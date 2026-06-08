// Benchmark the distributed global reductions (mac_max_abs, mac_dot, mac_remove_mean) -- the per-call
// cost (reduction kernel + MPI collective + device->host readback + any scratch alloc) that the PCG /
// V-cycle inner loop pays many times per iteration. Run before/after the Tier-1 reduction rework to
// measure the speedup. Run: mpirun -np N ./bench_reductions [res] [iters]
#include <mpi.h>

#include <cstdio>

#include "mac_halo.cuh"
#include "mac_reductions.cuh"

__global__ void fillk(double* a, long n, int seed) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = (double)(((unsigned)(i * 2654435761u) ^ (unsigned)seed) & 0xffff) / 65535.0 - 0.5;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = argc > 1 ? atoi(argv[1]) : 128;
  int iters = argc > 2 ? atoi(argv[2]) : 2000;
  int3 res = make_int3(N, N, N);

  MacGridHalo h;
  h.init(res, rank, size, {true, true, true}, /*ghost=*/2, MPI_COMM_WORLD);
  long n = (long)h.num_local_cells();
  double *a, *b;
  cudaMalloc(&a, n * 8);
  cudaMalloc(&b, n * 8);
  int t = 256, blocks = (int)((n + t - 1) / t);
  fillk<<<blocks, t>>>(a, n, 1);
  fillk<<<blocks, t>>>(b, n, 2);
  cudaDeviceSynchronize();

  auto bench = [&](const char* tag, auto call) {
    for (int w = 0; w < 50; ++w) call();  // warmup
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    volatile double acc = 0;
    for (int it = 0; it < iters; ++it) acc += call();
    cudaDeviceSynchronize();
    double us = (MPI_Wtime() - t0) / iters * 1e6;
    if (rank == 0) printf("  %-18s %8.2f us/call   (acc=%.3e)\n", tag, us, (double)acc);
    return us;
  };

  if (rank == 0)
    printf("=== reduction benchmark  (np=%d, %dx%dx%d, %d iters/op) ===\n", size, N, N, N, iters);
  double tm = bench("mac_max_abs", [&] { return cfdmpi::mac_max_abs(a, h, MPI_COMM_WORLD); });
  double td = bench("mac_dot", [&] { return cfdmpi::mac_dot(a, b, h, MPI_COMM_WORLD); });
  double tr = bench("mac_remove_mean", [&] {
    cfdmpi::mac_remove_mean(a, h, MPI_COMM_WORLD);
    return 0.0;
  });
  if (rank == 0)
    printf("  -> total per (max_abs + dot + remove_mean) = %.2f us\n", tm + td + tr);

  cudaFree(a);
  cudaFree(b);
  MPI_Finalize();
  return 0;
}
