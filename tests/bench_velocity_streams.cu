// Benchmark the 3-stream concurrent velocity solve (set_velocity_streams) vs the serial per-component
// loop. The u/v/w IBM RB-GS momentum solves are independent; streaming overlaps them when one stencil
// does not saturate the GPU (small per-component sizes), and is ~neutral once saturated. A velocity-
// heavy config (large n_diff, tiny n_pois) isolates the velocity solve. Run: mpirun -np N ./bench [iters]
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"

using dns::DistributedNS;

// sphere SDF (negative inside) on the extended block from global coords
static std::vector<double> sphere_sdf(const DistributedNS& s, int3 res) {
  int3 e = s.ext();
  int3 og = s.origin_incl_ghost();
  std::vector<double> sdf((size_t)e.x * e.y * e.z);
  double cx = res.x * 0.5, cy = res.y * 0.5, cz = res.z * 0.5, R = res.x * 0.30;
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx) {
        double gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        double d = std::sqrt((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz));
        sdf[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = d - R;  // <0 inside
      }
  return sdf;
}

static double run(int rank, int size, int N, int iters, bool streams) {
  int3 res = make_int3(N, N, N);
  DistributedNS s;
  s.init(res, rank, size, /*rho=*/1.0, /*mu=*/0.05, /*dt=*/0.1);
  s.set_velocity_streams(streams);
  s.set_body_force(1e-3, 0, 0);
  s.set_ibm_solid(sphere_sdf(s, res));
  // velocity-heavy: many diffusion sweeps, few pressure sweeps -> isolates the streamed velocity solve
  const int n_diff = 40, n_pois = 2;
  for (int w = 0; w < 3; ++w) s.step(n_diff, n_pois);  // warmup
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int it = 0; it < iters; ++it) s.step(n_diff, n_pois);
  cudaDeviceSynchronize();
  return (MPI_Wtime() - t0) / iters * 1e3;  // ms/step
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);
  int iters = argc > 1 ? atoi(argv[1]) : 30;

  if (rank == 0)
    printf("=== velocity-stream benchmark (np=%d, IBM sphere, n_diff=40 n_pois=2, %d steps) ===\n",
           size, iters);
  for (int N : {32, 64, 128}) {
    double tser = run(rank, size, N, iters, /*streams=*/false);
    double tstr = run(rank, size, N, iters, /*streams=*/true);
    if (rank == 0)
      printf("  N=%-4d  serial %8.3f ms/step   3-stream %8.3f ms/step   speedup %.2fx\n", N, tser,
             tstr, tser / tstr);
  }
  MPI_Finalize();
  return 0;
}
