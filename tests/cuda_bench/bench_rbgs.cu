// Native-CUDA throughput benchmark of the RB-GS Poisson sweep — the reference for the Kokkos-vs-CUDA
// efficiency comparison. The kernel is the same 7-point Red-Black sweep as cfd's pois_k and the Kokkos
// poisSmoothColor; timed identically (many full sweeps, min of repeats). Compare ns/sweep with
// tests/kokkos/bench_rbgs (run on the same GPU).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void pois_native_k(double* phi, const double* d, int ex, int ey, int ez, int g, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + g;
  int y = blockIdx.y * blockDim.y + threadIdx.y + g;
  int z = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (x >= ex - g || y >= ey - g || z >= ez - g) return;
  if (((x + y + z) & 1) != color) return;
  long sx = 1, sy = ex, sz = (long)ex * ey;
  long i = (long)x + (long)y * ex + (long)z * sz;
  double s = phi[i + sx] + phi[i - sx] + phi[i + sy] + phi[i - sy] + phi[i + sz] + phi[i - sz];
  phi[i] = (s - d[i]) / 6.0;
}

int main(int argc, char** argv) {
  const int N = (argc > 1) ? std::atoi(argv[1]) : 128;
  const int K = (argc > 2) ? std::atoi(argv[2]) : 200;
  const int g = 1;
  const int ex = N + 2 * g, ey = N + 2 * g, ez = N + 2 * g;
  const std::size_t n = (std::size_t)ex * ey * ez;
  double *phi, *d;
  cudaMalloc(&phi, n * sizeof(double));
  cudaMalloc(&d, n * sizeof(double));
  cudaMemset(phi, 0, n * sizeof(double));
  cudaMemset(d, 0, n * sizeof(double));

  dim3 blk(8, 8, 8);
  dim3 grd((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y, (N + blk.z - 1) / blk.z);
  auto sweep = [&]() {
    pois_native_k<<<grd, blk>>>(phi, d, ex, ey, ez, g, 0);
    pois_native_k<<<grd, blk>>>(phi, d, ex, ey, ez, g, 1);
  };
  for (int i = 0; i < 10; ++i) sweep();
  cudaDeviceSynchronize();

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    cudaEventRecord(a);
    for (int i = 0; i < K; ++i) sweep();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    if (ms < best) best = ms;
  }
  const double ns_per_sweep = best * 1e6 / K;
  const double cells = (double)N * N * N;
  const double gbps = cells * 64.0 / ns_per_sweep;
  std::printf("CUDA    N=%d sweeps=%d : %.3f ns/sweep, %.4f ns/cell, %.1f GB/s\n",
              N, K, ns_per_sweep, ns_per_sweep / cells, gbps);
  cudaFree(phi); cudaFree(d);
  return 0;
}
