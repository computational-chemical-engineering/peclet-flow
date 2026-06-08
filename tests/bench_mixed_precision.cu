// Benchmark: the speedup from the single-precision (cfdmpi::mreal) operator. The pressure/momentum
// solve is memory-bandwidth bound; its inner loop streams the 7-point operator (AC..AT) every call.
// This times the two bandwidth-bound kernels in isolation (no halo exchange / reductions, so only the
// operator-storage precision changes between builds):
//   - the variable-coefficient RB-GS smoother  (mgdetail::mg_smooth_var_k)
//   - the matvec  y = A x                       (mgdetail::mg_apply_var_k, the PCG inner product input)
// The iterate x and RHS b stay double (mreal only changes the operator), so the expected ceiling is
//   (7*8 + 6*8 + 8 + 8) / (7*S + 6*8 + 8 + 8) = 120 / (72 + 7*S)   bytes-per-cell ratio,
// i.e. ~1.30x for S=sizeof(float)=4, not 2x. Build once with mreal=float and once with mreal=double
// (flip the alias in mac_ibm.cuh + mac_multigrid.cuh) and compare. Run: mpirun -np 1 ./bench [res] [iters]
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_cutcell.cuh"
#include "mac_halo.cuh"
#include "mac_multigrid.cuh"

// 2x2x2 sphere packing, SDF negative inside (matches profile_sphere_packing) -> a real cut-cell operator
__host__ __device__ inline double packing_sdf(double x, double y, double z, int3 res) {
  double R = res.x * 0.18, best = 1e30;
  for (int cz = 0; cz < 2; ++cz)
    for (int cy = 0; cy < 2; ++cy)
      for (int cx = 0; cx < 2; ++cx) {
        double sx = (cx + 0.5) * res.x / 2.0, sy = (cy + 0.5) * res.y / 2.0,
               sz = (cz + 0.5) * res.z / 2.0;
        double dx = x - sx, dy = y - sy, dz = z - sz;
        dx -= res.x * round(dx / res.x); dy -= res.y * round(dy / res.y); dz -= res.z * round(dz / res.z);
        best = fmin(best, sqrt(dx * dx + dy * dy + dz * dz) - R);
      }
  return best;
}
__global__ void fill_sdf_ext(double* sdf, int3 ext, int3 og, int3 res) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  sdf[i] = packing_sdf(lx + og.x, ly + og.y, lz + og.z, res);
}
__global__ void fill_field_k(double* a, int3 ext, int3 og, int seed) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  unsigned h = (unsigned)((lx + og.x) * 73856093 ^ (ly + og.y) * 19349663 ^ (lz + og.z) * 83492791 ^ seed);
  a[i] = (h & 0xffff) / 65535.0 - 0.5;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = argc > 1 ? atoi(argv[1]) : 128;
  int iters = argc > 2 ? atoi(argv[2]) : 300;
  int3 res = make_int3(N, N, N);
  dim3 blk(8, 8, 8);

  cfdmpi::DistributedPoissonMG mg;
  mg.init(res, rank, size, 1.0, /*n_levels=*/1, MPI_COMM_WORLD, /*ghost=*/2);
  cfdmpi::MGLevel& l0 = mg.level(0);

  // real cut-cell operator (AC..AT in cfdmpi::mreal)
  double *sdf, *ox, *oy, *oz;
  for (double** p : {&sdf, &ox, &oy, &oz}) cudaMalloc(p, l0.n * 8);
  dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 7) / 8);
  fill_sdf_ext<<<gE, blk>>>(sdf, l0.ext, l0.og, res);
  cfdmpi::ccdetail::cc_build_open_k<<<gE, blk>>>(ox, oy, oz, sdf, l0.ext, 1.0, 1.0, 1.0);
  mg.setFineVariableOperator(ox, oy, oz, 1.0, 1.0, 1.0, /*galerkin=*/false);
  for (double* p : {sdf, ox, oy, oz}) cudaFree(p);

  fill_field_k<<<gE, blk>>>(l0.x, l0.ext, l0.og, 7);
  fill_field_k<<<gE, blk>>>(l0.rhs, l0.ext, l0.og, 13);

  dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
  double *y;
  cudaMalloc(&y, l0.n * 8);

  auto time_kernel = [&](const char* tag, auto launch, double bytes_per_cell) {
    for (int w = 0; w < 20; ++w) launch();  // warmup
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int it = 0; it < iters; ++it) launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    double per = ms / iters;                                   // ms per launch
    double ncells = (double)l0.inner.x * l0.inner.y * l0.inner.z;
    double gbps = ncells * bytes_per_cell / (per * 1e-3) / 1e9;
    if (rank == 0)
      printf("  %-22s %8.4f ms/call   %7.1f GB/s   (%.0f bytes/cell)\n", tag, per, gbps, bytes_per_cell);
    return per;
  };

  size_t S = sizeof(cfdmpi::mreal);
  if (rank == 0)
    printf("=== mixed-precision kernel benchmark  (np=%d, %dx%dx%d, %d iters, sizeof(mreal)=%zu) ===\n",
           size, N, N, N, iters, S);

  // smoother: read 7 op + 6 nbr-x(double) + 1 b(double), write 1 x(double)
  double sm_bytes = 7.0 * S + 6 * 8 + 8 + 8;
  auto smooth = [&]() {
    cfdmpi::mgdetail::mg_smooth_var_k<<<grd, blk>>>(l0.x, l0.rhs, l0.AC, l0.AW, l0.AE, l0.AS, l0.AN,
                                                    l0.AB, l0.AT, l0.ext, l0.og, l0.g, 0);
  };
  // matvec: read 7 op + 7 x(double), write 1 y(double)
  double mv_bytes = 7.0 * S + 7 * 8 + 8;
  auto matvec = [&]() {
    cfdmpi::mgdetail::mg_apply_var_k<<<grd, blk>>>(y, l0.x, l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB,
                                                   l0.AT, l0.ext, l0.g);
  };

  double ts = time_kernel("RB-GS smoother sweep", smooth, sm_bytes);
  double tm = time_kernel("matvec  y = A x", matvec, mv_bytes);
  if (rank == 0)
    printf("  -> smoother %.4f ms   matvec %.4f ms   (operator = %s)\n", ts, tm,
           S == 4 ? "float" : "double");

  cudaFree(y);
  mg.free();
  MPI_Finalize();
  return 0;
}
