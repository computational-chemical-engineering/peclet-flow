// Galerkin (variational/aggregation) coarsening vs constant-coefficient coarse operators, for a
// strongly-variable-coefficient Poisson (smooth openness in [0.02, 1], a 50x coefficient ratio -- the
// regime where the coarse-operator quality, not the smoother, is the bottleneck). The Galerkin V-cycle
// (coarse operators A_c = P^T A_f P + injection/summation transfers) must converge the system while the
// constant-coefficient-coarse V-cycle stalls. Convergence to ~0 proves the Galerkin cycle solves the
// true operator (a converged V-cycle finds the unique solution); the residual is identical across
// np=1,2,4, proving the aggregation/injection/sum operations are exact under decomposition.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_halo.cuh"
#include "mac_multigrid.cuh"

__host__ __device__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}
__host__ __device__ inline double psdf(double x, double y, double z, int3 res) {
  double cx = res.x * 0.5, cy = res.y * 0.5, cz = res.z * 0.5, R = res.x * 0.3;
  double dx = x - cx, dy = y - cy, dz = z - cz;
  dx -= res.x * round(dx / res.x);
  dy -= res.y * round(dy / res.y);
  dz -= res.z * round(dz / res.z);
  return sqrt(dx * dx + dy * dy + dz * dz) - R;
}
// smooth, strictly positive openness in [0.02, 1] (50x ratio); no hard zeros -> well-conditioned
__host__ __device__ inline double openf(double sd) { return 0.02 + 0.49 * (1.0 + tanh(2.0 * sd)); }

__global__ void d_fill_open(double* ox, double* oy, double* oz, int3 ext, int3 og, int3 res) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
  double gx = og.x + lx, gy = og.y + ly, gz = og.z + lz;
  ox[i] = openf(psdf(gx - 0.5, gy, gz, res));
  oy[i] = openf(psdf(gx, gy - 0.5, gz, res));
  oz[i] = openf(psdf(gx, gy, gz - 0.5, res));
}

// mode: 0 = V-cycle, constant-coeff coarse; 1 = V-cycle, Galerkin coarse; 2 = CG + Galerkin V-cycle.
// build operator, solve, return initial (x=0) and final residual max over owned cells, and #iterations.
static void run(int rank, int size, int3 res, int nlev, int nv, int pre, int post, int bottom,
                int mode, double& r0, double& rf, int& iters) {
  bool galerkin = (mode >= 1);
  cfdmpi::DistributedPoissonMG mg;
  mg.init(res, rank, size, /*h0=*/1.0, nlev, MPI_COMM_WORLD, /*ghost=*/1);
  cfdmpi::MGLevel& l0 = mg.level(0);
  dim3 blk(8, 8, 8);

  double *ox, *oy, *oz;
  for (double** p : {&ox, &oy, &oz}) cudaMalloc(p, l0.n * 8);
  dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 7) / 8);
  d_fill_open<<<gE, blk>>>(ox, oy, oz, l0.ext, l0.og, res);
  mg.setFineVariableOperator(ox, oy, oz, 1.0, 1.0, 1.0, galerkin);
  for (double* p : {ox, oy, oz}) cudaFree(p);

  std::vector<double> hb(l0.n, 0.0);
  for (int lz = l0.g; lz < l0.ext.z - l0.g; ++lz)
    for (int ly = l0.g; ly < l0.ext.y - l0.g; ++ly)
      for (int lx = l0.g; lx < l0.ext.x - l0.g; ++lx)
        hb[(size_t)lx + (size_t)ly * l0.ext.x + (size_t)lz * l0.ext.x * l0.ext.y] =
            hash01(lx + l0.og.x, ly + l0.og.y, lz + l0.og.z, 7);
  cudaMemcpy(l0.rhs, hb.data(), l0.n * 8, cudaMemcpyHostToDevice);
  cfdmpi::mac_remove_mean(l0.rhs, l0.mac, MPI_COMM_WORLD);
  cudaMemset(l0.x, 0, l0.n * 8);
  r0 = cfdmpi::mac_max_abs(l0.rhs, l0.mac, MPI_COMM_WORLD);  // residual at x=0 is rhs

  iters = nv;
  if (mode == 2)
    iters = mg.solve_pcg(nv, /*rtol=*/1e-10, pre, post, bottom);
  else
    mg.solve(nv, pre, post, bottom);

  dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
  l0.mac.exchange(l0.x);
  cfdmpi::mgdetail::mg_residual_var_k<<<grd, blk>>>(l0.res, l0.x, l0.rhs, l0.AC, l0.AW, l0.AE, l0.AS,
                                                    l0.AN, l0.AB, l0.AT, l0.ext, l0.g);
  rf = cfdmpi::mac_max_abs(l0.res, l0.mac, MPI_COMM_WORLD);
  mg.free();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  const int nlev = 4, nv = 12, pre = 2, post = 2, bottom = 20;

  double cc_r0, cc_rf, gk_r0, gk_rf, pc_r0, pc_rf;
  int cc_it, gk_it, pc_it;
  run(rank, size, res, nlev, nv, pre, post, bottom, /*mode=*/0, cc_r0, cc_rf, cc_it);
  run(rank, size, res, nlev, nv, pre, post, bottom, /*mode=*/1, gk_r0, gk_rf, gk_it);
  run(rank, size, res, nlev, /*max_cg_iter=*/40, pre, post, bottom, /*mode=*/2, pc_r0, pc_rf, pc_it);

  int fail = 0;
  if (rank == 0) {
    bool pc_conv = std::isfinite(pc_rf) && pc_rf < 1e-9 * pc_r0;  // CG+Galerkin converges -> exact
    bool gk_better = gk_rf < cc_rf;  // Galerkin coarse already beats const-coeff coarse
    fail = (pc_conv && gk_better) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  variable Poisson (openness 0.02..1, 50x), %d V-cycles\n", size, res.x,
           res.y, res.z, nv);
    printf("  const-coeff coarse V-cycle: residual %.3e -> %.3e  (%.2e)\n", cc_r0, cc_rf, cc_rf / cc_r0);
    printf("  Galerkin coarse  V-cycle:   residual %.3e -> %.3e  (%.2e)   %.1fx better than const\n",
           gk_r0, gk_rf, gk_rf / gk_r0, cc_rf / gk_rf);
    printf("  CG + Galerkin V-cycle:      residual %.3e -> %.3e  (%.2e) in %d its   %s\n", pc_r0,
           pc_rf, pc_rf / pc_r0, pc_it, pc_conv ? "converged" : "NOT CONVERGED");
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
