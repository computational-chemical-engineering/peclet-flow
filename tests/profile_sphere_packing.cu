// Profiling: the pressure Poisson solve for distributed incompressible flow through a periodic sphere
// packing (a porous medium -- the production solver's target case). The pressure solve is the dominant
// cost of incompressible porous flow. We build the real cut-cell pressure operator from a sphere-packing
// SDF and solve A x = b (b = a mean-removed deterministic field) to a target residual with three solvers
// on ONE GPU, reporting iterations AND wall time:
//   (A) pure Red-Black Gauss-Seidel   -- the "original" iterative solver (no coarse grids)
//   (B) geometric multigrid V-cycles  -- constant-coefficient coarse operators
//   (C) Galerkin multigrid + CG       -- variational coarse operators, CG-accelerated ("better")
// Run: mpirun -np 1 ./profile_sphere_packing
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_cutcell.cuh"
#include "mac_halo.cuh"
#include "mac_multigrid.cuh"

// a SMOOTH, low-frequency, zero-mean RHS -- representative of a flow divergence (high frequencies would
// excite the near-singular thin-cut-cell modes, an unrealistically hard problem for any solver)
__host__ inline double smooth_rhs(int gx, int gy, int gz, int3 res) {
  double kx = 2.0 * M_PI / res.x, ky = 2.0 * M_PI / res.y, kz = 2.0 * M_PI / res.z;
  return std::sin(kx * gx) * std::cos(ky * gy) + std::sin(ky * gy) * std::cos(kz * gz) +
         std::sin(kz * gz) * std::cos(kx * gx);
}
// 2x2x2 sphere packing, SDF negative inside (min-image distance to nearest centre)
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
// zero the RHS in solid cells (A_C == 0) so the system is compatible (rhs perpendicular to the
// isolated-solid null space), as it physically is -- divergence vanishes in the solid.
__global__ void mask_rhs_solid_k(double* rhs, const cfdmpi::mreal* AC, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && AC[i] < 1e-30f) rhs[i] = 0.0;
}
__global__ void fill_sdf_ext(double* sdf, int3 ext, int3 og, int3 res) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  sdf[i] = packing_sdf(lx + og.x, ly + og.y, lz + og.z, res);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  dim3 blk(8, 8, 8);
  const double target = 1e-6;  // residual reduction target (relative to the initial residual)

  // shared RHS (deterministic) so all solvers tackle the identical system; build the cut-cell openness
  auto setup_rhs = [&](cfdmpi::DistributedPoissonMG& mg) {
    cfdmpi::MGLevel& l0 = mg.level(0);
    std::vector<double> hb(l0.n, 0.0);
    for (int lz = l0.g; lz < l0.ext.z - l0.g; ++lz)
      for (int ly = l0.g; ly < l0.ext.y - l0.g; ++ly)
        for (int lx = l0.g; lx < l0.ext.x - l0.g; ++lx)
          hb[(size_t)lx + (size_t)ly * l0.ext.x + (size_t)lz * l0.ext.x * l0.ext.y] =
              smooth_rhs(lx + l0.og.x, ly + l0.og.y, lz + l0.og.z, res);
    cudaMemcpy(l0.rhs, hb.data(), l0.n * 8, cudaMemcpyHostToDevice);
    // zero RHS in solid cells (must be called after the operator is built), then remove the mean
    int t = 256, b = (int)((l0.n + t - 1) / t);
    mask_rhs_solid_k<<<b, t>>>(l0.rhs, l0.AC, (long)l0.n);
    cfdmpi::mac_remove_mean(l0.rhs, l0.mac, MPI_COMM_WORLD);
    cudaMemset(l0.x, 0, l0.n * 8);
  };
  auto build_op = [&](cfdmpi::DistributedPoissonMG& mg, bool galerkin) {
    cfdmpi::MGLevel& l0 = mg.level(0);
    double *sdf, *ox, *oy, *oz;
    for (double** p : {&sdf, &ox, &oy, &oz}) cudaMalloc(p, l0.n * 8);
    dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 7) / 8);
    fill_sdf_ext<<<gE, blk>>>(sdf, l0.ext, l0.og, res);
    cfdmpi::ccdetail::cc_build_open_k<<<gE, blk>>>(ox, oy, oz, sdf, l0.ext, 1.0, 1.0, 1.0);
    mg.setFineVariableOperator(ox, oy, oz, 1.0, 1.0, 1.0, galerkin);
    for (double* p : {sdf, ox, oy, oz}) cudaFree(p);
  };
  auto resid = [&](cfdmpi::DistributedPoissonMG& mg) {
    cfdmpi::MGLevel& l0 = mg.level(0);
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    l0.mac.exchange(l0.x);
    cfdmpi::mgdetail::mg_residual_var_k<<<grd, blk>>>(l0.res, l0.x, l0.rhs, l0.AC, l0.AW, l0.AE, l0.AS,
                                                      l0.AN, l0.AB, l0.AT, l0.ext, l0.g);
    return cfdmpi::mac_max_abs(l0.res, l0.mac, MPI_COMM_WORLD);
  };

  double r0 = 0;
  // (A) pure RB-GS: 1 level => solve() is just smoothing. Count sweeps to the target.
  double tA = 0; int itA = 0; double rA = 0;
  {
    cfdmpi::DistributedPoissonMG mg;
    mg.init(res, rank, size, 1.0, /*n_levels=*/1, MPI_COMM_WORLD, /*ghost=*/2);
    build_op(mg, false);
    setup_rhs(mg);
    r0 = resid(mg);
    const int chunk = 50;  // sweeps per measurement
    cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (; itA < 20000; itA += chunk) {
      mg.solve(/*vcycles=*/1, 0, 0, /*bottom=*/chunk);  // chunk RB-GS sweeps
      if (resid(mg) < target * r0) { itA += chunk; break; }
    }
    cudaDeviceSynchronize();
    tA = (MPI_Wtime() - t0) * 1e3;
    rA = resid(mg);
    mg.free();
  }
  // (B) geometric multigrid V-cycles (constant-coefficient coarse)
  double tB = 0; int itB = 0; double rB = 0;
  {
    cfdmpi::DistributedPoissonMG mg;
    mg.init(res, rank, size, 1.0, /*n_levels=*/4, MPI_COMM_WORLD, /*ghost=*/2);
    build_op(mg, false);
    setup_rhs(mg);
    cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (; itB < 500; ++itB) {
      mg.solve(1, 2, 2, 20);
      if (resid(mg) < target * r0) { ++itB; break; }
    }
    cudaDeviceSynchronize();
    tB = (MPI_Wtime() - t0) * 1e3;
    rB = resid(mg);
    mg.free();
  }
  // (C) Galerkin multigrid + CG
  double tC = 0; int itC = 0; double rC = 0;
  {
    cfdmpi::DistributedPoissonMG mg;
    mg.init(res, rank, size, 1.0, /*n_levels=*/4, MPI_COMM_WORLD, /*ghost=*/2);
    build_op(mg, true);
    setup_rhs(mg);
    cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    itC = mg.solve_pcg(/*max_iter=*/200, /*rtol=*/target, 2, 2, 10);
    cudaDeviceSynchronize();
    tC = (MPI_Wtime() - t0) * 1e3;
    rC = resid(mg);
    mg.free();
  }

  if (rank == 0) {
    double R = res.x * 0.18;
    double porosity = 1.0 - 8.0 * (4.0 / 3.0 * M_PI * R * R * R) / ((double)res.x * res.y * res.z);
    printf("=== sphere-packing pressure solve (np=%d, %dx%dx%d, porosity ~%.2f, target=%.0e) ===\n",
           size, res.x, res.y, res.z, porosity, target);
    printf("  (A) pure RB-GS          : %6d sweeps   %8.1f ms   final res/r0 = %.2e\n", itA, tA,
           rA / r0);
    printf("  (B) geometric multigrid : %6d V-cycles %8.1f ms   final res/r0 = %.2e\n", itB, tB,
           rB / r0);
    printf("  (C) Galerkin MG + CG    : %6d CG iters %8.1f ms   final res/r0 = %.2e\n", itC, tC,
           rC / r0);
    printf("  speedups vs RB-GS:  multigrid %.1fx   Galerkin+CG %.1fx  (wall time to target)\n",
           tA / tB, tA / tC);
  }
  MPI_Finalize();
  return 0;
}
