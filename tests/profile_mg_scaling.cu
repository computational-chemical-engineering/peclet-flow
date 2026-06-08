// Profiling to SIZE the iteration-reduction opportunity for the cut-cell pressure multigrid before
// investing in smoothed aggregation / Chebyshev / W-cycles. Builds the real Galerkin (unsmoothed-
// aggregation) cut-cell operator on a sphere packing and measures the two diagnostics that decide
// whether a better coarse correction (SA) is warranted vs a cheaper fix:
//
//   (1) Grid independence: CG iterations to a fixed rtol across N. Flat -> already ~optimal MG, little
//       room. Growing with N -> the V-cycle is not h-independent, SA/better coarsening has room.
//   (2) Smoother- vs coarse-limited: CG iterations at pre/post = 1, 2, 4 sweeps. Big drop with more
//       sweeps -> smoother-limited -> a stronger/cheaper smoother (Chebyshev) helps. Little drop ->
//       coarse-correction-limited -> SA (the expensive coarse-grid fix) is the relevant lever.
//
// Also reports the standalone V-cycle asymptotic convergence factor rho (geometric residual reduction
// per cycle, measured above the single-precision residual floor). Run: mpirun -np 1 ./profile_mg_scaling [N ...]
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_cutcell.cuh"
#include "mac_halo.cuh"
#include "mac_multigrid.cuh"

__host__ inline double smooth_rhs(int gx, int gy, int gz, int3 res) {
  double kx = 2.0 * M_PI / res.x, ky = 2.0 * M_PI / res.y, kz = 2.0 * M_PI / res.z;
  return std::sin(kx * gx) * std::cos(ky * gy) + std::sin(ky * gy) * std::cos(kz * gz) +
         std::sin(kz * gz) * std::cos(kx * gx);
}
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
__global__ void mask_rhs_solid_k(double* rhs, const cfdmpi::mreal* AC, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && AC[i] < 1e-30f) rhs[i] = 0.0;
}

// coarsen to ~4^3 (true h-independence test needs the coarsest grid fixed, not growing with N)
static int levels_for(int N) {
  int l = 1, n = N;
  while (n > 4 && (n % 2) == 0) { n /= 2; ++l; }
  return l;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);
  dim3 blk(8, 8, 8);

  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
  if (sizes.empty()) sizes = {32, 64, 128};

  if (rank == 0)
    printf("=== cut-cell pressure MG scaling (np=%d, Galerkin/unsmoothed-aggregation + CG) ===\n", size);

  for (int N : sizes) {
    int3 res = make_int3(N, N, N);
    int nlev = levels_for(N), bottom = 20;

    auto build = [&](cfdmpi::DistributedPoissonMG& mg) {
      mg.init(res, rank, size, 1.0, nlev, MPI_COMM_WORLD, /*ghost=*/2);
      cfdmpi::MGLevel& l0 = mg.level(0);
      double *sdf, *ox, *oy, *oz;
      for (double** p : {&sdf, &ox, &oy, &oz}) cudaMalloc(p, l0.n * 8);
      dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 3) / 4);
      fill_sdf_ext<<<gE, dim3(8, 8, 4)>>>(sdf, l0.ext, l0.og, res);
      cfdmpi::ccdetail::cc_build_open_k<<<gE, dim3(8, 8, 4)>>>(ox, oy, oz, sdf, l0.ext, 1.0, 1.0, 1.0);
      mg.setFineVariableOperator(ox, oy, oz, 1.0, 1.0, 1.0, /*galerkin=*/true);
      for (double* p : {sdf, ox, oy, oz}) cudaFree(p);
    };
    auto setup_rhs = [&](cfdmpi::DistributedPoissonMG& mg) {
      cfdmpi::MGLevel& l0 = mg.level(0);
      std::vector<double> hb(l0.n, 0.0);
      for (int lz = l0.g; lz < l0.ext.z - l0.g; ++lz)
        for (int ly = l0.g; ly < l0.ext.y - l0.g; ++ly)
          for (int lx = l0.g; lx < l0.ext.x - l0.g; ++lx)
            hb[(size_t)lx + (size_t)ly * l0.ext.x + (size_t)lz * l0.ext.x * l0.ext.y] =
                smooth_rhs(lx + l0.og.x, ly + l0.og.y, lz + l0.og.z, res);
      cudaMemcpy(l0.rhs, hb.data(), l0.n * 8, cudaMemcpyHostToDevice);
      int t = 256, b = (int)((l0.n + t - 1) / t);
      mask_rhs_solid_k<<<b, t>>>(l0.rhs, l0.AC, (long)l0.n);
      cfdmpi::mac_remove_mean(l0.rhs, l0.mac, MPI_COMM_WORLD);
      cudaMemset(l0.x, 0, l0.n * 8);
    };
    auto resid = [&](cfdmpi::DistributedPoissonMG& mg) {
      cfdmpi::MGLevel& l0 = mg.level(0);
      dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
      l0.mac.exchange(l0.x);
      cfdmpi::mgdetail::mg_residual_var_k<<<grd, blk>>>(l0.res, l0.x, l0.rhs, l0.AC, l0.AW, l0.AE,
                                                        l0.AS, l0.AN, l0.AB, l0.AT, l0.ext, l0.g);
      return cfdmpi::mac_max_abs(l0.res, l0.mac, MPI_COMM_WORLD);
    };

    // (A) standalone V-cycle asymptotic factor rho: residual reduction per cycle, above the float floor
    double rho = 0.0;
    {
      cfdmpi::DistributedPoissonMG mg;
      build(mg);
      setup_rhs(mg);
      double r0 = resid(mg);
      std::vector<double> ratios;
      double rprev = r0;
      for (int v = 0; v < 30; ++v) {
        mg.solve(1, 2, 2, bottom);
        double r = resid(mg);
        if (r > 1e-6 * r0 && r < 0.95 * rprev) ratios.push_back(r / rprev);  // clean decay window
        rprev = r;
      }
      if (!ratios.empty()) {
        std::sort(ratios.begin(), ratios.end());
        rho = ratios[ratios.size() / 2];  // median per-cycle reduction
      }
      mg.free();
    }

    // (B) CG iterations to rtol across smoother sweeps (smoother- vs coarse-limited)
    int it_pp[3] = {0, 0, 0};
    int pps[3] = {1, 2, 4};
    for (int s = 0; s < 3; ++s) {
      cfdmpi::DistributedPoissonMG mg;
      build(mg);
      setup_rhs(mg);
      it_pp[s] = mg.solve_pcg(/*max_iter=*/200, /*rtol=*/1e-8, pps[s], pps[s], bottom);
      mg.free();
    }

    // (C) Chebyshev smoother (degree=2, i.e. pre/post=2 -- equal matvec cost to RB-GS pre/post=2) at a
    // few spectral-band ratios; compare CG iterations to the RB-GS baseline it_pp[1].
    double ratios[3] = {8.0, 16.0, 30.0};
    int it_cheb[3] = {0, 0, 0};
    for (int s = 0; s < 3; ++s) {
      cfdmpi::DistributedPoissonMG mg;
      build(mg);
      mg.enableChebyshev(ratios[s]);
      setup_rhs(mg);
      it_cheb[s] = mg.solve_pcg(/*max_iter=*/200, /*rtol=*/1e-8, 2, 2, bottom);
      mg.free();
    }
    int best_cheb = it_cheb[0];
    for (int s = 1; s < 3; ++s)
      if (it_cheb[s] > 0 && (best_cheb == 0 || it_cheb[s] < best_cheb)) best_cheb = it_cheb[s];

    if (rank == 0) {
      double R = N * 0.18;
      double porosity = 1.0 - 8.0 * (4.0 / 3.0 * M_PI * R * R * R) / ((double)N * N * N);
      printf("\n  N=%-4d (levels=%d, porosity ~%.2f)\n", N, nlev, porosity);
      printf("    standalone V-cycle asymptotic factor rho = %.3f  (smaller=stronger; <0.2 is good MG)\n",
             rho);
      printf("    RB-GS     CG iters to 1e-8:  pre/post=1 -> %d    pre/post=2 -> %d    pre/post=4 -> %d\n",
             it_pp[0], it_pp[1], it_pp[2]);
      printf("    smoother sensitivity (it@1 / it@4) = %.2fx %s\n",
             it_pp[2] ? (double)it_pp[0] / it_pp[2] : 0.0,
             (it_pp[2] && (double)it_pp[0] / it_pp[2] > 1.8) ? "(smoother-limited)"
                                                            : "(coarse-correction-limited)");
      printf("    Chebyshev CG iters (deg=2):  eig_ratio=8 -> %d   16 -> %d   30 -> %d\n", it_cheb[0],
             it_cheb[1], it_cheb[2]);
      printf("    => best Chebyshev %d vs RB-GS(pre/post=2) %d   = %.2fx fewer iters (~equal cost)\n",
             best_cheb, it_pp[1], best_cheb ? (double)it_pp[1] / best_cheb : 0.0);
    }
  }
  if (rank == 0)
    printf(
        "\n  Read: CG iters ~flat across N => grid-independent (little SA room). Growing => SA room.\n"
        "        big drop 1->4 sweeps => smoother-limited (Chebyshev helps); small => coarse-limited.\n");
  MPI_Finalize();
  return 0;
}
