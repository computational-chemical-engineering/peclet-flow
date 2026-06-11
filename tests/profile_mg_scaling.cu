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
#include <chrono>
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
    printf("=== cut-cell pressure MG: coarse-operator comparison (np=%d) ===\n", size);

  // coarse-operator modes to compare (Phase 0 baseline; REDISC added in Phase 1)
  enum Mode { SINGLE = 0, CONSTC = 1, GALERKIN = 2, REDISC = 3 };
  const char* mode_name[] = {"1-level RB-GS ", "const-coeff MG", "Galerkin   MG ", "rediscret. MG "};

  for (int N : sizes) {
    int3 res = make_int3(N, N, N);
    int nlev_full = levels_for(N), bottom = 20;
    int cur_nlev = nlev_full;

    auto build = [&](cfdmpi::DistributedPoissonMG& mg, int mode) {
      cur_nlev = (mode == SINGLE) ? 1 : nlev_full;
      mg.init(res, rank, size, 1.0, cur_nlev, MPI_COMM_WORLD, /*ghost=*/2);
      cfdmpi::MGLevel& l0 = mg.level(0);
      double *sdf, *ox, *oy, *oz;
      for (double** p : {&sdf, &ox, &oy, &oz}) cudaMalloc(p, l0.n * 8);
      dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 3) / 4);
      fill_sdf_ext<<<gE, dim3(8, 8, 4)>>>(sdf, l0.ext, l0.og, res);
      cfdmpi::ccdetail::cc_build_open_k<<<gE, dim3(8, 8, 4)>>>(ox, oy, oz, sdf, l0.ext, 1.0, 1.0, 1.0);
      if (mode == REDISC)
        mg.setFineVariableOperatorRediscretized(ox, oy, oz, 1.0, 1.0, 1.0);
      else
        mg.setFineVariableOperator(ox, oy, oz, 1.0, 1.0, 1.0, /*galerkin=*/mode == GALERKIN);
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

    // standalone V-cycle asymptotic factor rho (median residual reduction per cycle, above float floor)
    auto measure_rho = [&](int mode) {
      cfdmpi::DistributedPoissonMG mg;
      build(mg, mode);
      setup_rhs(mg);
      double r0 = resid(mg), rprev = r0, rho = 0.0;
      std::vector<double> ratios;
      for (int v = 0; v < 30; ++v) {
        mg.solve(1, 2, 2, bottom);
        double r = resid(mg);
        if (r > 1e-6 * r0 && r < 0.95 * rprev) ratios.push_back(r / rprev);
        rprev = r;
      }
      if (!ratios.empty()) {
        std::sort(ratios.begin(), ratios.end());
        rho = ratios[ratios.size() / 2];
      }
      mg.free();
      return rho;  // 0 reported as n/a (e.g. single-level has no coarse correction)
    };
    // MG-preconditioned CG iterations to 1e-8 at pre/post=2
    auto measure_iters = [&](int mode) {
      cfdmpi::DistributedPoissonMG mg;
      build(mg, mode);
      setup_rhs(mg);
      int it = mg.solve_pcg(/*max_iter=*/400, /*rtol=*/1e-8, 2, 2, bottom);
      mg.free();
      return it;
    };

    if (rank == 0) {
      double R = N * 0.18;
      double porosity = 1.0 - 8.0 * (4.0 / 3.0 * M_PI * R * R * R) / ((double)N * N * N);
      printf("\n  N=%-4d (levels=%d, porosity ~%.2f)   [coarse-operator comparison]\n", N, nlev_full,
             porosity);
      printf("    %-16s %12s   %10s\n", "coarse operator", "V-cyc rho", "PCG iters");
    }
    for (int mode = SINGLE; mode <= REDISC; ++mode) {
      double rho = (mode == SINGLE) ? 0.0 : measure_rho(mode);  // no coarse grid for single level
      int it = measure_iters(mode);
      if (rank == 0) {
        if (rho > 0.0)
          printf("    %-16s %12.3f   %10d\n", mode_name[mode], rho, it);
        else
          printf("    %-16s %12s   %10d\n", mode_name[mode], "n/a", it);
      }
    }

    // --- timing: standalone V-cycle vs MG-PCG to the SAME residual tolerance (rediscretized operator) ---
    // PCG costs more per iteration (an extra matvec + 2 global dot-products/Allreduce + the axpys, plus a
    // symmetric V-cycle) but usually needs fewer iterations. This measures whether the standalone V-cycle
    // (a few more cheap cycles) beats PCG in wall-clock for the cut-cell pressure solve.
    {
      cfdmpi::DistributedPoissonMG mg;
      build(mg, REDISC);
      cfdmpi::MGLevel& l0 = mg.level(0);
      size_t nb = l0.n * 8;
      auto time_ms = [&](auto&& fn) {  // median of GPU-synced repeats, each cold-started (x=0)
        auto run = [&] { cudaMemset(l0.x, 0, nb); fn(); };
        run(); cudaDeviceSynchronize();  // warmup
        std::vector<double> ts;
        for (int r = 0; r < 7; ++r) {
          cudaDeviceSynchronize();
          auto a = std::chrono::high_resolution_clock::now();
          run();
          cudaDeviceSynchronize();
          auto b = std::chrono::high_resolution_clock::now();
          ts.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size() / 2];
      };
      if (rank == 0)
        printf("    --- standalone V-cycle vs MG-PCG to a fixed tolerance (rediscretized; pre/post=2) ---\n");
      double rtols[] = {1e-4, 1e-6, 1e-8};  // loose (per-step projection) -> tight
      for (double rtol : rtols) {
        setup_rhs(mg);
        double r0 = resid(mg);
        int kv = 1;
        for (; kv <= 500; ++kv) { mg.solve(1, 2, 2, bottom); if (resid(mg) < rtol * r0) break; }
        setup_rhs(mg);
        int kp = mg.solve_pcg(500, rtol, 2, 2, bottom);
        setup_rhs(mg);
        double tv = time_ms([&] { mg.solve(kv, 2, 2, bottom); });
        double tp = time_ms([&] { mg.solve_pcg(500, rtol, 2, 2, bottom); });
        if (rank == 0)
          printf("    rtol=%.0e :  V-cycle %2d cyc %7.3f ms (%.3f/cyc) | PCG %2d it %7.3f ms (%.3f/it)"
                 "  -> V-cycle %.2fx %s\n",
                 rtol, kv, tv, tv / kv, kp, tp, tp / kp, tp > tv ? tp / tv : tv / tp,
                 tp > tv ? "faster" : "slower");
      }
      mg.free();
    }
  }
  if (rank == 0)
    printf(
        "\n  Read: rho<0.2 & PCG iters flat across N => grid-independent MG (the goal).\n"
        "        rho->1 / iters growing => poor coarse model. 'rediscret. MG' is the Phase-1 target.\n");
  MPI_Finalize();
  return 0;
}
