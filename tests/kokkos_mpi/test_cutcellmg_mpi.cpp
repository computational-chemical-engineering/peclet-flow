// cfd-gpu — the PRODUCTION CutcellMG, multi-rank. Folds the distributed halo + reductions into the real
// class (gated behind CFD_MPI) rather than a test-only copy.
//
// Solves the periodic cut-cell pressure Poisson by the real CutcellMG::solvePCG (MG-PCG: CG preconditioned
// by one symmetric V-cycle) two ways: distributed (initMpi over MPI_COMM_WORLD) and single-rank (init() on
// the full grid, the validated single-GPU path). Every rank runs the single-rank reference redundantly and
// compares its own block. This proves CutcellMG.initMpi + the gated fill()->exchange +
// dot/maxabs/removeMean->Allreduce produce the same field as the single-GPU class, and (np=1) that the gating
// is byte-identical. Tolerance: np=1 is exact; np>1 agrees to ~1e-8 -- the MG-PCG is ADAPTIVE, so the
// inner-product Allreduce (summed in a different order than the single-rank local sum) steers the Krylov path
// slightly differently and, with the float-stored operator (residual floor ~1e-7 relative), the two converged
// solutions differ at ~1e-9. That is an operation-order/roundoff difference, not a method change (the
// deterministic V-cycle machinery is bit-tight to 1e-11 -- see test_distributed_mg). Build with -DCFD_MPI.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

#include "mac_cutcell_mg.hpp"

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"

using tpx::Index; using tpx::IVec;
using dns::CutcellMG; using dns::CCField; using dns::CCConst; using dns::C3;

static constexpr int G = 1, NLEV = 4;
static double source(int gx, int gy, int gz, IVec<3> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) + std::cos(4.0 * M_PI * gy / gs[1]) * std::sin(2.0 * M_PI * gz / gs[2]);
}

// fill level-0 inner cells of an extended block (ext = inner+2G) given its global inner origin.
static void fillSource(CCField b, C3 ext, C3 og, IVec<3> gs) {
  auto hb = Kokkos::create_mirror_view(b);
  for (std::size_t c = 0; c < b.extent(0); ++c) hb(c) = 0.0;
  for (int z = G; z < ext.z - G; ++z) for (int y = G; y < ext.y - G; ++y) for (int x = G; x < ext.x - G; ++x)
    hb((long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y) =
        source(x - G + og.x, y - G + og.y, z - G + og.z, gs);
  Kokkos::deep_copy(b, hb);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    IVec<3> gs{32, 32, 32};

    auto setupAndSolve = [&](CutcellMG& mg, C3 ext, C3 og) {
      const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
      CCField ox("ox", n), oy("oy", n), oz("oz", n);  // all-fluid openness (periodic Laplacian)
      Kokkos::deep_copy(ox, 1.0); Kokkos::deep_copy(oy, 1.0); Kokkos::deep_copy(oz, 1.0);
      mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
      CCField b("b", n), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
      fillSource(b, ext, og, gs);
      mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/200, /*rtol=*/1e-10, 2, 2, 8);
      auto hx = Kokkos::create_mirror_view(x); Kokkos::deep_copy(hx, x);
      return hx;
    };

    // distributed (real CutcellMG over MPI_COMM_WORLD)
    CutcellMG mg; mg.initMpi(gs[0], gs[1], gs[2], NLEV, MPI_COMM_WORLD);
    CutcellMG::Level& l0 = mg.level(0);
    auto hx = setupAndSolve(mg, l0.ext, l0.og);

    // single-rank reference (the validated single-GPU path) on the full grid; every rank, compare its block
    CutcellMG ref; ref.init(gs[0], gs[1], gs[2], NLEV);
    const C3 re{(int)gs[0] + 2 * G, (int)gs[1] + 2 * G, (int)gs[2] + 2 * G}, ro{0, 0, 0};
    auto hr = setupAndSolve(ref, re, ro);

    double maxdiff = 0.0;
    for (int z = G; z < l0.ext.z - G; ++z) for (int y = G; y < l0.ext.y - G; ++y) for (int x = G; x < l0.ext.x - G; ++x) {
      const int gx = x - G + l0.og.x, gy = y - G + l0.og.y, gz = z - G + l0.og.z;
      const double a = hx((long)x + (long)y * l0.ext.x + (long)z * (long)l0.ext.x * l0.ext.y);
      const double rr = hr((long)(gx + G) + (long)(gy + G) * re.x + (long)(gz + G) * (long)re.x * re.y);
      const double d = std::fabs(a - rr); if (d > maxdiff) maxdiff = d;
    }
    // np=1 (no Allreduce reordering) is bit-exact; np>1 is the float-op MG-PCG reduction-order floor (~1e-9).
    const double tol = (size == 1) ? 1e-13 : 1e-8;
    if (maxdiff > tol) { ++fail; std::fprintf(stderr, "[rank %d] max|distributed - single-rank CutcellMG| = %.3e (tol %.0e)\n", rank, maxdiff, tol); }
    else if (rank == 0) std::printf("  max|distributed - single-rank| = %.3e (np=%d, tol %.0e)\n", maxdiff, size, tol);
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0) std::printf("OK (np=%d): production CutcellMG MG-PCG distributed == single-rank\n", size);
    else std::fprintf(stderr, "FAILED (np=%d): %d ranks differ\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
