// cfd-gpu — the PRODUCTION VelocityMG (staircase velocity multigrid), multi-rank.
//
// Folds the distributed halo into the real velocity-MG (gated behind CFD_MPI). VelocityMG has NO global
// reductions (the backward-Euler velocity operator is non-singular -> no mean removal, and the V-cycle is a
// fixed iteration, not a Krylov method), so the fold is purely fill()->per-level halo exchange + the block-
// origin red-black parity -- and the distributed V-cycle is therefore BIT-tight to the single-rank one (no
// Allreduce reordering at all). Solves an all-fluid periodic backward-Euler diffusion (fine stencil =
// idiag*I - beta*Lap; staircase coarse op with theta=1, no pin, full clean-fluid mask) distributed (initMpi)
// vs single-rank (init() on the full grid). Build with -DCFD_MPI.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

#include "mac_velocity_mg.hpp"

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo_topology.hpp"

using tpx::IVec;
using sdflow::VelocityMG; using sdflow::CCField; using sdflow::CCConst; using sdflow::C3; using sdflow::FPV; using sdflow::FPC;

static constexpr int G = 2, NLEV = 4, NVCYC = 6;
static constexpr double IDIAG = 1.0, BETA = 0.2;
static double source(int gx, int gy, int gz, IVec<3> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) + std::cos(4.0 * M_PI * gy / gs[1]) * std::sin(2.0 * M_PI * gz / gs[2]);
}

static void fillSource(CCField b, C3 ext, C3 og, IVec<3> gs) {
  auto hb = Kokkos::create_mirror_view(b);
  for (std::size_t c = 0; c < b.extent(0); ++c) hb(c) = 0.0;
  for (int z = G; z < ext.z - G; ++z) for (int y = G; y < ext.y - G; ++y) for (int x = G; x < ext.x - G; ++x)
    hb((long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y) = source(x - G + og.x, y - G + og.y, z - G + og.z, gs);
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

    auto setupAndSolve = [&](VelocityMG& vmg, C3 ext, C3 og) {
      const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
      // fine stencil = backward-Euler diffusion (idiag+6beta diagonal, -beta off-diagonals)
      FPV AC("AC", n), AW("AW", n), AE("AE", n), AS("AS", n), AN("AN", n), AB("AB", n), AT("AT", n);
      Kokkos::deep_copy(AC, (float)(IDIAG + 6.0 * BETA));
      for (FPV* p : {&AW, &AE, &AS, &AN, &AB, &AT}) Kokkos::deep_copy(*p, (float)(-BETA));
      vmg.setFineStencil(FPC(AC), FPC(AW), FPC(AE), FPC(AS), FPC(AN), FPC(AB), FPC(AT));
      // all-fluid geometry: theta=1, no solid pin, full clean-fluid mask
      CCField th("th", n), solid("solid", n), clean("clean", n);
      Kokkos::deep_copy(th, 1.0); Kokkos::deep_copy(solid, 0.0); Kokkos::deep_copy(clean, 1.0);
      vmg.setStaircase(CCConst(th), CCConst(solid), CCConst(clean), BETA, IDIAG, 0.5);
      CCField b("b", n), x("x", n);
      fillSource(b, ext, og, gs); Kokkos::deep_copy(x, 0.0);
      vmg.solve(CCConst(b), x, NVCYC, 2, 2, 8);
      auto hx = Kokkos::create_mirror_view(x); Kokkos::deep_copy(hx, x);
      return hx;
    };

    VelocityMG vmg; vmg.initMpi(gs[0], gs[1], gs[2], NLEV, MPI_COMM_WORLD);
    VelocityMG::Level& l0 = vmg.level(0);
    auto hx = setupAndSolve(vmg, l0.ext, l0.og);

    VelocityMG ref; ref.init(gs[0], gs[1], gs[2], NLEV);
    const C3 re{(int)gs[0] + 2 * G, (int)gs[1] + 2 * G, (int)gs[2] + 2 * G}, ro{0, 0, 0};
    auto hr = setupAndSolve(ref, re, ro);

    double maxdiff = 0.0;
    for (int z = G; z < l0.ext.z - G; ++z) for (int y = G; y < l0.ext.y - G; ++y) for (int x = G; x < l0.ext.x - G; ++x) {
      const int gx = x - G + l0.og.x, gy = y - G + l0.og.y, gz = z - G + l0.og.z;
      const double a = hx((long)x + (long)y * l0.ext.x + (long)z * (long)l0.ext.x * l0.ext.y);
      const double rr = hr((long)(gx + G) + (long)(gy + G) * re.x + (long)(gz + G) * (long)re.x * re.y);
      const double d = std::fabs(a - rr); if (d > maxdiff) maxdiff = d;
    }
    if (maxdiff > 1e-11) { ++fail; std::fprintf(stderr, "[rank %d] max|distributed - single-rank VelocityMG| = %.3e\n", rank, maxdiff); }
    else if (rank == 0) std::printf("  max|distributed - single-rank| = %.3e (np=%d)\n", maxdiff, size);
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0) std::printf("OK (np=%d): production VelocityMG V-cycle distributed == single-rank\n", size);
    else std::fprintf(stderr, "FAILED (np=%d): %d ranks differ\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
