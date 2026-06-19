// cfd-gpu — distributed (MPI) Kokkos grid-halo integration: the foundational pattern for the multi-rank
// Kokkos sdflow.
//
// Solves a periodic implicit-diffusion system ((idiag)I - beta*Lap) u = b by Red-Black Gauss-Seidel, with
// the field decomposed over MPI ranks (ORB) and the per-colour ghost exchange done by transport-core's
// portable GPU-resident halo (DeviceGridExchangeKokkos) instead of a single-process periodic fill. The cfd
// CCField and tpx::View<double> are the SAME Kokkos type (x-fastest, default memory space), so the solver
// field exchanges directly. The distributed RB-GS is algebraically identical to the single-rank sweep (the
// halo supplies the same neighbour values + the GLOBAL red-black parity via the block's global origin), so
// the multi-rank result must equal a serial reference to machine precision. Validates the consumption
// pattern (replace fillGhosts with halo.exchange) the full sdflow MPI port will follow. Runs on any backend.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_stencils_kokkos.hpp"  // cfdk::diffSmoothColor, I3, SField

#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"
#include "tpx/halo/grid_halo_kokkos.hpp"

using tpx::Index;
using tpx::IVec;
using tpx::wrap;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DeviceGridExchangeKokkos;
using tpx::halo::GridHalo;
using cfdk::diffSmoothColor;
using cfdk::I3;
using cfdk::SField;

static constexpr int kDim = 3;
static constexpr int G = 1;                 // diffusion stencil reach
static constexpr double IDIAG = 1.0, BETA = 0.2, AC = IDIAG + 6.0 * BETA;
static constexpr int SWEEPS = 60;

// deterministic periodic source over the global grid.
static double source(int gx, int gy, int gz, IVec<kDim> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) * std::cos(2.0 * M_PI * gy / gs[1]) +
         0.5 * std::sin(2.0 * M_PI * gz / gs[2]);
}

// periodic ghost fill (3 axes, width G) of a single-block extended field on device.
static void periodicFill(SField f, I3 e) {
  cfdk::SExec sp; long st[3] = {1, e.x, (long)e.x * e.y}; int dims[3] = {e.x, e.y, e.z};
  int N3[3] = {e.x - 2 * G, e.y - 2 * G, e.z - 2 * G};
  for (int a = 0; a < 3; ++a) {
    const int b = (a + 1) % 3, c = (a + 2) % 3; const long sa = st[a], sb = st[b], sc = st[c]; const int N = N3[a];
    Kokkos::parallel_for("ref_pfill", Kokkos::MDRangePolicy<cfdk::SExec, Kokkos::Rank<2>>(sp, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) { f(base + (long)gl * sa) = f(base + (long)(gl + N) * sa);
          f(base + (long)(G + N + gl) * sa) = f(base + (long)(G + gl) * sa); } });
    sp.fence();
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    IVec<kDim> gs{24, 20, 16};
    std::array<bool, kDim> periodic{true, true, true};

    // ---- distributed solve ----
    BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gs);
    GridHalo<kDim> halo;
    halo.buildTopology(dec, rank, G, periodic, MPI_COMM_WORLD);
    const auto& idx = halo.indexer();
    const Index n = idx.numCellsInclGhost();
    const auto eg = idx.sizeInclGhost(), og = idx.originInclGhost();
    const I3 e{(int)eg[0], (int)eg[1], (int)eg[2]}, ogv{(int)og[0], (int)og[1], (int)og[2]};

    SField u("u", n), b("b", n);
    auto hb = Kokkos::create_mirror_view(b); auto hu = Kokkos::create_mirror_view(u);
    for (Index c = 0; c < n; ++c) { hb(c) = 0.0; hu(c) = 0.0; }
    idx.forEachInner([&](const IVec<kDim>& l) {
      hb(idx.localMdToLocal(l)) = source(l[0] + og[0], l[1] + og[1], l[2] + og[2], gs); });
    Kokkos::deep_copy(b, hb); Kokkos::deep_copy(u, hu);

    DeviceGridExchangeKokkos<double> dev;
    dev.init(halo);
    SField empty;
    for (int k = 0; k < SWEEPS; ++k)
      for (int color = 0; color < 2; ++color) {
        dev.exchange(u);  // cross-rank + periodic ghost exchange (replaces the single-process fillGhosts)
        diffSmoothColor(u, b, e, ogv, G, BETA, AC, color, empty);
      }
    Kokkos::deep_copy(hu, u);

    // ---- serial reference on the full global grid (every rank, redundant; compare its own block) ----
    const I3 ge{(int)gs[0] + 2 * G, (int)gs[1] + 2 * G, (int)gs[2] + 2 * G}, geo{-G, -G, -G};
    const Index gn = (Index)ge.x * ge.y * ge.z;
    SField ur("ur", gn), br("br", gn);
    auto hbr = Kokkos::create_mirror_view(br);
    for (Index c = 0; c < gn; ++c) hbr(c) = 0.0;
    for (int z = 0; z < gs[2]; ++z) for (int y = 0; y < gs[1]; ++y) for (int x = 0; x < gs[0]; ++x)
      hbr((long)(x + G) + (long)(y + G) * ge.x + (long)(z + G) * (long)ge.x * ge.y) = source(x, y, z, gs);
    Kokkos::deep_copy(br, hbr); Kokkos::deep_copy(ur, 0.0);
    for (int k = 0; k < SWEEPS; ++k)
      for (int color = 0; color < 2; ++color) { periodicFill(ur, ge); diffSmoothColor(ur, br, ge, geo, G, BETA, AC, color, empty); }
    auto hur = Kokkos::create_mirror_view(ur); Kokkos::deep_copy(hur, ur);

    // compare this rank's inner block against the global reference.
    double maxdiff = 0.0;
    idx.forEachInner([&](const IVec<kDim>& l) {
      const int gx = l[0] + og[0], gy = l[1] + og[1], gz = l[2] + og[2];
      const double a = hu(idx.localMdToLocal(l));
      const double r = hur((long)(gx + G) + (long)(gy + G) * ge.x + (long)(gz + G) * (long)ge.x * ge.y);
      const double d = std::fabs(a - r); if (d > maxdiff) maxdiff = d;
    });
    if (maxdiff > 1e-12) { ++fail; std::fprintf(stderr, "[rank %d] max|distributed - serial| = %.3e\n", rank, maxdiff); }
  }

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0) std::printf("OK (np=%d): distributed RB-GS diffusion == serial reference\n", size);
    else std::fprintf(stderr, "FAILED (np=%d): %d ranks differ\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
