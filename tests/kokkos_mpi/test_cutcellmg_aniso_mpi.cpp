// G4 under MPI — the ANISOTROPIC pressure hierarchy, multi-rank
// (flow/doc/anisotropic_metric.md §8.5 last bullet + §5.3/§5.4 and trap 6; Phase 2 commit C3).
//
// The SAME problem the single-rank `cutcellmg_aniso` gate solves — the Zick & Homsy SC sphere
// (phi = 0.216, R = 0.3722 L) on the stretched box (N, 2N, N/2) of the cube L^3, apertures from
// the solver's own `buildOpenness` with hp = (1, 1/2, 2), weights w = (1, 4, 1/4) — driven through
// the PRODUCTION `CutcellMG::initMpi` and compared against the validated single-rank `init()`.
//
// Four statements:
//   1. the LEVEL TABLE the aspect rule chooses is IDENTICAL on every rank.  It has to be: every
//      rank computes it from the same doubles and the same replicated decomposition, with no
//      communication (§5.3).  It is also the single-rank table wherever the even-block gate does
//      not bite.
//   2. np = 1 is BIT-EXACT to the single-rank solve (the gating is byte-identical), np = 2/4 agree
//      to the MG-PCG reduction-order floor — the tolerance pattern of `test_cutcellmg_mpi`.
//   3. TRAP 6: an axis DEFERRED by the aspect rule is NOT "blocked".  With the economic trigger
//      off (`setTelescopeMinExtent(0)`, so a merge happens only when an axis that CAN coarsen
//      globally is not even on every rank) and telescoping ON, NO level telescopes — even though
//      the rule defers x and z for two levels.  Deferral alone must never merge ranks.
//   4. a FORCED telescope at level 1 (`setTelescopeForceLevel(1)`) still reaches the same answer
//      on the deferred hierarchy: the merge path and the aspect rule compose.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "mac_cutcell.hpp"  // buildOpenness (the solver's own aperture model)
#include "mac_cutcell_mg.hpp"
#include "peclet/core/common/types.hpp"

using peclet::core::IVec;
using peclet::flow::buildOpenness;
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::CutcellMG;

static constexpr int G = 1, NLEV = 6;
static constexpr int N = 32;                        // stretched grid (N, 2N, N/2) == N^3 cells
static constexpr double LBOX = 1.0, RSPH = 0.3722;  // phi = 0.216 in the unit cell

// d' = (|x - c|_minimage - R)/dref at the PHYSICAL centre of the GLOBAL cell (gx, gy, gz).
static double sdfAt(int gx, int gy, int gz, int nx, int ny, int nz, double dref) {
  double px = (gx + 0.5) * (LBOX / nx) - 0.5 * LBOX, py = (gy + 0.5) * (LBOX / ny) - 0.5 * LBOX,
         pz = (gz + 0.5) * (LBOX / nz) - 0.5 * LBOX;
  px -= LBOX * std::round(px / LBOX);
  py -= LBOX * std::round(py / LBOX);
  pz -= LBOX * std::round(pz / LBOX);
  return (std::sqrt(px * px + py * py + pz * pz) - RSPH * LBOX) / dref;
}

// Assemble the operator on `mg`'s level-0 block and solve MG-PCG to 1e-10; returns the host
// iterate. `og` is this block's global inner origin (0 single-rank).
using HostVec = decltype(Kokkos::create_mirror_view(std::declval<CCField>()));
static HostVec setupAndSolve(CutcellMG& mg, C3 ext, C3 og, int nx, int ny, int nz, bool collective,
                             int* iters) {
  const double hx = LBOX / nx, hy = LBOX / ny, hz = LBOX / nz, dref = hx;
  const double hp[3] = {hx / dref, hy / dref, hz / dref};
  const double wx = 1.0 / (hp[0] * hp[0]), wy = 1.0 / (hp[1] * hp[1]), wz = 1.0 / (hp[2] * hp[2]);
  const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
  CCField sdf("sdf", n), ox("ox", n), oy("oy", n), oz("oz", n);
  {
    auto m = Kokkos::create_mirror_view(sdf);
    for (int z = 0; z < ext.z; ++z)
      for (int y = 0; y < ext.y; ++y)
        for (int x = 0; x < ext.x; ++x)
          m((long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y) =
              sdfAt(x - G + og.x, y - G + og.y, z - G + og.z, nx, ny, nz, dref);
    Kokkos::deep_copy(sdf, m);
  }
  buildOpenness(ox, oy, oz, CCConst(sdf), ext, hp[0], hp[1], hp[2], 1);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), wx, wy, wz);

  // The RHS of the single-rank gate: the smooth periodic mode on the rows that carry an open
  // face, mean-removed over exactly those rows.  The mean is GLOBAL, so it is an Allreduce here
  // and a local sum single-rank — the same number either way (a plain sum of the same set).
  std::vector<double> hb(n, 0.0);
  {
    auto hAC = Kokkos::create_mirror_view(mg.level(0).AC);
    Kokkos::deep_copy(hAC, mg.level(0).AC);
    const double k = 2.0 * M_PI / LBOX;
    double loc[2] = {0.0, 0.0};
    for (int z = G; z < ext.z - G; ++z)
      for (int y = G; y < ext.y - G; ++y)
        for (int x = G; x < ext.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * ext.x + (std::size_t)z * (std::size_t)ext.x * ext.y;
          if (!(hAC(i) > 0.0f))
            continue;
          const double px = (x - G + og.x + 0.5) * (LBOX / nx),
                       py = (y - G + og.y + 0.5) * (LBOX / ny),
                       pz = (z - G + og.z + 0.5) * (LBOX / nz);
          hb[i] = std::cos(k * px) * std::cos(k * py) * std::cos(k * pz);
          loc[0] += hb[i];
          loc[1] += 1.0;
        }
    double glob[2] = {loc[0], loc[1]};
    if (collective)  // the distributed build owns one block per rank; the reference owns them all
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double mean = glob[1] > 0.0 ? glob[0] / glob[1] : 0.0;
    for (int z = G; z < ext.z - G; ++z)
      for (int y = G; y < ext.y - G; ++y)
        for (int x = G; x < ext.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * ext.x + (std::size_t)z * (std::size_t)ext.x * ext.y;
          if (hAC(i) > 0.0f)
            hb[i] -= mean;
        }
  }
  CCField b("b", n), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
  {
    auto m = Kokkos::create_mirror_view(b);
    for (std::size_t i = 0; i < n; ++i)
      m(i) = hb[i];
    Kokkos::deep_copy(b, m);
  }
  *iters = mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/500, /*rtol=*/1e-10, 2, 2, 12);
  auto hx2 = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(hx2, x);
  return hx2;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int nx = N, ny = 2 * N, nz = N / 2;
    const double hp[3] = {1.0, 0.5, 2.0};

    // ---- the single-rank reference hierarchy (every rank builds it redundantly) ----------------
    CutcellMG ref;
    ref.setMetric(hp);
    ref.init(nx, ny, nz, NLEV);
    const C3 re{nx + 2 * G, ny + 2 * G, nz + 2 * G}, ro{0, 0, 0};
    int refIt = 0;
    auto hr = setupAndSolve(ref, re, ro, nx, ny, nz, /*collective=*/false, &refIt);
    const std::vector<C3> refTab = ref.levelRatios();

    // ---- PASS 1: the distributed hierarchy, telescoping ON, economic trigger OFF ---------------
    // so a merge can only be triggered by a genuinely BLOCKED axis (trap 6).
    CutcellMG mg;
    mg.setMetric(hp);
    mg.setTelescope(true);
    mg.setTelescopeMinExtent(0);
    mg.initMpi(nx, ny, nz, NLEV, MPI_COMM_WORLD);
    CutcellMG::Level& l0 = mg.level(0);
    int it = 0;
    auto hx = setupAndSolve(mg, l0.ext, l0.og, nx, ny, nz, /*collective=*/true, &it);
    const std::vector<C3> tab = mg.levelRatios();

    // 1. the level table is the same on every rank (a pure function of replicated data).
    {
      int nl = (int)tab.size(), nlMin = 0, nlMax = 0;
      MPI_Allreduce(&nl, &nlMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&nl, &nlMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (nlMin != nlMax) {
        ++fail;
        std::fprintf(stderr, "[rank %d] level COUNT differs across ranks: %d..%d\n", rank, nlMin,
                     nlMax);
      } else {
        std::vector<int> mine(3 * (std::size_t)nl), lo(3 * (std::size_t)nl),
            hi(3 * (std::size_t)nl);
        for (int L = 0; L < nl; ++L) {
          mine[3 * (std::size_t)L] = tab[(std::size_t)L].x;
          mine[3 * (std::size_t)L + 1] = tab[(std::size_t)L].y;
          mine[3 * (std::size_t)L + 2] = tab[(std::size_t)L].z;
        }
        MPI_Allreduce(mine.data(), lo.data(), 3 * nl, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(mine.data(), hi.data(), 3 * nl, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        for (int i = 0; i < 3 * nl; ++i)
          if (lo[(std::size_t)i] != hi[(std::size_t)i]) {
            ++fail;
            std::fprintf(stderr, "[rank %d] level table entry %d differs across ranks\n", rank, i);
            break;
          }
      }
      if (rank == 0) {
        std::printf("  level table (np=%d):", size);
        for (const C3& rr : tab)
          std::printf(" (%d,%d,%d)", rr.x, rr.y, rr.z);
        std::printf("\n  single-rank reference: ");
        for (const C3& rr : refTab)
          std::printf(" (%d,%d,%d)", rr.x, rr.y, rr.z);
        std::printf("\n");
      }
      // The aspect rule's own signature, whatever the even-block gate does below it.
      if (!(tab[0].x == 1 && tab[0].y == 2 && tab[0].z == 1)) {
        ++fail;
        std::fprintf(stderr, "[rank %d] level 0 ratio (%d,%d,%d), expected (1,2,1)\n", rank,
                     tab[0].x, tab[0].y, tab[0].z);
      }
    }

    // 3. TRAP 6: no level telescoped — deferral is not blocking.
    {
      int tele = 0;
      for (int L = 0; L < mg.nLevels(); ++L)
        if (mg.level(L).tele)
          ++tele;
      int teleAll = 0;
      MPI_Allreduce(&tele, &teleAll, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (teleAll != 0) {
        ++fail;
        std::fprintf(stderr,
                     "[rank %d] %d level(s) telescoped with the economic trigger off — an "
                     "aspect-DEFERRED axis was treated as BLOCKED (trap 6)\n",
                     rank, teleAll);
      } else if (rank == 0)
        std::printf("  trap 6: no telescope on the deferred hierarchy (min-extent trigger off)\n");
    }

    // 2. against the single-rank solve.
    double xref = 0.0;  // the reference solution's scale, for the RELATIVE floor below
    for (std::size_t i = 0; i < hr.extent(0); ++i)
      if (!(std::fabs(hr(i)) <= xref))
        xref = std::fabs(hr(i));
    auto blockDiff = [&](HostVec h, C3 ext, C3 og) {
      double m = 0.0;
      for (int z = G; z < ext.z - G; ++z)
        for (int y = G; y < ext.y - G; ++y)
          for (int x = G; x < ext.x - G; ++x) {
            const int gx = x - G + og.x, gy = y - G + og.y, gz = z - G + og.z;
            const double a = h((long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y);
            const double rr =
                hr((long)(gx + G) + (long)(gy + G) * re.x + (long)(gz + G) * (long)re.x * re.y);
            const double d = std::fabs(a - rr);
            if (!(d <= m))
              m = d;
          }
      return m;
    };
    // np = 1 is BIT-EXACT (the distributed gating is byte-identical there).  np > 1 is compared
    // RELATIVE to the solution scale, at the MG-PCG's own floor: the driver stops on a RESIDUAL
    // (rtol 1e-10 of |b|inf), and on a CUT-CELL operator a small-aperture row has a tiny effective
    // eigenvalue, so the same residual leaves a much larger solution spread than the all-fluid
    // hierarchy of test_cutcellmg_mpi does.  Measured here (max|phi| = 14.67): 1.01e-07 /
    // 2.97e-07 relative at np = 2 / 4, and a -DPECLET_FLOW_MREAL_DOUBLE build of the identical
    // source moves it only 4x (to 2.46e-08 / 1.21e-07), so it is the stopping rule and NOT the
    // float operator storage.
    // The iteration count -- the decomposition-independence statement that actually has teeth --
    // is gated to within 2 of the single-rank one.
    const double tol = (size == 1) ? 0.0 : 1e-6 * xref;
    const double d1 = blockDiff(hx, l0.ext, l0.og);
    if (!(d1 <= tol)) {
      ++fail;
      std::fprintf(stderr,
                   "[rank %d] pass 1 max|distributed - single-rank| = %.3e (rel %.3e, tol %.3e)\n",
                   rank, d1, d1 / xref, tol);
    } else if (rank == 0)
      std::printf(
          "  pass 1: %d iters (single-rank %d), max|dist - single| = %.3e = %.3e relative "
          "of max|phi| = %.4f (np=%d%s)\n",
          it, refIt, d1, d1 / xref, xref, size, size == 1 ? ", BIT-EXACT" : "");
    if (std::abs(it - refIt) > 2) {
      ++fail;
      std::fprintf(stderr, "[rank %d] pass 1 iterations %d vs single-rank %d (allowed +/-2)\n",
                   rank, it, refIt);
    }
    if (it >= 500 || refIt >= 500) {
      ++fail;
      std::fprintf(stderr, "[rank %d] pass 1 CAPPED (%d / %d iters)\n", rank, it, refIt);
    }

    // ---- PASS 2: a FORCED telescope at level 1 on the deferred hierarchy -----------------------
    if (size > 1) {
      CutcellMG mg2;
      mg2.setMetric(hp);
      mg2.setTelescope(true);
      mg2.setTelescopeMinExtent(0);
      mg2.setTelescopeForceLevel(1);
      mg2.initMpi(nx, ny, nz, NLEV, MPI_COMM_WORLD);
      CutcellMG::Level& m0 = mg2.level(0);
      int it2 = 0;
      auto hx2 = setupAndSolve(mg2, m0.ext, m0.og, nx, ny, nz, /*collective=*/true, &it2);
      int forced = mg2.level(1).tele ? 1 : 0, forcedAll = 0;
      MPI_Allreduce(&forced, &forcedAll, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      const double d2 = blockDiff(hx2, m0.ext, m0.og);
      if (forcedAll != 1) {
        ++fail;
        std::fprintf(stderr, "[rank %d] pass 2: setTelescopeForceLevel(1) did not merge\n", rank);
      }
      if (!(d2 <= 1e-6 * xref)) {
        ++fail;
        std::fprintf(stderr, "[rank %d] pass 2 max|telescoped - single-rank| = %.3e (rel %.3e)\n",
                     rank, d2, d2 / xref);
      } else if (rank == 0)
        std::printf(
            "  pass 2 (telescope forced at level 1): %d iters, max|dist - single| = %.3e "
            "= %.3e relative\n",
            it2, d2, d2 / xref);
      if (rank == 0 && mg2.level(0).ratio.x == 1 && mg2.level(0).ratio.y == 2)
        std::printf("  pass 2: the aspect rule still owns level 0 across the merge\n");
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): the anisotropic CutcellMG hierarchy, distributed\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d rank(s) failed\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail ? 1 : 0;
}
