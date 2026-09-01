// flow — coarse-level TELESCOPING of the production CutcellMG (docs/MG_TELESCOPING_PLAN.md).
//
// Three gates, all against the validated single-rank CutcellMG on the full grid (every rank runs
// the reference redundantly and compares its own block, the test_cutcellmg_mpi pattern):
//
//  A  telescoping OFF is the baseline this suite already validates (a control run here);
//  B  telescoping FORCED at level 1 on a grid where in-place coarsening is ALSO legal — the two
//     hierarchies solve the same problem to the same tolerance; the answer moves only at the
//     reduction-order floor (removeMean sums on a different partition), like coarse-first vs
//     aligned. The forced path exercises gather / sub-communicator recursion / scatter-add on a
//     hierarchy whose correctness the in-place path already vouches for;
//  C  a STARVED partition — 24^3 on np ranks under a PLAIN, unaligned ORB (handed to initMpi as
//     dec0; the solver's own factory would build a nesting partition at these rank counts, which
//     is exactly what it is for). The blocks turn odd (3 wide) at global 6^3 while the grid could
//     still halve, so the in-place hierarchy freezes those axes and its coarsest grid is NOT the
//     single-rank 3^3. The gate is the design goal itself: with telescoping, at
//     least one telescope happens and the coarsest GLOBAL grid equals the single-rank solver's
//     (the hierarchy is a property of the problem, not of the rank count); and the iteration count
//     matches the single-rank one to +-1 (reduction-order steering of the adaptive PCG). The
//     no-telescope count is printed for the record, not gated.
//
// np=1 has nothing to telescope (one block) and is byte-identical to the single-rank path by
// construction; the gates run at np=2 and 4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "mac_cutcell_mg.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::CutcellMG;

static constexpr int G = 1;

static double source(int gx, int gy, int gz, IVec<3> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) +
         std::cos(4.0 * M_PI * gy / gs[1]) * std::sin(2.0 * M_PI * gz / gs[2]);
}
static void fillSource(CCField b, C3 ext, C3 og, IVec<3> gs) {
  auto h = Kokkos::create_mirror_view(b);
  for (int z = 0; z < ext.z; ++z)
    for (int y = 0; y < ext.y; ++y)
      for (int x = 0; x < ext.x; ++x) {
        const long i = (long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y;
        const bool inner = x >= G && x < ext.x - G && y >= G && y < ext.y - G && z >= G &&
                           z < ext.z - G;
        h(i) = inner ? source(x - G + og.x, y - G + og.y, z - G + og.z, gs) : 0.0;
      }
  Kokkos::deep_copy(b, h);
}

struct Result {
  Kokkos::View<double*, Kokkos::HostSpace> x;
  int iters = 0;
};
static Result setupAndSolve(CutcellMG& mg, C3 ext, C3 og, IVec<3> gs) {
  const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
  CCField ox("ox", n), oy("oy", n), oz("oz", n);  // all-fluid periodic Laplacian
  Kokkos::deep_copy(ox, 1.0);
  Kokkos::deep_copy(oy, 1.0);
  Kokkos::deep_copy(oz, 1.0);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
  CCField b("b", n), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
  fillSource(b, ext, og, gs);
  Result res;
  res.iters = mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/300, /*rtol=*/1e-10, 2, 2, 8);
  res.x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(res.x, x);
  return res;
}

// max |distributed block - single-rank reference| over this rank's inner cells
static double compareBlock(const Result& d, const CutcellMG::Level& l0, const Result& ref,
                           C3 re) {
  double maxdiff = 0.0;
  for (int z = G; z < l0.ext.z - G; ++z)
    for (int y = G; y < l0.ext.y - G; ++y)
      for (int x = G; x < l0.ext.x - G; ++x) {
        const int gx = x - G + l0.og.x, gy = y - G + l0.og.y, gz = z - G + l0.og.z;
        const double a = d.x((long)x + (long)y * l0.ext.x + (long)z * (long)l0.ext.x * l0.ext.y);
        const double rr =
            ref.x((long)(gx + G) + (long)(gy + G) * re.x + (long)(gz + G) * (long)re.x * re.y);
        maxdiff = std::max(maxdiff, std::fabs(a - rr));
      }
  return maxdiff;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int fail = 0;
  {
    struct Case {
      const char* name;
      IVec<3> gs;
      int nlev;
      int force;   // force a telescope at this level (-1 = only where blocked)
      bool tele;   // telescope on
      int bottom;  // agglomeration mode (0 smoother, -1 auto)
    };
    // B: 32^3 at np=2/4 coarsens in place to the bottom; forcing a telescope at level 1 makes
    //    a hierarchy that differs ONLY in where its coarse levels live.
    // C: 48^3 = 2^4*3 halves four times single-rank (48->24->12->6->3); split over np ranks the
    //    blocks turn odd one or two halvings earlier and the in-place hierarchy stops there.
    //    Smoothed bottom on both so the bottom quality is what telescoping changes.
    const Case cases[] = {
        {"A control 32^3 in-place", {32, 32, 32}, 4, -1, false, -1},
        {"B forced telescope@1 32^3", {32, 32, 32}, 4, 1, true, -1},
        {"C0 starved 24^3, no telescope", {24, 24, 24}, 6, -1, false, 0},
        {"C1 starved 24^3, telescope", {24, 24, 24}, 6, -1, true, 0},
    };
    int itersC0 = -1, itersC1 = -1, itersRef = -1, teleC1 = 0;
    C3 cgC0{}, cgC1{}, cgRef{};
    for (const Case& c : cases) {
      if (size == 1 && c.tele)
        continue;  // nothing to telescope on one rank
      CutcellMG mg;
      mg.setAgglomerationMode(c.bottom);
      mg.setTelescope(c.tele);
      mg.setTelescopeForceLevel(c.force);
      peclet::core::decomp::BlockDecomposer<3> plain((std::size_t)size, c.gs);  // unaligned ORB
      mg.initMpi(c.gs[0], c.gs[1], c.gs[2], c.nlev, MPI_COMM_WORLD,
                 c.name[0] == 'C' ? &plain : nullptr);
      CutcellMG::Level& l0 = mg.level(0);
      Result d = setupAndSolve(mg, l0.ext, l0.og, c.gs);

      CutcellMG ref;
      ref.setAgglomerationMode(c.bottom);
      ref.init(c.gs[0], c.gs[1], c.gs[2], c.nlev);
      const C3 re{c.gs[0] + 2 * G, c.gs[1] + 2 * G, c.gs[2] + 2 * G}, ro{0, 0, 0};
      Result r = setupAndSolve(ref, re, ro, c.gs);

      double maxdiff = compareBlock(d, l0, r, re), gmax = 0.0;
      MPI_Allreduce(&maxdiff, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      // reduction-order floor for MG-PCG with the float-stored operator: ~1e-9 (see
      // test_cutcellmg_mpi); np=1 bit-exact.
      const double tol = (size == 1) ? 1e-13 : 1e-7;
      const bool ok = gmax <= tol;
      if (!ok)
        ++fail;
      if (rank == 0)
        std::printf("  %-42s np=%d levels(rank0)=%d iters=%3d  max|dist-ref|=%.3e  %s\n", c.name,
                    size, mg.localLevels(), d.iters, gmax, ok ? "ok" : "FAIL");
      if (c.name[0] == 'C' && c.name[1] == '0') {
        itersC0 = d.iters;
        cgC0 = mg.coarsestGlobal();
      }
      if (c.name[0] == 'C' && c.name[1] == '1') {
        itersC1 = d.iters;
        itersRef = r.iters;
        teleC1 = mg.telescopeCount();
        cgC1 = mg.coarsestGlobal();
        cgRef = ref.coarsestGlobal();
      }
    }
    if (size > 1 && itersC1 >= 0) {
      // rank 0 is always a group root, so it holds every level: its coarsestGlobal() is the
      // hierarchy's. Depth-equivalence with the single-rank solver + a telescope actually occurred.
      int tele0 = teleC1;
      MPI_Bcast(&tele0, 1, MPI_INT, 0, MPI_COMM_WORLD);
      int cg[3] = {cgC1.x, cgC1.y, cgC1.z};
      MPI_Bcast(cg, 3, MPI_INT, 0, MPI_COMM_WORLD);
      const bool sameBottom = (cg[0] == cgRef.x && cg[1] == cgRef.y && cg[2] == cgRef.z);
      const bool ok = (tele0 >= 1) && sameBottom && (std::abs(itersC1 - itersRef) <= 1);
      if (!ok)
        ++fail;
      if (rank == 0)
        std::printf("  starved 24^3: coarsest global no-telescope %dx%dx%d, telescope %dx%dx%d "
                    "(%d telescope%s), single-rank %dx%dx%d; iterations %d / %d / %d  %s\n",
                    cgC0.x, cgC0.y, cgC0.z, cg[0], cg[1], cg[2], tele0, tele0 == 1 ? "" : "s",
                    cgRef.x, cgRef.y, cgRef.z, itersC0, itersC1, itersRef,
                    ok ? "ok" : "FAIL (telescoped hierarchy != single-rank)");
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): telescoped CutcellMG == single-rank reference\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
