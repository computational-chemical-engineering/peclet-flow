// flow — the distributed host 'direct' pressure bottom (doc/vof_step_performance_design.md §14.3
// H-6): every rank holding the bottom solves the allgathered GLOBAL bottom gb_ redundantly.
//
// Why this test exists (WO-H6 review, findings 1 and 2): a wrong gb_ (a bad global-id map, gather
// order or boundary fill) only weakens the preconditioner, and the outer PCG still converges to
// rtol, so a tolerance comparison against a reference cannot see it. ITERATION IDENTITY can: the
// direct bottom and GraphAMG both solve the same global bottom to their tolerances, so their
// preconditioners agree to the bottom's round-off and the outer PCG takes the same number of
// iterations every step. Gates, on a walled (+-y), cut-cell (a sphere's fractional openness),
// variable-coefficient operator rebuilt every "step" (a dense blob moving through the domain):
//
//  A  in place: no telescope, the bottom on every rank;
//  B  a FORCED TELESCOPE at level 1: the bottom lives on the group roots only (bt.comm = roots,
//     the roots' og / gdim), the other ranks idle below the telescope point;
//  C  a REPARTITION stage (a weighted level-0 ORB with an odd root split + setRepartition, the
//     test_telescope_mpi case D knob);
//  in each: explicit 'direct' (it raises when not taken) vs explicit 'algebraic', pressure
//  iterations identical EVERY step and the final state's max|x_d - x_a| / max|x_a| <= 1e-12; B and
//  C also assert that the bottom actually sits below a telescope / repartition point (fewer ranks
//  hold it than run) and is agglomerated (A..C: 'direct' did not raise).
//  H  review note 1: an explicit 'direct' that is INELIGIBLE under a telescope (a 16^3 bottom: no
//     axis gives planes of at most 192 cells) raises on EVERY rank -- a rank idling below the
//     telescope point used to block in the transition scatter while the roots raised.
//
// Host backends only: the device keeps GraphAMG distributed (§13.9 Q-D7; 'direct' raises there by
// design), so on a device build the test reports SKIP (exit 77). np = 2 and 4.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

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
static constexpr int STEPS = 6;

static int wrap(int a, int n) {
  return ((a % n) + n) % n;
}

// Face coefficient (x-face of cell (gx, gy, gz) sits at gx - 1/2): the sphere's open fraction
// times 1 / rho of a dense blob moving along z with the step; 1 everywhere on a wall-free axis.
struct Scene {
  IVec<3> gs;
  int step = 0;
  double open(double x, double y, double z) const {
    const double cx = 0.5 * gs[0], cy = 0.5 * gs[1], cz = 0.5 * gs[2], R = 0.22 * gs[1];
    const double phi =
        std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)) - R;
    return std::clamp(phi + 0.5, 0.0, 1.0);
  }
  double coef(double x, double y, double z) const {
    const double bz = std::fmod(0.2 * gs[2] + 3.0 * step, (double)gs[2]);
    const double w = 0.12 * gs[0], dx = x - 0.3 * gs[0], dy = y - 0.35 * gs[1], dz = z - bz;
    return 1.0 / (1.0 + 9.0 * std::exp(-(dx * dx + dy * dy + dz * dz) / (w * w)));
  }
  double face(int a, double x, double y, double z) const {  // cell-centre coordinates
    if (a == 0)
      x -= 0.5;
    if (a == 1)
      y -= 0.5;
    if (a == 2)
      z -= 0.5;
    return open(x, y, z) * coef(x, y, z);
  }
};

// A cell with an open face (the operator's row is not empty).
static bool fluidCell(const Scene& sc, double cx, double cy, double cz) {
  return sc.open(cx - 0.5, cy, cz) + sc.open(cx + 0.5, cy, cz) + sc.open(cx, cy - 0.5, cz) +
             sc.open(cx, cy + 0.5, cz) + sc.open(cx, cy, cz - 0.5) + sc.open(cx, cy, cz + 0.5) >
         0.0;
}

// The rhs is zero in solid cells (all six faces closed) and has zero sum over the fluid cells,
// so the singular walled / cut-cell problem is compatible (one fluid component).
static void fillScene(CCField ox, CCField oy, CCField oz, CCField b, C3 ext, C3 og,
                      const Scene& sc) {
  auto hx = Kokkos::create_mirror_view(ox), hy = Kokkos::create_mirror_view(oy),
       hz = Kokkos::create_mirror_view(oz), hb = Kokkos::create_mirror_view(b);
  const IVec<3> gs = sc.gs;
  double loc[2] = {0.0, 0.0}, glb[2];
  for (int pass = 0; pass < 2; ++pass) {
    for (int z = 0; z < ext.z; ++z)
      for (int y = 0; y < ext.y; ++y)
        for (int x = 0; x < ext.x; ++x) {
          const long i = (long)x + (long)y * ext.x + (long)z * (long)ext.x * ext.y;
          // periodic x, z (wrapped ghosts); walls on y (the boundary fill closes the wall faces)
          const int gx = wrap(x - G + og.x, (int)gs[0]), gy = y - G + og.y,
                    gz = wrap(z - G + og.z, (int)gs[2]);
          const double cx = gx + 0.5, cy = gy + 0.5, cz = gz + 0.5;
          const bool inner =
              x >= G && x < ext.x - G && y >= G && y < ext.y - G && z >= G && z < ext.z - G;
          const bool fluid = fluidCell(sc, cx, cy, cz);
          const double f = std::sin(2.0 * M_PI * gx / gs[0]) * std::cos(M_PI * (gy + 0.5) / gs[1]) +
                           0.5 * std::sin(2.0 * M_PI * (gz + 0.3 * sc.step) / gs[2]) +
                           0.1 * (gy + 0.5) / gs[1];
          if (pass == 0) {
            hx(i) = sc.face(0, cx, cy, cz);
            hy(i) = sc.face(1, cx, cy, cz);
            hz(i) = sc.face(2, cx, cy, cz);
            if (inner && fluid) {
              loc[0] += f;
              loc[1] += 1.0;
            }
          } else {
            hb(i) = (inner && fluid) ? f - glb[0] / glb[1] : 0.0;
          }
        }
    if (pass == 0)
      MPI_Allreduce(loc, glb, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  Kokkos::deep_copy(ox, hx);
  Kokkos::deep_copy(oy, hy);
  Kokkos::deep_copy(oz, hz);
  Kokkos::deep_copy(b, hb);
}

struct Run {
  std::vector<int> iters;
  Kokkos::View<double*, Kokkos::HostSpace> x;
  int teleCount = 0, repCount = 0, localLevels = 0;
  C3 coarsest{}, ext{}, og{};
};

static void configure(CutcellMG& mg, int force, bool repart) {
  const int bc[6] = {0, 0, 1, 1, 0, 0};  // walls on -y / +y (hasBC_), periodic x and z
  mg.setBoundaryConditions(bc);
  mg.setAgglomerationMode(-1);  // 'auto' agglomeration (set_pressure_bottom_extent's path)
  mg.setTelescope(force >= 0 || repart);  // a Repartition stage is a telescope point
  mg.setTelescopeForceLevel(force);
  mg.setRepartition(repart);
}

static Run solveSteps(IVec<3> gs, int nlev, int force, bool repart,
                      peclet::core::decomp::BlockDecomposer<3>* dec0, int engine) {
  CutcellMG mg;
  configure(mg, force, repart);
  mg.setBottomSolver(engine);
  mg.initMpi(gs[0], gs[1], gs[2], nlev, MPI_COMM_WORLD, dec0);
  CutcellMG::Level& l0 = mg.level(0);
  const C3 ext = l0.ext, og = l0.og;
  const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
  CCField ox("ox", n), oy("oy", n), oz("oz", n), b("b", n), x("x", n), r("r", n), p("p", n),
      z("z", n), Ap("Ap", n);
  Run res;
  Scene sc{gs, 0};
  for (int s = 0; s < STEPS; ++s) {
    sc.step = s;
    fillScene(ox, oy, oz, b, ext, og, sc);
    mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);  // every step
    Kokkos::deep_copy(x, 0.0);
    res.iters.push_back(mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/300, /*rtol=*/1e-10, 2, 2, 8));
  }
  res.ext = ext;
  res.og = og;
  res.x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(res.x, x);
  res.teleCount = mg.telescopeCount();
  res.repCount = mg.repartitionCount();
  res.localLevels = mg.localLevels();
  res.coarsest = mg.coarsestGlobal();
  return res;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int fail = 0;
  constexpr bool host = std::is_same_v<peclet::flow::CCExec::memory_space, Kokkos::HostSpace>;
  if (!host) {
    if (rank == 0)
      std::printf("SKIP: the distributed 'direct' bottom is host-only (device keeps GraphAMG)\n");
    Kokkos::finalize();
    MPI_Finalize();
    return 77;
  }
  {
    struct Case {
      const char* name;
      IVec<3> gs;
      int nlev, force;
      bool repart, weighted;
    };
    const Case cases[] = {
        {"A in place 64x48x32", {64, 48, 32}, 4, -1, false, false},
        {"B forced telescope@1 64x48x32", {64, 48, 32}, 4, 1, false, false},
        {"C repartition 48x32x32 (weighted, odd root)", {48, 32, 32}, 4, -1, true, true},
    };
    for (const Case& c : cases) {
      peclet::core::decomp::BlockDecomposer<3>* dec0 = nullptr;
      std::vector<double> w;
      if (c.weighted) {  // test_telescope_mpi case D: a slab x < 3 weighs 20x -> odd root split
        w.resize((std::size_t)c.gs[0] * c.gs[1] * c.gs[2]);
        for (Index zz = 0; zz < c.gs[2]; ++zz)
          for (Index yy = 0; yy < c.gs[1]; ++yy)
            for (Index xx = 0; xx < c.gs[0]; ++xx)
              w[(std::size_t)(xx + yy * c.gs[0] + zz * c.gs[0] * c.gs[1])] = xx < 3 ? 20.0 : 1.0;
        dec0 = new peclet::core::decomp::BlockDecomposer<3>((std::size_t)size, c.gs, w);
      }
      Run rd, ra;
      std::string err;
      try {
        rd = solveSteps(c.gs, c.nlev, c.force, c.repart, dec0, CutcellMG::kBottomDirect);
        ra = solveSteps(c.gs, c.nlev, c.force, c.repart, dec0, CutcellMG::kBottomAlgebraic);
      } catch (const std::exception& e) {
        err = e.what();
      }
      delete dec0;
      int bad = err.empty() ? 0 : 1, anyBad = 0;
      MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (anyBad) {
        if (rank == 0)
          std::printf("  %-44s np=%d FAIL: %s\n", c.name, size,
                      err.empty() ? "raised on another rank" : err.c_str());
        ++fail;
        continue;
      }
      // the bottom's placement: ranks holding every level hold the bottom
      int maxLv = 0, nBottom = 0;
      MPI_Allreduce(&rd.localLevels, &maxLv, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      int mine = rd.localLevels == maxLv ? 1 : 0;
      MPI_Allreduce(&mine, &nBottom, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      int stages[2] = {rd.teleCount, rd.repCount};
      MPI_Bcast(stages, 2, MPI_INT, 0, MPI_COMM_WORLD);  // rank 0 always holds every level
      int cg[3] = {rd.coarsest.x, rd.coarsest.y, rd.coarsest.z};
      MPI_Bcast(cg, 3, MPI_INT, 0, MPI_COMM_WORLD);
      // over this rank's FLUID inner cells: a solid cell's row is empty, so its x is whatever the
      // preconditioner left there (the two bottoms treat a solid bottom cell differently) and is no
      // part of the solution; ghosts are not either.
      const C3 e = rd.ext, o = rd.og;
      const Scene sc{c.gs, STEPS - 1};
      double loc[2] = {0.0, 0.0}, glb[2];
      for (int zz = G; zz < e.z - G; ++zz)
        for (int yy = G; yy < e.y - G; ++yy)
          for (int xx = G; xx < e.x - G; ++xx) {
            if (!fluidCell(sc, xx - G + o.x + 0.5, yy - G + o.y + 0.5, zz - G + o.z + 0.5))
              continue;
            const long i = (long)xx + (long)yy * e.x + (long)zz * (long)e.x * e.y;
            loc[0] = std::max(loc[0], std::fabs(rd.x(i) - ra.x(i)));
            loc[1] = std::max(loc[1], std::fabs(ra.x(i)));
          }
      MPI_Allreduce(loc, glb, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const double rel = glb[0] / (glb[1] + 1e-300);
      const bool sameIt = rd.iters == ra.iters;
      bool placed = true;
      if (c.force >= 0)
        placed = stages[0] >= 1 && nBottom < size;
      if (c.repart)
        placed = stages[1] >= 1;  // the bottom's blocks are a proportional ORB's, not dec0's
      const bool ok = sameIt && rel <= 1e-12 && placed;
      if (!ok)
        ++fail;
      if (rank == 0) {
        std::printf(
            "  %-44s np=%d bottom %dx%dx%d on %d/%d rank(s), %d telescope / %d repartition "
            "stage(s); iters direct/algebraic:",
            c.name, size, cg[0], cg[1], cg[2], nBottom, size, stages[0], stages[1]);
        for (std::size_t s = 0; s < rd.iters.size(); ++s)
          std::printf(" %d/%d", rd.iters[s], ra.iters[s]);
        std::printf("  rel %.3e  %s%s\n", rel, ok ? "ok" : "FAIL",
                    placed ? "" : " (the bottom is not below a telescope / repartition point)");
      }
    }

    // H: an ineligible explicit 'direct' under a telescope raises on every rank (no hang).
    {
      const IVec<3> gs{64, 64, 64};  // 3 levels -> a 16^3 bottom: planes of 256 > 192 cells
      std::string err;
      int lvls = 0;
      try {
        Run r = solveSteps(gs, 3, 1, false, nullptr, CutcellMG::kBottomDirect);
        lvls = r.localLevels;
      } catch (const std::runtime_error& e) {
        err = e.what();
      }
      int raised = (err.find("not eligible here") != std::string::npos &&
                    err.find("192 cells") != std::string::npos)
                       ? 1
                       : 0,
          nRaised = 0;
      MPI_Allreduce(&raised, &nRaised, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      const bool ok = nRaised == size;
      if (!ok)
        ++fail;
      if (rank == 0)
        std::printf("  %-44s np=%d raised on %d/%d ranks%s  %s\n",
                    "H ineligible 'direct' under a telescope", size, nRaised, size,
                    err.empty() ? "" : (" (\"" + err + "\")").c_str(), ok ? "ok" : "FAIL");
      (void)lvls;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): distributed 'direct' bottom == 'algebraic' (iterations, 1e-12)\n",
                  size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
