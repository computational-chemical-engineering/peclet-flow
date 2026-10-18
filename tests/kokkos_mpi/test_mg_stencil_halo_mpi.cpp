// flow — the pressure MG's stencil halo (doc/vof_distributed_host_comm_design.md §5.2, WO-K3).
//
// The 7-point stencil exchanges (the non-CA smoother, the residual, the matvec and their FP32
// twins) run on a faces-only halo with the level's rank-local periodic axes M_L read in-kernel
// (bit a of M_L: the level is ONE block along a, no outflow face on a, even inner extent on a).
// The claim is class a: every stencil reads the values it read before -- a face ghost holds its
// owner's value, a wrapped read reaches the cell the full exchange copies into the ghost, and the
// edge / corner ghosts the faces-only halo no longer fills are never read. So the gate is BITWISE:
// the default against the test hook debugSetStencilHaloOff(true), which forces the full `dev`
// halo and wrapMask = 0 (today's code), on the same build and rank count -- per-step pressure
// iterations identical and the solution's inner cells memcmp-equal.
//
// Cases (a walled / cut-cell / variable-coefficient operator rebuilt every "step", as
// test_bottom_direct_mpi): walls +-y with periodic x, z; an inflow/outflow z (z never a wrap axis,
// the outflow ghost re-imposed after the faces-only exchange); an odd coarse z extent (R-1: that
// level keeps z unwrapped); a forced telescope (levels below it on fewer ranks, every axis wrapped
// on a single-block level: the exchange-free path at np > 1); all-periodic (CA smoothing on the
// coarse levels stays on `dev`); the FP32 V-cycle (sdevf); the band (non-exact) matvec.
//
// Structure checks: the stencil topology never has more neighbours or ghost cells than the full
// one; at np 1 every level wraps all three axes (zero stencil exchanges).
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "peclet/core/common/types.hpp"

using peclet::core::IVec;
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::CutcellMG;

static constexpr int G = 1;
static constexpr int STEPS = 4;

static int wrap(int a, int n) {
  return ((a % n) + n) % n;
}

// A sphere's open fraction times 1 / rho of a dense blob moving along z with the step.
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
  double face(int a, double x, double y, double z) const {
    if (a == 0)
      x -= 0.5;
    if (a == 1)
      y -= 0.5;
    if (a == 2)
      z -= 0.5;
    return open(x, y, z) * coef(x, y, z);
  }
};

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
          const int gx = wrap(x - G + og.x, (int)gs[0]), gy = wrap(y - G + og.y, (int)gs[1]),
                    gz = wrap(z - G + og.z, (int)gs[2]);
          const double cx = gx + 0.5, cy = gy + 0.5, cz = gz + 0.5;
          const bool inner =
              x >= G && x < ext.x - G && y >= G && y < ext.y - G && z >= G && z < ext.z - G;
          const double f = std::sin(2.0 * M_PI * gx / gs[0]) * std::cos(M_PI * cy / gs[1]) +
                           0.5 * std::sin(2.0 * M_PI * (gz + 0.3 * sc.step) / gs[2]) +
                           0.1 * cy / gs[1];
          const bool fluid = sc.open(cx, cy, cz) > 0.0 || sc.open(cx - 0.5, cy, cz) > 0.0;
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

struct Case {
  const char* name;
  IVec<3> gs;
  int nlev;
  int bc[6];
  int teleForce;
  int prec;    // CutcellMG::kVcycleFp64 / kVcycleFp32
  bool exact;  // the exact double level-0 apply (else the bands)
  bool wrap1;  // np 1: every level wraps all three axes (no outflow face, every extent even)
};

struct Run {
  std::vector<int> iters;
  std::vector<double> x;  // inner cells, x fastest
  int levels = 0, wrapAll = 1;
  long nbFull = 0, nbStencil = 0, ghFull = 0, ghStencil = 0;  // summed over this rank's levels
  bool fewer = true;
};

static Run solveSteps(const Case& c, bool hookOff) {
  CutcellMG mg;
  mg.setBoundaryConditions(c.bc);  // before initMpi
  mg.setTelescope(c.teleForce >= 0);
  mg.setTelescopeForceLevel(c.teleForce);
  mg.initMpi(c.gs[0], c.gs[1], c.gs[2], c.nlev, MPI_COMM_WORLD);
  mg.setExactResidual(c.exact);
  mg.setVcyclePrecision(c.prec);
  mg.debugSetStencilHaloOff(hookOff);
  CutcellMG::Level& l0 = mg.level(0);
  const C3 ext = l0.ext, og = l0.og;
  const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
  CCField ox("ox", n), oy("oy", n), oz("oz", n), b("b", n), x("x", n), r("r", n), p("p", n),
      z("z", n), Ap("Ap", n);
  Run res;
  Scene sc{c.gs, 0};
  for (int s = 0; s < STEPS; ++s) {
    sc.step = s;
    fillScene(ox, oy, oz, b, ext, og, sc);
    mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
    Kokkos::deep_copy(x, 0.0);
    res.iters.push_back(mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/300, /*rtol=*/1e-10, 2, 2, 8));
  }
  auto hx = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(hx, x);
  for (int zz = G; zz < ext.z - G; ++zz)
    for (int yy = G; yy < ext.y - G; ++yy)
      for (int xx = G; xx < ext.x - G; ++xx)
        res.x.push_back(hx((long)xx + (long)yy * ext.x + (long)zz * (long)ext.x * ext.y));
  res.levels = mg.localLevels();
  for (int L = 0; L < res.levels; ++L) {
    const CutcellMG::Level& lv = mg.level(L);
    if (!lv.shalo)
      continue;
    res.wrapAll = res.wrapAll && lv.wrapAxes == 7;
    const long nf = (long)lv.halo->numNeighbors(), ns = (long)lv.shalo->numNeighbors();
    const long gf = (long)lv.halo->numGhostRecv(), gsn = (long)lv.shalo->numGhostRecv();
    res.nbFull += nf;
    res.nbStencil += ns;
    res.ghFull += gf;
    res.ghStencil += gsn;
    res.fewer = res.fewer && ns <= nf && gsn <= gf;
  }
  return res;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int fail = 0;
  {
    const int F64 = CutcellMG::kVcycleFp64, F32 = CutcellMG::kVcycleFp32;
    const Case cases[] = {
        {"walls y, periodic x z", {64, 48, 32}, 4, {0, 0, 1, 1, 0, 0}, -1, F64, true, true},
        {"walls y, band matvec", {64, 48, 32}, 4, {0, 0, 1, 1, 0, 0}, -1, F64, false, true},
        {"inflow -z / outflow +z, walls y",
         {64, 48, 32},
         4,
         {0, 0, 1, 1, 2, 3},
         -1,
         F64,
         true,
         false},
        {"odd coarse z (48x40x30), walls y",
         {48, 40, 30},
         3,
         {0, 0, 1, 1, 0, 0},
         -1,
         F64,
         true,
         false},
        {"forced telescope @1, walls y", {64, 48, 32}, 4, {0, 0, 1, 1, 0, 0}, 1, F64, true, true},
        {"all periodic (CA coarse levels)",
         {64, 48, 32},
         4,
         {0, 0, 0, 0, 0, 0},
         -1,
         F64,
         true,
         true},
        {"FP32 V-cycle, walls y", {64, 48, 32}, 4, {0, 0, 1, 1, 0, 0}, -1, F32, true, true},
    };
    for (const Case& c : cases) {
      const Run a = solveSteps(c, /*hookOff=*/false);
      const Run h = solveSteps(c, /*hookOff=*/true);
      const bool sameIt = a.iters == h.iters;
      const bool sameX = a.x.size() == h.x.size() &&
                         std::memcmp(a.x.data(), h.x.data(), a.x.size() * sizeof(double)) == 0;
      int bad = (sameIt && sameX && a.fewer) ? 0 : 1;
      if (size == 1 && c.wrap1 && !a.wrapAll)
        bad = 1;  // np 1: every level wraps all three axes -> zero stencil exchanges
      int anyBad = 0;
      MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      long cnt[4] = {a.nbFull, a.nbStencil, a.ghFull, a.ghStencil}, tot[4];
      MPI_Allreduce(cnt, tot, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (anyBad)
        ++fail;
      if (rank == 0) {
        std::printf("  %-36s np=%d iters:", c.name, size);
        for (std::size_t s = 0; s < a.iters.size(); ++s)
          std::printf(" %d/%d", a.iters[s], h.iters[s]);
        std::printf("  x %s  nbrs full/stencil %ld/%ld ghosts %ld/%ld  %s\n",
                    sameX ? "bitwise" : "DIFFERS", tot[0], tot[1], tot[2], tot[3],
                    anyBad ? "FAIL" : "ok");
      }
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): stencil halo == full halo, bitwise\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d case(s)\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
