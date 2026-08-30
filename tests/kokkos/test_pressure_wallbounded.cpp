// WO-H — 3-D WALL-BOUNDED PRESSURE CONVERGENCE, all three Krylov drivers.
//
// The coverage gap this closes. Every shipped domain-BC verification script (lid cavity vs Ghia,
// developing channel, backward-facing step, de Vahl Davis) is quasi-2D — the third axis is 4 cells
// — and 4 cells is inside the healthy regime. So from July 2026 until WO-B (2026-08-30) nothing
// exercised the pressure solve on a genuinely 3-D wall-bounded grid, and MG-PCG stalled there:
// 200/200 iterations with max|div(open*u)|/u ~ 1e-5 once the third axis reached 8 cells, while
// Chebyshev took 13-14 on the identical operator. The cause (WO-C -> WO-H) was that the coarse
// levels' PERIODIC ghost fill left a walled face's ghost holding the value from the opposite side
// of the domain, and `prolongAdd` reads that ghost with weight 1/4 whatever the face openness — a
// long-range coupling present in the prolongation and absent from the restriction, i.e. an
// asymmetric V-cycle preconditioner, which is exactly what CG cannot tolerate.
// `CutcellMG::applyNeumannGhost` is the repair.
//
// What this test gates, on nz >= 8 (never 4):
//   A. lid box, CONSTANT density — the WO-B headline configuration (PCG was 200/200 here);
//   B. hydrostatic column, constant density, uniform gravity;
//   C. RESIDUAL #1 of the WO-C findings: the gravity-driven hydrostatic column with a GLOBAL
//      density stratification (varRho, sharp slab). Both CG drivers used to run a *stationary*
//      iteration here — the residual rose to r/r0 = 6.98 and froze for 199 iterations — while
//      Chebyshev returned the machine-exact rest state;
//   D. RESIDUAL #2 of the WO-C findings: a small coefficient rho0/rho_f adjacent to a
//      PRESCRIBED-VELOCITY (inflow-type, BC type 2) face — heavy fluid against the lid, so the
//      operator coefficient there is 1/ratio — at ratio 1e2 and 1e3.
// Each runs with driver in {MG-PCG, flexible MG-CG, Chebyshev} and is gated on BOTH the iteration
// count (never the cap) and the physics (max|div(open*u)| relative to the case's velocity scale) —
// a low iteration count alone is not a good solve, because the CG breakdown guard exits early and
// keeps the last finite iterate.
//
// Demonstrating the failure: PECLET_FLOW_MG_BCGHOST=0 restores the pre-repair periodic-wrap coarse
// ghost (measurement ablation only). Every PCG configuration below fails under it; the test prints
// which mode it is running so an ablation run is self-labelling.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

constexpr int NX = 24, NY = 24, NZ = 16;  // the third axis is 16 cells: 4x the quasi-2D regime
constexpr int LEVELS = 4;
constexpr int MAXIT = 200;
constexpr double RTOL = 1e-8;
constexpr int STEPS = 8;
constexpr double MU = 0.01, DT = 1.0, ULID = 1.0, GRAV = 0.1;
// Health floor: the WO-B stall plateaus at div/u ~ 1e-5, four orders above the round-off floor a
// converged solve reaches (1e-11 ... 1e-23). 1e-8 is the WO-B/WO-C battery's own `healthy` cut.
constexpr double DIV_TOL = 1e-8;

enum Driver { kPcg = 0, kFcg = 1, kCheb = 2 };
const char* driverName(Driver d) {
  return d == kPcg ? "MG-PCG" : (d == kFcg ? "FCG   " : "Cheb  ");
}

double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}

void selectDriver(peclet::flow::IbmSolver& s, Driver d) {
  // Exercises the WO-H selector repair as a side effect: on the varRho cases below this call comes
  // AFTER setDensityMode (which defaults the driver to Chebyshev), and `setPressurePcg(true, ...)`
  // must genuinely switch back — before the repair its `on` flag was discarded and the solve
  // silently stayed on Chebyshev.
  if (d == kPcg)
    s.setPressurePcg(true, MAXIT, RTOL);
  else if (d == kFcg)
    s.setPressureFcg(true, MAXIT, RTOL);
  else
    s.setPressureChebyshev(true, MAXIT, RTOL);
}

// One configuration x one driver. `varRho`: 0 = constant density, else the density ratio of a sharp
// z-slab; `heavyAbove` puts the heavy fluid (hence the SMALL coefficient rho0/rho_f) against +z.
// `lid`: +z is a prescribed-velocity face (BC type 2) instead of a wall; `gravity`: gravity closure
// (varRho) or a uniform body force (constant density).
struct Case {
  const char* name;
  bool lid;
  bool gravity;
  double ratio;  // 0 => constant density
  bool heavyAbove;
  // Which drivers are GATED here (bit d). The ungated ones are still run and printed, tagged
  // "open" — they are the residual defect WO-H measured and did NOT fix: at a high density
  // CONTRAST the arithmetic coarsening of the face coefficient makes the V-cycle preconditioner
  // INDEFINITE (measured directly: sym(M) assembled densely on an 8^3 walled box has a negative
  // LDL pivot from ratio ~1e3, and one at ratio 1e4 even fully PERIODIC), and no choice of the CG
  // beta survives an indefinite preconditioner. That is the coefficient-aware-coarsening item
  // (VOF_PLAN S3), not the boundary-symmetry item this test's cases A/B pin down; Chebyshev, which
  // needs only real spectrum bounds, is healthy on all of them. See doc/vof_workorders.md WO-H.
  int gate;
};

void run(const Case& cs, Driver d) {
  peclet::flow::IbmSolver s(NX, NY, NZ);
  s.setRho(1.0);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);  // creeping Stokes: isolate the pressure solve
  s.setDomainBc(4, 1, 0, 0, 0);
  if (cs.lid)
    s.setDomainBc(5, 2, ULID, 0.0, 0.0);  // prescribed velocity (inflow-type) face
  else
    s.setDomainBc(5, 1, 0, 0, 0);
  if (cs.lid) {  // the lid box also walls +-x (periodic y), as in the WO-B battery
    s.setDomainBc(0, 1, 0, 0, 0);
    s.setDomainBc(1, 1, 0, 0, 0);
  }
  s.setPressureLevels(LEVELS);
  s.setPressureGeometry(std::vector<double>((std::size_t)NX * NY * NZ, 10.0));  // all fluid

  if (cs.ratio > 0.0) {
    std::vector<double> rho((std::size_t)NX * NY * NZ);
    for (int z = 0; z < NZ; ++z) {
      const bool heavy = cs.heavyAbove ? (z >= NZ / 2) : (z < NZ / 2);
      for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
          rho[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] =
              heavy ? cs.ratio : 1.0;
    }
    s.addField("rho");
    s.setField("rho", rho);
    s.setDensityMode(true);  // defaults the driver to Chebyshev — selectDriver must override it
    if (cs.gravity)
      s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -GRAV});
  } else if (cs.gravity) {
    s.setBodyForce(0.0, 0.0, -GRAV);
  }
  selectDriver(s, d);

  long itMax = 0, itMed = 0;
  std::vector<long> its;
  double div = 0;
  for (int k = 0; k < STEPS; ++k) {
    s.step();
    its.push_back(s.lastPressureIterations());
    div = s.maxOpenDivergence();
  }
  const double umax = std::fmax(maxAbs(s.getVelocity(0)),
                                std::fmax(maxAbs(s.getVelocity(1)), maxAbs(s.getVelocity(2))));
  double uscale = std::fmax(umax, cs.lid ? ULID : GRAV * DT);
  const double divRel = div / uscale;
  for (long v : its)
    itMax = std::max(itMax, v);
  {
    std::vector<long> sorted = its;
    std::sort(sorted.begin(), sorted.end());
    itMed = sorted[sorted.size() / 2];
  }
  const bool gated = (cs.gate >> (int)d) & 1;
  std::printf("  %-22s %s  its median %3ld max %3ld   div/u %.2e  max|u| %.3e%s\n", cs.name,
              driverName(d), itMed, itMax, divRel, umax,
              gated ? "" : "   [open: S3 coefficient coarsening, not gated]");
  fflush(stdout);
  if (!gated)
    return;
  CHECK(std::isfinite(umax));
  CHECK(itMax < MAXIT);     // never the cap: the stall ran 200/200 on every step
  CHECK(divRel < DIV_TOL);  // and the projection must actually be converged
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const char* ab = std::getenv("PECLET_FLOW_MG_BCGHOST");
    const bool ablated = ab && std::atoi(ab) == 0;
    std::printf("3-D wall-bounded pressure convergence: %dx%dx%d, %d levels, %d steps, cap %d%s\n",
                NX, NY, NZ, LEVELS, STEPS, MAXIT,
                ablated ? "   [PECLET_FLOW_MG_BCGHOST=0 — pre-WO-H ablation, EXPECTED TO FAIL]"
                        : "");
    const int ALL = (1 << kPcg) | (1 << kFcg) | (1 << kCheb);
    const int FCG_CHEB = (1 << kFcg) | (1 << kCheb);
    const int CHEB_ONLY = (1 << kCheb);
    const Case cases[] = {
        // name                  lid    grav   ratio  heavyAbove  gated drivers
        {"A lid box const-rho", true, false, 0.0, false, ALL},
        {"B hydrostatic const", false, true, 0.0, false, ALL},
        {"C hydro strat 1e2", false, true, 1e2, false, FCG_CHEB},
        {"C hydro strat 1e3", false, true, 1e3, false, CHEB_ONLY},
        {"D inflow small-c 1e2", true, false, 1e2, true, FCG_CHEB},
        {"D inflow small-c 1e3", true, false, 1e3, true, FCG_CHEB},
    };
    for (const Case& c : cases)
      for (Driver d : {kPcg, kFcg, kCheb})
        run(c, d);
    if (failures == 0)
      std::printf("PASS: %zu wall-bounded configurations x 3 drivers (14 gated) on nz = %d\n",
                  sizeof(cases) / sizeof(cases[0]), NZ);
    else
      std::fprintf(stderr, "FAIL: %d check(s)\n", failures);
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
