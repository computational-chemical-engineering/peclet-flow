// ctest `pressure_chebyshev_bounds`: the life cycle of the Chebyshev pressure driver's spectral
// bounds (doc/mg_depth_policy_log.md 2026-10-30, the A3 review fixes).
//
// Case: the hydrostatic column of `pressure_wallbounded` (case B: 24 x 24 x 16, z walls, constant
// density, uniform gravity), Chebyshev driver. Constant coefficients, so the bounds are estimated
// once and kept across steps.
//
//   1. A V-cycle setter that CHANGES the V-cycle (set_pressure_vcycle_symmetric,
//      set_pressure_line_relaxation) invalidates the kept bounds: the next step re-estimates them
//      (one more estimate), and still converges. Setting the value already in force is inert (no
//      estimate). Before the fix the kept bounds of the old V-cycle survived the toggle.
//   2. The cold-bounds growth guard, symmetric and trilinear V-cycle: the test hook scales the
//      first estimate's bounds by 0.4 (lambda_max far below the top, so the solve diverges); the
//      guard trips once, the bounds are re-estimated with the bracketing margin, the step
//      converges, and later steps keep the new bounds without tripping. Without the hook the
//      guard never trips (case 1 also checks 0 trips).
//   3. The bounds ratchet (DECISIONS_GPU.log 2026-10-30 "A3 finish" (b)): after a trip, the
//      bounds are dropped as a coefficient rebuild drops them and the warm re-estimate is scaled
//      by 0.4 again. The ratchet keeps lambda_max at the trip's bracketed value, so the warm
//      solve neither trips its 3-iteration guard nor restarts cold. Without the ratchet the
//      warm bounds are the scaled ones and the step restarts (negative control: 1 restart).
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
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

constexpr int NX = 24, NY = 24, NZ = 16;
constexpr int LEVELS = 4;
constexpr int MAXIT = 200;
constexpr double RTOL = 1e-8;
constexpr double MU = 0.01, DT = 1.0, GRAV = 0.1;
constexpr double DIV_TOL = 1e-8;  // relative to GRAV * DT, as pressure_wallbounded

void setUp(peclet::flow::IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);
  s.setPressureLevels(LEVELS);
  s.setPressureGeometry(std::vector<double>((std::size_t)NX * NY * NZ, 10.0));  // all fluid
  s.setBodyForce(0.0, 0.0, -GRAV);
  s.setPressureChebyshev(true, MAXIT, RTOL);
}

// One step; the iteration count must stay below the cap and the projection converged.
void stepChecked(peclet::flow::IbmSolver& s, const char* what) {
  s.step();
  const long it = s.lastPressureIterations();
  const double div = s.maxOpenDivergence() / (GRAV * DT);
  std::printf("    %-34s its %3ld  div/u %.2e  estimates %ld\n", what, it, div,
              s.numPressureEigenEstimates());
  CHECK(it < MAXIT);
  CHECK(div < DIV_TOL);
}

// 1. A V-cycle change invalidates the kept bounds; the same value does not.
void toggleCase() {
  std::printf("  1. V-cycle setters invalidate the kept bounds\n");
  peclet::flow::IbmSolver s(NX, NY, NZ);
  setUp(s);
  stepChecked(s, "step 1");
  const long n1 = s.numPressureEigenEstimates();
  stepChecked(s, "step 2 (kept bounds)");
  stepChecked(s, "step 3 (kept bounds)");
  CHECK(s.numPressureEigenEstimates() == n1);  // constant coefficients: estimated once
  struct Toggle {
    const char* what;
    bool lineRelax;  // which setter
    bool value;
    long expect;  // estimates added by the next step
  };
  const Toggle toggles[] = {{"symmetric on (unchanged)", false, true, 0},
                            {"symmetric off", false, false, 1},
                            {"symmetric off (unchanged)", false, false, 0},
                            {"symmetric on", false, true, 1},
                            {"line relaxation on (unchanged)", true, true, 0},
                            {"line relaxation off", true, false, 1},
                            {"line relaxation on", true, true, 1}};
  for (const Toggle& t : toggles) {
    const long n0 = s.numPressureEigenEstimates();
    if (t.lineRelax)
      s.setPressureLineRelaxation(t.value);
    else
      s.setPressureVcycleSymmetric(t.value);
    stepChecked(s, t.what);
    CHECK(s.numPressureEigenEstimates() - n0 == t.expect);
  }
  CHECK(s.numPressureChebyshevGrowthTrips() == 0);
}

// 2. The growth guard on deliberately too-small bounds.
void guardCase(bool symmetric) {
  std::printf("  2. growth guard, %s V-cycle, first estimate x 0.4\n",
              symmetric ? "symmetric" : "trilinear");
  peclet::flow::IbmSolver s(NX, NY, NZ);
  setUp(s);
  s.setPressureVcycleSymmetric(symmetric);
  s.debugSetPressureEigenScale(0.4);
  stepChecked(s, "step 1 (scaled bounds)");
  CHECK(s.numPressureChebyshevGrowthTrips() == 1);
  CHECK(s.numPressureEigenEstimates() == 2);  // the scaled one + the bracketed re-estimate
  stepChecked(s, "step 2");
  stepChecked(s, "step 3");
  std::printf("    growth trips %ld\n", s.numPressureChebyshevGrowthTrips());
  CHECK(s.numPressureChebyshevGrowthTrips() == 1);
  CHECK(s.numPressureEigenEstimates() == 2);
}

// 3. The ratchet: a trip's bracketed lambda_max survives the warm re-estimate.
void ratchetCase() {
  std::printf("  3. bounds ratchet: trip, drop the bounds, warm re-estimate x 0.4\n");
  peclet::flow::IbmSolver s(NX, NY, NZ);
  setUp(s);
  s.debugSetPressureEigenScale(0.4);
  stepChecked(s, "step 1 (scaled bounds, trips)");
  CHECK(s.numPressureChebyshevGrowthTrips() == 1);
  s.debugDropPressureBounds();
  s.debugSetPressureEigenScale(0.4);
  const long n0 = s.numPressureEigenEstimates();
  stepChecked(s, "step 2 (warm x 0.4, ratcheted)");
  std::printf("    estimates +%ld  restarts %ld  growth trips %ld\n",
              s.numPressureEigenEstimates() - n0, s.numPressureChebyshevRestarts(),
              s.numPressureChebyshevGrowthTrips());
  CHECK(s.numPressureEigenEstimates() - n0 == 1);  // the warm one only
  CHECK(s.numPressureChebyshevRestarts() == 0);
  CHECK(s.numPressureChebyshevGrowthTrips() == 1);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Chebyshev pressure bounds: %dx%dx%d, %d levels, cap %d, rtol %.0e\n", NX, NY, NZ,
                LEVELS, MAXIT, RTOL);
    toggleCase();
    guardCase(true);
    guardCase(false);
    ratchetCase();
    if (failures == 0)
      std::printf("PASS: pressure_chebyshev_bounds\n");
    else
      std::fprintf(stderr, "FAIL: %d check(s)\n", failures);
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
