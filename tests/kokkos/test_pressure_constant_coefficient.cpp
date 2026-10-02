// E2(a) — the opt-in constant-coefficient (Dodd–Ferrante) pressure driver,
// `set_pressure_constant_coefficient` (doc/vof_step_performance_design.md §12).
//
// Gates, in the order they run:
//
//   SCOPE (WO-E2.2) — one raising check per precheck condition of §12.6 (each perturbs ONE thing
//      of an otherwise valid configuration, and the valid configuration itself steps), plus the
//      setter's own refusals: SolverColocated and startup_steps < 0.
//
//   BRACKET (WO-E2.2) — with the driver ON and startup_steps = 10**6 every step is an exact
//      start-up step. On a PCG-selected ratio-50 VoF drop (32^3) u, v, w, p, C must be BITWISE the
//      driver-off run over 20 steps (the bracket only copies), and "p_increment" must equal the
//      stored P^{n+1} - P^n bitwise at every step.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

std::size_t idx3(int x, int y, int z, int nx, int ny) {
  return (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
}

// Volume fractions of a sphere (test_vof_surface_tension's construction): exact in z, sub x sub
// sampling in (x, y).
std::vector<double> sphereC(int nx, int ny, int nz, double R, double cx, double cy, double cz,
                            int sub = 24) {
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  const double w = 1.0 / sub;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        double acc = 0.0;
        for (int b = 0; b < sub; ++b)
          for (int a = 0; a < sub; ++a) {
            const double px = i + (a + 0.5) * w, py = j + (b + 0.5) * w;
            const double r2 = R * R - (px - cx) * (px - cx) - (py - cy) * (py - cy);
            if (r2 <= 0.0)
              continue;
            const double h = std::sqrt(r2);
            const double lo = std::fmax(cz - h, (double)k), hi = std::fmin(cz + h, (double)k + 1);
            if (hi > lo)
              acc += hi - lo;
          }
        C[idx3(i, j, k, nx, ny)] = acc / (sub * sub);
      }
  return C;
}

bool bitwise(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}

struct State {
  std::vector<double> f[5];  // u, v, w, p, C
};
State snapshot(IbmSolver& s) {
  State st;
  for (int c = 0; c < 3; ++c)
    st.f[c] = s.getVelocity(c);
  st.f[3] = s.getField("p");
  st.f[4] = s.getVof();
  return st;
}
bool sameState(const State& a, const State& b) {
  for (int k = 0; k < 5; ++k)
    if (!bitwise(a.f[k], b.f[k]))
      return false;
  return true;
}

// ---------------------------------------------------------------- the ratio-50 VoF drop (32^3)
struct DropCfg {
  int nx = 32, ny = 32, nz = 32;
  double R = 8.0;
  double rhoIn = 50.0, rhoOut = 1.0;  // inside / outside
  double mu = 0.1, sigma = 1.0;
  double dtFac = 0.5;   // fraction of the capillary dt
  double kappa = -1.0;  // >= 0: constant curvature (the exactness configuration)
};

std::unique_ptr<IbmSolver> makeDrop(const DropCfg& c) {
  auto s = std::make_unique<IbmSolver>(c.nx, c.ny, c.nz);
  s->setRho(c.rhoOut);
  s->setMu(c.mu);
  s->setDt(1.0);
  s->setVelocityResidualTolerance(0.0);  // the fixed-sweep momentum loop: a bitwise gate
  s->setPressureGeometry(std::vector<double>((std::size_t)c.nx * c.ny * c.nz, 10.0));
  s->enableVof();
  s->setVof(sphereC(c.nx, c.ny, c.nz, c.R, c.nx / 2 + 0.13, c.ny / 2 + 0.27, c.nz / 2 + 0.11));
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {c.rhoOut, c.rhoIn - c.rhoOut});
  s->setSurfaceTension(c.sigma);
  if (c.kappa >= 0.0)
    s->setVofKappaConstant(c.kappa);
  s->setPressurePcg(true, 500, 1e-12);  // after the rho closure (which installs Chebyshev)
  s->setDt(c.dtFac * s->capillaryDt());
  return s;
}

// ---------------------------------------------------------------- SCOPE
// A small valid driver configuration: variable density (two layers), all-fluid, periodic.
constexpr int SN = 8;
void baseScope(IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(0.1);
  s.setDt(1.0);
}
void finishScope(IbmSolver& s, bool geometry = true, bool density = true) {
  if (geometry)
    s.setPressureGeometry(std::vector<double>((std::size_t)SN * SN * SN, 10.0));
  if (density) {
    s.setDensityMode(true);
    std::vector<double> r((std::size_t)SN * SN * SN);
    for (int z = 0; z < SN; ++z)
      for (int y = 0; y < SN; ++y)
        for (int x = 0; x < SN; ++x)
          r[idx3(x, y, z, SN, SN)] = z < SN / 2 ? 10.0 : 1.0;
    s.setField("rho", r);
  }
  s.setPressureConstantCoefficient(true, 2);
}

bool raisesOnStep(const char* what, const std::function<void(IbmSolver&)>& build) {
  IbmSolver s(SN, SN, SN);
  build(s);
  try {
    s.step();
  } catch (const std::runtime_error& e) {
    const bool named = std::strstr(e.what(), "set_pressure_constant_coefficient") != nullptr;
    std::printf("  %-34s raises%s\n", what, named ? "" : "  (BUT NOT the named message)");
    if (!named)
      std::printf("      what(): %s\n", e.what());
    return named;
  }
  std::printf("  %-34s DID NOT RAISE\n", what);
  return false;
}

void gateScope() {
  std::printf("\n=== SCOPE (WO-E2.2): every precheck condition raises, by name\n");
  {
    IbmSolver s(SN, SN, SN);
    baseScope(s);
    finishScope(s);
    bool ok = true;
    try {
      for (int k = 0; k < 3; ++k)
        s.step();
    } catch (const std::exception& e) {
      std::printf("  valid configuration raised: %s\n", e.what());
      ok = false;
    }
    std::printf("  %-34s %s\n", "valid configuration", ok ? "steps" : "FAILS");
    CHECK(ok);
  }
  CHECK(raisesOnStep("immersed solid", [](IbmSolver& s) {
    baseScope(s);
    std::vector<double> sdf((std::size_t)SN * SN * SN);
    for (int z = 0; z < SN; ++z)
      for (int y = 0; y < SN; ++y)
        for (int x = 0; x < SN; ++x) {
          const double dx = x + 0.5 - SN / 2.0, dy = y + 0.5 - SN / 2.0, dz = z + 0.5 - SN / 2.0;
          sdf[idx3(x, y, z, SN, SN)] = std::sqrt(dx * dx + dy * dy + dz * dz) - 2.0;
        }
    s.setSolid(sdf, true);
    finishScope(s, false);
  }));
  CHECK(raisesOnStep("no pressure geometry", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s, false);
  }));
  CHECK(raisesOnStep("porous continuity", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setPorousContinuity(true);
  }));
  CHECK(raisesOnStep("drag field", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.enableDrag();
  }));
  CHECK(raisesOnStep("constant density", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s, true, false);
  }));
  CHECK(raisesOnStep("inflow face", [](IbmSolver& s) {
    baseScope(s);
    s.setDomainBc(0, 2, 0.01, 0, 0);
    s.setDomainBc(1, 3, 0, 0, 0);
    finishScope(s);
  }));
  CHECK(raisesOnStep("outflow face", [](IbmSolver& s) {
    baseScope(s);
    s.setDomainBc(0, 1, 0, 0, 0);
    s.setDomainBc(1, 3, 0, 0, 0);
    finishScope(s);
  }));
  CHECK(raisesOnStep("Picard iterations", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setOuterIterations(2);
  }));
  CHECK(raisesOnStep("pressure under-relaxation", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setPressureUnderRelax(0.5);
  }));
  CHECK(raisesOnStep("harmonic face density", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setRhoFaceHarmonic(true);
  }));
  CHECK(raisesOnStep("divergence source", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setDivergenceSource(std::vector<double>((std::size_t)SN * SN * SN, 0.0));
  }));
  CHECK(raisesOnStep("non-incremental pressure", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setIncrementalPressure(false);
  }));
  {
    bool raised = false;
    try {
      peclet::flow::Solver<peclet::flow::Colocated> sc(SN, SN, SN);
      sc.setPressureConstantCoefficient(true, 2);
    } catch (const std::runtime_error&) {
      raised = true;
    }
    std::printf("  %-34s %s\n", "SolverColocated (setter)", raised ? "raises" : "DID NOT RAISE");
    CHECK(raised);
  }
  {
    bool raised = false;
    try {
      IbmSolver s(SN, SN, SN);
      s.setPressureConstantCoefficient(true, -1);
    } catch (const std::invalid_argument&) {
      raised = true;
    }
    std::printf("  %-34s %s\n", "startup_steps < 0 (setter)", raised ? "raises" : "DID NOT RAISE");
    CHECK(raised);
  }
}

// ---------------------------------------------------------------- BRACKET
void gateBracket() {
  std::printf(
      "\n=== BRACKET (WO-E2.2): startup_steps = 10**6 -> bitwise the driver-off run; p_increment "
      "= P^{n+1} - P^n\n");
  DropCfg c;
  auto off = makeDrop(c);
  auto on = makeDrop(c);
  on->setPressureConstantCoefficient(true, 1000000);
  int bad = 0, badInc = 0;
  for (int k = 0; k < 20; ++k) {
    const std::vector<double> p0 = on->getField("p");
    off->step();
    on->step();
    if (!sameState(snapshot(*off), snapshot(*on)))
      ++bad;
    const std::vector<double> p1 = on->getField("p"), dp = on->getField("p_increment");
    std::vector<double> want(p1.size());
    for (std::size_t i = 0; i < p1.size(); ++i)
      want[i] = p1[i] - p0[i];
    if (!bitwise(dp, want))
      ++badInc;
    if (on->lastPressureIterations() != off->lastPressureIterations())
      ++bad;
  }
  const auto st = on->pressureConstantCoefficientStats();
  std::printf(
      "  20 steps: state/iteration mismatches %d, p_increment mismatches %d; stats: enabled %d "
      "last_split %d startup_left %d rho0 %s\n",
      bad, badInc, st.enabled, st.lastStepSplit, st.startupStepsLeft,
      std::isnan(st.rho0) ? "NaN" : "set");
  std::printf("  max|u| %.3e  max|p_increment| %.3e  iters(last) %ld\n", maxAbs(on->getVelocity(0)),
              maxAbs(on->getField("p_increment")), on->lastPressureIterations());
  CHECK(bad == 0);
  CHECK(badInc == 0);
  CHECK(st.enabled && !st.lastStepSplit && st.startupStepsLeft == 1000000 - 20);
  CHECK(std::isnan(st.rho0));
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateScope();
    gateBracket();
  }
  Kokkos::finalize();
  std::printf("\ntest_pressure_constant_coefficient: %s (%d failure%s)\n",
              failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
