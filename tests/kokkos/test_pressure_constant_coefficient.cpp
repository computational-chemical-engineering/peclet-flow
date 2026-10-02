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
//
//   G-E2-RED (WO-E2.3, §12.5 P1) — at uniform density (rho closure [rho, 0], rho == set_rho) a
//      split step IS an exact MG-PCG step bit for bit: u, v, w, p, C and the per-step PCG
//      iterations after 20 steps, 32^3 periodic and an anisotropic 32x32x16 box of extent (1,1,1).
//
//   G-E2-REC (WO-E2.3, §12.5 P3) — in a horizontally uniform, inviscid, walled two-layer column
//      the split scheme's error obeys g_{n+1} = (1 - a_f)((1 + theta_n) g_n - theta_n g_{n-1})
//      EXACTLY per face (g = G P - F_f, a_f = rho0/rho_f), and the velocity stays zero. Ratios 3
//      and 50, 30 steps, dt x0.8 at step 10 (theta 0.8) and x1.25 at step 20 (theta capped at 1).
//      Fixes the sign and coefficient of S3, Delta P, theta and rho0 with no free parameter.
//
//   G-E2-BAL (WO-E2.3, §12.5 P2, revised §12.13.4) — the constant-kappa stationary droplet at
//      density contrast (inside/outside 50 and 0.02): mu = 0, 30 steps, max|u| < 1e-14 and
//      P = sigma*kappa*C + const to 1e-10 relative; mu = 0.1, split and exact in lockstep for 1000
//      steps (>= n_DF = 14 rho_max/rho0 = 700): no growth (max|u_split(n)| <= 1.5 x its max over
//      the first 10 steps), rate (D(1000) <= 1e-2 D(100), D(n) = max|u_split(n) - u_exact(n)|)
//      and fixed point (max|u_split(1000)| <= 2 max|u_exact(1000)| + 1e-14).
//
//   G-E2-RST (WO-E2.3, at mu = 0.1 per §12.13.6) — a ratio-50 rising bubble (32^3, walls in z):
//      40 continuous steps against 20 + the §12.7 restore (fresh solver, p_increment restored,
//      startup_steps = 0) + 20, bitwise; restoring WITHOUT p_increment must differ.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <Kokkos_Core.hpp>
#include <limits>
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
  CHECK(raisesOnStep("rotational weight > 2", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setRotationalWeight(2.5);
  }));
  CHECK(raisesOnStep("variable-rotational chi > 2", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setPropertyMode(true, false);
    s.setVariableRotational(1, 2.5);
  }));
  CHECK(raisesOnStep("balanced-force projection", [](IbmSolver& s) {
    baseScope(s);
    finishScope(s);
    s.setBalancedForceProjection(true);
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

// ---------------------------------------------------------------- G-E2-RED
bool redCase(const char* name, int nx, int ny, int nz, bool aniso, double rho) {
  auto build = [&]() {
    auto s = std::make_unique<IbmSolver>(nx, ny, nz);
    if (aniso)
      s->setPhysicalDomain({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {nx, ny, nz});
    s->setRho(rho);
    s->setMu(aniso ? 0.1 / (32.0 * 32.0) : 0.1);  // a comparable cell diffusion number
    s->setDt(1.0);
    s->setVelocityResidualTolerance(0.0);
    s->setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 10.0));
    s->enableVof();
    s->setVof(sphereC(nx, ny, nz, 0.25 * nz, nx / 2 + 0.13, ny / 2 + 0.27, nz / 2 + 0.11));
    s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {rho, 0.0});
    s->setSurfaceTension(aniso ? 1.0 / 32.0 : 1.0);
    s->setPressurePcg(true, 500, 1e-12);
    s->setDt(0.5 * s->capillaryDt());
    return s;
  };
  auto ex = build();
  auto sp = build();
  sp->setPressureConstantCoefficient(true, 0);
  int itBad = 0;
  long itSum = 0;
  bool rho0Ok = true;
  for (int k = 0; k < 20; ++k) {
    ex->step();
    sp->step();
    itSum += sp->lastPressureIterations();
    if (ex->lastPressureIterations() != sp->lastPressureIterations())
      ++itBad;
    if (k == 0) {
      const auto st = sp->pressureConstantCoefficientStats();
      rho0Ok = st.lastStepSplit && std::memcmp(&st.rho0, &rho, sizeof(double)) == 0;
      std::printf("  [%s] step 1: last_split %d rho0 %.17g (set_rho %.17g) %s\n", name,
                  st.lastStepSplit, st.rho0, rho, rho0Ok ? "bitwise" : "DIFFERS");
    }
  }
  const State a = snapshot(*ex), b = snapshot(*sp);
  const char* nm[5] = {"u", "v", "w", "p", "C"};
  bool all = true;
  std::printf("  [%s] 20 steps:", name);
  for (int q = 0; q < 5; ++q) {
    const bool same = bitwise(a.f[q], b.f[q]);
    all = all && same;
    std::printf(" %s %s", nm[q], same ? "bitwise" : "DIFFERS");
  }
  std::printf(" | iteration mismatches %d (mean %.2f) | max|u| %.3e\n", itBad, itSum / 20.0,
              maxAbs(b.f[0]));
  return all && itBad == 0 && rho0Ok;
}

void gateReduction() {
  std::printf("\n=== G-E2-RED (P1): uniform density -> split step == exact MG-PCG step, bitwise\n");
  CHECK(redCase("32^3 periodic, rho 2.5", 32, 32, 32, false, 2.5));
  CHECK(redCase("32x32x16 extent (1,1,1)", 32, 32, 16, true, 1.0));
}

// ---------------------------------------------------------------- G-E2-REC
constexpr int RX = 8, RY = 8, RZ = 32;
constexpr double RGRAV = 0.1;

bool recCase(double ratio) {
  IbmSolver s(RX, RY, RZ);
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(1.0);
  s.setAdvection(false);
  s.setVelocityResidualTolerance(0.0);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)RX * RY * RZ, 10.0));
  s.setDensityMode(true);
  std::vector<double> rho((std::size_t)RX * RY * RZ);
  for (int z = 0; z < RZ; ++z)
    for (int y = 0; y < RY; ++y)
      for (int x = 0; x < RX; ++x)
        rho[idx3(x, y, z, RX, RY)] = z < RZ / 2 ? ratio : 1.0;  // heavy below
  s.setField("rho", rho);
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "rho", "", {0.0, -RGRAV});
  s.setPressurePcg(true, 500, 1e-13);
  s.setPressureConstantCoefficient(true, 0);
  const double rho0 = 1.0;
  // the faces z = 1 .. RZ-1 between cells z-1 and z (column-uniform; read column (0,0))
  auto gOf = [&](const std::vector<double>& P) {
    std::vector<double> g(RZ, 0.0);
    for (int z = 1; z < RZ; ++z) {
      const double ra = rho[idx3(0, 0, z, RX, RY)], rb = rho[idx3(0, 0, z - 1, RX, RY)];
      const double Ff = 0.5 * ((-RGRAV * ra) + (-RGRAV * rb));
      g[z] = (P[idx3(0, 0, z, RX, RY)] - P[idx3(0, 0, z - 1, RX, RY)]) - Ff;
    }
    return g;
  };
  std::vector<double> g0 = gOf(s.getField("p"));  // P = 0: g_0 = -F_f
  double g0max = 0.0;
  for (double v : g0)
    g0max = std::fmax(g0max, std::fabs(v));
  std::vector<double> gPrev = g0, gCur = g0, gPred(RZ);  // g_{-1} = g_0 (no history)
  double dt = 1.0, dtPrev = 0.0, worst = 0.0, umax = 0.0, thetaErr = 0.0;
  for (int n = 0; n < 30; ++n) {
    if (n == 10)
      dt *= 0.8;
    if (n == 20)
      dt *= 1.25;
    s.setDt(dt);
    const double th = (dtPrev > 0 && dt < dtPrev) ? dt / dtPrev : 1.0;
    s.step();
    const auto st = s.pressureConstantCoefficientStats();
    thetaErr = std::fmax(thetaErr, std::fabs(st.theta - th));
    for (int z = 1; z < RZ; ++z) {
      const double ra = rho[idx3(0, 0, z, RX, RY)], rb = rho[idx3(0, 0, z - 1, RX, RY)];
      const double af = rho0 / (0.5 * (ra + rb));
      gPred[z] = (1.0 - af) * ((1.0 + th) * gCur[z] - th * gPrev[z]);
    }
    const std::vector<double> gm = gOf(s.getField("p"));
    double e = 0.0;
    for (int z = 1; z < RZ; ++z)
      e = std::fmax(e, std::fabs(gm[z] - gPred[z]));
    worst = std::fmax(worst, e / g0max);
    for (int c = 0; c < 3; ++c)
      umax = std::fmax(umax, maxAbs(s.getVelocity(c)));
    gPrev = gCur;
    gCur = gPred;  // the recurrence runs on its own prediction (no re-seeding from measurement)
    dtPrev = dt;
  }
  double gHeavy = 0.0;
  for (int z = 1; z < RZ / 2; ++z)
    gHeavy = std::fmax(gHeavy, std::fabs(gCur[z]));
  const bool ok = worst <= 1e-9 && umax <= 1e-13 && thetaErr == 0.0;
  std::printf(
      "  ratio %4.0f: max_n max_f |g_meas - g_pred|/max|g_0| = %.3e  max|u| = %.3e  theta err "
      "%.1e  (heavy-phase |g_30|/|g_0| = %.3e)  %s\n",
      ratio, worst, umax, thetaErr, gHeavy / g0max, ok ? "OK" : "FAIL");
  return ok;
}

void gateRecurrence() {
  std::printf("\n=== G-E2-REC (P3, 1-D): the discrete error recurrence holds exactly per face\n");
  CHECK(recCase(3.0));
  CHECK(recCase(50.0));
}

// ---------------------------------------------------------------- G-E2-BAL
std::unique_ptr<IbmSolver> makeBal(double rhoIn, double rhoOut, double mu, bool split) {
  DropCfg c;
  c.rhoIn = rhoIn;
  c.rhoOut = rhoOut;
  c.mu = mu;
  c.kappa = 0.25;
  auto s = makeDrop(c);
  s->setPressurePcg(true, 500, 1e-13);
  if (split)
    s->setPressureConstantCoefficient(true, 2);
  return s;
}
double maxVel(IbmSolver& s) {
  double m = 0.0;
  for (int q = 0; q < 3; ++q)
    m = std::fmax(m, maxAbs(s.getVelocity(q)));
  return m;
}
double maxVelDiff(IbmSolver& a, IbmSolver& b) {
  double m = 0.0;
  for (int q = 0; q < 3; ++q) {
    const auto ua = a.getVelocity(q), ub = b.getVelocity(q);
    for (std::size_t i = 0; i < ua.size(); ++i)
      m = std::fmax(m, std::fabs(ua[i] - ub[i]));
  }
  return m;
}
// mu = 0 item (§12.10, unchanged): 30 split steps, max|u| and |P - sigma kappa C - c| / sigma
// kappa.
struct BalOut {
  double umax, pres;
};
BalOut balInviscid(double rhoIn, double rhoOut) {
  auto s = makeBal(rhoIn, rhoOut, 0.0, true);
  for (int k = 0; k < 30; ++k)
    s->step();
  BalOut o{maxVel(*s), 0.0};
  const double sigma = DropCfg{}.sigma, kappa = 0.25;
  const auto P = s->getField("p");
  const auto C = s->getVof();
  double p0 = 0.0;
  long ng = 0;
  for (std::size_t i = 0; i < C.size(); ++i)
    if (C[i] == 0.0) {
      p0 += P[i];
      ++ng;
    }
  p0 /= (double)ng;
  for (std::size_t i = 0; i < C.size(); ++i)
    o.pres = std::fmax(o.pres, std::fabs(P[i] - p0 - sigma * kappa * C[i]));
  o.pres /= sigma * kappa;
  return o;
}
// mu = 0.1 item (§12.13.4): split and exact in lockstep; a raise (a blow-up trips the WY CFL cap)
// is a FAIL, not an abort.
constexpr int BAL_STEPS = 1000;
bool balViscous(double rhoIn, double rhoOut) {
  auto sp = makeBal(rhoIn, rhoOut, 0.1, true);
  auto ex = makeBal(rhoIn, rhoOut, 0.1, false);
  double early = 0.0, worst = 0.0, d100 = 0.0, d1000 = 0.0, us = 0.0, ue = 0.0;
  int worstAt = 0;
  for (int n = 1; n <= BAL_STEPS; ++n) {
    try {
      sp->step();
      ex->step();
    } catch (const std::runtime_error& e) {
      std::printf("    in/out %5.2f mu 0.1: step %d raised: %.70s\n", rhoIn / rhoOut, n, e.what());
      return false;
    }
    us = maxVel(*sp);
    if (n <= 10)
      early = std::fmax(early, us);
    if (us > worst) {
      worst = us;
      worstAt = n;
    }
    if (n == 100)
      d100 = maxVelDiff(*sp, *ex);
    if (n % 100 == 0)
      std::printf("    in/out %5.2f mu 0.1 step %4d: split max|u| %.3e  D %.3e\n", rhoIn / rhoOut,
                  n, us, n == 100 ? d100 : maxVelDiff(*sp, *ex));
  }
  d1000 = maxVelDiff(*sp, *ex);
  ue = maxVel(*ex);
  const bool noGrowth = worst <= 1.5 * early;
  const bool rate = d1000 <= 1e-2 * d100;
  const bool fixedPoint = us <= 2.0 * ue + 1e-14;
  std::printf(
      "  in/out %5.2f  mu 0.1 %d steps: no growth (max %.3e at %d <= 1.5 x %.3e) %s; rate "
      "D(1000)/D(100) = %.3e/%.3e = %.2e %s; fixed point split %.3e exact %.3e %s\n",
      rhoIn / rhoOut, BAL_STEPS, worst, worstAt, early, noGrowth ? "OK" : "FAIL", d1000, d100,
      d100 > 0 ? d1000 / d100 : 0.0, rate ? "OK" : "FAIL", us, ue, fixedPoint ? "OK" : "FAIL");
  return noGrowth && rate && fixedPoint;
}

void gateBalance() {
  std::printf("\n=== G-E2-BAL (P2, §12.13.4): constant-kappa droplet at a density contrast\n");
  const double pairs[2][2] = {{50.0, 1.0}, {1.0, 50.0}};
  for (const auto& pr : pairs) {
    const BalOut a = balInviscid(pr[0], pr[1]);
    const bool okA = a.umax < 1e-14 && a.pres <= 1e-10;
    std::printf("  in/out %5.2f  mu 0    30 steps: split max|u| %.3e  |P - sk C - c|/sk %.3e  %s\n",
                pr[0] / pr[1], a.umax, a.pres, okA ? "OK" : "FAIL");
    CHECK(okA);
    CHECK(balViscous(pr[0], pr[1]));
  }
}

// ---------------------------------------------------------------- G-E2-RST
constexpr int BN = 32;
std::unique_ptr<IbmSolver> makeBubble(bool split, int startup) {
  auto s = std::make_unique<IbmSolver>(BN, BN, BN);
  s->setRho(50.0);
  s->setMu(0.1);  // §12.13.6: the viscous restart is the meaningful one
  s->setDt(1.0);
  s->setVelocityResidualTolerance(0.0);
  s->setDomainBc(4, 1, 0, 0, 0);
  s->setDomainBc(5, 1, 0, 0, 0);
  s->setPressureGeometry(std::vector<double>((std::size_t)BN * BN * BN, 10.0));
  s->enableVof();
  s->setVof(sphereC(BN, BN, BN, 6.0, BN / 2 + 0.13, BN / 2 + 0.27, 10.11));
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {50.0, 1.0 - 50.0});  // light inside
  s->setPropertyModel("force_z", ClosureKind::LinearMix, "rho", "", {0.0, -1e-3});
  s->setSurfaceTension(0.5);
  s->setPressurePcg(true, 500, 1e-10);
  s->setDt(0.25 * s->capillaryDt());
  if (split)
    s->setPressureConstantCoefficient(true, startup);
  return s;
}

struct RstOut {
  bool same;
  double dmax;
};
RstOut restartRun(bool split, bool restoreIncrement) {
  auto cont = makeBubble(split, 2);
  for (int k = 0; k < 40; ++k)
    cont->step();
  auto first = makeBubble(split, 2);
  for (int k = 0; k < 20; ++k)
    first->step();
  // save (the §12.7 recipe), then a fresh solver
  std::vector<double> u = first->getVelocity(0), v = first->getVelocity(1),
                      w = first->getVelocity(2), p = first->getField("p"), C = first->getVof();
  std::vector<double> dp;
  if (split)
    dp = first->getField("p_increment");
  first.reset();
  auto rest = makeBubble(false, 0);
  rest->uploadVelocity(u, v, w);
  rest->setField("p", p);
  rest->setVof(C);
  rest->setVofStepParity(20);
  if (split) {
    rest->setPressureConstantCoefficient(true, 0);  // registers p_increment (zeros)
    if (restoreIncrement)
      rest->setField("p_increment", dp);
  }
  for (int k = 0; k < 20; ++k)
    rest->step();
  const State a = snapshot(*cont), b = snapshot(*rest);
  RstOut o{sameState(a, b), 0.0};
  for (int q = 0; q < 5; ++q)
    for (std::size_t i = 0; i < a.f[q].size(); ++i)
      o.dmax = std::fmax(o.dmax, std::fabs(a.f[q][i] - b.f[q][i]));
  return o;
}

void gateRestart() {
  std::printf("\n=== G-E2-RST: restart with p_increment and startup_steps = 0 is bitwise\n");
  const RstOut ex = restartRun(false, false);
  const RstOut sp = restartRun(true, true);
  const RstOut neg = restartRun(true, false);
  std::printf("  exact path restart:            %s (max|diff| %.3e)\n",
              ex.same ? "bitwise" : "differs", ex.dmax);
  std::printf("  split, p_increment restored:   %s (max|diff| %.3e)\n",
              sp.same ? "bitwise" : "differs", sp.dmax);
  std::printf("  split, p_increment NOT restored (negative control): %s (max|diff| %.3e)\n",
              neg.same ? "bitwise" : "differs", neg.dmax);
  // §12.10: if the exact path's own restart is not bitwise, the gate is "deviation <= exact's".
  CHECK(ex.same ? sp.same : sp.dmax <= ex.dmax);
  CHECK(!neg.same);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateScope();
    gateBracket();
    gateReduction();
    gateRecurrence();
    gateBalance();
    gateRestart();
  }
  Kokkos::finalize();
  std::printf("\ntest_pressure_constant_coefficient: %s (%d failure%s)\n",
              failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
