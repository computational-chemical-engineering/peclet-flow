// VoF rung V4 (WO-P) — balanced-force continuum surface force + the capillary time step.
//
// The one rule (Francois et al., JCP 213:141 (2006); Popinet, JCP 228:5838 (2009)): `sigma*kappa*
// grad C` must be evaluated at the same location and with the same discrete gradient operator as
// the pressure gradient. On this staggered grid that is the face difference `C(i) - C(i - s_c)`,
// the one `buildRhsVar` already applies to P. See `src/vof/surface_tension.hpp`.
//
// Gates, in the order they run:
//
//   P1 THE EXACTNESS GATE — a stationary droplet with a CONSTANT curvature. The force is then
//      exactly the discrete gradient of `sigma*kappa*C`, so the projection must annihilate it
//      completely and `max|u|` must stay at MACHINE ZERO, independent of viscosity and of
//      resolution. This is the momentum analogue of the hydrostatic acid test and it is the gate
//      that fails loudly on any force/pressure operator mismatch. Run at three resolutions and four
//      viscosities spanning three decades.
//
//   P2 THE ABLATION THAT MAKES P1 MEAN SOMETHING — the identical configuration with the identical
//      curvature, forced by a CELL-CENTRED sigma*kappa*grad(C) face-interpolated the way a rho*g
//      body force is (`set_csf_mode(1)`). Same physics, same accuracy order, one wrong operator
//      pairing: the currents must jump by many orders. Without this, P1 could be passing for the
//      wrong reason.
//
//   P3 YOUNG-LAPLACE — the discrete equilibrium pressure jump across the interface must be
//      sigma*kappa, and with the CONSTANT-kappa force the equilibrium field must be exactly
//      `P = sigma*kappa*C + const` (which is the algebraic statement P1 measures dynamically).
//
//   P4 THE CAPILLARY TIME STEP — `capillary_dt()` against the closed form
//      sqrt((rho1+rho2) h^3/(4 pi sigma)) (Brackbill 1992; Denner & van Wachem 2015), for the
//      declared phase pair and for the min+max of a closure-driven density field; and `step()`
//      must REFUSE a dt above it.
//
//   P5 INERT WHEN OFF — with `set_surface_tension` never called, u, P and C are BITWISE identical
//      to the V2a/V2b path. (The rung adds a kernel to the RHS loop and a curvature pass to the
//      head of the step; this is the statement that both are gated.)
//
//   P6 THE WISP GUARD, and the measurement that put it there. Weymouth-Yue leaves round-off colour
//      residue in every cell its sweeps touch; those cells satisfy `0 < C < 1`, so the curvature
//      cascade builds a zero-area PLIC polygon for them and returns |kappa| up to 1e8. Gated both
//      ways: with the guard (default eps = 1e-8) a real droplet's spurious currents DECAY, with
//      `set_vof_interface_eps(0)` they grow by three orders in 20 steps.
//
//   P7 SPURIOUS CURRENTS WITH THE REAL CURVATURE — the capillary number a resolved static droplet
//      actually settles at, and its convergence with resolution. A measurement, reported and
//      loosely bounded, not a pass/fail on a number the estimator cannot deliver.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
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

std::size_t idx(int x, int y, int z, int n) {
  return (std::size_t)x + (std::size_t)y * n + (std::size_t)z * (std::size_t)n * n;
}

// Volume fractions of a sphere: exact in z, sub x sub sampling in (x,y). Accurate to ~1e-6 in C,
// which matters — a crudely sampled sphere quantizes C and manufactures full/empty cell PAIRS whose
// shared face carries an interface no cell knows about (measured: 468 such faces at sub = 8 in all
// three directions, 0 with this construction).
std::vector<double> sphereC(int n, double R, double cx, double cy, double cz, int sub = 24) {
  std::vector<double> C((std::size_t)n * n * n, 0.0);
  const double w = 1.0 / sub;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
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
        C[idx(i, j, k, n)] = acc / (sub * sub);
      }
  return C;
}

double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}
double maxVel(IbmSolver& s) {
  return std::fmax(std::fmax(maxAbs(s.getVelocity(0)), maxAbs(s.getVelocity(1))),
                   maxAbs(s.getVelocity(2)));
}
double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  // WO-R2: NaN-PROPAGATING. `std::fmax(m, NaN) == m`, so the obvious loop returns 0.000e+00 for a
  // field that has gone entirely NaN and every bitwise gate built on it passes (WO-R found this on
  // a drained open-boundary run). A non-finite difference must fail, so return it.
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;  // NaN
    m = std::fmax(m, d);
  }
  return m;
}

struct Cfg {
  int n = 32;
  double R = 8.0;
  double sigma = 1.0;
  double mu = 0.1;
  double rhoG = 1.0;
  double rhoL = 1.0;
  double dtFac = 0.5;   // fraction of the capillary dt
  double kappa = -1.0;  // >= 0: freeze kappa at this constant (the exactness gate)
  int mode = 0;         // set_csf_mode
  double eps = -1.0;    // >= 0: set_vof_interface_eps
};

// A stationary droplet in a triply periodic box, off-centre by an irrational fraction of a cell so
// nothing is mesh-aligned.
std::unique_ptr<IbmSolver> makeDroplet(const Cfg& c) {
  auto s = std::make_unique<IbmSolver>(c.n, c.n, c.n);
  s->setRho(c.rhoL);
  s->setMu(c.mu);
  s->setDt(1.0);
  s->setPressureGeometry(std::vector<double>((std::size_t)c.n * c.n * c.n, 10.0));
  s->setPressureChebyshev(true, 500, 1e-14);
  s->enableVof();
  s->setVof(sphereC(c.n, c.R, c.n / 2 + 0.13, c.n / 2 + 0.27, c.n / 2 + 0.11));
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {c.rhoG, c.rhoL - c.rhoG});
  s->setSurfaceTension(c.sigma);
  if (c.eps >= 0.0)
    s->setVofInterfaceEps(c.eps);
  s->setCsfMode(c.mode);
  if (c.kappa >= 0.0)
    s->setVofKappaConstant(c.kappa);
  s->setDt(c.dtFac * s->capillaryDt());
  return s;
}

double runDroplet(const Cfg& c, int steps, IbmSolver** keep = nullptr) {
  auto s = makeDroplet(c);
  for (int k = 0; k < steps; ++k)
    s->step();
  const double m = maxVel(*s);
  if (keep)
    *keep = s.release();
  return m;
}

// ------------------------------------------------------------------ P1 + P2
void gateExactBalance() {
  std::printf("\n=== P1  THE EXACTNESS GATE: constant kappa -> max|u| at machine zero\n");
  const int ns[3] = {16, 32, 48};
  const double Rs[3] = {4.0, 8.0, 12.0};
  for (int i = 0; i < 3; ++i) {
    Cfg c;
    c.n = ns[i];
    c.R = Rs[i];
    c.kappa = 2.0 / Rs[i];
    const double m = runDroplet(c, 30);
    std::printf("  n = %2d  R = %4.1f  mu = 0.1        max|u| = %.4e   (Ca = %.2e)\n", ns[i], Rs[i],
                m, c.mu * m / c.sigma);
    CHECK(m < 1e-14);
  }
  const double mus[4] = {1e-3, 1e-2, 1e-1, 1.0};
  for (double mu : mus) {
    Cfg c;
    c.mu = mu;
    c.kappa = 0.25;
    const double m = runDroplet(c, 30);
    std::printf("  n = 32  R =  8.0  mu = %-8.4g  max|u| = %.4e\n", mu, m);
    CHECK(m < 1e-14);
  }
}

void gateAblation() {
  std::printf(
      "\n=== P2  the ablation: the SAME constant kappa, forced by a cell-centred sigma*kappa*grad "
      "C\n"
      "        face-interpolated like a body force (set_csf_mode(1))\n");
  Cfg bal;
  bal.kappa = 0.25;
  Cfg abl = bal;
  abl.mode = 1;
  const double mb = runDroplet(bal, 30), ma = runDroplet(abl, 30);
  std::printf("  balanced-force (face difference)   max|u| = %.4e   Ca = %.2e\n", mb,
              bal.mu * mb / bal.sigma);
  std::printf("  cell-centred force, interpolated   max|u| = %.4e   Ca = %.2e   -> %.1e x worse\n",
              ma, abl.mu * ma / abl.sigma, ma / std::fmax(mb, 1e-300));
  CHECK(mb < 1e-14);
  CHECK(ma > 1e6 * std::fmax(mb, 1e-300));  // the mismatch must be loud, not marginal
  CHECK(ma > 1e-4);
}

// ------------------------------------------------------------------ P3
void gateYoungLaplace() {
  std::printf("\n=== P3  Young-Laplace: the equilibrium pressure field is sigma*kappa*C + const\n");
  Cfg c;
  c.n = 32;
  c.R = 8.0;
  c.kappa = 0.25;
  IbmSolver* s = nullptr;
  runDroplet(c, 40, &s);
  std::unique_ptr<IbmSolver> own(s);
  const auto P = s->getPressure();
  const auto C = s->getVof();
  // Fit the offset from the gas (C == 0) cells, then measure the residual of P - sigma*kappa*C.
  double p0 = 0.0;
  long ng = 0;
  for (std::size_t i = 0; i < C.size(); ++i)
    if (C[i] == 0.0) {
      p0 += P[i];
      ++ng;
    }
  p0 /= (double)ng;
  double res = 0.0, jump = 0.0;
  long nl = 0;
  double pl = 0.0;
  for (std::size_t i = 0; i < C.size(); ++i) {
    res = std::fmax(res, std::fabs(P[i] - p0 - c.sigma * c.kappa * C[i]));
    if (C[i] == 1.0) {
      pl += P[i];
      ++nl;
    }
  }
  jump = pl / (double)nl - p0;
  const double exact = c.sigma * c.kappa;
  std::printf("  dP(liquid - gas) = %.10f   sigma*kappa = %.10f   rel err %.3e\n", jump, exact,
              std::fabs(jump - exact) / exact);
  std::printf("  max |P - p0 - sigma*kappa*C| over the WHOLE field = %.3e\n", res);
  CHECK(std::fabs(jump - exact) / exact < 1e-12);
  CHECK(res < 1e-11);
}

// ------------------------------------------------------------------ P4
void gateCapillaryDt() {
  std::printf("\n=== P4  the capillary time step\n");
  const double sigma = 2.0, rg = 1.0, rl = 700.0;
  const int n = 24;
  auto s = std::make_unique<IbmSolver>(n, n, n);
  s->setRho(rl);
  s->setMu(0.01);
  s->setDt(1e-3);
  s->setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 10.0));
  s->setPressureChebyshev(true, 300, 1e-13);
  s->enableVof();
  s->setVof(sphereC(n, 6.0, n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11));
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {rg, rl - rg});
  s->setSurfaceTension(sigma);
  const double want = std::sqrt((rg + rl) * 1.0 / (4.0 * M_PI * sigma));
  const double got = s->capillaryDt();  // from min+max of the closure-driven rho field
  std::printf("  closure-driven rho field:  dt_sigma = %.12g   closed form = %.12g   rel %.2e\n",
              got, want, std::fabs(got - want) / want);
  CHECK(std::fabs(got - want) / want < 1e-14);
  // The declared phase pair (momentum consistency) must give the same number.
  s->enableVofMomentum(rg, rl);
  const double got2 = s->capillaryDt();
  std::printf("  declared phase pair:       dt_sigma = %.12g\n", got2);
  CHECK(std::fabs(got2 - want) / want < 1e-14);
  // ... and step() must refuse a dt above it.
  s->setDt(1.01 * want);
  bool threw = false;
  try {
    s->step();
  } catch (const std::exception& e) {
    threw = true;
    std::printf("  step() at 1.01 dt_sigma REFUSED, as it must\n");
  }
  CHECK(threw);
  s->setCapillaryCfl(1e30);
  s->setDt(0.5 * want);
  s->step();  // and the escape hatch works
  std::printf("  set_capillary_cfl(1e30) disables the check\n");
}

// ------------------------------------------------------------------ P5
void gateInert() {
  std::printf("\n=== P5  inert when off: bitwise identical to the rung-V2a path\n");
  const int n = 24;
  auto build = [&](bool st) {
    auto s = std::make_unique<IbmSolver>(n, n, n);
    s->setRho(10.0);
    s->setMu(0.05);
    s->setDt(0.2);
    s->setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 10.0));
    s->setPressureChebyshev(true, 300, 1e-13);
    s->enableVof();
    s->setVof(sphereC(n, 6.0, n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11));
    s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {1.0, 9.0});
    if (st) {
      s->setSurfaceTension(0.0);  // explicitly OFF: must not arm anything
      s->setVofInterfaceEps(1e-8);
      s->setCsfMode(1);
    }
    const std::size_t nc = (std::size_t)n * n * n;
    std::vector<double> u(nc), v(nc), w(nc);
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
          u[idx(i, j, k, n)] = 0.02 * std::sin(2 * M_PI * (j + 0.5) / n);
          v[idx(i, j, k, n)] = 0.0;
          w[idx(i, j, k, n)] = 0.0;
        }
    s->uploadVelocity(u, v, w);
    return s;
  };
  auto a = build(false), b = build(true);
  for (int k = 0; k < 12; ++k) {
    a->step();
    b->step();
  }
  const double du = std::fmax(maxAbsDiff(a->getVelocity(0), b->getVelocity(0)),
                              maxAbsDiff(a->getVelocity(2), b->getVelocity(2)));
  const double dp = maxAbsDiff(a->getPressure(), b->getPressure());
  const double dc = maxAbsDiff(a->getVof(), b->getVof());
  std::printf("  du = %.3e   dp = %.3e   dC = %.3e\n", du, dp, dc);
  CHECK(du == 0.0 && dp == 0.0 && dc == 0.0);
}

// ------------------------------------------------------------------ P6
void gateWispGuard() {
  std::printf(
      "\n=== P6  the wisp guard: kappa must not be computed where there is no interface\n"
      "        The GATE is on the mechanism (max|kappa| over the cells the cascade served), which\n"
      "        is deterministic and separated by five orders. The velocity CONSEQUENCE is "
      "reported\n"
      "        rather than gated: whether a given run is merely degraded or destroyed depends on\n"
      "        whether a garbage-kappa cell happens to sit next to a real one, so it is loud but\n"
      "        configuration-dependent (measured at 64^3, 60 steps: 1.2e-2 unguarded and STILL\n"
      "        GROWING against 1.4e-4 decaying — see the study script).\n");
  const int ns[2] = {32, 48};
  const double Rs[2] = {8.0, 12.0};
  for (int i = 0; i < 2; ++i)
    for (double eps : {1e-8, 0.0}) {
      Cfg c;
      c.n = ns[i];
      c.R = Rs[i];
      c.eps = eps;
      auto s = makeDroplet(c);
      double u0 = 0, uMax = 0, kmax = 0;
      for (int k = 0; k < 40; ++k) {
        s->step();
        const double u = maxVel(*s);
        if (k == 0)
          u0 = u;
        uMax = std::fmax(uMax, u);
      }
      const double uEnd = maxVel(*s);
      const auto ka = s->getVofCurvature();
      const auto br = s->getVofCurvatureBranch();
      long interf = 0;
      for (std::size_t j = 0; j < ka.size(); ++j)
        if (br[j] > 0.5 && br[j] < 5.5) {
          ++interf;
          kmax = std::fmax(kmax, std::fabs(ka[j]));
        }
      std::printf(
          "  n = %2d  eps = %-6g  max|kappa| = %.4e  (2/R = %.4f)  cells served %4ld   "
          "max|u| %.3e -> %.3e (peak %.3e)\n",
          ns[i], eps, kmax, 2.0 / c.R, interf, u0, uEnd, uMax);
      if (eps > 0.0) {
        CHECK(kmax < 1.5 * 2.0 / c.R);  // no curvature beyond the sphere's own survives
        CHECK(uEnd < u0);               // and the spurious currents decay
      } else {
        CHECK(kmax > 100.0 * 2.0 / c.R);  // the mechanism, five orders wide
      }
    }
}

// ------------------------------------------------------------------ P7
void gateSpuriousCurrents() {
  std::printf(
      "\n=== P7  spurious currents with the REAL curvature (a measurement, not a pass/fail)\n"
      "        Ca = mu max|u| / sigma after 60 steps; La = sigma rho D / mu^2\n");
  const int ns[3] = {16, 32, 48};
  const double Rs[3] = {4.0, 8.0, 12.0};
  double prev = 0.0;
  for (int i = 0; i < 3; ++i) {
    Cfg c;
    c.n = ns[i];
    c.R = Rs[i];
    IbmSolver* s = nullptr;
    const double m = runDroplet(c, 60, &s);
    std::unique_ptr<IbmSolver> own(s);
    const auto d = s->csfDiagnostics();
    const double ca = c.mu * m / c.sigma;
    std::printf("  n = %2d  D/dx = %4.1f  La = %6.0f   Ca = %.3e%s   orphan faces %ld/%ld/%ld\n",
                ns[i], 2 * Rs[i], c.sigma * c.rhoL * 2 * Rs[i] / (c.mu * c.mu), ca,
                i ? "" : "            ", d.orphanFaces[0], d.orphanFaces[1], d.orphanFaces[2]);
    if (i)
      std::printf("             ratio to the coarser rung: %.2f x\n", prev / ca);
    prev = ca;
    CHECK(ca < 1e-3);
    CHECK(d.orphanFaces[0] + d.orphanFaces[1] + d.orphanFaces[2] < 20);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateExactBalance();
    gateAblation();
    gateYoungLaplace();
    gateCapillaryDt();
    gateInert();
    gateWispGuard();
    gateSpuriousCurrents();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "\nFAILED (%d)\n", failures);
    return 1;
  }
  std::printf("\nPASSED (0 failures)\n");
  return 0;
}
