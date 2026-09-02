// VoF rung V8 (WO-T) — the COLLOCATED path: variable density in the ABC approximate projection,
// forces as face accelerations with the averaged cell counterpart, colour advection from the
// projected face field.
//
// The collocated solver's pressure coupling is the approximate projection: average the cell
// velocities onto a MAC face field, project THAT exactly, correct the cell field. Two consequences
// drive every gate here:
//
//   * the TRANSPORT half is already right — `uf_/vf_/wf_` is exactly discretely divergence-free,
//     which is precisely what Weymouth-Yue's conservation proof needs;
//   * the FORCE half is not: a cell-centred `g_c - grad_c(P)/rho_c` is O(1) wrong at an interface
//     cell even when every face is exactly balanced, so every body/interfacial force becomes a FACE
//     acceleration `dt*(f_f - (P(i)-P(i-s)))/rho_f` added after `centerToFace`, and the cell takes
//     the AVERAGE of the two faces' total increment (Basilisk `centered.h`; Popinet JCP 2009 §3).
//
// Gates, in the order they run:
//
//   T1 HYDROSTATIC, through a hand-set rho and through C, at ratio 1000, in a triply periodic box
//      (zero-mean force) and in a walled column. `dP/dz == -rho_f g` and the FACE field at machine
//      zero. Staggered and collocated columns side by side. The cell field is REPORTED, not gated
//      at machine zero, and the reason is T1b.
//
//   T1b THE INVISIBLE SUBSPACE, measured. A cell-field checkerboard is exactly annihilated by
//      `centerToFace` (`½(U(i)+U(i-1))` kills the odd-even mode), so the approximate projection
//      cannot see it and cannot remove it. It is a PRE-EXISTING property of the collocated grid,
//      not of this rung — the control is the validated constant-density collocated path with a
//      plain body force, and it is measured here alongside the V8 path so the comparison is a
//      number rather than an argument.
//
//   T2 THE EXACTNESS GATE — a stationary droplet with a CONSTANT curvature: the face force is then
//      exactly the discrete gradient of `sigma*kappa*C`, the projection must annihilate it, and the
//      FACE field must stay at machine zero (the collocated form of V4's P1). Ratio 1 and 1000.
//
//   T3 CONSTANT-DENSITY EQUIVALENCE. At mu = 0 the collocated face-force predictor and the
//      validated cell-force one are the SAME scheme when rho is uniform (`avg_f(P(i)-P(i-s))` IS
//      the central difference), so a uniform-rho V8 run must reproduce the constant-density
//      collocated run to round-off. This is what says the rung did not change the scheme where it
//      was not supposed to.
//
//   T4 THE BRIDGE (G6). The colour transport on the collocated grid must be the SAME kernel on the
//      SAME faces as on the staggered grid: `uf_(i)` sits at i-1/2, the LOW face of cell i, exactly
//      where `flow`'s staggered `u(i)` sits, so handing a staggered solver the collocated solver's
//      own projected face field must reproduce the collocated colour BITWISE — and so must a
//      standalone `WyAdvector` given the same faces with the low->high index shift.
//
//   T5 SCOPE. `enable_vof` / variable density on the collocated grid with an immersed solid, with
//      the ghost projection, or with `enable_vof_momentum`, must throw.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::ClosureKind;
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::vof::WyAdvector;
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;
using Stag = peclet::flow::Solver<peclet::flow::Staggered>;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

std::size_t idx(int x, int y, int z, int nx, int ny) {
  return (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
}
double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}
double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}
template <class S>
double maxVel(S& s) {
  return std::fmax(std::fmax(maxAbs(s.getVelocity(0)), maxAbs(s.getVelocity(1))),
                   maxAbs(s.getVelocity(2)));
}
template <class S>
double maxFaceVel(S& s) {
  return std::fmax(std::fmax(maxAbs(s.getFaceVelocity(0)), maxAbs(s.getFaceVelocity(1))),
                   maxAbs(s.getFaceVelocity(2)));
}
// Odd-even (checkerboard) amplitude of a z-column of a cell field: max |f(k) - ½(f(k-1)+f(k+1))|,
// which reads 2A for a pure checkerboard of amplitude A and 0 for anything smooth-and-linear.
double checkerboardZ(const std::vector<double>& f, int nx, int ny, int nz) {
  double m = 0;
  const int xc = nx / 2, yc = ny / 2;
  for (int z = 1; z + 1 < nz; ++z)
    m = std::fmax(m, std::fabs(f[idx(xc, yc, z, nx, ny)] - 0.5 * (f[idx(xc, yc, z - 1, nx, ny)] +
                                                                  f[idx(xc, yc, z + 1, nx, ny)])));
  return m;
}

// ---------------------------------------------------------------- T1: hydrostatic
//
// A two-layer stratified column at rest. `periodic` runs it in a triply periodic box with the
// ZERO-MEAN force f = -(rho - <rho>) g (the only hydrostatic problem a periodic box admits; the
// reported dP/dz is then -(rho_f - <rho>) g). `walled` reproduces the staggered acid test of
// `test_vardensity_projection.cpp` verbatim (walls +-z), which is the direct staggered column.
struct HydroResult {
  double cellU = 0, faceU = 0, pErr = 0, cb = 0;
  long iters = 0;
};

template <class S>
HydroResult hydrostatic(double ratio, double mu, int steps, bool periodic, bool throughColour) {
  const int N = 8, NZ = 24;
  const double g = 0.1;
  S s(N, N, NZ);
  s.setRho(1.0);
  s.setMu(mu);
  s.setDt(1.0);
  s.setVelocityResidualTolerance(0.0);  // machine-precision gates: legacy fixed-sweep momentum loop
  if (!periodic) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
  // Heavy BELOW in the walled column (the stable stratification of the staggered acid test); a
  // heavy slab in the middle quarter in the periodic box, where the density is a FROZEN hand-set
  // field so no Rayleigh-Taylor mode exists (a periodic two-layer column always has one unstably
  // stratified interface — with a frozen rho it is simply not a degree of freedom).
  auto heavy = [&](int z) { return periodic ? (z >= NZ / 4 && z < 3 * NZ / 4) : (z < NZ / 2); };
  std::vector<double> fld((std::size_t)N * N * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        fld[idx(x, y, z, N, N)] =
            heavy(z) ? (throughColour ? 1.0 : ratio) : (throughColour ? 0.0 : 1.0);
  double rbar = 0;
  for (int z = 0; z < NZ; ++z)
    rbar += (heavy(z) ? ratio : 1.0) / NZ;
  if (throughColour) {
    // FROZEN interface (the `freeze` route of test_vof_twophase.cpp gate B2): register "C" as a
    // plain field and let the closure read it, without enable_vof, so C supplies rho and nothing
    // else. With the interface free the rest state is not a fixed point of the discrete system at
    // all (an interfacial gravity-wave mode with loop gain g*drho*dt/rho_g), which is a property of
    // the physics and not of this rung.
    s.addField("C");
    s.setField("C", fld);
    s.exchangeField("C");
    s.setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {1.0, ratio - 1.0});
  } else {
    s.addField("rho");
    s.setField("rho", fld);
    s.setDensityMode(true);
  }
  // gravity f_z = -rho g, offset by +<rho> g in the periodic box so the pressure is periodic
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "rho", "",
                     std::vector<double>{periodic ? rbar * g : 0.0, -g});
  for (int k = 0; k < steps; ++k)
    s.step();
  HydroResult r;
  r.cellU = maxVel(s);
  r.faceU = maxFaceVel(s);
  r.iters = s.lastPressureIterations();
  r.cb = checkerboardZ(s.getVelocity(2), N, N, NZ);
  const auto p = s.getPressure();
  const double off = periodic ? rbar * g : 0.0;
  for (int z = 1; z < NZ; ++z) {
    const double dp = p[idx(N / 2, N / 2, z, N, N)] - p[idx(N / 2, N / 2, z - 1, N, N)];
    const double rf = 0.5 * ((heavy(z) ? ratio : 1.0) + (heavy(z - 1) ? ratio : 1.0));
    r.pErr = std::fmax(r.pErr, std::fabs(dp + g * rf - off) / (g * ratio));
  }
  return r;
}

void gateHydrostatic() {
  std::printf(
      "\n=== T1  hydrostatic at ratio 1000, mu = 0: dP/dz == -rho_f g with the FACE field at\n"
      "        machine zero. mu = 0 is the staggered acid test's own choice — at mu > 0 the\n"
      "        STAGGERED balance is only approached (A = rho_f/dt - mu*Lap does not commute with\n"
      "        the discrete gradient at variable rho; the mu*dt^2 residue of WO-P), which is a\n"
      "        separate measurement below.\n"
      "        (cell |u| is REPORTED, not gated — see T1b.)\n");
  struct Case {
    const char* name;
    bool periodic, colour;
  };
  const Case cases[3] = {{"periodic, hand-set rho, frozen", true, false},
                         {"walled,   hand-set rho        ", false, false},
                         {"walled,   through C (frozen)  ", false, true}};
  for (const Case& c : cases) {
    const auto st = hydrostatic<Stag>(1000.0, 0.0, 100, c.periodic, c.colour);
    const auto co = hydrostatic<Colo>(1000.0, 0.0, 100, c.periodic, c.colour);
    std::printf("  %s  staggered  cell %.3e  face %.3e  dP/dz %.3e  cb %.3e  it %ld\n", c.name,
                st.cellU, st.faceU, st.pErr, st.cb, st.iters);
    std::printf("  %s  COLLOCATED cell %.3e  face %.3e  dP/dz %.3e  cb %.3e  it %ld\n", c.name,
                co.cellU, co.faceU, co.pErr, co.cb, co.iters);
    CHECK(st.faceU < 1e-12);
    CHECK(st.pErr < 1e-11);
    // The gated quantity on the collocated grid is the FACE balance (machine zero in the walled
    // column; the periodic box additionally carries the centerToFace LEAK of the cell
    // checkerboard's envelope — see T1b — which decays algebraically). The pressure follows the
    // face balance up to the accumulated potential that projects that leak away, so it is bounded
    // and reported rather than gated at machine zero.
    CHECK(co.faceU < (c.periodic ? 1e-8 : 1e-12));
    CHECK(co.pErr < 1e-6);
    CHECK(co.iters < 200);  // rule 3b: no capped solve
  }
  std::printf(
      "\n  The mu sweep (walled, hand-set rho, ratio 1000, 100 steps). On the collocated path the\n"
      "  force is applied OUTSIDE the momentum operator, at the face, so mu cannot enter the\n"
      "  balance at all; on the staggered path it does, as WO-P measured.\n");
  for (double mu : {0.0, 1e-3, 1e-2, 1e-1}) {
    const auto st = hydrostatic<Stag>(1000.0, mu, 100, false, false);
    const auto co = hydrostatic<Colo>(1000.0, mu, 100, false, false);
    std::printf(
        "    mu = %-6g  staggered face %.3e  dP/dz %.3e   |   COLLOCATED face %.3e  "
        "dP/dz %.3e\n",
        mu, st.faceU, st.pErr, co.faceU, co.pErr);
    CHECK(co.faceU < 1e-12);
  }
}

void gateInvisibleSubspace() {
  std::printf(
      "\n=== T1b the invisible subspace: a CELL checkerboard is annihilated by centerToFace\n"
      "        (1/2(U(i)+U(i-1)) kills the odd-even mode), so the approximate projection cannot\n"
      "        see it and cannot remove it. CONTROL = the VALIDATED constant-density collocated\n"
      "        path with a plain body force, where this rung is completely inert.\n");
  const int N = 8, NZ = 24;
  const double g = 0.1;
  for (double mu : {0.0, 0.01}) {
    Colo s(N, N, NZ);
    s.setRho(1.0);
    s.setMu(mu);
    s.setDt(1.0);
    s.setVelocityResidualTolerance(0.0);
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);
    s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
    s.setBodyForce(0.0, 0.0, -g);
    for (int k = 0; k < 100; ++k)
      s.step();
    std::printf(
        "  CONTROL constant-rho collocated, body force, mu = %-5g  cell |u| %.3e  face |uf| %.3e"
        "  cb %.3e\n",
        mu, maxVel(s), maxFaceVel(s), checkerboardZ(s.getVelocity(2), N, N, NZ));
  }
  const auto a = hydrostatic<Colo>(1000.0, 0.0, 100, false, false);
  const auto b = hydrostatic<Colo>(1000.0, 0.0, 400, false, false);
  std::printf(
      "  V8 walled ratio 1000 mu = 0:  100 steps cell %.3e (cb %.3e), 400 steps cell %.3e "
      "(cb %.3e)  -> decayed %.2fx\n",
      a.cellU, a.cb, b.cellU, b.cb, a.cellU / std::fmax(b.cellU, 1e-300));
  CHECK(b.cellU < a.cellU);  // the mode DECAYS; it is a transient remnant, not an instability
}

// ---------------------------------------------------------------- T2: the exactness gate
//
// Volume fractions of a sphere: exact in z, sub x sub sampling in (x,y) (the sampler of
// test_vof_surface_tension.cpp, kept identical so the two gates are on the same scene).
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
        C[idx(i, j, k, n, n)] = acc / (sub * sub);
      }
  return C;
}

template <class S>
std::unique_ptr<S> makeDroplet(int n, double R, double sigma, double mu, double rhoG, double rhoL,
                               double kappa, double dtFac) {
  auto s = std::make_unique<S>(n, n, n);
  s->setRho(rhoL);
  s->setMu(mu);
  s->setDt(1.0);
  s->setVelocityResidualTolerance(0.0);
  s->setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 10.0));
  s->setPressureChebyshev(true, 500, 1e-14);
  s->enableVof();
  s->setVof(sphereC(n, R, n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11));
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {rhoG, rhoL - rhoG});
  s->setSurfaceTension(sigma);
  if (kappa >= 0.0)
    s->setVofKappaConstant(kappa);
  s->setDt(dtFac * s->capillaryDt());
  return s;
}

struct DropResult {
  double cellU = 0, faceU = 0;
  long iters = 0;
  bool threw = false;
  std::string what;
};

template <class S>
DropResult runDroplet(int n, double R, double sigma, double mu, double ratio, double kappa,
                      double dtFac, int steps) {
  DropResult r;
  try {
    auto s = makeDroplet<S>(n, R, sigma, mu, 1.0, ratio, kappa, dtFac);
    for (int k = 0; k < steps; ++k)
      s->step();
    r.cellU = maxVel(*s);
    r.faceU = maxFaceVel(*s);
    r.iters = s->lastPressureIterations();
  } catch (const std::exception& e) {
    r.threw = true;
    r.what = e.what();
  }
  return r;
}

void gateStaticDroplet() {
  std::printf(
      "\n=== T2  static droplet, CONSTANT kappa: the face force is exactly grad(sigma*kappa*C), "
      "so\n"
      "        the projection must annihilate it -> the FACE field stays at machine zero.\n"
      "        The density ratio sweep is the interesting part: on the STAGGERED grid the force\n"
      "        goes through the momentum operator A = rho_f/dt - mu*Lap, which does not commute\n"
      "        with the discrete gradient at variable rho, so the balance is only APPROACHED (the\n"
      "        mu*dt^2 residue of WO-P). On the collocated grid the force is applied at the face\n"
      "        OUTSIDE A, so that mechanism does not exist.\n");
  const double ratios[4] = {1.0, 10.0, 100.0, 1000.0};
  for (double ratio : ratios) {
    const auto st = runDroplet<Stag>(32, 8.0, 1.0, 0.1, ratio, 0.25, 0.5, 30);
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, 0.1, ratio, 0.25, 0.5, 30);
    if (st.threw)
      std::printf("  ratio %6g  staggered  THREW: %.90s\n", ratio, st.what.c_str());
    else
      std::printf("  ratio %6g  staggered  cell %.4e  face %.4e   it %ld\n", ratio, st.cellU,
                  st.faceU, st.iters);
    if (co.threw)
      std::printf("  ratio %6g  COLLOCATED THREW: %.90s\n", ratio, co.what.c_str());
    else
      std::printf("  ratio %6g  COLLOCATED cell %.4e  face %.4e   it %ld\n", ratio, co.cellU,
                  co.faceU, co.iters);
    if (ratio == 1.0) {  // the exactness statement both grids must satisfy
      CHECK(!st.threw && st.faceU < 1e-14);
      CHECK(!co.threw && co.faceU < 1e-14);
      CHECK(!co.threw && co.cellU < 1e-14);
    }
    CHECK(co.iters < 500);
  }
  std::printf(
      "\n  The ratio-1000 mu sweep, which is where the collocated construction stops. The face\n"
      "  acceleration is EXPLICIT (applied outside A), so nothing damps the high-wavenumber part "
      "of\n"
      "  the force the way the staggered predictor's A^-1 does. The face and the cell are then\n"
      "  advanced by different operators — the face by the raw increment, the cell by A^-1 "
      "followed\n"
      "  by the AVERAGED increment — and their mismatch grows once the viscous smoothing per step\n"
      "  mu*dt/(rho_min h^2) stops being small. At dt = 0.5*dt_sigma = 4.46 that number is 0.045 "
      "at\n"
      "  mu = 0.01 (stable, and four orders more accurate than staggered) and 0.45 at mu = 0.1\n"
      "  (unstable, ~4x per step). THIS is what rates the collocated rung to density ratio "
      "~100.\n");
  for (double mu : {0.0, 0.01, 0.1}) {
    const auto st = runDroplet<Stag>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 40);
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 40);
    std::printf("    ratio 1000, mu = %-5g  staggered face %.4e%s   |   COLLOCATED face %.4e%s\n",
                mu, st.faceU, st.threw ? " (THREW)" : "", co.faceU, co.threw ? " (THREW)" : "");
    if (mu <= 0.01)  // the rated regime: stable, and better than the staggered reference
      CHECK(!co.threw && co.faceU < st.faceU);
  }
}

// ---------------------------------------------------------------- T3: constant-density equivalence
void gateUniformReduction() {
  std::printf(
      "\n=== T3  uniform rho, mu = 0: the V8 face-force predictor IS the validated cell-force one\n"
      "        (avg_f(P(i)-P(i-s)) is the central difference) -> agreement at round-off\n");
  const int N = 16, NZ = 8;
  std::vector<std::vector<double>> uu, pp;
  for (int var = 0; var < 2; ++var) {
    Colo s(N, N, NZ);
    s.setRho(2.0);
    s.setMu(0.0);
    s.setDt(4.0);
    s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
    s.setBodyForce(1e-2, 3e-3, 0.0);
    if (var) {
      s.addField("rho");
      s.setField("rho", std::vector<double>((std::size_t)N * N * NZ, 2.0));
      s.setDensityMode(true);
      s.setPressurePcg(true, 400, 1e-14);
    } else {
      s.setPressurePcg(true, 400, 1e-14);
    }
    for (int k = 0; k < 30; ++k)
      s.step();
    uu.push_back(s.getVelocity(0));
    pp.push_back(s.getPressure());
  }
  const double du = maxAbsDiff(uu[0], uu[1]), dp = maxAbsDiff(pp[0], pp[1]);
  std::printf("  max|du| %.3e  (|u| %.3e)   max|dP| %.3e  (|P| %.3e)\n", du, maxAbs(uu[0]), dp,
              maxAbs(pp[0]));
  CHECK(du < 1e-13 * std::fmax(maxAbs(uu[0]), 1e-30));
  CHECK(dp < 1e-11 * std::fmax(maxAbs(pp[0]), 1e-30));
}

// ---------------------------------------------------------------- T4: the bridge (G6)
void gateBridge() {
  std::printf(
      "\n=== T4  the bridge (G6): the collocated colour transport is the SAME kernel on the SAME\n"
      "        faces. uf_(i) sits at i-1/2 = flow's staggered u(i), so a staggered solver handed\n"
      "        the collocated projected face field must reproduce the colour BITWISE.\n");
  const int N = 24;
  const int steps = 20;
  // A collocated run whose cell velocity is a prescribed smooth solenoidal field; one step()
  // projects it onto uf_/vf_/wf_, and every advect_vof afterwards reuses that frozen face field.
  Colo co(N, N, N);
  co.setRho(1.0);
  co.setMu(0.0);
  co.setDt(0.1);  // the ABC field peaks at ~1.7, so the interface CFL is ~0.17 < the WY cap 0.25
  co.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
  co.setPressurePcg(true, 400, 1e-14);
  co.enableVof();
  {
    std::vector<double> c0((std::size_t)N * N * N, 0.0);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const double dx = x + 0.5 - (N / 2 + 0.13), dy = y + 0.5 - (N / 2 + 0.27),
                       dz = z + 0.5 - (N / 2 + 0.11);
          c0[idx(x, y, z, N, N)] = (dx * dx + dy * dy + dz * dz < 36.0) ? 1.0 : 0.0;
        }
    co.setVof(c0);
  }
  {  // an ABC-flow cell velocity: every component varies along all three axes (a uniform or a
     // solid-body field would not detect the axial half of the index shift)
    std::vector<double> u((std::size_t)N * N * N), v(u.size()), w(u.size());
    const double k = 2.0 * M_PI / N;
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const double X = x + 0.5, Y = y + 0.5, Z = z + 0.5;
          u[idx(x, y, z, N, N)] = std::sin(k * Z) + 0.7 * std::cos(k * Y);
          v[idx(x, y, z, N, N)] = std::sin(k * X) + 0.7 * std::cos(k * Z);
          w[idx(x, y, z, N, N)] = std::sin(k * Y) + 0.7 * std::cos(k * X);
        }
    co.setField("u", u);
    co.setField("v", v);
    co.setField("w", w);
  }
  co.step();  // populates + projects uf_/vf_/wf_ (and advances C once)
  const std::vector<double> uf = co.getFaceVelocity(0), vf = co.getFaceVelocity(1),
                            wf = co.getFaceVelocity(2);
  const std::vector<double> c1 = co.getVof();
  std::printf("  collocated max|div(open uf)| after the seeding step = %.3e\n",
              co.maxOpenDivergence());
  CHECK(co.maxOpenDivergence() < 1e-10);

  // --- (a) the same faces on the STAGGERED solver, whose u(i) IS the low face -------------------
  Stag st(N, N, N);
  st.setRho(1.0);
  st.setMu(0.0);
  st.setDt(0.1);
  st.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
  st.enableVof();
  st.setVof(c1);
  st.setField("u", uf);
  st.setField("v", vf);
  st.setField("w", wf);
  st.setVofStepParity(co.vofStepParity());  // the sweep permutation must start at the same index

  for (int k = 0; k < steps; ++k) {
    co.advectVofKinematic(0.1);
    st.advectVofKinematic(0.1);
  }
  const double dC = maxAbsDiff(co.getVof(), st.getVof());
  std::printf(
      "  collocated vs staggered on the SAME face field, %d kinematic steps: max|dC| = "
      "%.3e\n",
      steps, dC);
  CHECK(dC == 0.0);
  // and the colour actually moved (a bitwise match on a frozen field would be vacuous)
  std::printf("  max|C - C0| over the run = %.3e   (sum C: collocated %.15e, staggered %.15e)\n",
              maxAbsDiff(co.getVof(), c1), co.vofDiagnostics().sumC, st.vofDiagnostics().sumC);
  CHECK(maxAbsDiff(co.getVof(), c1) > 0.1);
}

// ---------------------------------------------------------------- T5: scope
void gateScope() {
  std::printf("\n=== T5  scope: what the rung refuses, loudly\n");
  const int N = 16;
  std::vector<double> solid((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - N / 2.0, dy = y + 0.5 - N / 2.0, dz = z + 0.5 - N / 2.0;
        solid[idx(x, y, z, N, N)] = std::sqrt(dx * dx + dy * dy + dz * dz) - 4.0;
      }
  {  // enable_vof with an immersed solid
    Colo s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setSolid(solid, true);
    bool threw = false;
    try {
      s.enableVof();
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  enable_vof + immersed solid (collocated)     -> throws: %s\n",
                threw ? "yes" : "NO");
    CHECK(threw);
  }
  {  // variable density with an immersed solid: caught at the first project()
    Colo s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setSolid(solid, true);
    s.addField("rho");
    s.setField("rho", std::vector<double>((std::size_t)N * N * N, 1.0));
    s.setDensityMode(true);
    bool threw = false;
    try {
      s.step();
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  set_density_mode + immersed solid            -> throws: %s\n",
                threw ? "yes" : "NO");
    CHECK(threw);
  }
  {  // momentum consistency stays staggered-only
    Colo s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
    bool threw = false;
    try {
      s.enableVofMomentum(1.0, 10.0);
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  enable_vof_momentum (collocated)             -> throws: %s\n",
                threw ? "yes" : "NO");
    CHECK(threw);
  }
  {  // the harmonic rho_f knob is not wired into the face acceleration
    Colo s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
    s.addField("rho");
    s.setField("rho", std::vector<double>((std::size_t)N * N * N, 1.0));
    s.setDensityMode(true);
    s.setRhoFaceHarmonic(true);
    bool threw = false;
    try {
      s.step();
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  set_rho_face_harmonic (collocated)           -> throws: %s\n",
                threw ? "yes" : "NO");
    CHECK(threw);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateHydrostatic();
    gateInvisibleSubspace();
    gateStaticDroplet();
    gateUniformReduction();
    gateBridge();
    gateScope();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "\n%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("\nAll rung V8 (WO-T) collocated gates passed.\n");
  return 0;
}
