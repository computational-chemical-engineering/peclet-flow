// VoF rung V8 — the COLLOCATED path: variable density and surface tension in the ABC approximate
// projection through the mass-adjoint pair (doc/collocated_varrho_forces.md), colour advection from
// the projected face field.
//
// The collocated solver's pressure coupling is the approximate projection: map the cell velocities
// onto a MAC face field, project THAT exactly, correct the cell field. On this path
//
//   * the TRANSPORT half: `uf_/vf_/wf_` is exactly discretely divergence-free, which is precisely
//     what Weymouth-Yue's conservation proof needs;
//   * the FORCE half: the pressure, the CSF and `set_body_force` (a mean pressure gradient) enter
//     the IMPLICIT momentum predictor as the finite-volume face integral
//     rho_c * avg_faces(o (f - (P(i)-P(i-s)))/rho_f); a per-cell force enters at the cell value
//     times the reconstruction's weight sum; the constraint reads the MOMENTUM-weighted face
//     velocity (rho_L u_L + rho_R u_R)/(rho_L + rho_R). Then M Gamma = -C^T: stable at every dt and
//     ratio, dt-independent steady state (tests/python/test_collocated_stability_guard.py).
//     The retired WO-T form -- every force a face acceleration added AFTER the viscous solve
//     (Basilisk centered.h) -- was balanced per step but non-incremental and unstable above
//     mu dt/(rho h^2) = 1/12; do not reintroduce it.
//
// Balance at rest. With the balanced-force projection (doc §4.6; the DEFAULT on this path) a static
// balance -- hydrostatic column, constant-kappa drop -- is exact from the first step. With it
// explicitly OFF the balance is still an exact FIXED POINT, but the pressure absorbs a gradient
// force only through the step, and the transient from the initial P decays. Both are gated: the
// T1/T1b/T2 blocks below run explicitly OFF; "ON" re-runs them at the default.
//
// Gates, in the order they run:
//
//   T1 HYDROSTATIC, through a hand-set rho and through C, at ratio 1000, in a triply periodic box
//      (the uniform offset <rho> g as set_body_force) and in a walled column. Staggered and
//      collocated side by side. Collocated: the FACE field at round-off; the cell field and
//      dP/dz are the decaying transient -- reported, with decay required (400 steps < 100 steps).
//
//   T1b THE INVISIBLE SUBSPACE, measured. A cell-field checkerboard is annihilated by the uniform
//      centre-to-face average (`½(U(i)+U(i-1))` kills the odd-even mode), so the approximate
//      projection cannot see it and cannot remove it -- a PRE-EXISTING property of the collocated
//      grid, measured on the constant-density control beside the V8 column. The transient of the
//      column lives largely in that mode, which is why it decays slowly.
//
//   T2 STATIONARY DROPLET with a CONSTANT curvature: the CSF is exactly a face gradient, so the
//      balance is a fixed point; ratio 1..1000 and the ratio-1000 mu sweep must not throw, and the
//      face field must decay (90 steps < 30 steps).
//
//   T3 CONSTANT-DENSITY EQUIVALENCE. A uniform-rho V8 run reproduces the constant-density
//      collocated run to round-off: the face integral of a uniform rho IS the central difference.
//
//   T4 THE BRIDGE (G6). The colour transport on the collocated grid is the SAME kernel on the SAME
//      faces as on the staggered grid: `uf_(i)` sits at i-1/2, exactly where the staggered `u(i)`
//      sits, so a staggered solver handed the collocated projected face field reproduces the
//      collocated colour BITWISE.
//
//   T5 SCOPE. Variable density / VoF on the collocated grid with an immersed solid, with
//      `enable_vof_momentum`, with `set_rho_face_harmonic` or with the non-incremental pressure,
//      must throw.
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
// `bfp`: -1 = the solver's default (balanced-force projection ON on the collocated V8 path, OFF on
// the staggered grid), 0 = explicitly OFF, 1 = explicitly ON.
HydroResult hydrostatic(double ratio, double mu, int steps, bool periodic, bool throughColour,
                        int bfp = -1, double chebRtol = -1.0) {
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
  // gravity f_z = -rho g as a per-cell field (volumetric: the cell value on the collocated grid);
  // in the periodic box the offset +<rho> g that makes the pressure periodic is a UNIFORM drive,
  // i.e. a mean pressure gradient, so it is set_body_force (a surface force on the collocated
  // variable-density path, doc/collocated_varrho_forces.md U1 / §5) and not part of the field.
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "rho", "", std::vector<double>{0.0, -g});
  if (periodic)
    s.setBodyForce(0.0, 0.0, rbar * g);
  if (bfp >= 0)
    s.setBalancedForceProjection(bfp == 1);
  if (chebRtol > 0.0)  // the driver last (set_density_mode re-selects Chebyshev)
    s.setPressureChebyshev(true, 4000, chebRtol);
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
      "\n=== T1  hydrostatic at ratio 1000, mu = 0: dP/dz == -rho_f g. Staggered: exact. "
      "Collocated:\n"
      "        the FACE field at round-off; the balance is an exact fixed point, so the cell "
      "field\n"
      "        and dP/dz are the transient from P = 0 -- reported, and required to DECAY.\n");
  struct Case {
    const char* name;
    bool periodic, colour;
  };
  const Case cases[3] = {{"periodic, hand-set rho, frozen", true, false},
                         {"walled,   hand-set rho        ", false, false},
                         {"walled,   through C (frozen)  ", false, true}};
  for (const Case& c : cases) {
    const auto st = hydrostatic<Stag>(1000.0, 0.0, 100, c.periodic, c.colour);
    const auto co = hydrostatic<Colo>(1000.0, 0.0, 100, c.periodic, c.colour, 0);
    const auto co4 = hydrostatic<Colo>(1000.0, 0.0, 400, c.periodic, c.colour, 0);
    std::printf("  %s  staggered  cell %.3e  face %.3e  dP/dz %.3e  cb %.3e  it %ld\n", c.name,
                st.cellU, st.faceU, st.pErr, st.cb, st.iters);
    std::printf(
        "  %s  COLLOCATED cell %.3e  face %.3e  dP/dz %.3e  cb %.3e  it %ld   (400 steps: "
        "cell %.3e  dP/dz %.3e)\n",
        c.name, co.cellU, co.faceU, co.pErr, co.cb, co.iters, co4.cellU, co4.pErr);
    CHECK(st.faceU < 1e-12);
    CHECK(st.pErr < 1e-11);
    // Collocated: the FACE field is the projected field of a velocity that is itself the decaying
    // transient, so it sits at the pressure solve's round-off of that transient. Walled: the
    // note's 1e-12 holds (3.2e-13 measured, double operator storage). Periodic: FROZEN at WO-V1
    // (2026-09-25) at 10x the measured 1.79e-10 (doc §9 G4 measure-then-freeze; the note's 1e-10
    // missed by 1.8x; the periodic box carries a 2.3e-1 cell checkerboard transient at mu = 0).
    CHECK(co.faceU < (c.periodic ? 1.8e-9 : 1e-12));
    CHECK(co4.cellU < co.cellU);  // the transient decays ...
    CHECK(co4.pErr < co.pErr);    // ... in the velocity and in the pressure gradient
    CHECK(co.iters < 200);        // rule 3b: no capped solve
  }
  std::printf(
      "\n  The mu sweep (walled, hand-set rho, ratio 1000, 100 steps). The staggered balance is\n"
      "  only approached at mu > 0 (A = rho_f/dt - mu*Lap does not commute with the discrete\n"
      "  gradient at variable rho: WO-P's mu*dt^2 residue).\n");
  for (double mu : {0.0, 1e-3, 1e-2, 1e-1}) {
    const auto st = hydrostatic<Stag>(1000.0, mu, 100, false, false);
    const auto co = hydrostatic<Colo>(1000.0, mu, 100, false, false, 0);
    std::printf(
        "    mu = %-6g  staggered face %.3e  dP/dz %.3e   |   COLLOCATED face %.3e  "
        "dP/dz %.3e\n",
        mu, st.faceU, st.pErr, co.faceU, co.pErr);
    CHECK(co.faceU < 1e-12);  // measured 4.2e-14 .. 3.2e-13 (double operator storage)
  }
}

void gateInvisibleSubspace() {
  std::printf(
      "\n=== T1b the invisible subspace: a CELL checkerboard is annihilated by centerToFace\n"
      "        (1/2(U(i)+U(i-1)) kills the odd-even mode), so the approximate projection cannot\n"
      "        see it and cannot remove it. CONTROL = the VALIDATED constant-density collocated\n"
      "        path with a plain body force, where rung V8 is completely inert. The V8 column's\n"
      "        transient lives largely in that mode, and must decay.\n");
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
  const auto a = hydrostatic<Colo>(1000.0, 0.0, 100, false, false, 0);
  const auto b = hydrostatic<Colo>(1000.0, 0.0, 400, false, false, 0);
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
                               double kappa, double dtFac, int bfp = -1) {
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
  if (bfp >= 0)
    s->setBalancedForceProjection(bfp == 1);
  return s;
}

struct DropResult {
  double cellU = 0, faceU = 0, face30 = 0;
  long iters = 0;
  bool threw = false;
  std::string what;
};

template <class S>
DropResult runDroplet(int n, double R, double sigma, double mu, double ratio, double kappa,
                      double dtFac, int steps, int bfp = -1) {
  DropResult r;
  try {
    auto s = makeDroplet<S>(n, R, sigma, mu, 1.0, ratio, kappa, dtFac, bfp);
    for (int k = 0; k < steps; ++k) {
      s->step();
      if (k + 1 == 30)
        r.face30 = maxFaceVel(*s);
    }
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
      "        the balance is an exact fixed point. On the STAGGERED grid the force goes through\n"
      "        A = rho_f/dt - mu*Lap, which does not commute with the discrete gradient at "
      "variable\n"
      "        rho or near walls, so the balance is only approached (WO-P's mu*dt^2 residue). The\n"
      "        collocated step (balanced-force projection OFF) no longer annihilates "
      "constant-kappa\n"
      "        CSF per step: a decaying transient residue remains (doc §4.6.5 model: 2.2e-4 at\n"
      "        ratio 1 after 30 steps, 1.6e-5 at ratio 1000, decaying to 2.4e-17 / 2.1e-7 in 300\n"
      "        steps; 1-2 decades above staggered at ratio 1000). Exact from step 1 needs the\n"
      "        balanced-force projection. Gated here: no throw, decay 30 -> 90 steps, and the\n"
      "        30-step face frozen at 10x its WO-V1 value.\n");
  // FROZEN at WO-V1 (2026-09-25, host-openmp): the collocated face |uf| after 30 steps, measured
  // 9.99e-5 / 2.40e-4 / 1.46e-4 / 6.22e-5 for ratio 1 / 10 / 100 / 1000 (mu = 0.1); the gate is
  // 10x that (doc §9 G4-OFF).
  const double ratios[4] = {1.0, 10.0, 100.0, 1000.0};
  const double frozen30[4] = {9.99e-4, 2.40e-3, 1.46e-3, 6.22e-4};
  for (int q = 0; q < 4; ++q) {
    const double ratio = ratios[q];
    const auto st = runDroplet<Stag>(32, 8.0, 1.0, 0.1, ratio, 0.25, 0.5, 30);
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, 0.1, ratio, 0.25, 0.5, 90, 0);
    if (st.threw)
      std::printf("  ratio %6g  staggered  THREW: %.90s\n", ratio, st.what.c_str());
    else
      std::printf("  ratio %6g  staggered  cell %.4e  face %.4e   it %ld\n", ratio, st.cellU,
                  st.faceU, st.iters);
    if (co.threw)
      std::printf("  ratio %6g  COLLOCATED THREW: %.90s\n", ratio, co.what.c_str());
    else
      std::printf(
          "  ratio %6g  COLLOCATED face %.4e (30 steps) -> %.4e (90 steps), cell %.4e   "
          "it %ld\n",
          ratio, co.face30, co.faceU, co.cellU, co.iters);
    if (ratio == 1.0)  // the exactness statement of the staggered grid
      CHECK(!st.threw && st.faceU < 1e-14);
    CHECK(!co.threw);
    CHECK(co.faceU < co.face30);
    CHECK(co.face30 <= frozen30[q]);
    CHECK(co.iters < 500);
  }
  std::printf(
      "\n  The ratio-1000 mu sweep (the WO-T face-acceleration form went unstable at mu = 0.1,\n"
      "  ~4x per step; the predictor form is stable at every mu dt). Decay 30 -> 90 steps.\n");
  // FROZEN at WO-V1: 30-step collocated face for mu = 0 / 0.01 / 0.1, measured 6.74e-5 / 6.50e-5 /
  // 6.22e-5; the gate is 10x that.
  const double mus[3] = {0.0, 0.01, 0.1};
  const double frozenMu[3] = {6.74e-4, 6.50e-4, 6.22e-4};
  for (int q = 0; q < 3; ++q) {
    const double mu = mus[q];
    const auto st = runDroplet<Stag>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 30);
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 90, 0);
    std::printf(
        "    ratio 1000, mu = %-5g  staggered face %.4e%s   |   COLLOCATED face %.4e -> "
        "%.4e%s\n",
        mu, st.faceU, st.threw ? " (THREW)" : "", co.face30, co.faceU, co.threw ? " (THREW)" : "");
    CHECK(!co.threw);
    CHECK(co.faceU < co.face30);
    CHECK(co.face30 <= frozenMu[q]);
  }
}

// ---------------------------------------------------------------- T1/T1b/T2 with the option ON
//
// The balanced-force projection (doc/collocated_varrho_forces.md §4.6) is the DEFAULT on this path
// (U2): once per step it moves the gradient part of the forces into P with the projection's own
// operator, so a static balance is exact from the FIRST step. These are the settled V8 exactness
// statements (constant-kappa CSF annihilated; hydrostatic dP/dz exact) at the solver default.
void gateBalancedOn() {
  std::printf("\n=== T1/T1b/T2 ON (the V8 default): static balances exact from step 1\n");
  struct Case {
    const char* name;
    bool periodic, colour;
  };
  const Case cases[3] = {{"periodic, hand-set rho, frozen", true, false},
                         {"walled,   hand-set rho        ", false, false},
                         {"walled,   through C (frozen)  ", false, true}};
  // The PERIODIC box at the default Chebyshev rtol 1e-9 misses by more than 10x (after 1 step: face
  // 5.6e-10, cell 2.0e-9, dP/dz 4.3e-11; after 100: face 9.9e-12) -- and passes at 1e-14 (face
  // 3.0e-14, cell 8.9e-14, dP/dz 1.2e-14), so it is the solve tolerance, not the scheme (doc §9 G4:
  // re-run at Chebyshev 1e-14; only a miss there is conceptual). It is gated at 1e-14; the walled
  // cases pass at the default. WO-P5 (the increment solve, full-RHS stop) does not change the
  // first step -- one solve at rtol 1e-9 relative to the full right-hand side -- and after it the
  // pre-projection skips (1e-9: 100 steps face 2.2e-10; 1e-14: 13 iterations at step 1, then 0,
  // never the cap).
  for (const Case& c : cases)
    for (int steps : {1, 100}) {
      const auto co = hydrostatic<Colo>(1000.0, 0.0, steps, c.periodic, c.colour, -1,
                                        c.periodic ? 1e-14 : -1.0);
      std::printf("  T1 %s %3d step(s)  cell %.3e  face %.3e  dP/dz %.3e  it %ld\n", c.name, steps,
                  co.cellU, co.faceU, co.pErr, co.iters);
      CHECK(co.faceU < 1e-12);
      CHECK(co.cellU < 1e-10);
      CHECK(co.pErr < 1e-10);
    }
  {  // T1 periodic at the DEFAULT Chebyshev rtol (1e-9), 100 steps: the frozen step-1 error.
    // Why 1e-9 and not the 1e-12 above: the pre-projection solves once at step 1 to rtol * |b|
    // relative to the FULL right-hand side (WO-P5), and |b| = |D(O c beta)| is large here (the
    // ratio-1000 hydrostatic force), so rtol * |b| leaves ~1e-10 in X; from step 2 the previous
    // split already meets that test and the solve is SKIPPED, which freezes that step-1 error in
    // the face field instead of letting later solves shave it. Measured 2.2e-10 (WO-P5); gated
    // at 1e-9 so a real regression (a second copy of the balanced pressure, a lost skip) shows.
    // The pre-projection tolerance stays EQUAL to the main solve's (DECIDED 2026-09-25): do not
    // tighten it to pass a tighter gate; the 1e-14 cases above are the exactness statement.
    const auto d = hydrostatic<Colo>(1000.0, 0.0, 100, true, false, -1, -1.0);
    std::printf("  T1 periodic at the default rtol, 100 steps: face %.3e  cell %.3e  dP/dz %.3e\n",
                d.faceU, d.cellU, d.pErr);
    CHECK(d.faceU < 1e-9);
  }
  {  // T1b: the walled column's cell field is exact too (the transient never exists)
    const auto a = hydrostatic<Colo>(1000.0, 0.0, 100, false, false);
    std::printf("  T1b walled ratio 1000, 100 steps: cell %.3e (cb %.3e)\n", a.cellU, a.cb);
    CHECK(a.cellU <= 1e-10);
  }
  const double ratios[4] = {1.0, 10.0, 100.0, 1000.0};
  for (double ratio : ratios) {
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, 0.1, ratio, 0.25, 0.5, 30);
    std::printf("  T2 ratio %6g  cell %.4e  face %.4e  it %ld%s\n", ratio, co.cellU, co.faceU,
                co.iters, co.threw ? "  THREW" : "");
    const double tol = (ratio == 1.0) ? 1e-14 : 1e-13;
    CHECK(!co.threw && co.faceU < tol && co.cellU < tol);
  }
  for (double mu : {0.0, 0.01, 0.1}) {
    const auto st = runDroplet<Stag>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 30);
    const auto co = runDroplet<Colo>(32, 8.0, 1.0, mu, 1000.0, 0.25, 0.5, 30);
    std::printf(
        "  T2 ratio 1000 mu %-5g  staggered (OFF) face %.4e   |   COLLOCATED face %.4e  "
        "cell %.4e\n",
        mu, st.faceU, co.faceU, co.cellU);
    CHECK(!co.threw && co.faceU < 1e-13);
    CHECK(co.faceU < st.faceU);
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
  {  // the pressure force lives in the implicit predictor: the non-incremental step has none
    Colo s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
    s.addField("rho");
    s.setField("rho", std::vector<double>((std::size_t)N * N * N, 1.0));
    s.setDensityMode(true);
    s.setIncrementalPressure(false);
    bool threw = false;
    try {
      s.step();
    } catch (const std::exception&) {
      threw = true;
    }
    std::printf("  non-incremental pressure (collocated V8)     -> throws: %s\n",
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
    gateBalancedOn();
    gateUniformReduction();
    gateBridge();
    gateScope();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "\n%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("\nAll rung V8 collocated gates passed.\n");
  return 0;
}
