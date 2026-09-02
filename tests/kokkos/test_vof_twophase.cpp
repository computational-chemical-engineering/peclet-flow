// VoF rung V2a (WO-J) — the colour field wired into the Navier-Stokes solver: C -> LinearMix
// closures -> rho(C)/mu(C) -> the variable-density projection, with C advected by the projected
// face velocities.
//
// Gates, in the order they are run:
//
//   A  THE BRIDGE. The solver-driven advection must reproduce a STANDALONE `WyAdvector` driven by
//      the same PHYSICAL velocity field. This is the gate that pins the two blocks' index
//      conventions against each other: `flow` puts u(i) on the LOW (-x) face of cell i while
//      `WyAdvector` puts uf(i) on the HIGH (+x) face, so the bridge must shift by one cell along
//      each component's own axis. Omitting that shift is invisible in a uniform flow and invisible
//      in each axis' own discrete divergence, and it destroys conservation (measured: +35 % volume
//      on gate C's scene). The scene is the LeVeque deformation field, whose three components each
//      vary along all three axes — a uniform or a solid-body-rotation field would NOT detect the
//      axial half of the shift.
//
//   B  THE HYDROSTATIC ACID TEST, DRIVEN THROUGH C. A two-layer column at rest under gravity, the
//      density coming from a LinearMix closure on the colour field. Two halves:
//        B1 the pressure: dP/dz must equal -rho_f*g to machine precision, measured against the
//           ACTUAL colour field. This is the loud half — it fails on any inconsistency between the
//           closure's face density and the projection coefficient (the harmonic-rho_f ablation of
//           gate E turns it into a 30 % error).
//        B2 the velocity, with the interface HELD FIXED: max|u| must reproduce the hand-set-rho
//           reference (`test_vardensity_projection.cpp`) BITWISE. Holding the interface fixed is
//           not a weakening of the gate, it is what isolates the quantity the gate exists to
//           measure — see the long note at `hydrostaticC()`.
//
//   C  VOLUME CONSERVATION through the coupled step: a ratio-10 sphere in a sheared periodic box,
//      1000 steps. The floor is the PROJECTION's own discrete divergence residual (WO-E finding 2),
//      so the test reports both.
//
//   D  C == const REDUCTIONS. Three separate statements, because "reduces to the single-phase
//      path" means different things depending on whether a rho closure is registered:
//        D1 a uniform C is EXACTLY stationary under the coupled advection (bitwise);
//        D2 VoF enabled with no closure is bitwise inert — u and P identical to VoF off;
//        D3 VoF + rho(C) with C == 1 is bitwise identical to the hand-set uniform-rho varRho path.
//      (A bitwise reduction to the CONSTANT-density solver is not attainable and never was: with
//      varRho on, the momentum RHS is a different kernel and the default pressure driver is
//      Chebyshev, not MG-PCG. That pre-existing reduction is measured at 2e-14 by
//      `test_vardensity_projection.cpp` and is not a VoF question.)
//
//   E  THE HARMONIC rho_f KNOB (WO-J item 5), measured: it must break the hydrostatic balance, and
//      loudly, because the momentum time term and the face body force keep the arithmetic mean.
//
//   F  THE INTERFACE-LOCAL CFL LIMITER: on a scene with a fast jet far from a quiescent interface,
//      the limiter must report the interface's Courant number, not the domain's.
#include <cmath>
#include <cstdio>
#include <exception>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::WyAdvector;

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
double sum(const std::vector<double>& v) {
  double s = 0;
  for (double x : v)
    s += x;
  return s;
}

// ---------------------------------------------------------------- gate A: the bridge
//
// The LeVeque vector-potential sampler of `vof_advect_scenes.hpp` writes the ADVECTOR's convention
// (u(i) at the high face, x = (i+1)h). The solver's u(i) sits at the LOW face, x = i*h. Both are
// samples of the SAME physical field at the SAME physical points once the index shift is applied,
// so writing the solver side by re-indexing the advector side is a statement about indices, not a
// tautology: `u_solver(i+1) = u_adv(i)`, i.e. solver cell i's low face carries advector cell i-1's
// value.
void bridge() {
  const int N = 24;
  const double h = 1.0 / N, dt = 0.1 * h, T = 3.0;
  // The solver's advector block runs in CELL units (h = 1), so the same Courant number
  // a = u*dt/h is reproduced by handing the solver the SAME face values with dt/h as its dt.
  const double dtSolver = dt / h;
  const int steps = 20;

  // --- reference: the standalone advector, prescribed field -------------------------------------
  WyAdvector ref;
  ref.init(N, N, N, h, peclet::flow::IbmSolver::kVofG);
  // WO-R2 item 4: the reference must be configured like the solver, wisp tolerance included —
  // otherwise this gate measures the DIFFERENCE OF TWO PREDICATES (8.481e-09 with the solver at
  // 1e-8 and the standalone at 0) instead of the bridge it exists to gate.
  ref.wispEps = peclet::flow::IbmSolver::defaultVofWispEps();
  const I3 eR = ref.extent();
  const int gR = ref.ghost();
  const vofscene::Block bR{ref.inner(), eR, gR, I3{0, 0, 0}};
  ref.exchange = [eR, gR](SField f) { vofscene::periodicFill(f, eR, gR, true, true, true); };
  vofscene::initSphere(ref.colour(), bR, h, 0.35, 0.35, 0.35, 0.15);
  ref.syncGhosts();
  for (int s = 0; s < steps; ++s) {
    vofscene::fillLeVeque(ref, bR, h, std::cos(M_PI * (s + 0.5) * dt / T));
    ref.advect(dt, s);
  }
  auto refHost = Kokkos::create_mirror_view(ref.colour());
  Kokkos::deep_copy(refHost, ref.colour());

  // --- the solver path: the same physical field handed to it through set_field("u"/"v"/"w") -----
  // mu = 0, no advection and no body force, so the momentum solve returns u unchanged and the
  // projection is a no-op on an already-divergence-free field: step() transports C and nothing
  // else. The velocity is re-prescribed every step (the field is time dependent).
  peclet::flow::IbmSolver s(N, N, N);
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(dtSolver);
  // bitwise / machine-precision gates: keep the legacy fixed-sweep momentum loop (the default
  // residual stop, 1e-5 since 2026-09-02, leaves an O(1e-12) momentum residual)
  s.setVelocityResidualTolerance(0.0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
  s.enableVof();
  {  // the same sphere, sampled by the same helper on a throwaway advector block
    WyAdvector seed;
    seed.init(N, N, N, h, peclet::flow::IbmSolver::kVofG);
    const vofscene::Block bS{seed.inner(), seed.extent(), seed.ghost(), I3{0, 0, 0}};
    vofscene::initSphere(seed.colour(), bS, h, 0.35, 0.35, 0.35, 0.15);
    auto sh = Kokkos::create_mirror_view(seed.colour());
    Kokkos::deep_copy(sh, seed.colour());
    const I3 eS = seed.extent();
    const int gS = seed.ghost();
    std::vector<double> c0((std::size_t)N * N * N);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          c0[idx(x, y, z, N, N)] = sh(L3(x + gS, y + gS, z + gS, eS));
    s.setVof(c0);
  }
  // Host samplers of the ADVECTOR's face values, so the solver side is the same numbers re-indexed.
  WyAdvector probe;
  probe.init(N, N, N, h, peclet::flow::IbmSolver::kVofG);
  const I3 eP = probe.extent();
  const int gP = probe.ghost();
  const vofscene::Block bP{probe.inner(), eP, gP, I3{0, 0, 0}};
  std::vector<double> uu((std::size_t)N * N * N), vv(uu.size()), ww(uu.size());
  for (int st = 0; st < steps; ++st) {
    vofscene::fillLeVeque(probe, bP, h, std::cos(M_PI * (st + 0.5) * dt / T));
    auto hu = Kokkos::create_mirror_view(probe.faceU());
    auto hv = Kokkos::create_mirror_view(probe.faceV());
    auto hw = Kokkos::create_mirror_view(probe.faceW());
    Kokkos::deep_copy(hu, probe.faceU());
    Kokkos::deep_copy(hv, probe.faceV());
    Kokkos::deep_copy(hw, probe.faceW());
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          // solver low face of cell i == advector high face of cell i-1 (periodic wrap)
          const int xm = (x + N - 1) % N, ym = (y + N - 1) % N, zm = (z + N - 1) % N;
          uu[idx(x, y, z, N, N)] = hu(L3(xm + gP, y + gP, z + gP, eP));
          vv[idx(x, y, z, N, N)] = hv(L3(x + gP, ym + gP, z + gP, eP));
          ww[idx(x, y, z, N, N)] = hw(L3(x + gP, y + gP, zm + gP, eP));
        }
    s.setField("u", uu);
    s.setField("v", vv);
    s.setField("w", ww);
    s.step();
  }
  const auto cs = s.getVof();
  double dmax = 0.0, l1 = 0.0;
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double d = cs[idx(x, y, z, N, N)] - refHost(L3(x + gR, y + gR, z + gR, eR));
        dmax = std::fmax(dmax, std::fabs(d));
        l1 += std::fabs(d);
      }
  std::printf("A bridge (LeVeque, %d steps): max|C_solver - C_standalone| %.3e  L1 %.3e\n", steps,
              dmax, l1);
  // Not bitwise: the solver re-projects the prescribed field each step, so the face values it
  // advects with differ from the prescribed ones at the projection's residual (~1e-16 relative).
  CHECK(dmax < 1e-11);
}

// ---------------------------------------------------------------- gate B: the acid test
//
// WHY THE VELOCITY HALF IS RUN WITH THE INTERFACE HELD FIXED.
//
// The acid test exists to catch an inconsistency between the momentum face density, the face body
// force and the projection face coefficient — the three-way agreement that makes
//   w* = w^n + (dt/rho_f)(-dP/dz + f_f) = w^n
// hold exactly (doc/variable_density_projection.md §3). The colour field is not part of that
// statement; it only supplies rho. So the sharpest form of the gate through the C chain is: freeze
// C and demand the hand-set-rho number, BITWISE. That is what `freeze = true` measures, and it
// returns 2.176e-17 == the reference to the last bit.
//
// With the interface FREE the rest state is no longer a stable fixed point of the DISCRETE system,
// and that is a property of the physics-plus-time-discretisation, not of the wiring:
//   * the colour field is an extra degree of freedom, and any residual velocity eps displaces it by
//     eps*dt;
//   * that displacement changes rho by (rho_l - rho_g)*eps*dt, so it changes the body force by
//     g*(rho_l-rho_g)*eps*dt and the resulting acceleration in the LIGHT layer by
//     g*(rho_l-rho_g)*eps*dt/rho_g — a loop gain of g*drho*dt/rho_g, which is 100 at ratio 1000,
//     g = 0.1, dt = 1;
//   * the projection removes only the part of that perturbation that is a discrete gradient. What
//     survives is an interfacial gravity-wave mode, and it is undamped at mu = 0.
// Measured, starting from the exact fixed point 2.176e-17 (200 steps): the residual reaches
//   gain 100 (ratio 1e3, g = 0.1, dt = 1)   -> 2.4e-12
//   gain  10 (ratio 1e3, g = 0.1, dt = 0.1) -> 7.9e-15
//   gain   1 (ratio 1e3, g = 1e-3, dt = 1)  -> 1.2e-16
//   gain 0.2 (ratio 3,   g = 0.1, dt = 1)   -> 3.0e-14
// i.e. the level it settles at is set by the loop gain, exactly as the mechanism predicts, and it
// is NOT set by anything in the bridge (which gate A pins independently, and which gate B2 pins to
// the last bit). The free-interface residual is therefore recorded and bounded, not gated at
// machine zero — see the WO-J findings entry.
double hydrostaticC(double ratio, bool freeze, bool harmonic, int steps, double g, double dt) {
  const int N = 8, NZ = 24;
  peclet::flow::IbmSolver s(N, N, NZ);
  s.setRho(1.0);
  s.setMu(0.0);  // inviscid: the balance is exact (no viscous wall layer in the predictor)
  // bitwise / machine-precision gates: keep the legacy fixed-sweep momentum loop (the default
  // residual stop, 1e-5 since 2026-09-02, leaves an O(1e-12) momentum residual)
  s.setVelocityResidualTolerance(0.0);
  s.setDt(dt);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z, periodic x,y
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
  std::vector<double> c0((std::size_t)N * N * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        c0[idx(x, y, z, N, N)] = (z < NZ / 2) ? 1.0 : 0.0;  // sharp: heavy (liquid) below
  if (freeze) {
    s.addField("C");
    s.setField("C", c0);
    s.exchangeField("C");
  } else {
    s.setVof(c0);
  }
  // rho = 1 + (ratio-1)*C  (enables the variable-density path), then gravity from rho.
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, ratio - 1.0});
  s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -g});
  if (harmonic)
    s.setRhoFaceHarmonic(true);
  double last = 0.0;
  for (int it = 0; it < steps; ++it) {
    s.step();
    last = std::fmax(maxAbs(s.getVelocity(0)),
                     std::fmax(maxAbs(s.getVelocity(1)), maxAbs(s.getVelocity(2))));
    CHECK(!std::isnan(last));
    if (std::isnan(last))
      return last;
  }
  // dP/dz against the ACTUAL colour column (with a free interface C is no longer the initial one).
  const auto cNow = freeze ? c0 : s.getVof();
  const auto p = s.getPressure();
  double perr = 0;
  const int xc = N / 2, yc = N / 2;
  for (int z = 1; z < NZ; ++z) {
    const double dp = p[idx(xc, yc, z, N, N)] - p[idx(xc, yc, z - 1, N, N)];
    const double ra = 1.0 + (ratio - 1.0) * cNow[idx(xc, yc, z, N, N)];
    const double rb = 1.0 + (ratio - 1.0) * cNow[idx(xc, yc, z - 1, N, N)];
    perr = std::fmax(perr, std::fabs(dp + g * 0.5 * (ra + rb)) / (g * ratio));
  }
  std::printf("B ratio %-6g %-14s: steady max|u| %.3e   dP/dz rel-err %.3e\n", ratio,
              harmonic ? "HARMONIC rho_f" : (freeze ? "frozen C" : "free C"), last, perr);
  if (!harmonic)
    CHECK(perr < 1e-11);
  return freeze ? last : perr;
}

// The hand-set-rho reference of test_vardensity_projection.cpp, on the identical grid, so the
// frozen-interface C chain can be compared to it BITWISE.
double hydrostaticRhoRef(double ratio, int steps, double g, double dt) {
  const int N = 8, NZ = 24;
  peclet::flow::IbmSolver s(N, N, NZ);
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(dt);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
  std::vector<double> rho((std::size_t)N * N * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        rho[idx(x, y, z, N, N)] = (z < NZ / 2) ? ratio : 1.0;
  s.addField("rho");
  s.setField("rho", rho);
  s.setDensityMode(true);
  s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -g});
  double last = 0.0;
  for (int it = 0; it < steps; ++it) {
    s.step();
    last = std::fmax(maxAbs(s.getVelocity(0)),
                     std::fmax(maxAbs(s.getVelocity(1)), maxAbs(s.getVelocity(2))));
  }
  return last;
}

// ---------------------------------------------------------------- gate C: conservation
void conservation(int steps) {
  const int N = 24;
  const double dt = 1.0, amp = 5e-4, ratio = 10.0;
  peclet::flow::IbmSolver s(N, N, N);
  s.setRho(1.0);
  s.setMu(0.05);
  s.setDt(dt);
  s.setAdvection(true);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
  s.enableVof();
  std::vector<double> c0((std::size_t)N * N * N), fx(c0.size());
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - N / 2.0, dy = y + 0.5 - N / 2.0, dz = z + 0.5 - N / 2.0;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        c0[idx(x, y, z, N, N)] = std::fmin(1.0, std::fmax(0.0, 0.5 - (r - N / 5.0)));
        fx[idx(x, y, z, N, N)] = amp * std::sin(2 * M_PI * (z + 0.5) / N);  // shear
      }
  s.setVof(c0);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, ratio - 1.0});
  s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {0.05, 0.45});
  s.enableCellForce();
  s.setField("force_x", fx);
  const double v0 = sum(s.getVof());
  double dmax = 0.0, cfl = 0.0, peak = 0.0, drift = 0.0;
  for (int it = 0; it < steps; ++it) {
    s.step();
    dmax = std::fmax(dmax, std::fabs(s.maxOpenDivergence()));
    cfl = std::fmax(cfl, s.vofLastCourant());
    drift = (sum(s.getVof()) - v0) / v0;  // the PEAK excursion, not just where it ends up
    peak = std::fmax(peak, std::fabs(drift));
  }
  const auto d = s.vofDiagnostics();
  std::printf(
      "C conservation (%d coupled steps, ratio %g): dV/V final %+.3e  PEAK %.3e   "
      "max|div(open u)| %.2e   max cfl %.4f   C in [%.2e, %.10f]   mixed %ld\n",
      steps, ratio, drift, peak, dmax, cfl, d.minC, d.maxC, d.mixed);
  // The floor here is the PROJECTION's discrete divergence residual, not the advection: WY adds
  // H(C-1/2)*div*dt/h to every full cell's budget, so the conservation floor is whatever the
  // pressure solve leaves (WO-E finding 2 stated this in advance). Measured on this scene the
  // residual wanders inside a band and does not accumulate. V1's own floor, with a prescribed
  // field at max|div| ~ 1e-15, was 5.7e-14 over 3200 steps.
  CHECK(peak < 1e-12);
  CHECK(d.maxC <= 1.0 + 1e-12 && d.minC > -1e-12);
}

// ---------------------------------------------------------------- gate D: C == const reductions
void constantColour() {
  const int N = 12;
  const std::size_t n = (std::size_t)N * N * N;
  // All-fluid periodic box (VoF with an immersed solid is refused at this rung) with a SHEARED body
  // force and advection on, so u and P are non-trivial fields and a bitwise comparison means
  // something.
  std::vector<double> fx(n), ones(n, 1.0);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        fx[idx(x, y, z, N, N)] = 1e-3 * std::sin(2 * M_PI * (z + 0.5) / N);
  auto build = [&](peclet::flow::IbmSolver& s) {
    s.setRho(2.0);
    s.setMu(0.1);
    s.setDt(2.0);
    s.setVelocityResidualTolerance(0.0);  // bitwise gates: legacy fixed-sweep momentum loop
    s.setAdvection(true);
    s.setPressureGeometry(std::vector<double>(n, 10.0));
    s.enableCellForce();
    s.setField("force_x", fx);
  };

  // D1 + D2: VoF on, NO closure. A uniform C must be bitwise stationary, and the flow must be
  // bitwise identical to the same run with VoF off.
  std::vector<std::vector<double>> u(2), p(2);
  std::vector<double> cEnd;
  for (int vofOn = 0; vofOn < 2; ++vofOn) {
    peclet::flow::IbmSolver s(N, N, N);
    build(s);
    if (vofOn)
      s.setVof(ones);
    for (int it = 0; it < 30; ++it)
      s.step();
    u[vofOn] = s.getVelocity(0);
    p[vofOn] = s.getPressure();
    if (vofOn)
      cEnd = s.getVof();
  }
  double dC = 0;
  for (double c : cEnd)
    dC = std::fmax(dC, std::fabs(c - 1.0));
  const double du = maxAbsDiff(u[0], u[1]), dp = maxAbsDiff(p[0], p[1]);
  std::printf(
      "D1 uniform C stationary: max|C-1| %.3e\nD2 VoF inert (no closure): du %.3e dp %.3e "
      "(|u| %.3e)\n",
      dC, du, dp, maxAbs(u[0]));
  CHECK(dC == 0.0);
  CHECK(du == 0.0 && dp == 0.0);

  // D3: VoF + rho(C) with C == 1 must be bitwise the hand-set uniform-rho varRho path.
  std::vector<std::vector<double>> uv(2), pv(2);
  for (int viaC = 0; viaC < 2; ++viaC) {
    peclet::flow::IbmSolver s(N, N, N);
    build(s);
    if (viaC) {
      s.setVof(ones);
      s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {0.0, 2.0});
    } else {
      s.addField("rho");
      s.setField("rho", std::vector<double>(n, 2.0));
      s.setDensityMode(true);
    }
    for (int it = 0; it < 30; ++it)
      s.step();
    uv[viaC] = s.getVelocity(0);
    pv[viaC] = s.getPressure();
  }
  const double duv = maxAbsDiff(uv[0], uv[1]), dpv = maxAbsDiff(pv[0], pv[1]);
  std::printf("D3 rho(C==1) == hand-set uniform rho: du %.3e dp %.3e\n", duv, dpv);
  CHECK(duv == 0.0 && dpv == 0.0);
}

// ---------------------------------------------------------------- gate F: interface-local CFL
//
// A localized jet far from a quiescent interface: the global max Courant number is the jet's, the
// interface-local one is the interface's. V1 measured the same effect on Zalesak (0.314 at a
// quiescent corner versus 0.157 at the interface) and it is why a global max must not set the VoF
// time step (VOF_PLAN.md §6).
void interfaceLocalCfl() {
  const int N = 24;
  const double dt = 1.0;
  peclet::flow::IbmSolver s(N, N, N);
  s.setRho(1.0);
  s.setMu(0.05);
  s.setDt(dt);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * N, 10.0));
  s.enableVof();
  std::vector<double> c0((std::size_t)N * N * N), fx(c0.size());
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - N / 2.0, dy = y + 0.5 - N / 2.0, dz = z + 0.5 - N / 2.0;
        c0[idx(x, y, z, N, N)] =
            std::fmin(1.0, std::fmax(0.0, 0.5 - (std::sqrt(dx * dx + dy * dy + dz * dz) - 3.0)));
        // a jet localized around z = 3, i.e. ~9 cells from the nearest interface cell
        const double a = (z + 0.5 - 3.0) / 1.5;
        fx[idx(x, y, z, N, N)] = 2e-3 * std::exp(-a * a);
      }
  s.setVof(c0);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, 9.0});
  s.enableCellForce();
  s.setField("force_x", fx);
  for (int it = 0; it < 40; ++it)
    s.step();
  const double local = s.vofLastCourant();
  double global = 0.0;
  for (int c = 0; c < 3; ++c)
    global = std::fmax(global, maxAbs(s.getVelocity(c)) * dt);
  std::printf("F CFL: interface-local %.4f   global %.4f   (over-throttle factor %.2fx)\n", local,
              global, global / std::fmax(local, 1e-300));
  CHECK(local < 0.5 * global);  // the whole point: the far-field jet must not set the VoF dt
  CHECK(local > 0.0);           // ... and the interface's own faces must still be seen
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bridge();
    // B1/B2: the acid test through the C chain. `freeze` isolates the closure/projection
    // consistency (bitwise against the hand-set-rho reference); the free-interface run records the
    // residual the moving-interface degree of freedom leaves.
    for (double ratio : {3.0, 1000.0}) {
      const double frozen = hydrostaticC(ratio, /*freeze=*/true, false, 100, 0.1, 1.0);
      const double ref = hydrostaticRhoRef(ratio, 100, 0.1, 1.0);
      std::printf("B2 ratio %-6g frozen-C chain vs hand-set rho: %.17g vs %.17g  (%s)\n", ratio,
                  frozen, ref, frozen == ref ? "BITWISE EQUAL" : "DIFFER");
      CHECK(frozen == ref);
      CHECK(frozen < 1e-12);
      hydrostaticC(ratio, /*freeze=*/false, false, 100, 0.1, 1.0);
    }
    // E: the harmonic rho_f ablation must break the balance loudly.
    //
    // WO-R2 item 3: "loudly" got LOUDER under the exact level-0 operator, which is now the default
    // once VoF is on. With the float bands the ablation settles at a dP/dz error of 0.3355; with
    // the exact operator the same run leaves the Weymouth-Yue boundedness cap before step 100 and
    // the advector throws. Both outcomes confirm the same statement (the harmonic face mean is not
    // the consistent one for this discretization), so the gate accepts either — and says which it
    // saw, because the two are different measurements. The ARITHMETIC runs above are unaffected
    // and in fact tighter under the exact operator (ratio 1000 free C: dP/dz 1.02e-15 -> 4.26e-16).
    {
      double perr = 0.0;
      bool blewUp = false;
      try {
        perr = hydrostaticC(1000.0, false, /*harmonic=*/true, 100, 0.1, 1.0);
        std::printf("E harmonic rho_f dP/dz rel-err %.3e (arithmetic is the consistent mean)\n",
                    perr);
      } catch (const std::exception& ex) {
        blewUp = true;
        std::printf("E harmonic rho_f: the ablation left the boundedness cap altogether — %s\n",
                    ex.what());
      }
      CHECK(blewUp || perr > 1e-3);
    }
    constantColour();
    interfaceLocalCfl();
    conservation(std::getenv("PECLET_VOF_TWOPHASE_LONG") ? 1000 : 200);
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
