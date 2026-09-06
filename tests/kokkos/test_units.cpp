// PHYSICAL DOMAINS — the two gates of suite/docs/PHYSICAL_UNITS_PLAN.md Phase 1.
//
// Run as `test_units identity` and `test_units scale` (two ctests off one binary).
//
// GATE 1 — units_identity.  The physical constructor at `extent == cells` reproduces the cell-unit
// solver exactly: every accessor reports the historical numbers, and with rho = 1 and dt = 1 (where
// all three reference scales are 1) the fields come back BITWISE equal to an extent=None run.  This
// is the anchor of the whole refactor: an armed solver that happens to sit on the unit lattice must
// be the unit-lattice solver.
//
// GATE 2 — units_scale_invariance.  The SAME physical problem, written in two unit systems whose
// length unit differs by a factor 1000, must give the same physical answer.  Under a pure change of
// length unit (time and mass units fixed, numerical lengths divided by lam):
//
//     L -> L/lam    u -> u/lam    rho -> rho*lam^3    mu -> mu*lam    F -> F*lam^2    p -> p*lam
//     dt unchanged  sdf -> sdf/lam
//
// Substituting those into the boundary conversions of `Solver::UnitScales` leaves EVERY internal
// quantity unchanged (mu' = mu*tRef/(rhoRef*hRef^2) is invariant, and so are dt', rho', F', the
// index velocity and the index SDF), so the two runs solve the identical discrete system and can
// only differ by the round-off of forming the conversion factors themselves.  Any surviving `h` in
// a kernel or at the API boundary breaks that invariance immediately — which is exactly what this
// gate is for (plan §5.2, target 1e-13 relative).
//
// Both a SAMPLED SDF (set_solid) and an ANALYTIC SCENE (set_scene + set_solid_from_scene) are
// carried through the scale change, on Poiseuille and on a periodic sphere in Stokes flow.
//
// GATE 3 — units_vof_sigma (`test_units vof`).  The V4 surface-tension gates with a PHYSICAL sigma
// at extent = 1e-2 * cells (plan §9.3 U5, §5.6).  Surface tension is a force per unit LENGTH, i.e.
// mass per time squared, so under a pure change of LENGTH unit sigma is numerically INVARIANT while
// mu -> mu*lam, rho -> rho*lam^3, kappa -> kappa*lam, p -> p*lam and u -> u/lam.  Two checks, the
// two the V4 rung rests on: the exactness identity (a CONSTANT curvature makes the CSF the exact
// discrete gradient of sigma*kappa*C, which the projection annihilates, so a static droplet stays
// at machine zero) and Young-Laplace (the pressure jump equals sigma*kappa).  Both must give the
// same PHYSICS in the two unit systems.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

void checkClose(double a, double b, double tol, const char* what) {
  if (!(std::fabs(a - b) <= tol)) {
    std::fprintf(stderr, "CHECK_CLOSE failed (%s): %.17g vs %.17g  (|d| = %.3e > %.3e)\n", what, a,
                 b, std::fabs(a - b), tol);
    ++failures;
  }
}

bool bitwiseEqual(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

double maxAbs(const std::vector<double>& a) {
  double m = 0.0;
  for (double v : a)
    if (!(std::fabs(v) <= m))
      m = std::fabs(v);
  return m;
}

/// max |a - k*b| / max|a| — the relative agreement of `a` with `b` rescaled by the exact unit
/// factor `k`.
double relDiffScaled(const std::vector<double>& a, const std::vector<double>& b, double k) {
  const double den = maxAbs(a);
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i] - k * b[i]);
    if (!(d <= m))
      m = d;  // NaN-propagating
  }
  return den > 0.0 ? m / den : m;
}

// --------------------------------------------------------------------------------------------
// The two problems, both written in ONE unit system parameterised by `lam` (lam = 1 is the
// cell-unit system: side N, spacing 1). Everything below is stated physically; nothing computes
// with a cell size.
// --------------------------------------------------------------------------------------------
struct Fields {
  std::vector<double> u, v, w, p;
  std::array<double, 3> spacing{};
  double div = 0.0;
};

/// Plane Poiseuille between two cut-cell walls, body-force driven, run to steady.
template <class Grid>
Fields runChannel(int nx, int ny, int nz, double lam, bool arm) {
  const double RHO = 1.0 * lam * lam * lam, MU = 0.1 * lam, F = 0.01 * lam * lam, DT = 50.0;
  const double H = 1.0 / lam;  // one cell, physically
  const double ylo = (std::round(0.30 * ny) + 0.5) * H, yhi = (std::round(0.70 * ny) + 0.5) * H;
  peclet::flow::Solver<Grid> s(nx, ny, nz);
  if (arm)
    s.setPhysicalDomain({nx * H, ny * H, nz * H}, {0.0, 0.0, 0.0}, {nx, ny, nz});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0.0, 0.0);
  s.setVelocityIterations(400);
  s.setVelocityResidualTolerance(1e-14);
  s.setPressurePcg(true, 80, 1e-13);
  // The SDF is a PHYSICAL signed distance sampled at the physical cell centres.
  const std::vector<double> cx = s.cellCentres(0), cy = s.cellCentres(1), cz = s.cellCentres(2);
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
            std::min(cy[y] - ylo, yhi - cy[y]);
  (void)cx;
  (void)cz;
  s.setSolid(sdf, /*cutcellPressure=*/false);
  for (int it = 0; it < 300; ++it)
    s.step();
  Fields f;
  f.u = s.getVelocity(0);
  f.v = s.getVelocity(1);
  f.w = s.getVelocity(2);
  f.p = s.getPressure();
  f.spacing = s.spacing();
  return f;
}

/// A periodic sphere in creeping flow, cut-cell pressure + multigrid. `scene = true` builds the
/// geometry from an ANALYTIC scene in physical coordinates instead of a sampled array.
template <class Grid>
Fields runSphere(int N, double lam, bool arm, bool scene) {
  const double RHO = 1.0 * lam * lam * lam, MU = 0.1 * lam, F = 1e-3 * lam * lam, DT = 60.0;
  const double H = 1.0 / lam;
  const double R = 0.30 * N * H;
  peclet::flow::Solver<Grid> s(N, N, N);
  if (arm)
    s.setPhysicalDomain({N * H, N * H, N * H}, {0.0, 0.0, 0.0}, {N, N, N});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0.0, 0.0);
  s.setAdvection(false);
  s.setVelocityIterations(200);
  s.setVelocityResidualTolerance(1e-14);
  s.setPressurePcg(true, 120, 1e-13);
  if (scene) {
    namespace g = peclet::core::geom;
    g::SceneBuilder<double> b;
    b.addLeaf(g::kSphere, {R});
    g::Transform<double> tr;
    // Centre of the box, in PHYSICAL coordinates.
    tr.translation = peclet::core::Vec3<double>{0.5 * N * H, 0.5 * N * H, 0.5 * N * H};
    b.addInstance(0, tr);
    std::vector<int> ni, ii;
    std::vector<double> nr, ir;
    b.encode(ni, nr, ii, ir);
    s.setScene(ni, nr, ii, ir, /*periodic=*/true);
    s.setSolidFromScene(/*cutcellPressure=*/true);
  } else {
    const std::vector<double> cx = s.cellCentres(0), cy = s.cellCentres(1), cz = s.cellCentres(2);
    const double c = 0.5 * N * H;
    std::vector<double> sdf((std::size_t)N * N * N);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
              std::sqrt((cx[x] - c) * (cx[x] - c) + (cy[y] - c) * (cy[y] - c) +
                        (cz[z] - c) * (cz[z] - c)) -
              R;
    s.setSolid(sdf, /*cutcellPressure=*/true);
  }
  for (int it = 0; it < 60; ++it)
    s.step();
  Fields f;
  f.u = s.getVelocity(0);
  f.v = s.getVelocity(1);
  f.w = s.getVelocity(2);
  f.p = s.getPressure();
  f.spacing = s.spacing();
  f.div = s.maxOpenDivergence();
  return f;
}

/// Exact volume fraction of a sphere of radius R (in CELLS) centred at (cx,cy,cz), sub-sampled in
/// x and y and integrated analytically in z. The colour field is dimensionless and identical in
/// every unit system — the geometry is the same cells — which is what makes the comparison below a
/// statement about sigma and nothing else.
std::vector<double> sphereColour(int n, double R, double cx, double cy, double cz, int sub = 24) {
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
            const double hh = std::sqrt(r2);
            const double lo = std::fmax(cz - hh, (double)k), hi = std::fmin(cz + hh, (double)k + 1);
            if (hi > lo)
              acc += hi - lo;
          }
        C[(std::size_t)i + (std::size_t)j * n + (std::size_t)k * n * n] = acc / (sub * sub);
      }
  return C;
}

struct Drop {
  double maxU = 0.0;   ///< max |u| over the three components, PHYSICAL
  double dp = 0.0;     ///< the measured Young-Laplace jump, PHYSICAL
  double kappa = 0.0;  ///< the curvature imposed or measured, PHYSICAL (1/length)
  double dtCap = 0.0;  ///< capillaryDt(), PHYSICAL
};

/// A stationary droplet in a triply periodic box, off-centre by an irrational fraction of a cell.
/// `lam` scales the LENGTH unit: lengths / lam, mu * lam, rho * lam^3, sigma unchanged.
Drop runDroplet(int n, double Rcells, double lam, bool constantKappa, int steps) {
  const double H = 1.0 / lam;              // one cell, physically
  const double SIGMA = 1.0;                // invariant under a change of LENGTH unit
  const double MU = 0.1 * lam, RHO = 1.0 * lam * lam * lam;
  peclet::flow::Solver<peclet::flow::Staggered> s(n, n, n);
  s.setPhysicalDomain({n * H, n * H, n * H}, {0.0, 0.0, 0.0}, {n, n, n});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(1.0);
  s.setVelocityResidualTolerance(0.0);  // machine-precision gate: keep the fixed-sweep loop
  s.setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 10.0 * H));
  s.setPressureChebyshev(true, 500, 1e-14);
  s.enableVof();
  const std::vector<double> C =
      sphereColour(n, Rcells, n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11);
  s.setVof(C);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {RHO, 0.0});
  s.setSurfaceTension(SIGMA);
  Drop d;
  d.kappa = 2.0 / (Rcells * H);  // physical 1/length
  if (constantKappa)
    s.setVofKappaConstant(d.kappa);
  else
    s.computeVofCurvature();
  d.dtCap = s.capillaryDt();
  s.setDt(0.5 * d.dtCap);
  for (int k = 0; k < steps; ++k)
    s.step();
  for (int c = 0; c < 3; ++c)
    d.maxU = std::fmax(d.maxU, maxAbs(s.getVelocity(c)));
  // Young-Laplace: the pressure inside the drop minus the pressure well outside it.
  const std::vector<double> p = s.getPressure();
  const std::size_t inside = (std::size_t)(n / 2) + (std::size_t)(n / 2) * n +
                             (std::size_t)(n / 2) * n * n;
  const std::size_t outside = 1 + (std::size_t)1 * n + (std::size_t)1 * n * n;
  d.dp = p[inside] - p[outside];
  return d;
}

// --------------------------------------------------------------------------------------------
void gateVofSigma() {
  std::printf("=== units_vof_sigma ===\n");
  const int N = 32;
  const double Rc = 8.0;
  const double lam = 1e2;  // extent = 1e-2 * cells

  // (a) the exactness identity, in both unit systems.
  const Drop a = runDroplet(N, Rc, 1.0, true, 20);
  const Drop b = runDroplet(N, Rc, lam, true, 20);
  std::printf("  constant kappa: max|u| %.3e (lam=1) vs %.3e (lam=%g)  ->  %.3e rescaled\n",
              a.maxU, b.maxU, lam, b.maxU * lam);
  // A velocity of exactly zero would also pass, so pin the SCALE the currents are measured against:
  // sigma*kappa/mu is the capillary velocity the naive CSF would produce.
  const double uScale = 1.0 * a.kappa / 0.1;
  CHECK(a.maxU < 1e-14 * uScale);
  CHECK(b.maxU * lam < 1e-14 * uScale);
  checkClose(a.dp, 1.0 * a.kappa, 1e-9 * a.kappa, "Young-Laplace at lam=1");
  checkClose(b.dp / lam, a.dp, 1e-9 * a.kappa, "Young-Laplace at lam=1e2 (rescaled)");
  std::printf("  Young-Laplace: dp %.9e vs sigma*kappa %.9e   (lam=%g: %.9e rescaled)\n", a.dp,
              a.kappa, lam, b.dp / lam);
  checkClose(b.kappa / lam, a.kappa, 1e-12 * a.kappa, "kappa scales as 1/length");
  checkClose(b.dtCap, a.dtCap, 1e-12 * a.dtCap, "capillary_dt invariant under a length-unit change");
  std::printf("  capillary_dt %.9e vs %.9e (invariant: sigma and the time unit are)\n", a.dtCap,
              b.dtCap);

  // (b) the COMPUTED curvature: the physical spurious currents and the jump must agree.
  const Drop ca = runDroplet(N, Rc, 1.0, false, 20);
  const Drop cb = runDroplet(N, Rc, lam, false, 20);
  const double ru = std::fabs(ca.maxU - cb.maxU * lam) / std::fabs(ca.maxU);
  const double rp = std::fabs(ca.dp - cb.dp / lam) / std::fabs(ca.dp);
  std::printf("  computed kappa: max|u| %.6e vs %.6e rescaled (rel %.3e); dp %.6e vs %.6e "
              "(rel %.3e)\n",
              ca.maxU, cb.maxU * lam, ru, ca.dp, cb.dp / lam, rp);
  CHECK(ru <= 1e-12);
  CHECK(rp <= 1e-12);
}

// --------------------------------------------------------------------------------------------
void gateIdentity() {
  std::printf("=== units_identity ===\n");

  // (a) No domain: the historical cell-unit reports.
  {
    peclet::flow::Solver<peclet::flow::Staggered> s(6, 8, 10);
    CHECK(!s.hasPhysicalDomain());
    const auto h = s.spacing();
    CHECK(h[0] == 1.0 && h[1] == 1.0 && h[2] == 1.0);
    const auto e = s.domainExtent();
    CHECK(e[0] == 6.0 && e[1] == 8.0 && e[2] == 10.0);
    const auto o = s.domainOrigin();
    CHECK(o[0] == 0.0 && o[1] == 0.0 && o[2] == 0.0);
    const auto c = s.cellCentres(1);
    CHECK(c.size() == 8u);
    CHECK(c[0] == 0.5 && c[7] == 7.5);
  }

  // (b) extent == cells with a shifted origin: spacing 1, centres shifted, scales all 1.
  {
    peclet::flow::Solver<peclet::flow::Staggered> s(6, 8, 10);
    s.setPhysicalDomain({6.0, 8.0, 10.0}, {-2.0, 0.5, 100.0}, {6, 8, 10});
    CHECK(s.hasPhysicalDomain());
    const auto h = s.spacing();
    CHECK(h[0] == 1.0 && h[1] == 1.0 && h[2] == 1.0);
    const auto o = s.domainOrigin();
    CHECK(o[0] == -2.0 && o[1] == 0.5 && o[2] == 100.0);
    const auto cx = s.cellCentres(0);
    CHECK(cx[0] == -1.5 && cx[5] == 3.5);
    const auto cz = s.cellCentres(2);
    CHECK(cz[0] == 100.5 && cz[9] == 109.5);
    const auto& u = s.unitScales();
    CHECK(u.hRef == 1.0);
    s.setRho(1.0);
    s.setDt(1.0);
    CHECK(u.rhoRef == 1.0 && u.tRef == 1.0);
    CHECK(u.muToInt() == 1.0 && u.pToInt() == 1.0 && u.forceToInt(0) == 1.0);
    CHECK(u.velToInt(2) == 1.0 && u.lenToInt() == 1.0 && u.sigmaToInt() == 1.0);
  }

  // (c) BITWISE: an armed run at extent == cells with rho = dt = 1 against extent=None.
  {
    auto run = [](bool arm) {
      const int nx = 8, ny = 24, nz = 8;
      peclet::flow::Solver<peclet::flow::Staggered> s(nx, ny, nz);
      if (arm)
        s.setPhysicalDomain({(double)nx, (double)ny, (double)nz}, {0.0, 0.0, 0.0}, {nx, ny, nz});
      s.setRho(1.0);
      s.setMu(0.1);
      s.setDt(1.0);
      s.setBodyForce(0.01, 0.0, 0.0);
      s.setVelocityIterations(60);
      s.setPressureIterations(4);
      const double ylo = 7.5, yhi = ny - 7.5;
      std::vector<double> sdf((std::size_t)nx * ny * nz);
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
                std::min((y + 0.5) - ylo, yhi - (y + 0.5));
      s.setSolid(sdf, /*cutcellPressure=*/true);
      for (int it = 0; it < 30; ++it)
        s.step();
      Fields f;
      f.u = s.getVelocity(0);
      f.v = s.getVelocity(1);
      f.w = s.getVelocity(2);
      f.p = s.getPressure();
      return f;
    };
    const Fields a = run(false), b = run(true);
    CHECK(maxAbs(a.u) > 0.0);
    CHECK(bitwiseEqual(a.u, b.u));
    CHECK(bitwiseEqual(a.v, b.v));
    CHECK(bitwiseEqual(a.w, b.w));
    CHECK(bitwiseEqual(a.p, b.p));
    std::printf("  extent == cells, rho = dt = 1: bitwise identical to the cell-unit run\n");
  }
}

// --------------------------------------------------------------------------------------------
void gateScale() {
  std::printf("=== units_scale_invariance ===\n");
  const double lam = 1e3;  // the second unit system's length unit is 1000x larger
  const double TOL = 1e-13;

  {
    const int nx = 8, ny = 24, nz = 8;
    const Fields a = runChannel<peclet::flow::Staggered>(nx, ny, nz, 1.0, true);
    const Fields b = runChannel<peclet::flow::Staggered>(nx, ny, nz, lam, true);
    checkClose(a.spacing[0], 1.0, 0.0, "channel spacing (system A)");
    checkClose(b.spacing[0], 1.0 / lam, 1e-18, "channel spacing (system B)");
    // u_B = u_A / lam and p_B = p_A * lam, exactly.
    const double ru = relDiffScaled(a.u, b.u, lam);
    const double rp = relDiffScaled(a.p, b.p, 1.0 / lam);
    std::printf("  Poiseuille (sampled SDF): u %.3e   p %.3e\n", ru, rp);
    CHECK(maxAbs(a.u) > 0.0);
    CHECK(ru <= TOL);
    CHECK(rp <= TOL);
  }

  for (int scene = 0; scene < 2; ++scene) {
    const int N = 24;
    const Fields a = runSphere<peclet::flow::Staggered>(N, 1.0, true, scene != 0);
    const Fields b = runSphere<peclet::flow::Staggered>(N, lam, true, scene != 0);
    const double ru = relDiffScaled(a.u, b.u, lam);
    const double rv = relDiffScaled(a.v, b.v, lam);
    const double rp = relDiffScaled(a.p, b.p, 1.0 / lam);
    // The divergence diagnostic is 1/time and the time unit did not change, so it is invariant —
    // but it is a solver RESIDUAL sitting at ~1e-12, not a component of the solution, so the
    // meaningful statement is that both runs are converged and that the two residuals agree to
    // round-off OF THAT RESIDUAL (measured 1.8e-18 absolute), not to 1e-13 of themselves.
    const double rd = std::fabs(a.div - b.div);
    std::printf("  sphere Stokes (%s): u %.3e  v %.3e  p %.3e  div %.3e vs %.3e (d %.3e)\n",
                scene ? "analytic scene" : "sampled SDF", ru, rv, rp, a.div, b.div, rd);
    CHECK(maxAbs(a.u) > 0.0);
    CHECK(ru <= TOL);
    CHECK(rv <= TOL);
    CHECK(rp <= TOL);
    CHECK(a.div < 1e-9 && b.div < 1e-9);  // both converged
    CHECK(rd <= 1e-15);                   // and the residuals agree to their own round-off
  }
}
}  // namespace

int main(int argc, char** argv) {
  const std::string gate = argc > 1 ? argv[1] : "identity";
  Kokkos::initialize(argc, argv);
  {
    if (gate == "identity")
      gateIdentity();
    else if (gate == "scale")
      gateScale();
    else if (gate == "vof")
      gateVofSigma();
    else {
      std::fprintf(stderr, "usage: test_units [identity|scale|vof]\n");
      ++failures;
    }
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
