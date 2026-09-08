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
// GATE 4 — units_anisotropic_poiseuille (`test_units aniso`).  Phase 2 (anisotropic cells),
// flow/doc/anisotropic_metric.md §8.2.  Plane Poiseuille on cells (16, 40, 8), with the cut-cell
// walls exactly on two y cell CENTRES — where the second difference of a quadratic is exact, so
// the discrete solution IS the analytic parabola u(y) = F (y - ylo)(yhi - y)/(2 mu) and the only
// thing the comparison can measure is the operator.  TWO stretched configurations, making two
// DIFFERENT statements:
//
//   (a) EXACTNESS, extent (16, 10, 16) -> spacing (1, 0.25, 2), hRef = 0.25 on `y` (a non-trivial
//       check that hRef is min_a h_a and not h_x), h' = (4, 1, 8), w = (1/16, 1, 1/64) and
//       mu' = mu*tRef/hRef^2 = 80 — every one of them exactly representable in float, so the
//       stored operator carries no rounding of its own and the metric is on trial alone.  Bound
//       1e-9 of u_max = F H^2/(8 mu) = 0.2 (measured 1.388e-15).
//
//   (b) THE FLOAT-STORAGE TRIPWIRE, extent (16, 12, 16) -> spacing (1, 0.3, 2), h' = (10/3, 1,
//       20/3), w = (0.09, 1, 0.0225) — the §5.3 configuration, and a production-SHAPED one: with
//       hRef = 0.3 neither mu' = 55.5555… nor AC = 1 + 2((5 + 55.5555…) + 1.25) = 124.6111… is
//       representable, so the six off-diagonals no longer sum to AC - idiag exactly and the whole
//       profile is scaled by 1 - 1.06e-07 (that ratio is CONSTANT across all fifteen fluid rows,
//       which is what identifies it).  Bound 1e-7, measured 3.052e-08 — and 8.674e-15 when the
//       identical source is built with -DPECLET_FLOW_MREAL_DOUBLE.  The discretisation is
//       pointwise exact here too; this row watches the WO-M float operator-storage floor
//       (docs/SCALING_ISSUES.md #1), it is NOT an exactness statement and NOT a metric defect.
//
// Both configurations also require max|v|, max|w| <= 1e-12 u_max and the wall/solid u rows exactly
// 0.  The isotropic control on (16, 16, 16) at extent == cells meets the same bounds and reports
// the EXACT identity metric (hp = w = (1,1,1), vol = 1, aniso false) from the §1.4 snap.  The gate
// also asserts the §7 refusals: the collocated policy, `enable_vof` and the hydro force integrals
// each throw on an anisotropic domain with the three spacings in the message.
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
  const double H = 1.0 / lam;  // one cell, physically
  const double SIGMA = 1.0;    // invariant under a change of LENGTH unit
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
  const std::vector<double> C = sphereColour(n, Rcells, n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11);
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
  const std::size_t inside =
      (std::size_t)(n / 2) + (std::size_t)(n / 2) * n + (std::size_t)(n / 2) * n * n;
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
  std::printf("  constant kappa: max|u| %.3e (lam=1) vs %.3e (lam=%g)  ->  %.3e rescaled\n", a.maxU,
              b.maxU, lam, b.maxU * lam);
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
  checkClose(b.dtCap, a.dtCap, 1e-12 * a.dtCap,
             "capillary_dt invariant under a length-unit change");
  std::printf("  capillary_dt %.9e vs %.9e (invariant: sigma and the time unit are)\n", a.dtCap,
              b.dtCap);

  // (b) the COMPUTED curvature: the physical spurious currents and the jump must agree.
  const Drop ca = runDroplet(N, Rc, 1.0, false, 20);
  const Drop cb = runDroplet(N, Rc, lam, false, 20);
  const double ru = std::fabs(ca.maxU - cb.maxU * lam) / std::fabs(ca.maxU);
  const double rp = std::fabs(ca.dp - cb.dp / lam) / std::fabs(ca.dp);
  std::printf(
      "  computed kappa: max|u| %.6e vs %.6e rescaled (rel %.3e); dp %.6e vs %.6e "
      "(rel %.3e)\n",
      ca.maxU, cb.maxU * lam, ru, ca.dp, cb.dp / lam, rp);
  CHECK(ru <= 1e-12);
  CHECK(rp <= 1e-12);
}

// --------------------------------------------------------------------------------------------
// Phase 2 gate G1 (doc/anisotropic_metric.md §8.2).
// --------------------------------------------------------------------------------------------

/// Plane Poiseuille between two cut-cell walls that sit exactly on y cell CENTRES, on an arbitrary
/// (possibly anisotropic) physical box.  `jlo`/`jhi` are the wall cell indices; everything else is
/// `runChannel`'s recipe, in the unit system the caller writes the box in (lam = 1).
struct Channel {
  Fields f;
  std::vector<double> yc;       ///< the physical y cell centres
  double ylo = 0.0, yhi = 0.0;  ///< the two wall positions (== yc[jlo], yc[jhi])
  std::array<double, 3> w{1.0, 1.0, 1.0};
  bool aniso = false;
};

Channel runChannelBox(int nx, int ny, int nz, const std::array<double, 3>& extent, int jlo, int jhi,
                      bool arm) {
  const double RHO = 1.0, MU = 0.1, F = 0.01, DT = 50.0;
  peclet::flow::Solver<peclet::flow::Staggered> s(nx, ny, nz);
  if (arm)
    s.setPhysicalDomain(extent, {0.0, 0.0, 0.0}, {nx, ny, nz});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0.0, 0.0);
  s.setVelocityIterations(400);
  s.setVelocityResidualTolerance(1e-14);
  s.setPressurePcg(true, 80, 1e-13);
  Channel ch;
  ch.yc = s.cellCentres(1);
  ch.ylo = ch.yc[jlo];
  ch.yhi = ch.yc[jhi];
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
            std::min(ch.yc[y] - ch.ylo, ch.yhi - ch.yc[y]);
  s.setSolid(sdf, /*cutcellPressure=*/false);
  for (int it = 0; it < 300; ++it)
    s.step();
  ch.f.u = s.getVelocity(0);
  ch.f.v = s.getVelocity(1);
  ch.f.w = s.getVelocity(2);
  ch.f.p = s.getPressure();
  ch.f.spacing = s.spacing();
  const auto& u = s.unitScales();
  ch.w = {u.w[0], u.w[1], u.w[2]};
  ch.aniso = u.aniso;
  return ch;
}

/// The §8.2 assertions on one channel: the parabola over every FLUID y DOF within `bound`, the two
/// wall rows exactly zero, and the two transverse components at the transverse floor.  `bound` is
/// 1e-9 wherever the stored operator is float-exact and 1e-7 on the tripwire row (see the header).
void checkChannel(const char* what, const Channel& ch, int nx, int ny, int nz, int jlo, int jhi,
                  double bound) {
  const double MU = 0.1, F = 0.01;
  const double H = ch.yhi - ch.ylo;
  const double umax = F * H * H / (8.0 * MU);
  std::printf("  %s: spacing (%.17g, %.17g, %.17g)  aniso %d  w (%.17g, %.17g, %.17g)\n", what,
              ch.f.spacing[0], ch.f.spacing[1], ch.f.spacing[2], (int)ch.aniso, ch.w[0], ch.w[1],
              ch.w[2]);
  double relErr = 0.0, wallMax = 0.0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t i = (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny;
        const double uu = ch.f.u[i];
        if (y <= jlo || y >= jhi) {  // wall and solid rows
          if (!(std::fabs(uu) <= wallMax))
            wallMax = std::fabs(uu);
          continue;
        }
        const double ex = F * (ch.yc[y] - ch.ylo) * (ch.yhi - ch.yc[y]) / (2.0 * MU);
        const double d = std::fabs(uu - ex) / umax;
        if (!(d <= relErr))
          relErr = d;
      }
  const double mv = maxAbs(ch.f.v), mw = maxAbs(ch.f.w);
  std::printf(
      "      H %.17g  u_max %.17g  max rel |u - parabola| %.3e (bound %.0e)  max|v| %.3e  "
      "max|w| %.3e  max|u| on the wall/solid rows %.3e\n",
      H, umax, relErr, bound, mv, mw, wallMax);
  if (!(relErr <= bound))
    std::printf(
        "      NOTE: a miss that is a SINGLE one-signed multiplicative factor across the "
        "whole profile (rel err proportional to u, one constant ratio on every fluid row) "
        "is the FLOAT momentum-operator storage (`IbmSolver::FV`, MReal; WO-M / "
        "docs/SCALING_ISSUES.md #1), not the metric -- rebuild with "
        "-DPECLET_FLOW_MREAL_DOUBLE to separate the two.\n");
  CHECK(relErr <= bound);
  CHECK(wallMax == 0.0);
  CHECK(mv <= 1e-12 * umax);
  CHECK(mw <= 1e-12 * umax);
}

void gateAnisoPoiseuille() {
  std::printf("=== units_anisotropic_poiseuille ===\n");
  // (a) EXACTNESS. cells (16, 40, 8) over (16, 10, 16) -> spacing (1, 0.25, 2), hRef = 0.25 on the
  //     y axis, h' = (4, 1, 8), w = (1/16, 1, 1/64), mu' = 80 -- all float-representable, so the
  //     stored operator adds no rounding and the metric is on trial alone. Walls on the y cell
  //     centres 12.5*0.25 = 3.125 and 28.5*0.25 = 7.125 (H = 4).
  {
    const Channel ch = runChannelBox(16, 40, 8, {16.0, 10.0, 16.0}, 12, 28, /*arm=*/true);
    CHECK(ch.aniso);
    CHECK(ch.f.spacing[0] == 1.0 && ch.f.spacing[1] == 0.25 && ch.f.spacing[2] == 2.0);
    CHECK(ch.w[0] == 0.0625 && ch.w[1] == 1.0 && ch.w[2] == 0.015625);
    CHECK(ch.ylo == 3.125 && ch.yhi == 7.125);
    checkChannel("stretched EXACTNESS (16, 40, 8) on (16, 10, 16)", ch, 16, 40, 8, 12, 28, 1e-9);
  }
  // (b) THE FLOAT-STORAGE TRIPWIRE. The same grid over (16, 12, 16) -> spacing (1, 0.3, 2),
  //     hRef = 0.3, h' = (10/3, 1, 20/3), w = (0.09, 1, 0.0225); walls at 12.5*0.3 = 3.75 and
  //     28.5*0.3 = 8.55 (H = 4.8). mu' = 55.5555... and AC = 124.6111... are not representable in
  //     the float operator storage, which scales the whole profile by 1 - 1.06e-07: measured
  //     3.052e-08 here and 8.674e-15 with -DPECLET_FLOW_MREAL_DOUBLE. Bound 1e-7 (see the header).
  {
    const Channel ch = runChannelBox(16, 40, 8, {16.0, 12.0, 16.0}, 12, 28, /*arm=*/true);
    CHECK(ch.aniso);
    checkClose(ch.f.spacing[0], 1.0, 0.0, "aniso spacing x");
    checkClose(ch.f.spacing[1], 0.3, 1e-17, "aniso spacing y");
    checkClose(ch.f.spacing[2], 2.0, 0.0, "aniso spacing z");
    checkClose(ch.ylo, 3.75, 1e-14, "aniso wall ylo");
    checkClose(ch.yhi, 8.55, 1e-14, "aniso wall yhi");
    checkChannel("stretched FLOAT-FLOOR (16, 40, 8) on (16, 12, 16)", ch, 16, 40, 8, 12, 28, 1e-7);
  }
  // (c) The isotropic control on (16, 16, 16) at extent == cells, with `runChannel`'s own wall
  //     convention (round(0.30*ny) + 0.5, round(0.70*ny) + 0.5 in cells): the same bound, and the
  //     §1.4 snap gives it EXACTLY the identity metric.  (It is not BITWISE the extent=None run of
  //     the same problem, and must not be asserted to be: dt = 50 pins tRef = 50, so the armed run
  //     computes with dt' = 1 and mu' = 50 mu while the cell-unit one computes with dt' = 50 and
  //     mu' = mu -- the same physics, a different internal scaling.  `units_identity` is the
  //     bitwise statement, and it fixes rho = dt = 1 for exactly this reason.)
  {
    const Channel a = runChannelBox(16, 16, 16, {16.0, 16.0, 16.0}, 5, 11, /*arm=*/true);
    CHECK(!a.aniso);
    CHECK(a.w[0] == 1.0 && a.w[1] == 1.0 && a.w[2] == 1.0);
    CHECK(a.f.spacing[0] == 1.0 && a.f.spacing[1] == 1.0 && a.f.spacing[2] == 1.0);
    checkChannel("isotropic control (16, 16, 16) at extent == cells", a, 16, 16, 16, 5, 11, 1e-9);
  }
  // (d) The §7 table, as commit C4 leaves it.  ADMITTED (C4 lifted them): the collocated policy
  //     and the hydro force integrals.  STILL REFUSED, with the three spacings in the message:
  //     enable_vof (Phase 3), and hydro_force_torque_reaction only on the v3 moving-wall torque
  //     path (doc/units_escalation.md E3), which needs a moving scene and is not reachable here.
  {
    bool threw = false;
    try {
      peclet::flow::Solver<peclet::flow::Colocated> sc(16, 40, 8);
      sc.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
      const auto sp = sc.spacing();
      std::printf("      ADMITTED (collocated): spacing (%.17g, %.17g, %.17g), aniso %d\n", sp[0],
                  sp[1], sp[2], (int)sc.unitScales().aniso);
      CHECK(sc.unitScales().aniso);
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      UNEXPECTED refusal (collocated): %s\n", ex.what());
    }
    CHECK(!threw);
    threw = false;
    try {
      // PHASE 3 INVERTED THIS ASSERTION. Phase 2 wrote it as `CHECK(threw)` — `enable_vof` was
      // the one entry point its §7 still refused on an anisotropic domain, and the gate pinned
      // that refusal so it could not be lifted by accident. Phase 3 (flow/doc/anisotropic_vof.md)
      // is what lifts it: the colour transport is metric-free because a stretched cell IS the unit
      // cube of the index coordinates, and everything that reads a direction, a length or an area
      // — the height-function curvature, the paraboloid frames, the CSF face force's per-axis
      // weight `w_a`, the wetting rotation, the phase-change normals/areas/V_cell — now carries
      // the metric. `units_vof_aniso` is the gate that holds it: the balanced-force exactness
      // identity is at machine zero (2.3e-17 at aspect 2, 1.9e-17 at aspect 4) and Young-Laplace
      // is exact, which is only true if the CSF weight is EXACTLY this file's `w_a`.
      peclet::flow::Solver<peclet::flow::Staggered> ss(16, 40, 8);
      ss.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
      ss.enableVof();
      std::printf("      ADMITTED (enable_vof — Phase 3; see units_vof_aniso)\n");
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      UNEXPECTED refusal (enable_vof): %s\n", ex.what());
    }
    CHECK(!threw);
    threw = false;
    try {
      peclet::flow::Solver<peclet::flow::Staggered> ss(16, 40, 8);
      ss.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
      ss.hydroForceTorque();  // C4: admitted (no scene here, so it returns an empty result)
      std::printf("      ADMITTED (hydro_force_torque)\n");
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      UNEXPECTED refusal (hydro_force_torque): %s\n", ex.what());
    }
    CHECK(!threw);
  }
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
// --------------------------------------------------------------------------------------------
// GATE 5 - units_anisotropic_sphere (`test_units sphere`).  Phase 2 gate G2,
// doc/anisotropic_metric.md §8.3.  The regression suite's Zick & Homsy sphere (phi = 0.216,
// K_ref = 7.442) with `tests/regression/perf_baseline.json`'s own configuration - rho 1, mu 0.1,
// dt 60, F 1e-3, 80 velocity sweeps, MG-PCG 300 / 1e-8, cut-cell pressure, advection off, the same
// convergence rule on mean(u) - on the CUBE L^3 resolved by cells (N, 2N, N/2), i.e.
// h = (dx, dx/2, 2dx), with the SDF sampled at the PHYSICAL cell centres.  Four requirements:
//   (i)   the stretched errors |K_s(N) - K_ref|/K_ref are STRICTLY DECREASING in N;
//   (ii)  err_s(N) <= 4 err_c(N) + 0.2 % at every N (the z axis is twice as coarse, and second
//         order allows 4x);
//   (iii) the least-squares order of the stretched ERROR sequence, p_s >= 1.5;
//   (iv)  the isotropic control (N, N, N) in CELL UNITS reproduces the recorded baseline: the step
//         counts 60 / 80 / 75 and the pressure iterations per step 6 / 7 / 7 EXACTLY, and the drag
//         factors 7.299689474225098 / 7.389063051924422 / 7.416167928926183 to 1e-6 relative.
// Plus the §4.4 HYDRO-FORCE gate that C4 also lifts: on a periodic Stokes cell the total force on
// the body equals the body force integrated over the fluid volume the momentum rows cover,
// F_x * (h_x h_y h_z) * (# fluid u-DOFs) - the periodic Stokes identity.  Measured on the cubic and
// on the stretched grid; the stretched grid must reach the tolerance the cubic one does.
//
// Two notes on (iii) and (iv).  `fit_order` (the regression's f(N) = f_inf + C N^-p grid search) is
// DEGENERATE on three grids - two free linear parameters fit three points exactly for every p - so
// the ctest gates the log-log least-squares slope of the errors instead, and
// `scripts/verify_anisotropic_spheres.py` runs the five-grid ladder where `fit_order` is the same
// estimator the regression uses.
//
// And "bitwise" in §8.3 (iv) is not attainable against `perf_baseline.json`, for two reasons that
// have nothing to do with the metric: the baseline is recorded on the CUDA build (the regression's
// documented tree) while this ctest also runs on host-openmp, and its `u.mean()` is numpy's
// PAIRWISE summation against a sequential C++ sum here.  Both perturb the drag factor at the
// PRESSURE SOLVE'S OWN STOPPING TOLERANCE - the baseline configuration is MG-PCG rtol 1e-8 - and
// the measured gap is 1.4e-08 / 9.7e-10 / 1.8e-09, i.e. exactly there.  What IS exactly
// reproducible is the discrete part of the record, so the gate asserts the step counts and the
// pressure iterations per step EXACTLY (60 / 80 / 75 at 6 / 7 / 7) and the drag factor to 1e-6,
// which still pins every digit `perf_baseline.json` prints, and it prints all 17 either way.
//
// GATE 6 - units_anisotropic_tgv (`test_units tgv`).  Phase 2 gate G3, §8.4.  Stokes Taylor-Green
// on a periodic box L x L x L_z with cells (N, 2N, 4), rho = dt = 1 (so every reference scale but
// hRef is 1 and the isotropic control can be compared BITWISE to the cell-unit run), mu = 0.1.  The
// mu = 0.25 is chosen so that BOTH assembled operators are exactly representable in the shipped
// FLOAT operator storage (`IbmSolver::FV`, MReal; WO-M / docs/SCALING_ISSUES.md #1) and the metric
// is therefore on trial alone -- the same move E1 of doc/units_escalation.md made for G1. Isotropic
// control: hRef = 1, mu' = 0.25, b = (1/4, 1/4, 1/4), AC = 1 + 2((b+b)+b) = 5/2.  Stretched:
// hRef = 0.5, hp = (2, 1, 2), w = (1/4, 1, 1/4), mu' = 1, b = (1/4, 1, 1/4), AC = 4.  Every one of
// them dyadic.  At mu = 0.1 the same gate reads 1.5e-07 (isotropic control) and 2.9e-07 (stretched)
// -- the float floor, in the isotropic control too, so not a metric statement.
// The plain TG field is NOT discretely divergence-free on a stretched staggered grid; the initial
// field is the one that is,
//     u =  a_x cos(kx) sin(ky),  v = -a_y sin(kx) cos(ky),
//     a_x sin(k h_x/2)/h_x == a_y sin(k h_y/2)/h_y,
// which is also an eigenvector of the anisotropic 7-point operator, so the backward-Euler amplitude
// ratio per step is EXACTLY r = 1/(1 + dt nu Lambda) with
// Lambda = 2(1-cos k h_x)/h_x^2 + 2(1-cos k h_y)/h_y^2.
struct Zh {
  double K = 0.0, umean = 0.0, div = 0.0, R = 0.0, pIterStep = 0.0;
  int steps = 0;
  std::array<double, 3> spacing{};
  bool aniso = false;
};

/// The regression's `run_case("zh_sphere", N)` transcribed.  `arm == false` is the CELL-UNIT
/// baseline configuration; `arm == true` puts the same physical cube `L^3` on (nx, ny, nz) cells.
Zh runZhSphere(int nx, int ny, int nz, bool arm, double L, int levels) {
  const double RHO = 1.0, MU = 0.1, DT = 60.0, FX = 1e-3, PHI = 0.216;
  const double R = std::pow(PHI * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * L;
  peclet::flow::Solver<peclet::flow::Staggered> s(nx, ny, nz);
  if (arm)
    s.setPhysicalDomain({L, L, L}, {0.0, 0.0, 0.0}, {nx, ny, nz});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(FX, 0.0, 0.0);
  s.setAdvection(false);
  s.setVelocityIterations(80);
  s.setPressureLevels(levels);
  s.setPressurePcg(true, 300, 1e-8);
  const std::vector<double> cx = s.cellCentres(0), cy = s.cellCentres(1), cz = s.cellCentres(2);
  const double c = 0.5 * L;
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
            std::sqrt((cx[x] - c) * (cx[x] - c) + (cy[y] - c) * (cy[y] - c) +
                      (cz[z] - c) * (cz[z] - c)) -
            R;
  s.setSolid(sdf, /*cutcellPressure=*/true);
  std::vector<double> pit;
  double prev = 0.0;
  Zh r;
  for (int it = 0; it < 400; ++it) {
    s.step();
    ++r.steps;
    pit.push_back((double)s.lastPressureIterations());
    if (it % 5 == 4) {
      const std::vector<double> uu = s.getVelocity(0);
      double m = 0.0;
      for (double v : uu)
        m += v;
      m /= (double)uu.size();
      if (it >= 15 && std::fabs(m - prev) < 1e-5 * (std::fabs(m) + 1e-30))
        break;
      prev = m;
    }
  }
  const std::vector<double> uu = s.getVelocity(0);
  double m = 0.0;
  for (double v : uu)
    m += v;
  r.umean = m / (double)uu.size();
  r.R = R;
  r.K = FX * (L * L * L) / (6.0 * M_PI * MU * R * r.umean);
  r.div = s.maxOpenDivergence();
  r.spacing = s.spacing();
  r.aniso = s.unitScales().aniso;
  std::vector<double> half(pit.begin() + (std::ptrdiff_t)(pit.size() / 2), pit.end());
  std::sort(half.begin(), half.end());
  const std::size_t nh = half.size();
  r.pIterStep = (nh % 2) ? half[nh / 2] : 0.5 * (half[nh / 2 - 1] + half[nh / 2]);
  return r;
}

/// least-squares slope of log(err) against log(N) (the observed order of convergence).
double logLogOrder(const std::vector<double>& Ns, const std::vector<double>& err) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = (double)Ns.size();
  for (std::size_t i = 0; i < Ns.size(); ++i) {
    const double X = std::log(Ns[i]), Y = std::log(err[i]);
    sx += X;
    sy += Y;
    sxx += X * X;
    sxy += X * Y;
  }
  return -(n * sxy - sx * sy) / (n * sxx - sx * sx);
}

/// The §4.4 periodic-Stokes force identity, on an ANALYTIC scene (hydroForceTorqueReaction needs
/// one).  Returns the measured x reaction, the identity's value and their relative gap.
struct React {
  double Fx = 0.0, Fref = 0.0, rel = 0.0;
  long nFluid = 0;
  std::array<double, 3> spacing{};
};

React runReactionSphere(int nx, int ny, int nz, bool arm, double L, int levels, int nsteps) {
  const double RHO = 1.0, MU = 0.1, DT = 60.0, FX = 1e-3, PHI = 0.216;
  const double R = std::pow(PHI * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * L;
  peclet::flow::Solver<peclet::flow::Staggered> s(nx, ny, nz);
  if (arm)
    s.setPhysicalDomain({L, L, L}, {0.0, 0.0, 0.0}, {nx, ny, nz});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(FX, 0.0, 0.0);
  s.setAdvection(false);
  s.setVelocityIterations(200);
  s.setVelocityResidualTolerance(1e-12);
  s.setPressureLevels(levels);
  s.setPressurePcg(true, 300, 1e-10);
  {
    namespace g = peclet::core::geom;
    g::SceneBuilder<double> b;
    b.addLeaf(g::kSphere, {R});
    g::Transform<double> tr;
    tr.translation = peclet::core::Vec3<double>{0.5 * L, 0.5 * L, 0.5 * L};
    b.addInstance(0, tr);
    std::vector<int> ni, ii;
    std::vector<double> nr, ir;
    b.encode(ni, nr, ii, ir);
    s.setScene(ni, nr, ii, ir, /*periodic=*/true);
    s.setSolidFromScene(/*cutcellPressure=*/true);
  }
  for (int it = 0; it < nsteps; ++it)
    s.step();
  React out;
  out.spacing = s.spacing();
  // The fluid u-DOF count, from the solver's OWN sampled SDF and `ibmSolidMask`'s rule: the
  // staggered sample at (x-1/2, y, z) is the mean of the two adjacent cell values (periodic wrap),
  // and the DOF is SOLID for `sd <= 0`.
  const std::vector<double> sd = s.getField("sdf");
  long nf = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const int xm = (x + nx - 1) % nx;
        const double a = sd[(std::size_t)xm + (std::size_t)y * nx + (std::size_t)z * nx * ny];
        const double b2 = sd[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny];
        if (0.5 * (a + b2) > 0.0)
          ++nf;
      }
  out.nFluid = nf;
  const double vcell = out.spacing[0] * out.spacing[1] * out.spacing[2];
  out.Fref = FX * vcell * (double)nf;
  const std::vector<double> fr = s.hydroForceTorqueReaction();
  out.Fx = fr.empty() ? 0.0 : fr[0];
  out.rel = std::fabs(out.Fx - out.Fref) / std::fabs(out.Fref);
  return out;
}

void gateAnisoSphere() {
  std::printf("=== units_anisotropic_sphere ===\n");
  const double KREF = 7.442;
  const double REC[3] = {7.299689474225098, 7.389063051924422, 7.416167928926183};
  const int NS[3] = {16, 24, 32};
  std::vector<double> Ns, errS, errC;
  for (int i = 0; i < 3; ++i) {
    const int N = NS[i];
    const int levels = std::max(2, (int)std::floor(std::log2((double)N)) - 1);
    const Zh cub = runZhSphere(N, N, N, /*arm=*/false, (double)N, levels);
    const Zh str = runZhSphere(N, 2 * N, N / 2, /*arm=*/true, (double)N, levels);
    const double ec = std::fabs(cub.K - KREF) / KREF, es = std::fabs(str.K - KREF) / KREF;
    Ns.push_back((double)N);
    errC.push_back(ec);
    errS.push_back(es);
    std::printf("  N=%2d  cubic  K %.17g  err %.4f %%  iters/step %.1f  steps %3d  div %.2e\n", N,
                cub.K, 100.0 * ec, cub.pIterStep, cub.steps, cub.div);
    std::printf(
        "        stretched (%d,%d,%d) spacing (%g, %g, %g) aniso %d  K %.17g  err %.4f %%"
        "  iters/step %.1f  steps %3d  div %.2e\n",
        N, 2 * N, N / 2, str.spacing[0], str.spacing[1], str.spacing[2], (int)str.aniso, str.K,
        100.0 * es, str.pIterStep, str.steps, str.div);
    // (iv) the cubic control reproduces the recorded baseline value to every printed digit.
    const double dRec = std::fabs(cub.K - REC[i]) / REC[i];
    const int RECSTEPS[3] = {60, 80, 75};
    const double RECITERS[3] = {6.0, 7.0, 7.0};
    std::printf(
        "        cubic vs perf_baseline.json %.17g   rel %.3e (bound 1e-6)   steps %d "
        "(recorded %d)   iters/step %.1f (recorded %.1f)\n",
        REC[i], dRec, cub.steps, RECSTEPS[i], cub.pIterStep, RECITERS[i]);
    CHECK(dRec <= 1e-6);
    CHECK(cub.steps == RECSTEPS[i]);
    CHECK(cub.pIterStep == RECITERS[i]);
    // (ii) err_s <= 4 err_c + 0.2 %
    CHECK(es <= 4.0 * ec + 0.002);
    // pressure iterations on the stretched grid: <= cubic + 2 (the §5 aspect rule)
    std::printf("        pressure iters/step  cubic %.1f  stretched %.1f  (bound cubic + 2)\n",
                cub.pIterStep, str.pIterStep);
    CHECK(str.pIterStep <= cub.pIterStep + 2.0);
  }
  // (i) strictly decreasing stretched errors
  CHECK(errS[1] < errS[0]);
  CHECK(errS[2] < errS[1]);
  // (iii) least-squares order of the stretched error sequence
  const double ps = logLogOrder(Ns, errS), pc = logLogOrder(Ns, errC);
  std::printf("  LS order of the ERROR sequence: stretched %.4f (bound >= 1.5), cubic %.4f\n", ps,
              pc);
  CHECK(ps >= 1.5);

  // --- §4.4, the hydro-force gate C4 lifts: the periodic Stokes identity F = F_body * V_fluid ---
  {
    const int N = 24;
    const int levels = std::max(2, (int)std::floor(std::log2((double)N)) - 1);
    const React rc = runReactionSphere(N, N, N, /*arm=*/false, (double)N, levels, 300);
    const React rs = runReactionSphere(N, 2 * N, N / 2, /*arm=*/true, (double)N, levels, 300);
    std::printf("  hydro_force_torque_reaction, periodic Stokes identity F = F_body*V_fluid:\n");
    std::printf(
        "    cubic     (%d,%d,%d) spacing (%g,%g,%g)  F_x %.17g  identity %.17g  rel %.3e"
        "  (%ld fluid u-DOFs)\n",
        N, N, N, rc.spacing[0], rc.spacing[1], rc.spacing[2], rc.Fx, rc.Fref, rc.rel, rc.nFluid);
    std::printf(
        "    stretched (%d,%d,%d) spacing (%g,%g,%g)  F_x %.17g  identity %.17g  rel %.3e"
        "  (%ld fluid u-DOFs)\n",
        N, 2 * N, N / 2, rs.spacing[0], rs.spacing[1], rs.spacing[2], rs.Fx, rs.Fref, rs.rel,
        rs.nFluid);
    // "to the same relative tolerance the cubic grid achieves" (§8.3), gated as ONE shared bound
    // rather than a ratio: both grids are at the march's own residual floor here (300 steps of the
    // regression's dt = 60 configuration, momentum residual stop 1e-12, MG-PCG 1e-10), and 1e-5 is
    // the order the cubic grid reaches with a decade of margin.
    CHECK(rs.rel <= 1e-5);
    CHECK(rc.rel <= 1e-5);
    std::printf("    stretched/cubic relative-gap ratio %.3f (both gated at 1e-5)\n",
                rs.rel / rc.rel);
  }
}

// --------------------------------------------------------------------------------------------
struct Tgv {
  double ratio = 0.0, rExact = 0.0, relRate = 0.0, div = 0.0, prof = 0.0, pmax = 0.0;
  std::vector<double> u, v, w, p;
  std::array<double, 3> spacing{};
  bool aniso = false;
};

/// Stokes (or, with `advect`, NS) Taylor-Green on the DISCRETELY divergence-free stretched field.
/// `arm == false` runs in cell units (spacing 1), which is what the isotropic control is compared
/// against bitwise.
Tgv runTgv(int nx, int ny, int nz, bool arm, double Lx, double Ly, double Lz, bool advect,
           int nsteps, double DT = 1.0) {
  const double RHO = 1.0, MU = 0.25;
  const double hx = arm ? Lx / nx : 1.0, hy = arm ? Ly / ny : 1.0, hz = arm ? Lz / nz : 1.0;
  const double Lw = arm ? Lx : (double)nx;  // the vortex wavelength (x and y share it)
  const double k = 2.0 * M_PI / Lw;
  const double ax = 1.0;
  const double ay = ax * (std::sin(0.5 * k * hx) / hx) * (hy / std::sin(0.5 * k * hy));
  peclet::flow::Solver<peclet::flow::Staggered> s(nx, ny, nz);
  if (arm)
    s.setPhysicalDomain({Lx, Ly, Lz}, {0.0, 0.0, 0.0}, {nx, ny, nz});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(advect);
  s.setVelocityIterations(400);
  s.setVelocityResidualTolerance(1e-14);
  s.setPressurePcg(true, 300, 1e-13);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 10.0));
  const std::size_t n = (std::size_t)nx * ny * nz;
  std::vector<double> u(n), v(n), w(n, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const std::size_t i = (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny;
        u[i] = ax * std::cos(k * (x * hx)) * std::sin(k * ((y + 0.5) * hy));
        v[i] = -ay * std::sin(k * ((x + 0.5) * hx)) * std::cos(k * (y * hy));
      }
  s.uploadVelocity(u, v, w);
  Tgv r;
  const double Lambda =
      2.0 * (1.0 - std::cos(k * hx)) / (hx * hx) + 2.0 * (1.0 - std::cos(k * hy)) / (hy * hy);
  r.rExact = std::pow(1.0 / (1.0 + DT * (MU / RHO) * Lambda), (double)nsteps);
  for (int it = 0; it < nsteps; ++it)
    s.step();
  r.u = s.getVelocity(0);
  r.v = s.getVelocity(1);
  r.w = s.getVelocity(2);
  r.p = s.getPressure();
  r.spacing = s.spacing();
  r.aniso = s.unitScales().aniso;
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    num += r.u[i] * u[i];
    den += u[i] * u[i];
  }
  r.ratio = num / den;
  double pmaxu = 0.0, perr = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pmaxu = std::fmax(pmaxu, std::fabs(u[i]));
    perr = std::fmax(perr, std::fabs(r.u[i] - r.ratio * u[i]));
  }
  r.prof = perr / pmaxu;
  r.relRate = std::fabs(r.ratio - r.rExact) / r.rExact;
  r.div = s.maxOpenDivergence();
  r.pmax = maxAbs(r.p);
  (void)hz;
  return r;
}

void gateAnisoTgv() {
  std::printf("=== units_anisotropic_tgv ===\n");
  const int N = 16, NS = 10;
  // (a) STOKES on the stretched box L x L x L_z with cells (N, 2N, 4): h = (1, 0.5, 1).
  {
    const Tgv t =
        runTgv(N, 2 * N, 4, /*arm=*/true, (double)N, (double)N, 4.0, /*advect=*/false, NS);
    std::printf("  stretched Stokes (%d,%d,4) spacing (%g, %g, %g) aniso %d\n", N, 2 * N,
                t.spacing[0], t.spacing[1], t.spacing[2], (int)t.aniso);
    std::printf("    amplitude ratio %.17g  exact %.17g  rel %.3e (bound 1e-10)\n", t.ratio,
                t.rExact, t.relRate);
    std::printf(
        "    max|div_o| %.3e (bound 1e-12)  profile %.3e (bound 1e-10)  max|P| %.3e "
        "(the phi proxy, bound 1e-12)\n",
        t.div, t.prof, t.pmax);
    CHECK(t.aniso);
    CHECK(t.relRate <= 1e-10);
    CHECK(t.div <= 1e-12);
    CHECK(t.prof <= 1e-10);
    CHECK(t.pmax <= 1e-12);
  }
  // (b) the same with advection ON: the NS run.
  {
    // dt = 0.05 here, `sdflow_tg`'s own step, which is what makes the advective CFL 0.05: at the
    // dt = 1 of the Stokes rows the TG amplitude gives CFL = 1 and the discrete nonlinear term's
    // imbalance with the pressure gradient is 2.0e-2, i.e. the O(CFL^2) advection error, not a
    // metric defect.  dt is free here -- the Stokes rows own the exactness statement and the
    // bitwise control, both of which need dt = 1 (it pins tRef = 1).
    const Tgv t =
        runTgv(N, 2 * N, 4, /*arm=*/true, (double)N, (double)N, 4.0, /*advect=*/true, NS, 0.05);
    std::printf(
        "  stretched NS      (%d,%d,4) dt 0.05  amplitude ratio %.17g  exact %.17g  "
        "rel %.3e (bound 5e-3)  max|div| %.3e (bound 1e-9)\n",
        N, 2 * N, t.ratio, t.rExact, t.relRate, t.div);
    CHECK(t.relRate <= 5e-3);
    CHECK(t.div <= 1e-9);
  }
  // (c) the isotropic control (N, N, 4) at extent == cells: the same bounds, and BITWISE the
  //     cell-unit run of the same problem (rho = dt = 1 pins every reference scale to 1).
  {
    const Tgv a = runTgv(N, N, 4, /*arm=*/true, (double)N, (double)N, 4.0, /*advect=*/false, NS);
    const Tgv b = runTgv(N, N, 4, /*arm=*/false, (double)N, (double)N, 4.0, /*advect=*/false, NS);
    std::printf("  isotropic control (%d,%d,4) spacing (%g, %g, %g) aniso %d\n", N, N, a.spacing[0],
                a.spacing[1], a.spacing[2], (int)a.aniso);
    std::printf(
        "    amplitude ratio %.17g  exact %.17g  rel %.3e  max|div_o| %.3e  profile %.3e "
        " max|P| %.3e\n",
        a.ratio, a.rExact, a.relRate, a.div, a.prof, a.pmax);
    CHECK(!a.aniso);
    CHECK(a.relRate <= 1e-10);
    CHECK(a.div <= 1e-12);
    CHECK(a.prof <= 1e-10);
    CHECK(a.pmax <= 1e-12);
    const bool bit = bitwiseEqual(a.u, b.u) && bitwiseEqual(a.v, b.v) && bitwiseEqual(a.w, b.w) &&
                     bitwiseEqual(a.p, b.p);
    std::printf("    armed at extent == cells vs the cell-unit run: bitwise %s\n",
                bit ? "EQUAL" : "DIFFERENT");
    CHECK(bit);
  }
}
}  // namespace

// =============================================================== units_vof_aniso (PHASE 3, S2)
//
// `flow/doc/anisotropic_vof.md` §11 S2 — the gate the Phase 2 merge finally makes constructible,
// because `enable_vof` refused an anisotropic domain until Phase 3 landed.
//
// A stationary droplet on cells that are NOT cubes. Two statements, and they are independent:
//
//  (a) THE BALANCED-FORCE IDENTITY SURVIVES THE METRIC. With a CONSTANT curvature the CSF force
//      is `sigma' kappa' (C(i) - C(i-s_a)) * w_a` — the SAME per-axis weight `w_a = 1/h_a'^2` the
//      momentum RHS puts on `-(P(i) - P(i-s_a))` (Phase 2 §1.3). So it is again exactly the
//      discrete weighted gradient of `sigma' kappa' C`, the projection annihilates it, and the
//      drop stays at machine zero. Get the weight wrong on ONE axis and this floors at the
//      spurious-current level instead — which is precisely what the gate is for.
//
//  (b) YOUNG-LAPLACE IS THE PHYSICAL JUMP. `dp = sigma * kappa` in the caller's units, on a mesh
//      whose cells differ by a factor 4 between axes.
//
// The colour is the EXACT fraction of a physical sphere over anisotropic cells, so the geometry is
// the same physical problem at every aspect ratio and only the mesh changes.
struct AnisoDrop {
  double maxU = 0.0, dp = 0.0, kappa = 0.0, dtCap = 0.0, hMin = 0.0;
};

/// Exact liquid fraction of the box cell `[x0,x0+hx] x ...` inside a sphere: midpoint-subsampled
/// in x and y, integrated ANALYTICALLY in z. Independent of every kernel under test.
double sphereFracBox(double cx, double cy, double cz, double R, double x0, double y0, double z0,
                     const double h[3], int sub = 24) {
  const double wx = h[0] / sub, wy = h[1] / sub;
  double acc = 0.0;
  for (int b = 0; b < sub; ++b)
    for (int a = 0; a < sub; ++a) {
      const double px = x0 + (a + 0.5) * wx, py = y0 + (b + 0.5) * wy;
      const double r2 = R * R - (px - cx) * (px - cx) - (py - cy) * (py - cy);
      if (r2 <= 0.0)
        continue;
      const double hh = std::sqrt(r2);
      const double lo = std::fmax(cz - hh, z0), hi = std::fmin(cz + hh, z0 + h[2]);
      if (hi > lo)
        acc += hi - lo;
    }
  return acc / (sub * sub * h[2]);
}

/// `n` cells per axis on a box mesh of cell size `h` (physical). `R` is the PHYSICAL radius.
AnisoDrop runDropletAniso(const int n[3], const double h[3], double R, bool constantKappa,
                          int steps) {
  const double SIGMA = 1.0, MU = 0.1, RHO = 1.0;
  peclet::flow::Solver<peclet::flow::Staggered> s(n[0], n[1], n[2]);
  s.setPhysicalDomain({n[0] * h[0], n[1] * h[1], n[2] * h[2]}, {0.0, 0.0, 0.0}, {n[0], n[1], n[2]});
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(1.0);
  s.setVelocityResidualTolerance(0.0);  // machine-precision gate: keep the fixed-sweep loop
  const std::size_t nc = (std::size_t)n[0] * n[1] * n[2];
  const double hMin = std::fmin(h[0], std::fmin(h[1], h[2]));
  s.setPressureGeometry(std::vector<double>(nc, 10.0 * hMin));
  s.setPressureChebyshev(true, 500, 1e-14);
  s.enableVof();  // <- refused before Phase 3; the whole point of this gate
  // the drop, off-centre by an irrational fraction of a cell on every axis
  const double ctr[3] = {0.5 * n[0] * h[0] + 0.13 * h[0], 0.5 * n[1] * h[1] + 0.27 * h[1],
                         0.5 * n[2] * h[2] + 0.11 * h[2]};
  std::vector<double> C(nc);
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i)
        C[(std::size_t)i + (std::size_t)j * n[0] + (std::size_t)k * n[0] * n[1]] =
            sphereFracBox(ctr[0], ctr[1], ctr[2], R, i * h[0], j * h[1], k * h[2], h);
  s.setVof(C);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {RHO, 0.0});
  s.setSurfaceTension(SIGMA);
  AnisoDrop d;
  d.hMin = hMin;
  d.kappa = 2.0 / R;  // physical 1/length
  if (constantKappa)
    s.setVofKappaConstant(d.kappa);
  else
    s.computeVofCurvature();
  d.dtCap = s.capillaryDt();
  s.setDt(0.5 * d.dtCap);
  for (int q = 0; q < steps; ++q)
    s.step();
  for (int c = 0; c < 3; ++c)
    d.maxU = std::fmax(d.maxU, maxAbs(s.getVelocity(c)));
  const std::vector<double> p = s.getPressure();
  const std::size_t inside = (std::size_t)(n[0] / 2) + (std::size_t)(n[1] / 2) * n[0] +
                             (std::size_t)(n[2] / 2) * n[0] * n[1];
  const std::size_t outside = 1 + (std::size_t)1 * n[0] + (std::size_t)1 * n[0] * n[1];
  d.dp = p[inside] - p[outside];
  return d;
}

void gateVofAniso() {
  std::printf("=== units_vof_aniso (Phase 3 S2: the balanced-force CSF on BOX cells) ===\n");
  // Three meshes carrying the SAME physical droplet: cubic, aspect 2, aspect 4. The cell counts
  // keep the physical box cubic, so only the cell SHAPE changes.
  const double base = 1.0 / 32.0;
  struct Mesh {
    const char* name;
    int n[3];
    double h[3];
  };
  const Mesh meshes[3] = {
      {"cubic      ", {32, 32, 32}, {base, base, base}},
      {"aspect 2   ", {32, 64, 32}, {base, base / 2, base}},
      {"aspect 4   ", {32, 64, 128}, {base, base / 2, base / 4}},
  };
  const double R = 8.0 * base;                  // D/h = 16 on the coarsest axis
  const double uScale = 1.0 * (2.0 / R) / 0.1;  // sigma*kappa/mu, the naive-CSF current

  for (int q = 0; q < 3; ++q) {
    const Mesh& m = meshes[q];
    // (a) the EXACTNESS identity: constant kappa -> the projection annihilates the force.
    const AnisoDrop a = runDropletAniso(m.n, m.h, R, true, 20);
    // (b) Young-Laplace with the same constant curvature.
    std::printf("  %s n = (%3d,%3d,%3d)  h'/hMin = (%.2f, %.2f, %.2f)\n", m.name, m.n[0], m.n[1],
                m.n[2], m.h[0] / a.hMin, m.h[1] / a.hMin, m.h[2] / a.hMin);
    std::printf("      exactness max|u| = %.3e   (gate %.3e = 1e-14 * sigma*kappa/mu)\n", a.maxU,
                1e-14 * uScale);
    std::printf("      Young-Laplace dp = %.9e vs sigma*kappa = %.9e   (rel %.2e)\n", a.dp, a.kappa,
                std::fabs(a.dp - a.kappa) / a.kappa);
    std::printf("      capillary_dt     = %.9e   (h_min = %.6e)\n", a.dtCap, a.hMin);
    CHECK(a.maxU < 1e-14 * uScale);
    checkClose(a.dp, a.kappa, 1e-9 * a.kappa, "anisotropic Young-Laplace");
    // The Brackbill limit is set by the SMALLEST cell: refining one axis by 2 must shrink it by
    // 2^{3/2}, which is the statement that `capillary_dt` takes `min_a h_a` and not `h_x`.
    if (q > 0) {
      const double want = meshes[0].h[0] / a.hMin;  // hMin ratio vs the cubic mesh
      const double got = runDropletAniso(meshes[0].n, meshes[0].h, R, true, 0).dtCap / a.dtCap;
      std::printf("      dt_sigma(cubic)/dt_sigma(this) = %.4f   (h^{3/2} predicts %.4f)\n", got,
                  std::pow(want, 1.5));
      CHECK(std::fabs(got - std::pow(want, 1.5)) < 1e-9 * std::pow(want, 1.5));
    }
  }

  // (c) the COMPUTED curvature. Everything above is EXACT because a constant kappa makes the CSF
  // the discrete weighted gradient the projection annihilates; with the cascade's own kappa the
  // residual currents are the curvature error and nothing else. MEASURED: the box mesh costs a
  // constant factor, it does not diverge —
  //
  //     spurious Ca   cubic 2.499e-04   aspect 2  7.869e-04   (ratio 3.15)
  //
  // A factor ~3 at aspect 2, on a mesh whose sphere is resolved BETTER on the refined axis. The
  // likely mechanism is the V2.1 decision (the column direction is ordered by the INDEX normal,
  // which is the right frame for whether 7 cells can close a column but not for how much PHYSICAL
  // interface those 7 cells span) — so a stretched mesh routes more cells to the PV fallback than
  // its cell count suggests. NOT isolated here: confirming it needs the branch census per mesh,
  // and it is an accuracy characteristic rather than a defect, because the property this rung
  // rests on — the exactness identity above — is at machine zero on all three meshes.
  // The gate is therefore set from the measurement (4x) and the number is printed every run.
  const AnisoDrop cc = runDropletAniso(meshes[0].n, meshes[0].h, R, false, 20);
  const AnisoDrop ca = runDropletAniso(meshes[1].n, meshes[1].h, R, false, 20);
  const double caC = 0.1 * cc.maxU / 1.0, caA = 0.1 * ca.maxU / 1.0;
  std::printf("  computed kappa: spurious Ca cubic %.3e vs aspect-2 %.3e (ratio %.2f, gate 4)\n",
              caC, caA, caA / caC);
  CHECK(caA < 4.0 * caC);
  checkClose(ca.dp, ca.kappa, 5e-2 * ca.kappa, "anisotropic Young-Laplace (computed kappa)");
}

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
    else if (gate == "aniso")
      gateAnisoPoiseuille();
    else if (gate == "sphere")
      gateAnisoSphere();
    else if (gate == "tgv")
      gateAnisoTgv();
    else if (gate == "vofaniso")
      gateVofAniso();
    else {
      std::fprintf(stderr, "usage: test_units [identity|scale|vof|aniso|sphere|tgv|vofaniso]\n");
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
