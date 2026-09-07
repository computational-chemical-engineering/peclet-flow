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
// Phase 2 gate G1 (doc/anisotropic_metric.md §8.2).
// --------------------------------------------------------------------------------------------

/// Plane Poiseuille between two cut-cell walls that sit exactly on y cell CENTRES, on an arbitrary
/// (possibly anisotropic) physical box.  `jlo`/`jhi` are the wall cell indices; everything else is
/// `runChannel`'s recipe, in the unit system the caller writes the box in (lam = 1).
struct Channel {
  Fields f;
  std::vector<double> yc;              ///< the physical y cell centres
  double ylo = 0.0, yhi = 0.0;         ///< the two wall positions (== yc[jlo], yc[jhi])
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
  std::printf("      H %.17g  u_max %.17g  max rel |u - parabola| %.3e (bound %.0e)  max|v| %.3e  "
              "max|w| %.3e  max|u| on the wall/solid rows %.3e\n",
              H, umax, relErr, bound, mv, mw, wallMax);
  if (!(relErr <= bound))
    std::printf("      NOTE: a miss that is a SINGLE one-signed multiplicative factor across the "
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
  // (d) The §7 refusals: an anisotropic domain is admitted by the staggered solver above and
  //     refused, with the three spacings in the message, by the collocated policy, by enable_vof
  //     and by the hydro force integrals (each lifted by a named later commit).
  {
    bool threw = false;
    try {
      peclet::flow::Solver<peclet::flow::Colocated> sc(16, 40, 8);
      sc.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      refused (collocated): %s\n", ex.what());
    }
    CHECK(threw);
    threw = false;
    try {
      peclet::flow::Solver<peclet::flow::Staggered> ss(16, 40, 8);
      ss.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
      ss.enableVof();
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      refused (enable_vof): %s\n", ex.what());
    }
    CHECK(threw);
    threw = false;
    try {
      peclet::flow::Solver<peclet::flow::Staggered> ss(16, 40, 8);
      ss.setPhysicalDomain({16.0, 12.0, 16.0}, {0.0, 0.0, 0.0}, {16, 40, 8});
      ss.hydroForceTorque();
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("      refused (hydro_force_torque): %s\n", ex.what());
    }
    CHECK(threw);
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
    else if (gate == "aniso")
      gateAnisoPoiseuille();
    else {
      std::fprintf(stderr, "usage: test_units [identity|scale|vof|aniso]\n");
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
