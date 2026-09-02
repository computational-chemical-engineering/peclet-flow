// VoF rung V5b (WO-S) — the theta-consistent solid-band fill, as a PURE KERNEL test.
//
// G0 THE FLAT-WALL LIMIT. A plane interface meeting a flat wall at angle theta, exact fractions
//    everywhere: the fill must reproduce the exact fractions of the CONTINUED plane in the solid
//    cells. That is the Afkhami & Bussmann (IJNMF 57:453, 2008) height-function contact-angle
//    boundary condition written as volume fractions, and it is an IDEMPOTENCE statement — the fill
//    is the identity on a configuration that already satisfies it, so theta is a fixed point of the
//    scheme. It is the gate that discriminates the pivot rules (see `vof/wetting.hpp`).
//
//    G0a idempotence at theta in {30,60,90,120,150}, several interface orientations, DEFAULT pivot.
//    G0b the same with the two ablation pivots  -> the measured error of each (no gate; a number).
//    G0c the rotation itself: m_theta . n_w == cos(theta) and the plane passes through p_f.
//    G0d the wetting limits: theta = 0 fills the band with liquid, theta = 180 empties it.
//    G0e the fluid-only Youngs normal recovered from exact fractions next to the wall -> the angle
//        error it injects (a number; it is the accuracy ceiling of the whole rung).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "vof/plic.hpp"
#include "vof/wetting.hpp"

namespace {
using namespace peclet::flow::vof;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

const double kDeg = M_PI / 180.0;

// The wall: n_w = +z, solid for z < zw. Cell (i,j,k) is the unit cube with corner (i,j,k).
const double zw = 3.0;

// A plane through the point `p` making angle `theta` with n_w, with its tangent direction
// `t_hat` = (cos(psi), sin(psi), 0) in the wall plane.
struct Plane {
  double m[3], A;
};
Plane makePlane(double thetaDeg, double psiDeg, const double p[3]) {
  Plane pl;
  const double th = thetaDeg * kDeg, ps = psiDeg * kDeg;
  pl.m[0] = std::sin(th) * std::cos(ps);
  pl.m[1] = std::sin(th) * std::sin(ps);
  pl.m[2] = std::cos(th);
  pl.A = pl.m[0] * p[0] + pl.m[1] * p[1] + pl.m[2] * p[2];
  return pl;
}
double frac(const Plane& pl, int i, int j, int k) {
  return planeCellFraction(pl.m[0], pl.m[1], pl.m[2], pl.A, i, j, k, 1.0);
}

// One (fluid cell f, solid cell s) pair of the fill, evaluated on the host with the SAME
// container-free kernels the device fill calls.
double fillOnce(const double mf[3], double cf, double thetaDeg, int pivot, const int f[3],
                const int s[3], double* cosApp = nullptr, int* branch = nullptr) {
  const double nw[3] = {0.0, 0.0, 1.0};
  const double sdfF = (f[2] + 0.5) - zw;
  double mth[3], alphaTh, ca;
  const int br = vofWettingPlane(mf, cf, nw, std::cos(thetaDeg * kDeg), std::sin(thetaDeg * kDeg),
                                 sdfF, pivot, 1e-6, mth, alphaTh, ca);
  if (cosApp)
    *cosApp = ca;
  if (branch)
    *branch = br;
  const int ds[3] = {s[0] - f[0], s[1] - f[1], s[2] - f[2]};
  return vofWettingFraction(mth, alphaTh, ds);
}

// ------------------------------------------------------------------ G0a / G0b
const char* kPivName[4] = {"volume (DEFAULT)", "interface p_f (Afkhami-Bussmann)",
                           "wall-normal (WO-S as written)", "contact-line"};

void idempotence() {
  const double thetas[5] = {30.0, 60.0, 90.0, 120.0, 150.0};
  const double psis[3] = {0.0, 37.0, 90.0};
  std::printf("G0a/G0b flat-wall idempotence (EXACT m_f): max |C_fill - C_exact| over the band\n");
  std::printf(
      "   theta |    pivot 0 volume |  pivot 1 interface | pivot 2 wall-normal |"
      " pivot 3 contact-line\n");
  double worst[4] = {0.0, 0.0, 0.0, 0.0};
  for (int ti = 0; ti < 5; ++ti) {
    double e[4] = {0.0, 0.0, 0.0, 0.0};
    for (int pi = 0; pi < 3; ++pi)
      for (int piv = 0; piv < 4; ++piv) {
        const double p[3] = {3.35, 3.6, zw};
        const Plane pl = makePlane(thetas[ti], psis[pi], p);
        for (int dx = -1; dx <= 1; ++dx)
          for (int dy = -1; dy <= 1; ++dy) {
            const int f[3] = {3 + dx, 3 + dy, 3};
            const double cf = frac(pl, f[0], f[1], f[2]);
            if (cf <= 1e-9 || cf >= 1.0 - 1e-9)
              continue;  // pure-phase donor: the fill takes the continuation branch
            for (int ks = 0; ks <= 2; ++ks) {
              const int s[3] = {f[0], f[1], ks};
              const double got = fillOnce(pl.m, cf, thetas[ti], piv, f, s);
              const double ref = frac(pl, s[0], s[1], s[2]);
              e[piv] = std::fmax(e[piv], std::fabs(got - ref));
            }
          }
      }
    std::printf("   %5.0f | %17.3e | %18.3e | %19.3e | %20.3e\n", thetas[ti], e[0], e[1], e[2],
                e[3]);
    for (int piv = 0; piv < 4; ++piv)
      worst[piv] = std::fmax(worst[piv], e[piv]);
  }
  for (int piv = 0; piv < 4; ++piv)
    std::printf("   pivot %d (%s): worst %.3e\n", piv, kPivName[piv], worst[piv]);
  CHECK(worst[0] <= 1e-12);  // G0, the shipped default
  CHECK(worst[1] <= 1e-12);
  CHECK(worst[3] <= 1e-12);
}

// ------------------------------------------------------------------ G0c
void rotation() {
  double worstAng = 0.0, worstPivot = 0.0;
  for (double thA = 20.0; thA <= 160.0; thA += 20.0)
    for (double thT = 15.0; thT <= 165.0; thT += 15.0) {
      const double p[3] = {3.35, 3.6, zw};
      const Plane pl = makePlane(thA, 37.0, p);
      const int f[3] = {3, 3, 3};
      const double cf = frac(pl, f[0], f[1], f[2]);
      if (cf <= 1e-6 || cf >= 1.0 - 1e-6)
        continue;
      const double nw[3] = {0, 0, 1};
      double mth[3], alphaTh, ca;
      vofWettingPlane(pl.m, cf, nw, std::cos(thT * kDeg), std::sin(thT * kDeg), (f[2] + 0.5) - zw,
                      kVofPivotInterface, 1e-6, mth, alphaTh, ca);
      worstAng = std::fmax(worstAng, std::fabs(mth[2] - std::cos(thT * kDeg)));
      // the plane must pass through p_f
      double mf[3] = {pl.m[0], pl.m[1], pl.m[2]};
      plicNormalizeL1(mf);
      double pf[3];
      vofPlicCentroid(mf, plicAlpha(mf[0], mf[1], mf[2], cf), pf);
      worstPivot = std::fmax(worstPivot,
                             std::fabs(mth[0] * pf[0] + mth[1] * pf[1] + mth[2] * pf[2] - alphaTh));
      // the measured apparent angle is the input angle
      worstAng = std::fmax(worstAng, std::fabs(std::acos(ca) / kDeg - thA) / 180.0);
    }
  std::printf(
      "G0c rotation: max |m_theta . n_w - cos(theta)| %.3e ; max |plane(p_f) - alpha| %.3e\n",
      worstAng, worstPivot);
  CHECK(worstAng <= 1e-12);
  CHECK(worstPivot <= 1e-12);
}

// ------------------------------------------------------------------ G0d
void limits() {
  const double p[3] = {3.35, 3.6, zw};
  const Plane pl = makePlane(85.0, 0.0, p);
  const int f[3] = {3, 3, 3};
  const double cf = frac(pl, f[0], f[1], f[2]);
  const int s[3] = {3, 3, 2};
  const double c0 = fillOnce(pl.m, cf, 0.0, kVofPivotInterface, f, s);
  const double c180 = fillOnce(pl.m, cf, 180.0, kVofPivotInterface, f, s);
  const double c30 = fillOnce(pl.m, cf, 30.0, kVofPivotInterface, f, s);
  const double c150 = fillOnce(pl.m, cf, 150.0, kVofPivotInterface, f, s);
  std::printf(
      "G0d wetting limits (C_f = %.4f at theta_a = 85 deg): C_band = %.6f (theta 0), "
      "%.6f (30), %.6f (150), %.6f (180)\n",
      cf, c0, c30, c150, c180);
  CHECK(c0 == 1.0);
  CHECK(c180 == 0.0);
  CHECK(c30 > cf && c150 < cf);  // wetting adds liquid to the band, non-wetting removes it
}

// ------------------------------------------------------------------ G0e
// What the fluid-only Youngs estimator can and cannot measure next to a wall, and what that costs
// the FILL. Two errors are reported per theta:
//   * the angle of the recovered normal to the wall  -> unusable (the one-sided wall-normal
//     difference reads a saturating profile over half the distance);
//   * the AZIMUTH of its in-wall component            -> what the construction actually uses;
// followed by the end-to-end band error of the fill driven by the fluid-only normal instead of the
// exact one, which is the accuracy of the rung as shipped.
void fluidOnlyYoungs() {
  std::printf(
      "G0e fluid-only Youngs next to a flat wall, from EXACT plane fractions:\n"
      "   theta | angle-to-wall err | full-stencil err |  azimuth err | fill err (band)\n");
  double worstAz = 0.0, worstFill = 0.0;
  for (double th = 30.0; th <= 150.0; th += 30.0) {
    double eAng = 0.0, eFull = 0.0, eAz = 0.0, eFill = 0.0;
    for (double psi = 0.0; psi <= 90.0; psi += 45.0)
      for (double sh = 0.0; sh < 1.0; sh += 0.25) {
        const double p[3] = {3.0 + sh, 3.5, zw};
        const Plane pl = makePlane(th, psi, p);
        const int f[3] = {3, 3, 3};
        double c27[27];
        unsigned char fl[27], flAll[27];
        for (int k = 0; k < 3; ++k)
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
              const int q = plicSt(i, j, k);
              const int cx = f[0] + i - 1, cy = f[1] + j - 1, cz = f[2] + k - 1;
              c27[q] = frac(pl, cx, cy, cz);
              fl[q] = (cz + 1 <= zw) ? 0u : 1u;  // a cell entirely below the wall is solid
              flAll[q] = 1u;
            }
        double m[3], mAll[3];
        if (!youngsNormalFluidOnly(c27, fl, m))
          continue;
        youngsNormalFluidOnly(c27, flAll, mAll);
        const double mn = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
        const double an = std::sqrt(mAll[0] * mAll[0] + mAll[1] * mAll[1] + mAll[2] * mAll[2]);
        eAng = std::fmax(
            eAng, std::fabs(std::acos(std::fmax(-1.0, std::fmin(1.0, m[2] / mn))) / kDeg - th));
        eFull = std::fmax(
            eFull, std::fabs(std::acos(std::fmax(-1.0, std::fmin(1.0, mAll[2] / an))) / kDeg - th));
        // azimuth of the in-wall component against the true one
        const double tx = m[0], ty = m[1];
        if (std::sqrt(tx * tx + ty * ty) > 1e-12) {
          double d = std::atan2(ty, tx) / kDeg - psi;
          while (d > 180.0)
            d -= 360.0;
          while (d < -180.0)
            d += 360.0;
          eAz = std::fmax(eAz, std::fabs(d));
        }
        // end-to-end: the fill with the fluid-only normal vs the exact continued plane
        const double cf = frac(pl, f[0], f[1], f[2]);
        if (cf > 1e-9 && cf < 1.0 - 1e-9)
          for (int ks = 0; ks <= 2; ++ks) {
            const int sc[3] = {f[0], f[1], ks};
            eFill = std::fmax(eFill, std::fabs(fillOnce(m, cf, th, kVofPivotVolume, f, sc) -
                                               frac(pl, sc[0], sc[1], sc[2])));
          }
      }
    std::printf("   %5.0f | %17.3f | %16.3f | %12.3f | %16.3e\n", th, eAng, eFull, eAz, eFill);
    worstAz = std::fmax(worstAz, eAz);
    worstFill = std::fmax(worstFill, eFill);
  }
  std::printf("   worst azimuth error %.3f deg ; worst end-to-end band error %.3e\n", worstAz,
              worstFill);
  CHECK(worstAz <= 3.0);
  CHECK(worstFill <= 0.05);
}

// ============================================================ G1/G6: the solver-level gates
//
// G1  THE PRESCRIBED ANGLE IS A FIXED POINT of the whole scheme. A liquid spherical cap of the
//     right volume for the SET angle is placed on a flat SDF wall, surface tension is turned on
//     and the drop is relaxed: the equilibrium must still be that cap. This is the statement the
//     rung actually makes (the fill is exactly idempotent at theta, gate G0a, so the continuum
//     equilibrium is a fixed point of the discrete scheme too), and it is what the study script
//     `tests/study/vof_wetting.py` then extends with the from-90-degrees relaxation.
//
// G6  INERTNESS AND THE 90-DEGREE LIMIT. `set_contact_angle(90)` is NOT bit-identical to WO-Q's
//     neutral fill and cannot be: the neutral rule is the MEAN of the fluid face neighbours, the
//     theta rule is the fraction of a plane, and the two agree only where the interface is already
//     perpendicular to the wall. The measured difference is the number reported here; the real
//     inertness statement — no `set_contact_angle` call leaves every V5a number byte-identical —
//     is gated by diffing the `vof_cutcell` ctest against the same binary built at V5a.
struct CapScene {
  int nx = 40, nz = 32;
  // A QUARTER-integer wall: genuinely CUT wall cells (eps = 3/4) whose tangential MAC faces
  // are still OPEN. At exactly k + 1/2 those faces sit on the SDF zero level, buildOpenness
  // closes them and the contact line cannot move at all — see the WO-S findings.
  double zw = 4.25, R = 10.0, sigma = 1.0, mu = 0.5;
  std::vector<double> sdf, colour;
  // A SLIT (a second wall at nz - zw): the box is periodic, so a lone ramp SDF puts a spurious
  // fluid/solid seam at the wrap plane three cells from the band.
  void build(double thetaDeg) {
    sdf.assign((std::size_t)nx * nx * nz, 0.0);
    colour.assign((std::size_t)nx * nx * nz, 0.0);
    // the cap of the same volume as a hemisphere of radius R, at contact angle theta
    const double ct = std::cos(thetaDeg * kDeg);
    const double V = 2.0 / 3.0 * M_PI * R * R * R;
    const double Rc = std::cbrt(3.0 * V / (M_PI * (1.0 - ct) * (1.0 - ct) * (2.0 + ct)));
    const double zc = zw - Rc * ct;  // the cap sphere's centre
    const double cx = nx * 0.5, cy = nx * 0.5;
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < nx; ++y)
        for (int x = 0; x < nx; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * nx;
          sdf[i] = std::fmin((z + 0.5) - zw, (nz - zw) - (z + 0.5));
          int in = 0, tot = 0;
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
              for (int c = 0; c < 4; ++c) {
                const double px = x + (a + 0.5) / 4.0, py = y + (b + 0.5) / 4.0,
                             pz = z + (c + 0.5) / 4.0;
                if (pz < zw || pz > nz - zw)
                  continue;
                ++tot;
                const double dx = px - cx, dy = py - cy, dz = pz - zc;
                if (dx * dx + dy * dy + dz * dz < Rc * Rc)
                  ++in;
              }
          colour[i] = tot ? (double)in / tot : 0.0;
        }
  }
};

// Relax the cap and return the apparent angle from the cap relations (h on the axis and the
// conserved volume; reading the contact radius off the first fluid plane instead biases it, see
// the study script).
struct CapResult {
  double theta = 0, h = 0, a = 0, drift = 0, ca = 0, umaxAll = 0, apparent = 0;
  long iters = 0, contact = 0, neighbour = 0, pure = 0, neutral = 0;
  std::vector<double> filled, kind, eps;
  int nx = 0, nz = 0;
};
CapResult relaxCap(double thetaSet, int steps, bool setAngle = true) {
  CapScene sc;
  sc.build(setAngle ? thetaSet : 90.0);
  peclet::flow::IbmSolver s(sc.nx, sc.nx, sc.nz);
  s.setRho(1.0);
  s.setMu(sc.mu);
  s.setSolid(sc.sdf, true);
  s.enableVof();
  s.setVof(sc.colour);
  s.setSurfaceTension(sc.sigma);
  if (setAngle)
    s.setContactAngle(thetaSet);
  const double dtc = 0.5 * s.capillaryDt();
  double dt = dtc;
  s.setDt(dt);
  const double v0 = s.vofDiagnostics().volume;
  CapResult r;
  r.filled = s.getVofFilledColour();
  for (int i = 0; i < steps; ++i) {
    const double c = s.vofMaxCourant();
    if (c > 0.15) {
      dt = std::fmin(dtc, dt * 0.15 / c);
      s.setDt(dt);
    } else if (dt < dtc && c < 0.075) {
      dt = std::fmin(dtc, 1.2 * dt);
      s.setDt(dt);
    }
    s.step();
    r.iters = std::max(r.iters, s.lastPressureIterations());
  }
  const auto d = s.vofDiagnostics();
  const auto cd = s.contactAngleDiagnostics();
  const auto cc = s.getVof();
  const auto eps = s.getVofGeometry(0);
  const int ix = sc.nx / 2;
  for (int z = 0; z < sc.nz; ++z) {
    const std::size_t i =
        (std::size_t)ix + (std::size_t)ix * sc.nx + (std::size_t)z * (std::size_t)sc.nx * sc.nx;
    r.h += cc[i] * eps[i];
  }
  r.a = std::sqrt(std::fmax((6.0 * d.volume / (M_PI * r.h) - r.h * r.h) / 3.0, 1e-12));
  r.theta = 2.0 * std::atan2(r.h, r.a) / kDeg;
  r.drift = std::fabs(d.volume - v0) / v0;
  const int z0 = (int)std::ceil(sc.zw);
  for (int c = 0; c < 3; ++c) {
    const auto v = s.getVelocity(c);
    for (int z = 0; z < sc.nz; ++z)
      for (int y = 0; y < sc.nx; ++y)
        for (int x = 0; x < sc.nx; ++x) {
          const double av = std::fabs(v[(std::size_t)x + (std::size_t)y * sc.nx +
                                        (std::size_t)z * (std::size_t)sc.nx * sc.nx]);
          r.umaxAll = std::fmax(r.umaxAll, av);
          if (z >= z0)
            r.ca = std::fmax(r.ca, sc.mu * av / sc.sigma);
        }
  }
  r.kind = s.getVofGeometry(4);
  r.eps = eps;
  r.nx = sc.nx;
  r.nz = sc.nz;
  r.apparent = cd.meanApparentAngle;
  r.contact = cd.contactCells;
  r.neighbour = cd.neighbourCells;
  r.pure = cd.pureCells;
  r.neutral = cd.neutralCells;
  r.filled = s.getVofFilledColour();
  return r;
}

void solverGates() {
  const double thetas[3] = {60.0, 90.0, 120.0};
  const int steps = std::getenv("PECLET_VOF_WETTING_LONG") ? 600 : 150;
  std::printf(
      "G1 the prescribed angle is a fixed point (cap of the SET angle on a flat SDF wall "
      "at z = 4.25 — CUT wall cells, D/dx = 20, sigma 1, mu 0.5, %d steps):\n",
      steps);
  double worst = 0.0;
  for (double th : thetas) {
    const CapResult r = relaxCap(th, steps);
    std::printf(
        "   theta_set %5.1f -> %7.3f (err %+6.3f)  h %6.3f a %6.3f  apparent %6.2f  "
        "dV/V %.2e  Ca(open) %.3e  max|u| all %.3e  band th/nbr/pure/neu "
        "%ld/%ld/%ld/%ld  iters %ld\n",
        th, r.theta, r.theta - th, r.h, r.a, r.apparent, r.drift, r.ca, r.umaxAll, r.contact,
        r.neighbour, r.pure, r.neutral, r.iters);
    CHECK(r.drift <= 1e-9);
    worst = std::fmax(worst, std::fabs(r.theta - th));
  }
  std::printf("   G1 worst |theta - theta_set| = %.3f deg (gate 3.0)\n", worst);
  CHECK(worst <= 3.0);

  // G6: the 90-degree limit against WO-Q's neutral fill, on the SAME scene.
  const CapResult a = relaxCap(90.0, 0, true), b = relaxCap(90.0, 0, false);
  double dm = 0.0, dmReach = 0.0;
  const int NX = a.nx, NZ = a.nz;
  const auto at = [&](const std::vector<double>& v, int x, int y, int z) {
    return v[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * (std::size_t)NX * NX];
  };
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NX; ++y)
      for (int x = 0; x < NX; ++x) {
        if (at(a.kind, x, y, z) < 0.5)
          continue;  // fluid cell: not part of the band
        const double d = std::fabs(at(a.filled, x, y, z) - at(b.filled, x, y, z));
        dm = std::fmax(dm, d);
        // ... and restricted to the band cells the V3 cascade can actually reach (a fluid cell
        // within the +/-3 the height-function columns and the MYC stencil span)
        bool reach = false;
        for (int dz = -3; dz <= 3 && !reach; ++dz)
          for (int dy = -3; dy <= 3 && !reach; ++dy)
            for (int dx = -3; dx <= 3 && !reach; ++dx) {
              const int px = x + dx, py = y + dy, pz = z + dz;
              if (px < 0 || py < 0 || pz < 0 || px >= NX || py >= NX || pz >= NZ)
                continue;
              if (at(a.kind, px, py, pz) < 0.5)
                reach = true;
            }
        if (reach)
          dmReach = std::fmax(dmReach, d);
      }
  const CapResult a1 = relaxCap(90.0, steps, true), b1 = relaxCap(90.0, steps, false);
  std::printf(
      "G6 theta = 90 vs WO-Q's neutral fill on the same scene, at t = 0: "
      "max |C_theta - C_neutral| over ALL solid cells %.3e, and over the solid cells the V3 "
      "cascade can reach (a fluid cell within +/-3) %.3e. NOT bitwise, and it cannot be — "
      "the neutral rule is the MEAN of the fluid face neighbours, the theta rule is a plane "
      "FRACTION, and the two coincide only where the interface is already perpendicular to "
      "the wall; the all-cells number is dominated by band cells deeper than the neutral "
      "rule's 3-pass reach, which it leaves untouched and the theta walk fills.\n"
      "   after %d steps: theta %.3f (theta-fill) vs %.3f (neutral), max|u| %.3e vs %.3e, "
      "Ca(open) %.3e vs %.3e\n",
      dm, dmReach, steps, a1.theta, b1.theta, a1.umaxAll, b1.umaxAll, a1.ca, b1.ca);
  CHECK(std::fabs(a1.theta - b1.theta) <= 1.0);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Kokkos::initialize(argc, argv);
  {
    idempotence();
    rotation();
    limits();
    fluidOnlyYoungs();
    if (!std::getenv("PECLET_VOF_WETTING_KERNEL_ONLY"))
      solverGates();
  }
  Kokkos::finalize();
  if (failures)
    std::fprintf(stderr, "%d check(s) FAILED\n", failures);
  else
    std::printf("all vof wetting (G0) checks passed\n");
  return failures ? 1 : 0;
}
