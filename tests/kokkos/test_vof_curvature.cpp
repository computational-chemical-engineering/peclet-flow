// VoF rung V3 (WO-O) — gate battery for the interface-curvature cascade
// (src/vof/curvature.hpp + src/vof/curvature_field.hpp). Standalone: no Solver, no MPI.
//
//   A  geometry primitives  : the plane^cube polygon against the analytic dV/dalpha area and the
//                             analytic interface centroid -dV/dm / dV/dalpha; the Green's-theorem
//                             monomial integrals against closed forms and a quadrature oracle
//   B  exact-fraction sphere: kappa vs the analytic 2/R at 16^3 / 32^3 / 64^3 -- L1 AND max error
//                             reported separately, convergence order, per-branch census; plus an
//                             initialization ablation showing the octree fractions are not the
//                             limiting error
//   B2 shape sanity         : a PLANE gives kappa = 0 and a CYLINDER gives 1/R, so the factor in
//                             kappa = 2H is gated and not assumed
//   C  resolution sweep     : D/dx from ~3 to ~40, including D/dx < 5 where the cascade MUST fall
//                             back; branch fractions per resolution; assertion that no interfacial
//                             cell returns NaN, and none returns kCurvNoEstimate (a curvature of
//                             exactly zero at an interfacial cell is only ever legal when the
//                             branch field says the interface really is flat)
//   D  translating droplet  : advection-realistic volume fractions from the V1 Weymouth-Yue
//                             advector; the error PLATEAU and the C*dx at which it sets in --
//                             a measurement, not a pass/fail (VOF_PLAN.md 6)
//   E  device / host        : the same kernels in a Kokkos parallel_for and in a serial host loop,
//                             compared bitwise on the same backend
//   F  fallback in isolation: the PV branch forced on every interfacial cell, so its own accuracy
//                             is a measured number rather than an inference from the mixture
//   G  tier 2b ablation     : the mixed height-position fit (VofCurvature::useMixedHeightFit) on
//                             and off on the SAME sphere -- the measurement that put it off by
//                             default, regenerated every run so it cannot go stale in a doc
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vof/advect_wy.hpp"
#include "vof/curvature.hpp"
#include "vof/curvature_field.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::VofCurvature;
using peclet::flow::vof::WyAdvector;
using vofscene::Block;

namespace vf = peclet::flow::vof;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

double order(double coarse, double fine) {
  return (fine > 0.0 && coarse > 0.0) ? std::log2(coarse / fine) : 0.0;
}

/// A single-block periodic case: colour storage + ghost fill + the curvature driver.
struct Case {
  WyAdvector adv;
  VofCurvature curv;
  Block blk;

  void setup(int nx, int ny, int nz, double h) {
    adv.init(nx, ny, nz, h, 3);
    curv.init(nx, ny, nz, 3);
    blk = vofscene::blockOf(adv, I3{0, 0, 0});
    const I3 e = adv.extent();
    const int g = adv.ghost();
    adv.exchange = [e, g](SField f) { vofscene::periodicFill(f, e, g, true, true, true); };
    // Measurement overrides for the tier-2b study recorded in the WO-O findings entry.
    if (const char* v = std::getenv("PECLET_VOF_CURV_PTW"))
      curv.ptWeightWidth = std::atof(v);
    if (const char* v = std::getenv("PECLET_VOF_CURV_FIT"))
      curv.useMixedHeightFit = (std::atoi(v) != 0);
  }
  SField c() const { return adv.colour(); }
  I3 e() const { return adv.extent(); }
  I3 n() const { return adv.inner(); }
  int g() const { return adv.ghost(); }
};

/// Error of the computed curvature against a constant analytic value, over the interfacial cells.
struct Err {
  double l1 = 0.0;   ///< mean |kappa - kExact| / |kExact|
  double max = 0.0;  ///< max  |kappa - kExact| / |kExact|
  long n = 0;
  long nan = 0;
};

/// `margin` excludes cells within that many cells of the inner-region boundary. Needed only where
/// the manufactured field is not periodic (an oblique plane in a periodic box has a second,
/// spurious interface at the wrap plane); every closed shape uses margin 0.
Err curvError(const Case& cs, double kExact, int margin = 0) {
  const I3 e = cs.e(), n = cs.n();
  const int g = cs.g() + margin;
  SField kap = cs.curv.kappa(), br = cs.curv.branch();
  // kExact = 0 (a flat interface) makes a relative error meaningless; report the absolute one.
  const double den = (kExact != 0.0) ? std::fabs(kExact) : 1.0;
  double sum = 0.0, mx = 0.0;
  long cnt = 0, nan = 0;
  Kokkos::parallel_reduce(
      "curv::err",
      Kokkos::MDRangePolicy<peclet::flow::SExec, Kokkos::Rank<3>>(peclet::flow::SExec(), {g, g, g},
                                                                  {g + n.x - 2 * margin,
                                                                   g + n.y - 2 * margin,
                                                                   g + n.z - 2 * margin}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc, double& m, long& c, long& bad) {
        const long i = L3(x, y, z, e);
        if (static_cast<int>(br(i)) == vf::kCurvNone)
          return;
        const double k = kap(i);
        if (!(k == k)) {  // NaN
          ++bad;
          return;
        }
        const double r = Kokkos::fabs(k - kExact) / den;
        acc += r;
        m = Kokkos::fmax(m, r);
        ++c;
      },
      sum, Kokkos::Max<double>(mx), cnt, nan);
  Kokkos::fence();
  Err r;
  r.n = cnt;
  r.nan = nan;
  r.l1 = cnt ? sum / static_cast<double>(cnt) : 0.0;
  r.max = mx;
  return r;
}

void printStats(const char* tag, const VofCurvature::Stats& s) {
  const double t = s.interfacial ? static_cast<double>(s.interfacial) : 1.0;
  std::printf("    %s interfacial %5ld | HF %5.2f%%  HFdir %5.2f%%  HFfit %5.2f%%  PV %5.2f%%"
              "  PVred %5.3f%%  none %ld\n",
              tag, s.interfacial, 100.0 * s.hf / t, 100.0 * s.hfMixed / t, 100.0 * s.hfFit / t,
              100.0 * s.pv / t, 100.0 * s.pvReduced / t, s.noEstimate);
}

// ========================================================== gate A: the geometry primitives
void gateGeometry() {
  std::printf("\n=== A  geometry primitives (polygon, area/centroid, Green monomials)\n");

  // A1/A2: polygon area and centroid of the plane^cube section, against the analytic identities
  //   |Gamma| = |m|_2 dV/dalpha        and       xbar_i = -(dV/dm_i) / (dV/dalpha)
  // both derived from V(m,alpha) = int_cell H(alpha - m.x) dx. Independent of the polygon code.
  unsigned s = 12345u;
  auto rnd = [&s]() {
    s = 1664525u * s + 1013904223u;
    return static_cast<double>(s >> 8) / 16777216.0;
  };
  double maxA = 0.0, maxC = 0.0;
  int nsamp = 0;
  for (int t = 0; t < 4000; ++t) {
    double m[3] = {2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0};
    const double l1 = std::fabs(m[0]) + std::fabs(m[1]) + std::fabs(m[2]);
    if (l1 < 1e-3)
      continue;
    for (int k = 0; k < 3; ++k)
      m[k] /= l1;
    const double V = 0.05 + 0.9 * rnd();
    const double al = vf::plicAlpha(m[0], m[1], m[2], V);
    double v[8][3], ctr[3], area;
    const int nv = vf::plicPolygon(m[0], m[1], m[2], al, v);
    if (nv < 3)
      continue;
    vf::polygonAreaCentroid(v, nv, ctr, area);
    const double d = 1e-6;
    const double dVda = (vf::plicVolume(m[0], m[1], m[2], al + d) -
                         vf::plicVolume(m[0], m[1], m[2], al - d)) /
                        (2 * d);
    const double l2 = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    maxA = std::max(maxA, std::fabs(area - l2 * dVda) / std::max(area, 1e-3));
    for (int k = 0; k < 3; ++k) {
      double mp[3] = {m[0], m[1], m[2]}, mm[3] = {m[0], m[1], m[2]};
      mp[k] += d;
      mm[k] -= d;
      const double dVdm = (vf::plicVolume(mp[0], mp[1], mp[2], al) -
                           vf::plicVolume(mm[0], mm[1], mm[2], al)) /
                          (2 * d);
      maxC = std::max(maxC, std::fabs(ctr[k] - (-dVdm / dVda)));
    }
    ++nsamp;
  }
  std::printf("  %d random planes:  max rel |Gamma| error %.3e   max centroid error %.3e\n", nsamp,
              maxA, maxC);
  CHECK(nsamp > 3000);
  CHECK(maxA < 1e-6);  // limited by the finite difference, not by the polygon
  CHECK(maxC < 1e-5);

  // A3: the Green's-theorem monomials on the unit square, closed form.
  {
    double xy[8][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    double q[6];
    vf::polygonMoments2d(xy, 4, q);
    const double ex[6] = {1.0, 0.5, 0.5, 1.0 / 3.0, 0.25, 1.0 / 3.0};
    double m = 0.0;
    for (int i = 0; i < 6; ++i)
      m = std::max(m, std::fabs(q[i] - ex[i]));
    std::printf("  unit square monomials (1,x,y,x2,xy,y2) = (%g,%g,%g,%g,%g,%g)  max err %.3e\n",
                q[0], q[1], q[2], q[3], q[4], q[5], m);
    CHECK(m < 1e-15);
  }
  // A3b: an offset/rotated random convex polygon vs a dense midpoint quadrature oracle.
  {
    double xy[8][2];
    const int nv = 5;
    for (int k = 0; k < nv; ++k) {
      const double a = 2.0 * M_PI * k / nv + 0.37;
      xy[k][0] = 1.3 + 0.9 * std::cos(a);
      xy[k][1] = -0.6 + 1.4 * std::sin(a);
    }
    double q[6];
    vf::polygonMoments2d(xy, nv, q);
    // oracle: midpoint quadrature over the bounding box with a convex inside test
    double lo[2] = {1e30, 1e30}, hi[2] = {-1e30, -1e30};
    for (int k = 0; k < nv; ++k)
      for (int d = 0; d < 2; ++d) {
        lo[d] = std::min(lo[d], xy[k][d]);
        hi[d] = std::max(hi[d], xy[k][d]);
      }
    const int NQ = 2000;
    const double dx = (hi[0] - lo[0]) / NQ, dy = (hi[1] - lo[1]) / NQ, dA = dx * dy;
    double o[6] = {0, 0, 0, 0, 0, 0};
    for (int j = 0; j < NQ; ++j)
      for (int i = 0; i < NQ; ++i) {
        const double x = lo[0] + (i + 0.5) * dx, y = lo[1] + (j + 0.5) * dy;
        bool in = true;
        for (int k = 0; k < nv && in; ++k) {
          const int k1 = (k + 1) % nv;
          const double cr = (xy[k1][0] - xy[k][0]) * (y - xy[k][1]) -
                            (xy[k1][1] - xy[k][1]) * (x - xy[k][0]);
          if (cr < 0.0)
            in = false;
        }
        if (!in)
          continue;
        o[0] += dA;
        o[1] += x * dA;
        o[2] += y * dA;
        o[3] += x * x * dA;
        o[4] += x * y * dA;
        o[5] += y * y * dA;
      }
    double m = 0.0;
    for (int i = 0; i < 6; ++i)
      m = std::max(m, std::fabs(q[i] - o[i]) / std::max(std::fabs(o[i]), 1e-3));
    std::printf("  random pentagon vs %dx%d quadrature: max rel err %.3e\n", NQ, NQ, m);
    CHECK(m < 5e-4);
  }
}

// ==================================================== gate B2: plane and cylinder shape sanity
void gateShapes() {
  std::printf("\n=== B2 shape sanity (plane -> 0, cylinder -> 1/R, i.e. kappa = 2H)\n");
  const int N = 48;
  const double h = 1.0 / N;

  // an oblique plane: kappa must be 0
  {
    Case cs;
    cs.setup(N, N, N, h);
    double m[3] = {0.47, -0.31, 0.86};
    const double l1 = std::fabs(m[0]) + std::fabs(m[1]) + std::fabs(m[2]);
    for (int k = 0; k < 3; ++k)
      m[k] /= l1;
    const double m0 = m[0], m1 = m[1], m2 = m[2];
    SField cc = cs.c();
    vofscene::forEachExtended(
        cs.blk, KOKKOS_LAMBDA(long i, int gx, int gy, int gz) {
          cc(i) = vf::planeCellFraction(m0, m1, m2, 0.5 * (m0 + m1 + m2), gx * h, gy * h, gz * h,
                                        h);
        });
    cs.adv.syncGhosts();
    const auto st = cs.curv.compute(cs.c());
    // margin 8: an oblique plane cannot be periodic, so the wrap planes carry a spurious second
    // interface. The census below covers the whole block (and reports it), the error metric only
    // the interior where the manufactured field is the plane it claims to be.
    const Err er = curvError(cs, 0.0, 8);  // kExact = 0 -> absolute error, in 1/cell
    printStats("plane  ", st);
    std::printf("    kappa (interior only): L1 %.3e  max %.3e  (analytic 0)  cells %ld  NaN %ld\n",
                er.l1, er.max, er.n, er.nan);
    CHECK(er.nan == 0);
    CHECK(er.n > 500);
    CHECK(er.max < 1e-9);
  }

  // a cylinder along z of radius R: kappa = 1/R (one principal curvature is zero)
  {
    Case cs;
    cs.setup(N, N, N, h);
    const double R = 0.3, cx = 0.5, cy = 0.5;
    const int NS = 128;
    SField cc = cs.c();
    vofscene::forEachExtended(
        cs.blk, KOKKOS_LAMBDA(long i, int gx, int gy, int) {
          const double x0 = gx * h, y0 = gy * h, w = h / NS;
          int in = 0;
          for (int q = 0; q < NS; ++q) {
            const double y = y0 + (q + 0.5) * w - cy;
            for (int p = 0; p < NS; ++p) {
              const double x = x0 + (p + 0.5) * w - cx;
              if (x * x + y * y < R * R)
                ++in;
            }
          }
          cc(i) = static_cast<double>(in) / (static_cast<double>(NS) * NS);
        });
    cs.adv.syncGhosts();
    const auto st = cs.curv.compute(cs.c());
    const double kEx = h / R;  // 1/R in cell units
    const Err er = curvError(cs, kEx);
    printStats("cylinder", st);
    std::printf("    kappa: L1 %.3e  max %.3e  (analytic %.6f = 1/R)   NaN %ld\n", er.l1, er.max,
                kEx, er.nan);
    CHECK(er.nan == 0);
    CHECK(st.noEstimate == 0);
    // 1 % is far inside the factor-2 that would separate 1/R from 2/R -- this gate exists to pin
    // the mean-curvature convention, not to measure accuracy.
    CHECK(er.l1 < 0.02);
  }
}

// ============================================ gate B: exact-fraction sphere, static convergence
void gateStaticSphere() {
  std::printf("\n=== B  exact-fraction sphere: static convergence of kappa vs 2/R\n");
  const int Ns[3] = {16, 32, 64};
  const int LEV = 6;
  double l1[3], mx[3];
  for (int q = 0; q < 3; ++q) {
    const int N = Ns[q];
    const double h = 1.0 / N;
    const double R = 0.3;
    Case cs;
    cs.setup(N, N, N, h);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, R, LEV);
    cs.adv.syncGhosts();
    const auto st = cs.curv.compute(cs.c());
    const double kEx = 2.0 * h / R;
    const Err er = curvError(cs, kEx);
    l1[q] = er.l1;
    mx[q] = er.max;
    std::printf("  N = %3d  D/dx = %5.1f  C*dx = %.4f\n", N, 2.0 * R / h, h / R);
    printStats("        ", st);
    std::printf("    kappa rel error:  L1 %.4e   max %.4e   NaN %ld\n", er.l1, er.max, er.nan);
    CHECK(er.nan == 0);
    CHECK(st.noEstimate == 0);
  }
  std::printf("  convergence order  L1: %.2f, %.2f   max: %.2f, %.2f\n", order(l1[0], l1[1]),
              order(l1[1], l1[2]), order(mx[0], mx[1]), order(mx[1], mx[2]));
  const double oL1 = order(l1[0], l1[2]) / 2.0;
  std::printf("  fitted order over 16->64:  L1 %.2f   max %.2f\n", oL1, order(mx[0], mx[2]) / 2.0);
  CHECK(oL1 > 1.7);

  // initialization ablation: if the octree fractions were the limiting error the measured error
  // would move with `levels`. Run 32^3 at levels 4 and 8 and print all three.
  double abl[3];
  const int levs[3] = {4, 6, 8};
  for (int q = 0; q < 3; ++q) {
    const int N = 32;
    const double h = 1.0 / N, R = 0.3;
    Case cs;
    cs.setup(N, N, N, h);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, R, levs[q]);
    cs.adv.syncGhosts();
    cs.curv.compute(cs.c());
    abl[q] = curvError(cs, 2.0 * h / R).l1;
  }
  std::printf("  init ablation at 32^3, octree levels 4/6/8:  L1 %.4e / %.4e / %.4e\n", abl[0],
              abl[1], abl[2]);
  CHECK(std::fabs(abl[1] - abl[2]) < 0.2 * abl[2]);  // levels 6 is not the limiting error
}

// ======================================================= gate C: resolution sweep incl. D/dx < 5
void gateSweep() {
  std::printf("\n=== C  sphere sweep incl. D/dx < 5 (the fallback MUST engage)\n");
  const double Rc[6] = {1.4, 2.2, 3.5, 6.0, 12.0, 24.0};  // radius in CELLS
  for (int q = 0; q < 6; ++q) {
    const double Rcell = Rc[q];
    const int N = static_cast<int>(std::ceil(2.6 * Rcell)) + 8;
    const double h = 1.0 / N, R = Rcell * h;
    Case cs;
    cs.setup(N, N, N, h);
    // off-centre by an irrational fraction of a cell so the sphere is never mesh-aligned
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5 + 0.137 * h, 0.5 - 0.311 * h, 0.5 + 0.241 * h, R,
                         6);
    cs.adv.syncGhosts();
    const auto st = cs.curv.compute(cs.c());
    const Err er = curvError(cs, 2.0 * h / R);
    std::printf("  D/dx = %5.1f  C*dx = %.4f  N = %3d\n", 2.0 * Rcell, 1.0 / Rcell, N);
    printStats("        ", st);
    std::printf("    kappa rel error:  L1 %.4e   max %.4e   NaN %ld\n", er.l1, er.max, er.nan);
    CHECK(er.nan == 0);
    CHECK(st.noEstimate == 0);
    CHECK(st.interfacial > 0);
    CHECK(st.hf + st.hfMixed + st.hfFit + st.pv + st.pvReduced == st.interfacial);
    if (2.0 * Rcell < 5.0)
      CHECK(st.pv + st.pvReduced > 0);  // below ~5 cells/diameter the HF cannot always close
  }
}

// ================================================= gate D: translating droplet, realistic fractions
void gateTranslating() {
  std::printf("\n=== D  translating droplet (advection-realistic fractions) -- a MEASUREMENT\n");
  std::printf("  the plateau is the method's physics (Han/Evrard/Desjardins 2024 3; "
              "VOF_PLAN.md 6)\n");
  const int Ns[3] = {16, 32, 64};
  double l1[3], mx[3], l1e[3];
  for (int q = 0; q < 3; ++q) {
    const int N = Ns[q];
    const double h = 1.0 / N, R = 0.2;
    Case cs;
    cs.setup(N, N, N, h);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, R, 6);
    cs.adv.syncGhosts();
    // curvature of the EXACT initial field, for reference
    cs.curv.compute(cs.c());
    l1e[q] = curvError(cs, 2.0 * h / R).l1;
    // transport it one full diameter along the diagonal at CFL 0.25
    const double dt = 0.25 * h;
    vofscene::fillUniform(cs.adv, cs.blk, 1.0, 1.0, 1.0);
    const long steps = static_cast<long>(std::lround(2.0 * R / (0.25 * h)));
    for (long s = 0; s < steps; ++s)
      cs.adv.advect(dt, s);
    const auto st = cs.curv.compute(cs.c());
    const Err er = curvError(cs, 2.0 * h / R);
    l1[q] = er.l1;
    mx[q] = er.max;
    std::printf("  N = %3d  D/dx = %5.1f  C*dx = %.4f  after %ld steps (1 diameter)\n", N,
                2.0 * R / h, h / R, steps);
    printStats("        ", st);
    std::printf("    kappa rel error:  L1 %.4e (exact fractions %.4e)   max %.4e   NaN %ld\n",
                er.l1, l1e[q], er.max, er.nan);
    CHECK(er.nan == 0);
    CHECK(st.noEstimate == 0);
  }
  std::printf("  ADVECTED  order L1: %.2f, %.2f    max: %.2f, %.2f\n", order(l1[0], l1[1]),
              order(l1[1], l1[2]), order(mx[0], mx[1]), order(mx[1], mx[2]));
  std::printf("  EXACT     order L1: %.2f, %.2f\n", order(l1e[0], l1e[1]), order(l1e[1], l1e[2]));
  std::printf("  ratio advected/exact L1: %.1fx, %.1fx, %.1fx  at C*dx %.3f, %.3f, %.3f\n",
              l1[0] / l1e[0], l1[1] / l1e[1], l1[2] / l1e[2], 1.0 / (0.2 * Ns[0]),
              1.0 / (0.2 * Ns[1]), 1.0 / (0.2 * Ns[2]));
  // No pass/fail on the order: the WO says report it. What IS gated is that the exact-fraction
  // reference on the same geometry converges, so a flat advected curve is transport noise and not
  // a broken estimator.
  CHECK(order(l1e[0], l1e[2]) / 2.0 > 1.5);
}

// ================================================================ gate E: device vs host oracle
// A serial host mirror of the driver, calling the SAME KOKKOS_INLINE_FUNCTIONs. Deliberately
// written out rather than shared with the driver, so it is an independent transcription of the
// cascade rather than the same code twice.
void gateDeviceHost() {
  std::printf("\n=== E  device kernel vs serial host loop (same backend, bitwise)\n");
  const int N = 32;
  const double h = 1.0 / N, R = 0.3;
  Case cs;
  cs.setup(N, N, N, h);
  // The oracle below mirrors the SHIPPED cascade (tier 1 -> tier 2a -> tier 3); tier 2b is off by
  // default and is pinned off here so the two transcriptions describe the same algorithm.
  cs.curv.useMixedHeightFit = false;
  vofscene::initSphere(cs.c(), cs.blk, h, 0.5 + 0.1 * h, 0.5, 0.5 - 0.2 * h, R, 5);
  cs.adv.syncGhosts();
  cs.curv.compute(cs.c());

  const I3 e = cs.e();
  const int g = cs.g();
  auto ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cs.c());
  auto kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cs.curv.kappa());
  auto bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cs.curv.branch());
  const long st[3] = {1, e.x, static_cast<long>(e.x) * e.y};
  auto C = [&](long i) { return ch(i); };

  long bitDiff = 0, cmp = 0;
  double maxDiff = 0.0;
  const int gr = vf::kPvHalf;
  // host planes over the inner region grown by kPvHalf
  std::vector<double> hm(3 * ch.extent(0), 0.0), ha(ch.extent(0), 0.0);
  for (int z = g - gr; z < g + N + gr; ++z)
    for (int y = g - gr; y < g + N + gr; ++y)
      for (int x = g - gr; x < g + N + gr; ++x) {
        const long i = L3(x, y, z, e);
        if (!vf::wyIsMixed(C(i)))
          continue;
        double stc[27], m[3];
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              stc[vf::plicSt(ii + 1, jj + 1, kk + 1)] = C(i + ii * st[0] + jj * st[1] + kk * st[2]);
        vf::mycNormal(stc, m);
        hm[3 * i] = m[0];
        hm[3 * i + 1] = m[1];
        hm[3 * i + 2] = m[2];
        ha[i] = vf::plicAlpha(m[0], m[1], m[2], C(i));
      }

  for (int z = g; z < g + N; ++z)
    for (int y = g; y < g + N; ++y)
      for (int x = g; x < g + N; ++x) {
        const long i = L3(x, y, z, e);
        double kappa = 0.0;
        int branch = vf::kCurvNone;
        if (vf::wyIsMixed(C(i))) {
          const double am[3] = {std::fabs(hm[3 * i]), std::fabs(hm[3 * i + 1]),
                                std::fabs(hm[3 * i + 2])};
          int ord[3] = {0, 1, 2};
          for (int p = 1; p < 3; ++p)
            for (int qq = p; qq > 0 && am[ord[qq]] > am[ord[qq - 1]]; --qq)
              std::swap(ord[qq], ord[qq - 1]);
          branch = -1;
          for (int t = 0; t < 3 && branch < 0; ++t) {
            const int d = ord[t], d1 = (d + 1) % 3, d2 = (d + 2) % 3;
            double hh[9];
            int orient0 = 0;
            bool ok = true;
            for (int qq = 0; qq < 3 && ok; ++qq)
              for (int p = 0; p < 3 && ok; ++p) {
                const long base = i + (p - 1) * st[d1] + (qq - 1) * st[d2];
                double col[vf::kHfColumn];
                for (int k = 0; k < vf::kHfColumn; ++k)
                  col[k] = C(base + (k - vf::kHfColumn / 2) * st[d]);
                double hv;
                int orient;
                if (!vf::hfColumnHeight(col, vf::kHfColumn, hv, orient, cs.curv.monoTol)) {
                  ok = false;
                  break;
                }
                if (orient0 == 0)
                  orient0 = orient;
                else if (orient != orient0) {
                  ok = false;
                  break;
                }
                hh[p + 3 * qq] = hv;
              }
            if (!ok)
              continue;
            kappa = vf::hfPatchKappa(hh);
            branch = (t == 0) ? vf::kCurvHf : vf::kCurvHfMixed;
          }
          if (branch < 0) {
            const double m0 = hm[3 * i], m1 = hm[3 * i + 1], m2 = hm[3 * i + 2];
            const double n2 = m0 * m0 + m1 * m1 + m2 * m2;
            if (!(n2 > 0.0)) {
              branch = vf::kCurvNoEstimate;
            } else {
              const double invn = 1.0 / std::sqrt(n2);
              const double nn[3] = {m0 * invn, m1 * invn, m2 * invn};
              double t1[3], t2[3];
              vf::curvFrame(nn, t1, t2);
              double v[8][3], ctr[3], area;
              const int nv = vf::plicPolygon(m0, m1, m2, ha[i], v);
              vf::polygonAreaCentroid(v, nv, ctr, area);
              const double org[3] = {ctr[0] - 0.5, ctr[1] - 0.5, ctr[2] - 0.5};
              vf::PvFit fit;
              vf::pvFitInit(fit);
              for (int oz = -gr; oz <= gr; ++oz)
                for (int oy = -gr; oy <= gr; ++oy)
                  for (int ox = -gr; ox <= gr; ++ox) {
                    const long j = L3(x + ox, y + oy, z + oz, e);
                    if (!vf::wyIsMixed(C(j)))
                      continue;
                    const double off[3] = {static_cast<double>(ox), static_cast<double>(oy),
                                           static_cast<double>(oz)};
                    vf::pvFitAdd(fit, hm[3 * j], hm[3 * j + 1], hm[3 * j + 2], ha[j], off, org, t1,
                                 t2, nn, cs.curv.weightWidth, cs.curv.cosMin);
                  }
              double a[6];
              bool red = false;
              if (!vf::pvFitSolve(fit, a, red)) {
                branch = vf::kCurvNoEstimate;
              } else {
                kappa = vf::paraboloidKappa(a);
                branch = red ? vf::kCurvPvReduced : vf::kCurvPv;
              }
            }
          }
        }
        ++cmp;
        if (kh(i) != kappa) {
          ++bitDiff;
          maxDiff = std::max(maxDiff, std::fabs(kh(i) - kappa));
        }
        CHECK(static_cast<int>(bh(i)) == branch);
      }
  std::printf("  %ld cells compared:  %ld bitwise identical, %ld differing (max |d| %.3e)\n", cmp,
              cmp - bitDiff, bitDiff, maxDiff);
  if (std::is_same<peclet::flow::SMem, Kokkos::HostSpace>::value) {
    CHECK(bitDiff == 0);
  } else {
    // Cross-backend (device kernel vs host loop) is a tolerance comparison by the suite policy
    // (`vof_workorders.md` shared preamble rule 2); WO-D measured the same 1e-14-class spread.
    CHECK(maxDiff < 1e-11);
  }
}

// ============================================= gate F: the fallback branch measured in isolation
void gateFallbackAlone() {
  std::printf("\n=== F  the PV fallback forced on EVERY interfacial cell (its own accuracy)\n");
  const int Ns[3] = {16, 32, 64};
  double l1[3];
  for (int q = 0; q < 3; ++q) {
    const int N = Ns[q];
    const double h = 1.0 / N, R = 0.3;
    Case cs;
    cs.setup(N, N, N, h);
    cs.curv.debugForceFallback = true;
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, R, 6);
    cs.adv.syncGhosts();
    const auto st = cs.curv.compute(cs.c());
    const Err er = curvError(cs, 2.0 * h / R);
    l1[q] = er.l1;
    std::printf("  N = %3d  C*dx = %.4f", N, h / R);
    std::printf("   PV %ld  PV-reduced %ld  none %ld   L1 %.4e  max %.4e\n", st.pv, st.pvReduced,
                st.noEstimate, er.l1, er.max);
    CHECK(st.noEstimate == 0);
    CHECK(er.nan == 0);
    CHECK(st.hf == 0 && st.hfMixed == 0 && st.hfFit == 0);
  }
  std::printf("  PV-only convergence order L1: %.2f, %.2f\n", order(l1[0], l1[1]),
              order(l1[1], l1[2]));
  CHECK(l1[2] < l1[0]);  // it must at least improve with refinement
}

// ================================= gate G: tier 2b (mixed height fit) on vs off, same geometry
void gateMixedHeightFitAblation() {
  std::printf("\n=== G  tier 2b (mixed height-position fit) ON vs OFF -- why it ships OFF\n");
  const int Ns[3] = {16, 32, 64};
  double l1[2][3], mx[2][3];
  for (int mode = 0; mode < 2; ++mode)
    for (int q = 0; q < 3; ++q) {
      const int N = Ns[q];
      const double h = 1.0 / N, R = 0.3;
      Case cs;
      cs.setup(N, N, N, h);
      cs.curv.useMixedHeightFit = (mode == 1);
      vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, R, 6);
      cs.adv.syncGhosts();
      cs.curv.compute(cs.c());
      const Err er = curvError(cs, 2.0 * h / R);
      l1[mode][q] = er.l1;
      mx[mode][q] = er.max;
    }
  for (int mode = 0; mode < 2; ++mode)
    std::printf("  tier 2b %-3s  L1 %.3e / %.3e / %.3e (order %.2f)   max %.3e / %.3e / %.3e"
                " (order %.2f)\n",
                mode ? "ON" : "OFF", l1[mode][0], l1[mode][1], l1[mode][2],
                order(l1[mode][0], l1[mode][2]) / 2.0, mx[mode][0], mx[mode][1], mx[mode][2],
                order(mx[mode][0], mx[mode][2]) / 2.0);
  std::printf("  the fit serves the cells tier 1 cannot; its data set is the columns that CLOSED,"
              " a slope-selected\n  and therefore asymmetric subset, so its bias is scale"
              " invariant. The PV fallback's 5^3 PLIC\n  polygons exist at every slope and are"
              " not.\n");
  // The shipped default must beat the alternative in the max norm, and must converge there.
  CHECK(order(mx[0][0], mx[0][2]) / 2.0 > 1.5);
  CHECK(order(mx[1][0], mx[1][2]) / 2.0 < 1.0);
  CHECK(mx[0][2] < 0.25 * mx[1][2]);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("VoF rung V3 (WO-O) - HF curvature cascade + PV fallback, backend: %s\n",
                peclet::flow::SExec::name());
    gateGeometry();
    gateShapes();
    gateStaticSphere();
    gateSweep();
    gateTranslating();
    gateDeviceHost();
    gateFallbackAlone();
    gateMixedHeightFitAblation();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
