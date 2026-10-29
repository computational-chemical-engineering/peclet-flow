// VoF momentum flux at second order (doc/vof_momentum_second_order.md, WO-3 (c)): the device
// advector (`WyAdvector` + `MomentumConsistentAdvector`, no solver, no projection) against the 1D
// a-priori operator test `tests/study/vof_mom2/apriori1d.py`.
//
//   G-M2-7  TRANSVERSE-SLAB ORACLE. x-faces carry U = 1, y-faces v(x) = 0.1 v0(x), z-faces 0, with
//           v0(s) = sin 2 pi s + 0.5 cos 6 pi s; C is a y,z-uniform liquid slab [0.3, 0.7);
//           ny = nz = 4, periodic, a_x = U dt/h = 0.2, one period (5 N steps). The field
//           (U, v(x), 0) reduces the step EXACTLY to the 1D x-sweep: the y-sweep sees y-uniform
//           data (equal fluxes, zero dilation, zero dv) and the z-sweep has a = 0. After each step
//           the advected velocity is written back as the next step's face velocity (the transported
//           v is the state). Criteria (pre-registered, §9): the L1 error of v within 1 % of
//           apriori1d.py's at every (N, ratio); order 64 -> 128 >= 1.6 at ratio 1e3;
//           L1(1e4)/L1(1e3) in [0.99, 1.01]; capped == 0; positive control: 'muscl_mixed' at 1e4,
//           N = 128 has L1 >= 3x 'muscl_phase''s.
//   G-M2-9  SINGLE-PHASE INACTIVITY. C == 1 and C == 0, a non-uniform divergence-free velocity,
//           ratio 1e3: the cap never bites (capped == 0 every step, §4.6).
//   ID      The uniform-velocity identity on the device advector, every scheme, ratios 1, 1e3,
//           1e4, a sharp sphere: max|u - U| == 0.0 bitwise (G-M2-1's direct-advector twin).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

#include "vof/colour_field.hpp"
#include "vof/momentum_advect.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
namespace vf = peclet::flow::vof;
using MS = vf::MomentumScheme;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

const char* schemeName(MS s) {
  switch (s) {
    case MS::Upwind:
      return "upwind";
    case MS::MusclMixed:
      return "muscl_mixed";
    case MS::MusclMixedCapped:
      return "muscl_mixed_capped";
    case MS::MusclPhase:
      return "muscl_phase";
  }
  return "?";
}

double v0(double s) {
  return std::sin(2 * M_PI * s) + 0.5 * std::cos(6 * M_PI * s);
}

// A periodic g = 3 advector of nx*ny*nz unit cells (h = 1: dt in cell units is dt/h).
struct Box {
  vf::WyAdvector w;
  vf::MomentumConsistentAdvector mom;
  I3 n, e;
  int g = 3;
  Box(int nx, int ny, int nz, double rhoG, double rhoL, MS scheme) {
    w.init(nx, ny, nz, 1.0, 3);
    n = w.inner();
    e = w.extent();
    g = w.ghost();
    const I3 ee = e;
    const int gg = g;
    w.exchange = [ee, gg](SField f) { vf::periodicFill(f, ee, gg, true, true, true); };
    mom.init(w, rhoG, rhoL);
    mom.momentumScheme = scheme;
  }
  // Inner values from f(x, y, z) (inner indices), then the periodic ghosts.
  template <class Fn>
  void put(SField dst, Fn f) {
    auto h = Kokkos::create_mirror_view(dst);
    Kokkos::deep_copy(h, dst);
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          h(L3(g + x, g + y, g + z, e)) = f(x, y, z);
    Kokkos::deep_copy(dst, h);
    w.exchange(dst);
  }
  std::vector<double> inner(SField src) const {
    auto h = Kokkos::create_mirror_view(src);
    Kokkos::deep_copy(h, src);
    std::vector<double> v((std::size_t)n.x * n.y * n.z);
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x)
          v[x + (std::size_t)n.x * (y + (std::size_t)n.y * z)] = h(L3(g + x, g + y, g + z, e));
    return v;
  }
  // The advected velocity becomes the next step's face velocity. Control volume i of component c
  // carries the advector face uf_c(i - s_c) (momentum_advect.hpp "Indexing"), so the inner face j
  // takes CV j + s_c, wrapped periodically.
  void feedBack() {
    for (int c = 0; c < 3; ++c) {
      const std::vector<double> u = inner(mom.advectedVelocity(c));
      put(w.faceVel(c), [&](int x, int y, int z) {
        int i[3] = {x, y, z};
        const int nn[3] = {n.x, n.y, n.z};
        i[c] = (i[c] + 1) % nn[c];
        return u[i[0] + (std::size_t)n.x * (i[1] + (std::size_t)n.y * i[2])];
      });
    }
  }
};

struct SlabResult {
  double l1 = 0.0;
  long capped = 0, clamped = 0, floored = 0;
  long diverged = -1;  ///< the step whose Courant check threw (the velocity blew up), else -1
};

// One period of the transverse slab at resolution N, density ratio `ratio` (rho_g = 1).
SlabResult slab(int N, double ratio, MS scheme) {
  Box b(N, 4, 4, 1.0, ratio, scheme);
  const double h = 1.0 / N, A = 0.2;
  b.put(b.w.colour(), [&](int x, int, int) {  // apriori1d.py's exact cell fractions
    const double xl = x * h;
    const double c = (std::fmin(xl + h, 0.7) - std::fmax(xl, 0.3)) / h;
    return c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
  });
  b.put(b.w.faceVel(0), [](int, int, int) { return 1.0; });
  b.put(b.w.faceVel(1), [&](int x, int, int) { return 0.1 * v0((x + 0.5) * h); });
  b.put(b.w.faceVel(2), [](int, int, int) { return 0.0; });
  const long nst = std::lround(1.0 / (A * h));
  SlabResult r;
  for (long k = 0; k < nst; ++k) {
    try {
      b.mom.advect(b.w, A, k);  // U = 1, h = 1: a_x = U dt/h = 0.2
    } catch (const std::runtime_error& ex) {
      // WyAdvector's Courant check: the transported v (|v0| <= 0.15) left the CFL cap, so the
      // one-period L1 does not exist; recorded as unbounded (only the positive control is expected
      // here: the mixed slope's stayer kick of note §2.2 is O(1) at ratio 1e4).
      std::printf("  [%s ratio %g N %d] step %ld: %s\n", schemeName(scheme), ratio, N, k,
                  ex.what());
      r.diverged = k;
      r.l1 = INFINITY;
      return r;
    }
    r.capped += b.mom.cappedFaces();
    r.clamped += b.mom.clampedFluxes();
    r.floored += b.mom.diagnostics().floored;
    b.feedBack();
  }
  const std::vector<double> v = b.inner(b.mom.advectedVelocity(1));
  double s = 0.0;
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < N; ++x) {
        const double xc = (x + 0.5) * h;
        s += std::fabs(v[x + (std::size_t)N * (y + 4 * z)] - 0.1 * v0(xc - nst * A / N));
      }
  r.l1 = s / (16.0 * N);
  return r;
}

// apriori1d.py, `run(N, ratio, 'muscl_phase', lambda s: 0.1 * v0(s))` (wisp 0, as here), L1 of v
// after one period; regenerate with
//   python -I -c "import apriori1d as A, numpy as np; v=lambda s: 0.1*A.v0(s); \
//     x,C,w,n,b=A.run(N,R,'muscl_phase',v); print(np.abs(w-v(x-n*A.A/N)).mean())"
// run from tests/study/vof_mom2.
constexpr int kNs[3] = {32, 64, 128};
constexpr double kRatios[3] = {1.0, 1e3, 1e4};
constexpr double kApriori[3][3] = {{1.1480651489e-02, 3.3904780129e-03, 1.0287161683e-03},
                                   {1.2372109717e-02, 3.9895354338e-03, 1.2389030843e-03},
                                   {1.2377683982e-02, 3.9924772277e-03, 1.2398866240e-03}};

void transverseSlab() {
  double l1[3][3];
  long capped = 0;
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k) {
      const SlabResult s = slab(kNs[k], kRatios[r], MS::MusclPhase);
      l1[r][k] = s.l1;
      capped += s.capped;
      const double rel = s.l1 / kApriori[r][k] - 1.0;
      std::printf(
          "G-M2-7 muscl_phase ratio %-6g N %-4d L1 %.10e  apriori %.10e  rel %+.2e  capped %ld "
          "clamped %ld floored %ld\n",
          kRatios[r], kNs[k], s.l1, kApriori[r][k], rel, s.capped, s.clamped, s.floored);
      CHECK(std::fabs(rel) <= 0.01);
      CHECK(s.capped == 0);
      CHECK(s.floored == 0);
    }
  const double ord = std::log2(l1[1][1] / l1[1][2]);
  const double rr = l1[2][2] / l1[1][2];
  std::printf(
      "G-M2-7 order 64->128 at 1e3: %.3f (>= 1.6)   L1(1e4)/L1(1e3) at N 128: %.5f "
      "([0.99, 1.01])   capped total %ld\n",
      ord, rr, capped);
  CHECK(ord >= 1.6);
  CHECK(rr >= 0.99 && rr <= 1.01);
  const SlabResult ctl = slab(128, 1e4, MS::MusclMixed);
  std::printf(
      "G-M2-7 positive control: muscl_mixed ratio 1e4 N 128 L1 %.6e = %.2fx muscl_phase "
      "(>= 3)%s\n",
      ctl.l1, ctl.l1 / l1[2][2], ctl.diverged >= 0 ? "  (left the Courant cap)" : "");
  if (ctl.diverged >= 0)
    std::printf(
        "G-M2-7 positive control: v left the Weymouth-Yue Courant cap at step %ld of %d "
        "(the mixed slope's ratio-1e4 stayer gain, note §2.2)\n",
        ctl.diverged, 5 * 128);
  CHECK(ctl.l1 >= 3.0 * l1[2][2]);
}

// A non-uniform, discretely divergence-free field: each face component is independent of its own
// coordinate. Courant <= 0.2 at dt/h = 0.2.
double fieldU(int c, int x, int y, int z, int n) {
  const double t = 2 * M_PI / n;
  if (c == 0)
    return 0.5 + 0.4 * std::sin(t * (y + 0.5)) * std::cos(t * z);
  if (c == 1)
    return -0.3 + 0.5 * std::cos(t * (z + 0.5)) + 0.2 * std::sin(2 * t * x);
  return 0.2 + 0.6 * std::sin(t * (x + 0.5) + 0.3) * std::cos(t * y);
}

void singlePhase() {
  const int n = 16, steps = 20;
  for (MS sch : {MS::MusclPhase, MS::MusclMixedCapped})
    for (double cval : {1.0, 0.0}) {
      Box b(n, n, n, 1.0, 1e3, sch);
      b.put(b.w.colour(), [&](int, int, int) { return cval; });
      for (int c = 0; c < 3; ++c)
        b.put(b.w.faceVel(c), [&](int x, int y, int z) { return fieldU(c, x, y, z, n); });
      long worst = 0, total = 0;
      for (int k = 0; k < steps; ++k) {
        b.mom.advect(b.w, 0.2, k);
        worst = std::max(worst, b.mom.cappedFaces());
        total += b.mom.cappedFaces();
        b.feedBack();
      }
      std::printf(
          "G-M2-9 %-18s C == %.0f ratio 1e3, %d steps: capped per step max %ld, total %ld\n",
          schemeName(sch), cval, steps, worst, total);
      if (sch == MS::MusclPhase)
        CHECK(total == 0);
    }
}

void identity() {
  const int n = 16;
  const double U[3] = {1.0, 0.6, -0.4};
  for (MS sch : {MS::Upwind, MS::MusclMixed, MS::MusclMixedCapped, MS::MusclPhase})
    for (double R : {1.0, 1e3, 1e4}) {
      Box b(n, n, n, 1.0, R, sch);
      b.put(b.w.colour(), [&](int x, int y, int z) {
        return vf::sphereCellFraction(0.5 * n, 0.5 * n, 0.5 * n, 0.3 * n, x, y, z, 1.0, 4);
      });
      for (int c = 0; c < 3; ++c)
        b.put(b.w.faceVel(c), [&](int, int, int) { return U[c]; });
      double dev = 0.0;
      for (int k = 0; k < 3; ++k) {
        b.mom.advect(b.w, 0.2, k);
        for (int c = 0; c < 3; ++c)
          for (double v : b.inner(b.mom.advectedVelocity(c)))
            dev = std::fmax(dev, std::fabs(v - U[c]));
      }
      std::printf("ID %-18s ratio %-6g 3 steps: max|u - U| = %.17g%s\n", schemeName(sch), R, dev,
                  dev == 0.0 ? "   (BITWISE)" : "");
      CHECK(dev == 0.0);
    }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Kokkos::initialize(argc, argv);
  {
    identity();
    singlePhase();
    transverseSlab();
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
