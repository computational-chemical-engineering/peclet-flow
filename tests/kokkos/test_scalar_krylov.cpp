// The cut-cell scalar BiCGStab (doc/scalar_ibm_design.md §5.1; src/scalar_krylov.hpp): the
// device-resident pass (WO-9b) against the host-scalar pass, bitwise — the iterate, the iteration
// count and the true residual — on a non-symmetric periodic advection-diffusion operator, one field
// and two coupled fields, on a healthy solve and on every breakdown exit (a non-finite (rh, r),
// (rh, v) or (t, t), and omega = 0), which no healthy problem reaches: the reductions are poisoned
// at a chosen call, identically for both passes.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++g_fail;                                                                          \
    }                                                                                    \
  } while (0)

namespace {
int g_fail = 0;
using namespace peclet::flow;

constexpr int N = 12, G = 2;
const C3 E{N + 2 * G, N + 2 * G, N + 2 * G};
const std::size_t NE = (std::size_t)E.x * E.y * E.z;

/// y = A x on the inner cells, periodic over the inner block (no ghosts read): A x = m x_i +
/// sum_f (d + up_f) (x_i - x_nb) with a fixed non-uniform upwind coefficient — an M-matrix,
/// non-symmetric.
void applyOp(CCField y, CCConst x, double m, double d) {
  ccFor3(
      "test_skr_apply", C3{G, G, G}, C3{E.x - G, E.y - G, E.z - G},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const int xm = (lx - G + N - 1) % N + G, xp = (lx - G + 1) % N + G;
        const int ym = (ly - G + N - 1) % N + G, yp = (ly - G + 1) % N + G;
        const int zm = (lz - G + N - 1) % N + G, zp = (lz - G + 1) % N + G;
        const long sy = E.x, sz = (long)E.x * E.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const long nb[6] = {xm + ly * sy + lz * sz, xp + ly * sy + lz * sz, lx + ym * sy + lz * sz,
                            lx + yp * sy + lz * sz, lx + ly * sy + zm * sz, lx + ly * sy + zp * sz};
        const double u = 0.7 + 0.3 * Kokkos::sin(0.5 * ly);  // inflow from -x, varying in y
        const double up[6] = {u, 0.0, 0.2, 0.0, 0.0, 0.1};
        double s = m * x(i);
        for (int f = 0; f < 6; ++f)
          s += (d + up[f]) * (x(i) - x(nb[f]));
        y(i) = s;
      });
}
/// The same-cell coupling of the two fields: y_f -= w x_s, y_s -= w x_f.
void couple(CCField yf, CCField ys, CCConst xf, CCConst xs, double w) {
  ccFor3(
      "test_skr_couple", C3{G, G, G}, C3{E.x - G, E.y - G, E.z - G},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * E.x + (long)lz * (long)E.x * E.y;
        yf(i) -= w * xs(i);
        ys(i) -= w * xf(i);
      });
}
/// Jacobi, a fixed linear operator: z = r / diag, per phase.
void jacobi(CCField zf, CCField zs, CCConst rf, CCConst rs, double df, double ds, bool two) {
  ccFor3(
      "test_skr_jacobi", C3{G, G, G}, C3{E.x - G, E.y - G, E.z - G},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * E.x + (long)lz * (long)E.x * E.y;
        zf(i) = df * rf(i);
        if (two)
          zs(i) = ds * rs(i);
      });
}
/// 0 on the ghost cells (the right-hand side lives on the inner block).
void zeroGhosts(CCField f) {
  ccFor3(
      "test_skr_zero_ghosts", C3{0, 0, 0}, E, KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= G && lx < E.x - G && ly >= G && ly < E.y - G && lz >= G && lz < E.z - G)
          return;
        f((long)lx + (long)ly * E.x + (long)lz * (long)E.x * E.y) = 0.0;
      });
}

struct Run {
  std::vector<double> xf, xs;
  int its = 0;
  double trueRes = 0.0;
  bool restarted = false;
};

/// One solve; `poisonDot` / `poisonDot2` = the 1-based call of dot / dot2 whose result becomes
/// `poison` (0: none). dot2's poison lands on (t, t) when `onTt`, else on (t, r).
Run solve(bool resident, bool two, int poisonDot, int poisonDot2, double poison, bool onTt) {
  auto mk = [](const char* n) { return CCField(n, NE); };
  ScalarVec b, x, r, rh, p, v, t, z, z2;
  ScalarVec* all[9] = {&b, &x, &r, &rh, &p, &v, &t, &z, &z2};
  for (ScalarVec* s : all) {
    s->f = mk("f");
    if (two) {
      s->s = mk("s");
      s->solid = true;
    }
  }
  {  // a fixed right-hand side, x0 = 0
    auto hb = Kokkos::create_mirror_view(b.f);
    for (std::size_t k = 0; k < NE; ++k)
      hb(k) = std::cos(1.3 * (double)k) + 0.25;
    Kokkos::deep_copy(b.f, hb);
    if (two)
      for (std::size_t k = 0; k < NE; ++k)
        hb(k) = std::sin(0.7 * (double)k);
    if (two)
      Kokkos::deep_copy(b.s, hb);
    zeroGhosts(b.f);
    if (two)
      zeroGhosts(b.s);
  }
  const double m = 1.0, d = 1.0, ms = 0.3, dsd = 2.0, w = 0.4;
  ScalarKrylovOps ops;
  ops.e = E;
  ops.g = G;
  ops.matvec = [&](const ScalarVec& y, const ScalarVec& xx) {
    applyOp(y.f, CCConst(xx.f), m, d);
    if (!two)
      return;
    applyOp(y.s, CCConst(xx.s), ms, dsd);
    couple(y.f, y.s, CCConst(xx.f), CCConst(xx.s), w);
  };
  ops.precond = [&](const ScalarVec& zz, const ScalarVec& rr) {
    jacobi(zz.f, zz.s, CCConst(rr.f), CCConst(rr.s), 1.0 / (m + 6.0 * d + 1.0),
           1.0 / (ms + 6.0 * dsd + 1.0), two);
  };
  int nDot = 0, nDot2 = 0;
  auto poisonHost = [&](double& val, int& n, int at) {
    if (++n == at)
      val = poison;
  };
  auto poisonSlot = [&](skr::Slot s, int& n, int at) {
    if (++n == at)
      Kokkos::deep_copy(s, poison);
  };
  ops.dot = [&](const ScalarVec& a, const ScalarVec& bb) {
    double s = two ? sco::dotTwoLocal(CCConst(a.f), CCConst(bb.f), CCConst(a.s), CCConst(bb.s), E, G)
                   : sco::dotLocal(CCConst(a.f), CCConst(bb.f), E, G);
    poisonHost(s, nDot, poisonDot);
    return s;
  };
  ops.dot2 = [&](const ScalarVec& a, const ScalarVec& bb, const ScalarVec& c, double& ab,
                 double& cc) {
    if (two)
      sco::dot2TwoLocal(CCConst(a.f), CCConst(bb.f), CCConst(c.f), CCConst(a.s), CCConst(bb.s),
                        CCConst(c.s), E, G, ab, cc);
    else
      sco::dot2Local(CCConst(a.f), CCConst(bb.f), CCConst(c.f), E, G, ab, cc);
    poisonHost(onTt ? cc : ab, nDot2, poisonDot2);
  };
  ops.maxabs = [&](const ScalarVec& a) {
    const double mf = sco::maxabsLocal(CCConst(a.f), E, G);
    return two ? std::fmax(mf, sco::maxabsLocal(CCConst(a.s), E, G)) : mf;
  };
  ops.removeMean = [](const ScalarVec&) {};
  struct Slots {
    Kokkos::View<double*, CCMem> ks;
    Kokkos::View<double*, Kokkos::HostSpace> pk;
  } st;
  if (resident) {
    residentKrylov(st, ops);
    ops.dotTo = [&](const ScalarVec& a, const ScalarVec& bb, skr::Slot s) {
      if (two)
        sco::dotTwoReduce(CCConst(a.f), CCConst(bb.f), CCConst(a.s), CCConst(bb.s), E, G, s);
      else
        sco::dotReduce(CCConst(a.f), CCConst(bb.f), E, G, s);
      poisonSlot(s, nDot, poisonDot);
    };
    ops.dot2To = [&](const ScalarVec& a, const ScalarVec& bb, const ScalarVec& c, skr::Slot s1,
                     skr::Slot s2) {
      if (two)
        sco::dot2TwoReduce(CCConst(a.f), CCConst(bb.f), CCConst(c.f), CCConst(a.s), CCConst(bb.s),
                           CCConst(c.s), E, G, s1, s2);
      else
        sco::dot2Reduce(CCConst(a.f), CCConst(bb.f), CCConst(c.f), E, G, s1, s2);
      poisonSlot(onTt ? s2 : s1, nDot2, poisonDot2);
    };
    ops.maxabsTo = [&](const ScalarVec& a, skr::Slot sf, skr::Slot ss) {
      sco::maxabsReduce(CCConst(a.f), E, G, Kokkos::Max<double, CCMem>(sf));
      if (two)
        sco::maxabsReduce(CCConst(a.s), E, G, Kokkos::Max<double, CCMem>(ss));
    };
  }
  const ScalarKrylovResult res =
      scalarBiCGStab(ops, b, x, r, rh, p, v, t, z, z2, 200, 1e-12, false);
  Run out;
  out.its = res.iterations;
  out.trueRes = res.trueRes;
  out.restarted = res.restarted;
  auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.f);
  out.xf.assign(hx.data(), hx.data() + NE);
  if (two) {
    auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.s);
    out.xs.assign(hs.data(), hs.data() + NE);
  }
  return out;
}

void compare(const char* tag, bool two, int pd, int pd2, double poison, bool onTt) {
  const Run h = solve(false, two, pd, pd2, poison, onTt);
  const Run d = solve(true, two, pd, pd2, poison, onTt);
  const bool same = h.xf.size() == d.xf.size() &&
                    std::memcmp(h.xf.data(), d.xf.data(), h.xf.size() * sizeof(double)) == 0 &&
                    h.xs.size() == d.xs.size() &&
                    (h.xs.empty() ||
                     std::memcmp(h.xs.data(), d.xs.data(), h.xs.size() * sizeof(double)) == 0);
  const bool resSame = std::memcmp(&h.trueRes, &d.trueRes, sizeof(double)) == 0;
  std::printf("  %-34s %s: iterations host %3d resident %3d, true residual %.3e / %.3e%s, iterate %s\n",
              tag, two ? "two fields" : "one field ", h.its, d.its, h.trueRes, d.trueRes,
              h.restarted ? " (restarted)" : "", same ? "bitwise" : "DIFFERS");
  CHECK(same);
  CHECK(h.its == d.its);
  CHECK(resSame);
  CHECK(h.restarted == d.restarted);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::printf("scalarBiCGStab: the resident pass against the host-scalar pass (%d^3, g = %d)\n", N,
                G);
    for (const bool two : {false, true}) {
      compare("healthy", two, 0, 0, 0.0, true);
      compare("(rh, r) NaN at iteration 3", two, 5, 0, nan, true);  // dot calls: 2 per iteration
      compare("(rh, v) NaN at iteration 2", two, 4, 0, nan, true);
      compare("(rh, r) = 0 at iteration 2", two, 3, 0, 0.0, true);
      compare("(t, t) NaN at iteration 2", two, 0, 2, nan, true);
      compare("(t, t) = 0 at iteration 1", two, 0, 1, 0.0, true);
      compare("omega = 0 ((t, r) = 0) at iteration 3", two, 0, 3, 0.0, false);
    }
  }
  Kokkos::finalize();
  if (g_fail) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
