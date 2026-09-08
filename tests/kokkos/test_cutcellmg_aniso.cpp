// G4 — the ANISOTROPIC pressure hierarchy: the OPERATOR converges at the cubic rate (commit C2)
// and the aspect-ratio COARSENING RULE gives it the cubic RATE too (commit C3)
// (flow/doc/anisotropic_metric.md §8.5).
//
// `CutcellMG` is driven DIRECTLY (no Solver, no velocity) on a triply periodic all-fluid box, the
// way the pressure gates of this suite are written: the only thing under test is the operator the
// per-axis weights of §1.3 assemble and the multigrid that inverts it.
//
//   cells (N, 2N, N/2) on the CUBE L^3  ->  h = (dx, dx/2, 2dx),  dx = L/N
//   setOpenness(o = 1, w_a = (dx/h_a)^2) = (1, 4, 1/4)                       [the §8.5 numbers]
//
// so the assembled operator is A = -dx^2 * (the anisotropic 7-point Laplacian).  With the
// MANUFACTURED solution
//
//   phi = cos(2 pi x/L) cos(2 pi y/L) cos(2 pi z/L),   A phi = -dx^2 lap(phi) = 3 k^2 dx^2 phi
//
// evaluated at the physical cell centres and the CONTINUOUS right-hand side b = 3 k^2 dx^2 phi
// (k = 2 pi / L), the discrete solution is exactly (3 k^2 dx^2 / Lambda) phi with
// Lambda = sum_a w_a 2(1 - cos(k h_a)), i.e. the L2 error is
// |3k^2dx^2/Lambda - 1| ||phi||_2 = k^2 (hx^2 + hy^2 + hz^2)/36 + O(h^4) -- SECOND ORDER in the
// mesh, with the coarse z axis (h_z = 2 dx) dominating.  That is the statement of the gate:
// giving the coarse axis its own w_a is what keeps the anisotropic operator second order, and a
// missing or mis-assigned weight destroys the rate rather than merely the constant.
//
// GATE (ORDER, commit C2): least-squares order over N = 16, 32, 64 >= 1.95 on the stretched grid;
// the cubic control (N, N, N) with w = (1,1,1) gives ~2.00.
//
// COMMIT C3 adds the other three halves of §8.5, all about the COARSENING RULE of §5:
//
//   (level table)  levelRatios() on (N, 2N, N/2) with hp = (1, 1/2, 2) is
//                  (1,2,1), (2,2,1), (2,2,2), (2,2,2), ... -- always coarsen the FINEST
//                  coarsenable axis, defer one already >= theta times coarser -- and on (N, N, N)
//                  it is today's (2,2,2), ... unchanged.  The isotropic table is asserted to be
//                  BITWISE the no-metric one, which is the whole G0 statement of this commit.
//   (rate a)       all-fluid, random mean-zero RHS: the residual reduction per V-cycle over
//                  cycles 2-8 <= 0.2.
//   (rate b/c)     with the Zick & Homsy sphere openness (phi = 0.216, sampled SDF through the
//                  solver's own buildOpenness with hp), MG-PCG iterations to rtol 1e-10 on the
//                  stretched grid <= cubic (SAME cell count) + 2 at N = 32 and 64; and the same
//                  measurement under PECLET_FLOW_MG_ASPECT=1e9 (today's full coarsening) must be
//                  worse or equal.  That ablation is a SEPARATE RUN of this binary (the threshold
//                  is read once per process); with the env var set the gates below print instead
//                  of asserting and the run is a measurement, not a test.
//
// Both hierarchies share one rule (CutcellMG::mgChooseRatio, called by VelocityMG too), so this
// file gates the rule itself.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell.hpp"  // buildOpenness (the solver's own aperture model)
#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {
int failures = 0;

struct Run {
  double l2 = 0.0;       ///< L2 norm of (phi_h - phi_exact) over the inner cells
  double l2exact = 0.0;  ///< the same, from the exact modal solution of the discrete system
  double rrel = 0.0;     ///< achieved max|r| / max|b|
  int iters = 0;         ///< Krylov iterations
  int levels = 0;
};

/// One solve. `stretched` selects the (N, 2N, N/2) box with w = (1, 4, 1/4); otherwise the cubic
/// control (N, N, N) with w = (1, 1, 1). Both are the SAME physical cube of side L.
Run solveOne(int N, bool stretched, bool fcg = false) {
  const int G = CutcellMG::G;  // level-0 ghost width (1)
  const int nx = N, ny = stretched ? 2 * N : N, nz = stretched ? N / 2 : N;
  const double L = 1.0;
  const double hx = L / nx, hy = L / ny, hz = L / nz;
  const double dref = hx;  // the reference length the weights are built on
  const double wx = (dref / hx) * (dref / hx), wy = (dref / hy) * (dref / hy),
               wz = (dref / hz) * (dref / hz);
  const C3 e{nx + 2 * G, ny + 2 * G, nz + 2 * G};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  const double k = 2.0 * M_PI / L;

  CCField ox("ox", n), oy("oy", n), oz("oz", n);
  Kokkos::deep_copy(ox, 1.0);  // all fluid
  Kokkos::deep_copy(oy, 1.0);
  Kokkos::deep_copy(oz, 1.0);

  // b and the exact solution at the physical cell centres of the inner block.
  std::vector<double> hb(n, 0.0), hex(n, 0.0);
  for (int z = G; z < e.z - G; ++z)
    for (int y = G; y < e.y - G; ++y)
      for (int x = G; x < e.x - G; ++x) {
        const double px = ((x - G) + 0.5) * hx, py = ((y - G) + 0.5) * hy,
                     pz = ((z - G) + 0.5) * hz;
        const double ph = std::cos(k * px) * std::cos(k * py) * std::cos(k * pz);
        const std::size_t i =
            (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
        hex[i] = ph;
        hb[i] = 3.0 * k * k * dref * dref * ph;  // A phi = -dref^2 lap(phi)
      }
  CCField b("b", n), xf("x", n), r("r", n), p("p", n), zz("z", n), Ap("Ap", n);
  {
    auto m = Kokkos::create_mirror_view(b);
    for (std::size_t i = 0; i < n; ++i)
      m(i) = hb[i];
    Kokkos::deep_copy(b, m);
  }
  Kokkos::deep_copy(xf, 0.0);

  CutcellMG mg;
  // C3: the per-axis spacings BEFORE init (trap 5) -- hp = h_a/dref = (1, 1/2, 2) stretched,
  // (1,1,1) cubic.  The rule is scale-free (it compares H's), so this IS the note's hp.
  const double hp[3] = {hx / dref, hy / dref, hz / dref};
  mg.setMetric(hp);
  mg.init(nx, ny, nz, 6);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), wx, wy, wz);

  Run out;
  out.levels = mg.nLevels();
  if (fcg) {
    CCField zp("zp", n);
    out.iters = mg.solveFCG(b, xf, r, p, zz, zp, Ap, 500, 1e-10, 2, 2, 12);
  } else {
    out.iters = mg.solvePCG(b, xf, r, p, zz, Ap, 500, 1e-10, 2, 2, 12);
  }
  // The EXACT solution of the discrete system: phi = cos*cos*cos is an eigenvector of the weighted
  // 7-point operator with eigenvalue Lambda = sum_a w_a 2(1 - cos(k h_a)), so
  // phi_h = (3 k^2 dref^2 / Lambda) phi and the L2 error is |3 k^2 dref^2 / Lambda - 1| * ||phi||_2
  // with ||phi||_2 = (1/2)^(3/2) exactly (a full period on every axis).  Printed beside the
  // measured one: it separates a discretization statement from an unconverged solve.
  {
    const double lam = wx * 2.0 * (1.0 - std::cos(k * hx)) + wy * 2.0 * (1.0 - std::cos(k * hy)) +
                       wz * 2.0 * (1.0 - std::cos(k * hz));
    out.l2exact = std::fabs(3.0 * k * k * dref * dref / lam - 1.0) * std::pow(0.5, 1.5);
  }
  {  // the residual solvePCG/solveFCG left behind, relative to |b|inf
    auto hr = Kokkos::create_mirror_view(r);
    Kokkos::deep_copy(hr, r);
    double rm = 0.0, bm = 0.0;
    for (int z = G; z < e.z - G; ++z)
      for (int y = G; y < e.y - G; ++y)
        for (int x = G; x < e.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
          if (!(std::fabs(hr(i)) <= rm))
            rm = std::fabs(hr(i));
          if (!(std::fabs(hb[i]) <= bm))
            bm = std::fabs(hb[i]);
        }
    out.rrel = bm > 0.0 ? rm / bm : rm;
  }

  auto hxv = Kokkos::create_mirror_view(xf);
  Kokkos::deep_copy(hxv, xf);
  // Both fields are defined up to a constant on a periodic box (the solver removes the mean of
  // its own iterate); compare them mean-free.
  double ma = 0.0, mb = 0.0;
  long cnt = 0;
  for (int z = G; z < e.z - G; ++z)
    for (int y = G; y < e.y - G; ++y)
      for (int x = G; x < e.x - G; ++x) {
        const std::size_t i =
            (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
        ma += hxv(i);
        mb += hex[i];
        ++cnt;
      }
  ma /= (double)cnt;
  mb /= (double)cnt;
  double acc = 0.0;
  for (int z = G; z < e.z - G; ++z)
    for (int y = G; y < e.y - G; ++y)
      for (int x = G; x < e.x - G; ++x) {
        const std::size_t i =
            (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
        const double d = (hxv(i) - ma) - (hex[i] - mb);
        acc += d * d;
      }
  out.l2 = std::sqrt(acc / (double)cnt);
  return out;
}

/// Least-squares slope of log(err) against log(N), negated -> the observed order.
double fitOrder(const std::vector<int>& Ns, const std::vector<double>& err) {
  const int m = (int)Ns.size();
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i) {
    const double X = std::log((double)Ns[i]), Y = std::log(err[i]);
    sx += X;
    sy += Y;
    sxx += X * X;
    sxy += X * Y;
  }
  return -(m * sxy - sx * sy) / (m * sxx - sx * sx);
}

void ladder(bool stretched, bool fcg, double minOrder) {
  const std::vector<int> Ns{16, 32, 64};
  std::vector<double> err;
  std::printf("  %s, %s:\n",
              stretched ? "stretched (N, 2N, N/2), w = (1, 4, 1/4)"
                        : "cubic control (N, N, N), w = (1, 1, 1)",
              fcg ? "FCG (the converged control)" : "MG-PCG (the §8.5 driver)");
  for (int N : Ns) {
    const Run rr = solveOne(N, stretched, fcg);
    err.push_back(rr.l2);
    std::printf(
        "    N = %3d  cells %4d x %4d x %4d  levels %d  %3d iters  r/|b| %.2e  "
        "L2 err %.6e  (exact discrete %.6e)",
        N, N, stretched ? 2 * N : N, stretched ? N / 2 : N, rr.levels, rr.iters, rr.rrel, rr.l2,
        rr.l2exact);
    if (err.size() > 1)
      std::printf("   order %.4f", std::log(err[err.size() - 2] / err.back()) / std::log(2.0));
    std::printf("\n");
  }
  const double ord = fitOrder(Ns, err);
  std::printf("    least-squares order = %.4f  (require >= %.2f)\n", ord, minOrder);
  if (!(ord >= minOrder)) {
    std::fprintf(stderr, "FAIL: observed order %.4f < %.2f\n", ord, minOrder);
    ++failures;
  }
}

// ---------------------------------------------------------------------------------------------
// C3 — the aspect-ratio coarsening rule of doc/anisotropic_metric.md §5
// ---------------------------------------------------------------------------------------------

/// Is this run the PECLET_FLOW_MG_ASPECT ablation (§8.5 item (c))?  The threshold is read once per
/// process, so "today's full coarsening" is a separate INVOCATION of this binary; in it the gates
/// print their numbers instead of asserting.
bool ablation() {
  return mgAspectTheta() != 2.0;
}

/// The level table a hierarchy of `levels` levels builds on (nx, ny, nz) with spacings `hp`.
/// `withMetric == false` reproduces the pre-C3 call sequence exactly (no setMetric at all).
std::vector<C3> levelTable(int nx, int ny, int nz, const double hp[3], int levels,
                           bool withMetric) {
  CutcellMG mg;
  if (withMetric)
    mg.setMetric(hp);
  mg.init(nx, ny, nz, levels);
  return mg.levelRatios();
}

void printTable(const char* what, const std::vector<C3>& t) {
  std::printf("    %-46s", what);
  for (const C3& r : t)
    std::printf(" (%d,%d,%d)", r.x, r.y, r.z);
  std::printf("\n");
}

bool sameTable(const std::vector<C3>& a, const std::vector<C3>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z)
      return false;
  return true;
}

void levelTables() {
  std::printf("  LEVEL TABLE (§8.5, C3), levels = 6, theta = %.4g:\n", mgAspectTheta());
  const double hpS[3] = {1.0, 0.5, 2.0};  // the stretched (N, 2N, N/2) box on a cube
  const double hpC[3] = {1.0, 1.0, 1.0};  // the cubic control
  const C3 want[4] = {{1, 2, 1}, {2, 2, 1}, {2, 2, 2}, {2, 2, 2}};
  for (int N : {16, 32, 64}) {
    const std::vector<C3> st = levelTable(N, 2 * N, N / 2, hpS, 6, true);
    const std::vector<C3> stOld = levelTable(N, 2 * N, N / 2, hpS, 6, false);
    const std::vector<C3> cu = levelTable(N, N, N, hpC, 6, true);
    const std::vector<C3> cuOld = levelTable(N, N, N, hpC, 6, false);
    std::printf("   N = %3d\n", N);
    printTable("stretched (N, 2N, N/2), hp = (1, 1/2, 2)", st);
    printTable("  the same WITHOUT the metric (today's rule)", stOld);
    printTable("cubic (N, N, N), hp = (1, 1, 1)", cu);
    printTable("  the same WITHOUT the metric", cuOld);
    if (ablation())
      continue;
    for (int L = 0; L < 4; ++L)
      if (!(st[(std::size_t)L].x == want[L].x && st[(std::size_t)L].y == want[L].y &&
            st[(std::size_t)L].z == want[L].z)) {
        std::fprintf(stderr,
                     "FAIL: N = %d stretched level %d ratio (%d,%d,%d), expected (%d,%d,%d)\n", N,
                     L, st[(std::size_t)L].x, st[(std::size_t)L].y, st[(std::size_t)L].z, want[L].x,
                     want[L].y, want[L].z);
        ++failures;
      }
    // The ISOTROPIC hierarchy must be today's BY CONSTRUCTION -- this is the G0 statement of C3.
    if (!sameTable(cu, cuOld)) {
      std::fprintf(stderr, "FAIL: N = %d cubic level table moved with the metric set\n", N);
      ++failures;
    }
    for (std::size_t L = 0; L + 1 < cu.size(); ++L)
      if (!(cu[L].x == 2 && cu[L].y == 2 && cu[L].z == 2)) {
        std::fprintf(stderr, "FAIL: N = %d cubic level %zu is not (2,2,2)\n", N, L);
        ++failures;
      }
  }
}

/// Host residual max|b - A x| of the level-0 band, inner cells, periodic wrap in INNER index
/// space (the all-fluid periodic box this file drives).  Independent of whatever the V-cycle left
/// in the ghosts.
double residualInf(CutcellMG& mg, const std::vector<double>& b) {
  CutcellMG::Level& l0 = mg.level(0);
  const C3 e = l0.ext;
  const int G = CutcellMG::G;
  const int nx = l0.inner.x, ny = l0.inner.y, nz = l0.inner.z;
  auto hx = Kokkos::create_mirror_view(l0.x);
  Kokkos::deep_copy(hx, l0.x);
  auto hAC = Kokkos::create_mirror_view(l0.AC);
  auto hAW = Kokkos::create_mirror_view(l0.AW);
  auto hAE = Kokkos::create_mirror_view(l0.AE);
  auto hAS = Kokkos::create_mirror_view(l0.AS);
  auto hAN = Kokkos::create_mirror_view(l0.AN);
  auto hAB = Kokkos::create_mirror_view(l0.AB);
  auto hAT = Kokkos::create_mirror_view(l0.AT);
  Kokkos::deep_copy(hAC, l0.AC);
  Kokkos::deep_copy(hAW, l0.AW);
  Kokkos::deep_copy(hAE, l0.AE);
  Kokkos::deep_copy(hAS, l0.AS);
  Kokkos::deep_copy(hAN, l0.AN);
  Kokkos::deep_copy(hAB, l0.AB);
  Kokkos::deep_copy(hAT, l0.AT);
  auto id = [&](int i, int j, int k) {
    const int wi = (i % nx + nx) % nx, wj = (j % ny + ny) % ny, wk = (k % nz + nz) % nz;
    return (std::size_t)(wi + G) + (std::size_t)(wj + G) * e.x +
           (std::size_t)(wk + G) * (std::size_t)e.x * e.y;
  };
  double m = 0.0;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const std::size_t c = id(i, j, k);
        const double Ax =
            (double)hAC(c) * hx(c) + (double)hAW(c) * hx(id(i - 1, j, k)) +
            (double)hAE(c) * hx(id(i + 1, j, k)) + (double)hAS(c) * hx(id(i, j - 1, k)) +
            (double)hAN(c) * hx(id(i, j + 1, k)) + (double)hAB(c) * hx(id(i, j, k - 1)) +
            (double)hAT(c) * hx(id(i, j, k + 1));
        const double r = b[c] - Ax;
        if (!(std::fabs(r) <= m))
          m = std::fabs(r);
      }
  return m;
}

/// §8.5 rate (a): the STANDALONE V-cycle on the stretched all-fluid box with a random mean-zero
/// RHS.  The reduction factor over cycles 2..8 is the asymptotic rate of the preconditioner the
/// two CG drivers wrap -- what §5.2 predicts the coarsening rule fixes.
void vcycleRate(int N, double maxFactor) {
  const int G = CutcellMG::G;
  const int nx = N, ny = 2 * N, nz = N / 2;
  const C3 e{nx + 2 * G, ny + 2 * G, nz + 2 * G};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  CCField ox("ox", n), oy("oy", n), oz("oz", n);
  Kokkos::deep_copy(ox, 1.0);
  Kokkos::deep_copy(oy, 1.0);
  Kokkos::deep_copy(oz, 1.0);
  CutcellMG mg;
  const double hp[3] = {1.0, 0.5, 2.0};
  mg.setMetric(hp);
  mg.init(nx, ny, nz, 6);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 4.0, 0.25);

  // Deterministic pseudo-random, mean-free over the inner cells (the operator's null space is the
  // constants on a periodic all-fluid box, so the RHS must be compatible).
  std::vector<double> hb(n, 0.0);
  unsigned int st = 12345u;
  double sum = 0.0;
  long cnt = 0;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        st = st * 1103515245u + 12345u;
        const double v = (double)((st >> 16) & 0x7fff) / 32767.0 - 0.5;
        hb[(std::size_t)(i + G) + (std::size_t)(j + G) * e.x +
           (std::size_t)(k + G) * (std::size_t)e.x * e.y] = v;
        sum += v;
        ++cnt;
      }
  const double mean = sum / (double)cnt;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        hb[(std::size_t)(i + G) + (std::size_t)(j + G) * e.x +
           (std::size_t)(k + G) * (std::size_t)e.x * e.y] -= mean;

  CutcellMG::Level& l0 = mg.level(0);
  {
    auto m = Kokkos::create_mirror_view(l0.rhs);
    for (std::size_t i = 0; i < n; ++i)
      m(i) = hb[i];
    Kokkos::deep_copy(l0.rhs, m);
  }
  Kokkos::deep_copy(l0.x, 0.0);
  std::printf(
      "  V-CYCLE RATE (§8.5 rate (a)), stretched (%d, %d, %d), 2/2 sweeps, random "
      "mean-zero RHS:\n",
      nx, ny, nz);
  double prev = 0.0, worst = 0.0;
  for (int c = 1; c <= 8; ++c) {
    mg.vcycle(0, /*sym=*/true);
    const double rn = residualInf(mg, hb);
    const double f = (c > 1 && prev > 0.0) ? rn / prev : 0.0;
    if (c >= 2) {
      std::printf("    cycle %d  max|r| %.6e   factor %.4f\n", c, rn, f);
      if (!(f <= worst))
        worst = f;
    } else {
      std::printf("    cycle %d  max|r| %.6e\n", c, rn);
    }
    prev = rn;
  }
  std::printf("    worst reduction factor over cycles 2-8 = %.4f  (require <= %.2f)\n", worst,
              maxFactor);
  if (!ablation() && !(worst <= maxFactor)) {
    std::fprintf(stderr, "FAIL: V-cycle reduction factor %.4f > %.2f\n", worst, maxFactor);
    ++failures;
  }
}

/// §8.5 rate (b)/(c): the Zick & Homsy SC sphere (phi = 0.216, R = 0.3722 L) sampled onto the
/// grid, apertures through the solver's own buildOpenness with hp, MG-PCG to rtol 1e-10.
Run solveSphere(int N, bool stretched, bool fcg) {
  const int G = CutcellMG::G;
  const int nx = N, ny = stretched ? 2 * N : N, nz = stretched ? N / 2 : N;
  const double L = 1.0;
  const double hx = L / nx, hy = L / ny, hz = L / nz;
  const double dref = hx;
  const double hp[3] = {hx / dref, hy / dref, hz / dref};
  const double wx = 1.0 / (hp[0] * hp[0]), wy = 1.0 / (hp[1] * hp[1]), wz = 1.0 / (hp[2] * hp[2]);
  const C3 e{nx + 2 * G, ny + 2 * G, nz + 2 * G};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  const double R = 0.3722 * L;  // phi = 0.216 in the unit cell

  CCField sdf("sdf", n), ox("ox", n), oy("oy", n), oz("oz", n);
  {  // d' = (|x - c|_minimage - R) / dref at the PHYSICAL cell centres, ghosts included
    auto m = Kokkos::create_mirror_view(sdf);
    for (int k = 0; k < e.z; ++k)
      for (int j = 0; j < e.y; ++j)
        for (int i = 0; i < e.x; ++i) {
          double px = ((i - G) + 0.5) * hx - 0.5 * L, py = ((j - G) + 0.5) * hy - 0.5 * L,
                 pz = ((k - G) + 0.5) * hz - 0.5 * L;
          px -= L * std::round(px / L);
          py -= L * std::round(py / L);
          pz -= L * std::round(pz / L);
          const double d = std::sqrt(px * px + py * py + pz * pz) - R;
          m((std::size_t)i + (std::size_t)j * e.x + (std::size_t)k * (std::size_t)e.x * e.y) =
              d / dref;
        }
    Kokkos::deep_copy(sdf, m);
  }
  buildOpenness(ox, oy, oz, CCConst(sdf), e, hp[0], hp[1], hp[2], 1);

  CutcellMG mg;
  mg.setMetric(hp);
  mg.init(nx, ny, nz, 6);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), wx, wy, wz);

  // A compatible RHS: the smooth mode of the order gate, masked to the rows that carry an open
  // face (AC > 0) and mean-removed over exactly those rows.
  std::vector<double> hb(n, 0.0);
  {
    auto hAC = Kokkos::create_mirror_view(mg.level(0).AC);
    Kokkos::deep_copy(hAC, mg.level(0).AC);
    const double k = 2.0 * M_PI / L;
    double sum = 0.0;
    long cnt = 0;
    for (int z = G; z < e.z - G; ++z)
      for (int y = G; y < e.y - G; ++y)
        for (int x = G; x < e.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
          if (!(hAC(i) > 0.0f))
            continue;
          const double px = ((x - G) + 0.5) * hx, py = ((y - G) + 0.5) * hy,
                       pz = ((z - G) + 0.5) * hz;
          const double v = std::cos(k * px) * std::cos(k * py) * std::cos(k * pz);
          hb[i] = v;
          sum += v;
          ++cnt;
        }
    const double mean = cnt ? sum / (double)cnt : 0.0;
    for (int z = G; z < e.z - G; ++z)
      for (int y = G; y < e.y - G; ++y)
        for (int x = G; x < e.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
          if (hAC(i) > 0.0f)
            hb[i] -= mean;
        }
  }
  CCField b("b", n), xf("x", n), r("r", n), pp("p", n), zz("z", n), Ap("Ap", n);
  {
    auto m = Kokkos::create_mirror_view(b);
    for (std::size_t i = 0; i < n; ++i)
      m(i) = hb[i];
    Kokkos::deep_copy(b, m);
  }
  Kokkos::deep_copy(xf, 0.0);
  Run out;
  out.levels = mg.nLevels();
  if (fcg) {
    CCField zp("zp", n);
    out.iters = mg.solveFCG(b, xf, r, pp, zz, zp, Ap, 500, 1e-10, 2, 2, 12);
  } else {
    out.iters = mg.solvePCG(b, xf, r, pp, zz, Ap, 500, 1e-10, 2, 2, 12);
  }
  {
    auto hr = Kokkos::create_mirror_view(r);
    Kokkos::deep_copy(hr, r);
    double rm = 0.0, bm = 0.0;
    for (int z = G; z < e.z - G; ++z)
      for (int y = G; y < e.y - G; ++y)
        for (int x = G; x < e.x - G; ++x) {
          const std::size_t i =
              (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
          if (!(std::fabs(hr(i)) <= rm))
            rm = std::fabs(hr(i));
          if (!(std::fabs(hb[i]) <= bm))
            bm = std::fabs(hb[i]);
        }
    out.rrel = bm > 0.0 ? rm / bm : rm;
  }
  return out;
}

void sphereGate() {
  std::printf(
      "  SPHERE RATE (§8.5 rate (b)/(c)), Zick & Homsy phi = 0.216, MG-PCG rtol 1e-10, "
      "theta = %.4g:\n",
      mgAspectTheta());
  for (int N : {32, 64}) {
    const Run st = solveSphere(N, /*stretched=*/true, /*fcg=*/false);
    const Run cu = solveSphere(N, /*stretched=*/false, /*fcg=*/false);
    const Run stF = solveSphere(N, /*stretched=*/true, /*fcg=*/true);
    std::printf(
        "    N = %3d  stretched %4d x %4d x %4d  levels %d  PCG %3d iters (r/|b| %.2e)"
        "   FCG %3d iters\n",
        N, N, 2 * N, N / 2, st.levels, st.iters, st.rrel, stF.iters);
    std::printf(
        "             cubic     %4d x %4d x %4d  levels %d  PCG %3d iters (r/|b| %.2e)"
        "   [require stretched <= cubic + 2 = %d]\n",
        N, N, N, cu.levels, cu.iters, cu.rrel, cu.iters + 2);
    if (!ablation() && !(st.iters <= cu.iters + 2)) {
      std::fprintf(stderr, "FAIL: N = %d stretched PCG %d iters > cubic %d + 2\n", N, st.iters,
                   cu.iters);
      ++failures;
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("=== cutcellmg_aniso (G4-order, doc/anisotropic_metric.md §8.5) ===\n");
    ladder(/*stretched=*/true, /*fcg=*/false, 1.95);
    ladder(/*stretched=*/false, /*fcg=*/false, 1.95);
    // The same operator through the FLEXIBLE CG driver, which flow/CLAUDE.md names as the one to
    // reach for when MG-PCG caps.  It is NOT a substitute for the §8.5 gate above; it is the
    // CONVERGED control that separates the discretization statement from the multigrid RATE.  On
    // the stretched grid today's FULL-coarsening rule leaves the V-cycle preconditioner stalling
    // (the MG-PCG rows above cap, and their L2 error then drifts a few percent from the exact
    // discrete value); §5 of doc/anisotropic_metric.md is the semi-coarsening rule that fixes it
    // and commit C3 is where the RATE part of G4 gates it.
    ladder(/*stretched=*/true, /*fcg=*/true, 1.95);
    ladder(/*stretched=*/false, /*fcg=*/true, 1.95);
    std::printf("=== C3 — the aspect-ratio coarsening rule (doc/anisotropic_metric.md §5) ===\n");
    if (ablation())
      std::printf(
          "  PECLET_FLOW_MG_ASPECT = %.4g -> ABLATION RUN: the gates below PRINT, they do "
          "not assert (§8.5 item (c)).\n",
          mgAspectTheta());
    levelTables();
    vcycleRate(32, 0.2);
    sphereGate();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
