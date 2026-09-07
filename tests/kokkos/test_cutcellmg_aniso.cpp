// G4-ORDER — the ANISOTROPIC pressure operator converges at the cubic rate
// (flow/doc/anisotropic_metric.md §8.5, the ORDER part; Phase 2 commit C2).
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
// GATE: least-squares order over N = 16, 32, 64 >= 1.95 on the stretched grid; the cubic control
// (N, N, N) with w = (1,1,1) gives ~2.00.  (The LEVEL TABLE and the V-cycle RATE parts of §8.5 are
// commit C3 -- this file deliberately says nothing about the coarsening rule.)
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {
int failures = 0;

struct Run {
  double l2 = 0.0;     ///< L2 norm of (phi_h - phi_exact) over the inner cells
  double l2exact = 0.0;///< the same, from the exact modal solution of the discrete system
  double rrel = 0.0;   ///< achieved max|r| / max|b|
  int iters = 0;       ///< Krylov iterations
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
    std::printf("    N = %3d  cells %4d x %4d x %4d  levels %d  %3d iters  r/|b| %.2e  "
                "L2 err %.6e  (exact discrete %.6e)",
                N, N, stretched ? 2 * N : N, stretched ? N / 2 : N, rr.levels, rr.iters, rr.rrel,
                rr.l2, rr.l2exact);
    if (err.size() > 1)
      std::printf("   order %.4f",
                  std::log(err[err.size() - 2] / err.back()) / std::log(2.0));
    std::printf("\n");
  }
  const double ord = fitOrder(Ns, err);
  std::printf("    least-squares order = %.4f  (require >= %.2f)\n", ord, minOrder);
  if (!(ord >= minOrder)) {
    std::fprintf(stderr, "FAIL: observed order %.4f < %.2f\n", ord, minOrder);
    ++failures;
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
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
