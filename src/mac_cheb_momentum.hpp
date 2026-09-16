/// @file
/// @brief flow — Chebyshev semi-iteration as a standalone momentum (implicit-diffusion) solver,
/// an alternative to the red-black Gauss-Seidel smoother on the same sharp cut-cell stencil.
///
/// WHY. The momentum solve is a screened Helmholtz, A = rho/dt + mu*(-Lap), whose condition number
/// is a property of the DIFFUSION NUMBER D = mu*dt/(rho*h^2) and not of the mesh: kappa = 1 + 12 D
/// on a uniform isotropic grid. Red-black Gauss-Seidel removes error at a rate set by kappa, so it
/// needs O(kappa) sweeps; Chebyshev semi-iteration needs O(sqrt(kappa)). The second, larger, gain
/// is per iteration: a Chebyshev step is ONE residual (one halo exchange) against a red-black
/// sweep's TWO colour passes and two exchanges, and every lane is active in every pass instead of
/// half of them. Where the momentum solve is halo-latency-bound -- the small per-rank blocks that
/// made setSolidVelocityMgAuto reach for the V-cycle -- that halving is the whole story.
///
/// THE INTERVAL IS NOT ESTIMATED. Chebyshev needs an interval that CONTAINS the spectrum; it does
/// not need a tight one. `ibmStencilJacobiBounds` takes Gershgorin bounds on the Jacobi-
/// preconditioned operator directly from the stored stencil, which is exact arithmetic on the
/// coefficients (one reduction, once per solve) rather than a power iteration that costs sweeps
/// and can under-estimate lambda_max -- the one error that makes Chebyshev DIVERGE. Pinned (solid)
/// rows carry the identity and contribute (1, 1), so they cannot drag the interval.
///
/// SIGN CONVENTION. The stencil stores signed coefficients: A x = AC*x + sum(A_nb * x_nb), with
/// A_nb <= 0 for the diffusion operator. So the Gershgorin row sum is AC + sum(A_nb) (the lower
/// bound) and AC - sum(A_nb) (the upper bound) -- see residualVarPin, which uses the same form.
#ifndef PECLET_FLOW_MAC_CHEB_MOMENTUM_HPP
#define PECLET_FLOW_MAC_CHEB_MOMENTUM_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell_mg.hpp"  // FPC
#include "mac_ibm.hpp"         // CCField/CCConst/CCExec/C3

namespace peclet::flow {

/// Gershgorin bounds [lo, hi] on the spectrum of the Jacobi-preconditioned cut-cell Helmholtz
/// diag(AC)^-1 A, over the UNPINNED rows of the inner block. Both are rigorous: lo <= lambda_min
/// and hi >= lambda_max, so Chebyshev on [lo, hi] is unconditionally convergent. A loose lo only
/// slows it down; a hi below lambda_max would diverge, which is why this is arithmetic and not an
/// estimate. Rows with a vanishing diagonal (never solved) are skipped.
template <class MC>
inline void ibmStencilJacobiBounds(MC AC, MC AW, MC AE, MC AS, MC AN, MC AB, MC AT,
                                   CCConst solidmask, C3 e, int g, double& lo, double& hi) {
  CCExec space;
  const bool hasMask = (solidmask.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  const MD pol(space, {g, g, g}, {e.x - g, e.y - g, e.z - g});
  double rlo = 0.0, rhi = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::cheb_bounds_lo", pol,
      KOKKOS_LAMBDA(int lx, int ly, int lz, double& m) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (hasMask && solidmask(i) > 0.5)
          return;
        const double ac = (double)AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double off = (double)AW(i) + (double)AE(i) + (double)AS(i) + (double)AN(i) +
                           (double)AB(i) + (double)AT(i);
        const double v = (ac + off) / ac;  // row sum / diagonal
        if (v < m)
          m = v;
      },
      Kokkos::Min<double>(rlo));
  Kokkos::parallel_reduce(
      "peclet::flow::cheb_bounds_hi", pol,
      KOKKOS_LAMBDA(int lx, int ly, int lz, double& m) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (hasMask && solidmask(i) > 0.5)
          return;
        const double ac = (double)AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double off = (double)AW(i) + (double)AE(i) + (double)AS(i) + (double)AN(i) +
                           (double)AB(i) + (double)AT(i);
        const double v = (ac - off) / ac;
        if (v > m)
          m = v;
      },
      Kokkos::Max<double>(rhi));
  lo = rlo;
  hi = rhi;
}

/// One Chebyshev update on the Jacobi-preconditioned system:  d <- beta*d + alpha*r/AC ;  x += d.
/// Pinned (solid) rows are held at zero exactly as the red-black smoother holds them, so the two
/// solvers share a fixed point. r is expected to be the pin-aware residual (residualVarPin), which
/// is already zero there.
inline void ibmChebUpdate(CCField x, CCField d, CCConst r, FPC AC, CCConst solidmask, C3 e, int g,
                          double alpha, double beta) {
  CCExec space;
  const bool hasMask = (solidmask.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cheb_update", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (hasMask && solidmask(i) > 0.5) {
          d(i) = 0.0;
          x(i) = 0.0;
          return;
        }
        const double ac = (double)AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double dn = beta * d(i) + alpha * r(i) / ac;
        d(i) = dn;
        x(i) += dn;
      });
}

/// The Chebyshev coefficient recurrence (Golub-Varga; the form multigrid Chebyshev smoothers use).
/// Iteration 0 takes alpha = 1/theta, beta = 0; iteration k takes rho_k = 1/(2 sigma - rho_{k-1}),
/// alpha = 2 rho_k / delta, beta = rho_k rho_{k-1}. Holding the recurrence in one small struct
/// keeps the driver free of solver algebra.
struct ChebCoeffs {
  double theta, delta, sigma, rho;
  ChebCoeffs(double lo, double hi)
      : theta(0.5 * (hi + lo)), delta(0.5 * (hi - lo)), sigma(theta / delta), rho(1.0 / sigma) {}
  /// Coefficients for iteration `it` (0-based). Advances the recurrence.
  void next(int it, double& alpha, double& beta) {
    if (it == 0) {
      alpha = 1.0 / theta;
      beta = 0.0;
      return;
    }
    const double rPrev = rho;
    rho = 1.0 / (2.0 * sigma - rPrev);
    alpha = 2.0 * rho / delta;
    beta = rho * rPrev;
  }
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CHEB_MOMENTUM_HPP
