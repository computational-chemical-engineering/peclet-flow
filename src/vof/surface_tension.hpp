/// @file
/// @brief flow — VoF rung V4 (WO-P): balanced-force continuum surface force + the capillary Δt.
///
/// Container-free `KOKKOS_INLINE_FUNCTION`s only (WO-D signature rule: scalars and small local
/// arrays, no `View`, no indexing, no halo types), so the promotion of `peclet::flow::vof` to
/// `peclet::core::vof` stays a file move. The block-walking driver is `Solver::addCsfRhs`, next
/// door in `flow_ibm.hpp`, for the same reason `curvature_field.hpp` sits next to `curvature.hpp`.
///
/// ## The one rule (Francois et al., JCP 213:141 (2006); Popinet, JCP 228:5838 (2009), §3)
///
/// The surface-tension force `σκ∇C` must be evaluated **at the same location and with the same
/// discrete gradient operator as the pressure gradient**. On this staggered MAC grid the pressure
/// gradient of component `c` at the velocity unknown `u_c(i)` is the face difference
///
///     gp(i) = P(i) - P(i - s_c)                          (`buildRhsVar`, and `projectCorrectVar`)
///
/// so the CSF force at that same unknown must be
///
///     F(i) = σ · κ_f(i) · ( C(i) - C(i - s_c) )                                              (1)
///
/// and NOT the face interpolation `½(f(i) + f(i−s_c))` of a cell-centred `f = σκ∇C`. The difference
/// is not cosmetic. Write `Φ = σκ̄C` for a constant `κ̄`. Then (1) is *exactly* the discrete
/// gradient `Φ(i) − Φ(i−s_c)`, i.e. it lies in the range of the same discrete gradient operator the
/// projection inverts, so the projection annihilates it completely and the discrete equilibrium is
/// exact in floating point. An interpolated cell-centred force is not in that range for any `Φ`,
/// the projection cannot remove it, and the leftover drives spurious currents of order `σκ/μ` that
/// no amount of curvature accuracy removes (Francois et al. §2; the measured 1e-2 vs 1e-7 spurious
/// capillary number quoted in `VOF_PLAN.md` §4/V4).
///
/// This is the momentum analogue of the three-way face-mean consistency that makes the hydrostatic
/// acid test exact (`doc/variable_density_projection.md` §3): there `ρ_f` must be the SAME mean in
/// the time term, the body force and the projection coefficient; here `∇C` must be the SAME
/// difference as `∇P`. Both fail loudly and both fail in the pressure/velocity residual rather than
/// in anything a convergence study would notice.
///
/// ## The face curvature
///
/// `κ` is defined only where the cascade produced an estimate — interfacial cells, branch 1..5
/// (`vof/curvature.hpp`, `CurvatureBranch`). At a face whose two cells both carry one, the face
/// value is their arithmetic mean; where only one does, that one is used; where neither does the
/// face has NO curvature and the force is dropped, which is a defect and is counted, not hidden
/// (Basilisk `tension.h` makes the same three-way choice and marks the last branch "this should not
/// happen"). Note what the exactness argument above needs from this rule: it needs `κ_f` to take
/// the SAME value at every face where `ΔC ≠ 0`, which a constant-curvature interface satisfies
/// under all three branches. It does not need `κ` itself to be accurate — curvature error moves the
/// currents off machine zero *smoothly*, whereas an operator mismatch puts a floor under them.
///
/// ## The capillary time step (Brackbill, Kothe & Zemach, JCP 100:335 (1992), eq. 44)
///
///     Δt_σ = sqrt( (ρ₁ + ρ₂) h³ / (4πσ) )
///
/// the period of the shortest resolvable capillary wave (wavelength `2h`) divided by `2π`. Denner &
/// van Wachem (JCP 285:24 (2015)) tested it directly and confirmed all three of its features: the
/// prefactor `1/(4π)` is the true stability boundary rather than a conservative estimate, the
/// scaling is `h^{3/2}`, and it is the SUM of the densities that enters (not the mean, not the
/// heavy phase) — because both phases participate in the oscillation. It is an *explicit*
/// surface-tension constraint: it does not soften with viscosity and it is not a CFL condition, so
/// at pore-scale capillary numbers it binds long before the Weymouth–Yue CFL cap does.
#ifndef PECLET_FLOW_VOF_SURFACE_TENSION_HPP
#define PECLET_FLOW_VOF_SURFACE_TENSION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "vof/curvature.hpp"  // CurvatureBranch

namespace peclet::flow::vof {

/// Is the curvature of a cell carrying branch code `branch` usable?
///
/// True for branches 1..5 (the four shipped cascade tiers). FALSE for `kCurvNone` (0 — not an
/// interfacial cell, where `kappa` is legitimately 0 and must not be read as a curvature) and for
/// `kCurvNoEstimate` (6 — the cascade failed, where `kappa` is 0 for want of anything better).
/// The branch field rides the ordinary field registry as a double, hence the rounded compare.
KOKKOS_INLINE_FUNCTION bool csfKappaDefined(double branch) {
  const int b = static_cast<int>(branch + 0.5);
  return b >= static_cast<int>(kCurvHf) && b <= static_cast<int>(kCurvPvReduced);
}

/// Face curvature for the balanced-force CSF, from the two cells sharing the face.
///
/// `k0/b0` are the LOW cell (`i - s_c`) and `k1/b1` the HIGH cell (`i`) — the two cells the face
/// difference `C(i) - C(i - s_c)` reads, in that order. Returns false (and `kf = 0`) when neither
/// cell carries a curvature; a caller that sees this on a face with `ΔC ≠ 0` has an orphan face and
/// should say so.
KOKKOS_INLINE_FUNCTION bool csfFaceCurvature(double k0, double b0, double k1, double b1,
                                             double& kf) {
  const bool d0 = csfKappaDefined(b0), d1 = csfKappaDefined(b1);
  if (d0 && d1) {
    kf = 0.5 * (k0 + k1);
    return true;
  }
  if (d0) {
    kf = k0;
    return true;
  }
  if (d1) {
    kf = k1;
    return true;
  }
  kf = 0.0;
  return false;
}

/// The balanced-force CSF contribution to the momentum RHS at one staggered velocity unknown, in
/// the same units and at the same place as the incremental scheme's `-(P(i) - P(i - s_c))`.
///
/// `dC = C(i) - C(i - s_c)` is formed by the caller from the SAME colour field the density closure
/// reads, with the projection's own face difference. `h` is the grid spacing (1 in this solver's
/// cell units) and enters as `1/h` because `kappa` is in `1/h` and `dC` is `h·∂C`: the product
/// `σ κ ∂C` is a force per unit volume, so one power of `h` survives in the denominator.
KOKKOS_INLINE_FUNCTION double csfFaceForce(double sigma, double kf, double dC, double h) {
  return sigma * kf * dC / h;
}

/// Brackbill capillary time-step limit `sqrt((ρ₁+ρ₂) h³ / (4πσ))`.
///
/// Returns `+inf` for `σ <= 0` (no constraint). `rhoSum` is the SUM of the two phase densities —
/// see the file header for why it is the sum and not a mean.
KOKKOS_INLINE_FUNCTION double capillaryDt(double rhoSum, double h, double sigma) {
  if (!(sigma > 0.0))
    return Kokkos::Experimental::infinity_v<double>;
  return Kokkos::sqrt(rhoSum * h * h * h / (4.0 * Kokkos::numbers::pi_v<double> * sigma));
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_SURFACE_TENSION_HPP
