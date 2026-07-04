/// @file
/// @brief flow — face/cell material-property accessors for the variable-coefficient momentum operator.
///
/// The variable-viscosity diffusion stencil (ibmBuildDiffusionVar) is templated on a small accessor
/// supplying the per-cell time diagonal idiag(i) = rho/dt and the per-face viscosity beta between two
/// adjacent cells. Two models:
///   - UniformFaceProps: two constants — reproduces the constant-mu operator (used to cross-check the
///     Var kernel against the scalar ibmBuildDiffusion at the solution level).
///   - FieldFaceProps: idiag from a (constant here) rho, beta from a per-cell mu field, averaged at
///     the face either arithmetically (default) or harmonically (correct for a viscosity jump — the
///     shear stress mu*du/dy is continuous across a material interface).
///
/// Averaging is done in double; the caller casts the assembled band to float once (matching the
/// constant path's `(float)(idiag + 6*beta)`), so no per-cell float rounding creeps into the sum.
#ifndef PECLET_FLOW_FACE_PROPS_HPP
#define PECLET_FLOW_FACE_PROPS_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

// Constant properties — the Var kernel with this accessor is the constant-mu operator (equivalence
// check for the variable path).
struct UniformFaceProps {
  double idiag_, beta_;
  KOKKOS_INLINE_FUNCTION double idiag(long) const { return idiag_; }
  KOKKOS_INLINE_FUNCTION double beta(long, long) const { return beta_; }
};

// Per-cell viscosity field (+ constant density for the time diagonal). harmonic=true uses the
// harmonic face mean (continuous shear stress across a viscosity jump); false = arithmetic.
struct FieldFaceProps {
  CCConst mu;
  double rhoIdt;  // rho / dt (constant-density momentum time term)
  bool harmonic;
  KOKKOS_INLINE_FUNCTION double idiag(long) const { return rhoIdt; }
  KOKKOS_INLINE_FUNCTION double beta(long i, long j) const {
    const double a = mu(i), b = mu(j);
    if (harmonic) {
      const double s = a + b;
      return (s > 0.0) ? (2.0 * a * b / s) : 0.0;
    }
    return 0.5 * (a + b);
  }
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FACE_PROPS_HPP
