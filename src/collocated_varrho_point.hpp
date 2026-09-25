/// @file
/// @brief flow — the POINTWISE formulas of the collocated variable-density ABC pair
///        (doc/collocated_varrho_forces.md §4.1, §4.9 L5).
///
/// Container-free on purpose: no Views, no solver state, `KOKKOS_INLINE_FUNCTION` only. They are
/// the three places where the collocated variable-density scheme's WEIGHTS live, so that promoting
/// them to `peclet::core::scheme` (for amr's multiphase package, §13 / Q10) is a file move, and so
/// that a change of reconstruction weights (the ghost scheme's one-sided pair, amr's coarse-fine
/// weights) changes these functions and nothing else.
///
/// The pair they define (face `j` of component `c` separates cells `j - s` and `j`):
///
///  * centre -> face, the projection's constraint:  u_f = (rho_L u_L + rho_R u_R) / (rho_L + rho_R)
///    (`faceMassWeighted`; the MOMENTUM-weighted map. Only with it is the pressure force exactly
///    -M^{-1} C^T, M Gamma = -(D O Pi_rho)^T for any openness 0 <= o <= 1 — §4.3);
///  * face -> cell, the pressure force and the projection's cell correction:
///    a_c = ½ (o_lo a_lo + o_hi a_hi)  (`cellFromFaces`; the face accelerations a are the bracket
///    [F_f - w G_f P] / rho_f WITHOUT the openness, which this reconstruction applies once);
///  * the weight sum of that reconstruction, W = ½ (o_lo + o_hi) (`weightSum`), which multiplies a
///    per-cell volumetric force so that f = rho g balances the pressure part rho_c sum r_f g
///    exactly next to a wall (§4.5).
///
/// L1: openness is a multiplicative WEIGHT, never a predicate. With o in {0, 1} (this package:
/// all-fluid, domain walls) it is bitwise what a predicate gives; with fractional apertures it is
/// the aperture-adjoint pair. `cellFromFaces` and `weightSum` must always be called with the SAME
/// two openness reads.
#ifndef PECLET_FLOW_COLLOCATED_VARRHO_POINT_HPP
#define PECLET_FLOW_COLLOCATED_VARRHO_POINT_HPP

#include <Kokkos_Core.hpp>

namespace peclet::flow::varrho {

// Momentum-weighted face velocity: (rho_L u_L + rho_R u_R) / (rho_L + rho_R).
KOKKOS_INLINE_FUNCTION double faceMassWeighted(double uL, double uR, double rL, double rR) {
  return (rL * uL + rR * uR) / (rL + rR);
}

// The cell value reconstructed from its two faces along one axis: ½ (o_lo a_lo + o_hi a_hi).
KOKKOS_INLINE_FUNCTION double cellFromFaces(double aLo, double aHi, double oLo, double oHi) {
  return 0.5 * (oLo * aLo + oHi * aHi);
}

// The weight sum of cellFromFaces for the same two openness values: ½ (o_lo + o_hi).
KOKKOS_INLINE_FUNCTION double weightSum(double oLo, double oHi) {
  return 0.5 * (oLo + oHi);
}

}  // namespace peclet::flow::varrho

#endif  // PECLET_FLOW_COLLOCATED_VARRHO_POINT_HPP
