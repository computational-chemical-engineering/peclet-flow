/// @file
/// @brief flow — collocated approximate (MAC) projection helpers (Almgren–Bell–Colella style).
///
/// The collocated solver stores all three velocity components at the cell center. Its pressure
/// coupling is the approximate projection: average the cell velocities onto a face (MAC) field,
/// make THAT field divergence-free with the existing cut-cell pressure machinery (buildCutcellOp /
/// divergOpen / projectCorrect from mac_pressure.hpp), then correct the cell-centered velocities
/// with the central-difference pressure gradient. The face field is exactly divergence-free; the
/// cell field is only approximately so — hence "approximate projection". These two kernels are the
/// only collocated-specific projection pieces; everything else (operator, divergence, MG solve,
/// rotational pressure update) is shared with the staggered solver.
#ifndef PECLET_FLOW_MAC_APPROX_PROJECTION_HPP
#define PECLET_FLOW_MAC_APPROX_PROJECTION_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // peclet::flow::C3, CCField, CCConst, CCExec

namespace peclet::flow {

// Average cell-centered velocities onto the staggered face layout: uf(i) is the velocity at the low
// (-x) face of cell i (located at i-1/2) = ½(U(i-1)+U(i)); likewise vf/wf along y/z. This
// reproduces the exact layout the staggered solver stores directly, so divergOpen / projectCorrect
// (mac_pressure.hpp) act on uf/vf/wf unchanged. Computed over the block where the axis neighbour
// exists ([1,e) per axis); the cell velocity ghosts must be filled first.
inline void centerToFace(CCField uf, CCField vf, CCField wf, CCConst U, CCConst V, CCConst W, C3 e,
                         int g) {
  (void)g;
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::center_to_face", MD(space, {1, 1, 1}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        uf(i) = 0.5 * (U(i) + U(i - sx));
        vf(i) = 0.5 * (V(i) + V(i - sy));
        wf(i) = 0.5 * (W(i) + W(i - sz));
      });
}

// Cell-centered velocity correction u_c -= grad_c(phi), grad_c per axis = ½·(g⁻ + g⁺) of the two
// adjacent FACE pressure-gradients, but a face that is fully CLOSED (openness 0 — a solid
// neighbour) contributes a ZERO gradient instead of reading that neighbour's φ. Rationale:
//   * interior fluid cell (both faces open): ½(φᵢ₊₁-φᵢ₋₁) — the central difference, bulk unchanged;
//   * immersed cut cell (solid neighbour): the closed face's φ is DECOUPLED (≈0, AC≈0 in the
//   operator), so
//     reading it corrupts the gradient — zeroing that face uses only the fluid-side (open)
//     gradient;
//   * domain-BC wall (Neumann, φ-ghost = interior): the closed wall face truly has ∂φ/∂n≈0, and
//   zeroing it
//     gives ½·g_open — identical to the previous central difference with the Neumann ghost (no
//     change there).
// Cut faces (0<o<1) keep their real gradient (the neighbour is fluid). Only the cell field is
// touched; the projection's face divergence-free guarantee is unaffected. phi ghosts + face
// openness must be filled first.
inline void projectCorrectCenter(CCField u, CCField v, CCField w, CCConst phi, CCConst ox,
                                 CCConst oy, CCConst oz, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::correct_center", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double gm_x = (ox(i) > 1e-12) ? (phi(i) - phi(i - sx)) : 0.0;
        const double gp_x = (ox(i + sx) > 1e-12) ? (phi(i + sx) - phi(i)) : 0.0;
        const double gm_y = (oy(i) > 1e-12) ? (phi(i) - phi(i - sy)) : 0.0;
        const double gp_y = (oy(i + sy) > 1e-12) ? (phi(i + sy) - phi(i)) : 0.0;
        const double gm_z = (oz(i) > 1e-12) ? (phi(i) - phi(i - sz)) : 0.0;
        const double gp_z = (oz(i + sz) > 1e-12) ? (phi(i + sz) - phi(i)) : 0.0;
        u(i) -= 0.5 * (gm_x + gp_x);
        v(i) -= 0.5 * (gm_y + gp_y);
        w(i) -= 0.5 * (gm_z + gp_z);
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_APPROX_PROJECTION_HPP
