/// @file
/// @brief sdflow — collocated approximate (MAC) projection helpers (Almgren–Bell–Colella style).
///
/// The collocated solver stores all three velocity components at the cell center. Its pressure coupling is
/// the approximate projection: average the cell velocities onto a face (MAC) field, make THAT field
/// divergence-free with the existing cut-cell pressure machinery (buildCutcellOp / divergOpen /
/// projectCorrect from mac_pressure.hpp), then correct the cell-centered velocities with the
/// central-difference pressure gradient. The face field is exactly divergence-free; the cell field is only
/// approximately so — hence "approximate projection". These two kernels are the only collocated-specific
/// projection pieces; everything else (operator, divergence, MG solve, rotational pressure update) is
/// shared with the staggered solver.
#ifndef CFD_MAC_APPROX_PROJECTION_HPP
#define CFD_MAC_APPROX_PROJECTION_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // sdflow::C3, CCField, CCConst, CCExec

namespace sdflow {

// Average cell-centered velocities onto the staggered face layout: uf(i) is the velocity at the low (-x)
// face of cell i (located at i-1/2) = ½(U(i-1)+U(i)); likewise vf/wf along y/z. This reproduces the exact
// layout the staggered solver stores directly, so divergOpen / projectCorrect (mac_pressure.hpp) act on
// uf/vf/wf unchanged. Computed over the block where the axis neighbour exists ([1,e) per axis); the cell
// velocity ghosts must be filled first.
inline void centerToFace(CCField uf, CCField vf, CCField wf, CCConst U, CCConst V, CCConst W, C3 e, int g) {
  (void)g;
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "sdflow::center_to_face", MD(space, {1, 1, 1}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        uf(i) = 0.5 * (U(i) + U(i - sx));
        vf(i) = 0.5 * (V(i) + V(i - sy));
        wf(i) = 0.5 * (W(i) + W(i - sz));
      });
}

// Cell-centered velocity correction u_c -= grad_c(phi) with the central-difference gradient
// ½(phi(i+1)-phi(i-1)) per axis (= the average of the two adjacent face gradients). Exact in all-fluid /
// domain-BC regions; a cut-cell openness-aware one-sided variant near immersed solids is a later
// refinement (see doc/sdflow_colocated_plan.md). phi ghosts must be filled first.
inline void projectCorrectCenter(CCField u, CCField v, CCField w, CCConst phi, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "sdflow::correct_center", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= 0.5 * (phi(i + sx) - phi(i - sx));
        v(i) -= 0.5 * (phi(i + sy) - phi(i - sy));
        w(i) -= 0.5 * (phi(i + sz) - phi(i - sz));
      });
}

}  // namespace sdflow

#endif  // CFD_MAC_APPROX_PROJECTION_HPP
