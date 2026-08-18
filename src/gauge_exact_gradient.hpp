/// @file
/// @brief flow — the gauge-exact directional cell-centre pressure gradient.
///
/// This is the operator that makes the COLLOCATED projection second order, and it is the default
/// (`set_collocated_scheme("gauge-exact")`, formerly `set_face_interp(9)`). It lived in
/// ghost_projection.hpp because the directional ghost projection introduced it; that path is now
/// quarantined, so the piece the production scheme depends on lives here instead. The name
/// `gpCenterGrad` is kept so the shared history with the ghost overlay stays greppable.
#ifndef PECLET_FLOW_GAUGE_EXACT_GRADIENT_HPP
#define PECLET_FLOW_GAUGE_EXACT_GRADIENT_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // CCField/CCConst, C3, CCExec

namespace peclet::flow {

/// Directional cell-center gradient (collocated ghost path) of a cell field p whose
/// solid-centered rows are DECOUPLED (hold 0): central difference where both axis neighbours are
/// fluid-centered; 2nd-order one-sided toward the fluid where a neighbour center is solid
/// ((-3p_i + 4p_{i+1} - p_{i+2})/2, falling back to the 2-point one-sided when the +/-2 cell is
/// solid too); 0 when sandwiched. Never reads a solid cell's value — reading the decoupled 0 is a
/// GAUGE-DEPENDENT O(1) gradient error (measured in tests/study/ghost_collocated_apriori.py [C2]:
/// the mode-0 central difference grows O(1/h) at cut cells, the openness-weighted kernels are
/// O(1)/O(h); this operator is O(h^2) and exactly gauge-invariant). Serves both the incremental
/// -grad(P^n) predictor and the projection's cell correction. The sdf must be the PROJECTION's
/// view (the fragmentation-guarded sdfGp), so pocket cells whose phi rows are decoupled are not
/// read either.
inline void gpCenterGrad(CCField out, CCConst p, CCConst sdf, int axis, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::gp_center_grad", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const long sa = (axis == 0) ? 1 : (axis == 1) ? sy : sz;
        if (sdf(i) < 0.0) {
          out(i) = 0.0;
          return;
        }
        const bool am = sdf(i - sa) >= 0.0, ap = sdf(i + sa) >= 0.0;
        double gr;
        if (am && ap)
          gr = 0.5 * (p(i + sa) - p(i - sa));
        else if (ap)
          gr = (sdf(i + 2 * sa) >= 0.0) ? 0.5 * (-3.0 * p(i) + 4.0 * p(i + sa) - p(i + 2 * sa))
                                        : (p(i + sa) - p(i));
        else if (am)
          gr = (sdf(i - 2 * sa) >= 0.0) ? 0.5 * (3.0 * p(i) - 4.0 * p(i - sa) + p(i - 2 * sa))
                                        : (p(i) - p(i - sa));
        else
          gr = 0.0;
        out(i) = gr;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_GAUGE_EXACT_GRADIENT_HPP
