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
/// grad2a selects the Guy–Fogelson "gradient 2a" one-sided branch (JCP 2005, eq. 95): the
/// gradient is linearly EXTRAPOLATED from the two interior central differences,
/// g_i = 2 g_{i+1} - g_{i+2} = (-2 p_i + p_{i+1} + 2 p_{i+2} - p_{i+3})/2, instead of formed by
/// quadratic extrapolation of the pressure ((-3 p_i + 4 p_{i+1} - p_{i+2})/2 — their "gradient
/// 2"). Both are 2nd order, but gradient 2 amplifies the highest-frequency (checkerboard) mode
/// at the boundary row and is the combination their stability analysis shows to be unstable
/// under the rotational pressure update on a cell-centered approximate projection — the exact
/// scheme family of this solver. Gradient 2a annihilates that mode. When the +/-3 cell is not
/// fluid, grad2a falls back to the 2-point one-sided difference (their stable "gradient 1"),
/// never to gradient 2. Default off: byte-identical to the shipped gauge-exact scheme.
inline void gpCenterGrad(CCField out, CCConst p, CCConst sdf, int axis, C3 e, int g,
                         bool grad2a = false) {
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
        // the +/-3 cell of the grad2a stencil can sit one past the g-wide ghost pad; the axis
        // coordinate guard falls back to the 2-point form there (never reads out of the block)
        const int ca = (axis == 0) ? x : (axis == 1) ? y : z;
        const int ea = (axis == 0) ? e.x : (axis == 1) ? e.y : e.z;
        double gr;
        if (am && ap)
          gr = 0.5 * (p(i + sa) - p(i - sa));
        else if (ap) {
          if (grad2a)
            gr = (ca + 3 < ea && sdf(i + 2 * sa) >= 0.0 && sdf(i + 3 * sa) >= 0.0)
                     ? 0.5 * (-2.0 * p(i) + p(i + sa) + 2.0 * p(i + 2 * sa) - p(i + 3 * sa))
                     : (p(i + sa) - p(i));
          else
            gr = (sdf(i + 2 * sa) >= 0.0) ? 0.5 * (-3.0 * p(i) + 4.0 * p(i + sa) - p(i + 2 * sa))
                                          : (p(i + sa) - p(i));
        } else if (am) {
          if (grad2a)
            gr = (ca - 3 >= 0 && sdf(i - 2 * sa) >= 0.0 && sdf(i - 3 * sa) >= 0.0)
                     ? 0.5 * (2.0 * p(i) - p(i - sa) - 2.0 * p(i - 2 * sa) + p(i - 3 * sa))
                     : (p(i) - p(i - sa));
          else
            gr = (sdf(i - 2 * sa) >= 0.0) ? 0.5 * (3.0 * p(i) - 4.0 * p(i - sa) + p(i - 2 * sa))
                                          : (p(i) - p(i - sa));
        } else
          gr = 0.0;
        out(i) = gr;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_GAUGE_EXACT_GRADIENT_HPP
