/// @file
/// @brief flow — GridLayout policy traits (placement of the velocity unknowns).
///
/// The orchestrator peclet::flow::Solver<Grid> is templated on a GridLayout policy that supplies
/// the grid-position-dependent pieces of the solver. Phase 1 of the collocated-grid plan
/// (doc/flow_colocated_plan.md) factors out the single seam needed to keep the staggered path
/// bit-identical: the per-component velocity sample offset that the cut-cell IBM overlay / openness
/// / volume-fraction kernels are built at. The Staggered policy places u,v,w on the low (-x/-y/-z)
/// faces; the future Colocated policy will place all three components at the cell center (offset 0)
/// and add the face-averaging + approximate-projection hooks. Header-only, constexpr — no runtime
/// cost.
#ifndef PECLET_FLOW_GRID_LAYOUT_HPP
#define PECLET_FLOW_GRID_LAYOUT_HPP

#include "colocated_advection.hpp"  // cadv::advect / advect_fou / fou_operator
#include "mac_ibm.hpp"              // peclet::flow::Off3
#include "staggered_advection.hpp"  // sadv::advect / advect_fou / fou_operator

namespace peclet::flow {

// A GridLayout policy supplies the two grid-position-dependent pieces the orchestrator needs:
//   - offset(c): where component c's velocity unknown sits (drives the cut-cell IBM overlay /
//   openness /
//     volume-fraction kernels);
//   - advect / advect_fou / fou_operator: the conservative momentum advection for that
//   control-volume
//     placement (forwarded to the sadv:: or cadv:: free functions).
// The policy is stateless; the advection methods are KOKKOS_INLINE_FUNCTION so they inline on
// device.

// Staggered MAC grid: component c (0=u,1=v,2=w) lives on the low face along axis c (offset -1/2
// there), i.e. u@(i-1/2,j,k), v@(i,j-1/2,k), w@(i,j,k-1/2). This is the existing flow grid; the
// offsets reproduce the previously hard-coded {-0.5,0,0}/{0,-0.5,0}/{0,0,-0.5} arrays exactly
// (bit-identical).
struct Staggered {
  static constexpr const char* name = "staggered";
  static constexpr bool collocated = false;  // stored velocity IS the face velocity (MAC)
  static constexpr Off3 offset(int c) {
    return c == 0   ? Off3{-0.5f, 0.0f, 0.0f}
           : c == 1 ? Off3{0.0f, -0.5f, 0.0f}
                    : Off3{0.0f, 0.0f, -0.5f};
  }
  // A cell-centred VOLUMETRIC field (per-cell body force, drag coefficient) at component c's
  // unknown, `s` = strideOf(c): the face between cells i - s and i, so the second-order face mean.
  // Surface forces (pressure, viscous stress, surface tension) never come through here -- they are
  // face integrals over the unknown's control volume (suite docs/decisions/flow.md, "Volumetric
  // forces at the velocity location").
  template <class A>
  KOKKOS_INLINE_FUNCTION static double atVelocity(const A& f, long i, long s) {
    return 0.5 * ((double)f(i) + (double)f(i - s));
  }
  // `uf` (the projected-face-field advecting velocity, see Colocated below) is meaningless here:
  // the stored staggered velocity IS the face velocity, and after the projection it IS the
  // divergence-free one. Accepted and ignored so the call sites stay grid-agnostic.
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect(int c, int x, int y, int z, A U, A V, A W, A F,
                                              bool /*uf*/) {
    return sadv::advect(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_sou(int c, int x, int y, int z, A U, A V, A W, A F,
                                                  bool /*uf*/) {
    return sadv::advect_sou(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_fou(int c, int x, int y, int z, A U, A V, A W, A F,
                                                  bool /*uf*/) {
    return sadv::advect_fou(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static void fou_operator(int c, int x, int y, int z, A U, A V, A W,
                                                  double dt, double& cC, double& cxm, double& cxp,
                                                  double& cym, double& cyp, double& czm,
                                                  double& czp, bool /*uf*/) {
    sadv::fou_operator(c, x, y, z, U, V, W, dt, cC, cxm, cxp, cym, cyp, czm, czp);
  }
};

// Collocated (cell-centered) grid: all three components live at the cell center (offset 0),
// advected on the cell control volume (cadv) by the projected divergence-free MAC face field the
// approximate (ABC) projection produced last step — the cell->face average only until that field
// exists, or under the `set_uf_advection(False)` ablation.
struct Colocated {
  static constexpr const char* name = "colocated";
  static constexpr bool collocated = true;  // cell-centered velocity; approximate (MAC) projection
  static constexpr Off3 offset(int /*c*/) { return Off3{0.0f, 0.0f, 0.0f}; }
  // A cell-centred volumetric field at the (cell-centred) unknown: the cell value itself.
  template <class A>
  KOKKOS_INLINE_FUNCTION static double atVelocity(const A& f, long i, long /*s*/) {
    return (double)f(i);
  }
  // `uf == true`: U/V/W are the projected divergence-free MAC face field (low-face convention),
  // read verbatim at the control volume's faces; `false`: they are the cell velocities and the
  // face value is their average. See colocated_advection.hpp and Solver::ufAdvVelocity().
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect(int c, int x, int y, int z, A U, A V, A W, A F,
                                              bool uf) {
    return cadv::advect(c, x, y, z, U, V, W, F, uf);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_sou(int c, int x, int y, int z, A U, A V, A W, A F,
                                                  bool uf) {
    return cadv::advect_sou(c, x, y, z, U, V, W, F, uf);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_fou(int c, int x, int y, int z, A U, A V, A W, A F,
                                                  bool uf) {
    return cadv::advect_fou(c, x, y, z, U, V, W, F, uf);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static void fou_operator(int c, int x, int y, int z, A U, A V, A W,
                                                  double dt, double& cC, double& cxm, double& cxp,
                                                  double& cym, double& cyp, double& czm,
                                                  double& czp, bool uf) {
    cadv::fou_operator(c, x, y, z, U, V, W, dt, cC, cxm, cxp, cym, cyp, czm, czp, uf);
  }
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_GRID_LAYOUT_HPP
