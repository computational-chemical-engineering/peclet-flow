/// @file
/// @brief sdflow — GridLayout policy traits (placement of the velocity unknowns).
///
/// The orchestrator sdflow::SdflowSolver<Grid> is templated on a GridLayout policy that supplies the
/// grid-position-dependent pieces of the solver. Phase 1 of the collocated-grid plan
/// (doc/sdflow_colocated_plan.md) factors out the single seam needed to keep the staggered path
/// bit-identical: the per-component velocity sample offset that the cut-cell IBM overlay / openness /
/// volume-fraction kernels are built at. The Staggered policy places u,v,w on the low (-x/-y/-z) faces;
/// the future Colocated policy will place all three components at the cell center (offset 0) and add the
/// face-averaging + approximate-projection hooks. Header-only, constexpr — no runtime cost.
#ifndef CFD_GRID_LAYOUT_HPP
#define CFD_GRID_LAYOUT_HPP

#include "mac_ibm.hpp"               // sdflow::Off3
#include "staggered_advection.hpp"   // sadv::advect / advect_fou / fou_operator
#include "colocated_advection.hpp"   // cadv::advect / advect_fou / fou_operator

namespace sdflow {

// A GridLayout policy supplies the two grid-position-dependent pieces the orchestrator needs:
//   - offset(c): where component c's velocity unknown sits (drives the cut-cell IBM overlay / openness /
//     volume-fraction kernels);
//   - advect / advect_fou / fou_operator: the conservative momentum advection for that control-volume
//     placement (forwarded to the sadv:: or cadv:: free functions).
// The policy is stateless; the advection methods are KOKKOS_INLINE_FUNCTION so they inline on device.

// Staggered MAC grid: component c (0=u,1=v,2=w) lives on the low face along axis c (offset -1/2 there),
// i.e. u@(i-1/2,j,k), v@(i,j-1/2,k), w@(i,j,k-1/2). This is the existing sdflow grid; the offsets
// reproduce the previously hard-coded {-0.5,0,0}/{0,-0.5,0}/{0,0,-0.5} arrays exactly (bit-identical).
struct Staggered {
  static constexpr const char* name = "staggered";
  static constexpr Off3 offset(int c) {
    return c == 0   ? Off3{-0.5f, 0.0f, 0.0f}
           : c == 1 ? Off3{0.0f, -0.5f, 0.0f}
                    : Off3{0.0f, 0.0f, -0.5f};
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect(int c, int x, int y, int z, A U, A V, A W, A F) {
    return sadv::advect(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_fou(int c, int x, int y, int z, A U, A V, A W, A F) {
    return sadv::advect_fou(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static void fou_operator(int c, int x, int y, int z, A U, A V, A W, double dt,
                                                  double& cC, double& cxm, double& cxp, double& cym,
                                                  double& cyp, double& czm, double& czp) {
    sadv::fou_operator(c, x, y, z, U, V, W, dt, cC, cxm, cxp, cym, cyp, czm, czp);
  }
};

// Collocated (cell-centered) grid: all three components live at the cell center (offset 0), advected on
// the cell control volume with cell->face-averaged advecting velocities (cadv). The pressure coupling
// (approximate/MAC projection) is added in a later phase; this policy carries the predictor pieces.
struct Colocated {
  static constexpr const char* name = "colocated";
  static constexpr Off3 offset(int /*c*/) { return Off3{0.0f, 0.0f, 0.0f}; }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect(int c, int x, int y, int z, A U, A V, A W, A F) {
    return cadv::advect(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static double advect_fou(int c, int x, int y, int z, A U, A V, A W, A F) {
    return cadv::advect_fou(c, x, y, z, U, V, W, F);
  }
  template <class A>
  KOKKOS_INLINE_FUNCTION static void fou_operator(int c, int x, int y, int z, A U, A V, A W, double dt,
                                                  double& cC, double& cxm, double& cxp, double& cym,
                                                  double& cyp, double& czm, double& czp) {
    cadv::fou_operator(c, x, y, z, U, V, W, dt, cC, cxm, cxp, cym, cyp, czm, czp);
  }
};

}  // namespace sdflow

#endif  // CFD_GRID_LAYOUT_HPP
