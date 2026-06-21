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

#include "mac_ibm.hpp"  // sdflow::Off3

namespace sdflow {

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
};

}  // namespace sdflow

#endif  // CFD_GRID_LAYOUT_HPP
