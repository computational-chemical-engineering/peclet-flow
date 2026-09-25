/// @file
/// @brief flow — the multidimensional iteration policies: iteration order follows storage.
///
/// Every flow field is x-fastest: cell (x, y, z) of an nx*ny*nz block lives at
/// I = x + y*nx + z*nx*ny (Kokkos LayoutLeft, Fortran order in Python, x-fastest in VTI;
/// suite/docs/CONVENTIONS.md). Every multidimensional kernel therefore iterates x FASTEST on EVERY
/// backend, never Kokkos' backend default: the default is Iterate::Left on CUDA/HIP (first index
/// fastest -- already right) but Iterate::Right on the host backends (LAST index fastest), which
/// made every OpenMP/Serial kernel stride nx*ny doubles between consecutive iterations.
///
/// The rule: an MDRange kernel is written `MDRange3<Exec>(space, {x0, y0, z0}, {x1, y1, z1})` with
/// the lambda taking (x, y, z) -- the storage-fastest index FIRST; a 2-D plane kernel should take
/// its two in-plane indices in storage order too (the cyclic (a+1)%3, (a+2)%3 face planes do not
/// for a = y: O(N^2) work, left as they are). `Kokkos::Rank<` and a bare `Kokkos::MDRangePolicy<`
/// may appear only in this header; tests/python/test_iteration_order.py (ctest `iteration_order`)
/// fails otherwise, in src/ and tests/ alike.
///
/// Numerics: on CUDA/HIP these are the defaults already, so the device code is bit-identical. On a
/// host backend a kernel that writes each cell from exactly one iteration is bit-identical as well;
/// an MDRange REDUCTION, or an atomic scatter into a shared cell, may change its summation order
/// (it follows the iteration order).
#ifndef PECLET_FLOW_POLICY_HPP
#define PECLET_FLOW_POLICY_HPP

#include <Kokkos_Core.hpp>

namespace peclet::flow {

/// 2-D range, first index fastest (both the tile order and the order within a tile).
template <class Exec>
using MDRange2 =
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;

/// 3-D range, x (the first index) fastest (both the tile order and the order within a tile).
template <class Exec>
using MDRange3 =
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;

}  // namespace peclet::flow

#endif  // PECLET_FLOW_POLICY_HPP
