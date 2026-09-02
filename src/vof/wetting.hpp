/// @file
/// @brief flow — thin shim: the container-free VoF theta-consistent solid-band fill (rung V5b) now
/// live in `core`.
///
/// **WO-W0 (2026-09-02), the L1 promotion of `VOF_PLAN.md` §11.** These kernels were always
/// container-free by contract (`plic.hpp`'s rule: scalars and small local arrays only, no
/// `Kokkos::View`, no grid indexing, no halo types), which is exactly the property that lets ONE
/// copy serve all three VoF containers — flow's structured colour field, the per-bubble block
/// container of Part III, and the future AMR path. The bodies therefore moved verbatim to
/// `core/include/peclet/core/vof/wetting.hpp` under `peclet::core::vof`; this header is the
/// compatibility shim that keeps every `peclet::flow::vof::` spelling in flow (and in the
/// `tests/kokkos` battery) working unchanged.
///
/// It is a **file move, nothing else**: the gate on the promotion was every VoF ctest bit-identical
/// before and after, on both backends. New container-free VoF math belongs in `core`, not here.
#ifndef PECLET_FLOW_VOF_WETTING_HPP
#define PECLET_FLOW_VOF_WETTING_HPP

#include "peclet/core/vof/wetting.hpp"

namespace peclet::flow::vof {
// A using-DIRECTIVE, not a list of using-declarations: qualified lookup into a namespace follows
// its using-directives ([namespace.qual]), so `peclet::flow::vof::plicVolume` and an unqualified
// call from inside `peclet::flow::vof` both resolve to the core kernel, and a kernel added in core
// later needs no edit here.
using namespace peclet::core::vof;  // NOLINT(google-build-using-namespace)
}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_WETTING_HPP
