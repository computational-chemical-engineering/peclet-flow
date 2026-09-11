/// @file
/// @brief The ONE compiled instantiation of the staggered MAC solver,
/// `peclet::flow::Solver<Staggered>`.
///
/// `Solver<Grid>` is a ~12 k-line class template whose 517 out-of-line members live in the twelve
/// `flow_ibm_*.hpp` domain headers.  Before QUALITY_PLAN G.8 every consumer — 45 test executables
/// and the nanobind module, the latter for both grids — instantiated all of it from scratch at
/// `-O3`, which is what made a full rebuild ~45-50 CPU-minutes.  This translation unit instantiates
/// it once; `flow_ibm.hpp` carries the matching `extern template` declaration, which suppresses the
/// implicit instantiation everywhere else (`PECLET_FLOW_INSTANTIATING` skips it here).
///
/// There is NO cross-TU optimization to lose: the build carries no LTO and no `-march`, so the
/// kernels compiled here are the same kernels, from the same source, under the same flags, that
/// each consumer used to compile for itself.  Members defined inside the class body stay implicitly
/// inline and are therefore still instantiated (and inlined) in the consumer — the standard exempts
/// inline functions from an explicit instantiation declaration — so the small accessors on the hot
/// path are unaffected.
///
/// `PECLET_FLOW_MPI` must agree between this TU and its consumers: it changes the class definition.
/// The build therefore compiles this file once per configuration (peclet_flow_solver without the
/// macro, peclet_flow_solver_mpi with it) and links each consumer against the matching one.
#define PECLET_FLOW_INSTANTIATING 1
#include "flow_ibm.hpp"

namespace peclet::flow {

template class Solver<Staggered>;

}  // namespace peclet::flow
