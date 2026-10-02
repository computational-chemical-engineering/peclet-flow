/// @file
/// @brief flow — the Anderson accelerator of a steady march: core's AndersonCore wrapped around
/// one `Solver<Grid>::step()` (doc/steady_acceleration.md, rev 1: D1, D13, §4.3, §5.1).
///
/// The data path of the steady acceleration. `peclet::core::solver::AndersonCore` owns the history,
/// the reductions, the least squares, the safeguards and the lazy mix; this adapter supplies the
/// solver's march state (`Solver::marchState()`: the velocity, P and, collocated with projected-
/// face advection, the face field), re-checks the refusals of §5.3 at every call, detects a change
/// of the parameter signature (internal dt, rho, mu, F) and calls the solver's step() unchanged.
/// The control path (phases, the stop instrument) is the pure-Python `peclet.flow.march_to_steady`.
///
/// One call of step(accelerate) is §4.3: the refusal re-check, the signature check, core.prepare
/// (the inner-entry state check and the lazy mix), solver.step(), core.complete. A step that
/// throws at a mixed iterate is absorbed (the last map output is restored and acceleration is
/// disabled); at a plain iterate it is the plain march's own failure and is rethrown.
///
/// Compiled once per grid beside the solver (src/flow_solver_{staggered,colocated}.cpp, the G.8
/// pattern): the member definitions below are out of line and the header ends with the matching
/// `extern template` declarations.
#ifndef PECLET_FLOW_ANDERSON_ACCELERATOR_HPP
#define PECLET_FLOW_ANDERSON_ACCELERATOR_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "flow_ibm.hpp"
#include "peclet/core/solver/anderson.hpp"
#ifdef PECLET_FLOW_MPI
#include "peclet/core/solver/anderson_mpi.hpp"
#endif

namespace peclet::flow {

template <class Grid>
class AndersonAccelerator {
 public:
  using Core = peclet::core::solver::AndersonCore;

  /// Builds the march state (throws std::runtime_error on a refused configuration, §5.3) and
  /// allocates the history ((2m+3)·n_s·8·n_pad bytes, §6.3; throws std::runtime_error naming the
  /// bytes when it does not fit). Collective under MPI.
  explicit AndersonAccelerator(Solver<Grid>& solver, int window = 5, double mixing = 1.0)
      : AndersonAccelerator(solver, solver.marchState(), window, mixing) {}

  AndersonAccelerator(const AndersonAccelerator&) = delete;
  AndersonAccelerator& operator=(const AndersonAccelerator&) = delete;

  /// One map evaluation (§4.3): the lazy mix when `accelerate` and a mix is pending, then
  /// solver.step(). Collective under MPI.
  void step(bool accelerate);

  // ---- status (forwarded from the core) ----
  const char* statusName() const { return core_.statusName(); }
  const std::string& reason() const { return core_.reason(); }
  /// Relative velocity residual of the last evaluation (design §3.2); +inf before the first.
  double residual() const { return core_.residual(); }
  int numRestarts() const { return core_.numRestarts(); }
  int numResets() const { return core_.numResets(); }
  int numColumns() const { return core_.numColumns(); }
  int window() const { return core_.window(); }
  double mixing() const { return core_.mixing(); }
  /// Bytes of the history (= Core::memoryBytesFor(window, n_s, n_pad), design §6.3).
  std::size_t memoryBytes() const { return core_.memoryBytes(); }
  /// Cumulative wall time spent in the accelerator, solver.step() excluded (device-fenced at both
  /// ends of each part; the instrument of gate G8), seconds, this rank.
  double seconds() const { return seconds_; }
  void reset() { core_.reset(); }
  void disable() { core_.disable(); }
  /// The core itself (tests: γ, engagement, the descriptor).
  const Core& core() const { return core_; }

 private:
  AndersonAccelerator(Solver<Grid>& solver, typename Solver<Grid>::MarchState ms, int window,
                      double mixing);
  static Core makeCore(typename Solver<Grid>::MarchState& ms, int window, double mixing);
  static double now();

  Solver<Grid>& s_;
  std::array<double, 6> sig_;
  Core core_;
  double seconds_ = 0.0;
};

template <class Grid>
AndersonAccelerator<Grid>::AndersonAccelerator(Solver<Grid>& solver,
                                               typename Solver<Grid>::MarchState ms, int window,
                                               double mixing)
    : s_(solver), sig_(ms.signature), core_(makeCore(ms, window, mixing)) {}

template <class Grid>
typename AndersonAccelerator<Grid>::Core AndersonAccelerator<Grid>::makeCore(
    typename Solver<Grid>::MarchState& ms, int window, double mixing) {
  peclet::core::solver::AndersonState st;
  st.fields = ms.fields;
  st.roles = ms.roles;
  st.extent = {ms.e.x, ms.e.y, ms.e.z};
  st.ghost = ms.G;
#ifdef PECLET_FLOW_MPI
  if (ms.distributed)
    st.comm = peclet::core::solver::andersonComm(ms.comm);
#endif
  try {
    return Core(std::move(st), window, mixing);
  } catch (const std::runtime_error& e) {  // the history does not fit (§6.3), on every rank
    throw std::runtime_error(std::string("AndersonAccelerator: ") + e.what() +
                             "; pass accelerate=False");
  }
}

template <class Grid>
double AndersonAccelerator<Grid>::now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class Grid>
void AndersonAccelerator<Grid>::step(bool accelerate) {
  double t0 = now();
  // §5.3: the refusals are re-evaluated at every call (host flags only), so a feature enabled
  // after construction is refused at the next call.
  const auto ms = s_.marchState();
  // The core mixes the buffers it was built on; a redistribute reallocates them, after which the
  // accelerator would act on buffers the solver no longer reads.
  const auto& held = core_.state().fields;
  bool same = held.size() == ms.fields.size();
  for (std::size_t f = 0; same && f < held.size(); ++f)
    same = held[f].data() == ms.fields[f].data() && held[f].extent(0) == ms.fields[f].extent(0);
  if (!same)
    throw std::logic_error(
        "AndersonAccelerator: the solver's state buffers were reallocated (a redistribute or a "
        "rebalance since construction); construct a new accelerator");
  // §4.3 step 0, the signature half: a parameter change is a new map.
  if (core_.status() == Core::Status::Active && ms.signature != sig_) {
    core_.invalidate();
    sig_ = ms.signature;
  }
  core_.prepare(accelerate);
  Kokkos::fence();
  seconds_ += now() - t0;
  try {
    s_.step();
  } catch (const std::exception& e) {
    t0 = now();
    const bool absorbed = core_.stepFailed(e.what());
    seconds_ += now() - t0;
    if (!absorbed)
      throw;
    return;
  }
  t0 = now();
  core_.complete(s_.pressureSolveFailed());
  Kokkos::fence();
  seconds_ += now() - t0;
}

}  // namespace peclet::flow

// Compiled once per grid in src/flow_solver_{staggered,colocated}.cpp (QUALITY_PLAN G.8).
#ifndef PECLET_FLOW_INSTANTIATING
namespace peclet::flow {
extern template class AndersonAccelerator<Staggered>;
extern template class AndersonAccelerator<Colocated>;
}  // namespace peclet::flow
#endif

#endif  // PECLET_FLOW_ANDERSON_ACCELERATOR_HPP
