/// @file
/// @brief nanobind module `flow` — the Kokkos cut-cell IBM Navier-Stokes solver
/// (`peclet.flow.Solver`).
///
/// Exposes peclet::flow::IbmSolver to Python: set rho/mu/dt, a body force, an SDF solid (cut-cell
/// IBM no-slip
/// + optional cut-cell pressure projection), step, read back the velocity/pressure, and query the
/// cut-cell flux divergence. Exercised by verify_poiseuille_sdflow (IBM channel) and
/// verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing).
/// rank()/bcast_from_root() are single-rank stubs (the multi-rank path lives in tests/kokkos_mpi).
///
/// Kokkos teardown follows the suite-wide pattern of peclet/core/python/kokkos_teardown.hpp: Kokkos
/// is initialized at import; every bound Solver is a `Releasable` (registered on construction) and
/// every zero-copy array (`field_view`) is a Releasable capsule; the module's single atexit hook
/// (also `flow.finalize()`) releases all of them and THEN calls Kokkos::finalize, so a Solver or
/// array still referenced at interpreter exit (script globals, a Jupyter/Quarto kernel) can no
/// longer be destroyed after finalize -- which is a Kokkos::abort (SIGABRT / exit 134, on OpenMP
/// as on CUDA), not a warning. Nothing needs to be `del`eted before exit any more.
///
/// Arrays cross the boundary through the shared zero-copy bridge (peclet::core::python, in core):
/// fields come back as Fortran-order (nx,ny,nz) float64 NumPy arrays referencing the field buffer,
/// and inputs are read as flat x-fastest buffers. See peclet/core/python/ndarray_interop.hpp.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <limits>
#include <string>
#include <vector>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#endif

#include "anderson_accelerator.hpp"
#include "flow_bindings_checks.hpp"
#include "flow_bindings_common.hpp"
#include "flow_bindings_core.hpp"
#include "flow_bindings_geometry.hpp"
#include "flow_bindings_phase_change.hpp"
#include "flow_bindings_scalar.hpp"
#include "flow_bindings_sharp_viscous.hpp"
#include "flow_bindings_tuning.hpp"
#include "flow_bindings_vof.hpp"
#include "flow_bindings_vof_blocks.hpp"
#include "flow_bindings_vof_momentum.hpp"
#include "flow_ibm.hpp"
#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

namespace nb = nanobind;
// The topic binders (flow_bindings_<topic>.hpp).
namespace fb = peclet::flow::bindings;
// The shared binding helpers (flow_bindings_common.hpp).
#ifdef PECLET_FLOW_MPI
using peclet::flow::bindings::ensure_mpi_init;
#endif
using peclet::flow::bindings::BoundAccelerator;
using peclet::flow::bindings::BoundSolver;
using peclet::flow::bindings::Diagnostics;

// The developer-tier Anderson accelerator of a steady march, bound as a PRIVATE class
// (`_AndersonAccelerator` / `_AndersonAcceleratorColocated`) and created only through
// `s.diagnostics.anderson_accelerator(...)`. The public entry is peclet.flow.march_to_steady.
template <class Grid>
static void bind_accelerator(nb::module_& m, const char* name) {
  using A = BoundAccelerator<Grid>;
  nb::class_<A>(
      m, name,
      "Anderson accelerator of a steady march (doc/steady_acceleration.md): type-II Anderson on "
      "the march state (velocity + P, plus the collocated face field under projected-face "
      "advection), velocity-only metric. Created by s.diagnostics.anderson_accelerator(); the "
      "public driver is peclet.flow.march_to_steady. step(accelerate) is ONE call of the solver's "
      "step() (preceded by the lazy mix when accelerate and a mix is pending); between calls the "
      "solver always holds a genuine step output.")
      .def("step", &A::step, nb::arg("accelerate") = true,
           "One map evaluation: when accelerate and a mix is pending, the solver's state is "
           "replaced by the Anderson mix first; then one solver.step(). accelerate=False is a "
           "plain step that still records history. Raises if the configuration became "
           "unsupported (the same refusals as the constructor).")
      .def_prop_ro("residual", &A::residual,
                   "Relative velocity residual ||g_u - x_u|| / ||g_u|| of the last evaluation "
                   "(inf before the first).")
      .def_prop_ro(
          "status", [](const A& a) { return std::string(a.statusName()); },
          "'active' or 'disabled' (acceleration stopped; steps are plain).")
      .def_prop_ro("reason", &A::reason, "Why the status is not 'active' ('' while active).")
      .def_prop_ro("num_restarts", &A::numRestarts, "History restarts so far.")
      .def_prop_ro("num_resets", &A::numResets,
                   "History resets because the state or the parameters (dt, rho, mu, body force, "
                   "advection on / scheme / implicit) "
                   "were changed from outside.")
      .def_prop_ro("num_columns", &A::numColumns, "Window columns in use.")
      .def_prop_ro("window", &A::window, "The window m.")
      .def_prop_ro("mixing", &A::mixing, "The mixing beta.")
      .def_prop_ro("memory_bytes", &A::memoryBytes,
                   "Bytes of the history: (2m+3) * n_fields * 8 * n_padded (this rank).")
      .def_prop_ro("seconds", &A::seconds,
                   "Cumulative wall time spent in the accelerator, solver.step() excluded "
                   "(device-fenced), seconds, this rank.")
      .def("reset", &A::reset, "Clear the history (the next call starts a new window).")
      .def("disable", &A::disable, "Stop accelerating for good; step() is then a plain step.");
}

// The developer tier of Solver<Grid>: every member that inspects, profiles, ablates or tunes
// the solver beyond what a user of the method needs (suite/docs/QUALITY_PLAN.md D2). Bound
// once per grid as `<name>Diagnostics`, reached only through `Solver.diagnostics`.
template <class Grid>
static void bind_diagnostics(nb::module_& m, const char* name) {
  using D = Diagnostics<Grid>;
  nb::class_<D> cls(
      m, name,
      "Developer tier of a Solver, reached as `s.diagnostics`: instruments (censuses, timers, "
      "budgets, ledgers), solver TUNING beyond the public driver selection, ablation switches "
      "and the raw zero-copy / MPI internals. Nothing here is needed to set up, run or read out "
      "a simulation. It holds a reference to the solver and no state of its own; the setters "
      "that CHANGE RESULTS say so in their docstring.");
  fb::bindDiagnosticsSolverOptions<Grid>(cls);
  fb::bindDiagnosticsScalar<Grid>(cls);
  fb::bindDiagnosticsPressure<Grid>(cls);
  fb::bindDiagnosticsGeometry<Grid>(cls);
  fb::bindDiagnosticsVof<Grid>(cls);
  fb::bindDiagnosticsVofBlocks<Grid>(cls);
  fb::bindDiagnosticsPhaseChange<Grid>(cls);
  fb::bindDiagnosticsVofForces<Grid>(cls);
  fb::bindDiagnosticsSharpViscous<Grid>(cls);
  fb::bindDiagnosticsVofMomentum<Grid>(cls);
  fb::bindDiagnosticsPorousMpi<Grid>(cls);
}

// Register a solver class for the given GridLayout policy (Staggered -> "Solver", Colocated ->
// "SolverColocated") and its developer tier. The Python API is identical across grids; only
// the velocity-unknown placement and the advection control volume differ inside Solver<Grid>.
template <class Grid>
static void bind_solver(nb::module_& m, const char* name, const char* diag_name) {
  using S = BoundSolver<Grid>;
  bind_diagnostics<Grid>(m, diag_name);
  nb::class_<S> cls(
      m, name,
      "One incompressible Navier-Stokes solver on a block of the grid (staggered MAC, the "
      "reference) -- or, for SolverColocated, the same API with cell-centered velocities and "
      "the ABC approximate projection. Construct on a (physical) domain; set_rho / set_mu / "
      "set_dt and the domain BCs; give it a geometry (set_solid, set_pressure_geometry or a "
      "scene); step(); read out with the get_* arrays. Developer instruments and tuning live "
      "on `.diagnostics`.");
  fb::bindSolverSetup<Grid>(cls);
  fb::bindSolverSharpViscous<Grid>(cls);
  fb::bindSolverControls<Grid>(cls);
  fb::bindSolverGeometry<Grid>(cls);
  fb::bindSolverState<Grid>(cls);
  fb::bindSolverScalar<Grid>(cls);
  fb::bindSolverVof<Grid>(cls);
  fb::bindSolverVofBlocks<Grid>(cls);
  fb::bindSolverPhaseChange<Grid>(cls);
  fb::bindSolverVofMomentum<Grid>(cls);
  fb::bindSolverPorousMpi<Grid>(cls);
}

NB_MODULE(_flow, m) {
  m.attr("__doc__") =
      "flow — Kokkos cut-cell IBM incompressible Navier-Stokes solver for porous media.\n\n"
      "Two solver classes share an identical API (only the velocity-unknown placement differs):\n"
      "  Solver          — staggered MAC grid (THE flow solver; permeability/drag accuracy "
      "default)\n"
      "  SolverColocated — collocated / cell-centered velocities (ABC approximate projection)\n\n"
      "Conventions: physical units throughout (density rho, viscosity mu, physical pressure p); "
      "SDFs\n"
      "are negative inside the solid; fields are Fortran-order (nx,ny,nz) float64 (x-fastest). "
      "This is\n"
      "the single-rank module — the multi-rank MPI path is exercised by the tests/kokkos_mpi "
      "suite.\n\n"
      "Kokkos is initialized at import and finalized via a Python atexit hook that first releases "
      "every live Solver and zero-copy field view (flow.finalize() does the same on demand); "
      "nothing needs to be released before interpreter exit. execution_space names the active "
      "Kokkos backend (\"OpenMP\", \"Cuda\", \"HIP\"), chosen by the build's install prefix.";
  // Kokkos init + the release-then-finalize atexit hook + finalize() + execution_space: the
  // suite-wide pattern (see the file comment and peclet/core/python/kokkos_teardown.hpp).
  peclet::core::python::install(m);

  // Staggered MAC grid (THE flow solver) + the collocated/cell-centered variant. Same Python API.
  bind_solver<peclet::flow::Staggered>(m, "Solver", "SolverDiagnostics");
  bind_solver<peclet::flow::Colocated>(m, "SolverColocated", "SolverColocatedDiagnostics");
  bind_accelerator<peclet::flow::Staggered>(m, "_AndersonAccelerator");
  bind_accelerator<peclet::flow::Colocated>(m, "_AndersonAcceleratorColocated");

#ifdef PECLET_FLOW_MPI
  // Module-level: this rank's ORB block of the global (gnx,gny,gnz) grid, matching the
  // deterministic BlockDecomposer the Solver's initMpi re-derives internally (and the C++
  // tests/kokkos_mpi template). Returns (origin=[ox,oy,oz], size=[lnx,lny,lnz]); slice the global
  // SDF with these to build the local block, then Solver(*size) + init_mpi(gnx,gny,gnz). MPI_Init
  // is called if needed.
  m.def(
      "mpi_block",
      [](int gnx, int gny, int gnz, int levels, double max_imbalance) {
        ensure_mpi_init();
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        // Derive it through the SAME factory Solver::initMpi uses, so this local block size matches
        // the solver's dec_ under either decomposition mode (see Solver.set_decomposition).
        auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), gnx, gny,
                                                          gnz, levels, max_imbalance);
        auto blk = dec.block(static_cast<std::size_t>(rank));
        std::vector<int> origin{(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
        std::vector<int> bsize{(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
        return std::make_pair(origin, bsize);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"), nb::arg("levels") = 0,
      nb::arg("max_imbalance") = 1.05,
      "Return this MPI rank's ORB block of the global (gnx,gny,gnz) grid as (origin, size), each a "
      "length-3 list [x,y,z]. Use it to slice the global SDF into this rank's local block for a "
      "distributed Solver, in this order: Solver(*size), init_mpi(gnx, gny, gnz), then the "
      "geometry (init_mpi raises after set_solid / set_pressure_geometry). MPI_Init is called if "
      "needed.\n\n"
      "levels / max_imbalance choose how the partition is built and MUST match the values given "
      "to Solver.set_decomposition() -- both derive the same partition and have to agree. "
      "levels=0 (DEFAULT) = AUTO: the aligned ORB (split positions on the FINE grid snapped to a "
      "power of two, capped at 16, i.e. 5 nested multigrid levels) whenever it is exactly "
      "balanced; otherwise the coarse-first partition at depth 5, then 4, with a balance-aware "
      "split axis, the first within max_imbalance (default 1.05) that nests at least as many "
      "levels in place as the aligned ORB (every block even and at least 4 cells wide at each); "
      "otherwise the balance-aware aligned ORB if it has a smaller largest block and nests at "
      "least min(3, the aligned ORB's) levels; otherwise the aligned ORB. AUTO never cuts an "
      "axis the aligned ORB "
      "leaves whole, and changes its set of cut axes only when the aligned ORB's largest block "
      "is at least 5/4 of the mean. levels=1 = the aligned ORB as is (the default before 2026-10: "
      "reproduces "
      "an older np > 1 partition). levels >= 2 = COARSE-FIRST: build the ORB on the grid "
      "coarsened levels-1 times and refine the partition upward, so every block is a multiple of "
      "the coarsening factor BY CONSTRUCTION and the hierarchy nests for the full requested "
      "depth. Coarse-first also balances better: the aligned ORB picks a split and then rounds "
      "it (a balanced 96|96 can round to 128|64), whereas on the coarse grid one cell IS the "
      "quantum. Each depth from `levels` down to 2 is BUILT and measured, and the deepest one "
      "whose max-block/min-block ratio stays within max_imbalance is taken, else the aligned "
      "ORB.");

#ifdef PECLET_FLOW_MPI
  m.def(
      "predict_hierarchy",
      [](int gnx, int gny, int gnz, int np, int levels, bool telescope, int min_extent,
         int decomposition_levels, double max_imbalance,
         const std::optional<std::vector<double>>& weights) -> nb::object {
        std::vector<std::tuple<std::tuple<int, int, int>, int, std::tuple<int, int, int>,
                               std::tuple<int, int, int>, bool>>
            out;
        int align = 1;
        for (const auto& r : peclet::flow::CutcellMG::predict(
                 gnx, gny, gnz, np, levels, telescope, min_extent, decomposition_levels,
                 max_imbalance, weights ? &*weights : nullptr, &align))
          out.emplace_back(std::make_tuple(r.global.x, r.global.y, r.global.z), r.ranks,
                           std::make_tuple(r.block.x, r.block.y, r.block.z),
                           std::make_tuple(r.ratio.x, r.ratio.y, r.ratio.z), r.tele);
#endif
        if (!weights)
          return nb::cast(out);
        return nb::make_tuple(nb::cast(out), align);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"), nb::arg("np"), nb::arg("levels"),
      nb::arg("telescope") = true, nb::arg("min_extent") = 4, nb::arg("decomposition_levels") = 0,
      nb::arg("max_imbalance") = 1.05, nb::arg("weights") = nb::none(),
      "The pressure-multigrid hierarchy init_mpi WOULD build for this grid / rank count / level "
      "request, for the level-0 decomposition that decomposition_levels / max_imbalance describe "
      "(pass what you will pass to Solver.set_decomposition) and with or "
      "without coarse-level telescoping -- a pure function, no MPI, no allocation, any rank "
      "count. Returns one row per level: (global dims, ranks holding the level, block-0 dims, "
      "ratio to the next level, telescopes-out). Pre-flight any job with it. telescope defaults to "
      "True, the Solver's own default (set_pressure_telescope), so the forecast is the hierarchy "
      "the solver builds; telescope=False forecasts the in-place-only hierarchy, which at np > 1 "
      "can stop coarsening early. levels=0 forecasts "
      "the automatic depth (doc/mg_depth_policy.md) for the singular operator (periodic / "
      "all-Neumann / immersed solid; no outflow face), at the default bottom extent 4.\n\n"
      "weights: per-cell weights, exactly what diagnostics.rebalance_by_weights takes (global "
      "x-fastest, gnx*gny*gnz values). Given, the forecast is the hierarchy AFTER that rebalance: "
      "level 0 is its aligned weighted ORB (split planes on multiples of 2^a, a the largest "
      "alignment whose weight imbalance stays within 1.05; decomposition_levels / max_imbalance "
      "are then ignored, as the rebalance ignores set_decomposition), and a stage may be a "
      "repartition onto fewer ranks (reported as telescopes-out). The return is then the pair "
      "(rows, align), align = 2^a -- the value rebalance_by_weights returns and a co-decomposing "
      "dem takes as migrate_to_weights(weights, align=...). None (default): the rows alone, "
      "unchanged. min_extent is in cells (a count, not a length).");
  m.attr("has_mpi") = true;
#else
  m.attr("has_mpi") = false;
#endif
}
