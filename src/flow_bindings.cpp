/// @file
/// @brief nanobind module `flow` — the Kokkos cut-cell IBM Navier-Stokes solver
/// (`peclet.flow.Solver`).
///
/// Exposes peclet::flow::IbmSolver to Python: set rho/mu/dt, a body force, an SDF solid (cut-cell
/// IBM no-slip
/// + optional cut-cell pressure projection), step, read back the velocity/pressure, and query the
/// cut-cell flux divergence. Exercised by verify_poiseuille_sdflow (IBM channel) and
/// verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing). Kokkos is initialized
/// at import and finalized via Python atexit (the solver holds Kokkos Views, so callers must
/// release the Solver before exit -- del + gc.collect()). rank()/bcast_from_root() are single-rank
/// stubs (the multi-rank path lives in tests/kokkos_mpi).
///
/// Arrays cross the boundary through the shared zero-copy bridge (peclet::core::python, in core):
/// fields come back as Fortran-order (nx,ny,nz) float64 NumPy arrays referencing the field buffer,
/// and inputs are read as flat x-fastest buffers. See tpx/python/ndarray_interop.hpp.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>
#include <nanobind/stl/tuple.h>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#endif

#include "flow_ibm.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

#ifdef PECLET_FLOW_MPI
// Ensure MPI_Init has been called (mirrors the dem init_mpi idiom); safe to call repeatedly.
static void ensure_mpi_init() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}
#endif

namespace nb = nanobind;

// A solver field (flat x-fastest, ghost-stripped) -> Fortran-order (nx,ny,nz) float64 NumPy array
// for [x,y,z] indexing. The vector is moved into the array's backing store (no extra copy vs the
// old to_xyz).
template <class S>
static nb::ndarray<nb::numpy, double> field_out(S& s, std::vector<double>&& v) {
  const auto nx = static_cast<std::size_t>(s.nx());
  const auto ny = static_cast<std::size_t>(s.ny());
  const auto nz = static_cast<std::size_t>(s.nz());
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

// A Fortran-order (nx,ny,nz) float64 array -> flat x-fastest host vector (F-contiguous data() is
// already x-fastest). nanobind casts/copies the input to f_contig double if needed.
static std::vector<double> grid_in(nb::ndarray<double, nb::f_contig> a) {
  return peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a));
}

// Zero-copy export of a registered field's padded device buffer as a Fortran-order 3-D array of the
// full block shape (ex,ey,ez) = (nx+2G, ny+2G, nz+2G), x-fastest strides {1,ex,ex*ey}. Includes the
// ghost band (the flat buffer is contiguous; a ghost-stripped view would not be). The capsule owns
// a copy of the managed CCField, so the allocation outlives the array — host → NumPy referencing
// the buffer, device → DLPack for CuPy/torch. Mirrors peclet::core::python::view_to_ndarray but
// with an explicit 3-D reshape of the flat 1-D field.
template <class S>
static auto field3d_out(S& s, peclet::flow::CCField f) {
  namespace pcp = peclet::core::python;
  using Mem = peclet::flow::CCMem;
  const auto bs = s.blockShape();
  std::array<std::size_t, 3> shape{static_cast<std::size_t>(bs[0]), static_cast<std::size_t>(bs[1]),
                                   static_cast<std::size_t>(bs[2])};
  std::array<std::int64_t, 3> strides{1, static_cast<std::int64_t>(bs[0]),
                                      static_cast<std::int64_t>(bs[0]) * bs[1]};
  auto* held = new peclet::flow::CCField(f);
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<peclet::flow::CCField*>(p); });
  double* data = f.data();
  if constexpr (pcp::is_host_space_v<Mem>) {
    return nb::ndarray<nb::numpy, double>(data, 3, shape.data(), owner, strides.data(),
                                          nb::dtype<double>(), nb::device::cpu::value, 0);
  } else {
    auto dev = pcp::dlpack_device<Mem>();
    return nb::ndarray<double>(data, 3, shape.data(), owner, strides.data(), nb::dtype<double>(),
                               dev.first, dev.second);
  }
}

// Register a solver class for the given GridLayout policy (Staggered -> "Solver", Colocated ->
// "SolverColocated"). The Python API is identical across grids; only the velocity-unknown placement
// and the advection control volume differ inside Solver<Grid>.
template <class Grid>
static void bind_solver(nb::module_& m, const char* name) {
  using S = peclet::flow::Solver<Grid>;
  nb::class_<S>(m, name)
      .def(nb::init<int, int, int>(), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"),
           "Create a solver on an nx x ny x nz unit-spacing grid (x-fastest, I = x + y*nx + "
           "z*nx*ny). "
           "Set physical parameters (rho/mu/dt) and any domain BCs before the geometry / first "
           "step.")
      .def("set_rho", &S::setRho, nb::arg("rho"),
           "Set fluid density rho (physical units). Set before geometry/first step.")
      .def("set_mu", &S::setMu, nb::arg("mu"), "Set dynamic viscosity mu (physical units).")
      .def("set_dt", &S::setDt, nb::arg("dt"),
           "Set the time step dt; the momentum solve is scaled by 1/dt (well-conditioned at large "
           "dt).")
      .def("set_body_force", &S::setBodyForce, nb::arg("fx"), nb::arg("fy"), nb::arg("fz"),
           "Set the body force per unit volume (fx, fy, fz) — e.g. a mean pressure gradient.")
      .def("set_advection", &S::setAdvection, nb::arg("on"),
           "Enable/disable explicit high-order momentum advection (default scheme SOU). Off ⇒ "
           "Stokes.")
      .def("set_advection_scheme", &S::setAdvectionScheme, nb::arg("scheme"),
           "High-order advection scheme: 0 = second-order upwind (SOU, default), 1 = Koren TVD.")
      .def("set_incremental_pressure", &S::setIncrementalPressure, nb::arg("on"),
           "Toggle the rotational incremental-pressure projection.")
      .def("set_pressure_warmstart", &S::setPressureWarmstart, nb::arg("on"),
           "Seed each pressure solve from the previous step's phi (default off).")
      .def(
          "set_face_interp", &S::setFaceInterp, nb::arg("mode"),
          "DEPRECATED integer form of set_collocated_scheme (0 = plain, 9 = gauge-exact, the "
          "default). Modes 1/2/5/6/7/10 were RETIRED 2026-08-18 (ablations; 10 measured divergent) "
          "and now raise. Modes 3/4 survive as FV-constraint ablations (4 pairs with "
          "set_fv_relax). No effect on the staggered solver.")
      .def(
          "set_collocated_scheme", &S::setCollocatedScheme, nb::arg("name"),
          "Collocated cut-cell projection scheme (no effect on the staggered solver). DEFAULT "
          "since 2026-08-25: AUTO = 'ghost' where supported, falling back to 'gauge-exact' with "
          "a stderr notice on porous/variable-rho/domain-BC/Chebyshev configurations (ghost v1 "
          "limits); any explicit selection here disables AUTO.\n"
          "  'gauge-exact' (default 2026-08-18..25; the AUTO fallback) — the aperture constraint with the "
          "directional gauge-exact pressure gradient. Cheapest scheme measured (symmetric MG-PCG, "
          "no fragmentation guard). CAVEATS from the 2026-08 campaign "
          "(doc/collocated_invisible_subspace.md): possesses an attractor FAMILY of steady states "
          "(support-inconsistent gradient/constraint pair) selected by march protocol, and a "
          "rotational-update instability whose wall-blend stopgap "
          "(set_rotational_wall_weight) has a resolution-dependent margin — UNSTABLE at "
          "(R>=16 cells/radius, dt>=600); use fixed dt<=60 protocols at high resolution. "
          "Clean-protocol bias vs the staggered reference: -2.5% (R=8) -> ~+0.2% asymptote.\n"
          "  'ghost' — the fluid-only constraint scheme (binary-openness divergence + directional "
          "closures + the same gauge-exact gradient). Family-free, unconditionally stable "
          "(dt 60..1e20, no stabilizer), protocol-independent (C2), both-bed clean ladder "
          "-1.4% (R=8) -> +0.22% asymptote, Z&H anchor -0.018% at N=128. Costs: nonsymmetric "
          "BiCGStab pressure solve (~2.3-2.7x), ~1.6 KB/cell overlay (single-GPU size cap), "
          "fragmentation guard. MPI-capable (np=1,2,4 ctests); at-scale np>=16 hardening in "
          "progress. Equivalent to set_ghost_projection(True, 2, 2); the (1, 2) mixed mode "
          "stays quarantined.\n"
          "  'plain' — the legacy path: plain 1/2-1/2 face average + central-difference grad(P). "
          "FIRST order at curved cut cells (the cell gradient reads the decoupled p=0 of "
          "solid-centred neighbours, an O(1/h) gauge error), and on a dense bed it also fails to "
          "reach steady state within 800 steps at coarse resolution. Kept for reproducing "
          "published results.")
      .def(
          "set_rotational_pressure", &S::setRotationalPressure, nb::arg("on"),
          "PM I ablation (Guy-Fogelson 2005): False drops the rotational -mu*div(u*) term from "
          "the incremental pressure accumulation (constant-mu path only). Default True = shipped "
          "rotational (Timmermans) update. Also: set_collocated_scheme accepts \"gauge-2a\" -- "
          "the experimental gradient-2a one-sided branch of the gauge-exact gradient.")
      .def(
          "set_rotational_filter", &S::setRotationalFilter, nb::arg("on"), nb::arg("eps") = 0.05,
          "Experimental filtered rotational update: smooth div(u*) (mask-aware axis-wise 1-2-1, "
          "one-sided into the fluid at walls) before accumulating -mu*div into P. Keeps the O(1) "
          "pressure-relaxation gain, removes the checkerboard feedback channel.")
      .def(
          "set_rotational_weight", &S::setRotationalWeight, nb::arg("w"),
          "Under-relax the rotational term: P += ct*phi - w*mu*div(u*). 1 = shipped, 0 = PM I; "
          "small w raises the boundary-mode stability threshold ~1/w at ~1/w slower smooth-mode "
          "pressure relaxation. phi=0 stays the unique fixed point for any w>0 at every dt.")
      .def(
          "set_rotational_wall_weight", &S::setRotationalWallWeight, nb::arg("w0"),
          "Wall-banded rotational blend: at fluid cells with a solid axis-neighbour use "
          "P += (rho/dt + w0*mu/dx^2)*phi - (1-w0)*mu*div(u*); bulk keeps the full rotational "
          "update. Stabilizes the boundary rows without slowing bulk pressure relaxation, and "
          "keeps them relaxing at dt->infinity. 0 (default) = off.")
      .def("set_aperture_order", &S::setApertureOrder, nb::arg("order"),
           "Face-aperture estimator (DEFAULT 2 since 2026-08-26): 1 = legacy one-sample linear model, 2 = "
           "marching-squares (5 samples/face, O(h^2); removes the convexity bias measured at "
           "+0.59/+0.27% bed permeability at R=8/12 -- see doc/collocated_paper_plan.md row 51). "
           "For analytic geometry, exact apertures via set_openness_override are better still. "
           "Call before set_solid.")
      .def("set_fluid_only_constraint", &S::setFluidOnlyConstraint, nb::arg("mode"),
           "Fluid-only pressure constraint (route 2b, call before set_solid; collocated "
           "experiment). 1 = Design A (openness filter), 2 = Design B (SPD Kron star "
           "elimination), 0 = off.")
      .def("set_fv_relax", &S::setFvRelax, nb::arg("w"),
           "Mode-4 FV wall-flux defect-correction under-relaxation (1=full; <1 damps the stiff "
           "explicit-lagged wall term). Steady state is independent of w.")
      .def("set_velocity_streams", &S::setVelocityStreams, nb::arg("on"),
           "Toggle overlapped per-component velocity solves.")
      .def("set_implicit_advection", &S::setImplicitAdvection, nb::arg("on"),
           "Use implicit-FOU advection with deferred-correction TVD.")
      .def("set_outer_iterations", &S::setOuterIterations, nb::arg("n"),
           "Set the number of Picard/outer iterations per step.")
      .def("set_outer_tolerance", &S::setOuterTolerance, nb::arg("tol"),
           "Set the outer (Picard) convergence tolerance.")
      .def("last_outer_iterations", &S::lastOuterIterations,
           "Return the outer-iteration count from the last step().")
      .def(
          "set_velocity_solver_params",
          [](S& s, int iters, double rtol, int min_iters) {
            s.setVelocityIterations(iters);
            s.setVelocityTolerance(rtol, min_iters);
          },
          nb::arg("iters"), nb::arg("rtol") = 0.0, nb::arg("min_iters") = 2,
          "Momentum smoother control: `iters` RB-GS sweeps per component, or with rtol > 0 a "
          "TOLERANCE STOP — end the loop once the sweep's max velocity increment has contracted "
          "to rtol of the first sweep's (iters becomes the cap, min_iters the floor). Easy "
          "regimes (small nu*dt/dx^2) exit after ~3-5 sweeps; stiff regimes run to the cap "
          "unchanged. rtol = 0 (default) is the legacy fixed count, byte-identical.")
      .def("set_velocity_residual_tolerance", &S::setVelocityResidualTolerance, nb::arg("rtol"),
           "Residual-based momentum stop: a component's implicit solve ends once max|b - A u| <= "
           "rtol * max(max|b|, max|A u|) over the solved unknowns (at least one sweep always runs; "
           "a round-off floor stops a solve that cannot improve). rtol < 0 (DEFAULT) FOLLOWS THE "
           "PRESSURE SOLVER'S rtol -- the projection consumes u* and resolves the divergence the "
           "momentum residual leaves to its own tolerance; rtol > 0 fixes it; rtol == 0 restores "
           "the legacy update criterion (relative to the FIRST sweep's update, which on a "
           "warm-started near-steady step is noise and cost the whole sweep cap). Every path.")
      .def("velocity_residual_tolerance", &S::velocityResidualTolerance,
           "The momentum residual tolerance in force (resolves the follow-the-pressure default).")
      .def("set_velocity_multigrid_auto", &S::setVelocityMultigridAuto, nb::arg("cells_per_rank"),
           nb::arg("min_global_cells") = -1,
           "AUTO velocity-MG rule (when set_velocity_multigrid was never called): under MPI (np > 1) "
           "use the V-cycle once the global cells per rank fall below cells_per_rank (default 65536; "
           "0 = never), for global problems of at least min_global_cells (default 8M). Env "
           "PECLET_FLOW_VMG_AUTO_CELLS / PECLET_FLOW_VMG_AUTO_MIN_GLOBAL.")
      .def("velocity_multigrid_active", &S::velocityMultigridActive)
      .def("last_momentum_residual", &S::lastMomentumResidual,
           "max over components of max|r|/max|b| at exit of the last step's momentum solves "
           "(residual mode only; -1 otherwise).")
      .def("set_deferred_correction", &S::setDeferredCorrection, nb::arg("on"),
           "Deferred-correction advection: True (default) = 2nd order (implicit FOU + explicit "
           "high-order correction, the high-order scheme being SOU by default or Koren TVD via "
           "set_advection_scheme); False = pure implicit FOU (1st-order upwind, more dissipative "
           "but "
           "unconditionally stable at sharp shear layers).")
      .def("set_backflow_stabilization", &S::setBackflowStab, nb::arg("beta"),
           "Outflow backflow-stabilization coefficient (Bazilevs 2009 / Esmaily-Moghadam 2011): "
           "beta "
           "in [0,1] scales the dissipative outflow term that prevents backflow divergence when "
           "flow "
           "reverses at the outlet (e.g. a separated wake / BFS recirculation). Default 0.2; 0 = "
           "off. "
           "Inert where the outlet is purely outgoing.")
      .def("set_pressure_solver_params", &S::setPressureIterations, nb::arg("iters"),
           "Set the pressure smoother iteration count.")
      .def(
          "set_pressure_multigrid", [](S& s, bool, int levels) { s.setPressureLevels(levels); },
          nb::arg("on"), nb::arg("levels") = 4,
          "Set the pressure multigrid depth (levels=1 => pure RB-GS, no coarse grid).")
      .def("set_pressure_chebyshev", &S::setPressureChebyshev, nb::arg("on"),
           nb::arg("max_iter") = 120, nb::arg("rtol") = 1e-9,
           "Use the communication-light Chebyshev pressure accelerator. Mutually exclusive with the "
           "two CG drivers (on=True clears an FCG selection); on=False deselects Chebyshev and "
           "falls back to FCG if that is selected, otherwise to MG-PCG.")
      .def(
          "set_pressure_mean_removal",
          [](S& s, const std::string& scope) {
            if (scope != "all" && scope != "fine")
              throw std::runtime_error("set_pressure_mean_removal: scope must be 'all' or 'fine'");
            s.setPressureMeanRemoval(scope == "all");
          },
          nb::arg("scope"),
          "Nullspace (mean) removal scope in the pressure solve: 'fine' (DEFAULT — only the "
          "projections the Krylov iteration needs: rhs/residual, the fine-level V-cycle exit, the "
          "final iterate; ~3x fewer global-reduction latency hits per iteration, the measured "
          "winner of the multi-GPU ablation) or 'all' (legacy — every V-cycle level + after every "
          "matvec). Iteration counts are identical (A preserves mean-freeness); results equal "
          "within solver tolerance, not bit-identical.")
      .def("set_pressure_telescope", &S::setPressureTelescope, nb::arg("on"),
           "Coarse-level TELESCOPING of the pressure multigrid (multi-rank). The geometric "
           "hierarchy coarsens a level in place, which needs every rank's block even on that axis; "
           "once a block turns odd the hierarchy used to STOP, leaving a coarsest grid far too "
           "large to solve (384^3 on 1536 ranks: blocks die at 3x6x4 with the coarsest extent "
           "still 48, and the pressure iteration count grows 16.6 -> 38.7 across the ladder). With "
           "this on, a blocked level instead merges ORB siblings onto fewer ranks (the merged "
           "block's origin is its parent's split value, so parity is restored), moves the residual "
           "down / correction up within rank groups, and the roots keep coarsening on a "
           "sub-communicator to a tiny bottom. Idle ranks below the telescope point cost nothing "
           "that matters; the iteration count becomes a property of the problem rather than the "
           "rank count. OFF by default (byte-identical to before); env PECLET_FLOW_TELESCOPE=1 "
           "is the no-rebuild switch. See docs/MG_TELESCOPING_PLAN.md.")
      .def("pressure_telescope", &S::pressureTelescope)
      .def(
          "set_pressure_bottom",
          [](S& s, const std::string& m) {
            if (m == "auto") s.setPressureBottomMode(-1);
            else if (m == "smoother") s.setPressureBottomMode(0);
            else if (m == "agglomerated") s.setPressureBottomMode(1);
            else throw std::runtime_error("set_pressure_bottom: 'auto' | 'smoother' | 'agglomerated'");
          },
          nb::arg("mode"),
          "Coarse-level (bottom) solve of the pressure multigrid. A V-cycle converges at a "
          "domain-independent rate only if its COARSEST level is effectively solved, and a geometric "
          "hierarchy cannot always get small enough: an axis stops coarsening once it turns odd, and "
          "under MPI once ANY rank's block turns odd -- so at fixed cells/rank the coarsest GLOBAL "
          "grid grows with the rank count and the bottom is progressively under-solved. "
          "'auto' (DEFAULT) agglomerates the coarsest level onto a global operator and solves it "
          "exactly whenever it exceeds PECLET_FLOW_AGGLOM_EXTENT (4) cells on any axis, and uses "
          "the cheap smoothed bottom otherwise (byte-identical to 'smoother' then). 'smoother' = "
          "never agglomerate (legacy). 'agglomerated' = always. Measured on one GPU (2048x64x64 "
          "channel): a smoothed bottom needs 13.5 pressure iterations/step at 4 levels and 6.0 at "
          "6, against 4.4 at full geometric depth; agglomerated it is 4.0 at BOTH depths, and "
          "faster in wall-clock than the deep hierarchy. Works on the cut-cell IBM and "
          "ghost-projection paths (per-fluid-component null-space projection, 2026-08-13).")
      .def("set_pressure_graph_amg", &S::setPressureGraphAmg, nb::arg("on"),
           "Solve the pressure MG's coarsest level with an agglomerated mesh-agnostic algebraic "
           "multigrid (core GraphAMG), decomposition-agnostic: with levels=1 this gives a "
           "mesh-independent pressure solve that works under a WEIGHTED ORB (where the geometric "
           "coarse levels can't cleanly coarsen). Applied at the next set_solid.")
      .def("set_pressure_pcg", &S::setPressurePcg, nb::arg("on"), nb::arg("max_iter") = 200,
           nb::arg("rtol") = 1e-8,
           "Use the MG-PCG pressure accelerator (single-GPU default) and set its iteration cap and "
           "relative tolerance. on=True GENUINELY selects: it clears both competing selections "
           "(Chebyshev and FCG), so it works after set_density_mode / set_porous -- last set wins. "
           "on=False RAISES: MG-PCG is the terminal fallback of the driver dispatch, so it cannot "
           "be deselected on its own -- select the driver you want instead "
           "(set_pressure_chebyshev(True, ...) or set_pressure_fcg(True, ...)). Under "
           "set_ghost_projection the nonsymmetric operator is solved by BiCGStab; this call still "
           "sets that solve's cap/tolerance (they are shared) and does not change its method.")
      .def("set_pressure_fcg", &S::setPressureFcg, nb::arg("on"), nb::arg("max_iter") = 200,
           nb::arg("rtol") = 1e-8,
           "Use the FLEXIBLE MG-CG pressure accelerator: the same V-cycle-preconditioned CG as "
           "set_pressure_pcg, the same stopping estimate/mean removal, and the same cap+tolerance "
           "(they are shared -- it is the same solve), differing only in the beta recurrence "
           "(Polak-Ribiere r^T(z_{k+1}-z_k)/r^T z_k instead of Fletcher-Reeves). Costs one extra "
           "level-0 vector and one extra global dot per iteration; buys tolerance of a "
           "preconditioner that is not symmetric with respect to the fine operator. On a symmetric "
           "preconditioner the two betas are algebraically identical, so FCG reproduces PCG's "
           "iteration count. Unlike set_pressure_pcg this flag GENUINELY selects: on=True clears "
           "the Chebyshev selection (so it works after set_density_mode/set_porous), on=False "
           "returns to MG-PCG, and a later set_pressure_chebyshev(True, ...) wins over it.")
      .def("set_exact_crossings", &S::setExactCrossings, nb::arg("t"),
           "Analytic-SDF capability: exact wall-crossing fractions overriding the "
           "linear-interpolated theta in the momentum cut-cell overlay AND the ghost-projection "
           "closures. Flat array of 9*nx*ny*nz values, blocks [(c*3+k)]: component c's staggered "
           "point at inner cell i toward its +k neighbour; NaN = no crossing. Call BEFORE "
           "set_solid; empty list clears. Single-rank, staggered momentum placement.")
      .def("set_openness_override", &S::setOpennessOverride, nb::arg("ox"), nb::arg("oy"),
           nb::arg("oz"),
           "Analytic-SDF capability: exact face-aperture (openness) fields for the cut-cell "
           "projection, overriding the sampled-SDF openness. Inner nx*ny*nz arrays, x-fastest; "
           "ox[i] = fluid fraction of the -x face of cell i. Call BEFORE set_solid; empty ox "
           "clears. Single-rank.")
      .def("set_ghost_projection", &S::setGhostProjection, nb::arg("on"),
           nb::arg("matrix_order") = 2, nb::arg("rhs_order") = 2,
           "QUARANTINED 2026-08-18 (verification only, unsupported): superseded by the gauge-exact collocated scheme, which matches its accuracy at 5-6x lower cost. Kept as the independent second discretization behind the cross-IBM physics gate. Enabling it on the collocated grid silently selects the plain face map, since it owns the operators the gauge-exact scheme replaces. EXPERIMENTAL directional ghost-cell projection (second staggered IBM): point-based FD "
           "divergence with wall-anchored directional closures instead of the openness-weighted "
           "cut-cell projection; solved by MG-preconditioned BiCGStab. Call BEFORE set_solid. "
           "Closure orders (1=linear, 2=quadratic): (matrix_order, rhs_order) = (2,2) full "
           "quadratic 13-point matrix; (1,1) linear 7-point; (1,2) mixed/deferred — 2nd-order "
           "steady constraint on a 7-point matrix. Collocated: the same closures/matrix on the "
           "face-averaged field, plus a directional (one-sided 2nd-order) cell gradient for the "
           "-grad(P) predictor and cell correction; requires face_interp 0. v1: single-rank, "
           "periodic + IBM, stationary walls; incompatible with "
           "porous/variable-rho/domain-BC/Chebyshev.")
      .def("set_velocity_multigrid", &S::setVelocityMultigrid, nb::arg("on"), nb::arg("levels") = 4,
           nb::arg("vcycles") = 8,
           "Enable velocity (momentum) multigrid for the implicit diffusion solve.")
      .def("last_pressure_iterations", &S::lastPressureIterations,
           "Return the pressure-solver iteration count from the last step().")
      .def(
          "last_step_timers",
          [](S& s) {
            nb::dict d;
            d["step"] = s.lastStepSeconds();
            d["predictor"] = s.lastPredictorSeconds();
            d["momentum"] = s.lastMomentumSeconds();
            d["projection"] = s.lastProjectionSeconds();
            d["pressure_allreduce"] = s.lastPressureAllreduceSeconds();
            d["pressure_allreduce_count"] = s.lastPressureAllreduceCount();
            d["momentum_sweeps"] = s.lastMomentumSweeps();
            return d;
          },
          "Per-phase wall times (seconds, THIS rank, device-fenced) of the last step(): 'predictor' "
          "(ghost fills + RHS/advection/stencil builds), 'momentum' (implicit-diffusion solves), "
          "'projection' (cut-cell pressure projection), 'step' (whole step). "
          "'pressure_allreduce'/'pressure_allreduce_count' = time in / number of global reductions "
          "(MPI_Allreduce) inside the pressure solve — the latency-bound term of the distributed "
          "solve (0 on a single rank).")
      .def("set_domain_bc", &S::setDomainBc, nb::arg("face"), nb::arg("type"), nb::arg("vx") = 0.0,
           nb::arg("vy") = 0.0, nb::arg("vz") = 0.0,
           "Set a per-face domain BC (face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic/1 wall/2 "
           "inflow/3 outflow).")
      .def(
          "set_domain_bc_profile",
          [](S& s, int face, nb::ndarray<double, nb::c_contig> prof) {
            if (prof.ndim() != 3 || prof.shape(2) != 3)
              throw std::runtime_error("profile must be (Nb,Nc,3)");
            const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
            s.setDomainBcProfile(
                face, peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)), nb_,
                nc);
          },
          nb::arg("face"), nb::arg("profile"),
          "Prescribe a per-position inlet velocity profile (Nb,Nc,3) over a face (sets it to "
          "inflow).")
      .def(
          "set_pressure_geometry",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf) { s.setPressureGeometry(grid_in(sdf)); },
          nb::arg("sdf"),
          "Set an all-fluid SDF for the cut-cell pressure operator without an immersed solid (the "
          "channel/BFS domain-BC path). For a no-slip immersed BODY in an inflow/outflow domain, "
          "call "
          "set_solid(sdf, cutcell_pressure=True) instead -- do NOT also call this (a second "
          "geometry "
          "setter overwrites the SDF and wipes the solid).")
      .def(
          "set_scene",
          [](S& s, nb::ndarray<int, nb::c_contig> ni, nb::ndarray<double, nb::c_contig> nr,
             nb::ndarray<int, nb::c_contig> ii, nb::ndarray<double, nb::c_contig> ir,
             bool periodic) {
            s.setScene(std::vector<int>(ni.data(), ni.data() + ni.size()),
                       std::vector<double>(nr.data(), nr.data() + nr.size()),
                       std::vector<int>(ii.data(), ii.data() + ii.size()),
                       std::vector<double>(ir.data(), ir.data() + ir.size()), periodic);
          },
          nb::arg("node_ints"), nb::arg("node_reals"), nb::arg("instance_ints"),
          nb::arg("instance_reals"), nb::arg("periodic") = false,
          "Install an analytic geometry scene from core's flat encoding (3 ints + 16 reals per "
          "node, 2 ints + 17 reals per instance). Geometry is in CELL UNITS on the GLOBAL inner "
          "grid. The scene is replicated on every rank, so scene-derived geometry needs no "
          "communication and -- unlike set_exact_crossings -- is NOT single-rank only. "
          "periodic=True treats the scene as min-image periodic over the global grid (one "
          "instance per body, no images); periodic=False leaves images to the caller.")
      .def("set_solid_from_scene", &S::setSolidFromScene, nb::arg("cutcell_pressure") = true,
           "Sample the installed scene onto this rank's inner grid and install it as the solid, "
           "entirely on device (no nx*ny*nz float64 host round trip).")
      .def("set_exact_crossings_from_scene", &S::setExactCrossingsFromScene,
           "Compute EXACT wall crossings from the installed scene, on device, on every rank -- the "
           "in-solver replacement for set_exact_crossings + scripts/exact_apertures_spheres.py. "
           "Bisection, not Newton: only SIGN correctness is guaranteed for bound-only leaves.")
      .def("has_scene", &S::hasScene)
      .def(
          "set_instance_motion",
          [](S& s, int i, std::array<double, 3> lin, std::array<double, 3> ang,
             std::optional<std::array<double, 3>> center) {
            s.setInstanceMotion(i, lin, ang, center ? center->data() : nullptr);
          },
          nb::arg("instance"), nb::arg("lin_vel") = std::array<double, 3>{0.0, 0.0, 0.0},
          nb::arg("ang_vel") = std::array<double, 3>{0.0, 0.0, 0.0},
          nb::arg("center") = nb::none(),
          "Rigid-body motion of one scene instance, in CELL UNITS per time (the scene lives on the "
          "global inner grid). Any nonzero component switches the solver onto the moving-geometry "
          "path: the momentum operator's no-slip datum becomes the local wall velocity and the "
          "cut-cell projection gains the wall's own volume flux. All-zero keeps the static path, "
          "bit for bit. Staggered grid only in v1 (no ghost projection / porous / variable rho).")
      .def("set_wall_flux_divergence", &S::setWallFluxDivergence, nb::arg("on"),
           "Rung 3 on/off (default on = correct physics). Off leaves the moving wall's no-slip "
           "datum in the momentum operator but drops the wall's own volume flux from the cut-cell "
           "projection -- the configuration the Galilean gate uses to exhibit the failure the term "
           "fixes.")
      .def("has_moving_instance", &S::hasMovingInstance,
           "True when at least one scene instance carries a nonzero velocity.")
      .def("scene_instance_count", &S::sceneInstanceCount)
      .def(
          "set_instance_transform",
          [](S& s, int i, std::array<double, 3> t, std::array<double, 4> q) {
            s.setInstanceTransform(i, t, q);
          },
          nb::arg("instance"), nb::arg("translation"),
          nb::arg("quat") = std::array<double, 4>{0.0, 0.0, 0.0, 1.0},
          "Move one scene instance (quat as (x,y,z,w)). Takes effect at the next "
          "rebuild_geometry(): the SDF, the cut-cell overlay, the apertures and the pressure "
          "operator are all derived from the transforms, so moving one without rebuilding would "
          "run the solver on stale geometry.")
      .def("set_fresh_cell_seed", &S::setFreshCellSeed, nb::arg("on"),
           "Fresh-cell policy for MOVING geometry. A point a body has just uncovered inherits "
           "whatever the solid held there -- zero, or a stale masked value -- which is one of the "
           "two textbook mechanisms behind spurious force oscillations in a moving-boundary IBM. "
           "True seeds it with the LOCAL WALL VELOCITY instead, so it starts moving with the "
           "surface that released it. Bounded (no extrapolation), no new field, and exactly the "
           "old behaviour when nothing moves. DEFAULT TRUE since 2026-08-30: measured on an "
           "oscillating sphere translating through the grid, it removes a resolution-INDEPENDENT "
           "+2.6..2.9% drag bias, cuts the spurious force oscillation 20-50x to within 17% of the "
           "non-moving floor, and improves the resolved CFD-DEM loop's total-momentum "
           "conservation 95x. Pass False for the pre-2026-08-30 behaviour.")
      .def("refresh_wall_velocity", &S::refreshWallVelocity,
           "Re-derive ONLY the wall-velocity fields and the momentum operator that folds them in, "
           "for a driver that changes an instance's VELOCITY every step while its transform stays "
           "put -- the linearised oscillating body, a shear cell driven by counter-moving plates. "
           "set_instance_motion alone does not reach those fields (they are built in "
           "set_solid_from_scene), so the alternative is a full rebuild_geometry() per step to "
           "update a boundary condition. Refuses if a transform has changed since the last "
           "geometry build, because it does NOT re-sample the SDF, the apertures, the ownership "
           "field or the pressure operator. Velocity and pressure are untouched.")
      .def("rebuild_geometry", &S::rebuildGeometry,
           "Re-derive all geometry from the current instance transforms. Velocity and pressure are "
           "PRESERVED across the rebuild (set_solid zeroes them by design). Full rebuild: the "
           "measured cost is ~65% momentum/IBM stencils, ~35% pressure/MG, scene sampling in the "
           "noise. Cells uncovered by the motion inherit zero, not an extrapolated fluid value.")
      .def(
          "hydro_force_torque_reaction",
          [](S& s) {
            std::vector<double> v = s.hydroForceTorqueReaction();
            const std::size_t n = v.size() / 6;
            return peclet::core::python::vector_to_ndarray(
                std::move(v), {2, n, 3},
                {static_cast<std::int64_t>(3 * n), 3, 1});
          },
          "Hydrodynamic force and torque per scene instance from the DISCRETE REACTION -- the "
          "momentum the fluid actually lost to each body, assembled from the composed step's "
          "budget (time term, body force, and the viscous fluxes at u*; the pressure telescopes "
          "inside each owner region and needs no field). Shape (2, n_instances, 3): [0] force, "
          "[1] torque about the instance centre. This is the RECOMMENDED coupling force: exactly "
          "conservative (sum = f * N_fluid-cells at steady state, to solver residual) and as "
          "accurate as the flow solution it sustains. The traction integral "
          "(hydro_force_torque) under-reads by a resolution-independent ~29% and remains as a "
          "diagnostic. Staggered; EXPLICIT advection is carried (the stashed advective RHS term "
          "is subtracted); implicit advection / porous / variable properties / domain BCs are "
          "refused loudly (v2). Atomics: tolerance-reproducible, not bitwise.")
      .def(
          "fluid_momentum_cells",
          [](S& s) {
            const auto n = s.fluidMomentumCells();
            return std::array<long, 3>{n[0], n[1], n[2]};
          },
          "Unmasked (fluid) staggered momentum cells per component -- the exact discrete datum of "
          "the reaction identity: at steady state, sum over bodies of F_c = f_c * N_c.")
      .def(
          "hydro_force_torque",
          [](S& s) {
            std::vector<double> v = s.hydroForceTorque();
            const std::size_t n = v.size() / 12;
            return peclet::core::python::vector_to_ndarray(
                std::move(v), {4, n, 3},
                {static_cast<std::int64_t>(3 * n), 3, 1});
          },
          "DIAGNOSTIC ONLY -- use hydro_force_torque_reaction for the coupling force. The "
          "reconstructed-traction loads, shape (4, n_instances, 3): [0] force, [1] torque about "
          "the instance centre, [2] the pressure part, [3] the viscous part ([0] == [2] + [3]). "
          "The cut-cell surface integral of (-p I + mu(grad u + grad u^T)) against the exact "
          "aperture wall-area vector; its central-difference gradient under-reads the drag by a "
          "resolution-independent ~29% (measured; see the design note), which is why it is kept "
          "only to keep that inconsistency visible. Atomics: tolerance-reproducible, not bitwise.")
      .def(
          "reaction_budget_terms",
          [](S& s) {
            std::vector<double> v = s.reactionBudgetTerms();
            return peclet::core::python::vector_to_ndarray(std::move(v), {2, std::size_t(3)},
                                                           {3, 1});
          },
          "Diagnostic decomposition of the reaction identity, shape (2, 3): [0] the unsteady sum "
          "sum_i (rho/dt)(u_i - u^n_i) and [1] the advective sum sum_i A_i, both over the FLUID "
          "momentum cells of each component. The full discrete identity is sum_bodies F_c = "
          "f_c*N_c + sum fb + [1]_c - [0]_c; the Stokes form drops both because they vanish at "
          "steady state with advection off. sum_i A_i is the advection operator's net momentum "
          "flux through the cut walls -- an O(h) property of that operator, not of the budget.")
      .def("periodic_image_overlap_cells", &S::periodicImageOverlapCells,
           "Cells on this rank whose solid/fluid sign was decided by a periodic IMAGE of an "
           "instance wider than the box (set_solid_from_scene warns when nonzero): the scene "
           "evaluates the UNION of images, so a slab wider than the box refills any cavity carved "
           "from it. 0 when no instance is that wide or the images agree.")
      .def("wall_area_probe", &S::wallAreaProbe,
           "Diagnostic: [sum_c x_c*A_wall_x, sum_c y_c*A_wall_y, sum_c z_c*A_wall_z] over all cut "
           "cells. Must equal -V_solid componentwise if the aperture wall-area vectors are right, "
           "so it separates a force-integral error in the GEOMETRY from one in the traction.")
      .def("wall_flux_imbalance", &S::wallFluxImbalance,
           "Sum over cells of u_wall . A_wall -- the compatibility datum of the singular pressure "
           "problem. Exactly zero for a translating body in a periodic box; small but nonzero for "
           "rotation. Reported, not corrected.")
      .def(
          "get_cut_owner",
          [](S& s) {
            std::vector<int> v = s.getCutOwner();
            const auto nx = static_cast<std::size_t>(s.nx());
            const auto ny = static_cast<std::size_t>(s.ny());
            const auto nz = static_cast<std::size_t>(s.nz());
            return peclet::core::python::vector_to_ndarray(
                std::move(v), {nx, ny, nz},
                {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
          },
          "Per-inner-cell owning scene instance (the argmin behind the sampled SDF), (nx,ny,nz) "
          "int32; -1 before set_solid_from_scene. Moving geometry reads its wall velocity off the "
          "owner and resolved CFD-DEM posts the hydrodynamic force back to it.")
      .def(
          "set_solid",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf, bool cutcell_pressure,
             const std::string& /*pressure_coarse*/) {
            s.setSolid(grid_in(sdf), cutcell_pressure);
          },
          nb::arg("sdf"), nb::arg("cutcell_pressure") = false, nb::arg("pressure_coarse") = "const",
          "Set the solid SDF as a Fortran-order (nx,ny,nz) float64 array (negative inside the "
          "solid, positive in fluid). cutcell_pressure=True enables the open-face-weighted "
          "cut-cell "
          "pressure operator (proper no-slip); it composes with domain BCs, so this is the single "
          "call for a no-slip immersed body in an inflow/outflow domain.")
      .def(
          "set_state",
          [](S& s, nb::ndarray<double, nb::f_contig> u, nb::ndarray<double, nb::f_contig> v,
             nb::ndarray<double, nb::f_contig> w) {
            s.uploadVelocity(grid_in(u), grid_in(v), grid_in(w));
          },
          nb::arg("u"), nb::arg("v"), nb::arg("w"),
          "Upload an initial velocity field (u,v,w each a Fortran-order (nx,ny,nz) float64 array).")
      .def("step", &S::step,
           "Advance the solver one time step (semi-implicit: diffusion + projection).")
      .def(
          "set_velocity",
          [](S& s, int c, nb::ndarray<double, nb::f_contig> a) { s.setVelocity(c, grid_in(a)); },
          nb::arg("c"), nb::arg("array"),
          "Write component c's inner velocity from a Fortran-order (nx,ny,nz) float64 array "
          "(solid rows re-masked). Initial-condition hook: a uniform stream around a fixed body "
          "is the Galilean twin of the same body towed through fluid at rest.")
      .def(
          "get_u", [](S& s) { return field_out(s, s.getVelocity(0)); },
          "Return the x-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_v", [](S& s) { return field_out(s, s.getVelocity(1)); },
          "Return the y-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_w", [](S& s) { return field_out(s, s.getVelocity(2)); },
          "Return the z-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_p", [](S& s) { return field_out(s, s.getPressure()); },
          "Return the physical pressure as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_ox", [](S& s) { return field_out(s, s.getOpenness(0)); },
          "TEMP: -x face openness (fluid area fraction) per inner cell, (nx,ny,nz).")
      .def(
          "get_oy", [](S& s) { return field_out(s, s.getOpenness(1)); },
          "TEMP: -y face openness per inner cell, (nx,ny,nz).")
      .def(
          "get_oz", [](S& s) { return field_out(s, s.getOpenness(2)); },
          "TEMP: -z face openness per inner cell, (nx,ny,nz).")
      .def(
          "get_ox_proj", [](S& s) { return field_out(s, s.getOpennessProj(0)); },
          "-x face openness whose fluxes the projection CONSERVES (binary/COUPLED under "
          "set_ghost_projection, geometric cut-cell otherwise). Use for flux bookkeeping "
          "(peclet.pnm extract_network_flow).")
      .def(
          "get_oy_proj", [](S& s) { return field_out(s, s.getOpennessProj(1)); },
          "-y face openness the projection conserves (see get_ox_proj).")
      .def(
          "get_oz_proj", [](S& s) { return field_out(s, s.getOpennessProj(2)); },
          "-z face openness the projection conserves (see get_ox_proj).")
      .def(
          "get_uf", [](S& s) { return field_out(s, s.getFaceVelocity(0)); },
          "Return the divergence-free FACE x-velocity (collocated: projected MAC field; staggered: "
          "== get_u).")
      .def(
          "get_vf", [](S& s) { return field_out(s, s.getFaceVelocity(1)); },
          "Return the divergence-free FACE y-velocity (collocated: projected MAC field; staggered: "
          "== get_v).")
      .def(
          "get_wf", [](S& s) { return field_out(s, s.getFaceVelocity(2)); },
          "Return the divergence-free FACE z-velocity (collocated: projected MAC field; staggered: "
          "== get_w).")
      // --- Named field registry (multiphysics field container) ---------------------------------
      .def(
          "add_field", [](S& s, const std::string& name) { s.addField(name); }, nb::arg("name"),
          "Register a new zero-initialised cell-centred field on the grid (for transported scalars "
          "or material properties). Idempotent.")
      .def(
          "has_field", [](S& s, const std::string& name) { return s.hasField(name); },
          nb::arg("name"), "Whether a field of this name is registered.")
      .def(
          "field_names", [](S& s) { return s.fieldNames(); },
          "Names of all registered fields (velocity u/v/w, p, sdf, plus any added), sorted.")
      .def(
          "get_field", [](S& s, const std::string& name) { return field_out(s, s.getField(name)); },
          nb::arg("name"),
          "Return a registered field's inner region as a Fortran-order (nx,ny,nz) float64 array.")
      .def(
          "set_field",
          [](S& s, const std::string& name, nb::ndarray<double, nb::f_contig> a) {
            s.setField(name, grid_in(a));
          },
          nb::arg("name"), nb::arg("array"),
          "Write a Fortran-order (nx,ny,nz) float64 array into a registered field's inner region "
          "(ghosts refilled on the next exchange_field/step).")
      .def(
          "field_view",
          [](S& s, const std::string& name) { return field3d_out(s, s.fieldView(name)); },
          nb::arg("name"),
          "Zero-copy view of a registered field's full padded buffer as a Fortran-order "
          "(nx+2g, ny+2g, nz+2g) array (g = ghost_width); host → NumPy, device → DLPack (CuPy).")
      .def(
          "exchange_field", [](S& s, const std::string& name) { s.exchangeField(name); },
          nb::arg("name"),
          "Fill a registered field's ghost cells (cross-rank + periodic under MPI; periodic "
          "single-rank).")
      .def(
          "exchange_field_add", [](S& s, const std::string& name) { s.exchangeFieldAdd(name); },
          nb::arg("name"),
          "Add-reduce halo: fold ghost-layer deposits back onto their owner (cross-rank + "
          "periodic). "
          "The particle->grid deposition primitive for MPI CFD-DEM; single-rank non-periodic "
          "no-op.")
      // --- Scalar transport (advection-diffusion) ----------------------------------------------
      .def(
          "add_scalar",
          [](S& s, const std::string& name, double diffusivity, int scheme, int iters) {
            s.addScalar(name, diffusivity, scheme, iters);
          },
          nb::arg("name"), nb::arg("diffusivity") = 0.0, nb::arg("scheme") = 1,
          nb::arg("iters") = 50,
          "Register a transported scalar (temperature/concentration/…): constant diffusivity (grid "
          "units), advection scheme 0=FOU/1=Koren TVD/2=SOU, and RB-GS diffusion sweeps. The "
          "scalar "
          "is a registered field (get_field/set_field/field_view). Requires geometry "
          "(set_solid/set_pressure_geometry) for the openness-weighted operators.")
      .def(
          "set_scalar_bc",
          [](S& s, const std::string& name, int face, int type, double value) {
            s.setScalarBc(name, face, type, value);
          },
          nb::arg("name"), nb::arg("face"), nb::arg("type"), nb::arg("value") = 0.0,
          "Scalar boundary condition on a domain face (0..5 = -x,+x,-y,+y,-z,+z): type 0 periodic, "
          "1 Neumann zero-flux (adiabatic), 2 Dirichlet value. Single-rank.")
      .def(
          "has_scalar", [](S& s, const std::string& name) { return s.hasScalar(name); },
          nb::arg("name"), "Whether a transported scalar of this name is registered.")
      .def(
          "advance_scalars", [](S& s) { s.advanceScalars(); },
          "Advance all registered scalars one dt with the current velocity (also done by step()).")
      // --- Geometric VoF colour field (rung V2a) -----------------------------------------------
      .def(
          "enable_vof", [](S& s) { s.enableVof(); },
          "Enable the geometric (PLIC + Weymouth-Yue) VoF colour field 'C'. Registers 'C' as an "
          "ordinary cell field (get_field/set_field/field_view/closure input) and allocates the "
          "colour field's OWN g=3 working block + halo; the solver's G=2 is untouched. C is "
          "advected at the end of every step() with the just-projected face velocities.\n\n"
          "RUNG V2a SCOPE — read before using: there is NO surface tension (rung V4) and NO "
          "momentum-consistent transport (rung V2b). Mass and momentum are advected by different "
          "fluxes, so a mixed cell carries a spurious interfacial momentum source of order "
          "delta-rho; the literature (Rudman 1998, Arrufat 2021) is unambiguous that this breaks "
          "down around density ratio 1000 unless the resolution is absurd. **V2a is valid only at "
          "MODEST density ratios** for cases with motion. A high-ratio case AT REST (the "
          "hydrostatic acid test) is exact, because there is no momentum to mis-advect. "
          "COLLOCATED (SolverColocated) is supported since rung V8 (2026-09-02), ALL-FLUID only: "
          "the colour is advected by the PROJECTED face field uf_/vf_/wf_ (the field the ABC "
          "approximate projection makes exactly divergence-free, which is what Weymouth-Yue's "
          "conservation proof needs), and every interfacial/body force is a FACE acceleration with "
          "the cell taking the average of its two faces. An immersed solid on that grid THROWS "
          "(the cut-cell colour transport of rung V5a is staggered-only), and so does "
          "enable_vof_momentum. On the collocated grid the SPURIOUS-CURRENT number to read is the "
          "FACE field (get_uf/get_vf/get_wf): the CELL field additionally carries the approximate "
          "projection's invisible odd-even mode, which centerToFace annihilates and the projection "
          "therefore cannot remove. An IMMERSED SOLID is supported since "
          "rung V5a: the geometric fluxes are openness-weighted and the update is done in "
          "fluid-volume units, so sum eps_eff*C is conserved exactly against the projection's own "
          "openness-weighted divergence; it needs set_solid(..., cutcell_pressure=True) (the "
          "staircase operator has no face openness to weight with) and it approximates the "
          "solid-clipped flux polygon by (whole-cell PLIC slab) x (open area) — see "
          "vof_diagnostics()['clipped_volume'].")
      .def(
          "set_vof",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setVof(grid_in(a)); }, nb::arg("array"),
          "Set the colour field from a Fortran-order (nx,ny,nz) float64 array: C = the LIQUID "
          "volume fraction of the cell, in [0,1]. Enables VoF if it is not on yet. Initialise it "
          "SHARP (0/1 with the exact fraction in the interface cells) — geometric VoF keeps a "
          "sharp interface and a diffuse initial profile is simply a different problem.")
      .def(
          "get_vof", [](S& s) { return field_out(s, s.getVof()); },
          "The colour field's inner region as a Fortran-order (nx,ny,nz) float64 array "
          "(== get_field('C')).")
      .def(
          "vof_max_courant", [](S& s) { return s.vofMaxCourant(); },
          "INTERFACE-LOCAL Courant number max|uf|*dt/h over the faces of mixed cells and their "
          "face neighbours, with the current velocity and dt (all-reduce max under MPI). This — "
          "not the global max — is the number Weymouth's boundedness bound applies to: a global "
          "max over-throttles badly (measured on Zalesak: 0.314 at a quiescent domain corner while "
          "the interface never exceeded 0.157). Pick dt as dt*cfl_target/vof_max_courant().")
      .def(
          "vof_last_courant", [](S& s) { return s.vofLastCourant(); },
          "The interface-local Courant number of the step just taken (0 before the first step).")
      .def(
          "set_vof_cfl_limit", [](S& s, double v) { s.setVofCflLimit(v); }, nb::arg("value"),
          "Weymouth-Yue boundedness cap on the interface-local Courant number. Default 0.25 — the "
          "PROVEN 3D bound 1/(2(N-1)) from Weymouth's thesis eq. A.33; the widely-quoted 0.5 is "
          "the 2D value. step() throws when it is exceeded. Conservation is INDEPENDENT of "
          "boundedness (the telescoping is algebraic), so an over-CFL run loses 0<=C<=1, not "
          "volume.")
      .def(
          "vof_cfl_limit", [](S& s) { return s.vofCflLimit(); },
          "The Weymouth-Yue boundedness cap currently in force.")
      // --- rung V5a (WO-Q): transport through an immersed solid --------------------------------
      .def(
          "advect_vof", [](S& s, double dt) { s.advectVofKinematic(dt); }, nb::arg("dt"),
          "KINEMATIC colour advection: advance C ONCE with the solver's CURRENT face velocity over "
          "`dt` (seconds), with NO Navier-Stokes step. This is the advection-benchmark entry point "
          "(Zalesak, LeVeque, and the cut-cell conservation gates): a frozen projected velocity "
          "advecting a colour field is a pure statement about the transport scheme, with the "
          "momentum and pressure solves out of the picture.\\n\\n"
          "It THROWS if the current velocity is not discretely divergence-free to 1e-10 "
          "(max_open_divergence()). That is not a nicety: Weymouth-Yue's exact conservation is "
          "CONDITIONAL on sum_f o_f u_f = 0 per cell, because the dilation term adds H(C-1/2) "
          "times that residual to every full cell's volume budget. Run step() to a steady state "
          "(or "
          "project()) and advect with the solver's own output; never with an analytic sample, "
          "which is solenoidal only to O(h^2).")
      .def(
          "set_vof_step_parity", [](S& s, long n) { s.setVofStepParity(n); }, nb::arg("n"),
          "Set the sweep-permutation counter of the NEXT colour advection: the Weymouth-Yue sweep "
          "order is kWySweepPerm[n % 6], cycled so no axis is systematically favoured. Exposed so "
          "a benchmark can hold the permutation fixed (n constant) or resume a run across a "
          "restart. Default: it increments once per advection from 0.")
      .def(
          "vof_step_parity", [](S& s) { return s.vofStepParity(); },
          "The sweep-permutation counter of the next colour advection.")
      .def(
          "set_vof_cutcell_flux_clamp", [](S& s, bool on) { s.setVofCutFluxClamp(on); },
          nb::arg("on"),
          "Ablation: Weymouth's admissible-interval clamp on the openness-weighted cut-cell flux "
          "(ON by default). The clamp bounds |F| by what the DONOR actually holds — at most "
          "eps*C liquid and at most eps*(1-C) gas of the fluid volume o*|a| swept through the face "
          "— applied to the one value both neighbours share, so conservation still telescopes "
          "bit-exactly. It is what makes the whole-cell-PLIC-times-open-area flux approximation "
          "BOUNDED. Measured with it off on a 24^3 packing at CFL 0.2: the [0,1] clip fires at "
          "3.2e-5 liquid volume per step and the conserved functional drifts 1.3e-8 in 30 steps; "
          "with it on the clip stops firing.")
      .def(
          "vof_filled_colour", [](S& s) { return field_out(s, s.getVofFilledColour()); },
          "The colour field INCLUDING the neutral solid-band fill — what the MYC and "
          "height-function stencils actually read — as a Fortran-order (nx,ny,nz) array. "
          "get_vof()/'C' is the canonical field and carries EXACTLY 0 in solid cells; the fill is "
          "a stencil device regenerated at every ghost exchange (three passes with a shrinking "
          "depth "
          "budget, src/vof/cutcell.hpp), and it is what makes the wall look 90-degree neutral "
          "instead of perfectly non-wetting.")
      .def(
          "vof_geometry", [](S& s, int which) { return field_out(s, s.getVofGeometry(which)); },
          nb::arg("which"),
          "The cut-cell geometry the colour block runs on, as a Fortran-order (nx,ny,nz) array: "
          "0 = the cell fluid fraction eps (4^3-subsampled, a multiple of 1/64), 1/2/3 = the "
          "openness of the +x/+y/+z face of each cell (the ADVECTOR's high-face convention, one "
          "cell shifted from the solver's ox/oy/oz), 4 = the classification (1 = SOLID, i.e. "
          "eps == 0 AND all six faces closed).")
      .def(
          "set_contact_angle", [](S& s, double deg) { s.setContactAngle(deg); }, nb::arg("theta"),
          "Rung V5b (WO-S). Prescribe a STATIC contact angle, in DEGREES, measured THROUGH THE "
          "LIQUID, on every immersed SDF wall. Needs set_solid(..., cutcell_pressure=True) + "
          "enable_vof; with no call the wall stays at the neutral 90-degree fill of rung V5a and "
          "every V5a number is byte-identical.\n\n"
          "What it changes is PASS 1 of the solid-band fill and nothing else: the band cells "
          "receive the volume fractions of the plane that continues the fluid-side interface into "
          "the solid at angle theta (m . n_w = cos theta, with n_w = grad(sdf)/|grad(sdf)| the "
          "solid->fluid wall normal and m the PLIC normal pointing into the gas). The unmodified V3 "
          "height-function/MYC cascade then returns the curvature of an interface meeting the wall "
          "at theta and the V4 balanced force does the rest — NO force is added at the wall. "
          "theta = 0 fills the band with liquid (complete wetting), theta = 180 empties it.\n\n"
          "Construction (src/vof/wetting.hpp): the anchor fluid cell is found by walking along n_w; "
          "its fluid-only Youngs normal supplies the AZIMUTH of the contact line (its in-wall "
          "component, which a fluid-restricted stencil measures to ~1e-14 on a plane) while the "
          "angle to the wall is replaced by the prescribed one — the wall-normal component of a "
          "fluid-only estimator is unusable (measured 23 deg of error at theta = 30 against 2.3 "
          "for the full stencil, gate G0e). The plane is then anchored by matching the anchor "
          "cell's own liquid volume, which makes the fill EXACTLY idempotent: an interface that "
          "already meets the wall at theta is reproduced to 1e-15 (gate G0a), so theta is a fixed "
          "point of the scheme.")
      .def(
          "set_contact_angle_field",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setContactAngleField(grid_in(a)); },
          nb::arg("theta"),
          "Per-cell static contact angle in DEGREES, as a Fortran-order (nx,ny,nz) array. Only the "
          "value AT the solid band cell being filled is read, so cells away from a wall are "
          "irrelevant. theta is a field so the dynamic-angle rung (V6) changes only what fills it.")
      .def(
          "contact_angle", [](S& s) { return s.contactAngle(); },
          "The prescribed static contact angle in degrees (90 if none was set).")
      .def(
          "set_contact_angle_pivot", [](S& s, int m) { s.setContactAnglePivot(m); }, nb::arg("mode"),
          "ABLATION — how the theta-plane is anchored in the fluid cell. 0 (DEFAULT) match the "
          "anchor cell's liquid volume with plicAlpha; 1 pass through the PLIC centroid p_f (the "
          "Afkhami-Bussmann / Basilisk contact.h rule); 2 the work order's c = p_f - sdf(p_f) n_w; "
          "3 the contact line on the wall. Modes 0/1/3 are exactly idempotent (1e-15 on gate G0a); "
          "mode 2 is NOT — projecting the centroid along n_w shifts the plane by -sdf(p_f) "
          "cos(theta), measured 0.26 in cell fraction at theta = 60, with the wrong sign (it "
          "removes liquid from the band for a wetting angle).")
      .def(
          "contact_angle_diagnostics",
          [](S& s) {
            const auto d = s.contactAngleDiagnostics();
            nb::dict r;
            r["contact_cells"] = d.contactCells;
            r["neighbour_cells"] = d.neighbourCells;
            r["pure_cells"] = d.pureCells;
            r["parallel_cells"] = d.parallelCells;
            r["neutral_cells"] = d.neutralCells;
            r["unfilled_cells"] = d.unfilledCells;
            r["mean_apparent_angle"] = d.meanApparentAngle;
            r["set_angle"] = d.setAngle;
            return r;
          },
          "The solid-band census of the CURRENT colour field on this rank: how many band cells each "
          "branch of pass 1 wrote ('contact_cells' the theta plane of the cell's own anchor, "
          "'neighbour_cells' the mean of the anchor's MIXED neighbours' planes where the anchor "
          "itself is pure phase, 'pure_cells' the pure-phase "
          "continuation C_s = C_f, 'parallel_cells' an interface parallel to the wall where no "
          "rotation is defined, 'neutral_cells' the WO-Q neutral-mean fallback, 'unfilled_cells' "
          "left untouched), plus 'mean_apparent_angle' — the mean angle the fluid-only normal "
          "reported at the contact cells BEFORE the rotation, in degrees. That last number is the "
          "direct read-out of how far the fluid-side interface still is from the prescribed angle, "
          "measured on the fill's own data rather than on a post-processed shape.")
      .def(
          "vof_has_geometry", [](S& s) { return s.vofHasGeometry(); },
          "True when the colour advection is running the CUT-CELL (openness-weighted) kernels, "
          "i.e. an immersed solid is present and set_solid ran with cutcell_pressure=True. False "
          "means "
          "the uncut rung-V1 kernels are running, byte-identically to a solid-free build.")
      .def(
          "vof_diagnostics",
          [](S& s) {
            const auto d = s.vofDiagnostics();
            nb::dict r;
            r["sum"] = d.sumC;
            r["min"] = d.minC;
            r["max"] = d.maxC;
            r["mixed"] = d.mixed;
            r["wisps"] = d.wisps;
            r["volume"] = d.volume;
            r["raw_volume"] = d.rawVolume;
            r["solid_sum"] = d.solidSumC;
            r["solid_fill_sum"] = d.solidFillSum;
            r["min_fluid"] = d.minCFluid;
            r["max_fluid"] = d.maxCFluid;
            r["clipped_volume"] = d.clippedVolume;
            r["clipped_signed"] = d.clippedSigned;
            r["cut_cells"] = d.cutCells;
            r["clamped_faces"] = d.clampedFaces;
            r["solid_cells"] = d.solidCells;
            // rung V-BC (WO-R): the boundary term of the exact colour budget
            const auto v = s.vofBcVolumes();
            double in = 0.0, out = 0.0;
            for (int f = 0; f < 6; ++f) {
              if (v[f] > 0.0)
                in += v[f];
              else
                out -= v[f];
            }
            r["inflow_volume"] = in;
            r["outflow_volume"] = out;
            return r;
          },
          "Colour census over THIS RANK's inner cells: sum (cell-volume units), min, max, the "
          "number of mixed cells (0<C<1) and of wisps (C within 1e-8 of 0 or 1). No clipping is "
          "applied at this rung, so min/max may leave [0,1] if the CFL cap is raised.\n\n"
          "With an immersed solid (rung V5a) the dict also carries the cut-cell quantities, all "
          "zero without one: 'volume' = sum eps_eff*C over fluid cells, the functional the "
          "openness-weighted scheme conserves EXACTLY (eps_eff = max(eps, 1/64), the 4^3 "
          "subsampling resolution — see src/vof/cutcell.hpp rule 1); 'raw_volume' = sum eps*C, "
          "which differs only on eps==0 cells that still own an open face and is therefore NOT the "
          "conserved functional; 'solid_sum' = sum of C over solid cells (0 by construction — the "
          "neutral band fill lives on the working block, not on 'C'); 'min_fluid'/'max_fluid' over "
          "UNCUT fluid cells (eps==1), where Weymouth's boundedness applies verbatim; "
          "'clipped_volume' / 'clipped_signed' = the liquid volume the cut-cell clip moved during "
          "the last advection (a TRIPWIRE on the flux approximation, not a mechanism: if it is not "
          "negligible the fix is the solid-clipped flux polygon); 'cut_cells' / 'solid_cells'.\n\n"
          "'inflow_volume' / 'outflow_volume' are the liquid volume that ENTERED / LEFT through "
          "the domain faces during the LAST colour advection, in the same cell-volume units as "
          "'sum' (0 unless a VoF boundary colour is set — rung V-BC). They are the advector's OWN "
          "face fluxes, so sum(C) - sum(C_0) = integral(inflow - outflow) holds to round-off "
          "whatever the interface does; vof_bc_volumes() breaks them out per face.")
      // --- two-phase open boundaries (rung V-BC, WO-R) ------------------------------------------
      .def(
          "set_vof_inflow", [](S& s, int face, double value) { s.setVofInflow(face, value); },
          nb::arg("face"), nb::arg("value"),
          "Colour of the fluid ENTERING through inflow face 'face' (0..5 = -x,+x,-y,+y,-z,+z); "
          "1 = liquid, 0 = gas. The face must already be an inflow (set_domain_bc(face, 2, ...)) "
          "or this raises.\n\n"
          "The value may be FRACTIONAL, and it then means 'this fraction of the incoming flux is "
          "liquid' — a flux statement, not a sub-cell interface position. That is exactly how it "
          "is implemented: on a face whose donor is outside the domain the geometric flux is "
          "replaced by the algebraic C_donor*a (wyFaceFluxBc), because a uniform prescribed ghost "
          "band has no usable MYC normal and the inflow colour is boundary DATA.\n\n"
          "Setting this also makes the property ghosts of that face follow the inflow colour "
          "through the registered closures (rho_ghost = rho(C_inflow)) instead of copying the "
          "interior — without it the inlet FACE density is the interior's, wrong by up to the "
          "density ratio at a liquid inlet into a gas domain. Default (nothing set): the "
          "zero-gradient copy, i.e. today's behaviour, bit for bit.")
      .def(
          "set_vof_inflow_profile",
          [](S& s, int face, nb::ndarray<double, nb::c_contig> prof) {
            if (prof.ndim() != 2)
              throw std::runtime_error("vof inflow profile must be (Nb,Nc)");
            const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
            s.setVofInflowProfile(
                face, peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)), nb_,
                nc);
          },
          nb::arg("face"), nb::arg("profile"),
          "Per-position inflow colour (Nb,Nc) on the INNER grid of the face's two perpendicular "
          "axes, resampled with the same clamp rule set_domain_bc_profile uses for the velocity. "
          "This is how a liquid distributor over part of an inlet is expressed: the colour "
          "profile says WHERE liquid enters, the velocity profile says how fast.")
      .def(
          "set_vof_backflow", [](S& s, int face, double value) { s.setVofBackflow(face, value); },
          nb::arg("face"), nb::arg("value") = 0.0,
          "inletOutlet colour on OUTFLOW face 'face' (default 0 = gas): where the boundary face "
          "velocity points back INTO the domain, the colour ghost carries this value instead of "
          "the zero-gradient copy. This is OpenFOAM's inletOutletFvPatchField (Rusche 2002 thesis "
          "section 4), the standard VoF outlet — an outlet is a place where you know what leaves "
          "(whatever is inside) but must STATE what comes back. Zero-gradient on a reversing face "
          "re-injects whatever happens to be sitting at the outlet, which for a draining film "
          "feeds the film back into the domain. Where the fluid leaves, zero-gradient is kept.")
      .def(
          "vof_bc_volumes", [](S& s) { return s.vofBcVolumes(); },
          "Signed liquid volume that crossed each of the six domain faces during the LAST colour "
          "advection, in cell-volume units, POSITIVE for liquid ENTERING the domain. Local to this "
          "rank. Six entries in face order (-x,+x,-y,+y,-z,+z); only faces carrying a domain BC "
          "are meaningful.")
      .def(
          "vof_bc_volumes_total", [](S& s) { return s.vofBcVolumesTotal(); },
          "The same, accumulated since enable_vof() or the last reset_vof_bc_volumes().")
      .def(
          "set_outflow_rho_correction", [](S& s, bool on) { s.setOutflowRhoCorrection(on); },
          nb::arg("on") = true,
          "ABLATION (WO-R item 4), DEFAULT FALSE — the measurement refuted the item.\n\n"
          "doc/variable_density_projection.md section 4 listed the missing 1/rho_f factor in "
          "bcCorrectOutflow as a defect to fix with a two-phase outflow case. Measured on that "
          "case (stratified duct, density ratio 10, max|div(open u)| of the PROJECTED field): "
          "WITHOUT the factor 8.76e-10, WITH it 9.24e-03 — seven orders worse. A projection "
          "correction cancels the divergence only if it uses the SAME face coefficient the "
          "operator row used, and the outflow face's coefficient is the RAW openness (buildRhoCoeff "
          "runs over inner cells only; the multigrid re-imposes the Dirichlet outflow face as "
          "simply open), so the plain phi difference IS the consistent correction. Removing the "
          "inconsistency would mean changing the OPERATOR, not this correction.\n\n"
          "Bitwise inert at constant density (rho_f == rho0 makes the factor exactly 1). "
          "PECLET_FLOW_OUTFLOW_RHO=1 turns it on process-wide.")
      .def(
          "outflow_rho_correction", [](S& s) { return s.outflowRhoCorrection(); },
          "Whether the 1/rho_f factor is applied to the outflow face correction (WO-R item 4).")
      .def(
          "reset_vof_bc_volumes", [](S& s) { s.resetVofBcVolumes(); },
          "Zero the per-face boundary liquid ledger (both the per-step and the running totals).")
      .def(
          "compute_vof_curvature",
          [](S& s) {
            s.computeVofCurvature();
            const auto d = s.vofCurvatureStats();
            nb::dict r;
            r["interfacial"] = d.interfacial;
            r["hf"] = d.hf;
            r["hf_mixed"] = d.hfMixed;
            r["hf_fit"] = d.hfFit;
            r["pv"] = d.pv;
            r["pv_reduced"] = d.pvReduced;
            r["no_estimate"] = d.noEstimate;
            return r;
          },
          "VoF rung V3: compute the interface curvature from the CURRENT colour field and store it "
          "in the registered cell fields 'kappa' and 'kappa_branch'. Returns THIS RANK's branch "
          "census as a dict.\n\n"
          "'kappa' is kappa = 2H in units of 1/h (CELL units — multiply by 1/h for physical "
          "units), POSITIVE for a convex blob of liquid: a sphere of liquid of radius R cells "
          "reads +2/R, an infinite cylinder +1/R, a plane 0.\n\n"
          "The cascade is Popinet's (2009): a standard height function on 7-cell column sums of C "
          "over the 3x3 transverse patch in the direction of the largest |n| (branch 1), the same "
          "in the other two directions (branch 2), then the PLIC-VOLUMETRIC paraboloid fit of "
          "Jibben et al. on a 5^3 Wendland-weighted stencil (branches 4/5) — the best "
          "cost/accuracy fallback per Han, Evrard & Desjardins, IJMF 174:104769 (2024). Branch 3 "
          "(the mixed height-position fit) is off by default; branch 0 is 'not an interfacial "
          "cell' and branch 6 is 'NO estimate', which must never occur.\n\n"
          "ALWAYS read 'kappa_branch' alongside 'kappa': kappa is 0 both where there is no "
          "interface (branch 0) and where no estimate could be made (branch 6).\n\n"
          "Two things the literature settles and this is NOT a bug: the fallback ALWAYS fires "
          "somewhere below ~5 cells per diameter (measured here: 100% of interfacial cells at "
          "D/dx = 2.8-4.4, ~19% at D/dx = 38-48 on a sphere), and with advection-realistic volume "
          "fractions the curvature error STOPS CONVERGING below C*dx ~ 1e-2 for every known "
          "method. Curvature order is not the thing to chase; transport fidelity is.")
      .def(
          "vof_curvature", [](S& s) { return field_out(s, s.getVofCurvature()); },
          "The curvature field's inner region as a Fortran-order (nx,ny,nz) float64 array, in 1/h "
          "(cell units). Call compute_vof_curvature() first. == get_field('kappa').")
      .def(
          "vof_curvature_branch", [](S& s) { return field_out(s, s.getVofCurvatureBranch()); },
          "Which cascade branch produced each cell's curvature, as a Fortran-order (nx,ny,nz) "
          "float64 array: 0 not interfacial, 1 height function, 2 height function in a "
          "non-preferred direction, 3 mixed height-position fit, 4 PLIC-volumetric paraboloid fit, "
          "5 the same with the rank-deficient 3-parameter model, 6 NO estimate. == "
          "get_field('kappa_branch').")
      .def(
          "set_vof_curvature_weight_width", [](S& s, double d) { s.setVofCurvatureWeightWidth(d); },
          nb::arg("width"),
          "Wendland support width of the PLIC-volumetric fallback fit, in CELL units. Default 2.5, "
          "which is Han et al.'s recommended pairing with the 5^3 stencil. Their translating "
          "droplet recovers first-order convergence of the spurious currents at 3.5 and loses "
          "convergence entirely at 4.5 (over-smoothing), so this is a real trade-off between the "
          "locality of the estimate and its robustness to transport error — not a free knob.")
      .def(
          "set_vof_curvature_mixed_height_fit",
          [](S& s, bool on) { s.setVofCurvatureMixedHeightFit(on); }, nb::arg("on") = true,
          "Enable cascade branch 3, the mixed height-position fit (a paraboloid through the "
          "interface positions of whichever columns closed). DEFAULT OFF and it should stay off: "
          "measured on an exact-fraction sphere at 16/32/64 it takes over the 19.5-59.6% of cells "
          "the height function cannot serve and DESTROYS the convergence of the max curvature "
          "error (order 0.00 vs 1.86 with the PLIC-volumetric fallback instead), because its data "
          "set is the columns that closed - a slope-selected, asymmetric subset whose lever-arm "
          "bias is scale invariant. Shipped as a re-measurable instrument, not a configuration.")
      // --- VoF rung V4 (WO-P): balanced-force surface tension ----------------------------------
      .def(
          "set_surface_tension", [](S& s, double sigma) { s.setSurfaceTension(sigma); },
          nb::arg("sigma"),
          "VoF rung V4: turn on the BALANCED-FORCE continuum surface force with coefficient sigma "
          "(0 = off, the default). Enables VoF and the curvature cascade, which then runs once per "
          "step at the head, from the same colour field the density closure sees.\n\n"
          "The force at the staggered velocity unknown u_c(i) is\n\n"
          "    F = sigma * kappa_f * (C(i) - C(i - s_c)) / h\n\n"
          "i.e. the colour difference is the projection's OWN face difference — the identical "
          "discrete gradient operator the pressure uses (Francois et al. 2006; Popinet 2009). That "
          "is the whole content of 'balanced force': with a constant kappa the force is exactly a "
          "discrete gradient, so the projection annihilates it and a static drop stays at machine "
          "zero. Face-interpolating a cell-centred sigma*kappa*grad(C) instead — the obvious way "
          "to reuse the per-cell body force — is not a discrete gradient of anything and leaves "
          "spurious currents of order sigma*kappa/mu that no curvature accuracy removes.\n\n"
          "UNITS: sigma is in the solver's units, in which the cell size is 1 (like rho, mu and "
          "set_body_force). SIGN: kappa is positive for a convex blob of liquid and C is the "
          "liquid fraction, so the equilibrium pressure is P = sigma*kappa*C + const — the "
          "Young-Laplace overpressure INSIDE the drop.\n\n"
          "Surface tension is EXPLICIT, so step() enforces the Brackbill capillary time step "
          "(see capillary_dt()).")
      .def(
          "surface_tension", [](S& s) { return s.surfaceTension(); },
          "The surface-tension coefficient (0 when the CSF is off).")
      .def(
          "capillary_dt", [](S& s) { return s.capillaryDt(); },
          "The Brackbill (1992) capillary time-step limit sqrt((rho_1+rho_2) h^3 / (4 pi sigma)), "
          "+inf when surface tension is off. Denner & van Wachem (2015) verified this IS the "
          "stability boundary of an explicit CSF — the prefactor 1/(4 pi), the h^{3/2} scaling and "
          "the SUM of the phase densities (both phases oscillate). The density sum is the declared "
          "phase pair under enable_vof_momentum, otherwise min(rho)+max(rho) over the field.")
      .def(
          "set_capillary_cfl", [](S& s, double f) { s.setCapillaryCfl(f); }, nb::arg("factor"),
          "Safety factor on the capillary limit: step() raises when dt > factor * capillary_dt(). "
          "Default 1.0 — the Brackbill formula was measured to be the boundary itself, so it "
          "carries no built-in margin. Set it huge to disable the check, the same escape hatch "
          "set_vof_cfl_limit is for the Weymouth-Yue cap.")
      .def(
          "vof_step_limits",
          [](S& s) {
            const auto L = s.vofStepLimits();
            nb::dict r;
            r["courant"] = L.courant;
            r["cfl_dt"] = L.cflDt;
            r["capillary_dt"] = L.capillaryDt;
            r["binding"] = L.binding;
            r["capillary_binds"] = L.capillaryBinds;
            return r;
          },
          "Both explicit two-phase step limits at the current state, and which one binds: the "
          "largest dt the interface-local Weymouth-Yue CFL cap admits ('cfl_dt'), the Brackbill "
          "capillary limit ('capillary_dt'), and min(cfl_dt, capillary_cfl*capillary_dt) with a "
          "'capillary_binds' flag. At pore-scale capillary numbers the capillary limit is expected "
          "to bind first.")
      .def(
          "csf_diagnostics",
          [](S& s) {
            const auto d = s.csfDiagnostics();
            nb::dict r;
            r["max_force"] = nb::make_tuple(d.maxForce[0], d.maxForce[1], d.maxForce[2]);
            r["orphan_faces"] =
                nb::make_tuple(d.orphanFaces[0], d.orphanFaces[1], d.orphanFaces[2]);
            r["forced_faces"] =
                nb::make_tuple(d.forcedFaces[0], d.forcedFaces[1], d.forcedFaces[2]);
            return r;
          },
          "Census of the CSF face force on THIS RANK: per component the max |F|, the number of "
          "faces that carried a force, and the number of ORPHAN faces — faces the colour jumps "
          "across but where neither cell has a curvature estimate, so the force was dropped. An "
          "orphan is a defect and must be 0; it is counted rather than hidden.")
      // --- Phase change, Part II rungs P0/P1 (WO-P01) ------------------------------------------
      .def(
          "enable_phase_change",
          [](S& s, double rho_g, double rho_l, double h_lv) {
            s.enablePhaseChange(rho_g, rho_l, h_lv);
          },
          nb::arg("rho_gas"), nb::arg("rho_liquid"), nb::arg("h_lv") = 1.0,
          "Enable VoF phase change (VOF_PLAN section 9, rungs P0/P1; the Boyd & Ling 2023 / Malan "
          "2021 pattern). Registers 'mdot' (the interfacial mass flux, kg m^-2 s^-1 in solver "
          "units, POSITIVE = evaporation) and 'pc_source' (the deposited divergence source, 1/s).\n"
          "\nWhat it does per step, at the HEAD of step(): (1) reconstruct the PLIC plane of every "
          "interfacial cell and take its POLYGON AREA A analytically (A = |m|_2 dV/dalpha, not a "
          "finite difference); (2) evaluate mdot (prescribed by set_mass_flux*, or computed by "
          "set_phase_change_thermal); (3) deposit S = mdot A (1/rho_g - 1/rho_l) into the nearest "
          "PURE GAS cell along +n, so the interfacial cell's own face velocity stays the LIQUID "
          "velocity and Weymouth-Yue keeps a field its conservation proof is entitled to; (4) "
          "regress the interface by a PLIC PLANE SHIFT, C -> C - mdot A dt/rho_l, with "
          "clip-and-redistribute along -n when a cell empties. There is NEVER a volume source in "
          "the C equation (the Hardt-Wondra smeared source leaves unresolvable residue and breaks "
          "the WY bounds).\n"
          "\nThe densities are given explicitly rather than read off a closure (the same contract "
          "as enable_vof_momentum). SCOPE at this rung: staggered grid, NO immersed solid, not "
          "composable with enable_vof_momentum; all three throw with a message. A CLOSED domain "
          "needs the net vapour production to leave somewhere: use an outflow face, or "
          "set_divergence_source() with a balancing sink, or rho_g = rho_l (then S == 0).")
      .def(
          "set_mass_flux_uniform", [](S& s, double v) { s.setMassFluxUniform(v); },
          nb::arg("mdot"),
          "P0: prescribe a UNIFORM interfacial mass flux (kg m^-2 s^-1, solver units; positive = "
          "evaporation, liquid consumed). Turns the thermal mass flux OFF.")
      .def(
          "set_mass_flux",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setMassFlux(grid_in(a)); },
          nb::arg("array"),
          "P0: prescribe a per-cell interfacial mass flux from a Fortran-order (nx,ny,nz) float64 "
          "array (== set_field('mdot') plus the ghost fill). Turns the thermal mass flux OFF.")
      .def(
          "set_phase_change_thermal",
          [](S& s, const std::string& name, double t_sat, double k_gas, double k_liquid,
             double r_int) { s.setPhaseChangeThermal(name, t_sat, k_gas, k_liquid, r_int); },
          nb::arg("scalar"), nb::arg("t_sat"), nb::arg("k_gas"), nb::arg("k_liquid"),
          nb::arg("r_int") = 0.0,
          "P1: compute mdot each step from a registered scalar (the temperature) instead of "
          "prescribing it:\n\n"
          "    mdot = ( k_g dT_g/dn - k_l dT_l/dn ) / h_lv ,   n = m/|m|_2 (LIQUID -> GAS)\n\n"
          "Each one-sided derivative is a weighted least-squares fit THROUGH the interface value "
          "over the PURE-PHASE cells of a 5^3 stencil on that side, with Malan's collinearity "
          "weight w = (d.n)^2/|d|^3 and the exact normal distance from the PLIC plane. Interfacial "
          "cells are pinned at T_G = t_sat + mdot*r_int in the energy solve through a per-cell "
          "Dirichlet mask on the scalar (r_int = 0 is the hard Dirichlet; a nonzero r_int is the "
          "Schrage/IHTR Robin condition of Bures & Sato 2021).\n\n"
          "SIGN CONVENTION — the interfacial energy balance is mdot*h_lv = (q_l - q_g).n with "
          "q = -k grad T and n pointing OUT OF THE LIQUID, which is the form above. Check it on "
          "the Stefan problem: superheated vapour behind the interface gives grad(T_g).n > 0 and "
          "mdot > 0, i.e. evaporation. The opposite pairing (k_l grad T_l - k_g grad T_g) with the "
          "SAME n would condense a superheated vapour.\n\n"
          "The energy scalar's diffusivity is whatever add_scalar was given (a CONSTANT); per-cell "
          "k(C) and the consistent rho c_p T geometric transport are the P3 upgrade. That is not a "
          "limitation for the Stefan gate, where the liquid is saturated and pinned at T_sat.")
      .def(
          "set_phase_change_thermal_off", [](S& s) { s.setPhaseChangeThermalOff(); },
          "Stop computing mdot from the temperature (back to the prescribed field).")
      .def(
          "set_divergence_source",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setDivergenceSource(grid_in(a)); },
          nb::arg("array"),
          "A PRESCRIBED extra divergence source (1/s) on the inner cells, added to the Poisson RHS "
          "beside the phase-change deposit, so the projection solves div(open u) = S. Registered "
          "as the field 'div_source'. Its use is to make a CLOSED (periodic) domain compatible "
          "with a net vapour production by carrying a balancing sink; with an outflow face the "
          "outflow does that job and this is unnecessary.")
      .def(
          "clear_divergence_source", [](S& s) { s.clearDivergenceSource(); },
          "Drop the prescribed divergence source (the field stays registered).")
      .def(
          "apply_phase_change", [](S& s, double dt) { s.applyPhaseChange(dt); }, nb::arg("dt"),
          "KINEMATIC phase-change step: build mdot / the PLIC areas / the normals, deposit the "
          "divergence source (census only — nothing is projected) and apply the interface "
          "regression over `dt`. No Navier-Stokes step, no advection. This is the P0a / P1 driver: "
          "an interface regressing under a prescribed or thermally-computed flux, with the "
          "momentum and pressure solves out of the picture.")
      .def(
          "phase_change_diagnostics",
          [](S& s) {
            const auto d = s.phaseChangeDiagnostics();
            nb::dict r;
            r["mdot_min"] = d.mdotMin;
            r["mdot_max"] = d.mdotMax;
            r["mdot_mean"] = d.mdotMean;
            r["interface_cells"] = d.interfaceCells;
            r["interface_area"] = d.area;
            r["removed_volume"] = d.removedVolume;
            r["redistributed"] = d.redistributed;
            r["deficit_cells"] = d.deficitCells;
            r["excess_cells"] = d.excessCells;
            r["source_sum"] = d.sourceSum;
            r["source_cells"] = d.sourceCells;
            r["fallback_cells"] = d.fallbackCells;
            r["unresolved"] = d.unresolved;
            r["min_C"] = d.minC;
            r["max_C"] = d.maxC;
            return r;
          },
          "Per-rank phase-change census of the LAST step: mdot extrema/mean and the interfacial "
          "cell count, the total PLIC interface area, the liquid volume the regression removed, "
          "the clip-and-redistribute ledger (|deficit| moved, and how many cells clipped at 0 / "
          "1), the deposited divergence source (its sum and how many cells received it), how many "
          "interfacial cells found NO pure gas cell within two cells along +n (the source then "
          "stays put — a fallback that must be 0 on a resolved interface), and the colour extrema.")
      .def(
          "set_csf_mode", [](S& s, int m) { s.setCsfMode(m); }, nb::arg("mode"),
          "ABLATION. 0 (default, the only production mode) evaluates the surface-tension force as "
          "sigma*kappa_f*(C(i)-C(i-s))/h at the face — the projection's own gradient operator. "
          "1 evaluates a CELL-CENTRED sigma*kappa*grad(C) and face-interpolates it with the "
          "arithmetic mean, exactly as the per-cell body-force machinery carries a rho*g field: "
          "consistent, convergent, and wrong for surface tension, because the result is not in the "
          "range of the operator the projection inverts. Shipped so the difference is a measured "
          "number (see the vof_surface_tension ctest), not an argument.")
      .def(
          "set_vof_interface_eps", [](S& s, double eps) { s.setVofInterfaceEps(eps); },
          nb::arg("eps"),
          "Wisp threshold on the curvature cascade's interfacial predicate once surface tension is "
          "on: a cell carries an interface only while eps < C < 1-eps. Default 1e-8.\n\n"
          "This is NOT optional and it is not cosmetic. Weymouth-Yue leaves round-off colour "
          "residue (measured down to -3e-35) in every cell its sweeps touch; those cells satisfy "
          "0 < C < 1, so the cascade builds a PLIC polygon of area ~0 for them and returns "
          "|kappa| up to 1e8 where the physical value is 0.125. A face between such a cell and a "
          "REAL interfacial cell then carries a surface-tension force eight orders too large. "
          "Measured on a 32^3 static droplet with eps = 0: max|u| 4.5e-4 at step 1 -> 2.7e-1 by "
          "step 20, and at 96^3 the run trips the Weymouth-Yue CFL cap. Setting eps = 0 "
          "reproduces that, which is the ablation. compute_vof_curvature() called WITHOUT surface "
          "tension keeps the rung-V3 predicate (0 < C < 1) unchanged.")
      .def(
          "vof_interface_eps", [](S& s) { return s.vofInterfaceEps(); },
          "The wisp threshold set by set_vof_interface_eps.")
      .def(
          "set_vof_kappa_frozen", [](S& s, bool on) { s.setVofKappaFrozen(on); },
          nb::arg("on") = true,
          "INSTRUMENT: stop recomputing the curvature at the head of each step and use whatever is "
          "in the 'kappa'/'kappa_branch' fields. With set_vof_kappa_constant this isolates the "
          "balanced-force identity from the curvature estimator.")
      .def(
          "set_vof_kappa_constant", [](S& s, double k) { s.setVofKappaConstant(k); },
          nb::arg("kappa"),
          "INSTRUMENT: set kappa to a constant everywhere (inner + ghosts), mark every cell as "
          "carrying a valid estimate, and freeze it. The CSF force is then EXACTLY the discrete "
          "gradient of sigma*kappa*C, so the projection must annihilate it to round-off from any "
          "colour field whatsoever — the exactness gate of rung V4, independent of curvature "
          "accuracy and of resolution.")
      .def(
          "set_rho_face_harmonic", [](S& s, bool on) { s.setRhoFaceHarmonic(on); },
          nb::arg("on") = true,
          "Use the HARMONIC instead of the arithmetic face mean of rho in the pressure projection "
          "(both the operator coefficient and the velocity correction, so the projection stays "
          "exact). DEFAULT OFF, and it should stay off: the arithmetic mean of rho IS the harmonic "
          "mean of the mobility 1/rho — the series-correct choice for a normal flux — and it is "
          "what makes the discrete hydrostatic balance EXACT, because the momentum time term and "
          "the face body force interpolate rho arithmetically and are not switched by this flag. "
          "Shipped as a measured knob for the coefficient-coarsening question (VOF_PLAN S3), not "
          "as an alternative scheme.")
      // --- Momentum-consistent VoF transport (rung V2b, WO-K) ----------------------------------
      .def(
          "enable_vof_momentum",
          [](S& s, double rg, double rl) { s.enableVofMomentum(rg, rl); }, nb::arg("rho_gas"),
          nb::arg("rho_liquid"),
          "Advect rho*u on the HALF-SHIFTED momentum control volumes with the SAME geometric "
          "fluxes, the same sweep order and one frozen dilation flag as the colour field, then "
          "recover u = (rho^c u)/rho^c. This is what makes the scheme usable above density ratio "
          "~100 (Rudman 1998; Arrufat 2021: a raindrop accurate within 15% at 15 cells/diameter "
          "instead of ~200).\n\n"
          "The two phase densities are REQUIRED, and must be the ones the rho closure produces at "
          "C=0 and C=1 — the momentum flux is rho_gas*(a-F) + rho_liquid*F with F the geometric "
          "liquid flux, so the scheme has to know which density each phase carries.\n\n"
          "Turning this on MOVES the VoF advection from the end of step() to its head (the momentum "
          "advection must precede the predictor that consumes it), so the advecting field is u^n — "
          "the previous step's projected output. u at step 0 must therefore be discretely "
          "divergence-free (rest or a uniform field both are). Requires the variable-density path, "
          "staggered layout, explicit advection, no immersed solid and no porous continuity.")
      .def(
          "vof_momentum_enabled", [](S& s) { return s.vofMomentumEnabled(); },
          "Whether momentum-consistent transport is on.")
      .def(
          "set_vof_rho_floor", [](S& s, double f) { s.setVofRhoFloorFrac(f); }, nb::arg("fraction"),
          "Floor on rho^c in the recovery divide u = (rho^c u)/rho^c, as a FRACTION of "
          "min(rho_gas, rho_liquid). Default 1e-6. rho^c leaves [rho_gas, rho_liquid] only through "
          "a wisp in the half-shifted colour and reaching zero would need C^c ~ -1/(ratio-1), so "
          "this is a guard, not a model — vof_momentum_diagnostics()['floored'] reports how many "
          "control volumes it actually touched.")
      .def(
          "vof_rho_floor", [](S& s) { return s.vofRhoFloor(); },
          "The absolute rho^c floor used by the last recovery.")
      .def(
          "set_vof_momentum_muscl", [](S& s, bool on) { s.setVofMomentumMuscl(on); },
          nb::arg("on") = true,
          "MinMod-limited linear reconstruction of the donor velocity in the momentum flux, "
          "instead of the DEFAULT plain donor-cell upwind. Both preserve the uniform-velocity "
          "identity exactly (a uniform field has a zero slope bit for bit), but on a control volume "
          "a sweep EMPTIES the slope's deviation from the volume's own velocity is amplified by "
          "drho*F/rho^c, which is unbounded in the density ratio. Measured at ratio 1e4, 50 steps: "
          "with the slope the uniform-velocity residual grows to 2.2e-10, without it it is flat at "
          "6.7e-16; at ratio 1e3 the slope is harmless. Turn it on deliberately and re-run the "
          "ratio sweep if you do.")
      .def(
          "set_vof_momentum_cell_flag", [](S& s, bool on) { s.setVofMomentumCellFlag(on); },
          nb::arg("on") = true,
          "ABLATION: use the PRESSURE-cell frozen dilation flag H(C^n-1/2) on the shifted control "
          "volume instead of its structural analogue H(C^c,n-1/2). The flag that must be shared is "
          "the one of the pair that telescopes, and both members of that pair live on the shifted "
          "volume — this switch is the literal reading of the work order, kept as a measurement.")
      .def(
          "set_vof_flux_clamp", [](S& s, bool on) { s.setVofFluxClamp(on); }, nb::arg("on") = true,
          "ABLATION: drop the Weymouth flux clamp max(0,|a|-(1-C^c_don)) <= |F| <= min(|a|,C^c_don) "
          "on the half-shifted control volume. The geometric flux is bounded by what the CURRENT "
          "cell planes see in the donor, not by the ADVECTED C^c; the gap is O(a^2) and at density "
          "ratio 1e4 a 2.6e-2 undershoot drives rho^c to -255, which the recovery would divide by. "
          "Default ON; off is how that statement stays a measured number.")
      .def(
          "vof_momentum_diagnostics",
          [](S& s) {
            const auto d = s.vofMomentumDiagnostics();
            nb::dict r;
            r["min_Cc"] = nb::make_tuple(d.minCc[0], d.minCc[1], d.minCc[2]);
            r["max_Cc"] = nb::make_tuple(d.maxCc[0], d.maxCc[1], d.maxCc[2]);
            r["sum_momentum"] = nb::make_tuple(d.sumM[0], d.sumM[1], d.sumM[2]);
            r["min_rho_c"] = d.minRhoC;
            r["floored"] = d.floored;
            r["clamped"] = d.clamped;
            return r;
          },
          "Census over THIS RANK's momentum control volumes: per-component min/max of the "
          "half-shifted colour C^c, the summed momentum rho^c u_c (the conservation census), the "
          "minimum rho^c before the floor, and the number of control volumes the floor touched.")
      .def(
          "vof_advected_velocity", [](S& s, int c) { return field_out(s, s.getVofAdvectedVelocity(c)); },
          nb::arg("component"),
          "The recovered advected velocity (rho^c u_c)/rho^c of component c on the inner cells — "
          "the momentum RHS's time base. Exposed so the uniform-velocity consistency identity can "
          "be gated on the ADVECTION ALONE, with the projection and the momentum solve out of the "
          "picture.")
      // --- Property closures + Boussinesq body force -------------------------------------------
      .def(
          "set_property_model",
          [](S& s, const std::string& target, const std::string& kind, const std::string& in0,
             const std::vector<double>& params, const std::string& in1) {
            peclet::flow::ClosureKind k;
            if (kind == "linear")
              k = peclet::flow::ClosureKind::LinearMix;
            else if (kind == "boussinesq")
              k = peclet::flow::ClosureKind::BoussinesqForce;
            else if (kind == "arrhenius")
              k = peclet::flow::ClosureKind::ArrheniusMu;
            else
              throw std::runtime_error("set_property_model: unknown kind '" + kind + "'");
            s.setPropertyModel(target, k, in0, in1, params);
          },
          nb::arg("target"), nb::arg("kind"), nb::arg("field"),
          nb::arg("params") = std::vector<double>{}, nb::arg("field2") = std::string{},
          "Register a device closure writing a property/body-force field from input field(s). "
          "target: a registered field (a property 'mu'/'rho'/… or a body-force component "
          "'force_x'/'force_y'/'force_z'). kind: 'linear' (params [p0,p1,p2]: "
          "p0+p1*field+p2*field2), "
          "'boussinesq' (params [rho0,g,beta,T0]: rho0*g*beta*(field-T0) buoyancy), 'arrhenius' "
          "(params [mu_ref,B,Tref]: mu_ref*exp(B*(1/field-1/Tref))). Applied at the top of step().")
      .def(
          "set_property_table",
          [](S& s, const std::string& target, const std::string& field,
             const std::vector<double>& x,
             const std::vector<double>& y) { s.setPropertyTable(target, field, x, y); },
          nb::arg("target"), nb::arg("field"), nb::arg("x"), nb::arg("y"),
          "Register a tabulated property: target = piecewise-linear interpolation of (x, y) at the "
          "input field value (x ascending, clamped at the ends).")
      .def(
          "update_properties", [](S& s) { s.updateProperties(); },
          "Apply all registered property/force closures now (also done at the top of step()).")
      .def(
          "enable_cell_force", [](S& s) { s.enableCellForce(); },
          "Allocate + register the per-cell body-force fields force_x/force_y/force_z and route "
          "them "
          "into the momentum RHS, for an external writer (e.g. CFD-DEM drag feedback) to fill "
          "directly via field_view('force_z'). They persist across steps until overwritten.")
      .def(
          "enable_drag", [](S& s) { s.enableDrag(); },
          "Enable implicit (semi-implicit) linear drag for CFD-DEM: allocate the per-cell "
          "'drag_beta' "
          "field (added to the momentum diagonal so a -beta*(u-u_p) source is treated implicitly "
          "-> "
          "unconditionally stable for the stiff beta of a dense bed) plus force_x/y/z (which carry "
          "beta*u_p, the RHS target). Fill 'drag_beta' and 'force_*' via field_view each step.")
      .def(
          "set_property_mode",
          [](S& s, const std::string& mode, bool harmonic) {
            s.setPropertyMode(mode == "variable", harmonic);
          },
          nb::arg("mode") = "variable", nb::arg("harmonic") = false,
          "Enable variable-coefficient momentum (variable viscosity): mode 'variable' binds the "
          "'mu' "
          "field (get/set_field('mu')) into the diffusion operator; 'constant' reverts. harmonic = "
          "harmonic face-viscosity mean (continuous shear stress across a jump) vs arithmetic. A "
          "closure targeting 'mu' enables this automatically. The incremental-rotational pressure "
          "scheme (large-dt / steady-Stokes) stays active — see set_variable_rotational.")
      .def(
          "set_variable_rotational",
          [](S& s, const std::string& mode, double chi) {
            int m = 0;
            if (mode == "min")
              m = 0;
            else if (mode == "full")
              m = 1;
            else if (mode == "off")
              m = 2;
            else
              throw std::runtime_error("set_variable_rotational: mode must be min/full/off");
            s.setVariableRotational(m, chi);
          },
          nb::arg("mode") = "min", nb::arg("chi") = 1.0,
          "Rotational-pressure term under variable viscosity (the constant-mu Timmermans term "
          "-mu*div(u*) is only valid for homogeneous viscosity — Deteix & Yakoubi 2018). 'min' "
          "(default): constant coefficient chi*mu_min — provably stable at any contrast, exact "
          "fallback to the constant-mu scheme for uniform mu. 'full': pointwise chi*mu(i) — better "
          "pressure consistency at MILD contrast only. 'off': plain incremental (no rotational "
          "term). All modes keep the incremental predictor (large-dt / steady-Stokes capability).")
      .def(
          "set_density_mode",
          [](S& s, const std::string& mode) { s.setDensityMode(mode == "variable"); },
          nb::arg("mode") = "variable",
          "Enable variable density: binds the 'rho' field "
          "(get/set_field('rho'), created seeded with set_rho's value if absent) into the momentum "
          "time term, the advection weight, the per-cell body force (face-interpolated), and the "
          "pressure projection (face coefficient openness*rho0/rho_f with the matching 1/rho_f "
          "velocity correction; rho0 = set_rho's value, so a uniform field reduces exactly to the "
          "constant solver). A closure targeting 'rho' (e.g. a linear mixture of a transported "
          "phase fraction) enables this automatically. For gravity, register a closure "
          "force_z = linear(rho, params=[0, -g]).\n\n"
          "COLLOCATED (SolverColocated, rung V8): supported since 2026-09-02, ALL-FLUID only "
          "(set_pressure_geometry; an immersed solid, the ghost projection and "
          "set_rho_face_harmonic throw). The face coefficient and the face correction are the same "
          "as on the staggered grid, applied to the ABC projection's MAC face field; the CELL "
          "correction is the AVERAGE OF THE TWO FACE CORRECTIONS of each axis (never a cell-centred "
          "grad(phi)/rho_c), and every body/interfacial force is likewise a FACE acceleration "
          "dt*(f_f - (P(i)-P(i-s)))/rho_f added after centerToFace, with the cell taking the average "
          "of the two faces' total increment. Rated to density ratio ~100 for cases WITH MOTION "
          "(momentum consistency needs Favre face states and is not in this rung); a high-ratio case "
          "at REST is exact - measured 0.0 spurious velocity and an exact dP/dz = -rho_f g at ratio "
          "1000.")
      .def(
          "ghost_width", [](S& s) { return s.ghostWidth(); },
          "Ghost-layer width g of the velocity block (field_view returns an (n+2g) buffer).")
      .def(
          "has_cutcell_pressure", [](S& s) { return s.hasCutcellPressure(); },
          "True once the cut-cell pressure operator exists (set_solid or set_pressure_geometry was "
          "called). The porous continuity requires it; the coupling driver auto-sets an all-fluid "
          "geometry when absent.")
      .def(
          "set_porous_continuity", [](S& s, bool on) { s.setPorousContinuity(on); },
          nb::arg("on") = true,
          "Enable the volume-averaged (porous) continuity for unresolved CFD-DEM (staggered only): "
          "the projection enforces d(eps)/dt + div(eps u) = 0 instead of div(u)=0, so the fluid "
          "velocity is NOT solenoidal where the void fraction changes (bubbling/expansion). Binds "
          "the 'eps' field (void fraction from the particle deposition, created seeded to 1 if "
          "absent; the coupling writes it each step BEFORE step()). eps=1 everywhere reduces "
          "exactly "
          "to div(u)=0. Pair with max_porous_residual() for the meaningful convergence check.")
      .def(
          "max_open_divergence_projected", [](S& s) { return s.maxOpenDivergenceProjected(); },
          "max|div(open*u)| of the field the projection ACTUALLY produced — the outflow-face "
          "correction included, and WITHOUT mutating the velocity.\n\n"
          "Use this instead of max_open_divergence() on any domain with an OUTFLOW face. "
          "max_open_divergence() re-imposes the zero-gradient outflow velocity before measuring "
          "(its own comment says so), which (a) DESTROYS bcCorrectOutflow's correction — the "
          "mechanism by which mass leaves — as a side effect, so calling it once per step inside a "
          "time loop changes the run, and (b) reports the divergence of a field the solver never "
          "used. Measured on a stratified ratio-1000 outflow box: 5e-3 from the mutating "
          "diagnostic, flat in the iteration count, flat in the density ratio and bit-identical in "
          "a -DPECLET_FLOW_MREAL_DOUBLE build (i.e. not a solver residual), against the projected "
          "field's own residual from this call. Identical to max_open_divergence() when there is "
          "no outflow face, and on the collocated grid (which already measures the face field).")
      .def("max_open_divergence", &S::maxOpenDivergence,
           "Return the max cut-cell velocity-flux divergence max|div(open*u)|. With porous "
           "continuity this is NOT ~0 -- it equals -d(eps)/dt (the bed expanding). Use "
           "max_porous_residual() for the continuity residual.")
      .def(
          "set_pressure_underrelax", [](S& s, double w) { s.setPressureUnderRelax(w); },
          nb::arg("omega"),
          "Pressure under-relaxation factor omega_p in (0,1] for the incremental accumulation "
          "(MFIX "
          "§10.1); 1.0 = off (default). <1 damps the incremental predictor overshoot on stiff "
          "porous+drag.")
      .def(
          "set_porous_deps_dt", [](S& s, bool on) { s.setPorousDepsDt(on); }, nb::arg("on"),
          "Include (default True) or drop the d(eps)/dt source in the porous projection RHS. Drop "
          "it "
          "to enforce div(eps u)=0 when the per-cell eps deposit's time-derivative is too jagged "
          "and "
          "destabilizes the eps-weighted pressure solve.")
      .def(
          "set_porous_conservative", [](S& s, bool on) { s.setPorousConservative(on); },
          nb::arg("on"),
          "eps-conservative porous momentum + projection pair (default True): time term "
          "(eps_f rho/dt) u, eps rho-weighted advective form, projection coefficients "
          "open*(eps rho idt)/(eps rho idt+beta) with matching correction. False = the legacy "
          "plain-u pair (A/B only; it lets the projection drag gas with the moving porosity at "
          "zero inertia cost — a spurious late-time energy source in clustering flows).")
      .def("sync_porous_prev", &S::syncPorousPrev,
           "Reseed eps^n = eps^{n+1} (d(eps)/dt=0 this step) — call once after the first "
           "void-fraction "
           "deposition so step 0 has no spurious source.")
      .def("max_porous_residual", &S::maxPorousResidual,
           "Residual of the volume-averaged continuity max|div(open*eps*u) + d(eps)/dt| -- the "
           "quantity the porous projection drives to zero. 0 unless set_porous_continuity(True).")
      .def(
          "get_resolution", [](S& s) { return std::vector<int>{s.nx(), s.ny(), s.nz()}; },
          "Return the LOCAL grid resolution [nx, ny, nz] (this rank's block under MPI).")
      .def(
          "global_resolution",
          [](S& s) {
            auto g = s.globalResolution();
            return std::vector<int>{g[0], g[1], g[2]};
          },
          "Return the GLOBAL grid resolution [gnx, gny, gnz] (== local single-rank). For the "
          "CFD-DEM co-decomposition weight field.")
      .def(
          "block_origin",
          [](S& s) {
            auto o = s.blockOrigin();
            return std::vector<int>{o[0], o[1], o[2]};
          },
          "This rank's inner-block origin in GLOBAL cells ([0,0,0] single-rank). Shift the "
          "coupling "
          "deposit origin by this so particles in global coordinates land in the local block.")
      .def(
          "get_spacing", [](S&) { return std::vector<double>{1.0, 1.0, 1.0}; },
          "Return the grid spacing [dx, dy, dz] (always unit on this grid).")
#ifdef PECLET_FLOW_MPI
      // Distributed path (built with -DPECLET_FLOW_MPI): construct the Solver with this rank's
      // LOCAL block dims (see the module-level mpi_block()), then init_mpi with the GLOBAL grid
      // dims. step() then does the g=2 velocity-block halo exchange + the distributed cut-cell
      // pressure MG. Bit-exact to single-rank.
      .def(
          "init_mpi",
          [](S& s, int gnx, int gny, int gnz) {
            ensure_mpi_init();
            s.initMpi(gnx, gny, gnz, MPI_COMM_WORLD);
          },
          nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
          "Wire the multi-rank step: pass the GLOBAL grid dims (gnx,gny,gnz). The Solver must have "
          "been "
          "constructed with this rank's LOCAL block dims (from mpi_block). MPI_Init is called if "
          "needed.")
      .def(
          "rebalance_by_weights",
          [](S& s, const std::vector<double>& w) { s.rebalanceByWeights(w); }, nb::arg("weights"),
          "Dynamic load balancing: redistribute the solver's state onto the weighted ORB of "
          "per-cell "
          "weights (global x-fastest, gnx*gny*gnz). Pass fluid work + gamma*particle_count and the "
          "coupled dem migrates onto the SAME partition from the same array. State-preserving "
          "(bit-exact at np=1, reduction floor at np>1).")
      .def(
          "rank",
          [](S&) {
            ensure_mpi_init();
            int r = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
          },
          "This rank's index in MPI_COMM_WORLD.")
      .def(
          "size",
          [](S&) {
            ensure_mpi_init();
            int n = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &n);
            return n;
          },
          "The number of ranks in MPI_COMM_WORLD.")
#else
      .def(
          "rank", [](S&) { return 0; },
          "MPI rank (always 0 in the single-rank Python module; the multi-rank path is the "
          "tests/kokkos_mpi suite).")
      .def(
          "size", [](S&) { return 1; }, "MPI size (1 in the single-rank Python module).")
#endif
      .def(
          "bcast_from_root", [](S&, nb::object v) { return v; }, nb::arg("value"),
          "Broadcast a value from rank 0 (identity in the single-rank module; mirrors the MPI "
          "API).");
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
      "Kokkos is initialized at import and finalized via a Python atexit hook. Release every "
      "Solver "
      "before interpreter exit (it goes out of scope, or `del s; gc.collect()`) so no Kokkos View "
      "outlives finalize.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // Register Kokkos::finalize via Python atexit. This is REQUIRED on CUDA: without it, Kokkos's
  // internal device state is torn down by static destructors AFTER the CUDA runtime unloads,
  // aborting with cudaErrorCudartUnloading at every exit. atexit runs the hook while the driver is
  // still up. (Returned fields are backed by host std::vectors, not device Views, so they never
  // block finalize; a live Solver still holding Views at exit must be released first — hence the
  // docstring note.)
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  }));
  // The active Kokkos backend ("OpenMP", "Cuda", "HIP"), chosen by the build's install prefix.
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

  // Staggered MAC grid (THE flow solver) + the collocated/cell-centered variant. Same Python API.
  bind_solver<peclet::flow::Staggered>(m, "Solver");
  bind_solver<peclet::flow::Colocated>(m, "SolverColocated");

#ifdef PECLET_FLOW_MPI
  // Module-level: this rank's ORB block of the global (gnx,gny,gnz) grid, matching the
  // deterministic BlockDecomposer the Solver's initMpi re-derives internally (and the C++
  // tests/kokkos_mpi template). Returns (origin=[ox,oy,oz], size=[lnx,lny,lnz]); slice the global
  // SDF with these to build the local block, then Solver(*size) + init_mpi(gnx,gny,gnz). MPI_Init
  // is called if needed.
  m.def(
      "mpi_block",
      [](int gnx, int gny, int gnz) {
        ensure_mpi_init();
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        // Derive it through the SAME factory Solver::initMpi uses, so this local block size matches
        // the solver's dec_ under either decomposition mode (see set_decomposition_levels).
        auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), gnx, gny,
                                                          gnz);
        auto blk = dec.block(static_cast<std::size_t>(rank));
        std::vector<int> origin{(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
        std::vector<int> bsize{(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
        return std::make_pair(origin, bsize);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
      "Return this MPI rank's ORB block of the global (gnx,gny,gnz) grid as (origin, size), each a "
      "length-3 list [x,y,z]. Use it to slice the global SDF into this rank's local block for a "
      "distributed Solver (see Solver.init_mpi). MPI_Init is called if needed.");

#ifdef PECLET_FLOW_MPI
  m.def(
      "predict_hierarchy",
      [](int gnx, int gny, int gnz, int np, int levels, bool telescope, int min_extent) {
        std::vector<std::tuple<std::tuple<int, int, int>, int, std::tuple<int, int, int>,
                               std::tuple<int, int, int>, bool>>
            out;
        for (const auto& r : peclet::flow::CutcellMG::predict(gnx, gny, gnz, np, levels, telescope,
                                                              min_extent))
          out.emplace_back(std::make_tuple(r.global.x, r.global.y, r.global.z), r.ranks,
                           std::make_tuple(r.block.x, r.block.y, r.block.z),
                           std::make_tuple(r.ratio.x, r.ratio.y, r.ratio.z), r.tele);
#endif
        return out;
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"), nb::arg("np"), nb::arg("levels"),
      nb::arg("telescope") = false, nb::arg("min_extent") = 4,
      "The pressure-multigrid hierarchy init_mpi WOULD build for this grid / rank count / level "
      "request, under the current decomposition mode (set_decomposition_levels) and with or "
      "without coarse-level telescoping -- a pure function, no MPI, no allocation, any rank "
      "count. Returns one row per level: (global dims, ranks holding the level, block-0 dims, "
      "ratio to the next level, telescopes-out). Pre-flight any job with it.");
  m.def(
      "set_decomposition_levels",
      [](int levels) { peclet::flow::CutcellMG::setDecompositionLevels(levels); },
      nb::arg("levels"),
      "Choose how the shared MPI decomposition is built. 0 (default) = the aligned ORB: split "
      "positions on the FINE grid are snapped to a power of two (capped at 16, i.e. 5 nested "
      "multigrid levels). levels >= 2 = COARSE-FIRST: build the ORB on the grid coarsened "
      "levels-1 times and refine the partition upward, so every block is a multiple of the "
      "coarsening factor BY CONSTRUCTION and the hierarchy nests for the full requested depth. "
      "Coarse-first also balances better: the aligned ORB picks a split and then rounds it (a "
      "balanced 96|96 can round to 128|64), whereas on the coarse grid one cell IS the quantum, so "
      "the rank count no longer has to be a power of two -- what matters is that the COARSE grid "
      "divides among the ranks. Backs off automatically if the coarse grid would have fewer cells "
      "than ranks. CALL BEFORE mpi_block() AND Solver.init_mpi(): both derive the same partition "
      "from this setting and must agree. Env override: PECLET_FLOW_DECOMP_LEVELS.");
  m.def("decomposition_levels", [] { return peclet::flow::CutcellMG::decompositionLevels(); },
        "Current decomposition mode (0 = aligned ORB, >= 2 = coarse-first with that depth).");

  m.attr("has_mpi") = true;
#else
  m.attr("has_mpi") = false;
#endif
}
