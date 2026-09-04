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
          "node, 2 ints + 18 reals per instance -- reals[17] is the centre-pinned flag; legacy 17-real "
          "records are accepted, reading an all-zero centre as 'follows the body'). Geometry is in CELL UNITS on the GLOBAL inner "
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
      .def(
          "moving_instance_cut_cells",
          [](S& s) {
            std::vector<long> v = s.movingInstanceCutCells();
            std::vector<double> d(v.begin(), v.end());
            return peclet::core::python::vector_to_ndarray(std::move(d), {v.size()}, {1});
          },
          "Per instance: inner cells on this rank it owns that touch a fractional face aperture, "
          "recounted whenever any instance moves. ZERO for a moving instance means its surface "
          "sits on grid planes (or is sub-cell) and its wall velocity is silently inert "
          "(set_solid_from_scene warns). Empty when no instance moves.")
      .def(
          "moving_instance_degenerate_points",
          [](S& s) {
            std::vector<long> v = s.movingInstanceDegeneratePoints();
            std::vector<double> d(v.begin(), v.end());
            return peclet::core::python::vector_to_ndarray(std::move(d), {v.size()}, {1});
          },
          "Per instance: staggered velocity points where the sampled SDF is EXACTLY zero -- a face "
          "on a lattice plane. Those points are fluid to the mask and not ghosts to the cut-cell "
          "fold, so a moving body's datum never enters there (set_solid_from_scene warns). Empty "
          "when no instance moves.")
      .def("instance_center", &S::instanceCenter, nb::arg("i"),
           "The resolved centre of rotation of instance i (world coordinates).")
      .def("instance_center_pinned", &S::instanceCenterPinned, nb::arg("i"),
           "True if instance i's centre of rotation is PINNED (an explicit finite centre in the "
           "encoding, or set_instance_motion(center=...)); False if it follows the body's "
           "translation (NaN in the encoding, the builder's default; legacy all-zero raw arrays).")
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
          "It THROWS if the current velocity is not discretely divergence-free to 1e-10, measured "
          "with max_open_divergence_projected() (the NON-mutating diagnostic). That is not a "
          "nicety: Weymouth-Yue's exact conservation is CONDITIONAL on sum_f o_f u_f = 0 per cell, "
          "because the dilation term adds H(C-1/2) times that residual to every full cell's volume "
          "budget. Run step() to a steady state (or project()) and advect with the solver's own "
          "output; never with an analytic sample, which is solenoidal only to O(h^2).\n\n"
          "It ALSO throws when no cut-cell pressure operator exists (WO-R2): without one the "
          "divergence guard measures nothing at all — max_open_divergence() returns 0.0 — and a "
          "cell-centre-sampled LeVeque field (true max|div| 0.612) was silently accepted and lost "
          "4.93 % of the liquid in 50 steps. Call set_pressure_geometry(sdf) on an all-fluid box, "
          "or set_solid(sdf, cutcell_pressure=True).")
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
          "set_vof_wisp_eps", [](S& s, double eps) { s.setVofWispEps(eps); }, nb::arg("eps"),
          "Wisp tolerance on the Weymouth-Yue mixed-cell predicate: a cell counts as carrying an "
          "interface only while eps < C < 1 - eps, and one outside that band is fluxed "
          "ALGEBRAICALLY as C_donor * a — its ACTUAL colour, so the exact telescoping conservation "
          "is untouched. The same threshold gates the interface-local Courant band "
          "(|C_i - C_j| > eps instead of an exact !=).\n\n"
          "DEFAULT 1e-8 (the same value the V3 curvature predicate uses under surface tension); "
          "0 restores the V1 predicate bit for bit. Two measured reasons it is not 0 (WO-R2 item "
          "4): (i) a domain that DRAINS through an open boundary leaves nothing but round-off "
          "residue, ~1e-18, which `0 < C < 1` still calls mixed — the MYC normal of that stencil "
          "is degenerate and plicAlpha divides by it (sum C -> -inf -> NaN within three steps on "
          "one backend); (ii) the round-off wake behind a passing interface (min C ~ -3.8e-17) "
          "kept the whole wake inside the Courant band, so vof_last_courant() on Zalesak read "
          "0.3110 by step 1000 on a case whose interface never exceeds 0.255.")
      .def(
          "vof_wisp_eps", [](S& s) { return s.vofWispEps(); },
          "The wisp tolerance currently in force (see set_vof_wisp_eps).")
      .def(
          "set_pressure_exact_residual",
          [](S& s, bool on) { s.setPressureExactResidual(on); }, nb::arg("on") = true,
          "Apply the level-0 pressure operator EXACTLY (matrix-free, double, flux form) in the "
          "residual and the Krylov matvec instead of reading the float band storage. P1 of the "
          "suite defect-correction campaign (docs/DEFECT_CORRECTION_PLAN.md); PROCESS-WIDE, and "
          "PECLET_FLOW_EXACT_RESIDUAL initialises it.\n\n"
          "enable_vof() turns it ON, because a two-phase coefficient contrast is exactly what "
          "amplifies the float operator's broken row-sum identity A*1 = 0. Measured on Hysing "
          "case 2 (64x128x4, adaptive dt, nvidia-cuda): max|div(open u)| 1.85e-03 -> 5.15e-11, "
          "with 116/600 pressure iterations, 1123 steps, the dt-limit census and both published "
          "functionals (v_rise max 0.2574 at t = 0.671, y_c(3) 1.1082) identical to every printed "
          "digit. Everything below level 0 stays float on purpose: it is a preconditioner and its "
          "errors change the convergence RATE, never the fixed point. Call it with False AFTER "
          "enable_vof for the ablation.")
      .def(
          "pressure_exact_residual", [](S& s) { return s.pressureExactResidual(); },
          "Whether the exact level-0 operator apply is in force (see set_pressure_exact_residual).")
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
            r["dynamic_cells"] = d.dynamicCells;
            r["pinned_cells"] = d.pinnedCells;
            r["advancing_cells"] = d.advancingCells;
            r["receding_cells"] = d.recedingCells;
            r["mean_imposed_theta"] = d.meanImposedTheta;
            r["mean_apparent_theta"] = d.meanApparentTheta;
            r["max_Ca_cl"] = d.maxCaCl;
            r["max_contact_speed"] = d.maxContactSpeed;
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
          "measured on the fill's own data rather than on a post-processed shape.\n\n"
          "Rung V6 (WO-V6) adds, all zero unless set_contact_angle_dynamic / "
          "set_contact_angle_hysteresis is configured: 'dynamic_cells' (band cells the V6 pass "
          "produced an angle for), 'pinned_cells' / 'advancing_cells' / 'receding_cells' (the "
          "hysteresis branch census), 'mean_imposed_theta' and 'mean_apparent_theta' in degrees, "
          "'max_Ca_cl' = max |mu_l U_cl / sigma| and 'max_contact_speed' = max |U_cl| in solver "
          "velocity units.")
      .def(
          "set_contact_angle_dynamic",
          [](S& s, double th, double slip, double mu, double sigma) {
            s.setContactAngleDynamic(th, slip, mu, sigma);
          },
          nb::arg("theta_e"), nb::arg("slip_length_cells"), nb::arg("mu_liquid"),
          nb::arg("sigma") = 0.0,
          "Rung V6 (WO-V6). The DYNAMIC contact angle: the angle imposed at the grid scale is the "
          "Cox-Voinov apparent angle of a contact line moving at speed U_cl, with the outer "
          "cut-off at the CELL SIZE and an EXPLICIT slip length lambda (Afkhami, Zaleski & "
          "Bussmann, JCP 228:5370, 2009):\n\n"
          "    theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta/lambda),   Ca_cl = mu_l U_cl / sigma\n\n"
          "angles in radians internally, arguments in DEGREES; Ca_cl > 0 advancing, < 0 receding; "
          "theta_Delta is clamped into [1, 179] degrees (set_contact_angle_clamp).\n\n"
          "`slip_length_cells` is lambda/Delta and must lie in (0, 1). It is not a tuning knob "
          "you may omit from a report: a VoF contact line's NUMERICAL slip is proportional to the "
          "cell size, so without an explicit lambda the imposed angle is silently grid-dependent "
          "(VOF_PLAN section 6). ALWAYS state lambda alongside a dynamic-wetting result.\n\n"
          "`mu_liquid` is the liquid dynamic viscosity entering Ca_cl (solver units); `sigma` "
          "defaults to whatever set_surface_tension holds. theta_e also becomes the static base "
          "angle, so this call replaces set_contact_angle. U_cl is measured as -u.t_hat at the "
          "anchor fluid cell of each band cell, with t_hat the IN-WALL direction of the fluid-side "
          "PLIC normal (m points into the gas, so the liquid advances along -t_hat), then smoothed "
          "with a 3-point mean along t_hat. Needs set_solid(..., cutcell_pressure=True) + "
          "enable_vof; with no call every V5b number is byte-identical.")
      .def(
          "set_contact_angle_hysteresis",
          [](S& s, double a, double r) { s.setContactAngleHysteresis(a, r); },
          nb::arg("theta_a"), nb::arg("theta_r"),
          "Rung V6 (WO-V6). Advancing / receding contact-angle hysteresis, in DEGREES. While the "
          "measured apparent angle lies in [theta_r, theta_a] the contact line is PINNED and the "
          "fill imposes the APPARENT angle itself — which reproduces the current interface exactly, "
          "because the V5b fill is idempotent (WO-S finding 1), so no Young force appears and the "
          "line does not move. Above theta_a the advancing angle is imposed and below theta_r the "
          "receding one, each with the Cox-Voinov correction of set_contact_angle_dynamic when "
          "that is also configured. Sets the static base to (theta_a+theta_r)/2 if no angle was "
          "set yet.")
      .def(
          "set_wall_slip_length", [](S& s, double lam) { s.setWallSlipLength(lam); },
          nb::arg("lambda_cells"),
          "Rung V6b (WO-V6b) - the VELOCITY half of the dynamic contact line. In the Robust-Scaled "
          "cut-cell IBM closure the TANGENTIAL wall datum stops being no-slip and becomes the "
          "NAVIER condition\n\n"
          "    u_t(wall) - u_body = lambda * d(u_t)/dn\n\n"
          "with `lambda_cells` = lambda/Delta (cell size 1, so this is the slip length in CELLS). "
          "The wall-NORMAL component is untouched: the wall stays impermeable, with the moving-body "
          "velocity as its datum. 0 (the default) restores the validated no-slip closure "
          "BIT-IDENTICALLY - the closure polynomials are the lambda = 0 members of the same "
          "family, and the lambda > 0 branch is never entered.\n\n"
          "It is ONE lambda with set_contact_angle_dynamic: either call overwrites the shared "
          "value (last call wins), so the Cox-Voinov inner cut-off and the momentum wall closure "
          "cannot disagree. Only this call switches the MOMENTUM half on, so every WO-V6 result "
          "taken with the angle half alone is unchanged.\n\n"
          "Why it exists, measured: with no-slip the contact line advances only at the scheme's "
          "own numerical slip, ~1/180 of Lucas-Washburn (WO-V6 finding 6), and WO-V7's pore-scale "
          "campaign found every IMBIBITION verdict inverted below the capillary number at which "
          "the imposed velocity crosses that numerical slip velocity. Typical values are 0.01-0.5 "
          "cells; state lambda with every dynamic-wetting result. Staggered grids only; needs "
          "set_solid(..., cutcell_pressure=True). Changing it rebuilds the three cut-cell overlays "
          "and the momentum operator in place (the velocity field is NOT reset).\n\n"
          "APPROXIMATIONS, both stated because they are invisible from the call: the tangential "
          "projector is taken DIAGONAL (the cross terms -n_c n_j u_j are dropped; they vanish "
          "exactly for an axis-aligned wall and are O(n_c n_j) otherwise), and an axis whose BOTH "
          "neighbours are solid (a one-cell fluid gap) keeps the no-slip closure - counted by "
          "wall_slip_sandwich_cells().")
      .def(
          "wall_slip_length", [](S& s) { return s.wallSlipLength(); },
          "The Navier slip length in force, in cells (0 = no-slip).")
      .def(
          "wall_slip_sandwich_cells",
          [](S& s) {
            auto a = s.wallSlipSandwichCells();
            return nb::make_tuple(a[0], a[1], a[2]);
          },
          "Per velocity component, the number of cut-cell AXES at which a one-cell fluid gap made "
          "the Navier closure inapplicable and the no-slip one was kept. Nonzero means part of the "
          "wall is silently no-slip; report it.")
      .def(
          "set_contact_angle_dynamic_off", [](S& s) { s.setContactAngleDynamicOff(); },
          "Turn the V6 dynamic angle and hysteresis off; the static V5b angle stands again.")
      .def(
          "set_contact_angle_smoothing", [](S& s, bool on) { s.setContactAngleSmoothing(on); },
          nb::arg("on"),
          "ABLATION - the 3-point in-wall mean of U_cl (default ON). A MAC velocity next to a wall "
          "is noisy cell to cell and the cube root of the Cox-Voinov relation puts that noise "
          "straight into the imposed angle; off measures what the smoothing is worth.")
      .def(
          "set_contact_angle_clamp",
          [](S& s, double lo, double hi) { s.setContactAngleClamp(lo, hi); }, nb::arg("lo"),
          nb::arg("hi"),
          "The clamp on the Cox-Voinov cube, in degrees (default 1 / 179). The cubic has no "
          "solution beyond the maximum receding capillary number (film entrainment) and the fill's "
          "plane construction degenerates at 0/180.")
      .def(
          "vof_dynamic_field", [](S& s, int w) { return field_out(s, s.getVofDynamicField(w)); },
          nb::arg("which"),
          "The V6 per-cell dynamic-wetting state on the inner region, as (nx,ny,nz): 0 the IMPOSED "
          "angle in degrees, 1 the measured APPARENT angle in degrees, 2 the smoothed U_cl "
          "(positive = the liquid ADVANCES), 3 Ca_cl, 4 the state (0 not a contact cell, 1 "
          "Cox-Voinov on the static base, 2 PINNED, 3 advancing, 4 receding). Regenerates the fill, "
          "so it is also the direct gate on the pass's decomposition independence.")
      .def(
          "vof_has_geometry", [](S& s) { return s.vofHasGeometry(); },
          "True when the colour advection is running the CUT-CELL (openness-weighted) kernels, "
          "i.e. an immersed solid is present and set_solid ran with cutcell_pressure=True. False "
          "means "
          "the uncut rung-V1 kernels are running, byte-identically to a solid-free build.")
      .def(
          "set_vof_timing", [](S& s, bool on) { s.setVofTiming(on); }, nb::arg("on"),
          "WO-V9: arm (or disarm) the VoF pipeline's per-stage timers, and reset them. OFF by "
          "default. When armed, every stage boundary calls Kokkos::fence() before reading the "
          "clock -- the same rule the step's three coarse phase timers already follow, because on "
          "a device backend queued work would otherwise be billed to whichever stage next reads "
          "the clock. A fence changes WHEN work happens, never WHAT it computes: a run with the "
          "timers armed is bit-identical to the same run without them (gated in "
          "tests/kokkos/test_vof_timing.cpp). When disarmed the cost is one branch per stage and "
          "no fence at all.")
      .def(
          "reset_vof_timing", [](S& s) { s.resetVofTiming(); },
          "Zero the VoF stage timers and the step counter without disarming them.")
      .def(
          "vof_timing",
          [](S& s) {
            const auto& v = s.vofTimingReport();
            const auto& k = s.vofKernelTiming();
            nb::dict r;
            r["steps"] = v.steps;
            // the step's three coarse phases, summed over the same window (seconds)
            r["step"] = s.vofTimingStepSeconds();
            r["predictor"] = s.vofTimingPredictorSeconds();
            r["momentum_solve"] = s.vofTimingMomentumSeconds();
            r["projection"] = s.vofTimingProjectionSeconds();
            // the VoF stages
            r["vof_advect"] = v.advect;
            r["vof_bridge"] = v.bridge;
            r["vof_momentum_advect"] = v.momAdvect;
            r["vof_momentum_bridge"] = v.momBridge;
            r["curvature"] = v.curvature;
            r["csf"] = v.csf;
            r["phase_change"] = v.phaseChange;
            // the advector's own kernels (shared by the colour, momentum and energy drivers)
            r["k_freeze"] = k.freeze;
            r["k_reconstruct"] = k.reconstruct;
            r["k_fluxes"] = k.fluxes;
            r["k_sweep"] = k.sweep;
            r["k_clip"] = k.clip;
            r["k_exchange"] = k.exchange;
            r["k_sweeps"] = k.sweeps;
            // the V3 curvature cascade's own passes
            const auto& q = s.vofCurvatureTiming();
            r["kc_calls"] = q.calls;
            r["kc_compact"] = q.compact;
            r["kc_planes"] = q.planes;
            r["kc_height"] = q.height;
            r["kc_fallback"] = q.fallback;
            r["kc_census"] = q.census;
            return r;
          },
          "WO-V9: cumulative VoF stage times in SECONDS since the last set_vof_timing/"
          "reset_vof_timing, on this rank. `steps` is the number of step() calls in the window, so "
          "every entry divides down to a per-step cost. `step`/`predictor`/`momentum_solve`/"
          "`projection` are the step's three coarse phases summed over the SAME window (the "
          "remainder of `step` is BC re-imposition, Picard bookkeeping and the scalars). "
          "`vof_advect` is the whole colour stage and `vof_bridge` the part of it spent in the "
          "G=2 <-> g=3 bridges and the C ghost policy; `vof_momentum_advect` is its V2b twin. "
          "`k_*` are the advector's own kernels, shared by the colour, momentum-consistent and "
          "consistent-energy drivers, so they are the per-KERNEL breakdown of whichever of the two "
          "stages is running: `k_reconstruct` the MYC + plicAlpha pass, `k_fluxes` the geometric "
          "face fluxes, `k_sweep` the update, `k_clip` the V5a cut-cell clip, `k_exchange` the "
          "g = 3 ghost exchange, `k_sweeps` the number of sweeps (3 per advect call). Returns "
          "zeros unless the timers are armed.")
      .def(
          "set_vof_worklist", [](S& s, bool on) { s.setVofWorklist(on); }, nb::arg("on"),
          "WyAdvector::useWorklist -- compact the PLIC reconstruction pass onto the mixed cells "
          "(a parallel_scan over the reconstruction region, then a dense parallel_for over the "
          "compacted list) instead of a guarded parallel_for over the whole region. Default ON. "
          "Pure optimization: the flux never reads a non-mixed cell's plane, so the two paths "
          "produce the same field BIT FOR BIT (gated in tests/kokkos/test_vof_advect.cpp and "
          "measured in WO-V9).")
      .def(
          "vof_worklist", [](S& s) { return s.vofWorklist(); },
          "Whether the reconstruction-pass compaction is on.")
      .def(
          "set_vof_curvature_worklist", [](S& s, bool on) { s.setVofCurvatureWorklist(on); },
          nb::arg("on"),
          "WO-V9: run the V3 curvature cascade (the PLIC plane pass, the height functions and the "
          "PV fallback) over a COMPACTED list of the interfacial cells instead of over the whole "
          "inner region. Default ON. The cascade is the most divergent kernel in the pipeline -- "
          "on a resolved droplet the interfacial cells are well under 1 % of the block, so a dense "
          "parallel_for puts one or two active lanes in a warp and the other thirty run the guard "
          "and then idle for the whole height function. Compaction is a pure re-ordering (each "
          "cell reads the same neighbours and writes the same value), so the two paths are "
          "BIT-IDENTICAL; tests/kokkos/test_vof_timing.cpp gates that and WO-V9 records the gain.")
      .def(
          "vof_curvature_worklist", [](S& s) { return s.vofCurvatureWorklist(); },
          "Whether the curvature-cascade compaction is on.")
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
      // --- Part III rung W0 (WO-W0): the per-bubble block container ----------------------------
      .def(
          "enable_vof_blocks",
          [](S& s, const std::vector<std::array<double, 4>>& seeds) { s.enableVofBlocks(seeds); },
          nb::arg("seeds"),
          "Carry each bubble on its OWN VoF block (the TBFsolver vofBlock pattern, VOF_PLAN §10): "
          "one Weymouth-Yue advector per marker on a small moving global index box (bubble extent "
          "+ 3 cells, its own g = 3 halo on top) with a master rank of its own, and the registered "
          "'C' the closures see is the UNION C = max_blocks C_block.\n\n"
          "WHY: two bubbles that touch CANNOT coalesce numerically — colliding markers overlap in "
          "space but never merge, so coalescence becomes an explicit model decision (rung W4) "
          "instead of a numerical accident of a single global colour field. Measured on the gate: "
          "236 cells carry BOTH markers at closest approach (30.0 cells of shared liquid) while "
          "each marker's own volume is conserved to 2.6e-15, and the single-field control merges "
          "them irreversibly (neck colour 0.77 against the blocks' 0.00 after the reversal).\n\n"
          "'seeds' is a list of (cx, cy, cz, radius) spheres in CELL units, global indices. The "
          "colour is the same exact sphere fraction set_vof would take.\n\n"
          "SCOPE at W0: ALL-FLUID (an immersed solid raises — the cut-cell block is rung W12) and "
          "KINEMATIC (advect_vof_blocks(dt); NS coupling is W12). Masters are assigned round robin "
          "by block id, deliberately independent of where the bubble's cells live — the weighted-"
          "ORB assignment is rung W1; vof_block_imbalance() is the number to beat.")
      .def(
          "disable_vof_blocks", [](S& s) { s.disableVofBlocks(); },
          "Drop the block container; the structured colour field ('C', advect_vof) is unaffected.")
      .def(
          "vof_blocks_enabled", [](S& s) { return s.vofBlocksEnabled(); },
          "True after enable_vof_blocks.")
      .def(
          "advect_vof_blocks", [](S& s, double dt) { s.advectVofBlocks(dt); }, nb::arg("dt"),
          "Advance every block by dt with the CURRENT (projected) face velocity and union the "
          "result into 'C' — the block twin of advect_vof(dt), and it carries the same "
          "precondition: it RAISES unless the face field is discretely divergence-free to 1e-10, "
          "because Weymouth-Yue's exact conservation is conditional on it.\n\n"
          "Per step: gather the face velocity from the ranks that own each block's cells to its "
          "master (plain Isend/Irecv — the block table and the decomposition are both replicated, "
          "so every rank knows every message size and an NBX handshake would only rediscover it), "
          "run the three sweeps on the dense block, re-centre the box, replicate the table, and "
          "scatter the inner colour back with UNPACK_MAX.\n\n"
          "NOTE on the union: max() from an empty union CLIPS the negative Weymouth-Yue round-off "
          "residue to an exact 0 — measured up to 6.2e-17 on the LeVeque gate, and every measured "
          "union/global-field difference was exactly that and nothing else.")
      .def(
          "vof_block_stats",
          [](S& s) {
            nb::list out;
            for (const auto& b : s.vofBlockStats()) {
              nb::dict r;
              r["id"] = b.id;
              r["master"] = b.master;
              r["lo"] = nb::make_tuple(b.lo[0], b.lo[1], b.lo[2]);
              r["hi"] = nb::make_tuple(b.hi[0], b.hi[1], b.hi[2]);
              r["cells"] = b.cells;
              r["volume"] = b.volume;
              r["centroid"] = nb::make_tuple(b.centroid[0], b.centroid[1], b.centroid[2]);
              r["velocity"] = nb::make_tuple(b.velocity[0], b.velocity[1], b.velocity[2]);
              r["moments"] = nb::make_tuple(b.moment[0], b.moment[1], b.moment[2], b.moment[3],
                                            b.moment[4], b.moment[5]);
              r["recentred"] = b.recentred;
              r["discarded"] = b.discarded;
              r["area"] = b.area;
              out.append(r);
            }
            return out;
          },
          "Per-bubble Lagrangian census, one dict per block, in block-id order. 'lo'/'hi' (the "
          "global index box) and 'master' are replicated on every rank; the MEASURED entries — "
          "'volume' (sum of C over the box, cell volumes), 'centroid', 'velocity' (d(centroid)/dt "
          "of the last advection), 'moments' (the central second moments xx, yy, zz, xy, xz, yz "
          "divided by the volume, i.e. the deformation) — are filled only on the block's MASTER "
          "and are zero elsewhere.\n\n"
          "'discarded' is the colour a re-centring dropped, cumulatively: Weymouth-Yue leaves "
          "round-off residue in every cell its sweeps touch and the box tracks the BUBBLE, not "
          "that wake, so the residue falling outside the new box is discarded. Never physical "
          "liquid (measured -9.5e-17 over a 20-cell translation, against a bubble volume of 524), "
          "but it is reported rather than hidden — a container that silently loses mass is not "
          "acceptable.")
      .def(
          "vof_block_imbalance", [](S& s) { return s.vofBlockImbalance(); },
          "max/mean of the per-rank block-cell load under the CURRENT master assignment "
          "(round robin by block id at rung W0). 1.0 is perfect; the weighted-ORB assignment of "
          "rung W1 is what this number is here to grade.")
      .def(
          "vof_block_census",
          [](S& s) {
            std::vector<long> m, c;
            s.vofBlockCensus(m, c);
            nb::dict r;
            nb::list lm, lc;
            for (long v : m)
              lm.append(v);
            for (long v : c)
              lc.append(v);
            r["masters"] = lm;
            r["cells"] = lc;
            return r;
          },
          "Per-rank load census of the block container: 'masters'[r] = blocks rank r masters, "
          "'cells'[r] = the inner cells those blocks carry (the actual VoF work).")
      // --- Part III rungs W1/W2 (WO-W12) --------------------------------------------------------
      .def(
          "enable_vof_blocks_from_field",
          [](S& s, const std::vector<std::array<int, 6>>& boxes) {
            s.enableVofBlocksFromField(boxes);
          },
          nb::arg("boxes"),
          "Seed the block container from the CURRENT colour field: one marker per given global "
          "index box (lo_x, lo_y, lo_z, hi_x, hi_y, hi_z; half-open), its colour copied out of the "
          "field set_vof installed. The general seeding path — enable_vof_blocks(spheres) is a "
          "convenience over it — and the way a marker of any shape enters (a quasi-2-D Hysing "
          "cylinder, a scanned bubble, an arbitrary blob found with a host connected-component "
          "labelling).\n\n"
          "The boxes are the BUBBLE extents; the container grows each by its own 3-cell margin. "
          "They must not share a cell: the seeding gather is a copy, not a union, so a cell inside "
          "two boxes would be handed to both markers.")
      .def(
          "set_vof_block_assign",
          [](S& s, int mode, long every) { s.setVofBlockAssign(mode, every); }, nb::arg("mode"),
          nb::arg("every") = 0,
          "Master assignment of the block container (rung W1). mode 0 = round robin by block id "
          "(rung W0's, blind to block SIZE), 1 = LONGEST-PROCESSING-TIME greedy on the block cell "
          "counts, 2 = weighted ORB over a 1-D block space (core's BlockDecomposer<1> with the "
          "cell counts as weights). 'every' > 0 re-runs the assignment every 'every' block steps "
          "and MIGRATES the colour of any block that changed master (nothing else in a block is "
          "state, so the migration is exact and the bitwise gates hold across it).\n\n"
          "All three are pure functions of the REPLICATED block table, so every rank computes the "
          "same assignment without an exchange — which is what lets a re-assignment happen "
          "mid-run without breaking a bitwise gate. LPT is the measured winner (see the WO-W12 "
          "findings): the ORB's blocks must be CONTIGUOUS in block id, which LPT is free of.")
      .def(
          "vof_block_assign", [](S& s) { return s.vofBlockAssign(); },
          "The current master-assignment mode (see set_vof_block_assign).")
      .def(
          "vof_block_imbalance_of", [](S& s, int mode) { return s.vofBlockImbalanceOf(mode); },
          nb::arg("mode"),
          "The max/mean per-rank block-cell load the given assignment mode WOULD give on the "
          "current blocks, without applying it — so a study can put the three modes side by side "
          "on one swarm without perturbing the run.")
      .def(
          "set_vof_block_device_staging",
          [](S& s, bool on) { s.setVofBlockDeviceStaging(on); }, nb::arg("on"),
          "Pack/unpack the block gather and scatter in the block's own MEMORY SPACE (rung W1 item "
          "b, default True), with a host staging copy only per MPI MESSAGE — and none at all for "
          "the master's own cells, which are a device-to-device copy. False selects rung W0's "
          "host-staged path (a full mirror of the local patch per step), which is what the "
          "device-vs-host measurement compares against. Every step is a copy of a double, so the "
          "two paths are BITWISE identical; the ctest gates that rather than asserting it.")
      .def(
          "set_vof_block_pool", [](S& s, bool on) { s.setVofBlockPool(on); }, nb::arg("on"),
          "Recycle the advectors a re-centring retires, keyed by the exact box extent (rung W1 "
          "item c, default True). A translating bubble keeps its box SIZE and only moves its "
          "origin, so the hit rate is ~100 % and the ten Views of the new box are not allocated. "
          "A recycled advector is handed back in the state a freshly initialised one is in "
          "(colour and the three face-velocity fields zeroed), so the pool is bitwise inert.")
      .def(
          "vof_block_pool_stats",
          [](S& s) {
            const auto q = s.vofBlockPoolStats();
            nb::dict r;
            r["hits"] = q[0];
            r["misses"] = q[1];
            return r;
          },
          "Block-pool census: 'hits' = advectors recycled, 'misses' = advectors allocated.")
      .def(
          "enable_vof_block_csf", [](S& s) { s.enableVofBlockCsf(); },
          "Form the surface-tension force PER BLOCK (rung W2): each marker runs its own curvature "
          "cascade on its own dense box and forms the V4 balanced-force face force there "
          "(sigma kappa_f (C(i) - C(i-s))/h with the same selective face-curvature rule "
          "addCsfRhs uses), and the three face fields are scattered into the global RHS with "
          "UNPACK_SUM — TBFsolver's VOF.f90::computeSurfaceTension structure on the suite's "
          "kernels.\n\n"
          "WHY THE FORCE AND NOT THE CURVATURE IS SCATTERED: kappa is not additive and the union "
          "colour is a max, so a face between two OVERLAPPING markers has no single (kappa, dC) "
          "pair to build a force from. The force is the additive quantity, and forming it where "
          "each marker's own colour still exists is the only place the balanced-force pairing (the "
          "same face difference the projection's gradient uses) is available per marker.\n\n"
          "Requires set_surface_tension(sigma) and the block container; STAGGERED only. Once on, "
          "step() drives the whole two-phase stage through the blocks: the union C feeds the "
          "closures exactly as before, and the colour is advected by the blocks in the "
          "advect_vof slot.")
      .def(
          "vof_block_curvature_stats",
          [](S& s) {
            const auto st = s.vofBlockCurvatureStats();
            nb::dict r;
            r["interfacial"] = st.interfacial;
            r["hf"] = st.hf;
            r["hf_mixed"] = st.hfMixed;
            r["hf_fit"] = st.hfFit;
            r["pv"] = st.pv;
            r["pv_reduced"] = st.pvReduced;
            r["no_estimate"] = st.noEstimate;
            return r;
          },
          "Branch census of the last block-CSF curvature pass, SUMMED over this rank's blocks "
          "(local to the rank). 'no_estimate' must be 0 on any gated case.")
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
          "The 1/rho_f mobility factor on the HIGH-side outflow face correction. DEFAULT TRUE "
          "since WO-R2; pass False (or PECLET_FLOW_OUTFLOW_RHO=0) for the ablation.\n\n"
          "A projection correction cancels the discrete divergence only if it uses the SAME face "
          "coefficient the operator row used. Until WO-R2 the multigrid re-imposed the literal "
          "openness 1.0 at every Dirichlet domain face, overwriting buildRhoCoeff's "
          "open_f*rho0/rho_f, so the plain phi difference was the consistent partner and WO-R "
          "measured this factor making things seven orders WORSE. WO-R2 fixed the operator "
          "(CutcellMG::setOutflowCoefficient) and the verdict inverted. Stratified duct, ratio "
          "10, 5 steps, max|div(open u)| of the PROJECTED field:\n"
          "                              old operator   fixed operator\n"
          "  without the factor            8.76e-10        9.97e-05\n"
          "  with    the factor            9.24e-03        8.31e-10\n"
          "(tests/kokkos/test_vof_bc.cpp gate F2.) Bitwise inert at constant density (rho_f == "
          "rho0 makes the factor exactly 1) and gated on the variable-density path.")
      .def(
          "outflow_rho_correction", [](S& s) { return s.outflowRhoCorrection(); },
          "Whether the 1/rho_f factor is applied to the outflow face correction (WO-R item 4, "
          "reversed by WO-R2's operator fix; default True).")
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
      // --- Phase change, Part II rungs P2/P3 (WO-P23) ------------------------------------------
      .def(
          "set_phase_change_plane_dirichlet",
          [](S& s, bool on) { s.setPhaseChangePlaneDirichlet(on); }, nb::arg("on"),
          "The PLANE-ANCHORED (ghost-fluid) interfacial Dirichlet condition. ON by default.\n\n"
          "Rungs P0/P1 pinned the whole interfacial CELL at T_G, so the numerical thermal boundary "
          "sat at the cell CENTRE while the mass-flux gradient is fitted from the PLIC PLANE — a "
          "mismatch of up to half a cell that CHANGES SIGN as the interface sweeps through a cell, "
          "and the first-order component of the P1 Stefan error (WO-P01 finding 6). With this on, "
          "the condition becomes PER FACE: for a pure cell i whose neighbour j is interfacial, that "
          "face's row carries k*open*(T_i - T_G)/theta with theta the distance in cells from i's "
          "centre to j's PLIC plane along the face's own axis, and the interfacial cell's own value "
          "is never read by any neighbour.\n\n"
          "A per-CELL value (the literal reading of the work order) does NOT work and the "
          "measurement is in the findings: giving the interfacial cell the value the one-sided "
          "profile takes at its centre is right for the side the fit came from and wrong for the "
          "other side — on the P1 Stefan ladder it heats the saturated liquid through the "
          "interfacial cell's liquid-side face, which gives the liquid a spurious gradient that "
          "feeds straight back into mdot: +6.20/+5.62/+5.42 % at N = 64/128/256, order 0.10, "
          "against +1.31/+0.59/+0.20 % and order 1.37 for the cell-centre pinning it was meant to "
          "improve on. The per-face form has no such asymmetry (the liquid-side face reads T_G at "
          "its own theta, i.e. exactly zero flux for a saturated liquid).\n\n"
          "False restores the rung P0/P1 behaviour bit-for-bit.")
      .def(
          "set_phase_change_quadratic_fit",
          [](S& s, bool on) { s.setPhaseChangeQuadraticFit(on); }, nb::arg("on"),
          "Fit the one-sided interfacial temperature gradients QUADRATICALLY through the interface "
          "value (T - T_G = G phi + Q phi^2) instead of linearly. This is VOF_PLAN section 9 item "
          "1's Aslam quadratic extrapolation in least-squares form — the same 5^3 pure-cell "
          "samples, the same Malan collinearity weights, one more basis function, no PDE sweeps. "
          "Once the plane-anchored Dirichlet has removed the cell-centre mismatch, the linear fit's "
          "O(T'' h) curvature bias is the leading error of the rung: its samples start about one "
          "cell from the plane and reach two and a half, so a curved profile tilts the straight "
          "line through the origin. ON by default (WO-P23): with the plane-anchored Dirichlet it takes "
          "the P1 Stefan interface position from +0.195 % to +0.003 % at N = 256, and the mdot "
          "kernel itself from order 1.1 to order 2.0. set(False) is the ablation.")
      .def(
          "set_phase_change_area", [](S& s, int mode) { s.setPhaseChangeArea(mode); },
          nb::arg("mode"),
          "WO-P3c: WHICH GEOMETRY the interfacial area A_Gamma comes from. A_Gamma sets the plane "
          "shift dV = mdot A dt / rho_l and the divergence source S = mdot A (1/rho_g - 1/rho_l), "
          "so a bubble grows as int mdot dA and a biased area is a biased growth rate.\n"
          "  0 = PLIC (DEFAULT, rungs P0/P1): plicArea = |m|_2 dV/dalpha on the MYC normal.\n"
          "  1 = cascade metric: the V3 curvature cascade's own geometry — the height function's "
          "area element sqrt(1 + h_x^2 + h_y^2) from the SAME central differences the curvature "
          "differentiates once more (tiers 1/2), the PV paraboloid's gradient (tier 3) — applied "
          "to the PLIC polygon's projected footprint, so the cells of a column still tile.\n"
          "  2 = cascade normal: the same normals, but the plane is rebuilt on them, "
          "plicArea(n*, plicAlpha(n*, C)).\n"
          "  3 = cascade footprint: the height function's OWN footprint times its own metric, the "
          "only per-cell variant whose pieces tile.\n"
          "  4..7 = WO-P3d, the JOINED sheet: marching tetrahedra on the cell-centre lattice, one "
          "watertight surface whose triangles are booked to cells — 4 the C = 1/2 level set with "
          "whole triangles to the cell holding the centroid, 5 the same sheet clipped to each "
          "cell's cube, 6 and 7 the same two deposits on the zero of the PLIC-reconstructed signed "
          "distance (exact on a TILTED plane, where interpolating C is not, because C(d) is the SZ "
          "piecewise cubic). Modes 4-7 are the only ones whose SUM converges on a curved "
          "interface: WO-P3c proved with two analytic controls that every PER-CELL area is first "
          "order in h/R because the pieces do not JOIN across cells.\n"
          "All of 0-3 are EXACT on a plane, so every planar gate (P0a/P0b/P1/P2) is unmoved. The "
          "default is 0 because the measurement says so: on a sphere whose colour field is "
          "resolved (16^3 sub-sampling), summed plicArea is within 0.5 % of 4 pi R^2 and the "
          "cascade does not improve it. WO-P3b's 5.5-9.3 % 'PLIC area deficit' was its probe's own "
          "4^3 sub-sampling, which quantizes C to 1/64 and drops a QUARTER of the interfacial "
          "cells (their volume is 1e-4 %, their area 6 %).")
      .def("phase_change_area", [](S& s) { return s.phaseChangeArea(); },
           "The set_phase_change_area mode in force (0 PLIC, 1 cascade metric, 2 cascade normal, "
           "3 cascade footprint, 4-7 the joined marching-tetrahedra sheet).")
      .def(
          "vof_interface_area", [](S& s) { return s.vofInterfaceArea(); },
          "Total interfacial area of the colour field, in cell units squared (h^2), summed over "
          "the inner region and globally reduced under MPI. Uses the geometry "
          "set_phase_change_area selects, so the number a page quotes and the number the phase "
          "change integrates are the same one. Needs enable_vof; phase change need not be on. "
          "A sphere of radius R cells reads 4 pi R^2 to about 0.2-0.8 % once its colour field is "
          "resolved (WO-P3c).")
      .def(
          "set_phase_change_energy",
          [](S& s, double rcp_gas, double rcp_liquid) {
            s.setPhaseChangeEnergy(rcp_gas, rcp_liquid);
          },
          nb::arg("rho_cp_gas"), nb::arg("rho_cp_liquid"),
          "CONSISTENT rho*c_p*T transport (VOF_PLAN section 9 item 6) for the scalar named by "
          "set_phase_change_thermal. Two things change together:\n"
          " (1) TRANSPORT: H = (rho c_p) T is advected with the colour advection's OWN geometric "
          "fluxes, sweep order and frozen dilation flag (vof/energy_advect.hpp), and T is recovered "
          "as H / (rho c_p)(C^{n+1}) — so a uniform temperature is preserved EXACTLY at any heat "
          "capacity ratio. With two different fluxes the heat carried into a mixed cell is divided "
          "by a capacity built from another flux: an error of order d(rho c_p), i.e. ~2000x at "
          "water/steam, which is the artificial interfacial heating this removes.\n"
          " (2) DIFFUSION: the implicit operator becomes A_C = rho c_p(C)/dt + sum_f k_f open_f "
          "with k_f the arithmetic mean of the cells' k(C) (the k_gas/k_liquid of "
          "set_phase_change_thermal) EXCEPT at a face touching a per-cell Dirichlet (interfacial) "
          "cell, where the pure neighbour's own k is used — a Dirichlet row is an identity row, so "
          "that coefficient's only job is the conductance with which the pure cell reaches a "
          "boundary condition that already sits at the interface.\n"
          "Units: rho*c_p in J/(cell^3 K), k in W/(cell K). Requires set_phase_change_thermal.")
      .def(
          "set_phase_change_energy_muscl",
          [](S& s, bool on) { s.setPhaseChangeEnergyMuscl(on); }, nb::arg("on"),
          "MinMod-limited donor reconstruction of the face temperature in the CONSISTENT energy "
          "flux, instead of the default plain donor-cell (first-order upwind) value. OFF by "
          "default. The consistency is what the geometric flux buys; its first-order upwind "
          "numerical diffusion |u| h (1 - CFL)/2 thickens the thermal boundary layer and therefore "
          "LOWERS the interfacial gradient, which is exactly what mdot is — measured on the P3 "
          "Scriven bubble at Ja = 0.5: -2.24 % on R(t) with plain upwind against -1.46 % for the "
          "scalar module's Koren TVD (which is NOT consistent). This buys the accuracy back; it is "
          "a switch and not a default because it is the energy twin of set_vof_momentum_muscl.")
      .def(
          "set_phase_change_energy_off", [](S& s) { s.setPhaseChangeEnergyOff(); },
          "Back to the constant-diffusivity scalar operator and the Koren TVD advective term "
          "(the rung P0/P1 energy path).")
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
          "set_divergence_sink",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setDivergenceSink(grid_in(a)); },
          nb::arg("weights"),
          "WO-P23: an AUTO-BALANCED sink region for the phase-change divergence source. Give a "
          "non-negative weight per cell (Fortran-order (nx,ny,nz)); after each deposit the solver "
          "subtracts (global sum of the phase-change source) * w / (global sum of w), so a CLOSED "
          "domain's Poisson RHS is exactly compatible every step with no user bookkeeping. Put the "
          "weights in the LIQUID far from the interface, where the exact solution simply has the "
          "liquid leaving. This is what makes a closed-box phase-change run possible without the "
          "variable-density outflow operator (whose density-ratio inconsistency is WO-R2's "
          "subject). Registered as the field 'div_sink'.")
      .def(
          "clear_divergence_sink", [](S& s) { s.clearDivergenceSink(); },
          "Drop the auto-balanced sink region.")
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
            r["area_hf_cells"] = d.areaHf;    // WO-P3c: the area cascade's branch census
            r["area_pv_cells"] = d.areaPv;
            r["area_no_cascade_cells"] = d.areaNone;
            r["area_orphan"] = d.areaOrphan;  // WO-P3d: area on cells the flux integral drops
            r["mdot_fit"] = d.mdotFit;        // WO-P3g: the least-squares estimator, diagnostic
            r["q_operator"] = d.qOperator;    // WO-P3g: the operator's own interfacial heat (W)
            r["q_orphan"] = d.qOrphan;        // ... on interfacial cells with no area
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
            r["band_div"] = d.bandDiv;
            r["T_min"] = d.Tmin;
            r["T_max"] = d.Tmax;
            return r;
          },
          "Per-rank phase-change census of the LAST step: mdot extrema/mean and the interfacial "
          "cell count, the total PLIC interface area, the liquid volume the regression removed, "
          "the clip-and-redistribute ledger (|deficit| moved, and how many cells clipped at 0 / "
          "1), the deposited divergence source (its sum and how many cells received it), how many "
          "interfacial cells found NO pure gas cell within two cells along +n (the source then "
          "stays put — a fallback that must be 0 on a resolved interface), the colour extrema, "
          "'band_div' = max|div(open u)| over the INTERFACIAL cells (WO-P23: the direct read-out of "
          "whether the field Weymouth-Yue advects with is the liquid velocity there — the "
          "band-extended velocity of VOF_PLAN section 9 item 3 is needed iff this is not at the "
          "projection floor), and the energy-scalar extrema under the consistent transport.")
      .def(
          "set_phase_change_fit_curvature",
          [](S& s, double k) { s.setPhaseChangeFitCurvature(k); }, nb::arg("kappa"),
          "WO-P3f INSTRUMENT (default 0, bitwise inert): prescribe the interface curvature "
          "`kappa = div(n)` that the one-sided temperature fits use to measure a sample's distance "
          "to the CURVED interface instead of to the interfacial cell's tangent plane "
          "(`vof::pcCurvedDistance`). For a spherical GAS bubble of radius R, whose PLIC normal "
          "points inward, kappa = -2/R.\n\n"
          "Why it exists: the tangent-plane distance makes the fit FIRST ORDER in h/R on a curved "
          "interface, and every off-axis sample is hotter than the plane model expects, so the "
          "fitted dT/dn and with it mdot come out HIGH. Measured a priori on an exact sphere with "
          "an exactly linear profile: +19.2 / +12.1 / +8.8 / +6.2 % at R = 6 / 10 / 14 / 20, "
          "observed order 0.91-0.98 in h/R. This entry point takes a PRESCRIBED kappa so that bias "
          "can be measured against a known geometry; it is not a curvature estimator.")
      .def(
          "set_phase_change_energy_order",
          [](S& s, int order) { s.setPhaseChangeEnergyOrder(order); }, nb::arg("order"),
          "WO-P3g: the ORDER of the interfacial energy operator. 1 (the shipped WO-P23...P3f "
          "scheme) or 2.\n\n"
          "`order = 2` turns on, TOGETHER, the four pieces WO-P3f's instruments indicated:\n"
          "  1. `set_phase_change_mdot_operator(True)` -- mdot is the energy operator's OWN "
          "interfacial flux q/(h_lv A) instead of a separate least-squares fit, so the heat the "
          "energy equation loses and the mass the regression produces are one discrete quantity "
          "(and the interfacial AREA cancels out of the mass balance entirely);\n"
          "  2. `set_phase_change_gfm_order(2)` -- the Gibou-Fedkiw three-point ghost-fluid row "
          "(2/((1+theta) theta), 2/(1+theta)) instead of the two-point (1/theta, 1), which is "
          "exact on a quadratic profile at every theta;\n"
          "  3. `set_phase_change_curvature_distance(True)` -- the row's theta and the one-sided "
          "fits' sample distances are measured to the CURVED interface, with kappa taken per cell "
          "from the V3 curvature cascade;\n"
          "  4. `set_phase_change_carry_conserve(True)` -- WO-P3f's enthalpy-conserving Dirichlet "
          "overwrite.\n\n"
          "Why together and not one at a time: WO-P3f measured the shipped scheme's 1 % Scriven "
          "error to be the residue of a CANCELLATION between the fit's +6 % curvature bias, the "
          "two-point row's -5 % flux deficit and the overwrite's -0.7...-4.3 % enthalpy "
          "destruction, so repairing any ONE alone makes the gate worse.")
      .def(
          "set_phase_change_deposit_fallback",
          [](S& s, bool on) { s.setPhaseChangeDepositFallback(on); }, nb::arg("on") = true,
          "WO-P3f open item 6 / WO-P3g (default from PECLET_PC_DEPOSIT_FALLBACK, i.e. OFF): give "
          "an interfacial cell whose two along-the-normal deposit candidates are BOTH still "
          "interfacial a target from the 5^3 box instead of leaving the divergence source in "
          "place. A cell that keeps its source carries div(open u) = S on its own faces, so "
          "Weymouth-Yue advects the colour with a field that is not the liquid velocity -- read it "
          "out with `phase_change_diagnostics()['band_div']` and the 'fallback_cells' count.")
      .def(
          "set_phase_change_mdot_operator",
          [](S& s, bool on) { s.setPhaseChangeMdotOperator(on); }, nb::arg("on") = true,
          "WO-P3g item 1 (default OFF): take mdot from the energy operator's own interfacial flux "
          "-- the sum of the ghost-fluid rows' Dirichlet couplings evaluated with the converged T "
          "-- instead of the one-sided least-squares fit, which stays as "
          "`phase_change_diagnostics()['mdot_fit']`.")
      .def(
          "set_phase_change_gfm_order", [](S& s, int o) { s.setPhaseChangeGfmOrder(o); },
          nb::arg("order"),
          "WO-P3g item 2 (default 1): the order of the one-sided (ghost-fluid) Dirichlet row. "
          "1 = the shipped two-point form k o (T_Gamma - T_i)/theta; 2 = Gibou-Fedkiw's "
          "three-point form through (T_behind, T_i, T_Gamma), which reproduces a quadratic "
          "temperature profile exactly at every theta.")
      .def(
          "set_phase_change_curvature_distance",
          [](S& s, bool on) { s.setPhaseChangeCurvatureDistance(on); }, nb::arg("on") = true,
          "WO-P3g item 3 (default OFF): measure the GFM row's theta and the one-sided fits' sample "
          "distances to the CURVED interface, with the mean curvature taken PER CELL from the V3 "
          "curvature cascade (the same kappa surface tension uses; positive for a convex blob of "
          "liquid, i.e. -2/R for a gas bubble). Supersedes `set_phase_change_fit_curvature`, which "
          "prescribes one curvature for the whole field; where both are set the cascade wins.")
      .def(
          "set_phase_change_carry_conserve",
          [](S& s, bool on) { s.setPhaseChangeCarryConserve(on); }, nb::arg("on") = true,
          "WO-P3f OPTION (default OFF): make the interfacial cells' per-cell Dirichlet OVERWRITE "
          "enthalpy-conserving.\n\n"
          "An interfacial cell's row is the identity `T = dval`, so whatever the geometric energy "
          "transport left there is discarded every step. Over a cell's interfacial lifetime that "
          "telescopes to rho c_p (T_entry - dval_exit): a liquid cell the interface sweeps enters "
          "with its superheat and leaves as vapour at T_sat, and the difference goes nowhere. "
          "Measured on Scriven 128^3 by `phase_change_budget()['d_overwrite']`: -0.7 % of "
          "mdot h_lv A_Gamma at Ja = 0.5 and -4.3 % at Ja = 2, one-signed.\n\n"
          "With this on, that enthalpy is handed to the interfacial cell's face neighbours that "
          "are still in the solve, weighted by n_d^2 (the clip-and-redistribute allocation), as a "
          "fixed-order gather so it is decomposition-independent. Which side receives is decided "
          "per axis by which pure neighbour deviates from T_Gamma in the same direction as the "
          "interfacial cell itself, i.e. the phase the enthalpy came from -- the superheated "
          "liquid on an evaporating bubble, the superheated vapour on the Stefan problem.")
      .def(
          "phase_change_carry_ledger",
          [](S& s) {
            nb::dict r;
            r["deposited"] = s.phaseChangeCarryDeposited();
            r["lost"] = s.phaseChangeCarryLost();
            return r;
          },
          "WO-P3f: the last `set_phase_change_carry_conserve` pass — the enthalpy actually handed "
          "back ('deposited') and the enthalpy of the interfacial cells that had NO neighbour left "
          "in the solve ('lost', which stays destroyed). Both 0 when the option is off.")
      .def(
          "set_phase_change_budget", [](S& s, bool on) { s.setPhaseChangeBudget(on); },
          nb::arg("on") = true,
          "WO-P3f INSTRUMENT: turn on the energy budget of the phase-change energy solve. Costs "
          "one extra cell field and two reductions per energy solve, and is skipped entirely when "
          "off (the solve is then bit-identical). Read it with `phase_change_budget()`.\n\n"
          "What it exists for: interfacial cells are Dirichlet rows, i.e. they are OUTSIDE the "
          "energy solve, so the set over which enthalpy is conserved changes membership every step "
          "as the interface sweeps. A liquid cell that becomes interfacial LEAVES that set carrying "
          "its superheat rho c_p (T - T_sat) and an interfacial cell that becomes pure RE-ENTERS it "
          "carrying `pcCarriedValue`; neither transfer appears in the latent-heat book-keeping.")
      .def(
          "phase_change_budget",
          [](S& s) {
            const auto b = s.phaseChangeBudgetValues();
            nb::dict r;
            r["h_open"] = b.hOpen;
            r["h_open_new"] = b.hOpenNew;
            r["h_liquid"] = b.hLiquid;
            r["h_masked"] = b.hMasked;
            r["d_overwrite"] = b.dEoverwrite;
            r["d_overwrite_new"] = b.dEoverwriteNew;
            r["e_enter"] = b.eEnter;
            r["e_leave"] = b.eLeave;
            r["q_gfm"] = b.qGfm;
            r["q_behind"] = b.qBehind;  // WO-P3g: the second-order row's one-sided band rescaling
            r["n_enter_liquid"] = b.nEnterLiquid;
            r["n_enter_gas"] = b.nEnterGas;
            r["n_leave_liquid"] = b.nLeaveLiquid;
            r["n_leave_gas"] = b.nLeaveGas;
            r["n_masked"] = b.nMasked;
            r["calls"] = b.calls;
            return r;
          },
          "WO-P3f: the ENERGY BUDGET of the LAST energy solve (units: J for the enthalpies, W for "
          "`q_gfm`; cell volume = 1).\n"
          "  h_open / h_open_new  sum rho c_p (T - T_sat) over the UNMASKED cells, before / after "
          "the solve\n"
          "  h_liquid, h_masked   the same over the pure-liquid and the interfacial cells\n"
          "  d_overwrite          sum rho c_p (dval - T) over masked cells: what the Dirichlet "
          "rows inject when they overwrite the transported temperature (negative = destroyed)\n"
          "  d_overwrite_new      the part of it on cells that were NOT masked last step\n"
          "  e_enter / e_leave    the enthalpy carried OUT of / INTO the solved set by the cells "
          "that changed class this step\n"
          "  q_gfm                the heat the plane-anchored rows deliver INTO the unmasked set "
          "(negative while a bubble grows); the energy equation's own interfacial flux, to be "
          "compared with the regression's `mdot h_lv A_Gamma`\n"
          "  n_enter_*/n_leave_*  the class-change census\n"
          "The discrete balance the entries close over one solve is "
          "`sum rho c_p (T^{n+1} - T*)/dt = q_gfm + (domain boundary flux) + (solve residual)`.")
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
