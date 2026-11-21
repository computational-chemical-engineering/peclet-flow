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
#include "flow_ibm.hpp"
#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

namespace nb = nanobind;
// The topic binders (flow_bindings_<topic>.hpp).
namespace fb = peclet::flow::bindings;
// Input validation and the checked setters bound in place of a member (flow_bindings_checks.hpp).
namespace chk = peclet::flow::binding_checks;
// The shared binding helpers (flow_bindings_common.hpp).
#ifdef PECLET_FLOW_MPI
using peclet::flow::bindings::ensure_mpi_init;
#endif
using peclet::flow::bindings::field_out;
using peclet::flow::bindings::enum_index;
using peclet::flow::bindings::kCsfModes;
using peclet::flow::bindings::BoundSolver;
using peclet::flow::bindings::BoundAccelerator;
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
  cls
      .def(
          "set_csf_mode",
          [](D& diag, const std::string& mode) {
            diag.s->setCsfMode(enum_index(mode, kCsfModes, "set_csf_mode", "mode"));
          },
          nb::arg("mode"),
          "ABLATION. 'face' (default, the only production mode) evaluates the surface-tension "
          "force as sigma*kappa_f*(C(i)-C(i-s))/h at the face -- the projection's own gradient "
          "operator. 'cell' evaluates a CELL-CENTRED sigma*kappa*grad(C) and face-interpolates it "
          "with the arithmetic mean, exactly as the per-cell body-force machinery carries a rho*g "
          "field: consistent, convergent, and wrong for surface tension, because the result is not "
          "in the range of the operator the projection inverts. Shipped so the difference is a "
          "measured number (see the vof_surface_tension ctest), not an argument.")
      .def(
          "set_vof_interface_eps", [](D& diag, double eps) { diag.s->setVofInterfaceEps(eps); },
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
          "vof_interface_eps", [](D& diag) { return diag.s->vofInterfaceEps(); },
          "The wisp threshold set by set_vof_interface_eps.")
      .def(
          "set_vof_block_kappa_clip",
          [](D& diag, bool enabled, std::optional<double> kappaMax) {
            diag.s->setVofBlockKappaClip(enabled, kappaMax ? *kappaMax : -1.0);
          },
          nb::arg("enabled"), nb::arg("kappa_max") = nb::none(),
          "Curvature admissibility clip of the BLOCK container (doc/vof_overlap_design.md 5.1, "
          "11): after each marker's cascade every interfacial cell gets |kappa| <= kappa_max. "
          "DEFAULT ON at kappa_max = 1/Delta_min. kappa_max is PHYSICAL (1/length); None selects "
          "the default. enabled=False restores the pre-clip block cascade verbatim (the "
          "ablation). The single-field cascade never clips: there a large |kappa| is legitimate "
          "(an under-resolved droplet, a contact-line cell).\n\n"
          "Why: a mixed cell with no body of its marker near it (debris left where two markers "
          "overlapped) has no closing height-function column and a PLIC-volumetric fit through "
          "polygons of area ~0, so its paraboloid is arbitrary (measured |kappa| 273 and 428 per "
          "cell against a true 0.4 on channel_18; one-step velocity blow-up). |kappa| <= 1/Delta "
          "is a sphere of R >= 2 Delta, the smallest the height function can see, and bounds every "
          "CSF face force by sigma |dC| / Delta^2. The value is written only where it changes, so "
          "a resolved interface is bit-identical; vof_block_curvature_stats() reports 'clipped'.")
      .def(
          "set_vof_kappa_frozen", [](D& diag, bool on) { diag.s->setVofKappaFrozen(on); },
          nb::arg("on") = true,
          "INSTRUMENT: stop recomputing the curvature at the head of each step and use whatever is "
          "in the 'kappa'/'kappa_branch' fields. With set_vof_kappa_constant this isolates the "
          "balanced-force identity from the curvature estimator.")
      .def(
          "set_vof_kappa_constant", [](D& diag, double k) { diag.s->setVofKappaConstant(k); },
          nb::arg("kappa"),
          "INSTRUMENT: set kappa to a constant everywhere (inner + ghosts), mark every cell as "
          "carrying a valid estimate, and freeze it. The CSF force is then EXACTLY the discrete "
          "gradient of sigma*kappa*C, so the projection must annihilate it to round-off from any "
          "colour field whatsoever — the exactness gate of rung V4, independent of curvature "
          "accuracy and of resolution.")
      .def(
          "set_rho_face_harmonic", [](D& diag, bool on) { diag.s->setRhoFaceHarmonic(on); },
          nb::arg("on") = true,
          "Use the HARMONIC instead of the arithmetic face mean of rho in the pressure projection "
          "(both the operator coefficient and the velocity correction, so the projection stays "
          "exact). DEFAULT OFF, and it should stay off: the arithmetic mean of rho IS the harmonic "
          "mean of the mobility 1/rho — the series-correct choice for a normal flux — and it is "
          "what makes the discrete hydrostatic balance EXACT, because the momentum time term and "
          "the face body force interpolate rho arithmetically and are not switched by this flag. "
          "Shipped as a measured knob for the coefficient-coarsening question (VOF_PLAN S3), not "
          "as an alternative scheme.")
      .def(
          "set_vof_rho_floor", [](D& diag, double f) { diag.s->setVofRhoFloorFrac(f); },
          nb::arg("fraction"),
          "Floor on rho^c in the recovery divide u = (rho^c u)/rho^c, as a FRACTION of "
          "min(rho_c0, rho_c1). Default 1e-6. rho^c leaves [rho_c0, rho_c1] only through "
          "a wisp in the half-shifted colour and reaching zero would need C^c ~ -1/(ratio-1), so "
          "this is a guard, not a model — vof_momentum_diagnostics()['floored'] reports how many "
          "control volumes it actually touched.")
      .def(
          "vof_rho_floor", [](D& diag) { return diag.s->vofRhoFloor(); },
          "The absolute rho^c floor used by the last recovery.")
      .def(
          "set_vof_momentum_scheme",
          [](D& diag, const std::string& name) { diag.s->setVofMomentumScheme(name); },
          nb::arg("name"),
          "The transported face velocity of the VoF momentum flux (flow "
          "doc/vof_momentum_second_order.md). 'upwind': donor cell, convex at every density "
          "ratio, first order (the default until 2026-10-31). 'muscl_mixed': a MinMod slope of "
          "the MIXED control-volume velocity; it converges to a WRONG limit at high ratio (P1, "
          "ratio ~1e3: +3.0/+3.1 % at d/h 16/24; the mechanism is open). 'muscl_mixed_capped': "
          "that slope times the mass-aware cap (the diagnosis ablation). 'muscl_phase': each "
          "phase's mass flux carries its own face velocity from an interface-biased MC slope, "
          "scaled by the cap - second order, a discrete maximum principle at any ratio, the "
          "uniform-velocity identity bit-exact; the DEFAULT. vof_momentum_diagnostics()['capped'] "
          "counts the faces the cap scaled in the last step. Any other name raises ValueError.")
      .def(
          "vof_momentum_scheme", [](D& diag) { return diag.s->vofMomentumScheme(); },
          "The selected VoF momentum scheme (see set_vof_momentum_scheme).")
      ;
  fb::bindDiagnosticsSharpViscous<Grid>(cls);
  cls
      .def(
          "set_vof_momentum_muscl", [](D& diag, bool on) { diag.s->setVofMomentumMuscl(on); },
          nb::arg("on") = true,
          "Former spelling of set_vof_momentum_scheme: on = 'muscl_mixed' (a MinMod slope of the "
          "mixed control-volume velocity), off = 'upwind'. Use set_vof_momentum_scheme; "
          "'muscl_phase' is the second-order scheme that stays bounded at high density ratio.")
      .def(
          "set_vof_momentum_cell_flag",
          [](D& diag, bool on) { diag.s->setVofMomentumCellFlag(on); }, nb::arg("on") = true,
          "ABLATION: use the PRESSURE-cell frozen dilation flag H(C^n-1/2) on the shifted control "
          "volume instead of its structural analogue H(C^c,n-1/2). The flag that must be shared is "
          "the one of the pair that telescopes, and both members of that pair live on the shifted "
          "volume — this switch is the literal reading of the work order, kept as a measurement.")
      .def(
          "set_vof_flux_clamp", [](D& diag, bool on) { diag.s->setVofFluxClamp(on); },
          nb::arg("on") = true,
          "ABLATION: drop the Weymouth flux clamp max(0,|a|-(1-C^c_don)) <= |F| <= "
          "min(|a|,C^c_don) "
          "on the half-shifted control volume. The geometric flux is bounded by what the CURRENT "
          "cell planes see in the donor, not by the ADVECTED C^c; the gap is O(a^2) and at density "
          "ratio 1e4 a 2.6e-2 undershoot drives rho^c to -255, which the recovery would divide by. "
          "Default ON; off is how that statement stays a measured number.")
      .def(
          "vof_momentum_diagnostics",
          [](D& diag) {
            const auto d = diag.s->vofMomentumDiagnostics();
            nb::dict r;
            r["min_Cc"] = nb::make_tuple(d.minCc[0], d.minCc[1], d.minCc[2]);
            r["max_Cc"] = nb::make_tuple(d.maxCc[0], d.maxCc[1], d.maxCc[2]);
            r["sum_momentum"] = nb::make_tuple(d.sumM[0], d.sumM[1], d.sumM[2]);
            r["min_rho_c"] = d.minRhoC;
            r["floored"] = d.floored;
            r["clamped"] = d.clamped;
            r["capped"] = d.capped;
            return r;
          },
          "Census over THIS RANK's momentum control volumes: per-component min/max of the "
          "half-shifted colour C^c, the summed momentum rho^c u_c (the conservation census), the "
          "minimum rho^c before the floor, and the number of control volumes the floor touched; "
          "'clamped' and 'capped' count the faces "
          "the Weymouth clamp and the mass-aware cap (set_vof_momentum_scheme) touched in the last "
          "step.")
      .def(
          "vof_advected_velocity",
          [](D& diag, int c) { return field_out(*diag.s, diag.s->getVofAdvectedVelocity(c)); },
          nb::arg("component"),
          "The recovered advected velocity (rho^c u_c)/rho^c of component c on the inner cells — "
          "the momentum RHS's time base. Exposed so the uniform-velocity consistency identity can "
          "be gated on the ADVECTION ALONE, with the projection and the momentum solve out of the "
          "picture.")
      .def(
          "set_property_mode",
          [](D& diag, const std::string& mode, bool harmonic) {
            diag.s->setPropertyMode(mode == "variable", harmonic);
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
          [](D& diag, const std::string& mode, double chi) {
            int m = 0;
            if (mode == "min")
              m = 0;
            else if (mode == "full")
              m = 1;
            else if (mode == "off")
              m = 2;
            else
              throw std::runtime_error("set_variable_rotational: mode must be min/full/off");
            diag.s->setVariableRotational(m, chi);
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
          [](D& diag, const std::string& mode) { diag.s->setDensityMode(mode == "variable"); },
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
          "as on the staggered grid, applied to the ABC projection's MAC face field, which is the "
          "MOMENTUM-weighted map (rho_L u_L + rho_R u_R)/(rho_L + rho_R) of the cell velocities; "
          "the "
          "CELL correction is the average of the two face corrections of each axis (never a "
          "cell-centred grad(phi)/rho_c). The pressure, the CSF and set_body_force (a mean "
          "pressure "
          "gradient) enter the implicit momentum predictor as the matching face integral "
          "rho_c * avg_faces((f - grad_f P)/rho_f); a per-cell force enters at the cell value. The "
          "pair is adjoint, so the step is stable at every dt and density ratio and its steady "
          "state is dt-independent (doc/collocated_varrho_forces.md). Rated to density ratio ~10 "
          "for cases WITH MOTION (a translating drop exceeds the VoF CFL cap at ratio 100 and "
          "1000; momentum consistency needs Favre face states and is not in this rung); exact at "
          "rest at ratio 1000.")
      ;
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
  cls
      // --- Momentum-consistent VoF transport (rung V2b, WO-K) ----------------------------------
      .def(
          "enable_vof_momentum",
          [](S& s, double rg, double rl) { s.enableVofMomentum(rg, rl); }, nb::arg("rho_c0"),
          nb::arg("rho_c1"),
          "Advect rho*u on the HALF-SHIFTED momentum control volumes with the SAME geometric "
          "fluxes, the same sweep order and one frozen dilation flag as the colour field, then "
          "recover u = (rho^c u)/rho^c. This is what makes the scheme usable above density ratio "
          "~100 (Rudman 1998; Arrufat 2021: a raindrop accurate within 15% at 15 cells/diameter "
          "instead of ~200).\n\n"
          "The two densities are REQUIRED, and must be the ones the rho closure produces at "
          "C=0 and C=1 — the momentum flux is rho_c0*(a-F) + rho_c1*F with F the geometric "
          "flux of the C = 1 phase, so the scheme has to know which density each side carries. "
          "ORDER: rho_c0 is the density at C = 0 and rho_c1 the density at C = 1, whichever "
          "phase that is (when C = 1 marks the gas, a bubble colour, rho_c1 is the gas density). "
          "When a 'rho' closure on 'C' is already "
          "registered (set_property_model / set_property_table), the pair is checked against it "
          "at C = 0 and C = 1 and a mismatch raises, naming both; registered after this call, it "
          "is not checked.\n\n"
          "Turning this on MOVES the VoF advection from the end of step() to its head (the momentum "
          "advection must precede the predictor that consumes it), so the advecting field is u^n — "
          "the previous step's projected output. u at step 0 must therefore be discretely "
          "divergence-free (rest or a uniform field both are). Requires the variable-density path, "
          "staggered layout, explicit advection, no immersed solid and no porous continuity, and is "
          "refused together with the per-bubble block container (enable_vof_blocks*, in either "
          "order): the momentum rides the structured colour's fluxes, not the markers'. "
          "rho_c0 and rho_c1 are in the caller's density units.")
      .def(
          "enable_vof_momentum",
          [](S& s, double rg, double rl) { s.enableVofMomentum(rg, rl); }, nb::arg("rho_gas"),
          nb::arg("rho_liquid"),
          "Former keyword names of enable_vof_momentum(rho_c0, rho_c1), the canonical spelling: "
          "rho_gas = rho_c0 (density at C = 0), rho_liquid = rho_c1 (density at C = 1), whichever "
          "phase each is. Deprecated in the next release, removed at 2.0.0.")
      .def(
          "vof_momentum_enabled", [](S& s) { return s.vofMomentumEnabled(); },
          "Whether momentum-consistent transport is on.")
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
            chk::finiteEach("set_property_model", "params", params, "the closure's units");
            s.setPropertyModel(target, k, in0, in1, params);
          },
          nb::arg("target"), nb::arg("kind"), nb::arg("field"),
          nb::arg("params") = std::vector<double>{}, nb::arg("field2") = std::string{},
          "Register a device closure writing a property/body-force field from input field(s). "
          "target: a registered field (a property 'mu'/'rho'/… or a body-force component "
          "'force_x'/'force_y'/'force_z'). kind: 'linear' (params [p0,p1,p2]: "
          "p0+p1*field+p2*field2), "
          "'boussinesq' (params [rho0,g,beta,T0]: rho0*g*beta*(field-T0) buoyancy), 'arrhenius' "
          "(params [mu_ref,B,Tref]: mu_ref*exp(B*(1/field-1/Tref))). Applied at the top of step().\n\n"
          "Units: the input fields are taken as UNSCALED quantities (a phase fraction, a "
          "temperature, a concentration -- the solver never rescales a transported scalar), so on "
          "a target 'rho', 'mu' or 'force_x/y/z' the parameters are in the solver's PHYSICAL "
          "units (a density, a viscosity, a force per volume: rho0*g*beta*(T-T0) as written); "
          "any other target takes its field's INTERNAL units (see unit_scales).")
      .def(
          "set_property_table", &chk::setPropertyTable<S>, nb::arg("target"), nb::arg("field"),
          nb::arg("x"), nb::arg("y"),
          "Register a tabulated property: target = piecewise-linear interpolation of (x, y) at the "
          "input field value (x ascending, clamped at the ends). Units as set_property_model: y in "
          "the target's physical units on 'rho'/'mu'/'force_*', x in the input field's.")
      .def(
          "update_properties", [](S& s) { s.updateProperties(); },
          "Apply all registered property/force closures now (also done at the top of step()).")
      .def(
          "enable_cell_force", [](S& s) { s.enableCellForce(); },
          "Allocate + register the per-cell body-force fields force_x/force_y/force_z and route "
          "them "
          "into the momentum RHS, for an external writer (e.g. CFD-DEM drag feedback) to fill "
          "directly via diagnostics.field_view('force_z'). They persist across steps until overwritten. "
          "They are raw registry fields: a force per unit volume in INTERNAL units (multiply a "
          "physical one by unit_scales['force_density_to_internal']).")
      .def(
          "enable_drag", [](S& s) { s.enableDrag(); },
          "Enable implicit (semi-implicit) linear drag for CFD-DEM: allocate the per-cell "
          "'drag_beta' "
          "field (added to the momentum diagonal so a -beta*(u-u_p) source is treated implicitly "
          "-> "
          "unconditionally stable for the stiff beta of a dense bed) plus force_x/y/z (which carry "
          "beta*u_p, the RHS target). Fill 'drag_beta' and 'force_*' via diagnostics.field_view each step.")
      ;
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
