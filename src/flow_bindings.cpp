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
using peclet::flow::bindings::grid_in;
using peclet::flow::bindings::enum_index;
using peclet::flow::bindings::kCsfModes;
using peclet::flow::bindings::kAreaModes;
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
  cls
      // --- Phase change, Part II rungs P2/P3 (WO-P23) ------------------------------------------
      .def(
          "set_phase_change_plane_dirichlet",
          [](D& diag, bool on) { diag.s->setPhaseChangePlaneDirichlet(on); }, nb::arg("on"),
          "The PLANE-ANCHORED (ghost-fluid) interfacial Dirichlet condition. ON by default.\n\n"
          "Rungs P0/P1 pinned the whole interfacial CELL at T_G, so the numerical thermal boundary "
          "sat at the cell CENTRE while the mass-flux gradient is fitted from the PLIC PLANE — a "
          "mismatch of up to half a cell that CHANGES SIGN as the interface sweeps through a cell, "
          "and the first-order component of the P1 Stefan error (WO-P01 finding 6). With this on, "
          "the condition becomes PER FACE: for a pure cell i whose neighbour j is interfacial, "
          "that "
          "face's row carries k*open*(T_i - T_G)/theta with theta the distance in cells from i's "
          "centre to j's PLIC plane along the face's own axis, and the interfacial cell's own "
          "value "
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
          [](D& diag, bool on) { diag.s->setPhaseChangeQuadraticFit(on); }, nb::arg("on"),
          "Fit the one-sided interfacial temperature gradients QUADRATICALLY through the interface "
          "value (T - T_G = G phi + Q phi^2) instead of linearly. This is VOF_PLAN section 9 item "
          "1's Aslam quadratic extrapolation in least-squares form — the same 5^3 pure-cell "
          "samples, the same Malan collinearity weights, one more basis function, no PDE sweeps. "
          "Once the plane-anchored Dirichlet has removed the cell-centre mismatch, the linear "
          "fit's "
          "O(T'' h) curvature bias is the leading error of the rung: its samples start about one "
          "cell from the plane and reach two and a half, so a curved profile tilts the straight "
          "line through the origin. ON by default (WO-P23): with the plane-anchored Dirichlet it "
          "takes "
          "the P1 Stefan interface position from +0.195 % to +0.003 % at N = 256, and the mdot "
          "kernel itself from order 1.1 to order 2.0. set(False) is the ablation.")
      .def(
          "set_phase_change_area",
          [](D& diag, const std::string& mode) {
            diag.s->setPhaseChangeArea(
                enum_index(mode, kAreaModes, "set_phase_change_area", "mode"));
          },
          nb::arg("mode"),
          "WO-P3c: WHICH GEOMETRY the interfacial area A_Gamma comes from. A_Gamma sets the plane "
          "shift dV = mdot A dt / rho_l and the divergence source S = mdot A (1/rho_g - 1/rho_l), "
          "so a bubble grows as int mdot dA and a biased area is a biased growth rate.\n"
          "  'plic' (DEFAULT, rungs P0/P1): plicArea = |m|_2 dV/dalpha on the MYC normal.\n"
          "  'cascade-metric': the V3 curvature cascade's own geometry -- the height function's "
          "area element sqrt(1 + h_x^2 + h_y^2) from the SAME central differences the curvature "
          "differentiates once more (tiers 1/2), the PV paraboloid's gradient (tier 3) -- applied "
          "to the PLIC polygon's projected footprint, so the cells of a column still tile.\n"
          "  'cascade-normal': the same normals, but the plane is rebuilt on them, "
          "plicArea(n*, plicAlpha(n*, C)).\n"
          "  'cascade-footprint': the height function's OWN footprint times its own metric, the "
          "only per-cell variant whose pieces tile.\n"
          "  'sheet-color-centroid' / 'sheet-color-split' / 'sheet-plic-centroid' / "
          "'sheet-plic-split' = WO-P3d, the JOINED sheet: marching tetrahedra on the cell-centre "
          "lattice, one watertight surface whose triangles are booked to cells -- the C = 1/2 "
          "level "
          "set with whole triangles to the cell holding the centroid, the same sheet clipped to "
          "each cell's cube, and the same two deposits on the zero of the PLIC-reconstructed "
          "signed distance (exact on a TILTED plane, where interpolating C is not, because C(d) is "
          "the SZ piecewise cubic). The sheet modes are the only ones whose SUM converges on a "
          "curved interface: WO-P3c proved with two analytic controls that every PER-CELL area is "
          "first order in h/R because the pieces do not JOIN across cells.\n"
          "All of the per-cell modes are EXACT on a plane, so every planar gate (P0a/P0b/P1/P2) is "
          "unmoved. The default is 'plic' because the measurement says so: on a sphere whose "
          "colour field is resolved (16^3 sub-sampling), summed plicArea is within 0.5 % of "
          "4 pi R^2 and the cascade does not improve it. WO-P3b's 5.5-9.3 % 'PLIC area deficit' "
          "was its probe's own 4^3 sub-sampling, which quantizes C to 1/64 and drops a QUARTER of "
          "the interfacial cells (their volume is 1e-4 %, their area 6 %).")
      .def(
          "phase_change_area", [](D& diag) { return kAreaModes.at(diag.s->phaseChangeArea()); },
          "The set_phase_change_area mode in force, as its name.")
      .def(
          "set_phase_change_energy_muscl",
          [](D& diag, bool on) { diag.s->setPhaseChangeEnergyMuscl(on); }, nb::arg("on"),
          "MinMod-limited donor reconstruction of the face temperature in the CONSISTENT energy "
          "flux, instead of the default plain donor-cell (first-order upwind) value. OFF by "
          "default. The consistency is what the geometric flux buys; its first-order upwind "
          "numerical diffusion |u| h (1 - CFL)/2 thickens the thermal boundary layer and therefore "
          "LOWERS the interfacial gradient, which is exactly what mdot is — measured on the P3 "
          "Scriven bubble at Ja = 0.5: -2.24 % on R(t) with plain upwind against -1.46 % for the "
          "scalar module's Koren TVD (which is NOT consistent). This buys the accuracy back; it is "
          "a switch and not a default because it is the energy twin of set_vof_momentum_muscl.")
      .def(
          "phase_change_diagnostics",
          [](D& diag) {
            const auto d = diag.s->phaseChangeDiagnostics();
            nb::dict r;
            r["mdot_min"] = d.mdotMin;
            r["mdot_max"] = d.mdotMax;
            r["mdot_mean"] = d.mdotMean;
            r["interface_cells"] = d.interfaceCells;
            r["interface_area"] = d.area;
            r["area_hf_cells"] = d.areaHf;  // WO-P3c: the area cascade's branch census
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
          "'band_div' = max|div(open u)| over the INTERFACIAL cells (WO-P23: the direct read-out "
          "of "
          "whether the field Weymouth-Yue advects with is the liquid velocity there — the "
          "band-extended velocity of VOF_PLAN section 9 item 3 is needed iff this is not at the "
          "projection floor), and the energy-scalar extrema under the consistent transport.\n"
          "Units are the solver's physical ones: mdot_* a mass flux, interface_area/area_orphan "
          "an area, q_* a power, removed_volume/redistributed/unresolved a liquid VOLUME (colour "
          "x cell volume), source_sum and band_div 1/time; colours and temperatures as given.")
      .def(
          "set_phase_change_fit_curvature",
          [](D& diag, double k) { diag.s->setPhaseChangeFitCurvature(k); }, nb::arg("kappa"),
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
          [](D& diag, int order) { diag.s->setPhaseChangeEnergyOrder(order); }, nb::arg("order"),
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
          [](D& diag, bool on) { diag.s->setPhaseChangeDepositFallback(on); }, nb::arg("on") = true,
          "WO-P3f open item 6 / WO-P3g (OFF by default): give "
          "an interfacial cell whose two along-the-normal deposit candidates are BOTH still "
          "interfacial a target from the 5^3 box instead of leaving the divergence source in "
          "place. A cell that keeps its source carries div(open u) = S on its own faces, so "
          "Weymouth-Yue advects the colour with a field that is not the liquid velocity -- read it "
          "out with `phase_change_diagnostics()['band_div']` and the 'fallback_cells' count.")
      .def(
          "set_phase_change_mdot_operator",
          [](D& diag, bool on) { diag.s->setPhaseChangeMdotOperator(on); }, nb::arg("on") = true,
          "WO-P3g item 1 (default OFF): take mdot from the energy operator's own interfacial flux "
          "-- the sum of the ghost-fluid rows' Dirichlet couplings evaluated with the converged T "
          "-- instead of the one-sided least-squares fit, which stays as "
          "`phase_change_diagnostics()['mdot_fit']`. Under a physical domain this needs "
          "set_phase_change_energy: the constant-diffusivity flux carries no heat capacity, so "
          "the step raises rather than assume one.")
      .def(
          "set_phase_change_gfm_order", [](D& diag, int o) { diag.s->setPhaseChangeGfmOrder(o); },
          nb::arg("order"),
          "WO-P3g item 2 (default 1): the order of the one-sided (ghost-fluid) Dirichlet row. "
          "1 = the shipped two-point form k o (T_Gamma - T_i)/theta; 2 = Gibou-Fedkiw's "
          "three-point form through (T_behind, T_i, T_Gamma), which reproduces a quadratic "
          "temperature profile exactly at every theta.")
      .def(
          "set_phase_change_curvature_distance",
          [](D& diag, bool on) { diag.s->setPhaseChangeCurvatureDistance(on); },
          nb::arg("on") = true,
          "WO-P3g item 3 (default OFF): measure the GFM row's theta and the one-sided fits' sample "
          "distances to the CURVED interface, with the mean curvature taken PER CELL from the V3 "
          "curvature cascade (the same kappa surface tension uses; positive for a convex blob of "
          "liquid, i.e. -2/R for a gas bubble). Supersedes `set_phase_change_fit_curvature`, which "
          "prescribes one curvature for the whole field; where both are set the cascade wins.")
      .def(
          "set_phase_change_carry_conserve",
          [](D& diag, bool on) { diag.s->setPhaseChangeCarryConserve(on); }, nb::arg("on") = true,
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
          [](D& diag) {
            nb::dict r;
            r["deposited"] = diag.s->phaseChangeCarryDeposited();
            r["lost"] = diag.s->phaseChangeCarryLost();
            return r;
          },
          "WO-P3f: the last `set_phase_change_carry_conserve` pass — the enthalpy actually handed "
          "back ('deposited') and the enthalpy of the interfacial cells that had NO neighbour left "
          "in the solve ('lost', which stays destroyed). Both 0 when the option is off. Units as "
          "phase_change_budget's enthalpies.")
      .def(
          "set_phase_change_budget", [](D& diag, bool on) { diag.s->setPhaseChangeBudget(on); },
          nb::arg("on") = true,
          "WO-P3f INSTRUMENT: turn on the energy budget of the phase-change energy solve. Costs "
          "one extra cell field and two reductions per energy solve, and is skipped entirely when "
          "off (the solve is then bit-identical). Read it with `phase_change_budget()`.\n\n"
          "What it exists for: interfacial cells are Dirichlet rows, i.e. they are OUTSIDE the "
          "energy solve, so the set over which enthalpy is conserved changes membership every step "
          "as the interface sweeps. A liquid cell that becomes interfacial LEAVES that set "
          "carrying "
          "its superheat rho c_p (T - T_sat) and an interfacial cell that becomes pure RE-ENTERS "
          "it "
          "carrying `pcCarriedValue`; neither transfer appears in the latent-heat book-keeping.")
      .def(
          "phase_change_budget",
          [](D& diag) {
            const auto b = diag.s->phaseChangeBudgetValues();
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
          "WO-P3f: the ENERGY BUDGET of the LAST energy solve, in the solver's physical units: "
          "energies for the enthalpies and powers for `q_gfm`/`q_behind` with "
          "set_phase_change_energy on; without it the constant-diffusivity scalar carries no heat "
          "capacity and they are temperature x volume (and per time).\n"
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
  cls
      // --- Phase change, Part II rungs P0/P1 (WO-P01) ------------------------------------------
      .def(
          "enable_phase_change",
          [](S& s, double rho_g, double rho_l, double h_lv) {
            s.enablePhaseChange(rho_g, rho_l, h_lv);
          },
          nb::arg("rho_gas"), nb::arg("rho_liquid"), nb::arg("h_lv") = 1.0,
          "Enable VoF phase change (VOF_PLAN section 9, rungs P0/P1; the Boyd & Ling 2023 / Malan "
          "2021 pattern). rho_gas/rho_liquid are densities and h_lv the latent heat (energy per "
          "mass), all in the solver's physical units. Registers 'mdot' (the interfacial mass flux, "
          "POSITIVE = evaporation) and 'pc_source' (the deposited divergence source); both are "
          "raw-registry fields and hold INTERNAL values -- multiply by 1/unit_scales"
          "['mass_flux_to_internal'] and 1/unit_scales['divergence_to_internal'] (both 1 "
          "without an extent).\n"
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
          "set_mass_flux_uniform", &chk::setMassFluxUniform<S>, nb::arg("mdot"),
          "P0: prescribe a UNIFORM interfacial mass flux (mass per area per time; positive = "
          "evaporation, liquid consumed). Turns the thermal mass flux OFF. Under an extent, call "
          "set_rho and set_dt first: the value is converted into the 'mdot' field once.")
      .def(
          "set_mass_flux",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setMassFlux(grid_in(a)); },
          nb::arg("array"),
          "P0: prescribe a per-cell interfacial mass flux from a Fortran-order (nx,ny,nz) float64 "
          "array, in the units of set_mass_flux_uniform (and with its call-order rule). Turns "
          "the thermal mass flux OFF.")
      .def(
          "set_phase_change_thermal",
          [](S& s, bool enabled, const std::string& name, double t_sat, double k_gas,
             double k_liquid, double r_int) {
            if (!enabled) {
              s.setPhaseChangeThermalOff();
              return;
            }
            if (name.empty())
              throw std::invalid_argument(
                  "set_phase_change_thermal(True, ...): name the registered temperature scalar");
            s.setPhaseChangeThermal(name, t_sat, k_gas, k_liquid, r_int);
          },
          nb::arg("enabled"), nb::arg("scalar") = "", nb::arg("t_sat") = 0.0,
          nb::arg("k_gas") = 0.0, nb::arg("k_liquid") = 0.0, nb::arg("r_int") = 0.0,
          "P1: compute mdot each step from a registered scalar (the temperature) instead of "
          "prescribing it; `enabled=False` stops that (back to the prescribed mdot field):\n\n"
          "    mdot = ( k_g dT_g/dn - k_l dT_l/dn ) / h_lv ,   n = m/|m|_2 (LIQUID -> GAS)\n\n"
          "Each one-sided derivative is a weighted least-squares fit THROUGH the interface value "
          "over the PURE-PHASE cells of a 5^3 stencil on that side, with Malan's collinearity "
          "weight w = (d.n)^2/|d|^3 and the exact normal distance from the PLIC plane. Interfacial "
          "cells are pinned at T_G = t_sat + mdot*r_int in the energy solve through a per-cell "
          "Dirichlet mask on the scalar (r_int = 0 is the hard Dirichlet; a nonzero r_int is the "
          "Schrage/IHTR Robin condition of Bures & Sato 2021).\n\n"
          "SIGN CONVENTION -- the interfacial energy balance is mdot*h_lv = (q_l - q_g).n with "
          "q = -k grad T and n pointing OUT OF THE LIQUID, which is the form above. Check it on "
          "the Stefan problem: superheated vapour behind the interface gives grad(T_g).n > 0 and "
          "mdot > 0, i.e. evaporation. The opposite pairing (k_l grad T_l - k_g grad T_g) with the "
          "SAME n would condense a superheated vapour.\n\n"
          "The energy scalar's diffusivity is whatever add_scalar was given (a CONSTANT); per-cell "
          "k(C) and the consistent rho c_p T geometric transport are set_phase_change_energy. That "
          "is not a limitation for the Stefan gate, where the liquid is saturated and pinned at "
          "T_sat.\n\n"
          "Units: t_sat in the temperature scalar's own units (never rescaled); k_gas/k_liquid "
          "are thermal conductivities (power per length per kelvin) and r_int an interfacial "
          "resistance (kelvin per mass flux), all in the solver's physical units.")
      .def(
          "vof_interface_area", [](S& s) { return s.vofInterfaceArea(); },
          "Total interfacial area of the colour field (an area in the solver's units; cells "
          "squared without an extent), summed over "
          "the inner region and globally reduced under MPI. Uses the geometry "
          "diagnostics.set_phase_change_area selects, so the number a page quotes and the number the phase "
          "change integrates are the same one. Needs enable_vof; phase change need not be on. "
          "A sphere of radius R reads 4 pi R^2 to about 0.2-0.8 % once its colour field is "
          "resolved (WO-P3c).")
      .def(
          "set_phase_change_energy",
          [](S& s, bool enabled, double rcp_gas, double rcp_liquid) {
            if (!enabled) {
              s.setPhaseChangeEnergyOff();
              return;
            }
            if (!(rcp_gas > 0.0) || !(rcp_liquid > 0.0))
              throw std::invalid_argument(
                  "set_phase_change_energy(True, ...): both heat capacities must be positive");
            s.setPhaseChangeEnergy(rcp_gas, rcp_liquid);
          },
          nb::arg("enabled"), nb::arg("rho_cp_gas") = 0.0, nb::arg("rho_cp_liquid") = 0.0,
          "CONSISTENT rho*c_p*T transport (VOF_PLAN section 9 item 6) for the scalar named by "
          "set_phase_change_thermal; `enabled=False` returns to the constant-diffusivity scalar "
          "operator and the Koren TVD advective term (the rung P0/P1 energy path). Two things "
          "change together:\n"
          " (1) TRANSPORT: H = (rho c_p) T is advected with the colour advection's OWN geometric "
          "fluxes, sweep order and frozen dilation flag (vof/energy_advect.hpp), and T is recovered "
          "as H / (rho c_p)(C^{n+1}) -- so a uniform temperature is preserved EXACTLY at any heat "
          "capacity ratio. With two different fluxes the heat carried into a mixed cell is divided "
          "by a capacity built from another flux: an error of order d(rho c_p), i.e. ~2000x at "
          "water/steam, which is the artificial interfacial heating this removes.\n"
          " (2) DIFFUSION: the implicit operator becomes A_C = rho c_p(C)/dt + sum_f k_f open_f "
          "with k_f the arithmetic mean of the cells' k(C) (the k_gas/k_liquid of "
          "set_phase_change_thermal) EXCEPT at a face touching a per-cell Dirichlet (interfacial) "
          "cell, where the pure neighbour's own k is used -- a Dirichlet row is an identity row, so "
          "that coefficient's only job is the conductance with which the pure cell reaches a "
          "boundary condition that already sits at the interface.\n"
          "Units: rho_cp_* are VOLUMETRIC heat capacities (energy per volume per kelvin) in the "
          "solver's physical units, like the k of set_phase_change_thermal. Requires "
          "set_phase_change_thermal.")
      .def(
          "set_divergence_source",
          [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setDivergenceSource(grid_in(a)); },
          nb::arg("array"),
          "A PRESCRIBED extra divergence source (1/time; under an extent call set_dt first) on "
          "the inner cells, added to the Poisson RHS "
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
          "momentum and pressure solves out of the picture. dt is in the caller's time units.")
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
