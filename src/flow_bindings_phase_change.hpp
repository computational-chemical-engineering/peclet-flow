/// @file
/// @brief Python binding of peclet.flow: phase change (mass flux, energy, divergence source).
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): phase change.
template <class Grid, class C>
inline void bindSolverPhaseChange(C& cls) {
  using S = BoundSolver<Grid>;
  // --- Phase change, Part II rungs P0/P1 (WO-P01) ------------------------------------------
  cls.def(
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
      "set_divergence_source() with a balancing sink, or rho_g = rho_l (then S == 0).");
  cls.def("set_mass_flux_uniform", &chk::setMassFluxUniform<S>, nb::arg("mdot"),
          "P0: prescribe a UNIFORM interfacial mass flux (mass per area per time; positive = "
          "evaporation, liquid consumed). Turns the thermal mass flux OFF. Under an extent, call "
          "set_rho and set_dt first: the value is converted into the 'mdot' field once.");
  cls.def(
      "set_mass_flux", [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setMassFlux(grid_in(a)); },
      nb::arg("array"),
      "P0: prescribe a per-cell interfacial mass flux from a Fortran-order (nx,ny,nz) float64 "
      "array, in the units of set_mass_flux_uniform (and with its call-order rule). Turns "
      "the thermal mass flux OFF.");
  cls.def(
      "set_phase_change_thermal",
      [](S& s, bool enabled, const std::string& name, double t_sat, double k_gas, double k_liquid,
         double r_int) {
        if (!enabled) {
          s.setPhaseChangeThermalOff();
          return;
        }
        if (name.empty())
          throw std::invalid_argument(
              "set_phase_change_thermal(True, ...): name the registered temperature scalar");
        s.setPhaseChangeThermal(name, t_sat, k_gas, k_liquid, r_int);
      },
      nb::arg("enabled"), nb::arg("scalar") = "", nb::arg("t_sat") = 0.0, nb::arg("k_gas") = 0.0,
      nb::arg("k_liquid") = 0.0, nb::arg("r_int") = 0.0,
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
      "resistance (kelvin per mass flux), all in the solver's physical units.");
  cls.def(
      "vof_interface_area", [](S& s) { return s.vofInterfaceArea(); },
      "Total interfacial area of the colour field (an area in the solver's units; cells "
      "squared without an extent), summed over "
      "the inner region and globally reduced under MPI. Uses the geometry "
      "diagnostics.set_phase_change_area selects, so the number a page quotes and the number the "
      "phase "
      "change integrates are the same one. Needs enable_vof; phase change need not be on. "
      "A sphere of radius R reads 4 pi R^2 to about 0.2-0.8 % once its colour field is "
      "resolved (WO-P3c).");
  cls.def(
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
      "set_phase_change_thermal.");
  cls.def(
      "set_divergence_source",
      [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setDivergenceSource(grid_in(a)); },
      nb::arg("array"),
      "A PRESCRIBED extra divergence source (1/time; under an extent call set_dt first) on "
      "the inner cells, added to the Poisson RHS "
      "beside the phase-change deposit, so the projection solves div(open u) = S. Registered "
      "as the field 'div_source'. Its use is to make a CLOSED (periodic) domain compatible "
      "with a net vapour production by carrying a balancing sink; with an outflow face the "
      "outflow does that job and this is unnecessary.");
  cls.def(
      "clear_divergence_source", [](S& s) { s.clearDivergenceSource(); },
      "Drop the prescribed divergence source (the field stays registered).");
  cls.def(
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
      "subject). Registered as the field 'div_sink'.");
  cls.def(
      "clear_divergence_sink", [](S& s) { s.clearDivergenceSink(); },
      "Drop the auto-balanced sink region.");
  cls.def(
      "apply_phase_change", [](S& s, double dt) { s.applyPhaseChange(dt); }, nb::arg("dt"),
      "KINEMATIC phase-change step: build mdot / the PLIC areas / the normals, deposit the "
      "divergence source (census only — nothing is projected) and apply the interface "
      "regression over `dt`. No Navier-Stokes step, no advection. This is the P0a / P1 driver: "
      "an interface regressing under a prescribed or thermally-computed flux, with the "
      "momentum and pressure solves out of the picture. dt is in the caller's time units.");
}

/// `SolverDiagnostics` (and its collocated twin): phase-change options, ledgers and budgets.
template <class Grid, class C>
inline void bindDiagnosticsPhaseChange(C& cls) {
  using D = Diagnostics<Grid>;
  // --- Phase change, Part II rungs P2/P3 (WO-P23) ------------------------------------------
  cls.def(
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
      "False restores the rung P0/P1 behaviour bit-for-bit.");
  cls.def(
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
      "kernel itself from order 1.1 to order 2.0. set(False) is the ablation.");
  cls.def(
      "set_phase_change_area",
      [](D& diag, const std::string& mode) {
        diag.s->setPhaseChangeArea(enum_index(mode, kAreaModes, "set_phase_change_area", "mode"));
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
      "the interfacial cells (their volume is 1e-4 %, their area 6 %).");
  cls.def(
      "phase_change_area", [](D& diag) { return kAreaModes.at(diag.s->phaseChangeArea()); },
      "The set_phase_change_area mode in force, as its name.");
  cls.def(
      "set_phase_change_energy_muscl",
      [](D& diag, bool on) { diag.s->setPhaseChangeEnergyMuscl(on); }, nb::arg("on"),
      "MinMod-limited donor reconstruction of the face temperature in the CONSISTENT energy "
      "flux, instead of the default plain donor-cell (first-order upwind) value. OFF by "
      "default. The consistency is what the geometric flux buys; its first-order upwind "
      "numerical diffusion |u| h (1 - CFL)/2 thickens the thermal boundary layer and therefore "
      "LOWERS the interfacial gradient, which is exactly what mdot is — measured on the P3 "
      "Scriven bubble at Ja = 0.5: -2.24 % on R(t) with plain upwind against -1.46 % for the "
      "scalar module's Koren TVD (which is NOT consistent). This buys the accuracy back; it is "
      "a switch and not a default because it is the energy twin of set_vof_momentum_muscl.");
  cls.def(
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
      "x cell volume), source_sum and band_div 1/time; colours and temperatures as given.");
  cls.def(
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
      "can be measured against a known geometry; it is not a curvature estimator.");
  cls.def(
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
      "destruction, so repairing any ONE alone makes the gate worse.");
  cls.def(
      "set_phase_change_deposit_fallback",
      [](D& diag, bool on) { diag.s->setPhaseChangeDepositFallback(on); }, nb::arg("on") = true,
      "WO-P3f open item 6 / WO-P3g (OFF by default): give "
      "an interfacial cell whose two along-the-normal deposit candidates are BOTH still "
      "interfacial a target from the 5^3 box instead of leaving the divergence source in "
      "place. A cell that keeps its source carries div(open u) = S on its own faces, so "
      "Weymouth-Yue advects the colour with a field that is not the liquid velocity -- read it "
      "out with `phase_change_diagnostics()['band_div']` and the 'fallback_cells' count.");
  cls.def(
      "set_phase_change_mdot_operator",
      [](D& diag, bool on) { diag.s->setPhaseChangeMdotOperator(on); }, nb::arg("on") = true,
      "WO-P3g item 1 (default OFF): take mdot from the energy operator's own interfacial flux "
      "-- the sum of the ghost-fluid rows' Dirichlet couplings evaluated with the converged T "
      "-- instead of the one-sided least-squares fit, which stays as "
      "`phase_change_diagnostics()['mdot_fit']`. Under a physical domain this needs "
      "set_phase_change_energy: the constant-diffusivity flux carries no heat capacity, so "
      "the step raises rather than assume one.");
  cls.def(
      "set_phase_change_gfm_order", [](D& diag, int o) { diag.s->setPhaseChangeGfmOrder(o); },
      nb::arg("order"),
      "WO-P3g item 2 (default 1): the order of the one-sided (ghost-fluid) Dirichlet row. "
      "1 = the shipped two-point form k o (T_Gamma - T_i)/theta; 2 = Gibou-Fedkiw's "
      "three-point form through (T_behind, T_i, T_Gamma), which reproduces a quadratic "
      "temperature profile exactly at every theta.");
  cls.def(
      "set_phase_change_curvature_distance",
      [](D& diag, bool on) { diag.s->setPhaseChangeCurvatureDistance(on); }, nb::arg("on") = true,
      "WO-P3g item 3 (default OFF): measure the GFM row's theta and the one-sided fits' sample "
      "distances to the CURVED interface, with the mean curvature taken PER CELL from the V3 "
      "curvature cascade (the same kappa surface tension uses; positive for a convex blob of "
      "liquid, i.e. -2/R for a gas bubble). Supersedes `set_phase_change_fit_curvature`, which "
      "prescribes one curvature for the whole field; where both are set the cascade wins.");
  cls.def(
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
      "liquid on an evaporating bubble, the superheated vapour on the Stefan problem.");
  cls.def(
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
      "phase_change_budget's enthalpies.");
  cls.def(
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
      "carrying `pcCarriedValue`; neither transfer appears in the latent-heat book-keeping.");
  cls.def(
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
      "`sum rho c_p (T^{n+1} - T*)/dt = q_gfm + (domain boundary flux) + (solve residual)`.");
}

}  // namespace peclet::flow::bindings
