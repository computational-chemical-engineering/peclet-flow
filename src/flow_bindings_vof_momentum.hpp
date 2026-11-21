/// @file
/// @brief Python binding of peclet.flow: the VoF momentum coupling, interface forces and property
/// closures.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): VoF momentum and property closures, cell force and drag.
template <class Grid, class C>
inline void bindSolverVofMomentum(C& cls) {
  using S = BoundSolver<Grid>;
  // --- Momentum-consistent VoF transport (rung V2b, WO-K) ----------------------------------
  cls.def(
      "enable_vof_momentum", [](S& s, double rg, double rl) { s.enableVofMomentum(rg, rl); },
      nb::arg("rho_c0"), nb::arg("rho_c1"),
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
      "rho_c0 and rho_c1 are in the caller's density units.");
  cls.def(
      "enable_vof_momentum", [](S& s, double rg, double rl) { s.enableVofMomentum(rg, rl); },
      nb::arg("rho_gas"), nb::arg("rho_liquid"),
      "Former keyword names of enable_vof_momentum(rho_c0, rho_c1), the canonical spelling: "
      "rho_gas = rho_c0 (density at C = 0), rho_liquid = rho_c1 (density at C = 1), whichever "
      "phase each is. Deprecated in the next release, removed at 2.0.0.");
  cls.def(
      "vof_momentum_enabled", [](S& s) { return s.vofMomentumEnabled(); },
      "Whether momentum-consistent transport is on.");
  // --- Property closures + Boussinesq body force -------------------------------------------
  cls.def(
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
      "any other target takes its field's INTERNAL units (see unit_scales).");
  cls.def("set_property_table", &chk::setPropertyTable<S>, nb::arg("target"), nb::arg("field"),
          nb::arg("x"), nb::arg("y"),
          "Register a tabulated property: target = piecewise-linear interpolation of (x, y) at the "
          "input field value (x ascending, clamped at the ends). Units as set_property_model: y in "
          "the target's physical units on 'rho'/'mu'/'force_*', x in the input field's.");
  cls.def(
      "update_properties", [](S& s) { s.updateProperties(); },
      "Apply all registered property/force closures now (also done at the top of step()).");
  cls.def(
      "enable_cell_force", [](S& s) { s.enableCellForce(); },
      "Allocate + register the per-cell body-force fields force_x/force_y/force_z and route "
      "them "
      "into the momentum RHS, for an external writer (e.g. CFD-DEM drag feedback) to fill "
      "directly via diagnostics.field_view('force_z'). They persist across steps until "
      "overwritten. "
      "They are raw registry fields: a force per unit volume in INTERNAL units (multiply a "
      "physical one by unit_scales['force_density_to_internal']).");
  cls.def(
      "enable_drag", [](S& s) { s.enableDrag(); },
      "Enable implicit (semi-implicit) linear drag for CFD-DEM: allocate the per-cell "
      "'drag_beta' "
      "field (added to the momentum diagonal so a -beta*(u-u_p) source is treated implicitly "
      "-> "
      "unconditionally stable for the stiff beta of a dense bed) plus force_x/y/z (which carry "
      "beta*u_p, the RHS target). Fill 'drag_beta' and 'force_*' via diagnostics.field_view each "
      "step.");
}

/// `SolverDiagnostics` (and its collocated twin): CSF mode, interface width, curvature clips, face
/// density and the VoF momentum scheme.
template <class Grid, class C>
inline void bindDiagnosticsVofForces(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
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
      "measured number (see the vof_surface_tension ctest), not an argument.");
  cls.def(
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
      "tension keeps the rung-V3 predicate (0 < C < 1) unchanged.");
  cls.def(
      "vof_interface_eps", [](D& diag) { return diag.s->vofInterfaceEps(); },
      "The wisp threshold set by set_vof_interface_eps.");
  cls.def(
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
      "a resolved interface is bit-identical; vof_block_curvature_stats() reports 'clipped'.");
  cls.def(
      "set_vof_kappa_frozen", [](D& diag, bool on) { diag.s->setVofKappaFrozen(on); },
      nb::arg("on") = true,
      "INSTRUMENT: stop recomputing the curvature at the head of each step and use whatever is "
      "in the 'kappa'/'kappa_branch' fields. With set_vof_kappa_constant this isolates the "
      "balanced-force identity from the curvature estimator.");
  cls.def(
      "set_vof_kappa_constant", [](D& diag, double k) { diag.s->setVofKappaConstant(k); },
      nb::arg("kappa"),
      "INSTRUMENT: set kappa to a constant everywhere (inner + ghosts), mark every cell as "
      "carrying a valid estimate, and freeze it. The CSF force is then EXACTLY the discrete "
      "gradient of sigma*kappa*C, so the projection must annihilate it to round-off from any "
      "colour field whatsoever — the exactness gate of rung V4, independent of curvature "
      "accuracy and of resolution.");
  cls.def(
      "set_rho_face_harmonic", [](D& diag, bool on) { diag.s->setRhoFaceHarmonic(on); },
      nb::arg("on") = true,
      "Use the HARMONIC instead of the arithmetic face mean of rho in the pressure projection "
      "(both the operator coefficient and the velocity correction, so the projection stays "
      "exact). DEFAULT OFF, and it should stay off: the arithmetic mean of rho IS the harmonic "
      "mean of the mobility 1/rho — the series-correct choice for a normal flux — and it is "
      "what makes the discrete hydrostatic balance EXACT, because the momentum time term and "
      "the face body force interpolate rho arithmetically and are not switched by this flag. "
      "Shipped as a measured knob for the coefficient-coarsening question (VOF_PLAN S3), not "
      "as an alternative scheme.");
  cls.def(
      "set_vof_rho_floor", [](D& diag, double f) { diag.s->setVofRhoFloorFrac(f); },
      nb::arg("fraction"),
      "Floor on rho^c in the recovery divide u = (rho^c u)/rho^c, as a FRACTION of "
      "min(rho_c0, rho_c1). Default 1e-6. rho^c leaves [rho_c0, rho_c1] only through "
      "a wisp in the half-shifted colour and reaching zero would need C^c ~ -1/(ratio-1), so "
      "this is a guard, not a model — vof_momentum_diagnostics()['floored'] reports how many "
      "control volumes it actually touched.");
  cls.def(
      "vof_rho_floor", [](D& diag) { return diag.s->vofRhoFloor(); },
      "The absolute rho^c floor used by the last recovery.");
  cls.def(
      "set_vof_momentum_scheme",
      [](D& diag, const std::string& name) { diag.s->setVofMomentumScheme(name); }, nb::arg("name"),
      "The transported face velocity of the VoF momentum flux (flow "
      "doc/vof_momentum_second_order.md). 'upwind': donor cell, convex at every density "
      "ratio, first order (the default until 2026-10-31). 'muscl_mixed': a MinMod slope of "
      "the MIXED control-volume velocity; it converges to a WRONG limit at high ratio (P1, "
      "ratio ~1e3: +3.0/+3.1 % at d/h 16/24; the mechanism is open). 'muscl_mixed_capped': "
      "that slope times the mass-aware cap (the diagnosis ablation). 'muscl_phase': each "
      "phase's mass flux carries its own face velocity from an interface-biased MC slope, "
      "scaled by the cap - second order, a discrete maximum principle at any ratio, the "
      "uniform-velocity identity bit-exact; the DEFAULT. vof_momentum_diagnostics()['capped'] "
      "counts the faces the cap scaled in the last step. Any other name raises ValueError.");
  cls.def(
      "vof_momentum_scheme", [](D& diag) { return diag.s->vofMomentumScheme(); },
      "The selected VoF momentum scheme (see set_vof_momentum_scheme).");
}

/// `SolverDiagnostics` (and its collocated twin): VoF momentum options and read-outs, property and
/// density modes.
template <class Grid, class C>
inline void bindDiagnosticsVofMomentum(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "set_vof_momentum_muscl", [](D& diag, bool on) { diag.s->setVofMomentumMuscl(on); },
      nb::arg("on") = true,
      "Former spelling of set_vof_momentum_scheme: on = 'muscl_mixed' (a MinMod slope of the "
      "mixed control-volume velocity), off = 'upwind'. Use set_vof_momentum_scheme; "
      "'muscl_phase' is the second-order scheme that stays bounded at high density ratio.");
  cls.def(
      "set_vof_momentum_cell_flag", [](D& diag, bool on) { diag.s->setVofMomentumCellFlag(on); },
      nb::arg("on") = true,
      "ABLATION: use the PRESSURE-cell frozen dilation flag H(C^n-1/2) on the shifted control "
      "volume instead of its structural analogue H(C^c,n-1/2). The flag that must be shared is "
      "the one of the pair that telescopes, and both members of that pair live on the shifted "
      "volume — this switch is the literal reading of the work order, kept as a measurement.");
  cls.def(
      "set_vof_flux_clamp", [](D& diag, bool on) { diag.s->setVofFluxClamp(on); },
      nb::arg("on") = true,
      "ABLATION: drop the Weymouth flux clamp max(0,|a|-(1-C^c_don)) <= |F| <= "
      "min(|a|,C^c_don) "
      "on the half-shifted control volume. The geometric flux is bounded by what the CURRENT "
      "cell planes see in the donor, not by the ADVECTED C^c; the gap is O(a^2) and at density "
      "ratio 1e4 a 2.6e-2 undershoot drives rho^c to -255, which the recovery would divide by. "
      "Default ON; off is how that statement stays a measured number.");
  cls.def(
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
      "step.");
  cls.def(
      "vof_advected_velocity",
      [](D& diag, int c) { return field_out(*diag.s, diag.s->getVofAdvectedVelocity(c)); },
      nb::arg("component"),
      "The recovered advected velocity (rho^c u_c)/rho^c of component c on the inner cells — "
      "the momentum RHS's time base. Exposed so the uniform-velocity consistency identity can "
      "be gated on the ADVECTION ALONE, with the projection and the momentum solve out of the "
      "picture.");
  cls.def(
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
      "scheme (large-dt / steady-Stokes) stays active — see set_variable_rotational.");
  cls.def(
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
      "term). All modes keep the incremental predictor (large-dt / steady-Stokes capability).");
  cls.def(
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
      "rest at ratio 1000.");
}

}  // namespace peclet::flow::bindings
