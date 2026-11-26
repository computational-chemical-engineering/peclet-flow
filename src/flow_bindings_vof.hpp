/// @file
/// @brief Python binding of peclet.flow: the VoF colour field, its advection, contact angle and
/// wetting.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): VoF colour field, advection, contact angle and wall slip.
template <class Grid, class C>
inline void bindSolverVof(C& cls) {
  using S = BoundSolver<Grid>;
  // --- Geometric VoF colour field (rung V2a) -----------------------------------------------
  cls.def(
      "enable_vof", [](S& s) { s.enableVof(); },
      "Enable the geometric (PLIC + Weymouth-Yue) VoF colour field 'C'. Registers 'C' as an "
      "ordinary cell field (get_field/set_field/diagnostics.field_view/closure input) and "
      "allocates the "
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
      "conservation proof needs), and the CSF enters the implicit momentum predictor as the "
      "mass-adjoint face integral (doc/collocated_varrho_forces.md). An immersed solid on that "
      "grid THROWS "
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
      "diagnostics.vof_diagnostics()['clipped_volume'].");
  cls.def(
      "set_vof", [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setVof(grid_in(a)); },
      nb::arg("array"),
      "Set the colour field from a Fortran-order (nx,ny,nz) float64 array: C = the LIQUID "
      "volume fraction of the cell, in [0,1]. Enables VoF if it is not on yet. Initialise it "
      "SHARP (0/1 with the exact fraction in the interface cells) — geometric VoF keeps a "
      "sharp interface and a diffuse initial profile is simply a different problem.");
  cls.def(
      "get_vof", [](S& s) { return field_out(s, s.getVof()); },
      "The colour field's inner region as a Fortran-order (nx,ny,nz) float64 array "
      "(== get_field('C')).");
  cls.def(
      "vof_max_courant", [](S& s) { return s.vofMaxCourant(); },
      "INTERFACE-LOCAL Courant number max|uf|*dt/h over the faces of mixed cells and their "
      "face neighbours, with the current velocity and dt (all-reduce max under MPI). This — "
      "not the global max — is the number Weymouth's boundedness bound applies to: a global "
      "max over-throttles badly (measured on Zalesak: 0.314 at a quiescent domain corner while "
      "the interface never exceeded 0.157). Pick dt as dt*cfl_target/vof_max_courant().");
  cls.def("set_vof_cfl_limit", &chk::setVofCflLimit<S>, nb::arg("value"),
          "Weymouth-Yue boundedness cap on the interface-local Courant number. Default 0.25 — the "
          "PROVEN 3D bound 1/(2(N-1)) from Weymouth's thesis eq. A.33; the widely-quoted 0.5 is "
          "the 2D value. step() throws when it is exceeded. Conservation is INDEPENDENT of "
          "boundedness (the telescoping is algebraic), so an over-CFL run loses 0<=C<=1, not "
          "volume.");
  cls.def(
      "vof_cfl_limit", [](S& s) { return s.vofCflLimit(); },
      "The Weymouth-Yue boundedness cap currently in force.");
  // --- rung V5a (WO-Q): transport through an immersed solid --------------------------------
  cls.def(
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
      "or set_solid(sdf, cutcell_pressure=True).");
  cls.def(
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
      "point of the scheme.");
  cls.def(
      "set_contact_angle_field",
      [](S& s, nb::ndarray<double, nb::f_contig> a) { s.setContactAngleField(grid_in(a)); },
      nb::arg("theta"),
      "Per-cell static contact angle in DEGREES, as a Fortran-order (nx,ny,nz) array. Only the "
      "value AT the solid band cell being filled is read, so cells away from a wall are "
      "irrelevant. theta is a field so the dynamic-angle rung (V6) changes only what fills it.");
  cls.def(
      "contact_angle", [](S& s) { return s.contactAngle(); },
      "The prescribed static contact angle in degrees (90 if none was set).");
  cls.def(
      "set_contact_angle_dynamic",
      [](S& s, bool enabled, double th, double slip, double mu, double sigma) {
        if (!enabled) {
          s.setContactAngleDynamicOff();
          return;
        }
        chk::finite("set_contact_angle_dynamic", "sigma", sigma, "force/length");
        if (!(slip > 0.0 && slip < 1.0) || !(mu > 0.0))
          throw std::invalid_argument(
              "set_contact_angle_dynamic(True, ...): slip_length_cells must lie in (0, 1) and "
              "mu_liquid must be positive");
        s.setContactAngleDynamic(th, slip, mu, sigma);
      },
      nb::arg("enabled"), nb::arg("theta_e") = 90.0, nb::arg("slip_length_cells") = 0.0,
      nb::arg("mu_liquid") = 0.0, nb::arg("sigma") = 0.0,
      "Rung V6 (WO-V6). The DYNAMIC contact angle -- `enabled=False` turns it (and the "
      "hysteresis) off so the static angle stands again; `enabled=True` imposes at the grid "
      "scale the Cox-Voinov apparent angle of a contact line moving at speed U_cl, with the "
      "outer cut-off at the CELL SIZE and an EXPLICIT slip length lambda (Afkhami, Zaleski & "
      "Bussmann, JCP 228:5370, 2009):\n\n"
      "    theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta/lambda),   Ca_cl = mu_l U_cl / sigma\n\n"
      "angles in radians internally, arguments in DEGREES; Ca_cl > 0 advancing, < 0 receding; "
      "theta_Delta is clamped into [1, 179] degrees (diagnostics.set_contact_angle_clamp).\n\n"
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
      "enable_vof; with no call every V5b number is byte-identical.");
  cls.def(
      "set_contact_angle_hysteresis",
      [](S& s, double a, double r) { s.setContactAngleHysteresis(a, r); }, nb::arg("theta_a"),
      nb::arg("theta_r"),
      "Rung V6 (WO-V6). Advancing / receding contact-angle hysteresis, in DEGREES. While the "
      "measured apparent angle lies in [theta_r, theta_a] the contact line is PINNED and the "
      "fill imposes the APPARENT angle itself — which reproduces the current interface exactly, "
      "because the V5b fill is idempotent (WO-S finding 1), so no Young force appears and the "
      "line does not move. Above theta_a the advancing angle is imposed and below theta_r the "
      "receding one, each with the Cox-Voinov correction of set_contact_angle_dynamic when "
      "that is also configured. Sets the static base to (theta_a+theta_r)/2 if no angle was "
      "set yet.");
  cls.def(
      "set_wall_slip_length", [](S& s, double lam) { s.setWallSlipLength(lam); },
      nb::arg("lambda_cells"),
      "Rung V6b (WO-V6b) - the VELOCITY half of the dynamic contact line. In the Robust-Scaled "
      "cut-cell IBM closure the TANGENTIAL wall datum stops being no-slip and becomes the "
      "NAVIER condition\n\n"
      "    u_t(wall) - u_body = lambda * d(u_t)/dn\n\n"
      "with `lambda_cells` the slip length in the caller's own units (CELLS without a physical "
      "domain, where the cell size is 1); the model needs lambda < one cell. "
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
      "diagnostics.wall_slip_sandwich_cells().");
  cls.def(
      "wall_slip_length", [](S& s) { return s.wallSlipLength(); },
      "The Navier slip length in force, in the caller's units (0 = no-slip).");
  cls.def(
      "vof_has_geometry", [](S& s) { return s.vofHasGeometry(); },
      "True when the colour advection is running the CUT-CELL (openness-weighted) kernels, "
      "i.e. an immersed solid is present and set_solid ran with cutcell_pressure=True. False "
      "means "
      "the uncut rung-V1 kernels are running, byte-identically to a solid-free build.");
}

/// `SolverDiagnostics` (and its collocated twin): VoF step controls, geometry, contact angle,
/// timing and worklists.
template <class Grid, class C>
inline void bindDiagnosticsVof(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "vof_last_courant", [](D& diag) { return diag.s->vofLastCourant(); },
      "The interface-local Courant number of the step just taken (0 before the first step).");
  cls.def(
      "set_vof_step_parity", [](D& diag, long n) { diag.s->setVofStepParity(n); }, nb::arg("n"),
      "Set the sweep-permutation counter of the NEXT colour advection: the Weymouth-Yue sweep "
      "order is kWySweepPerm[n % 6], cycled so no axis is systematically favoured. Exposed so "
      "a benchmark can hold the permutation fixed (n constant) or resume a run across a "
      "restart. Default: it increments once per advection from 0.");
  cls.def(
      "vof_step_parity", [](D& diag) { return diag.s->vofStepParity(); },
      "The sweep-permutation counter of the next colour advection.");
  cls.def(
      "set_vof_wisp_eps", [](D& diag, double eps) { diag.s->setVofWispEps(eps); }, nb::arg("eps"),
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
      "0.3110 by step 1000 on a case whose interface never exceeds 0.255.");
  cls.def(
      "vof_wisp_eps", [](D& diag) { return diag.s->vofWispEps(); },
      "The wisp tolerance currently in force (see set_vof_wisp_eps).");
  cls.def(
      "set_pressure_exact_residual", [](D& diag, bool on) { diag.s->setPressureExactResidual(on); },
      nb::arg("on") = true,
      "Apply the level-0 pressure operator EXACTLY (matrix-free, double, flux form) in the "
      "residual and the Krylov matvec instead of reading the float band storage. P1 of the "
      "suite defect-correction campaign (suite/docs/archive/DEFECT_CORRECTION_PLAN.md). Per "
      "solver, OFF at construction.\n\n"
      "enable_vof() turns it ON, because a two-phase coefficient contrast is exactly what "
      "amplifies the float operator's broken row-sum identity A*1 = 0. Measured on Hysing "
      "case 2 (64x128x4, adaptive dt, nvidia-cuda): max|div(open u)| 1.85e-03 -> 5.15e-11, "
      "with 116/600 pressure iterations, 1123 steps, the dt-limit census and both published "
      "functionals (v_rise max 0.2574 at t = 0.671, y_c(3) 1.1082) identical to every printed "
      "digit. Everything below level 0 stays float on purpose: it is a preconditioner and its "
      "errors change the convergence RATE, never the fixed point. A flag read at every solve, "
      "so it may be changed at any time; the ablation is False AFTER enable_vof.");
  cls.def(
      "pressure_exact_residual", [](D& diag) { return diag.s->pressureExactResidual(); },
      "Whether the exact level-0 operator apply is in force (see set_pressure_exact_residual).");
  cls.def(
      "set_vof_cutcell_flux_clamp", [](D& diag, bool on) { diag.s->setVofCutFluxClamp(on); },
      nb::arg("on"),
      "Ablation: Weymouth's admissible-interval clamp on the openness-weighted cut-cell flux "
      "(ON by default). The clamp bounds |F| by what the DONOR actually holds — at most "
      "eps*C liquid and at most eps*(1-C) gas of the fluid volume o*|a| swept through the face "
      "— applied to the one value both neighbours share, so conservation still telescopes "
      "bit-exactly. It is what makes the whole-cell-PLIC-times-open-area flux approximation "
      "BOUNDED. Measured with it off on a 24^3 packing at CFL 0.2: the [0,1] clip fires at "
      "3.2e-5 liquid volume per step and the conserved functional drifts 1.3e-8 in 30 steps; "
      "with it on the clip stops firing.");
  cls.def(
      "vof_filled_color", [](D& diag) { return field_out(*diag.s, diag.s->getVofFilledColour()); },
      "The colour field INCLUDING the neutral solid-band fill — what the MYC and "
      "height-function stencils actually read — as a Fortran-order (nx,ny,nz) array. "
      "get_vof()/'C' is the canonical field and carries EXACTLY 0 in solid cells; the fill is "
      "a stencil device regenerated at every ghost exchange (three passes with a shrinking "
      "depth "
      "budget, src/vof/cutcell.hpp), and it is what makes the wall look 90-degree neutral "
      "instead of perfectly non-wetting.");
  cls.def(
      "vof_geometry",
      [](D& diag, int which) { return field_out(*diag.s, diag.s->getVofGeometry(which)); },
      nb::arg("which"),
      "The cut-cell geometry the colour block runs on, as a Fortran-order (nx,ny,nz) array: "
      "0 = the cell fluid fraction eps (4^3-subsampled, a multiple of 1/64), 1/2/3 = the "
      "openness of the +x/+y/+z face of each cell (the ADVECTOR's high-face convention, one "
      "cell shifted from the solver's ox/oy/oz), 4 = the classification (1 = SOLID, i.e. "
      "eps == 0 AND all six faces closed).\n\n"
      "On an ALL-FLUID solver (set_pressure_geometry, or no cut-cell pressure operator) there "
      "is no geometry block and this returns the TRIVIAL geometry the V1 transport kernels "
      "actually execute: eps = 1, openness = 1, classification = 0. It used to raise, which "
      "meant one diagnostic could not serve a packed scene and its all-fluid control.");
  cls.def(
      "set_contact_angle_pivot",
      [](D& diag, const std::string& mode) {
        diag.s->setContactAnglePivot(enum_index(mode, kPivots, "set_contact_angle_pivot", "mode"));
      },
      nb::arg("mode"),
      "ABLATION -- how the theta-plane is anchored in the fluid cell. 'volume' (DEFAULT) match "
      "the anchor cell's liquid volume with plicAlpha; 'centroid' pass through the PLIC "
      "centroid p_f (the Afkhami-Bussmann / Basilisk contact.h rule); 'projected-centroid' the "
      "work order's c = p_f - sdf(p_f) n_w; 'contact-line' the contact line on the wall. "
      "'volume', 'centroid' and 'contact-line' are exactly idempotent (1e-15 on gate G0a); "
      "'projected-centroid' is NOT -- projecting the centroid along n_w shifts the plane by "
      "-sdf(p_f) cos(theta), measured 0.26 in cell fraction at theta = 60, with the wrong sign "
      "(it removes liquid from the band for a wetting angle).");
  cls.def(
      "contact_angle_diagnostics",
      [](D& diag) {
        const auto d = diag.s->contactAngleDiagnostics();
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
      "The solid-band census of the CURRENT colour field on this rank: how many band cells "
      "each "
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
      "velocity units.");
  cls.def(
      "wall_slip_sandwich_cells",
      [](D& diag) {
        auto a = diag.s->wallSlipSandwichCells();
        return nb::make_tuple(a[0], a[1], a[2]);
      },
      "Per velocity component, the number of cut-cell AXES at which a one-cell fluid gap made "
      "the Navier closure inapplicable and the no-slip one was kept. Nonzero means part of the "
      "wall is silently no-slip; report it.");
  cls.def(
      "set_contact_angle_smoothing", [](D& diag, bool on) { diag.s->setContactAngleSmoothing(on); },
      nb::arg("on"),
      "ABLATION - the 3-point in-wall mean of U_cl (default ON). A MAC velocity next to a wall "
      "is noisy cell to cell and the cube root of the Cox-Voinov relation puts that noise "
      "straight into the imposed angle; off measures what the smoothing is worth.");
  cls.def(
      "set_contact_angle_clamp",
      [](D& diag, double lo, double hi) { diag.s->setContactAngleClamp(lo, hi); }, nb::arg("lo"),
      nb::arg("hi"),
      "The clamp on the Cox-Voinov cube, in degrees (default 1 / 179). The cubic has no "
      "solution beyond the maximum receding capillary number (film entrainment) and the fill's "
      "plane construction degenerates at 0/180.");
  cls.def(
      "vof_dynamic_field",
      [](D& diag, int w) { return field_out(*diag.s, diag.s->getVofDynamicField(w)); },
      nb::arg("which"),
      "The V6 per-cell dynamic-wetting state on the inner region, as (nx,ny,nz): 0 the IMPOSED "
      "angle in degrees, 1 the measured APPARENT angle in degrees, 2 the smoothed U_cl "
      "(positive = the liquid ADVANCES), 3 Ca_cl, 4 the state (0 not a contact cell, 1 "
      "Cox-Voinov on the static base, 2 PINNED, 3 advancing, 4 receding). Regenerates the "
      "fill, "
      "so it is also the direct gate on the pass's decomposition independence.");
  cls.def(
      "set_vof_timing", [](D& diag, bool on) { diag.s->setVofTiming(on); }, nb::arg("on"),
      "WO-V9: arm (or disarm) the VoF pipeline's per-stage timers, and reset them. OFF by "
      "default. When armed, every stage boundary calls Kokkos::fence() before reading the "
      "clock -- the same rule the step's three coarse phase timers already follow, because on "
      "a device backend queued work would otherwise be billed to whichever stage next reads "
      "the clock. A fence changes WHEN work happens, never WHAT it computes: a run with the "
      "timers armed is bit-identical to the same run without them (gated in "
      "tests/kokkos/test_vof_timing.cpp). When disarmed the cost is one branch per stage and "
      "no fence at all.");
  cls.def(
      "reset_vof_timing", [](D& diag) { diag.s->resetVofTiming(); },
      "Zero the VoF stage timers and the step counter without disarming them.");
  cls.def(
      "vof_timing",
      [](D& diag) {
        const auto& v = diag.s->vofTimingReport();
        const auto& k = diag.s->vofKernelTiming();
        nb::dict r;
        r["steps"] = v.steps;
        // the step's three coarse phases, summed over the same window (seconds)
        r["step"] = diag.s->vofTimingStepSeconds();
        r["predictor"] = diag.s->vofTimingPredictorSeconds();
        r["momentum_solve"] = diag.s->vofTimingMomentumSeconds();
        r["projection"] = diag.s->vofTimingProjectionSeconds();
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
        const auto& q = diag.s->vofCurvatureTiming();
        r["kc_calls"] = q.calls;
        r["kc_compact"] = q.compact;
        r["kc_planes"] = q.planes;
        r["kc_height"] = q.height;
        r["kc_fallback"] = q.fallback;
        r["kc_clip"] = q.clip;
        r["block_advect"] = v.blockAdvect;
        r["block_debris"] = v.blockDebris;
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
      "zeros unless the timers are armed.");
  cls.def(
      "set_vof_worklist", [](D& diag, bool on) { diag.s->setVofWorklist(on); }, nb::arg("on"),
      "WyAdvector::useWorklist -- compact the PLIC reconstruction pass onto the mixed cells "
      "(a parallel_scan over the reconstruction region, then a dense parallel_for over the "
      "compacted list) instead of a guarded parallel_for over the whole region. Default ON. "
      "Pure optimization: the flux never reads a non-mixed cell's plane, so the two paths "
      "produce the same field BIT FOR BIT (gated in tests/kokkos/test_vof_advect.cpp and "
      "measured in WO-V9).");
  cls.def(
      "vof_worklist", [](D& diag) { return diag.s->vofWorklist(); },
      "Whether the reconstruction-pass compaction is on.");
  cls.def(
      "set_vof_curvature_worklist", [](D& diag, bool on) { diag.s->setVofCurvatureWorklist(on); },
      nb::arg("on"),
      "WO-V9: run the V3 curvature cascade (the PLIC plane pass, the height functions and the "
      "PV fallback) over a COMPACTED list of the interfacial cells instead of over the whole "
      "inner region. Default ON. The cascade is the most divergent kernel in the pipeline -- "
      "on a resolved droplet the interfacial cells are well under 1 % of the block, so a dense "
      "parallel_for puts one or two active lanes in a warp and the other thirty run the guard "
      "and then idle for the whole height function. Compaction is a pure re-ordering (each "
      "cell reads the same neighbours and writes the same value), so the two paths are "
      "BIT-IDENTICAL; tests/kokkos/test_vof_timing.cpp gates that and WO-V9 records the gain.");
  cls.def(
      "vof_curvature_worklist", [](D& diag) { return diag.s->vofCurvatureWorklist(); },
      "Whether the curvature-cascade compaction is on.");
}

}  // namespace peclet::flow::bindings
