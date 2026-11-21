/// @file
/// @brief Python binding of peclet.flow: construction and domain read-outs, solver controls, state
/// and stepping, and the porous / MPI tail of both tiers.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): the diagnostics tier, constructors, domain read-outs,
/// density and viscosity.
template <class Grid, class C>
inline void bindSolverSetup(C& cls) {
  using S = BoundSolver<Grid>;
  using D = Diagnostics<Grid>;
  cls.def_prop_ro(
      "diagnostics", [](S& s) { return D{&s}; }, nb::keep_alive<0, 1>(),
      "The developer tier of this solver (instruments, tuning, ablations, internals).");
  cls.def("__init__", &chk::initCells<S>, nb::arg("nx"), nb::arg("ny"), nb::arg("nz"),
          "Create a solver on an nx x ny x nz grid of UNIT cells (x-fastest, I = x + y*nx + "
          "z*nx*ny). Everything is then in cell units: the cell size is 1 and lengths, "
          "velocities and the SDF are measured in cells. Prefer the physical form "
          "Solver((nx,ny,nz), extent=(Lx,Ly,Lz)). Set the domain BCs before the geometry (a later "
          "set_domain_bc raises).");
  cls.def("__init__", &chk::initPhysical<S>, nb::arg("cells"), nb::arg("extent") = nb::none(),
          nb::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
          nb::arg("global_cells") = nb::none(),
          "Create a solver on a PHYSICAL domain: `cells` = (nx, ny, nz), `extent` = the box side "
          "lengths (Lx, Ly, Lz) and `origin` = its lower corner, all in the caller's own "
          "consistent units. The cell size is extent/cells and is never written by the user: "
          "rho, mu, dt, body forces, boundary velocities, surface tension, the SDF handed to "
          "set_solid and the coordinates of set_scene are all physical, and the velocity, "
          "pressure and curvature that come back are physical too.\n\n"
          "extent=None (the default) keeps the historical CELL-UNIT behaviour, bit for bit: the "
          "spacing is 1 and every quantity is in cells.\n\n"
          "ANISOTROPIC cells (extent/cells different per axis) are supported since Phase 2 on the "
          "staggered Solver AND (since commit C4) on SolverColocated: sampled or scene geometry, "
          "constant or variable properties, variable density, porous continuity with or without "
          "implicit drag, every domain-BC type, every pressure driver and bottom, the velocity "
          "multigrid, scalar transport, the hydrodynamic force and torque integrals, and MPI. "
          "moving geometry included. What still REFUSES one, with a message naming the three "
          "spacings and what lifts it: enable_vof (Phase 3). "
          "Three spacings that agree to 1e-12 relative are SNAPPED "
          "to one, so an isotropic domain stays bit-identical however its extent was written. "
          "Under MPI `cells` is this rank's block (from "
          "mpi_block) while `extent`/`origin` describe the GLOBAL domain, so pass the global "
          "grid as `global_cells` - the same numbers init_mpi gets.");
  cls.def_prop_ro(
      "cells", [](S& s) { return std::array<long, 3>{s.nx(), s.ny(), s.nz()}; },
      "This rank's inner block resolution (nx, ny, nz) -- the shape of every field array "
      "(get_u, get_p, the SDF for set_solid). Under MPI it is the LOCAL block from mpi_block().");
  cls.def_prop_ro(
      "global_cells", [](S& s) { return s.globalCells(); },
      "The GLOBAL grid resolution (gnx, gny, gnz); == `cells` single-rank, the numbers "
      "init_mpi gets under MPI (the CFD-DEM co-decomposition weight field is this shape).");
  cls.def_prop_ro(
      "extent", [](S& s) { return s.domainExtent(); },
      "The GLOBAL domain side lengths. Equal to the cell counts without a physical domain.");
  cls.def_prop_ro(
      "origin", [](S& s) { return s.domainOrigin(); },
      "The physical lower corner of the GLOBAL domain (0,0,0 by default).");
  cls.def_prop_ro(
      "spacing", [](S& s) { return s.spacing(); },
      "The cell size [dx, dy, dz] = extent/cells as three floats; [1.0, 1.0, 1.0] "
      "without a physical domain (the cell-unit default). Three spacings within 1e-12 "
      "relative are snapped to one value, so an isotropic box is exactly isotropic.");
  cls.def_prop_ro(
      "unit_scales",
      [](S& s) {
        const auto& u = s.unitScales();
        nb::dict d;
        d["identity"] = !u.physical;
        d["h_ref"] = u.hRef;
        d["rho_ref"] = u.rhoRef;
        d["t_ref"] = u.tRef;
        d["length_to_internal"] = u.lenToInt();
        d["velocity_to_internal"] =
            std::vector<double>{u.velToInt(0), u.velToInt(1), u.velToInt(2)};
        d["time_to_internal"] = u.timeToInt();
        d["density_to_internal"] = u.rhoToInt();
        d["viscosity_to_internal"] = u.muToInt();
        d["pressure_to_internal"] = u.pToInt();
        d["force_density_to_internal"] =
            std::vector<double>{u.forceToInt(0), u.forceToInt(1), u.forceToInt(2)};
        d["force_to_physical"] = u.forceTotalToPhys();
        d["torque_to_physical"] = u.torqueToPhys();
        // scalar transport + the phase-change energy path (a transported scalar itself, a
        // temperature or a concentration, is never rescaled)
        d["diffusivity_to_internal"] = u.diffToInt();
        d["mass_flux_to_internal"] = u.mdotToInt();
        d["latent_heat_to_internal"] = u.latentToInt();
        d["conductivity_to_internal"] = u.condToInt();
        d["heat_capacity_to_internal"] = u.rcpToInt();  // volumetric, rho*c_p
        d["interface_resistance_to_internal"] = u.rintToInt();
        d["divergence_to_internal"] = u.divToInt();
        return d;
      },
      "The factors between the caller's units and the ones the solver computes in.\n\n"
      "The solver keeps computing on the UNIT LATTICE and folds the metric into constants at "
      "its API boundary, so every array reached through the RAW field registry "
      "(diagnostics.field_view / get_field / set_field) is in those INTERNAL units, unlike "
      "get_u/get_p, "
      "which convert. `identity` is True when no extent was given, and then every factor is "
      "exactly 1.0 and internal == physical. Multiply a physical value by *_to_internal, and "
      "an internal one by force_to_physical (a total force) to come back. The registered "
      "'mdot' field is in mass_flux_to_internal units and 'pc_source' / 'div_source' in "
      "divergence_to_internal units; a transported scalar is never rescaled.");
  cls.def_prop_ro(
      "physical_units", [](S& s) { return s.hasPhysicalDomain(); },
      "True when the solver was given an extent, i.e. lengths are physical rather than cells.");
  cls.def(
      "cell_centers",
      [](S& s) { return nb::make_tuple(s.cellCentres(0), s.cellCentres(1), s.cellCentres(2)); },
      "The physical cell-center coordinates of THIS rank's inner block as three 1-D float64 "
      "arrays (x, y, z), i.e. origin + (i + 1/2) * spacing per axis. "
      "np.meshgrid(*s.cell_centers(), indexing='ij') is the grid an SDF for set_solid is "
      "sampled on. Under MPI these are the LOCAL block's centers in GLOBAL coordinates. "
      "CELL-UNIT TRAP: without an extent this is i + 1/2, while set_scene puts cell i's centre "
      "at i -- the same body given as a scene and as an SDF sampled here lands half a cell "
      "apart. With an extent both use origin + (i + 1/2) * spacing. Unifying the cell-unit "
      "convention changes results; deferred to 2.0.0.");
  cls.def("set_rho", &chk::setRho<S>, nb::arg("rho"),
          "Set the fluid density rho (the caller's units). Under a physical domain the FIRST call "
          "pins the reference density the internal scales are built on; a later change rebuilds "
          "the momentum operator at the next step.");
  cls.def("set_mu", &chk::setMu<S>, nb::arg("mu"),
          "Set the dynamic viscosity mu (the caller's units); a change after the geometry "
          "rebuilds the momentum operator at the next step.");
  cls.def("set_viscous_stress", &S::setViscousStress, nb::arg("form"),
          "Viscous stress form under variable viscosity: 'auto' (default), 'full' or "
          "'laplacian'. 'full' = div(mu (grad u + grad u^T)), the transpose term explicit on u^n "
          "as conservative MAC stress elements (doc/full_viscous_stress.md); 'laplacian' = "
          "div(mu grad u), the form before the AUTO default (it reproduces earlier variable-mu "
          "numbers). The Laplacian form imposes the wrong tangential stress condition at a "
          "viscosity jump (an "
          "h-independent interface force: +19.5 % rise velocity for a drop at viscosity ratio "
          "0.1). 'auto' resolves at each step to 'full' on the staggered Solver with the "
          "arithmetic face viscosity, no porous momentum, no moving scene instances and no "
          "immersed solid, else to 'laplacian' with a one-time stderr notice. Inert for a "
          "constant mu. An explicit 'full' is staggered-only (SolverColocated raises) and is "
          "refused at step() with the harmonic face viscosity, porous momentum or moving scene "
          "instances. With solids it stays allowed, but it drops the transpose flux across the "
          "staircase face: an O(gamma h) near-wall error on oblique walls (gamma the wall shear "
          "rate), present for any non-uniform mu; the repair is open. "
          "The form is a name, not a physical quantity.");
}

/// `Solver` (and its collocated twin): time step, body force, advection, decomposition and the
/// pressure / velocity solver controls.
template <class Grid, class C>
inline void bindSolverControls(C& cls) {
  using S = BoundSolver<Grid>;
  cls.def("set_dt", &chk::setDt<S>, nb::arg("dt"),
          "Set the time step dt (the caller's time unit); the momentum solve is scaled by 1/dt "
          "(well-conditioned at large dt). Under a physical domain the FIRST call pins the "
          "reference time the internal scales are built on; any later call (step_adaptive "
          "makes them) rebuilds the momentum operator at the next step.");
  cls.def("set_body_force", &chk::setBodyForce<S>, nb::arg("force"),
          "Set the body force per unit volume (fx, fy, fz) as one 3-sequence — e.g. a mean "
          "pressure gradient.");
  cls.def(
      "set_superficial_velocity",
      [](S& s, bool enabled, const std::string& axis, double velocity) {
        chk::finite("set_superficial_velocity", "velocity", velocity, "length/time");
        s.setSuperficialVelocity(
            enabled, enum_index(axis, kAxes, "set_superficial_velocity", "axis"), velocity);
      },
      nb::arg("enabled"), nb::arg("axis") = "x", nb::arg("velocity") = 0.0,
      "Hold the SUPERFICIAL velocity along a periodic axis at `velocity` (physical units): "
      "the volume mean of that velocity component, i.e. the volume flux of the MIXTURE "
      "through any cross-section divided by the cross-section's full area. In multiphase "
      "flow 'velocity' alone is ambiguous between superficial and interstitial (per-phase); "
      "this is the superficial one. At the end of every step() one uniform shift is "
      "added to that component on every face; on an axis with no boundary a uniform shift "
      "of the normal velocity leaves every cell divergence unchanged, so the field stays "
      "discretely divergence-free. `velocity = 0` is a closed (batch) column: the flow can "
      "circulate but not pass through (TBFsolver's constant flow rate at 0). The mean is a "
      "device reduction, global under MPI; single-rank the step makes no host round trip for "
      "it. Staggered Solver, all-fluid domain (set_pressure_geometry), periodic `axis` only "
      "-- step() raises otherwise. set_superficial_velocity(False) turns it off. "
      "diagnostics.last_superficial_velocity_shift() is the shift the last step applied.");
  cls.def("set_balanced_force_projection", &S::setBalancedForceProjection, nb::arg("enabled"),
          "Balanced-force projection (flow doc/collocated_varrho_forces.md §4.6), on both grids. "
          "Once per step, before the momentum predictor, the gradient part of the forces "
          "(set_body_force, the per-cell force fields, the CSF) is moved into the pressure with "
          "one extra solve of the projection's own operator, so static balances -- a hydrostatic "
          "column, a constant-curvature drop -- are exact from the first step at every "
          "viscosity, dt, density ratio and immersed-solid aperture. The solve does not depend "
          "on the flow state, so stability and the converged steady state are identical on and "
          "off; off leaves a decaying transient balance residue. get_p() stays the total "
          "physical pressure; its balanced part is the registered field 'p_balanced', which is "
          "RESTART STATE: registered as soon as the option is active, restoring it with 'p' "
          "(set_field) keeps the split, and omitting it (or switching the option on after steps "
          "with it off) re-splits P -- never a second copy of the balanced pressure. "
          "Cost: one extra pressure solve per step while the forces change, "
          "solved for the increment of the previous split and skipped when that is already "
          "converged -- a static interface costs no iterations. Default: "
          "ON on SolverColocated with variable density or surface tension, OFF otherwise; with "
          "an inflow/outflow face that default resolves OFF (a one-time stderr notice says "
          "so); an explicit call always wins. Refused (named error) with porous continuity, the "
          "ghost projection, "
          "set_fluid_only_constraint(2), the block CSF, inflow/outflow faces, the "
          "non-incremental pressure, and on the constant-density SolverColocated path (nothing "
          "to balance there).");
  cls.def_prop_ro(
      "balanced_force_projection", [](S& s) { return s.balancedForceProjection(); },
      "Whether the balanced-force projection runs (see "
      "set_balanced_force_projection).");
  cls.def("set_advection", &S::setAdvection, nb::arg("on"),
          "Enable/disable momentum advection (high-order scheme SOU by default; implicit FOU + "
          "deferred correction on the domain-BC path). ON by default since 2026-10-23; False => "
          "Stokes (no inertia), which a creeping-flow script must now ask for. Inert under "
          "enable_vof_momentum, whose VoF fluxes carry the momentum advection.");
  cls.def(
      "set_advection_scheme",
      [](S& s, const std::string& scheme) {
        s.setAdvectionScheme(enum_index(scheme, kAdvSchemes, "set_advection_scheme", "scheme"));
      },
      nb::arg("scheme"),
      "High-order momentum advection scheme: 'sou' (second-order upwind, the default) or "
      "'koren' (Koren TVD).");
  cls.def("set_collocated_scheme", &S::setCollocatedScheme, nb::arg("name"),
          "Collocated cut-cell projection scheme (no effect on the staggered solver). DEFAULT "
          "since 2026-08-25: AUTO = 'ghost' where supported, falling back to 'gauge-exact' with "
          "a stderr notice on porous/variable-rho/domain-BC/Chebyshev configurations (ghost v1 "
          "limits); any explicit selection here disables AUTO.\n"
          "  'ghost' -- the fluid-only constraint scheme (binary-openness divergence + directional "
          "closures + the same gauge-exact gradient). Family-free, unconditionally stable "
          "(dt 60..1e20, no stabilizer), protocol-independent (C2), both-bed clean ladder "
          "-1.4% (R=8) -> +0.22% asymptote, Z&H anchor -0.018% at N=128. Costs: nonsymmetric "
          "BiCGStab pressure solve (~2.3-2.7x), ~1.6 KB/cell overlay (single-GPU size cap), "
          "fragmentation guard. MPI-capable (np=1,2,4 ctests); at-scale np>=16 hardening in "
          "progress. Equivalent to diagnostics.set_ghost_projection(True, 2, 2); the (1, 2) "
          "mixed mode stays quarantined there.\n"
          "  'gauge-exact' (the AUTO fallback) -- the aperture constraint with the directional "
          "gauge-exact pressure gradient. Cheapest scheme measured (symmetric MG-PCG, no "
          "fragmentation guard). CAVEATS from the 2026-08 campaign "
          "(doc/collocated_invisible_subspace.md): possesses an attractor FAMILY of steady states "
          "(support-inconsistent gradient/constraint pair) selected by march protocol, and a "
          "rotational-update instability whose wall-blend stopgap "
          "(diagnostics.set_rotational_wall_weight) has a resolution-dependent margin -- UNSTABLE "
          "at (R>=16 cells/radius, dt>=600); use fixed dt<=60 protocols at high resolution. "
          "Clean-protocol bias vs the staggered reference: -2.5% (R=8) -> ~+0.2% asymptote.\n"
          "  'embed' -- the complete Basilisk embed.h port (doc/history/"
          "collocated_embed_port_plan.md): the finite-volume momentum operator with the "
          "true-normal dirichlet_gradient wall drag applied as a defect correction on the IBM "
          "matrix, the openness-weighted -grad(P) predictor and cell correction, the wall-aware "
          "face map and the solid-cut-cell sliver mask. The live candidate for removing the "
          "collocated accuracy ceiling; its two intermediate rungs are "
          "diagnostics.set_face_interp(5 | 6).\n"
          "  'plain' -- the legacy path: plain 1/2-1/2 face average + central-difference grad(P). "
          "FIRST order at curved cut cells (the cell gradient reads the decoupled p=0 of "
          "solid-centred neighbours, an O(1/h) gauge error), and on a dense bed it also fails to "
          "reach steady state within 800 steps at coarse resolution. Kept for reproducing "
          "published results.");
  cls.def("set_decomposition", &S::setDecomposition, nb::arg("levels"),
          nb::arg("max_imbalance") = 1.05,
          "Choose how this solver's shared MPI decomposition is built. levels=0 (DEFAULT) = AUTO: "
          "the aligned ORB (split positions on the FINE grid snapped to a power of two, capped at "
          "16, i.e. 5 nested multigrid levels) whenever it is exactly balanced; otherwise the "
          "coarse-first partition at depth 5, then 4, built with a balance-aware split axis, the "
          "first whose max-block/min-block ratio stays within max_imbalance (default 1.05) and "
          "that nests at least as many levels in place as the aligned ORB (every block even and "
          "at least 4 cells wide at each); otherwise the balance-aware aligned ORB if it has a "
          "smaller largest block and nests at least min(3, the aligned ORB's) levels; otherwise "
          "the aligned ORB. AUTO "
          "never cuts an axis the aligned ORB leaves whole, and changes its set of cut axes only "
          "when the aligned ORB's largest block is at least 5/4 of the mean. levels=1 = the "
          "aligned ORB as is "
          "(the default before 2026-10: use it to reproduce an older np > 1 partition). "
          "levels >= 2 = COARSE-FIRST: build the ORB on the grid coarsened levels-1 times and "
          "refine the partition upward, so every block is a multiple of the coarsening factor BY "
          "CONSTRUCTION and the hierarchy nests for the full requested depth; each depth from "
          "`levels` down to 2 is built and MEASURED, and the deepest one within max_imbalance is "
          "taken, else the aligned ORB. levels < 0 or max_imbalance <= 1.0 raise ValueError. Call "
          "BEFORE init_mpi() (a later call raises), and pass "
          "the SAME levels / max_imbalance to flow.mpi_block() -- both derive the same partition "
          "and must agree.");
  cls.def_prop_ro("decomposition_levels", &S::decompositionLevels,
                  "This solver's decomposition mode (0 = auto, 1 = aligned ORB, >= 2 = "
                  "coarse-first with that depth).");
  cls.def_prop_ro("decomposition_max_imbalance", &S::decompositionMaxImbalance,
                  "The load-imbalance budget the auto and coarse-first searches stay within.");
  cls.def("set_pressure_bottom_extent", &S::setPressureBottomExtent, nb::arg("cells"),
          "The threshold set_pressure_bottom('auto') applies: agglomerate the coarsest level "
          "onto a global operator once the coarsest GLOBAL grid exceeds this many cells on any "
          "AXIS (not cell count -- what a few smoother sweeps cannot fix is a mode spanning many "
          "cells along one axis). DEFAULT 4. Under the automatic depth (set_pressure_multigrid "
          "levels=0) it also enters the depth, at the geometry build: a later call changes the "
          "agglomeration but not the depth already resolved. "
          "`cells` is in cells (a count, not a length); <= 0 restores the default.");
  cls.def_prop_ro("pressure_bottom_extent", &S::pressureBottomExtent,
                  "The agglomerated-bottom extent threshold (see set_pressure_bottom_extent). "
                  "Measured in cells (a count, not a length).");
  cls.def("set_implicit_advection", &S::setImplicitAdvection, nb::arg("on"),
          "Use implicit-FOU advection with deferred-correction TVD.");
  cls.def("set_velocity_residual_tolerance", &chk::setVelocityResidualTolerance<S>, nb::arg("rtol"),
          "Residual-based momentum stop: a component's implicit solve ends once max|b - A u| <= "
          "rtol * max(max|b|, max|A u|) over the solved unknowns (at least one sweep always runs; "
          "a round-off floor stops a solve that cannot improve). rtol < 0 (DEFAULT) FOLLOWS THE "
          "PRESSURE SOLVER'S rtol -- the projection consumes u* and resolves the divergence the "
          "momentum residual leaves to its own tolerance; rtol > 0 fixes it; rtol == 0 restores "
          "the legacy update criterion (relative to the FIRST sweep's update, which on a "
          "warm-started near-steady step is noise and cost the whole sweep cap). Every path.");
  cls.def("velocity_residual_tolerance", &S::velocityResidualTolerance,
          "The momentum residual tolerance in force (resolves the follow-the-pressure default).");
  cls.def("set_backflow_stabilization", &chk::setBackflowStabilization<S>, nb::arg("beta"),
          "Outflow backflow-stabilization coefficient (Bazilevs 2009 / Esmaily-Moghadam 2011): "
          "beta "
          "in [0,1] scales the dissipative outflow term that prevents backflow divergence when "
          "flow "
          "reverses at the outlet (e.g. a separated wake / BFS recirculation). Default 0.2; 0 = "
          "off. "
          "Inert where the outlet is purely outgoing.");
  cls.def("set_pressure_solver_params", &chk::setPressureSolverParams<S>, nb::arg("iters"),
          "Set the pressure smoother iteration count.");
  cls.def("set_pressure_multigrid", &chk::setPressureMultigrid<S>, nb::arg("on"),
          nb::arg("levels") = 0,
          "Set the pressure multigrid depth, fixed at the next geometry build. levels=0 (the "
          "default) derives it: the shallowest depth >= 4 whose global coarsest grid the exact "
          "bottom solver admits (<= 8192 cells, a plane of <= 192; an operator with an outflow "
          "face keeps depth 4), or where coarsening ends by itself. levels=1 => pure RB-GS, no "
          "coarse grid; levels >= 2 => exactly that depth. The resolved depth is "
          "len(diagnostics.pressure_mg_level_ratios()). set_pressure_bottom_extent must precede "
          "the geometry to enter the depth (doc/mg_depth_policy.md).");
  cls.def("set_pressure_chebyshev", &chk::setPressureChebyshev<S>, nb::arg("on"),
          nb::arg("max_iter") = 120, nb::arg("rtol") = 1e-9,
          "Use the communication-light Chebyshev pressure accelerator. Mutually exclusive with the "
          "two CG drivers (on=True clears an FCG selection); on=False deselects Chebyshev and "
          "falls back to FCG if that is selected, otherwise to MG-PCG.");
  cls.def("set_pressure_telescope", &S::setPressureTelescope, nb::arg("on"),
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
          "rank count. ON by default since 2026-09-02 (the FoxBerry ladder); False restores the "
          "in-place-only hierarchy, byte-identical to before telescoping existed. See "
          "suite/docs/archive/MG_TELESCOPING_PLAN.md.");
  cls.def("pressure_telescope", &S::pressureTelescope,
          "Whether the pressure multigrid's coarse levels telescope (merge ORB siblings onto "
          "fewer ranks once a level's block turns odd) rather than stop coarsening in place. "
          "Reads back set_pressure_telescope's current setting; ON by default since 2026-09-02 "
          "(the FoxBerry ladder), so this is normally True even though it was never called "
          "explicitly -- check it rather than assuming the legacy in-place-only hierarchy.");
  cls.def(
      "set_pressure_bottom",
      [](S& s, const std::string& m) {
        if (m == "auto")
          s.setPressureBottomMode(-1);
        else if (m == "smoother")
          s.setPressureBottomMode(0);
        else if (m == "agglomerated")
          s.setPressureBottomMode(1);
        else
          throw std::runtime_error("set_pressure_bottom: 'auto' | 'smoother' | 'agglomerated'");
      },
      nb::arg("mode"),
      "Coarse-level (bottom) solve of the pressure multigrid. A V-cycle converges at a "
      "domain-independent rate only if its COARSEST level is effectively solved, and a geometric "
      "hierarchy cannot always get small enough: an axis stops coarsening once it turns odd, and "
      "under MPI once ANY rank's block turns odd -- so at fixed cells/rank the coarsest GLOBAL "
      "grid grows with the rank count and the bottom is progressively under-solved. "
      "'auto' (DEFAULT) agglomerates the coarsest level onto a global operator and solves it "
      "exactly whenever it exceeds set_pressure_bottom_extent (4) cells on any axis, and uses "
      "the cheap smoothed bottom otherwise (byte-identical to 'smoother' then). 'smoother' = "
      "never agglomerate (legacy). 'agglomerated' = always. Measured on one GPU (2048x64x64 "
      "channel): a smoothed bottom needs 13.5 pressure iterations/step at 4 levels and 6.0 at "
      "6, against 4.4 at full geometric depth; agglomerated it is 4.0 at BOTH depths, and "
      "faster in wall-clock than the deep hierarchy. Works on the cut-cell IBM and "
      "ghost-projection paths (per-fluid-component null-space projection, 2026-08-13). Under "
      "the automatic depth (set_pressure_multigrid levels=0) the mode also enters the depth, at "
      "the geometry build: a later call changes the bottom solve but not the depth already "
      "resolved.");
  cls.def("set_pressure_pcg", &chk::setPressurePcg<S>, nb::arg("on"), nb::arg("max_iter") = 200,
          nb::arg("rtol") = 1e-8,
          "Use the MG-PCG pressure accelerator (single-GPU default) and set its iteration cap and "
          "relative tolerance. on=True GENUINELY selects: it clears both competing selections "
          "(Chebyshev and FCG), so it works after diagnostics.set_density_mode / set_porous -- "
          "last set wins. "
          "on=False RAISES: MG-PCG is the terminal fallback of the driver dispatch, so it cannot "
          "be deselected on its own -- select the driver you want instead "
          "(set_pressure_chebyshev(True, ...) or set_pressure_fcg(True, ...)). Under "
          "diagnostics.set_ghost_projection the nonsymmetric operator is solved by BiCGStab; this "
          "call still "
          "sets that solve's cap/tolerance (they are shared) and does not change its method.");
  cls.def("set_pressure_fcg", &chk::setPressureFcg<S>, nb::arg("on"), nb::arg("max_iter") = 200,
          nb::arg("rtol") = 1e-8,
          "Use the FLEXIBLE MG-CG pressure accelerator: the same V-cycle-preconditioned CG as "
          "set_pressure_pcg, the same stopping estimate/mean removal, and the same cap+tolerance "
          "(they are shared -- it is the same solve), differing only in the beta recurrence "
          "(Polak-Ribiere r^T(z_{k+1}-z_k)/r^T z_k instead of Fletcher-Reeves). Costs one extra "
          "level-0 vector and one extra global dot per iteration; buys tolerance of a "
          "preconditioner that is not symmetric with respect to the fine operator. On a symmetric "
          "preconditioner the two betas are algebraically identical, so FCG reproduces PCG's "
          "iteration count. Unlike set_pressure_pcg this flag GENUINELY selects: on=True clears "
          "the Chebyshev selection (so it works after diagnostics.set_density_mode/set_porous), "
          "on=False "
          "returns to MG-PCG, and a later set_pressure_chebyshev(True, ...) wins over it.");
  cls.def("set_velocity_multigrid", &chk::setVelocityMultigrid<S>, nb::arg("on"),
          nb::arg("levels") = 4, nb::arg("vcycles") = 8,
          "Enable velocity (momentum) multigrid for the implicit diffusion solve.");
}

/// `Solver` (and its collocated twin): state, stepping, velocity and field read-outs.
template <class Grid, class C>
inline void bindSolverState(C& cls) {
  using S = BoundSolver<Grid>;
  cls.def(
      "set_state",
      [](S& s, nb::ndarray<double, nb::f_contig> u, nb::ndarray<double, nb::f_contig> v,
         nb::ndarray<double, nb::f_contig> w) {
        s.uploadVelocity(grid_in(u), grid_in(v), grid_in(w));
      },
      nb::arg("u"), nb::arg("v"), nb::arg("w"),
      "Upload an initial velocity field (u,v,w each a Fortran-order (nx,ny,nz) float64 array).");
  cls.def("step", &S::step,
          "Advance the solver one time step (semi-implicit: diffusion + projection).\n\n"
          "ATOMIC across the two explicit two-phase stability caps: the Weymouth-Yue "
          "boundedness cap and the Brackbill capillary limit are evaluated at the HEAD of the "
          "call, so a rejected dt leaves every field bitwise as it was and a retry at a smaller "
          "dt is exact. (Before 2026-09-04 the momentum half had already advanced by the "
          "rejected dt, so catch-and-halve desynchronised colour and momentum.) The WY "
          "pre-check reads the field the call starts from; where the projection itself "
          "accelerates the field inside the step, use step_adaptive().");
  cls.def(
      "step_adaptive",
      [](S& s, double cfl, double capCfl, double dtMax) {
        return s.stepAdaptive(cfl, capCfl, dtMax);
      },
      nb::arg("cfl_target") = 0.4, nb::arg("capillary_cfl") = 0.4,
      nb::arg("dt_max") = std::numeric_limits<double>::infinity(),
      "One step at a dt re-picked from the CURRENT state; returns the dt used.\n\n"
      "dt = min(cfl_target * cfl_dt, capillary_cfl * capillary_dt, dt_max) with both limits "
      "taken from vof_step_limits() at the moment of the call, then set_dt(dt) + step(). "
      "Needs enable_vof().\n\n"
      "WHY: both limits are INSTANTANEOUS. Re-picking every ten steps is safe for a free "
      "bubble or a falling film and is NOT safe in a packing -- a throat jet can grow "
      "max|uf| by more than 50 %% inside ten steps (measured, examples/bubble-through-packing) "
      "and the WY cap is a hard throw; and capillary_dt itself reads the density field, so a "
      "value taken before the closures have run can be 7x too large (measured, "
      "examples/pore-scale-imbibition). This is bitwise the "
      "`dt = 0.4*min(L['cfl_dt'], L['capillary_dt']); s.set_dt(dt); s.step()` loop the "
      "gallery drivers write by hand (IEEE multiplication is monotone, so "
      "f*min(a,b) == min(f*a,f*b)). set_dt/step semantics are unchanged. "
      "cfl_target and capillary_cfl are dimensionless; dt_max and the returned dt are in the "
      "caller's time units.");
  cls.def(
      "set_velocity",
      [](S& s, int c, nb::ndarray<double, nb::f_contig> a) { s.setVelocity(c, grid_in(a)); },
      nb::arg("c"), nb::arg("array"),
      "Write component c's inner velocity from a Fortran-order (nx,ny,nz) float64 array "
      "(solid rows re-masked). Initial-condition hook: a uniform stream around a fixed body "
      "is the Galilean twin of the same body towed through fluid at rest. "
      "The array is in the caller's velocity units (length/time), as get_u returns it.");
  cls.def(
      "get_u", [](S& s) { return field_out(s, s.getVelocity(0)); },
      "Return the x-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
      "[x,y,z]).");
  cls.def(
      "get_v", [](S& s) { return field_out(s, s.getVelocity(1)); },
      "Return the y-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
      "[x,y,z]).");
  cls.def(
      "get_w", [](S& s) { return field_out(s, s.getVelocity(2)); },
      "Return the z-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
      "[x,y,z]).");
  cls.def(
      "get_p", [](S& s) { return field_out(s, s.getPressure()); },
      "Return the physical pressure as a Fortran-order (nx,ny,nz) float64 array (index "
      "[x,y,z]).");
  cls.def(
      "get_ox", [](S& s) { return field_out(s, s.getOpenness(0)); },
      "The -x face openness field of the cut-cell operator: the fluid area fraction (0..1, "
      "dimensionless) of the face between cells (i-1,j,k) and (i,j,k), as a Fortran-order "
      "(nx,ny,nz) float64 array over THIS rank's inner cells. Computed by set_solid from the "
      "SDF (1 everywhere without a solid); it is what the momentum operator and the geometric "
      "cut-cell projection see. get_ox_proj is the openness the PROJECTION actually conserves "
      "fluxes through, which differs under diagnostics.set_ghost_projection (binary/coupled "
      "faces).");
  cls.def(
      "get_oy", [](S& s) { return field_out(s, s.getOpenness(1)); },
      "The geometric -y face openness per inner cell, (nx,ny,nz); see get_ox.");
  cls.def(
      "get_oz", [](S& s) { return field_out(s, s.getOpenness(2)); },
      "The geometric -z face openness per inner cell, (nx,ny,nz); see get_ox.");
  cls.def(
      "get_ox_proj", [](S& s) { return field_out(s, s.getOpennessProj(0)); },
      "-x face openness whose fluxes the projection CONSERVES (binary/COUPLED under "
      "diagnostics.set_ghost_projection, geometric cut-cell otherwise). Use for flux bookkeeping "
      "(peclet.pnm extract_network_flow).");
  cls.def(
      "get_oy_proj", [](S& s) { return field_out(s, s.getOpennessProj(1)); },
      "-y face openness the projection conserves (see get_ox_proj).");
  cls.def(
      "get_oz_proj", [](S& s) { return field_out(s, s.getOpennessProj(2)); },
      "-z face openness the projection conserves (see get_ox_proj).");
  cls.def(
      "get_uf", [](S& s) { return field_out(s, s.getFaceVelocity(0)); },
      "Return the divergence-free FACE x-velocity (collocated: projected MAC field; staggered: "
      "== get_u).");
  cls.def(
      "get_vf", [](S& s) { return field_out(s, s.getFaceVelocity(1)); },
      "Return the divergence-free FACE y-velocity (collocated: projected MAC field; staggered: "
      "== get_v).");
  cls.def(
      "get_wf", [](S& s) { return field_out(s, s.getFaceVelocity(2)); },
      "Return the divergence-free FACE z-velocity (collocated: projected MAC field; staggered: "
      "== get_w).");
  // --- Named field registry (multiphysics field container) ---------------------------------
  cls.def(
      "add_field", [](S& s, const std::string& name) { s.addField(name); }, nb::arg("name"),
      "Register a new zero-initialised cell-centred field on the grid (for transported scalars "
      "or material properties). Idempotent.");
  cls.def(
      "has_field", [](S& s, const std::string& name) { return s.hasField(name); }, nb::arg("name"),
      "Whether a field of this name is registered.");
  cls.def(
      "field_names", [](S& s) { return s.fieldNames(); },
      "Names of all registered fields (velocity u/v/w, p, sdf, plus any added), sorted.");
  cls.def(
      "get_field", [](S& s, const std::string& name) { return field_out(s, s.getField(name)); },
      nb::arg("name"),
      "Return a registered field's inner region as a Fortran-order (nx,ny,nz) float64 array.");
  cls.def(
      "set_field",
      [](S& s, const std::string& name, nb::ndarray<double, nb::f_contig> a) {
        s.setField(name, grid_in(a));
      },
      nb::arg("name"), nb::arg("array"),
      "Write a Fortran-order (nx,ny,nz) float64 array into a registered field's inner region "
      "(ghosts refilled on the next diagnostics.exchange_field/step).");
}

/// `Solver` (and its collocated twin): ghost width, porous continuity, divergence read-outs and
/// MPI.
template <class Grid, class C>
inline void bindSolverPorousMpi(C& cls) {
  using S = BoundSolver<Grid>;
  cls.def(
      "ghost_width", [](S& s) { return s.ghostWidth(); },
      "Ghost-layer width g of the velocity block (diagnostics.field_view returns an (n+2g) "
      "buffer).");
  cls.def(
      "has_cutcell_pressure", [](S& s) { return s.hasCutcellPressure(); },
      "True once the cut-cell pressure operator exists (set_solid or set_pressure_geometry was "
      "called). The porous continuity requires it; the coupling driver auto-sets an all-fluid "
      "geometry when absent.");
  cls.def(
      "set_porous_continuity", [](S& s, bool on) { s.setPorousContinuity(on); },
      nb::arg("on") = true,
      "Enable the volume-averaged (porous) continuity for unresolved CFD-DEM (staggered only): "
      "the projection enforces d(eps)/dt + div(eps u) = 0 instead of div(u)=0, so the fluid "
      "velocity is NOT solenoidal where the void fraction changes (bubbling/expansion). Binds "
      "the 'eps' field (void fraction from the particle deposition, created seeded to 1 if "
      "absent; the coupling writes it each step BEFORE step()). eps=1 everywhere reduces "
      "exactly "
      "to div(u)=0. Pair with max_porous_residual() for the meaningful convergence check.");
  cls.def(
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
      "a -DPECLET_FLOW_OPERATOR_DOUBLE build (i.e. not a solver residual), against the projected "
      "field's own residual from this call. Identical to max_open_divergence() when there is "
      "no outflow face, and on the collocated grid (which already measures the face field).");
  cls.def("max_open_divergence", &S::maxOpenDivergence,
          "Return the max cut-cell velocity-flux divergence max|div(open*u)|. With porous "
          "continuity this is NOT ~0 -- it equals -d(eps)/dt (the bed expanding). Use "
          "max_porous_residual() for the continuity residual.");
  cls.def("sync_porous_prev", &S::syncPorousPrev,
          "Reseed eps^n = eps^{n+1} (d(eps)/dt = 0 for the next step). Call it once the first "
          "void-fraction field has been written into 'eps' and before the first step, so that "
          "step sees no spurious d(eps)/dt source; any later call simply restarts the "
          "d(eps)/dt bookkeeping from the current field.");
  cls.def("max_porous_residual", &S::maxPorousResidual,
          "Residual of the volume-averaged continuity max|div(open*eps*u) + d(eps)/dt| -- the "
          "quantity the porous projection drives to zero. 0 unless set_porous_continuity(True).");
  cls.def(
      "block_origin",
      [](S& s) {
        auto o = s.blockOrigin();
        return std::vector<int>{o[0], o[1], o[2]};
      },
      "This rank's inner-block origin in GLOBAL cells ([0,0,0] single-rank). Shift the "
      "coupling "
      "deposit origin by this so particles in global coordinates land in the local block. "
      "The offset is in cells (a count, not a length).");
#ifdef PECLET_FLOW_MPI
  // Distributed path (built with -DPECLET_FLOW_MPI): construct the Solver with this rank's
  // LOCAL block dims (see the module-level mpi_block()), then init_mpi with the GLOBAL grid
  // dims. step() then does the g=2 velocity-block halo exchange + the distributed cut-cell
  // pressure MG. Bit-exact to single-rank.
  cls.def(
      "init_mpi",
      [](S& s, int gnx, int gny, int gnz) {
        chk::globalDims("init_mpi", gnx, gny, gnz);
        ensure_mpi_init();
        s.initMpi(gnx, gny, gnz, MPI_COMM_WORLD);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
      "Wire the multi-rank step: pass the GLOBAL grid dims (gnx,gny,gnz). The Solver must have "
      "been "
      "constructed with this rank's LOCAL block dims (from mpi_block). MPI_Init is called if "
      "needed. Call it BEFORE the geometry (set_solid / set_pressure_geometry / "
      "set_solid_from_scene): the geometry builds the pressure and velocity multigrids for "
      "the partition in force when it runs, so init_mpi raises if a geometry already exists "
      "(built first, every rank would solve its own block-periodic pressure problem).\n\n"
      "Placement (host OpenMP builds): threads per rank should divide the cores per L3 cache "
      "domain (AMD EPYC Zen 4: 8; Threadripper 5965WX: 6), ideally one rank per domain -- a "
      "rank whose threads span two domains runs measurably slower (up to +18 % step time) "
      "and no partition fixes it. With more than one rank, when every OpenMP thread is "
      "pinned to one CPU, rank 0 prints one stderr line naming the ranks that straddle; "
      "nothing else changes.");
  cls.def(
      "rank",
      [](S&) {
        ensure_mpi_init();
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
      },
      "This rank's index in MPI_COMM_WORLD.");
  cls.def(
      "size",
      [](S&) {
        ensure_mpi_init();
        int n = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &n);
        return n;
      },
      "The number of ranks in MPI_COMM_WORLD.");
#else
  cls.def(
      "rank", [](S&) { return 0; },
      "MPI rank (always 0: this module was built without PECLET_FLOW_MPI; a build with it "
      "exposes init_mpi / mpi_block and the real rank).");
  cls.def(
      "size", [](S&) { return 1; },
      "MPI size (always 1: this module was built without PECLET_FLOW_MPI).");
#endif
}

/// `SolverDiagnostics` (and its collocated twin): pressure under-relaxation, porous options,
/// rebalance and broadcast.
template <class Grid, class C>
inline void bindDiagnosticsPorousMpi(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "set_pressure_underrelax", [](D& diag, double w) { diag.s->setPressureUnderRelax(w); },
      nb::arg("omega"),
      "Pressure under-relaxation factor omega_p in (0,1] for the incremental accumulation "
      "(MFIX "
      "§10.1); 1.0 = off (default). <1 damps the incremental predictor overshoot on stiff "
      "porous+drag.");
  cls.def(
      "set_porous_deps_dt", [](D& diag, bool on) { diag.s->setPorousDepsDt(on); }, nb::arg("on"),
      "Include (default True) or drop the d(eps)/dt source in the porous projection RHS. Drop "
      "it "
      "to enforce div(eps u)=0 when the per-cell eps deposit's time-derivative is too jagged "
      "and "
      "destabilizes the eps-weighted pressure solve.");
  cls.def(
      "set_porous_conservative", [](D& diag, bool on) { diag.s->setPorousConservative(on); },
      nb::arg("on"),
      "eps-conservative porous momentum + projection pair (default True): time term "
      "(eps_f rho/dt) u, eps rho-weighted advective form, projection coefficients "
      "open*(eps rho idt)/(eps rho idt+beta) with matching correction. False = the legacy "
      "plain-u pair (A/B only; it lets the projection drag gas with the moving porosity at "
      "zero inertia cost — a spurious late-time energy source in clustering flows).");
#ifdef PECLET_FLOW_MPI
  // Distributed path (built with -DPECLET_FLOW_MPI): construct the Solver with this rank's
  // LOCAL block dims (see the module-level mpi_block()), then init_mpi with the GLOBAL grid
  // dims. step() then does the g=2 velocity-block halo exchange + the distributed cut-cell
  // pressure MG. Bit-exact to single-rank.
  cls.def(
      "rebalance_by_weights",
      [](D& diag, const std::vector<double>& w) { return diag.s->rebalanceByWeights(w); },
      nb::arg("weights"),
      "Dynamic load balancing: redistribute the solver's state onto the weighted ORB of "
      "per-cell weights (global x-fastest, gnx*gny*gnz), its split planes aligned to 2^a "
      "cells with a the largest alignment whose weight imbalance stays within 1.05, so the "
      "pressure multigrid keeps coarsening in place for a levels. Returns that alignment "
      "(1 = the plain weighted ORB). Pass fluid work + gamma*particle_count; a coupled dem "
      "migrates onto the SAME partition with migrate_to_weights(weights, align=<the "
      "return value>). State-preserving (bit-exact at np=1, reduction floor at np>1).");
#else
#endif
  cls.def(
      "bcast_from_root", [](D&, nb::object v) { return v; }, nb::arg("value"),
      "Broadcast a value from rank 0 (identity in the single-rank module; mirrors the MPI "
      "API).");
}

}  // namespace peclet::flow::bindings
