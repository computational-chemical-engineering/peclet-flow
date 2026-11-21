/// @file
/// @brief Python binding of peclet.flow: the developer tier's solver tuning, ablation switches and
/// pressure / momentum instruments.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `SolverDiagnostics` (and its collocated twin): face interpolation, pressure projection and
/// advection switches, outer iterations, velocity solver.
template <class Grid, class C>
inline void bindDiagnosticsSolverOptions(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "set_face_interp",
      [](D& diag, int mode) {
        if (mode != 5 && mode != 6)
          throw std::invalid_argument(
              "diagnostics.set_face_interp: mode must be 5 or 6 (the two intermediate rungs "
              "of the embed port); the complete schemes are Solver.set_collocated_scheme("
              "'plain' | 'gauge-exact' | 'embed' | 'ghost')");
        diag.s->setFaceInterp(mode);
      },
      nb::arg("mode"),
      "The two INTERMEDIATE rungs of the Basilisk embed.h port on the collocated grid "
      "(doc/history/collocated_embed_port_plan.md), reachable for the ladder that measures "
      "what each piece of the complete port ('embed' = mode 7) is worth: 5 = the FV momentum "
      "operator with the true-normal dirichlet_gradient wall drag as a defect correction, on "
      "the wall-aware face map with the transpose-gradient pressure force; 6 = plus the "
      "openness-weighted -grad(P) predictor and cell correction, on the plain face map "
      "(7 adds the wall-aware face map and the solid-cut-cell sliver mask). CHANGES RESULTS; "
      "no effect on the staggered solver. Modes 1-4 and 10-13 (ablations) were deleted with "
      "their kernels at 1.0.0.");
  cls.def(
      "set_incremental_pressure",
      [](D& diag, bool on) { return diag.s->setIncrementalPressure(on); }, nb::arg("on"),
      "Toggle the rotational incremental-pressure projection.");
  cls.def(
      "set_pressure_warmstart", [](D& diag, bool on) { return diag.s->setPressureWarmstart(on); },
      nb::arg("on"), "Seed each pressure solve from the previous step's phi (default off).");
  cls.def(
      "set_rotational_pressure", [](D& diag, bool on) { return diag.s->setRotationalPressure(on); },
      nb::arg("on"),
      "PM I ablation (Guy-Fogelson 2005): False drops the rotational -mu*div(u*) term from "
      "the incremental pressure accumulation (constant-mu path only). Default True = shipped "
      "rotational (Timmermans) update.");
  cls.def(
      "set_rotational_filter",
      [](D& diag, bool on, double eps) { return diag.s->setRotationalFilter(on, eps); },
      nb::arg("on"), nb::arg("eps") = 0.05,
      "Experimental filtered rotational update: smooth div(u*) (mask-aware axis-wise 1-2-1, "
      "one-sided into the fluid at walls) before accumulating -mu*div into P. Keeps the O(1) "
      "pressure-relaxation gain, removes the checkerboard feedback channel.");
  cls.def(
      "set_rotational_weight", [](D& diag, double w) { return diag.s->setRotationalWeight(w); },
      nb::arg("w"),
      "Under-relax the rotational term: P += ct*phi - w*mu*div(u*). 1 = shipped, 0 = PM I; "
      "small w raises the boundary-mode stability threshold ~1/w at ~1/w slower smooth-mode "
      "pressure relaxation. phi=0 stays the unique fixed point for any w>0 at every dt.");
  cls.def(
      "set_rotational_wall_weight",
      [](D& diag, double w0) { return diag.s->setRotationalWallWeight(w0); }, nb::arg("w0"),
      "Wall-banded rotational blend: at fluid cells with a solid axis-neighbour use "
      "P += (rho/dt + w0*mu/dx^2)*phi - (1-w0)*mu*div(u*); bulk keeps the full rotational "
      "update. Stabilizes the boundary rows without slowing bulk pressure relaxation, and "
      "keeps them relaxing at dt->infinity. 0 (default) = off.");
  cls.def(
      "set_aperture_order", [](D& diag, int order) { return diag.s->setApertureOrder(order); },
      nb::arg("order"),
      "Face-aperture estimator (DEFAULT 2 since 2026-08-26): 1 = legacy one-sample linear "
      "model, 2 = "
      "marching-squares (5 samples/face, O(h^2); removes the convexity bias measured at "
      "+0.59/+0.27% bed permeability at R=8/12 -- see doc/collocated_paper_plan.md row 51). "
      "For analytic geometry, exact apertures via set_openness_override are better still. "
      "Call before set_solid (a later call raises).");
  cls.def(
      "set_advection_wall_velocity",
      [](D& diag, bool on) { return diag.s->setAdvectionWallVelocity(on); }, nb::arg("on"),
      "A0: fill the advection inputs' masked (solid) rows with the instantaneous WALL "
      "velocity instead of zeros, so a MOVING body's advective term sees the body's own "
      "motion. DEFAULT True; False is the pre-A0 ablation (measurement only). Inert on a "
      "static scene and on the collocated grid.");
  cls.def_prop_ro(
      "advection_wall_velocity", [](D& diag) { return diag.s->advectionWallVelocity(); },
      "Whether the wall-velocity advection inputs are in force.");
  cls.def(
      "set_uf_advection", [](D& diag, bool on) { return diag.s->setUfAdvection(on); },
      nb::arg("on"),
      "Advect the COLLOCATED momentum with the projected, divergence-free MAC face field "
      "uf/vf/wf of the last projection instead of the un-projected cell->face average "
      "0.5*(u_i+u_j). DEFAULT True -- the shipped scheme: doc/flow_colocated_plan.md "
      "step 3, the Almgren-Bell-Colella prescription (the field the projection just made "
      "solenoidal IS the conservative advective flux), and what the FOU operator's "
      "conservative row-sum identity needs. False is the ablation that restores the phase-2 "
      "average (measurement only; see doc/uf_advection.md). Inert on the staggered grid (the "
      "stored velocity already IS the projected face velocity) and before the first "
      "projection or set_velocity has built a face field.");
  cls.def_prop_ro(
      "uf_advection", [](D& diag) { return diag.s->ufAdvection(); },
      "Whether the projected face field is the collocated advecting velocity.");
  cls.def(
      "set_comm_avoiding",
      [](D& diag, const std::string& mode) {
        int mask = 0;
        if (mode == "both")
          mask = peclet::flow::kCaBoth;
        else if (mode == "off")
          mask = 0;
        else if (mode == "momentum")
          mask = peclet::flow::kCaMomentum;
        else if (mode == "pressure")
          mask = peclet::flow::kCaMg;
        else
          throw std::runtime_error("set_comm_avoiding: 'both' | 'off' | 'momentum' | 'pressure'");
        diag.s->setCommAvoiding(mask);
      },
      nb::arg("mode"),
      "Communication-avoiding red-black smoothing (multi-rank): exchange a 2-deep ghost layer "
      "once per red-black PAIR instead of a 1-deep layer before every colour, redundantly "
      "re-smoothing the first colour's ghost ring so the second colour reads exactly what a "
      "per-colour exchange would have delivered. BIT-IDENTICAL, at half the halo events "
      "(measured on host: np=32 weak efficiency 35% -> 62%). 'both' (DEFAULT on a host "
      "backend) | 'off' (DEFAULT on a device backend: USER 2026-10-29, H100 weak scaling "
      "measured CA off 21-39 ms/step faster) | 'momentum' "
      "(the velocity RB-GS only) | 'pressure' (the pressure-multigrid coarse levels only) -- "
      "the split exists to ATTRIBUTE a measured regression to one subsystem. Set it BEFORE "
      "init_mpi (a later call raises: the momentum half is latched with the halo topology); "
      "either half is inert on blocks smaller than 4 cells on any axis, and on a periodic axis "
      "of odd global extent (the pressure half per multigrid level).");
  cls.def_prop_ro(
      "comm_avoiding",
      [](D& diag) {
        const int m = diag.s->commAvoiding();
        return std::string(m == peclet::flow::kCaBoth       ? "both"
                           : m == peclet::flow::kCaMomentum ? "momentum"
                           : m == peclet::flow::kCaMg       ? "pressure"
                                                            : "off");
      },
      "The communication-avoiding smoothing mode in force (see set_comm_avoiding).");
  cls.def(
      "set_multigrid_aspect_threshold",
      [](D& diag, double theta) { return diag.s->setMultigridAspectThreshold(theta); },
      nb::arg("theta"),
      "Anisotropic-coarsening aspect threshold theta shared by the pressure and velocity "
      "multigrids (doc/anisotropic_metric.md §5.1): an axis is coarsened only while its cell "
      "size stays within theta of the smallest one, so a stretched grid semi-coarsens instead "
      "of building levels whose operator is dominated by one direction. DEFAULT 2.0; must be "
      "> 1; 1e9 restores full coarsening on every axis (the §8.5 item (c) ablation). Only "
      "ever consulted on an ANISOTROPIC metric, so it is inert on cubic cells.");
  cls.def_prop_ro(
      "multigrid_aspect_threshold", [](D& diag) { return diag.s->multigridAspectThreshold(); },
      "The anisotropic-coarsening threshold in force (see "
      "set_multigrid_aspect_threshold).");
  cls.def(
      "set_pressure_strict", [](D& diag, bool on) { return diag.s->setPressureStrict(on); },
      nb::arg("on"),
      "Raise instead of report when the pressure preconditioner returns a non-finite "
      "correction. DEFAULT False: the shipped behaviour caps the solve, prints one line, and "
      "raises the pressure_solve_failed() flag, so a check can catch it without changing "
      "control flow. True turns the same event into an exception.");
  cls.def_prop_ro(
      "pressure_strict", [](D& diag) { return diag.s->pressureStrict(); },
      "Whether a non-finite pressure preconditioner output raises (see "
      "set_pressure_strict).");
  cls.def(
      "set_pressure_coarse_ghost",
      [](D& diag, bool on) { return diag.s->setPressureCoarseGhost(on); }, nb::arg("on"),
      "Zero-gradient (Neumann) coarse ghost on wall/inflow faces before the pressure "
      "multigrid's prolongation -- the WO-H symmetry repair. DEFAULT True. False restores "
      "the pre-2026-08-30 periodic-wrap ghost, which reinstates the asymmetry that stalls "
      "MG-PCG on every 3-D wall-bounded grid: a MEASUREMENT ABLATION, never a production "
      "setting. Inert on periodic / IBM problems, where the two are the same.");
  cls.def_prop_ro(
      "pressure_coarse_ghost", [](D& diag) { return diag.s->pressureCoarseGhost(); },
      "Whether the Neumann coarse ghost is in force (see set_pressure_coarse_ghost).");
  cls.def(
      "set_fluid_only_constraint",
      [](D& diag, const std::string& mode) {
        diag.s->setFluidOnlyConstraint(
            enum_index(mode, kFluidOnly, "set_fluid_only_constraint", "mode"));
      },
      nb::arg("mode"),
      "Fluid-only pressure constraint (route 2b, a collocated mechanism instrument; call "
      "before "
      "set_solid): 'filter' = Design A (openness filter), 'star' = Design B (SPD Kron star "
      "elimination), 'off'. Any setting other than 'off' disables the AUTO scheme.");
  cls.def(
      "set_outer_iterations", [](D& diag, int iters) { return diag.s->setOuterIterations(iters); },
      nb::arg("n"), "Set the number of Picard/outer iterations per step.");
  cls.def(
      "set_outer_tolerance", [](D& diag, double tol) { return diag.s->setOuterTolerance(tol); },
      nb::arg("tol"), "Set the outer (Picard) convergence tolerance.");
  cls.def(
      "last_outer_iterations", [](D& diag) { return diag.s->lastOuterIterations(); },
      "Return the outer-iteration count from the last step().");
  cls.def(
      "set_velocity_solver",
      [](D& diag, const std::string& name) {
        if (name == "auto")
          diag.s->setVelocitySolverAuto();
        else if (name == "gauss_seidel") {
          diag.s->setVelocityChebyshev(false);
          diag.s->setVelocityMultigrid(false, 3, 40);
        } else if (name == "multigrid") {
          diag.s->setVelocityChebyshev(false);
          diag.s->setVelocityMultigrid(true, 3, 40);
        } else if (name == "chebyshev") {
          diag.s->setVelocityMultigrid(false, 3, 40);
          diag.s->setVelocityChebyshev(true);
        } else
          throw std::invalid_argument(
              "set_velocity_solver: expected 'auto', 'gauss_seidel', 'multigrid' or "
              "'chebyshev', got '" +
              name + "'");
      },
      nb::arg("name"),
      "Which solver runs the implicit momentum (diffusion) equation. 'auto' (DEFAULT) picks by "
      "the operator's condition number kappa = 1 + 4 dt mu (sum w)/rho -- the multigrid at "
      "kappa >= 13, the smoother below -- and names no geometry, so an immersed body and a "
      "domain-BC box at the same dt get the same solver. 'gauss_seidel' is the red-black "
      "smoother; 'multigrid' the 3-level velocity V-cycle; 'chebyshev' Chebyshev "
      "semi-iteration over a Gershgorin interval. ALL FOUR SOLVE THE SAME EQUATION TO THE SAME "
      "TOLERANCE, so this changes cost and not the answer -- except that 'gauss_seidel' runs "
      "out of sweeps above kappa ~ 50 and then returns a solution that never met it. "
      "MEASURED (384^3 cut-cell bed, Snellius): multigrid beats gauss_seidel 1.66x-2.23x "
      "across the CPU ladder and 2.19x on one H100; chebyshev beats gauss_seidel by the "
      "sqrt(kappa) factor theory predicts but loses to multigrid except at the smallest "
      "per-rank blocks, where the evidence is suggestive and not conclusive.");
  cls.def(
      "set_velocity_mg_smoother",
      [](D& diag, const std::string& name, int degree, double eig_ratio) {
        if (name == "gauss_seidel")
          diag.s->setVelocityMgChebyshev(false);
        else if (name == "chebyshev")
          diag.s->setVelocityMgChebyshev(true, degree, eig_ratio);
        else
          throw std::invalid_argument(
              "set_velocity_mg_smoother: expected 'gauss_seidel' or 'chebyshev', got '" + name +
              "'");
      },
      nb::arg("name"), nb::arg("degree") = 0, nb::arg("eig_ratio") = 10.0,
      "Which smoother the velocity multigrid uses on every level -- orthogonal to "
      "set_velocity_solver, and read only when that selects 'multigrid'. 'gauss_seidel' "
      "(DEFAULT) or 'chebyshev'. ABLATION, AND CHEBYSHEV IS MEASURED SLOWER: at degree 4 / "
      "eig_ratio 6 it IS the stronger smoother per cycle (8.0 V-cycles against red-black's "
      "9.3) and still loses 1.21x on wall time, because matching red-black's smoothing "
      "strength costs the same four halo exchanges two sweeps cost, while Chebyshev needs a "
      "separate residual and update pass where a Gauss-Seidel sweep fuses the two. Kept "
      "because it is the apparatus for a question that gets asked repeatedly. degree=0 "
      "follows the V-cycle's own pre/post/bottom counts.");
  cls.def(
      "velocity_solver",
      [](D& diag) {
        return diag.s->velocityChebyshevActive()   ? std::string("chebyshev")
               : diag.s->velocityMultigridActive() ? std::string("multigrid")
                                                   : std::string("gauss_seidel");
      },
      "The momentum solver actually in force, resolving 'auto'. Only meaningful after the "
      "first step(), which is where the choice is made (dt and mu are final only then).");
  cls.def(
      "velocity_mg_smoother",
      [](D& diag) {
        return diag.s->velocityMgChebyshev() ? std::string("chebyshev")
                                             : std::string("gauss_seidel");
      },
      "The velocity multigrid's smoother.");
  cls.def(
      "set_velocity_solver_params",
      [](D& diag, int iters, double rtol, int min_iters) {
        diag.s->setVelocityIterations(iters);
        diag.s->setVelocityTolerance(rtol, min_iters);
      },
      nb::arg("iters"), nb::arg("rtol") = 0.0, nb::arg("min_iters") = 2,
      "Momentum smoother control: `iters` RB-GS sweeps per component, or with rtol > 0 a "
      "TOLERANCE STOP — end the loop once the sweep's max velocity increment has contracted "
      "to rtol of the first sweep's (iters becomes the cap, min_iters the floor). Easy "
      "regimes (small nu*dt/dx^2) exit after ~3-5 sweeps; stiff regimes run to the cap "
      "unchanged. rtol = 0 (default) is the legacy fixed count, byte-identical.");
  cls.def(
      "set_velocity_multigrid_auto",
      [](D& diag, long cellsPerRank, long minGlobalCells, int minBlockExtent) {
        return diag.s->setVelocityMultigridAuto(cellsPerRank, minGlobalCells, minBlockExtent);
      },
      nb::arg("cells_per_rank"), nb::arg("min_global_cells") = -1, nb::arg("min_block_extent") = -1,
      "Narrow the rule that selects the momentum solver. Since 2026-09-15 the DEFAULT is the "
      "velocity V-cycle on every validated operator mode and at every rank count: measured on "
      "the 1.0.0 scaling benchmark it is faster at every rung of both the CPU and the GPU "
      "ladder (2.23x at 147k cells/rank, 2.19x on one H100), and the red-black smoother it "
      "replaced was running at its sweep cap -- not converging -- on every rung of both. "
      "cells_per_rank sets an upper bound in cells per rank (0 = never use the V-cycle); "
      "set_velocity_multigrid_auto(65536, 1 << 23) restores the 1.0.0 behaviour exactly. "
      "min_block_extent (default 16) is the shortest per-rank inner extent worth building a "
      "hierarchy on. An explicit set_velocity_multigrid() overrides all of it.");
  cls.def(
      "velocity_multigrid_active", [](D& diag) { return diag.s->velocityMultigridActive(); },
      "Whether the velocity (momentum) solve uses the multigrid V-cycle rather than plain "
      "RB-GS. Reads back the OUTCOME, not the setting: an explicit set_velocity_multigrid("
      "on=...) call always wins, but absent one the AUTO rule (set_velocity_multigrid_auto) "
      "decides it when the geometry is built (set_solid / set_pressure_geometry / "
      "set_solid_from_scene) -- on for a distributed run (np > 1) in an eligible operator "
      "mode with a global problem of at least min_global_cells (default 8M) and fewer than "
      "cells_per_rank (default 65536) cells per rank, off otherwise. So the value can differ "
      "from what the caller last set explicitly, and is only meaningful after geometry has "
      "been installed (False on a fresh solver).");
}

/// `SolverDiagnostics` (and its collocated twin): momentum and pressure-solver options, pressure
/// instruments, step timers.
template <class Grid, class C>
inline void bindDiagnosticsPressure(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "outflow_backflow",
      [](D& diag) {
        const auto ob = diag.s->outflowBackflow();
        nb::dict d;
        d["max_reverse"] = ob.maxReverse;
        d["fraction"] = ob.fraction;
        d["energy_influx"] = ob.energyInflux;
        d["reversed_faces"] = ob.reversed;
        d["outlet_faces"] = ob.total;
        return d;
      },
      "Outflow REVERSAL census over the outflow faces (collective under MPI): 'max_reverse' = "
      "the largest reversed normal velocity max(0, -u.n), 'fraction' = reversed / all outlet "
      "faces, 'energy_influx' = sum over the reversed faces of rho |u.n| |u|^2 / 2 -- the "
      "kinetic-energy production the zero-gradient (do-nothing) outflow admits where the flow "
      "re-enters, which is the mechanism that diverges an inflow/outflow run whose "
      "recirculation reaches the outlet (a BFS bubble at x_r ~ L, a shed vortex). The "
      "backflow stabilization (set_backflow_stabilization, default beta 0.2) removes beta rho "
      "|u.n| |u_n|^2 of it; beta >= 0.5 is the unconditional energy bound. All zeros without "
      "an outflow face or without reversal. step() warns once, on stderr, when reversal "
      "appears while the stabilization is switched off.");
  cls.def(
      "last_momentum_residual", [](D& diag) { return diag.s->lastMomentumResidual(); },
      "max over components of max|r|/max|b| at exit of the last step's momentum solves "
      "(residual mode only; -1 otherwise).");
  cls.def(
      "set_deferred_correction", [](D& diag, bool on) { return diag.s->setDeferredCorrection(on); },
      nb::arg("on"),
      "Deferred-correction advection: True (default) = 2nd order (implicit FOU + explicit "
      "high-order correction, the high-order scheme being SOU by default or Koren TVD via "
      "set_advection_scheme); False = pure implicit FOU (1st-order upwind, more dissipative "
      "but "
      "unconditionally stable at sharp shear layers).");
  cls.def(
      "set_pressure_mean_removal",
      [](D& diag, const std::string& scope) {
        if (scope != "all" && scope != "fine")
          throw std::runtime_error("set_pressure_mean_removal: scope must be 'all' or 'fine'");
        diag.s->setPressureMeanRemoval(scope == "all");
      },
      nb::arg("scope"),
      "Nullspace (mean) removal scope in the pressure solve: 'fine' (DEFAULT — only the "
      "projections the Krylov iteration needs: rhs/residual, the fine-level V-cycle exit, the "
      "final iterate; ~3x fewer global-reduction latency hits per iteration, the measured "
      "winner of the multi-GPU ablation) or 'all' (legacy — every V-cycle level + after every "
      "matvec). Iteration counts are identical (A preserves mean-freeness); results equal "
      "within solver tolerance, not bit-identical.");
  cls.def(
      "set_pressure_bottom_solver",
      [](D& diag, const std::string& engine) {
        if (engine == "auto")
          diag.s->setPressureBottomSolver(0);
        else if (engine == "direct")
          diag.s->setPressureBottomSolver(1);
        else if (engine == "algebraic")
          diag.s->setPressureBottomSolver(2);
        else
          throw std::runtime_error("set_pressure_bottom_solver: 'auto' | 'direct' | 'algebraic'");
      },
      nb::arg("engine"),
      "Engine of the AGGLOMERATED pressure bottom (set_pressure_bottom 'auto' / "
      "'agglomerated'). 'auto' (DEFAULT): single rank, the singular (periodic / wall) "
      "operator, a bottom of at most 8192 cells with at most 64 fluid components (solids "
      "allowed) and an axis whose planes hold at most 192 cells, 'direct': one team launch "
      "runs flexible CG in FP64 (inner tolerance 1e-5, cap 100) preconditioned by a "
      "block-tridiagonal FP32 direct factor of the bottom (planes along one axis, explicit "
      "Schur-complement inverses), refactored once per operator change -- no host transfer "
      "on a GPU; on a host backend the team has min(8, team_size_max) threads and the FCG's "
      "reductions run on one lane, so the result does not depend on the thread count. "
      "Everywhere else the GraphAMG solve. 'direct' forces that engine and raises at the "
      "bottom solve, naming the failed condition, where it is not eligible; 'algebraic' "
      "forces GraphAMG (the A/B instrument). doc/vof_step_performance_design.md §13, §14 "
      "H-1.");
  cls.def(
      "set_pressure_line_relaxation",
      [](D& diag, bool enabled) { diag.s->setPressureLineRelaxation(enabled); }, nb::arg("enabled"),
      "Zebra line relaxation in the pressure multigrid (DEFAULT True). On a level whose "
      "finest axis has stopped coarsening (its extent no longer halves) while the other two "
      "keep coarsening, so that the coarse cells are more than 2:1 elongated across it, the "
      "smoother solves each line along that axis exactly (tridiagonal; cyclic for a periodic "
      "axis) instead of relaxing point by point, which damps the error modes the coarse grid "
      "cannot see. Every other level, and every grid without such a level, is unchanged; "
      "False restores the point red-black smoother everywhere, bit for bit. Under MPI a level "
      "whose line axis is split across ranks keeps the point smoother. Live. "
      "doc/mg_depth_policy.md Amendment A2.");
  cls.def(
      "set_pressure_vcycle_symmetric",
      [](D& diag, bool enabled) { diag.s->setPressureVcycleSymmetric(enabled); },
      nb::arg("enabled"),
      "The symmetric pressure V-cycle (DEFAULT True since doc/mg_depth_policy.md A3, WO-S3). "
      "True: every coarse-to-fine transfer adds the coarse parent's value to each fine cell "
      "(piecewise-constant prolongation, the transpose of the cell-average restriction up to "
      "the factor 2^k) and the smoothed coarsest level sweeps forward then in reverse, so the "
      "V-cycle is a symmetric positive definite preconditioner and the CG drivers keep their "
      "optimality on high-contrast and thin-slab grids. False: the pre-A3 trilinear "
      "prolongation and forward bottom, bit for bit (the A/B lever). Applies to every "
      "pressure driver (PCG, FCG, Chebyshev, the standalone V-cycle, the ghost projection's "
      "BiCGStab), single rank and distributed, FP64 and FP32. Live. "
      "doc/mg_depth_policy.md Amendment A3.");
  cls.def(
      "set_pressure_vcycle_precision",
      [](D& diag, const std::string& mode) {
        if (mode == "auto")
          diag.s->setPressureVcyclePrecision(0);
        else if (mode == "fp64")
          diag.s->setPressureVcyclePrecision(1);
        else if (mode == "fp32")
          diag.s->setPressureVcyclePrecision(2);
        else
          throw std::runtime_error("set_pressure_vcycle_precision: 'auto' | 'fp64' | 'fp32'");
      },
      nb::arg("mode"),
      "Precision of the V-cycle that preconditions the pressure PCG / FCG (single rank and "
      "distributed). 'auto' (DEFAULT since 2026-10-19): 'fp32' where eligible, else 'fp64', "
      "on a DEVICE backend (CUDA, HIP), with the FP64 restart below; on a host backend "
      "(OpenMP, Serial) 'auto' is 'fp64' -- the host saves too little (~5-8 %) to pay for "
      "FP32's risks. 'fp64': the FP64 V-cycle. 'fp32': FP32 face weights (the diagonal "
      "derived, never "
      "stored), FP32 iterates and flux-form arithmetic on every level above the bottom, the "
      "input scaled by an exact power of two; the Krylov matvec, dots, mean projection and "
      "stop test stay FP64 -- raises at the solve, naming the failed condition, where not "
      "eligible (an outflow face, an overlay solve, a non-bottom level with an odd inner "
      "dimension, a float outer operator, a face weight outside [1e-30, 1e30], the "
      "mean-removal scope 'all', a driver other than PCG / FCG); forced 'fp32' runs on host "
      "backends too. Distributed, FP32 runs down to the first telescope point (FP64 below) "
      "and an np=1 run is bitwise the single-rank run in every mode. Takes effect at once. "
      "Its weight rounding is not invariant under a non-power-of-two change of units, and it "
      "can cap where 'fp64' converges (hence the restart); pass 'fp64' for an FP64-only run. "
      "RESTART: under 'auto' a PCG / FCG solve whose FP32 attempt does not meet "
      "its stop test (a cap, a non-finite breakdown) is re-solved from x = 0 with the FP64 "
      "V-cycle -- the 'fp64' solve of the same right-hand side -- with one warning line, and "
      "'auto' then stays FP64 for the rest of the run (pressure_vcycle_precision() reads "
      "'fp64'); calling this setter again re-arms it. Forced 'fp32' is never restarted (the "
      "A/B mode). doc/vof_projection_distributed_design.md §4, §9 (the default). It saves ~19 "
      "% of the "
      "projection on a GPU "
      "(bubble column, RTX 5080), ~11 % on a CPU. Outside the CG drivers' rated contrast "
      "(density ratio above ~1e3 with walls) both precisions may cap; FP32 may cap earlier. "
      "doc/vof_projection_cost_design.md §4.");
  cls.def(
      "set_pressure_initial_guess",
      [](D& diag, const std::string& mode, int history) {
        int m = -1;
        if (mode == "auto")
          m = 0;
        else if (mode == "zero")
          m = 1;
        else if (mode == "projected")
          m = 2;
        else
          throw std::runtime_error("set_pressure_initial_guess: 'auto' | 'zero' | 'projected'");
        diag.s->setPressureInitialGuess(m, history);
      },
      nb::arg("mode"), nb::arg("history") = 2,
      "Initial guess of the pressure PCG / FCG solve of the projection (single rank and "
      "distributed). 'zero' "
      "(DEFAULT): x0 = 0. 'projected': the A-norm projection of the solution onto the span of "
      "the last `history` (1 or 2) projection solutions, x0 = c1 phi^n + c2 phi^(n-1) with "
      "the 2x2 Galerkin system of the current operator (never worse than x0 = 0 in the A-norm;"
      " 0 on non-fluid cells), and the stop stays relative to the ZERO-guess residual "
      "max|P b|, so the solve is never looser than the cold one. Raises at the solve, naming "
      "the failed condition, where not eligible (the Chebyshev driver, the ghost projection, "
      "set_pressure_warmstart(True)). 'auto': 'projected' where eligible, else 'zero', on "
      "every "
      "backend. 'projected' and 'auto' are opt-in, not the default (decided 2026-10-15). "
      "Distributed, the Gram system is allreduced, so every rank takes the same guess; an "
      "np=1 run is bitwise the single-rank run. The history "
      "restarts empty (a cold solve, then one increment, then two) after "
      "this call, a geometry change (set_solid, set_pressure_geometry, init_mpi, "
      "redistribute / rebalance_by_weights), a state write (set_field of u/v/w/p/phi, "
      "set_state, set_velocity), a "
      "driver change, and a failed or capped solve. RESTARTS: no history is checkpointed, so"
      " a run restarted from saved fields takes those two cold-history steps and agrees with "
      "the uninterrupted run to the solver tolerance, not bitwise. "
      "doc/vof_projection_cost_design.md §5, doc/vof_projection_distributed_design.md §6.");
  cls.def(
      "set_pressure_prefactor_hoist",
      [](D& diag, bool on) { diag.s->setPressurePrefactorHoist(on); }, nb::arg("on"),
      "Switch the side-instance bottom prefactor (doc/vof_gpu_coarse_tail_design.md §5.6, "
      "default ON) off: the variable-rho coefficients are then built in the projection and "
      "the direct bottom factored lazily in the bottom solve. Same state bit for bit; for "
      "tests and timing comparisons.");
  cls.def_prop_ro(
      "num_pressure_prefactors", [](D& diag) { return diag.s->pressurePrefactorCount(); },
      "Side-instance bottom factors launched so far (0 where the prefactor is ineligible: "
      "host backends, MPI, constant density, a balanced-force or Chebyshev configuration).");
  cls.def(
      "pressure_vcycle_precision",
      [](D& diag) { return std::string(diag.s->pressureVcycleFp32() ? "fp32" : "fp64"); },
      "The precision ('fp32' or 'fp64') the last pressure solve's V-cycles used "
      "(set_pressure_vcycle_precision); 'fp64' before any FP32 solve.");
  cls.def(
      "set_pressure_graph_amg", [](D& diag, bool on) { return diag.s->setPressureGraphAmg(on); },
      nb::arg("on"),
      "Solve the pressure MG's coarsest level with an agglomerated mesh-agnostic algebraic "
      "multigrid (core GraphAMG), decomposition-agnostic: with levels=1 this gives a "
      "mesh-independent pressure solve that works under a WEIGHTED ORB (where the geometric "
      "coarse levels can't cleanly coarsen). Applied at the next set_solid.");
  cls.def(
      "set_exact_crossings",
      [](D& diag, const std::vector<double>& t) { return diag.s->setExactCrossings(t); },
      nb::arg("t"),
      "Analytic-SDF capability: exact wall-crossing fractions overriding the "
      "linear-interpolated theta in the momentum cut-cell overlay AND the ghost-projection "
      "closures. Flat array of 9*nx*ny*nz values, blocks [(c*3+k)]: component c's staggered "
      "point at inner cell i toward its +k neighbour; NaN = no crossing. Call BEFORE "
      "set_solid (a later call raises); empty list clears. Single-rank, staggered momentum "
      "placement.");
  cls.def(
      "set_openness_override",
      [](D& diag, const std::vector<double>& ox, const std::vector<double>& oy,
         const std::vector<double>& oz) { return diag.s->setOpennessOverride(ox, oy, oz); },
      nb::arg("ox"), nb::arg("oy"), nb::arg("oz"),
      "Analytic-SDF capability: exact face-aperture (openness) fields for the cut-cell "
      "projection, overriding the sampled-SDF openness. Inner nx*ny*nz arrays, x-fastest; "
      "ox[i] = fluid fraction of the -x face of cell i. Call BEFORE set_solid (a later call "
      "raises); empty ox clears. Single-rank.");
  cls.def(
      "set_ghost_projection",
      [](D& diag, bool on, int matrixOrder, int rhsOrder) {
        return diag.s->setGhostProjection(on, matrixOrder, rhsOrder);
      },
      nb::arg("on"), nb::arg("matrix_order") = 2, nb::arg("rhs_order") = 2,
      "QUARANTINED 2026-08-18 (verification only, unsupported): superseded by the gauge-exact "
      "collocated scheme, which matches its accuracy at 5-6x lower cost. Kept as the "
      "independent second discretization behind the cross-IBM physics gate. Enabling it on the "
      "collocated grid silently selects the plain face map, since it owns the operators the "
      "gauge-exact scheme replaces. EXPERIMENTAL directional ghost-cell projection (second "
      "staggered IBM): point-based FD "
      "divergence with wall-anchored directional closures instead of the openness-weighted "
      "cut-cell projection; solved by MG-preconditioned BiCGStab. Call BEFORE set_solid (a "
      "later call raises). "
      "Closure orders (1=linear, 2=quadratic): (matrix_order, rhs_order) = (2,2) full "
      "quadratic 13-point matrix; (1,1) linear 7-point; (1,2) mixed/deferred — 2nd-order "
      "steady constraint on a 7-point matrix. Collocated: the same closures/matrix on the "
      "face-averaged field, plus a directional (one-sided 2nd-order) cell gradient for the "
      "-grad(P) predictor and cell correction; requires face_interp 0. v1: single-rank, "
      "periodic + IBM, stationary walls; incompatible with "
      "porous/variable-rho/domain-BC/Chebyshev.");
  cls.def(
      "pressure_mg_level_ratios", [](D& diag) { return diag.s->pressureMgLevelRatios(); },
      "The pressure multigrid's per-level coarsening ratio, one (rx, ry, rz) per level "
      "(read-only).\n\n"
      "On an isotropic domain every coarsenable axis halves, as it always has. On an "
      "ANISOTROPIC one (extent giving different spacings per axis) the hierarchy coarsens "
      "the FINEST coarsenable axis and defers one that is already at least "
      "set_multigrid_aspect_threshold (default 2) times coarser, so the coarse operators stay "
      "close "
      "to isotropic and the point smoother keeps its rate -- e.g. cells (N, 2N, N/2) on a "
      "cube gives (1,2,1), (2,2,1), (2,2,2), ... Empty until set_solid/"
      "set_pressure_geometry has built the operator. See flow/doc/anisotropic_metric.md "
      "\u00a75.");
  cls.def(
      "last_superficial_velocity_shift",
      [](D& diag) { return diag.s->lastSuperficialVelocityShift(); },
      "The uniform velocity shift set_superficial_velocity applied at the end of the last "
      "step() (physical velocity units; 0 when it is off). Reading it synchronises with the "
      "device.");
  cls.def(
      "last_balanced_force_iterations",
      [](D& diag) { return diag.s->lastBalancedForceIterations(); },
      "Pressure-driver iterations of the last step's balanced-force solve "
      "(set_balanced_force_projection); 0 when it is off, the force divergence was zero, or "
      "the previous split already met the stop.");
  cls.def(
      "last_balanced_force_failed", [](D& diag) { return diag.s->lastBalancedForceFailed(); },
      "Whether the last step's balanced-force solve FAILED (driver breakdown or a non-finite "
      "result). A failed solve keeps P and p_balanced unchanged for that step; the first one "
      "warns on stderr.");
  cls.def(
      "balanced_force_failures", [](D& diag) { return diag.s->balancedForceFailures(); },
      "How many balanced-force solves have failed over the solver's lifetime (see "
      "last_balanced_force_failed).");
  cls.def(
      "anderson_accelerator",
      [](D& diag, int window, double mixing) {
        return new BoundAccelerator<Grid>(*diag.s, window, mixing);
      },
      nb::arg("window") = 5, nb::arg("mixing") = 1.0, nb::rv_policy::take_ownership,
      nb::keep_alive<0, 1>(),
      "A new Anderson accelerator of this solver's steady march "
      "(doc/steady_acceleration.md; the public driver is peclet.flow.march_to_steady). "
      "window in [1, 8] (default 5), mixing in (0, 1] (default 1.0). Raises on an "
      "unsupported configuration (a collocated scheme other than 'ghost', VoF, phase change, "
      "scalars, porous, variable properties, drag, cell forces, moving scenes, superficial "
      "velocity, pressure warm start, the balanced-force projection) and when the history, "
      "(2m+3) * n_fields * 8 * n_padded bytes, does not fit. Collective under MPI.");
  cls.def(
      "last_pressure_iterations", [](D& diag) { return diag.s->lastPressureIterations(); },
      "Return the pressure-solver iteration count from the last step().\n\n"
      "A solve that BROKE DOWN (non-finite preconditioner output) reports the iteration "
      "CAP, so the usual rule-3b 'a capped pressure solve invalidates the run' check sees "
      "it; pressure_solve_failed() distinguishes the two.");
  cls.def_prop_ro(
      "num_pressure_chebyshev_restarts",
      [](D& diag) { return diag.s->numPressureChebyshevRestarts(); },
      "How many Chebyshev pressure solves were redone on cold spectral bounds since "
      "construction. Under variable density the bounds are re-estimated every step "
      "from the previous estimate's power iterates (5 + 5 iterations instead of "
      "15 + 15 from the right-hand side); a solve on such bounds that reaches the "
      "cap, or whose residual after 3 iterations exceeds the initial one, is redone "
      "on a cold estimate and counted here; the abandoned attempt's V-cycles count in "
      "last_pressure_iterations().");
  cls.def_prop_ro(
      "num_pressure_chebyshev_growth_trips",
      [](D& diag) { return diag.s->numPressureChebyshevGrowthTrips(); },
      "How many Chebyshev pressure solves on cold spectral bounds (the main projection's "
      "and the balanced-force pre-projection's) saw their residual rise above the initial "
      "one, since construction. Chebyshev converges for every eigenvalue below the sum of "
      "its two bounds, so a growing residual means lambda_max was underestimated: the bounds "
      "are re-estimated with a bracketing margin (1.15 on lambda_max) and the solve is "
      "redone once from its starting iterate; the first such case prints a stderr notice. "
      "Expected 0; a solve that does not trip is unchanged bit for bit. "
      "doc/mg_depth_policy_log.md 2026-10-30.");
  cls.def_prop_ro(
      "num_pressure_unconverged_solves",
      [](D& diag) { return diag.s->numPressureUnconvergedSolves(); },
      "How many MG-PCG / flexible-CG pressure solves (main and balanced-force projection) "
      "since construction ended UNCONVERGED: capped, broken down, or stopped on their "
      "recurrence residual while the TRUE residual max|b - A x| (the stop test's norm and "
      "mean removal) exceeded max(10 rtol, 1e-8) * r0 -- the floor keeps a round-off-limited "
      "solve at a near-machine rtol silent. Each is also reported on "
      "stderr (the first, then the 10th, 100th, ...). Under the trilinear V-cycle "
      "(set_pressure_vcycle_symmetric(False)) above a density ratio of ~1e3 the preconditioner "
      "turns indefinite and both CG drivers can stop 'converged' on a field that is not a "
      "projection; the default SPD V-cycle keeps them healthy there. The check reads the "
      "result only: fields and iteration counts are unchanged. "
      "Identical on every rank.");
  cls.def(
      "last_pressure_true_residual", [](D& diag) { return diag.s->lastPressureTrueResidual(); },
      "The last MG-PCG / flexible-CG pressure solve's TRUE residual max|b - A x| divided by "
      "its stop tolerance rtol * r0 (r0 the initial residual, or the balanced-force "
      "projection's full right-hand side): <= 1 means the exit really meets the tolerance; "
      "above max(10, 1e-8 / rtol) the solve counts in num_pressure_unconverged_solves. "
      "Unit-free. Not updated "
      "by Chebyshev or ghost-projection (BiCGStab) solves; 0 before the first CG solve and "
      "after one with a zero right-hand side.");
  cls.def(
      "pressure_solve_failed", [](D& diag) { return diag.s->pressureSolveFailed(); },
      "Did the last pressure solve break down on a non-finite recurrence scalar (a "
      "preconditioner or operator that produced NaN/Inf)?\n\n"
      "The driver then zeroes the correction and continues -- so the projection was handed "
      "NOTHING for that step and the run is invalid. Until 2026-09-04 the only trace was one "
      "line on stdout ('preconditioner produced non-finite z') while "
      "last_pressure_iterations() reported 0, i.e. a perfectly healthy-looking solve "
      "(measured: examples/pore-scale-imbibition, a 3.1-cell throat). Set "
      "set_pressure_strict(True) to raise instead.");
  cls.def(
      "last_step_timers",
      [](D& diag) {
        nb::dict d;
        d["step"] = diag.s->lastStepSeconds();
        d["predictor"] = diag.s->lastPredictorSeconds();
        d["momentum"] = diag.s->lastMomentumSeconds();
        d["projection"] = diag.s->lastProjectionSeconds();
        d["pressure_allreduce"] = diag.s->lastPressureAllreduceSeconds();
        d["pressure_allreduce_count"] = diag.s->lastPressureAllreduceCount();
        d["momentum_sweeps"] = diag.s->lastMomentumSweeps();
        return d;
      },
      "Per-phase wall times (seconds, THIS rank, device-fenced) of the last step(): "
      "'predictor' "
      "(ghost fills + RHS/advection/stencil builds), 'momentum' (implicit-diffusion solves), "
      "'projection' (cut-cell pressure projection), 'step' (whole step). "
      "'pressure_allreduce'/'pressure_allreduce_count' = time in / number of global reductions "
      "(MPI_Allreduce) inside the pressure solve — the latency-bound term of the distributed "
      "solve (0 on a single rank).");
}

}  // namespace peclet::flow::bindings
