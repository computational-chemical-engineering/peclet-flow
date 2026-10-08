# CLAUDE.md reference — detail moved out of CLAUDE.md, 2026-10-08

`CLAUDE.md` was compacted from ~65 KB on 2026-10-08. The text below is what it said, moved here
verbatim (light edits only: stale statements corrected, link targets re-rooted for `doc/`). Each
rule it contains is still stated, shorter, in `CLAUDE.md`; this file carries the measured numbers,
the rationale and the long API notes. **Paths in backticks are relative to the flow repo root**, as
in `CLAUDE.md`. Rewrite in place when the code changes; do not let it become a diary.

Sections: §1 Build, §2 Layout, §3 Settled decisions — the long forms, §4 Conventions and hard rules, §5 Python API and units, §6 Pressure solve, §7 Collocated solver, §8 Domain boundary conditions, §9 Geometric VoF, §10 Cut-cell scalar transport, §11 Steady marches, §12 Open items — long forms. The velocity-solver selection rule and its evidence moved to [`velocity_mg_plan.md`](velocity_mg_plan.md) (last section).

## 1. Build

*The host flags, the mpiexec trap, the G.8 single instantiation and why ccache cannot work on a device prefix.*

**Host backends compile flow with `-ffp-contract=off`** (always; no compiler default may fuse an
FMA), and `-DPECLET_FLOW_HOST_ARCH=<arch>` (default empty = generic x86-64) adds `-march=<arch>` —
`native` on a dev box, `znver4` on Snellius genoa, `znver3` on the workstation. With contraction
off the wider ISA vectorizes without changing a bit (state hashes and the bubble-column dumps are
identical across generic, contract-off and `-march=native`). Neither flag reaches a CUDA/HIP build,
and PyPI wheels stay generic (`doc/vof_step_performance_design.md` §5.1). On aarch64 contract-off
does change bits against the old compiler default: it makes host results ISA-independent.

**Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`.** FindMPI may pick ParaView's bundled `mpiexec` off
`PATH`, which launches the OpenMPI-linked binaries as singletons — every `*_np4` then silently runs
four independent np = 1 jobs.

**`Solver<Grid>` is compiled ONCE per grid, not once per consumer** (QUALITY_PLAN G.8, 2026-09-11).
`src/flow_solver_staggered.cpp` and `src/flow_solver_colocated.cpp` hold the explicit instantiations;
`cmake/PecletFlowSolver.cmake` builds them into a static library that the module and all 45 tests
link, and `flow_ibm.hpp` ends with the matching `extern template` declarations. It also includes the
thirteen domain headers **only in those two TUs** (`PECLET_FLOW_INSTANTIATING`), so everything else
sees declarations alone and an edit to one domain header rebuilds four objects, not forty-nine. The
one rule that follows: a member *template* of `Solver` that a test or the bindings calls must be
defined in `flow_ibm.hpp` itself — an explicit instantiation of the class does not cover member
templates, so a definition left in a domain header would not link. Two consequences for
anyone editing the build: a target that includes `flow_ibm.hpp` must link `peclet_flow_solver`
(tests/kokkos links every target in the directory, so a new single-rank test needs no edit at all),
and it must link the variant matching its `PECLET_FLOW_MPI` — which is why the macro is a PUBLIC
property of `peclet_flow_solver_mpi` and no test declares it itself. Adding a third grid policy means
adding a third instantiation TU beside those two.

**`ccache` is opt-in, never forced, and HOST-ONLY**: pass `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` at
configure time on a `host-openmp` / `host-serial` prefix (measured on this tree: the solver library
cold 69 s, warm 0.3 s, 100 % direct hits). Nothing turns it on, so a tree configured without it
behaves exactly as before. **It cannot work on a CUDA/HIP prefix, and the reason is structural:**
Kokkos routes device compilation by putting `kokkos_launch_compiler <nvcc_wrapper> <c++>` in the
global `RULE_LAUNCH_COMPILE`, and that script redirects only when the executable immediately
following it is the compiler it was handed. A compiler launcher is expanded exactly there, so the
script sees `ccache`, declines to redirect, and plain `c++` gets nvcc's flags
(`unrecognized command-line option '-arch=sm_120'`) — at the end of a long build. The CMake now
rejects the combination at configure time with that explanation, so do not re-attempt it blind.

## 2. Layout

*Which member definitions live in which domain header.*

- `src/flow_ibm.hpp` — `template <class Grid> class Solver` (`IbmSolver` = `Solver<Staggered>`):
  includes, the nested structs, every member DECLARATION (with its docstring) and the whole state
  block (QUALITY_PLAN G.1: split 2026-09-11, 11886 -> 4046 lines). The out-of-line member
  DEFINITIONS live in thirteen domain headers included at the bottom — since G.8, only in the two
  instantiation TUs (each reopens
  `namespace peclet::flow`; declarations + state never move):
  `flow_ibm_core.hpp` (allocation, rho/mu/dt + driver setters, stencils/ghosts/advection inputs,
  the momentum-solve smoother family, reductions, gather/scatter, the field registry),
  `flow_ibm_project.hpp` (`step()`, the `project()` stages, the `buildRhs` family,
  `applyFaceAcceleration`), `flow_ibm_scene.hpp` (scene instances + motion upload, wall velocity,
  wall-flux divergence, `setSolidFromScene`), `flow_ibm_geometry.hpp` (`setSolid`/`setSolidDevice`
  + its nine stages, pressure geometry), `flow_ibm_hydro.hpp` (hydro force/torque, reaction-budget
  terms, wall probes), `flow_ibm_vof.hpp` (colour field, cut-cell VoF, contact angle, VoF blocks,
  curvature, surface tension / CSF RHS, advection, `stepAdaptive`), `flow_ibm_phase_change.hpp`
  (interface area, mass flux, energy transport, budgets, the `pc*` machinery),
  `flow_ibm_closures.hpp` (property closures/modes, porous continuity, drag, prop/eps ghosts),
  `flow_ibm_scalars.hpp` (`addScalar`/`setScalarBc`/`advanceScalars` + BC application),
  `flow_ibm_scalars_cutcell.hpp` (the cut-cell scalar path: geometry build, setters, assembly + solve,
  diagnostics),
  `flow_ibm_bc.hpp` (domain BCs/profiles, `pressureBcGhost`, `fillVelGhosts*`,
  `setupBcDiffusion`), `flow_ibm_mpi.hpp` (`initMpi`, `redistribute`, `rebalanceByWeights`, the
  post-repartition field-resize passes; `#ifdef PECLET_FLOW_MPI`-guarded), `flow_ibm_diagnostics.hpp`
  (state getters, divergence probes, timers, the outflow/backflow census). `src/flow_bindings.cpp`
  — the nanobind module. `src/flow_solver_staggered.cpp` / `src/flow_solver_colocated.cpp` — the
  two explicit instantiations of the class, the only TUs that compile it (see "Build"), and of
  `AndersonAccelerator<Grid>` (`src/anderson_accelerator.hpp`, the steady-march accelerator).
- `src/mac_cutcell_mg.hpp` (`CutcellMG`, pressure MG), `src/mac_velocity_mg.hpp` (`VelocityMG`),
  the `src/mac_*.hpp` operators, `src/cut_cell_ibm.hpp` (the Robust-Scaled overlay: `poly_*`,
  K/M/X/Nbc/R, `D_rescale`), `src/staggered_advection.hpp` (`sadv::advect`),
  `src/gauge_exact_gradient.hpp` (`gpCenterGrad`), `src/ghost_projection.hpp`.
- `src/vof/` — the VoF stack (below). Its container-free kernels were promoted to
  `peclet::core::vof` in `../core`, so `src/vof/{plic,curvature,cutcell,wetting}.hpp` are thin
  includes. **New container-free VoF math belongs in `core`**; the drivers stay here.
- `tests/{kokkos,kokkos_mpi,python,regression,study}`, `scripts/`, `doc/` + `doc/history/`.

## 3. Settled decisions — the long forms

*The evidence behind four of the compressed bullets in CLAUDE.md.*

- **Collocated pressure and forces go INSIDE the implicit momentum predictor — never a face
  acceleration added after the viscous solve (the Basilisk `centered.h` "kick").** Added after
  the implicit solve, the projection removes the lagged pressure exactly, so the velocity update
  is non-incremental. That gives Chorin's dt-dependent steady state, and with the rotational
  update an explicit pressure diffusion growing like −12κ·dt/(ρh²) per step (measured −12.0000;
  it capped V8 at density ratio ~100). Variable density keeps balance through the mass-adjoint
  pair (the ρ-weighted face-average pressure force + the momentum-weighted centre→face map) and
  the optional balanced-force projection — not through face placement. The guard is the
  stability/dt-independence gate on every collocated path, not a grep. Register: suite-wide
  "Collocated forces stay in the implicit predictor"; design: flow `doc/collocated_varrho_forces.md`.
- **The force on a body is the discrete REACTION — `hydro_force_torque_reaction()` — never the
  traction integral** (USER-approved 2026-09-30). The traction (`diagnostics.hydro_force_torque_traction()`)
  differences the velocity across the wall with a central difference and under-reads the viscous
  part: traction/reaction 0.685–0.730 over φ 0.008–0.45 and N 24–128, at every resolution; its
  spinning-sphere torque reads 0.59–0.72 of the exact value where the reaction reads 1.019–1.027.
  It is a diagnostic (it carries the pressure/viscous split, and runs where the reaction refuses:
  collocated, porous, variable properties, domain BCs). Its old public name `hydro_force_torque()`
  is DEPRECATED and warns; do not hand it out as "the force". A one-sided difference to the wall
  over θ is a closed dead end (17× too large as θ → 0); the planned fix is under "Open items".
  Register: coupling "The public force API returns the reaction".

- **COLLOCATED momentum advection uses the PROJECTED, divergence-free face field** `uf_/vf_/wf_`
  of the previous step's projection — never the un-projected cell→face average ½(u_i+u_j) (the
  phase-2 form, kept only as the ablation `diagnostics.set_uf_advection(False)`). It is
  `doc/flow_colocated_plan.md` §1 step 3, the Almgren–Bell–Colella prescription, and what the FOU
  operator's conservative row sum needs; it closed the last uniform-grid difference against
  `peclet.amr` (2.5e-4 → 1.9e-11 over 20 NS steps). The implicit FOU operator and the explicit
  (SOU−FOU) deferred correction MUST read the same field — one predicate, `ufAdvVelocity()`.
  Landed 2026-09-21; evidence and limits in `doc/uf_advection.md`. Staggered is unaffected (its
  stored velocity already IS the projected face velocity) and proved bit-identical.
- **Operator storage is double by default** (`PECLET_FLOW_OPERATOR_DOUBLE=ON` since 2026-09-11).
  Float storage silently breaks A·1=0 at high MG contrast — it fails without an error, so it will
  not announce itself; that is why it is no longer the default. Opting out costs correctness on
  dense beds, not just accuracy. The double-*diagonal* fallback is a different thing and stays
  retired (it converges to the float-face operator; 65× worse on divergence).

## 4. Conventions and hard rules

*Env knobs, the two API tiers, and the call-order rules with the init_mpi incident.*

- **No environment variable changes a result** (`../docs/QUALITY_PLAN.md` D3, executed 2026-09-08
  in `ad917b1`). Every numerics-changing or algorithm-selecting read became a per-solver setter
  with the old unset behaviour as its default, or was deleted with the ablation it served; the
  process-global statics went with them. Only debug *printing* reads remain, and
  `tests/python/test_no_env_knobs.py` (ctest `no_env_knobs`) fails if a new one appears in `src/`.
  The old-name → setter table is in the `ad917b1` message. Do not add one, and do not document a
  workflow that sets one.
- **Two API tiers (QUALITY_PLAN D2, since 2026-09-10).** `Solver` / `SolverColocated` carry the
  PUBLIC surface — what a user of single-phase, VoF, porous, moving-geometry, scalar-transport or
  thermal flow needs to set up, run and read out a run (~140 members). Everything a developer uses
  to inspect, profile, ablate or tune — the `*_diagnostics/_stats/_census/_budget/_ledger/_probe/
  _timing` families, `last_*`, solver tuning beyond the driver selection, the ablation switches,
  the zero-copy/MPI internals `field_view`/`exchange_field`/`rebalance_by_weights` — lives on
  `s.diagnostics` (a view holding a reference to the solver, one class per grid). Integer codes are
  strings everywhere (`set_domain_bc('-x', 'inflow', …)`, `set_advection_scheme('koren')`,
  `add_scalar(scheme='koren')`, `set_scalar_bc(name, '+z', 'dirichlet', v)`), and every on/off pair
  is one setter with a leading `enabled` bool (`set_phase_change_thermal(False)`).
- **Call order is enforced where a wrong order gives a wrong result.** The settings that are folded
  into the operators when the geometry is built — `set_domain_bc`/`set_domain_bc_profile`,
  `diagnostics.set_aperture_order`, `set_exact_crossings`, `set_openness_override`,
  `set_fluid_only_constraint`, `set_ghost_projection` / `set_collocated_scheme('ghost')` — raise
  after `set_solid`/`set_pressure_geometry`/`set_solid_from_scene`; `set_decomposition` and
  `diagnostics.set_comm_avoiding` raise after `init_mpi`. **`init_mpi` itself raises after the
  geometry** (2026-09-24): the geometry builds the pressure and velocity multigrids for the
  partition in force when it runs, so built first they were single-rank and every rank ran its own
  block-periodic pressure solve — PCG converged, nothing failed, and the velocity kept each block's
  mean divergence (4.0e-2 of max|u| 0.36 after one step at np = 4; exact at np = 1, whose block is
  the domain, which is why it hid). The order is `Solver(*size)` → `init_mpi` → geometry;
  `test_cellforce_mpi` gates it. `set_rho`/`set_mu`/`set_dt` may be
  called at any time (a change after the geometry rebuilds the momentum operator at the next step;
  under a physical domain the FIRST `set_rho` and the FIRST `set_dt` pin the reference scales).
  `set_decomposition(levels, max_imbalance)` and `flow.mpi_block(..., levels=, max_imbalance=)`
  must get the **same** values. Select the pressure driver **last**: `set_property_model("rho", …)`
  fires the density mode, which re-selects Chebyshev and discards an earlier choice.

## 5. Python API and units

*The physical-units surface, the internal-units trap, anisotropic cells.*

Everything in and out is in one consistent system of the caller's choosing — properties, `dt`,
forces, boundary velocities and profiles, `sigma`, the slip length, the SDF and scene coordinates
in; `get_u/v/w`, `get_p`, `get_uf`, `vof_curvature()`, `max_open_divergence()`, the hydro force and
torque out. **The scalar/energy surface too** (since 2026-10-02 — U3 had missed it): `add_scalar`'s
diffusivity, the phase-change densities, latent heat, conductivities, heat capacities, IHTR
resistance, prescribed mass flux and divergence source, and the closure parameters on
`rho`/`mu`/`force_*` (Boussinesq included) go in physical; `vof_interface_area` and the
phase-change diagnostics/budget come out physical. A transported scalar (temperature,
concentration) is **never rescaled**. Constants are kept verbatim and re-derived when a scale is
pinned, so their order against `set_rho`/`set_dt` does not matter; the two FIELD setters
(`set_mass_flux*`, `set_divergence_source`) convert once and therefore raise under an extent until
`set_rho`/`set_dt` have run. The operator-flux `mdot` (`set_phase_change_mdot_operator`,
`energy_order(2)`) without `set_phase_change_energy` raises under an extent: its constant-D flux
carries an implicit ρc_p of one internal unit.

Under MPI the constructor takes **this rank's** block; pass the global grid as
`global_cells` and the global box as `extent`/`origin`. `extent=None` keeps **cell units**
(spacing 1, origin 0) and is bit-identical to the pre-2026-09 code.

**The trap.** The solver computes on the unit lattice; the metric is folded into constants at the
API boundary (`Solver::UnitScales` in `src/flow_ibm.hpp`, derivation in the comment above it), with
`hRef = min h`, `rhoRef` = the first `set_rho` and `tRef` = the first `set_dt`, so stored float
operator coefficients stay O(1) in any unit system. **The raw field registry —
`get_field`/`set_field` and `diagnostics.field_view`/`diagnostics.exchange_field` — hands out those
INTERNAL arrays**, unlike `get_u`/`get_p`, which convert. A driver writing `force_x` or `drag_beta` directly (CFD-DEM does)
must convert with `s.unit_scales` — and so must anyone reading `mdot`, `pc_source` or `div_source`
(`mass_flux_to_internal`, `divergence_to_internal`).

**Anisotropic cells** work on both solvers, single phase and VoF
([`doc/anisotropic_metric.md`](anisotropic_metric.md),
[`doc/anisotropic_vof.md`](anisotropic_vof.md)). Spacings agreeing to 1e-12 relative are snapped
to one, so an isotropic domain stays bit-identical however its extent was written; both multigrids
defer an axis already `theta` (2.0) times coarser than the finest coarsenable one, without which a
stretched grid does not solve at all. Two guards remain: the CFD-DEM coupling driver refuses an
anisotropic box (its *particle model* has no single length to form `Re_p` — not a metric gap), and
`scripts/check_decomposition.py` still models the isotropic coarsening rule.

## 6. Pressure solve

*Contrast, operator precision (and the 2026-09-11 battery record), the bottom engines, depth, telescoping, Repartition, the aligned weighted rebalance, redistribute, decomposition.*

- **High coefficient CONTRAST makes the V-cycle preconditioner indefinite** and both CG drivers cap
  above density ratio ~10³ — the cause is the arithmetic coarsening of the face coefficient, not
  float storage. Only Chebyshev is healthy there; coefficient-aware coarsening is the open fix.
- **Operator STORAGE precision is a separate axis, and a typed CMake option (QUALITY_PLAN G.6).**
  `MReal` (`mac_cutcell_mg.hpp`) types the pressure hierarchy and, via `IbmSolver::FV` and
  `IbmOverlay` (the cut-cell overlay: `cut_cell_ibm.hpp`'s `poly_*`/`ibmFillEntry`/
  `ibmModifyStencil`, templated on `Real`), the momentum stencil and the closure factors K/M/X/
  Nbc/R/D_rescale — and, since 2026-09-11, `GpOverlayMReal` (`ghost_projection.hpp`'s
  `GpOverlayT`/`GpOverlayReal<Real>`, aliased in `mac_ibm.hpp` next to `IbmOverlay`), the
  ghost-projection overlay `IbmSolver::gpOv_` behind the AUTO-default `'ghost'` collocated scheme.
  `option(PECLET_FLOW_OPERATOR_DOUBLE)` makes all of it double at +12 % step time and is ON by
  default since 2026-09-11 (`dc6ae78`); `-DPECLET_FLOW_OPERATOR_DOUBLE=OFF` (`pip install . -C
  cmake.define.PECLET_FLOW_OPERATOR_DOUBLE=OFF`) is the float storage, bit-identical to before G.6. Float rounding breaks `A·1 = 0`, and on a high-contrast
  bed the residual floors and then **rebounds** — the run is **invalid, not degraded**
  (`../docs/SCALING_ISSUES.md` #1). `tests/python/test_no_float_operator_casts.py` (ctest
  `no_float_operator_casts`) fails on a new hard `(float)` cast / `float`-typed operator view in
  `src/` outside its allow-list (a `// PRECISION-EXEMPT: <reason>` marker, or one of the whole-file
  exemptions it documents). **`GpOverlay`'s two remaining `PRECISION-EXEMPT` casts are a real,
  cross-repo gap, not dead code:** `buildGpOverlay`'s per-face SDF/theta samples still narrow to
  `float` before reaching `peclet::core::scheme::gpFillRow`/`gpClassifyFace`
  (`ghost_closure.hpp`), which are float-hardcoded — templating them needs the same
  `Real`-templating `IbmOverlayT`'s siblings got, done in `core`, out of this repo's scope.
  `PECLET_FLOW_OPERATOR_DOUBLE=ON` therefore widens `gpOv_`'s SoA storage (so it no longer
  round-trips a computed value through a *second*, separate float narrowing) without changing the
  ghost closure's own arithmetic. **Double is the DEFAULT since 2026-09-11** and the battery is
  clean at it: 155/155 non-bench on host-openmp. An earlier reading of "151/155, four float-tuned
  gates" (`build_q_dbl`, 2026-09-11) is superseded — two of those four (`vof_bc_mpi_np2`,
  `vof_bc_mpi_np4`) do not reproduce, and the other two were defects in the GATES, not tolerance
  noise, each fixed with its cause recorded in the commit:
  `verify_lid_cavity_sdflow` measured steady state by the plane-MEAN of u, which is near zero by
  symmetry and therefore mostly pressure-solve round-off, so it halted the march at step 400 instead
  of 650 and compared an unconverged field to Ghia; it now measures the largest velocity change on
  the plane, and float and double then agree to four digits (both 1000 steps, u_rms 0.0067, v_rms
  0.0037, centreline min u −0.2124). `vardensity_mpi_np4` demanded EXACT Chebyshev V-cycle-count
  equality between the distributed and single-rank solves, which is not well posed above np = 1
  because the stopping test reads a global reduction whose summation order differs; the count gate
  is now exact at np = 1 and ±1 above it, while the answer tolerances are untouched.
- `set_pressure_bottom("auto" | "smoother" | "agglomerated")` — **`"auto"` is the default**: it
  agglomerates the coarsest level into a global, decomposition-independent operator and solves it
  exactly whenever that grid exceeds `set_pressure_bottom_extent` (4) cells on any axis. Porous and
  variable-ρ rebuild it every step; avoid `auto` with a badly-factored grid there.
- **Two engines solve an agglomerated bottom** (`doc/vof_step_performance_design.md` §13): the
  host GraphAMG (every host backend, every multi-rank run, every ineligible case) and, on a GPU
  backend, the **device bottom** — one single-team launch of flexible CG in FP64 on the bottom's own
  operator, inner tolerance 1e-5 (relative, ∞-norm; E3), cap 100, preconditioned by a
  **block-tridiagonal FP32 direct factor** (`src/mg_bottom_direct.hpp`: planes along one axis, each
  plane's Schur complement inverted explicitly, the null space lifted by an exact plane-local
  rank-1 term per component; refactored by its own launch once per operator change, keyed on a host
  flag), no host transfer. A float factor is admissible only because A·1 = 0, the mean projections
  and the stopping residual stay FP64 — an unrefined float bottom is not. Eligible: single rank, the
  singular operator (no outflow face), `auto`/`agglomerated` bottom, ≤ 8192 bottom cells with 1–64
  fluid components (labelled on the device at geometry time; solids keep x = 0), and an axis whose
  planes hold ≤ 192 cells (`directBottomIneligible()` names the first failure).
  `diagnostics.set_pressure_bottom_solver('auto' | 'direct' | 'algebraic')` A/Bs them on one build;
  `'direct'` raises where ineligible. Host results are untouched by it; on the GPU it is a recorded
  numerics change. B1's V-cycle-preconditioned engine (`'geometric'`) is retired (history: the WO-6 / WO-11 commits).
- **Depth follows the factors of two, per axis.** An axis coarsens only while it stays even, so an
  **odd dimension never coarsens at all** (384×128×256 → 5.0 pressure iterations/step, ×255 →
  16.2), and under MPI only if *every rank's block* is even on it. **Telescoping is the default**
  (`set_pressure_telescope`): a level that cannot coarsen in place merges ORB siblings onto fewer
  ranks and continues to 3³. The in-place requirement on intermediate levels is the top open item
  at scale (`../docs/DECOMPOSITION_AND_MULTIGRID.md` §2.8).
- **The stage machinery is core's** (2026-09-24, S4 of `../amr/docs/amr_mg_core_boundary.md`,
  proved byte-identical): the depth search is `peclet::core::decomp::chooseStageTarget` with
  flow's lift rule `CutcellMG::teleLiftable` as the predicate (the forced test telescope is
  `shallowestLiftableMerge`), the communicators `makeStageComm`, the gather / scatter one
  `RedistributeTopology` per stage. `Telescope` stays flow's per-level record; the scatter's ADD
  and the WO-R2 outflow ghost-plane gather stay here. Do not re-inline any of it.
- **A weighted level-0 decomposition gets Repartition stages** (S5). After
  `diagnostics.rebalance_by_weights` the ORB has odd splits, and the sibling-merge search alone
  collapsed the level at the shallowest odd split onto ONE rank when the root split was odd
  (`../docs/SCALING_ISSUES.md` #2 TRAP: projection ×2.25–2.6, iterations unchanged). The Solver
  now calls `CutcellMG::setRepartition(true)` for a weighted `dec0` only, which caps the block a
  stage hands a rank at the largest level-0 block and repartitions the level onto a proportional
  ORB on `np_L` ranks instead (`[mg]` trace: `-> REPARTITION`). A run that never rebalances takes
  none (maxBlockCells = 0), and a Repartition computes the same bits as the collapse it replaces
  (coarse arithmetic is pointwise) — `test_telescope_mpi` gate D. The escapes the old source
  comment recommended (`nLevels = 1`, the GraphAMG bottom) do not escape; do not re-recommend them.
- **`rebalance_by_weights` builds the ALIGNED weighted ORB and RETURNS the alignment** (S5, landed
  2026-09-25). The partition is core's `chooseAlignedWeighted(np, G, w)`: split planes on multiples
  of `2^a`, `a` the largest alignment whose weight imbalance stays within 1.05 (`a = 0` = the plain
  weighted ORB, bit for bit), so the pressure MG coarsens in place for `a` levels before any stage
  fires (`[mg] rebalanceByWeights: aligned weighted ORB a = …` under `PECLET_FLOW_MG_DEBUG`). It
  returns `2^a`, and **a co-decomposing code must build its partition from the same weights AND
  that alignment** — dem's `migrate_to_weights(w, align=…)`, which coupling's `CfdDem.rebalance()`
  calls with the return value; a code that uses the weights alone owns different blocks. Probe
  (96³, np = 4/8, pinned, median of 5): projection ÷ momentum after the rebalance 1.05–1.10 against
  0.98–1.07 unweighted (Repartition alone: up to 1.10; the collapse: up to 1.87). **Forecast it
  without a run:** `flow.predict_hierarchy(..., weights=w)` returns `(rows, align)` — the ladder
  after the rebalance (its partition, Repartition stages included) and the `2^a` the call will
  return; `scripts/check_decomposition.py --predict --weights w.npy` logs both.
  `test_predict_weighted_mpi` holds the forecast to the built hierarchy row for row, before and
  after the rebalance (np = 1, 2, 4). `weights=None` is the old call, byte for byte.
- **`redistribute` must carry every piece of cross-step state, and seed it AFTER the scatter.**
  Its step 3 reallocates every buffer whose block changed size (fresh zeros) and only step 4
  scatters the migrated registry fields into them, so state DERIVED from a registry field must be
  re-derived after step 4 — eps^n (`epsPrev_`, the porous d(eps)/dt) was copied from the zeroed
  `eps` in step 3 until 2026-09-25, and the first porous projection after any size-changing
  rebalance saw d(eps)/dt = eps/dt (pressure off by 2.2e+02 in a CfdDem rebalance).
  `test_porous_redistribute_mpi` gates it; a rebalance that keeps every block's size hides it.
- Decomposition: `set_decomposition(0)` (default) is aligned ORB (fine-grid splits snapped to a
  power of two); `set_decomposition(L>=2)` is coarse-first — decompose the grid coarsened `L-1`
  times, then refine the partition upward, so blocks nest for the full depth and balance better.
  `decomposition()` takes the deepest candidate within `max_imbalance` (1.05), a pure function of
  (ranks, grid, levels). Check a combination with `scripts/check_decomposition.py` first.

## 7. Collocated solver

*Scheme strings and their internals, V8.*

Read [`doc/collocated_invisible_subspace.md`](collocated_invisible_subspace.md) (mechanism)
before touching this path; [`doc/collocated_paper_plan.md`](collocated_paper_plan.md) tracks the
results and [`doc/fluid_only_constraint_plan.md`](fluid_only_constraint_plan.md) is the
production plan. **The default is AUTO = `"ghost"`** (the fluid-only scheme) wherever the
configuration supports it, falling back to gauge-exact with a stderr notice on porous / variable-ρ
/ domain-BC / Chebyshev. Any explicit `set_collocated_scheme` / `diagnostics.set_face_interp` /
`diagnostics.set_ghost_projection` disables AUTO; tests and baselines pin schemes explicitly.

`set_collocated_scheme` takes one of four strings. `"ghost"` ==
`diagnostics.set_ghost_projection(True, 2, 2)`: fluid-only binary-openness constraint, directional
closures, gauge-exact gradient — family-free, no stabilizer, but BiCGStab (~2.3–2.7× the pressure
stage) and ~1.6 KB/cell of overlay. `"gauge-exact"` converges yet carries an attractor family;
`"plain"` is first order and legacy; `"embed"` is the complete Basilisk embed.h port (the FV
momentum operator with the true-normal wall drag as a defect correction, the openness-weighted
pressure force, the wall-aware face map and the sliver mask — the live candidate for the accuracy
ceiling; its two intermediate rungs are `diagnostics.set_face_interp(5 | 6)`). Inside the C++
these are `faceInterp_` 0 / 9 / 7 (5, 6); the integer modes 1–4 and 10–13 and the `gauge-2a`
gradient branch were deleted with their kernels at 1.0.0. MPI validated np = 1, 2, 4 (np ≥ 16
unresolved); the mixed `(matrix_order=1, rhs_order=2)` ghost mode is **do-not-use**,
march-unstable above ~2000 spheres.

**Variable density and surface tension (rung V8)** run all-fluid through the mass-adjoint ABC pair
of [`doc/collocated_varrho_forces.md`](collocated_varrho_forces.md) (immersed solids on that
path: the next package, `doc/collocated_multiphase_solids_plan.md`); the balanced-force projection
(`set_balanced_force_projection`) defaults ON there and makes static balances exact from step 1.

## 8. Domain boundary conditions

*Open boundaries, rank awareness, ghost fills, solid cutting an open face, the asymmetric axis ends, outlet divergence.*

`set_domain_bc(face, type, velocity=(vx, vy, vz))` with `face` one of `'-x'`, `'+x'`, `'-y'`, `'+y'`,
`'-z'`, `'+z'` and `type` one of `'periodic'` (default), `'wall'` (no-slip), `'inflow'` (Dirichlet
velocity), `'outflow'`, `'slip'` (free-slip/symmetry, which **mirrors the SDF ghost band** about
that face — `extendSdfDomainGhosts`, which gives every other non-periodic face the constant normal
extension instead — or a half channel closed by a symmetry plane would see the far wall as a
solid). Tangential walls use a face-fold in the implicit diffusion so `u_inner` stays
implicit. `set_domain_bc_profile(face, profile[Nb,Nc,3])` prescribes a per-position inlet (and
sets the face to inflow) — the backward-facing step is realized purely this way. Only a call that
would CHANGE a face's TYPE must precede the geometry (it raises afterwards); a VALUE update
(`velocity=`, or a new profile) on a face whose type is unchanged is allowed at any time and takes
effect the next step — ramp an inflow jet or a lid's tangential speed after `set_solid`. With no
immersed solid, use `set_pressure_geometry(all_fluid_sdf)`.

- **Open boundaries** split face openness in two: the *operator* openness (pressure matrix) is 0 at
  walls and inflow and open at outflow (Dirichlet p = 0, mean-removal off); the *flux* openness
  stays open at both so their flux is counted. **Outlet reversal** is the one conditionally-stable
  regime and it is instrumented: `set_backflow_stabilization` (default β = 0.2, β ≥ ½ the
  unconditional bound) adds β ρ |u·n| to the reversed row's diagonal, `diagnostics.outflow_backflow()` returns
  the census, and `step()` warns once on stderr when reversal appears with β = 0. Purely outgoing
  outlets are byte-identical.
- **Rank-aware:** every per-face application is guarded by `touchesGlobalFace` — a rank imposes a
  face's BC iff its own block touches that global face (before that fix, a partition cutting a
  walled axis split the domain into independent sub-domains, visible only in the pressure). A
  multi-rank inlet profile must be handed to each rank as its own slice; no scatter helper.
- **Ghost fills that are easy to lose:** `step()` fills the cell body-force and `drag_beta` ghosts
  right after `updateProperties()` because every staggered RHS builder and `addDragDiagonal`
  face-interpolate them (`Grid::atVelocity`: a VOLUMETRIC force sits where the velocity lives);
  skip either and the first inner plane of every block silently carries half the value.
- **Solid CUTTING an open face** (fixed 2026-09-16, `test_openbc_solid{,_mpi}` — the first tests to
  combine `set_domain_bc` with `set_solid`): the SDF ghost outside a non-periodic face was filled by
  PERIODIC WRAP, so the boundary-face aperture was teleported from the far side — a solid cell
  against the inlet came out fully open and the prescribed inflow fed a closed pressure row
  (inconsistent: MG-PCG capped, `max|div|` = U). `extendSdfDomainGhosts` now extends the SDF
  constant out of every non-periodic face (type 4 keeps its mirror). Separately, a Dirichlet outlet
  row carried the literal openness 1.0 instead of the face aperture — silently wrong, mass leaving
  through solid — and now carries the aperture through the WO-R2 save/restore/coarsen machinery
  (the C++ ablation `setOutflowOperatorCoefficient(false)` — not bound in Python — restores the old row). Geometry that SEALS fluid cells against
  an inlet is rejected by `set_solid` with a named error, not stalled on. **Both grids** — the fix
  is all geometry, so it is grid-independent, and the pre-fix collocated inlet-cut bed capped at 200
  with `max|div|` = U exactly as the staggered one did; `test_openbc_solid` runs on both.
  [`doc/cutcell_openbc_convergence.md`](cutcell_openbc_convergence.md).
- **The two ends of an axis are ASYMMETRIC and the halo knows nothing about domain BCs.** The LOW
  domain face of an axis is an INNER index; the HIGH one is the first GHOST index, and the
  decomposition wraps periodically on every axis (the non-periodic conditions are imposed on top),
  so any high-side boundary plane comes back from an exchange carrying the OPPOSITE boundary's
  value. Two things depend on it and **fixing either alone makes things worse** — they used to wrap
  together and stay mutually consistent while both were wrong about the geometry: the cut-cell
  openness (`buildOpennessHighFace` re-derives it from the SDF after the exchange) and the outflow
  face velocity (`fillVelGhostsTo(..., doOutflow = false)` now saves and restores the plane over the
  INNER transverse range; the transverse ghost rows are the neighbour's inner values and must stay).
  `test_vof_bc_mpi`'s composed budget is the gate: 1.25e-14 → 2.46e-02 with the openness half alone
  → 2.89e-14 with both, at np=1. `docs/SCALING_ISSUES.md` #8.
- **Which divergence to read at an outlet:** `max_open_divergence()` refills the outflow ghost with
  the zero-gradient extrapolation before measuring, so at a partly blocked outlet it reports how far
  zero-gradient is from the mass-conserving face and does NOT decay.
  `max_open_divergence_projected()` is the residual of the constraint the projection solved.

## 9. Geometric VoF

*What enable_vof switches on, the capillary step, the WY dilation flag, scope.*

Sharp-interface two-phase flow: PLIC planes (SZ2000 / Lehmann–Gekle, MYC normals), Weymouth–Yue
split geometric advection on its own g=3 block (CFL cap 0.25 — WY's 3-D bound; the familiar 0.5 is
the 2-D value), height-function curvature with a PLIC-volumetric paraboloid fallback,
balanced-force CSF surface tension, transport through an SDF solid, static and dynamic contact
angles, open boundaries, a per-bubble block container, and phase change. `"C"` is an ordinary
registered `G=2` cell field, so ρ(C)/μ(C) go through the existing closures and the field accessors
work on it unchanged. Entry points: `enable_vof()`, `set_vof`/`get_vof`, `advect_vof`,
`compute_vof_curvature`, `set_surface_tension`, `enable_vof_momentum`, `set_contact_angle*`,
`set_vof_inflow*`, `enable_vof_blocks*`, `enable_phase_change`; their `*_diagnostics` censuses and
the ablation knobs (`set_csf_mode`, `set_vof_kappa_*`, the `set_phase_change_*` wall) are on
`s.diagnostics`.

- **`enable_vof()` turns on two things**: the exact level-0 pressure operator
  (`diagnostics.set_pressure_exact_residual`, per solver — a two-phase contrast is exactly what
  amplifies the float operator's broken `A·1 = 0`) and the wisp guard `diagnostics.set_vof_wisp_eps`
  at **1e-8** (0 is the
  bit-for-bit V1 predicate and still the standalone `WyAdvector`'s default). Any gate comparing the
  solver against a standalone `WyAdvector` must copy the knob (`IbmSolver::defaultVofWispEps()`).
  **`enable_phase_change` then sets the wisp eps back to 0** — the guard and phase change are
  incompatible on a curved interface; setting it afterwards is the deliberate override.
- **The capillary step** `Δt < sqrt((ρ₁+ρ₂)Δx³/(4πσ))` is `capillary_dt()` and is **enforced by
  `step()`** (`set_capillary_cfl` is the safety factor). At pore scale it binds everywhere, more so
  under refinement (`dt_σ ~ h^{3/2}` against `dt_CFL ~ h`).
- **The WY dilation flag is frozen once per step** — recompute it per sweep and exact conservation
  silently dies (2.3e-15 → 1.5e-2). Momentum consistency freezes its coefficient the same way,
  clamps the flux into WY's admissible interval on the *shifted* volume's own colour, and uses
  plain donor-cell upwind (a MUSCL slope is a density-ratio amplifier). `enable_vof_momentum` moves
  VoF advection to the head of `step()` and needs variable density, staggered layout, explicit
  advection, no solid, no porous continuity; it makes ratios above ~100 usable.
- **Scope, and say it to users:** staggered is the reference; collocated is all-fluid, ratio ~10
  with motion (WO-V3, 2026-09-25: a translating drop tracks staggered at ratio 10 and throws the
  Weymouth-Yue CFL cap at ratio 100 and 1000, balanced-force projection on or off; exact at rest
  at ratio 1000). The next package re-measures it. The block container is all-fluid and
  staggered-only for its CSF. **Colliding
  markers are rated since 2026-09-25** (`doc/vof_overlap_design.md`): the blow-up was garbage
  curvature on sub-cell *debris* one marker leaves inside another, not the SUM-force/MAX-colour
  pairing. The fix is per-step debris + sub-`wispEps` residue removal with exact return to the
  marker's own interface, a block-only `|κ| ≤ 1/Δ_min` clip, and a gas–gas capillary dt bound
  while markers overlap. **Never assemble the block force from the union colour** (rejected: it
  re-creates numerical coalescence through the rim crease); the parked `vof-w4` branch is
  superseded.

The rung-by-rung record — every work order, gate number and refuted hypothesis — is
`doc/history/vof_workorders{,_v2,_v34,_v5,_v6}.md`.

## 10. Cut-cell scalar transport

*The full user-facing summary: method, files, supported setters, refusals, traps, do-not-reverse, open.*

An opt-in per scalar (landed on main 2026-10-04). The contract is
[`doc/scalar_ibm_design.md`](scalar_ibm_design.md), with Amendments A1–A3 and the open
questions in §13; every measured number and every ruling (D-WO*) is in
[`doc/scalar_ibm_log.md`](scalar_ibm_log.md). Legacy scalars are untouched: the dispatch is the
first line of the `advanceScalars` loop, and the 12 state hashes are the gate.

- **Status.** The `cutcell=` spelling is DEFAULT-PENDING Frank (§13 Q5). The path stays opt-in until
  G1–G13 pass and one release has shipped it (Q7).
- **Method.**
  - Cut-cell FV with κ (fluid fraction) in storage and sources.
  - Plain aperture two-point faces, on the scalar's own apertures (ungated, snapped at both ends).
    `ox_` is never touched.
  - Geometry from one fan-tetrahedron PL model of the SDF (core `scheme/cut_cell_geometry.hpp`).
  - Walls: one facet per cut cell, with a **probe-flux** closure. The probe sits on the normal at
    s = 1.1·½Σ|n_a|h_a, interpolated trilinearly, with a fallback ladder R0/R1a/R1b/R2 (core
    `scheme/probe_flux.hpp`). The BC is eliminated per facet.
  - Conjugate solids: two fields on one grid, in ψ = c/K.
  - Solver: BiCGStab on the true operator, preconditioned by one `ScalarMG` V-cycle on the SPD
    lumped-probe surrogate (the advective surrogate for steady advection, A2). The stop is a max-norm
    relative residual, default 1e-10.
  - Accuracy: second order for every BC, every Robin number and every conjugate contrast (gates
    G1–G8: orders 1.84–2.02).
- **Files.** `src/scalar_cutcell_geometry.hpp`, `src/scalar_cutcell_operator.hpp`,
  `src/scalar_krylov.hpp`, `src/scalar_mg.hpp`, the domain header `src/flow_ibm_scalars_cutcell.hpp`;
  gates in `tests/python/test_scalar_cutcell_gates.py`; 2-D oracles in `tests/study/scalar_ibm/`.
- **Supported** (all physical units):
  - wall `set_scalar_wall(name, type, value, coefficient, instance=None)`: `'neumann'` (value = flux
    into the fluid; the default is 0), `'dirichlet'`, `'robin'` (coefficient = k, value = g);
  - conjugate solids `set_scalar_solid(name, diffusivity, capacity, partition, contact_resistance,
    instance=None)`, i.e. D_s, C_s, K and R_c; `get_scalar_solid` returns c_s;
  - the mean-gradient closure mode `set_scalar_mean_gradient` (θ periodic, c = G·x + θ) with
    `scalar_mean_flux` (k*, dispersion), and `solve_scalar_steady`;
  - `set_scalar_source`, `set_scalar_tolerance`, `scalar_wall_flux` (per body); `set_scalar_bc` takes
    a per-face profile;
  - domain faces: periodic, `'dirichlet'`, `'neumann'`, and flow inflow/outflow faces **on the
    staggered grid**. Their flux is the one the projection constrained, captured by `step()`, and an
    inflow face needs a scalar `'dirichlet'` value.
- **Refusals** (RuntimeError at the first advance or solve; the message names the remedy):
  - **the collocated `'ghost'` scheme**, the AUTO default of `SolverColocated` (and the staggered
    `diagnostics.set_ghost_projection`). Its face field is divergence-free under no openness, and a
    constant drifts 0.80 in 50 steps (G9b). Remedy: `set_collocated_scheme('gauge-exact' | 'plain' |
    'embed')` before `set_solid`, or the staggered `Solver`;
  - **collocated inflow/outflow faces.** The projected `uf_` does not keep the high-side boundary
    flux (D-WO5b-1). Remedy: the staggered `Solver`;
  - **a moving fluid without the cut-cell projection.** Remedy: `set_solid(..., cutcell_pressure=True)`
    or `set_pressure_geometry`. A fluid at rest needs neither. Relatedly, an open face before the first
    `step()` is refused;
  - **collocated, a moving fluid before the face field exists.** `set_solid` zeroes `uf_` and a raw
    `set_field('u', …)` does not seed it, so the scalar would silently not advect (D-WOR-1, as
    `advect_vof`). Remedy: `set_state`/`set_velocity`, or `step()`;
  - porous continuity; moving scene instances; the per-cell Dirichlet mask; `set_phase_change_thermal`
    or `_energy` naming a cut-cell scalar (ValueError); a mean gradient along a non-periodic axis; a
    face periodic for one of scalar and flow but not the other.
- **Traps.**
  - **Koren is bounded only at bulk Courant ≤ ½.** Forward Euler with the legacy limiter is TVD only
    to ½; with no solid at all it reaches min −21 at C = 0.9 (D-WO5-1). FOU stays positive at any C.
    Koren above ½ warns once per scalar (D-WOR-7).
  - **Steady backflow through an `'outflow'` face** enters at the zero-gradient value c_i: counted
    (census `num_backflow_faces`) and warned once per scalar; A2's coarse mass stays clamped there
    (D-WOR-2a). An open face carrying flux is never the uniform-mean singular case (D-WOR-2b). With
    open faces but no inflow face the operator is still singular, with a NON-uniform left null
    vector: it converges for a compatible right-hand side (no source, insulating walls) and reports
    non-convergence otherwise — there is no steady state then. Give the fluid an `'inflow'` face.
  - The census residual of a singular solve is measured after the gauge shift (D-WOR-5).
  - **The unknown sets follow the snapped apertures, not κ.**
    - A fluid cell with κ > 0 and every aperture 0 is *sealed*: it is not an unknown and is held at 0.
    - A solid cell is an unknown iff κ_s > 0 and it has an open solid face or a conjugate facet
      (D-WO7-1).
    - Both sealed volumes are in the census (`sealed_volume`, `sealed_solid_volume`). A fluid one
      above 1e-6 of the fluid volume prints a warning.
    - Non-unknowns read exactly 0 (NaN in `get_scalar_solid`) and are re-zeroed every advance.
  - **The iteration count measures the MG level table.** An axis coarsens only while it stays even,
    so an odd or 2-poor box gives a shallow table: G5b at n = 77 takes 78 iterations, at n = 80 six.
    Gate and benchmark on boxes with factors of two (D-WO4-1;
    `../docs/DECOMPOSITION_AND_MULTIGRID.md` §3.1). The bottom is smoothing only, so a full-table
    solve (steady, or κ_A ≥ 25) whose table stops above `set_pressure_bottom_extent` (4) on any axis
    warns once per scalar (D-WOR-4).
  - **rtol and MPI parity.**
    - np = 1 is bitwise.
    - np > 1 agrees to the Krylov reduction-order floor, i.e. inside the stopping tolerance, so MPI
      parity is gated at rtol 1e-13 (≤ 1e-10 relative, iterations ±1; D-WO4-2).
    - The thread count and the backend move Koren results by ~1e-9 over hundreds of steps: the limiter
      carries reduction-order noise that FOU damps (D-WO9-1). Gate koren at 1e-7, or on integrals.
  - Steady advection is first-order FOU in v1 (WO-11 optional), gated to Pe_h ≤ 10 (census
    `max_cell_peclet`).
  - The transient level rule: κ_A = 1 + 4 dt'D'Σw_a < 25 runs level 0 alone, otherwise the full table
    (`ScalarMG::kFullTableKappa`, D-WO9-3).
  - **Conjugate contacts and sub-cell gaps are not resolved.** The point-sampled SDF closes fluid
    wedges thinner than ~h, so a contact neck comes out √(a² + Rh) instead of a, and conducting
    sub-cell gaps fuse.
- **Do not reverse** (register: `../docs/decisions/flow.md`, the cut-cell scalar entries of 2026-10-04):
  - a short or κ-dependent probe (AMReX; erratic order 0.4–1.2);
  - unit storage (first order; it loses 2–12 % of the mass);
  - the series-resistance / GFM wall flux (first order);
  - coarse wall terms at the level's own probe distance (divergent, contraction 3.1) or Galerkin RAP;
  - a symmetric surrogate for steady advection (no convergence at Pe_h 1);
  - a post-solve mass fix-up;
  - Peters' directional one-field conjugate scheme (L∞ first order, 2–50× worse).
- **Open.**
  - The conjugate contact model (§13 Q4, triggered; G13's conjugate row self-converges at −2.84 and is
    INFO, marked OPEN) → Frank.
  - Collocated open faces.
  - `advanceScalars`' plain velocity ghost fill still wraps the high-side open plane of the live flow
    state, also for a run with only cut-cell scalars (review note 7; `uf-outlet-diag` owns the fix).
    It cannot simply be skipped: the cut-cell advance reads the block's high-face velocity ghost,
    which `projectCorrect` (inner faces only) leaves at u* — skipped, a constant drifts 4.1e-2 in 6
    staggered steps (D-WOR-7, measured).
  - **Agglomerated exact bottom for ScalarMG (as CutcellMG `'auto'`)** — follow-up WO (review
    finding 4, D-WOR-4). Today the coarsest level gets 16 RB-GS sweeps wherever the table stops, so
    steady closure/dispersion solves at scale (grids sized by the physics, ORB blocks at np ≫ 1)
    lose the grid-independent iteration count; only the warning exists.
  - Host performance: advance ÷ projection ≈ 1.05 at OMP 4 after D-WO9-3, and 1.4–1.6 before it (the
    smoother is 38 % of the advance). The GPU sits at 0.96–0.98 under contention.
  - Q5 (spelling), Q7 (default), and the NAMING rows in `doc/scalar_ibm_naming_rows.md`.

## 11. Steady marches

*The stop instrument, the Anderson phases, scope, memory, checkpointing, the developer tier.*

`peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, max_steps=5000, accelerate=True, window=5,
check_every=5, num_passes=3, slow_rate=0.997, roundoff=1e-11, callback=None)` marches a solver to its
steady state and returns a frozen `MarchResult` (`converged`, `steps`, `accelerated_steps`, `reason`
∈ {"certified", "max_steps", "diverged"}, `num_restarts`, `monitor`). Design and every measured
number: [`doc/steady_acceleration.md`](steady_acceleration.md) (rev 2) and its log.
`converged=True` certifies that the state passed the stop test on consecutive plain steps at this dt
(stationarity); it does not certify that a plain march from the initial state would reach it.

```python
s = peclet.flow.Solver((N, N, N), extent=(L, L, L)); ...; s.set_solid(sdf, cutcell_pressure=True)
res = peclet.flow.march_to_steady(s, lambda: float(s.get_u().mean()))   # <u_x> along the force
```

- **The stop instrument is the study's** (geometric-remainder bound on the block change of
  `monitor()`); `accelerate=False` is that march step for step (ctest `march_to_steady`, G0(b)).
  Under MPI `monitor()` must return the same value on every rank (a global reduction).
- **Acceleration is type-II Anderson on the march state** (u, v, w + P; collocated with projected-face
  advection + uf/vf/wf), velocity-only metric. Phase A mixes until the relative velocity residual is
  `(1 - slow_rate) * rtol`; phase B certifies on consecutive PLAIN steps with the unchanged
  instrument (budget `2 * (num_passes + 3)` blocks, early "slow" exit, ×0.1 and resume), so the
  reported state is a plain-march state. There is no instability guard (rev 2: a Ritz radius of
  this non-normal map is no stability test); an unstable plain map shows only on plain steps
  (growth exit, R ≥ 1 never passes, stagnation fallback). Phase A "stagnates" when its residual
  has not fallen by `slow_rate**(10 window)` (the plain march's assumed rate) in `10 window` calls
  (review R1; a halving rule lost the dense bed at ν dt/h² = 60). A core restart restores the last
  kept map output (review R2). `Solver.step()` is untouched. Data
  path: core's `AndersonCore` + `src/anderson_accelerator.hpp`; control path:
  `packaging/flow_steady.py` (installed as `peclet/flow/steady.py`).
- **Scope.** Staggered `Solver`; `SolverColocated` with the `'ghost'` scheme only. Refused with a
  named error (`Solver::marchState()`, re-checked every step): the other collocated schemes, VoF,
  phase change, scalars, porous continuity, variable rho/mu, `drag_beta`, cell forces, moving scenes,
  `set_superficial_velocity`, the pressure warm start, the balanced-force projection. Pass
  `accelerate=False` for those.
- **Memory:** `(2 window + 3) * n_fields * 8 * n_padded` bytes per rank (`n_fields` 4, or 7 collocated
  with face advection): 416 B per inner cell at window 5, ~7.5 M cells max on a 16 GB card beside the
  solver. `acc.memory_bytes` reports it; an allocation that does not fit raises with the byte count.
- **Checkpoint with `get_field`/`set_field` of u, v, w, p** — not `set_state` (velocity only, loses P).
  The Anderson history is not checkpointed; a restart rebuilds it in a few steps.
- Developer tier: `s.diagnostics.anderson_accelerator(window=5, mixing=1.0)` → `acc.step(accelerate)`,
  `acc.residual`, `acc.status` ("active" | "disabled"), `acc.reason`, `acc.num_restarts`,
  `acc.num_resets`, `acc.num_columns`, `acc.memory_bytes`, `acc.seconds`, `acc.reset()`,
  `acc.disable()`. A redistribute
  reallocates the state buffers, and collocated `set_advection` changes the field count (a
  configuration change, history reset); the accelerator then refuses to step (construct a new one).
  A change of dt, rho, mu, the body force or the advection settings (on / scheme / implicit) between
  calls resets the history (`num_resets`).

## 12. Open items — long forms

*The multigrid/redistribute items and the planned consistent wall traction.*

Intermediate-level multigrid repartitioning at scale (the Repartition kind exists and fires for
a weighted `dec0`; unweighted ladders are unchanged); coefficient-aware coarsening for high
contrast; `vof-w4` (superseded by `doc/vof_overlap_design.md`, but `quality.yml` still excludes its six files from clang-format); **the collocated advecting face field `uf_`
is not carried through
`redistribute`** (pre-existing, not fixed by the V8 package): a collocated run WITH advection that
is rebalanced at np = 1 already drifts 5.0e-8 in u from the never-rebalanced run, option off
(measured 2026-09-25 while gating `balanced_force_mpi`, whose rebalance case is therefore Stokes).

**PLANNED — a consistent wall traction** (2026-09-30): the wall shear from the momentum operator's
Robust-Scaled (small-cell-robust) wall reconstruction instead of the central difference in
`hydroForceTorque()` (`src/flow_ibm_hydro.hpp`), so that traction → reaction under refinement.
Unlocks wall-shear maps, the pressure/viscous split of the *accurate* force, and local surface
fluxes; a candidate section of the M3 method paper (`~/Codes/peclet-papers` PLAN.md, item D8).
Known dead end: the one-sided difference to the wall over the crossing distance θ — 1/θ is
unbounded on small cut cells and the drag came out 17× too large. Gate: traction/reaction → 1 on
`tests/python/test_hydro_force_units.py`'s spheres, with the steady spinning-sphere torque at the
reaction's 1.02 rather than the traction's 0.59.
