# Velocity (momentum) multigrid for the cut-cell IBM — implementation plan

**Goal.** A *convergent* geometric V-cycle for the per-component implicit-diffusion (Helmholtz) velocity
solve `(I − νΔt·L)u = b`, so the steady-state march (and large-Δt steady-Stokes solves) is
resolution-robust instead of O(N) in RB-GS sweeps. **Standalone RB-GS V-cycle only** (no PCG/Chebyshev
wrapper), **optional** (`set_velocity_multigrid`, default off; default RB-GS path byte-identical).

> **STATUS (2026-06-16): cavity/BFS diffusion vmg + upwind-FOU coarse op DONE (committed). IBM-path
> diffusion fix = the scheme below; Phase 2 (`D_rescale` un-scale) is REJECTED, do not revive it.**
>
> The first IBM attempt (masked staircase coarse + `D_rescale` residual un-scale) left the +2–4% Z&H drag
> bias unchanged. Diagnosis: the dominant error is the **unmasked transfers** (corrections move across the
> immersed boundary, biasing the cut-cell skin), not the coarse operator. The un-scale (Phase 2) is both
> **dangerous** (`D_rescale = min|D|` cut-face factor →0 for thin slivers; dividing the residual by it
> amplifies near-solid noise) and **unnecessary**: the correction-scheme V-cycle already restricts the TRUE
> residual `b' − A_f x` (level-0 op = the IBM `As_[c]`), so the fixed point is the exact sharp solution for
> ANY coarse op — only the *rate* depends on it. `src/mac_ibm.hpp` is explicit that the row-based IBM op is
> "NEVER multigridded; D_rescale is GS-invariant but not MG-invariant." **⇒ leave the residual scaled; fix
> the coarse op + transfers (Phases 1+3 below), not the scaling.**

## Why the current `vmg_` fails (measured)
`ensure_vmg_built` wires `VelocityMG` with a `setDiffusionCoarse` = **constant-coefficient** coarse
(`mg_const_diffusion_op_k`, geometry-free) and feeds it the **row-scaled** fine IBM stencil. Result: at
moderate Δt the drag is biased +2% (dt=60) and it **NaNs at dt=200**. Two root causes:
1. the coarse operator throws the solid geometry away → the coarse correction is inconsistent at the skin;
2. the fine residual `b − A_fine x` is **`D_rescale`-scaled** at cut cells (the Robust-Scaling row factor),
   so 8:1-restricting it mixes scaled and unscaled rows → the coarse "sees" a wrong residual.

The design principle (from `doc/ibm_overlay.md`): keep the **row-scaled sharp IBM only on the fine-level
RB-GS smoother**; build the **coarse operators from a clean, un-scaled staircase/volume-fraction
discretization**; the V-cycle's fine residual+smoother guarantee the converged answer is the exact *fine*
(sharp-IBM) solution regardless of how crude the coarse is — the coarse only sets the convergence *rate*.

## Design (mirrors the proven pressure `CutcellMG::setOpenness` path)

### Phase 1 — rediscretized staircase / volume-fraction coarse operator (the core)
New velocity coarse operator on `VelocityMG` (built once per component; static
geometry), mirroring `CutcellMG::setOpenness`:
1. **Coarsen the fluid volume fraction** `θ` (per the component's staggered grid) 2:1 per level — new
   `mg_coarsen_volfrac_k` (volume-average θ, like `mg_coarsen_open_avg_k` area-averages openness). Stash θ
   per level in the existing `tx` scratch.
2. **Build the coarse Helmholtz** per level — new `mg_build_velocity_coarse_k` into `AC..AT` (mreal):
   - face openness `α_f = min(θ_i, θ_nbr)` (or harmonic) from the coarse θ; `β_f = νΔt·α_f/h_L²`;
   - `AC = θ_i·(1 + Σ_f β_f)`, off-diagonal slot `= −β_f`;
   - **fully-solid cells (θ≈0) decouple**: `AC=1`, off-diag=0 (identity row → masked Dirichlet).
   At Re=0 this is the **symmetric masked Laplacian** (the clean case to land first). It is an M-matrix →
   `mg_smooth_var_k` (RB-GS) smooths it cleanly and ρ<1. *(Re>0 upwind: Phase 4.)*
3. Replace the `setDiffusionCoarse` call in `ensure_vmg_built` with three per-component velocity-coarse
   builds (`VelocityMG::setStaircase` / `buildUpwindCoarse`) and a
   per-component coarse-operator swap (`AC..AT` per level), exactly like `setDiffusionFine` swaps level 0.

### Phase 2 — ~~un-scale the fine residual~~ **REJECTED — do not implement**
~~Divide the restricted residual by `descale_`.~~ This is wrong on two counts: (a) `D_rescale = min|D|`
cut-face factor →0 for thin slivers, so `res/descale` amplifies near-solid noise (blows up); (b) it is
unnecessary — the correction scheme already restricts the TRUE residual `b' − A_f x` (level-0 op = `As_[c]`
via `setDiffusionFine`), so `r=0 ⇒ correction=0` and the fixed point is the exact sharp IBM solution for any
coarse op. The coarse op only sets the convergence RATE. **Leave the residual scaled.** The +2–4% bias is a
*finite-cycle* artifact of unmasked transfers (Phase 3), not of the residual scaling. (`src/mac_ibm.hpp`: the
IBM op is row-based, non-conservative across the wall, "NEVER multigridded; D_rescale … not MG-invariant".)

### Phase 3 — clean-fluid-interior coarse coupling (the actual fix that landed)
The volume-weighted-transfer idea was tried and dropped (it suppressed useful coupling and was unstable; the
proven pressure cut-cell MG uses a geometry-aware *operator* with *plain* transfers). What actually works is
restricting **which cells the coarse grid is allowed to touch**: the coarse grid couples ONLY where its clean
operator matches the fine one — the **clean fluid interior** (a fluid cell with no solid neighbour).
- A mask `ibm_clean_fluid_mask_k` = 1 at clean-fluid cells, **0 at IBM cut cells AND solid cells**.
- **Restriction:** zero the fine residual at the masked cells before restricting (`mg_mul_mask_k` on
  `lv.res_mask`). The inconsistent row-scaled cut-cell residuals and the stiff `1+6β` solid residuals never
  reach the coarse grid.
- **Prolongation:** `mg_prolong_masked_k` — no coarse correction is added INTO the masked cells.
- The fine IBM RB-GS smoother owns the cut-cell band + the solid interior; the coarse grid solves the clean
  interior. Excluding the **solid** cells (not just the cut cells) is what removes the large-β overshoot:
  during the V-cycle the fine op carries solid cells as full `1+6β` (masked only at the end) while the coarse
  op treats them as identity (θ<ε) — a factor-`(1+6β)` mismatch that diverges at large Δt without the mask.

### Phase 4 — finite-Re (advection) and gating
- For Re>0 add an **upwind advective** term to `mg_build_velocity_coarse_k` from the coarsened advecting
  velocity (donor-cell; numerical diffusion is *welcome* on the coarse — keeps the M-matrix). Gate on
  `advection_`. Stokes (Re=0) needs none.
- `set_velocity_multigrid(on, levels, vcycles)` stays the toggle (default off). On ⇒ staircase coarse +
  residual un-scale + vol-weighted transfers; default RB-GS path untouched.

## Files / touch list
- `src/mac_velocity_mg.hpp` (`VelocityMG`): new kernels `mg_coarsen_volfrac_k`, `mg_build_velocity_coarse_k`,
  `mg_restrict_volwt_k`, `mg_unscale_res_k`; the per-component velocity coarse operator; a `vel_unscale_` flag +
  `descale_lvl0_` and the level-0 residual-unscale + masked/weighted transfer hooks in `vcycle`.
- `src/mac_ibm.hpp`: a θ (fluid-fraction) kernel from the staggered SDF (reuse `ibm_solid_mask_k` shape);
  the residual-unscale kernel may live here next to `ibm_scale_k`.
- `src/sdflow_ibm.hpp` (`SdflowIbm`): `ensure_vmg_built` — build the 3 per-component staircase coarse hierarchies
  from the staggered θ; the vmg branch — set `descale_lvl0_ = descale_[c]` and enable `vel_unscale_`;
  keep the smoother on `As_[c]` (row-scaled).
- `src/sdflow_bindings.cpp`: no change (`set_velocity_multigrid` already exposed).

## Validation (gate each phase)
1. **Z&H SC sphere** (`scripts/validate_zick_homsy_sdflow.py`): vel-MG drag == RB-GS drag to
   <0.01%; per-cycle Helmholtz residual drops geometrically (ρ≲0.3); fewer outer Picard steps than RB-GS.
2. **Ring bed at high res** (the `D_rescale≠1` stressor): converges (no NaN/bias), matches RB-GS, and the
   **outer Picard count becomes ~resolution-independent** (the whole point — fixes the 600-step-cap stall).
3. **Regression:** default (vmg off) byte-identical; **72/72 ctests** green (np=1,2,4).

## Sequencing & risk
- **Do Phase 3 (masked transfers) FIRST/together with 1+2, Stokes-only** — it's where the bias lives
  (measured; see STATUS). Concretely: mask the prolongation (no correction into/at a fine solid-skin cell)
  and volume-fraction-weight the restriction (`Σθ_f r_f / Σθ_f`), so the staircase-coarse ↔ sharp-fine
  transfer is consistent at the boundary. *Then* the masked coarse (Phase 1) + residual un-scale (Phase 2)
  set the convergence rate. **Validation gate stays Z&H** (drag == RB-GS to <0.01%, ρ≲0.3); only after that
  passes does the ring bed test mean anything.
- A cheap first probe before wiring the full vol-weighting: just **mask** the existing `mg_restrict_k` /
  `mg_prolong_k` at solid cells (zero θ<ε entries) — if that alone removes the +2–4% Z&H bias, it confirms
  the transfers are the culprit and the vol-weighting is then the refinement for partial cells.
- Add Phase 4 (upwind/advection) only after Stokes converges.
- The biggest new surface is the **per-component staggered θ hierarchy** (3 mask sets); everything else
  reuses the pressure MG machinery. The `D_rescale` un-scale is small but non-optional for the hard cases.
- Keep the const-coeff coarse as a selectable fallback for A/B comparison.

---

## UPDATE (cavity path landed; upwind-convective design)

**Done + committed (main 94b64b2, 87d69c4):** velocity-MG on the **domain-BC** path (cavity/BFS), which
the dispatch never reached before (it lived only inside `if(ibm_enabled_)`). Operator = const-coeff
`I − νΔt∇²` + per-component no-slip face-fold (`mg_diffusion_bc_fold_k`, the const-coeff analogue of
`bc_dcorr_`); per-axis-β **semi-coarsening** (`mg_const_diffusion_op_aniso_k`, β_a=νΔt/(h0·cfac_a)²) so a
quasi-2D grid builds a deep hierarchy (nz=4 cavity → 7 levels). Validated: lid cavity Re=100 == Ghia;
V-cycle θ≈0.015; stiff regime (dt=30) −33% steps/wall vs RB-GS; 72/72 green. This is **diffusion-only**
(convection explicit in the RHS) — correct + sufficient for time-accurate unsteady inertial (moderate dt).

**Upwind-convective coarse operator (Phase 4, task #56) — DONE + LANDED (2026-06-16, IBM path).** For
*implicit-advection* large-dt high-Re **steady-state acceleration** (where the velocity operator is
advection-dominated/stiff). Opt-in: `set_velocity_multigrid(on,…)` **together with**
`set_implicit_advection(on)` now takes the upwind-convective vel-MG on **both** the IBM and the domain-BC
(cavity/BFS) paths (the combination was previously gated out → RB-GS; no current test sets both, so default
paths are byte-identical, 72/72 green). What shipped:
- Fine operator (pre-existing): `build_adv_stencil_k` → `sadv::fou_operator` adds first-order-upwind
  advection to the diffusion stencil from the advecting u,v,w at unit spacing (`As_[c]`).
- **Coarse** stencil builder `buildAdvCoarse` / `VelocityMG::buildUpwindCoarse` (`src/mac_velocity_mg.hpp`): aniso const-coeff
  diffusion (per-axis β from `cfac`) **+** a coarse FOU via new `sadv::fou_operator_aniso` on the
  *restricted* coarse velocity, scaling vel by `s_a = 1/cfac_a` per face axis (the FOU coefficient is
  `dt·vel/h_a`, and `fou_operator` assumes h=1). Launched over inner cells (smoother/residual read AC..AT
  only there; the ±1 reach hits exchanged ghosts).
- Per Picard iteration: `restrict_vmg_adv_velocities()` restricts u,v,w to every coarse level (reuse
  `mg_restrict_k`, 8:1 volume average + ghost exchange), `build_vmg_adv_stencil(c, include_fine)` (re)builds
  the velocity stencils, solve, mask. Per-level velocity scratch `vadv_{u,v,w}_` allocated in
  `ensure_vmg_built` (whenever `implicit_fou_`), zero-init so non-periodic boundary ghosts read 0 (no
  spurious coarse advective wall flux).
  - **IBM path** (`ibm_enabled_`): level 0 stays the fine row-scaled `As_[c]` (`setDiffusionFine`,
    `include_fine=false`); coarse levels = aniso diffusion + restricted FOU. Solution masked by `solidmask_`.
  - **Domain-BC path** (cavity/BFS, `has_domain_bc_`, no IBM stencil): `build_vmg_adv_stencil` builds the
    operator on **every** level (`include_fine=true`, level 0 from u_/v_/w_), then `setDiffusionBoundaryFold`
    applies the no-slip/inflow/outflow wall fold on all levels; the dt·FOU deferred-correction RHS
    (`add_fou_rhs_k`) is added for the `implicit_fou_ && vmg_enabled_` case. Semi-coarsening (quasi-2D)
    works (per-axis `cfac` β + 1/cfac advective scaling).
- Upwind keeps every level an **M-matrix** (diagonally dominant) → RB-GS smoothing + coarse correction
  stable in advection-dominated rows, at arbitrary CFL.
- **Validated:**
  - IBM (`scripts/verify_velocity_mg_upwind_sdflow.py`, sphere + cut-cell, dt=5): stable at high Re (U~2,
    CFL≫1); converges to the **same field as RB-GS to machine precision** (U_max/U_mean diff 0.000%);
    V-cycle ρ≈0.02–0.05 (~8 cycles to machine ε).
  - Domain-BC (`scripts/verify_velocity_mg_upwind_cavity_sdflow.py`, lid cavity Ghia Re=100, quasi-2D
    nz=4): at advective **CFL=40** (dt=40, grid units) **explicit advection blows up** while implicit-FOU +
    upwind vmg stays **bounded and converges to the Ghia centreline** (u_min −0.2113 vs Ghia −0.2058),
    machine-precision divergence (3.6e-16), steady in 201 steps; the 6-level semi-coarsened V-cycle drops
    the residual geometrically (51→0.26→0.014→…, ρ≈0.06–0.13).
  - **72/72 ctests green** (default paths byte-identical).

## IBM-path coarse operator — CONSOLIDATED to the STAIRCASE (opt-in; RB-GS stays the DEFAULT)

The IBM velocity-MG (`set_velocity_multigrid(on, levels, vcycles)` with `set_ibm_solid`) uses the **staircase**
coarse operator — the *only* IBM coarse operator now (the geometry-blind const-coeff and the area-fraction
variants were built, measured, and **removed** after the staircase dominated both). Off by default — plain
RB-GS remains the default IBM velocity solve; 72/72 ctests green (default path byte-identical).

**The staircase operator** (`setVelocityStaircaseCoarse`, `mg_build_velocity_op_staircase_k`):
- **Fine level** = the sharp row-based IBM stencil `As_[c]` (so the residual + smoother use the TRUE operator
  and the fixed point is the exact sharp solution). Its solid cells are pinned to 0 in the smoother, and its
  **IBM-cell residuals are filtered before restriction** (the clean-fluid exclude mask `ibm_clean_fluid_mask_k`
  = 0 at cut+solid; restriction `mg_mul_mask_k`, masked prolong `mg_prolong_masked_k`). This exclude is
  **required** — coupling the row-scaled IBM cut/solid cells to any coarse grid overshoots at large β (even at
  one level; a pore-scale coarsening *cap* does not help — measured).
- **Coarse levels** use the volume fraction θ ONLY to **classify** cells: θ≥0.5 fluid, θ<0.5 solid (pinned to
  0 — a first-order staircase no-slip). A **plain const-coeff Helmholtz** is built at fluid cells (per-axis β;
  **no area/volume fractions in the coefficients** — that was the area-fraction op, whose weak partial-cell
  coupling capped it). Pinning is `MGLevel::pin` + a `pin` arg on `mg_smooth_var_k`/`mg_residual_var_k`
  (nullptr ⇒ the pressure MG is bit-identical). The binary classification disconnects fluid pockets across
  resolved walls — the failure mode of the geometry-blind const-coarse — so it's stable AND geometry-aware.

**Validated** — exact (== RB-GS, 0.000%) and stable to very large Δt:
- SC sphere (`scripts/verify_velocity_mg_staircase_zh_sdflow.py`): exact; stable to dt≈6400 (β=640) both φ.
- Packed beds (`scripts/verify_velocity_mg_staircase_packing_sdflow.py`, random periodic sphere packings):
  moderate (21 sph, φ=0.245) and dense thin-neck (53 sph, φ=0.29, ~1-cell necks) — exact (0.000–0.010%),
  stable to dt=800–3200 (β up to 320; the ceiling rises with V-cycle count), and **exact across coarsening
  levels 2–6** (deep coarsening only affects the *rate*, not the answer, since the fine smoother + exclude
  mask own the boundary — so `levels` is a tuning knob, not a correctness requirement). V-cycle ρ≈0.23.

**Why conserve this** (it is not a big efficiency win on the current uniform grid — RB-GS converges in
O(pore-cells) sweeps for beds, and at high res the pressure solve dominates the step): the staircase
multigrid is the natural coarse operator for a future **AMR with extreme refinement near contact points**,
where the velocity solve becomes genuinely stiff/multiscale and an O(1)-V-cycle solver pays off. The exact,
unconditionally-stable behaviour (fine IBM smoother + clean-fluid exclude + staircase coarse) is exactly what
an AMR hierarchy needs. Keep it intact.

**Measured dead-ends (do not retry blindly):** (a) un-scaling the residual by 1/D_rescale — amplifies thin
slivers, and unnecessary (the true residual already gives the exact fixed point); (b) coupling the partial
cells (dropping the exclude mask) — diverges at dt=200 even at one coarsening level, so a pore-scale cap
cannot rescue it; (c) volume/area fractions AS COEFFICIENTS — strictly worse than the staircase. All removed.
