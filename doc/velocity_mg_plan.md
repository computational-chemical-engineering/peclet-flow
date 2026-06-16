# Velocity (momentum) multigrid for the cut-cell IBM — implementation plan

**Goal.** A *convergent* geometric V-cycle for the per-component implicit-diffusion (Helmholtz) velocity
solve `(I − νΔt·L)u = b`, so the steady-state march (and large-Δt steady-Stokes solves) is
resolution-robust instead of O(N) in RB-GS sweeps. **Standalone RB-GS V-cycle only** (no PCG/Chebyshev
wrapper), **optional** (`set_velocity_multigrid`, default off; default RB-GS path byte-identical).

> **STATUS (2026-06-16): Phase 1 + 2 implemented and TESTED — insufficient alone; reverted.** The masked
> staircase coarse (Phase 1) + the `D_rescale` residual un-scale (Phase 2) compile cleanly, but on Z&H the
> +2–4% drag bias and non-convergence are **unchanged** from the const-coeff version. That null result is
> the key finding: **the dominant error is NOT the coarse operator — it is the unmasked transfers.** The
> restriction/prolongation move corrections across the immersed boundary inconsistently, biasing the
> cut-cell skin regardless of how good the coarse operator is. **⇒ Phase 3 (masked, volume-weighted
> transfers) is a CORRECTNESS prerequisite, not an efficiency nicety — do it FIRST/together with 1+2.** The
> masked coarse and the un-scale are correct building blocks underneath it. (Reverted to keep the 72-ctest
> solver clean; the working tree is unchanged.)

## Why the current `vmg_` fails (measured)
`ensure_vmg_built` wires `DistributedPoissonMG` with `setDiffusionCoarse` = **constant-coefficient** coarse
(`mg_const_diffusion_op_k`, geometry-free) and feeds it the **row-scaled** fine IBM stencil. Result: at
moderate Δt the drag is biased +2% (dt=60) and it **NaNs at dt=200**. Two root causes:
1. the coarse operator throws the solid geometry away → the coarse correction is inconsistent at the skin;
2. the fine residual `b − A_fine x` is **`D_rescale`-scaled** at cut cells (the Robust-Scaling row factor),
   so 8:1-restricting it mixes scaled and unscaled rows → the coarse "sees" a wrong residual.

The design principle (from `doc/ibm_overlay.md`): keep the **row-scaled sharp IBM only on the fine-level
RB-GS smoother**; build the **coarse operators from a clean, un-scaled staircase/volume-fraction
discretization**; the V-cycle's fine residual+smoother guarantee the converged answer is the exact *fine*
(sharp-IBM) solution regardless of how crude the coarse is — the coarse only sets the convergence *rate*.

## Design (mirrors the proven pressure `setFineVariableOperatorRediscretized` path)

### Phase 1 — rediscretized staircase / volume-fraction coarse operator (the core)
New `DistributedPoissonMG::setVelocityCoarse(comp, θ_fine, nu_dt, h0)` (built once per component; static
geometry), mirroring `setFineVariableOperatorRediscretized`:
1. **Coarsen the fluid volume fraction** `θ` (per the component's staggered grid) 2:1 per level — new
   `mg_coarsen_volfrac_k` (volume-average θ, like `mg_coarsen_open_avg_k` area-averages openness). Stash θ
   per level in the existing `tx` scratch.
2. **Build the coarse Helmholtz** per level — new `mg_build_velocity_coarse_k` into `AC..AT` (mreal):
   - face openness `α_f = min(θ_i, θ_nbr)` (or harmonic) from the coarse θ; `β_f = νΔt·α_f/h_L²`;
   - `AC = θ_i·(1 + Σ_f β_f)`, off-diagonal slot `= −β_f`;
   - **fully-solid cells (θ≈0) decouple**: `AC=1`, off-diag=0 (identity row → masked Dirichlet).
   At Re=0 this is the **symmetric masked Laplacian** (the clean case to land first). It is an M-matrix →
   `mg_smooth_var_k` (RB-GS) smooths it cleanly and ρ<1. *(Re>0 upwind: Phase 4.)*
3. Replace the `setDiffusionCoarse` call in `ensure_vmg_built` with three `setVelocityCoarse(c,…)` and a
   per-component coarse-operator swap (`AC..AT` per level), exactly like `setDiffusionFine` swaps level 0.

### Phase 2 — un-scale the fine residual before restriction (the missing must-do)
In `vcycle`, level 0 only: after `residual(lv)` (`lv.res = b − A_fine x`, `D_rescale`-scaled at cut cells),
divide by `descale_`: new `mg_unscale_res_k(lv.res, descale, n)` (`res /= descale`), gated on a new
`vel_unscale_` flag + a `const double* descale_lvl0_` pointer set by the solver. The **smoother keeps the
row-scaled operator** (GS-invariant); only the *restricted residual* is un-scaled. (This is the piece the
earlier code missed; it's a no-op for Z&H where `D_rescale≈1` but essential for thin walls / near-contacts
where it departs from 1.)

### Phase 3 — volume-fraction-weighted transfers + solid masking
- **Restriction:** `mg_restrict_volwt_k` (variant of `mg_restrict_k`): `coarse = Σ θ_f r_f / Σ θ_f` —
  weight the fine residual by its fluid fraction so fine-partial → coarse-staircase is consistent.
- **Prolongation:** keep `mg_prolong_k` (trilinear) but **mask**: zero the correction at coarse solid
  cells and don't add into fine solid cells (multiply by θ). Prevents the coarse correction pumping error
  into the skin that the smoother then fights every cycle.
- Masks come from the per-component staggered solid field already built by `ibm_solid_mask_k`
  (`solidmask_[c]`), coarsened to θ in Phase 1.

### Phase 4 — finite-Re (advection) and gating
- For Re>0 add an **upwind advective** term to `mg_build_velocity_coarse_k` from the coarsened advecting
  velocity (donor-cell; numerical diffusion is *welcome* on the coarse — keeps the M-matrix). Gate on
  `advection_`. Stokes (Re=0) needs none.
- `set_velocity_multigrid(on, levels, vcycles)` stays the toggle (default off). On ⇒ staircase coarse +
  residual un-scale + vol-weighted transfers; default RB-GS path untouched.

## Files / touch list
- `src/mac_multigrid.cuh`: new kernels `mg_coarsen_volfrac_k`, `mg_build_velocity_coarse_k`,
  `mg_restrict_volwt_k`, `mg_unscale_res_k`; method `setVelocityCoarse(comp,…)`; a `vel_unscale_` flag +
  `descale_lvl0_` and the level-0 residual-unscale + masked/weighted transfer hooks in `vcycle`.
- `src/mac_ibm.cuh`: a θ (fluid-fraction) kernel from the staggered SDF (reuse `ibm_solid_mask_k` shape);
  the residual-unscale kernel may live here next to `ibm_scale_k`.
- `src/distributed_ns.cuh`: `ensure_vmg_built` — build the 3 per-component staircase coarse hierarchies
  from the staggered θ; the vmg branch — set `descale_lvl0_ = descale_[c]` and enable `vel_unscale_`;
  keep the smoother on `As_[c]` (row-scaled).
- `src/sdflow_bindings.cu`: no change (`set_velocity_multigrid` already exposed).

## Validation (gate each phase)
1. **Z&H SC sphere** (`scripts/validate_zick_homsy_sdflow_vs_pnm.py`): vel-MG drag == RB-GS drag to
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

**Upwind-convective coarse operator (Phase 4, the remaining piece, task #56).** Only needed for
*implicit-advection* large-dt high-Re **steady-state acceleration** (where the velocity operator is
advection-dominated/stiff). Design, now fully mapped:
- Fine operator already exists: `build_adv_stencil_k` → `sadv::fou_operator` adds first-order-upwind
  advection to the diffusion stencil from the advecting u,v,w at unit spacing (`As_[c]`).
- **Coarse** kernel `mg_build_adv_stencil_k`: aniso diffusion (per-axis β) **+** a coarse FOU built from
  `sadv::adv_vel` on the *restricted* coarse velocity, with per-axis advective spacing `dt·(1/cfac_fd)`
  (the FOU coefficient is `dt·vel/h_L`; `fou_operator` assumes h=1, so scale vel by 1/cfac per face dir).
- Per Picard iteration: restrict the advecting u,v,w to every coarse level (reuse `mg_restrict_k`),
  rebuild coarse stencils, fine = `As_[c]`, solve. Upwind keeps the operator an **M-matrix** (diagonally
  dominant) → RB-GS smoothing + coarse correction stay stable for advection-dominated rows.
- **Gate:** lift `!implicit_fou_`. NB **implicit advection is currently IBM-only** (`implicit_fou_ &&
  ibm_enabled_`, distributed_ns.cuh:633) — to use this on cavity/BFS it must also be wired into the
  domain-BC velocity path. Validate: a high-Re implicit-advection steady case, non-symmetric MG stability,
  drag/x_r == RB-GS. Large multi-file change — do with fresh context.
