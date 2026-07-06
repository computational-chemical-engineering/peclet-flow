# Consistent porous (Model B) + semi-implicit drag: momentum & pressure — implementation plan

Status: **plan / for review** (not implemented). Companion to `variable_density_projection.md`
(the validated variable-coefficient projection this mirrors) and
`../../docs/MULTIPHYSICS_PLAN.md` §Phase 6 (which *deferred* Model-A porous terms — this is that
follow-up, done consistently). Reproducer for the current defect: `scratchpad/pureflow.py`.

## 1. Goal

Make the volume-averaged (porous) CFD-DEM fluid step numerically consistent and stable by routing the
drag through the **same face-consistent coefficient machinery** the variable-density projection uses,
instead of the ad-hoc `addDragDiagonal` (cell β) + `buildPorousCoeffDrag` (face β) mismatch. Target:
the fixed-bed and fluidized-bed cases run to ≥100 steps, finite, on **both** CUDA and OpenMP, and the
incompressible / single-phase / non-drag paths stay byte-identical.

### Decisions (fixed by review)

- **β lives inside `idiag`** — the momentum diagonal, correction, and Poisson operator all read one
  face value `A_{p,f}`. No separate `addDragDiagonal`.
- **Model B** pressure: the fluid feels the full `−∇p` (no `−ε∇p`); ε-buoyancy is carried by the drag.
- **Non-conservative inertia** with `ρε` as a *prefactor*: `ρε(∂u/∂t + u·∇u)` — `ρε` stays **outside**
  both the time and convection derivatives (chosen for stability: a clean `ρε_f/dt` diagonal, no
  `ε^n`/`ε^{n+1}` split, and convection is just a weight change `ρ → ρε_f`, not a new flux operator).
- **Full deviatoric viscous stress** `∇·[εμ(∇u + ∇uᵀ)]` — `ε` and `μ` **inside** the divergence, and
  the transpose (deformation-tensor) term included (the variable-viscosity path today is Laplacian-only
  `∇·(μ∇u)`; the transpose is its known deferred upgrade — see `variable_viscosity_projection.md` §6.1).
- **Convection**: implicit FOU + optional deferred-correction TVD, reusing the existing advection with
  the weight `ρ → ρε_f` (exactly how varRho uses `ρ_f`).
- **Drag-consistent incremental predictor**: keep the rotational/incremental pressure, but with `β_f`
  in `A_{p,f}` the predictor's pressure response `−∇P^n/A_{p,f}` and the corrector's `w_f∇φ` share the
  same `A_{p,f}` — which is exactly what the current cell-β diagonal breaks (largest at the
  porosity/β jump; §2 shows why that diverges today).

### Cross-check vs MFIX (Syamlal 1998, *MFIX Documentation: Numerical Technique*)

What MFIX confirms we have right:
- **Drag in the momentum diagonal `a_p` → pressure-correction coefficient `d = 1/a_p`** (§7). This is
  exactly our `β_f` in `A_{p,f}` → `w_f = idt/A_{p,f}`. The core of this plan.
- **Deferred-correction higher-order convection** via down-wind factors (§2, App. C) — our implicit
  FOU + deferred TVD.

What MFIX has that we **do not** — candidate missing stability ingredients:
- **Under-relaxation of velocity AND pressure** (§10.1): "To ensure the stability of the calculations
  it is *necessary* to under-relax the changes in the field variables." Applied to the *coefficient*
  (Eq. 10.2: `(a_p/ω)φ_P = Σ a_nb φ_nb + b + ((1−ω)/ω)a_p φ_P^old`, `0<ω<1`), plus **selective
  under-relaxation in packed regions** where the solids fraction is rising (§8.5). We have **none** —
  and an under-relaxed pressure directly damps the incremental-predictor overshoot I measured. This is
  the most likely additional lever and should be a first-class part of the scheme (§3.6, Phase 5).
- **Partial Elimination Algorithm (PEA, Spalding 1980)** (§6): MFIX makes the *two-way* interphase drag
  implicit — "decoupling by [lagging] the interphase transfer terms will make the iterations unstable
  or force the time step to be very small" for strong coupling. Our semi-implicit drag makes the
  *fluid* side implicit (`β` on the diagonal) but the *particle* side stays explicit (`u_p` frozen over
  the DEM substeps) — a half-PEA. Fine for a fixed bed (`u_p=0`); the proper fix for the *moving*-bed
  coupling (the "Layer-2" `u→β` feedback), noted as a follow-up (§6).

MFIX default momentum is **conservative + Model A** (`−ε∇p`); we chose non-conservative + Model B by
your call. Both are supported models; noting the deviation, not flagging it.

## 2. The defects (RESOLVED 2026-07-06 — measured root causes, superseding the first diagnosis)

The porous fixed/fluidized-bed blowups were **two independent code defects**, found by bisection on
`scratchpad/pureflow.py` (synthetic fixed bed, hand-set ε/β, no particles):

1. **GraphAMG bottom solve diverges on domain-BC operators.** `configurePorousDragSolver()` forced
   `mg_.setGraphAmgBottom(true)` for every porous+drag run; on a domain-BC problem (Dirichlet-outflow
   rows + drag-weighted supplied coefficients) the bottom solve diverges — a hard-converged PCG goes
   NaN **within one solve** (β=0 exactly, i.e. numerically identical coefficients to the working
   no-drag path, still NaNs: pure code-path defect). Periodic/IBM GraphAMG is fine. A second API bug
   hid this: `set_pressure_graph_amg(False)` only set the flag, applied at the next `set_solid` — so
   it silently never turned AMG off. *Fix:* setter propagates live; porous+drag defaults GraphAMG only
   when `!hasBc_`. (GraphAMG∧domain-BC itself remains an open bug, documented here.)

2. **The drag never entered the momentum operator on the all-fluid domain-BC path.** With no immersed
   solid and no advection (`advect_=false` default — the fluidized-bed example never enables it),
   `bcStencilPath()` was false and `smoothComp` fell through to the **const-coefficient** fold smoother
   (`Ac = ρ/dt + 6μ` computed inline) — `rebuildStencils`+`addDragDiagonal` faithfully built the
   drag-loaded band every step and the solve never read it. The momentum response to the incremental
   `∇P^n` was `1/idt` while the projection's operator+correction used `w_f = idt/(idt+β_f)`: pressure
   loop gain `= |1 − (idt+β)/idt| = β·dt/ρ` — **measured 3.84 vs predicted 3.85** at β=77, idt=20;
   marginal at β=idt (measured 0.99); stable below. *Fix:* `bcStencilPath()` now includes `hasDrag_`,
   so drag problems use the assembled stencil (the path validated for BC+solid/implicit-advection).
   The cylinder case (`set_solid` ⇒ `hasSolid_`) always used the stencil path — it only suffered
   defect 1.

The face-vs-cell β consistency (the original §2 diagnosis) is real but secondary: `addDragDiagonal`
now face-averages β under `porous_` (matching `buildPorousCoeffDrag`/`projectCorrectPorousDrag`), so
the exact-cancellation identity of §3 holds at coefficient jumps too. Result: square + cylinder fixed
beds run stable with the incremental scheme ON, porous residual → 1e-11 (synthetic) / steady (real
deposits), ΔP_bed = Σβ_f·w_f·h; the moving cylinder bed fluidizes 200+ steps, finite, CUDA + OpenMP.

## 3. Target scheme (staggered, divided-by-dt, component `c` at face `f` between cells `i−1, i`)

Notation: `idt ≡ ρ/dt` (scalar, constant gas ρ); face means `ε_f = ½(ε_i+ε_{i−1})`,
`β_f = ½(β_i+β_{i−1})`, `open_f` = cut-cell face openness. Solver pressure `φ`, physical `P`, with the
existing scaling `δp = P^{n+1}−P^n = idt·φ`.

### 3.1 Continuous (Model B, non-conservative inertia, constant ρ)

```
∂ε/∂t + ∇·(εu) = 0
ρε( ∂u/∂t + u·∇u ) = −∇p + ∇·[ εμ(∇u + ∇uᵀ) ] − β(u − u_p) + ρε g
```

`ρε` is a prefactor on the inertia (outside the derivatives); `εμ` and the transpose are inside the
viscous divergence.

### 3.2 Discrete momentum predictor (u*, incremental — old pressure P^n)

```
A_{p,f} u*_f  +  ρε_f·(u·∇u)*_f  −  [∇·(εμ∇u*)]_f  =  idt·ε_f u^n_f  +  β_f u_{p,f}  +  f_f  +  V^T_f  −  (P^n_i − P^n_{i−1})
```

with the **face diagonal (β and ε folded in)**

```
A_{p,f} = idt·ε_f + β_f            ← this is the whole point: one face value   (idt ≡ ρ/dt)
```

- **Time term** `ρε_f/dt = idt·ε_f` (non-conservative `ρε ∂u/∂t`: diagonal `idt·ε_f`, RHS
  `idt·ε_f u^n` — the *same* `ε_f`, no `ε^n`/`ε^{n+1}` split).
- **Drag** `β_f` on the diagonal; target `β_f u_{p,f}` on the RHS (face-averaged — today `force_*`
  is used cell-wise; §4 Phase 3).
- **Convection** non-conservative `u·∇u` with weight `ρε_f` — reuse the existing implicit-FOU +
  deferred-correction advection, only the FOU weight changes `ρ → ρε_f` (as varRho uses `ρ_f`). No new
  conservative-flux operator.
- **Viscous** full deviatoric stress `∇·[εμ(∇u+∇uᵀ)]`, split:
  - *normal* `∇·(εμ∇u)` **implicit** — the variable-coefficient band with `FaceProps.beta(i,j) =
    faceMean(ε_iμ_i, ε_jμ_j)` (extends `ibmBuildDiffusionVar`);
  - *transpose* `V^T = ∇·(εμ∇uᵀ)` **explicit / deferred-correction** in the RHS (couples components via
    cross-derivatives; NEW — variable-viscosity is Laplacian-only). Vanishes for uniform `εμ`.

### 3.3 Continuity (the constraint the projection enforces)

```
∇·(open · ε_f · u^{n+1}) = −(ε^{n+1} − ε^n)/dt
```

### 3.4 Pressure Poisson (derived so it telescopes)

Correction `u^{n+1}_f = u*_f − (1/A_{p,f}) ∂_f δp`; substitute into 3.3, use `δp = idt·φ`:

```
∇·(C_f ∂φ) = ∇·(open·ε_f·u*) + ∂_t ε
C_f = open_f · ε_f · w_f            w_f = idt / A_{p,f} = idt / (idt·ε_f + β_f)
```

Difference from today: `w_f` denominator gains the `ε_f` on the time term (from the conservative
inertia). Limits: `ε=1 ⇒ w_f = idt/(idt+β_f)` (recovers today's form); dense bed `β_f ≫ idt·ε_f ⇒
w_f → 0` (drag holds the velocity, pressure barely moves it — the intended stiff-drag behaviour).

### 3.5 Correction & rotational pressure update

```
u^{n+1}_f = u*_f − w_f ∂_f φ
P^{n+1}   = P^n + idt·φ − μ·r        r = ∇·(open·ε_f·u*) + ∂_t ε   (the porous residual → 0)
```

`div_` already holds exactly this porous residual `r`, so the rotational term is unchanged.

### 3.6 Consistency conditions (acceptance for an a-priori check)

1. `A_{p,f} = idt·ε_f + β_f` identical in: momentum diagonal, `w_f` (correction), `w_f` (operator).
2. Continuity flux `open·ε_f·u` and operator `ε_f` weight match.
3. On a converged solve of a *fixed* bed, the discrete `r = ∇·(open ε u) + ∂_tε → machine zero`
   (telescoping check — the current scheme cannot reach this; the fixed version must).
4. ε≡1 ⇒ every porous factor is 1.0 in FP ⇒ **incompressible path byte-identical** (guard on
   `porous_ && hasDrag_`).

## 4. Implementation phases

Each phase is independently testable; non-porous/non-drag paths are guarded so they never change.

### Phase 1 — β (and ε) into `idiag`; delete `addDragDiagonal`
- `flow/src/face_props.hpp`: add `PorousDragFaceProps` (mirrors `VarFaceProps`): carries `ε`, `β`
  fields + `idt` + component stride `sc`; `idiag(i) = idt·½(ε_i+ε_{i−sc}) + ½(β_i+β_{i−sc})`;
  `beta(i,j)` = constant μ (v1).
- `flow/src/flow_ibm.hpp`: in the momentum-stencil build path, when `porous_ && hasDrag_`, assemble
  the diffusion band with `ibmBuildDiffusionVar` + `PorousDragFaceProps` (the validated Var kernel),
  instead of the constant `ibmBuildDiffusion` + `addDragDiagonal`.
- **Remove** the three `addDragDiagonal(c)` call sites (813, 1143, 1189); keep the method only if the
  non-porous `enable_drag()` (Model-B incompressible) still wants it — otherwise fold that path in too.
- RHS: `buildRhsForced` time term becomes `idt·ε_f^n·u^n` (ε-weighted) for the porous+drag case.

### Phase 2 — convection weight `ρ → ρε_f`
- Non-conservative `u·∇u` (existing `sadv::advect` / implicit-FOU + deferred correction), with the FOU
  weight `fouw` and the explicit advection weight changed from `ρ` to `ρε_f` for `porous_ && hasDrag_`
  — the same hook varRho already uses for `ρ_f`. No new operator; guard leaves constant-ρ/varRho
  untouched.

### Phase 3 — full deviatoric viscous `∇·[εμ(∇u+∇uᵀ)]`
- *Normal* part (implicit): `FaceProps.beta(i,j) = faceMean(ε_iμ_i, ε_jμ_j)` in the
  `PorousDragFaceProps` from Phase 1, so `ibmBuildDiffusionVar` assembles `∇·(εμ∇u)` directly.
- *Transpose* part (explicit, deferred): a new kernel for `V^T_i = ∂_j(εμ ∂_i u_j)` on the staggered
  grid (cross-derivatives; `εμ` at the edge/corner between the two face velocities), added to the RHS
  each outer iteration. Vanishes for uniform `εμ` → guarded, so no effect on the non-porous paths.
- This is the largest new piece and the main accuracy (not stability) item — could ship after Phases
  1–2/4 land the stability fix, if you want to stage it.

### Phase 4 — projection coefficient + face-consistent drag target
- `flow/src/mac_pressure.hpp`: `buildPorousCoeffDrag` → `C_f = open·ε_f·w_f`,
  `w_f = idt/(idt·ε_f + β_f)`; `projectCorrectPorousDrag` → same `w_f`. (Add the `idt·ε_f` term to the
  denominator; today it is `idt`.) Pass `ε_f` (already available as `eps1_`) alongside `β`.
- Drag **target**: use the face value `β_f u_{p,f}` in the momentum RHS. Cleanest: keep the coupling
  depositing `force_* = β u_p` (cell) and face-average it in `buildRhsForced`
  (`½(fb_i + fb_{i−sc})`), consistent with `β_f`.

### Phase 5 — stability: verify predictor + add under-relaxation
- With `A_{p,f}` now identical in predictor/corrector/operator, confirm on `pureflow.py` (fixed ε+β
  jump) that pmax stays bounded with incremental **on** (this alone may suffice — the jump was the
  defect).
- **Implement pressure (and optionally velocity) under-relaxation, but default OFF** — the face-fix is
  expected to suffice (user experience: under-relaxation usually not needed); keep it as a lever to
  switch on only if instability persists. `P^{n+1} = P^n + ω_p·(idt·φ) − μ r`, `0 < ω_p ≤ 1`, via
  `set_pressure_underrelax(ω_p)`, **default `ω_p = 1.0` (off)**. Optional selective under-relaxation in
  packed cells (MFIX §8.5) only if needed.
- Decision gate: if the face-consistent `A_{p,f}` + a modest `ω_p` hold `pureflow`/`frozen`/fluidized
  bed, keep the incremental scheme; else default `set_incremental_pressure(False)` for porous+drag.

### Phase 6 — validation (see §5)

## 5. Files & tests

**Files:** `flow/src/face_props.hpp` (`PorousDragFaceProps`: `idiag = idt·ε_f+β_f`, `beta = εμ` face
mean), `flow/src/flow_ibm.hpp` (stencil build via `ibmBuildDiffusionVar`, `ρε_f` advection weight,
ε-weighted time RHS, face-averaged drag target, remove `addDragDiagonal`, transpose viscous kernel),
`flow/src/mac_pressure.hpp` (`w_f = idt/(idt·ε_f+β_f)` in `buildPorousCoeffDrag`/`projectCorrectPorousDrag`).
No binding/API change.

**Tests (both backends, CUDA + OpenMP):**
- `scratchpad/pureflow.py` — fixed ε+β bed: **must stay finite** with incremental on (the minimal
  reproducer, currently NaN at step ~8).
- **New** `coupling/tests/test_fixed_bed_ergun_porous.py` — the Ergun ΔP benchmark with `porous=True`
  (the existing test is Model B / `porous=False`); must land on the Ergun curve.
- `frozen.py` (settled bed, `move_particles=False`) and the full `examples/fluidized_bed.py`
  (`porous=True`) — ≥100 steps, finite, and the bed fluidizes (95th-pct height rises with `U>U_mf`).
- `flow/tests/regression/sdflow_regression.py` + `tests/kokkos_mpi` — unchanged (guards keep
  non-porous byte-identical).
- A-priori consistency check (§3.6.3): on a converged fixed-bed solve, `max_porous_residual → ~1e-12`.

## 6. Resolved (from review) + remaining questions

**Resolved:**
- **Inertia** — non-conservative, `ρε` prefactor (`ρε(∂u/∂t + u·∇u)`), for stability. `w_f =
  idt/(idt·ε_f + β_f)`. (Edge note: `β=0, ε<1 ⇒ w_f = 1/ε_f > 1`; doesn't occur physically — particles
  present ⇒ drag present — but I'll add a guard.)
- **Viscous** — full deviatoric `∇·[εμ(∇u+∇uᵀ)]`, `εμ` inside; normal part implicit, transpose
  explicit/deferred (Phase 3).

**Remaining:**
1. **Drag target face-averaging** — face-average `force_*` in the RHS (Phase 4), or have the coupling
   deposit a face-consistent target? I lean toward face-averaging in flow (keeps the coupling simple).
2. **Non-porous `enable_drag()` (Model-B incompressible)** — leave its `addDragDiagonal` path as-is
   (validated Ergun), or migrate it to the FaceProps diagonal too for uniformity? I'd leave it.
3. **Transpose viscous term staging** — land Phases 1–2 + 4–5 first (the stability fix) and add the
   transpose (Phase 3, accuracy) after, or do it all in one pass?
4. **Bulk term** — the deviatoric stress sometimes carries `−⅔μ(∇·u)I`; you specified `∇u+∇uᵀ` only, so
   I've left it out. Confirm.
5. **Incremental default** (Phase 5 gate) — keep incremental for porous+drag if `pureflow` is stable,
   else default non-incremental.

## 7. Future: two-way coupling stability (PEA) — deferred decision

The scheme above stabilizes the **fluid** solve (drag on the fluid diagonal, face-consistent). It does
**not** address the *moving-bed* two-way instability (the `u→β` feedback: the particle drag is applied
frozen over the DEM sub-steps, which overshoots when the drag is stiff, `m_p/β < Δt`). MFIX handles
this with the **Partial Elimination Algorithm** (§6 of Syamlal 1998); documenting what CFD-DEM PEA
would need, so it's ready when we decide to do it.

**TFM PEA is a cell-local 2×2** (both phases Eulerian, co-located). **CFD-DEM PEA is per-particle**:
eliminate `u_p` from the particle EoM and deposit a *modified* coefficient — the P2G/G2P machinery is
unchanged. Per particle (`γ_p=β_pV_p`, `a_p=m_p/Δt`, `ũ_g`=fluid velocity interpolated to `x_p`,
`f_p^{ext}`=gravity+contacts):

```
u_p' = (a_p u_p + f_p^{ext} + γ_p ũ_g)/(a_p+γ_p)
reaction on fluid = γ_p^{eff}·(ǔ_p − ũ_g),   γ_p^{eff} = γ_p·a_p/(a_p+γ_p),   ǔ_p = u_p + f_p^{ext}/a_p
```

So PEA ⇒ **deposit `γ_p^{eff}` and `γ_p^{eff}·ǔ_p`** into `drag_beta`/`force_*` instead of `γ_p` and
`γ_p u_p` — same trilinear deposit, same summation. `γ_p^{eff}` saturates at `m_p/Δt` for stiff drag
(the fluid never sees more than the particle's inertial resistance). P2G being the adjoint of G2P
keeps it conservative.

**The catch (why it's deferred, not trivial):** `a_p=m_p/Δt` is the **free-particle** inertia. In a
packed bed the grain is held by the contact network — its response is set by the bed's rigidity, so the
correct limit is the fixed bed (`γ_p^{eff}→γ_p`, i.e. `a_p→∞`), which the naive formula **over-reduces**.
TFM gets this rigidity from the solids-stress term in `a_p`; CFD-DEM has no continuum stand-in — it's the
resolved contacts in the DEM sub-steps.

**Options, in order of preference:**
1. **Implicit particle-drag per DEM sub-step** — `u_p' = (a_p u_p + f + γ_p ũ_g)/(a_p+γ_p)` each
   sub-step instead of a frozen drag force. Cheap, local, no deposition change; correct dilute, and
   degrades gracefully in dense regions (the XPBD contacts in the same sub-step supply the rigidity).
2. **Fluid↔DEM Picard sub-iterations** (2–3) per fluid step — captures the true contact-constrained
   response; robust, higher cost.
3. **Avoid** the deposited-`γ^{eff}` form with `a_p=m_p/Δt` alone — it over-reduces the drag in packed
   cells (needs a contact-aware `a_p`, i.e. reinventing the solids stress).

**Recommendation when we take this up:** start with (1); add (2) only if the dense two-way coupling
still bites. The fluid side is already implicit (raw `β` deposited → stable sink on the diagonal, Poisson
unaffected), so PEA is purely a *particle-side / two-way-consistency* upgrade — not needed for the
stationary bed.
