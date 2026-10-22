# Full deviatoric viscous stress for variable μ in `peclet.flow` — design note

Status: DESIGN (architect, 2026-10-22). It is to be committed to flow as `doc/full_viscous_stress.md`.
Brief: `bubble_column_perf/validation/ARCH_BRIEF_FULLSTRESS.md`. Evidence: `CASE_A_SETUP.md`,
`CASE_A_ABLATE.md`. Current operator: flow `doc/variable_viscosity_projection.md`, cited below as VVP.

## 0. Summary

- **Diagnosis.** The +20 % in case A comes from the omitted transpose term. This is a quantitative verdict.
  - The Laplacian form ∇·(μ∇u) imposes the wrong tangential jump condition, [μ ∂ₙu_t] = 0. The correct
    condition is [μ(∂ₙu_t + ∂_t uₙ)] = 0.
  - For a Hadamard–Rybczynski drop at viscosity ratio κ = 0.1, the Laplacian form gives drag 3.5πμUR
    instead of 4.18πμUR. The rise velocity is therefore **+19.5 %**, independent of h.
  - Measured: our d/h 40 result divided by Basilisk's d/h 40 result is **+20.1 %**. At κ = 1 the predicted
    error is 0, and A3 shows ratio 1 converged.
  - Nothing else is needed to explain the offset.
- **Decision.**
  - Discretization: the full stress on the staggered grid. The transpose term is the **conservative MAC
    stress-element form**:
    - the normal stresses sit at cell centres, the shear stresses on cell edges;
    - each uses the *same* μ that the existing Laplacian band uses at that location.
  - Time treatment: the term is **explicit and lagged on uⁿ** (the projected field). It is added to the RHS of
    the existing per-component solve and is unconditionally stable for the predictor (§6).
  - Bit-identity: the term is skipped exactly when μ is globally uniform.
  - Collocated: refused in v1, planned in §9.
- **Pre-registered prediction:** case A at d/h 40 gives Re ≈ 25.54/1.195 = **21.4 ± 0.4**. Basilisk gives
  21.275.

---

## 1. Problem and scope

**Built.** On the staggered `Solver`, whenever the viscosity is variable (`varProps_`), the momentum
predictor gains the transpose term

  T = ∇·(μ (∇u)ᵀ),  T_c = Σ_a ∂_a(μ ∂_c u_a),

so that the predictor approximates ∇·(μ(∇u + ∇uᵀ)) = ∇·(2μD). The term covers:
- VoF with a μ closure, single-field or blocks;
- the `enable_vof_momentum` path;
- `set_field("mu")` and closure-driven μ (Arrhenius) in all-fluid domains and with static SDF solids.

**Out of scope:**
- SRP (not required, §7);
- the collocated implementation (planned, not built: §9);
- AMR;
- cases B–D;
- near-contact models;
- release work;
- a coupled or implicit 3-component solve, and variable-coefficient velocity MG (deferred, §4.2).

**Refused in v1, with a clear error:** collocated; the harmonic face mean; porous (ε-weighted) momentum;
moving scene instances.

## 2. Diagnosis and a-priori magnitude

**2.1 What the Laplacian form gets wrong.**
- Inside each phase (μ constant, ∇·u = 0) both forms are identical.
- They differ only through the interface jump condition. Integrating the u_i equation across a thin layer
  gives:
  - with ∇·(μ∇u): [μ ∂ₙu] − [p]n = σκn;
  - with the full stress: [μ(∂ₙu + (∇u)ᵀn)] − [p]n = σκn.
- (∇u)ᵀn is continuous across the interface (both of its parts, ∂_t uₙ and ∂ₙuₙ = −∇_s·u_t, are), so the
  product with the δ-function is well defined.
- The defect is therefore a **surface force [μ](∇u)ᵀn**, which does not shrink with h. It is zero for:
  - uniform μ;
  - layered μ(y) with flow along the layers (two-layer Couette).

  It is nonzero for a circulating drop. On a sphere with uₙ = 0 it equals Δμ·u_θ/R, tangential.

**2.2 Stokes-limit size (exact).** Hadamard–Rybczynski with the tangential condition generalized:
- s = 2 is the full stress: μ_l(∂_r u_θ − s u_θ/R)ₗ = μ_b(∂_r u_θ − s u_θ/R)_b, written with
  f″, f′ of the Stokes stream function;
- s = 1 is the Laplacian form.

Solved (4×4 system, checked numerically):

| κ = μ_b/μ_l | F_full / (πμ_l U R) | F_Laplacian / (πμ_l U R) | U_Lap / U_full |
|---|---|---|---|
| 0 | 4.000 | 3.000 | 1.333 |
| **0.1** | **4.182** | **3.500** | **1.195** |
| 1 | 5.000 | 5.000 | 1.000 |
| 10 | 5.818 | 5.857 | 0.993 |

Closed form: F_Lap = 3πμUR(1+4κ)/(1+2κ), against Hadamard–Rybczynski 2πμUR(2+3κ)/(1+κ).

**2.3 Against the evidence.**

| observation | the omitted-term model predicts | measured |
|---|---|---|
| Re(κ = 0.1) / Re_lit | 1.195 (Stokes; finite Re moves this by a few %) | 25.54/21.275 = **1.201** at d/h 40; ≈ 1.22 extrapolated |
| κ = 1 (A3) | no error, converges | 17.19 / 17.14, converged |
| the error does not shrink with h | O(1) surface force | increments 4.4 % → 1.6 %, converging to ≈ 25.8 (a consistent scheme for the wrong jump condition) |
| harmonic vs arithmetic face μ | O(h): the face mean multiplies first derivatives, not δ-like terms | gap 0.96 / 0.49 / 0.24 |
| dt/2 | no effect | +0.02 % |

**2.4 The other candidates.**
- **The one-step μ lag** displaces μ by U·dt behind the interface, which is proportional to dt. A2 (dt/2 →
  +2e-4) bounds its whole effect to ≲ 0.05 %. **No package.**
- **`enable_vof_momentum`** (+11–15 %): the fix should bring the velocity form to ≈ 21.4. The conservative
  form would then still sit ≈ +10 % high. It looks like a *second, independent* inconsistency in the opt-in
  path.
  - Re-measure it in WO-5.
  - If the gap is still > 2 % at d/h 20 and 40, open a separate package. Its first test: a translating
    two-phase field (u ≡ U, ρ and μ jumps, no g, no σ) must keep u ≡ U to rounding in both forms.
  - It does not block case A.
- **If G5 misses**, investigate in this order:
  1. the steady closure acceleration (≈ 0.02 g, a net mean force of the velocity form). It shifts the
     effective buoyancy by ~2 %, which is ≈ 1.7 % in Re: report it with full stress on;
  2. the vof_momentum package;
  3. the Re frame.

## 3. Constraints and invariants

- **Grid conventions.**
  - Cell index i = x + y·e.x + z·e.x·e.y (x fastest). Stride s_a: s_x = 1, s_y = e.x, s_z = e.x·e.y.
  - The staggered node (c, i) is u_c on the −c face of cell i, between cells i − s_c and i. This is
    consistent with `gp = w_c·(P(i) − P(i−s_c))`.
  - Loop range: inner cells [G, e−G) per axis, the same as `rhs_var`. Loops go through `ccFor3`/`MDRange3`,
    x fastest (ctest `iteration_order`).
- **Units.** Index velocity v_a = u_a·t_ref/h_a. Metric weight w_a = (h_ref/h_a)², exactly 1.0 on the
  isotropic path (flow `doc/anisotropic_metric.md` §1.2).
  - The Laplacian band of u_c across an a-face is β·w_a, i.e. the weight of the derivative axis.
  - Change of variables for the transpose term: **T_c carries w_c, the weight of the component axis**, for
    every a. Derivation: ∂_a(μ∂_c u_a) = (1/(h_c t_ref))∂_ξa(μ∂_ξc v_a), and the c-equation is divided by
    ρ_ref h_c/t_ref².
- **μ placement.** The transpose term uses *the same doubles* as the existing bands:
  - β_c(i, j) = `makeFaceProps(c).beta(i, j)` before the ×w multiplication;
  - along c this is μ(i) or μ(i−s_c);
  - across a it is the edge value ½[μ_f(i, i±s_a) + μ_f(i−s_c, i−s_c±s_a)].

  This is what makes rigid rotation exact (§5.3).
- **Bit-identity.**
  - Feature off: every existing path is byte-identical. The gate is `tests/regression/state_hash.py`, 14
    cases, host and CUDA.
  - Feature on with uniform μ (μ ratio 1 included): byte-identical to off.
  - The two-layer Couette stays exact.
- **Platform rules.**
  - On-device, MPI-distributable, deterministic per node (no atomics).
  - No global `Kokkos::fence` in `step()` (ctest `device_sync_budget`). Instance-scoped reductions to the
    host are allowed, as in `minMuInner()`.
  - No environment variable; physical-unit API only.
- **Settled, untouched.**
  - Backward Euler.
  - The incremental predictor and the rotational update with μ_rot = χ·μ_min.
  - Forces inside the predictor.
  - Velocity MG off under varProps.
  - Never assemble the block force from the union colour.

## 4. The decision and the rejected alternatives

### 4.1 Discretization: conservative stress elements on the MAC grid (chosen)

T_c is the divergence of stress elements:
- normal stresses μ∂_c u_c at **cell centres**;
- shear stresses μ_e∂_c u_a at **cell edges**.

Each element uses the μ that the Laplacian already uses at the same place.

Properties, all by construction:
- the stress tensor μ_e(∂_a u_c + ∂_c u_a) at an edge is symmetric;
- **exact telescoping**, so the net force is zero on a periodic box;
- **rigid rotation gives exactly zero force** (to rounding);
- the a-priori energy inequalities of §6 hold, which makes the explicit treatment unconditionally stable;
- second order for smooth μ;
- the correct integrated jump condition at a sharp μ jump (conservation).

Rejected:
- **(R1) The product form (∇μ)·(∇u)ᵀ.** Its discrete version Σ_a (δ_aμ)·avg(δ_c v_a) differs from the
  conservative form by Σ_a μ̄_a δ_aδ_c v_a. Here μ̄_a is the a-direction average of μ, and it differs between
  directions by O(Δμ) inside the interface band. At an oblique interface the velocity *gradient* jumps:
  [∇u] = a⊗n with a·n = 0, so [∂_y u_y] = a_y n_y ≠ 0. The second differences are then δ-like (∝ 1/h), and
  their weighted difference integrates to an **O(1) surface-force error**. That is the same class of defect
  being fixed. The form is exact only for grid-aligned interfaces. It also passes the rigid-rotation test,
  so the arbiter is gate G2 (kinked manufactured solution), not rigid rotation.
- **(R2) The excess-viscosity form Σ_a δ_a((μ_e − μ_ref)δ_c v_a)**, with a global constant μ_ref. Its
  appeal is that it is bitwise zero for uniform μ. But:
  - it needs the same global reduction as the chosen skip;
  - it drops a +μ_ref·G·D term, which breaks the bound T ≤ K for compressive modes unless
    μ_ref ≤ (2/3)μ_min, because (∇·u)² ≤ 3|D|²;
  - it gains nothing over (C) of §4.4.
- **(R3) Changing the Laplacian edge μ.** For example, a symmetric 4-cell harmonic for both parts. This is
  not needed under the arithmetic mean, and it would move the validated Laplacian bands.

### 4.2 Time treatment: explicit, lagged on uⁿ, once per step (chosen)

T_c(uⁿ) is evaluated from `old_` (the time base uⁿ, the projected field). It is added to the RHS in
every Picard iteration and is constant across iterations. Nothing changes in the implicit operator, the
smoother, the projection or the rotational update.

- **Stability (§6).** The predictor (M/dt + K)u* = (M/dt)uⁿ + T uⁿ + … has amplification
  λ = (m + ⟨T⟩)/(m + ⟨K⟩) ∈ (−1, 1] for *every* dt. Large dt is safe.
- **Accuracy.** The lag error is O(dt), the same order as backward Euler. In case A the viscous number is
  D = μ dt/(ρh²) ≈ 0.016, so the splitting error is negligible.
- **The steady state** is the fully implicit one: uⁿ⁺¹ = uⁿ ⇒ T(uⁿ) = T(uⁿ⁺¹). It is dt-independent.

Rejected:
- **(R4) The term inside the Picard loop on the iterate u^k.**
  - u* is not solenoidal, so T(u*) carries μ∇(∇·u*), whose explicit amplification tends to −1 at large
    dt.
  - It perturbs the gradient part that the rotational update −χμ_min∇·u* assumes.
  - It converges to the implicit solution at the same rate λ_∞ per iteration, so there is no gain.
- **(R5) A semi-implicit diagonal.** This would make the a = c normal part implicit by doubling the along-c
  band.
  - It is stable too, but changes the validated 7-band kernel and its cut-cell overlay.
  - It does not help the slow modes of §6, which are rotation-dominated and have T_cc ≈ 0.
- **(R6) A fully implicit coupled solve.** This means block-GS across components, or a coupled 3-component
  RB-GS.
  - The slow modes at large dt are *physically* weakly dissipated (⟨μD²⟩ ≪ ⟨μ|∇u|²⟩), so (M/dt + K + T)
    is ill-conditioned there and every stationary smoother is slow on them.
  - Only a coupled variable-coefficient velocity multigrid fixes that, and it is deferred.
  - A coupled smoother without it is a large solver change with no rate gain.

### 4.3 Where it enters `step()` (flow `src/flow_ibm_project.hpp`)

- **Prelude.** Run once per step, right after `old_[c] = C[c].u` (≈ :111), gated by `fullStressActive()`:
  1. If `rebuildStencils()` did not run this step (the `implicitAdv()` path), call `fillMuGhosts()`.
  2. `muUniform_ = (muMin == muMax)`, from a fused min/max reduction over cells [G−1, e−G+1)³ (the inner
     cells plus the one ghost ring that T reads). Use `MPI_Allreduce`, MIN and MAX, under `distributed_`.
     Exact comparison.
  3. If `!muUniform_`: for c = 0, 1, 2, call `fillVelGhostsTo(old_[c], c, 0, /*doOutflow=*/false)`. This
     gives the ghosts of uⁿ incl. edges. It leaves `C[c].u` untouched, so the existing loop is unaffected.
     `doOutflow=false` keeps the projected outflow face.
  4. If solids exist and `regDirty_`, rebuild the regular-node masks (§5.4).
- **In the Picard loop.** Right after the CSF block (≈ :197), before the implicit-FOU stencil rebuild:
  `if (fullStressActive() && !muUniform_) for c: addFullStressRhs(c);`.
  - This is additive, like the CSF.
  - It is independent of which of `buildRhs` / `buildRhsForced` / `buildRhsVar` / `buildRhsVarMom` ran.
- `fullStressActive() := fullStress_ && varProps_ && !Grid::collocated`. It is checked once per step, and
  the refusals of §5.6 are raised at step head.

### 4.4 Bit-identity mechanism

Three alternatives were considered:
- (A) A formula that is bitwise zero for uniform μ. This forces R1 or R2, both rejected.
- (B) A per-node local-uniformity skip.
  - It needs no reduction.
  - But a skipped node drops a full element flux from the telescoping sum. The leak is
    O(μ·∇(∇·uⁿ)) ~ projection tolerance, small but real.
  - Its semantics are harder to state.
- **(C) Chosen: a global exact uniformity test (§4.3 step 2).** T is either the complete conservative term
  or nothing at all.
  - It costs one instance-scoped reduction and a host read per step. There is precedent: `minMuInner()`
    already does this every step. On GPU this is ~20 µs.

## 5. The discretization

### 5.1 Elements (component c, node i; v_a = `old_[a]`; β_c = `makeFaceProps(c).beta`)

```
Normal element at cell k:        N_c(k)    = β_c(k, k+s_c) · ( v_c(k+s_c) − v_c(k) )          [= μ(k)·δ_c v_c]
Shear element, edge (+a), a≠c:   S_c(i,+a) = β_c(i, i+s_a) · ( v_a(i+s_a) − v_a(i+s_a−s_c) )
Shear element, edge (−a), a≠c:   S_c(i,−a) = β_c(i, i−s_a) · ( v_a(i)     − v_a(i−s_c) )

T_c(i) = w_c · ( ( (N_c(i) − N_c(i−s_c)) + (S_c(i,+a1) − S_c(i,−a1)) ) + (S_c(i,+a2) − S_c(i,−a2)) )
bb_c(i) += rs_c(i) · T_c(i)          (a1 < a2: the two other axes, ascending; this association order is fixed)
```

Note: β_c(k, k+s_c) = μ(k) and β_c(i, i−s_c) = μ(i−s_c), as returned by `VarFaceProps::beta`.

The edges:
- edge (+a) is shared by cells i, i−s_c, i+s_a, i−s_c+s_a and by the nodes u_c(i), u_c(i+s_a),
  u_a(i+s_a), u_a(i+s_a−s_c);
- edge (−a) is shared by cells i, i−s_c, i−s_a, i−s_c−s_a.

β is symmetric under (i, j) swap, bitwise: arithmetic `mean2` commutes. Hence S_c(i,+a) ≡ S_c(i+s_a,−a),
and telescoping is exact.

The sign is the force on the RHS: ρ Du/Dt = … + ∇·τ. There is no ρ factor, as for the Laplacian.

### 5.2 Stencil footprint (for MPI)

The kernel reads:
- v_c at i, i±s_c;
- v_a at i, i−s_c (edge −a) and i+s_a, i−s_c+s_a (edge +a), for each a ≠ c.
- μ at i, i−s_c, i±s_a, i−s_c±s_a for both a ≠ c (10 cells).

Ghost width 1. **It needs the edge ghosts in the (a, c) planes and no corner ghosts.**

### 5.3 Exactness identities (they define the a-priori gates)

- **Rigid rotation.** For a linear velocity field the MAC differences are exact. The Laplacian flux
  β·w_a·δ_a v_c and the transpose flux β·w_c·δ_c v_a at the same edge are μ_e∂_a u_c and μ_e∂_c u_a in
  physical units, so they cancel for u = Ω×x.
  - Full force: 0 to rounding.
  - Laplacian alone: (δ_aμ_e)·(∂_a u_c) ≠ 0.
- **Pure strain** u = (γx, −γy, 0): S_x(i) = 2·w_x·γ′·(μ(i) − μ(i−s_x)) exactly, with γ′ = γ t_ref. The
  Laplacian alone gives half of it.
- **Simple shear** u_x = γy: S_y(i) = w_y·γ′·(h_y/h_x)·(β_y(i,i+s_x) − β_y(i,i−s_x)) exactly. The
  Laplacian alone gives 0.
- **Uniform μ:** T_c = μ·w_c·δ_c(D(v)), the gradient of the discrete divergence. It is small, but not
  bitwise zero, hence §4.4.

### 5.4 Solids: regular-node rule (static SDF solids only)

- reg_c(i) = 1 iff `C[c].mask(i) == 0` (not solid) **and** i is not in the cut list of `C[c].ov`. It is
  stored as a `uint8` View per component (3 B/cell, only when solids exist and the feature is on).
- Built in two kernels: from the mask, then zeroing the overlay's `cell_index` list. Exchange the ghosts
  incl. edges.
- `regDirty_` is set by every geometry build.
- Inclusion rules:
  - an element is included iff all velocity nodes it touches are regular: N_c(k) needs k and k+s_c; S_c
    needs its 2 u_c nodes and 2 u_a nodes;
  - a row receives only included elements, so non-regular rows get nothing automatically.
- Consequences:
  - the rule is symmetric in (a, c), so T stays symmetric;
  - telescoping holds on the regular subgraph;
  - near a wall the transpose term is staircase-truncated, an O(h) effect. At a stationary no-slip wall the
    exact transpose traction (∇u)ᵀn = n·∂ₙuₙ = 0 anyway, so **no reaction-force contribution** is lost.
- All-fluid domains take a compile-time `kSolid = false` path with no mask reads.

### 5.5 Domain faces

- No transpose element reads a ghost beyond a wall: the wall-normal face values are imposed data, and
  periodic ghosts are images.
- **Never add T to a row whose value a BC imposes.** These are the low-face node of component c on a
  non-periodic global c-face, on ranks that touch it. The high-face node lies outside the loop range.
- Inflow and outflow face values are read as data.
- Gate (WO-2): in a walled channel with variable μ and full stress on, the wall-normal velocity on wall
  faces stays bitwise equal to its imposed value.

### 5.6 Refusals (raised at step head when `fullStress_` is requested)

- `SolverColocated`. The setter itself raises.
- `harmonic=True`: the series direction is component-dependent, which breaks the symmetry of T.
- `porous_`: the ε-weighted stress is a different equation.
- Moving scene instances: the reg masks would need to follow them.

If μ is constant (`!varProps_`), the feature is inactive with no error: the term is identically zero.

## 6. Time treatment: stability analysis

Write K = −(Laplacian part) (SPD) and T as above, both in the physical-energy inner product
⟨u,v⟩ = Σ_c Σ (1/w_c)·v_c·u_c·V. With arithmetic β, summation by parts edge by edge gives:

```
⟨u,Ku⟩       = Σ_cells μ (∂_c u_c)²  + Σ_edges μ_e [ (∂_a u_c)² + (∂_c u_a)² ]
⟨u,Tu⟩       = −Σ_cells μ (∂_c u_c)² − Σ_edges 2μ_e (∂_a u_c)(∂_c u_a)
⟨K⟩ − ⟨T⟩   = 2Σ μ (∂_c u_c)² + Σ_e μ_e (∂_a u_c + ∂_c u_a)²   = 2⟨μ D:D⟩ ≥ 0   (the full-stress dissipation)
⟨K⟩ + ⟨T⟩   = Σ_e μ_e (∂_a u_c − ∂_c u_a)²                                 ≥ 0
```

T is symmetric in this inner product. M = ρ_F/dt is diagonal and positive. For the predictor
(M + K)u* = (M + T)uⁿ, the iteration matrix is self-adjoint in the (M + K)-norm, and its Rayleigh
quotients are

  λ = (m + ⟨T⟩)/(m + ⟨K⟩) = 1 − 2⟨μD:D⟩/(m + ⟨K⟩) ∈ [(m − ⟨K⟩)/(m + ⟨K⟩), 1] ⊂ (−1, 1].

The predictor is therefore **unconditionally stable** for any dt and any contrast.

- λ = 1 only for rigid motions.
- λ → −1 only for irrotational modes (⟨K⟩ + ⟨T⟩ = 0). These are compressive gradient fields, which the
  projection removes.
- **Uniform μ, solenoidal periodic u:** ⟨K⟩ = 2⟨μD:D⟩, so λ_∞ = 0, identical to the implicit scheme.
- **Large dt, variable μ:** λ_∞ = 1 − 2⟨μD:D⟩/⟨μ|∇u|²⟩. It is close to 1 only for *rotation-dominated
  modes of the high-μ phase* (a high-μ blob rotating rigidly in low-μ fluid), where λ_∞ ≈ 1 − O(μ_lo/μ_hi).
  - These modes are weakly dissipated in reality too (the implicit λ → 0 there only because the implicit
    scheme over-damps them per step).
  - The **cost is steady-march convergence**, not stability or the fixed point. It is measured in G3; see
    Q3.
- What is *not* proven: the composite of predictor, M-orthogonal projection and incremental pressure
  accumulation, for general variable μ. That is covered by the large-dt gates G3.
  - It is consistent with the historical failure mode: the rejected 'full' rotational term failed by
    injecting a non-gradient into the *pressure accumulation*. T never touches the pressure.
- With solids (§5.4): K carries the Robust-Scaled overlay at cut rows, which have no T. T is symmetric on
  the regular subgraph. G3(iii) covers it.

## 7. Solver impact

- **Operator, smoother, bands, overlay:** unchanged. The 7-band kernel, `ibmBuildDiffusionVar`, the cut-cell
  bake, RB-GS and the AUTO-VMG rule (still off under varProps) are all untouched.
- **Projection and rotational update:** unchanged. T acts on the solenoidal uⁿ, so the implicit operator's
  gradient part is still μ∇(∇·u*), and −χμ_min∇·u* stays the consistent rotational term.
- **SRP is not required.** The kernel of §5.1 is exactly the transpose half of ∇·(2μD), so a later SRP
  can assemble ∇·∇·(2μD) from it.
- **Momentum residual stop:** it is evaluated on the modified b, with no change of rule.

## 8. Bit-identity guarantee (stated as checks)

1. **Feature off (the default through WO-6).**
   - Every new call is behind `fullStressActive()`. No existing kernel, loop condition or ghost fill
     changes.
   - Gate: state_hash, 14 cases, byte-identical on host and CUDA.
   - Note: the state_hash VoF cases use constant μ (ρ closure only), so even WO-7 cannot change them. Verify
     this.
2. **Feature on, μ uniform** (constant field or a μ_b = μ_l closure).
   - The prelude reads μ and may refill the μ ghosts with identical values. It writes no velocity, no
     `old_` ghost and no RHS.
   - Gate: A3 (μ ratio 1), d/h 10, full stress on against off: the full Re time series is identical.
3. **Two-layer Couette (μ(y)).** Here δ_x of x-independent quantities vanishes, so T = 0 to rounding.
   - Gate: the max error equals today's value (0.0006 % at dt = 20; 0.0002 % at dt = 100) to a relative
     1e-6.

## 9. Collocated plan (later, not v1)

`SolverColocated.set_viscous_stress("full")` raises in v1. The planned design:
- Compute T at MAC faces from the **projected face field** `uf_/vf_/wf_` with the §5.1 kernel. The face
  field is discretely solenoidal, and there is no wide cell cross-derivative, hence no checkerboard
  coupling.
- Inject it through the V8 mass-adjoint face-force channel of `buildRhsColoVar`, the one the CSF uses.
  This keeps it inside the predictor.

Why it is deferred:
- The collocated Laplacian acts on *cell* velocities with plain face means, while this T acts on *face*
  velocities. The energy pairing of §6 does not hold, so stability needs its own analysis.
- It interacts with the invisible-subspace mechanism (`doc/collocated_invisible_subspace.md`).

Gates when it is built: the §5.3 identities on the face field, `collocated_stability_guard`, and case A at
d/h 20 within 2 % of the staggered result.

## 10. MPI ghost needs

| field | width | edges ((a,c) planes) | corners | provided by |
|---|---|---|---|---|
| μ (`muField_`) | 1 | **yes** (i−s_c±s_a) | no | `fillMuGhosts` → `fillPropGhosts(edges=true)` → `fillGhosts`. Distributed GridHalo g = 2; its declaration says "all ghosts incl. corners". **Unverified → WO-0.** |
| v_a (`old_[a]`) | 1 | **yes** (i−s_c+s_a) | no | `fillVelGhostsTo(old_[a], a, 0, false)`. Single rank: `fillPeriodicAxes`, sequential axes, covers edges. Distributed: velocity halo. **WO-0.** |
| reg_c (solids) | 1 | yes | no | one exchange at build |

Per step this is one extra 3-field velocity exchange and one 2-double allreduce.

If WO-0 finds the μ edge ghosts missing across ranks, the *existing* Laplacian cross-face read is already
wrong under MPI. Fix it first (WO-0b).

## 11. Cost

- **Kernel.** One fused kernel per component per Picard iteration (the default `outerIters_` = 1). It
  reads ~7 double streams: v_c, v_a1, v_a2, μ, rs, and bb read and write. That is ≈ 56 B per node per
  component, ≈ 170 B/cell per step.
- **Bubble column, 786k cells, RTX 5080:**
  - ≈ 133 MB/step, ≈ 0.19 ms;
  - plus 3 periodic ghost fills of `old_` (≈ 0.02 ms) and one reduction with host read (≈ 0.03 ms);
  - total ≈ 0.25 ms on 13.5 ms, i.e. **≈ +1.8 %**.
- **Host:** a similar fraction.
- **Memory:** none persistent on all-fluid domains (no buffer). With solids, 3 B/cell.
- **Case A grids:** small grids are launch-bound, adding ≈ 4 launches and 1 sync ≈ 40 µs on 4–7 ms.

## 12. Work orders (dependency order)

The implementation branch is `full-stress` in a sibling worktree `suite/flow-full-stress`.

**WO-0 — Facts, no production change.**
- A ctest `ghost_edges_mpi_np{2,4}`:
  - np 4 is a 2×2 split, cutting two axes;
  - fill μ-like and velocity-like fields with f = gx + 1e3·gy + 1e6·gz;
  - run `fillPropGhosts(edges=true)` and `fillVelGhostsTo(…, false)`;
  - compare every ghost within width 1, edges and corners included, with the periodic-wrapped global value.
- Log, without gating, whether case A takes the `rebuildStencils()` path and runs `outerIters_ == 1`.
- **Gate G0:** 0 mismatches, host (OpenMP) and CUDA-MPI if built.
- On failure: **WO-0b** fixes the fill in flow's wrapper. Re-run G0, plus the `variable_mu` and
  `mirror_symmetry` ctests.

**WO-1 — Kernel and a-priori operator test.**
- `src/full_stress.hpp`: `addFullStressRhs<FaceProps, bool kSolid>` (§5.1, §5.4) and `muRangeInner()`.
- Host oracle in `tests/kokkos/test_full_stress.cpp`, ctest `full_stress_operator`.
- Not wired into `step()`.
- **Gate G1** (§13).

**WO-2 — Wiring, setter, refusals.**
- §4.3 and §5.5–5.6.
- Public setter `set_viscous_stress("full" | "laplacian")`, default `"laplacian"` (Q1, Q2).
- The reg masks for static solids.
- **Gates:** G-bit (§8 items 1–3); G2 (the rotating-core problem); `mirror_symmetry` with full stress on
  (same tolerance as off); wall-normal face values unchanged (§5.5); `variable_mu` and
  `velocity_solver_variable_mu` unchanged.

**WO-3 — Large dt and steady marches.**
- Study script `tests/study/full_stress_large_dt.py`.
- A short ctest `full_stress_large_dt`: the rotating core at μ ratio 100, D = 1e4, 500 steps.
- **Gate G3.**

**WO-4 — MPI and CUDA.**
- ctests `full_stress_operator_np{2,4}`: a random periodic field, T bitwise against np 1.
- Case A d/h 20 serial ↔ np 2, 4.
- CUDA = host.
- **Gate G4.**

**WO-5 — Physics and performance.**
- Case A d/h 10, 20, 40 with full stress.
- Re-run A3 (ratio 1) and the `enable_vof_momentum` ablation.
- Run d/h 80 only after G5 passes (local GPU, ≈ 65 min).
- Record results in `CASE_A_ABLATE.md` and the state file.
- **Gates G5, G7.**

**WO-6 — Documents.**
- This note goes to `doc/full_viscous_stress.md`.
- VVP: §2 points to it, and §6.1 is marked "resolved on staggered".
- Add the register entry (§14) and the CHANGELOG line.
- One line in flow `CLAUDE.md`, under Settled decisions: "variable μ uses the full stress: explicit
  conservative MAC elements on uⁿ, never the product form, never on the Picard iterate. Guard: ctest
  `full_stress_operator` and case A."

**WO-7 — User-gated: the default.**
- If approved (Q1): AUTO resolves to full on staggered when μ is variable, arithmetic, not porous and not
  moving. Otherwise laplacian, with a one-time stderr notice, as the AUTO-scheme fallback does.
- This is one isolated commit.
- Re-verify state_hash. It is expected to stay unchanged (§8.1).
- Announce that published variable-μ numbers change.

## 13. Verification gates (numbers)

**G0** — Ghost values: 0 mismatches (exact).

**G1** — Operator test, isotropic and anisotropic metric (h ratio 1 : 1.5 : 2). μ jump sharp (sphere,
cell-sampled, κ = 0.1) and smeared (tanh, width 2h).
- (a) Rigid rotation: max|L + T| ≤ 1e-12·max|L|, and max|L| > 1e3·max|L + T|. The second condition shows
  that the defect exists and is cancelled.
- (b) Pure strain and (c) simple shear: §5.3 identities to a relative 1e-13.
- (d) 100 random periodic (u, μ) pairs, μ ratio up to 1e3, arithmetic, no solids:
  - telescoping: |Σ_i T_c| ≤ 1e-13·Σ|T_c|;
  - W-symmetry: |⟨v,Tu⟩ − ⟨u,Tv⟩| ≤ 1e-12·|⟨v,Tu⟩|;
  - ⟨u,(K ± T)u⟩ ≥ −1e-12·⟨u,Ku⟩, with K from `ibmBuildDiffusionVar` at idiag = 0.
  - With a random solid mask, T symmetry only.
- (e) Smooth μ = μ₀(1 + ½ sin 2πx sin 2πy cos 2πz) and a solenoidal trigonometric u: max-norm truncation
  error of (L + T) against the analytic ∇·(2μD) on the unit periodic cube, order ≥ 1.9 for both 32→64
  and 64→128. The Laplacian
  alone must not converge.
- (f) Uniform μ: `muRangeInner` returns min == max, and the wiring skips.

**G2** — Rotating-core manufactured steady Stokes, with a sharp, kinked, curved μ jump. This test
distinguishes R1 from the chosen form.
- Box (N, N, 4) with physical extent (L, L, 4L/N), periodic, centre (L/2, L/2). R = 0.25L, R₂ = 0.45L,
  azimuthal flow.
- u_θ = U₀(r/R)³ for r ≤ R.
- For R < r < R₂, u_θ = H(r), the quintic with:
  - H(R) = U₀;
  - H′(R) = (U₀/R)(1 + 2μ_in/μ_out): continuity of μ(∂_r u_θ − u_θ/r), so the velocity *gradient* jumps;
  - H″(R) = 0;
  - H = H′ = H″ = 0 at R₂.
- u_θ = 0 beyond R₂.
- Forcing f_θ = −μ(u_θ″ + u_θ′/r − u_θ/r²) per region, with no interfacial force. The exact pressure is
  constant.
- μ = μ_in·C + μ_out·(1 − C), with C the cell volume fraction (8×8 sub-samples).
- Advection off (Stokes), or Re = ρU₀R/μ_out ≤ 1e-3. March at D ≥ 10 until the step change is < 1e-12·U₀.
- Error e = max|u − u_ex|/U₀ at N = 32, 64, 128, for μ_in/μ_out = 0.1 and 10.
- Gates:
  - full stress: e(64)/e(128) ≥ 1.75 (order ≥ 0.8);
  - Laplacian: e(128) ≥ 5·e_full(128), and e(64)/e(128) ≤ 1.25 (not converging).

**G3** — Large dt.
- (i) Two-layer Couette at dt = 100 with full stress: error 0.0002 % ± 1e-6 relative, converged by 200
  steps.
- (ii) G2 at μ ratio 0.01 and 100, D = 1e4, 2000 steps: no growth (max|u| non-increasing after 50 steps).
  It converges to the same steady error as at D = 10 (relative 1e-8).
- (iii) Steady Stokes through a periodic sphere array (sdflow set-up) with a smooth 10× μ(x):
  - dt = 20 and dt = 100 give the same steady K to 1e-6;
  - report the number of steps to steady with full stress against laplacian. Target ≤ 2×; otherwise Q3.

**G4** — Parallel and device.
- T bitwise equal at np 1, 2, 4.
- Case A d/h 20, 100 steps: max|u_np − u_1|/U ≤ max(1e-12, 1.1 × the same metric with full stress off).
  If the off path is bitwise decomposition-invariant, the on path must be too.
- CUDA against host, case A d/h 10 to t* = 20: |ΔRe|/Re ≤ 1e-4 (today 6e-5).

**G5** — Physics (case A, t* = 20, 2L-window drift Re).
- d/h 40 ∈ **[20.6, 21.7]** (±2 % of 21.0–21.3). Pre-registered prediction: 21.4 ± 0.4.
- Converging: |Re₄₀ − Re₂₀| ≤ 0.6·|Re₂₀ − Re₁₀|, or |Re₄₀ − Re₂₀|/Re₄₀ ≤ 1 %.
- A3 d/h 10: bitwise equal to the off path (§8.2).
- Report:
  - the `enable_vof_momentum` result at d/h 10 and 20. If it is > 2 % from the velocity form at both, open
    the separate package (Q6);
  - the steady closure acceleration.

**G6** — Regression.
- state_hash: 14 cases byte-identical, host and CUDA.
- ctests: `variable_mu`, `mirror_symmetry`, `velocity_solver_variable_mu`, `collocated_stability_guard`,
  `device_sync_budget`, `no_env_knobs`, `iteration_order`, clang-format.
- The full battery with `-LE bench` (248) green.

**G7** — Performance.
- Bubble column, 786k cells, RTX 5080: ms/step with full stress on is ≤ 1.03 × off.
- Host, 8 threads: ≤ 1.03 × off.
- Case A d/h 40 GPU: ≤ 1.05 × off.

## 14. Register entry (draft, for `docs/decisions/flow.md` and the DECISIONS.md index)

```
### Variable viscosity uses the full deviatoric stress (staggered): explicit conservative MAC transpose term on uⁿ
- area: flow
- source: flow doc/full_viscous_stress.md; bubble_column_perf/validation/CASE_A_ABLATE.md
- decided: 2026-10-22 (design); status: proposed → settled at WO-5 G5; default per USER (Q1)
- quote: |
    The Laplacian form ∇·(μ∇u) imposes [μ∂ₙu]=0 instead of [μ(∇u+∇uᵀ)n]=0: an h-independent interface
    force. Hadamard–Rybczynski at κ=0.1: drag 3.5 vs 4.18 πμUR → +19.5 % rise velocity; case A measured
    +20.1 % vs Basilisk at d/h 40. Transpose term = divergence of stress elements (normal at cell centres,
    shear on edges) with the SAME β as the Laplacian bands; explicit on uⁿ (unconditionally stable
    predictor: λ=(m+⟨T⟩)/(m+⟨K⟩)∈(−1,1]); skipped exactly iff μ globally uniform.
- rejected: product form (∇μ)·(∇u)ᵀ; excess-viscosity form with global μ_ref; transpose on the Picard
  iterate u*; implicit normal-stress diagonal; coupled implicit 3-component solve; harmonic edge μ with
  full stress (v1); collocated now
- why: product form leaves an O(1) interface error at oblique kinked interfaces (direction-dependent μ
  averages × δ-like second differences); excess form needs the same reduction and breaks T≤K for
  compressive modes; u* is not solenoidal (λ→−1, perturbs the rotational term) and gains no rate;
  implicit/coupled variants change the validated operator without speeding the physically slow
  rotation-dominated modes (that needs a variable-coefficient VMG, deferred); harmonic makes the edge μ
  component-dependent (asymmetric T); collocated breaks the cell-Laplacian/face-transpose energy pairing.
```

## 15. Risks and open questions (each has a default, so work proceeds)

| # | question | needs | default (proceed with) |
|---|---|---|---|
| Q1 | Should "full" be the default for VoF and closures, given that it changes published variable-μ numbers? | **user preference** | Implement opt-in (`"laplacian"` default) through WO-6. **Recommend** flipping to AUTO = full (WO-7): the Laplacian form is a different PDE for smooth μ and an O(1) interface error at jumps. |
| Q2 | Public name and placement | **user preference** (NAMING.md) | `Solver.set_viscous_stress("full" \| "laplacian")`, public, because the stress form is a modelling choice. It stays public after a flip (reproduces old numbers). |
| Q3 | Steady-march slowdown for Arrhenius users at large dt (rotation-dominated high-μ modes) | **fact** (G3 iii) | Ship regardless: stability and the fixed point are unaffected. If > 2× steps, open "coupled variable-coefficient VMG" as the deferred package; do not build a coupled smoother without it. |
| Q4 | Harmonic + full stress | **user preference** | Refuse in v1. Option later: a symmetric 4-cell harmonic edge μ for both parts (exact for layered μ, symmetric T). |
| Q5 | Collocated full stress: when | **user preference** (scope) | After case A closes, as §9. Until then, collocated refuses explicit "full"; AUTO resolves to laplacian with a notice. |
| Q6 | The `enable_vof_momentum` gap after the fix | **fact** (G5) | A separate package only if > 2 % at d/h 20 and 40, with the translation a-priori test first. Not blocking. |
| Q7 | μ and velocity edge ghosts across ranks | **fact** (G0) | If missing, WO-0b fixes them first: it is a latent bug of the existing Laplacian too. |
| Q8 | Porous and moving solids with full stress | **user preference** (scope) | Refuse in v1. |
| Q9 | Any state_hash case affected by WO-7? | **fact** | Expected none: all VoF hash cases have constant μ. Verify when flipping. |
| Q10 | The swarm +11 % (TBFsolver, Loisy) | **fact** | Re-run one swarm point after WO-5 before attributing any remainder elsewhere. |

Other risks:
- **Composite stability (§6).** It is not proven in general. Covered by G3.
- **The BC-imposed-row predicate (§5.5).** The implementer must locate it in the domain-BC path. It is
  guarded by the wall-face gate.
- **Float operator storage opt-out.** Rigid-rotation cancellation then holds only to float precision. This
  is acceptable: it is not the default.

## Amendments (coordinator, 2026-10-22)

- **A1 — G2 at μ ratio 10.** The Laplacian sub-gate e_lap(128) ≥ 5·e_full(128) reads 1.108 / 0.239 = 4.63 at N = 128
  (full-stress order 0.85 over 32/64/128, which passes). At N = 256 on CUDA it reads 0.952 / 0.1285 = 7.42, with full
  order 0.90. The design's premise holds: the full form converges and the Laplacian does not. The threshold was too
  tight at N = 128 for ratio 10. **G2 at ratio 10 is judged at N = 256.** Ratio 0.1 passes at N = 128 (lap/full 17.4).
- **A2 — G3(i).** The quoted 0.0006 % / 0.0002 % Couette errors are harmonic-mean values, and v1 refuses harmonic
  with full stress (Q4). For layered μ(y) the transpose term vanishes identically. G3(i) therefore reads: on the
  arithmetic mean, full and Laplacian give the same Couette error to ≤ 1e-8 relative at dt 20 and dt 100. Measured:
  1.881699 % for both, with differences of 3e-13 and 2e-9.
- **Smoke (not a gate):** case A, d/h 10, host, full stress gives Re 21.3588. The Laplacian form gives 24.08;
  Basilisk 23.05 at d/h 10 and 21.275 at d/h 40.
