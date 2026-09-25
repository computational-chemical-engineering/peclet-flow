# Collocated variable density (rung V8), the balanced-force projection, and the guard against face-acceleration kicks

*Design note, 2026-09-25 (architect pass, amended the same day with the user's decisions). Branch
`v8-cellforce` of flow (base `1fdee46`).*

**What this note covers:**

- It replaces the V8 (WO-T) step of `SolverColocated`, i.e. whenever
  `Grid::collocated && (varRho_ || csfActive())`.
- It adds an optional **balanced-force projection** on both grids.
- It installs a suite-wide guard against the defect V8 reintroduced.

The collocated multiphase-with-solids phase is the NEXT package. Its plan is
[`collocated_multiphase_solids_plan.md`](collocated_multiphase_solids_plan.md).

**Evidence labels:**

- **[MODEL]** — the exact-solve step-operator models of Appendix A.
- **[CODE]** — measured in this worktree on host-openmp.

**User decisions of 2026-09-25, not open:**

- **U1.** The predictor form with the momentum-weighted centre-to-face map is the new collocated
  variable-ρ scheme. `set_body_force` is a mean pressure gradient in face form (Q1 = yes).
  Per-cell forces keep the cell-value rule across density jumps (Q2 = yes).
- **U2.** The balanced-force projection is an **option** (user: "optional"). Its default is an
  orchestrator decision (2026-09-25, logged; reversible in one commit): **ON on the collocated
  variable-ρ/CSF path (V8)**, **OFF on the staggered grid and on constant-ρ collocated**. The reason
  is the settled V8 requirement that constant-κ CSF stays annihilated (brief §5). With the option off,
  V8 would lose that (§4.6.5: 2.2e-4 at ratio 1 after 30 steps). Off the V8 path, OFF keeps
  byte-identity.
- **U3.** The option is available on the staggered grid as well.
- **U4.** There is one public setter for both grids.
- **U5.** The face-acceleration defect is guarded suite-wide (§7).
- **U6.** Collocated multiphase must reach immersed solids (next package; §4.9 lists this
  package's load-bearing choices).
- **U7.** The end goal is amr + solids + multiphase (§13).

## 0. The decision

1. **V8's two defects have one cause.** V8 adds the pressure gradient as a face acceleration after
   the viscous solve.
   - The projection removes exactly that term, so the velocity update is *non-incremental*:
     $P^n$ never reaches the velocity.
   - With κ = 0 this is Chorin's splitting error: the steady state depends on dt (cause B).
   - With κ > 0 the rotational term becomes an explicit pressure diffusion with multiplier
     $-4\kappa\Delta tS/(\rho h^2)$. That is −12 at (π,π,π) for μdt = 1, measured −12.0000
     (cause A).
2. **The pressure and every force go back inside the implicit predictor.**
   - Pressure, CSF and the uniform drive act on the cell as
     $\rho_c\cdot\tfrac12\sum_{\rm faces}o_f\,(F_f-wG_fP)/\rho_f$. This is the finite-volume face
     integral with the acceleration-continuous face pressure.
   - Per-cell volumetric forces act at the **cell value**, multiplied by the weight sum
     $W=\tfrac12(o_-+o_+)$.
3. **The projection's face field is momentum-weighted:** $(\rho_Lu_L+\rho_Ru_R)/(\rho_L+\rho_R)$.
   - Only with this map is the pressure force exactly $-M^{-1}C^T$ (C the constraint), with or
     without fractional openness.
   - That adjointness is the stability proof: [MODEL] spectral radius exactly 1 in ≈ 1 200 step
     operators, ratio up to 10⁶, dt up to 10⁴, κ = μ and κ = 0, including an immersed disk with
     fractional apertures.
   - The steady state is dt-independent. At uniform ρ this *is* the validated constant-ρ scheme
     (and amr's).
4. **The balanced-force projection is optional, with one setter on both grids. The default is ON on V8 and OFF elsewhere (U2):**
   `set_balanced_force_projection(enabled)`.
   - Mechanism: one extra Poisson solve per step with the projection's own operator. It moves the
     gradient part of the forces into P before the viscous operator acts.
   - ON makes static balances exact from step 1: hydrostatic columns and constant-κ drops, at every
     μ, dt and ratio, **including immersed solids on the staggered grid**. It removes WO-P's
     "μdt² residue".
   - OFF (the staggered and constant-ρ collocated default) leaves the scheme's stability and steady state exactly as they are, because the
     solve is state-independent. It leaves a decaying transient balance residue (numbers in §4.6).
   - On the staggered grid OFF is byte-identical to today.
5. **The rotational (Timmermans) update stays, unchanged.**
6. **The Basilisk face-acceleration form is removed, and guarded.** The guard has four parts:
   - a suite-wide register entry;
   - prohibitions inlined in three CLAUDE.md files;
   - a design-rationale comment at the predictor site;
   - a stability/dt-independence gate on *every* collocated density path in flow and amr.

---

## 1. Problem and scope

**Being built:**

- **(a)** The V8 step replacement: all-fluid, periodic or wall domain faces.
- **(b)** The balanced-force projection option on the staggered grid (constant and variable ρ,
  immersed solids, VoF CSF) and on V8.
- **(c)** The guard (§7).
- **(d)** The design choices that the next package (collocated multiphase + solids) and amr
  multiphase depend on (§4.9, §13).

**Out of scope, and unchanged byte for byte:**

- the staggered path with the option OFF;
- the constant-ρ collocated path without CSF (ghost / gauge-exact / plain / embed);
- CFD-DEM coupling, gamma calibration, dem.

**Refused with a named error:**

- V8 with an immersed solid, the ghost projection or harmonic ρ_f (today's
  `requireCollocatedFaceForceScope`);
- the option in the configurations of §4.6.4.

**Deferred:**

- collocated multiphase + solids (companion plan);
- momentum-consistent VoF on the collocated grid;
- inflow/outflow faces under the option (Q7).

## 2. Root cause (brief §4 A and B, confirmed and sharpened)

**Notation.**

| symbol | meaning |
|---|---|
| $M=\mathrm{diag}(\rho_c)$, $M_f=\mathrm{diag}(\rho_f)$ | cell and face density, with $\rho_f(i)=\tfrac12(\rho(i)+\rho(i-s))$ |
| $O$ | face openness |
| $\Pi$ | centre to face, $\tfrac12(u(i)+u(i-s))$ |
| $R=\Pi^T$ | face to cell, $\tfrac12(a(i)+a(i+s))$ |
| $G_f$ | face difference $P(i)-P(i-s)$ |
| $D=-G_f^T$ | divergence |
| $A=M/\Delta t-\mu L$ | implicit momentum operator |
| $\lvert K\rvert=-DM_f^{-1}G_f$ | pressure operator divided by ρ₀ |

**2.1 V8 is non-incremental in the velocity.**

- V8 forms $u_f^*=\Pi u^*+\Delta t\,(F_f-G_fP^n)/\rho_f$ with $u^*=A^{-1}(Mu^n/\Delta t)$.
- The pressure term lies exactly in the range the projection removes. So
  $\varphi=\varphi_0-(\Delta t/\rho_0)P^n$, where $\varphi_0$ is the potential of the
  pressure-free field.
- The cell update therefore contains **no $P^n$**, and
  $P^{n+1}=\tfrac{\rho_0}{\Delta t}\varphi_0-\kappa D(\Pi u^*+\Delta tF_f/\rho_f)-\kappa\Delta t\,|K|\,P^n$.

**Cause A** (rotational term, κ > 0):

- The multiplier is $-4\kappa\Delta tS/\rho$ on **every** pressure mode. The largest eigenvalue,
  at (π,π,π), gives the explicit-diffusion limit $12\kappa\Delta t/(\rho h^2)<1$.
- It matches the brief's measured −12.0000 / −1.2000 and 11.8478.
- [MODEL]: $8\mu\Delta t$ (2-D) and $12\mu\Delta t$ (3-D) at every ratio.
- *Correction to the brief:* in V8 no pressure mode reaches u. The checkerboard is merely the
  fastest mode.

**Cause B** (κ = 0, pure Chorin):

- The steady state is $-\mu Lu=(I-\Delta t\mu LM^{-1})(\dots)$, a factor $(1+\Delta t\mu\Lambda)$.
- The register already records the symptoms: Z&H −40 %; face-primary mode 8 −75 %
  (`docs/decisions/flow.md:866`).
- [MODEL]: V8 steady states at dt 0.3 / 3 / 30 differ by 3e-2 to 0.51.

**Placement.** At uniform ρ, V8's cell sees $R\Pi f=\tfrac14(1,2,1)f$.

**2.2 The obvious repairs fail [MODEL, Appendix B]:**

- **κ = 0.** Stable, but still Chorin.
- **Predictor form with the arithmetic Π (D2).** Unstable at high contrast, 1.018 per step at
  ratio 10⁴, for κ = 0 and κ = μ. The pair is not adjoint.
- **Naive $g-G_cP/\rho_c$.** 245× per step at ratio 1000.

## 3. Constraints and invariants

**Conventions.**

- The face $i$ of component c separates cells $i-s$ and $i$. Inner cells span $[G,e-G)$; face
  kernels span $[G,e-G]$.
- $o$ = `ox_/oy_/oz_` enters **multiplicatively** everywhere (§4.9 L1).
- $\rho_0$ = `rho_`; $w_c$ = `u_.w[c]`; internal units.

**Algorithmic invariants.**

- ABC: the face field is re-derived from the cell each step, and only the increment φ corrects it.
  Never Rhie–Chow.
- `projectPressureUpdate` is unchanged. κ = `rotWeight_·mu_`, or χ·μ_min under varProps.

**Placement rule (user, 2026-09-25).**

- Volumetric forces at the velocity location: the cell value on the collocated grid, the face mean
  on the staggered grid.
- Surface forces (pressure, viscous stress, CSF) as face integrals.
- `set_body_force` is a mean pressure gradient (U1).

**Precision and parallelism.**

- Double precision, no float casts.
- np = 1 exactly decomposition-independent; np > 1 within the existing derived tolerances.
- On-device only.

**Byte-identity.**

- Staggered with the option OFF: byte-identical.
- Constant-ρ collocated without CSF: byte-identical.
- All 12 `state_hash` cases (none is V8).

## 4. The design

### 4.1 Operators (per component c; a closed face has o = 0)

| symbol | definition | kernel |
|---|---|---|
| $\rho_f(j)$ | $\tfrac12(\rho(j)+\rho(j-s))$ | as today |
| $\Pi_\rho$ | $u_f(j)=\dfrac{\rho(j)u(j)+\rho(j-s)u(j-s)}{\rho(j)+\rho(j-s)}$ | **new** `centerToFaceMassWeighted` (`centerToFace` when `!effVarRho()`) |
| $\Phi_c(j)$ | $o(j)\,[f^{const}_c+\mathrm{CSF}_c(j)-w_c(P(j)-P(j-s))]/\rho_f(j)$ | `buildFaceAccelVar(haveFb=false, incr, scale=1)` + `addFaceAccelCsf(scale=1)`, with the openness predicate made multiplicative |
| $W_c(i)$ | $\tfrac12(o(i)+o(i+s))$, the **weight sum** of the cell reconstruction | **new** inline in the RHS kernel, computed from the same `o` reads as the reconstruction |
| $k_c(j)$ | $o(j)\,w_c\,(\rho_0/\rho_f(j))(\varphi(j)-\varphi(j-s))$ | written exactly as `projectCorrectVar` groups it |
| cell correction | $u_c(i)\mathrel{-}=\tfrac12(k(i)+k(i+s))$ | **new** `correctCellFaceAverageVar` |
| $\beta_c(j)$ | $o(j)\,[f^{const}_c+\tfrac12(f_c(j)+f_c(j-s))+\mathrm{CSF}_c(j)]$ | **new** `buildBalancedFaceForce` (option only; both grids, §4.6) |

- Rename `buildFaceAccelVar`/`addFaceAccelCsf`'s `dt` parameter to `scale`; the arithmetic is
  unchanged.
- When ρ is constant (CSF only): $\rho_f$ is the scalar `rho_` and $k$ is the plain
  `projectCorrect` difference.
- CSF(j) is the V4 face form `vof::csfFaceForce(σ, κ_f, ΔC, 1/w)`.

### 4.2 The V8 step

**Step head.** Unchanged: `vofStepPrecheck`, `updateProperties`, ghost fills, `updateVofCurvature`,
`rebuildStencils`, `old_ ← u`, and the `P_` ghost fill. Then:

- **(S0)** `requireCollocatedFaceForceScope`. It additionally throws if `!incremental_`.
- **(B)** *Only if the option is ON:* `balancedForceProjection()` (§4.6). It runs once per step,
  before the Picard loop.

**Per Picard iteration:**

- **(S5)** Build $\Phi_c$ into `faceAcc_[c]`. It reads the live `P_`.
- **(S6)** New `buildRhsColoVar(c)`, replacing `buildRhsColoFF`:

  $$b_c(i)=r_s\Big[\tfrac{\rho_c}{\Delta t}u^n_c-\rho_ca_K+\rho_ca_F+W_c(i)\,f_c(i)+\rho_c\tfrac12\big(\Phi_c(i)+\Phi_c(i+s)\big)\Big]+(\text{bc ? }b_{bc}:-\text{inh})$$

  Everything except the two force terms is as in today's `buildRhsColoFF`.
- **(S7)** Implicit momentum solve, unchanged.
- **(S8)** `project()` for V8:
  1. $u_f^*=\Pi_\rho u^*$ (it reads ρ's depth-1 ghosts; there is **no** face acceleration);
  2. `div_ = divergOpen(u_f^*)`;
  3. `projectSolve`;
  4. face correction by `projectCorrectVar`/`projectCorrect`, then fills and `buildOpenFaceField`
     as today;
  5. cell correction by `correctCellFaceAverageVar`;
  6. `maskVelocity`;
  7. `projectPressureUpdate` (unchanged; `div_` is now $D\Pi_\rho u^*$, Timmermans' quantity).

**Retired:** `applyFaceAcceleration`, `addFaceIncrement` and `applyCellFaceAverageCorrection`. They
are deleted, and the header text that described them is retired as in §7.5.

### 4.3 Stability: the pair is adjoint in the kinetic-energy inner product

Define $\Gamma P:=R\,O\,M_f^{-1}G_fP$ (pressure acceleration) and $C:=D\,O\,\Pi_\rho$ with
$\Pi_\rho=M_f^{-1}\Pi M$. Then

$$M\Gamma=M\Pi^TOM_f^{-1}G_f=-(D\,O\,M_f^{-1}\Pi M)^T=-C^T,$$

and this holds **for any openness $0\le o\le1$**, fractional apertures included. Consequences:

- **Schur complement.** $S=CA^{-1}C^T$ is SPD.
- **Pressure-correction bound (a proof).** Cauchy–Schwarz on the average gives
  $\Pi M\Pi^T\le M_f$, and $O^2\le O$. So $CM^{-1}C^T\le|L_p|/\rho_0$, and with $A\ge M/\Delta t$
  the φ part of the Uzawa operator is bounded by 1 at every dt and ratio.
- **Rotational part (measured).** It adds $\kappa\lambda_{\max}(C(-\mu L)^{-1}C^T)$, which is
  0.98–1.40 in the walled models.
- **[MODEL], radius exactly 1.000000000 in:**
  - 1 008 2-D operators (N ∈ {10, 11, 13}, periodic/walled, ratio 10–10⁶, dt 0.01–10⁴, κ = μ/0);
  - 80 3-D operators (N = 6, 7);
  - variable μ with κ = χμ_min (κ = μ_max is unstable at 27–104 per step, confirming the
    register's χ·μ_min);
  - 90 operators with an **immersed disk** (fractional apertures, masked solid cells, ratio 1–10⁶).
- **Neutral modes.** They are the known uniform-ρ checkerboard family. Interfaces add none.
- **The option does not change the step operator.** $P_b$ depends only on the forces, not on
  $(u,P)$. Stability is therefore identical ON and OFF.

### 4.4 dt-independence (ON and OFF identical)

At a fixed point, Prop. 1 of `collocated_invisible_subspace.md` gives φ = 0 and $u=u^*$, so

$$-\mu Lu=Wf+MR\,O\,\frac{f^{const}+\mathrm{CSF}-wG_fP}{\rho_f},\qquad D\,O\,\Pi_\rho u=0.$$

This contains no dt. The option only splits P.

[MODEL] (walled, generic volumetric plus non-gradient face forcing; direct fixed-point solve),
identical ON and OFF:

| ratio | $\lvert u(0.3)-u(3)\rvert$ | $\lvert u(3)-u(30)\rvert$ | $\lvert u\rvert$ |
|---|---|---|---|
| 1 | 1.1e-13 | 2.0e-14 | 1.56 |
| 1000 | 1.5e-10 | 4.6e-12 | 1.56 |

V8 gives 3e-2 to 0.51.

### 4.5 Placement and the weight-sum rule

- **Interior.** The volumetric force is the cell value. At uniform ρ the step is algebraically the
  validated constant-ρ collocated scheme (placement to round-off, TG order 2, exact drag).
- **The weight-sum rule.** A volumetric force enters multiplied by $W_c=$ the sum of the cell's
  face-to-cell reconstruction weights along its axis.
  - All-fluid with walls, $W_c=\tfrac12(o_-+o_+)$: the wall carries the Neumann pressure
    $\partial p/\partial n=f_n$ for the missing half.
  - Equilibrium requires this. For $f=\rho g$, the pressure part is
    $\rho_c\sum r_f\,g=W\rho_cg$, which balances exactly.
  - It holds with **fractional apertures**: [MODEL] a disk crossing a ratio-1000 column gives cell
    1.5e-13 with the option ON.
  - It holds with any other reconstruction (the ghost scheme's one-sided (1.5, −0.5), and amr's
    coarse–fine weights), because only the weight sum enters.
- **[MODEL] without the rule:** the ratio-1000 column leaves 7e-2 in the wall cells.

### 4.6 The balanced-force projection (option; both grids)

**4.6.1 What it computes.** With $\tilde c_f=\rho_0/\rho_f^{op}$:

- $\rho_f^{op}$ is the pressure operator's own face mean: arithmetic, or harmonic under the
  staggered WO-J knob. $\tilde c_f=1$ at constant ρ.
- Solve $D(O\tilde c\,wG\,X)=D(O\tilde c\,\beta)$ with the projection's **own** operator,
  constraint divergence and driver.
- Then $P\mathrel{+}=X-P_b$ and $P_b:=X$.

`P_` stays the total physical pressure. The predictor, whichever grid it is on, then sees
$\beta-wG_fP=\rho_f\,r-wG_fP_{dyn}$, where $r$ is the $O\tilde c$-solenoidal remainder, i.e. the part
of the forces that physically drives flow.

**Exactness.** If $\beta=wG_f\Psi$ on every face the operator couples, then $X=\Psi$ up to per-pocket
constants. The predictor RHS then vanishes identically, so $u^*=0$ and the state stays at rest from
step 1. That holds for any μ and dt, any ratio, any openness, and any IBM momentum operator (static
no-slip walls).

**4.6.2 Step placement.** After `vofStepPrecheck` (so a throw leaves the fields bitwise unchanged,
as today) and after properties, curvature and ghost fills; before the Picard loop:

- **(B1) Coefficients.** When ρ is variable, `projectBuildCoefficients()` runs here once, and a
  per-step flag makes `project()` skip its variable-ρ rebuild. ρ is frozen within the step.
- **(B2) Forces.** `buildBalancedFaceForce(c)` fills `faceAcc_[c]` with $\tilde c\beta$. It is one
  kernel for both grids:
  - the face mean $\tfrac12(f(j)+f(j-s))$ is written with the same expression as
    `Grid::atVelocity` on the staggered grid;
  - CSF comes from the **same** code the predictor uses, via `addCsfRhs` / `addCsfRhsCellInterp`
    refactored to take a destination and a `rowScaled` flag. The existing calls pass
    `(C[c].b, true)` and stay bitwise.
- **(B3) Divergence.** $b=$ `constraintDivergence(faceAcc_)`. This is the constraint-divergence
  routine `projectAssembleDivergence` already uses, extracted:
  - flux openness: `divergOpen(·, ox_, oy_, oz_)`;
  - for the next package's scheme it also carries that scheme's overlay (§4.9 L2).
- **(B4) Guard and solve.**
  - If $\max|b|=0$ (a global reduction), set X := 0.
  - Otherwise solve `solvePressureSystem(rhs1 = −b, pb1_)`, which is the dispatch extracted from
    `projectSolve`: the same driver and the same rtol, warm-started from `Pb_`.
- **(B5) Bookkeeping.** In one fused kernel: $P\mathrel{+}=X-P_b$; $P_b:=X$. Then the existing
  `fillGhosts(P_)` and `pressureBcGhost()`.
- **State.** `Pb_` is a registered g=2 cell field (`"p_balanced"`, so `redistribute` carries it),
  registered (zero) **as soon as the option is active** — by `set_balanced_force_projection(True)`,
  or on V8 by the setter that makes the path V8 (`set_density_mode`, `set_surface_tension`) — so a
  restart can restore it before the first step; `pb1_` is g=1 scratch. Neither exists while the
  option has never been active.
- **The split is restart state; an invalid split is RE-SPLIT (review fix, 2026-09-25).** B5's
  $P\mathrel{+}=X-P_b$ is right only while $P_b$ is the balanced part of the current $P$. Two
  events break that: a step with the option off (P moves, $P_b$ does not), and a restart that
  restored `"p"` (`set_field`) without `"p_balanced"` (or `"p_balanced"` without `"p"`). The
  continuing update would then add the balanced pressure a second time — measured on a walled
  ratio-1000 V8 column (8×8×24, μ 0.1, dt 1, g 0.1): 1.25e-2 after a restart and 1.27e-2 after
  OFF 400 steps → ON, against 7.8e-14 continued. The rule: whenever $P_b$ is not a valid split
  (`pbValid_`), the step sets $P_b:=X$ and leaves $P$ **unchanged**; restoring `"p"` and
  `"p_balanced"` together keeps the split. A fresh solver ($P=P_b=0$) is a valid split, so the
  first step still does $P\mathrel{+}=X$ and static balances stay exact from step 1. On a static
  force a re-split step IS the OFF step (the predictor sees $\beta-wG_fP$ either way), so an
  OFF → ON switch continues the OFF trajectory exactly — it never jumps, and it relaxes at the
  OFF rate. Gate: `tests/python/test_balanced_force_restart.py` (ctest `balanced_force_restart`),
  both grids — continued / restart / restart + `p_balanced` / OFF → ON against their twins
  (§9 G8(g)). A write through `diagnostics.field_view` is invisible to the rule: restore with
  `set_field`.

**4.6.3 API.** Names follow `docs/NAMING.md`: `set_` + a leading `enabled` bool, a readback without
`get_`, public tier.

- **Python:** `s.set_balanced_force_projection(enabled)` and `s.balanced_force_projection`, on
  **both** `Solver` and `SolverColocated` (one binding in the shared template).
- **Diagnostics:** `s.diagnostics.last_balanced_force_iterations`.
- **C++:** `setBalancedForceProjection(bool)`, `balancedForceProjection()`,
  `lastBalancedForceIterations()`.
- **Default: ON when `colocatedFaceForce()` (V8: collocated && (varRho || CSF)), OFF otherwise (U2).**
  An explicit `set_balanced_force_projection(False)` on V8 is honoured. It may be toggled at any
  time; toggling only re-splits P (§4.6.2) and never moves a converged steady state. Switching
  OFF keeps `Pb_` inside P; the first ON step after OFF steps re-splits.

**4.6.4 Scope.** The option raises a named error when first used (at the setter if the
configuration is already known, otherwise at the next step's head) with:

- `porous_` (the VANS operator carries ε and the drag relaxation; CFD-DEM is out of scope);
- the ghost projection (either grid; its overlay reaches β in the next package);
- `fluidOnlyMode_ == 2`;
- `vofBlockCsf()`;
- any inflow/outflow domain face (Q7);
- `!incremental_`;
- `!cutcellPressure_`;
- the collocated constant-ρ single-phase path. There the pressure force is `gpCenterGrad`, not Γ,
  and the option has nothing to balance: say so, do not silently no-op.

**Supported:** staggered constant and variable ρ (arithmetic or harmonic), VoF with V4 or
cell-interp CSF, **immersed solids**, periodic, wall and slip faces, and V8.

**4.6.5 What OFF leaves.** This is U2's price, stated. [MODEL] 2-D N = 12, max|u| after 30 steps
(collocated: cell / face):

| case | V8 new, OFF | staggered, OFF (today) | either grid, ON |
|---|---|---|---|
| κ-const drop, periodic, μdt = 0.05, ratio 1 | 2.2e-4 / 7.8e-5 | 1.4e-17 | ≤ 3e-17 |
| same, ratio 10 / 100 / 1000 | 5.1e-4 / 1.1e-4 / **1.6e-5** | 2.2e-5 / 3.2e-6 / **2.2e-7** | ≤ 1e-17 |
| same at μdt = 0.5, ratio 1 / 1000 | 1.0e-3 / 9.8e-5 | 2.2e-17 / 6.9e-6 | ≤ 9e-17 |
| walled drop, low Ca, μdt/ρ_min h² = 20, ratio 1 / 1000 | 3.0e-5 / 4.2e-6 | 5.9e-12 / 1.7e-6 | ≤ 2e-18 |
| walled column ratio 1000, 30 / 100 / 400 steps | cell 1.6e-3 / 4.6e-4 / 6.0e-5; face ≤ 8e-14; dP/dz rel 1.5e-2 / 7.8e-3 / 3.4e-4 | exact (1-D) | cell ≤ 2e-13, dP/dz ≤ 9e-13 **from step 1** |
| staggered column ratio 1000 + staircase solid, μ = 0.1 / 10 | — | 8.9e-5 / 1.1e-3 | ≤ 4.4e-15 |
| staggered drop cut by a solid, ratio 1, μ = 0.1 | — | 1.1e-4 | 1.6e-17 |
| aperture pair, column ratio 1000 + disk (next package) | 1.4e-2 (cell) | — | 1.5e-13 |

- OFF transients **decay**: the ratio-1 drop goes 2.2e-4 → 2.4e-17 in 300 steps, ratio 1000
  1.6e-5 → 2.1e-7.
- Staggered T2 [CODE] for comparison: ratio 10 / 100 / 1000 = 2.36e-5 / 2.29e-5 / 1.01e-5, and the
  ratio-1000 μ sweep gives 1.35e-10 / 4.1e-6 / 6.6e-6 (`v8gate_norot.log`).
- **Two consequences to say plainly:**
  1. With OFF, V8 no longer annihilates constant-κ CSF *per step*. V8 did, through the face
     acceleration, and the brief lists that as settled. The requirement is now met with the option
     ON, where the T2 exactness gates run (§9 G4).
  2. With OFF, the like-for-like model puts V8's static residue 1–2 decades above staggered's at
     ratio 1000, and T1's dP/dz < 1e-6 cannot hold at 100 steps (model 7.8e-3).

### 4.7 The option on the staggered grid

- **The predictor needs no change.** Every staggered RHS builder (`buildRhs`, `buildRhsForced`,
  `buildRhsVar`, `buildRhsVarMom`) reads the live `P_` through `gp = w(P(i) − P(i−s))` and adds
  $f^{const}+$ `atVelocity(fb)` and then CSF. So after (B5) it sees exactly
  $\beta-wG_fP_b-wG_fP_{dyn}$.
- **β is the predictor's own face force.** The face mean equals `atVelocity`, and the CSF comes from
  the same refactored kernels. `rs` (IBM row scaling) is not in β, because it multiplies the whole
  row.
- **Immersed solids are supported** (the user's porous-media target).
  - The staggered cut-cell projection is the aperture operator $D(O\tilde cwG)$ with the matching
    face correction, so the exactness of §4.6.1 holds on every cut face.
  - IBM rows see $r_s\cdot0$ at equilibrium, and pockets get their own constants.
  - [MODEL] (staircase solids) removes 1e-4 to 1e-3 residues, **even at ratio 1**: with walls or
    solids, $A$ does not commute with $G_f$ at any ratio.
- **Rotational update.** Unchanged. The option is state-independent, so staggered stability is
  unchanged.
- **WO-P's "μdt² residue".** At variable ρ or near walls, $A^{-1}G_f\Psi\notin\mathrm{range}(M_f^{-1}G_f)$,
  so a gradient force that enters the predictor lagged by one step leaves a non-gradient face
  current. The option removes the gradient before $A^{-1}$ acts. [MODEL] 2e-5 to 2e-7 → ≤ 1e-17.

### 4.8 Performance, data, communication

**OFF.** V8 costs the same as today's V8 minus the face-acceleration kernels. Staggered is
untouched.

**ON, per step:**

- the β kernels and one divergence;
- one global max-reduction;
- one extra Poisson solve: same operator, same MG hierarchy, same driver and rtol. It solves for
  the INCREMENT, $A\,\delta P_b = D(Oc\beta) - A P_b^{n-1}$, with the stop relative to the FULL
  right-hand side, $\lVert r\rVert \le \mathrm{rtol}\cdot\max(\lVert D(Oc\beta)\rVert,\,\text{tiny})$, and
  is skipped (zero iterations) when $P_b^{n-1}$ already meets that test (WO-P5, orchestrator
  decision 2026-09-25). The MAIN projection's Chebyshev bounds are estimated on the main
  right-hand side only (Q9) and their logic is unchanged. The pre-projection has bounds of its
  OWN (review fix 2026-09-25): estimated once on its own right-hand side and **kept across
  steps** — the per-step variable-ρ coefficient refresh (B1) invalidates only the main bounds —
  and re-estimated after a structural operator change (the driver, density, porous and ρ-face
  setters, a new geometry's pressure-MG rebuild — not a `redistribute`, which moves the same
  operator) or when a pre-projection Chebyshev solve **diverges** (it ran to
  the cap and its residual ended above its initial one; the solve then restarts from
  $P_b^{n-1}$ with fresh bounds). Before this, every non-skipped pre-projection under variable ρ
  estimated bounds (15 V-cycle iterations) and discarded them;
- two copies.
- **A failed pre-projection is visible and harmless** (review fix 2026-09-25). The driver's
  breakdown flag is captured right after the solve, and a non-finite X is checked (one
  reduction, agreed across ranks). On either, $P$ and $P_b$ are kept for the step (its predictor
  sees the forces as with the option off), `diagnostics.last_balanced_force_failed()` is set,
  `diagnostics.balanced_force_failures()` counts it, and the first one warns on stderr.

No new halo exchange of state.

**Measured [CODE] (32³, ratio 1000, Chebyshev rtol 1e-9, host-openmp, 4 threads, on a shared
host at load 75–120; `tests/study/balanced_force_cost.py` at 5 repeats, OFF and ON interleaved,
median; before = WO-P5 (`cbc1366`), after = the kept pre-projection bounds; two runs each, both
shown):**

| case | grid | step ON / OFF before | step ON / OFF after | main iterations OFF / ON (before = after) | pre-projection iterations before → after |
|---|---|---|---|---|---|
| moving drop (U = 0.02) | staggered | 1.76, 1.75 | 1.20, 1.19 | 9.9 / 9.8 | 10.2 → 11.1 |
| moving drop | collocated | 1.78, 1.68 | 1.22, 1.19 | 9.8 / 9.4 | 10.2 → 11.0 |
| drop at rest, height-function κ | staggered | 1.73, 1.86 | 1.12, 1.20 | 10.4 / 10.0 | 10.0 → 10.0 |
| drop at rest, height-function κ | collocated | 2.01, 1.81 | 1.22, 1.22 | 10.5 / 10.6 | 10.0 → 10.0 |
| drop at rest, constant κ (exact equilibrium) | staggered | 0.95, 1.00 | 1.02, 1.01 | 10.0 / 10.0 | 0.0 → 0.0 |
| drop at rest, constant κ | collocated | 0.97, 1.06 | 1.04, 1.10 | 11.2 / 11.0 | 0.0 → 0.0 |
| hydrostatic column, 20 steps (`balanced_force_mpi`) | both | — | — | — | 14 at step 1, then 0 |

(The WO-P4/P5 record of this table, 1.59–2.13 ON/OFF, was one run at 3 repeats; the "before"
columns above re-measure it under the same load as the "after" ones.)

- A static balance costs nothing after the first step. A changing force costs a full solve: the
  pre-projection needs about as many iterations as the main projection (one more on the moving
  drop, where the kept bounds are a few steps old), and the step costs 1.1–1.2× (G7's 1.5× is
  met; its second criterion, pre-projection iterations ≤ main, misses by about one iteration on
  the moving drop). Before the kept bounds it cost 1.7–2.0×, most of the difference being the two
  per-step spectral estimates (the pre-projection's, discarded, and the main solve's). A drop
  "at rest" with height-function curvature is not static: its parasitic currents move the
  interface, and the force changes every step.
- Main iterations are equal ON and OFF (the 2× budget of Q9 is not approached), and identical
  before and after the kept bounds (the main bound logic is untouched).
- The first implementation (WO-P1..P4: warm start from `Pb_`, stop relative to the initial
  residual) did NOT make a static interface cheap: the initial residual is the round-off of the
  previous solve and the stop asked it to fall by rtol again, i.e. a full solve at rtol 1e-9 and
  the iteration cap at 1e-14.

Memory: one g=2 and one g=1 field, only once the option is ON.

### 4.9 Load-bearing choices for the next package (collocated solids) and amr

These are made **now** so that neither the solids phase nor amr requires a rewrite.

- **L1. Openness is a multiplicative weight, never a predicate**, in $\Phi$, $W$, $k$, $\beta$ and
  $\Pi_\rho$'s divergence.
  - With $o\in\{0,1\}$ this is bitwise what a predicate gives. With fractional apertures it *is*
    the aperture-adjoint pair (§4.3 proof, [MODEL] disk).
  - $W$ must be computed from the **same** reads as the reconstruction, as its weight sum, never
    as a separate formula. A later change of reconstruction weights (ghost one-sided, coarse–fine)
    then stays balanced automatically.
- **L2. The option's RHS is "the projection's constraint operator applied to $\tilde c\beta$",
  obtained by calling the extracted `constraintDivergence`**, never `divergOpen` directly. The
  ghost scheme adds `gpDivergDelta`, and amr adds refluxing, without touching the option.
- **L3. $\Gamma$ is *defined* as $-M^{-1}C^T$ face by face.** The code keeps the
  reconstruct-from-faces structure (face accelerations first, then the weighted cell sum) rather
  than a cell stencil. Coarse–fine faces and closure faces then change weights, not structure.
- **L4. The three-way consistency is generalised.** The operator's $\rho_f$, $\beta$'s face mean,
  and $\Pi_\rho$'s mass weights use the **same** face-interpolation weights $w_k$ (½, ½ on a
  uniform face).
- **L5. The pointwise formulas live in ONE container-free header,** `src/collocated_varrho_point.hpp`
  (`KOKKOS_INLINE_FUNCTION`, no Views, namespace `peclet::flow::varrho`):
  - `faceMassWeighted(uL, uR, rL, rR)`;
  - `cellFromFaces(aLo, aHi, oLo, oHi)`;
  - `weightSum(oLo, oHi)`.

  They are written so that promotion to `peclet::core::scheme` is a file move (§13, Q10).

## 5. Balance of pressure, gravity and surface tension

**5.1 The identity.** Take the half-cell momentum balances with piecewise-constant ρ and f and
continuous acceleration. The face pressure is then

$$p_f=\frac{\rho_Rp_L+\rho_Lp_R}{\rho_L+\rho_R}+h\,\frac{\rho_Rf_L-\rho_Lf_R}{2(\rho_L+\rho_R)} .$$

The FV cell balance with this $p_f$ is algebraically V8's cell rule.

- V8 = cell-value force **plus** a force-aware reconstruction. At uniform ρ that folds
  $-\tfrac14\delta^2f$ into "the pressure", which is why the user rule rejects it (3.8 % misfit in
  the placement test).
- This design keeps the ρ-weighted first term and the cell-value $f_c$.
- The two agree whenever $f/\rho$ is continuous across the face. For **gravity** there is no
  conflict: $f_c/\rho_c=g=\overline{\rho_fg/\rho_f}$.

**5.2 Table.**

| force | enters | static balance, option OFF | option ON |
|---|---|---|---|
| pressure | Γ, FV face integral, acceleration-continuous | defines it | defines it |
| gravity (closure ρg) | cell value, $Wf$ | exact **fixed point**; transient from $P^0$ decays (§4.6.5) | exact from step 1, interior and walls, any ratio |
| CSF (V4 face form) | face form in Φ | exact fixed point; decaying transient | exact from step 1 (≤ 1e-16 model, every ratio, μ, dt) |
| `set_body_force` | face form in Φ (U1: mean pressure gradient) | exact fixed point | exact from step 1 (periodic column: cell 2.4e-13; as a per-cell field it would be 20.5) |
| per-cell forces not ∝ ρ, across density jumps | cell value (U1) | consistent; residual $O(\Delta(f/\rho))$ in interface cells | same (not a gradient) |
| drag β | target and diagonal at the cell | $u=U_0$ exact | same |

The O(1) interface-cell imbalance that motivated V8 is gone in both modes. It came from the
arithmetic $G_cP/\rho_c$, not from where the force sits.

**5.3 Rating.**

- Static and balanced at ratio 1000 is stable in both modes. It is exact with the option ON.
- With motion the collocated path is rated ratio ~10 (WO-V3: a translating drop throws the VoF
  CFL cap at ratio 100 and 1000, option ON and OFF; RT at ratio 3 and the drop at ratio 10 track
  staggered). Momentum transport is untouched by this package.

## 6. Rejected alternatives

| alternative | why rejected |
|---|---|
| κ = 0 on V8 | Cures A only; still Chorin (TG order −0.1, drag 2.48 [CODE]); contradicts "rotational update must be restored" |
| rotational term on $D\Pi u^*$ before $a_f$ | P drifts linearly (brief pre-check); B untouched |
| filtered rotational update | Damps A without removing its source; B untouched |
| completing Basilisk `centered.h` (lagged g around the viscous solve) | The face field keeps $\Delta t(I-\Pi R)a_f$, a Rhie–Chow-type term: the steady constraint depends on dt; non-incremental P |
| arithmetic Π with the ρ-weighted force (D2) | Non-adjoint: 1.0002 per step (ratio 1000), 1.018 per step (10⁴) |
| naive $g-G_cP/\rho_c$ | 245× per step at ratio 1000, plus O(1) imbalance |
| force-aware reconstruction for volumetric forces | Places ¼(1,2,1)f; violates the user rule |
| **pre-projection always on** (this note's first draft) | Superseded by U2: an option, per the user. Default ON only on V8, where the settled constant-κ balance needs it |
| **default OFF on V8 too** | Breaks the settled V8 requirement (constant-κ CSF annihilated): 2.2e-4 at ratio 1, and the column dP/dz 7.8e-3 at 100 steps [MODEL] |
| the option as a new staggered predictor | Unnecessary: the staggered predictor already reads the total P |
| lagged pre-projection (exact first step, then reuse) | First-order in time for moving interfaces, two code paths |
| option kernels born in `core` now | One consumer today; a core release for three one-line functions buys nothing. Promotion is a file move when amr consumes them (Q10) |

## 7. Register reconciliation, and the guard against face-acceleration kicks (U5)

**7.1 Reconciliation.**

- **Suite-wide "Basilisk face-acceleration form REJECTED"** (`docs/decisions/suite-wide.md:300`).
  WO-T violated it; this design removes it. The entry's positive rule (the transpose pair in
  predictor and correction) is kept and extended to variable ρ as $M\Gamma=-C^T$.
- **ABC, never Rhie–Chow.** Kept.
- **Rotational update and χ·μ_min.** Kept.
- **Wall-banded blend.** Not needed (all-fluid).
- **User rule 2026-09-25.** Kept, with Q1 and Q2 now settled by U1.

**7.2 Why one register entry was not enough, and what replaces it.**

- The entry lived under "flow is the method reference" in the suite-wide file. It named the form
  but not the **mechanism**, and it gave no **signature** a test could detect.
- WO-T ported Basilisk `centered.h` in good faith as the "balanced-force" answer, under a different
  name, and it passed every gate the rung defined. None of those gates exercised dt·μ/(ρh²) > 1/12
  or dt-independence.

The guard therefore has four parts: the register entry (7.3), inlined prohibitions (7.4), comments
and docs (7.5), and a gate (7.6).

**7.3 Suite-wide register entry.** Covers flow, amr and voro collocated paths. Text in §12, entry C.

**7.4 Inlined prohibitions.** Add this bullet to the "Settled decisions" lists of `flow/CLAUDE.md`,
`amr/CLAUDE.md` and the suite `CLAUDE.md` cross-cutting list (the amr copy says "collocated
Navier–Stokes"):

> - **Collocated pressure and forces go INSIDE the implicit momentum predictor — never a face
>   acceleration added after the viscous solve (the Basilisk `centered.h` "kick").** Added after
>   the implicit solve, the projection removes the lagged pressure exactly, so the velocity update
>   is non-incremental. That gives Chorin's dt-dependent steady state, and with the rotational
>   update an explicit pressure diffusion growing like −12κ·dt/(ρh²) per step (measured −12.0000;
>   it capped V8 at density ratio ~100). Variable density keeps balance through the mass-adjoint
>   pair (the ρ-weighted face-average pressure force + the momentum-weighted centre→face map) and
>   the optional balanced-force projection — not through face placement. The guard is the
>   stability/dt-independence gate on every collocated path, not a grep. Register: suite-wide
>   "Collocated forces stay in the implicit predictor"; design: flow `doc/collocated_varrho_forces.md`.

**7.5 Design-rationale comment and doc trail.**

**(a) Predictor site.** At `buildRhsColoVar`'s definition (and a one-line pointer at the V8 branch
of the RHS dispatch in `step()`), verbatim:

```
// WHY the pressure and every force are HERE, inside the implicit predictor, and never a face
// acceleration added after the viscous solve (the Basilisk centered.h "kick" that rung V8/WO-T
// used until 2026-09-25): added after A^{-1}, the lagged pressure -dt*G P^n/rho_f lies exactly in
// the range the projection removes, so P^n never reaches u (the step is non-incremental Chorin:
// steady state scaled by (1 + dt*mu*Lambda)), and the rotational term -kappa*div then acts on it
// as an explicit diffusion, P^{n+1} = -4*kappa*dt*S*P^n/rho (-12.0000 measured at (pi,pi,pi)).
// Balance at a density jump comes from the operators, not the placement: the pressure force is
// rho_c * avg_faces(G_f P / rho_f) (FV face integral, acceleration-continuous face pressure), and
// the constraint reads the momentum-weighted face velocity, so M*Gamma = -C^T. Gates:
// tests/python/test_collocated_stability_guard.py. Design: doc/collocated_varrho_forces.md.
```

**(b) `src/collocated_varrho.hpp` header.** Rewritten to describe this scheme. The retired design
survives as one paragraph:

> HISTORY — do not reintroduce. WO-T (2026-09-02) made every force and the lagged pressure a MAC
> face acceleration added after the viscous solve (Basilisk centered.h; Popinet JCP 2009 §3). It
> was balanced but non-incremental and unstable above μdt/(ρh²) = 1/12, and it was retired
> 2026-09-25. See doc/collocated_varrho_forces.md §2 and the suite-wide register.

**(c) `tests/kokkos/test_vof_collocated.cpp`.** The header and the T1/T1b/T2 narratives are
rewritten. The "THIS is what rates the collocated rung to density ratio ~100" paragraph is replaced
by the §4.6.5 statement.

**(d) `doc/collocated_invisible_subspace.md`.** Gets a new §10, "Variable density: the
face-acceleration detour (WO-T → retired 2026-09-25)": the mechanism of §2.1, the numbers, the
adjointness fix, and a pointer here. About 15 lines.

**(e) `doc/history/vof_workorders_v5.md`.** Banners at the WO-T sections (lines 331 and 847):
"SUPERSEDED 2026-09-25 — the face-acceleration form was retired; see
doc/collocated_varrho_forces.md. Do not re-implement." History is otherwise untouched.

**(f) `doc/variable_density_projection.md` §4** gets the collocated paragraph. **`flow/CLAUDE.md`
"Collocated solver"** gets the scope line.

**7.6 The gate that actually guards: `tests/python/test_collocated_stability_guard.py`** (ctest
`collocated_stability_guard`, plus an amr counterpart).

**Paths**, every collocated density path. A throw FAILS the path at any step, unless the path is
on the guard's explicit refusal whitelist (empty today). (Until the 2026-09-25 review a refusal at
the first step counted as a pass, so a path that threw at once passed silently.)

- constant ρ: `ghost` (PINNED, and its build must print no AUTO fallback notice; its G3 walls are
  an immersed solid, because the ghost v1 takes no domain-BC walls — the old "AUTO (ghost)" G3
  case with ±y domain walls silently ran gauge-exact), `gauge-exact`, `plain`, `embed`;
- V8 variable ρ, uniform ρ field and a frozen ratio-1000 slab, option OFF and ON;
- V8 constant-ρ CSF (`enable_vof`, σ > 0, frozen C), OFF and ON.

**Per path**, Stokes (advection off, so the step is affine and the perturbation evolves exactly by
the step operator):

- **G2, growth.** Seed P with ε(−1)^{x+y+z} plus a random field of amplitude ε. Run 100 steps at
  dt ∈ {0.1, 100} (μ = 1, h = 1, so μdt/ρ_min h² up to 100) beside the unseeded twin, periodic;
  once more WALLED (±y) at dt = 100 on the ratio-1000 slab (option OFF: the option does not touch
  the homogeneous step operator, ON and OFF measure the same growth), so the weight sum, the
  high-face openness and the Neumann ρ ghosts are in it. Measure the (π,π,π) FFT amplitude of P − P_twin and
  max|u − u_twin|. Pass iff at step 100 both are ≤ 2× their step-0 values and ≤ (1 + 1e-6)×
  their step-50 values.
- **G3, dt-independence.** Walled at ±y, TG forcing, dt ∈ {100, 10, 1}, march to
  max|Δu| ≤ 1e-10·max|u| (dt = 10 and 1 warm-started from the dt = 100 state; a V8 ON path's
  dt = 100 march warm-started from the OFF path's converged state — the same fixed point, and the
  first ON step re-splits, §4.6.2). Pass iff max|u(dt_i) − u(dt_j)| ≤ 1e-7·max|u|: the Chorin
  signature is 2.5e-2–0.5, five decades above. Pressure rtol 1e-12 throughout. (Trimmed at the
  2026-09-25 review from dt ∈ {0.1, 1, 10, 100} / 1e-12 / 1e-9 / rtol 1e-14, which took 915 s.)

**Why a gate and not a grep.**

- The defect is a *property of the discrete step*: where the lagged pressure enters relative to the
  implicit operator.
- It can come back under any name and any structure:
  - a `centered.h` port;
  - an "explicit pressure correction after the viscous solve";
  - a defect-correction stage that moves −G Pⁿ to a later explicit update;
  - an amr/voro port that "saves a solve" by applying forces after it.
- None of these share a token. A grep for `applyFaceAcceleration` or "face acceleration" misses
  them and flags legitimate code (the ABC face correction *is* a face increment).
- Every variant has an unconditional signature:
  - growth ≈ 12κdt/(ρh²) per step once μdt/(ρh²) > 1/12 (G2);
  - with κ = 0, the dt-dependent steady state (G3).
- G3 matters because the likely next mistake is "turn off the rotational term", which silences G2.
- **The gate must fail on today's V8.** WO-V0 records that it does.

## 8. Work orders (for `opus-implementer`)

The tracks are ordered so that **Track V does not depend on Track P**. WO-V1 is one atomic commit.
Record the `state_hash` hashes before the first WO; G6 re-checks them after every WO.

### Track V — the V8 redesign

**WO-V0 — instruments (no production change).**

- `tests/study/colocated_varrho_symbol.py` = Appendix A.
- Write `test_collocated_stability_guard.py` (§7.6). Run it on the current build and record in the
  commit which paths fail (expected: V8, G2 at dt ≥ 0.1 and G3) and that the constant-ρ paths pass.
  Not yet registered.
- T1 periodic: the offset moves to `setBodyForce(0, 0, rbar*g)`, with the closure
  `LinearMix("rho", {0, -g})`. Gate: T1 passes its current thresholds on current code, and dP/dz
  agrees with the pre-edit run to 1e-12 relative.

**WO-V1 — the V8 step, OFF-mode (atomic).**

- Kernels (`collocated_varrho.hpp` + the new `collocated_varrho_point.hpp`, L5):
  `centerToFaceMassWeighted`, `correctCellFaceAverageVar`, `buildRhsColoVar` with W computed as the
  weight sum (L1). Openness is multiplicative (L1). The `dt`→`scale` rename.
- The §7.5(a) comment.
- S5–S8 in `step()` / `projectAssembleDivergence` / `projectCorrectVelocities`, inside the existing
  `colocatedFaceForce()` branches only. Delete the three retired functions. The `!incremental_`
  throw.
- Rewrite the headers per §7.5(b)(c).
- Adjust tests to OFF-mode gates (G4-OFF). The exactness checks listed there as "ON" move to WO-P2.
- **Gates:** G1, G2/G3 (V8 paths now pass; others unchanged), G4-OFF, G6.

**WO-V2 — V8 MPI, OFF.** `test_vof_collocated_mpi`: np = 1, 2, 4 × `OMP_NUM_THREADS` 1/2/4/8,
`--bind-to none`. Gate G5.

**WO-V3 (optional measurement; decides the motion rating).** Translating drop at ratio 1000 and RT
at ratio 3 against staggered. Report only.

### Track P — the balanced-force projection option

**WO-P0 — pure refactors (byte-identical).**

- Extract `solvePressureSystem(rhs1, x1)` from `projectSolve`.
- Extract `constraintDivergence(fx, fy, fz, out)` from `projectAssembleDivergence`, covering the
  non-porous, non-ghost branches of both grids.
- Give `addCsfRhs` / `addCsfRhsCellInterp` a destination and a `rowScaled` flag.
- Gate: G6, and `test_vof_*` output identical line for line.

**WO-P1 — the option on the staggered grid.**

- State: `balancedForceProj_`, `Pb_` (registered, lazy), `pb1_`, the B1 flag,
  `lastBalancedForceIters_`.
- `buildBalancedFaceForce` and `balancedForceProjection()` (B1–B5), called from `step()` at §4.6.2's
  point **only when the option is ON**.
- The §4.6.4 refusals.
- The setter, readback and diagnostics bindings (both grids).
- **Gates:** G6 (OFF byte-identical); G8 (staggered ON).

**WO-P2 — the option on V8** (depends on WO-V1 and WO-P1).

- Hook B into the V8 step, and **make ON the V8 default (U2)**. An explicit `False` is honoured.
- Tests: G4-ON variants of T1/T1b/T2 (now the default on V8, so T1/T2 are back at their exactness
  thresholds), G4-OFF kept as explicit-OFF cases, and the guard test's V8 ON paths.
- Gates: G4-ON, G2/G3 on ON paths, G6.

**WO-P3 — MPI for the option (both grids).**

- np = 1, 2, 4. The iteration rule (exact at np = 1, ±1 above) extends to
  `last_balanced_force_iterations`.
- A redistribute case: `rebalance_by_weights`, then 5 steps ON, compared against no rebalance.
- Gate G5-ON.

**WO-P4 — performance.** The G7 measurement.

### Track G — guard and documents

**WO-G1 — register the guard.**

- Register `collocated_stability_guard` in flow's ctest list.
- **amr counterpart** (amr worktree `amr-stability-guard`): `tests/test_amr_stability_guard.cpp`
  or a Python ctest. Run §7.6's G2/G3 protocol on `Flow` at lmax = 0 and with one refined region
  (all-fluid and with a solid).
- The perturbation is seeded through whichever pressure/velocity accessor amr exposes. If none can
  seed P, seed u with the same pattern; the growth signature is the same.
- Gate: passes on amr `main` today (amr is already predictor-form), and the ctest count rises by
  one.

**WO-G2 — documents and register.**

- §12's entries: C → `docs/decisions/suite-wide.md`; A, B, D → `docs/decisions/flow.md`.
- §7.4's bullet in the three CLAUDE.md files.
- §7.5(d)(e)(f).
- `docs/DECISIONS.md` index lines.
- Rename `colocatedFaceForce()` → `colocatedVarRho()` (mechanical).
- `NAMING.md` needs no divergence row (the name is canonical).

## 9. Verification gates

**G1 — placement** (`test_cell_force_placement.py`, SKIP removed), collocated var-ρ:

- cell < 1e-6;
- TG order > 1.8;
- drag < 1e-8.

Other combinations unchanged.

**G2 / G3 — `collocated_stability_guard`** (§7.6), every path, OFF and ON. Pre-change V8 must fail.

**G4 — `test_vof_collocated`.** T1/T1b at the default Chebyshev rtol (1e-9); T2 at 1e-14.

- **OFF (default):**
  - T1 face < 1e-12 walled and < 1e-10 periodic.
  - T1 cell and dP/dz are reported, with decay required: the 400-step value < the 100-step value.
  - T1b: `b.cellU < a.cellU` (kept; model 4.6e-4 → 6.0e-5).
  - T2, all ratios and the ratio-1000 μ ∈ {0, 0.01, 0.1} sweep: no throw.
  - T2 face at 90 steps < face at 30 steps.
  - T2 face at 30 steps ≤ 10× the value measured at WO-V1 (frozen with a comment).
  - T3 and T4: tolerances unchanged. T5: plus the non-incremental throw.
- **ON:**
  - T1: face < 1e-12, cell < 1e-10, dP/dz < 1e-10 — **after 1 step and after 100 steps** (the
    periodic box at Chebyshev 1e-14, the walled ones at the default).
  - T1 periodic at the DEFAULT rtol (1e-9), 100 steps: face < 1e-9, measured 2.2e-10. The
    pre-projection solves once at step 1 to rtol·|b| relative to the full right-hand side, and
    |b| (the ratio-1000 hydrostatic force) is large; from step 2 the previous split meets that
    test and the solve is skipped, which freezes the step-1 error. The pre-projection tolerance
    stays EQUAL to the main solve's (DECIDED 2026-09-25) — the 1e-14 case is the exactness
    statement, this one guards against a regression at the shipped default.
  - T1b: cell ≤ 1e-10.
  - T2: ratio 1 face and cell < 1e-14; ratios 10–1000 < 1e-13.
  - Ratio-1000 μ sweep: < 1e-13 and < the staggered face.
- **Measure-then-freeze.**
  - A miss by less than 10× freezes the gate at 10× the measured value, with a comment. For T2
    ratio 1 first try a pre-projection rtol of 0.1× the main rtol (a named constant).
  - A miss by more than 10×: re-run at Chebyshev 1e-14 first. Only a miss there is conceptual;
    stop and report it.

**G5 — MPI.** `test_vof_collocated_mpi`:

- np = 1 exact;
- np > 1: u/uf 1e-11, P 1e-9, C 1e-11;
- iterations ±1 (both counts);
- thread counts 1/2/4/8;
- `wo_vof_mpi_parity_gates.md` §8.

**G6 — byte-identity.**

- `state_hash.py`: all 12 hashes unchanged.
- `sdflow_regression.py` baselines unchanged.
- `test_vardensity_projection` output identical.
- The battery (`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`, `--bind-to none`, np8 last): 167/167 →
  169/169 with the two new flow tests. amr +1.

**G7 — performance.**

- ON step time ≤ 1.5× OFF on a moving ratio-1000 32³ drop (both grids).
- Mean pre-projection iterations ≤ mean main iterations.
- On a miss: report; the orchestrator decides (Q3).

**G8 — staggered with the option ON** (new cases in `test_vof_surface_tension` /
`test_vardensity_projection`, or a new `test_balanced_force.cpp`):

- **(a) OFF identity.** OFF equals the pre-change code: every existing staggered line is
  identical.
- **(b) T2 staggered drop.** Ratio 1 / 10 / 100 / 1000, 30 steps: face < 1e-13 (today 2.4e-17 /
  2.36e-5 / 2.29e-5 / 1.01e-5). Ratio-1000 μ ∈ {0, 0.01, 0.1}: < 1e-13 (today 1.35e-10 / 4.1e-6 /
  6.6e-6).
- **(c) Hydrostatic column.** Walled, ratio 1000, μ ∈ {0, 1e-3, 1e-2, 1e-1}: dP/dz < 1e-10
  **after one step** (OFF: converges over steps), face < 1e-12.
- **(d) Low-Ca walled drop.**
  - Setup: 32³, walls on all faces, R = 8, constant κ = 0.25, σ = 0.01, μ = 10,
    dt = 0.5·`capillary_dt()`, so μdt/(ρ_min h²) = 20 (ratio 1) or ≈ 450 (ratio 1000).
  - ON: face < 1e-13. OFF: reported (model 5.9e-12 / 1.7e-6).
- **(e) Immersed solid.** Both cases with ON at < 1e-12; OFF reported (model 1e-4 to 1e-3):
  - **(i)** A ratio-1000 walled column with a solid sphere (radius 5) crossing the interface
    (`set_solid`), μ = 0.1: face < 1e-12 and dP/dz at fluid cells < 1e-10 after 30 steps.
  - **(ii)** A constant-κ sessile cap: the drop of (d) cut by an immersed plane 0.5 R below its
    centre, ratio 1 and 1000, μ = 0.1.
- **(f) Refusals.** porous, ghost, block CSF, inflow/outflow and non-incremental each raise a
  named error.
- **(g) The split survives a restart and a switch** (`balanced_force_restart`, both grids, the
  column of §4.6.2): continued, restart without and with `p_balanced`, OFF 400 → ON after 1 and 6
  steps, each within 1e-12 of its twin (the continued run; for OFF → ON the OFF run continued);
  the restarts also < 1e-12 absolute. Measured: V8 restart 6.3e-14 (twin diff 6.7e-14), with
  `p_balanced` bit-identical, OFF → ON 0 / 8.4e-14; staggered 1.8e-14 (4.5e-14), bit-identical,
  0 / 3.0e-14. Before the fix: 1.25e-2 and 1.27e-2 on V8.

## 10. Byte-identity outside V8

- **The option.** Its code runs only under `balancedForceProj_`, which defaults to false. With it
  off, the staggered step and every non-V8 collocated step are unchanged. WO-P0's refactors are
  proved by G6 before anything else moves.
- **The V8 redesign.** It lives inside the existing `colocatedFaceForce()` branches, i.e.
  `Grid::collocated && (varRho_ || csfActive())`. Staggered folds that to false; constant-ρ
  collocated without CSF evaluates it false.
- **State.** `Pb_`/`pb1_` exist only once the option has been active (V8, or an explicit ON), so
  registry, redistribution and memory are unchanged otherwise. The split-validity flags
  (`pbValid_`, set by `set_field("p")`/`("p_balanced")` and by an OFF step) are plain bools that no
  OFF computation reads.
- **`state_hash`.** No case is V8 (`vof_droplet` is staggered), and none enables the option.

## 11. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1, Q2 | — | settled (U1) | — |
| Q3 | Cost of the option | fact (WO-P4; re-measured with the kept pre-projection bounds) | Accept up to 1.5×; beyond that the orchestrator asks the user about the lagged variant. MEASURED after the kept bounds: 1.1–1.2× on a changing force (before: 1.7–2.0×), 1.0–1.1× static (§4.8) |
| Q4 | Moving-interface effect of Π_ρ's O(h) interface-local weighting | fact (WO-V3) | Accept (stability requires it); report |
| Q5 | Motion rating | fact (WO-V3) | MEASURED: a translating drop throws the VoF CFL cap at ratio 100 and 1000 (ON and OFF); ratio 10 tracks staggered. Collocated with motion is rated ratio ~10 until the next package re-measures |
| Q6 | 3-D rotational margin (κλ_max < 2 is measured, not proved) | fact | G2 covers dt ≤ 100. If G2 fails with κ = μ but passes with κ = 0, stop and report; never ship a κ change without the architect |
| Q7 | Option with inflow/outflow faces | preference (scope) + fact | Refused in v1. It needs the operator-openness vs flux-openness rule at inflow and the WO-R2 outflow planes. Next package if the porous-media drainage target needs it |
| Q8 | Staggered VoF with the default OFF keeps WO-P's μdt² residue | preference | Resolved for V8 by U2's default ON. On staggered, the docs recommend enabling the option for static and low-Ca work (the porous-media drainage target) |
| Q9 | Chebyshev bounds when ON | fact | RESOLVED (WO-P5 + review fix): the main bounds are estimated on the MAIN right-hand side only; the pre-projection keeps bounds of its OWN across steps, re-estimated on a structural operator change or a diverging solve, never shared. Main iterations ON = OFF (§4.8) |
| Q10 | Pointwise formulas (L5) and the CSF pair → `core::scheme` / `core::vof`: now or when amr consumes them | preference (process) | When amr's multiphase package starts. Its WO-A0 moves them verbatim, with flow byte-identity as the gate |
| Q11 | Two collocated families in amr (ghost single-phase, aperture-adjoint multiphase+solids) | preference | See the companion plan §3: accept for S1; revisit at S2 |

## 12. Draft register entries

**A — flow** (`docs/decisions/flow.md`):

```
### Collocated variable density (V8): the mass-adjoint ABC pair; the face-acceleration form is retired
- area: flow
- source: flow doc/collocated_varrho_forces.md; flow <WO-V1 commit>
- decided: 2026-09-25
- status: settled
- quote: |
    V8 (WO-T) added the pressure gradient and every force as a MAC face acceleration AFTER the
    implicit viscous solve (Basilisk centered.h). The velocity update was therefore non-incremental:
    P^n cancelled exactly in the projection, the steady state was scaled by (1 + dt mu Lambda) (TG
    order -0.1, drag 2.48), and the rotational term became an explicit pressure diffusion,
    -12 kappa dt/(rho h^2) at (pi,pi,pi), measured -12.0000. The step now puts pressure and forces
    inside the implicit predictor. Pressure, CSF and set_body_force act on the cell as
    rho_c * avg_faces[o (F_f - w G_f P)/rho_f]; per-cell forces act at the cell value times the
    reconstruction weight sum W. The projection reads the momentum-weighted face velocity
    (rho_L u_L + rho_R u_R)/(rho_L + rho_R), so M*Gamma = -(D O Pi_rho)^T for any openness. Model:
    spectral radius exactly 1 in ~1200 step operators (ratio to 1e6, dt to 1e4, kappa = mu and 0,
    immersed disk); steady state dt-independent to 1e-10 (model) and <= 1e-12 relative in the
    code (guard G3, every V8 path). The rotational update is unchanged.
- rejected: face acceleration after the viscous solve (non-incremental; unstable with the rotational
    term); kappa = 0 (Chorin); completing centered.h with a lagged g (a Rhie-Chow-type dt-dependent
    constraint); arithmetic centre->face with the rho-weighted force (1.018/step at ratio 1e4); the
    naive g - G_c P/rho_c (245x/step); force-aware reconstruction for volumetric forces (places
    1/4(1,2,1) f)
- why: stability at every dt and ratio, a dt-independent steady state and cell-value placement
    need the incremental predictor with an adjoint pair
- conflict: conforms to suite-wide "Collocated forces stay in the implicit predictor" (entry C) and
    "ABC never Rhie-Chow"; keeps "rotational update must be restored" and "chi*mu_min under varProps"
```

**B — flow:**

```
### The balanced-force projection is an option on both grids, default ON on collocated variable-rho (V8), OFF elsewhere
- area: flow
- source: flow doc/collocated_varrho_forces.md §4.6-4.7; USER DECISION 2026-09-25
- decided: 2026-09-25
- status: settled
- quote: |
    set_balanced_force_projection(enabled) (Solver and SolverColocated; default True on the
    collocated variable-rho/CSF path, False elsewhere) solves, once
    per step and with the projection's own operator, constraint divergence and driver,
    D(O c w G X) = D(O c beta) (beta = the predictor's own face force: f_const + face mean of the
    cell forces + CSF; c = rho0/rho_f^op), then P += X - P_b, P_b = X. X is solved as the
    INCREMENT on P_b with the stop relative to the full right-hand side, and skipped when P_b
    already meets it: a static interface costs no iterations, a changing force about one extra
    pressure solve (step 1.6-2.1x at 32^3). Gradient forces (hydrostatics, constant-kappa CSF)
    are then balanced exactly from step 1 at every mu, dt, density ratio and openness. That
    includes immersed solids on the staggered grid, where it removes WO-P's mu*dt^2 residue
    (code: staggered drop 2.0e-5 / 1.0e-5 at ratio 10 / 1000 -> <= 2e-17; sphere across a
    ratio-1000 column 9.3e-4 -> 8.7e-14; sessile cap 1.0e-4 -> 2.8e-17). It is state-independent,
    so stability and the steady state are identical ON and OFF; OFF leaves a decaying transient
    (code, 3-D: V8 constant-kappa drop face 1.0e-4 at ratio 1 and 6.2e-5 at ratio 1000 after
    30 steps, decaying; ON <= 2e-17). Staggered OFF is byte-identical.
- rejected: always-on (the user asked for an option); default off on V8 (loses the settled
    constant-kappa CSF annihilation); a new staggered predictor (the existing one reads the
    total P); lagged P_b (first-order for moving interfaces)
- why: exact static balance is a physics choice with a cost (one extra solve), so the user makes it;
    one setter, one meaning on both grids
```

**C — suite-wide** (`docs/decisions/suite-wide.md`; supersedes the scope, not the content, of
line 300's clause):

```
### Collocated pressure and forces stay in the implicit predictor — never a face acceleration after the viscous solve
- area: suite-wide (flow, amr, voro collocated paths)
- source: flow doc/collocated_varrho_forces.md §2, §7; USER DIRECTIVE 2026-09-25
- decided: 2026-09-25
- status: settled
- quote: |
    On a collocated grid the lagged pressure gradient and every force enter the momentum equation
    INSIDE the implicit (viscous/drag) predictor. They are never added as a face acceleration
    after the implicit solve (Basilisk centered.h's uf += dt*a, and any equivalent "kick",
    whatever it is called). Mechanism: after A^{-1}, -dt G P^n/rho_f lies exactly in the range
    the projection removes, so P^n never reaches u. With kappa = 0 the scheme is non-incremental
    Chorin: the steady state is scaled by (1 + dt mu Lambda) (V8: TG error 9.2/9.8/10.6 at
    N = 16/32/64, order -0.1; drag 2.48). With the rotational update the pressure obeys
    P^{n+1} = -4 kappa dt S P^n/(rho h^2): -12 at (pi,pi,pi), measured -12.0000 at mu dt = 1 and
    -1.2000 at 0.1, unstable once mu dt/(rho h^2) > 1/12. The same instability rated flow's
    collocated VoF to density ratio ~100 (T2 "~4x per step"). The WO-T port reintroduced this
    in 2026-09 despite the earlier clause, which named the form but not the mechanism. The guard
    is therefore a GATE, not a grep: every collocated density path runs the seeded-checkerboard
    growth test (dt 0.1-100) and the dt-independence test (flow
    tests/python/test_collocated_stability_guard.py; amr test_amr_stability_guard). Variable
    density keeps balance through the mass-adjoint pair and the optional balanced-force
    projection.
- rejected: the Basilisk face-acceleration ("centered.h") form on any collocated path; a text
    prohibition alone (already existed and was bypassed)
- why: the defect is structural and silent at small mu*dt, so only a signature test at large
    mu*dt catches it
- conflict: sharpens suite-wide "flow is the reference ... Basilisk face-acceleration form
    REJECTED" (2026-09-03)
```

**D — flow** (Q1, now settled by U1):

```
### On the collocated variable-density path set_body_force is a mean pressure gradient (face form)
- area: flow
- source: flow doc/collocated_varrho_forces.md §5; USER DECISION 2026-09-25
- decided: 2026-09-25
- status: settled
- quote: |
    A uniform per-volume drive in a two-phase fluid stands in for a mean pressure gradient, so it
    is a surface force and enters as rho_c * avg_faces(o f_const / rho_f), balanced exactly by a
    linear pressure. Per-cell force fields stay volumetric (cell value), including across density
    jumps (residual O(delta(f/rho)) in interface cells, accepted). On the staggered grid and at
    uniform density the two readings coincide bit for bit.
- rejected: f_const at the cell value (periodic hydrostatic box: cell 20.5 after 30 steps at
    ratio 1000, against 2.4e-13); the force-aware reconstruction for per-cell forces (places
    1/4(1,2,1) f)
- why: exact balance of the only uniform drives two-phase users apply; matches the staggered semantics
```

## 13. Transfer to amr (the end goal: amr + solids + multiphase)

1. **At uniform ρ this scheme *is* amr's current step.** `amr/flow.hpp` and `flow_oracle.hpp` are
   incremental-rotational, with −∇pⁿ and f inside the implicit momentum and the ABC face average.
   Adopting the design changes nothing in amr's single-phase numerics. Variable density is a change
   of face weights (ρ_c/ρ_f in Γ, ρ-weights in Π), not of structure.
2. **Coarse–fine faces** (L3, L4). With amr's C/F face interpolation weights $w_k$ (`cf_scheme.hpp`,
   standard or quadratic):
   - the constraint's face velocity is $\sum w_k\rho_ku_k/\sum w_k\rho_k$;
   - the operator's face density is $\sum w_k\rho_k$ (the same weights);
   - β's face mean uses the same $w_k$;
   - each cell k receives the transpose share of the face's pressure acceleration.

   Then $M\Gamma=-C^T$ holds across the seam, and the §4.3 proof covers it.
   - **Fact to establish first** (amr multiphase WO-A1): is amr's *uniform-ρ* pair already
     transpose-exact at C/F faces? If not, the seam is the one place the proof does not reach, and
     WO-G1's refined-region guard is what catches it.
3. **The balanced-force projection** is amr's own constraint operator (the flux-matched, refluxed
   divergence) applied to $\tilde c\beta$ (L2), solved by amr's driver, with the same bookkeeping.
   amr projects once per step, and the option also runs once per step, so there is no divergence.
4. **Ghost projection.** amr's single-phase production scheme stays. Multiphase with solids takes the
   aperture-adjoint pair (companion plan §3), i.e. amr's aperture-projection family with these
   weights. Q11 flags the two-family consequence.
5. **Core** (Q10). The pointwise formulas of L5 go to `peclet::core::scheme`, next to
   `ghost_closure.hpp`, and the CSF pair (`csfFaceForce`, `csfFaceCurvature`, today in flow's
   `src/vof/surface_tension.hpp`) goes to `peclet::core::vof`, following the precedent of
   plic / curvature / cutcell / wetting. Nothing else moves, because the drivers are
   container-bound. This happens in amr multiphase's first WO, with a core minor release before
   any consumer pushes.
6. **Divergences flagged for amr's package:**
   - **(i) Pressure driver.** amr's settled "pressure is MG-PCG". Flow found PCG caps above ratio
     ~10³ (arithmetic coefficient coarsening), so amr multiphase at high ratio needs Chebyshev or
     coefficient-aware coarsening. That is an amr decision, not this package's.
   - **(ii) The guard.** amr's copy of §7.6 exists from WO-G1, before any amr multiphase code.
   - **(iii) Nothing else.** Nothing in this package's code shape (L1–L5) diverges from what amr
     needs.

---

## Appendix A — the step-operator model

A dense, exact-solve, 2-D, h = 1 model of the step, with ρ₀ = 1. Run it with
`OMP_NUM_THREADS=8 python colocated_varrho_symbol.py`. Expected output:

- V8 at dt 0.2: 1.6 = 8μdt;
- D2 at ratio 10⁴: 1.0136 and 1.0145 at dt 3 and 10;
- D3: 1.00000 everywhere.

```python
import numpy as np, itertools
def model(N, rho, mu, walls=False):
    n, I = N * N, np.eye(N * N)
    def sh(dx, dy):
        S = np.zeros((n, n))
        for x, y in itertools.product(range(N), repeat=2):
            S[x + N * y, (x + dx) % N + N * ((y + dy) % N)] = 1
        return S
    Sm, Sp = [sh(-1, 0), sh(0, -1)], [sh(1, 0), sh(0, 1)]
    o = [np.ones(n), np.ones(n)]
    if walls: o[1][:N] = 0.0                                   # closed y-faces at y = 0
    L = np.zeros((n, n))                                        # no-slip reflection at closed faces
    for a in range(2):
        for i in range(n):
            for S, op in ((Sm[a], o[a][i]), (Sp[a], (Sp[a] @ o[a])[i])):
                j = int(np.argmax(S[i]))
                if op: L[i, j] += mu; L[i, i] -= mu
                else: L[i, i] -= 2 * mu
    rf = [0.5 * (rho + Sm[a] @ rho) for a in range(2)]
    G = [I - Sm[a] for a in range(2)]; D = [Sp[a] - I for a in range(2)]
    Pi = [0.5 * (I + Sm[a]) for a in range(2)]; R = [0.5 * (I + Sp[a]) for a in range(2)]
    Lpi = np.linalg.pinv(sum(D[a] @ np.diag(o[a] / rf[a]) @ G[a] for a in range(2)), rcond=1e-12)
    return locals()
def step_matrix(m, dt, kap, scheme):                            # "V8" | "D2" | "D3" (chosen)
    n, I, rho, o, rf, G, D, Pi, R, L, Lpi = (m[k] for k in "n I rho o rf G D Pi R L Lpi".split())
    Ai, Z, T = np.linalg.inv(np.diag(rho / dt) - L), np.zeros((n, n)), np.zeros((3 * n, 3 * n))
    Fp = [np.diag(o[a] / rf[a]) @ G[a] for a in range(2)]      # face pressure accel per unit P
    us, uf = [], []
    for a in range(2):
        blk = [Z, Z, Z if scheme == "V8" else -Ai @ np.diag(rho) @ R[a] @ Fp[a]]
        blk[a] = Ai @ np.diag(rho / dt); us.append(blk)
        C2F = np.diag(1 / rf[a]) @ Pi[a] @ np.diag(rho) if scheme == "D3" else Pi[a]
        f = [np.diag(o[a]) @ C2F @ b for b in blk]
        if scheme == "V8": f[2] = f[2] - dt * Fp[a]
        uf.append(f)
    div = [D[0] @ uf[0][k] + D[1] @ uf[1][k] for k in range(3)]
    phi = [Lpi @ d for d in div]
    for a in range(2):
        for k in range(3):
            acc = -Fp[a] @ phi[k] + (-dt * Fp[a] if (scheme == "V8" and k == 2) else 0)
            T[a*n:(a+1)*n, k*n:(k+1)*n] = (us[a][k] + R[a] @ acc if scheme == "V8"
                                           else us[a][k] - R[a] @ Fp[a] @ phi[k])
    for k in range(3):
        T[2*n:, k*n:(k+1)*n] = phi[k] / dt - kap * div[k] + (I if k == 2 else 0)
    return T
radius = lambda T: np.max(np.abs(np.linalg.eigvals(T)))
```

**How the other numbers are produced:**

- **Forced runs** (§4.4–§5) add the affine terms to the predictor RHS:
  - $Wf_c$ and $\rho_cR(\Phi)$;
  - when the option is ON, $X=L_p^{+}D(o\tilde c\beta)$ and $Q\mathrel{+}=X-P_b$.
- **The staggered model** (§4.6.5, G8) is the MAC analogue: face unknowns, face-lattice Laplacian,
  staircase solids with pinned closed faces. Same option and projection.
- **The aperture runs** use fractional $o$ from a linear SDF along each face segment, pinned rows
  and masked solid-centred cells.

The scripts are in the session scratchpad: `stag.py`, `aperture.py`, `aperture_hydro.py`,
`amend_runs.py`, `col_perr.py`. WO-V0 copies the model into `tests/study/`.

## Appendix B — evidence [MODEL unless marked]

**B.1 Spectral radius, 2-D N = 12, μ = 1.** Columns are dt = 0.01, 0.1, 1, 10, 100, 10⁴. D3 is
1.000000 at every dt on all six periodic and walled configurations.

| config | V8 κ=μ | V8 κ=0 | D2 κ=μ | D3 κ=μ |
|---|---|---|---|---|
| uniform | 1, 1, **8**, 80, 800, 8e4 | 1 | 1 | 1 |
| slab ratio 1000 | 1, 1, **7.73**, 77, 773, 7.7e4 | 1 | 1 | 1 |
| disk ratio 1000, walled | 1, 1, **7.71**, 77, 771, 7.7e4 | 1 | 1 | 1 |

3-D (N = 6): V8 κ = μ gives 3.6 at dt 0.3 (= 12μdt); D3 gives 1.

**B.2 D2 vs D3, random shapes.** Worst radius:

| ratio | D2 κ=0 | D2 κ=μ | D3 κ=0 | D3 κ=μ | aperture pair + disk |
|---|---|---|---|---|---|
| 10 / 100 | 1 | 1 | 1 | 1 | — |
| 10³ | 1.0002 | 1.00008 | 1 | 1 | 1 |
| 10⁴ | 1.0149 | 1.0183 | 1 | 1 | — |
| 10⁶ | — | — | 1 | 1 | 1 |

**B.3 Balance with the option.** See §4.6.5.

Additional numbers:

- The periodic hydrostatic box with the offset supplied as a per-cell field gives 20.5 (μ = 0) and
  0.23 (μ = 0.1); in face form it gives 2.4e-13 and 1.7e-14.
- Removing the wall rule leaves 7.0e-2 in the column's wall cells.

**B.4 dt-dependence of the forced steady state.** Walled; ON and OFF identical.

| ratio | D3 κ=μ, $\lvert u(0.3)-u(3)\rvert$ / $\lvert u(3)-u(30)\rvert$ | V8 κ=0 |
|---|---|---|
| 1 | 1.1e-13 / 2.0e-14 | 5.4e-2 / 5.1e-1 |
| 1000 | 1.5e-10 / 4.6e-12 | 2.9e-2 / 9.3e-2 |

**B.5 [CODE].**

- V8 T2 is stable only with the rotational term off (face 4.3e-18 and 5.9e-18).
- Staggered T2 face: ratio 1 / 10 / 100 / 1000 = 2.43e-17 / 2.36e-5 / 2.29e-5 / 1.01e-5; the
  ratio-1000 μ sweep gives 1.35e-10 / 4.08e-6 / 6.59e-6.
- V8 T1 dP/dz today: 2.3e-8.
- T1b CONTROL cell: 3.1e-2.
