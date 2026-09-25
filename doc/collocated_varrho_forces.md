# Collocated variable density (rung V8): forces, pressure and the mass-adjoint approximate projection

*Design note, 2026-09-25 (architect pass). Branch `v8-cellforce` of flow (base `1fdee46`). It
replaces the V8 (WO-T) step of `SolverColocated` whenever `Grid::collocated && (varRho_ ||
csfActive())`. Everything else in flow stays byte-identical. Numbers marked [MODEL] come from the
exact-solve step-operator model in Appendix A. Numbers marked [CODE] were measured in this worktree
on host-openmp.*

## 0. The decision in ten lines

1. **V8's two defects have one cause.** V8 adds the pressure gradient as a face acceleration after
   the viscous solve. That makes its velocity update *non-incremental*: $P^n$ cancels exactly in
   the projection, so the velocity never sees it. Cause B is Chorin's splitting error. Cause A is
   the rotational term, which turns into an explicit pressure diffusion (§2).
2. **The pressure and every force go back inside the implicit predictor**, as they already do on
   every validated flow path.
3. **Pressure force on the cell: the finite-volume face integral.** The face pressure is
   reconstructed by continuity of acceleration, which gives
   $\rho_c\,\overline{(G_fP/\rho_f)}$, the average of the two face pressure accelerations.
   Surface tension and the uniform `set_body_force` drive enter the same way. Per-cell
   volumetric forces enter at the **cell value**.
4. **The face field handed to the projection is momentum-weighted:**
   $u_f = (\rho_L u_L + \rho_R u_R)/(\rho_L+\rho_R)$. This is the one centre-to-face map for which
   the cell pressure force is the *exact adjoint* of the constraint in the kinetic-energy inner
   product. It is the variable-density form of flow's gauge-exact transpose pair, and it is what
   makes the scheme stable at every density ratio.
5. **A balanced-force pre-projection runs once per step.** One extra Poisson solve with the same
   operator, warm-started, puts the gradient part of the forces into the pressure *before* the
   viscous operator acts. This keeps the hydrostatic column and the constant-κ droplet exactly at
   rest from step 1, at every μ, dt and ratio.
6. **The rotational (Timmermans) update stays, unchanged.** It again acts on the divergence of the
   viscously filtered predictor, as on the constant-ρ path.
7. **Result [MODEL]:** spectral radius exactly 1 in every one of about 1 100 step operators
   evaluated (2-D and 3-D, ratio 1 to 10⁶, dt 0.01 to 10⁴, κ = μ and κ = 0). The steady state is dt-independent to round-off. The
   static balances stay at machine zero. At uniform ρ the scheme reduces to the validated
   constant-ρ collocated scheme, so placement is at the cell value with order 2.
8. **The Basilisk face-acceleration form is removed from V8.** The suite-wide register entry that
   rejected it stands, and is now obeyed (§7).

---

## 1. Problem and scope

**Being built.** A new step for `SolverColocated` in the V8 configuration, i.e. variable density
and/or CSF surface tension, all-fluid, with periodic or domain-BC walls. The step must:

- be stable at every dt;
- have a dt-independent steady state;
- place volumetric forces at the cell value (user rule, 2026-09-25);
- keep the balanced-force properties exact: the ratio-1000 hydrostatic column (T1) and the
  constant-κ static droplet at ratio 1 and 1000 (T2).

**Out of scope, and unchanged byte for byte:**

- both staggered paths (constant and variable ρ, porous CFD-DEM);
- the constant-ρ collocated path without CSF (all schemes: ghost, gauge-exact, plain, embed);
- CFD-DEM coupling, gamma calibration, dem;
- V8 with an immersed solid, with the ghost projection or with harmonic ρ_f. These keep throwing
  (`requireCollocatedFaceForceScope`).
- V8's inflow and outflow faces keep today's openness predicate but get no new gate (§11, Q7).
- The momentum *transport* at high density ratio. There is no momentum-consistent VoF on the
  collocated grid; `enable_vof_momentum` is staggered-only and stays so.

## 2. Root cause (brief §4 A and B, confirmed and sharpened)

**Notation.** Cell ρ is $M=\mathrm{diag}(\rho_c)$. Face ρ is $M_f=\mathrm{diag}(\rho_f)$ with
$\rho_f(i)=\tfrac12(\rho(i)+\rho(i-s))$. The other operators are:

- $\Pi$: centre to face, $\tfrac12(u(i)+u(i-s))$;
- $R=\Pi^T$: face to cell average, $\tfrac12(a(i)+a(i+s))$;
- $G_f$: face difference $P(i)-P(i-s)$;
- $D=-G_f^T$: divergence;
- $A = M/\Delta t-\mu L$;
- $|K| = -D M_f^{-1}G_f \ge 0$, the pressure operator divided by ρ₀.

**2.1 V8 is non-incremental in the velocity. This single fact produces both A and B.**
V8 forms $u_f^*=\Pi u^* + \Delta t\,(F_f-G_fP^n)/\rho_f$ with $u^*=A^{-1}(Mu^n/\Delta t)$. The term
$-\Delta t M_f^{-1}G_fP^n$ lies exactly in the range the projection removes. So
$\varphi=\varphi_0-(\Delta t/\rho_0)P^n$, where $\varphi_0$ is the potential of the
pressure-free field. The cell update $u^*+R(\tilde a)$ then contains
$\Delta tF_f/\rho_f-\rho_0G_f\varphi_0/\rho_f$ and **no $P^n$ at all**. The pressure update becomes

$$P^{n+1}=\tfrac{\rho_0}{\Delta t}\varphi_0-\kappa\,D(\Pi u^*+\Delta tF_f/\rho_f)\;-\;\kappa\Delta t\,|K|\,P^n .$$

- **Cause A.** $P^n$ feeds back only through the last term. The per-mode multiplier is
  $-\kappa\Delta t\,\lambda(|K|)=-4\kappa\Delta t S/\rho$ for **every** pressure mode, not only
  the checkerboard. The instability threshold is set by the largest eigenvalue of $|K|$, which
  belongs to the $(\pi,\pi,\pi)$ mode ($S=3$). That gives $12\kappa\Delta t/(\rho h^2) < 1$, the
  explicit-diffusion limit.
  - The brief's measured multipliers (−12.0000 at μdt = 1, −1.2000 at 0.1) and the (0.875π, π, π)
    value 11.8478 are exactly this.
  - [MODEL] reproduces $8\mu\Delta t$ in 2-D and $12\mu\Delta t$ in 3-D, at every density ratio.
  - *Correction to the brief:* the checkerboard's invisibility to $\Pi$ is **not** why it
    decouples. In V8 no pressure mode reaches the velocity. The checkerboard is simply the
    fastest mode.
- **Cause B.** With κ = 0 the scheme is pure Chorin. Its steady state solves
  $-\mu Lu=(I-\Delta t\mu LM^{-1})(\dots)$, i.e. the forcing carries the factor $(1+\Delta t\mu\Lambda)$.
  - The register already records both symptoms: non-incremental Chorin gives −40 % Z&H drag, and
    face-primary mode 8 gives −75 % drag (`docs/decisions/flow.md:866ff`).
  - [MODEL] steady states at dt = 0.3/3/30 differ by 3e-2 to 9e-2 at ratio 1000, and by up to
    0.51 at ratio 1 (walled box, §B.4).
- **Placement.** At uniform ρ, V8's cell sees $R\Pi f=\tfrac14(f_{i-1}+2f_i+f_{i+1})$. §5.1 shows
  what that rule is at variable ρ.

**2.2 The obvious repairs fail [MODEL, Appendix B].**

- **κ = 0.** Stable, but still Chorin: dt-dependent and ¼(1,2,1) placement.
- **Pressure and forces inside the predictor, with the arithmetic face field Π (called D2
  below).** Weakly **unstable** at high contrast, for κ = 0 and κ = μ alike:
  - 1.0002 per step at ratio 1000 in one walled case;
  - up to 1.018 per step at ratio 10⁴, dt 3–10, confirmed by power iteration.

  The pair is not adjoint (§4.3), and nothing bounds the Uzawa spectrum.
- **The naive cell-centred $g-G_cP/\rho_c$ pair.** This is the imbalance that motivated V8. It
  grows 1.99× per step at ratio 10 and 245× per step at ratio 1000.
- **Predictor form without the pre-projection.** Stable and dt-independent. But the constant-κ
  droplet leaves face and cell at 1e-5 to 2e-3 after 30 steps (ratio 1 and 1000), instead of
  machine zero.

## 3. Constraints and invariants

- **Conventions.**
  - Face index $i$ of component $c$ separates cells $i-s$ and $i$, with $s=$ `strideOf(c)`.
  - Inner cells span $[G,e-G)$. The face kernels span $[G,e-G]$ (widened by one high plane, as
    V8 does today).
  - "Open" means `o(i) > 1e-12`, with `o` = `ox_/oy_/oz_`: the openness V8 uses today.
  - $\rho_0$ = `rho_`. The anisotropic weight is $w_c$ = `u_.w[c]` on every pressure difference.
  - Everything is in the solver's internal units.
- **ABC projection.** The face field is re-derived from the cell field every step, and only the
  increment φ corrects it. Never Rhie–Chow.
- **Incremental and rotational update.** $P\mathrel{+}=\tfrac{\rho_0}{\Delta t}\varphi-\kappa\,\mathrm{div}$,
  with κ = `rotWeight_·mu_`, or χ·μ_min under varProps. `projectPressureUpdate` is **unchanged**.
- **Placement.**
  - Volumetric forces (`cellForce_` = force_x/y/z incl. Boussinesq and closure ρg, and
    `drag_beta`) enter at the cell value. That is `Grid::atVelocity`, which is already the case
    for the drag diagonal.
  - Surface forces (pressure, viscous stress, CSF) enter as face integrals over the cell's
    control volume.
- **Precision.** Double throughout, no float casts in the new kernels (`no_float_operator_casts`).
- **Parallel.** np = 1 must be exactly decomposition-independent. np > 1 must stay within the
  existing derived tolerances of `test_vof_collocated_mpi`. Everything runs on-device; there is
  no host path.
- **Byte-identity.** Everything outside V8 is byte-identical (§10).

## 4. The design

### 4.1 Operators (per component c; closed faces per the rule above)

| symbol | definition (cell $i$ / face $j$) | kernel |
|---|---|---|
| $\rho_f(j)$ | $\tfrac12(\rho(j)+\rho(j-s))$ | as today |
| $\Pi_\rho$ (face field) | $u_f(j)=\dfrac{\rho(j)\,u(j)+\rho(j-s)\,u(j-s)}{\rho(j)+\rho(j-s)}$ | **new** `centerToFaceMassWeighted` |
| $\beta_c(j)$ (balanced force × ρ₀) | open ? $\rho_0\,[f_c^{const}+\tfrac12(f(j)+f(j-s))+\sigma\kappa_f\Delta C/h^2_{grad}]/\rho_f(j)$ : 0 | `buildFaceAccelVar(haveFb=true, incr=false, scale=ρ0)` + `addFaceAccelCsf(scale=ρ0)` |
| $\Phi_c(j)$ (surface acceleration) | open ? $[f_c^{const}+\sigma\kappa_f\Delta C/h^2_{grad}-w_c(P(j)-P(j-s))]/\rho_f(j)$ : 0 | `buildFaceAccelVar(haveFb=false, incr, scale=1)` + `addFaceAccelCsf(scale=1)` |
| $W_c(i)$ (wall weight) | $\tfrac12(\mathrm{open}(i)+\mathrm{open}(i+s))\in\{0,\tfrac12,1\}$ | inline |
| cell correction | $u_c(i)\mathrel{-}=\tfrac12(k(i)+k(i+s))$, $k(j)=$ open ? $w_c\,(\rho_0/\rho_f(j))(\varphi(j)-\varphi(j-s))$ : 0 | **new** `correctCellFaceAverageVar` |

Notes on the table:

- Rename `buildFaceAccelVar`'s and `addFaceAccelCsf`'s `dt` parameter to `scale`; the arithmetic
  is unchanged.
- In `correctCellFaceAverageVar`, $k$ must be written with **exactly** `faceAccelSubGradPhi`'s
  and `projectCorrectVar`'s grouping, so that the face correction and the cell correction see
  the same bits.
- **When ρ is constant** (CSF only, `!effVarRho()`):
  - $\Pi_\rho$ is the plain `centerToFace`;
  - $\rho_f$ is the scalar `rho_`;
  - $k$ uses the plain `projectCorrect` difference.

### 4.2 The step (V8 only; everything outside V8 unchanged)

**Step head.** Run the existing sequence: `updateProperties`, `fillCellForceGhosts`,
`fillDragBetaGhosts`, `updateVofCurvature`, `rebuildStencils`, `old_ ← u`. Then:

- **(S0)** `requireCollocatedFaceForceScope`. It additionally throws if `!incremental_`: V8 is
  defined for the incremental pressure-correction only.
- **(S1) Coefficients.** Call `projectBuildCoefficients()` once and set a per-step flag. The
  flag makes the later call inside `project()` skip the variable-ρ rebuild; ρ is frozen within
  the step. The flag is cleared at the head of every step and is only ever set on V8.
- **(S2) Balanced-force right-hand side.** Build $\beta_c$ on the faces into `faceAcc_[c]`. Then
  $b=\mathrm{divergOpen}(\beta)$ into `div_`, which is free at this point.
  - Take $m=\max|b|$, **globally** reduced under MPI.
  - If $m=0$: set $P_b^{new}:=0$ and skip S3.
- **(S3) Pre-projection solve.** Solve $A_p X=-b$, where $A_p$ is this step's MG operator
  $-D(o\rho_0w/\rho_f)G$. That is the main projection's operator and sign convention: bridge
  with the same `copyInner` + negate into `rhs1_`.
  - Use the same driver the V8 projection uses (Chebyshev / FCG / PCG dispatch) via
    `solvePressureSystem(rhs1_, pb1_)` (WO-1). Its rtol equals the main driver's.
  - Warm start: copy `Pb_`'s inner values into `pb1_` before the solve. `pb1_` is pure scratch
    and holds no cross-step state.
  - Store the iteration count in `lastBalancedIters_`.
- **(S4) Bookkeeping.** In one fused inner-cell kernel (g=1 → g=2 index map):
  $P(i)\mathrel{+}=X(i)-P_b(i)$; $P_b(i):=X(i)$.
  - `P_` stays the **total** physical pressure $Q$.
  - `Pb_` is the balanced part last added. It is a registered g=2 cell field
    (`"p_balanced"`, allocated lazily at the first V8 step, so `redistribute` carries it).
  - Then run the existing `fillGhosts(P_)` and `pressureBcGhost()`.

**Per Picard iteration:**

- **(S5)** Build $\Phi_c$ into `faceAcc_[c]`. It reads the live `P_`, which changes between outer
  iterations.
- **(S6) Predictor RHS.** New kernel `buildRhsColoVar(c)`, which replaces `buildRhsColoFF`:

  $$b_c(i)=r_s(i)\Big[\tfrac{\rho_c}{\Delta t}u^n_c-\rho_c\,a_K+\rho_c\,a_F+W_c(i)\,f_c(i)+\rho_c\,\tfrac12\big(\Phi_c(i)+\Phi_c(i+s)\big)\Big]+(\text{bc ? } b_{bc}:-\text{inh})$$

  - $\rho_c$, $a_K$, $a_F$, `dg` and the bc/inh term are exactly as in today's `buildRhsColoFF`.
  - $f_c$ = `cellForce_[c]` (0 if absent).
  - $f^{const}$ and CSF are **not** in the cell term; they are in Φ.
- **(S7)** Implicit momentum solve, unchanged (`smoothComp`; drag diagonal at the cell value).
- **(S8) `project()` for V8:**
  1. `fillVelGhosts`;
  2. $u_f^*=\Pi_\rho u^*$ (**no** `applyFaceAcceleration`). It reads ρ's depth-1 ghost ring,
     which S1's `fillPropGhosts` has just written;
  3. `div_ = divergOpen(u_f^*)`;
  4. rhs bridge;
  5. `projectSolve`;
  6. face correction by the existing `projectCorrectVar`/`projectCorrect`, then ghost fills,
     outflow and `buildOpenFaceField` as today;
  7. cell correction by `correctCellFaceAverageVar` (replaces `applyCellFaceAverageCorrection`);
  8. `maskVelocity`;
  9. `projectPressureUpdate`, **unchanged**. Its `div_` is now $D\Pi_\rho u^*$, the divergence of
     the viscously filtered predictor, which is Timmermans' quantity.

**Retired from the step:** `applyFaceAcceleration`, `addFaceIncrement` and
`applyCellFaceAverageCorrection`. Delete them; git keeps the history.

**Ablation switch.** `setBalancedPressure(bool)` (Python: `diagnostics.set_balanced_pressure(enabled)`,
developer tier, default on). When off, S2–S3 are replaced by $P_b^{new}:=0$, and S4 still runs.
Switching it off mid-run therefore removes the stored balanced part once. It exists for the one
gate that proves the pre-projection is load-bearing.

### 4.3 Why it is stable: the pair is adjoint in the kinetic-energy inner product

Let the cell pressure acceleration be $\Gamma P:=R\,O\,M_f^{-1}G_fP$ (O = openness) and the
constraint be $C:=D\,O\,\Pi_\rho$, with $\Pi_\rho=M_f^{-1}\Pi M$. Then

$$M\Gamma=M\Pi^TO M_f^{-1}G_f=-\big(D\,O\,M_f^{-1}\Pi M\big)^T=-C^T .$$

- The pressure force per unit volume is **exactly** minus the transpose of the constraint. The
  pressure therefore does no work on the discretely divergence-free fields in the energy norm
  $\tfrac12u^TMu$.
- At uniform ρ, $\Pi_\rho=\Pi$ and $M\Gamma=\Pi^TOG_f=-(DO\Pi)^T$. That is flow's gauge-exact
  pair (`gpCenterGrad` = transpose of `centerToFace`) verbatim.
- $\Pi_\rho$ is the unique centre-to-face map that makes the hydrostatically balanced Γ (§5.1)
  adjoint. The arithmetic Π (D2) breaks this, which is why D2 is unstable.

Consequences:

- **Real Uzawa spectrum.** The Schur complement $S=CA^{-1}C^T$ is SPD, because $A$ is SPD. The
  rotational update is the Cahouet–Chabard preconditioner
  $W=\rho_0(\Delta t\,|L_p|)^{-1}+\kappa I$, also SPD. So $WS$ has a real positive spectrum.
- **Pressure-correction part bounded by 1.** By Cauchy–Schwarz on the average,
  $z^T\Pi M\Pi^Tz=\sum_c\rho_c(\tfrac{z_-+z_+}{2})^2\le\sum_f\rho_fz_f^2$. Hence
  $CM^{-1}C^T\le|L_p|/\rho_0$ and, since $A\ge M/\Delta t$, the φ part of $WS$ is bounded by 1
  at **every** dt and density ratio. With κ = 0 this is a proof of stability for the pressure
  iteration.
- **Rotational part (not proved, measured).** It adds $\kappa\,\lambda_{\max}(C(-\mu L)^{-1}C^T)$.
  In the walled model $\lambda_{\max}\cdot\mu$ is 0.98 at ratio 1 and 1.28–1.40 at ratios 10³–10⁶.
- **[MODEL] spectral radius exactly 1.000000000** in:
  - 72 2-D density configurations × κ ∈ {μ, 0} × 7 dt, i.e. 1 008 operators (N ∈ {10, 11, 13},
    periodic/walled, ratio 10–10⁶, smeared and sharp, dt 0.01–10⁴);
  - 8 3-D configurations × κ × 5 dt (N = 6, 7, ratio 1–10⁶, walls);
  - variable μ (ratio 100 and 0.01) with κ = χμ_min.

  With κ = μ_max it is unstable (27–104 per step), which confirms the existing register rule
  χ·μ_min.
- **The radius-1 modes are the known neutral family.** It is the cell checkerboard that is
  invisible in the uniform-ρ interior (`doc/collocated_invisible_subspace.md`). Interfaces create
  no new neutral modes, because $\Pi_\rho$ sees the checkerboard where ρ jumps.

### 4.4 Why the steady state does not depend on dt

At a joint fixed point, P-stationarity gives $(\rho_0/\Delta t)\varphi=\kappa\,D\Pi_\rho u^*$.
Proposition 1 of `collocated_invisible_subspace.md` (the bracket is SPD) then gives φ = 0,
$u=u^*$, and

$$-\mu Lu = W f+ M R\,O\,\frac{f^{const}+F^\sigma-w\,G_fP}{\rho_f},\qquad D\,O\,\Pi_\rho u=0,$$

which contains no dt. $P_b$ is dt-free: it is solved in pressure units, $\beta$ carries no dt,
and it only splits $P$.

[MODEL] (walled box, ratio 1 and 1000, generic volumetric force plus a non-gradient face force,
fixed points by direct solve): $|u(0.3)-u(3)|\le1.5\times10^{-10}$ and $|u(3)-u(30)|\le4.6\times10^{-12}$
at $|u|=1.56$. V8 gives 3e-2 to 0.51.

### 4.5 Placement, and the wall rule

- **Interior.** The volumetric force is $f_c(i)$, the cell value.
  - At uniform ρ, Γ is the central difference and $\Pi_\rho=\Pi$. The step is then algebraically
    the validated constant-ρ collocated scheme; floating point differs only at ulp level.
  - Hence placement at the cell value to round-off, Taylor–Green order 2, and exact drag
    consistency ($W f$ and the drag diagonal both at the cell).
- **Wall rule, $W_c$.**
  - At a closed face the no-penetration condition holds the face's normal acceleration at zero.
    The wall pressure is therefore the one that balances the cell's own volumetric force: the
    Neumann condition $\partial p/\partial n=f_n$ in finite-volume form.
  - In the cell balance the closed face contributes "half of $f_c$" in place of a pressure
    difference. Equivalently, the normal volumetric force is multiplied by
    $W_c=\tfrac12(\text{open}_-+\text{open}_+)$.
  - Tangential components are untouched.
  - Surface terms (Φ, and the correction) take 0 at closed faces, as V8 does today.
  - [MODEL] Without the rule the ratio-1000 column leaves 7e-2 in the wall cells; with it,
    1e-13.
  - This is a V8-only rule. The constant-ρ path keeps its zero-gradient ghost and its known
    wall-cell leak (T1b CONTROL: cell 3.1e-2 at μ = 0 [CODE]).

### 4.6 The balanced-force pre-projection

- **Why it is needed.** The viscous operator does not map a face gradient to a face gradient:
  $\Pi_\rho A^{-1}M\Gamma G\Psi\notin\mathrm{range}(M_f^{-1}G_f)$. It fails even at uniform ρ,
  because $\Pi\Pi^T$ smooths each axis differently.
  - A balanced force that passes through the predictor therefore leaves a non-gradient face
    residue that the projection cannot remove.
  - The staggered path has the same defect at variable ρ: the "μdt² residue", with a T2 ratio-1000
    face of 1.3e-10 to 6.6e-6 [CODE].
  - The only exact remedy is to remove the gradient part **before** the viscous operator acts.
- **What S2–S4 do.**
  - S2–S3 compute $P_b$ with $w\,G_fP_b/\rho_f=\beta/\rho_0-r$, where $r$ is the
    $M_f$-solenoidal remainder, exactly solenoidal under $D\,O$.
  - S4 adds it to the pressure. The predictor then sees only the remainder $r$ (the part that
    physically drives flow) plus $P_{dyn}=Q-P_b$.
  - For gravity and constant-κ CSF, $r=0$, so $u^*=0$ exactly: the static state holds from the
    first step, at every μ and ratio.
- **Why the volumetric forces are also in β.** Nothing in flow tells a closure-generated ρg from
  any other cell force. Including them costs nothing extra and makes the hydrostatic column exact
  from step 1. For forces that are not balanced (drag, drives), $P_b$ only shifts the pressure's
  starting guess: the steady state is unaffected (§4.4).
- **Exactness in floating point.** Exactness holds to the driver's tolerance. See §9 G4 for the
  fallback if T2's 1e-14 is missed narrowly.

### 4.7 Performance, data and communication

- **New state.**
  - `Pb_`: one g=2 cell field, registered.
  - `pb1_`: one g=1 scratch.
  - Two scalars (the step flag, `lastBalancedIters_`).
  - Nothing is allocated outside V8.
- **New work per step.** Once per step:
  - build β (three face kernels, already written);
  - one `divergOpen`;
  - one global max-reduction;
  - **one extra Poisson solve** with the same operator, the same MG hierarchy and the same
    Chebyshev bounds. The bounds are estimated once per coefficient rebuild; with the step flag
    the main solve reuses them;
  - two copy kernels.

  Per Picard iteration, Φ replaces V8's face acceleration at equal cost. $\Pi_\rho$ reads two
  extra ρ values per face, which is negligible.
- **Communication.**
  - The extra solve's halo exchanges and reductions (PCG: 2 allreduces per iteration; Chebyshev:
    fewer).
  - One allreduce for the zero-RHS guard.
  - No new halo exchange of state: Φ and $\Pi_\rho$ read only ghosts that are already filled
    (ρ, `cellForce_`, C, κ, and `P_` after its existing fill).
- **Expected cost.**
  - The pre-projection warm-starts from last step's $P_b$. For a static interface its residual
    starts at rtol, so it costs about 0–1 iteration.
  - For a moving interface expect roughly 50–80 % of a cold solve. That is up to about 1.5–1.8×
    the V8 pressure cost, where the pressure solve dominates at ratio 1000.
  - Measured in WO-5 against a hard budget. The fallback is in §11 Q3.
- **Device residency.** Every new kernel is a Kokkos `parallel_for` on existing views. No host
  round-trip beyond the scalar reduction the drivers already do.

## 5. What changes for the balance of pressure, gravity and surface tension

### 5.1 The identity that settles the user-rule conflict

Take the half-cell momentum balance on each side of a face, with piecewise-constant $\rho$ and
$f$, and require the acceleration to be continuous across the face. That gives the face pressure

$$p_f=\frac{\rho_Rp_L+\rho_Lp_R}{\rho_L+\rho_R}+h\,\frac{\rho_Rf_L-\rho_Lf_R}{2(\rho_L+\rho_R)} .$$

The finite-volume cell balance per unit volume (h = 1), $\rho_ca_c=f_c-(p_{f+}-p_{f-})$, is then
**algebraically**
$a_c=\overline{(\tfrac12(f_L+f_R)-G_fp)/\rho_f}$, which is V8's cell rule.

- **V8** was therefore the cell-value force *plus a force-aware face-pressure reconstruction* (the
  second term). At uniform ρ that reconstruction folds $-\tfrac14\delta^2f$ into "the pressure",
  so the cell effectively sees ¼(1,2,1)f. The user rule rejects that; the placement test measures
  a 3.8 % misfit at N = 16.
- **This design** drops the force term from the reconstruction. It keeps the ρ-weighted first term
  (a surface-force face integral, as the rule demands) and the cell-value $f_c$.
- The two coincide exactly **whenever $f/\rho$ is continuous across the face**: gravity, and any
  force proportional to ρ. For gravity there is thus no conflict at all. Gravity is a volumetric
  force at the cell value, $f_c/\rho_c=g$, and it is balanced to round-off at every interior cell,
  because $\overline{(\rho_fg/\rho_f)}=g$.

### 5.2 Table

| force | enters | static balance |
|---|---|---|
| pressure | Γ: FV face integral, acceleration-continuous reconstruction | defines the balance |
| gravity (closure ρg in `cellForce_`) | cell value, $W f$ | **exact** at interior and wall cells, from step 1 (pre-projection). Column ratio 1000 [MODEL]: cell and face ~1e-13 |
| CSF (V4 pairing, `csfFaceForce`) | face form in Φ (surface force) | **exact** for constant κ at every ratio, μ and dt, from step 1. [MODEL]: ≤ 4.5e-16. Closed faces carry 0, as today |
| `set_body_force` $f^{const}$ | face form in Φ: **a uniform per-volume drive is a mean pressure gradient** (default, Q1) | **exact**; this is how the periodic hydrostatic box is balanced |
| per-cell forces not ∝ ρ (Boussinesq, user fields) at density jumps | cell value (user rule) | consistent but not exact at interface cells: a residual of $O(\Delta(f/\rho))$ in the cell layer. At uniform ρ, the same balance as the validated constant-ρ path |
| drag β | target and diagonal at the cell | exact consistency ($u=U_0$) |

- **The O(1) interface-cell imbalance that motivated V8** ($g_c-G_cP/\rho_c$) is gone. It came from
  the arithmetic cell gradient divided by $\rho_c$; Γ is the ρ-weighted average of face
  accelerations. The face-acceleration placement was never needed to fix it.
- **Periodic hydrostatic box.** Its offset $\bar\rho g$ must be supplied through `set_body_force`
  (face form), not folded into the per-cell field. [MODEL], ratio 1000, 30 steps:

  | offset supplied as | cell, μ = 0 | cell, μ = 0.1 |
  |---|---|---|
  | per-cell field | 20.5 | 0.23 |
  | face form | 2.4e-13 | 1.7e-14 |

  T1's periodic case changes accordingly (WO-0). On today's V8 that change is bitwise inert:
  $\tfrac12(x+x)=x$.

### 5.3 Does the density-ratio-~100 rating lift? (brief §6.4)

- **Static and balanced rating: yes.** T2's "~4× per step at μdt/(ρ_min h²) = 0.45" was cause A,
  as the rotational ablation already showed [CODE]. This design is stable at ratio 1000 for every
  T2 μ, and to 10⁶ in the model. T2 gates it at 1000 (WO-3).
- **Rating with motion: not yet.** CLAUDE.md's "collocated ratio ≲ 100 with motion" also rests on
  momentum transport. The collocated grid has no momentum-consistent VoF, and this design does not
  touch that. It stays until a moving-interface gate is run (Q5).

## 6. Rejected alternatives (with the evidence)

| alternative | why rejected |
|---|---|
| **κ = 0 on V8** (the one-line fix) | Cures A only. Still Chorin (§2.1): dt-dependent steady state (TG error 9.2/9.8/10.6, order −0.1; drag 2.48 [CODE]) and ¼(1,2,1) placement. It also contradicts the register's "rotational update must be restored" |
| **Rotational term on $D(\Pi u^*)$ only, before $a_f$** | At steady state $D\Pi u^*=-Da_f\neq0$, so P drifts linearly (the brief's pre-check). Cause B untouched |
| **Filtered rotational update** (`setRotationalFilter`) | Damps A's feedback without removing its source (P still never reaches u); B untouched |
| **Complete Basilisk `centered.h`** (the lagged $g$ added before and removed after the viscous solve) | Removes the gross (1+dtμΛ) factor. But the face field keeps $\Delta t(I-\Pi R)\,a_f$, a Rhie–Chow-type term. The steady constraint becomes $D\Pi u=-\Delta t\,D(I-\Pi R)a_f$, which depends on dt. The pressure is also non-incremental. Violates both the ABC and the rotational-update entries |
| **Predictor form with the arithmetic Π (D2)** | Non-adjoint pair: unstable at 1.0002 (ratio 1000) and 1.018 per step (ratio 10⁴) [MODEL] |
| **Naive cell-centred $g-G_cP/\rho_c$ with $(\rho_0/\rho_c)\,\overline{G\varphi}$** | 245× per step at ratio 1000 [MODEL], plus the O(1) hydrostatic imbalance |
| **Predictor form without the pre-projection** | Fails the settled balanced-force requirement: droplet 2e-4 (ratio 1) and 1.6e-5 (ratio 1000) after 30 steps, instead of machine zero [MODEL] |
| **Force-aware reconstruction for volumetric forces** (V8's cell rule inside the predictor) | Balances *every* conservative force exactly, but places ¼(1,2,1)f. Violates the user rule and fails the placement test. Kept only as the alternative in Q2 |
| **Lagged pre-projection** (exact first step, then reuse $P_b^{n-1}$) | Saves the second solve but makes the balanced force first-order in time for moving interfaces, with two code paths. Held as the performance fallback (Q3), not the default |

## 7. Reconciliation with the register

- **Suite-wide "Basilisk face-acceleration form REJECTED — do not reintroduce it"**
  (`docs/decisions/suite-wide.md:300`). V8/WO-T had reintroduced it, unregistered, and §2.1 shows
  it produced exactly the defects the entry names: rotational-update instability and the
  non-adjoint pair. This design **removes** it.
  - No velocity is ever incremented on faces outside the projection's own correction.
  - The pre-projection computes a pressure, not a velocity increment.
  - The entry's positive content ("the incremental predictor with the cell pressure gradient = the
    EXACT TRANSPOSE of the centre→face constraint, used in both the predictor and the
    correction") is kept. At variable density the transpose is taken in the kinetic-energy inner
    product ($M\Gamma=-C^T$) and reduces to the old statement at uniform ρ.
  - No new decision is needed to *keep* a face-acceleration form. A new entry records that V8 now
    conforms, and extends the transpose rule to variable ρ (§12).
- **ABC, never Rhie–Chow** (`flow.md:494`, `:2053`). Kept: the face field is $\Pi_\rho u^*$ plus
  the increment correction only. The rejected Basilisk-complete variant is the one that would have
  carried a Rhie–Chow term.
- **Rotational (Timmermans) update must be restored** (`flow.md:2281`); **χ·μ_min under varProps**
  (`flow.md:2395`). Both kept unchanged and now stable on V8.
- **Wall-banded blend** (`flow.md:1249`). Not needed: V8 is all-fluid.
- **User rule 2026-09-25** (`flow.md:3162`). Satisfied: cell value for volumetric forces, face
  integrals for surface forces. §5.1 records the one place where the rule has a physical cost
  (non-ρ-proportional forces across density jumps). The single reinterpretation, `set_body_force`
  as a mean pressure gradient, is flagged as Q1.

## 8. Work orders (for `opus-implementer`; in dependency order)

Every WO ends with the byte-identity gate (G6) and a green battery. Record the state_hash hashes
before WO-1 and compare after every WO.

**WO-0 — instruments first (no production change).**

- Copy Appendix A to `tests/study/colocated_varrho_symbol.py` as an instrument.
- Write `tests/python/test_colocated_varrho_stability.py` (G2 + G3, §9). Run it on the
  **current** build, record that it **fails**, with the numbers, in the commit message. Do not
  register it in ctest yet.
- Edit T1's periodic case in `test_vof_collocated.cpp`:
  - the closure becomes `LinearMix("rho", {0, -g})`;
  - add `setBodyForce(0, 0, rbar*g)`;
  - the reported dP/dz offset is unchanged.

  Gate, on the current code:
  - T1 still passes its existing thresholds, staggered and collocated;
  - dP/dz agrees with the pre-edit run to 1e-12 relative. The two forms differ only in
    summation order, so round-off-level face and cell numbers may change in their last digits.

**WO-1 — pure refactor.**

- Extract `solvePressureSystem(CCField rhs1, CCField x1)` from `projectSolve`: the driver dispatch
  plus the Chebyshev bound estimate on `rhs1`, returning the iteration count.
- `projectSolve` calls it with `(rhs1_, phi1_)` and keeps its post-solve pinning, copy-back and
  ghost code.
- Gate: G6, plus `test_vof_collocated` output identical line for line.

**WO-2 — the V8 step (one atomic commit, so the battery stays green).**

- **Kernels** (`collocated_varrho.hpp`):
  - `centerToFaceMassWeighted`, with the same index range as `centerToFace`;
  - `correctCellFaceAverageVar`;
  - rename `dt`→`scale` in `buildFaceAccelVar` and `addFaceAccelCsf`;
  - rewrite the header comment to this note's §4 (drop the Basilisk narrative).
- **Solver:**
  - S0–S4 as `balancedForcePressure()`, called from `step()` at the point in §4.2, V8 only;
  - the S1 step flag in `projectBuildCoefficients`;
  - S5–S6 as `buildRhsColoVar` in place of `buildRhsColoFF`;
  - S8 in `projectAssembleDivergence` and `projectCorrectVelocities`;
  - delete the three retired functions;
  - the `!incremental_` throw;
  - the `Pb_`/`pb1_` state;
  - `setBalancedPressure` / `lastBalancedIterations`, with bindings on `diagnostics` (names
    checked against `docs/NAMING.md`; the canon wins).
- **Minimal test edits to stay green.** T1b's `CHECK(b.cellU < a.cellU)` becomes
  `CHECK(b.cellU <= 1e-10 && a.cellU <= 1e-10)`: there is no remnant left to decay, and both
  values are solver-tolerance-level.
- **Gates:**
  - G1 (the placement test with the SKIP removed passes all three collocated var-ρ checks);
  - G2 and G3 pass;
  - G4 at the existing thresholds;
  - G6.

**WO-3 — gates.**

- Register `test_colocated_varrho_stability` in the Python ctest list.
- Tighten T1/T1b/T2 to G4's numbers.
- T2's μ sweep: gate μ = 0.1 and the base μ as well, with no throw.
- Add the ablation gate (pre-projection off: T2 ratio-1 face ≥ 1e-8).
- T5: V8 with non-incremental pressure throws.
- Rewrite the T2 narrative (the "ratio ~100" paragraph) and the placement test's SKIP comment.
- Gate: the battery, 168/168 with the new test.

**WO-4 — MPI.**

- `test_vof_collocated_mpi` at np = 1, 2, 4 and `OMP_NUM_THREADS` ∈ {1, 2, 4, 8}, with
  `--bind-to none`.
  - np = 1 stays exact (tolerance 0).
  - np > 1: u/uf within 1e-11, P within 1e-9, C within 1e-11.
  - The iteration rule (exact at np = 1, ±1 above it) extends to `lastBalancedIterations`.
- Add a V8 redistribute case: `rebalance_by_weights` with non-uniform weights on hydro-z, then
  5 steps. Compare against the same run without the rebalance at the np > 1 tolerances. This
  proves `Pb_` is carried.

**WO-5 — performance (measure; do not tune numerics).**

- Setup: host-openmp, `OMP_NUM_THREADS=8`. A ratio-1000 droplet of 32³ at dt = 0.5·dt_σ, with a
  uniform imposed translation (or a Rayleigh–Taylor case).
- Report, per step:
  - main and pre-projection iterations;
  - `tProjection_`;
  - step time, against the pre-change V8 with `set_rotational_pressure(False)` (the only stable
    pre-change reference).
- Gate G7.

**WO-6 — documents.**

- `CLAUDE.md`:
  - "Collocated solver" and VoF scope lines (§5.3);
  - one new bullet under settled decisions.
- `doc/variable_density_projection.md` §4: the collocated paragraph.
- Hand the §12 entries to the orchestrator for `../docs/decisions/flow.md` (the umbrella repo).
- Optional mechanical rename `colocatedFaceForce()` → `colocatedVarRho()` (Q8).

**WO-7 (optional measurement, decides Q5).** A translating droplet at ratio 1000 (Galilean test)
and the RT growth rate at ratio 3 against the staggered `tests/study/rayleigh_taylor.py`. Report
only.

## 9. Verification gates (numbers)

**G1 — placement** (`tests/python/test_cell_force_placement.py`, SKIP removed; `OMP_NUM_THREADS=4
PYTHONPATH=build_omp`), collocated var-ρ:

- cell misfit < 1e-6 (expected ≈ 5e-14, as the constant-ρ path);
- TG order > 1.8 (expected 1.99–2.00);
- drag < 1e-8 (expected ≤ 1.5e-12).

The other three combinations are unchanged.

**G2 — stability probe** (new test). Setup:

- 16×16×4, μ = 1, h = 1, rotational update at its default;
- dt ∈ {0.1, 1, 10, 100} (μdt/ρ_min h² up to 100);
- densities: (a) the uniform-ρ variable mode; (b) a frozen hand-set slab of ratio 1000
  (`set_field("rho")` + density mode, Chebyshev rtol 1e-13);
- forces: the brief's TG force and $f_x(x)$;
- seed `set_field("p")` += ε(−1)^{x+y+z} with ε = 1e-6·max|f|; run 100 steps. The physical
  pressure is z-uniform, so it has no (π,π,π) content: the amplitude measures the seed alone.

Gate:

- the (π,π,π) FFT amplitude of P − mean satisfies $A_{100}\le A_0(1+10^{-6})$;
- no NaN;
- at uniform ρ, max|u| at step 100 is within 1e-9 relative of the unseeded run (the seed is
  exactly invisible there);
- in the slab, max|u| ≤ 2× the unseeded run. The seed is legitimately visible at interface
  cells, so this is boundedness only.

Pre-change it must fail, with multiplier ≈ 12μdt.

**G3 — dt-independence** (new test). Setup:

- 16×16×4 with no-slip walls at ±y;
- frozen slab of ratio 1000 (and ratio 1);
- a force with a non-gradient part (the TG force);
- dt ∈ {1, 10, 100}, marched until max|Δu| per step ≤ 1e-12·max|u| (cap 20000 steps);
- Chebyshev rtol 1e-13 for the ratio-1000 slab, PCG 1e-13 for ratio 1.

Gate: max|u(dt_i) − u(dt_j)| ≤ 1e-9·max|u|. Pre-change (κ = 0 ablation) it must fail at
≥ 1e-3.

**G4 — `test_vof_collocated`.** The WO-3 values:

T1 and T1b run at the **default** Chebyshev rtol (1e-9); T2 runs at an explicit 1e-14.

- **T1**, collocated, walled and periodic:
  - face < 1e-12 walled (unchanged) and < 1e-10 periodic (was 1e-8);
  - dP/dz < 1e-6 (unchanged; report it; expected ≤ 1e-9);
  - **new** cell < 1e-10 (today's V8: 2.8e-8);
  - μ sweep: face < 1e-12 and cell < 1e-10.
- **T1b:** the V8 cell ≤ 1e-10 at 100 and 400 steps. CONTROL reported unchanged.
- **T2:**
  - ratio 1: face and cell < 1e-14 (unchanged);
  - ratios 10–1000: face and cell < 1e-13 (new);
  - ratio-1000 μ sweep over {0, 0.01, 0.1}: no throw, face < 1e-13, and face < staggered face.
- **T3, T4:** unchanged tolerances.
- **T5:** + the non-incremental throw.

"Measure then freeze":

- **Missed by less than 10×.** Set the threshold at 10× the measured value, with a comment. For
  T2 ratio 1 specifically, first set the pre-projection rtol to 0.1× the main rtol (a named
  constant in `balancedForcePressure`, not a knob).
- **Missed by more.** First re-run the case locally, not committed, with
  `setPressureChebyshev(true, 500, 1e-14)`. A T1 number that sits at the default rtol's level is
  the solver, not the scheme; the expectation at 1e-14 is ≤ 1e-13. Only a miss at 1e-14 is
  conceptual: then stop and report.

**G5 — MPI.** WO-4's numbers; also `suite/docs/wo_vof_mpi_parity_gates.md` §8 at thread counts
1/2/4/8.

**G6 — byte-identity.** `tests/regression/state_hash.py`: all 12 cases (staggered_bed,
colocated_ghost, colocated_gauge_exact, colocated_plain, colocated_embed, colocated_advect,
colocated_advect_bc, channel, vof_droplet, scalar, porous, scene_moving) give hashes identical to
before WO-1. Also:

- the full battery, 167/167 → 168/168 (`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`,
  `--bind-to none`, np8 last);
- `sdflow_regression.py` baselines unchanged;
- `test_vardensity_projection` output unchanged line for line.

**G7 — performance.** V8 step time ≤ 1.5× the pre-change κ=0 V8 on WO-5's moving case. Mean
pre-projection iterations ≤ mean main iterations. On a miss: report, keep the numerics, and the
orchestrator decides Q3.

## 10. Byte-identity argument outside V8

Every change is reached only when `colocatedFaceForce()` is true, i.e.
`Grid::collocated && (varRho_ || csfActive())`:

- **`step()`.** The new `balancedForcePressure()` call sits under that predicate. Staggered folds
  it to false, and the constant-ρ collocated path without CSF evaluates it false.
- **`projectBuildCoefficients`.** The skip flag is only ever set by `balancedForcePressure`.
- **`projectSolve`.** Changed by a pure extraction (WO-1), gated separately by G6 before anything
  else moves.
- **Other edits.** The momentum-RHS dispatch, `projectAssembleDivergence` and
  `projectCorrectVelocities` are edited only inside their existing `coloFF` / `colocatedFaceForce()`
  branches. The non-V8 text of those functions is unchanged: the reviewer verifies this on the
  diff.
- **New fields.** `Pb_` and `pb1_` are allocated and registered lazily at the first V8 step, so
  the registry, redistribution and memory of every other run are unchanged.
- **V8 kernels.** `collocated_varrho.hpp` is included by others but its kernels are called only
  from V8. The `dt`→`scale` rename is inert.
- **No state_hash case is V8.** `vof_droplet` is staggered.

## 11. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1 | On V8, is `set_body_force` a mean pressure gradient (face form, balanced) or a volumetric force (cell value)? | **user preference** | **Face form.** It matches the staggered semantics and makes periodic hydrostatic / pressure-driven two-phase boxes exact. The cell-value alternative leaves $O(\bar\rho g/\rho_{min})$ interface-cell errors (20.5 in §5.2) |
| Q2 | Per-cell volumetric forces not ∝ ρ across density jumps: cell value (user rule; interface-cell residual $O(\Delta(f/\rho))$) or force-aware face form (exact balance, ¼(1,2,1) placement)? | **user preference** | **Cell value**, per the rule. The consequence is recorded in the register entry |
| Q3 | Cost of the pre-projection | **fact** (WO-5) | Pre-project every step. If G7 fails, the orchestrator asks the user between accepting the cost and the lagged variant (§6, last row) |
| Q4 | Is the O(h) interface-local error of the momentum-weighted face velocity visible in moving-interface accuracy? | **fact** (WO-7) | Accept (it is the face momentum/mass velocity; stability requires it); report |
| Q5 | Does the "collocated ratio ≲ 100 with motion" rating lift? | **fact** (WO-7) | Keep it; re-rate only the static/balanced rating to 1000 |
| Q6 | 3-D margin of κ = μ ($\kappa\lambda_{max}(C(-\mu L)^{-1}C^T)<2$ is measured, not proved) | **fact** | G2 at dt ≤ 100 in code covers it. If G2 fails with κ = μ but passes with κ = 0, stop and report. Do not ship a κ change without the architect |
| Q7 | V8 inflow/outflow faces: the design reuses today's openness predicate; nothing gates them | **fact** | Out of scope; no gate; noted in CLAUDE.md scope |
| Q8 | Rename `colocatedFaceForce()` (the name now lies) | preference (minor) | Rename, mechanically, in WO-6 |
| Q9 | Chebyshev bounds are now estimated on the pre-projection RHS (V8 only) | **fact** | Accept. If T1/T2 main iteration counts rise > 2×, estimate on the main RHS instead |

## 12. Draft register entries (for `docs/decisions/flow.md`)

```
### Collocated variable density (V8): the mass-adjoint ABC pair + a balanced-force pre-projection; the face-acceleration form is retired
- area: flow
- source: flow doc/collocated_varrho_forces.md; flow <commit of WO-2>
- decided: 2026-09-25
- status: settled
- quote: |
    V8 (WO-T) added the pressure gradient and every force as a MAC face acceleration AFTER the
    implicit viscous solve (Basilisk centered.h). That makes the velocity update non-incremental:
    P^n cancels exactly in the projection. Chorin's splitting error followed (steady state x(1 +
    dt mu Lambda), TG order -0.1, drag 2.48), and the rotational term became an explicit pressure
    diffusion with multiplier -12 kappa dt/(rho h^2) at (pi,pi,pi) (measured -12.0000). The step now
    puts the pressure and all forces inside the implicit predictor.
    Pressure, CSF and set_body_force act on the cell as rho_c * avg_faces[(F_f - w G_f P)/rho_f]:
    the FV face integral with the acceleration-continuous face pressure. Per-cell forces act at the
    cell value, with the normal component halved per closed face (the wall's Neumann pressure).
    The projection's face field is the momentum-weighted interpolation
    (rho_L u_L + rho_R u_R)/(rho_L + rho_R), the unique map with M*Gamma = -(D O Pi_rho)^T (the
    gauge-exact transpose, in the kinetic-energy inner product). A per-step pre-projection with the
    same operator puts the gradient part of the forces into P before the viscous operator acts.
    The rotational update is unchanged. Model: spectral radius exactly 1 in ~1100 step operators
    (ratio to 1e6, dt to 1e4, kappa = mu and 0); steady state dt-independent to 1e-10; hydrostatic and
    constant-kappa droplet at machine zero from step 1.
- rejected: the face-acceleration form after the viscous solve (non-incremental; unstable with the
    rotational term); kappa = 0 (still Chorin); completing Basilisk centered.h with the lagged g
    (a Rhie-Chow-type dt-dependent steady constraint); the arithmetic centre->face map with the
    rho-weighted pressure force (non-adjoint: 1.018/step at ratio 1e4); the naive g - G_c P/rho_c
    pair (245x/step at ratio 1000); the predictor form without the pre-projection (static droplet
    1e-5..2e-3); the force-aware reconstruction for volumetric forces (places 1/4(1,2,1) f)
- why: stability at every dt and ratio, a dt-independent steady state, and cell-value placement
    need the incremental predictor with an adjoint pair. Exact per-step balanced force needs the
    gradient part removed before the viscous operator, which does not map face gradients to face
    gradients even at uniform rho.
- conflict: conforms to suite-wide "Basilisk face-acceleration form REJECTED" (which WO-T had
    violated) and to "ABC never Rhie-Chow"; keeps "rotational update must be restored" and
    "chi*mu_min under varProps". Consequence of the 2026-09-25 user rule, recorded: a per-cell force
    whose f/rho jumps across a density interface is balanced only to O(delta(f/rho)) in the
    interface cells.

### On the collocated variable-density path set_body_force is a mean pressure gradient (face form)
- area: flow
- source: flow doc/collocated_varrho_forces.md §5.2, Q1
- decided: 2026-09-25 (default; user preference pending)
- status: provisional
- quote: |
    A uniform per-volume drive in a two-phase fluid is the stand-in for a mean pressure gradient
    (the periodic hydrostatic offset rho_bar*g, a pressure-driven channel), so it is a SURFACE force
    and enters as rho_c * avg_faces(f_const / rho_f), exactly balanced by a linear pressure.
    Per-cell force fields stay volumetric (cell value). On the staggered grid and at uniform
    density the two readings coincide bit for bit.
- rejected: f_const at the cell value on V8 (periodic hydrostatic box: cell 20.5 after 30 steps at
    ratio 1000, mu = 0, against 2.4e-13)
- why: exact balance of the only uniform drives two-phase users apply; matches the staggered semantics
```

---

## Appendix A — the step-operator model (reproduces §2, §4.3 and Appendix B)

This is a dense, exact-solve, 2-D, h = 1 model of the step, with ρ₀ = 1. Run it with
`OMP_NUM_THREADS=8 python colocated_varrho_symbol.py`. Its expected output:

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

The forced (affine) runs of §4.4–§5 add these terms to the same step:

- $W f_c$ and $\rho_cR(\Phi)$ to the predictor RHS;
- the pre-projection $P_b=\rho_0L_p^{+}D(o\beta)$;
- $Q\mathrel{+}=P_b^{new}-P_b^{old}$.

The fixed points of §4.4 come from solving $(I-T)s=b$ with the affine vector $b$ taken with $P_b$
already in place.

## Appendix B — evidence tables [MODEL unless marked]

**B.1 Spectral radius, 2-D N = 12, μ = 1.** Columns are dt = 0.01, 0.1, 1, 10, 100, 10⁴.

(D3 was checked on the same six periodic and walled configurations: 1.000000 at every dt.)

| config | V8 κ=μ | V8 κ=0 | D2 κ=μ | D3 κ=μ |
|---|---|---|---|---|
| uniform | 1, 1, **8**, 80, 800, 8e4 | 1 | 1 | 1 |
| slab ratio 1000 | 1, 1, **7.73**, 77, 773, 7.7e4 | 1 | 1 | 1 |
| disk ratio 1000, walled | 1, 1, **7.71**, 77, 771, 7.7e4 | 1 | 1 | 1 |

In 3-D (N = 6), V8 with κ = μ gives 3.6 at dt 0.3 (= 12μdt).

**B.2 D2 vs D3 over random smeared and sharp shapes.** N ∈ {10, 11, 13}, periodic and walled,
worst radius over dt ∈ [0.3, 10⁴] (D3 over [0.01, 10⁴]):

| ratio | D2 κ=0 | D2 κ=μ | D3 κ=0 | D3 κ=μ |
|---|---|---|---|---|
| 10 / 100 | 1 | 1 | 1 | 1 |
| 10³ | 1.0002 | 1.00008 | 1 | 1 |
| 10⁴ | 1.0149 | 1.0183 | 1 | 1 |
| 10⁶ | — | — | 1 | 1 |

**B.3 Balanced-force transients.** N = 12, 30 steps, D3, cell / face:

| case | with pre-projection | without pre-projection |
|---|---|---|
| walled column ratio 1000, μ 0 / 0.1 | 1.2e-13 / 8e-14 · 3.0e-14 / 2.9e-14 | 1.6e-3 / 7e-14 · 1.5e-3 / 2.9e-14 |
| same, wall rule off | 7.0e-2 (cell) | — |
| droplet κ const, ratio 1, dt 0.5 / 5 / 50 | ≤ 4.5e-16 | 2.2e-4 · 1.1e-3 · 2.1e-3 |
| droplet κ const, ratio 1000, dt 0.5 / 5 / 50 | ≤ 2.1e-16 | 1.6e-5 · 9.8e-5 · 4.7e-4 |

**B.4 dt-dependence of the forced steady state.** Walled box; max|u(dt_a) − u(dt_b)| for
(0.3, 3) / (3, 30):

| ratio | D3 κ=μ | V8 κ=0 |
|---|---|---|
| 1 | 1.1e-13 / 2.0e-14 | 5.4e-2 / 5.1e-1 |
| 1000 | 1.5e-10 / 4.6e-12 | 2.9e-2 / 9.3e-2 |

**B.5 [CODE], from the brief and the worktree logs.**

- V8 T2 at ratio 1000 is stable only with the rotational term off: face 4.3e-18 and 5.9e-18.
- Staggered T2 face at ratio 1000: 1.3e-10 / 4.1e-6 / 6.6e-6 at μ = 0 / 0.01 / 0.1.
- V8 T1 dP/dz today: 2.3e-8.
- T1b CONTROL cell (constant-ρ collocated): 3.1e-2.
