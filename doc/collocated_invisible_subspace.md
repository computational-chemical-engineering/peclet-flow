# Steady-state defects of a collocated cut-cell approximate projection: instability, attractor families, and the invisible pressure subspace

*Working note, 2026-08-21 (F. Peters / Claude session). Math is LaTeX-ready ($ inline, $$ display).
Every numerical claim carries its artifact (log path or commit). Status labels: [PROVED] = follows
from the stated assumptions; [MEASURED] = observed in the solver; [INFERRED] = consistent with all
measurements but not independently computed.*

## 1. Setting

Cell-centered (collocated) velocities $u \in \mathbb{R}^{3N}$, pressures $P, \phi \in \mathbb{R}^{N}$
on a periodic Cartesian grid with an immersed solid described by a cell-centered SDF; $h = 1$
(cell units). Discrete operators of the gauge-exact aperture scheme (`SolverColocated`,
`set_collocated_scheme("gauge-exact")`):

- $A_u = (\rho/\Delta t) I - \mu L$: momentum operator; $L$ the IBM (Robust-Scaled cut-cell)
  vector Laplacian, masked at solid-centered cells.
- $\Pi: u \mapsto \bar u$: the face average, $\bar u_f = \tfrac12(u_i + u_j)$ over the two
  adjacent cell centers (solid-centered cells contribute their masked value $0$).
- $D_\alpha$: the aperture (open-area–weighted) divergence acting on face fields;
  $D = D_\alpha \Pi$ is the constraint operator on cell fields.
- $G_f$: the compact face-difference gradient (cell field to faces); the Poisson operator of the
  projection is $A_p = D_\alpha G_f$ (SPD up to the constant nullspace; solved by MG-PCG).
- $G$: the cell-centered *gauge-exact* gradient (`gpCenterGrad`): central where both axis
  neighbours are fluid-centered, one-sided quadratic
  $(-3p_i + 4p_{i+1} - p_{i+2})/2$ toward the fluid at a solid-centered neighbour,
  two-point fallback, $0$ at sandwiched cells. **$G$ never reads a solid-centered value.**

One step of the scheme (steady Stokes; body force $F$; exact linear solves assumed here,
inexactness addressed in §5):

$$\begin{aligned}
&\text{(momentum)}   && A_u u^* = (\rho/\Delta t)\,u^n + F - G P^n,\\
&\text{(Poisson)}    && A_p \phi = -\,D_\alpha \Pi u^*,\\
&\text{(faces)}      && u_f^{n+1} = \Pi u^* - G_f \phi \quad(\Rightarrow D_\alpha u_f^{n+1} = 0),\\
&\text{(cells)}      && u^{n+1} = M\,(u^* - G \phi), \qquad M = \text{solid mask},\\
&\text{(pressure)}   && P^{n+1} = P^n + (\rho/\Delta t)\,\phi - \mu\, D_\alpha \Pi u^*.
\end{aligned}$$

The last line is the incremental-rotational (Timmermans / "PM II") update.

**A key structural fact.** In the aperture scheme, a face between a cut fluid cell and a
*solid-centered* cell can have open area $\alpha_f > 0$. Hence $A_p$ has non-trivial rows for
solid-centered cells: **solid-centered pressures are coupled degrees of freedom of the
constraint**, while $G$ (which delivers pressure forces to the momentum equation and the cell
correction) never reads them.

## 2. The naive fixed point [PROVED]

**Proposition 1.** *Under exact solves, a joint stationary point ($u^{n+1}=u^n$ and
$P^{n+1}=P^n$) has $\phi = 0$, $D_\alpha \Pi u^* = 0$, $u = u^*$, and satisfies the
$\Delta t$-free steady system*
$$ -\mu L u = F - G P, \qquad D_\alpha \Pi u = 0. $$

*Proof.* $P$-stationarity gives $(\rho/\Delta t)\phi = \mu D_\alpha \Pi u^* = -\mu A_p \phi$,
i.e. $((\rho/\Delta t) I + \mu A_p)\phi = 0$. The bracket is SPD on the mean-free complement and
strictly positive on constants ($A_p \mathbf 1 = 0$, $\rho/\Delta t > 0$), so $\phi = 0$. Then
$D_\alpha\Pi u^* = 0$, the corrections vanish, $u = u^*$, and the momentum equation loses its
$\Delta t$ terms. $\square$

So *if* the iteration converged jointly, the steady state would be unique (up to pressure
constants) and $\Delta t$-independent (design requirement C2). Both predictions fail in
practice; the next two sections say why, in two separate layers.

## 3. Layer 1: linear instability of the rotational update [MEASURED + literature]

On the $\phi=0$ background the step map is linear; its spectrum decides convergence.

- [MEASURED] On a dense random sphere bed ($\varphi = 0.60$, $R = 12$ cells/radius, $192^3$),
  the gauge-exact iteration converges pre-asymptotically and then grows a nearly mean-free
  near-wall mode **exponentially**: doubling time $\approx 750$ steps at $\Delta t = 60$,
  $\approx 82$ steps at $\Delta t = 600$ (rate $\propto \Delta t$); at $\Delta t{=}600$ the
  defect grows four decades ($m_1 \to 12$) while $\langle u\rangle$ moves only $0.25\%$ — the
  mode is invisible to permeability-based convergence monitors.
  (Artifacts: `scratchpad/longmarch_R12*.log`, notes `collocated_stall_notes.md`.)
- [MEASURED] The staggered twin (same rotational update, same bed) is $\Delta t$-flat to
  8 digits. The instability needs $R \gtrsim 12$ resolution structure; $R=6$ is stable.
- Literature anchor: Guy & Fogelson, *JCP* 2005, analyze exactly this family (cell-centered
  approximate projection; PM I/PM II updates; boundary pressure-gradient variants). Their
  result: on MAC grids all variants are stable; on cell-centered grids PM II is destabilized by
  precisely the quadratic one-sided boundary gradient ("gradient 2") that our $G$ uses at cut
  cells, through high-frequency boundary modes; standard $n$-th order pressure extrapolation
  amplifies the checkerboard mode by $2^n/h$ at the boundary row.
- [MEASURED] Ablations at $R=12$: their "gradient 2a" analogue (extrapolate the gradient, not
  the pressure) slows growth $\times 0.85$ but does **not** stabilize; removing the rotational
  term (PM I, $\chi = 0$) **does** stabilize but loses all pressure relaxation as
  $\Delta t \to \infty$ (gain $\rho/\Delta t \to 0$); a smoothing filter $S$ on the rotational
  term with one-sided wall rows *destabilizes further* (new boundary-row coupling).

**The $\Delta t\to\infty$ structure [PROVED, modulo exact solves].** At $\Delta t = \infty$ the
update is a unit-step Uzawa iteration for the steady Stokes Schur complement: the pressure error
obeys $\delta P^{n+1} = (I - W\,\Sigma)\,\delta P$ with
$\Sigma = D_\alpha \Pi A_u^{-1} G$ (evaluated at $\Delta t = \infty$) and $W$ the weighting of
the rotational term. For the *adjoint* pair ($G = -D^T$) $\Sigma$ is symmetric positive
semidefinite and scalar under-relaxation suffices. Our pair is **not adjoint** ($G \ne -D^T$),
$\Sigma$ is nonsymmetric, and no scalar or diagonal *rescaling* can stabilize an eigenvalue with
$\operatorname{Re}\lambda < 0$ (|1 - w\lambda| > 1 for all $w>0$).

- [MEASURED] Uniform $w = 0.3$: stable at $\Delta t = 60$; marginal at $600$ (rate cut
  $\times 6$, still growing); **blows up at $\Delta t = 10^{20}$** (doubling $\sim 230$ steps).
- [MEASURED] The adjoint ablation (mode 3, wall-aware $(T, T^T)$ pair): **unconditionally stable
  including $\Delta t = 10^{20}$** — but $\approx -12\%$ accurate on the bed (its known defect).
  This confirms non-adjointness as the structural origin. (`lm192_mode3_dt*.log`)

**The wall-banded stabilization (F. Peters) [MEASURED].** Replace the update by
$$ P^{n+1}_i = P^n_i + \Big(\tfrac{\rho}{\Delta t} + w_i\, \tfrac{\mu}{\Delta x^2}\Big)\phi_i
   \;-\; (1 - w_i)\,\mu\,(D_\alpha \Pi u^*)_i, \qquad
   w_i = \begin{cases} w_0 & \text{cell } i \text{ has a solid axis-neighbour}\\ 0 &\text{else.}\end{cases} $$
Bulk rows keep the exact rotational gain ($\mu A_p \phi$ per mode — the correct
Schur-complement gain at every smooth mode as $\Delta t \to \infty$); the deficient boundary
rows trade $(1-w_0)$ of the destabilizing off-diagonal for a *diagonal* gain
$w_0 \mu \phi$ that survives $\Delta t \to \infty$. Fixed-point statement of Prop. 1 is
unchanged for any $w_0 \in (0,1]$ (the bracket stays SPD). With $w_0 = 0.5$, $R=12$:
**stable and settled to 11 digits at $\Delta t = 60$, $600$, and $10^{20}$**
(`lm192_wall05_dt*.log`). This satisfies design requirement C2's stability half at all
$\Delta t$; it does not by itself restore steady-state uniqueness — see Layer 2.

## 4. Layer 2: the invisible subspace and the attractor family

Define the *invisible subspace*
$$ \mathcal K \;=\; \ker G \;\supseteq\; \{\text{fields supported on solid-centered cells}\}
   \,\oplus\, \{\text{constants per fluid component}\}, $$
which is large (dimension $\gtrsim$ number of solid cells) because $G$ never reads
solid-centered values. The constraint side does *not* share this kernel: $A_p$ couples
solid-centered pressures through $\alpha_f > 0$ faces.

**Proposition 2 (stationarity is quotiented by $\mathcal K$).** *For the iteration of §1 (any
of the pressure-update variants of §3), $u^{n+1} = u^n$ for all subsequent steps holds iff the
pressure increment lies in $\mathcal K$:*
$$ \big(\text{diag-blend}\big)\,\phi \;\in\; \mathcal K,
   \qquad \text{diag-blend} = \big(\tfrac{\rho}{\Delta t} I + W \tfrac{\mu}{\Delta x^2}\big)
   + (I - W)\,\mu A_p . $$
*The set of $u$-stationary states is therefore an affine family of dimension
$\dim \mathcal K$, parameterized by which member the transient selects; on it $P$ grows
linearly in time along $\mathcal K$ while $u$, $u_f$, $\phi$ are exactly stationary with
$\phi_\infty \ne 0$ in general.*

*Proof sketch.* $u^{n+1}$ depends on $(u^n, P^n)$ only through $G P^n$ (momentum) and $G\phi$,
$G_f \phi$ (corrections), and $\phi$ depends on $u^*$ only. If the increment
$\Delta P \in \mathcal K$ then $G P^{n+1} = G P^n$, so $u^*, \phi, u^{n+1}$ repeat exactly;
conversely stationary $u$ forces $G \Delta P = 0$. The family is affine because the map is
affine. $\square$

*(Prop. 1 is the special case $\mathcal K = \{0\}$-effective: it implicitly assumed the
increment must vanish, i.e. that no nonzero increment is invisible. For the staggered scheme the
gradient reads every cell the divergence couples, $\ker G_{\rm stag} = \text{constants}$, and
the family degenerates to the harmless uniform gauge drift — Prop. 2 explains staggered
immunity.)*

**Measurements confirming the family [MEASURED].**

1. *Exact, $\Delta t$-labeled, path-independent attractors:* cycling
   $\Delta t: 60 \to 600 \to 6 \to 10^{20} \to 60$ under $w_0 = 0.5$, each segment snaps to its
   own $k$ (3.9883632 / 3.9878386 / 3.9864118 / 3.9877125 $\times 10^{-3}$), and the return to
   $\Delta t = 60$ reproduces 3.9883632e-3 **to 10 digits**. No ratcheting: the reconciliation
   defect $m_1 = \mathrm{rms}|u_f - \Pi u|/\langle u\rangle \approx 1.8\times 10^{-2}$ at every
   $\Delta t$. (`lm192_wall05_cycle.log`)
2. *Pressure runaway on $\mathcal K$:* at the settled $R=8$ state,
   $\max|\Delta P| = 0.217$ per 500 steps with $|P|_{\max}$ grown to $1.30$ — $75\times$ the
   staggered pressure scale — while $u$ is frozen to 9 digits. Staggered control: small uniform
   drift only. (`ladder_R8_wall05.log`, `ladder_R8_stag.log`)
3. *$\phi_\infty \ne 0$:* $m_1 > 0$ persists in every stabilized state (it equals
   $|\Pi G \phi_\infty - G_f \phi_\infty|$ up to masking), and the cell-field constraint
   residual $m_2 \approx 1.9\times 10^{-2}$ while the face field's is $10^{-8}$.

**Corollary (the "plateau").** The permeability bias of the collocated scheme contains a
contribution from the selected family member (through the $A_p$-coupling of the invisible DOFs
at the wall band). It is *not* a truncation-error plateau of the discretization: the previous
"$\sim 0.3\%$ accuracy ceiling" narrative conflated (i) instability contamination (Layer 1) and
(ii) family selection (Layer 2). Same-rung gaps re-measured with the stabilized scheme and deep
convergence: $R=8: -2.53\%$, $R=12: -0.76\%$ (previously reported $-1.43\%$, $-0.50\%$).
$R = 16/24/32$ in progress. [MEASURED]

## 5. Inexact solves [PROVED, brief]

For stationary linear inner solvers (fixed-sweep RB-GS; iteration matrix $T$, $1 \notin
\operatorname{spec} T$), the fixed points of the inexact outer map coincide with the exact ones:
$u = T^m u + (I - T^m)A_u^{-1} b \Rightarrow u = A_u^{-1} b$. PCG with a relative tolerance
leaves a residual that scales with the (vanishing) RHS at the fixed point. Neither changes
Props. 1–2. The measured attractors are therefore properties of the scheme, not of solver
tolerances. (This was also verified empirically: $100\times$ tighter marches move $k$ by
nothing to six digits.)

## 6. Solutions

**S0 (shipped as of flow `9919a86..8a6be43`, opt-in): wall-banded blend.** Cures Layer 1 at all
$\Delta t$ incl. $10^{20}$; leaves Layer 2 (reproducible attractors; residual
$\Delta t$-sensitivity of $k$ about $5\times 10^{-4}$ relative across $\Delta t \in [6, 10^{20}]$,
vs $1.8\times10^{-3}$ before). Suitable as a stopgap default once validated on the ladder.

**S1 (the principled fix): support-consistent gradients.** Make every operator that delivers
pressure to the momentum/corrections read the same DOFs the constraint couples, collapsing
$\mathcal K$ to constants (Prop. 2 then gives uniqueness as in Prop. 1):
- The openness-weighted corrector/predictor of the *embed* line (modes 6/7,
  `projectCorrectCenterOpen`) has exactly this support: face-difference averages weighted by
  $\alpha_f$, reading solid-centered $p$ where $\alpha_f > 0$. Basilisk's embed operates this
  way. The embed modes are currently broken on beds (a preconditioner NaN at $R=8$, never
  root-caused) — **fixing that bug is now the highest-value implementation step**, followed by a
  stability check (Layer-1 analysis applies to the new pair too) and the accuracy ladder.
- Alternatively a *fluid-only aperture operator*: eliminate solid-centered DOFs from $A_p$ by
  folding their $\alpha_f > 0$ connections into wall closures. Larger surgery; changes the
  constraint discretization itself (ghost-projection-adjacent territory).

**S2 (containment, if S1 disappoints): protocol standardization.** The attractors are exact and
path-independent; fixing the march protocol (one $\Delta t$, fixed convergence criterion) gives
reproducibility to 10 digits. C2 is then violated only across protocols, quantified at
$\sim 0.05\%$ in $k$.

## 7. Open questions

1. **Is the (collocated, gauge-exact) scheme second order once Layer 2 is removed?** Unknown —
   and *currently unanswerable*, because every reachable steady state is a family member, not
   the $\phi = 0$ solution of Prop. 1. The clean-ladder series ($-2.53\%, -0.76\%, \ldots$)
   measures the family-selected states; its $R=8\to12$ contraction ($\times 3.3$ on a $1.5\times$
   refinement, formally order $\approx 3$) is *not* interpretable near a possible sign change
   (the documented zero-crossing trap). If S1 collapses the family, the order of the true scheme
   becomes well-posed and measurable; the a-priori constraint-consistency result (defect
   $O(h^{1.2\text{–}1.4})$, net three decades below the bias) suggests the *operators* are not
   the obstruction.
2. Direct spectral verification of the $\operatorname{Re}\lambda < 0$ claim for $\Sigma$
   (Arnoldi on the assembled boundary block) — currently [INFERRED].
3. Whether the ghost projection's half-height plateau is quantitatively predicted by its
   (smaller) invisible set.

## 8. Publication assessment

There is a paper here, with the following novel content: (i) identification of the
**support-inconsistency mechanism** — a collocated cut-cell approximate projection whose
gradient and constraint couple different pressure supports possesses an affine *family* of
$u$-stationary states with runaway invisible pressure, selected by protocol ($\Delta t$), and
masquerading as a grid-convergence plateau; we found no prior statement of this in the
approximate-projection or momentum-interpolation literature (nearest: the free-variable pinning
in binary-openness ghost methods, and the relaxation/time-step dependence results of
Majumdar/Choi for Rhie–Chow, which concern the face-interpolation transient term, not the
support of the gradient); (ii) the manifestation of the Guy–Fogelson cell-centered PM II
instability inside a production cut-cell IBM solver, with growth-rate scaling
$\propto \nu\Delta t/h^2$ measured over four decades of $\Delta t$, and a **wall-banded
diagonal-augmented rotational update** that is stable through $\Delta t = 10^{20}$ with a
one-line implementation; (iii) a methodological arc (same-rung gap instrument, a-priori operator
exoneration, dt-cycling attractor diagnosis, P-drift instrumentation) that is genuinely
reusable. A complete paper wants: the clean ladder (running), one support-consistent scheme
demonstrated (S1), and the Arnoldi verification (Q2). Venue: JCP or J. Comput. Phys.-adjacent
(Computers & Fluids). The refutation catalogue (march protocol, constraint operator, estimator,
wall-band localisation, gradient-2a, filters) belongs in the paper — it is what makes the
mechanism identification credible.

## 9. Measurement index

| fact | value | artifact |
|---|---|---|
| baseline growth, $\Delta t{=}60$, $R{=}12$ | doubling $\sim$750 steps | `longmarch_R12_staleop.log` |
| baseline growth, $\Delta t{=}600$ | doubling $\sim$82 steps; $m_1{\to}12$, $k$ moves 0.25% | `longmarch_R12_dt600.log` |
| staggered control | $\Delta t$-flat to 8 digits (dt 6..600) | `dtsweep_R12_stag_*.json` (peclet-examples) |
| gradient-2a | still exponential, rate $\times 0.85$ | `lm192_2a.log` |
| PM I | stable, no $\Delta t{\to}\infty$ gain | `lm192_pmI.log` (killed at 4000, decisive) |
| uniform $w{=}0.3$ | stable@60, marginal@600, blows up@1e20 | `lm192_w03*.log` |
| wall-banded $w_0{=}0.5$ | stable + 11-digit convergence @60/600/1e20 | `lm192_wall05_dt*.log` |
| adjoint pair (mode 3) | stable @600/1e20, $\approx -12\%$ accuracy | `lm192_mode3_dt*.log` |
| dt-cycling | exact reversible attractors, no draining | `lm192_wall05_cycle.log` |
| P runaway | $\max|\Delta P| = 0.217$/500 steps, $|P|$ 75$\times$ staggered | `ladder_R8_wall05.log` |
| clean gaps | $R8: -2.53\%$, $R12: -0.76\%$ | `ladder_R8_*.log`, `lm192_wall05_dt60.log` |
| flat-wall isolation | E1/E1b identical stag==col; E2 leak $O(h^6)$; $s{=}0.5$ incidence $O(1)$ all solvers | `flatwall_sweep.log` |
| TGV no-solid control | col$-$stag order $+2.00$ exactly | `tgv_control.log` |
