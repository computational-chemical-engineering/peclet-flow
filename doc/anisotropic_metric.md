# Anisotropic cells, single phase — the metric in every discrete operator (Phase 2 design note)

**Status:** design settled 2026-09-06 (Fable); **IMPLEMENTED AND LANDED 2026-09-07** (flow `12cac0f`,
`735fb46`, `6cf870b`, `0d8417b`, `f168436` — see §10 and its final table). Plan: `suite/docs/PHYSICAL_UNITS_PLAN.md` §3.2, §5, §9.4, §9.8. Phase 1 (isotropic
physical domains, flow `1e3d67d`) is the base and is not reopened. Decisions D1–D5 stand.

The two ⚑ design points of §9.4 are decided here: **§5** the coarsening order for aspect ratios in both
geometric multigrids, and **§6** the Robust-Scaled / embedded ghost closures with the anisotropic
index-space normal. **§8** states the acceptance gates with the numbers they must hit. **§9** is the
commit plan. Everything else is the per-operator bookkeeping the implementer works from.

---

## 0. The one idea

The solver keeps computing on the **unit lattice** (plan §3.2 option B). An anisotropic cell changes
**no algorithm**: it changes a handful of per-axis constants at operator assembly and at the API
boundary. Where a kernel today carries `mu_`, `1.0`, or `6.0*mu_` there is now a per-axis
`beta_b = mu' / h_b'^2`, a per-axis pressure weight `w_a = 1 / h_a'^2`, or `2*(beta_x+beta_y+beta_z)`.
The only places that need more than a constant are the ones that read the SDF **as a distance** along
a **normal**: the embedded-boundary closures (§6). Every consumer that is not carried in this phase
must **refuse** an anisotropic domain explicitly (§7) — no silent isotropy assumption survives.

**The exact-1.0 contract (the first gate of every commit).** With `extent=None` the metric is
`h' = (1,1,1)` and every new constant is *exactly* `1.0`. Multiplying or dividing an IEEE-754 double by
`1.0` is the identity, so an operator written as *today's expression times a constant* is bit-identical.
Two things break that identity and are called out where they occur: (i) a per-axis form that changes
the **order** of floating-point operations (e.g. `beta*(s1+s2+…+s6)` versus `bx*(s1+s2)+by*(…)+…`), and
(ii) a normalisation that is only approximately 1 (a `sqrt` of a unit normal). In both cases the kernel
dispatches on a host-side `aniso` flag and runs **today's code literally** when it is false. That is
not a workaround; it is the statement that the isotropic path is the reference and the anisotropic
path must reduce to it.

---

## 1. The metric

### 1.1 Notation

| symbol | meaning | where it lives |
|---|---|---|
| `h_a` | physical cell size on axis `a` = `extent_a / cells_a` | `UnitScales::h[3]` (Phase 1) |
| `hRef` | `min_a h_a` — the reference length; every `d'`, `mu'`, `p'`, `sigma'` is built on it (Phase 1, unchanged) | `UnitScales::hRef` |
| `h_a'` | `h_a / hRef` ≥ 1, **exactly 1.0 on the finest axis** (`x/x == 1` in IEEE-754) | **new** `UnitScales::hp[3]` |
| `w_a` | `1 / h_a'^2` ≤ 1 — the pressure-gradient / Laplacian weight of axis `a` | **new** `UnitScales::w[3]` |
| `beta_b` | `mu' * w_b` — the viscous coefficient of derivative axis `b` | computed at assembly |
| `V'` | `h_x' h_y' h_z'` — the cell volume in `hRef^3` | **new** `UnitScales::vol` |
| `aniso` | `true` iff some `h_a' != 1.0` after the snap of §1.4 | **new** `UnitScales::aniso` |
| `v_a` | index velocity, cells per `tRef` = `u_a * tRef / h_a` (Phase 1 `velToInt(a)`) | fields |
| `m` | the physical unit normal pulled back to index space, `m_a = n_a / h_a'` (§6) | closures |

The four new members go at the **END** of `UnitScales` (the Phase 3 session appends to the same
struct; never reorder). In cell units and on the armed-isotropic path they are `1,1,1 / 1,1,1 / 1 /
false`.

### 1.2 The momentum equation per component (exact)

Substituting `x_a = org_a + h_a xi_a`, `t = tRef t'`, `u_a = (h_a/tRef) v_a`, `rho = rhoRef rho'` into
`rho (du/dt + u·grad u) = -grad p + mu lap u + F` and dividing the `a`-component by
`rhoRef h_a / tRef^2` gives, with **no approximation**,

```
rho' ( dv_a/dt' + sum_b v_b d v_a/d xi_b )
    = - w_a  dp'/d xi_a  +  sum_b beta_b  d^2 v_a / d xi_b^2  +  F_a'          (M)

beta_b = mu' w_b,   mu' = mu tRef/(rhoRef hRef^2),   p' = p tRef^2/(rhoRef hRef^2),
F_a'   = F_a tRef^2/(rhoRef h_a)          (Phase 1's forceToInt(a) — already per axis)
```

Term by term: the time derivative and the advection are **untouched** (the `h_a` of the component
cancels against the `1/h_b` of the derivative and the `h_b` of the advecting velocity), the pressure
gradient carries `w_a` (`hRef^2/h_a^2`), each second derivative carries `beta_b`, and the body force
and every boundary velocity are already per-axis in Phase 1. Continuity is the unit-lattice
`sum_a d v_a / d xi_a = 0`, unchanged; the open-flux divergence `divergOpen` is unchanged (a face flux
`o_f u_f A_f / V` gives `1/h_a` per axis, which the velocity conversion removes).

**Consequence for the operator storage (SCALING_ISSUES #1 stays solved):** `beta_b ≤ mu'` and
`w_a ≤ 1`, so every stored float coefficient is bounded by the isotropic one; the finest axis carries
exactly the Phase 1 numbers.

### 1.3 The projection (incremental-rotational, cut-cell)

With `phi = dt' p_inc' / rho'` as today:

```
L_w phi  :=  sum_a w_a [ o_{a+} (phi_{+a} - phi) - o_{a-} (phi - phi_{-a}) ]  =  div_o(v*)      (P1)
v_a      <-  v_a - w_a (phi - phi_{-a})                     on every open face of axis a      (P2)
P'       <-  P' + (rho'/dt') phi - mu' div(v*)               (unchanged — mu' carries hRef^2)   (P3)
```

(P1) is exactly what `CutcellMG::setOpenness(ox,oy,oz, idx2,idy2,idz2)` already assembles with
`idx2 = w_x` etc. (its three call sites pass `1.0,1.0,1.0` today), and the coarse levels already form
`w_a / cfac_a^2` (`idx2*sx`). (P2) is `projectCorrect` and its siblings with a per-axis weight. The
predictor `-grad P^n` in `buildRhs` carries the same `w_a`. Variable density, porous and drag
coefficients (`c_f = o_f rho0/rho_f`, `o_f eps_f`, …) ride the same rails unchanged — the weight
multiplies the coefficient field inside `setOpenness` and the same `w_a` multiplies the correction.

### 1.4 The snap (what the relaxed assert becomes)

`setPhysicalDomain` today refuses `|h_a - h_0| > 1e-12 |h_0|`. Phase 2 replaces the throw by a
**snap**: if all three spacings agree to `1e-12` relative, set `h_a := h_0` for all `a` (so
`h' = (1,1,1)` and `aniso = false` **exactly**, even when `extent_a/cells_a` differ in the last ulp
across axes, as `(N_a * H)/N_a` can). Otherwise `hRef = min_a h_a`, `hp[a] = h_a/hRef`,
`w[a] = 1/(hp[a]*hp[a])`, `vol = hp[0]*hp[1]*hp[2]`, `aniso = true`. The snap is what keeps the Phase 1
gates (`units_identity`, `units_scale_invariance`) on **literally** the Phase 1 arithmetic. The
configurations the relaxed assert admits are listed in §7 and named in the commit that relaxes it.

---

## 2. Momentum — every site, with its constant (U10)

`grep -n 'beta = mu_' src/flow_ibm.hpp` gives the four fold sites; `beta = backflowBeta_` is not one of
them (§4.4).

| site | today | Phase 2 | reduction |
|---|---|---|---|
| `ibmBuildDiffusion` (`cut_cell_ibm.hpp`), called from `rebuildStencils()` and `buildAdvStencil()` | `AC = idiag + 6*beta`, off `= -beta` | `AC = idiag + 2*((bx+by)+bz)`, `AW=AE=-bx`, `AS=AN=-by`, `AB=AT=-bz` with `b_a = mu'*w[a]` | `bx=by=bz=mu`: `(mu+mu)` exact, `+mu` rounds once to `round(3mu)`, `*2` exact `= round(6mu) = 6.0*mu` — **bit-identical**, keep this association order |
| `ibmBuildDiffusionVar` + `FaceProps` (`face_props.hpp`) | `b_face = fp.beta(i,j)` | `b_face = fp.beta(i,j) * w[axis of the face]`; `AC = idiag(i) + sum` unchanged | `*1.0` exact |
| `smoothComp()` domain-BC const-coeff path (`~:5750`) | `beta = mu_, Ac = rho_/dt_ + 6.0*mu_` | `Ac = idiag + 2*((bx+by)+bz)`; `diffSmoothColor`/`diffSmoothColorDu`/`constCoeffResidual` take `(bx,by,bz)` | the smoother's `beta*(s1+…+s6)` becomes `bx*(sW+sE)+by*(sS+sN)+bz*(sB+sT)` — **different rounding order** → keep the legacy kernel under `!aniso` (template bool), aniso kernel otherwise |
| `setupBcDiffusion()` (`~:6074`) | `dval = ±beta`, `bval = 2*beta*u_wall` | `dval = ±b_a` for the face on axis `a`, `bval = 2*b_a*bcVel_[face][c]` | `b_a = mu*1.0` |
| `buildAdvStencil` / `buildAdvStencilVar` FOU part (`Grid::fou_operator`, `fouw = rho_`) | index velocities | **unchanged** — advection is index-native (§1.2) | — |
| Koren / SOU / FOU explicit RHS, deferred correction `rho*(aF - aK)`, the `comp` term | index-native | **unchanged** | — |
| `buildRhs` predictor `gp = P(i) - P(i - strd)` (staggered) and `0.5*(P(i+s)-P(i-s))` (collocated) | weight 1 | `w[c] * (…)` | `*1.0` |
| `addDragDiagonal` (CFD-DEM implicit drag) | per-cell diagonal | **unchanged** (a rate, converted at the registry boundary in Phase 1) | — |
| Robust-Scaled bake `ibmModifyStencil` (`cut_cell_ibm.hpp`) | modifies the *original* off-diagonals `orig[k]` | **unchanged**: `orig[k]` already is `-b_axis(k)`; `K/M/X/Nbc/D/R` are functions of the crossing fraction `theta` along the axis, which is unit-free | — |
| `ibmFillEntry` slip length `lamAxis[a]` | `s*lambda'/|n_a|` | `s*lambda'/(h_a'*|n_a|)` with `n` the **physical** normal (§6.3) | `hp = 1.0f` exact |
| velocity MG level 0 (`setFineStencil`) | the fine stencil | carries the fold automatically | — |
| velocity MG coarse (`setStaircase`, `setStaircaseBc`, `buildUpwindCoarse`, `setDomainBcOp`) | `b_a^L = nu_dt / cfac_a^2` | `b_a^L = nu_dt * w[a] / cfac_a^2` (one new `VelocityMG::setMetric(w)` before `init`; `buildConstAniso` and `buildAdvCoarse` already take per-axis `bx,by,bz`) | `*1.0` |
| velocity MG upwind coarse `s_a = 1/cfac_a` | index velocity restricted | **unchanged** | — |
| velocity MG transfers `restrictAvg` / `prolongAdd` / `prolongMasked` / `fillProlongBcGhosts` | per-axis `ratio` | **unchanged** (already semi-coarsening-aware for odd axes) | — |
| staircase classification `ibmVolfrac` (`mac_ibm.hpp`) | `theta = clamp(0.5 + d')` | `theta = clamp(0.5 + d' * |m|)`, `|m| = sqrt(sum_a n_a^2 w_a)` (§6.4), **rate-only**; literal legacy expression under `!aniso` (the isotropic `|m|` is a `sqrt`, not exactly 1) | dispatch |
| scalar transport `scalarBuildDiffusionOpen(D, …)` (`scalar_transport.hpp`) | `t_f = D*o_f` | `t_f = D*w[axis]*o_f` per face axis; advection unchanged | `*1.0` |
| momentum residual stop `max|b - A u| / max|b|` | ratio | **unchanged** | — |

**Wall velocity, inflow profiles, body force, drag rate:** already per-axis in Phase 1
(`velToInt(a)`, `forceToInt(a)`); nothing to do.

---

## 3. Pressure — every site (U11)

| site | today | Phase 2 |
|---|---|---|
| `mg_.setOpenness(…, 1.0, 1.0, 1.0)` at `flow_ibm.hpp` ~:2356 (geometry), ~:6299 (variable density), ~:6334 (porous) | unit weights | `w[0], w[1], w[2]` |
| `buildCutcellOp(…, gfx,gfy,gfz)` fine and coarse (`idx2*sx`) | already per axis | **unchanged** |
| exact matrix-free level 0 (`gfx_`…) | stored from `setOpenness` | **unchanged** |
| `coarsenOpenAvg`, `restrictAvg`, `prolongAdd`, `applyNeumannGhost`, `applyOutflowGhost`, the RB-GS smoother, deflation, MG-PCG / FCG / Chebyshev drivers, the agglomerated / GraphAMG bottom | operate on the assembled operator | **unchanged** (Chebyshev bounds are re-estimated on every coefficient change already: `chebBoundsSet_ = false`) |
| `projectCorrect` (`mac_pressure.hpp`) | `u -= phi - phi_-x` … | `u -= w_x*(…)`, `v -= w_y*(…)`, `w -= w_z*(…)` |
| `projectCorrectVar`, `projectCorrectVarHarm`, `projectCorrectPorousCons`, `projectCorrectPorousDrag` | `rho0/rho_f * (…)` etc. | the same `w_a` factor per axis, multiplying **outside** the existing expression (`w_a * (rho0/rho_f * (…))`) so the `1.0` reduction is exact |
| `bcCorrectOutflow`, `bcCorrectOutflowVar` (`mac_bc.hpp`) | high-side outflow face | `f(bf) -= w_a * (…)` |
| collocated face field (`projectCorrect(uf_,vf_,wf_,…)`) | shared kernel | carried by the shared kernel |
| collocated cell corrections `projectCorrectCenter`, `projectCorrectCenterOpen`, and the gradients `centerGradOpen`, `centerGradAperture`, `centerGradApertureScaled`, `centerGradOpenCapped`, `gpCenterGrad`, `transposeGradWallAware` (`mac_approx_projection.hpp`, `gauge_exact_gradient.hpp`) | index gradients | every kernel gains a `double w` (per-axis) applied to its output; the `-grad P^n` predictor uses the same `w[c]`. These read the SDF only by **sign** (verified: `gauge_exact_gradient.hpp`, `star_elimination.hpp`, `ghost_projection.hpp` use `>= 0` tests and `theta` ratios) — no metric inside |
| rotational update `P += (rho/dt) phi - mu div(u*)` and the rotational filter `filterCellField` (sign-only SDF reads) | — | **unchanged** |
| `maxOpenDivergence`, `div` reports | index divergence → `divToPhys` | **unchanged** |
| pressure Dirichlet ghost `bcZeroPressureGhost`, Neumann ghost | values | **unchanged** |

**Chebyshev / PCG spectra.** The weighted operator's spectrum is bounded by `4 sum_a w_a ≤ 12`;
`estimateEigenvalues` handles it, nothing to tune.

---

## 4. Domain boundaries, the outflow census, backflow, forces

### 4.1 Velocity BCs (`mac_bc.hpp`)
`bcVelocityComp`, `bcSlipComp`, `bcMirrorGhost`, `bcOutflowComp`, `bcNeumannGhost`: values and
profiles are already index velocities per axis (Phase 1 `resampleBcProfile`) — **unchanged**. The
implicit folds (`bcDiffusionFold`) take `±b_a` of the **face axis** (§2). Free-slip tangential fold
`-b_a`; outflow zero-gradient fold `-b_a`; the velocity MG's `boundaryFold` per level uses `b_a^L`.

### 4.2 Outflow census (`outflowBackflow()`, `~:2624`)
`fraction` and `reversed/total` count faces — unit-free. `maxReverse` must be reported through
`velToPhys(a)` of the face axis. `energyInflux` sums `rho' back * 0.5*(back^2 + tb^2 + tc^2)`, which
mixes three components with three different velocity scales; the physical kinetic-energy influx per
unit area is `rho |u_n| 0.5 |u|^2`, so inside the reduce use `k_a = velToPhys(a)` etc.:
`rhoRef*rho' * (k_a back) * 0.5*((k_a back)^2 + (k_b tb)^2 + (k_c tc)^2)`. In cell units every `k` is
`1.0`. (Phase 1 left this diagnostic in index units; this is the one place Phase 2 completes it because
the anisotropic mix cannot be converted after the fact.)

### 4.3 Backflow stabilisation (`applyBackflowStab`, `~:5477`) — **unchanged, with proof**
The Bazilevs term is a surface integral; per unit volume of the outlet-adjacent cell it is
`beta rho |u_n| u_a / h_a`. Divide by the component-`a` normalisation `rhoRef h_a/tRef^2` and insert
`u = (h/tRef) v`: `beta rho' |v_n| v_a` — the `1/h_a` of surface-over-volume cancels against the
velocity conversion. The code's `AC += beta*rho*back` with `back` an index velocity is already the
anisotropic form.

### 4.4 Hydrodynamic force and torque
*Traction integral* `hydroForceTorque()` (`~:3994`, diagnostic): with `A_a = W_a V'/h_a'` the physical
fragment area vector in `hRef^2` (`W_a = o_{a-} - o_{a+}`), `gu[a][b] = ½(v_a(i+e_b) - v_a(i-e_b))`, and
the total-force scale `forceTotalToPhys = rhoRef hRef^4/tRef^2` unchanged:

```
dFp_a = p' * W_a * V'/h_a'
dFv_a = - mu' * sum_b W_b * (V'/h_b') * [ (h_a'/h_b') gu[a][b] + (h_b'/h_a') gu[b][a] ]
```

(the `h_a'/h_b'` pair is the physical strain rate `du_a/dx_b + du_b/dx_a` written in index velocities).
The lever arm `r = rp * sm.dToInt` is a physical displacement in `hRef` units on every axis already —
unchanged; `torqueToPhys` unchanged.

*Reaction force* `hydroForceTorqueReaction()` (`~:4203`, the CFD-DEM source): the momentum row of
component `a` is a force density in the component-`a` normalisation, so the body force is
`F_a = - sum_owner R_a * h_a' * V'` in `forceTotalToPhys` units (the isotropic `h_a' V' = 1`).

*The v3 wall-torque term* is a **traction**, not a momentum row, so it takes the metric on its AREA
VECTOR and not on its force component: `F' = mu' (A' x Omega')` with `A'_b = a_b * V'/h_b'` — the
same physical area vector in `hRef^2` the traction paragraph above forms — and `Omega' = Omega tRef`,
with the physical lever arm. (Corrected 2026-09-07; the first version of this sentence said "the same
`h_a'` on its force factor", which is wrong: a wall patch with `n dA = (A_x, 0, 0)` and
`Omega = Omega_z e_z` produces a force in **y**, so the per-axis factor belongs to the area component
the cross product consumes, not to the force component — the two agree only when `h_y' = 1/h_x'`.
`doc/units_escalation.md` E3 carries the derivation; commit C4b implements it.)

### 4.5 Geometry inputs (unchanged unless listed)
`set_solid`: `d' = d/hRef` (Phase 1; a **distance** on every axis, not a per-axis scaling — the
crossing fractions `theta = sdf_c/(sdf_c - sdf_n)` are invariant under any positive scaling of the
SDF, so the sampled cut-cell path needs no metric at all). `SceneMap`: already per-axis
(`b[a] = h_a`). `setExactCrossingsFromScene`: bisection along the **physical** one-cell segment per
axis — unchanged. `buildOpenness(…, dx,dy,dz)` at `~:2010` is handed `1.0,1.0,1.0` today and **must
be handed `hp[0],hp[1],hp[2]`**: `ccFractionCore` (the shipped order-1 aperture model) computes the
gradient `(s+ - s-)/(2 dx)` and the face extent across the normal `|n_b| dy + |n_c| dz` — it was
written for a metric and has been running with the identity. The marching-squares path
(`ccFaceOpenMS`, `order = 2`) is a linear-simplex area fraction on the face: unit-free, unchanged.
`cellFluidFraction` (64-point sign sampling), `ibmSolidMask`, `ibmCleanFluidMask`, `ibmIsCut`: signs
only, unchanged.

---

## 5. ⚑ A — coarsening order for aspect ratios (both geometric multigrids)

### 5.1 The rule

Let `cfac_a^L` be the accumulated coarsening factor of axis `a` at level `L` (already tracked as
`Level::cfac`), and the **level spacing** `H_a^L = h_a' * cfac_a^L`. Let `C^L` be the set of axes that
*can* coarsen at level `L` by today's tests: `can(gdim_a)` (even and `≥ 4`) and, under MPI,
`evenOn(dec, a)` (every rank's block even on that axis). Then

```
aniso == false :  ratio_a^L = 2   iff   a ∈ C^L                                   (today's rule, verbatim)
aniso == true  :  ratio_a^L = 2   iff   a ∈ C^L   and   H_a^L < theta * min_{b ∈ C^L} H_b^L,   theta = 2
                  ratio_a^L = 1   otherwise.
```

**Always coarsen the finest coarsenable axis; defer an axis while it is already at least `theta`
times coarser than the finest one.** `theta` is read once from `PECLET_FLOW_MG_ASPECT` (default
`2.0`) — a measurement knob for the gate of §8.5, not a user setting. The rule is **engaged only on
an anisotropic domain**: on an isotropic one the level table is today's by construction, including
after a telescoping merge, where an axis that was blocked for some levels re-enters `C` with a
smaller `cfac` than its neighbours (an *operator* anisotropy on isotropic cells — see §5.4).

### 5.2 Why this rule, and why `theta = 2`

- **The isotropic hierarchy is reproduced bit for bit.** With `h' = (1,1,1)` every `H_a^L` in `C^L` is
  equal (axes leave `C` when they turn odd, and the survivors coarsen in lockstep), so `1 < 2` and every
  coarsenable axis coarsens, exactly today's decision. The existing semi-coarsening of odd axes
  (`DECOMPOSITION_AND_MULTIGRID.md` §1) is untouched: an axis that cannot coarsen is simply not in `C`,
  and the rule never waits for it (waiting would freeze the hierarchy on a blocked fine axis; the fix
  for *that* anisotropy is line relaxation, open item 11 of that document, out of scope).
- **The spread halves per level until it is below 2, then stays there.** Let
  `s^L = max_C H / min_C H`. Axes with `H < 2 H_min` double into `[2 H_min, 4 H_min)`, the others stay
  in `[2 H_min, s H_min]`; the new minimum is `2 H_min`, so `s^{L+1} = max(s^L/2, < 2)`. On the gate
  grid `(dx, 0.5dx, 2dx)`: `L0 (1, ½, 2)` → coarsen `y` → `L1 (1, 1, 2)` → coarsen `x, y` →
  `L2 (2, 2, 2)` → full coarsening from there. Every level's coefficient ratio is `≤ 4`, and after two
  levels it is 1.
- **Why not coarsen everything (today's rule as-is):** the spread would then *persist* on every level,
  `(1,½,2) → (2,1,4) → (4,2,8)`; the coarse operators stay 4:1 anisotropic in coefficient, and a point
  red-black smoother does not damp the error components that are smooth along the strongly coupled
  axis only — the V-cycle rate degrades with the spread and the semi-coarsened hierarchy is the textbook
  remedy (coarsen along the strong coupling, i.e. the finest axis). Cost of the remedy on the gate
  grid: one extra level with `½` the fine cells (V-cycle work `1 + ½ + ⅛ + …` ≈ `1.64` vs `1.14`,
  +44 %), against a rate that does not depend on the aspect ratio at all.
- **Why `theta = 2` and not `sqrt 2`:** both bound the per-level spread by 2 (the argument above works
  for any `theta ∈ (1, 2]`); `sqrt 2` additionally inserts a semi-coarsened level whenever the spread is
  in `(sqrt 2, 2)`, buying a coarse-level aspect of `≤ 1.41` instead of `≤ 2` at the price of an extra
  level with 2–4× the cells of a fully coarsened one. On the gate grid the two thresholds give the
  **same** hierarchy, so the gate cannot separate them; `theta = 2` is the cheaper default and
  `PECLET_FLOW_MG_ASPECT=1.4142` is one run away if a stretched production case ever shows the
  iteration count drifting with aspect ratio. Record that run in this note if it is made.
- **Levels.** `nLevels_` (default 4, `set_pressure_multigrid(on, levels)`) keeps its meaning (a count
  of levels); the deferred axes consume levels, so a stretched hierarchy bottoms out on a finer grid
  for the same `levels`. The auto agglomerated / GraphAMG bottom absorbs that; the gate of §8.5 runs at
  `levels = 6`.

### 5.3 Where it enters

One helper, shared by both hierarchies (put it in `mac_cutcell_mg.hpp`):

```cpp
// H[a] = hp[a]*cfac[a]; canA[a] = today's per-axis coarsenability. Returns the per-axis ratio.
static C3 mgChooseRatio(const double H[3], const bool canA[3], double theta);
```

- `CutcellMG::init` (single-rank loop, the `can(inner.x)` block) and `CutcellMG::initMpi` (the
  `can(gs.*) && evenOn` block). The telescoping trigger keeps its meaning: `blocked` is "an axis that
  *can* coarsen globally but is not even on every rank"; an axis deferred by the aspect rule is **not**
  blocked and must not trigger a merge. `H` needs `hp`, so **`CutcellMG::setMetric(hp)` is called before
  `init`/`initMpi`** (`setSolidDevice` ~:2350 and the MPI init path), not with `setOpenness`, which
  comes after the level table exists.
- `VelocityMG::init` / `initMpi` (`mac_velocity_mg.hpp` ~:330–470): the same helper, the same `hp`, so
  the two hierarchies share one level table and `dec.coarsened(ratio)` is the same on both.
- `coarsenAlignment`, `refineFactor`, `decomposition()` (coarse-first): **unchanged**. The aspect rule
  only *defers* coarsenings, so by any level an axis has coarsened at most as often as under today's
  rule: today's alignment is a valid over-alignment (harmless, as the code already notes), and the
  coarse-first candidate grid (fully coarsened `L-1` times) remains the conservative one for the
  imbalance search. Under MPI every rank computes `H` from the same doubles and reaches the same
  `ratio` without communication.
- `mgDebugLevel()` prints the level table; the gate asserts it through a new
  `CutcellMG::levelRatios()` accessor.

### 5.4 Noted, not done: the same rule on isotropic cells after a telescoping merge

With `h' = 1` the rule reduces to today's only because it is gated on `aniso`. Without the gate it
would, after a telescoping merge unblocks an axis at `cfac = 1` while its neighbours sit at `cfac = 4`,
defer the neighbours until the unblocked axis catches up — `(1,4,4) → (2,4,4) → (4,4,4)` instead of
today's `(2,8,8)`. That is arguably the better hierarchy (the coarse operator is 16:1 anisotropic in
coefficient today), but it changes bits at `extent=None` on every telescoped multi-rank case, which
is exactly what Phase 2 must not do. It is a separate, measured change for the scaling campaign
(`docs/SCALING_ISSUES.md`); the helper and the knob make it a one-line experiment.

---

## 6. ⚑ B — the ghost closures with the anisotropic index-space normal

### 6.1 The normal, and the direction the closures actually need

The sampled SDF on the lattice is `d'(xi) = phi(org + h∘xi)/hRef`, a **distance in hRef units**, not a
signed distance of the index space. Its index-space central difference is

```
g_a = ½ (d'(i+e_a) - d'(i-e_a))  ≈  d d'/d xi_a  =  h_a' * n_a        (n = the physical unit normal)
```

so the code's `nx = ½(sdf(i+sx) - sdf(i-sx))`, renormalised, is *not* `n` when the cells are
anisotropic — it is the covariant gradient, tilted toward the coarse axes. Two different vectors are
needed, and both are per-axis rescalings of `g`:

```
physical unit normal        n_a = (g_a / h_a') / | g / h' |             (plan §3.2's "h_a dphi/dx_a renormalised")
index direction of n        m_a = n_a / h_a'                              (the contravariant pull-back)
```

`m` is the vector the closures march along: moving `t` along `m` in index space moves `t` **in
physical hRef units** along the true normal (`|h'∘m| = |n| = 1`). The foot point of a cell centre
(the physical point `x - d n`, pulled back) is

```
xi* = xi - d' * m           (per axis: xi_a - d' n_a / h_a' — the plan's "d·n_a/h_a")
```

With `h' = 1`: `n = g/|g|` and `m = n`, and the arithmetic is literally today's (`g_a/1.0`, `n_a/1.0`)
— bit-identical without a dispatch.

### 6.2 Collocated FV closure (`fvViscousApply`, `mac_approx_projection.hpp` ~:317–:380)

```
L_FV(U)_i = idt cs_i U_i + mu' [ sum_a w_a ( o_{a-}(U_i - U_{-a}) + o_{a+}(U_i - U_{+a}) )  -  sum_a w_a W_a g_a(xi*) ]
```

- face terms: `mu A_f (U_i - U_nbr)/d_f / V` with `A_f = V'/h_a'`, `d_f = h_a'` → `w_a` per axis;
- wall drag: `mu sum_a A_a dU/dx_a / V` with `A_a = W_a V'/h_a'` and `dU/dx_a = (1/h_a') dU/dxi_a` → the
  same `w_a`; `g_a(xi*) = sg (2 u1 - ½ u2)` is the one-sided index derivative at the foot point,
  sampled at `xi* ± sigma e_a` exactly as today, with `xi*` from §6.1 (and the same clamp into the
  sampleable block).

Interior cells (`cs = 1, o_f = 1, W = 0`) reduce to `idt U - sum_a beta_a lap_a U`, the same operator
as the IBM matrix `M`, so the defect correction still vanishes there.

### 6.3 Embedded true-normal closure (`embedViscousApply`, `embedDirichletGradient`, ~:475–:520)

```
wall = mu' * |A'| / V' * dU/dn,      |A'|/V' = sqrt( sum_a W_a^2 w_a )
```

`embedDirichletGradient` receives `m` (not `n`) and the foot point `-d' m`:
- dominant axis `da = argmax_a |m_a|` — it is the **index** slope `m_t/m_da` that must stay `≤ 1` for
  the transverse offsets to remain within the `±1` stencil the Basilisk `j/k` rounding assumes;
- image plane `l` at axial offset `io`: `t_l = (io - p_da)/m_da` is the **physical** (hRef) distance
  from the wall point, transverse offsets `p_t + t_l m_t`; the two-point quadratic fit in `(t_0, t_1)`
  returns `dU/dn` per `hRef` directly;
- the degenerate 1-point estimate `U/d0` with `d0 = |p_da/m_da|` and the same `0.5` floor (the floor
  bounds the explicit lagged gain; it is a physical distance now, which is the right thing to floor).

### 6.4 The Robust-Scaled staggered closure (`cut_cell_ibm.hpp`, `mac_ibm.hpp`)

Nothing in `ibmFillEntry` reads a distance: `theta`, `D`, `K`, `M`, `X`, `Nbc`, `R`, `D_rescale` are
functions of crossing fractions along an axis. **The metric enters only through the Navier slip
length** `lamAxis[a] = s * lambda' / (h_a' |n_a|)`: the Robin condition along axis `a` reduces with
`d/dn = (1/(h_a |n_a|)) d/dxi_a`, so the slip length measured in cells along the axis is the physical
one divided by the physical cell size along that axis and by `|n_a|` of the **physical** normal
(`n` from the 7-sample gradient as in §6.1, in float). `s = 1 - n_c^2` uses the same `n`. The `1e-3`
floor on `|n_a|` stays.

`ibmVolfrac` (velocity-MG staircase classification, **rate only**): `theta = clamp(0.5 + d' |m|)`
with `|m| = sqrt(sum_a n_a^2 w_a)` the index distance per unit physical distance along the normal —
under `!aniso` the literal `0.5 + d'` (a `sqrt` of a unit vector is not exactly 1).

### 6.5 What this decision deliberately does not do

- It does not rotate the closure into a metric-orthogonal frame or change the Basilisk stencil
  topology: same taps, same fallbacks, per-axis constants only (§9.6 rule 2 of the plan).
- It does not touch the ghost projection (`ghost_projection.hpp`) or the gauge-exact gradient: they
  classify by sign and interpolate by `theta`; the metric reaches them as the `w_a` on their output.
- VoF (the PLIC normal, the height-function columns, the contact-line normal) is Phase 3's, on the
  same `n`/`m` definitions; the Phase 3 note should cite §6.1.

---

## 7. What Phase 2 admits and what it refuses

After the commit that relaxes the assert, an anisotropic `extent` is **admitted** for: the staggered
`Solver`, constant or variable properties, variable density, porous continuity with or without implicit
drag, immersed cut-cell geometry (sampled or scene), every domain-BC type, every pressure driver and
bottom, the velocity MG, scalar transport, MPI (`init_mpi`, telescoping, coarse-first), and — after the
⚑ B commit — the collocated policy and its face-interpolation modes.

**Refused, with a `throw` that prints the three spacings** (one helper `requireIsotropic(const char*
what)` on the solver; the message names the phase that lifts it):

| refused | where the guard goes | lifted by | state after C4 / C4b |
|---|---|---|---|
| `enable_vof` and every VoF entry point | `enableVof()` | Phase 3 | **ADMITTED** since Phase 3 landed (`flow/doc/anisotropic_vof.md`); gate `units_vof_aniso` |
| the AMR module (`Octree(cells, extent)`) | `core/python/amr_bindings.cpp` | Phase 3 | **ADMITTED** since Phase 3 (`core/docs/amr_anisotropic.md`); the octree's cells are boxes, and only the scalar `spacing_from_extent` helper keeps a cubic contract |
| the CFD-DEM coupling driver | `coupling/python/peclet_coupling/driver.py` (still refuses) | NOT a metric item — see the note under this table | **still refused, and deliberately** — the obstacle is the PARTICLE MODEL, not the map |
| the collocated policy, until the ⚑ B commit lands | `Solver<Colocated>::setPhysicalDomain` | this phase, commit C4 | **ADMITTED (C4)** |
| `hydro_force_torque*` until its constants land (§4.4) | those two functions | this phase, commit C4 | **ADMITTED (C4)** |
| the v3 transposed-stress WALL TORQUE of `hydro_force_torque_reaction` (a MOVING instance under cut-cell pressure) | `hydroForceTorqueReaction()`, beside the other v2 scope refusals | `doc/units_escalation.md` **E3**, RESOLVED | refused by C4, **ADMITTED (C4b)** |

The last row is the one thing C4 narrowed and **C4b restored**. §4.4's original sentence about that
term — "the wall-torque term (v3) takes the same `h_a'` on its force factor" — admitted two readings
that differ on an anisotropic grid, so C4 refused the path rather than choose. E3 was decided for
**reading 2** (the metric on the AREA vector, `A'_b = a_b V'/h_b'`, because the term is a TRACTION
and not a momentum row), §4.4 now states that, and C4b implements it and drops the guard. Everything
§7 promised for Phase 2 is therefore admitted; only `enable_vof` and the two consumers that carry
their own guards still refuse — after Phase 3 that is the CFD-DEM coupling driver alone.

**And the coupling driver's refusal is NOT a metric gap — an earlier version of this table said it
was, and that was wrong.** Three specific claims, corrected:

- *"`gmap()` collapses one spacing onto all three axes."* It does not, and there is nothing to
  fix: the driver poses the WHOLE coupling in the solver's INTERNAL units (`driver.py`, "THE UNIT
  LAYER"), so particle positions reach `gmap` already converted to INDEX coordinates. A spacing of
  1 is the *correct* map there, on a cubic mesh and on a box mesh alike, and `_inv_vcell_i` is
  exactly 1.0 for the same reason.
- *"`inv_vcell` is `1/h^3`."* True when written; fixed since (`1/(h_x h_y h_z)`, coupling
  `9f7cd24`). It was a diagnostic, never on the kernel path.
- *"the velocity/force conversions take component 0 of a per-axis vector."* True, and it is the
  one real per-axis item — `velocity_to_internal` is a three-vector on a box mesh and the driver
  reads `[0]`. It is a few lines.

What actually blocks the driver is the **particle model**, and no amount of per-axis bookkeeping
reaches it: an unresolved CFD-DEM particle has ONE radius, and every drag law in the suite
(Stokes, Schiller–Naumann, Ergun, Di Felice, Wen & Yu, Gidaspow, Beetstra, Tang) is a function of
ONE particle Reynolds number `Re_p = rho |u_s| d_p / mu`. Posed on the unit lattice of a box mesh
there is no single length with which to form `d_p` — `hRef` is the finest axis, so a particle
would be one diameter across in `x` and half of one in `y`, and the drag law would silently be
evaluated at the wrong `Re_p`. The same is true of the deposit stencil's support and of the
volume-averaging validity condition. The honest resolution is to pose the particle side in
PHYSICAL units and convert per axis at the grid interface, which is a coupling design change and
not a metric one; until then the guard stands and its message says exactly this.

The one part of the coupling that **is** posed per axis already is the porosity volume FILTER,
because its width is a physical length that has nothing to do with the lattice: `smooth_length`
with `alpha_a = C/h_a^2`, `C = 1/(2 sum_a 1/h_a^2)` (coupling `9f7cd24`, gate
`coupling/tests/test_smoothing_isotropy.py`). It reduces to `alpha = 1/6` on a cubic mesh bitwise.

---

## 8. Acceptance gates — with the numbers

All gates run with `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, fresh `build_a_*` trees, on
`host-openmp` **and** `nvidia-cuda`; byte comparisons on OpenMP.

### 8.1 G0 — bit identity at `extent=None` (the first gate of every commit)
- `tests/kokkos`: all **39** ctests green (the 36 of the plan plus the three `units_*`), OpenMP and CUDA.
- `tests/kokkos_mpi`: every ctest, `np = 1, 2, 4`, OpenMP and CUDA; np = 1 bit-exact to single-rank.
- `tests/regression/sdflow_regression.py` (never `--update`) against `perf_baseline.json`: the recorded
  `zh_sphere` K values `7.29969 / 7.38906 / 7.41617 / 7.43609 / …` at N = 16–64, order `2.29`,
  extrapolated `7.44704` (reference `7.442`), and the recorded pressure iterations per step
  `6 / 7 / 7 / 6 / …` must be **identical**, not within tolerance.
- The five verify scripts (`verify_poiseuille_flow`, `verify_periodic_spheres_sdflow`, channel, bfs,
  lid cavity): `get_u/v/w/p` dumped to `.npz` before and after the commit, `np.array_equal` **True**
  for all four fields in all five.
- `units_identity`, `units_scale_invariance`, `units_vof_sigma`: unchanged, including their printed
  numbers.

### 8.2 G1 — `units_anisotropic_poiseuille` (ctest, `test_units.cpp`, commit C2)
`runChannel` on cells `(16, 40, 8)`, walls exactly on two `y` cell CENTRES (where the second
difference of a quadratic is exact, so the discrete solution IS the analytic parabola and the
comparison measures only the operator), `rho = 1, mu = 0.1, F = 0.01, dt = 50`, 300 steps,
`velTol 1e-14`, `cutcellPressure = false`.

- **Two stretched configurations.**
  **(a) Exactness:** `extent = (16, 10, 16)`, `spacing() == (1, 0.25, 2)`, `hRef = 0.25` on `y` (the
  finest axis is `y`, a non-trivial check of `hRef = min`), `h' = (4, 1, 8)`,
  `w = (1/16, 1, 1/64)`, `mu' = mu tRef/hRef^2 = 80` — all float-representable, so the stored
  operator carries no rounding of its own; walls at `12.5*0.25 = 3.125` and `28.5*0.25 = 7.125`
  (`H = 4`, `u_max = F H^2/(8 mu) = 0.2`):
  `max_j |u(y_j) - F (y_j - ylo)(yhi - y_j)/(2 mu)| / u_max ≤ 1e-9` over every fluid DOF.
  **(b) The §5.3 configuration** `extent = (16, 12, 16)`, `spacing() == (1, 0.3, 2)`,
  `h' = (10/3, 1, 20/3)`, walls at `3.75` / `8.55` (`H = 4.8`, `u_max = 0.288`): `≤ 1e-7`.
  The discretisation is pointwise exact in both; (b) sits at the float operator-storage floor
  (WO-M, `docs/SCALING_ISSUES.md` #1) because `hRef = 0.3` makes `mu' = 55.5…` and
  `AC = 124.61…` unrepresentable — a one-signed multiplicative factor of `1.06e-7` on every row,
  `8.7e-15` under `-DPECLET_FLOW_MREAL_DOUBLE`. Not a metric defect; recorded. (b) is a
  production-SHAPED configuration and is kept as the tripwire for that floor, not as an exactness
  statement.
- The DOFs on the walls are exactly `0`, and `max |v|, max |w| ≤ 1e-12 * u_max`, in both.
- The isotropic control `(16, 16, 16)` at `extent = cells` in the same test: the same bounds, and it
  reports the EXACT identity metric (`hp = w = (1,1,1)`, `vol = 1`, `aniso` false) from the §1.4
  snap. (It is NOT bitwise the `extent=None` run of the same problem and must not be asserted to
  be: `dt = 50` pins `tRef = 50`, so the armed run computes with `dt' = 1` and `mu' = 50 mu` while
  the cell-unit one computes with `dt' = 50` and `mu' = mu` — the same physics, a different internal
  scaling. `units_identity` is the bitwise statement, and it fixes `rho = dt = 1` for that reason.)
- Print `spacing()`, `unit_scales["aniso"]`, and the three `w`.

### 8.3 G2 — `units_anisotropic_sphere` (ctest, `test_units.cpp`) + `scripts/verify_anisotropic_spheres.py` (commit C4)
The regression's Zick & Homsy sphere (`phi = 0.216`, `R = 0.3722 L`, `K_ref = 7.442`, config of
`perf_baseline.json`: `rho 1, mu 0.1, dt 60, F 1e-3`, PCG `rtol 1e-8`) on the cube `L^3` with cells
`(N, 2N, N/2)` → `h = (dx, ½dx, 2dx)`, sampled SDF at the physical cell centres, `cutcellPressure = true`.
- ctest: `N ∈ {16, 24, 32}` stretched and cubic in one binary. Require (i) stretched errors
  `|K_s(N) - K_ref|/K_ref` **strictly decreasing** in N; (ii) `err_s(N) ≤ 4 * err_c(N) + 0.2 %` at every
  N (the `z` axis is twice as coarse; second order allows 4×); (iii) the least-squares order of the
  stretched sequence `p_s ≥ 1.5` (cubic: `2.29`); (iv) the isotropic control reproduces the recorded
  `7.29969 / 7.38906 / 7.41617` bitwise.
- script: the ladder to `N = 64` (stretched `64×128×32`), Richardson-extrapolated `K_s,inf` within
  **2 %** of `7.442` (the regression's `extrap_rel`), `p_s ∈ [1.5, 2.5]`; pressure iterations per step
  on the stretched grid `≤ cubic + 2` at every N (the aspect rule of §5). Record the table in §10.

### 8.4 G3 — `units_anisotropic_tgv` (ctest, `test_units.cpp`, commit C4)
Stokes Taylor–Green on a periodic box `L × L × L_z`, cells `(N, 2N, 4)` → `h = (L/N, L/(2N), …)`,
`k = 2 pi / L`, `nu = mu/rho`, **discretely divergence-free initial field**
`u = a_x cos(kx) sin(ky)`, `v = -a_y sin(kx) cos(ky)` with `a_x sin(k h_x/2)/h_x = a_y sin(k h_y/2)/h_y`
(on a stretched staggered grid the plain TG field is *not* discretely divergence-free — the isotropic
test relies on `h_x = h_y`). The field is an eigenvector of the anisotropic 7-point operator, so the
backward-Euler amplitude ratio per step is exactly

```
r = 1 / (1 + dt nu Lambda),   Lambda = 2(1 - cos k h_x)/h_x^2 + 2(1 - cos k h_y)/h_y^2
```

- `|r_measured - r| / r ≤ 1e-10` (the isotropic reference figure is the "Taylor–Green ~2e-15" of
  `CLAUDE.md`; the ctest bound is set by `velTol 1e-14` and PCG `1e-13`), `max |div_o| ≤ 1e-12`,
  profile preserved to `1e-10`, `phi` identically `≤ 1e-12`.
- With advection on (NS): the amplitude ratio within `5e-3` of `r` (as `sdflow_tg` requires) and
  `max |div| ≤ 1e-9`.
- The isotropic control `(N, N, 4)` at `extent = cells`, `a_x = a_y = 1`, in the same test: identical
  bounds, and bitwise equal to the cell-unit run.

### 8.5 G4 — `cutcellmg_aniso` (ctest; pressure only; commits C2 for the order part, C3 for the rate part)
`CutcellMG` driven directly (as `test_pressure_wallbounded` does), periodic box, cells `(N, 2N, N/2)`
with `setOpenness(…, 1, 4, ¼)`; `levels = 6`, MG-PCG `rtol 1e-10`, 2/2 sweeps.
- **Order (C2):** all-fluid, manufactured `phi = cos(2 pi x/L) cos(2 pi y/L) cos(2 pi z/L)` with the
  continuous RHS: L2 error at `N = 16, 32, 64` of order `≥ 1.95` (the cubic grid gives `2.00`).
- **Level table (C3):** `levelRatios()` on `(N, 2N, N/2)` is `(1,2,1), (2,2,1), (2,2,2), (2,2,2), …`
  and on `(N, N, N)` is unchanged `(2,2,2), …`.
- **Rate (C3):** (a) all-fluid random mean-zero RHS: the residual reduction per V-cycle over cycles
  2–8 `≤ 0.2` (the isotropic `mg` test achieves ~0.1); (b) with the Zick & Homsy sphere openness,
  MG-PCG iterations to `1e-10` on the stretched grid `≤ cubic (same cell count) + 2` at `N = 32` and
  `64`; (c) the same (b) with `PECLET_FLOW_MG_ASPECT=1e9` (i.e. today's full coarsening) must be
  **worse or equal** — the measurement that justifies §5; record all three counts in §10.
- Under MPI (`tests/kokkos_mpi`): the stretched hierarchy at `np = 1, 2, 4` bit-exact across ranks
  for the same problem (the existing MPI pressure gate pattern).

### 8.6 Stop conditions (plan §9.6, restated)
A G0 miss not found in two focused attempts; an isotropic gate that moves; a regression iteration
count that changes; MPI np = 1 not bit-exact; a kernel that would need a different algorithm. Write
`flow/doc/units_escalation.md` and stop.

---

## 9. Work orders → commits (one each, in this order, each pushed only with G0 green)

| commit | work order | scope | gate |
|---|---|---|---|
| **C0** | U9 | this note | — |
| **C1** | U10 | `UnitScales` tail (`hp, w, vol, aniso`), the snap-less `refreshUnitDerived` plumbing, §2 momentum sites incl. domain-BC folds, the `!aniso` legacy dispatch of the const-coeff smoother, `VelocityMG::setMetric`, scalar transport; **assert intact** | G0 + a stencil-level unit test in `test_stencils.cpp` (`ibmBuildDiffusion` / `buildConstAniso` with `w = (1, 4, ¼)` produce `AC = idiag + 2(bx+by+bz)` and the six off-diagonals, and `w = (1,1,1)` is bitwise today's) |
| **C2** | U11 (weights) | §3 pressure weights at the three `setOpenness` sites and every correction / gradient kernel, `buildOpenness(hp)`, §4.2 census, the **snap + relaxed assert** (§1.4, admits the list of §7 minus the collocated policy and the hydro forces), `requireIsotropic` guards | G0, **G1**, G4-order |
| **C3** | U11 (⚑ A) | `mgChooseRatio`, `setMetric` before `init`/`initMpi` in both MGs, `levelRatios()`, `PECLET_FLOW_MG_ASPECT` | G0, **G4** full (incl. MPI) |
| **C4** | U12 (⚑ B) | §6.1–6.4 closures, slip length, `ibmVolfrac`, §4.4 forces; admits the collocated policy and the hydro forces | G0, **G2**, **G3** |
| **C5** | docs | `flow/CLAUDE.md` "Physical domains and units" (anisotropic admitted, what Phase 3 still refuses), `suite/docs/CONVENTIONS.md` §7 one line, `PHYSICAL_UNITS_PLAN.md` §9.4 marked landed with the gate numbers of §10 | — |

Commit messages name the gate and its numbers; trailer per the session brief. Push the submodule
(`git pull --rebase origin main && git push origin HEAD:main`), bump the umbrella pointer last. Phase 3
rebases onto C5.

---

## 10. Measured (filled in by the implementing session, one line per gate, with backend and np)

### C1 (U10, the momentum fold) — flow `12cac0f`

- **G0**, `tests/kokkos` **39/39** host-openmp and **39/39** nvidia-cuda; `tests/kokkos_mpi`
  **103/103** on both backends (68 tests at np = 1, 2, 4); `sdflow_mpi_np1` bit-exact to
  single-rank, `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00`.
- **Regression** (CUDA, never `--update`): PASS, every recorded number `+0.00 %` — zh_sphere
  `K 7.2997 / 7.3891 / 7.4162 / 7.4361 / 7.4404`, order `2.29`, `K_inf 7.447`, pressure iters/step
  `6/7/7/6/6`; random_spheres `0.0064513…0.0062621` at `7/7/7/7`; hollow_rings
  `0.018629…0.017608` at `9/10/9/9`.
- **Five verify scripts** vs a build of `5285aae`, host-openmp: `np.array_equal` TRUE on every
  field (poiseuille 24/24 arrays, periodic_spheres 8/8, channel 4/4, bfs 8/8, lid_cavity 4/4).
- **Stencil unit test** (`test_stencils.cpp`): at `w = (1, 4, 1/4)`,
  `AC = 1.0700000524520874`, `AW/AE = -0.10000000149011612`, `AS/AN = -0.40000000596046448`,
  `AB/AT = -0.02500000037252903`; at `w = (1,1,1)`, `AC = 0.62000000476837158` == the pre-change
  `idiag + 6.0*beta` bitwise, both backends.

### C2 (U11 weights, the pressure metric + the §1.4 snap) — flow `735fb46`

- **G0**, `tests/kokkos` **41/41** host-openmp and **41/41** nvidia-cuda (the 39 of C1 plus
  `cutcellmg_aniso` and `units_anisotropic_poiseuille`); `tests/kokkos_mpi` **103/103**
  host-openmp and **103/103** nvidia-cuda at np = 1, 2, 4; `sdflow_mpi_np1` bit-exact to
  single-rank, `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00`, `div 5.86e-12`.
- **Regression** (CUDA, never `--update`): PASS, every recorded number and every iteration count
  identical to C1's line above (`+0.00 %` throughout).
- **Five verify scripts** vs a build of `12cac0f`, host-openmp, `np.array_equal` TRUE on every
  array: poiseuille **848/848**, periodic_spheres **76/76**, channel **31/31**, bfs **8/8**
  (`x_r/S` 5.26 and 8.16 at an identical 5500 / 16400 steps), lid_cavity **4/4**.
- **Phase 1 units gates unchanged**, printed numbers included — `units_identity` bitwise;
  `units_scale_invariance` u `8.882e-17`, p `0.000e+00`, sphere sampled
  `4.070e-16 / 1.503e-15 / 1.761e-15` with div `6.915e-12` (d `1.850e-18`), scene
  `6.105e-16 / 1.224e-15 / 1.370e-15` (d `0.000e+00`); `units_vof_sigma` Young-Laplace
  `2.500000000e-01`, `capillary_dt 3.989422804e-01`, computed kappa rel `0.000e+00` / `2.202e-16`.
- **G1** (§8.2), identical on host-openmp and nvidia-cuda:

  | configuration | `spacing()` | `w` | max rel \|u − parabola\| | bound |
  |---|---|---|---|---|
  | (a) exactness, extent `(16, 10, 16)` | `(1, 0.25, 2)` | `(1/16, 1, 1/64)` | **1.388e-15** | 1e-9 |
  | (b) tripwire, extent `(16, 12, 16)` | `(1, 0.3, 2)` | `(0.09, 1, 0.0225)` | **3.052e-08** | 1e-7 |
  | (c) isotropic control `(16,16,16)` at extent == cells | `(1, 1, 1)` | `(1, 1, 1)` | **3.701e-15** (CUDA 4.441e-15) | 1e-9 |

  `max|v| = max|w| = 0.000e+00` exactly and the wall/solid `u` rows exactly `0` in all three.
  Row (b)'s error is a single one-signed multiplicative factor — rel err / u = `1.06e-07` on all
  fifteen fluid rows — and the identical source built with `-DPECLET_FLOW_MREAL_DOUBLE` reads
  **8.674e-15**: the WO-M float operator-storage floor, not the metric.
- **G4-order** (§8.5, the order half), `cutcellmg_aniso`, host-openmp (nvidia-cuda identical to the
  digit). L2 error against the exact modal solution of the discrete system; `levels = 6`,
  `rtol 1e-10`, 2/2 sweeps:

  | ladder | N = 16 | N = 32 | N = 64 | LS order |
  |---|---|---|---|---|
  | stretched `(N,2N,N/2)`, `w = (1,4,¼)`, **MG-PCG** | 500 it, r/\|b\| 5.2e-06, 7.997679e-03 | 500, 1.9e-04, 1.990492e-03 | 500, 1.5e-04, 4.789682e-04 | **2.0308** |
  | cubic `(N,N,N)`, `w = (1,1,1)`, MG-PCG | 8 it, 1.0e-11, 4.578780e-03 | 8, 4.2e-11, 1.138076e-03 | 8, 4.9e-11, 2.841076e-04 | **2.0052** |
  | stretched, **FCG** (converged control) | 19 it, 4.5e-11, 7.996594e-03 | 22, 3.9e-11, 1.990660e-03 | 27, 7.5e-11, 4.971282e-04 | **2.0038** |
  | cubic, FCG | 7 it, 4.578780e-03 | 8, 1.138076e-03 | 8, 2.841076e-04 | **2.0052** |

  The FCG rows reproduce the exact discrete solution to every printed digit.
- **The MG-PCG cap is §5's "before" number** (C3's rate gate is measured against it): MG-PCG caps
  at 500 on every stretched rung, a stall (non-monotone around 1e-4…1e-5 relative), and the L2
  error then drifts up to 3.6 % from the exact discrete value at N = 64, which is what inflates
  the stretched order to 2.0308 against FCG's 2.0038. It is not the bottom solve
  (`setAgglomerationMode` 0 / −1 / 1 identical to the digit) and not the depth (`levels` 2, 3, 4,
  5, 6 all cap). A 2×2 ablation at N = 32 isolates it to the COMBINATION: stretched shape with
  `w = (1,1,1)` converges in **8** iterations, cubic shape with `w = (1,4,¼)` in **20**, and only
  stretched + stretched weights stalls — exactly §5.2's prediction for today's full-coarsening
  rule.

### C3 (U11 ⚡ A, the aspect-ratio coarsening rule) — flow, Phase 2 commit C3

**The rule, and where it lives.** One helper,
`CutcellMG::mgChooseRatio(H, canA, aniso, theta)` (§5.1 verbatim: with `aniso` false it returns
today's decision, ratio 2 on every axis with `canA` true; with `aniso` true, ratio 2 iff
`canA[a] && H[a] < theta * min_{b: canA[b]} H[b]`), called from all four level loops —
`CutcellMG::init`, `CutcellMG::initMpi`, `VelocityMG::init`, `VelocityMG::initMpi` — with
`H[a] = hp[a] * cfac[a]` at that level. `CutcellMG::setMetric(hp)` and the extended
`VelocityMG::setMetric(w, hp)` are called BEFORE `init`/`initMpi` at their single call sites in
`flow_ibm.hpp` (`setSolidDevice`, and the velocity-MG build just above it) — trap 5.
`theta` comes from `PECLET_FLOW_MG_ASPECT` (`mgAspectTheta()`, read once, default `2.0`).
`levelRatios()` on both classes, `Solver.pressure_mg_level_ratios()` in Python.
`coarsenAlignment`, `refineFactor`, `decomposition()` and the telescope trigger are UNCHANGED
(§5.3, trap 6): `blocked` still reads `can()`/`evenOn()` alone.

- **G0**, `tests/kokkos` **41/41** host-openmp and **41/41** nvidia-cuda; `tests/kokkos_mpi`
  **106/106** host-openmp and **106/106** nvidia-cuda at np = 1, 2, 4 (the 103 of C2 plus
  `cutcellmg_aniso_mpi` × 3); `sdflow_mpi_np1` bit-exact to single-rank,
  `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00`, `div 5.86e-12`.
- **Regression** (CUDA, never `--update`): PASS, every recorded number `+0.00 %` and every
  iteration count identical to C1/C2 — zh_sphere `K 7.2997 / 7.3891 / 7.4162 / 7.4361 / 7.4404`,
  order `2.29`, `K_inf 7.447`, pressure iters/step `6/7/7/6/6`; random_spheres
  `0.0064513…0.0062621` at `7/7/7/7`; hollow_rings `0.018629…0.017608` at `9/10/9/9`.
- **Five verify scripts** vs a build of `735fb46`, host-openmp, `np.array_equal` TRUE on every
  array: poiseuille **800/800**, periodic_spheres **60/60**, channel **23/23**, bfs **221/221 (x_r/S 5.26 and 8.16 at an identical 5500 / 16400 steps)**,
  lid_cavity **15/15**.
- **Phase 1 + G1 units gates unchanged**, printed numbers included — `units_identity` bitwise;
  `units_scale_invariance` u `8.882e-17`, p `0.000e+00`, sphere sampled
  `4.070e-16 / 1.503e-15 / 1.761e-15` (div `6.915e-12`, d `1.850e-18`), scene
  `6.105e-16 / 1.224e-15 / 1.370e-15` (d `0.000e+00`); `units_vof_sigma` Young-Laplace
  `2.500000000e-01`, `capillary_dt 3.989422804e-01`, kappa rel `0.000e+00` / `2.202e-16`;
  `units_anisotropic_poiseuille` **1.388e-15 / 3.052e-08 / 3.701e-15** on its three rows.

**G4 — the level table** (§8.5), `levels = 6`, `theta = 2`, identical on both backends and
(under MPI) on every rank:

| grid | `levelRatios()` |
|---|---|
| stretched `(N, 2N, N/2)`, `hp = (1, ½, 2)`, N = 16 | `(1,2,1) (2,2,1) (2,2,2) (2,2,2) (1,1,1)` |
| stretched, N = 32 and 64 | `(1,2,1) (2,2,1) (2,2,2) (2,2,2) (2,2,2) (1,1,1)` |
| the SAME grid with no metric set (today's rule), N = 32 | `(2,2,2) (2,2,2) (2,2,2) (2,2,1) (1,2,1) (1,1,1)` |
| cubic `(N, N, N)`, N = 16 / 32 / 64 | `(2,2,2)…(1,1,1)` — **bitwise the no-metric table**, asserted |

`PECLET_FLOW_MG_ASPECT=1e9` reproduces the no-metric table exactly on every rung, which is what
makes the ablation below a clean "today's rule" control.

**G4 — the rate.** All host-openmp; nvidia-cuda identical to the digit.

| measurement | today's rule (`MG_ASPECT=1e9`) | the §5 rule (default) |
|---|---|---|
| (a) V-cycle reduction, worst over cycles 2–8, stretched 32×64×16, random mean-zero RHS, 2/2 sweeps | **0.6920** | **0.1501** (gate ≤ 0.20) |
| (b/c) Z&H sphere `φ = 0.216`, MG-PCG to `1e-10`, stretched N = 32 | **24** iters | **10** (cubic control **9**; gate ≤ cubic + 2 = 11) |
| (b/c) the same at N = 64 | **25** iters | **10** (cubic control **10**; gate ≤ 12) |
| the same, FCG | 23 / 24 | 10 / 10 |
| the C2 ORDER ladder, stretched MG-PCG, N = 16/32/64 | **500 / 500 / 500 CAPPED** (r/\|b\| 5.2e-06 / 1.9e-04 / 1.5e-04), LS order 2.0308 | **7 / 8 / 8** (r/\|b\| 1.3e-11 / 3.0e-11 / 3.1e-11), LS order **2.0038** |

The stretched MG-PCG rows now reproduce the exact discrete solution to every printed digit
(7.996594e-03 / 1.990660e-03 / 4.971282e-04), i.e. they equal the FCG control — **E2 of
`doc/units_escalation.md` is resolved by the coarsening rule alone**, with no change to the
smoother, the post-smoothing colour order or the driver selection.

**The symmetry read-out.** `pr` (`PECLET_FLOW_MG_DEBUG=2`, zero iff the V-cycle is symmetric w.r.t.
the fine operator), median over the FCG iterations at N = 32:

| problem | today's rule | the §5 rule | the cubic control |
|---|---|---|---|
| all-fluid stretched | **3.555e-01** (max 1.08) | **6.818e-02** | 2.814e-02 |
| Z&H sphere stretched | 5.345e-02 | **4.002e-02** | — |

i.e. the stretched hierarchy's asymmetry falls by 5.2× into the neighbourhood of the isotropic
periodic hierarchy's own 0.062 (`flow/CLAUDE.md`, WO-H) — which is the mechanism: today's rule
keeps `(1, ½, 2) → (2, 1, 4) → (4, 2, 8)`, the point RB-GS smoother stops damping along the
strongly coupled axis, and the V-cycle preconditioner stops being symmetric enough for PCG.

**G4 under MPI** (`tests/kokkos_mpi/test_cutcellmg_aniso_mpi.cpp`, the Z&H sphere on
`(32, 64, 16)`, `levels = 6`, MG-PCG `rtol 1e-10`):

| np | level table | iters (single-rank 10) | max\|dist − single-rank\| | telescope, min-extent trigger off |
|---|---|---|---|---|
| 1 | `(1,2,1) (2,2,1) (2,2,2) (2,2,2) (2,2,2) (1,1,1)` | 10 | **0.000e+00 — BIT-EXACT** | none |
| 2 | identical, on every rank | 10 | 1.486e-06 = **1.013e-07** of max\|φ\| = 14.67 | none (trap 6) |
| 4 | identical, on every rank | 10 | 4.351e-06 = **2.965e-07** relative | none (trap 6) |

The np > 1 spread is the MG-PCG's own floor on a CUT-CELL operator (it stops on a RESIDUAL, and a
small-aperture row has a tiny effective eigenvalue): a `-DPECLET_FLOW_MREAL_DOUBLE` build of the
identical source moves it only 4× (3.602e-07 / 1.772e-06 absolute), so it is the stopping rule and
not the float operator storage. **Trap 6 is gated directly**: with `setTelescopeMinExtent(0)` (merge
only when an axis that CAN coarsen is not even on every rank) NO level telescopes, even though the
rule defers x and z for two levels; and a FORCED merge at level 1 reaches the same answer
(1.486e-06 / 4.351e-06) with the aspect rule still owning level 0.

**One place the note's site list did not match the code.** `CutcellMG::predict()` (the
`scripts/check_decomposition.py` pre-flight, `mac_cutcell_mg.hpp` ~:2880) is a FIFTH copy of the
level loop that §5.3 does not list. It takes no metric, so it still models the isotropic rule; a
comment there now says so and points at `levelRatios()`. Threading `hp` through it and through the
Python pre-flight is a follow-up, not part of C3.

### C4 (U12 ⚑ B, the embedded-boundary closures + the hydrodynamic forces) — flow, Phase 2 commit C4

**The closures, and where the metric enters.** §6.1's two vectors are built in three kernels and
nowhere else. From the index-space central difference `g_a = ½(sdf(i+e_a) − sdf(i−e_a)) = h_a' n_a`,
the **physical unit normal** is `n = (g/h')/|g/h'|` and the **index direction** it marches along is
`m_a = n_a/h_a'`; the foot point is `xi* = xi − d' m`. `fvViscousApply` and `embedViscousApply`
(`mac_approx_projection.hpp`) carry `w_a` on both the two-point face flux and the wall term
(`wall_a = w_a W_a sg (2u1 − ½u2)`), `embedViscousApply`'s fragment area becomes
`|A'|/V' = sqrt(Σ_a W_a² w_a)`, and `embedDirichletGradient` is handed `m` and `−d' m` — so its image
distance `t_l = (io − p_da)/m_da`, its dominant axis `argmax|m_a|`, its degenerate `d0 = |p_da/m_da|`
with the 0.5 floor and the derivative it returns are all PHYSICAL, in hRef, with no metric left to
apply outside. `buildIbmOverlay`'s Navier slip length (`mac_ibm.hpp`) is
`lamAxis[a] = lamEff/(h_a' |n_a|)` with the same `n` in float, `s = 1 − n_c²` from it, the 1e-3 floor
kept; `ibmVolfrac` becomes `theta = clamp(0.5 + d' |m|)` with `|m| = sqrt(Σ n_a² w_a)`, under a
host-bool `aniso` dispatch that runs the literal `0.5 + sd` otherwise (a `sqrt` of a unit vector is
not exactly 1 — the C1 smoother pattern). §4.4's forces land in `hydroForceTorque` (`dFp_a` and
`dFv_a` verbatim, both factors applied OUTSIDE the existing expressions in the same association
order) and `hydroForceTorqueReaction` (`F_a = −Σ R_a h_a' V'`). `Solver<Colocated>::setPhysicalDomain`
and both force integrals no longer refuse an anisotropic domain; `enable_vof` still does.

**Three sites the note's list did not name.**

1. **`starCorrectFaces`** (`star_elimination.hpp`, collocated `fluid_only` mode 2) — C2's report
   flagged it. It is a FIX-UP of what `projectCorrect` applied, and `projectCorrect` now applies
   `−w_a (phi_hi − phi_lo)`, so the `±phibar_s` it adds back carries the same `w_a` of the face's own
   axis. Forced, not chosen.
2. **The V8 face acceleration** (`collocated_varrho.hpp`, C2's other flag) — of
   `a_f = dt (f_c + f_b − (P(i) − P(i−s)))/rho_f` only the PRESSURE difference carries a metric
   (`w_a`): `f_c`/`f_b` are already per-axis from Phase 1 and `1/rho_f` is unit-free.
   `faceAccelSubGradPhi` takes the same `w_a` `projectCorrectVar` takes, spelled the same way,
   because bit-for-bit pairing with the face correction is the whole point of that kernel. The cell
   average of the two faces (`applyCellFaceAverage`) is unchanged, as §3 says. `addFaceAccelCsf` is
   untouched: it is VoF-only, and Phase 3 carried it (`flow/doc/anisotropic_vof.md` §5).
3. **The v4 owner-boundary attribution correction** in `hydroForceTorqueReaction` — §4.4 does not
   mention it. It REMOVES a term that is already inside `F_a = −Σ R_a h_a' V'`, so it must carry
   exactly what that term carries there: the momentum row's pressure gradient is
   `w_c (P(i) − P(i−s))` since C1/C2, and `F` multiplies by `h_c' V'`, so the factor on `pi(i)` is
   `w_c h_c' V' = V'/h_c'` — the physical area of the face. Also forced: any other factor breaks the
   pairwise cancellation the correction exists to keep.

**One thing C4 deliberately did NOT do** — see §7's table and `doc/units_escalation.md` **E3**: the
v3 transposed-stress WALL TORQUE of `hydroForceTorqueReaction` (a moving instance under cut-cell
pressure) kept an explicit refusal, because §4.4's one sentence about it admitted two readings that
differ on an anisotropic grid and choosing between them was a design decision. Every C4 gate was
green without it; the term is unreachable without `set_instance_motion`. **Commit C4b closes it**
(E3 decided for reading 2) — see the C4b entry below.

**The SDF-as-a-length audit** (`grep 'sdf(' mac_approx_projection.hpp gauge_exact_gradient.hpp
star_elimination.hpp ghost_projection.hpp collocated_varrho.hpp colocated_advection.hpp`): after C4
the only reads of `|sdf|` as a DISTANCE anywhere on the collocated path are the two foot points C4
just fixed. Everything else is a sign test or a per-axis crossing fraction, both invariant under a
positive scaling of the SDF — `wallAwareFaceStencil`'s `th = sc/(sc − ss)`, its abscissae
`xe/xc/xf/xg` (cells along ONE axis), `buildFaceCentroidDist`'s `d = s0/(s0 − s1)` and its uniform
index-space subsampling (an affine, per-axis map, so the index centroid IS the physical one pulled
back), `gpCenterGrad`'s `>= 0` branches, `starAval`, `gpBinaryOpenness`'s face means and `gpFillRow`'s
`th`. `ccFractionCore` already took `dx,dy,dz` and got `hp` in C2. Nothing was found that needs a
different algorithm.

- **G0**, `tests/kokkos` **43/43** host-openmp and **43/43** nvidia-cuda (the 41 of C3 plus
  `units_anisotropic_sphere` and `units_anisotropic_tgv`); `tests/kokkos_mpi` **106/106**
  host-openmp and **106/106** nvidia-cuda at np = 1, 2, 4; `sdflow_mpi_np1` bit-exact to
  single-rank, `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00`, `div 5.86e-12`.
- **Regressions** (CUDA, never `--update`), all three baselines **PASS** with every recorded number
  at `+0.00 %`, every fitted order and extrapolation at `d = 0.00` / `rel = 0.00 %`, and every
  iteration and step count equal:
  * `perf_baseline.json` (staggered): zh_sphere `K 7.2997 / 7.3891 / 7.4162 / 7.4361 / 7.4404`,
    order `2.29`, `K_inf 7.447`, pressure iters/step `6/7/7/6/6` at steps `60/80/75/150/245`;
    random_spheres `0.0064513 / 0.0063515 / 0.0062821 / 0.0062621` at `7/7/7/7`, order `2.19`,
    `k*_inf 0.0062362`; hollow_rings `0.018629 / 0.018228 / 0.017659 / 0.017608` at `9/10/9/9`,
    order `1.38`, `k*_inf 0.017184`.
  * `perf_baseline_colocated_ghost.json` (`--solver colocated --scheme ghost`): zh_sphere
    `K 7.3909 / 7.415 / 7.4283 / 7.436 / 7.4383` at `7/9/9/11/10`, order `1.59`, `K_inf 7.445`;
    random_spheres order `1.46`, `k*_inf 0.0062399`; hollow_rings order `1.75`, `k*_inf 0.017201`.
  * `perf_baseline_colocated.json` (`--solver colocated`): zh_sphere
    `K 7.4879 / 7.4463 / 7.4495 / 7.4437 / 7.4417` at `6/7/7/7/7`, order `4.00`, `K_inf 7.4424`;
    random_spheres order `0.30`, `k*_inf 0.0062504`; hollow_rings order `1.86`, `k*_inf 0.0172`.

  The two COLLOCATED baselines are part of G0 for the first time here, because C4 is the commit that
  admits the collocated policy on an anisotropic domain — and they are the deepest exercise the
  collocated cell gradients, the star modes and the approximate projection get.
- **Six verify scripts** vs a build of `6cf870b`, host-openmp, `np.array_equal` TRUE on every array:
  poiseuille **800/800**, periodic_spheres **60/60**, channel **23/23**, bfs **221/221**,
  lid_cavity **15/15**, and the collocated **colocated_taylor_green 4/4**.
- **The collocated kernels no script and no ctest reaches** (`set_face_interp` 4/5/6/7 —
  `fvViscousApply`, `embedViscousApply`, `embedDirichletGradient` — and
  `set_fluid_only_constraint` 1/2 — `starCorrectFaces`), on the Z&H sphere at N = 24, 40 steps:
  **24/24 arrays bitwise** at `OMP_NUM_THREADS=1`. At 4 threads the four `fluid_only 2` arrays
  differ by 1.665e-16 — and so do **two runs of the UNCHANGED tree against each other, by
  2.220e-16**: `starEliminate`'s `Kokkos::atomic_add` makes that one path run-to-run
  non-deterministic under OpenMP, which is why the statement is made at one thread.
- **Phase 1 + G1 units gates unchanged**, printed numbers included — `units_identity` bitwise;
  `units_scale_invariance` u `8.882e-17`, p `0.000e+00`, sphere sampled
  `4.070e-16 / 1.503e-15 / 1.761e-15` (div `6.915e-12`, d `1.850e-18`), scene
  `6.105e-16 / 1.224e-15 / 1.370e-15` (d `0.000e+00`); `units_vof_sigma` Young-Laplace
  `2.500000000e-01`, `capillary_dt 3.989422804e-01`, kappa rel `0.000e+00` / `2.202e-16`;
  `units_anisotropic_poiseuille`'s three rows still **1.388e-15 / 3.052e-08 / 3.701e-15** with
  `max|v| = max|w| = 0.000e+00` and the wall rows exactly 0. The one thing that changed in any
  printed output is that gate's (d) block, which now records the collocated policy and
  `hydro_force_torque` as **ADMITTED** instead of refused, and still asserts `enable_vof`'s refusal
  (with the three spacings in the message).

**G2 — `units_anisotropic_sphere`** (§8.3, the ctest half), host-openmp; the Z&H sphere `phi = 0.216`,
`K_ref = 7.442`, the regression's own configuration, cells `(N, 2N, N/2)` over the cube:

| N | cubic `K` | err_c | stretched `K` | err_s | iters/step c → s | steps c / s |
|---|---|---|---|---|---|---|
| 16 | 7.2996895730187736 | 1.9123 % | 7.2845639537087239 | 2.1155 % | 6.0 → 6.0 | 60 / 105 |
| 24 | 7.3890630590938464 | 0.7113 % | 7.3890694982915299 | 0.7112 % | 7.0 → 6.0 | 80 / 100 |
| 32 | 7.4161679424561768 | 0.3471 % | 7.4236776333311072 | 0.2462 % | 7.0 → 7.0 | 75 / 70 |

(i) strictly decreasing ✓; (ii) `err_s ≤ 4 err_c + 0.2 %` ✓; (iii) the least-squares order of the
stretched ERROR sequence **3.0759** (cubic 2.4603), bound 1.5 ✓; (iv) the cubic control reproduces
`perf_baseline.json`'s `steps` (60 / 80 / 75) and `iters/step` (6 / 7 / 7) **exactly** and its drag
factors to **1.353e-08 / 9.703e-10 / 1.824e-09** relative. That last number is NOT bit-for-bit and
cannot be: the baseline is a CUDA run and its `u.mean()` is numpy's pairwise sum against a sequential
C++ one here, and both perturb `K` at the pressure solve's own `rtol = 1e-8` stop. `fit_order` is
degenerate on three grids (two free linear parameters fit three points exactly at every `p`), which
is why the ctest gates the log-log error slope and the SCRIPT gates `fit_order`.

**G2 — the ladder** (`scripts/verify_anisotropic_spheres.py`, §8.3's script half), nvidia-cuda:

| N | cells (stretched) | spacing | `K_cubic` | `K_stretch` | err_c | err_s | it_c | it_s |
|---|---|---|---|---|---|---|---|---|
| 16 | 16×32×8 | (1, 0.5, 2) | 7.29969 | 7.28456 | 1.9123 % | 2.1155 % | 6.0 | 6.0 |
| 24 | 24×48×12 | (1, 0.5, 2) | 7.38906 | 7.38907 | 0.7113 % | 0.7112 % | 7.0 | 6.0 |
| 32 | 32×64×16 | (1, 0.5, 2) | 7.41617 | 7.42368 | 0.3471 % | 0.2462 % | 7.0 | 7.0 |
| 48 | 48×96×24 | (1, 0.5, 2) | 7.43609 | 7.44711 | 0.0794 % | 0.0687 % | 6.0 | 7.0 |
| 64 | 64×128×32 | (1, 0.5, 2) | 7.44041 | 7.45116 | 0.0213 % | 0.1231 % | 6.0 | 7.0 |

`fit_order`: cubic `p = 2.2900`, `K_inf = 7.44704` (**0.0677 %** of `K_ref`) — i.e. the ladder's cubic
column reproduces `perf_baseline.json`'s recorded `order 2.29` and `extrapolated 7.44704` and its
`6/7/7/6/6` iterations exactly; stretched `p_s = 2.2500` ∈ [1.5, 2.5] ✓ and
`K_s,inf = 7.46034`, **0.2465 %** of `K_ref` (bound 2 %) ✓. Pressure iterations per step on the
stretched grid are `≤ cubic + 2` at every rung ✓ — the §5 aspect rule is what buys that.

**The hydro-force gate** (§4.4 + §8.3), the periodic Stokes identity `F = F_body · V_fluid` with
`V_fluid = h_x h_y h_z ×` (# fluid u-DOFs), on the analytic-scene sphere at N = 24 after 300 steps of
the regression's `dt = 60` configuration, host-openmp:

| grid | spacing | `F_x` | the identity | relative gap |
|---|---|---|---|---|
| cubic `(24,24,24)` | (1, 1, 1) | 10.878000070904704 | 10.878 (10878 DOFs) | **6.518e-09** |
| stretched `(24,48,12)` | (1, 0.5, 2) | 10.844000905143286 | 10.844 (10844 DOFs) | **8.347e-08** |

Both are at the march's own residual floor (at 120 steps they read 7.351e-07 and 2.234e-06, i.e. the
remaining transient, and they fall together with more steps); the gate is the shared 1e-5 bound.

**G3 — `units_anisotropic_tgv`** (§8.4), host-openmp. Stokes Taylor–Green on `L × L × L_z` with cells
`(N, 2N, 4)`, `N = 16`, `L = 16`, `L_z = 4` → spacing `(1, 0.5, 1)`, `rho = dt = 1` (so every
reference scale but `hRef` is 1 and the isotropic control CAN be compared bitwise), 10 steps, the
DISCRETELY divergence-free initial field with `a_x sin(k h_x/2)/h_x = a_y sin(k h_y/2)/h_y`:

| row | amplitude ratio | exact `r^10` | rel (bound) | max\|div_o\| | profile | max\|P\| (the φ proxy) |
|---|---|---|---|---|---|---|
| stretched Stokes | 0.47852135218094743 | 0.47852135218094644 | **2.088e-15** (1e-10) | 1.074e-16 | 1.618e-15 | 1.280e-16 |
| stretched NS, `dt = 0.05` | 0.96143993730091892 | 0.96254747532385743 | **1.151e-03** (5e-3) | 2.678e-16 | — | — |
| isotropic control `(16,16,4)` at extent == cells | 0.48016564490786506 | 0.48016564490785385 | **2.335e-14** | 6.160e-17 | 1.415e-15 | 2.682e-16 |

and the isotropic control is **BITWISE** the cell-unit run of the same problem in `u, v, w, p`.

Two configuration choices this gate needed, both inside §8.4's freedom and both recorded because
they matter: **`mu = 0.25`**, which makes every assembled coefficient dyadic and therefore exactly
representable in the shipped FLOAT operator storage — `AC = 5/2` isotropic (`b = (¼,¼,¼)`) and
`AC = 4` stretched (`hRef = ½`, `hp = (2,1,2)`, `w = (¼,1,¼)`, `mu' = 1`, `b = (¼,1,¼)`) — the same
move E1 made for G1; at `mu = 0.1` the identical gate reads **1.5e-07 (isotropic control)** and
2.9e-07 (stretched), i.e. the WO-M float floor showing up in the ISOTROPIC row too, which is what
identifies it as not a metric statement. And **`dt = 0.05` on the NS row only** (`sdflow_tg`'s own
step): at the `dt = 1` the Stokes rows need for `tRef = 1`, the TG amplitude gives an advective
CFL of 1 and the discrete nonlinear term's imbalance with the pressure gradient reads 2.0e-02 —
the O(CFL²) advection error, not the metric.

### C4b (E3 — the v3 wall-torque reading) — flow, Phase 2 commit C4b

**E3 decided: reading 2.** The v3 transposed-stress wall torque is a **traction**, not a momentum
row, so `F' = mu' (A' x Omega')` with `A'_b = a_b V'/h_b'` the physical area vector in `hRef^2` — the
same `A` §4.4's traction paragraph forms — and `Omega' = Omega tRef`. The metric sits on the AREA
component the cross product consumes; `V'` enters only through `A'`. The counter-example that
separates the readings: a wall patch with `n dA = (A_x, 0, 0)` under `Omega = Omega_z e_z` produces a
force in **y**, which reading 1 would have scaled by `h_y' V'` and reading 2 scales by `V'/h_x'` —
equal only when `h_y' = 1/h_x'`. `hydro_reaction_torque_transpose` now forms
`Ax = ax*kA0` etc. with `kA_b = V'/h_b'` applied OUTSIDE the existing expression (exactly `1.0`
isotropic, so an identity multiplication), the `requireIsotropic` C4 left on
`hasMotion_ && cutcellPressure_` is gone, and §4.4 and §7 say so. **Every consumer §7 promised for
Phase 2 is now admitted**; `enable_vof` followed in Phase 3, and the two components with their own guards
still refuse — after Phase 3 that is the CFD-DEM coupling driver alone.

- **G0**, `tests/kokkos` **43/43** host-openmp and **43/43** nvidia-cuda; `tests/kokkos_mpi`
  **106/106** host-openmp and **106/106** nvidia-cuda at np = 1, 2, 4; `sdflow_mpi_np1` bit-exact to
  single-rank, `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00`, `div 5.86e-12`.
- **All three regression baselines** (CUDA, never `--update`): **PASS**, every recorded number
  `+0.00 %`, every iteration and step count equal — identical to C4's line above.
- **`units_identity` / `units_scale_invariance` / `units_vof_sigma` / `units_anisotropic_poiseuille`
  printed output unchanged**, digit for digit (u `8.882e-17`, p `0.000e+00`; sphere
  `4.070e-16 / 1.503e-15 / 1.761e-15`, div `6.915e-12`, d `1.850e-18`; scene
  `6.105e-16 / 1.224e-15 / 1.370e-15`, d `0.000e+00`; Young-Laplace `2.500000000e-01`; `capillary_dt
  3.989422804e-01`; kappa rel `0.000e+00 / 2.202e-16`; G1's three rows
  `1.388e-15 / 3.052e-08 / 3.701e-15`).
- **The decisive check — `movingscene_advect_mpi`, the only moving-geometry gate** (a 48³ towed
  sphere, `hasMotion_ && cutcellPressure_` both true, i.e. the exact path C4 refused), np = 1, 2, 4
  on both backends: PASS, with `du = 0.000e+00` and `dp = 0.000e+00` at np = 1 and
  `dF <= 2.9e-15`, `dT <= 1.5e-14`, `div 2.37e-10`. Run against a build of C4 (`0d8417b`) at
  **`OMP_NUM_THREADS=1`** the reaction force is **BITWISE identical on all 17 digits and
  reproducibly so**:

  ```
  C4  r1/r2   F_ref = (-0.11367215421887535, -0.14541750499799105, -0.00070001790246804435)
  C4b r1/r2   F_ref = (-0.11367215421887535, -0.14541750499799105, -0.00070001790246804435)
  ```

  At 4 threads the two builds differ in the last 2-3 digits — and so does **C4 against itself**: three
  runs of the unchanged binary spread `5.6e-16` in `F_x` against `1.1e-15` between the builds, i.e.
  the same size. That is `hydroForceTorqueReaction`'s own `Kokkos::atomic_add` accumulation floor,
  which the test's header already documents ("NOT bitwise even at np = 1, and not because of MPI"),
  and it is why the statement is made at one thread — the same reason C4's `starCorrectFaces` probe
  was.
- **The six verify scripts were deliberately NOT re-run**: none of them moves a body, so none reaches
  the changed kernel, and C4's byte comparison already covers every path they exercise.

**OPEN ITEM — the anisotropic torque path is implemented, isotropic-bitwise, and UNTESTED against a
reference.** No gate in the tree measures the torque of a ROTATING body against the analytic Stokes
couple `8 pi mu R^3 Omega`: `movingscene_advect_mpi` TRANSLATES, so `Omega = 0` and the v3 kernel
returns before its arithmetic runs. Building such a gate was deliberately out of C4b's scope. It
belongs to the coupling campaign, and it is the one place in Phase 2 where a per-axis constant is
carried on argument alone rather than on a measurement.

### Phase 2, the final table (what shipped, and the number that proves each half)

| commit | work order | what it put in | the number |
|---|---|---|---|
| `5285aae` | U9 | this note | — |
| `12cac0f` | U10 | the per-axis MOMENTUM fold `b_a = mu' w_a`, `AC = idiag + 2((bx+by)+bz)`, the `!aniso` smoother dispatch, `VelocityMG::setMetric`, scalar transport | `AC` bitwise the old `idiag + 6.0*beta` at `w = (1,1,1)`; `1.0700000524520874` at `w = (1,4,¼)` |
| `735fb46` | U11 weights | the per-axis PRESSURE weight `w_a` at the three `setOpenness` sites and every correction / cell-gradient kernel, `buildOpenness(hp)`, the §4.2 census, the §1.4 **snap**, the §7 refusals | **G1** Poiseuille pointwise exact **1.388e-15** at `spacing (1, 0.25, 2)` |
| `6cf870b` | U11 ⚑ A | `mgChooseRatio` in all four level loops, `levelRatios()`, `PECLET_FLOW_MG_ASPECT` | **G4** stretched MG-PCG **500/500/500 CAPPED → 7/8/8**; Z&H sphere **24 → 10** iters; V-cycle rate **0.6920 → 0.1501** |
| `0d8417b` | U12 ⚑ B | the §6 closures on the index-space normal `m`, the slip length, `ibmVolfrac`, the §4.4 forces; the collocated policy and the force integrals ADMITTED | **G2** `K_s,inf = 7.46034` = 0.2465 % of 7.442, `p_s = 2.2500`; **G3** amplitude ratio to **2.088e-15** |
| `f168436` | E3 | the v3 wall-torque traction on its AREA vector; the last refusal lifted | `movingscene_advect_mpi` **bitwise** C4 vs C4b at one thread, all 17 digits |

**G0, on every one of them:** `tests/kokkos` **43/43** and `tests/kokkos_mpi` **106/106** (np = 1, 2,
4) on host-openmp AND nvidia-cuda; `sdflow_mpi_np1` bit-exact to single-rank; all THREE regression
baselines at `+0.00 %` with every iteration and step count equal; the six verify scripts
`np.array_equal` TRUE on every array.

**The three things Phase 2 measured that were not asked for, and are worth carrying forward:**

1. **The float operator-storage floor is the binding constraint on any anisotropic exactness claim**
   (E1). At `spacing (1, 0.3, 2)` neither `mu' = 55.5…` nor `AC = 124.61…` is representable and the
   whole profile is scaled by `1 − 1.06e-07`; `-DPECLET_FLOW_MREAL_DOUBLE` reads `8.674e-15`. Every
   exactness gate in this phase is therefore stated at a float-representable metric, with the
   production-shaped configuration kept beside it as a tripwire. `docs/SCALING_ISSUES.md` #1.
2. **Two solver paths are not run-to-run deterministic under OpenMP**, both through
   `Kokkos::atomic_add` over an overlay: the fluid-only star elimination (`starEliminate`, 2.2e-16 on
   the collocated Z&H sphere) and `hydroForceTorqueReaction` (5.6e-16 on the moving-scene force).
   Neither is a Phase 2 change — both are pre-existing — but any future byte comparison that touches
   them has to be made at `OMP_NUM_THREADS=1`, as this phase's were.
3. **Two isotropic-rule copies were left behind on purpose**: `CutcellMG::predict()` (the
   `scripts/check_decomposition.py` pre-flight, C3) still models the full-coarsening rule, and §5.4's
   aspect rule on isotropic cells after a telescoping merge is deliberately not engaged because it
   would change bits at `extent=None`. Both are one-line experiments for the scaling campaign.

**The one open item.** The anisotropic v3 wall torque is implemented and isotropic-bitwise but
**untested against a reference** — no gate measures a ROTATING body's torque against `8 pi mu R^3
Omega`, and the only moving-geometry test translates. It is the single place in Phase 2 where a
per-axis constant rests on argument rather than measurement; it belongs to the coupling campaign.

---

## 11. Traps, in the order you will meet them

1. **`hRef` is `min h`, and `h' = 1.0` is exact only on that axis.** Never compute `h'` as
   `extent/(cells*hRef)`; compute `h[a]/hRef` from the stored doubles.
2. **The snap must precede everything.** `units_scale_invariance` builds `extent = N_a * H` per axis;
   without the snap `h'` can be `1 ± 1 ulp` on two axes and the "isotropic" run stops being Phase 1's
   arithmetic.
3. **Association order in `Ac`.** `idiag + 2.0*((bx + by) + bz)`; any other grouping breaks G0 at the
   last bit in the domain-BC and const-coeff paths.
4. **The smoother dispatch.** `diffSmoothColor`/`Du` and `constCoeffResidual` must run the legacy body
   under `!aniso` (template `<bool Aniso>`), not a per-axis body with `w = 1.0` — the summation order
   differs.
5. **`setMetric` before `init`.** The level table is built in `init`/`initMpi` (called from
   `setSolidDevice` and the MPI init), long before `setOpenness`; the ratio rule needs `hp` at that
   point.
6. **Telescoping.** An aspect-deferred axis is not "blocked". Test with `PECLET_FLOW_TELESCOPE=1` on a
   stretched `np = 4` case that the merge is not triggered by deferral alone.
7. **`buildOpenness` has taken `dx,dy,dz` all along** (`~:2010` passes `1.0`). Hand it `hp`; the
   marching-squares path ignores them.
8. **`ccSampleExt` at the foot point** clamps integer indices, not a NaN coordinate — the existing
   guard on `sdi` stays. Since `h_a' ≥ 1` on every axis, `|m_a| = |n_a|/h_a' ≤ |n_a| ≤ 1`: the foot
   point moves at most `d'` cells per axis, never farther than today, so the clamp into `[1, e-2]`
   is still sufficient.
9. **Float slip length.** `lamAxis` is `float`; `hp` enters as `(float)hp[a]` and `1.0f` is exact.
10. **The census reduce mixes components.** Convert inside the lambda (§4.2), not after.
11. **`hydro_force_torque_reaction` is the CFD-DEM source.** Its `h_a' V'` factor must land in the same
    commit as the guard is lifted (C4), or the coupling reads a wrong force silently — which the
    coupling's own isotropy guard currently prevents; keep that guard until its deposit kernel is
    per-axis.
12. **CUDA is not bitwise-deterministic across runs** for the reductions; byte comparisons are an
    OpenMP statement (plan §9.2). CUDA gates compare to tolerance.
