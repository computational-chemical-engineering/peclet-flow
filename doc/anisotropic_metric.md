# Anisotropic cells, single phase — the metric in every discrete operator (Phase 2 design note)

**Status:** design settled 2026-09-06 (Fable). Implementation follows this note; nothing below is
implemented yet. Plan: `suite/docs/PHYSICAL_UNITS_PLAN.md` §3.2, §5, §9.4, §9.8. Phase 1 (isotropic
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
`F_a = - sum_owner R_a * h_a' * V'` in `forceTotalToPhys` units (the isotropic `h_a' V' = 1`). The
wall-torque term (v3) takes the same `h_a'` on its force factor and the physical lever arm.

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

| refused | where the guard goes | lifted by |
|---|---|---|
| `enable_vof` and every VoF entry point | `enableVof()` | Phase 3 |
| the AMR module (`Octree(cells, extent)`) | `core/python/amr_bindings.cpp:h0FromExtent` (already refuses) | Phase 3 |
| the CFD-DEM coupling driver | `coupling/python/peclet_coupling/driver.py` (already refuses) | Phase 2 follow-up (per-axis deposit) |
| the collocated policy, until the ⚑ B commit lands | `Solver<Colocated>::setPhysicalDomain` | this phase, commit C4 |
| `hydro_force_torque*` until its constants land (§4.4) | those two functions | this phase, commit C4 |

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
`runChannel` on cells `(16, 40, 8)` with `extent = (16, 12, 16)` so `spacing() == (1.0, 0.3, 2.0)`
(`hRef = 0.3`, `h' = (10/3, 1, 20/3)` — the finest axis is `y`, a non-trivial check of `hRef = min`),
walls at the `y` cell centres `ylo = 12.5*0.3 = 3.75`, `yhi = 28.5*0.3 = 8.55` (`H = 4.8`),
`rho = 1, mu = 0.1, F = 0.01, dt = 50`, 300 steps, `velTol 1e-14`, `cutcellPressure = false`.
- `max_j |u(y_j) - F (y_j - ylo)(yhi - y_j)/(2 mu)| / u_max ≤ 1e-9` over every fluid DOF,
  `u_max = F H^2/(8 mu) = 0.288`; the DOFs on the walls are exactly `0`.
- `max |v|, max |w| ≤ 1e-12 * u_max`.
- The isotropic control `(16, 16, 16)` at `extent = cells` in the same test: the same bound, and its
  fields `np.array_equal` to the `units_identity` run (G0 covers this; the test asserts it anyway).
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

*(empty until the commits land)*

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
