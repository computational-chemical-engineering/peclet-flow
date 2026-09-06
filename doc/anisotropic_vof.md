# Anisotropic VoF — the geometric two-phase stack on stretched cells

*Design note, 2026-09-06, Phase 3 (VoF half) of `suite/docs/PHYSICAL_UNITS_PLAN.md` (§9.5).
Status: DESIGN — every ⚑ of §9.5's VoF half is decided below; the implementation follows the
work orders of §10 against the gates of §11. The AMR half is `core/docs/amr_anisotropic.md`.
Decisions D1–D5 of the plan are taken as given and are not reopened here.*

## 0. The result in one paragraph

Under the unit-lattice representation Phase 1 adopted (plan §3.2 (B)), a stretched cell
`h = (h_x, h_y, h_z)` is the unit cube of the index coordinates `xi_a = (x_a - o_a)/h_a`, and
**every volume-fraction operation is already the textbook normalised-stretched-cell formulation**:
PLIC reconstruction, the plane↔volume relations, slab/box fluxes, the Weymouth–Yue sweeps and
their conservation and boundedness proofs, the cut-cell rules, the wisp guards, the dilation term,
the momentum-consistent fluxes and the solid-band fill state machine do not change by a single
operation. What the map does **not** preserve is angles, lengths and areas — so everything that
treats a normal as a *direction*, a height as a *length* or a polygon as an *area* takes the
metric: the height-function curvature (physical column heights over physical transverse spacings),
the two paraboloid fits (physical coordinates), the CSF face force (the projection's own per-axis
gradient weight), the capillary time step (`h_min`), the contact-angle rotation (physical frame),
the interfacial area and the phase-change layer's normal distances and `V_cell`. Each of those is a
per-axis constant folded at kernel entry, written so that at equal spacings the code multiplies by
an exact `1.0` and the isotropic battery stays **bit-identical**. The metric is three numbers
`h'_a = h_a / h_ref` (plus their product and maximum) appended to `Solver::UnitScales`.

## 1. Inherited contract and the two rules

Phase 1 (`flow/src/flow_ibm.hpp`, the derivation block above `Solver::UnitScales`) fixes the
boundary conversions. Everything internal is on the unit lattice in reference units:
`h_ref = min_a h_a` (plan §3.2; Phase 1 wrote `hRef = h[0]` because the cells were equal — Phase 2
turns that into the `min`, and nothing below depends on which of the two it is because every
per-axis factor is spelled with an explicit `min`/`max` where the choice matters), `rhoRef` the
first `set_rho`, `tRef` the first `set_dt`. The VoF-relevant ones, verbatim from Phase 1:

    sigma' = sigma * tRef^2 / (rhoRef*hRef^3)      kappa' = kappa * hRef       d' = d / hRef
    v_a    = u_a * tRef / h_a                       dt'    = dt / tRef          p' = p*tRef^2/(rhoRef*hRef^2)

and the two deviations recorded at the top of the plan stand: `WyAdvector::h_` stays `1.0` (its
Courant number `v_a dt'` IS `u_a dt/h_a`, per axis, already), and the four property setters convert.

Two rules govern every change in this note.

**Rule A — per-axis constants only.** No kernel gets a different *algorithm* for `h_x != h_y`;
the metric enters as at most three constants per kernel family, evaluated once per call. If a
kernel needed more than that it would be an escalation under plan §9.6 item 2. None does (§3–§7
is the proof, kernel by kernel).

**Rule B — the isotropic path executes today's floating-point operations.** Every per-axis
factor is written in one of these three forms, and the implementer picks the first that applies:

1. **A multiplicative ratio that is exactly `1.0` at equal spacings.** `r = h'_d / h'_1` is
   `1.0/1.0 = 1.0` exactly, `x * 1.0 == x` and `x / 1.0 == x` bitwise in IEEE-754. So
   `hx = 0.5*(h[2]-h[0]) * r` reproduces `hx = 0.5*(h[2]-h[0])` bit for bit. Products of
   ratios (`h'_x h'_y h'_z`) and ratios of identical doubles (`|m|/|m|`) are `1.0` exactly too.
2. **The same value on every axis in the same loop order.** `x_a = h'_a * xi_a` for each `a`
   is today's `h * xi_a` with the same `h` on every axis: identical operations, identical bits.
3. **A guarded isotropic branch** — only where the anisotropic expression has a genuinely
   different operation tree (e.g. `sqrt(3)*w` versus `|w|`). Rule B(3) is the last resort and each
   use is listed in the work order that introduces it. Sums that used to be a single product
   (`6*beta` → `2b_x + 2b_y + 2b_z`) must be parenthesised as one sum before anything is added
   to them, so the isotropic value is the single correctly-rounded `6*beta` (the sum of exact
   power-of-two multiples rounds once, to the same double).

The gate for Rule B is not an argument but the `extent=None` battery plus the three isotropic
units gates (`units_identity`, `units_scale_invariance`, `units_vof_sigma`), all of which must
stay byte-identical (§11 G0).

## 2. The geometry of a stretched cell — what the index map preserves and what it does not

Write `H = diag(h'_x, h'_y, h'_z)` (reference units; `H = I` in cell units and in every isotropic
run) and `x' = H xi` for the physical position (in `h_ref`) of an index point `xi`. The cell is
the unit cube in `xi`. Then, exactly:

- **Volumes scale by the constant `det H`**, so every *fraction* of a cell — the colour, the
  fluid fraction `eps`, a slab or box fraction, a PLIC volume — is the same number in both frames.
  This is why §3 changes nothing.
- **Ratios along an axis are invariant** (a plane crosses a grid line at the same fraction of
  the cell in both frames): face Courant numbers `a_f = v_f dt'`, cut-cell apertures, the GFM
  `theta` of the phase-change energy rows and the Robust-Scaled wall crossings are unit-free.
- **A plane `n . x' = d` (unit physical normal `n`) is `m . xi = alpha` with `m = H n`,
  `alpha = d`** — and conversely a plane reconstructed on the unit cube with index normal `m` is
  the physical plane with `n = H^{-1} m / |H^{-1} m|`. The MYC/Youngs estimators difference the
  colour on the index lattice, so what they return IS `m` (they estimate `-grad_xi C = -H grad C`),
  which is exactly the normal the unit-cube PLIC needs. A stretched cell therefore needs no new
  normal estimator and no new plane↔volume routine.
- **Three quantities acquire a per-cell, per-orientation factor** `s(m) = |H^{-1} m| / |m|`
  (with `m` any scale of the index normal; `s = 1` exactly when `H = I` because it is a double
  divided by itself, Rule B(1)):

      physical unit normal       n   = H^{-1} m / (|m| s)
      normal distance            phi_phys = phi_xi / s          (phi_xi = (m.xi - alpha)/|m|)
      interfacial area           A_phys   = A_xi * det(H) * s   (A_xi = the unit-cube polygon area)

  The area rule is the cofactor transformation of the area vector, `A n = det(H) H^{-T} (A_xi m̂)`;
  the distance rule is `(H d).(H^{-1} m̂) = d . m̂` rescaled by `|H^{-1} m̂|`.
- **Curvature is a second derivative of a graph and needs both a length and two spacings**:
  a height `h_idx` along axis `d` is the physical height `h'_d h_idx`, differenced over `h'_1`,
  `h'_2`. §4 writes that out.

Everything in §3–§7 is one of these five facts applied to one kernel.

## 3. PLIC in normalised stretched cells — DECISION V1: the existing kernels, unchanged

`core/include/peclet/core/vof/plic.hpp` (the L1 container-free layer; `flow/src/vof/plic.hpp` is
a thin include) is the normalised-cell formulation by construction: it maps *every* cell to
`[0,1]^3` and never asks what the cell's physical size is. Its plane↔volume relation `V(m, alpha)`
is invariant under `(m, alpha) -> (lambda m, lambda alpha)` and `plicSlabVolume`/`plicBoxVolume`
are coordinate rescales inside that same cube. With `m` the index normal of §2 the reconstruction
`plicAlpha(m, C)` returns the plane whose *physical* fluid volume fraction is `C` — the
Scardovelli–Zaleski "stretched cell = rescaled normal" construction, already in the code because
the code never left the unit cube. Consequently:

- `mycNormal`, `youngsNormal`, `plicAlpha`, `plicVolume`, `plicSlabVolume`, `plicBoxVolume`,
  `faceFluxVolume`, `plicPolygon` (its vertices are unit-cube vertices), `planeCellFraction` and
  `sphereCellFraction` (test helpers; the sphere one gains a per-axis `h[3]` overload for the
  stretched test scenes only) — **unchanged**.
- `WyAdvector` (`flow/src/vof/advect_wy.hpp`): sweeps, `c_i = H(C^n - 1/2)`, the exact
  telescoping conservation, the `1/4` 3-D CFL cap, `maxCourantInterface*`, the wisp predicates,
  the clip census — **unchanged**; `h_` stays `1.0` (plan deviation 1). The per-face Courant numbers
  the Solver hands it are already `v_f dt' = u_f dt/h_f` per axis because `velToInt(a)` is per
  axis.
- The cut-cell rules (`core/.../vof/cutcell.hpp`): `eps_eff`, the admissible flux interval, the
  `1/eps` Courant amplification, the band-fill passes 2–3 — all fractions and axis ratios —
  **unchanged**.
- `momentum_advect.hpp` (V2b): its half-shifted control volumes and `plicBoxVolume` fluxes are
  fractions of the same unit cubes; the density it forms per face is a colour mix — **unchanged**.
  Its one metric-carrying argument is the CSF term it receives (`flow_ibm.hpp` ~:5196 passes
  `sigmaCsf_, vofAdv_.h()`), which follows §5.
- `energy_advect.hpp` (P2/P3): advects `H = rho c_p T` with the colour's own fluxes — fractions —
  **unchanged**.

The advector-level gate is therefore an *identity*: with the same index face velocities the
stretched and the cubic run are bitwise equal (§11 K1 states it so that a future change that
sneaks a length into the advector fails loudly). The physics gate is at Solver level: the same
physical translation on a stretched extent conserves volume to the WY floor (§11 S1).

## 4. Height-function curvature with physical column heights — DECISION V2

`core/.../vof/curvature.hpp` keeps its cascade (tier 1 → 2a → [2b off] → 3) and its data
sources (column sums, PLIC volumes; never a differenced normal). Three things change, all
per-axis constants:

**V2.1 Column direction ordering stays in index space.** Tier 1 tries the axis `d` with the
largest `|m_d|` of the cell's *index* normal. That is the right frame: a 7-cell column closes
when the interface, as a graph over the index transverse plane, varies by at most ~2.5 cells
across the 3×3 patch, and that variation is `|m_1/m_d| + |m_2/m_d|` in index space whatever the
physical aspect ratio. The ordering is categorical (never differenced), so there is no accuracy
argument for the physical frame either. Basilisk orders by the physical normal only because its
cells are cubes. **Unchanged.**

**V2.2 `hfColumnHeight` is unchanged; the height it returns is in cells along `d`.** The physical
height (in `h_ref`) is `h'_d * h_idx`. Monotonicity, the pure-cell window, the orientation rule
and `monoTol` are colour-space statements — unchanged.

**V2.3 `hfPatchKappa` takes the metric.** With the nine heights `h[p + 3q]` (p along `d1`, q along
`d2`), Han et al. (2024) eqs. (4)–(5) on spacings `(h'_{d1}, h'_{d2})` for the graph
`f = h'_d h_idx` read

    f_x  = 0.5*(h[2,1]-h[0,1]) * (h'_d/h'_d1)               f_y  = 0.5*(h[1,2]-h[1,0]) * (h'_d/h'_d2)
    f_xx = (h[2,1]-2h[1,1]+h[0,1]) * (h'_d/h'_d1^2)          f_yy = (h[1,2]-2h[1,1]+h[1,0]) * (h'_d/h'_d2^2)
    f_xy = 0.25*(h[2,2]-h[0,2]-h[2,0]+h[0,0]) * (h'_d/(h'_d1 h'_d2))
    kappa' = -(f_xx + f_yy + f_xx f_y^2 + f_yy f_x^2 - 2 f_xy f_x f_y) / (1 + f_x^2 + f_y^2)^{3/2}

in `1/h_ref` — the unit the rest of the solver already assumes for `kappa'` (`curvToPhys() =
1/hRef`, `p' = sigma' kappa'`). Signature: `hfPatchKappa(const double h[9], const VofMetric& g,
int d)` with `struct VofMetric { double h[3]; }` (container-free, `h = {1,1,1}` default); the
one-argument overload delegates with the unit metric so every existing test compiles and stays
bitwise. The five ratios are formed as `h'_d / h'_d1` etc. — Rule B(1) — so at equal spacings each
is exactly `1.0` and `f_x = hx_idx * 1.0`.

**V2.4 The paraboloid fits (tier 2b, tier 3) run in physical coordinates.** `PtFit`/`PvFit`
fit `z = a0 + a1 x + a2 y + a3 x^2 + a4 xy + a5 y^2` in an orthonormal frame `(t1, t2, n)` about
the target's PLIC centroid; `paraboloidKappa` is then a physical curvature only if the frame is
orthonormal *in physical space* and the points are physical positions. So:

- the target's unit normal is `n = H^{-1} m / |H^{-1} m|` (§2), `curvFrame(n)` unchanged;
- `pvFitAdd(...)` maps each polygon vertex `X_idx = off + v - 1/2 - org_idx` to `X = H X_idx`
  before projecting into the frame (`org` is the centroid in index units, mapped the same way);
  the polygon's own plane normal `np` is formed from `H^{-1} m_j`; `polygonMoments2d` and the
  normal equations are unchanged (they act on frame coordinates);
- `ptFitAdd(...)` receives `X = H X_idx` with `X_idx[d] = orient * h_idx`, `X_idx[d1] = p`,
  `X_idx[d2] = q` (the interface position along the column, in cells, times `h'_d`);
- the Wendland supports: `dW = kPvWeightWidth * max_a h'_a` and `ptWeightWidth * max_a h'_a`,
  so the 5³ stencil's farthest cell along the long axis stays inside the support exactly as the
  cubic `d = 2.5` cells keeps it today. `r` is the frame distance of the physical centroid,
  computed as today from the frame coordinates (bitwise at `H = I`). `cosMin` is a physical
  angle test and is now evaluated between physical normals — correct where the index test was an
  approximation.

At `H = I` every mapped quantity is `x * 1.0` and the frames are today's (Rule B(1)/(2)):
bitwise. `kappa'` from the fits comes out in `1/h_ref` because the coordinates are in `h_ref`.

**V2.5 Tier 2b stays OFF** (measured verdict in the file header; nothing here changes its data
set, which is the mechanism).

**Units out.** `vof_curvature()` keeps returning `kappa' * curvToPhys()` = `1/length`; nothing
at the API changes. The driver `VofCurvature` (`flow/src/vof/curvature_field.hpp`) gains a
`VofMetric metric` member (default unit) that the Solver sets from `UnitScales` and passes into
`curvHeightCell`/`curvFallbackCell`; `VofInterfaceArea` (same skeleton) the same.

## 5. The CSF face force and the capillary time step — DECISION V3

**V3.1 The force carries the projection's own per-axis gradient weight.** The balanced-force
identity (`vof/surface_tension.hpp`) is that `F_c(i)` must be the *same discrete operator* applied
to `sigma*kappa*C` as the momentum RHS applies to `P`. Phase 2 (plan §9.4 U11) gives the pressure
face difference of component `a` the per-axis weight of §3.2, `-(P(i) - P(i-s_a)) / h'_a^2`.
Deriving the CSF term the same way — `F_a = sigma kappa dC/h_a` physical, times
`forceToInt(a) = tRef^2/(rhoRef h_a)`, with `sigma`, `kappa` substituted from §1 — gives

    F'_a(i) = sigma' * kappa_f'(i) * (C(i) - C(i - s_a)) / h'_a^2

i.e. **exactly the pressure weight**. With a constant `kappa'` the force is again the discrete
(weighted) gradient of `sigma' kappa' C`, in the range of the operator the projection inverts, so
the projection annihilates it and the stationary droplet stays at machine zero on a stretched grid
as it does on a cubic one (§11 S2). Implementation: the last argument of `csfFaceForce(sigma, kf,
dC, hGrad)` becomes the per-axis gradient denominator `hGrad = h'_a^2` (documented; `1.0` in cell
units — `x / 1.0 == x`, Rule B(1)). The three call sites — `addCsfRhs`, `addCsfRhsCellInterp`
(the ablation, same weight), the block-container CSF (`block_container.hpp` ~:606, per block the
same `h'_a^2`) — and the momentum-consistent path (`flow_ibm.hpp` ~:5196) pass `u_.hr[c]*u_.hr[c]`
instead of `vofAdv_.h()`. **Contract with Phase 2:** whatever *spelling* Phase 2 gives the pressure
weight in `buildRhs*`/`projectCorrect*`, the CSF term uses the same symbol; the exactness gate,
not the formula, is what is gated. `csfFaceCurvature` (the three-way face rule) is unchanged.

**V3.2 The capillary time step uses the smallest spacing.** Brackbill's constraint is the
period of the shortest resolvable capillary wave, wavelength `2 h_min` (Denner & van Wachem 2015
confirm the prefactor and the `h^{3/2}` scaling; Basilisk uses `Delta = min`). So
`capillaryDt(rhoSum, min_a h'_a, sigma')` — spelled with the explicit `min` so it does not depend
on Phase 2's `hRef` choice; `min_a h'_a` is `1.0` today. `vof_step_limits()`, `step()`'s guard and
the adaptive step read the same function.

**V3.3 Everything else in the CSF is unit-free**: `csfKappaDefined`, the orphan census, the
wisp threshold `interfaceEps = 1e-8`, `set_vof_kappa_constant` (already converts through
`curvToPhys`).

## 6. Wetting — DECISION V4: rotate in the physical frame, walk in index space

**V4.1 Static contact angle (`core/.../vof/wetting.hpp`, the WO-S fill).** `cos(theta) = m . n_w`
is a physical angle, so the theta-plane is built in the physical frame and mapped back:

    n_w  = H^{-1} g_w / |H^{-1} g_w|      g_w = the index central difference of the exchanged SDF
    m_f  = H^{-1} g_f / |H^{-1} g_f|      g_f = youngsNormalFluidOnly (an index normal)
    t = m_f - (m_f.n_w) n_w ;  m_theta = cos(theta) n_w + sin(theta) t/|t|        (unchanged)
    m_idx = H m_theta ;  alphaTh = plicAlpha(m_idx, C_f) ;  C_s = vofWettingFraction(m_idx, ...)

`vofWettingPlane` gains the `VofMetric` argument and does the two maps at entry (`H^{-1}`) and
exit (`H`); `plicAlpha`/`plicVolume` renormalise internally so `m_idx` needs no L1 pass. The
default `kVofPivotVolume` anchor reads nothing else. The three ablation pivots
(`kVofPivotInterface/WallNormal/ContactLine`) map their pivot points `p_f`, `c0` through `H`
before the dot products and back through `H^{-1}` for `alphaTh` — mechanical, and they stay
ablations. `cosApp` (the diagnostic the dynamic model reads) is the physical cosine. At `H = I`
every map is `x / 1.0` or `x * 1.0` and the L2 normalisations act on the same numbers: bitwise.

**V4.2 The band walk stays an index walk, along the physical normal.** The fill walks from the
band cell `step = 1..4` cells along `round(step * w)` to find the anchor. The direction that
continues the interface is the physical `n_w`, expressed in index steps: `w ∝ H^{-1} n_w ∝ H^{-2}
g_w`. Spelled as `w_a = g_w[a] / (h'_a h'_a)` followed by ONE L2 normalisation — at `H = I` that is
`g_w / |g_w|`, today's vector, bitwise (Rule B(1); a second normalisation of an already-unit vector
would NOT be bitwise and is forbidden). The neighbour-average branch, the pure-anchor rule and the
neutral fallback are unchanged.

**V4.3 Dynamic contact angle (`flow/src/vof/wetting_dynamic.hpp`, WO-V6).** Cox–Voinov
`theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta/lambda)` needs one length `Delta`, "the cell size" —
on a stretched grid the cell size *across the wall*. DECISION: `Delta = h'_{d*}` with
`d* = argmax_a |n_w,a|` per contact cell (the axis the wall is most normal to). Then
`logRatio = -ln(slip') + ln(Delta)` with `slip' = lambda/h_ref` (`lenToInt`, as today); at `H = I`,
`ln(1.0) = 0.0` exactly and `x + 0.0 == x`, so the stored `-ln(slip)` is untouched (Rule B(1)).
The admissibility check becomes `slip' < min_a h'_a` (lambda below the smallest cell), spelled
with the explicit `min`. `U_cl = u . t_hat` is physical: the cell-centre velocity is converted per
axis `u'_a = v_a h'_a` (`v` the index velocity the driver already averages from the faces) and
`t_hat` is the in-wall direction of the physical `m_f`, `n_w` of V4.1; the 3-point smoothing along
`t_hat` steps in index space along `H^{-1} t_hat` (one normalisation, as in V4.2).
`Ca_cl = mu'_l U'_cl / sigma'` is dimensionless in reference units (`mu' U'/sigma' = mu U/sigma`
exactly). Hysteresis, pinning, the clamps and the census are angle statements — unchanged.

**V4.4 Navier slip in the momentum wall closure (`set_wall_slip_length`, WO-V6b)** is a length
in the Robust-Scaled closure — Phase 2's U12 owns the closure; the setter's `lenToInt` conversion
stays, and Phase 2 is told (plan §9.8 table) that `slipLambda_` is in `h_ref` units.

## 7. Phase change — DECISION V5: physical normals, areas and `V_cell` at the driver

`flow/src/vof/phase_change.hpp` stays container-free and mostly unchanged; the *driver*
(`flow_ibm.hpp`, the `pcCompute*`/`pcRegress`/`pcBudget*` kernels ~:9560–:10480) forms physical
arguments before calling it. The pieces:

**V5.1 Interfacial area** (`A_G`, the unit conversion of `mdot`; `interface_area.hpp`,
`marching_cubes.hpp`, `interface_area_field.hpp`, `marching_cubes_field.hpp`). Every construction
returns a physical area in `h_ref^2`:

- `kAreaPlic`: `plicArea(m, alpha) * det(H) * s(m)` (§2), wrapped as `plicAreaMetric(m, alpha, g)`;
  the factor is `(h'_x h'_y h'_z) * (|H^{-1} m| / |m|)`, exactly `1.0 * (q/q) = 1.0` at `H = I`.
- `kAreaMetric` / `kAreaFootprint` / `kAreaNormal` (the height-function constructions): the area
  element `sqrt(1 + f_x^2 + f_y^2)` with the *physical* slopes of V2.3 times the footprint
  `h'_{d1} h'_{d2}`; `hfSurfaceNormal` returns the physical normal (frame of V2.4).
- the marching-tetrahedra sheet (`kAreaMc*`, the DEFAULT since WO-P3d): the vertex positions are
  mapped through `H` before `mcTwiceArea`/`mcPolygonArea`; the crossing parameter `t` along a
  lattice edge is an axis ratio and stays; the clip to a cell's cube is done in index space
  (the cube is the unit cube) and the *clipped piece* is then mapped. The deposit rules
  (centroid/split) are index-space bookkeeping — unchanged.

**V5.2 The one-sided gradient fit (Malan; `PcGradFit`) takes physical offsets.** The driver
forms, per sample at integer offset `d`, the physical offset `delta = H d` and the physical unit
normal `n` of §2, then calls the unchanged `pcGradWeight(delta, n)`, `pcOffsetDistance(phi_c,
n, delta)`, `pcCurvedDistance(phi, delta, n, kappa')` and `pcGradAdd`. `phi_c` is
`pcCentreDistance(m, alpha) / s(m)` (the physical centre distance). The fit then returns a physical
`dT/dn` (per `h_ref`), which is what `pcMassFlux` needs. At `H = I`: `delta = d * 1.0`, `phi_c / 1.0`
— bitwise. The pure-cell classification (`cj <= pureEps`) and the sign tests are unchanged.

**V5.3 The plane-anchored (GFM) rows.** `pcGfmTheta(phiC_idx, nd_idx, s, ...)` measures the
distance along a grid line *in cells*, which is an axis ratio: with the INDEX centre distance
and INDEX unit normal it is already right on a stretched cell (§2, "ratios along an axis") —
**unchanged**, and the operator-flux `mdot` (`pcOperatorMassFlux`, the shipped `energy_order=2`
path) reads the same `theta`. `pcGfmThetaK` (the curvature correction, on with
`set_phase_change_curvature_distance`) mixes a physical `kappa'` with an index `rho^2 = 1 - nd^2`;
its stretched form is `rho^2 = h'_d^2 (1 - n_d^2)` with the *physical* `n_d` and the correction
applied to the *physical* distance, then divided by `h'_d` to return to cells:
`|phi_phys - s h'_d n_d + kappa' h'_d^2 (1 - n_d^2)/4| / (h'_d |n_d|)`, which at `H = I` is the
shipped expression term for term. `pcGfmRow` is a function of `theta` — unchanged.

**V5.4 The scalar/energy operator gets per-axis constants** (`flow/src/scalar_transport.hpp`:
`scalarBuildDiffusionOpen`, `scalarMaskGfm`, `scalarMaskGfm2`, and the variable-`k` variant). The
diffusion coefficient of a face normal to axis `a` becomes `D'/h'_a^2` (resp. `k'_f/h'_a^2`),
the scalar twin of Phase 2's U10 `beta_b = mu'/h'_b^2`: `tw = (D/hx2)*ox(i)` with `hx2 = h'_x^2`
(`D/1.0 == D`, Rule B(1)); the GFM rows `k o (T_i - T_G)/theta` carry the same face factor; the
diagonal is the sum of the six face coefficients as today. Advection (`scalarBuildRhs`, FOU/Koren/
SOU on face Courant numbers) is index-space — unchanged. This file is claimed by **Phase 3** (it
is neither in Phase 2's list nor in `flow/src/vof/`); at rebase time check `git log --
src/scalar_transport.hpp` on flow's main and, if Phase 2 touched it, keep Phase 2's spelling.
The single-phase thermal gates (Rayleigh–Bénard, the scalar ctests) are part of §11 G0.

**V5.5 `V_cell` and the divergence source.** In reference units `V'_cell = h'_x h'_y h'_z`
(`vCellR` on `UnitScales`, `1.0` today). The plane shift removes the *fraction*
`dV = mdot' A' dt' / (rho'_l V'_cell)` (`pcRegressVolume(...) / vCellR`) and the source is
`S' = mdot' A' (1/rho'_g - 1/rho'_l) / V'_cell` (`pcDivSource(...) / vCellR`); with the operator
`mdot` both are `A`-free (`q dt'/(h_lv' rho'_l V'_cell)` and `q (1/rho'_g - 1/rho'_l)/(h_lv' V'_cell)`),
which is where the division is essential — `q` is a heat flow through unit-lattice faces carrying
the V5.4 per-axis factors, `dV` is a fraction of THIS cell. `x / 1.0 == x` keeps the cubic path
bitwise. The deposit search for the receiving pure-gas cell scores candidates with Malan's
collinearity weight on `(delta, n)` of V5.2 (physical), so on a stretched grid it still prefers the
cell that is nearest *and* most along the normal.

**V5.6 Boundary conversions for the phase-change setters (optional, last).** RELEASE_PREP §8.2
records that the phase-change layer "stays in solver-internal units" for 0.8.0. The conversions
are one table — `mdot' = mdot tRef/(rhoRef hRef)`, `k' = k tRef/(rhoRef c_p,Ref hRef^2)` with the
energy path's `rho c_p` closure, `h_lv` and `T` unchanged when `c_p` is given in the same energy
unit, `R_int' = R_int rhoRef hRef/tRef` (the inverse of `mdot'/T`) — and belong in
`refreshUnitDerived()` beside `sigmaCsf_`. Work order V5c below; it does NOT gate Phase 3's done-ness
and is skipped if the rebase onto Phase 2 leaves no time — say so in the report if skipped.

## 8. The block container (Part III)

`VofBlockSet::init(gs, per, rank, size, h = 1.0)` keeps `h_ = 1`: the block bounds
(`floor((c - r)/h_)`) are index-space. Each block's `VofCurvature` and the block CSF take the
same `VofMetric`/`h'_a^2` the global field takes (`curvProto.metric`, the `csfFaceForce` call at
~:606). Nothing else in the container (union-`max` colour, force scatter with UNPACK_SUM,
migration, checkpoint) knows a length. The block ctest battery (`vof_blocks`, `vof_blocks_mpi`)
is in G0.

## 9. What is added to `Solver::UnitScales` (appended at the END, never reordered)

    double hr[3]   = {1.0, 1.0, 1.0};  ///< h_a / hRef  (Phase 3; exactly 1.0 when isotropic)
    double hrMax   = 1.0;              ///< max_a hr[a]
    double hrMin   = 1.0;              ///< min_a hr[a]  (== 1 once Phase 2 sets hRef = min h)
    double vCellR  = 1.0;              ///< hr[0]*hr[1]*hr[2]  — V_cell in hRef^3
    bool   isotropic = true;           ///< all three hr equal (Rule B(3) guards read this)

`refreshUnitDerived()` fills them from `h[]`/`hRef` and pushes a `vof::VofMetric{hr}` into
`vofAdv_` (wetting/dynamic drivers), `vofCurv_`, `pcArea*_`, the block container and the
phase-change driver state. If Phase 2's diff already appends an equivalent field (it may want
`hr[3]` for U10), keep Phase 2's name at rebase and drop ours — never carry two copies.

## 10. Work orders (one commit each; the message names the gate and its numbers)

Files: `core/include/peclet/core/vof/{plic,curvature,wetting}.hpp` (kernels — core is Phase 3's
alone), `flow/src/vof/*` (drivers), `flow/src/scalar_transport.hpp`, the VoF region of
`flow/src/flow_ibm.hpp`, `flow/tests/kokkos/test_units.cpp` (extended, never a new file for the
units gates) and the existing `test_vof_*.cpp` for the kernel gates.

| WO | what | gates |
|---|---|---|
| **V0** | `VofMetric` in core `plic.hpp`; `UnitScales` fields of §9; `refreshUnitDerived` plumbing; `s(m)`, `plicAreaMetric`, the physical-normal helper; no kernel behaviour change | G0 |
| **V1** | the PLIC/advector identity gate K1 (test-only) + the stretched `sphereCellFraction` helper | K1, G0 |
| **V2** | `hfPatchKappa(h, g, d)`, the physical-frame `pvFitAdd`/`ptFitAdd`, `dW` scaling, `VofCurvature::metric`; `VofInterfaceArea` in step | K2, K4, G0 |
| **V3** | `csfFaceForce(…, hGrad)` at the four sites; `capillaryDt(…, min h')`; step guards | G0 (S2–S4 after Phase 2) |
| **V4** | `vofWettingPlane(…, g)`; the walk direction; the dynamic driver's `Delta`, `U_cl`, admissibility | K3, G0 |
| **V5a** | phase-change driver: physical offsets/normals/`phi_c`; `pcGfmThetaK` stretched form; `V_cell` in regress + source; deposit weight | K5, G0 |
| **V5b** | `scalar_transport.hpp` per-axis face constants (diffusion + GFM rows) | G0 (S5 after Phase 2) |
| **V5c** | (optional) phase-change setter conversions | `units_phase_change` scale-invariance at `extent = 1e-2·cells` |
| **V6** | after Phase 2 is on main: rebase, `units_vof_aniso` in `test_units.cpp` (S1–S5), full battery, push | S1–S5, G0 |

Order of execution: V0–V5 are implementable and kernel-gated (K-gates) **now**, before Phase 2
lands, because the kernels and the standalone drivers (`WyAdvector`, `VofCurvature`,
`VofInterfaceArea`, the wetting drivers) take the metric directly. The Solver-level gates S1–S5
need Phase 2's momentum/pressure operators and the relaxed `setPhysicalDomain` assert and run
only after the rebase (V6). Do not relax that assert; do not touch `mac_*.hpp`,
`cut_cell_ibm.hpp` or the momentum/pressure/domain-BC regions of `flow_ibm.hpp`.

## 11. Acceptance gates and the numbers they must hit

**G0 — bit-identity and the isotropic gates (every commit, before every push).** With
`extent=None`: flow `tests/kokkos` OpenMP and CUDA (36 + 3 units gates), `tests/kokkos_mpi`
np = 1, 2, 4 (103), `tests/regression/sdflow_regression.py` (never `--update`; every metric +0.00 %,
every `p_iter_tot` and step count equal), the five verify scripts byte-identical
(`get_u/v/w/p` → `.npz`, `np.array_equal`), and the three isotropic units gates UNCHANGED —
`units_identity` bitwise, `units_scale_invariance` ≤ 1e-13, `units_vof_sigma` with its recorded
numbers (exactness `max|u|` 1.9e-17-class, Young–Laplace `dp = 0.25 = sigma·kappa`,
`capillary_dt = 3.989422804e-01`, computed-curvature currents bitwise after rescaling). MPI np = 1
bit-exact to single-rank on OpenMP. A changed digit anywhere is a bug in the change.

**K1 — PLIC/advector identity on stretched cells** (`test_vof_advect.cpp`, new case). The
LeVeque deformation field and the Zalesak disk run twice on the same index face velocities with
`VofMetric = (1,1,1)` and `(1, 0.5, 2)`: the colour fields are **bitwise equal** at every step and
`sum eps C` is conserved to **≤ 1e-15** relative (the WY floor). Plus `planeCellFraction` on
1e5 random `(m, alpha, h)` with `h_a ∈ [0.25, 4]` against the independent oracle of
`test_vof_plic.cpp`: **≤ 1e-14** absolute.

**K2 — curvature on stretched cells** (`test_vof_curvature.cpp`, new ladder). Exact-fraction
sphere (physical-space fractions, sub-sampled 24³) of radius `R = 8 h_min` on `h' = (1, 0.5, 2)`,
min-axis counts 16 → 32 → 64: `L1(|kappa' - 2/R|)` order **≥ 1.8** and max order **≥ 1.5** (cubic:
2.26 / 1.86); `noEstimate = 0` on every rung; the tier census printed. Plane through the stretched
lattice at three random orientations: `|kappa'| ≤ 1e-13` (cubic 1.5e-14). Cylinder along the LONG
and along the SHORT axis, `R = 8 h_min`: `|kappa' R - 1| ≤ 5e-3` (cubic 2.8e-3). The
cubic ladder's printed L1/max digits are unchanged (G0).

**K3 — wetting idempotency on stretched cells** (`test_vof_wetting.cpp`, G0a/G0e stretched). A
plane meeting a flat axis-aligned wall at `theta = 30, 60, 90, 120, 150 deg` (physical), wall
normal along the short axis and along the long axis of `h' = (1, 0.5, 2)`: the band fractions the
fill writes reproduce the exact continued plane to **≤ 1e-14** (cubic 1e-15); `cosApp` returns the
imposed angle to **≤ 1 deg** where the fluid-only azimuth is defined.

**K4 — interfacial area on stretched cells** (`test_vof_phase_change.cpp`). A plane through
stretched cells: `plicAreaMetric` and the joined marching-tetrahedra sheet equal the analytic area
to **≤ 1e-13** relative. A sphere `R = 8 h_min` on `(1, 0.5, 2)`: the sheet's `sum A` within
**0.5 %** of `4 pi R^2` (cubic: +0.04 % at the P3d gate; the stretched number is reported next to it).

**K5 — the one-sided gradient on physical distances** (`test_vof_phase_change.cpp`). An exact
plane at 3 orientations on `(1, 0.5, 2)` carrying `T = T_G + G phi_phys` in the gas: the fitted
`dT/dn` equals `G` to **≤ 1e-12** relative (the linear fit through the interface value is exact for
a linear profile iff the sample distances are the physical ones — with index distances it is off by
`s(m)`, up to a factor 2 here, which is what makes this the discriminating gate).

**S1 — conservation and shape under a stretched translation** (post Phase 2; `units_vof_aniso`
(a) in `test_units.cpp`). A sphere `D = 16 h_min` translated at physical velocity `(1, 1, 1)`
through one period of a periodic box of extent `(32, 16, 64) h_min` on `32³` cells: `sum C V`
conserved to **≤ 1e-15** relative; `L1` shape error against the exact fraction **≤ 1.5×** the cubic
run at the same `h_min`.

**S2 — Laplace pressure on stretched cells** (`units_vof_aniso` (b)). Static droplet
`R = 8 h_min`, box `32³` cells at `h' = (1, 0.5, 2)`, `rho = mu = sigma = 1` in a 1e-2 length unit:
with the constant curvature, `max|u| < 1e-14 · sigma kappa/mu` (the exactness identity, as
`units_vof_sigma`) and `dp = sigma kappa` to **1e-9** relative; with the computed curvature the
spurious `Ca ≤ 2×` the cubic value at the same `h_min` (cubic: 2.5e-4 at `D/h = 8`, 5.9e-5 at 16).

**S3 — capillary wave on stretched cells** (`units_vof_aniso` (c); the `gate_wave` setup of
`tests/study/vof_surface_tension.py`). The wave along the short axis and, separately, along the
long axis of a quasi-2D box stretched `(1, ·, 0.5)` in `x–z`, 32 and 64 cells per wavelength along
the wave axis, exact viscous two-fluid reference (`vof_capillary_references.wave_mode`): the
frequency error stays inside **[-6 %, 0 %]** and within **1.5 %** of the cubic run at equal cells
per wavelength (cubic: -2.1 … -3.7 %).

**S4 — oscillating drop** (`units_vof_aniso` (d); `gate_lamb`, mode 2, `phi = 0.8 %`,
ratio 100). On `(1, 0.5, 2)` at equal `h_min` the mode-2 frequency is within **2 %** of the cubic run
(cubic: -6.3 … -7.0 % of Lamb, recorded as a measured deviation, not a pass — the anisotropic gate
is relative to it).

**S5 — the two-phase thermal pair** (`vof_phase_change` P1 Stefan and the P3 Scriven probe on
`(1, 0.5, 2)`, energy order 2): the Stefan front position within **2×** the cubic error at N = 64
(cubic +0.20 % at N = 256 class); Scriven `beta_eff` within **3 %** of the cubic run at equal `h_min`.

## 12. Dependencies, ownership and the things that could move a bit

- **Phase 2 first.** `setPhysicalDomain`'s isotropy assert, `hRef = min h`, the per-axis pressure
  weight and the momentum fold are Phase 2's; S1–S5 cannot run before it is on main. K1–K5 and G0 can.
- **`flow/src/scalar_transport.hpp`** is claimed by Phase 3 (V5b). Check at rebase.
- **Rule B(3) guards** are expected in exactly zero places on the VoF half (every factor is a
  ratio or a same-double quotient); if the implementer needs one, it is listed in the commit message.
- **STOP conditions** (plan §9.6 and the brief): a G0 digit moves and two focused attempts do
  not find it; an isotropic units gate moves; the regression suite's iteration counts move at
  `extent=None`; MPI np = 1 stops being bit-exact; or a kernel would need a non-constant
  metric (none is expected — §2 is the proof).
