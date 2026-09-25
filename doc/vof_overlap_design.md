# Design — bounded surface tension for overlapping markers in the block VoF container

Fable design note, 2026-09-24, branch `vof-overlap`. Answers `doc/vof_overlap_design_brief.md`.
Implementer: an Opus engineer who has seen neither the brief nor the conversation; every number
and predicate needed is in this note.

## 1. Problem and scope

Two markers of the block container (`src/vof/block_container.hpp`) that interpenetrate produce,
some hundreds of steps later, a one-step velocity blow-up. The container must never produce an
unbounded face force, whatever the markers do, and it must keep every marker's volume exact.

In scope: (a) a curvature admissibility rule, (b) per-marker debris removal with exact volume
return, (c) the pipeline placement, MPI treatment, bit-identity statement, gates and work orders,
(d) a verdict on WO-W4 item 1 (union-based force assembly).

Out of scope (unchanged from the brief): coalescence/breakup models, a contact model, the
bubble-column example, the TBFsolver benchmark set-up, anisotropic specifics beyond the metric
already in the code, collocated block CSF, cut cells, the single-field path's satellite policy.

## 2. Diagnosis — accepted, with one refinement and one hypothesis

**Accepted.** The proximate cause is the curvature the cascade assigns to *debris*: mixed cells
of a marker with no body of that marker near them. On such cells every height-function column
fails (no pure end within the column) and the PLIC-volumetric fit is made from a handful of
corner-sliver polygons of total area ~0; the paraboloid through them is arbitrary (measured
|κ| = 273 and 428 per cell against a true 0.4). The face force `σ κ_f ΔC/h` then exceeds the
physical one by 3–100× (measured −306 and −10 810 against a typical ≤ 120) and acts on the light
phase. The SUM-force / MAX-colour pairing is *not* the cause: static overlapping pairs are
balanced at every depth (brief §6.5), and the algebra says why (with constant κ_k the summed
force is the gradient of `σ Σ_k κ_k C_k`, which the projection annihilates). Raising
`interfaceEps` cannot fix it because debris exists at C up to 0.08 (brief §6.7).

**Refinement — how debris is made (hypothesis, cheaply testable).** The segment of marker A's
surface that lies inside marker B's body is advected by B's *internal* gas circulation, while the
rest of A moves with A. Two colliding bubbles have different velocities, so that segment is
sheared off at the lens rim into slivers thinner than a cell, which WY advection fragments. When
the pair separates, the entrained colour of A stays behind inside B — exactly the "residual wisps
of one marker inside the other, 0.01–0.03 cells" that kill the shear reproducer *after*
separation (brief §6.6). Prediction: A's debris cells sit where `C_B ≥ 0.5` (or in B's band) and
their displacement per step tracks B's centroid velocity, not A's. The existing channel_18 dump
(step 10702, markers 2/3) can confirm this without a new run. Consequence for the design: debris
is intrinsic to independent markers in relative motion under SUM/MAX — it is not a bug that a
better advector removes — so the container needs a *standing* hygiene rule, not a one-off fix.
(A contact model that limits overlap depth, W4 item 2, would reduce the rate; out of scope.)

**Two side observations for the record, neither causal here.** (i) `block_container.hpp` says in
three places that the ghost ring is "pure gas" while `C = 1` is the bubble (the shear reproducer
sets `rho = rho_l + C (rho_g − rho_l)`); the ghost is `C = 0`, the continuous phase — a doc slip,
fix the words in passing. (ii) TBFsolver counts a full cell touching an empty one as mixed so the
face gets a κ; peclet leaves that face an *orphan* (force 0, counted in `csfDiagnostics`). Not
this campaign's problem; noted so nobody rediscovers it as the cause.

## 3. Constraints and invariants

- Cell units internally; `hRef = min h`, so the finest spacing is 1 in index units; the block
  metric is `VofMetric` (`minH()` available).
- Marker volumes exact: `Σ C_k` over the inner box changes only by advection flux and by what
  the ledger reports. Target 1e-12 relative on closed cases.
- Non-colliding runs bit-identical: every existing ctest (`vof_*`, MPI twins np 1/2/4, the
  Hysing block == global gate, `state_hash.py`) unchanged unless a counter proves a rule fired.
- Balanced force (static droplet at machine-level parasitic currents) untouched — neither rule
  touches a cell of a resolved static interface (§6, gate G0/G2).
- One master rank per block; a rule that needs another marker's colour needs an exchange. Both
  rules below are block-local: **no new exchange**.
- Kokkos device-only; the block state must stay decomposition-independent bitwise (W0), so
  anything that enters the *state* through a floating-point reduction must use a fixed summation
  order.
- No numerics-changing environment variables; new behaviour behind per-solver setters with the
  old behaviour reachable (`enabled=False`).
- `core/` is not touched this session: the clip lives in flow's cascade driver
  (`src/vof/curvature_field.hpp`), not in `core/.../curvature.hpp`.

## 4. The decision

**Two rules, with different roles.**

1. **Curvature clip (the safety invariant):** `|κ| ≤ κ_max = 1/Δ_min` on every cell,
   both VoF paths (block and global). This alone bounds every face force by
   `σ |ΔC| / Δ_min²` — the same scale the Brackbill capillary step
   `Δt < sqrt((ρ₁+ρ₂)Δ³/4πσ)` is built to hold, so a clipped debris force is by construction no
   worse than the steepest *resolvable* capillary wave. Applied to the brief's two sites:
   step 10706 face `κ_f` 137 → (0.294 + 1)/2 = 0.65, marker 2's contribution −306 → −1.5;
   step 11331 `κ` −428 → −1, force −10 810 → ≈ −26. Both sub-typical.
2. **Debris removal with exact volume return (hygiene):** per master block, every step, after
   advection: an interfacial cell whose 5³ fit stencil holds less than one cell volume of the
   marker's own colour is debris; its colour is zeroed and the removed volume is returned to the
   marker's own attached interface, weighted `C(1−C)`, in a fixed summation order. The marker's
   volume is unchanged to round-off; every quantity is ledgered.

Why both: the clip guarantees boundedness whatever happens; without it any future debris
generator (a new advector, a contact model, breakup) reopens the failure. Removal is needed for
*accuracy over long runs*: debris grew from 6 to 24 cells in 600 steps on channel_18 (brief
§6.4/6.7); each debris cell keeps exerting a bounded but spurious force (κ pinned at 1/Δ), sits
in the union colour as a density defect where it lies in liquid, and — if merely zeroed as
TBFsolver does — costs volume: at the measured ~0.07–0.16 cells per colliding pair per ~500
steps, a bubble in continuous collision over the 20-turnover benchmark (~1.4e5 steps) would lose
of order 5 % of its volume. TBFsolver's own `offset_volume_difference` (commented out at
`VOF.f90:395`) shows they saw that loss; peclet's selling point is that it does not have one.

**Rejected alternatives.**

- *Raise `interfaceEps`* — moves the failure (measured); debris exists at C = 0.08.
- *Clip only, no removal* — bounded but accumulating debris, growing spurious forcing and
  density defects; fails the long-run accuracy goal though not the stability one.
- *Removal only, no clip* — the fragment predicate is a size rule; anything it does not catch
  (a sub-cell drop of Σ > 1, a thin ligament) can still get an arbitrary fit κ. Safety must not
  depend on a hygiene predicate.
- *TBFsolver's predicate verbatim ("no full cell in 5³")* — deletes whole unresolved satellites
  (a 3-cell blob at C = 0.9 everywhere sums to ~24 cells = 5 % of a D = 10 bubble) and thin
  sheets of a strongly deformed bubble. With the volume-return step that would move 5 % of a
  bubble onto its rim in one pass. A *size* criterion (`Σ_{5³} C < 1`) catches the measured debris
  (0.07, 0.16, 0.01–0.03 cells; individual 5³ sums ≤ 0.2) with a ≥ 16× margin to any resolved
  surface cell (a surface cell of a sphere with R ≥ 2Δ has ≥ ~16 cell volumes in its 5³), and
  leaves satellites — whose policy is W4 item 4 — alone with a clipped, bounded κ.
- *Discard without return (TBFsolver)* — the volume argument above.
- *Connected-component labelling* — correct but not needed; the local 5³ sum is one 125-read
  scan per interfacial cell and needs no labels, no iteration, no exchange.
- *Every 10 steps (TBFsolver)* — the force must never see a debris cell; per step the rule costs
  less than the cascade and keeps each event's perturbation smallest.
- *Curvature from the union / union-based force (W4 item 1)* — §9.

## 5. The design

### 5.1 Curvature clip

Location: `VofCurvature` (`src/vof/curvature_field.hpp`), a final pass after `fallbackPass`,
over `listI_` in worklist mode and over the inner region in dense mode (same cells by
construction).

```
kappaMax  : double, index units; default 1.0 / metric.minH()  (== 1 when hRef = min h); 0 = off
for each inner cell i with csfKappaDefined(branch(i)):
    if (fabs(kappa(i)) > kappaMax) { kappa(i) = copysign(kappaMax, kappa(i)); ++clipped; }
```

- Exact no-op where it does not fire: the value is written only when it changes (never
  `k * min(1, kmax/|k|)`), so an unclipped field is bit-identical.
- `Stats.clipped` (long) joins the census (reduction of an integer — deterministic).
- Applied to the CELL value, so `csfFaceCurvature`'s average, `vof_curvature()` and every
  diagnostic see the clipped field; `|κ_f| ≤ κ_max` follows.
- Both paths: the global `vofCurv_` and every block's `curv_` via `curvProto` (same propagation
  as `interfaceEps`, `block_container.hpp:603`).
- Solver setter, diagnostics tier: `diagnostics.set_vof_block_kappa_clip(enabled, kappa_max=None)`;
  physical `kappa_max` (1/length) converted with the unit scales (`κ' = κ hRef`); `None` = the
  default `1/Δ_min`; `enabled=False` = the pre-clip cascade verbatim. Two justifications for the
  default, both resolution statements: `|κ| ≤ 1/Δ` is a sphere of `R ≥ 2Δ` (`D ≥ 4Δ`), the
  smallest the 7-cell height function can see at all; and the force bound of §4.

### 5.2 Debris predicate

Per master block `b`, over the inner box, reading the block's own colour `C` (ghosts are `0` by
the margin invariant; reads may equivalently be clamped to the inner box — state which in the
code, both give the same answer):

```
kDebrisVolume = 1.0            // cell volumes; a discretization constant like kPvHalf, not a setter
ieps          = curvProto.interfaceEps   // 1e-8: "interfacial" means the same thing as to the cascade

debris(i)  :=  vofIsInterface(C(i), ieps)  &&  Σ_{|ox|,|oy|,|oz| ≤ 2} C(i+o)  <  kDebrisVolume
```

`Σ_{5³} C < 1` implies there is no full cell in the stencil (TBFsolver's condition) and adds the
size bound. The stencil is exactly the PV fit's: a cell whose own fit data set holds less than one
cell of colour cannot carry a meaningful paraboloid. The predicate is evaluated on the colour
*before* any removal in this pass (two-phase: mark, then act), so the result does not depend on
traversal order.

### 5.3 Removal and volume return

```
1. mark: parallel_scan over the inner box in index order ->
         listD (debris cells), nD;   listA (attached interfacial cells: interfacial && !debris), nA
   if nD == 0: return                        // the common case; nothing else runs -> bitwise no-op
2. dV   = Σ_{i in listD} C(i)     in list order, sequential (single-thread kernel or host)
   W    = Σ_{i in listA} C(i)(1-C(i))   in list order, sequential
   VA   = Σ_{i in listA} C(i)     (same pass)
3. guard: if VA < 1.0 (the marker has no attached interface worth a cell) -> do NOT remove,
          stats.debrisUnresolved += nD, return          // clip still bounds it
4. act:  for i in listD: C(i) = 0.0
         for i in listA: d = dV * C(i)(1-C(i)) / W;  Cn = C(i) + d;
                         if (Cn > 1.0) { lost += Cn - 1.0; Cn = 1.0; }  C(i) = Cn
   (`lost` accumulated in list order; expected identically 0 — d ≤ ~1e-3 at realistic sizes.)
5. ledger (VofBlockStats, master-owned, migrated with the block like `discarded`):
   debrisCells (this step), debrisVolume (this step, = dV), debrisReturned (cumulative Σ dV),
   debrisLost (cumulative Σ lost), debrisUnresolved (cumulative count from step 3)
```

Conservation statement, exact by construction: `volume_k(t) + discarded_k + debrisLost_k =
volume_k(0) + boundary flux_k` to summation round-off; on a closed box, to 1e-12 relative.

Why fixed-order sums: `dV/W` enters the *state*. A tree reduction's order is not guaranteed
across launches or ranks; index-order sequential accumulation over lists produced by a
`parallel_scan` is, so the W0 np-independence gate keeps holding bitwise *even when debris
fires*. The lists are a few hundred entries on a block; the cost is nil.

Setter: `diagnostics.set_vof_block_debris(enabled)`; default ON when block CSF is enabled
(`enable_vof_block_csf`), OFF otherwise (a kinematic-only block run keeps today's behaviour).
Rationale: debris is harmless without surface tension and the container is then bit-identical to
today; with surface tension it is the rule that makes the rating hold.

### 5.4 Pipeline placement (one step)

```
step n head:  updateVofCurvature -> per master block: cascade -> CLIP (5.1) -> buildCsfForce
              -> scatterForceSum -> addCsfRhsBlocks -> momentum -> projection
after projection: advectVof -> VofBlockSet::advect:
              gatherFaceVel -> per block: WY advect -> DEBRIS (5.2/5.3) -> recentre
              -> syncTable -> (reassign) -> scatterColourMax -> measure
```

Debris runs before `recentre` so the bubble box excludes it and the returned volume is never
double-counted with recentre's `discarded`; the curvature of step n+1 and the union colour that
sets ρ/μ at n+1 both see clean colour. No new exchange anywhere: the clip is per cell, the debris
step reads and writes one block on its master.

### 5.5 What changes bitwise, and for which runs

- Clip: only cells with `|κ| > 1/Δ_min` — by definition unresolved (`R < 2Δ`). Expected count on
  every existing gate: 0 (resolved spheres, `R ≥ 4` cells at the coarsest ladder rung, κ ≤ 0.5).
  If a gate does fire it (a pinch-off tail on Hysing case 2 is the only candidate), that case
  changes only there; re-baseline with the count and pre-clip max|κ| in the commit message.
- Debris: only when `nD > 0` on a block with CSF on; then debris cells → 0 and attached
  interfacial cells move by ≤ ~1e-3 each. Zero on every non-colliding existing gate (no cell of
  a gated case lies detached above 1e-8 — the V4 finding).
- The proof of no-op is the counters, not a hash alone: G0 asserts `clipped == 0` and
  `debrisCells == 0` per test, and the hashes then follow.
- Hysing block == global stays exact: the clip is in the shared cascade and debris never fires
  on a single non-colliding marker.

### 5.6 Phantom interfaces and the capillary time step (addendum, 2026-09-24, after review)

**The premise (measured by the reviewer).** Under SUM/MAX the part of marker A's surface inside
B's body is a *gas–gas* interface with tension σ: the union colour is 1 on both sides, so ρ = ρ_g
on both sides. The explicit capillary bound on it is Brackbill with `2ρ_g`, i.e.
`sqrt(2ρ_g/(ρ_l+ρ_g))` of the bound `step()` enforces: 0.43 at ratio 10 (channel_18 at 0.25 is
inside), **0.198 at ratio 50** (Loisy E1, the bubble-column benchmark, dt capillary-bound).
Static pair R = 5, σ = 320, ratio 50, μ ratio 1, 400 steps, peak max|u|: separated 0.142 at every
safety factor; d = 8 (2-cell overlap) 0.26 @0.15, 0.586 @0.25, 1.163 @0.40 — a transient that
grows with dt and crosses the single-bubble level where the gas–gas bound says, damped here only
because the gas carries the liquid's viscosity (viscosity relaxes the capillary bound, Denner &
van Wachem 2015; at μ_g/μ_l = 0.02 that relief is gone). So this is an explicit-stability
boundary of the model, not a debris effect, and neither rule of §4 touches it.

**Decision (revised after review): two things, at different levels.**

*Safety, default ON, model unchanged — phantom-aware capillary bound while an overlap exists.*
A SUM scatter of the markers' colour beside the existing MAX one gives `S = Σ_k C_k` on the
owner (same machinery and deterministic block order as `scatterForceSum`). Whenever block CSF is
on and any cell has `S > 1 + 1e-8`, `vof_step_limits()['capillary_dt']` and the `step()`
enforcement use `sqrt(2 ρ_min Δ_min³ / 4πσ)` instead of the `(ρ_l+ρ_g)` bound; otherwise nothing
changes. This is option (a) gated by the overlap itself. Cost: the rating factor
`sqrt(2ρ_g/(ρ_l+ρ_g))` (0.43 at ratio 10, 0.198 at ratio 50) on dt **only while a pair is in
contact** — for channel_18 at safety 0.25 it never binds; for the bubble column at ratio 50 it is
1.6× (0.25 → 0.16) during contacts, and at φ = 4.4 %, Bo = 2 Loisy report no close contact below
φ ≈ 5 %, so contact should be rare there (the overlap census from `S` measures it). The overlap is
detected at the end of step n and binds from step n+1 — a one-step lag on a transient that takes
tens of steps to grow (the 400-step peaks above); acceptable, stated. The drivers re-pick dt every
step; a fixed-dt driver at ratio ≥ ~31 (where 0.25 exceeds the factor) will see `step()` throw at
first contact, which is the same hard-boundary policy `step()` already applies to the base bound.

*Model, OPT-IN, default OFF — the "tent": the overlap volume as film liquid.*
`diagnostics.set_vof_block_overlap_density(enabled)`; with it the union colour that feeds ρ(C),
μ(C) becomes `C_eff = (S <= 1 + 1e-8) ? max_k C_k : max(0, 2 − S)`. Its justification is
**stiffness and inertia only, and it is a model**: across A's phantom surface `S` goes 1 → 2 and
ρ goes ρ_g → ρ_l, so the phantom is an ordinary liquid–gas interface under the base bound (a
thin-overlap band cell with `C_A = 0.5, C_B = 1` gets `C_eff = 0.5`, the mean density every
real interfacial cell has), and the lens acquires the inertia and viscosity the missing film would
have. The balanced-force property holds for any ρ field (with constant κ_k the force is
`∇(σ Σ κ_k C_k)`, and `p = σ Σ κ_k C_k` solves the variable-density projection exactly), so the
static balance of brief §6.5 survives by construction.

**What the tent costs — stated honestly (my first draft had this backwards).** The real state
is gas `V_A + V_B` with a sub-grid film. MAX shows gas `V_A + V_B − V_ov`: it already has `V_ov`
*too much* liquid. The tent shows `V_A + V_B − 2V_ov`: it doubles that error. Per contacting pair
the mixture weight is off by `2 V_ov Δρ g` while the drivers' `⟨ρ⟩` uses the markers' volumes,
so a column feels a small net force during contacts (`V_ov ~ 5` of 523 cells ≈ 2 %, transient),
and each colliding bubble loses `V_ov` of buoyancy while in contact. It also changes collision
dynamics (harder deceleration on approach) — and the partner code of the benchmark, TBFsolver, is
MAX (with ρ smoothing), so a tent default would turn the peclet–TBFsolver comparison into a
comparison of two collision models. Hence opt-in, and a default flip only on measurement (G7),
recorded as a decision.

**Rejected.** (b) *always 2ρ_min*: pays the factor when no phantom exists. (d) *drop the phantom
force where covered by another marker*: the masked force is no longer a gradient, so the rim
acquires unbalanced-CSF spurious currents of order `σκ/(ρ_g Δ)` in statics — the very thing V4
removed; it needs B's colour on A's master (a pairwise exchange); and without its own tension A's
inner part becomes a passive scalar in B's circulation, the debris generator of §2 made worse.
(e) *a fixed rating* `set_capillary_cfl ≤ 0.8·sqrt(2ρ_g/(ρ_l+ρ_g))`: correct but pays the factor
always; the gated bound is (e) applied only when it is needed, and it is what a fixed-dt driver
should set by hand.

**Measured vs assumed.** Measured: the dt-scaling spike and its threshold (above). Assumed and
gated: that the gated bound removes the spike at ratio 50 (G7a); that contact is rare in the
column (overlap census, G7c); for the tent, that it removes the spike at the base bound and does
not deepen or prolong contact (G7a/b). Not gated here: bounce statistics against a reference
(W4 G3 and the benchmark).

**Gate G7 (ratio 50, R = 5, σ = 320, the reviewer's static table).** (a) MAX + gated bound: at
requested safety 0.25 and 0.40 the run's dt drops to the phantom bound while d = 8 overlaps and
the peak is ≤ 1.1 × single (≤ 0.16); end values unchanged; separated cases bitwise; repeated at
μ_g/μ_l = 0.02. The same rows with the tent at the base bound, side by side. (b) Shear
reproducer, both configurations: completes, max|u| ≤ 18, the pair separates; report max overlap
volume, contact duration, debris ledger, and the number of steps the gated bound bound.
(c) No-op proof: `clipped`, `debrisCells`, the overlap census all zero on every existing ctest,
hashes unchanged; then the bubble column's overlap census over its first turnovers (fraction of
steps with any `S > 1 + 1e-8`). The default stays MAX + gated bound unless (a)/(b) show the tent
better on something that matters (peak, contact depth, debris rate) — then a recorded decision.

**WO-7** (after WO-3): `scatterColourSum` in the block exchange; the per-step overlap census
(`max S`, `Σ (S−1)⁺`, cell count) in `vof_block_stats()`; the gated capillary bound in
`vof_step_limits()` and the `step()` check; `C_eff` in `harvestVofBlockUnion` behind
`set_vof_block_overlap_density(enabled)` (default OFF = MAX verbatim); G7 runs. **Q7** (fact,
tent only): whether μ follows the same map or stays MAX (default: same map; if G7b shows
over-damped collisions, try ρ-only).

## 6. Gates (falsifiable)

- **G0 battery no-op.** `ctest -LE bench` all green on `build_cuda` (and host if available);
  `tests/regression/state_hash.py` VoF entries unchanged; every VoF ctest reports
  `clipped == 0` and `debrisCells == 0` (helper assertion added to the block and curvature
  tests). Any test with a non-zero count is listed with its pre-clip max|κ| and decided per §5.5.
- **G1 seeded-debris ctest** (`tests/kokkos/test_vof_blocks.cpp`, via
  `enable_vof_blocks_from_colours`): marker A sphere R = 5, marker B sphere R = 5 at centre
  distance 16 (no overlap), plus three cells of B's colour hand-painted in A's interface band:
  `1e-5, 1e-2, 3e-3` in a line. σ = 320, ρ ratio 10, u = 0.
  (i) clip ON, debris OFF: B's κ at the three cells satisfies `|κ| ≤ 1`, and at every face
  `|F_total − F_A-alone| ≤ σ · max|ΔC_speck| / Δ²` (`vof_block_force` vs. the A-only run).
  (ii) clip ON, debris ON: after one `step()` the three cells read exactly `0.0`,
  `debrisVolume == 1.3e-2 ± 1e-14`, `debrisLost == 0`, B's volume unchanged to 1e-12 relative,
  A's colour bitwise unchanged (A has no debris), max|u| after the step ≤ the A-alone value + 1e-6.
- **G2 static pair sweep** (brief §6.5, `vof_blocks_overlap.py static`): the peak/end table
  reproduced to 5 % with both rules on; report `clipped` and `debrisCells` per distance (expected
  0 everywhere — then bitwise).
- **G3 shear reproducer** (`vof_blocks_overlap.py shear`, ~20 min): runs to its full `T`
  (= 3·gap/rel ≈ 12.5) without throwing (today: WY-CFL throw at step 2910); max|u| ≤ 18
  throughout (today's plateau 15.67); each marker's volume to 1e-12 relative; cumulative
  `debrisReturned` per marker ≤ 0.1 cells reported; `debrisLost == 0`.
- **G4 channel_18.** (a) Restart from the step-10000 checkpoint to step 13000 (≈ 15 min at
  0.3 s/step): passes 10706, 11213 and 11331; max|u| ≤ 52 (1.5× the healthy 34.65); volumes
  1e-12; per-marker `debrisReturned` and `clipped` history reported. (b) A fresh chain to
  ≥ 5 eddy turnovers with the same bounds (≈ 35 000 steps ≈ 3 h on the RTX 5080; the full 20 for
  the TBFsolver statistics is a billing decision, §8).
- **G5 MPI.** The G1 scene with a linear shear velocity `u = γ(y − y_c)` and 20 steps, np 1/2/4
  with the ORB cutting between the markers: union colour, block volumes and the five ledger
  fields bitwise identical across np.
- **G6 curvature order ladder** (`test_vof_curvature` 16/32/64): bitwise (subsumed by G0; named
  because it is the accuracy contract the clip must not touch).

## 7. Work orders (commit-sized, dependency order)

- **WO-1 clip.** `kappaMax` + clip pass + `Stats.clipped` in `curvature_field.hpp`; prototype
  propagation in `allocateCsf`; `setVofBlockKappaClip` + binding on `diagnostics`; expose `clipped` in
  the curvature-stats dict. Acceptance: G0 counters and hashes; a unit assertion in
  `test_vof_curvature.cpp` that a hand-seeded detached cell at C = 1e-6 gets `|κ| ≤ 1` with the
  clip and `> 1` without (documents that the clip is live).
- **WO-2 debris census only.** Predicate (5.2), lists, the five stats fields, no removal;
  `vof_block_stats()` reports them. Acceptance: on the channel_18 step-10702 dump the census
  reads marker 2 ≈ 6 cells / 0.07 and marker 13 ≈ 5 cells / 4.5e-3 (the brief's counts; any
  difference is the size bound and is reported); the refinement check of §2 (debris of 2 inside
  3, moving with 3) run and its result written to the STATE file.
- **WO-3 removal + return + ledger.** 5.3 in `VofBlockSet::advect` before `recentre`;
  fixed-order sums; ledger migration in `serializeAux`; `setVofBlockDebris` + binding; default ON
  under `enable_vof_block_csf`. Acceptance: G1(ii), G0, G5.
- **WO-4 tests.** G1 ctest + its MPI twin (G5) registered; block-test helper asserting zero
  counters on the existing block scenes. Acceptance: the two new ctests green at np 1/2/4.
- **WO-5 reproducers.** G2, G3, G4(a) run; numbers into `doc/vof_overlap_STATE.md`; the
  CLAUDE.md scope sentence ("colliding markers are outside the rating") rewritten to the measured
  state; a decision-register entry queued for `../docs/decisions/flow.md` (umbrella is off-limits
  this session — write the entry text into this note's §10 and land it later).
- **WO-6 the chain.** G4(b) ≥ 5 turnovers; where it runs is the user's call (§8 Q6).

## 8. Risks and open questions (each with a default, so work proceeds unattended)

- **Q1 clip value** (fact). `1/Δ_min` vs `2/Δ_min`. Settled by WO-1's counter on the battery
  plus a pre-clip max|κ| histogram over healthy channel_18 steps (are there legitimate HF cells
  above 1 on strongly deformed bubbles?). Default `1/Δ_min` (TBFsolver's value, both resolution
  arguments of §5.1).
- **Q2 `kDebrisVolume`** (fact). 1.0 vs 0.5 cell volumes. Settled by the WO-2 histogram of
  `Σ_{5³} C` over debris cells vs attached cells on the dump; the two populations should be
  separated by more than a decade. Default 1.0.
- **Q3 return vs ledger-only** (user preference). Exact conservation is the container's selling
  point and the loss estimate is percent-level over the benchmark; default return.
- **Q4 clip on the global path too** (preference + fact). A curvature admissibility rule is not a
  block concept and the single-field path has the same degenerate-fit exposure above 1e-8.
  Default on for both; if a single-field regression baseline moves, re-baseline with the count.
- **Q5 per-step vs periodic removal** (fact, performance). Default per step; measure the block
  step time before/after on G3 (expected < 1 %).
- **Q6 where the 5- and 20-turnover chains run** (user: billing). Default the local RTX 5080
  overnight for 5 turnovers; the 20-turnover statistics run waits for the user.
- **Risk: debris rate.** If the returned volume per turnover is not small (say > 1e-2 of a
  marker), the model's SUM/MAX overlap is generating too much torn colour and a contact rule
  (W4 item 2) moves up the queue. The ledger makes this visible; the gate reports it.
- **Risk: the shear reproducer's second failure.** If G3 survives the wisps but dies later of
  something else, the new site is instrumented exactly as before (κ, C, force per marker) before
  any further rule is added.

## 9. Verdict on WO-W4 item 1 — union-based force assembly: REJECT

Not needed: the summed per-marker force is statically balanced at every overlap depth (measured),
the blow-up is debris curvature (measured), and W4's own union-force branch still died at 1.53
turnovers because it did not touch debris κ. Not wanted: the union colour `max_k C_k` has a
*crease* at the lens rim where two spheres intersect; a curvature of the union sees that crease as
a near-singular κ and pulls the rim — which is precisely the numerical-coalescence mechanism the
container exists to avoid. Per-marker κ on per-marker colour gives the rim no special force, and a
thin film between two *non*-overlapping bubbles correctly feels both Laplace pressures (two
interfaces, each with σ) — SUM is the physics, not an approximation to the union. The one thing
the union does define is ρ and μ, and it keeps doing that. No exchange of `(κ, ΔC)` pairs, no
owner-side assembly. W4 items 2–5 (contact census, coalescence and breakup as explicit models)
remain valid future work and are unaffected by this note.

## 10. Decision-register entry (to land in `../docs/decisions/flow.md` when the umbrella is free)

*Block VoF surface tension: per-step per-marker debris removal (interfacial cells with no
`C > 1/2` in their 5³ neighbourhood, §11) and sub-`wispEps` residue clearing (§13), both with exact
volume return to the marker's attached interface; a per-cell `|κ| ≤ 1/Δ_min` clip on the BLOCK
path only (§11); a gated gas–gas capillary dt bound while markers overlap (§5.6).* Rejected: union-based force assembly (W4 item 1 — statically unnecessary, would
reintroduce numerical coalescence through the rim crease), raising `interfaceEps` (moves the
failure), TBFsolver's unsized fragment predicate (deletes satellites and thin sheets), discarding
without return (percent-level loss over 20 turnovers). Evidence: `doc/vof_overlap_design_brief.md`
§6, this note, the G0–G5 numbers in `doc/vof_overlap_STATE.md`.

## 11. Addendum (Opus, 2026-09-24 14:30) — three premises measured false, three decisions

The implementer's gates (STATE, commit 94eeb34) falsified three premises; decided on the evidence,
reviewer to check at the end.

1. **Clip scope → BLOCK PATH ONLY** (Q4 reversed). On the single-field path the clip fires in
   existing gates on *legitimately* large curvature — under-resolved droplets (vof_curvature
   D/Δ 2.8/4.4: κ = 2/R > 1 by definition), contact-line cells (vof_wetting 2.9, vof_wetting_mpi
   8.7, vof_cutcell 1.6), the eps = 0 wisp ablation of vof_surface_tension P6 — and would move
   validated numbers. The single-field path has never shown the debris failure (one colour field
   cannot hold one marker's debris inside another). Default `1/Δ_min`, prototype-propagated to
   blocks only; `vofCurv_` unchanged.
2. **Debris predicate → `interfacial(C, 1e-8) && max_{5³} C < 0.5`** (replaces `Σ_{5³} C < 1`).
   Measured: the size criterion does not separate (fragments Σ 0.10–1.24, attached ≥ 1.19), and a
   connected-component rule would not either — the debris is mostly CONNECTED to its own body
   through trails of 1e-8…1e-3 (blob of m13 detaches only at a 1e-4 threshold; m2's tails not even
   at 1e-3). "No cell above ½ within two cells" is TBFsolver's fragment rule with "full" read as
   C > ½; on the dumps it selects exactly the census sets (m2 6/0.069, m13 5/4.2e-3; eps-1e-3 dump
   m13 24/0.162), and it selects NOTHING at the healthy step-10000 checkpoint (all 18 markers).
   §4's objection to TBFsolver's rule (it deletes a C = 0.9 satellite) does not apply: such a
   satellite has cells above ½ and is kept. What it removes is colour more than two cells from
   any cell the marker at least half fills — sub-cell sheets and tails with no resolvable
   interface. Volume return, ledger, fixed-order sums, placement and default exactly as §5.3–5.4.
3. **Phantom-bound trigger → `S > 1 + 1e-2`** (was 1 + 1e-8). Measured: WY residue of one marker
   inside another keeps S just above 1 on essentially every step after a contact (shear r10: bound
   on 5028/5907 steps, also after separation; channel_18: every step, dt 2.24e-3 → 1.76e-3), so the
   1e-8 trigger is a permanent 27 % dt tax for no stability benefit. A phantom interface worth a
   capillary mode needs a marker's interface band inside another's body, i.e. S − 1 = O(0.1–1).

Also: **box guard.** When a periodic bubble box would reach the full domain length, `vofClampBox`
snaps it to [0, L) and `recentre`'s copy by unwrapped index silently wipes the marker (channel_18
G4(a), ~step 12150, marker 2 → volume 0). Replace the silent path with a loud error naming the
block, its box and the likely cause (debris / a marker as large as the domain); debris removal is
the fix for the cause. And **G7 verdict: the tent stays opt-in** — at μ_g/μ_l = 0.02 it is worse
than MAX + gated bound (d = 8 peak 0.42–0.58 vs 0.32).

## 12. Addendum (Opus, 2026-09-24 16:10) — the block container's residue policy

G4(a) after §11 (STATE a18e891): removal ON dies at 10908 (marker 12 non-finite in cells holding
1e-33 residue; box 57×28×30 grown on 1e-12…1e-8 wisps); removal OFF dies at 12518 as before; a
diagnostic build with block wispEps = 1e-8 has no non-finite colour but EVERY marker's box grows
to 128 and is wiped. One inconsistency explains all three: the global path runs WY with the
wisp guard 1e-8 (`enable_vof`), the block advectors at 0 (they reconstruct 1e-33 residue → NaN);
and with the guard on, residue below it is frozen in place while `bubbleBox` still tracks
C > 1e-12, so a travelling bubble drags its box along its own frozen wake until it spans Lx.

1. **Block advectors take the global advector's `wispEps`** (the `enable_vof` value, 1e-8;
   copied in the same place `curvProto.pureEps` already follows `vofAdv_.wispEps`). The
   block == global gates compare against a global path that already runs at that value.
2. **The box tracks `C > max(bubbleEps, wispEps)`.** Residue below the wisp threshold never
   defines the box; what the next recentre leaves outside the box is dropped and ledgered in
   the existing `discarded` (never physical; measured ≤ 1e-10 per event on W0). `discarded` must
   now migrate with the block like the three debris fields (implementer's note: it does not).
3. **A periodic box that needs ≥ the domain length is snapped to [0, L) and copied WITH the
   periodic wrap** (source index mod L), which is lossless — a marker genuinely as long as the
   domain (a slug) is legal. No throw; `st_.fullAxis` counter for the census. The §11 guard
   branch (98f53ea) is superseded.
4. Accepted: G3 r10 `debrisReturned` 0.144 / 0.105 per marker exceeds the note's "≤ 0.1"; that
   number was a guess, not derived — the gate is "lost == 0 and completes".

## 13. Addendum (Opus, 2026-09-24) — residue is returned, not left to leak

§12 made channel_18 stable (10000→13000: max|u| 35.81, boxes ≤ 23×21×22, lost 0) but broke
exact volume: 1.5e-9 relative over 3000 steps, on every marker including non-colliding ones;
reproduced kinematically (one marker, uniform flow, 1500 advections: −3.18e-7, of which only
1.69e-7 ledgered as `discarded`). Cause: colour below `wispEps` no longer defines the box, so the
margin invariant no longer covers it; it reaches the ghost ring and is zeroed uncounted.

Rule: after every block advection, every cell of the inner box with `0 < |C| <= wispEps` is set to
exactly 0 and its SIGNED colour is added to the removed volume `dV` of the §5.3 pass (same
index-order lists and sums), which returns it to the marker's attached interfacial cells weighted
`C(1−C)`. Runs whenever the block `wispEps > 0` (kinematic block runs included — the leak is not
a CSF matter); debris removal itself keeps its §5.3 default (ON under block CSF). Ledger: a
separate cumulative `residueReturned` (signed) so debris statistics stay interpretable. With it no
colour exists below the box threshold, the margin invariant holds again, `discarded` should read
0 (reported), and volume is exact to summation round-off. Gate: the leak.py scene and G4(a)
marker volumes to 1e-12 relative.

## 14. Addendum (Opus, 2026-09-25 01:40) — review2 (Phase B re-check)

- **MUST-FIX closed.** `prepareVofBlocks` now calls `setMetric(u_.vofMetric())`: the only other
  push (`refreshUnitDerived`) fires before the set exists in the documented call order, so every
  anisotropic block run had used the unit metric in its advectors and CSF weights. Gate M
  (`test_vof_blocks_overlap`, a 1.5:1:2 box) fails with the line removed (verified) and passes with it.
- `finishVofBlocks` takes the prototype by ONE `copyTunablesFrom(vofCurv_)` and overrides only
  `interfaceEps`, `pureEps`, `kappaMax` (bitwise for every existing configuration: the newly copied
  fields are the metric and the two debug switches at their defaults). Gate P sets
  `debugForceFallback` so a copy that drops it is caught.
- Open (recorded, not fixed): a curvature setter called AFTER `enable_vof_block_csf` reaches only
  the structured cascade; re-enabling the phantom bound takes effect from the next overlap update
  (one step late); the review's #6/#8/#9/#10 notes (np-dependent SUM order with ≥3 markers per
  cell, stale ledger on release, ledger not checkpointed, global phantom trigger); the debris-pass
  cost on channel_18 is unmeasured.

## 15. Addendum (Opus, 2026-09-25) — a marker at a LOW wall leaked through the wall face

Bubble column (walls y = 0, NY): every marker with colour in the y = 0 layer drifted, both signs,
up to 1e-5 of its volume per step (ckpt t = 86, 300 steps: markers 2/3/9/10 at 1.1e-3, all others
≤ 4e-15; no ledger entry moved). Cause: the block's face velocity outside a non-periodic domain was
a zero-gradient clamp of index 0, but `uf(i)` is the HIGH face of cell i, so the LOW domain face is
index -1 — outside the domain. The clamp put `v(y = 1/2)` (up to 9 cells/time) on the no-slip wall;
the global field reads the solver's 0 there. The high wall face (index NY-1) is inside the domain
and was always gathered, hence the asymmetry. Fix (not a design change — it restores the header's
"every double the block consumes is the global field's"): the face-velocity gather also covers the
`ghost` cells beyond a non-periodic domain face, read from the boundary rank's patch ghosts
(`vofBuildPieces(…, outside)`); the clamp is gone. Inert unless a marker has colour in the low
wall layer (walled probes with boxes hanging out of both walls, and with 68 cells of colour on
the HIGH wall: all fields bitwise equal). Gate W (`test_vof_blocks`): sphere cut by y = 0, 240
steps, block |V/V0−1| 3.0e-15 and block == global field (2.6e-3 and max|d| 0.16 with the clamp
restored).
