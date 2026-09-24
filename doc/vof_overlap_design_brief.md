# Design brief (Fable) — robust surface tension for OVERLAPPING markers in the block VoF container

Author: Opus session 2026-09-24, branch `vof-overlap` (worktree `suite/flow-vof-overlap`, from flow
main 0ae29d6). State file: `doc/vof_overlap_STATE.md`.

## 1. The question

What should the block-VoF surface-tension pipeline do so that two (or more) interpenetrating
markers can never produce an unbounded face force? Specifically: which combination of (a) a
curvature admissibility rule, (b) per-marker debris removal and (c) an explicit contact/overlap
rule is the right production design — stated as a rule precise enough for Opus to implement,
with the conservation, bit-identity and stability gates it must pass. Secondary: is anything
beyond (a)+(b) needed for the force itself (the WO-W4 "one force per face from the union" rule),
given the evidence in §6 that the summed force is statically balanced?

## 2. Why Fable

The previous diagnosis (WO-W3 §7, and the WO-W4 design it produced) blamed a *balanced-force*
defect — UNPACK_SUM face force against a `max`-union colour. Local measurement today says the
proximate cause is something else (§6). The fix touches the V4 balanced-force CSF and the V3
curvature cascade — both validated to machine-zero / order-2 gates — and every future bubbly-flow
case (the next deliverable is a bubble-column swarm example and a TBFsolver benchmark) depends on
it. A wrong rule here either reintroduces the blow-up at the next collision or silently degrades
the curvature accuracy of every non-colliding run.

## 3. Current state (code)

**Block CSF** (`src/vof/block_container.hpp`, `buildCsfForce`): per master block, over its inner
box, face force from the block's OWN colour and OWN curvature:

```cpp
const double dC = cv(i) - cv(i - strd);
double f = 0.0;
if (dC != 0.0) {
  double kf = 0.0;
  vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
  f = vof::csfFaceForce(sig, kf, dC, wc);        // sigma * kf * dC / h
}
ff(i) = f;
```
then `exch_->scatterForceSum(...)` sums all markers' face forces into the global face-force patch
(UNPACK_SUM); `Solver::addCsfRhsBlocks` adds `rscale * fb` to the momentum RHS (rscale = the same
1/rho_f the projection uses). Union colour for rho/mu: `C = max_k C_k` (UNPACK_MAX), no smoothing.

**Face curvature** (`src/vof/surface_tension.hpp`):
```cpp
bool csfFaceCurvature(k0, b0, k1, b1, kf):   // branch-flag "defined" = HF / HF-mixed / HF-fit / PV / PV-reduced
  both defined -> kf = 0.5(k0+k1); one defined -> that one; none -> 0 (force 0)
```
**Interfacial predicate of the cascade** (`src/vof/curvature_field.hpp`):
`vofIsInterface(c, eps) = eps < c < 1-eps`, with `interfaceEps = csfInterfaceEps_ = 1e-8`
(V4 wisp guard; set on the block prototype by `enable_vof_block_csf`). Every cell passing it gets
a curvature from the cascade: HF (7-column) → HF mixed direction → HF fit → PLIC-volumetric 5³
paraboloid fit → reduced fit → "no estimate". **There is no clip on |kappa| anywhere.**

The block's WY advection: per-marker, exact telescoping conservation; round-off wisps discarded
at re-centring only if outside the new box (ledger `discarded`).

## 4. Constraints and invariants

- Marker volumes conserved to ~1e-12 relative (the container's selling point); any removal of
  colour must be exactly redistributed or explicitly ledgered.
- Non-overlapping runs: the block path is bit-identical to the global-field path on Hysing
  (W2 gate 1) and to itself across np 1/2/4 (W0); G1/G7 of WO-W4 ask that every existing block
  ctest stays bit-identical when no overlap occurs. A rule that changes non-overlap numerics must
  say so and justify it (and then the Hysing-block == Hysing-global gate must be re-stated).
- Balanced force: static droplet/bubble at machine-level parasitic currents (V4 acid test).
- Kokkos device-only production path; MPI: a block has one master rank, markers of a pair may
  live on different ranks — a rule that needs BOTH markers' colour on one rank needs an exchange
  (the pair census on `vof-w4` built one; cost matters).
- Cell units internally; staggered only for the block CSF; all-fluid only.
- channel_18 numbers (cell units): D = 10 cells, rho_g/rho_l = 0.1, mu_g = mu_l = 0.533,
  sigma = 320, Brackbill dt ~ 1.65e-2, run dt ~ 2e-3 (CFL-bound, Courant 0.0625).

## 5. Settled / not open

- No numerical coalescence by default (multiple markers; coalescence only as an explicit model).
- VoF stays geometric PLIC/WY/HF (VOF_PLAN §1); no CLSVOF, no smoothing of the colour that the
  curvature sees.
- TBFsolver is the reference implementation of the pattern (Cifani et al. C&F 2018).
- W4's parked branch `vof-w4` (f29c8e7: union-colour CSF, pair census, coalescence/breakup models,
  32/33 battery, still died at 1.53 turnovers) is NOT merged; this design may reuse its ideas.

**Genuinely open:** the rule set in §1, and whether union-based force assembly (WO-W4 item 1) is
needed at all.

## 6. Evidence (measured today) — the high-value section

Instrument: restart of W3's healthy step-10000 checkpoint of `channel_18` (128×80×64, 18 markers)
on an RTX 5080 with new per-marker diagnostics (`diagnostics.vof_block_kappa(id)`,
`diagnostics.vof_block_force(c)`), rolling state dump.

1. **Overlap is present and harmless for 700 steps.** At step 10000 markers 2/3 already share 6.2
   cells of volume (14 cells with C2+C3 > 1, max sum 1.08); pairs 1/2, 3/4, 12/13, 12/17 come and go.
   Nothing happens.
2. **The blow-up is ONE step**, 10705 → 10706: max|u| 34.65 → 135.5 at v(26,41,12) (the v-face
   between cells y=40 and 41), inside the 2/3 pair's boxes. The global face force there is
   **−405** (typical interface faces −20…−120).
3. **Decomposition of that force**: marker 3 (the real interface, C3 = 0.826 | 0.029, kappa 0.37 /
   0.33) contributes ≈ −89. Marker 2 contributes ≈ −306: its colour at the two cells is
   **7.25e-3 | 2.08e-5**, kappa **0.294 | 273.15** (1/cell; the bubble's true kappa ≈ 0.4). Face
   kappa 137, dC −7.2e-3, sigma 320 → −316.
4. **Marker 2 carries DEBRIS**: 6 cells with 1e-8 < C < 1−1e-8 and no C > 0.5 cell within the 5³
   neighbourhood (TBFsolver's `resetFragments` criterion), total volume 0.07, max C 4.2e-2, lying in
   marker 3's interface band. Marker 13 (the 12/13 pair) carries 5 such cells (volume 4.5e-3,
   |kappa| up to 4.6). **At step 10000 NO marker has any fragment**; by 10702 exactly the two markers
   of overlapping pairs have them. So the debris is generated where one marker's interface enters
   the other's interface band (the lens rim), and it is fatal only through the curvature the
   cascade assigns to near-empty cells (the 2.08e-5 cell passes `interfaceEps = 1e-8`).
5. **Static pairs are balanced.** Two spheres R = 5 at rest, no gravity, ratio 10, sigma 320,
   400 steps at 0.25 Brackbill: parasitic max|u| peak / end —
   single 0.131/0.047; d=14 0.131/0.048; d=10.5 0.231/0.093; d=10 0.133/0.127; d=9 (1-cell
   overlap) 0.162/0.049; d=8 (2 cells) 0.135/0.055; d=6 (4 cells) 0.217/0.049. Volumes 5e-15.
   So SUM-vs-MAX is NOT a static balanced-force defect (as expected: Σ_k σ κ_k ∇_f C_k is the
   discrete gradient of σ Σ κ_k C_k for constant κ_k, and 1/ρ_f multiplies force and pressure alike).
6. **Controlled collision (Couette, U=±16, offset b=6, We≈0.7)**: the pair collides, overlaps up to
   ~5 cells, orbits and separates; 2800 steps stable, max|u| flat at 15.67, volumes exact. A gentle
   collision alone does not reproduce the failure. (A harder one, U=±40, We≈4.5, is running.)
7. **Raising the wisp threshold only MOVES the failure.** Same restart with
   `set_vof_interface_eps(1e-3)` / `(1e-2)` (cells with C < eps get no curvature): both pass 10706,
   then die at **11213** / **11331** (t u_tau/h 1.586 / 1.597) — a different place, v/w faces near
   (104,41,54), inside the 12/17 and 12/13 pairs. There, marker 13 has grown **24 fragment cells,
   volume 0.16**: an isolated blob with C up to 0.081 and NO full cell within 2 (C slice shows a
   3×3 speck: 3e-3 / 8.1e-2 / 3.4e-3 surrounded by 1e-5…1e-7), curvature **−428/cell**; the global
   face force at the site is **−10 810** (max over the field; typical ≤ 120). Same mechanism, debris
   above any sensible eps. So the fix must bound the curvature/force of debris AND/OR remove the
   debris; a threshold on C alone cannot.
8. **TBFsolver (the reference, runs 20 turnovers on this case)** does exactly SUM-force/MAX-colour
   too, and has NO collision/overlap model. Its guards that peclet lacks: (i) per-cell
   `|kappa| <= 1/min(dx,dy,dz)` clip (`VOF.f90:2201`); (ii) `resetFragments` every 10 steps —
   zero mixed cells with no full cell in the 5³ stencil (`VOF.f90:1272`, volume NOT redistributed);
   (iii) `resetSatellites` — delete the smaller of two connected components; (iv) per-sweep clip of
   C to [0,1]; (v) face kappa averaged over MIXED neighbours only (mixed = eps 1e-12, and a full
   cell touching an empty one counts as mixed); (vi) rho, mu from a SMOOTHED union (2 passes
   cell→vertex→cell), force from the sharp marker; (vii) D/Δ = 15–20 (peclet 10), CFL 0.1.

Rejected / already tried (W3/W4): reducing dt 4× (failure time unchanged); W4's union-colour CSF +
pair census (still died at 1.53 turnovers — consistent with §6: it did not touch debris kappa).

## 7. How the answer will be verified

- channel_18 restart from step 10000 past 10706 and on (local, RTX 5080, ~0.3 s/step); then a
  fresh chain to ≥ 5 turnovers (ideally the full 20 for the TBFsolver statistics comparison).
- Static pair sweep (§6.5) unchanged; single static bubble parasitic current unchanged.
- Marker volumes to 1e-12 relative (or an explicit, reported ledger if removal is chosen).
- Existing ctests: `vof_blocks` (+MPI twin np 1/2/4), `vof_surface_tension`, the Hysing
  block==global gate — bit-identical, or a stated reason why not.
- New: a deterministic debris test (hand-seeded marker with a fragment of C = 1e-5…1e-2 beside a
  full interface of another marker; the force must stay O(σ/R)).

## 8. Deliverable

A design note `flow/doc/vof_overlap_design.md` with: the diagnosis accepted/amended; the chosen
rules (exact predicates, thresholds with their justification, where in the pipeline they run,
MPI/exchange needs); conservation treatment; what changes bitwise and for which runs; the gates
above made concrete; work orders for Opus. Also a verdict on WO-W4 item 1 (union-based force).

## 9. Out of scope

Coalescence/breakup models (W4 items 3–4); the bubble-column example and the TBFsolver benchmark
set-up (Opus does those after the fix); anisotropic cells; collocated block CSF; cut-cell blocks.
