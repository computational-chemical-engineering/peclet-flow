# Paper plan: "Steady-state attractor families in collocated cut-cell approximate projections"

*Living document — update the tracker (§4) whenever a result lands. Companion to
`collocated_invisible_subspace.md` (the technical note, LaTeX-ready) and
`collocated_stall_notes.md` (the chronological lab record).*

## 1. Thesis and target

**Claim.** A collocated cut-cell approximate projection whose pressure gradient and divergence
constraint couple *different pressure supports* possesses an affine family of velocity-stationary
states (with runaway invisible pressure), selected by march protocol; combined with a
boundary-row instability of the rotational pressure update, this masquerades as a
grid-convergence plateau. We identify both mechanisms, give a one-line stabilization valid
through $\Delta t = 10^{20}$, restore uniqueness by support-consistent gradients, and re-measure
the scheme's true convergence.

Venue: JCP (first choice) / Computers & Fluids. Format: full-length with a reproducibility
appendix (the refutation catalogue).

## 2. Outline (section -> source material)

1. Introduction: collocated projections for AMR/cut-cell IBM; the observed plateau; related work
   (ABC approximate projection; Rider filters; Almgren-Bell-Crutchfield; Guy-Fogelson 2005;
   Brown-Cortez-Minion; Majumdar/Choi/Yu momentum-interpolation dt-dependence; Timmermans/GMS
   rotational updates; Basilisk embed).
2. The scheme and its operators. (note §1)
3. The naive fixed point and design requirement C2. (note §2, Prop. 1)
4. Layer 1 — instability of the rotational update: measurements, GF anchoring, dt->infty Uzawa
   structure, ablation table (2a / PM I / filter / uniform w / mode 3), wall-banded update +
   stability at 1e20. (note §3)
5. Layer 2 — the invisible subspace: Prop. 2, attractor family, dt-cycling, P-runaway,
   staggered immunity. (note §4-5)
6. Support-consistent repair (embed-style corrector), collapse of the family (m1 -> 0, P-drift
   -> gauge-only, dt-independence restored), clean convergence study. [PENDING - the S1 arc]
7. Accuracy results: same-rung gap ladders (contaminated vs clean vs support-consistent),
   Z&H cross-check, order verdict. [PARTIAL]
8. Discussion: implications for AMR (core::amr::AmrFlow is this scheme), ghost projection's
   half-height plateau, methodology (same-rung instrument, a-priori exoneration, refutation
   catalogue).

## 3. Checklist

**In hand (see tracker):**
- [x] Instability measured (rates at dt=60/600, mean-free character, R-threshold)
- [x] Staggered dt-flat control (8 digits)
- [x] GF gradient-2 identification (verbatim stencil match)
- [x] Ablations: gradient-2a (refuted), PM I (stable/no-gain), filter (worse), uniform w
      (fails at 1e20), mode-3 adjoint (stable at 1e20, -12%)
- [x] Wall-banded blend stable + 11-digit convergence at dt=60/600/1e20
- [x] dt-cycling attractor proof (10-digit reversibility)
- [x] P-runaway instrumentation (0.217/500 steps, |P| 75x staggered)
- [x] Props. 1-2 with proofs (note)
- [x] Controls: TGV no-solid (col-stag order +2.00), flat-wall E1/E1b (stag==col identical),
      E2 hydrostatic (stag exact, gauge-exact O(h^6), plain O(h^1.6)), s=0.5 incidence pathology
- [x] Refutation catalogue from the ceiling investigation (march protocol, constraint operator,
      estimator, wall-band localisation) — `collocated_accuracy_ceiling.md`
- [x] set_dt staleness fix (needed for dt-switch experiments)

**Running:**
- [ ] Clean ladder R=16/24/32 + staggered refs (Snellius job 25914731) -> tracker on arrival

**Needed for submission:**
- [ ] Embed-NaN root cause + fix (S1 prerequisite)
- [ ] Support-consistent scheme validated: stability (Layer-1 check on the new pair, blend if
      needed), family collapse (m1 -> 0, P-drift gauge-only, dt-independence i.e. C2 across
      dt=6..1e20), THEN the convergence ladder (R=8..32, both beds phi=0.50/0.60) vs staggered
- [ ] Z&H clean cross-check (external anchor) with the final scheme
- [ ] Arnoldi/eigs verification of the boundary-block spectrum claim (Re lambda < 0) —
      assemble Sigma = D Pi A_u^-1 G on a small bed (N=48?), compute rightmost/leftmost
      eigenvalues for gauge-exact vs mode-3 vs embed pairs
- [ ] Minimal reproducer for the paper (smallest geometry showing the instability + family;
      candidate: few-sphere periodic cell or tilted channel — flat wall does NOT reproduce)
- [ ] phi=0.50 bed replication of the headline numbers
- [ ] Literature due-diligence pass (esp. Armfield & Street; Zang/Street/Koseff collocated
      fractional step; any prior statement of support inconsistency)
- [ ] Figures: growth curves, dt-cycling staircase, P-runaway, ladder plots, mechanism schematic
- [ ] Decide author list / acknowledge compute (Snellius grant tes24005)

## 4. Results tracker (append-only; cite artifact + commit)

| # | result | value | artifact | status |
|---|---|---|---|---|
| 1 | baseline growth dt=60 R=12 | doubling ~750-780 steps | `longmarch_R12_staleop.log` | final |
| 2 | baseline growth dt=600 | doubling ~82 steps; m1->12; k moves 0.25% | `longmarch_R12_dt600.log` | final |
| 3 | rate ratio dt600/dt60 | ~9.5 (rate prop. to dt) | ibid | final |
| 4 | stag dt-sweep | dt 6 vs 60 identical to 8 digits | `dtsweep_R12_stag_*.json` (peclet-examples 0a04990) | final |
| 5 | col dt-sweep (contaminated) | +0.125%/+0.180% (6->60->600) | `dtsweep_R12_col_*.json` | final |
| 6 | gradient-2a | exponential, ~0.85x rate | `lm192_2a.log` | final |
| 7 | PM I | stable @dt60; no dt->infty gain | `lm192_pmI.log` | final |
| 8 | rot-filter (one-sided S) | destabilizes (m1 7e-2 @2000) | `lm192_rotf.log` | final |
| 9 | uniform w=0.3 | stable@60 / marginal@600 / blowup@1e20 (~230 steps doubling) | `lm192_w03*.log` | final |
| 10 | wall-blend w0=0.5 | stable + 11-digit k @60/600/1e20 | `lm192_wall05_dt*.log` | final |
| 11 | mode-3 adjoint | stable @600/1e20; k~3.54e-3 (-12%) | `lm192_mode3_dt*.log` | final |
| 12 | dt-cycling | exact reversible attractors; k(60)=3.9883632e-3 to 10 digits; no drain | `lm192_wall05_cycle.log` | final |
| 13 | P runaway | max|dP|=0.217/500 steps; |P|max=1.30 vs stag 0.017 | `ladder_R8_wall05.log` | final |
| 14 | clean gap R=8 | -2.53% (wall05 3.91964e-3 vs stag 4.02144e-3) | `ladder_R8_*.log` | final |
| 15 | clean gap R=12 | -0.76% (3.98836e-3 @dt60 vs stag 4.01803e-3) | `lm192_wall05_dt60.log` | final |
| 16 | attractor dt-spread (blend) | ~5e-4 rel over dt 6..1e20 (was 1.8e-3) | cycle log | final |
| 17 | TGV control | col-stag order +2.00 (N16..128) | `tgv_control.log` | final |
| 18 | flat-wall E1/E1b | stag==col identical all (N,s); L2 2e-7 | `flatwall_sweep.log` | final |
| 19 | flat-wall E2 | stag exact (1e-14); gauge-exact O(h^6); plain O(h^1.6) | ibid | final |
| 20 | s=0.5 incidence | O(1) error ALL solvers (robustness note) | ibid | final |
| 21 | N=96 (R=6) stability | no growth any variant 6000 steps | `lm96_*.log` | final |
| 22 | clean ladder R=16/24/32 | — | Snellius 25914731 `cleanladder_*.log` | RUNNING |
| 23 | embed NaN root cause | — | — | TODO |
| 24 | support-consistent ladder | — | — | TODO |
| 25 | Arnoldi spectrum | — | — | TODO |

*(All cited scratchpad logs are archived in `flow/doc/data/collocated_campaign/`.)*

## 5. Repo state (for the next session)

- flow main @ `c055894` (8 commits ahead of origin, NOT pushed): wall-blend + weight + filter
  (retired) + gauge-2a flags, set_dt fix, harnesses, notes, this plan. Defaults byte-identical;
  staggered untouched.
- peclet-examples @ `0a04990` (not pushed): dt-sweep harness + local results.
- umbrella pointer: bump pending.
- Snellius: source rsync-OVERLAY of `flow/src` + `tests/study/collocated_longmarch.py` sits on
  the checkout (dirty, no push); job 25914731 rebuilds build_cuda_mpi and runs the clean ladder.
