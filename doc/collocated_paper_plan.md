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
- [x] Embed-NaN root cause + fix (rows 23/26; flow 4732b17) -- and the repaired modes 6/7 are
      measured unconditionally unstable (non-adjoint Uzawa), closing that branch
- [x] Support-consistent scheme validated for STABILITY + COLLAPSE (rows 27-29): adjoint-aperture
      modes 11/12/13 -- m1 -> 1e-5 monotone, P frozen, C2 restored dt=60..1e20 with NO blend;
      accuracy measured O(h) with -8..-11% at R=8 (row 30: x1.49 contraction R=8->12) => the
      support/accuracy tension is fundamental for this pair (note S6). Ladder R=16..32 queued
      (Snellius 25922444); phi=0.50 bed replication still open
- [x] Z&H clean cross-check (row 45): ghost (2,2) -0.018% at N=128, order ~1.6-1.8
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
| 22 | clean ladder R=16/24/32 | PROTOCOL INVALIDATED at R>=16 for wall05@dt600 (row 32); job 25914731 TIMED OUT after R24 (5h; divergent rungs ran ~300 PCG iters/step); harvested: R16_stag FINAL 4.0173693e-3, R24_stag partial (step 1500: 4.0192623e-3); re-run at dt=60 = job 25937469 (cleanladder2, incl. R24/R32 stag insurance) | `cleanladder_*.log` harvested to peclet-examples + doc/data | superseded by 25937469 |
| 23 | embed NaN root cause | FOUND+FIXED: degenerate-sliver wall drag U/d0, d0=|sdf| floored at 1e-3 -> explicit lagged gain mu*area/d0 ~ 1e2 x diag at grazing centres (dt-free x~100/step, localized solid-ctr cells |sdf|~1e-3; Basilisk-safe only because implicit *coef*); fix = floor d0 at 0.5 (gain <= ~0.6 for any mu,rho,dt) | `probe_localize.py` runs + src/mac_approx_projection.hpp | fixed |
| 24 | support-consistent ladder | — | — | TODO |
| 25 | Arnoldi spectrum | — | — | TODO |
| 26 | mode-6 post-fix instability | UNSTABLE: P exp-growth x~11-15/250 steps, dt-FREE (60==600), wall-blend w0=0.5 AND w0=1.0 ineffective; PM I (ROT=0) STABLE (m1 8e-3->4e-4) -> rotational-update Uzawa instability of the NON-ADJOINT normalized pair, not wall-band-local; rotw=0.3 only SLOWS it (re-grows from ~step 1000 at dt=60) -> no scalar/diagonal rescaling works, modes 6/7 structurally dead as S1 | `lm128_mode6_*.log` | final |
| 27 | mode 11 (adjoint-aperture pair) | NEW SCHEME G=-(D_a Pi)^T (adjointness verified 8e-16 a-priori): COLLAPSE CONFIRMED at R=8 -- m1 -> 1e-5 monotone (gauge-exact freezes at 1.8e-2), |P| frozen at stag scale, k(dt=60/600/1e20) = 3.5743523/3.5743595/3.5745995e-3 all monotone to one limit (C2 to ~2e-6, vs 5e-4 attractor spread), NO blend needed incl. dt=1e20; accuracy price -11.1% vs stag at R=8 | `lm128_mode11_dt*.log` | final |
| 28 | mode 12 (per-cell rescale S=6/sum(o)) | stable, collapse identical (m1 8.8e-6, k 10-digit settled 3.6958252e-3); gap -8.10% at R=8 | `lm128_mode12_dt600.log` | final |
| 29 | mode 13 (capped per-axis normalize, floor 0.25) | stable, collapse identical (3.6776869e-3); gap -8.55% at R=8 == mode 12 -> weighting magnitude is NOT the accuracy lever; adjoint family shares structural ~-8..-11% at R=8 | `lm128_mode13_dt600.log` | final |
| 30 | adjoint-family order check R=12 | mode 12: 3.8000478e-3 = -5.425% (ratio 1.494 vs R=8), mode 13: 3.7898681e-3 = -5.678% (ratio 1.506) -> textbook O(h), two independent variants; support-consistency via multiplier-reading is FIRST order, period | `lm192_mode1{2,3}_dt600.log` | final |
| 32 | wall-blend margin is R-DEPENDENT | w0=0.5 UNSTABLE at (R=16, dt=600): k negative by step 50, doubling ~4-8 steps, H100 AND OMP identical (code-deterministic, backend-independent); healthy at (R=12, dt=600) and (R=16, dt=60); incidence refuted (grazing-cell counts smooth in N); mechanism: GF gain ~ mu*dt/h^2 grows with refinement while the banded diagonal is fixed w0*mu -> the blend is NOT a uniform stabilizer; clean-ladder R>=16 rungs re-run at dt=60 (job 25937469); w0=1.0@dt600 diagnostic: diverges FASTER (1e14 by step 50) -> no w0 rescues dt>=600 at R>=16, blend stability domain = (R<=12, dt<=600) u (R=16, dt<=60) | `cleanladder_R16_wall05.log`, `cleanladder2_R16_w10_dt600.log`, `omp256_*.log`, `incidence_check.py` | final |
| 33 | clean ladder stag refs COMPLETE | R=8..32: 4.02144 / 4.01803 / 4.01737 / 4.01926 / 4.0213 e-3 (R24 longmarch dt60 converged; R32 via np=4 MPI bench, tol 1e-9, dt60 -- staggered protocol-insensitivity measured, so protocols mix validly) -- the reference itself WANDERS +-0.05%, the same scale as the collocated plateaus: final asymptote statements need Richardson on BOTH series, not single-rung gaps | snellius `cleanladder_R16_stag.log`, `stagr24_R24_stag.log`, `stagr32mpi_R32.log` | final |
| 34 | phi=0.50 replication R=8 | stag 1.0923667e-2; wall05 1.0738996e-2 (-1.69%); mode12 1.0277467e-2 (-5.92%, m1 -> 4e-6) -- phi=0.60 story replicates | `p50_R8_*.log` | final |
| 37 | clean gauge-exact ladder (dt=60) | R=12: -0.76%, R=16: -0.1335% (4.0120051e-3, dk 8e-12, STABLE at dt=60 w/ wall05; m1 frozen 1.25e-2 + |P| drift to 8.3 = Layer-2 family alive as expected); R=8 dt-60 re-run: 3.9194118e-3 = -2.536% == dt-600 value (protocol-robust); homogeneous dt-60 series -2.54 -> -0.76 -> -0.13 -> ~+0.20 (R=8/12/16/24; R24 = 4.0272055e-3, CONVERGED dk 3e-11 before its job timed out; gap provisional on the final R24 stag ref) -- sign crossed, heading toward the old +0.39% asymptote: the gauge-exact plateau appears REAL and family-borne under the clean protocol (m1 frozen 1.25e-2); R=32 + exact stag refs pending | `cleanladder2_R{16,24}_wall05_dt60.log`, `ladder_R8_wall05_dt60.log` | partial |
| 47 | ghost clean ladder VERDICT (4 rungs) | R=8/12/16/24: -1.431 / -0.265 / +0.077 / +0.219 % (R24: ghost 4.0280824e-3 vs stag ref 4.0192593e-3, both converged; dt-6000+VIT-50 protocol, levers validated to ~5e-4%) -- the ghost has its OWN real plateau ~ +0.22%, matching the OLD record's R32 asymptote (+0.218%) from an independent protocol; NOT contamination: family-free (m1 3e-5), stable, C2. R32 single-GPU impossible (OOM, ~1.6KB/cell); rung dropped as decision-irrelevant | `ghr24b_R24_ghost.log`, `stagr24_R24_stag.log` | final |
| 46 | np>=16 ghost debt: de-confounding | recorded weak np16/32 rungs ran the PURE (2,2) mode (json gporder) -> real open problem; s116 (7823-sphere) bed SINGLE-RANK: R=8 CONVERGED k/R^2=0.0114366 (s116probe), R=12 OOM (ghost ~1.6KB/cell ceiling confirmed) -> single-rank stability extends to R=8; THE Phase-A discriminator (same bed, native R=16, np=8) submitted as job 25975965 | `s116probe_R8.json.log` (snellius) | partial |
| 45 | Z&H clean anchor, ghost (2,2) | err -0.175 -> -0.084 -> -0.056 -> -0.029 -> -0.018 % (N=32..128), order ~1.6-1.8, div ~1e-15 -- external anchor converges onto the Z&H constant | `zh_ghost22.log` | final |
| 44 | B+ spectral gates | B+sym DEAD (sym(A_ghost) indefinite: persistent negative family + 1+-73j preconditioned pair); B+kron PASS (A_star SPD, spec(A_star^-1 A_ghost) real positive [0.02, 2.2-3.0] at N=32/48) -> star-aperture structure as the ghost's PRECONDITIONER is the efficiency lever (reuses the Design-B solvePCG overlay); pairing lesson corollary: unstable-as-scheme, excellent-as-base | `bplus_gate.py` (doc/data) | final |
| 42 | ghost phi=0.50 replication | R=8: 1.0810846e-2 = -1.033% (stag 1.0923667e-2), R=12: 1.0848649e-2 = -0.164% (stag 1.0866476e-2) -- x6.3 contraction per 1.5x, family-free (m1 -> 7e-7): replicates the phi=0.60 pattern on the second bed | `p50_R{8,12}_ghost.log`, `p50_R12_stag.log` | final |
| 43 | VIT insensitivity | ghost R=12 VIT=50 vs 150: k identical to 8 digits (4.0073681 vs 4.0073684e-3) -> velocity-sweep count does not move fixed points (S5 confirmed); R=32 rung runs at VIT=50 | `lm192_ghost_vit50.log` | final |
| 41 | Design B (SPD Kron star, set_fluid_only_constraint(2)) | IMPLEMENTED (star_elimination.hpp; solvePCG star overlay; filtered-surrogate preconditioner; phibar face fix-up; defaults byte-identical) but UNSTABLE with the gauge-exact gradient: P exp-growth from step 1, doubling ~4 steps at (R=8, dt=600), ~80 steps at N=64 -- PCG solve itself converges cleanly => outer-loop (pair) instability; PM I ALSO unstable (m1 3.8, |P| 1.6e4 @600 steps) and dt=60 too -> NOT the rotational update: the projection loop itself amplifies (G_ge structurally mismatched to the star rows); fluidonly2+mode13 pairing probe CONFOUNDED (mode-13 G reads the pinned phi_s=0 -> worse, inf@50) -- a clean matched-pair test needs a phibar-substituting gradient, not built (mode 11 already demonstrates matched-stable); design lesson recorded in fluid_only_constraint_plan.md: stability = pair matching, accuracy = closure-value consistency, ghost architecture = the only measured scheme with both | `lm128_fluidonly2_dt600.log`, `lm128_fo2_{pmI,dt60,m13}.log` | final |
| 40 | mode 14a (Design A fluid-only filter) | family-free + stable as designed (m1 -> 1e-6) but accuracy-dead: R=8 3.7326033e-3 = -7.18%, R=12 3.8477056e-3 = -4.24% (order ~1.3, constant ~ adjoint family) -> the Neumann-zero closure is insufficient; B/B+ carry route 2b (see fluid_only_constraint_plan.md) | `lm1{28,92}_fluidonly_dt600.log` | final |
| 39 | ladder job economics | ALL THREE R>=16 jobs timed out after their R16 rungs: measured per-step 0.72 s (collocated 256^3), 3.3 s (384^3), ghost 1.94/9.6 s (256^3/384^3) under the fixed protocol -> R32@dt600 ghost ~25 h/rung, infeasible; re-plan: stag refs R24/32 as own job (25945148), ghost high rungs pend the dt-6000 C2 probe (large-dt short marches), adjoint R24/32 SKIPPED (O(h) proven) | sacct + partial logs | final |
| 38 | adjoint-family O(h), third rung | mode 12 R=16: 3.8527726e-3 = -4.10% (-8.10 -> -5.42 -> -4.10: perfect 1/R line); mode 11 R=16: -5.21% (x2.14 per 2x) | snellius `mode11ladder_R16_*.log` | final |
| 35 | ker(G_ge) fluid-side is TRIVIAL | a-priori (numpy mirror of gpCenterGrad, real bed, percolating component): exactly ONE zero eigenvalue (the constant) at N=32 AND N=48; next cluster at the smooth-mode scale with SPREAD eigenvectors (PR 0.44-0.62) -> the ENTIRE attractor family lives in the solid-row flux-balance equations; a fluid-only constraint + gauge-exact gradient is unique-by-construction (route 2 structurally de-risked) | `kernel_study.py` (doc/data) | final |
| 36 | ghost CLEAN protocol R=8/R=12 | R=8: 3.9639024e-3 (-1.431%), R=12: 4.0073684e-3 (-0.265%) -- x5.4 contraction per 1.5x (super-quadratic => zero-crossing near R~13-16, R=16..32 decide: job 25937956); family-free signature BOTH rungs (m1 -> 1e-5 monotone, |P| frozen) AND NO Layer-1 growth at (R=12, dt=600) unblended where gauge-exact doubled every 82 steps -- the fluid-only constraint removed the instability with the family; R=16 (H100): 4.0204793e-3 = +0.0774% -- zero CROSSED between R=12 and 16, family-free + stable unblended; C2 CERTIFICATE PASS: k(dt=60/600/6000/1e20) = 3.9639025/3.9639024/3.9639210/3.9639208e-3, spread <= 5e-6 rel over 19 decades, all monotone to one limit; dt=6000 within 5e-6 at 2000 steps -> large-dt short marches licensed for R=32; R=24 running (job 25951131) | `lm128_ghost_dt*.log`, `lm192_ghost_dt600.log`, snellius `ghostladder_R16_ghost.log` | partial |
| 31 | mode-12 dt-cycling (C2 flagship) | ONE fixed point across dt=60->600->6->1e20->60: segment-end k 3.6958242/3.6958241/3.6958214/3.6958229e-3 -- spread 7.6e-7 rel over 19 decades of dt (gauge-exact attractor spread 5e-4), m1 ~ 3e-6 throughout, stable at 1e20 mid-cycle with no blend | `lm128_mode12_cycle.log` | final |

*(All cited scratchpad logs are archived in `flow/doc/data/collocated_campaign/`.)*

## 5. Repo state (for the next session)

- flow main @ `c055894` (8 commits ahead of origin, NOT pushed): wall-blend + weight + filter
  (retired) + gauge-2a flags, set_dt fix, harnesses, notes, this plan. Defaults byte-identical;
  staggered untouched.
- peclet-examples @ `0a04990` (not pushed): dt-sweep harness + local results.
- umbrella pointer: bump pending.
- Snellius: source rsync-OVERLAY of `flow/src` + `tests/study/collocated_longmarch.py` sits on
  the checkout (dirty, no push); job 25914731 rebuilds build_cuda_mpi and runs the clean ladder.
