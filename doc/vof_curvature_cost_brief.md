# Architect brief — G: the cost of block-VoF curvature on the CPU (and GPU)

Date 2026-10-08. Orchestrator: the Opus session driving the VoF performance campaign
(`doc/vof_perf_STATE.md`). USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on
every case it handles; yardstick TBFsolver on the bubble column. USER DECISION 2026-10-08: design
D and G now.

## 1. The question
Which change(s) cut the block-VoF curvature stage from **9.7 ms/step** (genoa 1×24, of a 93 ms step;
TBFsolver's whole step is 46 ms) to ≲ 4 ms **without degrading curvature accuracy** (convergence
order and max error on the existing gates) — and which of them are bitwise vs recorded numerics
changes? Deliver a design note with ranked options, a recommendation, work orders and gates.

## 2. Why the architect
Curvature IS the surface-tension accuracy budget (register: "CSF force is exact; curvature error is
the real budget — improve κ, never the force"). A cheaper tier-3 that is less accurate would
silently degrade every capillary result. Several cheaper-looking alternatives were already rejected
on accuracy (see §6). Needs judgement on accuracy vs cost, and on which algorithmic restructurings
are bitwise.

## 3. Current state (facts from recon of flow main f19ac4b; verify anchors yourself)
**IMPORTANT correction to the orchestrator's earlier framing:** "height functions first, fit only
where HF fails" is ALREADY the cascade. The cost question is tier 3 itself and how many cells reach it.

Cascade (`src/vof/curvature_field.hpp` `curvHeightCell` 57–133, `curvFallbackCell` 199–224;
container-free bodies in `core/include/peclet/core/vof/curvature.hpp`):
- Tier 1 `kCurvHf`: HF along the largest |n_d|: 3×3 columns × `kHfColumn` = 7 cells; fails if any
  column has no pure end within 7 cells, non-monotone (monoTol 1e-6), or orientations disagree.
- Tier 2a `kCurvHfMixed`: HF along the other two directions.
- Tier 2b `kCurvHfFit` (point fit to the closed columns): implemented, **OFF** (`useMixedHeightFit
  = false`, l.494; comment 464–493: it destroyed max-error convergence, order 1.37 vs 2.26).
- Tier 3 `kCurvPv` / `kCurvPvReduced`: PV paraboloid fit over the 5³ neighbourhood; for every
  interfacial neighbour its PLIC polygon (`plicPolygon`, ≤ 8 vertices) is projected into the
  target's frame, Wendland-weighted (width 2.5), 6×6 normal equations, LDLᵀ solve (`curvSolveSym`);
  reduced {1,x²,y²} model if < 6 polygons; polygons with frame-normal component ≤ cosMin 0.2 dropped.
  NOTE: each neighbour's polygon is rebuilt for every target that sees it (~125 targets).
- Then `kCurvNoEstimate` (gated == 0). Block path: |κ| clip at 1/Δmin (block only, register).
- CSF face curvature = arithmetic mean of the two cells' κ if both defined (`surface_tension.hpp:87`).
Excerpt, tier 3 body:
```
PvFit fit; pvFitInit(fit);
for oz,oy,ox in [-2,2]^3: j = i + ox + oy*sy + oz*sz;
    if (!vofIsInterface(c(j), ieps)) continue;
    pvFitAdd(fit, mx(j), my(j), mz(j), al(j), off, org, t1, t2, nn, dW, cmin, g);
curvFallbackStore(i, fit, kap, br);   // pvFitSolve -> kCurvPv / kCurvPvReduced / kCurvNoEstimate
```
Block launch layout (`src/vof/block_batch.hpp` 1086–1430, `block_container.hpp:1855–1945`): per
chunk of ≤ 16 markers: compaction scans, PLIC over the grown list, HF pass (pass 0), tier-3 pass
(pass 1), clip (pass 2), census (pass 3), CSF force — all one launch each over all markers. Host:
dynamic schedule chunk 16 (`vofHostListFor`, §14 H-4(a)); tier 3 = one thread per target. Device:
tier 3 = one warp-team per target, canonical-order accumulation (register "PV curvature fallback:
one team per target cell", 2026-10-03).

Measured (genoa 1×24, S-1, flow 38e80e6, 30-step kprof): curvature stage 9.7 ms/step;
`vof::block::batch_curv_list` 7.17 ms (4 launches) — tier 3 dominates; `batch_plic` 0.63; CSF
0.78. Tier 3 handles ≈ 35 % of interfacial cells in this case (doc design §3.6, from an earlier
census — re-measure with `diagnostics.vof_block_curvature_stats()`). Sphere tests: tier 1 cannot
serve 19.5–59.6 % (curvature_field.hpp:474). Bubbles are D/h ≈ 16 (case
`peclet-examples/benchmarks/bubble-column`, block VoF, marker count: check the case file).
GPU (RTX 5080) step 36–37 ms; curvature share there unknown — measure.

## 4. Constraints and invariants
- Accuracy gates must not regress: `tests/kokkos/test_vof_curvature.cpp` gate B sphere convergence
  (CHECK oL1 > 1.7 at 16/32/64), C sweep D/dx 3–40 (noEstimate == 0), F PV alone, G 2b ablation,
  H purity, I clip; `test_vof_surface_tension` P1–P7 (P1 machine-zero currents for constant κ);
  study gates `tests/study/vof_surface_tension.py` static/wave/lamb/hysing1/hysing2; block tests
  `test_vof_blocks*`, MPI np 1/2/4. Bubble-column statistics against the published rtol-1e-8 run
  (`~/Codes/bubble_column_perf/peclet_runs/rtol1e-8/`).
- Device and host must stay bitwise equal to each other where they are today (test E device vs host
  bitwise) unless a recorded decision says otherwise; host builds use `-ffp-contract=off`.
- No GPU↔host transfers in the step (USER DIRECTIVE); no new env vars that change results; new
  container-free VoF math belongs in `core` (core tag + flow pin bump needed).
- Distributed: block VoF runs multi-rank (MPI tests) — the design must keep MPI bitwise to np = 1.

## 5. Already decided, not open
- HF cascade + PV paraboloid fallback is the method (register "Method verdict: WY-split PLIC (B)").
- Improve κ, never the force; no κ smoothing that changes the balanced-force property.
- Block-only |κ| clip; debris removal (register 2026-09-25).
- Purity tolerance is TOLD to consumers (wispEps), not chosen per consumer.
**Genuinely open:** how to make tier 3 cheaper (e.g. per-cell polygon/moment caching so each
neighbour's polygon is built once per step instead of once per target; precomputed per-polygon
moments transformed per target; a point/centroid fit as in Popinet 2009 where accuracy allows;
smaller support where safe); how to make fewer cells reach tier 3 (longer/adaptive HF columns, as
Popinet's HF uses larger stencils; the conditions under which the 3×3×7 HF fails at D/h 16); launch
and memory layout; whether any of this is bitwise; GPU applicability.

## 6. Already tried / rejected (with evidence)
- Tier 2b mixed-HF point fit as a cheaper middle tier: max-error convergence order 1.37 vs 2.26 →
  OFF (curvature_field.hpp:464–493). Reopening needs a reason that addresses that number.
- A warp reduction of the PV normal equations (non-canonical order) — rejected (bitwise device=host).
- Union-colour force assembly; raising interfaceEps; κ clip on the structured path — rejected.
- §14 H-4(a) already fixed the host load balance of the list passes (dynamic schedule).
- TBFsolver reference: its source is at `~/Codes/bubble_column_perf/tbf_builds/tbf` — read how it
  computes curvature and what it costs per step (it runs the same case at 46 ms total); that is the
  SOTA yardstick for this stage.

## 7. How the answer will be verified
The gates in §4; per-tier census before/after on the bubble-column checkpoint
(`~/Codes/bubble_column_perf/ckpt_t43.npz`, `tests/study/vof_perf/prof.py`); kernel timer on genoa
(S-1 scripts `~/Codes/bubble_column_perf/s1/`) and on the RTX 5080; G-BIT (state_hash + 50-step
dump, scripts in `~/Codes/bubble_column_perf/`; NOTE their `rc $?` bug — they always print rc 0).

## 8. Deliverable
`flow/doc/vof_curvature_cost_design.md`, committed (named paths; push to flow main — USER
DIRECTIVE, no PR; another session may share the checkout: `git pull --rebase` first, commit only
your file). Sections: measured breakdown (re-measure the tier census and tier-3 cost split:
polygon build vs accumulate vs solve), options ranked with estimated savings (host and GPU),
accuracy risk, bitwise or recorded, recommendation, work orders sized for an `opus-implementer`,
gates per work order, register entries to add, open questions with defaults.

## 9. Out of scope
The pressure solve (separate brief D), PLIC/advection, the CSF force formula, contact angles,
phase change, implementing anything.
