# The collocated accuracy ceiling — evidence, refuted mechanisms, state of the diagnosis

*Definitive record as of 2026-08-20, superseding the diagnosis section of
`collocated_ceiling_plan.md` (whose "defect (a)" attribution is refuted below; its Step-0/1
execution results are folded in here). Written to bootstrap the next session. Every number in this
document is a measurement with a committed artifact; nothing here is conjecture unless labelled.*

---

## 1. The phenomenon

The collocated solver (`SolverColocated`, ABC approximate projection) converges **to a fixed
offset from the staggered/continuum answer, not to it**. Stokes permeability ladders on two
DEM-grown periodic sphere beds (one fixed physical packing per φ, resampled at R = 5…32 cells per
sphere radius, march tol 1e-6, staggered cut-cell as the same-rung reference):

**Gap = (k_collocated − k_staggered)/k_staggered at the SAME rung, %:**

| variant / metric | R=8 | R=12 | R=16 | R=24 | R=32 |
|---|---|---|---|---|---|
| φ=0.50 gauge-exact, cell | −1.461 | −0.211 | +0.145 | +0.299 | **+0.297** |
| φ=0.50 gauge-exact, flux | −2.064 | −0.503 | −0.027 | +0.218 | **+0.251** |
| φ=0.50 ghost, cell | −1.035 | −0.166 | +0.063 | +0.154 | **+0.151** |
| φ=0.50 ghost, flux | −1.528 | −0.410 | −0.081 | +0.087 | **+0.112** |
| φ=0.50 plain (mode 0), cell | −5.184 | −3.351 | −2.463 | −1.630 | −1.220 |
| φ=0.60 gauge-exact, cell | −2.251 | −0.503 | +0.071 | +0.357 | **+0.385** |
| φ=0.60 gauge-exact, flux | −3.482 | −1.084 | −0.272 | +0.195 | **+0.291** |
| φ=0.60 ghost, cell | −1.433 | −0.267 | +0.075 | +0.218 | **+0.218** |
| φ=0.60 plain (mode 0), cell | −7.198 | −4.683 | −3.462 | −2.290 | −1.716 |

(Data: `peclet-examples/benchmarks/porous-scaling/results/snellius-h100/colcmp*_R*_flux.json`,
commits d260975 / ade42a3 / 6b1be1d / 77c0ace. The staggered reference itself is converged:
−0.009 % (φ=0.50) / on its ±0.05 % noise floor (φ=0.60) at R=32 vs its own Richardson k∞.)

Structure of every collocated series: fast pre-asymptotic convergence (apparent order 2.2–3.3
over R=5…12), a **zero crossing at ~12–16 cells per sphere radius**, then a plateau. Between
R=24 and R=32 the gauge-exact gap moves by −0.001 (φ=0.50 cell) to +0.079 (φ=0.60 cell) — no
detectable decay, slight growth on the dense bed.

**Why it matters:** the AMR solver (`core::amr::AmrFlow`) is collocated, and the whole point of
AMR is to drive error down by refinement. A ~0.3 % floor that refinement cannot cross blocks
that. If the offset is a true constant the scheme is formally **inconsistent** (zeroth-order),
which is a defect to fix, not a quality tradeoff.

## 2. What is firmly established

1. **Four independent series plateau** (2 beds × 2 estimators), reproducing rung-to-rung to
   ~0.01 % on φ=0.50 (+0.299 → +0.297).
2. **The ladder can see convergence when it is there**: mode 0 (plain), first order throughout
   (0.86–1.29 per rung), is still improving by 0.4–0.6 %/rung at R=32 in both metrics — a
   working control alongside the flat series.
3. **The same phenomenon exists on Zick & Homsy**, ~4× smaller: gauge-exact vs K_ZH is a flat
   −0.04…−0.10 % band from N=32 to 128 (−0.056/−0.099/−0.082/−0.091/−0.043), while staggered
   converges cleanly through it (−0.314 → +0.013 %). Direct same-grid gap
   (`zh_collocated_gap.py`, benchmark precision cancels): +0.259 (N=32), then −0.017, −0.064,
   −0.100, −0.056 (N=48…128) — nonzero but scattering by 6× between rungs.
4. **The sign is physically consistent across geometries** once K (drag) and k (permeability)
   are put on one axis: at coarse resolution the collocated schemes **under-predict flow**
   (beds: k gap < 0 for R≤12; Z&H: K gap +0.26 % at N=32); past ~12–16 cells/radius both flip
   to **over-predicting flow** (beds: k gap > 0; Z&H: K gap < 0). Same crossover resolution,
   same asymptotic direction, on both geometries. Asymptotic flow excess: ~0.05–0.10 % (Z&H,
   φ=0.125) → +0.25–0.30 % (bed φ=0.50) → +0.29–0.39 % (bed φ=0.60): **grows with confinement**
   (wall area per unit through-flux), the signature of a wall-sourced error.
5. **The ghost projection plateaus at half the gauge-exact level, same sign, both beds, both
   metrics.** Since ghost replaces the *entire constraint closure* (binary openness +
   wall-anchored quadratics instead of apertures) and the plateau survives at ~½ magnitude, the
   machinery **shared** by mode 9 and ghost is the prime suspect pool: the cell-centred momentum
   solve with its cell-centre-classified IBM, the ½/½ cell→face averaging of near-wall COUPLED
   faces, `gpCenterGrad` (predictor + cell correction), the ABC cell/face reconciliation, and
   the incremental-rotational pressure accumulation. The constraint closure itself is what
   *differs* between the two — and also what §3.2 exonerates directly.
6. **Dimensional oddity (Frank's argument, unexplained):** fitting e = C·(h/R)² + B gives
   C ≈ 0.8 (order 1, healthy) but B ≈ 0.003 — a zeroth- and second-order Taylor coefficient
   separated by ~250×. Genuine inconsistency usually does not hide two decades below the
   leading truncation constant. Consistent with this, **no mechanism for a true zeroth-order
   term has survived direct measurement** (§3). Caveat honestly: R=24→32 is only a 1.33×
   refinement — a very-slowly-decaying term (h^½, h·log h) is not yet distinguishable from a
   constant on φ=0.50; the slight *growth* on φ=0.60 argues against a decaying tail there.

## 3. Refuted mechanisms and artifacts (each with its measurement)

**3.1 The march / steady-state protocol — exonerated.** Re-running the decisive cases at
`MARCH_TOL=1e-8` (100× tighter), `CHECK_EVERY=20`, cap 3000, with per-step TRACE (Snellius job
25805513, results `tol_*`): k unchanged **to six digits** while taking 9–20× more steps
(φ=0.60 R=16 mode9: 0.0040203 at 70 and at 1380 steps; R=24: 0.0040337 at 60 and 540 steps;
staggered R=16: 0.0040174 at 260 and 2380). ⟨u⟩ monotone, flat to 7 digits from step ~120,
max|div| pinned at 3.9e-9. Short marches at fine R = better initial guess, not early exit.

**3.2 The constraint operator (the "½/½ truncation" story) — exonerated.** A-priori test
(`tests/study/collocated_constraint_consistency.py`): feed the discrete divergence an exact
solenoidal no-slip field (Stokes past a sphere — used as a *test field*, not as the periodic
solution) and compare each model face flux against the **exact open-area flux by 8×8
quadrature**. N=32→128: L1 defect converges at O(h^1.1) staggered / O(h^1.2–1.4) collocated —
no plateau; the **signed net** defect (what a permeability feels) is 1e-5 → 1e-6 of the
through-flux, sign-flipping, at the quadrature floor — **three decades below the 0.3 % bias**.
This kills the "aperture constraint's ½/½ truncation is the ceiling" claim (previously in
CLAUDE.md and the example page) and the attribution of the plateau to the embed port's
"defect (a)". A back-of-envelope also shows an O(1) per-cut-face relative flux error yields an
O(h) global error, not O(1) — the original mechanism never had the right arithmetic.

**3.3 The estimator/diagnostic — mostly exonerated.** Measuring k from the conserved face flux
`mean(α_f·uf)` (the quantity `div` actually drives to zero; `k_over_R2_face` in the bench)
instead of the cell mean shifts the plateau down 16–25 % and leaves it just as flat (φ=0.50:
+0.218 → +0.251 over R=24→32). The two estimators *bracket* the answer (staggered R=8: +0.72 %
cell, −0.19 % flux) — the flux one converges more slowly because α_f itself carries the planar-
interface O(h) truncation. Estimator choice is worth ~1 % at R=8 and ~0.05 % at R=32; it is not
the phenomenon.

**3.4 Wall-band localisation — the test failed three ways (record so it is not repeated).**
`tests/study/zh_wallband_diff.py` split the col-vs-stag face-flux difference by wall distance.
(i) **Unit artifact:** the "amplitude grows as 1/h" signal (orders −1.16/−1.00/−1.51 on the bed,
−1.16 on Z&H) was fictitious — these runs fix F and μ in *cell units*, so the velocity scale
itself grows as R²; normalised by ⟨u⟩ the near-wall difference **decays** at O(h^0.5–1.0)
(relative amplitude 0.649 → 0.461 → 0.346 → 0.284 for R=8…24). *Credit: Frank caught this.*
(ii) The band-share statistic passes through a **zero crossing** of ⟨ΔFlux⟩ at R≈16 (share
−0.594 there — a vanishing denominator), and where defined it *declines* (0.618 → 0.371 →
0.249 at R=8/12/24): the wall band does not dominate the net difference. (iii) Design flaw:
the system is elliptic, so localising the *response* can never localise the *cause* — a
wall-sourced defect is smeared into the bulk by the pressure solve. The share result therefore
neither confirms nor refutes a wall source; only §2.4's confinement scaling speaks to that,
and it says wall-sourced.

**3.5 A benchmark-table artifact — excluded by construction.** The bed gaps are two solvers on
identical geometry at identical rungs; K_ZH's four significant figures cancel entirely in
`zh_collocated_gap.py`. Also excluded: a broken staggered reference (it converges to its own
Richardson k∞ to −0.009 %), and Richardson pathology (the φ=0.60 staggered series sits on its
noise floor and the raw fit returned p=−0.21 with k∞ 0.9 % below every rung —
`analyze_collocated_ab.py` now rejects implausible fits and falls back to the finest rung).

## 4. Instruments — what to trust, hard-won

- **The bed is the sharp instrument, Z&H is not** (inverting the natural intuition): the bed gap
  reproduces to three digits between rungs (+0.299/+0.297) because the effect is 4× larger
  there; the Z&H gap scatters 6× between rungs at the protocol's noise level. Use Z&H only for
  what it is good at: an exact-drag anchor and the cleanest demonstration of mode 0's first
  order.
- **Never read convergence order across a zero crossing.** Both original "second order to a
  floor" claims (R≤16 beds, N≤128 Z&H) were the signed error passing through zero. The
  crossing sits right in the comfortable-resolution window (~12–16 cells/radius), which is what
  made this trap so effective. Go to R≥24 before concluding anything on these beds.
- **Normalise before reading growth/decay** — in cell units with fixed F, μ, every velocity-like
  quantity gains R² for free (§3.4.i).
- **Same-rung gaps beat absolute errors** whenever a second scheme is available: benchmark
  precision, geometry error and protocol bias all cancel.
- The a-priori pattern (exact field + exact quadrature reference, no solver in the loop) settled
  in an afternoon what marches could not settle in days. Prefer it for the next mechanism too.

## 5. What remains standing — suspects for the next session

By elimination (constraint out, protocol out, estimator out), the offset lives in how the
**solution** is formed, in machinery shared by mode 9 and ghost (§2.5):

- **S1 — the ABC cell/face reconciliation.** Collocated-only: two velocity representations (cell
  field, face field) corrected by *different* discrete gradients of the same φ (`gpCenterGrad`
  one-sided at cut cells vs plain face differences), reconciled only approximately. Its steady
  fixed point is exactly the kind of place a small consistent offset can live. No direct test
  run yet. Cheap probe: measure the steady distribution of `u_cell − avg(uf)` and its scaling.
- **S2 — the incremental-rotational pressure accumulation** interacting with the approximate
  projection. Cheap, decisive probe: **a dt sweep at a plateau rung** (φ=0.60 R=16, mode 9, DT
  6 / 60 / 600). Any dt dependence of the plateau indicts the accumulation channel (and would
  violate the C2 design requirement outright); dt independence kills S2 in one run. *Do this
  first — it is one job.*
- **S3 — the ½/½ average at solid-adjacent faces using masked-zero cell values.** The a-priori
  test says its *net* is tiny for an exact field, but the solve responds locally to the local
  error; ghost (which replaces exactly these closures) halving the plateau is circumstantial
  support. Probe: the `col_nm` (no-mask) variant exists in the a-priori harness; the solve-side
  equivalent is a small kernel ablation.
- **S4 — the cell-centred momentum IBM** (θ classified from cell centres vs staggered's face
  points). Both are nominally 2nd-order Robust-Scaled closures, but the collocated one feeds
  S1's loop. Field-level test: compare each solver's velocity field against a fine-staggered
  reference (restricted), relative L2 over fluid — if the collocated *field* error plateaus
  while its drag converges pre-asymptotically, the field norm localises which term.

**Ruled-out fixes, do not revisit:** making the preconditioner ρ-aware (ghost phase A: harmful);
mode 10 open-centroid quadrature (O(h), divergent on slivers — breaks D/G adjointness;
conservation is the binding constraint on any constraint-side repair); the Seo–Mittal
pressure-only split (measured 1st order, worse than mode 0); reverting to the ghost projection
as production (same plateau family at half height, 2.3–5× cost, nonsymmetric + fragmenting +
the (1,2) march instability).

## 6. Design constraints on any fix (Frank, 2026-08-19)

- **C1** Cell centres carry conserved quantities (ρ, momentum density as velocity, energy);
  faces carry the volume flux; every flux = face volume flux × quantity interpolated to the
  face (flux-form FV shape).
- **C2** Steady states must be **dt-independent**. This disqualifies Basilisk's acceleration-
  event `uf = fs·(face_avg(u)+dt·a)` *as written* and both salvage routes in the embed-port
  notes (undivided convention; implicit face body-force source) — all leave dt inside uf.
  It also makes the S2 probe above double as an acceptance test.
- **C3** Must generalise to AMR 2:1 coarse–fine faces (face-primary fluxes reflux naturally —
  a point in their favour).
- **C4** No robustness regressions: no fragmenting pressure graph / connectivity guard, keep
  the operator symmetric (CG) if at all possible.

Step-2 candidates from `collocated_ceiling_plan.md`, still open, to be re-ranked once §5
localises the defect: (A) momentum-interpolated face flux with a **dt-free** coefficient
(needs a literature check for a published dt-independent formulation with curved-wall order);
(B) flux-form conservative EB finite volume (EBChombo/Trebotich–Graves line — matches C1,
AMR-native; small-cell redistribution vs C4 untested on contact-tight throats).

## 7. Related state (context a fresh session needs)

- **Defaults shipped and standing:** `set_collocated_scheme("gauge-exact")` (= old
  `set_face_interp(9)`) is the collocated default (flow 8a8dcdb); AMR `setGhostGradient`
  default ON (core eb73c6c). Justified by mode 0 being first order and 4–6× worse at every
  rung — the plateau does not change that ranking, it caps it. Budget **~0.3 % as the current
  collocated accuracy ceiling** (CLAUDE.md, example page 9d99c5b already say so).
- **Ghost projection:** quarantined (verification only); asymptotically ~2× more accurate than
  gauge-exact, 2.3–5× dearer; its `(1,2)` mixed mode is march-unstable on beds >~2000 spheres
  (`ghost_hardening_findings_A.md`).
- **Embed line (modes 5/6/7)** re-instated after being wrongly retired (flow acc940b): healthy
  on Z&H (N=32, 60 steps: K = 4.3245/4.2095/4.2808 for 5/6/7) but **NaNs on both beds** at R=8
  (`CutcellMG::solvePCG: preconditioner produced non-finite z`) — never exercised on packings;
  debugging that is a prerequisite to evaluating the embed route at all. Mode 8 (face-primary
  uf) fails under the divided convention (−75 % drag) and is barred as-written by C2.
- **Separate open thread:** the AMR aperture-under-advection solver defect
  (`core/docs/amr_aperture_advection_plan.md` + `amr_advection_session_prompt.md`) — unrelated
  mechanism, do not conflate.

## 8. Harnesses, data, reproduction

Harnesses (flow, all committed): `tests/study/collocated_constraint_consistency.py` (a-priori
constraint vs exact quadrature), `tests/study/zh_collocated_gap.py` (same-grid col−stag gap on
Z&H), `tests/study/zh_wallband_diff.py` (band localisation; flawed per §3.4 — kept as a record,
takes `BED=<packing.npz>`), `tests/study/collocated_zh_schemes.py` (Z&H plain vs gauge-exact in
the example page's protocol; "plain" reproduces the published columns exactly — the protocol
control). All fall back to `set_face_interp` on builds predating `set_collocated_scheme`.

Bench (peclet-examples `benchmarks/porous-scaling`): `spheres_bench.py` knobs `GRID`,
`FACEINTERP` (always passed explicitly — a falsy-guard bug that silently turned mode-0 baselines
into mode 9 after the default change was fixed in 77c0ace), `TRACE`, `PUNDER`, `GPORDER`;
`k_over_R2` + `k_over_R2_face` in every march JSON. Ladder: `snellius/collocated_ab_gpu.sh
<R> <tag> <phi050|phi060>` (6 variants incl. embed); verdict: `analyze_collocated_ab.py
<dir> <colcmp|colcmp060>` (Richardson-guarded). Wall-band sbatch: `snellius/wallband_gpu.sh`.
Data: `results/snellius-h100/` — `colcmp*_R*_flux.*` (the definitive both-metric ladders),
`tol_*` (protocol exoneration), `wallband_N*.log`. Beds: `results/packings/…s100.npz` (φ=0.50),
`…phi0.60_s3.npz` (φ=0.60, contact-tight, median NN gap 2e-4 R). **Do not use
`flow/tests/study/rcp_pack_seed3.npz` for periodic work — it is non-periodic** (92/180 spheres
cross the box face).

Snellius: ssh alias `snellius` works non-interactively; billing per *allocated* GPU
(right-size `--gpus-per-node`); the flow checkout there needs `git pull --ff-only origin main`
(no upstream configured — a bare `git pull` fails silently in `&&` chains).
