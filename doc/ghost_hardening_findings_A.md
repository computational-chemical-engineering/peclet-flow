# Ghost-projection hardening — phase A findings

*Analysis phase of [`ghost_hardening_plan.md`](ghost_hardening_plan.md), run 2026-08-17 on the
workstation (RTX 5080 16 GB / Threadripper 5965WX) against flow `b96dd74` (overlap + CA smoothing
already in). No production code changed: the only additions are opt-in instrumentation
(`PECLET_FLOW_GP_DEBUG`, `src/ghost_projection_debug.hpp`) and new probes in
`tests/study/ghost_projection_apriori.py` (`--split`).*

**Headline.** The plan's lead hypothesis (**H1**: a handful of extreme thin-gap rows with huge
weights) is **refuted**, three independent ways: the θ clamp never fires on any bed we can build;
the worst closure weights are O(10²) and their rows are correctly rescaled to O(1); and the worst
rows are not at thin gaps at all but at grid/surface *incidence* coincidences, giving a fixed
density (~0.3 per sphere) rather than an extreme-value tail. The **conditioning of the
preconditioned operator is also fine** — measured on a real bed, spec(M⁻¹A) has |λ|min = 0.32,
λmax ≈ 2.15, spread ≈ 6.7, and making the preconditioner ρ-aware makes it *much worse*, not better.
So the ρ rescale is doing its job and is not the defect.

**H2** (compatibility bias) is **real but not the driver**: the left null vector really is not the
constants (|w·1|/(|w||1|) = 0.9925–0.9983 on beds), yet an rtol sweep on the running solver shows the
ghost solve converging cleanly and linearly to 1e-10 with no floor — so nothing is being burned on
the stagnation guard at the production tolerance. The cost is a *rate*: 13.5 iterations per decade
against cut-cell's 2.3 on the same bed.

**What we did find is a local failing case with a measured growth channel.** The documented mixed
mode `set_ghost_projection(True, matrix_order=1, rhs_order=2)` — the seam phase C was to be built on
— **diverges on the workstation** with exactly the Snellius signature (exponential, ×1.62/step,
k → 4.5e77) and with the *same control variable*: a bed-size threshold between 1956 and 7823
spheres, resolution-independent, small bed fine / big bed blows up. The cheapest instance is
**1.8 M cells, visible by step 40, ~1 minute on one GPU**. On it:

- the trajectory is **identical to five digits at rtol 1e-4 and 1e-12** — the instability is a
  property of the scheme, not of how well the linear system is solved;
- **under-relaxing the incremental pressure `set_pressure_underrelax(ω_p)` controls it**: at R=3,
  ω_p = 0.5 gives max|u| = 1.790e-02 against the stable (2,2) control's 1.795e-02 (and 0.2, 0.05 are
  stable too) — but at R=5 the same knob only *slows* the growth: max|u| at step 120 is 2.9e+04
  (ω_p = 0.2), 1.1e+02 (0.1), 1.2e+00 (0.02) against a correct 4.9e-02, and ω_p = 0.5 still reaches
  k = 8.3e40 over a 400-step march. It damps the amplifier without removing it;
- a 10× smaller dt makes it *worse*, not better.

So the amplifier is the **incremental-rotational pressure accumulation** `P += (ρ/dt)φ` integrating a
per-step projection residual that feeds back positively — ω_p turns the gain down monotonically, and
a smaller dt (more accumulation steps per unit physical time) turns it up. It is a diagnostic and a
damper, not a cure.

Go/no-go: **B = GO but almost entirely re-scoped** — "full Robust-Scaling" has little left to add
(every specific item in the plan's B list is measured to be a non-issue); the work that remains is
the compatibility projector plus two hygiene items. **C = GO but re-specified**: "the correction
converges through the time stepping" is precisely the thing that diverges, so the outer iteration
must live inside the step.

---

## A1 — reproduce the divergence locally

**The plan's configuration does not fit in 16 GB.** 512×256×256 ghost dies in `set_solid`
(`Kokkos ERROR: Cuda memory space failed to allocate 128 MiB (label="ci")`); 384×192×192 dies in
the gp overlay itself (`label="gpZ2"`), *before* the MG hierarchy is built, so reducing `MGLEVELS`
to 5 or 4 changes nothing. The ghost path's ceiling on this card is **≈10 M cells** — it carries a
worst-case-sized overlay (`gpMakeOverlay(nInner)`, ~160 B per *grid* cell whether or not the cell
has a row) plus three extra level-0 BiCGStab scratch fields. Cut-cell fits considerably more.

**The CPU fallback was measured and rejected.** OpenMP build, 2.1 M cells, np=1 × 24 threads:
35.5 s/step (28 s of it projection) — ~50× worse per cell than the GPU, extrapolating to ~570 s/step
at 33.6 M cells, i.e. >12 h for one march. np=4 × 6 threads is worse: stack sampling puts every rank
in `MPI_Waitall`, latency-bound on the small blocks.

**What the budget does allow — all converge, no growth exponent to measure:**

| bed | spheres | grid | R | march | k/R² |
|---|---|---|---|---|---|
| s116 (the np=16 bed) | 7823 | 320×160×160 | 5 | converged, 210 steps | 0.012097 |
| s100 | 489 | 224³ | 14 | converged, 90 steps | 0.011039 |
| s108 | 3911 | 192³ | 6 | converged, 185 steps | 0.011838 |
| s132 (the np=32 bed) | 15646 | 256×256×128 | 4 | 250-step cap, still descending | 0.012631 |

**Constructed stress case** (a periodic sphere lattice at the *reference* R=16 in which *every*
neighbour pair is near-tangent — jittered gaps down to 0.02 R = 0.32 cells and 0.002 R = 0.032
cells; `make_lattice_bed.py`): 64 and 216 spheres, both gaps, **all converge** in 95–105 steps.
The tighter gap needed *fewer* iterations (51 vs 61), not more.

**The decisive control was already in the committed Snellius data.** `refine_np32_ghost` is
1024³, np=32, R=64, 489 spheres → k/R² = 0.010874, perfectly sane (`conv=False` only because it hit
the 400-step cap). So a 1024³ grid, np=32 and full MG depth are individually innocent. Divergence
appears only on the weak rungs and only from ≥7823 spheres **at R=16** (np=8 / 3911 spheres
converges; np=16 / 7823 → −1.7e58; np=32 / 15646 → +1.4e57).

**Verdict.** The default (2,2) divergence needs the number of near-tangent contacts resolved at
R≈16, i.e. 268 M cells for the s116 bed — not reproducible under 16 GB. *Not escalated.* The two
remaining confounders (bed size vs np, since the weak ladder moves them together) would need one
Snellius rung: the same 7823-sphere bed at R=16 on np=8 fat ranks vs np=16.

## A2 — row forensics

Instrumentation: `PECLET_FLOW_GP_DEBUG=1` prints the overlay census after every build (`=2` dumps
every row to a binary file for offline correlation). Census over the bed family:

| geometry | grid | sph | R | overlay rows | QUAD | LIN | BC_ONLY | EXPLICIT | EXTENDED θ | θ clamped | ρ<1e-2 rows | min ρ | max abs w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s100 | 64³ | 489 | 4 | 77 603 | 93 053 | 23 224 | 21 273 | 2 358 | 59 225 | **0** | 139 | 6.2e-3 | 159 |
| s100 | 128³ | 489 | 8 | 331 279 | 499 297 | 46 005 | 31 220 | 2 208 | 278 055 | **0** | 208 | 2.3e-3 | 439 |
| s100 | 224³ | 489 | 14 | 1 021 137 | 1 686 076 | 64 548 | 37 862 | 2 370 | 890 024 | **0** | 210 | 4.8e-3 | 209 |
| s108 | 192³ | 3911 | 6 | 1 470 853 | 2 049 771 | 292 848 | 219 657 | 17 684 | 1 196 123 | **0** | 1 196 | 4.6e-3 | 217 |
| s116 | 320×160×160 | 7823 | 5 | 2 010 078 | 2 633 161 | 489 383 | 395 738 | 35 856 | 1 593 003 | **0** | 2 182 | 1.0e-3 | 971 |
| s132 | 256×256×128 | 15646 | 4 | 2 485 500 | 2 983 925 | 744 345 | 682 374 | 71 268 | 1 901 021 | **0** | 3 749 | 1.7e-3 | 583 |
| lattice gap 0.02 R | 128³ | 64 | 16 | 163 362 | 258 144 | 24 164 | 3 892 | 246 | 142 410 | **0** | 99 | 4.4e-3 | 224 |
| lattice gap 0.002 R | 128³ | 64 | 16 | 163 701 | 258 210 | 25 324 | 6 614 | 254 | 142 626 | **0** | 141 | 2.3e-2 | 42 |

What this says:

1. **`GP_THETA_MIN` (1e-4) never binds.** Not once, on any geometry, including the lattice whose
   every gap is a third of a cell. The smallest ρ observed anywhere is 1.0e-3 — three decades above
   the clamp. θ→0 requires the *near* face point to sit on the wall (θ = s_near/(s_near−s_ghost)),
   which is much rarer than a small gap; a barely-solid ghost face gives θ→1, not θ→0. **The clamp
   is not the problem and does not need widening or tightening.**
2. **Weights are bounded and the rescale does its job.** max|w| ≤ 971 across ~10⁷ closure faces, and
   because ρ = min_f D_f while w ~ 1/D_f, the *applied* coefficients ρ·w are O(1). No blow-up in the
   coefficients themselves.
3. **The EXTENDED-θ sliver branch is benign** — as its own docstring claims. θ ∈ (1,2) gives
   D = θ(1+θ) ∈ (2,6) > 1, so a sliver *never* lowers ρ, its weights are ≤ 1 in magnitude, and the
   high clamp (θ=2) never fires either. It is ~50 % of all closure faces and none of them is a
   problem. The plan's suspicion of this branch is **refuted**.
4. **The bad rows are the LIN branch with small θ**, and they are few and identifiable. The worst
   row on s116/R=5:
   `(104,29,151) rho=1.03e-3 maxw=9.7e+02 | LIN(th=1.03e-03) COUPLED COUPLED COUPLED LIN(th=2.6e-02) QUAD(th=0.65)`
   — one face with θ=1e-3 sets ρ for the *whole* row, so the row's three healthy COUPLED faces and
   its diagonal are all multiplied by 1e-3 as well.
5. **Pathological rows scale linearly with sphere count and are R-independent**: rows with ρ<1e-2 per
   sphere = 0.28 (s100/R4), 0.43 (s100/R8), 0.43 (s100/R14), 0.31 (s108), 0.28 (s116), 0.24 (s132).
   So the count saturates above R≈8 at ≈0.3–0.4 rows per sphere. This is *not* extreme-value
   statistics — it is a fixed density of bad rows per contact.
6. **The bad rows are NOT at thin gaps.** Correlating the 20 worst rows of s116/R=5 against the
   packing (distance from the cell centre to the two nearest sphere *surfaces*): every one of them
   sits essentially ON a single sphere's surface (d₁ = 0.000–0.002 R) with the second sphere
   0.007–0.20 R away — only one of the twenty is at a genuine near-tangent pair. Small θ comes from
   a **grid/surface incidence coincidence** (a face point landing on the wall), not from thin-gap
   geometry. That is why its density is proportional to sampled interfacial area — i.e. to sphere
   count — and why the all-pairs-near-tangent lattice is not worse. H1 as stated is refuted twice
   over.
7. **Row-scaling asymmetry is real but local.** The instrumentation reports
   max_nb |log₁₀(ρ_r/ρ_nb)|: ≤3 decades everywhere, with 460 (s100/R14) → 5 500 (s116) → 9 660
   (s132) rows above one decade — again linear in sphere count.
8. Fragmentation-guard pockets scale the same way: 345 (s100/R8) → 1 907 (s108/R6) → 3 646
   (s116/R5) decoupled cells.

**Conclusion for A2.** There is no small identifiable set of catastrophic rows. There is a *fixed
density* (~0.3 per sphere) of rows carrying a ρ of 1e-3–1e-2, whose LEFT scaling by ρ is invisible to
the preconditioner. The failure is therefore expected to be **cumulative in the number of such
rows**, which is exactly the bed-size dependence the Snellius ladder shows.

## A3 — the "fully Robust-Scaled" gap list (the phase-B work list)

Systematic diff against `cut_cell_ibm.hpp`'s momentum Robust-Scaling. The important finding is that
ghost's one-scalar ρ is **not** the crude transplant it was believed to be — for a single-sided
closure it is algebraically *identical* to cut-cell's `R = D_rescale/D_axis` machinery, because
ghost divides each face's weights by that face's own `D` (`gpOrderWeights`) and then scales the row
by `min_f D_f`, while cut-cell folds both steps into one stored coefficient
`K = poly_Nc(θ)·D_rescale/D_axis`. The real gaps are elsewhere.

| # | cut-cell mechanism | on the ghost path | what "full" means |
|---|---|---|---|
| 1 | per-direction `D_vals[k]`, `1e9` for fluid dirs | **present** (`gpOrderWeights` D; COUPLED faces excluded from the min) | — |
| 2 | closure weights divided by their OWN axis `D` | **present** (`w = poly/D_f`) | — |
| 3 | `D_rescale` = smallest-magnitude active `D`, applied to the whole row | **present** (`ρ = min(1, min_f D_f)`) | — |
| 4 | rescale threaded to the RHS as well (`rhs_scale`, used as `rs(i)*` in the momentum RHS) | **present** — `gpDivergDelta` scales the RHS and the diagnostic with the same ρ | — |
| 5 | the preconditioner/smoother sees the rescaled operator (RB-GS and VelocityMG are built on the modified `AC/AW/…`) | **absent** — the ghost MG hierarchy is rediscretized from *binary openness* and carries no ρ at any level | **Measured to be a NON-issue, and "fixing" it is harmful.** spec(M⁻¹A) on a real bed: \|λ\|min = 0.32, λmax ≈ 2.15, spread ≈ 6.7 — a healthy preconditioned operator. Scaling the preconditioner by ρ as well takes the deferred-correction radius from 1.15 to **15.4** (the rescale is precisely what keeps the closure delta commensurate with S). **Do not do this.** |
| 6 | order reduction driven by conditioning | **absent in both** — QUAD→LIN→BC_ONLY is chosen purely by *source availability* (`src1`/`src2`) | **B1**: add a conditioning gate — when `D_f` < τ (measured: τ≈1e-2 touches ~0.3 rows/sphere), reduce the order at that face rather than emit a row whose ρ is 1e-3. Cut-cell's precedent says isolated order reduction is invisible globally. Gate with a counter and assert zero on the stable regression geometries |
| 7 | weights stored already-rescaled (`K = poly·R`, all O(1)) | **absent** — the overlay stores unscaled `w = poly/D` (max 971 measured, unbounded in principle) and multiplies by ρ at apply time | **B2**: store ρ·w. Bounded by construction, makes a magnitude cap enforceable at build time, and removes a large float intermediate |
| 8 | two-sided `D_sandwich = ξ⁻ξ⁺` closure with `poly_N_c_sandwich` + two-wall `Nbc`, entering the `D_rescale` min | **absent** — a sandwiched axis goes to `BC_ONLY` on both sides: zero flux, no φ coupling, ρ untouched | Judgement call, not obviously a bug: for a *flux* closure "no fluid unknown between the two walls ⇒ no flux" is defensible. Measured impact: BC_ONLY faces are 4 % (s100) to 27 % (s132) of closure faces, and they grow with bed size. **B3 (optional)**: a two-sided flux closure so a sub-cell slit still transmits |
| 9 | `if (|D_axis|<1e-9) R = 1` guard | no analogue needed (θ ≥ 1e-4 ⇒ D ≥ 1e-4) | — |
| 10 | left null vector is exactly the constants (symmetric operator), so mean removal is the exact compatibility projector | **ABSENT — this is the gap that survives every check.** Measured `\|w·1\|/(\|w\|\|1\|)` = 0.9925–0.9983 on real beds; `solveBiCGStab` already carries an explicit "compatibility-floor" stagnation guard because of it | **B0 (revised)**: make the compatibility projector match the operator — project the residual (and the RHS) on the *measured* left null vector instead of the constants, or symmetrise the split so the constant is exact (phase C). Deviation from the constants grows with the number of closure rows, i.e. linearly in sphere count — the only measured quantity that scales like the failure |
| 11 | no fragmentation concept needed (openness handles pockets) | ghost needs the host-BFS guard + a global allgather; pocket count grows with sphere count | works; noted as an extra global operation |
| 12 | — | EXTENDED-θ sliver branch (ghost-only) | **no gate needed** — measured benign (A2 §3) |

**Phase-B priority, revised by the measurements: B0 (compatibility projector) ≫ B1 > B2 > B3.**
The plan assumed closure hardening was the fix. The census says the closure coefficients are already
bounded, the clamp never fires, the sliver branch is benign, and the preconditioned spectrum is
healthy — so "full Robust-Scaling" in the literal sense has almost nothing left to add. The one
mechanism that is genuinely absent relative to cut-cell, and the only measured quantity that grows
the way the failure grows, is the **compatibility defect of the nonsymmetric operator** (row 10).
B1/B2 remain worthwhile as insurance and hygiene, not as the fix.

## A4 — symmetric split: ρ(S⁻¹N), and the mixed mode

`tests/study/ghost_projection_apriori.py --split` adds a sparse split probe (power iteration on
S⁻¹N with the solver's own gauge handling — mean removal around a pinned dof) that runs on
geometries the existing dense probe cannot reach. It reproduces the dense
`deferred-correction rate max|1−λ|` to 4 digits at N=16 (1.1191 both), so the two agree.

| geometry | matrix_order | ρ(S⁻¹N) | |1ᵀA|₁/(n·scale) | |w·1|/(|w||1|) |
|---|---|---|---|---|
| analytic sphere N=16 | 2 | 1.119 | 1.25e-2 | 0.9965 |
| analytic sphere N=24 | 2 | 1.172 | 7.78e-3 | 0.9978 |
| analytic sphere N=32 | 2 | 1.182 | 5.80e-3 | 0.9983 |
| analytic sphere N=24 | **1** | **0.381** | 1.86e-3 | 0.9992 |
| s100 bed N=32 (R=2) | 2 | 1.150 | 1.23e-2 | 0.9925 |
| s100 bed N=48 (R=3) | 2 | 1.202 | 9.89e-3 | 0.9957 |
| s100 bed N=32 (R=2) | **1** | **0.592** | 5.26e-3 | 0.9955 |
| s100 bed N=48 (R=3) | **1** | **0.665** | 3.55e-3 | 0.9979 |

The same probe also reports the health of what the solver actually runs — `spec(M⁻¹A)` with M the
binary-openness surrogate, via `|λ|min = 1/ρ(A⁻¹M)` by power iteration:

| geometry | matrix_order | \|λ\|min | λmax ≈ 1+ρ(S⁻¹N) | spread | ρ(S⁻¹N) with a ρ-aware M |
|---|---|---|---|---|---|
| analytic sphere N=24 | 2 | 0.423 | 2.17 | 5.1 | 1.71 |
| s100 bed N=32 | 2 | 0.321 | 2.15 | 6.7 | **15.4** |
| s100 bed N=32 | 1 | 0.408 | 1.59 | 3.9 | 7.8 |

**The preconditioned operator is healthy** — no near-zero eigenvalues, spread under 7 — so the
56–160 iterations/step are *not* explained by the preconditioner failing on the ρ-scaled rows. The
last column kills the tempting "make the preconditioner ρ-aware" fix outright: it takes the split
radius from 1.15 to 15.4, because ρ is exactly what keeps the closure delta commensurate with S.
The ρ rescale is right; the cost and the instability have to come from somewhere else, and A5 says
where.

**At the production order (2,2), ρ(S⁻¹N) ≈ 1.15–1.20 > 1: plain deferred correction DIVERGES.**
Phase C as literally specified ("S φ^{k+1} = b − N φ^k, under-relaxation knob in case ρ approaches
1") cannot be built on the order-2 matrix without damping, and ρ is *above* 1, not near it.

At matrix_order = 1 the split is contractive: ρ = 0.38 (single sphere) to 0.59–0.67 (real bed,
rising with R). So phase C's split has to be the **7-point (1,·) matrix**, which is what the plan
intended by "the existing two-weight-set seam". Outer count at ρ=0.67 is ~16 for 1e-3 and ~45 for
1e-8 — so a *Richardson* outer is not competitive; the outer must stay Krylov-accelerated (which is
what today's BiCGStab already is, just with a one-V-cycle rather than a converged inner).

**The mixed mode is not a baseline — it is a divergence generator, and it is our local repro.**
Running `matrix_order=1, rhs_order=2` through the actual march:

| bed | spheres | grid | cells | R | (2,2) | (1,2) |
|---|---|---|---|---|---|---|
| s100 | 489 | 128³ | 2.1 M | 8 | stable | stable |
| s100 | 489 | 224³ | 11.2 M | 14 | converged, k=0.0110391 | converged, k=0.0110391 (identical) |
| s102 | 978 | 256×128×128 | 4.2 M | 8 | — | stable |
| s104 | 1956 | 256×256×128 | 8.4 M | 8 | stable | stable |
| s108 | 3911 | 192³ | 7.1 M | 6 | stable | stable |
| s116 | 7823 | 192×96×96 | **1.8 M** | 3 | stable (max\|u\|=0.018) | **DIVERGES** (max\|u\|=1.8e2 @80) |
| s116 | 7823 | 256×128×128 | 4.2 M | 4 | stable (max\|u\|=0.030) | **DIVERGES** (max\|u\|=6.0e7 @80) |
| s116 | 7823 | 320×160×160 | 8.2 M | 5 | converged, k=0.012097 | **DIVERGES, k = 4.5e+77** |

The (1,2) threshold sits between **1956 and 7823 spheres and is resolution-independent** (s116
diverges at R = 3, 4 and 5; s104 and s108 are stable at R = 8 and 6). That is the *same control
variable* as the production (2,2) failure, whose threshold sits between 3911 and 7823 spheres —
(1,2) simply reaches it at a coarse resolution the workstation can hold. **The cheapest failing case
is s116 at R=3: 1.8 M cells, visible by step ~40, about a minute on one GPU.**

The s116 (1,2) blow-up is exponential with a *constant* rate — max|u| goes 0.15 (step 5) → 1.4e2
(21) → 1.1e6 (41) → 3.7e52 (261), i.e. **×1.62 per step, 0.21 decades/step** — while ⟨u⟩ stays
physical until ~step 20. A localized mode grows and eventually swamps the mean: the same signature
as the Snellius (2,2) k → ±1e57/1e58, and the same bed-size dependence (small bed fine, big bed
blows up).

This is decisive for phase C's *design*: the (1,2) mode **is** "S-split with N deferred, relaxed
through the time stepping", and the documented claim that "the operator mismatch converges through
the time stepping (measured rate ~0.4)" is only true where the bed is small. The linear DC rate
(ρ=0.38–0.67, matching the documented ~0.4) does **not** control the march, because each step also
advances momentum with the mismatched pressure — the amplification factor of the *step map* is a
different, bed-size-dependent quantity, and it exceeds 1 here.

**⇒ Phase C must converge the outer iteration WITHIN a step** (outer residual measured on the true
operator A, as the plan already says), never across steps. Deferring the correction to the next time
step is exactly the thing that is broken today.

## A5 — compatibility bias, and the actual growth channel

**Right null space is clean.** `A·1` = 1e-15 on every geometry — the closure deltas telescope for a
constant φ, exactly as designed.

**The left null vector is not the constants**, so mean removal is not the exact compatibility
projector: `|w·1|/(|w||1|)` = 0.9925–0.9983 on real beds (a 6–12 % orthogonal component), and the
relative column-sum defect `|1ᵀA|₁/(n·scale)` is 5e-3 – 1.3e-2 at order 2, 2–4× smaller at order 1.
That is a real structural difference from cut-cell and is why `solveBiCGStab` carries its
"no improvement for 30 iterations" guard.

**But it is not what sets the iteration count, and it is not the divergence driver.** An rtol sweep
on the running solver settles that — if the solve were floor-limited, tightening rtol would buy
nothing:

| case | rtol 1e-4 | 1e-6 | 1e-8 | 1e-10 | iters per decade |
|---|---|---|---|---|---|
| ghost (2,2), s116 R=5 | 56.2 | 83.2 | 109.9 | 137.2 | **13.5** |
| ghost (2,2), s100 R=8 | 37.4 | 55.2 | 70.6 | 85.4 | **8.0** |
| cut-cell, s116 R=5 | — | 10.9 | 15.5 | — | **2.3** |

The ghost solve converges cleanly and linearly in log(rtol) all the way to 1e-10 — **no floor above
1e-10**, so no iterations are being burned on stagnation at the production tolerance. (The a-priori
`|w·b|/|b|` ≈ 1e-3 gap uses a *manufactured* RHS; the real `−div(u*)` is produced by the previous
projection and is far closer to compatible, which is why the floor never shows.) The honest cost
statement is a **rate** difference: 13.5 iterations per decade against cut-cell's 2.3, i.e. a
per-iteration contraction of 0.84 vs 0.37, grid-independent but growing with geometric complexity
(8.0 → 13.5 from 489 to 7823 spheres). That is where the reported 4–7× lives.

**The growth channel, measured directly.** On the cheap failing case (s116 R=3, mixed (1,2), 1.8 M
cells) three knobs separate the candidates:

| knob | result at step 100 |
|---|---|
| baseline (ω_p = 1) | max\|u\| = 5.96e+03 — diverging |
| `set_pressure_underrelax(0.5)` | max\|u\| = **1.790e-02** — stable; the (2,2) control gives 1.795e-02 |
| `set_pressure_underrelax(0.2)` / `(0.05)` | stable (1.76e-02 / 1.81e-02) |
| the same knob on the harder R=5 instance | max\|u\| at step 120: 2.9e+04 (ω_p=0.2), 1.1e+02 (0.1), 1.2e+00 (0.02) vs a correct 4.9e-02; ω_p=0.5 still reaches k = 8.3e40 over 400 steps — monotone in ω_p, never zero |
| PRTOL 1e-12 | max\|u\| = 5.9569e+03 — **identical to baseline to 5 digits** |
| PRTOL 1e-4 | max\|u\| = 6.00e+03 — same divergence, 25 iters/step instead of 72 |
| DT 6 (10× smaller) | max\|u\| = 4.75e+92 — **much worse** |

So: the instability is **completely insensitive to how well the linear system is solved** (1e-4 and
1e-12 give the same trajectory to five digits — it is a property of the scheme, not of the solver),
and its growth rate is **monotonically controlled by under-relaxing the incremental-rotational
pressure accumulation** — enough to stabilise the mildest instance outright, and enough to slow the
harder one by 40 decades, but never enough to remove it. The amplifier is `P += (ρ/dt)·φ`
integrating a per-step residual that feeds back positively; ω_p turns the gain down, and a smaller
dt (more accumulation steps per unit physical time) turns it up.

For the (1,2) mode the per-step residual has an obvious source: the solve enforces `A₁φ = −div₂(u*)`
while the true post-correction divergence is `div₂(u* − ∇φ) = −(A₂−A₁)φ`, i.e. the operator/RHS
mismatch is re-injected every step. That also explains the bed-size threshold — the mismatch lives
on closure rows, whose count is linear in sphere count.

ω_p is therefore a **gain knob on the amplifier, not a repair of the source** — useful as a
diagnostic and, at ω_p = 0.5, measured to be accuracy-neutral (go/no-go below), so it is free to try
on the production rungs as one more data point.

## Go / no-go and predicted payoff

**Phase B: GO, but almost entirely re-scoped.** Every specific item the plan listed under "full
Robust-Scaling" is measured to be a non-issue — the θ clamp never fires, the sliver branch is benign,
the closure weights are bounded, the rescale is threaded consistently through matrix, RHS and
diagnostic, and the preconditioned spectrum is healthy. Ghost's ρ is not a partial transplant of
`D_rescale`; for a single-sided closure it is algebraically the same thing. What remains:

- **B0 — compatibility projector (the one real structural gap).** Project the residual and RHS on
  the operator's actual left null direction instead of the constants, or symmetrise so the constant
  is exact. Predicted payoff: it is the only measured structural difference from cut-cell that grows
  linearly with sphere count. *But note it does not explain today's iteration count* (no floor above
  rtol 1e-10), so do not expect a cost win from it — expect a robustness win.
- **B1 — conditioning-driven order reduction (τ on `D_f`).** Insurance, not the fix: QUAD→LIN→BC_ONLY
  is currently chosen by source availability only, and a τ ≈ 1e-2 gate would retire ~0.3 rows per
  sphere at a cost that cut-cell's precedent says is invisible. Gate with a counter; assert zero
  firings on the current regression geometries so those builds stay byte-identical.
- **B2 — store ρ·w rather than w.** Mechanical; bounds the stored coefficients by construction and
  makes B1's cap enforceable at build time.
- **B3 — two-sided sandwich flux closure.** Optional. BC_ONLY faces run 4 % (s100) to 27 % (s132) of
  closure faces, so it is worth one measurement, but nothing points at it.
- **Do NOT** widen/tighten `GP_THETA_MIN`, add an EXTENDED-θ gate, or make the preconditioner ρ-aware
  (that last one is actively harmful: split radius 1.15 → 15.4).
- **New acceptance test, free:** the (1,2) mode on s116 must stop diverging — cheapest instance
  192×96×96, 1.8 M cells, ~1 min. Also available: the all-pairs-near-tangent lattice generator
  (`make_lattice_bed.py`) for the plan's near-tangent case.

**Phase C: GO, but re-specified — and re-cost it first.**

- The split must use the **7-point matrix_order = 1 operator** as S+N: ρ(S⁻¹N) = 0.38 (single sphere)
  to 0.59–0.67 (real bed). At order 2 the split is **not contractive** (ρ ≈ 1.15–1.20), so the plan's
  "under-relaxation knob in case ρ approaches 1" does not apply — ρ is above 1, not near it.
- **The outer iteration must live inside the step**, converged on the true operator A. "The operator
  mismatch converges through the time stepping" is exactly what the (1,2) mode does, and it is now
  measured to diverge above a bed-size threshold. This is a fact, not a risk.
- Re-cost before building. The plan's premise was that a symmetric S makes the inner solve
  cut-cell-like (~12–30 iterations) so the total lands within 2× cut-cell. But the *measured* per-
  decade rates are ghost 13.5 vs cut-cell 2.3, with a healthy exact-S spectrum (spread 6.7) — i.e.
  most of the gap is the quality of the **binary-openness V-cycle** on a staircased, pocketed domain,
  which an S-split does not change. With ρ(S⁻¹N) ≈ 0.67 the outer needs Krylov acceleration anyway,
  so phase C's structure ends up close to today's (BiCGStab on A preconditioned by S), just with a
  more converged inner. **The cheap experiment that decides C's value: measure how many V-cycles the
  binary-openness operator needs on its own** (versus the geometric-openness cut-cell operator on the
  same bed). If the binary V-cycle is the bottleneck, the productive work is improving *that*
  hierarchy — not restructuring the outer solve.
- The claim that S's clean null space "structurally eliminates the H2 bias channel" holds for the
  inner solve only; the outer still stops on A's residual and inherits A5's defect.

**A cheap thing to try on the production failure, with expectations set correctly.**
`set_pressure_underrelax(ω_p)` is measured to be **accuracy-neutral on a steady march**: ghost (2,2)
gives k/R² = 0.0113644 (ω_p = 1) vs 0.0113645 (ω_p = 0.5) on s100 R=8, and **0.0120974 vs 0.0120974**
on s116 R=5 — identical to six digits, at the cost of ~20–25 % more march steps (150 → 185, 210 →
250). That is expected: at steady state φ → 0, so the accumulation term ω_p multiplies vanishes and
only the path changes. So it is free to try. But it is a damper: on the harder local instance it
reduced the growth rate monotonically without ever stabilising it, so **do not plan on it as the
fix** — plan on it as one extra data point (if the np=16 rung survives at ω_p = 0.5, the production
blow-up shares this amplifier; if it merely grows more slowly, that is equally informative).

**Still open / needs Snellius (one rung).** The (2,2) production divergence itself is *not*
reproduced and its trigger is still confounded: the weak ladder moves bed size and np together, while
the refine ladder cleanly acquits np=32, a 1024³ grid and full MG depth (489 spheres at R=64 is
perfectly stable). Note also that (2,2) has no operator/RHS mismatch by construction, so it cannot be
the (1,2) mechanism verbatim — but it can share the accumulation channel. The disambiguating rung:
the same 7823-sphere bed at R=16 on np=8 fat ranks vs np=16, plus one repeat with ω_p = 0.5.

## Reproducing this

New knobs, all additive and off by default: `PECLET_FLOW_GP_DEBUG` (+ `_FILE`) in flow;
`TRACE`, `GPORDER`, `PUNDER` in `peclet-examples/benchmarks/porous-scaling/spheres_bench.py`;
`--split / --bed / --pair / --split-n / --matrix-order / --no-leftnull` in the a-priori harness.

```bash
# overlay census / per-row dump (no-op unless the env var is set)
PECLET_FLOW_GP_DEBUG=1 PYTHONPATH=<build> python spheres_bench.py
PECLET_FLOW_GP_DEBUG=2 PECLET_FLOW_GP_DEBUG_FILE=rows PYTHONPATH=<build> python spheres_bench.py

# THE local failing case: 1.8 M cells, diverging by step ~40, ~1 min on one GPU
cd peclet-examples/benchmarks/porous-scaling
PACK=results/packings/packing_1024x512x512_r16_phi0.50_s116.npz \
GNX=192 GNY=96 GNZ=96 IBM=ghost GPORDER=1,2 TRACE=1 WARMUP=0 NSTEPS=1 MARCH_MAX=100 \
  mpirun -np 1 python spheres_bench.py
# ... and the knob that switches it off:            PUNDER=0.5
# ... proof it is not the linear solve:             PRTOL=1e-12   (identical trajectory)
# ... the stable control:                           GPORDER=2,2

# all-pairs-near-tangent lattice bed (phase-B acceptance geometry)
python make_lattice_bed.py lat.npz <n_per_axis> <gap_in_R> <jitter_in_R>

# A4/A5 probes (CPU only)
python tests/study/ghost_projection_apriori.py --split --split-n 16 24 32 [--matrix-order 1]
python tests/study/ghost_projection_apriori.py --split --bed <cubic packing.npz> --split-n 32 48
python tests/study/ghost_projection_apriori.py --split --pair 0.02 --split-n 32
```

Raw logs for every run quoted here are in the session scratchpad (A1 ladder + lattice, A2 census,
A4 threshold ladder, A5 rtol/mechanism/omega sweeps); the numbers in this note are copied from them.
