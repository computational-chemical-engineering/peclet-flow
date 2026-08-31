# Work orders — V3 (curvature), V4 (balanced-force surface tension), and the precision campaign

Companion to `vof_workorders.md` (**its "Shared preamble" applies verbatim** — build, hard
rules, escalation policy, the never-`git add -A` and capped-solve rules) and to
`suite/docs/VOF_PLAN.md` §4.

All three authorized by the user 2026-08-31. **WO-P (V4) depends on WO-O (V3)**; WO-M is
independent of both but touches shared solver internals, so run it concurrently with at most
one other heavy agent (the np=4 `MPI_ERR_TRUNCATE` race of WO-L is load-triggered).

State entering these rungs: V0 (PLIC toolbox), V1 (WY advection), V2a (colour field in the
solver), V2b (momentum-consistent transport) all landed and gated on host + CUDA. Four
production defects fixed along the way (rank-unaware domain BCs, body-force ghosts,
`drag_beta` ghosts, the MG prolongation ghost).

---

## WO-M (precision campaign) — is the solver's fp32 operator storage costing us accuracy?

**Authorized.** Two independent loci are now on the table and they are *not* the same code:

1. **The multigrid operator** (`mac_cutcell_mg.hpp:51`, `using MReal = float`). The
   collocated-paper session A/B'd it with a compile switch on a high-contrast periodic bed:
   float → PCG rebounds 8e-7 → 3.8e-5 and cap-burns, FCG floors at 2e-6, Chebyshev cap-burns;
   double → clean monotone convergence to rtol 1e-8 in ~86 iterations. Only the storage
   changed. Mechanism: float rounding breaks `A·1 = 0` per row at eps_f32, and under ~3
   decades of contrast the defect on rows mixing large and tiny couplings is ~1e-4 *relative
   to the tiny coupling*, shifting the near-null vector off the constant that mean-removal
   deflates — i.e. the documented `:1411` agglomerated-bottom failure generalised to every
   level.
2. **The momentum/velocity operator** (`flow_ibm.hpp:61`, `using FV = View<MReal*>`). Found
   independently by WO-K: `buildRhsVar*` forms `ρ_f/dt·u` in double while
   `ibmBuildDiffusionVar` stores the same `ρ_f/dt` in the float stencil, so `u* = b/diag`
   differs from its input at float epsilon **whenever ρ varies** (uniform ρ gives exactly
   0.0). This is what floors V2b's uniform-velocity identity at ~1e-7; in a double build it
   is ~1e-15. No pressure solve came near its cap in those runs, so this locus is distinct
   from (1).

**Do, in this order — the first step may cancel the third.**
1. **Settle the S3 question.** Rerun WO-H's dense-preconditioner probe (assemble M by one
   V-cycle per unit basis vector, LDLᵀ of `sym(M)`, look for a negative pivot) on a
   **double** build. WO-H measured the pivot at contrast ~1e3 wall-bounded and ~1e4 periodic
   on a *float* hierarchy, so that evidence is contaminated. **Pivot disappears in double ⇒
   S3's coefficient-coarsening theory loses its evidence and must be struck from the plan;
   pivot survives ⇒ S3 stands as an independent mechanism.** Report either way; this is the
   single most valuable measurement in this work order.
2. **Quantify what precision actually costs us**, on cases we care about rather than
   synthetic ones: the varRho hydrostatic acid tests, V2b's uniform-velocity identity at
   ratios 1e1–1e4, the Ergun porous bed, the Z&H sphere drag, and the single-phase regression
   suite — each float vs double, reporting accuracy AND wall-time/memory. A precision change
   that costs 2× memory bandwidth for an accuracy gain nobody needs is not worth shipping.
3. **Then choose a policy from the measurements**, not in advance. The candidate the
   collocated session proposes, and the one I'd expect to win, is the **double-diagonal**
   variant: keep the six face coefficients in float, store and resum the diagonal in double
   so `A·1 = 0` holds exactly — +4 B/cell against +28 B for a full fp64 hierarchy, and it
   generalises the fix already proven at `:1411`. Evaluate it against full-fp64 and against
   doing nothing. **If the measurements say the current float storage is fine for the
   accuracies we actually need, say so and ship nothing** — that is a valid and valuable
   outcome.

**Coordinate, do not duplicate.** The collocated-paper session owns the `MReal` compile
switch (`-DPECLET_FLOW_MREAL_DOUBLE`, smoothers/matvec templated on the coefficient view
type) and has an experiment tree at `/projects/0/prjs1022/peclet/suite/flow_mreal64/`, with
data under `flow/doc/data/collocated_campaign/`. Read what is on flow main before writing
anything, and prefer extending their switch to inventing a second one. **Do not rsync into
the shared cluster tree `/projects/0/prjs1022/peclet/suite/flow/src`** — it is a known
collision zone; if you need the cluster, use your own tree.

**Gates.** Default build byte-identical (the switch must be inert when off): single-phase
regression +0.00% with identical iteration counts; kernel and MPI ctests green host + CUDA.
Any policy you ship must state its measured accuracy gain and its measured cost.
**Build note:** CUDA 12.x nvcc ICEs on the pre-templating code; use their templates or CUDA 13.

---

## WO-N (re-baseline) — record the post-fix porous numbers  [small, run after WO-M or alone]

**Authorized.** WO-I's `drag_beta` fix moved the porous Ergun bed from **3.2258 %** relative
error to **~5e-8**, and identified the recorded "~3 % Ergun agreement" in
`flow/doc/porous_drag_scheme.md` §5 as *that bug* rather than a closure residual. WO-I
deliberately left every recorded number untouched pending this authorization.

**Do.** Re-run the porous verification set, update `porous_drag_scheme.md` §5 (and any other
doc or `perf_baseline*.json` carrying a superseded porous number) with the post-fix values,
and **keep the old number visible with its explanation** — a struck-through or "was 3.2258 %,
identified as the halved face drag of WO-I" line, not a silent overwrite. Anyone who read the
old number needs to find out why it changed. Note in passing whether the fix changes any
conclusion drawn from those numbers (e.g. whether a closure was tuned to compensate).

**Gates.** Numbers reproducible on a re-run; the provenance note present; nothing else edited.

---

## WO-O (rung V3) — height-function curvature with a paraboloid-fit fallback  [OPUS]

**The design is settled** (`VOF_PLAN.md` §2, §4 V3) — implement it, do not re-litigate it:
the Popinet (2009) HF cascade — standard height function where monotone columns exist →
mixed-direction HF → **PLIC-volumetric paraboloid fit on a 5³ Wendland-weighted stencil**
where they cannot be assembled (the best cost/accuracy fallback per Han, Evrard & Desjardins,
*IJMF* 2024).

**Two facts from the literature that the gates must respect, not fight:**
- HF *always* needs the fallback somewhere below ~4–5 cells per diameter — i.e. in every
  under-resolved pore throat. Han et al. measured the fallback firing in up to 0.9 % of
  interfacial cells even at D/Δ ≈ 102. A cascade that never falls back is a cascade with a
  bug, not a triumph.
- With advection-realistic (not exact) volume fractions, **curvature error stops converging
  below CΔ ≈ 1e-2 for every known method**. So a plateau on fine grids is the physics of the
  method, not a defect to chase — measure it and report it.

**Precedent from this campaign to reuse**: V0 measured that MYC's *normal* error does not
converge while its *reconstruction* error does. Curvature must come from column sums of C
(and the fit), **not** from the MYC normal — that is the whole reason the plan rejected
∇C-based geometry. If you find yourself differentiating a normal field, stop.

**Do.** `src/vof/curvature.hpp`, container-free kernels under the WO-D signature rule
(scalars and small local arrays only — these are scheduled for promotion to
`peclet::core::vof` so AMR and the bubble-block container can share them). Register a
curvature field; expose it for inspection. The colour field's g=3 halo already gives the
7-cell column reach.

**Gates.**
- Static convergence with **exact** volume fractions on a sphere: 2nd order, 16³→32³→64³.
- Sphere sweep including **D/Δ < 5** where the fallback must engage: report the fraction of
  interfacial cells taking each cascade branch at each resolution, and assert no cell returns
  NaN or a silently-zero curvature.
- Curvature of a **translating** droplet (advection-realistic fractions): report the error
  plateau and the CΔ at which it sets in — this is a measurement, not a pass/fail.
- Compare against an analytic κ = 2/R everywhere; report max and L1 error separately (the max
  is where the cascade's weakest branch shows).
- Device/host bitwise agreement on the same backend; np 1/2/4 bitwise; single-phase
  regression +0.00%.

---

## WO-P (rung V4) — balanced-force CSF and the capillary time step  [OPUS, after WO-O]

**The one rule that makes this work** (Francois et al. 2006; Popinet 2009/2018): the surface
tension force `σκ∇C` must be evaluated **at the same location and with the same discrete
gradient operator as the pressure gradient**. On the staggered MAC grid that means face
centres, using the identical face-difference the projection uses. Get this right and the
discrete equilibrium is exact to machine precision; get it wrong by even a consistent-looking
interpolation and spurious currents appear at O(σκ/μ) and no amount of curvature accuracy
will remove them.

**Do.** The CSF force through the existing per-cell force-field machinery (which
`buildRhsVar` already face-interpolates — WO-G fixed its ghosts, so that path is now sound),
with the face value formed by the projection's own operator. Add the capillary time-step
constraint `Δt < sqrt((ρ₁+ρ₂)Δx³/(4πσ))` (Brackbill 1992; Denner & van Wachem 2015 confirmed
the prefactor is the true stability boundary and that it scales with Δx^{3/2} and the *sum*
of densities) and fold it into the step limiter alongside V2a's interface-local CFL.

**The loud gate is the stationary droplet**: an exactly balanced σκ∇C against the pressure
jump must leave **max|u| at machine zero**, independent of viscosity and resolution
(Popinet 2009). This is the momentum analogue of the hydrostatic acid test and it fails
loudly on any force/pressure operator mismatch. Report the spurious capillary number; the
pore-scale budget is Ca ≲ 1e-7 and naive CSF gives ~1e-2.

**Further gates**: capillary wave vs Prosperetti's analytical dispersion; oscillating droplet
vs Lamb's frequency; the Hysing rising-bubble benchmark (both cases) against the published
reference; **and the falling-drop terminal velocity WO-K had to defer** — it needs surface
tension to hold the drop together, and WO-K's substitute reached only 9 % of
Hadamard–Rybczynski with a suspected under-resolved momentum solve, so re-run it here with a
converged momentum solve before claiming Arrufat's "within 15 % at 15 cells/diameter".
np 1/2/4 bitwise; single-phase regression +0.00%.

**Watch for**: at pore-scale Ca the capillary Δt, not the WY CFL, becomes the binding step
limit — confirm that and record the resulting step-count economics, because it decides
whether implicit surface tension ever needs revisiting (Popinet 2018 says not yet).

---

# Findings

### WO-M step 1 — the S3 question, settled: **the negative pivot SURVIVES in double**

**Verdict: S3 (coefficient-aware coarsening) stands as an independent mechanism.** The V-cycle
preconditioner's indefiniteness under high coefficient contrast is *not* an artefact of the float
operator storage. Float and double hierarchies produce the **same preconditioner spectrum to three
or four significant figures** on every configuration measured.

**Instrument** (committed, so this is reproducible rather than throwaway):
`tests/study/mg_precond/` — `mg_dense_precond.cpp` assembles the preconditioner `M` densely, one
mean-removed unit basis vector per column through the exact `precond` lambda `solvePCG` uses (a
zero-iteration `solvePCG` first puts the V-cycle on the production `(pre, post, bottom) = (2, 2, 12)`
schedule), plus the fine operator `A` via `matvecOverlap`; `mg_precond_analyze.py` reports the skew
`||M−Mᵀ||_F/||M||_F`, the unpivoted LDLᵀ pivots (WO-H's instrument, for comparability) **and** the
eigenvalues of `sym(M)` restricted to the mean-free subspace. Build it twice against the same
prefix, the second time with `-DPECLET_FLOW_MREAL_DOUBLE`; nothing else differs.

8³, 3 levels, sharp mid-height ρ-slab, `c_f = ρ₀/ρ_f` (`buildRhoCoeff`), host-openmp:

| geom | ratio | prec | skew | LDL neg | λ_min(sym M) | λ_max | neg eigenvalues |
|---|---|---|---|---|---|---|---|
| periodic | 1 | float / double | 1.14e-02 / 1.14e-02 | 0 / 0 | 8.333e-02 | 1.534e+00 | 0 / 0 |
| periodic | 1e2 | float / double | 2.82e-02 / 2.82e-02 | 0 / 0 | 8.750e-02 | 8.947e+01 | 0 / 0 |
| periodic | 1e3 | float / double | 2.86e-02 / 2.86e-02 | 0 / 0 | 8.222e-02 | 8.907e+02 | 0 / 0 |
| **periodic** | **1e4** | float / double | 2.87e-02 / 2.87e-02 | **6 / 6** | **−3.301e+00** | 8.903e+03 | **6 / 6** |
| periodic | 1e5 | float / double | 2.87e-02 / 2.87e-02 | 14 / 14 | −4.627e+01 | 8.902e+04 | 14 / 14 |
| periodic | 1e6 | float / double | 2.87e-02 / 2.87e-02 | 20 / 20 | −4.759e+02 | 8.902e+05 | 20 / 20 |
| wallz | 1 | float / double | 9.38e-03 / 9.38e-03 | 0 / 0 | 8.431e-02 | 5.914e+00 | 0 / 0 |
| wallz | 1e2 | float / double | 5.26e-02 / 5.26e-02 | 0 / 0 | 8.748e-02 | 3.499e+02 | 0 / 0 |
| **wallz** | **1e3** | float / double | 5.35e-02 / 5.35e-02 | **1 / 1** | **−1.077e+00** | 3.485e+03 | **1 / 1** |
| wallz | 1e4 | float / double | 5.35e-02 / 5.35e-02 | 5 / 5 | −2.375e+01 | 3.483e+04 | 5 / 5 |
| wallz | 1e5 | float / double | 5.35e-02 / 5.35e-02 | 9 / 9 | −2.508e+02 | 3.483e+05 | 9 / 9 |
| wallz | 1e6 | float / double | 5.35e-02 / 5.35e-02 | 12 / 12 | −2.521e+03 | 3.483e+06 | 12 / 12 |

The onsets reproduce WO-H's: **first real negative at ratio ~1e3 wall-bounded and ~1e4 fully
periodic**. Confirmed at 16³ / 4 levels (float and double again identical to every printed digit,
periodic 1e3 now carrying **2** negative eigenvalues where 8³/3-levels carried 0 — the defect grows
with hierarchy depth, exactly as the ladder's "degrades with depth" observation).

**Two methodological corrections to the WO-H record, both of which strengthen it:**
1. WO-H's "1 negative pivot (**−1.1e-12**), ratio 1e2" is **not** indefiniteness. `sym(M)` is
   singular by construction (the constant is an exact null direction: the input is mean-removed and
   the level-0 V-cycle mean-removes its output), so an unpivoted LDLᵀ produces one pivot at
   round-off whose *sign is noise* — it flips between the float and double builds on the same
   configuration in this battery. The real negatives (−1.08, −3.30, −23.8, …) are 12 orders of
   magnitude larger and are unambiguous. Read the eigenvalue column, not the pivot count.
2. Conversely, an unpivoted LDLᵀ **breaks down** near the transition: at wallz 1e3 with the (2,2,4)
   schedule it reported a −1.25 pivot while `sym(M)` was still positive semi-definite. The
   restricted spectrum is the reliable instrument.

**The float row-sum defect is real, is contrast-amplified exactly as claimed, and is nevertheless
not this failure.** The probe measures `A·1` per row directly:

| ratio | float, abs | float, /max\|a\| | float, /min\|a\| | double, /min\|a\| |
|---|---|---|---|---|
| 1 | 0 (exact) | 0 | 0 | 0 |
| 1e2 | 1.14e-07 | 1.14e-07 | 5.74e-06 | 1.39e-15 |
| 1e3 | 5.43e-08 | 2.33e-07 | 2.72e-05 | 7.73e-14 |
| 1e4 | 1.85e-07 | 4.37e-07 | 9.26e-04 | 1.66e-13 |
| 1e6 | 9.27e-08 | 2.27e-07 | **4.63e-02** | 4.99e-11 |

So the collocated session's mechanism is confirmed as a *fact about the stored operator* — at three
decades of contrast the row-sum defect reaches ~1e-4 …1e-2 **relative to the small couplings**,
which is what shifts the near-null vector off the constant that mean-removal deflates — but it does
**not** move `M`'s spectrum. The two loci are therefore genuinely separate failures with separate
signatures: **contrast ⇒ M indefinite (precision-independent, kills the CG-family β, Chebyshev
survives)**, and **float storage ⇒ A·1 ≠ 0 (precision-dependent, floors/rebounds the residual near
the deflation floor, hurts every driver including Chebyshev)**. S3 is not struck from the plan.

Reproduce:
```bash
cmake -S tests/study/mg_precond -B build_wom_probe_f -DCMAKE_PREFIX_PATH=$PWD/../extern/install/host-openmp
cmake -S tests/study/mg_precond -B build_wom_probe_d -DCMAKE_PREFIX_PATH=$PWD/../extern/install/host-openmp \
      -DCMAKE_CXX_FLAGS=-DPECLET_FLOW_MREAL_DOUBLE
cmake --build build_wom_probe_f -j && cmake --build build_wom_probe_d -j
OMP_NUM_THREADS=8 OMP_PROC_BIND=false ./build_wom_probe_f/mg_dense_precond --sweep --outdir doc/data/wom
OMP_NUM_THREADS=8 OMP_PROC_BIND=false ./build_wom_probe_d/mg_dense_precond --sweep --outdir doc/data/wom
python tests/study/mg_precond/mg_precond_analyze.py doc/data/wom
```
