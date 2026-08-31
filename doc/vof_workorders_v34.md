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

**⚠ PREMISE CORRECTED BY WO-M — do not record "~5e-8" as machine precision.** WO-M measured
the same flow-side porous path against an *exact* reference and found the post-fix residual
**4.768e-08 in float and 2.776e-16 in double**. So WO-I's post-fix Ergun figure is **the fp32
operator-storage floor, not machine precision and not the closure's true accuracy**. Record it
that way: the drag bug accounted for the 3.2258 %, and what remains is an arithmetic floor of
the current storage precision, which a double build (or the recommended double-diagonal)
removes by eight orders. Stating "~5e-8, machine precision" would bake a second wrong number
into the docs in the act of fixing the first.

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

---

## Findings log (v3/v4 work orders)

### WO-O (rung V3) — HF curvature cascade + PLIC-volumetric fallback — **DONE 2026-08-31**, one design deviation

Delivered: `src/vof/curvature.hpp` (container-free `KOKKOS_INLINE_FUNCTION`s only, WO-D signature
rule — no `View`, no indexing, no halo types, so the V4 promotion to `peclet::core::vof` is a file
move) + `src/vof/curvature_field.hpp` (the view-level driver, the same `plic.hpp`/`colour_field.hpp`
split V2a already uses) + `IbmSolver::computeVofCurvature()` registering the `"kappa"` and
`"kappa_branch"` G=2 fields + five Python entry points + `tests/kokkos/test_vof_curvature.cpp`
(ctest `vof_curvature`, 7 gates, ~3 s) + `tests/kokkos_mpi/test_vof_curvature_mpi.cpp`
(`vof_curvature_mpi_np{1,2,4}`).

**The cascade as built** (Popinet 2009; Han, Evrard & Desjardins, *IJMF* 174:104769 (2024),
arXiv:2304.08643 — retrieved and read, all equations followed rather than approximated):

| tier | branch code | what it is | ships |
|---|---|---|---|
| 1 | `kCurvHf` = 1 | standard HF: 7-cell column sums of C over the 3×3 transverse patch in the direction of the largest \|n_d\|, Han eqs. (3)–(5) | ON |
| 2a | `kCurvHfMixed` = 2 | the same in the two remaining directions, in order of decreasing \|n_d\| (Basilisk `height_curvature`'s `foreach_dimension`) | ON |
| 2b | `kCurvHfFit` = 3 | Popinet's *generalised* HF: a paraboloid through the interface positions of whichever of the 27 columns closed | **OFF** — see the deviation below |
| 3 | `kCurvPv` = 4 / `kCurvPvReduced` = 5 | the **PV** method (Jibben et al. 2019; Han et al. §2.3): fit a paraboloid so that the volume under it matches the volume under the PLIC polygons, over a **5³ stencil with a Wendland C2 weight of width 2.5 cells** — Han's recommended configuration. Normal equations eqs. (13)–(15), assembled from Green's-theorem monomial integrals over each polygon's projection | ON |
| — | `kCurvNoEstimate` = 6 | fewer than 3 usable polygons: no estimate, and it says so | must never fire |

Sign/units: `kappa = 2H` in 1/h (cell units), **positive for a convex blob of liquid** — sphere
`+2/R`, cylinder `+1/R`, plane `0`, all three gated (gate B2 exists so the factor in `kappa = 2H` is
measured and not assumed). **Curvature never differentiates a normal field** (WO-D finding 2, and
the whole reason `VOF_PLAN` §0 rejected ∇C geometry): tiers 1–2 read column sums of C, tier 3 reads
PLIC volumes. The MYC normal appears twice, both times categorically — it *orders* the candidate
column directions (Basilisk does the same) and it *defines the fit frame*.

**No new halo, by design and it fits exactly.** Tier 1/2 reach 3×3×7 = ±3; tier 3 reaches a 5³ of
MYC stencils = 7³ of colour = ±3. The colour field's existing g = 3 is precisely the reach and not
one cell more. The cascade has **no reduction in it at all**, so it is bitwise
decomposition-independent by construction — which is why the MPI gate below is literally bitwise and
not at the usual reduction-order floor.

**Gate results** (host-openmp and nvidia-cuda unless noted).

| gate | result |
|---|---|
| A geometry primitives: the plane∧cube polygon vs the analytic `\|Γ\| = \|m\|₂ dV/dα` and `x̄ᵢ = −(dV/dmᵢ)/(dV/dα)`, 4000 random planes | max rel area error **3.6e-10**, max centroid error **9.6e-10** (both limited by the finite difference, not the polygon) |
| A Green's-theorem monomials on the unit square | **0.0e+00** exact; random pentagon vs a 2000² quadrature oracle **5.2e-6** |
| B2 oblique plane, exact fractions | kappa L1 **4.5e-15**, max **1.5e-14** against the analytic 0 |
| B2 cylinder R = 0.3 at 48³ | L1 **2.8e-3**, max **4.8e-3** against `1/R` — the mean-curvature factor is gated |
| **B static convergence, exact fractions, sphere R = 0.3, 16³→32³→64³** | L1 **3.03e-2 / 5.94e-3 / 1.32e-3**, fitted order **2.26**; max **5.01e-2 / 1.32e-2 / 3.79e-3**, order **1.86** |
| B initialization ablation (octree `levels` 4 / 6 / 8 at 32³) | L1 5.9377e-3 / 5.9410e-3 / 5.9412e-3 — the manufactured fractions are **not** the limiting error |
| C sphere sweep D/Δ = 2.8 … 48, no NaN, no silently-zero | branch table below; **0 NaN and 0 `kCurvNoEstimate` on every rung** |
| D translating droplet, advection-realistic fractions | plateau measured, table below |
| E device kernel vs serial host loop, same backend | host-openmp **32768/32768 bitwise**, max \|d\| 0.0. On CUDA the "serial host loop" is a *different* backend, so by the suite tolerance policy it is a tolerance comparison: 32062/32768 bitwise, max \|d\| **1.19e-15** (WO-D measured the same 1e-14-class spread on its round-trip battery) |
| F the PV fallback forced on every interfacial cell | L1 3.62e-2 / 9.27e-3 / 2.33e-3 — order **1.96, 1.99** |
| G tier-2b ablation | see the deviation |
| MPI np 1/2/4, host + CUDA, periodic and walls-on-the-cut-axis | **0 of 8192 cells differ, bitwise, on kappa AND on the branch field; the summed branch census equals the single-rank census exactly** |
| single-phase regression | **+0.00 % on all 13 grid points**, identical pressure-iteration totals, per-step medians and step counts on `zh_sphere` / `random_spheres` / `hollow_rings`; order p and the Richardson extrapolate unchanged to every printed digit. Run in an **isolated git worktree carrying only this WO's files** (the checkout is shared with WO-M, which has `mac_cutcell_mg.hpp` / `mac_velocity_mg.hpp` in flight), so the +0.00 % is attributable |
| full ctest batteries | `tests/kokkos` **25/25** on host-openmp AND nvidia-cuda; `tests/kokkos_mpi` **57/57** on host-openmp AND nvidia-cuda (was 54; the 3 new ones are `vof_curvature_mpi_np{1,2,4}`). On the CUDA leg two np=4 tests — `dragbeta_ghost_mpi_np4` and `vof_twophase_mpi_np4`, both pre-existing and both unrelated to this WO — hit the known **load-triggered `MPI_ERR_TRUNCATE` race of WO-L** while the GPU was time-slicing a second agent's full battery; each passes on re-run (8.7 s and 803 s). Not chased, per the work order. |

**Branch statistics per resolution** (exact-fraction sphere, off-centre by an irrational fraction of
a cell so it is never mesh-aligned):

| D/Δ | CΔ | interfacial | HF | HF other dir | PV | no estimate |
|---|---|---|---|---|---|---|
| 2.8 | 0.714 | 41 | 0 % | 0 % | **100 %** | 0 |
| 4.4 | 0.455 | 92 | 0 % | 0 % | **100 %** | 0 |
| 7.0 | 0.286 | 239 | 20.9 % | 0 % | 79.1 % | 0 |
| 12.0 | 0.167 | 681 | 54.3 % | 0 % | 45.7 % | 0 |
| 24.0 | 0.083 | 2728 | 74.0 % | 0 % | 26.0 % | 0 |
| 48.0 | 0.042 | 11437 | 80.6 % | 0 % | 19.4 % | 0 |

**The fallback rate is ~19 % at D/Δ = 48, not the ~0.9 % Han et al. report at D/Δ = 102 — and that
is geometry, not a defect.** Their 0.9 % is a **2-D** droplet, where the height patch is 1×3 and the
worst-case transverse slope at the preferred axis is 1, so a 7-cell column always closes. In **3-D**
the patch is 3×3 and the corner column at offset (±1,±1) must span `√2 · s` where the slope in the
preferred direction reaches `s = √2` on the octant diagonal (the largest normal component is
`≥ 1/√3`, with equality exactly there). That needs `√2·√2 + ½ = 2.5` cells of reach — the exact
capacity of a 7-column — so the diagonal band is permanently marginal and the failing fraction tends
to a **resolution-independent** ~19 %, which is what the table shows. Han et al. use **NH = 11** for
their 3-D static tests for precisely this reason ("a column height of NH = 11 is necessary to ensure
well-defined interface heights"); NH = 11 needs a g = 5 halo. **This is the one place where a wider
halo would buy something**, and it is not needed: the PV fallback serves those cells at measured
order 1.96–1.99 (gate F) and the combined cascade converges at 2.26 / 1.86.

**The advection-realistic plateau** (sphere R = 0.2 transported one full diameter along the diagonal
at CFL 0.25 by the V1 Weymouth–Yue advector, then curvature; the *same* geometry with exact
fractions is the control):

| D/Δ | CΔ | advected L1 | exact-fraction L1 | ratio | advected max |
|---|---|---|---|---|---|
| 6.4 | 0.3125 | 1.04e-1 | 7.79e-2 | 1.3× | 4.12e-1 |
| 12.8 | 0.1562 | 4.91e-2 | 1.43e-2 | 3.4× | 1.60e-1 |
| 25.6 | 0.0781 | 3.29e-2 | 3.20e-3 | 10.3× | 1.53e-1 |
| | | order 1.09 → **0.58** | order 2.45 → 2.16 | | order 1.36 → **0.07** |

**The plateau sets in between CΔ ≈ 0.16 and CΔ ≈ 0.08** and the max error is flat there already
(1.60e-1 → 1.53e-1, order 0.07) while the exact-fraction control on the identical geometry keeps
converging at 2.16. That is Han et al. §3 / Remmerswaal & Veldman's Lemma 3 reproduced: volume
fractions that are first-order in L∞ make the curvature zeroth-order, and the transition happens at
a *coarser* CΔ than their quoted ~1e-2 because our droplet is transported a full diameter (their
random-perturbation study injects a fixed `k·C` error instead). Nothing to chase; the number to
carry into V4 is that **at pore-scale resolutions the curvature error is set by the transport, not
by the estimator**, so V4's spurious-current budget should be spent on the force discretisation
(the balanced-force identity) rather than on a fancier curvature.

**DEVIATION — tier 2b (the mixed height function) is implemented, measured, and ships OFF.**
The WO specifies "standard HF → mixed-direction HF → PV fit". Both readings of "mixed" were built:
2a (try the other two column directions) and 2b (Popinet's generalised HF / Basilisk's
`height_curvature_fit`: a paraboloid through the interface positions of whichever columns closed).
Measured on the exact-fraction sphere at 16/32/64, everything else identical (ctest gate G, so the
numbers regenerate on every run):

```
tier 2b OFF   L1 3.03e-2 / 5.94e-3 / 1.32e-3  (order 2.26)   max 5.01e-2 / 1.32e-2 / 3.79e-3  (order 1.86)
tier 2b ON    L1 2.83e-2 / 7.27e-3 / 4.21e-3  (order 1.37)   max 6.08e-2 / 4.64e-2 / 6.07e-2  (order 0.00)
```

Tier 2b takes over exactly the 19.5–59.6 % of cells tier 1 cannot serve, and **destroys the
convergence of the max error** on cells the PV fallback handles at second order. It is not a
parameter choice: Wendland widths 1.5, 2.0, 2.5, 3.5 and 6.0 cells were swept and **none** converges
in the max (`PECLET_VOF_CURV_PTW` in the ctest reproduces the sweep). The mechanism is structural —
**its data set is the columns the height function could close, which is a *slope-selected* and
therefore asymmetric subset.** At a cell whose normal is near an octant diagonal the failing columns
are exactly the corner ones on the steep side, so the surviving points sample the interface
asymmetrically about the target and the quadratic fit picks up a lever-arm bias; that selection
depends on the normal direction and not on h, so the bias is **scale invariant** — which is
precisely the flat max-error curve. The PV fit is immune because a PLIC polygon exists in every
mixed cell whatever the slope, so its 5³ data set is symmetric. Note the WO's own gate design is
what caught this: *"report max and L1 separately, since the max is where the cascade's weakest
branch shows"* — the L1 barely moves (2.26 → 1.37) while the max goes 1.86 → 0.00.
Kept as `set_vof_curvature_mixed_height_fit(True)` / `VofCurvature::useMixedHeightFit`, an
instrument rather than a configuration. Han et al. independently rank the PV fallback *above* HF
once the volume fractions carry transport error — which is exactly the regime the fallback fires in.

**Two smaller findings.**
- **Tier 2a fires on 0.00–0.06 % of a sphere's interfacial cells** and is therefore near-vacuous as
  a tier. That is structural too: the direction with the largest |n_d| is by construction the one
  whose columns are most likely to close, so when it fails the other two usually fail as well. It is
  free (the same colour read along another axis) and it does help on transported fields, so it
  stays; but a cascade design that leans on it is leaning on nothing.
- **A published sign error in Han et al. eq. (14f)**, found and corrected. Their `∫y'² dA` line
  integral carries the edge factor `(x'_{v+1} − x'_v)`; evaluated on the counter-clockwise unit
  square it returns `−1/3` where `∫y² dA = +1/3`. The correct factor is `(x'_v − x'_{v+1})`, the
  natural x↔y antisymmetry of Green's theorem. All six monomials are re-derived in
  `curvature.hpp` from `∫∫ x^a y^b dA = ∮ x^{a+1}y^b/(a+1) dy` rather than transcribed, verified
  against the unit square (exact) and a quadrature oracle (5e-6), and (14a)–(14e) do check out —
  (14d) agrees term for term after `(x0+x1)(x0²+x1²) = x0³+x0²x1+x0x1²+x1³`. *This is the second
  transcription defect this campaign has found in a published listing (WO-D found one in Lehmann &
  Gekle's Listing 1); the habit of gating a hand-computable case is what catches them, because the
  randomized batteries do not.*

**Build note for whoever adds a driver class next:** nvcc rejects an extended
`__host__ __device__` lambda inside a **private or protected** member function ("The enclosing
parent function ... cannot have private or protected access within its class"). `VofCurvature`'s
four passes are public for that reason alone, as `WyAdvector`'s are. The host-openmp build compiles
it happily, so this only shows up on the CUDA leg.

### WO-M steps 2 + 3 — what float storage costs, and the precision policy that follows

All numbers below are one A/B: the **same commit** built twice against `nvidia-cuda`, the second time
with `-DPECLET_FLOW_MREAL_DOUBLE`, run on one RTX 5080. Harness: `tests/study/precision_ab.py`
(+ `mg_trace_parse.py` for the residual traces). Because the shared checkout was mid-flight with
WO-O's V3 work, every number here was produced in a **`git worktree` at HEAD + this WO's diff only**,
so none of it is contaminated by concurrent work.

#### 0. The A/B switch was incomplete — four families of hard `(float)` casts (fixed, byte-identical)

`MReal` types the operator views, but a set of assignment sites cast to `float` *literally*, so
their operator stayed fp32 even in a double build. Found by the porous case below reading
**4.768e-08 in both builds**:

| site | what it clamps |
|---|---|
| `IbmSolver::buildAdvStencil` / `buildAdvStencilVar` (`flow_ibm.hpp`) | the implicit-FOU momentum stencil, all 7 coefficients, both the constant and the varRho path |
| `IbmSolver::addDragDiagonal` | the CFD-DEM face-drag momentum diagonal |
| `IbmSolver::applyBackflowStab` | the backflow-stabilization diagonal increment (found in the same audit) |
| `mac_velocity_mg.hpp` × 5 | the velocity-MG staircase / upwind-coarse / const-aniso operators, the identity row, and the no-slip boundary fold |

All now cast to `MReal`, which is **byte-identical when `MReal` is float** (verified: the float
build reproduces every pre-change digit of the hydrostatic, porous, Z&H and contrast cases). Without
this the whole step-2 table would have been wrong in the momentum half.

#### 1. Accuracy — float vs double on cases we ship

| case | metric | float (default) | double | verdict |
|---|---|---|---|---|
| **varRho hydrostatic acid test**, ρ ratio 1e1…1e6 | steady max\|u\| | 1.8e-17 … 6.2e-17 | 1.2e-17 … 4.5e-17 | **no gain** — both at machine zero |
| " | ∂P/∂z rel err | 3.4e-16 … 9.1e-16 | 2.2e-16 … 9.1e-16 | **no gain** |
| " | pressure its (ratio 1e1→1e6) | 10,12,13,14,15,**22** | 10,11,12,13,14,**15** | −7 its at ratio 1e6 |
| **V2b uniform-velocity identity**, ratio 1e1…1e4 | max\|u−U\|/\|U\| | 1.32e-7 … 1.68e-7 | 8.9e-16 … 2.4e-15 | **8 orders** |
| " | max\|div(open·u)\| | 1.8e-14 … 9.8e-12 | **3.3e-16, flat in ratio** | 2–4 orders |
| **porous drag balance** f = β·u (exact reference) | rel err | **4.768e-08** | **2.776e-16** | **8 orders** |
| **Zick & Homsy** SC drag φ=0.125, N=32/48/64 | K (ref 4.292) | 4.277001 / 4.288006 / 4.291023 (−0.35 / −0.09 / −0.02 %) | 4.277003 / 4.288012 / 4.291033 | **no gain** — the two agree to the 6th digit; the discretization error is 4 orders larger |
| **RCP permeability** (φ=0.63) Ng=44 / 56 | k | 1.00352402e-3 / 1.00879970e-3 | 1.00344076e-3 / **1.00879970e-3** | 8e-5 relative / **identical** |
| **RCP bed, PCG rtol 1e-8 cap 300**, Ng=48/64/96 | its per step | 24 / 33 / **300 — CAPPED, INVALID** | 14 / 14 / **28** | **the solve is valid instead of invalid** |
| " | max\|div(open·u)\| | 4.5e-6 / 6.5e-7 / 4.4e-6 | 9.5e-12 / 9.5e-12 / 3.2e-11 | **5 orders** |

**The single-phase regression, run on BOTH builds** (13 grid points, `zh_sphere` /
`random_spheres` / `hollow_rings`): **+0.00 % on every metric in both**, with **identical**
`p_iter_tot`, iterations/step and step counts. The only thing that moves is the flux divergence,
which drops about two orders (e.g. `zh_sphere` N=64 4.5e-11 → 2.1e-12; `hollow_rings` N=24 5.9e-11 →
4.6e-12). That is the cleanest possible statement of the trade: on everything this suite exists to
protect, fp64 changes *nothing* — it only lowers a residual that was already four orders below the
gate.

**On the Ergun bed specifically.** The work order named `coupling/tests/test_fixed_bed_ergun_porous.py`,
which needs `peclet.dem` + `peclet.coupling`; the local builds of both predate the 2026-08-30
OpenMP-prefix switch, and composing an old-prefix and a new-prefix module in one interpreter is
explicitly forbidden (`suite/CLAUDE.md`), so rebuilding two projects was not the cheapest route to
the answer. The `porous` case above exercises the *same flow-side code path*
(`buildPorousCoeffCons` + `addDragDiagonal` + `projectCorrectPorousCons`) against an **exact**
reference instead of an empirical correlation — and it settles the Ergun question anyway:
**WO-I's post-fix Ergun agreement of "2e−8 … 7e−8 (machine precision)" is precisely this float
floor**, not double machine precision. The identical 4.768e-08 here, and its collapse to 2.776e-16
in a double build, identify that recorded number as fp32-limited. WO-N should record it as such.

The pattern is sharp and it is not "double is better everywhere". **Every case that only needs an
approximation is unchanged** (Z&H drag, the converged permeability, the hydrostatic rest state —
which is exact in float because it never divides by a stencil). **Every case that depends on a
discrete identity moves by eight orders**: the uniform-velocity identity, the exact drag balance,
and the singular pressure operator's deflation.

#### 2. Cost — measured, at identical iteration counts

RCP bed, `rtol` 1e-6 so neither build caps, 12 timed steps after 4 warm-up, three interleaved
repeats each (spread ≤ 5 %):

| grid | cells | its/step | float ms/step | double ms/step | Δt | float GPU | double GPU | ΔB/cell |
|---|---|---|---|---|---|---|---|---|
| 128³ | 2.10 M | 14 | **171.1** | **192.8** | **+12.7 %** | 2532 MiB | 2774 MiB | **+121 B** |
| 160³ | 4.10 M | 21 | **350.1** | **390.3** | **+11.5 %** | 4442 MiB | 4908 MiB | **+119 B** |

So a full fp64 hierarchy + momentum stencil costs **+12 % wall time and +120 B/cell (+10 % of total
solver memory)**. The measured +120 B/cell is a good cross-check that the instrument is measuring
what it should: the arrays that change are the pressure hierarchy's seven coefficients (7 × 4 B ×
8/7 for the geometric tail = 32 B/cell) plus the three momentum components' seven each (3 × 7 × 4 =
84), i.e. **116 B/cell predicted against 119–121 measured**. By the same accounting the
double-diagonal's one array per operator is 4 × 8/7 + 3 × 4 = **17 B/cell**. On the high-contrast beds it is nonetheless *faster to solution* (28 iterations
instead of a burnt 300).

#### 3. The attainable floor, and why a fixed rtol of 1e-8 is not a meaningful target

Driving PCG with an unreachable `rtol` = 1e-14 and letting it run to 400–500 iterations shows where
each build's residual actually stops, and whether it then walks back up (`min` vs `final`):

| Ng | float floor → final | double floor → final | double + **diagresum** floor → final |
|---|---|---|---|
| 48 | 5.5e-09 → 8.7e-04 (**×7.8e4**) | 4.0e-15 → 4.0e-15 | 4.0e-15 → 4.0e-15 |
| 64 | 4.3e-09 → 6.2e-04 (**×7.6e4**) | 5.0e-15 → 5.0e-15 | 5.0e-15 → 5.0e-15 |
| 96 | 2.4e-08 → 3.8e-03 (**×1.6e5**) | 5.9e-15 → 5.9e-15 | 9.9e-15 → 9.9e-15 |
| 128 | 4.9e-09 → 2.4e-03 (**×4.9e5**) | 5.0e-15 → 5.0e-15 | 4.8e-15 → 4.8e-15 |
| 160 | 3.8e-08 → 6.9e-04 (**×1.8e4**) | 1.7e-14 → **6.0e-08** (×3.5e6) | 6.4e-15 → 6.4e-15 (solve 0); 1.5e-14 → 5.3e-07 (solve 1) |

Three readings.
- **The float floor is ~5e-9…6e-8 and is resolution-independent**, and it is *always* followed by a
  rebound of 1e4–5e5. That is the eps_f32 signature: the residual reaches the level at which the
  stored operator no longer satisfies `A·1 = 0`, then the CG recurrence loses conjugacy against a
  right-hand side it can no longer represent.
- **It is also depth-independent.** Ng=96 at levels 3/4/5/6: float 2.49e-8 / 2.43e-8 / 2.43e-8 /
  2.39e-8; double 9.0e-15 / 9.6e-15 / 5.9e-15 / 4.4e-15. So this is a **fine-level storage** defect,
  not a coarsening one — the discriminator the coordinator asked for, applied.
- **At Ng=160 the double build starts to rebound too** (floor 1.7e-14, final 6.0e-8), which is the
  same phenomenon the collocated session records at 384³/R=24 (their "double floors at 4e-8"). One
  precision level up, same shape.

**Condition numbers**, from the dense probe (`mg_precond_analyze.py` now reports κ(A) on the
mean-free subspace and the O(eps·κ) attainable residual it implies):

| grid | contrast | κ(A) | O(eps·κ) f32 | O(eps·κ) f64 |
|---|---|---|---|---|
| 8³ periodic | 1 | 2.05e+01 | 1.2e-06 | 2.3e-15 |
| 8³ periodic | 1e2 / 1e3 / 1e4 / 1e6 | 1.15e3 / 1.14e4 / 1.14e5 / 1.14e7 | 6.8e-5 … 6.8e-1 | 1.3e-13 … 1.3e-9 |
| 16³ periodic | 1e3 / 1e4 | 4.64e4 / 4.64e5 | 2.8e-3 / 2.8e-2 | 5.2e-12 / 5.2e-11 |
| 8³ / 16³ walls ±z | 1e3 | 4.47e4 / 1.85e5 | 2.7e-3 / 1.1e-2 | 5.0e-12 / 2.1e-11 |

κ is **exactly linear in the contrast and quadratic in N** (8³→16³ at ratio 1e3: 1.14e4 → 4.64e4, a
factor 4.07). So a usable design rule for this operator:

> **κ ≈ 0.18 · N² · contrast** (periodic; ×4 for wall-bounded), and the attainable relative residual
> is **O(eps · κ)** — no preconditioner and no storage policy crosses it.

Consequences worth acting on:
- At **ratio 1000 on a 256³ grid**, κ ≈ 1.2e7, so the fp64 attainable residual is ~1e-9 *before* the
  O(1)–O(10) constant. **A VoF gate demanding `rtol` 1e-8 there is asking for something within a
  factor of a few of the arithmetic limit**, and at 512³ or ratio 1e4 it is asking for the
  impossible. V3/V4 should adopt a **resolution- and contrast-aware tolerance**, e.g.
  `rtol = max(1e-8, C · eps · 0.18 N² · Δρ/ρ)` with C ~ 1e2, rather than a fixed number — which is
  the same conclusion the collocated session reached empirically when it moved its ladder to
  `PRTOL = 2e-7` at 384³.
- The eps·κ bound is an **order-of-magnitude ceiling, not a predictor**: the measured float floors
  (5e-9) sit well below the bound's 1e-6 for the same operator. Use it to decide whether a target is
  *impossible*, not to predict where a solve will stop.

#### 4. Does the cheap fix work? — the double-diagonal ablation, against a matched full-double control

`PECLET_FLOW_MG_DIAGRESUM=1` (new, off by default, `mac_cutcell_mg.hpp`) rounds every stored face
coefficient back to float and recomputes the diagonal as the **exact double sum of those rounded
faces** — bit-for-bit the arithmetic a "float faces + double diagonal" hierarchy performs, on fp64
storage, so it isolates the numerics from the storage saving. Judged the way the coordinator
specified: **does it recover the matched full-double control**, not does it reach 1e-8.

**It does, at every configuration measured** (the third column of the floor table): identical to
full double at Ng=48 and 64, within 3 % at Ng=96 and 128, and at Ng=160 — the one grid where full
double itself becomes marginal — it converged solve 0 in 81 iterations with no rebound where full
double rebounded, and matched full double on solve 1. Iteration counts likewise: 22/24, 21/23,
43/54, 41/47, 81/… — the same numbers as full double.

**So the six off-diagonals carry only float information and it does not matter. The diagonal's
exactness is the entire story.**

#### 5. THE POLICY (step 3)

The same root cause has now surfaced **three times in unrelated code**, which is the real argument:
it is not three bugs, it is one missing rule.

| # | where | the identity that was broken | how it was cured |
|---|---|---|---|
| 1 | pressure hierarchy, every level | `A·1 = 0` per row (the operator is singular *by construction*; mean-removal deflates exactly the constant) | open — this WO |
| 2 | momentum / velocity stencil | `rowsum(A) = ρ_f/dt (+β_f)` — the *same* coefficient `buildRhsVar*` forms in double, so `u* = b/diag` must return `u` | open — this WO |
| 3 | agglomerated bottom AMG (`buildAmg`) | `A·1 = 0` on the bottom operator | **already cured, 2026-08-13, by a targeted double resum of the fluid diagonals** — and it has held since |

**The rule, stated once:**

> **A quantity that an algorithm requires to satisfy an exact discrete identity must be stored in the
> precision in which that identity is asserted. A quantity that only carries an approximation may
> stay float.**

Applied to this solver, the identity-bearing quantity is in both cases the operator **diagonal**, and
the approximation-bearing quantities are the six **face couplings**. The measurement in §4 is the
direct test of that split and it passes: perturb the faces at eps_f32 and nothing happens; perturb
the diagonal at eps_f32 and the solve dies.

**Recommendation, in priority order.**

1. **Ship the double-diagonal in both operators.** Store `AC` in double while `AW…AT` stay float, and
   *define* `AC` as the resum of what was stored: for the pressure operator `AC = −Σ(stored faces)`;
   for the momentum operator `AC = (ρ_f/dt + β_f) + Σ(−stored faces)` with the time term at the RHS's
   own precision. **Measured gain**: everything in §1's identity rows (8 orders on the uniform-velocity
   identity and the drag balance) plus §3's floor (5e-9 → 5e-15) and §1's 300-iteration cap.
   **Cost**: +4 B/cell per diagonal → **≈ +17 B/cell** (pressure hierarchy ≈ 4.6, three momentum
   components 12) against the **measured +120 B/cell** of full fp64, i.e. **14 % of the memory** and,
   since the diagonal is one of seven coefficient arrays, ≈ **+1.7 % of the measured +12 % time**.
   This is a generalisation of a fix already proven in-tree (row 3 above), not a new idea.
   *Implementation cost is the honest objection*: `AC` needs a view type distinct from `AW…AT`
   through the smoother, residual, matvec, CA ring and AMG assembly. That is a follow-on work order,
   not a line edit — which is exactly why it was worth proving the numerics first with `DIAGRESUM`.
2. **Until then, `-DPECLET_FLOW_MREAL_DOUBLE` is a validated escape hatch**, not a science project:
   it is the only way to get a *valid* solve on a high-contrast bed above ~Ng 96 today, and it costs
   a measured +12 % time / +10 % memory. Document it for high-contrast porous and high-ratio VoF work.
3. **Do NOT make fp64 the default.** It buys nothing on the accuracy metrics anyone quotes (Z&H drag,
   permeability, the hydrostatic acid test — §1), and +12 % time / +10 % memory on every run to fix a
   problem that a +1.7 % / +1.4 % change fixes as completely (§4).
4. **Adopt a resolution- and contrast-aware pressure tolerance** in place of the fixed 1e-8 (§3). A
   fixed 1e-8 is already within a small factor of the fp64 arithmetic limit at ratio 1000 on 256³.
5. **Next thing to measure (not this WO):** the *geometry* is still float and `MReal` does not reach
   it — `IbmOverlay`'s `K/M/X/Nbc` and `D_rescale`, and `mac_ibm.hpp`'s `(float)ccSampleExt` θ
   sampling. That is a third storage locus and the leading suspect for the residual floor the
   collocated session sees in a *double* build at 384³ (§3, Ng=160 here).

**And what the policy explicitly does NOT buy** — so that nobody expects it to:
- **S3 stands.** The V-cycle preconditioner goes indefinite at contrast ≳1e3 wall-bounded / 1e4
  periodic in *both* precisions (step 1). No storage policy touches it; Chebyshev remains the
  varRho/porous default for that reason.
- **The attainable floor O(eps·κ) stands.** A perfect `A·1 = 0` repair still cannot cross it.

#### 6. What this WO actually shipped (and what it deliberately did not)

**Shipped** (all inert in the default build):
- the three families of missing `MReal` casts (§0) — plus a fourth found while auditing, the
  backflow-stabilization diagonal `AC(i) += (float)(beta*rho*back)`;
- `CutcellMG::resumDiagonal` + `PECLET_FLOW_MG_DIAGRESUM`, the double-diagonal ablation;
- `tests/study/mg_precond/` (dense `M` and `A`, LDL + mean-free spectrum + κ(A) + attainable floor);
- `tests/study/precision_ab.py` + `mg_trace_parse.py`, the whole step-2 battery, reproducible in
  one command per build.

**Not shipped, deliberately: the production double-diagonal itself.** Its numerics are proven (§4)
but the change is type surgery, not a line edit — `AC` must become a view type distinct from
`AW…AT` through `buildCutcellOp`, `cutcellSmoothColor`, `residualCutcell(Box)`, the matvec,
`removeMean`, `maskSolid`, `estimateEigenvalues`, `buildAmg`, the CA operator ring and its
`GridHalo<MReal>` twin, and the whole momentum half in `cut_cell_ibm.hpp` / `flow_ibm.hpp` /
`mac_velocity_mg.hpp`. That is a work order of its own, with the full ctest + MPI + regression
battery behind it, and WO-M's charter was to decide the policy from measurement — which it now has,
with the decisive experiment already in the tree so the implementer can re-run it after every step.

#### 7. Process note — three of this WO's hunks were swept into someone else's commit

The `(float)` → `(MReal)` hunks in `src/flow_ibm.hpp` were staged and committed by the concurrent
WO-O session in `9047e1b` ("VoF V3 (WO-O): height-function curvature…"), which staged
`src/flow_ibm.hpp` by name while this WO's edit to that file was in the working tree. The code is
correct and complete on main (verified line by line against this WO's own worktree copy), and history
was **not** rewritten — main is shared with two other live sessions. It happened a second time a few hours later: the memory-accounting paragraph of §2 was
carried into `cb7fcbb` ("WO-O findings: record the CUDA kokkos_mpi 57/57 leg"), which staged this
same doc by name. Recording both here because the commit messages attribute the changes to curvature
work, and because named-path staging in a shared checkout has now crossed sessions four times in two
days. The mitigation this WO used for
everything downstream of it — a `git worktree` at HEAD carrying only its own diff — is the one that
actually works, and is worth making the default for any measurement campaign run beside another agent.

#### 8. Gates

| gate | result |
|---|---|
| **Default build byte-identical** — single-phase regression | **PASS, +0.00 %** on every metric and **identical** `p_iter_tot`, iterations/step, step count and divergence on all 13 grid points of `zh_sphere` / `random_spheres` / `hollow_rings`. Structurally guaranteed too: `(float)x` → `(MReal)x` is the same cast when `MReal` is float, and `resumDiagonal` returns on `!mgDiagResum()` (env-gated, default off) *and* refuses to run at all when `sizeof(MReal) == sizeof(float)` |
| `tests/kokkos` ctests | **PASS 24/24 on nvidia-cuda AND host-openmp** in the isolated worktree, and **25/25 on nvidia-cuda on the merged main tree** (WO-O's V3 `vof_curvature` included) |
| `tests/kokkos_mpi` ctests | **PASS 54/54 on nvidia-cuda AND 54/54 on host-openmp**, np 1/2/4, zero failures. Both suites had to be run in two segments because the first pass crawled under a 3-way GPU/CPU contention with the concurrent WO-O battery (`graphamg_mpi_np4` 543 s against its normal ~2 s; `velocitymg_mpi_np4` on host stalled >45 min inside ctest yet passed standalone with `max\|distributed − single-rank\| = 0.000e+00`). That is the load-triggered slowness WO-L documents, not a defect: once the competing battery finished, the remaining 23 CUDA legs took 1889 s and the remaining 40 host legs 1511 s, all green |
| The float A/B is unchanged by the cast completion | **PASS** — every float number in §1 (hydrostatic, porous, Z&H, contrast) reproduces its pre-change digits exactly |
| Measurements isolated from concurrent work | **PASS** — every number produced in a `git worktree` at HEAD + this WO's diff only |

---

### WO-P (rung V4) — balanced-force CSF + the capillary time step — **DONE 2026-08-31**, one design deviation, one V3 defect found, one inherited gate corrected

Delivered: `src/vof/surface_tension.hpp` (container-free `KOKKOS_INLINE_FUNCTION`s only, WO-D
signature rule — the face-curvature rule, the face force and the Brackbill limit) +
`Solver::addCsfRhs` / `addCsfRhsCellInterp` / `updateVofCurvature` / `capillaryDt` /
`vofStepLimits` / `csfDiagnostics` in `src/flow_ibm.hpp` + `VofCurvature::interfaceEps` in
`src/vof/curvature_field.hpp` + ten Python entry points +
`tests/kokkos/test_vof_surface_tension.cpp` (ctest `vof_surface_tension`, 7 gates) +
`tests/kokkos_mpi/test_vof_surface_tension_mpi.cpp` (`vof_surface_tension_mpi_np{1,2,4}`) +
`tests/study/vof_surface_tension.py` (the physics battery: `static`, `wave`, `lamb`, `hysing1/2`,
`falling`, `limits`).

**The force, and exactly which operator forms the face value.** At the staggered velocity unknown
`u_c(i)`,

```
F_c(i) = sigma * kappa_f(i) * ( C(i) - C(i - s_c) ) / h                                       (1)
```

added to the momentum RHS at the same point, in the same units and with the same cut-cell rescale
`rs(i)` as the incremental scheme's `-(P(i) - P(i - s_c))`. **The difference `C(i) - C(i - s_c)` is
the projection's own face difference** — the identical operator `buildRhsVar` applies to `P` and
`projectCorrectVar` applies to `phi`. `kappa_f` is the arithmetic mean of the two cells' curvatures
where both carry one, the single available one where only one does, and 0 where neither does (an
*orphan* face, counted by `csf_diagnostics()`, never silently dropped — Basilisk's `tension.h` makes
the same three-way choice and marks the last branch "this should not happen").

**DEVIATION — the force does NOT go through the per-cell force-field machinery, and that is the
point of the rung.** The WO says to use `cellForce_` (whose ghosts WO-G repaired) "with the face
value formed by the projection's own operator". The two are not simultaneously satisfiable:
`cellForce_`'s face rule is the **arithmetic interpolation** `1/2 (f(i) + f(i-s_c))` of a
cell-centred force, which is exactly right for `rho*g` (the pair `f_f/rho_f` is then the intended
acceleration — `variable_density_projection.md` §1) and exactly wrong for `sigma*kappa*grad C`. So
`addCsfRhs` is an additive sibling kernel at the same insertion point, and the interpolated variant
ships as the **ablation** `set_csf_mode(1)` rather than as the mechanism. The reason, and the
measurement:

> With a constant `kappa`, (1) is *exactly* the discrete gradient of `sigma*kappa*C`. It therefore
> lies in the range of the operator the projection inverts, the projection annihilates it
> completely, and the discrete equilibrium is exact in floating point. An interpolated cell-centred
> `sigma*kappa*grad C` is not in that range for any potential, the projection cannot remove it, and
> what is left drives the classical spurious current.

| stationary droplet, 32^3, constant kappa, 30 steps | max\|u\| | spurious Ca |
|---|---|---|
| **balanced force, (1)** — the shipped path | **1.93e-17** | **1.9e-18** |
| `set_csf_mode(1)` — cell-centred force, face-interpolated | **5.76e-02** | **5.8e-03** |

**3.0e+15 x.** That is the literature's "balanced force gives machine zero, naive CSF gives ~1e-2"
(Francois et al., JCP 213:141 (2006); Popinet, JCP 228:5838 (2009)) reproduced as a switch on one
line of one kernel, with the curvature, the grid, the time step and the pressure driver held
identical. It is the momentum analogue of the harmonic-rho_f ablation that makes WO-J's hydrostatic
gate mean something.

**Gate results** (host-openmp and nvidia-cuda; `tests/kokkos/test_vof_surface_tension.cpp`).

| gate | result |
|---|---|
| **P1 THE EXACTNESS GATE** — stationary droplet, constant kappa, uniform rho | max\|u\| **3.64e-17 / 1.93e-17 / 2.43e-17** at 16^3/32^3/48^3 (R = 4/8/12) and **9.38e-17 / 7.47e-17 / 1.93e-17 / 4.28e-18** at mu = 1e-3 / 1e-2 / 1e-1 / 1 — machine zero, independent of resolution and of viscosity, as Popinet (2009) requires |
| P2 the operator ablation | the table above |
| P3 Young-Laplace | dP(liquid-gas) = sigma*kappa to **2.2e-16** relative; max\|P - p0 - sigma*kappa*C\| over the WHOLE field **2.6e-15** — the equilibrium pressure field IS the discrete solution `sigma*kappa*C + const` |
| P4 the capillary dt | `capillary_dt()` reproduces `sqrt((rho1+rho2) h^3/(4 pi sigma))` to **0.0e+00** both from the declared phase pair (`enable_vof_momentum`) and from `min(rho)+max(rho)` of a closure-driven field; `step()` REFUSES `dt = 1.01 dt_sigma`; `set_capillary_cfl(1e30)` is the escape hatch |
| P5 inert when off | du = dp = dC = **0.0** bitwise against the V2a/V2b path, with `set_surface_tension(0)`, `set_vof_interface_eps` and `set_csf_mode` all touched |
| P6 the wisp guard | the mechanism below |
| P7 spurious currents, real curvature | Ca **2.54e-4 / 5.90e-5 / 2.65e-5 / 1.39e-5** at D/dx = 8 / 16 / 24 / 32 (60 steps), fitted order **2.10**; dkappa_rms 4.57e-3 -> 1.22e-4; orphan faces 0-1 per component at every rung |
| MPI np 1/2/4, host + CUDA, periodic and walls-on-the-cut-axis | np = 1 **bitwise** on u, P, C and kappa; np = 2/4 du <= 1.5e-15, dP <= 2.2e-15, dC <= 6.7e-15, dkappa <= 8.8e-15 (the pressure driver's documented reduction-order floor, propagated into the colour and hence into kappa); the curvature **BRANCH** field bitwise at every np, and the exactness gate at machine zero on every decomposition |

**THE V3 DEFECT THIS RUNG FOUND: the curvature cascade returns `|kappa|` up to 2.9e+11 for cells
that carry no interface, and under a force that is fatal.**

Weymouth-Yue leaves **round-off colour residue** in every cell its sweeps touch — measured on a
static droplet at 64^3 after 10 steps: `C` down to **-3.2e-35**, and some **5300 extra cells** at
`0 < C < 1e-30`, i.e. more "interfacial" cells than the interface itself has (10018 against 4719).
Those cells satisfy `wyIsMixed` (`C > 0 && C < 1`), so the cascade dutifully builds a PLIC polygon of
area ~0 for them and fits a paraboloid to it. Measured `|kappa|` in that population: **2.2e+02 ...
2.9e+11**, where the physical value is 0.125.

On its own that is a wrong number in a field nobody was reading — V3's own gates run on exact colour
fields, where the population is empty, which is why WO-O never saw it. Under V4 it is fatal: a face
between such a cell and a REAL interfacial cell has `dC = O(1)` and a face curvature
`(kappa_real + 1e11)/2`. Measured, 32^3 static droplet, unguarded: max\|u\| **4.5e-4 at step 1 ->
2.7e-1 by step 20**; at 96^3 the run trips the Weymouth-Yue CFL cap outright; the SAME run with the
curvature frozen at its clean initial value stays bounded at 4e-3 — which is what isolates the
curvature feedback as the mechanism rather than the transport or the force.

The fix is a wisp threshold on the interfacial predicate — `VofCurvature::interfaceEps`: a cell
carries an interface only while `eps < C < 1 - eps`. **Default 0, which is `wyIsMixed` verbatim, so
rung V3 is byte-identical** (the `vof_curvature` ctest reproduces WO-O's numbers digit for digit:
order 2.26 / 1.86, PV-only 1.96 / 1.99, tier-2b 1.37 / 0.00). `set_surface_tension` sets it to
**1e-8**, the same threshold `WyAdvector::diagnostics` already reports wisps against.
`set_vof_interface_eps(0)` is the ablation; measured at 64^3 over 60 steps:

| interface_eps | cells served | max\|kappa\| (2/R = 0.125) | max\|u\| step 1 -> step 60 |
|---|---|---|---|
| **1e-8 (default)** | 4828 | **0.1258** | 2.11e-3 -> **1.39e-4** (decaying) |
| 0 (the V3 predicate) | 10018 | **2.87e+11** | 2.11e-3 -> **1.85e-3** (not decaying) |

This is VOF_PLAN §6's trap — *"with surface tension clipping is unavoidable (Arrufat)"* — arriving
exactly where the plan said it would, and in the cheapest available form: a threshold on the
*predicate* rather than clipping the colour field, so exact conservation is untouched. The wisps
themselves are left alone; they are at 1e-30 and neither the density closure nor the force can see
them.

**WHY THE EXACTNESS GATE IS EXACT AT UNIFORM DENSITY AND ASYMPTOTIC AT VARIABLE DENSITY — a
mechanism, measured, and NOT the float-storage floor.**

At ratio 1 the gate is at 1e-17 from the first step. At ratio 10-1000 it starts at ~2e-4 and
*decays*: **1.12e-4 / 5.90e-5 / 2.04e-5 / 4.07e-6 / 4.97e-7** at steps 1 / 10 / 30 / 100 / 300
(ratio 10). Two ablations pin the mechanism:

| ratio 100, step 1 | mu = 0 | 1e-4 | 1e-2 | 0.1 | 1 |
|---|---|---|---|---|---|
| max\|u\| | **3.30e-09** | 4.82e-07 | 4.35e-05 | 2.38e-04 | 7.17e-04 |

| ratio 100, mu = 0.1, step 1 | dt/dt_sigma = 1e-4 | 1e-2 | 0.1 | 0.5 |
|---|---|---|---|---|
| max\|u\| | **1.95e-11** | 1.89e-07 | 1.60e-05 | 2.38e-04 |

i.e. the residue scales as **mu . dt^2** and vanishes at either limit. The reason is the one that
makes the gate exact at uniform density: the semi-implicit momentum operator
`A = rho_f/dt - mu*Lap` **commutes with the discrete gradient only when `rho_f` is constant**, so
`A^{-1} grad(Phi) = grad(A^{-1} Phi)` — a pure discrete gradient, which the projection annihilates
exactly — only there. With `rho_f` varying, `u*` picks up a non-gradient part of order
`mu dt^2 grad(rho_f)`; the fixed point (`u = 0`, `grad P = sigma kappa grad C`) still exists and is
still exact, but it is *approached* instead of hit. The same statement applies at a **wall** at
constant density (the face-folded viscous operator does not leave a gradient field invariant next to
a Dirichlet boundary): measured on the MPI `exact-walls-z` configuration, 2.11e-10 at 10 steps ->
1.18e-10 at 30 -> 2.91e-11 at 100 -> 3.59e-12 at 300, and **1.10e-16 at mu = 0**. That is why the
MPI gate asserts machine zero on the periodic configuration and a bound plus decomposition-
independence on the walled one.

**It is NOT the float operator storage.** The same battery in a `-DPECLET_FLOW_MREAL_DOUBLE` build
reproduces every number above to five significant figures (2.0374e-05 against 2.0374e-05 at ratio 10,
mu = 0.1) even though `max|div(open u)|` improves from 9.4e-14 to 7.6e-21. What the double build
*does* fix is the residue **at mu = 0**, where the mu-mechanism is absent:

| mu = 0, 30 steps | ratio 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| float `MReal` (shipped) | 7.51e-17 | 8.60e-10 | 4.01e-10 | 1.35e-10 |
| double `MReal` | 6.36e-17 | **6.84e-17** | **2.29e-17** | **8.86e-18** |

So the balanced-force identity is exact at **every** density ratio to 1000 in a double build, and the
float default carries WO-M's `A.1 = 0` floor at ~1e-10 — two orders below the mu-mechanism at any
viscosity anyone would run, so it does not bind in practice. Both loci are separately identified,
which is the point of measuring them apart.

**THE CAPILLARY TIME STEP, AND IT BINDS EVERYWHERE AT PORE SCALE.**

`capillary_dt()` = `sqrt((rho1+rho2) h^3/(4 pi sigma))` (Brackbill, Kothe & Zemach, JCP 100:335
(1992), eq. 44; Denner & van Wachem, JCP 285:24 (2015) verified all three of its features — the
`1/(4 pi)` prefactor IS the stability boundary rather than a conservative estimate, the scaling is
`h^{3/2}`, and it is the **sum** of the densities because both phases oscillate). It is enforced by
`step()` beside the Weymouth-Yue CFL cap, with `set_capillary_cfl(f)` (default 1.0) the deliberate
escape hatch, and `vof_step_limits()` reports both limits and which binds.

Swept over pore diameters 50 / 200 um, 16 / 32 / 64 cells per diameter and Ca = 1e-6 / 1e-4 / 1e-2
for water/air (sigma 0.072 N/m, rho 1000/1.2, mu 1e-3): **the capillary limit binds in 18 of 18
combinations**, by factors 6 (200 um, 16 cells, Ca 1e-2) to 5.9e+04 (50 um, 16 cells, Ca 1e-6) — and
it becomes **more** binding under refinement, since `dt_sigma ~ h^{3/2}` against `dt_CFL ~ h`, so
their ratio grows as `h^{-1/2}`. The step-count economics that follows, per pore volume traversed:

| d [um] | cells/d | Ca | dt_sigma [s] | dt_CFL [s] | steps for one pore |
|---|---|---|---|---|---|
| 50 | 16 | 1e-6 | 1.84e-7 | 1.09e-2 | **3.8e+06** |
| 50 | 64 | 1e-6 | 2.30e-8 | 2.71e-3 | **3.0e+07** |
| 50 | 16 | 1e-2 | 1.84e-7 | 1.09e-6 | 3.8e+02 |
| 200 | 64 | 1e-4 | 1.84e-7 | 1.09e-4 | 1.5e+05 |

**So the answer to the WO's question is yes: at pore scale the capillary dt is the binding limit, by
four to five orders at the capillary numbers of interest, and the cost is 1e6-1e7 steps per pore
volume.** That is the number an implicit or semi-implicit surface-tension treatment would have to
beat. Popinet (2018) says not yet, and nothing here contradicts him for *dynamic* problems (where
`dt_CFL` shrinks alongside), but for a slow imbibition/drainage sweep at Ca ~ 1e-6 the explicit
constraint is essentially the entire cost of the calculation.

**THE PHYSICS BATTERY** (`tests/study/vof_surface_tension.py`, nvidia-cuda). Every run records the
max pressure-iteration count against its cap and `max|div(open u)|`; none of the runs below capped.

**Hysing rising bubble** (Hysing et al., IJNMF 60:1259 (2009)), quasi-2D on 64x128x4, adaptive
`dt = 0.4 min(dt_CFL, dt_sigma)`, t = 0..3, both cases, against the published reference:

| case | quantity | measured | reference | deviation |
|---|---|---|---|---|
| **1** (Re 35, Eo 10, ratio 10) | max rise velocity | **0.2497** at t = 0.886 | 0.2417 at t = 0.921 | **+3.3 %** |
| 1 | y_c(3) | **1.0808** | 1.0810 | **-0.02 %** |
| **2** (Eo 125, ratio 1000, mu ratio 100) | max rise velocity | **0.2574** at t = 0.671 | 0.2502 at t = 0.732 | **+2.9 %** |
| 2 | y_c(3) | **1.1082** | 1.1376 | **-2.6 %** |

Case 1: 2032 steps, and the binding limit was the CAPILLARY dt on **204 of 204** dt re-picks.
Case 2: 1123 steps, and the binding limit was the **WY CFL on 108 of 113** — at ratio 1000 with
sigma 12.5x smaller, `dt_sigma` is 3.4x larger while the velocities are the same, so the transport
limit takes over. That is the clean statement of when each binds. Case 2's pressure solve is the
weak point of the pair: 116/600 iterations and `max|div(open u)| = 1.85e-03` (1e-4 relative to
|u| ~ 17 cells/s) against 20/600 and 9.1e-06 for case 1 — the density-ratio-1000 operator
conditioning WO-M quantified (kappa ~ 0.18 N^2 x contrast), and the reason Hysing themselves report
case 2 as not grid-converged across the three reference codes (y_c(3) = 1.1249 / 1.1376 / 1.1512).

*Lateral BC*: the benchmark prescribes free-slip side walls, which this solver does not have. The
runs use PERIODIC, and for case 1 that is **not an approximation** — mirroring a laterally symmetric
bubble about x = 0 and x = 1 puts images at spacing 1, and the mirror of a symmetric bubble is its
translate. Case 2 develops skirts that break the symmetry, so there it is an approximation.
Circularity is not reported: it needs the PLIC interface length, which the solver does not expose.

**MOMENTUM CONSISTENCY (V2b) IS WORTH 14 % ON HYSING CASE 1, AT DENSITY RATIO 10.** The case-1 run
above has `enable_vof_momentum` on. With it OFF, everything else identical: max rise velocity
**0.2827 (+16.9 %)** and y_c(3) **1.2083 (+11.8 %)**, against +3.3 % / -0.02 % with it on. WO-K
finding 1 recorded that its uniform-velocity gate could not discriminate the consistent scheme from
the inconsistent one on this solver (the momentum advection is in advective form, so a uniform
velocity is a fixed point either way) and concluded that "the contrast this rung needs must come
from a case where the momentum actually transports". This is that case, and the contrast is a
factor 5 in the error against a published reference — **at ratio 10**, an order of magnitude below
where the literature says consistency starts to matter.

**Falling drop — the gate WO-K deferred. Both halves of WO-K's account turn out to be wrong, and the
gate as written measured the wrong quantity.**

WO-K's substitute reached 9 % of Hadamard-Rybczynski and named an under-resolved momentum solve as
the suspected cause. Re-run here with surface tension holding the drop together (ratio 800,
mu ratio 100, sigma 16, D/h = 15, 6 Stokes relaxation times, `dt = 0.5 dt_sigma`):

1. **A periodic box driven by a zero-mean body force conserves total MOMENTUM, not total volume
   flux.** The box's mean velocity is a free mode of the projection, and at ratio 800 with
   phi = 1.6 % the light ambient recoils at ~19x the drop's speed
   (`rho_g (1-phi) U_a = -rho_l phi U_d`). Measured: `U_drop = -1.64e-03` while
   `U_ambient = +3.06e-02`. **The lab-frame drop velocity is a near-cancellation of two much larger
   numbers and is not the settling velocity**; what Hadamard-Rybczynski predicts is
   `U_drop - U_ambient = -3.23e-02`.
2. **Converging the momentum solve changes nothing physical.** Driving the smoother to a tolerance
   instead of a fixed count moves the *lab-frame* number a factor 5.7 — and leaves the *relative*
   velocity alone to four digits:

| momentum smoother | max sweeps/step | U_drop (lab) | U_rel | U_rel / (U_HR/K) |
|---|---|---|---|---|
| fixed 20 (WO-K's setting) | 60 | -8.90e-03 | -3.2202e-02 | **0.826** |
| tolerance 1e-4, cap 400 | 857 | -1.64e-03 | -3.2284e-02 | **0.828** |
| tolerance 1e-6, cap 2000 | 2649 | -1.57e-03 | -3.2287e-02 | **0.828** |

   So WO-K's suspected mechanism is refuted outright: the sweep count is irrelevant to the physics
   and only moves the cancellation.
3. **Two formula corrections were needed to make the comparison mean anything**, and both matter at
   the 20-50 % level: Hadamard-Rybczynski is `U = (2/3)(d rho) g R^2/mu_o (mu_o + mu_i)/(2 mu_o +
   3 mu_i)` — the denominator carries `3 mu_i`, and the rigid limit must return Stokes'
   `2 (d rho) g R^2/(9 mu_o)` — and the periodic-array correction is Hasimoto's (1959)
   **denominator** form `K = 1/(1 - 1.7601 c^{1/3} + c - 1.5593 c^2)`, which is 1.748 at c = 0.016
   against 1.443 for the linearised `1 + 1.7601 c^{1/3}`.
4. **The result, with the momentum solve converged:**

| D/h | phi | K | U_rel / (U_HR/K) |
|---|---|---|---|
| 10 | 0.0047 | 1.409 | **0.786** |
| 15 | 0.0160 | 1.748 | **0.828** |
| 20 | 0.0160 | 1.748 | **0.869** |

   monotone in resolution and converging toward 1. **At 15 cells per diameter the drop reaches
   83 % of the analytic terminal velocity — 17 % low, i.e. just OUTSIDE Arrufat's "within 15 %" —
   and at 20 cells per diameter it is 13 % low, inside it.** That is the honest statement; the claim
   is *not* reproduced at the resolution it is made for, and the residual is a resolution effect
   rather than a solver-convergence one. What remains unattributed is whether the last ~15 % is the
   one-cell-diffuse viscosity jump standing in for a mu-ratio-100 drop, or the array correction's
   two-term expansion at L/D ~ 3-5; the gate does not separate them and this WO does not claim it.

**Capillary wave against the analytical dispersion** (`omega^2 = sigma k^3/(rho_1+rho_2)`, the
inviscid limit of Prosperetti 1981), quasi-2D standing wave, `a_0 = lambda/100`, matched kinematic
viscosities, walls +-z, 2.5 periods:

| cells / lambda | nu | omega measured | omega theory | err | decay measured | 2 nu k^2 | err |
|---|---|---|---|---|---|---|---|
| 32 | 0.005 | 0.06017 | 0.06152 | **-2.20 %** | 1.087e-3 | 3.855e-4 | +182 % |
| 64 | 0.005 | 0.02130 | 0.02175 | **-2.06 %** | 4.628e-4 | 9.638e-5 | +380 % |
| 32 | 0.020 | 0.05928 | 0.06152 | **-3.65 %** | 2.746e-3 | 1.542e-3 | +78 % |

The frequency is 2-4 % low and the deficit grows with viscosity, which is the right sign (the
viscous correction lowers omega) but far larger than the weak-damping estimate `(gamma/omega)^2/2 ~
1e-4` allows. The decay rate is 2-5x the leading-order `2 nu k^2`; note that number is a two-extremum
fit over 2.5 periods on an initial-value problem whose early behaviour is not a pure exponential
(that is precisely why Prosperetti's full solution exists), so it is an upper bound on the true rate
rather than a measurement of it. **Reported, not gated**: the honest comparison for the decay needs
Prosperetti's Laplace-transform solution, which this WO did not implement.

**Oscillating droplet against Lamb** (1932, art. 275; mode n = 2,
`omega^2 = n(n-1)(n+1)(n+2) sigma / (R^3((n+1) rho_in + n rho_out))`), prolate perturbation of 5 % of
R released from rest, frequency from the zero crossings of the `<2z^2-x^2-y^2>` moment:

| grid | R | phi (drop/box) | rho ratio | omega measured | omega Lamb | err |
|---|---|---|---|---|---|---|
| 32^3 | 8 | 6.54 % | 1 | 0.09010 | 0.09682 | **-6.95 %** |
| 48^3 | 12 | 6.54 % | 1 | 0.04925 | 0.05270 | **-6.56 %** |
| 48^3 | 8 | 1.94 % | 1 | 0.09063 | 0.09682 | **-6.39 %** |
| 64^3 | 8 | 0.82 % | 1 | 0.09069 | 0.09682 | **-6.33 %** |
| 48^3 | 12 | 6.54 % | 100 | 0.00657 | 0.00678 | **-3.14 %** |

**A 6.5 % frequency deficit that is neither confinement nor resolution**, and both were ruled out by
measurement rather than assumed: an eightfold reduction in the drop-to-box volume ratio (6.54 % ->
0.82 %, i.e. the periodic images' added mass down by ~8x) moves it from -6.95 % to -6.33 %, and R = 8
vs R = 12 at fixed phi moves it from -6.95 % to -6.56 %. The ratio-100 row is consistent with a
*small* confinement contribution (a 100x lighter outer fluid has much less added mass to confine, and
the deficit halves), but it cannot be the bulk of it. The candidates this WO did not separate are the
finite perturbation amplitude (Tsamopoulos & Brown's nonlinear frequency shift is negative, but at
epsilon = 0.05 it is O(0.3 %)) and a systematic bias in the discrete curvature of a mode-2 shape.
**Recorded as a measured deviation, not a pass** — the same sign and the same order as the capillary
wave's, which suggests one mechanism rather than two.

Four further ablations on the Lamb case, all at 32^3 / R = 8, each of which could have explained it
and none of which does:

| what was varied | values | omega error |
|---|---|---|
| perturbation amplitude `epsilon` | 0.10 / 0.05 / 0.02 / 0.01 | -7.11 / -6.95 / -6.52 / **-6.25 %** (extrapolates to ~-6.1 % at epsilon = 0) |
| time step `dt/dt_sigma` | 0.5 / 0.125 | -6.95 / **-6.92 %** (no temporal component) |
| initialisation sub-sampling | 16 / 48 per axis | -6.95 / **-6.96 %** (not the initial fractions) |
| PV Wendland support width | 1.2 / 1.8 / 2.5 / 3.5 / 5.0 | **diverged** / n/a (-12.97 % at R = 12) / -6.95 / -6.58 / **-5.36 %** |

The width sweep is the interesting one and it points the *opposite* way to the obvious guess: a
**wider**, smoother PV kernel gives a **better** frequency (-6.95 % -> -5.36 % going 2.5 -> 5.0
cells) and a narrower one is worse and eventually unstable (width 1.2 trips the WY CFL cap outright).
So curvature *noise* costs about 1.5 % of the frequency, but a very smooth curvature still leaves
-5.4 %, and the smoothing-damps-the-restoring-force hypothesis is refuted rather than supported.
What remains is an open, measured item for whoever takes the next curvature rung; it is NOT a
balanced-force defect (the exactness gate is at 1e-17 on the same machinery), and the residual has
the same sign and order in both oscillation benchmarks.

**Full gate battery for this WO.**

| battery | result |
|---|---|
| `tests/kokkos` (26 ctests) | **26/26 on host-openmp AND nvidia-cuda**; `vof_surface_tension` is the new one (93 s host, 34 s CUDA) |
| `tests/kokkos_mpi` (60 ctests) | **60/60 on host-openmp**; the three new `vof_surface_tension_mpi_np{1,2,4}` also run green standalone on nvidia-cuda, where **np = 1 AND np = 2 are bitwise** on every field |
| V3 regression (`vof_curvature`) | reproduces WO-O's record **digit for digit** — order 2.26 / 1.86, PV-only 1.96 / 1.99, tier-2b ON 1.37 / 0.00 — i.e. `interfaceEps` defaulting to 0 is byte-identical, as designed |
| single-phase regression | **+0.00 % on all 13 grid points**, identical `p_iter_tot`, iterations/step, step counts and flux divergence on `zh_sphere` / `random_spheres` / `hollow_rings`; order p and the Richardson extrapolate unchanged to every printed digit. Run in an **isolated `git worktree` at this WO's commit only** (the checkout is shared with two other sessions), so the +0.00 % is attributable |
| the shipped build with surface tension OFF | bitwise inert (ctest P5), and structurally so: `csfActive()` gates the RHS kernel, `updateVofCurvature()` returns immediately, and `VofCurvature::interfaceEps` defaults to the V3 predicate |

**What a follow-on rung should pick up, in priority order.**
1. **The 5-7 % frequency deficit shared by the two oscillation benchmarks.** Not a balanced-force
   defect (the exactness gate is at 1e-17 on the same code) and not amplitude, dt, initialisation,
   confinement or resolution — all measured. It is the one number in this rung that is unexplained.
2. **A curvature that is defined where the force needs it.** The wisp guard fixes the fatal case, but
   the orphan-face count is not identically zero (1-2 faces per component at 48^3-64^3), and a
   face whose colour jumps with no curvature on either side silently loses its force. Extending
   kappa one cell into the non-interfacial neighbours would close it.
3. **Case 2 of Hysing shows the ratio-1000 pressure operator at its limit** — 116/600 iterations and
   `max|div(open u)| = 1.85e-03` against 20/600 and 9.1e-06 for case 1. That is WO-M's `kappa ~ 0.18
   N^2 x contrast` conditioning, and it is the first VoF case in this campaign where it is the
   dominant error rather than a footnote.

