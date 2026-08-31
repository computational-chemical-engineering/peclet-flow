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
was **not** rewritten — main is shared with two other live sessions. Recording it here because the
commit message attributes the change to curvature work, and because it is the third time in two days
that named-path staging in a shared checkout has crossed sessions. The mitigation this WO used for
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
