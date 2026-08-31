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
