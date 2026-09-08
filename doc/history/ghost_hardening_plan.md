# Ghost-projection hardening: full Robust-Scaling + symmetrized deferred correction

*Plan for a dedicated session: analysis first, then two implementation phases. Written
2026-08-17 from the porous-scaling campaign findings (peclet-examples
`benchmarks/porous-scaling`, suite `docs/COMMUNICATION_SCALING.md`).*

## Why (strategic)

The ghost-cell IBM is the AMR-bearing formulation: its per-face closure cascade is already
lifted into `peclet::core::scheme::ghost_closure` and shared verbatim with core's octree AMR
band — pointwise stencil closures generalize to coarse–fine level boundaries where cut-cell
openness bookkeeping does not. So ghost must become production-grade, not stay the
verification method. Two things stand in the way, both measured on Snellius:

1. **Fragility**: the steady Stokes march diverges (k → ±1e57) on the np=16/32 upscale beds
   (7.8k/15.6k spheres) while every smaller/refined case converges. Lead hypothesis:
   extreme-value statistics of thin gaps — the worst near-tangent sphere pair in a big random
   bed produces an ill-conditioned closure row; cut-cell survives slivers by construction
   (Robust-Scaled, `D_rescale`), ghost's current per-row `rho = min(1, min_f D_f)` scaling is
   only a partial transplant. Secondary hypothesis: per-solve compatibility bias of the
   nonsymmetric operator (left null vector ≠ constant) accumulating through the
   incremental-rotational pressure.
2. **Cost**: 4–7× cut-cell's pressure iterations (56–160/step vs 12–33). Structural: BiCGStab
   on the nonsymmetric 13-point gp matrix, preconditioned by an MG hierarchy built on the
   *binary-openness surrogate* — the gp rows never reach the hierarchy, so the preconditioner
   approximates a different operator and the mismatch is paid every iteration.

## What exists today (read these before coding)

- `flow/src/ghost_projection.hpp` — the overlay: face-state cascade (COUPLED / sandwich /
  QUAD / LIN / sliver-EXTENDED-θ / BC_ONLY / EXPLICIT), per-row conditioning rescale
  `rho = min(1, min_f D_f)`, and **two weight sets**: `wm_*` (`matrix_order`) for the implicit
  φ couplings (`gpApplyDelta`) and `w_*` (`rhs_order`) for the divergence (`gpDivergDelta`).
  `matrix_order=1, rhs_order=2` is a documented "mixed/deferred-correction" mode: 7-point,
  near-symmetric matrix with the 2nd-order steady constraint on the RHS — the seam phase C
  builds on.
- `flow/src/cut_cell_ibm.hpp` — the full Robust-Scaled reference: per-axis `D_vals`,
  `D_sandwich`, `D_rescale`, `R = D_rescale / D_axis` renormalization threaded through K/M/X/
  Nbc — this is what "fully implemented" means; phase A itemizes what the ghost path lacks.
- `flow/src/flow_ibm.hpp` (~line 900–1130) — gp overlay build + the MG-preconditioned
  BiCGStab driver; `set_ghost_projection(on, matrix_order, rhs_order, ...)` in
  `flow_bindings.cpp`.
- `peclet::core::scheme::ghost_closure` (core) — the PURE closure pieces, shared with the AMR
  band. **Constraint: closure-weight changes go here, stay pure, and keep float arithmetic
  identical between flow and core call sites.**
- Tests: `tests/study/ghost_projection_apriori.py` (reference harness + gates),
  `tests/kokkos_mpi` ghost ctests (np=1,2,4 bit-exactness), regression ghost cases,
  `scripts/validate_zick_homsy_sdflow.py`.
- Data: the diverging beds are committed —
  `peclet-examples/benchmarks/porous-scaling/results/packings/packing_1024x512x512_r16_phi0.50_s116.npz`
  (7823 spheres, np=16's bed) and `..._1024x1024x512_...s132.npz`; the stable reference bed is
  `..._256x256x256_...s100.npz`. Diverged-run logs/JSONs: `results/snellius-h100/weak_np16_ghost*`.

## Phase A — analysis (no production code; deliverable = findings note + go/no-go per fix)

- **A1 Reproduce small.** Run the s116 bed at 512×256×256 (R=8) ghost on the workstation GPU
  (fits 16 GB). Expected: it diverges → the bug is bed-geometry-portable and locally
  debuggable; if it doesn't, escalate resolution/try s132 before concluding scale matters.
  Add a per-step `max|u|` growth log to the march; measure the growth exponent. Controls: a
  fresh-seed bed of the same size (converges?), dt=6 (stabilizes?).
- **A2 Row forensics.** Instrument the gp build (debug env, style of `PECLET_FLOW_AGMG_DEBUG`)
  to dump per-row: cascade branch taken, θ values (incl. EXTENDED θ∈(1,2) slivers), weight
  magnitudes, `rho`, row asymmetry. Correlate the worst rows with local gap width (sphere-pair
  surface distance from the packing). Compare distributions: s100 (stable) vs s116
  (diverging). Hypothesis to confirm/refute: the divergence is seeded by a handful of
  identifiable pathological rows (thin-gap sliver/EXTENDED branch), not spread diffusely.
- **A3 The "fully Robust-Scaled" gap list.** Systematic diff of ghost's conditioning treatment
  vs cut-cell's: one scalar `rho` per row vs per-axis rescale; is `rho` applied consistently
  to matrix couplings AND RHS AND the diagnostic; are QUAD→LIN fallbacks triggered by
  *conditioning* or only by source availability; sandwich/sliver branch audit; θ clamp
  (`GP_THETA_MIN`) adequacy for near-tangent pairs. Output: an itemized list of what phase B
  must add.
- **A4 Symmetric-split feasibility.** With A = S + N (S = the binary-openness 7-point operator
  the CutcellMG hierarchy already rediscretizes; N = the gp overlay delta): measure ‖N‖
  row-wise on the sample beds and estimate ρ(S⁻¹N) by power iteration (a python-level
  experiment via the existing apriori harness). ρ < 1 → plain deferred correction converges,
  outer count ≈ log(tol)/log(ρ); ρ ≥ 1 on thin-gap rows → phase B's hardening must shrink it
  first, and/or under-relax the correction. Also just *try* the existing
  `matrix_order=1, rhs_order=2` mode on the diverging case — it may already be the answer in
  embryo, and its current behavior (iteration count, stability, accuracy on zh_sphere) is the
  baseline phase C must beat.
- **A5 Compatibility bias.** On a small periodic bed, compute the residual of `1ᵀA` (how far
  the left null vector is from constant) and the per-solve compatibility bias under the
  current mean-removal; estimate the per-step momentum injection it implies. Decides how much
  of the divergence is H2 (bias accumulation) vs H1 (bad rows) — they compose, but the fix
  priority differs.

## Phase B — full Robust-Scaling of the ghost closure (fix the fragility)

Driven by A3's list; the principles, mirroring cut-cell:

- Every closure row bounded: weight magnitudes capped by a conditioning threshold; when a
  branch's `D` falls below it, *reduce the order locally* (QUAD → LIN → BC_ONLY) rather than
  emit large weights — order reduction at isolated pathological faces is invisible in the
  global error (cut-cell's precedent).
- The rescale threaded consistently: matrix couplings, RHS, and diagnostic all see the same
  per-row (or per-axis, if A3 says needed) factor — no half-scaled rows.
- The EXTENDED-θ sliver branch gets an explicit conditioning gate (it interpolates with
  D > 2 by construction, but its *source* values can still sit in a near-degenerate line).
- Changes land in `peclet::core::scheme::ghost_closure` (pure, shared with AMR) + the overlay
  assembly; flow and core stay float-identical.

Acceptance: A1's diverging case converges; on beds where no new gate fires the build is
**byte-identical** (gate the trigger with a counter, assert zero on the stable regression
geometries); zh_sphere drag and the ghost regression cases unchanged within tolerance;
`tests/kokkos_mpi` bit-exact np=1,2,4; `ghost_projection_apriori.py` extended with a
near-tangent two-sphere pathological case that fails before B and passes after.

## Phase C — symmetrized deferred correction (fix the cost)

Formulation: outer iteration `S φ^{k+1} = b − N φ^k`, with

- **S = the binary-openness operator — which the CutcellMG hierarchy already rediscretizes
  exactly.** The current "surrogate preconditioner" becomes the *true operator* of the inner
  solve: symmetric → **MG-PCG applies**, grid-independent at cut-cell-like counts (~12–30),
  and the agglomerated bottom becomes meaningful on the ghost path for free.
- N = the gp overlay delta (already isolated as `gpApplyDelta`), moved to the RHS via the
  existing two-weight-set seam (`wm_*` becomes unused in the matrix; keep the BiCGStab path
  compiled and selectable for A/B).
- Compatibility per outer iteration: S has the clean constant null space → standard mean
  removal is *exact* — this structurally eliminates the H2 bias channel.
- Under-relaxation knob on the correction (default 1.0) in case ρ(S⁻¹N) approaches 1 on rough
  beds; A4's measurement calibrates it, B's hardening shrinks it.

Cost model: m_outer × (cut-cell-like inner). A4's ρ predicts m; target **total ≤ 2× cut-cell
iterations** (vs today's 4–7×). Stopping: outer residual on the TRUE operator (A), so accuracy
is not weakened.

Acceptance: iteration target on the regression beds; k values unchanged within tolerance;
refine-ladder ghost k(N) curve reproduced (the 0.1 % k∞ agreement is the physics gate);
np=16/np=32 upscale rungs rerun on Snellius with a result TAG — both must converge and the
study page's incident entry gets closed out.

## Order and gates

A (1–2 sessions, mostly instrumentation + python harness) → B (closure hardening; small,
surgical, heavily gated) → C (driver restructuring; keep old path selectable). Separate
commits per phase; regression baselines re-recorded only with stated justification. B before
C matters: hardening shrinks ρ(S⁻¹N), which C's outer count depends on.

## Practicalities

CUDA+MPI build `flow/build_cuda_mpi_ch`, OpenMP-MPI `build_mpi`; kmpi ctests need
`-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`. Host memory is capped (cfd-slice, 450G) — sequential
jobs. The GPU may be shared with the comm-scaling session — coordinate. Snellius packing must
use the CPU dem build (H100 corruption bug, see porous-scaling README). Commit at milestones;
push only when asked.
