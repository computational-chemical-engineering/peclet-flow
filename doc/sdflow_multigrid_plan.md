# sdflow multigrid: design review & plan

Goal: a **functional, efficient multigrid pressure solver** for sdflow that works on large multi-GPU
(MPI block-decomposed) cut-cell problems, where plain RB-GS does not scale (its iteration count grows
~O(N) with resolution). Motivation: at scale, the pressure-Poisson solve dominates and needs a solver
whose iteration count is ~O(1) in N.

> **STATUS (2026-06-11): Phase 0 + Phase 1 DONE.** A *rediscretized* cut-cell pressure coarse operator
> (`DistributedPoissonMG::setFineVariableOperatorRediscretized`) is **grid-independent** (V-cycle ρ≈0.15
> flat in N, **8 PCG iters flat** vs RB-GS 18→47 / Galerkin 15→30 / const-MG 10→13) and **correct**
> (matches Zick & Homsy to <0.05% at N=128 across φ). Exposed as `set_solid(..., pressure_coarse=
> "rediscretized")` (new default). All 24 MG ctests green (np=1,2,4). **Discovery along the way:** the
> "+4% drift" previously blamed on the Galerkin *pressure* path is actually the **velocity diffusion
> multigrid** under-converging the IBM operator (confound). Every *pressure* path is correct; the velocity
> MG is the next fix (see "Velocity-diffusion MG" below). Tooling: `tests/profile_mg_scaling.cu` compares
> all coarse modes; `scripts/validate_zick_homsy_sdflow_vs_pnm.py` checks correctness.

## 0. Where we are (grounded in the code)

**pnm_backend's MG works** (`src/cfd_solver_multigrid.cu`) and is, concretely, a **geometric multigrid
with rediscretized coarse operators**:
- `build_sdf_hierarchy_host` — coarsen the *geometry* (SDF) level by level.
- `compute_pressure_operator_kernel` — on each level, **rebuild** the 7-point cut-cell Poisson operator
  from that level's face fractions + SDF. Every coarse level is therefore a *genuine discretization*,
  not an algebraic product.
- smoother: `solve_rbgs_mg_kernel` = **Red-Black Gauss-Seidel** (exactly the local smoother intuition).
- transfers: `restrict_average_kernel` (8:1 average) + `prolongate_trilinear_add_kernel` (trilinear
  correction). `subtract_mean_mg_kernel` handles the singular (all-Neumann/periodic) null space.
- Used as a **standalone V-cycle solver** (no outer Krylov). It nails Zick & Homsy to <0.1%.

**sdflow already has most of the machinery** (`src/mac_multigrid.cuh`, `DistributedPoissonMG`):
- distributed per-level blocks with 2:1 local coarsening + per-level halo exchange (transport-core),
- RB-GS smoother `mg_smooth_var_k` (red-black `color`), residual, **geometric** restriction
  (`mg_restrict_k`, 8:1 average) + trilinear prolongation (`mg_prolong_k`), per-level mean removal,
- an optional Chebyshev (point-Jacobi) smoother,
- both a standalone V-cycle (`solve`) and an MG-preconditioned CG (`solve_pcg`).

**What sdflow is missing is exactly the piece that makes pnm correct: a consistent variable-coefficient
coarse operator.** Its two current coarse-operator options are both inadequate (the code comments admit
it):
- `galerkin=false`: coarse levels are **constant-coefficient** (ignore the geometry) — *"a poor coarse
  model for stiff cut cells"*. Converges to the right answer (it's only a preconditioner) but the coarse
  correction is weak, so it barely beats single-level RB-GS for dense packings.
- `galerkin=true`: **unsmoothed-aggregation Galerkin** (`A_c = PᵀA_fP` via summation restriction +
  injection prolongation). This is the **buggy path** — on the SC sphere it drifts +4% in drag at N=128
  and worsens with N (see `doc/sdflow_pnm_parity.md`). The aggregation operator + injection/summation
  transfers are not a consistent pair for the singular cut-cell operator.

So the plan is **not** "build an MG from scratch" — it's "add the rediscretized-coarse-operator mode
that pnm proved, into sdflow's existing distributed V-cycle, and then make the coarse grid scale on
multi-GPU."

## 1. Review — the multigrid design axes (what can be envisioned)

### (a) Coarse-operator construction — the decisive choice
| option | idea | for cut cells | verdict |
|---|---|---|---|
| **Geometric rediscretization** | coarsen the geometry (face openness / SDF), rebuild the cut-cell operator per level | consistent by construction — each level is a real discretization | **recommended** (pnm-proven) |
| Galerkin / RAP `A_c=RA_fP` | algebraic coarse operator from transfers | naive aggregation is what's broken now; a *proper* operator-induced P (smoothed aggregation, de Zeeuw matrix-dep. P) can work but grows stencils and is complex | avoid on a structured grid |
| Constant-coefficient coarse | ignore geometry on coarse levels | weak coarse correction for dense packings | keep only as a cheap fallback |

The grid is a **structured Cartesian** block — geometric MG is the natural, cheap fit. Full **AMG**
(algebraic, setup-heavy, irregular coarsening, hard on GPU + MPI) buys robustness we don't need once
rediscretization handles the geometry. **Skip AMG.**

### (b) Smoother (local relaxation)
| option | parallel | strength | note |
|---|---|---|---|
| **Red-Black GS** | red/black colored, 1 halo exchange per color | strong smoothing factor | **recommended / already present**, pnm-proven |
| Damped Jacobi | fully parallel | weaker (more sweeps) | simple fallback |
| Chebyshev / polynomial (Jacobi-pre) | matvec-only, **communication-light** | tunable band | **already present**; attractive at scale / on coarse levels where RB-GS's per-color halo dominates |
| ILU / line-SOR | poor on GPU+MPI | strong | not for us |

Recommendation: **RB-GS as the default smoother** (your intuition; it's what pnm uses and what works for
small per-rank blocks), with the existing **Chebyshev smoother as an option** for communication-bound
regimes (deep/coarse levels, very many ranks).

### (c) Cycle type
- **V-cycle** — default, cheapest. Usually enough for Poisson with a good smoother+coarse operator.
- **W-cycle** — stronger coarse work, more coarse-level comm; rarely needed for Poisson, can help very
  stiff operators.
- **F-cycle / Full-Multigrid (FMG)** — solve coarse-first and prolong up for a near-converged initial
  guess. **Valuable for steady marches** (each timestep's pressure solve starts near the answer) and for
  the very first solve.

### (d) Standalone solver vs Krylov-preconditioner
- **Standalone V-cycle** — cheapest per solve; pnm does this and it's accurate. Good default.
- **MG-preconditioned CG (MG-PCG)** — robust for the singular / mildly ill-conditioned cut-cell
  operator; needs a *symmetric* V-cycle (forward pre-smooth + reverse post-smooth — already supported via
  `smooth(reverse=true)`). Best insurance for stiff/dense cases. **Recommendation: support both; default
  to standalone V-cycle, expose MG-PCG for hard cases.**

### (e) Distributed / multi-GPU coarse-grid strategy (the real scaling problem)
As levels deepen, per-rank blocks shrink to the halo width and **coarsening must stop**; the coarsest
grid then stays *distributed and not-small*, so (i) the coarse correction is weak and (ii) coarse-level
halo messages are tiny and **latency-bound** — the classic "MG doesn't strong-scale" wall.
| option | idea | trade-off |
|---|---|---|
| **Agglomerated redundant coarse solve** | below a size threshold, gather the coarse level onto 1 (or a few) rank(s), solve redundantly (serial MG / direct), scatter back | removes deep-coarse latency; standard at scale (hypre/AMReX); needs a gather (transport-core has one) + a serial coarse solver | **recommended** |
| Keep-distributed coarse Krylov | run a CG with many iters on the coarsest affordable level | no gather, but comm-heavy and slow | fallback |
| Stop early + heavy smoothing | few levels, many RB-GS sweeps on a mid-size coarsest grid | simplest; loses O(1) scaling | interim |

## 2. Proposed plan (phased)

**Phase 0 — baseline + decisions. [DONE]**
`tests/profile_mg_scaling.cu` (extended to compare all coarse modes) on the SC packing, V-cycle ρ and
MG-PCG iters to 1e-8 vs N:

| N | 1-level RB-GS | const-coeff MG | Galerkin MG | rediscretized MG |
|---|---|---|---|---|
| 32 | 18 iters | ρ0.36 / 10 | ρ0.65 / 15 | ρ0.15 / 8 |
| 64 | 25 | ρ0.42 / 12 | ρ0.81 / 20 | ρ0.15 / 8 |
| 128 | 47 | ρ0.45 / 13 | ρ0.90 / 30 | ρ0.16 / 8 |

Every non-rediscretized path's iteration count grows with N (Galerkin's V-cycle ρ blows up toward 1).
Decisions confirmed: standalone-V default + MG-PCG option; RB-GS smoother default.

**Phase 1 — geometric rediscretized coarse operator (the core fix). [DONE]**
`DistributedPoissonMG::setFineVariableOperatorRediscretized` (new `mg_coarsen_open_avg_k` kernel):
1. **Coarsen the geometry** — area-average the staggered face openness (`ox,oy,oz`) 2:1 per level (coarse
   face = average of its 4 fine sub-faces; *not* the `mg_agg_T_k` sum the Galerkin path uses), with a
   periodic ghost exchange.
2. **Rebuild the operator per level** via the existing `mg_build_op_k` at coarse spacing (idx2 → idx2/4ᴸ)
   — every level is a genuine cut-cell discretization.
3. **Geometric V-cycle** unchanged (`galerkin_=false`: average restriction + trilinear prolongation +
   RB-GS + per-level mean removal). Coarse openness stashed in the per-level `tx/ty/tz` scratch.
Result: grid-independent (ρ≈0.15 flat, 8 PCG iters flat) and correct (Z&H <0.05% at N=128 across φ).
Exposed via `set_solid(..., pressure_coarse="rediscretized")` (default). All 24 MG ctests green.

**Phase 2 — robustness & steady-march efficiency. [done, with one deferral]**
- **Velocity-diffusion MG — rediscretization ATTEMPTED, diverges; DEFERRED.** Applying the Phase-1
  technique (`I − νΔt·Lap_cutcell` from coarsened cell-face openness) *diverges* (K→4.99 at 4 cycles,
  NaN beyond): the velocity fine stencil is **row-scaled by `D_rescale`** (the Robust-Scaled cut-cell
  trick), so it is inconsistent with a clean `I−β·L` coarse operator under geometric transfers — and the
  staggered velocity geometry is *not* the cell-face openness. A proper velocity MG needs the per-
  component velocity geometry and a coarse operator consistent with the Robust-Scaling (research effort,
  secondary payoff). Reverted to the const-coeff coarse (converges, slowly). **Velocity RB-GS
  (`set_velocity_solver_params`) is exact and the recommended default** — the velocity Helmholtz is the
  *easy*, non-singular operator; the pressure solve is the one that needed MG.
- **Warm-start — DONE (opt-in, `set_pressure_warmstart`, default off).** Seeds the pressure solve from
  the previous step's projection potential. Correct (same K on/off, PCG converges to tolerance); default
  off preserves the bit-exact cold-start cell-for-cell ctests; turn on for steady production marches.
- MG-PCG already available (`set_pressure_pcg`). FMG and Chebyshev-on-coarse left as future tuning.

**Phase 3 — distributed coarse-grid scaling (makes it usable at scale). [characterized; impl pending]**
- **Measured first:** the rediscretized MG gives *identical* convergence at np=1/2/4 (ρ≈0.15, **8 PCG
  iters** at N=64/128, independent of rank count). So the distributed coarsening + per-level halo are
  correct and grid-independence already survives through np=4 — **agglomeration is not yet a bottleneck
  at these scales.** The wall appears only at large np (many GPUs), where per-rank blocks become too
  small to coarsen to a small global grid (the coarsest level stays distributed and latency-bound).
- **Agglomerated redundant coarse solve** (the fix for large np): once distributed coarsening caps out,
  gather the coarsest level's operator+RHS to root, solve there (serial), scatter the correction. Removes
  the latency wall and decouples the level count from the per-rank block size. **Correctness is testable
  at np≤4; the scaling benefit needs many-rank / multi-GPU hardware (not available on the 1-GPU dev box),
  so it is deferred until that can be validated** — implementing it blind would ship unverifiable infra.

**Phase 4 — generalize & clean up.**
- (velocity-diffusion rediscretized coarse operator moved up to Phase 2 — it's the active defect.)
- API already migrated to a mode (`set_solid(..., pressure_coarse=...)` / `set_cutcell_pressure_operator`
  `coarse_mode`). The Galerkin pressure path is **not buggy** (it converges to the right answer, just the
  slowest coarse model) — keep it for comparison; the rediscretized mode is the default. Update
  `doc/sdflow_pnm_parity.md` as findings settle.

## DEFERRED WORK — agglomerated coarse solve (required for large-np / multi-GPU)

**Status:** not implemented. Deferred because its payoff is unverifiable on the 1-GPU dev box (np≤4 is
already grid-independent at 8 iters, and the trigger condition below would not even fire at np≤4). Pick
this up **before any production run at large rank counts / on real multi-GPU**, where it is required to
keep the pressure MG grid-independent.

**Why it is needed.** `DistributedPoissonMG::init` coarsens 2:1 per level while each rank's block stays
even and the partition stays aligned (it asserts this). The number of levels is therefore bounded by the
*per-rank* block size, not the global size. At large np the per-rank block is small, so coarsening stops
early → the coarsest *global* grid is still large → the coarse correction is weak (iteration count grows
with np) **and** the coarse smoother's halo exchanges are tiny, latency-bound messages. This is the
classic "MG stops strong-scaling" wall. Measured here: np=1/2/4 all give 8 PCG iters (the wall is past
np=4 on these grids), so it does not show on the dev box.

**Design (the standard hypre/AMReX approach: redundant agglomerated coarse grid).**
1. **Distributed phase (existing).** Coarsen + smooth (RB-GS) + restrict as now, down to the coarsest
   level `K` that the aligned per-rank coarsening allows.
2. **Trigger.** When the level-`K` *global* grid is ≤ a threshold (e.g. 16³–32³ total cells) **or** the
   per-rank block can no longer coarsen, switch to agglomeration for the bottom solve instead of the
   current distributed `bottom_` RB-GS sweeps.
3. **Gather.** Assemble level-`K`'s global operator (the 7 stencil arrays `AC..AT`) and RHS onto root
   (and optionally a small subset of ranks for redundancy) via the ORB→global mapping. transport-core
   already has a gather-to-root for fields (used for VTI output); reuse it for the RHS and each stencil
   band. The operator only needs gathering once per `setFineVariableOperator*` (geometry is static); the
   RHS gathers every V-cycle.
4. **Redundant serial solve.** On root, solve the (small) coarse system to tight tolerance — a serial
   geometric MG continuing the coarsening to ~1 cell, or a direct solve. Cheap because the grid is tiny.
   Handle the singular pressure null space with the same mean removal.
5. **Scatter.** Broadcast/scatter the coarse correction back to the level-`K` blocks; prolong up as now.
6. **Variants / knobs.** redundant on 1 vs a few ranks (latency vs memory); threshold tuning; for the
   non-singular velocity Helmholtz the same machinery applies (no mean removal).

**Validation plan (when picked up).** (a) Correctness: the agglomerated V-cycle must converge to the
*same* solution as the all-distributed path at np=1,2,4 (and match the serial reference cell-for-cell on
a case small enough to trigger agglomeration). (b) Scaling (needs the hardware): PCG iteration count
**flat** in np at fixed problem size (strong scaling) and flat per-rank work (weak scaling) at np = 8,
16, 32, … on real multi-GPU — the deliverable that justifies "MG for large multi-GPU". Until (b) can be
run, do not ship it (untestable infra). The hook points are `DistributedPoissonMG::vcycle` (bottom
branch) and `init` (record `K` / the agglomeration threshold).

## 3. Open questions / decisions
1. **Default solver shape:** standalone V-cycle vs MG-PCG default. *(Current: MG-PCG available via
   `set_pressure_pcg`; standalone V-cycle via fixed `n_pois`. Lean V-cycle default.)*
2. **Coarse-grid strategy for scale (Phase 3):** agglomerated redundant coarse solve (recommended) vs a
   keep-distributed coarse Krylov?
3. **Keep the constant-coefficient coarse mode** as a cheap fallback, or expose only rediscretized? *(All
   three modes currently exposed via `pressure_coarse`.)*
4. **Coarsening rule for the face openness** — plain area-average (current, validated to <0.05% vs Z&H)
   vs a series/harmonic transmissibility combination near boundaries. Area-average works; revisit only if
   a denser packing underperforms.
