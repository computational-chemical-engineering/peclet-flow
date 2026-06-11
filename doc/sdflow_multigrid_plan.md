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

**Phase 2 — robustness & steady-march efficiency. [partial]**
- **[NEW — top priority] Fix the velocity-diffusion MG.** Phase-1 isolation revealed
  `set_velocity_multigrid` is the real "+4% at N=128" defect: it under-converges the IBM diffusion
  (a *fixed* 4 V-cycles over a geometry-blind constant-coefficient coarse operator, `setDiffusionCoarse`).
  Fix by **rediscretizing the velocity-diffusion coarse operator with the cut-cell geometry** — the exact
  Phase-1 technique applied to `I − νΔt·Lap_cutcell` (coarsen the openness, build the Helmholtz operator
  per level) — and/or wrap in PCG / use more V-cycles. Until done, velocity RB-GS is exact; use it.
- MG-PCG default-available (symmetric V-cycle) for stiff/dense packings. *(already wired:
  `set_pressure_pcg`.)*
- **FMG / warm-start**: reuse the previous timestep's pressure as the initial guess (cheap, big win for
  steady marches) and FMG for the first solve.
- smoother tuning (RB-GS pre/post sweeps; optional Chebyshev on coarse levels).

**Phase 3 — distributed coarse-grid scaling (makes it usable at scale).**
- **Agglomerated redundant coarse solve**: below a per-rank-block threshold, gather the coarse level to
  one rank, solve there, scatter the correction. Removes the latency-bound deep-coarse comm.
- per-level halo tuning (coalesce/condense small coarse messages).
- weak- and strong-scaling tests at np = 1,2,4,8 (and multi-GPU) — iteration count flat, time/solve
  scaling cleanly. This is the deliverable that justifies "MG for large multi-GPU."

**Phase 4 — generalize & clean up.**
- (velocity-diffusion rediscretized coarse operator moved up to Phase 2 — it's the active defect.)
- API already migrated to a mode (`set_solid(..., pressure_coarse=...)` / `set_cutcell_pressure_operator`
  `coarse_mode`). The Galerkin pressure path is **not buggy** (it converges to the right answer, just the
  slowest coarse model) — keep it for comparison; the rediscretized mode is the default. Update
  `doc/sdflow_pnm_parity.md` as findings settle.

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
