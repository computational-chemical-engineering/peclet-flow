# Pressure-MG depth that follows the grid, and an on-device bottom at every rank count

Design note, 2026-10-24. Branch `mgdepth` (worktree `suite/flow-mgdepth`, from main `89e5fa7`).
Answers `bubble_column_perf/gpu_diag/BRIEF_ARCH_MGDEPTH.md` (recon digest `RECON_MGDEPTH.md`).
Implementer: `opus-implementer`, who has seen neither the brief nor the conversation; everything
needed is here. Anchors are `file:line` at `89e5fa7`. Parents: `doc/vof_step_performance_design.md`
§13 (the direct bottom engine) and §14 (H-1 host, H-6 distributed host), `doc/vof_gpu_coarse_tail_design.md`,
`../docs/DECOMPOSITION_AND_MULTIGRID.md`.

Terms. **Depth D** = number of levels including the fine one (`nLevels`); level ℓ = 0 … D−1; the
**bottom** is level D−1. **g_ℓ** = level ℓ's GLOBAL inner dims (`Level::gdim`). **Singular** operator =
`removeMean_` (periodic / all-Neumann / IBM, no outflow face); **anchored** = Dirichlet outflow present.

## 0. Outcome first

1. **Depth rule.** The default depth is derived, not 4: the **shallowest D ≥ 4 whose global coarsest
   grid g_{D−1} the bottom engine admits**, or the natural end of coarsening if no such D exists.
   "Admits" is the direct engine's own size test (n ≤ 8192 cells and a plane of ≤ 192 cells), or
   "every axis ≤ the bottom extent (4)" for a smoothed bottom. Decided on global dims only, so the
   global hierarchy is independent of np and backend. `set_pressure_multigrid(on, levels)` with
   `levels ≥ 1` stays an exact override; `levels=0` (the new DEFAULT) = AUTO, as `set_decomposition`.
2. **Bottom engine.** No new engine. The existing FP32 block-tridiagonal factor + FP64 FCG
   (`src/mg_bottom_direct.hpp`) is the bottom everywhere the rule lands; the rule exists to keep the
   bottom inside its admission. Under device MPI it becomes eligible by running H-6's redundant
   global-bottom scheme on device (WO-3). Host GraphAMG remains only as a warned fallback.
3. **Rebuild policy.** The per-step `buildAmg` leaves the production path because GraphAMG does;
   the direct factor's refactor per operator change (already designed) stays. No numeric-only
   GraphAMG refresh.
4. **Bitwise** wherever the depth-4 bottom is already admissible: all 14 `state_hash` cases, the
   bubble column 128×96×64, every isotropic grid ≤ 96³. Moved (recorded, census-listed): grids whose
   depth-4 bottom is not admissible (≥ 128³ isotropic, long channels, multi-rank weak scaling).

## 1. Problem and scope

**Problem.** The depth is a constant (`int nLevels_ = 4;`, `src/flow_ibm.hpp:4931`). A level halves
every coarsenable axis, so at 256³ the depth-4 bottom is 32³ = 32 768 cells per rank, and under
weak scaling the GLOBAL bottom grows ∝ np (np 32 at 256³/rank: 128×128×64 ≈ 1 M cells). Above the
direct engine's caps the singular path's agglomerated bottom goes to host GraphAMG
(`mac_cutcell_mg.hpp:3806-3830` → `graphAmgSolveBottom` :5502), gathered to every bottom rank,
solved by host PCG each V-cycle, and **rebuilt every step** on the porous / variable-ρ paths because
`setOpenness` resets it (`amg_.reset()` :2246, called from `flow_ibm_project.hpp:1340/1382`). Under
device MPI the direct engine is never eligible (`if (distributed_ && !kHostMemory)` :4778), so every
device run at np > 1 with an agglomerated bottom is on that host path regardless of size.

**Measured** (brief §4; `gpu_diag/HANDOFF_WEAK.md`, `gpu_diag/weak/results/`):

| config | depth | ms/step | pressure share | iters/step |
|---|---|---|---|---|
| H100, VoF swarm 256³, np1 | 4 | 450.5 | 91 % | 7.93 |
| H100, same, np4 2×2 host-staged | 4 | 2137.8 | 95 % | 7.93 |
| workstation GPU, 192³, np1 | 4 / 6 / 7 | 558 / 87.4 / 90.1 | — | 7.9 / 9.0 / 9.0 |

nsys np1: kernels 96 ms, CUDA API 109 ms, ≈ 245 ms host. py-spy 192³: 84 % of `step()` in
`graphAmgSolveBottom` (`buildAmg` 29 %, `pcgAmg` 39 %, `HostCsrOp::apply` 26 %). The bubble column
128×96×64 at depth 4 (bottom 16×12×8 = 1536 cells, direct engine on device) runs 13.5 ms/step on an
RTX 5080: the pathology starts exactly where the bottom leaves the direct engine's admission.

**Evidence that depth is not free** (decisive for the rule): `doc/vof_step_performance_log.md` E1,
bubble column, exact bottom one/two levels deeper than 16×12×8: 13.00 → 21.50 / 22.05 iterations
(`auto` and `agglomerated` identical at each depth, so the cost is the deeper *coarse operators*,
not the bottom solve). `DECOMPOSITION_AND_MULTIGRID.md` §2.7, single-phase 2048×64×64 channel:
exact bottom 4.0 iterations at depths 4, 6 and 8 — depth is free there. The penalty is a
high-contrast phenomenon and it saturates (E1: L5 21.5, L6 22.05; swarm 192³: L6 9.0, L7 9.0). Its
mechanism is on the register (flow.md, 2026-08-31, "S3 coarsening indefiniteness is real …" and
"WO-M FINAL … S3 indefiniteness deepens with depth"): `coarsenOpenAvg` averages face coefficients
arithmetically, so across a 1000:1 jump the coarse coefficient is ≈ 0.5 where ≈ 2e-3 is right, and
the V-cycle preconditioner turns indefinite — more so the deeper it goes. That is also why Chebyshev
is the variable-ρ / porous default driver.

**In scope:** the default depth of the pressure `CutcellMG` (staggered and collocated/VoF solvers,
the one `mg_` per solver; `flow_ibm_geometry.hpp:863/866`), its forecast (`CutcellMG::predict`,
`predict_hierarchy`), the device-MPI eligibility of the direct bottom, the GraphAMG rebuild policy.

**Out of scope:** halo transport and the SURF issue; the 2×2 CUDA-aware pathology; the velocity MG
(`vmgLevels_`), ScalarMG, amr's MG; the weak-scaling driver; coefficient-aware coarsening (S3, the
real fix for E1's penalty); factor-3 coarsening; the decomposition (partitions are untouched — the
decomposition never reads `nLevels_`, whose only consumers are `flow_ibm_geometry.hpp:863/866`).

## 2. Constraints and invariants

- **Device-first, MPI-distributable;** host only as oracle. The production device path must not
  compute on the host. MPI transport through pinned staging is transport, as the host-staged level
  halos already are (§4.5, Q4).
- **No env var changes a result.** The depth is an API setting; `mgDebugLevel()` stays debug output.
- **Physical inputs only.** The depth is derived; `levels` remains an explicit override. The bottom
  extent stays a per-axis cell count (suite-wide register, 2026-09-24, USER).
- **Rank- and backend-independence of the hierarchy.** D and every g_ℓ are functions of the global
  grid, the operator class, the bottom-mode setters and the metric — never of np, the partition,
  thread count, or host vs device. (This is what keeps np1-distributed ≡ single-rank and the
  np-equivalence gates meaningful, and is why a per-rank rule is rejected.)
- **The MG is a preconditioner.** A depth change cannot change the converged solution beyond the
  outer tolerance, provided every outer solve meets its stop test; it CAN change iteration counts
  and the attainable floor (flow.md:199's anchored round-off floor). Gates check exactly those.
- **Singular/anchored distinction (flow.md:199) stands:** auto never agglomerates an anchored bottom.
- **Three default outer drivers see the depth** (`flow/CLAUDE.md` "Pressure solve"): MG-PCG
  (single rank; the VoF benchmarks select it), the standalone V-cycle (multi-rank, constant ρ — no
  Krylov wrapper, so a worse coarse correction shows directly), and Chebyshev (variable ρ / porous;
  its bounds are re-estimated per hierarchy, `eigWarm_ = false` in `init`). Gates cover all three.
- **Bitwise where the rule returns 4.** Everything that resolves to D = 4 must be byte-identical.
- Never Rhie–Chow; forces stay inside the implicit predictor (untouched here).

## 3. The decision, and what was rejected

### 3.1 Depth rule: shallowest admissible, floor 4

D_auto = min( D_nat , min{ D ≥ 4 : A(g_{D−1}) } ), where D_nat is where coarsening ends by itself
(no axis can halve; under MPI with telescoping off, where in-place coarsening stops) and A is §4.1.

Why: per V-cycle the fine levels cost what they cost; the choice of D trades **bottom cost** (falls
≈ 32× per level: the solve ∝ n·b, n ÷ 8, b ÷ 4) against **coarse-operator quality** (E1: +65 %
iterations one level past 16×12×8 on the bubble column; ≈ +14 % for the swarm). An extra outer
iteration at 256³ costs ≈ 12 ms on an H100 (96 ms of kernels / 7.93); an admitted bottom costs a few
ms per step at most (§4.7). So the iteration count dominates, and the exact bottom should be as fine
as the exact engine can take cheaply — i.e. the shallowest admissible level. The risks are
asymmetric: being one level too shallow costs ≤ ≈ 5 % (bottom cost), one level too deep can cost
+65 % (E1). It is also design rule 3 of `DECOMPOSITION_AND_MULTIGRID.md` ("hold the coarsest grid
fixed, not the level count") made automatic: under refinement or weak scaling the bottom stays
between ≈ 1/8 and 1× the admission size.

The floor 4 is a compatibility choice: every grid whose depth-4 bottom is already admissible keeps
D = 4 bit for bit. (Small grids might do marginally better shallower; Q2.)

Rejected:
- **Fixed 4** (today): the bottom grows as N; measured 84 % of the step at 192³.
- **Per-rank coarsest extent E**: the global bottom still grows ∝ np under weak scaling, and the
  preconditioner would change with np (iteration counts a property of the rank count — the failure
  telescoping was built to remove).
- **Deepest (coarsen until every axis ≤ 4, smoothed tiny bottom):** pays E1's penalty for nothing;
  swarm 192³ L7 (3³ smoothed) 90.1 ms vs L6 (6³ exact) 87.4, same iterations.
- **A cost model choosing D by predicted time:** needs an iteration predictor; the penalty is
  problem-dependent (E1 vs swarm vs channel) and unknowable at geometry build.
- **Device-specific or host-specific depths:** host and device would run different preconditioners.

### 3.2 Bottom engine: the existing direct factor, made distributed on device

Rejected:
- **Device AMG at a fixed (shallow) depth** (core `GraphAMGDevice`, unused by flow): ≈ 150 launches
  per bottom solve × ≈ 14 per step, launch-bound (`vof_step_performance_design.md` §5 rejected list);
  GraphAMG on GPU measured 3–8× worse (flow.md, 2026-08-09); and a bottom that grows with np scales
  the wrong quantity (flow.md:56, the retired "hand the stopping level to GraphAMG").
- **Device FCG + smoother as the bottom:** iterations ∝ bottom extent² (Gauss–Seidel needs O(L²)
  sweeps per wavelength L; `agglomerateAuto` comment, :4757-4760); §2.7 smoothed depth-4 bottom
  13.5 vs 4.0 iterations exact.
- **Rank-0 solve + broadcast / telescope to one root:** rejected by the H-6 register entry; the
  redundant solve costs no wall time (the other ranks would idle) and saves the scatter.
- **Requiring CUDA-aware MPI for the device bottom** (§13 Q-D7.1's assumption): on a host-staged
  stack the alternative is the host GraphAMG, i.e. host *computation*; ≤ 64 KB of pinned staging per
  V-cycle is the same transport class the level halos use (Q4).

### 3.3 Rebuild policy

The direct engine refactors once per operator change (per step under variable ρ / porous), as now:
single-rank device on the side instance after `updateProperties()`; distributed (WO-3) in-stream
after the openness gather. GraphAMG keeps `amg_.reset()` in `setOpenness` and its lazy rebuild —
correct for the fallback, and now reached only on fallback configurations. The comment at
:2240-2245 ("the bottom level is tiny, so the per-step rebuild is negligible") becomes true by
construction; WO-4 points it at this note. **Rejected: a numeric-only GraphAMG refresh** (fixed
aggregates, new Galerkin values): complexity paid forever for a path the rule removes; under moving
interfaces the strength graph changes, so it would need its own re-setup trigger; the H-1 register
entry already rejected "GraphAMG setup reuse" in favour of the direct engine.

### 3.4 Telescoping, odd and non-power-of-two extents

- At np > 1 the rule needs telescoping (default ON, min extent 4) to reach the admitted level once
  blocks turn odd or fall below 4; with telescoping OFF, D_nat is where in-place coarsening stops,
  and the bottom may stay inadmissible (fallback + warning). The rule never adds a telescope stage
  at the bottom level (the bottom level is treated as the last, so `doTele` is not evaluated there).
- Odd / badly factored grids (400³ = 25·2⁴): coarsening ends at 25³ (15 625 cells); the rule returns
  D_nat = 5 (better than today's 50³), the bottom is inadmissible → host GraphAMG fallback with a
  one-time warning naming the remedy (dimensions with more factors of two; `predict_hierarchy`).
  Factor-3 coarsening is out of scope (rejected for the bubble column in the §13 register entry).
- Semi-coarsening and the anisotropic aspect rule are respected automatically: the test reads the
  actual g_ℓ the level loop produces.

## 4. The design

### 4.1 The admissibility predicate (host, pure, evaluated once per hierarchy build)

```
// g = a candidate bottom's GLOBAL inner dims. Static so that predict() calls the same code.
static bool bottomAdmissible(C3 g, int extent, bool anchored, int agglomMode, const int bc[6]) {
  const int e = extent > 0 ? extent : 4;                      // set_pressure_bottom_extent
  if (g.x <= e && g.y <= e && g.z <= e) return true;          // a smoothed bottom suffices
  if (anchored) return !kAnchoredFollowsGrid;                 // WO-2: true (=> D = 4); WO-5: false
  if (agglomMode == 0) return false;                          // 'smoother' forced: only small grids
  const long n = (long)g.x * g.y * g.z;
  return n <= kBottomMaxCells && bottomChoosePlanes(g, bc).valid();    // the engine's own test
}
// member use: bottomAdmissible(g, agglomExtent_, !removeMean_ || hasOutflow_, agglomMode_, bc_),
// all set BEFORE init/initMpi (WO-1 moves setAgglomerationMode before init; §4.2).
// predict(levels=0): extent 4, anchored false, mode -1, all-periodic bc.
```

- `kBottomMaxCells = 8192` (`mac_cutcell_mg.hpp:655`), `kBottomMaxPlane = 192` inside
  `bottomChoosePlanes` (`mg_bottom_direct.hpp:63/87`). **Single source of truth:** the depth target
  is the engine's admission; changing a cap moves both together. Validity of `bottomChoosePlanes` is
  bc-independent (it tries all three axes; bc only picks among valid ones): valid ⇔ the product of
  the two smallest extents ≤ 192. `predict` (no bc) may pass an all-periodic bc.
- Not in the predicate: the ≤ 64 fluid components condition (runtime, from the openness). A bottom
  that fails it takes the GraphAMG fallback, as today.
- `kAnchoredFollowsGrid` is a `constexpr bool` in `mac_cutcell_mg.hpp`, `false` until WO-5 flips it.
- Independent of `bottomSolver_` ('direct'/'algebraic') and of the backend, by design.

### 4.2 Resolution in the level loops

`nLevels == 0` means AUTO in `CutcellMG::init` (:1242), `initMpi` (:1628) and `predict` (:6940).
In AUTO the loops run with a cap of `kAutoMaxLevels = 30` and one change to the "is this the last
level" test, at :1263 (single rank, dims `inner`, which equal g_ℓ) and :1737 (MPI, dims `gs`):

```
const bool autoStop = autoDepth && (L + 1 >= kAutoMinLevels /*4*/) && bottomAdmissible(dims);
if (L + 1 < nLevelsCap && !autoStop) { ... compute ratio / telescope as today ... }
```

Everything else — `can()`, the aspect rule, the `next == inner` / `next == gs` natural-end break,
telescoping triggers and stages, `idleBelow`, `allocFp64(lv_.back())`, `buildBottomStore()`,
`computeFp32Last()` — is unchanged. Explicit `nLevels ≥ 1` takes the existing code path verbatim.

- **Replicated decision, no collective:** `gs`, the bc, the mode and the extent are identical on
  every rank, so every rank resolves the same D. With telescoping ON the per-level global dims at
  np > 1 must equal single-rank's; WO-1's MPI ctest proves it rather than this note assuming it.
- **Ordering in `Solver::setSolidInitPressureMg()`** (`flow_ibm_geometry.hpp:848-886`): call
  `mg_.setAgglomerationMode(pressGraphAmg_ ? 1 : pressAgglomMode_)` (today :883) BEFORE the
  `initMpi`/`init` call (keep the post-init call or delete it — `setAgglomerationMode` only assigns
  `agglomMode_`, :6855, so both are bitwise). `setBoundaryConditions` (which sets `removeMean_`,
  `hasOutflow_`, :1926-1936) already precedes init. `set_pressure_bottom_extent` writes `mg_`
  immediately (`flow_ibm_core.hpp:892`), so it is in force at the build if called before the
  geometry; a later call still changes agglomeration but not the depth already resolved — say so
  in both docstrings (the depth is fixed per hierarchy, as `levels` is today).
- `mgDebugLevel()`'s init line (:1298) prints `requested auto -> D (legacy 4 -> D4)`; for the
  census (WO-1) it also prints, while the default is still 4, what AUTO *would* resolve: run the
  predicate over the built levels' gdims, no second hierarchy.

### 4.3 API

- `set_pressure_multigrid(on, levels=0)`: `levels=0` (DEFAULT from WO-2) = AUTO; `1` = pure RB-GS;
  `≥ 2` explicit depth. `Solver::setPressureLevels`: `nLevels_ = levels < 0 ? 1 : levels` (0 is
  no longer clamped to 1 — no caller passes 0: grep of flow, coupling, pnm, peclet-examples
  2026-10-24). `nLevels_` default `0` from WO-2. The docstring states the rule in one sentence and
  the observable: the resolved depth is `len(diagnostics.pressure_mg_level_ratios())` (existing,
  `flow_bindings.cpp:934`) — **no new public name**.
- `predict_hierarchy(..., levels=0)` forecasts AUTO for the singular operator (docstring says so;
  for an outflow case before WO-5 the answer is 4; WO-5 adds an `outflow=False` keyword, Q6).
- Takes effect at the next geometry build, as today.

### 4.4 What the rule resolves (single rank unless stated; isotropic metric)

| global grid | depth-4 bottom | admissible? | D_auto | bottom | changes? |
|---|---|---|---|---|---|
| 16³, 16×16×8, 16×16×4 (state_hash) | ≤ 2×2×2 | small | 4 | same | no |
| 24×12×8 (state_hash channel, advect_bc) | 3×3×2 | small | 4 | same | no |
| 48³ / 64³ / 96³ | 6³ / 8³ / 12³ | direct | 4 | same | no |
| 128×96×64 bubble column | 16×12×8 = 1536 | direct | 4 | same | **no** |
| 64×48×32 (`bottom_direct_mpi`) | 8×6×4 | direct | 4 | same | no |
| 128³ | 16³ (plane 256) | no | 5 | 8³ | yes |
| 192³ (swarm, workstation) | 24³ = 13 824 | no | 5 | 12³ = 1728 | yes |
| 256³ (swarm, H100 np1) | 32³ | no | 6 | 8³ | yes |
| 256×192×128 (bubble column ×2) | 32×24×16 | no | 5 | 16×12×8 | yes |
| 2048×64×64 channel | 256×8×8 = 16 384 | no | 5 | 128×4×4 = 2048 | yes |
| 512×512×256 (np4 2×2 at 256³) | 64×64×32 | no | 6 | 16×16×8 = 2048 | yes |
| 1024×1024×512 (np32 at 256³) | 128×128×64 | no | 7 | 16×16×8 | yes |
| 400³ | 50³ | no | 5 = D_nat | 25³ inadmissible | yes, fallback |
| 128×32×32 outflow (anchored) | 16×4×4 | WO-2: forced true | 4 | same | no (WO-5: 6, 4×2×2) |

All 14 `state_hash` cases are unchanged: 8 default-depth cases resolve to 4 with the same bottom; 6
pass `set_pressure_multigrid(True, 3)` explicitly (`state_hash.py:76/300`). `sdflow_regression.py`
passes explicit levels (:128-136), unaffected.

### 4.5 The distributed device direct bottom (WO-3)

H-6 (`buildGlobalBottomLevel` :4923, `buildGlobalBottomOp` :4989, `gatherGlobalBottomRhs` :5042,
`scatterGlobalBottomX` :5064) builds `gb_`, a private single-rank-shaped copy of the global bottom on
every rank holding the bottom (`bt.comm`), from allgathered openness (per operator change) and rhs
(per V-cycle); each rank factors and solves it redundantly and extracts its own block. Its data
movement is host code (`create_mirror_view_and_copy`, host loops, `std::vector`s allocated per call).
WO-3 makes that data movement device-resident and **unified for both backends** (it is pure data
movement, so the host results stay bitwise):

1. **Per hierarchy** (`buildGlobalBottomLevel`): as today compute `gbLocal_`, `gbCnt1_/Dsp1_/Cnt3_/Dsp3_`,
   `gbGid_`. Additionally allocate, once: `gbGidDev_` (`View<int*, CCMem>`, copy of `gbGid_`);
   `gbSend_` (`CCField`, 3·nloc), `gbRecv_` (3·ntot); and, only when `CCMem` is not host-accessible,
   pinned mirrors `gbSendH_`, `gbRecvH_` (`Kokkos::SharedHostPinnedSpace`, as `PinnedHost` :2540).
   On host builds the "mirrors" are the same Views (no copy).
2. **Pack** (kernel on `CCExec`): slot s ∈ [0, nloc), i = s mod nx_b, j = (s / nx_b) mod ny_b,
   k = s / (nx_b·ny_b) (the order of today's host loops, x fastest); p = (i+g) + (j+g)·ex + (k+g)·ex·ey;
   rhs: `gbSend_(s) = bt.rhs(p)`; openness: `gbSend_(3s+c) = bt.o{x,y,z}(p)` (today's interleave).
3. **Transport:** device builds `deep_copy(CCExec, gbSendH_, gbSend_)`; `CCExec().fence()`
   (instance-scoped, never a bare `Kokkos::fence()`); `MPI_Allgatherv(gbSendH_, …, gbRecvH_,
   gbCnt*, gbDsp*, MPI_DOUBLE, bt.comm)`; `deep_copy(CCExec, gbRecv_, gbRecvH_)`. Always pinned
   staging (Q4). Host builds: `CCExec().fence()` then `MPI_Allgatherv` on the Views directly.
4. **Unpack** (kernel): r ∈ [0, ntot): `gb_.rhs(gbIndex(gbGidDev_(r))) = gbRecv_(r)` (openness:
   three fields), `gbIndex` made `KOKKOS_INLINE_FUNCTION` (it is index arithmetic only, :4979).
   Then, for the openness, the existing `fillLocal`, `applyBoundaryOpenness`, `buildCutcellOpFace`
   on `gb_` (already Kokkos kernels).
5. **Extract** (`scatterGlobalBottomX`, kernel over `bt`'s extended index): inner cells read
   `gb_.x(gbIndex(gid))`, gid = (og.x+i) + (og.y+j)·G.x + (og.z+k)·G.x·G.y with G = `bt.gdim`
   (the `gbLocal_` formula, :4950); ghost cells 0, as today.
6. **Eligibility edits:** delete the device exclusions at :4778, :4824, :4888 (`!kHostMemory` /
   `kHostMemory &&`) and the `kHostMemory` condition at :2256 (`agreeDirectBottom`). Everything else
   in `directBottomIneligible[All]` stays (singular only, agglomerated, n ≤ 8192, planes, 1–64
   components, replicated agreement).
7. **Sync budget:** per V-cycle one instance fence + two small copies; buffers preallocated per
   hierarchy; zero global fences, zero View allocations in `step()` (`device_sync_budget` rule).
   The single-rank side-instance prefactor hoist is NOT extended to the distributed path in WO-3
   (the openness gather needs MPI on the host thread); refactor in-stream after the gather. A hoist
   is a later, separately gated optimisation.
8. **Determinism:** every bottom rank holds identical gathered bytes and runs the same kernels;
   the factor and FCG are bitwise independent of team size (§13 register entry), so copies agree
   bitwise on a homogeneous GPU partition. On a heterogeneous one they agree to the FCG's τ = 1e-5,
   the same order as the inexact bottom already is; no check is added (Q10).

Cost at np 32 (bottom 16×16×8 = 2048 cells): rhs 16 KB per V-cycle, openness 48 KB per step,
one `Allgatherv` ≈ 20–60 µs per V-cycle; factor ≈ 16·128³ ≈ 3.4e7 FP32 FMA per step on one SM,
redundantly on every GPU in parallel.

### 4.6 Fallback and its visibility (WO-4)

Remaining GraphAMG cases: inadmissible natural-end bottoms (badly factored grids, telescoping OFF),
> 64 fluid components, forced `set_pressure_bottom_solver('algebraic')`, forced `'agglomerated'` on
an anchored operator. On the first bottom solve of a hierarchy that takes `graphAmgSolveBottom` on a
device backend, or with an inadmissible bottom on host, print once (rank 0 of `comm_`, stderr):
`[peclet.flow] pressure multigrid bottom gx×gy×gz (n cells) is solved by host GraphAMG (<reason
from directBottomIneligibleAll()>), rebuilt at every operator change; choose dimensions with more
factors of two (flow.predict_hierarchy) or a smaller set_pressure_bottom_extent` — suppressed when
the user forced `'algebraic'`. Output only; no result depends on it.

### 4.7 Performance expectations (to be checked by the gates, not assumed)

- Extra levels: ≈ 12 launches per level per V-cycle leg × ≈ 4.5 µs; two extra levels × 9 V-cycles
  ≈ 2 ms/step at np 1; at np > 1 each also adds latency-bound halo exchanges (telescope min extent,
  Q3).
- Bottom at the admitted sizes: solve ∝ 2·n·b FMA, factor ∝ n·b². Worst admitted corner (n 8192,
  b 192) ≈ 3e8 FMA factor, 3e6 per solve; near-cubic bottoms (b ≤ 192 ⇒ n ≲ 2660) ≤ 1e8 / 1e6.
  The bubble column bottom (n 1536) is the measured, acceptable reference point on host and device.
- Expected H100 256³ np1: ≈ kernels 96 ms × 9/7.93 + ≈ 5 ms ≈ 115–130 ms/step (from 450.5).

## 5. Register entries

Kept (the design depends on them; unchanged):
- flow.md "Agglomerated/'auto' bottom … gated to the singular (mean-removed) path only" (2026-08-14).
- flow.md "'Hand the stopping level to GraphAMG' telescoping idea is retired" (2026-09-01) —
  reinforced: the bottom is bounded independent of np.
- flow.md "Telescoping is default on for the pressure MG since 2026-09-02".
- flow.md "Device pressure bottom: block-tridiagonal FP32 direct factor …" (2026-10-03), including
  "rejected: a deeper / factor-3 bottom (+8.5 outer iterations)" — the rule never goes past the
  first admitted level. Its distributed clause is superseded **for device** by WO-3 (new entry N2).
- flow.md "Host backends use the 'direct' pressure bottom where eligible" (2026-10-08).
- flow.md "The distributed host 'direct' bottom runs redundantly on every rank" (WO-H6) — extended
  to device by N2, with its rejection of rank-0 + broadcast.
- flow.md "Coupling partial/cut+solid cells to the coarse grid fails regardless of coarsening depth"
  — unrelated (velocity MG); depth is not offered as a fix for it.
- flow.md "GraphAMG bottom solve (on GPU)" rejected (2026-08-09).
- flow.md "S3 coarsening indefiniteness is real, but coarsenOpenAvg must NOT switch to harmonic"
  (2026-08-31) and the "WO-M FINAL" finding that it deepens with depth — the reason the rule stops
  at the FIRST admitted level; this package does not touch `coarsenOpenAvg`.
- suite-wide "The coarsest multigrid level's size is a count of cells per axis" (2026-09-24, USER).

Hygiene: flow.md "Coarse-level solve policy default stays 'smoother'…" (2026-08-11) says `status:
settled` although the 2026-08-14 flip superseded it; WO-2 sets `status: superseded (2026-08-14) by
"Agglomerated/'auto' bottom … singular path only"`.

New (written in the commit that carries each):
- **N1 (WO-2)** "The pressure MG depth is derived: the shallowest D ≥ 4 whose global coarsest grid
  the bottom engine admits". Rejected: fixed 4; per-rank coarsest extent; deepest coarsening; cost
  model; backend-specific depth. Evidence: this note §1, WO-0 log, G-PERF numbers.
- **N2 (WO-3)** "The distributed device 'direct' bottom runs redundantly on every bottom rank,
  gathered through pinned staging". Rejected: host GraphAMG on device MPI; rank-0 + broadcast;
  CUDA-aware-only transport.
- **N3 (WO-2)** "No numeric-only GraphAMG refresh; the per-step rebuild leaves the production path
  with the depth rule". Rejected: numeric-only refresh.
- **N4 (WO-5)** anchored operators follow the grid (or, if G-ANCH fails, "anchored keeps depth 4",
  with the failing numbers).

Docs to update with WO-2/WO-5: `flow/CLAUDE.md` (:229, :267-270, :290-293), `doc/CLAUDE_reference.md`
(:253-278), `doc/anisotropic_metric.md:281`, `../docs/DECOMPOSITION_AND_MULTIGRID.md` §1.1 (:32) and
§3 rule 3, `../docs/SCALING_ISSUES.md` row #2, the comment at `mac_cutcell_mg.hpp:1376-1380` ("cap at
2^(default nLevels − 1)": the cap 16 stays; deeper AUTO levels telescope), CHANGELOG.

## 6. Work orders

Run gates through a one-line-summary script with the log in a file (the `gpu_diag/gate` pattern).

**WO-0 — measurement, no `src/` change.** Study scripts only (`tests/study/mg_depth/`). Explicit
depths via existing knobs (`weak_swarm.py --levels`, `prof.py --levels --bottom`, a small IBM driver
on `sdflow_regression.py`'s geometries with explicit `levels`). Matrix, single rank:
M1 swarm 192³ device L4–L7 (L5 is the missing point); M2 bubble column 256×192×128 device L4–L6
(add a resolution option to the study script if absent); M3 IBM `zh_sphere` and `random_spheres`
N = 128 device L4–L6; M4 channel 2048×64×64 device L4–L6; M5 swarm 128³ host (1×6 threads) L4–L6;
M6 anchored 128×32×32 inflow/outflow channel L4–L6 (divergence floor); M7 the other default
drivers: swarm 192³ with the default (Chebyshev) driver, and a porous bed ≥ 128³ (CFD-DEM-style,
`porous=True`, default driver) L4–L6; M8 the channel 2048×64×64 at np 2 (standalone V-cycle, the
multi-rank default) L4–L6. Per row: bottom dims and
engine, ms/step (A/B/A/B in one session, medians), iterations mean/min/max, FP32 restarts,
`max_open_divergence_projected()`. Write `doc/mg_depth_policy_log.md` (append-only). **Checkpoint:**
evaluate G-PERF-WS (a)–(c) on M1; if (a) fails because L5 loses to L6 by > 8 %, apply Q1's
pre-registered fallback before WO-2. If job 28551969 (H100 L7) has reported, add its row.

**WO-1 — AUTO plumbing, inert.** §4.1 predicate, §4.2 loops (`init`, `initMpi`, `predict`),
`setAgglomerationMode` before init, `setPressureLevels` accepts 0, binding accepts `levels=0`,
`predict_hierarchy(levels=0)`; defaults stay explicit 4 (C++ and Python). New ctests: serial
`mg_depth_policy` (CutcellMG init only, no solve: every row of §4.4 incl. 400³ natural end, the
'smoother'-forced singular case, and `predict(levels=0)` equal to the built hierarchy) and MPI
`mg_depth_policy_mpi` np 1 2 4 + an np 8 rung (`--bind-to none`): resolved D and the per-level
gdim sequence equal single rank's for 192³, 256×192×128, 2048×64×64 with telescoping ON; with
telescoping OFF a blocked case documents D < single-rank. Debug census line (§4.2).
*Bit-identity:* the default path does not change — every existing gate byte-identical (G-BIT).

**WO-2 — default AUTO on the singular path (recorded numerics change).** `nLevels_ = 0`, binding
default `levels=0`, docstrings, docs (§5), register N1 + N3 + hygiene, CHANGELOG ("Changed").
Step 1: run the full flow battery and coupling's battery with the WO-1 census line; list every test
whose AUTO depth ≠ 4 (expected: no `state_hash` case). Step 2: flip; re-baseline exactly the listed
tests with an old → new table (depth, iterations, the test's metric) in the commit message.
*Bit-identity:* byte-identical wherever D_auto = 4 (every unlisted test; all 14 `state_hash` hashes
host and CUDA); partitions unchanged everywhere. Moved: the census list only. Gates: G-BIT,
G-CENSUS, G-CONV, G-ITER, G-PERF-WS, G-PERF-BC, G-PERF-H, G-IBM.

**WO-3 — distributed device direct bottom.** §4.5. Register N2; `flow/CLAUDE.md:267-273` text
("GraphAMG otherwise (… distributed GPU)") updated. Extend `bottom_direct_mpi` to the CUDA build
(np 2 4, ranks may share one GPU). Depends on WO-1 only. *Bit-identity:* host bitwise (data
movement only: `bottom_direct_mpi`, `graphamg_mpi`, `state_hash` `case_mpi` identical); device np1
with `init_mpi` bitwise to single-rank device; device np > 1 recorded change ('algebraic' →
'direct') under G-NUM-D. Gates: G-NUM-D, G-SYNC, G-BIT (host).

**WO-4 — fallback visibility, inert.** §4.6 warning; `setOpenness` comment (:2240-2245) points
here; `predict_hierarchy` docstring. *Bit-identity:* all results byte-identical (stderr only).

**WO-5 — anchored operators follow the grid (recorded).** `kAnchoredFollowsGrid = true` (anchored
admissible iff every axis ≤ e; still smoothed, never agglomerated); `predict_hierarchy(outflow=)`
(Q6). Census as WO-2. *Bit-identity:* anchored cases with D_auto = 4 byte-identical (`case_channel`,
`case_colocated_advect_bc`); moved: the census list. Gates: G-ANCH, G-CONV, G-CENSUS. If G-ANCH
fails, do not land; record N4 as "anchored keeps 4" with the numbers.

**WO-6 — Snellius confirmation (billed; the user schedules it, bundled with other ready packages).**
`weak_swarm.slurm pilot` at np 1 and np 4 on flow main with WO-2 + WO-3, default depth. Gate G-SNEL.

## 7. Gates (thresholds pre-registered; a miss escalates to Opus, never silently re-thresholded)

- **G-BIT.** `ctest -LE bench` host and CUDA green (the whole battery, ≈ 248–253 by the day's
  count; record it); `tests/regression/state_hash.py` host and
  CUDA 14/14 hashes equal to the pre-change baseline; `bottom_direct_mpi`, `graphamg_mpi`,
  `telescope*_mpi`, `decomposition_policy`, `pcg_breakdown` green with identical printed values;
  two runs bitwise.
- **G-CENSUS.** The moved set equals the WO-1 census list exactly; zero `state_hash` cases.
- **G-CONV** (every moved case, at its own settings, plus a rerun at rtol 1e-10 where it is a
  time-dependent case): (a) every outer solve meets its stop test (never the cap); FP32 restarts do
  not increase (0 stays 0); (b) max relative difference over u, v, w, p (and C) at the case's end
  ≤ 100 × rtol, against the same case at explicit depth 4; (c) ~~`max_open_divergence_projected()`
  new/old ≤ 2 on every step~~ — **superseded 2026-10-25 by Amendment A1 below**; (d) the case's physics metric (K, k*, static drop, Hysing rise) within
  1e-6 relative or its own gate tolerance, whichever is looser.
- **G-ITER** (moved cases): mean pressure iterations/step ≤ 1.15 × (explicit depth 4) + 0.5.
  G-CONV and G-ITER must each be exercised on at least one moved case per default driver: MG-PCG
  (M1), Chebyshev (M7, swarm and porous bed), standalone V-cycle (M8).
- **G-PERF-WS** (workstation GPU, swarm 192³ np1, 30 timed steps, interleaved A/B, medians;
  pilot reference L4 558, L6 87.4, L7 90.1 ms; 7.9 / 9.0 / 9.0 iterations): (a) t(AUTO) ≤ 1.08 ×
  min(t(L6), t(L7)); (b) t(AUTO) ≤ 0.20 × t(L4); (c) iterations(AUTO) ≤ iterations(L6);
  (d) `graphAmgSolveBottom` never called (debug counter asserted in the gate script's C++ hook or
  absence in py-spy).
- **G-PERF-BC** (bubble column 256×192×128, device np1): t(AUTO) ≤ 0.8 × t(L4); G-ITER.
- **G-PERF-H** (host, swarm 128³, 1×6): t(AUTO) ≤ 1.03 × t(L4); G-ITER.
- **G-IBM** (`zh_sphere`, `random_spheres`): N = 48 byte-identical (D = 4); N = 128 (4 → 5):
  K / k* within 1e-6 relative of L4; median iterations/step ≤ 1.15 × min(L4, L6) + 0.5; ~~flux
  divergence and~~ deep-solid velocity ≤ 2 × L4's. (The flux-divergence clause is **superseded
  2026-10-25 by Amendment A1 below**; the deep-solid clause stands.)
- **G-NUM-D** (WO-3; device; swarm 64³ per rank at np 2 (128×64×64) and np 4 (128×128×64), and the
  bubble column 128×96×64 at np 2 and 4; 'direct' vs 'algebraic' on the same build): 50 steps at
  rtol 1e-10 max relative difference over u v w p C ≤ 1.499e-11; outer iterations identical every
  step; divergence ratio ≤ 2; inner FCG max ≤ 3, mean ≤ 2.5, shift restarts 0; at the case's own
  rtol iterations identical every step; two runs bitwise; np1 `init_mpi` state SHA == single-rank.
- **G-SYNC** (WO-3; device np 2): per step 0 global fences, 0 View allocations (the
  `device_sync_budget` instrumentation run under MPI); `graphAmgSolveBottom` calls 0 at np 1, 2, 4.
- **G-ANCH** (WO-5; 128×32×32 inflow/outflow channel, the flow.md:199 case): divergence floor ≤ 2 ×
  depth-4's at identical budgets; iterations ≤ depth-4's; anchored `state_hash` cases byte-identical.
- **G-SNEL** (H100, 256³/GPU swarm, 30 timed steps): np1 ≤ 150 ms/step (pilot 450.5); iterations
  ≤ 9.5 (pilot 7.93); np1 single SHA == np1 `init_mpi` SHA; np4 2×2 host-staged: no GraphAMG frames
  in the profile, bottom (kernels + gather) ≤ 3 % of the step; E(4) recorded, not gated (the halo
  pathology is out of scope).

### Amendment A1 (2026-10-25, USER) — the divergence gates judge the projection's contract

Provenance: USER decision of 2026-10-25 ("I go with your recommendation"; corrected version: "Yes, that
is fine"), registered in the umbrella `docs/decisions/flow.md`, entry "A pressure-solver change is judged
on the divergence contract, not on a per-step divergence ratio" (umbrella 7c569d6; index line in
`docs/DECISIONS.md`). Evidence: the WO-2 gate record in `doc/mg_depth_policy_log.md` (25929ec) and the
diagnosis `bubble_column_perf/gpu_diag/mgd/WO2_DIAG.md`.

G-CONV (c) and G-IBM's "flux divergence ≤ 2 × L4's" are REPLACED by two conditions:

1. **At the production tolerance:** every step's post-projection max|div u| ≤ rtol · r0 of that step,
   for both depths. This is the stop test of PCG and Chebyshev (`src/mac_cutcell_mg.hpp:3204` resident
   PCG, `:6240` Chebyshev; `|r|inf < rtol · rref`, rref = r0 = |r0|inf unless a stop reference is set).
2. **At a fixed tight tolerance (rtol 1e-13):** the worst divergence over the run, depth 5 / depth 4, ≤ 2.

Why: the post-projection max divergence IS the final max-norm residual, and a correct solve stops
anywhere in (ρ · rtol, rtol] · r0 (ρ = per-iteration contraction, 0.06–0.2), so two correct depths differ
per step by up to 1/ρ; non-flip depth pairs fail "≤ 2 per step" as often as 5/4 (L6/L5 M3z 9.98, L6/L4
M7a 30/30 steps). A gate a correct change cannot pass is no gate. Rejected: the per-step max-norm ratio
≤ 2; a worst-over-run ratio at the production tolerance (still a stopping-point draw); solving every gate
case to 1e-13 in production. The original wording above is kept, struck through, and the WO-2 MISSes
against it stay on record in the log. G-IBM is judged on the same two conditions. The other gates that
carry a divergence ratio (G-NUM-D, G-ANCH) are not amended by this decision.

## 8. Risks and open questions

Risks:
- **R1 — the E1 penalty on high-contrast problems** whose admitted level is coarser than the
  interface scale (e.g. a swarm at small d/h): the arithmetic coarse coefficients make the V-cycle
  indefinite, more so with depth. The rule minimises it but cannot remove it; S3 (coefficient-aware
  coarsening, a separate design) is the fix. G-ITER / G-PERF-BC / the FP32-restart count detect it;
  a PCG cap at the new depth is a miss, not a re-threshold.
- **R2 — bottom cost at the caps' corner on host** (one thread for the solve, team 6 for the
  factor; n 8192 × b 192 would be ≈ 10× the bubble column's solve). G-PERF-H covers a near-cubic
  case only; an elongated host case is not gated (Q1 covers the response).
- **R3 — latency at np ≫ 32:** tiny-block levels and an `Allgatherv` over many bottom ranks. Not
  measured; Q3.
- **R4 — badly factored grids** keep a (smaller) host GraphAMG bottom; warned, not fixed.
- **R5 — census completeness:** a C++ test building a `Solver` at a large grid with the default
  depth moves; the WO-1 census covers every test the battery runs, coupling's included.
- **R6 — gallery/benchmark results at large grids move** (within tolerance); re-checked at the next
  release's gallery pass (`docs/RELEASE.md`).

Open questions (each with a default; work proceeds on the default):
- **Q1 [fact] — is the engine's admission the right target size?** Default: yes (n ≤ 8192, plane
  ≤ 192). Pre-registered fallback if G-PERF-WS (a) fails on WO-0's M1 because the 12³ bottom costs
  more than it saves: admit only bottoms whose two smallest extents multiply to ≤ 96 (a separate
  `kAutoMaxPlane = 96` used by the predicate only; it keeps the bubble column's 16×12×8 at D = 4
  and sends 192³ to D = 6), then re-run G-PERF-WS. Settled by M1–M5.
- **Q2 [pref] — keep the floor D ≥ 4?** Default: keep (bit-identity of every small case). Dropping
  it would make small grids shallower and move many tests for a marginal gain.
- **Q3 [fact] — telescope min extent at np ≥ 8** (merge earlier to cut tiny-block latency). Default:
  4, unchanged; WO-6 records per-level time to decide in a separate package.
- **Q4 [pref] — bottom-gather transport.** Default: always pinned staging (stack-independent, ≤ 64 KB
  per V-cycle, no core change; this sidesteps the 2×2 CUDA-aware pathology). Alternative: follow
  core's `gpuAwareMpi()` like the halos — it lives in `peclet::core::halo::detail`, so it needs a
  public core accessor and a core release first. Revisit only if the user reads the "no GPU↔host
  transfers" directive as covering MPI staging.
- **Q5 [fact] — raise the plane cap to 256** (would admit 16³: 256³ at D = 5). Default: no; §13
  Q-D5 says measure direct against GraphAMG at b = 256 first.
- **Q6 [pref] — the keyword `outflow=False` on `predict_hierarchy`** (WO-5, a public name).
  Default: that spelling.
- **Q7 [pref] — Snellius spend for WO-6.** Default: one pilot (np 1, 4) at the next billed
  milestone, bundled with other ready packages; the package lands without it, on G-PERF-WS.
- **Q8 [fact] — H100 L7 job 28551969.** Default: add to WO-0's log when it reports; it changes the
  design only if L7 beats L6 by > 5 % on H100 (then revisit Q1 toward smaller bottoms).
- **Q9 [pref] — GraphAMG numeric-only refresh for the fallback.** Default: no (N3).
- **Q10 [fact] — heterogeneous GPUs among bottom ranks.** Default: no check; it affects bitwise
  reproducibility across such a partition only, not convergence (§4.5 item 8).

Open items from the review of 2026-10-25 (`bubble_column_perf/gpu_diag/mgd/REVIEW_MGDEPTH.md`; notes only):
- **Closed-cell pressure differs between depths** (M3: 12.7 % in solid cells, K equal to 15 digits). It is
  not part of the solved system, but `get_p`/VTI show it, and it is the initial guess when a moving scene
  opens a cell.
- **Amendment A1 condition (1) at the production tolerance restates the stop test** (div/|r|inf = 1 within
  1e-4); the checks that bite are G-CONV (b) and condition (2).
- **M7b/M7c (porous) would want `max_porous_residual()`** for the divergence contract, not
  `max_open_divergence_projected()`.
- **Weak scaling: iterations at depth 6/7 are to be recorded on Snellius** (WO-6, coordinator-owned).
