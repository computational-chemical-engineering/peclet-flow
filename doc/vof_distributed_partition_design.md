# flow — distributed host partition and coarse-solve policy on one 24-core NUMA domain

Design note, 2026-10-19. Branch `partition` (worktree `suite/flow-partition`, from flow `be35841`,
core `v1.6.0` = `fcb3b06`). Brief: `bubble_column_perf/dist_cpu/BRIEF_ARCH_PART.md`; diagnosis:
`dist_cpu/IMBAL_REPORT.md`. No production code here. Tags: **[meas]** measured (source given),
**[model]** predicted by the arithmetic stated next to it, **[sim]** computed by an exact Python replica
of the core ORB (`initImpl` + `splitPosition` + flow's factory) in
`bubble_column_perf/dist_cpu/part/orb_sim/` (`cf2.py "GxGxG" np,np,…` prints legacy vs auto; §5.1 is
the rule it implements).

---

## 0. Outcome

> **Amended 2026-10-22 (§12, §13).** Genoa WO-7 (job 28464076) falsified two predictions of this
> section: the B1 per-call cost (0.25–0.40 ms, designed ≤ 0.08) and the 12×2 gain (+6.8 ms, not
> −3…−6). §13 attributes both, corrects one claim of §4(a), and gives the follow-up work orders.
> The tables below are kept as the pre-registered predictions; §13.8 supersedes their numbers.

1. **Two causes are removed by this design. Both are general fixes, not tuned to the case.**
   - **P1, partition.** The default aligned ORB gives 24 ranks a 2.000 cell imbalance on 128×96×64,
     because it always bisects the longest axis and then snaps the cut to 16 cells. The default
     becomes an *auto* policy:
     - keep the legacy partition whenever it is exactly balanced, so those runs stay bit-identical;
     - otherwise use a coarse-first partition built with a new **balance-aware split-axis rule**
       (core), at nesting depth 5, else 4;
     - otherwise use the better of legacy and the balance-aware aligned ORB.

     On the bubble column every layout comes out exactly balanced with full nesting [sim].
     Layouts 2/4/6/8/16/48 keep today's partition bit for bit. Only 3/12/24/32/96 change.
   - **B1, bottom solve.** On host, the replicated `'direct'` bottom is dominated by its
     *preconditioner application*, not by the factor: 7.0 of 9.6 ms/step at 1 thread/rank [meas,
     genoa 24×1]. Each row's dot product is a 128-long dependent FP32 chain. B1 is a host lane
     schedule for that application, run on a single thread, and it is **bitwise**. It helps every
     layout including single-rank, and most at low threads per rank.
2. **Expected genoa ms/step** (cores 0–23, N=200, `ckpt_t43`; now = S-2C job 28302113):

| layout np×T | ranks/CCD | now [meas] | after B1 [model] | after B1+P1 [model] | ≈ 1×24? |
|---|---|---|---|---|---|
| 1×24 single | — | 68.6 | 68.2 (−0.2…−1) | 68.2 | reference |
| m1×24 (np 1 + init_mpi) | — | 66.7 | 66.3 | 66.3 | — |
| 3×8 | 1 | not measured | −0.7 | further −3…−6 (classic max/mean 1.125 → 1.000) | **candidate**; S-PART decides |
| 6×4 | 2 | 80.5 | 78.8 (−1.7) | 78.8 (partition unchanged) | no: +~10 residual, §4(e) |
| 12×2 | 4 | not measured | −3.4 | further −3…−6 (1.125 → 1.000) | no [model 76–82] |
| 24×1 | 8 | 100.8 | 94.4 (−6.4) | **79 ± 4** (−15, report's per-rank arithmetic) | no: halo, VoF spread, factor |
| 8×3 | straddles | 100.4 | 98.2 (−2.2) | 98.2 | **no: hardware** (+17.6 CCD straddle) |
| 4×1×2 (8×3, test dims) | straddles | 98.4 | 96.2 | 96.2 | no: hardware |

3. **What cannot reach ≈ 68 ms, and what users are told.**
   - **8×3 and any layout whose threads per rank do not divide the L3 (CCD) core count cannot.**
     Two ranks straddle two CCDs, and their small, synchronization-bound kernels run 1.4–2.7× slower
     [meas]. No partition fixes that reproducibly. Users are told: threads per rank must divide the
     cores per L3 domain (genoa: 1, 2, 4, 8; 5965WX: 1, 2, 3, 6). The fastest hybrid is **one rank
     per L3 domain** (workstation 4×6: 47.1 vs 1×24 52.3 [meas]).
   - **Balanced, non-straddling layouts (6×4, 12×2, 24×1) keep a ~10 ms distributed residual on
     genoa** after P1 and B1. The residual is halo gather+unpack (5–9 ms/rank), reduction skew,
     the VoF spread, and the bottom factor at few threads. It is not a partition or coarse-solve
     problem. §4(e) specifies the genoa measurement that attributes it, and the next design (if
     wanted) starts from that attribution.
4. **Not changed:**
   - the H-6 replicated `'direct'` bottom (kept);
   - `'auto'` with no thread-count term;
   - MG depth (default 4; bottom 16×12×8 = 1536 cells);
   - the alignment cap (16);
   - telescoping;
   - the weighted ORB (`rebalance_by_weights`, CFD-DEM);
   - core's default split rule for every other consumer;
   - the device path's numerics.

## 1. Problem and scope

USER goal (2026-10-17): "24 cores solve equally fast almost independent of the distribution". On
one genoa NUMA domain (EPYC 9654, NPS4: cores 0–23 = 3 CCDs × 8 cores, 32 MB L3 each), flow's
distributed bubble column (128×96×64; x vertical periodic, y walls, z periodic; 16 bubbles, D/h = 16;
variable ρ; MG-PCG rtol 1e-8; 4 MG levels; agglomerated `'direct'` bottom 16×12×8) runs 66.7–100.8
ms/step depending on the layout.

**In scope:**
- (a) the default level-0 partition and its level-count rule;
- (b) the replicated bottom at low threads per rank;
- (c) VoF-cost weights;
- (d) CCD/L3 advice;
- (e) what to measure for the genoa-only 6×4 gap;
- the genoa confirmation run.

**Out of scope:**
- device multi-GPU transport (SURF ticket);
- AMR;
- velocity-MG internals;
- the gallery;
- halo pack/unpack kernel efficiency (the comm design's domain: `doc/vof_distributed_host_comm_design.md`; flagged as the next lever in §10);
- the host factor's flop efficiency (`doc/vof_step_performance_design.md` §13–14; flagged in §10).

## 2. Facts this design rests on

| # | fact | source |
|---|---|---|
| F1 | Legacy ORB (`initImpl`, core `block_decomposer.hpp:432`): split the longest *physical* axis; `n1 = ⌊n/2⌋`; cut = round(S·n1/n); then snap to `align` (flow `coarsenAlignment`, capped at 16), and skip the snap if the box is narrower than 2·align | code |
| F2 | 24 ranks on 128×96×64 [sim, matches IMBAL]: a 3-rank node (32,48,64) cuts z at round(64/3)=21 → snapped to 16. Its 2-rank sibling (32,48,48) cuts y at 24 → snapped to 32. That gives 8 blocks of 32×32×48 = 49152 and 16 of 24576, imbalance 2.000. The same mechanism gives 12 ranks 1.500 (max/mean 1.125), 3 ranks 1.200 (1.125), 32 ranks 2.000, 96 ranks 4.000 | sim; IMBAL Q1 |
| F3 | Coarse-first search (`CutcellMG::decomposition`, `mac_cutcell_mg.hpp:1415`): for L = levels…2, build the ORB on the grid coarsened L−1 times and refine; take the deepest L with max/min ≤ budget (1.05). At 12 ranks only L = 2 fits the budget with the longest-axis rule. Telescoping then puts coarse levels and the bottom on ⅓ of the ranks: local 12×2 48.9 → 52.3 | code; IMBAL Q3.2 |
| F4 | Coarse-first L5 locally at 24×1: 59.4 → 53.4 (imbalance 1.048), iterations equal (10.05). Genoa per-rank arithmetic: heavy "rest" 54.0 → mean 39.2, ≈ −15 ms | IMBAL Q3.2 |
| F5 | Genoa 24×1 trace: `mg_bottom_factor` 2.63 ms/step (1 call); `mg_bottom_direct` 6.98 ms/step (10 calls = 0.70 ms per V-cycle) on every rank. 8×3, non-straddling rank: 1.45 + 2.83 (0.28 per call) | `imbal/k24x1.csv`, `k8x3.csv` |
| F6 | Bottom planes for the case: slow axis = the only wall axis y → P = 12 planes, b = 16·8 = 128, no border (`bottomChoosePlanes`). M application (`BottomDirect::apply`, `mg_bottom_direct.hpp:218`): per plane, `TeamThreadRange` over rows i, each row a serial `acc += Q(k,j,i)*v_j` over j = 0…b−1. That is 23 plane mat-vecs per M, each row a dependent chain of 128 FP32 FMAs: ≈ 65 k cycles/plane at 1 thread, ≈ 0.4 ms per M. The inner FCG takes ≤ 3 iterations (G-NUM-H #4), which matches 0.70 ms per solve | code; arithmetic |
| F7 | Host builds compile with `-ffp-contract=off` (`CMakeLists.txt:129`), and `mg_bottom_direct.hpp` already has `hostAccumulate`: a lane-vectorized `acc[i] += x_m[i]*y_m` over m in order, bitwise to the serial loop | code |
| F8 | 8×3 genoa: ranks 2 and 5 straddle a CCD, +17.6 ms; halo unpack 2.7×, `bc_vel` 2.4×, streaming kernels +5–8 %. The workstation reproduces it with roles swapped (6×4 on 6-core CCDs: 67.6 vs 4×6 47.1) | IMBAL Q2 |
| F9 | VoF block work at 24×1 genoa: 8–10.5 ms on ranks 0–15, 1.5–3.0 on 16–23 (max 10.5 vs mean 6.4) | IMBAL Q1 |
| F10 | Velocity MG uses the solver's decomposition in place, without telescoping (`mac_velocity_mg.hpp:447`), so it needs ≥ 3 in-place halvings for `vmgLevels = 4` | code |
| F11 | Every level-0 decomposition in flow goes through `CutcellMG::decomposition`: `flow_ibm_mpi.hpp:23`, `flow_bindings.cpp:3842` (`mpi_block`), `mac_cutcell_mg.hpp:1495, 6803` (`predict_hierarchy`). `scripts/check_decomposition.py` calls the bindings | grep |
| F12 | CFD-DEM: a moving multi-rank run co-decomposes through the *weighted* ORB (`rebalance_by_weights` + dem `migrate_to_weights`). A fixed bed adopts flow's init partition through `block_origin()` | `coupling/python/peclet_coupling/driver.py:351` |
| F13 | Register invariant: "the ORB must never split the wall-normal axis (y) in wall-bounded flow". It comes from the channel, where the driver hard-aborts. The factory is BC-agnostic and does not enforce it. The legacy default already splits the bubble column's wall axis y at np 6, 8, 12 and 24, and the 8×3 bench layout (4×2×1) splits y. Those runs pass G-NUM-H against single-rank (H-6: 2.3–3.6e-14, iterations identical), and the campaign recorded "no rule change" (DECISIONS_DISTCPU.log, Q-7). The y-unsplit 4×1×2 is not slower (98.4 vs 100.4) | `docs/decisions/flow.md:1273`; [sim]; S-2C |

## 3. Constraints and invariants

- x-fastest indexing; per-axis factor-of-two coarsening (`can(d) = d even && d/2 ≥ 2`); telescoping on.
- **Serial ↔ np 1 bitwise** (one block: every policy returns the whole grid).
- **np > 1:** results may move only within **G-NUM-H** (§7), and only for configurations whose
  partition changes. Configurations whose partition does not change stay **G-BIT** at the same np.
  Every re-baselined hash is listed in the commit.
- **Pure function, no communication:** the partition is a function of (np, grid, levels,
  max_imbalance). Every rank and every call site (`mpi_block`, `IbmSolver::initMpi`,
  `CutcellMG::initMpi`, `predict_hierarchy`) computes the identical answer.
- **No environment variable changes a result.** The policy is selected by the existing setter
  values only.
- MPI collectives rank-consistent; GridHalo init/destroy collective (core 1.6.0). Nothing here
  changes that call pattern.
- **Device:** same factory, so multi-GPU partitions change exactly where host partitions do. B1 is
  host-only and the device bottom kernels are untouched.
- **core consumers** (dem, voro, amr, coupling, pnm) see no change: core's default rule stays.
- **Wall-normal invariant (F13): the auto policy never splits an axis that the legacy partition
  at the same (np, grid) leaves whole.** A channel whose legacy partition keeps y whole keeps it
  whole. The status quo for the bubble column, whose legacy partition already splits y, is
  unchanged in kind. Whether the invariant should also bind the bubble column is Q-W (§10). This
  design neither relaxes nor extends it.
- **No new public Python name** (NAMING canon, QUALITY_PLAN D2).

## 4. Decisions

### (a) Default partition: the *auto* policy, one rule for every layout

**Decision.** `levels = 0` (the default of `Solver.set_decomposition` and `flow.mpi_block`) becomes
*auto*:
1. **Legacy if exact.** Build the legacy aligned ORB A. If max cells == min cells, return A. This
   proves bitwise identity for every currently-balanced run by construction, with no census
   argument.
2. **Else coarse-first, balance-aware, depth 5 then 4.** For L = 5, 4, build the coarse-first
   partition C with the new core split rule `SplitAxis::Balanced` (§5.1). Return the first one whose
   max/min ≤ `max_imbalance` (default 1.05) and whose nestDepth(C) ≥ nestDepth(A): a coarse-first
   candidate that nests less deep than legacy is rejected. nestDepth counts the minimum-extent rule
   (§5.2): a C whose blocks are 8 cells wide nests 2 levels. Amended 2026-10-19 (the original rule
   checked depth only for A′); in the census this rejects C at 40³ np 5, 48×32×40 np 8,
   48×40×30 np 5 (depth 2 < 3) and 96³ np 7 (3 < 4), and on the channel grids 480×80×160 np 4, 8,
   22, 24, 32 and 768×128×256 np 13, 20, 28, 30, 31 (3 < 4; these fall back to A or A′).
   - L = 5 is the depth the alignment cap 16 already encodes.
   - L = 4 keeps 3 in-place halvings, which the default 4-level pressure MG and the velocity MG
     need (F10).
3. **Else the better aligned ORB.** Build A′ = the aligned ORB (same `coarsenAlignment`) with
   `SplitAxis::Balanced`. Return A′ iff maxCells(A′) < maxCells(A) and
   nestDepth(A′) ≥ min(nestDepth(A), 3). Otherwise return A.

Every candidate in steps 2–3 must also satisfy **splitAxes(C) ⊆ splitAxes(A)**, where axis k is
split iff some block has size[k] < G[k]. A candidate that would cut an axis the legacy partition
leaves whole is skipped (F13).

`levels = 1` returns **A** exactly: today's default, kept for reproducing pre-change np > 1
partitions. `levels ≥ 2` = the legacy coarse-first search (L…2, `SplitAxis::Longest`, legacy
fallback), unchanged: the pre-partition partitions bit for bit (review 2026-10-20 #1, coordinator).

Results [sim]:
- 128×96×64: np 3, 12, 24, 32, 96 → exact (max/min 1.000), ≥ 4 in-place halvings. Every other
  np ≤ 192 tested is unchanged.
- 48³ at np 2: 2.000 (legacy 32 | 16) → 1.000 (24 | 24), the coupling's documented case; legacy
  does not balance 48³ at np 2. At np 4: 4.000 → 1.000 at L4. Tests that rely on the legacy
  32 | 16 cut pin `levels = 1` (WO-3).
- 64³ at np 12: 1.500 → stays A (no balanced option within budget at L5/L4).

**Rejected:**
- **Legacy as is.** 2.000 at 24×1 costs ≈ 15 ms on genoa (F4). It is also 1.5 at 12 ranks and
  1.2 at 3, the layouts the CCD advice points users to.
- **Plain coarse-first as default (longest-axis rule, search to L2).** It trades depth for balance
  below the MG depth. That is the measured local 12×2 regression (+3.4 ms, F3).
- **A Cartesian process grid** (px·py·pz = P, MPI_Dims_create style). It is not an ORB tree, so
  it breaks telescoping (sibling merge = tree truncation), the weighted path and `refined()`. It
  also fails for prime counts. The balance-aware ORB reproduces a Cartesian factorization whenever
  one fits the quanta. **[False — corrected in §13.3(e): at np 12 and 24 on 128×96×64 it gives
  mixed shapes, a slab plus two columns per 3-group.]**
- **Changing core's default rule.** dem/voro/amr/pnm partitions would move for no measured
  benefit.
- **Deriving the alignment cap from the MG depth** (DECOMP §4 item 2). It is not needed for the
  bubble column (exact at cap 16), and it would be a second change of the same partition in the
  same package.
- **A level count per layout.** One rule fits all layouts.

### (b) The replicated bottom at low threads per rank: keep H-6, make the solve cheap (B1)

**Decision.**
- Keep H-6 (every rank factors and solves redundantly).
- Add **B1**: on host backends the per-V-cycle bottom solve runs on **one thread**, and the M
  application uses a **lane schedule** (rows as SIMD lanes, j ascending), bitwise to today (§5.3).
- Expected [model]: per-solve 0.70 → ≤ 0.08 ms at 1 thread, and ≤ 0.08 ms at any T. Step: −6.4
  (24×1), −3.4 (12×2), −2.2 (8×3), −1.7 (6×4), −0.7 (3×8), −0.2…−1 (1×24) on genoa.
- The factor (2.63 ms at 1 thread, ~6 % of FP32 peak) stays as is.
- Size caps (8192 cells / 192-cell plane) unchanged. MG depth unchanged.

**Rejected:**
- **Agglomerating onto a subset of ranks.** A subset rank has the same T, so the critical path
  is unchanged.
- **Rank-0 solve plus broadcast.** H-6's reason (a latency step per V-cycle; the other ranks
  idle anyway). After B1 the solve is ~50 µs, so there is nothing to save.
- **A cooperative on-node factor across ranks over a shared-memory window.** It would be bitwise
  (each factor scalar is computed by one lane in a fixed order) and would gain ≈ 2 ms at 24×1
  only. The cost is shared windows, ~600 node barriers per factor, and failure semantics. It is
  deferred as F-B2 (§10).
- **A deeper default hierarchy** (5 levels → bottom 8×6×4 = 192 cells, factor ≈ 0). It changes
  serial numerics for every flow user. It is offered as a case setting after experiment E1
  (WO-6), never as a default here.
- **A thread-count rule choosing `'algebraic'`.** Register 2026-10-18 (results would depend on
  `OMP_NUM_THREADS`). B1 also removes the reason for it: the 1-thread near-cap edge was the M
  application.

### (c) VoF-cost weights: none in the default; dynamic VoF rebalancing deferred

**Decision.** The default partition balances cells. No static or dynamic VoF weights now.
- Payoff: ≤ 4 ms at 24×1 (F9: max 10.5 vs mean 6.4), and ≤ 1–2 ms at ≥ 3 threads/rank.
- Any weight shifts MG cells off exact balance, which costs back part of the gain.
- Rebalancing cost is not yet known: re-building the hierarchy, halos, telescope, bottom store and
  VoF block structures, plus migrating the VoF state through `redistribute`, whose completeness
  for VoF state is unproven.
- Trigger to revisit (F-C, §10): VoF max − mean ≥ 3 ms at a recommended layout (3×8 / 6×4 /
  12×2) in S-PART.

**Rejected:**
- **Static weights from the initial state.** Bubbles rise one block height in O(100) steps, so
  the initial state has no special status, and it breaks exact cell balance permanently.
- **A column (y–z only) partition.** 16 bubbles over 24 columns is a worse VoF imbalance and gives
  +40 % halo surface.

### (d) CCD/L3: documentation plus a one-line init-time warning; no placement-aware partition

**Decision.**
- **WO-4:** document the rule "threads per rank divide the cores per L3 domain; prefer one rank
  per L3 domain", with launch recipes.
- **WO-5:** at `init_mpi` with np > 1, when every OpenMP thread is pinned to one CPU, rank 0
  prints one stderr line naming the ranks whose threads span more than one L3 domain. No API, no
  result effect. A one-rank job stays silent: it has no better layout to move to (ruling
  2026-10-19).

**Rejected:**
- **Speed- or placement-weighted partitions** (give straddling ranks fewer cells). Results would
  then depend on where the scheduler put the ranks, which breaks reproducibility in the spirit of
  QUALITY_PLAN D3.
- **Rank→block remapping.** Blocks are equal, so the straddling rank is slow whichever block it
  owns.
- **Fixing the straddle in code.** It is cross-CCD cache-line and barrier latency inside one
  rank's thread team (F8).

### (e) The genoa-only 6×4 gap: measure, do not design

**Decision.**
- The reference for 6×4 is **m1×24** (same distributed code path, 66.7 ms), not 1×24.
- S-PART (§8) traces m1×24, 3×8, 6×4 and 24×1 with `kp_trace` + `pmpi_trace`, and runs two
  microbenchmarks.
- Pre-registered hypotheses, their expected share of the 13.8 ms (6×4 − m1×24) [model], and how
  each is attributed:

| # | hypothesis | expected | attribution |
|---|---|---|---|
| e1 | halo gather + unpack: per-thread ghost volume ≈ 1.8× m1×24's self-copy | +4…+7 | per-kernel diff |
| e2 | bottom at 4 threads | +2 | removed by B1, so S-PART shows the residual without it |
| e3 | reduction latency + skew | +2…+4 | pmpi `MPI_Allreduce` time and count per rank; microbenchmark μ1 |
| e4 | VoF spread | +1…+2 | `vof::*` max − mean |
| e5 | fork-join | ≤ 0 (4 threads in one CCD is cheaper than 24 across 3) | microbenchmark μ2 |

- **Falsifiable statement:** e1 + e3 + e4 explain ≥ 80 % of the post-B1 gap. If they do, the gap is
  ordinary distributed overhead, visible on genoa because genoa's 1×24 is efficient. If they do
  not, the unattributed residual (step wall − Σkernel − ΣMPI on the critical rank) is the next
  thing to explain, and that goes back to the caller with numbers, not to a design.

## 5. Design detail

### 5.1 core: `SplitAxis::Balanced` (opt-in)

**API** (core `block_decomposer.hpp`; minor version 1.7.0):
- `enum class SplitAxis { Longest, Balanced };`
- member `splitAxis_ = SplitAxis::Longest`;
- a new overload `init(numBlocks, globalSize, align, cellExtent, SplitAxis rule)` that sets
  `splitAxis_`, then follows the existing 4-argument path.

Every existing overload, `refined()`, `coarsened()`, `chooseStageTarget` / Repartition and the
weighted inits keep `Longest`. The rule acts only at construction (`initImpl`); the stored tree is
an ordinary ORB tree. Telescoping and sibling merges are unaffected.

**Rule at a node** (box size S[0..D), count m ≥ 2, n1 = ⌊m/2⌋; unweighted only):
1. **Candidate cut** c_k on each axis k with S[k] ≥ 2, computed by today's expression:
   - c = round_half_away(S[k]·n1/m), clamped as today;
   - if align[k] > 1 and S[k] ≥ 2·align[k]: snap to the nearest multiple of align[k] within
     [align[k], S[k] − align[k]], exactly as today.
2. **Excluded axes:** an axis with align[k] > 1 and S[k] < 2·align[k] is excluded from the
   alternatives. An unaligned cut there would cost nesting depth: in [sim] the unrestricted rule
   took 64³ np 48 to 0 halvings.
3. **Deviation:** D_k = |c_k·m − S[k]·n1| (int64). The relative deviation of the left child's cell
   fraction is D_k/(S[k]·m). Compare axes i, j exactly by D_i·S[j] vs D_j·S[i].
4. **Choice:**
   - kL = the legacy axis (largest S[k]·cellExtent[k], ties to the lower index).
   - If kL has the minimum relative deviation (in particular, if D_kL = 0), choose kL.
   - Else choose the minimum-deviation axis; ties go to the larger physical extent, then the
     lower index.
   - If no axis qualifies, choose kL.
5. Split as today with the chosen axis and its c_k.

**Properties** (each a core unit test, §7):
- **(P-i)** Legacy exact ⇒ every node's legacy cut is exact (a balanced final partition forces
  every left subtree to hold n1/m of its box) ⇒ the rule picks kL at every node ⇒ an identical
  tree.
- **(P-ii)** Pure and deterministic.
- **(P-iii)** No unaligned cut is introduced where the legacy rule had an aligned one.

The coarse-first call has align = 1 and cellExtent = r, so step 2 never excludes anything there.
`n1` stays ⌊m/2⌋. Rank counts with two odd prime factors, on grids that need them (e.g. 9 on
3·2^k × 3·2^k), stay inexact; this is a known limitation (§10).

### 5.2 flow: the factory

`CutcellMG::decomposition(numBlocks, gnx, gny, gnz, levels = 0, maxImbalance = 1.05)`:

```
if numBlocks == 1:            return whole grid                    # unchanged
A = BlockDecomposer(numBlocks, G, coarsenAlignment(G))             # legacy, today's levels=0
if levels == 1:               return A
if levels >= 2:               return coarseFirstSearch(L = levels..2, Longest) else A   # legacy, unchanged
# levels == 0: auto
if maxCells(A) == minCells(A): return A                            # bitwise-unchanged branch
for L in (5, 4):
    r = refineFactor(G, L); if r == (1,1,1): break
    if cells(G / r) < numBlocks: continue
    C = BlockDecomposer.init(numBlocks, G/r, align 1, cellExtent r, Balanced).refined(r)
    if minCells(C) > 0 and maxCells(C)/minCells(C) <= maxImbalance
       and splitAxes(C) ⊆ splitAxes(A)
       and nestDepth(C) >= nestDepth(A): return C
A2 = BlockDecomposer.init(numBlocks, G, coarsenAlignment(G), cellExtent 1, Balanced)
if maxCells(A2) < maxCells(A) and nestDepth(A2) >= min(nestDepth(A), 3)
   and splitAxes(A2) ⊆ splitAxes(A): return A2
return A
```

- **`coarseFirstSearch`** is today's loop (`mac_cutcell_mg.hpp:1440–1462`) with
  `init(…, {1,1,1}, r, SplitAxis::Balanced)`.
- **`nestDepth(dec, G)`** is the number of consecutive in-place coarsenings from level 0, up to 4,
  before the first telescope trigger. At each step:
  - every axis that can(g_k) coarsen must have every block origin and size even;
  - `minBlockExtent ≥ 4` (the `teleMinExtent_` default), checked on the current level before each
    coarsening: blocks 8 cells wide give depth 2 (8 → 4 → 2, stop).

  The count includes the minimum-extent rule (ruling 2026-10-19), for every candidate (C5, C4
  against nestDepth(A), A2 against min(nestDepth(A), 3)) and in the census; the [sim] replica
  (`dist_cpu/part/orb_sim/`) counts it the same way.
  Reuse `teleLiftable` / `cdec::minBlockExtent` rather than re-deriving them.
- **Debug print:** `mgDebugLevel()` prints which branch fired (A-exact / C5 / C4 / A2 / A-fallback)
  and the max/min.
- **Bindings:** update the docstrings of `set_decomposition` and `mpi_block`
  (`flow_bindings.cpp:2381`, `:3834`) and the flow CLAUDE.md "Decomposition" bullet:
  - 0 = auto (default);
  - 1 = the legacy aligned ORB (pre-2026-10 default, for reproducing old np > 1 partitions);
  - ≥ 2 = coarse-first search.

  `set_decomposition` and `mpi_block` must still receive the same values.
- **Check:** confirm F11 (`grep -n "BlockDecomposer<3>(" src/`). The only other construction is
  the velocity MG's standalone `initMpi(gnx,…)` test path, which keeps its plain ORB.

### 5.3 B1: host bottom-solve schedule (bitwise)

**Scope.** `BottomDirect::apply` and the bottom FCG launch (`'direct'`, host backends only; the
device keeps the team code).

1. **Single thread.** On a host backend, the per-V-cycle bottom solve launch (the
   `mg_bottom_direct` kernel and anything else that runs per V-cycle on `gb_`) runs with team
   size 1.
   - The FCG's host reductions are already single-lane (H-1), and every other operation is
     elementwise, so results do not depend on the team size.
   - At T ≥ 8 the 23 per-plane team barriers (cross-CCD at T = 24) would cost more than the whole
     single-thread M (≈ 5–8 µs).
2. **Lane M.** For each plane mat-vec, forward (k = 0…Pt−1), backward (k = Pt−2…0), and the
   border's two loops:

   ```
   v[j] = (forward, k>0) u(o+j) - e(k,j)*g(op+j)  |  (k==0) u(o+j)
          (backward)     e(k+1,j)*g(on+j)          |  (border) (u(oB+j)-e(0,j)*g(j)) - e(P-1,j)*g(oL+j)
   acc[0..b) = 0
   for j = 0..b-1:  acc[i] += Q(k,j,i) * v[j]   for all i, as a SIMD lane loop over i
   forward: g(o+i) = acc[i];  backward: g(o+i) = g(o+i) - acc[i]
   ```

   - Each acc[i] sees the same operands in the same order as the team code's serial
     `acc += Q(k,j,i)*(…)`.
   - v[j] is formed with the same expression, hence the same two roundings under
     `-ffp-contract=off`, so the result is bitwise.
   - Use `hostAccumulate` (generalize its y-stride) or an equivalent loop. Keep ≥ 8 row chunks per
     j in flight, which `hostAccumulate`'s 4-way m-unroll × the lane loop over i already gives.
   - Q(k,j,·) must be unit-stride in i. If the View's layout is not, copy Q once per factor into a
     host scratch with i fastest (a bitwise copy) — never read Q(k,i,j) instead.
   - The Y loop (border) is identical with Y(k,j,i).
3. **Unchanged:** the cast in and out, `keep`, and the null-space augmentation.

**Cost [model].** M = 23 × 128 × 8 vector FMAs over 786 KB of L2-resident FP32 Q, about 5–8 µs.
The FCG at ≤ 3 iterations with the 1536-cell operator and single-lane dots comes to ≈ 50–60 µs per
solve, against 700 µs today at T = 1.

### 5.4 Placement warning (WO-5) and documentation (WO-4)

**Warning.** In `init_mpi` (host OpenMP backend, Linux):
1. In one host parallel region, each thread reads `sched_getaffinity(0, …)` (the calling thread's
   mask).
2. If every mask holds exactly one CPU, map each CPU to `/sys/devices/system/cpu/cpu<N>/cache/index3/id`.
3. Rank status:
   - 0 = unknown (any thread unpinned, sysfs unreadable, non-Linux);
   - 1 = one L3;
   - 2 = more than one L3.
4. `MPI_Gather` one int to rank 0. All ranks call it, inside the already-collective `init_mpi`.
5. If np > 1 and any status is 2, rank 0 prints once (np = 1 stays silent):

   `peclet.flow: ranks 2,5 run OpenMP threads on more than one L3 cache domain (measured up to
   +18 % step time); use threads per rank dividing the cores per L3 domain (AMD EPYC Zen 4: 8),
   ideally one rank per domain.`

   List at most 8 ranks.

No API, no flag, no result effect.

**Documentation:**
- flow CLAUDE.md (MPI/decomposition bullets);
- the docstrings of `init_mpi` / `mpi_block`;
- `docs/DECOMPOSITION_AND_MULTIGRID.md`:
  - §3: new rule 9, "threads per rank divide the L3 domain; prefer one rank per L3 domain",
    with the 8×3 / 6×4 numbers;
  - §4 item 13: "auto policy since …, legacy = levels 1";
  - items 2/3: partially addressed by `Balanced`; the cap is unchanged;
- `docs/SNELLIUS.md`: a genoa recipe for 3×8 and 6×4 inside one NUMA domain (`--cpus-per-task=T
  --cpu-bind=cores`, `OMP_PLACES=cores OMP_PROC_BIND=close`). Verify it against the file's sbatch
  conventions; ctests keep `--bind-to none`.
- Genoa numbers enter only after S-PART.

### 5.5 What does not change

- Telescoping and its stage machinery (core's);
- `rebalance_by_weights` and `chooseAlignedWeighted`;
- the weighted coarse-first init;
- the H-6 gather;
- `'auto'` bottom selection;
- the device bottom;
- the velocity MG;
- VoF;
- all communication code.

## 6. Work orders (dependency order)

Classes:
- **G-BIT**: bitwise `state_hash` (§7);
- **G-NUM-H**: §7;
- **NONE**: no numerics;
- **MEAS**: measurement only.

Local perf = workstation, `dist_cpu/imbal/run.sh` + `bench_lv.py`, N=200, 2 reps, median, load ≤ 2,
bound threads. Every Δ is against the WO-0 baseline.

| WO | content | class | gate (acceptance) | expected |
|---|---|---|---|---|
| **WO-0** | Baselines on flow `be35841`. Local: `state_hash` (50 steps, rtol 1e-8 and 1e-10) at 1×24, m1×24, np 2/3/4/6/8/12/24. Local ms/step at 1×24, m1×24, 4×6, 8×3, 6×4, 12×2, 24×1. A `kp_trace` of 1×24 and 24×1 (bottom per-call times) | MEAS | files under `dist_cpu/part/base/`, one summary line | — |
| **WO-1** | B1 (§5.3) on flow `partition`; independent of core; lands first | G-BIT | (i) new `bottom_direct` case U9: host lane M == team M bitwise for random r on three bottoms (the case's 16×12×8 y-planes; a periodic slow axis with border, P ≥ 3; a b = 192 plane) and both shifts paths. (ii) `ctest -LE bench` host and CUDA green with no hash change. (iii) bubble column `state_hash` == WO-0 at 1×24, m1×24, np 2/4/8. (iv) per-call `mg_bottom_direct` at T=1 ≤ 0.10 ms. (v) local Δ: 24×1 ≤ −3.0, 12×2 ≤ −1.2, 8×3 ≤ −0.6, 4×6 and 1×24 ≤ +0.3 | local: 24×1 −4.2, 12×2 −2.2, 8×3 −1.5, 6×4 −1.1, 4×6 −0.7, 1×24 −0.2; genoa §0 |
| **WO-2** | core `SplitAxis::Balanced` (§5.1) on core branch `partition` (worktree `suite/core-partition`); unit tests P-i…P-iii + exactness | G-BIT (core: every existing call byte-identical) | core ctests np 1–8 green. New tests: (P-i) over the census set (§7) Balanced == Longest (origins, sizes, tree split positions) whenever Longest is exact; exactness on 128×96×64, align 16, np 3/12/24/32/96; nesting: `coarsened({2,2,2})` four times valid for those | — |
| **WO-3** | flow factory auto policy (§5.2), bindings docstrings, CLAUDE.md line; PecletDeps on the core `partition` commit for local builds (sibling headers) | np 1 G-BIT; unchanged partitions G-BIT; changed G-NUM-H | (i) new host ctest `decomposition_policy` (§7). (ii) G-BIT at np 2/4/6/8 vs WO-0. (iii) G-NUM-H at np 3 (T=8), 12 (T=2), 24 (T=1) vs single-rank. (iv) `ctest -LE bench` host and CUDA; every changed hash is in the census's changed list, old → new table in the commit. A ctest whose purpose is legacy-ORB behaviour pins `levels=1`; all others are re-baselined. (v) coupling MPI tests green. (vi) local Δ (B1 already in): 24×1 ≤ −4.5, 12×2 ≤ −1.0, unchanged layouts within ±1 % | local 24×1 −6, 12×2 −2.8; genoa 24×1 −15, 12×2/3×8 −3…−6 |
| **WO-4** | docs (§5.4) | NONE | review by the coordinator | — |
| **WO-5** | placement warning (§5.4) | NONE | warning printed for local 6×4 (straddles 6-core CCDs) and silent for 4×6, 12×2, 1×24, a one-rank `init_mpi` job (m1×24) and for `--bind-to none` without OpenMP binding; `ctest -LE bench` green | — |
| **WO-6** | E1 depth experiment on the WO-1+WO-3 build, local: bubble column at `set_pressure_multigrid(True, 5)` vs 4, at 1×24 and 24×1. Report ms/step, mean/max PCG iterations, bottom ms | MEAS | numbers in the campaign log | 5 levels: bottom 8×6×4 = 192 cells (P = 6, b = 32), factor ≈ 0; win iff iterations ≤ 4-level + 0.1 |
| **WO-7** | S-PART genoa run (§8), user OK | MEAS | §8 falsification criteria | §0 table |

Landing:
- WO-1 lands on flow main on its own (bitwise; no review needed beyond the G-BIT proof).
- WO-2 and WO-3 need a **core v1.7.0 tag before flow main** (USER directive; outward-facing, so
  user OK).
- WO-3 is numerics-class (np > 1) and changes a default, so it gets a **reviewer pass** before
  landing.
- Umbrella last: pointers plus the register entries (§11).

## 7. Verification gates

**G-BIT.** `state_hash` of the bubble column from `ckpt_t43`, 50 steps, rtol 1e-8 and 1e-10, equal
to WO-0 at the same layout. Host battery (`ctest -LE bench`, 252 registered) green with unchanged
hashes.

**G-NUM-H** (`doc/vof_step_performance_design.md` §14.6 items 1, 2, 3, 5, 9; scripts
`dist_cpu/gate/h6num.sh` / `numcmp.py` pattern). Candidate = the new partition at np P; reference =
single-rank `'direct'` on the same build.

| # | check | threshold | expected |
|---|---|---|---|
| 1 | rtol 1e-10, 50 steps: max rel. diff over u v w p C | ≤ 1.499e-11 | ≤ 1e-13 (H-6: 2.3–3.6e-14) |
| 2 | iterations | identical every step | — |
| 3 | projected divergence ratio | ≤ 2 | — |
| 5 | rtol 1e-8, 50 steps: iterations | identical every step | — |
| 9 | two runs | bitwise | — |

Run at np 3, 12, 24. A failure of #2 or #5 means the gathered bottom openness or the hierarchy is
not partition-independent. **Stop and report; do not loosen.**

**Census ctest `decomposition_policy`** (host, no MPI, default label).
- Census set:
  - grids {128×96×64, 48³, 64³, 96³, 128³, 192³, 256×128×128, 384×288×192, 40³, 100×50×50};
  - np ∈ {1…8, 12, 16, 24, 32, 48, 96};
  - plus every (grid, np) used by flow's and coupling's MPI ctests (grep `mpi_block` / `init_mpi`
    in `tests/`).
- For each configuration, compare auto (levels 0) with legacy (levels 1). Assert:
  1. legacy exact ⇒ auto ≡ legacy (origins, sizes, split positions);
  2. maxCells(auto) ≤ maxCells(legacy);
  3. nestDepth(auto) ≥ min(nestDepth(legacy), 3);
  4. 128×96×64 at np 3/12/24/32/96: max == min;
  5. 48³ at np 4: max == min;
  6. two calls identical;
  7. splitAxes(auto) ⊆ splitAxes(legacy);
  8. a channel-shaped grid whose legacy partition keeps y whole keeps y whole: 768×128×256 and
     480×80×160 at np 2…32.
- Print the changed list. That list is the hash re-baseline list.

**Batteries.**
- flow host and CUDA `ctest -LE bench`;
- core np 1–8;
- coupling MPI (host);
- dem unchanged (no dem code touched; skip unless the coupling tests fail).

**Perf (local).** WO-1 / WO-3 rows of §6. All Δ use the WO-0 baseline, N=200, 2 reps, median,
load ≤ 2. Noise band ±1 %.

**Limiting cases.**
- np 1: auto == legacy == whole grid.
- np = number of quanta (128×96×64 at np 192): one 16³ block per rank, exact, 4 halvings [sim].
- Grids without factors of two (e.g. 30³, 100×50×50): auto falls back to A or an A2 that is no
  worse.

## 8. S-PART: the genoa confirmation run (one billed job, user OK, ≤ 60 SBU)

**Harness.** Copy `dist_cpu/s2c/` to `dist_cpu/spart/`. Keep `bench_peclet.py`, `tabulate_s2c.py`,
`l3map.py` (CCD audit column) and the per-layout table in `s2c.slurm`. Add 3×8 and 12x2:
`np cpus-per-task threads bind cpu-bind` = `3 8 8 close cores` and `12 2 2 close cores`. Use
cores 0–23 only, an exclusive node, OpenMPI 5.0.3 + UCX as in S-2C.

**Builds** (generic, same modules):
- **old** = flow `be35841` + core `fcb3b06`;
- **new** = flow `partition` tip (WO-1…WO-5) + core `partition` tip, from bundles as in S-2C.

**Runs, in priority order** (drop from the bottom if the estimate exceeds 60 SBU; each run is about
40 s):
1. new, 2 reps, N=200: 1×24, m1×24, 3×8, 6×4, 12×2, 24×1, 8×3.
2. old, 1 rep, N=200: 3×8, 12×2, 24×1, 6×4.
3. traces (new, N=50, `kp_trace` + `pmpi_trace`): m1×24, 6×4, 3×8, 24×1.
4. microbenchmarks:
   - μ1: mpi4py 8-byte `Allreduce` × 10⁴ at 6 and 24 ranks with the 6×4 / 24×1 binding;
   - μ2: an empty Kokkos host `parallel_for` × 10⁴ at T = 4, 8, 24 with the same binding (≤ 40-line
     C++ tool in `dist_cpu/tools`, built against the S-2C Kokkos prefix).
5. E1 (WO-6) at 1×24 and 24×1, new, 1 rep.

**Falsification** (ms/step, new build, median):
- **B1 + P1 model rejected** if any of the following holds; then report numbers, do not tune:
  - 24×1 > 85;
  - 6×4 > 79.8;
  - 8×3 > 99.4;
  - 1×24 > 69.3;
  - per-call `mg_bottom_direct` > 0.15 ms on any rank.
- **Layout advice confirmed** if 3×8 ≤ min(6×4, 12×2, 24×1). Otherwise the docs recommend the
  measured fastest.
- **(e)** is answered by the e1–e5 attribution table (§4e). Deliverable: one table per layout,
  step = Σkernel + ΣMPI + residual on the critical rank, and the 6×4 − m1×24 diff by category.

## 9. Numbers the implementer must reproduce before trusting the design

[sim] ORB replica for 128×96×64, align (16,16,16):

| np | legacy | auto |
|---|---|---|
| 3 | blocks 48×96×64 / 80×48×64 (1.200) | C5: 128×32×64 + 2× 64×64×64 |
| 12 | 16×48×64 / 48×48×32 (1.500) | C5: 32×32×64 / 64×16×64 |
| 24 | 1 heavy 32×32×48 + 2 light per 3-group (2.000) | C5: 32×16×64 + 2× 32³ per 3-group |
| 2, 4, 6, 8, 16, 48 | unchanged | unchanged |

If the C++ factory disagrees with this table, stop: the rule in §5.1 is mis-implemented, or the
replica is wrong. Either needs the caller.

## 10. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1 | Accept that 8×3 / 4×1×2 (and any T not dividing the CCD) are documented as slow rather than engineered around? | preference | **yes**: docs + warning (WO-4/5) |
| Q2 | Is 3×8 ≈ 1×24 on genoa? | fact (S-PART) | recommend "one rank per L3 domain" from the workstation evidence until S-PART |
| Q3 | Bubble column at 5 MG levels (E1)? It changes the benchmark's numerics | fact + preference | keep 4; if E1 shows iterations ≤ +0.1 and 1×24 not slower, put it to the user as a case setting |
| Q4 | Pursue the ~10 ms residual (halo pack/unpack 5–9 ms/rank, reduction skew) as a new design? | preference, after the (e) fact | yes, as a new architect brief built on the S-PART attribution; not before |
| Q5 | VoF-weighted dynamic rebalancing (F-C) | fact (S-PART VoF spread) | deferred; trigger: max − mean ≥ 3 ms at 3×8 / 6×4 / 12×2 |
| Q6 | core v1.7.0 tag + flow default change (np > 1 partitions move for inexact configurations; published FoxBerry ladders reproduce with `levels=1`) | preference (outward-facing) | implement and gate on branches; land after user OK; CHANGELOG states the behaviour change and the `levels=1` escape |
| Q7 | Semantics `levels`: 0 = auto, 1 = legacy, instead of a new setter value or name | preference | reuse the values (no new public name) |
| Q8 | Cooperative on-node factor (F-B2), ≈ −2 ms at 24×1 only | preference | not now; trigger: the user wants 24×1 parity |
| Q9 | Host factor at ~6 % of FP32 peak (2.63 ms at 1 thread; F-B3) | fact (profile) | not now; belongs to the §13/14 owner |
| Q10 | Mixed block shapes at 24 ranks (32×16×64 with 32³) add halo neighbours | fact (WO-3 local trace: halo ms/rank before/after) | accept if ≤ +0.5 ms/rank; otherwise prefer a Cartesian-like tie-break (an extra criterion in §5.1 step 4, new decision) |
| Q11 | Rank counts with two odd factors (9, 15, 45, …) stay inexact (`n1 = ⌊m/2⌋`) | fact (census) | accept; revisit only if a user layout needs it |
| Q12 | Placement warning: unsuppressable stderr line | preference | print once on rank 0 when detected; no opt-out |
| Q13 | `max_imbalance` 1.05 for the auto search | fact | keep 1.05 (exact is reached on every bubble-column layout) |
| Q-W | Should the wall-normal invariant (F13) bind the bubble column? If it does, the factory needs the BCs (a `periodic=`-style input, new API). The 3 in 3/6/12/24 ranks lives only in y = 96, so those counts could then never be exact on this grid; only power-of-two counts could. | fact (does the channel's "centreline decoupling" still reproduce on today's code? one np 2 y-split channel run against single-rank) + preference (re-scope or enforce) | P1 as designed (status quo preserved: never a new split axis); the G-NUM-H runs at np 3/12/24 are the bubble-column evidence; put the re-scope question to the user with that evidence |

**Risk (superseded 2026-10-19).** C4 would give up a fourth in-place halving that legacy had: [sim]
480×80×160 at np 4, 8 and 24 would go from 1.14–1.50 to exact, at 3 halvings instead of 4. The
ruling of 2026-10-19 rejects such a C (nestDepth(C) ≥ nestDepth(A)): those configurations keep
legacy (or A′ where it is better and nests as deep). The census prints the changed cases.

**Risk.** A ctest that silently depends on the legacy partition at a changed (grid, np). The census
list is the safeguard: any hash change outside it is a bug, not a re-baseline.

## 11. Register entries (drafts for `docs/decisions/flow.md`, `core.md`)

1. **The default level-0 decomposition is the auto policy.**
   - Policy: legacy aligned ORB when exactly balanced; else coarse-first with the balance-aware
     split axis at depth 5 then 4 within `max_imbalance`; else the better of legacy and the
     balance-aware aligned ORB. `levels=1` = legacy.
   - Supersedes in part "Coarse-first decomposition ships opt-in, legacy remains the default"
     (2026-08-11).
   - The policy never splits an axis the legacy partition leaves whole, which keeps the
     wall-normal invariant's status quo.
   - Rejected: legacy as is (24×1 2.000 imbalance, ≈ −15 ms genoa); plain coarse-first default
     (12×2 local +3.4 ms: depth for balance below the MG depth); a Cartesian process grid (not an
     ORB tree); a core axis mask (more API for the same guarantee).
   - Evidence: IMBAL_REPORT; §9 sim; WO-3 gates.
2. **core `SplitAxis::Balanced` is opt-in; `Longest` stays core's default.**
   - Rejected: changing the default (dem/voro/amr/pnm partitions would move without a measured
     need).
   - Why: identical tree whenever Longest is exact (P-i).
3. **Host `'direct'` bottom solve: single thread, lane-scheduled M, bitwise; H-6 replication
   kept.**
   - Rejected: subset agglomeration, rank-0 solve plus broadcast (no critical-path gain at equal
     T); cooperative on-node factor (complexity for ≈ 2 ms; deferred); a thread-count `'algebraic'`
     rule (2026-10-18 entry).
   - Evidence: F5/F6, WO-1 gates.
4. **No VoF weights in the default partition; no static initial-state weights.** Dynamic VoF
   rebalancing deferred, with the trigger in Q5.
5. **CCD/L3 placement is documentation plus an init-time warning, never a placement-weighted
   partition** (results would depend on the scheduler's placement).
6. **8×3-type layouts (threads per rank not dividing the L3 domain) are unsupported for speed.** The
   fastest measured hybrid is one rank per L3 domain.

## 12. Landing record

- WO-0…WO-6 landed: flow `dfd62c4` (B1 `91964b7`, auto policy `bd7b6ce` plus review fixes,
  warning `0e6d22b`, docs `25a4473`), core `v1.7.0` (`SplitAxis::Balanced`), umbrella `3bdb0cd`
  (register: "The default level-0 decomposition is the AUTO policy").
- The local quiet run of 2026-10-20 (`dist_cpu/part/perf/final/summary.txt`) already showed both
  defects that genoa then confirmed:
  - 12×2: 49.2 (base) → 47.4 (WO-1) → **51.5 (WO-3)**, so WO-3 gate (vi) failed;
  - WO-1 (iv), the per-call bottom solve at T = 1: 0.203 ms against a gate of ≤ 0.10.
- WO-6 (E1, 5 MG levels): no gain locally or on genoa. 4 levels stay (closes §10 Q3).

## 13. Genoa WO-7 outcome (2026-10-22)

**Sources.**
- S-PART: job 28464076 (tcn1000, cores 0–23, N = 200, median of 2 new reps / 1 old rep),
  `dist_cpu/spart/results/part-353baa8/`.
- Trace attribution (`trace_attr.py`, `kper.py`, 45 timed steps).
- The local 12×2 pair, in `dist_cpu/part2/` (evidence, scripts and copies of the genoa
  attribution): flow `0ec6a9b` module (= the auto policy) on the 5965WX, cores 0–23 pinned,
  3 ranks per 6-core CCD, machine idle (load 0.12 at start).

### 13.1 Verdict against §8

| criterion | predicted | measured | verdict |
|---|---|---|---|
| 24×1 ≤ 85 | 79 ± 4 | 85.53 | rejected (by 0.5) |
| 6×4 ≤ 79.8 | 78.8 | 81.69 | rejected |
| 8×3 ≤ 99.4 | 98.2 | 100.06 | rejected (by 0.7) |
| 1×24 ≤ 69.3 | 68.2 | 67.96 | holds |
| per-call `mg_bottom_direct` ≤ 0.15 ms | ≤ 0.08 | 0.245 (m1×24) … 0.395 (24×1) | **rejected** (§13.2) |
| 12×2 | 76–82 | 88.96 (old 82.13) | **rejected** (§13.3) |
| 3×8 ≤ min(6×4, 12×2, 24×1) | — | 78.47 | **confirmed**: one rank per L3 domain |
| §4(e): e1 + e3 + e4 ≥ 80 % of 6×4 − m1×24 | — | ≈ 120 %, offset by e5 | holds (§13.5) |

How the measured changes (old → new) split:
- 24×1, −16.3: B1 −3.0 (`mg_bottom_direct` 6.98 → 3.95 ms/step) and P1 ≈ −13.3 (predicted
  −15 ± 4: holds).
- 3×8, −5.2: P1 (predicted −3…−6: holds); B1 ≈ 0.
- 6×4, +0.6: the partition is unchanged, and B1 ≈ 0 at T = 4 (0.262 ms per solve; before B1 it
  was 0.28 at T = 3).
- 12×2, +6.8: §13.3.
- E1 (5 levels) is slower at 1×24 (68.99 vs 67.96) and at 24×1 (86.10 vs 85.53).

P1's partition model held except at np 12. B1 held only at T = 1, and there at about half its
designed size.

### 13.2 B1: why the bottom solve still costs 0.25–0.40 ms

**Measured.**
- Per solve (10 per step):
  - 0.395 ms at 24×1 (T = 1);
  - 0.262 at 6×4;
  - 0.251 at 3×8;
  - 0.245 at m1×24, where B1 runs it on one thread at any T;
  - locally, 0.203 at 24×1.
- Factor (once per step): 2.63 ms at T = 1, 1.29 at T = 4, 1.18–1.19 at T = 8 and 24.
- So the bottom still costs 3.65 ms/step at m1×24 and 6.6 at 24×1.

**Why the model was wrong.** §5.3 assumed Q is L2-resident and the M application is bound by FMA
latency (5–8 µs). Two errors fit the numbers [model]:
- **Q is not L2-resident.**
  - Q is 12 planes × 128² FP32 = 786 KB. One M reads it about twice: 23 plane mat-vecs × 64 KB
    ≈ 1.5 MB.
  - The L2 is 512 KB on Zen 3 and 1 MB on Zen 4, where it is shared with the V-cycle's data. Q
    therefore streams from L3, at ≈ 30–60 µs per M.
  - With ≈ 4 M per solve (≤ 3 FCG iterations + 1), a solve costs 0.15–0.25 ms. With eight ranks
    per CCD each streaming its own Q (24×1), it costs 0.40.
  - That 24×1 figure is the one number that favours this hypothesis.
- **The lane loop is not vectorized.** Scalar mul + add under `-ffp-contract=off` is 2 × 377 k
  flops per M; at 2–4 per cycle that is ≈ 50–100 µs per M.

Which of the two holds (or both) is a fact, settled by WO-M. Nothing is redesigned before that.
B1 stays: it is bitwise and saves 3.0 ms at T = 1.

### 13.3 Q1: why 12×2 regressed

**(a) The partitions** [sim = C++, `check_decomposition.py`]. Both split 128×96×64 into four
64×48×64 quarters (x halves × y halves), with three ranks per quarter:

| | blocks per quarter | max/mean | axes cut |
|---|---|---|---|
| legacy | one 16×48×64, two 48×48×32 | 1.125 | x, y **and z** |
| auto (C5) | one 64×16×64 "slab", two 32×32×64 "columns" | 1.000 | x, y; **z is whole on every rank** |

`predict_hierarchy` gives the same hierarchy for both: 4 levels, no telescoping, bottom 16×12×8 on
all 12 ranks. **Telescoping and MG depth are ruled out.**

**(b) The measurement.** Local, one build; only `set_decomposition` differs.

| | auto (`levels=0`) | legacy (`levels=1`) |
|---|---|---|
| ms/step, N = 200, two reps | 52.42, 52.27 | 47.93, 47.69 |
| traced (45 steps) | 51.85 | 48.45 |
| kernel ms/rank, mean / max | 35.4 / 39.7 | 36.9 / 39.9 |
| MPI ms/rank, mean | 4.29 | 4.38 |
| untraced gap ms/rank, mean [min, max] | 10.9 [7.2, 12.5] | 5.8 [4.7, 7.9] |
| gap just after a stencil exchange's unpack, ms/step | slabs 1.0; **columns 6.2–6.8** | 48×48×32: 0.95–1.04; 16×48×64: 0.09 |

- The loss is +9.4 % here and +8.3 % on genoa, whose old build also lacks B1: the same relative
  loss on two machines with different CCD sizes.
- Balance worked: the critical rank's kernel time is equal, and the mean dropped 1.5 ms.
- The loss is untraced time on the 8 column ranks:
  - 54 µs after each of the 80 smoother colour passes per step;
  - 93 µs after each of the 10 residuals;
  - 98 µs after each of the 10 operator applies;
  - total 6.2 ms/step.

**(c) Mechanism** [code + meas].
1. K3 (`doc/vof_distributed_host_comm_design.md` §5.2) makes an axis a level **wrap axis** iff the
   level is one block along it (`mac_cutcell_mg.hpp:1107`). Legacy at np 12 cuts z, so no level
   wraps. Auto never cuts z, so z wraps on every rank.
2. On a wrap axis the overlapped box passes span the whole block. The boundary *shell*, i.e. the
   cells the post-exchange launch visits (`ccBoxShellRuns`, `mac_cutcell.hpp:370`), therefore loses
   its two z-slabs.
3. For the columns that drops the shell under `kHostSerialCellCutoff = 8192` visited cells
   (`mac_cutcell.hpp:284`). The shell then runs **serially, on one of the rank's two threads**,
   outside any Kokkos launch, which is why the trace does not see it:

| level-0 block | wrap | shell cells | x-runs per pass | launch |
|---|---|---|---|---|
| auto column 32×32×64 | z | 7936 | 3840 | **serial** |
| auto slab 64×16×64 | z | 9984 | 1792 | parallel |
| legacy 48×48×32 | — | 10248 | 2760 | parallel |
| legacy 16×48×64 | — | 9224 | 5704 | parallel |

4. A serial column-shell pass costs 54 µs. The cost comes from the 3840 single-cell x-runs: the
   shell's x faces make one `run(...)` call per row and side. A run costs ≈ 11 ns, against
   ≈ 1.5 ns per cell of a whole row (interior pass: 46 µs for 57600 cells on 2 threads).

Why other configurations escape:
- Legacy cuts z, so every block's shell keeps its z-slabs and stays above 8192 cells.
- np 3 and 6 have shells of ≥ 12032 cells (T = 8, 4).
- At np 24, T = 1, so serial and parallel cost the same.

**(d) What §0/§4(a) missed.** Per-rank cost is not a function of cells. Here it has a
*discontinuity*: the host launch policy (8192 visited cells, the same at every T), selected by
block shape and the level's wrap set. The other candidates do not explain the loss:
- not halo surface: the auto slab has 10240 remote face cells against legacy's 9216, worth
  ≈ +0.5 ms;
- not neighbour count;
- not VoF: the bubbles are the same, and per-rank VoF time is 4.1–6.2 ms (auto) against 4.2–6.4
  (legacy);
- not MG depth.

**(e) Correction to §4(a)** [sim]. The balance-aware ORB does **not** reproduce a Cartesian
factorization at np 12 and 24.
- The factor 3 is split at an m = 3 node whose y extent is already halved (48 cells = 3 quanta).
  The 1 : 2 cut makes a 16-thick slab, and the 2-rank remainder is then cut across x.
- Exact uniform partitions exist: 32×32×64 or 64×32×32 at np 12, and 32³ at np 24.
- A separate, unattributed observation: at 24×1, the slab ranks carry +4.1 ms of kernel time over
  the 32³ ranks at equal cells (genoa trace). About 0.9 ms of that is VoF placement; the rest is
  spread at +4–15 % over the volume kernels and the halo gather. With T = 1 this is not the cutoff.
  It is Q-U (§13.9).

**(f) Confirming measurement.** Done: (b) is the decisive pair. The cutoff counts cells, not
seconds, so the mechanism does not depend on the machine, and no billed run is needed to confirm
it. A genoa 12×2 trace pair is optional at the next billed milestone (WO-S).

### 13.4 Q2: the policy fix

**Decision: fix the discontinuity, not the partition rule. Add a narrow partition guard only if
the fixed code still loses at 12×2.**

1. **WO-A (G-BIT): the shell's serial test counts its x-runs.**
   - **Change.** In `ccBoxShellRuns`, the shell branch (non-empty skip box) calls
     `hostRunSerial(cells + kShellRunCells * runs)` instead of `hostRunSerial(cells)`, where:
     - `runs = nC * ((nxLo > 0) + (nxHi > 0))`;
     - `kShellRunCells = 8` is a `constexpr` next to `kHostSerialCellCutoff`, commented "a
       one-cell x-run costs about 8 streamed cells (5965WX: serial pass 54 µs = 7936 cells × 1.5 ns
       + 3840 runs × 11 ns)".
   - **Unchanged:** the interior branch, `kHostSerialCellCutoff`, and every other `hostRunSerial`
     caller. The serial/parallel choice never changes a bit, by `ccBoxShellRuns`' contract (cells
     are independent within a launch).
   - **Which shells change** (cells + 8·runs):

     | shell | before | after | path |
     |---|---|---|---|
     | auto column, level 0 | 7936 | 38656 | → parallel |
     | auto column, level 1 | 1920 | 9088 | → parallel |
     | 6×4, level 1 | 2944 | 10112 | → parallel |
     | auto slab, level 1 | 2432 | 5504 | stays serial |
     | levels ≥ 2 | — | ≤ 1984 | stay serial |

     Slab and legacy level-0 shells are already parallel and stay so.
   - **Expected** [model]:
     - 12×2 auto, local: −3 ms (column shells 6.2 ms serial → ≈ 3.1 at T = 2, plus ≈ 100 forks
       × 0.8 µs);
     - 6×4: −0.3…−0.6;
     - every other layout within ±0.3.
2. **WO-D (conditional): a partition guard.** Every auto candidate (C5, C4, A2) must satisfy,
   besides `splitAxes(X) ⊆ splitAxes(A)`:

   `splitAxes(X) == splitAxes(A)  OR  4·maxCells(A)·numBlocks ≥ 5·cells(G)`

   The second clause says legacy max/mean ≥ 5/4; the integer form is rank-consistent.
   - **Why.** The wrap set (the axes no rank cuts) selects the K3/H-7 code paths, and the cell count
     cannot see what they cost: +9.4 % at a legacy max/mean of 1.125 (§13.3(c)). A candidate that
     changes the wrap set must therefore buy a ≥ 20 % cut in the critical rank's cells, twice the
     worst code-path penalty measured.
   - **Census replica** (`dist_cpu/part2/guard.py`; np ∈ {2…10, 12, 15, 16, 20, 24, 32, 48, 96}
     over the §7 grids plus the two channel grids). It reverts exactly four configurations to
     legacy (legacy max/mean 1.03–1.15):
     - 128×96×64 at np 12;
     - 384×288×192 at np 9;
     - 100×50×50 at np 5 and np 10.

     It keeps 48³ at np 3/5/7 (legacy max/mean ≥ 1.25) and 128×96×64 at np 3/24/32/96.
   - **Amended 2026-10-22 (WO-D review): the guard reverts 14 census configurations.** The C++
     census ctest (`decomposition_policy`) covers more grids than `guard.py`, and on them the same
     rule reverts ten more, all with legacy max/mean < 5/4 (1.03–1.24):
     - the four of `guard.py`'s set above;
     - ten small test grids that flow's and coupling's MPI tests decompose: 30³ at np 3/5/6/7,
       36³ at np 3/5/6, 48×32×40 at np 3 and 48×40×30 at np 3;
     - 480×80×160 at np 30 (the channel loop), the only production-shaped one.

     These are not a deviation from the design. They are the same rule applied to more grids, and
     every revert returns to the pre-2026-10 (legacy) partition, so no published configuration
     moves. Restricting the guard to a list of configurations would be tuning to cases. The census
     ctest asserts the 14 by name, and asserts that 48³ at np 3/5/7 stays on auto.
   - **Landing rule (pre-registered, WO-C).** After WO-A, compute
     R = median(12×2 auto) / median(12×2 legacy) locally. Land WO-D iff R > 1.01. If WO-C cannot
     run within 48 h, land WO-D (the default).

**Rejected:**
- **"Coarse-first only when legacy imbalance ≥ X."** np 3 and np 12 have the same legacy
  max/mean, 1.125. One gains 6.3 % and the other loses 8.3 %.
- **An MG-depth or telescope criterion.** The hierarchies are identical and nothing telescopes.
- **A smooth cost term** (surface, neighbour count, rows per cell). The loss is a step in the
  launch policy. A term fitted to it is tuning to one case, and a rows term even favours auto at
  np 12, because legacy's 16×48×64 block has the most rows.
- **A thread-count-aware partition.** Partitions stay pure functions of global integers (the
  register precedent of 2026-10-18).
- **A thread-count-aware `hostRunSerial`.** It is legal, since it is bitwise, but it needs
  per-machine fork-join constants (μ2: 0.77 / 1.21 / 4.36 µs at T = 4 / 8 / 24 on genoa), which is
  tuning. The x-run term removes the measured defect with one constant that describes the code,
  not the machine.
- **Lowering `kHostSerialCellCutoff` globally.** H-7 measured its benefit at the coarse levels.
  There is no evidence for touching the interior launches.
- **"Odd factor last" / uniform shapes (§13.3(e)).** The benefit is unmeasured, and it would move
  the np 12/24/96 partitions a second time in one campaign. This is Q-U.

### 13.5 Q3: the remaining distributed overhead

Genoa traces, 45 timed steps. The exact coverage decomposition is the rank mean in ms/step
(`trace_attr` (a)); the attribution by cause below uses the `kper` categories.

| | 6×4 − m1×24 | 24×1 − m1×24 |
|---|---|---|
| step | +15.2 (82.65 vs 67.44) | +19.3 (86.76) |
| kernels | +1.45 | −3.9 |
| MPI | +4.29 | +8.0 |
| deep copies | +0.24 | +3.3 (halo receive staging) |
| untraced gaps | +9.05 | +11.8 |

By cause:
- **Halo exchange, the dominant term.**
  - Gather + unpack kernels: +5.7 (6×4), +7.0 (24×1).
  - Post-exchange gaps (shared-memory completion, `MPI_Iprobe` polling, receive copies): ≈ +9 and
    ≈ +11.5.
  - Together that is ≈ 57 µs per exchange at 257 exchanges per step (6×4).
  - **Caveat:** these gaps mix waiting with untraced serial host work, and §13.3 found 6.2 ms of
    such work on one layout. Until WO-T, the split overstates the waiting.
- **Reductions.** 78 `MPI_Allreduce` calls per step add +2.2 (6×4) and +4.8 mean (24×1). Latency
  is only 0.23 / 0.29 ms of that (μ1: 3.0 / 3.7 µs); the rest is skew. At 24×1 the skew is the VoF
  imbalance: 9.2–10.6 ms of VoF work on ranks 0–15 against 2.6–3.4 on ranks 16–23.
- **VoF block exchange** (4 `Waitall` per step): +1.6 and +2.5.
- **Pressure-MG kernels:** +2.8 at 6×4.
- **Bottom:** +0.3 at 6×4; +2.95 at 24×1 (factor 2.63 + solve 3.95 at T = 1).
- **Thread efficiency (a gain):** VoF −2.7, velocity −1.75, other −2.45 at 6×4. Four-thread
  teams beat one 24-thread team (μ2: 0.77 vs 4.36 µs per launch, and m1×24 makes 877 launches per
  step).

**Packages, with expected effect and risk:**

| package | where | expected [model] | risk | recommendation |
|---|---|---|---|---|
| WO-A (this note) | 12×2; level-1 shells at 6×4 | −3 (12×2 local), −0.3…−0.6 (6×4) | low, bitwise | now |
| WO-T (this note) | attribution | 0 | none | now; needed before any comm design |
| P-CA: CA smoothing with domain BCs | every np > 1 (`caSmooth` is off for the bubble column: `hasBC_`, the y walls) | −1.5…−3 (≈ 80 fewer exchanges per step) | medium: the bitwise ring argument must be re-derived with BC ghosts | new architect brief to the comm-design owner, after WO-T |
| P-B2: bottom Q traffic | every layout, 1×24 included | −1.5…−3 | medium: storage changes may move bits | after WO-M |
| P-sync: per-exchange completion cost | every np > 1 | unknown until WO-T | — | re-attribute first |
| P-VoF: VoF weights (F-C) | 24×1 only (≈ 3 ms on the critical path) | ≤ 3 | high | deferred: the trigger (≥ 3 ms at 3×8/6×4/12×2) is not met; VoF spread is 1.0 at 6×4, 0.5 at 3×8 |
| P-shell: restructure the x-runs in `ccBoxShellRuns` | every np > 1 that cuts x | −0.5…−2 | low–medium (the H-7 run interface) | only if WO-T shows shell x-runs above 15 % of box-pass time at 3×8/6×4 |

**Is ≈ 1×24 reachable at every layout?** Not with today's halo engine.
- 3×8 and 6×4 carry +11…+15 ms, ≈ 14 ms of it halo exchange. Even with WO-A, P-CA and P-B2 the
  model gives ≈ 74–76 ms (+10–13 %).
- Parity needs the per-exchange cost to fall from ≈ 57 µs to ≈ 15–20 µs. That is P-sync's question.
- 24×1 also carries the VoF imbalance (≈ 3 ms) and the one-thread bottom (factor 2.6 + solve
  4.0 ms). Its realistic floor is ≈ 78–80 ms.

**What users are told** (genoa, one NUMA domain = 24 cores = 3 CCDs; this case size):
1. If MPI is not needed: one rank with 24 threads (68.0 ms).
2. If the run is distributed: **one rank per CCD with 8 threads** (3×8: 78.5 ms, +15 %).
   - 6×4 (81.7) is acceptable.
   - Use 24×1 (85.5) only when the code needs 1 thread per rank.
   - Avoid 12×2 until WO-A, and WO-D if it lands: 89.0 now, ≈ 82 expected after.
3. Never use a thread count per rank that does not divide the CCD (8×3: 100.1). `init_mpi` names
   the offending ranks.
4. Multi-node: the same rule per node, i.e. 24 ranks × 8 threads on a 2-socket genoa node. The
   recipe is in `docs/SNELLIUS.md`.

### 13.6 Work orders (dependency order)

Local perf below means: 5965WX, cores 0–23, `pin.sh`, idle (load ≤ 1 at start), N = 200, 3
interleaved reps, median. The baseline is a flow `dfd62c4` host MPI build.

| WO | content | class | gate | expected |
|---|---|---|---|---|
| **WO-T** | Name the host serial runs in the profile. In flow `src/`, in each `hostRunSerial(...)` true branch (`grep -n 'hostRunSerial(' src/`), wrap the serial loop in `if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Profiling::pushRegion("<launch name>@serial")` … `popRegion()`. Each site uses a string literal, so nothing is allocated when no tool is loaded. Tools (outside flow): `trace_attr.py`/`kper.py` count type-6 regions named `*@serial` as kernel time; confirm that `kp_trace` records push/pop with both timestamps | NONE | (i) G-BIT: `state_hash` 1×24, m1×24, np 2/12 at levels 0 and 1. (ii) Untraced 12×2 within ±1 %. (iii) Traced auto 12×2: the column ranks show `cc_smooth_box@serial` ≈ 4.3 ms/step, and their post-unpack gap is ≤ 1.2 ms | — |
| **WO-A** | §13.4(1) | G-BIT | (i) G-BIT: `state_hash` 1×24, m1×24, np 2/3/4/8/12/24 at levels 0 and 1; `ctest -LE bench` host and CUDA; no hash changes. (ii) Local: 12×2 auto ≤ 50.0 ms; 12×2 legacy within ±1 %; 6×4 ≤ base + 0.3; 4×6, 24×1, m1×24 and 1×24 within ±1 %. (iii) With WO-T: no `cc_*_box@serial` region above 0.1 ms/step at level 0 on any auto 12×2 rank | 12×2 auto −3; 6×4 −0.3…−0.6 |
| **WO-C** | On the WO-A build, run 12×2 at levels 0 and 1, with traces. Apply the §13.4(2) landing rule; log R in the campaign log | MEAS | one summary line with R | R ≈ 1.03 [model], so WO-D lands |
| **WO-D** | §13.4(2) in `CutcellMG::decomposition`, in all three candidate tests. The replica (`part/orb_sim/cf2.py`) gets the same condition. Census ctest `decomposition_policy`: assertion 7 adds the condition and prints the reverted list. Update the flow CLAUDE.md "Decomposition" bullet and the `set_decomposition`/`mpi_block` docstrings | G-BIT per configuration (reverted ones equal `levels=1`; all others equal the `dfd62c4` auto partition) | C++ census == replica: exactly the 14 reverts of §13.4(2) (asserted by name in the census). np 12 `state_hash` == the `levels=1` hash; np 3/24 hashes unchanged. Coupling MPI tests green. Local 12×2 at levels 0 and 1 equal within ±1 % | 12×2 = legacy |
| **WO-M** | Bottom-solve facts: (1) inner FCG iterations per bottom solve (mean and max over 50 steps); (2) per-M time of `BottomDirect::laneSubstitute` at T = 1, on the case's 12-plane bottom and on a 4-plane bottom (Q = 262 KB, L2-resident on both machines); (3) whether the lane loop is vectorized (packed `vmulps`/`vaddps` in `objdump -d` of the host TU, or `-fopt-info-vec`) | MEAS | one table in the campaign log | decides P-B2 |
| **WO-G** | Docs: `docs/DECOMPOSITION_AND_MULTIGRID.md` rule 9 with the S-PART numbers (the §13.5 list). flow CLAUDE.md: correct the B1 line ("halves the 1-thread solve; 0.25–0.40 ms per solve on genoa, §13.2") and the decomposition bullet if WO-D lands. `docs/SNELLIUS.md`: the 3×8 recipe, now confirmed | NONE | coordinator review | — |
| **WO-S** | Genoa confirmation, folded into the next billed milestone (no dedicated job): 12×2 at levels 0 and 1, 6×4, 3×8 and 24×1 on the WO-A(+D) build, plus one 12×2 trace pair | MEAS, billed, user OK | the §13.8 numbers within their bands | — |

**Order and landing.**
- WO-T and WO-A are independent and bitwise. They land on flow main directly with the G-BIT proof
  and need no reviewer pass, because nothing numeric moves.
- WO-C then decides WO-D.
- WO-D changes the default partition of 14 census configurations, so it gets a reviewer pass and
  a register entry (§13.10). It needs no core change, so there is no core tag.

### 13.7 Verification gates

- **G-BIT** (§7) for WO-T and WO-A:
  - `state_hash` from `ckpt_t43`, 50 steps, rtol 1e-8 and 1e-10;
  - at 1×24, m1×24 and np 2, 3, 4, 8, 12, 24, each with both `set_decomposition(0)` and `(1)`;
  - host battery `ctest -LE bench` (248) and CUDA unchanged.
- **WO-D:** G-BIT per configuration against the stated reference (reverted ones against
  `levels=1`, all others against the `dfd62c4` auto partition), and C++ census == replica. G-NUM-H
  is not needed: no partition appears outside those two known sets.
- **Perf:** the local bands in §13.6, noise ±1 %. A WO-A result outside its band is reported with
  the WO-T trace, not tuned: `kShellRunCells` is never adjusted to pass the gate.
- **Limiting cases:**
  - np 1 and m1×24 are untouched (every axis wraps: one launch, no shell);
  - T = 1 layouts keep the same times within noise (serial = parallel on one thread).

### 13.8 Expected genoa ms/step [model] (supersedes §0's table)

| layout | S-PART new | after WO-A | after WO-A + WO-D (if it lands) |
|---|---|---|---|
| 1×24 | 67.96 | 68.0 | 68.0 |
| m1×24 | 66.87 | 66.9 | 66.9 |
| 3×8 | 78.47 | 78.5 ± 0.5 | 78.5 |
| 6×4 | 81.69 | 81.2 ± 0.5 | 81.2 |
| 12×2 | 88.96 | 84 ± 1.5 | ≈ 81.6 (legacy 82.13, with B1 at T = 2) |
| 24×1 | 85.53 | 85.5 ± 0.5 | 85.5 |
| 8×3 | 100.06 | 100 | 100 |

Closing the rest of the gap to 1×24 is the work of P-CA, P-B2 and P-sync (§13.5), not of the
partition.

### 13.9 Open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q-A | Is `kShellRunCells` = 8 right? | fact (WO-T per-run cost) | 8. Change it only through a new measured derivation, never to pass a gate |
| Q-D | Land the guard WO-D? | fact (WO-C) | land iff R > 1.01; land it if WO-C cannot run within 48 h |
| Q-T | The guard threshold 5/4 | preference (how much unpriced code-path risk to accept) | 5/4 (twice the measured 9 % penalty) |
| Q-U | Uniform-shape partitions ("odd factor last") at np 12/24/96 | fact (an A/B of explicit partitions, which needs a test-only hook for an explicit decomposition) + preference (a second partition change) | not now. Revisit with P-CA/P-sync, once per-shape costs can be attributed (WO-T) |
| Q-B | Bottom redesign (P-B2) | fact (WO-M) | not before WO-M |
| Q-C | CA smoothing with domain BCs (P-CA) | preference (scope) + architect | brief after WO-T; expected −1.5…−3 ms at every np > 1 |
| Q-V | VoF weights (F-C) | fact | deferred; the §10 Q5 trigger is unchanged and not met |
| Q-S | Confirm WO-A/WO-D on genoa? | preference (billed) | fold into the next milestone. The local pair is decisive because the cutoff counts cells |
| Q-P | The "≈ 1×24 everywhere" target | preference | target 1×24 + 10 % for one-rank-per-CCD layouts; 24×1 is not a target layout |

Status of the §10 questions:
- Q2: answered. 3×8 is the fastest distributed layout, +15 % over 1×24.
- Q3: closed; 4 levels stay.
- Q4: superseded by Q-C and Q-P.
- Q5: not triggered.
- Q10: answered by §13.3. The cost was the cutoff, not the neighbour count; what remains is Q-U.

### 13.10 Register entries (drafts)

1. **Host box-shell launches count their x-runs in the serial test.** `ccBoxShellRuns` runs a shell
   serially only if cells + 8·runs < 8192.
   - Rejected: a thread-count-aware cutoff (it needs per-machine constants); lowering the global
     cutoff (no evidence); no change (12×2 +9.4 %).
   - Evidence: §13.3, `dist_cpu/part2/`.
2. *(Only if WO-D lands.)* **The auto partition keeps legacy's split-axis set unless legacy
   max/mean ≥ 5/4.**
   - Supersedes in part "The default level-0 decomposition is the AUTO policy".
   - Rejected: an imbalance threshold alone (np 3 vs np 12); a depth or telescope criterion (the
     hierarchies are identical); smooth cost terms (the loss is a step in the launch policy);
     thread-aware partitions.
3. **B1's cost premise is falsified; B1 is kept.**
   - Measured 0.25–0.40 ms per host bottom solve on genoa, against ≤ 0.08 designed.
   - B1 gains only at T = 1 (−3.0 ms at 24×1).
   - No bottom redesign before WO-M.
4. **"One rank per L3 domain" is confirmed on genoa.** 3×8 (78.5 ms) is the fastest distributed
   layout. On a single NUMA domain, one rank (68.0 ms) is still faster.
