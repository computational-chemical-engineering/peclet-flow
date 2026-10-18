# Device pressure coarse tail: the bottom factor, the bottom solve and the host syncs

Design note, 2026-10-18. Branch `gpucoarse` (worktree `suite/flow-gpucoarse`, from main `b820652`).
Answers `bubble_column_perf/gpu_diag/BRIEF_ARCH_GPU.md`. Implementer: `opus-implementer`, who has
seen neither the brief nor the conversation; every number, layout, ordering and gate needed is here.
Parents: `doc/vof_step_performance_design.md` §13 (the device direct bottom, decisions D-1..D-5,
cited below as "§13"), `doc/vof_projection_cost_design.md`, `doc/vof_projection_distributed_design.md`.
Evidence: `bubble_column_perf/gpu_diag/GPU_REPORT.md` and its `logs/ana_{auto,fp64}.txt`.
Anchors are `file:line` at `b820652`.

## 0. Outcome first

Bubble column (RTX 5080, CUDA, no MPI, 128×96×64, `ckpt_t100`, rtol 1e-8, 12.7 PCG iterations/step):
**`'auto'` 18.92 → ≈ 13.7 ms/step (13.1–14.8); `'fp64'` 21.86 → ≈ 16.6.** With the separate
batch_stats/bbox package (−2.0, not here) `'auto'` ≈ 11.7. The bottom code is shared by both
V-cycle precisions, so every saving applies to `'fp64'` and `'auto'` alike; the pending WO-F default
flip changes nothing below.

| WO | change | class | `'auto'` ms/step | gate |
|---|---|---|---|---|
| 0 | measurements P1–P5; no production change | — | 0 | numbers in the log |
| S1 | per-level device fluid count; no combined (multi-result) reducer in the PCG | a | −0.25 … −0.45 | G-BIT, fence count |
| F1 | the factor in team shared memory, in place; same scalars | a | −1.9 … −2.5 | U9, G-BIT |
| B2 | bottom FCG v2: segment-split M, fixed-chunk reductions, T = 512 | **b** | −1.3 … −1.7 | U10, G-NUM |
| S2 | the per-iteration packet read and `readSlot` instance-scoped, pinned | a | −0.05 … −0.15 | G-BIT |
| S4 | step-wide fence hygiene (exec-less copies, fills, global fences) | a | −0.2 … −0.5 | G-BIT, G-SYNC |
| X | early coefficient build; factor on a side execution instance | a | −0.4 … −1.1 (F1 + X ≤ 3.1, the whole factor) | G-BIT, G-OVL |
| F1b | (conditional) assemble by rows | a | 0 or −0.1 … −0.3 | U9 |
| D | log, register, CLAUDE.md, state file | — | 0 | — |

Class **a** = structural, proved bitwise (G-BIT). Class **b** = recorded numerics change (G-NUM,
re-baseline). **D-1 stands**: nothing here reverses a register entry. B2 amends two normative clauses
of §13 on device only (Q-1). Three findings drive the design (§3): the factor's working set lives in
global memory (it was designed for 0.35–0.55 ms and measures 3.1); each of M's 24 plane phases keeps
only 4 warps' worth of loads in flight; and Kokkos 5.1.1 fences twice on every reduction with two
results, so the "device-resident" `{sum, count}` mean costs two host syncs per call.

## 1. Problem and scope

Per step, `'auto'`: kernel sum 17.1 ms, GPU idle 2.75 ms (1106 gaps < 5 µs = 1.24 ms; 144 gaps of
5–20 µs = 1.45 ms), 1163 launches, 202 host syncs (133.7 `cudaStreamSynchronize`, 68.9
`cudaDeviceSynchronize`). The bottom (16×12×8 = 1536 cells; walls ±y, so slow axis s = y, P = 12
planes of b = 128 cells, no border) costs:

- `mg_bottom_factor` 3.11 ms: one launch per step, one team of 768 threads, so 1 of 84 SMs busy. It
  is refactored every step because ρ changes the operator (`setOpenness` → `facStale_`,
  `mac_cutcell_mg.hpp:1993`; lazy launch at `:4833`).
- `mg_bottom_direct` 2.65 ms: 12.7 launches × 209 µs, one team of 896 threads.

Both run on the critical path with the other 83 SMs idle.

**In scope:** the single-rank device path: the factor and solve kernels, the device Krylov scalar
plumbing, the host syncs inside `Solver::step()` on device backends, and where the factor is
launched. **Out of scope:** the host and distributed CPU paths (their bits must not move);
batch_stats/batch_bbox (another engineer, `src/vof/block_batch.hpp`: do not edit that file);
curvature numerics; AMR; Snellius runs; CUDA graphs and the fine-level Krylov fusions (deferred,
§4 D-T7, Q-5).

## 2. Constraints and invariants

1. **The pinned bits.** The `'fp64'` V-cycle and the default bottom are what the CUDA `state_hash` and
   the np-tests pin. A class-a WO is proved bitwise against its parent commit (G-BIT). A class-b WO
   passes G-NUM against `b820652` and re-baselines the CUDA hashes in its own commit, with the
   old → new table in the message.
2. **Defect correction.** The Krylov matvec and residual use the exact FP64 operator. Preconditioners
   below may be float. Inside the bottom: the stored operator, the A·1 = 0 projections and the residual
   that decides convergence stay FP64 (§13 D-3). Bottom semantics are unchanged: relative ∞-norm
   residual ≤ τ = 1e-5, cap 100, and the non-finite flag through the A6 packet.
3. **The host stays bitwise.** Host results, and np = 1 under `init_mpi` (the distributed host-scalar
   PCG), stay memcmp-equal to `b820652` and to single rank. Every device-only change sits behind
   `if constexpr (!kHostMemory)` / `!kHostLane` / `!kHostFactor`, or on the `!distributed_` device
   branch.
4. **Reproducibility.** The factor and M stay bitwise independent of the team size T (§13.2). After
   B2 the whole bottom solve is too (Q-1). Every change is bitwise run-to-run.
5. **No transfers.** No new D↔H copy inside `step()`. The factor launch stays decided by a host flag,
   not by a device read.
6. **Conventions.** No environment variable changes a result, and no new public name (all new names
   are internal C++). Kokkos device code lives in `.hpp`. One Kokkos per process. Device-first and
   MPI-distributable: §5.7 says how each choice extends to the distributed device path.
7. **Iteration order** follows storage (`policy.hpp`): new MDRange kernels use `MDRange3`.

## 3. Diagnosis (first principles; this is what the decisions rest on)

**3.1 Why the factor takes 3.1 ms, not 0.35–0.55.**
- Per plane (b = 128) the team algorithm (`mg_bottom_direct.hpp:388–521`) runs ≈ 170 barrier phases:
  8 tiles × (17 diagonal-tile column phases + panel + trailing), 16 for W, 1 for Q, plus assemble.
- Σ, L and W are b×b FP32 views in **global** memory (`D.Sg/L/W`, `:146`), 192 KB in all. That
  exceeds the SM's 128 KB unified L1, which is shared with the scratch carve-out.
- So each dependent access batch in the panel, trailing update, W and Q phases pays L2 latency,
  ≈ 0.5–0.7 µs. 3.11 ms / 12 planes = 259 µs per plane. A latency model of those phases gives
  ≈ 120–250 µs per plane, against ≈ 20 µs for the 136 diagonal-tile phases, which already work in
  scratch.
- With the working set resident in shared memory, each dependent step costs ≈ 30 cycles instead of
  ≈ 600. Model: ≈ 30–50 µs per plane, so **0.4–0.7 ms per factor**. The FLOP floor is ≈ 12.6 M FP32
  FMA, about 40 µs on one SM.
- The §13.4.7 layout table already permits this ("level-0 scratch is a permitted, bitwise-neutral
  optimisation"). It was not taken because Σ, L and W do not all fit in scratch side by side. The
  fix is to make them one buffer, in place.

**3.2 Why one bottom solve takes 209 µs, not 85–125.**
- M is 2P = 24 dependent plane phases plus cast and uncast. Each plane phase computes b = 128 outputs,
  one thread per output, so only 4 warps are active.
- Each of those warps keeps ≤ 16 loads in flight (`PECLET_BOTTOM_UNROLL`), about 8 KB in all. Covering
  L2 latency on one SM needs ≈ 160 GB/s × 0.6 µs ≈ 100 KB in flight. A 64 KB Q plane therefore costs
  ≈ 3 µs instead of ≈ 1 µs. §13.9's risk R-D1 ("if the FP32 gemv phases are memory-bound above
  1 µs …") is what happened.
- About 12 Kokkos team reductions per solve add ≈ 1–1.5 µs each. They run on the 5080's FP64 pipe
  (2 lanes/clk/SM), and their bits depend on T. T = `team_size_max`, which the compiler's register
  allocation decides, so any edit of the kernel can change the bottom's bits.
- Breakdown for 2 inner iterations, ≈ 2 M applications: M ≈ 2 × 70 µs, the rest ≈ 70 µs.

**3.3 Where the host syncs come from (Kokkos 5.1.1, `extern/install/nvidia-cuda/include`).**
- **Combined reducers always fence twice.** `impl/Kokkos_Combined_Reducer.hpp:486–532` handles a
  `parallel_reduce` with ≥ 2 results ("Since we don't support asynchronous combined reducers yet").
  It reduces into host memory, fences, copies back, and fences again, even when every result is a
  device View.
  - `removeMean`'s single-rank device branch (`mac_cutcell_mg.hpp:5958`, `{sum, count}` into
    `slot(kMsum)`, `kcnt_`) and `exitFp32`'s `joint` (`:4074`) are such reductions: 14.7 + 12.7
    calls/step.
  - nsys counts 40.5 "fence due to result being value" plus 40.3 "fence after copying values back"
    per step. Their authors meant them to be async ("no host read", A6).
  - The register entry "device-resident Krylov scalars" assumed one host read per iteration, and this
    is a premise of it that was false. Fixing it is consistent with that entry, not a reversal.
- **The packet read is exec-less.** `deep_copy(kpk_, subview(ks_, …))` (`:2875`) runs a global fence,
  a synchronous copy into pageable memory, and a global fence: the "40 syncs after `mgmax`".
  `readSlot` (`:2280`) is exec-less too.
- **Exec-less copies, fills and fences elsewhere in the step.** "copy between contiguous views"
  39.3/step (12.7 of them the packet), "Unnamed Global Fence" 25.1, "scalar copy" 15.1 pre plus 15.1
  post, 20.8 `cudaMalloc` and 2.4 `cudaFree`.
  - Example: `Kokkos::deep_copy(old_[c], C[c].u)` at `flow_ibm_project.hpp:103`.
  - Each first fence drains the GPU, and the GPU then idles ≈ 10 µs until the host relaunches. That is
    the 144 gaps of 5–20 µs.
  - Global fences (`cudaDeviceSynchronize`, `cudaFree`) also wait on every other stream. This matters
    for X.
- **The MDRange reduction tree is functor-independent on CUDA.** Tiles come from device properties
  only (`Cuda/Kokkos_Cuda_MDRangePolicy.hpp:76–86`). The reduce block is
  max(pow2(tile product), 256), with 256 unless the kernel needs > 255 registers
  (`Cuda/Kokkos_Cuda_Parallel_MDRange.hpp:300–360`). The grid is min(block, tiles).
- So a sum over the same range has the same tree whether it runs alone or as the first component of
  `{sum, count}`. This is why S1 is expected to be bitwise.

**3.4 What does not pay** (the reasons behind the rejections in §4):
- Cross-SM work per bottom phase: a launch or grid barrier costs ≈ 2.5–4 µs against ≈ 1 µs in one
  SM, and one phase moves a single SM's latency-bandwidth product (64 KB).
- Fusing the L2 level (12 288 cells) into the bottom team: one SM sweeps L2 in ≈ 2–3 µs (0.4 MB at
  ≈ 160 GB/s), against 1.5–2 µs for a full-GPU launch, and L2 costs only 0.15–0.20 ms/step in all.

## 4. Decisions

| # | decision | rejected, and why |
|---|---|---|
| D-T1 | **Factor in team shared memory** (F1): Σ → L → W in one packed lower-triangular scratch buffer, overwritten in place; every scalar the same expression in the same order as today, so it is bitwise | **Lagged factor** (reuse across k steps, or a drift criterion): class b. After B2 each extra inner FCG iteration costs ≈ 35 µs × 12.7 ≈ 0.45 ms/step, comparable to the whole post-F1 factor. It needs topology-change invalidation, and X hides the factor at zero numerics cost. **Multi-SM factor** (block cyclic reduction, SPIKE): off-diagonal blocks fill in dense (2–3× the flops), log-depth pays only with cross-SM syncs, and it changes D-1's elimination order. **Dense inverse or vendor solver**: still rejected by D-1 (a); see D-T8 |
| D-T2 | **Hide the residual factor** (X): build the variable-ρ coefficients right after `updateProperties()`, launch the factor on a second Kokkos execution instance, and join with one instance fence before the first bottom solve | **Overlap without moving the build**: the factor is needed ≈ 0.1 ms into the projection, so there is no window. **Native CUDA events**: portable instance fences cost ≈ 20 µs/step, which does not justify a backend `#if` |
| D-T3 | **Bottom FCG v2** (B2, class b): M's plane products split into 4 fixed j-segments over 512 work items and combined in fixed order; all reductions in 64 fixed chunks; T = 512 by `LaunchBounds`. The bottom's bits become T-independent | **Bitwise high-unroll M** (more loads in flight, one accumulator kept): raising the register count changes `team_size_max`, hence T, hence the T-dependent dots, so it is not bitwise anyway. **Multi-SM apply**: §3.4. **Another slow axis**: x (periodic) needs the border, 1.73 MB and 36 phases per M against y's 1.57 MB and 24; z gives b = 192 and 2.4 MB. **Smaller bottom**: E1, +8.5 outer iterations. **Fixed two-step refinement instead of FCG**: amends D-1's FCG-to-τ, for ≈ 0.3 ms |
| D-T4 | **Device fluid count per level** (S1), counted once per operator build into a device scalar. The per-call mean is a single-result async reduction | the `{sum, count}` combined reducer into device Views (fences twice, §3.3); a host read of the count at every `setOpenness` (a sync per step, plus a transfer) |
| D-T5 | **Instance-scoped host reads** (S2): `deep_copy(CCExec(), pinned, dev)` + `CCExec().fence(label)` | the exec-less copy (two global fences) |
| D-T6 | **No global synchronization inside `step()` on device** (S4): exec-less device copies and fills become `deep_copy(CCExec(), …)`, `Kokkos::fence()` becomes `CCExec().fence(label)`, and no per-step View allocation | leaving them (≈ 10 µs drain each; they also defeat X) |
| D-T7 | **No CUDA graphs and no fused coarse V-cycle in this package** | Graphs: the register already rejected fill-then-sweep with graph capture (`docs/decisions/flow.md:3374`). The per-iteration host decision cuts every PCG iteration into a graph segment anyway, the FP32 V-cycle takes a host-side scale exponent per call, and Kokkos 5.1 graphs would mean restructuring every launch site. Worth ≤ 0.4–0.8 ms of the 1.24 ms of < 5 µs gaps. Re-measure after S4 (Q-5). Fused coarse V-cycle: §3.4 (B1's failure mode, and L2 is 0.15–0.2 ms in all). **Lagged or speculative stop test**: rejected by the same register entry (not bitwise), and a speculative iteration costs a whole V-cycle |
| D-T8 | **D-1 stands.** Re-examined: a multi-SM dense FP32 inverse applied as a GEMV (9.4 MB, ≈ 10 µs) still needs FP64 refinement (D-1 (e)), so the FCG becomes ≈ 20 launches per solve (≈ 50 µs), and the inverse must be rebuilt every step from the factor (≈ 0.1–0.2 ms). Net ≈ 0.15 ms/step better than B2, for a second engine and 9.4 MB more | reversing D-1 toward (a) dense, a vendor solver, or (d) deeper coarsening; no new evidence favours any of them |

## 5. The design (normative)

### 5.1 S1: the device fluid count and single-result means

- **Storage.** `Level` gains `Kokkos::View<long, CCMem> nFluidDev` (allocated with the level).
- **Count kernel.** New `noteFluidCellsDevice(Level&)`. On `!kHostMemory && !distributed_` it runs
  `ccReduce3("mgfluidcount", inner range, if (ac(i) > 1e-30f) k += 1, Kokkos::Sum<long, CCMem>(lv.nFluidDev))`.
  That is a single-result reduction into a device View, so it is asynchronous. The host branch of
  `noteFluidCells` (`:1833`) is unchanged.
- **Where it is called.** Wherever an AC is (re)built: in `setOpenness` (`:1871`), for level 0 and for
  every coarse level whose AC is built there (that is, not deferred), right where `noteFluidCells(c)`
  is called; and in `ensureFp64` for a level whose AC it builds. A deferred level's `nFluidDev` is not
  read before `ensureFp64` (deferral requires the 'fine' scope, where only level 0 removes means).
  Assert this in debug: host flag `lv.nFluidDevValid`.
- **`removeMean`, single-rank device branch** (`:5958–5972`):
  `ccReduce3Lanes("mgmeanr", …, [](x,y,z,double& s){ i = …; if (ac(i) > 1e-30f) s += ff(i); }, Kokkos::Sum<double, CCMem>(slot(kMsum)))`.
  It keeps the same predicate, addend, range and policy. The `mgmeans` subtract reads
  `lv.nFluidDev()` where it read `kc()`. The rest is unchanged, including `stopGuard`.
- **`exitFp32`, single-rank device branch** (`:4074`): the `joint` call becomes the same single-result
  sum, with the predicate `ac(i) > 1e-30f` (not the host's `all ||` shortcut). The body's `cnt` reads
  `l0.nFluidDev()` where it read `kc()`.
- **Other combined reducers.** Any other `parallel_reduce` / `ccReduce3Lanes` with ≥ 2 results that
  runs every step on the single-rank device path (P4 lists them) is converted by the same rule if it
  is in `mac_cutcell_mg.hpp` / `mg_fp32_vcycle.hpp`. Otherwise it goes into the S4 list.
  `kcnt_` stays for the paths that still use it.
- **Bits.** Expected bitwise by §3.3. If G-BIT fails, fall back to one custom reducer whose
  `value_type` is `struct {double s; long k;}` (the combined reducer's 16-byte value, join
  componentwise, `init` zeros), written to a device View: the same value size and the same joins as
  today. If that also fails, STOP and report. Default for the caller: land S1 as its own class-b
  commit right after B2, under the same re-baseline.
- **Distributed.** The distributed branch (host Allreduce) is unchanged; it is one sync per mean by
  MPI's nature. A CUDA-aware device Allreduce is a later option.

### 5.2 F1: the factor in shared memory (`mg_bottom_direct.hpp`)

**Scratch** (team level 0, FacReal = float), per launch:

| name | shape | role |
|---|---|---|
| `Sp` | 1-D, b(b+1)/2 FacReal; index `pk(i,j) = i(i+1)/2 + j` for j ≤ i | Σ's lower triangle (assemble, trailing updates), overwritten in place by L(i,j) when the reference stores L(i,j), then by W(i,j) when the reference stores W(i,j) |
| `Ls` | [b][kBottomTile+1] | as today (the tile column of L; aliased as `A` in the W phase) |
| `St` | [kBottomTile][kBottomTile+1] | as today |

For b = 128 that is (8256 + 2176 + 272) × 4 B = 42.8 KB, under the 48 KB default.

**Schedule selection.** `smemBytes(b)` is the Kokkos `shmem_size` sum of the three buffers. It is
chosen at hierarchy build and stored as `BottomDirect::smemFactor`. The shared-memory schedule is used
iff the backend is a device (`!kHostFactor`), FacReal = float, and
`smemBytes(b) ≤ TeamPolicy<CCExec>(CCExec(), 1, 1).scratch_size_max(0)`. Otherwise the current
global schedule runs. For b = 192 that is 74 KB + 13 KB, so the global schedule unless the backend
reports more. The `double` test instantiation always takes the global schedule. Both schedules give
the same bits, so this is a performance switch, not a numerics knob.

Test hook: `int forceGlobal` (U9).

**The mapping** (reference = today's `assemble`/`invert`, `:366–521`). Each reference statement is
copied **verbatim** and only its accessor changes:
- `D.Sg(i,j)`, `L(i,j)`, `W(i,j)` with j ≤ i → `Sp(pk(i,j))`.
- `D.L(i,i)` → `Sp(pk(i,i))`.
- Stores to `D.L` and `D.W` become stores into `Sp`. `D.Sg`, `D.L` and `D.W` are no longer touched
  on this path.
- Do not introduce temporaries that split an expression the compiler may contract. For example,
  never hoist `Ls(i,m) * Ls(j,m)` out of `acc += …`, and never precompute `e_i·Q`. FMA contraction
  must see the same expression trees.

Why in place is safe:
1. L(i,j) is written exactly once, from Σ(i,j), and Σ(i,j) is never read again: trailing updates only
   touch rows and columns ≥ j1 of later tiles.
2. In the W phase, row block i0's phase 1 reads L rows of its own block (still L) and W rows < i0
   (already W). Its phase 2 reads only `St` (the staged L tile), `A` and registers. A barrier
   separates the two (already present), so phase 2's writes of W(i, j ≤ i) over L are race-free.
3. Q = WᵀW reads W from `Sp` and writes `D.Q(k,i,j)` and `D.Q(k,j,i)` to global, as today.

Mode 1 still reads `D.Q(k−1,i,j)`, and mode 2 and `border()` read `D.Q` and `D.Y`, all from global
and unchanged.

Free (bitwise-neutral) choices:
- Which thread computes which item inside a phase. For example, map only the lower triangle in the
  trailing update and in Q, so no thread idles on j > i.
- The unroll factor.
- T. The factor is T-independent (U3), and `directTeam` keeps choosing `team_size_max`.

**F1b (conditional; P2 decides).** Do it if assemble costs ≥ 0.15 ms per factor after F1: today it
evaluates `entry()` (6 face evaluations, FP64) for every one of the b² pairs. The row form is:
- One thread per row i walks the six faces in the order −x, +x, −y, +y, −z, +z. It keeps the in-plane
  neighbour factor index `jf[f]` (or −1) and `sa[f] = si * af` (FP64) in registers.
- Then, for j = 0..i: `v = (i==j) ? 1.0 : 0.0; for f in 0..5: if (jf[f] == j) v += sa[f] * D.s(fj);`,
  then the aug test (a per-plane `compOfFactor[b]` int in scratch), then `if (i==j) v += delta;`,
  then the cast and the mode-1/2 adjustment, verbatim.
- `sa[f] * D.s(fj) + v` is the same tree as the reference's `si * af * D.s(fj)` added to `v`, since
  (si·af) is rounded first in both.

**Probe for P2.** Only under the compile definition `PECLET_FLOW_BOTTOM_PROBE`, set by the bench
target alone. Thread 0 records `Kokkos::Impl::clock_tic()` at each phase boundary into a member
`View<long long*>`. Production builds compile no probe code. It is not an env knob.

### 5.3 B2: bottom FCG v2 (class b; device instantiation only)

**Team.** `kBottomSolveTeam = 512`. The device launch (`launchDirectKernel`, `:4812`) uses
`TeamPolicy<CCExec, Kokkos::LaunchBounds<512, 1>>(1, min(512, team_size_max))`. The host keeps
`kBottomHostTeam` and its single-lane path, bitwise. Every output below comes from fixed work items
mapped by `TeamThreadRange`, so the bits do not depend on T.

**M plane products** (`BottomDirect::apply`, `:217–283`; device branch
`if constexpr (!std::is_same_v<CCMem, Kokkos::HostSpace>)`):
- `kBottomSeg = 4`. The segment length is `Lg = ceil(b/4)` and segment s covers
  j ∈ [s·Lg, min(b,(s+1)·Lg)).
- Scratch: `part[kBottomSeg][b]` FacReal (≤ 3 KB).
- Forward plane k:
  - **Phase A**: work items w ∈ [0, 4b) with `s = w / b`, `i = w % b`, so a warp holds consecutive i
    and the `Q(k,j,i)` loads coalesce. Each item computes
    `acc = 0; for j in seg(s) ascending: acc += Q(k,j,i) * (k==0 ? u(o+j) : (u(o+j) - e(k,j)*g(op+j)))`
    and stores `part[s][i] = acc`. Barrier.
  - **Phase B**: work items i ∈ [0,b):
    `g(o+i) = ((part[0][i] + part[1][i]) + part[2][i]) + part[3][i]`. Barrier.
- Backward plane k: phase A accumulates `Q(k,j,i) * (e(k+1,j)*g(on+j))`, and phase B sets
  `g(o+i) = g(o+i) - (((p0+p1)+p2)+p3)`.
- Border: the x̃_B phase is split the same way. The Y phase (oB outputs) and the cast and uncast stay
  as they are, one thread per output, single accumulator.
- Inside a segment the unroll is free (16 or 32, by measurement; bitwise-neutral).
- With T = 512 and b = 128 that is one item per thread and 16 warps × 32 loads × 128 B = 64 KB in
  flight: one L2 round trip per plane.

**Reductions** (`BottomKernel::teamSum` / `teamMax`, `mac_cutcell_mg.hpp:≈820–855`; device branch):
- `kBottomRedChunks = 64`. Chunk c covers q ∈ [c·Lc, min(n,(c+1)·Lc)), with Lc = ceil(n/64).
- Work item c accumulates its chunk in ascending q into `red[c]` (FP64 scratch). Barrier.
- Then `Kokkos::single(PerTeam, [&](double& v){ v = 0; for c in 0..63: v += red[c]; }, s)` broadcasts
  the result. For max, the same with max.
- `removeMeanBottom`, `dot`, `rmax` and the Polak–Ribière sum all go through these.
- Everything else in the FCG is unchanged: algebra, τ, cap, masking, the failure path, `info`.

**Expected cost** (2 inner iterations): M ≈ 27 µs (24 × ≈ 1 µs, plus cast, uncast and mean), the FCG
work outside M ≈ 11 µs per iteration, about 80 µs per solve in all (≤ 110 µs gate). That is
1.0 ms/step, against 2.65.

**Distributed.** With B2 the bits no longer depend on T. That removes the "same T" condition of §13
Q-D7 item 2 for a future redundant device bottom; the same GPU architecture and build suffice.

### 5.4 S2: instance-scoped packet reads

- `kpk_` and `kgp_` (`:2274–2275`) are allocated in `Kokkos::SharedHostPinnedSpace`, which is
  HostSpace on host builds.
- The packet read (`:2875`) becomes `Kokkos::deep_copy(CCExec(), kpk_, sub); CCExec().fence("peclet::flow::mg_packet");`.
- `readSlot` (`:2280`) uses a 1-element pinned View with the same pattern. Apply the same pattern to
  every other exec-less D→H read in `mac_cutcell_mg.hpp` / `mg_fp32_vcycle.hpp` on the per-step
  device path, including the guess packet.
- Host reductions into a host value (`maxabs`, `dot`) already fence only the instance: leave them.

### 5.5 S4: step-wide fence hygiene

**Census.** P4 produces the list of sync sites per step from nsys, attributed to NVTX ranges and code.
For each site in flow's `src/`:
- (a) A device↔device `Kokkos::deep_copy(dst, src)` becomes `Kokkos::deep_copy(CCExec(), dst, src)`.
- (b) `Kokkos::deep_copy(view, scalar)` on a device View becomes `Kokkos::deep_copy(CCExec(), view, scalar)`.
- (c) `Kokkos::fence()` becomes `CCExec().fence("<site label>")` where a host consumer follows. It may
  be deleted only where nothing on the host consumes device data before the next device operation;
  each deletion is justified in the commit message. When in doubt, convert.
- (d) A D→H read becomes S2's pattern.
- (e) A View allocated per step becomes a persistent member, sized at first use and kept.

Do not edit `src/vof/block_batch.hpp` (owned by the batch_stats package) or the curvature kernels.
List their sites in the log instead. MPI-path sites may also be converted: all device work outside
X's factor is on `CCExec()`.

**Gate G-SYNC** (ctest `device_sync_budget`, CUDA, not `bench`):
- 10 steps of a small single-rank VoF bubble case.
- In-process Kokkos Tools fence callbacks (`Kokkos::Tools::Experimental::set_begin_fence_callback`)
  count per step: global fences (the names "Unnamed Global Fence", "deep_copy … pre/post …",
  "scalar copy …"), and the combined-reducer fence names.
- Assert global ≤ the post-S4 measured value, and combined = 0.
- If the in-process callback API is unavailable in this Kokkos build, the same check runs in
  `gpu_diag/ana.py` as part of G-PERF (no ctest), and the log says so.

### 5.6 X: the factor on a side instance

**Hoist** (`flow_ibm_project.hpp`, `Solver::step()`):

```
68:  updateProperties();
     const bool hoisted = prefactorHoistEligible();
     if (hoisted) { projectBuildCoefficients(); mg_.prefactorBottom(); }
...
121: coeffBuiltThisStep_ = hoisted;        // was: = false (the reset must not undo the hoist)
127: applyBalancedForceProjection();        // its early build: add mg_.prefactorBottom() after it
```

- `prefactorHoistEligible()` = `varRho_ && !porous_ && outerIters_ == 1 &&
  !balancedForceActive() && mg_.directPrefactorEligible()`.
- `directPrefactorEligible()` = single rank, device backend, `bottomStore_`, the direct engine is
  what `directBottomSolve` will run, and the driver is PCG or FCG.
- For every other configuration nothing moves.
- **Audit before coding.** Between line 68 and `project()` (`:1145`), no code may write `rhoField_`,
  `rho1_`, `ox1_..oz1_`, `cx1_..cz1_` or the MG level-0 coefficients. No code may read the MG level-0
  `ox/oy/oz/AC/AF*` other than through the predicate `AC > 1e-30` (which ρ does not change). This
  covers `projectAssembleDivergence` and the incremental predictor's ∇Pⁿ in particular.
- If the audit finds a reader, the hoist point moves to just before that reader. The fallback point is
  right after the curvature block (`:84`): the window is then momentum alone, ≈ 1.85 ms, still more
  than the post-F1 factor.
- The audit also fixes the window's other end: the first `directBottomSolve`.

**Side instance** (`CutcellMG`, device only):
- Member `CCExec side_`, created lazily by
  `Kokkos::Experimental::partition_space(CCExec(), std::vector<int>{1})[0]` (on CUDA, a new stream;
  `Cuda/Kokkos_Cuda_Instance.hpp:335`). It is released with the other Views.
- If P3 shows that it serializes with the default instance, `side_` instead comes from a non-blocking
  stream: one 15-line factory function, `#if defined(KOKKOS_ENABLE_CUDA)` using
  `cudaStreamCreateWithFlags(…, cudaStreamNonBlocking)` + `Kokkos::Cuda(stream, ManageStream::yes)`,
  and a HIP twin.

**Methods:**
- `prefactorBottom()`: if not eligible, or `!facStale_`, return. Otherwise
  `CCExec().fence("peclet::flow::mg_prefactor_ready")` (the operator build has landed), launch
  `BottomFactorKernel` on `side_` (`launchDirectFactor` gains an exec-space argument), then set
  `facStale_ = false` and `facPending_ = true`.
- `joinPrefactor()`: `if (facPending_) { side_.fence("peclet::flow::mg_prefactor_join"); facPending_ = false; }`.
- Call `joinPrefactor()` first in `directBottomSolve` (`:4829`, before its `facStale_` test), and in
  `setOpenness`, `buildBottomStore`, every rebuild or redistribute of the hierarchy, every test hook
  touching `dir_`, and the destructor. Nothing may write the bottom operator or `dir_` while the side
  factor runs.
- The `[mg]` factor debug print moves after the join.

**Window rule.** Between the side launch and the join there must be no global synchronization. The
P4 census, after S4, must show zero `cudaDeviceSynchronize`, `cudaFree` and synchronous memcpy in the
window. Main-instance fences are fine: they do not wait on `side_`.

**Distributed.** Per rank the same, after H-6's bottom gather, once the device path gets the
redundant direct bottom (not here). Host backends: none; the lazy factor as today.

### 5.7 How each choice extends to the distributed device path

Today the distributed device path runs the host-scalar `solvePCG` with GraphAMG at the bottom, and
none of these WOs changes it. When the device path adopts the redundant global direct bottom (§13
Q-D7, H-6 on device):
- F1 and B2 apply unchanged.
- B2 lifts Q-D7's same-T condition.
- X launches after the gather.
- S1's device count is single-rank. Its distributed twin is a device Allreduce on `nFluidDev` with
  CUDA-aware MPI.
- S2 and S4 rules apply as written.

## 6. Work orders (in this order; one commit each; `opus-implementer`)

Set-up:
- Work in `suite/flow-gpucoarse`, with its own `build_cuda` and `build_omp`; copy the `-D` options from
  `suite/flow-projdist/build_{cuda,omp}/CMakeCache.txt`. Use cuda-13.4 and
  `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`.
- The gate scripts (`bubble_column_perf/gate_dist/{grun,gcmp}.sh`, `numcmp.py`,
  `gpu_diag/{run1.sh,ana.py}`, `page_update/bench_gpu.py`) are repointed to this worktree in a copy:
  `bubble_column_perf/gpucoarse/`.
- State file `bubble_column_perf/gpucoarse/STATE.md`, rewritten at every WO. Numbers go to
  `doc/vof_step_performance_log.md` (append).

- **WO-0: measure (no production change; ≈ 2 h).**
  - P1: inner FCG iterations per bottom solve. 30 steps from `ckpt_t100` with
    `PECLET_FLOW_MG_DEBUG=2` (debug printing only; raise the debug-solve limit if `mgDebugSolves()`
    caps it), giving a histogram, mean and max.
  - P2: the factor's phase split (assemble, diagonal tiles, panel, trailing, W, Q), from
    `bench_bottom_factor` walls-y on CUDA with `PECLET_FLOW_BOTTOM_PROBE`; cycles are scaled to the
    measured ms. Also the solve's M share: time `launchDirectKernel(precondOnly)` against the full
    solve in the same bench.
  - P3: a 40-line two-instance concurrency probe. One team runs ≈ 3 ms on `partition_space`'s
    instance while 200 large kernels run on the default instance. Pass if the wall time ≈ max, not
    the sum.
  - P4: the sync census from nsys sqlite, per step. For each `cudaDeviceSynchronize`,
    `cudaStreamSynchronize`, `cudaFree`, `cudaMalloc` and synchronous `cudaMemcpy`: its NVTX range
    and preceding kernel, mapped to the source line. Flag those between `updateProperties` and the
    first `mg_bottom_direct`.
  - P5: the baseline. `bench_gpu.py` interleaved fp64/auto, 3 × 200 steps, warm 50, quiet GPU
    (`nvidia-smi` utilization 0 before start), minimum of the rounds.
  - *Accept:* P1–P5 in the log. G-PERF targets are restated from P2 and P5. If P1 shows a mean ≥ 3
    inner iterations, record it: B2's gain then rises.
- **WO-S1** (§5.1), class a. *Accept:* G-BIT on CUDA and host. Combined-reducer fences ≤ 2/step (from
  80.8) in `ana.py`, fp64 and auto. Host syncs/step reduced by ≥ 50. G-PERF: step −0.15 ms or better.
- **WO-F1** (§5.2), class a.
  - *Accept:* U9. G-BIT on CUDA and host. `bench_bottom_factor` dump `cmp`-identical to the
    `b820652` build on walls-y and periodic.
  - `mg_bottom_factor` ≤ 1.2 ms/launch (expected 0.4–0.7); step ≤ parent − 1.8 ms.
- **WO-B2** (§5.3), class b.
  - *Accept:* U10, G-NUM (§7), the CUDA `state_hash` re-baselined in the commit (old → new table), and
    the register entry R-1 drafted.
  - `mg_bottom_direct` ≤ 110 µs mean per launch (expected 75–90); step ≤ parent − 1.1 ms.
- **WO-S2** (§5.4), class a. *Accept:* G-BIT. "copy between contiguous views" fences per step reduced
  by ≥ 12. Packet-read latency in nsys ≤ parent's.
- **WO-S4** (§5.5), class a. May be split by file. A `sonnet-implementer` may do the mechanical edits
  from the census list, reviewed on Opus. *Accept:* G-BIT, G-SYNC. `cudaDeviceSynchronize` ≤ 5/step,
  and none in X's window outside the files excluded in §5.5.
- **WO-X** (§5.6), class a. Do it only if P3 passes (or its fallback stream does) and the window census
  is clean after S4.
  - *Accept:* G-BIT (the hoist must be bitwise: same coefficients, same factor), G-OVL, and the
    batteries.
  - Step ≤ parent − 0.8 × the post-F1 solo factor time.
- **WO-F1b** (§5.2), class a. Only if X is not landed and P2 shows assemble ≥ 0.15 ms/factor after F1.
  *Accept:* U9, G-BIT.
- **WO-D**:
  - Register entries R-1..R-5 (§9), the CLAUDE.md "Pressure solve" bullets (the side instance, the
    fence rule, the test counts), and the log.
  - Close the state file.
  - Hand off: final `bench_gpu.py` numbers against §0.

## 7. Verification gates

- **G-BIT** (class a; the parent design's §8 G-BIT, every item):
  - `grun.sh <module> cuda` and `grun.sh <module> omp` against the parent commit's outputs via
    `gcmp.sh`. This covers `state_hash` all cases + np2, and 50-step bubble-column dumps from
    `ckpt_t43` at rtol 1e-8 and 1e-10 (u v w p C bitwise, per-step iterations identical).
  - `ctest -LE bench` green on CUDA and host (247 registered / 243 non-bench at `b820652`, plus the
    new tests), `ctest -R '_np[0-9]+$'` green, and np 1 ↔ single-rank memcmp tests unchanged.
  - Where operator storage is touched, the `PECLET_FLOW_OPERATOR_DOUBLE=OFF` tree's hashes are
    unchanged.
- **U9** (ctest `bottom_direct`, CUDA): on every U-problem of §13.8 A (walls-y is the bubble
  column's bottom shape; the real operator is covered by G-BIT's dumps), the factor storage (Q, Y, e, s, stat) of the
  shared-memory schedule is **bitwise** equal to the global schedule (`forceGlobal = 1`), for
  T ∈ {128, 256, 512, T_max}. U1–U8 still pass.
- **U10** (ctest `bottom_direct`, CUDA): the full bottom solve x and M(r) are bitwise equal for
  T ∈ {64, 128, 256, 512}. U2 (relative residual ≤ 1e-3), U4 (≤ 3 iterations), U6 and U7 (≤ 1e-3
  against host FP64 dense) pass with B2's kernel. The host instantiation's outputs are bitwise
  unchanged against `b820652`.
- **G-NUM** (B2; parent §8 G-NUM, via `numcmp.py` against `b820652`):
  - (1) the max relative difference over u v w p C after 50 steps from `ckpt_t43` is ≤ N50_8 =
    1.48e-8 at rtol 1e-8 and ≤ N50_10 = 2.52e-9 at rtol 1e-10. Expected ≤ 1e-10 and ≤ 1e-12.
  - (2) per-step outer iterations within ±1, and the 50-step total within ±2 % (expected identical).
  - (3) `max_open_divergence_projected()` ≤ 2× per step.
  - (4) static drop max|u| within 5 %; Hysing 1 peak rise velocity and its time within 0.2 %.
  - Also: inner FCG iterations max ≤ 3 and mean ≤ 2.5 over 300 steps, shift restarts 0, the flag
    never set, two runs bitwise identical, and host G-BIT bitwise. Run in both `'fp64'` and `'auto'`.
- **G-OVL** (X; nsys, 30 steps, sqlite):
  - The `mg_bottom_factor` kernel runs on the side stream and overlaps main-stream kernels for ≥ 90 %
    of its duration.
  - Zero global syncs between its launch and the join.
  - The join fence returns in ≤ 20 µs, so the factor is already done.
- **G-SYNC** (S4): §5.5.
- **G-PERF** (every WO): `bench_gpu.py` interleaved fp64/auto, 3 × 200 steps, warm 50, from
  `ckpt_t100`, quiet 5080, minimum of the rounds. Plus nsys 30 steps → `ana.py`: kernel ms/step,
  launches, fences by label. Report against §0. Noise is ≈ ±0.1 ms: a WO whose expected gain is
  below 0.2 ms reports its nsys counts instead of a wall gate.
  **Package gate: `'auto'` ≤ 14.8 ms/step and `'fp64'` ≤ 17.8** (the pessimistic end of §0). A miss by > 1 ms against §0 goes to
  Opus for analysis; the implementer does not tune around it.

## 8. Risks and open questions (each has a default; the implementer decides none of them)

- **Q-1 [pref]: B2 amends §13.4.4 and §13.2.** These are "one thread per output, ascending j, a single
  accumulator" for M, and "the FCG's dots stay team reductions, deterministic for a fixed T", on
  device only. It stays inside D-1 (block-tridiagonal FP32 direct factor preconditioning the FP64 FCG
  to τ), and it is a recorded change. *Default:* proceed, register entry R-1. If the user reads
  §13.4.4 as part of D-1 itself, B2 waits and everything else proceeds.
- **Q-2 [fact]: S1 bitwise.** It rests on §3.3's reduction-tree argument. *Default:* the fallback chain
  in §5.1. The decision to re-baseline goes to the caller, not the implementer.
- **Q-3 [fact]: F1 bitwise.** It depends on identical FMA contraction of the moved expressions.
  *Default:* if U9 fails, find the differing statement (the reference is verbatim). If it is
  irreducible, STOP and report. The caller's default is then to accept F1 as class b with G-NUM:
  the factor is an FP32 preconditioner inside a τ-converged FCG.
- **Q-4 [fact]: P2 might show the factor is not latency-bound** (for example the diagonal tiles
  dominate). *Default:* F1 as specified still lands if U9 passes and it saves ≥ 0.5 ms. Otherwise
  report P2 to Opus before WO-B2.
- **Q-5 [pref]: CUDA graphs or a launch-recording layer for the V-cycle.** It would recover ≤ 0.4–0.8
  ms of < 5 µs gaps. *Default:* not in this package. After S4, re-measure the gap histogram and bring
  a separate architect question only if > 1 ms of < 5 µs gaps remains inside the projection.
- **Q-6 [pref]: next levers outside this package.**
  - Device versions of the host Krylov fusions C1–C3 (matvec + p·Ap, update + Σr, mean-subtract +
    max), about −0.4 to −0.5 ms. Trap: the update kernel runs over the full extent including ghosts,
    while the sums run over the inner MDRange, so a fused kernel must keep both ranges.
  - Velocity-MG residual reads (`vmg_resid_max`, 9/step).
  - *Default:* separate packages; the caller schedules them.
- **Q-7 [pref]: twisted (two-ended) block factorization.** M phases 24 → 12 (≈ −0.3 ms solve), and the
  factor's critical path halves. It changes D-1's elimination order, so it needs the user. *Default:*
  not now.
- **Q-8 [fact, needs billed H100 time → user]: an FP64 factor** (§13 Q-D2). On the 5080 an FP64 M
  apply costs ≈ 76 µs on one SM's FP64 pipe: rejected there. *Default:* unchanged, `float`.
- **R-1 [fact]: the side kernel slows down when it shares an SM** with momentum kernels (latency-bound
  warps competing for issue and L1). G-OVL measures it. With F1 the factor is ≈ 0.5 ms against a
  ≈ 3 ms window, so even a 2× slowdown is hidden.
- **R-2 [fact]: global syncs in the window** come from code the implementer may not edit
  (`block_batch.hpp`, curvature). *Default:* move the hoist point after them (§5.6). If no clean
  window ≥ 1.5× the post-F1 factor remains, skip X and report.
- **R-3 [fact]: scratch limits on other GPUs.** H100 227 KB, MI250X 64 KB LDS; b = 128 needs 43 KB.
  The rule in §5.2 falls back automatically. HIP bits are not pinned locally.
- **R-4: `side_` lifetime.** It is a Kokkos handle and must be released before `Kokkos::finalize`,
  like a View. Release it where the hierarchy's Views are released.

## 9. Register entries (drafted in WO-D; the caller adds them to `docs/DECISIONS.md` / `docs/decisions/flow.md`)

- **R-1** Device bottom FCG v2: M's plane products in 4 fixed j-segments, reductions in 64 fixed
  chunks, T = 512. The bottom's bits are independent of T. *Rejected:* T-dependent team reductions
  (bits follow the compiler's register allocation); bitwise high-unroll M (changes T anyway);
  multi-SM apply (cross-SM sync ≥ 2.5 µs per phase against ≈ 1 µs). *Evidence:* WO-B2 log.
- **R-2** The device direct factor runs in team shared memory, Σ → L → W in place, bitwise to the
  global schedule, which stays the fallback above the scratch limit. *Rejected:* lagged factor;
  cyclic reduction / SPIKE; dense or vendor (D-1).
- **R-3** Single-rank device mean removal uses a per-level device fluid count. No per-iteration
  reduction has two results (Kokkos 5.1 combined reducers fence twice). *Rejected:* `{sum, count}`
  into device Views.
- **R-4** On device, `step()` holds no global synchronization: instance-scoped copies, fills and
  fences, pinned packets. Guarded by `device_sync_budget`. *Rejected:* exec-less `deep_copy` /
  `Kokkos::fence()` in the step.
- **R-5** Variable-ρ single-rank device: the coefficients are built right after `updateProperties()`,
  and the bottom factor runs on a side execution instance, joined before the first bottom solve.
  *Rejected:* lazy factor at the first bottom solve; CUDA graphs in this package; native events.
