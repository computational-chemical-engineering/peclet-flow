# Distributed host pressure solve: the communication and synchronization structure

Design note, 2026-10-18. Branch `distcomm` (worktree `suite/flow-distcomm`, from flow main
`b820652`). Core work goes on branch `distcomm` of a core worktree `suite/core-distcomm`, from core
main `2398dc7` (v1.5.0). Answers `bubble_column_perf/dist_cpu/BRIEF_ARCH_CPU.md`. Implementer:
`opus-implementer`, who has seen neither the brief nor the conversation.

Evidence:
- `bubble_column_perf/dist_cpu/s2t/summary_28257785.txt`: genoa job 28257785. EPYC 9654,
  cores 0–23 = one NUMA domain = 3 CCDs of 8 cores. Bubble column 128×96×64, rtol 1e-8,
  10.29 PCG iterations/step, 10 V-cycles/step.
- The previous CPU package: `dist_cpu/{DIAG,IMPL,FIX}_REPORT.md`.
- Its design: `doc/vof_step_performance_design.md` §14, cited as "§14".

Anchors are `file:line` at flow `b820652` / core `2398dc7`.

## 0. Outcome first

All three work orders are **class a**: proved bitwise at every np against base at the same np, and
np 1 against single rank. There is no recorded numerics change in this package. Expected genoa
ms/step:

| WO | change | where | 8×3 close | 24×1 | np 1 (`init_mpi`) |
|---|---|---|---|---|---|
| 0 | attribution: trace tool, halo microbenchmark, local reproduction | tools | 0 | 0 | 0 |
| K1 | GridHalo host fast path: self-copy fused into pack, calling-thread copies below a cutoff, persistent requests | core | −4 … −7 (all GridHalo users) | −2 … −3 | −1 … −2 |
| K3 | stencil halo: faces only, rank-local periodic axes read in-kernel | core option + flow | −2 … −4 | −4 … −8 | −8 … −12 |
| K2 | communication-avoiding coarse smoothing with wall/slip/inflow BCs; the post-smooth reuses the ring rhs | flow | −2.5 … −3.5 | −2 … −3.5 | 0 |
| K4 | distributed host Krylov fusion (C2, C3, C4 mirrored) | flow | −1 … −2 | −1 … −2 | −2 … −3 |
| K5 | (conditional) on-node shared-memory halo transport | core | −3 … −6 | −4 … −8 | 0 |
| D | register, docs, state | — | 0 | 0 | 0 |

Total without K5 → with K5:

| layout | today | without K5 | with K5 |
|---|---|---|---|
| 8×3 close | 104.2 | ≈ 89 (84–95) | ≈ 84 (78–90) |
| 24×1 | 108.4 | ≈ 95 (90–100) | ≈ 89 (82–96) |
| np 1 (`init_mpi`) | 87.1 | ≈ 73 (70–77) | ≈ 73 (70–77) |

**The brief's 1.1× of 1×24 single (74.5 ms) is not reached at 8×3 close or 24×1 by this package.**
§3.5 itemizes the remainder. Of the 35 ms of non-kernel time, about 12 ms lies outside the projection
(velocity MG, field and VoF exchanges); WO-0 attributes it into a follow-up.

Two further terms are not communication:
- 24×1's redundant single-thread bottom costs 10.45 ms, against 3.17 ms single-rank.
- 24×1's uneven z blocks (16/24/24) and 8×3 close's two L3-straddling ranks leave imbalance.

np 1 reaches ≈ 1.08×.

## 1. Problem and scope

Per step at 8×3 close, the projection costs 65.3 ms against 34.3 single-rank:
- Compute kernels are equal: ≈ 31.4 against 31.0.
- The difference is halo kernels (9.47 ms, 1119 launches), halo MPI time (post 1.9, wait 11.6),
  allreduce (2.1), and ≈ 7–9 ms of host time between kernels that nothing attributes.
- Outside the projection, kernels are cheaper distributed (−8 ms), but the stages are +3.9 ms. That
  hides ≈ 12 ms of non-kernel time (§3.4).

**In scope:**
- The pressure MG's halo exchanges and their count, structure and transport.
- The distributed host-scalar PCG's Krylov passes.
- The attribution of the non-kernel time.
- GridHalo itself (core), whose improvements reach every user.

**Out of scope:** the device path (its own note), AMR, curvature, the gallery, the velocity MG's own
algorithm (Q-6), the bottom engine (Q-5), and decomposition or layout policy (Q-4).

## 2. Constraints and invariants

1. **Bits.**
   - Serial ↔ np 1 bitwise: `init_mpi` at np 1 runs the distributed host-scalar `solvePCG`, and it
     stays memcmp-equal to single rank.
   - np > 1: every WO here must be bitwise against base **at the same np**, which is stronger than
     the G-NUM-H tolerance class.
   - Single-rank results are untouched. Defaults are unchanged in bits.
2. **The np 1 distributed path stays the distributed code** (§14 Q-H2: the np 1 gate needs it).
   Speed it up inside the distributed path; never by routing np 1 to single-rank engines.
3. **No environment variable changes a result.** Kill switches for transports are allowed only if
   they are bitwise-neutral and debug-only, like `PECLET_CORE_GPU_AWARE_MPI`.
4. **MPI.** Collectives stay rank-consistent: every new collective happens at topology/hierarchy
   build, where all ranks already participate. MPI is called from the master thread only
   (FUNNELED).
5. **Core first.** A core change lands in core, is tagged and published **before** flow consumes it
   (suite rule). Tagging is outward-facing: Q-2. Locally, flow builds against the core worktree with
   `-DPECLET_SIBLING_PECLET_CORE=$PWD/../core-distcomm` (`cmake/PecletDeps.cmake:158–161`).
6. **Device path.** K1's fused gather and persistent requests, and K2's and K3's CutcellMG paths,
   run on device builds too and must be bitwise there. The serial-copy cutoff, K4 and K5 are host-only
   (`if constexpr` on the memory space).
7. **Register.**
   - "Domain-BC paths stay blocking (VelocityMG precedent)" (`docs/decisions/flow.md:747`, undated,
     no reason given) is already not true of the pressure MG: its distributed smoother and residual
     overlap on every path (`mac_cutcell_mg.hpp:4290–4307`, `:3545–3556`). K2 touches this entry
     (Q-1).
   - "Refresh the MG halo before the residual" (DECISIONS.md:103) is kept: K2 does not drop that
     exchange.

## 3. Diagnosis

**3.1 The exchange count.** Per PCG iteration there are 31 exchanges, from the code:
- the matvec: 1 (L0);
- L0: pre-smooth 2 sweeps × 2 colours = 4, the residual 1, post-smooth 4;
- L1 and L2: 4 + 1 + 4, plus 1 fill before prolongation (`vcycleImpl`, `:3507–3606`);
- L3 (the replicated direct bottom): 1 fill.

That gives 1 + 9 + 10 + 10 + 1 = 31, and 31 × 10.29 ≈ 323 per step (measured 323). The
communication-avoiding (CA) ring would cut L1/L2 smoothing, but it is off with any domain BC
(`v.g = … !hasBC_ …`, `:1494`; `caSmooth`, `:3633`). The code's reason: ring rows "would need
post-BC ghost openness the exchange does not deliver" (`:3628–3632`).

**3.2 Each exchange costs 71 µs at 8×3 close** (post 6.0 + wait 36.0 + kernels 29.3). It costs
36.6 µs per-CCD and 67.7 µs at 24×1. Three structural reasons:
- **Kernels.** `GridHalo::exchangeBegin` (`core/include/peclet/core/halo/grid_halo.hpp:194–236`)
  launches selfCopy and pack as two parallel regions. `exchangeEnd` launches unpack. On a 3-thread
  rank each launch is ≈ 8.5 µs, mostly fork/join, for copies that are a few hundred elements on L2
  and L3. 1119 launches/step.
- **No overlap.** The begin/interior/end split does not overlap on host. Shared-memory MPI
  (OpenMPI ob1/sm-vader or UCX) moves a message above its eager limit by rendezvous: the receiver
  copies (CMA/XPMEM) and acknowledges inside an MPI call. Nobody is inside MPI while all threads run
  the interior sweep, so the transfer happens in `exchangeEnd`, with both neighbours present.
  - L0 messages at 8×3 are 16–33 KB, so rendezvous. L1 is 6–8 KB, around the limit.
  - The wait is therefore (neighbour skew) + (handshake) + (single-threaded copy).
  - This is a hypothesis with a mechanism. M2 in WO-0 confirms it; §6.
- **26 neighbours.** GridHaloTopology fills every ghost cell, edges and corners included
  (`grid_halo_topology.hpp:3–8`). The 7-point smoother, residual and matvec read only face ghosts
  (`mac_pressure.hpp:475–510`: `i ± s_a`).
  - 8×3 (4×2×1 blocks): 5 distinct partners per exchange where faces need 3.
  - 24×1: ≈ 17 where faces need ≈ 5.
  - Every extra partner adds posts and, worse, another neighbour whose lateness the rank waits for.
    That is a max over more random arrivals. It is the likeliest reason 24×1 waits as long as 8×3
    with a third of the data.

**3.3 np 1 under `init_mpi`** (+19 ms against single rank):
- every exchange is a self-copy (373 launches, 6.3 ms);
- the smoother runs as box interior + shell (400 launches, 17.3 ms, against 240 and 9.4);
- the Krylov loop runs unfused.

The single-rank path avoids all three: A3 in-kernel wrap, and C1–C4.

**3.4 The unattributed non-kernel time** (8×3 close):
- Step non-kernel 35.0 ms. Attributed: halo wait 11.6, post 1.9, allreduce 2.1.
- Of the rest, ≈ 7–9 ms is inside the projection (≈ 5 µs per launch between kernels; single rank
  shows ≈ 2.5 µs per launch). ≈ 12 ms is outside it: momentum +2.9 ms of stage time with
  ibm/momentum kernels 2.5 ms cheaper, "other" +0.9 with kernels 3 ms cheaper, block_advect equal
  with VoF kernels 2.5 ms cheaper. That is velocity-MG exchanges and residual allreduces, field
  ghost exchanges and VoF container messages, none of which section C counts.
- The kernel timer brackets the parallel region, so fork/join is *inside* kernel time, not in this
  remainder. At 24×1 (1 thread/rank) the projection still has ≈ 10 ms unattributed.
- WO-0's trace (§6 M1) separates the hypotheses: launch overhead, MPI outside the MG timers,
  allocation/page faults, and skew.

**3.5 What remains after the package** (8×3 close, ≈ 89 ms):
- kernels ≈ 68;
- L0 halo copies ≈ 3 (real work: 16–33 KB faces);
- per-exchange wait at the post-K3 count, ≈ 260 × 20–30 µs ≈ 6;
- allreduce ≈ 1.5;
- host gaps ≈ 4;
- outside-projection communication ≈ 8–10, reduced only by K1/K5.

The reference that would show "communication-free" cost on the same memory system is 1×24 single
(67.7). The per-CCD layout (68.9) is not that reference: each rank there has its own 32 MB L3 for a
≈ 24 MB working set, so its kernels are 18 ms faster.

## 4. Decisions

| item (brief §6) | decision | rejected, and why |
|---|---|---|
| fewer exchanges | **K2**: CA (g = 2) on 0 < L < bottom with wall/slip/inflow BCs (no outflow face), ring openness re-exchanged after the BC re-imposition, and the post-smooth reusing the pre-smooth's ring rhs. 31 → 25 exchanges per iteration | **CA on L0** (g = 2 at level 0): the solver's g = 1 bridge (openness/rhs/φ staging, ghost-projection staging) assumes g = 1 there, so it is too invasive for ≈ −3 ms. **Deep CA** (g = 4, one exchange per smoothing phase): ≈ −2 ms after K5 for a shrinking-ring generalization and 1.5× coarse smoothing work; Q-3. **Dropping the residual or prolong refresh**: the register keeps the residual refresh (DECISIONS.md:103) and the prolong fill is required. **Aggregating the setOpenness ox/oy/oz exchanges**: 8/step, ≈ 0.5 ms |
| merging across sweeps/colours | that *is* CA. Two colours of one sweep cannot share an exchange at g = 1 | — |
| deeper coarse halos | not now (Q-3) | see above |
| overlap | **today's split does not overlap on host** (§3.2). The fix is the transport: K5, conditional on M2 | **MPI progress inside the interior sweep** (chunked sweeps with `MPI_Testall`): more fork/joins, and the copy is still single-threaded. **A progress thread / MPI_THREAD_MULTIPLE**: fragile with Kokkos OpenMP, implementation-dependent. **Eager-limit tuning** (`btl_sm_eager_limit`, `UCX_RNDV_THRESH`) as the production fix: site- and version-specific and two copies; kept as an M2 diagnostic |
| messages per exchange | **K3**: stencil exchanges (smoother off-CA, residual, matvec) use a faces-only halo | keeping 26-neighbour fills for 7-point stencils |
| np 1 path | **K3**: rank-local periodic axes (one block along the axis, even inner extent, no outflow face) read the wrap in-kernel. At np 1 the stencil exchange vanishes and the box kernels run one full-range launch, still the distributed code path (constraint 2) | routing np 1 to the single-rank engines (§14 Q-H2: no) |
| GridHalo cost | **K1**: one gather launch (self-copy + pack), calling-thread copies below `kHaloSerialCopy` elements on host, persistent requests | merging unpack into the shell sweep (an indirection per ghost read in every stencil); `MPI_Neighbor_alltoallv` (blocking completion, no gain on one node) |
| Krylov | **K4**: mirror C2 (update + Σr), C3 (mean subtract + max) and C4 (V-cycle L0 exit subtract + r·z) in the distributed host-scalar loop, host only. Same lanes and the same per-rank partials, then the *same* Allreduce calls (incl. the 2-double `allreduceSum2`) | **C1** (matvec + p·Ap): the box matvec splits rows between interior and shell, so the lane order cannot be kept. **Merging max\|r\| into the Σr reduction** (max/min of r before the subtract give max\|r−m\| exactly, but MAX and SUM in one call need a user op or a rank-ordered Allgather, which changes the np > 1 sum order): class b for ≈ 0.4 ms. **Pipelined CG**: class b and less stable |
| unattributed time | **WO-0**: a per-rank event trace (Kokkos Tools + PMPI) on one node-wide clock, gap attribution, and a skew/protocol split per exchange | guessing |
| transport | **K5 only if** WO-0/M2 shows a protocol share (wait minus neighbour lateness) of ≥ 10 µs per exchange after K1/K3, on the workstation's straddled layout or at genoa | building it unconditionally |

## 5. The design (normative)

### 5.1 K1: GridHalo host fast path (core `grid_halo.hpp`)

- **(a) One gather launch.** `exchangeBegin` runs one `parallel_for` over `[0, nSelf_ + nSend_)`:
  `i < nSelf_` → `f(selfDst(i)) = f(selfSrc(i))`, else `sendBuf(i−nSelf_) = f(sendIdx(i−nSelf_))`.
  - It is race-free because self-copy writes ghost cells only and pack reads inner cells only, for
    any ghost width.
  - Label `peclet::core::halo::gather`.
  - On every backend.
- **(b) Calling-thread copies** (host memory space only). Constant
  `inline constexpr Index kHaloSerialCopy = 2048`, which M2 fixes before commit (the choice is
  bitwise-neutral).
  - If `nSelf_ + nSend_ ≤ kHaloSerialCopy`, the gather is a plain loop on the calling thread, with
    no parallel region. The same rule applies to unpack with `nRecv_`.
  - Device builds are unchanged.
- **(c) Persistent requests.**
  - In `init`, after the buffers exist: `MPI_Recv_init` / `MPI_Send_init` per neighbour, on the
    buffers that the current `gpuAwareMpi()` choice selects.
  - The choice is cached at init. A later flip re-inits the requests.
  - `exchangeBegin` → `MPI_Startall`; `exchangeEnd` → `MPI_Waitall`; free them in the destructor.
  - A call with `tag != 0` keeps the `Isend`/`Irecv` path (flow passes none; check with grep).
  - The `PECLET_CORE_HALO_TIMEOUT` diagnostic keeps working: test-and-poll on the persistent
    requests.
- **Bits:** identical bytes moved. **Gate:** G-HALO (§10).

### 5.2 K3: the stencil halo (core option + flow)

**Core.** `GridHaloTopology::buildTopology(dec, rank, g, per, comm, Neighborhood nb = Neighborhood::Full)`.
`Faces` keeps a ghost cell only if exactly one coordinate lies outside the inner block. Existing
callers are unchanged.

**Flow** (`mac_cutcell_mg.hpp`, hierarchy build `:1476–1500`):
- Each distributed level gets `Level::sdev` (`GridHalo<double>`), and `sdevf` (`GridHalo<VReal>`) on
  FP32 levels. Both are built with `Neighborhood::Faces` and with `per` cleared on the level's
  **wrap axes** `M_L`.
- Axis a ∈ M_L iff all of:
  1. the level's decomposition has exactly one block along a (decided from the global
     decomposition, so every rank agrees);
  2. no outflow face on a;
  3. `lv.inner[a]` is even (red-black consistency across the wrap; the A3 condition);
  4. `!overlaySolve_` at solve time. If an overlay is active, use `dev` and `wrapMask = 0`.
- `dev` (full) stays for prolong fills, operator/openness fills and CA smoothing.

**Kernels.** The box kernels gain `int wrapMask` (default 0, which is today's code bit-for-bit):
`cutcellSmoothColorBoxFace`, `residualCutcellBoxFace`, `applyCutcellOpExactBox`,
`applyCutcellOpBoxFace` and the FP32 box twins. For a masked axis a:
- the interior box spans `[g, ext_a − g)` on a, so a contributes no shell;
- the **field operand** reads along a use the wrapped index, as A3's `ccWrapNbrs` does on that axis:
  at the low inner face `i − s_a → i + (n_a − 1) s_a`, at the high face `i + s_a → i − (n_a − 1) s_a`.
  Coefficient and openness reads stay as today; their ghosts are filled by `dev` at operator build.

**Call sites.**
- The distributed branch of `smooth()` (non-CA), the residual's distributed branch, the
  distributed `matvecOverlap`, and the FP32 distributed twins exchange on `sdev`/`sdevf` with
  `wrapMask = M_L`.
- If `M_L` is all three axes (np 1), skip the exchange calls and run the box over the full inner
  range in one launch per colour.

**Bits.** Every stencil reads the same values: a face ghost holds exactly the value its owner
holds, and a wrapped read reaches the same cell. Edge and corner ghosts are never read by these
kernels. **Gate:** G-BIT-D plus `mg_stencil_halo_np{1,2,4}`, which compares a test hook forcing
`dev`/`wrapMask = 0` against the default, bitwise.

### 5.3 K2: CA coarse smoothing with wall BCs (flow)

- **Eligibility** (`:1494`): `v.g = (L > 0 && L + 1 < nLevels && (caMode_ & kCaMg) && minBlockExtent(dec) >= 4 && (!hasBC_ || !hasOutflow_)) ? 2 : 1`.
  - The bottom level keeps g = 1, because the direct bottom requires ghost width 1 (§13.4.7).
  - For `!hasBC_`, the rule `L > 0` applied to every level including the bottom. That case must
    stay byte-identical, so keep `L + 1 < nLevels` out of the `!hasBC_` branch: today's periodic
    behaviour is untouched.
  - `caSmooth(lv)` = `distributed_ && lv.caOk && !hasOutflow_ && M_L ≠ all axes`. When every axis
    is masked, K3's exchange-free path is cheaper.
- **Ring openness after the BCs.** In `setOpenness`, on every level with g = 2 and `hasBC_`, run the
  full `fill` of `ox`, `oy`, `oz` once more *after* `applyBoundaryOpenness(c, &fin)`. The ring rows'
  operator, built over `c.g − 1` ghost layers, then sees post-BC openness identical to the owner's.
  This touches no level that exists today with BCs.
- **Ring rhs reuse.** CA `smooth()` exchanges `lv.rhs` at entry. Add `lv.rhsRingFresh`: set it after
  that exchange, and clear it wherever `lv.rhs` is written (the restriction into `cs.rhs`, the
  telescope paths, `setOpenness`). Skip the exchange when it is set. The post-smooth then reuses the
  pre-smooth's ring (the same values). This applies to the periodic CA path too, also bitwise.
- **Count:** CA levels go from 10 exchanges per V-cycle to 7 (rhs ring 1, pre 2, residual 1, post 2,
  prolong fill 1). With L1 and L2: 31 → 25 per iteration.
- **Bits.** The CA scheme is bitwise to the per-colour exchange by construction (`:4259–4266`). With
  walls the new elements are the post-BC ring openness and zero-openness wall faces (weight 0 on the
  periodic image beyond the wall).
- **Gate:** the existing setter makes a direct proof possible:
  `diagnostics.set_comm_avoiding('off')` against `'both'` on the same build, bubble column and the
  solids battery, np 2/4/8, bitwise (new ctest `mg_ca_walls_np{2,4}`), plus G-BIT-D.
  A sign-of-zero difference is possible in principle (0·image summed into a zero). If the gate shows
  any difference, STOP and report.
- **Odd periodic extent (2026-10-26, in force for every CA path, K2 included).** The bitwise
  argument above needs each ghost-ring slot to carry its image's red-black parity. Both CA pairs
  sweep the ring in colour 0 with the slot's OWN parity (`parityOg` / `og_`); across a periodic
  wrap of ODD global extent the slot (z = −1, image of z = nz − 1) has the opposite parity, so the
  ring was swept in the wrong colour: np 1 ≠ single rank (M10 128×128×25: max|Δp| 7.37e-7 of 820,
  CA off bitwise; `~/Codes/bubble_column_perf/gpu_diag/mgd/WO7_DIAG.md` §D1). CA is therefore off on an
  MG level, and on the velocity decomposition, with any periodic axis of odd GLOBAL extent (global
  dims only, so np-independent); those take the per-colour exchange, exactly like the domain-BC
  levels. Even extents are bitwise unchanged. Rejected: correcting the ring parity in the kernels
  (more code in a hot kernel for a class with no production case; the per-colour path is the
  validated reference). Gate: ctest `ca_odd_periodic_mpi_np{1,2}` (32×32×25).

### 5.4 K4: distributed host Krylov fusion (flow, `kHostMemory` only)

- In the distributed host-scalar `solvePCG` (and `solveFCG` where it shares the helpers), port:
  - **C2**: the update `x += αp, r −= αAp` over the inner box's rows, accumulating the fluid Σr and
    the local count in `ccReduce3Lanes` order. Ghost entries of x and r are no longer updated; check
    by grep that nothing reads them before the next exchange (§ C amendment 4 of
    `doc/vof_projection_cost_design.md`).
  - **C3**: the mean subtract fused with the local max|r|.
  - **C4**: the V-cycle L0 exit subtract fused with the local r·z.
- Then the **same** MPI calls as today: `allreduceSum2(sum, count)`, the MAX Allreduce and the SUM
  Allreduce, so the global values are bitwise unchanged at every np.
- C1 is not ported (§4).
- **Gate:** G-BIT-D; np 1 stays bitwise to single rank (single rank already runs the fused kernels,
  which are bitwise to the unfused ones).

### 5.5 K5 (conditional): on-node shared-memory transport (core, host memory only)

- **Init** (collective, at GridHalo init):
  - `MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED)` → `node_`. Use the shm path iff
    `size(node_) > 1` and the window's `MPI_WIN_MODEL == MPI_WIN_UNIFIED`.
  - Each rank allocates, through `MPI_Win_allocate_shared`, its on-node receive slots (2 parities ×
    bytes per on-node sender) plus per-sender flags `seq[2]` and `consumed` (uint64, 64-byte
    aligned).
  - Senders learn their slot offsets once, via an all-to-all over `node_`, and resolve base pointers
    with `MPI_Win_shared_query`.
  - Off-node neighbours keep K1's persistent MPI path.
- **Exchange s** (s = ++seq of this instance; one exchange in flight per instance, as today):
  - **Begin:** for each on-node receiver, spin until `consumed ≥ s − 2`. Run the gather kernel with
    on-node payloads written straight into the receivers' parity-(s & 1) slots. Then a release
    fence, then `atomic_ref(seq[s&1]).store(s, release)` per on-node receiver, then `MPI_Startall`
    for off-node.
  - **End:** off-node `MPI_Waitall`; per on-node sender, spin on `seq[s&1] == s` (acquire; bounded
    spin, then `sched_yield`); unpack from the slots and the private buffer; then store
    `consumed = s` (release) per on-node sender.
- **Bits:** the bytes are identical. **Gate:** G-HALO stress.
- **Kill switch:** `PECLET_CORE_HALO_SHM=0` (debug, bitwise-neutral).
- Data then lands during the sender's pack, so the existing interior-sweep overlap becomes real.

### 5.6 Device builds and the distributed device path

- K1(a) and K1(c) apply on device builds; persistent requests on device buffers on the GPU-aware
  path, cached per init.
- K2 and K3 change CutcellMG code that the device distributed path shares, so the CUDA MPI tree's
  np tests must stay bitwise.
- K1(b), K4 and K5 are host-only.

## 6. WO-0: measurement design (attribution)

- **M1, trace** (campaign tools in `bubble_column_perf/distcomm/tools/`, not in a repo's `src/`):
  - (i) A Kokkos Tools library `kp_trace.so`. It records begin/end of
    parallel_for/reduce/scan, deep_copy, fence, allocate/deallocate and push/pop region, with name
    ids and `CLOCK_MONOTONIC` ns (node-wide on one node), into a per-rank binary file.
  - (ii) An LD_PRELOAD PMPI interposer `pmpi_trace.so` for `MPI_{Isend,Irecv,Start,Startall,Wait,
    Waitall,Test*,Allreduce,Allgather(v),Bcast,Barrier,Alltoall*,Neighbor_*}`: call, bytes, peer,
    t0, t1.
  - (iii) `trace_attr.py`, per rank and per step (a step marker: the first kernel label of
    `step()`):
    - (a) time covered by kernels, by MPI, and by gaps;
    - (b) gaps attributed to (previous event → next event) pairs, top 20 by ms/step;
    - (c) MPI ms by call and by the enclosing region or previous kernel (the call-site proxy);
    - (d) per exchange k, the same sequence index on every rank (GridHalo label + ordinal):
      **lateness** = max over its partners of (partner's post time) − (own post time), and
      **protocol** = own wait − max(0, lateness);
    - (e) per-rank kernel totals (the imbalance);
    - (f) allocations and bytes per step.
- **M2, `bench_grid_halo`** (core `tests/`, label `bench`, MPI np 2–24). GridHalo exchanges on the
  column's per-rank level blocks (L0–L3 at 8×3 and 24×1), in modes:
  - back-to-back;
  - begin → synthetic interior kernel of D µs (0, 20, 50, 100) → end (does the wait fall with D?);
  - full against faces-only;
  - variants: Isend, persistent, eager limit raised (diagnostic env of the MPI library), shm (after
    K5).
  - Output: µs per exchange per level, and the protocol share.
- **M3, launch overhead.** An empty `parallel_for` and a 1-cell MDRange at T = 1, 3 (one CCD),
  3 (straddled) and 24: µs per launch.
- **Decision rules:**
  - If protocol ≥ 10 µs/exchange after K1/K3, do K5.
  - If gaps explain ≥ 50 % of the unattributed time at M3's per-launch cost, the launch reductions
    in K2–K4 are the remedy and nothing new is needed.
  - If outside-projection MPI is ≥ 4 ms, report it to Opus for the velocity-MG/field follow-up
    (Q-6).
  - If allocations are ≥ 1 ms/step, add WO-K7 (hoist per-step Views, bitwise) with the list.

## 7. Local reproduction plan (workstation TR PRO 5965WX, 4 CCDs × 6 cores; free)

Layouts, all with `pin.sh` explicit per-rank core lists, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`
and `--bind-to none`:

| id | layout | why |
|---|---|---|
| L-a | 1×24 single | reference |
| L-b | 8×3 aligned (2 ranks/CCD) | today 53.6; no straddling |
| L-c | 8×3 **straddled-2**: r1 = {3,4,6}, r2 = {5,7,8}, others aligned | genoa's 2-of-8 L3 straddle pattern |
| L-d | 24×1 | genoa 1.6×; workstation ≈ 1.36× (72.0 against ≈ 53) |
| L-e | 1×24 + `init_mpi` | the np 1 path |

**Accept the reproduction (end of WO-0) if** L-c's non-kernel ms/step is ≥ 1.5× L-b's, or its halo
wait per exchange is ≥ 1.5× L-b's, **and** L-d shows ≥ 15 ms non-kernel. Then every K-WO is
judged locally on L-c, L-d and L-e.

If L-c does not separate from L-b, the straddle hypothesis is locally falsified. Proceed with
K1–K4, which are layout-independent, and move the genoa run forward (Q-2).

Not reproduced: genoa's per-NUMA bandwidth (≈ 115 GB/s for 24 cores) and its UCX stack. Record
`ompi_info | grep -iE 'pml|btl|ucx'`. If the workstation runs ob1/vader and genoa runs UCX, the
protocol share is decided at genoa (M2 in the confirmation run).

**Per WO:**
- `bench` protocol: 300 steps from `ckpt_t43`, 5 interleaved reps, median ms/step, load average
  < 2 before start.
- M1 at L-c, L-d and L-e: 30 steps, attribution table.
- Exchanges/step and messages/exchange (from M1).

## 8. The genoa confirmation run (one billed job; user OK required, Q-2)

- **When:** after K1–K4 land locally (or right after WO-0 if L-c fails to reproduce).
- **Node:** one exclusive genoa node, 14 min, ≤ 45 SBU, as job 28257785.
- **Builds:** the clean base `flow-s2g/build_omp_mpi_s2g` (b820652, exists), and the distcomm build
  (generic arch, the same flags).
- **Runs:**
  - (1) M2 at 8×3 close, 8×3 per-CCD and 24×1: ≈ 1.5 min.
  - (2) Unfenced N = 200 × 2 reps, base and distcomm, for 1×24 single, 1×24 `init_mpi`, 8×3 close,
    24×1 and 8×3 per-CCD: ≈ 7 min.
  - (3) M1 trace, distcomm, 30 steps, at 8×3 close and 24×1: ≈ 2 min.
- Masks and srun options as in `dist_cpu/s2t/s2t.slurm`. Read `suite/docs/SNELLIUS.md` first, and
  pass positional args, never `VAR=x sbatch`.
- **Output:** table A (ms/step, iterations, ratio), the M1 attribution, and the M2 protocol share.
  These decide K5 and the follow-up.

## 9. Work orders (in order; one commit each; class a throughout)

Set-up:
- Worktrees `suite/flow-distcomm` (exists) and `suite/core-distcomm`
  (`cd suite/core && git worktree add ../core-distcomm -b distcomm main`).
- Each has its own `build_omp` with `-DPECLET_FLOW_HOST_ARCH=znver3` for the workstation, and an
  MPI build `-DPECLET_FLOW_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`. Flow configures with
  `-DPECLET_SIBLING_PECLET_CORE=$PWD/../core-distcomm`.
- Gate scripts are copied from `bubble_column_perf/dist_cpu/gate/` to
  `bubble_column_perf/distcomm/` and repointed.
- State file `bubble_column_perf/distcomm/STATE.md`, rewritten per WO. Numbers are appended to
  `doc/vof_step_performance_log.md`.

The work orders:
- **WO-0** (no `src/` change): M1, M2, M3 and the L-a…L-e baseline (§6, §7). *Accept:* the
  reproduction verdict and the attribution table in the log. The K-ranges in §0 are restated from
  them.
- **WO-K1** (core, 3 commits: (a), (b), (c)).
  - *Accept:* G-HALO, core's ctests at np 1–8, flow G-BIT-D against core main.
  - `bench_grid_halo` before/after. Expected: L2/L3 exchange −40 %, launches/step −323 (gather) and
    ≈ −500 (serial).
- **WO-K3** (core `Neighborhood::Faces`, then flow).
  - *Accept:* G-BIT-D and `mg_stencil_halo_np{1,2,4}`.
  - Messages per stencil exchange: ≤ 3 at 8×3, ≤ 6 at 24×1. np 1 has zero stencil exchanges and
    one smoother launch per colour pass.
- **WO-K2** (flow). *Accept:* `mg_ca_walls_np{2,4}` (CA on against off, bitwise), G-BIT-D, the
  solids battery at np 2/4 bitwise. Exchanges/step 323 → ≤ 262.
- **WO-K4** (flow). *Accept:* G-BIT-D. At np 1, Krylov launches per iteration ≤ single-rank's + 2.
- **WO-K5** (core; only under §6's rule). *Accept:* G-HALO stress, G-BIT-D, and M2 protocol share
  ≤ 3 µs.
- **WO-D:**
  - register entries (§12);
  - flow CLAUDE.md: the "Pressure solve" bullet on the CA-with-walls scope, the stencil halo and the
    test counts;
  - core CLAUDE.md: the GridHalo variants;
  - the log, and closing the state file.
- **Genoa run** (§8, user OK).

## 10. Verification gates

- **G-BIT-D** (every WO):
  1. Host `state_hash.py` all cases + np2: identical to base.
  2. Distributed bubble column, 50 steps from `ckpt_t43`, 3 threads/rank, at np 1 (`init_mpi`),
     2, 4 and 8, rtol 1e-8 and 1e-10, bottom `'auto'` (= `'direct'`) and forced `'algebraic'`:
     u v w p C bitwise to base at the same np, and per-step iterations identical. np 1 is also
     bitwise to single rank. Use the `dist_cpu/gate/drun.sh` + `h6cmp.py` pattern.
  3. The p3 solids battery at np 2/4, 20 steps: bitwise to base.
  4. An outflow case at np 2/4 (`test_vof_bc_mpi` and the outflow np-tests): bitwise. This covers
     K2's and K3's exclusions.
  5. `ctest -LE bench` host (243 at `b820652`, plus the new tests), `ctest -R '_np[0-9]+$'`, and
     the CUDA tree's battery and CUDA `state_hash` unchanged.
  6. Core ctests at np 1–8.
- **G-HALO** (core ctest `grid_halo_variants_np{2,4,8}`):
  - For random fields and per-rank random delays (`usleep` 0–50 µs), 10⁴ exchanges per variant
    (Isend, persistent, serial-copy, fused gather, faces-only on face ghosts, shm).
  - Ghost values are bitwise equal to the reference exchange.
  - No hang within 60 s (the `PECLET_CORE_HALO_TIMEOUT` diagnostic names a stuck partner).
  - Two decompositions: 4×2×1 and irregular ORB 24.
- **G-PERF-D** (every WO, local): §7's protocol at L-c, L-d and L-e, with the M1 table; reported
  against §0.
  - Package gate, local: L-c non-kernel ms/step ≤ 50 % of WO-0's; L-e ≤ 1.10× L-a.
  - Package gate, genoa (confirmation): 8×3 close ≤ 95 ms/step, 24×1 ≤ 100, np 1 ≤ 77 (the
    pessimistic ends of §0).
  - A miss goes to Opus. The implementer does not tune around it.

## 11. Risks and open questions (each has a default; the implementer decides none)

- **Q-1 [pref]: the register entry "Domain-BC paths stay blocking"** (undated, no reason, VelocityMG
  precedent). The pressure MG already overlaps on BC paths; K2 adds CA there. *Default:* proceed and
  record a new entry scoping the old one to the velocity MG. If the user wants it kept for the
  pressure MG, drop K2 (−3 ms).
- **Q-2 [pref; outward-facing and billed]: tagging core and the genoa run.** *Default:* develop on
  the core branch, with flow consuming the core worktree. Do not tag or push core, and do not queue
  genoa, without the user's OK. The genoa run goes after K1–K4.
- **Q-3 [fact]: deep CA (g = 4 on L1/L2).** *Default:* not in this package. Revisit if the post-K5
  per-exchange cost is ≥ 30 µs.
- **Q-4 [pref]: the production 24-core layout.** *Default:* 6×4 CCD-aligned (§14 Q-H9). 8×3 close
  straddles two L3s and 24×1 has uneven z blocks: no communication design removes either.
- **Q-5 [pref]: 24×1's redundant single-thread bottom** (10.45 ms against 3.17). A node-cooperative
  bottom (the node's ranks sharing one factor and M through MPI shared memory) is a separate design.
  *Default:* not now.
- **Q-6 [fact → pref]: the ≈ 12 ms of non-kernel time outside the projection** (velocity MG, field
  and VoF exchanges). K1/K5 reach it through GridHalo. Bringing K2/K3's structure to the velocity MG
  runs into the same register entry. *Default:* WO-0 quantifies it, then a follow-up design if it
  is ≥ 4 ms after K1.
- **Q-7 [fact]: the 8×3 decomposition splits the wall-normal y** (blocks 32×48×64). The register
  says the ORB must never split it ("validated bug, the driver hard-aborts": the channel driver,
  DECISIONS.md:136). Yet this case passes G-NUM-H at np 2/4/8. *Default:* no change here; the caller
  checks whether that entry is channel-specific.
- **R-1:** K3's wrap reads require even inner extents on masked axes; odd ones keep `dev`.
- **R-2:** K5 is a concurrency protocol, so a bug is a race, not a crash. G-HALO stress plus the
  bitwise column runs (each run twice) are the guard.
- **R-3:** persistent requests and GPU-aware MPI: if `MPI_Send_init` rejects device pointers on a
  stack, keep `Isend` on device builds (fall back at init).

## 12. Register entries (drafted in WO-D; the caller adds them)

1. Pressure-MG stencil exchanges use a faces-only halo with rank-local periodic axes read
   in-kernel. *Rejected:* 26-neighbour fills for 7-point stencils.
2. Communication-avoiding coarse smoothing also runs with wall/slip/inflow BCs: post-BC ring
   openness, never on the bottom level or with an outflow face, and the ring rhs is exchanged once
   per V-cycle. *Rejected:* CA only for periodic operators; CA on L0.
3. GridHalo on host: one gather launch, calling-thread copies below `kHaloSerialCopy`, persistent
   requests. *Rejected:* separate self-copy and pack launches; Isend per exchange.
4. The distributed host PCG runs C2–C4 with unchanged MPI calls. *Rejected:* merged or pipelined
   reductions (class b).
5. (If K5) on-node halo by MPI-3 shared windows with sequence flags. *Rejected:* eager-limit tuning
   as the fix; MPI progress threads.
