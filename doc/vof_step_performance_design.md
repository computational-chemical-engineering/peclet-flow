# VoF step performance: design note

**Status:** DESIGN, 2026-09-25. Nothing here is implemented yet. Branch `vof-perf` (worktree
`suite/flow-vof-overlap`). The brief is `doc/vof_step_performance_brief.md`. Numbers marked [brief]
are measurements quoted from it. Numbers marked [model] are derived below and must be re-measured.

**Reading order for an implementer:** §0 (the decisions), §2 (invariants), then your work order
in §7, the gates it cites in §8, and the design section in §5 that the work order points to. §10
contains options that belong to the user. Do not implement any of them unless the user says so.

---

## 0. Summary

The bubble-column step (128×96×64, 16 VoF block markers, ρ and μ ratio 0.02, MG-PCG at rtol 1e-10)
takes 187 ms on 24 genoa cores and 54 ms on one RTX 5080 [brief]. TBFsolver takes 45 ms on the
same cores [brief]. The measured gap has structural causes, and each cause gets one fix below.

| rank | item | kind | GPU saving [model] | CPU 1×24 saving [model] | build | risk |
|---|---|---|---|---|---|---|
| 1 | **B1** geometric-Krylov bottom: the host GraphAMG bottom is replaced on GPU by a single-team in-kernel FCG over the geometric sub-hierarchy | numerics (preconditioner internals); recorded | −12 to −14 ms | 0 (the host keeps GraphAMG) | L | med |
| 2 | **C3** PV curvature fallback: one team per target cell, neighbour terms in parallel, accumulated in canonical order (`pvFitAdd` split in core) | bitwise | −8 to −9 | 0 | M | low |
| 3 | **C1** block container: one launch per stage for all markers, device-side counts, ≤ 2 host reads per container step | bitwise on state | −6 to −7 | −6 to −10 | L | med |
| 4 | **A3** fused periodic wrap in smoother/residual/matvec, plus a division-free ghost fill | bitwise | −1 | −15 to −18 | M | low |
| 5 | **A1** host flags: `-ffp-contract=off` always, `-march` opt-in | bitwise | 0 | −10 to −30 (all stages) | S | low |
| 6 | **A4** host pencil launch rule (MDRange only on device) | bitwise (elementwise) | 0 | −15 to −25 | M | low |
| 7 | **A2** face-form operator storage {AC,TX,TY,TZ} replaces 7 bands | bitwise | −2 | −5 to −12 | M | low |
| 8 | **D1** measure the tolerance the physics needs (no code) | measurement → user | ×0.6–0.8 on projection | same | S | low |
| 9 | **A6** device-resident Krylov scalars (1 host read per iteration) | bitwise | −1 | 0 | M | med |
| 10 | **C2** batched block statistics, deferred read | diagnostics order; recorded | −0.5 to −1.5 | −1 | M | low |
| 11 | **A5** no full-field copies around the preconditioner | bitwise | −0.4 | −2 to −4 | S | low |
| 12 | **B2** host sum-reductions in pencil order | round-off; recorded | 0 | −3 to −8 | S | low |
| 13 | **B1b** B1 eligible with solids (component labels) | recorded | IBM/porous cases | 0 | M | med |
| 14 | **D3** Chebyshev bounds re-estimated warm (S-ladder S2) | recorded | Chebyshev users −40 % projection | same | M | med |

**Expected end state** (§6), main line = items 1–7 and 9–13:
- **GPU:** ≈ 17–21 ms/step (54 now). With D1 it is ≈ 15–19, which is 2.4–3× faster than
  TBFsolver on 24 cores. The ≤ 8 ms GPU projection target is met only together with D1, or with the
  user option E1.
- **CPU (24 cores):** ≈ 80–125 ms/step (187 now), so still ≈ 2× slower than TBFsolver. The ≤ 20 ms
  CPU projection target is **not reachable** in the main line. The main line reaches ≈ 45–80 ms, and
  30–50 ms with D1 plus the separately designed S3. **CPU parity with TBFsolver needs a
  constant-coefficient pressure equation, and that reverses a settled decision.** §10 frames this
  as the user's decision (E2).

---

## 1. Problem and scope

**What is built:** faster execution of the variable-density multi-marker VoF step in `peclet.flow`
(staggered, Kokkos, CUDA/HIP/OpenMP):
1. the pressure projection: MG-PCG and Chebyshev drivers, the `CutcellMG` hierarchy and its
   bottom solve;
2. the host launch structure of every kernel in the step;
3. the execution of the block VoF container: WY advection, debris, recentre, statistics, the
   curvature cascade, the CSF force;
4. a stated end state per stage.

**Out of scope:**
- the physics difference between the codes (bubbles collecting at a wall);
- AMR; multi-node runs;
- the collocated grid;
- the momentum solver's algorithms (it only inherits the host launch rule, A1/A4);
- the container's *data* design (one WY field per marker on a moving box, union MAX colour,
  per-marker CSF SUM; settled in `doc/vof_overlap_design.md`);
- multi-GPU. The new device paths (A6, B1) are single-rank in v1. Distributed paths keep today's
  behaviour bit for bit.

---

## 2. Constraints and invariants

- **Storage and order.** Every field is x-fastest, `I = x + y·ex + z·ex·ey`. Iteration is x-fastest
  on every backend. `Kokkos::Rank<` and `MDRangePolicy<` may appear only in `src/policy.hpp`
  (ctest `iteration_order`). Device code lives in `.hpp` files compiled as C++.
- **Pressure equation (unchanged).** The solver solves `A φ = b` with `b = −div(open·u*)` and face
  coefficient `c_f = open_f·ρ0/ρ_f` (`buildRhoCoeff`, arithmetic face mean, ρ0 = `set_rho` =
  ρ_liquid in the case). In band form `AC = Σ_f t_f`, the off-diagonal across face f is `−t_f`,
  and `t_f = c_f·gf` with `gf` the per-level metric scalar (1/cfac² on coarse levels, exact powers
  of two on isotropic grids). The operator is singular on the periodic/wall path (`removeMean_`):
  the mean is removed over fluid cells (`AC > 1e-30`). The level-0 Krylov matvec is the exact flux
  form (`exactResidual_`, switched on by `enable_vof()`). The V-cycle reads the bands.
- **Driver semantics (unchanged).** PCG stops on `max|r| < rtol·max|r0|`. The start is `x0 = 0`
  (`pwarm_` is off by default). The V-cycle uses pre 2 / post 2 RB-GS sweeps, a reversed colour
  order on the post-smooth, and 12 bottom sweeps; the mean is removed at the L0 exit ("fine"
  scope). The default depth is 4. The `auto` bottom agglomerates on the singular path when the
  coarsest level exceeds 4 cells on any axis (the case: L3 = 16×12×8) and solves it with host
  GraphAMG-PCG to an inner tolerance of 1e-8 (cap 100), rebuilt at every `setOpenness`.
- **Bitwise** means: byte-identical `u, v, w, p, C`, and the block colours, after N steps, plus
  identical per-step pressure iteration counts, on the same build and backend. Host runs are
  compared at a fixed `OMP_NUM_THREADS` (sum reductions depend on the thread count). A
  **structural** work order must be bitwise at the old configuration on host-openmp **and**
  nvidia-cuda, and on the `PECLET_FLOW_OPERATOR_DOUBLE=OFF` tree where it touches operator storage.
  A **numerics** work order is a recorded decision that passes the §8 G-NUM gates.
- **Register decisions that bind this design:**
  - operator storage is double by default (MReal);
  - PCG, not RB-GS, for the pressure;
  - Chebyshev stays the variable-density default;
  - `auto` agglomeration is gated to the singular path;
  - `coarsenOpenAvg` must NOT become harmonic;
  - Dodd–Ferrante has been rejected twice;
  - no environment variable changes a result (use setters or CMake options);
  - the per-kernel `fence()` removal stands;
  - the host-serial threshold (8192 cells) stands;
  - x-fastest order on every backend (USER DIRECTIVE 2026-09-25).
- **Device residency.** No bulk D↔H transfer inside `step()` on a GPU build (USER DIRECTIVE). A
  scalar packet (≤ 64 B) that a host decision needs is allowed, and each one must be justified.
- **MPI.** Decomposition independence and np = 1 bitwise identity must be preserved. Every new
  single-rank fast path has an eligibility predicate; `distributed_` falls back to today's code.
- **Generality.** IBM cut cells, SDF solids, porous, arbitrary factor-of-two structure and
  telescoping all keep working. A fast path is selected by an eligibility predicate, and the
  general path stays intact beside it.

---

## 3. Cost model (the premises of the decisions)

**3.1 Anatomy of one PCG iteration (single rank, today).**
- Work: 1 V-cycle, 1 exact matvec, 2 dots, 1 max, 2 mean removals.
- Host reads: 5 (`dot` ×2, `maxabs`, and the `removeMean` sums of `r` and of the V-cycle exit
  `x`). Each read is a pipeline drain on the GPU.
- Kernel launches: ≈ 100. Per non-bottom level and V-cycle: 8 colour passes, each preceded by a
  full ghost fill; 1 fill plus residual; restrict; zero `cs.x`; fill plus Neumann ghost before
  prolongation; prolong.
- There are 13–15 iterations per step, so ≈ 1400 launches and ≈ 70 host reads per step.
- Profile evidence [brief]: `cc_smooth` 312 launches and `mg_pfill3` 416 launches per step.

**3.2 Level-0 bytes per cell per iteration (double, today) [model].**

| component | bytes/cell | makeup |
|---|---|---|
| RB colour pass ×8 | 8 × 80 | red-black at stride 2 touches whole lines: 7 bands 56 + rhs 8 + x 8 + write 8 |
| residual | 80 | same makeup as one colour pass |
| restrict + prolong | ≈ 26 | |
| V-cycle entry/exit copies | 40 | `rhs ← r`, `x ← 0`, `z ← x` |
| mean removal at exit | 40 | |
| Krylov | ≈ 216 | exact matvec 40, 2 dots 48, 2 axpy 48, mean(r) 40, max 16, aypx 24 |
| **total** | **≈ 1040** | |

At 786 k cells that is 0.82 GB per iteration. On the RTX 5080 (≈ 850 GB/s) that is ≈ 1 ms per
iteration, which is consistent with the measured non-bottom projection of ≈ 12.3 ms at 14
iterations. Of that, ≈ 2–3 ms is launch latency on levels L1–L3.

**3.3 The 5080's FP64 rate is 1/64 of FP32, ≈ 10 GFLOP/s per SM.** Any FP64 work that runs on
one SM or at low occupancy is slow on this card. This bounds B1 and C3 below. On H100 or MI250X
(FP64 1:2) the same kernels are 20–30× cheaper.

**3.4 CPU (1×24 = 125 ms projection) is overhead/compute-bound, not DRAM-bound.**
- The effective rate is ≈ 90 GB/s at 24 cores. With `OMP_PROC_BIND=spread` over a whole node that
  is one core per CCD, and the working set (≈ 120 MB) is plausibly L3-resident (fact to confirm,
  Q4).
- The measured overheads [brief]:
  - host MDRange at 2.4× the cost of a flat x loop;
  - no `-march` (SSE2 only);
  - `mg_pfill3` at 85 µs per launch: a 64-bit div/mod per ghost cell, no serial cutoff;
  - `removeMean`'s `MDRange3` uses Kokkos' **default** host tiles, not `ccFor3`'s `{nx,2,2}`:
    19.6 ms per 56 launches on the workstation.
- The host GraphAMG bottom is serial, ≈ 4–8 ms per step [model].

**3.5 Container.**
- Each master block runs its own `WyAdvector::advect`:
  - 1 CFL reduction read on the host;
  - 3 sweeps × (worklist scan → count to host, PLIC, flux, sweep, ghost fill);
  - then debris scans (counts to host), recentre bbox (2 reductions plus a fence), and `measure()`
    (3 reductions plus fences).
- That is ≈ 60 launches and ≈ 8 host reads per block, so ≈ 1000 launches and ≈ 130 reads per step
  for 16 blocks. The launches are tiny (a block is ≈ 24³). On the GPU the cost is pure latency
  (≈ 5 ms of the 8 ms [brief]).

**3.6 PV fallback.**
- About 35 % of interfacial cells, ≈ 7 k targets, each run a 5³ = 125-neighbour paraboloid fit
  on **one thread** (`curvFallbackCell`).
- That is 7 k threads, ≈ 220 warps on 84 SMs: latency- and occupancy-bound, ≈ 10 ms [brief].
- The FP64 work itself is ≈ 170 MFLOP, ≈ 0.4 ms at the 5080's FP64 peak.

---

## 4. Decisions and rejected alternatives

### 4.1 Bottom solve on the GPU: the geometric-Krylov bottom (B1)

**Decision.** Where eligible (§5.7), a device build replaces `graphAmgSolveBottom` with one kernel
launch. That launch is a single Kokkos team that runs flexible CG (Polak–Ribière) on the bottom
level's own operator. The FCG is preconditioned by one symmetric V-cycle over the geometric
sub-levels *below* the bottom. Those sub-levels are built by the existing coarsening rule and held
outside `lv_`. The inner tolerance (1e-8 relative, ∞-norm, cap 100) is the same as GraphAMG's
today, so the V-cycle keeps its "exact bottom" semantics and the outer iteration count should not
move. Host backends keep GraphAMG. Ineligible configurations keep GraphAMG, and there the
transfers remain as a documented limitation until B1b.

**Rejected alternatives:**
- **Deeper hierarchy with a smoothed bottom.** Measured [brief]: depth 6 raises MG-PCG from 13 to
  21 iterations (+60 % V-cycles). The deep levels' arithmetic-coarsened operators are the weak
  link, and fixing them is S3 (a separate design).
- **Dense direct or explicit inverse of the 1536-cell bottom.** Forming the inverse is
  2n³ = 7.2 GFLOP per step (the operator changes every step under variable ρ). That is ≈ 8 ms in
  FP64 on the 5080. A Cholesky followed by triangular solves is 2n sequential, barrier-bound steps
  per V-cycle. Neither works for large bottoms.
- **Device GraphAMG.** Setup plus an AMG-PCG solve is ≈ 150 launches per bottom solve, × 14 per
  step: launch-bound again.
- **Host bottom with asynchronous copies.** This violates the no-transfer directive, and the
  V-cycle needs the bottom result immediately (there is nothing to overlap).
- **A multi-launch K-cycle.** It is launch-bound.
- **Float inverse.** Float in the pressure path is register-adjacent; see E1.

**Why one team is enough.** The kernel is launch-free and sync-free. Its cost is FP64 throughput
on one SM (§3.3): ≈ 140 kFLOP per inner iteration × ≈ 12–15 iterations, so ≈ 0.2 ms per bottom
and ≈ 2.5 ms per step [model]. That compares with 15.5 ms today. Two open facts can shrink it:
Q1 (move the exact solve one level deeper) and Q2 (loosen the inner tolerance).

**Why the host keeps GraphAMG.** The same team kernel runs serially on a host (team size 1) and is
estimated at 8–24 ms per step, against GraphAMG's ≈ 4–8. Keeping GraphAMG leaves host results
bitwise unchanged. This means two bottom engines are maintained; GraphAMG has to exist anyway as
the fallback.

### 4.2 Operator storage: face form {AC, TX, TY, TZ} (A2)

**Decision.** Each level stores its diagonal `AC` (built exactly as today) and three face arrays,
`TX(i) = ox(i)·gfx`, `TY(i) = oy(i)·gfy`, `TZ(i) = oz(i)·gfz`, in `MReal`, over the full extent.
The seven bands go away:

```
AW(i) = −TX(i)   AE(i) = −TX(i+1)
AS(i) = −TY(i)   AN(i) = −TY(i+ex)
AB(i) = −TZ(i)   AT(i) = −TZ(i+ex·ey)
```

**Bitwise argument.** The old `AW(i)` was `−(ox(i)·gfx)`, stored. `TX` holds the same rounded
product, and negation is exact. In float, `(float)(−t) = −(float)t` because rounding is symmetric.
`AC` is untouched. Every consumer computes the same products with the same operands, as long as
it keeps the expression shape (the order `AE, AW, AN, AS, AT, AB` in the smoother sum).

**Saving.** 56 → 32 B/cell per operator read, i.e. −216 B/cell/iteration (−21 %), and memory
−3 arrays per level.

**Rejected alternatives:**
- **Recompute AC on the fly.** It saves 8 more B/cell, but the stored AC may carry the build
  kernel's FMA contraction, so recomputing it is not provably bitwise on device.
- **Colour-packed layout.** It breaks the x-fastest storage convention.
- **Float bands.** Register (the float-operator trap).

**Implementation note (WO-3, 2026-09-25; coordinator decision): the arrays store the band sign.**
As written above (`TX = +o·gf`, consumers reading `−TX`), A2 was bitwise on host-openmp and on the
float tree but NOT on CUDA: the bubble column drifted by 8.8e-15 in u (relative, 50 steps), and the
staggered/collocated and np2 state hashes changed. nvcc contracts the `(−TX(i+1))·φ + (−TX(i))·φ + …`
chain into different FMAs than the band expression (risk R1; the host does not contract). The
arrays therefore hold the off-diagonal coefficient of the cell's LOW face itself, named
`AFX/AFY/AFZ`: `AFX(i) = −o_x(i)·gfx` = the old `AW(i)`, `AE(i) = AFX(i+1)`, and so on, so every
consumer reads the literal band expression. Same storage (3 arrays, 32 B/cell) and the same bitwise
argument; bitwise on host, CUDA and float (`vof_step_performance_log.md`, WO-1…5 section).

### 4.3 Ghost fills: fused periodic wrap (A3)

**Decision.** On the single-rank path with no outflow face and no overlay:
- the smoother, residual and Krylov matvec read periodic neighbours through wrapped indices
  instead of a ghost fill launched before each pass;
- the smoother does this only when **all three inner dims of the level are even**, so the wrapped
  neighbour always has the other colour and holds exactly the value the fill would have copied;
- residual and matvec may always wrap, because they are read-only;
- wall faces multiply the wrapped value by a coefficient of 0, exactly as the wrapped ghost was
  multiplied.

The fills that remain (before prolongation, the openness fills, ineligible paths) are rewritten
row-wise, without per-cell division, and get the serial cutoff. Both changes are pure copies or
reads of identical values, so they are bitwise.

**Rejected:** keeping every fill and hiding its latency with graph capture (Kokkos Graph is
experimental, it does not reduce the host work, and it does not fix the 85 µs host fill). See F2.

### 4.4 Krylov scalars on device (A6)

**Decision.** On the single-rank path, reductions land in 0-d device Views. α, β and the mean are
computed on the device by the consuming kernels, with the same IEEE divisions. The host reads one
packed scalar struct per iteration, which the stop test needs.

**Rejected:** a lagged or every-k convergence check. It overshoots, so it is not bitwise and it
wastes V-cycles.

### 4.5 The host launch rule (A1 + A4 + B2)

**Rule H**, which from now on applies to every host kernel in the step:
1. **One cell body per kernel.** A `KOKKOS_INLINE_FUNCTION` taking `(x, y, z)` or a flat index.
   All launch forms call the same body: device MDRange, host pencil, and team.
2. **On the device:** `MDRange3` with the default tiling (unchanged).
   **On a host backend:** a `RangePolicy` over the `(y, z)` rows of the box, with the x loop inside
   the helper, marked `#pragma omp simd`, calling the body. This is `ccFor3`'s host branch, which
   today is an MDRange with `{nx,2,2}` tiles. **A new host MDRange in a hot path is a review
   defect.** The rows are statically partitioned over threads, identically for every kernel on a
   level, so a thread keeps touching the same rows and stays L3-local.
3. Below `kHostSerialCellCutoff` (8192 cells) a host launch runs serially. This is already so for
   `ccFor3` and the smoother; it now also covers fills and mean removal.
4. **Contract for the simd helper:** an iteration writes only its own cell(s) and reads no cell
   another iteration of the same row writes. RB colour passes qualify (same-colour cells are
   independent). Ghost copies with overlapping source and destination do not; they use the
   non-simd row helper.
5. **Reductions** keep a fixed, documented order. Changing a host reduction's order is a recorded
   decision (B2), never a side effect.
6. **Host flags (A1):** `-ffp-contract=off` on every host-backend build of flow's targets, so that
   no compiler default forms FMAs. It is inert on today's x86-64 baseline, which has no FMA in the
   ISA. `-march=<arch>` is opt-in through a CMake option. With contraction off, SIMD vectorization
   of non-reduction loops is bit-identical to scalar SSE2. Neither flag is applied to CUDA/HIP
   builds; nvcc keeps `--fmad=true` for device code.

**Rejected:** team policies on host (no benefit, and more overhead); rebuilding the shared Kokkos
prefix with `Kokkos_ARCH_*` (that would change every other repo's host bits).

### 4.6 Block container execution: batched job tables (C1)

**Decision.**
- Each per-block `WyAdvector` keeps ownership of its Views. Nothing about the container's data
  changes.
- A batch driver executes each stage for **all master blocks in one launch**, through a
  by-value job table of raw device pointers plus extents and offsets. This is the pattern already
  proved in `DebrisTable` (`kDebrisBatch = 16`) and `VofCurvFallbackTable<N>`. With more than 16
  blocks the stage runs in chunks of 16.
- A flat index is mapped to (job, local cell) by searching the job's offset table.
- Compaction counts stay on the device. Consumers launch over the upper bound (the region) and
  exit early at `t ≥ count[job]`.
- Host reads per container step: **1** (CFL, because the throw needs it), plus **1** combined read
  of {debris sums, bounding boxes} (the recentre decision needs it). Statistics are deferred
  (C2). Every per-cell body is the existing inline function, so the state is bitwise.

**Rejected:**
- **Per-block streams** (`partition_space`). The per-block host reads still serialize the host
  thread, and the launch count does not drop.
- **One pooled "atlas" allocation for all blocks.** It changes the settled data design and needs
  reallocation logic when boxes resize.
- **Atomics for the overlapping force sum.** They are non-deterministic. The force scatter is a
  gather in block order.

### 4.7 PV fallback: team per target (C3)

**Decision.**
- **In core:** `pvFitAdd` is split into `pvFitTerm` (per neighbour: polygon, frame transform,
  moments, weight → {ok, w, s[6], B}) and `pvFitAccum` (the `+=` into the 6×6 system, the
  identical expressions). `pvFitAdd` becomes their composition, which is bitwise.
- **In flow:** the device fallback launches one team (a warp: 32 on CUDA, 64 on HIP) per target.
  Lanes compute the 125 terms into team scratch. One lane accumulates them in the original
  (oz, oy, ox) order and then solves. The map runs in parallel; the reduction keeps its order, so
  the result is bitwise.
- The host keeps today's one-thread-per-target loop (`if constexpr` on the memory space).

**Rejected:** a warp reduction of the normal equations (it changes the summation order); a
precomputed polygon cache (more storage and code, and not needed to reach the target).

### 4.8 Tolerance (D1): not decided. It is measured.

A structural estimate:
- **Parasitic currents** do not constrain rtol above ≈ 1e-6. The scheme is incremental, so the
  solve error is corrected the next step and does not accumulate. The velocity error it leaves,
  ≈ rtol·|div u*|·dt, is orders of magnitude below height-function curvature errors.
- **Marker volume conservation** is the binding constraint. WY changes a marker's volume by
  `Σ C·div(u)·dt`, and a systematic residual divergence of `rtol·|div u*|dt ≈ rtol·1e-2` per cell
  per step would drift a D/h = 16 bubble by ≈ rtol·1e-2 relative per step.

These say 1e-8 is probably safe and 1e-6 marginal for long runs. That is not enough to change a
default, so §5.12 specifies the experiment. The momentum solve's rtol follows the pressure rtol by
default, so D1 speeds up the momentum stage too.

### 4.9 Deliberately not in the main line

- **S3**, coefficient-aware coarsening: the planned S-ladder. It needs its own design, because
  `coarsenOpenAvg` must not become harmonic.
- **E1**, float V-cycle, and **E2**, Dodd–Ferrante: user decisions, §10.
- **F1**, two-colour temporal blocking of the smoother: bitwise, but L to build. Only if the gates
  after WO-6 miss the target.
- **F2**, Kokkos Graph capture of the single-rank V-cycle: experimental. Only if the coarse-level
  launches still exceed 25 % of the GPU projection after WO-6.
- **F3**, communication-avoiding smoothing with wall BCs (multi-rank CPU layouts 8×3 and 64×3):
  needs a design for the post-BC ring openness.
- **D4**, warm start with a |b|-relative stop: blocked by the registered, unexplained divergence of
  `set_pressure_warmstart` on the steady Stokes march (Q11). A plausible mechanism: the
  r0-relative stop under-solves whenever the warm guess is worse than zero.

---

## 5. Design detail

### 5.1 A1: host compile flags
- In flow's CMake, when the Kokkos backend is a host backend (no CUDA, no HIP), add
  `-ffp-contract=off` to every flow target (solver library, module, tests) for GCC and Clang.
- New cache option `PECLET_FLOW_HOST_ARCH` (string, default empty). When it is non-empty and the
  backend is a host backend, add `-march=${PECLET_FLOW_HOST_ARCH}` to the same targets. Typical
  values: `native` for dev boxes, `znver4` for Snellius genoa, `znver3` for the workstation.
- Do not touch `extern/install/*`. Defaults (Q6): the PyPI wheels stay generic; site builds and
  dev trees set it. Document the option next to `PECLET_FLOW_OPERATOR_DOUBLE` in `CLAUDE.md` and
  the site scripts.
- On aarch64 hosts, `-ffp-contract=off` changes bits relative to today's compiler default.
  Record this in the register entry. It makes host results ISA-independent.

### 5.2 A4 (with A0): cell bodies and the host pencil helper
- **Cell bodies.** Extract the per-cell bodies of these kernels into `KOKKOS_INLINE_FUNCTION`s
  (in `mac_pressure.hpp` / `mac_cutcell_mg.hpp`), without changing any expression:
  - `cutcellSmoothColor` (device MDRange branch and host pencil branch alike);
  - `residualCutcell` / `residualCutcellBox`;
  - `applyCutcellOp` / `applyCutcellOpExact`;
  - `restrictAvg`, `prolongAdd`, `buildCutcellOp`, `coarsenOpenAvg`;
  - the `removeMean` subtract;
  - `applyNeumannGhost` / `applyOutflowGhost`.
- **`ccFor3` host branch:**

  ```
  rows = (hi.y−lo.y)·(hi.z−lo.z)
  if hostRunSerial(cells): the existing serial triple loop
  else: parallel_for(RangePolicy<CCExec>(0, rows), [=](long r){
          y = lo.y + r % ny; z = lo.z + r / ny
          #pragma omp simd
          for (x = lo.x; x < hi.x; ++x) f(x, y, z); })
  ```

  Add a sibling `ccForRows3` without `omp simd` for bodies that break the contract in §4.5 item 4.
- **Route through the helpers.** Convert every raw `MDRange3<CCExec>` elementwise kernel in the
  pressure MG, the variable-ρ coefficient builders (`buildRhoCoeff`, `copyBlockShifted`) and the
  momentum hot kernels in the profile (`ibm_rbgs` colour pass, `vmg_resid`) to `ccFor3`, or give
  them a host pencil branch of the same shape as `cutcellSmoothColor`'s. The device branches stay
  `MDRange3` and are unchanged.
- **Reductions** (`ccReduce3`, the `mgmeanr` part of `removeMean`, `dot`, `maxabs`) stay on their
  current policy in A4. B2 changes them.
- **Audit.** The commit message lists every call site converted and states which helper each uses.

### 5.3 A2: face-form storage
- **`Level` fields.** Replace `AW, AE, AS, AN, AB, AT` with `TX, TY, TZ` (FPV = MReal views of
  length `lv.n`, full extent). `AC` stays.
- **Build.** `buildCutcellOp` writes `AC` exactly as today on the same box, and `TX/TY/TZ` as
  `o·gf` rounded to MReal over **[0, ext)** on every axis. The openness is ghost-filled over the
  full extent before the build (it already is: `fillOpenness` plus `applyBoundaryOpenness`), so
  every index a band ever read, including the CA ring and the `i+1` of the last ring cell, is
  defined.
- **Consumers.** Replace the band reads with `−T` reads in the identical expression order:
  - the smoother and box smoother, device and host;
  - the residual and box residual;
  - `applyCutcellOp` and its box variant;
  - `buildAmg`'s CSR assembly (it mirrors 4 arrays instead of 7, and its double row-sum on the
    singular path is unchanged);
  - the telescope and Repartition stage builds;
  - the ghost-projection BiCGStab and the star overlay wherever they read the hierarchy;
  - the tests that inspect `Level` (the implementer adapts accessors, never assertions).
- **Order.** Write the neighbour sum as
  `(−TX(i+1))·φ(i+1) + (−TX(i))·φ(i−1) + (−TY(i+ex))·φ(i+ex) + (−TY(i))·φ(i−ex) + (−TZ(i+exy))·φ(i+exy) + (−TZ(i))·φ(i−exy)`,
  which is the old AE, AW, AN, AS, AT, AB order. Do not factor the negation out of the sum. That
  would change the sign of an exactly-zero partial sum.
- **Float tree.** `PECLET_FLOW_OPERATOR_DOUBLE=OFF` must remain bitwise too.

### 5.4 A3: fused periodic wrap and the fill rewrite
- **Eligibility of a level for fused reads:** `!distributed_ && !hasOutflow_` and no overlay
  attached (`star == nullptr`, not the ghost-projection BiCGStab path).
  - Smoother: additionally, `inner.x, inner.y, inner.z` are all even.
  - Residual and matvec: no parity condition.
- **Index rule, per axis a, for a cell at inner coordinate `k ∈ [g, g+n_a)`:**
  - minus-neighbour offset `= (k == g) ? +(n_a−1)·s_a : −s_a`;
  - plus-neighbour offset `= (k == g+n_a−1) ? −(n_a−1)·s_a : +s_a`.

  This applies to the field being smoothed or applied (`φ`, `v`) only. Coefficients are static per
  solve and read at their ghost indices as today (TX(i+1) at the high ghost holds the periodic
  face).
- **Launches removed:**
  - the `fill` before each colour pass in `smooth()`;
  - the `fill` + `applyOutflowGhost` before the residual in `vcycleImpl` (single-rank branch);
  - the `fill` + `applyOutflowGhost` in `matvecOverlap` (single-rank branch).

  The `fill` + `applyNeumannGhost` before `prolongAdd` stays.
- **`fillWrap` rewrite.** One `RangePolicy` over *rows* of the three disjoint shell slabs:
  - z-ghost planes: rows `(y, zg)`, contiguous x copy;
  - y-ghost rows at inner z: rows `(yg, z)`, contiguous x copy;
  - x-ghost cells at inner (y, z): rows `(y, z)`, 2·g cells each.

  Per-row index math only, and `hostRunSerial` below the cutoff. The slabs are disjoint and each
  ghost cell is written once from an inner cell, so the result is bitwise.
- **Post-condition change.** After A3, a solve no longer leaves `p`'s or `x`'s ghosts filled as a
  side effect of the matvec. If G-BIT fails only because some consumer reads unfilled ghosts of
  `phi1_`, STOP and report. That consumer already reads ghosts that are inconsistent with the
  mean-removed interior.

### 5.5 A5: no copies around the preconditioner
- `precond(zz, rr)` in `solvePCG`, `solveFCG`, `solveChebyshev` and `estimateEigenvalues` does
  three things:
  1. rebinds `l0.rhs := rr` and `l0.x := zz` (shallow View assignment);
  2. zero-fills `zz`;
  3. runs `vcycle(0)`, then restores both members.
- This relies on level 0 never writing `lv.rhs`. The CA smoother exchanges `rhs`, but only on
  `caOk` levels, which are coarse. Add a debug assertion that `lv_[0].caOk` is false.
- Also fold the zero-fill of `cs.x` before recursion into a **new variant** of `restrictAvg` that
  writes `cs.x = 0` on the same inner cells in the same kernel. `VelocityMG` shares `restrictAvg`
  and must keep the original. The coarse `x` ghosts are always filled, wrap-read or overwritten
  (GraphAMG writes the whole array) before anyone reads them. Bitwise; G-BIT decides.

### 5.6 A6: device-resident Krylov scalars (single rank)
- **Device scalars** (0-d Views, `CCExec` memory): `rz, pAp, rznew, beta, rn, msum, mcnt, stop`.
  - `dot`, `maxabs` and the sum half of `removeMean` reduce into them with the same
    policy and functor, so the order is the same.
  - The subtract kernel computes `mean = msum/mcnt` itself.
  - `distributed_`: after the local reduce, copy to host, `MPI_Allreduce`, copy back into the same
    View. This is today's sync count; on host backends the copies are no-ops.
- **Per solve:** one read of `r0`, and one read of the initial `rz` guard.
- **Per iteration**, in the original order:

  ```
  matvec(Ap,p); pAp=dot(p,Ap)
  update: if (!stop && isfinite(pAp) && pAp > 1e-300) { a = rz/pAp; x += a*p; r -= a*Ap }
          else stop |= BRK_PAP            (x and r untouched)
  removeMean(r); rn = maxabs(r)
  HOST READ {pAp, rn, stop}  →  BRK_PAP: failed iff !finite(pAp); break without ++it
                                BRK_RZ (set last iteration): failed; break, report the index of that iteration
                                rn < rtol*r0: ++it; break
  precond(z,r); rznew=dot(r,z)
  one-thread kernel: if isfinite(rznew) { beta = rznew/rz; rz = rznew } else stop |= BRK_RZ
  aypx: if (!stop) p = z + beta*p
  ```

- **Requirement:** every exit path returns the same `it` and sets the same `solveFailed_` as
  today. A unit test injects a NaN into `pAp` and into `rznew`, and checks the counts and that `x`
  equals the pre-breakdown iterate.
- The update kernel may do the `x` and `r` updates in one kernel: pointwise, so bitwise.
- Chebyshev: device mean, one read of `rn` per iteration. `estimateEigenvalues`: unchanged (it
  becomes rare after D3).
- **Assumption to verify in the gate:** a Kokkos reduction into a device View produces the same
  bits as the same reduction into a host scalar. If it does not, STOP and report. Do not relax the
  gate.

### 5.7 B1: the geometric-Krylov bottom

**Eligibility.** Computed when the geometry or openness hierarchy is built, and cached. The ρ and
ε rescaling never closes a face (ρ0/ρ_f > 0, ε_f > 0), so it does not change the answer. All of
the following must hold:
1. the Kokkos memory space of `CCExec` is not a host space;
2. `!distributed_`;
3. `removeMean_` (singular path) and no outflow face;
4. `agglomerateBottom()` returns true (this already includes the `agglomMode_` and singular-path
   gating);
5. the bottom level has ≤ `kGeoBottomMaxCells = 8192` inner cells;
6. every **interior** face coefficient of the bottom level (periodic wrap faces included, domain
   wall faces excluded) is > 0, which guarantees one fluid component. B1b lifts this.
7. at least one geometric sub-level exists below the bottom.

Otherwise the code takes the GraphAMG path, unchanged.

**Sub-hierarchy.**
- `sub_`, a separate `std::vector<Level>`, is **not** part of `lv_`, so no existing loop over
  `lv_` changes.
- It is built from the bottom by `mgChooseRatio` and the same init code, down to the first level
  where no axis can coarsen, or every coarsenable axis has ≤ 2 cells.
- `setOpenness` continues its coarsening loop into `sub_`: `coarsenOpenAvg`, `fillOpenness`,
  `applyBoundaryOpenness`, `buildCutcellOp`, with the same scale factors. This happens only when
  eligible.
- Each sub-level gets `x, rhs, res`. The bottom level gets FCG vectors `r, p, z, zp, Ap` and a
  solution vector (6 × 1536 doubles in the case).

**Kernel.** One `TeamPolicy<CCExec>(1, T)` launch per bottom solve.
`T = min(1024, team_size_max(functor, ParallelForTag))`; the chosen `T` is recorded in the
`[mg]` debug trace. `team_barrier()` separates every phase. Team reductions (`parallel_reduce`
over `TeamThreadRange`) produce dots, max-norms and fluid means. For a fixed T on a given GPU these
are deterministic. All loops over cells call the A0 cell bodies, so the per-cell arithmetic is the
same as the per-kernel path's.

```
b := lv.rhs (inner); b -= mean_fluid(b)
x = 0; r = b; r0 = max|r|; if r0 == 0: x = 0, done
z = M(r); p = z; rz = <r,z>
for k in 1..100:
    Ap = A_b p            (team phase: periodic wrap fill of p, then the band apply)
    pAp = <p,Ap>; if !(isfinite(pAp) && pAp > 1e-300): break (flag if non-finite)
    a = rz/pAp; x += a p; r -= a Ap; r -= mean_fluid(r)
    if max|r| <= 1e-8 * r0: break
    zp = z; z = M(r)
    beta = <r, z - zp> / rz      (Polak–Ribière; robust to the V-cycle's R != P^T asymmetry)
    rz = <r,z>; if !isfinite: break (flag)
    p = z + beta p
x -= mean_fluid(x); lv.x (inner) := x
```

`M(r)` is **one V-cycle over `sub_`** starting at the bottom level itself:
- pre-smooth 2 sweeps (colour 0 then 1), each colour preceded by a team periodic-wrap fill phase,
  as `smooth()` does today;
- residual (after a fill), `restrictAvg` into `sub_[0].rhs`, `sub_[0].x = 0`;
- recurse through `sub_`: the last sub-level gets 12 sweeps;
- on the way up: fill, `applyNeumannGhost`, `prolongAdd`;
- post-smooth 2 sweeps (colour 1 then 0), and remove the fluid mean at the bottom level's exit.

The fixed constants are τ = 1e-8, a cap of 100, and pre/post/bottom = 2/2/12. They are not setters,
and E3 is the only route to changing τ. On a non-finite scalar the kernel sets `x = 0` on the
bottom and raises a device flag, which the next host read (A6's packet) turns into
`solveFailed_ = true`.

**Selection.** A new developer-tier setter `diagnostics.set_pressure_bottom_solver("auto" |
"geometric" | "algebraic")`, default `"auto"`. `"auto"` means geometric where eligible on a
device backend and algebraic otherwise. `"geometric"` on an ineligible configuration raises with
the failed condition named. Check the spelling against `../docs/NAMING.md` (Q15). It exists so
every gate can A/B the two engines on the same build.

**Unit gate (new ctest, CUDA tree).** The kernel's `M(r)` equals, bitwise, the result of running
the existing per-kernel `vcycle` code over the same levels. That code is reachable by a test-only
hook that builds a `CutcellMG` whose `lv_` is the bottom plus `sub_`, with `agglomMode_ = 0`. The
inner FCG then converges to τ on a manufactured variable-coefficient bottom (ratio 50) within the
cap.

### 5.8 B2: host reduction order (recorded)
- `dot`, the `removeMean` sum (`mgmeanr`) and every `ccReduce3` sum use a host pencil reduction:
  - rows in (y, z) order over the same static row partition as `ccFor3`;
  - x ascending inside a row, scalar, with **no** `omp simd` reduction;
  - per-thread partials combined by Kokkos in thread order.
- Max-reductions are order-free and change nothing.
- Device code is unchanged. The host state hashes are re-baselined in the same commit.

### 5.9 C1: batched container stages
- **Job record.** A host struct copied by value into the functor, ≤ 16 per table:
  - raw pointers: `c, mx, my, mz, alpha, flux, uf, vf, wf, cc (dilation flag), list, outside`
    (null when there is no WO-R mask);
  - `I3 e, n`; `int g`;
  - `long regOff, facOff` (prefix offsets of this job's region and face ranges in the flat index);
  - `bool hasOutsideMask`.
- **Direction.** `d = kWySweepPerm[step_ % 6][s]` is the same for every block, because the
  container passes one `step_`.
- **Stage order.** Each block's stage sequence is kept, **stage-synchronously across blocks**.
  Blocks share no cells, so running stage k for all blocks before stage k+1 reorders nothing that
  any block reads.
- **Stages, one launch each (per chunk of ≤ 16 blocks):**
  1. **CFL.** A per-job max of the Courant number, using the body the job's `interfaceLocalCfl`
     selects (`maxCourantInterface` or `maxCourant`), in one launch (league = jobs). Max is
     order-free. Then **read #1**, which returns every job's value. The host sets each advector's
     `lastCfl_` from it and throws, with today's message, if any value exceeds `cflLimit`.
  2. **Freeze.** The dilation flag, pointwise.
  3. **Sweep loop, s = 0..2:**
     - (a) worklist: one `parallel_scan` over the concatenated `(n+2)³` regions. It writes cell
       indices at global positions and each job's count `cnt[job] = P(end) − P(start)` to a
       device array;
     - (b) PLIC over the upper bound `Σ region`: thread t maps to a job by searching the device
       prefix; exit if `t − P(start_job) ≥ cnt[job]`; otherwise `wyReconstructCell`;
     - (c) flux: `wyFaceFlux` or `wyFaceFluxBc` per the job's flag;
     - (d) the sweep update body;
     - (e) the ghost fill: the three passes of `fillBlockGhosts` (zero, periodicFill, clampFill),
       each batched, in the same pass order.

     The BC face-volume ledger (`accumulateBcFaceVolume`) becomes one thread per (job, side),
     summing its plane in (p0, p1) row order. That is a C2 order change.
  4. **Debris.**
     - The mark scans (D, R, A) are concatenated like (a).
     - The existing single-thread-per-block sums read their counts from the device (their
       canonical list order is unchanged, so they stay bitwise).
     - The act kernels launch over upper bounds with device guards, so "nothing marked" is a
       device no-op.
     - `debrisOut` is **not** read here.
  5. **Bounding boxes.** Batched min/max (order-free), then **read #2**: {per-block boxes, the
     `debrisOut` sums}. The host sets the debris ledgers from it and decides recentres exactly as
     today. The recentre copies themselves are rare and stay per block.
  6. **Curvature and CSF** (`computeCsf`): the tiers 1–2 kernels of `VofCurvature::computeBegin`,
     `buildCsfForce`, and the stats counters (integer sums, order-free) are batched the same way.
     The force scatter into the caller's patches becomes a **gather**: for each face of the union
     of block boxes, loop over the jobs in block order and add the ones whose box contains the
     face. That is the same addition sequence as the per-block scatter.
- **Scope of the batched path.** It covers the plain `advect()` configuration of the block
  container: all-fluid, kinematic plus CSF. Any other configuration (a momentum-consistent or
  energy advector attached, `debugRecomputeDilation`) keeps the per-block path. This is decided by
  an eligibility predicate, as in B1.
- **Before coding**, the implementer greps for consumers of `st_.volume/centroid/velocity/area/
  moment` and of `bcVol_`. If any consumer feeds state (dt, forces, colour), STOP and report
  (Q5).

### 5.10 C2: batched statistics (recorded; diagnostics only)
- `measure()` becomes one `TeamPolicy(league = master blocks)` launch. Each team reduces
  `moments1`, and then `moments2` and `interfaceArea`, with team reductions, into a device stats
  array.
- The centroid velocity uses a device copy of `prevCentroid`. The migration path, which is rare,
  reads it to the host for `serializeAux`.
- The stats reach the host at the next container step's read #1, or in `statsAll()`, whichever
  comes first. They are the same values, one step later in wall time; nothing reads them in
  between (Q5).
- `bubbleBox` min/max and `outOfBoxSum` keep their semantics. `outOfBoxSum` is a recentre-time sum,
  which is rare, and stays as it is.

### 5.11 C3: PV fallback
- **Core** (`core/include/peclet/core/vof/curvature.hpp`):

  ```
  struct PvTerm { double w, B, s[6]; bool ok; };
  KOKKOS_INLINE_FUNCTION bool pvFitTerm(PvTerm&, mx,my,mz,alpha, off,org,t1,t2,nn, dW, cosMin, g)
  KOKKOS_INLINE_FUNCTION void pvFitAccum(PvFit& f, const PvTerm& t)
  ```

  - `pvFitTerm` is the body of `pvFitAdd` up to and including `B`.
  - `pvFitAccum` holds the loops `f.b[i] += w*s[i]*B; f.A[i][j] += w*s[i]*s[j]; ++npoly`, with the
    identical expressions.
  - `pvFitAdd = pvFitTerm` plus `if ok: pvFitAccum`.
  - A core ctest (host and CUDA) checks that `pvFitAdd` equals the composition, bitwise, on 10⁵
    random planes and frames, including the rejected cases.
  - Core is committed and tagged **before** flow uses the change. The tag and push are the
    coordinator's call under the suite's release rule (Q12).
- **Flow** (`src/vof/curvature_field.hpp`, `VofCurvature::fallbackBatch`). On device:
  `TeamPolicy(league = Σ fallback-list lengths over the table, team = warp)` with a scratch of
  `(2gr+1)³ × sizeof(PvTerm)` per team.
  - Every lane computes the target's frame (`nn, t1, t2, org`) redundantly. It is deterministic,
    so no broadcast is needed.
  - `TeamThreadRange(0, (2gr+1)³)`: offset `k → (ox, oy, oz)` in the canonical order
    `k = ((oz+gr)(2gr+1) + (oy+gr))(2gr+1) + (ox+gr)`. The lane writes `ok = false` if
    `!vofIsInterface(c(j))`, else it calls `pvFitTerm`.
  - `team_barrier`, then `single(PerTeam)`: `pvFitInit`, `pvFitAccum` for `k = 0..` in order where
    `ok`, `pvFitSolve`, and write `kap/br` exactly as `curvFallbackCell` does.
  - The early exits (`br(i) ≥ 0`, a failed normal) are taken before any team work.
- On host the current kernel stays unchanged.

### 5.12 D1: the tolerance study (no code in `src/`)
- **Runs.** rtol ∈ {1e-10 (reference), 1e-9, 1e-8, 1e-7, 1e-6} on the pressure driver, with the
  momentum rtol following it (its default), on:
  1. static drop: `tests/study/vof_surface_tension.py static`, gated on max|u|;
  2. Hysing case 1 through the block path: `tests/study/vof_blocks_ns.py hysing`, gated on the
     peak rise velocity and its time, and the final centroid;
  3. the bubble column from `ckpt_t43`, 2000 steps: per-bubble and total gas volume drift, mean
     rise velocity over the last 1000 steps, pressure and momentum iterations per step, and
     max|div u| per step.
- **Acceptance of a candidate rtol:**
  - (1) max|u| ≤ 1.05× the reference;
  - (2) every quantity within 0.2 % of the reference (well inside Hysing's own spread);
  - (3) total volume drift ≤ max(1e-8, 2× the reference's drift) relative, and mean rise velocity
    within 1 % (the column is chaotic, so this is a time average).
- **Outcome.** The loosest passing rtol is recorded with the table. The default action (Q8) is to
  change `run_peclet.py` only; the solver's default stays 1e-10.
- **Measured (WO-13, 2026-10-02; flow ed05b6f, table in `vof_step_performance_log.md`).** Every
  rtol in {1e-9, …, 1e-6} passes all three criteria: static max|u| within 3.6e-8, Hysing within
  1.8e-5, column total drift 1.6e-9 (≤ 1e-8) and rise velocity within 7.8e-9 at 1e-6. The loosest
  passing rtol is **1e-6**; pressure iterations on the column 13.85 → 7.43 per step. The solver
  default stays 1e-10, and so does the published bubble-column case (session decision 2026-10-02: the 1e-6 volume drift is systematic and the comparison code solves its pressure exactly; revisit after the D1 rerun on main). Three caveats are recorded with the table: the
  study predates the variable-μ MAC-face fix (b273031; a confirmation rerun is scripted), the
  2000-step column window does not decorrelate, and the 1e-6 volume drift is systematic
  (≈ 5e-8 extrapolated over a production window).

### 5.13 D3: Chebyshev bounds re-estimated from a warm start (S-ladder S2; recorded)
- Keep the last `v_max` and `v_min` iterates of `estimateEigenvalues`.
- On a re-estimation after a coefficient rebuild, seed each power iteration from them (masked and
  mean-removed as `seedf` does) and run `k_w = 5` iterations instead of 15.
- **Guard:** if the Chebyshev solve then hits `chebMaxit_`, or its residual after 3 iterations
  exceeds `r0`, re-estimate cold (15 + 15 from the rhs) and redo the solve. Count guard firings in
  the solver's diagnostics.
- Expected: 44 → ≈ 24 V-cycles per step on the case under Chebyshev.
- Gates: the `vardensity` suite (hydrostatic acid ratios 3 and 1000, `walls-z` and `jump-z` at
  np 1, 2, 4), and 2000 bubble-column steps under Chebyshev with no guard firing after step 1.

### 5.14 B1b: geometric-Krylov bottom with solids
- Replace eligibility condition 6 with **component labels** on the bottom level, computed on the
  device at geometry time by deterministic min-label propagation over faces with coefficient > 0
  (iterate to a fixed point inside one team kernel).
- If there are ≤ 64 components, the kernel's mean removal becomes a per-component loop: one team
  reduction per label, in label order.
- Cells with `AC ≤ 1e-30` keep `x = 0`.
- If there are more than 64 components, use GraphAMG. The register already requires a
  per-component projector for the agglomerated bottom, and this is the same rule.

---

## 6. Expected end state per stage (ms/step)

| stage | CPU 1×24 now [brief] | CPU main line [model] | GPU now [brief] | GPU main line [model] | main line + D1(1e-8) |
|---|---|---|---|---|---|
| projection | 125.2 | 45–80 | 27.8 | 9–11 | CPU 36–64 / GPU 7.5–9 |
| momentum | 23.4 | 12–18 | 3.9 | 3.9 | CPU 11–16 / GPU 3.5 |
| curvature | 14.4 | 8–11 | 13.0 | 2–3 | same |
| block advect | 13.2 | 4–7 | 8.0 | 1–2 | same |
| predictor + debris + csf + other | ≈ 11 | 7–9 | ≈ 1.4 | ≈ 1 | same |
| **total** | **187** | **≈ 80–125** | **54.1** | **≈ 17–21** | **CPU ≈ 65–105 / GPU ≈ 15–19** |
| TBFsolver, same 24 cores [brief] | 45 | | | | |

"Main line" means WO-1 to WO-11. The CPU ranges are wide because the 24-core per-kernel split is
extrapolated from a workstation profile at 8 threads; WO-0 measures it. The ≤ 20 ms CPU projection
target would need, stacked: a host roofline efficiency ≥ 50 % (§8 G-ROOF), D1, S3, and perhaps E1.
The only way to reach CPU parity with TBFsolver is E2.

---

## 7. Work orders

Every work order lands as one or more commits of its own. **A structural work order commits only
after G-BIT. A numerics work order commits only after G-NUM, its register entry (§11) and
re-baselined hashes, with the old → new table in the commit message.** Each work order also runs
G-PERF and records the numbers in the campaign log. A miss on G-PERF is reported; it does not
cause a revert. Stage named paths only.

WO-7 and WO-8 touch only `src/vof/*` and core. They are independent of WO-1…6 and may run in
parallel in a separate worktree.

**WO-0. Harness, baselines, and the experiments that need no code.**
- Copy `prof.py`, `cmp.py`, `run_mpi.py` and `bench_cpu.sh` from the brief's scratchpad
  (`/tmp/claude-1003/…/scratchpad/perf/`) into `tests/study/vof_perf/`. Make the checkpoint path a
  CLI argument.
- Copy `ckpt_t43.npz` to `~/Codes/peclet-examples-bubble-column/benchmarks/bubble-column/data/`
  (do not commit it) and to the Snellius project directory.
- Record the baselines:
  - `state_hash.py`, host (`OMP_NUM_THREADS=8`) and CUDA;
  - 50-step dumps, host 1×24, host 1×8 and GPU;
  - **N50_rtol**, host and GPU (§8);
  - per-stage timings (host 1×24 on Snellius or a quiet host; GPU);
  - kernel profiles (nsys on GPU; the Kokkos simple kernel timer on host);
  - Snellius `hwloc-ls` and `taskset -cp` of both layouts (Q4).
- Run **E1**: `prof.py --levels {4,5,6} --bottom {auto,agglomerated}`, 20 steps, iterations per
  step, host build (Q1).
- Start the D1 runs (§5.12) in the background.
- **Accept when** the tools are committed and every number above is in the campaign log.

**WO-1. A1 flags.**
- Commit 1: `-ffp-contract=off` on host builds.
- Commit 2: `PECLET_FLOW_HOST_ARCH`.
- **Accept:** G-BIT on the generic, contract-off and `-march=native` trees, all with the same
  hashes as the WO-0 baseline. G-PERF: expect CPU −10 to −30 ms per step overall.

**WO-2. A0 + A4: cell bodies, pencil helper, conversions** (§5.2).
- **Accept:** G-BIT. G-PERF: `cc_residual`, `prolong` and `mgmeans` host time at least halved.

**WO-3. A2 face-form storage** (§5.3).
- **Accept:** G-BIT, including the float-MReal tree. Operator memory −3 arrays per level
  (report). G-PERF: GPU projection −1.5 to −2.5 ms; host smoother −20 % or more.

**WO-4. A3 fused wrap and the fill rewrite** (§5.4).
- **Accept:** G-BIT. `mg_pfill3` launches per step drop from 416 to ≤ 60 on the case (host
  profile). G-PERF: CPU projection −15 ms or more.

**WO-5. A5 + A6** (§5.5, §5.6). Two commits.
- **Accept:** G-BIT plus the NaN-injection unit test. Host reads during the projection ≤
  iterations + 4 per step (nsys: count of D→H copies).

**WO-6. B1 geometric-Krylov bottom** (§5.7). Recorded decision.
- **Accept:** host G-BIT (the host path is untouched). CUDA: G-NUM with outer iterations within
  ±1 of the `"algebraic"` engine on every one of the 50 steps, plus the new unit gate.
- **Zero bulk transfers:** no D↔H memcpy ≥ 1 KiB inside `step()` on the case (nsys).
- G-PERF: GPU projection ≤ 15 ms; bottom ≤ 3.5 ms per step.
- Then run **E3** (Q2): τ ∈ {1e-8, 1e-6, 1e-5}. Adopt the loosest τ with per-step outer
  iterations within +1, as its own recorded commit.

**WO-7. C3 PV fallback.**
- WO-7a (core): the split, its ctest, commit, tag.
- WO-7b (flow): the team fallback.
- **Accept:** core ctest bitwise; flow G-BIT on host and CUDA. G-PERF: GPU fallback ≤ 1.5 ms
  (from ≈ 10).

**WO-8. C1 batched container** (§5.9). Sub-commits a–d in the stage order of §5.9:
- a: advection;
- b: debris;
- c: boxes plus the single read;
- d: curvature, CSF and the gather.
- **Accept, each:** G-BIT on the state (`u, v, w, p, C`, block colours). The block stats and the
  BC ledger may differ only in the fields C2 names (compare them separately). The MPI VoF block
  ctests at np 1, 2, 4 pass.
- **Accept, end of WO-8:** ≤ 3 host reads per container step; block-advect launches per step ≤ 40.
  G-PERF: GPU block advect ≤ 2 ms, curvature (with WO-7) ≤ 3 ms.

**WO-9. C2 batched statistics** (§5.10). Recorded.
- **Accept:** G-BIT on the state. Stats relative difference ≤ 1e-12 against the per-block
  reductions over 50 steps.

**WO-10. B2 host reduction order** (§5.8). Recorded.
- **Accept:** G-NUM, host only. The CUDA hashes are unchanged.

**WO-11. B1b solids** (§5.14). Recorded.
- **Accept:** G-NUM on the IBM and porous state-hash cases on CUDA; `vardensity` and porous ctests
  green; zero bulk transfers on `packing_ring`.

**WO-12. D3 Chebyshev warm bounds** (§5.13). Recorded.

**WO-13. D1 outcome.** Commit the D1 table to the campaign log and `doc/`, and change
`run_peclet.py` if the user agrees (Q8).

**Conditional (only if the gates after WO-6 and WO-8 miss the targets, or the user asks):** F1,
F2, F3, the S3 design, D4 once Q11 is resolved, E1, E2.

---

## 8. Verification gates

**G-BIT** (structural work orders; every item must hold):
1. `tests/regression/state_hash.py`: all cases plus np2, identical hashes to the WO-0 baseline, on
   host-openmp (`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`) and nvidia-cuda.
2. Bubble column: 50 steps from `ckpt_t43` (`prof.py 50 --pcg --dump`). `cmp.py` reports
   `bitwise=True` for `u, v, w, p, C` against the same configuration's WO-0 dump, for host 1×24,
   host 1×8 and GPU. Per-step pressure iterations are identical.
3. `ctest -LE bench`: 175/175 on host (`OMP_NUM_THREADS=8`); the CUDA tree's registered suite is
   green; `ctest -R '_np[0-9]+$'` is green.
4. Where storage is touched (WO-3): the `PECLET_FLOW_OPERATOR_DOUBLE=OFF` tree gives hashes
   identical to its own baseline.

**G-NUM** (numerics work orders):
1. **Noise floor.** `N50_rtol` = max over `u, v, w, p, C` of `max|a−b| / max|a|` after 50 steps
   from `ckpt_t43`, where a is at rtol 1e-10 and b at rtol 1e-9, same build. It is measured in
   WO-0, per backend. A tenfold looser tolerance is always attainable; a tighter one may floor.
   **Gate:** the change's 50-step difference from the reference is ≤ N50_rtol. In words: a
   solver-internal change may move the result no more than loosening the tolerance tenfold does.
2. Per-step outer iterations within ±1 of the reference, and the 50-step total within ±2 %.
3. Per-step `max_open_divergence_projected()` ≤ 2× the reference's value at the same step.
4. Physics: static-drop max|u| within 5 % of the reference; Hysing case 1 (block path) peak rise
   velocity and its time within 0.2 %.
5. G-BIT items 3 and 4 still pass, with tolerances untouched. The hashes are re-baselined in the
   same commit, with the old → new table in the message.

**G-PERF** (every work order): per-stage timings from `prof.py --timing`, and the relevant kernel
counts and times, on the same machine as the baseline (GPU: the 5080 under quiet conditions; CPU:
Snellius genoa through `bubble_cpu.slurm`, or the workstation at load average < 2, taking the
minimum of interleaved rounds as `bench_cpu.sh` does). Report the measured value against the
expected one in the table in §0.

**G-ROOF** (host, informational, WO-2 onward): for each top host kernel, achieved GB/s = model
bytes per cell (§3.2, updated for the work order) × cells × launches / time. Compare it with the
STREAM triad on the same core set. A hot kernel below 50 % is overhead-bound and is the next host
fix; name it in the log.

**Transfer gate** (WO-5 onward, GPU): nsys over 20 steps:
- no D↔H or H↔D memcpy ≥ 1 KiB inside `step()`, except WO-6's A/B engine `"algebraic"` when it is
  selected explicitly;
- small reads per step ≤ (pressure iterations + 4) + 3 (container) + the momentum solve's own.

**Serial ↔ parallel:** the np-independence ctests (`vardensity_mpi`, `telescope_mpi`, the VoF block
MPI tests) are part of G-BIT item 3. The new fast paths are single-rank by eligibility. The
distributed branches of every modified function must be byte-identical in source, apart from the
face-form reads.

---

## 9. Risks and open questions

Each item is labelled **[fact]** (knowable by measurement) or **[pref]** (the user's call), and
each has a default so work can proceed unattended.

- **Q1 [fact]: where should the exact bottom sit?** Does an exact bottom at L4 (levels 5) or L5
  (levels 6) keep outer iterations within +1 of the exact L3 bottom? E1 in WO-0 answers it with
  existing options. **Default:** the bottom is the `auto` level (L3). If E1 shows parity one
  level deeper, pass that level to B1's bottom (a smaller team problem, about 8× cheaper), recorded
  with the E1 table.
- **Q2 [fact]: the inner τ of the geometric-Krylov bottom.** The register notes parity held at 1e-5
  for the GraphAMG bottom. **Default:** 1e-8 until E3.
- **Q3 [fact]: the rtol the physics needs.** D1. **Default:** 1e-10 until measured.
- **Q4 [fact]: Snellius core placement of the 1×24 and 8×3 layouts** (CCD map, L3 residency). It
  decides whether bytes (A2) or instructions (A1, A4) dominate on the CPU. **Default:** proceed in
  the order given; record hwloc output in WO-0.
- **Q5 [fact]: do block statistics or `bcVol_` feed state?** **Default assumption:** no (they are
  replicated diagnostics). WO-8 greps before coding. If they do, keep that sum single-threaded in
  canonical order.
- **Q6 [pref]: `-march` in PyPI wheels?** **Default:** no. Site packages and dev trees only.
- **Q7 [pref]: two bottom engines** (GraphAMG on host, geometric-Krylov on device). The directive
  on serial host paths could be read strictly. **Default:** keep both. The host bottom is tiny, and
  running the team kernel serially is slower.
- **Q8 [pref]: if D1 finds headroom, change the solver's default rtol or only the case script?**
  **Default:** only the case script, with the finding documented.
- **Q9 [pref]:** E1, the float V-cycle (§10). **Default:** not pursued.
- **Q10 [pref, register reversal]:** E2, the scoped Dodd–Ferrante driver (§10). **Default:** not
  pursued.
- **Q11 [fact]: root cause of the registered `set_pressure_warmstart` divergence on the steady
  Stokes march.** D4 depends on it. **Default:** D4 is not pursued in this campaign.
- **Q12 [fact/process]: core tag for `pvFitTerm`.** **Default:** WO-7a lands on core main and is
  tagged by the coordinator. In a dev tree, WO-7b builds against the sibling `../core`. Nothing is
  pushed without the coordinator.
- **Q13 [fact]: the checkpoint and harness live in a volatile `/tmp` scratchpad.** **Default:**
  WO-0 copies them first, before anything else.
- **Q14 [fact]: `team_size_max` of the B1 kernel on the 5080** (register pressure). With T = 256
  instead of 1024, B1 costs ≈ 1.5–2× more. **Default:** accept; report T in the trace.
- **Q15 [pref]: the spelling of `diagnostics.set_pressure_bottom_solver`.** **Default:** as written;
  the implementer checks `../docs/NAMING.md` and adds the row if one is required.
- **Q16 [fact]: remaining GPU launch latency after WO-6.** It decides F2. **Default:** F2 only if
  L1–L3 kernels exceed 25 % of the projection.
- **Risk R1: a bitwise claim fails on device through FMA contraction** when an expression moves
  between functions. **Mitigation:** keep expression shapes verbatim. If device G-BIT still fails,
  STOP and report. Never relax G-BIT, and never add `--fmad=false`.
- **Risk R2: A3 exposes a consumer that reads unfilled `phi` ghosts.** That is a latent defect.
  STOP and report it (§5.4).
- **Risk R3: B1 on the 5080 costs more than modelled**, because of the FP64 per-SM bound.
  **Mitigations:** Q1 and Q2. On H100 the issue disappears.
- **Risk R4: the batched path throws the CFL abort before any block has advected.** Today, blocks
  earlier in the list have already advected when a later one throws. The state after an exception
  therefore differs; successful steps do not. **Default:** accept, and document it beside the CFL cap in
  the block-container header comment. A caller that catches the throw and retries with a smaller dt now retries from an
  untouched state, which is the better behaviour.

---

## 10. For the user: options that touch settled decisions

**E1: mixed-precision V-cycle (register-adjacent: "operator storage is double by default").**
- **What:** store the V-cycle's face coefficients and diagonal (A2's four arrays) and its
  iterates (`x, rhs, res` at every level inside the preconditioner) in float, with double
  arithmetic. The diagonal is formed in double from the float faces, so the preconditioner's
  `A·1 = 0` holds to double rounding. This is the structural fix for the trap the register records:
  float bands used to round the diagonal independently. The outer Krylov (exact double matvec,
  double vectors) is unchanged. Use FCG if PCG loses orthogonality.
- **Gain:** about −45 % of V-cycle bytes, so −25 to −35 % of the projection on both backends. On
  the GPU it closes the ≤ 8 ms target without D1.
- **Cost and risk:** L. It would have to pass the register's own criterion on the high-contrast
  battery (`tests/study/vardensity_solver_probe.py`, ratio 10³ walled, `packing_ring` ratio 10⁴):
  iteration parity and an attainable residual floor ≤ the full-double floor. It re-opens a
  decision taken because the float operator failed silently.
- **Recommendation:** only after the main line, if the GPU target is missed.

**E2: scoped constant-coefficient pressure driver (Dodd–Ferrante), which REVERSES two register
entries** ("Part III… Dodd–Ferrante FFT not needed"; "S-ladder; Dodd–Ferrante splitting
rejected: error ~ σκ").
- **What:** an opt-in driver for solid-free boxes (periodic plus wall BCs, uniform grid). It solves
  `∇²p^{n+1}/ρ0 = ∇·u*/dt + ∇·((1/ρ0 − 1/ρ)∇p̂)` with `p̂ = 2p^n − p^{n−1}` and `ρ0 = min ρ`,
  as TBFsolver does (Dodd & Ferrante 2014; Cifani 2019). The general path stays the default.
- **Two solver options:**
  - (a) the existing MG-PCG on the constant operator. Nothing is rebuilt per step, no contrast
    enters the coarse operators, the bottom is built once, and ≈ 6–8 iterations are expected. The
    projection would be ≈ 25–40 ms on CPU and ≈ 4–6 ms on GPU [model].
  - (b) FFT in x and z plus a tridiagonal solve in y: ≈ 7 ms on CPU [brief] and < 1 ms on GPU. It
    needs a new FFT dependency (FFTW, cuFFT/rocFFT, or kokkos-fft), which is a separate decision.
- **Accuracy cost, to be quantified before any adoption:** the splitting error is
  `(1/ρ − 1/ρ0)·∇(p^{n+1} − p̂)`. It is largest where the capillary pressure jump moves between
  cells, which is exactly the register's objection.
- **Gates:** the static drop (spurious currents), Hysing cases 1 and 2, and bubble-column
  statistics (hold-up, rise velocity, the wall-collection tendency) against the MG projection at
  equal resolution.
- **Recommendation:** decide only if CPU parity with TBFsolver is a goal. Nothing else in this note
  gets there.

---

## 11. Register entries this note creates (for the caller to add)

1. **Face-form pressure operator storage {AC, TX, TY, TZ}.** Rejected: 7 bands; AC recomputed on
   the fly (not provably bitwise under FMA contraction). Bitwise; WO-3.
2. **Geometric-Krylov bottom on GPU backends** (single-team FCG over the geometric sub-hierarchy,
   τ = 1e-8); host keeps GraphAMG. Rejected: deeper hierarchy with a smoothed bottom (13 → 21
   iterations), dense inverse (FP64 cost), device GraphAMG (launch-bound), host bottom with async
   copies (directive). WO-6 plus E3.
3. **Host launch rule H** (pencils on host, MDRange on device, one cell body, `-ffp-contract=off`
   on host, `-march` opt-in). Rejected: host MDRange in hot paths (2.4×); rebuilding the shared
   prefix with `Kokkos_ARCH` (it changes other repos' bits). WO-1 and WO-2.
4. **Fused periodic wrap** on single-rank, even-dimension levels. Rejected: fill-then-sweep plus
   graph capture. WO-4.
5. **Device-resident Krylov scalars**, 1 read per iteration. Rejected: lagged stop checks (not
   bitwise). WO-5.
6. **Batched block-container stages** with raw-pointer job tables; ≤ 3 host reads per container
   step; force scatter as a block-ordered gather. Rejected: per-block streams; a pooled atlas;
   atomics. WO-8.
7. **PV fallback, team per target**, with canonical-order accumulation (`pvFitTerm`/`pvFitAccum`
   in core). Rejected: warp reduction of the normal equations (order change). WO-7.
8. **Recorded order changes:** batched stats and the BC ledger (C2, WO-9); host sum-reduction
   order (B2, WO-10).

---

## 12. Addendum (2026-09-25): E2(a), the opt-in constant-coefficient pressure driver

**Status:** DESIGN. Nothing is implemented. The user decided E2(a) on 2026-09-25 (register entry
"Scoped constant-coefficient (Dodd–Ferrante) pressure driver: opt-in, solid-free boxes,
accuracy-gated", umbrella `e390d73`). That decision supersedes §9 Q10 and the "not pursued" default
of §10 E2. The brief is `doc/vof_constcoef_pressure_brief.md`. **Order:** E2 is implemented after
the main line (§7 WO-0…WO-13). The one exception is WO-E2.0, a measurement with no code, which may
run at any time. Numbers marked [model] are derived here and must be measured.

**Reading order for an implementer:** §12.0, §12.3 (the discrete scheme), then your work order in
§12.9, the gates it cites in §12.10, and §12.6 (placement). §12.4 and §12.5 give the reasons. Read
them before you change any decision.

### 12.0 Decisions at a glance

| # | question | decision | rejected |
|---|---|---|---|
| D-E2.1 | form of the scheme | flow's incremental predictor is unchanged. One explicit face pre-correction `u** = u* − q` is built from the last pressure increment. Then today's constant-density projection runs: the openness operator, `projectCorrect`, the rotational update | TBFsolver's non-incremental form with the pressure removed from the predictor |
| D-E2.2 | surface tension / balanced force | the CSF face force stays in the predictor, untouched. `q` contains no force. Static equilibrium is an exact fixed point | adding σκ∇C to the extrapolated-pressure term, which double counts it in the incremental form |
| D-E2.3 | face density in `q` | `ρ_f = 0.5*(ρ(i)+ρ(i−s))`, the predictor's own expression | the arithmetic mean of 1/ρ (TBFsolver); the harmonic knob, which raises |
| D-E2.4 | ρ0 | the global minimum of the density field, exact, recomputed every split step, device-resident. No safety factor | a fixed ρ0; a user-supplied ρ0; a factor s ≠ 1 |
| D-E2.5 | extrapolation with variable dt | `p̂ − Pⁿ = θ·ΔPⁿ` with `θ = min(1, dtⁿ/dtⁿ⁻¹)` | the uncapped linear extrapolation, which is unstable when dt grows; ignoring dt |
| D-E2.6 | history state | the registry field `"p_increment"` holds `ΔPⁿ = Pⁿ − Pⁿ⁻¹` of the stored values. The runtime scalar dtⁿ⁻¹ is not checkpointed. 2 exact start-up steps. A restart passes `startup_steps=0` | storing `Pⁿ⁻¹`; hooks inside `set_field` |
| D-E2.7 | solver for option (a) | MG-PCG (the `set_pressure_pcg` cap and rtol) on the `set_pressure_geometry` operator, built once. It is independent of the Krylov driver selection, which then serves only the exact start-up steps. No special bottom | following the driver selection; a dedicated bottom engine |
| D-E2.8 | rotational pressure term | reads `div(u**)`. This is the exact Timmermans correction for the constant-coefficient part | `div(u*)` |
| D-E2.9 | slot for option (b) | a single solve function with a written contract (§12.3, S5) | — |

### 12.1 Problem and scope

**What is built:**
- an opt-in pressure-velocity coupling for the staggered `Solver`:
  `set_pressure_constant_coefficient(True)`;
- it replaces the variable-coefficient projection `∇·((dt/ρ_f)∇δp) = ∇·u*` with Dodd & Ferrante
  (2014) / Cifani (2019) splitting:

  ```
  ∇²p^{n+1}/ρ0 = ∇·u*/dt + ∇·((1/ρ0 − 1/ρ)∇p̂)
  u^{n+1} = u* − dt[∇p^{n+1}/ρ0 + (1/ρ − 1/ρ0)∇p̂]
  ```

  written in flow's incremental form (§12.3);
- the Poisson operator is constant, so it is built once, and MG-PCG iterations do not depend on the
  density ratio.

**Scope in which it runs (anything else raises; §12.6 lists the conditions):**
- staggered grid;
- solid-free box: `set_pressure_geometry` with an all-fluid SDF, no immersed solid, no scene;
- every domain face is `'periodic'`, `'wall'` or `'slip'`;
- variable density (a `"rho"` field, normally from `set_property_model("rho", …)`);
- one Picard iteration;
- no porous continuity, no drag field, no divergence source (phase change, user source);
- the arithmetic face density.

Anisotropic cells are allowed: the per-axis weight `w_c` enters exactly as in `projectCorrect`.
MPI is allowed. Momentum-consistent VoF (`enable_vof_momentum`), the block container,
`set_superficial_velocity`, and variable viscosity are allowed.

**Out of scope:**
- option (b), the FFT solver (§12.3 S5 only defines its slot);
- the collocated grid. It is **not trivial**: rung V8 applies every force as a face acceleration
  (`applyFaceAcceleration`) and corrects cells by `applyCellFaceAverageCorrection`, so the explicit
  term would have to enter that machinery and the cell-average correction. That needs its own
  design. The setter raises on `SolverColocated`;
- making the driver the default;
- inflow/outflow faces, Picard iterations, phase change, porous and IBM (§12.11 R-E2.5).

### 12.2 Constraints and invariants

- **Conventions:**
  - x-fastest storage and iteration; `MDRange3` only through `src/policy.hpp`;
  - new host kernels follow Rule H (§4.5);
  - device code in `.hpp`;
  - double operator storage (the default);
  - no environment variable: selection is by setter only.
- **The default path stays bitwise.** With the driver disabled, every existing function computes the
  same bits. §8 G-BIT holds on every WO-E2.x commit. Every edit to an existing function is a branch
  on a flag that is false by default, or an inert bookkeeping assignment.
- **Balanced force.** The CSF face force σκ∇C and the pressure gradient share one face difference
  and one face density: `buildRhsVar`, `addCsfRhsBlocks`, and the block container's per-marker SUM
  paired with the union MAX colour (`doc/vof_overlap_design.md`). The split projection must keep an
  equilibrium exactly stationary. §12.5 P2 proves that it does.
- **Hydrostatic three-way consistency** (`doc/variable_density_projection.md` §1): one arithmetic
  ρ_f in the time term, the body force and the pressure gradient. The explicit term uses the same
  ρ_f (D-E2.3).
- **dt-divided convention.** The momentum RHS is `rs·(ρ_f uⁿ/dt + … − G P)`. The projection works
  on φ with `δP = (ρ0/dt)·φ`. Everything below is in the solver's internal (unit-lattice) units, as
  the kernels compute.
- **MPI.** The only new global operation is a MIN reduction, which is exact and order-independent.
  All new kernels are pointwise with ghost fills. np-independence is therefore of the same kind as
  the constant-density projection: bitwise at np = 1, and at np > 1 the `test_vardensity_mpi`
  tolerances.
- **Device residency.** A single-rank split step adds **no** host read. Under MPI it adds one
  scalar allreduce per step, the same pattern as `applySuperficialVelocity`.

### 12.3 The discrete scheme (normative)

**Notation.**
- For velocity component c: stride `s_c`, per-axis weight `w_c = u_.w[c]` (1.0 isotropic).
- Face difference `G_c q(i) = w_c·(q(i) − q(i − s_c))`. This is the operator `buildRhsVar` applies to
  `P` and `projectCorrect` applies to φ.
- Density: `ρ = rhoField_` (internal units), with ghosts from `fillPropGhosts`.
- Face density: `ρ_f(i) = 0.5 * (ρ(i) + ρ(i − s_c))`.
- `o_c = ox_/oy_/oz_`: the g = 2 openness that `divergOpen` reads. It is 1 on interior and periodic
  faces and 0 on wall and slip faces.
- `P = P_`, and `ΔP` = the registry field `"p_increment"` (member `pIncrement_`).

**One split step, n → n+1** (outerIters = 1):

- **S0, predictor (unchanged).** `buildRhsVar` / `buildRhsVarMom` + `addCsfRhsBlocks` (or
  `addCsfRhs`) + `smoothComp`. It solves `ρ_f(u* − uⁿ)/dt + adv − visc = −G Pⁿ + F_body + F_csf`.
- **S1, ρ0.**
  ```
  ρ0 = min over inner cells and all ranks of ρ(i)
  ```
  Kokkos `Min` into a 0-d device View `constCoefRho0_`. Under MPI: `deep_copy` to host,
  `MPI_Allreduce(MIN)`, `deep_copy` back (the `applySuperficialVelocity` pattern,
  `flow_ibm_project.hpp:295`).
- **S2, θ** (host scalar):
  ```
  θ = (pIncrementDt_ > 0 && dt_ < pIncrementDt_) ? dt_ / pIncrementDt_ : 1.0
  ```
- **S3, explicit face pre-correction.**
  - Fills first: `fillPropGhosts(rhoField_)`, `fillGhosts(pIncrement_)`.
  - Then for c = 0, 1, 2 over the inner faces `[G, e − G)`, with `r0 = constCoefRho0_()`:
    ```
    const double cq = th * dt / r0;                         // th = θ
    const double rf = 0.5 * (rho(i) + rho(i - s));
    u_c(i) -= o_c(i) * (w_c * ((r0 / rf - 1.0) * (cq * (dp(i) - dp(i - s)))));
    ```
    The expression shape is normative (FMA reproducibility; §9 R1).
  - `r0/rf − 1.0 ≤ 0`, and it is exactly 0 on a face whose two cells both hold ρ0.
  - The result is `u** = u* − dt(1/ρ_f − 1/ρ0)·G(θΔPⁿ)`.
- **S4, divergence (unchanged).** `projectAssembleDivergence()` on u**: ghosts, `divergOpen`,
  sources (none in scope), `rhs1_ = −div`.
- **S5, the constant solve: the option-(a)/(b) slot `long solveConstantPressure()`.**
  - It solves `A0 φ = rhs1_` on the g = 1 block. A0 is the operator `setSolidInitPressureMg`
    builds (`mg_.setOpenness(ox1_, oy1_, oz1_, w…)`).
  - Option (a), v1: `phi1_ = 0` (or warm if `pwarm_`), then `projectSolve`'s final MG-PCG call
    **verbatim, arguments included**: `mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_,
    pcgRtol_, 2, 2, 12, fluidOnlyMode_ == 2 ? &starOv_ : nullptr, nStar_, C3{nx_, ny_, nz_})`. The
    star overlay is unreachable without a solid, and keeping the call identical is what makes
    G-E2-RED bitwise. Then `lastPressureFailed_ = mg_.lastSolveFailed()` and `projectSolveTail()`.
  - **Contract for any engine (option (b) later):** same inputs (`rhs1_`, whose fluid-cell sum is
    zero to rtol); φ is defined up to a constant, and the engine should return it in the solver's
    mean-removed gauge; it returns an iteration count and sets `lastPressureFailed_`. It may assume
    A0 is the constant 7-point operator with periodic or Neumann (wall/slip) faces and per-axis
    weights `w_a`.
  - Engine selection is an internal enum `constCoefEngine_ {MgPcg}`. v1 has no public engine switch.
- **S6, correction (unchanged kernel).** `projectCorrect(C[0].u, C[1].u, C[2].u, phi_, …, w…)`,
  then `maskVelocity`. So `u^{n+1} = u** − Gφ`.
- **S7, pressure update and history.** Over all `n_` cells (the same range as the existing
  `press*` kernels), with `ct = r0 / dt` computed in the kernel:
  ```
  const double p0 = P(i);
  P(i) += ct * ph(i) - mr * d(i);          // textual twin of the existing "press" kernel
  dp(i) = P(i) - p0;                       // ΔP^{n+1}: the stored pressures' own difference
  ```
  - `mr` is the rotational coefficient of the existing branch logic, unchanged:
    - `varProps_` with `varRotMode_ == 1`: `varRotChi_·μ(i)`;
    - `varProps_`, min mode: `varRotChi_·minMuInner()`;
    - `varProps_`, off: 0;
    - otherwise: `rotationalP_ ? rotWeight_·mu_ : 0`. The wall-blend branch reduces to this with no
      solid.
  - `d = div_ = div(u**)` (D-E2.8).
  - Finally, on the host: `pIncrementDt_ = dt_`.

**The same step in continuous form.** Let `δ = P^{n+1} − Pⁿ + μ_r∇·u**`. Then

```
u^{n+1} = u* − (dt/ρ0)∇δ − dt(1/ρ_f − 1/ρ0)∇(θΔPⁿ),      ∇·u^{n+1} = 0
```

Substitute `u* = ũ − (dt/ρ_f)∇Pⁿ`, where ũ is the momentum update without pressure. This is
exactly the brief's scheme `u^{n+1} = ũ − dt[∇p^{n+1}/ρ0 + (1/ρ − 1/ρ0)∇p̂]` with `p̂ = Pⁿ + θΔPⁿ`,
which is `2Pⁿ − Pⁿ⁻¹` at constant dt. The ∇Pⁿ terms cancel identically. The Poisson equation
`∇²δ/ρ0 = ∇·u*/dt − ∇·((1/ρ_f − 1/ρ0)∇(θΔPⁿ))` is S3 + S4 + S5.

**Exact (start-up) steps while the driver is enabled.**
- The projection is today's five stages, untouched.
- It is bracketed by `deep_copy(pIncrement_, P_)` before `projectPressureUpdate()` and
  `pIncrement_(i) = P_(i) − pIncrement_(i)` (all `n_` cells) after it.
- Then `pIncrementDt_ = dt_` and `--constCoefStartupLeft_`.
- So ΔP has the same definition, "stored P^{n+1} minus stored Pⁿ", after either kind of step.

### 12.4 Decisions and the reasons

**D-E2.1 and D-E2.2: the incremental pre-correction form; the CSF stays in the predictor.**

TBFsolver writes the non-incremental scheme and puts σκ∇C inside the old-pressure divergence:
`computeOldPressDiv(…, psi, st, …)`. Its velocity update is
`u = ũ − dt[(∇p^{n+1} − F)/ρ0 + (1/ρ − 1/ρ0)(∇p̂ − F)]`. Expand it:

```
= ũ + dt F/ρ − dt[∇p^{n+1}/ρ0 + (1/ρ − 1/ρ0)∇p̂]
```

because F/ρ0 + (1/ρ − 1/ρ0)F = F/ρ.
- TBFsolver merges F with p only so that F meets the same face 1/ρ as the pressure. In flow, F
  already sits in the predictor, beside `−G Pⁿ`, divided by the same ρ_f through the momentum
  operator.
- The DF explicit term then involves only the pressure increment (§12.3, continuous form). Putting
  F into `q` as well would count it twice.
- The incremental form keeps every validated piece of the projection bit-for-bit: the divergence,
  the constant-density operator, `projectCorrect`, `maskVelocity`, and the rotational update. The
  new code is one pointwise kernel, one reduction, and one twin of the pressure-update kernel.
- The non-incremental alternative would need a second predictor without `−G Pⁿ`. That is a second
  RHS family, and the incremental Timmermans machinery would have to be re-derived for it.
  Rejected.

**D-E2.3: ρ_f is the arithmetic mean of ρ, not the mean of 1/ρ.** The explicit term must cancel
the predictor's `(dt/ρ_f)∇Pⁿ` exactly (the continuous-form derivation). A different face density
ρ̃_f leaves an inconsistency `dt(1/ρ̃_f − 1/ρ_f)∇ΔPⁿ` that is first order in time and has a
ratio-sized coefficient at interface faces. TBFsolver's mean of 1/ρ is consistent inside
TBFsolver's own discretization; it is not consistent with flow's. The harmonic knob
(`rhoFaceHarmonic_`) changes only the projection and not the predictor, so under this driver it
raises.

**D-E2.4: ρ0 = min ρ, exactly, per step.**
- The linear analysis (§12.5 P3) gives the per-mode factor `μ = 1 − ρ0/ρ_f`. Stability at constant
  dt needs `μ > −1/3`, i.e. ρ0 < (4/3)·min ρ_f.
- `ρ0 = min ρ` is the largest ρ0 with μ ≥ 0 on every face, the regime in which the error modes decay
  monotonically in modulus. It is also the most accurate stable choice: the explicit part
  `(1/ρ0 − 1/ρ_f)` grows as ρ0 shrinks.
- A factor s < 1 makes the explicit term nonzero in the light phase, which is exact at s = 1, and
  slows the light-phase modes. A factor s > 1 buys at most a few per mille on the heavy-phase rate
  and leaves a band of stability that no configuration needs. So s = 1.
- Per step rather than fixed, because the density field is arbitrary: a closure, a table, or a
  transported field.
- The minimum is exact, so it is identical on every rank and at every np. It costs one 8 B/cell
  read.
- On a face between two light cells, `r0/rf − 1.0` is exactly 0: 0.5·(2ρ0) = ρ0 in IEEE-754.
  Ghosts are copies of inner values, so a boundary face cannot go below ρ0.
- No override in v1 (§12.11 R-E2.7).

**D-E2.5: θ = min(1, dtⁿ/dtⁿ⁻¹).**
- With a history ratio r, the error recurrence (§12.5 P3) is `z² − (1+r)μz + rμ = 0`.
- By the Jury criterion it is stable for every μ ∈ [0,1) iff r ≤ 1. For r > 1 it is stable only
  while `r < 1/μ_max = ρ_max/(ρ_max − ρ0)`, which is 1.02 at ratio 50.
- The linear-in-time extrapolation `r = dtⁿ/dtⁿ⁻¹` therefore amplifies heavy-phase errors whenever
  dt grows by more than 2 % in one step. `step_adaptive` and a driver's `set_dt` both do that.
- Capping at 1 keeps the second-order linear extrapolation when dt shrinks. When dt grows it
  under-extrapolates, which is first order for that step, and it is unconditionally stable in the
  model. At constant dt, θ = 1.0 exactly, and `cq = 1.0*dt/r0`, so the constant-dt bits do not
  depend on the rule.

**D-E2.6: the history.**
- ΔP is a registry field, `"p_increment"`, G = 2, cell-centred, internal units like `"p"`.
  Consequences:
  - it is device-resident and allocated once, when the driver is first enabled (zeros);
  - `get_field` and `set_field` save and restore it with no new API;
  - `redistribute` migrates it, because the FieldSet owns it. The member alias `pIncrement_` is
    re-bound beside `rhoField_` (`flow_ibm_mpi.hpp:235`, `bind(pIncrement_, "p_increment")`).
- It is not `Pⁿ⁻¹`. The increment is what the kernel consumes. It is small (O(dt·∂p/∂t)), so the
  S3 difference carries no cancellation against the large hydrostatic and capillary P. And it
  makes "no history" mean ΔP = 0, which is a valid (zero-order) extrapolation rather than a
  pressure of zero.
- **dtⁿ⁻¹** (`pIncrementDt_`) is a runtime scalar and is not checkpointed. It is reset to 0
  ("unknown", θ = 1) by every call of the setter. A restart at constant dt is therefore bitwise
  continuous, because the continuous run also has θ = 1.0.
- **Start-up:** `startup_steps` exact steps (default 2) run after every call of
  `set_pressure_constant_coefficient(True, startup_steps)`.
  - From a cold P (zero, or any non-equilibrium field), the first exact step jumps P to near
    equilibrium. Its ΔP is that jump, not a time increment.
  - The second exact step records a genuine increment.
  - A restart that restores `"p"` and `"p_increment"` passes `startup_steps=0`. A restart that
    restores `"p"` only passes 1.
- Hooks in `set_field` were rejected: a name-keyed special case in a generic accessor, with a
  silent dependence on the order of the calls. An explicit argument makes the restart contract
  visible in the driver script.

**D-E2.7: the solver and the driver selection.** Option (a) is by definition MG-PCG on the constant
operator.
- The split solve always uses MG-PCG with the MG-PCG cap and rtol (`pcgMaxit_`, `pcgRtol_`; C++
  defaults 500 and 1e-10; set by `set_pressure_pcg`).
- It does **not** read `useChebyshev_` or `useFcg_`. The Krylov selection then governs only the
  exact start-up steps. That is right: at ratio ≥ 10³ those steps want the density mode's Chebyshev
  default (PCG caps there, as the CLAUDE.md pressure section records), while the constant operator
  wants PCG at any ratio.
- Consequence: the new setter is **order-independent** with respect to `set_property_model("rho")`.
  "Select the driver last" still applies to the start-up driver only.
- The momentum tolerance rule "follows the active pressure rtol" returns `pcgRtol_` while the driver
  is enabled (`velocityResidualTolerance()`, `flow_ibm_core.hpp:430`). It does not flip between
  start-up and split steps.
- **Bottom:** nothing special. The constant operator's hierarchy and bottom are built by one
  `setOpenness` and then reused:
  - host: `auto` agglomeration → GraphAMG, set up once;
  - GPU: the B1 geometric-Krylov bottom (§5.7), sub-hierarchy built once;
  - before WO-6 the GPU keeps GraphAMG's per-V-cycle transfers, which is B1's concern, not E2's.
- Whether the constant operator makes the agglomerated bottom unnecessary altogether (levels 6 with
  the smoothed bottom on 4×3×2) is a fact WO-E2.0 measures (R-E2.6).

**D-E2.8: the rotational term reads div(u**).** From u** to u^{n+1} the correction is the pure
gradient Gφ of a constant-coefficient Poisson problem. So `Lap(Gφ) = G(Lap φ) = G(div u**)`, and
`−μ_r div(u**)` is the exact Timmermans correction for the implicit part. The explicit q is lagged
data and belongs with the predictor. The difference from `div(u*)` is `μ_r div q`, which is
O(dt²). It keeps `projectAssembleDivergence` unchanged, with one divergence per step.

### 12.5 What the design guarantees (the premises of the gates)

**P1, reduction.** If ρ ≡ ρ0 everywhere (bitwise), then:
- `r0/rf − 1.0 = 0` → S3 leaves u unchanged;
- the exact path's `buildRhoCoeff` gives `cx1 = ox1·ρ0/ρ_f = ox1` exactly, so both paths hand
  identical coefficients to `setOpenness`;
- `projectCorrectVar` equals `projectCorrect` (ratio 1.0);
- `ct` is the same division;
- `mr` is the same.

So a split step equals an exact MG-PCG step **bit for bit**. Gate G-E2-RED.

**P2, identical fixed points; balanced force is exact at equilibrium.** Hold the interface, κ and
the forces fixed, and suppose a steady state of the split scheme exists. Then
`ΔP = P^{n+1} − Pⁿ = 0`, so q = 0. The update gives `(ρ0/dt)φ = μ_r div(u**)` with
`Lap φ = div u**`. So `(ρ0/dt − μ_r Lap)φ = 0`. That operator is SPD on mean-free fields, also with
a pointwise μ(i) > 0, so φ = 0. Hence `u** = u* = u^{n+1}`, divergence-free, and the momentum
equation reads `ρ_f(u − u)/dt + adv − visc = −G P + F`. That is **the same steady-state system as
the exact projection** (whose φ vanishes by the same argument with coefficient ρ0/ρ_f). In
particular:
- a static drop in exact discrete balance (`G P = F_csf` on every face, as P1/P3 of
  `test_vof_surface_tension` construct with constant κ) is a fixed point of the split scheme, to
  round-off;
- the steady spurious-current field of a height-function drop, when one exists, is the same field.

The register objection "error ~ σκ" therefore **does not apply at equilibrium**. It applies to
transients (P3).

**P3, the pressure-error dynamics (linear model).** Take an inviscid, advection-free perturbation
about a steady state, with velocity kept solenoidal. Let `K = L0⁻¹L_ρ`, where
`L_ρ = ∇·((1/ρ_f)∇·)` and `L0 = (1/ρ0)∇²`. For a pressure error `eⁿ`:

```
e^{n+1} = (I − K)·((1+θ)eⁿ − θ e^{n−1}),     spectrum of K ⊂ [ρ0/ρ_max, 1]
```

The exact projection gives `e^{n+1} = 0`. Per mode with `μ = 1 − λ`:
- **Stability.** See D-E2.4 and D-E2.5.
- **Decay (θ = 1).** The factor is |z| = √μ, a complex pair with argument ≈ √(1−μ).
  - Heavy-phase modes (λ ≈ ρ0/ρ_max): 0.990 per step at ratio 50 (e-folding in 99 steps, ringing
    period 2π√50 ≈ 44 steps); 0.9995 at ratio 1000 (e-folding 2000, period 199).
  - Light-phase modes: λ = 1, removed in one step.
- **Forced response.** A per-step defect f (the extrapolation truncation, ∝ dt²·∂²p/∂t²) settles to
  e ≈ f/λ, i.e. **amplified by ρ_f/ρ0 in the heavy phase**. The splitting error is second order in
  dt with a density-ratio-sized constant.
- **Where it bites.** It is largest where ∂²p/∂t² is large. When a capillary jump crosses a cell,
  that cell's P ramps by σκ·ΔC per step and kinks as the interface enters or leaves. That is the
  register's objection, precisely. The model predicts:
  - the split–exact difference falls ≈ 4× when dt halves (gate A2);
  - it grows with ρ_max/ρ0 (gate A3, case 2 vs case 1).
- **1-D exactness.** In a horizontally uniform, inviscid, walled column the recurrence is exact and
  pointwise per face:
  ```
  g_{n+1} = (1 − a_f)((1+θ_n)g_n − θ_n g_{n−1}),   g_n = G Pⁿ − F_f,   a_f = ρ0/ρ_f
  ```
  with `g_{−1} = g_0` when there is no history, and the velocity stays exactly zero. This is an
  implementation gate with no free parameter (G-E2-REC).

**P4, conservation.** The constraint is enforced by the constant solve to `pcgRtol_`, as the exact
path enforces its own. The Weymouth–Yue volume conservation, which depends only on the discrete
divergence of the advecting field, is unaffected.

### 12.6 Placement and code layout

**In `step()`** (`flow_ibm_project.hpp:14`):
- `constCoefPrecheck()` sits next to `superficialVelocityPrecheck()`, before the first mutator. So a
  refused configuration throws with every field untouched.
- Nothing else in `step()` changes.

**In `project()`** (`flow_ibm_project.hpp:770`):

```
const bool split = constCoefP_ && constCoefStartupLeft_ == 0;
constCoefLastSplit_ = split;
if (split) {
  constCoefPrepare();              // S1 (rho0), S2 (theta), fills, S3 kernel
  projectAssembleDivergence();     // S4, unchanged
  constCoefEnsureOperator();       // A0 into the MG iff !constCoefOpReady_
  solveConstantPressure();         // S5 (includes projectSolveTail())
  projectCorrectVelocities();      // S6: its staggered chain takes projectCorrect when split
  constCoefPressureUpdate();       // S7
} else {
  projectAssembleDivergence(); projectBuildCoefficients(); projectSolve();
  projectCorrectVelocities();
  if (constCoefP_) { deep_copy(pIncrement_, P_); }
  projectPressureUpdate();
  if (constCoefP_) { pIncrement = P − pIncrement; pIncrementDt_ = dt_; --constCoefStartupLeft_; }
}
```

**Edits to existing functions.** Each is inert when the driver is off.
- `projectSolve()`: extract its tail (from `copyInner(phi_, …)` through the outflow block) into
  `projectSolveTail()`, which is called by both. This is structural and bitwise.
- `projectCorrectVelocities()`, staggered chain: `else if (varRho_)` becomes
  `else if (varRho_ && !constCoefLastSplit_)`.
- `constCoefOpReady_` is set false at every other `mg_.setOpenness` site: `projectBuildCoefficients`,
  both branches; `setSolidInitPressureMg`; any MG (re)initialisation in the MPI and redistribute
  paths. It is set true only by `constCoefEnsureOperator()`, which runs:
  `mg_.setBoundaryConditions(bc_); mg_.setOutflowCoefficient(false);
  mg_.setOpenness(ox1_, oy1_, oz1_, u_.w[0], u_.w[1], u_.w[2]);`
  That is the geometry path's own sequence minus `init`, and it leaves `setAgglomerationMode` as the
  geometry set it.
- `velocityResidualTolerance()`: `if (constCoefP_) return pcgRtol_;` after the explicit-tolerance
  test.

**Files.**
- The S3 kernel is a free function `projectExplicitSplit(...)` in `mac_pressure.hpp`, beside
  `projectCorrectVar`, with the loop per Rule H.
- The S1/S2/S7 members go in `flow_ibm_project.hpp`.
- The setter and `velocityResidualTolerance` are in `flow_ibm_core.hpp`.
- The precheck goes beside `superficialVelocityPrecheck`.
- State and declarations with docstrings are in `flow_ibm.hpp`, near the pressure-driver state:

  ```
  bool constCoefP_ = false;              // set_pressure_constant_coefficient
  int constCoefStartupLeft_ = 0;         // exact steps still to run before splitting
  bool constCoefLastSplit_ = false;      // the last project() ran the split scheme
  bool constCoefOpReady_ = false;        // the pressure MG holds A0 (see constCoefEnsureOperator)
  double pIncrementDt_ = 0.0;            // dt of the step that produced p_increment; 0 = unknown
  CCField pIncrement_;                   // aliases the registry field "p_increment"
  Kokkos::View<double, CCMem> constCoefRho0_;  // per-step min rho (internal), device-resident
  enum class ConstCoefEngine { MgPcg } constCoefEngine_ = ConstCoefEngine::MgPcg;
  ```
- `bind(pIncrement_, "p_increment")` goes into the redistribute re-alias list
  (`flow_ibm_mpi.hpp:235`), and `constCoefOpReady_ = false` goes there after the MG rebuild.

**`constCoefPrecheck()` raises (named message, "…under set_pressure_constant_coefficient…") if:**
- an immersed solid is present (`hasSolid_`);
- `!cutcellPressure_` (call `set_pressure_geometry` with an all-fluid SDF);
- `porous_`;
- `hasDrag_`;
- `!varRho_` (a constant-density run already solves a constant operator);
- any domain face is inflow or outflow;
- `outerIters_ > 1`;
- `pressUnderRelax_ != 1.0`;
- `rhoFaceHarmonic_`;
- `pcEnabled_ || pcHasUser_`;
- `!incremental_`.

The setter itself raises on `SolverColocated` and on `startup_steps < 0`.

**MPI summary.**
- S1: device MIN + one host allreduce.
- S3: `fillPropGhosts` and `fillGhosts`, the existing halo paths.
- S5: the existing distributed MG-PCG, including telescoping, on A0.
- S7: pointwise.
- redistribute: migration by the registry, re-alias, operator flag cleared.

No other global operation is added.

### 12.7 API

- **Public** (`Solver`):
  `set_pressure_constant_coefficient(enabled: bool, startup_steps: int = 2)`.
  - The docstring states: the scope and what raises; that the split solve uses MG-PCG with
    `set_pressure_pcg`'s cap and rtol while the selected driver serves the start-up steps; that the
    call is order-independent of `set_property_model`; the restart recipe; and that adoption per
    case follows the accuracy gates.
  - Calling it again re-arms the start-up count and resets dtⁿ⁻¹. `enabled=False` returns to the
    exact projection, which rebuilds its coefficient operator on the next step by itself. The field
    is kept.
- **Registry field** `"p_increment"` (internal units, like `"p"`), through
  `get_field`/`set_field`. It is registered at the first `enabled=True`.
- **Restart recipe** (documented; a driver script follows it verbatim):
  1. build the solver exactly as before;
  2. `set_field` for `u, v, w, p, C` (and the block state the script already restores), then
     `set_field("p_increment", …)`;
  3. `set_pressure_constant_coefficient(True, startup_steps=0)`.

  Restore without `p_increment` → pass `startup_steps=1`. Bitwise continuity holds at constant dt
  across the restart.
- **Developer** (`s.diagnostics`): `pressure_constant_coefficient_stats()` returns
  `{'enabled', 'last_step_split', 'startup_steps_left', 'rho0' (physical units; NaN before the first
  split step), 'theta'}`. `rho0` is read on demand from the device View.
- **NAMING:** new names only. WO-E2.5 confirms that no existing spelling of the concept exists
  before adding them.

### 12.8 Expected cost, bubble column (ms/step) [model]

**Premises:**
- 7–9 MG-PCG iterations at rtol 1e-10 on the constant operator (13–15 today on the variable one);
  WO-E2.0 measures it;
- the per-iteration cost of the main line from §6: CPU 3.0–5.5 ms, GPU 0.65–0.8 ms including the B1
  bottom;
- no per-step `setOpenness` or bottom setup;
- S1 + S3 + S7 ≈ 110 B/cell of extra traffic, about 0.1 of one PCG iteration (≈ 1 ms CPU, ≈ 0.1 ms
  GPU).

| | CPU 1×24 main line (§6) | CPU + E2(a) | GPU main line (§6) | GPU + E2(a) |
|---|---|---|---|---|
| projection | 45–80 | **22–50** | 9–11 | **4.5–7** |
| rest of the step | 35–45 | 35–45 | 8–10 | 8–10 |
| **total** | 80–125 | **57–95** | 17–21 | **12.5–17** |
| E2(a) + D1 at rtol 1e-8 (≈ 6 iterations) | | 54–80 | | 12–15 |
| TBFsolver, same 24 cores [brief] | 45 | | | |

**Conclusions:**
- E2(a) roughly halves the projection on both backends. The iteration count no longer depends on
  the density ratio: at ratio 10³ the variable operator needs Chebyshev at 44 V-cycles/step.
- **CPU parity with TBFsolver is still not reached by E2(a).** It reaches 57–95 against 45.
- Parity needs option (b): a direct solve of ≈ 7–10 ms, for a total of ≈ 42–55. That is the
  separate decision, and this design leaves its slot ready.
- On the GPU, E2(a) gives ≈ 3× TBFsolver's 24-core step.
- Built before the main line (not planned), E2(a) would give CPU ≈ 55–75 and GPU ≈ 12–16 ms
  projection, because the host bottom's per-V-cycle transfers remain.

### 12.9 Work orders

Each work order lands as its own commit or commits and stages named paths only.
- WO-E2.1 and WO-E2.2 are **structural** and commit after §8 G-BIT.
- WO-E2.3 and WO-E2.4 are **numerics on a new opt-in path**. They commit after their G-E2 gates,
  and after G-BIT for the default path.
- Dependency order: E2.0 → E2.1 → E2.2 → E2.3 → E2.4 → E2.5. E2.6 and E2.7 come after E2.3. The
  bubble-column part of E2.6 comes after the low-wall physics fix.

**WO-E2.0. Measure the premise (no code; may run before the main line).**
- Run a ratio-1 control of the bubble column: the rho closure with params `[ρ_l, 0]`, everything
  else as the case. From `ckpt_t43`, run 50 steps with MG-PCG at rtol {1e-10, 1e-8}. The MG then
  holds the constant operator.
- Record per-step PCG iterations at levels {4 auto, 5 auto, 6 smoother}, on host 1×8 and on GPU.
- In the ratio-50 exact run, time `projectBuildCoefficients` (with the bottom setup) per step.
- **Accept:** the numbers are in the campaign log.
- **STOP and report** if the levels-4 control needs ≥ 12 iterations at 1e-10. The premise of
  option (a) is then false.

**WO-E2.1. Structural preparation.**
- Extract `projectSolveTail()`.
- Add the `constCoefOpReady_` bookkeeping at every `setOpenness` and MG-init site.
- Add `solveConstantPressure()` (S5), not yet called.
- **Accept:** G-BIT.

**WO-E2.2. State, API, start-up bookkeeping** (no split numerics yet).
- The setter (C++ `setPressureConstantCoefficient(bool enabled, int startupSteps)`) and its
  binding.
- The registry field and the alias; the redistribute re-bind.
- `constCoefPrecheck()`; the `velocityResidualTolerance` rule; `diagnostics.…_stats()`.
- The exact-step bracket of §12.3. In this WO, `split` is forced false.
- **Accept, all of:**
  - G-BIT with the driver off;
  - with the driver on and `startup_steps=10**6`, on a PCG-selected ratio-50 VoF drop (32³):
    u, v, w, p, C bitwise identical to the driver-off run over 20 steps, and `p_increment` equal
    to P^{n+1} − Pⁿ bitwise at every step;
  - one raising check per precheck condition and for `SolverColocated`, in the new ctest
    `tests/kokkos/test_pressure_constant_coefficient.cpp`.

**WO-E2.3. The split projection** (§12.3 S1–S7, the operator switch, θ).
- **Accept:** G-E2-RED, G-E2-REC, G-E2-BAL and G-E2-RST on host-openmp **and** nvidia-cuda, all in
  the ctest above; G-BIT for the default path.
- If a gate misses, STOP and report. Do not tune a threshold.

**WO-E2.4. MPI gate.**
- The S1 allreduce may already be in WO-E2.3; its gate lands here.
- Add `tests/kokkos_mpi/test_pressure_constant_coefficient_mpi.cpp` at np = 1, 2, 4.
- **Accept:** G-E2-MPI; `ctest -R '_np[0-9]+$'` green.

**WO-E2.5. Documentation.**
- The "Pressure solve" section of `CLAUDE.md`: one paragraph with the scope, the order independence,
  and the restart recipe; update "select the driver last" and the ctest counts.
- A new section, "Constant-coefficient splitting (opt-in)", in `doc/variable_density_projection.md`,
  with §12.3, §12.5 and the recipe.
- The NAMING check (§12.7). Hand the register entries of §12.12 to the caller.
- **Accept:** `no_env_knobs`, `no_float_operator_casts` and `iteration_order` green; links resolve.

**WO-E2.6. Accuracy study** (decides adoption per case).
- `tests/study/vof_constcoef_pressure.py` with gates `static`, `translate`, `hysing1`, `hysing2`.
  Reuse `sphere_fractions` and friends from `vof_surface_tension.py`, and `build_hysing` and
  `run_hysing` from `vof_blocks_ns.py`.
- A `--constcoef` flag on the WO-0 bubble-column harness `tests/study/vof_perf/prof.py`.
- **Accept:** the G-E2-ACC table is complete, with a verdict for each case, and it is logged and
  reported to the user.

**WO-E2.7. Performance.**
- G-E2-PERF against §12.8.
- **Accept:** the numbers are logged. A miss is reported, not reverted.

### 12.10 Verification gates

**G-BIT** (§8), with the driver off, on every WO-E2.x.

**G-E2-RED — reduction** (ctest; P1).
- Case: a uniform-density VoF drop. The rho closure `[ρ, 0]` with ρ equal to the `set_rho` value;
  first assert that `rho0 == rho_` bitwise.
- Setup: σ > 0, μ = 0.1, 32³ triply periodic; also an anisotropic 32×32×16 box with extent
  (1, 1, 1). Both runs select `set_pressure_pcg(True, 500, 1e-12)`.
- Compare the split driver (`startup_steps=0`) with the exact path.
- **Pass:** u, v, w, p, C bitwise identical after 20 steps, and identical per-step PCG iterations.
- On CUDA a mismatch means the S7 twin or the S3 shape diverged textually. Fix it; never relax it.

**G-E2-REC — the discrete recurrence** (ctest; P3, 1-D).
- Case: a two-layer column at rest, walls in z, periodic x and y, 8×8×32. Gravity through the cell
  force `force_z = −g·ρ`. μ = 0 (if the solver rejects 0, use 1e-12 and a tolerance of 1e-7).
  `startup_steps=0`, from P = 0, PCG rtol 1e-13.
- Ratios 3 and 50, 30 steps. dt ×0.8 at step 10 (θ = 0.8) and ×1.25 at step 20 (θ capped at 1).
- Predict `g_n` per face from the recurrence of P3 (a_f from the actual ρ_f; `g_{−1} = g_0`).
- **Pass:**
  - `max_faces |g_meas − g_pred| / max|g_0| ≤ 1e-9` at every step;
  - `max|u| ≤ 1e-13` throughout.

This gate fixes the sign and coefficient of S3, the definition of ΔP, θ and ρ0 without a free
parameter.

**G-E2-BAL — balanced force at a density contrast** (ctest; P2).
- Case: `test_vof_surface_tension`'s P1 stationary droplet with constant κ (`setVofKappaConstant`),
  32³, R = 8.
- ρ_inside/ρ_outside ∈ {50, 0.02}, default `startup_steps`.
- **Pass:**
  - μ = 0, 30 steps: split `max|u| < 1e-14`, and P equal to σκC + const to 1e-10 relative (P3 of
    that test);
  - μ = 0.1, 200 steps: split `max|u| ≤ max(1e-14, 2 × exact path)` on the identical case.

**G-E2-RST — restart** (ctest).
- Case: a global-colour ratio-50 rising bubble, 32³.
- Compare 40 continuous split steps with 20 steps, save, a fresh solver, the §12.7 restore with
  `startup_steps=0`, and 20 more steps.
- **Pass:** u, v, w, p, C bitwise identical to the continuous run.
- Negative control: restoring without `p_increment` must differ, which proves the field matters.
- If the exact path's own restart of this case is not bitwise, the gate becomes "deviation ≤ the
  exact path's", and that is reported.

**G-E2-MPI** (np = 1, 2, 4; the `test_vardensity_mpi` protocol).
- Cases:
  - `walls-z`, hydrostatic, ratio 1000;
  - `jump-z`, periodic, ratio 1000, with the jump on the rank boundary;
  - a ratio-50 VoF drop with σ, 32³.
  All three are cut on the loaded axis and run with the driver on.
- **Pass:**
  - np = 1 bitwise against the single-rank reference;
  - np > 1: `du ≤ max(1e-15, 1e-11·umag)`, `dp ≤ max(1e-12, 1e-11·pmag)`, and PCG iterations ±1;
  - ρ0 bitwise identical on every rank and every np;
  - walls-z: `max|u| < 1e-14` and dP/dz error `< 1e-11` at every np.
- Rebalance sub-case, np = 2: `rebalance_by_weights` with a size-changing weight at step 10,
  continued to step 20. It must match the no-rebalance np = 2 run within the same tolerances, which
  proves that `p_increment` migrated.

**G-E2-ACC — accuracy against the exact projection at equal resolution, dt and rtol** (study).
These gates decide adoption per case. The exact path uses the case's production driver.
- **A1, static bubble with height-function curvature.**
  - Setup: 32³/R = 8 and 64³/R = 16; ρ_l/ρ_g ∈ {50, 1000} with the light phase inside; μ ratio
    0.02; dt = 0.5·dt_σ; 500 steps.
  - **Pass:** window-max Ca and final Ca of the split driver both ≤ 1.5 × the exact path's.
  - More than 10× → STOP: a defect, not a splitting error.
  - Report the time series; P3 predicts ringing with a period of ≈ 44 and ≈ 199 steps.
- **A2, translating bubble** (the register objection's own test).
  - Setup: triply periodic 32³/R = 8 and 64³/R = 16; ratio 50 and 1000 (the latter with
    `enable_vof_momentum`); uniform initial velocity U0·(1, 0.5, 0.25) with We = ρ_l U0² 2R/σ = 1;
    one domain crossing; `E = max_t max|u − U0|`.
  - **Pass:**
    - (i) E_split ≤ 1.5 × E_exact at dt = 0.25·dt_σ, the column's operating point;
    - (ii) `E_split − E_exact` at 0.25·dt_σ ≤ 0.5 × its value at 0.5·dt_σ. This identifies the
      splitting error by its dt-convergence. If it does not shrink, STOP. Criterion (ii) is
      vacuous if the difference is ≤ 0 at 0.5·dt_σ;
    - (iii) relative volume change ≤ 1e-12 in both.
- **A3, Hysing.**
  - Case 1: block path (`vof_blocks_ns.py hysing`), nx = 64. v_max, t(v_max) and y_c(3) each
    within **0.2 %** of the exact path.
  - Case 2: global colour with momentum consistency, nx = 64. v_max and y_c(3) within **1 %**.
  - Also report the exact path's nx 32 → 64 change beside each, so the user sees the splitting
    error against the discretization error.
- **A4, bubble column** (after the low-wall defect fix).
  - Run 2000 steps from `ckpt_t43`: the exact path at rtol 1e-10 (the reference) and at 1e-9 (the
    chaos control), and the split driver at 1e-10. Define `σ_S = |S(1e-10) − S(1e-9)|` for each
    statistic S.
  - **Pass:**
    - mean rise velocity over the last 1000 steps: `|split − exact| ≤ max(2σ_S, 1 %)`;
    - wall-region gas hold-up (gas volume within one bubble diameter of each y wall, averaged over
      the last 1000 steps), each wall: `≤ max(2σ_S, 5 %)` relative;
    - total and maximum per-bubble volume drift ≤ max(1e-8, 2 × exact);
    - per-step `max_open_divergence_projected()` ≤ 2 × the exact path's at the same step.
- **Adoption.** A case is rated for the split driver when all its gates pass. The table of rated
  cases goes into `doc/variable_density_projection.md`. Until then a case is not rated.

**G-E2-PERF** (bubble column, 50 split steps after start-up; same machines as §8 G-PERF).
- Report, against §12.8:
  - PCG iterations per step (model 7–9);
  - exactly one `setOpenness` after start-up;
  - ms per stage, CPU 1×24 and GPU.
- nsys: a split step adds no D→H read on a single-rank GPU run.

### 12.11 Risks and open questions

Each item carries a label and a default, so work can proceed unattended.

- **R-E2.1 [fact]: iterations on the constant operator.** WO-E2.0 measures it. **Default:** proceed
  if ≤ 11 at rtol 1e-10. At ≥ 12, stop and report, because §12.8 is then wrong.
- **R-E2.2 [fact]: accuracy at ratio 10³.** P3 predicts heavy-phase error modes at 0.9995 per step
  and a forced response amplified by ~10³. **Default:** the driver is rated only up to ratio 100
  until A2 at ratio 1000 and A3 case 2 pass.
- **R-E2.3 [fact]: does the bubble column pass A4?** **Default:** not rated until it does, and A4
  runs only after the low-wall defect fix.
- **R-E2.4 [pref]: the public names.** **Default:** `set_pressure_constant_coefficient`, the field
  `"p_increment"`, and `diagnostics.pressure_constant_coefficient_stats`.
- **R-E2.5 [pref]: widening the scope.** Candidates: inflow/outflow (the Dirichlet row and the
  outflow correction would need the explicit term), Picard iterations (ΔP per step rather than per
  iteration), phase change and divergence sources, drag, IBM, collocated. **Default:** they raise.
  Each is a later decision with its own gate.
- **R-E2.6 [fact]: can the constant operator drop the agglomerated bottom** (levels 6, smoothed
  4×3×2 bottom)? WO-E2.0 answers it. **Default:** the main-line bottom. If levels 6 is within +1
  iteration, recommend it in the case script only, with no code.
- **R-E2.7 [pref]: a user override of ρ0.** **Default:** none (D-E2.4).
- **R-E2.8 [pref]: a Chebyshev engine with fixed bounds for the constant operator** (no global dot
  products, for multi-rank scale). **Default:** not in v1. The `constCoefEngine_` slot takes it
  later.
- **R-E2.9 [fact]: is the exact path's restart of a VoF case bitwise?** **Default:** G-E2-RST
  compares against the exact path's deviation if not.
- **R-E2.10 [pref]: billed Snellius runs** for long bubble-column statistics or CPU timing.
  **Default:** workstation GPU for A4 (2000 steps). CPU timing uses the Snellius job already in use
  for §8 G-PERF only if the user has approved that allocation. Otherwise use the workstation at
  1×8, labelled as such.
- **R-E2.11 [risk, accepted]: restart continuity.** A restart across a dt change is not bitwise,
  because dtⁿ⁻¹ is not checkpointed (θ = 1 on the first step). This is documented.
- **R-E2.12 [risk]: start-up at ratio ≥ 10³ with PCG selected** can cap on the variable operator.
  **Default:** documented idiom:
  - leave the density mode's Chebyshev for the start-up steps;
  - to set the split solve's rtol, call `set_pressure_pcg(True, maxit, rtol)` and then
    `set_pressure_chebyshev(True, …)`. The second call keeps `pcgRtol_`.

### 12.12 Register entries this section creates (for the caller to add)

1. **Split pressure in incremental form.** A face pre-correction `u** = u* − dt(1/ρ_f − 1/ρ0)∇(θΔPⁿ)`,
   followed by the unchanged constant-density projection; the rotational term reads div(u**).
   Rejected: TBFsolver's non-incremental form (a second predictor family); div(u*).
2. **The CSF stays in the predictor under the split driver.** An equilibrium is an exact fixed
   point, and the split and exact schemes share their steady states. This scopes the "error ~ σκ"
   objection to transients, where G-E2-ACC measures it. Rejected: σκ∇C inside the extrapolated term
   (double counting).
3. **Face density of the explicit term = the predictor's arithmetic ρ_f.** Rejected: the mean of 1/ρ
   (TBFsolver), which is inconsistent with flow's predictor to first order.
4. **ρ0 = the exact global min of ρ per step, no safety factor.** Rejected: a fixed or user ρ0, and
   s ≠ 1 (see the P3 analysis).
5. **Extrapolation θ = min(1, dtⁿ/dtⁿ⁻¹).** Rejected: the uncapped linear form, unstable for a dt
   growth above ρ_max/(ρ_max − ρ0).
6. **History = the registry field `"p_increment"` + 2 exact start-up steps + explicit
   `startup_steps=0` on restart.** Rejected: storing Pⁿ⁻¹; `set_field` hooks.
7. **The split solve is MG-PCG on A0, independent of the driver selection;** the setter is
   order-independent of the rho closure. Rejected: following `useChebyshev_`/`useFcg_`.

---

## 13. Addendum (2026-10-03): the device bottom solve, redesigned (B1's premise measured false)

**Status:** DESIGN. Branch `vof-b1` (worktree `suite/flow-vof-b1`), on top of WO-6, E3 and WO-11.
§13 **supersedes** §4.1, §5.7's preconditioner, kernel, sub-hierarchy, eligibility condition 7 and
unit gate, and the B1 row of §0. It keeps §5.7's selector, flag plumbing and FCG skeleton, and it
keeps §5.14's component labels. Numbers marked [meas] are from the quiet RTX 5080 run quoted below.
Numbers marked [inf] are inferred from those measurements. Numbers marked [model] are derived here.

### 13.0 Decisions at a glance

| # | decision | rejected |
|---|---|---|
| D-1 | The device bottom is a **block-tridiagonal direct factorization** of the bottom operator in FP32. Planes run along one axis, and each plane's Schur complement is inverted explicitly (block Thomas). It is the preconditioner of B1's existing **FP64 FCG**, which runs to the same τ = 1e-5. There are two launches: a factor launch once per operator change, and one solve launch per V-cycle. | B1's V-cycle preconditioner (measured +5.3 ms/step); a dense 1536² factor or inverse; all-SM Krylov; Chebyshev to τ; a deeper or factor-3 bottom; an unrefined FP32 bottom; a vendor dense solver |
| D-2 | The null space is handled by **exact plane-local augmentation**: per component, a rank-1 term on its cells in its last-eliminated plane. The augmented solve is exact for compatible right-hand sides. | a diagonal shift (costs accuracy); pinning one cell (raises κ by about n) |
| D-3 | A float factor is admissible. Every quantity that carries the identity stays FP64: the stored operator, the A·1 = 0 projection, and the residual that decides convergence. Float changes only the iteration count, and failure is loud. | float anywhere the identity is asserted (the register's float-storage failure mode) |
| D-4 | **B1's V-cycle preconditioner and `sub_` are deleted** once the direct engine passes its gates. History keeps them at `ba8f769`. | keeping `'geometric'` as an option |
| D-5 | Selector: `diagnostics.set_pressure_bottom_solver('auto' \| 'direct' \| 'algebraic')`. `'auto'` selects `'direct'` where eligible. Host backends keep GraphAMG, bitwise. | — |

### 13.1 Problem, evidence, scope

**Measured** [meas] (2026-10-03, quiet RTX 5080, bubble column 128×96×64, rtol 1e-10, 300 steps
from `ckpt_t43`, `prof.py --bottom-solver`):
- `'algebraic'` (host GraphAMG): step 42.8 / 43.2 ms, projection 23.7 / 24.0 ms.
- `'geometric'` (B1, τ 1e-5): step 48.1 / 48.2 ms, projection 29.1 ms.
- Both engines: 13.32 PCG iterations per step, so 13.32 bottom solves per step.
- Bottom: 16×12×8 = 1536 cells, rebuilt every step (variable ρ).
- E1 (WO-0): an exact bottom one or two levels deeper costs +8.5 / +9.05 outer iterations per step.

**Split of the projection** [inf]. With G = the B1 bottom, A = the GraphAMG bottom and N = the rest
of the projection: G − A = 5.3 ms is measured. The single-SM `GeoBottomKernel` measured 18.9 ms/step
at τ 1e-5 on a shared GPU, and a one-SM kernel is barely slowed by co-tenants. That gives G ≈ 18,
A ≈ 13 and N ≈ 11 ms (each ±1). This is consistent with §3.2's ≈ 0.85 ms per outer iteration after
WO-2/3/5. WO-D0 confirms it.

**Why B1 is slow** (from the WO-6 measurements and the kernel source):
- Each FCG iteration is ≈ 112 dependent team phases. The in-kernel V-cycle accounts for ≈ 102 of
  them, 48 of which are the 12 coarsest sweeps on a 12-cell level. At 10–12 iterations that is
  ≈ 1250 phases per solve.
- 1.35 ms per solve divided by ≈ 1250 phases gives **≈ 1.0 µs per dependent phase**. That is the
  calibration constant used below.
- About a third of the cost is FP64 work on one SM: ≈ 2 M DP instructions per solve at the 5080's
  2 DP lanes/clk/SM.
- So B1 is *phase-bound* on every GPU, not only FP64-bound on this one. H100 would not rescue it.

**Scope:** single rank, device backends, the singular agglomerated bottom (the B1/B1b eligibility),
bubble column first.

**Out of scope:**
- the host bottom (GraphAMG, bitwise);
- MPI (see §13.9 Q-D7);
- better coarse operators (S3);
- an FP64 factor on FP64-strong GPUs (Q-D2).

### 13.2 Constraints and invariants

- **FP64 operator.** The FCG applies the bottom level's own stored operator (AC, AFX, AFY, AFZ),
  through the A0 cell body, exactly as B1 does. A·1 = 0, the per-component mean projections and the
  stopping residual all stay FP64.
- **Same bottom semantics.** Relative ∞-norm residual ≤ τ = 1e-5 (E3, recorded) and cap 100. On a
  non-finite scalar: x = 0 plus the device flag that rides the A6 packet (unchanged).
- **Solids (B1b).** Solid cells (comp < 0) keep x = 0 and enter r as 0. There are 1–64 components;
  more go to GraphAMG.
- **Reproducibility.**
  - The factor and M are **bitwise independent of the team size T**. Every stored or output scalar
    is computed by one thread, in a summation order fixed by the algorithm and the compile-time tile
    size, never by T. There are no team reductions inside the factor or M.
  - The FCG's dots stay team reductions, as in B1: deterministic for a fixed T and GPU.
  - Runs are bitwise run-to-run.
- **No transfers.** No D↔H copy inside `step()` on the eligible path. The factor launch is decided
  by a host flag, not by a read.
- **Unchanged elsewhere.** Host results, the distributed path and the `'algebraic'` engine stay
  bitwise.

### 13.3 The decision, and why the alternatives lose

The bottom needs about 13 exact-to-1e-5 solves per step with an operator that changes every step,
on a GPU where one SM delivers ≈ 5 G FP64 FMA/s and where each dependent in-kernel phase costs
≈ 1 µs. So the cost of a candidate is roughly (dependent phases per solve) × 1 µs + (FP64 work on
one SM). A direct method attacks both terms:
- it needs 2P ≈ 24–34 phases per application instead of ≈ 112 per FCG iteration;
- it needs 2 applications per solve instead of 11;
- its heavy arithmetic moves to FP32, which runs at 64× the FP64 rate on this card.

The structured sparsity is what makes the direct method cheap. With the longest suitable axis
slowest, b = cells per plane ≤ 128. Then:
- the factor is P·b³/2 ≈ 13 M FMA, against the dense n³/3 ≈ 1.2 G;
- an application reads 2·P·b² FP32 values ≈ 1.6 MB, against 9.4 MB for a dense factor.

| candidate | why it loses (5080 unless stated) |
|---|---|
| (a) dense Cholesky or inverse of the 1536² matrix | FP64 factor ≈ 1.4 ms/step even at full-GPU peak, through a tiled multi-launch library. In one team, each solve streams 9–19 MB, ≥ 60–120 µs. An FP32 dense factor with FP64 refinement fixes the factor but still moves 6× the block-tridiagonal traffic per application. It also needs cuSOLVER/rocSOLVER plus host LAPACK, or Kokkos-Kernels: a new dependency per backend, buying nothing once the sparsity is used. The primitives D-1 needs are b×b ≤ 192² team kernels (≈ 300 lines). |
| (b) all-SM Krylov with device scalars | Every dependent step becomes a launch (≈ 4–5 µs). A V-cycle-preconditioned iteration is ≥ 110 launches. Unpreconditioned CG needs ≈ √κ·ln(1/τ) ≈ 20–60 iterations × ≈ 4 launches. Both cost ≥ 0.4 ms per solve. |
| (c) Chebyshev or block-Jacobi to τ | Degree ≈ √κ·ln(2/τ) ≈ 240 matvecs per solve (scaled κ ≈ 400): ≈ 1 ms in one team, 240 launches across all SMs. |
| (d) factor-3 / deeper coarsening, bottom ≈ 3³–6³ | E1 measured +8.5 outer iterations ≈ +7 ms/step: more than the whole bottom. It is blocked on S3. If S3 ever allows a smaller bottom, D-1 only gets cheaper (∝ n·b²). |
| (e) FP32 bottom used *unrefined* in the FP64 V-cycle | Its error ≈ ε32·κ is unverified and grows silently with contrast. That is exactly the register's float-storage failure mode ("Float MReal operator storage silently breaks A·1=0"). D-3 is admissible under the register's precision rule ("identity-bearing quantities stored in the precision the identity is asserted") precisely because it is refined in FP64 to τ. |
| B1 with an FP32, shared-memory V-cycle | It is still ≈ 112 phases × 11 iterations ≥ 0.5 ms per solve. |
| FP64 factor everywhere | On the 5080 the factor runs on one SM in FP64: ≈ 13 M / 5.2 G ≈ 2.5 ms/step. On H100 it saves one FCG iteration (Q-D2). |

### 13.4 The scheme (normative)

**13.4.1 Matrix, scaling, ordering.**
- **Diagonal.** On fluid cells, d_i = −Σ over the six faces of AF, resummed in FP64 as GraphAMG
  does, so the factored matrix has A·1 = 0 exactly per component.
- **Scaling.** s_i = 1/√d_i in FP64. Set s_i = 0 where comp(i) < 0 or d_i ≤ 0.
- **Scaled matrix.** Ã = S·A·S, assembled in FP64 and cast to FacReal (= `float`) when stored.
  - Diagonal 1 on fluid cells. Rows with s_i = 0 are identity rows.
  - Off-diagonal Ã_ij = s_i·AF_face·s_j, **accumulated over each cell's six faces in the order
    −x, +x, −y, +y, −z, +z**. Periodic axes of length 1 or 2 therefore need no special case.
- **Slow axis s, decided at hierarchy build from the domain BC flags.**
  - The candidates are the non-periodic axes; if there are none, all three axes.
  - Choose the longest candidate; on a tie the higher index wins (z > y > x).
  - If its plane size b = n/n_s exceeds `kBottomMaxPlane = 192`, try the remaining axes in
    decreasing length. If none fits, the bottom is ineligible.
- **Indexing.** Plane k is the coordinate along s, and P = n_s. The in-plane index is
  q = (lower other axis) + n_lower·(higher other axis). The factor-order index is k·b + q.
- **Block structure.**
  - Â_k: the dense b×b in-plane block.
  - e_k ∈ R^b: the diagonal coupling of (q,k) to (q,k−1). For P ≥ 3 with a periodic slow axis,
    e_0 is the wrap coupling of plane 0 to plane P−1.
  - With P = 2 and a periodic slow axis, both faces accumulate into e_1, and the problem is treated
    as non-periodic.

**13.4.2 Null space: plane-local augmentation (exact).**
- For each component c, let k_c = the largest plane index among its cells, and m_c = the number of
  its cells in plane k_c. The label kernel computes both (§5.14 extended; geometry time).
- Add 1/m_c to Â_{k_c}(i,j) for every pair i, j ∈ c ∩ plane k_c.
- **Why this is exact.** The augmented matrix is A′ = A + Σ_c σ_c·w_c·w_cᵀ, with
  w_c = S⁻¹·1_{c∩k_c} > 0 supported on c. For a compatible r (1_cᵀ r = 0) we have
  1_cᵀ A′x = σ_c (1_cᵀ w_c)(w_cᵀ x) = 0, so w_cᵀ x = 0 and A x = r **exactly**.
- **Why it is well conditioned.** The lifted eigenvalue is ≈ m_c/|c| ≈ 1/P, above the smallest
  nonzero eigenvalue of Ã (≈ 0.006–0.025), so κ is not inflated.
- M's output is then made fluid-mean-free per component in FP64 (`removeMeanBottom`, unchanged).

**13.4.3 Factor (one team, FacReal arithmetic, at the first bottom solve after each `setOpenness`).**

*Non-periodic slow axis* (block Thomas with explicit inverses):
```
Σ_0 = Â_0;                          Q_0 = Σ_0⁻¹
Σ_k = Â_k − diag(e_k) Q_{k−1} diag(e_k);  Q_k = Σ_k⁻¹      k = 1..P−1
```

*Periodic slow axis, P ≥ 3:*
- T = planes 0..P−2, factored as above (Q_0..Q_{P−2}).
- Y = T⁻¹·B, where B has block 0 = diag(e_0), block P−2 = diag(e_{P−1}), and zeros elsewhere.
  It is computed as a matrix recurrence: one phase per plane, forward then backward, the same
  arithmetic per column as the solve.
- Σ_B = Â_{P−1} − diag(e_0)·Y_0 − diag(e_{P−1})·Y_{P−2}; Q_B = Σ_B⁻¹, stored as block P−1.

*Dense SPD inverse* Q = Σ⁻¹. Only the lower triangle of Σ is read. Tile Bt = 16 (constexpr). For
each tile t:
1. **Diagonal tile.** One thread does an unblocked column Cholesky. For each column j:
   - pivot p = Σ_jj − Σ_{m<j, m∈t} L_jm² (ascending m);
   - fail if !(p > τ_piv = 1e-6);
   - L_jj = √p;
   - for each i > j in the tile: L_ij = (Σ_ij − Σ_m L_im L_jm) / L_jj.
2. **Panel.** One thread per row below the tile, with the same formula over the tile's columns in
   ascending order.
3. **Trailing update.** For i ≥ j below the tile: Σ_ij −= (Σ_{m∈t} L_im L_jm), as one partial sum
   in ascending m.

Barriers separate 1 / 2 / 3. After all tiles:
- **W = L⁻¹**, one thread per column j: W_jj = 1/L_jj; for i > j in ascending order,
  W_ij = −(Σ_{m=j}^{i−1} L_im W_mj) / L_ii.
- **Q = WᵀW**, with Q_ij = Q_ji = Σ_{m=max(i,j)}^{b−1} W_mi W_mj (ascending m). Q is
  therefore bitwise symmetric.

*Failure handling:*
- On a pivot failure or a non-finite value, restart the whole factor with δ added to every scaled
  diagonal: δ = 1e-4, then 1e-2.
- A third failure can only come from non-finite input. Then `facOk = 0`, the solve kernel takes
  B1's failure path (x = 0 plus the flag), and the result is `solveFailed_`.
- The restart count goes to `info` and into the `[mg]` debug trace.

**13.4.4 M(r) (inside the solve kernel; r and z FP64 in the level's ext layout).**
1. Cast phase: r̃ = FacReal(s ∘ r) in factor order. B1's z → zp copy is fused into this phase.
2. Forward, one phase per plane:
   - g_0[i] = Σ_j Q_0[j][i]·r̃_0[j];
   - g_k[i] = Σ_j Q_k[j][i]·(r̃_k[j] − e_k[j]·g_{k−1}[j]).
   The bracket is evaluated inline by each thread. Ascending j, a single accumulator, one thread per
   output i.
3. Backward: x̃_{P−1} = g_{P−1}; x̃_k[i] = g_k[i] − Σ_j Q_k[j][i]·(e_{k+1}[j]·x̃_{k+1}[j]).
4. Border (periodic, P ≥ 3):
   - run steps 2–3 over T to get u;
   - x̃_B = Q_B·(r̃_B − e_0 ∘ u_0 − e_{P−1} ∘ u_{P−2});
   - one phase: x̃_k = u_k − Y_k·x̃_B.
5. Uncast phase: z = s ∘ double(x̃) on fluid cells, 0 elsewhere. Then `removeMeanBottom(z)`.

**13.4.5 FCG.** B1's loop (§5.7 pseudocode, `GeoBottomKernel::operator()`) is unchanged except that
`vcycle(t, r, z, cnt, &zp)` becomes M (§13.4.4). τ, the cap, Polak–Ribière, the masking, the
per-component means and the flag are all unchanged. The expected iteration count is 2, because
M's relative error is ≈ c·ε32·κ(Ã) ≈ 1e-5–1e-3.

**13.4.6 Eligibility.** §5.7 conditions 1–6 (6 = B1b's 1–64 components), plus:
- n ≤ 8192 (unchanged);
- a slow axis with b ≤ 192 exists.

Condition 7 (a sub-level exists) is dropped. Otherwise GraphAMG runs, unchanged.

**13.4.7 Layout and placement.**
- **Device storage** (allocated with the hierarchy; ≈ 1.5–1.8 MB for the case):

  | name | type and shape | contents |
  |---|---|---|
  | `Q` | `View<FacReal***>` [P][b][b], LayoutRight | read as Q(k, j, i), so thread i's reads are coalesced; it equals row i by symmetry |
  | `Y` | [P−1][b][b] | stored transposed, Y(k, j, i) = Y_k(i, j) |
  | `e` | [P][b] | the slow-axis couplings |
  | `s` | `View<double*>` [n] | the scaling |
  | `g`, `u` | [n] FacReal | substitution vectors |
  | Σ, L, W scratch | 3 × b² | global memory; level-0 scratch is a permitted, bitwise-neutral optimisation |
  | `kc`, `aug` | [64] | the per-component augmentation |
  | `facOk`, `restarts` | | factor status |

- **New header `src/mg_bottom_direct.hpp`.** Container-free: the factor functor
  `BottomFactorKernel<FacReal>` and the M device function, taking Views. It is included by
  `mac_cutcell_mg.hpp`.
- FacReal is a template parameter; `float` is the only production instantiation. The unit test also
  instantiates `double`.

### 13.5 Cost model, checked against the measurements

Calibration from B1 [inf]:
- ≈ 1.0 µs per dependent single-team phase;
- FP64 ≈ 5.2 G FMA/s per SM; FP32 ≈ 333 G FMA/s per SM;
- L2 → one SM ≈ 160 GB/s.

The case's BCs decide the shape:
- walls in y: s = y, P = 12, b = 128, no border;
- fully periodic: s = x, P = 16, b = 96, with border.

| item (5080, quiet) | phases | [model] |
|---|---|---|
| M | 2P + 4 (≈ 28; border case ≈ 38) | 28–45 µs (memory ≈ 10 µs, hidden under latency) |
| FCG iteration excluding M (FP64 matvec ≈ 4 µs, 4 reductions, mean) | ≈ 10 | ≈ 14 µs |
| one bottom solve (2 iterations) | ≈ 90 | 85–125 µs |
| solves per step (× 13.32) | | 1.1–1.7 ms |
| factor (≈ 25 phases + ≈ 10 µs serial chains per plane; border adds Y ≈ 30 phases) | | 0.35–0.55 ms |
| **direct bottom per step** | | **1.5–2.2 ms** |
| B1 / GraphAMG today | | 18 / 13 [inf] |

Expected result:
- projection ≈ 11 + 1.9 ≈ **13 ms** (23.7 today);
- step ≈ **32 ms** (42.8 today; −11 ms);
- per bottom solve, ≈ 90 phases instead of B1's ≈ 1250, and ≈ 50 k DP instructions instead of
  ≈ 2 M.

**H100** [model]:
- FP32 factor: ≈ 1.2–1.8 ms/step, with the same phase latency.
- An FP64 factor would need 1 FCG iteration: ≈ 0.8–1.2 ms/step.
- So the FP64 factor saves ≈ 0.4–0.6 ms/step there and costs ≈ +2 ms/step on the 5080 (Q-D2).

If WO-D0 finds A much smaller than 13 ms, the design stands: it removes the transfers, and saves
A − 2 ms. Only the expected step time changes.

### 13.6 What happens to B1

| part | fate |
|---|---|
| `GeoLabelKernel` | **kept**, extended with `kc` and `aug` |
| eligibility skeleton | **kept** |
| FCG loop | **kept** |
| flag through the A6 packet | **kept** |
| `info` / team-size trace | **kept** |
| selector | **kept** |
| in-kernel V-cycle (`smooth`/`residual`/`restrictZero`/`prolong`/`fill`/`neumannCell`/`prolongGhosts`/`vcycle`) | **deleted** in WO-D3 |
| `sub_`, `buildGeoSub`, the lazy sub-level coarsening | **deleted** in WO-D3 |
| `kGeoPre/Post/Sweeps`, `neu[]` | **deleted** in WO-D3 |
| `'geometric'` selector value | **deleted** in WO-D3 |
| `Geo*` / `kGeo*` identifiers | renamed `Bottom*` / `kBottom*` in WO-D3; they no longer name what the code is |

The P3 handoff's open items:
1. The bitwise-M unit gate is **moot**.
2. `'auto'` selects `'direct'`.
3. The spelling question becomes Q-D4.
4. The register text is in §13.10.

### 13.7 Work orders

- **WO-D0 (no code, about 20 min on a quiet GPU).**
  - Run the P3 handoff's `xfer.sh quiet …/frozen_w11/cuda --flux device --fixdt 1.5e-3`.
  - Record `GeoBottomKernel` ms/step = G, then N = 29.1 − G and A = 23.85 − N.
  - Record the bubble column's BC per axis, and hence s, P and b.
  - *Accept:* the numbers are in the log, and G-PERF's targets are fixed from them.
- **WO-D1: factor and M, not wired.**
  - `mg_bottom_direct.hpp` and the label-kernel extension.
  - The host-side slow-axis rule and caps.
  - Storage, and test hooks to run the kernels on a host backend.
  - The new ctest `bottom_direct` (CUDA and host).
  - *Accept:* U1–U7 pass; both batteries are green; host and CUDA `state_hash` are unchanged
    (nothing selects the engine yet).
- **WO-D2: wire M into the FCG kernel** (a `precond` member: V-cycle | direct).
  - Selector value `'direct'`; `'auto'` → direct.
  - Lazy factor launch keyed on a host stale flag set by `setOpenness`.
  - *Accept:* §13.8 B, C, D and E. This is the recorded numerics change; the reference is
    `'algebraic'` on the same build.
- **WO-D3: retire B1's preconditioner** (§13.6 deletions and renames; drop `'geometric'`).
  - *Accept:* the CUDA 50-step bubble-column dump is bitwise identical to WO-D2's (all arrays);
    host G-BIT holds; batteries are green; clang-format 18.1.8 is clean.
- **WO-D4: log, register text, NAMING row text, handoff.**
- *(Not main line)* **WO-D5: `FacReal = double` on FP64-strong architectures.** Only after Q-D2.

### 13.8 Verification gates

**A. Unit (`bottom_direct`, CUDA and host).** The problems are: periodic with border (P ≥ 3);
periodic with P = 2; walls-y (no border); the B1b solid sheet with 3 components, periodic and
walls-y; a 1-cell pocket. All at coefficient ratio 50.

| id | check | threshold |
|---|---|---|
| U1 | FP64 exactness: one M on a compatible r, `double` instantiation | ‖Ax − r‖∞/‖r‖∞ ≤ 1e-12; per-component mean of x ≤ 1e-15·max\|x\| |
| U2 | float accuracy: one M | relative residual ≤ 1e-3 |
| U3 | T-independence: factor storage and M(r) | **bitwise** identical for T ∈ {32, 64, 128, 256, T_max} (CUDA) and {1, 2, 4} (OpenMP) |
| U4 | FCG convergence to τ | ≤ 3 iterations; x = 0 exactly on solids; per-component mean ≤ 4e-16·max\|x\| |
| U5 | shift path: test hook sets τ_piv = 1e30 for the first attempt | restarts = 1; FCG reaches τ; no flag |
| U6 | non-finite coefficient | flag set → `lastSolveFailed()`; x = 0 |
| U7 | float M against a host FP64 dense solve of A′ (16×12×8) | max relative difference ≤ 1e-3 |

**B. Numerics (bubble column, CUDA, 50 steps from `ckpt_t43`, against `'algebraic'` on the same
build).**
- N50: max relative difference over u, v, w, p, C ≤ **1.499e-11** (WO-0 N50; B1 reached 2.0e-14;
  expect ≤ 1e-13).
- Outer iterations **identical on every step** (653 total); fallback per §8 G-NUM 2.
- Divergence ratio ≤ 2 (expect ≤ 1.001).
- §8 G-NUM 4 physics (static drop, Hysing 1).
- Over 300 steps: inner FCG iterations max ≤ 3 and mean ≤ 2.5; shift restarts 0; flag never set.
- Two runs bitwise identical.

**C. Solids battery** (`p3/solids.py`: slab1, slab2, cyl, rings, pack; constant and variable ρ).
- Outer iterations identical on every step against `'algebraic'`.
- Velocities within each case's rtol×10 floor.
- `'algebraic'` bitwise to origin/main.

**D. Bitwise, transfers, batteries.**
- Host `state_hash` + np2, and the bubble column 1×8: identical / bitwise.
- `ctest -LE bench` green on host, CUDA and the `PECLET_FLOW_OPERATOR_DOUBLE=OFF` tree. CUDA
  `state_hash` is identical, or re-baselined in the commit with the old → new table if a case newly
  qualifies.
- Transfer gate:
  - bubble column, ≥ 1 KiB per step: H→D 0, D→H 2 (the WO-8 packets);
  - `pack:slab`: 0 / 0;
  - small reads per step unchanged from WO-11.

**E. G-PERF** (quiet 5080, the brief's harness, 300 steps; nsys kernel sums).
- Factor plus solve kernels ≤ **2.5 ms/step**; factor launch ≤ 0.6 ms.
- Projection ≤ N + 2.5 ms.
- Step ≤ 43.0 − (A − 2.5) ms. With the [inf] split: projection ≤ 13.5 ms and step ≤ 32.5 ms.
- Report the measured values against §13.5.

### 13.9 Risks and open questions (each has a default)

- **Q-D1 [fact]: the N / A / G split.** WO-D0 measures it. *Default:* N ≈ 11, A ≈ 13, G ≈ 18. The
  design does not depend on it.
- **Q-D2 [fact, needs billed H100 time → user]: FP32 or FP64 factor on FP64-strong GPUs.**
  *Default:* `float` everywhere. A per-architecture compile-time choice of `double` would be a new
  recorded decision, taken after measurement (WO-D5).
- **Q-D3 [pref]: delete B1's preconditioner.** It is dominated: slower than GraphAMG on the 5080,
  phase-bound on every GPU, and selected by nothing. *Default:* delete (WO-D3).
- **Q-D4 [pref]: spelling `'direct'`.** *Default:* `'direct'`. The implementer checks
  `../docs/NAMING.md` and drafts the row (umbrella file, for the caller).
- **Q-D5 [fact]: the caps b ≤ 192, n ≤ 8192.** *Default:* as stated. When a 16³ bottom (b = 256)
  first matters, measure direct against GraphAMG before raising the cap.
- **Q-D6 [pref]: the direct engine on host backends.** *Default:* no. GraphAMG stays and host
  results stay bitwise; the CPU projection is a separate question.
- **Q-D7 [pref]: MPI.** *Default:* not in this package; eligibility stays single-rank. What would
  change:
  1. Allgatherv the bottom face form once per operator change, and the rhs once per V-cycle, on
     device buffers. CUDA-aware MPI is needed to honour the no-transfer directive.
  2. Every rank factors and solves redundantly, as GraphAMG does today. Bitwise agreement across
     ranks needs the same GPU model and the same T (check T with an Allreduce min/max; on mismatch,
     use GraphAMG).
  3. Each rank extracts its own block.
- **Q-D8 [pref]: promote `mg_bottom_direct.hpp` to `core::solver`.** *Default:* not now; it is
  container-free so it can move later.
- **Risk R-D1: per-phase cost.** If the FP32 gemv phases are memory-bound above 1 µs, M rises to
  ≈ 45–55 µs and the bottom to ≈ 2.5 ms/step. Even then it saves ≥ 8 ms if A ≈ 13. *Levers, in
  order:*
  1. level-0 scratch for the current Q_k (bitwise-neutral);
  2. a FIXED 2–4-way split of each dot (a new fixed order: re-run U1–U3 and B; it is not
     T-dependent);
  3. Y on many teams as a separate launch (bitwise-neutral: the same per-column arithmetic).
- **Risk R-D2: high contrast with ε32·κ ≳ 1** (cut-cell beds at ρ ratio 1e4). FCG iterations rise
  or the shift restarts. Correctness is held by the FP64 τ, and the cost shows in the trace. Gate C
  covers it.
- **Risk R-D3: the stored AC differs from the resummed d** (float-storage tree). M then
  preconditions an exactly singular neighbour of the stored operator, and the FCG still converges on
  the stored operator. Gate D's OFF tree covers it.

### 13.10 Register entries this section creates (for the caller to add)

1. **Device pressure bottom = block-tridiagonal FP32 direct factor (explicit Schur-complement
   inverses) preconditioning FP64 FCG to τ = 1e-5; one factor launch per operator change, one solve
   launch per V-cycle.**
   - Rejected: B1's single-team V-cycle preconditioner (measured +5.3 ms/step on the 5080; ≈ 1250
     dependent phases per solve); dense 1536² Cholesky or inverse; all-SM Krylov; Chebyshev to τ;
     a deeper or factor-3 bottom (E1: +8.5 iterations); a vendor solver dependency.
2. **Singular bottom handled by exact plane-local augmentation (rank-1 per component on its
   last-eliminated plane).** Rejected: a diagonal shift (accuracy), single-cell pinning
   (κ × ≈ n).
3. **A float factor is admissible inside an FP64-verified bottom; an unrefined float bottom is
   not.** This is the precision-policy rule applied: A·1 = 0, the projection and the stopping
   residual stay FP64.
4. **B1's V-cycle preconditioner and sub-hierarchy are retired** (code at `ba8f769`). The component
   labels, FCG, flag and selector carry over.
5. **Selector spelling** `set_pressure_bottom_solver('auto'|'direct'|'algebraic')` (pending
   Q-D4).

---

## 14. Addendum (2026-10-04): the host (OpenMP) step, from 3.1× TBFsolver toward parity

**Status:** DESIGN. Worktree `suite/flow-cpu14` (branch `cpu14` = flow origin/main `6719f01`). Brief:
`~/Codes/bubble_column_perf/BRIEF_CPU_PROJECTION.md`. §14 **supersedes** the CPU columns of §0 and
§6, the host clause of §13 D-5 ("host backends keep GraphAMG") and §13.9 Q-D6's default. WO-9/10/12
are in flight and are assumed to land as designed in §5.10/§5.8/§5.13.

Markers: **[meas-S]** Snellius genoa, job 27519212, flow `d02d3b0`, 1×24 and 8×3, rtol 1e-10
(`/projects/0/prjs1022/peclet/bubble-cpu/results/kprof-d02d3b0/summary.txt`); **[meas-W]** this
note's workstation runs (5965WX, load 39–57: only serial code and launch structure are read from
them); **[model]** derived here. All savings are ms/step.

### 14.0 Decisions at a glance

| rank | item | kind | 1×24 | 8×3 | build | risk |
|---|---|---|---|---|---|---|
| 1 | **H-0** benchmark protocol: np = 1 runs without `init_mpi`; the case's rtol 1e-8; Zen layouts that never straddle an L3; `PECLET_FLOW_HOST_ARCH=znver4` | no `src/` change | −9 (np 1), −16 (rtol) | −20 to −25 (layout), −15 (rtol) | S | low |
| 2 | **H-1** `'direct'` bottom on host backends (single rank); the FCG's reductions single-lane on host | recorded numerics | −13 | 0 (see H-6) | S | low |
| 3 | **H-4** container on host: exact-count list launches with a dynamic schedule, row-mapped region kernels, flattened bbox, row scans, one-launch moves | bitwise | −11 to −16 | −8 to −12 | M | low |
| 4 | **H-3** single-launch stragglers and fill/BC launches under rule H | bitwise | −10 to −13 | −5 to −7 | S–M | low |
| 5 | **H-2** prolong with integer coarse indices and weights | bitwise | −5 | −4 | S | low |
| 6 | **H-6** distributed host `'direct'` (§13 Q-D7, host half): redundant per rank | recorded | 0 | −13 | M | med |
| 7 | **H-7** distributed box passes: shell-only shell launch, simd interior | bitwise | 0 | −3 to −5 | S | low |
| 8 | **H-5** Krylov reduction fusion on host (after WO-10) | bitwise vs post-WO-10 | −2 | −2 | M | low |

**End state [model]** (§14.4): 1×24 ≈ **69 ms** (60–78) against TBFsolver's 46, i.e. ≈ 1.5×; 8×3 ≈ 83
(a CCD-aligned 6×4 is the recommended 24-core MPI layout); 192 cores ≈ 55 (45–65, low confidence)
against 25. **The brief's ≤ 50 ms at 1×24 is not reached by structural work plus H-1.** Closing the
remaining ≈ 19 ms needs user-owned options stacked (U-1 case rtol 1e-6 ≈ −9; U-2 E1 float V-cycle ≈ −8)
plus the conditional F4 (≈ −3); see Q-H6, Q-H7.

### 14.1 What the profile measured: five findings that reorder the work

1. **The 1×24 profile ran the distributed code path.** `bench_peclet.py` (and
   `tests/study/vof_perf/run_mpi.py`) call `mpi_block` + `init_mpi` at np = 1, so
   `CutcellMG::distributed_` is true and none of WO-3…5's single-rank paths runs: A3's wrap reads
   (`fusedWrapReads()` needs `!distributed_`), A5's fused zero, A6's resident loop. Fingerprints
   [meas-S]: `core::halo::selfCopy` 493.8 launches (7.90 ms) is GridHalo's periodic self-exchange;
   `cc_smooth_box` 626.4 launches (17.9 ms) is an interior plus a shell launch per colour pass;
   `cc_residual_box` 78.3, `cc_apply_exact_box` 28.1, `zeromemset` 78.1 (the distributed
   `deep_copy(cs.x, 0)`). The single-rank profile of the same code has `cc_smooth` 313.2 launches,
   no selfCopy and no box kernels [meas-W, `bubble_column_perf/perf/main_host/kern_omp8.txt`]. The
   brief's "fuse the periodic self-copy into the stencils" is therefore built (A3) but was not
   measured.
2. **The ≈ 22 ms outside kernels is the host GraphAMG bottom.** [meas-W] `PECLET_FLOW_MG_DEBUG=3`
   level timer, frozen main host module (`perf3/frozen_main/omp`), `prof.py 20 --warm 3 --pcg`, 1×8,
   two runs: L3 (16×12×8: `graphAmgSolveBottom` including its per-step `buildAmg`) **0.375 s and
   0.374 s per 250 V-cycles = 1.50 ms per V-cycle**, while the parallel levels moved 14 % between the
   runs. At 13.0 iterations (≈ 13.5 V-cycles) that is ≈ 20 ms/step, which is nearly all of the
   143.6 − 121.2 = 22.4 ms outside kernels [meas-S]. At rtol 1e-8 (≈ 11.6 V-cycles) it is ≈ 17 ms.
3. **The batched container's list kernels are serial per block on host.** Under GCC, Kokkos'
   OpenMP backend compiles a static `RangePolicy` to `schedule(static)` without a chunk
   (`OpenMP/Kokkos_OpenMP_Parallel_For.hpp`): one contiguous block of iterations per thread. The
   WO-8 list kernels launch over Σ upper bounds (each block's inner region) and exit at `q ≥ end`;
   a block's active entries (≈ 10 % of its sub-range) sit at its head and land on one thread. With
   16 blocks on 24 threads at least 8 threads idle and the largest block sets the time. Tier 3
   (`batch_curv_list` pass 1: PV fits on ≈ 35 % of interfacial cells) is the most expensive of
   these. The region kernels map a flat index with three 64-bit div/mod per cell plus a job search.
   This finding is from the code; WO-H0 measures it before WO-H4 relies on it.
4. **The baseline ran rtol 1e-10; the case runs 1e-8** (register, 2026-10-03; 10.6 iterations per
   step instead of 13.0).
5. **The 8×3 imbalance is placement.** `--distribution=block:block --cpu-bind=cores` on cores 0–23
   (CCDs 0–2 of an EPYC 9654, 8 cores per CCD) gives rank 2 cores 6–8 and rank 5 cores 15–17: each
   straddles two L3s. Ranks 2 and 5 show 122.9 and 124.4 ms kernel time against 98.0–101.7 for the
   others [meas-S], and the others wait in MPI.

**Scope:** host execution of the bubble-column step; the projection first, then the container and
the momentum stragglers. **Out of scope:** device paths (every change here is either bitwise on
device or a host-only `if constexpr` branch); S3; E1/E2; core changes (merging GridHalo's selfCopy
into pack needs a core tag: deferred); the momentum algorithms; multi-node.

### 14.2 Constraints added to §2

- A host-only branch selects on `std::is_same_v<typename Exec::memory_space, Kokkos::HostSpace>`,
  as `ccdetail::ccRows3` does. The device branch keeps its bits.
- Bitwise items pass G-BIT (§8) on host 1×8 and 1×24, nvidia-cuda, `state_hash` + np2 and the MPI
  np-tests. Recorded items pass G-NUM-H (§14.6).
- The register holds "np = 1 bit-exactness is the gate for every distributed default". That gate
  needs `init_mpi` at size 1 to keep running the distributed path, so H-0 is a **protocol** change,
  not a change of `init_mpi`.
- No new host `MDRange3<` in a hot path (rule H, §4.5). The host-serial cutoff (8192) stands.

### 14.3 The items

**H-0. Benchmark protocol (no `src/` change).**
- **np = 1 is single-rank.** When `comm.Get_size() == 1`, skip `mpi_block` and `init_mpi`: in
  `tests/study/vof_perf/run_mpi.py` (and add `--dump` there, same format as `prof.py`), and in the
  caller's Snellius `bench_peclet.py`. [model] −9 at 1×24 and rtol 1e-8: selfCopy −4.9 (net of the
  single-rank fills before prolong), smoother launches and shell scans −3.7, residual/matvec box
  −0.65, zero fills −0.65; the single-rank momentum fills (`ibm_pfill`, 3 MDRange2 per fill) give
  back +1 to +2 until H-3(d) lands.
- **rtol.** `bench_peclet.py` uses the case's registered 1e-8 in both `set_pressure_pcg` calls.
  [model] projection −14.2, momentum −1.5 (its rtol follows the pressure rtol).
- **Layouts on Zen.** A rank's cpuset never spans two CCDs. On genoa (8 cores per CCD) the
  24-core layouts are 1×24, 3×8, 6×4, 12×2 and 24×1 on cores 0–23; 8×3 only as 8 CCDs × 3 cores
  (`--cpu-bind=mask_cpu`, one 3-core mask per CCD). At 192 cores, 24×8 (one rank per CCD) is
  measured next to 64×3.
- **Site build.** `PECLET_FLOW_HOST_ARCH=znver4`. Under A1 (`-ffp-contract=off`) its bits equal
  the generic build's; confirm once with G-BIT on the workstation at `znver3`.
- `OMP_WAIT_POLICY=active` and spread placement (24 threads over 12 CCDs) are measured in S-1,
  not assumed (Q-H5).

**H-1. `'direct'` bottom on host backends, single rank (recorded numerics change).**
- **Decision.** `'auto'` selects `'direct'` on host where §13.4.6 holds; the `kHostMemory`
  condition of `directBottomIneligible()` is removed. The engine is §13's, unchanged: the FP32
  block-tridiagonal factor (the case: walls in y, so s = y, P = 12, b = 128) preconditions FP64 FCG
  to τ = 1e-5. `'algebraic'` stays selectable.
- **Host team size.** `T_host = min(kBottomHostTeam = 8, team_size_max)` for the factor and the
  solve launches (`directTeam`, `mac_cutcell_mg.hpp:2918`). 8 is one Zen CCX. The factor and M are
  bitwise independent of T by construction (§13.2, U3).
- **Host reductions single-lane.** On host, every team reduction of the FCG skeleton (the dots,
  the max-norms, the per-component sums of the mean projections) is computed by one lane,
  `Kokkos::single(Kokkos::PerTeam(t), …, result)` with the broadcast form, ascending ext index over
  the same cell set. Cost ≈ 8 × 1536 FMA ≈ 5 µs per FCG iteration. The host bottom's bits are then
  independent of T and of `OMP_NUM_THREADS`, which H-6 needs. The device keeps team reductions.
- **Factor launch.** Lazy, at the first bottom solve after `setOpenness`, as on device.
- **Cost [model].** Factor ≈ 17–21 M FMA in ordered chains over 8 threads ≈ 1.3–1.8 ms per step.
  A solve takes one FCG iteration (§13.8 B: mean 1.02), ≈ 0.13 ms. So ≈ 3 ms per step at 11.6
  V-cycles, against ≈ 17: **−13**.
- **Numerics.** GraphAMG's inner 1e-8 is replaced by FCG to τ = 1e-5 with an M accurate to 2e-7…5e-6
  (U2). On CUDA the same switch measured 2.67e-14 over 50 steps with identical iterations (653).
- **Rejected:** keeping GraphAMG for host bitwise (17 ms per step, the largest single host term);
  GraphAMG at τ = 1e-5 (at best halves its iterations: still ≈ 8–10 ms); reusing GraphAMG's setup
  across steps (still ≈ 1 ms of serial sparse work per solve); an explicit dense inverse (2n³ =
  7.2 GFLOP per step).
- **Lever if the host bottom exceeds 4 ms per step at 1×24 (not v1):** in M and in the factor's
  Q = WᵀW and trailing update, loop j outer and outputs i inner (`omp simd` over i). Each output
  keeps its ascending-j sum, so the bits are unchanged, and the loops vectorize 4–8 wide.

**H-2. Prolong with integer coarse indices (bitwise).**
`prolongAddCell` (`mac_cutcell_mg.hpp:406`) derives the coarse index and weight from
`floor(0.5·i − 0.25 + gc)`: three floors and six FP↔int conversions per fine cell, scalar. Per axis:

```
ratio 2:  x0 = (i >> 1) + gc - 1 + (i & 1);   wx = (i & 1) ? 0.25 : 0.75
ratio 1:  x0 = i + gc;                         wx = 0.0
```

These are exactly the values the floor path produces: 0.5·i − 0.25 + gc is exact in double, so its
floor is exact and the weight is exactly 0.25, 0.75 or 0. The interpolation expression stays
**textually unchanged**, including `(1 - wx)` written as a subtraction, so nvcc sees the same DAG
(A2's FMA lesson, R1). The body is shared by host and device; if CUDA G-BIT fails, the change goes
to the host branch only. [model] 0.64 → ≈ 0.22 ms per V-cycle: **−4.9**.

The brief's question about mean removal and dot at ≈ 170 µs per launch has two causes: host MDRange
(`mgmeanr` is a raw `MDRange3`; `dot` is `ccReduce3`'s MDRange) and a loop-carried add chain of
≈ 4 cycles per cell. WO-10 removes the first. H-5 removes passes. The chain stays unless WO-10
adopts lanes (Q-H4).

**H-3. Stragglers and fill/BC launches under rule H (bitwise; one commit per family).**
- **(a) `Solver::copyInner`** (`flow_ibm_core.hpp:983`; also `flow_reference.hpp:131`). Today a
  flat index with three 64-bit div/mod per cell. Change: `ccFor3` over the inner box. 7 launches,
  3.07 ms [meas-S] → ≈ 0.2.
- **(b) `vof::block::gather_local_sum`** (`block_exchange.hpp:461`). Today, per domain cell,
  16 jobs × 3 axes of `((v % L) + L) % L`. Change: `ccFor3`, plus per-axis job masks.
  - For each axis d, a host-built array over the patch's coordinates holds 16-bit masks
    `m_d[x_d]` = {k : job k's box contains `x_d + o_d` under today's periodic predicate}.
  - Per cell, `m = mx[x] & my[y] & mz[z]`; the jobs in m are visited in ascending k, with `l[d]`
    computed as today.
  - Same jobs, same order, same additions, so bitwise. 2.83 → ≈ 0.2.
- **(c) `rhs_var`** (`flow_ibm_project.hpp:988`) and **`ibm_build_diff_var`**
  (`cut_cell_ibm.hpp:366`). `MDRange3` → `ccFor3` with the body verbatim; each kernel keeps its
  execution space (add an `Exec` parameter to the helper if they differ). 2.52 + 3.78 → ≈ 1.5 + 2.2.
- **(d) Momentum periodic fill** (`Solver::fillAxis`, `ibm_pfill`, one MDRange2 per axis). Where a
  call site fills x, y and z back to back:
  - one launch over the rows of the three disjoint ghost slabs, exactly as `CutcellMG::fillWrap`
    (§5.4), with the serial cutoff;
  - single-axis callers keep `fillAxis`;
  - bitwise by §5.4's argument (each ghost cell is written once, from the inner cell the sequential
    fill propagates);
  - −1 to −2. This is what makes H-0's np = 1 switch pay in the momentum stage.
- **(e) `bc_vel`** (`mac_bc.hpp:54`, one MDRange2 per face and component). One launch per (axis,
  BC application) covers both faces and every component the call site applies back to back, when
  the two ghost planes are disjoint (`ext_a > 2g`). Different axes stay separate launches in their
  original order, so edges keep their write order. 138.5 launches → ≈ 25: ≈ −2.

**H-4. The container on host (bitwise).** Host branches only; the device launches are untouched.
The statistics kernels (`area`, moments) belong to WO-9 and are not touched here.
- **(a) List kernels.** These are the bodies with the `q ≥ end(base + k)` guard:
  `vofCurvListPass` passes 0–3, the PLIC over the worklist, the debris act kernels, and every other
  batch kernel of that shape.
  - Launch `RangePolicy<SExec, Schedule<Dynamic>>(0, Σ_k cnt_k).set_chunk_size(16)`.
  - `cnt_k = end − start` is read directly: the View is HostSpace, so no transfer exists. A
    host-built exact offset table maps t → (k, q = start(base + k) + t − off_k). The body is
    verbatim.
  - Each q is visited once, and every body writes only its own cell or adds integers atomically,
    so the result is bitwise.
- **(b) Region kernels.** These map a flat index with `%`/`/` per cell: ghost zero, periodic,
  clamp, the sweep update, hf reset, CSF force, freeze, flux.
  - Iterate rows (k, y, z) with the x loop inside, `omp simd` where `ccFor3`'s contract holds, and
    find the job once per row.
  - Ghost-only kernels (ghost zero, periodic) iterate only their ghost shell's rows, using
    `fillWrap`'s three-slab split per job.
  - Clamp skips any row that lies inside the domain on every non-periodic axis (no cell in it
    writes).
- **(c) `batch_bbox`.** Today `TeamPolicy(nj, AUTO)`, which on host is one thread per block.
  Change: one row-parallel pass over all jobs' inner rows. Each row computes its six extrema over
  x, then integer `atomic_min` / `atomic_max` into the job's slots. Order-free, so bitwise.
- **(d) Scans** (`batch_worklist`, `batch_curv_compact`, the debris mark scans).
  - A `parallel_scan` over rows; a row's value is its hit count in x order.
  - The final pass writes the row's hits at [prefix, prefix + count) in x order, which gives the
    identical list.
- **(e) `movePiece`** (`move_local`: one MDRange3 per piece per component, 75 launches). One launch
  per call over all (piece, component) pairs, through a by-value job table (≤ 16 per chunk), rows
  inside. Pieces write disjoint block regions; add a debug assert for it.
- [model] curvature 16.9 → ≈ 7.5 (with H-3b), block advect 15.3 → ≈ 6, debris 1.9 → ≈ 1.

**H-5. Krylov reduction fusion on host (after WO-10; bitwise against the post-WO-10 state).**
- In the single-rank resident PCG, host branch:
  - (i) the matvec kernel also accumulates p·Ap;
  - (ii) the update kernel (x += αp, r −= αAp) also accumulates {Σr, count} over fluid cells;
  - (iii) the mean subtract of r also takes max|r|.
- Each fused loop visits the cells in WO-10's order: `ccFor3`'s row partition, x ascending,
  per-thread partials combined in thread order. If WO-10's order cannot be expressed inside a
  `ccFor3`-shaped row loop, STOP and report.
- The distributed path fuses the same way, before the Allreduce. Net: −3 L0 passes and −3 launches
  per iteration.
- **Q-H4 (WO-10 not yet committed):** adopt 4 fixed x-lanes per row.
  - lane = (x − lo.x) & 3, each lane ascending; the row value is (l0 + l1) + (l2 + l3).
  - ISA-independent: there is no reassociation, and vectorizing changes no lane's order.
  - It breaks the add chain that bounds a pencil reduction at ≈ 40 µs per L0 pass at 24 threads:
    ≈ −2 more at 1×24.

**H-6. Distributed host `'direct'` (§13 Q-D7, host half; recorded).**
- **Eligibility.** On host-memory builds the `!distributed_` condition is dropped. The device keeps
  it, because CUDA-aware MPI is unresolved.
- **Once per operator change:**
  - every rank allgathers the bottom level's openness ox, oy, oz (inner cells, by global id),
    reusing GraphAMG's `amgGlobalOfLocal_` map and gatherv helper;
  - the data goes into a private, single-rank-shaped global bottom level `gb_`;
  - `gb_` runs the single-rank `fillOpenness`, `applyBoundaryOpenness`, `buildCutcellOpFace` (same
    gf), the label kernel and the factor.
- **Per V-cycle:** allgatherv the rhs (as GraphAMG does today), run the FCG on `gb_`, and each rank
  copies its own block of x out.
- **Why it is bitwise.** With H-1's single-lane reductions, the result equals the single-rank
  `'direct'` result bitwise at any thread count, provided the gathered openness equals single-rank's
  bottom openness bitwise. The np-gates assert that property of the hierarchy, and GraphAMG's
  gather already relies on it.
- Every rank factors redundantly, which is GraphAMG's pattern today. [model] 8×3 −13; 192 cores
  −12 to −15.
- **Rejected:** a rank-0 solve plus broadcast. It adds a latency step per V-cycle, and the other
  ranks idle anyway.

**H-7. Distributed box passes (bitwise).** In the host branch of `cutcellSmoothColorBoxFace`,
`residualCutcellBoxFace` and the apply box:
- the shell launch iterates only the shell (the z-slabs, then the y-slabs inside them, then the two
  x-runs of the remaining rows), instead of every row of the box with a per-cell skip test;
- the interior launch (empty skip box) uses a variant with `omp simd` and no skip test;
- same cells, same colour rule. [model] 8×3: −3 to −5.

**Conditional (not main line; Q-H7):**
- **F4: a persistent host team for the coarse tail** (L1, L2 and the bottom in one
  `TeamPolicy(1, pool)` launch with team barriers).
  - After the main line, ≈ 1000 launches/step × 6–8 µs ≈ 6–8 ms of fork/join remain; F4 recovers
    about half, at L cost.
  - It is B1's retired in-kernel V-cycle rebuilt for host, and Kokkos has no barrier inside
    `parallel_for`.
- **F1: two-colour temporal blocking.** It needs its own design: at 24 threads the 64 z-planes give
  each thread 2–3 planes, so z-slab lagging is mostly boundary planes.
- **Prolong reading coarse ghosts through a composed wrap/clamp map.** It removes the fill + Neumann
  launches before each prolong: ≈ −0.6.

### 14.4 Expected end state [model]

**1×24, rtol 1e-8**, ms/step:

| stage | now [meas-S] (rtol 1e-10, np 1 distributed) | after H-0 | main line (H-0…H-5 + WO-10) |
|---|---|---|---|
| projection | 85.2 | 60.5 | 37 (32–42) |
| momentum | 15.0 | 15.0 | 8.5 (7.5–10.5) |
| curvature | 16.9 | 16.9 | 7.5 (6–9) |
| block advect | 15.3 | 15.3 | 6 (5–8) |
| predictor + debris + csf | 6.2 | 6.2 | 5.5 |
| outside the stage timers | 5.0 | 5.0 | 4.5 |
| **step** | **143.6** | **≈ 119** | **≈ 69 (60–78)** |
| TBFsolver, same 24 cores [brief] | 46 | | |

**Measured, Snellius S-1 (2026-10-08) [meas-S]** — flow 38e80e6 = H-0…H-4 without H-5, znver4,
1×24 contiguous cores 0–23, rtol 1e-8, `'direct'` bottom; stages from the 30-step kernel-profile
run (step 93.7), step from the 300-step runs (log, "Snellius S-1"):

| stage | model "main line" | S-1 measured |
|---|---|---|
| projection | 37 (32–42) | 55.3 |
| momentum | 8.5 (7.5–10.5) | 10.6 |
| curvature | 7.5 (6–9) | 9.7 |
| block advect | 6 (5–8) | 7.6 |
| predictor + debris + csf | 5.5 | 4.5 |
| outside the stage timers | 4.5 | 6.1 |
| **step** (best median, 300 steps) | **≈ 69 (60–78)** | **93.1** (`'algebraic'` 95.8) |
| TBFsolver, 8×3 same node | | 45.5–46.0 |

The projection carries the miss: `mg_bottom_factor` (the FP32 factor, rebuilt every step) alone is
16.9 ms/step against the model's ≈ 3 ms for the whole bottom, and H-5 (−2) is not in. Other
layouts: spread over 12 CCDs 83.8 (`'algebraic'` 89.4); 8×3 one rank per CCD 90.7, 6×4 106.5, 3×8
107.9 (all `'algebraic'`: the distributed `'direct'` is H-6); generic build 98.0 (znver4 −5 %).
F4 not triggered (937 launches/step < 1100).

Projection in the main line, per V-cycle or iteration [model]:
- smoother 1.1 ms per V-cycle;
- residual 0.3;
- prolong 0.22;
- restrict + fills 0.19;
- the bottom ≈ 3 ms per step;
- the Krylov part 0.87 ms per iteration;
- setup ≈ 6 ms per step.

The smoother (≈ 13 ms per step) is the largest remaining term, and it is near its L3 bandwidth.
Only F1, E1 or fewer iterations (S3, U-1) move it.

**Beyond the main line**, each approximate at 1×24:
- U-1, case rtol 1e-6 (D1 passed every physics gate; 7.4 iterations): −9;
- U-2, E1 float V-cycle: −7 to −10;
- F4: −3 to −4;
- lanes (Q-H4): −2;
- spread placement: unknown (fact, S-1).

**8×3** (rank 0): 166.3 [meas-S] → aligned ≈ 143 → rtol 1e-8 ≈ 128 → main line with H-6 and H-7
≈ **83** (72–95). A 6×4 run is expected at or below that.

**192 cores** (64×3): 82–85 [brief] → ≈ **55** (45–65). The contributions are H-6 (−13), rtol (−6)
and the per-thread items scaled by ⅛ (−4). Confidence is low: the profile at this size is missing,
and exchange latency (≈ 31 exchanges per iteration on the box path) is the probable next term.
TBFsolver: 25.

### 14.5 Work orders

Order: WO-H0, then WO-H1…H3 in sequence; WO-H4 may run in parallel in its own worktree (it touches
`src/vof/*` only); then S-1; WO-H5 waits for WO-10; then WO-H6, WO-H7, S-2, WO-H8. Every commit stages
named paths only.

- **WO-H0. Protocol and baselines** (§14.3 H-0).
  - `run_mpi.py`: the size-1 skip and `--dump`.
  - The `bench_peclet.py` diff (size-1 skip, rtol 1e-8) and the slurm layouts are handed to the
    caller.
  - Workstation baselines on current main:
    - host 50-step dumps at 1×8 and 1×24, rtol 1e-10 and 1e-8;
    - `PECLET_FLOW_MG_DEBUG=3` level times, 2 runs;
    - kernel-timer profiles of the container kernels at `OMP_NUM_THREADS=1` and `8`, interleaved,
      for H-4's scaling gate.
  - *Accept:*
    - the kernel listing of `run_mpi.py` at np 1, 1×8, shows 0 `selfCopy` and 0 `cc_smooth_box`
      launches;
    - the np = 1 dump against the `prof.py` dump is bitwise, or the difference is logged (it must
      be ≤ N50);
    - all numbers are in the log.
- **WO-H1. H-1.**
  - *Accept:* G-NUM-H (§14.6).
  - U3 is extended on OpenMP to T ∈ {1, 2, 4, 8}: the factor + M, **and the whole FCG solve**, are
    bitwise across T.
  - G-PERF: L3 self time ≤ 0.35 ms per V-cycle at 1×8 by the [meas-W] method (1.50 today).
- **WO-H2. H-2.**
  - *Accept:* G-BIT on host and CUDA.
  - The host `prolong` kernel time is ≤ 0.45× the pre-change module's, in an interleaved A/B on
    the workstation (1×8, kernel timer, 3 rounds, minimum per kernel).
- **WO-H3. H-3 (a)…(e)**, one commit each.
  - *Accept:* G-BIT for each commit.
  - Interleaved A/B ratios: `copyInner` ≤ 0.15×, `gather_local_sum` ≤ 0.2×, `rhs_var` and
    `ibm_build_diff_var` ≤ 0.75×.
  - Launches per step on the case: `ibm_pfill` ÷ 3, `bc_vel` ≤ 30.
- **WO-H4. H-4 (a)…(e)**, commits a–e.
  - *Accept:* G-BIT on the state (u v w p C and the block colours). The statistics are bitwise as
    well, since H-4 does not touch their kernels.
  - The VoF block MPI ctests pass at np 1, 2, 4.
  - Scaling gate: time(OMP 1) / time(OMP 8) ≥ 5.5 for `batch_curv_list` and ≥ 5 for the region
    batch kernels (interleaved; WO-H0 recorded the ratio before the change, predicted ≤ 3 for the
    list passes).
  - `move_local` ≤ 10 launches per step.
- **S-1. Snellius run 1**, after WO-H0…H4 (and WO-9/10/12 if they have landed); znver4 build;
  rtol 1e-8.
  - Runs: 1×24; 1×24 with `OMP_WAIT_POLICY=active`; 1×24 spread over 12 CCDs
    (`--cpus-per-task=96`, `OMP_NUM_THREADS=24`, `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`); 6×4
    CCD-aligned; 8×3 on the old map (for continuity).
  - kprof for 1×24 and 6×4.
  - *Accept:* the numbers are logged, §14.4 is re-stated with them, and Q-H3, Q-H5 and Q-H7 are
    answered by their triggers.
- **WO-H5. H-5**, after WO-10.
  - *Accept:* G-BIT against the post-WO-10 state; −3 launches per PCG iteration.
- **WO-H6. H-6.**
  - *Accept:* G-NUM-H at np 2 and 4 against `'algebraic'`.
  - `'direct'` at np 1, 2, 4 is bitwise to single-rank `'direct'` at equal `OMP_NUM_THREADS`
    (50-step column through `run_mpi.py --dump`).
  - `vardensity_mpi` and `telescope_mpi` are green; distributed `'algebraic'` is bitwise to before.
- **WO-H7. H-7.**
  - *Accept:* G-BIT, including the np-tests.
- **S-2. Snellius run 2.**
  - Runs: 1×24; 6×4; 8×3 on 8 CCDs; 192 cores as 24×8 and 64×3; kprof of 64×3 and 24×8.
  - *Accept:* G-PERF-H.
- **WO-H8.** Log, register text (§14.8), state file and handoff.

### 14.6 Verification gates

**G-NUM-H** (WO-H1, WO-H6). Host, `'direct'` against `'algebraic'` on the same build:

| # | check | threshold |
|---|---|---|
| 1 | 50 steps from `ckpt_t43`, rtol 1e-10, 1×8: max rel. diff over u v w p C | ≤ 1.499e-11 (host N50, WO-0); expected ≤ 1e-13 |
| 2 | outer iterations | identical on every step (1×8 baseline: 13 × 49, 14 × 1) |
| 3 | `max_open_divergence_projected()` ratio | ≤ 2 per step (expected ≤ 1.001) |
| 4 | 300 steps, inner FCG | max ≤ 3, mean ≤ 2.5; shift restarts 0; flag never set |
| 5 | rtol 1e-8 (the case), 50 steps | iterations identical on every step |
| 6 | `p3/solids.py` battery on host | iterations identical per step; u v w within rtol × 10 |
| 7 | §8 G-NUM 4 physics (static drop, Hysing 1) on host | as §8 |
| 8 | `ctest -LE bench` | green. Host `state_hash` re-baselined in the commit, with an old → new table, for the cases that newly select `'direct'`; the CUDA `state_hash` is unchanged |
| 9 | two runs | bitwise |

**G-BIT:** §8, unchanged.

**Workstation A/B method** (robust to the shared host's load):
- two modules in the same session, run A B A B A B;
- the kernel timer's 20-step difference, taking each kernel's minimum;
- the gates are ratios, never absolute times.

**G-PERF-H** (S-2: Snellius, rtol 1e-8, znver4, 1×24 single-rank), each against its expected value:

| quantity | gate | expected |
|---|---|---|
| step | ≤ 78 | 69 |
| projection | ≤ 42 | 37 |
| curvature | ≤ 9 | 7.5 |
| block advect | ≤ 8 | 6 |
| momentum | ≤ 10.5 | 8.5 |
| launches per step | ≤ 1300 | |
| step − kernel time | ≤ 8 ms | |
| 6×4 or 8×3 on 8 CCDs | ≤ 95 | |

A miss is reported, not reverted (§7).

### 14.7 Open questions and risks (each has a default)

- **Q-H1 [pref] Reverse §13 D-5 / Q-D6 (`'direct'` on host).** *Default:* yes (H-1); host G-NUM-H
  replaces host bitwise. Revert = restore the `kHostMemory` line in `directBottomIneligible()`.
- **Q-H2 [pref] Should `init_mpi` at size 1 select the single-rank engines in code?** *Default:* no.
  The register's np = 1 gate needs the distributed path at np = 1, so this stays a protocol choice
  (H-0).
- **Q-H3 [fact] T_host.** *Default:* 8. Experiment: L3 self time at T ∈ {1, 2, 4, 8, 16, 24} on the
  workstation (`PECLET_FLOW_MG_DEBUG=3`); H-1's single-lane reductions keep the bits. Take the
  fastest, as a constant, not a setter.
- **Q-H4 [fact] Lanes in WO-10's order.** *Default:* adopt them if WO-10 has not committed its
  order. Otherwise there is no second re-baseline, unless S-1 shows reductions > 2.5 ms per step.
- **Q-H5 [fact; a TBFsolver rerun is the user's (billed)] Placement for the comparison.** *Default:*
  publish contiguous cores (as so far). S-1 measures peclet at spread placement; TBFsolver is not
  rerun.
- **Q-H6 [pref] Close ≤ 50 ms with U-1 (case rtol 1e-6) and/or U-2 (E1).** *Default:* neither.
  Report the main-line number.
- **Q-H7 [fact] F4 / F1.** *Default:* design F4 only if S-1 shows ≥ 1100 launches per step at 1×24
  and a step above 78. F1 needs its own architect pass.
- **Q-H8 [fact] The next step at 192 cores.** *Default:* the S-2 kprof of 64×3 and 24×8. The
  expected next design is F3 (communication-avoiding smoothing with wall BCs) plus exchange
  aggregation.
- **Q-H9 [pref] The published 24-core MPI layout.** *Default:* 6×4.
- **Q-H10 [fact] The row-pair prolong** (x-interpolated coarse lines shared by four fine rows;
  bitwise: the same operands and expressions per output). *Default:* only if prolong exceeds
  1.5 ms per step at 1×24 after H-2.
- **R-H1.** H-4's premise comes from the code. If WO-H0's 1→8 scaling ratio is already ≥ 5, H-4
  shrinks to its index-math part (−4 to −6).
- **R-H2.** Between H-0 and H-3(d), the single-rank momentum fills cost +1 to +2.
- **R-H3.** At spread placement, T_host = 8 spans CCDs and the barrier cost about doubles (Q-H3).
- **R-H4.** H-2 on CUDA could contract differently. Keep the expression text; the fallback is the
  host branch only.
- **R-H5.** The 1×24 end state misses ≤ 50 ms (centre 69).
- **R-H6.** Other host cases (IBM, porous) take `'direct'` only under §13.4.6 (≤ 64 components,
  b ≤ 192, n ≤ 8192). Otherwise GraphAMG runs, unchanged.

### 14.8 Register entries this section creates (for the caller to add)

1. **Host backends use the `'direct'` bottom where eligible.** This reverses §13 D-5's host clause
   and Q-D6's default.
   - Evidence: GraphAMG costs 1.50 ms per V-cycle [meas-W], which is ≈ the 22 ms/step outside
     kernels [meas-S].
   - Rejected: GraphAMG kept for host bitwise; GraphAMG at τ = 1e-5; GraphAMG setup reuse.
2. **On host, the bottom FCG's reductions are single-lane in index order**, so the bits are
   independent of T and of the thread count; T_host = 8 affects speed only.
   - Rejected: team reductions on host.
3. **The distributed host `'direct'` runs redundantly on every rank**, from the allgathered bottom
   openness.
   - Rejected: a rank-0 solve plus broadcast; GraphAMG.
4. **One-rank benchmarks run without `init_mpi`.** `init_mpi` at size 1 stays distributed, because
   the np = 1 gate needs it.
   - Rejected: a silent single-rank switch inside `init_mpi`.
5. **Host list-driven batch kernels launch over exact counts with a dynamic schedule; host region
   kernels iterate rows.**
   - Rejected: upper-bound launches under GCC's contiguous static schedule; per-cell div/mod index
     maps.
6. **Benchmark layouts on Zen never straddle an L3 (CCD).**
   - Rejected: 8×3 on contiguous cores 0–23.
