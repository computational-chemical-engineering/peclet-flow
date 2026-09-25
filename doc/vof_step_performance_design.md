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
