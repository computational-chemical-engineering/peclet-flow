# Distributed FP32 V-cycle (D), distributed projected guess (E), FP64 restart: design note

> **Landed (2026-10-17), defaults unchanged:** WO-R 99dd50b, WO-A 85b66fb, WO-D1 acf495a, WO-D2a 87cdd62,
> WO-D2b0 5baf24b, WO-D2b1 afa062b, WO-E a7d3e9e, WO-G 471b03b, review fixes e4f75df (review:
> `bubble_column_perf/gate_dist/REVIEW.md`, WO-F GO, no blockers). Gate numbers: `doc/vof_step_performance_log.md`.
> **WO-F landed (2026-10-19, USER "Yes, make the GPU V-cycle default 'auto'"): the V-cycle default is
> `'auto'`** at every np, after S-4 (Snellius H100, one rank per GPU: projection auto/fp64 0.970 np 1,
> 0.974/0.711/0.645 strong np 2/4/8, 0.979/0.981/0.829 weak np 2/4/8; 0 restarts) and F-1…F-7 on the
> flip commit (numbers: `doc/vof_step_performance_log.md`). CUDA results recorded, host bitwise; E stays
> `'zero'`.
> **Open issue (pre-existing, not this package):** the staggered cut-cell cylinder Stokes case is not
> bitwise serial ↔ np = 1 even under `'zero'` (step 1: du 1.0e-17, dp 4.6e-11, at 0cc1a3a).

> **Outcome (design, 2026-10-16).** Three pieces, then one isolated default flip.
> **R — restart.** Under `'auto'`, a PCG/FCG solve whose FP32-preconditioned attempt does not meet
> its stop test (cap, non-finite breakdown, degenerate direction) is **re-solved cold with the FP64
> V-cycle** — bitwise the `'fp64'` solve of the same right-hand side — and `'auto'` stays on FP64 for
> the rest of the run (re-armed only by `set_pressure_vcycle_precision`). Forced `'fp32'` unchanged.
> One warning line; no new public name. **D-dist.** The FP32 V-cycle runs on the distributed
> host-scalar PCG/FCG as an op-for-op transcription of today's distributed FP64 V-cycle: one
> `GridHalo<float>` per FP32 level on the existing topology, the CA ring in FP32, halo kernels that
> share the single-rank FP32 cell bodies, FP64 from the first telescope point down, the bottom FP64
> and unchanged, no extra collectives (σ from the already-global max|r|; the exit mean costs the
> Allreduce today's `removeMean` costs). **np = 1 is bitwise to single-rank in every mode.**
> **E-dist.** The Gram packet is allreduced (one SUM of five, one MAX), the guard unchanged, history
> reset on `init_mpi` and every `redistribute`. Opt-in parity only. **Flip.** `'auto'` becomes
> FP32-where-eligible **on device backends only** (host `'auto'` = FP64), then one commit makes it the
> default once gates F-1…F-7 (§9) pass on the commit itself (done 2026-10-19, WO-F).

**Status:** DESIGN, 2026-10-16. Brief: `~/Codes/bubble_column_perf/gate_dist/BRIEF_ARCH.md`, recon
digest `…/gate_dist/RECON.md`. Code read at flow `ab8a4ee` (branch `projdist`). Parent design:
`doc/vof_projection_cost_design.md` (cited "P§N"); this note implements its §15 Q-P3 and the Outcome
block's flip conditions, and **supersedes P§4.4.1 items 1–2** (single rank) and **P§5.3's "single rank
in v1"**. Line numbers below are at `ab8a4ee` (`src/mac_cutcell_mg.hpp` unless named).

---

## 0. Decisions at a glance

| item | decision | rejected | class |
|---|---|---|---|
| R trigger | FP32 attempt under `'auto'` that did not meet its stop test (cap, non-finite, `pAp ≤ 1e-300` exit) | stagnation heuristic; `it ≥ maxit` inference | bitwise except where it fires |
| R restart point | **x = 0, cold**, fresh `maxit`, the caller's stop reference: bitwise the `'fp64'` cold solve | current iterate (keeps progress, unverifiable quality, never bitwise) | |
| R persistence | latch FP64 for the run; the setter re-arms | per solve (pays `maxit` every step in a bad regime) | |
| R scope | `'auto'` only; PCG + FCG (both call sites: projection and balanced force) | forced `'fp32'` (it is the A/B tool) | |
| R reporting | one line (rank 0), `pressure_vcycle_precision()` reads `'fp64'`, C++ counters | a new Python getter (Q-4) | |
| D driver | the existing host-scalar `solvePCG` loop + `solveFCG` | a distributed device-resident PCG | |
| D halos | separate `GridHalo<float>` per FP32 level on the shared `GridHaloTopology` | reusing the double buffers (type-unsafe, core change) | |
| D telescope | FP32 on levels 0…Lf, Lf = first telescope level; FP64 below through the unchanged stage code | FP32 through the stages (core gather/scatter in float) | |
| D bottom | FP64, unchanged engine (RB-GS / GraphAMG), fed by today's FP32→FP64 interface | FP32 GraphAMG input | |
| D collectives | none added per V-cycle; +1 Allreduce per operator build (range flags) | a separate σ Allreduce | |
| E-dist | Gram packet allreduced; history reset on `init_mpi`/`redistribute` | carrying φⁿ⁻¹ through `redistribute` | |
| Flip | `'auto'` device-only (WO-A), then default `'auto'` (WO-F) after F-1…F-7 | host `'auto'` = FP32 (gain 0.92–0.95 < the 0.88 rule) | recorded (CUDA), bitwise (host) |

---

## 1. Problem and scope

**Build:** (R) an FP64 restart for FP32-preconditioned PCG/FCG solves under `'auto'`; (D) the FP32
V-cycle preconditioner on the distributed (`init_mpi`, any np including 1) PCG/FCG; (E) the projected
initial guess on the distributed PCG/FCG; (F) the criterion and the isolated commit that make
`'auto'` the default of `set_pressure_vcycle_precision` on device backends.

**Out of scope:** E's default (stays `'zero'`); Snellius or any billed run; `amr/`; D under BiCGStab,
the standalone V-cycle, Chebyshev; the velocity MG; curvature; performance beyond "D and E are no
slower than FP64 at np > 1"; a distributed device-resident Krylov driver; D5 (FP64-data deferral) on
the distributed path; FP32 below a telescope point; editing the decision register (proposed entries
in §12).

---

## 2. Constraints and invariants

**Mathematics.** Unchanged from P§2: the outer PCG/FCG, matvec, dots, mean projection and stop test
are exact FP64; the V-cycle is a preconditioner and may be FP32 (register: defect-correction rule).
FP32 bodies are P§4.4.3 verbatim (flux form, diagonal derived never stored, no double variable or
unsuffixed literal inside an FP32 body). σ = 2^-ilogb(max|r|). The bottom is FP64.

**Today's contracts, stated precisely (the targets of this note).**
- *Serial ↔ np = 1 (FP64):* **bitwise.** Eight assembled-solver ctests compare an `init_mpi` np = 1
  run with the single-rank run by memcmp (tol 0) on host and CUDA — `sdflow_colocated`, `varmu`,
  `cellforce`, `cutcellmg_aniso`, `vof_wetting`, `vof_wetting_dynamic`, `movingscene_advect`,
  `wall_slip` (`_mpi_np1`) — although single rank runs `solvePCGResident` (device scalars, fused host
  reductions) and np = 1 runs the host-scalar loop of `solvePCG` (:1965) with halo exchanges. It holds
  because every V-cycle op is pointwise per colour, the halo delivers the values the in-kernel wrap
  reads, the reductions use the same bodies (host lane order, C0) and a one-rank Allreduce is the
  identity. `cutcellmg_mpi_np1` asserts 1e-13 (its comment: "bit-exact").
- *np > 1 ↔ np = 1 (FP64):* **solve-tolerance class.** `cutcellmg_mpi` asserts
  max|x_np − x_single| ≤ 1e-8 at rtol 1e-10 (`test_cutcellmg_mpi.cpp:103`); the V-cycle alone is
  "bit-tight to 1e-11" (`test_distributed_mg`); iteration counts may differ (Allreduce order,
  hierarchy depth, telescoping, bottom).
- **This note keeps both, for every mode** (`'fp64'`/`'fp32'`/`'auto'` × `'zero'`/`'projected'`/
  `'auto'`, and a restarted solve): np = 1 memcmp-equal to single rank with identical per-solve
  iterations, host and CUDA; np > 1 in today's tolerance class.

**Defaults bitwise.** Until WO-F, `'fp64'`/`'zero'` are byte-identical to `ab8a4ee` on host and CUDA,
single rank and every np (G-BIT). The single-rank FP32 and E code paths are also unchanged in results
(their references `ref/d3f32_omp`, `gd4_cuda` stay valid): every new branch is behind `distributed_`
or behind a restart that fires only on a non-converged FP32 attempt.

**Mirror rule (normative for D).** The distributed FP32 V-cycle is the distributed branch of
`vcycleImpl`/`smooth` transcribed op for op: each FP64 kernel replaced by an FP32 kernel built on the
single-rank FP32 cell bodies, each `lv.dev` exchange by `lv.devf` on the FP32 field, every zeroing,
fill, Neumann and overlap split kept at its position. Exactly three places differ, all already
present in the single-rank FP32 path: the L0 entry, the FP32→FP64 interface (bottom or telescope),
the L0 exit. Bitwise np = 1 ↔ single rank then follows from the FP64 argument above.

**Global decisions.** Every branch that can differ between ranks must be taken from global data:
eligibility (hierarchy, flags allreduced), σ (global max), the restart (derived from allreduced
scalars only). No rank-local value may select a path.

**Rules.** Kokkos device code in `.hpp`; host `-ffp-contract=off`; rule H (host pencils, no host
`MDRange3` in hot paths — mirror the FP64 box kernels' launch forms); x-fastest; no env var changes a
result; modes are strings on the existing setters; any new `(VReal)`/`(float)` cast carries
`// PRECISION-EXEMPT: FP32 V-cycle preconditioner (doc/vof_projection_distributed_design.md)` (ctest
`no_float_operator_casts`); MPI ctests with `--bind-to none`, `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`.

---

## 3. Decisions, with the rejected alternatives

### 3.1 Restart (open item 1)

- **Trigger:** `vprec_ == kVcycleAuto && lastVcycleFp32_ && !lastSolveConverged_ && maxit > 0`, where
  `lastSolveConverged_` (new, per attempt) is true iff the attempt met `rn < rtol·rref` or started
  with r0 == 0. This covers the cap, both non-finite breakdowns (the early `rz` return included) and
  the finite `pAp ≤ 1e-300` exit — which with a sound FP64 iterate is preceded by the stop test, so
  under FP32 it signals a degenerate z. *Rejected:* a stagnation trigger — it needs a threshold that
  is a fact nobody has measured, and false positives would latch FP64 for the run; inferring the cap
  from `it ≥ maxit` — a solve converging exactly at `maxit` would latch needlessly.
- **Restart point: x = 0, cold.** *Why not the current iterate:* PCG's step is the exact A-norm line
  search for any z only while `r_k·p_{k−1} = 0` holds numerically; a capped or broken FP32 attempt is
  exactly where that is in doubt, and its iterate's quality cannot be certified without the very
  solve we are about to run. Cold costs at most one extra FP64 solve **once per run** (the latch) and
  buys a falsifiable property: **the restarted solve is bitwise the `'fp64'` cold solve** of the same
  `b` (r = b − A·0 = b bitwise, H-2c; same stop reference; the FP64 V-cycle is stateless across
  solves; D5's run-time FP64 rebuild is gated bitwise).
- **Budget:** a fresh `maxit` (a cap leaves no remainder). Total ≤ 2·maxit, once per run.
- **Persistence:** latch `fp32Latched_` until the next `set_pressure_vcycle_precision` call (any
  mode); it survives `init`, `init_mpi`, `redistribute`, geometry rebuilds (it is mode state, like
  `vprec_`). *Rejected:* per-solve — in a regime where FP32 fails (walled slab ≥ 1e3) every step would
  pay `maxit` iterations before its FP64 solve.
- **Scope:** `'auto'` only; forced `'fp32'` keeps today's behaviour (it is the A/B and gate tool; a
  restart would hide what the gates measure). PCG (resident and host-scalar) and FCG; implemented in
  CutcellMG's public `solvePCG`/`solveFCG`, so the projection solve **and** the balanced-force solve
  (`flow_ibm_project.hpp:~640-760`) are covered. Chebyshev, BiCGStab, standalone V-cycle: D is
  ineligible there; nothing to restart.
- **E:** the restart clears the projected guess for its attempt (φⁿ⁻¹ lived in the output buffer and
  is overwritten), so the restarted solve is cold — bitwise `'fp64'`+`'zero'`. If it converges the
  history is kept and advances (φⁿ was valid); if it caps, the caller's existing rule
  (`flow_ibm_project.hpp:1429`) resets it.
- **Warm-started callers** (balanced force, legacy `pwarm_`): the restart is cold too, with the
  caller's `stopRef_` kept. Result in the tolerance class of the warm `'fp64'` solve, not bitwise
  (Q-7).
- **Flags:** `solveFailed_` is per attempt, so after a restart it reflects the FP64 attempt. A cap
  that persists under FP64 does **not** set `pressure_solve_failed()` — the restarted solve is
  indistinguishable from an `'fp64'` solve, whose cap today is reported through the iteration count;
  changing that flag's meaning is an API change outside this package (Q-6).
- **`set_pressure_strict`:** inside an FP32 attempt under `'auto'` a breakdown does not throw (the
  restart handles it); the FP64 restart attempt honours strict exactly as any FP64 solve.
- **Iterations reported** (`last_pressure_iterations`): the restart attempt's. The caller's cap
  inference `iters ≥ pcgMaxit_` stays correct with no caller change. The discarded FP32 iterations
  go to the warning line and a C++ counter.

### 3.2 Distributed D (open item 2)

- **Driver: the host-scalar loop of `solvePCG` and `solveFCG`.** A distributed device-resident
  driver saves only small host copies: each iteration still needs three Allreduces with data
  dependencies between them (pAp → update → rn; precond → rz), C1–C4's fusions are single-rank host
  only, and a fourth driver would have to be proved bitwise to two others at np = 1. *Rejected for
  this package.* Consequence for np = 1: the host-scalar loop must take r·z **from the FP32 exit's
  fused reduction** (the resident FP32 path does, :2604, `precondDot`), not from a separate `dot` —
  on CUDA a separate kernel can change the reduction tree. FCG single rank already uses `dot` after
  an exit with k = −1, so FCG needs no driver change.
- **Halos: one `GridHalo<float>` (`Level::devf`) per FP32 level, built from the level's existing
  `GridHaloTopology` (same g, same neighbours).** `GridHalo<T>` packs `MPI_BYTE × sizeof(T)` and is
  already instantiated for float (flow_ibm.hpp:4954): no core change. *Rejected:* reusing the double
  object's buffers (no `exchange(View<float>)` overload; would need a core release and
  reinterpretation). Cost: halo-surface float buffers per FP32 level — negligible.
- **CA ring (g = 2 coarse levels): FP32 transcription of the pair smoother (:3812ff).** The ring
  rows are bitwise to the owner's because the FP32 weights are built by the same kernel over
  `[0, ext)` from the same double operands as the double face form (P§4.4.2); gate G-K2 checks it,
  and if it fails the pre-authorized remedy is three `devf` exchanges of `WX, WY, WZ` per level per
  build.
- **In-kernel wrap vs halo at np = 1:** values identical by construction — the new box kernels call
  the same `fp32SmoothCell`/`fp32ResidualCell` (mg_fp32_vcycle.hpp:35, :46) with plain offsets where
  the wrap kernels pass `ccWrapNbrs` indices; a colour pass reads only the other colour, so any
  launch partition gives the same values. Gate G-K1 compares the two kernels directly.
- **Telescope: FP32 on levels 0…Lf, FP64 below.** Lf = the first level carrying `tele` (any
  `StageKind`), else `bottom − 1`. Below a telescope point blocks are by definition too small to
  coarsen in place: those levels cost little in any precision, while FP32 there needs float
  gather/scatter in core's proved-byte-identical stage machinery and FP32 data on idle-rank layouts.
  At np = 1 there is never a telescope (`numBlocks() > 1` is required, :1496), so single-rank
  equivalence is untouched. L0 alone carries ≈ 86 % of the V-cycle bytes (P§4.3), so the benefit is
  kept.
- **Bottom: FP64, unchanged** (smoothed RB-GS by default under MPI, agglomerated GraphAMG when
  selected), reached through the existing FP32→FP64 restriction / FP64→FP32 prolongation interface.
  GraphAMG's input precision is therefore FP64 — unchanged.
- **σ:** no new collective. The host already holds the **global** max|r|: r0 / rref before the loop
  and the stop test's rn inside it, both from `maxabs` (MPI MAX). Max is order-free, so e is the same
  integer at every np.
- **Exit mean:** global, through `allreduceSum2` on `l0.comm` — the one Allreduce today's
  `removeMean` (:5214) spends. The z pass then forms the local r·z, allreduced by the driver — the one
  Allreduce today's `dot` spends. Per V-cycle the collective count is unchanged.
- **Build-time collective:** the per-level face-range flags (P§4.4.1 (5)) are read back lazily at the
  first eligibility check after a build (:3503) and, distributed, **allreduced with MPI_MAX over
  `comm_`** at that read-back: +1 Allreduce per operator build.
- **Memory:** distributed keeps the FP64 level data (no D5 deferral): +24 B/cell on L0 (+≈ 3 B/cell
  coarse) on FP32-carrying runs, ≈ 1.3 % of the case's 2145 B/cell device peak. *Rejected for now:*
  extending D5 — it would need the FP64 ring rows rebuilt on demand collectively (Q-5).

### 3.3 Distributed E (open item 3)

Gram entries: local reductions exactly as today, then **one** `MPI_Allreduce(SUM)` over the five sums
{G11, g1, G22, G12, g2} and **one** `MPI_Allreduce(MAX)` over rref, both on `l0.comm`, before the
host 2×2 solve; every rank then computes the same c. Guard unchanged (`d > 1e-10·G22`, G-E2 holds at
rtol 1e-10). History reset: `init_mpi`, **every** `redistribute`/rebalance (size-changing or
size-preserving — a size-preserving one passes the existing `phiPrev1_` extent check unnoticed),
plus P§5.3's list. *Rejected:* carrying φⁿ⁻¹ through `redistribute` (a field for an opt-in mode,
two cold steps per rebalance cost ≈ 1.7 iterations). Restart: §3.1. Size: opt-in parity only.

### 3.4 The flip criterion (open item 4)

§9: F-1…F-7 on the flip commit itself; any failure reverts it.

---

## 4. R — normative scheme

**State (CutcellMG):** `bool fp32Latched_ = false; bool lastSolveConverged_ = false;
int vcycleRestarts_ = 0; int lastDiscardedIters_ = 0;` C++ accessors `vcycleRestarts()`,
`lastDiscardedIters()`, `lastSolveConverged()`. Test hook (C++ only, no Python, no env):
`debugSetFp32Fault(int kind, int at)`; kind 1 = on the `at`-th FP32 V-cycle of an attempt the exit
scale is NaN (z and r·z non-finite on every rank); kind 2 = an FP32 attempt runs with
`maxit = min(maxit, at)`; kind 0 = off. Inert unless the V-cycle is FP32.

**Effective mode:** `effectiveVprec() = (vprec_ == kVcycleAuto && (fp32Latched_ || hostOnly)) ?
kVcycleFp64 : vprec_`, where `hostOnly` is introduced by WO-A (`kHostMemory`); WO-R introduces the
latch term only. `fp32Mode()` (:3430) returns `effectiveVprec() != kVcycleFp64`, so the FP32 build,
D5 deferral and eligibility all follow the latch. When the latch trips, run `fp32Reconfigure` as a
mode change would (FP64 data are allocated by `ensureFp64()` on the restart's first V-cycle; FP32
data go stale and are not rebuilt). `setVcyclePrecision` (:3728) clears the latch **before** its
existing logic.

**Per attempt:** each driver sets `lastSolveConverged_ = false` at entry and `true` exactly where the
stop test `rn < rtol·rref` fires, or when the loop is skipped because r0 == 0. The `if (strictPressure_)`
throws inside the drivers (:2026, :2029, :2642 and the FCG copies) become
`if (strictPressure_ && !restartArmed())`, `restartArmed() = vprec_ == kVcycleAuto && lastVcycleFp32_`.
The host-scalar `solvePCG` sets `lastVcycleFp32_` (WO-D2a; until then false).

**Wrapper** (the public `solvePCG`/`solveFCG` keep their signatures; today's bodies become
`solvePCGAttempt`/`solveFCGAttempt`, unchanged):
```
it = attempt(maxitAttempt)                 // maxitAttempt = maxit, or the kind-2 fault limit
if (!(vprec_ == kVcycleAuto && lastVcycleFp32_ && !lastSolveConverged_ && maxitAttempt > 0))
  return it
lastDiscardedIters_ = it; ++vcycleRestarts_; latch (as above)
deep_copy(x, 0.0)                           // full extent, ghosts included
scope: guessV1_ = {}, guessHist_ = 0, zeroGuess_ = true   // restored on exit (RAII)
it2 = attempt(maxit)                        // FP64 by the latch; stopRef_ as the caller set it
rank 0 prints once:  peclet::flow CutcellMG: the FP32 V-cycle did not converge this solve
  (<capped after N | non-finite at iteration N | degenerate direction at N>); re-solved with the
  FP64 V-cycle in M iterations; 'auto' stays FP64 for the rest of the run
  (set_pressure_vcycle_precision re-arms)
return it2
```
No second restart. The decision reads only `it`, `lastSolveConverged_`, `lastVcycleFp32_`, `vprec_` —
all derived from allreduced scalars or global mode state, hence identical on every rank.

---

## 5. D — normative scheme (distributed)

**5.1 Level data and lifetime.** `Level` gains (under `PECLET_FLOW_MPI`)
`std::shared_ptr<GridHalo<float>> devf` and, on the level Lf only when it telescopes,
`CCField tcorr` (double, level extent). `devf` is reset wherever `dev` is (re)created (`initMpi`
:1452ff, every rebuild); it is created in `fp32Build(L)` when `distributed_` and the level carries
FP32 data: `devf->init(*halo)`, label `"mg L<L> g<g> f32"`. `fp32Build` runs inside the collective
operator build and its decision is global, so creation is collective.

**5.2 Region and eligibility.** `fp32Last_` (int, recomputed whenever `lv_` is built): single rank
`lv_.size() − 2`; distributed `min{L : lv_[L].tele}` if any level telescopes, else `lv_.size() − 2`
(every rank holds levels ≤ the first telescope point, so it is global). `Level::globalInner` (C3) is
stored in `initMpi` from the loop's global size `gs`. Distributed predicates (the single-rank
branches stay byte-identical in source):
- `fp32Carriable()`: `!hasOutflow_ && fp32Last_ ≥ 0 && !meanRemovalAll_ && (exactResidual_ ||
  MReal = double)` and every level 0…`fp32Last_` has even `globalInner` on all three axes;
  `fp64Deferrable()` gains `&& !distributed_`; `fp32Build(L)` carries iff `L ≤ fp32Last_`.
- `fp32VcycleIneligible(krylov)`: the `distributed_` clause is replaced, for `distributed_` only, by:
  `!krylov`; outflow; overlay; `fp32Last_ < 0` → "the hierarchy has a single level"; odd
  `globalInner` on a level 0…`fp32Last_` → "a level the FP32 V-cycle runs on has an odd global inner
  dimension"; float outer; `meanRemovalAll_`; `!fp32WeightsOk(L)` for L ≤ `fp32Last_`. **WO-D2a
  only:** any telescoping level → "the hierarchy telescopes (WO-D2b)". At np = 1 every one of these
  equals its single-rank counterpart (global = local dims, no telescope), so `'auto'` picks the same
  precision as single rank.
- `fp32WeightsOk`: in distributed, the lazy read-back (:3507) is followed by
  `MPI_Allreduce(MPI_IN_PLACE, wBadHost_, nLevels, MPI_INT, MPI_MAX, comm_)`; it is collective and
  happens before any per-level early return that depends on the read-back.

**5.3 New kernels** (`mg_fp32_vcycle.hpp`, each a launch form around the existing cell bodies; host
launch forms mirror the FP64 box kernels' rule-H forms; parity expression copied from
`cutcellSmoothColorBoxFace`):
- `cutcellSmoothColorFp32Box(VField x, VConst b, VConst WX, VConst WY, VConst WZ, C3 e, C3 og,
  int color, C3 lo, C3 hi, C3 exLo, C3 exHi)` — cells of `color` in [lo, hi) \ [exLo, exHi),
  neighbours i±1, i±sy, i±sz, body `fp32SmoothCell`;
- `residualFp32Box(VField r, VConst x, VConst b, W…, C3 e, C3 lo, C3 hi, C3 exLo, C3 exHi)` — body
  `fp32ResidualCell`;
- `fp32ToDouble(CCField d, VConst f, C3 e)` — full extent, `d(i) = (double)f(i)`;
- `fp32AddDouble(VField f, CCConst d, C3 e, int g)` — inner cells, `f(i) = (VReal)((double)f(i) +
  d(i))` (the `prolongFp32FromD` rounding, P§4.4.4 item 3);
- `fill(Level&, VField)` distributed branch: `lv.devf->exchange(f)`.

**5.4 The V-cycle** (`precondVcycleFp32` keeps its signature; it calls `vcycleFp32Impl` when
`!distributed_` — unchanged — and `vcycleFp32Dist` otherwise):
```
vcycleFp32Dist(L, sym):
  smoothFp32Dist(lv, pre_, reverse=false, skipFirst = (L == 0 && pre_ > 0))
  residualFp32Dist(lv):  devf->exchangeBegin(xf); residualFp32Box(interior lo..hi);
                         devf->exchangeEnd(xf);   residualFp32Box(g..ext-g minus lo..hi)
  if (lv.tele)                                   // L == Lf (WO-D2b)
    fp32ToDouble(lv.res, resf); deep_copy(lv.tcorr, 0.0)
    teleTransition(L, sym, /*src=*/lv.res, /*target=*/lv.tcorr)   // §5.6
    fp32AddDouble(lv.xf, lv.tcorr)
  else if (L == fp32Last_)                       // bottom interface (L + 2 == lv_.size())
    restrictFp32ToDZeroX(cs.rhs, cs.x, resf, …); deep_copy(cs.x, 0.0)
    vcycle(L+1, sym)                             // FP64, unchanged (RB-GS or GraphAMG)
    fill(cs, cs.x); applyOutflowGhost(cs, cs.x, cs.g); applyNeumannGhost(cs, cs.x, cs.g)
    prolongFp32FromD(lv.xf, cs.x, …)
  else
    restrictFp32ZeroX(cs.rhsf, cs.xf, resf, …); deep_copy(cs.xf, 0.0f)
    vcycleFp32Dist(L+1, sym)
    fill(cs, cs.xf); applyNeumannGhost(cs, cs.xf, cs.g)
    prolongFp32(lv.xf, cs.xf, …)
  smoothFp32Dist(lv, post_, reverse=sym, false)

smoothFp32Dist(lv, sweeps, reverse, skipFirst):
  if caSmooth(lv):                               // never L0 (g = 1), so never skipFirst
    lv.devf->exchange(lv.rhsf)                   // ring rhs, once per call
    per sweep: c0 = reverse ? 1 : 0, c1 = 1 - c0
      devf->exchangeBegin(xf); Box(c0, lo, hi, none); devf->exchangeEnd(xf)
      Box(c0, rlo, rhi, excl lo..hi); Box(c1, g..ext-g, none)
  else per sweep, per colour s: color = reverse ? 1-s : s; if skipFirst {skipFirst = false; continue}
      devf->exchangeBegin(xf); Box(color, lo, hi, none); devf->exchangeEnd(xf)
      Box(color, g..ext-g, excl lo..hi)
```
lo/hi/rlo/rhi exactly as in `smooth` (:3812ff). No outflow (eligibility), so the FP64 path's
`applyOutflowGhost` on the FP32 field has no counterpart. The entry (`entryFp32`, no neighbour
reads) is used unchanged.

**5.5 Exit** (`exitFp32`, distributed branch only; `nFluid = −1` there, so the joint {sum, count}
reduction of the uncounted branch runs, as on a single-rank device): after the mean reduction into
`slot(kMsum)`/`kcnt_`, read both to the host, `allreduceSum2(sum, dcnt, l0.comm)`, write
the global values back into the same slots asynchronously on `CCExec()` (no fence), then the
**unchanged** z-pass body (with k ≥ 0 its fused r·z into `slot(k)` is the local partial). Ghosts of
z are not written; a distributed solve that runs FP32 zeroes z (full extent) once at its start, so
the never-exchanged wall ghosts of z and p stay 0.

**5.6 Telescope transition refactor (WO-D2b0, inert).** The `if (lv.tele) { … }` block of
`vcycleImpl` (:3363-3383) becomes `teleTransition(int L, bool sym, CCField src, CCField target)`;
`vcycleImpl` calls it with `(L, sym, lv.res, lv.x)`. Nothing else moves.

**5.7 Driver.** Host-scalar `solvePCG` (:1965ff): `f32 = fp32VcycleSolve(); lastVcycleFp32_ = f32;`
after the initial residual. The FP64 statements stay as they are inside the `else` of
```
if (f32) { if first call: deep_copy(z, 0.0)
           precondVcycleFp32(z, r, ilogb(rnow), kSlot, CCField());
           rz = allreduce(readSlot(kSlot), MPI_SUM, l0.comm) }
else     { precond(z, r); …; rz = dot(l0, r, z) }      // today
```
with rnow = r0 before the loop (rref under E, as in `solvePCGResident`) and the stop test's rn
inside it; kSlot = `kRz` before the loop, `kRzNew` inside (the resident driver's slots);
`deep_copy(p, z)` before the loop as today. `solveFCG` needs no change (its `precond` lambda already
routes to `precondVcycleFp32` with k = −1; its dots are the distributed ones). Forced `'fp32'` raises
where ineligible through `fp32VcycleSolve()`.

**5.8 Cost [model].** Per V-cycle the exchanges and Allreduces are those of the FP64 V-cycle, at half
the halo bytes; compute as single-rank D (CUDA projection 0.80 at np = 1). Added: one tiny
async H2D per V-cycle (exit write-back), one Allreduce per build, two conversion passes on the
telescope level. Expected CUDA np > 1 projection ratio 0.8–0.9.

---

## 6. E — normative scheme (distributed)

1. `projectedGuessIneligible()` (flow_ibm_core.hpp:720) drops the distributed clause;
   `setProjectedGuess` (:5033) drops its throw.
2. `projectedGuess` (:2412): `guessMatvec` is the driver's distributed matvec (`matvecOverlap` +
   `starApplyDelta` when a star is present); `fused` is false whenever distributed (already:
   `hostKrylovFusion()` requires `!distributed_`); after the one packet read (:2514), when
   `distributed_`: `MPI_Allreduce(SUM)` over the five sum entries, then `MPI_Allreduce(MAX)` over
   rref, both on `l0.comm`, in that order; then the host 2×2 solve unchanged. The trace helpers
   (`guessTraceSave/Exit`) use the distributed matvec/dot/maxabs, are collective, print on rank 0.
3. Host-scalar `solvePCG` gains the guess branch of `solvePCGResident` (:2559-2563) verbatim:
   `if (guessOn && guessHist_ > 0) r0 = projectedGuess(…, /*fused=*/false, …); else {today}`.
   `solveFCG` already has it.
4. `pguessCount_ = 0` in `init_mpi` and in every `redistribute`/rebalance path (verify each; add
   where missing).
5. Audit: no MPI-path member (redistribute carry lists, field registries in `flow_ibm_mpi.hpp`)
   holds a long-lived copy of the `phi1_`/`phiPrev1_` handles across the swap; a carry list must read
   the member at call time. List what was checked in the commit message.

---

## 7. Work orders (for `opus-implementer`; one commit each, named paths only, in this order)

| WO | content | class | acceptance |
|---|---|---|---|
| **R** | §4: latch, `lastSolveConverged_`, wrapper on `solvePCG`/`solveFCG`, strict deferral, warning line, counters, fault hook; new ctest `pressure_restart` (`tests/kokkos`); `set_pressure_vcycle_precision` docstring: the restart and latch | bitwise wherever no FP32 attempt fails | G-BIT (defaults; `'auto'`/`'fp32'` refs bitwise); G-R (a)–(g) host + CUDA |
| **A** | `'auto'` = FP32 where eligible **on device backends only**: `hostOnly` term of `effectiveVprec()`; docstring; host test arms that meant FP32 under `'auto'` switch to forced `'fp32'` | bitwise defaults; host `'auto'` becomes `'fp64'` (opt-in mode) | G-BIT; host `'auto'` ≡ `'fp64'` memcmp (column 50 steps, `mg_fp32_vcycle`); CUDA `'auto'` bitwise to WO-R |
| **D1** | §5.1–5.3: `devf`, `fp32Last_`, `globalInner`, distributed `fp32Carriable`/`fp64Deferrable`/`fp32Build`, range-flag Allreduce, the four kernels, FP32 `fill`; eligibility still excludes distributed (inert) | bitwise all modes | G-BIT; G-K1; G-K2 |
| **D2a** | §5.4 (no telescope branch), §5.5, §5.7, distributed eligibility with the temporary telescope exclusion; new ctest `mg_fp32_vcycle_mpi` (np 1, 2, 4, 8); forced-`'fp32'` arms in `cutcellmg_aniso_mpi` and `vof_wetting_mpi` | bitwise defaults and single rank; recorded (new) distributed `'fp32'`/`'auto'` | G-BIT; G-DD1; G-DD2 (a)–(e), (g), (h) on non-telescoped grids; G-R (h) |
| **D2b0** | §5.6 refactor | bitwise | G-BIT; `telescope_mpi`, `telescope_varrho_mpi`, `cutcellmg_mpi` np 1/2/4 green |
| **D2b1** | §5.4 telescope branch, `tcorr`; drop the telescope exclusion | recorded (distributed FP32 on telescoped hierarchies) | G-DD2 (f) + (a)–(c) at np 4, 8 with natural and forced (`teleForce_`) telescopes; G-BIT |
| **E** | §6; new ctest `projected_guess_mpi` (np 1, 2, 4); `set_pressure_initial_guess` docstring drops "single rank" | bitwise `'zero'` and single rank; recorded (new) distributed `'projected'` | G-BIT; G-ED1…G-ED5 |
| **G** | gate campaign, no solver source: G-DD3, G-DD4, F-3…F-6 measurements; extend `tests/study/vardensity_solver_probe.py` with the per-solve (d) table and `'auto'`; numbers appended to `doc/vof_step_performance_log.md` | — | every gate reported with numbers; a miss stops the package before WO-F |
| **F** | the flip (isolated, §9): default `kVcycleAuto` (:5445); `units_scale_invariance` pins `'fp64'`; CUDA hashes re-baselined (old → new table in the message); docstrings; `CLAUDE.md` pressure-solve paragraph; Outcome of this note | recorded (CUDA), bitwise (host) | F-1…F-7 on this commit |

Each WO's commit message names its class and gate numbers. If a gate fails and the cause is not a
local bug, the implementer stops and reports; the remedies pre-authorized here are the only ones
that may be applied without asking (G-K2 → weight exchange; R-1 → shared kernel template; R-6 → see
§10).

---

## 8. Verification gates

Common: host = host-openmp prefix, `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` (MPI ctests: threads per
rank, 2 on a loaded host); CUDA = nvidia-cuda prefix; MPI with `--bind-to none`. Scripts in `~/Codes/bubble_column_perf/gate_proj/` (`build.sh`,
`battery.sh`, `grun.sh`/`gcmp.sh`, `numcmp.py`); one-line summaries, logs to files.

**G-BIT** (every WO before F): P§13 G-BIT unchanged — `state_hash.py` all cases + np2 identical (host
and CUDA); 50-step dump from `ckpt_t43` bitwise for u v w p C with identical per-step iterations (host
1×8, 1×24, CUDA); `ctest -LE bench` green (233 non-bench + the new tests); `-R '_np[0-9]+$'` green;
two runs bitwise. Plus: the forced-`'fp32'` and (CUDA) `'auto'` 50-step dumps bitwise to `ref/d3f32_omp`
/ `gd4_cuda` (single-rank FP32 untouched).

**G-R** (restart; `pressure_restart`, CutcellMG level on the column, cyl and pack states of
`test_mg_fp32_vcycle` plus one assembled VoF case; PCG and FCG; host and CUDA; rtol 1e-10):
- (a) `'auto'`, fault kind 1 at call 3: returned x memcmp-equal to the `'fp64'` cold solve; iterations
  equal to `'fp64'`'s; `lastSolveFailed()` false; `vcycleRestarts() == 1`; `lastVcycleFp32()` false;
- (b) `'auto'`, kind 2 at 2: as (a), `lastDiscardedIters() == 2`;
- (c) forced `'fp32'` with kind 1 / kind 2: no restart; failed flag set / 2 iterations returned;
- (d) `'fp64'` with either fault: bitwise to no fault;
- (e) strict + `'auto'` + kind 1: no throw, (a) holds; strict + forced `'fp32'` + kind 1: throws;
- (f) after (a), the next `'auto'` solve runs FP64 with no restart; after `setVcyclePrecision(auto)`
  the next solve runs FP32 and is bitwise to a fresh `'auto'` solve;
- (g) assembled, `'projected'` with history 2, kind 1 on step k: step k's p memcmp-equal to the
  `'fp64'`+`'zero'` solve from the same state; the history counter advances (not reset);
- (h) MPI, in `mg_fp32_vcycle_mpi` np 1/2/4/8: (a)–(c); `vcycleRestarts()` min == max over ranks;
  np 1 memcmp-equal to the single-rank restart;
- (i) study, natural trigger: walled hydro slab, box, ratio 1e4, PCG (`vardensity_solver_probe.py`),
  host and CUDA: where forced `'fp32'` caps at step 1, `'auto'` restarts exactly once at step 1 and
  its p is memcmp-equal to `'fp64'`'s at step 1. If forced `'fp32'` does not cap on a backend,
  record it; (a)–(h) carry the gate.

**G-K1** (kernels, single rank, host + CUDA): on every non-bottom level of column, cyl, pack (ratio
1e4) and a 64³ periodic box, with ghosts filled by `fillWrap`: one `cutcellSmoothColorFp32Box` pass
over the inner box (each colour) memcmp-equal on inner cells to `cutcellSmoothColorFp32Wrap`; the same
for `residualFp32Box` vs `residualFp32Wrap`; interior + shell split equal to one full-box pass.

**G-K2** (np 2, 4, 8, CA on): after a build, for every FP32 level, `WX, WY, WZ` on all ghost cells
within depth g memcmp-equal to the owner's values (exchange copies with `devf` into scratch and
compare). Also `'fp64'` builds create no `devf` and no FP32 arrays.

**G-DD1** (np = 1 ≡ single rank, bitwise, host + CUDA; `mg_fp32_vcycle_mpi_np1` on column, cyl, pack
and a domain-BC walled slab at 1e2): (a) one FP32 V-cycle: z memcmp on inner cells; (b) forced
`'fp32'` PCG and FCG solves (maxit 200, rtol 1e-10): x memcmp, iterations equal; (c) CUDA `'auto'`
the same; (d) the forced-`'fp32'` arms of `cutcellmg_aniso_mpi_np1` and `vof_wetting_mpi_np1`
memcmp-equal to their single-rank `'fp32'` runs.

**G-DD2** (np 2, 4, 8 vs np 1, host + CUDA, forced `'fp32'` and CUDA `'auto'`):
- (a) solve: max|x_np − x_np1| ≤ 1e-8 at rtol 1e-10 (the `cutcellmg_mpi` class); iterations per
  solve within ±max(1, 5 %) of np 1;
- (b) a 64³ periodic box and the cyl state on a grid whose blocks stay even to the bottom at every np
  (no telescope, equal depth): one V-cycle z, max|Δz|/max|z| ≤ 1e-11 vs np 1;
- (c) `'fp32'` vs `'fp64'` at the same np, on configurations where `'fp64'` converges every solve:
  iterations ≤ fp64 + max(1, 5 %);
- (d) attainable floor (rtol 1e-14, cap 300) at np 2: floor_fp32 ≤ 2 × floor_fp64;
- (e) CA on vs `set_comm_avoiding` off at the same np: x bitwise equal;
- (f) telescoped hierarchies (natural at np 8 on a grid that telescopes, and `teleForce_` = 1 at
  np 4): (a)–(c);
- (g) two runs at np 2 bitwise;
- (h) no hang: every MPI ctest under a 120 s timeout.

**G-DD3** (case, WO-G): bubble column, np 2 and 4, 50 steps from the same initial state; CUDA
(ranks share the GPU) under `'auto'`, host under forced `'fp32'`; against `'fp64'` at the same np:
G-NUM-P 1 (max relative difference over u v w p C ≤ N50_8 = 1.48e-8 at rtol 1e-8), 2 (per step ±1,
50-step total ±2 %), 3 (`max_open_divergence_projected()` ≤ 2× per step). And `'auto'` np 2 vs
`'auto'` np 1 within the np-class tolerances of `vof_*_mpi`.

**G-DD4** (perf, WO-G; min of 3 rounds, projection ms/step, bubble column): CUDA np 2 `'auto'` ≤ 1.02 ×
`'fp64'` (expected 0.85–0.9); host np 4 (×6 threads) forced `'fp32'` ≤ 1.02 × `'fp64'`
(informational: host `'auto'` is FP64). A CUDA miss blocks WO-F.

**G-ED** (E distributed; `projected_guess_mpi` + assembled cases, host + CUDA):
1. np 1 ≡ single rank: `'projected'` (history 1 and 2) over 10 steps on a VoF var-ρ case and the cyl
   Stokes case: u v w p (C) memcmp, iterations equal;
2. np 2/4 vs np 1 `'projected'`: the np-class tolerances; iterations per step ≤ `'zero'` at the same
   np + 1;
3. P§13 G-E 1–4 at np 2;
4. a size-preserving rebalance under `'projected'`: the next solve memcmp-equal to a `'zero'` solve
   from the same state; the following one uses history 1;
5. `'zero'`: G-BIT.

---

## 9. The flip (WO-F) and its criterion

**Preconditions:** WO-R, A, D1, D2a, D2b0, D2b1, E, G landed on `projdist`, each with its gate.

**Commit (isolated):** `vprec_` default `kVcycleFp64 → kVcycleAuto`; `units_scale_invariance` sets
`'fp64'` explicitly (its subject is units handling; FP32 weight rounding is not covariant under a
non-power-of-two unit change — measured p 2.0e-11 / 1.05e-10 vs its 1e-13 at 37f8f5f — and D's
power-of-two covariance is gated by P§13 G-D2(a)); CUDA state hashes re-baselined with the old → new
table; docstrings (`'auto'` DEFAULT on device: FP32 where eligible, single rank and distributed, FP64
restart on failure); `CLAUDE.md` pressure-solve paragraph; this note's Outcome.

**Criterion — all on the WO-F commit; any miss reverts it:**
- **F-1** CUDA battery `ctest -LE bench` green: the eight np1 bitwise tests, every `_np` test, the new
  tests (`pressure_restart`, `mg_fp32_vcycle_mpi_np1..8`, `projected_guess_mpi_np1..4`).
- **F-2** host battery green and host state hashes bitwise to the WO-G head (host `'auto'` = FP64).
- **F-3** CUDA bubble column: the 50-step dump under the new default memcmp-equal to `gd4_cuda`
  (G-D4's `'fp32'` reference, so P§13 G-NUM-P 1–6 carry over); if that reference no longer applies,
  re-run G-NUM-P 1–6 on CUDA.
- **F-4** CUDA probe, per solve (review finding 3): on box/pack × slab/blob × ratios 1e2, 1e3, 1e4 ×
  PCG/FCG plus the walled hydro slab, every solve where `'fp64'` converges also converges under
  `'auto'` (final attempt < maxit); every restart is listed (configuration, step, discarded
  iterations) and occurs only where forced `'fp32'` capped or broke down.
- **F-5** G-DD3 under the new default (CUDA np 2, 4).
- **F-6** G-DD4 met; CUDA np 1 projection under the default ≤ 0.85 × `'fp64'` (measured 0.80); zero
  restarts over the 2000-step column run of G-NUM-P 5.
- **F-7** G-R (a)–(h) on CUDA, single rank and np 2.

---

## 10. Risks

- **R-1 [fact] FMA contraction on CUDA could differ between the wrap and box FP32 kernels** (same
  inline body, different surrounding code). G-K1 catches it. Pre-authorized remedy: one kernel
  template per pass whose only template parameter is the neighbour-index policy (wrap vs offset),
  so both instantiations share one body expression. The FP64 wrap/box pair is bitwise on CUDA today
  (the eight np1 tests), so this is unlikely.
- **R-2 [fact] `GridHalo<float>::init` might be collective.** Creation happens only in the collective
  build with a global decision (§5.1); G-DD2 (h) catches a hang.
- **R-3 [fact] Ghost face weights in distributed.** G-K2; remedy pre-authorized (§3.2).
- **R-4 [fact] A restart costs up to `maxit` discarded iterations once per run** (plus up to one
  `'fp64'` solve). F-4 lists every restart with its cost.
- **R-5 [pref] The latch keeps FP64 after a transient failure** for the rest of the run (≈ −20 %
  projection speed lost on device). Accepted for predictability; Q-1.
- **R-6 [fact] E at np 1 on host:** single rank forms the Gram dots in the fused host kernel
  (`matvecGuessDots`), distributed unfused. If G-ED1 fails on host for that reason only, the remedy
  is pre-decided: single-rank host E uses the unfused dots (recorded for host `'projected'`, an opt-in
  mode; ≈ +0.1 ms per solve). Any other cause: stop and report.
- **R-7 [fact] Telescope-heavy decompositions keep less of the gain** (FP64 below Lf). At genoa 6×4
  the FP32 region should still cover L0–L1. Measured later on Snellius (not in this package).
- **R-8 [pref] `units_scale_invariance` no longer exercises the default precision.** Accepted: its
  subject is units; precision is gated by G-D2(a) and G-NUM-P.
- **R-9 [fact] The multi-rank default driver is the standalone V-cycle** (CLAUDE.md driver table): D
  and E act at np > 1 only where PCG/FCG is selected. Nothing to change; stated so nobody expects a
  gain elsewhere.

---

## 11. Open questions (each with the default the implementer follows)

- **Q-1 [pref] Latch scope.** *Default:* sticky until `set_pressure_vcycle_precision` is called again.
  Alternatives: re-arm at every geometry rebuild; per-solve. Reversible in one function.
- **Q-2 [pref] Restart on forced `'fp32'`.** *Default:* none (forced is the measurement mode; review
  finding 3 agrees).
- **Q-3 [fact] Discarded-attempt cost under the caller's `maxit`.** *Default:* no cap on the FP32
  attempt below `maxit`; revisit only if F-4 shows a restart costing more than one `'fp64'` solve's
  wall time on a realistic case.
- **Q-4 [pref] A Python-visible restart counter.** *Default:* no new public name (warning line,
  `pressure_vcycle_precision()` reads `'fp64'` after the latch, C++ counters for tests). If wanted:
  `diagnostics.num_pressure_vcycle_restarts()` per `docs/NAMING.md` (`num_*`), one commit.
- **Q-5 [fact] D5 deferral on the distributed path.** *Default:* off (+≈ 27 B/cell, ≈ 1.3 % of the
  device peak). Revisit when multi-GPU memory binds.
- **Q-6 [pref] Should any cap set `pressure_solve_failed()`?** *Default:* no — unchanged semantics;
  a cap is read from the iteration count, as today. Raise it in the next API cycle if wanted.
- **Q-7 [pref] Restart of warm-started callers** (balanced force, `pwarm_`). *Default:* cold restart
  with the caller's stop reference; tolerance class, not bitwise, vs the warm `'fp64'` solve. The
  alternative (saving x0 before every FP32 attempt: one copy per solve) costs every solve for an
  event that happens once per run.
- **Q-8 [pref] Stagnation trigger** (e.g. FP32 22 → 104 iterations on box 1e4 FCG, converged).
  *Default:* none in v1; it is a performance loss, not a correctness one. Revisit with F-4's table.
- **Q-9 [fact] FP32 through telescope stages.** *Default:* no; revisit only if a Snellius profile
  shows the FP64 region below Lf above 10 % of V-cycle time.
- **Q-10 [pref] Tell the user before WO-F** that the flip rests on P§13's rescoped G-D3 (per
  configuration) plus F-4's per-solve check, as the review asked. *Default:* the coordinator reports
  it with the WO-G numbers.

---

## 12. Register entries proposed (the coordinator adds them as the WOs land)

1. *Restart-on-failure (WO-R):* `'auto'` re-solves a non-converged FP32 attempt cold with the FP64
   V-cycle and latches FP64 for the run. Rejected: restart from the current iterate; per-solve retry;
   a stagnation trigger; restart under forced `'fp32'`.
2. *`'auto'` precision is device-only (WO-A).* Rejected: host `'auto'` = FP32 (gain 0.92–0.95, rule
   0.88).
3. *Distributed FP32 V-cycle (WO-D2a/b):* host-scalar driver, `GridHalo<float>` per level, mirror
   rule, FP64 below the first telescope point, bottom FP64. Rejected: distributed device-resident
   PCG; float telescope stages; reusing double halo buffers.
4. *Distributed E (WO-E):* allreduced Gram packet; history reset on `init_mpi`/`redistribute`.
   Rejected: carrying φⁿ⁻¹ through `redistribute`.
5. *D default `'auto'` on device (WO-F),* with F-1…F-7 and the `units_scale_invariance` pin;
   supersedes the 2026-10-15 "defaults stay `'fp64'`" ruling's condition.
