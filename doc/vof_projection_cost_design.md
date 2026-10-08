# VoF projection and momentum cost (D + C, E, F, H): design note

**Status:** DESIGN, 2026-10-08. Brief: `doc/vof_projection_cost_brief.md`. Code read at flow main
`afbc8b6` (= `f19ac4b` for every file this note touches). Parent design:
`doc/vof_step_performance_design.md` (cited as "perf §N"); this note **supersedes perf §10 E1 and
§14 U-2** and **confirms with amendments perf §14.3 H-5**. Curvature (G) is a separate note
(`doc/vof_curvature_cost_design.md`) and is not touched here.

Markers: **[meas-S]** Snellius genoa 1×24 contiguous cores 0–23, znver4, rtol 1e-8, `'direct'`
bottom, 30-step kernel profile, S-1 job 27770799, flow `38e80e6`
(`~/Codes/bubble_column_perf/s1/summary_kprof.txt`); **[model]** derived here; **[fact]** to be
measured by a named work order. Savings are ms/step on the bubble column (128×96×64, x and z
periodic, walls in y, ρ_g/ρ_l = 0.02, rtol 1e-8, 10.19 PCG iterations/step).

---

## 0. Decisions at a glance (ranked by host saving)

| rank | item | what | host 1×24 | GPU 5080 | numerics class | risk | build |
|---|---|---|---|---|---|---|---|
| 1 | **D** | FP32 V-cycle preconditioner below the exact FP64 PCG: FP32 face weights, diagonal **derived, never stored**, flux/correction form, FP32 iterates and arithmetic, power-of-two input scaling | −6.5 (−5…−8.5) | −2.5 (−1.5…−3.5) | recorded (precision); default after gates on eligible single-rank PCG/FCG | M | L |
| 2 | **C** | host Krylov reductions: 4-lane row order (re-baseline; Q-H4 triggered), then fuse p·Ap→matvec, Σr→update, max→mean-subtract, r·z→V-cycle exit | −4.5 (−3…−5.5) | 0 | C0 recorded (host only); C1–C4 bitwise vs C0 | L | M |
| 3 | **F** | momentum: VMG stop residual as a pure max-reduction (no write, no separate max/zero-plane launches); remaining `bc_vel` launch merges | −2.3 (−1.5…−3) | −0.3 | bitwise (host + device) | L | S |
| 4 | **H** | all-fluid level flag + precomputed fluid count (host); staging copies removed (setOpenness, PCG, VMG); skip the zero-guess matvec | −2.2 (−1.5…−2.5) | −0.2 | bitwise | L | S |
| 5 | **E** | initial guess = A-norm projection onto the last two increments; stop relative to the **zero-guess** residual (bitwise to today at x0 = 0); Krylov drivers only | −2 (−0.5…−4) [fact] | −0.6 (−0.2…−1.4) | recorded (iteration path) | M, mitigated by construction | M |
| — | **S** (conditional) | host wrap kernels without per-cell wrap selects (peel row ends): bitwise | 0…−4 [fact] | 0 | bitwise | L | S |
| — | **F4** | persistent host team for the coarse tail | not in this package (§10) | | | | |

**Package total [model]:** host −17.5 (−12…−23), GPU −3.6. **End state (§16):** 1×24 ≈ 61 ms
after A(b) and this package, ≈ 55 ms with G's −5.7 target, against TBFsolver's 46 (≈ 1.2×).

**Order of work:** P0 → H → F → C0 → C1–C4 → (S if P0 says go) → D0–D3 → D5 → E0 → E1; X1
independent. Every item has its own commit(s) and gate (§12, §13).

---

## 1. Problem and scope

The single-rank host step is 93.1 ms against TBFsolver's 46 on the same 24 genoa cores [meas-S].
The projection is 55.3 ms of it, the momentum solve 10.6. A(b) (done, being confirmed) removes
≈ 15 of the 16.9 ms `mg_bottom_factor`. This note designs what comes next in those two stages.

**In scope:** the single-rank MG-PCG pressure solve (`CutcellMG::solvePCGResident`, the V-cycle
`precondVcycle`/`vcycleImpl`, the host Krylov reductions), the initial guess of the projection
solve, the velocity-MG solve loop (`VelocityMG::solve`), the staging copies and per-step rebuild
plumbing around both. Host and device, one Kokkos source.

**Out of scope:** curvature (G); distributed H-6/H-7 (perf §14); the case tolerance (1e-8, user);
FFT / constant-coefficient drivers (E2(a) parked, E2(b) dropped); the momentum algorithms; the
`'direct'` bottom engine (unchanged; A(b) is its own package); distributed D and E (eligibility
predicates keep the distributed code byte-identical; §15 Q-P3); implementing anything.

---

## 2. Constraints and invariants

**Mathematics (unchanged).** The solver solves `A φ = b`, `b = −div(open·u*)`, face coefficient
`c_f = open_f·ρ0/ρ_f` (arithmetic face mean of ρ), `t_f = c_f·gf_L` with `gf_L` the level metric
(exact powers of two on isotropic grids). In band form `AC_i = Σ_f t_f`, off-diagonal `−t_f`. The
operator is singular on the periodic/wall path; its null space is one constant per connected fluid
component; the mean is removed over fluid cells (`AC > 1e-30f`). The level-0 Krylov matvec is the
exact flux form from double openness (`exactResidual_`, on under `enable_vof()`); otherwise the
double bands (`MReal = double`, the default). Coarse operators are **rediscretized** from averaged
openness (`coarsenOpenAvg`, must not become harmonic). Restriction is the 8-child average,
prolongation trilinear: **R ≠ c·Pᵀ, so today's V-cycle is already a non-symmetric preconditioner**
(the reason `solveFCG` exists).

**Register rules that bind D** (`../docs/decisions/flow.md`):
- *Defect-correction rule* (settled 2026-09-01): Krylov matvec and residual = the exact double
  operator in flux form; V-cycles, smoothers and AMG below are preconditioners and **may be float**.
- *Precision policy rule* (2026-08-31): a quantity an algorithm requires to satisfy an exact
  discrete identity is stored in the precision in which the identity is asserted.
- *Double operator storage is the default* (2026-09-11): float storage silently broke `A·1 = 0`.
  D does **not** change `MReal`; the outer operator stays double.
- *Success criterion* (2026-08-31): attainable residual floor ≤ the full-double floor at matched
  configuration and iteration parity — never "reaches rtol" alone.
- *Device bottom* (2026-10-03): an FP32 factor inside an FP64-verified iteration is admissible; an
  unrefined FP32 bottom is not. A·1 = 0, the mean projection and the stopping residual stay FP64.
- No environment variable changes a result; ablations are `s.diagnostics` setters; strings for
  modes; one setter per concept (`../docs/NAMING.md`).
- x-fastest iteration; rule H on host (pencils/rows, no host `MDRange3` in hot paths); host
  `-ffp-contract=off`; the host-serial cutoff 8192 stands; host-only branches select on
  `std::is_same_v<memory_space, HostSpace>`; device bits unchanged unless recorded.
- MPI: every new fast path is single-rank by an eligibility predicate; `distributed_` takes today's
  code, byte-identical in source except shared helpers whose double instantiation is unchanged.

**Classes.** *Bitwise:* byte-identical u, v, w, p, C and block colours after N steps plus identical
per-step iterations, same build/backend/thread count (G-BIT, §13). *Recorded:* a register entry plus
G-NUM-P (§13). A host-only recorded change must leave CUDA bitwise.

**Device reduction caveat (decides C's and H-1's scope).** On CUDA, Kokkos sizes a reduction's
blocks from the functor's resources; fusing a reduction into another kernel, or removing a read
from it, can change the block size and so the summation tree. Sum-fusions are therefore **bitwise
only on host** (whose order is the recorded B2 pencil order). Max/min reductions are order-free and
bitwise on both.

---

## 3. Measured breakdown and byte model

**3.1 Profile [meas-S]** (ms/step, launches): projection 55.3 = `mg_bottom_factor` 16.9 (1) +
`cc_smooth` 12.4 (240) + `mgmeanr`+`mgmeans` 5.6 (44) + `mgdot` 3.5 (20) + `prolong` 2.7 (30) +
`mg_bottom_direct` 1.9 (10) + `rhs_var` 1.8 + `cc_residual` 1.5 + `cc_apply_exact` 1.4 +
`restrict` 0.9 + `mgpcg_update` 0.8 + `rho_coeff` 0.5 + `cc_build_op` 0.5 + `correct_var` 0.45 +
≈ 4.4 other (max, aypx, fills, one-thread kernels). Momentum 10.6 = `vmg_resid` 3.6 +
`ibm_build_diff_var` 3.5 + `ibm_rbgs` 2.9 + `bc_vel` 2.3 (47) + `vmg_maxabs`+`maxabsdiff` 1.1.
Host deep copies 1.8 (11), zero memsets 1.2 (62). 937 launches/step. Spread over 12 CCDs is 10 %
faster than contiguous.

**3.2 What bounds the host kernels [model].** A level-0 colour pass moves 56 B/cell × 841 k cells
(with ghosts) = 47 MB. Fitting the 240 launches (8 passes × ≈ 11.2 V-cycles × 3 levels) gives
≈ 105 µs per L0 pass, i.e. ≈ 420 GB/s — above what three CCDs draw from DRAM, far below their L3
peak. The projection's active working set is ≈ 94 MB (L0: operator 32, openness 24, Krylov x r p z
Ap 40, res 8 B/cell; plus ≈ 7 MB of coarse levels) against 96 MB L3 + 24 MB L2 on cores 0–23.
So the hierarchy is **cache-resident, and the smoother is bound by per-update work** (stride-2
colour access, the wrap selects of `ccWrapNbrs` that turn the x-neighbour loads into gathers, one
division) at ≈ 24 cycles per updated cell, not by DRAM bytes. The reductions are slower still:
`mgdot` ≈ 175 µs and `mgmeanr` ≈ 100 µs per launch against ≈ 50 µs of bandwidth time — the
loop-carried scalar add chain of the recorded B2 order. Consequences: a byte cut speeds the host
smoother by less than the byte ratio (D's host range is wide and WO-D0 measures it), FP32 helps
host compute by doubling SIMD width, and the reductions want lanes (C0) more than fewer bytes.

**3.3 GPU [model].** RTX 5080: ≈ 900 GB/s DRAM, 64 MB L2, FP64 at 1/64 of FP32 (≈ 0.44 G DP
instructions/µs). A double GS update costs ≈ 15 DP instructions (6 FMA, a software DP division):
≈ 13 µs of DP issue per L0 pass against ≈ 10 µs (L2) to 50 µs (DRAM) of data. The device smoother
is therefore co-limited by **FP64 arithmetic**; FP32 arithmetic removes that limit, FP32 storage
halves the traffic and lets the V-cycle's level-0 data (24 B/cell → 20 MB) live in L2.

**3.4 Level-0 bytes per cell per PCG iteration** (bubble column; "after" = C + H + D):

| component | today | after | how |
|---|---|---|---|
| exact matvec | 40 | 40 | p·Ap fused into it (C1) |
| p·Ap dot | 24 | 0 | C1 |
| x, r update | 48 | 48 | Σr fused (C2) |
| mean of r: reduce + subtract | 40 | 16 | sum in C2; all-fluid, no mask read (H-1) |
| max\|r\| | 16 | 0 | fused into the subtract (C3) |
| z = 0 per V-cycle | 8 | 0 | D's entry pass |
| smoother, 8 colour passes | 448 | 196 | D: entry 28 + 7 × 24 |
| residual for restriction | 56 | 24 | D |
| restrict + prolong | 27 | 14 | D |
| L0 exit mean of z | 40 | 24 | D's exit pass (reduce x_f 4; pass x_f 4, z 8, r 8 for r·z) |
| r·z dot | 24 | 0 | C4 (fused into the exit pass) |
| p = z + βp | 24 | 24 | |
| levels 1, 2 | ≈ 76 | ≈ 33 | D |
| **total** | **≈ 871** | **≈ 419** | ×0.48 |

---

## 4. D — the FP32 V-cycle preconditioner

### 4.1 Decision

On eligible solves (§4.4.1) the V-cycle that preconditions the FP64 PCG/FCG runs in FP32 on every
level above the bottom:
- **storage:** per level, three FP32 face-weight arrays `WX, WY, WZ` (`w_f = fl32(t_f) > 0`) and
  FP32 iterates `xf, rhsf, resf`. **No diagonal is stored in any precision**: `D_i = Σ_f w_f` is
  formed from the six weights the update reads anyway;
- **arithmetic:** FP32 throughout the V-cycle, in **flux/correction form**:
  `r_i = b_i − Σ_f w_f (x_i − x_j(f))`, `x_i ← x_i + r_i / D_i`;
- **interfaces:** the input r (FP64) is scaled by an exact power of two and rounded once; the
  output z is converted once, mean-removed and rescaled in FP64; the bottom keeps its FP64
  vectors and its unchanged `'direct'`/`'algebraic'` engine;
- **outer:** PCG stays (FCG is the contingent fallback, WO-D4c). Matvec, dots, mean projection,
  stop test, x, r, p, Ap stay FP64 — the defect-correction rule, verbatim.

### 4.2 Why the retired float storage failed, and why D cannot (the null-space argument)

Let `w_f > 0` be a level's exact weights, `w̃_f = fl32(w_f) = w_f(1+η_f)`, `|η_f| ≤ u = 2⁻²⁴`.

**P1 (exact null space).** `(Ãx)_i = Σ_f w̃_f (x_i − x_j)` gives `Ã·1 = 0` **exactly**, in exact
arithmetic and in FP32 arithmetic: every difference of equal values is exactly 0.

**P2 (symmetry).** Each face weight is stored once and read by both cells: `Ã = Ãᵀ` exactly.

**P3 (contrast-free spectral equivalence).** `xᵀÃx = Σ_f w̃_f (x_i − x_j)² ∈ [(1−u), (1+u)]·xᵀAx`.
Ã and A have the same null space (same connectivity, given the range check of §4.4.1) and
`κ(Ã⁺A) ≤ (1+u)/(1−u)` on the range — independent of the coefficient contrast, the grid size and
κ(A).

**P4 (what the old storage did).** Rounding AC independently adds a diagonal
`Δ = diag(δ_i)`, `|δ_i| ≤ u·AC_i`, which is not a Laplacian: `Ã'·1 = δ ≠ 0`. On a near-null mode
v (eigenvalue λ) the perturbation is `vᵀΔv/vᵀv ~ u·AC_max`, so the relative change is
`u·AC_max/λ_min = u·κ(A)`. Bubble column: κ ≈ 6·50/(π/128)² ≈ 5·10⁵ → u·κ ≈ 0.03. A 256³ bed at
contrast 10⁴: κ ≈ 4·10⁸ → u·κ ≈ 24 — the near-null eigenvalue changes sign, the preconditioner is
indefinite, the mean projection no longer matches its null vector, and the solve floors, rebounds
or returns a non-finite z. That is SCALING_ISSUES #1 and the porous path's "non-finite
preconditioner on 2 of 5 steps", quantitatively. Evaluating `AC·x_i − Σ w x_j` in FP32 does the
same per application (error ~u·AC_i|x_i|): **the AC form is barred from FP32 arithmetic**.

**P5 (FP32 arithmetic in flux form).** The evaluation error of `fl(r_i)` is
≲ 8u·(|b_i| + Σ_f w̃_f|x_i − x_j|): proportional to the local fluxes, zero for locally constant x.
Near-null modes of high-contrast operators (piecewise constant on strongly coupled islands) keep
relative accuracy O(u). The diagonal `D_i` computed in FP32 only perturbs the relaxation weight
(`x_i += r_i/D̃_i`); the smoother's fixed point is still `r = 0` of Ã.

**P6 (iterate storage).** Storing x in FP32 perturbs it by `δx ~ u|x|`; in the A-norm
`‖δx‖_A/‖x‖_A ≲ u·√κ(A)`: 4·10⁻⁵ on the bubble column, 1.2·10⁻³ at κ = 4·10⁸. This is the
preconditioner's rounding variability δ. It is **√κ, not κ**, because it is a perturbation of a
vector, not of the operator.

**P7 (what PCG does with that variability).** PCG's step `α_k = r_k·z_k / p_k·Ap_k` is the exact
A-norm line search along `p_k` for **any** preconditioner output z_k (fixed, varying,
non-symmetric or indefinite), because `r_k·p_{k−1} = 0` follows from the previous step alone
(§5.1 T1). Variability δ only erodes conjugacy with older directions. The operator is symmetric
(P2) but today's V-cycle already is not (R ≠ cPᵀ), and PCG converges in ≈ 10 iterations with it;
δ ≤ 10⁻³ on top of that structural non-symmetry is invisible. The record agrees: P1 of the
defect-correction campaign (register "P1 passed") ran a float-band hierarchy — with the bad
independently rounded diagonal — under the exact double matvec and matched full double exactly
(RCP bed 14/14/28 iterations). D removes the diagonal defect P1 still carried.

**P8 (the singular coarse problems).** Every FP32 level operator is exactly singular with the
per-component constant null space (P1). The L0 exit removes the mean in FP64 (fine scope, as today).
The bottom receives an FP32-rounded restricted residual whose mean is not exactly zero; its FCG
already projects its rhs (`removeMeanBottom`), as it does today with double rounding.

### 4.3 Rejected alternatives

- **FP32 faces + FP64 arithmetic + FP64 iterates ("D-lite").** 36 instead of 24 B per pass and no
  conversions, but the device stays FP64-issue-bound (§3.3): ≈ half of D's host saving and a third
  of its GPU saving. Rejected as the end state; it is WO-D0's control.
- **FP32 faces with a stored FP64 diagonal (20 B).** Correct (P1–P3 hold if the stored diagonal is
  the FP64 sum of the FP32 faces), but 8 B/cell more per pass for a value the pass can form from
  operands it already loads. Rejected.
- **Float storage of today's bands (the retired default).** P4. Rejected (register).
- **The double-diagonal fallback.** Retired by the register (it made the *outer* operator the
  float-face one). Not applicable: D never touches the outer operator.
- **FP32 only on coarse levels.** Levels 1–2 are 14 % of the bytes. Rejected.
- **BF16/FP16 faces.** Spectral equivalence 1 ± 4·10⁻³ would still precondition well, but host
  arithmetic, Kokkos portability and the iterate range need their own design for ≈ 25 % more.
  Not now.
- **D for the velocity MG.** The VMG solve is a stationary V-cycle iteration: its level-0 smoother
  defines the fixed point, so FP32 there would solve the FP32 operator (the double-diagonal lesson).
  Only levels ≥ 1 could be FP32, ≤ 0.4 ms. Rejected.
- **D under Chebyshev / standalone V-cycle / BiCGStab.** Not needed by the case; Chebyshev's bounds
  assume a fixed preconditioner. Out of v1 (eligibility).

### 4.4 The scheme (normative)

**4.4.1 Eligibility** (`CutcellMG::fp32VcycleEligible()`, evaluated per solve; all must hold):
1. the solve's preconditioner is `precondVcycle` called from `solvePCGResident` or `solveFCG`;
2. `!distributed_`, `!overlaySolve_`, `!hasOutflow_` (that is, `fusedWrapReads()`);
3. every non-bottom level satisfies `fusedWrapSmooth(lv)` (even inner dims) and has at least one
   coarser level (with a single level D does not apply);
4. the outer operator is double: `exactResidual_ || std::is_same_v<MReal, double>`;
5. the level face-range check of the last build passed: every face weight t satisfies `t == 0` or
   `1e-30 ≤ t ≤ 1e30` (one min/max reduction per level in the build, stored as a flag). This makes
   "all six FP32 weights are 0" the same cell set as the FP64 `AC < 1e-30` set; the build asserts
   the count equality in debug builds.
If any fails, the solve uses today's FP64 V-cycle, unchanged.

**4.4.2 Data** (per level L < bottom, single rank, g = 1):
- `Kokkos::View<VReal*, CCMem> WX, WY, WZ, xf, rhsf, resf`, extent = the level's full extent,
  zero-initialized. `using VReal = float;` is declared once, with
  `// PRECISION-EXEMPT: FP32 V-cycle preconditioner below the exact FP64 Krylov
  (doc/vof_projection_cost_design.md §4)`.
- Built in the same kernel as the double face form (`buildCutcellOpFace`, `mac_pressure.hpp:171`):
  `WX(i) = (VReal)(ox(i) * gfx)` (and y, z), over [0, ext), from the identical double operands
  (`tw = ox(i)*gfx` of `cutcellBuildFaceOpCell`). Positive sign. In v1 both forms are built
  (runtime A/B without a rebuild; +12 B/cell writes per build ≈ 0.1 ms); WO-D5 removes the dead
  ones.
- The bottom level keeps its double `x, rhs` and its engine.

**4.4.3 Cell bodies** (new, shared by every launch form; the FP64 bodies stay untouched):
```
D_i  = ((((WX(i+sx) + WX(i)) + WY(i+sy)) + WY(i)) + WZ(i+sz)) + WZ(i)      // FP32, this order
q_i  = WX(i)*(x_i - x_xm) + WX(i+sx)*(x_i - x_xp) + WY(i)*(x_i - x_ym)
     + WY(i+sy)*(x_i - x_yp) + WZ(i)*(x_i - x_zm) + WZ(i+sz)*(x_i - x_zp)  // left to right
res_i = b_i - q_i                                                           // residual body
smooth (colour cell, D_i != 0):  x_i = x_i + (b_i - q_i) / D_i              // D_i == 0: skip
```
Neighbour indices come from `ccWrapNbrs` exactly as the FP64 wrap kernels take them. Every literal
in an FP32 body carries `f`; no double variable enters an FP32 body.

**4.4.4 The V-cycle on eligible solves** (same schedule as today: pre 2, post 2 with reversed
colours, bottom 12; colour parity `parityOg`):
1. **Entry (L0, replaces the z = 0 memset and the first pre-smooth colour pass).** One pass over
   L0's inner rows: `rhsf(i) = (VReal)(rr(i) * σ)` for every inner cell; on the cells of the first
   colour of the first pre-smooth sweep (today's order) with `D_i != 0`, `xf(i) = rhsf(i) / D_i`;
   on every other inner cell `xf(i) = 0`. σ = 2⁻ᵉ with
   `e = ilogb(rn)`, rn = max|r| of the vector being preconditioned (the host already holds it: r0
   before the loop, the packet's rn inside it). Multiplying by a power of two is exact. rn = 0 or
   non-finite never reaches the preconditioner (today's guards).
2. Remaining pre-smooth colour passes, residual, restriction, recursion, prolongation,
   post-smooth: as today on every non-bottom level, with the FP32 bodies on `xf, rhsf, resf, W*`.
   Restriction FP32→FP32 (the 8-child sum times `0.125f`) zeroes the coarse `xf`; prolongation
   FP32←FP32 with weights `0.25f/0.75f` and `(1.0f − w)` written as a subtraction. The coarse-ghost
   `fill` and `applyNeumannGhost` before each prolongation get FP32 overloads (same copies).
3. **Bottom interface.** Restriction FP32→FP64: the children summed in double from `(double)resf`,
   times the double inverse count, into the bottom's `rhs`, zeroing its `x`. The bottom runs as
   today. Its fill/Neumann run on the double `x` as today. Prolongation FP64→FP32:
   `xf(i) = (VReal)((double)xf(i) + interp_d)` with the interpolation in double.
4. **Exit (L0, replaces `removeMean(lv, lv.x)` and the subsequent `dotTo(r, z)`).** A reduction
   `m = Σ_fluid (double)xf(i) / nFluid` (C0's lane order on host); then one pass: on fluid cells
   `z(i) = s·((double)xf(i) − m)`, on non-fluid inner cells `z(i) = s·(double)xf(i)` (today's
   semantics), with `s = 2ᵉ`, accumulating `r(i)·z(i)` over fluid cells into the caller's slot
   (C4's order). z's ghosts are not written (the wrap path never reads them).
5. The V-cycle is called through `precondVcycle(zz, rr, e, rzSlot)`; the FP64 path ignores e.

**4.4.5 Outer driver.** `solvePCGResident` passes e (from r0 / the packet rn) and receives r·z in
the slot it uses today (`kRz` before the loop, `kRzNew` inside). `solveFCG` (host loop) passes e and
keeps its own dot. PCG stays the default driver. **Health instrument** (debug printing only, under
`PECLET_FLOW_MG_DEBUG ≥ 2`, which is an allowed print knob): the orthogonality-loss ratio
`|r_{k+1}·z_k| / |r_{k+1}·z_{k+1}|` per iteration (one extra dot only when tracing).

**4.4.6 API.** `s.diagnostics.set_pressure_vcycle_precision(mode)`, mode ∈ `'auto'` (FP32 where
§4.4.1 holds, else FP64), `'fp64'`, `'fp32'` (raises `RuntimeError` naming the failed condition
when not eligible, as `set_pressure_bottom_solver('direct')` does); query
`s.diagnostics.pressure_vcycle_precision()` → the precision the last solve used. The setter takes
effect at once (the FP32 weights are rebuilt from the stored level openness). Default `'fp64'`
until WO-D3's promotion commit sets `'auto'`.

**4.4.7 Memory (WO-D5).** When the effective precision is FP32 and the outer is exact, these are
dead and are allocated on demand only: L0 `AFX, AFY, AFZ`, L0 `x, rhs, res` (A5 rebinds x and rhs;
res is the FP64 V-cycle's), and on levels 1…bottom−1 `AC, AFX, AFY, AFZ, x, rhs, res`. Kept: L0
`AC` (fluid predicate and `nFluid`), every `ox, oy, oz` (exact matvec, coarsening, rebuild), the
bottom level whole. Net L0 change: +24 B/cell (FP32 data) − 48 = **−24 B/cell**. Any other reader
of a listed array found by grep keeps that array; the implementer lists it in the commit.

### 4.5 Cost per kernel [model]

| kernel | S-1 host | D host | note |
|---|---|---|---|
| `cc_smooth` | 12.4 | 6.5–9.5 | 56 → 24 B/pass; FP32 SIMD ×2; launch floor ≈ 1.7 ms |
| `cc_residual` | 1.5 | 0.8 | 56 → 24 B |
| `prolong` | 2.7 | 1.8–2.2 | 17 → 9 B, FP32 SIMD |
| `restrict` | 0.9 | 0.5 | |
| L0 exit (after C) | ≈ 1.4 | ≈ 0.8 | |
| **D total** | | **−6.5 (−5…−8.5)** | smoother speedup 1.3–2× decides the range (WO-D0) |

GPU: the V-cycle share of the iteration falls from ≈ 655 to ≈ 290 B/cell and its FP64 issue limit
disappears: −2.5 (−1.5…−3.5) of ≈ 17.5 ms projection. Working set: host projection ≈ 94 → ≈ 75 MB
(fits the 96 MB L3 of cores 0–23; may shrink the contiguous-vs-spread gap — S-3 measures it).

### 4.6 Failure modes and the gate that catches each

| failure | caught by |
|---|---|
| an FP32 diagonal is stored or the AC form is evaluated in FP32 | G-D1(a) `Ã·1 == 0` bitwise; review rule §4.4.3 |
| a face underflows/flushes and changes connectivity | eligibility §4.4.1(5); G-D1(c) decoupled-set equality |
| range trouble at tiny/huge residuals | power-of-two scaling; G-D2(a) exact scale covariance |
| accidental FP64 promotion in a body (host) | G-D1(e) host scalar-float reference, bitwise |
| loss of conjugacy from variability | G-D3 iteration parity; health ratio; WO-D4c FCG |
| new indefiniteness of the preconditioner | G-D2(b) dense `sym(M)` spectrum (`tests/study/mg_precond`) |
| converging to the wrong fixed point | impossible by construction (outer exact); G-D3(c) true residual |
| silent cap | G-D3(d) cap never reached; `pressure_solve_failed()` |
| ineligible path changed | G-D1(f): `'auto'` on ineligible configs is bitwise to `'fp64'` |
| GPU memory growth | WO-D5 memory report |

---

## 5. E — a better initial guess (and Q11)

### 5.1 Q11 from first principles

**T1 (PCG/FCG never amplify the guess).** In exact arithmetic, for any sequence of preconditioner
outputs z_k (fixed or not, symmetric or not, definite or not), with `p_0 = z_0`,
`p_k = z_k + β_{k−1}p_{k−1}`, `α_k = r_k·z_k / p_k·Ap_k`:
`r_k·p_{k−1} = r_{k−1}·p_{k−1} − α_{k−1}p_{k−1}·Ap_{k−1} = r_{k−1}·(p_{k−1} − z_{k−1}) =
β_{k−2} r_{k−1}·p_{k−2} = 0` by induction, so `r_k·p_k = r_k·z_k` and α_k is the exact A-norm line
search along p_k. Hence `‖e_{k+1}‖_A² = ‖e_k‖_A² − (r_k·p_k)²/(p_k·Ap_k) ≤ ‖e_k‖_A²`. The same
holds for FCG (same α). Rounding perturbs `r_k·p_{k−1}` at O(ε₆₄·κ): negligible.

**T2 (an r0-relative stop can only under-solve, boundedly).** With x0 = φⁿ⁻¹ and a stop
`max|r| < rtol·max|r0_warm|`, the defect left is ≤ rtol·max|r0_warm|, and
`r0_warm = P b_n − Aφⁿ⁻¹ ≈ P(b_n − b_{n−1}) + r_{n−1}` on a fixed operator. So
`|r_n| ≲ rtol·(|b_n − b_{n−1}| + |r_{n−1}|)`: a contraction of factor rtol on the defect sequence.
It cannot grow geometrically over steps.

**Conclusion.** Under PCG/FCG on an SPD-on-range double operator, a warm start **cannot** produce
the registered exponential divergence (k → −1.7·10¹²⁰ by step 400 ≈ a factor 2 per step). Three
mechanisms can, and the registered run (undated, `porous-scaling-benchmark.md`, 192³ bed) plausibly
had at least one:
- **M2, a non-Krylov driver.** The standalone V-cycle (the multi-rank default) and Chebyshev (the
  porous/variable-ρ default) give `e_k = p_k(BA)e_0`, with |p_k| > 1 on components of BA outside
  the damped interval — present because the V-cycle is non-symmetric and, at high contrast,
  indefinite (register; the coarsening). A cold start begins from e_0 = x*, which carries little of
  those components; a warm start begins from the previous solve's error, which carries their
  *amplified* content, so the amplification compounds geometrically across steps.
- **M3, a float outer operator** (the pre-2026-09-11 default on the non-VoF path, whose matvec read
  float bands): P4's indefinite near-null mode makes the "A-norm" no norm, so T1 fails, and the warm
  start accumulates the near-null component across steps.
- **M1, under-solve (the perf §9 Q11 hypothesis).** Real but bounded (T2); not a divergence.

**Prediction (falsifiable, WO-E0b):** the registered setup diverges with `set_pressure_warmstart`
under the standalone V-cycle or Chebyshev, or under a float operator build; it does not under
MG-PCG with the double operator. E is immune to M1–M3 by construction (next section), so E does
not wait for the experiment.

### 5.2 Decision

The projection solve starts from the **A-norm projection of the solution onto the span of the last
two pressure increments**, and stops relative to the **zero-guess residual**:
- `x0 = c1·φⁿ + c2·φⁿ⁻¹` with `[φᵢᵀAφⱼ] c = [φᵢᵀ P b]` (Galerkin on span{φⁿ, φⁿ⁻¹}, current A).
  So `‖x* − x0‖_A ≤ min over the span`, which contains 0, φⁿ and 2φⁿ − φⁿ⁻¹: **never worse than
  the cold start**, and it picks the dt-dependent scaling (φ ∝ dt²∂P/∂t under variable dt) by
  itself;
- stop `max|r_k| < rtol·rref` with `rref = max|P b|` — exactly today's r0 when x0 = 0, so the cold
  path stays bitwise; never under-solves relative to the cold solve (M1);
- **Krylov drivers only** (PCG, FCG): excludes M2; the outer operator is double (M3);
- the guess is zero on non-fluid cells (no free-variable carry-over).

**Rejected:** the existing `set_pressure_warmstart` semantics (x0 = φⁿ, r0-relative stop: M1, and
M2 under the drivers that allow it); unprojected `2φⁿ − φⁿ⁻¹` (worse than zero when φ oscillates or
decays faster than 2× per step; needs dt-ratio caps); Fischer's projection with L > 2 (the operator
changes every step under variable ρ, so the A-orthonormal basis must be rebuilt each step: L
matvecs; revisit for fixed-geometry IBM marches); warm starts under Chebyshev, the standalone
V-cycle or BiCGStab (no T1).

### 5.3 The scheme (normative)

**Eligibility:** driver PCG or FCG; not the balanced-force solve (it has its own warm start and
`stopRef_`); `!pwarm_` (the legacy setter keeps its exact old behaviour); `!ghostProjection_`;
single rank in v1.

**State (Solver):** `phiPrev1_` (n1_ doubles, zero-initialized) and `histCount_ ∈ {0, 1, 2}`.
`histCount_ = 0` on: geometry/hierarchy (re)build (`set_solid`, `set_pressure_geometry`,
`init_mpi`, `redistribute`/rebalance), any state write (`set_field`/`set_state`/`set_u`… of u, v,
w, p, phi), a driver or guess-mode change, and a failed or capped solve.

**Per solve** (`projectSolve`, replacing `deep_copy(phi1_, 0.0)` at `flow_ibm_project.hpp:1372`
when E applies):
1. Buffers: A = `phi1_` (holds φⁿ), B = `phiPrev1_` (holds φⁿ⁻¹ if histCount_ = 2). The solve
   writes its solution into B. After the solve, swap the Views (`phi1_ ↔ phiPrev1_`) so `phi1_` is
   the new solution and `phiPrev1_` = φⁿ; `histCount_ = min(histCount_ + 1, 2)`. No copies. The
   implementer verifies that no member stores a long-lived copy of the `phi1_` handle (grep;
   Chebyshev's `chebX0_` is a separate buffer).
2. `histCount_ == 0`: B is zeroed (full extent) and the solve runs cold — bitwise to today.
3. Otherwise, inside the solver before its loop (a helper shared by `solvePCGResident` and
   `solveFCG`):
   - `r = P b` (today's `r = b; removeMean(r)`), `rref = max|r|` (today's r0 read);
   - `Ap ← A·v1` fused with `G11 = v1·Av1`, `g1 = v1·r`; if histCount_ = 2, `z ← A·v2` fused with
     `G22 = v2·Av2`, `G12 = v1·Av2`, `g2 = v2·r` (fluid cells; host lane order);
   - **one** host packet read {rref, G11, G12, G22, g1, g2} (it replaces today's r0 read);
   - host 2×2 solve in double: if `!(G11 > 0)` or non-finite → c = (0, 0). `l11 = √G11`,
     `l21 = G12/l11`, `d = G22 − l21²`; if histCount_ < 2 or `!(d > 1e-10·G22)`: `c1 = g1/G11`,
     `c2 = 0`; else `y1 = g1/l11`, `y2 = (g2 − l21·y1)/√d`, `c2 = y2/√d`, `c1 = (y1 − l21·c2)/l11`.
     Any non-finite c → (0, 0);
   - c = (0, 0): zero B (full extent), keep r — the cold state exactly;
   - else one pass: `x(i) = fluid_i ? c1·v1(i) + c2·v2(i) : 0` into B (in place is safe:
     elementwise), ghosts of B set to 0; `r(i) −= c1·Av1(i) + c2·Av2(i)` on inner cells; then
     `removeMean(r)` with max|r| fused (C3) → r0w (trace only);
   - the loop runs as today (the existing `r0 > 0 && isfinite` guard uses rref) with the stop
     `rn < rtol·rref`. At least one iteration runs whenever rref > 0 (today's structure already
     does; an exact guess exits through the converged-direction branch).
4. A later `removeMean(x)` at exit (today's) removes any constant the basis carried.

**API.** `s.diagnostics.set_pressure_initial_guess(mode, history=2)`, mode ∈ `'auto'`
(projected where eligible), `'zero'`, `'projected'` (raises where ineligible); `history` ∈ {1, 2}
for the measurement. Default `'zero'` until WO-E1; `'auto'` after a passed promotion. The public
`set_pressure_warmstart` is untouched (Q-P5).

### 5.4 Cost and saving [model]

Cost: 2 exact matvecs with fused dots + 1 pass + 1 host read ≈ +0.4 ms host, +0.15 ms GPU. A PCG
iteration costs ≈ 3.4 ms host today and ≈ 1.9 ms after C + H + D (≈ 1.2 → 0.7 ms GPU). The V-cycle
reduces the error ≈ 6× per iteration (10.19 iterations to 1e-8); a guess with relative A-norm error
ε_g saves log(ε_g)/log(0.164): ε_g = 0.3 → 0.7, 0.1 → 1.3, 0.03 → 1.9 iterations. Interface motion
is ≤ ¼ cell per step (capillary/CFL dt), favouring small ε_g; height-function curvature noise is
uncorrelated step to step and limits it. **ε_g is a fact** (WO-E0). Expected −2 (−0.5…−4) host,
−0.6 GPU. Promotion rule in WO-E1.

---

## 6. C — host Krylov reduction fusion (perf §14.3 H-5: confirmed, amended)

**Confirmed:** (i) the matvec accumulates p·Ap; (ii) the update `x += αp, r −= αAp` accumulates Σr
over fluid cells; (iii) the mean subtract of r takes max|r|. Host branch only, single-rank resident
PCG (and the host-loop paths that share the helpers).

**Amendments:**
1. **C0, lanes first (recorded, host only).** Q-H4's trigger ("unless S-1 shows reductions
   > 2.5 ms per step") is met: `mgmeanr`+`mgmeans` 5.6 + `mgdot` 3.5 [meas-S]. New helper
   `ccReduce3Lanes` beside `ccReduce3` (`src/mac_cutcell.hpp:385`): the same row partition; within
   a row, lane ℓ = (x − lo.x) & 3 accumulates its cells in ascending x from 0.0; the row value is
   `(s0 + s1) + (s2 + s3)`; the thread accumulator adds row values in row order; thread partials
   combine in Kokkos' thread order (as B2). A masked-out cell is skipped (no `+0.0`). Used **only**
   by CutcellMG's sum reductions (`dot`/`dotTo`, the `removeMean` sum) and by every fused kernel of
   C1–C4, D and E — no other `ccReduce3` user changes. Max reductions keep `ccReduce3`.
2. **(iv) r·z into the L0 exit.** The exit's mean-subtract pass (`mgmeans` on z) becomes a lane
   reduction accumulating r·z after subtracting; it replaces `dotTo(r, z)` both before the loop and
   inside it. Under D it is D's exit pass (§4.4.4).
3. **The fluid count is precomputed** (H-1); the sum reductions carry no count.
4. **The fused update iterates the inner box's rows** (to share the dot's order); ghost entries of
   x and r are no longer updated. The implementer confirms by grep that no reader consumes them
   (the wrap paths read wraps; `copyInner` reads inner cells); G-BIT is the proof.
5. **Device unchanged** (§2 caveat). A device fusion (≈ −0.5 ms GPU) would be a recorded device
   re-baseline: Q-P12, default no.
6. **Rejected:** folding α/β into the update/aypx kernels (−20 launches, ≈ 0.1 ms, but moves the
   breakdown flags); fusing aypx into the next matvec (redundant per-neighbour recomputation; device
   contraction makes "bitwise" unprovable).

Saving [model]: p·Ap −1.6, r·z −1.2, Σr −1.0, max −0.6, exit-mean lanes −0.5 → −4.5 (−3…−5.5);
−5 launches per iteration (≈ −50 per step).

---

## 7. F — momentum-solve fusion (bitwise, host and device)

**F-1. The VMG stop residual becomes a pure reduction** (`VelocityMG::solve`,
`src/mac_velocity_mg.hpp:762–777`). Today: `fill`, `bcApplyL0_`, `residualVarPin` writes `res`,
`zeroPlane` on the held normal-Dirichlet plane, `maxAbsInner` — per V-cycle — plus
`maxAbsDiffInner(rhs, res)` once per solve. New: after the same `fill` and `bcApplyL0_`, one
max-reduction kernel computes each cell's residual with the **verbatim** `residualVarPin` body,
excludes the held plane exactly where `zeroPlane` zeroes it, and returns `max|r|` (and on the first
call also `max|b|` and `max|b − r|`, the latter computed as `b(i) − r(i)` from the value it just
formed, as today). No write to `l0.res`. The V-cycle-internal residual is untouched. Max is
order-free: bitwise on host and device. Precondition: `l0.res` is not read after the stop test
(grep `lastResidualRatio` and the VMG debug paths). Saves the `res` write and read, the
`vmg_maxabs`, `vmg_maxabsdiff` and `zeroPlane` launches: ≈ −1.3 ms host, −0.1 GPU.

**F-2. `bc_vel`.** 47 launches, 2.3 ms [meas-S] against perf §14 H-3(e)'s acceptance of ≤ 30:
either H-3(e) missed call sites or the count differs. Attribute the 47 launches to call sites
(kernel-timer call-site names), then apply H-3(e)'s rule (both faces and back-to-back components of
one axis in one launch; different axes stay separate, in order) at the missed sites. Target
≤ 25 launches, ≤ 1.0 ms: ≈ −1.3 host, −0.2 GPU.

`ibm_build_diff_var` (3.5 ms against perf §14's 2.2 expected after H-3(c)) is not designed here:
WO-H4 measures why (Q-P11).

---

## 8. H — copies, zeros, rebuilds (bitwise)

- **H-1, all-fluid flag and fluid count (host only).** `setOpenness` computes per level
  `nFluid_L` (the count of `AC > 1e-30f`, the existing predicate) and `allFluid_L`. Host branches of
  `dot`/`dotTo`/`maxabs`/`maxabsTo`/`removeMean` skip the AC read when `allFluid_L`, and
  `removeMean` uses `nFluid_L` instead of reducing a count. Same cells, same order: bitwise on host;
  the device keeps its kernels (§2 caveat). Not-all-fluid levels keep the AC read (a 1-byte mask is
  not worth an array here).
- **H-2a, setOpenness copies.** `setOpenness` deep-copies the three coefficient fields into L0
  (`mac_cutcell_mg.hpp:1638–1640`), then fills ghosts and re-imposes boundary faces in place. Give
  `buildRhoCoeff`/`buildRhoCoeffOutflowFace` (and the porous builders) the MG's L0 `ox, oy, oz` as
  destinations through an accessor, and skip the copies — **only if** nothing else reads
  `cx1_/cy1_/cz1_` after `setOpenness` (if `projectCorrectVar` or a diagnostic reads them, keep the
  copies: the in-place ghost/boundary edits would leak). G-BIT decides.
- **H-2b, PCG staging.** In `solvePCGResident` drop `deep_copy(l0.x, x)` at entry and the exit round
  trip `l0.x ← x; removeMean(l0.x); x ← l0.x` (→ `removeMean(l0, x)`), after checking no reader of
  `lv_[0].x` between solves. Fuse the initial `p = z` copy into the first r·z reduction pass.
- **H-2c, the zero-guess matvec.** When the caller guarantees x0 = 0 (projectSolve's cold path),
  skip `matvec(Ap, x)` and `axpy(r, −1, Ap)`: `r = b`. Bitwise: A·0 = +0 everywhere and
  `b + (−1·(+0)) = b` for every b including ±0.
- **H-2d, VMG staging.** `VelocityMG::solve` copies `l0.rhs ← b`, `l0.x ← x`, `x ← l0.x`
  (`mac_velocity_mg.hpp:739–740`, end of solve): rebind `l0.rhs`/`l0.x` to the caller's Views for
  the solve and restore them (perf A5's pattern), after asserting the VMG V-cycle never writes
  `l0.rhs` and that b, x have the level-0 extent.
- **Not done:** fusing the first pre-smooth pass with the z = 0 memset in the FP64 V-cycle (≈ 0.2
  ms; bitwise only through a sign-of-zero argument `(b + 0.0)/ac`; D's entry pass subsumes it on
  eligible paths); fusing `buildRhoCoeff` with the level-0 face build (≤ 0.5 ms; the ghost fill and
  boundary re-imposition sit between them).

Saving [model]: copies −1.6 (11 → ≤ 2 per step), all-fluid −0.4, zero matvec −0.13 → −2.2
(−1.5…−2.5) host, −0.2 GPU.

---

## 9. S — host wrap kernels without per-cell wrap selects (conditional, bitwise)

`ccWrapNbrs` (`mac_cutcell_mg.hpp:≈290`) selects every neighbour index per cell. Inside the host
`omp simd` x-loop the y/z selects are loop-invariant (hoistable) but the x selects are not, so the
two x-neighbour loads become gathers. Variant: per row, compute the y/z offsets once; peel the first
and last cell of the row (scalar, today's body with `ccWrapNbrs`); run the interior with `i ± 1` —
affine, stride-2 loads. Same cells, same body, same operands, independent same-colour updates:
bitwise. Applies to the host branches of `cutcellSmoothColorFaceWrap`,
`residualCutcellFaceWrap`, `applyCutcellOpExactWrap` and (later) their FP32 siblings. **Go/no-go
from WO-P0:** land it if the peeled FP64 smoother pass is ≥ 1.2× faster in the microbenchmark;
otherwise drop it. Saving 0…−4 [fact]. It is measured before D so that D's gain is measured on top
of it.

---

## 10. F4 after all? No, not in this package

- perf §14's own trigger (≥ 1100 launches and a step > 78) is not met: 937 launches [meas-S].
- This package removes ≈ 100 launches (C −50, F −20, H −10, D −20), leaving ≈ 840.
- F4 (a persistent host team for L1, L2 and the bottom) would have to be written for the FP32
  V-cycle as well; doing it before D builds it twice.
- **Re-evaluation rule** (S-3): measure the empty-launch cost t_L at 24 threads (WO-P0). If
  `launches × t_L > 5 ms` at 1×24 after this package, F4 gets its own architect pass. Expected
  ≈ 840 × 6 µs ≈ 5 ms: borderline, so F4 is the likely next host item after G.

---

## 11. Recon defects, checked

- **(i) confirmed.** `solvePCGResident` stops on `rn < rtol * r0` (`mac_cutcell_mg.hpp:2017`),
  ignoring `stopRef_`, which the host loop (`:1754, :1811`), FCG (`:2098, :2145`) and Chebyshev
  (`:3863, :3875`) honour. Only `solveBalancedForceSystem` sets it (`flow_ibm_project.hpp:737`), so
  the single-rank balanced-force solve (collocated variable ρ, BFP on by default) stops relative to
  its warm residual instead of the full right-hand side, and differs in semantics from the
  distributed path. The staggered bubble column is unaffected. Fix = one line, a recorded change for
  that path only: WO-X1, isolated, default yes (Q-P7).
- **(ii) confirmed.** `pcgMaxit_ = 500`, `pcgRtol_ = 1e-10` (`flow_ibm.hpp:4846–4847`) against the
  binding's `set_pressure_pcg(on, max_iter=200, rtol=1e-8)` (`flow_bindings.cpp:2386–2387`): a
  script that calls `set_pressure_pcg(True)` without arguments gets 200/1e-8, one that never calls it
  gets 500/1e-10. API question, not numerics of this package: Q-P8, default no change here.

---

## 12. Work orders (for `opus-implementer`; one commit per bullet unless stated; stage named paths)

Branch/worktree: `suite/flow-projcost` (sibling convention). Every WO logs its numbers in
`doc/vof_step_performance_log.md` and updates `doc/vof_perf_STATE.md` at milestones.

- **WO-P0. Baselines and instruments** (no `src/` change).
  - Fix the G-BIT scripts in `~/Codes/bubble_column_perf/` (the `rc $?` after `$(basename …)`
    always prints 0): gate on the real exit status and on rebuilt-module timestamps.
  - On post-A(b) main: 50-step dumps from `ckpt_t43` at rtol 1e-8, 1e-7, 1e-10, 1e-9 (host 1×8,
    1×24; CUDA) → `N50_8` = diff(1e-8, 1e-7), `N50_10` = diff(1e-10, 1e-9); per-step iteration logs.
  - Empty `parallel_for` cost t_L at 24 threads (host).
  - Microbenchmark (extend `tests/kokkos/bench_rbgs.cpp`, label `bench`): one L0 colour pass at
    128×96×64 with the case's coefficients, OMP 24 and CUDA: (a) today's FP64 wrap pass, (b) the
    peeled FP64 pass (§9), (c) D-lite, (d) the FP32 flux pass (§4.4.3). Also single-thread cycles per
    updated cell for (a).
  - *Accept:* numbers logged; S go/no-go recorded (§9); D's expected host range re-stated from
    (d)/(a).
- **WO-H1** (§8 H-1). *Accept:* G-BIT (host and CUDA hashes unchanged — device untouched).
- **WO-H2a…d** (§8). *Accept:* G-BIT each; host deep copies ≤ 2 per step; if H-2a's precondition
  fails, record why and drop H-2a.
- **WO-H4.** Measure `ibm_build_diff_var` (call count, bytes, per-launch time, OMP 1 vs 24); report
  only.
- **WO-F1, WO-F2** (§7). *Accept:* G-BIT host and CUDA; `vmg_maxabs`, `vmg_maxabsdiff`, VMG
  `zeroPlane` launches 0 per step; `bc_vel` ≤ 25 launches and ≤ 1.0 ms (workstation A/B ratio
  ≤ 0.5).
- **WO-C0** (§6.1, recorded). *Accept:* G-NUM-P host; CUDA bitwise; host `state_hash` re-baselined
  in the commit with the old → new table; interleaved A/B: `mgdot` and `mgmeanr` per-launch time
  ≤ 0.5×.
- **WO-C1…C4** (§6). *Accept:* G-BIT against C0; −4 to −5 launches per PCG iteration.
- **WO-S** (§9, only on go). *Accept:* G-BIT host; CUDA untouched; A/B ratio of `cc_smooth` as
  measured in P0 ± 10 %.
- **WO-D0.** Extend `no_float_operator_casts` to also flag `<float>` template arguments and
  `class X = float` defaults; mark `BottomDirect<float>` and `FR = float` `// PRECISION-EXEMPT:`
  with their reason; add the single `VReal` alias line. *Accept:* the guard fails on a planted
  `BottomDirect<float>` without the marker; ctest green.
- **WO-D1. FP32 data and eligibility** (§4.4.1, §4.4.2, §4.4.6 with default `'fp64'`). New ctest
  `tests/kokkos/test_mg_fp32_vcycle.cpp` with G-D1. *Accept:* G-D1; G-BIT (default unchanged).
- **WO-D2. FP32 kernels** (§4.4.3, §4.4.4 items 1–4): smoother (host pencil + device MDRange),
  residual, restrict f→f and f→d, prolong f←f and f←d, FP32 `fill`/`applyNeumannGhost`, entry,
  exit. *Accept:* G-D1(e), G-D2 on the dense tool (`tests/study/mg_precond`, add a
  `--precision fp32` switch); G-BIT with `'fp64'`.
- **WO-D3. Integration** (§4.4.5) and promotion. Commit 1: wiring, default `'fp64'`; G-D3 and G-D4
  run with `'fp32'` forced. Commit 2 ("D default"): `'auto'`, register entry, re-baselined hashes
  with the table. *Accept:* G-NUM-P, G-D3, G-D4, and the perf gate: projection A/B ratio ≤ 0.88 on
  the 5080 and ≤ 0.88 on host (workstation 1×8, interleaved). If only the GPU passes, `'auto'`
  selects FP32 on device backends only (`kHostMemory` clause) and the log says why.
- **WO-D4c (contingent).** Only if G-D3 fails on iterations or the health ratio: a resident FCG
  (PCG + `r_{k+1}·z_k` fused into the C2 update kernel, β = (r·z_new − r·z_old)/rz). Recorded for
  `set_pressure_fcg` single rank. Then re-run G-D3 with FCG.
- **WO-D5. Memory** (§4.4.7). *Accept:* bitwise to WO-D3 with `'fp32'`; device memory per cell
  reported before/after on the case and a 256³ box; switching to `'fp64'` at run time works (test).
- **WO-E0. E behind `'zero'` default** (§5.3, all modes). *Accept:* G-BIT with `'zero'`; with
  `'projected'`: G-E1…E5, and the measurement: per-step iterations over 50 and 300 steps from
  `ckpt_t43` for history 1 and 2, logged.
- **WO-E0b (optional, Q-P6).** Reproduce Q11 at reduced size (a dense bed ≈ 64³, steady Stokes
  march, 400 steps) under {standalone V-cycle, Chebyshev, PCG} × {warmstart off, on}; ≤ 1 GPU-hour
  on the workstation. *Accept:* the table, and a register entry closing or narrowing Q11.
- **WO-E1. Promotion** if the 300-step mean saving is ≥ 0.5 iterations/step and G-NUM-P passes:
  `'auto'` default, register entry. Otherwise E stays opt-in and the log records ε_g.
- **WO-X1 (independent; Q-P7).** `solvePCGResident` stops on `rtol * (stopRef_ > 0 ? stopRef_ :
  rref)`. *Accept:* bitwise everywhere `stopRef_` is unset; collocated variable-ρ tests green; the
  balanced-force single-rank iteration counts logged before/after.
- **WO-S3. Snellius S-3** (after D3/E1; billed — the coordinator queues it): 1×24 contiguous and
  spread, znver4, rtol 1e-8, 300 steps, kprof; RTX 5080 timing. *Accept:* G-PERF-P logged, §16
  re-stated, the F4 rule of §10 evaluated.
- **WO-W. Wrap-up:** register entries (§14), state file, flow `CLAUDE.md` (pressure-solve section:
  the V-cycle precision; the initial guess), handoff.

---

## 13. Verification gates

**G-BIT** (bitwise WOs): perf §8 G-BIT unchanged — `state_hash.py` all cases + np2 identical
(host-openmp `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`, and nvidia-cuda); 50-step dump from
`ckpt_t43` bitwise for u v w p C with identical per-step iterations (host 1×8, 1×24, GPU); `ctest
-LE bench` green (231) on host and CUDA; `-R '_np[0-9]+$'` green; two runs bitwise. Host-only
changes: CUDA hashes unchanged.

**G-NUM-P** (recorded WOs: C0, D3, E1, X1):
1. 50-step max relative difference over u v w p C against the reference ≤ `N50_8` at rtol 1e-8 and
   ≤ `N50_10` at rtol 1e-10 (WO-P0), per backend.
2. Iterations: per step within ±1 and 50-step total within ±2 % (C0, D3); E1: per step
   ≤ reference + 1 and 300-step total ≤ reference.
3. `max_open_divergence_projected()` ≤ 2× the reference per step.
4. Static drop max|u| within 5 %; Hysing case 1 peak rise velocity and its time within 0.2 %
   (`tests/study/vof_surface_tension.py`), host and CUDA.
5. 2000-step bubble column: gas-volume drift ≤ 1e-11; time-averaged hold-up and mean rise velocity
   differ from the reference by less than the reference differs from its own rtol-1e-7 run.
6. `ctest -LE bench` green; hashes re-baselined in the commit with the old → new table.

**G-D1** (unit, `test_mg_fp32_vcycle`, every non-bottom level of the case, of `cyl` and of `pack`
from the probe at ratio 1e4):
(a) the FP32 residual of `x = c·1`, `b = 0`, is exactly 0 for c ∈ {1, −3.7, 1e5, 2⁻⁶⁰};
(b) `|uᵀÃv − vᵀÃu| ≤ 1e-6·|u|·|v|·max D` for 10 random pairs;
(c) `max_f |w̃_f/t_f − 1| ≤ 2⁻²⁴`, and the FP32 decoupled set equals the FP64 `AC < 1e-30` set;
(d) eligibility false on a distributed, outflow, overlay and odd-dimension configuration;
(e) host: one FP32 smoother pass equals a scalar plain-`float` reference of §4.4.3 bitwise;
(f) on each ineligible configuration `'auto'` is bitwise to `'fp64'`.

**G-D2** (preconditioner; dense tool n = 8 and 16, `periodic` and `wallz`, ratio 1, 1e2, 1e3,
1e4): (a) `B(2ᵏr) = 2ᵏB(r)` bitwise for k ∈ {−40, 0, 37}; (b) λ_min of sym(M_fp32) on the mean-free
subspace ≥ λ_min(sym(M_fp64)) − 1e-3·λ_max (no new indefiniteness); (c)
`‖M_fp32 − M_fp64‖_F/‖M_fp64‖_F ≤ 1e-4` at ratio ≤ 1e3 and ≤ 1e-3 at 1e4; (d) nonlinearity
`‖B(r1+r2) − Br1 − Br2‖₂/‖Br1‖₂ ≤ 1e-4` on the bubble column's L0.

**G-D3** (solver; `tests/study/vardensity_solver_probe.py` with `--drivers pcg,fcg`, geometries box
and pack, shapes slab and blob, sharp edges, ratios 1e2–1e4; `tests/study/precision_ab.py
contrast` (the RCP bed of register P1); the bubble column), `'fp32'` against `'fp64'` on the same
build:
(a) outer iterations per solve ≤ fp64 + max(1, 5 %); on the RCP bed at rtol 1e-8, 14/14/28 ± 1;
(b) attainable floor (rtol 1e-14, cap 300): `floor_fp32 ≤ 2 × floor_fp64`;
(c) at exit, the freshly recomputed `max|P(b − Ax)|/max|Pb| ≤ 1.5·rtol`;
(d) no capped solve where fp64 converges; `pressure_solve_failed()` never set;
(e) health ratio (traced): median ≤ 1e-2 per solve. Failing (a) or (e) triggers WO-D4c.

**G-D4** (case): G-NUM-P 1–6 for `'fp32'` against post-C main, host and CUDA.

**G-E** (WO-E0/E1, `'projected'`):
1. every step, traced: `‖x − x0‖_A ≤ ‖x‖_A` with x the converged solution (never worse than cold);
2. every step: freshly recomputed `max|P(b − Ax)|/rref ≤ 1.5·rtol`;
3. steps with histCount_ = 0 (step 1, after `set_field`) bitwise to `'zero'`;
4. an IBM case: x0 = 0 on every non-fluid cell;
5. G-NUM-P 3–5;
6. a steady march (`cyl`, Stokes, 400 steps, PCG): final permeability within 1e-6 of `'zero'`.

**G-PERF-P** (S-3, Snellius 1×24 contiguous, rtol 1e-8, znver4, post-A(b); a miss is reported,
not reverted):

| quantity | gate | expected |
|---|---|---|
| step | ≤ 66 | 61 |
| projection | ≤ 31 | 26 |
| momentum | ≤ 8.5 | 7.3 |
| launches/step | ≤ 900 | ≈ 840 |
| RTX 5080 projection | ≤ 15.5 | ≈ 14 |
| RTX 5080 step | ≤ 34.5 | ≈ 33 |

**Serial ↔ parallel:** D and E are single-rank by eligibility; distributed branches are
byte-identical in source except templated shared helpers whose double instantiation is unchanged;
the np-tests (`vardensity_mpi`, `telescope_mpi`, VoF block MPI) are bitwise to before for every
WO except C0, whose host order change applies at every np (np-tests at their tolerances).

---

## 14. Register entries this note creates (for the caller to add when each lands)

1. **FP32 V-cycle preconditioner below the exact FP64 PCG** (D, default `'auto'` on eligible
   single-rank PCG/FCG). Rejected: float band storage with a stored/independently rounded diagonal
   (u·κ on near-null modes); FP32 faces with FP64 arithmetic (device FP64-bound); a stored FP64
   diagonal (8 B/cell for a derivable value); FP32 coarse levels only; BF16 (not now). Evidence: §4.2,
   G-D1–G-D4.
2. **Reduced-precision operators are stored as face weights; the diagonal is derived, never stored,
   and FP32 arithmetic uses the flux/correction form** (the precision-policy rule's corollary).
   Rejected: the AC form in FP32.
3. **The FP32 V-cycle's input is scaled by an exact power of two** (2⁻ᵉ, e = ilogb max|r|).
4. **The velocity MG stays FP64** (stationary iteration; the level-0 smoother defines the fixed
   point).
5. **Initial guess = A-norm projection onto the last two increments; stop relative to the zero-guess
   residual; Krylov drivers only** (E). Rejected: x0 = φⁿ with an r0-relative stop; unprojected
   2φⁿ − φⁿ⁻¹; Fischer L > 2 under per-step operator rebuilds; warm starts under Chebyshev, the
   standalone V-cycle or BiCGStab.
6. **Q11 bounded:** under PCG/FCG with a double operator a warm start cannot amplify (T1) and an
   r0-relative stop only under-solves boundedly (T2); the registered divergence needs a non-Krylov
   driver or a float outer operator. (Updated with WO-E0b's table if run.)
7. **Host Krylov sum reductions in CutcellMG use 4 x-lanes per row** (Q-H4 triggered by S-1).
8. **Host Krylov reduction fusion** (C1–C4), host only; device fusion rejected for bitwise reasons.
9. **All-fluid level flag and precomputed fluid count** (host).
10. **The VMG stop residual is a pure max-reduction.**
11. **F4 deferred** with the §10 re-evaluation rule.
12. **The resident PCG honours `stopRef_`** (if WO-X1 lands).

---

## 15. Open questions and risks (each has a default)

- **Q-P1 [pref] D as the default.** It re-opens the precision question the user closed on
  2026-09-11, but only below the exact outer operator, which that decision did not touch.
  *Default:* yes, `'auto'`, after G-NUM-P and G-D1–G-D4 pass; one isolated commit ("D default");
  revert = set the default back to `'fp64'`.
- **Q-P2 [fact] The host smoother's bound** (§3.2). *Default:* WO-P0's microbenchmark sets D's host
  range; the host default follows WO-D3's A/B rule (device-only `'auto'` if the host gains < 12 %).
- **Q-P3 [pref/scope] Distributed D and E.** *Default:* not in this package; eligibility keeps the
  distributed path unchanged. Needed for 6×4/8×3 later (`GridHalo<float>`, CA ring, telescope
  stages, GraphAMG input).
- **Q-P4 [fact] E's saving on the bubble column (ε_g).** *Default:* WO-E0 measures; WO-E1's rule
  (≥ 0.5 iterations/step) decides the default.
- **Q-P5 [pref] The public `set_pressure_warmstart`.** *Default:* untouched (it disables E); propose
  its deprecation in favour of E in the next API cycle.
- **Q-P6 [fact] Q11's actual mechanism.** *Default:* run WO-E0b only if the porous-scaling bed and
  script are available locally (≤ 1 GPU-hour); otherwise record the bound (§5.1) alone.
- **Q-P7 [pref] Defect (i).** *Default:* fix in WO-X1, isolated, recorded for the collocated
  balanced-force single-rank path only.
- **Q-P8 [pref] Defect (ii), the PCG default mismatch.** *Default:* no change in this package; an
  open item for the naming/API register.
- **Q-P9 [pref] Restart reproducibility under E.** A restart without the two increments starts cold
  for two steps (results agree to the solver tolerance, not bitwise with an uninterrupted run).
  *Default:* document it; do not add checkpoint fields.
- **Q-P10 [fact] F4.** *Default:* §10's rule after S-3.
- **Q-P11 [fact] `ibm_build_diff_var`.** *Default:* WO-H4 measures; a design only if it exceeds
  2 ms after diagnosis.
- **Q-P12 [pref] Device fusion of the Krylov reductions** (≈ −0.5 ms GPU, a device re-baseline).
  *Default:* no.
- **Q-P13 [pref] Setter names** (`set_pressure_vcycle_precision`, `set_pressure_initial_guess`, on
  `s.diagnostics`). *Default:* these names; check them against `../docs/NAMING.md` in WO-D1.
- **R-1.** D's host gain may be at the low end if the smoother is gather-bound (S first, §9).
- **R-2.** E's gain may be small if curvature noise dominates φ's step-to-step change.
- **R-3.** C0 changes every host pressure result at the 1e-15 level; the host hashes move once.
- **R-4.** FP32 at extreme contrast and size (κ ≳ 10¹¹, δ ≈ 2·10⁻²) may cost iterations; G-D3 at
  1e4 is the evidence boundary, and FCG (WO-D4c) the fallback.
- **R-5.** H-2a may be impossible (aliasing of the coefficient fields); it is then dropped (−0.4).

---

## 16. End state, honestly [model]

1×24 contiguous genoa, rtol 1e-8, ms/step:

| stage | S-1 [meas-S] | after A(b) | after this package | with G (−5.7 target) |
|---|---|---|---|---|
| projection | 55.3 | ≈ 40.4 | ≈ 26 (22–31) | ≈ 26 |
| momentum | 10.6 | 10.6 | ≈ 7.3 | ≈ 7.3 |
| curvature | 9.7 | 9.7 | 9.7 | ≈ 4 |
| block advect, predictor, debris, CSF, outside timers | 17.5 | 17.5 | 17.5 | 17.5 |
| **step** | **93.1** | **≈ 78** | **≈ 61 (55–66)** | **≈ 55 (49–61)** |
| TBFsolver, same node | 45.5–46.0 | | | |

This package with A(b) and G gives ≈ 1.2× TBFsolver at 1×24 contiguous (≈ 50 ms at spread
placement, the measured −10 %). It does not reach parity. What remains, largest first: block advect
(7.6), launch overhead (≈ 5 ms; F4), `ibm_build_diff_var` (3.5), the unavoidable PCG iterations
(≈ 8–9 × ≈ 1.9 ms). RTX 5080: ≈ 36.5 → ≈ 33 ms (D −2.5, E −0.6, F −0.3, H −0.2).
