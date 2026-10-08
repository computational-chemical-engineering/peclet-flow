# Design G — the cost of block-VoF curvature (CPU and GPU)

Architect design note, 2026-10-08. Brief: `doc/vof_curvature_cost_brief.md`. Campaign state:
`doc/vof_perf_STATE.md`. Sibling package D (the projection) is `doc/vof_projection_cost_design.md`;
nothing here touches the pressure solve.

**Decision in five lines.** The cascade is already HF-first, and the cells that reach tier 3 are the
cells where a height function is *less accurate* than the PV fit. Measured below, longer columns
make the max error worse. So the number of tier-3 cells stays as it is, and what changes is the cost
of each one. Of tier 3's time, 62 % goes to rebuilding every neighbour's PLIC polygon about 16 times
per pass, and another 29 % to integrating it again in each target's frame. The fix: build each
cell's polygon once per pass and cache it (bitwise), then cache the polygon's 3-D area moments and
transform them per target (round-off only, a recorded change). On the GPU the same kernel is FP64-issue-bound,
because of serial one-lane accumulation and a sparse 125-slot map. The fix there is neighbour
compaction plus entry-parallel accumulation in canonical order, which is bitwise.

| | host tier 3, genoa 1×24 | curvature stage, genoa | GPU tier 3, RTX 5080 | numerics |
|---|---|---|---|---|
| today (measured) | ≈ 6.9 ms | 9.69 ms | 4.90 ms | — |
| WO-2: polygon cache + support prefilter + lower-triangle accumulation; device compaction + entry-parallel accumulation; host fused list pass | ≈ 2.2 ms | ≈ 4.8 ms | ≈ 1.3 ms | **bitwise** |
| WO-3: moment cache replaces polygon cache | ≈ 1.4 ms | ≈ 3.9–4.2 ms | ≈ 0.55 ms | **recorded** (max rel. Δκ 7.7e-15) |

The genoa and GPU figures after a change are *estimates*: single-thread harness ratios, and an
FP64-issue model anchored on the measured 4.90 ms (§2.5). The gates in §7 measure them.

---

## 1. Problem and scope

**In scope.** Tier 3 of the curvature cascade (the PV paraboloid fit), its launch shape on host and
device, and the per-cell data it consumes. This covers all three paths that run tier 3: the batched block container
(`vofCurvListPass` / `vofCurvFallbackTeams`, `src/vof/block_batch.hpp`), the per-block and
structured `VofCurvature` (`fallbackPass` / `fallbackBatch`, `src/vof/curvature_field.hpp`), and
the container-free kernels in `core/include/peclet/core/vof/curvature.hpp`. Also in scope is the
question of how many cells reach tier 3.

**Out of scope.** The pressure solve (package D), PLIC/advection, the CSF force formula
(`csfFaceCurvature` stays the arithmetic mean of two defined cell values), contact angles, phase
change, and the interfacial-area driver `src/vof/interface_area_field.hpp`. That driver keeps its own
`pvFitAdd` walk unchanged; see §9 R4. Also out: the curvature-stage kernels outside the cascade
(compaction scans, zero sweeps, CSF force), about 2.5 ms on genoa (§9 Q2); accuracy improvements to κ; and any change to tier
1/2 or to the cascade order.

---

## 2. Measured breakdown (2026-10-08)

The data and tools are kept outside git in `~/Codes/bubble_column_perf/curv_cost/` (README there):
`census.py` (prof.py plus a census and block dump), `harness.cpp` (replays flow's cell bodies),
the outputs, and the kernel tables.

### 2.1 Tier census, bubble column (`ckpt_t43`, 16 markers, D/h ≈ 16)
From `diagnostics.vof_block_curvature_stats()` after 1 warm-up and 2 steps, host build
`flow-main-base/build_omp` (5795fb0; the curvature math is identical on main):

| interfacial | HF (tier 1) | HF mixed (2a) | HF fit (2b) | PV | PV reduced | none | clipped |
|---|---|---|---|---|---|---|---|
| 19 961 | 12 992 (65.1 %) | 35 (0.18 %) | 0 (off) | **6 934 (34.7 %)** | 0 | 0 | 0 |

The brief's "≈ 35 %" is confirmed.

### 2.2 Tier-3 work per target (harness replay, the same blocks, 6 955 targets)
Per target: **44.8** interfacial cells in the 5³ stencil. All 44.8 pass `cosMin` (0.00 rejected) and
all get a polygon built (4.00 vertices on average). **12.2 (27 %) are then discarded** because their
vertex average lies beyond the Wendland support d = 2.5, which leaves 32.6 accepted terms. In one
pass, 311 k polygons are built from **19 988 distinct interfacial cells, about 15.6 times each**,
plus one more per target for its own frame.

### 2.3 Tier-3 cost split (single thread, Threadripper PRO 5965WX, `-O3 -march=native -ffp-contract=off`, best of 15, min over 3 runs)

| piece | ms | share of tier 3 |
|---|---|---|
| tier-3 pass (all 16 blocks) | 72.7 | 100 % |
| `plicPolygon` inside it | 45.8 | **63 %** |
| projection into the target frame + Green's-theorem moments + weight | ≈ 22 | **30 %** |
| accumulate + LDLᵀ solve | 1.74 | 2.4 % |
| frame (`curvFallbackFrame`) | 1.18 | 1.6 % |
| the 125-offset interface scan | 0.71 | 1.0 % |
| *for comparison:* HF pass (tiers 1-2, all interfacial) | 2.67 | — |
| *for comparison:* PLIC planes over the grown list | 4.15 | — |

The accumulation is not the cost: accumulate and solve together are 2.4 %.

**Variants, measured in the same harness against the replay** (bitwise means a `memcmp` of κ on all 6 955 targets):

| variant | tier-3 ms | + cache build | total | vs today | κ vs today |
|---|---|---|---|---|---|
| V1 polygon cache | 25.2 | 3.3 | 28.5 | 2.6× | **bitwise** |
| V2 = V1 + lower-triangle accumulation | 23.6 | 3.3 | 26.9 | 2.7× | **bitwise** |
| **V5 = V2 + conservative support prefilter** | **19.5** | 3.3 | **22.8** | **3.2×** | **bitwise** |
| V3 moment cache + lower triangle | 10.1 | 4.0 | 14.1 | 5.2× | max rel 7.67e-15 |
| **V6 = V3 + origin from the cached moments** | **10.6** | 4.0 | **14.6** | **5.0×** | **max rel 7.20e-15** |

Runs on the loaded workstation vary by ±15 % in absolute terms. The ratios are stable.

### 2.4 Genoa (Snellius S-1, flow 38e80e6, 1×24; stage timers over 30 steps, kernel timer 20-step difference)
The curvature stage is 9.69 ms of a 93 ms step. `vof::block::batch_curv_list` takes **7.17 ms** over 4
launches (HF, tier 3, clip, census). Applying the harness proportions (tier 3 : HF = 72.7 : 2.67)
gives **≈ 6.9 ms of tier 3 and ≈ 0.25 ms of HF**. The other ≈ 2.5 ms of the stage is compaction,
zeroing, planes, HF reset and the CSF face force. The local summary does not itemize it (§9 Q2).
Projection factor: **0.095 genoa-ms per single-thread harness ms**. It assumes the change does
not alter parallel efficiency.

### 2.5 GPU (RTX 5080, `flow-main-base/build_cuda134` = 5795fb0, nsys, 20-step difference)
The step is 29.3 ms with 26.7 ms of kernel time. **`vof::vofCurvFallbackTeams` = 4.90 ms/step, the single
largest kernel of the GPU step (18 %).** HF, clip and census together are 0.10 ms. (The design
target of WO-7 was ≤ 1.5 ms.)

**Why it is 4.90 ms.** A GB203 runs FP64 at 1/64 of its FP32 rate, about 2 FP64 lanes per SM, so a
warp's FP64 instruction holds the SM's FP64 pipe for about 16 cycles, *whatever its active mask*.
The cost is therefore set by the number of FP64 warp instructions:
- the term map runs ⌈125/32⌉ = 4 rounds of `pvFitTerm` (≈ 1 750 FP64 instructions each, six FP64
  `atan2` among them), although only about 45 of the 125 slots are interfacial;
- each lane recomputes the frame, including the target's own polygon (≈ 1 000);
- **one lane accumulates 32.6 terms × 42 entries × 2 = ≈ 2 500 serial FP64 instructions**;
- one lane runs the solve (≈ 500).

That is about 10 k instructions per target × 6 934 targets / 84 SMs × 16 cycles ≈ 4.5 ms, which
matches the measurement. Section 5.4 uses this model.

### 2.6 Accuracy facts that decide the options (gate B sphere, replayed by the harness)
The harness reproduces gate B exactly. Today's cascade gives L1 3.032e-2 / 5.941e-3 / 1.324e-3
and max 5.007e-2 / 1.325e-2 / 3.795e-3 at N = 16 / 32 / 64 (R = 0.3), for orders L1 2.26 and
max 1.86.

- **PV-only** (gate F): L1 3.616e-2 / 9.271e-3 / 2.326e-3, max **identical** to the cascade. The
  max error sits on cells that both routes send to PV. HF is better than PV on average only
  where the slope is mild.
- **Longer HF columns.** These are the Popinet/TBFsolver variable-length walks, emulated with
  `hfColumnHeight(col, nh)`. Each cell keeps its 7-column result when it has one; nh applies only to
  cells that fail at 7.

| column | tier-3 cells at 16/32/64 | L1 at 64 | max at 32 / 64 | max order |
|---|---|---|---|---|
| 7 (today) | 248 / 536 / 1352 | 1.324e-3 | 1.325e-2 / **3.795e-3** | **1.86** |
| 9 | 200 / 224 / 344 | 1.117e-3 | 1.778e-2 / 6.296e-3 | 1.50 |
| 11 | 200 / 128 / 56 | 1.202e-3 | 1.778e-2 / 1.406e-2 | 0.92 |

  On the bubble column, 9-cell columns would rescue 42.4 % of the tier-3 cells and 11-cell columns
  60.7 %. On the rescued cells, κ_HF − κ_PV has an rms of 1.85e-2 and 2.0e-2 against an rms κ of
  0.18–0.21, which moves those cells by about 10 %. **The cells HF cannot close are exactly the
  steep-slope cells where a height function is inaccurate.** This is the same structural verdict as
  tier 2b, now measured for column length.

### 2.7 The SOTA yardstick (TBFsolver, `tbf_builds/tbf/src/VOF/VOF.f90`)
`computeBlockCurvature` is Popinet's 2009 generalised HF, run per block under OpenMP:
- HF in up to three directions, with **unbounded** column walks bounded only by the block
  (`hfColumn`);
- then a paraboloid fit through the HF interface *positions* (`parabFittedCurvature(...,fromHF)`,
  which is peclet's tier 2b);
- then a fit through PLIC centroids (`interCentroids`);
- `spreadCurvature` is disabled.

Its fallback is therefore the combination peclet measured and rejected on max-error convergence
(§2.6, and the 2b note at `curvature_field.hpp:464-493`). That makes it cheaper per cell, since there
are no polygon moments, and less accurate in the max norm. It is the yardstick for cost, not for
accuracy. Its per-step curvature cost has not been measured (§9 Q3).

---

## 3. Constraints and invariants

- **Mathematics fixed.** The HF cascade and PV fallback (5³, Wendland C2 with d = 2.5 cells
  scaled by `metric.maxH()`, `cosMin` 0.2, a 6-parameter model and the reduced {1, x², y²} model
  below 6 polygons, LDLᵀ with pivTol 1e-12) are unchanged. So are the tier order, the tolerances
  (`interfaceEps`, `pureEps`, `monoTol`), the 7-cell column, the block-only |κ| clip, and the
  meaning of κ = 2H in 1/hRef, positive for a convex liquid blob.
- **Bitwise classes.** "Bitwise" means host build vs host baseline AND CUDA build vs CUDA baseline,
  each on its own backend (G-BIT). Gate E stays as it is: bitwise on a host backend, ≤ 1e-11
  across backends. Gate C1 (batched == per-block, bitwise) must hold after **every** work order,
  so the batched path and `VofCurvature` always change together.
- **Canonical order.** On every backend each normal-equation entry is summed over the stencil
  terms in increasing canonical offset index k = ((oz+2)·5 + (oy+2))·5 + (ox+2), starting from 0.0
  and with the expression `a += w * s[i] * s[j]` (`b[i] += w * s[i] * B`). A warp reduction
  stays rejected (register).
- **Host FP.** `-ffp-contract=off` on host backends; nvcc keeps `--fmad=true`.
- **MPI.** Every new quantity is a pure function of one cell's PLIC plane and the block metric,
  consumed inside the existing ±3 reach. No halo, exchange or reduction is added, so np-bitwise
  holds by construction on the structured path and the block path alike.
- **Residency.** No host↔device transfer and no new host read of a device count (WO-8 holds the
  container to ≤ 3 host reads). Caches are device Views written and read inside one chunk.
- **Placement.** Container-free math goes in `core` (one tag, then a flow pin bump). Existing core
  functions keep their bits (`pvFitAdd`, `pvFitTerm`, `pvFitAccum`, `pvFitSolve`,
  `plicPolygon`), because other containers and amr may call them. Everything new is additive.
- **No env var** and no new public API. Tunables keep flowing through `copyTunablesFrom` and
  none is added.

---

## 4. Options, ranked

| rank | option | host saving (genoa, est.) | GPU saving (5080, est.) | accuracy risk | class | verdict |
|---|---|---|---|---|---|---|
| 1 | **Moment cache (V6)**: per-cell 3-D area moments of the PLIC polygon, built once in the planes pass and transformed per target | 6.9 → 1.4 ms | with #3: 4.90 → ≈ 0.55 ms | none (same integral; 7.7e-15) | recorded, round-off | **DO (WO-3)** |
| 2 | **Polygon cache + support prefilter + lower triangle (V5)** | 6.9 → 2.2 ms | with #3: → ≈ 1.3 ms | none | **bitwise** | **DO (WO-2a)**, superseded in payload by #1 |
| 3 | **Device: neighbour compaction + entry-parallel canonical accumulation** | — | 4.90 → ≈ 2.5 ms alone | none | **bitwise** | **DO (WO-2b)** |
| 4 | **Host: one fused list pass** (HF → tier 3 → clip → census per entry) | ≈ 0.1–0.3 ms (3 fork/joins + fences) | — | none | **bitwise** | **DO (WO-2c)** |
| — | longer HF columns (9/11, Popinet/TBF walks) | tier-3 count −42 % / −61 % | similar | **measured: max order 1.86 → 1.50 / 0.92; max at 64³ ×1.7 / ×3.7** | recorded | **REJECT** |
| — | tier 2b point fit (reopen) | large | large | measured: max order 0.00 | recorded | REJECT (standing) |
| — | centroid point fit (Popinet 2009 last resort, TBF `interCentroids`) | ≈ rank 1 | ≈ rank 1 | lower: fits facet centroids, whose O(κh²) sag varies with each cut | recorded | **REJECT**: no cost gain over V6 (centroid = m1/a is in the cache) |
| — | smaller support (S = 3 or d < 2.5) | ≈ 0.4 ms after V6 | ≈ 0.15 ms | Han et al.'s measured optimum is d = 2.5; gates B/F would move | recorded | REJECT now (§9 Q5) |
| — | FP32 fit | — | large on consumer GPUs | ill-conditioned 6×6 under a 1e-12 pivot guard | recorded | REJECT (precision policy) |
| — | warp-shuffle reduction of the normal equations | — | ≈ like #3 | none | not bitwise | REJECT (register) |
| — | thread-per-target on device | — | ≈ like #1+#3 | none | bitwise | REJECT: reverses the registered team decision, high register pressure, divergent loops |
| — | build caches only near tier-3 targets (planes → HF → mark → build) | ≈ 0.08 ms | ≈ 0 | none | bitwise | REJECT: an extra pass and ordering complexity for < 0.1 ms |

**Why rank 1 over rank 2 as the destination.** V5 still evaluates Green's theorem on each
projected polygon for every pair: about 6 edges × 5 true divisions (`/6`, `/12`, `/3`) plus two
slope divisions. V6 replaces the whole per-pair geometry with about 70 flops and one square root.
It is the same integral:

- The projected polygon's moments are the polygon's own 3-D moments contracted with the target
  frame and scaled by the projection factor `n̂_j · n̂_t`.
- B, the integral of the polygon's plane height over the projected polygon, is the first moment
  dotted with the target normal.

The only difference is the rounding sequence. V6 is also what makes the GPU cheap, because FP64
instructions are the GPU's currency. V5 is built first anyway: it proves the cache plumbing
**bitwise** before any number moves (§6). Without that step a plumbing bug would hide inside a
"recorded" difference.

**Why longer columns are wrong here, from first principles.** In the preferred direction a
height function has slopes |h_x|, |h_y| ≤ 1. Its second-difference truncation error grows with
slope like (1 + 5s²)(1 + s²) along a 2-D section (relative error (h²/12)·h''''/h''). That is a
factor of 12 between unit slope and a flat patch, and the octant-diagonal cells of a sphere carry the
largest slopes the preferred direction admits. The 7-cell column fails precisely when the corner
columns need more than ±3 cells, roughly when (|n₁| + |n₂|)/|n_d| ≳ 1.6. A longer column therefore
admits exactly the steepest, least accurate HF cells. §2.6 measures the consequence.

---

## 5. The design

### 5.1 New container-free kernels (core, `peclet/core/vof/curvature.hpp`, namespace `peclet::core::vof`)

All are `KOKKOS_INLINE_FUNCTION`, take scalars and small arrays only, and follow the existing
"unit-metric overload" convention where the existing functions have one.

**(a) A bitwise split of `pvFitTerm`.**
```
// the normal half: n2 > 0 check, mi = m / g.h, invn = 1/sqrt(|mi|^2), np = (mi.t1, mi.t2, mi.nn)*invn,
// cosMin guard — verbatim from pvFitTerm. Returns false where pvFitTerm returns false before plicPolygon.
bool pvTermNormal(double mx, double my, double mz, const double t1[3], const double t2[3],
                  const double nn[3], double cosMin, const VofMetric& g, double np[3]);
// the polygon half: everything pvFitTerm does after `if (nv < 3) return false;`, verbatim
// (vertex transform with toPhys, px/py/zc, polygonMoments2d, the |s0| > 1e-14 test, b0/b1/b2,
// r, wendlandWeight, B), given the polygon. Sets t.ok and returns it.
bool pvTermPolygon(PvTerm& t, const double np[3], const double v[8][3], int nv,
                   const double off[3], const double org[3], const double t1[3],
                   const double t2[3], const double nn[3], double dW, const VofMetric& g);
// pvFitTerm becomes: t.ok = false; if (!pvTermNormal(..)) return false; v[8][3]; nv = plicPolygon(..);
// if (nv < 3) return false; return pvTermPolygon(..);   -- same arithmetic, same order.
```

**(b) The V5 cache entry and its prefilter.**
```
struct alignas(32) PvPolygon {   // 224 B
  double v[8][3];   // plicPolygon's output, cell-local [0,1]^3, exactly as returned (slots >= nv unused)
  double vbar[3];   // (sum_k v[k]) * (1.0/nv) - 0.5 : cell-centred INDEX-space vertex average (prefilter only)
  int nv;           // plicPolygon's return value (0 if the normal is degenerate); vbar = 0 when nv == 0
};
void pvPolygonBuild(double mx, double my, double mz, double alpha, PvPolygon& P);
// true iff the polygon is CERTAINLY outside the Wendland support, so pvTermPolygon would reject it
// with w == 0. org is in index units, as pvFitTerm's.
//   if (P.nv < 3) return false;   (the polygon half rejects it on its own)
//   X = g.toPhys({vbar + off - org});  return X.X > dW*dW*(1.0 + 1e-9);
bool pvOutsideSupport(const PvPolygon& P, const double off[3], const double org[3], double dW,
                      const VofMetric& g);
```
*Why the prefilter is bitwise.* pvTermPolygon's r is the norm of the frame projections of the mean
of X_k = toPhys(off + v_k − 0.5 − org). The prefilter's norm is the same vector computed in a
different order. The two differ by a few ulp, about 1e-15 relative. The margin (1 + 1e-9) on r² is 5e-10
on r, so every skipped polygon has r_orig > dW, which gives q ≥ 1 and w = 0, and pvTermPolygon would
have rejected it. The accepted set is unchanged, and so are npoly and every accumulated byte.

**(c) Lower-triangle accumulation.**
```
// pvFitAccum restricted to j <= i: for i in 0..5 { b[i] += w*s[i]*B; for j in 0..i A[i][j] += w*s[i]*s[j]; } ++npoly.
// The A[i][j] for j > i stay at pvFitInit's 0.0.
void pvFitAccumLower(PvFit& f, const PvTerm& t);
// entry e in [0,27): e < 21 -> (i, j) with e = i(i+1)/2 + j, j <= i:  a += t.w * t.s[i] * t.s[j];
//                    e >= 21 -> b[e-21]:                              a += t.w * t.s[e-21] * t.B;
void pvFitAccumEntry(double& a, const PvTerm& t, int e);
// fill a PvFit from 27 entries (+ npoly); upper triangle zero
void pvFitFromEntries(PvFit& f, const double ent[27], int npoly);
```
*Why this is bitwise.* `curvSolveSym` reads only the diagonal and the strict lower triangle of
its input after the Jacobi scaling (the scaling statement scales the upper half, but nothing reads
it). The reduced model's `Ar` copies `A[3][0]`, `A[5][0]`, `A[5][3]`, also lower. The harness
confirms this (V2 bitwise). Each entry is the same sequential sum from 0.0 in the same order, so a
per-entry loop equals pvFitAccum's interleaved loop bit for bit.

**(d) The V6 cache entry: polygon area moments in physical, cell-centred coordinates.**
```
struct alignas(64) PvMoments {   // 16 doubles = 128 B
  double n[3];    // unit PHYSICAL normal: vofPhysNormalInv(m, g, n); (0,0,0) if degenerate
  double c[3];    // physical vertex average, cell-centred: (1/nv) sum_k Y_k        [Wendland point]
  double a;       // area (physical); 0 when nv < 3 or the normal is degenerate
  double m1[3];   // int_P Y dA
  double m2[6];   // int_P Y Y^T dA in the order xx, yy, zz, xy, xz, yz
};
void pvMomentsBuild(double mx, double my, double mz, double alpha, const VofMetric& g, PvMoments& P);
```
`pvMomentsBuild`:
1. Zero all 16 entries.
2. If `vofPhysNormalInv(m, g, n) <= 0`, return.
3. Set `nv = plicPolygon(m, alpha, v)`; if nv < 3, return with a = 0 and n kept.
4. Set Y_k = g.toPhys(v_k − 0.5) and c = (Σ_k Y_k)·(1/nv).
5. Fan from Y_0. For k = 1 … nv−2, with (A, B, C) = (Y_0, Y_k, Y_{k+1}):
   - cr = (B−A)×(C−A), At = 0.5·|cr|, S = A + B + C;
   - a += At; m1 += At·S/3;
   - m2_pq += (At/12)·(A_p A_q + B_p B_q + C_p C_q + S_p S_q).

   These are the exact triangle moments (∫λ_pλ_q dA = A(1+δ_pq)/12). The polygon is convex, so
   every At ≥ 0.
```
bool pvFrameMoments(const PvMoments& P, const VofMetric& g, double nn[3], double t1[3],
                    double t2[3], double org[3]);
//   if (!(n.n > 0)) return false;  nn = P.n;  curvFrame(nn, t1, t2);
//   org = (P.a > 0) ? P.m1 / P.a : g.toPhys({-0.5,-0.5,-0.5});   // PHYSICAL, target-centred
//   (the second branch is today's degenerate ctr = 0, org = -0.5; it is unreachable when n != 0)
bool pvTermMoments(PvTerm& t, const PvMoments& P, const double off[3], const double org[3],
                   const double t1[3], const double t2[3], const double nn[3], double dW,
                   double cosMin, const VofMetric& g);
```
`pvTermMoments`, in this order:
1. Set t.ok = false and cj = P.n · nn; reject if `!(cj > cosMin)` (this also rejects n = 0).
2. Reject if `!(P.a > 0)`.
3. Set δ = g.toPhys(off) − org, r = |P.c + δ| and w = wendlandWeight(r, dW); reject if `!(w > 0)`.
4. Set Q1 = m1 + δ·a.
5. Set Q2 = m2 + δ m1ᵀ + m1 δᵀ + δδᵀ·a, with diagonal entries `m2_pp + 2 δ_p m1_p + δ_p δ_p a` and
   off-diagonal entries `m2_pq + δ_p m1_q + m1_p δ_q + δ_p δ_q a`.
6. Set u = Q2·t1 and v = Q2·t2. Then:
   s0 = cj·a, s1 = cj·(t1·Q1), s2 = cj·(t2·Q1), s3 = cj·(t1·u), s4 = cj·(t2·u), s5 = cj·(t2·v).
7. Reject if `!(|s0| > 1e-14)`.
8. Set B = cj·(nn·Q1), t = {w, B, s}, ok.

*Equivalence.* For the polygon P_j, with X = Y + δ relative to the target origin, the projected
polygon's moments are ∫_{P'} φ(x', y') dA' = (n̂_j·n̂_t) ∫_{P} φ(t1·X, t2·X) dA. Two facts make this
hold: projecting a planar region scales its area by the cosine, and the projected polygon is the
polygon of the projected vertices. Also, ∫_{P'} z' dA' = (n̂_j·n̂_t) n̂_t·∫_P X dA, because X lies on
the plane. Today's B = b0 s0 + b1 s1 + b2 s2 is that same integral. In V0 the Wendland point is the
vertex average of X_k, which is P.c + δ here. On an anisotropic metric the area centroid maps
linearly under H, so the physical org = H·(index org).

### 5.2 Ownership, layout and lifetime of the cache

- **Slot map.** `VofCurvature` gets `slot_`, a `Kokkos::View<int*, SMem>` over its extended block
  (the same extent as `mx_`, allocated in `init()`). Within the grown region,
  `slot(i) >= 0` ⟺ `vofIsInterface(c(i), interfaceEps)`. The value is i's position in the list
  that owns the cache. Outside the grown region `slot` is never read.
- **Caches.** Entries are AoS, one per grown interfacial cell, in grown-list order. 128 B
  (`PvMoments`) or 224 B (`PvPolygon`), aligned, so one cell's entry is one contiguous read. That
  is ideal when one host thread or one device lane consumes one neighbour.
  - `VofCurvature::cache_` is sized to `listG_`'s capacity, allocated with it, and used by its
    own `compute()` / `fallbackBatch`.
  - On the batched block path the chunk-level `VofBlockSet::cCacheG_` is sized and reallocated
    exactly like `cListG_` (capacity = Σ of the chunk's grown-region volumes, because the counts
    stay on the device). It is passed in `VofCurvTable` as one pointer; positions are the absolute
    `cListG_` positions q.
- **Memory.** The batched capacity on the column is 473 k entries: 60.5 MB (V6), 106 MB (V5,
  transitional), plus 4 B/cell for `slot_`. Only the 20 k used entries are ever touched, so a host
  never maps the rest. On the GPU the allocation is physical; §9 R3 gives the scaling.
- **Lifetime.**
  1. `planes_zero` writes `slot = -1` over the grown region together with its four plane zeros.
  2. `planes_list` (one thread per grown-list entry q, cell i) runs `wyReconstructCell`, then
     builds the entry from the plane it just wrote (`pvPolygonBuild` or `pvMomentsBuild(…, J.gm)`)
     into `cache(q)`, and sets `slot(i) = q`.
  3. Tier 3 of the same chunk reads the cache.
  4. Nothing reads it afterwards.

  The cast to int is guarded by a host check that the list capacity is < 2³¹.
- **Dense mode** (`useWorklist = false`, the compaction oracle). There is no list and no slot. The
  tier-3 body builds each neighbour's entry on the fly from `(mx, my, mz, al)(j)` with the same
  build function, and the interface test stays `vofIsInterface(c(j))`. The body is one template over an
  *entry accessor*: `Cached{slot, cache}` or `OnTheFly{mx, my, mz, al, c, ieps, g}`. Both return
  bit-identical entries because both run the same function on the same inputs.

### 5.3 Host tier-3 body (thread per target; `curvFallbackCell`'s replacement)
```
if (br(i) >= 0) return;
frame:  V5: P_i = entry(i); polygonAreaCentroid(P_i.v, P_i.nv) -> org = ctr - 0.5 (index units);
            nn from vofPhysNormalInv(m(i)); curvFrame                        [== curvFallbackFrame, bitwise]
        V6: pvFrameMoments(entry(i), g, nn, t1, t2, org)                     [org physical]
        not framed -> kappa 0, kCurvNoEstimate (as today)
PvFit f; pvFitInit(f);
for k in canonical order (oz, oy, ox in [-2,2]^3, ox fastest):
    j = i + ox + oy*sy + oz*sz;  if (!present(j)) continue;            // slot(j) >= 0, or vofIsInterface
    V5: if (pvOutsideSupport(entry(j), off, org, dW, g)) continue;
        PvTerm t; if (pvTermNormal(m(j), t1,t2,nn, cmin, g, np) &&
                      entry(j).nv >= 3 && pvTermPolygon(t, np, entry(j).v, entry(j).nv, off, org, t1,t2,nn, dW, g))
            pvFitAccumLower(f, t);
    V6: PvTerm t; if (pvTermMoments(t, entry(j), off, org, t1,t2,nn, dW, cmin, g)) pvFitAccumLower(f, t);
curvFallbackStore(i, f, kap, br);                                      // unchanged
```
(In V5 the `nv >= 3` test sits where pvFitTerm has it: after the normal test, before the
polygon half.)

### 5.4 Device tier-3 body (`curvFallbackTeam`'s replacement; one warp-team per target, persistent league unchanged)
```
if (br(i) >= 0) return;                         // uniform, as today
all lanes: frame exactly as 5.3 (deterministic, no broadcast)
if framed:
  (1) parallel_scan(TeamThreadRange(tm, 125), k -> present(j_k) ? 1 : 0)   // canonical order kept
        final: list[p] = slot(j_k) (or j_k in dense mode), kidx[p] = k;      n = total
  barrier
  (2) parallel_for(TeamThreadRange(tm, n), p -> term[p] = TERM(entry(list[p]), off(kidx[p]), ...))
        TERM = the V5 or V6 sequence of 5.3, writing ok = false where 5.3 would `continue`
  barrier
  (3) parallel_for(TeamThreadRange(tm, 27), e -> { double a = 0.0;
          for p in 0..n-1: if (term[p].ok) pvFitAccumEntry(a, term[p], e);  ent[e] = a; })
  barrier
single lane: not framed -> kCurvNoEstimate; else npoly = #ok over term[0..n);
             pvFitFromEntries(f, ent, npoly); curvFallbackStore(i, f, kap, br)
barrier (existing, before the next entry of the persistent stride)
```
Team scratch: `term` PvTerm[125] (72 B each), `list` int[125], `kidx` int[125], `ent` double[27],
about 10.2 KB. Today the kernel uses 9 KB, so occupancy is unchanged. `term[p]` reads in (3) are
broadcasts (every lane reads the same address), so there are no bank conflicts. HIP (64 lanes)
runs 2 scan rounds, and 27 entries fit. The same body serves `vofCurvFallbackTeams` (batched) and
`VofCurvature::fallbackBatch`'s team path.

*Model estimate (FP64 warp instructions per target, §2.5):*
- today: 4 × 1 750 + 1 000 + 2 500 + 500 ≈ 10 000;
- WO-2b alone: 2 × 1 750 + 1 000 + 100 + 500 ≈ 5 100, about 2.5 ms;
- WO-2a + 2b (V5 term about 750, frame from cache about 200): about 2 300, about 1.25 ms;
- WO-3 (V6 term about 160, frame about 60): about 980, **about 0.55 ms**, after which the
  single-lane solve is about half of what remains (§9 Q4).

### 5.5 Host list launches (batched path, host backends only)
`vofCurvListPass` on a host executes ONE `vofHostListFor` over the inner list (dynamic schedule,
chunk 16, as H-4(a)). Per entry, it calls in order:
1. `curvHeightCell`;
2. if `br(i) < 0`, the 5.3 body;
3. the clip, if `J.km > 0`, with today's expression;
4. the census atomics.

This is bitwise: the tier-3 body of cell i reads only its own `br(i)`, written by step 1 of the
same entry, plus planes, colour and cache, which were all written before the launch. The clip
touches only `kap(i)`, and the census uses integer atomics, which are order-free. Device backends
keep the four launches (tier 3 runs as teams). The CSF force stays a separate launch because it
reads neighbours' κ.

### 5.6 What does not change
- Tiers 1-2 and their 7-cell columns.
- The cascade order and every tolerance.
- The clip.
- The census layout and its single host read.
- `csfFaceCurvature`.
- The interface-area driver.
- The existing core functions' bits: `pvFitTerm`'s composition is gated bitwise against a frozen
  copy (WO-1).
- MPI.
- The structured path's API.

---

## 6. Work orders (dependency order; each is one commit unless stated)

Every flow WO is done in a sibling worktree `suite/flow-curvcost` (flow CLAUDE.md), and stages named
paths only. "Bitwise" WOs must pass G-BIT. WO-3 must pass G-NUM. Every WO reports G-PERF (a miss is
reported, not blocking).

**WO-0: baseline (measurement only, no commit to code).**
- On the current flow main, record the following:
  - the G-BIT reference set (state_hash 12 + np2, the 50-step `prof.py --dump` at host 1×8 and on
    the GPU), or reuse the S-1 merge baselines if main has not moved in `src/vof`;
  - the curvature census (§2.1);
  - `prof.py --timing` plus kernel tables for the workstation (1×16 host, at load average < 2)
    and the 5080.
- Rebuild `~/Codes/bubble_column_perf/curv_cost/harness.cpp` against the main tree and confirm
  that it reproduces §2.3's split within ±15 %.
- **Accept:** the numbers are written to `doc/vof_step_performance_log.md`.

**WO-1: core additions (core repo; then tag, then flow pin bump in WO-2a's commit).**
- Add §5.1 (a)–(d) to `curvature.hpp`, with doc comments that carry the equivalence arguments above.
- Tests in `core/tests/test_vof_pvfit.cpp`, or a new `test_vof_pvcache.cpp` registered the same
  way. Use the existing random-case generator (planted rejections, random metrics) and run on the
  default execution space and on the host space:
  - **T1.** The refactored `pvFitTerm` equals a frozen verbatim copy of today's `pvFitTerm`,
    bitwise (t.ok, and w, B, s[6] when ok), over 10⁵ cases.
  - **T2 (cache pattern).** Kernel 1 runs `pvPolygonBuild` per stencil cell into a View; kernel 2
    evaluates `pvOutsideSupport` → `pvTermNormal` → `pvTermPolygon` from that View. The result must
    equal T1's reference bitwise. Report the number of prefilter skips that the reference accepted;
    it **must be 0**. The prefilter must skip ≥ 50 % of the planted outside-support cases.
  - **T3.** `pvFitAccumLower`, and the 27-lane `pvFitAccumEntry` team kernel, against
    `pvFitAccum`: lower triangle, diagonal, b and npoly all bitwise. `pvFitSolve` output on both
    must be bitwise for full-rank systems and for systems forced rank-deficient (npoly 3–5, reduced
    model).
  - **T4 (V6 vs today).** Over 10⁵ cases: the accept flags agree on every case that is not within
    1e-9 relative of a threshold (r = dW, s0 = 1e-14, cj = cosMin); report the threshold count. For
    accepted cases with |off| ≤ 2, with s0 as the projected area:
    - |Δs_k| ≤ 1e-13·(|s_k| + s0·3.5^{deg k}), where deg = 0, 1, 1, 2, 2, 2;
    - |ΔB| ≤ 1e-13·(|B| + 3.5 s0);
    - |Δw| ≤ 1e-14.

    `pvFrameMoments` against `curvFallbackFrame`: |Δorg| ≤ 1e-14 (physical) and the frame vectors
    are bitwise.
  - **T5.** Analytic moments.
    - m = (0,0,1), α = 0.5: a = 1, m1 = 0, m2 = diag(1/12, 1/12, 0) to 1e-15.
    - The same plane on the metric h = (2, 1, 0.5): a = 2, m2_xx = 2/3 and m2_yy = 1/6, both to
      1e-15.
    - A tilted plane against a fine triangulated quadrature of the polygon to 1e-12.
- **Accept:** the T1–T5 core ctests pass on host-openmp and CUDA; core CI is green; a tag is cut
  per `docs/RELEASE.md` (additive, so a minor version).

**WO-2a: flow polygon cache, support prefilter, lower-triangle accumulation (bitwise). Pin bump.**
- Add `slot_` to `VofCurvature` and `slot` to `VofCurvJob` and `VofCurvFallbackJob`.
- `VofCurvature::cache_` holds `PvPolygon`; `VofBlockSet::cCacheG_` likewise.
- The planes kernels follow §5.2: both `vofCurvPlanesZero`/`vofCurvPlanesList` and
  `VofCurvature::reconstructPlanes` (worklist mode). Use the host row form of H-4(b) for the zero
  sweep.
- The tier-3 bodies follow §5.3 (host) and, for now, today's team shape with `TERM` from the cache
  on the device. The entry-accessor template covers the dense mode.
- **Accept:**
  - G-BIT (host 1×8, host contract-off and `-march=native` trees, CUDA), bitwise;
  - gates C1 and E pass as today;
  - the full battery is as on main (231/231 host and CUDA at the S-1 merge, with GPU-OOM flukes
    rerun serially);
  - G-PERF: host tier-3 work ÷ ≥ 2.5 (harness or kernel timer, same machine); the GPU fallback is
    measured and reported.

**WO-2b: device team body (bitwise).**
- Replace `curvFallbackTeam` with the §5.4 body: compaction, the term map over n, entry-parallel
  accumulation, and the single-lane solve.
- **Accept:** CUDA G-BIT bitwise, host untouched (host hashes identical), and core T3's team kernel
  proven in WO-1. G-PERF on the 5080: `vofCurvFallbackTeams` ≤ 1.6 ms (from 4.90).

**WO-2c: host fused list pass (bitwise).**
- Apply §5.5 to the batched host path.
- **Accept:** host G-BIT bitwise; host `batch_curv_list` launches per chunk drop from 4 to 1;
  G-PERF is reported.

**WO-3: the moment cache (recorded numerics change).**
- Swap the payload from `PvPolygon` to `PvMoments` on every path (batched, `VofCurvature`, dense
  accessor) in ONE commit:
  - `pvMomentsBuild` goes into the planes kernels;
  - `pvFrameMoments` and `pvTermMoments` go into the 5.3 and 5.4 bodies;
  - `pvOutsideSupport` and the `PvPolygon` code are deleted.
- The commit carries the state_hash re-baseline with an old → new table and the register entries
  (§8).
- **Accept:** G-NUM items 1–5, plus:
  - `test_vof_curvature` gates B, C, D, F, G, H, I and the anisotropic gate print **identical
    numbers to all printed digits** against WO-2 (κ moves at about 1e-14), with every CHECK
    unchanged;
  - C1 bitwise (batched == per-block);
  - E as today;
  - `test_vof_surface_tension` P1–P7 pass unchanged;
  - the study gates `tests/study/vof_surface_tension.py` (static, wave, lamb, hysing1, hysing2)
    give numbers equal to WO-2's within 1e-10 relative;
  - the bubble-column census at the checkpoint: pv and pv_reduced counts within ±2 cells of WO-2,
    and max |κ_WO3 − κ_WO2| / max |κ| ≤ 1e-12 over all blocks (`vof_block_kappa`).
- G-PERF:
  - host tier-3 work ÷ ≥ 1.4 against WO-2 (harness 22.8 → 14.6 ms);
  - GPU `vofCurvFallbackTeams` ≤ 0.8 ms;
  - genoa curvature stage ≤ 4.5 ms (target ≤ 4.0), measured in the next scheduled Snellius S-2
    run (§9 Q6).

**WO-4: record.**
- Update `doc/vof_perf_STATE.md` and `doc/vof_step_performance_log.md`.
- Add the register entries in `../docs/decisions/flow.md` (and the `DECISIONS.md` index line) if
  WO-3 did not already.
- Update the flow CLAUDE.md VoF paragraph only if a prohibition must be inlined (the longer-column
  rejection, one line).

Commit isolation: WO-3 is the only commit that moves a bit. Reverting it restores the WO-2 state
exactly, apart from the new `pvMoments*` functions, which can stay in core.

---

## 7. Verification gates (consolidated)

| gate | invariant | how | tolerance | configurations | WOs |
|---|---|---|---|---|---|
| core T1–T3 | split, cache pattern and lower or entry accumulation equal today's | 10⁵ random cases | bitwise; prefilter false-skips = 0 | host-openmp + CUDA (default + host space) | 1 |
| core T4–T5 | moment terms equal the Green's-theorem terms; analytic moments | random + analytic | §6 WO-1 numbers | host + CUDA | 1 |
| G-BIT | state unchanged | state_hash 12 + np2; 50-step dump `u, v, w, p, C` + block colours; per-step pressure iterations | bitwise | host 1×8, host contract-off and `-march=native`, CUDA; np 1/2/4 block MPI ctests | 2a, 2b, 2c |
| C1 | batched == per-block curvature + CSF | `test_vof_blocks` C1 | bitwise | host + CUDA | all |
| E | device kernel vs serial host oracle | `test_vof_curvature` E | host bitwise; CUDA ≤ 1e-11 | host + CUDA | all |
| accuracy | convergence order and max error unchanged | gates B (oL1 > 1.7; today 2.26 / max 1.86), C (D/dx 3–40, noEstimate = 0), F, G, H, I, anisotropic | WO-2 bitwise; WO-3 identical printed digits | host + CUDA | all |
| G-NUM | recorded change within noise | §8 of `vof_step_performance_design.md`: 50-step diff ≤ N50_rtol; iterations ±1 and ±2 %; divergence ≤ 2×; static drop within 5 %; Hysing 1 within 0.2 % | as stated | host + CUDA | 3 |
| census | tier counts unchanged | `vof_block_curvature_stats()` at `ckpt_t43` | WO-2 equal; WO-3 ±2 cells | host + CUDA | all |
| serial↔parallel | np-independence | block VoF MPI ctests np 1/2/4; state_hash np2 | bitwise (WO-2), re-baselined consistently (WO-3) | host + CUDA | all |
| transfers | no new D↔H traffic | nsys 20 steps (Transfer gate, design §8) | no memcpy ≥ 1 KiB in `step()`; small reads unchanged | CUDA | 2a, 2b, 3 |
| G-PERF | cost | harness split; kernel timer / nsys; S-2 | the targets in §6 | workstation, 5080, genoa | all |

A falsifiable limiting case is built into T5 (an exact plane). The harness's 0-mismatch HF replay
(`hfCell(nh = 7)` == `curvHeightCell`) and its exact reproduction of gate B are the evidence that the
measurements in §2 are the shipped algorithm's numbers.

---

## 8. Register entries to add (`suite/docs/decisions/flow.md`, area flow; index lines in `DECISIONS.md`)

1. **Tier-3 PV fit: each interfacial cell's PLIC polygon data is built ONCE per curvature pass (a
   per-cell cache filled in the planes pass, indexed by a slot map), never once per target.**
   - Rejected: rebuilding the polygon per target (15.6 builds per cell per pass, 63 % of tier 3);
     a dense per-cell cache over the extended block (2.25 GB at 256³ on the structured path).
   - Evidence: this note §2.3. Decided 2026-10-08 (WO-2a commit).
2. **Tier-3 PV terms come from per-cell 3-D polygon area moments (`PvMoments`), transformed into
   each target's frame. A RECORDED numerics change at round-off (max rel Δκ 7.7e-15).**
   - Rejected: per-target Green's-theorem moments of the projected polygon (2.6× slower here and
     FP64-heavy on GPUs); a centroid point fit (no cost advantage once the centroid is cached;
     less accurate); a smaller support.
   - Supersedes nothing. Evidence §2.3 and §4; WO-3 commit with its hash table.
3. **Height-function columns stay 7 cells; longer, Popinet/TBFsolver-style column walks are
   REJECTED.**
   - Measured on gate B: max-error order 1.86 → 1.50 (9 cells) and 0.92 (11 cells); max at 64³
     3.8e-3 → 6.3e-3 and 1.4e-2. The cells only a long column closes are the steep-slope cells
     where HF is inaccurate.
   - Rejected alternative: "fewer tier-3 cells via longer columns". Evidence §2.6.
4. **Device tier 3: neighbour compaction plus ENTRY-PARALLEL canonical accumulation (each of the
   27 lower or b entries summed serially in canonical order by its own lane), bitwise.**
   - Amends "PV curvature fallback: one team per target cell, canonical-order accumulation" (its
     team shape and canonical order stand). The one-lane accumulation is retired.
   - Rejected: one lane accumulating all 42 entries (about 2 500 serial FP64 instructions per
     target; FP64-issue-bound on consumer GPUs, 4.90 ms on the 5080); thread-per-target.
5. **Only the lower triangle and diagonal of the PV normal equations is accumulated
   (`curvSolveSym` reads nothing else). Bitwise.**
   - Rejected: the full 6×6 accumulation (dead work).

---

## 9. Risks and open questions (each with a recommended default)

- **Q1 (user's preference): accept WO-3's recorded round-off change?**
  - The mathematics is identical, κ moves by ≤ 7.7e-15 relative, and it buys about 0.8 ms on genoa
    plus about 0.7 ms on the 5080 beyond the bitwise WO-2. Precedent: H-1 'direct' bottom was
    recorded and kept.
  - **Default: yes, proceed.** WO-3 is one isolated commit; a revert restores WO-2 bit for bit.
- **Q2 (fact): what is the 2.5 ms of the curvature stage outside the list passes on genoa?**
  - It covers compaction scans, zero sweeps, the HF reset, the CSF face force and fences.
    Locally, before H-4, at 16 threads: compaction 1.42, CSF force 0.50, planes zero 0.42,
    planes 0.41 ms.
  - After WO-3 it is the larger part of the stage, so the ≲ 4 ms target hinges on it.
  - **Default:** itemize it from the S-1 kprof `.dat` on Snellius (no new run). If it exceeds
    2 ms after WO-3, open a follow-up (fold the slot/plane zeroing into the compaction scan, merge
    the HF reset into the fused pass). It is not designed here.
- **Q3 (fact): TBFsolver's own curvature cost per step.**
  - It is not needed for this decision, which rests on peclet's own cost structure and accuracy.
  - **Default:** do not measure. If the gallery cost table wants a per-stage comparison, wrap
    `computeCurvature` in its existing `info` timer in the next TBF run.
- **Q4 (fact): is a lane-parallel LDLᵀ solve on the device worth it?**
  - After WO-3 the single-lane solve is about half of the GPU tier 3 (≈ 0.25 ms).
  - **Default:** not built. Build it only if WO-3's G-PERF on the 5080 exceeds 0.8 ms, with each
    lane computing the same expressions (bitwise) and its own design check.
- **Q5 (user's preference): revisit the support (d = 2.5, S = 5) for cost?**
  - **Default: no.** Accuracy comes first, and after WO-3 the saving is ≈ 0.4 ms host and
    ≈ 0.15 ms GPU.
- **Q6 (user's preference; billed hours): genoa G-PERF for WO-2 and WO-3.**
  - **Default:** do not queue a dedicated job. Fold the measurement into the already-planned S-2
    run (STATE "Next action" 2) and use the workstation plus the 5080 for acceptance meanwhile.
- **R1 (fact; risk): nvcc contraction in the entry-parallel and cached forms.**
  - The equality rests on nvcc compiling `a += w*s[i]*s[j]` and the inlined `plicPolygon` the same
    way in the planes kernel and in the team kernel. Core T2 and T3 prove it on CUDA before any
    flow change.
  - **Default:** if T2 or T3 fails on CUDA, STOP and escalate. Do not add `__fmul_rn` or similar.
- **R2 (fact): HIP is untestable locally.**
  - **Default:** compile-only, with kVofWarp 64 handled by the same body.
- **R3 (fact): batched cache memory on the GPU.**
  - The capacity is the chunk's grown-region volume, because the counts stay on the device. At
    V6's 128 B: 60 MB on the column; about 163 MB for 16 blocks at D/h 32; about 0.86 GB at
    D/h 64. That is roughly equal to the 16 blocks' existing field memory.
  - **Default:** accept. If a case runs short of memory, add the grown count to the census read
    (no extra read) and size the next step's cache from it with a device overflow flag. That is a
    separate design.
- **R4 (fact): the interface-area driver (phase change) keeps `pvFitAdd`.**
  - Tier selection (HF vs PV) stays identical. After WO-3, PV vs PV-reduced can differ from
    curvature only through a threshold flip (measure-zero; census gate ±2).
  - **Default:** leave it untouched (out of scope). Migrating it to the cache is a later
    performance item.
- **R5 (fact): anisotropic metrics.**
  - V6 works in physical coordinates throughout (§5.1d). Covered by T4 (random metrics), T5 and
    `test_vof_curvature`'s anisotropic gate.
  - **Default:** as designed. Any T4 failure on an anisotropic case blocks WO-3.
- **R6 (fact): AMR or other containers calling `pvFitAdd`/`pvFitTerm`** are bit-unchanged (T1).
  **Default:** no action.
