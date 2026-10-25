# MG-depth policy — measurement log (append-only)

Companion of `doc/mg_depth_policy.md`. Dated entries, newest last; never rewritten.

## 2026-10-24/25 — WO-0: the depth matrix on the unmodified base (flow 246541f)

**Modules** (MPI builds of 246541f, unmodified): CUDA
`/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/ref/mod_base_cuda` (RTX 5080, Kokkos CUDA +
OpenMP host), host `/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/ref/mod_base_host`
(OpenMP, znver3). Raw output (logs, per-run JSON, probe logs, the M3s dumps):
`/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/wo0/`.

**Command** (one session, 2026-10-24 22:08 → 2026-10-25 04:35, each row under one hold of
`gpu_diag/mgd/heavy.lock`):

```bash
M=/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd
tests/study/mg_depth/run_matrix.sh $M/ref/mod_base_cuda $M/ref/mod_base_host $M/wo0 \
    M6 M1 M2 M3z M3r M3s M4 M5 M7a M7b M7c M8
python tests/study/mg_depth/summarize.py $M/wo0 <row> [--md]      # the table below
```

Protocol: per row, one 2-step probe per depth with `PECLET_FLOW_MG_DEBUG=1
PECLET_FLOW_AGMG_DEBUG=1` (debug printing only; read for the bottom engine: `[mg] direct bottom`
→ direct, `[agmg-build]` → GraphAMG, neither → smoothed), then 3 interleaved rounds
(L_a L_b … L_a L_b …), 5 warm-up + 30 timed steps (M2: 10 warm-up), ms/step = mean of the timed
`step()` walls, median over rounds. `OMP_NUM_THREADS=4 OMP_PROC_BIND=false` (M5: 6; M8: 2 per
rank). Load average ≤ 17.2 on every run (all judged); no other GPU process during any timed run
(nvidia-smi before/after). Every (row, depth) state hash is identical across its 3 rounds.
Explicit depth = `set_pressure_multigrid(True, L)` before the geometry; L = 0 = no call.

**Case definitions** (`tests/study/mg_depth/mgd_case.py`; the choices the note does not fix are
marked *measurement choice*):
- M1 / M7a swarm 192³: `weak_swarm.py --single --cells 192` code for code (case A, d/h 20,
  2³ bubbles); M1 MG-PCG cap 800 rtol 1e-8 (as weak_swarm), M7a no driver call (Chebyshev).
- M2 bubble column 256×192×128 = `run_peclet.build()` at `BUBBLE_COLUMN_CELLS_PER_D=32`;
  *measurement choice*: FRESH start from the seeded spheres at rest (the t43 checkpoint exists only
  at 128×96×64), 10 warm-up steps; MG-PCG rtol 1e-8 as built, device superficial-velocity constraint.
- M3 `zh_sphere` / `random_spheres`: `sdflow_regression.py`'s geometry, CFG and march (stop test,
  ≤ 400 steps); ms/step over the whole march.
- M4 / M8 channel 2048×64×64: *measurement choice* (the §2.7 case's setup is not recorded):
  periodic x, walls ±y ±z, μ 0.05, dt 1, body force 1e-5, plus a fixed-seed non-solenoidal
  8-mode perturbation of amplitude 1e-2 at global coordinates (without it the channel is
  unidirectional and the projection is trivial: smoke run, div 6.6e-21). Default driver.
- M6 anchored 128×32×32: *measurement choice* (the 2026-08 sweep's setup is not recorded):
  uniform inflow U 0.05 at −x, outflow +x, walls ±y ±z, μ 0.1, dt 1, MG-PCG cap 300 rtol 1e-12
  (the identical budget at every depth).
- M7b porous bed 128³: *measurement choice*: fixed bed ε = 0.45 ± 0.1 (smoothed fixed-seed noise)
  for x < 0.6 N, 1 above (4-cell tanh), `enable_drag` + `set_porous_continuity(True)`,
  drag_beta = 150 μ (1−ε)²/(ε d_p²) with d_p 4, advection + implicit advection ON, periodic,
  body force 1e-4 along x; default driver. M7c the same without `enable_drag`.

**Results** (iterations: round 1, identical in all rounds; div = max over the timed steps of
`max_open_divergence_projected()`; FP32 restarts = the solver's stderr warning, summed over rounds;
failed = `pressure_solve_failed()` steps):

| row | L | levels | bottom (cells) | engine | ms/step median (rounds) | iters mean [min,max] | div max | FP32 restarts | failed | load max | extra |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 4 | 4 | 24x24x24 (13824) | GraphAMG | 155.57 (155.6, 155.9, 155.6) | 8.00 [8,8] | 7.10e-11 | 0 | 0 | 6.3 |  |
| M1 | 5 | 5 | 12x12x12 (1728) | direct | 81.12 (81.2, 81.1, 81.0) | 9.00 [9,9] | 4.23e-11 | 0 | 0 | 6.0 |  |
| M1 | 6 | 6 | 6x6x6 (216) | direct | 79.36 (79.4, 79.3, 79.4) | 9.00 [9,9] | 5.77e-11 | 0 | 0 | 5.6 |  |
| M1 | 7 | 7 | 3x3x3 (27) | smoothed | 79.70 (79.6, 79.7, 79.7) | 9.00 [9,9] | 5.80e-11 | 0 | 0 | 5.2 |  |
| M2 | 4 | 4 | 32x24x16 (12288) | GraphAMG | 203.31 (203.0, 203.3, 205.4) | 10.03 [9,12] | 6.56e-10 | 0 | 0 | 3.7 |  |
| M2 | 5 | 5 | 16x12x8 (1536) | direct | 91.25 (91.3, 91.3, 91.2) | 11.70 [11,13] | 4.00e-10 | 0 | 0 | 3.5 |  |
| M2 | 6 | 6 | 8x6x4 (192) | direct | 97.22 (97.2, 97.3, 97.2) | 13.77 [12,16] | 7.27e-10 | 0 | 0 | 3.3 |  |
| M3z | 4 | 4 | 16x16x16 (4096) | GraphAMG | 65.00 (65.0, 65.2, 64.9) | 7.00 [7,8] | 4.42e-11 | 0 | 0 | 9.2 | K=7.554169743 steps=400 itmed2=7 umaxsolid=0.000e+00 |
| M3z | 5 | 5 | 8x8x8 (512) | direct | 52.80 (52.8, 52.9, 52.8) | 7.00 [7,8] | 4.21e-11 | 0 | 0 | 8.2 | K=7.554169743 steps=400 itmed2=7 umaxsolid=0.000e+00 |
| M3z | 6 | 6 | 4x4x4 (64) | smoothed | 52.59 (52.6, 52.6, 52.6) | 7.01 [7,8] | 6.49e-11 | 0 | 0 | 8.2 | K=7.554169743 steps=400 itmed2=7 umaxsolid=0.000e+00 |
| M3r | 4 | 4 | 16x16x16 (4096) | GraphAMG | 65.12 (65.1, 65.1, 65.1) | 7.09 [7,9] | 5.86e-11 | 0 | 0 | 3.4 | k_star=0.006245475062 steps=340 itmed2=7 umaxsolid=0.000e+00 |
| M3r | 5 | 5 | 8x8x8 (512) | direct | 51.10 (51.0, 51.1, 51.1) | 7.09 [7,8] | 1.00e-10 | 0 | 0 | 3.0 | k_star=0.006245475062 steps=340 itmed2=7 umaxsolid=0.000e+00 |
| M3r | 6 | 6 | 4x4x4 (64) | smoothed | 50.84 (50.8, 50.8, 50.8) | 7.11 [7,8] | 1.58e-10 | 0 | 0 | 2.5 | k_star=0.006245475062 steps=340 itmed2=7 umaxsolid=0.000e+00 |
| M3s_0 | 0 | 4 | 6x6x6 (216) | direct | 8.24 (8.3, 8.2, 8.2) | 6.51 [6,7] | 1.28e-10 | 0 | 0 | 8.5 | K=7.436078194 steps=150 itmed2=6 umaxsolid=0.000e+00 |
| M3s_1 | 0 | 4 | 6x6x6 (216) | direct | 9.54 (9.5, 9.6, 9.5) | 7.32 [7,8] | 1.53e-10 | 0 | 0 | 8.2 | k_star=0.006282103323 steps=60 itmed2=7 umaxsolid=0.000e+00 |
| M4 | 4 | 4 | 256x8x8 (16384) | GraphAMG | 175.91 (175.8, 176.1, 175.9) | 8.00 [8,8] | 4.44e-16 | 0 | 0 | 6.5 |  |
| M4 | 5 | 5 | 128x4x4 (2048) | direct | 109.11 (109.1, 109.1, 109.1) | 8.00 [8,8] | 7.10e-16 | 0 | 0 | 5.6 |  |
| M4 | 6 | 6 | 64x2x2 (256) | direct | 107.25 (107.3, 107.2, 107.2) | 8.00 [8,8] | 7.45e-16 | 0 | 0 | 4.9 |  |
| M5 | 4 | 4 | 16x16x16 (4096) | GraphAMG | 607.26 (606.4, 607.3, 608.6) | 7.90 [7,8] | 1.41e-10 | 0 | 0 | 17.2 |  |
| M5 | 5 | 5 | 8x8x8 (512) | direct | 615.09 (617.5, 614.6, 615.1) | 8.90 [8,9] | 1.94e-10 | 0 | 0 | 14.0 |  |
| M5 | 6 | 6 | 4x4x4 (64) | smoothed | 617.27 (617.4, 616.5, 617.3) | 9.00 [9,9] | 6.67e-11 | 0 | 0 | 11.3 |  |
| M6 | 4 | 4 | 16x4x4 (256) | smoothed | 16.02 (16.0, 16.1, 15.9) | 29.53 [28,31] | 1.11e-16 | 0 | 0 | 8.2 |  |
| M6 | 5 | 5 | 8x2x2 (32) | smoothed | 14.88 (14.8, 14.9, 14.9) | 23.00 [23,23] | 3.81e-17 | 0 | 0 | 7.6 |  |
| M6 | 6 | 6 | 4x2x2 (16) | smoothed | 59.08 (59.1, 58.8, 59.2) | 90.37 [60,113] | 1.02e-16 | 0 | 0 | 7.6 |  |
| M7a | 4 | 4 | 24x24x24 (13824) | GraphAMG | 358.07 (357.8, 358.1, 358.2) | 11.00 [11,11] | 2.72e-12 | 0 | 0 | 3.6 |  |
| M7a | 5 | 5 | 12x12x12 (1728) | direct | 225.72 (225.6, 225.8, 225.7) | 11.97 [11,12] | 1.89e-11 | 0 | 0 | 3.2 |  |
| M7a | 6 | 6 | 6x6x6 (216) | direct | 214.83 (214.8, 214.7, 214.8) | 12.00 [12,12] | 6.48e-12 | 0 | 0 | 2.9 |  |
| M7b | 4 | 4 | 16x16x16 (4096) | GraphAMG | 54.55 (54.5, 54.6, 54.6) | 9.00 [9,9] | 2.69e-04 | 0 | 0 | 1.6 |  |
| M7b | 5 | 5 | 8x8x8 (512) | direct | 30.06 (30.1, 30.1, 30.1) | 9.00 [9,9] | 2.69e-04 | 0 | 0 | 1.6 |  |
| M7b | 6 | 6 | 4x4x4 (64) | direct | 28.92 (28.9, 28.9, 28.9) | 9.00 [9,9] | 2.69e-04 | 0 | 0 | 1.7 |  |
| M7c | 4 | 4 | 16x16x16 (4096) | GraphAMG | 101.62 (101.4, 101.7, 101.6) | 9.00 [9,9] | 4.29e-03 | 0 | 0 | 1.3 |  |
| M7c | 5 | 5 | 8x8x8 (512) | direct | 60.53 (60.5, 60.5, 60.5) | 9.00 [9,9] | 4.29e-03 | 0 | 0 | 1.4 |  |
| M7c | 6 | 6 | 4x4x4 (64) | smoothed | 59.03 (59.0, 59.0, 59.0) | 9.00 [9,9] | 4.29e-03 | 0 | 0 | 1.4 |  |
| M8 | 4 | 4 | 256x8x8 (16384) | GraphAMG | 282.14 (282.4, 282.1, 282.1) | 8.00 [8,8] | 4.44e-16 | 0 | 0 | 2.0 |  |
| M8 | 5 | 5 | 128x4x4 (2048) | GraphAMG | 245.70 (245.9, 245.4, 245.7) | 8.00 [8,8] | 7.10e-16 | 0 | 0 | 2.0 |  |
| M8 | 6 | 6 | 64x2x2 (256) | GraphAMG | 266.85 (266.8, 266.8, 267.2) | 8.00 [8,8] | 7.45e-16 | 0 | 0 | 2.0 |  |

M7b / M7c: `max_open_divergence_projected()` is not the porous constraint (2.7e-4 / 4.3e-3 at every
depth, the ε-gradient term); `max_porous_residual()` max over the timed steps: M7b 1.86e-16 /
1.95e-16 / 1.98e-16, M7c 1.85e-15 / 7.89e-16 / 5.96e-16 (L4/L5/L6). M3s state hashes (for G-IBM's
byte-identity at N = 48): zh_sphere `4e5493f4b8fe5bf7…`, random_spheres `ee7970632d20eedd…`
(full hashes in the JSON).

**Checkpoint G-PERF-WS (a)–(c) on M1, AUTO = L5** (§4.4):
- (a) t(L5) 81.12 ≤ 1.08 × min(t(L6) 79.36, t(L7) 79.70) = 85.71: **PASS** (L5 is +2.2 % on L6;
  the Q1 fallback is not triggered).
- (b) t(L5) 81.12 ≤ 0.20 × t(L4) 155.57 = 31.11: **MISS** (ratio 0.52). L4 measured 155.6 ms/step
  here, not the pilot's 558: the pilot's L4 ran its host GraphAMG bottom on a loaded host (load
  90–190, `gpu_diag/HANDOFF_WEAK.md`); on this quiet host (load ≤ 6.3) the host bottom is 3.6×
  cheaper while L5–L7 (on-device bottoms) moved 87.4/90.1 → 79.4/79.7. The threshold was derived
  from the pilot's L4; the miss is reported, not re-thresholded (§7 header).
- (c) iterations(L5) 9.00 ≤ iterations(L6) 9.00: **PASS**.

**Findings against the note's premises** (for the coordinator):
1. **No standalone V-cycle on the cut-cell path.** On the CutcellMG path (`set_pressure_geometry` /
   `set_solid`), `solvePressureSystem` (`src/flow_ibm_project.hpp:1482-1585`) dispatches
   Chebyshev / BiCGStab (ghost) / FCG / else MG-PCG — MG-PCG is the terminal default at every np
   (`setPressurePcg`'s own message). M8 (np 2, no driver selected) therefore ran MG-PCG
   (`pressure_vcycle_precision()` = fp32, iterations identical to M4). §2's "standalone V-cycle
   (multi-rank, constant ρ)" and G-ITER's "standalone V-cycle (M8)" coverage have no CutcellMG
   instance on this base.
2. **The CFD-DEM porous default is MG-PCG, not Chebyshev.** `set_porous_continuity` selects
   Chebyshev, but with drag on `configurePorousDragSolver()` (`src/flow_ibm_closures.hpp:394-406`)
   switches to MG-PCG and forces agglomeration (mode 1). M7b (CFD-DEM style, drag) = MG-PCG; M7c
   (no drag) = Chebyshev; M7a (variable-ρ swarm) = Chebyshev.
3. **M8 at np 2 on device: GraphAMG at every depth** (the distributed device bottom is never
   direct, as §1 states); the best depth there is L5 (245.7 vs 282.1 ms at L4, 266.9 at L6).
4. **M6 anchored, smoothed bottom at every depth:** L5 23.0 iterations vs L4 29.5, but L6
   90.4 [60, 113] (×3.1, 59 ms/step vs 16) — relevant to WO-5 (`kAnchoredFollowsGrid`, D 4 → 6 for
   this grid in §4.4). No divergence floor appeared at rtol 1e-12 (≤ 1.1e-16 at every depth).
5. **M5 host 128³ (G-PERF-H shape):** L5 615.1 vs L4 607.3 ms/step = 1.013 (≤ 1.03); iterations
   8.90 vs 7.90 (G-ITER bound 1.15 × 7.90 + 0.5 = 9.59).
6. M2 (G-PERF-BC shape): L5 91.25 vs L4 203.31 = 0.449 (≤ 0.8); iterations 11.70 vs 10.03 (bound
   12.03). M3: K and k* identical to 10 digits at L4/L5/L6 (N = 128).

**H100 job 28551969 (L7):** no report in `gpu_diag/weak/results/` (pilot1–pilot4 only); not added.

## 2026-10-25 — WO-2: gates of the AUTO-default flip (flow d2558a2)

**Modules** `gpu_diag/mgd/ref/mod_wo2_{cuda,host}` (d2558a2). Raw output: `gpu_diag/mgd/logs/wo2/` (census,
battery, grun), `gpu_diag/mgd/wo2/matrix/` (the WO-0 matrix rerun with the `0` column, DUMP=1, 3 rounds,
2026-10-25 09:06 → 11:08), `gpu_diag/mgd/wo2/rtol/` (rtol 1e-10 reruns, L0 vs L4, one run each).
Judging script `gpu_diag/mgd/wo2/judge.py <matrix dir>`. In every row the L0 (AUTO) state hash equals the
explicit depth AUTO resolves to (L5; M6 anchored: L4), all rounds stable.

**G-CENSUS** (PECLET_FLOW_MG_DEBUG=1 battery pre 05c43f6 / post d2558a2, outputs diffed after normalising the
request wording and MPI line order, `wo2/outdiff.py`): moved = {scalar_cutcell_gadv, units_vof_aniso} on host
and CUDA = the census list; coupling 3/3 outputs identical; no state_hash case. Five more tests differ in the
last digits of printed force/torque reductions (hydro_force_units, movingscene_advect_mpi_np{1,2,4},
units_anisotropic_sphere; explicit depth 3 or auto == 4): they differ between two runs of the SAME build
(`logs/wo2/rr3*`), i.e. run-to-run reduction order, not the flip. CUDA scalar_cutcell_g4/g6_* failed only in the
parallel pass (GPU contention, pre and post); serial reruns identical.
Movers (no stored baseline; threshold tests, both pass, nothing re-baselined): gadv 128³ pressure depth 4 → 5,
bottom 16³ → 8³, scalar iterations identical (R/h=32: 10/17/33), budget identity host 9.4e-13 → 9.3e-13 (gate
1e-11); units_vof_aniso 32×64×128 depth 4 → 5, 16³ → 8³, exactness max|u| host 1.244e-17 → 1.145e-17, CUDA
2.018e-17 → 1.540e-17 (gate 8e-13), Young–Laplace rel 0.

**G-BIT** (`wo_gate.sh wo2`): -LE bench 105 + 153 = 258 pass host and CUDA (= WO-1's 258 = base 253 + the 5
WO-1 tests; skips as base); state_hash 14/14 + case_mpi + prof dumps bitwise vs base on both; printed values
identical (diffs = ctest numbering, MPI PIDs only).

**Matrix, L0 vs L4** (ms/step medians; load ≤ 13.1 except M5):

| row | AUTO | ms L0 / L4 / L5 / L6 | iters L0 / L4 (bound) | relcmp L0 vs L4 (tol) | div ratio per step max (steps > 2) | max-over-steps ratio |
|---|---|---|---|---|---|---|
| M1 | 5, 12³ direct | 80.89 / 155.94 / 80.90 / 79.28 (L7 79.68) | 9.00 / 8.00 (9.70) | 6.4e-11 (1e-6) | 1.76 (0/30) | 0.60 |
| M2 | 5, 16×12×8 | 91.14 / 203.99 / 91.22 / 97.11 | 11.70 / 10.03 (12.04) | 2.7e-10 (1e-6) | 3.78 (1/30) | 0.61 |
| M3z | 5, 8³ | 52.82 / 65.13 / 52.89 / 52.58 | 7.00 / 7.00 (8.56) | u,v,w 1.0e-12; p 0.127 all in solid cells, fluid 1.2e-10 | 3.71 (378/400) | 0.95 |
| M3r | 5, 8³ | 51.03 / 65.20 / 51.10 / 50.83 | 7.09 / 7.09 (8.65) | u,v,w 1.0e-13; p 0.121 solid only, fluid 9.7e-11 | 8.06 (1/340) | 1.71 |
| M4 | 5, 128×4×4 | 109.13 / 176.11 / 109.21 / 107.27 | 8 / 8 (9.70) | 4.3e-14 (1e-8) | 1.79 (0/30) | 1.60 |
| M5 host | 5, 8³ | 556.2 / 495.3 / 505.9 / 532.8 — load 33.6 | 8.90 / 7.90 (9.58) | 8.5e-11 (1e-6) | 6.10 (2/30) | 1.38 |
| M6 anch. | 4 (= L4 bitwise) | 16.10 / 16.08 | 29.53 / 29.53 | 0 | 1.00 | 1.00 |
| M7a Cheb | 5, 12³ | 225.84 / 359.17 / 225.71 / 214.93 | 11.97 / 11.00 (13.15) | 1.8e-11 (1e-7) | 30.6 (1/30) | 6.95 |
| M7b | 5, 8³ | 30.07 / 54.80 / 30.07 / 28.89 | 9 / 9 (10.85) | 6.2e-14 (1e-8) | 1.00 | 1.00 |
| M7c Cheb | 5, 8³ | 60.53 / 101.80 / 60.53 / 59.14 | 9 / 9 (10.85) | 1.1e-13 (1e-7) | 1.00 | 1.00 |
| M8 np2 | 5, GraphAMG | 245.61 / 282.13 / 245.68 / 267.04 | 8 / 8 (9.70) | 4.2e-14 (1e-8) | 1.79 (0/30) | 1.60 |

rtol 1e-10 reruns (L0 vs L4, tol 1e-8): M1 6.2e-13, iters 11.2 / 10.07 (bound 12.08), div ratio max 6.91 (4/30
steps > 2); M2 2.4e-12, 15.3 / 13.5 (16.03), 3.87 (7/30); M4 4.3e-14, 8 / 8, 1.79 (0/30); M5 9.5e-13, 11.2 / 9.97
(11.96), 3.50 (3/30). Every outer solve met its stop test (0 failed, max iterations far below the caps), FP32
restarts 0 / 0, Chebyshev restarts 0. M3: K 7.5541697434796 (L0) vs …638 (L4), k* identical to 15 digits;
deep-solid |u| 0 / 0; end-state `max_open_divergence()` M3z 1.110e-12 vs 3.013e-13 (ratio 3.68; L6 4.38e-13),
M3r 7.24e-13 vs 7.30e-13. M3s (N = 48) hashes = WO-0's (4e5493f4…, ee797063…). G-PERF-WS (d): M1 L0 probe has
no `[agmg-build]` (L4: 6).

**Verdicts** (§7 thresholds as registered): G-BIT PASS; G-CENSUS PASS; G-CONV (a) PASS, (b) PASS (M3 on fluid
cells — solid-cell p is outside the system), (c) **MISS** (per-step ratio > 2 on M2, M3z, M3r, M5, M7a and the
1e-10 reruns of M1, M2, M5; worst M7a 30.6), (d) PASS; G-ITER PASS on every row (standalone V-cycle line N/A,
coordinator ruling); G-PERF-WS (a) PASS 80.89 ≤ 85.62, (b) MISS 0.52 (on record, non-blocking), (c) PASS
9.00 ≤ 9.00, (d) PASS; G-PERF-BC PASS 0.447 ≤ 0.8; G-PERF-H NOT JUDGED (load 33.6 / 26.4); G-IBM N = 48 PASS
(bitwise), N = 128 K / k* PASS, iterations PASS, deep-solid PASS, flux divergence **MISS** on M3z if read as
the end-state `max_open_divergence()` (3.68), PASS if read as the per-step projected maximum (0.95).
Stopped on the misses for the coordinator; nothing re-thresholded.

## 2026-10-25 — WO-2 re-gate under Amendment A1 (flow 7d37fdc amendment, a5dcc9a Chebyshev trace)

The divergence gates as amended (§7 Amendment A1, USER 2026-10-25; register umbrella `docs/decisions/flow.md`
7c569d6). The WO-2 MISSes above stand on record against the old wording. Host module at a5dcc9a (inert:
below), the matrix steps, `PECLET_FLOW_MG_DEBUG=2` traces; L0 = AUTO (5), L4 = explicit depth 4. Condition (1):
max over steps of div_k / (rtol·r0_k), div_k = `max_open_divergence_projected()`, r0_k and rtol from the trace of
that step's solve (one solve per step on every row); div_k / final |r|inf = 1 within 1e-4 on every judged run.
Condition (2): max_k div_k (L0) / max_k div_k (L4) at rtol 1e-13. Instruments: `bubble_column_perf/gpu_diag/mgd/
wo2c/{cond1,cond2}.py`; M1, M2, M3z, M5 and M7a's 1e-13 runs are the diagnosis runs (`mgd/wo2diag/out`, d2558a2,
whose result equals a5dcc9a's — M7a traced hashes equal); M7a at its rtol and M3r are new runs on a5dcc9a.

| row (driver) | rtol | cond (1) L0 | cond (1) L4 | steps | cond (2) at 1e-13 | verdict |
|---|---|---|---|---|---|---|
| M1 (PCG) | 1e-8 | 0.636 | 0.449 | 30 | 0.756 | PASS |
| M1 | 1e-10 | 0.749 | 0.966 | 30 | — | PASS |
| M2 (PCG) | 1e-8 | 0.937 | 0.991 | 30 | 1.243 | PASS |
| M2 | 1e-10 | 0.966 | 0.991 | 30 | — | PASS |
| M3z (PCG, G-IBM zh_sphere) | 1e-8 | 0.884 | 0.928 | 400 | 1.033 | PASS |
| M3r (PCG, G-IBM random_spheres) | 1e-8 | 0.979 | 0.977 | 340 | 0.868 | PASS |
| M5 (PCG, host) | 1e-8 | 0.978 | 0.845 | 30 | 1.048 | PASS |
| M5 | 1e-10 | 0.825 | 0.756 | 30 | — | PASS |
| M7a (Chebyshev) | 1e-9 | 0.971 | 0.140 | 30 | 1.193 | PASS |

**Verdicts:** G-CONV (c) as amended PASS on every judged row (M1, M2, M3z, M3r, M5, M7a, and the 1e-10 reruns);
the rows the brief did not list (M4, M7b, M7c, M8; per-step ratio ≤ 2 under the old wording, every solve met its
stop test per G-CONV (a)) were not re-traced: condition (1) NOT JUDGED there. G-IBM as amended PASS (zh_sphere
0.884 / 0.928, 1.033; random_spheres 0.979 / 0.977, 0.868); its other clauses as recorded above. Information, not
a gate: at rtol 1e-13 the true divergence decouples from the recurrence residual (M3r: div / |r| up to 95,
div/(rtol·r0) 9.0 / 9.3 at L0 / L4, worst div 3.0e-15 / 3.5e-15 — the round-off floor at both depths), which is
why condition (1) is set at the production tolerance only.

Inertness of a5dcc9a against the WO-2 build: `state_hash.py` 13 + np2 1 hashes identical host and CUDA; prof.py
dumps bitwise (host 4, CUDA 2); `ctest -LE bench` 258 / 258 pass host and CUDA (same 2 SKIPs each); printed
values identical (CUDA: only the prterun PID in the two SKIP banners differs); Chebyshev on CUDA, `--dump` vs the
WO-2 matrix, `np.array_equal` on every field: M7a L0 / L4 and M7c L0 / L4 switch off, M7c L0 switch on.

## 2026-10-25 — review fixes (REVIEW_MGDEPTH.md S1–S3; coordinator decisions in DECISIONS_GPU.log)

**S3 — G-PERF-H verdict, from the quiet WO-0 numbers.** AUTO on M5 (host swarm 128³, 1 × 6 threads) resolves
to depth 5 and is bitwise the explicit depth-5 run (WO-2 matrix: "In every row the L0 (AUTO) state hash equals
the explicit depth AUTO resolves to (L5 …)"). The WO-0 reading (this log, WO-0 note 5: "M5 host 128³ (G-PERF-H
shape): L5 615.1 vs L4 607.3 ms/step = 1.013 (≤ 1.03); iterations 8.90 vs 7.90 (G-ITER bound … 9.59)", load
≤ 17.2) therefore is the AUTO reading: t(AUTO) / t(L4) = 615.09 / 607.26 = 1.013 ≤ 1.03 → **G-PERF-H PASS**;
G-ITER 8.90 ≤ 9.59 PASS. The WO-2 rerun at load 33.6 stays NOT JUDGED.

**S1 — gate row M9: SolverColocated + solid, the ghost projection's BiCGStab driver** (thresholds pre-registered
by the coordinator, 2026-10-25). `mgd_case.py --case zh_sphere|random_spheres --n 128 --solver colocated` (no
scheme call: the AUTO default 'ghost'; BiCGStab at the case's MG-PCG cap 300, rtol 1e-8), the steady march of
`sdflow_regression.py`; one run per depth on the CUDA module of 8796951 (`gpu_diag/mgd/ref/mod_s1_cuda`) with
`PECLET_FLOW_MG_DEBUG=2 PECLET_FLOW_MG_DEBUG_SOLVES=1000000` (the 8796951 exit trace, every solve; debug output
only — off vs on bitwise at N = 48). Raw: `gpu_diag/mgd/fix/m9/`; judge `gpu_diag/mgd/fix/m9_judge.py`.
G-CONV (b) = max over u, v, w, p of max_fluid|a − b| / max_fluid|a| (fluid = sdf > 0 at the cell centres).

| case | AUTO depth, bottom (L4 bottom) | steps L0 / L4 | exits L0 / L4 | breakdowns L0 / L4 | G-CONV (b) fluid (all cells) | iters L0 / L4 (bound) | metric L0 / L4, rel |
|---|---|---|---|---|---|---|---|
| zh_sphere | 5, 8³ direct (16³ GraphAMG) | 400 / 400 | 400 × stop test / 400 × stop test | 0 / 0 | p 2.43e-14, v w 2.30e-14, u 2.50e-15 (same) | 11.990 / 12.023 (14.326) | K 7.557283136435 / …435, 2.4e-16 |
| random_spheres | 5, 8³ direct (16³ GraphAMG) | 340 / 340 | 340 × stop test / 340 × stop test | 0 / 0 | p 1.16e-13, v 3.44e-15 (same) | 18.015 / 18.012 (21.214) | k* 0.006247594073822 both, 0 |

Stagnation exits 0 at both depths; deep-solid |u| 0 / 0; end-state `max_open_divergence()` zh 3.44e-14 / 1.16e-14,
random 1.704e-6 / 1.704e-6 (equal). Load 24–29 during the runs (another job outside the lock; iteration counts
and fields are load-independent, no timing is judged in M9). **Verdicts:** G-CONV (a) PASS (every solve exits on
its stop test; breakdowns 0 = 0), G-CONV (b) PASS (≤ 1.2e-13 against 1e-6), G-ITER PASS, physics PASS (K, k*
within 2.4e-16).

**S2 — probe row M10: an axis that stops coarsening early** (thresholds pre-registered by the coordinator).
`run_matrix.sh … M10 M10b` on the CUDA module of 8796951 (+ the inert follow-ups), 3 interleaved rounds, 5 warm-up
+ 30 timed steps; `mgd_case.py --case swarm --dims 128 128 25`: case A's swarm (d/h 20, MG-PCG rtol 1e-8) on the
anisotropic box; *measurement choice*: a bubble lattice of max(1, N_axis // 64) per axis = 2 × 2 × 1, 4 bubbles,
α 0.0409. Raw: `gpu_diag/mgd/fix/m10/`; judge `gpu_diag/mgd/fix/m10_judge.py`.

| row | AUTO depth, levels (global) | bottom AUTO / L4 | iters AUTO / L4 (bound) | ms/step AUTO / L4 (rounds) | load max |
|---|---|---|---|---|---|
| M10 128×128×25 | 5: 128×128×25, 64×64×25, 32×32×25, 16×16×25, 8×8×25 | 8×8×25 direct / 16×16×25 GraphAMG | 15.967 [max 17] / 9.933 [max 10] (11.923) | 16.57 (16.6, 16.6, 16.6) / 55.37 (50.7, 59.9, 55.4) | 26.0 |
| M10b 512×512×100 | not run: CUDA out of memory at construction (211 MiB "AT", RTX 5080 16 GB) | | | | |

z (25) never coarsens, so level ℓ has cells 2^ℓ : 2^ℓ : 1 — the last smoothed level (3) at 8:1, the bottom at
16:1, as the review predicted. Iterations identical in all rounds; 0 failed solves, 0 FP32 restarts.
**Verdicts:** G-ITER **MISS** (15.967 > 11.923; +61 % over depth 4); t(AUTO) ≤ t(L4) NOT JUDGED (load 26.0 > 24;
AUTO 3.3× faster in the loaded reading). Stopped on the MISS as the brief requires: a coarse-cell aspect bound for
AUTO is a design decision for the architect. Not run: the rebase onto flow main and its re-gate.
