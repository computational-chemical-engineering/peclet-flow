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

## 2026-10-26 — WO-A2a stopped on a fact; WO-A2b G-ASP partial (Amendment A2.5)

**WO-A2a (not committed: MISS).** A2.3 implemented as written (`Level::lineAxis`, `strongAspect`, census/table
print; diff `gpu_diag/mgd/a2/woa2a.patch`). G-BIT on that build: `state_hash` 14 (13 + np2) and prof dumps bitwise
vs the unmodified a5b6552 (= the s1 refs) host and CUDA; printed values identical (CUDA: prterun PIDs only).
`mg_depth_policy` FAILS one row: **16×16×4 (§4.4, the `state_hash` `scalar` case) triggers** — L1 8×8×2 blocks z,
L3 2×2×2 has A' = 4 > θ and is the SMOOTHED bottom (2 ≤ extent 4, not agglomerated), so {3} is a triggered smoothed
level. A2.3 says every §4.4 row has none; the rule and the claim disagree. All 8 A2.3 rows pass (128×128×25 → D 5
{2,3}; its explicit 4 → {2}; 512×512×100 → D 7 {4,5}; 512×64×25 → D 5 {2,3}; 34×34×116, 8×8×18 → none).
Stopped for the architect (WO-7 would move that hash).

**WO-A2b** on the WO-A2a build (bitwise the unmodified one), `mgd_case.py` via `a2/tdiag.py` (traced, MG_DEBUG=2,
debug output only) and `run_matrix.sh` rows M10, M10h (3 rounds, 5 + 30, L 0 4 6). Raw `gpu_diag/mgd/a2/b/`;
judge `a2/asp_judge.py`. D_auto = 5 on all three rows (bottoms 8×8×25, 8×8×25, 32×4×25 direct; L4 16×16×25,
16×16×25, 64×8×25 GraphAMG).

| row | ms L0 / L4 / L6 | iters L0 / L4 / L6 | (1b) vs L4 prod; 1e-10 | A1 (1) L0 / L4 prod; 1e-10 | A1 (2) 1e-13 | load |
|---|---|---|---|---|---|---|
| M10 (PCG 1e-8) | 16.41 / 47.71 / 20.79 | 15.967 / 9.933 / 27.333 | 2.2e-10 (1e-6); 1.6e-12 (1e-8) | 0.982 / 0.859; 0.962 / 0.669 | 0.774 | 36.4 |
| M10h (Cheb 1e-9) | 179.73 / 232.25 / 280.29 | 21.967 / 13.000 / 45.533 | 1.6e-11 (1e-7); 1.6e-12 (1e-8) | 0.789 / 0.732; 0.973 / 0.980 | 1.390 | 29.5 |
| M10v (PCG 1e-10, np 2) | not run | 20.0 / 13.0 (traced) | not run | 0.728 / 0.377 | 1.143 | — |

Every solve met its stop test (0 failed, max 17 / 22 / 20 ≪ caps), FP32 and Chebyshev restarts 0 everywhere,
iterations identical across rounds. **Verdicts:** G-ASP (1) PASS M10, M10h; M10v (1a) (traced) and A1 (1), (2) PASS,
(1b) NOT RUN. G-ASP (3) iterations PASS M10 (15.967 ≤ 27.333), M10h (21.967 ≤ 45.533). G-ASP (2) and (3) timing
NOT JUDGED (load 36.4 / 29.5 > 24; loaded readings AUTO 0.34 × L4 on M10, 0.77 × on M10h). M10v matrix not run
(queued behind another job's lock; cancelled at the context cap).

## 2026-10-26 — WO-A2a committed (coordinator ruling); WO-A2b timing attempt 2

**WO-A2a committed (d2e525f).** Coordinator ruling 2026-10-26: A2.3's rule stands (a smoothed bottom included),
its claim is corrected — 16×16×4 triggers {3}, so WO-7 moves the `scalar` state_hash (recorded change); rejected:
exempt the smoothed bottom from T. Ctest expects {3} on that row. G-BIT on the final diff: state_hash 14 + np2 +
prof dumps PASS host and CUDA vs `ref/a2base_*`; printed identical (CUDA: PIDs only); `mg_depth_policy*` 5/5 both
(8/8 line rows); battery 105/105 + 153/153 both (CUDA `scalar_cutcell_g4` failed in the parallel pass only).

**WO-A2b attempt 2** (`a2/b/attempt2.out`, `judge_a2.txt`; 04:08–07:12, rows M10 M10h M10v, mod_a2a).

| row | ms L0 / L4 / L6 | iters L0 / L4 / L6 | (1b) prod vs L4 | load max | (2) AUTO ≤ L4 | (3) AUTO ≤ 1.03 × L6 |
|---|---|---|---|---|---|---|
| M10 | 16.37 / 47.64 / 20.77 | 15.967 / 9.933 / 27.333 | — (as attempt 1) | 26.7 | NOT JUDGED | iters PASS; t NOT JUDGED |
| M10h | 199.66 / 230.41 / 282.61 | 21.967 / 13.000 / 45.533 | PASS | 20.1 | PASS (≤ 1.03 × 230.41) | PASS |
| M10v (np 2) | 356.22 / 396.16 / 458.94 | 20.000 / 13.000 / 32.800 | PASS 3.1e-13 (tol 1e-8) | 10.2 | PASS (≤ 1.03 × 396.16) | PASS |

Every solve met its stop test (0 failed), FP32 flips and Chebyshev restarts 0, iterations identical across rounds
and attempts. M10v G-ASP (1) PASS (A1(1) 0.728 / 0.377, A1(2) 1.143). **Verdicts:** M10h and M10v G-ASP (1)–(3)
PASS. M10 timing still NOT JUDGED (load 26.7 > 24; loaded AUTO 0.34 × L4, 0.79 × L6, rounds within 0.2 %).

**WO-A2b attempt 3 (M10 only; `a2/b_attempt_m10.sh 3`, `judge_a3.txt`; 07:44, load max 0.3).** ms L0 / L4 / L6
16.40 / 47.70 / 20.80 (rounds within 1.5 %), iters 15.967 / 9.933 / 27.333 (identical to attempts 1–2), 0 failed.
G-ASP (1) PASS; (2) 16.40 ≤ 1.00 × 47.70 PASS; (3) iters PASS, t 16.40 ≤ 1.03 × 20.80 PASS. **WO-A2b: G-ASP (1)–(3)
PASS on M10, M10h and M10v.** (The judge's GASP_ALL line reads M10h/M10v NOT RUN in this matrix; they are judged in
attempt 2.)

## 2026-10-26 — rebase onto f4ec2ef (flow origin/main 661eee6 = f4ec2ef + a study-script commit)

`mgdepth` (21 commits, old HEAD 04517fb on 89e5fa7) rebased onto origin/main 661eee6 without conflicts; new tip
8a7616e (the WO-A2b attempt-3 log commit). 661eee6 touches only `tests/study/vof_perf/weak_swarm.py` (outside ctest
and state_hash), so main's f4ec2ef references stand: logs `bubble_column_perf/validation/wo7/{host,cuda}_mgdmain.*`.
Gate `validation/wo7/gate.sh <host|cuda> mgdrebase` (incremental rebuild, `-LE bench -V -j 4`, OMP 4, under the
heavy lock); census `compare_name.py` (by test name, worktree paths normalised).

| gate | host | CUDA | verdict |
|---|---|---|---|
| -LE bench battery | 267/267 (= main 262 + 5 mg_depth_policy*) | 267/267 | PASS (no flake this time) |
| state_hash np1 14 + np2 digest | 460e6fd14885673e (= main) | 20f528f0c295ca15 (= main) | PASS |
| census movers vs main | scalar_cutcell_gadv (budget identity 9.4e-13 → 9.3e-13), units_vof_aniso (max\|u\| 1.244e-17 → 1.145e-17) | gadv (Pe_h 0.1 budget 4.9e-16 → 3.5e-17), units_vof_aniso (2.018e-17 → 1.540e-17) | PASS (= WO-2 movers, same values) |
| other output diffs | hydro_force_units, movingscene_advect_mpi_np{1,2,4}, units_anisotropic_sphere: last digits of force/torque reductions (the WO-2 run-to-run set); scalar_cutcell_solve_mpi_np4: 4-rank stderr interleaving of one warning, no numeric line | same five run-to-run tests; bottom_direct_mpi_np{2,4} (skipped both sides): prterun PIDs | not movers |
| full stress × depth movers | none | none | — |
| mg_depth_policy* | 5/5, 8/8 line rows; output = pre-rebase a2a/mgdp_host.txt except test numbers | 5/5, 8/8; = a2a/mgdp_cuda.txt except numbers | PASS |
| M10 trigger rows | 128×128×25 D5 {2,3}, explicit 4 {2}; 512×512×100 D7 {4,5}; 512×64×25 D5 {2,3} | identical | PASS (unchanged) |

Raw: `validation/wo7/{host,cuda}_mgdrebase.{build,ctest,hash}.*`, `gpu_diag/mgd/rebase_gate.out`.

## 2026-10-26 — U-A2 answered: P1 (USER "I agree with P1")

Hierarchies with a triggered smoothed level (A2.3) are judged for WO-2 on G-ASP, not G-ITER. G-ASP (WO-A2b
attempt 3, 8a7616e) is PASS on all three rows: M10 16.40 / 47.70 / 20.80 ms (AUTO / L4 / L6, load 0.3), M10h 199.66 /
230.41 / 282.61, M10v 356.22 / 396.16 / 458.94. G-ITER keeps its threshold and baseline and becomes WO-7's acceptance
gate. The M10 MISS (AUTO D5 15.967 iterations vs bound 11.923) stays on record. Q-A3 is not answered; its default
applies: LR is built (WO-7, branch `mglr`). WO-2 lands on main with this commit; next are WO-3, WO-4 and WO-7.

## 2026-10-26 — WO-7: zebra line relaxation (Amendment A2.4), gates G-LR / G-BIT

**Implementation** (branch `mglr` from db1468b): 067bb38 (A2.4 kernel `src/mg_line_relax.hpp`, CutcellMG wiring,
setter `setLineRelaxation`, `Solver::setPressureLineRelaxation`, binding
`diagnostics.set_pressure_line_relaxation(enabled)`, default True), 7a740b5 + c4f3459 (ctest `mg_line_relax`; line
checks in `mg_fp32_vcycle_mpi`), 210bdac (A2.6 latency remedy: c, y, z in thread-local arrays for m ≤ 64, bitwise).
Study tools: `mgd_case.py --line-relaxation on|off`, `run_matrix.sh LR_OFF_LEVELS` (arms `<L>off`). Gate scripts:
`gpu_diag/mgd/wo7/` (`glr_attempt.sh`, `glr_judge.py`, `trace.sh`, `np1_gate.sh`, `hash_lr_off.py`, `gbit.sh`).

**G-LR (5) `mg_line_relax`** (host, CUDA): one colour pass = the dense (cyclic) column solve, 32×32×{25, 2, 3},
s = z periodic and walls, 5 % closed faces, one sealed column: FP64 ≤ 6.7e-16, FP32 ≤ 3.0e-7 relative; other colour
bitwise unchanged; all finite; FP32 x = c·1, b = 0 bitwise unchanged; wall weights 0. **PASS** both.

**G-LR (4) bit identity** (`validation/wo7/{host,cuda}_mglr.*`, vs `*_mgdrebase.*`):

| check | host | CUDA | verdict |
|---|---|---|---|
| -LE bench battery | 268/268 (267 + mg_line_relax) | 268/268 | PASS |
| state_hash 13 np1 + np2 | 12 + np2 equal; `scalar` fc99aa51…c584 → b9fc5744…89f8 | 12 + np2 equal; `scalar` 83554517…5a92 → 27445fd8…63d1 | PASS (ruled mover only) |
| digest (14 lines) | 460e6fd14885673e → 51fec031a160c88b | 20f528f0c295ca15 → 8337752781faf71b | — |
| LR off ≡ pre-WO-7 | state_hash 13 with LR off (`hash_lr_off.py`) = main incl. `scalar` | M10 30 steps LR off vs the db1468b module: u v w p C bitwise, iteration + div lists equal | PASS |
| census movers (old → new) | full_stress_large_dt (d_999/d_0 4.96e-11 → 4.97e-11), open_face_plane_mpi_np3 (B_census 1.6e-14 → 3.8e-14 …), verify_colocated_taylor_green (div 8.445e-16 → 6.545e-16), vof_bc (F2 outflow u 0.091916056118741554 → …742984) | same four (taylor_green 8.397e-16 → 6.464e-16, vof_bc …741582 → …74297, B_census 6.9e-15 → 7.8e-15) | every mover triggers (census) |
| other diffs | the WO-2 run-to-run set (hydro_force_units, movingscene_advect_mpi_np{1,2,4}, units_anisotropic_sphere, scalar_cutcell_solve_mpi_np4 stderr) | same + mg_fp32_vcycle_mpi_np* (new line checks) | not movers |

Triggered tests (MG_DEBUG=1 census on the unmodified host build, `[mg-census] … A2 line-trigger`): verify_lid_cavity_sdflow,
verify_colocated_taylor_green, colocated_open_boundary, open_face_plane(_mpi_np3), cell_force_placement, mirror_symmetry,
full_stress_large_dt(_outflow), collocated_stability_guard, scalar_cutcell_g9, scalar_mg, vof_bc (along y),
mg_depth_policy (hierarchy only), units_anisotropic_tgv. The rest print identical values.

**np 1 ≡ single rank.** As written (M10 swarm host, init_mpi at np 1 vs single rank, LR on): **NOT JUDGEABLE** — the
LR-OFF control already differs (host PCG: max|Δp| 7.4e-7 of 820, iteration lists equal; host Chebyshev and CUDA, and
the M10v channel on CUDA, likewise), a pre-existing solver-level np 1 difference independent of WO-7. At the CutcellMG
level (`mg_fp32_vcycle_mpi` lineChecks, MG-PCG rtol 1e-10, host and CUDA): anisoW 64×64×25 z walls (AUTO) and thinW
64×64×4 z walls (5 levels, 'fp64' and forced 'fp32') np 1 memcmp-equal with LR on (and the LR-off control equal);
aniso 64×64×25 periodic: the LR-off control differs at np 1 (102398 cells), so n/a; np 2, 4 (8 on CUDA): max|dx|
≤ 1.8e-11, iterations equal. Every triggered level a rank holds is line-relaxed.

**G-LR (1)–(3), attempt 1** (`gpu_diag/mgd/wo7/glr_attempt1.out`, matrix `wo7/matrix_a1`, traces `wo7/trace`; module
mod_wo7 = 067bb38; 3 interleaved rounds of 5 + 30 steps; arms L0 = AUTO, L0off, L4, L4off, L6; load max 22.7 / 19.3 /
19.1, no other GPU app; iterations identical across rounds; 0 failed solves, 0 FP32 / Chebyshev restarts). The
rebased tree (advection + full stress defaults) moved the LR-off baselines: M10 AUTO 16.1 (was 15.967), L4 10.1 (was
9.933).

| row | iters L0 / L0off / L4 / L4off / L6 | ms L0 / L0off / L4 / L4off / L6 | (1) ≤ 1.15 × L4off + 0.5 | (2) t(L0) ≤ k × t(L0off) | (3) G-ASP (1), LR on |
|---|---|---|---|---|---|
| M10 (PCG 1e-8) | 8.100 / 16.100 / 8.067 / 10.100 / 8.200 | 18.22 / 17.17 / 47.64 / 49.74 / 20.29 | 8.100 ≤ 12.115 PASS | 18.22 > 17.17 (k 1.00) **MISS** | **MISS**: A1 (2) r13 worst div L0 2.345e-15 / L4 7.910e-16 = 2.965 > 2; (1a), (1b) (1.8e-10 prod, 1.2e-12 r10), A1 (1) (0.986 / 0.906; 0.886 / 0.687) PASS |
| M10h (Cheb 1e-9) | 10.033 / 21.967 / 10.000 / 13.000 / 11.000 | 161.45 / 196.31 / 229.84 / 239.81 / 169.51 | 10.033 ≤ 15.450 PASS | 161.45 ≤ 1.03 × 196.31 PASS | PASS (A1 (2) 0.428) |
| M10v (np 2, V-cycles) | 9.000 / 20.000 / 9.000 / 13.000 / 9.000 | 227.15 / 317.78 / 312.55 / 362.53 / 227.14 | 9.000 ≤ 15.450 PASS | 227.15 ≤ 1.03 × 317.78 PASS | PASS (A1 (2) 0.889) |

G-ASP (2), (3) with LR on (information): M10 18.22 ≤ 47.64, iters 8.100 ≤ 8.200, t ≤ 1.03 × 20.29; M10h, M10v PASS.
**G-ITER (Q-A2): LR reaches it on every row** — AUTO 8.1 vs depth 4 without LR 10.1 on M10.
The (2) MISS on M10 is latency (iterations halve, step time +6 %), so A2.6's remedy was applied (210bdac: c, y, z in
thread-local arrays for m ≤ 64; bitwise — M10 dump vs attempt 1, state_hash digests host 51fec031a160c88b, CUDA
8337752781faf71b unchanged) and (2) re-judged once: attempt 2 below. (3) is numerics, which 210bdac does not change.

**Observation (not a gate; for the architect): LR hurts MG-PCG where the only triggered level is a SMOOTHED bottom of a
thin slab.** CutcellMG-level, 64×64×4 between z walls, blob 1e2, CutcellMG's own default agglomeration 'never',
rtol 1e-10 (host): PCG 125 → 158 iterations (4 levels), 89 → 126 (5 levels) with LR; FCG on the same problems
17 → 12 and 17 → 12. 64×64×25 (several triggered levels): PCG 32 → 12, 34 → 12; FCG 30 → 10. The smoothed bottom
runs `bottom_` forward sweeps (colour 0 then 1, never reversed), so the V-cycle is not symmetric there; FCG tolerates
that, PCG does not, and the stronger LR bottom makes it worse. The `scalar` state_hash case (16×16×4, triggered
smoothed bottom L3 2×2×2) is of this class. Not tuned; reported.

**G-LR (2) re-judge (attempt 2, mod_wo7r = 210bdac):** PENDING — `gpu_diag/mgd/wo7/glr_attempt.sh 2 wo7r` waits for no
other GPU compute app and load ≤ 24 (up to 4 h), then the same matrix; verdict in `wo7/glr_attempt2.out` (GLR lines).

## 2026-10-29 — WO-7 landing prep: rebase onto eddb790, G-LR (4) re-gate (np 1 judged), G-LR (3) re-judged under A1 (2′)

**Rebase.** `mglr` rebased onto flow `origin/main` eddb790 (CA red-black pairs off across an odd periodic global
extent; ctest `ca_odd_periodic_mpi_np{1,2}`). The source commits applied cleanly; the only conflict was this log
(both sections kept, main's first). Branch commits after the rebase: d2fc25c (A2.4 kernel), 14f3728 + 8771ff5 (ctests),
71f57c4 (A2.6 remedy), ae199ff (gate record), 1b4459a (Amendment A3). Gate scripts: `gpu_diag/mgd/wo7l/` (`land2.sh`,
`land3.sh`, `judge3.py`, `tdiag2.py`); module snapshot `ref/mod_wo7l_{host,cuda}`.

**G-LR (4) bit identity, re-gated** (`validation/wo7/gate.sh <host|cuda> mglr2`; raw `validation/wo7/{host,cuda}_mglr2.*`,
census vs `*_mgdrebase.*` = pre-WO-7 main by `gpu_diag/mgd/caodd/compare_any.py`):

| check | host | CUDA | verdict |
|---|---|---|---|
| -LE bench battery | 270/270 (main 269 + mg_line_relax) | 270/270 (clean rerun `cuda_mglr2b`; the first run `cuda_mglr2` had two -j4 failures, scalar_mg = CUDA allocation failure of 96 MiB while another session's process held the GPU, and the known flake scalar_cutcell_g4; both passed serially with printed values identical to main, `cuda_mglr2.rerun.ctest.log`) | PASS |
| state_hash 13 np1 + np2 | 12 + np2 = main; `scalar` fc99aa51…c584 → b9fc5744…89f8 | 12 + np2 = main; `scalar` 83554517…5a92 → 27445fd8…63d1 | PASS (ruled mover only) |
| digest (14 lines) | 51fec031a160c88b (= the pre-rebase WO-7 digest) | 8337752781faf71b (= pre-rebase) | — |
| census movers vs main | full_stress_large_dt, open_face_plane_mpi_np3, verify_colocated_taylor_green, vof_bc — printed values identical to the pre-rebase WO-7 record | same four, identical to the pre-rebase record | PASS |
| other diffs | mg_fp32_vcycle_mpi_np{1,2,4,8} (the 14 new line-check lines only; nothing old changed); noise: hydro_force_units, movingscene_advect_mpi_np{1,2,4}, units_anisotropic_sphere, scalar_cutcell_solve_mpi_np4, balanced_force_mpi_np2 and varmu_mpi_np4 (interleaved stderr) | mg_fp32 alike; noise: the same set + bottom_direct_mpi_np{2,4} skip text | not movers |
| M10 host np 1 `init_mpi` ≡ single rank, LR on (`wo7/np1_gate.sh`, 10 + 2 steps) | u v w p C memcmp-equal, per-step iteration and divergence lists equal (D 5, 8.7 it/step); LR-off control equal too (16.5) | — | **PASS** (was NOT JUDGEABLE: the CA bug, fixed by eddb790) |

The CA fix also closes the CutcellMG-level gap of the pre-rebase record: `mg_fp32_vcycle_mpi` aniso 64×64×25 periodic,
np 1, LR-off control now memcmp-equal (CUDA was 102398 cells differ), LR on equal; CUDA np 2 control max|dx| 1.2e-10 → 1.1e-13
(main's change, not WO-7's).

**G-LR (3) re-judged = G-ASP (1) with LR on, divergence condition (2′)** (A3.6, USER 2026-10-27; `wo7l/land3.sh`,
judge `wo7l/judge3.py`, raw `wo7l/trace/`, verdicts `wo7l/glr3.txt`). Module mod_wo7l (rebased `mglr`). new = AUTO,
old = explicit 4, both with LR at its default (on; LR triggers in both hierarchies), so the arms differ only in the
depth. Per row and arm: the production rtol (dumps), the 1e-10 rerun (dumps; M10, M10h), and (2′): rtol 1e-17, cap 300
(`tdiag2.py`: the driver re-selection of the A1 (2) runs, `--pcg-rtol` / `--cheb-rtol`, with the cap as an argument),
validity checked per run (min_k div_k / (rtol r0_k) ≥ 10; the 1e-19 fallback was not needed). G-CONV (a), (b) use the
traced runs (the attempt-1/2 matrix dumps are not reused; the traced production dumps are bitwise equal to attempt 2's,
below).

| row | (1a) stop test, restarts | (1b) AUTO vs 4: prod; 1e-10 | A1 (1) L0 / L4: prod; 1e-10 | (2′) validity min div/(rtol·r0) L0 / L4; it/step | (2′) worst div L0 / L4 = ratio (≤ 4) | G-LR (3) |
|---|---|---|---|---|---|---|
| M10 (PCG 1e-8) | 0 failed, max it 9 / 12 (cap 800), 0 FP32 flips | 1.790e-10 (≤ 1e-6); 1.150e-12 (≤ 1e-8) | 0.986 / 0.906; 0.886 / 0.687 | 214 / 244; 18.9 / 18.4 | 3.608e-16 / 3.192e-16 = **1.130** | **PASS** |
| M10h (Cheb 1e-9) | 0 failed, max it 11 / 11 (cap 120), 0 restarts | 5.193e-12 (≤ 1e-7); 8.679e-13 (≤ 1e-8) | 0.985 / 0.411; 0.679 / 0.297 | 275 / 214; 19.0 / 18.2 | 3.331e-16 / 3.331e-16 = **1.000** | **PASS** |
| M10v (np 2, 1e-10) | 0 failed, max it 9 (cap 500) | 1.341e-14 (≤ 1e-8); — | 0.161 / 0.158; — | 7.5e3 / 8.1e3; 15.0 / 15.0 | 1.735e-18 / 1.735e-18 = **1.000** | **PASS** |

No (2′) run reached its cap (max 23 iterations). The old A1 (2) MISS on M10 (rtol 1e-13: 2.345e-15 / 7.910e-16 =
2.965, attempt 1) stays on record; under (2′) the same pair sits at its round-off floor, 1.13 apart.

**G-LR (1), (2): attempt 2 (`wo7/glr_attempt2.out`, mod_wo7r = 210bdac before the rebase; load max 19.6 / 22.7 / 17.0,
no other GPU app; iterations identical across the three rounds).** They stand on the rebased tree: its production
runs are bitwise equal to attempt 2's round-1 dumps (u v w p C, every row, AUTO and explicit 4).

| row | iters L0 / L0off / L4 / L4off / L6 | ms L0 / L0off / L4 / L4off / L6 | (1) ≤ 1.15 × L4off + 0.5 | (2) t(L0) ≤ k × t(L0off) |
|---|---|---|---|---|
| M10 | 8.100 / 16.100 / 8.067 / 10.100 / 8.200 | 16.61 / 17.00 / 48.97 / 75.82 / 18.89 | 8.100 ≤ 12.115 PASS | 16.61 ≤ 1.00 × 17.00 **PASS** |
| M10h | 10.033 / 21.967 / 10.000 / 13.000 / 11.000 | 165.06 / 207.34 / 231.88 / 241.85 / 168.46 | 10.033 ≤ 15.450 PASS | 165.06 ≤ 1.03 × 207.34 PASS |
| M10v | 9.000 / 20.000 / 9.000 / 13.000 / 9.000 | 229.50 / 324.28 / 305.12 / 371.18 / 241.68 | 9.000 ≤ 15.450 PASS | 229.50 ≤ 1.03 × 324.28 PASS |

**WO-7 gate summary on `mglr` (rebased): G-LR (1) PASS, (2) PASS (attempt 2, after the A2.6 remedy), (3) PASS under
(2′), (4) PASS (np 1 included), (5) PASS (`mg_line_relax` green in both batteries).** Not pushed, not merged.

## 2026-10-26 — Amendment A3: WO-S1 (symmetric V-cycle, switch off) and WO-S2 (stopped at G-SYMCONV)

Branch `mgsym` from `mglr` cb981c9 (worktree `suite/flow-mgsym`). WO-S1 = f81ecbd: A3.3 items 1–5 behind
`CutcellMG::setSymmetricVcycle` / `s.diagnostics.set_pressure_vcycle_symmetric` (default off), ctests
`mg_vcycle_symmetry` and `mg_vcycle_symmetry_mpi_np{1,2,4}`. Tools and raw output: `bubble_column_perf/gpu_diag/mgd/sym/`
(build/gate scripts, `logs/`, `symconv/`); study tools here: `tests/study/mg_depth/symconv.cpp` (G-SYMCONV),
`mgd_case.py --vcycle-symmetric on|off`, `run_matrix.sh` tokens `<L>sym` (`SYM_LEVELS`, `ONLY_LEVELS`), `summarize.py`
accepting depth tokens.

**G-SYM** (switch on; host, f81ecbd): a = |xᵀMy − yᵀMx| / (|x||My|) of the production V-cycle (schedule 2/2/12):

| case | a trilinear/forward (switch off, recorded) | a symmetric | bound | np 2 / np 4 |
|---|---|---|---|---|
| thinW 64×64×4 z walls, ρ×100 blob, D5 | 8.12e-5 | 1.45e-17 | 1e-14 | 2.9e-17 / 2.2e-17 |
| thinW, LR off | 4.51e-5 | 2.28e-17 | 1e-14 | 1.1e-17 / 3.3e-17 |
| thin periodic slab, D5 | 1.90e-4 | 2.52e-17 | 1e-14 | 8.1e-18 / 1.3e-17 |
| `scalar` 16×16×4 (x/y walls, D4, bottom 2×2×2) | 1.27e-4 | 1.15e-17 | 1e-14 | 0 / 3.8e-18 |
| 64³ periodic ρ×100 blob, D5 | 2.87e-5 | 5.7e-19 | 1e-14 | 4.5e-18 / 2.3e-18 |
| 64³ walls ρ×100 blob, D5 | 2.17e-5 | 4.9e-19 | 1e-14 | 4.0e-18 / 0 |
| `cylProblem` 32³, D4 | 1.32e-5 | 6.7e-18 | 1e-14 | 1.1e-17 / 1.3e-17 |
| outflow-anchored channel 32×16×16 (−x inflow, +x outflow, y/z walls), D4 | 8.39e-5 | 3.42e-17 | 1e-14 | 4.4e-17 / 3.1e-17 |
| 64³ periodic blob D4, agglomerated bottom (forced) | 3.01e-5 | 2.35e-11 | 1e-8 | 2.3e-11 / 2.3e-11 |

np 1 `initMpi` one V-cycle memcmp-equal to single rank on every case. FP32 V-cycle (forced 'fp32', host; the CUDA
row comes with the CUDA battery log): thinW a 3.6e-10, PCG 10 (not capped). Guard (production PCG, rhsOf, rtol 1e-10):
thinW 10 (LR off 13; bound 16), slab 11 (bound 14); trilinear 126 / 89 / cap 500. Host verdict: **PASS**.
The channel case (sizes, BCs) is this package's choice; G-SYM names only "an outflow-anchored channel".

**G-BIT** (switch off ≡ mglr): PENDING — `gpu_diag/mgd/sym/s1_gate.sh` (detached, under the heavy lock): per backend
`gbit.sh` (ctest -LE bench -V -j4; state_hash np1+np2 twice, digest as `validation/wo7/gate.sh`; printed values of
`bottom_direct_mpi`, `graphamg_mpi`, `telescope*_mpi`, `decomposition_policy`, `pcg_breakdown` vs mglr's gate log)
and prof.py dumps of the unmodified build (cb981c9, `sym/ref/mod_base_*`) vs f81ecbd (`mod_s1_*`), `gcmp.sh`.
Pass = battery green (272 = mglr's 268 + 4), digests host 51fec031a160c88b / CUDA 8337752781faf71b both runs,
printed identical, `GBIT PASS` from gcmp. Verdict lines: `sym/logs/s1_gate.out`.

**WO-S2, G-SYMCONV** (production `solvePCG` / `solveFCG`, rtol 1e-10, zero guess, rhsOf, closed cells zero, cap 500,
host single rank, one hierarchy per row, switch off then on; `sym/symconv/judge.out`; (d) bound = 1.25 C + 1 with
C the matrix's constant-P PCG):

| row | off PCG / FCG | on PCG / FCG | (a) no cap | (b) on ≤ min(off)+1 | (c) \|FCG−PCG\| ≤ 1 | (d) |
|---|---|---|---|---|---|---|
| thinW D5 LR off | 89 / 17 | 13 / 13 | PASS | PASS | PASS | 13 ≤ 16 PASS |
| thinW D5 LR on | 126 / 12 | 10 / 10 | PASS | PASS | PASS | 10 ≤ 13.5 PASS |
| slab D5 | cap / 14 | 11 / 11 | PASS | PASS | PASS | 11 ≤ 14.75 PASS |
| 64³ blob ×100 D4 | 30 / 12 | 10 / 10 | PASS | PASS | PASS | 10 ≤ 13.5 PASS |
| 64³ blob ×100 D5 | 30 / 12 | 10 / 10 | PASS | PASS | PASS | PASS |
| 64³ walls blob ×100 D5 | 27 / 12 | 10 / 10 | PASS | PASS | PASS | PASS |
| 64³ blob ×10⁴ D5 | cap / 143 | 10 / 10 | PASS | PASS | PASS | PASS |
| cylinder D4 / D5 | 9 / 9 | 9 / 9 | PASS | PASS | PASS | 9 ≤ 12.25 PASS |
| staircase sphere D5 | 9 / 9 | 10 / 10 | PASS | 10 ≤ 10 PASS | PASS | 10 ≤ 13.5 PASS |
| 128×128×25 D5 | cap / 57 | 29 / 29 | PASS | PASS | PASS | 29 ≤ 32.25 PASS |
| 128×128×25 D6 | cap / 71 | 28 / 28 | PASS | PASS | PASS | 28 ≤ 34.75 PASS |
| 128³ swarm D5 | cap / 247 | 85 / 85 | PASS | PASS | PASS | 85 ≤ 101 PASS |
| **128³ swarm D6** | cap / 249 | **80 / 83** | PASS | PASS | **MISS (3)** | 80 ≤ 103.5 PASS |
| uniform 128³ D5 / D6 | 8 / 8 | 8 / 8 | PASS | PASS | PASS | 8 ≤ 11 PASS |
| uniform 256³ D6 / D7 | 8 / 8 | 8 / 8 | PASS | PASS | PASS | 8 ≤ 11 PASS |
| uniform 256³ walls D7 | 9 / 9 | 9 / 9 | PASS | PASS | PASS | 9 ≤ 12.25 PASS |

The miss reproduces (rerun 80 / 83); the V-cycle on that hierarchy is symmetric, a(on) = 2.6e-18 (D5: 5.5e-18,
85 / 85). The matrix's replica (40-sweep symmetric bottom) had 82 / 82 there; the production bottom is 6 + 6.
**Verdict: G-SYMCONV MISS on (c), row 128³ swarm D6** (all else PASS). Per the brief, WO-S2 stops here: G-ITER-S,
G-PERF-S, G-CONV-S, G-NP and G-CONTRAST NOT RUN (tooling ready: `ONLY_LEVELS=0 SYM_LEVELS=0 ROUNDS=3 DUMP=1
run_matrix.sh … M1 M2 M3s M5 M7a M7b M8 M10 M10h M10v`). Nothing re-thresholded; escalated.
