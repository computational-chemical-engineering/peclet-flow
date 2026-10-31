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

**G-BIT** (switch off ≡ mglr; `gpu_diag/mgd/sym/s1_gate.sh`, per backend `gbit.sh` + prof.py dumps of cb981c9
(`sym/ref/mod_base_*`) vs f81ecbd (`mod_s1_*`) compared by `gcmp.sh`; verdicts `sym/logs/s1_gate.out`):
host 272/272 (-LE bench), state_hash np1 + np2 rc 0/0 twice, digests 51fec031a160c88b(14) both runs, printed values
identical, dumps `GBIT PASS` (fail=0 both runs); CUDA 272/272, digests 8337752781faf71b(14) both runs, dumps `GBIT PASS`;
its printed values differ in 4 lines, all OpenMPI `Process name: [prterun-…@1,k]` lines of a warning block in
`bottom_direct_mpi_np{2,4}` (a PID), no value. **G-BIT PASS** both backends.
**G-SYM CUDA** (`sym/logs/cuda_s1.ctest.log`, `mg_vcycle_symmetry`): a symmetric thinW 1.7e-17, LR off 2.1e-17, slab
2.2e-17, `scalar` 3.8e-18, 64³ blob D5 5.7e-19, walls 0, cyl 2.7e-18, channel 6.8e-18 (bound 1e-14); agglomerated
1.5e-11 (1e-8); FP32 V-cycle (forced 'fp32') thinW a 1.15e-9, PCG 10 (not capped); guard thinW 10 / LR off 13, slab 11.
**G-SYM PASS** on CUDA (host above).

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

**(c) diagnosis** (coordinator ruling 2026-10-26: the MISS stays on record, WO-S3 blocked; `symconv <case> <D> 1 <bottom> 1`,
host loop MG-CG with the production symmetric V-cycle as M, Fletcher–Reeves β (solvePCG's) vs single-direction
Polak–Ribière β (solveFCG's), same M, from x = 0; `sym/symconv/diag_b{12,40}.txt`). 128³ swarm D6:
- bottom 12 (6 + 6, production): FR 80, PR 83 (= production 80 / 83). |β_PR − β_FR| / β_FR ≤ 1.5e-12 on every iteration
  (typically 1e-13: round-off, ~eps·κ level, no O(1e-6) structure); residual histories agree to 3 digits through it 72
  (|r|/r0 1.5e-8), first > 1 % split at it 77 (|r|/r0 8.4e-10 vs 8.5e-10), i.e. at the stop level, where |r|inf
  oscillates in 1e-10…4e-9 (non-monotone max norm); FR lands below 1e-10 at 80, PR at 83.
- bottom 40 (20 + 20, the replica's): FR 82, PR 82; |Δβ|/β ≤ 2.2e-12; split at it 66 (|r|/r0 2e-8), both stop at 82.
Reading (for the architect, not a verdict): the 3-iteration gap is where two round-off-separated recurrences cross a
non-monotone max-norm stop level, a stopping draw like A3.6's, not a defect of M (a(on) 2.6e-18).

**Remaining WO-S2 gates (measurement only, ruling above), switch off vs on, one build (f81ecbd).** Data:
`gpu_diag/mgd/sym/s2_campaign.sh` → `sym/s2/` (A matrix rows, L0 off vs L0sym, 3 interleaved rounds of 5 + 30 steps,
dumps round 1; B traced runs (MG_DEBUG=2, debug output only) at the production rtol, 1e-13, 1e-17; C G-NP on a local
default-on worktree `suite/flow-mgsym-on`, removed afterwards; D bubble column 50 steps PCG on) and the corrections
`sym/s2fix/s2fix.sh` (below). Judge `sym/s2fix/s2_judge.py` → `sym/s2fix/judge.out`. 66/66 timed runs rc 0.

Two campaign defects, corrected before judging: (i) `mgd_case.py --rtol` never reaches the Chebyshev driver of
`--case swarm --driver default` (M7a, M10h): the campaign's 1e-13 / 1e-17 traces of those rows ran at the default
rtol, so (2) / (2′) there were re-run through `s2fix_case.py --cheb-rtol R` (set_pressure_chebyshev(True, cap, R)
after the build; cap 120 at 1e-13, 300 at 1e-17); (ii) the queued G-CONTRAST case, the bubble column, has
ρ_g/ρ_l = 0.02 (ratio 50), not 10³: re-run as the 128³ swarm with `--rho-ratio 1e-3` (μ ratio, Ar, Bo unchanged),
MG-PCG cap 800 rtol 1e-8, 50 steps, on and off. The PCG rows' (2′) runs used mgd_case's cap 800, not A3.6's 300:
no step reached 300 (max 36), so the difference is moot.

| row | iters off → on (≤ 1.10 off + 0.5) | ms off → on (median of 3) | G-PERF-S | G-CONV-S (b) max rel (≤ 100 rtol) |
|---|---|---|---|---|
| M1 swarm 192³ PCG, D5 | 9.100 → 8.000 PASS | 83.79 → 78.55 (0.938) | ≤ 1.02 PASS (load ≤ 15.4) | 1.2e-10 PASS |
| M2 bubble column 256×192×128, D5 | 11.900 → 9.133 PASS | 94.13 → 82.90 (0.881) | ≤ 1.02 PASS (13.5) | 2.4e-10 PASS |
| M3s zh_sphere N 48, D4 | 6.507 → 6.393 PASS | 8.38 → 8.24 (info) | — | p 0.255 **MISS as written**; u v w 6.6e-12; p on non-solid cells 5.1e-11 |
| M3s random_spheres N 48, D4 | 7.317 → 7.133 PASS | 9.63 → 9.30 (info) | — | p 0.286 **MISS as written**; u v w 1.9e-11; p non-solid 4.2e-10 |
| M5 host swarm 128³ 1×6 (G-PERF-H) | 9.000 → 7.900 PASS | 514.11 → 387.45 (0.754) | ≤ 1.03 PASS (17.7) | 1.3e-10 PASS |
| M7a swarm 192³ Chebyshev | 11.967 → 11.000 PASS | 228.61 → 216.51 (info) | — | 1.1e-11 PASS |
| M7b porous bed 128³ | 9.000 → 8.100 PASS | 30.15 → 29.14 (info) | — | 8.3e-14 PASS |
| M8 channel np 2, V-cycles | 8.000 → 8.000 PASS | 247.04 → 239.02 (info) | — | 2.2e-14 PASS |
| M10 128×128×25 PCG | 8.100 → 8.100 PASS | 16.72 → 16.49 (info) | — | 3.4e-10 PASS |
| M10h host, Chebyshev | 10.033 → 11.000 PASS (≤ 11.537) | 180.28 → 158.74 (info) | — | 1.2e-11 PASS |
| M10v host np 2, V-cycles | 9.000 → 10.000 PASS (≤ 10.400) | 246.54 → 230.77 (0.936) | ≤ 1.03 PASS (16.3) | 1.2e-13 PASS |

Iterations identical across rounds on every arm; 0 failed solves, 0 FP32 restarts, 0 Chebyshev restarts, both arms.
M5's rounds spread (off 503/553/514, on 462/387/365 ms): PASS on the medians, with a margin far beyond the spread.
K (zh_sphere) 7.436078194 both arms (rel 7.6e-15), k* (random_spheres) 0.006282103323 (rel 7.9e-14): PASS.
The M3s p MISS lives entirely in solid cells (every face velocity 0; 22272 / 18595 cells, |p| there ≈ 1e-2 in BOTH arms):
pressure in closed cells is undetermined and the two prolongations put different values there. On cells with an
open face the difference is 5e-11 / 4e-10. G-CONV (b) as written compares p over all cells; WO-2 never met it (N 48
was byte-identical). Not re-thresholded: **MISS as written**, for the architect.

**A1** (B traces; per step the last solve's r0 and rtol from the trace, div = `max_open_divergence_projected()`):

| row | (1) max div/(rtol r0), off / on | (2) 1e-13 worst div off / on, ratio (≤ 2) | (2′) 1e-17 worst div off / on, ratio (≤ 4); validity |
|---|---|---|---|
| M1 | 0.583 / 0.368 PASS | 1.50e-15 / 1.41e-15, 0.940 PASS | 3.33e-16 / 3.33e-16, 1.000 PASS; 30/30 valid both |
| M2 | 0.988 / 0.984 PASS | 5.38e-15 / 2.50e-15, 0.464 PASS | 3.89e-16 / 3.89e-16, 1.000 PASS; 30/30 |
| M5 | 0.836 / 0.745 PASS | 1.30e-15 / 8.76e-16, 0.672 PASS | 3.89e-16 / 3.47e-16, 0.893 PASS; 30/30 |
| M7a (Chebyshev, s2fix) | 0.967 / 0.502 PASS | 1.87e-15 / 1.73e-15, 0.926 PASS | 3.33e-16 / 3.33e-16, 1.000 PASS; 30/30 |
| M10 | 0.986 / 0.961 PASS | 2.35e-15 / 1.36e-15, 0.580 PASS | 3.61e-16 / 3.61e-16, 1.000 PASS; 30/30 |
| M10h (Chebyshev, s2fix) | 0.985 / 0.850 PASS | 5.22e-16 / 8.60e-16, 1.648 PASS | 3.33e-16 / 3.33e-16, 1.000 PASS; 30/30 |

(2′) iterations max 36 (M2 off), no cap. Not traced: M7b, M8, M10v (A1 NOT JUDGED there).
**G-CONV-S: A1 (1) PASS, (2) PASS as written, (2′) PASS; (b) PASS on 9 rows, MISS as written on M3s (solid-cell p
only); physics K / k* PASS; Hysing rise and static drop NOT RUN** (no instrument in this campaign).

**G-NP** (default-on build, the one-line flip `symVcycle_ = true`, never committed): host and CUDA 20/20 —
`mg_stencil_halo_np{1,2,4}`, `telescope_mpi_np{1,2,4}`, `telescope_varrho_mpi_np{1,2,4}`, `bottom_direct_mpi_np{2,4}`,
`mg_fp32_vcycle_mpi_np{1,2,4,8}`, `mg_vcycle_symmetry(_mpi_np{1,2,4})`, `device_sync_budget` (0 global fences,
0 allocations per step). **PASS.** Halo exchanges per V-cycle: no counter; D − 1 fewer by construction (ruled).

**G-CONTRAST** (`sym/s2fix/contrast3_{on,off}.json`, swarm 128³ ρ ratio 10³, MG-PCG, 50 steps): switch on 11.76
iterations/step mean, max 14, 0 capped, div_max 2.1e-8; switch off (control) mean 229.4, 13 of 50 steps capped at
800, div_max 7.2e-2. **No capped solve with the switch on** → per A3.5 the CLAUDE.md contrast bullet is rewritten
to the measured result at WO-S3. (The campaign's bubble column, ratio 50: on 9.28, max 12, 0 capped — information.)

**Verdict WO-S2 (measurement):** G-BIT PASS, G-SYM PASS (host, CUDA), G-SYMCONV MISS (c) (on record, diagnosed),
G-ITER-S PASS (11 rows), G-PERF-S PASS (M1 0.938, M2 0.881, M5 0.754, M10v 0.936), G-CONV-S PASS except (b) on M3s
as written (solid-cell pressure) and Hysing / static drop not run, G-NP PASS, G-CONTRAST: no cap.
Not re-thresholded; WO-S3 stays blocked (coordinator).

## 2026-10-29 — Amendment A3: G-CONV (d) / G-CONV-S physics metrics, Hysing rise and static drop, switch on vs off

Before WO-S3 (user condition (ii) on U-A3). Instrument: `tests/study/vof_surface_tension.py` copied to
`gpu_diag/mgd/sym/phys/vst_sym.py` (diff `vst_sym.diff`: `--sym on|off` sets
`diagnostics.set_pressure_vcycle_symmetric` on every Solver, metrics dumped at full precision; nothing else).
CUDA build `build_cuda` (f81ecbd source), default resolution (static 16/32/48/64³; Hysing nx 64, T 3), Chebyshev
driver at the script's rtol (static 1e-14, Hysing 1e-12), one arm each, OMP 4, under `heavy.lock`.
Tolerance: 1e-6 relative — the script and its docstring state no tolerance of their own for these metrics.
Data `sym/phys/logs/{static,hysing1,hysing2}_{off,on}.{log,json}`; table from `sym/phys/judge.py`.

| metric | off | on | rel | verdict |
|---|---|---|---|---|
| static Ca, 16³ / 32³ / 48³ / 64³ | 2.5520e-4 / 5.9173e-5 / 2.8138e-5 / 1.4471e-5 | same | 4.7e-14 / 3.7e-14 / 1.8e-13 / 1.7e-12 | PASS |
| static Δp vs Laplace 2σ/R | — | — | — | NOT JUDGED (the script does not measure it) |
| Hysing 1 v_rise max | 0.24291129853633 | 0.24291129853633 | 2.3e-16 | PASS |
| Hysing 1 t(v_rise max) | 0.95691509308147 | equal | 0 | PASS |
| Hysing 1 y_c(3) | 1.08409344631311 | 1.08409344631311 | 2.0e-16 | PASS |
| Hysing 2 v_rise max | 0.25223606197221 | 0.25223606197221 | 3.7e-15 | PASS |
| Hysing 2 t(v_rise max) | 0.75979564297055 | 0.75979564297055 | 5.7e-15 | PASS |
| Hysing 2 y_c(3) | 1.12584606332378 | 1.12584606332372 | 5.5e-14 | PASS |
| Hysing 1, 2 circularity min | — | — | — | NOT JUDGED (the script does not compute it: no PLIC length) |

Supplementary, not pre-registered: Δp = mean p(C > 1−1e-6) − mean p(C < 1e-6) after 60 steps is bit-equal on and off
(0.5226 / 0.2521 / 0.1672 / 0.1252 against 2σ/R 0.5 / 0.25 / 0.1667 / 0.125).

**Stop test, both arms: no capped solve** (cap 500 static, 600 Hysing). Pressure iterations per step, mean (max),
off → on: static 16³ 12.00 (12) → 11.93 (12), 32³ 13.00 (13) → 11.98 (12), 48³ 13.00 (13) → 12.05 (13), 64³ 13.00 (13)
→ 11.98 (12); Hysing 1 14.46 (17) → 15.32 (18), 2032 steps both; Hysing 2 20.88 (53) → 20.28 (121), 856 steps both —
the on-arm max is a two-step transient (steps 718–719: 104, 121; next highest 37), off's max is steps 625–626 (45, 53).
max |div(open u)| off / on: static ≤ 1.8e-15 / 3.2e-16; Hysing 1 5.2e-13 / 4.5e-13; Hysing 2 4.2e-11 / 4.8e-11.

**Verdict: G-CONV (d) / G-CONV-S physics PASS on every metric the instrument measures (max rel 1.7e-12, static 64³
Ca); static Δp and circularity NOT JUDGED (not measured by the script).** Not re-thresholded; WO-S3 not started.

## 2026-10-29 — WO-S3 gate mgsym3 triage (flip 4175de8): four re-pins, CA device default, one STOP (Chebyshev bounds)

Gate mgsym3 (`validation/wo7/{host,cuda}_mgsym3.*`): host 6/274 failed, CUDA 5/274; the summary lists also carry
skips. Every entry, classified:

| test | backend | class | evidence | action |
|---|---|---|---|---|
| device_sync_budget, prefactor_hoist_bitwise | host | skip by design (exit 77, device-only) | skipped on main host_mglr2 too; CUDA mgsym3 Passed | none |
| bottom_direct_mpi_np2/np4 | CUDA | skip by design | skipped on main cuda_mglr2/2b too; host mgsym3 Passed | none |
| scalar_mg, scalar_cutcell_g4 | CUDA | (c) flake: GPU OOM under a shared GPU (`Cuda memory space failed to allocate 17.55 MiB` / `3.092 MiB`) | same two failed with the same OOM on main cuda_mglr2; serial CUDA rerun on c6c0494: both Passed | none |
| units_anisotropic_sphere | both | (a) pinned iters/step | 7.0 → 6.0 at N = 24, 32 (N = 16: 6.0); K and steps unchanged | 41a7257 re-pin |
| mg_fp32_vcycle | both | (a) pinned relative criterion | G-D3 (e) fp64 health 0.06–0.44 (trilinear) → 3e-15–3e-13 (symmetric: round-off); fp32 2e-8–1.6e-7 (FP32 rounding) | 5f59812 floor max(1.1×fp64, 1e-2) (G-D3's original absolute) |
| vof_collocated | both | (a) stopping draw at rtol 1e-9 | T1 walled step 1 face 6.6e-13 → 1.53e-12 (bound 1e-12); at 1e-14 5.1e-14 (cell 1.2e-13, dP/dz 2.3e-14) | b5efbcf walled at 1e-14, the periodic case's protocol |
| vof_twophase_mpi_np2/np4 | host | (a) stopping draw; the test's premise "|u| is round-off" | walls-z max|u| 2.7e-12 → 1.6e-10 at rtol 1e-9, du 4.2e-12 / 5.2e-12 (tol 1e-12); at 1e-12 and 1e-14 both arms |u| ≈ 3e-13 | 9c01268 walls-z at rtol 1e-12: du 4.4e-13 / 3.9e-13 |
| pressure_wallbounded | host (CUDA symptomatic) | (b) **Chebyshev bounds do not bracket the symmetric V-cycle — STOP, design question** | case B Cheb 7 → 200 (cap), max|u| 4e84 host; CUDA 7 → 24 (passes, same mechanism) | none (below) |

**pressure_wallbounded case B, root cause (measured, host, diagnostic prints not committed).** The symmetric V-cycle's
λ_max(M⁻¹A) is above 1 (constant P with rediscretized coarse operators over-corrects): 300 power iterations from a
rich seed give 1.0566 on this 24×24×16 D4 box (trilinear 0.9999); vof_collocated 1.059; the varRho cases C/D's warm
estimates climb 1.02 → 1.12 over 8 steps (trilinear ≤ 1.010). The Chebyshev driver brackets with 1.05 × the
15-iteration estimate seeded by the right-hand side. Case B's rhs (uniform body force, walls ±z, periodic x, y) is
x-y-uniform, an invariant subspace of M⁻¹A, so the estimate sees only the 1-D spectrum: [0.0194, 0.9714] → b = 1.020
< 1.0566; constant density never re-estimates, round-off seeds the other modes and they grow (200 iterations,
|u| 4e84). Trilinear: the 1-D estimate 0.967 × 1.05 = 1.015 brackets 0.9999, so the same trap was latent. A rich seed
alone (15 iterations) gives b = 1.039 and 12 iterations: still under λ_max. The A3 note puts Chebyshev in scope and
is silent on its bounds. Options for the decision: (i) a decomposition-independent mean-free pseudo-random component
in the cold seed (moves every Chebyshev digest); (ii) a larger upper margin in symmetric mode, or a floor on b;
(iii) a residual-growth guard on cold-bound solves (re-estimate) — needs (i) to help. Default if asked: (i) + (ii).

**CA device default (c6c0494)**: `sym/flip/ca_patch.py` applied as one commit. Host state_hash b5054ad271849e15
(= mgsym3); CUDA 1c59f91ccf487086 (= mgsym3): bitwise. ca_odd_periodic_mpi + mg_stencil_halo host 5/5, CUDA 5/5; the re-pinned four CUDA 6/6 (walls-z du 0 / 6.6e-13). clang-format clean on the CI-covered
files (flow_ibm.hpp, flow_bindings.cpp are on the CI exclude list).

**G-NP (mgsym3 logs, switch on):** host 19/19 run green (device_sync_budget skip by design), CUDA 18/18 run green
(device_sync_budget PASS; bottom_direct_mpi skip by design, host PASS): every one of the 20 green on its backend.

**Not done (blocked on the Chebyshev decision):** the full gate re-run, the census vs mglr2 and the 14-digest table
(gate mgsym3 digests: host b5054ad271849e15, CUDA 1c59f91ccf487086; pre-flip 51fec031a160c88b / 8337752781faf71b;
a Chebyshev-bounds change would move the VoF/Chebyshev cases again).

## 2026-10-29 — Chebyshev bounds under the symmetric V-cycle: (i) seed + (ii) upper margin measured; STOP on (ii)

Coordinator decision: (i) an np-independent pseudo-random component in the cold eigen seed, (ii) an upper margin
chosen from measurement. Measured on host (diagnostic build, not committed; evidence and the WIP patch in
`gpu_diag/mgd/sym/flip/cheb/`). (i) = rhs + max|rhs| · h(global cell), h a splitmix64 hash in [−0.5, 0.5), gated on
the symmetric mode (off stays the pre-A3 estimate bit for bit). Reference λ_max(M⁻¹A): 150 power iterations from the
broadband seed (CG-Lanczos agrees within 2 % wherever M is symmetric; it is invalid for the trilinear V-cycle).

**Underestimate ratio λ_max / estimate, worst over the -LE bench battery (53 Chebyshev ctests), with (i):**
symmetric: cold 1.129 (vof_blocks_ns_mpi), 1.126 (vof_momentum), 1.111 (pressure_wallbounded C), 1.100
(vof_redistribute_mpi), 1.099 (balanced_force_mpi); warm 1.122. Trilinear: 1.041 (inside its 1.05). So a
bracketing margin needs ≥ 1.13 total: kChebSymUpper ≥ 1.076 on top of solveChebyshev's 1.05; 1.15 gives 7 %
headroom.

**Cost, Chebyshev V-cycles summed over the 53 tests:** trilinear 237 889; (i) only 239 557; (i) + factor 1.15
on λ_max (bracketing, total 1.21) 260 757 (+9.6 % vs trilinear; per test +5…+29 %, collocated_stability_guard
+29 %, cell_force_placement +22 %, vof_twophase +14 %); 1.20: 269 535 (+13.3 %); 1.25: 279 714 (+17.6 %).

**Mechanism:** Chebyshev converges for every λ in (0, a + b), so an unbracketed top costs little (the (i)-only arm
is within 1 % of trilinear). The failure (pressure_wallbounded B) is the LOWER bound collapsing: the min iteration
is shifted by the unmargined λ_max estimate, and a shift below λ_max makes it converge to the wrong end (estimate
0.0194 / 0.16 against a true 0.90), so a + b < λ_max. Applying the factor to the SHIFT only and returning the raw
λ_max ("shift-only"): total 232 854 at 1.15 (−2.1 % vs trilinear), 232 573 at 1.25; case B 8 iterations (trilinear 7;
mgsym3 200 = cap; (i) only 34).

**Verdict:** (ii) as decided (bracket the top) costs about 10 % of the Chebyshev V-cycles (the VoF default driver)
and would likely break G-ITER-S (≤ 1.10 × off + 0.5): STOP per the decision's own condition. Not committed. The
shift-only variant fixes the measured failure at no cost but is a different design choice: it goes to the
coordinator / architect. In every symmetric arm vof_collocated (lines 241, 257: exactness at the default rtol 1e-9)
fails on the changed stopping draw; it passes at factor 1.25 bracketing. Not triaged further.

## 2026-10-29 — Chebyshev bounds landed ((i) + shift-only), vof_collocated triage, full gate mgsym4

Coordinator decision ((i) + shift-only, supersedes the (i) + (ii) line). **c54a828** (production only): cold eigen
seed = rhs + max|rhs| · h(global cell) (splitmix64 of the global index, np-independent), symmetric mode only; the min
power iteration is shifted by 1.15 × the λ_max estimate (`kChebSymShift`), λ_max returned raw. Switch off: shift =
λ_max, no noise — the pre-A3 estimate bit for bit (porous / non-Chebyshev hashes unchanged, below). Chebyshev
V-cycles over the 53 Chebyshev ctests 232 854 (trilinear 237 889, −2.1 %; = cb63826's shift-only arm exactly);
pressure_wallbounded B 8 iterations (mgsym3: 200 = cap).

**Unit check, lmin estimate within 20 % of the true lmin** (host, diagnostic layer not committed:
`gpu_diag/mgd/sym/flip/cheb/prod_shiftonly_with_diag.patch`). True lmin = 300 shifted-power iterations from a
broadband seed with re-projection each iteration. CG-Lanczos agrees to 1e-4 on pressure_wallbounded / vof_momentum
/ balanced_force_mpi, but gives spurious 0.10–0.28 on vof_blocks_ns_mpi and vof_redistribute_mpi (the earlier
table's "est_lo/lanczos_lo 8.3" there is the Lanczos reference, not the estimate). Worst row per case:

| case | est lmin | true lmin | est/true |
|---|---|---|---|
| pressure_wallbounded B (cold) | 0.9223 | 0.9018 | 1.023 PASS |
| pressure_wallbounded C (cold; warm 1.005–1.015) | 0.9250 | 0.9028 | 1.025 PASS |
| vof_blocks_ns_mpi_np1 | 0.7105 | 0.7136 | 0.996 PASS |
| vof_momentum | 0.8187 | 0.7795 | 1.050 PASS |
| vof_redistribute_mpi_np1 | 0.8903 | 0.8637 | 1.031 PASS |
| balanced_force_mpi_np1 | 0.9338 | 0.9048 | 1.032 PASS |

**vof_collocated lines 241 / 257 (face ≤ 1e-12), class (a) stopping draw — 921b778.** Default rtol 1e-9: symmetric
face 1.99e-12 in 10 iterations (mu sweep 1.27–1.99e-12), trilinear 2.25e-13 in 13. At 1e-14 (the test's G4
protocol, as b5efbcf): 8.2e-14 in 15 iterations, mu sweep 2.0e-14–1.1e-13; cell / dP/dz transient unchanged
(2.862e-04 / 4.180e-03). Bounds unchanged; the periodic case stays at the default rtol.

**Gate mgsym4 (`validation/wo7/{host,cuda}_mgsym4.*`, head 921b778):** host 274/274, CUDA 274/274 (0 failed; skips
by design only: host device_sync_budget / prefactor_hoist_bitwise, CUDA bottom_direct_mpi_np2/np4; the OOM flakes
did not recur). **PASS.**

state_hash (14 cases; first 12 hex, pre-flip → mgsym3 → mgsym4): 13 move with the flip, `porous` unchanged on both
backends; the Chebyshev change moves `vof_droplet` only (the one Chebyshev case):

| case | host pre → mgsym4 | CUDA pre → mgsym4 |
|---|---|---|
| channel | d2512e2c3486 → f4a5922cbeab | 4fd52a6be32f → 1178dcd06774 |
| colocated_advect | 89324a797ea8 → feda0b6bff0d | 739a644d6bab → 99f52abf0a9c |
| colocated_advect_bc | 5df7c013c9db → 27ce08217e47 | 44c2be37d16f → bb2feb5f60c3 |
| colocated_bfp | d234ba65950a → f483970b809f | b2b5f6188bba → a893c10cf563 |
| colocated_embed | d22cd2f07ae5 → 785fb19576c9 | d3224126afbc → 21f4c1e32b25 |
| colocated_gauge_exact | bf0698145f75 → e2a657b264fc | f51471270a07 → d1dc25455101 |
| colocated_ghost | 4b2762dc018c → 927b9278b85c | ba618738445d → b2a1b68a38b0 |
| colocated_plain | 6d61f3ef3f47 → ed983cffe773 | b9e4ccc87102 → 72a74eba5770 |
| mpi_np2 | 1ca4ff4d85ae → 1b676469c56d | 637f7ebdc30f → eab10d6b39d3 |
| porous | 8226b7ee9fe6 (same) | 7ca6f39d89e7 (same) |
| scalar | b9fc57440bb6 → f21301301875 | 27445fd86a99 → 8ee60cf84562 |
| scene_moving | f23c8c5a666b → b3f9b0fd48ef | 14fd3b6138f2 → c074c0318bc1 |
| staggered_bed | d12cfa34e39d → e2adb7800aaa | 56b52c908927 → ab98c36f2539 |
| vof_droplet | 41a51923c28e → 70fc2c7ea7b0 (mgsym3 bf7fd5249fdb) | d12105490c40 → e5e4a0a5508a (mgsym3 fb494755dbed) |

Digests (sha of all 14 lines): host 51fec031a160c88b → **f36332b3a2b1c46d**, CUDA 8337752781faf71b →
**81ed80eebb3c905a**.

**Census** (`sym/flip/compare_name.py`, printed output per test): host mglr2 → mgsym4 126 of 270 common tests print
differently, CUDA mglr2b → mgsym4 124 of 270 (noise set cuda_mglr2 vs mglr2b: 8); every status unchanged or
flake → Passed; 4 tests new since main's logs. mgsym3 → mgsym4: host 49, CUDA 50 — the Chebyshev / VoF set
(balanced_force*, vardensity*, vof_*, units_vof_*, pressure_wallbounded, cell_force_placement, multiphysics_mpi),
the four re-pinned tests, and the run-to-run printers of the noise set.

**G-NP** (mgsym4 logs): host 19 Passed + device_sync_budget skip by design, CUDA 18 Passed + bottom_direct_mpi_np2/4
skip by design (host Passed): 20/20 green on their backends. **PASS.**

**G-ITER-S on the Chebyshev rows** (`sym/flip/giters4.out`, --levels 0, 30 steps, off vs on, this build): M7a swarm
192³ 11.967 → 11.000 PASS; M7b porous bed 9.000 → 8.100 PASS; M10h 10.033 → 11.000 PASS (≤ 11.537); 0 failed
solves, 0 FP32 / Chebyshev restarts. Iterations equal to WO-S2's; M7a div_max 9.29e-12 → 8.21e-12.

**CA device default c6c0494 bitwise on top:** CUDA state_hash with c6c0494 reverted in the working tree
81ed80eebb3c905a = with it 81ed80eebb3c905a. **PASS.**

## 2026-10-30 — A3 review fixes (fa1303c … 02848cf): two STOP findings on the Chebyshev bounds; CA np 2/4 bitwise

Measured on host at 164384d (`bubble_column_perf/gpu_diag/mgd/sym/flip/cheb2/`: arms over the 53 Chebyshev ctests, a
temporary diagnostic injected and reverted; trip diagnostic `cheb2/tripdiag.log`, per-test sums `cheb2/pertest.py`).
Gate mgsym5 NOT run: both findings below are design questions.

**STOP 1 — safety ratio s = (0.95·λmin_est + 1.05·λmax_est)/λ_true** (Chebyshev converges for λ < a + b, so s ≥ 1
brackets; the bar is 1.05). Symmetric mode: cold min **0.9997**, warm min **1.0035**, both at collocated_stability_guard
(est [0.0448, 0.9953], λ_true 1.0879: the 1a A-norm Rayleigh quotient still sees λmax ≈ 1.00 of 1.088). Every other
test ≥ 1.4189. Trilinear: warm min 1.0918 (cold "0.0000" rows are zero-rhs estimates, not misses). **MISS.**

**STOP 2 — the growth guard (1b) is not inert on healthy solves.** Trip test `rn > r0` in the max norm at ANY iteration:

| test | cold solves | trips | trip iteration, rn/r0 | s (bounds bracket?) | verdict |
|---|---|---|---|---|---|
| vof_blocks_ns_mpi_np{1,2,4} | 6 | 4 | it 1, 1.55 | ≥ 1.53 (yes) | false positive: max-norm transient; redo b 1.050 → 1.207, 19 V-cycles |
| multiphysics_mpi_np{1,2,4} | 4 | 2 | it 1, 5.70 | ≥ 1.42 (yes) | false positive, same mechanism |
| collocated_stability_guard | 460 | 421 | it 3, 1.035–1.135 | 0.9997–1.0035 (no) | genuine underestimate, but it recurs every step |
| vof_collocated, pressure_wallbounded | 271, 20 | 0 | — (max rn/r0 0.37, 0.36) | yes | inert |

With correct bounds only the M⁻¹-norm of the Chebyshev residual is non-increasing; the max norm can rise (×5.7 at
it 1). In collocated_stability_guard the redo's bracketed bounds do not persist: the next warm estimate is again
unbracketed, guard3 trips (420 warm trips), the cold re-estimate trips again, redo — +839 solves. The "21 trips" of
`cheb2/run.out` are the once-per-process stderr notices (7 tests × 3), not the trip count.

**Chebyshev V-cycles over the 53 tests: 232 854 (mgsym4) → 253 081 (+20 227, +8.7 %)**: collocated_stability_guard
+13 305 (the trip loop above), vof_collocated +6 929 with 0 trips (1a's estimate: bounds moved, every solve on
them), vof_blocks_ns +58 ×3 and multiphysics +2 ×3 (false-positive redos), the rest ±≤ 112 (1a).

**Item 4 — CA device default at np > 1** (`sym/flip/item45.out`, CUDA, 8 steps, CA both vs off): periodic np 2, np 4,
walled np 2, np 4 all **BITWISE** (u, v, w, p, every rank); iterations identical. **PASS.**

**Item 5a** (02848cf): perf_baseline.json zh_sphere iteration fields re-recorded from a CUDA run (6.0 at every grid;
totals 366/494/485/959/1587); K kept (moved ≤ 3.3e-6 rel), steps unchanged.

## 2026-10-31 — A3 finish (60fac53, 88d196f, cf3a2d3, 1527238): both STOP findings closed; gate mgsym5 PASS

Fixes for the two STOPs of 2026-10-30 (decisions in `bubble_column_perf/gpu_diag/DECISIONS_GPU.log`, "A3 finish"
(a)(b)(c)). Measured on host over the 53 Chebyshev ctests (`gpu_diag/mgd/sym/flip/cheb3/run.sh c53`; logs
`cheb3/c53_{on,off,vc}.log`, trips per solve by `cheb3/trips.py`).

- **(a) 60fac53 — the growth guard tests ρ_k = r·z (the M⁻¹-norm), not max|r|**, lagged one iteration: a `dotTo`
  slot `kChebRz` and a pinned copy, read after the stop test's maxabs fence; one Allreduce of {max, sum}
  (`maxabsWithLocalSum`, pair MPI_Op). No new synchronisation; unguarded solves bitwise.
- **(b) 88d196f — bounds ratchet**: a growth trip's bracketed λmax persists (`chebLmaxFloor_`, `bfpLmaxFloor_`), so
  the next warm estimate cannot fall back below it. ctest `pressure_chebyshev_bounds` case 3 (hook
  `debugDropPressureBounds`). Negative control, ratchet off: estimates +2, restarts 1 → FAIL; on: +1, 0 → PASS.
- **(c) cf3a2d3 — symmetric-mode cold λmax = top Ritz value of ≤ 10 PCG steps** (stop at relative change < 1e-3);
  ColdKeep repeats the pass to build the top Ritz vector as the warm seed. λmin by the shifted power iteration
  (unchanged); trilinear and warm paths untouched.
- 1527238 — vof_collocated T2 at μ = 0 compares `co.faceU < max(st.faceU, 1e-16)`: both faces are round-off
  (staggered 7.4e-18, collocated 4.8e-17).

**Safety ratio s = (0.95·λmin_est + 1.05·λmax_est)/λ_true**, symmetric: cold min **1.0864**, warm min **1.0867**
(collocated_stability_guard, est 1.0850 vs true 1.0879); was 0.9997 / 1.0035. Trilinear warm min 1.0862. **PASS.**

**Trips per solve**, symmetric (on / vc arms): growth **0**, guard3 **0** over 18 131 solves (887 cold-guarded).
vof_blocks_ns and multiphysics 0 (were 4 and 2 per test with the max norm); the collocated_stability_guard loop is
gone (was 421 growth + 420 guard3). (a) alone removed the growth trips there; the 420 guard3 trips remained until (c).

**Chebyshev V-cycles over the 53 tests: 232 854 (mgsym4) → 240 206 (+3.2 %).** Movers > 1 %:

| test | Δ V-cycles | note |
|---|---|---|
| vof_collocated | +7341 (+13.5 %) | 0 trips; the higher, now bracketing bounds |
| balanced_force | +147 (+1.0 %) | |
| vof_surface_tension_mpi_np* | +30 (+3.1 %) | |
| vof_blocks | +13 | 2 → 3 solves |
| vof_momentum_rho_names | +9 (+8 %) | |
| pressure_wallbounded | +6 (+1.6 %) | |
| vof_collocated_mpi_np* | −20 (−5 %) | |
| vof_blocks_ns_mpi | −48 (−3.7 %) | |
| vardensity_mpi | −17 (−2.4 %) | |
| vof_cutcell_mpi | −6 (−1.1 %) | |

**Trilinear guard (coordinator decision 2026-10-31): keep the r·z growth test in all modes, no code change.** M is
not symmetric in trilinear mode, so (a)'s monotonicity argument does not apply there and the test is a heuristic.
Measured (53/53 pass both): r·z guard 420 growth + 439 guard3 = 859 trips over 19 022 solves (vof_collocated 221/287
cold, collocated_stability_guard 64 + 430, balanced_force_restart 88); the old max-norm guard (temporary patch
`cheb3/trilmax_patch.py`, reverted) 541 + 436 = 977 over 19 160. The trilinear trips predate (a). Rejected: r·z only
when `symVcycle_`, max norm in trilinear (more code, more trips).

**Gate mgsym5 at 1527238** (`sym/flip/gate5.out`): host **275/275**, digest **805e3e99a93333c7**; CUDA **273/275**,
digest **e7edbb471ea7a742** — the two failures, scalar_cutcell_g3a/_g4, are GPU out-of-memory ("Cuda memory space
failed to allocate"), 2/2 Passed rerun serially (`flip/gate5_rerun.log`). Digests vs mgsym4: 13 of 14 identical on
both backends; only vof_droplet moved (host 70fc2c7ea7b0 → 289af88934b0, CUDA e5e4a0a5508a → dfc69aa6d5a3), as
expected. vs main: porous identical, the rest as mgsym4. CA np 2/4 periodic + walled: all bitwise. G-ITER-S: M7a
11.967 → 11.000, M7b 9.000 → 8.100, M10h 10.033 → 10.933 (≤ 11.537), growth trips 0: **PASS.** G-NP 20/20 (host 19 +
device_sync_budget skip; CUDA 18 + bottom_direct np 2/4 skip, as mgsym4). SDFLOW PASS. **Battery green.**
