# VoF step performance campaign — STATE (rewritten in place; history in vof_step_performance_log.md)

**Objective.** USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles; no GPU<->host transfers in the step. Yardstick: TBFsolver on the bubble column
(peclet-examples `benchmarks/bubble-column`, live page), same Snellius genoa node.

**Where we are (2026-10-16).**
- On flow main (all gated, register entries in suite/docs/decisions/flow.md): main line WO-0..5, 8,
  WO-7, §13 GPU 'direct' bottom (GPU default; RTX 5080 step 36-37 ms), WO-12, WO-9, WO-10, and now
  the **CPU §14 package** WO-H0..H4 (design §14), landed 2026-10-08 after Snellius S-1, rebased on
  5795fb0: H-0 one-rank benchmark protocol, H-1 host 'auto' -> 'direct' bottom where eligible
  (recorded numerics change, kept on the S-1 numbers), H-2 prolong, H-3(a)-(e) stragglers,
  H-4(a)-(e) host container kernels. Merge gate: CUDA bitwise to main (state_hash 12 + np2, 50-step
  dump), host bitwise with `--bottom-solver algebraic`; batteries 231/231 host and CUDA (4 GPU-OOM
  failures at -j6 on a shared GPU pass serially). CUDA now builds with nvcc 13.4 (system 13.2
  removed 2026-10-08: old CUDA build trees need a fresh configure).
- **S-1 (genoa, rtol 1e-8, 300 steps; jobs 27770798/27770799)**, ms/step direct / algebraic: 1x24
  contiguous 93.10 / 95.81; spread over 12 CCDs 83.81 / 89.38; active wait 94.86 / 96.22. Multi-rank
  (algebraic only until H-6): 8x3 per CCD 90.72, 6x4 106.53, 3x8 107.87. Generic build 97.95
  (znver4 5 % faster). TBFsolver 8x3 same node 45.45-46.04. Old published 144 ms (old protocol).
  kprof 1x24: 937 launches/step (F4 not triggered), projection 55.3 of 93.7 ms.
- **A(b) landed 2026-10-08** (flow 62f91ef, 591bde4): the host `mg_bottom_factor` runs a bitwise
  host schedule (same scalars, ~2.5 barriers per tile, vectorized lanes); factor bytes, U8, host
  G-BIT (OMP 1/8/24, `direct`) and CUDA identical to main. Workstation: 5.6-6.5 -> 1.05-1.5 ms per
  factor (kprof 1x24: 6.6 -> 1.5-1.8 ms/step). **Genoa confirmed** (job 27783375, base afbc8b6 vs
  d8cd5ca interleaved, 1x24 `direct`, N=100 x2): step 97.4-97.7 -> 80.9-81.1 ms (-16.6, -17 %);
  `mg_bottom_factor` 13.2 -> 1.56 ms/step. Genoa factor sweep (walls-y P12 b128): host T=6 1.08 ms
  best (T=2 1.51, T=8 1.20; team kernel 9-18 ms) -> `kBottomHostFactorTeam` 2 -> 6 landed with the
  projection package (1090b21, bitwise). Peclet 1x24 ~81 vs TBFsolver 46 before that package.
- **G (tier-3 curvature cost) LANDED 2026-10-09**: WO-1 kernels in core main 4806f87 (core tag
  NOT cut: a `v*` tag publishes to PyPI); on flow main WO-2a, WO-2c, WO-2b (bitwise host + CUDA) and
  WO-3 (moment cache + device entry-parallel accumulation, RECORDED: 50-step 2.7e-14 / 2.5e-14,
  iterations identical). The moment build and the team body are device-`__noinline__` so C1 stays
  bitwise on CUDA. RTX 5080 (shared): tier 3 3.55 -> 0.575 ms, curvature stage 6.19 -> ~3.3 ms.
  Genoa G-PERF (stage <= 4.5 ms) belongs in the next S-2 run. Numbers: log 2026-10-09.
- **Projection-cost package LANDED 2026-10-16** (flow 60126ec..191bba3: + the opt-in D `'auto'`
  2a59131 and the E0/X1 review fixes 191bba3, ctest `projected_guess`, state_hash `colocated_bfp`;
  design `doc/vof_projection_cost_design.md`, Outcome block): H (deep copies/step 11 -> 4), C0
  (4-lane host sums, recorded, host hashes moved once, <= 6.5e-14), C1-C4 (host Krylov fusion,
  -4.0 launches per PCG iteration), S (peeled host wrap rows, `cc_smooth` 1.35x per launch), F-1/F-2/
  F-2b (`bc_vel` 67.25 -> 59.0), X1 (resident PCG honours `stopRef_`), all bitwise except C0/X1.
  Launches/step 1164.2 -> 1061.7 (kp); projection kernels 1x24 76.1 -> 63.8 ms (min, phase 2).
  **Opt-in, defaults unchanged (USER 2026-10-15):** D, the FP32 V-cycle (`set_pressure_vcycle_precision`
  'fp64' default | 'fp32' | 'auto'; projection 0.811 on CUDA, 0.884-0.892 host; D5: MG arrays 119.3 ->
  71.1 B/cell under FP32) and E, the projected guess (`set_pressure_initial_guess` 'zero' default |
  'projected' | 'auto'; -0.83...-0.87 PCG its/step over 300 steps). Single-rank-only defaults would
  break serial <-> np=1 bit-exactness; flip only with an FP64 restart-on-cap + distributed FP32/E.
  Batteries 232/232 host + CUDA at 9cff2a0; 2a59131 / 191bba3 gated by G-BIT + targeted ctests.
- PARKED: E2(a) constant-coefficient driver; E2(b) FFT solver DROPPED 2026-10-08 (register).

**Next action.**
0. S-3 on genoa (`bubble_column_perf/s3/s3_proj.slurm`): package total vs the note's model (host
   -17.5 ms/step), C0's verdict (its per-launch gate was not met on the workstation: bandwidth-bound
   at 24 threads) and F4's t_L. Reviewer pass on E0 and X1 still owed (C0, D, G reviewed:
   `bubble_column_perf/gate_proj/REVIEW_C0_D_G.md`).
1. Gallery cost table on peclet-examples `benchmarks/bubble-column` from S-1 (re-render + commit the
   `_freeze`; check the publish job) — not done; Snellius numbers above, TBFsolver 45.5-46.0.
2. USER 2026-10-08: device A(b) wanted later — the device factor costs 7–9 ms per launch on the
   RTX 5080 (walls-y), out of a 36–37 ms step; same goal: bitwise-faster schedule first.
3. WO-H5 (Krylov reduction fusion on host), H6 (distributed host 'direct'), H7, then S-2 (incl. a
   64x3 and 24x8 profile), H8 (flow CLAUDE.md: host no longer "keeps GraphAMG").

**Open.** Projection follow-ups: S on `applyCutcellOpFaceWrap` and C1's fused `matvecDotTo` (the
PCG loop's level-0 matvec) not done; the first balanced-force solve of every X1 case caps at 500
(pre-existing); `set_pressure_warmstart` deprecation (Q-P5) proposed for the next API cycle.
A(a) (factor reuse across steps, a numerics change) not
designed. The DEVICE factor: next action 2. Q-H3:
factor T measured (genoa: 6 best, change pending); the solve kernel's `kBottomHostTeam = 8` not re-measured. Spread placement 10 % faster than contiguous:
publish contiguous, record spread alongside (Q-H5). The vof_momentum 1.84 acceleration seen once in
a D/h-16 pair run (not reproduced at D/h 20); PECLET_FLOW_OPERATOR_DOUBLE=OFF fails 6 tests on main
too (pre-existing); the 15-19 % swarm-drift difference to TBFsolver = near-contact treatment.

**Where things are.** Working data + gate scripts: /home/frankp/Codes/bubble_column_perf/ (README;
S-1 raw in `s1/`, merge gate `gate_s1merge.sh` / `gate_s1merge/`). Baseline worktree
../flow-main-base (detached at the pre-merge main 5795fb0). Snellius:
/projects/0/prjs1022/peclet/bubble-cpu (scripts, TBF build, results; S-1 in `s1/results/`).
