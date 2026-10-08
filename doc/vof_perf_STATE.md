# VoF step performance campaign — STATE (rewritten in place; history in vof_step_performance_log.md)

**Objective.** USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles; no GPU<->host transfers in the step. Yardstick: TBFsolver on the bubble column
(peclet-examples `benchmarks/bubble-column`, live page), same Snellius genoa node.

**Where we are (2026-10-08).**
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
- PARKED: E2(a) constant-coefficient driver; E2(b) FFT solver DROPPED 2026-10-08 (register).

**Next action.**
1. Gallery cost table on peclet-examples `benchmarks/bubble-column` from S-1 (re-render + commit the
   `_freeze`; check the publish job) — not done; Snellius numbers above, TBFsolver 45.5-46.0.
2. WO-H5 (Krylov reduction fusion on host), H6 (distributed host 'direct'), H7, then S-2 (incl. a
   64x3 and 24x8 profile), H8 (flow CLAUDE.md: host no longer "keeps GraphAMG").

**Open.** `mg_bottom_factor` = 16.9 ms/step at 1x24 (the FP32 factor rebuilt every step; 1 launch),
the largest single host kernel — the obvious next host target (factor reuse across steps / more
lanes); NOT designed. Q-H3 (T_host) not measured. Spread placement 10 % faster than contiguous:
publish contiguous, record spread alongside (Q-H5). The vof_momentum 1.84 acceleration seen once in
a D/h-16 pair run (not reproduced at D/h 20); PECLET_FLOW_OPERATOR_DOUBLE=OFF fails 6 tests on main
too (pre-existing); the 15-19 % swarm-drift difference to TBFsolver = near-contact treatment.

**Where things are.** Working data + gate scripts: /home/frankp/Codes/bubble_column_perf/ (README;
S-1 raw in `s1/`, merge gate `gate_s1merge.sh` / `gate_s1merge/`). Baseline worktree
../flow-main-base (detached at the pre-merge main 5795fb0). Snellius:
/projects/0/prjs1022/peclet/bubble-cpu (scripts, TBF build, results; S-1 in `s1/results/`).
