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
- **A(b) landed 2026-10-08** (flow 62f91ef, 591bde4): the host `mg_bottom_factor` runs a bitwise
  host schedule (same scalars, ~2.5 barriers per tile, vectorized lanes); factor bytes, U8, host
  G-BIT (OMP 1/8/24, `direct`) and CUDA identical to main. Workstation: 5.6-6.5 -> 1.05-1.5 ms per
  factor (kprof 1x24: 6.6 -> 1.5-1.8 ms/step). DECISION `kBottomHostFactorTeam = 2` (alternative 4)
  pending the genoa confirmation `bubble_column_perf/s1/s1_bfac.slurm` (NOT submitted: user OK).
- PARKED: E2(a) constant-coefficient driver; E2(b) FFT solver DROPPED 2026-10-08 (register).

**Next action.**
1. Gallery cost table on peclet-examples `benchmarks/bubble-column` from S-1 (re-render + commit the
   `_freeze`; check the publish job) — not done; Snellius numbers above, TBFsolver 45.5-46.0.
2. USER 2026-10-08: device A(b) wanted later — the device factor costs 7–9 ms per launch on the
   RTX 5080 (walls-y), out of a 36–37 ms step; same goal: bitwise-faster schedule first.
3. WO-H5 (Krylov reduction fusion on host), H6 (distributed host 'direct'), H7, then S-2 (incl. a
   64x3 and 24x8 profile), H8 (flow CLAUDE.md: host no longer "keeps GraphAMG").

**Open.** Genoa confirmation of A(b) (S-1 had `mg_bottom_factor` 16.9 ms/step at 1x24): job
`s1_bfac.slurm`, needs the user's OK. A(a) (factor reuse across steps, a numerics change) not
designed. The DEVICE factor: next action 2. Q-H3:
factor T measured (2 chosen, above); the solve kernel's `kBottomHostTeam = 8` not re-measured. Spread placement 10 % faster than contiguous:
publish contiguous, record spread alongside (Q-H5). The vof_momentum 1.84 acceleration seen once in
a D/h-16 pair run (not reproduced at D/h 20); PECLET_FLOW_OPERATOR_DOUBLE=OFF fails 6 tests on main
too (pre-existing); the 15-19 % swarm-drift difference to TBFsolver = near-contact treatment.

**Where things are.** Working data + gate scripts: /home/frankp/Codes/bubble_column_perf/ (README;
S-1 raw in `s1/`, merge gate `gate_s1merge.sh` / `gate_s1merge/`). Baseline worktree
../flow-main-base (detached at the pre-merge main 5795fb0). Snellius:
/projects/0/prjs1022/peclet/bubble-cpu (scripts, TBF build, results; S-1 in `s1/results/`).
