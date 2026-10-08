# VoF step performance campaign — STATE (rewritten in place; history in vof_step_performance_log.md)

**Objective.** USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles; no GPU<->host transfers in the step. Yardstick: TBFsolver on the bubble column
(peclet-examples `benchmarks/bubble-column`, live page), same Snellius genoa node.

**Where we are (2026-10-08, session handover).**
- On flow main (all gated, register entries in suite/docs/decisions/flow.md): main line WO-0..5, 8
  (bitwise), WO-7 team PV fallback (core v1.3.2 released for it; flow now pins v1.4.0), §13 GPU
  'direct' bottom (GPU default; RTX 5080 step 42.6 -> 36-37 ms), WO-12 Chebyshev warm bounds
  (44 -> 24 V-cycles/step), WO-9, WO-10. Full battery 194/194 with a correct MPI launcher.
- Numbers (published on the page): Snellius 24 cores peclet 144 ms (old protocol: distributed path at
  np 1, rtol 1e-10) vs TBFsolver 46; 192 cores 82-85 vs 25; GPU 36-37 ms. Case rtol is 1e-8 (USER).
- PARKED: E2(a) constant-coefficient driver (fails static balance ~5000x; branches vof-e2,
  vof-e2-e23-stopped2 on origin; register "PARKED").

**Next action.** CPU §14 package (design: §14 of vof_step_performance_design.md) is DONE but HELD on
branch `cpu14-h4` (pushed; worktree ../flow-cpu14-h4; 16 commits on main 935ffaf): WO-H0 protocol,
H1 host 'direct' bottom (recorded numerics change), H2 prolong, H3a-e stragglers, H4a-e container
on host. Gate vs main: bitwise on CUDA and on host with --bottom-solver algebraic; batteries 194/194
both. HELD because H-1's host speed is unproven (serially 1.35x GraphAMG; 8-lane factor scaling
unmeasurable on the loaded workstation). 1) S-1 on Snellius — ASK THE USER FIRST (budget "running
low"): build cpu14-h4 (core v1.4.0) generic + znver4, apply patches/bench_peclet_H0.patch, layouts
1x24, 6x4, 3x8, 8x3 per-CCD masks (never a rank across two CCDs), host bottom 'direct' vs
'algebraic', + TBFsolver same node, + kernel profile (kprof.slurm). 2) Set the host 'auto' default
from S-1 (revert = the kHostMemory line in directBottomIneligible(), commit 3687a09 on the branch),
rebase on main, battery, push, umbrella pointer + §14.8 register entries (texts in the H0-H3 and H4
reports, summarised in the campaign log). 3) WO-H5 (Krylov reduction fusion), H6 (distributed host
'direct'), H7, S-2 (incl. a 64x3 profile), H8 (CLAUDE.md: host no longer "keeps GraphAMG").
4) Update the page's cost table from S-1 (re-render + commit the _freeze; check the publish job).

**Open.** E2(b) FFT never decided; the vof_momentum 1.84 acceleration seen once in a D/h-16 pair run
(not reproduced at D/h 20); PECLET_FLOW_OPERATOR_DOUBLE=OFF fails 6 tests on main too (pre-existing);
the 15-19 % swarm-drift difference to TBFsolver = near-contact treatment (both codes grid-set).

**Where things are.** Working data + gate scripts: /home/frankp/Codes/bubble_column_perf/ (README).
Worktrees: ../flow-cpu14 (branch cpu14 = H0-H3 only), ../flow-cpu14-h4 (the combined branch),
../flow-main-base (detached baseline; build_mpi_ok has the correct MPI launcher), ../core-v140
(detached v1.4.0, used by the cpu14 builds via PECLET_SIBLING_PECLET_CORE).
Snellius: /projects/0/prjs1022/peclet/bubble-cpu (scripts, TBF build, results).
