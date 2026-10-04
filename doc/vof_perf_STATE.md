# VoF step performance campaign — STATE (rewritten in place)

**Objective.** USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles; no GPU<->host transfers in the step. Yardstick: TBFsolver on the bubble column
(peclet-examples benchmarks/bubble-column), same node.

**Where we are (2026-09-25 evening).**
- Landed (flow main 7fdec0d, umbrella d0182e0): 8 bitwise perf commits (GPU 207->104 Chebyshev,
  54 ms MG-PCG), `set_superficial_velocity` (USER name), x-fastest MDRange alias `src/policy.hpp` +
  ctest `iteration_order` (register suite-wide), coupling 412b067.
- Same-node Snellius genoa (tcn538, 24 cores, 7fdec0d): TBFsolver 45-46 ms/step, peclet 186-203
  (projection 125 = 67 %); 192 cores 25-33 vs 76-81. GPU RTX 5080 peclet 54 ms.
- Design: `doc/vof_step_performance_design.md` (architect, 48c2548; brief beside it). Main line
  WO-0..WO-13 -> GPU ~17-21 ms, CPU ~80-125 ms. USER DECISION: main line first, then E2(a) =
  opt-in constant-coefficient (Dodd-Ferrante) driver solved by MG-PCG, accuracy-gated (register
  umbrella e390d73); E2(b) FFT decided later. Defaults taken: -march only site/dev builds; two
  bottom solvers (host GraphAMG, GPU geometric-Krylov); tolerance changes in the case script only;
  no float V-cycle; setter names per the note.

**Status (2026-10-02).** PAGE PUBLISHED: peclet-examples main 008c120, benchmarks/bubble-column
(the comparison; isolated bubble / in-line wake / drafting pair / resolution / Loisy free swarm;
the three peclet defects; cost). Physics findings: single bubbles + in-line wakes agree with TBF
(no smoothing) 1-4 %; swarm differs ~15 % via near-contact (peclet 1.6-2.6-cell film, TBF ~0.8-cell
overlap, both grid-set, converging slowly); Loisy E1 free array at Loisy's ratios: peclet U/U0 0.89
vs ~0.80 (0.99 at ratios 0.02). GPU now 42.7 ms/step (main 035121a+). Landed since: guard
vof_momentum+blocks (2d0a0d0). OPEN: vof_momentum isolated-bubble acceleration seen in the D/h 16
pair run (1.84) but not at D/h 20 (on/off within 7 % transient, 1.6 % settled).

**Status (2026-10-03).** LANDED on flow main: WO-0..5, WO-8 (bitwise), WO-7b/c team-per-target PV
fallback (5a34c69, bitwise, core v1.3.2 released for it; now pinned v1.4.0 by the Anderson session),
device bottom 'direct' = §13 (23a3631; GPU default; quiet RTX 5080 bubble column step 42.6 ->
36.1-37.4 ms, projection 23.7 -> ~17.5; misses §13's 12.5 ms target — L2-latency-bound factor chains;
the next lever splits dots = numerics change; B1 retired). Umbrella fce1c06 (register + NAMING).
PARKED: E2(a) (branch vof-e2, worktree flow-vof-e2): stable in the column but fails static balance
by ~5000x; USER asked to choose park vs third design round (recommended: park). D1: published case
stays rtol 1e-10; D1 rerun on main scripted (~/Codes/bubble_column_perf/d1_main/run_d1_main.sh).
Known pre-existing: PECLET_FLOW_OPERATOR_DOUBLE=OFF (non-default) fails 6 tests on main too
(cell_force_placement, collocated_stability_guard, balanced_force_restart, hydro_force_units,
vof_collocated, balanced_force) — float-storage tolerances.

**Snellius rerun 2026-10-03** (flow d02d3b0, genoa, same-node TBF): 24 cores 1x24 peclet 144 ms
(znver4 140) vs TBF 46 (3.1x, was 4.1x); 8x3 166-169 vs 45-46; 192 cores 82-85 vs 25 (unchanged).
1x24 stages: projection 84.6 (60 %), momentum 14.2, curvature 16.7, block_advect 15.0. Page cost
table updated (peclet-examples aaaeaa0, deployed). Snellius budget warning: "running low".
D1 rerun on main (a0afc9b): same pattern — rtol 1e-8: iterations 13.93 -> 10.55, column volume
drift 5.2e-12 / 2000 steps; 1e-6: 7.49, 1.5e-9. USER decision pending: published case rtol (now 1e-10).
E2(a) PARKED (register 43c78cb; branches vof-e2, vof-e2-e23-stopped2 on origin).

**CPU §14 (2026-10-04, architect; branch cpu14 = 8e5fc53, worktree flow-cpu14):** Snellius kernel
profile (job 27519212): 1x24 kernel 121 ms over 2104 launches + ~25 ms host GraphAMG bottom (serial,
1.5 ms/V-cycle). FINDINGS: the Snellius np=1 runs took the DISTRIBUTED path (init_mpi at size 1) —
single-rank fusions never ran; container kernels serial per block on host (static schedule); 8x3
imbalance = CCD-straddling placement. Plan -> 1x24 ~69 ms (60-78; TBF 46); <50 needs user options
(rtol 1e-6, float V-cycle). DECISIONS: Q-H1 'direct' bottom on host YES (recorded numerics change);
Q-H2 NO (np=1 keeps the distributed path; script fix only). Running: WO-H0..H3 (flow-cpu14),
WO-H4 container (flow-cpu14-h4), WO-12/9/10 (vof-perf3; told to fold Q-H4's 4-lane order into WO-10
if not committed). Then S-1 Snellius (needs user OK: budget low), WO-H5..H8, S-2.
Production column at case rtol 1e-8 running (peclet-examples worktree peclet-examples-rtol, branch
rtol-1e8; frozen module scratchpad/flow_prod6 = a0afc9b) -> page update.

**2026-10-04 later.** LANDED: WO-12/9/10 (flow 935ffaf; Chebyshev 44 -> 24 V-cycles/step); full
battery on main with a CORRECT MPI launcher 194/194 (fresh trees otherwise pick ParaView mpiexec =
singleton MPI tests). Page updated with the rtol 1e-8 production run (peclet-examples dee1a32; drift
0.965 +- 0.042, volume 1.3e-10). CPU §14: WO-H0..H3 (branch cpu14) + WO-H4 (cpu14-h4) DONE, all
bitwise except H-1 (host 'direct' bottom: G-NUM-H passes, G-PERF open — serially 1.35x GraphAMG,
team barriers under load); H4 container kernels 132.9 -> 23.8 ms/step at 8 threads. Branches
COMBINED as cpu14-h4 (16 commits on origin/main 935ffaf); gate running (gate_cpu14.sh). HELD until
S-1 (Snellius, needs USER OK, budget low) measures host 'direct' vs 'algebraic' and the protocol fix
(patch /home/frankp/Codes/bubble_column_perf/patches/bench_peclet_H0.patch; CCD-respecting layouts).
Builds use /home/frankp/Codes/suite/core-v140 (detached v1.4.0) via PECLET_SIBLING_PECLET_CORE.

**Next.** Snellius same-node rerun (bubble_cpu.slurm) + update the page's cost table; host MG launch
structure (CPU still ~4x TBF); WO-9/10/12; E2(a) per the user's answer.

**Old next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then UPDATE the page's cost table (Snellius rerun) as WO-6/E2(a) land.

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
