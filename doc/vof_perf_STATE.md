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

**Next.** Snellius same-node rerun (bubble_cpu.slurm) + update the page's cost table; host MG launch
structure (CPU still ~4x TBF); WO-9/10/12; E2(a) per the user's answer.

**Old next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then UPDATE the page's cost table (Snellius rerun) as WO-6/E2(a) land.

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
