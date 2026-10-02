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

**Running / results (2026-10-02 late).**
- P3 DONE (branch vof-b1, worktree flow-vof-b1, not pushed; handoff in its doc/vof_step_performance_log.md):
  WO-6 device geometric-Krylov bottom (tau 1e-5 via E3) + WO-11 solids — numerically clean (column
  2e-14, identical iterations; host bitwise; bulk transfers 13/19 -> 0/2), BUT slower on the RTX 5080:
  74 vs 50 ms/step (bottom 2.8 ms/solve; single team, FP64-weak card; measured on a 90-99 % shared
  GPU). DECISION: land with the host GraphAMG still the GPU default, B1 opt-in, until a quiet-GPU and
  an H100 measurement; if confirmed, back to the architect (premise R3; a device dense LU of the
  1536-unknown bottom looks ~1-2 ms/step). CUDA battery was cut at 47/189 (0 failed): rerun at merge.
- P5 DONE (vof-pvfit, flow-vof-pvfit; core cb4c7ba on core-pvfit, NOT pushed — release session
  suite-73 asked to release it): WO-7b team kernel bitwise (fallbackBatch 5.37 -> 1.61 ms/launch);
  WO-13 D1: rtol 1e-6 passes all criteria, iterations 13.9 -> 7.4, but systematic ~5e-8 volume
  drift over production -> DECISION: published case stays 1e-10 (peclet-examples ac35a14 reverts
  f28b2ce); D1 rerun on main scripted (~/Codes/bubble_column_perf/d1_main/run_d1_main.sh) for a
  quiet GPU. WO-7c (persistent teams on the WO-8 batched tier-3 path, DECISION option C) running.
- P4 E2(a) STOPPED at WO-E2.3 (vof-e2, flow-vof-e2; handoff in its log): E2.1/E2.2 gated (bitwise;
  RED, REC, RST pass); G-E2-BAL FAILS with viscosity — the D-E2.8 pressure update feeds +mu_r div q
  into the rotational increment (anti-diffusion): mu=0.01 x1.04/step, mu=0.1 blows up; bubble column
  unstable. Without the viscous term: stable but transients decay 0.99/step (40x the exact path).
  Benefit while stable: projection 95.7 -> 55.3 ms/step (shared GPU), PCG its 13.06 -> 7.96.
  ARCHITECT RESUMED (the §12 author) for §12.13: stable-with-viscosity D-E2.8 (incl. TBF's
  non-incremental form as an option), a realistic BAL threshold, impl. questions 2-4.
- Gallery publish broken since 2026-09-21 (11 pages changed without re-freeze -> CI re-executes ->
  6 h timeout); engineer re-freezing them on branch refreeze-0921 (brief
  /home/frankp/Codes/bubble_column_perf/BRIEF_GALLERY_REFREEZE.md). The bubble-column page goes live
  once that lands.
- Core cb4c7ba (pvFit split) PUSHED to core main; core TAG = USER decision (v1.3.2 / v1.4.0 / wait);
  flow + amr pins still v1.3.0. WO-7b/c parked on vof-pvfit until the tag.

**Next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then UPDATE the page's cost table (Snellius rerun) as WO-6/E2(a) land.

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
