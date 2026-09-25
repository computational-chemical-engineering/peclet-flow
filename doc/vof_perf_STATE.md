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

**Running.**
- opus-implementer WO-0..WO-5 (host flags, cell bodies, face-form operator, fused wrap, no copies,
  device Krylov scalars) — worktree ../flow-vof-mg, branch vof-mg.
- opus-implementer WO-8/WO-9 (batched block container + statistics) — worktree
  ../flow-vof-container, branch vof-container.
- architect §12 E2(a) addendum — this worktree, branch vof-perf (brief doc/vof_constcoef_pressure_brief.md).
- DONE: the LOW-wall asymmetry was variable-mu placement (face mean shifted h/2 towards +c): flow
  main fe377a5 (b273031 fix, 392bf9a ctest mirror_symmetry, fe377a5 velocity MG refuses variable
  mu), umbrella 5c328f2 (register). vof-mg / vof-container branches predate it: rebase at merge.
- production column rerunning on the fix (frozen scratchpad/flow_prod4), peclet/run.
- §12 E2(a) design landed (8ac2ea5). WO-E2.0 PASSED: ratio 1 -> 7.3 MG-PCG iterations (max 8) vs
  13.1 at ratio 50. E2(a) CPU model 57-95 ms/step (TBF 45): parity needs E2(b) FFT = USER decision
  (proposed: after E2(a) accuracy gates).

**Next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then the bubble-column page (cost + physics sections).

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
