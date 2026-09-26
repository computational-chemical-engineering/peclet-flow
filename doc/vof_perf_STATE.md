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

**Running / status (2026-09-27).** Agents hit the weekly usage limit on 09-26; work resumed by hand.
- vof-mg (WO-0..WO-5, band-sign WO-3): all committed, per-commit bitwise gates passed on CUDA/host/float
  (pre-rebase); rebased onto flow main ed1eea1 (balanced-force projection campaign landed there);
  rebase2 gate running: ~/Codes/bubble_column_perf/rebase2_gate.sh (vof-mg vs main ed1eea1 built in
  ../flow-main-base, state_hash + 50-step dump, CUDA + host, then both batteries).
- vof-container: WO-8a-d committed, each bitwise-gated; rebased onto ed1eea1 cleanly; needs its own
  post-rebase gate. WO-9 (diagnostics only) DEFERRED; G-PERF timing not done.
- Physics: production column on fe377a5 drift 0.949 vs TBF 0.778. Isolated bubble (scratchpad/single):
  peclet 1.300 vs TBF-nosmooth 1.309 (4x4x4), 1.077 vs 1.122 (8x4x4); TBF default 8-11 % slower ->
  TBF smoothing explains ~half. TBF NO-SMOOTHING column to t=150 running:
  peclet-examples .../tbfsolver/run150_nosmooth (~2.5 h), reduce with scripts/tbf_read.py.
- E2(a) design §12 (8ac2ea5) done; WO-E2.0 passed (7.3 vs 13.1 iterations).

**Next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then the bubble-column page (cost + physics sections).

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
