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

**Status (2026-09-27 06:00).** Main line WO-0..5 + WO-8 ON FLOW MAIN 035121a (all bitwise vs main,
CUDA + host 184/184), umbrella b27fd39 (register: face-form operator with band sign, host launch
rule, fused wrap + device Krylov scalars, batched container). Deferred: WO-9. Not yet: WO-6 (B1
device bottom), WO-7 (C3, core change + tag), WO-10..13, E2(a) (§12, premise passed).
Physics (t 50-150): peclet 0.949 / TBF default 0.778 / TBF no-smoothing 0.811; local slip 0.922 vs
0.804. D/h=24 (t 30-60): slip peclet 0.886->0.935, TBF-ns 0.772->0.813 — the gap is NOT resolution;
t=1..5 transient: peclet 7-10 % faster at both resolutions (column bubbles start 2 D apart in-line).
USER: go ahead with (1) Loisy, Naso & Spelt 2017 E1 free array (paper in scratchpad/loisy; fig. 21:
Nb=8, U/U0 ~0.80 at phi 3.8 %, U0 ~1.03 from Re0 31 -> U ~0.82): peclet run scratchpad/loisy/run_loisy.py
(96^3, D/h 20, triply periodic, 8 bubbles, t=100) running; (2) single bubble in a 2 D tall periodic box
(own wake at the column's spacing), both codes, scratchpad/single/run_x2.sh running.

**Next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then the bubble-column page (cost + physics sections).

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
