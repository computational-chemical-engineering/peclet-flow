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
RESULTS 10-01: in-line 2 D box agrees (1.770 vs 1.746). Drafting pair (scratchpad/pair): lateral
dynamics match TBF before contact; peclet holds a ~1.6-cell film (|r|min 0.93 vs TBF 0.74 =
interpenetration), tumbles slowly (pass t 20 vs 12-14); single-field VoF reproduces it to 1e-5 D ->
flow solver, not the block container. Loisy run: swarm LESS aligned than random; one periodic
in-line chain persisted ~15 time units (0.085 of the excess); Loisy used ratios 1e-3/1e-2 and a
single level-set field -> the 25 % is NOT like-for-like. USER-approved plan: B (film cause +
resolution D/h 16/24/32 both codes, ablations: vof_momentum, harmonic mu) — pair agent resumed;
A (single-field VoF at our ratios and at Loisy's 1e-3/1e-2 with enable_vof_momentum, + guard:
enable_vof_momentum with blocks must RAISE) — brief scratchpad/loisy/BRIEF_A.md, branch
vofmom-guard; then C (publish the page with the finding).
B DONE (scratchpad/pair, res_b.txt): near contact is GRID-SET in both codes — peclet film 2.6 -> 1.6
cells (0.163 -> 0.066 D) from D/h 16 -> 24 (thins faster than h, not converged lubrication); TBF
overlaps ~0.8 cells at both resolutions (|r|min 0.74). Pass-through gap peclet-TBF 8.5 -> 5.05
time units; peclet late rise converged (1.062/1.059). Harmonic mu doubles peclet's film (default
arithmetic is the less cushioning). NEW DEFECT: enable_vof_momentum (single field, no blocks) in
this case family accelerates an isolated bubble to 1.84 vs 1.06 and merges the pair — A2 at risk;
Loisy agent told to check on/off first. Guard commit 9b72708 on vofmom-guard (blocks+vof_momentum raise).
A (scratchpad/loisy): single field 0.02/0.02 U/U0 0.995 = block 0.993 (container irrelevant; one
coalescence at t 91). Loisy's ratios 1e-3/1e-2 + vof_momentum: interim t 59 U/U0 0.87 and falling
(Loisy 0.80) -> property ratios explain most of the earlier "25 %". vof_momentum on/off isolated
bubble: transient lead <= 7 %, settles 1.6 % (the pair run's 1.84 did not reproduce at D/h 20 ->
OPEN question, not a confirmed defect). Guard landed: flow main 2d0a0d0 (vof_momentum + blocks
raise; the markers never moved), umbrella 63039e7. A2 run PID 233204 finishing; summarise with
scratchpad/loisy/summarise.py run_a1 run_a2. Then C: the page.

**Next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then the bubble-column page (cost + physics sections).

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.
