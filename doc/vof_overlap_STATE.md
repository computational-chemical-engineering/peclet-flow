# VoF overlapping markers + bubble column — campaign STATE (rewrite in place, ~1 screen)

**Objective.** Find (by tests) why the multiple-marker block VoF blows up when two markers
overlap (channel_18 dies at ~1.5 eddy turnovers, WO-W3 §7), fix it (design with Fable, consensus),
then a bubble-column example in `peclet-examples` + if feasible a peclet-vs-TBFsolver benchmark.

**Where.** flow worktree `suite/flow-vof-overlap`, branch `vof-overlap` (from main 0ae29d6).
Build tree `build_cuda` (CUDA prefix, tests+MPI on). An AMR session is doing a release in
parallel — do not touch umbrella docs/RELEASE*, core, amr. W4 WIP stays parked on `vof-w4`
(f29c8e7); not merged here.

**Now (2026-09-24 evening, implementer).** vof-overlap: WO-2 census `3b7376b`, WO-7 `29b1bb8`
landed. WO-1 clip is on branch `vof-overlap-wo1-pending` (`795765c`), NOT merged: STOPPED on Q4 --
the clip fires in existing ctests: vof_surface_tension P6 (eps=0 wisp ablation, 205639 clips, pre
max 3.5e26) FAILS `kmax > 100*2/R`; vof_curvature C sweep D/dx 2.8/4.4 (40/92, pre 2.31/1.14,
errors change), wrap-seam of periodic planes B2/K2 (202; 2/68/30/140, pre <= 12.3, gated numbers
unchanged); vof_cutcell G5 (968, 1.60), vof_wetting G1 (10640, 2.90), vof_wetting_mpi (968, 8.70),
vof_wetting_dynamic_mpi (576, 1.18) -- printed output unchanged. WO-3 NOT implemented: STOPPED on
Q2 -- 5^3-sum histogram not a decade apart at 1.0 (dump 10702: fragments 0.10-0.89 vs attached
>= 1.19; eps1e-3 dump: a fragment at 1.24). Census on the dump = brief (m2 6/0.069, m13 5/4.2e-3).
Refinement check: m2's debris sits in m3's band (C3 0.10-0.95), m13's in NO marker; motion not
resolvable in the 3-step window. G1(i) scene is trivial: the speck gets NO estimate (kappa 0).
No-op proofs: 48 arrays + G2 8 cases + state_hash 13/13 bitwise vs unmodified tree (clip on/off).
Gates (clip + census + gated bound, no removal): G2 ratio 10 = brief table exactly, bitwise.
G3 shear r10: completes T 12.5 (5907 steps, was dead at 2910), max|u| 15.69, dV 6e-14, debris <= 34
cells / 0.22, bound on 5028 steps (stays on after separation: residual wisps keep S > 1+1e-8).
G4(a) channel_18: passes 10706/11213/11331 (max|u| <= 35.9) but at ~12150 marker 2's debris-
stretched box reaches 128 = Lx, vofClampBox snaps it to [0,128) and recentre's copy by unwrapped
index WIPES the marker (volume 0); at 12518 marker 13 (box 118 wide) goes non-finite. Debris
<= 52 cells / 0.22 per marker, clip fires 2198/2518 steps; phantom bound on EVERY step, dt 2.24e-3
-> 1.76e-3 (the note's "never binds" is false). G7 ratio 50 peak/end, single | d=8 MAX+gated |
d=8 tent: mu_g=mu_l @0.25 0.142/0.050 | 0.144/0.115 (dt 0.05) | 0.145/0.055; @0.40 0.145/0.066 |
0.144/0.075 | 0.149/0.069; mu_g=0.02: single itself grows (0.437 @0.25, 0.546 @0.40), MAX+gated
d=8 0.318/0.245, 0.317/0.204, tent 0.423/0.423, 0.576/0.523. Shear r50: MAX+gated 8305 steps,
contact 810-7249; tent 3140 steps, contact 810-2089; both complete, max|u| 15.69, dV <= 1e-13.

**Instruments.** `tests/study/vof_blocks_overlap.py` (static pair vs d); scratch
`diag_c18.py` (restart channel_18 from W3's healthy step-10000 checkpoint
`flow-w4/tests/study/channel_18/runs/prod/ckpt.npz`, dump the window around the blow-up).
New diagnostics bound: `diagnostics.vof_block_kappa(id)`, `diagnostics.vof_block_force(c)`.

**Second failure mode (same family).** set_vof_interface_eps 1e-3 / 1e-2 pass 10706 but die at
11213 / 11331: marker 13 grew a 24-cell debris blob (vol 0.16, C up to 0.08, kappa -428, face force
-10810). Couette pair (`vof_blocks_overlap.py shear`, U 16, 20 min) dies at step 2910 AFTER
separating = cheap deterministic reproducer. U 40 run: stable through 1540+ steps.

**Design — CONSENSUS (Fable note doc/vof_overlap_design.md, commits 4ffb431, 78f2ac3, 36704e6).**
(1) |kappa| <= 1/Delta_min clip, both VoF paths (safety); (2) per-step per-marker debris removal,
Sum_5^3 C < 1 cell, exact volume return to the marker's attached interface weighted C(1-C),
ledgered (hygiene); (3) S = Sum_k C_k scatter; while any S > 1+1e-8 the capillary limit uses the
GAS-GAS bound sqrt(2 rho_min h^3 / 4 pi sigma) (phantom interfaces; measured: ratio-50 static pair
d=8 peak 0.26/0.59/1.16 at safety 0.15/0.25/0.40 vs 0.14 separated); (4) "tent" liquid-lens density
opt-in only (default MAX = TBFsolver's model); W4 item 1 (union force) REJECTED. Opus pushback
accepted: tent's volume argument was backwards (MAX already over-counts liquid by V_ov).
Implementation: opus-implementer on WO-1..4 + WO-7 in build_impl (not build_cuda).

**Bubble column (independent).** DECISION: walled periodic column (x vertical periodic, walls +-y,
z periodic), 128x96x64, D=16 cells, 16 bubbles phi 4.36 %, Loisy E1 (Ar 29.9, Bo 2) with density
and viscosity ratios 0.02; reason: TBFsolver FAST_MODE Poisson is tridiagonal in y (no fully
periodic swarm). Case module `~/Codes/peclet-examples-bubble-column/benchmarks/bubble-column/
scripts/case.py` (worktree of peclet-examples, branch bubble-column). TBFsolver built locally in
scratchpad/tbf (patched to read specs/bubblePositions; patch file beside it). DECISION: CLOSED
column (zero net volume flux, TBFsolver flowCtrl 2 flow_rate 0; peclet driver subtracts mean u_x
each step) — flowCtrl 3 grew a net upflow from wall friction. TBFsolver closed run: engineer
restarting (8x3 cores); open-column output kept in tbfsolver/run_open/. peclet run waits for the fix.

**Next action.** Review Fable's design note; iterate to consensus; implement (opus); gate on the
shear reproducer + channel_18 restart; then the peclet side of the bubble column.

**Gates (to be set with the design).** channel_18 ≥ 5 turnovers; static pair parasitic current
flat in d; marker volumes 1e-12; every existing block ctest bit-identical when no overlap.
