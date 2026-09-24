# VoF overlapping markers + bubble column — campaign STATE (rewrite in place, ~1 screen)

**Objective.** Find (by tests) why the multiple-marker block VoF blows up when two markers
overlap (channel_18 dies at ~1.5 eddy turnovers, WO-W3 §7), fix it (design with Fable, consensus),
then a bubble-column example in `peclet-examples` + if feasible a peclet-vs-TBFsolver benchmark.

**Where.** flow worktree `suite/flow-vof-overlap`, branch `vof-overlap` (from main 0ae29d6).
Build tree `build_cuda` (CUDA prefix, tests+MPI on). An AMR session is doing a release in
parallel — do not touch umbrella docs/RELEASE*, core, amr. W4 WIP stays parked on `vof-w4`
(f29c8e7); not merged here.

**Now.** ROOT CAUSE LOCALISED (2026-09-24, local restart of W3's step-10000 checkpoint, RTX 5080):
blow-up in ONE step 10705->10706, max|u| 34.65 -> 135.5 at v(26,41,12), inside the boxes of the
overlapping pair (markers 2,3; they have shared ~4-6 cells of volume since step 10000 without harm).
The face force there is -405 = marker 3's regular -89 (kappa 0.35) + marker 2's -306: marker 2 has
DEBRIS (6 fragment cells, vol 0.07, C 2e-5..4e-2, no C>0.5 cell within 2) stranded in marker 3's
interface band; the cell with C2 = 2.08e-5 counts as interfacial (interfaceEps = 1e-8) and gets
kappa = 273/cell (true ~0.4). sigma*kappa_f*dC = 320*137*7e-3 on low-rho fluid -> blow-up.
Marker 13 (the 12/13 pair) carries the same kind of debris. So: NOT the SUM-vs-MAX balanced-force
argument (WO-W4 premise); a curvature estimate on marker debris created in the overlap region.
TBFsolver has exactly the two guards peclet lacks: |kappa| <= 1/Delta clip, and resetFragments
(zero mixed cells with no full cell in 5^3, every 10 steps).
Static pair: single 0.13 peak, d=14 0.13, d=10.5 0.23 (bounded) — static H1 not supported so far.
Running: restart with set_vof_interface_eps 1e-3 / 1e-2 (does removing kappa on C<eps cells survive?);
shear collision test (vof_blocks_overlap.py shear).

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
