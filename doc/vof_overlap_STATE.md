# VoF overlapping markers + bubble column — campaign STATE (rewrite in place, ~1 screen)

**Objective.** Find (by tests) why the multiple-marker block VoF blows up when two markers
overlap (channel_18 dies at ~1.5 eddy turnovers, WO-W3 §7), fix it (design with Fable, consensus),
then a bubble-column example in `peclet-examples` + if feasible a peclet-vs-TBFsolver benchmark.

**Where.** flow worktree `suite/flow-vof-overlap`, branch `vof-overlap` (from main 0ae29d6).
Build tree `build_cuda` (CUDA prefix, tests+MPI on). An AMR session is doing a release in
parallel — do not touch umbrella docs/RELEASE*, core, amr. W4 WIP stays parked on `vof-w4`
(f29c8e7); not merged here.

**Now (2026-09-24 evening, implementer, after design §11).** Landed on vof-overlap: WO-1 clip,
BLOCK path only `85b659d`; WO-3 debris removal + return + ledger, predicate max_{5^3} C < 1/2
`042f9c2`; phantom trigger S > 1.01 `d8183af`; WO-4 tests `3d13924` (WO-2 `3b7376b`, WO-7
`29b1bb8` earlier). G0: ctest -LE bench 163/163 (build_impl, CUDA+MPI); block-CSF ctests assert
clip/debris/overlap counters 0 (vof_blocks_ns_mpi np1/2/4: 0/0/0); single-field cannot clip;
48 dumped arrays + G2 8 cases + state_hash 13/13 bitwise vs the unmodified tree. G1: speck cells
-> 0.0, account removed + sub-1e-8 residue = painted to 1.7e-18 (literal +-1e-14 on debrisVolume
unreachable: advection precedes removal, residue 8.6e-9), B volume 2.2e-16, A bitwise, max|u| =
reference. G5 vof_blocks_debris_mpi np1/2/4 bitwise (union C, volumes, 5 ledger fields). G2 r10:
brief table, bitwise. G3 r10: completes (5104 steps), max|u| 15.69, dV 5.7e-14, returned
0.144/0.105 (> the 0.1 gate), lost 0, bound 3627 steps (= contact 779-4447, off after
separation). G3 r50: 8247 steps, 15.69, 9.8e-14, returned 0.077/0.079, bound 6368 (= contact).
STOPPED: (1) box guard parked on `vof-overlap-box-guard-pending` (98f53ea) -- as specified it
fires in test_vof_blocks G1 (LeVeque, bubbleEps = 0 set: box 34 > 32 on y, only WY residue
lost); needs a threshold on the dropped colour or a wrap-aware copy. (2) G4(a) channel_18 FAILS:
debris ON dies at 10908 (marker 12 colour non-finite in 1e-33 residue cells; box 12 had grown to
57 x 28 x 30 on 1e-12..1e-8 wisps the predicate cannot see); debris OFF dies at 12518 exactly as
before (m2 box 128 -> wiped, m13 inf). Block advectors run WY at wispEps = 0 while curvProto
.pureEps = 1e-8; diagnostic build with block wispEps = 1e-8: no inf, but every box grows to 128
and the snap wipes the markers (volumes -> 0 by ~12400). Phantom bound on every channel step
(real overlaps S up to 1.47). Residue/wake handling of the block container is the open question.

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
