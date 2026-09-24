# VoF overlapping markers + bubble column — campaign STATE (rewrite in place, ~1 screen)

**Objective.** Find (by tests) why the multiple-marker block VoF blows up when two markers
overlap (channel_18 dies at ~1.5 eddy turnovers, WO-W3 §7), fix it (design with Fable, consensus),
then a bubble-column example in `peclet-examples` + if feasible a peclet-vs-TBFsolver benchmark.

**Where.** flow worktree `suite/flow-vof-overlap`, branch `vof-overlap` (from main 0ae29d6).
Build tree `build_cuda` (CUDA prefix, tests+MPI on). An AMR session is doing a release in
parallel — do not touch umbrella docs/RELEASE*, core, amr. W4 WIP stays parked on `vof-w4`
(f29c8e7); not merged here.

**Now (2026-09-24 night, implementer, after design §12-13).** On vof-overlap: WO-1 `85b659d`,
WO-3 `042f9c2`, trigger S>1.01 `d8183af`, WO-4 `3d13924`, §12.1-2 `97f9ca0` (block wispEps =
global, box threshold, discarded migrates), §12.3 `db37e32` (full-axis wrap copy, full_axis), §13
`05e2ea3` (sub-wispEps residue cleared + returned, residueReturned; runs on kinematic block runs
too), tests `03b1660`. leak.py: rel dV 1.3e-15, discarded 0 (§12 alone: -6.1e-10 / 1.7e-7).
G4(a) channel_18 10000->13000: PASS -- 3000 steps, max|u| 35.64, marker volumes |dV|/V <= 1.8e-14,
boxes <= 23x22x22 all run, full_axis 0, discarded 0, debris returned m2 0.098 m4 0.069 m13 0.280
m17 0.185 (others < 3e-3), lost 0, unresolved 0, residue returned ~1e-5 per marker, clip on
2711/3000 steps (<= 24 cells), phantom bound on 3000/3000 steps (real overlaps; S to ~1.5).
-R vof: 61/61 after the G1/G5 tests adopted §13 (only vof_blocks D1 failed: its sub-threshold
account is now residue). G1: speck window 0.0, B volume 2.2e-16, A bitwise. G5 np1/2/4: union C,
volumes and 7 ledger fields bitwise; volumes exact. G2 r10: peaks moved <= 0.6 % (d=10.5 0.2301
vs 0.2314), end values <= 3.5 % (d=10 0.1315 vs 0.1270), dV 1e-15 (was 5e-15); within 5 %.
G3 r10: completes (4972 steps), max|u| 15.69, dV 1.3e-15, debris returned 0.0096/0.0196, lost 0,
bound 3397 steps (contact 751-4212). G3 r50: completes (6547), 15.68, dV 1.3e-15, returned
0.0054/0.0054, bound 4248 (contact 822-5131). Hysing gate (vof_blocks_ns.py hysing, nx 64,
T 3): global path bitwise unchanged; block path moved: v_rise max 0.2827287000 -> 0.2827271797,
y_c 1.2085701812 -> 1.2085639735, block-vs-global -0.0005 % / -0.0005 % (was -0.0000 %); local
block-vs-global max|dC| 1.1e-5 -> 2.5e-3, max|w| diff 9.1e-5 -> 5.1e-2 of max|w| 43 (bubble band,
z 67-89); was never bitwise (C convention flip). vof_blocks_ns_mpi / vof_redistribute_mpi: all
gated diffs stay at round-off (np>1 |dV| vs np=1 0 -> 1.1e-13, tol 1e-11). G0: ctest -LE
bench 163/163 (OMP 4, -j8, 12358 s); state_hash 13/13 identical.

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

**Next action (2026-09-25 00:20, before a session compaction).** USER AUTHORIZED push + publish.
Running: (a) opus-implementer Phase B of the review fixes (doc/vof_overlap_review.md, untracked):
real clip test, curvProto-propagation test, ctests for wrap-recentre / residue return / phantom
bound (np 1/2/4), cost measurement + parallel per-cell writes, setter
diagnostics.set_vof_phantom_capillary_bound (default on), rename set_vof_kappa_clip ->
set_vof_block_kappa_clip, kinematic interfaceEps consistency, G2/G3/G4(a)/Hysing re-runs with
counters, a block case in state_hash.py. (b) GPU chain scratchpad/c18_g4b/chain.sh on frozen module
scratchpad/flow_prod (= 14d9483): peclet bubble column t=150 (ETA ~05:15) -> peclet D/h=24 to t=5
-> channel_18 10000 -> 5 turnovers (G4(b)). Branches vof-overlap (flow) and bubble-column
(peclet-examples) pushed to origin as backups. THEN: reviewer re-check of Phase B (short) ->
rebase vof-overlap onto moved flow origin/main (core 1.2.0 pin, init_mpi raise), rebuild, battery
once, push flow main; umbrella pointer bump LAST (release session active: core 1.2.0 out, AMR in
progress — coordinate, stage named paths); update flow/CLAUDE.md VoF scope sentence (colliding
markers now rated) + decision-register entry (design §10); finish page
peclet-examples/benchmarks/bubble-column/index.qmd (prose placeholders), peclet_reduce.py ->
data/peclet_closed.npz (+ _d24_t5), render, merge bubble-column to main, push = publish.
Cleanup after: tbfsolver/run, run_open, run150 raw snapshots (~17 GB), peclet/run_degraded_pureEps.

**Gates (to be set with the design).** channel_18 ≥ 5 turnovers; static pair parasitic current
flat in d; marker volumes 1e-12; every existing block ctest bit-identical when no overlap.
