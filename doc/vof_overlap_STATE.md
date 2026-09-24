# VoF overlapping markers + bubble column — campaign STATE (rewrite in place, ~1 screen)

**Objective.** Find (by tests) why the multiple-marker block VoF blows up when two markers
overlap (channel_18 dies at ~1.5 eddy turnovers, WO-W3 §7), fix it (design with Fable, consensus),
then a bubble-column example in `peclet-examples` + if feasible a peclet-vs-TBFsolver benchmark.

**Where.** flow worktree `suite/flow-vof-overlap`, branch `vof-overlap` (from main 0ae29d6).
Build tree `build_cuda` (CUDA prefix, tests+MPI on). An AMR session is doing a release in
parallel — do not touch umbrella docs/RELEASE*, core, amr. W4 WIP stays parked on `vof-w4`
(f29c8e7); not merged here.

**Now (2026-09-24 late, implementer, after design §12).** On vof-overlap: WO-1 `85b659d`, WO-3
`042f9c2`, trigger `d8183af`, WO-4 `3d13924` (G0 163/163, G1, G2 bitwise, G3 r10/r50, G5 np1/2/4
bitwise -- all BEFORE §12); §12.1-2 `97f9ca0` (block wispEps = global, box threshold
max(bubbleEps, wispEps), discarded migrates), §12.3 `db37e32` (full-axis wrap copy, full_axis).
Side branches wo1-pending / box-guard-pending deleted.
G4(a) channel_18 10000 -> 13000 with §12: RUNS THROUGH (no blow-up, 3000 steps, max|u| 35.81
<= 52; boxes <= 23x21x22 all run, full_axis 0 everywhere; debris returned m2 0.104, m4 0.027,
m12 1.6e-4, m13 0.111, m17 0.240, lost 0, unresolved 0; clip fired on 2697/3000 steps (max 22
cells); phantom bound on 3000/3000 steps (real overlaps, S to ~1.5)). STOPPED on volumes: max
rel marker volume change 1.5e-9 (gate 1e-12), a steady loss on EVERY marker incl. non-colliding
(m0 -1.4e-7 over 2400 steps). Reproduced in isolation (scratchpad leak.py: one R = 5 marker,
uniform u, 1500 advect_vof_blocks(0.2)): unmodified tree / block wispEps 0: dV +6e-12, ledger
consistent; block wispEps 1e-8 (§12): dV -3.18e-7, discarded 1.69e-7, UNLEDGERED -1.49e-7. The
global field at wispEps 1e-8 conserves (9e-13 over 1500 steps). Hypothesis: sub-1e-8 residue is
fluxed algebraically (pure-cell branch) and, now that it no longer defines the box, drifts out
of the inner box into the ghost ring, which the fill zeroes -- unledgered; the rest is dropped at
re-centring (ledgered, but 1e-7, not the "<= 1e-10 per event" §12.2 assumed). Needs a design
choice (ledger the inner-box boundary flux / keep a residue margin / clear sub-eps colour
explicitly). Not run per the order: -R vof ctests, G2/G3 reruns, full battery.

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
