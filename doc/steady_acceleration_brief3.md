# Architect brief 3 — steady acceleration: the Ritz guard's noise floor is false at tight settings

Worktree `/home/frankp/Codes/suite/flow-anderson`, branch `anderson` (head 47479aa). Design note
`doc/steady_acceleration.md` (revision 1; D5 is the entry in question, lines ~195–215). Evidence:
`doc/steady_acceleration_log.md`, entry "2026-10-02 — WO-5: the gate campaign", §G1 and §G7 (lines
~611–890). The C++ path is implemented (WO-2/WO-4 done) and matches the NumPy oracle
`tests/study/anderson_oracle.py --rev 1` to 2.5e-9 per step; the oracle reproduces the failure below
exactly, so you can test a revision there cheaply (host build `build_omp`, `PYTHONPATH=$PWD/build_omp`,
`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`, venv `../.venv`). The C++ gate driver is
`tests/study/steady_acceleration_gates.py`.

## 1. The question

What should the instability guard (D5) be — eligibility, threshold, consequence — so that it has no
false alarms at production AND tight settings while still catching what it exists to catch? Or
should it go? One decision, with the oracle evidence, and the exact note/work-order delta.

## 2. Why the architect

Revision 1's premise — the Ritz estimate is meaningful above kRitzFloor = max(1e-10, 1000·τ) — is
false at tight settings (τ = 1e-12 → floor 1e-9), and a false alarm currently makes
`march_to_steady` return `converged=False` on a converging case (§7: `if acc.status == "unstable":
return result(False, "unstable")`).

## 3. Measured (WO-5; host and CUDA agree)

Production (τ = 1e-8, floor 1e-5): 96 accelerated runs, 946 eligible readings, max 0.9825, none >
1.0005 — passes. Tight: every eligible reading > 1.0005, as (step, residual, radius):
- coll §11 N14: (19, 5.47e-6, 1.0021) (20, 5.41e-6, 1.0037) (22, 5.42e-6, 1.0047) (31, 4.78e-6,
  1.0007) (33, 4.78e-6, 1.0062) (62, 6.65e-9, 1.0041) (63, 6.16e-9, 1.0053) (64, 5.85e-9, 1.0053)
  → "unstable" at 64, converged=False.
- stag §11 N20: (39, 5.35e-9, 1.0010) (42, 5.26e-9, 1.0015) (44, 5.25e-9, 1.0013) (47, 5.23e-9,
  1.0026) (49, 5.23e-9, 1.0011) (50, 5.18e-9, 1.0011) (51, 5.17e-9, 1.0014) → "unstable" at 51.
- coll N12: (16, 1.49e-5, 1.0120); stag N18: (22, 1.73e-6, 1.0111) (23, 1.71e-6, 1.0047); stag N24:
  (58, 1.78e-8, 1.0072); Z&H φ 0.343 max 1.0030, φ 0.45 max 1.0018.
- Two families: (a) the velocity residual stalls at 5–7e-9 at tight settings (5× the floor) and the
  estimate there is noise; (b) isolated 1.002–1.012 readings at residual 1e-6–1.5e-5 on stagnating
  stretches of phase A (coll N14: residual ≈ 5e-6 for steps 19–37). All plain marches converge.

What the guard must still catch:
- G7a (staggered §11 N16, μ = 0.0158, dt = 1.234e-2; plain diverges at step 440): at m = 8 the
  guard reads 1.124 at residual 9.4e-3 three times → "unstable" at step 18 (the only true positive
  seen). At m = 3/5 Anderson does not trip the guard; the march reports diverged.
- Core unit test U4 (`core-anderson/tests/test_anderson.cpp`): a synthetic map with growth 1.02 /
  rotation modulus 1.01 reads Ritz ≈ 1.0105 and must go "unstable" within 3m steps of engagement.
  Note: 1.0105 overlaps family (b)'s noise (up to 1.012), so a threshold alone cannot separate them.
- Other safeguards already in the design: restarts (4× min), the certification growth exit, the
  plain stop instrument on plain steps, NaN/failed-solve restore.

## 4. Constraints

Fixed point unchanged; `step()` untouched; on-device + MPI (any guard decision must be identical on
all ranks — rank-0 broadcast exists); no env vars; no constant tuned to pass a gate without a
mechanism argument. A core change is allowed (core branch `anderson` in
`/home/frankp/Codes/suite/core-anderson`, untagged, head 9ff3bd2; an engineer is concurrently fusing
the pass-2 reductions there for G8 — numerics unchanged — so specify any core delta as a work order,
do not edit core).

## 5. Decided, not open

Everything else in revision 1. Q1/Q13: `accelerate=True` is the default (staggered bed wall 3.4–3.9×,
measured). The orchestrator has separately decided: G1's bed reference becomes a long plain march
(the plain stop instrument certifies 1.4e-8 early on the bed's ~0.9999 tail); G3's bed run is
extended to reach its depth criterion. Do not re-open those.

## 6. Open (decide)

1. Eligibility / threshold / consecutive count for the guard, or its removal.
2. The consequence of "unstable": return not-converged (current) vs fall back to the plain march
   from the last plain output and let the plain instrument decide (revision 0's §7 did this; a
   true positive then ends as a diverged/not-converged plain march — check G7a both ways).
3. The G7c census criterion to replace "every eligible reading ≤ 1.0005" if (1)/(2) change it.

## 7. Deliverable

Revision 2 of the note in place (short "Revision 2" section at top; D5, §4.3 step 7, §7, §8 G7, §9
work orders for flow and, if needed, core). Oracle evidence over the WO-5 tight + production matrix
and G7a, appended to the log with commands. Commit named paths on `anderson`, trailers:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015fNTZgmz6U1bv2MRTM3bbF
No production code, no push/merge, do not touch core-anderson or the shared checkouts.

## 8. Out of scope

G8 overhead, G1/G3 bed references, the plain stop instrument's premature certificate on the bed
(reported to the user separately), anything outside the guard.
