# Architect brief 2 — steady-march acceleration: WO-1 falsified three premises

Branch `anderson`, worktree `/home/frankp/Codes/suite/flow-anderson` (commit 16393b9). Design note:
`doc/steady_acceleration.md` (revision 0, flow main f6b89fe). Measured numbers with their commands:
`doc/steady_acceleration_log.md`. The NumPy oracle that produced them, an exact implementation of
note §3/§4/§7: `tests/study/anderson_oracle.py` (host build `build_omp`, `PYTHONPATH=$PWD/build_omp`,
`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`, suite venv `../.venv`). You may run it and extend it in
scratch to test a revision; it is cheap (dense bed N = 64 is 0.08 s/step on host).

## 1. The question

Can a revision of the design deliver a worthwhile speed-up on **dense random beds** (the A1 use
case; worthwhile = the user's bar of ≥ 1.5× wall time, D11) while fixing the two driver defects
below — and if so, what exactly changes in the note? If not, say so plainly: the answer may be that
acceleration is only for symmetric/collocated cases, or that this campaign should stop.

## 2. Why it needs the architect

The three findings contradict premises of the note itself (§7's target, §4.3 step 7's guard scope,
§7's certification assumption), and the A1 dense-bed gain is 1.07–1.26× in steps against an
expected 2–4×. Whether the remedy is a different metric, a different phase-A target, a different
state, or no remedy is a design decision.

## 3. Current state (measured, WO-1; host-openmp, production settings unless noted)

The oracle's plain march reproduces the study exactly (collocated N=16: 395 steps, K 4.2663;
staggered: 75 steps, K 4.2120).

**Steps to `converged=True`** (plain / m=3 / m=5 / m=8):

| case | plain | m=3 | m=5 | m=8 |
|---|---|---|---|---|
| §11 collocated N=16 | 395 | 125 | 112 | 163 |
| §11 collocated N=24 | 90 | 65 | **101** | 63 |
| §11 staggered N=16 | 75 | 49 | **77** | 47 |
| §11 staggered N=24 | 135 | 52 | 58 | 114 |
| Re≈10 (staggered) | 345 | 166 | 114 | 109 |
| dense bed φ 0.6, N=64, staggered | 325 | 154 | 257 | 334 |
| dense bed φ 0.6, N=64, collocated | 190 | 193 | 178 | 213 |

All K agree with plain to ≤ 1.1e-5 (production) and G1-tight to 4.95e-10 / 1.8e-11. G7a passes
(plain diverges at step 440 with default SOU; m = 3/5/8 also report not converged). 0 restarts
anywhere. Window rule (m=3 within 10 % of m=5 everywhere): not met. Bed = A1's phi-scan config
(`random_arrays.py`, 64 spheres, `grow`, seed 0, D/dx 16 → N = 64, dt = 6h², PCG(200, 1e-8)).

**Finding 1 — false "unstable" in certification.** §4.3 step 7 runs the Ritz guard on every
active call including the plain certification steps. After ~m plain steps the window holds only
plain-march differences, nearly collinear: the scaled condition of XX falls to 1.6e-11–4e-11 (above
kCondMin = 1e-12, so the pseudo-inverse keeps the near-null direction), the Ritz radius reads 12.6–56,
and three readings trip "unstable" in phase B. 5 of 12 §11 production runs (coll N=16 m=5,8; coll N=24
m=8; stag N=16 m=5,8). In phase A the radius stays ≤ 1.0000004. Verified not an oracle bug (Gram
blocks match recomputation to 4.5e-16; Gelfand radius matches numpy's eigenvalue to 1e-5). The march
still returns converged (§7 checks "unstable" only in phase A, then falls back to plain), but steps
are wasted and the status is wrong.

**Finding 2 — certification rarely passes at its first blocks.** §7 assumes it does. The 30-step
budget ran out in 9 of 24 runs: after phase A the monitor's block ratios start above 1 (1.8, 1.36,
1.07, 1.01 on coll N=16 m=5), so the 0 < R < 1 test keeps failing. Note estimated ~80 steps on coll
N=16; measured 112.

**Finding 3 — on the dense bed the residual and the monitor decouple by two decades.** Phase A never
reaches its target (1 − slow_rate)·rtol = 3e-7: W-residual 8.3e-5 / 1.4e-5 / 4.4e-6 at accelerated
step 40 / 80 / 120 (staggered, m=5). Not an inner-solve floor (PCG and velocity tolerances at 1e-12
give the same sequence to 3 digits). The plain march's W-residual is still 7.85e-5 at step 300, yet
its ⟨u_x⟩ monitor certifies at step 325. The window's Ritz estimate on the bed is 0.99979–0.99999:
the slowest content of the W-residual is a mode that barely moves ⟨u_x⟩. §7's premise ("the target is
the residual at which the slowest mode can carry at most rtol of remaining change") fails here.

Other: the oracle cannot run collocated with face advection (u_f is not in the field registry, as D1
says). Wall time in the oracle is slower than plain on the bed (NumPy overhead; not meaningful).

## 4. Constraints and invariants (unchanged from the note)

- The fixed point must be unchanged (C2, Δt-independence); no change to discrete equations,
  operators or defaults; `step()` untouched and bit-identical.
- On-device, MPI-distributable; host only as oracle. No numerics-changing env vars. Physical inputs.
- ABC projection, never Rhie–Chow; no post-solve face kick.
- Memory: the 16 GB card bound of §6.3 still applies.

## 5. Already decided — not open

- D13: `AndersonCore` lives in core. **WO-3 is DONE** in `/home/frankp/Codes/suite/core-anderson`
  branch `anderson` (7212582 descriptor `AndersonState`, e300e29 core, 2defc9c tests): U1a–U7 pass
  host + CUDA, MPI np 1/2/4. The descriptor uses `AndersonComm` callables instead of a raw MPI_Comm
  (accepted). Unit test U1 was re-specified by the orchestrator: the note's linspace spectrum is a
  continuum (a window of 5 cannot deflate it; 3.3e-8 at step 400 — now U1b); U1a = fast bulk + one
  isolated mode 0.996 converges in 32 steps; three isolated modes need 330 steps at m=5 (77 at m=8).
  A revision may change the core, but say exactly what and why — it is tested and committed.
- Q1 rule (user): `accelerate` defaults to True iff the dense bed gains ≥ 1.5× wall time at G2;
  otherwise False and opt-in. Q11 (user): the §3.2 instrument and its constants are library API, used
  verbatim as the acceptance test (`march_to_steady(accelerate=False)` == the study's `march()`).
- All other §10 defaults accepted by the user.

## 6. Genuinely open (what you decide)

1. The guard scope (Finding 1) — e.g. phase A only, or a different column test.
2. Phase B / certification (Finding 2) — budget, entry condition, or how phase A hands over.
3. Finding 3 — the phase-A target and/or the metric/state, so the accelerator spends its window on
   modes the answer depends on. Explain what the slow bed mode physically is (Ritz ≈ 0.9998–0.99999,
   invisible in ⟨u_x⟩) — measure it with the oracle if needed (e.g. its spatial support: pressure in
   near-closed pores, the gauge per disconnected region, near-contact films?).
4. Whether the window default stays 5 given the table (m=3 is better on 5 of 7 cases).
5. Whether the expected bed gain after the revision clears 1.5×. Prove it with the oracle in steps
   before writing work orders that depend on it. If it does not, recommend the scope cut.

## 7. Already tried and rejected (with evidence)

- Tighter inner solves to break the bed's phase-A floor: no effect (Finding 3).
- Larger window on the bed: m=8 is worse than m=5 (334 vs 257 staggered).
- Velocity-only state: stalls at 1.9e-4 (note §1.1) — P must stay in the state on collocated.

## 8. How the answer will be verified

The oracle on: §11 N=16, 24 (collocated + staggered), Re≈10, G7a, the dense bed (staggered +
collocated). Bars: G1 tight ≤ 1e-8; steps_acc ≤ steps_plain on EVERY case; no false "unstable"; ≥ 3×
on coll N=16; the dense-bed step ratio reported against the 1.5× bar. Then the C++ path (WO-4/5).

## 9. Deliverable

Revise `doc/steady_acceleration.md` in place as **revision 1**: change the affected D-entries
(keep the old reading as "superseded by rev 1" one-liners), §4/§7/§8/§9 as needed, update the work
orders (WO-1 re-run with the revision, WO-2, WO-4–6; any WO-3 delta as a separate work order on core),
and add a short "Revision 1" section at the top listing what changed and the oracle evidence for it.
Record every number you measure in `doc/steady_acceleration_log.md` (append-only) with its command.
If you change the oracle, commit it. Commit on branch `anderson` with named paths only; trailer lines:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015fNTZgmz6U1bv2MRTM3bbF
No production code. Do not push or merge. Do not touch the shared checkouts or core-anderson.

## 10. Out of scope

VoF, collocated face-field registration (Q7), float history (Q8), MPI/GPU performance tuning,
releases and tags.
