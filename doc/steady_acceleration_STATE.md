# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (WO-1 … WO-6): Anderson acceleration of steady
marches, `peclet.flow.march_to_steady`. History and every number: `doc/steady_acceleration_log.md`.

**Where we are (2026-10-02, evening).**
- **Branch:** `anderson` (worktree `suite/flow-anderson`), not pushed. Core: `../core-anderson`
  `9ff3bd2` (WO-3 + WO-3b done there), built with `-DPECLET_SIBLING_PECLET_CORE=…/core-anderson`.
- **WO-2 DONE** (`311d0cb`): `Solver::marchState()` + the 12 refusals, ctest `march_state`.
- **WO-4 DONE**: adapter `src/anderson_accelerator.hpp`, bindings, `peclet.flow.march_to_steady`,
  ctests `march_to_steady` + `anderson_mpi` (G4c). C++ == oracle (2.5e-9 over 30 steps; step
  counts equal the R-6 table everywhere compared). `accelerate=True` default as the note states.
- **WO-5 STOPPED on G1 / G7c (the Ritz guard at TIGHT settings).** False "unstable" on §11
  collocated N = 14 (step 64) and staggered N = 20 (step 51): eligible readings 1.001–1.005 at
  residual 5–7e-9, i.e. 5–7× the tight floor 1e-9 (= 1000·τ, τ = 1e-12), where the tight map's
  residual stalls. More readings > 1.0005 on stag N18/N24, coll N12, Z&H 0.343/0.45 (up to 1.012,
  some at residual 1e-6 – 1.5e-5). The oracle reproduces it exactly → a design question (Q15),
  not a C++ bug. Production runs: no reading > 1.0005 except the G7a true positive.
- G0(c) battery on the WO-4 tree: running (`ctest_g0c_{serial,mpi}.log`).

**Next action.** Report the guard finding to the caller/architect (numbers in the log, WO-5
entry). Finish the guard-independent WO-5 measurements (G2 host, G3, G5, G6, G8), then WO-6 parts
that do not depend on the guard.

**Gates (C++ build).**

| gate | number | verdict |
|---|---|---|
| G0(a) | 13 state hashes identical (WO-2, WO-4) | pass |
| G0(b) | N16/N24 both schemes: equal steps, bit-identical <u_x> | pass |
| oracle | max rel residual diff 2.54e-9 / 30 steps | pass |
| G2 (CUDA, m = 5) | coll N16 395/89 = 4.44× steps, 4.6× wall; bed stag wall 3.6–3.9×, coll 2.3–2.7× | pass; D11 → True (host pending) |
| G4 / G4c | hash np1 == serial; K np2/np4 ≤ 2.6e-12; u ≤ 2.6e-15, γ bitwise | pass |
| G1 tight | coll N14, stag N20: converged=False (false "unstable") | **FAIL** |
| G7c census | eligible readings up to 1.012 > 1.0005 (tight) | **FAIL** |

**Open decisions (defaults in force; note §10).** Q12–Q17 as in the note; Q15 now has evidence
against 1000·τ at tight settings (see above).

**Anchors.** Note "Revision 1", D5, §4.1 (`kRitzFloorFactor`), §4.3 step 7; core
`AndersonCore::complete` step 7; gates script `tests/study/steady_acceleration_gates.py`.
