# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (WO-1 … WO-6): Anderson acceleration of steady
marches, `peclet.flow.march_to_steady`. History and every number: `doc/steady_acceleration_log.md`.

**Where we are (2026-10-02).**
- **Branch:** `anderson` (worktree `suite/flow-anderson`, from main `f6b89fe`), not pushed.
- **Design note:** at **revision 1** (architect pass on `doc/steady_acceleration_brief2.md`). Its
  section "Revision 1" lists the changes:
  - velocity-only metric, P carried;
  - phase A hands over on the velocity residual;
  - the Ritz guard only on mixed windows above 1000·τ;
  - certification budget 12 blocks plus the early "slow" exit;
  - window stays 5.
- **WO-1:** DONE at rev 1. The oracle `tests/study/anderson_oracle.py` defaults to `--rev 1`;
  `--rev 0` reproduces the stopped WO-1.
- **WO-3:** DONE in `../core-anderson`. WO-3b (core delta for rev 1) is not started.
- **Not started:** WO-2, WO-4, WO-5, WO-6.

**Next action.**
1. Route WO-3b (§9) to opus-engineer in `../core-anderson`, branch `anderson`:
   - part 1 — `innerTolerance`, `kRitzFloorFactor`, the per-slot mixed flag, the eligibility rule;
   - part 2 — delete the Pressure role, `sdf`, `cP`, `gauged` and pass 1 (Q14, default yes), as a
     separate commit.
2. Then WO-2 → WO-4 → WO-5 in flow. WO-5 decides D11 on the staggered dense bed (Q13).

**Gates (revision-1 oracle, production, m = 5 unless stated).**

| gate | case | number | verdict |
|---|---|---|---|
| G1 tight | §11 N=16 coll / stag, m 3/5/8 | ≤ 5.1e-10 / ≤ 1.9e-11 | pass |
| G2 ≥ 3× | §11 coll N16 | 395 / 89 = 4.4× (m=3 6.1×, m=8 5.4×) | pass |
| steps_acc ≤ steps_plain | every case, m 3/5/8 | worst: bed coll m=3 163 vs 190 | pass |
| G7a | stag N16 μ 0.0158 SOU | converged=False everywhere (m=8 via the guard at step 18) | pass |
| false "unstable" | all runs | 0 (max eligible Ritz 0.9974) | pass |
| dense bed (D11 early indication) | stag / coll | steps 3.5× / 2.3×; pressure iterations 3.5× / 2.4×; host map time 2.7–3.2× / 2.2× | clears 1.5× in steps |

**Open decisions (defaults in force; note §10).**
- Q12 (fact): the cost of a mixed iterate. Default: proceed; G2 wall time decides.
- Q13 (preference): which bed decides D11. Default: staggered (A1's solver).
- Q14 (preference): WO-3b part 2, deleting the core pressure metric. Default: yes.
- Q15 (fact): the Ritz floor factor. Default: 1000.
- Q16 (fact): the budget of 12 blocks. Default: as stated.
- Q17 (preference): the A1 bed's sealed pockets. Default: no action, tell the A1 owner.

**Anchors.**
- The design: note "Revision 1" (top), D3/D5/D6, §4.3 step 7, §7 (driver), §9 WO-3b.
- Oracle: `AndersonOracle.step` (§4.3), `march_to_steady(rev=1)` (§7).
- Probes: `tests/study/anderson_rev1_probes.py` (slowmode / phase-a / certify).
