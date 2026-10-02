# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (WO-1 … WO-6): Anderson acceleration of steady
marches, `peclet.flow.march_to_steady`. History and every number: `doc/steady_acceleration_log.md`.

**Where we are (2026-10-02).** Branch `anderson` (worktree `suite/flow-anderson`, from main
`f6b89fe`), not pushed. WO-1 oracle written (`tests/study/anderson_oracle.py`) and measured.
**WO-1 is STOPPED on its own stop rule: a false "unstable"** (§9 WO-1). WO-2 … WO-6 not started
here (WO-3 runs in `core-anderson`, separately).

**Next action.** Architect question (the caller routes it), then re-run the WO-1 matrix:
1. The Ritz guard (§4.3 step 7, §4.6) is evaluated on plain `acc.step(False)` calls, whose window
   holds near-collinear plain-march differences (scaled cond(XX) 1.6e-11–4e-11 > kCondMin 1e-12);
   radii up to 56 → "unstable" in 5/12 production §11 runs, all in phase B. Should the guard run on
   certification steps, and with what truncation?
2. Certification often exhausts its num_passes + 3 budget (9/24 runs): the plain monitor's block
   ratio R starts > 1 after leaving phase A (1.8, 1.36, 1.07, 1.01 on coll N16 m5).
3. Dense bed: phase A's W-residual target (3e-7) decouples from the ⟨u_x⟩ certificate by ~2
   decades (plain W-residual 7.9e-5 when its monitor certifies) → 1.07–1.26× only.

**Gates (WO-1, literal §4.1 constants).**

| gate | case | number | verdict |
|---|---|---|---|
| G1 tight | §11 N=16 coll, m 3/5/8 | \|K_acc/K_plain−1\| = 4.95e-10 | pass |
| G1 tight | §11 N=16 stag, m 3/5/8 | 1.8e-11 | pass |
| G2 ≥ 3× steps | §11 N=16 coll, m=5, production | 395/112 = 3.53 (K vs K_G1 3.2e-7) | pass (path includes the false "unstable") |
| G7a | stag N=16 mu 0.0158 dt 1.234e-2 (SOU) | converged=False plain (diverged @440) and m 3/5/8 | pass |
| restart storm | all runs | 0 restarts | none |
| false "unstable" | §11 production, m=5/8 | 5 of 12 runs, phase B | **STOP** |

**Window rule.** Not met (m=3 vs m=5: coll N16 +11.6 %, Re≈10 +46 %); default stays 5 —
provisional until the guard/certification answer.

**Dense bed (Q2 default: A1 phi-scan phi 0.6, 64 spheres, seed 0, D/dx 16 → N 64).** Host-feasible.
Production m=5 step ratio 1.26× stag / 1.07× coll — the early Q1 indication is below 1.5×.

**Open items.** The three questions above; the oracle cannot run collocated + advection (uf_ not
in the registry, D1) — G6's collocated Re≈10 waits for WO-4.

**Anchors.** `tests/study/anderson_oracle.py` (`AndersonOracle.step` = §4.3, `march_to_steady` = §7);
note §4.3 step 7 (Ritz on every active call), §4.6 (pinv truncation), §7 (budget, target).
