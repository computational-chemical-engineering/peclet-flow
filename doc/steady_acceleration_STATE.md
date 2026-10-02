# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (rev 2; WO-1 … WO-8): Anderson acceleration of
steady marches, `peclet.flow.march_to_steady`. History and every number:
`doc/steady_acceleration_log.md`.

**Where we are (2026-10-03).**
- **Branch:** `anderson` in `suite/flow-anderson` and `suite/core-anderson`, not pushed. Flow
  builds read core from `../core-anderson` (`-DPECLET_SIBLING_PECLET_CORE`).
- **Done:** WO-1 … WO-6, **WO-7** (flow `d4d71b5`: no guard in driver / adapter / bindings,
  `converged=True` = stationarity, Q18 default), **WO-3c** (core `f9956ed`, on the fusion
  `0290998`: guard, `innerTolerance`, GG / GR deleted; pass 2 = RR and b only), **WO-8** re-gate.
- **Gates now (log "WO-8"):** G0 (hashes 13/13, ctests 63 + 130, core 96 host + 96 CUDA) pass;
  G1 14/14 pass (coll N14 2.7e-10, stag N20 5.8e-10; staggered bed ≤ 5.1e-10 against the long
  plain march K∞ ∈ [98.9156994, 98.9156995], orchestrator decision 1); G2 production 42/42 step
  counts identical to WO-5; G4 identical to WO-5; G7 a/b/c pass (G7c closed); G8 CUDA staggered
  3.5 % (bar 5 %), collocated 1.2 %, host ≤ 4.8 %.
- **Still FAILING (twice — reported, not tuned):** G3 staggered bed, K agreement over Δt: spread
  3.8e-7 CUDA / 3.3e-7 host at running-min residual ≤ 1e-9 (reached at 400–426 steps; every other
  G3 criterion passes). Supplementary 3000-step run: Δt 600 / 1e4 agree to 4.7e-10, but Δt 60 is
  disabled by the restart rule at step 2588 (5 restarts after reaching 2.7e-12).
- Not re-run on the new core: G5, G6 (WO-5 values stand; the guard never fired there).

**Next action.** Caller: decide on the G3 bed K-agreement criterion (and the Δt-60 restarts in the
long run); then review, register entries (log R2-7), push core → tag → flow → umbrella.

**Anchors.** Note "Revision 2", D5, §7, §8 G3, §9 WO-8; core `AndersonCore::complete` (pass 2,
step 5 restart rule); instrument `tests/study/steady_acceleration_gates.py` (`g3 --extend-to`);
driver `packaging/flow_steady.py`; adapter `src/anderson_accelerator.hpp`.
