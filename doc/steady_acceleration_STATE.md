# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (WO-1 … WO-6): Anderson acceleration of steady
marches, `peclet.flow.march_to_steady`. History and every number: `doc/steady_acceleration_log.md`.

**Where we are (2026-10-02, night).**
- **Branch:** `anderson` (worktree `suite/flow-anderson`), not pushed. Core: `../core-anderson`
  `9ff3bd2` (WO-3 + WO-3b), built with `-DPECLET_SIBLING_PECLET_CORE=…/core-anderson`.
- **WO-2 DONE** (`311d0cb`), **WO-4 DONE** (`2ba357c`). G0(c): 193/193 ctests pass.
- **WO-5 STOPPED** — gates that FAIL (numbers in the log, entry "WO-5"; note §1.3):
  - G1 / G7c, the Ritz guard at TIGHT settings: false "unstable" on §11 coll N14 (step 64) and
    stag N20 (step 51), readings 1.001–1.005 at residual 5–7e-9 (floor 1e-9); isolated readings
    up to 1.012 at residual 1e-6–1.5e-5. Oracle reproduces it → design question (Q15).
  - G1 staggered bed: |K_acc/K_plain − 1| = 1.44e-8 — the PLAIN tight certificate is premature
    (tail ~0.9999/step vs slow_rate 0.997); the accelerated K is within ~5e-10 of a 60 000-step march.
  - G3 staggered bed: 400 steps reach residual 1–2e-9 (not 1e-9); K spread 3.8e-7; no instability.
  - G8 CUDA staggered: 5.3 % > 5 % (latency-bound; the §6.2 remedy is a core change).
- **Passing:** G0, G2 (all bars; D11 → `accelerate=True`, staggered bed wall 3.4–3.9×), G4/G4c,
  G5, G6, G7a, G7c at production (0 of 946 readings > 1.0005), G8 memory, G8 CUDA collocated.
- **WO-6:** CLAUDE.md "Steady marches", note §1.3 "Measured", register drafts in the log (the
  guard-scope entry marked PENDING); docs build (doxygen) exit 0, no warning from the new files.
- Host G1 / G6-array / G8 re-runs were still running at the time of this file (log addendum).

**Next action.** Caller / architect: the four failures above. Nothing here changes a constant.

**Anchors.** Note "Revision 1", D5, §4.1 (`kRitzFloorFactor`), §4.3 step 7, §1.3; core
`AndersonCore::complete` step 7; instrument `tests/study/steady_acceleration_gates.py`; adapter
`src/anderson_accelerator.hpp`; driver `packaging/flow_steady.py`.
