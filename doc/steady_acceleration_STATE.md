# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (rev 2 + review fixes): Anderson acceleration of
steady marches, `peclet.flow.march_to_steady`. History and every number:
`doc/steady_acceleration_log.md`.

**Where we are (2026-10-03).**
- **Branch:** `anderson` in `suite/flow-anderson` and `suite/core-anderson`, not pushed. Flow
  builds read core from `../core-anderson` (`-DPECLET_SIBLING_PECLET_CORE`).
- **Done:** WO-1 … WO-8; **review fixes** (log "Review fixes"): core R6 `323a23a` (validate in the
  collective), R2 `868938f` (a restart restores the last kept output; U11); flow R5 `c5201b2`
  (field-count change = configuration change), R4 `48b9174` (adapter ctest), R1 `03e5757`
  (stagnation against `slow_rate**(10·window)`), R3 `fddcec7` (G3 bed depth 5e-11, K at a
  non-restart call); note §4.3 / §7 / §8 amended.
- **Then (orchestrator decisions):** R5 signature `a6da9eb` (advection on / scheme / implicit in
  the signature); gate script `d2ff05a` (G1 bed vs the plain certificate's error; G3 bed fixed 3000
  calls); note §4.2 / §5.1 / §8 G1 / G3 amended.
- **Gates now:** G0 hashes 13/13; flow ctests 63 + 130 (full battery at `fddcec7`), march_state +
  march_to_steady pass at `a6da9eb`; core 96 + 96; G2 42/42 identical to WO-8; G8 CUDA 3.4 % / 1.2 %;
  G1 14/14 old cases; G1 bed Δt 6 under the new criterion: pass (acc ≤ 2.1e-9 vs plain 1.5e-8).
- **STOPPED (reported, not tuned):** (1) G1 bed Δt 60: the plain tight march does NOT certify in
  20 000 steps (K 2.6e-7 off at 20 000), so the criterion has no K_plain; acc is 1.7e-8 (CUDA) /
  2.0e-8 (host) from K∞. (2) G3 bed fixed 3000: K spread passes (4.7e-9 CUDA / 5.5e-9 host), but
  Δt 60 is disabled by "too many restarts" on both backends (2699 / 1917 calls) and Δt 600 CUDA has
  3 restarts (2 per 100): status / restart criteria fail.

**Next action.** Caller: decide on G1 bed Δt 60 (no plain certificate) and on the G3 restart
criteria at the inexact-solve floor; then register entries, push core → tag → flow → umbrella.

**Anchors.** Note §4.3 "Restart (amended)", §7 "Why stagnation is measured against slow_rate",
§8 G1 / G3; core `AndersonCore::complete` (step 5 restart, step 6 after the broadcast); driver
`packaging/flow_steady.py` phase A; adapter `src/anderson_accelerator.hpp` step();
`tests/study/steady_acceleration_gates.py` `G3_DEPTH`; queue `<scratch>/fix/queue.sh`.
