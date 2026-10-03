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
- **Gates now:** G0 hashes 13/13, flow ctests 63 + 130, core 96 + 96 pass; G1 14/14 old cases pass
  (bed stag ν dt/h² 6: 695 CUDA / 809 host steps, ≤ 2.0e-9 vs K∞); G2 42/42 identical to WO-8; G8
  CUDA 3.4 % / 1.2 %; unit digests identical except U5 (its one restart) and the new U11.
- **FAILING (reported, not tuned):** (1) new G1 case, staggered bed ν dt/h² = 60 tight: certified
  in 724 (CUDA) / 686 (host) steps, |K/K∞ − 1| = 1.7e-8 / 2.0e-8 > 1e-8. (2) G3 bed K agreement at
  depth 5e-11: spread 2.5e-8 (CUDA) / 3.1e-8 (host); Δt 600 / 1e4 sit 2.6–4.1e-8 from K∞ — the
  R3 ratio premise (37–170) does not hold at large Δt (500–830).
- **Open (not implemented):** R5's staggered `set_advection` invalidation — the staggered field
  list never changes, so it needs a new configuration key; a design choice for the caller.

**Next action.** Caller: decide on the two failing bars (G1 bed Δt 60, G3 bed depth/spread) and on
the staggered-advection signature; then register entries, push core → tag → flow → umbrella.

**Anchors.** Note §4.3 "Restart (amended)", §7 "Why stagnation is measured against slow_rate",
§8 G1 / G3; core `AndersonCore::complete` (step 5 restart, step 6 after the broadcast); driver
`packaging/flow_steady.py` phase A; adapter `src/anderson_accelerator.hpp` step();
`tests/study/steady_acceleration_gates.py` `G3_DEPTH`; queue `<scratch>/fix/queue.sh`.
