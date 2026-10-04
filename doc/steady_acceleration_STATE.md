# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (rev 2 + review fixes): Anderson acceleration of
steady marches, `peclet.flow.march_to_steady`. History and every number:
`doc/steady_acceleration_log.md`.

**Where we are (2026-10-03): LANDED.**
- **core 1.4.0 RELEASED** (user GO 2026-10-03): core main `c656ccb`, tag `v1.4.0`, peclet-halo +
  peclet-core 1.4.0 on PyPI. Gates on that tip: host+MPI 77/77, Kokkos CUDA+MPI 97/97.
- **flow main** carries `march_to_steady` (branch `anderson` rebased onto VoF's `5a34c69`, pin
  `PECLET_CORE_TAG v1.4.0`, which supersedes VoF's v1.3.2). Re-gate on the combined tree against
  the tagged headers: host serial 63/63, MPI 130/130, CUDA march_* 2/2, G0 hashes 13/13 + np2
  byte-identical to the pre-rebase head.
- **Umbrella**: register (2 superseded, 5 flow + 1 core new; 593 -> 599), CLAUDE.md counts,
  CHANGELOG core 1.4.0, pointers core + flow.
- **Q18 DECIDED (user, 2026-10-03): stationarity only.** MarchResult docstring states the
  caveats: uniqueness holds for Stokes only (note §2.3 scope line added); with advection Anderson
  can land on an unstable steady branch; the stop test under-reads slow tails (~1.4e-8 at rtol
  1e-10 on the dense bed — open limitation).

**Next action.** None in this campaign. flow's next release (1.3.0 in the pending family 1.4.0)
ships march_to_steady; its CHANGELOG entry goes under [Unreleased] at that release.

**Anchors.** Note §4.3 "Restart (amended)", §7 "Why stagnation is measured against slow_rate",
§8 G1 / G3; core `AndersonCore::complete` (step 5 restart, step 6 after the broadcast); driver
`packaging/flow_steady.py` phase A; adapter `src/anderson_accelerator.hpp` step();
`tests/study/steady_acceleration_gates.py` `G3_DEPTH`; queue `<scratch>/fix/queue.sh`.
