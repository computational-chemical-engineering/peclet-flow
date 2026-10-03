# Steady-march Anderson acceleration — STATE (rewritten in place)

**Objective.** Execute `doc/steady_acceleration.md` (rev 2 + review fixes): Anderson acceleration of
steady marches, `peclet.flow.march_to_steady`. History and every number:
`doc/steady_acceleration_log.md`.

**Where we are (2026-10-03, landing).**
- **core 1.4.0 RELEASED** (user GO 2026-10-03): core main `c656ccb` = 1.3.2 merge `5763934` +
  version bump, tag `v1.4.0`, peclet-halo + peclet-core 1.4.0 on PyPI. Gates on that tip:
  host+MPI 77/77, Kokkos CUDA+MPI 97/97; landing pages OK. Release worktree `suite/core-release-1.4.0`.
- **flow `anderson`** rebased onto origin/main (2 doc-only commits, clean) + `f1de647` pins
  `PECLET_CORE_TAG v1.4.0`; builds read core from `../core-release-1.4.0` (= the tag).
- **Re-gate in progress** (`<scratch>/flow_battery.sh`): build_omp serial + MPI ctests, G0 hashes
  vs the previous session's `fix/hash2_{serial,mpi}.txt`, build_cuda `march_*`.
- **Umbrella** worktree `.claude/worktrees/anderson-register`, branch `anderson-register`,
  `7baa4c6` (local): register (2 superseded, 5 flow + 1 core new, 593 -> 599), CLAUDE.md counts,
  CHANGELOG core 1.4.0. Pointer bumps (core `c656ccb`, flow head) come after flow is pushed.
- Previous gates (pre-rebase head): flow host serial 63/63, MPI 130/130, CUDA march_* 2/2, G0 13/13,
  G2 42/42, G8 CUDA 3.4 % / 1.2 %, G1 14/14 + bed, G3 bed K agreement (restarts documented limitation).
- **Q18 OPEN (user)**: converged=True promises stationarity only (documented default) vs also
  plain-march reachability. Recorded as open in the register; "also reachability" -> architect.

**Next action.** Battery green -> push flow main (ff-only) -> umbrella: gitlinks core + flow on
`anderson-register`, ff onto origin/main, push LAST -> remove worktrees core-anderson,
core-release-1.4.0, flow-anderson, anderson-register.

**Anchors.** Note §4.3 "Restart (amended)", §7 "Why stagnation is measured against slow_rate",
§8 G1 / G3; core `AndersonCore::complete` (step 5 restart, step 6 after the broadcast); driver
`packaging/flow_steady.py` phase A; adapter `src/anderson_accelerator.hpp` step();
`tests/study/steady_acceleration_gates.py` `G3_DEPTH`; queue `<scratch>/fix/queue.sh`.
