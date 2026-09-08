# Repository guidelines for coding agents

`peclet-flow` — the `peclet.flow` cut-cell IBM incompressible Navier–Stokes solver. Everything an
agent needs is in one place: **read [`CLAUDE.md`](CLAUDE.md)** for the build and test recipes, the
module layout, the conventions and call orders, and the traps. This file exists only so that tools
looking for `AGENTS.md` find the pointer; it is not a second, diverging copy.

Short version:

- Build one tree per backend against `../extern/install/<backend>`, with
  `-DPECLET_FLOW_BUILD_TESTS=ON` (and `-DPECLET_FLOW_MPI=ON` for the distributed suites); run
  `ctest --test-dir <tree> -LE bench` with the OpenMP pool bounded.
- Device code is Kokkos C++ in `.hpp` headers — never a `.cu`. C++ is 2-space, `clang-format`
  18.1.8 with the repo `.clang-format`, and the check is **blocking** in CI. Python is 4-space and
  must pass ruff's critical-error set.
- **No environment variable may change a numerical result** (`../docs/QUALITY_PLAN.md` D3); add a
  solver setter instead. The ctest `no_env_knobs` enforces it.
- Design notes for shipped behaviour are in [`doc/`](doc/README.md); campaign records and work
  orders are in [`doc/history/`](doc/history/README.md) and are not maintained.
- Commits use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `test:`), stay scoped to one
  change, and state the numerical or physical intent.
