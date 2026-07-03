# Repository Guidelines

## Project Structure & Module Organization
- `src/`: header-only **Kokkos** C++ sources for the CFD solver + **nanobind** bindings (`.hpp` + `.cpp`; no `.cu/.cuh` — CUDA retired).
- `tests/`: C++ Kokkos kernel tests (`tests/kokkos/`) + multi-rank ctests (`tests/kokkos_mpi/test_*.cpp`) + a couple of pore-extraction Python scripts.
- `scripts/`: `flow` verification/analysis scripts (e.g., `verify_poiseuille_flow.py`).
- `build/`: CMake build output (expects `build/peclet.flow.*.so` + `build/pnm.*.so`).
- `doc/`, `notebooks/`, `data/`: design notes, experiments, and input datasets.

## Build, Test, and Development Commands
```bash
# Build/install the Python extensions via scikit-build-core (Kokkos prefix from ../tools/bootstrap_deps.sh)
CMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>" pip install .
# Or a dev cmake build (nanobind found via the active interpreter; CMake 3.24+):
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>" && cmake --build build -j

# Activate venv and run flow verification (canonical scripts are scripts/verify_*_sdflow.py)
source .venv/bin/activate
PYTHONPATH=$PWD/build python scripts/verify_poiseuille_flow.py
PYTHONPATH=$PWD/build python scripts/verify_periodic_spheres_sdflow.py
PYTHONPATH=$PWD/build python scripts/verify_lid_cavity_sdflow.py
```

## Coding Style & Naming Conventions
- C++ uses 2-space indentation and K&R-style braces; device code is plain Kokkos C++ in `.hpp` headers.
- Python uses 4-space indentation.
- Files and functions use `snake_case`; types/classes use `PascalCase`.
- Kokkos device work is `parallel_for`/`parallel_reduce` over `Kokkos::View`s; keep functor types suffixed `_kernel`/`_op`.

## Testing Guidelines
- Tests are executable Python scripts under `tests/`.
- Name tests as `test_*.py` and keep import paths pointed at `build/`.
- No formal coverage target; prefer adding a small validation script when fixing numerics.

## Commit & Pull Request Guidelines
- Recent history uses Conventional Commit prefixes (`feat:`, `fix:`); follow this when possible.
- Keep commits scoped to one change; include the numerical/physical intent in the message.
- PRs should describe the change, list commands run, and attach plots or outputs if results change.

## Agent Notes
- For more detailed architecture and build guidance, see `CLAUDE.md`.
