# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++/CUDA sources for the CFD solver and pybind11 bindings (`.cpp`, `.cu`, `.cuh`).
- `tests/`: C++ multi-rank ctests (`tests/kokkos_mpi/test_*.cpp`) + a couple of pore-extraction Python scripts.
- `scripts/`: `sdflow` verification/analysis scripts (e.g., `verify_poiseuille_sdflow.py`).
- `build/`: CMake build output (expects `build/sdflow.*.so` + `build/pnm.*.so`).
- `doc/`, `notebooks/`, `data/`: design notes, experiments, and input datasets.

## Build, Test, and Development Commands
```bash
# Build the Python extensions (requires CUDA Toolkit + CMake 3.18+)
cmake -S . -B build && cmake --build build -j

# Activate venv (if used) and run sdflow verification
source .venv/bin/activate
PYTHONPATH=$PWD/build python scripts/verify_poiseuille_sdflow.py
PYTHONPATH=$PWD/build python scripts/verify_periodic_spheres_sdflow.py

# Run verification scripts
python scripts/verify_poiseuille.py
python scripts/verify_periodic_spheres.py
python scripts/verify_divergence.py
```

## Coding Style & Naming Conventions
- C++/CUDA uses 2-space indentation and K&R-style braces.
- Python uses 4-space indentation.
- Files and functions use `snake_case`; types/classes use `PascalCase`.
- Keep CUDA kernels suffixed with `_kernel` and headers as `.cuh`/`.h`.

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
