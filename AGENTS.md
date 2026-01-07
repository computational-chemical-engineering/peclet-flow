# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++/CUDA sources for the CFD solver and pybind11 bindings (`.cpp`, `.cu`, `.cuh`).
- `tests/`: Python test scripts (`test_*.py`) that import the built extension.
- `scripts/`: Verification and analysis scripts (e.g., `verify_poiseuille.py`).
- `build/`: CMake build output (expects `build/pnm_backend.so`).
- `doc/`, `notebooks/`, `data/`: design notes, experiments, and input datasets.

## Build, Test, and Development Commands
```bash
# Build the Python extension (requires CUDA Toolkit + CMake 3.18+)
mkdir -p build && cd build && cmake .. && cmake --build .

# Activate venv (if used) and run core tests
source .venv/bin/activate
python tests/test_cfd_solver.py
python tests/test_implicit_poiseuille.py
python tests/test_fluid_fractions.py

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
