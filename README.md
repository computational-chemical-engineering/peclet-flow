# flow

[![PyPI version](https://img.shields.io/pypi/v/peclet-flow.svg)](https://pypi.org/project/peclet-flow/)
[![Python versions](https://img.shields.io/pypi/pyversions/peclet-flow.svg)](https://pypi.org/project/peclet-flow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-flow/actions/workflows/ci.yml)

GPU-accelerated incompressible **Navier–Stokes** solver for flow in complex geometry, built around a
staggered **MAC** grid, a signed-distance-field (**SDF**) description of the solid, a cut-cell **Immersed
Boundary Method**, and a pressure-projection step with a geometric **multigrid** Poisson solve. The code is
written in **Kokkos** C++ (one source runs on the CUDA, HIP/AMD, and OpenMP backends, selected at build
time) and exposed to Python through **nanobind** (zero-copy, on `core`'s View↔ndarray bridge);
simulations are driven from Python.

> The repository is also known as `pnm_from_sdf` (its GitLab origin) — it computes pore-network–scale flow
> directly from segmented SDF geometry.

## Modules

| Module | Role |
|--------|------|
| **`flow`** | **The CFD solver** — a **distributed** (MPI-optional) GPU cut-cell IBM Navier–Stokes solver in physical units, built on the shared `core` block-decomposition + async halo layer. One code / one API / MPI-optional, with native domain boundary conditions. Exposes `peclet.flow.Solver` (staggered MAC, default) and `peclet.flow.SolverColocated` (collocated/cell-centered velocities, ABC approximate projection) — identical API via a `GridLayout` policy. Validated against analytics and **Zick & Homsy** sphere-array drag (`scripts/validate_zick_homsy_sdflow.py`). |
| **`pnm`** | **Pore-network extraction** — SDF VTI reading + pore/segmentation/topology extraction (`SDFReader`, `extract_pores`, `segment_volume`, `extract_topology_gpu`). The repo's namesake "pnm_from_sdf" feature. |

The original CUDA implementation has been **retired** (Kokkos became canonical, 2026-06); `flow` was
validated bit-identical to the CUDA solver — to machine precision, and against Zick & Homsy sphere-array
drag — before the CUDA sources were deleted (restore point: git tag `pre-cuda-retirement`). The shared
cut-cell IBM primitives now live in `src/cut_cell_ibm.hpp`; the operator headers are `src/mac_*.hpp` +
`src/flow_ibm.hpp`.

## Capabilities

- **Geometry:** SDF solids (negative inside); the cut-cell IBM applies a Robust-Scaled no-slip / moving-wall
  condition and a matching cut-cell pressure operator (face openness from the SDF).
- **Native domain boundary conditions** (`flow`): per-face periodic / no-slip wall / Dirichlet velocity
  (inflow) / outflow, plus per-position **inlet velocity profiles**. Validated on the lid-driven cavity
  (Ghia et al.), the developing plane channel (Poiseuille), and the backward-facing step (Armaly/Gartling).
- **Pressure multigrid:** rediscretized geometric V-cycle, grid-independent, with MG-PCG and Chebyshev
  outer accelerators. Works on periodic, IBM, and non-periodic (BC) domains, including **semi-coarsening**
  for thin (quasi-2D) grids.
- **Time integration:** pressure projection with optional incremental pressure, explicit (Koren) or
  implicit-deferred-correction advection, and Picard outer iteration.

## Build

```bash
# Canonical: build + install both modules via scikit-build-core
CMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>" pip install .   # -> flow + pnm
# Or a dev cmake build (nanobind found via the active interpreter, no cmakedir needed):
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>" && cmake --build build -j
# distributed flow build (opt-in MPI):
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON -DCMAKE_PREFIX_PATH="$PWD/../extern/install/<backend>" \
  && cmake --build build_mpi -j
```

`<backend>` is one of `nvidia-cuda` / `host-openmp` / `lumi-hip` under `../extern/install/`, produced once
by `../tools/bootstrap_deps.sh` (a hard build dependency). Requirements: a Kokkos backend (CUDA/HIP/OpenMP
— CUDA is just one option, not required), a C++20 host compiler, **nanobind + scikit-build-core**, and —
for distributed `flow` — MPI. Python dependencies live in a virtual environment (`.venv`).

## Run / verify

Simulations are scripts, not C++ mains. The `scripts/verify_*_sdflow.py` files are the canonical
verification entry points:

```bash
source .venv/bin/activate
python scripts/verify_lid_cavity_sdflow.py     # lid-driven cavity vs Ghia, Ghia & Shin (1982)
python scripts/verify_channel_sdflow.py        # developing plane channel -> Poiseuille
python scripts/verify_bfs_sdflow.py            # backward-facing step (reattachment length)
ctest --test-dir build_mpi --output-on-failure # the multi-rank C++ test suite
```

## Documentation

API documentation (C++ classes/kernels and Python scripts) is generated with **Doxygen** and published to
GitHub Pages by the `Documentation` CI workflow. Build it locally with:

```bash
doxygen docs/Doxyfile      # output in docs/html/index.html
```

The architecture, conventions, and design rationale are described in `CLAUDE.md` and the design notes
under `doc/` in the repository.
