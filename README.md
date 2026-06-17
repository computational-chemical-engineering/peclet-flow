# cfd-gpu

GPU-accelerated incompressible **Navier–Stokes** solver for flow in complex geometry, built around a
staggered **MAC** grid, a signed-distance-field (**SDF**) description of the solid, a cut-cell **Immersed
Boundary Method**, and a pressure-projection step with a geometric **multigrid** Poisson solve. The code is
written in CUDA C++ and exposed to Python through `pybind11`; simulations are driven from Python.

> The repository is also known as `pnm_from_sdf` (its GitLab origin) — it computes pore-network–scale flow
> directly from segmented SDF geometry.

## Modules

| Module | Role |
|--------|------|
| **`sdflow`** | **The CFD solver** — a **distributed** (MPI-optional) GPU cut-cell IBM Navier–Stokes solver in physical units, built on the shared `transport-core` block-decomposition + async halo layer. One code / one API / MPI-optional, with native domain boundary conditions. Validated against analytics and **Zick & Homsy** sphere-array drag (`scripts/validate_zick_homsy_sdflow.py`). |
| **`pnm_backend`** | **Pore-network extraction** — SDF VTI reading + pore/segmentation/topology extraction (`SDFReader`, `extract_pores`, `segment_volume`, `extract_topology_gpu`). The repo's namesake "pnm_from_sdf" feature. |

The original single-GPU `CFDSolver` reference (`src/cfd_solver*.cu`) has been **retired**; `sdflow` was
validated bit-identical to it before removal (restore point: git tag `pnm_backend-reference`). The shared
cut-cell IBM primitives now live in `src/cut_cell_ibm.cuh`.

## Capabilities

- **Geometry:** SDF solids (negative inside); the cut-cell IBM applies a Robust-Scaled no-slip / moving-wall
  condition and a matching cut-cell pressure operator (face openness from the SDF).
- **Native domain boundary conditions** (`sdflow`): per-face periodic / no-slip wall / Dirichlet velocity
  (inflow) / outflow, plus per-position **inlet velocity profiles**. Validated on the lid-driven cavity
  (Ghia et al.), the developing plane channel (Poiseuille), and the backward-facing step (Armaly/Gartling).
- **Pressure multigrid:** rediscretized geometric V-cycle, grid-independent, with MG-PCG and Chebyshev
  outer accelerators. Works on periodic, IBM, and non-periodic (BC) domains, including **semi-coarsening**
  for thin (quasi-2D) grids.
- **Time integration:** pressure projection with optional incremental pressure, explicit (Koren) or
  implicit-deferred-correction advection, and Picard outer iteration.

## Build

```bash
cmake -S . -B build && cmake --build build -j   # -> build/sdflow.so (CFD solver) + build/pnm_backend.so (pore extraction)
# distributed sdflow build (opt-in MPI):
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON && cmake --build build_mpi -j   # -> build_mpi/sdflow*.so
```

Requirements: CUDA (the device arch is pinned for the dev box's RTX 5080), a C++17/20 host compiler,
`pybind11`, and — for `sdflow` — MPI. Python dependencies live in a virtual environment (`.venv`).

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
