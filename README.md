# cfd-gpu

GPU-accelerated incompressible **Navier–Stokes** solver for flow in complex geometry, built around a
staggered **MAC** grid, a signed-distance-field (**SDF**) description of the solid, a cut-cell **Immersed
Boundary Method**, and a pressure-projection step with a geometric **multigrid** Poisson solve. The code is
written in CUDA C++ and exposed to Python through `pybind11`; simulations are driven from Python.

> The repository is also known as `pnm_from_sdf` (its GitLab origin) — it computes pore-network–scale flow
> directly from segmented SDF geometry.

## Two solvers

| Module | Role |
|--------|------|
| **`pnm_backend`** | The production solver — GPU cut-cell IBM Navier–Stokes for porous media. Validated cell-for-cell against analytics and against **Zick & Homsy** sphere-array drag. |
| **`sdflow`** | The canonical **distributed** (MPI-optional) solver built on the shared `transport-core` block-decomposition + async halo layer. One code / one API / MPI-optional. It reproduces `pnm_backend`'s physics and adds native domain boundary conditions. |

`sdflow` is the convergence target that will eventually replace `pnm_backend`; both are kept in sync and
cross-validated (`scripts/cross_validate_sdflow_vs_pnm.py`, `scripts/validate_zick_homsy_sdflow_vs_pnm.py`).

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
mkdir -p build && cd build && cmake .. && cmake --build .   # -> build/pnm_backend.so
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
