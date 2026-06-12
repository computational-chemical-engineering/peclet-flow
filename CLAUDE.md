# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated incompressible Navier-Stokes CFD solver for porous media simulations. Uses staggered MAC grid with Immersed Boundary Method (IBM) to handle complex geometries defined by Signed Distance Functions (SDF). The solver implements Newton-Raphson iteration with pressure projection.

## Build Commands

```bash
# Build the Python extension module
mkdir -p build && cd build && cmake .. && cmake --build .

# Output: build/pnm_backend.so
```

**Requirements:** CUDA Toolkit, CMake 3.18+, Python 3.10+, C++17

## Running Tests and Verification

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run unit tests
python tests/test_cfd_solver.py
python tests/test_implicit_poiseuille.py
python tests/test_fluid_fractions.py

# Run verification scripts
python scripts/verify_poiseuille.py      # Plane Poiseuille (analytical solution)
python scripts/verify_periodic_spheres.py # Flow around sphere packing
python scripts/verify_divergence.py       # Check incompressibility
```

## Architecture

### Core Components

- **`pnm_backend`** - Python extension module (built via pybind11)
- **`CFDSolver`** - Main solver class with GPU-accelerated kernels
- **`MacGrid`** - Staggered grid data structure (SoA layout for GPU coalescing)
- **`SDFData`/`SDFReader`** - Geometry loading from VTI files

### Memory Layout

- Linear indexing: `I = x + y*nx + z*nx*ny` (x is fastest)
- Python arrays: Fortran order `order='F'` with shape `(nx, ny, nz)`
- Periodic boundaries with wrapping: `(x % res.x + res.x) % res.x`

### Numerical Method

1. **Advection**: TVD scheme with Koren limiter
2. **Diffusion**: Crank-Nicolson (θ configurable 0.5-1.0)
3. **Momentum Solve**: Implicit Jacobian with Red-Black Gauss-Seidel
4. **Pressure Projection**: Poisson solve for auxiliary scalar φ
5. **IBM**: Robust Scaled method with D_rescale for near-wall handling

### Key Source Files

- `src/cfd_solver.cuh` - Data structures (MacGrid, IBM_Data) and public API
- `src/cfd_solver.cu` - Main solver implementation
- `src/cfd_solver_ibm.cu` - Immersed Boundary Method kernels
- `src/bindings.cpp` - Python/C++ interop via pybind11
- `doc/implementation_plan.md` - Detailed architecture documentation

## Python API Usage

```python
import pnm_backend

# Create solver
res = pnm_backend.int3(32, 32, 32)
spacing = pnm_backend.float3(dx, dx, dx)
solver = pnm_backend.CFDSolver(res, spacing)

# Initialize geometry from SDF
sdf_data = pnm_backend.SDFData(values, res, origin, spacing)
solver.initialize(sdf_data)

# Configure solver
solver.set_rho(1.0)
solver.set_mu(0.01)
solver.set_body_force(pnm_backend.float3(1e-2, 0, 0))
solver.set_cfl(0.5)
solver.set_pressure_solver_params(max_iter=500, tol=1e-5)
solver.set_velocity_solver_params(max_iter=50, tol=1e-5)
solver.set_diffusion_theta(1.0)  # 0.5=CN, 1.0=Implicit

# Time step
solver.step(dt)

# Get results (returns flat array, reshape with order='F')
u = np.array(solver.get_u()).reshape((nx, ny, nz), order='F')
```

## Conventions

- **SDF sign**: Negative inside solid, positive in fluid
- **CUDA kernels**: Named with `_kernel` suffix
- **Staggered grid**: u at (i+1/2,j,k), v at (i,j+1/2,k), w at (i,j,k+1/2), p at cell centers

## MPI distribution (transport-core integration)

There is a **complete, validated distributed incompressible Navier–Stokes solver** built on the shared
`transport-core` library (sibling repo `../transport-core`), opt-in and **separate from the production
`pnm_backend` module** (which is untouched). Full step-by-step status is in
[`doc/mpi_parallelization_status.md`](doc/mpi_parallelization_status.md).

Key pieces (all `src/*.cuh`, header-only, on branch `mpi-halo-integration`):
- `mac_halo.cuh` — `MacGridHalo`: decomposes the global MAC cell grid (ORB) into rank-owned blocks and
  exchanges a ghost layer (configurable width; 2 for the Koren advection reach) for `double`
  cell-fields on the extended local block. cfd's x-fastest indexing means the halo drops in directly.
- `staggered_advection.cuh` — `sadv::advect`: cfd's exact staggered Koren TVD advection, templated on a
  field accessor so the same operator serves the full grid and a local block.
- `distributed_ns.cuh` — `dns::DistributedNS`: the reusable solver. `step(n_diff, n_pois)`
  does per-component implicit diffusion (RB-GS, halo exchange between sweeps) + Chorin projection, with
  `set_advection(true)` (full Navier–Stokes), `set_body_force`, `set_solid` (no-slip masking), and
  `gather_to_root` (assemble the global field for VTI output via `tpx::geom`).

### Pressure solver options (the `sdflow` module / `DistributedNS`)

The canonical solver is exposed as the `sdflow` Python module (`src/sdflow_bindings.cu`, class
`sdflow.Solver`). Its cut-cell pressure Poisson is solved by a geometric **multigrid** whose smoother is
**Red-Black Gauss-Seidel** and whose coarse operator is the **rediscretized** cut-cell operator
(`mac_multigrid.cuh`). Three outer drivers wrap that V-cycle — **select one per solver**:

| driver | select with | use |
|---|---|---|
| **Standalone V-cycle** | default (neither below set) | multi-rank default. `set_pressure_multigrid(True, levels=1)` ⇒ pure RB-GS (no coarse grid) |
| **MG-PCG** | `set_pressure_pcg(True, max_iter, rtol)` | **single-GPU default** (auto-enabled on 1 rank); ~1.2× faster than the V-cycle to a fixed tolerance |
| **Chebyshev** | `set_pressure_chebyshev(True, max_iter, rtol)` | communication-light (no per-iteration global dot-products) — for large multi-GPU where PCG's reductions are latency-bound. ≈ PCG iteration count; bounds estimated once on step 1 |

- **PCG and Chebyshev are mutually exclusive** (last set wins); either overrides the single-rank auto-PCG
  default. With neither set, the solve is `n_pois` standalone V-cycles.
- Coarse-operator mode: `set_solid(..., pressure_coarse="rediscretized")` (default; also `"galerkin"` /
  `"const"`). `set_pressure_multigrid(on, levels)` sets the multigrid depth (`levels=1` == pure RB-GS).
- `set_pressure_warmstart(True)` seeds each solve from the previous step's φ (opt-in, off by default).
- Validated against Zick & Homsy SC-sphere drag, bit-identical to `pnm_backend`. Design + benchmarks:
  [`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md); parity: [`doc/sdflow_pnm_parity.md`](doc/sdflow_pnm_parity.md).

Validated cell-for-cell vs serial and against analytics (Taylor–Green ~2e-15, Poiseuille, momentum
conservation), **36/36 ctests, real multi-rank np=1,2,4**.

Build/test (opt-in; default module build untouched):
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/build/_deps/pybind11-src
cmake --build build_mpi -j
ctest --test-dir build_mpi --output-on-failure
```
**Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`** — FindMPI may pick ParaView's bundled `mpiexec` on
`PATH`, which launches the OpenMPI-linked test binaries as singletons (so `*_np4` silently runs 4×np=1).

**Remaining (largest task):** the *in-place* `cfd_solver.cu` rewrite — extended-block state/scratch +
MPI global reductions (`max_abs`, `remove_mean`, pressure pin) + distributed **multigrid** (halo per
level, restriction/prolongation across blocks) + the full Robust-Scaled cut-cell IBM (vs masking). The
`DistributedNS` solver is the working single-level path and proves the mechanism end to end.
