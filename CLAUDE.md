# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated incompressible Navier-Stokes CFD solver for porous media simulations. Uses a staggered MAC grid with the Immersed Boundary Method (IBM) over complex geometries defined by Signed Distance Functions (SDF), with cut-cell pressure projection.

**Two solvers live in this repo:**
- **`pnm_backend`** (`src/cfd_solver*.cu`, single-GPU) — the original implementation, validated against the Zick & Homsy sphere-array Stokes drag. Now kept as the **numerical reference** (git tag `pnm_backend-reference`); not deleted, retiring it needs explicit sign-off.
- **`sdflow`** (`src/sdflow_bindings.cu` → `dns::DistributedNS` in `src/distributed_ns.cuh`) — the **canonical solver going forward**: the same cut-cell IBM physics, MPI-optional (single-GPU by default), with a grid-independent geometric **multigrid** pressure solve. Validated **bit-identical to `pnm_backend`** against Zick & Homsy. See the "MPI / sdflow" section below, [`doc/sdflow_pnm_parity.md`](doc/sdflow_pnm_parity.md), and [`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md).

Much of the detail below (CFDSolver/MacGrid internals, the pnm_backend Python API) describes the **reference** solver; for new work use `sdflow`.

## Build Commands

```bash
# Default build (single-GPU, no MPI): builds BOTH the reference and the canonical module
cmake -S . -B build && cmake --build build -j
# Output: build/pnm_backend.so (reference) + build/sdflow.so (canonical, single-rank, TPX_NO_MPI)

# Multi-rank build (opt-in): adds the MPI-linked sdflow + the *_mpi ctest suite (see the MPI section)
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun && cmake --build build_mpi -j
```

(An existing `build/` may predate the `sdflow` target and hold only `pnm_backend.so` — reconfigure to pick up `sdflow`.)

**Requirements:** CUDA Toolkit, CMake 3.18+, Python 3.10+, C++17 (+ MPI & `../transport-core` for the multi-rank build)

## Running Tests and Verification

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run unit tests
python tests/test_cfd_solver.py
python tests/test_implicit_poiseuille.py
python tests/test_fluid_fractions.py

# Run verification scripts (pnm_backend, the reference)
python scripts/verify_poiseuille.py      # Plane Poiseuille (analytical solution)
python scripts/verify_periodic_spheres.py # Flow around sphere packing
python scripts/verify_divergence.py       # Check incompressibility
```

For the canonical **`sdflow`** solver (built into `build_mpi/`), drive verification from Python and run
the C++ ctest suite:
```bash
PYTHONPATH=$PWD/build_mpi python scripts/verify_periodic_spheres_sdflow.py   # cut-cell Stokes (RB-GS)
PYTHONPATH=$PWD/build_mpi python scripts/verify_poiseuille_sdflow.py         # analytic parabola
PYTHONPATH=$PWD/build_mpi python scripts/validate_zick_homsy_sdflow_vs_pnm.py # ground-truth vs Z&H + pnm
ctest --test-dir build_mpi --output-on-failure                              # 72 multi-rank tests (np=1,2,4)
mpirun -np 1 ./build_mpi/profile_mg_scaling 32 64 128                       # pressure-solver scaling/timing
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

**`pnm_backend` (reference):**
- `src/cfd_solver.cuh` - Data structures (MacGrid, IBM_Data) and public API
- `src/cfd_solver.cu` - Main solver implementation
- `src/cfd_solver_ibm.cu` - Immersed Boundary Method kernels
- `src/cfd_solver_multigrid.cu` - geometric multigrid (rediscretized cut-cell coarse operator, RB-GS)
- `src/bindings.cpp` - Python/C++ interop via pybind11

**`sdflow` (canonical):**
- `src/distributed_ns.cuh` - `DistributedNS`: the solver (diffusion, projection, three pressure drivers)
- `src/mac_multigrid.cuh` - `DistributedPoissonMG`: distributed MG (V-cycle / PCG / Chebyshev), coarse-op modes
- `src/mac_ibm.cuh`, `src/mac_cutcell.cuh`, `src/mac_halo.cuh`, `src/mac_reductions.cuh` - IBM stencil, cut-cell openness, halo, reductions
- `src/sdflow_bindings.cu` - the `sdflow` pybind module (`sdflow.Solver`)

## Python API Usage (pnm_backend, the reference)

For the canonical `sdflow` API see the "Pressure solver options" table and `scripts/*_sdflow.py`; the
`pnm_backend` reference API is below.

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

## MPI / sdflow (the canonical solver, transport-core integration)

The **`sdflow`** module is the canonical incompressible Navier–Stokes solver, built on the shared
`transport-core` library (sibling repo `../transport-core`). It is **MPI-optional** — single-GPU by
default, multi-rank with `-DCFD_BUILD_MPI=ON` — and is validated **bit-identical to `pnm_backend`**
(which it is converging to replace; `pnm_backend` stays as the reference). Full status:
[`doc/mpi_parallelization_status.md`](doc/mpi_parallelization_status.md) and
[`doc/sdflow_pnm_parity.md`](doc/sdflow_pnm_parity.md).

Key pieces (all `src/*.cuh`, header-only, on `main`):
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

### Domain boundary conditions

Beyond periodic + IBM no-slip on immersed solids, sdflow has **native per-face domain BCs** (`mac_bc.cuh`):
`set_domain_bc(face, type, vx, vy, vz)` for the 6 faces (0=−x,1=+x,2=−y,3=+y,4=−z,5=+z); `type` 0=periodic
(default), 1=no-slip wall, 2=Dirichlet velocity / inflow, 3=outflow. Velocity ghosts are filled in the
MAC-staggered convention. Tangential walls use a **face-fold** in the implicit diffusion (drop the wall
face, fold its β into the diagonal + RHS) so `u_inner` stays implicit — no Gauss–Seidel lag; explicit
advection keeps the reflection ghost. Call **before** geometry/first step. For a domain-BC problem with no
immersed solid, use `set_pressure_geometry(all_fluid_sdf)` (the cut-cell pressure operator without the IBM).

**Open boundaries** (outflow, or inflow with a non-zero normal velocity) split the face openness into two
roles: the **operator** openness α (pressure matrix) is 0 at walls + inflow (Neumann) and open at outflow
(Dirichlet p=0, ghost held at 0 → non-singular, mean-removal off); the **flux** openness β
(divergence/correction) stays open at inflow + outflow so their flux is counted. Outflow velocity is
zero-gradient (∂/∂n=0); the projection corrects the outflow face so mass leaves.

**Non-uniform inlets:** `set_domain_bc_profile(face, profile[Nb,Nc,3])` prescribes a per-position inlet
velocity over the face's perpendicular plane (sets the face to inflow). Used for a parabolic channel inlet
or the **backward-facing step**, whose step is realized purely as the inlet condition — the developed
parabola over the open upper half, zero over the step face (no immersed solid needed).

**Validated:** lid-driven cavity vs Ghia et al. Re=100 to ~0.7% rms (`scripts/verify_lid_cavity_sdflow.py`);
developing plane channel (uniform inlet → parabolic Poiseuille outlet, `u_max/U_mean`→1.5, exact mass
conservation, machine-precision divergence; `scripts/verify_channel_sdflow.py`); backward-facing step
(Gartling expansion-ratio-2, `scripts/verify_bfs_sdflow.py`) — reattachment `x_r/S` 5.3 (Re_S=100) → 8.3
(Re_S=200) on the Armaly/Biswas curve, `SDFLOW_BFS_RE800=1` pushes to the Gartling Re=800 benchmark.

The **rediscretized geometric pressure multigrid is multilevel on these non-periodic domains** (not just the
periodic/IBM case): each coarse level re-imposes the boundary face openness (Neumann wall/inflow → 0,
Dirichlet outflow → open) and the trilinear prolongation fills the non-periodic boundary ghosts
(Neumann → zero-gradient, Dirichlet → 0). Gated on `has_bc_`, so the periodic/IBM path is byte-identical.
Convergence is grid-independent — e.g. a 256×64 channel at a fixed 10 V-cycles/step drives the flux
divergence from `2e-3` (1 level) to `5e-7` (3 levels) at ~the same cost.

**Semi-coarsening** handles thin (quasi-2D) grids: uniform 2:1 coarsening caps an `nz=4` grid at 2 levels,
so `init(..., semi=true)` halves an axis only while it stays even and ≥2 — a thin axis freezes while the
wide axes keep coarsening (`MGLevel::ratio`/`cfac`; the transfer + openness kernels take a per-axis
`int3 ratio`, the operator uses per-axis `idx2/cfac²`). The solver enables it only for native-BC problems
(`has_domain_bc_`, `semi_level_count`); the periodic/IBM porous path stays uniform + `clamp_levels`, so it
is byte-identical (72/72). A quasi-2D 256×64×4 channel now builds up to 8 levels (was 2): raw V-cycle flux
divergence at a fixed 8 cycles drops `1.7e-4`→`8.6e-13`. The BC verify scripts request `levels=8`
(auto-capped). *Follow-ups:* convective outflow for unsteady wakes, multi-rank inlet-profile scatter
(validated single-rank).

Validated cell-for-cell vs serial and against analytics (Taylor–Green ~2e-15, Poiseuille, momentum
conservation) **and against Zick & Homsy sphere-array drag (bit-identical to `pnm_backend`)** —
**72/72 ctests, real multi-rank np=1,2,4**.

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

**Status:** `DistributedNS`/`sdflow` is the full solver, not a prototype — extended-block state, MPI
global reductions, the Robust-Scaled cut-cell IBM, and a grid-independent geometric **multigrid**
pressure solve (rediscretized cut-cell coarse operator; three selectable outer drivers — see the
"Pressure solver options" table above) are all done and validated bit-identical to `pnm_backend` against
Zick & Homsy. **Remaining toward retiring `pnm_backend`** (the only open items): the large-np scaling
work — an **agglomerated coarse solve** and the communication-light **Chebyshev** accelerator's at-scale
benchmark, both needing real multi-GPU hardware (designs in [`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md)) — plus an explicit
sign-off. `pnm_backend` is **not** deleted; tag `pnm_backend-reference` is the restore point.
