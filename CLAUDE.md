# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated incompressible Navier-Stokes CFD solver for porous media simulations. Uses a staggered MAC grid with the Immersed Boundary Method (IBM) over complex geometries defined by Signed Distance Functions (SDF), with cut-cell pressure projection.

**`sdflow` is THE solver** (`src/sdflow_bindings.cu` → `dns::DistributedNS` in `src/distributed_ns.cuh`):
cut-cell IBM physics on a staggered MAC grid, MPI-optional (single-GPU by default), with a grid-independent
geometric **multigrid** pressure solve. It solves the equations in **physical units** (density `rho`,
dynamic viscosity `mu`, physical pressure `p`). See the "MPI / sdflow" section below and
[`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md).

The original single-GPU `CFDSolver` reference (`src/cfd_solver*.cu`) has been **retired** — `sdflow` was
validated bit-identical to it (and to the Zick & Homsy sphere-array Stokes drag) before its removal; the
restore point is the git tag `pnm_backend-reference`. The shared cut-cell IBM primitives it provided
(`get_idx`, `IBM_Data`, the Robust-Scaled `poly_*`) now live in `src/cut_cell_ibm.cuh`.

**`pnm_backend` is now the pore-network-extraction module** (`src/bindings.cpp` + `src/pore_extraction.cu`):
`SDFReader`, `extract_pores`, `segment_volume`, `extract_topology_gpu` — the repo's namesake "pnm_from_sdf"
feature, unrelated to the CFD solve.

## Build Commands

```bash
# Default build (single-GPU, no MPI): the sdflow solver + the pnm_backend pore-extraction module
cmake -S . -B build && cmake --build build -j
# Output: build/sdflow.so (the CFD solver, single-rank, TPX_NO_MPI) + build/pnm_backend.so (pore extraction)

# Multi-rank build (opt-in): adds the MPI-linked sdflow + the *_mpi ctest suite (see the MPI section)
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun && cmake --build build_mpi -j
```

**Requirements:** CUDA Toolkit, CMake 3.18+, Python 3.10+, C++17 (+ MPI & `../transport-core` for the multi-rank build)

## Running Tests and Verification

Drive `sdflow` verification from Python and run the C++ ctest suite:
```bash
source .venv/bin/activate
PYTHONPATH=$PWD/build_mpi python scripts/verify_periodic_spheres_sdflow.py   # cut-cell Stokes
PYTHONPATH=$PWD/build_mpi python scripts/verify_poiseuille_sdflow.py         # analytic parabola
PYTHONPATH=$PWD/build_mpi python scripts/verify_lid_cavity_sdflow.py         # lid cavity vs Ghia
PYTHONPATH=$PWD/build_mpi python scripts/verify_channel_sdflow.py            # developing channel
PYTHONPATH=$PWD/build_mpi python scripts/verify_bfs_sdflow.py               # backward-facing step
PYTHONPATH=$PWD/build_mpi python scripts/validate_zick_homsy_sdflow.py       # external ground truth (Z&H drag)
ctest --test-dir build_mpi --output-on-failure                              # 72 multi-rank tests (np=1,2,4)
mpirun -np 1 ./build_mpi/profile_mg_scaling 32 64 128                       # pressure-solver scaling/timing
```

Pore-network extraction (the `pnm_backend` module): `python scripts/test_extraction.py`,
`python scripts/verify_segmentation.py`.

## Architecture

### Memory Layout

- Linear indexing: `I = x + y*nx + z*nx*ny` (x is fastest)
- Python arrays: Fortran order `order='F'` with shape `(nx, ny, nz)`
- Periodic boundaries with wrapping: `(x % res.x + res.x) % res.x`

### Numerical Method (`sdflow`)

The physical incompressible momentum equation `rho*(du/dt + (u.grad)u) = -grad(p) + mu*Lap(u) + f`, solved
each step (semi-implicit), **scaled by 1/dt** (the "divided" convention — the operator is `(rho/dt)*I -
mu*Lap`, well-conditioned at large dt / steady state):

1. **Advection**: explicit Koren TVD, or implicit-FOU + deferred-correction TVD (`set_implicit_advection`)
2. **Diffusion**: backward-Euler implicit, Red-Black Gauss-Seidel (or opt-in velocity multigrid)
3. **Pressure projection**: `u = u* - grad(phi)`, `Lap(phi) = div(u*)`; physical pressure `p = (rho/dt)*phi`,
   accumulated rotationally under the default incremental-pressure scheme. Geometric multigrid (V-cycle /
   MG-PCG / Chebyshev).
4. **IBM**: Robust-Scaled cut-cell method with D_rescale for near-wall handling.

### Key Source Files

**`sdflow` (the CFD solver):**
- `src/distributed_ns.cuh` - `DistributedNS`: the solver (diffusion, projection, three pressure drivers)
- `src/mac_multigrid.cuh` - `DistributedPoissonMG`: distributed MG (V-cycle / PCG / Chebyshev), coarse-op modes
- `src/mac_ibm.cuh`, `src/mac_cutcell.cuh`, `src/mac_halo.cuh`, `src/mac_reductions.cuh` - IBM stencil, cut-cell openness, halo, reductions
- `src/cut_cell_ibm.cuh` - shared cut-cell IBM primitives (`get_idx`, `IBM_Data`, Robust-Scaled `poly_*`)
- `src/staggered_advection.cuh` - staggered Koren TVD advection
- `src/sdflow_bindings.cu` - the `sdflow` pybind module (`sdflow.Solver`)

**`pnm_backend` (pore-network extraction):**
- `src/pore_extraction.cu` / `.cuh`, `src/sdf_reader.cpp` / `.h` - SDF VTI reading + pore/topology extraction
- `src/bindings.cpp` - the `pnm_backend` pybind module (`SDFReader`, `extract_pores`, `segment_volume`, ...)

## Python API Usage (`sdflow`)

```python
import sdflow
s = sdflow.Solver(nx, ny, nz)
s.set_rho(1.0); s.set_mu(0.01); s.set_dt(60.0)   # physical units; fix before geometry
s.set_body_force(1e-2, 0, 0)                       # force per unit volume
s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")  # SDF [x,y,z], <0 inside
for _ in range(n_steps):
    s.step()
u = s.get_u()   # 3-D numpy array [x,y,z];  p = s.get_p() is the physical pressure
```
See the "Pressure solver options" table below and `scripts/*_sdflow.py` for the full API.

## Conventions

- **SDF sign**: Negative inside solid, positive in fluid
- **CUDA kernels**: Named with `_kernel` suffix
- **Staggered grid**: u at (i+1/2,j,k), v at (i,j+1/2,k), w at (i,j,k+1/2), p at cell centers

## MPI / sdflow (the CFD solver, transport-core integration)

The **`sdflow`** module is the incompressible Navier–Stokes solver, built on the shared
`transport-core` library (sibling repo `../transport-core`). It is **MPI-optional** — single-GPU by
default, multi-rank with `-DCFD_BUILD_MPI=ON`. It was validated bit-identical to the retired `CFDSolver`
reference (tag `pnm_backend-reference`) and against external analytics. Full status:
[`doc/mpi_parallelization_status.md`](doc/mpi_parallelization_status.md).

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
- Validated against Zick & Homsy SC-sphere drag. Design + benchmarks:
  [`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md).

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
conservation) **and against Zick & Homsy sphere-array drag** — **72/72 ctests, real multi-rank np=1,2,4**.

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
"Pressure solver options" table above) are all done and validated against Zick & Homsy (and, before its
retirement, bit-identical to the `CFDSolver` reference). The old `CFDSolver` reference is **retired** (tag
`pnm_backend-reference` is the restore point). **Remaining open items:** the large-np scaling work — an
**agglomerated coarse solve** and the communication-light **Chebyshev** accelerator's at-scale benchmark,
both needing real multi-GPU hardware (designs in [`doc/sdflow_multigrid_plan.md`](doc/sdflow_multigrid_plan.md)).
