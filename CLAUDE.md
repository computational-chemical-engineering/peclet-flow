# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Performance-portable incompressible Navier-Stokes CFD solver for porous media simulations. Uses a staggered MAC grid with the Immersed Boundary Method (IBM) over complex geometries defined by Signed Distance Functions (SDF), with cut-cell pressure projection. Built on **Kokkos** — the same source runs on **CUDA, HIP (AMD/LUMI), and OpenMP** backends, selected at build time by the install prefix.

**`sdflow` is THE solver** (`src/sdflow_bindings.cpp` → `sdflow::SdflowIbm` in `src/sdflow_ibm.hpp`):
cut-cell IBM physics on a staggered MAC grid, with a grid-independent geometric **multigrid** pressure
solve, and a multi-rank MPI path (core grid halo). It solves the equations in **physical units**
(density `rho`, dynamic viscosity `mu`, physical pressure `p`). See the "MPI / sdflow" section below.

The CUDA implementation was **retired** (Kokkos became canonical, 2026-06): `sdflow` was validated
bit-identical to the CUDA solver (machine-precision, and against the Zick & Homsy sphere-array Stokes drag)
before the CUDA sources were deleted. Restore point: the git tag `pre-cuda-retirement`. The cut-cell IBM
primitives live in `src/cut_cell_ibm.hpp`; the operator headers are `src/mac_*.hpp` + `src/sdflow_ibm.hpp`.

**`pnm` is the pore-network-extraction module** (`src/pnm_bindings.cpp` + `src/pore_extraction.hpp`
Kokkos compute + the pure-C++ `src/sdf_reader.cpp` VTI reader): `SDFReader`, `extract_pores`,
`segment_volume`, `extract_topology_gpu` — the repo's namesake "pnm_from_sdf" feature, unrelated to the CFD solve.

## Build Commands

Kokkos is found via `find_package` against the bootstrapped install prefix
(`../extern/install/<backend>`, built once by `../tools/bootstrap_deps.sh` — a **hard build dependency**;
backend = `nvidia-cuda` / `host-openmp` / `lumi-hip`); **nanobind** is provisioned by the shared
`SuiteNanobind` helper (found through the active Python interpreter, no cmakedir prefix needed). With
`nvcc`, put it on `PATH` (`export PATH=/usr/local/cuda-13.2/bin:$PATH`).

```bash
source .venv/bin/activate
# Canonical: build + install both modules (sdflow solver + pnm) via scikit-build-core.
CMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" pip install .

# Or a dev cmake build (single-rank Python modules):
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j
# Output: build/sdflow.*.so (the CFD solver) + build/pnm.*.so (pore extraction)

# OpenMP backend: same source, just swap the prefix (extern/install/host-openmp).
```

**Requirements:** Kokkos 5.x (C++20), CMake 3.24+, Python 3.10+, nanobind + scikit-build-core;
`../core` (header-only) + MPI for the multi-rank test suite. The Kokkos/ArborX install prefix is
produced by `../tools/bootstrap_deps.sh`.

## Running Tests and Verification

Drive `sdflow` verification from Python:
```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/build
python scripts/verify_periodic_spheres_sdflow.py   # cut-cell Stokes through a sphere packing
python scripts/verify_poiseuille_sdflow.py         # analytic parabola
python scripts/verify_lid_cavity_sdflow.py         # lid cavity vs Ghia
python scripts/verify_channel_sdflow.py            # developing channel
python scripts/verify_bfs_sdflow.py                # backward-facing step
python scripts/verify_chebyshev_sdflow.py          # Chebyshev pressure driver == MG-PCG
python scripts/validate_zick_homsy_sdflow.py       # external ground truth (Z&H drag)
```

C++ kernel + multi-rank test suites (own `find_package` projects; build against the same prefix):
```bash
# Single-rank Kokkos kernel unit tests:
cmake -S tests/kokkos -B build_kokkos -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure   # 14 tests
# Multi-rank (MPI) tests, np=1,2,4:
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure       # 18 tests (6 x np)
```

Single-GPU **accuracy + efficiency regression suite** (grid-convergence + recorded solver-iteration
counts, checked against a saved baseline so regressions are caught — Z&H sphere, random-sphere bed,
hollow-ring bed): `PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py` (`--update` to
re-record the baseline). See [`tests/regression/README.md`](tests/regression/README.md).

Pore-network extraction (the `pnm` module): `python scripts/test_extraction.py`,
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

All Kokkos, header-only (`namespace sdflow`), C++20.

**`sdflow` (the CFD solver):**
- `src/sdflow_ibm.hpp` - `sdflow::SdflowIbm`: the solver (diffusion, projection, three pressure drivers, Picard, MPI)
- `src/mac_cutcell_mg.hpp` - `CutcellMG`: geometric pressure MG (V-cycle / PCG / Chebyshev), MPI-folded
- `src/mac_velocity_mg.hpp` - `VelocityMG`: velocity multigrid (staircase / upwind / domain-BC), MPI-folded
- `src/mac_ibm.hpp`, `src/mac_cutcell.hpp`, `src/mac_pressure.hpp`, `src/mac_bc.hpp`, `src/mac_reductions.hpp` - IBM stencil, cut-cell openness, projection, domain BCs, reductions
- `src/cut_cell_ibm.hpp` - the Robust-Scaled cut-cell IBM overlay (`poly_*`, K/M/X/Nbc/R, D_rescale)
- `src/staggered_advection.hpp` - `sadv::advect`: staggered Koren TVD advection (+ implicit-FOU operator)
- `src/sdflow_bindings.cpp` - the `sdflow` nanobind module: `sdflow.Solver` (staggered MAC, default) and
  `sdflow.SolverColocated` (collocated/cell-centered velocities via the `GridLayout` policy + ABC
  approximate projection — identical Python API; see [`doc/sdflow_colocated_plan.md`](doc/sdflow_colocated_plan.md))

**`pnm` (pore-network extraction):**
- `src/pore_extraction.hpp` (`namespace pnm`, Kokkos compute), `src/sdf_reader.cpp` / `.h` (pure-C++ VTI reader)
- `src/pnm_bindings.cpp` - the `pnm` nanobind module (`SDFReader`, `extract_pores`, `segment_volume`, ...)

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
- **Kokkos kernels**: `parallel_for` / `parallel_reduce` over `Kokkos::View`s (`MDRangePolicy` for 3-D loops); device sources are `.hpp` compiled as C++ (the launch compiler routes through `nvcc`/`hipcc`), never `.cu`
- **Staggered grid**: u at (i+1/2,j,k), v at (i,j+1/2,k), w at (i,j,k+1/2), p at cell centers

## MPI / sdflow (the CFD solver, core integration)

The **`sdflow`** solver (`sdflow::SdflowIbm`) is built on the shared `core` library (sibling repo
`../core`), whose **Kokkos** grid halo (`tpx::halo::GridHalo`) carries the
multi-rank ghost exchange. The single-rank Python module is built by the main `CMakeLists.txt`; the
multi-rank path is exercised by the `tests/kokkos_mpi` ctests (gated behind `CFD_MPI`, so the single-rank
module is byte-identical). It was validated bit-identical (machine precision) to the retired CUDA solver
and against external analytics.

Key pieces (all `src/*.hpp`, Kokkos, header-only, `namespace sdflow`):
- `tpx::halo::GridHalo` (core) — per-level ORB block ghost exchange for the
  `double` cell-fields on the extended local block. cfd's x-fastest layout matches `tpx::Field3D`.
- `staggered_advection.hpp` — `sadv::advect`: staggered Koren TVD advection, templated on a field accessor.
- `sdflow_ibm.hpp` — `sdflow::SdflowIbm`: the solver. `step()` does per-component implicit diffusion
  (RB-GS or velocity-MG, halo exchange between sweeps) + cut-cell incremental-rotational projection, with
  `set_advection`/`set_implicit_advection`, `set_body_force`, `set_solid` (cut-cell IBM no-slip), domain
  BCs, and `initMpi(gnx,gny,gnz,comm)` for the multi-rank step.

### Pressure solver options (the `sdflow` module)

The cut-cell pressure Poisson is solved by a geometric **multigrid** (`mac_cutcell_mg.hpp`, `CutcellMG`)
whose smoother is **Red-Black Gauss-Seidel** and whose coarse operator is the **rediscretized** cut-cell
operator. Three outer drivers wrap that V-cycle — **select one per solver**:

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

Beyond periodic + IBM no-slip on immersed solids, sdflow has **native per-face domain BCs** (`mac_bc.hpp`):
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
is byte-identical. A quasi-2D 256×64×4 channel now builds up to 8 levels (was 2): raw V-cycle flux
divergence at a fixed 8 cycles drops `1.7e-4`→`8.6e-13`. The BC verify scripts request `levels=8`
(auto-capped). *Follow-ups:* convective outflow for unsteady wakes, multi-rank inlet-profile scatter
(validated single-rank).

Validated against analytics (Taylor–Green ~2e-15, Poiseuille, momentum conservation) **and against Zick &
Homsy sphere-array drag**; the multi-rank step is bit-exact to the single-rank — **18 `tests/kokkos_mpi`
ctests, real multi-rank np=1,2,4, on CUDA + OpenMP**.

Build/test the multi-rank ctests:
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S tests/kokkos_mpi -B build_kmpi \
  -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_kmpi -j
ctest --test-dir build_kmpi --output-on-failure
```
**Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`** — FindMPI may pick ParaView's bundled `mpiexec` on
`PATH`, which launches the OpenMPI-linked test binaries as singletons (so `*_np4` silently runs 4×np=1).

**Status:** `sdflow::SdflowIbm`/`sdflow` is the full solver — the Robust-Scaled cut-cell IBM, a grid-independent
geometric **multigrid** pressure solve (rediscretized cut-cell coarse operator; three selectable outer
drivers), velocity multigrid, implicit-FOU + Picard, all domain BCs, and a bit-exact multi-rank step
(`CutcellMG` + `VelocityMG` MPI-folded). The CUDA implementation is **retired** (restore tag
`pre-cuda-retirement`). **Remaining open items:** the large-np scaling work — an **agglomerated coarse
solve** and the communication-light **Chebyshev** accelerator's at-scale benchmark, both needing real
multi-GPU hardware.
