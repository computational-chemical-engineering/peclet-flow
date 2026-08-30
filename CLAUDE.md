# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Performance-portable incompressible Navier-Stokes CFD solver for porous media simulations. Uses a staggered MAC grid with the Immersed Boundary Method (IBM) over complex geometries defined by Signed Distance Functions (SDF), with cut-cell pressure projection. Built on **Kokkos** — the same source runs on **CUDA, HIP (AMD/LUMI), and OpenMP** backends, selected at build time by the install prefix.

**`flow` is THE solver** (`src/flow_bindings.cpp` → `peclet::flow::IbmSolver` in `src/flow_ibm.hpp`):
cut-cell IBM physics on a staggered MAC grid, with a grid-independent geometric **multigrid** pressure
solve, and a multi-rank MPI path (core grid halo). It solves the equations in **physical units**
(density `rho`, dynamic viscosity `mu`, physical pressure `p`). See the "MPI / flow" section below.

The CUDA implementation was **retired** (Kokkos became canonical, 2026-06): `flow` was validated
bit-identical to the CUDA solver (machine-precision, and against the Zick & Homsy sphere-array Stokes drag)
before the CUDA sources were deleted. Restore point: the git tag `pre-cuda-retirement`. The cut-cell IBM
primitives live in `src/cut_cell_ibm.hpp`; the operator headers are `src/mac_*.hpp` + `src/flow_ibm.hpp`.

**Pore-network extraction moved out** (2026-07): the former `peclet.flow.pnm` module is now its own
suite project, `../pnm` (`peclet.pnm`, repo `peclet-pnm`) — the "pnm_from_sdf" namesake feature,
unrelated to the CFD solve. This repo builds only the solver.

## Build Commands

Kokkos is found via `find_package` against the bootstrapped install prefix
(`../extern/install/<backend>`, built once by `../tools/bootstrap_deps.sh` — a **hard build dependency**;
backend = `nvidia-cuda` / `host-openmp` / `lumi-hip`); **nanobind** is provisioned by the shared
`SuiteNanobind` helper (found through the active Python interpreter, no cmakedir prefix needed). With
`nvcc`, put it on `PATH` (`export PATH=/usr/local/cuda-13.2/bin:$PATH`).

```bash
source ../.venv/bin/activate   # THE suite venv (suite/CLAUDE.md "One venv")
# Canonical: build + install the flow solver via scikit-build-core.
CMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" pip install .

# Or a dev cmake build (single-rank Python modules):
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j
# Output: build/peclet.flow.*.so (the CFD solver) + build/pnm.*.so (pore extraction)

# OpenMP backend: same source, just swap the prefix (extern/install/host-openmp).
```

**Requirements:** Kokkos 5.x (C++20), CMake 3.24+, Python 3.10+, nanobind + scikit-build-core;
`../core` (header-only) + MPI for the multi-rank test suite. The Kokkos/ArborX install prefix is
produced by `../tools/bootstrap_deps.sh`.

## Running Tests and Verification

Drive `flow` verification from Python:
```bash
source ../.venv/bin/activate   # THE suite venv (suite/CLAUDE.md "One venv")
export PYTHONPATH=$PWD/build
python scripts/verify_periodic_spheres_sdflow.py   # cut-cell Stokes through a sphere packing
python scripts/verify_poiseuille_flow.py         # analytic parabola
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
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure   # 19 tests
# Multi-rank (MPI) tests, np=1,2,4:
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure       # 42 tests (14 x np)
```

Single-GPU **accuracy + efficiency regression suite** (grid-convergence + recorded solver-iteration
counts, checked against a saved baseline so regressions are caught — Z&H sphere, random-sphere bed,
hollow-ring bed): `PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py` (`--update` to
re-record the baseline). See [`tests/regression/README.md`](tests/regression/README.md).


## Architecture

### Memory Layout

- Linear indexing: `I = x + y*nx + z*nx*ny` (x is fastest)
- Python arrays: Fortran order `order='F'` with shape `(nx, ny, nz)`
- Periodic boundaries with wrapping: `(x % res.x + res.x) % res.x`

### Numerical Method (`flow`)

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

All Kokkos, header-only (`namespace flow`), C++20.

**`flow` (the CFD solver):**
- `src/flow_ibm.hpp` - `peclet::flow::IbmSolver`: the solver (diffusion, projection, three pressure drivers, Picard, MPI)
- `src/mac_cutcell_mg.hpp` - `CutcellMG`: geometric pressure MG (V-cycle / PCG / Chebyshev), MPI-folded
- `src/mac_velocity_mg.hpp` - `VelocityMG`: velocity multigrid (staircase / upwind / domain-BC), MPI-folded
- `src/mac_ibm.hpp`, `src/mac_cutcell.hpp`, `src/mac_pressure.hpp`, `src/mac_bc.hpp`, `src/mac_reductions.hpp` - IBM stencil, cut-cell openness, projection, domain BCs, reductions
- `src/cut_cell_ibm.hpp` - the Robust-Scaled cut-cell IBM overlay (`poly_*`, K/M/X/Nbc/R, D_rescale)
- `src/staggered_advection.hpp` - `sadv::advect`: staggered Koren TVD advection (+ implicit-FOU operator)
- `src/flow_bindings.cpp` - the `flow` nanobind module: `peclet.flow.Solver` (staggered MAC, default) and
  `peclet.flow.SolverColocated` (collocated/cell-centered velocities via the `GridLayout` policy + ABC
  approximate projection — identical Python API; see [`doc/flow_colocated_plan.md`](doc/flow_colocated_plan.md))
- `src/gauge_exact_gradient.hpp` - `gpCenterGrad`: the directional, gauge-exact cell-centre pressure
  gradient. This is what makes the COLLOCATED projection second order, and it is the default.

### Collocated schemes (`SolverColocated`) — REWRITTEN 2026-08-23 after the attractor campaign

**Read first if touching this path:** `doc/collocated_invisible_subspace.md` (the mechanism note),
`doc/collocated_paper_plan.md` (results tracker, rows 1–47), `doc/fluid_only_constraint_plan.md`
(the production plan). The 2026-08-18 "accuracy ceiling" narrative below the old table was
**superseded**: the "~0.3 % ceiling" conflated (i) a rotational-update instability
(Guy–Fogelson type) and (ii) an attractor FAMILY of steady states caused by the gradient and the
aperture constraint coupling different pressure supports (solid-centered φ are constraint DOFs
the gauge-exact gradient never reads). Under the clean protocol each scheme's true record is:

**DEFAULT since 2026-08-25: AUTO = `"ghost"`** (the fluid-only scheme) where the configuration
supports it, with a gauge-exact fallback + stderr notice on porous / variable-ρ / domain-BC /
Chebyshev (ghost v1 limits). Any explicit `set_collocated_scheme` / `set_face_interp` /
`set_ghost_projection` call disables AUTO. Baselines and tests pin schemes explicitly.

| scheme (`set_collocated_scheme`) | clean gaps vs staggered, φ=0.60, R=8→24 | stability / uniqueness |
|---|---|---|
| `"gauge-exact"` (AUTO fallback; default 08-18..25) | −2.54 → −0.76 → −0.13 → ~+0.20 % | attractor family (m1 frozen ~1e-2, P drift); wall-blend `set_rotational_wall_weight` is a STOPGAP whose margin dies at (R≥16, dt≥600) — fixed dt≤60 protocols at high R |
| `"ghost"` (the AUTO DEFAULT) | −1.43 → −0.265 → +0.077 → +0.219 % | family-free (m1→1e-5), NO stabilizer needed dt=60…1e20, C2/protocol-independent; Z&H −0.018 % @N=128; φ=0.50 replicated |
| `"plain"` | first order, −6.4 % @R=8 | legacy |

Both converging schemes share a small **real** asymptote ~+0.2 % vs the staggered reference
(which itself moves non-monotonically at the 5e-4 level over these rungs — the R=32 references
pin this down). The adjoint-aperture ablations (modes 11–13) and the fluid-only filter/star
designs (mode 14a / `set_fluid_only_constraint`) are retained as mechanism instruments: they
prove uniqueness needs support-consistency, that support-consistent *values* are O(h)
(multiplier/average reads), and that stability requires the (gradient, constraint) pair to be
structurally matched — the ghost architecture (shared directional closures) is the only measured
scheme that is stable, unique, AND converging. The B+ deferred-correction/preconditioner-swap
ideas were spectrally gated: sym(A_ghost) is indefinite (dead); the star base equals the binary
surrogate's λmax (no quick iteration win) — see the plan doc.

`"ghost"` == `set_ghost_projection(True, 2, 2)`: fluid-only binary-openness constraint +
directional closures + the gauge-exact gradient. Costs: BiCGStab (nonsymmetric, ~2.3–2.7× the
pressure stage), ~1.6 KB/cell overlay (single-GPU cap ≈ 10 M cells / 16 GB, ≈ 60 M / 94 GB),
fragmentation guard. MPI: validated np=1,2,4 (`ghost_projection_mpi` ctest, collocated included);
the at-scale np≥16 weak-rung divergence is under active de-confounding (it ran the PURE (2,2)
mode; single-rank on the same 7823-sphere bed is stable at R=8 — see tracker row 46). The
`(matrix_order=1, rhs_order=2)` mixed mode remains **do-not-use** (march-unstable above ~2000
spheres, `doc/ghost_hardening_findings_A.md`).

The old (contaminated-protocol) record is kept for the paper's refutation catalogue in
`doc/collocated_accuracy_ceiling.md`; its asymptotes (+0.29/+0.39 gauge-exact, +0.14/+0.22
ghost) match the clean-protocol asymptotes — the coarse-rung values were the contaminated part.
Old integer API `set_face_interp`: modes 1/2/10 retired; 3/4 FV ablations; 11–14 campaign
instruments (opt-in, single-rank).

**Regression**: `tests/regression/sdflow_regression.py --solver colocated --scheme ghost`
(baseline `perf_baseline_colocated_ghost.json`).

## Python API Usage (`flow`)

```python
import peclet.flow
s = peclet.flow.Solver(nx, ny, nz)
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

## MPI / flow (the CFD solver, core integration)

The **`flow`** solver (`peclet::flow::IbmSolver`) is built on the shared `core` library (sibling repo
`../core`), whose **Kokkos** grid halo (`peclet::core::halo::GridHalo`) carries the
multi-rank ghost exchange. The single-rank Python module is built by the main `CMakeLists.txt`; the
multi-rank path is exercised by the `tests/kokkos_mpi` ctests (gated behind `PECLET_FLOW_MPI`, so the single-rank
module is byte-identical). It was validated bit-identical (machine precision) to the retired CUDA solver
and against external analytics.

Key pieces (all `src/*.hpp`, Kokkos, header-only, `namespace flow`):
- `peclet::core::halo::GridHalo` (core) — per-level ORB block ghost exchange for the
  `double` cell-fields on the extended local block. cfd's x-fastest layout matches `peclet::core::Field3D`.
- `staggered_advection.hpp` — `sadv::advect`: staggered Koren TVD advection, templated on a field accessor.
- `flow_ibm.hpp` — `peclet::flow::IbmSolver`: the solver. `step()` does per-component implicit diffusion
  (RB-GS or velocity-MG, halo exchange between sweeps) + cut-cell incremental-rotational projection, with
  `set_advection`/`set_implicit_advection`, `set_body_force`, `set_solid` (cut-cell IBM no-slip), domain
  BCs, and `initMpi(gnx,gny,gnz,comm)` for the multi-rank step.

**Distributed smoother communication** (see `../docs/COMMUNICATION_SCALING.md`): every RB-GS sweep
(momentum + each MG level) overlaps its halo exchange with the interior sweep (post / smooth
interior / finish / smooth boundary shell — bit-identical by construction), and
**communication-avoiding smoothing** (`PECLET_FLOW_CA`, default ON, `=0` kills it) exchanges a
2-deep ghost layer ONCE per red-black pair instead of 1-deep per colour, redundantly re-smoothing
the 1-deep ghost ring so the second colour needs no exchange — bit-identical, half the halo
events. CA engages on the periodic/IBM operator where every rank's block extent is ≥ 4: in the
momentum sweeps (the velocity block is g=2 already; stencil/mask/rhs ring exchanged per operator
rebuild) and on `CutcellMG`'s coarse levels (per-level runtime ghost width `Level::g`, width-2
topologies, operator ring assembled from exchanged openness; level 0 and domain-BC hierarchies
keep g=1 — byte-identical). Parity trap: mixed ghost widths need the g-independent red-black
origin (`CutcellMG::parityOg`) or the colours swap on g=2 levels.

### Pressure solver options (the `flow` module)

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
- `set_pressure_bottom("smoother" | "auto" | "agglomerated")` — coarse-level solve. A V-cycle is
  domain-independent only if its coarsest level is effectively solved, and the hierarchy cannot always
  get there (§1.1/1.2 of [`../docs/DECOMPOSITION_AND_MULTIGRID.md`](../docs/DECOMPOSITION_AND_MULTIGRID.md)).
  `"auto"` agglomerates the coarsest level into a global operator (keyed by global cell id, so it is
  decomposition-independent — np=6 vs np=1 to 4.5e-16) and solves it exactly whenever that grid
  exceeds `PECLET_FLOW_AGGLOM_EXTENT` (4) cells on any axis. Measured, 2048×64×64 channel: 4 levels
  13.5 → 4.0 iters/step, 6 levels 6.0 → 4.0 (91 → 69.5 ms) — better than full geometric depth (4.4,
  77.2 ms). The former IBM anomaly (+41 % on `random_spheres`) is RESOLVED (2026-08-13): the bottom
  null-space projection is now per-fluid-component (solid identity rows excluded), the fluid
  diagonals are resummed in double so `A·1 = 0` exactly despite float level storage, and the inner
  tolerance is 1e-8 — cut-cell beds run at parity, the long-box case wins 25 → 7 iters/step, and the
  ghost-projection path is verified (its MG hierarchy is the binary-openness surrogate — the
  nonsymmetric gp rows never reach the bottom). (`PECLET_FLOW_AGMG_DEBUG=1` prints the
  bottom-operator anatomy + inner-CG stats; see `../docs/DECOMPOSITION_AND_MULTIGRID.md` §2.7.)
  **`"auto"` is the DEFAULT since 2026-08-13** (suite sweep: staggered regression +0.00 %, colocated
  13–27 % FEWER pressure iterations at identical accuracy, domain-BC verifies unchanged, MPI ctests
  green); `"smoother"` remains available, and `PECLET_FLOW_AGGLOM_EXTENT=1000000` reproduces the
  legacy behaviour without a code change. Porous / variable-ρ rebuild the operator every step and so
  rebuild the bottom AMG every step — negligible for the intended few-cells-per-axis bottoms, but
  avoid `auto` + a badly-factored grid (huge bottom) on those paths.

**Multigrid depth vs the decomposition (multi-rank).** An axis coarsens only while it stays even
(`d % 2 == 0 && d / 2 >= 2`), **per axis independently** — so semi-coarsening is automatic (the long
axis keeps halving after the short ones stop) and an axis's usable depth is its number of factors of
two, not its size. **An odd dimension never coarsens at all**: measured on one GPU at 384×128×GNZ,
`GNZ=256` needs 5.0 pressure iterations/step and `GNZ=255` needs 16.2 (3.2× the step time). Under
MPI there is a second gate — a level coarsens an axis only if *every rank's block* is even on it, so
the achievable depth is set by the per-rank block, not the global grid.

Two ways to build a decomposition that survives that, selected by
`flow.set_decomposition_levels(L)` (or `PECLET_FLOW_DECOMP_LEVELS`), which **must be set before
`mpi_block()` and `Solver.init_mpi()` — both derive the same partition from it**:

| `L` | how the level-0 partition is built |
|---|---|
| `0` (default) | **aligned ORB** — split positions chosen on the fine grid, then snapped to a power of two (capped at 16 = 5 nested levels) |
| `>= 2` | **coarse-first** — decompose the grid coarsened `L-1` times, then `refined()` the partition upward; blocks are multiples of the coarsening factor by construction, so the hierarchy nests for the full depth |

Coarse-first also balances better, because snapping can round a balanced split into an unbalanced one
while on the coarse grid one cell *is* the quantum. Measured, 480×80×160 with 6 levels requested
(max block / min block): np=4 1.143→1.000, np=7 1.125→1.029, np=12 1.333→1.038, np=16 1.333→1.000,
**np=24 1.500→1.000**. Note none of these rank counts is a power of two and several are the *best*
cases — what matters is that the coarse grid divides among the ranks, not that N is 2^k.

Depth and balance genuinely trade off (each extra level doubles the quantum), so `decomposition()`
builds each candidate depth and **measures** its imbalance, taking the deepest that stays within
`PECLET_FLOW_DECOMP_MAX_IMBALANCE` (default 1.05) and otherwise falling back to the aligned ORB. The
search is a pure function of (ranks, grid, levels), so every rank reaches the same answer with no
communication.

Check any grid/rank-count combination **before** submitting a job — no GPU needed, runs oversubscribed:
```bash
PYTHONPATH=$PWD/build_mpi_omp python scripts/check_decomposition.py --grid 480,80,160 --levels 6 \
  --np 4,7,24 --mode 0,coarse          # halvings, block, imbalance, splits, levels ACHIEVED
```
Full background, measurements and open problems: [`../docs/DECOMPOSITION_AND_MULTIGRID.md`](../docs/DECOMPOSITION_AND_MULTIGRID.md).
- Validated against Zick & Homsy SC-sphere drag. Design + benchmarks:
  [`doc/flow_multigrid_plan.md`](doc/flow_multigrid_plan.md).

### Domain boundary conditions

Beyond periodic + IBM no-slip on immersed solids, flow has **native per-face domain BCs** (`mac_bc.hpp`):
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
(Re_S=200) on the Armaly/Biswas curve, `PECLET_FLOW_BFS_RE800=1` pushes to the Gartling Re=800 benchmark.

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
Homsy sphere-array drag**; the multi-rank step is bit-exact to the single-rank — **42 `tests/kokkos_mpi`
ctests, real multi-rank np=1,2,4, on CUDA + OpenMP**. The variable-density and variable-viscosity
layers are multi-rank + CUDA gated since 2026-08-30 (`vardensity_mpi`, `varmu_mpi`).

**Domain BCs are rank-aware (WO-F, 2026-08-30).** Every per-face domain-BC application is guarded by
an ownership test — `touchesGlobalFace(f)` in `IbmSolver`, `touchesGlobalFace(lv, f)` per level in
`CutcellMG` — so a rank imposes a face's BC **iff its own block touches that global face**; the halo
exchange (built periodic on all three axes) owns every interior ghost, and the BC overwrite wins on
the owning rank, exactly the fill-then-BC order the single-rank path uses. Single-rank the test is
identically true, so all of it is byte-identical there. Before the fix, every rank imposed the wall
on its OWN block faces, so a partition cutting a walled/inflow/outflow axis split the domain into
independent sub-domains — **invisible in the velocity** (each sub-domain is separately consistent)
and visible only in the pressure (measured: max|u| 4.5e-17 while max|P_dist − P_ref| = 4.0e+02).
Guarded sites: velocity BCs, the flux-openness construction, the implicit wall fold, the pressure /
φ ghosts, `bcCorrectOutflow`, backflow stabilization, and the pressure MG's per-level boundary
openness + outflow ghost. `fillPropGhosts` / `fillPorousEpsGhosts` used to key their override on
`if (!distributed_)` — wrong at every np including 1 — and now use the same per-face test.
Details + measured numbers: `doc/variable_density_projection.md` §4.

**Per-cell body forces get a ghost fill (WO-G, 2026-08-30).** `applyClosure` writes the inner cells
only and an external CFD-DEM writer folds its ghost deposit onto the owners without refilling the
ghost band, and nothing exchanged `force_x/y/z` — so `buildRhsVar`'s face interpolation
`0.5*(fb(i) + fb(i−s_c))` read the registration zero (or the deposit residue) on the **first inner
plane of every block**, halving the face body force there. `step()` now calls `fillCellForceGhosts()`
right after `updateProperties()` — the force fields go through the same rank-aware `fillPropGhosts`
the other cell properties use (halo/periodic base + Neumann copy on an owned domain-BC face), which
is required because the RHS face-interpolates the force with the *same* mean it uses for ρ and the
physical content of the pair is f_f/ρ_f. Reach: only `buildRhsVar` (variable density, or the
eps-conservative porous momentum) reads that ghost — `buildRhsForced` reads `fb(i)` alone, so the
constant-density Boussinesq path is unaffected. On a **periodic** axis the old behaviour was a net
body-force deficit of exactly 1/(2·N_axis) (the projection removes the non-uniform part and the
deficient mean survives); at a **wall** it was fully masked, because the halved face is the
wall-normal velocity plane the Dirichlet BC pins and whose flux openness is 0 — which is why the
hydrostatic acid test, Rayleigh–Taylor and de Vahl Davis are all bit-identical across the fix and
only the multi-rank/periodic cases moved. Gate: `tests/kokkos_mpi/test_bodyforce_ghost_mpi.cpp`.

Two related MPI restrictions remain, both now explicit rather than silent: the solver's **velocity
multigrid is single-rank** (`IbmSolver` never calls `VelocityMG::initMpi`; `set_velocity_multigrid`
is disabled with a stderr notice under MPI), and a **multi-rank inlet profile** must be handed to
each rank as its own block's slice — `set_domain_bc_profile` resamples onto the local face plane and
there is no scatter helper.

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

**Status:** `peclet::flow::IbmSolver`/`flow` is the full solver — the Robust-Scaled cut-cell IBM, a grid-independent
geometric **multigrid** pressure solve (rediscretized cut-cell coarse operator; three selectable outer
drivers), velocity multigrid, implicit-FOU + Picard, all domain BCs, and a bit-exact multi-rank step
(`CutcellMG` + `VelocityMG` MPI-folded). The CUDA implementation is **retired** (restore tag
`pre-cuda-retirement`). **Remaining open items:** the large-np scaling work — an **agglomerated coarse
solve** and the communication-light **Chebyshev** accelerator's at-scale benchmark, both needing real
multi-GPU hardware.
