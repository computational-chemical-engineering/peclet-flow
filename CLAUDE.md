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
cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure   # 22 tests
# Multi-rank (MPI) tests, np=1,2,4:
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure       # 48 tests (16 x np)
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
operator. Four outer drivers wrap that V-cycle — **select one per solver**:

| driver | select with | use |
|---|---|---|
| **Standalone V-cycle** | default (neither below set) | multi-rank default. `set_pressure_multigrid(True, levels=1)` ⇒ pure RB-GS (no coarse grid) |
| **MG-PCG** | `set_pressure_pcg(True, max_iter, rtol)` | **single-GPU default** (auto-enabled on 1 rank); ~1.2× faster than the V-cycle to a fixed tolerance. Healthy on 3-D wall-bounded grids since the WO-H symmetry repair (below); still capped by a **high density/void CONTRAST** — see the last bullet |
| **Flexible MG-CG** | `set_pressure_fcg(True, max_iter, rtol)` | the same solve with the Polak–Ribière β, tolerant of a preconditioner that is not symmetric w.r.t. the fine operator. +1 vector, +1 global dot/iteration, ≤2 % projection-time overhead. The most forgiving CG driver — the one to try when PCG caps |
| **Chebyshev** | `set_pressure_chebyshev(True, max_iter, rtol)` | communication-light (no per-iteration global dot-products) — for large multi-GPU where PCG's reductions are latency-bound. ≈ PCG iteration count on periodic/IBM; bounds estimated once on step 1 (per step under varRho/porous, which costs 30 extra V-cycles) |

- **The three Krylov drivers are mutually exclusive in both directions, last set wins** (repaired by
  WO-H, 2026-08-30): `set_pressure_pcg(True, …)` clears both competing selections, `set_pressure_fcg(True, …)`
  clears the Chebyshev selection, `set_pressure_chebyshev(True, …)` clears the FCG selection — so each
  works after `set_density_mode` / `set_porous`. `set_pressure_fcg(False)` and
  `set_pressure_chebyshev(False)` fall back to MG-PCG; `set_pressure_pcg(False)` **raises**, because
  MG-PCG is the terminal fallback of the dispatch and cannot be deselected on its own — name the
  driver you want instead. With none set, the solve is `n_pois` standalone V-cycles (or auto-PCG on 1
  rank). *Before WO-H `set_pressure_pcg`'s `on` flag was silently discarded*, so after
  `set_density_mode("variable")` / `set_porous` there was no way to select PCG at all and the working
  spelling was `set_pressure_chebyshev(False, …)`; every "PCG under varRho/porous" measurement made
  that way actually measured Chebyshev (the tell: it capped at 120, `chebMaxit_`'s default).
- **The domain-BC V-cycle used to be an asymmetric preconditioner, and MG-PCG stalled on every 3-D
  wall-bounded grid because of it** — 200/200 iterations with max|div(open·u)| ≈ 1e-5 once the third
  axis reached 8 cells. **FIXED 2026-08-30 (WO-H)**: the per-level ghost fill is periodic on all three
  axes, so a walled face's coarse ghost held the value from the *opposite* side of the domain; every
  operator consumer is immune (the wall face openness is 0, so the smoother/residual/matvec multiply
  that ghost by a zero coefficient) but the **trilinear prolongation is not** — it reads the coarse
  ghost with weight ¼ whatever the openness. That teleport is a long-range coupling present in the
  prolongation and absent from the restriction. `CutcellMG::applyNeumannGhost` now imposes the
  zero-gradient ghost on owned wall/inflow faces before every prolongation, the pressure-side
  counterpart of `VelocityMG::fillProlongBcGhosts` (the velocity MG always had both halves; the
  pressure MG only ever got the Dirichlet one, `applyOutflowGhost`). Measured, 24×24×16 lid box at
  constant density: **PCG 200/200 → 6, FCG 22 → 6, Chebyshev 12 → 7**; the direct symmetry read-out
  `pr` (`PECLET_FLOW_MG_DEBUG=2`, zero iff M is symmetric w.r.t. the fine operator) drops from
  0.42–0.52 median wall-bounded to **0.008–0.086**, at or below the 0.062 of the *periodic* hierarchy.
  Periodic/IBM is byte-identical (`hasBC_` gates it) and the single-phase regression is +0.00 % with
  identical iteration counts. `PECLET_FLOW_MG_BCGHOST=0` restores the old ghost as a measurement
  ablation. Gate: `tests/kokkos/test_pressure_wallbounded.cpp` (nz = 16, all three drivers).
- **What remains: a high coefficient CONTRAST makes the V-cycle preconditioner indefinite.** Both CG
  drivers still cap on a wall-bounded *stratified* column at density ratio ≳ 10³ and PCG on the
  small-coefficient-against-an-inflow-face case, while **Chebyshev is healthy on all of them**. Cause,
  measured directly (dense `sym(M)` assembled from unit V-cycles on an 8³ box): a **negative LDL
  pivot** from ratio ~10³ walled and ~10⁴ even fully periodic — i.e. the arithmetic coarsening of the
  face coefficient (`coarsenOpenAvg`) produces a coarse operator inaccurate enough to make the
  correction indefinite, which no choice of the CG β survives. This is the coefficient-aware-coarsening
  item (VOF_PLAN S3), not a boundary-treatment one, and it is why **Chebyshev stays the varRho/porous
  default**. Measurements: WO-B/WO-C/WO-H in [`doc/vof_workorders.md`](doc/vof_workorders.md);
  reproduce with `tests/study/vardensity_solver_probe.py --drivers pcg,fcg`.
  **WO-M (2026-08-31) re-ran that dense probe on a `-DPECLET_FLOW_MREAL_DOUBLE` hierarchy and the
  negative pivot SURVIVES, unchanged to 3-4 significant figures** — so the indefiniteness is the
  coarsening, not the float storage, and S3 stands. (Two corrections to the method: `sym(M)` is
  singular by construction, so an unpivoted LDL's 1e-12 pivot is sign noise and the *mean-free
  restricted spectrum* is the reliable read-out. Instrument: `tests/study/mg_precond/`.)
- **Operator STORAGE precision (`MReal`) is a separate, measured axis.** `mac_cutcell_mg.hpp`'s
  `MReal` types the pressure hierarchy AND (via `IbmSolver::FV`) the momentum stencil;
  `-DPECLET_FLOW_MREAL_DOUBLE` switches both to double. Float rounding breaks the singular row-sum
  identity `A·1 = 0` at ~eps_f32 per row, which is amplified to ~1e-4 relative to the *small*
  couplings under three decades of aperture/density contrast. Measured (WO-M,
  `tests/study/precision_ab.py`, RCP bed φ=0.63): **the residual floors at 5e-9…6e-8 and then
  REBOUNDS by 1e4…5e5**, resolution- AND depth-independent, so MG-PCG burns its cap on any
  high-contrast bed above ~96³ — a run that is INVALID, not degraded. In double the same solves
  converge in 21-55 iterations to 5e-15. It also floors the V2b uniform-velocity identity at 1.3e-7
  (double: 9e-16) and the porous CFD-DEM drag balance at 4.8e-8 (double: 2.8e-16). It is
  **irrelevant** to every approximation-limited metric: Z&H drag, the converged permeability and the
  hydrostatic acid test agree between the two builds to 6 digits or better. **Cost of full fp64,
  measured on the same GPU at identical iteration counts: +12 % step time, +120 B/cell (+10 % GPU
  memory).** The recommended production fix is the **double-diagonal** (faces float, diagonal stored
  and resummed in double so `A·1 = 0` holds exactly — the fix already proven at the agglomerated
  bottom, generalised); `PECLET_FLOW_MG_DIAGRESUM=1` **on a double build** emulates exactly its
  arithmetic and matches full double at every grid measured. Full findings + policy:
  `doc/vof_workorders_v34.md` (WO-M). **Independently reproduced in the field (2026-09-01,
  peclet-examples `benchmarks/foxberry-scaling`): a 5000-sphere φ=0.45 bed caps MG-PCG at 384³
  (R = 10.7 cells) while the SAME bed converges in 11 iterations at 128³ and 20 at 256³, at np=1
  as well as np≥24 — and the giveaway is that `<u>` and `max|div|` are identical to seven digits
  between the capped rtol=1e-8 run and a converged rtol=1e-6 one (36.5 iters). Not dt (a 100×
  sweep moves 9.5 → 11), not the bottom mode, not advection, not MPI. The fp64 build is 2×
  FASTER in wall clock there (71 iters vs a 200 cap), so on high-contrast beds fp64 is the
  performance choice, not a correctness tax.**
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

**That second gate is an implementation limit and it is the top open item at scale.** Coarse levels
are required to be the fine decomposition `coarsened()` in place (so restrict/prolong stay purely
local), which is why the hierarchy simply stops when a block hits an odd extent. Measured cost
(peclet-examples `benchmarks/foxberry-scaling`, 384³, 24→1536 ranks): pressure iterations rise
16.6 → 38.7 and strong-scaling efficiency falls to 67 %, while time *per iteration* scales
**super-linearly** (156 %) — i.e. the whole loss is the iteration count, not communication. The fix
is to let a coarse level live on its own coarser partition and redistribute inside the transfer
(PETSc `PCTELESCOPE`, MueLu `RepartitionFactory`, hypre's redundant coarse solve); the endpoint of
this already exists as `set_pressure_bottom("auto")`, only the intermediate steps are missing. See
[`../docs/DECOMPOSITION_AND_MULTIGRID.md`](../docs/DECOMPOSITION_AND_MULTIGRID.md) §2.8 and open
problem 1.

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

### Geometric VoF — two-phase flow (rungs V2a + V2b; `src/vof/`)

Campaign plan: [`../docs/VOF_PLAN.md`](../docs/VOF_PLAN.md); work orders + findings:
[`doc/vof_workorders.md`](doc/vof_workorders.md) (phase 0),
[`doc/vof_workorders_v2.md`](doc/vof_workorders_v2.md) (V2),
[`doc/vof_workorders_v34.md`](doc/vof_workorders_v34.md) (V3/V4) and
[`doc/vof_workorders_v5.md`](doc/vof_workorders_v5.md) (V5a cut cells, and the open V-BC/V5b/V8
work orders).

- `src/vof/plic.hpp` (V0) — SZ2000/Lehmann–Gekle plane↔volume, MYC normals, slab flux volumes.
  Container-free `KOKKOS_INLINE_FUNCTION`s only (no `View`, no indexing), so the V4 promotion to
  `peclet::core::vof` is a file move.
- `src/vof/advect_wy.hpp` (V1) — `WyAdvector`: Weymouth–Yue split geometric advection on its **own
  g=3 block** with its own ghost callback. CFL cap 0.25 (Weymouth's proven **3D** bound
  1/(2(N−1)); the familiar 0.5 is the 2D value). The dilation flag is frozen once per step —
  recompute it per sweep and exact conservation silently dies (measured 2.3e-15 → 1.5e-2).
- `src/vof/colour_field.hpp` (V2a) — the `G=2` ↔ `g=3` bridge and the colour ghost policy.
- `src/vof/wetting.hpp` (V5b) — the theta-consistent band fill: the fluid-only Youngs normal,
  the rotation to the prescribed angle and the plane->fraction rule, container-free like `plic.hpp`.
- `src/vof/cutcell.hpp` (V5a) — the cut-cell rules of the colour transport, container-free like
  `plic.hpp`: the effective fluid volume `eps_eff = max(eps, 1/64)`, Weymouth's admissible flux
  interval generalized to a cut donor, the effective Courant number, the [0,1] clip and the
  solid-band fill state machine.
- `src/vof/momentum_advect.hpp` (V2b) — `MomentumConsistentAdvector`: `rho^c u_c` on the
  half-shifted MAC control volumes, driven by the SAME PLIC planes, sweep order and frozen state as
  the colour advection of that step. Opt-in (`enable_vof_momentum(rho_gas, rho_liquid)`), and what
  makes the scheme usable above density ratio ~100.
- `src/vof/curvature.hpp` + `curvature_field.hpp` (V3) — interface curvature as the Popinet (2009)
  height-function cascade with a **PLIC-volumetric paraboloid fallback** (Jibben 2019 / Han, Evrard
  & Desjardins IJMF 2024, 5³ stencil, Wendland width 2.5). `curvature.hpp` is container-free like
  `plic.hpp`; the driver is next door. **No new halo** — the whole cascade reaches exactly ±3, so
  the colour field's g = 3 is the design and no reduction appears anywhere in it (hence *bitwise*
  MPI, not the reduction-order floor).

**Curvature (V3).** `compute_vof_curvature()` fills two registered `G=2` fields from the current
colour field and returns this rank's branch census: **`"kappa"`** = κ = 2H in **1/h (cell units)**,
positive for a convex blob of liquid (sphere `+2/R`, cylinder `+1/R`, plane `0`), and
**`"kappa_branch"`** = which tier produced it. *Always read the branch field alongside κ*: κ is 0
both where there is no interface (branch 0, correct) and where no estimate could be made (branch 6,
which must never happen). Curvature comes from **column sums of C and PLIC volumes, never from
differentiating the MYC normal** — V0 measured that normal's error not converging (order 0.83) while
its reconstruction error does (1.98), which is the whole reason the plan rejected ∇C geometry.
Measured (`tests/kokkos/test_vof_curvature.cpp`): exact-fraction sphere 16³→32³→64³ **order 2.26
(L1) / 1.86 (max)**; plane 1.5e-14; cylinder 2.8e-3 against 1/R. Two literature facts the API
docstrings repeat because they are physics and not defects: the **fallback always fires** below ~5
cells/diameter (measured 100 % at D/Δ = 2.8–4.4, ~19 % at D/Δ = 38–48 — higher than Han's 2-D 0.9 %
because a 3×3 patch in 3D needs 2.5 cells of column reach on the octant diagonal, exactly the
capacity of a 7-column), and with **advection-realistic fractions the curvature error stops
converging**, here between CΔ ≈ 0.16 and 0.08. Cascade tier 2b (the mixed height-position fit) is
implemented and ships **OFF** — it destroys the max-error convergence (order 0.00 vs 1.86) because
its data set is the slope-selected subset of columns that closed; `doc/vof_workorders_v34.md` WO-O
has the mechanism and the width sweep.

**Balanced-force surface tension (V4).** `set_surface_tension(sigma)` turns on the continuum
surface force at the staggered face, formed with **the projection's own difference operator**:

```
F_c(i) = sigma * kappa_f(i) * ( C(i) - C(i - s_c) ) / h
```

added to the momentum RHS at the same place, in the same units and with the same cut-cell rescale
as the incremental scheme's `-(P(i) - P(i - s_c))`. `kappa_f` is the mean of the two cells'
curvatures where both carry one, the single available one where only one does. **That pairing is the
whole rung.** With a constant κ the force is *exactly* the discrete gradient of `σκC`, so it lies in
the range of the operator the projection inverts, the projection annihilates it, and a stationary
droplet stays at machine zero — measured **max|u| = 3.6e-17 / 1.9e-17 / 2.4e-17 at 16³/32³/48³** and
**9.4e-17 … 4.3e-18 over μ = 1e-3 … 1** (Francois et al. 2006; Popinet 2009). The ablation
`set_csf_mode(1)` — a *cell-centred* `σκ∇C` face-interpolated exactly as the per-cell body-force
machinery carries `ρg` — is the same physics with one wrong operator pairing and reads **5.8e-2,
i.e. Ca = 5.8e-3 and 3.0e+15× the balanced-force value**: that is the literature's "naive CSF gives
~1e-2", reproduced as a switch. Equilibrium `P = σκC + const` holds to **2.6e-15** over the whole
field and the Young–Laplace jump to **2.2e-16**.

With the *computed* curvature the residual currents are the curvature error and nothing else:
measured **Ca = 2.5e-4 / 5.9e-5 / 2.6e-5 at D/Δ = 8 / 16 / 24**, converging. Note what that means
for the pore scale: `Ca ≈ δκ·h` in cell units, so the ≲1e-7 budget is a *curvature* requirement, and
WO-O measured that curvature error stops converging with advection-realistic fractions — the force
discretization is exact, the estimator is the ceiling.

Two things the rung had to pay for, both measured:
- **A wisp guard on the curvature's interfacial predicate is not optional once κ feeds a force.**
  Weymouth–Yue leaves round-off colour residue (down to −3e-35) in every cell its sweeps touch;
  those cells satisfy `0 < C < 1`, so the V3 cascade builds a zero-area PLIC polygon for them and
  returns **|κ| up to 1.2e+08** where the physical value is 0.125. A face between one of them and a
  real interfacial cell then carries a force eight orders too large. `set_surface_tension` therefore
  sets `VofCurvature::interfaceEps = 1e-8` (`set_vof_interface_eps`); the V3 default stays 0, so
  `compute_vof_curvature()` without surface tension is unchanged. Ablation at `eps = 0`: a 32³ static
  droplet goes 4.5e-4 → 2.7e-1 in 20 steps and a 96³ one trips the WY CFL cap.
- **At variable density the equilibrium is approached, not hit.** The semi-implicit momentum
  operator `A = ρ_f/dt − μ∇²` commutes with the discrete gradient only when `ρ_f` is constant (and
  away from a wall), so `A⁻¹∇Φ` is a pure ρ-weighted gradient — which the projection removes
  exactly — only there. With ρ varying the first-step residue scales as **μ·dt²** (measured at ratio
  100: 3.3e-9 at μ=0 against 2.4e-4 at μ=0.1; 1.9e-11 at dt/dt_σ = 1e-4 against 2.4e-4 at 0.5) and
  then **decays**: 1.1e-4 → 5.0e-7 over 300 steps at ratio 10. It is a property of the projection
  splitting, not of the CSF, and it is NOT the float-storage floor — the numbers reproduce to five
  figures in a `-DPECLET_FLOW_MREAL_DOUBLE` build. At μ = 0 that build reads **6.8e-17 … 8.9e-18 at
  every ratio to 1000** while the float default floors at ~1e-10 (WO-M's `A·1 = 0` defect, two
  orders below the μ mechanism at any realistic viscosity).

**The capillary time step** `Δt < sqrt((ρ₁+ρ₂)Δx³/(4πσ))` (Brackbill 1992; Denner & van Wachem 2015
verified the prefactor IS the stability boundary, the h^{3/2} scaling, and that it is the *sum* of
the densities) is exposed as `capillary_dt()` and **enforced by `step()`** alongside the
Weymouth–Yue CFL cap; `set_capillary_cfl(f)` is the safety factor (default 1.0, huge to disable).
`vof_step_limits()` reports both limits and which binds. **At pore scale the capillary limit binds
everywhere**: swept over pore diameters 50/200 µm, 16/32/64 cells per diameter and Ca = 1e-6…1e-2
for water/air, `dt_σ` is the binding limit in 18 of 18 combinations, by factors 6 to 5.9e4 — and it
gets *more* binding under refinement, since `dt_σ ~ h^{3/2}` against `dt_CFL ~ h`. The cost is real
(3.8e6 steps to traverse one 50 µm pore at Ca = 1e-6), which is the measurement that decides whether
implicit surface tension is ever worth revisiting.

**Python:** `enable_vof()`, `set_vof(C)` / `get_vof()`, `vof_max_courant()` / `vof_last_courant()`,
`set_vof_cfl_limit()`, `vof_diagnostics()`, `set_rho_face_harmonic()`; V2b adds
`enable_vof_momentum(rho_gas, rho_liquid)`, `vof_advected_velocity(c)`,
`vof_momentum_diagnostics()`, `set_vof_rho_floor()`, `set_vof_momentum_muscl()`,
`set_vof_momentum_cell_flag()`, `set_vof_flux_clamp()`; V3 adds `compute_vof_curvature()`,
`vof_curvature()`, `vof_curvature_branch()`, `set_vof_curvature_weight_width()`,
`set_vof_curvature_mixed_height_fit()`; V4 adds
`set_surface_tension()` / `surface_tension()`, `capillary_dt()`, `set_capillary_cfl()`,
`vof_step_limits()`, `csf_diagnostics()`, `set_vof_interface_eps()` / `vof_interface_eps()`,
`set_vof_kappa_frozen()`, `set_vof_kappa_constant()`, `set_csf_mode()`. `"C"` is an ordinary
registered `G=2` cell field, so ρ(C)/μ(C) go through the existing `LinearMix` closures
(`set_property_model("rho","linear","C",[rho_g, rho_l-rho_g])`, which enables the varRho path) and
`get_field`/`set_field`/`field_view`/`redistribute` work on it unchanged. The **g=3 working block**
is the advector's, with its own `GridHaloTopology` under MPI — flow's global `G=2` is untouched.

**Momentum consistency (V2b, opt-in).** `enable_vof_momentum(rho_gas, rho_liquid)` advects
`rho^c u_c` on the half-shifted momentum control volumes with the same geometric fluxes as `C`,
then recovers `u = (rho^c u)/rho^c`. Without it, mass and momentum are advected by different fluxes
and a mixed cell carries a spurious interfacial momentum source of order Δρ — the literature is
unambiguous that this breaks down around ratio 1000. Turning it on **moves the VoF advection from
the end of `step()` to its head** (the momentum advection must precede the predictor that consumes
it), so the advecting field is `u^n`, the previous step's projected output, and `u` at step 0 must
be discretely divergence-free. Requires the variable-density path, staggered layout, explicit
advection, no immersed solid, no porous continuity. The decisive gate — an arbitrary sharp `C` with
a uniform velocity — is **bitwise** exact at ratios 1e1..1e4, single-rank and at np 1/2/4.
Three things this construction paid for, all in `doc/vof_workorders_v2.md` (WO-K findings):
- **the flux must be clamped into Weymouth's admissible interval on the shifted volume's own
  colour.** The geometric flux is bounded by what the CURRENT cell planes see in the donor, not by
  the ADVECTED `C^c`; the gap is O(a²) and at ratio 1e4 a 2.6e-2 undershoot drives `rho^c` to −255.
  `set_vof_flux_clamp(False)` is the ablation: divergence at step 2.
- **the dilation coefficient `rho^ u` must be frozen across the three sweeps**, exactly as WY freeze
  `H(C^n−½)`. Measured: per-step momentum drift 1.4e-7 with the running velocity, 2.2e-13 frozen.
- **a MUSCL slope in the momentum flux is a density-ratio amplifier** on any control volume a sweep
  empties (gain `Δρ·F/rho^c`); plain donor-cell upwind is the default and
  `set_vof_momentum_muscl(True)` the opt-in, measured 2.2e-10 vs 6.7e-16 at ratio 1e4.

**Cut cells — VoF through an immersed solid (rung V5a, WO-Q).** `advectVof`'s `hasSolid_` throw is
LIFTED: the colour field is transported through an SDF solid with **openness-weighted geometric
fluxes**. `C` is the liquid fraction of the **fluid** volume of a cell (VOF_PLAN §3 rule 2), the
transported quantity is `eps_i C_i`, every flux is `F_f = o_f · wyFaceFlux(a_f, …)` and the dilation
term uses the **same** `o_f a_f` — so the flux sum telescopes and the dilation sum is `H(C−½)` times
the projection's own openness-weighted divergence, i.e. `Σ_i eps_eff_i C_i` is conserved **exactly**
to the projection's residual. Measured (24³ periodic sphere array, 200 kinematic steps at interface
CFL 0.2): drift **7.7e-12** against a `max|div(open·u)|` floor of **3.0e-11**, colour in solid cells
**exactly 0**, `C ∈ [0,1]` in uncut fluid cells to the last bit, clipped volume **5.4e-19**. Needs
`set_solid(..., cutcell_pressure=True)` (the staircase operator has no face openness to weight
with) and it throws otherwise. New Python: `advect_vof(dt)` (kinematic advection with the current
face velocity, no NS step — it **throws** unless the field is discretely divergence-free to 1e-10,
because WY conservation is conditional on that), `set_vof_step_parity()`, `vof_has_geometry()`,
`vof_filled_colour()`, `vof_geometry(which)`, `set_vof_cutcell_flux_clamp()`,
`set_vof_solid_colour_zero()`; `vof_diagnostics()` gains `volume` / `raw_volume` / `solid_sum` /
`min_fluid` / `max_fluid` / `clipped_volume` / `cut_cells` / `solid_cells` / `clamped_faces`.
`src/vof/cutcell.hpp` holds the container-free rules; the geometry branch in `advect_wy.hpp` and
`momentum_advect.hpp` is taken **outside** the lambda, so a solid-free run executes the V1 kernel
bodies verbatim and the whole V1/V2a/V3/V4 battery is byte-identical.

Five things this rung paid for, all measured (`doc/vof_workorders_v5.md`, WO-Q findings):
- **What it approximates.** The PLIC polyhedron is reconstructed on the WHOLE unit cell and its slab
  volume multiplied by the open area, rather than clipped against the solid as well (Huang, *JCP*
  2025/2026 solid-clipped flux polygons). Conservative either way, exact where interface and wall
  are parallel or the cell is whole, O(1) wrong in the *distribution* inside a cell whose interface
  crosses its wall. `vof_diagnostics()['clipped_volume']` is the tripwire.
- **The cut-cell Courant number is `max(|a_f|, o_f |a_f| / max(eps_i, 0.1))`.** The second term
  alone (the work order's rule) is *smaller* than `|a_f|` wherever `o_f < eps_i` and licensed
  `dt = 1.85` on the packing gate, which lost **70 % of the liquid volume** in 200 steps while the
  flux sum still telescoped — over-CFL WY loses boundedness, not volume, which is why it is quiet.
  Consequence: in a packing the cut-cell limiter is up to **6×** tighter than the plain one.
- **Boundedness comes from clamping the FLUX, not from clipping C.** With the [0,1] clip as the only
  bound it fired at 3.2e-5 liquid volume per step and the drift reached 1.3e-8 in 30 steps.
  `vofCutFluxClamp` bounds `|F|` by what the donor holds (`o|a|` swept fluid volume, of which at
  most `eps·C` liquid and `eps·(1−C)` gas), applied to the one value both neighbours share — so
  conservation still telescopes bit-exactly — and **only for a MIXED donor**, because a pure-phase
  donor's algebraic flux is already exactly bounded and clamping it would break the exact full-cell
  cancellation. `set_vof_cutcell_flux_clamp(False)` is the ablation.
- **The conserved functional is `Σ eps_eff C` with `eps_eff = max(eps, 1/64)`, not `Σ eps C`.**
  `buildCellFraction` subsamples 4³, so a cell can read `eps == 0` while owning an open face; it is
  fluid, it receives flux, and the raw sum silently drops it. Both are reported.
- **The solid-band fill is a stencil device, and the canonical `"C"` is 0 in the solid.** Three
  passes with a *shrinking* depth budget (pass k writes solid cells at ghost depth ≤ 3−k, reads only
  fluid cells or cells filled in an earlier pass) plus a second exchange give a zero-slope
  continuation of the colour into the wall — the 90° Afkhami–Bussmann limit — so the MYC and
  height-function stencils see a consistent interface. Measured on a liquid cap resting on a flat
  SDF wall at a **half-integer** z (D/Δ = 24, σ = 1): apparent contact angle **89.94°**,
  Young–Laplace to **4.8e-3**, volume drift 4.5e-15. Writing the fill into `"C"` instead of 0 is
  **bitwise identical** in every measured quantity (the faces onto a solid cell have openness 0), so
  the "no colour in the solid" contract is free. Near-wall spurious currents are **Ca = 4.9e-4** in
  the open fluid, ~20× the V4 free-droplet 2.6e-5 at the same D/Δ. WO-S replaces only the pass-1
  rule with the θ-consistent one.

**Static contact angle on SDF solids (rung V5b, WO-S).** `set_contact_angle(theta_deg)` (or
`set_contact_angle_field`) replaces **pass 1 only** of the V5a solid-band fill by the volume
fractions of the plane that continues the fluid-side interface into the solid at the prescribed
angle, measured **through the liquid**: with `n_w = grad(sdf)/|grad(sdf)|` (solid -> fluid) and the
PLIC normal `m` (liquid -> gas), the contact-line condition is `m . n_w = cos(theta)`, so theta = 0
fills the band with liquid and theta = 180 empties it. The unmodified V3 height-function/MYC cascade
then returns the curvature of an interface meeting the wall at theta and the V4 balanced force does
the rest — **no force is added at the wall**. With no call the neutral 90-degree fill runs and the
whole V5a battery is **byte-identical**. Construction (`src/vof/wetting.hpp`, container-free like
`plic.hpp`): walk from the band cell along `n_w` to the first fluid cell, take the AZIMUTH of its
**fluid-only** Youngs normal (the 27-point weights restricted to non-solid cells, renormalized per
half-plane), build `m_theta = cos(theta) n_w + sin(theta) t_hat`, and anchor the plane by matching
that cell's own liquid volume (`plicAlpha`). New Python: `set_contact_angle`,
`set_contact_angle_field`, `contact_angle`, `set_contact_angle_pivot` (ablation),
`contact_angle_diagnostics`.

Four things this rung paid for, all measured (`doc/vof_workorders_v5.md`, WO-S findings):
- **The fill is exactly idempotent, and that is the whole design.** An interface that already meets
  the wall at theta is reproduced to **1e-15** (`tests/kokkos/test_vof_wetting.cpp` G0a), so theta is
  a fixed point of the discrete scheme. The work order's anchor — the PLIC centroid projected onto
  the wall along `n_w` — is **not** idempotent: it shifts the plane by `-sdf(p_f) cos(theta)`, with
  the wrong sign, measured **0.26 in cell fraction at theta = 60**. It ships as ablation mode 2 of
  `set_contact_angle_pivot`.
- **Only the IN-WALL half of the fluid-only normal is usable.** A fluid-restricted stencil cannot
  measure the WALL-NORMAL component — there is no colour below the first fluid row and the one-sided
  substitute reads a saturating profile over half the distance: **23 deg of error at theta = 30**
  against 2.3 deg for the same stencil with the solid rows present. Its AZIMUTH is exact for a plane
  (0.000 deg over the sweep), and since the rung OVERWRITES the angle anyway, using the azimuth and
  discarding the rest makes the end-to-end band error **1.1e-15** instead of inheriting the defect.
- **The anchor's own column is not enough.** Walking strictly along `n_w` gives a PURE-phase anchor
  in every column just outside the contact circle, where the continued interface still puts liquid
  in the band; the band then under-extends and the equilibrium angle comes out **5 deg biased
  towards 90**. Where the anchor is pure phase the fill averages the theta-planes of the anchor's
  MIXED neighbours (branch `neighbour_cells`), which removes the bias.
- **Accuracy, honestly rated.** Starting from the spherical cap of the prescribed angle on a flat
  SDF wall (D/dx = 24, Oh = 0.1, 500 steps, ratio 1) the equilibrium angle comes back
  **30.69 / 59.98 / 88.84 / 116.86 / 146.23** for theta = 30 / 60 / 90 / 120 / 150, i.e. within
  **1.2 deg up to 90 deg** and **3.5-3.8 deg** above it — and the 120-degree row was run to rest
  (`max|u|` down to 1.8e-4) to confirm that is a converged bias and not an unfinished transient. The
  two failing rows are also the two whose CONTACT RADIUS is under ten cells (9.1 and 5.3 against
  20.2 at theta = 30), and the same bias is *larger*, not smaller, with the wall on a cell face
  where the cut-cell reconstruction is exactly absent — so it is a contact-line resolution effect,
  not the cut-cell approximation. At ratio 100 the same protocol reads 29.92 / 58.36 / 88.44 /
  116.42 / 142.80, i.e. the equilibrium the fill selects is a property of the FILL and not of the
  density contrast up to 120 deg, with only the (under-resolved, contact radius 5.8 cells)
  150-degree row degrading further. On an
  SDF SPHERE (Rs = 12, drop of Rd = 8) the equilibrium CAP RADIUS is within **2.11 %** of the
  two-sphere reference at theta = 60/90/120, while the angle inferred from the volume and the apex
  height is off by 4.5 / -2.7 / -6.2 deg — that inversion carries `dtheta/dH ~ 10.5 deg per cell`,
  so a half-cell error in a colour column is worth five degrees and the cap radius is the
  well-conditioned reading. Full tables and the corrected gates: `doc/vof_workorders_v5.md`.
- **Where the SDF wall sits INSIDE the cell decides whether the contact line can move at all**, and
  it is a property of the cut-cell IBM, not of this rung. At exactly `k + 1/2` — the placement WO-Q's
  G5 and WO-S's G1 both ask for — the tangential MAC faces of the wall-adjacent cell sit ON the SDF
  zero level and `buildOpenness` (`sdf > 0` is fluid) closes them: measured `ox = oy = 0.000` on a
  cell with `eps = 0.5`. That cell is tangentially isolated, the contact line **cannot move**, and
  the unrelieved Young force appears as `max|u| ~ 1` on DOFs whose face openness is 0. At `k + 1/4`
  the same wall gives `eps = ox = oy = 0.75`, the contact line is mobile and the raw `max|u|`
  collapses from **7.9e-1 to 1.7e-3**. **This also answers WO-Q's open question 8**: the 0.788
  "wall-band spurious current" of the V5a G5 cap was entirely on zero-openness DOFs and is an
  artefact of the half-integer wall, not a surface-tension defect.

**Momentum consistency in cut cells (V5a item 8).** `enable_vof_momentum` composes with a solid: the
half-shifted CV gets fluid volume `½(eps(i−s_e) + eps(i))`, transverse face openness
`½(o_d(i−s_e) + o_d(i))` and, on the axial (cell-centre) face, the cell fraction `eps` of the cell
whose centre it sits on; both updates are done in fluid-volume units. Because every term of the
deviation form is a *difference of velocities*, WO-K's uniform-velocity identity is **bitwise in cut
cells too** — measured `max|u_adv − U| = 0` at ratios 10/100/1000 on a packing. Coupled draining
through the packing at ratio 10 (zero-mean buoyancy, 200 steps): colour drift **6.0e-14 per step**,
11 pressure iterations. **Open**: with a NON-zero-mean body force (an unbounded acceleration in a
periodic box) the momentum-consistent path runs clean for ~155 steps and then takes `C^e` to `+inf`
inside the advection, while the colour-only path completes; no bounded-quantity precursor was found
(WO-Q finding 9).

**The collocated path — `SolverColocated` (rung V8, WO-T).** `enable_vof` and
`set_density_mode("variable")` no longer throw on the collocated grid. That grid's pressure coupling
is the **ABC approximate projection** (average the cell velocities onto a MAC face field, project
THAT exactly, correct the cell field), and two facts follow:

- the **transport** half was already right — `uf_/vf_/wf_` is exactly discretely divergence-free,
  which is precisely what Weymouth–Yue's conservation proof needs — so `bridgeVelocityToVof` reads
  the FACE field (`uf_(i)` sits at i−1/2, the same low-face convention as flow's staggered `u(i)`,
  so it is the identical `copyFaceVelocity` shift on a different source view);
- the **force** half was not. A cell-centred `g_c − ∇_c P/ρ_c` is O(1) wrong at an interface cell
  even when every face is exactly balanced, so on this path the predictor carries **no force at
  all** (`buildRhsColoFF`) and every body/interfacial force enters as a **face acceleration**
  `a_f = dt (f_f − (P(i) − P(i−s)))/ρ_f` added after `centerToFace` (`src/collocated_varrho.hpp`),
  with the cell taking the **average of its two faces' TOTAL increment** `a_f − (ρ₀/ρ_f)Δφ` — the
  same averaging operator `projectCorrectCenter` applies to φ differences, a closed face (openness
  0) contributing 0. Basilisk's `centered.h` pattern (Popinet JCP 2009 §3). The face coefficient
  `c_f = o_f ρ₀/ρ_f` and `projectCorrectVar` on the face field are the staggered kernels unchanged.

Measured (`tests/kokkos/test_vof_collocated.cpp`, both backends): hydrostatic at **ratio 1000** with
the interface frozen — face `max|uf|` **8.1e-15** (host) / **9.2e-15** (CUDA) in the walled column,
`dP/dz = −ρ_f g` to **2.3e-8** relative, *independent of μ* over 0…0.1 (the force never passes
through `A = ρ_f/dt − μ∇²`, so the μ·dt² non-commutation of the staggered predictor does not exist
here); static droplet with a **constant curvature**, face `max|uf|` **2.8e-17** — the V4 exactness
identity, reproduced at the face.

**Read the FACE field, not the cell field, for a spurious-current number on this grid.** A cell
checkerboard is exactly annihilated by `centerToFace` (`½(U(i)+U(i−1))` kills the odd-even mode), so
the approximate projection is structurally blind to it — the invisible subspace of
`doc/collocated_invisible_subspace.md`. It is a property of the grid, not of this rung: the CONTROL
(the validated **constant-density** collocated path with a plain body force, where every V8 branch
is inert) reads cell `max|u|` **3.1e-2** at μ=0 / **8.7e-4** at μ=0.01 on the same walled column
while its face field sits at 5.1e-13; the V8 ratio-1000 run reads **2.8e-8**, six orders better, and
it **decays** (2.8e-8 at 100 steps → 8.0e-9 at 400). The accumulated pressure inherits it through
the `centerToFace` leak of the checkerboard's envelope, which is the 2.3e-8 above.

Scope of the collocated rung: **all-fluid only** (`set_pressure_geometry`); an immersed solid, the
ghost projection and `set_rho_face_harmonic` throw with a message naming the reason.
`enable_vof_momentum` stays staggered-only (the collocated construction needs Favre face states,
AMR-Wind's pattern), so the collocated path is rated to density ratio **≲ 100 for cases with
motion**; a high-ratio case at REST is exact. Under `set_density_mode` on this grid the AUTO scheme
falls back from the ghost projection to gauge-exact (which, all-fluid, IS the plain central
difference). Known gap, recorded not guarded: an OUTFLOW face's `bcCorrectOutflow` adjustment of
`uf_` is not mirrored into the stored face increment, so open boundaries on this path are untested.
Gates: `tests/kokkos` `vof_collocated`, `tests/kokkos_mpi` `vof_collocated_mpi_np{1,2,4}`,
`tests/study/vof_collocated.py` (the staggered/collocated columns of the V4 physics battery).

**Scope — say this to users:** **Staggered is the reference**; the collocated path is rung V8 (the
paragraph above) and is all-fluid, ratio ≲ 100 with motion. An **immersed solid is supported since
rung V5a** — `set_solid(...,
cutcell_pressure=True)` — with the openness-weighted flux above (STAGGERED only); an all-fluid
`set_pressure_geometry` is of course also fine. Without
`enable_vof_momentum` the rung is **valid only at modest density ratios for cases with motion**; a
high-ratio case at REST (the hydrostatic acid test) is exact either way. **With** it, the shipped
build is honestly rated to ratio ~1e3: the uniform-velocity residual through the coupled step is
floored at 1.2e-7 by the solver's FLOAT momentum-operator storage (`Solver::FV`, a pre-existing
defect unrelated to VoF — a `-DPECLET_FLOW_MREAL_DOUBLE` build measures 1.2e-15, flat across four
decades of ratio), and at ratio 1e4 that injection is enough to destabilise a long run.

Three traps this rung paid for, all in `doc/vof_workorders_v2.md` (WO-J findings):
- **The face index conventions differ by one cell.** flow's `u(i)` is the **low** (−x) face of cell
  `i`; `WyAdvector`'s `uf(i)` is the **high** (+x) face. The bridge shifts along each component's
  own axis. Omitting it is invisible in a uniform flow, invisible in each axis' own divergence and
  invisible in `max|div(open·u)|` — and cost **35 % of the colour volume** over 1000 steps, because
  the advector sums the three axes at ONE cell. Gate A of `tests/kokkos/test_vof_twophase.cpp`
  (solver vs a standalone advector on the same physical LeVeque field) exists for this.
- **The VoF dt limit is interface-LOCAL** (mixed cells and their neighbours, `maxCourantInterface`),
  not a global max — measured 22× over-throttling on a jet-plus-quiescent-interface scene. The band
  predicate is a colour *difference*, not `mixed`: a grid-aligned sharp interface has no mixed cell.
- **The colour conservation floor is the projection's divergence residual**, not the advection
  (measured: dV/V bounded at 5.6e-13 with `max|div|` 1e-12…1e-11; V1's standalone floor at
  `max|div| ~ 1e-15` was 5.7e-14 over 3200 steps).

`set_rho_face_harmonic(True)` switches the projection's ρ_f (coefficient **and** correction) to the
harmonic mean. Default OFF and it should stay off: arithmetic ρ_f is the harmonic mean of the
mobility 1/ρ — the series-correct choice for a normal flux — and it is what makes hydrostatic
balance exact, since the momentum time term and the face body force keep the arithmetic mean.
Measured with it on: ∂P/∂z relative error **0.34** instead of 1e-15. It ships as a measured knob for
the coefficient-coarsening question (VOF_PLAN S3), not as an alternative scheme.

Gates: `tests/kokkos` ctests `vof_plic`, `vof_advect`, `vof_twophase`, `vof_momentum`,
`vof_curvature`, `vof_surface_tension`, `vof_cutcell`, `vof_wetting`, `vof_collocated`;
`tests/kokkos_mpi` `vof_advect_mpi_np{1,2,4}`,
`vof_twophase_mpi_np{1,2,4}`, `vof_momentum_mpi_np{1,2,4}`, `vof_curvature_mpi_np{1,2,4}`,
`vof_surface_tension_mpi_np{1,2,4}`, `vof_cutcell_mpi_np{1,2,4}`, `vof_wetting_mpi_np{1,2,4}`,
`vof_collocated_mpi_np{1,2,4}`;
`tests/study/vof_collocated.py` (the V8 staggered/collocated columns);
`tests/study/vof_cutcell.py` (the V5a battery: conservation through a packing, coupled draining,
the 90° cap on a cut wall); `tests/study/vof_momentum_consistency.py` (the ratio sweep, the
falling drop, the RT near-Nyquist check — every gate there records the pressure iteration count
against its cap and treats a capped run as INVALID); `tests/study/vof_surface_tension.py` (the V4
physics battery: `static`, `wave`, `lamb`, `hysing1`, `hysing2`, `falling`, `limits`);
`tests/study/rayleigh_taylor.py` (diffuse and sharp records side by side).

**Benchmarks (V4, `tests/study/vof_surface_tension.py`).** **Hysing rising bubble**, quasi-2D
64×128×4, adaptive `dt`, against the published reference: case 1 max rise velocity **0.2497 vs
0.2417 (+3.3 %)** and `y_c(3)` **1.0808 vs 1.0810 (−0.02 %)**; case 2 **0.2574 vs 0.2502 (+2.9 %)**
and `y_c(3)` **1.1082 vs 1.1376 (−2.6 %)**. **Momentum consistency (V2b) is worth 14 % on case 1 at
density ratio 10** — with `enable_vof_momentum` OFF the same run reads +16.9 % / +11.8 %, which is
the discriminating case WO-K's uniform-velocity gate could not provide. Capillary wave vs the
analytical dispersion: frequency **−2.1 to −3.7 %** at 32–64 cells/λ. Lamb mode-2 droplet: **−6.3 to
−7.0 %**, and neither confinement (φ 6.5 % → 0.8 %) nor resolution explains it — recorded as a
measured deviation, not a pass. **Falling drop** (the gate WO-K deferred): the periodic zero-mean
body force conserves *momentum*, not volume flux, so at ratio 800 the light ambient recoils at ~19×
the drop's speed and the LAB-FRAME drop velocity — what WO-K measured — is a near-cancellation. The
relative velocity reaches **0.786 / 0.828 / 0.869 of the Hasimoto-corrected Hadamard–Rybczynski at
D/h = 10 / 15 / 20**, and is **insensitive to the momentum-sweep count to four digits** (0.826 at 60
sweeps/step, 0.828 at 2649) — so WO-K's suspected under-resolved momentum solve is refuted. At 15
cells/diameter it is 17 % low, just outside Arrufat's "within 15 %"; at 20 it is 13 % low, inside.

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

**OPEN DEFECT — solid intersecting an OPEN domain face (2026-09-01).** Every validation below is
**all-fluid**; no test or script anywhere combines `set_domain_bc` with `set_solid`. Within that
gap: when immersed solid **cuts an inflow/outflow face**, the cut-cell pressure solve runs to its
iteration cap with `max|div|` 4e-3 and diverges outright at MG depth ≤ 2. Measured A/B at 128³, all
else identical, only the bed moved: spheres clipped by the inlet/outlet planes give 260.8 iters
(5 of 6 steps capped, `max|div|` 4.0e-3); the same bed pulled clear of those faces gives **32.7
iters, no capping, `max|div|` 1.95e-06 over 42 steps**. Not the agglomerated bottom
(`BOTTOM=smoother` also caps); FCG caps and Chebyshev NaNs. Suspected: how a cut cell on an open
face reconciles the operator openness α (Dirichlet, mean-removal off) with the flux openness β.
*A bed clear of the open faces — the normal case — is fine, so this is narrower than a blocker.*
Full A/B, mechanism and a minimal-reproducer plan:
[`doc/cutcell_openbc_convergence.md`](doc/cutcell_openbc_convergence.md).

**Validated:** lid-driven cavity vs Ghia et al. Re=100 to ~0.7% rms (`scripts/verify_lid_cavity_sdflow.py`);
developing plane channel (uniform inlet → parabolic Poiseuille outlet, `u_max/U_mean`→1.5, exact mass
conservation, machine-precision divergence; `scripts/verify_channel_sdflow.py`); backward-facing step
(Gartling expansion-ratio-2, `scripts/verify_bfs_sdflow.py`) — reattachment `x_r/S` 5.3 (Re_S=100) → 8.3
(Re_S=200) on the Armaly/Biswas curve, `PECLET_FLOW_BFS_RE800=1` pushes to the Gartling Re=800 benchmark.

The **rediscretized geometric pressure multigrid is multilevel on these non-periodic domains** (not just the
periodic/IBM case): each coarse level re-imposes the boundary face openness (Neumann wall/inflow → 0,
Dirichlet outflow → open) and the trilinear prolongation fills the non-periodic boundary ghosts
(Neumann → zero-gradient via `applyNeumannGhost`, Dirichlet → 0 via `applyOutflowGhost`). Gated on
`has_bc_`, so the periodic/IBM path is byte-identical. *Until WO-H (2026-08-30) only the Dirichlet
half existed and the Neumann ghost held the periodic wrap — that was the MG-PCG stall; see the
pressure-driver bullets above.*
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
Homsy sphere-array drag**; the multi-rank step is bit-exact to the single-rank — **48 `tests/kokkos_mpi`
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

**The drag coefficient gets the same fill, one phase earlier (WO-I, 2026-08-30) — and it moves the
porous CFD-DEM numbers.** Under `set_porous_continuity`, `addDragDiagonal` builds the momentum
diagonal from the FACE drag `β_f = ½(β(i) + β(i−s_c))`, deliberately, so it matches the projection's
SIMPLE coefficient and correction (`doc/porous_drag_scheme.md` §2). But every writer of `drag_beta`
writes inner cells only — `set_field`/`applyClosure`, and the CFD-DEM driver, which folds its
ghost-band deposit onto the owners and then **zeroes** that band — and the only
`fillPropGhosts(dragBeta_)` was inside `project()`, i.e. *after* the momentum stencil builds at the
top of `step()`. So on the first inner plane of every block the momentum diagonal carried **β/2**
while the projection used the full exchanged β: the β_f mismatch `addDragDiagonal`'s own comment
warns about. `step()` now calls `fillDragBetaGhosts()` next to `fillCellForceGhosts()`, right after
`updateProperties()` — the one point after both writers and before the first consumer;
`project()`'s fill is thereby redundant but is kept (its own contract, one exchange on the porous
path). Reach: **only** `porous_` reads that ghost (`addDragDiagonal` uses the cell value alone
otherwise), so the incompressible-drag CFD-DEM path, the single-phase regression and every non-drag
case are bit-identical. Measured on the periodic uniform fixed bed at N = 16: the mean face drag was
low by exactly one half-plane in 16, `Σβ_f/N = 31β/32`, and the Ergun agreement recorded in
`doc/porous_drag_scheme.md` §5 as "~3 %" is precisely that — `32/31 = 1.032258` against a measured
`3.22581 %`. With the fill it drops to **2e−8…7e−8** (machine precision, matching the incompressible
path) at all three drive levels. Gate: `tests/kokkos_mpi/test_dragbeta_ghost_mpi.cpp` (np 1/2/4,
gating the assembled diagonal directly via `getMomentumDiagonal`). **Nothing was re-baselined** — the
full before/after table is in `doc/vof_workorders.md`, WO-I findings.

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
