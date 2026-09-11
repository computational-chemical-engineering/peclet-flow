# CLAUDE.md

Working reference for `peclet-flow` (`peclet.flow`): what the code does, how to build and test it,
and the conventions and traps someone editing this repo must know. Design notes that describe
shipped behaviour are in [`doc/`](doc/README.md); **campaign narrative, dated investigations and
superseded design discussion are in [`doc/history/`](doc/history/README.md)**. Suite-wide contracts
are in `../docs/`.

## What this is

Performance-portable incompressible Navier–Stokes solver for flow in complex geometry: staggered MAC
grid, signed-distance-field solids, a cut-cell Immersed Boundary Method, and a pressure projection
with a geometric multigrid Poisson solve, **in physical units**. One Kokkos source runs on CUDA, HIP
and OpenMP; simulations are driven from Python, never from a C++ main.

`peclet::flow::IbmSolver` (`src/flow_ibm.hpp`) is the solver; `src/flow_bindings.cpp` exposes it as
`peclet.flow.Solver` (staggered MAC, the reference) and `peclet.flow.SolverColocated`
(cell-centered velocities + ABC approximate projection) — the same Python API through a `GridLayout`
policy. The raw-CUDA implementation was retired 2026-06 after a bit-identical validation (restore
tag `pre-cuda-retirement`); pore-network *extraction* is the sibling `../pnm` project since 2026-07.
Validation: Ghia lid cavity, developing channel, backward-facing step, Taylor–Green, Poiseuille and
Zick & Homsy sphere-array drag.

## Build

Kokkos is found with `find_package` against a bootstrapped prefix `../extern/install/<backend>`
(`nvidia-cuda` / `host-openmp` / `lumi-hip`, built once by `../tools/bootstrap_deps.sh` — a **hard
build dependency**); nanobind comes from the active interpreter via `SuiteNanobind`, so no cmakedir
is needed. Put `nvcc` on `PATH` for CUDA. Requirements: Kokkos 5.x (C++20), CMake 3.24+, Python
3.10+, nanobind + scikit-build-core, `../core`, and MPI for the distributed path.

```bash
source ../.venv/bin/activate                  # THE suite venv (../CLAUDE.md "One venv")
CMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" pip install .     # canonical install
cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DPECLET_FLOW_BUILD_TESTS=ON -DPECLET_FLOW_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_dev -j8
export PYTHONPATH=$PWD/build_dev              # -> build_dev/peclet/flow/_flow.*.so
```

`PECLET_FLOW_BUILD_TESTS=ON` registers every suite in the module's own tree; `PECLET_FLOW_MPI=ON`
adds MPI, `Solver.init_mpi`/`rank`/`size` and `flow.mpi_block()` (OFF leaves the single-rank module
byte-identical). Swapping backend = swapping the prefix and using a second tree.

**Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`.** FindMPI may pick ParaView's bundled `mpiexec` off
`PATH`, which launches the OpenMPI-linked binaries as singletons — every `*_np4` then silently runs
four independent np = 1 jobs.

## Test

```bash
ctest --test-dir build_dev -N                                   # 157 registered, nothing hidden
OMP_NUM_THREADS=8 OMP_PROC_BIND=false ctest --test-dir build_dev --output-on-failure -LE bench
ctest --test-dir build_dev -R '_np[0-9]+$' --output-on-failure   # the distributed suite only
```

157 registered / **155 with `-LE bench`** (G.6 added `no_float_operator_casts`): 45 from
`tests/kokkos` — of which `bench_rbgs` and `vof_timing` carry the `bench` label and are
instruments, not gates — 106 from `tests/kokkos_mpi` (35 cases at np = 1, 2, 4 plus one np = 8
rung), and 6 Python ctests on the module built in that tree (`regression_staggered`,
`verify_poiseuille_flow`, `verify_lid_cavity_sdflow`, `verify_colocated_taylor_green`,
`no_env_knobs`, `no_float_operator_casts`). Always bound the OpenMP pool — an unbounded one on a
many-core host is an hour-long trap.

More verification lives in `scripts/verify_*_sdflow.py` and `validate_zick_homsy_sdflow.py` (the
external ground truth), run with `PYTHONPATH=<tree>`. `tests/regression/sdflow_regression.py` is
the accuracy + iteration-count regression against a saved baseline (`--update` re-records;
`--solver colocated --scheme ghost` has its own); `tests/regression/state_hash.py` prints the
SHA-256 of the final state of one fixed-seed run per public entry path (the byte gate for any
refactor that must not change numerics — the hashes at each package-F milestone are in the commit
messages); `tests/study/` holds instruments, not gates.

CI: `ci.yml` builds one host tree (kernel ctests + regression + verify), then the same tree with
MPI at np = 1, 2 (4/8 if time allows). `quality.yml` runs ruff critical errors and a **blocking**
clang-format 18.1.8 check over `src/` and `tests/`, with six files the `vof-w4` WIP branch owns
temporarily excluded. `docs.yml` publishes Doxygen from `docs/Doxyfile`.

## Layout

All header-only Kokkos C++20 in `namespace peclet::flow`.

- `src/flow_ibm.hpp` — `template <class Grid> class Solver` (`IbmSolver` = `Solver<Staggered>`):
  includes, the nested structs, every member DECLARATION (with its docstring) and the whole state
  block (QUALITY_PLAN G.1: split 2026-09-11, 11886 -> 4046 lines). The out-of-line member
  DEFINITIONS live in twelve domain headers included at the bottom (each reopens
  `namespace peclet::flow`; declarations + state never move):
  `flow_ibm_core.hpp` (allocation, rho/mu/dt + driver setters, stencils/ghosts/advection inputs,
  the momentum-solve smoother family, reductions, gather/scatter, the field registry),
  `flow_ibm_project.hpp` (`step()`, the `project()` stages, the `buildRhs` family,
  `applyFaceAcceleration`), `flow_ibm_scene.hpp` (scene instances + motion upload, wall velocity,
  wall-flux divergence, `setSolidFromScene`), `flow_ibm_geometry.hpp` (`setSolid`/`setSolidDevice`
  + its nine stages, pressure geometry), `flow_ibm_hydro.hpp` (hydro force/torque, reaction-budget
  terms, wall probes), `flow_ibm_vof.hpp` (colour field, cut-cell VoF, contact angle, VoF blocks,
  curvature, surface tension / CSF RHS, advection, `stepAdaptive`), `flow_ibm_phase_change.hpp`
  (interface area, mass flux, energy transport, budgets, the `pc*` machinery),
  `flow_ibm_closures.hpp` (property closures/modes, porous continuity, drag, prop/eps ghosts),
  `flow_ibm_scalars.hpp` (`addScalar`/`setScalarBc`/`advanceScalars` + BC application),
  `flow_ibm_bc.hpp` (domain BCs/profiles, `pressureBcGhost`, `fillVelGhosts*`,
  `setupBcDiffusion`), `flow_ibm_mpi.hpp` (`initMpi`, `redistribute`, `rebalanceByWeights`, the
  post-repartition field-resize passes; `#ifdef PECLET_FLOW_MPI`-guarded), `flow_ibm_diagnostics.hpp`
  (state getters, divergence probes, timers, the outflow/backflow census). `src/flow_bindings.cpp`
  — the nanobind module.
- `src/mac_cutcell_mg.hpp` (`CutcellMG`, pressure MG), `src/mac_velocity_mg.hpp` (`VelocityMG`),
  the `src/mac_*.hpp` operators, `src/cut_cell_ibm.hpp` (the Robust-Scaled overlay: `poly_*`,
  K/M/X/Nbc/R, `D_rescale`), `src/staggered_advection.hpp` (`sadv::advect`),
  `src/gauge_exact_gradient.hpp` (`gpCenterGrad`), `src/ghost_projection.hpp`.
- `src/vof/` — the VoF stack (below). Its container-free kernels were promoted to
  `peclet::core::vof` in `../core`, so `src/vof/{plic,curvature,cutcell,wetting}.hpp` are thin
  includes. **New container-free VoF math belongs in `core`**; the drivers stay here.
- `tests/{kokkos,kokkos_mpi,python,regression,study}`, `scripts/`, `doc/` + `doc/history/`.

## Settled decisions — do not reverse silently

Each of these was chosen *against* the obvious or textbook alternative, on measured evidence. They
are the ones that have actually been re-proposed by mistake. Full entries, with verbatim quotes and
provenance, in [`../docs/decisions/flow.md`](../docs/decisions/flow.md); reversing one takes a new
recorded decision, not a judgement call in the moment.

- **Collocated pressure coupling is the Almgren–Bell–Colella approximate (MAC) projection — NEVER
  Rhie–Chow.** The residual cell divergence is *intrinsic* to cell-centred velocity placement, and
  the permeability gap lives in the momentum solve, not the projection. Rhie–Chow has been proposed
  by mistake repeatedly; it is not an "upgrade".
- **The pressure solve is PCG (Krylov), not RB-GS**, for cut-cell IBM.
- **Backward Euler is the default time integrator.** Crank–Nicolson was ported and reverted.
- **Porous beds use ε-weighted momentum with a matched projection.** Plain incompressible continuity
  with ε only in the drag term is the *wrong* constraint.
- **Masking excludes both cut cells and solid cells**, never cut cells alone.
- **IBM velocity-MG must never un-scale the residual by `1/D_rescale`.**
- **The ORB must never split the wall-normal axis** in wall-bounded flow.
- **Geometric const-coeff operators + masking are the validated defaults**; Galerkin/CG is opt-in.
- **Momentum advection uses the actual wall velocity field**, not `maskVelocity`'s solid zeros.
- **Float `MReal` operator storage silently breaks A·1=0** at high MG contrast — it fails without an
  error, so it will not announce itself.
- **The rotational (Timmermans) pressure update must be restored, not the non-rotational Goda form.**
- **`set_ghost_projection(True)` must be called before `set_solid`** — call order is load-bearing.
- **Distributed cut-cell MG coarse levels must be nested**, never independently re-decomposed.
- **Fresh (newly-uncovered) cells are seeded with the local wall velocity**, not the stale interior.
- **Closed dead ends, do not re-attempt:** mode-10 quadrature, the Seo–Mittal pressure-only split,
  ghost-as-production, and the double-diagonal fallback (measurably worse, not merely unnecessary).

⚠️ **Unresolved — do not rely on either reading until settled:** the interstitial-vs-superficial
drag normalisation. `ibm-accuracy-sphere-validation.md` states both that our K (=Zick–Homsy) is
interstitial while vdH/Tenneti report superficial (l.45), and that K is reported superficial (=Z&H)
while vdH/Beetstra is interstitial (l.110). Both note the dilute limit hides the difference. This
decides a (1−φ) factor on published permeabilities.

## Conventions and hard rules

- **SDF sign:** negative inside solid, positive in fluid.
- **Indexing:** `I = x + y*nx + z*nx*ny` (x fastest); Python arrays are `order='F'`, shape
  `(nx, ny, nz)`. **Staggered placement:** u at (i+½,j,k), v at (i,j+½,k), w at (i,j,k+½), p at
  cell centres.
- **Kokkos device code lives in `.hpp` compiled as C++** (the launch compiler routes it through
  `nvcc`/`hipcc`) — never a `.cu`. `parallel_for`/`parallel_reduce` over `Kokkos::View`,
  `MDRangePolicy` for 3-D loops.
- **No environment variable changes a result** (`../docs/QUALITY_PLAN.md` D3, executed 2026-09-08
  in `ad917b1`). Every numerics-changing or algorithm-selecting read became a per-solver setter
  with the old unset behaviour as its default, or was deleted with the ablation it served; the
  process-global statics went with them. Only debug *printing* reads remain, and
  `tests/python/test_no_env_knobs.py` (ctest `no_env_knobs`) fails if a new one appears in `src/`.
  The old-name → setter table is in the `ad917b1` message. Do not add one, and do not document a
  workflow that sets one.
- **Two API tiers (QUALITY_PLAN D2, since 2026-09-10).** `Solver` / `SolverColocated` carry the
  PUBLIC surface — what a user of single-phase, VoF, porous, moving-geometry, scalar-transport or
  thermal flow needs to set up, run and read out a run (~140 members). Everything a developer uses
  to inspect, profile, ablate or tune — the `*_diagnostics/_stats/_census/_budget/_ledger/_probe/
  _timing` families, `last_*`, solver tuning beyond the driver selection, the ablation switches,
  the zero-copy/MPI internals `field_view`/`exchange_field`/`rebalance_by_weights` — lives on
  `s.diagnostics` (a view holding a reference to the solver, one class per grid). Integer codes are
  strings everywhere (`set_domain_bc('-x', 'inflow', …)`, `set_advection_scheme('koren')`,
  `add_scalar(scheme='koren')`, `set_scalar_bc(name, '+z', 'dirichlet', v)`), and every on/off pair
  is one setter with a leading `enabled` bool (`set_phase_change_thermal(False)`).
- **Call order is enforced where a wrong order gives a wrong result.** The settings that are folded
  into the operators when the geometry is built — `set_domain_bc`/`set_domain_bc_profile`,
  `diagnostics.set_aperture_order`, `set_exact_crossings`, `set_openness_override`,
  `set_fluid_only_constraint`, `set_ghost_projection` / `set_collocated_scheme('ghost')` — raise
  after `set_solid`/`set_pressure_geometry`/`set_solid_from_scene`; `set_decomposition` and
  `diagnostics.set_comm_avoiding` raise after `init_mpi`. `set_rho`/`set_mu`/`set_dt` may be
  called at any time (a change after the geometry rebuilds the momentum operator at the next step;
  under a physical domain the FIRST `set_rho` and the FIRST `set_dt` pin the reference scales).
  `set_decomposition(levels, max_imbalance)` and `flow.mpi_block(..., levels=, max_imbalance=)`
  must get the **same** values. Select the pressure driver **last**: `set_property_model("rho", …)`
  fires the density mode, which re-selects Chebyshev and discards an earlier choice.

## Python API and units

```python
import peclet.flow
s = peclet.flow.Solver((nx, ny, nz), extent=(Lx, Ly, Lz))   # PHYSICAL box; spacing derived
s.set_rho(1.0); s.set_mu(0.01); s.set_dt(60.0)
s.set_body_force((1e-2, 0, 0))                              # force per unit volume
x, y, z = s.cell_centers()
s.set_solid(sdf, cutcell_pressure=True)                     # SDF [x,y,z], < 0 inside
for _ in range(n_steps):
    s.step()
u = s.get_u()                                               # [x,y,z]; get_p() = physical pressure
```

Everything in and out is in one consistent system of the caller's choosing — properties, `dt`,
forces, boundary velocities and profiles, `sigma`, the slip length, the SDF and scene coordinates
in; `get_u/v/w`, `get_p`, `get_uf`, `vof_curvature()`, `max_open_divergence()`, the hydro force and
torque out. Under MPI the constructor takes **this rank's** block; pass the global grid as
`global_cells` and the global box as `extent`/`origin`. `extent=None` keeps **cell units**
(spacing 1, origin 0) and is bit-identical to the pre-2026-09 code.

**The trap.** The solver computes on the unit lattice; the metric is folded into constants at the
API boundary (`Solver::UnitScales` in `src/flow_ibm.hpp`, derivation in the comment above it), with
`hRef = min h`, `rhoRef` = the first `set_rho` and `tRef` = the first `set_dt`, so stored float
operator coefficients stay O(1) in any unit system. **The raw field registry —
`get_field`/`set_field` and `diagnostics.field_view`/`diagnostics.exchange_field` — hands out those
INTERNAL arrays**, unlike `get_u`/`get_p`, which convert. A driver writing `force_x` or `drag_beta` directly (CFD-DEM does)
must convert with `s.unit_scales`.

**Anisotropic cells** work on both solvers, single phase and VoF
([`doc/anisotropic_metric.md`](doc/anisotropic_metric.md),
[`doc/anisotropic_vof.md`](doc/anisotropic_vof.md)). Spacings agreeing to 1e-12 relative are snapped
to one, so an isotropic domain stays bit-identical however its extent was written; both multigrids
defer an axis already `theta` (2.0) times coarser than the finest coarsenable one, without which a
stretched grid does not solve at all. Two guards remain: the CFD-DEM coupling driver refuses an
anisotropic box (its *particle model* has no single length to form `Re_p` — not a metric gap), and
`scripts/check_decomposition.py` still models the isotropic coarsening rule.

## Pressure solve

Geometric multigrid (`CutcellMG`), RB-GS smoother, **rediscretized** cut-cell coarse operators.
Four outer drivers, **one per solver**; the three Krylov ones are mutually exclusive both ways,
last set wins:

| driver | select with | use |
|---|---|---|
| standalone V-cycle | default | multi-rank default; `set_pressure_multigrid(True, levels=1)` = pure RB-GS |
| MG-PCG | `set_pressure_pcg(True, iters, rtol)` | single-rank default (auto-enabled at np = 1) |
| flexible MG-CG | `set_pressure_fcg(True, iters, rtol)` | Polak–Ribière β, tolerant of a non-symmetric preconditioner — try it when PCG caps |
| Chebyshev | `set_pressure_chebyshev(True, iters, rtol)` | no global dot products per iteration; the variable-density / porous default |

`set_pressure_pcg(False)` **raises** — MG-PCG is the terminal fallback of the dispatch and cannot be
deselected on its own; name the driver you want instead.

- **High coefficient CONTRAST makes the V-cycle preconditioner indefinite** and both CG drivers cap
  above density ratio ~10³ — the cause is the arithmetic coarsening of the face coefficient, not
  float storage. Only Chebyshev is healthy there; coefficient-aware coarsening is the open fix.
- **Operator STORAGE precision is a separate axis, and a typed CMake option (QUALITY_PLAN G.6).**
  `MReal` (`mac_cutcell_mg.hpp`) types the pressure hierarchy and, via `IbmSolver::FV` and
  `IbmOverlay` (the cut-cell overlay: `cut_cell_ibm.hpp`'s `poly_*`/`ibmFillEntry`/
  `ibmModifyStencil`, templated on `Real`), the momentum stencil and the closure factors K/M/X/
  Nbc/R/D_rescale. `option(PECLET_FLOW_OPERATOR_DOUBLE)` (`pip install . -C
  cmake.define.PECLET_FLOW_OPERATOR_DOUBLE=ON`) makes all of it double at +12 % step time; OFF
  (default) is bit-identical to before G.6. Float rounding breaks `A·1 = 0`, and on a high-contrast
  bed the residual floors and then **rebounds** — the run is **invalid, not degraded**
  (`../docs/SCALING_ISSUES.md` #1). `tests/python/test_no_float_operator_casts.py` (ctest
  `no_float_operator_casts`) fails on a new hard `(float)` cast / `float`-typed operator view in
  `src/` outside its allow-list (a `// PRECISION-EXEMPT: <reason>` marker, or one of the whole-file
  exemptions it documents — `ghost_projection.hpp`'s `GpOverlay` is float-only still, a known,
  deferred gap: it needs the same `Real`-templating `IbmOverlayT` got, and it is NOT dead code —
  it backs the AUTO-default `'ghost'` collocated scheme).
- `set_pressure_bottom("auto" | "smoother" | "agglomerated")` — **`"auto"` is the default**: it
  agglomerates the coarsest level into a global, decomposition-independent operator and solves it
  exactly whenever that grid exceeds `set_pressure_bottom_extent` (4) cells on any axis. Porous and
  variable-ρ rebuild it every step; avoid `auto` with a badly-factored grid there.
- **Depth follows the factors of two, per axis.** An axis coarsens only while it stays even, so an
  **odd dimension never coarsens at all** (384×128×256 → 5.0 pressure iterations/step, ×255 →
  16.2), and under MPI only if *every rank's block* is even on it. **Telescoping is the default**
  (`set_pressure_telescope`): a level that cannot coarsen in place merges ORB siblings onto fewer
  ranks and continues to 3³. The in-place requirement on intermediate levels is the top open item
  at scale (`../docs/DECOMPOSITION_AND_MULTIGRID.md` §2.8).
- Decomposition: `set_decomposition(0)` (default) is aligned ORB (fine-grid splits snapped to a
  power of two); `set_decomposition(L>=2)` is coarse-first — decompose the grid coarsened `L-1`
  times, then refine the partition upward, so blocks nest for the full depth and balance better.
  `decomposition()` takes the deepest candidate within `max_imbalance` (1.05), a pure function of
  (ranks, grid, levels). Check a combination with `scripts/check_decomposition.py` first.

## Velocity solve

Per-component backward-Euler diffusion, RB-GS or an optional V-cycle (`set_velocity_multigrid`),
both distributed. Three operator modes: IBM-periodic (staircase), all-fluid domain-BC (folded
constant-coefficient), and **mixed** for an immersed solid *with* domain BCs. `bcStencilPath()` and
`implicitAdv()` must agree with the solver in use — unenforced, turning velocity MG on silently
made advection explicit and two converged solves sat 3e-4 apart.

`set_velocity_residual_tolerance(rtol)` stops a component once
`max|b − A u| <= rtol · max(max|b|, max|A u|)`. **The default (`rtol < 0`) follows the active
pressure driver's rtol** — the projection consumes u* and resolves its divergence to *its*
tolerance, so "no less accurately than pressure" is the rule with no free constant; `0` restores the
legacy update criterion. At least one sweep always runs, and there is deliberately no early return
for a warm start that already meets the tolerance (skipping it drifts the hydrostatic acid test by
1e-8 in dP/dz). With `set_velocity_multigrid` never called, `diagnostics.set_velocity_multigrid_auto` takes the
3-level V-cycle on a distributed run of ≥ 8 M cells once cells/rank fall below 65536; the V-cycle
needs no depth on a pore-confined bed, so no telescoping.

## Collocated solver (`SolverColocated`)

Read [`doc/collocated_invisible_subspace.md`](doc/collocated_invisible_subspace.md) (mechanism)
before touching this path; [`doc/collocated_paper_plan.md`](doc/collocated_paper_plan.md) tracks the
results and [`doc/fluid_only_constraint_plan.md`](doc/fluid_only_constraint_plan.md) is the
production plan. **The default is AUTO = `"ghost"`** (the fluid-only scheme) wherever the
configuration supports it, falling back to gauge-exact with a stderr notice on porous / variable-ρ
/ domain-BC / Chebyshev. Any explicit `set_collocated_scheme` / `diagnostics.set_face_interp` /
`diagnostics.set_ghost_projection` disables AUTO; tests and baselines pin schemes explicitly.

`set_collocated_scheme` takes one of four strings. `"ghost"` ==
`diagnostics.set_ghost_projection(True, 2, 2)`: fluid-only binary-openness constraint, directional
closures, gauge-exact gradient — family-free, no stabilizer, but BiCGStab (~2.3–2.7× the pressure
stage) and ~1.6 KB/cell of overlay. `"gauge-exact"` converges yet carries an attractor family;
`"plain"` is first order and legacy; `"embed"` is the complete Basilisk embed.h port (the FV
momentum operator with the true-normal wall drag as a defect correction, the openness-weighted
pressure force, the wall-aware face map and the sliver mask — the live candidate for the accuracy
ceiling; its two intermediate rungs are `diagnostics.set_face_interp(5 | 6)`). Inside the C++
these are `faceInterp_` 0 / 9 / 7 (5, 6); the integer modes 1–4 and 10–13 and the `gauge-2a`
gradient branch were deleted with their kernels at 1.0.0. MPI validated np = 1, 2, 4 (np ≥ 16
unresolved); the mixed `(matrix_order=1, rhs_order=2)` ghost mode is **do-not-use**,
march-unstable above ~2000 spheres.

## Domain boundary conditions

`set_domain_bc(face, type, velocity=(vx, vy, vz))` with `face` one of `'-x'`, `'+x'`, `'-y'`, `'+y'`,
`'-z'`, `'+z'` and `type` one of `'periodic'` (default), `'wall'` (no-slip), `'inflow'` (Dirichlet
velocity), `'outflow'`, `'slip'` (free-slip/symmetry, which also **mirrors the SDF ghost band**
about that face, `mirrorSdfSlipFaces`, or a half channel closed by a symmetry plane would see the
far wall as a solid). Tangential walls use a face-fold in the implicit diffusion so `u_inner` stays
implicit. `set_domain_bc_profile(face, profile[Nb,Nc,3])` prescribes a per-position inlet (and
sets the face to inflow) — the backward-facing step is realized purely this way. Only a call that
would CHANGE a face's TYPE must precede the geometry (it raises afterwards); a VALUE update
(`velocity=`, or a new profile) on a face whose type is unchanged is allowed at any time and takes
effect the next step — ramp an inflow jet or a lid's tangential speed after `set_solid`. With no
immersed solid, use `set_pressure_geometry(all_fluid_sdf)`.

- **Open boundaries** split face openness in two: the *operator* openness (pressure matrix) is 0 at
  walls and inflow and open at outflow (Dirichlet p = 0, mean-removal off); the *flux* openness
  stays open at both so their flux is counted. **Outlet reversal** is the one conditionally-stable
  regime and it is instrumented: `set_backflow_stabilization` (default β = 0.2, β ≥ ½ the
  unconditional bound) adds β ρ |u·n| to the reversed row's diagonal, `diagnostics.outflow_backflow()` returns
  the census, and `step()` warns once on stderr when reversal appears with β = 0. Purely outgoing
  outlets are byte-identical.
- **Rank-aware:** every per-face application is guarded by `touchesGlobalFace` — a rank imposes a
  face's BC iff its own block touches that global face (before that fix, a partition cutting a
  walled axis split the domain into independent sub-domains, visible only in the pressure). A
  multi-rank inlet profile must be handed to each rank as its own slice; no scatter helper.
- **Ghost fills that are easy to lose:** `step()` fills the cell body-force and `drag_beta` ghosts
  right after `updateProperties()` because `buildRhsVar`/`addDragDiagonal` face-interpolate them;
  skip either and the first inner plane of every block silently carries half the value.
- **OPEN DEFECT:** an immersed solid *cutting* an inflow/outflow face breaks the pressure solve
  (iteration cap, `max|div|` 4e-3, divergence at MG depth ≤ 2). A bed clear of the open faces is
  fine. [`doc/cutcell_openbc_convergence.md`](doc/cutcell_openbc_convergence.md).

## Geometric VoF (`src/vof/`)

Sharp-interface two-phase flow: PLIC planes (SZ2000 / Lehmann–Gekle, MYC normals), Weymouth–Yue
split geometric advection on its own g=3 block (CFL cap 0.25 — WY's 3-D bound; the familiar 0.5 is
the 2-D value), height-function curvature with a PLIC-volumetric paraboloid fallback,
balanced-force CSF surface tension, transport through an SDF solid, static and dynamic contact
angles, open boundaries, a per-bubble block container, and phase change. `"C"` is an ordinary
registered `G=2` cell field, so ρ(C)/μ(C) go through the existing closures and the field accessors
work on it unchanged. Entry points: `enable_vof()`, `set_vof`/`get_vof`, `advect_vof`,
`compute_vof_curvature`, `set_surface_tension`, `enable_vof_momentum`, `set_contact_angle*`,
`set_vof_inflow*`, `enable_vof_blocks*`, `enable_phase_change`; their `*_diagnostics` censuses and
the ablation knobs (`set_csf_mode`, `set_vof_kappa_*`, the `set_phase_change_*` wall) are on
`s.diagnostics`.

- **`enable_vof()` turns on two things**: the exact level-0 pressure operator
  (`diagnostics.set_pressure_exact_residual`, per solver — a two-phase contrast is exactly what
  amplifies the float operator's broken `A·1 = 0`) and the wisp guard `diagnostics.set_vof_wisp_eps`
  at **1e-8** (0 is the
  bit-for-bit V1 predicate and still the standalone `WyAdvector`'s default). Any gate comparing the
  solver against a standalone `WyAdvector` must copy the knob (`IbmSolver::defaultVofWispEps()`).
  **`enable_phase_change` then sets the wisp eps back to 0** — the guard and phase change are
  incompatible on a curved interface; setting it afterwards is the deliberate override.
- **The capillary step** `Δt < sqrt((ρ₁+ρ₂)Δx³/(4πσ))` is `capillary_dt()` and is **enforced by
  `step()`** (`set_capillary_cfl` is the safety factor). At pore scale it binds everywhere, more so
  under refinement (`dt_σ ~ h^{3/2}` against `dt_CFL ~ h`).
- **The WY dilation flag is frozen once per step** — recompute it per sweep and exact conservation
  silently dies (2.3e-15 → 1.5e-2). Momentum consistency freezes its coefficient the same way,
  clamps the flux into WY's admissible interval on the *shifted* volume's own colour, and uses
  plain donor-cell upwind (a MUSCL slope is a density-ratio amplifier). `enable_vof_momentum` moves
  VoF advection to the head of `step()` and needs variable density, staggered layout, explicit
  advection, no solid, no porous continuity; it makes ratios above ~100 usable.
- **Scope, and say it to users:** staggered is the reference; collocated is all-fluid, ratio ≲ 100
  with motion. The block container is all-fluid, staggered-only for its CSF, and **colliding
  markers are outside the rating** (a contacting pair drives through the 2-cell film at ~1.5 eddy
  turnovers, dt-independent; the fix is the parked `vof-w4` branch).

The rung-by-rung record — every work order, gate number and refuted hypothesis — is
`doc/history/vof_workorders{,_v2,_v34,_v5,_v6}.md`.

## Open items

Intermediate-level multigrid repartitioning at scale; coefficient-aware coarsening for high
contrast; double-diagonal operator storage; solid cutting an open face; `vof-w4`.
