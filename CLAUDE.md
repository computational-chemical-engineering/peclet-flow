# CLAUDE.md

Working reference for `peclet-flow` (`peclet.flow`): what the code does, how to build and test it,
and the rules and traps for editing it. Design notes for shipped behaviour are in
[`doc/`](doc/README.md); campaign narrative and superseded design in
[`doc/history/`](doc/history/README.md). **The measured numbers, rationale and long API notes behind
the rules below are in [`doc/CLAUDE_reference.md`](doc/CLAUDE_reference.md)** (cited as "ref §N").
Suite-wide contracts are in `../docs/`.

## What this is

Performance-portable incompressible Navier–Stokes for complex geometry, **in physical units**:
staggered MAC grid, signed-distance-field solids, a cut-cell IBM, and a pressure projection with a
geometric-multigrid Poisson solve. One Kokkos source runs on CUDA, HIP and OpenMP; simulations are
driven from Python, never from a C++ main. `peclet::flow::IbmSolver` (`src/flow_ibm.hpp`) is the
solver; `src/flow_bindings.cpp` exposes `peclet.flow.Solver` (staggered, the reference) and
`peclet.flow.SolverColocated` (cell-centred + ABC approximate projection) through a `GridLayout`
policy. Raw CUDA retired 2026-06 (tag `pre-cuda-retirement`); pore networks are `../pnm`.

## Build

```bash
source ../.venv/bin/activate                  # THE suite venv (../CLAUDE.md "One venv")
CMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" pip install .     # canonical install
cmake -S . -B build_dev -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DPECLET_FLOW_BUILD_TESTS=ON -DPECLET_FLOW_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_dev -j8
export PYTHONPATH=$PWD/build_dev              # -> build_dev/peclet/flow/_flow.*.so
```

Kokkos via `find_package` on `../extern/install/<backend>` (`nvidia-cuda` / `host-openmp` /
`lumi-hip`, built once by `../tools/bootstrap_deps.sh` — a **hard build dependency**); nanobind from
the active interpreter (`SuiteNanobind`); `nvcc` on `PATH` for CUDA. Kokkos 5.x (C++20), CMake 3.24+,
Python 3.10+, `../core`, MPI for the distributed path. `PECLET_FLOW_BUILD_TESTS=ON` registers every
suite; `PECLET_FLOW_MPI=ON` adds `init_mpi`/`rank`/`size` and `flow.mpi_block()` (OFF leaves the
single-rank module byte-identical). Another backend = another prefix and another tree.

- **Force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`.** FindMPI may pick ParaView's `mpiexec` off `PATH`;
  every `*_np4` then silently runs four independent np = 1 jobs.
- **Host builds always compile with `-ffp-contract=off`**; `-DPECLET_FLOW_HOST_ARCH=<arch>` (default
  generic x86-64; `native` dev box, `znver4` Snellius genoa, `znver3` workstation) adds `-march`
  without changing a bit on x86. Neither reaches CUDA/HIP; PyPI wheels stay generic. On aarch64
  contract-off does change bits against the old default (it makes host results ISA-independent).
- **`Solver<Grid>` is compiled ONCE per grid** (QUALITY_PLAN G.8, done 2026-09-11, `61e9f58`,
  `8f79c3b`): explicit instantiations in `src/flow_solver_{staggered,colocated}.cpp`, built into
  `peclet_flow_solver` by `cmake/PecletFlowSolver.cmake`; `flow_ibm.hpp` ends with `extern template`
  and includes the domain headers only in those TUs (`PECLET_FLOW_INSTANTIATING`). Rules: a member
  *template* that a test or the bindings call must be defined in `flow_ibm.hpp` itself (else it does
  not link); a target including `flow_ibm.hpp` links the `peclet_flow_solver` variant matching its
  `PECLET_FLOW_MPI` (never declared in a test); a third grid policy needs a third TU. Ref §1.
- **`ccache` is opt-in and HOST-ONLY** (`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` on a host prefix).
  On CUDA/HIP it structurally defeats Kokkos' launch-compiler routing; CMake rejects the combination
  at configure time — do not re-attempt it. Ref §1.

## Test

```bash
ctest --test-dir build_dev -N                                   # 234 registered, nothing hidden
OMP_NUM_THREADS=8 OMP_PROC_BIND=false ctest --test-dir build_dev --output-on-failure -LE bench
ctest --test-dir build_dev -R '_np[0-9]+$' --output-on-failure   # the distributed suite only
```

234 registered / **231 with `-LE bench`** (counted 2026-10-08 on `66e82e6`): 59 from `tests/kokkos`
(`bench_rbgs`, `vof_timing`, `bench_scalar_cutcell` carry `bench` — instruments, not gates), 136
from `tests/kokkos_mpi` (45 cases at np = 1, 2, 4 + one np = 8 rung), and 39 Python ctests on the
module of that tree (regression/verify, the guards `no_env_knobs`, `no_float_operator_casts`,
`iteration_order`, and 24 `scalar_cutcell_*` gates; the six `_g6_*` take 20–35 min each).

- **Always bound the OpenMP pool** — unbounded on a many-core host is an hour-long trap. On a shared,
  loaded host run `_np*` with `OMP_NUM_THREADS=2` (threads per RANK): at 8 per rank, load ~110 on 48
  cores, `velocitymg_bc_mpi_np2` took 45 min instead of seconds.
- `tests/regression/state_hash.py` prints the SHA-256 of one fixed-seed run per public entry path —
  **the byte gate for any refactor that must not change numerics** (package-F hashes: commit
  messages). `tests/regression/sdflow_regression.py` is the accuracy + iteration regression
  (`--update` re-records; `--solver colocated --scheme ghost` has its own baseline). External
  ground truth: `scripts/validate_zick_homsy_sdflow.py`, plus `scripts/verify_*_sdflow.py`.
  `tests/study/` holds instruments, not gates.
- CI: `ci.yml` (single-rank tree, then MPI np = 1, 2, 4/8 if time allows); `quality.yml` (ruff +
  **blocking** clang-format 18.1.8 on `src/`, `tests/`; six `vof-w4` files excluded); `docs.yml`.

## Layout

All header-only Kokkos C++20 in `namespace peclet::flow`.

- `src/flow_ibm.hpp` — `template <class Grid> class Solver` (`IbmSolver` = `Solver<Staggered>`):
  nested structs, every member DECLARATION with its docstring, the whole state block. Out-of-line
  DEFINITIONS live in thirteen domain headers `src/flow_ibm_{core,project,scene,geometry,hydro,vof,
  phase_change,closures,scalars,scalars_cutcell,bc,mpi,diagnostics}.hpp`, each reopening the
  namespace; declarations and state never move out. Which member lives where: ref §2.
  `src/flow_bindings.cpp` is the nanobind module; `src/flow_solver_*.cpp` the instantiations (also
  of `AndersonAccelerator<Grid>`, `src/anderson_accelerator.hpp`).
- `src/mac_cutcell_mg.hpp` (`CutcellMG`), `src/mac_velocity_mg.hpp` (`VelocityMG`), the `src/mac_*.hpp`
  operators, `src/cut_cell_ibm.hpp` (Robust-Scaled overlay: `poly_*`, K/M/X/Nbc/R, `D_rescale`),
  `src/staggered_advection.hpp`, `src/gauge_exact_gradient.hpp`, `src/ghost_projection.hpp`.
- `src/vof/` — the VoF drivers. Its container-free kernels were promoted to `peclet::core::vof`
  (`src/vof/{plic,curvature,cutcell,wetting}.hpp` are thin includes): **new container-free VoF math
  belongs in `core`**.
- `tests/{kokkos,kokkos_mpi,python,regression,study}`, `scripts/`, `doc/` + `doc/history/`.

## Settled decisions — do not reverse silently

Each was chosen *against* the textbook alternative on measured evidence and has been re-proposed by
mistake. Full entries with quotes and provenance:
[`../docs/decisions/flow.md`](../docs/decisions/flow.md). Reversing one takes a new recorded decision.

- **Collocated pressure coupling is the Almgren–Bell–Colella approximate (MAC) projection — NEVER
  Rhie–Chow.** The residual cell divergence is intrinsic to cell-centred placement; the permeability
  gap lives in the momentum solve, not the projection. Rhie–Chow is not an "upgrade".
- **Collocated pressure and forces go INSIDE the implicit momentum predictor — never a face
  acceleration after the viscous solve (the Basilisk `centered.h` "kick")**: that gives Chorin's
  dt-dependent steady state and an explicit pressure diffusion (−12κ·dt/(ρh²) per step). Guard: the
  gate `collocated_stability_guard`, not a grep. `doc/collocated_varrho_forces.md`; ref §3.
- **The force on a body is the discrete REACTION — `hydro_force_torque_reaction()` — never the
  traction integral** (USER-approved 2026-09-30). `diagnostics.hydro_force_torque_traction()` reads
  0.69–0.73 of it at every resolution — a diagnostic only. The old `hydro_force_torque()` is
  DEPRECATED and warns — never hand it out as "the force". Register: coupling "The public force API
  returns the reaction"; planned fix under Open items; ref §3.
- **The pressure solve is PCG (Krylov), not RB-GS**, for cut-cell IBM.
- **Backward Euler is the default time integrator.** Crank–Nicolson was ported and reverted.
- **Porous beds use ε-weighted (ε-conservative) momentum with a matched projection** (`2d1564a`,
  `doc/porous_drag_scheme.md`). Plain incompressible continuity with ε only in the drag is the *wrong*
  constraint.
- **Masking excludes cut AND solid cells**, never cut cells alone.
- **IBM velocity-MG must never un-scale the residual by `1/D_rescale`.**
- **The ORB must never split the wall-normal axis** in wall-bounded flow.
- **Geometric const-coeff operators + masking are the validated defaults**; Galerkin/CG is opt-in.
- **Momentum advection uses the actual wall velocity field**, not `maskVelocity`'s solid zeros.
- **COLLOCATED momentum advection uses the PROJECTED face field** `uf_/vf_/wf_` — never the
  un-projected ½(u_i+u_j) (ablation `diagnostics.set_uf_advection(False)` only). Implicit FOU and the
  explicit (SOU−FOU) correction MUST read the same field: one predicate, `ufAdvVelocity()`
  (`doc/uf_advection.md`).
- **Operator storage is double by default** (`PECLET_FLOW_OPERATOR_DOUBLE=ON` since 2026-09-11,
  `dc6ae78`). Float silently breaks A·1 = 0 at high contrast — opting out costs correctness on dense
  beds. The double-*diagonal* fallback is different and stays retired (65× worse on divergence).
- **The rotational (Timmermans) pressure update must be restored, not the non-rotational Goda form.**
- **`set_ghost_projection(True)` must be called before `set_solid`** — call order is load-bearing.
- **Distributed cut-cell MG coarse levels must be nested**, never independently re-decomposed.
- **Fresh (newly-uncovered) cells are seeded with the local wall velocity**, not the stale interior.
- **Closed dead ends, do not re-attempt:** mode-10 quadrature, the Seo–Mittal pressure-only split,
  the double-diagonal fallback (measurably worse, not merely unnecessary). "Ghost as production",
  listed dead on 2026-08-20, was **reversed on 2026-08-25**: `'ghost'` is the collocated AUTO
  default (flow `c672014`, amr/core `7472306`).
- **Drag normalisation (settled by study A1, 2026-10):** our K is the TOTAL force per particle
  (F = f V / N_p, including the V_p∇p part) at the SUPERFICIAL velocity — the Zick & Homsy
  convention. The van der Hoef/Beetstra/Tang F_D (drag only, superficial) = (1−φ)K; van Wachem's
  F̃_D is also (1−φ)K, but because it is the total force at the interstitial velocity. The dilute
  limit cannot tell them apart; φ = 0.5 differs by 2×. Source: `~/Codes/peclet-study-A1-drag-audit`
  `work/RERUN_PLAN.md` §10 and `work/PORT_NOTES.md` F5. Still OPEN there: our drag is ~9 % below
  van Wachem's on the same bed.

## Conventions and hard rules

- **SDF sign:** negative inside solid, positive in fluid.
- **Indexing:** `I = x + y*nx + z*nx*ny` (x fastest); Python arrays `order='F'`, shape `(nx, ny, nz)`.
  **Staggered:** u at (i+½,j,k), v at (i,j+½,k), w at (i,j,k+½), p at cell centres.
- **Kokkos device code lives in `.hpp` compiled as C++** — never a `.cu`. Loops use `MDRange3<Exec>` /
  `MDRange2<Exec>` (`src/policy.hpp`), lambda `(x, y, z)`. **Iteration order follows storage — x
  fastest on every backend, never Kokkos' default** (z-fastest on OpenMP/Serial;
  `../docs/CONVENTIONS.md` §1). A bare `Kokkos::Rank<` / `MDRangePolicy<` outside `policy.hpp` fails
  ctest `iteration_order`.
- **No environment variable changes a result** (QUALITY_PLAN D3, `ad917b1`, 2026-09-08). Every
  numerics-changing read became a per-solver setter (old unset behaviour = default) or was deleted;
  the old-name → setter table is in the `ad917b1` message. Only debug *printing* reads remain
  (`PECLET_FLOW_MG_DEBUG`, `…_GP_DEBUG`, `…_AGMG_DEBUG`); ctest `no_env_knobs` fails on a new one.
  Do not add one, and do not document a workflow that sets one.
- **Two API tiers (QUALITY_PLAN D2).** `Solver` / `SolverColocated` carry the PUBLIC surface (set
  up, run, read out); inspection, profiling, ablation, tuning beyond driver selection and the MPI
  internals live on `s.diagnostics`. Integer codes are strings; an on/off pair is one setter with a
  leading `enabled` bool. Ref §4.
- **Call order is enforced where a wrong order gives a wrong result.** Settings folded into the
  operators at geometry build — `set_domain_bc` (a TYPE change), `set_domain_bc_profile`,
  `diagnostics.set_aperture_order`, `set_exact_crossings`, `set_openness_override`,
  `set_fluid_only_constraint`, `set_ghost_projection` / `set_collocated_scheme('ghost')` — raise after
  `set_solid`/`set_pressure_geometry`/`set_solid_from_scene`; `set_decomposition` and
  `diagnostics.set_comm_avoiding` raise after `init_mpi`. **`init_mpi` raises after the geometry**
  (built first, every rank ran its own block-periodic pressure solve, silently; gate
  `test_cellforce_mpi`): the order is `Solver(*size)` → `init_mpi` → geometry. `set_rho`/`set_mu`/
  `set_dt` may come any time (under a physical domain the FIRST `set_rho` and FIRST `set_dt` pin the
  reference scales). `set_decomposition(levels, max_imbalance)` and `flow.mpi_block(...)` must get the
  **same** values. Select the pressure driver **last**: `set_property_model("rho", …)` re-selects
  Chebyshev. Ref §4.

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

Everything in and out is in one consistent unit system of the caller's choosing — including the
scalar/energy surface since 2026-10-02 (full list ref §5); a transported scalar is **never
rescaled**. The two FIELD setters (`set_mass_flux*`, `set_divergence_source`) raise under an extent
until `set_rho`/`set_dt` have run; the operator-flux `mdot` without `set_phase_change_energy` raises
under an extent. Under MPI the constructor takes **this rank's** block, with `global_cells` and the
global `extent`/`origin`. `extent=None` keeps cell units and is bit-identical to the pre-2026-09 code.

- **The trap:** the solver computes on the unit lattice (`Solver::UnitScales`; `hRef = min h`,
  `rhoRef`/`tRef` = first `set_rho`/`set_dt`). **The raw registry — `get_field`/`set_field`,
  `diagnostics.field_view`/`exchange_field` — hands out INTERNAL arrays**, unlike `get_u`/`get_p`. A
  driver writing `force_x`/`drag_beta` (CFD-DEM does) or reading `mdot`/`pc_source`/`div_source` must
  convert with `s.unit_scales`.
- **Anisotropic cells** work on both solvers, single phase and VoF (`doc/anisotropic_*.md`); CFD-DEM
  refuses an anisotropic box, and `scripts/check_decomposition.py` models only isotropic coarsening.

## Pressure solve

Geometric multigrid (`CutcellMG`), RB-GS smoother, **rediscretized** cut-cell coarse operators. Four
outer drivers, **one per solver**; the three Krylov ones are mutually exclusive, last set wins:

| driver | select with | use |
|---|---|---|
| standalone V-cycle | default | multi-rank default; `set_pressure_multigrid(True, levels=1)` = pure RB-GS |
| MG-PCG | `set_pressure_pcg(True, iters, rtol)` | single-rank default (auto-enabled at np = 1) |
| flexible MG-CG | `set_pressure_fcg(True, iters, rtol)` | tolerant of a non-symmetric preconditioner — try it when PCG caps |
| Chebyshev | `set_pressure_chebyshev(True, iters, rtol)` | no global dots per iteration; the variable-density / porous default |

`set_pressure_pcg(False)` **raises** (the terminal fallback cannot be deselected; name the driver
you want). Detail and numbers for every bullet: ref §6.

- **High coefficient CONTRAST makes the V-cycle preconditioner indefinite**: both CG drivers cap
  above density ratio ~10³ (the coarsening, not float storage); only Chebyshev is healthy there.
- **Operator storage precision** (`MReal`; double by default, +12 % step time) covers the pressure
  hierarchy, momentum stencil, K/M/X/Nbc/R/D_rescale and the ghost overlay. Float = **invalid, not
  degraded** on a high-contrast bed (`../docs/SCALING_ISSUES.md` #1). Ctest `no_float_operator_casts`
  fails on a new `(float)` cast outside `// PRECISION-EXEMPT: <reason>`; `GpOverlay`'s two exempt
  casts are a real cross-repo gap (core's `gpFillRow` is float-hardcoded), not dead code.
- `set_pressure_bottom("auto")` (default) agglomerates the coarsest level into a global operator solved
  exactly when it exceeds `set_pressure_bottom_extent` (4) cells on an axis; porous and variable-ρ
  rebuild it every step — avoid `auto` with a badly-factored grid there. Engines: host GraphAMG, or on
  an eligible single-rank GPU the device bottom (`src/mg_bottom_direct.hpp`, design
  `doc/vof_step_performance_design.md` §13); A/B with `diagnostics.set_pressure_bottom_solver`.
- **Depth follows the factors of two, per axis**: an odd dimension never coarsens; under MPI only if
  every rank's block is even. **Telescoping is the default** (`set_pressure_telescope`); the in-place
  requirement on intermediate levels is the top open item at scale
  (`../docs/DECOMPOSITION_AND_MULTIGRID.md` §2.8).
- **The stage machinery is core's** (`chooseStageTarget`, `makeStageComm`, `RedistributeTopology`;
  proved byte-identical): **do not re-inline any of it**. `Telescope`, the scatter's ADD and the WO-R2
  outflow ghost-plane gather stay here.
- **A weighted level-0 decomposition gets Repartition stages** (gate `test_telescope_mpi` D), not a
  collapse onto one rank. `nLevels = 1` and the GraphAMG bottom do not escape it — do not
  re-recommend them.
- **`rebalance_by_weights` builds the ALIGNED weighted ORB and RETURNS `2^a`; a co-decomposing code
  must use the same weights AND that alignment** (dem `migrate_to_weights(w, align=…)`, as
  `CfdDem.rebalance()` does). Forecast: `flow.predict_hierarchy(..., weights=w)`.
- **`redistribute` must carry all cross-step state and seed derived state AFTER the scatter**
  (gate `test_porous_redistribute_mpi`; a size-preserving rebalance hides a miss).
- Decomposition: `set_decomposition(0)` (default) aligned ORB; `set_decomposition(L>=2)` coarse-first
  (blocks nest for the full depth). Check a combination with `scripts/check_decomposition.py` first.

## Velocity solve

Per-component backward-Euler diffusion, RB-GS or a V-cycle, both distributed; operator modes
IBM-periodic (staircase), all-fluid domain-BC, and **mixed**. `bcStencilPath()` and `implicitAdv()`
must agree with the solver in use (unenforced; a mismatch silently made advection explicit).

- `set_velocity_residual_tolerance`: default follows the pressure driver's rtol. **No early return
  for a converged warm start** (it drifts the hydrostatic acid test by 1e-8).
- **The solver is chosen by the PHYSICS, not the configuration** (2026-09-15): `kappa = 1 +
  4·dt·mu·(w_x+w_y+w_z)/rho ≥ 13` (D ≳ 1) → the 3-level V-cycle, else RB-GS. A **correctness**
  threshold (above D≈4 RB-GS stops meeting its tolerance); geometry-independent. Decided at the
  **first `step()`** (`set_solid` resets it). `set_velocity_multigrid_auto(65536, 1 << 23)` = the
  1.0.0 rule, `(0)` disables. Rationale, numbers, exceptions: `doc/velocity_mg_plan.md` (last section).

## Collocated solver (`SolverColocated`)

Read [`doc/collocated_invisible_subspace.md`](doc/collocated_invisible_subspace.md) (mechanism)
before touching this path; `doc/collocated_paper_plan.md` tracks results,
`doc/fluid_only_constraint_plan.md` is the production plan. **The default is AUTO = `"ghost"`** (the
fluid-only scheme, since 2026-08-25). AUTO falls back to `"gauge-exact"` with a stderr notice where
ghost v1 is unsupported: porous, variable-ρ, domain BCs, Chebyshev, exact-crossing / openness
overrides, a fluid-only instrument mode, and V8 (variable density / CSF enabled after the geometry).
Any explicit `set_collocated_scheme` / `diagnostics.set_face_interp` / `set_ghost_projection` /
`set_fluid_only_constraint` disables AUTO; tests and baselines pin schemes explicitly.

- Schemes: `"ghost"` (family-free, no stabilizer; BiCGStab ~2.3–2.7× the pressure stage, ~1.6 KB/cell
  overlay), `"gauge-exact"` (converges, carries an attractor family), `"plain"` (first order, legacy),
  `"embed"` (the Basilisk embed.h port — the accuracy-ceiling candidate). Internals: ref §7.
- MPI validated np = 1, 2, 4 (**np ≥ 16 unresolved**). The mixed `(matrix_order=1, rhs_order=2)` ghost
  mode is **do-not-use**.
- Variable density + surface tension (V8) run **all-fluid** through the mass-adjoint ABC pair
  (`doc/collocated_varrho_forces.md`); immersed solids on that path are the next package
  (`doc/collocated_multiphase_solids_plan.md`). The balanced-force projection defaults ON there.

## Domain boundary conditions

`set_domain_bc(face, type, velocity=…)`, faces `'-x' … '+z'`, types `'periodic'` (default), `'wall'`,
`'inflow'`, `'outflow'`, `'slip'` (mirrors the SDF ghost band; other non-periodic faces get the
constant normal extension). `set_domain_bc_profile(face, profile[Nb,Nc,3])` prescribes an inlet. Only
a TYPE change must precede the geometry; a VALUE update is allowed any time. No immersed solid → use
`set_pressure_geometry(all_fluid_sdf)`. Detail: ref §8 and `doc/cutcell_openbc_convergence.md`.

- **Outlet reversal** is the one conditionally-stable regime: `set_backflow_stabilization` (β = 0.2
  default; β ≥ ½ unconditional), census `diagnostics.outflow_backflow()`, `step()` warns once at β = 0.
- **Rank-aware:** every per-face application is guarded by `touchesGlobalFace`; a multi-rank inlet
  profile goes to each rank as its own slice (no scatter helper).
- `step()` fills the cell body-force and `drag_beta` ghosts right after `updateProperties()` — skip
  either and the first inner plane of every block silently carries half.
- **A solid cutting an open face** (`e144a00`, `test_openbc_solid{,_mpi}`, both grids): SDF ghosts are
  extended, not wrapped; outlet rows carry the aperture; fluid SEALED against an inlet is rejected.
- **The two ends of an axis are ASYMMETRIC and the halo knows nothing about domain BCs**: the high
  boundary plane comes back from an exchange with the OPPOSITE boundary's value. The openness
  (`buildOpennessHighFace`) and the outflow face velocity (`fillVelGhostsTo(..., doOutflow=false)`)
  must both handle it — **fixing either alone makes things worse**. Gate: `test_vof_bc_mpi`'s
  composed budget (`../docs/SCALING_ISSUES.md` #8).
- `max_open_divergence()` measures zero-gradient extrapolation and does NOT decay at a partly blocked
  outlet; `max_open_divergence_projected()` is the residual the projection solved.

## Geometric VoF (`src/vof/`)

PLIC (SZ2000 / Lehmann–Gekle, MYC normals), Weymouth–Yue split advection on its own g=3 block (**CFL
cap 0.25**, WY's 3-D bound), height-function curvature, balanced-force CSF, transport through an SDF
solid, contact angles, open boundaries, a per-bubble block container, phase change. `"C"` is an
ordinary `G=2` registered cell field. Entry points: `enable_vof()`, `set_vof`/`get_vof`,
`advect_vof`, `compute_vof_curvature`, `set_surface_tension`, `enable_vof_momentum`,
`set_contact_angle*`, `set_vof_inflow*`, `enable_vof_blocks*`, `enable_phase_change`; censuses and
ablations on `s.diagnostics`. Ref §9; rung record `doc/history/vof_workorders*.md`.

- **`enable_vof()` turns on** the exact level-0 pressure operator and the wisp guard
  `diagnostics.set_vof_wisp_eps` at 1e-8. A gate comparing against a standalone `WyAdvector` must
  copy the knob (`IbmSolver::defaultVofWispEps()`). **`enable_phase_change` sets the wisp eps back to
  0** (incompatible on a curved interface); setting it afterwards is the deliberate override.
- **The capillary step** `capillary_dt()` is **enforced by `step()`** (`set_capillary_cfl`).
- **The WY dilation flag is frozen once per step** (per sweep, exact conservation silently dies); so
  is the momentum-consistency coefficient, which uses donor-cell upwind (MUSCL amplifies density
  ratio). `enable_vof_momentum`: variable ρ, staggered, explicit advection, no solid, not porous.
- **Scope — say it to users:** staggered is the reference; collocated is all-fluid, ratio ~10 with
  motion (exact at rest at 1000). Colliding markers are rated since 2026-09-25
  (`doc/vof_overlap_design.md`). **Never assemble the block force from the union colour** (it
  re-creates numerical coalescence); the parked `vof-w4` branch is superseded.

## Cut-cell scalar transport (`add_scalar(..., cutcell=True)`)

Opt-in per scalar, landed on main 2026-10-04. Contract
[`doc/scalar_ibm_design.md`](doc/scalar_ibm_design.md) (Amendments A1–A3, open questions §13); numbers and rulings (D-WO*) in `doc/scalar_ibm_log.md`; state
`doc/scalar_ibm_STATE.md`; the full user-facing summary (method, setters, refusals, traps) is ref
§10. Legacy scalars are untouched (dispatch on the first line of the `advanceScalars` loop; the 12
state hashes are the gate). Second order for every BC, Robin number and conjugate contrast (G1–G8).
The `cutcell=` spelling and making it the default are pending Frank (§13 Q5/Q7).

- **Refused** (RuntimeError naming the remedy): the collocated `'ghost'` scheme — the
  `SolverColocated` default; collocated open faces; a moving fluid without the cut-cell projection;
  porous continuity; moving scenes; and the rest listed in ref §10.
- **Traps:** Koren bounded only at Courant ≤ ½; steady backflow through `'outflow'` enters at c_i —
  give the fluid an `'inflow'` face; unknowns follow the snapped apertures, not κ; the iteration count
  measures the MG table — **gate and benchmark on boxes with factors of two**; MPI parity gated at
  rtol 1e-13, Koren at 1e-7; steady advection is FOU, Pe_h ≤ 10; contacts/sub-cell gaps unresolved.
- **Do not reverse** (register `../docs/decisions/flow.md`, cut-cell scalar entries of 2026-10-04): a
  short or κ-dependent probe; unit storage; the series-resistance / GFM wall flux; coarse wall terms at
  the level's own probe distance or Galerkin RAP; a symmetric surrogate for steady advection; a
  post-solve mass fix-up; Peters' directional one-field conjugate scheme.

## Steady marches (`march_to_steady`)

`peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, max_steps=5000, accelerate=True, …)` returns
a frozen `MarchResult` (`converged`, `steps`, `reason` ∈ {"certified", "max_steps", "diverged"}, …).
Design and numbers: [`doc/steady_acceleration.md`](doc/steady_acceleration.md); full API notes ref §11.
`converged=True` certifies stationarity at this dt, not that a plain march would reach that state.

- The stop instrument is the study's; `accelerate=False` reproduces that march step for step (ctest
  `march_to_steady`). Under MPI `monitor()` must return the same value on every rank.
- Type-II Anderson (`src/anderson_accelerator.hpp`, control `packaging/flow_steady.py`), certified
  on PLAIN steps; deliberately **no instability guard** (a Ritz radius is no stability test here).
- **Scope:** staggered, or collocated `'ghost'`; the rest is refused by name — pass
  `accelerate=False`. Memory 416 B per inner cell at window 5.
- **Checkpoint with `get_field`/`set_field` of u, v, w, p** — not `set_state` (loses P).

## Open items

**Cut-cell scalar transport — pending FRANK** (evidence `doc/scalar_ibm_STATE.md`):
1. **The conjugate contact model** (§13 Q4): particles in contact do not converge at R/h 8–32 (G13
   conjugate row INFO). Recommended: a body-aware geometry record + a contact conductance.
2. **Naming** (`doc/scalar_ibm_naming_rows.md`): `set_scalar_tolerance` → `set_scalar_residual_tolerance`,
   `maxit` → `max_iter`, the `cutcell=` spelling (Q5), when cut-cell becomes the default (Q7).
3. **Branch `origin/uf-outlet-diag`** (pushed, NOT merged; `doc/uf_outlet_fix.md` on that branch): fixes
   the legacy scalars at high-side open faces on both grids; needs Frank's OK on `c002c8e` and
   `af88101`; changes the state_hash `scalar` case. Until it lands, `advanceScalars`' plain velocity
   ghost fill wraps the high-side open plane — and it cannot simply be skipped (a constant drifts
   4.1e-2 in 6 steps, D-WOR-7).
4. Also: collocated open faces for cut-cell scalars; the test-battery cadence; tagging core; the
   ScalarMG agglomerated exact bottom (follow-up WO — only a warning exists today); host advance ÷
   projection ≈ 1.05.

**Solver:** intermediate-level MG repartitioning at scale (Repartition fires only for a weighted
`dec0`); coefficient-aware coarsening for high contrast; collocated np ≥ 16; the −9 % drag vs van
Wachem (above); `vof-w4` (superseded, but still holds `quality.yml`'s six clang-format exclusions);
**the collocated face field `uf_` is re-seeded from the cell velocity, not migrated, by
`redistribute`** — a rebalanced collocated run with advection drifts 5.0e-8 in u at np = 1 (so
`balanced_force_mpi`'s rebalance case is Stokes).

**PLANNED — a consistent wall traction** (2026-09-30): the wall shear from the momentum operator's
Robust-Scaled wall reconstruction instead of the central difference in `hydroForceTorque()`
(`src/flow_ibm_hydro.hpp`), so traction → reaction under refinement (M3 paper, `~/Codes/peclet-papers`
PLAN.md D8). Dead end: the one-sided difference over θ (17× too large). Gate and detail: ref §12.
