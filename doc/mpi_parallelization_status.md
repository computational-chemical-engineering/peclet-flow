# MPI parallelization of cfd-gpu — status

Living tracker for the step-wise MPI parallelization of the solver on the shared `transport-core`
library (`../transport-core`). Each step is built and **verified against the serial solver's own
discretization** before the next, and committed separately so any step is revertible.

## Strategy

Distribute the global MAC cell grid into rank-owned blocks (transport-core orthogonal recursive
bisection). Replace the solver's in-kernel periodic wrapping (`get_idx`) with **extended local blocks**
(inner cells + a ghost layer) whose ghosts are filled by an **asynchronous halo exchange**
(`MacGridHalo`, host-staged GPU exchange). Each distributed operator is validated cell-for-cell
against the serial full-grid result (computed with cfd's *own* `get_idx`) before it is threaded into
`step()`.

Build/run the MPI tests (opt-in; default `pnm_backend` build untouched):
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/build/_deps/pybind11-src
cmake --build build_mpi -j
ctest --test-dir build_mpi --output-on-failure     # *_np{1,2,4}, real multi-rank
```
**Important:** force `-DMPIEXEC_EXECUTABLE=/usr/bin/mpirun`. FindMPI may otherwise pick a different
`mpiexec` on `PATH` (e.g. ParaView's bundled one), which is incompatible with the system-OpenMPI-linked
binaries and silently launches the ranks as MPI *singletons* (so `*_np4` would run as 4×np=1).

## Steps

### Step 1 — halo adapter + diffusion-width validation ✅ verified
- `src/mac_halo.cuh` (`MacGridHalo`): decompose global res, exchange a ghost layer for a `double`
  cell-field on the extended local block.
- `tests/test_mac_halo.cu`: distributed width-1 (7-point) periodic Laplacian == serial `get_idx`
  Laplacian, np=1,2,4.

### Step 2 — arbitrary ghost width + advection-width validation ✅ verified
- `MacGridHalo::init` takes a ghost width. cfd's Koren TVD advection flux reads `phi_LL..phi_RR`
  (reach ±2), so the solver needs **ghost width 2** for advection (1 for diffusion/pressure).
- `tests/test_mac_halo.cu`: distributed separable ±1..R stencils == serial for **R=1 and R=2**,
  np=1,2,4.

### Step 3 — distributed explicit advection–diffusion time loop ✅ verified
- `tests/test_advdiff_mpi.cu`: explicit scalar transport with the **Koren TVD** advection limiter
  (the same reconstruction cfd uses; reach ±2) + central diffusion, constant velocity, periodic.
  Serial (full grid, `get_idx`) and distributed (extended block, halo exchange each step) share one
  `cell_update` device function and match cell-for-cell over 25 steps, np=1,2,4.
- Establishes the per-step **exchange → update-inner** pattern the distributed solver will use, and
  confirms the width-2 halo stays correct on every block boundary across a multi-step time loop.

### Step 4 — distributed implicit diffusion (Red-Black Gauss-Seidel) ✅ verified
- `tests/test_gs_implicit_mpi.cu`: solve `(I - beta*Lap) phi = b` with RB-GS, **halo exchange after
  each colour sweep**. Red/black uses **global parity** `(gx+gy+gz)&1` so colouring is consistent
  across block boundaries. Distributed matches serial RB-GS cell-for-cell over 30 iterations,
  np=1,2,4. This is the iterative implicit-solver pattern the momentum and pressure solves use.

### Step 5 — distributed pressure projection (Chorin) ✅ verified
- `tests/test_projection_mpi.cu`: the full incompressible update on a staggered MAC grid — exchange
  (u,v,w) → cell-centred divergence → Poisson solve for phi (RB-GS, exchange between sweeps) →
  subtract grad(phi). Distributed matches serial **cell-for-cell** for u,v,w, np=1,2,4. Composes
  multi-field exchange + staggered operators + the iterative solve — the canonical incompressible
  operation, distributed.

## Summary: all core distributed patterns are verified

The halo (widths 1 & 2), explicit advection–diffusion (Koren TVD), implicit Red-Black Gauss-Seidel
(exchange between sweeps), and the full staggered Chorin projection all reproduce the serial result
cell-for-cell across np=1,2,4. **12/12 MPI ctests pass.** The mechanism for distributing cfd's solver
is proven on cfd's own conventions and field layout.

## Integration into the solver: chosen approach

`step()` is a monolithic outer Picard loop (per-component fused advection+implicit-diffusion stencil
build → RB-GS/multigrid delta solve → correction; then pressure RHS → RB-GS/multigrid Poisson →
projection → pressure pin), all in a **single-process pybind module** with several **global
reductions** (`max_abs`, `remove_mean`, pressure pin) and a **multigrid** hierarchy.

A literal in-place rewrite (extended-block allocation + MPI-ifying every reduction + distributing
multigrid restriction/prolongation across blocks) is a multi-week effort that would leave the
production solver broken mid-way — not compatible with "carefully, commit and test in between". So the
solver is parallelised by **assembling the validated building blocks** (Steps 1–5) into a complete
distributed solver that runs under `mpirun`, reuses cfd's discretisation and the shared
decomposition+halo, and is validated both **cell-for-cell vs a serial reference** and against an
**analytic solution**. The production `pnm_backend` build stays untouched. Full in-place threading of
`cfd_solver.cu` (incl. multigrid + global-reduction MPI collectives) remains the longer-term path on
top of this proven foundation.

### Step 6 — distributed staggered incompressible solver (unsteady Stokes) ✅ verified
- `tests/test_stokes_mpi.cu`: a full timestep = per-component **implicit diffusion** (backward Euler,
  RB-GS with exchange between sweeps) + **Chorin projection** (Step 5), staggered MAC, periodic.
  Validated (a) distributed == serial **cell-for-cell** over multiple steps, np=1,2,4; (b) **physical**
  — initialised with the 2D Taylor–Green vortex (discretely divergence-free on the MAC grid, so
  projection is an exact no-op), the solver reproduces the analytic backward-Euler decay rate to
  <0.1%. This is a correct, MPI-parallel incompressible-flow solver assembled from the verified pieces.

### Step 7 — distributed flow around an SDF-described solid ✅ verified
- `tests/test_stokes_solid_mpi.cu`: the Step 6 solver + a static solid (sphere, SDF < 0 inside)
  handled by a no-slip immersed boundary (velocity zeroed on faces inside the solid each step,
  applied last). Validated: (a) distributed == serial **cell-for-cell** over 20 steps, np=1,2,4 — the
  decomposition + halo are correct with a solid present; (b) **exact no-slip** (velocity identically
  zero in the solid). This is a complete MPI-parallel incompressible solver handling SDF solids.

### Step 8 — distributed Poiseuille channel flow (analytic validation) ✅ verified
- `tests/test_poiseuille_mpi.cu`: body-force-driven flow between two masked no-slip walls (hard
  Dirichlet enforced after every Gauss-Seidel sweep, ghosts masked by wrapped global-y), periodic in
  x,z. Validated: (a) distributed == serial **cell-for-cell**, np=1,2,4; (b) **physical** — the steady
  interior momentum balance `nu*Lap_y(u)+g` is ~5e-9, and the peak velocity matches the analytic
  parabolic profile `g*W^2/(8*nu)` to 0.16%. Mirrors cfd's own `verify_poiseuille.py`.

### Step 9 — reusable `DistributedStokes` solver component ✅ verified
- `src/distributed_stokes.cuh`: consolidates the validated kernels into a `dstokes::DistributedStokes`
  class — fields + `MacGridHalo` + a `step(n_diff, n_pois)` doing per-component implicit diffusion +
  Chorin projection, with `set_body_force()` and `set_solid()` (per-cell no-slip mask). A clean,
  reusable API instead of loose test code.
- `tests/test_distributed_stokes.cu`: drives the class through its public API to reproduce both the
  Taylor–Green decay (rel_err ~2e-15) and the Poiseuille profile (peak to 0.16%, residual ~5e-9),
  np=1,2,4.

### Step 10 — distributed staggered momentum advection operator ✅ verified
- `tests/test_advection_mpi.cu`: replicates cfd's exact nonlinear advection — `get_advection_velocity`
  (2-point staggered interpolation of the advecting velocity) + Koren TVD flux
  (`psi=max(0,min(2r,min((1+2r)/3,2)))`, sign-upwinded), conservative form `A = sum_dir(F+ - F-)`,
  reach ±2 (ghost width 2). A templated field accessor lets the same operator serve the full grid
  (wrapping) and the local extended block. Validated: (a) distributed == serial **cell-for-cell**,
  np=1,2,4; (b) **conservation** — global sum of A is ~1e-14 (flux form telescopes), so advection
  conserves total momentum. This is the operator that upgrades the Stokes solver to full Navier–Stokes.

### Step 11 — full distributed Navier–Stokes ✅ verified
- `src/staggered_advection.cuh`: the advection operator factored into a shared header (single source
  for the solver and the Step-10 test). `DistributedStokes::set_advection(true)` folds explicit Koren
  advection into the momentum RHS (`b = u - dt*A + dt*f`, all components from the n-level velocity);
  the class now uses ghost width 2 to cover the advection reach.
- `tests/test_navier_stokes_mpi.cu`: the distributed solver (advection on) matches an independent
  serial full-grid integration of the identical scheme **cell-for-cell** over 10 steps, np=1,2,4 —
  the rigorous distribution check for the full nonlinear solver.

### Step 12 (capstone) — distributed Navier–Stokes flow around an SDF solid ✅ verified
- `tests/test_ns_solid_mpi.cu`: the `DistributedStokes` solver with **both** nonlinear advection and
  an SDF solid (sphere, no-slip by masking), body-force-driven flow past the sphere. Matches an
  independent serial full-grid integration of the identical scheme **cell-for-cell** over 8 steps,
  np=1,2,4 — the complete capability (decomposition + async halo + advection + projection + solids).

### Step 13 — gather-to-root + VTI output ✅ verified
- `DistributedStokes::gather_to_root()` assembles the global field from the rank-owned blocks onto
  rank 0 (block geometry known to every rank from the replicated decomposition).
- `tests/test_gather_vti_mpi.cu`: gathers the TGV u-field and checks it matches the analytic decayed
  pattern cell-for-cell (maxerr ~1e-15 — a wrong assembly would scramble it), then writes it via
  `tpx::geom::writeVti` and reads it back bit-exactly. Completes the output pipeline and ties the
  distributed solver to transport-core's geometry/field I/O. np=1,2,4.

### Step 14 — distributed global reductions (`max_abs`, `remove_mean`) ✅ verified
- `src/mac_reductions.cuh`: reduce a `double` cell-field over each rank's **inner** (owned) cells —
  skipping the ghost layer that duplicates neighbours — then `MPI_Allreduce`. `mac_reduce` returns the
  global sum and global max|.|; `mac_max_abs` is the CFL/convergence max; `mac_remove_mean` subtracts
  the global mean (sum / full-grid cell count) over the **whole** extended block, so every rank
  subtracts the same constant and the ghost layer stays consistent with neighbours' inner cells — no
  halo exchange needed afterward. (`atomicMaxDouble` via `atomicCAS`; inner-cell reduce kernel.)
- `tests/test_reductions_mpi.cu`: the global field is a deterministic hash over global coords, so the
  host computes the EXACT global sum / max|.| / mean by sweeping all global cells. Distributed result
  matches **cell-for-cell**: max|.| exact (order-independent), `remove_mean` residual sum ~5e-16
  relative, max|f−mean| exact, sum to ~1e-16 relative. np=1,2,4.
- These are the prerequisite for the distributed **multigrid** (mean removal between V-cycles for the
  pure-Neumann pressure) and for any CFL/convergence gate in the in-place solver. `DistributedStokes`
  itself needs no mean removal on its periodic all-fluid Poisson (consistent iteration from φ=0).

### Step 15 — distributed geometric multigrid (periodic Poisson) ✅ verified
- `src/mac_multigrid.cuh`: `DistributedPoissonMG` — a V-cycle on a hierarchy of `MacGridHalo` levels
  (level L = global grid at `res>>L`). Because transport-core's ORB cuts power-of-two grids at
  midpoints, **each rank owns the same spatial sub-box halved at every level** (asserted in `init` via
  the inner-block start `og+ghost`), so **restriction (8:1 average) and prolongation (trilinear) are
  local within a rank's block** — only the per-level ghost exchange and the per-level mean removal
  (Step 14 reductions) cross ranks. Operator: constant-coefficient periodic Laplacian (`A=-Lap`,
  spacing doubling per level); RB-GS smoother coloured by **global** parity (halo exchange between
  colours); residual; restrict; recurse; prolong (needs the coarse ghost layer first); post-smooth;
  remove mean. All `double`.
- `tests/test_multigrid_mpi.cu`: runs the identical V-cycle as a **serial full-grid reference** (wrap
  indexing) and the distributed solver, and compares cell-for-cell. **max|d| ~1e-15** (machine
  precision — halo vs in-kernel wrap give identical neighbours; the only divergence is the mean
  reduction order) AND a convergence gate (residual max 5.0e-1 → 1.9e-4 over 3 V-cycles, **identical
  across np**, proving the solver is real, not a no-op). 64³, 4 levels, np=1,2,4.
- This is the core of the distributed pressure solve. Remaining for the full in-place solver: the
  **variable-coefficient / cut-cell** operator on level 0 (the `A_C..A_T` from `frac_u/v/w + sdf`, vs
  the constant-coefficient operator here) and wiring the V-cycle into `DistributedStokes::step`.

### Step 16 — variable-coefficient fine-level operator for the multigrid ✅ verified
- `mac_multigrid.cuh` gains a **variable-coefficient** fine level: per-cell 7-point coefficients
  `A_C..A_T` assembled (`mg_build_op_k`) from staggered **face transmissibilities** — `ox[i]` = openness
  of the −x face of cell i (== +x face of cell i−1), so a face is shared and every rank derives the
  same coefficient → the operator is **symmetric across block boundaries**. Mirrors the serial
  `compute_pressure_operator_kernel` (`A_E = -frac_r·inv_dx2`, …); `mg_smooth_var_k` / `mg_residual_var_k`
  read the per-cell coefficients with a halo-exchanged neighbour `x` (and an `A_C` guard skips fully
  closed solid cells). `setFineVariableOperator(ox,oy,oz, idx2,idy2,idz2)` installs it on level 0;
  **coarse levels stay constant-coefficient** (mirrors the serial `use_periodic_operator = level>0`).
- `tests/test_multigrid_var_mpi.cu`: the fine operator is built from an analytic **periodic sphere SDF**
  (face openness in [0.1, 1] — a 10× coefficient ratio; the production path swaps in cfd's
  gradient-normalised fluid fraction with the same assembly). A serial full-grid reference V-cycle runs
  the identical operations (wrap-indexed twins). Distributed matches **cell-for-cell ~1e-15** at
  np=1,2,4, with residual reduction 5.0e-1 → 3.2e-2 over 4 V-cycles, **identical across np**.
- With this the distributed pressure-solve machinery is complete end to end: variable fine operator +
  constant-coefficient coarse hierarchy + block-local transfers + global mean removal. Remaining is
  computing the openness from the real SDF/`frac_u/v/w` on the extended block (cfd's fraction kernel,
  ghost width 2 for its stencil) and threading the V-cycle into `DistributedStokes::step`.

### Step 17 — multigrid V-cycle wired into `DistributedStokes::step` ✅ verified
- `DistributedStokes::set_pressure_multigrid(on, n_levels, pre, post, bottom)` makes the projection's
  pressure Poisson use the distributed geometric multigrid (Step 15) instead of the single-level RB-GS.
  Both solve the **same** periodic constant-coefficient Laplacian, so the V-cycle is a drop-in that
  converges far faster per unit work; `step()`'s `n_pois` then counts V-cycles. The MG owns its own
  level hierarchy on the **same ORB decomposition and ghost width (2)** as the solver, so MG level-0
  blocks share the layout — `div` is copied straight into the V-cycle RHS (`b = -div`) and the solution
  `phi` straight back, no remap. Built lazily on the first step; the single-level path stays the default
  (every prior cell-for-cell test is untouched).
- `tests/test_mg_projection_mpi.cu`: pure projection (`nu=0`, `n_diff=0`) of a divergent random field.
  Initial max|div| 2.6 → after 8 iterations: single-level GS leaves it large, the **V-cycle reaches
  6.5e-10** (~8×10⁷ lower), and the value is **identical across np=1,2,4** (distribution-exact). np=1,2,4.

### Step 18 — real SDF/fraction cut-cell pressure operator ✅ verified
- `src/mac_cutcell.cuh`: builds the staggered face openness `ox/oy/oz` from an SDF on the extended block
  using cfd's **gradient-normalised fluid fraction** (`cc_fraction_core` = `compute_fluid_fraction_kernel`
  math: `0.5 + sd/denom`, normal from the SDF gradient, clamped) plus the operator mask (`sd≤0 → closed`,
  from `compute_pressure_operator_kernel`). The fraction math is shared with a serial reference so the
  distributed and serial builds use identical arithmetic. Fed to `setFineVariableOperator` it gives the
  true cut-cell `A_C..A_T` on the MG fine level.
- `DistributedStokes::set_cutcell_pressure_operator(sdf_ext)` installs it (enables + builds the MG) and
  keeps the openness for the **cut-cell flux divergence** `div(open·u)` (`diverg_open_k`) — the quantity
  the cut-cell projection is consistent with. `step()` uses the flux divergence as the Poisson RHS when
  the cut-cell operator is active (plain divergence otherwise); `max_open_divergence()` reports it.
- `tests/test_cutcell_operator_mpi.cu`: [1] the distributed `A_C..A_T` from a sphere SDF match the serial
  full-grid reference **bit-for-bit (max|d| = 0)** at np=1,2,4 (coefficients are deterministic functions
  of the SDF — no reduction-order divergence). [2]/[3] the `DistributedStokes` cut-cell projection (solved
  with Galerkin MG + CG, Step 19) drives the flux divergence to RMS ~3e-11, identical across np.

### Step 19 — Galerkin coarsening + CG for the cut-cell multigrid ✅ verified
This step replaces the constant-coefficient coarse operators of Steps 15–18 with **operator-dependent
(Galerkin) coarse operators**, and wraps the V-cycle in **Conjugate Gradients**, so the stiff cut-cell
pressure system actually converges. It is the largest deviation from the original plan; see
"Deviations from the original plan" below for the full rationale.

**Galerkin (aggregation) coarse operators** (`mac_multigrid.cuh`):
- With piecewise-constant **aggregation** transfers — injection prolongation `P` (`mg_prolong_inject_k`,
  each fine cell takes its 2×2×2 parent's value) and its transpose restriction `R = Pᵀ`
  (`mg_restrict_sum_k`, sum of the 8 children) — the variational coarse operator `A_c = Pᵀ A_f P` stays a
  **7-point stencil** whose face transmissibility is the **sum of the 4 fine faces spanning the coarse
  face** (`mg_agg_T_k`). So the coarse operators are *derived from the fine cut-cell operator* and see the
  geometry, unlike the re-discretised constant-coefficient operator. Every level is variable; the
  variable smoother/residual (Step 16) run on all of them. `setFineVariableOperator(..., galerkin=true)`
  builds the hierarchy: fine transmissibility `T = openness/h²`, then `mg_agg_T_k` down each level, then
  `mg_build_op_k` per level. Transfers are local (no halo).
- Limitation (documented, not a bug): **unsmoothed aggregation is a good *preconditioner* but a poor
  standalone solver** — its V-cycle reduces the residual only ~6× more than the constant-coefficient
  coarse V-cycle and still stalls (`test_galerkin_mpi`: 12 V-cycles → residual 1.6e-3).

**CG acceleration** (`solve_pcg`): the symmetric (forward pre-smooth + reverse post-smooth, `Rb=Pᵀ`,
`A_c=PᵀA_fP`) Galerkin V-cycle is an SPD preconditioner; one V-cycle per CG iteration. Needs a
distributed dot product (`mac_dot`, added to `mac_reductions.cuh`) and a matvec (`mg_apply_var_k`).
`test_galerkin_mpi` (strongly-variable smooth Poisson, openness 0.02–1, 50× ratio): the const-coeff
V-cycle stalls at 2e-2, the Galerkin V-cycle at 3e-3, and **CG+Galerkin converges to 2.5e-11 in 28
iterations** — identical across np=1,2,4 (distribution-exact). Wired into `DistributedStokes` via
`set_pressure_pcg(on, max_iter, rtol)`.

**Cut-cell projection consistency** (two corrections needed to make the sphere actually converge):
- The Poisson RHS must be the **open-weighted flux divergence** `div(open·u)` (Step 18), not the plain
  divergence.
- **Do not mask the velocity** when using the cut-cell operator. Velocity masking (`set_solid`) zeros
  partially-open solid-adjacent faces *after* the projection, reintroducing divergence and pinning the
  residual; the open fractions already encode the geometry. `test_cutcell_operator_mpi`: with the
  cut-cell operator + CG and **no masking**, the flux divergence converges to RMS 2.8e-11 (max 2e-10),
  ~1.3×10⁶ better than the Galerkin V-cycle alone, identical across np=1,2,4.

## Deviations from the original plan

The original plan (the suite-level multigrid design and Steps 15–18 here) specified a **geometric
multigrid**: a constant-coefficient periodic Laplacian on every coarse level, with trilinear
prolongation and full-weighting (1/8-average) restriction. That is what Steps 15–17 built and validated
to machine precision against a serial reference, and it is still the default path (`set_pressure_multigrid`,
`setFineVariableOperator(..., galerkin=false)`). The constant-coefficient coarse operator is a fine model
for the all-fluid Poisson but a poor one for the cut-cell operator, so Step 19 deviates. The deviations,
and why each was necessary:

1. **Coarse operator: constant-coefficient → Galerkin (operator-dependent).** A re-discretised
   constant-coefficient coarse grid does not see the solid, so the multigrid could not converge the
   stiff cut-cell system (Step 18 stalled at a ~8× reduction). Step 19 builds variational coarse
   operators `A_c = Pᵀ A_f P` instead, so the coarse grids inherit the geometry. *Sub-deviation:* to keep
   the coarse operator a cheap 7-point stencil I used **aggregation** (piecewise-constant) transfers —
   injection prolongation + summation restriction — rather than the trilinear/full-weighting transfers of
   the geometric path. (The strictly-Galerkin operator for trilinear transfers is a 27-point stencil,
   needing a corner-inclusive halo and far more code; aggregation is the pragmatic, still-variational
   choice.) Both transfer schemes coexist behind the `galerkin_` flag; the geometric path is unchanged.

2. **Standalone V-cycle → CG-preconditioned.** Unsmoothed aggregation is a good *preconditioner* but a
   poor standalone solver (its V-cycle reduces the residual only ~6× more than the constant-coefficient
   coarse V-cycle and then stalls). The deviation is to run the symmetric Galerkin V-cycle as the SPD
   preconditioner inside **Conjugate Gradients** (`solve_pcg`), which converges the system (~1e-11). This
   added a distributed dot product (`mac_dot`) and a symmetric smoother option (reverse post-smooth);
   neither affects the existing geometric path (`solve()` uses the original forward-only smoother, so the
   cell-for-cell tests are unchanged).

3. **Projection RHS: plain divergence → open-weighted flux divergence.** The cut-cell operator is
   `A = -div(open·grad)`, so the consistent projection requires the **flux divergence** `div(open·u)` as
   its RHS, not the plain divergence (which only matches the constant-coefficient operator). Plain
   divergence is still used when no cut-cell operator is set.

4. **No-slip handling: velocity masking is incompatible with the cut-cell operator.** The earlier solver
   imposes no-slip by zeroing the velocity in solid cells (`set_solid`). With the cut-cell operator this
   is *inconsistent*: masking zeros partially-open solid-adjacent faces after the projection and
   reintroduces divergence, pinning the residual. The cut-cell open fractions already encode the
   geometry, so the cut-cell path must **not** mask velocities. Masking remains correct for the
   masking-based (constant-coefficient) path.

Net effect: the constant-coefficient geometric multigrid (Steps 15–17) and the masking-based no-slip
remain the validated defaults; the cut-cell operator + Galerkin + CG + flux-divergence + no-masking form
a separate, self-consistent opt-in path. Everything is validated distributed-vs-serial (bit-exact
coefficients) and by convergence + np-invariance (`test_galerkin_mpi`, `test_cutcell_operator_mpi`).

**Possible further work:** smoothed aggregation or a 27-point geometric-Galerkin operator (better V-cycle
convergence factor, so CG needs fewer iterations); a proper cut-cell no-slip (open-weighted velocity BC
instead of masking); and folding the stack into the in-place `cfd_solver.cu`.

## Status: a working, reusable distributed Navier–Stokes solver with solids and I/O

Steps 1–13 deliver, on the shared decomposition + halo: the async ghost exchange (widths 1 & 2), Koren
advection–diffusion, RB-GS implicit solves, staggered Chorin projection, a full unsteady-Stokes
timestep (Taylor–Green-verified to ~2e-15), flow around an SDF solid, channel flow matching the
analytic Poiseuille profile, a **reusable `DistributedStokes` solver class**, staggered nonlinear
momentum advection (cfd's scheme, momentum-conserving), the **full distributed Navier–Stokes** step,
**Navier–Stokes flow around an SDF solid**, **gather-to-root + VTI output**, **distributed global
reductions** (`max_abs`, `remove_mean`), and a **distributed geometric multigrid** V-cycle for the
Poisson — both constant-coefficient and **variable-coefficient (SDF / cut-cell) fine operators**, with
block-local restriction/prolongation and machine-precision match to a serial reference — **wired as an
opt-in pressure solver in `DistributedStokes::step`** (drives projection divergence to ~1e-9), including
the **real SDF/fraction cut-cell operator** (bit-exact coefficients vs serial) solved with
**Galerkin-coarsened multigrid + CG** (the stiff cut-cell projection converges to ~1e-11). **55/55 MPI
ctests pass**, np=1,2,4. The production `pnm_backend` build is untouched.

## Demo

`tests/demo_flow_sphere.cu` (target `demo_flow_sphere`) is a runnable example: distributed
Navier–Stokes flow past an SDF sphere, gathered and written as a ParaView `.vti`.
```bash
mpirun -np 4 ./build_mpi/demo_flow_sphere [N] [steps] [out.vti]   # defaults: 48, 60, flow_sphere.vti
```
CI runs only a tiny smoke case (`demo_flow_sphere_smoke`).

## Remaining (further work, same pattern)

- **Nonlinear staggered advection** in the distributed step (use the Step-3 Koren operator with the
  staggered advecting-velocity interpolation; ghost width 2) → full Navier–Stokes, not just Stokes.
- **Full Robust-Scaled cut-cell IBM** (D_rescale stencil edits) in place of velocity masking, with cut
  cells straddling block boundaries.
- **Distributed pressure-solve stack is complete** through `DistributedStokes`: MPI global reductions
  (Step 14), the geometric multigrid V-cycle (Step 15), the variable-coefficient fine operator
  (Step 16), the V-cycle wired into the projection (Step 17), the **real SDF/fraction cut-cell operator
  + flux-divergence projection** (Step 18), and **Galerkin coarsening + CG** that converges the stiff
  cut-cell system to ~1e-11 (Step 19). Possible further work (see "Deviations from the original plan"):
  smoothed aggregation / 27-point geometric-Galerkin operators (fewer CG iterations), a proper
  open-weighted cut-cell no-slip BC (vs masking), and folding the stack into the in-place
  `cfd_solver.cu` (extended-block state/scratch) to replace the production single-GPU solver's globals +
  multigrid with the distributed equivalents.
- **Non-periodic BCs & load balance**: physical boundaries as today; ORB balances cell counts, IBM
  load imbalance may warrant weighting later.

## Constraints / notes

- **CUDA-aware MPI is unavailable** on this box (stock OpenMPI built without CUDA); the halo is
  host-staged. See `../transport-core/docs/cuda-aware-mpi.md`.
- Periodicity is exact via the halo (owner-based, any ghost width); non-periodic physical BCs are
  filled by the solver as today.
- Work lives on branch `mpi-halo-integration`; the default module build is unaffected.
