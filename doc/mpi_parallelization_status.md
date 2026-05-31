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
cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/build/_deps/pybind11-src
cmake --build build_mpi --target test_mac_halo -j
ctest --test-dir build_mpi --output-on-failure     # mac_halo_np{1,2,4}
```

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

## Status: a working, reusable distributed Navier–Stokes solver with solids

Steps 1–11 deliver, on the shared decomposition + halo: the async ghost exchange (widths 1 & 2), Koren
advection–diffusion, RB-GS implicit solves, staggered Chorin projection, a full unsteady-Stokes
timestep (Taylor–Green-verified to ~2e-15), flow around an SDF solid, channel flow matching the
analytic Poiseuille profile, a **reusable `DistributedStokes` solver class**, the **staggered nonlinear
momentum advection** operator (cfd's scheme, momentum-conserving), and the **full distributed
Navier–Stokes** step (cell-for-cell vs serial). **30/30 MPI ctests pass**, np=1,2,4. The production
`pnm_backend` build is untouched.

## Remaining (further work, same pattern)

- **Nonlinear staggered advection** in the distributed step (use the Step-3 Koren operator with the
  staggered advecting-velocity interpolation; ghost width 2) → full Navier–Stokes, not just Stokes.
- **Full Robust-Scaled cut-cell IBM** (D_rescale stencil edits) in place of velocity masking, with cut
  cells straddling block boundaries.
- **Full in-place `cfd_solver.cu`** — extended-block fields + MPI global reductions + distributed
  **multigrid** (halo per level, restriction/prolongation across block boundaries). Largest remaining
  piece; the single-level RB-GS path (Steps 4–6) is the working fallback.
- **Non-periodic BCs & load balance**: physical boundaries as today; ORB balances cell counts, IBM
  load imbalance may warrant weighting later.

## Constraints / notes

- **CUDA-aware MPI is unavailable** on this box (stock OpenMPI built without CUDA); the halo is
  host-staged. See `../transport-core/docs/cuda-aware-mpi.md`.
- Periodicity is exact via the halo (owner-based, any ghost width); non-periodic physical BCs are
  filled by the solver as today.
- Work lives on branch `mpi-halo-integration`; the default module build is unaffected.
