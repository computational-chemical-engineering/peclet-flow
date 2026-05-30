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

## Remaining — the integration into the real solver (large)

- **Step 6** — thread the validated exchanges into the actual `step()`: allocate the state/scratch
  fields (`MacGrid::u,v,w,p`, residuals, explicit terms) as **extended local blocks**, replace the
  in-kernel `get_idx` periodic wrapping with local indexing, and insert `MacGridHalo::exchange()` at
  the right points (after each Red-Black/Jacobi sweep, before each explicit operator, ghost width 2
  where advection is involved). This touches the 3000-line `cfd_solver.cu` pervasively; do it
  operator-by-operator, re-validating against the single-rank references
  (`scripts/verify_poiseuille.py`, `verify_divergence.py`, `verify_periodic_spheres.py`) after each.
- **Step 7** — **IBM**: cut-cell SoA data (`ibm_data*`, `ibm_id_map*`) is built per block; verify cut
  cells straddling block boundaries and that `D_rescale` stencil edits are applied consistently with
  exchanged ghosts. **Multigrid**: the pressure/velocity V-cycles need halo exchange at every level
  and correct restriction/prolongation across block boundaries (coarse-grid decomposition) — the
  hardest remaining piece; a single-level RB-GS fallback (Step 4/5 pattern) works in the meantime.
- **Non-periodic BCs & load balance**: physical boundaries handled as today (BC fills ghosts that fall
  outside the domain); ORB already balances cell counts, but IBM/solid load imbalance may warrant
  weighting later.

## Constraints / notes

- **CUDA-aware MPI is unavailable** on this box (stock OpenMPI built without CUDA); the halo is
  host-staged. See `../transport-core/docs/cuda-aware-mpi.md`.
- Periodicity is exact via the halo (owner-based, any ghost width); non-periodic physical BCs are
  filled by the solver as today.
- Work lives on branch `mpi-halo-integration`; the default module build is unaffected.
