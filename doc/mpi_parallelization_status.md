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

## Remaining (planned)

- **Step 3** — distributed *explicit* operators on an all-fluid periodic domain vs serial:
  divergence (pressure RHS) and the explicit advection–diffusion term. Establishes the per-operator
  exchange pattern (exchange → compute inner).
- **Step 4** — distributed *implicit* velocity diffusion (Red-Black Gauss-Seidel): halo exchange
  after each colour sweep; validate vs the serial diffusion solve (all-fluid, no IBM).
- **Step 5** — distributed pressure Poisson (single-level Jacobi/RB-GS) vs serial (all-fluid).
- **Step 6** — thread the validated exchanges into `step()`: allocate state/scratch as extended local
  blocks, run the existing kernels on them with exchanges between sweeps; validate vs single-rank
  `scripts/verify_poiseuille.py`, `verify_divergence.py`.
- **Step 7** — IBM (cut-cell data is local per block; verify cut cells near block boundaries) and the
  **multigrid hierarchy** (restriction/prolongation across block boundaries) — the hardest parts.

## Constraints / notes

- **CUDA-aware MPI is unavailable** on this box (stock OpenMPI built without CUDA); the halo is
  host-staged. See `../transport-core/docs/cuda-aware-mpi.md`.
- Periodicity is exact via the halo (owner-based, any ghost width); non-periodic physical BCs are
  filled by the solver as today.
- Work lives on branch `mpi-halo-integration`; the default module build is unaffected.
