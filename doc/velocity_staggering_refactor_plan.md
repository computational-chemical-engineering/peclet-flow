# Velocity Staggering Refactor Plan

## Goal
Switch the MAC staggering convention so that:
- SDF cell center at index (i,j,k) is at (i+0.5, j+0.5, k+0.5) * spacing.
- u(i,j,k) is at (i, j+0.5, k+0.5) * spacing.
- v(i,j,k) is at (i+0.5, j, k+0.5) * spacing.
- w(i,j,k) is at (i+0.5, j+0.5, k) * spacing.
The last face plane in each direction is intentionally unused/empty.

## Status Key
- [ ] pending
- [x] done
- [~] in progress

## Task List

### 1) Audit current assumptions
- [x] List all kernels/functions that assume u/v/w are right/top/front faces.
  - Updated: `compute_divergence_kernel`, `compute_pressure_stencil_kernel`,
    `project_velocity_kernel`, `compute_explicit_terms_kernel`,
    `compute_momentum_stencil_kernel`, `get_advection_velocity`,
    `apply_face_sdf_mask_kernel`, `apply_velocity_mask_kernel`.
- [x] Identify all SDF sampling calls and face offsets.
- [x] Find plotting/verification scripts that assume old staggering.

### 2) Core discretization changes
- [x] Divergence: ensure u(i,j,k) is the LEFT face of cell (i,j,k) not right.
- [x] Pressure gradient projection: update gradient stencils to match new faces.
- [x] Advection (PPM/TVD): update control volumes, face flux locations.
- [x] Diffusion/viscous terms: update indexing of Laplacians (unchanged; same neighbor stencil).
- [x] Boundary/periodic wrapping logic: periodic wrapping keeps x=L face implicit.

### 3) IBM + SDF sampling
- [x] Update face-based SDF offsets for u/v/w (now -0.5 in the aligned axis).
- [x] Update compute_ibm_geometry_kernel offsets and ghost detection directions.
- [x] Update face mask kernel and get_face_velocity_for_flux logic.
- [x] Update fluid fraction computation to match new face locations.

### 4) Tests, scripts, and plots
- [x] Update verification scripts to use new index-to-position mapping.
- [x] Regenerate Poiseuille/angled-Poiseuille plots with new convention.
- [ ] Re-run tests and record outcomes.
  - Ran: `python tests/test_fluid_fractions.py` (PASS).
  - Ran: `python scripts/verify_projection_unit_test.py` (PASS after pin-cell masking).

### 5) Documentation
- [x] Add a concise “Grid indexing convention” comment in core headers.
- [x] Update any doc references that mention face locations (none found).

## Notes / Checkpoints
- Previous convention placed u at (i+1, j+0.5, k+0.5) * dx.
- New convention shifts u/v/w by -0.5 cell in their aligned axis.
- Expect to touch both `src/cfd_solver.cu` and `src/cfd_solver_ibm.cu` heavily.
