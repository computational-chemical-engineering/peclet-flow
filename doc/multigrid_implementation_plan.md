# Geometric Multigrid Implementation Plan for the GPU 7-Point Stencil Solver

## Goal

Replace the current iteration-heavy Red-Black Gauss-Seidel (RBGS) solves with a geometric multigrid (GMG) framework that is tuned to this codebase:

- structured 3D periodic grids
- explicit 7-point stencil coefficients stored in SoA form
- GPU execution
- ghost-cell IBM baked into the fine-grid operator
- the existing `phi`-based pressure correction solve plus three staggered momentum/Helmholtz solves

Important constraint:

- GMG should accelerate the linear solve for the auxiliary field `phi`
- the current `phi -> velocity projection -> pressure update` path must remain intact
- this plan does not propose switching to a different direct-pressure formulation

The target use case is exactly the kind of simulation in [simulate_flow_around_packing.py](/home/frankp/Codes/pnm_from_sdf/tests/simulate_flow_around_packing.py): flow through or around complex SDF-defined packings at moderate to high resolution, where the current RBGS iteration counts become the dominant cost.

## Current Solver Summary

### What exists today

The current solver path in [cfd_solver.cu](/home/frankp/Codes/pnm_from_sdf/src/cfd_solver.cu) is:

1. Build a 7-point operator on the fine grid.
2. For momentum:
   - launch `compute_fused_fluid_kernel`
   - launch `compute_fused_ibm_kernel`
   - solve the correction system with repeated RBGS sweeps
3. For pressure:
   - launch `compute_pressure_stencil_kernel`
   - solve the linear system for `phi` with repeated RBGS sweeps
   - apply `project_velocity_kernel`
   - apply `update_pressure_from_phi_kernel`

The stencil arrays are:

- `A_C`
- `A_W`, `A_E`
- `A_S`, `A_N`
- `A_B`, `A_T`
- `B_RHS`

These are already in the right form for a matrix-free geometric multigrid implementation with 7-point operators.

### Why this is a good GMG target

This solver already has the properties GMG wants:

- structured index space
- periodic neighbors via `get_idx`
- explicit operator coefficients
- regular 7-point topology
- static geometry in typical runs
- a pressure operator that is geometry-dependent but state-independent

The main missing pieces are:

- level hierarchy data structures
- residual/restriction/prolongation kernels
- coarse-grid operators
- V-cycle orchestration

## Recommended Scope

### Phase 1: pressure multigrid first

Implement GMG for the `phi` solve first.

This is the best first target because:

- the pressure operator is the cleanest 7-point operator in the code
- the operator depends on geometry and fractions, not on the evolving advective state
- the pressure matrix can be cached across time steps for static geometry
- `simulate_flow_around_packing.py` currently uses a very large `p_max_iter`

This gives the highest return for the lowest implementation risk.

### Phase 2: momentum multigrid second

Extend the same GMG framework to the three momentum correction solves.

This is more involved because:

- the operator changes every outer iteration
- the operator is nonsymmetric because of upwind advection terms
- the staggered-grid geometry differs for `u`, `v`, and `w`
- coarse-grid IBM handling is harder than for pressure

The Phase 2 recommendation is to use multigrid first as a fast inner solver or preconditioner for the momentum systems, not as the very first part of the rollout.

## Multigrid Design for This Solver

## 1. Hierarchy Type

Use geometric multigrid with factor-2 coarsening in each direction:

- fine grid: `Nx x Ny x Nz`
- coarse levels: `Nx/2 x Ny/2 x Nz/2`, continuing until a small bottom grid

Because the code is periodic and structured, this is the natural hierarchy.

Initial implementation assumption:

- all dimensions must be divisible by `2^(L-1)` for the chosen number of levels

If a dimension becomes odd, stop coarsening there and use the previous level as the coarsest grid.

## 2. Level Storage

Add new multigrid-specific level structs, rather than overloading `MacGrid`.

Suggested split:

### `MGPressureLevel`

- `int3 res`
- `float3 spacing`
- `int num_elements`
- operator arrays: `A_C`, `A_W`, `A_E`, `A_S`, `A_N`, `A_B`, `A_T`
- vectors: `x`, `rhs`, `residual`, `correction`
- geometry arrays:
  - `sdf`
  - `frac_u`, `frac_v`, `frac_w`

### `MGMomentumLevel`

Per component hierarchy, or one generic level reused component-by-component:

- `int3 res`
- `float3 spacing`
- `int num_elements`
- operator arrays: `A_C`, `A_W`, `A_E`, `A_S`, `A_N`, `A_B`, `A_T`
- vectors: `x`, `rhs`, `residual`, `correction`
- geometry arrays:
  - `sdf`
  - component-specific IBM data
  - component-specific `ibm_id_map`
  - component-specific `fluid_indices`

### Why separate pressure and momentum hierarchies

Pressure and momentum do not use the same geometry preprocessing:

- pressure uses fraction-weighted operator construction
- momentum uses fluid lists plus ghost-cell IBM geometry

Keeping them separate will make the implementation much easier to reason about and tune.

## 3. Smoother

Use the existing RBGS kernel as the first multigrid smoother.

Recommendation:

- pre-smoothing: 2 sweeps
- post-smoothing: 2 sweeps
- bottom solve: 16 to 64 sweeps depending on coarse grid size

Reasons:

- it already exists
- it matches the current solver
- it uses the same 7-point arrays
- it keeps the first GMG version low-risk

Later optimization option:

- replace or augment RBGS with weighted Jacobi or Chebyshev smoothing if kernel launch overhead or red/black synchronization becomes limiting

## 4. Transfer Operators

### Restriction

Use full-weighting restriction for residuals.

For cell-centered pressure-like fields:

- standard 3D full-weighting with periodic wrap

For staggered momentum fields:

- use the same logical index-space full-weighting on the component grid
- the field offset is handled by the geometry/operator build, not by the restriction stencil itself

This keeps transfer operators simple and consistent with the solver’s storage model.

### Prolongation

Use trilinear prolongation for coarse-grid corrections.

For momentum components:

- prolongate the correction on the component grid itself
- then add the prolongated correction into `du`, `dv`, or `dw`

This is the right tradeoff between quality and GPU simplicity.

## 5. Residual Kernel

Add a generic residual kernel for 7-point operators:

```text
r = b - A x
```

This kernel should accept:

- `x`
- `A_C`, `A_W`, `A_E`, `A_S`, `A_N`, `A_B`, `A_T`
- `b`
- `res`

This will be reused for:

- pressure V-cycles
- momentum V-cycles
- convergence diagnostics

## 6. Coarse-Grid Operators

This is the most important design choice in the whole plan.

Because you want geometric multigrid with 7-point stencils, the coarse operators should also be 7-point operators built geometrically, not arbitrary sparse RAP products.

### Pressure / `phi` operator

Recommended approach:

1. Build a coarse SDF hierarchy once.
2. Recompute coarse face fractions from the coarse SDF.
3. Build the pressure stencil on each level from those coarse fractions.

This stays geometric, preserves the 7-point structure, and fits the current fraction-based pressure formulation.

### Momentum operators

Recommended approach:

1. Build a coarse SDF hierarchy once.
2. For each momentum component and each nonlinear outer iteration:
   - restrict the state needed to assemble the operator
   - rebuild the coarse 7-point operator using the same logic as on the fine level
   - rebuild the coarse IBM geometry from the coarse SDF

The state to restrict for coarse momentum operators is:

- the component field being corrected
- the advecting velocities needed by the upwind terms
- the pressure field entering the current linearization

This is more work than pressure, but it keeps the method geometric and keeps all levels on 7-point stencils.

### What not to do in version 1

Do not start with exact Galerkin `A_c = R A_f P` because:

- it generally destroys the strict 7-point structure
- it adds implementation complexity
- it does not match the user’s stated goal of a geometric 7-point multigrid

## 7. Coarse SDF Hierarchy

Build a geometry hierarchy once after `initialize()` or `update_ibm_geometry()`.

Recommended method:

- define each coarse grid by halving the resolution
- sample the fine SDF onto the coarse cell centers using trilinear interpolation

Then for each level:

- pressure hierarchy:
  - compute `frac_u`, `frac_v`, `frac_w`
  - build cached pressure operator
- momentum hierarchy:
  - compute component-specific IBM geometry from the level SDF
  - classify fluid indices on the level grid

Why this is the right fit:

- the original geometry source is already an SDF
- the fine-grid IBM implementation is SDF-based
- coarse-grid operator construction stays consistent with the fine-grid method

## 8. Nullspace Handling for Pressure

The pressure solve is effectively a periodic Neumann problem and therefore has a constant nullspace.

This must be handled correctly in multigrid.

Recommended treatment:

- enforce zero-mean residual on every level before or after restriction
- enforce zero-mean correction on every level after smoothing
- keep the current final pressure shift on the fine grid

Do not rely on coarse-grid RBGS alone to control the nullspace drift.

This is a solver-specific requirement and should be part of the initial implementation, not an afterthought.

## Pressure GMG Plan

This section refers specifically to the linear solve for the auxiliary field `phi`, not to a direct solve for `p`.

## A. Refactor the pressure path

Split the current pressure kernel responsibilities.

Current kernel:

- `compute_pressure_stencil_kernel` builds both operator and RHS

Refactor into:

1. `compute_pressure_operator_kernel`
   - build `A_*` only
   - depends only on geometry/fractions
2. `compute_pressure_rhs_kernel`
   - build `B_RHS` for the `phi` equation from divergence of the current velocity field

Why:

- the operator can then be cached for static geometry
- the RHS remains dynamic per outer iteration
- the cached fine-level `phi` operator becomes the level-0 matrix in GMG

## B. Build the pressure hierarchy once

During initialization or geometry update:

1. allocate `MGPressureLevel` objects
2. build coarse SDF levels
3. compute coarse fractions on each level
4. build and store `A_*` on each level

This hierarchy should be reused across time steps as long as the geometry does not change.

## C. Add pressure V-cycle kernels

Add:

- `compute_residual_kernel`
- `restrict_full_weighting_kernel`
- `prolongate_trilinear_and_add_kernel`
- `subtract_mean_kernel` generalized per level

Then add a host-side recursive or iterative V-cycle driver.

Recommended first version:

- host-side control flow
- GPU kernels for all level operations

This keeps debugging much easier than a fully device-side recursion scheme.

## D. Integrate into `step()`

Replace:

- `for (k = 0; k < p_max_iter_; ++k) { RBGS }`

with:

- `n_pressure_cycles` V-cycles for the `phi` solve only

Initial mapping:

- `p_max_iter_` becomes either:
  - `pressure_v_cycles`
  - or a compatibility wrapper that translates to V-cycles

Recommended starting value:

- `2` to `4` V-cycles per outer iteration

After the GMG solve for `phi`, keep the rest of the current correction path unchanged:

1. `project_velocity_kernel`
2. `update_pressure_from_phi_kernel`
3. final pressure shift/pinning

## Momentum GMG Plan

## A. Reuse the same multigrid framework

Momentum should use the same core multigrid machinery:

- level operators in 7-point SoA form
- RBGS smoother
- residual/restriction/prolongation kernels

Only the operator build and geometry preparation differ.

## B. Refactor fine-grid momentum assembly

The current fine-grid assembly is split into:

- `compute_fused_fluid_kernel`
- `compute_fused_ibm_kernel`

For multigrid, create a level-generic assembly interface so the same logic can be applied on coarse levels.

Suggested direction:

1. keep the current fine kernels for level 0 if that is simpler
2. add coarse-level variants or generalized kernels that accept:
   - level resolution
   - level spacing
   - level SDF
   - level IBM data
   - restricted state fields

## C. Build coarse momentum geometry from coarse SDF

For each coarse level and each component:

- run the same IBM geometry logic used on the fine grid
- classify fluid and IBM cells for that level

This is essential. Restricting the fine-grid operator without rebuilding the IBM geometry would undermine the whole point of a geometric method.

## D. Use GMG first as the inner correction solver

Replace:

- `v_max_iter_` RBGS sweeps

with:

- `n_velocity_cycles` V-cycles per component per outer iteration

Recommended first setting:

- `1` or `2` V-cycles for `u`
- `1` or `2` V-cycles for `v`
- `1` or `2` V-cycles for `w`

Because the momentum systems are nonsymmetric, the first target is practical acceleration, not perfect textbook multigrid optimality.

## GPU-Specific Tuning Decisions

## 1. Keep the 7-point SoA layout

Do not convert to CSR or another sparse format.

The current SoA layout:

- matches coalesced memory access
- works with existing kernels
- keeps neighbor fetches predictable

## 2. Prefer full-domain level vectors over masked sparse solves

Even though fluid and IBM cells are sparse lists on the fine grid for assembly, the smoother and residual kernels should operate on full level arrays.

Reason:

- simpler kernels
- regular memory access
- easier restriction/prolongation
- no need for sparse coarse-grid indexing logic

This matches the current solver style.

## 3. Cache geometry-heavy data

For static geometry:

- coarse SDF levels
- coarse fractions
- coarse IBM lists

should be built once and reused.

Pressure operator hierarchy should definitely be cached.

Momentum geometry hierarchy should also be cached. Only the dynamic momentum operator coefficients need rebuilding per outer iteration.

## 4. Mixed precision

Recommended initial precision:

- operators and MG vectors in `float`
- keep the main state fields `u`, `v`, `w`, `p` in their current precision

This matches the current correction solves, which already use `float` for the linear solve vectors and coefficients.

## 5. Coarsest solve

Do not over-engineer the coarsest solve in version 1.

Use:

- many RBGS sweeps on the coarsest level

Only if that becomes a bottleneck should you consider:

- a tiny direct CPU solve
- a dense GPU solve

## Suggested File-Level Changes

## New files

Add:

- `src/cfd_solver_multigrid.cuh`
- `src/cfd_solver_multigrid.cu`

These should contain:

- level structs
- allocation/free helpers
- restriction/prolongation/residual kernels
- V-cycle drivers
- pressure hierarchy build
- momentum hierarchy build

## Existing files to modify

### [src/cfd_solver.cuh](/home/frankp/Codes/pnm_from_sdf/src/cfd_solver.cuh)

Add:

- multigrid configuration parameters
- hierarchy members
- helper methods:
  - `build_pressure_multigrid()`
  - `build_momentum_multigrid()`
  - `pressure_v_cycle(...)`
  - `momentum_v_cycle(...)`

### [src/cfd_solver.cu](/home/frankp/Codes/pnm_from_sdf/src/cfd_solver.cu)

Refactor:

- pressure operator/RHS split
- `step()` integration for pressure GMG
- later, `step()` integration for momentum GMG

### [src/cfd_solver_ibm_kernels.cuh](/home/frankp/Codes/pnm_from_sdf/src/cfd_solver_ibm_kernels.cuh)

Potentially reuse:

- SDF interpolation
- IBM geometry kernel

for coarse-level geometry builds.

### [src/bindings.cpp](/home/frankp/Codes/pnm_from_sdf/src/bindings.cpp)

Expose:

- MG enable/disable
- number of V-cycles
- smoother sweep counts
- max number of levels

## Validation Plan

## 1. Pressure-only validation

Before touching momentum GMG:

1. verify that the fine-level `phi` operator built by the refactored code matches the current operator
2. compare RBGS and GMG residual reduction for the `phi` solve on the same RHS
3. verify that the downstream `phi -> projection -> pressure update` behavior remains unchanged
4. verify nullspace handling by checking zero-mean residual and stable pressure shift

Test cases:

- all-fluid periodic domain
- planar cut-cell channel
- sphere packing case from `simulate_flow_around_packing.py`

## 2. End-to-end pressure speedup

On `simulate_flow_around_packing.py`:

- compare runtime per `step()`
- compare number of pressure iterations/cycles needed to reach equivalent divergence reduction
- confirm unchanged qualitative flow field

## 3. Momentum GMG validation

After pressure GMG is stable:

1. compare one-step momentum corrections against the RBGS baseline
2. compare residual reduction per unit time
3. verify stability in angled-wall and sphere-packing cases

## 4. Scaling tests

Run:

- `64^3`
- `128^3`
- if practical, `256^3`

Measure:

- wall time per step
- pressure solve fraction of total runtime
- momentum solve fraction of total runtime
- achieved convergence per cycle

## Implementation Order

## Phase 1: infrastructure

1. add multigrid level structs
2. add residual, restriction, prolongation kernels
3. generalize mean-subtraction per level
4. add host-side V-cycle driver

## Phase 2: pressure GMG

1. split pressure operator build from RHS build
2. build coarse SDF and fraction hierarchy
3. cache pressure operators
4. replace pressure RBGS loop with V-cycles
5. validate on all-fluid and packing cases

## Phase 3: momentum GMG

1. generalize momentum operator assembly to level data
2. build coarse IBM geometry hierarchies
3. add one component at a time, starting with `u`
4. integrate all three components
5. tune pre/post sweeps and bottom sweeps

## Phase 4: tuning

1. profile kernel launch overhead
2. compare RBGS vs weighted Jacobi smoother
3. tune number of levels
4. tune cycle counts per solve type

## Design Decisions Requiring Feedback

These are the decisions I would want your feedback on before implementing.

### 1. Pressure-only first, or pressure plus momentum immediately?

My recommendation:

- implement pressure GMG first
- then extend to momentum once the framework is stable

Reason:

- largest immediate win
- much lower implementation risk

### 2. Is the first goal a fast pressure solver only, or a full replacement of all RBGS loops?

My recommendation:

- first milestone: pressure only
- second milestone: replace velocity RBGS loops too

### 3. Are we allowed to require power-of-two-compatible grid sizes for version 1?

My recommendation:

- yes for version 1
- later generalize if needed

This keeps the first hierarchy implementation much cleaner.

### 4. For momentum coarse operators, do you want a strictly geometric rediscretization from coarse SDF and restricted state, or would you accept a temporary approximate coefficient restriction?

My recommendation:

- strict geometric rediscretization

Reason:

- matches your request
- preserves the 7-point formulation
- stays consistent with the IBM design

### 5. Do you want multigrid to be a standalone solver, or a preconditioner-like accelerator that still keeps a few RBGS sweeps?

My recommendation:

- pressure: standalone V-cycle solver
- momentum: use V-cycles as the inner solver, but keep the surrounding nonlinear outer loop exactly as it is

### 6. Is static geometry a valid assumption for the first implementation?

My recommendation:

- yes

That allows caching:

- coarse SDF levels
- pressure operator hierarchy
- IBM geometry hierarchies

### 7. Is it acceptable to keep the coarsest solve as repeated RBGS instead of a direct solve?

My recommendation:

- yes for version 1

That is much simpler and probably sufficient on small bottom grids.

### 8. Do you want to preserve the current mixed-precision approach?

My recommendation:

- yes

Keep:

- state in current precision
- MG operators and corrections in `float`

### 9. Should we expose multigrid controls through Python immediately?

My recommendation:

- yes, but minimally

Expose:

- enable/disable multigrid
- max levels
- pre-sweeps
- post-sweeps
- bottom sweeps
- V-cycles per solve

## Recommended First Milestone

If the goal is practical speedup soonest, the first milestone should be:

1. pressure-only geometric multigrid
2. cached `phi` hierarchy from static SDF/fractions
3. fine-grid `phi` RHS built each outer iteration
4. 2 to 4 V-cycles replacing the `p_max_iter_` RBGS loop for `phi`
5. benchmark on `simulate_flow_around_packing.py`

That is the shortest path to a meaningful solver speedup in this codebase.
