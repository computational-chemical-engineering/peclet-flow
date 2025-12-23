# Implementation Plan: Accelerated CFD Solver with IBM

## Goal
Implement a GPU-accelerated Incompressible Navier-Stokes solver using a Staggered Grid arrangement (MAC Grid) and the Immersed Boundary Method (IBM) to handle complex geometries (porous media).

## User Review Required
> [!IMPORTANT]
> **Solver Choice**: detailed below. We will proceed with **Red-Black Gauss-Seidel (RB-GS)** for the Pressure Poisson Equation in Phase 1.
> *Reasoning*: RB-GS is easy to implement "matrix-free" on GPU and parallelizes well (checkerboard pattern). While strictly slower convergence-wise than Multigrid or PCG for large grids, it is sufficient for testing the IBM formulation. We can upgrade to a PCG solver later if needed.

> [!NOTE]
> The solver will be implemented in C++/CUDA, integrating with the existing SDF-based geometry representation found in `src/sdf_reader.h`.

## Proposed Architecture

### 1. Data Structures: Staggered Grid (MAC)
We will use a Structure of Arrays (SoA) layout for coalesced memory access on the GPU.

#### Grid Variables
- $u$: Velocity x-component, defined at $(i+1/2, j, k)$
- $v$: Velocity y-component, defined at $(i, j+1/2, k)$
- $w$: Velocity z-component, defined at $(i, j, k+1/2)$
- $p$: Pressure, defined at cell centers $(i, j, k)$
- $f_{solid}$: Solid volume fraction (from SDF), defined at cell centers.

**Memory Layout & Indexing**:
-   **Device/C++**: Linear Index $I = x + N_x \cdot y + N_x \cdot N_y \cdot z$.
    -   $x$ ($i$) is the fastest moving index (stride 1).
    -   $z$ ($k$) is the slowest moving index.
    -   This matches the existing `pore_extraction.cu` helper: `z * res.y * res.x + y * res.x + x`.
-   **Python/NumPy**:
    -   To maintain `arr[i, j, k]` indexing in Python with this memory layout, we must use **Fortran Ordering** (`order='F'`) when creating arrays.
    -   Shape: `(Nx, Ny, Nz)`.
    -   If standard C-ordering is used, the shape would need to be `(Nz, Ny, Nx)` (transposed). We will strictly use `order='F'` to keep $(x,y,z)$ coordinates intuitive.

**Periodic Dimensions**:
Since the Domain is periodic, face counts equal cell counts.
Faces $i=0$ and $i=N_x$ are identical. We store indices $0 \dots N_x-1$.
Indices wrap modulo dimension size.

**Refined Struct Definition:**
We will define this in a new header `src/cfd_solver.cuh`.
```cpp
struct MacGrid {
    int3 res;       // Grid resolution (cells)
    float3 spacing; // dx, dy, dz
    
    // Periodic boundaries: Faces = Cells in count
    // SoA for reduced memory transactions
    float* u; // Size: nx * ny * nz (u-faces at i+1/2, wrapping)
    float* v; // Size: nx * ny * nz (v-faces at j+1/2, wrapping)
    float* w; // Size: nx * ny * nz (w-faces at k+1/2, wrapping)
    float* p; // Size: nx * ny * nz
    float* rhs; // Divergence of intermediate velocity
    
    // Body force (Macroscopic Pressure Gradient)
    float3 body_force; 
    
    // IBM fields
    float* sdf; // Signed Distance Field (from existing loader)
};
```

### 2. Time Integration: Chorin's Projection Method
1.  **Advection**: Dimension-Split (Cascade) PPM.
    -   **Approach**: Split transport into X, Y, Z passes. $\Phi^* = \mathcal{A}_x(\mathcal{A}_y(\mathcal{A}_z(\Phi^n)))$.
    -   **Method**: PPM (Piecewise Parabolic Method) on Staggered Control Volumes.
        -   Reconstruct face values $\phi_L, \phi_R$ using 4-point stencil of cell averages.
        -   Compute Upwind Flux at faces based on transverse velocity.
        -   Update conservative variable: $\phi^{new} = \phi - \frac{\Delta t}{\Delta x} (F_{right} - F_{left})$.
    -   **Conservation**: This is strictly conservative (Flux Form).
    -   **Handling Staggered Variables**:
        -   $u$-velocity control volume is shifted half-cell in X.
        -   We treat $u, v, w$ as scalars defined on their respective grids.
    -   **Directional Splitting**: Strang Splitting ($X-Y-Z$, then $Z-Y-X$) or simple Lie Splitting ($X-Y-Z$) depending on order requirements. We will start with **Lie Splitting** ($X-Y-Z$) for simplicity and performance, which is $O(\Delta t)$.
    -   **Periodic boundaries**: PPM stencil wraps around.
3.  **Diffusion**: Crank-Nicolson (Implicit-Explicit) using Red-Black Gauss-Seidel.
    -   Equation: $\frac{u^* - u^{adv}}{\Delta t} = \nu \frac{1}{2} (\nabla^2 u^* + \nabla^2 u^{adv})$.
    -   Rearranged: $(I - \mu \nabla^2) u^* = (I + \mu \nabla^2) u^{adv}$, where $\mu = \frac{\nu \Delta t}{2}$.
    -   **Step 1**: Compute RHS $b = (I + \mu \nabla^2) u^{adv}$ (Explicit convolution).
    -   **Step 2**: Solve Helmholtz system $A u^* = b$ using **Red-Black Gauss-Seidel**.
        -   Operator $A = I - \mu \nabla^2$.
        -   Similar to Pressure solve but with diagonal dominance (better convergence).
    -   Applied independently to $u, v, w$.
4.  **Body Force**: Add `body_force` term.
5.  **Pressure Projection**: Solve Poisson equation $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot u^*$.
    -   *Note*: For periodic domains, the pressure is determined up to a constant. We may need to fix one node ($p(0)=0$) or enforce mean zero to prevent drift, though algorithms usually handle floating levels purely by gradients.
4.  **Correction**: Update velocity $u^{n+1} = u^* - \frac{\Delta t}{\rho} \nabla p$.

## Immersed Boundary Method (IBM)
### Phase 2a: Staircase Approximation
-   **Concept**: Voxelize the geometry. If a cell center $SDF < 0$, it is Solid.
-   **Velocity Masking**:
    -   At every step (after advection/diffusion/force), enforce $u=0$ on solid faces.
    -   Face $(i+1/2)$ is solid if Cell $i$ OR Cell $i+1$ is solid.
-   **Pressure Boundary Condition**:
    -   At Fluid-Solid interface, flow is 0.
    -   Implies $\nabla p \cdot n = \text{force} \cdot n \approx 0$ (if we ignore external forcing at wall).
    -   Simplest robust BC: **Homogeneous Neumann** ($dp/dn = 0$) at solid faces.
    -   **Solver Mod**: Check neighbors in RB-GS. If neighbor is Solid, do not add its $p$ term and reduce diagonal coefficient (effectively treating neighbor $p$ as equal to center $p$).

### Phase 2b: Partial Cell (Future)
-   ...

### 3. Linear System Solver (Pressure Poisson)
To solve $A p = b$ on the GPU:
-   **Method**: Red-Black Gauss-Seidel (SOR) with **Periodic Neighbors**.
    -   Stencil uses `(x+1)%nx`, `(x-1+nx)%nx`, etc.
-   **Implementation**:
    -   Two kernels: `solve_pressure_red` and `solve_pressure_black`.
    -   Use `__syncthreads()` if doing block-level shared memory optimization, but global memory ping-pong is easier for large grids.
-   **Convergence**: compute residual norm reduction.

### 4. Immersed Boundary Method (IBM)
For handling complex porous structures:
-   **Direct Forcing**: Modify momentum equation.
-   **Interpolation**:
    -   Use the SDF value to determine the exact distance to the wall.
    -   For velocity nodes inside the solid, or near the interface, we force $u_{interp} = u_{wall}$ (0).
    -   *Implementation*: `apply_ibm_forcing_kernel` that runs after advection-diffusion but before projection? Recalibrating: standard IBM modifies the Laplacian stencil or applies a forcing term.
    -   *Simpler Approach*: Ghost-Cell Immersed Boundary Method (GCIBM) or simply masking for Phase 1.
    -   *Plan*: We will start with a **Step-Stair approximation** (voxelized) first.
        -   SDF is defined at cell centers $(i,j,k)$.
        -   Velocity $u_{i+1/2, j, k}$ is adjudged "solid" if the average SDF at the face is $< 0$ (or if either neighbor is solid).
        -   Strategy: Precompute a `byte` or `bit` mask for each velocity component ($u,v,w$) indicating if it is active or solid-constrained (0).
        -   *Note*: Periodic boundaries are handled *before* IBM check. IBM simply zeros out velocity at specific locations regardless of where they wrapped from.

## Code Strategy: Periodic Helpers
We will adopt the integer math helpers found in `src/pore_extraction.cu` for robust wrapping:

```cpp
__device__ int get_idx(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}
```

## Proposed Changes

### `src/`
#### [NEW] [cfd_solver.cuh](file:///home/frankp/Codes/pnm_from_sdf/src/cfd_solver.cuh)
-   Define `MacGrid` struct.
-   Declare kernels: `advect`, `project`, `solve_pressure`.

#### [NEW] [cfd_solver.cu](file:///home/frankp/Codes/pnm_from_sdf/src/cfd_solver.cu)
-   Implement memory management (`init_grid`, `free_grid`).
-   Implement the solver kernels.
-   Implement the host stepper function `step_simulation(...)`.

#### [MODIFY] [bindings.cpp](file:///home/frankp/Codes/pnm_from_sdf/src/bindings.cpp)
-   Expose the `CFDSolver` class/functions to Python.

## Verification Plan

### Automated Tests
-   **Lid-Driven Cavity**:
    -   Create a Python script `tests/test_lid_driven_cavity.py`.
    -   Initialize solver with $U_{top} = 1.0$.
    -   Run for sufficient steps to reach steady state.
    -   Compare vertical centerline $u$-velocity with benchmarks (Ghia et al.).
    -   *Success criteria*: $L_2$ error < threshold.

### Manual Verification
-   **Divergence Check**:
    -   After projection, compute $\nabla \cdot u$.
    -   Ensure it is close to machine epsilon (or solver tolerance).
-   **Visual**:
    -   Export .vti files (already supported in project?).
    -   Visualize in ParaView used `Structure` (SDF) and `Velocity`.
