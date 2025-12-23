# Implementation Plan: Accelerated CFD Solver with IBM

## Goal
Implement a GPU-accelerated Incompressible Navier-Stokes solver using a Staggered Grid arrangement (MAC Grid) and the Immersed Boundary Method (IBM) to handle complex geometries (porous media).

## User Review Required
> [!IMPORTANT]
> Confirm the choice of **Red-Black Gauss-Seidel** for the pressure Poisson equation. While easy to parallelize, Multigrid methods might be required for faster convergence on large grids. We will start with RB-GS for simplicity.

> [!NOTE]
> The solver will be implemented in C++/CUDA, integrating with the existing SDF-based geometry representation.

## Proposed Architecture

### 1. Data Structures: Staggered Grid (MAC)
We will use a Structure of Arrays (SoA) layout for coalesced memory access on the GPU.

#### Grid Variables
- $u$: Velocity x-component, defined at $(i+1/2, j, k)$
- $v$: Velocity y-component, defined at $(i, j+1/2, k)$
- $w$: Velocity z-component, defined at $(i, j, k+1/2)$
- $p$: Pressure, defined at cell centers $(i, j, k)$
- $f_{solid}$: Solid volume fraction (from SDF), defined at cell centers.

```cpp
struct MacGrid {
    int3 res;
    float3 spacing;
    
    // SoA for reduced memory transactions
    float* u; // Size: (nx+1) * ny * nz
    float* v; // Size: nx * (ny+1) * nz
    float* w; // Size: nx * ny * (nz+1)
    float* p; // Size: nx * ny * nz
    float* rhs; // Divergence of intermediate velocity
};
```

### 2. Time Integration: Chorin's Projection Method
1.  **Advection-Diffusion**: Predict intermediate velocity $u^*$ ignoring pressure.
    -   Advection: Semi-Lagrangian (stable for large $\Delta t$) or Upwind (conservative). *Recommendation: Semi-Lagrangian.*
    -   Diffusion: Explicit central differences (subject to stability limit) or Implicit (Crank-Nicolson).
2.  **Pressure Projection**: Solve Poisson equation $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot u^*$.
3.  **Correction**: Update velocity $u^{n+1} = u^* - \frac{\Delta t}{\rho} \nabla p$.

### 3. Linear System Solver (Pressure Poisson)
To solve $A p = b$ on the GPU:
-   **Method**: Red-Black Gauss-Seidel (SOR).
-   **Implementation**: Two kernels per iteration (Update Red nodes, then Update Black nodes).
-   **Convergence**: Check residual $L_2$ norm every $N$ iterations.

### 4. Immersed Boundary Method (IBM)
For handling the complex porous structures defined by the SDF:
-   **Direct Forcing**: Modify the momentum equation to force velocity to zero (or wall velocity) at the solid interface.
-   **INTERPOLATION**:
    -   Since the grid lines don't align with the geometry, we use interpolation to enforce $u=0$ at the exact surface distance $d=0$ (from SDF).
    -   For a grid point inside the solid close to the boundary, the stencil is modified to incorporate the boundary condition.

## Step-by-Step Implementation

### Phase 1: Basic Solver (No IBM)
1.  Implement `MacGrid` allocation/deallocation on GPU.
2.  Implement `advect_velocity_kernel` (Semi-Lagrangian).
3.  Implement `solve_pressure_rbgs_kernel` (Red-Black GS).
4.  Implement `project_velocity_kernel`.
5.  Verify with a simple lid-driven cavity or channel flow case.

### Phase 2: IBM Integration
1.  Integrate SDF data: Map SDF values to the MAC grid.
2.  Implement `classify_cells_kernel`: Determine Fluid, Solid, and Ghost cells.
3.  Modify Advection: Ensure backtracking works near boundaries (clamping or valid region checks).
4.  Modify Pressure Solve: Homogeneous Neumann $\partial p / \partial n = 0$ at solid boundaries.

## Verification Plan

### Automated Tests
-   **Lid-Driven Cavity**: Compare center-line velocities with Ghia et al. benchmark data.
-   **Poiseuille Flow**: Verify parabolic profile in a straight pipe.

### Manual Verification
-   Visual inspection of velocity vectors in ParaView (using VTI export).
-   Monitor divergence of velocity field (should stay near zero).
