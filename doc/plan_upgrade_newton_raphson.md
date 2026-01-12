# Implementation Plan: Mixed-Precision Newton-Raphson Solver

## Objective
Upgrade the current single-precision (Float32) "Total Form" solver to a mixed-precision (Double state, Float solver) "Delta Form" Newton-Raphson solver. This improves accuracy and robustness while maintaining GPU performance.

## Target Architecture

*   **State (`MacGrid`)**:
    *   `u, v, w, p`: **Double** (FP64). High precision state integration.
    *   `phi, rhs`: **Float** (FP32). Linear solver working variables.
    *   `du, dv, dw`: **Float** (FP32). Newton corrections.
    *   `res_u, res_v, res_w`: **Float** (FP32). Residuals.
*   **Kernels**:
    *   `compute_residual`: Reads `double` state, computes `double` non-linear terms, writes `float` residual.
    *   `solve_linear`: Reads `double` state (coefficients), solves for `float` correction.
    *   `apply_correction`: Reads `float` correction, accumulates into `double` state.

## Phase 1: Type Promotion & Memory Refactor

### 1.1 Header Update (`src/cfd_solver.cuh`)
*   Update `MacGrid` struct:
    *   Change `float *u, *v, *w, *p` to `double *u, *v, *w, *p`.
    *   Change `float3 u_bc_` to `double3 u_bc_` (optional, for consistency).
    *   Ensure `du, dv, dw` exist as `float*`.
*   Update `CFDSolver` methods:
    *   `get_u()` -> `std::vector<double>`.
    *   `set_u(const std::vector<double>&)`.

### 1.2 Host Implementation (`src/cfd_solver.cu`)
*   Update `initialize`: Allocate `double` buffers. Cast input SDF/Values to double.
*   Update `step`: Temporary logic to handle double pointers (casting to float for old kernels if needed during transition, or updating signatures).
*   Update Memory Management: `cudaMalloc` sizes.

### 1.3 Kernel Signatures
*   Systematically update all kernel signatures to accept `double*` for velocity/pressure fields.
*   *Transitional Logic*: Inside kernels, use `(float)val` if logic is not yet upgraded to double math, to ensure compilation.

### 1.4 Bindings (`src/bindings.cpp`)
*   Update `pybind11` interface to expose `std::vector<double>` for field accessors.

## Phase 2: Kernel Implementation (Delta Form)

### 2.1 Residual Kernel (`compute_residual`)
*   Create/Update `compute_momentum_residual_kernel`:
    *   Input: `double* u`, `double* u_old`, ...
    *   Output: `float* res_u`.
    *   Logic: Compute `(u - u_old)/dt + Adv + Diff + Grad P` in **Double Precision**.
    *   Store result cast to `float`.

### 2.2 Jacobian / Linear Solver (`rbgs_delta_step`)
*   Update `solve_velocity_implicit_kernel`:
    *   Input: `double* u` (for coefficients), `float* residual` (RHS).
    *   Output: `float* du` (Correction).
    *   Logic:
        *   Compute coefficients $A_{nb}$ using `u` (if non-linear) or standard stencil.
        *   Solve $A \delta u = -R$.
        *   Use Red-Black Gauss-Seidel.

### 2.3 Correction Kernel (`apply_correction`)
*   New Kernel `apply_correction_kernel`:
    *   Input: `double* u`, `float* du`.
    *   Output: `u += du`.

## Phase 3: Host Logic Refactor (`step`)

### 3.1 Newton Loop
*   Replace the current `step` logic with:
    ```cpp
    // Predictor / Explicit Terms (if any)
    
    for (iter = 0; iter < outer_iters; iter++) {
        // 1. Compute Residual (Double -> Float)
        compute_momentum_residual(u, res_u);
        
        // 2. Clear Correction
        cudaMemset(du, 0, ...);
        
        // 3. Inner Linear Solve (Float) for Correction
        for (inner = 0; inner < inner_iters; inner++) {
            solve_linear(u, res_u, du);
        }
        
        // 4. Update State (Float -> Double)
        apply_correction(u, du);
    }
    ```

## Phase 4: Pressure Update
*   Update `project_velocity_kernel` and `update_pressure_from_phi`:
    *   Ensure they handle `double` p and u.
    *   Pressure update `p += delta_p` should be double.

## Verification
*   Compile and run `scripts/verify_poiseuille.py`.
*   Check residual history for convergence.
