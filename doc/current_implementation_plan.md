# Implementation Plan: Robust Scaled IBM Solver Refactor

## Overview

Complete refactor of the CFD solver to implement the incremental pressure correction scheme from `doc/Robust_Scaled_IBM_Solver.tex`. This plan covers implementation, unit testing, verification, and debugging.

---

## Current State Analysis

**What works:**
- Newton-Raphson momentum solve with TVD advection (Koren limiter)
- Robust Scaled IBM with D_rescale for velocity Dirichlet BCs
- Basic pressure Poisson solve with RB-GS
- Verification scripts exist (Poiseuille, periodic spheres)

**Problem:**
- Drag factor K shows O(Δt) dependence (see `scripts/verify_dt_dependence.py`)
- Standard projection has first-order splitting error
- `project()` function is stubbed out

**Goal:**
- Implement incremental pressure correction (Eq. 11-14 in LaTeX)
- Achieve O(Δt²) temporal accuracy
- K should be independent of timestep

---

## Phase 1: Data Structure Updates

### Task 1.1: Add new fields to MacGrid
**File:** `src/cfd_solver.cuh`

```cpp
// Add to MacGrid struct:
float *phi;      // Auxiliary pressure correction scalar
float *p_old;    // Previous timestep pressure
float *du, *dv, *dw;  // Velocity increments δu*, δv*, δw*
```

### Task 1.2: Allocate/free memory in constructor/destructor
**File:** `src/cfd_solver.cu`

- Allocate `phi`, `p_old`, `du`, `dv`, `dw` in `CFDSolver::CFDSolver()`
- Free in `CFDSolver::~CFDSolver()`
- Initialize to zero

### Task 1.3: Unit test - memory allocation
**File:** `tests/test_memory_allocation.py`

```python
def test_solver_creation():
    solver = pnm_backend.CFDSolver(res, spacing)
    # Should not crash, memory allocated correctly
```

---

## Phase 2: Area Fraction Computation

### Task 2.1: Implement proper area fraction kernel
**File:** `src/cfd_solver_ibm.cu`

Current `compute_fluid_fraction_kernel` computes fractions but needs verification.

Formula (Eq. 20 in LaTeX):
```
α_f ≈ clamp(0.5 + ψ/(|n_y|Δy + |n_z|Δz), 0, 1)  // for x-face
```

### Task 2.2: Unit test - area fractions
**File:** `tests/test_area_fractions.py`

```python
def test_plane_interface():
    """Plane at x=0.5 should give α=0.5 at that face"""

def test_sphere_fractions():
    """Sphere should have smooth α transition"""

def test_fraction_sum():
    """Sum of α*A should equal fluid volume"""
```

### Task 2.3: Debug visualization
**File:** `scripts/visualize_fractions.py`

- Export α_u, α_v, α_w to VTI
- Visualize in ParaView to verify correctness

---

## Phase 3: Divergence Operator with Area Fractions

### Task 3.1: Refactor divergence kernel
**File:** `src/cfd_solver.cu`

Current `compute_divergence_kernel` signature:
```cpp
__global__ void compute_divergence_kernel(
    const float *u, const float *v, const float *w,
    const float *frac_u, const float *frac_v, const float *frac_w,
    float *rhs, int3 res, float3 spacing, float dt, float rho);
```

Ensure it computes:
```
div = (α_e*u_e - α_w*u_w)/Δx + (α_n*v_n - α_s*v_s)/Δy + (α_t*w_t - α_b*w_b)/Δz
```

### Task 3.2: Handle solid faces with α > 0
**File:** `src/cfd_solver.cu`

For faces where center is solid but α_f > 0:
- Extrapolate velocity from fluid neighbor
- Fallback to u_bc=0 if no fluid neighbor

### Task 3.3: Unit test - divergence
**File:** `tests/test_divergence.py`

```python
def test_uniform_flow_divergence():
    """Uniform flow should have zero divergence"""

def test_divergence_with_sphere():
    """Divergence should be zero in fluid region after projection"""
```

---

## Phase 4: Pressure Stencil with IBM

### Task 4.1: Refactor pressure stencil kernel
**File:** `src/cfd_solver.cu`

The pressure Laplacian should use area fractions:
```
L_p = (α_e(p_E-p_C) - α_w(p_C-p_W))/Δx² + ...
```

Diagonal: `A_C = -(α_e + α_w)/Δx² - (α_n + α_s)/Δy² - (α_t + α_b)/Δz²`

### Task 4.2: Pressure IBM geometry (Neumann BC)
**File:** `src/cfd_solver_ibm.cu`

Current implementation uses bc_type=1 (Neumann) for pressure.
Verify K=1, M=0 logic is correct for dp/dn=0.

### Task 4.3: Solid cell pressure extrapolation
**File:** `src/cfd_solver.cu`

For cells entirely in solid (all α=0):
```
D_s G p = 0  (Laplace smoothing)
```

This prevents pressure oscillations at interface.

### Task 4.4: Unit test - pressure Poisson
**File:** `tests/test_pressure_poisson.py`

```python
def test_manufactured_solution():
    """Set up known p field, verify L*p = b"""

def test_neumann_bc():
    """Gradient at wall should be zero"""
```

---

## Phase 5: Incremental Pressure Correction Scheme

### Task 5.1: Implement φ solve
**File:** `src/cfd_solver.cu`

New kernel `compute_phi_rhs_kernel`:
```cpp
// RHS = div(u + δu*)
__global__ void compute_phi_rhs_kernel(
    const float *u, const float *v, const float *w,
    const float *du, const float *dv, const float *dw,
    const float *frac_u, const float *frac_v, const float *frac_w,
    float *rhs, int3 res, float3 spacing);
```

### Task 5.2: Implement pressure update from φ
**File:** `src/cfd_solver.cu`

Equation 14:
```cpp
// δp = (ρ/(θΔt) + ρ|u|/Δx + μ/Δx²) * φ
__global__ void compute_pressure_from_phi_kernel(
    const float *phi, const float *u, const float *v, const float *w,
    float *dp, int3 res, float3 spacing,
    float dt, float theta, float rho, float mu);
```

For solid cells: `δp = c * φ` where c matches fluid diagonal.

### Task 5.3: Implement velocity correction
**File:** `src/cfd_solver.cu`

```cpp
// u = u + δu* - G*φ
__global__ void correct_velocity_kernel(
    float *u, float *v, float *w,
    const float *du, const float *dv, const float *dw,
    const float *phi, const float *frac_u, const float *frac_v, const float *frac_w,
    int3 res, float3 spacing);
```

### Task 5.4: Unit test - projection
**File:** `tests/test_projection.py`

```python
def test_divergence_free_after_projection():
    """After projection, div(u) should be ~0"""

def test_pressure_update_consistency():
    """Pressure should converge as iterations increase"""
```

---

## Phase 6: Refactor step() Function

### Task 6.1: Restructure main time step
**File:** `src/cfd_solver.cu`

```cpp
void CFDSolver::step(float dt) {
    // 0. Initialize IBM geometry (once, or if geometry changes)

    // 1. Save previous state
    copy(u_old, u); copy(v_old, v); copy(w_old, w); copy(p_old, p);

    // 2. Compute explicit terms (for Crank-Nicolson)
    compute_explicit_terms();

    // 3. Outer Newton iteration
    for (int outer = 0; outer < max_outer_iter; outer++) {

        // 3a. Momentum predictor (solve for δu*, δv*, δw*)
        for (int inner = 0; inner < max_inner_iter; inner++) {
            solve_momentum_u(du);  // J_u * δu* = -f_u
            solve_momentum_v(dv);  // J_v * δv* = -f_v
            solve_momentum_w(dw);  // J_w * δw* = -f_w
        }

        // 3b. Pressure correction
        compute_phi_rhs();           // RHS = div(u + δu*)
        solve_poisson(phi);          // L*φ = RHS
        correct_velocity();          // u += δu* - G*φ
        update_pressure_from_phi();  // p += δp(φ)

        // 3c. Check convergence
        float residual = compute_residual_norm();
        if (residual < tol) break;
    }
}
```

### Task 6.2: Convergence monitoring
**File:** `src/cfd_solver.cu`

Add residual computation and expose to Python:
```cpp
float CFDSolver::get_last_residual() const;
int CFDSolver::get_last_iterations() const;
```

---

## Phase 7: Branchless GPU Optimization

### Task 7.1: Review all kernels for branching
**Files:** `src/cfd_solver.cu`, `src/cfd_solver_ibm.cu`

Checklist:
- [ ] TVD limiter uses `fmin/fmax` only
- [ ] Upwind selection uses `signbit` or arithmetic masks
- [ ] IBM stencil modification is fully pre-computed
- [ ] No `if (is_solid)` in inner loops

### Task 7.2: Memory access pattern review
Checklist:
- [ ] All field arrays use SoA layout
- [ ] Stencil access is coalesced (x is fastest index)
- [ ] IBM sparse lists are sorted by cell index for locality

### Task 7.3: Profile with nvprof/nsight
**Script:** `scripts/profile_solver.py`

```python
def profile_step():
    """Run solver step with profiling enabled"""
    # Use: nvprof --print-gpu-trace python profile_solver.py
```

---

## Phase 8: Verification Suite

### Task 8.1: Taylor-Green Vortex (Advection)
**File:** `scripts/verify_taylor_green_advection.py`

- Initialize TGV with constant convective velocity U=(1,1,1)
- Verify kinetic energy decay matches analytical profile
- Tests Galilean invariance of advection scheme

### Task 8.2: Plane Poiseuille Flow
**File:** `scripts/verify_poiseuille.py`

- Pressure-driven flow between parallel plates
- Compare with analytical parabolic profile
- Use cell-averaged analytical solution for O(Δx²) verification

**Expected results:**
- Velocity profile matches analytical within solver tolerance
- Error decreases as O(Δx²) with grid refinement

### Task 8.3: Flow Past Periodic Spheres
**File:** `scripts/verify_periodic_spheres.py`

- Simple cubic array of spheres
- Compare drag factor K with Zick & Homsy (1982)
- Test cases: φ=0.05 (K≈4.95), φ=0.50 (K≈39.5)

### Task 8.4: Time Step Independence
**File:** `scripts/verify_dt_dependence.py`

- Run same case with Δt = 0.05, 0.1, 0.25, 0.5, 1.0, 2.0
- K should be constant (within 1%) across all Δt
- **This is the primary success metric**

### Task 8.5: Divergence Check
**File:** `scripts/verify_divergence.py`

- After projection, ||div(u)||_∞ < 10⁻⁶
- Automated test that runs after each verification

---

## Phase 9: Debugging Tools

### Task 9.1: Field export to VTI
**File:** `scripts/export_fields.py`

```python
def export_all_fields(solver, filename):
    """Export u,v,w,p,sdf,div to VTI for ParaView"""
```

### Task 9.2: Residual history plotting
**File:** `scripts/plot_convergence.py`

```python
def plot_newton_convergence(solver):
    """Plot residual vs iteration for debugging"""
```

### Task 9.3: Stencil coefficient visualization
**File:** `scripts/visualize_stencil.py`

```python
def visualize_A_coefficients(solver, cell_idx):
    """Print A_C, A_W, A_E, ... for debugging IBM"""
```

### Task 9.4: IBM geometry validation
**File:** `scripts/validate_ibm_geometry.py`

```python
def check_ibm_consistency():
    """Verify D_rescale, K, M, X, B values are sensible"""
    # D_rescale should be in (0, 1]
    # K + M should preserve row sum
    # etc.
```

---

## Phase 10: Final Integration & Cleanup

### Task 10.1: Remove dead code
- Delete stubbed functions
- Remove commented-out old implementations
- Clean up debug prints

### Task 10.2: Update CLAUDE.md
- Document new solver parameters
- Update API usage examples

### Task 10.3: Run full test suite
```bash
source .venv/bin/activate
python -m pytest tests/ -v
python scripts/verify_poiseuille.py
python scripts/verify_periodic_spheres.py
python scripts/verify_dt_dependence.py
python scripts/verify_divergence.py
```

### Task 10.4: Commit with comprehensive message
```
feat: Implement incremental pressure correction scheme

- Add φ auxiliary variable for pressure projection
- Implement Eq. 11-14 from Robust_Scaled_IBM_Solver.tex
- Achieve O(Δt²) temporal accuracy
- Drag factor K now independent of timestep

Verified against:
- Plane Poiseuille flow (analytical)
- Periodic sphere array (Zick & Homsy 1982)
- Taylor-Green vortex advection
```

---

## Success Criteria

1. **dt-independence**: K varies < 1% across Δt range [0.05, 2.0]
2. **Poiseuille accuracy**: L2 error < 1% vs analytical
3. **Sphere drag**: K within 2% of Zick & Homsy values
4. **Divergence**: ||div(u)||_∞ < 10⁻⁶ after projection
5. **No branching**: nvprof shows no warp divergence in main kernels

---

## Execution Order

Start with Phase 1-2 (data structures, area fractions) as foundation.
Then Phase 3-5 (divergence, pressure, projection) for core algorithm.
Phase 6 integrates everything into step().
Phase 7-8 for optimization and verification.
Phase 9-10 for polish.

**Estimated order of files to modify:**
1. `src/cfd_solver.cuh` - data structures
2. `src/cfd_solver.cu` - memory allocation
3. `tests/test_memory_allocation.py` - verify
4. `src/cfd_solver_ibm.cu` - area fractions
5. `tests/test_area_fractions.py` - verify
6. `src/cfd_solver.cu` - divergence, pressure kernels
7. `tests/test_divergence.py`, `tests/test_pressure_poisson.py` - verify
8. `src/cfd_solver.cu` - φ solve, velocity correction
9. `src/cfd_solver.cu` - refactor step()
10. Run all verification scripts
11. Profile and optimize
12. Cleanup and commit
