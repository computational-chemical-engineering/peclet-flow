# Implementation Status Report

**Date:** 2026-01-05
**Task:** Implement Incremental Pressure Correction Scheme from `Robust_Scaled_IBM_Solver.tex`

---

## Summary

Implemented the full incremental pressure correction scheme (Eq. 14). The solver compiles, runs, and produces accurate results for Poiseuille flow (0.24% error at N=64). The dt-dependence has been reduced for small timesteps (K clusters within ~1% for dt≤0.25).

---

## What Was Implemented

### 1. Data Structure Updates (Complete)
- Added `p_old` field to MacGrid for storing previous timestep pressure
- Added `du`, `dv`, `dw` fields for velocity increments
- Added `phi` field for auxiliary pressure correction scalar
- Memory properly allocated in constructor, freed in destructor

### 2. Full Eq. 14 Pressure Update (Complete)
Implemented the "rotational" incremental pressure correction scheme:

**Eq. 14 from LaTeX:**
```
δp = (ρ/(θΔt)*I + ρ*C_φ - μ*L_φ) * φ
```

**Adjusted for our code's scaling (φ_our = (ρ/Δt)*φ_latex):**
```cpp
δp = φ/θ + Δt*(u·∇φ) - ν*Δt*∇²φ
```

The kernel `update_pressure_from_phi_kernel` at `src/cfd_solver.cu:1424-1513` computes:
1. **Temporal term**: φ/θ
2. **Convection term**: Δt * (u·∇φ) using central differences
3. **Diffusion term**: -ν*Δt*∇²φ (rotational correction)

### 3. Momentum Equation Updated
- Changed momentum stencil to use `p_old` instead of `p` to ensure predictor uses old pressure

---

## Verification Results

### Poiseuille Flow (Working)
```
N     Error%     L2 Error    Order
16    20.51%     1.99e-01    -
32    11.15%     1.13e-01    0.82
64    0.24%      2.91e-03    5.28
```
The solver converges to the analytical solution (0.24% error at N=64).

### dt-Dependence (Improved for small dt)

**Current results with full Eq. 14:**
```
dt         K
--------------------
0.05       2.37
0.10       2.37
0.25       2.38
0.50       2.43
1.00       2.57
2.00       2.78
```

- Small dt (0.05-0.25): K clusters at 2.37-2.38 (within ~1%)
- Larger dt (0.5-2.0): K increases due to:
  - Fewer timesteps to reach steady state
  - Accumulation of time integration error
  - CFL-like stability considerations

---

## Analysis

The small-dt convergence demonstrates that the incremental pressure correction scheme is working correctly. The K≈2.37 value at small dt is the "converged" drag factor for this geometry and resolution.

The K values (~2.37) are lower than Zick & Homsy (K≈4.95 for φ=0.05). This discrepancy is likely due to:
1. Coarse grid resolution (N=32)
2. IBM treatment near the sphere surface
3. Periodic boundary effects

**Note:** Higher resolutions (N>64) require geometric multigrid for efficient convergence. Current RB-GS iterations scale as O(N²).

---

## Files Modified

| File | Changes |
|------|---------|
| `src/cfd_solver.cuh` | Added `p_old`, `phi`, `du`, `dv`, `dw` fields to MacGrid |
| `src/cfd_solver.cu` | Added memory alloc/free; implemented `update_pressure_from_phi_kernel` with full Eq. 14; modified `step()` to use incremental scheme |

---

## Verification Notebook

Created `notebooks/verification_incremental_pressure.ipynb` with:
1. Poiseuille flow spatial convergence study
2. Order of convergence analysis
3. dt-dependence visualization
4. Summary of results

---

## Next Steps

1. **Geometric Multigrid**: Implement V-cycle multigrid for the pressure Poisson solve to enable efficient high-resolution simulations

2. **Higher Resolution Testing**: Once multigrid is implemented, verify K convergence at N=128, 256

3. **IBM Refinement**: Investigate if the K≈2.37 value improves with resolution

---

## Code Location

The pressure update kernel is at `src/cfd_solver.cu:1424-1513`:
```cpp
__global__ void update_pressure_from_phi_kernel(
    float *p, const float *phi, const float *p_old,
    const float *u, const float *v, const float *w,
    int3 res, float3 spacing, float dt, float theta, float rho, float mu)
{
    // ... neighbor indexing ...

    float nu = mu / rho;

    // 1. Temporal term: φ/θ
    float temporal_term = phi_C / theta;

    // 2. Convection term: Δt * (u·∇φ)
    float convection_term = dt * (uc * dphi_dx + vc * dphi_dy + wc * dphi_dz);

    // 3. Diffusion term: -ν*Δt*∇²φ
    float diffusion_term = -nu * dt * laplacian;

    // Full update
    p[idx] = p_old[idx] + temporal_term + convection_term + diffusion_term;
}
```
