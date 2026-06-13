"""
Reproduction Script for N=32 Angled Poiseuille Error (Point-Value Scheme)
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def generate_angled_slab_sdf(res_n, L, slab_thickness):
    dx = L / res_n
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    sqrt2 = np.sqrt(2.0)
    d = (Y - X) / sqrt2
    period = L / sqrt2
    d_wrapped = d - period * np.round(d / period)
    sdf = np.abs(d_wrapped) - slab_thickness / 2.0
    return sdf.ravel(order='F').astype(np.float32)

def analytical_component_at_points(x, y, L, slab_thickness, f_mag, nu, comp):
    sqrt2 = np.sqrt(2.0)
    period = L / sqrt2
    half_t = slab_thickness / 2.0
    H = period - slab_thickness
    y = np.mod(y, L)
    d = (y - x) / sqrt2
    d_wrapped = d - period * np.round(d / period)
    u_parallel = np.zeros_like(d_wrapped)
    abs_d = np.abs(d_wrapped)
    fluid = abs_d > half_t
    d_wall = abs_d[fluid] - half_t
    u_parallel[fluid] = (f_mag / (2.0 * nu)) * d_wall * (H - d_wall)
    if comp == 'u' or comp == 'v':
        return u_parallel / sqrt2
    return np.zeros_like(u_parallel)

def run_reproduction():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    ibm_scheme = 0 # Point-Value
    verbose = True
    
    print("=" * 60)
    print(f"Reproducing N={res_n} Error with Scheme={ibm_scheme} (Point-Value)")
    print("=" * 60)

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n
    period = L / sqrt2
    H = period - slab_thickness

    # Generate SDF
    sdf_values = generate_angled_slab_sdf(res_n, L, slab_thickness)

    # Create solver
    sdf_data = pnm_backend.SDFData(
        sdf_values,
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0, 0, 0),
        pnm_backend.float3(dx, dx, dx)
    )
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(sdf_data)
    solver.set_ibm_scheme(ibm_scheme)

    # Physical parameters
    rho = 1.0
    nu = 0.01
    mu = nu * rho
    f_mag = 1.0e-2
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # Initialize Analytical Solution (Point-Sampled)
    x_faces = np.linspace(0, L, res_n, endpoint=False)
    x_centers = x_faces + 0.5 * dx
    X_u, Y_u, Z_u = np.meshgrid(x_faces, x_centers, x_centers, indexing='ij')
    X_v, Y_v, Z_v = np.meshgrid(x_centers, x_faces, x_centers, indexing='ij')
    X_w, Y_w, Z_w = np.meshgrid(x_centers, x_centers, x_faces, indexing='ij')
    
    u_init = analytical_component_at_points(X_u, Y_u, L, slab_thickness, f_mag, nu, 'u')
    v_init = analytical_component_at_points(X_v, Y_v, L, slab_thickness, f_mag, nu, 'v')
    w_init = analytical_component_at_points(X_w, Y_w, L, slab_thickness, f_mag, nu, 'w')
    
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    solver.set_v(v_init.ravel(order='F').astype(np.float32))
    solver.set_w(w_init.ravel(order='F').astype(np.float32))
    
    # Solver parameters
    solver.set_pressure_solver_params(iter=500)
    solver.set_velocity_solver_params(iter=50)
    solver.set_diffusion_theta(1.0)

    # Time stepping
    dt = 10.0
    
    # Run simulation
    solver.step(dt)
    
    # Get final fields
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')

    # --- Operator Analysis ---
    print("  Extracting Diffusion Stencils...")
    stencil_u_bare_flat = solver.get_diffusion_stencil(0, False)
    stencil_u_ibm_flat = solver.get_diffusion_stencil(0, True)
    ibm_scaling_flat = solver.get_ibm_scaling(0)
    ibm_scaling = np.array(ibm_scaling_flat).reshape((res_n, res_n, res_n), order='F')
    
    def reshape_stencil(flat_stencil, n):
        return [np.array(arr).reshape((n, n, n), order='F') for arr in flat_stencil]
        
    stencil_u_bare = reshape_stencil(stencil_u_bare_flat, res_n)
    stencil_u_ibm = reshape_stencil(stencil_u_ibm_flat, res_n)
    
    def apply_stencil(field, stencil):
        res = stencil[0] * field
        res += stencil[1] * np.roll(field, 1, axis=0)
        res += stencil[2] * np.roll(field, -1, axis=0)
        res += stencil[3] * np.roll(field, 1, axis=1)
        res += stencil[4] * np.roll(field, -1, axis=1)
        res += stencil[5] * np.roll(field, 1, axis=2)
        res += stencil[6] * np.roll(field, -1, axis=2)
        return res

    u_init_reshaped = u_init.reshape((res_n, res_n, res_n), order='F')
    
    # Apply operators
    Lu_bare_ana = apply_stencil(u_init_reshaped, stencil_u_bare)
    Lu_ibm_ana = apply_stencil(u_init_reshaped, stencil_u_ibm)
    
    safe_scaling = np.where(np.abs(ibm_scaling) < 1e-4, 1.0, ibm_scaling)
    Lu_ibm_ana_norm = Lu_ibm_ana / safe_scaling
    
    expected_val = -rho * fx
    
    # Use Face-Centered SDF for staggered component U
    # u[i,j,k] is at (i, j+0.5, k+0.5) logical or (i*dx, (j+0.5)*dx, (k+0.5)*dx) physical
    # Wait, the SDF interpolated by the GPU uses offset (-0.5, 0, 0) relative to cell index?
    # No, compute_ibm_geometry_kernel uses offset (-0.5, 0, 0).
    # This means for index i, it samples at i - 0.5.
    
    def get_face_sdf(sdf_reshaped):
        # Linear interpolation to face x=i
        # (SDF[i-1] + SDF[i]) / 2
        return 0.5 * (sdf_reshaped + np.roll(sdf_reshaped, 1, axis=0))

    sdf_reshaped = sdf_values.reshape((res_n, res_n, res_n), order='F')
    sdf_face_u = get_face_sdf(sdf_reshaped)
    
    deep_fluid_mask = sdf_face_u > 0.05
    fluid_mask = sdf_face_u > 0
    ibm_mask = (sdf_face_u > 0) & (sdf_face_u < 0.05)
    
    print(f"  Expected Viscous Term L(u) = {expected_val:.6e}")
    
    if np.any(ibm_mask):
         D_vals = ibm_scaling[ibm_mask]
         print(f"  IBM Scaling (D) at Boundary: Min={np.min(D_vals):.4f}, Max={np.max(D_vals):.4f}, Mean={np.mean(D_vals):.4f}")
         
    if np.any(deep_fluid_mask):
        print(f"  Bare Operator on Ana U (Deep Fluid): Mean={np.mean(Lu_bare_ana[deep_fluid_mask]):.6e}, Std={np.std(Lu_bare_ana[deep_fluid_mask]):.6e}")
        print(f"  IBM Operator on Ana U (Normalized, Deep Fluid):  Mean={np.mean(Lu_ibm_ana_norm[deep_fluid_mask]):.6e}, Std={np.std(Lu_ibm_ana_norm[deep_fluid_mask]):.6e}")
        Lu_ibm_raw = Lu_ibm_ana[deep_fluid_mask]
        expected_scaled = expected_val * ibm_scaling[deep_fluid_mask]
        diff = Lu_ibm_raw - expected_scaled
        print(f"  IBM Scaled Residual Accuracy (A'u - D*f) at Deep Fluid: Mean={np.mean(diff):.6e}, Std={np.std(diff):.6e}, L_inf={np.max(np.abs(diff)):.6e}")

    Lu_ibm_raw = Lu_ibm_ana[fluid_mask]
    expected_scaled = expected_val * ibm_scaling[fluid_mask]
    diff = Lu_ibm_raw - expected_scaled
    print(f"  IBM Scaled Residual Accuracy (A'u - D*f) at Fluid: Mean={np.mean(diff):.6e}, Std={np.std(diff):.6e}, L_inf={np.max(np.abs(diff)):.6e}")
    Lu_ibm_raw = Lu_ibm_ana_norm[fluid_mask]
    diff = Lu_ibm_raw - expected_val
    print(f"  IBM Residual Accuracy (Normalized, 1/D*A'u - f) at Fluid: Mean={np.mean(diff):.6e}, Std={np.std(diff):.6e}, L_inf={np.max(np.abs(diff)):.6e}")

    if np.any(ibm_mask):
        Lu_ibm_raw = Lu_ibm_ana
        expected_scaled = expected_val * ibm_scaling
        diff = Lu_ibm_raw[ibm_mask] - expected_scaled[ibm_mask]
        print(f"  IBM Scaled Residual Accuracy (A'u - D*f) at Boundary: Mean={np.mean(diff):.6e}, Std={np.std(diff):.6e}")
        
        # Also print normalized stats to confirm reproduction
        print(f"  IBM Operator on Ana U (Normalized, Boundary):    Mean={np.mean(Lu_ibm_ana_norm[ibm_mask]):.6e}, Std={np.std(Lu_ibm_ana_norm[ibm_mask]):.6e}")

    # --- Debug Specific Cell ---
    if np.any(ibm_mask):
        # Find max error
        diff = Lu_ibm_raw[ibm_mask] - expected_scaled[ibm_mask]
        max_err_idx_local = np.argmax(np.abs(diff))
        
        # Get global index
        indices = np.argwhere(ibm_mask)
        target_idx_tuple = tuple(indices[max_err_idx_local])
        
        print(f"\n--- WORST Cell {target_idx_tuple} ---")
        print(f"  SDF (Face): {sdf_face_u[target_idx_tuple]:.6f}")
        print(f"  Scaling (D): {ibm_scaling[target_idx_tuple]:.6f}")
        print(f"  A_C: {stencil_u_ibm[0][target_idx_tuple]:.6e}")
        print(f"  u_C: {u_init_reshaped[target_idx_tuple]:.6e}")
        print(f"  R_ibm: {Lu_ibm_raw[target_idx_tuple]:.6e}")
        print(f"  R_exp: {expected_scaled[target_idx_tuple]:.6e}")
        print(f"  Error: {Lu_ibm_raw[target_idx_tuple] - expected_scaled[target_idx_tuple]:.6e}")
    else:
        print("\nNO IBM CELLS DETECTED BY FACE MASK!")
    
if __name__ == "__main__":
    run_reproduction()
