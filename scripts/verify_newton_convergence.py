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

def run_newton_verification():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    ibm_scheme = 0 # Point-Value
    
    print("=" * 60)
    print(f"Newton Convergence Verification N={res_n}")
    print("=" * 60)

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n
    
    # Generate SDF
    sdf_values = generate_angled_slab_sdf(res_n, L, slab_thickness)
    sdf_reshaped = sdf_values.reshape((res_n, res_n, res_n), order='F')

    # Create solver
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(pnm_backend.SDFData(
        sdf_values,
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0, 0, 0),
        pnm_backend.float3(dx, dx, dx)
    ))
    solver.set_ibm_scheme(ibm_scheme)

    # Physical parameters
    rho = 0.01
    mu = 0.01
    f_mag = 1.0e-2
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # Initialize Analytical Solution
    # Note: Analytical solution depends on mu, not rho (for steady state)
    # The function expects 'nu' in the denominator: u = f / (2*nu) ...
    # We want u = f_dens / (2*mu) ...
    # So we pass 'mu' as the 'nu' argument.
    
    x_faces = np.linspace(0, L, res_n, endpoint=False)
    x_centers = x_faces + 0.5 * dx
    X_u, Y_u, Z_u = np.meshgrid(x_faces, x_centers, x_centers, indexing='ij')
    X_v, Y_v, Z_v = np.meshgrid(x_centers, x_faces, x_centers, indexing='ij')
    X_w, Y_w, Z_w = np.meshgrid(x_centers, x_centers, x_faces, indexing='ij')
    
    u_init = analytical_component_at_points(X_u, Y_u, L, slab_thickness, f_mag, mu, 'u')
    v_init = analytical_component_at_points(X_v, Y_v, L, slab_thickness, f_mag, mu, 'v')
    w_init = analytical_component_at_points(X_w, Y_w, L, slab_thickness, f_mag, mu, 'w')
    
    # Set Initial Condition
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    solver.set_v(v_init.ravel(order='F').astype(np.float32))
    solver.set_w(w_init.ravel(order='F').astype(np.float32))
    
    # Solver parameters
    solver.set_pressure_solver_params(iter=500)
    solver.set_velocity_solver_params(iter=50) # Tight tol to see drift
    solver.set_diffusion_theta(1.0) # Implicit

    # Run a dummy step to ensure IBM geometry and stencils are fully populated
    # (Paranoia check for initialization order)
    solver.step(1e-9)
    
    # Reset to Analytical Initial Condition
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    solver.set_v(v_init.ravel(order='F').astype(np.float32))
    solver.set_w(w_init.ravel(order='F').astype(np.float32))
    
    # --- Helper Functions ---
    def get_stencils_and_scaling():
        stencil_u_bare = [np.array(arr).reshape((res_n, res_n, res_n), order='F') 
                          for arr in solver.get_diffusion_stencil(0, False)]
        stencil_u_ibm = [np.array(arr).reshape((res_n, res_n, res_n), order='F') 
                         for arr in solver.get_diffusion_stencil(0, True)]
        scaling = np.array(solver.get_ibm_scaling(0)).reshape((res_n, res_n, res_n), order='F')
        return stencil_u_ibm, scaling

    def apply_stencil(field, stencil):
        res = stencil[0] * field
        res += stencil[1] * np.roll(field, 1, axis=0)
        res += stencil[2] * np.roll(field, -1, axis=0)
        res += stencil[3] * np.roll(field, 1, axis=1)
        res += stencil[4] * np.roll(field, -1, axis=1)
        res += stencil[5] * np.roll(field, 1, axis=2)
        res += stencil[6] * np.roll(field, -1, axis=2)
        return res

    def compute_div(u, v, w):
        fu = np.array(solver.get_fluid_fraction(1, pnm_backend.float3(-0.5, 0, 0))).reshape((res_n, res_n, res_n), order='F')
        fv = np.array(solver.get_fluid_fraction(2, pnm_backend.float3(0, -0.5, 0))).reshape((res_n, res_n, res_n), order='F')
        fw = np.array(solver.get_fluid_fraction(3, pnm_backend.float3(0, 0, -0.5))).reshape((res_n, res_n, res_n), order='F')
        
        div_u = (np.roll(u, -1, axis=0) * np.roll(fu, -1, axis=0) - u * fu) / dx
        div_v = (np.roll(v, -1, axis=1) * np.roll(fv, -1, axis=1) - v * fv) / dx
        div_w = (np.roll(w, -1, axis=2) * np.roll(fw, -1, axis=2) - w * fw) / dx
        return div_u + div_v + div_w

    def analyze_state(label, u_curr, v_curr, w_curr):
        print(f"\n--- State: {label} ---")
        
        # 1. Momentum Residual
        stencil_u, scaling = get_stencils_and_scaling()
        Lu = apply_stencil(u_curr, stencil_u)
        
        # Mask
        # Face-centered SDF for U (matching staggered location)
        # u[i] is at i-0.5. Compute SDF at i-0.5.
        # This matches get_face_sdf in reproduce_n32_error.py
        def get_face_sdf(s): return 0.5 * (s + np.roll(s, 1, axis=0))
        sdf_face = get_face_sdf(sdf_reshaped)
        
        # Mask: Fluid cells near boundary
        ibm_mask = (sdf_face > 0) & (sdf_face < 0.05)
        
        # Expected forcing term in the discrete equation
        # Solver solves: (rho/dt)u - mu*L(u) = f_density
        # Or mu*L(u) = -f_density (steady state)
        # So L(u) should match -fx.
        expected = -fx
        
        # Calculate Error
        # Resid = A'u - D*f
        resid = Lu - scaling * expected
        
        if np.any(ibm_mask):
            resid_ibm = resid[ibm_mask]
            print(f"  Max Momentum Residual (IBM Boundary): {np.max(np.abs(resid_ibm)):.6e}")
            print(f"  Mean Momentum Residual (IBM Boundary): {np.mean(np.abs(resid_ibm)):.6e}")
            
            # Find worst cell index for debug
            idx_flat = np.argmax(np.abs(resid_ibm))
            # Getting global index is tricky with masked array, skip for summary
        else:
            print("  No IBM Boundary cells found (check mask logic).")
        
        # 2. Divergence
        div = compute_div(u_curr, v_curr, w_curr)
        print(f"  Max Weighted Divergence: {np.max(np.abs(div)):.6e}")
        
        # 3. Estimate Discrete Advection (Upwind)
        # u * du/dx + v * du/dy + w * du/dz
        # For U-component:
        # u_c * (u_c - u_w)/dx if u_c > 0
        u_adv = np.zeros_like(u_curr)
        # Simple Upwind X
        u_vel = u_curr # Approximation
        u_grad_x = np.where(u_vel > 0, 
                            (u_curr - np.roll(u_curr, 1, axis=0))/dx,
                            (np.roll(u_curr, -1, axis=0) - u_curr)/dx)
        # Simple Upwind Y (using average V at U-center)
        v_at_u = 0.25 * (v_curr + np.roll(v_curr, -1, axis=0) + np.roll(v_curr, -1, axis=1) + np.roll(np.roll(v_curr, -1, axis=0), -1, axis=1))
        u_grad_y = np.where(v_at_u > 0,
                            (u_curr - np.roll(u_curr, 1, axis=1))/dx,
                            (np.roll(u_curr, -1, axis=1) - u_curr)/dx)
        
        adv_term = u_vel * u_grad_x + v_at_u * u_grad_y
        
        if np.any(ibm_mask):
            adv_mag = np.max(np.abs(adv_term[ibm_mask]))
            print(f"  Max Discrete Advection (Est): {adv_mag:.6e}")

        return np.max(np.abs(resid[ibm_mask])) if np.any(ibm_mask) else 0.0

    # Need to run a dummy step to init IBM geometry?
    # Initialize calls update_ibm_geometry, so lists should be populated.
    
    u_field = u_init.reshape((res_n, res_n, res_n), order='F')
    v_field = v_init.reshape((res_n, res_n, res_n), order='F')
    w_field = w_init.reshape((res_n, res_n, res_n), order='F')
    
    analyze_state("Initial (Analytical)", u_field, v_field, w_field)
    
    # Time Step 1
    dt = 10.0
    solver.step(dt)
    
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w_field = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    
    analyze_state("After Step 1 (dt=10)", u_field, v_field, w_field)
    
    # Time Step 2
    solver.step(dt)
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w_field = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    analyze_state("After Step 2 (dt=10)", u_field, v_field, w_field)

if __name__ == "__main__":
    run_newton_verification()
