"""Verify the projected velocity field is discretely divergence-free."""
import sys
import os
import numpy as np

# Add build directory to path to import pnm_backend
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'build')))
import pnm_backend

def generate_sphere_sdf(res_n, L=1.0, R=0.2):
    """Generates SDF for a single sphere in center."""
    dx = L/res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    xc, yc, zc = L/2, L/2, L/2
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)
    sdf = dist - R
    
    return sdf.ravel(order='F').astype(np.float32), (dx, dx, dx)

def check_divergence():
    res_n = 32
    L = 1.0
    R = 0.25
    
    sdf_vals, (dx, dy, dz) = generate_sphere_sdf(res_n, L, R)
    
    nx, ny, nz = res_n, res_n, res_n
    
    sdf_data = pnm_backend.SDFData(
        sdf_vals, 
        pnm_backend.int3(nx, ny, nz),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dy, dz)
    )
    
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(nx, ny, nz), 
        pnm_backend.float3(dx, dy, dz)
    )
    solver.initialize(sdf_data)
    
    solver.set_rho(1.0)
    solver.set_mu(0.01)
    solver.set_body_force(pnm_backend.float3(1.0, 0, 0)) # Force in X
    
    dt = 0.05
    solver.set_pressure_solver_params(iter=5000)
    
    print("Running 5 steps...")
    for i in range(5):
        solver.step_implicit(dt)

    u_flat = np.array(solver.get_u())
    v_flat = np.array(solver.get_v())
    w_flat = np.array(solver.get_w())
    
    u = u_flat.reshape((nx, ny, nz), order='F')
    v = v_flat.reshape((nx, ny, nz), order='F')
    w = w_flat.reshape((nx, ny, nz), order='F')
    sdf = sdf_vals.reshape((nx, ny, nz), order='F')
    
    # Compute Divergence
    u_right = np.roll(u, -1, axis=0) # u[i+1]
    u_left = u
    v_top = np.roll(v, -1, axis=1)
    v_bottom = v
    w_front = np.roll(w, -1, axis=2)
    w_back = w
    
    div = (u_right - u_left)/dx + (v_top - v_bottom)/dy + (w_front - w_back)/dz
    
    max_div = np.max(np.abs(div))
    mean_div = np.mean(np.abs(div))
    
    print(f"Max Divergence (Overall): {max_div:.6e}")
    print(f"Mean Divergence (Overall): {mean_div:.6e}")
    
    # Check max velocity inside solid (parasitic currents)
    fluid_mask = sdf >= 0
    solid_mask = ~fluid_mask
    if np.any(solid_mask):
        max_u_solid = np.max(np.abs(u[solid_mask]))
        print(f"Max U inside solid: {max_u_solid:.6e}")
    
    if max_div < 1e-4:
        print("PASS: Divergence is low.")
    else:
        print("FAIL: Divergence is high.")

if __name__ == "__main__":
    check_divergence()
