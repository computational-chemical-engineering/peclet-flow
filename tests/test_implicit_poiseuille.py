
import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
# Add build to path (assuming pnm_backend.so is there)
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pnm_backend

def test_implicit_poiseuille():
    print("Testing Implicit Convection (Poiseuille Flow)...")

    # Domain
    nx, ny, nz = 32, 32, 32
    res = pnm_backend.int3(nx, ny, nz)
    dx = 1.0 / nx
    spacing = pnm_backend.float3(dx, dx, dx)
    
    # Physics
    rho = 1.0
    nu = 0.01
    g = 1.0
    # Analytical Max Velocity for Channel Width H (Full Width?)
    # Setup: Channel in Y defined by SDF.
    # Walls at y=0.25 and y=0.75?
    # Let's use simple walls at edges or defined by planes.
    
    # Create planar channel SDF
    # Top and Bottom walls.
    # d = min(y - y_min, y_max - y)  (Positive inside)
    y_min, y_max = 0.2, 0.8
    H = y_max - y_min
    
    sdf_vals = np.zeros((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                y = (j + 0.5) * dx
                d = min(y - y_min, y_max - y)
                sdf_vals[k, j, i] = d
    
    sdf_data = pnm_backend.SDFData(sdf_vals.flatten(), res, 
                                   pnm_backend.float3(0,0,0), spacing)

    solver = pnm_backend.CFDSolver(res, spacing)
    solver.initialize(sdf_data)
    solver.set_rho(rho)
    # Update to use Dynamic Viscosity mu
    mu = nu * rho
    solver.set_mu(mu)
    solver.set_diffusion_theta(1.0)
    solver.set_body_force(pnm_backend.float3(g, 0.0, 0.0)) # Flow in X
    
    # Implicit Parameters
    solver.set_pressure_solver_params(iter=100)
    solver.set_velocity_solver_params(iter=20) # Inner iterations
    
    # Determine Time Step for High CFL
    # u_max_analytical = g H^2 / (8 nu)
    u_max = g * H**2 / (8 * nu)
    # CFL = u * dt / dx
    # Target CFL = 5.0
    dt_cfl_1 = dx / u_max
    dt = 5.0 * dt_cfl_1
    
    print(f"Analytical Max Velocity: {u_max:.4f}")
    print(f"Grid dx: {dx:.4f}")
    print(f"Time Step: {dt:.4f} (CFL ~ 5.0)")
    
    # Run to Steady State
    # T_diff = H^2 / nu
    T_final = 2.0 * (H**2 / nu)
    num_steps = int(T_final / dt) + 1
    
    print(f"Running {num_steps} implicit steps...")
    
    start_time = time.time()
    for n in range(num_steps):
        solver.step(dt)
        if n % 10 == 0:
            u_field = np.array(solver.get_u()).reshape(nz, ny, nx)
            curr_max = np.max(u_field)
            print(f"Step {n}: Max U = {curr_max:.4f}")
            
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f}s")
    
    # Analyze Results
    u_field = np.array(solver.get_u()).reshape(nz, ny, nx)
    # Extract centerline profile (along Y, at mid X, Z)
    mid_x, mid_z = nx // 2, nz // 2
    profile = u_field[mid_z, :, mid_x]
    
    # Analytical Profile
    # u(y) = g/(2nu) * (y - y_min)(y_max - y)
    y_coords = (np.arange(ny) + 0.5) * dx
    u_analytical = np.zeros_like(profile)
    for j in range(ny):
        y = y_coords[j]
        if y > y_min and y < y_max:
            u_analytical[j] = (g / (2 * nu)) * (y - y_min) * (y_max - y)
            
    # Error
    error_L2 = np.sqrt(np.mean((profile - u_analytical)**2))
    max_sim = np.max(profile)
    err_rel = abs(max_sim - u_max) / u_max
    
    print("\nResults:")
    print(f"Simulated Max U: {max_sim:.4f}")
    print(f"Analytical Max U: {u_max:.4f}")
    print(f"Relative Error (Max): {err_rel*100:.2f}%")
    print(f"L2 Error Profile: {error_L2:.6f}")
    
    if err_rel < 0.05:
        print("SUCCESS: Error within 5%")
    else:
        print("FAILURE: Error too high")
        sys.exit(1)

if __name__ == "__main__":
    test_implicit_poiseuille()
