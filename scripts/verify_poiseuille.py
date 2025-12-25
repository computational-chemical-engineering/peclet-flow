import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def generate_slab_sdf(res_n, L, slab_thickness):
    # Slab centered at L/2 (Solid inside)
    # Fluid domain: y in [0, L] excluding [Lc - t/2, Lc + t/2]
    # Fluid Height H = L - t
    
    dx = L/res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    
    # Shape (Nx, Ny, Nz) F-order
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    yc = L/2
    # Distance to surface: |y - yc| - t/2
    # If d < 0: Inside slab (Solid)
    # If d > 0: Outside (Fluid)
    # SDF Convention: Negative is Solid? 
    # Usually: SDF > 0 fluid, SDF < 0 solid.
    # Current code: if (sdf < 0) -> solid.
    
    # So we want SDF < 0 inside slab.
    # d = |y - yc| - t/2
    # Inside: |y-yc| < t/2 => d < 0. Correct.
    
    # We want to represent the distance to the CLOSEST surface.
    dist = np.abs(Y - yc) - slab_thickness/2.0
    
    # NOTE: Since strictly speaking SDF is distance to boundary, 
    # and boundary is at y = yc +/- t/2.
    # This formula is correct signed distance.
    
    return dist.ravel(order='F').astype(np.float32)

def run_simulation(res_n, save_plot=False):
    L = 1.0
    slab_thickness = 0.2
    H = L - slab_thickness # Fluid Channel Height
    
    sdf_values = generate_slab_sdf(res_n, L, slab_thickness)
    dx = L/res_n
    
    # Setup Data
    sdf_data = pnm_backend.SDFData(
        sdf_values, 
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dx, dx)
    )
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n), 
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(sdf_data)
    
    rho = 1.0
    nu = 0.01
    solver.set_rho(rho)
    # mu = nu * rho
    mu = nu * rho
    solver.set_mu(mu)
    
    g_x = 1.0e-2 # Body force acceleration
    solver.set_body_force(pnm_backend.float3(g_x, 0, 0))

    # Set solver parameters
    # Scale iterations with resolution
    if res_n == 16:
        pressure_max_iter = 100
    elif res_n == 32:
        pressure_max_iter = 500
    else:
        pressure_max_iter = 2000
        
    pressure_tol = 1e-5
    velocity_max_iter = 50 # Increase to ensure diffusion converges
    velocity_tol = 1e-5
    cfl = 0.5
    
    solver.set_pressure_solver_params(max_iter=pressure_max_iter, tol=pressure_tol)
    solver.set_velocity_solver_params(max_iter=velocity_max_iter, tol=velocity_tol)

    solver.set_cfl(cfl)
    
    # Run
    max_steps = 10000
    u_mean_history = []
    
    print(f"Running N={res_n} with CFL={cfl}...")
    for i in range(max_steps):
        # Pass dt=-1.0 to trigger adaptive stepping
        solver.step(-1.0)
        
        if i % 100 == 0:
            u_field = np.array(solver.get_u())
            u_mean = np.mean(u_field)
            u_mean_history.append(u_mean)
            
            if len(u_mean_history) > 5:
                err = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-12)
                if err < 1e-6:
                    break

    u_field = np.array(solver.get_u())
    
    # Get Mean Velocity in Fluid Phase only?
    # Solver.get_u() returns domain average? No, get_u returns full grid.
    # Masked values are 0.
    # U_sup (Superficial) = sum(u) / V_total
    # U_fluid_avg = sum(u) / V_fluid = U_sup / porosity
    # Porosity phi_f = H / L
    
    u_field = np.array(solver.get_u())
    
    # Check Centerline (Max) Velocity to avoid integration errors with cut-cells
    U_sim_max = np.max(u_field)
    
    # Analytical Max Velocity (Plane Poiseuille)
    # U_max = (g * H^2) / (8 * nu)
    U_ana_max = (g_x * H**2) / (8.0 * nu)
    
    error = 100 * abs(U_sim_max - U_ana_max) / U_ana_max
    
    print(f"  U_max_sim={U_sim_max:.6e}, U_max_ana={U_ana_max:.6e}, Error={error:.2f}%")
    
    return res_n, error, U_sim_max

def run_study():
    resolutions = [16, 32, 64] # Add 128 if fast enough
    results = []
    
    print("Plane Poiseuille Verification")
    print("-----------------------------")
    for res in resolutions:
        results.append(run_simulation(res))
        
    print("\nGrid Convergence Results:")
    print(f"{'N':<5} {'Error%':<10} {'Order':<10}")
    
    for i, (n, err, u) in enumerate(results):
        order = "-"
        if i > 0:
            # Order ~ log2(err_prev / err_curr)
            prev_err = results[i-1][1]
            if err > 1e-10:
                o = np.log2(prev_err / err)
                order = f"{o:.2f}"
            
        print(f"{n:<5} {err:<10.2f} {order:<10}")

if __name__ == "__main__":
    run_study()
