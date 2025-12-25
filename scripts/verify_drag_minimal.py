import sys
import os
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import pnm_backend

def generate_sc_sdf(res, radius, spacing):
    x = np.linspace(0.5 * spacing[0], res[0] * spacing[0] - 0.5 * spacing[0], res[0])
    y = np.linspace(0.5 * spacing[1], res[1] * spacing[1] - 0.5 * spacing[1], res[1])
    z = np.linspace(0.5 * spacing[2], res[2] * spacing[2] - 0.5 * spacing[2], res[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Sphere at center
    cx, cy, cz = res[0] * spacing[0] / 2, res[1] * spacing[1] / 2, res[2] * spacing[2] / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
    sdf = dist - radius
    
    # Invert for fluid (Fluid > 0, Solid < 0 in this solver? )
    # CFDSolver expects positive SDF for fluid.
    # dist - radius: Positive OUTSIDE sphere. Negative INSIDE sphere.
    # So Fluid is OUTSIDE. Correct.
    
    return sdf.flatten(order='F').astype(np.float32)

def verify_drag():
    N = 32
    res = pnm_backend.int3(N, N, N)
    spacing = pnm_backend.float3(1.0/N, 1.0/N, 1.0/N) # L=1.0
    
    # Parameters
    phi = 0.05
    # R^3 = 3 phi / (4 pi)
    R = (3 * phi / (4 * np.pi))**(1/3)
    print(f"Phi: {phi}, Radius: {R}")
    
    sdf = generate_sc_sdf((N, N, N), R, (1.0/N, 1.0/N, 1.0/N))
    
    solver = pnm_backend.CFDSolver(res, spacing)
    origin = pnm_backend.float3(0.0, 0.0, 0.0)
    solver.initialize(pnm_backend.SDFData(sdf, res, origin, spacing))
    
    # Parameters (Scaled for Stability + Low Re)
    # rho=1.0 to avoid 1/rho splitting error
    # mu=10.0 to keep Re low (~0.01) despite higher rho
    solver.set_rho(1.0)
    mu = 10.0
    solver.set_mu(mu)
    
    # Body Force
    body_force = 1.0 # Resulting U should be ~0.1/10 ~ 0.01 => Re ~ 0.001
    solver.set_body_force(pnm_backend.float3(body_force, 0.0, 0.0))
    
    solver.set_pressure_solver_params(2000, 1e-6) # Tight tolerance
    solver.set_velocity_solver_params(100, 1e-6)
    
    dt = 0.05 # Safe timestep for convergence
    
    print("Running simulation (Scaled Params: rho=1, mu=10, dt=0.05)...")
    for i in range(2000): # More steps for small dt
        solver.step_implicit(dt)
        if i % 20 == 0:
            u = np.array(solver.get_u())
            print(f"Step {i}: Max U = {np.max(u):.4f}, Mean U = {np.mean(u):.4f}")
            
    u_field = np.array(solver.get_u())
    u_avg = np.mean(u_field)
    
    print(f"Final U_avg: {u_avg}")
    
    if u_avg < 1e-9:
        print("Velocity too low!")
        return

    F_drag = body_force * (1.0 - phi) # Force balancing
    K = F_drag / (6 * np.pi * mu * R * u_avg)
    
    print(f"Calculated K: {K:.4f}")
    
    # Zick & Homsy approx for phi=0.05 SC
    # Expected K approx 2 to 3.
    # If splitting error was present, velocity would be much higher, so K much lower.
    
if __name__ == "__main__":
    verify_drag()
