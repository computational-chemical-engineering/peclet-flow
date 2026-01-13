import sys
import os
import numpy as np
import time

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import pnm_backend

def generate_sc_sdf(res, radius, spacing):
    x = np.linspace(0.5 * spacing[0], res[0] * spacing[0] - 0.5 * spacing[0], res[0])
    y = np.linspace(0.5 * spacing[1], res[1] * spacing[1] - 0.5 * spacing[1], res[1])
    z = np.linspace(0.5 * spacing[2], res[2] * spacing[2] - 0.5 * spacing[2], res[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cx, cy, cz = res[0] * spacing[0] / 2, res[1] * spacing[1] / 2, res[2] * spacing[2] / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
    sdf = dist - radius
    return sdf.flatten(order='F').astype(np.float32)

def run_simulation(dt_val, rho=1.0, mu=1.0, outer_iters=1):
    N = 32
    res = pnm_backend.int3(N, N, N)
    spacing = pnm_backend.float3(1.0/N, 1.0/N, 1.0/N)
    
    phi_solid = 0.05
    R = (3 * phi_solid / (4 * np.pi))**(1/3)
    sdf = generate_sc_sdf((N, N, N), R, (1.0/N, 1.0/N, 1.0/N))
    
    solver = pnm_backend.CFDSolver(res, spacing)
    origin = pnm_backend.float3(0.0, 0.0, 0.0)
    solver.initialize(pnm_backend.SDFData(sdf, res, origin, spacing))
    
    solver.set_rho(rho)
    solver.set_mu(mu)
    
    body_force = 10.0
    solver.set_body_force(pnm_backend.float3(body_force, 0.0, 0.0))
    
    # We use sufficient iterations for large dt (stiff Poisson)
    solver.set_pressure_solver_params(iter=2000)
    solver.set_velocity_solver_params(iter=100)
    
    # Set Outer Iterations
    try:
        solver.set_outer_iterations(outer_iters)
    except AttributeError:
        print("Error: set_outer_iterations not exposed in binding!")
        return 0.0

    print(f"Running dt={dt_val}, OuterIters={outer_iters} ...")
    
    # Run slightly fewer steps since outer loop adds cost
    if outer_iters > 1:
        # If iterating, we expect convergence faster in 'time'
        # Total compute = TimeSteps * OuterIters * InnerIters
        max_steps = 100 
    else:
        # Standard splitting requires more steps to reach T=final
        max_steps = 200

    start_t = time.time()
    for i in range(max_steps):
        solver.step_implicit(dt_val)
    end_t = time.time()
    
    u_field = np.array(solver.get_u())
    u_avg = np.mean(u_field)
    
    if u_avg < 1e-9: 
        print(f"  Failed: U_avg={u_avg}")
        return 0.0
    
    F_drag = body_force * (1.0 - phi_solid)
    K = F_drag / (6 * np.pi * mu * R * u_avg)
    
    print(f"  Time: {end_t - start_t:.2f}s | U_avg: {u_avg:.5f} | K: {K:.4f}")
    return K

def verify_dt_independence():
    print("Verifying Timestep Independence (Comparing dt=1.0 vs dt=2.0 with SIMPLE)...")
    
    # 1. Test: dt=1.0, SIMPLE (20 iters)
    k_dt1 = run_simulation(dt_val=1.0, outer_iters=20)
    
    # 2. Test: dt=2.0, SIMPLE (20 iters)
    k_dt2 = run_simulation(dt_val=2.0, outer_iters=20)
    
    # 3. Reference for context
    k_ref = 2.37
    
    print("\n---------------------------------------------------")
    print(f"SIMPLE (dt=1.0): K = {k_dt1:.4f}")
    print(f"SIMPLE (dt=2.0): K = {k_dt2:.4f}")
    print(f"Low-dt Ref     : K = {k_ref:.4f}")
    print("---------------------------------------------------")
    
    # Check consistency between dt=1.0 and dt=2.0
    diff = abs(k_dt1 - k_dt2)
    rel_diff = diff / k_dt1
    
    print(f"Difference: {diff:.4f} ({rel_diff*100:.2f}%)")
    
    if rel_diff < 0.01: # < 1% difference
        print("\nSUCCESS: Results are timestep independent!")
    else:
        print("\nFAILURE: Significant dependence remains.")

if __name__ == "__main__":
    verify_dt_independence()
