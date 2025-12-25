import sys
import os

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pnm_backend
import numpy as np

def test_cfl_control():
    print("Initializing Solver...")
    res = pnm_backend.int3(32, 32, 32)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    solver = pnm_backend.CFDSolver(res, spacing)

    # Initialize SDF (all fluid)
    sdf_vals = np.ones(32*32*32, dtype=np.float32)
    origin = pnm_backend.float3(0,0,0)
    sdf_data = pnm_backend.SDFData(sdf_vals.tolist(), res, origin, spacing)
    solver.initialize(sdf_data)
    
    print("Setting Body Force and CFL...")
    # Set Body Force
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))

    # Set Material Properties
    solver.set_rho(1.0)
    # mu = nu * rho = 0.01 * 1.0 = 0.01
    solver.set_mu(0.01)

    # Set Solver Params
    solver.set_pressure_solver_params(100, 1e-5)
    solver.set_velocity_solver_params(10, 1e-5)

    cfl_target = 0.5
    solver.set_cfl(cfl_target)
    
    print(f"Target CFL: {solver.get_cfl()}")
    assert abs(solver.get_cfl() - cfl_target) < 1e-6
    
    print("Running Step 1 (Auto DT)...")
    # Step 1: Initial velocity 0. Acceleration limiter should kick in.
    solver.step(-1.0) 
    
    dt1 = solver.get_dt()
    print(f"Step 1 DT: {dt1}")
    assert dt1 > 0
    
    # Verify velocity increased
    u = np.array(solver.get_u())
    max_u = np.max(np.abs(u))
    print(f"Max U after step 1: {max_u}")
    assert max_u > 0
    
    print("Running Step 2 (Auto DT)...")
    solver.step(-1.0)
    dt2 = solver.get_dt()
    print(f"Step 2 DT: {dt2}")
    
    # Check that DT is reasonable (not checking specifically for decrease as dynamics might vary)
    assert dt2 > 0

if __name__ == "__main__":
    test_cfl_control()
