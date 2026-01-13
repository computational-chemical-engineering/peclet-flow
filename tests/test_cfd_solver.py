import sys
import os
import numpy as np

# Add build directory to path to find the module
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import pnm_backend
    print("Successfully imported pnm_backend")
except ImportError as e:
    print(f"Failed to import pnm_backend: {e}")
    sys.exit(1)

def test_periodic_channel_flow():
    """
    Test a simple periodic channel flow driven by a body force.
    """
    nx, ny, nz = 32, 32, 32
    res = pnm_backend.int3(nx, ny, nz)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    
    # Create Solver
    solver = pnm_backend.CFDSolver(res, spacing)
    
    # Apply Body Force in X direction
    force_x = 1.0
    solver.set_body_force(pnm_backend.float3(force_x, 0.0, 0.0))
    
    # Run Simulation
    # Re=100? No viscosity yet (Inviscid). 
    # Velocity should increase linearly v = F*t if no advection and no pressure gradient opposes it.
    # But pressure gradient will likely not oppose it in periodic x-direction unless we block it (but it's empty space).
    # So u should grow indefinitely.
    
    dt = 0.1
    rho = 1.0
    steps = 10
    
    print(f"Running {steps} steps with Fx={force_x}...")
    
    solver.set_rho(rho)
    solver.set_mu(0.0)
    solver.set_pressure_solver_params(iter=10)
    
    for i in range(steps):
        solver.step(dt)
        
    u = np.array(solver.get_u()).reshape((nx, ny, nz), order='F')
    
    mean_u = np.mean(u)
    expected_u = force_x * dt * steps # f = ma => a = f/rho (rho=1) => v = a*t
    
    print(f"Mean U: {mean_u:.4f}")
    print(f"Expected U (approx): {expected_u:.4f}")
    
    if np.abs(mean_u - expected_u) < 1e-2:
        print("PASS: Velocity matches acceleration from body force.")
    else:
        print("FAIL: Velocity mismatch.")

    # Check Divergence free (should be 0 for uniform flow)
    # The solver projects it.
    
def test_cavity_like_setup():
    # Not actually lid driven since we don't support BCs yet (only periodic),
    # but we can test if the solver runs without crashing on random init?
    pass

if __name__ == "__main__":
    test_periodic_channel_flow()
