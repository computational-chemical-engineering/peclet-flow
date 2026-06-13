"""Unit test for the pressure-projection operator."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add build directory to path
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import pnm_backend
except ImportError:
    print("Error: Could not import pnm_backend. Make sure you are running from the project root and build exists.")
    sys.exit(1)

def compute_divergence(u, v, w, dx):
    """
    Compute divergence of staggered velocity field.
    u: (nz, ny, nx) -- Defined at x=i (left faces) in simulation.
       Divergence at cell center (i,j,k) = (u_{i+1} - u_i + ...)
    """
    div = np.zeros_like(u)
    
    # Simple divergence for staggered grid
    # div[k,j,i] = (u[i+1] - u[i])/dx + (v[j+1] - v[j])/dy + (w[k+1] - w[k])/dz
    
    # We need to handle periodicity or boundary. 
    # Simulation uses periodic sample field in divergence calculation.
    # Let's assume standard internal cells for this check.
    
    # Shift arrays to get u_{i+1}
    u_p1 = np.roll(u, -1, axis=2) # Axis 2 is X
    v_p1 = np.roll(v, -1, axis=1) # Axis 1 is Y
    w_p1 = np.roll(w, -1, axis=0) # Axis 0 is Z
    
    div = (u_p1 - u) + (v_p1 - v) + (w_p1 - w)
    return div / dx

def run_test():
    res = pnm_backend.int3(8, 8, 8)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    
    solver = pnm_backend.CFDSolver(res, spacing)
    
    # Create SDF: All fluid
    # Size needs to match: flattened size
    # But wait, initialize takes SDFData object
    
    total_elements = 8 * 8 * 8
    sdf_values = [1.0] * total_elements # All fluid > 0
    
    origin = pnm_backend.float3(0,0,0)
    sdf_data = pnm_backend.SDFData(sdf_values, res, origin, spacing)
    
    solver.initialize(sdf_data)
    
    # Initialize Random Velocity
    np.random.seed(42)
    u_init = np.random.rand(total_elements).astype(np.float32) - 0.5
    v_init = np.random.rand(total_elements).astype(np.float32) - 0.5
    w_init = np.random.rand(total_elements).astype(np.float32) - 0.5
    
    solver.set_u(u_init)
    solver.set_v(v_init)
    solver.set_w(w_init)
    
    # Reshape for analysis
    u_grid_init = u_init.reshape((8, 8, 8))
    v_grid_init = v_init.reshape((8, 8, 8))
    w_grid_init = w_init.reshape((8, 8, 8))
    
    div_init = compute_divergence(u_grid_init, v_grid_init, w_grid_init, 1.0)
    
    print(f"Initial Max Divergence: {np.max(np.abs(div_init)):.6f}")
    print(f"Initial Mean Divergence: {np.mean(np.abs(div_init)):.6f}")
    
    # Project
    dt = 0.1
    solver.set_rho(1.0)
    solver.set_pressure_solver_params(iter=1000) # Ensure tight convergence
    solver.project(dt)
    
    # Get Results
    u_final = np.array(solver.get_u()).reshape((8, 8, 8))
    v_final = np.array(solver.get_v()).reshape((8, 8, 8))
    w_final = np.array(solver.get_w()).reshape((8, 8, 8))
    p_final = np.array(solver.get_p()).reshape((8, 8, 8))

    print(f"Pressure Stats: Min {np.min(p_final):.4f}, Max {np.max(p_final):.4f}, Mean {np.mean(p_final):.4f}")
    
    div_final = compute_divergence(u_final, v_final, w_final, 1.0)
    
    max_div = np.max(np.abs(div_final))
    mean_div = np.mean(np.abs(div_final))
    print(f"\nFinal Max Divergence: {max_div:.6e}")
    print(f"Final Mean Divergence: {mean_div:.6e}")

    # Exclude pinned pressure cell from diagnostics
    div_masked = div_final.copy()
    div_masked[0, 0, 0] = 0.0
    max_div_masked = np.max(np.abs(div_masked))
    mean_div_masked = np.mean(np.abs(div_masked))
    print(f"Final Max Divergence (masked pin): {max_div_masked:.6e}")
    print(f"Final Mean Divergence (masked pin): {mean_div_masked:.6e}")
    
    max_loc = np.unravel_index(np.argmax(np.abs(div_final)), div_final.shape)
    print(f"Location of Max Divergence: {max_loc}")
    
    sum_div = np.sum(div_final)
    print(f"Sum of Divergence (Global Mass Check): {sum_div:.6e}")

    sum_div_init = np.sum(div_init)
    print(f"Initial Sum of Divergence: {sum_div_init:.6e}")
    
    # Analysis
    if max_div_masked < 1e-4:
        print("\n[PASS] Divergence is effectively zero.")
    else:
        print("\n[FAIL] Divergence is too high.")
        
    # Energy Analysis
    ke_init = 0.5 * np.sum(u_grid_init**2 + v_grid_init**2 + w_grid_init**2)
    ke_final = 0.5 * np.sum(u_final**2 + v_final**2 + w_final**2)
    print(f"\nKinetic Energy: {ke_init:.4f} -> {ke_final:.4f}")
    print("Kinetic Energy should decrease (projection onto divergence-free subspace removes energy).")

if __name__ == "__main__":
    run_test()
