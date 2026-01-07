
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import pnm_backend

def run_simulation(dt, T, res_x=32):
    res = pnm_backend.int3(res_x, res_x, 3) 
    spacing = pnm_backend.float3(1.0/res_x, 1.0/res_x, 1.0/res_x)
    solver = pnm_backend.CFDSolver(res, spacing)
    
    # Create Channel SDF (Walls at y=0.1 and y=0.9)
    # y coords: (0.5 to N-0.5) * h
    y_coords = np.linspace(0.5*spacing.y, 1.0-0.5*spacing.y, res_x)
    X, Y, Z = np.meshgrid(y_coords, y_coords, np.array([0.5, 1.5, 2.5])*spacing.z, indexing='ij')
    
    # Walls at y=0.1 and y=0.9
    # SDF positive inside channel
    # Dist to bottom wall: y - 0.2
    # Dist to top wall: 0.8 - y
    # Channel from 0.2 to 0.8 width 0.6
    sdf = np.minimum(Y - 0.2, 0.8 - Y).flatten().astype(np.float32)
    
    solver.initialize(pnm_backend.SDFData(sdf, res, pnm_backend.float3(0,0,0), spacing))
    
    solver.set_rho(1.0)
    solver.set_mu(0.01)
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0)) # Force in X
    
    # Use Newton Solver (Picard)
    # solver.set_outer_iterations(5) # Deprecated/Unused
    solver.set_velocity_solver_params(20, 1e-5) # Picard Iterations
    solver.set_pressure_solver_params(50, 1e-5)
    
    num_steps = int(T / dt)
    print(f"Running DT={dt}, Steps={num_steps}")
    
    for i in range(num_steps):
        solver.step(dt)
        
    return np.array(solver.get_u()).reshape((3, res_x, res_x)).transpose(2,1,0) # X, Y, Z

# Parameters
T_final = 1.0
dts = [0.1, 0.05, 0.025, 0.0125]
errors = []
velocities = []

# Reference run (finest)
print("Running Reference...")
u_ref = run_simulation(dts[-1], T_final)
center_y = int(32/2)
v_ref_max = np.max(u_ref[:, center_y, 1]) # Max velocity in channel center
print(f"Reference Max Vel: {v_ref_max}")

# Compare
for dt in dts[:-1]:
    u = run_simulation(dt, T_final)
    v_max = np.max(u[:, center_y, 1])
    # Compute L2 error relative to reference (interpolated? No, same grid)
    # Just compare grid values directly
    diff = np.abs(u - u_ref)
    l2_err = np.linalg.norm(diff) / np.linalg.norm(u_ref)
    
    print(f"DT: {dt}, Max Vel: {v_max}, L2 Error: {l2_err}")
    errors.append(l2_err)

# Plot
plt.figure()
plt.loglog(dts[:-1], errors, '-o', label='Error')
plt.loglog(dts[:-1], [e * (dts[i]/dts[0]) for i,e in enumerate(errors)], '--', label='1st Order')
plt.xlabel('dt')
plt.ylabel('L2 Relative Error')
plt.title('Time Step Convergence (Newton Solver)')
plt.legend()
plt.grid(True)
plt.savefig('convergence_plot.png')
print("Saved convergence_plot.png")
