import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Add build directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'build')))
import pnm_backend

def generate_slab_sdf(nx, ny, nz, L, slab_thickness):
    # Anisotropic resolution
    dx = L / nx
    dy = L / ny
    dz = L / nz
    
    x = np.linspace(0, L, nx, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, ny, endpoint=False) + 0.5 * dy
    z = np.linspace(0, L, nz, endpoint=False) + 0.5 * dz
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    yc = L/2
    dist = np.abs(Y - yc) - slab_thickness/2.0
    
    return dist.ravel(order='F').astype(np.float32), (dx, dy, dz)

def get_analytical_profile(y_coords, L, H, g, nu):
    u_prof = np.zeros_like(y_coords)
    U_max = (g * H**2) / (8.0 * nu)
    
    for i, y in enumerate(y_coords):
        if y < L/2:
            y_prime = y
        else:
            y_prime = y - L
            
        if abs(y_prime) <= H/2:
            u_prof[i] = U_max * (1.0 - (2.0 * y_prime / H)**2)
        else:
            u_prof[i] = 0.0
            
    return u_prof

def run_poiseuille(res_tuple, n_steps=5000):
    nx, ny, nz = res_tuple
    L = 1.0
    slab_thickness = 0.2
    H = L - slab_thickness
    
    sdf_values, spacing = generate_slab_sdf(nx, ny, nz, L, slab_thickness)
    dx, dy, dz = spacing
    
    sdf_data = pnm_backend.SDFData(
        sdf_values, 
        pnm_backend.int3(nx, ny, nz),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dy, dz)
    )
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(nx, ny, nz), 
        pnm_backend.float3(dx, dy, dz)
    )
    solver.initialize(sdf_data)
    
    rho = 1.0
    nu = 0.01
    g_x = 1.0 # Boosted force
    solver.set_body_force(pnm_backend.float3(g_x, 0, 0))
    # Timestep stability: CFL < 1, Diff < 0.5
    # dx ~ 0.015. dt < 0.015 / U. If U ~ 8, dt < 0.002.
    # Diff limit: dt < 0.5 * dx^2 / nu = 0.5 * 0.000225 / 0.01 = 0.01.
    # So CFL is stricter if U is large.
    # Let's try dt = 0.001 (safe).
    dt = 0.001 
    
    print(f"Running Grid={nx}x{ny}x{nz} with {n_steps} steps (dt={dt:.5f}, g={g_x})...")
    start_time = time.time()
    
    chunk_size = 1000
    for i in range(0, n_steps, chunk_size):
        steps_to_run = min(chunk_size, n_steps - i)
        solver.step(dt, rho, nu, steps_to_run, 1e-5)
        u = np.array(solver.get_u())
        max_u = np.max(u)
        print(f"  Step {i+steps_to_run}: Max U = {max_u:.6e}")
        
    print(f"  Done in {time.time() - start_time:.2f}s")
            
    # Extract Profile
    u_flat = np.array(solver.get_u())
    u_grid = u_flat.reshape((nx, ny, nz), order='F') # [x, y, z]
    
    ix = nx // 2
    iz = nz // 2
    u_profile = u_grid[ix, :, iz]
    
    y_coords = np.linspace(0, L, ny, endpoint=False) + 0.5 * dy
    u_ana = get_analytical_profile(y_coords, L, H, g_x, nu)
    
    return y_coords, u_profile, u_ana

def main():
    os.makedirs('output', exist_ok=True)
    res = (16, 64, 16)
    steps_list = [5000, 20000, 100000]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['r', 'g', 'b', 'c']
    
    first = True
    for i, n_steps in enumerate(steps_list):
        y, u_sim, u_ana = run_poiseuille(res, n_steps=n_steps)
        
        plt.plot(y, u_sim, 'o--', color=colors[i % len(colors)], label=f'Sim {n_steps} steps')
        
        if first:
            plt.plot(y, u_ana, 'k-', linewidth=2, label='Analytical')
            first = False
            
    plt.xlabel('y')
    plt.ylabel('Velocity u(y)')
    plt.title(f'Poiseuille Flow Convergence Investigation (Ny={res[1]})')
    plt.legend()
    plt.grid(True)
    output_path = 'output/investigation_ny64.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
