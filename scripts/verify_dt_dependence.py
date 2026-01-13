import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

def run_simulation(dt_val, phi=0.05):
    N = 32
    res = pnm_backend.int3(N, N, N)
    spacing = pnm_backend.float3(1.0/N, 1.0/N, 1.0/N)
    
    R = (3 * phi / (4 * np.pi))**(1/3)
    sdf = generate_sc_sdf((N, N, N), R, (1.0/N, 1.0/N, 1.0/N))
    
    solver = pnm_backend.CFDSolver(res, spacing)
    origin = pnm_backend.float3(0.0, 0.0, 0.0)
    solver.initialize(pnm_backend.SDFData(sdf, res, origin, spacing))
    
    # Stable parameters
    rho = 1.0
    mu = 1.0
    solver.set_rho(rho)
    solver.set_mu(mu)
    
    body_force = 10.0
    solver.set_body_force(pnm_backend.float3(body_force, 0.0, 0.0))
    
    solver.set_pressure_solver_params(iter=2000)
    solver.set_velocity_solver_params(iter=100)
    
    # Run to steady state
    # Max steps adjusted for dt to ensure roughly same physical time T=10.0
    max_steps = int(10.0 / dt_val) + 10
    
    for i in range(max_steps):
        solver.step(dt_val)
        
    u_field = np.array(solver.get_u())
    u_avg = np.mean(u_field)
    
    if u_avg < 1e-9: return 0.0
    
    F_drag = body_force * (1.0 - phi)
    K = F_drag / (6 * np.pi * mu * R * u_avg)
    return K

def verify_dt_dependence():
    dts = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    ks = []
    
    print(f"{'dt':<10} {'K':<10}")
    print("-" * 20)
    
    for dt in dts:
        k = run_simulation(dt)
        ks.append(k)
        print(f"{dt:<10.2f} {k:<10.4f}")
        
    # Analyze slope
    # Error ~ O(dt) or O(dt^2)
    # If standard projection, Error ~ dt. K = K_true + C * dt
    # If incremental, Error ~ dt^2 (for velocity).
    
    plt.figure()
    plt.plot(dts, ks, 'o-')
    plt.xlabel('dt')
    plt.ylabel('K')
    plt.title('Drag Factor K vs Timestep dt')
    plt.grid(True)
    plt.savefig('dt_dependence.png')
    print("Saved dt_dependence.png")

if __name__ == "__main__":
    verify_dt_dependence()
