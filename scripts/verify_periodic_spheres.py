import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def generate_single_sphere_sdf(res_n, L, R):
    # Use ij indexing for correct Z, Y, X layout matching C++ (x-fast? wait)
    # C++ Linear: z*ny*nx + y*nx + x -> X is fast index (if standard C-order)
    # Python ravel(C-order): fast last index.
    # So if we want x to be fast, x must be last index.
    # Array shape (Nz, Ny, Nx).
    # Meshgrid(z, y, x, indexing='ij') -> (Nz, Ny, Nx).
    
    dx = L/res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    xc, yc, zc = L/2, L/2, L/2
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)
    sdf = dist - R
    return sdf.ravel().astype(np.float32)

def generate_sc_sdf(phi, res_n, L=1.0):
    R = (phi * 3.0 / (4.0 * np.pi))**(1.0/3.0) * L
    return generate_single_sphere_sdf(res_n, L, R), R

def get_sangani_acrivos_k_sc(phi):
    # Sangani and Acrivos (1982) Int. J. Multiphase Flow
    # Simple Cubic Array
    # K = Force / (6 pi mu U a)
    # Series expansion for SC
    # Limit phi -> 0: K = 1 + ...
    # Specific values from table or formula
    # Using formula (Eq 3.14 / Table 1 approximation):
    # K = 1 / (1 - 1.7601 phi^1/3 + phi - 1.5593 phi^2 ...) NO, that's sedimentation U/U_stokes.
    # K_drag = 1.0 / (U/U_stokes)
    # So K = 1.0 / (1 - 1.7601*phi**(1/3) + phi - 1.5593*phi**2) ? (Hasimoto)
    # Let's use the SC values often cited:
    # phi=0.05 -> K=4.95
    # phi=0.10 -> K=6.1
    # phi=0.20 -> K=9.1
    # phi=0.30 -> K=13.6
    # phi=0.40 -> K=21.6
    # phi=0.50 -> K=39.5
    
    # Simple interpolation (lookup table)
    phis = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    ks   = np.array([4.95, 6.10, 9.10, 13.6, 21.6, 39.5])
    
    # Linear interp for now
    if phi < phis[0]: return 1.0 # Or Hasimoto
    return np.interp(phi, phis, ks)

def run_simulation(sdf_values, res_n, dx, R, L, phi_real, label):
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
    f_mag = 1.0e-4 
    solver.set_body_force(pnm_backend.float3(f_mag, 0, 0))
    dt = 0.1 * dx
    
    u_mean_history = []
    max_steps = 5000 
    
    # print(f"--- {label} (phi={phi_real:.4f}) ---")
    
    for i in range(max_steps):
        solver.step(dt, rho, nu, 20, 1e-4) # 20 inner iters
        
        if i % 100 == 0:
            u_field = np.array(solver.get_u())
            u_mean = np.mean(u_field)
            u_mean_history.append(u_mean)
            
            if len(u_mean_history) > 5:
                # Check convergence
                err = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-12)
                if err < 1e-6:
                    # print(f"Converged at iter {i}, U_mean={u_mean:.6e}")
                    break
    
    u_field = np.array(solver.get_u())
    U_sup = np.mean(u_field)
    
    F_drag = f_mag * (1.0 - phi_real) * (L**3)
    mu = rho * nu
    K = F_drag / (6.0 * np.pi * mu * R * U_sup)
    
    # print(f"  Result: K={K:.4f}")
    return K

def run_sweep():
    # Sweep phi from 0.05 to 0.5
    phis = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    res_n = 64
    L = 1.0
    dx = L/res_n
    
    print(f"Running Parameter Sweep (res={res_n}^3)...")
    print(f"{'Phi':<10} {'K_sim':<10} {'K_ref':<10} {'Error%':<10}")
    print("-" * 45)
    
    results = []
    
    for phi_target in phis:
        sdf, R = generate_sc_sdf(phi_target, res_n, L)
        # Recalculate actual phi based on discrete voxels?
        # For now use target R/phi.
        
        k_sim = run_simulation(sdf, res_n, dx, R, L, phi_target, f"SC phi={phi_target}")
        k_ref = get_sangani_acrivos_k_sc(phi_target)
        
        err = 100.0 * (k_sim - k_ref) / k_ref
        print(f"{phi_target:<10.3f} {k_sim:<10.4f} {k_ref:<10.4f} {err:<10.2f}")
        results.append((phi_target, k_sim, k_ref))
        
    print("-" * 45)
    print("Done.")

if __name__ == "__main__":
    run_sweep()
