import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../cfd_utils')))
import pnm_backend
from vti import save_vti

def generate_sc_sdf(phi, res_n, L=1.0):
    """Generates Signed Distance Field for a Simple Cubic array of spheres."""
    # Use (Nx, Ny, Nz) shape with Fortran-order (x varies fastest) for consistency with user preference
    dx = L/res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    
    # indexing='ij' gives shape (Nx, Ny, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    R = (phi * 3.0 / (4.0 * np.pi))**(1.0/3.0) * L
    
    xc, yc, zc = L/2, L/2, L/2
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)
    sdf = dist - R
    
    # ravel(order='F') ensures we flatten (x,y,z) as x-fast, y-med, z-slow
    # which matches the C++ linear index: idx = z*ny*nx + y*nx + x
    return sdf.ravel(order='F').astype(np.float32), R, (dx, dx, dx)

def get_analytical_k(phi):
    """Returns reference Drag Factor K for Simple Cubic."""
    #phis = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50]) # Sangani & Acrivos (1982)
    #ks   = np.array([4.95, 6.10, 9.10, 13.6, 21.6, 39.5])
    phis = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236] # Zick & Homsy (1982)
    ks = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]
    
    if phi < phis[0]: return 1.0 # Dilute limit approximation
    return np.interp(phi, phis, ks)

def compute_velocity_at_centers(u, v, w):
    """Interpolate staggered velocities to cell centers and compute magnitude."""
    # u is at (i, j+0.5, k+0.5), interpolate to (i+0.5, j+0.5, k+0.5)
    u_c = 0.5 * (u + np.roll(u, -1, axis=0))
    # v is at (i+0.5, j, k+0.5), interpolate to (i+0.5, j+0.5, k+0.5)
    v_c = 0.5 * (v + np.roll(v, -1, axis=1))
    # w is at (i+0.5, j+0.5, k), interpolate to (i+0.5, j+0.5, k+0.5)
    w_c = 0.5 * (w + np.roll(w, -1, axis=2))

    return u_c, v_c, w_c

class SimulationRunner:
    def __init__(self, phi, res_n, L=1.0):
        self.phi = phi
        self.res_n = res_n
        self.L = L
        
        self.sdf_values, self.R, self.spacing_tuple = generate_sc_sdf(phi, res_n, L)
        self.dx, self.dy, self.dz = self.spacing_tuple
        
        # Setup Data Objects
        self.sdf_data = pnm_backend.SDFData(
            self.sdf_values, 
            pnm_backend.int3(res_n, res_n, res_n),
            pnm_backend.float3(0,0,0),
            pnm_backend.float3(self.dx, self.dy, self.dz)
        )
        self.solver = pnm_backend.CFDSolver(
            pnm_backend.int3(res_n, res_n, res_n), 
            pnm_backend.float3(self.dx, self.dy, self.dz)
        )
        self.solver.initialize(self.sdf_data)
        
        # Parameters (Low Re)
        self.rho = 0.0
        self.mu = 1.0
        self.f_mag = 1.0
        self.solver.set_body_force(pnm_backend.float3(self.f_mag, 0, 0))
        self.solver.set_rho(self.rho)
        self.solver.set_mu(self.mu)
        
        # Solver Parameters for Implicit Scheme
        self.solver.set_pressure_solver_params(50)
        self.solver.set_velocity_solver_params(2)
        #self.solver.set_diffusion_theta(1.0)
        self.solver.set_outer_iterations(800)
        self.solver.set_outer_tolerance(0.0)

    def run(self, dt=1.0, max_steps=100, save_output=False, filename=None):
        # Explicit limit is roughly 0.5*dx^2/nu ~ 0.5 * (1/64)^2 / 0.01 ~ 0.01
        # We use a much larger step to speed up convergence to steady state.
        
        u_mean_history = []
        
        print(f"Running phi={self.phi:.3f}, Res={self.res_n}^3, dt={dt:.4f}...")
        
        start_time = time.time()
        for i in range(max_steps):
            # Use Implicit Step
            self.solver.step(dt)
            
            if i % 50 == 0:
                u_field = np.array(self.solver.get_u())
                u_mean = np.mean(u_field)
                u_mean_history.append(u_mean)
                
                if len(u_mean_history) > 5:
                    # Relative change convergence check
                    if abs(u_mean_history[-1]) > 1e-12:
                        err = abs(u_mean_history[-1] - u_mean_history[-2]) / abs(u_mean_history[-1])
                        if err < 1e-5:
                            print(f"  Converged at step {i}. Mean Vel = {u_mean:.6e}")
                            break
        
        elapsed = time.time() - start_time
        u = np.array(self.solver.get_u())
        U_sup = np.mean(u)
        
        # Calculate K
        F = self.f_mag * (self.L**3)
        K = F / (6.0 * np.pi * self.mu * self.R * U_sup)
        
        print(f"  K_sim = {K:.4f} (Time: {elapsed:.2f}s)")
        
        u = u.reshape((self.res_n,self.res_n,self.res_n), order='F')
        p = np.array(self.solver.get_p()).reshape((self.res_n,self.res_n,self.res_n), order='F')
        if save_output and filename:            
            v = np.array(self.solver.get_v()).reshape((self.res_n,self.res_n,self.res_n), order='F')
            w = np.array(self.solver.get_w()).reshape((self.res_n,self.res_n,self.res_n), order='F')
            u_c, v_c, w_c = compute_velocity_at_centers(u, v, w)
            save_vti(filename, (self.res_n, self.res_n, self.res_n), (self.dx, self.dy, self.dz), u_c, v_c, w_c, p)

        p_prof_1d = np.mean(p, axis=(1,2))
        u_prof_1d = np.mean(u, axis=(1,2))

        return K, U_sup, p_prof_1d, u_prof_1d

def run_sweep():
    # 1. Define parameter space
    phis = [0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
    resolutions = [16, 32, 64, 128, 256]

    # Cartesian product (Grid Search)
    param_grid = list(product(phis, resolutions))

    results_list = []

    # 2. Run Loop with Progress Bar
    for phi, res in tqdm(param_grid, desc="Simulating"):
        
        # Initialize and Run
        simulator = SimulationRunner(phi, res) 
        k_sim, u_sup, p_prof_1d, u_prof_1d = simulator.run(
            save_output=True, 
            filename=f"output/phi_{phi:.2f}_N_{res}.vti", 
            dt=1.0, 
            max_steps=100
        )
        
        k_ref = get_analytical_k(phi)
        R = (phi * 3.0 / (4.0 * np.pi))**(1.0/3.0)
        
        results_list.append({
            'phi': phi,
            'R': R,
            'resolution': res,
            'k_sim': k_sim,
            'k_ref': k_ref,
            'error': (k_sim - k_ref) / k_ref,
            'p_profile': p_prof_1d,
            'u_profile': u_prof_1d
        })

    # 3. Create DataFrame
    df = pd.DataFrame(results_list)
    df.to_csv('output/drag_dimensionless_sc.csv', index=False)
    # --- Output: Table ---
    # Display table with background gradient (Pandas native feature)
    display_cols = ['phi', 'resolution', 'k_sim', 'k_ref', 'error_pct']
    display(df[display_cols].style.background_gradient(subset=['error_pct'], cmap='coolwarm', axis=None))


if __name__ == "__main__":
    # Run convergence test first
    run_sweep()
