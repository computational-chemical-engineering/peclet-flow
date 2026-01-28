import sys
import os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../cfd_utils')))
import pnm_backend
from vti import save_vti

def compute_velocity_at_centers(u, v, w):
    """
    Interpolate staggered velocities to cell centers.
    Input shapes: (Nz, Ny, Nx)
    """
    # u is at (i, j+0.5, k+0.5) [Face X], interpolate along X-axis
    u_c = 0.5 * (u + np.roll(u, -1, axis=2))
    # v is at (i+0.5, j, k+0.5) [Face Y], interpolate along Y-axis
    v_c = 0.5 * (v + np.roll(v, -1, axis=1))
    # w is at (i+0.5, j+0.5, k) [Face Z], interpolate along Z-axis
    w_c = 0.5 * (w + np.roll(w, -1, axis=0))

    return u_c, v_c, w_c

class SimulationRunner:
    def __init__(self, filename_sdf):
        print(f"Loading geometry: {filename_sdf}")
        # New binding returns: (sdf_3d, origin, spacing) in ZYX order
        self.sdf_3d, self.origin, self.spacing = pnm_backend.SDFReader.read_vti(filename_sdf)
        
        # sdf_3d.shape is (Nz, Ny, Nx)
        self.shape = self.sdf_3d.shape

        # Instantiate solver using the shape directly
        # Updated binding: CFDSolver(shape_tuple, spacing_tuple)
        self.solver = pnm_backend.CFDSolver(self.shape, self.spacing)
        
        # Initialize with the 3D array and metadata
        self.solver.initialize(self.sdf_3d, self.origin, self.spacing)
        
        # Physics Parameters (Stokes flow regime)
        self.rho = 0.0
        self.mu = 1.0
        self.f_mag = 1.0
        
        # Use Pythonic set methods
        self.solver.set_body_force(pnm_backend.float3(self.f_mag, 0, 0))
        self.solver.set_rho(self.rho)
        self.solver.set_mu(self.mu)
        
        # Solver Settings
        self.solver.set_pressure_solver_params(50)
        self.solver.set_velocity_solver_params(2)
        self.solver.set_outer_iterations(800) # We control outer loop in Python
        self.solver.set_outer_tolerance(0.0)

    def run(self, dt=1.0, max_steps=100, save_output=False, filename=None):
        u_mean_history = []
        start_time = time.time()
        
        print("Starting Simulation Loop...")
        for i in range(max_steps):
            self.solver.step(dt)
            
            if i % 10 == 0:
                # get_u() now returns a 3D (Nz, Ny, Nx) array automatically
                u_field = self.solver.get_u()
                u_mean = np.mean(u_field)
                u_mean_history.append(u_mean)
                
                if len(u_mean_history) > 2:
                    err = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-15)
                    if err < 1e-8:
                        print(f"  Converged at step {i}. Mean Vel = {u_mean:.6e}")
                        break

        elapsed = time.time() - start_time
        print(f"Simulation finished in {elapsed:.2f}s")

        # Gather final 3D fields (already shaped as Nz, Ny, Nx)
        u = self.solver.get_u()
        v = self.solver.get_v()
        w = self.solver.get_w()
        p = self.solver.get_p()
        
        if save_output and filename:
            u_c, v_c, w_c = compute_velocity_at_centers(u, v, w)
            # Stack for vector representation (Nz, Ny, Nx, 3)
            vel_vec = np.stack([u_c, v_c, w_c], axis=-1)
            
            fields = {
                "Velocity": vel_vec,
                "Pressure": p,
                "SDF": self.sdf_3d
            }
            save_vti(filename, fields, self.spacing, self.origin)

        # Profiles (averaging over Y and X planes to get Z-profile)
        p_prof_1d = np.mean(p, axis=(1, 2))
        u_prof_1d = np.mean(u, axis=(1, 2))

        return np.mean(u), p_prof_1d, u_prof_1d

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    
    filename_sdf = "ring_packing_sdf.vti"
    if not os.path.exists(filename_sdf):
        print(f"Error: {filename_sdf} not found.")
    else:
        simulator = SimulationRunner(filename_sdf)
        u_sup, p_prof, u_prof = simulator.run(
            save_output=True, 
            filename="output/ring_packing_flow.vti", 
            dt=1.0, 
            max_steps=100
        )
        print(f"Final Superficial Velocity: {u_sup:.6e}")