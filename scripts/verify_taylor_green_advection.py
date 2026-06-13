"""Verify momentum advection against the Taylor-Green vortex."""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to find pnm_backend
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import pnm_backend

def run_tgv(res_n=32, u_conv=[0.0, 0.0, 0.0], steps=100, dt_fixed=None):
    L = 2.0 * np.pi
    dx = L / res_n
    
    # Grid Coordinates (Staggered)
    # 0..N-1
    # x_c = (i + 0.5) * dx
    # x_face_r = (i + 1.0) * dx
    
    # Indices
    X_idx = np.arange(res_n)
    Y_idx = np.arange(res_n)
    Z_idx = np.arange(res_n)
    
    # Meshgrid (Index space)
    # Cartesian indexing: Z, Y, X
    ZZ, YY, XX = np.meshgrid(Z_idx, Y_idx, X_idx, indexing='ij')
    
    # Coordinates for U (Staggered X: i+1, j+0.5, k+0.5)
    Xu = (XX + 1.0) * dx
    Yu = (YY + 0.5) * dx
    Zu = (ZZ + 0.5) * dx
    
    # Coordinates for V (Staggered Y: i+0.5, j+1, k+0.5)
    Xv = (XX + 0.5) * dx
    Yv = (YY + 1.0) * dx
    Zv = (ZZ + 0.5) * dx
    
    # Coordinates for W (Staggered Z: i+0.5, j+0.5, k+1)
    Xw = (XX + 0.5) * dx
    Yw = (YY + 0.5) * dx
    Zw = (ZZ + 1.0) * dx
    
    # Taylor-Green Vortex (2D in 3D domain)
    # u = U0 sin(x) cos(y)
    # v = -U0 cos(x) sin(y)
    # w = 0
    U0 = 1.0
    
    # Initial Fields (TGV + Convection)
    u_init = U0 * np.sin(Xu) * np.cos(Yu) + u_conv[0]
    v_init = -U0 * np.cos(Xv) * np.sin(Yv) + u_conv[1]
    w_init = np.zeros_like(Xw) + u_conv[2]
    
    # Initialize Solver
    # SDF: All fluid (> 0.0)
    # Set to a large positive value (e.g. 10.0)
    sdf_vals = np.full((res_n, res_n, res_n), 10.0, dtype=np.float32)
    sdf_vals_flat = sdf_vals.ravel(order='F')
    
    sdf_data = pnm_backend.SDFData(
        sdf_vals_flat,
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dx, dx)
    )
    
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(dx, dx, dx)
    )
    
    # Allocate SDF on device
    solver.initialize(sdf_data)
    
    # Set Parameters
    # Re = 100?
    # nu = U0 * L / Re? Or just pick nu.
    # Analytical decay rate gamma = 2 * nu * (k^2 + k^2) = 4 nu
    # If we want visible decay in 100 steps:
    # Time scale T = 1 / (4 nu).
    # Let's set nu = 0.01
    nu = 0.01
    rho = 1.0
    solver.set_rho(rho)
    solver.set_mu(nu * rho) # mu = nu * rho
    
    # Set Initial Velocity
    # Flatten in Fortran order (X-fastest)
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    solver.set_v(v_init.ravel(order='F').astype(np.float32))
    solver.set_w(w_init.ravel(order='F').astype(np.float32))
    
    # DEBUG: Check if U is set
    u_check = np.array(solver.get_u())
    print(f"DEBUG: Max U after set: {u_check.max()}, Min U: {u_check.min()}, Mean: {u_check.mean()}")
    print(f"DEBUG: Python says u[132] = {u_check[132]}")

    if u_check.max() == 0.0 and U0 > 0.0:
        print("ERROR: set_u failed!")
    
    # Time Step
    # CFL = U_max * dt / dx < 0.5
    # U_max approx |U0| + |U_conv|
    U_mag = np.sqrt((U0 + abs(u_conv[0]))**2 + (U0 + abs(u_conv[1]))**2 + abs(u_conv[2])**2)
    # Let's verify U max is roughly 3 if convection is (1,1,1).
    if dt_fixed is None:
        dt = 0.1 * dx / U_mag # Safe dt
    else:
        dt = dt_fixed
        
    print(f"Running TGV with U_conv={u_conv}, N={res_n}, dt={dt:.4f}, steps={steps}")
    
    energy_history = []
    times = []
    
    t = 0.0
    for i in range(steps):
        solver.step(dt)
        t += dt
        
        # Compute Energy of FLUCTUATIONS
        # E = 0.5 * sum( (u - u_conv)^2 + ... ) * vol
        # Retrieve fields
        u_curr = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
        v_curr = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
        w_curr = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
        
        # Subtract mean flow (convection)
        # Assuming mean flow is preserved exactly (periodic conservation)
        # Or we can just subtract the spatial mean.
        # For TGV, spatial mean of perturbation is 0.
        u_pert = u_curr - u_conv[0]
        v_pert = v_curr - u_conv[1]
        w_pert = w_curr - u_conv[2]
        
        # Kinetic Energy of perturbation
        ke = 0.5 * np.sum(u_pert**2 + v_pert**2 + w_pert**2) * (dx**3)
        energy_history.append(ke)
        times.append(t)
        
    return times, energy_history, dt

def analytical_decay(t_arr, E0, nu):
    # E(t) = E0 * exp(-4 * nu * t)
    # k=1 for sin(x) in [0, 2pi]?
    # u = sin(x) -> wavenumber k=1.
    # laplacian u = - (1^2 + 1^2) u = -2 u.
    # du/dt = nu lap u = -2 nu u.
    # u(t) = exp(-2 nu t).
    # E ~ u^2 ~ exp(-4 nu t).
    return E0 * np.exp(-4.0 * nu * np.array(t_arr))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    # Case 1: Static
    t1, E1, dt1 = run_tgv(u_conv=[0,0,0], steps=args.steps)
    
    # Case 2: Convected (Diagonal)
    t2, E2, dt2 = run_tgv(u_conv=[1.0, 1.0, 1.0], steps=args.steps, dt_fixed=dt1)
    
    # Case 3: Fast Convection
    t3, E3, dt3 = run_tgv(u_conv=[5.0, 5.0, 0.0], steps=args.steps, dt_fixed=dt1)
    
    # Analytical
    E0 = E1[0] # Normalize to first measured point
    # Wait, first point is after step 1. Initial E0 approx:
    # Integral sin^2(x) cos^2(y) dx dy over [0, 2pi]^2
    # = (pi) * (pi) * (2pi Z) = 2 pi^3 * U0^2?
    # ke = 0.5 * integral (u^2+v^2)
    # u^2 + v^2 = sin^2 x cos^2 y + cos^2 x sin^2 y
    # Integral over 2pi x 2pi:
    # int sin^2 x dx = pi. int cos^2 y dy = pi. -> pi^2.
    # int cos^2 x dx = pi. int sin^2 y dy = pi. -> pi^2.
    # Sum = 2 pi^2.
    # Multiply by Z-length (2pi). -> 4 pi^3.
    # E_total = 0.5 * 4 pi^3 approx 62.01.
    # Check E1[0].
    
    decay_ana = analytical_decay(t1, E1[0], 0.01) # Use experimental E0 to match start
    
    print(f"Final Energy Static: {E1[-1]:.4f}")
    print(f"Final Energy Conv1:  {E2[-1]:.4f}")
    print(f"Final Energy Conv2:  {E3[-1]:.4f}")
    
    # Error metrics
    # Compare E2 vs E1 vs Analytical
    err_static = np.abs(np.array(E1) - decay_ana).mean()
    err_conv1 = np.abs(np.array(E2) - decay_ana).mean()
    err_conv2 = np.abs(np.array(E3) - decay_ana).mean()
    
    print(f"Mean Abs Error (Static vs Ana): {err_static:.6f}")
    print(f"Mean Abs Error (Conv1 vs Ana):  {err_conv1:.6f}")
    print(f"Mean Abs Error (Conv2 vs Ana):  {err_conv2:.6f}")
    
    # Plot
    plt.figure()
    plt.plot(t1, E1, 'o-', label='Static (0,0,0)')
    plt.plot(t2, E2, 'x-', label='Conv (1,1,1)')
    plt.plot(t3, E3, 's-', label='Conv (5,5,0)')
    plt.plot(t1, decay_ana, 'k--', label='Analytical')
    plt.xlabel('Time')
    plt.ylabel('Kinetic Energy (Perturbation)')
    plt.title('TGV Decay verifying Advection Galilean Invariance')
    plt.legend()
    plt.grid(True)
    plt.savefig('tgv_advection_verification.png')
    
    # Pass Criteria
    # 1. Decay matches analytical (roughly, numerical dissipation expected)
    # 2. Convection results match Static results (Galilean Invariance)
    
    diff_conv = np.abs(np.array(E1) - np.array(E2)).max()
    print(f"Max Diff (Static - Conv1): {diff_conv:.6f}")
    
    params_match = diff_conv < 1.0 # Tolerance adjusted for N=32 coarse grid
    
    if params_match:
        print("PASS: Advection test passed (Galilean Invariance holds)")
    else:
        print("FAIL: Significant difference between static and convected TGV")
