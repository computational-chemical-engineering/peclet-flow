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

def get_zick_homsy_k_sc(phi):
    # Simple interpolation (lookup table)
    phis = np.array([0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236])
    ks   = np.array([1.096, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1, 42.1])
    
    # Linear interp for now
    if phi < phis[0]: return 1.0 # Or Hasimoto
    return np.interp(phi, phis, ks)

def run_simulation(sdf_values, res_n, dx, R, L, phi_real, label, high_accuracy=True):
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

    rho = 0.1
    mu = 1.0
    f_mag = 1.0e-2
    solver.set_body_force(pnm_backend.float3(f_mag, 0, 0))
    solver.set_rho(rho)
    solver.set_mu(mu)

    if high_accuracy:
        # High-accuracy settings for mixed-precision solver
        dt = 50.0
        max_steps = 300
        solver.set_pressure_solver_params(max_iter=2000, tol=1e-11)
        solver.set_velocity_solver_params(max_iter=100, tol=1e-9)
        solver.set_diffusion_theta(1.0)  # Fully implicit
        solver.set_outer_iterations(6)
        solver.set_outer_tolerance(1e-9)
        conv_tol = 1e-9
        check_interval = 10
    else:
        # Original settings
        dt = 5.0
        max_steps = 500
        solver.set_pressure_solver_params(max_iter=200, tol=1e-9)
        solver.set_velocity_solver_params(max_iter=100, tol=1e-6)
        conv_tol = 1e-6
        check_interval = 100

    u_mean_history = []

    for i in range(max_steps):
        solver.step(dt)

        if i % check_interval == 0:
            u_field = np.array(solver.get_u())
            u_mean = np.mean(u_field)
            u_mean_history.append(u_mean)

            if len(u_mean_history) > 5:
                # Check convergence
                err = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-12)
                if err < conv_tol:
                    break

    u_field = np.array(solver.get_u())
    U_sup = np.mean(u_field)

    F_drag = f_mag * (L**3)
    K = F_drag / (6.0 * np.pi * mu * R * U_sup)

    return K

def run_sweep(res_n=64, high_accuracy=True):
    # Sweep phi from 0.05 to 0.5
    phis = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.5236]
    L = 1.0
    dx = L/res_n

    mode = "HIGH-ACCURACY" if high_accuracy else "STANDARD"
    print(f"\n{'='*60}")
    print(f"Periodic Sphere Array - Zick & Homsy Comparison")
    print(f"Resolution: {res_n}^3, Mode: {mode}")
    print(f"{'='*60}")
    print(f"{'Phi':<10} {'K_sim':<12} {'K_ref':<12} {'Error%':<10}")
    print("-" * 50)

    results = []

    for phi_target in phis:
        sdf, R = generate_sc_sdf(phi_target, res_n, L)

        k_sim = run_simulation(sdf, res_n, dx, R, L, phi_target,
                               f"SC phi={phi_target}", high_accuracy=high_accuracy)
        k_ref = get_zick_homsy_k_sc(phi_target)

        err = 100.0 * (k_sim - k_ref) / k_ref
        print(f"{phi_target:<10.4f} {k_sim:<12.4f} {k_ref:<12.4f} {err:<+10.2f}")
        results.append((phi_target, k_sim, k_ref, err))

    print("-" * 50)

    # Summary statistics
    errors = [abs(r[3]) for r in results]
    print(f"Mean |Error|: {np.mean(errors):.2f}%")
    print(f"Max  |Error|: {np.max(errors):.2f}%")
    print()

    return results

def run_convergence_test():
    """Test grid convergence at a single phi value."""
    phi_target = 0.20
    resolutions = [32, 48, 64]
    L = 1.0

    print(f"\n{'='*60}")
    print(f"Grid Convergence Test at phi={phi_target}")
    print(f"{'='*60}")

    k_ref = get_zick_homsy_k_sc(phi_target)
    print(f"Reference K (Zick & Homsy): {k_ref:.4f}")
    print()
    print(f"{'N':<8} {'K_sim':<12} {'Error%':<12} {'Order':<8}")
    print("-" * 45)

    results = []
    for res_n in resolutions:
        dx = L / res_n
        sdf, R = generate_sc_sdf(phi_target, res_n, L)
        k_sim = run_simulation(sdf, res_n, dx, R, L, phi_target,
                               f"phi={phi_target}", high_accuracy=True)
        err = 100.0 * (k_sim - k_ref) / k_ref

        order = "-"
        if len(results) > 0:
            prev_err = results[-1][2]
            prev_n = results[-1][0]
            if abs(err) > 1e-6 and abs(prev_err) > 1e-6:
                order = f"{np.log(abs(prev_err)/abs(err)) / np.log(res_n/prev_n):.2f}"

        print(f"{res_n:<8} {k_sim:<12.4f} {err:<+12.2f} {order:<8}")
        results.append((res_n, k_sim, err))

    print("-" * 45)
    return results


if __name__ == "__main__":
    # Run convergence test first
    run_convergence_test()

    # Then run full sweep at default resolution
    run_sweep(res_n=64, high_accuracy=True)
