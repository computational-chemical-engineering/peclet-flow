"""
High-Accuracy Angled Poiseuille Flow Verification

Tests whether residuals far below 1e-8 can be achieved with the
mixed-precision Newton-Raphson solver.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend


def generate_angled_slab_sdf(res_n, L, slab_thickness):
    """
    Generate SDF for a slab at 45 degrees.
    """
    dx = L / res_n
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    sqrt2 = np.sqrt(2.0)
    d = (Y - X) / sqrt2
    period = L / sqrt2
    d_wrapped = d - period * np.round(d / period)
    sdf = np.abs(d_wrapped) - slab_thickness / 2.0

    return sdf.ravel(order='F').astype(np.float32)


def run_high_accuracy_test(res_n, L=1.0, slab_thickness=0.2):
    """Run simulation targeting very low residuals."""

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n
    period = L / sqrt2
    H = period - slab_thickness

    print(f"\n{'='*70}")
    print(f"High-Accuracy Test: N={res_n}")
    print(f"{'='*70}")
    print(f"  Grid spacing: dx = {dx:.6e}")
    print(f"  Channel height: H = {H:.6f}")

    # Generate SDF
    sdf_values = generate_angled_slab_sdf(res_n, L, slab_thickness)

    # Create solver
    sdf_data = pnm_backend.SDFData(
        sdf_values,
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0, 0, 0),
        pnm_backend.float3(dx, dx, dx)
    )
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(sdf_data)

    # Physical parameters
    rho = 1.0
    mu = 0.01

    # Body force magnitude and direction (parallel to wall)
    f_mag = 1.0e-4
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # HIGH ACCURACY SETTINGS
    # Increased iterations and tighter tolerances
    pressure_max_iter = 5000
    velocity_max_iter = 100
    theta = 1.0  # Fully implicit for stability

    solver.set_pressure_solver_params(iter=pressure_max_iter)
    solver.set_velocity_solver_params(iter=velocity_max_iter)
    solver.set_diffusion_theta(theta)
    solver.set_outer_iterations(8)  # More Newton iterations
    solver.set_outer_tolerance(1e-10)

    # Analytical solution
    U_ana_max = (f_mag * H**2) / (8.0 * mu)
    print(f"  Analytical U_max = {U_ana_max:.10e}")
    print(f"  Pressure solver: max_iter={pressure_max_iter}")
    print(f"  Velocity solver: max_iter={velocity_max_iter}")

    # Use large dt for steady state (implicit scheme)
    dt = 100.0
    max_steps = 200
    target_residual = 1e-12

    print(f"  dt = {dt}, max_steps = {max_steps}")
    print(f"  Target residual: {target_residual:.0e}")
    print()

    residual_history = []

    print(f"{'Step':<8} {'NS_res_max':<15} {'Div_max':<15} {'U_max':<15} {'Error%':<12}")
    print("-" * 65)

    for step in range(max_steps):
        solver.step(dt)

        # Get current fields
        u_curr = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
        v_curr = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
        w_curr = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')

        residual = solver.get_momentum_residual_max()
        div_max = solver.get_divergence_max(dt)
        residual_history.append(residual)

        # Compute velocity in flow direction
        u_center = 0.5 * (u_curr + np.roll(u_curr, -1, axis=0))
        v_center = 0.5 * (v_curr + np.roll(v_curr, -1, axis=1))
        u_parallel = (u_center + v_center) / sqrt2
        U_sim_max = np.max(u_parallel)

        error_pct = 100.0 * abs(U_sim_max - U_ana_max) / U_ana_max

        # Print progress
        if step < 10 or step % 10 == 0 or residual < target_residual:
            print(f"{step:<8} {residual:<15.6e} {div_max:<15.6e} {U_sim_max:<15.10e} {error_pct:<12.6f}")

        # Check convergence
        if residual < target_residual:
            print(f"\n*** Converged to residual {residual:.2e} at step {step} ***")
            break

    # Final results
    print()
    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"  Final residual:    {residual:.6e}")
    print(f"  Final max div:     {div_max:.6e}")
    print(f"  U_sim_max:         {U_sim_max:.10e}")
    print(f"  U_ana_max:         {U_ana_max:.10e}")
    print(f"  Relative error:    {error_pct:.6f}%")
    print(f"  Absolute error:    {abs(U_sim_max - U_ana_max):.6e}")

    # Check if we achieved target
    if residual < 1e-8:
        print(f"\n  SUCCESS: Residual < 1e-8 achieved!")
    else:
        print(f"\n  WARNING: Residual {residual:.2e} > 1e-8")

    if residual < 1e-10:
        print(f"  SUCCESS: Residual < 1e-10 achieved!")

    if residual < 1e-12:
        print(f"  SUCCESS: Residual < 1e-12 achieved!")

    return {
        'res': res_n,
        'residual': residual,
        'residual_history': residual_history,
        'U_sim_max': U_sim_max,
        'U_ana_max': U_ana_max,
        'error_pct': error_pct,
    }


def main():
    print("=" * 70)
    print("HIGH-ACCURACY ANGLED POISEUILLE VERIFICATION")
    print("Testing mixed-precision Newton-Raphson solver")
    print("=" * 70)

    # Test at multiple resolutions
    resolutions = [16, 32]

    results = []
    for res in resolutions:
        r = run_high_accuracy_test(res)
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'N':<8} {'Final Residual':<18} {'Error%':<12} {'Converged?':<12}")
    print("-" * 50)

    for r in results:
        converged = "YES" if r['residual'] < 1e-10 else "NO"
        print(f"{r['res']:<8} {r['residual']:<18.6e} {r['error_pct']:<12.6f} {converged:<12}")


if __name__ == "__main__":
    main()
