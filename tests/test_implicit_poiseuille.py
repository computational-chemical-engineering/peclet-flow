import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

import pnm_backend


def test_implicit_poiseuille():
    print("Testing Implicit Convection (Poiseuille Flow)...")

    nx = ny = nz = 32
    dx = 1.0 / nx
    spacing = [dx, dx, dx]

    rho = 1.0
    nu = 0.01
    g = 1.0
    y_min, y_max = 0.2, 0.8
    h = y_max - y_min

    sdf_vals = np.zeros((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            y = (j + 0.5) * dx
            d = min(y - y_min, y_max - y)
            sdf_vals[k, j, :] = d

    solver = pnm_backend.CFDSolver([nx, ny, nz], spacing)
    solver.initialize(sdf_vals, [0.0, 0.0, 0.0], spacing)
    solver.set_rho(rho)
    solver.set_mu(nu * rho)
    solver.set_theta_(1.0)
    solver.set_body_force(pnm_backend.float3(g, 0.0, 0.0))
    solver.set_pressure_solver_params(100)
    solver.set_velocity_solver_params(20)

    u_max = g * h**2 / (8.0 * nu)
    dt = 5.0 * (dx / u_max)
    t_final = 2.0 * (h**2 / nu)
    num_steps = int(t_final / dt) + 1

    start_time = time.time()
    for step in range(num_steps):
        solver.step(dt)
        if step % 10 == 0:
            u_field = np.array(solver.get_u(), copy=False)
            print(f"Step {step}: Max U = {np.max(u_field):.4f}")
    print(f"Simulation took {time.time() - start_time:.2f}s")

    u_field = np.array(solver.get_u(), copy=False)
    profile = u_field[nz // 2, :, nx // 2]

    y_coords = (np.arange(ny) + 0.5) * dx
    u_analytical = np.zeros_like(profile)
    for j, y in enumerate(y_coords):
        if y_min < y < y_max:
            u_analytical[j] = (g / (2.0 * nu)) * (y - y_min) * (y_max - y)

    error_l2 = np.sqrt(np.mean((profile - u_analytical) ** 2))
    max_sim = np.max(profile)
    err_rel = abs(max_sim - u_max) / u_max

    print(f"Simulated Max U: {max_sim:.4f}")
    print(f"Analytical Max U: {u_max:.4f}")
    print(f"Relative Error (Max): {err_rel * 100:.2f}%")
    print(f"L2 Error Profile: {error_l2:.6f}")

    assert err_rel < 0.05


if __name__ == "__main__":
    test_implicit_poiseuille()
