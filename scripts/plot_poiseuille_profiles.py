import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend


def generate_slab_sdf(res_n, L, slab_thickness):
    dx = L / res_n
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    yc = L / 2.0
    sdf = np.abs(Y - yc) - slab_thickness / 2.0
    return sdf.ravel(order='F').astype(np.float32)


def analytical_profile(y, g, nu, L, slab_thickness):
    y0 = 0.5 * L - 0.5 * slab_thickness
    y1 = 0.5 * L + 0.5 * slab_thickness
    H = L - slab_thickness
    u = np.zeros_like(y)

    outside = (y <= y0) | (y >= y1)
    d = np.zeros_like(y)
    d[y >= y1] = y[y >= y1] - y1
    d[y <= y0] = (L - y1) + y[y <= y0]
    u[outside] = (g / (2.0 * nu)) * d[outside] * (H - d[outside])
    return u


def run_simulation(res_n, L=1.0, slab_thickness=0.2):
    dx = L / res_n
    sdf_values = generate_slab_sdf(res_n, L, slab_thickness)

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

    rho = 1.0
    nu = 0.01
    mu = nu * rho
    g_x = 1.0e-2

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(g_x, 0, 0))

    if res_n <= 8:
        pressure_max_iter = 100
    elif res_n == 16:
        pressure_max_iter = 100
    elif res_n == 32:
        pressure_max_iter = 500
    else:
        pressure_max_iter = 2000

    solver.set_pressure_solver_params(max_iter=pressure_max_iter, tol=1e-5)
    solver.set_velocity_solver_params(max_iter=50, tol=1e-5)
    solver.set_cfl(0.5)
    solver.set_diffusion_theta(1.0)

    H = L - slab_thickness
    U_ana_max = (g_x * H**2) / (8.0 * nu)
    dt = 0.5 * dx / (U_ana_max + 1e-12)

    max_steps = 20000
    u_mean_history = []
    for i in range(max_steps):
        solver.step(dt)
        if i % 100 == 0:
            u_field = np.array(solver.get_u())
            u_mean = np.mean(u_field)
            u_mean_history.append(u_mean)
            if len(u_mean_history) > 5:
                err = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-12)
                if err < 1e-6:
                    break

    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    U_sim_max = np.max(u_field)
    error = 100.0 * abs(U_sim_max - U_ana_max) / U_ana_max

    i = 0
    k = res_n // 2
    profile = u_field[i, :, k]
    y = (np.arange(res_n) + 0.5) * dx

    return {
        'res': res_n,
        'y': y,
        'u': profile,
        'error': error,
        'U_sim_max': U_sim_max,
        'U_ana_max': U_ana_max,
    }


def main():
    resolutions = [8, 16, 32]
    L = 1.0
    slab_thickness = 0.2
    g_x = 1.0e-2
    nu = 0.01

    results = []
    for res in resolutions:
        results.append(run_simulation(res, L, slab_thickness))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    y_fine = np.linspace(0.0, L, 400)
    u_ana = analytical_profile(y_fine, g_x, nu, L, slab_thickness)
    ax.plot(y_fine, u_ana, 'k-', linewidth=2, label='Analytical')

    markers = ['o', 's', '^']
    colors = ['C0', 'C1', 'C2']

    for i, r in enumerate(results):
        ax.scatter(r['y'], r['u'], s=60, marker=markers[i], color=colors[i],
                   edgecolors='k', linewidths=0.5,
                   label=f"N={r['res']} (err={r['error']:.1f}%)")

    y0 = 0.5 * L - 0.5 * slab_thickness
    y1 = 0.5 * L + 0.5 * slab_thickness
    ax.axvspan(y0, y1, color='0.85', alpha=0.6, label='Solid')

    ax.set_title('Poiseuille Flow Velocity Profiles')
    ax.set_xlabel('y')
    ax.set_ylabel('u(y)')
    ax.set_xlim(0.0, L)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    N = np.array([r['res'] for r in results])
    err = np.array([r['error'] for r in results])
    ax.loglog(N, err, 'o-', color='C0', linewidth=2, markersize=10, label='Simulation')

    N0 = N[0]
    err0 = err[0]
    ref1 = err0 * (N / N0) ** -1.0
    ref2 = err0 * (N / N0) ** -2.0
    ax.loglog(N, ref1, '--', color='red', label='1st order')
    ax.loglog(N, ref2, '--', color='green', label='2nd order')

    ax.set_title('Grid Convergence')
    ax.set_xlabel('Resolution N')
    ax.set_ylabel('Error (%)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = 'output/poiseuille_verification_n8_16_32.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
