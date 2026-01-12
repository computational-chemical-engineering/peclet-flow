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

    # Get all fields
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w_field = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    p_field = np.array(solver.get_p()).reshape((res_n, res_n, res_n), order='F')
    sdf_field = sdf_values.reshape((res_n, res_n, res_n), order='F')

    U_sim_max = np.max(u_field)
    error = 100.0 * abs(U_sim_max - U_ana_max) / U_ana_max

    # Extract profiles along y at x=0, z=mid
    i = 0
    k = res_n // 2
    u_profile = u_field[i, :, k]
    p_profile = p_field[i, :, k]
    y = (np.arange(res_n) + 0.5) * dx

    return {
        'res': res_n,
        'y': y,
        'u': u_profile,
        'p': p_profile,
        'error': error,
        'U_sim_max': U_sim_max,
        'U_ana_max': U_ana_max,
        'u_field': u_field,
        'v_field': v_field,
        'w_field': w_field,
        'p_field': p_field,
        'sdf': sdf_field,
        'L': L,
        'dx': dx,
    }


def main():
    os.makedirs('output', exist_ok=True)

    resolutions = [8, 16, 32]
    L = 1.0
    slab_thickness = 0.2
    g_x = 1.0e-2
    nu = 0.01

    results = []
    for res in resolutions:
        print(f"Running N={res}...")
        results.append(run_simulation(res, L, slab_thickness))

    # =====================================================================
    # Figure 1: Velocity profiles and convergence
    # =====================================================================
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

    # =====================================================================
    # Figure 2: Pressure profiles
    # =====================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for i, r in enumerate(results):
        # Subtract mean pressure for comparison
        p_centered = r['p'] - np.mean(r['p'])
        ax2.plot(r['y'], p_centered, marker=markers[i], color=colors[i],
                 markersize=6, linewidth=1.5,
                 label=f"N={r['res']}")

    ax2.axvspan(y0, y1, color='0.85', alpha=0.6, label='Solid')

    ax2.set_title('Pressure Profile (mean subtracted)')
    ax2.set_xlabel('y')
    ax2.set_ylabel('p - mean(p)')
    ax2.set_xlim(0.0, L)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = 'output/poiseuille_pressure_profile.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # =====================================================================
    # Figure 3: 2D velocity slices (finest resolution)
    # =====================================================================
    r = results[-1]  # Use finest resolution
    mid_z = r['res'] // 2

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))

    u_slice = r['u_field'][:, :, mid_z]
    v_slice = r['v_field'][:, :, mid_z]
    w_slice = r['w_field'][:, :, mid_z]
    sdf_slice = r['sdf'][:, :, mid_z]

    extent = [0, r['L'], 0, r['L']]

    for ax, data, title in zip(axes3, [u_slice, v_slice, w_slice],
                                ['u (x-velocity)', 'v (y-velocity)', 'w (z-velocity)']):
        im = ax.imshow(data.T, origin='lower', extent=extent, cmap='RdBu_r')
        ax.contour(sdf_slice.T, levels=[0], colors='k', linewidths=2,
                   extent=extent, origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} (N={r["res"]})')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out_path = 'output/poiseuille_2d_velocity.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # =====================================================================
    # Figure 4: 2D pressure slice (finest resolution)
    # =====================================================================
    fig4, ax4 = plt.subplots(figsize=(6, 5))

    p_slice = r['p_field'][:, :, mid_z]
    p_centered = p_slice - np.mean(p_slice)

    im = ax4.imshow(p_centered.T, origin='lower', extent=extent, cmap='RdBu_r')
    ax4.contour(sdf_slice.T, levels=[0], colors='k', linewidths=2,
               extent=extent, origin='lower')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(f'Pressure (mean subtracted) at z={mid_z*r["dx"]:.2f} (N={r["res"]})')
    ax4.set_aspect('equal')
    plt.colorbar(im, ax=ax4, label='p - mean(p)')

    plt.tight_layout()
    out_path = 'output/poiseuille_2d_pressure.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    plt.close('all')
    print("\nAll plots saved to output/")


if __name__ == '__main__':
    main()
