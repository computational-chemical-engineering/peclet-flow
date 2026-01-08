"""
Angled Poiseuille Flow Verification

Poiseuille flow between two parallel walls at 45 degrees.
- Wall normal: n = (-1/√2, 1/√2, 0)
- Body force: f = (1/√2, 1/√2, 0) (parallel to walls)
- Geometry is periodic

This tests the IBM with multiple active velocity components but no curvature.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend


def generate_angled_slab_sdf(res_n, L, slab_thickness):
    """
    Generate SDF for a slab at 45 degrees.

    The slab is centered on the line y = x (passing through domain center).
    Wall normal: (-1/√2, 1/√2, 0)

    For periodicity, we need to account for periodic images of the plane.
    The plane y - x = 0 has periodic images at y - x = ±L.
    """
    dx = L / res_n

    # Cell centers
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    # Signed distance to center plane y - x = 0
    # d = (y - x) / sqrt(2)  (positive when y > x)
    sqrt2 = np.sqrt(2.0)
    d = (Y - X) / sqrt2

    # For periodic domain, find distance to nearest periodic image
    # Images are at y - x = 0, ±L, ±2L, ...
    # In terms of d: d = 0, ±L/√2, ...
    period = L / sqrt2

    # Wrap d to [-period/2, period/2]
    d_wrapped = d - period * np.round(d / period)

    # SDF: positive outside slab (fluid), negative inside (solid)
    sdf = np.abs(d_wrapped) - slab_thickness / 2.0

    return sdf.ravel(order='F').astype(np.float32)


def run_simulation(res_n, L=1.0, slab_thickness=0.2, verbose=True):
    """Run angled Poiseuille simulation and return results."""

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n

    # Channel height (fluid gap)
    # The slab occupies thickness t in the normal direction
    # Due to periodicity, effective channel height = period - t = L/√2 - t
    period = L / sqrt2
    H = period - slab_thickness

    if verbose:
        print(f"N={res_n}: period={period:.4f}, slab_t={slab_thickness:.4f}, H={H:.4f}")

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
    nu = 0.01
    mu = nu * rho

    # Body force magnitude and direction (parallel to wall)
    f_mag = 1.0e-2
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # Match settings used in scripts/plot_poiseuille_profiles.py
    if res_n <= 8:
        pressure_max_iter = 100
    elif res_n == 16:
        pressure_max_iter = 100
    elif res_n == 32:
        pressure_max_iter = 500
    else:
        pressure_max_iter = 2000

    velocity_max_iter = 50
    cfl = 0.5
    theta = 1.0
    max_steps = 40000
    check_interval = 100

    solver.set_pressure_solver_params(max_iter=pressure_max_iter, tol=1e-5)
    solver.set_velocity_solver_params(max_iter=velocity_max_iter, tol=1e-5)
    solver.set_cfl(cfl)
    solver.set_diffusion_theta(theta)

    # Analytical max velocity (Poiseuille between parallel plates)
    U_ana_max = (f_mag * H**2) / (8.0 * nu)

    # Time stepping
    dt = 0.5 * dx / (U_ana_max + 1e-12)

    if verbose:
        print(f"  U_ana_max={U_ana_max:.6e}, dt={dt:.6e}")

    # Run to steady state with convergence check
    u_mean_history = []
    for i in range(max_steps):
        solver.step(dt)

        if i % check_interval == 0:
            u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
            v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')

            u_center = 0.5 * (u_field + np.roll(u_field, -1, axis=0))
            v_center = 0.5 * (v_field + np.roll(v_field, -1, axis=1))

            u_parallel = (u_center + v_center) / sqrt2
            u_mean = np.mean(u_parallel)
            u_mean_history.append(u_mean)

            if verbose:
                print(f"  Step {i}/{max_steps}")

            if len(u_mean_history) > 5:
                rel_change = abs(u_mean_history[-1] - u_mean_history[-2]) / (abs(u_mean_history[-1]) + 1e-12)
                if rel_change < 5e-7:
                    if verbose:
                        print(f"  Converged at step {i}")
                    break

    # Get final fields
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w_field = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    p_field = np.array(solver.get_p()).reshape((res_n, res_n, res_n), order='F')

    u_center = 0.5 * (u_field + np.roll(u_field, -1, axis=0))
    v_center = 0.5 * (v_field + np.roll(v_field, -1, axis=1))

    # Velocity in flow direction at cell centers
    u_parallel = (u_center + v_center) / sqrt2

    # Max velocity
    U_sim_max = np.max(u_parallel)

    # Error
    error = 100.0 * abs(U_sim_max - U_ana_max) / U_ana_max

    if verbose:
        print(f"  U_sim_max={U_sim_max:.6e}, U_ana_max={U_ana_max:.6e}, Error={error:.2f}%")

    return {
        'res': res_n,
        'error': error,
        'U_sim_max': U_sim_max,
        'U_ana_max': U_ana_max,
        'H': H,
        'u_field': u_field,
        'v_field': v_field,
        'w_field': w_field,
        'p_field': p_field,
        'u_parallel': u_parallel,
        'sdf': sdf_values.reshape((res_n, res_n, res_n), order='F'),
        'dx': dx,
        'L': L,
        'slab_thickness': slab_thickness,
    }


def extract_profile_along_normal(result):
    """Extract velocity profile along the wall-normal direction."""
    res_n = result['res']
    u_parallel = result['u_parallel']
    sdf = result['sdf']
    L = result['L']
    dx = result['dx']
    sqrt2 = np.sqrt(2.0)

    # Sample along the normal direction (-1/√2, 1/√2, 0)
    # Start from corner (0, L, L/2) and go towards (L, 0, L/2)
    # Parameter s goes from 0 to L*√2

    n_samples = res_n * 2
    s_vals = np.linspace(0, L * sqrt2, n_samples)

    # Points along the normal: (s/√2, L - s/√2, L/2)
    # Actually, let's go from center outward in normal direction
    # Center: (L/2, L/2, L/2)
    # Normal direction: (-1/√2, 1/√2, 0)

    center = np.array([L/2, L/2, L/2])
    normal = np.array([-1/sqrt2, 1/sqrt2, 0])

    # Sample from -L/2 to L/2 in normal direction (covers full periodic cell)
    period = L / sqrt2
    d_vals = np.linspace(-period/2, period/2, n_samples)

    u_profile = []
    sdf_profile = []

    for d in d_vals:
        # Point in 3D
        p = center + d * normal

        # Wrap to domain [0, L]³
        p = p % L

        # Find nearest cell (using trilinear would be better, but nearest is ok)
        ix = int(p[0] / dx) % res_n
        iy = int(p[1] / dx) % res_n
        iz = int(p[2] / dx) % res_n

        u_profile.append(u_parallel[ix, iy, iz])
        sdf_profile.append(sdf[ix, iy, iz])

    return d_vals, np.array(u_profile), np.array(sdf_profile)


def analytical_profile(d_vals, H, slab_thickness, f_mag, nu):
    """Compute analytical Poiseuille profile."""
    sqrt2 = np.sqrt(2.0)

    u_ana = np.zeros_like(d_vals)
    half_t = slab_thickness / 2.0
    half_H = H / 2.0

    for i, d in enumerate(d_vals):
        if abs(d) <= half_t:
            # Inside solid
            u_ana[i] = 0.0
        else:
            # In fluid: parabolic profile
            # Distance from wall (at |d| = half_t)
            d_from_wall = abs(d) - half_t
            # Parabolic: u = (f/(2ν)) * y * (H - y) where y is distance from wall
            u_ana[i] = (f_mag / (2*nu)) * d_from_wall * (H - d_from_wall)

    return u_ana


def analytical_component_at_points(x, y, L, slab_thickness, f_mag, nu, comp):
    """
    Analytical component value at physical points for angled Poiseuille flow.

    comp: 'u', 'v', or 'w'
    """
    sqrt2 = np.sqrt(2.0)
    period = L / sqrt2
    half_t = slab_thickness / 2.0
    H = period - slab_thickness

    # Wrap y into [0, L)
    y = np.mod(y, L)

    d = (y - x) / sqrt2
    d_wrapped = d - period * np.round(d / period)

    u_parallel = np.zeros_like(d_wrapped)
    abs_d = np.abs(d_wrapped)
    fluid = abs_d > half_t
    d_wall = abs_d[fluid] - half_t
    u_parallel[fluid] = (f_mag / (2.0 * nu)) * d_wall * (H - d_wall)

    if comp == 'u' or comp == 'v':
        return u_parallel / sqrt2
    return np.zeros_like(u_parallel)


def trilinear_sample(field, x_idx, y_idx, z_idx):
    """Trilinear sample on periodic grid using index-space coordinates."""
    n = field.shape[0]
    i0 = int(np.floor(x_idx))
    j0 = int(np.floor(y_idx))
    k0 = int(np.floor(z_idx))
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    tx = x_idx - i0
    ty = y_idx - j0
    tz = z_idx - k0

    i0 %= n
    j0 %= n
    k0 %= n
    i1 %= n
    j1 %= n
    k1 %= n

    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz


def sample_component_at_point(result, comp, x, y, z):
    """Sample a component at a physical point using trilinear interpolation."""
    dx = result['dx']
    if comp == 'u':
        field = result['u_field']
        x_idx = x / dx
        y_idx = y / dx - 0.5
        z_idx = z / dx - 0.5
    elif comp == 'v':
        field = result['v_field']
        x_idx = x / dx - 0.5
        y_idx = y / dx
        z_idx = z / dx - 0.5
    elif comp == 'w':
        field = result['w_field']
        x_idx = x / dx - 0.5
        y_idx = y / dx - 0.5
        z_idx = z / dx
    else:
        field = result['p_field']
        x_idx = x / dx - 0.5
        y_idx = y / dx - 0.5
        z_idx = z / dx - 0.5

    return trilinear_sample(field, x_idx, y_idx, z_idx)


def extract_line_samples(result, comp, axis, x0, y0, z0):
    """
    Extract a line of a component varying along one axis at fixed x,y,z.

    For MAC grid with cell centers at (i+0.5)dx, component locations are:
    - u: (i)dx, (j+0.5)dx, (k+0.5)dx
    - v: (i+0.5)dx, (j)dx, (k+0.5)dx
    - w: (i+0.5)dx, (j+0.5)dx, (k)dx
    """
    res_n = result['res']
    dx = result['dx']

    if axis == 'x':
        if comp == 'v':
            line = (np.arange(res_n) + 0.5) * dx
        else:
            line = (np.arange(res_n) + 0.5) * dx
        x = line
        y = np.full_like(line, y0)
        z = np.full_like(line, z0)
    elif axis == 'y':
        if comp == 'v':
            line = (np.arange(res_n) + 0.0) * dx
        else:
            line = (np.arange(res_n) + 0.5) * dx
        x = np.full_like(line, x0)
        y = line
        z = np.full_like(line, z0)
    else:
        if comp == 'w':
            line = (np.arange(res_n) + 0.0) * dx
        else:
            line = (np.arange(res_n) + 0.5) * dx
        x = np.full_like(line, x0)
        y = np.full_like(line, y0)
        z = line

    values = np.array([
        sample_component_at_point(result, comp, x_i, y_i, z_i)
        for x_i, y_i, z_i in zip(x, y, z)
    ])

    return x, y, z, values


def make_staggered_line_plots(results):
    """Plot u, v, w, p along fixed lines and compare to high-res profile."""
    f_mag = 0.01
    nu = 0.01

    comps = ['u', 'v', 'w', 'p']
    titles = ['u (x-velocity)', 'v (y-velocity)', 'w (z-velocity)', 'p (pressure)']
    markers = ['o', 's', '^']
    colors = ['C0', 'C1', 'C2']
    line_specs = {
        'u': {'axis': 'y', 'x': 0.5, 'y': None, 'z': 0.5},
        'v': {'axis': 'x', 'x': None, 'y': 0.5, 'z': 0.5},
        'w': {'axis': 'z', 'x': 0.5, 'y': 0.5, 'z': None},
        'p': {'axis': 'y', 'x': 0.5, 'y': None, 'z': 0.5},
    }

    for comp, title in zip(comps, titles):
        fig, ax = plt.subplots(figsize=(6, 4))
        spec = line_specs[comp]
        axis = spec['axis']
        x0 = spec['x'] if spec['x'] is not None else 0.5
        y0 = spec['y'] if spec['y'] is not None else 0.5
        z0 = spec['z'] if spec['z'] is not None else 0.5

        # High-resolution reference line from finest grid
        r_ref = results[-1]
        fine = np.linspace(0.0, r_ref['L'], 400)
        if axis == 'x':
            x_ref, y_ref, z_ref = fine, np.full_like(fine, y0), np.full_like(fine, z0)
            axis_label = 'x'
        elif axis == 'y':
            x_ref, y_ref, z_ref = np.full_like(fine, x0), fine, np.full_like(fine, z0)
            axis_label = 'y'
        else:
            x_ref, y_ref, z_ref = np.full_like(fine, x0), np.full_like(fine, y0), fine
            axis_label = 'z'

        ref_vals = np.array([
            sample_component_at_point(r_ref, comp, xr, yr, zr)
            for xr, yr, zr in zip(x_ref, y_ref, z_ref)
        ])
        ax.plot(fine, ref_vals, 'k-', linewidth=2, label='High-res reference')

        for i, r in enumerate(results):
            x, y, z, vals = extract_line_samples(r, comp, axis, x0, y0, z0)
            coord = x if axis == 'x' else (y if axis == 'y' else z)
            ax.scatter(
                coord, vals, s=30, marker=markers[i], color=colors[i],
                edgecolors='k', linewidths=0.5,
                label=f"N={r['res']}"
            )

        ax.set_title(f"{title} along {axis_label} at x={x0:.4f}, y={y0:.4f}, z={z0:.4f}")
        ax.set_xlabel(axis_label)
        if comp == 'p':
            ax.set_ylabel('pressure')
        else:
            ax.set_ylabel('velocity')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        out_path = f'output/angled_poiseuille_staggered_{comp}.png'
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Staggered line plot saved to {out_path}")


def run_convergence_study():
    """Run grid convergence study."""
    resolutions = [8, 16, 32]
    L = 1.0
    slab_thickness = 0.2
    print("=" * 60)
    print("Angled Poiseuille Flow Verification (45° walls)")
    print("=" * 60)
    print(f"Wall normal: (-1/√2, 1/√2, 0)")
    print(f"Body force:  ( 1/√2, 1/√2, 0) * 0.01")
    print(f"Slab thickness: {slab_thickness}")
    print("=" * 60)

    results = []
    for res in resolutions:
        print(f"\n--- Resolution N={res} ---")
        r = run_simulation(res, L, slab_thickness)
        results.append(r)

    # Print summary
    print("\n" + "=" * 60)
    print("Grid Convergence Results:")
    print(f"{'N':<6} {'Error%':<12} {'U_sim_max':<14} {'U_ana_max':<14} {'Order':<8}")
    print("-" * 60)

    for i, r in enumerate(results):
        order = "-"
        if i > 0:
            prev_err = results[i-1]['error']
            if r['error'] > 1e-10 and prev_err > 1e-10:
                order = f"{np.log2(prev_err / r['error']):.2f}"
        print(f"{r['res']:<6} {r['error']:<12.2f} {r['U_sim_max']:<14.6e} {r['U_ana_max']:<14.6e} {order:<8}")

    return results


def make_plots(results):
    """Create verification plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sqrt2 = np.sqrt(2.0)
    f_mag = 0.01
    nu = 0.01

    # Plot 1: Velocity profiles
    ax1 = axes[0]

    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', '^']

    for i, r in enumerate(results):
        d_vals, u_profile, sdf_profile = extract_profile_along_normal(r)
        ax1.scatter(sdf_profile, u_profile, c=colors[i], marker=markers[i], s=40,
                   label=f"N={r['res']} (err={r['error']:.1f}%)",
                   edgecolors='k', linewidths=0.5, zorder=2)

    # Analytical profile (use finest grid parameters)
    r = results[-1]
    d_ana = np.linspace(-r['L']/(2*sqrt2), r['L']/(2*sqrt2), 200)
    u_ana = analytical_profile(d_ana, r['H'], r['slab_thickness'], f_mag, nu)
    ax1.plot(d_ana, u_ana, 'k-', linewidth=2, label='Analytical', zorder=1)

    # Shade solid region
    half_t = r['slab_thickness'] / 2.0
    ax1.axvspan(-half_t, half_t, alpha=0.2, color='gray', label='Solid')

    ax1.set_xlabel('Distance along normal (d)')
    ax1.set_ylabel('Velocity parallel to wall')
    ax1.set_title('Angled Poiseuille (45°) Velocity Profile')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Convergence
    ax2 = axes[1]

    ns = [r['res'] for r in results]
    errs = [r['error'] for r in results]

    ax2.loglog(ns, errs, 'bo-', markersize=10, linewidth=2, label='Simulation')

    # Reference slopes
    n_ref = np.array([8, 32])
    err_ref_1st = errs[0] * (ns[0] / n_ref)
    err_ref_2nd = errs[0] * (ns[0] / n_ref)**2
    ax2.loglog(n_ref, err_ref_1st, 'r--', alpha=0.5, label='1st order')
    ax2.loglog(n_ref, err_ref_2nd, 'g--', alpha=0.5, label='2nd order')

    ax2.set_xlabel('Resolution N')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Grid Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    plt.savefig('output/angled_poiseuille_verification.png', dpi=150)
    print(f"\nPlot saved to output/angled_poiseuille_verification.png")

    # Also create a 2D slice visualization
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    r = results[-1]  # Use finest resolution
    mid_z = r['res'] // 2

    # Plot u, v, and velocity magnitude
    u_slice = r['u_field'][:, :, mid_z]
    v_slice = r['v_field'][:, :, mid_z]
    mag_slice = r['u_parallel'][:, :, mid_z]
    sdf_slice = r['sdf'][:, :, mid_z]

    extent = [0, r['L'], 0, r['L']]

    for ax, data, title in zip(axes2, [u_slice, v_slice, mag_slice],
                                ['u (x-velocity)', 'v (y-velocity)', 'u_parallel (flow dir)']):
        im = ax.imshow(data.T, origin='lower', extent=extent, cmap='RdBu_r')
        ax.contour(sdf_slice.T, levels=[0], colors='k', linewidths=2,
                   extent=extent, origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} (N={r["res"]})')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('output/angled_poiseuille_2d_slices.png', dpi=150)
    print(f"2D slice plot saved to output/angled_poiseuille_2d_slices.png")

    # 2D pressure field
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    p_slice = r['p_field'][:, :, mid_z]
    p_centered = p_slice - np.mean(p_slice)
    im = ax3.imshow(p_centered.T, origin='lower', extent=extent, cmap='RdBu_r')
    ax3.contour(sdf_slice.T, levels=[0], colors='k', linewidths=2,
               extent=extent, origin='lower')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Pressure (mean subtracted) at z={mid_z*r["dx"]:.2f} (N={r["res"]})')
    ax3.set_aspect('equal')
    plt.colorbar(im, ax=ax3, label='p - mean(p)')
    plt.tight_layout()
    plt.savefig('output/angled_poiseuille_2d_pressure.png', dpi=150)
    print(f"2D pressure plot saved to output/angled_poiseuille_2d_pressure.png")


if __name__ == "__main__":
    results = run_convergence_study()
    make_plots(results)
    make_staggered_line_plots(results)
