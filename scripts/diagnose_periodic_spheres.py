"""
Diagnostic script for periodic sphere array verification.
- Runs in Stokes limit (very small rho)
- Creates 2D plots of residual, divergence, velocity magnitude
- Outputs VTI files for ParaView visualization
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend


def generate_single_sphere_sdf(res_n, L, R):
    """
    Generate SDF for a single sphere centered in domain.

    Returns:
        sdf_flat: Flattened array in x-fastest order for C++ solver
        sdf_3d: 3D array with shape (nx, ny, nz) matching solver convention
    """
    dx = L / res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx

    # Create meshgrid with shape (nx, ny, nz) - same as solver convention
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    xc, yc, zc = L/2, L/2, L/2
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)
    sdf = dist - R

    # Flatten with F-order (x-fastest) for C++ solver
    return sdf.ravel(order='F').astype(np.float32), sdf


def generate_sc_sdf(phi, res_n, L=1.0):
    R = (phi * 3.0 / (4.0 * np.pi))**(1.0/3.0) * L
    sdf_flat, sdf_3d = generate_single_sphere_sdf(res_n, L, R)
    return sdf_flat, R, sdf_3d


def compute_divergence(u, v, w, dx):
    """Compute divergence field at cell centers."""
    # Forward differences for staggered grid
    div_u = (np.roll(u, -1, axis=0) - u) / dx
    div_v = (np.roll(v, -1, axis=1) - v) / dx
    div_w = (np.roll(w, -1, axis=2) - w) / dx
    return div_u + div_v + div_w


def compute_velocity_magnitude_at_centers(u, v, w):
    """Interpolate staggered velocities to cell centers and compute magnitude."""
    # u is at (i, j+0.5, k+0.5), interpolate to (i+0.5, j+0.5, k+0.5)
    u_c = 0.5 * (u + np.roll(u, -1, axis=0))
    # v is at (i+0.5, j, k+0.5), interpolate to (i+0.5, j+0.5, k+0.5)
    v_c = 0.5 * (v + np.roll(v, -1, axis=1))
    # w is at (i+0.5, j+0.5, k), interpolate to (i+0.5, j+0.5, k+0.5)
    w_c = 0.5 * (w + np.roll(w, -1, axis=2))

    return np.sqrt(u_c**2 + v_c**2 + w_c**2), u_c, v_c, w_c


def save_vti(filename, res, spacing, u, v, w, p, sdf):
    """
    Save fields to VTI (ImageData) format.
    Uses appended raw binary format (Little Endian, Float64).

    All fields are cell-centered:
    - velocity: interpolated to cell centers from staggered grid
    - pressure: already cell-centered
    - sdf: already cell-centered

    For nx cells, WholeExtent is "0 nx" which defines nx+1 points
    and thus nx cells. CellData has nx*ny*nz values.
    """
    nx, ny, nz = res
    dx, dy, dz = spacing

    with open(filename, 'wb') as f:
        # XML Header
        f.write(b'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" Spacing="{dx} {dy} {dz}">\n'.encode('ascii'))
        f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n'.encode('ascii'))
        f.write(b'      <CellData Scalars="pressure" Vectors="velocity">\n')

        offset = 0

        # Velocity (3 components * 8 bytes for Float64)
        vel_len = nx * ny * nz * 3 * 8
        f.write(f'        <DataArray type="Float64" Name="velocity" NumberOfComponents="3" format="appended" offset="{offset}"/>\n'.encode('ascii'))
        offset += vel_len + 8  # 8 bytes for UInt64 length header

        # Pressure (1 component * 8 bytes)
        p_len = nx * ny * nz * 1 * 8
        f.write(f'        <DataArray type="Float64" Name="pressure" NumberOfComponents="1" format="appended" offset="{offset}"/>\n'.encode('ascii'))
        offset += p_len + 8

        # SDF (keep as Float32 since it's input as float32)
        sdf_len = nx * ny * nz * 1 * 4
        f.write(f'        <DataArray type="Float32" Name="sdf" NumberOfComponents="1" format="appended" offset="{offset}"/>\n'.encode('ascii'))

        f.write(b'      </CellData>\n')
        f.write(b'    </Piece>\n')
        f.write(b'  </ImageData>\n')

        # Appended Data Section
        f.write(b'  <AppendedData encoding="raw">\n')
        f.write(b'    _')

        def write_chunk_f64(data):
            """Write Float64 data chunk with UInt64 header."""
            flat = data.astype('<f8')  # Little-endian float64
            nbytes = flat.nbytes
            f.write(np.array([nbytes], dtype='<u8').tobytes())
            flat.tofile(f)

        def write_chunk_f32(data):
            """Write Float32 data chunk with UInt64 header."""
            flat = data.astype('<f4')  # Little-endian float32
            nbytes = flat.nbytes
            f.write(np.array([nbytes], dtype='<u8').tobytes())
            flat.tofile(f)

        # 1. Velocity: Interleave (u,v,w) as 3-component vector
        # Flatten in Fortran order (x-fastest) to match C++ memory layout
        # column_stack creates (u0,v0,w0), (u1,v1,w1), ... then flatten
        u_flat = u.flatten('F')
        v_flat = v.flatten('F')
        w_flat = w.flatten('F')
        vel_stacked = np.column_stack((u_flat, v_flat, w_flat)).flatten()
        write_chunk_f64(vel_stacked)

        # 2. Pressure
        write_chunk_f64(p.flatten('F'))

        # 3. SDF
        write_chunk_f32(sdf.flatten('F'))

        f.write(b'\n  </AppendedData>\n')
        f.write(b'</VTKFile>\n')

    print(f"  Written: {filename}")


def run_stokes_diagnostic(phi_target=0.20, res_n=64):
    """Run diagnostic in Stokes limit."""
    L = 1.0
    dx = L / res_n

    print(f"\n{'='*70}")
    print(f"Stokes Limit Diagnostic: phi={phi_target}, N={res_n}")
    print(f"{'='*70}")

    # Generate SDF
    sdf_values, R, sdf_3d = generate_sc_sdf(phi_target, res_n, L)

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

    # STOKES LIMIT: very small rho
    rho = 1e-1  # Very small for Stokes limit
    mu = 1.0
    f_mag = 1.0e-2

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(f_mag, 0, 0))

    # High accuracy settings
    solver.set_pressure_solver_params(max_iter=5000, tol=1e-12)
    solver.set_velocity_solver_params(max_iter=200, tol=1e-10)
    solver.set_diffusion_theta(1.0)
    solver.set_outer_iterations(8)
    solver.set_outer_tolerance(1e-10)

    print(f"  rho = {rho} (Stokes limit)")
    print(f"  mu = {mu}")
    print(f"  f_mag = {f_mag}")
    print(f"  Sphere radius R = {R:.4f}")

    # Time stepping
    dt = 10.0  # Large dt for steady state
    max_steps = 200

    print(f"  dt = {dt}, max_steps = {max_steps}")
    print()

    # Store previous for residual
    u_prev = np.zeros((res_n, res_n, res_n))
    v_prev = np.zeros((res_n, res_n, res_n))
    w_prev = np.zeros((res_n, res_n, res_n))

    print(f"{'Step':<8} {'dU_max':<15} {'Div_max':<15} {'U_mean':<15}")
    print("-" * 55)

    for step in range(max_steps):
        solver.step(dt)

        # Get fields
        u = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
        v = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
        w = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')

        # Compute change (residual proxy)
        du_max = max(np.max(np.abs(u - u_prev)),
                     np.max(np.abs(v - v_prev)),
                     np.max(np.abs(w - w_prev)))

        # Compute divergence
        div = compute_divergence(u, v, w, dx)
        div_max = np.max(np.abs(div))

        u_mean = np.mean(u)

        if step < 10 or step % 20 == 0 or du_max < 1e-12:
            print(f"{step:<8} {du_max:<15.6e} {div_max:<15.6e} {u_mean:<15.6e}")

        u_prev = u.copy()
        v_prev = v.copy()
        w_prev = w.copy()

        if du_max < 1e-14:
            print(f"\n*** Converged at step {step} ***")
            break

    # Get final fields
    u = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    p = np.array(solver.get_p()).reshape((res_n, res_n, res_n), order='F')

    # Compute derived fields
    div = compute_divergence(u, v, w, dx)
    vel_mag, u_c, v_c, w_c = compute_velocity_magnitude_at_centers(u, v, w)

    # Compute K
    U_sup = np.mean(u)
    F_drag = f_mag * (L**3)
    K_sim = F_drag / (6.0 * np.pi * mu * R * U_sup)

    # Reference value
    phis = np.array([0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236])
    ks = np.array([1.096, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1, 42.1])
    K_ref = np.interp(phi_target, phis, ks)

    error = 100.0 * (K_sim - K_ref) / K_ref

    print()
    print("=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"  K_sim = {K_sim:.4f}")
    print(f"  K_ref = {K_ref:.4f} (Zick & Homsy)")
    print(f"  Error = {error:+.2f}%")
    print(f"  Max |divergence| = {np.max(np.abs(div)):.6e}")
    print(f"  Mean velocity U = {U_sup:.6e}")

    # Create output directory
    out_dir = Path("output/sphere_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2D Plots at z = N/2
    mid_z = res_n // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Velocity magnitude, u, v
    extent = [0, L, 0, L]

    ax = axes[0, 0]
    im = ax.imshow(vel_mag[:, :, mid_z].T, origin='lower', extent=extent, cmap='viridis')
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='r', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'Velocity Magnitude at z={mid_z*dx:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='|u|')

    ax = axes[0, 1]
    im = ax.imshow(u_c[:, :, mid_z].T, origin='lower', extent=extent, cmap='RdBu_r')
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='k', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'u (x-velocity) at z={mid_z*dx:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='u')

    ax = axes[0, 2]
    im = ax.imshow(v_c[:, :, mid_z].T, origin='lower', extent=extent, cmap='RdBu_r')
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='k', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'v (y-velocity) at z={mid_z*dx:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='v')

    # Row 2: Divergence, Pressure, SDF
    ax = axes[1, 0]
    div_plot = div[:, :, mid_z].T
    vmax = max(abs(div_plot.min()), abs(div_plot.max()))
    if vmax < 1e-15:
        vmax = 1e-15
    im = ax.imshow(div_plot, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='k', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'Divergence at z={mid_z*dx:.3f}\nmax|div|={np.max(np.abs(div_plot)):.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='div(u)')

    ax = axes[1, 1]
    p_plot = p[:, :, mid_z].T
    p_centered = p_plot - np.mean(p_plot)
    im = ax.imshow(p_centered, origin='lower', extent=extent, cmap='RdBu_r')
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='k', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'Pressure (mean subtracted) at z={mid_z*dx:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='p - mean(p)')

    ax = axes[1, 2]
    im = ax.imshow(sdf_3d[:, :, mid_z].T, origin='lower', extent=extent, cmap='RdBu')
    ax.contour(sdf_3d[:, :, mid_z].T, levels=[0], colors='k', linewidths=2, extent=extent, origin='lower')
    ax.set_title(f'SDF at z={mid_z*dx:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='SDF')

    plt.suptitle(f'Periodic Sphere Array: phi={phi_target}, N={res_n}, K_sim={K_sim:.2f}, K_ref={K_ref:.2f}, Error={error:+.1f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f'sphere_diagnostic_phi{phi_target}_N{res_n}.png', dpi=150)
    print(f"\n  2D plot saved to {out_dir}/sphere_diagnostic_phi{phi_target}_N{res_n}.png")
    plt.close()

    # Velocity near sphere surface plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sample velocity magnitude along a line through sphere center
    center_idx = res_n // 2
    x_line = np.linspace(0, L, res_n)
    vel_line = vel_mag[:, center_idx, center_idx]
    sdf_line = sdf_3d[:, center_idx, center_idx]

    ax.plot(x_line, vel_line, 'b-', linewidth=2, label='|u|')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Mark sphere surface
    surface_x = L/2 - R
    ax.axvline(surface_x, color='r', linestyle='--', alpha=0.5, label=f'Surface (x={surface_x:.3f})')
    ax.axvline(L/2 + R, color='r', linestyle='--', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('Velocity magnitude')
    ax.set_title(f'Velocity profile through sphere center (y=z={L/2:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'sphere_velocity_profile_phi{phi_target}_N{res_n}.png', dpi=150)
    print(f"  Velocity profile saved to {out_dir}/sphere_velocity_profile_phi{phi_target}_N{res_n}.png")
    plt.close()

    # Write VTI files
    print("\n  Writing VTI files for ParaView...")

    # All arrays are in (nx, ny, nz) shape with F-order (x-fastest)
    save_vti(
        str(out_dir / f'sphere_fields_phi{phi_target}_N{res_n}.vti'),
        (res_n, res_n, res_n),
        (dx, dx, dx),
        u_c, v_c, w_c, p, sdf_3d.astype(np.float32)
    )

    return {
        'K_sim': K_sim,
        'K_ref': K_ref,
        'error': error,
        'div_max': np.max(np.abs(div)),
        'u_mean': U_sup,
    }


def main():
    print("=" * 70)
    print("PERIODIC SPHERE DIAGNOSTIC - STOKES LIMIT")
    print("=" * 70)

    # Run at phi=0.20 with resolution 64
    result = run_stokes_diagnostic(phi_target=0.40, res_n=64)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  K_sim = {result['K_sim']:.4f}")
    print(f"  K_ref = {result['K_ref']:.4f}")
    print(f"  Error = {result['error']:+.2f}%")
    print(f"  Max |div| = {result['div_max']:.6e}")


if __name__ == "__main__":
    main()
