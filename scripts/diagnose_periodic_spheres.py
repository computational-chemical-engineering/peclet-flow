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


def sample_sdf_interp_local(sdf, x, y, z):
    """Trilinear SDF sampling with periodic wrapping (matches C++ helper)."""
    nx, ny, nz = sdf.shape
    fx = np.floor(x)
    fy = np.floor(y)
    fz = np.floor(z)

    wx = x - fx
    wy = y - fy
    wz = z - fz

    ix = int(fx)
    iy = int(fy)
    iz = int(fz)

    x0 = ix % nx
    y0 = iy % ny
    z0 = iz % nz
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    z1 = (z0 + 1) % nz

    c000 = sdf[x0, y0, z0]
    c100 = sdf[x1, y0, z0]
    c010 = sdf[x0, y1, z0]
    c110 = sdf[x1, y1, z0]
    c001 = sdf[x0, y0, z1]
    c101 = sdf[x1, y0, z1]
    c011 = sdf[x0, y1, z1]
    c111 = sdf[x1, y1, z1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy

    return c0 * (1.0 - wz) + c1 * wz


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


def save_debug_vti(filename, res, spacing, debug_fields):
    """
    Save debug residual/divergence fields to VTI (Float32).

    debug_fields order:
      0: res_u_pre
      1: res_v_pre
      2: res_w_pre
      3: res_u_post
      4: res_v_post
      5: res_w_post
      6: div_pre
      7: div_post
      8: pressure_fluid_flag
    """
    nx, ny, nz = res
    dx, dy, dz = spacing

    names = [
        "res_u_pre",
        "res_v_pre",
        "res_w_pre",
        "res_u_post",
        "res_v_post",
        "res_w_post",
        "div_pre",
        "div_post",
        "pressure_fluid_flag",
    ]

    with open(filename, "wb") as f:
        f.write(b'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" Spacing="{dx} {dy} {dz}">\n'.encode("ascii"))
        f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n'.encode("ascii"))
        f.write(b'      <CellData Scalars="res_u_pre">\n')

        offset = 0
        field_len = nx * ny * nz * 4
        for name in names:
            f.write(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="1" format="appended" offset="{offset}"/>\n'.encode("ascii"))
            offset += field_len + 8

        f.write(b'      </CellData>\n')
        f.write(b'    </Piece>\n')
        f.write(b'  </ImageData>\n')

        f.write(b'  <AppendedData encoding="raw">\n')
        f.write(b'    _')

        def write_chunk_f32(data):
            flat = data.astype("<f4")
            nbytes = flat.nbytes
            f.write(np.array([nbytes], dtype="<u8").tobytes())
            flat.tofile(f)

        for field in debug_fields:
            write_chunk_f32(field.flatten("F"))

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
    solver.set_debug_stats(False)

    # STOKES LIMIT: very small rho
    rho = 0.0  # Very small for Stokes limit
    mu = 1.0
    f_mag = 1.0

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(f_mag, 0, 0))

    # High accuracy settings
    solver.set_pressure_solver_params(iter=50)
    solver.set_velocity_solver_params(iter=2)
    solver.set_diffusion_theta(1.0)
    solver.set_outer_iterations(800)
    solver.set_outer_tolerance(0.0)

    print(f"  rho = {rho} (Stokes limit)")
    print(f"  mu = {mu}")
    print(f"  f_mag = {f_mag}")
    print(f"  Sphere radius R = {R:.4f}")


#    debug_cell = (18, 16, 10)
#    print(f"  Debug cell (i,j,k) = {debug_cell}")

#    solver.set_debug_cell(pnm_backend.int3(*debug_cell))

#    sdf_u_r = sample_sdf_interp_local(sdf_3d, debug_cell[0] + 0.5,
#                                      debug_cell[1], debug_cell[2])
#    sdf_u_l = sample_sdf_interp_local(sdf_3d, debug_cell[0] - 0.5,
#                                      debug_cell[1], debug_cell[2])
#    sdf_v_n = sample_sdf_interp_local(sdf_3d, debug_cell[0], debug_cell[1] + 0.5,
#                                      debug_cell[2])
#    sdf_v_s = sample_sdf_interp_local(sdf_3d, debug_cell[0], debug_cell[1] - 0.5,
#                                      debug_cell[2])
#    sdf_w_t = sample_sdf_interp_local(sdf_3d, debug_cell[0], debug_cell[1],
#                                      debug_cell[2] + 0.5)
#    sdf_w_b = sample_sdf_interp_local(sdf_3d, debug_cell[0], debug_cell[1],
#                                      debug_cell[2] - 0.5)
#    all_faces_solid = (
#        (sdf_u_r <= 0.0)
#        and (sdf_u_l <= 0.0)
#        and (sdf_v_n <= 0.0)
#        and (sdf_v_s <= 0.0)
#        and (sdf_w_t <= 0.0)
#        and (sdf_w_b <= 0.0)
#    )
#    print(
#        "  Debug cell face SDFs: "
#        f"u_r={sdf_u_r:.6e} u_l={sdf_u_l:.6e} "
#        f"v_n={sdf_v_n:.6e} v_s={sdf_v_s:.6e} "
#        f"w_t={sdf_w_t:.6e} w_b={sdf_w_b:.6e} "
#        f"(all_faces_solid={all_faces_solid})"
#    )

    # Time stepping
    dt = 1.0  # Large dt for steady state
    max_steps = 1000

    print(f"  dt = {dt}, max_steps = {max_steps}")
    print()

    # Store previous for residual
    u_prev = np.zeros((res_n, res_n, res_n))
    v_prev = np.zeros((res_n, res_n, res_n))
    w_prev = np.zeros((res_n, res_n, res_n))

    print(f"{'Step':<8} {'dU_max':<15} {'NS_res_max':<15} {'Div_max':<15} {'U_mean':<15}")
    print("-" * 75)

    # debug_prev_div = 0.0
    # dbg_len = 22
    # dbg_div = 0
    # dbg_du_dx = 1
    # dbg_dv_dy = 2
    # dbg_dw_dz = 3
    # dbg_u_r = 4
    # dbg_u_l = 5
    # dbg_v_n = 6
    # dbg_v_s = 7
    # dbg_w_t = 8
    # dbg_w_b = 9
    # dbg_frac_u_r = 10
    # dbg_frac_u_l = 11
    # dbg_frac_v_n = 12
    # dbg_frac_v_s = 13
    # dbg_frac_w_t = 14
    # dbg_frac_w_b = 15
    # dbg_u0 = 44
    # dbg_v0 = 45
    # dbg_w0 = 46
    # dbg_u1 = 47
    # dbg_v1 = 48
    # dbg_w1 = 49
    # dbg_u2 = 50
    # dbg_v2 = 51
    # dbg_w2 = 52
    # dbg_u3 = 53
    # dbg_v3 = 54
    # dbg_w3 = 55
    # dbg_u4 = 56
    # dbg_v4 = 57
    # dbg_w4 = 58

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

        u_mean = np.mean(u)

        if step < 10 or step % 20 == 0 or du_max < 1e-12:
            res_max = solver.get_momentum_residual_max(fluid_only=True)
            div_max = solver.get_divergence_max(dt, fluid_only=True)
            stats = solver.get_debug_stats()
            res_before = max(stats[0:3])
            res_after = max(stats[3:6])
            corr_max = max(stats[6:9])
            div_before = stats[9]
            div_after = stats[10]

            print(f"{step:<8} {du_max:<15.6e} {res_max:<15.6e} {div_max:<15.6e} {u_mean:<15.6e}")
            # print(f"         NR_res_pre={res_before:.6e} NR_res_post={res_after:.6e} "
            #       f"du_max={corr_max:.6e} div_pre={div_before:.6e} div_post={div_after:.6e}")

            # debug_fields = solver.get_debug_fields()
            # div_pre = np.array(debug_fields[6]).reshape((res_n, res_n, res_n), order="F")
            # div_post = np.array(debug_fields[7]).reshape((res_n, res_n, res_n), order="F")
            # dbg_div_pre = div_pre[debug_cell]
            # dbg_div_post = div_post[debug_cell]
            # dbg_info = solver.get_debug_cell_info()
            # dbg_pre = dbg_info[:dbg_len]
            # dbg_post = dbg_info[dbg_len:]
            # print(
            #     "         dbg_cell div_prev={:.6e} div_pre={:.6e} "
            #     "div_post={:.6e} (gpu_pre={:.6e} gpu_post={:.6e})".format(
            #         debug_prev_div, dbg_div_pre, dbg_div_post,
            #         dbg_pre[dbg_div], dbg_post[dbg_div]
            #     )
            # )
            # print(
            #     "         dbg_cell pre du_dx={:.6e} dv_dy={:.6e} dw_dz={:.6e}".format(
            #         dbg_pre[dbg_du_dx], dbg_pre[dbg_dv_dy], dbg_pre[dbg_dw_dz]
            #     )
            # )
            # print(
            #     "         dbg_cell pre u_r={:.6e} u_l={:.6e} v_n={:.6e} v_s={:.6e} "
            #     "w_t={:.6e} w_b={:.6e}".format(
            #         dbg_pre[dbg_u_r], dbg_pre[dbg_u_l], dbg_pre[dbg_v_n],
            #         dbg_pre[dbg_v_s], dbg_pre[dbg_w_t], dbg_pre[dbg_w_b]
            #     )
            # )
            # print(
            #     "         dbg_cell pre frac u_r={:.6e} u_l={:.6e} v_n={:.6e} v_s={:.6e} "
            #     "w_t={:.6e} w_b={:.6e}".format(
            #         dbg_pre[dbg_frac_u_r], dbg_pre[dbg_frac_u_l],
            #         dbg_pre[dbg_frac_v_n], dbg_pre[dbg_frac_v_s],
            #         dbg_pre[dbg_frac_w_t], dbg_pre[dbg_frac_w_b]
            #     )
            # )
            # print(
            #     "         dbg_cell post du_dx={:.6e} dv_dy={:.6e} dw_dz={:.6e}".format(
            #         dbg_post[dbg_du_dx], dbg_post[dbg_dv_dy], dbg_post[dbg_dw_dz]
            #     )
            # )
            # print(
            #     "         dbg_cell u/v/w pre={:.6e}/{:.6e}/{:.6e} "
            #     "after_u={:.6e}/{:.6e}/{:.6e} after_v={:.6e}/{:.6e}/{:.6e} "
            #     "after_w={:.6e}/{:.6e}/{:.6e} post_proj={:.6e}/{:.6e}/{:.6e}".format(
            #         dbg_info[dbg_u0], dbg_info[dbg_v0], dbg_info[dbg_w0],
            #         dbg_info[dbg_u1], dbg_info[dbg_v1], dbg_info[dbg_w1],
            #         dbg_info[dbg_u2], dbg_info[dbg_v2], dbg_info[dbg_w2],
            #         dbg_info[dbg_u3], dbg_info[dbg_v3], dbg_info[dbg_w3],
            #         dbg_info[dbg_u4], dbg_info[dbg_v4], dbg_info[dbg_w4],
            #     )
            # )
            # debug_prev_div = dbg_div_post

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
    res_max_cpp = solver.get_momentum_residual_max(fluid_only=True)
    div_max_cpp = solver.get_divergence_max(dt, fluid_only=True)
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
    print(f"  Max |NS residual| = {res_max_cpp:.6e}")
    print(f"  Max |divergence| (C++ kernel) = {div_max_cpp:.6e}")
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

    debug_fields = solver.get_debug_fields()
    debug_arrays = [
        np.array(field).reshape((res_n, res_n, res_n), order="F")
        for field in debug_fields
    ]
    div_pre = debug_arrays[6]
    abs_div_pre = np.abs(div_pre)
    flat_idx = int(np.argmax(abs_div_pre))
    max_idx = np.unravel_index(flat_idx, div_pre.shape, order="F")
    max_val = float(div_pre[max_idx])
    print(
        "  Max |div_pre| cell index (i,j,k) = "
        f"{max_idx}, value = {max_val:.6e}"
    )
    print(f"  Max |div_pre| flat index (F-order) = {flat_idx}")
    sdf_u_r = 0.5 * (sdf_3d + np.roll(sdf_3d, -1, axis=0))
    sdf_u_l = 0.5 * (sdf_3d + np.roll(sdf_3d, 1, axis=0))
    sdf_v_n = 0.5 * (sdf_3d + np.roll(sdf_3d, -1, axis=1))
    sdf_v_s = 0.5 * (sdf_3d + np.roll(sdf_3d, 1, axis=1))
    sdf_w_t = 0.5 * (sdf_3d + np.roll(sdf_3d, -1, axis=2))
    sdf_w_b = 0.5 * (sdf_3d + np.roll(sdf_3d, 1, axis=2))
    solid_mask = (
        (sdf_u_r <= 0.0)
        & (sdf_u_l <= 0.0)
        & (sdf_v_n <= 0.0)
        & (sdf_v_s <= 0.0)
        & (sdf_w_t <= 0.0)
        & (sdf_w_b <= 0.0)
    )
    if np.any(solid_mask):
        solid_div_max = np.max(np.abs(div_pre[solid_mask]))
        solid_all_zero = np.allclose(div_pre[solid_mask], 0.0, atol=1e-12)
        print(
            "  Solid-cell div_pre max = "
            f"{solid_div_max:.6e} (all_zero={solid_all_zero})"
        )
    else:
        print("  Solid-cell div_pre max = n/a (no fully solid cells)")
    fluid_flag = (
        (sdf_u_r > 0.0)
        | (sdf_u_l > 0.0)
        | (sdf_v_n > 0.0)
        | (sdf_v_s > 0.0)
        | (sdf_w_t > 0.0)
        | (sdf_w_b > 0.0)
    ).astype(np.float32)
    debug_arrays.append(fluid_flag)
    save_debug_vti(
        str(out_dir / f'sphere_debug_phi{phi_target}_N{res_n}.vti'),
        (res_n, res_n, res_n),
        (dx, dx, dx),
        debug_arrays,
    )

    return {
        'K_sim': K_sim,
        'K_ref': K_ref,
        'error': error,
        'div_max': div_max_cpp,
        'residual_max': res_max_cpp,
        'u_mean': U_sup,
    }


def main():
    print("=" * 70)
    print("PERIODIC SPHERE DIAGNOSTIC - STOKES LIMIT")
    print("=" * 70)

    # Run at phi=0.20 with resolution 64
    result = run_stokes_diagnostic(phi_target=0.30, res_n=32)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  K_sim = {result['K_sim']:.4f}")
    print(f"  K_ref = {result['K_ref']:.4f}")
    print(f"  Error = {result['error']:+.2f}%")
    print(f"  Max |NS residual| = {result['residual_max']:.6e}")
    print(f"  Max |div| (C++ kernel) = {result['div_max']:.6e}")


if __name__ == "__main__":
    main()
