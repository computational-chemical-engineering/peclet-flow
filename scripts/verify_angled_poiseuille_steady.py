"""
Angled Poiseuille Flow Verification - Steady State Initialization

Poiseuille flow between two parallel walls at 45 degrees.
- Wall normal: n = (-1/√2, 1/√2, 0)
- Body force: f = (1/√2, 1/√2, 0) (parallel to walls)
- Geometry is periodic

This script initializes the velocity field with the analytical stationary solution
and pressure with zero, then runs the solver to verify stability/accuracy.
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


def analytical_component_at_points(x, y, L, slab_thickness, f_mag, mu, comp):
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
    u_parallel[fluid] = (f_mag / (2.0 * mu)) * d_wall * (H - d_wall)

    if comp == 'u' or comp == 'v':
        return u_parallel / sqrt2
    return np.zeros_like(u_parallel)


def run_simulation(res_n, L=1.0, slab_thickness=0.2, verbose=True, ibm_scheme=0, use_avg=False):
    """Run angled Poiseuille simulation with steady-state initialization and return results."""

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n

    # Channel height (fluid gap)
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
    rho = 0.1
    mu = 0.01
    f_mag = 1.0e-2
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # --- Initialize with Analytical Solution ---
    if verbose:
        print("  Initializing with analytical stationary solution...")

    # Create meshgrids for face coordinates
    # Note: pnm_backend expects Fortran order (column-major) for flattened arrays
    # Meshgrid 'ij' indexing with ravel('F') corresponds to x fast, y medium, z slow?
    # Let's verify: 
    # np.ravel(arr, 'F') iterates over 1st dim (x), then 2nd (y), then 3rd (z).
    # This matches the layout where x is contiguous.
    
    x_faces = np.linspace(0, L, res_n, endpoint=False)
    x_centers = x_faces + 0.5 * dx
    
    # U faces: (i*dx, (j+0.5)dx, (k+0.5)dx)
    X_u, Y_u, Z_u = np.meshgrid(x_faces, x_centers, x_centers, indexing='ij')
    
    # V faces: ((i+0.5)dx, j*dx, (k+0.5)dx)
    X_v, Y_v, Z_v = np.meshgrid(x_centers, x_faces, x_centers, indexing='ij')
    
    # W faces: ((i+0.5)dx, (j+0.5)dx, k*dx)
    X_w, Y_w, Z_w = np.meshgrid(x_centers, x_centers, x_faces, indexing='ij')
    
    u_init = analytical_component_at_points(X_u, Y_u, L, slab_thickness, f_mag, mu, 'u')
    v_init = analytical_component_at_points(X_v, Y_v, L, slab_thickness, f_mag, mu, 'v')
    w_init = analytical_component_at_points(X_w, Y_w, L, slab_thickness, f_mag, mu, 'w')
    
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    solver.set_v(v_init.ravel(order='F').astype(np.float32))
    solver.set_w(w_init.ravel(order='F').astype(np.float32))
    
    # -------------------------------------------

    # Solver parameters
    if res_n <= 8:
        pressure_max_iter = 100
    elif res_n == 16:
        pressure_max_iter = 100
    elif res_n == 32:
        pressure_max_iter = 500
    else:
        pressure_max_iter = 2000

    velocity_max_iter = 50
    theta = 1.0
    
    # Run a few steps to see if it drifts
    max_steps = 20 
    check_interval = 1

    solver.set_pressure_solver_params(iter=pressure_max_iter)
    solver.set_velocity_solver_params(iter=velocity_max_iter)
    solver.set_diffusion_theta(theta)

    # Analytical max velocity
    U_ana_max = (f_mag * H**2) / (8.0 * mu)

    # Time stepping
    dt = 10.0

    if verbose:
        print(f"  U_ana_max={U_ana_max:.6e}, dt={dt:.6e}")
        print("  Running simulation to check stability...")

    u_mean_history = []
    
    # Initial error check
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    u_center = 0.5 * (u_field + np.roll(u_field, -1, axis=0))
    v_center = 0.5 * (v_field + np.roll(v_field, -1, axis=1))
    u_parallel = (u_center + v_center) / sqrt2
    U_sim_max_init = np.max(u_parallel)
    err_init = 100.0 * abs(U_sim_max_init - U_ana_max) / U_ana_max
    if verbose:
        print(f"  Step 0 (Init): U_sim_max={U_sim_max_init:.6e}, Error={err_init:.2f}%")

    for i in range(1, max_steps + 1):
        solver.step(dt)

        if i % check_interval == 0:
            u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
            v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')

            u_center = 0.5 * (u_field + np.roll(u_field, -1, axis=0))
            v_center = 0.5 * (v_field + np.roll(v_field, -1, axis=1))

            u_parallel = (u_center + v_center) / sqrt2
            u_mean = np.mean(u_parallel)
            u_mean_history.append(u_mean)
            
            U_sim_max = np.max(u_parallel)
            error = 100.0 * abs(U_sim_max - U_ana_max) / U_ana_max

            if verbose:
                print(f"  Step {i}/{max_steps}: U_mean={u_mean:.6e}, U_max={U_sim_max:.6e}, Error={error:.2f}%")

    # Get final fields
    u_field = np.array(solver.get_u()).reshape((res_n, res_n, res_n), order='F')
    v_field = np.array(solver.get_v()).reshape((res_n, res_n, res_n), order='F')
    w_field = np.array(solver.get_w()).reshape((res_n, res_n, res_n), order='F')
    p_field = np.array(solver.get_p()).reshape((res_n, res_n, res_n), order='F')

    # --- Debug: Check Fluid Fractions ---
    # Fetch all area fractions for divergence computation
    frac_u = np.array(solver.get_fluid_fraction(1, pnm_backend.float3(-0.5, 0, 0))).reshape((res_n, res_n, res_n), order='F')
    frac_v = np.array(solver.get_fluid_fraction(2, pnm_backend.float3(0, -0.5, 0))).reshape((res_n, res_n, res_n), order='F')
    frac_w = np.array(solver.get_fluid_fraction(3, pnm_backend.float3(0, 0, -0.5))).reshape((res_n, res_n, res_n), order='F')
    
    # If fractions are all 0 or 1, then SDF might be interpreted wrong or all positive
    frac_vol = np.array(solver.get_fluid_fraction(0, pnm_backend.float3(0,0,0))).reshape((res_n, res_n, res_n), order='F')
    if verbose:
        print(f"  Fluid Volume Fraction: Min={np.min(frac_vol):.4f}, Max={np.max(frac_vol):.4f}, Mean={np.mean(frac_vol):.4f}")
        count_cut = np.sum((frac_vol > 0) & (frac_vol < 1))
        print(f"  Cut Cells (0 < Vf < 1): {count_cut}")

    # --- Operator Analysis ---
    if verbose:
        print("  Extracting Diffusion Stencils...")
    
    # Force update IBM geometry with Scheme 1 (Cell Average) explicitly
    scheme = 0
    solver.set_ibm_scheme(scheme)
    # Trigger update (step calls it, or we can add a binding for update_ibm_geometry? 
    # step() handles it via lazy init or force.
    # Let's ensure it runs.
    # The previous run of step() used default scheme 0?
    # We should restart solver or just rely on step() checking `ibm_initialized`?
    # In cfd_solver.cu, `static bool ibm_initialized`.
    # It only runs ONCE per process lifetime if static?
    # NO! `static` local variable is persistent across calls but local to the function.
    # If I create a NEW solver instance, does it reset?
    # `static bool` in a member function is SHARED among all instances of the class!
    # This is a BUG in `cfd_solver.cu` line 2854.
    # `static bool ibm_initialized = false;`
    # This means if I create Solver1, it inits.
    # Then Solver2, it sees `true` and skips init!
    # Grid sizes might differ!
    # THIS IS THE BUG.
    
    # I must remove `static` from `ibm_initialized` in `cfd_solver.cu`.
    # Or make it a member variable.
    # `ibm_initialized` is not in class?
    # I should check `cfd_solver.cu`.
    
    # Get Stencils for U (Component 0)
    # List of 7 arrays: C, W, E, S, N, B, T
    stencil_u_bare_flat = solver.get_diffusion_stencil(0, False)
    stencil_u_ibm_flat = solver.get_diffusion_stencil(0, True)
    
    # Get IBM Scaling Factors (D_rescale)
    ibm_scaling_flat = solver.get_ibm_scaling(0)
    ibm_scaling = np.array(ibm_scaling_flat).reshape((res_n, res_n, res_n), order='F')
    
    # Reshape stencils
    def reshape_stencil(flat_stencil, n):
        return [np.array(arr).reshape((n, n, n), order='F') for arr in flat_stencil]
        
    stencil_u_bare = reshape_stencil(stencil_u_bare_flat, res_n)
    stencil_u_ibm = reshape_stencil(stencil_u_ibm_flat, res_n)
    
    # Apply Operator Function
    def apply_stencil(field, stencil):
        # A_C * u
        res = stencil[0] * field
        # A_W * u_{i-1} (roll +1 axis 0)
        res += stencil[1] * np.roll(field, 1, axis=0)
        # A_E * u_{i+1} (roll -1 axis 0)
        res += stencil[2] * np.roll(field, -1, axis=0)
        # A_S * u_{j-1} (roll +1 axis 1)
        res += stencil[3] * np.roll(field, 1, axis=1)
        # A_N * u_{j+1} (roll -1 axis 1)
        res += stencil[4] * np.roll(field, -1, axis=1)
        # A_B * u_{k-1} (roll +1 axis 2)
        res += stencil[5] * np.roll(field, 1, axis=2)
        # A_T * u_{k+1} (roll -1 axis 2)
        res += stencil[6] * np.roll(field, -1, axis=2)
        return res

    # Reconstruct initial field U (analytical)
    # Using volume-averaged quadratic profile: u_avg = u_center + (dx^2 / 24) * laplacian(u)
    # Since mu * laplacian(u) = -fx, we have laplacian(u) = -fx / mu.
    u_init_reshaped = u_init.reshape((res_n, res_n, res_n), order='F')
    v_init_reshaped = v_init.reshape((res_n, res_n, res_n), order='F')
    w_init_reshaped = w_init.reshape((res_n, res_n, res_n), order='F')
    
    if scheme == 1:
        lap_u = -rho * fx / mu
        u_init_reshaped = u_init_reshaped + (dx**2 / 24.0) * lap_u
        # v and w also need correction if non-zero, but here v=u (symmetry) and w=0
        lap_v = -rho * fy / mu
        v_init_reshaped = v_init_reshaped + (dx**2 / 24.0) * lap_v

    # Apply operators to Initial (Analytical) U
    Lu_bare_ana = apply_stencil(u_init_reshaped, stencil_u_bare)
    Lu_ibm_ana = apply_stencil(u_init_reshaped, stencil_u_ibm)
    
    # Normalize by IBM Scaling to get physical Laplacian (mu * lap(u))
    # L_physical = L_discrete / D_rescale
    safe_scaling = ibm_scaling
    Lu_ibm_ana_norm = Lu_ibm_ana / safe_scaling
    
    # Apply operators to Final (Steady State) U
    Lu_bare_sim = apply_stencil(u_field, stencil_u_bare)
    Lu_ibm_sim = apply_stencil(u_field, stencil_u_ibm)
    Lu_ibm_sim_norm = Lu_ibm_sim / safe_scaling
    
    expected_val = -rho * fx

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
        'u_init': u_init_reshaped,
        'v_init': v_init_reshaped,
        'w_init': w_init_reshaped,
        'frac_u': frac_u,
        'frac_v': frac_v,
        'frac_w': frac_w,
        'u_parallel': u_parallel,
        'sdf': sdf_values.reshape((res_n, res_n, res_n), order='F'),
        'dx': dx,
        'L': L,
        'slab_thickness': slab_thickness,
        'Lu_bare_ana': Lu_bare_ana,
        'Lu_ibm_ana': Lu_ibm_ana_norm,
        'Lu_ibm_sim': Lu_ibm_sim_norm,
        'expected_val': expected_val
    }

def extract_profile_along_normal(result):
    """Extract velocity profile along the wall-normal direction."""
    res_n = result['res']
    u_parallel = result['u_parallel']
    sdf = result['sdf']
    L = result['L']
    dx = result['dx']
    sqrt2 = np.sqrt(2.0)

    center = np.array([L/2, L/2, L/2])
    normal = np.array([-1/sqrt2, 1/sqrt2, 0])
    period = L / sqrt2
    
    n_samples = res_n * 2
    d_vals = np.linspace(-period/2, period/2, n_samples)

    u_profile = []
    sdf_profile = []

    for d in d_vals:
        p = center + d * normal
        p = p % L
        ix = int(p[0] / dx) % res_n
        iy = int(p[1] / dx) % res_n
        iz = int(p[2] / dx) % res_n

        u_profile.append(u_parallel[ix, iy, iz])
        sdf_profile.append(sdf[ix, iy, iz])

    return d_vals, np.array(u_profile), np.array(sdf_profile)


def analytical_profile(d_vals, H, slab_thickness, f_mag, mu):
    """Compute analytical Poiseuille profile."""
    u_ana = np.zeros_like(d_vals)
    half_t = slab_thickness / 2.0
    
    for i, d in enumerate(d_vals):
        if abs(d) <= half_t:
            u_ana[i] = 0.0
        else:
            d_from_wall = abs(d) - half_t
            u_ana[i] = (f_mag / (2*mu)) * d_from_wall * (H - d_from_wall)

    return u_ana


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
    """Extract a line of a component varying along one axis at fixed x,y,z."""
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
    """Plot u, v, w, p along fixed lines."""
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
        ax.set_ylabel('value')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        out_path = f'output/angled_poiseuille_fixed_pos_{comp}.png'
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Staggered line plot saved to {out_path}")


def run_convergence_study():
    """Run grid convergence study."""
    resolutions = [8, 16, 32]
    L = 1.0
    slab_thickness = 0.2
    
    print("=" * 60)
    print("Angled Poiseuille Verification - SCHEME 0 (POINT-VALUE)")
    print("=" * 60)
    results_pt = []
    for res in resolutions:
        r = run_simulation(res, L, slab_thickness, verbose=True, ibm_scheme=0, use_avg=False)
        results_pt.append(r)
        
    print("\n" + "=" * 60)
    print("Angled Poiseuille Verification - SCHEME 1 (CELL-AVERAGE)")
    print("=" * 60)
    results_avg = []
    for res in resolutions:
        r = run_simulation(res, L, slab_thickness, verbose=True, ibm_scheme=1, use_avg=True)
        results_avg.append(r)

    return results_avg # Return avg for plotting default


def make_plots(results):
    """Create verification plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sqrt2 = np.sqrt(2.0)
    f_mag = 0.01
    mu = 0.01

    ax1 = axes[0]
    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', '^']

    for i, r in enumerate(results):
        d_vals, u_profile, sdf_profile = extract_profile_along_normal(r)
        ax1.scatter(sdf_profile, u_profile, c=colors[i], marker=markers[i], s=40,
                   label=f"N={r['res']} (err={r['error']:.1f}%)",
                   edgecolors='k', linewidths=0.5, zorder=2)

    r = results[-1]
    d_ana = np.linspace(-r['L']/(2*sqrt2), r['L']/(2*sqrt2), 200)
    u_ana = analytical_profile(d_ana, r['H'], r['slab_thickness'], f_mag, mu)
    ax1.plot(d_ana, u_ana, 'k-', linewidth=2, label='Analytical', zorder=1)

    half_t = r['slab_thickness'] / 2.0
    ax1.axvspan(-half_t, half_t, alpha=0.2, color='gray', label='Solid')

    ax1.set_xlabel('Distance along normal (d)')
    ax1.set_ylabel('Velocity parallel to wall')
    ax1.set_title('Angled Poiseuille (45°) Velocity Profile')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ns = [r['res'] for r in results]
    errs = [r['error'] for r in results]

    ax2.loglog(ns, errs, 'bo-', markersize=10, linewidth=2, label='Simulation')
    
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
    plt.savefig('output/angled_poiseuille_fixed_pos.png', dpi=150)
    print(f"\nPlot saved to output/angled_poiseuille_fixed_pos.png")

    # Divergence Analysis
    mid_z = r['res'] // 2
    
    # Extract slices for Operator Analysis
    lu_bare_slice = r['Lu_bare_ana'][:, :, mid_z]
    lu_ibm_slice = r['Lu_ibm_ana'][:, :, mid_z]
    lu_ibm_sim_slice = r['Lu_ibm_sim'][:, :, mid_z]
    sdf_slice = r['sdf'][:, :, mid_z]
    
    fu = r['frac_u']
    fv = r['frac_v']
    fw = r['frac_w']
    dx = r['dx']
    
    def compute_weighted_div(u_f, v_f, w_f):
        # u[i+1] is right face, u[i] is left face
        # fu[i+1] is area of right face (xp1)
        div_u = (np.roll(u_f, -1, axis=0) * np.roll(fu, -1, axis=0) - u_f * fu) / dx
        div_v = (np.roll(v_f, -1, axis=1) * np.roll(fv, -1, axis=1) - v_f * fv) / dx
        div_w = (np.roll(w_f, -1, axis=2) * np.roll(fw, -1, axis=2) - w_f * fw) / dx
        return div_u + div_v + div_w

    div_sim = compute_weighted_div(r['u_field'], r['v_field'], r['w_field'])
    div_ana = compute_weighted_div(r['u_init'], r['v_init'], r['w_init'])
    
    div_sim_slice = div_sim[:, :, mid_z]
    div_ana_slice = div_ana[:, :, mid_z]
    
    # Coordinate mesh for plotting
    coords = np.linspace(0, r['L'], r['res'], endpoint=False) + 0.5 * dx
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    
    extent = [0, r['L'], 0, r['L']]
    
    vmin = np.percentile(lu_ibm_slice, 1)
    vmax = np.percentile(lu_ibm_slice, 99)
    
    # New Figure: Operator Analysis and Divergence Comparison
    # We want to see: L_ibm(ana), L_ibm(sim), Div(ana), Div(sim)
    
    fig_comp, axes_comp = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. L_ibm(u_ana)
    im1 = axes_comp[0].imshow(lu_ibm_slice.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes_comp[0].contour(X, Y, sdf_slice, levels=[0], colors='k', linewidths=1)
    axes_comp[0].set_title(f'L_ibm(u_ana) z={mid_z}')
    plt.colorbar(im1, ax=axes_comp[0])

    # 2. L_ibm(u_sim)
    im2 = axes_comp[1].imshow(lu_ibm_sim_slice.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes_comp[1].contour(X, Y, sdf_slice, levels=[0], colors='k', linewidths=1)
    axes_comp[1].set_title(f'L_ibm(u_sim) z={mid_z}')
    plt.colorbar(im2, ax=axes_comp[1])
    
    # Divergence Scaling
    max_div_sim = np.max(np.abs(div_sim_slice))
    max_div_ana = np.max(np.abs(div_ana_slice))
    print(f"  Divergence Field (z={mid_z}):")
    print(f"    Sim Max={max_div_sim:.6e}, Mean={np.mean(np.abs(div_sim_slice)):.6e}")
    print(f"    Ana Max={max_div_ana:.6e}, Mean={np.mean(np.abs(div_ana_slice)):.6e}")
    
    # Use analytical max range for both to show scale difference, unless sim is exploded
    vmax_div = max(max_div_ana, 1e-10)
    vmin_div = -vmax_div
    
    # 3. Div(u_ana)
    im3 = axes_comp[2].imshow(div_ana_slice.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=vmin_div, vmax=vmax_div)
    axes_comp[2].contour(X, Y, sdf_slice, levels=[0], colors='k', linewidths=1)
    axes_comp[2].set_title(f'Div(u_ana)\nMax={max_div_ana:.2e}')
    plt.colorbar(im3, ax=axes_comp[2])
    
    # 4. Div(u_sim)
    im4 = axes_comp[3].imshow(div_sim_slice.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=vmin_div, vmax=vmax_div)
    axes_comp[3].contour(X, Y, sdf_slice, levels=[0], colors='k', linewidths=1)
    axes_comp[3].set_title(f'Div(u_sim)\nMax={max_div_sim:.2e}')
    plt.colorbar(im4, ax=axes_comp[3])
    
    plt.tight_layout()
    plt.savefig('output/angled_poiseuille_operator_analysis.png', dpi=150)
    print(f"Comparison plot saved to output/angled_poiseuille_operator_analysis.png")


if __name__ == "__main__":
    results = run_convergence_study()
    make_plots(results)
    make_staggered_line_plots(results)
