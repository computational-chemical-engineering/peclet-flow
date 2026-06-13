
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def generate_angled_slab_sdf(res_n, L, slab_thickness):
    dx = L / res_n
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    sqrt2 = np.sqrt(2.0)
    d = (Y - X) / sqrt2
    period = L / sqrt2
    d_wrapped = d - period * np.round(d / period)
    sdf = np.abs(d_wrapped) - slab_thickness / 2.0
    return sdf.ravel(order='F').astype(np.float32)

def analytical_component_at_points(x, y, L, slab_thickness, f_mag, nu, comp):
    sqrt2 = np.sqrt(2.0)
    period = L / sqrt2
    half_t = slab_thickness / 2.0
    H = period - slab_thickness
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

def debug_residual_budget():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    ibm_scheme = 0 # Point-Value
    
    print(f"Debug Residual Budget N={res_n}")

    sqrt2 = np.sqrt(2.0)
    dx = L / res_n
    
    sdf_values = generate_angled_slab_sdf(res_n, L, slab_thickness)
    
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(
        pnm_backend.SDFData(
            sdf_values,
            pnm_backend.int3(res_n, res_n, res_n),
            pnm_backend.float3(0, 0, 0),
            pnm_backend.float3(dx, dx, dx)
        )
    )
    solver.set_ibm_scheme(ibm_scheme)

    rho = 1.0
    nu = 0.01
    mu = nu * rho
    f_mag = 1.0e-2
    f_dir = np.array([1.0/sqrt2, 1.0/sqrt2, 0.0])
    fx, fy, fz = f_mag * f_dir

    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))

    # Analytical Init
    x_faces = np.linspace(0, L, res_n, endpoint=False)
    x_centers = x_faces + 0.5 * dx
    X_u, Y_u, Z_u = np.meshgrid(x_faces, x_centers, x_centers, indexing='ij')
    
    u_init = analytical_component_at_points(X_u, Y_u, L, slab_thickness, f_mag, nu, 'u')
    solver.set_u(u_init.ravel(order='F').astype(np.float32))
    
    # Extract Stencils
    stencil_u_ibm_flat = solver.get_diffusion_stencil(0, True)
    ibm_scaling_flat = solver.get_ibm_scaling(0)
    
    def reshape_stencil(flat_stencil, n):
        return [np.array(arr).reshape((n, n, n), order='F') for arr in flat_stencil]
        
    stencil_u_ibm = reshape_stencil(stencil_u_ibm_flat, res_n)
    ibm_scaling = np.array(ibm_scaling_flat).reshape((res_n, res_n, res_n), order='F')
    u_field = u_init.reshape((res_n, res_n, res_n), order='F')
    
    # Calculate Residual Field
    def apply_stencil(field, stencil):
        res = stencil[0] * field
        res += stencil[1] * np.roll(field, 1, axis=0)
        res += stencil[2] * np.roll(field, -1, axis=0)
        res += stencil[3] * np.roll(field, 1, axis=1)
        res += stencil[4] * np.roll(field, -1, axis=1)
        res += stencil[5] * np.roll(field, 1, axis=2)
        res += stencil[6] * np.roll(field, -1, axis=2)
        return res

    Lu = apply_stencil(u_field, stencil_u_ibm)
    expected = -rho * fx
    resid = Lu - ibm_scaling * expected
    
    # Find Worst Cell
    # Use Face Mask
    def get_face_sdf(s): return 0.5 * (s + np.roll(s, 1, axis=0))
    sdf_reshaped = sdf_values.reshape((res_n, res_n, res_n), order='F')
    sdf_face = get_face_sdf(sdf_reshaped)
    ibm_mask = (sdf_face > 0) & (sdf_face < 0.05)
    
    resid_masked = np.zeros_like(resid)
    resid_masked[ibm_mask] = resid[ibm_mask]
    
    idx_flat = np.argmax(np.abs(resid_masked))
    idx = np.unravel_index(idx_flat, (res_n, res_n, res_n), order='F')
    
    print(f"\nWorst Cell Index: {idx}")
    print(f"  SDF (Face): {sdf_face[idx]:.6e}")
    print(f"  Scaling (D): {ibm_scaling[idx]:.6f}")
    print(f"  Residual: {resid[idx]:.6e}")
    print(f"  L(u): {Lu[idx]:.6e}")
    print(f"  D * f: {ibm_scaling[idx] * expected:.6e}")
    print(f"  Analytic f: {expected:.6e}")
    
    # Detailed Stencil Analysis
    print("\nStencil Breakdown:")
    names = ["C", "W", "E", "S", "N", "B", "T"]
    offsets = [(0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    sum_Au = 0.0
    
    for k, name in enumerate(names):
        off = offsets[k]
        nb_idx = ((idx[0]+off[0])%res_n, (idx[1]+off[1])%res_n, (idx[2]+off[2])%res_n)
        coeff = stencil_u_ibm[k][idx]
        val = u_field[nb_idx]
        term = coeff * val
        sum_Au += term
        print(f"  {name}: Coeff={coeff:12.4e}, U_nb={val:12.4e}, Term={term:12.4e} (Idx {nb_idx})")
        
    print(f"  Sum(A*u) = {sum_Au:.6e}")
    print(f"  Mismatch = {sum_Au - ibm_scaling[idx]*expected:.6e}")

if __name__ == "__main__":
    debug_residual_budget()
