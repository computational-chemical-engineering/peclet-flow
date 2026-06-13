import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def poly_D(xi): return xi * (1.0 + xi)
def poly_Nc(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nnb(xi): return xi * (1.0 - xi)
def poly_Nbc(xi): return 2.0

def debug_n32_worst_cell():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    dx = L/res_n
    
    # 45 degree setup
    sqrt2 = np.sqrt(2.0)
    
    # 1. Generate SDF
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    d = (Y - X) / sqrt2
    period = L / sqrt2
    d_wrapped = d - period * np.round(d / period)
    sdf_vals = np.abs(d_wrapped) - slab_thickness / 2.0
    sdf_flat = sdf_vals.ravel(order='F').astype(np.float32)
    
    # 2. Setup Solver
    sdf_data = pnm_backend.SDFData(
        sdf_flat, 
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
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
    f_mag = 1.0e-2
    fx = f_mag / sqrt2
    fy = f_mag / sqrt2
    fz = 0.0
    solver.set_rho(rho)
    solver.set_mu(mu)
    solver.set_body_force(pnm_backend.float3(fx, fy, fz))
    solver.set_ibm_scheme(0) # Point-Value
    
    # 3. Analytical U (Point Sampled)
    x_faces = np.linspace(0, L, res_n, endpoint=False)
    x_centers = x_faces + 0.5 * dx
    X_u, Y_u, Z_u = np.meshgrid(x_faces, x_centers, x_centers, indexing='ij')
    
    # d at U-face
    d_u = (Y_u - X_u) / sqrt2
    d_u_wrapped = d_u - period * np.round(d_u / period)
    
    H = period - slab_thickness
    half_t = slab_thickness / 2.0
    
    u_ana = np.zeros_like(d_u_wrapped)
    fluid_mask = np.abs(d_u_wrapped) > half_t
    d_wall = np.abs(d_u_wrapped[fluid_mask]) - half_t
    U_prof = (f_mag / (2.0 * nu)) * d_wall * (H - d_wall)
    u_ana[fluid_mask] = U_prof / sqrt2
    
    solver.set_u(u_ana.ravel(order='F').astype(np.float32))
    
    # 4. Extract Stencils & Scaling
    solver.step(0.0) 
    
    stencils_bare = solver.get_diffusion_stencil(0, False)
    stencils_ibm = solver.get_diffusion_stencil(0, True)
    scaling = np.array(solver.get_ibm_scaling(0)).reshape((res_n, res_n, res_n), order='F')
    
    def reshape_stencil(flat_stencil):
        return [np.array(arr).reshape((res_n, res_n, res_n), order='F') for arr in flat_stencil]

    SB = reshape_stencil(stencils_bare)
    SI = reshape_stencil(stencils_ibm)
    
    # 5. Compute Residuals
    def get_val(arr, i, j, k): return arr[i % res_n, j % res_n, k % res_n]
        
    R_ibm = np.zeros_like(u_ana)
    R_bare = np.zeros_like(u_ana)
    
    for i in range(res_n):
        for j in range(res_n):
            for k in range(res_n):
                # IBM
                val = SI[0][i,j,k] * u_ana[i,j,k]
                val += SI[1][i,j,k] * get_val(u_ana, i-1, j, k)
                val += SI[2][i,j,k] * get_val(u_ana, i+1, j, k)
                val += SI[3][i,j,k] * get_val(u_ana, i, j-1, k)
                val += SI[4][i,j,k] * get_val(u_ana, i, j+1, k)
                val += SI[5][i,j,k] * get_val(u_ana, i, j, k-1)
                val += SI[6][i,j,k] * get_val(u_ana, i, j, k+1)
                R_ibm[i,j,k] = val
                
                # Bare
                val = SB[0][i,j,k] * u_ana[i,j,k]
                val += SB[1][i,j,k] * get_val(u_ana, i-1, j, k)
                val += SB[2][i,j,k] * get_val(u_ana, i+1, j, k)
                val += SB[3][i,j,k] * get_val(u_ana, i, j-1, k)
                val += SB[4][i,j,k] * get_val(u_ana, i, j+1, k)
                val += SB[5][i,j,k] * get_val(u_ana, i, j, k-1)
                val += SB[6][i,j,k] * get_val(u_ana, i, j, k+1)
                R_bare[i,j,k] = val

    # 6. Expected Residual (Scaled)
    f_phys = -rho * fx
    R_exp = scaling * f_phys
    
    # 7. Find Worst Cell
    # Filter for cut cells only to find the specific issue
    mask_cut = (scaling != 1.0)
    if np.sum(mask_cut) == 0:
        print("No cut cells found! Check geometry.")
        return

    Error = R_ibm - R_exp
    Error_cut = Error[mask_cut]
    indices_cut = np.argwhere(mask_cut)
    
    max_err_idx = np.argmax(np.abs(Error_cut))
    max_err = Error_cut[max_err_idx]
    worst_idx_tuple = indices_cut[max_err_idx]
    
    i, j, k = worst_idx_tuple
    
    print(f"Worst Error: {max_err:.6e} at ({i}, {j}, {k})")
    
    # --- Deep Dive Report ---
    def sample_sdf(ix, iy, iz):
        return sdf_vals[ix%res_n, iy%res_n, iz%res_n]

    # Helper for trilinear (copy from previous)
    def interp_sdf(x_idx, y_idx, z_idx):
        i0 = int(np.floor(x_idx)); i1 = i0 + 1
        j0 = int(np.floor(y_idx)); j1 = j0 + 1
        k0 = int(np.floor(z_idx)); k1 = k0 + 1
        tx = x_idx - i0; ty = y_idx - j0; tz = z_idx - k0
        c000 = sample_sdf(i0, j0, k0); c100 = sample_sdf(i1, j0, k0)
        c010 = sample_sdf(i0, j1, k0); c110 = sample_sdf(i1, j1, k0)
        c001 = sample_sdf(i0, j0, k1); c101 = sample_sdf(i1, j0, k1)
        c011 = sample_sdf(i0, j1, k1); c111 = sample_sdf(i1, j1, k1)
        # Interpolate X
        c00 = c000*(1-tx) + c100*tx
        c10 = c010*(1-tx) + c110*tx
        c01 = c001*(1-tx) + c101*tx
        c11 = c011*(1-tx) + c111*tx
        # Interpolate Y
        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty
        return c0*(1-tz) + c1*tz

    # U-face SDF values
    sdf_c = interp_sdf(i - 0.5, j, k)
    sdf_w = interp_sdf(i - 1.5, j, k) # i-1
    sdf_e = interp_sdf(i + 0.5, j, k) # i+1
    sdf_s = interp_sdf(i - 0.5, j - 1, k)
    sdf_n = interp_sdf(i - 0.5, j + 1, k)
    
    print("\n--- SDF Values ---")
    print(f"C: {sdf_c:.6f}")
    print(f"W: {sdf_w:.6f}")
    print(f"E: {sdf_e:.6f}")
    print(f"S: {sdf_s:.6f}")
    print(f"N: {sdf_n:.6f}")
    
    print("\n--- IBM Geometry ---")
    D_rescale = scaling[i,j,k]
    print(f"D_rescale: {D_rescale:.6f}")
    
    def report_dir(sn, name):
        if sn < 0:
            th = sdf_c / (sdf_c - sn)
            D = poly_D(th)
            print(f"{name} Ghost: theta={th:.4f}, D={D:.4f}")
        else:
            print(f"{name} Fluid")
            
    report_dir(sdf_w, "West")
    report_dir(sdf_e, "East")
    report_dir(sdf_s, "South")
    report_dir(sdf_n, "North")
    
    print("\n--- Stencil Coefficients ---")
    print(f"{ 'Dir':<6} { 'Bare':<14} { 'IBM':<14} {'Ratio/Diff'}")
    
    dirs = ['C', 'W', 'E', 'S', 'N', 'B', 'T']
    for d_idx, d_name in enumerate(dirs):
        bare = SB[d_idx][i,j,k]
        ibm = SI[d_idx][i,j,k]
        print(f"{d_name:<6} {bare:<14.4e} {ibm:<14.4e}")
        
    print("\n--- Values ---")
    print(f"u_C: {u_ana[i,j,k]:.6e}")
    print(f"u_W: {get_val(u_ana, i-1, j, k):.6e}")
    print(f"u_E: {get_val(u_ana, i+1, j, k):.6e}")
    print(f"u_S: {get_val(u_ana, i, j-1, k):.6e}")
    print(f"u_N: {get_val(u_ana, i, j+1, k):.6e}")
    
    print(f"\n--- Residuals ---")
    print(f"R_bare: {R_bare[i,j,k]:.6e}")
    print(f"R_ibm:  {R_ibm[i,j,k]:.6e}")
    print(f"R_exp:  {R_exp[i,j,k]:.6e}")
    print(f"Error:  {Error[i,j,k]:.6e}")

if __name__ == "__main__":
    debug_n32_worst_cell()
