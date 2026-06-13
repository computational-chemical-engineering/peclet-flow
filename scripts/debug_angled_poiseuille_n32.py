import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def poly_D(xi): return xi * (1.0 + xi)
def poly_Nc(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nnb(xi): return xi * (1.0 - xi)

def debug_n32():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    dx = L/res_n
    
    # 45 degree setup
    sqrt2 = np.sqrt(2.0)
    
    # 1. Generate SDF (Slab)
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    # Center plane y = x. Distance d = (y-x)/sqrt(2)
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
    # U faces: (i*dx, (j+0.5)dx, (k+0.5)dx)
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
    
    # Set to solver
    solver.set_u(u_ana.ravel(order='F').astype(np.float32))
    
    # 4. Extract Stencils & Scaling
    # Trigger geometry update
    solver.step(0.0) 
    
    stencils = solver.get_diffusion_stencil(0, True) # 0 for U
    scaling = np.array(solver.get_ibm_scaling(0)).reshape((res_n, res_n, res_n), order='F')
    
    # Reshape stencils
    A_C = np.array(stencils[0]).reshape((res_n, res_n, res_n), order='F')
    A_W = np.array(stencils[1]).reshape((res_n, res_n, res_n), order='F')
    A_E = np.array(stencils[2]).reshape((res_n, res_n, res_n), order='F')
    A_S = np.array(stencils[3]).reshape((res_n, res_n, res_n), order='F')
    A_N = np.array(stencils[4]).reshape((res_n, res_n, res_n), order='F')
    A_B = np.array(stencils[5]).reshape((res_n, res_n, res_n), order='F')
    A_T = np.array(stencils[6]).reshape((res_n, res_n, res_n), order='F')
    
    # 5. Compute Residual R = A' u
    # Need to handle periodicity manually or use roll
    # Note: Stencil A_W applies to u[i-1].
    
    def get_val(arr, i, j, k):
        return arr[i % res_n, j % res_n, k % res_n]
        
    R = np.zeros_like(u_ana)
    
    for i in range(res_n):
        for j in range(res_n):
            for k in range(res_n):
                val = A_C[i,j,k] * u_ana[i,j,k]
                val += A_W[i,j,k] * get_val(u_ana, i-1, j, k)
                val += A_E[i,j,k] * get_val(u_ana, i+1, j, k)
                val += A_S[i,j,k] * get_val(u_ana, i, j-1, k)
                val += A_N[i,j,k] * get_val(u_ana, i, j+1, k)
                val += A_B[i,j,k] * get_val(u_ana, i, j, k-1)
                val += A_T[i,j,k] * get_val(u_ana, i, j, k+1)
                R[i,j,k] = val
                
    # 6. Expected Residual
    # L(u) = -rho * fx (since mu*lap(u) + rho*g = 0 -> mu*lap = -rho*g)
    # Scaled: D * (-rho * fx)
    f_phys = -rho * fx
    R_exp = scaling * f_phys
    
    Error = R - R_exp
    
    # 7. Identify Bad Cells
    # Check only fluid cells
    # Note: u_ana definition uses d_u > half_t.
    # Solver uses interpolated SDF > 0.
    # Interpolation of SDF to U-face might differ from analytical d_u calculation?
    # Analytical d_u is exact distance to plane.
    # Solver SDF is trilinear interp of cell-centered SDF.
    # This difference might be the cause?
    # But for a plane aligned at 45 deg, trilinear interp of exact SDF is exact?
    # d(x,y) = (y-x)/sqrt2. Linear in x, y.
    # Trilinear interp of linear function is exact.
    # So SDF interpolation should be exact.
    
    # We rely on solver's fluid classification.
    # If scaling == 1.0, it might be fluid or interior.
    # If scaling != 1.0, it is cut cell.
    
    # Let's filter for cut cells first
    mask_cut = (scaling != 1.0)
    
    print(f"Total Cut Cells: {np.sum(mask_cut)}")
    
    # Filter Error
    Error_cut = Error[mask_cut]
    indices_cut = np.argwhere(mask_cut)
    
    max_err_idx = np.argmax(np.abs(Error_cut))
    max_err = Error_cut[max_err_idx]
    worst_idx_tuple = indices_cut[max_err_idx] # (i, j, k) in flattened argwhere? No, argwhere returns list of coords.
    
    print(f"Worst Error in Cut Cells: {max_err:.6e}")
    
    i, j, k = worst_idx_tuple
    print(f"\n--- Investigating Cell ({i}, {j}, {k}) ---")
    
    # Get SDF values used by kernel
    # Need to emulate sample_sdf_interp
    # U-face offset: (-0.5, 0, 0)
    # Centered SDF array: sdf_data.sdf_values
    
    def sample_sdf(ix, iy, iz):
        # Emulate device trilinear interp
        # For U-face at (i, j+0.5, k+0.5), we sample at idx - 0.5 in x.
        # coords: x = i - 0.5.

        # This python script SDF is Fortran (x,y,z).
        # To match device, we need to be careful with indices.
        # device: z*ny*nx + ...
        # python: [x, y, z]
        
        # Let's just use the analytical formula for debugging, 
        # but also check if grid values match.
        
        # Grid values at (i, j, k)
        val = sdf_vals[ix%res_n, iy%res_n, iz%res_n]
        return val

    # Helper for trilinear
    # x in index space.
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
        
        # Interpolate Z
        return c0*(1-tz) + c1*tz

    # U-face location in index space (relative to centers)
    # Center at (i+0.5).
    # U-face at i.
    # So offset from center is -0.5.
    # interp at (i-0.5, j, k)
    
    sdf_c = interp_sdf(i - 0.5, j, k)
    sdf_e = interp_sdf(i + 0.5, j, k) # i+1
    sdf_w = interp_sdf(i - 1.5, j, k) # i-1
    sdf_n = interp_sdf(i - 0.5, j + 1, k)
    sdf_s = interp_sdf(i - 0.5, j - 1, k)
    
    print(f"SDF Interp:")
    print(f"  C: {sdf_c:.6f}")
    print(f"  W: {sdf_w:.6f} (ghost? {sdf_w < 0})")
    print(f"  E: {sdf_e:.6f} (ghost? {sdf_e < 0})")
    print(f"  S: {sdf_s:.6f} (ghost? {sdf_s < 0})")
    print(f"  N: {sdf_n:.6f} (ghost? {sdf_n < 0})")
    
    # Calculate Xi and D
    def get_D(sc, sn):
        if sn < 0:
            theta = sc / (sc - sn)
            theta = max(1e-4, min(1.0, theta))
            return poly_D(theta)
        return 1e9
        
    Dw = get_D(sdf_c, sdf_w)
    De = get_D(sdf_c, sdf_e)
    Ds = get_D(sdf_c, sdf_s)
    Dn = get_D(sdf_c, sdf_n)
    
    D_list = [Dw, De, Ds, Dn]
    min_D = min(D_list)
    
    # Check sandwich (W and E ghost?)
    is_sw_x = (sdf_w < 0 and sdf_e < 0)
    is_sw_y = (sdf_s < 0 and sdf_n < 0)
    
    print(f"  Dw={Dw:.4f}, De={De:.4f}, Ds={Ds:.4f}, Dn={Dn:.4f}")
    print(f"  Sandwich X: {is_sw_x}, Sandwich Y: {is_sw_y}")
    
    print(f"  Solver D_rescale: {scaling[i,j,k]:.6f}")
    print(f"  Calculated min_D: {min_D:.6f}")
    
    # Check Coefficients
    print(f"Stencils:")
    print(f"  A_C: {A_C[i,j,k]:.6e}")
    print(f"  A_W: {A_W[i,j,k]:.6e}")
    print(f"  A_E: {A_E[i,j,k]:.6e}")
    print(f"  A_S: {A_S[i,j,k]:.6e}")
    print(f"  A_N: {A_N[i,j,k]:.6e}")
    
    # Manual Calc A_C
    # A_orig_C = -2 * mu * (1/dx^2 + 1/dy^2 + 1/dz^2) = -6 * mu/dx^2
    # A_orig_nb = mu/dx^2
    
    # If West is ghost:
    # A_C += A_orig_W * K_w
    # K_w = Nc(theta_w) * R
    # R = D_res / D_w
    
    h2 = dx**2
    base_ac = -6.0 * mu / h2
    base_anb = mu / h2
    
    print(f"  Base A_C: {base_ac:.6e}")
    print(f"  Base A_nb: {base_anb:.6e}")
    
    # Reconstruct A_C
    ac_calc = base_ac * scaling[i,j,k]
    
    # Check all 6 neighbors
    def add_correction(sn, name):
        nonlocal ac_calc
        if sn < 0:
            th = sdf_c / (sdf_c - sn)
            th = max(1e-4, min(1.0, th))
            R = scaling[i,j,k] / poly_D(th)
            K = poly_Nc(th) * R
            ac_calc += base_anb * K
            print(f"  + {name} Correction: K={K:.4f} (th={th:.4f}, R={R:.4f})")

    add_correction(sdf_w, "West")
    add_correction(sdf_e, "East")
    add_correction(sdf_s, "South")
    add_correction(sdf_n, "North")
    add_correction(interp_sdf(i, j, k-1.5), "Bottom") # Approximate z neighbor check
    add_correction(interp_sdf(i, j, k+0.5), "Top")
        
    print(f"  Reconstructed A_C: {ac_calc:.6e}")
    print(f"  Diff A_C: {ac_calc - A_C[i,j,k]:.6e}")
    
    # Check Residual
    # R = ac_calc * u_c + ...
    # u_c = u_ana[i,j,k]
    
    print(f"  A_W from solver: {A_W[i,j,k]:.6e}")
    if sdf_w < 0:
        print("  West is ghost, A_W should be 0.")
        
    # Is u_c correct?
    print(f"  u_c (ana): {u_ana[i,j,k]:.6e}")
    
    # Let's calculate R manually
    R_man = ac_calc * u_ana[i,j,k]
    if sdf_w >= 0: R_man += base_anb * scaling[i,j,k] * u_ana[i-1,j,k]
    if sdf_e >= 0: R_man += base_anb * scaling[i,j,k] * u_ana[i+1,j,k]
    if sdf_s >= 0: R_man += base_anb * scaling[i,j,k] * u_ana[i,j-1,k]
    if sdf_n >= 0: R_man += base_anb * scaling[i,j,k] * u_ana[i,j+1,k]
    # Z neighbors (always fluid in this case)
    R_man += base_anb * scaling[i,j,k] * (u_ana[i,j,k-1] + u_ana[i,j,k+1])
    
    print(f"  Manual R: {R_man:.6e}")
    print(f"  Solver R: {R[int(i),int(j),int(k)]:.6e}")
    print(f"  Expected R: {R_exp[int(i),int(j),int(k)]:.6e}")
    
    # Calculate geometric error term
    # Exact Laplacian = f_phys. 
    # Discrete Op = D * f_phys + Error?
    # Or is it D * f_phys exactly?
    # For 1D quadratic it was exact.
    # For 3D rotated, we have cross terms?
    # Stencil is 7-point.
    # 7-point Laplacian is NOT exact for rotated quadratic?
    # Rotated quadratic: x^2 + y^2 - 2xy.
    # d2/dx2 + d2/dy2 = 2 + 2 = 4.
    # Cross term -2xy. 
    # d2/dx2 (-2xy) = 0.
    # So 7-point stencil IS exact for rotated quadratic!
    
    # BUT, only if grid points align?
    # No, finite difference of quadratic is exact on any uniform grid.
    
    # So error must be in IBM boundary handling.
    # If multiple ghosts?
    # Case: West Ghost (xi_w) AND South Ghost (xi_s).
    # Corner.
    # D_rescale = min(|Dw|, |Ds|).
    # Say Dw is min. R_w = 1. R_s = Dw/Ds.
    # A_C = Dw * A_c_orig + K_w * A_nb + K_s * R_s * A_nb.
    
    # Is this composition exact?
    # 1D: L_1 = (N u + ...)/D -> f.
    # 2D Corner:
    # We enforce BC in x-dir: u_g_w = ...
    # We enforce BC in y-dir: u_g_s = ...
    # Does the sum of modifications equal D * f?
    # This is the question.
    
if __name__ == "__main__":
    debug_n32()
