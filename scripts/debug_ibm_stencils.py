import numpy as np
import sys
import os

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

def debug_stencils():
    res_n = 32
    L = 1.0
    slab_thickness = 0.2
    dx = L/res_n
    mu = 0.01
    
    print(f"Debug N={res_n} Slab - Missed Neighbor Scan")
    
    sdf_values = generate_angled_slab_sdf(res_n, L, slab_thickness)
    sdf_reshaped = sdf_values.reshape((res_n, res_n, res_n), order='F')
    
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n), 
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(pnm_backend.SDFData(
        sdf_values,
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dx, dx)
    ))
    solver.set_mu(mu)
    solver.set_ibm_scheme(0) # Point Value
    
    # Run dummy step
    solver.step(0.01)
    
    # Get scaling to detect IBM status
    ibm_scaling_flat = solver.get_ibm_scaling(0) # U component
    ibm_scaling = np.array(ibm_scaling_flat).reshape((res_n, res_n, res_n), order='F')
    
    # Interpolation Helper
    def sample_sdf_interp(x_idx, y_idx, z_idx):
        # Emulates GPU texture fetching / linear interpolation
        # x_idx, y_idx, z_idx are FLOATS representing the sampling coordinate
        # relative to cell CENTERS (0,0,0) -> (N-1, N-1, N-1).
        
        # Wrapping
        x = x_idx % res_n
        y = y_idx % res_n
        z = z_idx % res_n
        
        fx = np.floor(x)
        fy = np.floor(y)
        fz = np.floor(z)
        
        wx = x - fx
        wy = y - fy
        wz = z - fz
        
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        
        x0, x1 = ix, (ix+1)%res_n
        y0, y1 = iy, (iy+1)%res_n
        z0, z1 = iz, (iz+1)%res_n
        
        c000 = sdf_reshaped[x0, y0, z0]
        c100 = sdf_reshaped[x1, y0, z0]
        c010 = sdf_reshaped[x0, y1, z0]
        c110 = sdf_reshaped[x1, y1, z0]
        c001 = sdf_reshaped[x0, y0, z1]
        c101 = sdf_reshaped[x1, y0, z1]
        c011 = sdf_reshaped[x0, y1, z1]
        c111 = sdf_reshaped[x1, y1, z1]
        
        c00 = c000*(1-wx) + c100*wx
        c10 = c010*(1-wx) + c110*wx
        c01 = c001*(1-wx) + c101*wx
        c11 = c011*(1-wx) + c111*wx
        
        c0 = c00*(1-wy) + c10*wy
        c1 = c01*(1-wy) + c11*wy
        
        return c0*(1-wz) + c1*wz

    # Scan for Solid IBM Cells
    solid_ibm_count = 0
    fluid_ibm_count = 0
    
    for k in range(res_n):
        for j in range(res_n):
            for i in range(res_n):
                u_center_loc = (i - 0.5, j, k)
                sdf_c = sample_sdf_interp(*u_center_loc)
                is_ibm = (ibm_scaling[i,j,k] != 1.0)
                
                if is_ibm:
                    if sdf_c < 0:
                        solid_ibm_count += 1
                        if solid_ibm_count <= 3:
                            print(f"SOLID IBM CELL (U) at ({i},{j},{k}): SDF_C={sdf_c:.6e}, D={ibm_scaling[i,j,k]:.4f}")
                    else:
                        fluid_ibm_count += 1

    print(f"\nTotal Fluid IBM Cells: {fluid_ibm_count}")
    print(f"Total Solid IBM Cells: {solid_ibm_count}")

    # Specific check for (6,1,0)
    target = (6, 1, 0)
    u_loc_c = sample_sdf_interp(target[0]-0.5, target[1], target[2])
    print(f"\n--- Specific Check ({target[0]},{target[1]},{target[2]}) ---")
    print(f"  SDF (Face): {u_loc_c:.6e}")
    print(f"  Is IBM: {(ibm_scaling[target] != 1.0)}")
    print(f"  D: {ibm_scaling[target]:.4f}")

if __name__ == "__main__":
    debug_stencils()
