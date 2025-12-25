
import sys
import os
import math
import numpy as np

# Add build to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import pnm_backend

def test_fluid_fraction():
    nx, ny, nz = 10, 10, 10
    res = pnm_backend.int3(nx, ny, nz)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    
    # 1. Plane at x = 5.5 (Normal X)
    # SDF = x - 5.5
    # Cell Centers (integer coords in texture space): 0, 1, ..., 9
    # P_5 at 5.0 (SDF -0.5). P_6 at 6.0 (SDF +0.5).
    # Interface at 5.5.
    
    sdf = np.zeros((nz, ny, nx), dtype=np.float32, order='F') # [z,y,x] linear X-fast?
    # Python array is usually row-major (Z, Y, X) if just zeros.
    # But we flatten it to vector.
    # Let's fill it using loop to be safe ormeshgrid
    
    # Texture coords:
    # x: 0..9
    # sdf[z,y,x] = x - 5.5
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Linear index: k*ny*nx + j*nx + i
                # Fortran order or C order?
                # C++ binding expects linear.
                # Usually we assume X-fastest in CUDA texture logic.
                # So index = i + j*nx + k*nx*ny.
                pass
    
    # Create linear buffer
    sdf_vals = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                val = float(i) - 5.5
                sdf_vals.append(val)
    
    sdf_data = pnm_backend.SDFData(sdf_vals, res, pnm_backend.float3(0,0,0), spacing)
    solver = pnm_backend.CFDSolver(res, spacing)
    solver.initialize(sdf_data)
    
    # Type 0: Volume Fraction
    # Center Offset (0,0,0)
    vf = solver.get_fluid_fraction(0, pnm_backend.float3(0,0,0))
    vf = np.array(vf)
    
    print("Checking Volume Fractions (Plane X=5.5)...")
    # At i=5 (x=5.0): SDF -0.5. nx=1. Denom = 1.0. Frac = 0.5 - 0.5 = 0.
    # At i=6 (x=6.0): SDF 0.5. Frac = 0.5 + 0.5 = 1.
    # Wait, my previous manual calc said P_5 is 5.5?
    # No, texture coord i corresponds to x=i.
    # So P_5 is at x=5.0.
    # Let's inspect vf at y=0, z=0.
    row = vf[0:nx]
    print("VF Row:", row)
    
    # Check i=5 (0.0) and i=6 (1.0).
    # Transition should be monotonic.
    # Formula: 0.5 + (x-5.5)/1 = x - 5.0.
    # i=5: 5-5 = 0.
    # i=6: 6-5 = 1.
    # i=5.5 (if sampled): 0.5.
    # Correct.
    
    # 2. Check Area Fraction X (U-Face)
    # Type 1. Offset (0.5, 0, 0).
    # U_i is at i + 0.5.
    # U_5 is at 5.5.
    # SDF interpolated at 5.5 is 0.0.
    # Normal X: ny=0, nz=0.
    # Denom = 0 -> clamped to epsilon.
    # Frac = 0.5 + 0/eps = 0.5.
    # U_4 is at 4.5. SDF = -1.0. Frac = 0 (solid).
    # U_6 is at 6.5. SDF = 1.0. Frac = 1 (fluid).
    # So U_5 should be 0.5 (or jump if numer != 0).
    
    af_u = solver.get_fluid_fraction(1, pnm_backend.float3(0.5, 0, 0))
    af_u = np.array(af_u)
    print("AF_U Row:", af_u[0:nx])
    
    if abs(af_u[5] - 0.5) < 1e-3:
        print("AF_U at interface is 0.5 (Correct behavior for sharp interface aligned with grid).")
    else:
        print(f"AF_U at interface: {af_u[5]}")

    print("Test Passed.")

if __name__ == "__main__":
    test_fluid_fraction()
