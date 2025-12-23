import numpy as np
import sys
import os

# Add build/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import pnm_backend

def save_vti(filename, res, spacing, u, v, w, p, sdf):
    """
    Save fields to VTI (XML PolyData) format.
    Uses appended raw binary format (Little Endian).
    """
    nx, ny, nz = res
    dx, dy, dz = spacing
    
    with open(filename, 'wb') as f:
        # XML Header
        f.write(b'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="0 0 0" Spacing="{dx} {dy} {dz}">\n'.encode('ascii'))
        f.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n'.encode('ascii'))
        f.write(b'      <PointData Scalars="pressure" Vectors="velocity">\n')
        
        offset = 0
        
        # Velocity (3 components * 4 bytes)
        vel_len = nx * ny * nz * 3 * 4
        f.write(f'        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="appended" offset="{offset}"/>\n'.encode('ascii'))
        offset += vel_len + 8 # 8 bytes for UInt64 length header
        
        # Pressure (1 component * 4 bytes)
        p_len = nx * ny * nz * 1 * 4
        f.write(f'        <DataArray type="Float32" Name="pressure" NumberOfComponents="1" format="appended" offset="{offset}"/>\n'.encode('ascii'))
        offset += p_len + 8
        
        # SDF
        f.write(f'        <DataArray type="Float32" Name="sdf" NumberOfComponents="1" format="appended" offset="{offset}"/>\n'.encode('ascii'))
        
        f.write(b'      </PointData>\n')
        f.write(b'    </Piece>\n')
        f.write(b'  </ImageData>\n')
        
        # Appended Data Section
        f.write(b'  <AppendedData encoding="raw">\n')
        f.write(b'    _')
        
        def write_chunk(data):
            # Ensure Little Endian floats
            flat = data.astype('<f4')
            nbytes = flat.nbytes
            f.write(np.array([nbytes], dtype=np.uint64).tobytes())
            flat.tofile(f)
            
        # 1. Velocity: Interleave (u,v,w)
        # u, v, w are F-order (x-fastest), numpy default flatten is C-order.
        # We need (u0, v0, w0), (u1, v1, w1)...
        # column_stack does exactly this row-wise if inputs are 1D arrays.
        u_flat = u.flatten('F')
        v_flat = v.flatten('F')
        w_flat = w.flatten('F')
        vel_stacked = np.column_stack((u_flat, v_flat, w_flat)).flatten()
        write_chunk(vel_stacked)
        
        # 2. Pressure
        write_chunk(p.flatten('F'))
        
        # 3. SDF
        write_chunk(sdf.flatten('F'))
        
        f.write(b'\n  </AppendedData>\n')
        f.write(b'</VTKFile>\n')
        
    print(f"Saved {filename} (XML VTI)")

def create_sphere_packing(res, radius):
    nx, ny, nz = res
    sdf = np.ones((nx, ny, nz), dtype=np.float32) * 1e9
    
    # Sphere at center
    cx, cy, cz = nx//2, ny//2, nz//2
    
    # Create coordinate grids
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - radius
    
    return dist

def simulate_flow():
    nx, ny, nz = 64, 64, 64
    res = (nx, ny, nz)
    spacing = (1.0, 1.0, 1.0)
    
    solver = pnm_backend.CFDSolver(pnm_backend.int3(nx, ny, nz), 
                                   pnm_backend.float3(1.0, 1.0, 1.0))
    
    # Create SDF: Sphere of radius 10 in center
    sdf_field = create_sphere_packing(res, 10.0)
    
    # Flatten for C++ (F-order)
    sdf_flat = sdf_field.flatten(order='F')
    
    solver.initialize(pnm_backend.SDFData(sdf_flat, pnm_backend.int3(nx, ny, nz), pnm_backend.float3(0,0,0), pnm_backend.float3(1,1,1)))
    
    # Body Force in X
    force_x = 0.001
    solver.set_body_force(pnm_backend.float3(force_x, 0.0, 0.0))
    
    # Parameters
    dt = 0.01
    rho = 1.0
    nu = 0.5 # Viscous flow
    max_iter = 20 # Pressure iterations
    tol = 1e-4
    
    steps = 100
    print(f"Running {steps} steps flow around sphere...")
    
    for i in range(steps):
        solver.step(dt, rho, nu, max_iter, tol)
        if i % 20 == 0:
            print(f"Step {i}")
            
    # Get results
    u = np.array(solver.get_u()).reshape(res, order='F')
    v = np.array(solver.get_v()).reshape(res, order='F')
    w = np.array(solver.get_w()).reshape(res, order='F')
    p = np.array(solver.get_p()).reshape(res, order='F')
    
    # Save
    save_vti("flow_around_sphere.vti", res, spacing, u, v, w, p, sdf_field)
    
    # Stats
    print("Max U:", np.max(u))
    print("Min U (should be 0 or <0 in recirculation):", np.min(u))
    print("SDF check: U inside sphere should be 0.")
    
    # Verify center
    center_idx = (nx//2, ny//2, nz//2)
    print(f"U at center {center_idx} (Solid): {u[center_idx]}")

if __name__ == "__main__":
    simulate_flow()
