import numpy as np
import sys
import os

# Add build/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import pnm_backend

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

    print(f"Saved {filename} (XML VTI)")

def interpolate_to_cell_centers(u, v, w):
    """
    Interpolate staggered velocities to cell centers.

    Staggered grid positions (MAC grid):
    - u at (i, j+0.5, k+0.5) - on x-faces
    - v at (i+0.5, j, k+0.5) - on y-faces
    - w at (i+0.5, j+0.5, k) - on z-faces

    Cell centers at (i+0.5, j+0.5, k+0.5).

    With periodic boundaries, we average u[i] and u[i+1] to get u at cell center i.
    """
    # u: average in x direction
    u_c = 0.5 * (u + np.roll(u, -1, axis=0))
    # v: average in y direction
    v_c = 0.5 * (v + np.roll(v, -1, axis=1))
    # w: average in z direction
    w_c = 0.5 * (w + np.roll(w, -1, axis=2))
    return u_c, v_c, w_c


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
    max_iter = 1000 # Pressure iterations
    tol = 1e-4
    
    solver.set_rho(rho)
    solver.set_mu(nu * rho)
    solver.set_pressure_solver_params(iter=max_iter)
    
    steps = 100
    print(f"Running {steps} steps flow around sphere...")
    
    for i in range(steps):
        solver.step(dt)
        if i % 20 == 0:
            print(f"Step {i}")
            
    # Get results (staggered grid)
    u = np.array(solver.get_u()).reshape(res, order='F')
    v = np.array(solver.get_v()).reshape(res, order='F')
    w = np.array(solver.get_w()).reshape(res, order='F')
    p = np.array(solver.get_p()).reshape(res, order='F')

    # Interpolate velocities to cell centers for VTI output
    u_c, v_c, w_c = interpolate_to_cell_centers(u, v, w)

    # Save (cell-centered data)
    save_vti("flow_around_sphere.vti", res, spacing, u_c, v_c, w_c, p, sdf_field)
    
    # Stats
    print("Max U:", np.max(u))
    print("Min U (should be 0 or <0 in recirculation):", np.min(u))
    print("SDF check: U inside sphere should be 0.")
    
    # Verify center
    center_idx = (nx//2, ny//2, nz//2)
    print(f"U at center {center_idx} (Solid): {u[center_idx]}")

if __name__ == "__main__":
    simulate_flow()
