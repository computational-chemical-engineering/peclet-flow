import sys
import os
import numpy as np

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import pnm_backend
except ImportError as e:
    print(f"Failed to import pnm_backend: {e}")
    sys.exit(1)

def test_ibm_sphere_flow():
    """
    Test flow past a sphere using IBM to verify stability and crash fix.
    """
    nx, ny, nz = 32, 32, 32
    res = pnm_backend.int3(nx, ny, nz)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    origin = pnm_backend.float3(0.0, 0.0, 0.0)
    
    # Create Solver
    solver = pnm_backend.CFDSolver(res, spacing)
    
    # Create SDF: Sphere at center
    cx, cy, cz = nx/2, ny/2, nz/2
    radius = 8.0
    
    # Grid coordinates
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Dist = sqrt(dx^2 + dy^2 + dz^2) - R
    # Inside sphere (Solid) -> dist < 0
    # But convention is often: SDF < 0 is Fluid?
    # Let's check cfd_solver code: "if (sdf_c < 0.0f) ... return" (Ignore)
    # This implies Negative = Solid / Locked / Omitted?
    # Wait, in populate_ghost: "if (idx >= num_ibm_cells) return"
    # In check_id_map: "if (sdf[idx] < 0.0f) u=0"
    # This clearly means Negative SDF = Solid.
    # So Dist < 0 = Solid.
    # Dist > 0 = Fluid.
    
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) - radius
    
    # Invert sign if I want Sphere to be Solid?
    # dist < 0 inside sphere. So inside is Solid. Correct.
    
    # However, standard signed distance:
    # d(x) = min dist to boundary. Negative inside.
    
    # Flatten array (order must match C++ expectation)
    # C++ reader expects flat array. 
    # Python binding: SDFData takes vector<float>.
    # layout: z*ny*nx + y*nx + x? 
    # Binding: py::array_t<float>({d.resolution[2], ...}) -> Shape (Z, Y, X)
    # The binding constructor takes vector.
    # We should flatten in 'C' order if we pass to vector, but Binding might copy?
    # Wait, binding lambda:
    # "return SDFData{sdf_values...}"
    # So it just takes the list/array.
    # If I pass numpy array, pybind converts.
    # Memory layout:
    # The helper `get_idx` uses: z * res.y * res.x + y * res.x + x.
    # This corresponds to Z-slowest, X-fastest.
    # In Numpy 'C' order for shape (D, H, W) -> (Z, Y, X), the last dim is X (fastest).
    # So flattening a (Nz, Ny, Nx) array in 'C' order gives appropriate layout.
    
    # My meshgrid (X, Y, Z) with indexing='ij':
    # Shape is (Nx, Ny, Nz).
    # If I flatten this: x changes fastest? No, with 'ij', x is dim 0.
    # In 'C' order flatten: dim 0 is slowest.
    # So if I have (Nx, Ny, Nz), flattening makes X slowest. This is WRONG.
    # I want X fastest.
    # So I should Transpose to (Nz, Ny, Nx) then flatten? 
    # Or just use indexing='xy' (Cartesian)? No, 'xy' is weird for 3D.
    
    # Let's verify coordinates:
    # get_idx(x,y,z) = z*... + y*... + x
    # This assumes x is innermost.
    # So array should be stored as [z0 y0 x0, z0 y0 x1, ... ]
    # In Python shape (Nz, Ny, Nx):
    # arr[z, y, x].
    # Flat: z changes slowest. x changes fastest.
    # So I need to construct (Nz, Ny, Nx) array.
    
    X_t = X.transpose((2, 1, 0)) # (Nz, Ny, Nx)
    Y_t = Y.transpose((2, 1, 0))
    Z_t = Z.transpose((2, 1, 0))
    
    dist_t = np.sqrt((X_t-cx)**2 + (Y_t-cy)**2 + (Z_t-cz)**2) - radius
    
    # Ensure -1 inside, +1 outside (roughly)
    # Current: inside sphere dist < 0.
    # Outside > 0.
    # Solvers expects SDF < 0 -> Solid. (Verified).
    # So this is correct.
    
    sdf_flat = dist_t.flatten().astype(np.float32)
    
    # Create SDFData
    sdf_obj = pnm_backend.SDFData(sdf_flat, res, origin, spacing)
    
    # Initialize Solver
    print("Initializing Solver with Sphere SDF...")
    solver.initialize(sdf_obj)
    
    # Apply Force
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    
    steps = 50
    dt = 0.05
    rho = 1.0
    mu = 0.01 # Viscosity
    
    print(f"Running {steps} steps with IBM...")
    for i in range(steps):
        if i % 10 == 0:
            print(f"Step {i}")
        solver.step(dt, rho, mu, 10, 1e-4) # 10 iter pressure
        
    print("Simulation completed successfully.")
    
    # Check output
    u = np.array(solver.get_u()).reshape((nz, ny, nx)) # Z Y X
    # Check velocity inside sphere (should be 0)
    # Center is at [nz/2, ny/2, nx/2]
    # Indices:
    cz_i, cy_i, cx_i = int(nz/2), int(ny/2), int(nx/2)
    
    u_center = u[int(nz/2), int(ny/2), int(nx/2)]
    print(f"Velocity at center (Solid): {u_center}")
    
    if abs(u_center) < 1e-5:
        print("PASS: Solid velocity is zero.")
    else:
        print("FAIL: Solid velocity is non-zero.")

if __name__ == "__main__":
    test_ibm_sphere_flow()
