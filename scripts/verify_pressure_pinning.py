import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def generate_sdf(res_n, L):
    # Simple sphere sdf
    dx = L/res_n
    x = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    y = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    z = np.linspace(0, L, res_n, endpoint=False) + 0.5 * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Sphere at center
    xc, yc, zc = L/2, L/2, L/2
    R = L/4
    dist = R - np.sqrt((X-xc)**2 + (Y-yc)**2 + (Z-zc)**2)
    # Fluid inside sphere (dist > 0), Solid outside (dist < 0)
    
    return dist.ravel(order='F').astype(np.float32)

def test_pinning(method='step'):
    res_n = 32
    L = 1.0
    dx = L/res_n
    
    sdf_values = generate_sdf(res_n, L)
    
    # Find expected pin_idx (max SDF value)
    pin_idx = np.argmax(sdf_values)
    print(f"Expected Pin Index: {pin_idx} (Max SDF: {sdf_values[pin_idx]:.4f})")
    
    sdf_data = pnm_backend.SDFData(
        sdf_values, 
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dx, dx)
    )
    
    solver = pnm_backend.CFDSolver(
        pnm_backend.int3(res_n, res_n, res_n), 
        pnm_backend.float3(dx, dx, dx)
    )
    solver.initialize(sdf_data)
    
    # Set params to ensure non-trivial pressure
    solver.set_body_force(pnm_backend.float3(1.0, 0, 0)) # Force to drive flow
    solver.set_rho(1.0)
    solver.set_mu(0.01)
    
    # Run
    print(f"Running {method}...")
    if method == 'step':
        solver.step(0.01)
    elif method == 'project':
        # Project usually requires an initial velocity field or force?
        # If u=0, project might result in p=0?
        # Need to set u first?
        # Or depend on body force? 
        # project() takes dt.
        # Inside project: 
        #   compute_divergence(u) -> rhs. 
        #   solve phi. 
        #   u -= grad(phi). 
        #   update p from phi.
        # If u is 0, div is 0. phi is 0. p remains p_old (0).
        # So we need some velocity.
        
        # Set a dummy velocity field
        u_init = np.random.rand(res_n*res_n*res_n).astype(np.float32)
        solver.set_u(u_init)
        solver.project(0.01, False)
        
    p_field = np.array(solver.get_p())
    p_pin = p_field[pin_idx]
    
    print(f"Pressure at pin_idx: {p_pin:.6e}")
    
    if abs(p_pin) > 1e-5:
        print("FAIL: Pressure not pinned to 0")
        return False
    else:
        print("PASS: Pressure pinned to 0")
        return True

if __name__ == "__main__":
    print("Testing 'step' method pinning...")
    pass_step = test_pinning('step')
    
    print("\nTesting 'project' method pinning...")
    pass_project = test_pinning('project')
    
    if pass_step and pass_project:
        print("\nAll Tests Passed")
        sys.exit(0)
    else:
        print("\nSome Tests Failed")
        sys.exit(1)
