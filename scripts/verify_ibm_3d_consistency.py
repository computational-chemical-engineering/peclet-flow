import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def poly_D(xi): return xi * (1.0 + xi)
def poly_Nc(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nnb(xi): return xi * (1.0 - xi)

def apply_stencil(field, stencil):
    res = stencil[0] * field
    res += stencil[1] * np.roll(field, 1, axis=0)
    res += stencil[2] * np.roll(field, -1, axis=0)
    res += stencil[3] * np.roll(field, 1, axis=1)
    res += stencil[4] * np.roll(field, -1, axis=1)
    res += stencil[5] * np.roll(field, 1, axis=2)
    res += stencil[6] * np.roll(field, -1, axis=2)
    return res

def test_ibm_3d_accuracy():
    """
    Test IBM operator on a 3D parabolic field.
    
    The field is u(x,y,z) = (x - x_b)^2.
    It should yield L(u) = 2*mu exactly at boundary cells if consistent.
    """
    res_n = 32
    L = 1.0
    dx = L/res_n
    mu = 0.01
    
    # 1. Setup Solver with a plane at 45 degrees
    # Normal n = (1, 0, 0) for simplest test
    def generate_plane_sdf(n, shift):
        coords = np.linspace(0, L, n, endpoint=False) + 0.5 * dx
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        # Plane at x = shift
        sdf = X - shift
        return sdf.ravel(order='F').astype(np.float32)

    # Shift so boundary is at xi from a cell center
    # Cell centers at 0.5, 1.5, ...
    # Let center be at i=10. x = 10.5 * dx.
    # Boundary at x_b = (10.5 - theta) * dx.
    theta = 0.3
    x_b = (10.5 - theta) * dx
    
    sdf_values = generate_plane_sdf(res_n, x_b)
    
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
    solver.set_mu(mu)
    
    # 2. Extract IBM Stencil
    # component 0 (U)
    stencil_ibm_flat = solver.get_diffusion_stencil(0, True)
    stencil_ibm = [np.array(arr).reshape((res_n, res_n, res_n), order='F') for arr in stencil_ibm_flat]
    
    # 3. Create Analytical Field u = (x - x_b)^2
    coords = np.linspace(0, L, res_n, endpoint=False)
    # U is staggered in X: faces at i*dx
    X_u, Y_u, Z_u = np.meshgrid(coords, coords + 0.5*dx, coords + 0.5*dx, indexing='ij')
    u_ana = (X_u - x_b)**2
    # Ensure u=0 in solid for clean application?
    # Actually, u_ana is analytical everywhere.
    
    # 4. Apply Operator
    Lu_ibm = apply_stencil(u_ana, stencil_ibm)
    
    # Identify the IBM cell just to the right of boundary
    # x_b = (10.5 - theta)*dx. 
    # Center at i=10 is (10.5)*dx. Distance is theta*dx. Correct.
    # U faces are at i*dx.
    # Boundary at x_b = 10.5*dx - 0.3*dx = 10.2*dx.
    # Fluid cell center at 10.5*dx.
    # West neighbor face is at 10*dx. (Solid).
    # East neighbor face is at 11*dx. (Fluid).
    idx_target = 10
    
    # Expected: 2 * mu
    expected = 2.0 * mu
    
    # The Lu_ibm we computed is A' * u.
    # This should equal D * expected.
    # But D varies per cell. We must use the D_rescale factors extracted from solver.
    
    ibm_scaling_flat = solver.get_ibm_scaling(0)
    ibm_scaling = np.array(ibm_scaling_flat).reshape((res_n, res_n, res_n), order='F')
    
    # Check target cell slice
    Lu_target_slice = Lu_ibm[idx_target, :, :]
    D_eff_slice = ibm_scaling[idx_target, :, :]
    
    # Normalize Lu by D_rescale to recover L(u)
    # L(u) should be 2*mu
    Lu_normalized = Lu_ibm / ibm_scaling
    
    val = np.mean(Lu_normalized[idx_target, :, :])
    
    print(f"Theta={theta}: Lu_ibm[target]={np.mean(Lu_target_slice):.6e}, Expected_unscaled={expected:.6e}")
    print(f"Normalized L(u) = {val:.6e}, Expected = {expected:.6e}")
    
    if abs(val - expected) < 1e-8:
        print("PASS: 3D Operator matches 1D prediction exactly.")
    else:
        print("FAIL: 3D Operator discrepancy.")

    # 5. Check multi-directional consistency (Angled Plane)
    print("\nTesting 45 degree plane consistency...")
    # Plane x + y = shift
    # d = (x + y - shift) / sqrt(2)
    sqrt2 = np.sqrt(2.0)
    shift = 0.5
    coords = np.linspace(0, L, res_n, endpoint=False) + 0.5*dx
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    sdf_angled = (X + Y - shift) / sqrt2
    
    solver.initialize(pnm_backend.SDFData(
        sdf_angled.ravel(order='F').astype(np.float32),
        pnm_backend.int3(res_n, res_n, res_n),
        pnm_backend.float3(0,0,0),
        pnm_backend.float3(dx, dx, dx)
    ))
    
    stencil_angled_flat = solver.get_diffusion_stencil(0, True)
    stencil_angled = [np.array(arr).reshape((res_n, res_n, res_n), order='F') for arr in stencil_angled_flat]
    
    # Get scaling for angled case
    scaling_angled_flat = solver.get_ibm_scaling(0)
    scaling_angled = np.array(scaling_angled_flat).reshape((res_n, res_n, res_n), order='F')
    
    # Analytical Field u = d^2 = (x + y - shift)^2 / 2
    # lap(u) = d2u/dx2 + d2u/dy2 = 1/2 * (2) + 1/2 * (2) = 2.
    # mu * lap(u) = 2 * mu.
    coords_u = np.linspace(0, L, res_n, endpoint=False)
    X_u, Y_u, Z_u = np.meshgrid(coords_u, coords_u + 0.5*dx, coords_u + 0.5*dx, indexing='ij')
    u_ana_angled = ((X_u + Y_u - shift)**2) / 2.0
    
    Lu_angled = apply_stencil(u_ana_angled, stencil_angled)
    
    # Compare in fluid cells (normalized)
    fluid_mask = (sdf_angled.reshape((res_n, res_n, res_n), order='F') > 0.05)
    
    Lu_angled_norm = Lu_angled / scaling_angled
    
    expected_viscous = 2.0 * mu
    ratios = Lu_angled_norm[fluid_mask] / expected_viscous
    print(f"Angled Plane: Ratio Mean={np.mean(ratios):.6f}, Std={np.std(ratios):.6f}")
    # In interior fluid, ratios should be 1.0. At boundary, should also be 1.0 (if exact).

if __name__ == "__main__":
    test_ibm_3d_accuracy()
