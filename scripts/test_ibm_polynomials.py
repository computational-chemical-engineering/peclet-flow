import numpy as np

def poly_D(xi):
    return xi * (1.0 + xi)

def poly_N_c(xi):
    return 2.0 * (xi**2 - 1.0)

def poly_N_nb(xi):
    return xi * (1.0 - xi)

def poly_N_bc(xi):
    return 2.0

def verify_1d_operator():
    """
    Verify 1D IBM Operator consistency with second derivative.
    
    Setup:
    - Domain: x in [0, 1]
    - Grid: N cells, dx = 1/N
    - Boundary: Located at x_b = (i_ghost + xi) * dx
    - Function: u(x) = (x - x_b)^2
      u(x_b) = 0
      u'(x) = 2(x - x_b)
      u''(x) = 2
    
    Discrete Operator L(u) should equal d^2u/dx^2 = 2.
    L(u)_c = (A_c * u_c + A_nb * u_nb + B_rhs) / D_scale
    
    Wait, in our code:
    A_c_mod = D*A_c + K*A_g
    A_nb_mod = D*A_nb + M*A_g
    B_mod = D*B + B_fac
    
    Standard Stencil (Diffusion):
    A_c = -2/dx^2, A_nb = 1/dx^2, A_g = 1/dx^2
    B = 0
    
    Modified Stencil:
    A_c' = D*(-2) + N_c*(1)
    A_nb' = D*(1) + N_nb*(1)  <- (Here neighbor is fluid, ghost is on other side)
    So if ghost is West:
      A_E' = D*(1) + N_E*(1) = D + 0 (since N_E applies to ghost dir)
      Wait, N_nb in code applies to the FLUID neighbor in the interpolation stencil?
      Table 1 says "N_{nb,d} applies to the fluid neighbor used in interpolation".
      Interpolation uses: Ghost(at -1), Center(0), Neighbor(1).
      So N_nb applies to Neighbor at +1.
      
      So:
      A_E' = D * A_E + N_nb * A_W (Cross term? No.)
      Let's look at code:
      modify_stencil_ibm_kernel:
      A_c += A_nb * K  (A_nb is the coeff OF the ghost direction in standard stencil)
      
      Example: Ghost is West (index 1 in loop).
      orig_AW is the coefficient for u_W in standard stencil.
      K = N_c * R (R=1 for 1D)
      M = 0
      X = N_nb * R
      
      Update:
      A_C += orig_AW * K = -2*D + 1*N_c
      
      Neighbor updates:
      mod_AW += orig_AW * (D*M - 1) = 1 * (0 - 1) = -1. 
      So A_W becomes 1 + (-1) = 0. Correct.
      
      mod_AE += orig_AE * X = 1 * N_nb = N_nb.
      So A_E becomes D*1 + N_nb.
      
      So Discrete Eq:
      (A_C' * u_C + A_E' * u_E) * (1/dx^2) / D_scale
      = [ (-2D + N_c)*u_C + (D + N_nb)*u_E ] / D
      
      RHS correction:
      B_val for West ghost:
      Code says B = 0 if is_ghost is true? 
      Wait, in `compute_ibm_geometry_kernel`:
      if (is_ghost[kp]) {
          K = N_c * R
          X = N_nb * R
          B = 0
      }
      Where is u_bc handled?
      Ah, `compute_ibm_geometry_kernel` assumes B_val is 0 in the code shown in `src/cfd_solver_ibm.cu`?
      Let's check lines 220-230.
      
      Line 226: ibm_data.B_val[...] = 0.0f;
      
      This looks suspicious! The text says N_bc * u_bc.
      The code seems to set B_val = 0.
      
      Let's verify if u_bc is handled elsewhere or if this is the Bug.
      If u_bc = 0 (Homogeneous Dirichlet), then B=0 is correct.
      Our verification case uses u=0 at boundary, so B=0 is fine for now.
      
    Test for u_bc = 0.
    """
    
    print("Testing 1D Operator Consistency (u_bc = 0)...")
    
    # Arbitrary theta (0 < theta <= 1)
    # distance = theta * dx
    thetas = [0.1, 0.3, 0.5, 0.8, 0.99]
    
    for theta in thetas:
        # Polynomials
        D = poly_D(theta)
        Nc = poly_N_c(theta)
        Nnb = poly_N_nb(theta)
        
        # Standard Stencil Coeffs (scaled by dx^2)
        # u'' = (u_E - 2u_C + u_W) / dx^2
        a_W = 1.0
        a_C = -2.0
        a_E = 1.0
        
        # Modified Stencil (Ghost is West)
        # a_C' = D*a_C + Nc*a_W = -2D + Nc
        # a_W' = 0
        # a_E' = D*a_E + Nnb*a_W = D + Nnb
        
        # Note: In code `mod_AE += orig_AE * (descale * M - 1.0f)`?
        # No, for East (k=0), M=1, X=0 (if fluid).
        # Wait, if West is ghost, East is fluid.
        # For East direction (k=0): is_ghost=False -> K=0, M=1, X=0.
        # mod_AE += orig_AE * (D*1 - 1) -> A_E becomes D*A_E.
        #
        # For West direction (k=1): is_ghost=True -> K=Nc, M=0, X=Nnb.
        # A_C += orig_AW * Nc
        # mod_AW += orig_AW * (D*0 - 1) = -orig_AW -> A_W = 0.
        # mod_AE += orig_AW * X = 1 * Nnb.
        #
        # Total A_E = D*A_E + Nnb*A_W = D + Nnb.
        # Total A_C = D*A_C + Nc*A_W = -2D + Nc.
        
        a_C_mod = -2.0 * D + Nc
        a_E_mod = D + Nnb
        
        # Test Field: u(x) = x^2 (Parabola with vertex at 0)
        # Shift so boundary is at x = -theta.
        # Center at x=0. East at x=1. West at x=-1.
        # Boundary at x_b = -theta.
        # u(x) = (x - x_b)^2 = (x + theta)^2
        # u_C = (0 + theta)^2 = theta^2
        # u_E = (1 + theta)^2 = 1 + 2theta + theta^2
        # u_bc = 0
        
        u_C = theta**2
        u_E = (1.0 + theta)**2
        
        # Apply Operator
        # L(u) = (a_C' * u_C + a_E' * u_E) / D
        
        val = (a_C_mod * u_C + a_E_mod * u_E) / D
        
        # Expected Second Derivative: d2/dx2 (x+theta)^2 = 2
        expected = 2.0
        
        print(f"Theta={theta:.2f}: L(u)={val:.6f}, Expected={expected:.6f}, Diff={val-expected:.6e}")
        
        if abs(val - expected) > 1e-5:
            print("  FAIL: Operator inconsistent")
            return False
            
    print("PASS: 1D Operator Consistent for Quadratic Profile")
    return True

if __name__ == "__main__":
    verify_1d_operator()
