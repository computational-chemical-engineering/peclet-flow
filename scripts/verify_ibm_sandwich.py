"""Verify the IBM on a sandwiched-channel (two-wall) configuration."""
import numpy as np

# Point-Value Polynomials (Reference)
def poly_D_pt(xi): return xi * (1.0 + xi)
def poly_Nc_pt(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nnb_pt(xi): return xi * (1.0 - xi)
def poly_Nbc_pt(xi): return 2.0

# Sandwich Point-Value Polynomials
def poly_D_sw_pt(xm, xp): return xm * xp
def poly_Nc_sw_pt(xm, xp): return (xm + 1.0) * (xp - 1.0)

# Inhomogeneous Sandwich factors (from Table 2 derivation/paper)
# u_ghost_+ approx w_nb u_nb + ...
# Actually, for sandwich, we solve:
# u_c = ... 
# D * u_c = ...
# D * b_c - N_bc_pp * u_bc_p * a_gp ...
# We need the N_bc terms.
# Paper Table 2:
# N_bc,+,+ : xi_- /(xi_-+xi_+) * (1 + xi_-)
# N_bc,-,+ : xi_+ /(xi_-+xi_+) * (1 - xi_+)
# Symmetry:
# N_bc,-,- : xi_+ /(xi_-+xi_+) * (1 + xi_+) (Swap + and -)
# N_bc,+,- : xi_- /(xi_-+xi_+) * (1 - xi_-)

def poly_Nbc_pp_sw_pt(xm, xp): return xm / (xm + xp) * (1.0 + xm)
def poly_Nbc_mp_sw_pt(xm, xp): return xp / (xm + xp) * (1.0 - xp)
def poly_Nbc_mm_sw_pt(xm, xp): return xp / (xm + xp) * (1.0 + xp)
def poly_Nbc_pm_sw_pt(xm, xp): return xm / (xm + xp) * (1.0 - xm)

def verify_sandwich_inhomogeneous():
    print("Verifying Sandwich Inhomogeneous Terms...")
    
    # 1D Diffusion: -u'' = f
    # Discrete: (-u_E + 2u_C - u_W) / h^2 = f
    # a_c = 2/h^2, a_nb = -1/h^2.
    # Wait, my code uses: a_c = -2, a_nb = 1.
    # L(u) = a_c u_c + a_nb u_nb.
    # So L(u) ~ u'' ~ 2.
    
    # Case: u(x) = (x - x_m)(x_p - x) + u_wall
    # u(-xm) = u_wall
    # u(xp) = u_wall
    # u(0) = -xm*(-xp) + u_wall = xm*xp + u_wall
    # u'' = -2.
    # Standard stencil on u_c, u_E, u_W (if fluid) would give -2.
    
    # Sandwich Operator:
    # a'_c u_c = b'_c
    # b'_c = D * b_c - Sum( N_bc * u_bc * a_ghost )
    # b_c (standard) = L(u)_target = -2 (if u''=-2).
    # a_ghost = 1 (standard neighbor coeff).
    
    # So:
    # a'_c u_c = D * (-2) - Sum(N_bc) * u_wall * 1.
    
    # Let's verify if this holds for u_c = xm*xp + u_wall.
    
    xis = [(0.5, 0.5), (0.2, 0.8), (0.1, 0.4)]
    u_wall = 5.0
    
    for xm, xp in xis:
        D = poly_D_sw_pt(xm, xp)
        Nc = poly_Nc_sw_pt(xm, xp)
        
        Npp = poly_Nbc_pp_sw_pt(xm, xp)
        Nmp = poly_Nbc_mp_sw_pt(xm, xp)
        Nmm = poly_Nbc_mm_sw_pt(xm, xp)
        Npm = poly_Nbc_pm_sw_pt(xm, xp)
        
        # Standard coeffs
        a_c = -2.0
        a_g = 1.0 # a_E and a_W
        b_c = 0.0 # Laplacian of u_wall + quadratic?
        # u(x) = -x^2 + (xp-xm)x + xm*xp + u_wall
        # u'' = -2.
        # Discrete L(u) = -2 (Exact).
        # So A*u = -2.
        # b_c should be -2?
        # The equation is A u = b.
        # If we check consistency, we check if A' u_c = b'.
        # Let's compute A' u_c.
        
        # A'_c = D*a_c + Nc * a_g (Is Nc sum of Nc+ and Nc-?)
        # Paper: A'_c = D*a_c + Nc,- * a_g- + Nc,+ * a_g+
        # Here Nc in code seems to be single value?
        # Table 2 lists "Nc+".
        # By symmetry "Nc-" should be swapping xi.
        # Actually code `compute_ibm_geometry_kernel` calculates:
        # N_c_plus = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]) -> (xm, xp)
        # N_c_minus = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]) -> (xp, xm)
        # Note: In code km is Minus, kp is Plus.
        # `poly_N_c_sandwich(xm, xp)` returns `(xm+1)(xp-1)`.
        # Wait, `xp-1` term corresponds to the Plus boundary?
        # Let's verify Table 2.
        # Nc+ = (xm+1)(xp-1).
        # Nc- = (xp+1)(xm-1) (Swap xm, xp).
        
        Nc_p = poly_Nc_sw_pt(xm, xp)
        Nc_m = poly_Nc_sw_pt(xp, xm)
        
        A_prime_c = D * a_c + Nc_p * a_g + Nc_m * a_g
        
        # b' = D * b_c - Sum terms.
        # If u(x) satisfies the equation, b_c should be the forcing.
        # u'' = -2.
        # Solver: mu * lap(u) = f.
        # Let mu=1. f = -2.
        # b_c = f = -2.
        
        # Terms:
        # u_bc,- * a_g- * (Nmm + Npm) ?
        # u_bc,+ * a_g+ * (Npp + Nmp) ?
        # In our case u_bc,- = u_bc,+ = u_wall. a_g- = a_g+ = 1.
        # SumN = Nmm + Npm + Npp + Nmp.
        
        SumN = Nmm + Npm + Npp + Nmp
        
        B_prime_c = D * (-2.0) - SumN * u_wall * 1.0
        
        # Check: A'_c * u_c == B'_prime_c?
        u_c_val = xm * xp + u_wall
        
        lhs = A_prime_c * u_c_val
        rhs = B_prime_c
        
        diff = lhs - rhs
        print(f"xi=({xm}, {xp}): LHS={lhs:.4f}, RHS={rhs:.4f}, Diff={diff:.4e}")
        
        if abs(diff) > 1e-10:
            print("  FAIL")
        else:
            print("  PASS")

if __name__ == "__main__":
    verify_sandwich_inhomogeneous()
