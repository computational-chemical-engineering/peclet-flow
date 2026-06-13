"""Unit-verify the cut-cell IBM stencil mathematics."""
import numpy as np

def poly_D_pt(xi): return xi * (1.0 + xi)
def poly_Nnb_pt(xi): return xi * (1.0 - xi)
def poly_Nc_pt(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nbc_pt(xi): return 2.0

def poly_D_avg(xi): return xi * (1.0 + xi) - 1.0/12.0
def poly_Nnb_avg(xi): return xi * (1.0 - xi) + 1.0/12.0
def poly_Nc_avg(xi): return 2.0 * (xi**2 - 1.0) - 1.0/6.0
def poly_Nbc_avg(xi): return 2.0

def test_point_value_exactness():
    print("Testing Point-Value IBM Polynomials for Quadratic Exactness...")
    thetas = [0.1, 0.3, 0.5, 0.8]
    for xi in thetas:
        D = poly_D_pt(xi)
        Nc = poly_Nc_pt(xi)
        Nnb = poly_Nnb_pt(xi)
        Nbc = poly_Nbc_pt(xi)
        
        # Test 1: Constant field u=1
        # u_ghost = (Nnb*1 + Nc*1 + Nbc*1) / D
        u_ghost = (Nnb + Nc + Nbc) / D
        if abs(u_ghost - 1.0) > 1e-12:
            print(f"  FAIL Constant (xi={xi}): u_ghost={u_ghost}")
            
        # Test 2: Quadratic u = (x + xi)^2
        # Boundary at -xi where u=0.
        # u_c (x=0) = xi^2
        # u_nb (x=1) = (1+xi)^2
        # u_ghost (x=-1) = (-1+xi)^2
        u_c = xi**2
        u_nb = (1.0 + xi)**2
        u_ghost_target = (-1.0 + xi)**2
        
        u_ghost_calc = (Nnb * u_nb + Nc * u_c + Nbc * 0.0) / D
        if abs(u_ghost_calc - u_ghost_target) > 1e-12:
            print(f"  FAIL Quadratic (xi={xi}): calc={u_ghost_calc}, target={u_ghost_target}")
        else:
            print(f"  PASS xi={xi}")

def test_cell_average_exactness():
    print("\nTesting Cell-Average IBM Polynomials for Quadratic Exactness...")
    # u_avg = integral_{-0.5}^{0.5} (Ax^2 + Bx + C) dx = A/12 + C
    # u_point(x) = Ax^2 + Bx + C
    thetas = [0.1, 0.3, 0.5, 0.8]
    for xi in thetas:
        D = poly_D_avg(xi)
        Nc = poly_Nc_avg(xi)
        Nnb = poly_Nnb_avg(xi)
        Nbc = poly_Nbc_avg(xi)
        
        # Boundary at -xi where u_point(-xi) = 0.
        # Let u_point(x) = (x + xi)^2 = x^2 + 2*xi*x + xi^2
        # u_c = integral_{-0.5}^{0.5} (x + xi)^2 dx = 1/12 + xi^2
        # u_nb = integral_{0.5}^{1.5} (x + xi)^2 dx = integral_{-0.5}^{0.5} (x + 1 + xi)^2 dx = 1/12 + (1+xi)^2
        # u_ghost_target = integral_{-1.5}^{-0.5} (x + xi)^2 dx = integral_{-0.5}^{0.5} (x - 1 + xi)^2 dx = 1/12 + (-1+xi)^2
        
        u_c = 1.0/12.0 + xi**2
        u_nb = 1.0/12.0 + (1.0 + xi)**2
        u_ghost_target = 1.0/12.0 + (xi - 1.0)**2
        
        u_ghost_calc = (Nnb * u_nb + Nc * u_c + Nbc * 0.0) / D
        if abs(u_ghost_calc - u_ghost_target) > 1e-12:
            print(f"  FAIL Quadratic (xi={xi}): calc={u_ghost_calc}, target={u_ghost_target}")
        else:
            print(f"  PASS xi={xi}")

def poly_D_sw_pt(xm, xp): return xm * xp
def poly_Nc_sw_pt(xm, xp): return (xm + 1.0) * (xp - 1.0)
def poly_Nbc_pp_sw_pt(xm, xp): return xm / (xm + xp) * (1.0 + xm)
def poly_Nbc_mp_sw_pt(xm, xp): return xp / (xm + xp) * (1.0 - xp)

def poly_D_sw_avg(xm, xp): return xm * xp - 1.0/12.0
def poly_Nc_sw_avg(xm, xp): return (xm + 1.0) * (xp - 1.0) - 1.0/12.0
def poly_Nbc_pp_sw_avg(xm, xp): return xm / (xm + xp) * (1.0 + xm) - 1.0/12.0
def poly_Nbc_mp_sw_avg(xm, xp): return xp / (xm + xp) * (1.0 - xp) + 1.0/12.0

def test_sandwich_point_value():
    print("\nTesting Sandwiched Point-Value Polynomials...")
    xis = [(0.5, 0.5), (0.2, 0.8), (0.1, 0.4)]
    for xm, xp in xis:
        D = poly_D_sw_pt(xm, xp)
        Nc = poly_Nc_sw_pt(xm, xp)
        Nbc_pp = poly_Nbc_pp_sw_pt(xm, xp)
        Nbc_mp = poly_Nbc_mp_sw_pt(xm, xp)
        
        # Test: Quadratic u(x) = (x + xm)(xp - x)
        # u(-xm) = 0. u(xp) = 0.
        # u(0) = xm * xp
        # u(1) = (1 + xm)(xp - 1)
        u_c = xm * xp
        u_ghost_target = (1.0 + xm) * (xp - 1.0)
        
        # u_ghost = (Nc * u_c + ...) / D
        u_ghost_calc = (Nc * u_c) / D
        if abs(u_ghost_calc - u_ghost_target) > 1e-12:
            print(f"  FAIL (xm={xm}, xp={xp}): calc={u_ghost_calc}, target={u_ghost_target}")
        else:
            print(f"  PASS xm={xm}, xp={xp}")

def test_sandwich_cell_average():
    print("\nTesting Sandwiched Cell-Average Polynomials...")
    xis = [(0.5, 0.5), (0.2, 0.8), (0.1, 0.4)]
    for xm, xp in xis:
        D = poly_D_sw_avg(xm, xp)
        Nc = poly_Nc_sw_avg(xm, xp)

        # Quadratic with zero wall values at x=-xm and x=xp:
        # u(x) = (x + xm)(xp - x) = -x^2 + (xp-xm)x + xm*xp
        # Cell average over a unit cell centered at x0 is u(x0) - 1/12.
        u_c = xm * xp - 1.0/12.0
        u_ghost_target = (1.0 + xm) * (xp - 1.0) - 1.0/12.0

        u_ghost_calc = (Nc * u_c) / D
        if abs(u_ghost_calc - u_ghost_target) > 1e-12:
            print(f"  FAIL (xm={xm}, xp={xp}): calc={u_ghost_calc}, target={u_ghost_target}")
        else:
            print(f"  PASS xm={xm}, xp={xp}")

if __name__ == "__main__":
    test_point_value_exactness()
    test_cell_average_exactness()
    test_sandwich_point_value()
    test_sandwich_cell_average()
