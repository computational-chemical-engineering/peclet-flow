import numpy as np

def poly_D_pt(xi): return xi * (1.0 + xi)
def poly_Nnb_pt(xi): return xi * (1.0 - xi)
def poly_Nc_pt(xi): return 2.0 * (xi**2 - 1.0)
def poly_Nbc_pt(xi): return 2.0

def poly_D_avg(xi): return xi * (1.0 + xi) - 1.0/12.0
def poly_Nnb_avg(xi): return xi * (1.0 - xi) + 1.0/12.0
def poly_Nc_avg(xi): return 2.0 * (xi**2 - 1.0) - 1.0/6.0
def poly_Nbc_avg(xi): return 2.0

def apply_ibm_1d(u_c, u_nb, u_bc, xi, mu, dx, scheme='point'):
    if scheme == 'point':
        D = poly_D_pt(xi)
        Nc = poly_Nc_pt(xi)
        Nnb = poly_Nnb_pt(xi)
        Nbc = poly_Nbc_pt(xi)
    else:
        D = poly_D_avg(xi)
        Nc = poly_Nc_avg(xi)
        Nnb = poly_Nnb_avg(xi)
        Nbc = poly_Nbc_avg(xi)
        
    # Standard stencil coeffs (diffusion)
    a_nb = mu / dx**2
    a_c = -2.0 * mu / dx**2
    a_g = mu / dx**2
    
    # Modified stencil (A' u = f')
    # Row scaling factor D
    # a_c' = D*a_c + Nc*a_g
    # a_nb' = D*a_nb + Nnb*a_g
    # f' = D*f - Nbc*u_bc*a_g
    
    a_c_mod = D * a_c + Nc * a_g
    a_nb_mod = D * a_nb + Nnb * a_g
    
    # Inhomogeneous correction (term that moves to RHS)
    # Eq: a'u = Df - Nbc*ubc*ag  => Df = a'u + Nbc*ubc*ag
    rhs_corr = Nbc * u_bc * a_g
    
    return (a_c_mod * u_c + a_nb_mod * u_nb + rhs_corr) / D

def test_exactness():
    mu = 0.01
    dx = 0.1
    thetas = [0.1, 0.3, 0.5, 0.8]
    
    print("Testing 1D IBM Exactness...")
    
    for xi in thetas:
        # 1. Point-Value Quadratic
        # u(x) = (x + xi)^2. Boundary at -xi where u=0.
        u_pt = lambda x: (x + xi)**2
        u_c = u_pt(0)
        u_nb = u_pt(dx)
        u_bc = 0.0
        
        val = apply_ibm_1d(u_c, u_nb, u_bc, xi/dx, mu, dx, 'point')
        expected = 2.0 * mu
        print(f"Point Quadratic xi={xi}: calc={val:.6e}, expected={expected:.6e}")
        assert abs(val - expected) < 1e-12
        
        # 2. Cell-Average Quadratic
        # u_avg = 1/dx * integral_{-dx/2}^{dx/2} (x+xi)^2 dx = xi^2 + dx^2/12
        u_avg = lambda x: (x + xi)**2 + dx**2/12.0
        u_c_avg = u_avg(0)
        u_nb_avg = u_avg(dx)
        u_bc = 0.0 # Point value at boundary
        
        val_avg = apply_ibm_1d(u_c_avg, u_nb_avg, u_bc, xi/dx, mu, dx, 'avg')
        print(f"Cell-Avg Quadratic xi={xi}: calc={val_avg:.6e}, expected={expected:.6e}")
        assert abs(val_avg - expected) < 1e-12

    # 3. Inhomogeneous BC
    xi = 0.3
    u_wall = 5.0
    u_pt_inhom = lambda x: (x + xi)**2 + u_wall
    u_c = u_pt_inhom(0)
    u_nb = u_pt_inhom(dx)
    u_bc = u_wall
    val = apply_ibm_1d(u_c, u_nb, u_bc, xi/dx, mu, dx, 'point')
    print(f"Inhomogeneous (u_bc={u_wall}): calc={val:.6e}, expected={expected:.6e}")
    assert abs(val - expected) < 1e-12

if __name__ == "__main__":
    test_exactness()
