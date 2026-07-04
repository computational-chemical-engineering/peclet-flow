import numpy as np, gc, time
import peclet.flow as F

def allfluid(shape): return np.asfortranarray(np.full(shape, 10.0))

def two_layer_couette(harmonic, mu1=1.0, mu2=0.1, N=32, U=1.0, steps=4000, tol=1e-8):
    # Plane Couette: bottom wall (y=0) fixed, top wall (y=Ly) moves at U in x; periodic x,z.
    # Two viscosity layers: mu1 for lower half, mu2 for upper half. Steady Stokes (no advection).
    nz = 4
    s = F.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(mu1); s.set_dt(20.0)   # large dt -> steady Stokes
    s.set_domain_bc(2, 1, 0.0, 0.0, 0.0)            # -y no-slip wall (fixed)
    s.set_domain_bc(3, 2, U, 0.0, 0.0)              # +y moving wall (tangential Dirichlet)
    s.set_pressure_geometry(allfluid((N, N, nz)))
    # viscosity field: lower half mu1, upper half mu2 (interface at y = N/2 cell face)
    y = np.arange(N)
    muy = np.where(y < N//2, mu1, mu2).astype(np.float64)
    mu = np.repeat(muy[None, :, None], N, 0).repeat(nz, 2)
    s.add_field("mu"); s.set_field("mu", np.asfortranarray(mu))
    s.set_property_mode("variable", harmonic)
    prev = None
    for it in range(steps):
        s.step()
        if it % 200 == 199:
            u = s.get_u(); um = np.abs(u).max()
            if prev is not None and np.abs(u-prev).max()/(um+1e-30) < tol: break
            prev = u.copy()
    u = s.get_u()[N//2, :, 0]   # u(y) at mid-x
    # analytic piecewise-linear: interface velocity u_i/U = mu2/(mu1+mu2)
    ui = U*mu2/(mu1+mu2)
    yc = (y + 0.5)/N
    exact = np.where(yc < 0.5, ui*(yc/0.5), ui + (U-ui)*((yc-0.5)/0.5))
    err = np.max(np.abs(u-exact))/U
    del s; gc.collect()
    return err, u, exact, ui

if __name__ == "__main__":
    eh, u, ex, ui = two_layer_couette(harmonic=True)
    print(f"[couette harmonic] max-err {eh*100:.3f}%  (interface u_i/U={ui:.4f})")
    ea, _, _, _ = two_layer_couette(harmonic=False)
    print(f"[couette arithmetic] max-err {ea*100:.3f}%  (should be worse -> harmonic matters)")
    assert eh < 0.005, f"harmonic two-layer Couette err {eh*100:.2f}% > 0.5%"
    assert ea > eh, "arithmetic should be less accurate than harmonic at a viscosity jump"
    print("PHASE 4: PASS")
