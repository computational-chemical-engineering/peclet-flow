#!/usr/bin/env python
"""Two-layer plane Couette — validates variable viscosity (Phase 4) INCLUDING the
incremental-rotational pressure scheme (large-dt / steady-Stokes capability) at strong contrast.

Bottom wall fixed, top wall moving at U; two viscosity layers (mu1 lower, mu2 upper, 10x jump).
Analytic steady profile: piecewise linear with interface velocity u_i/U = mu2/(mu1+mu2). The
harmonic face-viscosity mean reproduces it to ~1e-5; the arithmetic mean shows O(1%) error at the
jump (why 'harmonic' matters).

Rotational pressure under variable viscosity: the constant-mu Timmermans term -mu*div(u*) is only
valid for HOMOGENEOUS viscosity (Deteix & Yakoubi, Appl. Math. Lett. 79 (2018) 111-117; arXiv:
1902.05643 for the full shear-rate-projection). The default here ('min') uses the constant
coefficient chi*mu_min - stable at any contrast by domination, exact fallback to the constant-mu
scheme for uniform mu, and it KEEPS the incremental predictor: dt=100 converges in ~200 steps below.
'full' (pointwise mu(i)) is demonstrated to diverge at 10x contrast - mild-contrast use only.

Run:  PYTHONPATH=<build> python two_layer_couette.py
"""
import numpy as np
import peclet.flow as F


def couette(harmonic, rot=None, chi=1.0, dt=20.0, N=32, steps=4000, tol=1e-9):
    nz = 4
    mu1, mu2, U = 1.0, 0.1, 1.0
    s = F.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(mu1); s.set_dt(dt)
    s.set_domain_bc(2, 1, 0.0, 0.0, 0.0)          # -y fixed wall
    s.set_domain_bc(3, 2, U, 0.0, 0.0)            # +y moving wall
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, nz), 10.0)))
    y = np.arange(N)
    muy = np.where(y < N // 2, mu1, mu2).astype(np.float64)
    s.add_field("mu")
    s.set_field("mu", np.asfortranarray(np.repeat(muy[None, :, None], N, 0).repeat(nz, 2)))
    s.set_property_mode("variable", harmonic)
    if rot is not None:
        s.set_variable_rotational(rot, chi)
    prev, conv = None, -1
    for it in range(steps):
        s.step()
        if np.isnan(s.get_u()).any():
            return float("nan"), -(it + 1)        # diverged at step it
        if it % 100 == 99:
            u = s.get_u(); um = np.abs(u).max()
            if prev is not None and np.abs(u - prev).max() / (um + 1e-30) < tol:
                conv = it + 1
                break
            prev = u.copy()
    u = s.get_u()[N // 2, :, 0]
    ui = U * mu2 / (mu1 + mu2)
    yc = (y + 0.5) / N
    exact = np.where(yc < 0.5, ui * yc / 0.5, ui + (U - ui) * (yc - 0.5) / 0.5)
    return np.max(np.abs(u - exact)) / U, conv


if __name__ == "__main__":
    e, c = couette(True)                          # default: incremental + rotational('min')
    print(f"harmonic, incr+rot(min), dt=20 : err {e*100:.4f}%  conv@{c}")
    assert c > 0 and e < 0.005

    e2, c2 = couette(True, dt=100.0, steps=3000)  # LARGE dt — the steady-Stokes capability
    print(f"harmonic, incr+rot(min), dt=100: err {e2*100:.4f}%  conv@{c2}")
    assert c2 > 0 and e2 < 0.005

    ea, ca = couette(False)                       # arithmetic mean: O(1%) at the jump
    print(f"arithmetic, incr+rot(min)      : err {ea*100:.3f}%  conv@{ca}  (harmonic matters)")
    assert ca > 0 and ea > e

    ef, cf = couette(True, rot="full", steps=800)  # pointwise rotational at 10x: diverges
    print(f"harmonic, rot('full') 10x      : {'DIVERGED@'+str(-cf) if cf < 0 else 'stable'} "
          f"(documented mild-contrast-only mode)")
    print("PASS")
