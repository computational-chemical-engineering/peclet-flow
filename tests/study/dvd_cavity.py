#!/usr/bin/env python
"""Differentially heated square cavity (de Vahl Davis 1983) — validates the Boussinesq
field->momentum coupling (property closures + per-cell body force + scalar transport).

Left wall hot T=1, right wall cold T=0, top/bottom adiabatic, no-slip everywhere, gravity in -y
with a Boussinesq buoyancy body force. Benchmark average Nusselt number on the hot wall:

    Ra    Nu_avg   u_max*   v_max*
    1e3   1.118    3.649    3.697
    1e4   2.243   16.178   19.617
    1e5   4.519   34.73    68.59

(* velocity extrema normalised by alpha/L.) Run:  PYTHONPATH=<build> python dvd_cavity.py
"""
import numpy as np, time
import peclet.flow as F

REF = {1e3: 2.243}  # (Nu ref filled per-Ra below)
NU_REF = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519}


def cavity(N, Ra, Pr=0.71, mu=0.05, dt=8.0, steps=3000, tol=1e-5, verbose=False):
    nz = 4
    s = F.Solver(N, N, nz)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_implicit_advection(True); s.set_outer_iterations(2)
    for f in (0, 1, 2, 3):
        s.set_domain_bc(f, 1, 0.0, 0.0, 0.0)              # no-slip walls (z periodic)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, nz), 10.0)))
    alpha = mu / Pr
    s.add_scalar("T", diffusivity=alpha, scheme=1, iters=50)
    s.set_scalar_bc("T", 0, 2, 1.0); s.set_scalar_bc("T", 1, 2, 0.0)   # hot / cold walls
    s.set_scalar_bc("T", 2, 1, 0.0); s.set_scalar_bc("T", 3, 1, 0.0)   # adiabatic top/bottom
    coeff = Ra * mu * mu / (Pr * N**3)                    # rho0*g*beta so that Ra hits target
    s.set_property_model("force_y", "boussinesq", "T", [1.0, coeff, 1.0, 0.5])
    x = np.arange(N)
    T0 = np.repeat((1.0 - (x + 0.5) / N)[:, None, None], N, 1).repeat(nz, 2)
    s.set_field("T", np.asfortranarray(T0.astype(np.float64)))
    t0 = time.time(); prev = None
    for it in range(steps):
        s.step()
        if it % 100 == 99:
            u = s.get_u(); um = np.abs(u).max()
            if prev is not None and np.abs(u - prev).max() / (um + 1e-30) < tol:
                break
            prev = u.copy()
    T = s.get_field("T")
    Nu = 2.0 * N * (1.0 - T[0, :, :].mean())              # avg -dT/dx * L on the hot wall
    u, v = s.get_u(), s.get_v()
    return dict(Nu=Nu, umax=np.abs(u).max() * N / alpha, vmax=np.abs(v).max() * N / alpha,
                it=it + 1, t=time.time() - t0)


if __name__ == "__main__":
    for N, Ra in [(24, 0.0), (32, 1e4)]:
        r = cavity(N, Ra)
        if Ra == 0.0:
            print(f"conduction (Ra=0)  N={N}: Nu={r['Nu']:.4f} (expect 1.0)")
        else:
            e = abs(r['Nu'] - NU_REF[Ra]) / NU_REF[Ra] * 100
            print(f"Ra={Ra:.0e} N={N}: Nu={r['Nu']:.3f} (ref {NU_REF[Ra]}, err {e:.1f}%)  "
                  f"u*={r['umax']:.1f} v*={r['vmax']:.1f}  [{r['it']} steps, {r['t']:.0f}s]")
