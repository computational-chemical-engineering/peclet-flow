#!/usr/bin/env python
"""Variable-density validation (Phase 5): hydrostatic acid tests + Rayleigh-Taylor demonstrator.

1. Hydrostatic balance (the acid test): a stratified two-layer fluid at rest under gravity must
   STAY at rest, with the discrete pressure gradient exactly rho_face*g. This detects any
   inconsistency between the momentum face density, the body-force face value, and the projection
   face coefficient. Inviscid: max steady velocity ~1e-16, P-gradient error ~1e-16 at density
   ratios 3 AND 1000. (The C++ ctest `vardensity_projection` runs the same case.)

2. Rayleigh-Taylor: heavy over light (ratio 3, Atwood 0.5) through the FULL two-phase chain — a
   TRANSPORTED phase fraction c drives rho via a linear-mixture closure (auto-enabling the
   variable-density path), gravity is a closure force_z = -g*rho, momentum + projection carry the
   variable density. The interface amplitude grows ~exponentially then nonlinearly; the measured
   early growth rate is ~0.74x the inviscid sqrt(A g k) (viscous + finite-interface damping).

Run:  PYTHONPATH=<build> python rayleigh_taylor.py
"""
import numpy as np
import peclet.flow as F


def hydrostatic(ratio, mu=0.0, steps=100, g=0.1, dt=1.0, N=8, NZ=24):
    s = F.Solver(N, N, NZ)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_domain_bc(4, 1, 0, 0, 0); s.set_domain_bc(5, 1, 0, 0, 0)   # walls +-z
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, NZ), 10.0)))
    z = np.arange(NZ)
    rz = np.where(z < NZ // 2, ratio, 1.0).astype(np.float64)        # heavy below (stable)
    s.add_field("rho")
    s.set_field("rho", np.asfortranarray(np.broadcast_to(rz[None, None, :], (N, N, NZ)).copy()))
    s.set_density_mode("variable")                                    # Chebyshev pressure driver
    s.set_property_model("force_z", "linear", "rho", [0.0, -g])
    m = None
    for _ in range(steps):
        s.step()
        m = max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(), np.abs(s.get_w()).max())
    p = s.get_p()
    dp = p[N//2, N//2, 1:] - p[N//2, N//2, :-1]
    rf = 0.5 * (rz[1:] + rz[:-1])
    perr = np.max(np.abs(dp + g * rf)) / (g * ratio)
    return m, perr


def rayleigh_taylor(N=48, NZ=96, g=0.005, mu=0.002, dt=1.0, steps=240):
    s = F.Solver(N, 4, NZ)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt)
    s.set_advection(True)
    s.set_domain_bc(4, 1, 0, 0, 0); s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, 4, NZ), 10.0)))
    s.add_scalar("c", diffusivity=0.0, scheme=1, iters=1)
    s.set_scalar_bc("c", 4, 1, 0.0); s.set_scalar_bc("c", 5, 1, 0.0)
    s.set_property_model("rho", "linear", "c", [1.0, 2.0])            # rho = 1 + 2c (ratio 3)
    s.set_property_model("force_z", "linear", "rho", [0.0, -g])
    x, z = np.arange(N), np.arange(NZ)
    zi = NZ / 2 + 1.5 * np.cos(2 * np.pi * x / N)
    c0 = np.zeros((N, 4, NZ))
    for i in range(N):
        c0[i, :, :] = 0.5 * (1.0 + np.tanh((z[None, :] - zi[i]) / 1.5))  # heavy on top
    s.set_field("c", np.asfortranarray(c0))

    def amp():
        c = s.get_field("c")[:, 1, :]
        zc = np.array([np.interp(0.5, c[i, :], z) for i in range(N)])
        return 0.5 * (zc.max() - zc.min())

    hist = [amp()]
    for it in range(steps):
        s.step()
        if it % 40 == 39:
            hist.append(amp())
    return hist


if __name__ == "__main__":
    for ratio in (3.0, 1000.0):
        m, perr = hydrostatic(ratio)
        print(f"hydrostatic ratio {ratio:g}: steady max|u| {m:.2e}  P-grad rel-err {perr:.2e}")
        assert m < 1e-12 and perr < 1e-11
    hist = rayleigh_taylor()
    growth = hist[-1] / hist[0]
    print(f"Rayleigh-Taylor amplitude: {' -> '.join(f'{h:.2f}' for h in hist)}  (x{growth:.1f})")
    assert growth > 3.0 and all(hist[i+1] >= hist[i] * 0.98 for i in range(len(hist) - 1))
    print("PASS")
