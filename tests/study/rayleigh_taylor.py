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
   growth rate is ~0.75x the inviscid sqrt(A g k) (viscous + finite-interface damping).

   TWO records, same physics, reported side by side (VoF rung V2a / WO-J):
     * DIFFUSE C — the pre-VoF baseline: a tanh profile of width 1.5 advected by the Koren TVD
       scalar transport. Amplitude 1.50 -> 19.52 (x13.0), rate 0.75x sqrt(A g k).
     * SHARP C — geometric VoF (PLIC + Weymouth-Yue), adaptive dt from the INTERFACE-LOCAL
       Courant number. Amplitude 1.50 -> 20.20 (x13.5), rate 0.77x sqrt(A g k), colour volume
       drift -6.6e-09. The sharp interface grows marginally FASTER, which is the expected sign:
       a two-cell-wide density ramp damps the mode that a sharp jump does not.

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


def rayleigh_taylor_vof(N=48, NZ=96, g=0.005, mu=0.002, steps=240, cfl=0.2, dt0=1.0):
    """SHARP-interface Rayleigh-Taylor through the geometric VoF chain (rung V2a, WO-J).

    Identical physics to `rayleigh_taylor()` above — same grid, same Atwood number (ratio 3),
    same gravity, same viscosity, same initial interface z_i(x) = NZ/2 + 1.5 cos(2 pi x / N) —
    with ONE difference: the phase field is a sharp geometric VoF colour field advected by
    Weymouth-Yue instead of a tanh profile of width 1.5 advected by Koren TVD. Reporting both is
    the point: the diffuse record is the pre-VoF baseline this rung has to reproduce.

    Two deliberate departures from the diffuse driver, both forced by the method rather than
    chosen:

    * dt is ADAPTIVE. Weymouth-Yue is bounded only for an interface-local Courant number below
      1/(2(N-1)) = 0.25 in 3D, and an RT interface accelerates, so a fixed dt = 1 that is safe at
      t = 0 is not safe at t = 240. dt is therefore chosen from `vof_max_courant()` — the
      INTERFACE-LOCAL Courant number, which is the quantity the bound applies to (a global max
      would throttle on the far-field return flow; V1 measured 2x over-throttling on Zalesak, and
      the WO-J ctest measures 22x on a jet-plus-quiescent-interface scene). Amplitudes are
      therefore reported against PHYSICAL TIME, sampled at the same t = 40, 80, ... 240 as the
      diffuse record.

    * the amplitude comes from the COLUMN SUM of C, not from interpolating a C = 0.5 crossing.
      For a sharp interface the column sum IS the interface height to machine precision
      (z_i = NZ - sum_z C, heavy on top), while a 0.5-crossing interpolation of a two-cell-wide
      profile carries an O(h) bias that the diffuse case cannot avoid and this one need not.
    """
    s = F.Solver(N, 4, NZ)
    s.set_rho(1.0); s.set_mu(mu); s.set_dt(dt0)
    s.set_advection(True)
    s.set_domain_bc(4, 1, 0, 0, 0); s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, 4, NZ), 10.0)))
    s.enable_vof()
    x, z = np.arange(N), np.arange(NZ)
    zi = NZ / 2 + 1.5 * np.cos(2 * np.pi * x / N)
    # exact cell fractions of {z > z_i(x)} (heavy on top), i.e. a SHARP interface
    c0 = np.clip((z[None, :] + 1.0) - zi[:, None], 0.0, 1.0)
    s.set_vof(np.asfortranarray(np.repeat(c0[:, None, :], 4, axis=1)))
    s.set_property_model("rho", "linear", "C", [1.0, 2.0])           # rho = 1 + 2C (ratio 3)
    s.set_property_model("force_z", "linear", "rho", [0.0, -g])

    def amp():
        h = s.get_vof()[:, 1, :].sum(axis=1)      # cells above the interface, per x-column
        zc = NZ - h
        return 0.5 * (zc.max() - zc.min())

    v0 = s.get_vof().sum()
    t, dt, sample, hist, tt = 0.0, dt0, 40.0, [amp()], [0.0]
    while t < steps - 1e-9:
        c = s.vof_max_courant()                    # interface-local, with the CURRENT dt
        if c > 0:                                  # target `cfl`, never grow dt by more than 2x
            dt = min(2.0 * dt, dt * cfl / c, dt0)
        dt = min(dt, sample - t)
        s.set_dt(dt)
        s.step()
        t += dt
        if t >= sample - 1e-9:
            hist.append(amp()); tt.append(t); sample += 40.0
    return hist, tt, (s.get_vof().sum() - v0) / v0


if __name__ == "__main__":
    for ratio in (3.0, 1000.0):
        m, perr = hydrostatic(ratio)
        print(f"hydrostatic ratio {ratio:g}: steady max|u| {m:.2e}  P-grad rel-err {perr:.2e}")
        assert m < 1e-12 and perr < 1e-11
    hist = rayleigh_taylor()
    growth = hist[-1] / hist[0]
    print(f"Rayleigh-Taylor  DIFFUSE C amplitude: {' -> '.join(f'{h:.2f}' for h in hist)}  "
          f"(x{growth:.1f})")
    assert growth > 3.0 and all(hist[i+1] >= hist[i] * 0.98 for i in range(len(hist) - 1))

    vh, vt, vdrift = rayleigh_taylor_vof()
    vgrowth = vh[-1] / vh[0]
    print(f"Rayleigh-Taylor  SHARP  C amplitude: {' -> '.join(f'{h:.2f}' for h in vh)}  "
          f"(x{vgrowth:.1f})   colour dV/V {vdrift:+.2e}")
    # Linear theory: the inviscid interfacial growth rate n = sqrt(A g k), with A = (r-1)/(r+1)
    # = 0.5 at ratio 3 and k = 2 pi / N. The rate is fitted over t in [40, 160] -- NOT over the
    # first window. The linear solution from REST is a cosh, not a pure exponential, so the first
    # window measures the establishment of the mode rather than its growth rate (measured: 0.30x
    # there against 0.75x once established); by t = 160 the amplitude is ~10 cells = 0.2 lambda
    # and the nonlinear (bubble/spike) stage is taking over. Both records use the same window.
    A, k = 0.5, 2 * np.pi / 48
    n_lin = np.sqrt(A * 0.005 * k)

    def fit(a, t, t0=40.0, t1=160.0):
        m = [i for i, ti in enumerate(t) if t0 - 1e-9 <= ti <= t1 + 1e-9]
        return np.polyfit([t[i] for i in m], [np.log(a[i]) for i in m], 1)[0]

    td = [40.0 * i for i in range(len(hist))]
    n_dif, n_vof = fit(hist, td), fit(vh, vt)
    print(f"growth rate over t in [40,160]: linear theory sqrt(A g k) = {n_lin:.5f}   "
          f"diffuse {n_dif:.5f} ({n_dif/n_lin:.2f}x)   sharp {n_vof:.5f} ({n_vof/n_lin:.2f}x)")
    assert vgrowth > 3.0 and all(vh[i+1] >= vh[i] * 0.98 for i in range(len(vh) - 1))
    # The colour-volume floor here is the PROJECTION's divergence residual, not the advection
    # (WO-E finding 2): most of it is deposited by the pressure driver's first solve on a fresh
    # field, which leaves a transient velocity the advector faithfully transports once.
    assert abs(vdrift) < 1e-7
    print("PASS")
