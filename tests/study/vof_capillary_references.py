#!/usr/bin/env python3
"""Exact VISCOUS references for the two capillary-oscillation benchmarks of the VoF campaign
(`tests/study/vof_surface_tension.py` gates `wave` and `lamb`), replacing the inviscid formulas
those gates were first compared against. Derived 2026-09-02 (VOF_NEXT_SESSION.md, Item 2).

1. Planar capillary wave on the interface between two semi-infinite fluids of EQUAL density rho
   and kinematic viscosity nu (the `wave` gate's configuration). Normal modes ~ exp(s t):

        s^2 + w0^2 (1 - k / sqrt(k^2 + s/nu)) = 0,     w0^2 = sigma k^3 / (2 rho),   Re sqrt > 0

   (even vertical-velocity solution A e^{-k|z|} + B e^{-m|z|}, m^2 = k^2 + s/nu; continuity of
   w, w', w''; normal-stress jump mu [w'''] = -sigma k^4 w(0)/s). First order in
   eps = k sqrt(nu/w0):  s = i w0 - (1 + i) w0 eps/(2 sqrt 2), i.e. the frequency deficit equals
   the damping rate over w0. NOTE the damping is the O(sqrt(nu)) interfacial boundary-layer
   rate, 3-4x larger than the free-surface 2 nu k^2 at the gate's parameters. Walls at +-H
   enter only through w0^2 -> sigma k^3/(2 rho coth(kH)) (0.4 % at kH = pi).
   Measured (WO-P, nvidia-cuda): omega agrees to -0.03 / -0.23 / +0.52 % at
   (32 cells/lambda, nu 0.005) / (64, 0.005) / (32, 0.02).

2. Mode-n shape oscillation of a drop in a host of EQUAL rho and nu (the `lamb` gate). Unsteady
   Stokes normal modes: potential part (r^n inside, r^-(n+1) outside) + poloidal vortical part
   (j_n(q r) inside, h_n^(1)(q r) outside, q^2 = -s/nu, Im q > 0), matched at r = R by
   continuity of u_r, u_theta, tau_rtheta and the normal-stress jump sigma (n-1)(n+2) eta/R^2
   with s eta = u_r. 4x4 determinant, root near Lamb's i w0. First order:
   d omega/w0 = -0.885 sqrt(nu/(w0 R^2)) for n = 2.
   Measured: the simulation captures only part of this shift (its damping is 25-43 % below
   -Re s) because the interfacial boundary layer sqrt(nu/w0) is 0.1-0.45 cells at R = 8; and
   a further ~4 % INVISCID, resolution-independent deficit remains (Item 2 addendum).

Usage:  python tests/study/vof_capillary_references.py            # prints both tables
        from vof_capillary_references import wave_mode, drop_mode  # in a page or a gate
Requires mpmath for the drop (pip install mpmath); the wave needs only numpy/scipy.
"""
import math

import numpy as np


# --------------------------------------------------------------------------- planar wave
def wave_mode(k, nu, sigma=1.0, rho=1.0, H=None):
    """Complex growth rate s (Re s < 0 decay, Im s frequency) of the two-fluid capillary wave."""
    from scipy.optimize import fsolve

    w0sq = sigma * k ** 3 / (2 * rho)
    if H is not None:
        w0sq *= math.tanh(k * H)
    w0 = math.sqrt(w0sq)

    def f(v):
        s = complex(v[0], v[1])
        m = np.sqrt(k * k + s / nu)
        if m.real < 0:
            m = -m
        r = s * s + w0sq * (1.0 - k / m)
        return [r.real, r.imag]

    eps = k * math.sqrt(nu / w0)
    v = fsolve(f, [-w0 * eps / (2 * math.sqrt(2)), w0 * (1 - eps / (2 * math.sqrt(2)))], xtol=1e-12)
    return complex(v[0], v[1]), w0


# --------------------------------------------------------------------------- drop modes
def drop_mode(n, R, nu, sigma=1.0, rho=1.0, dps=50):
    """Complex growth rate s of mode n of a drop in an identical fluid; returns (s, w0_Lamb)."""
    import mpmath as mp

    mp.mp.dps = dps
    mu = rho * nu

    def jn(z):
        return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)

    def hn(z):  # spherical Hankel of the first kind, via mpmath's hankel1 (no j + i y cancellation)
        return mp.sqrt(mp.pi / (2 * z)) * mp.hankel1(n + mp.mpf(1) / 2, z)

    def det(s):
        q = mp.sqrt(-s / nu)
        if mp.im(q) < 0:
            q = -q  # the outer vortical solution must decay
        Rm = mp.mpf(R)
        jR, hR = jn(q * Rm), hn(q * Rm)  # column scaling (root-neutral): keeps the Hankel
        basis = [(lambda r: (r / Rm) ** n, lambda r: mp.mpf(0), 1),  # column from underflowing
                 (lambda r: mp.mpf(0), lambda r: jn(q * r) / jR, 1),
                 (lambda r: (Rm / r) ** (n + 1), lambda r: mp.mpf(0), -1),
                 (lambda r: mp.mpf(0), lambda r: hn(q * r) / hR, -1)]
        M = mp.matrix(4, 4)
        for j, (phi, f, sg) in enumerate(basis):
            U = lambda x, phi=phi, f=f: mp.diff(phi, x) + n * (n + 1) * f(x) / x
            V = lambda x, phi=phi, f=f: (phi(x) + mp.diff(lambda y: y * f(y), x)) / x
            T = mu * (Rm * mp.diff(lambda y: V(y) / y, Rm) + U(Rm) / Rm)
            N = rho * s * phi(Rm) + 2 * mu * mp.diff(U, Rm)
            M[0, j] = sg * U(Rm)
            M[1, j] = sg * V(Rm)
            M[2, j] = sg * T
            M[3, j] = -sg * N - (sigma * (n - 1) * (n + 2) / (s * Rm ** 2)) * U(Rm) * (1 if sg == 1 else 0)
        return mp.det(M)

    w0 = mp.sqrt(n * (n - 1) * (n + 1) * (n + 2) * sigma / (R ** 3 * (2 * n + 1) * rho))
    eps = mp.sqrt(nu / (w0 * R * R))
    s0 = w0 * mp.mpc(-0.885 * eps, 1 - 0.885 * eps)
    d0 = det(s0)
    s = mp.findroot(lambda s: det(s) / d0, s0, tol=1e-24, maxsteps=200)
    return complex(s), float(w0)


if __name__ == "__main__":
    print("Planar two-fluid capillary wave (equal rho, nu), sigma = rho = 1, cell units")
    print(f"{'nx':>4} {'nu':>7} {'w0':>9} {'Im s':>9} {'dw/w0':>8} {'-Re s':>10} {'2 nu k^2':>10}   WO-P measured omega / gamma")
    for nx, nu, mw, mg in [(32, 0.005, 0.06017, 1.087e-3), (64, 0.005, 0.02130, 4.628e-4), (32, 0.02, 0.05928, 2.746e-3)]:
        k = 2 * math.pi / nx
        s, w0 = wave_mode(k, nu, H=nx / 2.0)
        print(f"{nx:>4} {nu:>7.3f} {w0:>9.5f} {s.imag:>9.5f} {100*(s.imag/w0-1):>7.2f}% {-s.real:>10.3e} {2*nu*k*k:>10.3e}"
              f"   {mw:.5f} ({100*(mw/s.imag-1):+.2f}%) / {mg:.3e} ({100*(mg/(-s.real)-1):+.1f}%)")
    print("\nMode-2 drop in an identical fluid, R = 8, sigma = rho = 1")
    print(f"{'nu':>8} {'eps':>7} {'w0':>9} {'Im s':>9} {'dw/w0':>8} {'-Re s':>10}   32^3 measured omega / gamma")
    meas = {0.02: (0.09010, 3.983e-3), 0.01: (0.09061, 2.300e-3), 0.005: (0.09072, 1.474e-3),
            0.0025: (0.09075, 1.011e-3), 0.00125: (0.09066, 7.108e-4)}
    try:
        for nu in [0.02, 0.01, 0.005, 0.0025, 0.00125]:
            s, w0 = drop_mode(2, 8.0, nu)
            m = meas[nu]
            eps = math.sqrt(nu / (w0 * 64))
            print(f"{nu:>8g} {eps:>7.4f} {w0:>9.5f} {s.imag:>9.5f} {100*(s.imag/w0-1):>7.2f}% {-s.real:>10.3e}"
                  f"   {m[0]:.5f} ({100*(m[0]/s.imag-1):+.2f}%) / {m[1]:.3e} ({100*(m[1]/(-s.real)-1):+.1f}%)")
    except ImportError:
        print("  (mpmath not installed)")
