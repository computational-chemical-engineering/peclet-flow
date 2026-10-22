#!/usr/bin/env python
"""Large dt and steady marches with the full viscous stress (doc/full_viscous_stress.md §6, WO-3, G3).

  couette  G3(i)   two-layer Couette (10x mu(y), arithmetic mean) at dt = 20 and 100: full stress
                   against laplacian, the same error to 1e-8 relative (design Amendment A2).
  rotcore  G3(ii)  the rotating-core problem of G2 (full_stress_gates.py) at mu_in/mu_out = 0.01 and
                   100, D = 1e4, 2000 steps: no growth as design Amendment A3 states it -- with
                   d_n = max|u^n - u^2000| (all components), d_n <= 2 d_0 for every n < 2000 and
                   d_1999 <= 1e-8 d_0 (u^0 = the continuum field the march starts from); the
                   laplacian control is info. And the same steady error as at D = 10 (relative
                   1e-8).  --n N (default 64).
  spheres  G3(iii) steady Stokes through the periodic sphere array (sdflow set-up, N = 32) with a
                   smooth 10x mu(x) = mu0 10^((1 - cos 2 pi x/L)/2): dt = 20 and dt = 100 give the
                   same steady K = mu0 <u>/F to 1e-6; steps to steady (max|du| <= 1e-10 max|u|),
                   full against laplacian (target <= 2x, else Q3).
  ctest            the short ctest `full_stress_large_dt`: rotating core N = 32, ratio 100, D = 1e4,
                   500 steps, full stress: the A3 no-growth test against u^500.

Run: OMP_NUM_THREADS=8 PYTHONPATH=<build> python tests/study/full_stress_large_dt.py [cmd ...]
Exit 0 pass, 1 miss, 77 skipped (no module).
"""
import argparse
import os
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_stress_gates import couette, rotcore, rotcore_exact

F = np.asfortranarray
OK = True


def verdict(name, good, detail):
    global OK
    OK &= bool(good)
    print(f"{name:34s} {'PASS' if good else 'MISS'}  {detail}", flush=True)


def no_growth(d):
    """Amendment A3: d_n = max|u^n - u^N| for n = 0..N-1; d_n <= 2 d_0 for all n, d_{N-1} <= 1e-8 d_0."""
    d = np.asarray(d)
    if len(d) < 2 or not np.isfinite(d).all() or d[0] <= 0.0:
        return False, float("inf"), float("inf")
    peak, last = float(d.max() / d[0]), float(d[-1] / d[0])
    return peak <= 2.0 and last <= 1e-8, peak, last


def initial_field(N, ratio, nz=4, U0=1e-3):
    """u^0: the exact (continuum) face velocity rotcore() sets before the first step."""
    u_theta = rotcore_exact(N, ratio, U0)[4]
    c0 = N / 2.0
    i = np.arange(N, dtype=np.float64)
    Xu, Yu = np.meshgrid(i, i + 0.5, indexing="ij")
    Xv, Yv = np.meshgrid(i + 0.5, i, indexing="ij")
    ru, rv = np.hypot(Xu - c0, Yu - c0), np.hypot(Xv - c0, Yv - c0)
    ue = -u_theta(ru) * (Yu - c0) / np.maximum(ru, 1e-300)
    ve = u_theta(rv) * (Xv - c0) / np.maximum(rv, 1e-300)
    rep3 = lambda a: np.repeat(a[:, :, None], nz, axis=2)
    return np.stack([rep3(ue), rep3(ve), np.zeros((N, N, nz))])


def march_rotcore(N, ratio, D, steps, full=True):
    """Fixed-length march (the G2 set-up, exact initial field). Returns the solver, max|u| after
    every step, and d_n = max|u^n - u^steps| for n = 0..steps-1 (Amendment A3)."""
    *_, s = rotcore(N, ratio, full, D=D, max_steps=1, tol=-1.0)
    state = lambda: np.stack([np.array(s.get_u()), np.array(s.get_v()), np.array(s.get_w())])
    hist = [initial_field(N, ratio), state()]
    for _ in range(steps - 1):
        s.step()
        hist.append(state())
    mx = [float(np.abs(h[:2]).max()) for h in hist[1:]]
    ref = hist[-1]
    d = [float(np.abs(h - ref).max()) for h in hist[:-1]]
    return s, mx, d


def rotcore_error(s, N, ratio, U0=1e-3):
    u_theta = rotcore_exact(N, ratio, U0)[4]
    c0 = N / 2.0
    i = np.arange(N, dtype=np.float64)
    Xu, Yu = np.meshgrid(i, i + 0.5, indexing="ij")
    Xv, Yv = np.meshgrid(i + 0.5, i, indexing="ij")
    ru, rv = np.hypot(Xu - c0, Yu - c0), np.hypot(Xv - c0, Yv - c0)
    ue = -u_theta(ru) * (Yu - c0) / np.maximum(ru, 1e-300)
    ve = u_theta(rv) * (Xv - c0) / np.maximum(rv, 1e-300)
    u, v = np.array(s.get_u())[:, :, 0], np.array(s.get_v())[:, :, 0]
    return max(np.abs(u - ue).max(), np.abs(v - ve).max()) / U0


def cmd_couette():
    # Amendment A2: on the arithmetic mean the transpose term vanishes for layered mu(y), so full and
    # laplacian give the same Couette error to <= 1e-8 relative at dt 20 and 100.
    for dt in (20.0, 100.0):
        e0, c0 = couette(False, dt)
        e1, c1 = couette(True, dt)
        rel = abs(e1 - e0) / e0
        verdict(f"G3(i) couette dt={dt:g}", rel <= 1e-8 and c1 > 0,
                f"laplacian {e0 * 100:.6f}% conv@{c0}; full {e1 * 100:.6f}% conv@{c1}; rel {rel:.2e}")


def cmd_rotcore(N):
    for ratio in (0.01, 100.0):
        e10, n10, *_ = rotcore(N, ratio, True, D=10.0)
        s, mx, d = march_rotcore(N, ratio, 1e4, 2000)
        good_g, peak, last = no_growth(d)
        e4 = rotcore_error(s, N, ratio)
        rel = abs(e4 - e10) / e10
        print(f"  ratio {ratio:g} N {N}: D=10 e {e10:.10e} ({n10} steps); D=1e4 2000 steps e "
              f"{e4:.10e}; max|u| first/50/last {mx[0]:.6e} {mx[49]:.6e} {mx[-1]:.6e}; d_0 "
              f"{d[0]:.3e}, d_1/50/200/1000/1999 {d[1]:.3e} {d[50]:.3e} {d[200]:.3e} {d[1000]:.3e} "
              f"{d[-1]:.3e}", flush=True)
        verdict(f"G3(ii) ratio {ratio:g} no growth (A3)", good_g,
                f"max d_n/d_0 {peak:.4f} (<= 2), d_1999/d_0 {last:.2e} (<= 1e-8)")
        _, _, dl = march_rotcore(N, ratio, 1e4, 2000, full=False)
        gl, pl, ll = no_growth(dl)
        print(f"  (info) laplacian ratio {ratio:g}: A3 {'pass' if gl else 'miss'}, max d_n/d_0 "
              f"{pl:.4f}, d_1999/d_0 {ll:.2e}", flush=True)
        verdict(f"G3(ii) ratio {ratio:g} same steady", rel <= 1e-8,
                f"|e(D=1e4) - e(D=10)| / e = {rel:.2e} (<= 1e-8)")


def spheres(full, dt, N=32, mu0=0.1, Fb=1e-3, max_steps=20000, tol=1e-10):
    R = 0.18 * N
    cs = np.array([0.25 * N, 0.75 * N])
    xs = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    sdf = np.full((N, N, N), 1e30)
    for sx in cs:
        for sy in cs:
            for sz in cs:
                dx = X - sx
                dx -= N * np.round(dx / N)
                dy = Y - sy
                dy -= N * np.round(dy / N)
                dz = Z - sz
                dz -= N * np.round(dz / N)
                sdf = np.minimum(sdf, np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    mu = mu0 * 10.0 ** ((1.0 - np.cos(2 * np.pi * X / N)) / 2.0)
    s = pf.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(mu0)
    s.set_dt(dt)
    s.set_body_force((Fb, 0.0, 0.0))
    s.set_advection(False)
    s.set_pressure_pcg(True, 400, 1e-12)
    s.set_solid(F(sdf), cutcell_pressure=True)
    s.diagnostics.set_property_mode("variable", False)
    s.set_field("mu", F(mu))
    if full:
        s.set_viscous_stress("full")
    prev = np.array(s.get_u())
    for n in range(1, max_steps + 1):
        s.step()
        u = np.array(s.get_u())
        d = np.abs(u - prev).max() / max(np.abs(u).max(), 1e-300)
        prev = u
        if d <= tol:
            break
    return mu0 * float(u.mean()) / Fb, n, d


def cmd_spheres():
    res = {}
    for full in (False, True):
        for dt in (20.0, 100.0):
            res[(full, dt)] = spheres(full, dt)
            K, n, d = res[(full, dt)]
            print(f"  spheres {'full' if full else 'laplacian':9s} dt={dt:5g}: K {K:.10e}  steps "
                  f"{n}  last rel du {d:.1e}", flush=True)
    for full in (False, True):
        k20, k100 = res[(full, 20.0)][0], res[(full, 100.0)][0]
        rel = abs(k20 - k100) / abs(k100)
        verdict(f"G3(iii) {'full' if full else 'laplacian'} K(dt=20)==K(100)", rel <= 1e-6,
                f"rel {rel:.2e} (<= 1e-6)")
    for dt in (20.0, 100.0):
        r = res[(True, dt)][1] / res[(False, dt)][1]
        verdict(f"G3(iii) steps full/laplacian dt={dt:g}", r <= 2.0,
                f"{res[(True, dt)][1]} / {res[(False, dt)][1]} = {r:.2f} (target <= 2, else Q3)")
    dK = abs(res[(True, 100.0)][0] - res[(False, 100.0)][0]) / res[(False, 100.0)][0]
    print(f"  (info) K full vs laplacian at dt=100: rel diff {dK:.3e}")


def cmd_ctest():
    _, mx, d = march_rotcore(32, 100.0, 1e4, 500)
    good, peak, last = no_growth(d)
    verdict("full_stress_large_dt ratio 100 D=1e4", good,
            f"500 steps, max|u| {mx[-1]:.6e}, max d_n/d_0 {peak:.4f} (<= 2), d_499/d_0 {last:.2e} "
            f"(<= 1e-8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="*", default=["couette", "rotcore", "spheres"])
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    for c in a.cmd:
        if c == "rotcore":
            cmd_rotcore(a.n)
        else:
            globals()["cmd_" + c]()
    print("PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
