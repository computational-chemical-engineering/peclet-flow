#!/usr/bin/env python
"""Gates of the full viscous stress wiring (doc/full_viscous_stress.md §8, §5.5, §5.6, §13 G2).

Subcommands (default: all but rotcore):
  bit      §8.2  feature on with a UNIFORM mu field (variable mode, walled channel + periodic box):
                 u, v, w, p bitwise equal to the feature off.
  wall     §5.5  walled channel (+-y walls) and an inflow/outflow x channel with a 3-D variable mu,
                 full stress on: the wall-normal / inflow-normal face values stay bitwise equal to
                 the imposed value.
  couette  §8.3  two-layer plane Couette (mu(y), 10x, arithmetic mean): full stress against
                 laplacian at dt = 20 and dt = 100, same error to a relative 1e-6.
  refuse   §5.6  the refusals: collocated setter, harmonic, porous; a bad form name.
  rotcore  G2    the rotating-core manufactured steady Stokes problem (sharp, kinked, curved mu
                 jump): e = max|u - u_ex|/U0 at N = 32, 64, 128 for mu_in/mu_out = 0.1 and 10, full
                 stress (order >= 0.8) against laplacian (>= 5x larger at 128, not converging).
                 Options: --n 32,64,128 --ratios 0.1,10 --D 10 --max-steps N.

Run: OMP_NUM_THREADS=8 PYTHONPATH=<build> python tests/study/full_stress_gates.py [cmd ...]
Prints one verdict line per gate and exits 1 on any MISS.
"""
import argparse
import sys
import time

import numpy as np
import peclet.flow as pf

F = np.asfortranarray
OK = True


def verdict(name, good, detail):
    global OK
    OK &= bool(good)
    print(f"{name:34s} {'PASS' if good else 'MISS'}  {detail}", flush=True)


def fields(s):
    return [np.array(a) for a in (s.get_u(), s.get_v(), s.get_w(), s.get_p())]


# --- bit ------------------------------------------------------------------------------------------
def make_channel(full, mu_field, walls=True, n=(16, 12, 8), steps=12):
    nx, ny, nz = n
    s = pf.Solver(nx, ny, nz)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_dt(0.5)
    if walls:
        s.set_domain_bc("-y", "wall", (0, 0, 0))
        s.set_domain_bc("+y", "wall", (0, 0, 0))
    s.set_pressure_geometry(F(np.full(n, 10.0)))
    s.set_body_force((1e-3, 0.0, 0.0))
    s.add_field("mu")
    s.set_field("mu", F(mu_field))
    s.diagnostics.set_property_mode("variable", False)
    if full:
        s.set_viscous_stress("full")
    rng = np.random.default_rng(3)
    for k in range(3):
        a = F(0.01 * rng.standard_normal(n))
        if walls and k == 1:
            a[:, 0, :] = 0.0
        s.set_velocity(k, a)
    for _ in range(steps):
        s.step()
    return fields(s)


def gate_bit():
    n = (16, 12, 8)
    for walls in (True, False):
        mu = np.full(n, 0.05)
        off = make_channel(False, mu, walls)
        on = make_channel(True, mu, walls)
        same = all(np.array_equal(a, b) for a, b in zip(off, on))
        verdict(f"G-bit uniform mu ({'walls' if walls else 'periodic'})", same,
                "u,v,w,p bitwise on == off" if same else "DIFFERS")
    # the term is really active once mu varies (guards against a skip that is always taken)
    mu = np.full(n, 0.05)
    mu[4:9, 3:8, 2:6] = 0.5
    off, on = make_channel(False, mu), make_channel(True, mu)
    d = max(np.abs(a - b).max() for a, b in zip(off[:3], on[:3]))
    verdict("G-bit control: variable mu acts", d > 0, f"max|du| on vs off = {d:.3e}")


# --- wall -------------------------------------------------------------------------------------------
def gate_wall():
    n = (16, 12, 8)
    x, y, z = np.meshgrid(*(np.arange(k) + 0.5 for k in n), indexing="ij")
    mu = 0.05 * (1.0 + 4.0 * (np.sqrt((x - 6) ** 2 + (y - 4) ** 2 + (z - 4) ** 2) < 3.5))
    v = make_channel(True, mu, walls=True, steps=20)[1]
    lo = v[:, 0, :]
    verdict("wall: v on the -y wall face", np.all(lo == 0.0),
            f"max|v(-y face)| = {np.abs(lo).max():.3e} (imposed 0)")
    # inflow -x (U, 0, 0), outflow +x, walls +-y: u on the inflow face must stay U bitwise
    U = 0.02
    s = pf.Solver(*n)
    s.set_rho(1.0)
    s.set_mu(0.05)
    s.set_dt(0.5)
    s.set_domain_bc("-x", "inflow", (U, 0.0, 0.0))
    s.set_domain_bc("+x", "outflow")
    s.set_domain_bc("-y", "wall", (0, 0, 0))
    s.set_domain_bc("+y", "wall", (0, 0, 0))
    s.set_pressure_geometry(F(np.full(n, 10.0)))
    s.add_field("mu")
    s.set_field("mu", F(mu))
    s.diagnostics.set_property_mode("variable", False)
    s.set_viscous_stress("full")
    for _ in range(20):
        s.step()
    u, vv = np.array(s.get_u()), np.array(s.get_v())
    good = np.all(u[0] == U) and np.all(vv[:, 0, :] == 0.0) and np.isfinite(u).all()
    verdict("wall: inflow u and wall v faces", good,
            f"max|u(-x)-U| = {np.abs(u[0] - U).max():.3e}, max|v(-y)| = "
            f"{np.abs(vv[:, 0, :]).max():.3e}")


# --- couette ----------------------------------------------------------------------------------------
def couette(full, dt, N=32, steps=4000, tol=1e-9):
    nz = 4
    mu1, mu2, U = 1.0, 0.1, 1.0
    s = pf.Solver(N, N, nz)
    s.set_rho(1.0)
    s.set_mu(mu1)
    s.set_dt(dt)
    s.set_domain_bc("-y", "wall", (0.0, 0.0, 0.0))
    s.set_domain_bc("+y", "inflow", (U, 0.0, 0.0))
    s.set_pressure_geometry(F(np.full((N, N, nz), 10.0)))
    y = np.arange(N)
    muy = np.where(y < N // 2, mu1, mu2).astype(np.float64)
    s.add_field("mu")
    s.set_field("mu", F(np.repeat(muy[None, :, None], N, 0).repeat(nz, 2)))
    s.diagnostics.set_property_mode("variable", False)
    if full:
        s.set_viscous_stress("full")
    prev, conv = None, -1
    for it in range(steps):
        s.step()
        if it % 100 == 99:
            u = np.array(s.get_u())
            um = np.abs(u).max()
            if prev is not None and np.abs(u - prev).max() / (um + 1e-30) < tol:
                conv = it + 1
                break
            prev = u.copy()
    u = np.array(s.get_u())[N // 2, :, 0]
    ui = U * mu2 / (mu1 + mu2)
    yc = (y + 0.5) / N
    exact = np.where(yc < 0.5, ui * yc / 0.5, ui + (U - ui) * (yc - 0.5) / 0.5)
    return np.max(np.abs(u - exact)) / U, conv


def gate_couette():
    for dt in (20.0, 100.0):
        e0, c0 = couette(False, dt)
        e1, c1 = couette(True, dt)
        rel = abs(e1 - e0) / e0
        verdict(f"couette arithmetic dt={dt:g}", rel <= 1e-6 and c1 > 0,
                f"laplacian err {e0 * 100:.6f}% conv@{c0}; full err {e1 * 100:.6f}% conv@{c1}; "
                f"rel diff {rel:.2e}")


# --- refuse -----------------------------------------------------------------------------------------
def expect_raise(fn):
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        return str(e).splitlines()[0][:90]
    return None


def gate_refuse():
    msgs = {}
    msgs["collocated setter"] = expect_raise(
        lambda: pf.SolverColocated(8, 8, 8).set_viscous_stress("full"))
    msgs["bad form"] = expect_raise(lambda: pf.Solver(8, 8, 8).set_viscous_stress("fulll"))

    def var_solver(harmonic=False, porous=False):
        s = pf.Solver(8, 8, 8)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_dt(0.5)
        s.set_pressure_geometry(F(np.full((8, 8, 8), 10.0)))
        s.add_field("mu")
        mu = np.full((8, 8, 8), 0.1)
        mu[2:5, 2:5, 2:5] = 1.0
        s.set_field("mu", F(mu))
        s.diagnostics.set_property_mode("variable", harmonic)
        if porous:
            s.set_porous_continuity(True)
        s.set_viscous_stress("full")
        return s

    msgs["harmonic at step"] = expect_raise(lambda: var_solver(harmonic=True).step())
    msgs["porous at step"] = expect_raise(lambda: var_solver(porous=True).step())
    ok_laplacian = expect_raise(lambda: pf.SolverColocated(8, 8, 8).set_viscous_stress("laplacian"))
    for k, m in msgs.items():
        verdict(f"refuse: {k}", m is not None, m or "did NOT raise")
    verdict("refuse: collocated 'laplacian' accepted", ok_laplacian is None, ok_laplacian or "ok")


# --- rotcore (G2) -----------------------------------------------------------------------------------
def quintic(R, R2, U0, dH):
    # H(R)=U0, H'(R)=dH, H''(R)=0, H(R2)=H'(R2)=H''(R2)=0
    def row(r, d):
        c = np.zeros(6)
        for k in range(6):
            if k >= d:
                f = 1.0
                for j in range(d):
                    f *= (k - j)
                c[k] = f * r ** (k - d)
        return c

    A = np.array([row(R, 0), row(R, 1), row(R, 2), row(R2, 0), row(R2, 1), row(R2, 2)])
    b = np.array([U0, dH, 0.0, 0.0, 0.0, 0.0])
    return np.linalg.solve(A, b)


def rotcore_exact(N, ratio, U0):
    L = float(N)
    R, R2 = 0.25 * L, 0.45 * L
    mu_out, mu_in = 1.0, ratio
    a = quintic(R, R2, U0, (U0 / R) * (1.0 + 2.0 * mu_in / mu_out))

    def H(r, d=0):
        out = np.zeros_like(r)
        for k in range(d, 6):
            f = 1.0
            for j in range(d):
                f *= (k - j)
            out += a[k] * f * r ** (k - d)
        return out

    def u_theta(r):
        return np.where(r <= R, U0 * (r / R) ** 3, np.where(r < R2, H(r), 0.0))

    def f_theta(r):  # -mu (u'' + u'/r - u/r^2) per region
        inner = -mu_in * 8.0 * U0 * r / R ** 3
        rr = np.maximum(r, 1e-300)
        outer = -mu_out * (H(r, 2) + H(r, 1) / rr - H(r) / rr ** 2)
        return np.where(r <= R, inner, np.where(r < R2, outer, 0.0))

    return R, R2, mu_in, mu_out, u_theta, f_theta


def rotcore(N, ratio, full, D=10.0, U0=1e-3, max_steps=200000, tol=1e-12, nz=4, init=None):
    R, _R2, mu_in, mu_out, u_theta, f_theta = rotcore_exact(N, ratio, U0)
    c0 = N / 2.0
    i = np.arange(N, dtype=np.float64)
    # mu: cell volume fraction of the core, 8x8 sub-samples per cell (z-independent)
    sub = (np.arange(8) + 0.5) / 8.0
    xs = (i[:, None] + sub[None, :]).ravel()
    XS, YS = np.meshgrid(xs, xs, indexing="ij")
    inside = (np.hypot(XS - c0, YS - c0) < R).astype(np.float64)
    C = inside.reshape(N, 8, N, 8).mean(axis=(1, 3))
    mu = mu_in * C + mu_out * (1.0 - C)
    rep = lambda a: F(np.repeat(a[:, :, None], nz, axis=2))
    # forcing at cell centres
    Xc, Yc = np.meshgrid(i + 0.5, i + 0.5, indexing="ij")
    rc = np.hypot(Xc - c0, Yc - c0)
    ft = f_theta(rc)
    with np.errstate(invalid="ignore", divide="ignore"):
        sn, cs = np.where(rc > 0, (Yc - c0) / rc, 0.0), np.where(rc > 0, (Xc - c0) / rc, 0.0)
    fx, fy = -ft * sn, ft * cs
    # exact velocity on the faces
    Xu, Yu = np.meshgrid(i, i + 0.5, indexing="ij")
    Xv, Yv = np.meshgrid(i + 0.5, i, indexing="ij")
    ru, rv = np.hypot(Xu - c0, Yu - c0), np.hypot(Xv - c0, Yv - c0)
    ue = -u_theta(ru) * (Yu - c0) / np.maximum(ru, 1e-300)
    ve = u_theta(rv) * (Xv - c0) / np.maximum(rv, 1e-300)
    s = pf.Solver(N, N, nz)
    s.set_rho(1.0)
    s.set_mu(mu_out)
    s.set_dt(D / min(mu_in, mu_out))  # D = mu_min dt / (rho h^2), h = 1
    s.set_advection(False)
    s.set_pressure_pcg(True, 400, 1e-13)
    s.diagnostics.set_velocity_solver_params(400, 1e-12, 2)
    s.set_pressure_geometry(F(np.full((N, N, nz), 10.0)))
    s.enable_cell_force()
    s.diagnostics.set_property_mode("variable", False)  # registers the "mu" field
    s.set_field("mu", rep(mu))
    s.diagnostics.set_property_mode("variable", False)
    if full:
        s.set_viscous_stress("full")
    for nm, a in (("force_x", rep(fx)), ("force_y", rep(fy)), ("force_z", rep(0 * fx))):
        s.set_field(nm, a)
        s.diagnostics.exchange_field(nm)
    if init is None:
        s.set_velocity(0, rep(ue))
        s.set_velocity(1, rep(ve))
    else:
        for k in range(3):
            s.set_velocity(k, F(init[k]))
    prev = np.array(s.get_u())
    steps, hist = 0, []
    for steps in range(1, max_steps + 1):
        s.step()
        u = np.array(s.get_u())
        du = np.abs(u - prev).max()
        prev = u
        if steps % 50 == 0:
            hist.append(np.abs(u).max())
        if du < tol * U0:
            break
    u, v = np.array(s.get_u())[:, :, 0], np.array(s.get_v())[:, :, 0]
    e = max(np.abs(u - ue).max(), np.abs(v - ve).max()) / U0
    return e, steps, du / U0, hist, s


def gate_rotcore(Ns, ratios, D, max_steps):
    for ratio in ratios:
        ef, el = {}, {}
        for N in Ns:
            t0 = time.time()
            ef[N], nf, df, _, _ = rotcore(N, ratio, True, D=D, max_steps=max_steps)
            el[N], nl, dl, _, _ = rotcore(N, ratio, False, D=D, max_steps=max_steps)
            print(f"  rotcore ratio {ratio:g} N {N:4d}: full e {ef[N]:.4e} ({nf} steps, last "
                  f"du {df:.1e}); laplacian e {el[N]:.4e} ({nl} steps, last du {dl:.1e})  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
        if len(Ns) >= 3:
            a, b = Ns[-2], Ns[-1]
            rf, rl = ef[a] / ef[b], el[a] / el[b]
            verdict(f"G2 ratio {ratio:g} full order", rf >= 1.75,
                    f"e({a})/e({b}) = {rf:.3f} (>= 1.75, order {np.log2(rf):.2f})")
            verdict(f"G2 ratio {ratio:g} laplacian", el[b] >= 5 * ef[b] and rl <= 1.25,
                    f"e_lap({b}) / e_full({b}) = {el[b] / ef[b]:.2f} (>= 5); e_lap({a})/e_lap({b}) "
                    f"= {rl:.3f} (<= 1.25)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="*", default=["bit", "wall", "couette", "refuse"])
    ap.add_argument("--n", default="32,64,128")
    ap.add_argument("--ratios", default="0.1,10")
    ap.add_argument("--D", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=200000)
    a = ap.parse_args()
    for c in a.cmd:
        if c == "rotcore":
            gate_rotcore([int(x) for x in a.n.split(",")], [float(x) for x in a.ratios.split(",")],
                         a.D, a.max_steps)
        else:
            globals()["gate_" + c]()
    print("ALL PASS" if OK else "SOME MISS")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
