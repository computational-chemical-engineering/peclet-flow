#!/usr/bin/env python3
"""The slow (pi,0,0) pressure-checkerboard tail of the collocated steady march (2026-10-01).

Instrument behind doc/collocated_invisible_subspace.md §11. It marches the case of
study_avg_velocity_spheres.py (one sphere per periodic unit cell, phi = 0.125, centred on a grid
vertex, rho = mu = f = 1, nu dt / h^2 = beta, MG-PCG rtol 1e-8, velocity cap 200) and records per
step <u_x>, rms(du) and the rms of the fluid pressure increment, plus the last four iterates, so
that the asymptotic contraction rate and the spatial structure of the slow component can be read.

  geom N1,N2,..                    fluid-centre / crossing-fraction census of the sphere per N
  run TAG N beta stag|col auto|ghost|gauge-exact STEPS [shift_h] [opts]
                                   march STEPS steps; shift_h moves the centre by (s,s,s) h;
                                   opts (comma list): norot (PM I ablation), prtol12 (rtol 1e-12)
                                   -> $PROBE_OUT/TAG.npz (default ./probe_out)
  analyze TAG..                    tail rate per step, its amplitude, the extrapolated limit
  structure TAG..                  where the slow component lives: wall band, Fourier content
  stops TAG..                      replay the history through the study's old (5bcf353) and
                                   current stop tests: step and K bias against the limit

Run (from flow/, pool bounded): OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<flow build>
python tests/study/collocated_checkerboard_tail.py run col16 16 6 col auto 800
"""
import math
import os
import sys

import numpy as np

L, RHO, MU, F, PHI = 1.0, 1.0, 1.0, 1.0, 0.125
OUT = os.environ.get("PROBE_OUT", "probe_out")
HERE = os.path.dirname(os.path.abspath(__file__))


def radius():
    return (3.0 * PHI / (4.0 * math.pi)) ** (1.0 / 3.0) * L


def sdf_on(cx, cy, cz, shift_h, h):
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    c = 0.5 * L + shift_h * h
    return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - radius()


def old_stop(m, tol=1e-6, every=5):
    """The stop of study_avg_velocity_spheres.py at 5bcf353: |m_n - m_{n-5}| < tol |m_n|."""
    prev = 0.0
    for it in range(len(m)):
        if it % every == every - 1:
            if it > 10 and abs(m[it] - prev) < tol * (abs(m[it]) + 1e-30):
                return it + 1
            prev = m[it]
    return -1


def new_stop(m):
    """The study's current march() replayed on a recorded history (-1 if it never stops)."""
    sys.path.insert(0, HERE)
    import study_avg_velocity_spheres as st
    k = {"i": -1}

    def step():
        k["i"] += 1
        if k["i"] >= len(m):
            raise IndexError

    try:
        n, ok = st.march(step, lambda: float(m[k["i"]]))
    except IndexError:
        return -1
    return n if ok else -1


def geom(Ns):
    R = radius()
    for N in Ns:
        h = L / N
        c = (np.arange(N) + 0.5) * h
        s = sdf_on(c, c, c, 0.0, h) / h
        th = []
        for ax in range(3):
            a, b = s, np.roll(s, -1, axis=ax)
            m1, m2 = (a > 0) & (b < 0), (a < 0) & (b > 0)
            th += [a[m1] / (a[m1] - b[m1]), b[m2] / (b[m2] - a[m2])]
        th = np.concatenate(th)
        print(f"N={N:3d} R/h={R / h:7.4f}  min fluid sdf/h={s[s > 0].min():.4f}  "
              f"#fluid<0.05h={np.sum((s > 0) & (s < 0.05)):3d}  min theta={th.min():.4f}  "
              f"#theta<0.05={np.sum(th < 0.05)}")


def run(tag, N, beta, solver, scheme, steps, shift=0.0, opts=""):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()
    opts = set(o for o in opts.split(",") if o)
    s = (flow.Solver if solver == "stag" else flow.SolverColocated)((N, N, N), extent=(L, L, L))
    h = L / N
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(beta * RHO * h * h / MU)
    s.set_body_force((F, 0.0, 0.0)); s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, max_iter=200, rtol=1e-12 if "prtol12" in opts else 1e-8)
    if solver == "col" and scheme != "auto":
        s.set_collocated_scheme(scheme)
    if "norot" in opts:
        s.diagnostics.set_rotational_pressure(False)
    cx, cy, cz = s.cell_centers()
    sdf = sdf_on(cx, cy, cz, shift, h)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True)

    m, du, dp = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    pit, mres = np.zeros(steps, int), np.zeros(steps)
    keep, prevU, prevP, fluid = {}, None, None, sdf > 0
    for it in range(steps):
        s.step()
        U = np.stack([s.get_u(), s.get_v(), s.get_w()])
        P = s.get_p().copy()
        m[it] = float(U[0].mean())
        pit[it] = s.diagnostics.last_pressure_iterations()
        mres[it] = s.diagnostics.last_momentum_residual()
        if prevU is not None:
            du[it] = np.sqrt(np.mean((U - prevU) ** 2))
            d = (P - prevP)[fluid]
            dp[it] = np.sqrt(np.mean((d - d.mean()) ** 2))
        prevU, prevP = U, P
        keep[it] = (U, P)
        keep.pop(it - 4, None)
    its = sorted(keep)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), m=m, du=du, dp=dp, pit=pit, mres=mres,
                        sdf=sdf / h, U=np.stack([keep[i][0] for i in its]),
                        P=np.stack([keep[i][1] for i in its]), N=N, beta=beta)
    print(f"[{tag}] N={N} beta={beta} {solver}/{scheme} shift={shift} opts={sorted(opts)} "
          f"old_stop={old_stop(m)} vmg={bool(s.diagnostics.velocity_multigrid_active())} "
          f"p-iters(last)={pit[-1]} mom-res(last)={mres[-1]:.2e} m_final={m[-1]:.10e}", flush=True)


def tail(m, floor=1e-11):
    """(per-step rate, window, back-projected amplitude, geometric-extrapolated limit)."""
    rel = np.abs(np.diff(m)) / abs(m[-1])
    ok = np.where(rel > floor)[0]
    hi = ok[-1] + 1 if len(ok) else len(m) - 1
    lo = max(5, hi // 2)
    r = float(np.exp((np.log(rel[hi - 1]) - np.log(rel[lo])) / (hi - 1 - lo)))
    return r, (lo, hi), rel[lo] / r ** lo, m[hi] + (m[hi] - m[hi - 1]) * r / (1.0 - r)


def analyze(tag):
    z = np.load(os.path.join(OUT, tag + ".npz"))
    m, du, dp, beta = z["m"], z["du"], z["dp"], float(z["beta"])
    r, (lo, hi), A0, m_inf = tail(m)
    print(f"[{tag}] N={int(z['N'])} beta={beta} tail rate {r:.5f}/step over [{lo},{hi}) "
          f"(as a backward-Euler mode lambda L^2 = {(1 / r - 1) / beta * int(z['N']) ** 2:.2f}); "
          f"|dm|/m amplitude at step 0 {A0:.2e}; m_inf = {m_inf:.10e}")
    rel = np.abs(np.diff(m)) / abs(m[-1])
    print("   step  |dm|/|m|    rms du      rms dP(fluid)")
    for k in [0, 4, 9, 19, 29, 49, 69, 99, 149, 199, 299, 399, 599, 799, 1199]:
        if k < len(m) - 1:
            print(f"   {k + 1:4d}  {rel[k]:.3e}   {du[k + 1]:.3e}   {dp[k + 1]:.3e}")


def structure(tag):
    """Where the last step's increment lives: wall band and Fourier content."""
    z = np.load(os.path.join(OUT, tag + ".npz"))
    U, P, sdf = z["U"], z["P"], z["sdf"]
    N, fluid = sdf.shape[0], sdf > 0
    d = U[-1] - U[-2]
    dP = P[-1] - P[-2]
    dP = np.where(fluid, dP - dP[fluid].mean(), 0.0)
    e = (d ** 2).sum(axis=0)
    print(f"  [{tag}] last-step increment:")
    for lo, hi, name in [(-1e9, 0, "solid"), (0, 1, "0<sdf<h"), (1, 2, "h<sdf<2h"),
                         (2, 1e9, "sdf>2h")]:
        msk = (sdf > lo) & (sdf <= hi)
        print(f"    {name:9s} {100 * e[msk].sum() / e.sum():5.1f}% of |du|^2, "
              f"{100 * (dP[msk] ** 2).sum() / (dP ** 2).sum():5.1f}% of |dP|^2 "
              f"({100 * msk.mean():4.1f}% of cells)")
    h2 = N // 2
    for name, f in (("du_x", d[0]), ("dP", dP), ("P", np.where(fluid, P[-1] - P[-1][fluid].mean(), 0))):
        E = np.abs(np.fft.fftn(f)) ** 2
        E /= E.sum()
        cb = {"(pi,0,0)": E[h2, 0, 0], "(0,pi,0)": E[0, h2, 0], "(0,0,pi)": E[0, 0, h2],
              "(pi,pi,pi)": E[h2, h2, h2]}
        print(f"    {name:5s} energy fraction " + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in cb.items()))


def stops(tag):
    z = np.load(os.path.join(OUT, tag + ".npz"))
    m = z["m"]
    m_inf = tail(m)[3]
    f = [old_stop(m), new_stop(m)]
    g = ["%5d (K bias %+.1e)" % (n, m_inf / m[n - 1] - 1) if n > 0 else "  never" for n in f]
    print(f"[{tag}] old stop {g[0]}   current stop {g[1]}")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "geom":
        geom([int(x) for x in a[0].split(",")])
    elif cmd == "run":
        run(a[0], int(a[1]), float(a[2]), a[3], a[4], int(a[5]),
            float(a[6]) if len(a) > 6 else 0.0, a[7] if len(a) > 7 else "")
    else:
        for t in a:
            {"analyze": analyze, "structure": structure, "stops": stops}[cmd](t)
