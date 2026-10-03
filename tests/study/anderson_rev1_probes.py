#!/usr/bin/env python3
"""Probes behind revision 1 of doc/steady_acceleration.md (an INSTRUMENT, not a gate).

Three subcommands, each on the oracle's cases (tests/study/anderson_oracle.py builds the solvers):

  slowmode   plain march on a case, per-step residual split (velocity / c_P-weighted pressure) and
             the full residual field at chosen steps; then the structure of the late residual:
             decay rates, alignment, energy shares, the fluid components of the face-aperture graph
             (sealed pockets) and the share of the pressure residual that sits off the main
             component at aperture thresholds 0 ... 0.2.                       (note §1.2)
  phase-a    phase A run unconditionally (acc.step(True) every call, no handover) with the monitor
             read after every map output: steps until the monitor stays within 1e-5 / 3e-5 of the
             reference and until the velocity / W residual first reaches 3e-7.  ("Revision 1" R1)
  certify    the full revision-1 driver with the monitor recorded after every step, then the
             certification blocks replayed: d, R and the remainder bound per block.   (§7, R4)

Run from the flow tree with the OpenMP pool bounded, e.g.

    OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build_omp \\
        python tests/study/anderson_rev1_probes.py phase-a --case sphere --scheme collocated --N 16 \\
        --window 5 --metric V --steps 300 --out e4.npz

The dense bed needs `--case bed --arrangement <npz>` (centres, radii, L; see the log for how it is
generated from the A1 scripts).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anderson_oracle as AO  # noqa: E402


def build_case(flow, a, settings="production"):
    """The oracle's sphere / bed cases (anderson_oracle.main's geometry), returning (solver,
    oracle kwargs, settings, K-of-monitor)."""
    st = AO.settings_of(settings)
    if a.case == "sphere":
        L, phi = 1.0, a.phi
        R = (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0) * L

        def sdf_fn(cx, cy, cz):
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
            return np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - R

        def K(u):
            return L ** 3 / (6.0 * math.pi * a.mu * R * u)
    else:
        from scipy.spatial import cKDTree
        z = np.load(a.arrangement)
        cen, rad, L = np.asarray(z["centres"], float), np.asarray(z["radii"], float), float(z["L"])

        def sdf_fn(cx, cy, cz):
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
            pts = np.stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")], axis=1)
            d, idx = cKDTree(np.mod(cen, L), boxsize=L).query(np.mod(pts, L), k=min(4, len(cen)))
            return (d - rad[idx]).min(axis=1).reshape(X.shape, order="C")

        def K(u):
            return L ** 3 / (3.0 * math.pi * a.mu * 2.0 * float(rad.mean()) * u * len(rad))
    h = L / a.N
    dt = a.dt if a.dt is not None else a.beta * h * h / a.mu
    s, okw = AO.build_solver(flow, a.scheme, a.N, sdf_fn, L, rho=1.0, mu=a.mu, F=1.0, dt=dt,
                             advection=a.advection, settings=st)
    return s, okw, st, K


def inner(v, G=2):
    return np.array(v[G:-G, G:-G, G:-G])


def cmd_slowmode(flow, a):
    s, okw, _, K = build_case(flow, a)
    names = ["u", "v", "w", "p"]
    views = [s.diagnostics.field_view(n) for n in names]
    fluid = inner(s.diagnostics.field_view("sdf")) > 0.0
    cP = okw["c_p"]
    save_at = {int(k) for k in a.save.split(",")} if a.save else set()
    x = [inner(v) for v in views]
    rows, saved = [], {}
    for k in range(1, a.steps + 1):
        s.step()
        y = [inner(v) for v in views]
        r = [y[f] - x[f] for f in range(4)]
        rp = r[3][fluid] - r[3][fluid].mean()
        rows.append((k, *[float(np.sum(r[f] ** 2)) for f in range(3)], cP * cP * float(np.sum(rp ** 2)),
                     float(sum(np.sum(y[f] ** 2) for f in range(3))), float(s.get_u().mean())))
        if k in save_at:
            saved[f"r{k}"] = np.stack(r)
        x = y
    rows = np.array(rows)
    W = np.sqrt(rows[:, 1:5].sum(1) / rows[:, 5])
    V = np.sqrt(rows[:, 1:4].sum(1) / rows[:, 5])
    P = np.sqrt(rows[:, 4] / rows[:, 5])
    ref = a.ref if a.ref else rows[-1, 6]
    print(f"reference monitor {ref:.12e} (K {K(ref):.8f})")
    marks = [k for k in (1, 10, 50, 100, 150, 200, 250, 300, 325, 400, 500, 600, 800, 1000, 1200,
                         1500, 2000) if k <= a.steps]
    for k in marks:
        print(f"step {k:5d}  W {W[k-1]:.3e}  vel {V[k-1]:.3e}  pres {P[k-1]:.3e}  "
              f"monitor err {abs(rows[k-1, 6] / ref - 1):.1e}")
    for k0, k1 in zip(marks[:-1], marks[1:]):
        if k0 >= 50:
            rt = lambda q: (q[k1 - 1] / q[k0 - 1]) ** (1.0 / (k1 - k0))  # noqa: E731
            print(f"rate {k0:4d}-{k1:4d}: W {rt(W):.6f} vel {rt(V):.6f} pres {rt(P):.6f}")
    # structure: fluid components of the aperture graph. get_ox[i] is the -x face of cell i, i.e.
    # the face between cells i-1 and i (the binding's docstring); rolled by -1 so that opn[ax][i]
    # is the face between cells i and i+1. (Corrected 2026-10-03: until then this read get_ox[i]
    # itself as the face (i, i+1), one cell off, which turned 205 sealed pockets / 217 cells into
    # 1,085 / 1,168 on the A1 bed -- tests/study/pocket_pressure_probe.py checks the convention.)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = fluid.shape
    idx = np.arange(fluid.size).reshape(n, order="F")
    opn = [np.roll(np.asarray(o).reshape(n, order="F"), -1, axis=ax)
           for ax, o in enumerate((s.get_ox(), s.get_oy(), s.get_oz()))]
    mods = sorted(saved)
    # the cut-cell operator's own graph at aperture 0: every open face couples its two cells,
    # whatever their sdf sign, so a fluid cell joined to the main space only through a solid-
    # CENTRED cut cell is not sealed. 'operator' below = fluid cells outside its main component.
    # Its apertures are the PROJECTION's (get_ox_proj: binary under the collocated ghost scheme,
    # identical to get_ox on the staggered grid).
    opp = [np.roll(np.asarray(o).reshape(n, order="F"), -1, axis=ax)
           for ax, o in enumerate((s.get_ox_proj(), s.get_oy_proj(), s.get_oz_proj()))]
    node = fluid.copy()
    for ax in range(3):
        o = opp[ax] > 0.0
        node |= o | np.roll(o, 1, axis=ax)
    for th, nodes in ((0.0, node), (0.0, fluid), (1e-2, fluid), (5e-2, fluid), (0.1, fluid),
                      (0.2, fluid)):
        Ii, Jj = [], []
        for ax in range(3):
            o = opp[ax] if nodes is node else opn[ax]
            m = (o > th) & nodes & np.roll(nodes, -1, axis=ax)
            Ii.append(idx[m])
            Jj.append(np.roll(idx, -1, axis=ax)[m])
        Ii, Jj = np.concatenate(Ii), np.concatenate(Jj)
        A = coo_matrix((np.ones(Ii.size), (Ii, Jj)), shape=(fluid.size, fluid.size))
        _, lab = connected_components(A, directed=False)
        fl = lab.reshape(n, order="F")[fluid]
        sizes = np.bincount(fl)
        main = int(np.argmax(sizes))
        line = []
        for key in mods:
            v = saved[key][3][fluid]
            v = v - v.mean()
            line.append(f"{key}: off-main {float(np.sum(v[fl != main] ** 2) / np.sum(v ** 2)):.3f}")
        tag = "operator graph, " if nodes is node else ""
        print(f"{tag}aperture > {th:g}: {len(np.unique(fl))} fluid components, main {sizes[main]} "
              f"of {fl.size} fluid cells ({fl.size - sizes[main]} off main); " + "; ".join(line))
    for key in mods:
        r = saved[key]
        rp = r[3][fluid] - r[3][fluid].mean()
        ev = float(sum(np.sum(r[f] ** 2) for f in range(3)))
        ep = cP * cP * float(np.sum(rp ** 2))
        e = np.sort(rp ** 2)[::-1]
        n90 = int(np.searchsorted(np.cumsum(e) / e.sum(), 0.9)) + 1
        print(f"{key}: pressure share of W^2 {ep / (ev + ep):.6f}; 90% of the pressure energy in "
              f"{n90} of {rp.size} fluid cells")


def cmd_phase_a(flow, a):
    s, okw, _, K = build_case(flow, a)
    okw = dict(okw, metric=a.metric, ritz_scope="mixed")
    acc = AO.AndersonOracle(s, window=a.window, **okw)
    rows = []
    for k in range(1, a.steps + 1):
        acc.step(True)
        if acc.status != "active":
            print(f"status {acc.status} at step {k}: {acc.reason}")
            break
        R = acc.Rprev
        vel = sum(float(np.dot(R[f, acc.inner], R[f, acc.inner])) for f in acc.vel)
        rp = R[3, acc.fluid] - R[3, acc.fluid].mean()
        pres = okw["c_p"] ** 2 * float(np.dot(rp, rp))
        U2 = acc.usq(acc.Gprev)
        rows.append((k, math.sqrt((vel + pres) / U2), math.sqrt(vel / U2), float(s.get_u().mean()),
                     acc.ritzRadius))
    rows = np.array(rows)
    if a.out:
        np.savez(a.out, rows=rows)
    ref = a.ref if a.ref else rows[-1, 3]
    err = np.abs(rows[:, 3] / ref - 1)

    def stays(tol):
        bad = np.where(err > tol)[0]
        return bad[-1] + 2 if bad.size else 1

    def first(col, tol):
        i = np.where(rows[:, col] <= tol)[0]
        return int(i[0]) + 1 if i.size else None
    print(f"{a.case} {a.scheme} N={a.N} m={a.window} metric={a.metric}: reference {ref:.12e}; "
          f"monitor within 3e-5 from step {stays(3e-5)}, within 1e-5 from {stays(1e-5)}; velocity "
          f"residual <= 3e-7 at {first(2, 3e-7)}; W residual <= 3e-7 at {first(1, 3e-7)}")


def cmd_certify(flow, a):
    s, okw, st, K = build_case(flow, a)
    okw = dict(okw, metric=a.metric, ritz_scope="mixed")
    mons, holder = [], {}

    def cb(steps, phase):
        mons.append((steps, phase, float(s.get_u().mean())))
    res = AO.march_to_steady(s, lambda: float(s.get_u().mean()), rtol=st["rtol"],
                             max_steps=st["max_steps"], window=a.window, callback=cb, oracle_kw=okw,
                             holder=holder, budget=a.budget, rev=1)
    print(f"{a.case} {a.scheme} N={a.N} m={a.window}: converged={res['converged']} "
          f"{res['reason']} steps={res['steps']} accelerated={res['accelerated_steps']} "
          f"K={K(res['monitor']):.10f}")
    runs = []
    for k, ph, m in mons:
        if not runs or runs[-1][0] != ph:
            runs.append((ph, []))
        runs[-1][1].append((k, m))
    for ph, rows in runs:
        if ph == "accelerate":
            print(f"  accelerate {rows[0][0]}-{rows[-1][0]}")
            continue
        out, prev, dprev = [], None, None
        for k, m in [rows[i] for i in range(4, len(rows), 5)]:  # the instrument's samples
            if prev is not None:
                d = m - prev
                if dprev:
                    R = d / dprev
                    bound = abs(d) / (1 - max(R, 0.997 ** 5)) if R < 1 else math.inf
                    out.append(f"[{k}: d/|m| {d / abs(m):+.1e} R {R:+.3f} bound/|m| "
                               f"{bound / abs(m):.1e}]")
                else:
                    out.append(f"[{k}: d/|m| {d / abs(m):+.1e}]")
                dprev = d
            prev = m
        print(f"  {ph} {rows[0][0]}-{rows[-1][0]}: " + " ".join(out))


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("slowmode", "phase-a", "certify"))
    ap.add_argument("--case", choices=("sphere", "bed"), default="sphere")
    ap.add_argument("--scheme", choices=("staggered", "collocated"), required=True)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--phi", type=float, default=0.125)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--advection", default=None)
    ap.add_argument("--arrangement", default=None)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--metric", choices=("V", "W"), default="V")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--save", default="", help="slowmode: steps whose residual field is kept")
    ap.add_argument("--ref", type=float, default=None, help="reference monitor value")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"slowmode": cmd_slowmode, "phase-a": cmd_phase_a, "certify": cmd_certify}[a.command](flow, a)


if __name__ == "__main__":
    main()
