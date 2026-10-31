"""G-SV1, G-SV1c, G-SV1d (doc/vof_sharp_viscous_stress.md §11): the 3-D sharp viscous operator on
planar-interface Stokes flows, through the production stepper.

Cases (one planar interface, phase A = C = 1 on the side xi < 0; mu_B = 1, mu_A = ratio):
  normal   n = (cos b cos g, cos b sin g, sin b), b, g in {0, 22.5, 45} deg (9 orientations)
  plane    xi = n.(x - c) - s,  c = (1/2, 1/2, 1/2),  s = (k/8) h,  k = 0..7
  tangents (coordinator ruling B2, 2026-10-31):
           t1 = normalize(e_z x n)  (t1 = e_x if |n x e_z| < 1e-12),  t2 = n x t1,
           t  = (t1 + t2)/sqrt(2)   -- the shear direction AND the extension tangent
  couette     u = U(xi) t,  U = xi/mu                      (tau = 1)
  poiseuille  u = U(xi) t,  U = (xi - xi^2/2)/mu,  uniform body force along t (tau_0 = 1)
  extension   u = eps (xi n - eta t),  eta = t.x (B2),  eps = 1  (the stepper finds the pressure
              jump 2 eps [mu] itself)
  general     poiseuille + extension
Gate sets: gsv1 = couette, poiseuille at ratio 1e-2, 1e2; gsv1c = extension, general at 1e-2, 1e2;
gsv1d = couette, poiseuille at 1e-3, 1e3. N = 24, 32, 48, 64 (note §5.7 Q-C).
PASS (note §11, USER 2026-10-31): per kind x ratio x orientation, the LS order of the offset envelope
e_env(N) = max over k of err is >= 1.8 over N 24/32/48/64; couette also passes if err <= 1e-10 at every
N; extension needs err <= 1e-9 at every N. Per-offset orders are reported, not gating.

Discretization of the data: C is the exact cell volume fraction of the half-space; every velocity
datum is the exact face average of the analytic field (piecewise polynomial in xi, integrated
exactly). Boundary (coordinator ruling B1, option A): the production Dirichlet profile
(set_domain_bc_profile) on all six faces, fed with the EXACT wall-plane values: the normal
component's average over the wall face, each tangential component's average over the wall-plane
segment of its own face. Uniform rho = 1, D_hi = mu_hi dt/(rho h^2) = 100, momentum advection off,
sharp viscous interface. The colour is frozen: set_vof(C0) after every step. Start from the exact
face averages; plain production steps until the velocity increment of one step is
<= 1e-12 max|u_exact| (ruling B3's criterion). Error = max|u_h - u_face| / max|u_face| over the
three face arrays.

Also recorded per case (instruments, not gate measures): the max error binned by the face's
distance to the nearest wall (half-cell bins, `wallmap`), the same joined with the distance to the
interface (`wximap[wall bin 0..3, >=4][|xi|/h in [0,1), [1,2), [2,4), >=4]`), and the location of
the max error (its wall distance and |xi|/h).

  nice -n 10 python gsv1.py run  --set gsv1 --N 24,32 --procs 4 --out res.jsonl   (resumes)
  python gsv1.py table --out res.jsonl                                           (verdict lines)
  python gsv1.py one KIND BETA GAMMA K RATIO N                                   (one JSON line)
"""
import argparse
import itertools
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

GX, GW = np.polynomial.legendre.leggauss(6)
ANGLES = (0.0, 22.5, 45.0)
SETS = {
    "gsv1": (("couette", "poiseuille"), (1e-2, 1e2)),
    "gsv1c": (("extension", "general"), (1e-2, 1e2)),
    "gsv1d": (("couette", "poiseuille"), (1e-3, 1e3)),
}
D_HI, TOL, MAX_STEPS = 100.0, 1e-12, 200000


def cube_frac(m, al):
    """Volume fraction of {y in [0,1]^3 : m.y < al} (inclusion-exclusion, exact)."""
    m = np.array(m, float)
    al = al - sum(x for x in m if x < 0)
    m = np.abs(m)
    nz = [x for x in m if x > 1e-300]
    d = len(nz)
    if d == 0:
        return 1.0 if al > 0 else 0.0
    tot = 0.0
    for S in itertools.product((0, 1), repeat=d):
        r = al - sum(x for x, s in zip(nz, S) if s)
        if r > 0:
            tot += (-1) ** sum(S) * r**d
    return min(1.0, max(0.0, tot / (math.factorial(d) * float(np.prod(nz)))))


def tangents(n):
    ez = np.array([0.0, 0.0, 1.0])
    c = np.cross(ez, n)
    t1 = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(np.cross(n, ez)) < 1e-12 else c / np.linalg.norm(c)
    t2 = np.cross(n, t1)
    return t1, t2, (t1 + t2) / math.sqrt(2.0)


class Case:
    def __init__(self, kind, beta, gamma, k, ratio, N):
        b, g = math.radians(beta), math.radians(gamma)
        self.n = np.array([math.cos(b) * math.cos(g), math.cos(b) * math.sin(g), math.sin(b)])
        self.t = tangents(self.n)[2]
        self.kind, self.N, self.h = kind, N, 1.0 / N
        self.s = k / 8.0 * self.h
        self.muA, self.muB = ratio, 1.0
        self.c = np.array([0.5, 0.5, 0.5])

    def xi(self, x):
        return self.n @ (x - self.c) - self.s

    def U(self, xi):
        mu = np.where(xi < 0, self.muA, self.muB)
        if self.kind == "couette":
            return xi / mu
        if self.kind in ("poiseuille", "general"):
            return (xi - 0.5 * xi * xi) / mu
        return 0.0 * xi

    def ext(self, x):
        if self.kind not in ("extension", "general"):
            return np.zeros(3)
        return self.xi(x) * self.n - float(self.t @ x) * self.t

    def avgU(self, xi0, wa, wb):
        """E[U(xi0 + X)], X = sum of two centred uniforms of widths |wa|, |wb| (exact)."""
        a, b = abs(wa), abs(wb)
        if a < 1e-15 and b < 1e-15:
            return float(self.U(np.array(xi0)))
        if b < 1e-15 or a < 1e-15:
            w = max(a, b)
            pts = sorted({-w / 2, w / 2} | ({-xi0} if -w / 2 < -xi0 < w / 2 else set()))
            dens = lambda x: np.full_like(x, 1.0 / w)
        else:
            br = {-(a + b) / 2, (a + b) / 2, -abs(a - b) / 2, abs(a - b) / 2}
            if -(a + b) / 2 < -xi0 < (a + b) / 2:
                br.add(-xi0)
            pts = sorted(br)
            dens = lambda x: np.clip((a + b) / 2 - np.abs(x), 0.0, min(a, b)) / (a * b)
        tot = 0.0
        for lo, hi in itertools.pairwise(pts):
            if hi - lo <= 0:
                continue
            x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * GX
            tot += 0.5 * (hi - lo) * float(np.dot(GW, self.U(xi0 + x) * dens(x)))
        return tot

    def face_avg(self, comp, x, axes):
        """Average of u_comp over the h-square (two axes) or h-segment (one axis) centred x."""
        w = [self.n[a] * self.h for a in axes] + [0.0] * (2 - len(axes))
        return self.t[comp] * self.avgU(self.xi(x), w[0], w[1]) + self.ext(x)[comp]


def wall_dist(N):
    """Per component: distance (cells) of each face position to the nearest domain wall."""
    out = []
    for comp in range(3):
        g = []
        for a in range(3):
            idx = np.arange(N, dtype=float)
            pos = idx if a == comp else idx + 0.5
            g.append(np.minimum(pos, N - pos))
        out.append(np.minimum(np.minimum(g[0][:, None, None], g[1][None, :, None]), g[2][None, None, :]))
    return np.stack(out)


def run_case(key):
    kind, beta, gamma, k, ratio, N = key
    os.environ.setdefault("OMP_PROC_BIND", "false")
    import peclet.flow as pf

    t0 = time.time()
    cs = Case(kind, beta, gamma, k, ratio, N)
    h = cs.h
    C = np.zeros((N, N, N))
    ex = np.zeros((3, N, N, N))
    xin = np.zeros((3, N, N, N))
    for i, j, kk in itertools.product(range(N), repeat=3):
        xc = np.array([i, j, kk], float) * h
        C[i, j, kk] = cube_frac(cs.n, (cs.s - cs.n @ (xc - cs.c)) / h)
        for comp in range(3):
            x = (np.array([i, j, kk], float) + 0.5) * h
            x[comp] -= 0.5 * h
            ex[comp, i, j, kk] = cs.face_avg(comp, x, tuple(a for a in range(3) if a != comp))
            xin[comp, i, j, kk] = abs(cs.xi(x)) / h
    dt = D_HI * h * h / max(cs.muA, cs.muB)
    amp = 0.01 * h / (dt * float(np.max(np.abs(ex))))  # colour CFL <= 0.01; linear: scale-free
    ex *= amp
    s = pf.Solver((N, N, N), extent=(1.0, 1.0, 1.0))
    s.set_dt(dt)  # first: the first set_dt pins the internal time scale
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3  # resampleBcProfile's (b, c) = ((a+1)%3, (a+2)%3)
        for side in (0, 1):
            prof = np.zeros((N, N, 3))
            for ib in range(N):
                for ic in range(N):
                    x = np.zeros(3)
                    x[a] = float(side)
                    x[b], x[c] = (ib + 0.5) * h, (ic + 0.5) * h
                    prof[ib, ic, a] = cs.face_avg(a, x, (b, c))
                    xb = x.copy()
                    xb[b] = ib * h
                    prof[ib, ic, b] = cs.face_avg(b, xb, (c,))
                    xc = x.copy()
                    xc[c] = ic * h
                    prof[ib, ic, c] = cs.face_avg(c, xc, (b,))
            s.set_domain_bc_profile(("-" if side == 0 else "+") + "xyz"[a], np.ascontiguousarray(amp * prof))
    s.set_rho(1.0)
    s.set_mu(max(cs.muA, cs.muB))
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    C = np.asfortranarray(C)
    s.set_vof(C)
    s.set_property_model("mu", "linear", "C", [cs.muB, cs.muA - cs.muB])
    s.set_advection(False)
    if kind in ("poiseuille", "general"):
        s.set_body_force(tuple(amp * cs.t))
    s.set_pressure_pcg(True, 800, 1e-12)
    s.diagnostics.set_vof_viscous_interface("sharp")
    for comp in range(3):
        s.set_velocity(comp, np.asfortranarray(ex[comp]))
    U = float(np.max(np.abs(ex)))
    prev = np.stack([s.get_u(), s.get_v(), s.get_w()])
    inc, it, status = math.inf, 0, "max_steps"
    for it in range(1, MAX_STEPS + 1):
        s.step()
        s.set_vof(C)
        cur = np.stack([s.get_u(), s.get_v(), s.get_w()])
        inc = float(np.max(np.abs(cur - prev))) / U
        prev = cur
        if not math.isfinite(inc):
            status = "blowup"
            break
        if inc <= TOL:
            status = "ok"
            break
    e = np.abs(prev - ex) / U
    err = float(np.max(e))
    wd = wall_dist(N)
    bins = np.floor(2.0 * wd + 1e-9).astype(int)  # half-cell bins
    wallmap = [float(np.max(e[bins == q])) if np.any(bins == q) else 0.0 for q in range(int(bins.max()) + 1)]
    xbins = np.digitize(xin, (1.0, 2.0, 4.0))  # |xi|/h in [0,1), [1,2), [2,4), >= 4
    wb = np.minimum(bins, 4)
    wximap = [[float(np.max(e[(wb == q) & (xbins == r)])) if np.any((wb == q) & (xbins == r)) else 0.0
               for r in range(4)] for q in range(5)]
    am = np.unravel_index(int(np.argmax(e)), e.shape)
    return {"kind": kind, "beta": beta, "gamma": gamma, "k": k, "ratio": ratio, "N": N, "status": status, "steps": it,
                "inc": inc, "err": err, "at_comp": int(am[0]), "at_cell": [int(v) for v in am[1:]],
                "at_walldist": float(wd[am]), "at_xi": float(xin[am]), "wallmap": wallmap, "wximap": wximap,
                "secs": round(time.time() - t0, 1)}


def keyof(r):
    return (r["kind"], r["beta"], r["gamma"], r["k"], r["ratio"], r["N"])


def load(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[keyof(r)] = r
    return out


def line(r):
    return (f"GSV1CASE {r['kind']} b={r['beta']:g} g={r['gamma']:g} k={r['k']} ratio={r['ratio']:g} N={r['N']} "
            f"{r['status']} steps={r['steps']} inc={r['inc']:.1e} err={r['err']:.3e} "
            f"at(wall={r['at_walldist']:g},xi={r['at_xi']:.2f}) {r['secs']}s")


def cmd_run(a):
    kinds, ratios = SETS[a.set]
    Ns = [int(v) for v in a.N.split(",")]
    done = load(a.out)
    keys = [(kd, b, g, k, r, n) for n in Ns for r in ratios for kd in kinds
            for b in ANGLES for g in ANGLES for k in range(8)]
    todo = [kk for kk in keys if kk not in done]
    print(f"GSV1RUN set={a.set} N={a.N} total={len(keys)} todo={len(todo)}", flush=True)
    with Pool(a.procs) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(run_case, todo, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(line(r), flush=True)
    print("EXIT", flush=True)


def lsorder(Ns, es):
    x, y = np.log(1.0 / np.array(Ns, float)), np.log(np.maximum(np.array(es, float), 1e-300))
    return float(np.polyfit(x, y, 1)[0])


def cmd_table(a):
    res = load(a.out)
    for name, (kinds, ratios) in SETS.items():
        for kd in kinds:
            exact_thr = {"couette": 1e-10, "extension": 1e-9}.get(kd)
            for r in ratios:
                for b, g in itertools.product(ANGLES, ANGLES):
                    rows = {k: sorted((n, res[(kd, b, g, k, r, n)]) for n in (24, 32, 48, 64)
                                      if (kd, b, g, k, r, n) in res) for k in range(8)}
                    if not any(rows.values()):
                        continue
                    Ns = sorted({n for v in rows.values() for n, _ in v})
                    bad = [rr["status"] for v in rows.values() for _, rr in v if rr["status"] != "ok"]
                    full = Ns == [24, 32, 48, 64] and all(len(v) == len(Ns) for v in rows.values())
                    orders, exact_all = [], True
                    for v in rows.values():
                        es = [rr["err"] for _, rr in v]
                        exact_all &= exact_thr is not None and all(e <= exact_thr for e in es)
                        if len(v) >= 2:
                            orders.append(lsorder([n for n, _ in v], es))
                    env = [max(res[(kd, b, g, k, r, n)]["err"] for k in range(8) if (kd, b, g, k, r, n) in res)
                           for n in Ns]
                    envo = lsorder(Ns, env) if len(Ns) >= 2 else float("nan")
                    if exact_thr is not None and kd == "extension":
                        ok = exact_all
                    elif exact_thr is not None:
                        ok = exact_all or (len(Ns) >= 2 and envo >= 1.8)  # USER 2026-10-31: envelope order
                    else:
                        ok = len(Ns) >= 2 and envo >= 1.8  # USER 2026-10-31 (Q13): envelope order binds
                    verdict = ("PASS" if ok else "MISS") if (full and not bad) else "NOT_JUDGED(partial)"
                    if bad:
                        verdict += f" status={','.join(sorted(set(bad)))}"
                    print(f"{name.upper()} {kd} ratio={r:g} b={b:g} g={g:g} N={','.join(map(str, Ns))} "
                          f"{'exact' if exact_all else 'not-exact'} min_offset_order="
                          f"{min(orders) if orders else float('nan'):.2f} env={','.join(f'{e:.2e}' for e in env)} "
                          f"env_order={envo:.2f} envN2={','.join(f'{e * n * n:.3g}' for e, n in zip(env, Ns))} "
                          f"{verdict}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--set", choices=tuple(SETS), required=True)
    r.add_argument("--N", default="24,32")
    r.add_argument("--procs", type=int, default=4)
    r.add_argument("--out", required=True)
    t = sp.add_parser("table")
    t.add_argument("--out", required=True)
    o = sp.add_parser("one")
    o.add_argument("case", nargs=6)
    a = p.parse_args()
    if a.cmd == "one":
        kd, b, g, k, r, n = a.case
        print(json.dumps(run_case((kd, float(b), float(g), int(k), float(r), int(n)))), flush=True)
        return
    cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    main()
