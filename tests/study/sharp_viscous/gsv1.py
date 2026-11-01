"""G-SV1 family (doc/vof_sharp_viscous_stress.md §11, third pass §6.3/§6.4): the 3-D sharp viscous
operator on planar-interface Stokes flows, through the production stepper; steady states by WO-6s.

STEADY STATE (WO-6s, §6.4). The colour is frozen (set_vof(C0) after every step), so the step is an
affine map S(x) = G x + b on x = (u, v, w, p), p through get_field/set_field("p") in internal units
with its mean removed in and out. GMRES(60) solves (I - G) x = b with the matvec v -> v - S(v) + S(0)
(one production step; S(0) once), started from the exact face averages with p = 0. Every restart
cycle begins with a PLAIN production step: its increment S(x) - x is the residual and the
certificate, ||S(x) - x||_inf <= 1e-12 max|u| over u (the pre-registered criterion); cap 3000
steps, a case at the cap is NOT CONVERGED. Pressure: the production driver (Chebyshev under
variable rho, MG-PCG under uniform rho at np 1) with rtol 1e-12 and a cap of 20000 (as WO-6a).

PERIODIC GATES (USER Q16 pending: they replace the walled box; WO-6p):
  gsv1p   G-SV1P layered flow: m in NORMALS, n = m/|m|, P = 1/|m|; phase A where
          frac((n.x - s0)/P) < 3/8, s0 = n.(1/2,1/2,1/2) + (k/8) h, k = 0..7; mu_A/mu_B = 1e-2, 1e2;
          rho_A/rho_B = 1/2; force (rho(C) - rho_bar) g t1 through the closures
          force_c = linear(rho, [-rho_bar g t1_c, g t1_c]), t1 = tangents(n)[0]; exact u = U(n.x) t1,
          U periodic piecewise quadratic (mu_k U'' = -(rho_k - rho_bar) g, U and mu U' continuous).
  gsv1dp  the same at mu ratio 1e-3, 1e3.
  gsv1cp  the along mode: phase A where s' mod P in (P/4, 3P/4), s' = n.x - s0; t1 = t/|t| from
          TANG_CP; u = -a sin(a eta) cos(b s') n + b cos(a eta) sin(b s') t1, a = 2 pi |t|,
          b = 2 pi |m|, eta = t1.x; p_k = 2 mu_k a b sin sin; force mu(C_i) F(x_i) as a cell field
          (set_field, internal units); kinds 'along' (rho = 1) and 'along+layered' (plus G-SV1P's
          layered flow on the same interfaces, rho_A/rho_B = 1/2). Face averages: 4x4 Gauss-Legendre
          for the smooth part, exact for the layered part.
  Gauge: per component the rho_F-weighted mean of the exact face averages is removed (the discrete
  momentum sum rho_F u_c is invariant, 0 from the gauged start). D_hi = 100. Error e = max|u_h -
  u_bar| / max|u_bar|. PASS per normal x ratio (x kind): the LS order of the offset envelope
  e_env(N) over N = 24, 32, 48, 64 is >= 1.8; e_env N^2 and the per-offset orders are reported.
  gsv1w   G-SV1W: walled Couette 22.5/22.5, offset 0, mu ratio 1e-2, N 24/32, option-A BCs with
          wall-face-AVERAGED Dirichlet data (tangential components averaged over the h-square of the
          wall plane), both modes; PASS iff at both N the 'sharp' max error over faces within one
          cell of a wall <= the 'averaged' one.
  pilot   the WO-6s pilot (cmd_prun).
  check   WO-6p acceptance: equations and jump conditions at 100 random points, the layered
          profile's closure, the exact face averages against brute-force quadrature.

  nice -n 10 python gsv1.py run --set gsv1p --N 24,32,48,64 --mode sharp --procs 8 --out res.jsonl
  python gsv1.py table --out res.jsonl ;  python gsv1.py check

WALLED (the pre-registered box; a reported diagnostic since the third pass).
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
face averages; the steady state by WO-6s (GMRES, below) to a step increment
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
    s.set_pressure_pcg(True, PCAP, PRTOL)  # uniform rho: the single-rank default driver
    s.diagnostics.set_vof_viscous_interface("sharp")
    for comp in range(3):
        s.set_velocity(comp, np.asfortranarray(ex[comp]))
    U = float(np.max(np.abs(ex)))
    Smap = StepMap(s, C, N)  # the steady state by WO-6s (GMRES on the fixed-point residual)
    x0 = np.concatenate([ex[c].ravel(order="F") for c in range(3)] + [np.zeros(N**3)])
    xs, status, it, inc, _ = gmres_steady(Smap, x0, 3 * N**3, U)
    prev = np.stack([xs[c * N**3:(c + 1) * N**3].reshape((N, N, N), order="F") for c in range(3)])
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


# ---------------------------------------------------------------------------------------------
# Periodic gates (note §6.3, §11 G-SV1P / G-SV1cP / G-SV1dP; WO-6p)
# ---------------------------------------------------------------------------------------------
NORMALS = ((1, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 2), (1, 0, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 1, 1))
TANG_CP = {(1, 0, 0): (0, 1, 0), (2, 1, 0): (-1, 2, 0), (1, 1, 0): (-1, 1, 0), (0, 1, 2): (0, -2, 1),
           (1, 0, 1): (-1, 0, 1), (2, 1, 1): (-1, 2, 0), (1, 2, 1): (2, -1, 0), (1, 1, 2): (1, -1, 0),
           (1, 1, 1): (1, -1, 0)}
PSETS = {  # name: (kinds, mu ratios)
    "gsv1p": (("layered",), (1e-2, 1e2)),
    "gsv1dp": (("layered",), (1e-3, 1e3)),
    "gsv1cp": (("along", "along+layered"), (1e-2, 1e2)),
}
RHO_A_LAYERED = 0.5  # rho_A / rho_B = 1/2, rho_B = 1 (G-SV1P)
PCAP, PRTOL = 20000, 1e-12  # the production pressure driver's cap (never binds) and rtol (as WO-6a)
GM_RESTART, GM_CAP = 60, 3000  # WO-6s defaults (§12 Q19)
Pl = np.polynomial.Polynomial


class Layered:
    """Periodic layered shear profile U(r), r = (xi - o) mod P, phase A on [0, a), B on [a, P):
    mu_k U'' = -q_k, U and mu U' continuous (periodic), mean of U over a period removed. Also V,
    W0: V(r) = int_0^r U (periodic), W0(r) = int_0^r (V - Vbar) (periodic), for exact face
    averages by double differences (the linear part Vbar*xi of the second antiderivative cancels)."""

    def __init__(self, P, a, o, muA, muB, qA, qB):
        self.P, self.a, self.o, self.mu = P, a, o, (muA, muB)

        def build(t0):
            tA = Pl([t0, -qA])
            UA = (tA / muA).integ()
            tB = Pl([tA(a) + qB * a, -qB])  # tau_B(r) = tau(a) - qB (r - a)
            UB = (tB / muB).integ()
            return UA, UB + (UA(a) - UB(a))

        _, B0 = build(0.0)
        _, B1 = build(1.0)
        t0 = -B0(P) / (B1(P) - B0(P))  # U(P) = U(0) = 0
        UA, UB = build(t0)
        mean = ((UA.integ()(a) - UA.integ()(0.0)) + (UB.integ()(P) - UB.integ()(a))) / P
        UA, UB = UA - mean, UB - mean
        VA = UA.integ()
        VB = UB.integ()
        VB = VB + (VA(a) - VB(a))
        Vbar = (VA.integ()(a) + VB.integ()(P) - VB.integ()(a)) / P
        WA = (VA - Vbar).integ()
        WB = (VB - Vbar).integ()
        WB = WB + (WA(a) - WB(a))
        self.U, self.V, self.W = (UA, UB), (VA - 0.0, VB), (WA, WB)
        self.closure = (abs(VB(P)), abs(WB(P)))  # both 0 to round-off (periodic V, W0)

    def _ev(self, pair, xi):
        r = np.mod(np.asarray(xi, float) - self.o, self.P)
        return np.where(r < self.a, pair[0](r), pair[1](r))

    def Uf(self, xi):
        return self._ev(self.U, xi)

    def avg(self, xi, wa, wb):
        """E[U(xi + Y + Z)], Y, Z centred uniforms of widths |wa|, |wb| (exact)."""
        a, b = abs(wa), abs(wb)
        if a < b:
            a, b = b, a
        if a < 1e-14:
            return self.Uf(xi)
        if b < 1e-14:
            return (self._ev(self.V, xi + a / 2) - self._ev(self.V, xi - a / 2)) / a
        W = lambda z: self._ev(self.W, z)
        return (W(xi + (a + b) / 2) - W(xi + (a - b) / 2) - W(xi - (a - b) / 2) + W(xi - (a + b) / 2)) / (a * b)


def layered_fractions(n, s0, P, a, o, N):
    """Exact cell volume fraction of {(n.x - s0 - o) mod P < a} on the N^3 unit grid."""
    h = 1.0 / N
    lo, hi = h * sum(min(v, 0.0) for v in n), h * sum(max(v, 0.0) for v in n)
    C = np.zeros((N, N, N))
    for i, j, k in itertools.product(range(N), repeat=3):
        x0 = np.array([i, j, k], float) * h
        base = float(n @ x0) - s0
        F = lambda c: cube_frac(n, (c - base) / h)
        tot = 0.0
        for q in range(int(math.floor((base + lo - o) / P)) - 1, int(math.ceil((base + hi - o) / P)) + 1):
            c0 = q * P + o
            tot += F(c0 + a) - F(c0)
        C[i, j, k] = min(1.0, max(0.0, tot))
    return C


def face_centres(N, c):
    h = 1.0 / N
    g = (np.arange(N) + 0.5) * h
    axes = [g.copy(), g.copy(), g.copy()]
    axes[c] = np.arange(N) * h
    return np.meshgrid(*axes, indexing="ij")


def rho_face(rho):
    """The solver's face density (face_props.hpp idiag): the mean of the two cells, periodic."""
    return [0.5 * (rho + np.roll(rho, 1, axis=c)) for c in range(3)]


class PCase:
    """One periodic case: kind in layered | along | along+layered; m the integer normal."""

    def __init__(self, kind, m, k, ratio, N):
        self.kind, self.m, self.k, self.ratio, self.N = kind, tuple(m), k, ratio, N
        mv = np.array(m, float)
        self.n = mv / np.linalg.norm(mv)
        self.P = 1.0 / np.linalg.norm(mv)
        self.h = 1.0 / N
        self.s0 = float(self.n @ np.full(3, 0.5)) + k / 8.0 * self.h
        self.muA, self.muB = ratio, 1.0
        if kind == "layered":
            self.a, self.o = 3.0 * self.P / 8.0, 0.0
            self.rhoA = RHO_A_LAYERED
        else:  # phase A where s' mod P in (P/4, 3P/4): cos(b s') = 0 on the interfaces
            self.a, self.o = self.P / 2.0, self.P / 4.0
            self.rhoA = RHO_A_LAYERED if kind == "along+layered" else 1.0
        self.rhoB = 1.0
        self.t1 = tangents(self.n)[0]  # G-SV1P's layered tangent
        t = np.array(TANG_CP[self.m], float)
        self.tc = t / np.linalg.norm(t)  # G-SV1cP's along-mode tangent
        self.ka, self.kb = 2 * math.pi * np.linalg.norm(t), 2 * math.pi * np.linalg.norm(mv)

    # --- the along mode (G-SV1cP), analytic in (eta, s'): u = A n + B t1
    def along_u(self, X):
        """X: (..., 3) points -> (..., 3) velocity."""
        eta, sp = X @ self.tc, X @ self.n - self.s0
        a, b = self.ka, self.kb
        A = -a * np.sin(a * eta) * np.cos(b * sp)
        B = b * np.cos(a * eta) * np.sin(b * sp)
        return A[..., None] * self.n + B[..., None] * self.tc

    def along_F(self, X):
        """The force divided by the phase viscosity: f_k = mu_k F."""
        eta, sp = X @ self.tc, X @ self.n - self.s0
        a, b = self.ka, self.kb
        Fn = a * (b * b - a * a) * np.sin(a * eta) * np.cos(b * sp)
        Ft = b * (3 * a * a + b * b) * np.cos(a * eta) * np.sin(b * sp)
        return Fn[..., None] * self.n + Ft[..., None] * self.tc

    def along_p(self, X, mu):
        eta, sp = X @ self.tc, X @ self.n - self.s0
        return 2.0 * mu * self.ka * self.kb * np.sin(self.ka * eta) * np.sin(self.kb * sp)

    def phaseA(self, X):
        r = np.mod(X @ self.n - self.s0 - self.o, self.P)
        return r < self.a

    def build(self):
        """C, the exact face averages (3, N, N, N) before the gauge, the cell force F_cell (3,
        N,N,N) with g = 1 (layered part) or unit amplitude (along part), and the Layered."""
        N, h = self.N, self.h
        C = layered_fractions(self.n, self.s0, self.P, self.a, self.o, N)
        rho = self.rhoB + (self.rhoA - self.rhoB) * C
        rbar = float(np.mean(rho))
        ex = np.zeros((3, N, N, N))
        force = np.zeros((3, N, N, N))
        lay = None
        if self.kind in ("layered", "along+layered"):
            phi = self.a / self.P
            rbar_c = phi * self.rhoA + (1 - phi) * self.rhoB
            lay = Layered(self.P, self.a, self.o, self.muA, self.muB, self.rhoA - rbar_c, self.rhoB - rbar_c)
            for c in range(3):
                X = face_centres(N, c)
                xi = self.n[0] * X[0] + self.n[1] * X[1] + self.n[2] * X[2] - self.s0
                w = [self.n[b] * h for b in range(3) if b != c]
                ex[c] += self.t1[c] * lay.avg(xi, w[0], w[1])
            force += ((rho - rbar)[None] * self.t1[:, None, None, None])
        if self.kind in ("along", "along+layered"):
            gx, gw = np.polynomial.legendre.leggauss(4)
            for c in range(3):
                X = np.stack(face_centres(N, c), axis=-1)
                bb = [b for b in range(3) if b != c]
                acc = np.zeros((N, N, N, 3))
                for (p, wp), (q, wq) in itertools.product(zip(gx, gw), zip(gx, gw)):
                    Y = X.copy()
                    Y[..., bb[0]] += 0.5 * h * p
                    Y[..., bb[1]] += 0.5 * h * q
                    acc += 0.25 * wp * wq * self.along_u(Y)
                ex[c] += acc[..., c]
            Xc = np.stack(np.meshgrid(*([(np.arange(N) + 0.5) * h] * 3), indexing="ij"), axis=-1)
            muC = self.muB + (self.muA - self.muB) * C
            force += np.moveaxis(muC[..., None] * self.along_F(Xc), -1, 0)
        return C, rho, rbar, ex, force, lay

    def gauge(self, ex, rho):
        rf = rho_face(rho)
        return np.stack([ex[c] - float(np.sum(rf[c] * ex[c]) / np.sum(rf[c])) for c in range(3)])


# ---------------------------------------------------------------------------------------------
# The steady driver (WO-6s, note §6.4): GMRES(60) on the production step's fixed-point residual
# ---------------------------------------------------------------------------------------------
class StepMap:
    """S: x = (u, v, w, p) -> one production step, colour frozen (set_vof(C0) after the step); p
    through get_field/set_field("p") in internal units, its mean removed on input and output."""

    def __init__(self, s, C0, N, rho=None):
        self.s, self.C0, self.shp, self.n = s, np.asfortranarray(C0), (N, N, N), N**3
        self.steps, self.max_pit, self.max_msw, self.mom, self.max_mres = 0, 0, 0, 0.0, -1.0
        self.rf = rho_face(rho) if rho is not None else None
        self.rfs = [float(np.sum(f)) for f in self.rf] if self.rf is not None else None

    def gauge(self, x):
        """Remove each velocity component's rho_F-weighted mean (periodic box only) and p's mean."""
        if self.rf is None:
            return x
        y, n = x.copy(), self.n
        for c in range(3):
            a = y[c * n:(c + 1) * n].reshape(self.shp, order="F")
            a -= float(np.sum(self.rf[c] * a)) / self.rfs[c]
            y[c * n:(c + 1) * n] = a.ravel(order="F")
        y[3 * n:] -= y[3 * n:].mean()
        return y

    def __call__(self, x, track=False):
        s, n = self.s, self.n
        for c in range(3):
            s.set_velocity(c, np.asfortranarray(x[c * n:(c + 1) * n].reshape(self.shp, order="F")))
        p = x[3 * n:]
        s.set_field("p", np.asfortranarray((p - p.mean()).reshape(self.shp, order="F")))
        s.step()
        s.set_vof(self.C0)
        self.steps += 1
        dg = s.diagnostics
        self.max_pit = max(self.max_pit, int(dg.last_pressure_iterations()))
        self.max_msw = max(self.max_msw, int(dg.last_step_timers()["momentum_sweeps"]))
        self.max_mres = max(self.max_mres, float(dg.last_momentum_residual()))
        U = [np.asarray(a) for a in (s.get_u(), s.get_v(), s.get_w())]
        if track and self.rf is not None:  # sum_faces rho_F u_c over sum rho_F max|u| (WO-6p)
            um = max(float(np.max(np.abs(a))) for a in U)
            for c in range(3):
                self.mom = max(self.mom, abs(float(np.sum(self.rf[c] * U[c]))) / (float(np.sum(self.rf[c])) * um))
        P = np.asarray(s.get_field("p")).ravel(order="F")
        return np.concatenate([a.ravel(order="F") for a in U] + [P - P.mean()])


def gmres_steady(S, x0, nu, umax, m=GM_RESTART, cap=GM_CAP, tol=TOL):
    """Restarted GMRES(m) on (I - G) x = b, G v = S(v) - S(0). Every cycle starts with a PLAIN
    production step: r = S(x) - x is both the residual and the certificate (increment of u <=
    tol * max|u|). Returns (state after the certifying step, status, steps, increment, cycles).
    Periodic box: I - G is singular on the uniform-velocity modes (the momentum invariant), and the
    solver's round-off-level momentum error would put a component there that GMRES amplifies by
    1/lambda; the Krylov vectors are therefore kept in the gauge subspace (rho_F-weighted mean of
    each velocity component removed), as p is kept mean-free. The certificate is the unprojected
    plain-step increment."""
    gauge = S.gauge
    S0 = S(np.zeros_like(x0))
    x, cycles = x0.copy(), 0
    while True:
        Sx = S(x, track=True)
        r = Sx - x
        inc = float(np.max(np.abs(r[:nu]))) / max(float(np.max(np.abs(x[:nu]))), 1e-300)
        if not math.isfinite(inc):
            return Sx, "blowup", S.steps, inc, cycles
        if inc <= tol:
            return Sx, "ok", S.steps, inc, cycles
        if S.steps >= cap:
            return Sx, "not_converged", S.steps, inc, cycles
        r = gauge(r)
        beta = float(np.linalg.norm(r))
        V = [r / beta]
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        kk = 0
        for j in range(m):
            if S.steps >= cap:
                break
            v = V[j]
            sc = umax / max(float(np.max(np.abs(v[:nu]))), float(np.max(np.abs(v))), 1e-300)
            w = gauge(v - (S(sc * v) - S0) / sc)
            for _ in range(2):  # MGS, twice
                for i in range(j + 1):
                    hij = float(V[i] @ w)
                    H[i, j] += hij
                    w = w - hij * V[i]
            H[j + 1, j] = float(np.linalg.norm(w))
            for i in range(j):  # apply the previous rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            d = math.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = H[j, j] / d, H[j + 1, j] / d
            hn = H[j + 1, j]
            H[j, j], H[j + 1, j] = d, 0.0
            g[j + 1], g[j] = -sn[j] * g[j], cs[j] * g[j]
            kk = j + 1
            if hn <= 1e-300 or abs(g[j + 1]) <= 0.1 * tol * umax:
                break
            V.append(w / hn)
        if kk == 0:
            continue
        y = np.linalg.solve(np.triu(H[:kk, :kk]), g[:kk])
        x = x + sum(yi * V[i] for i, yi in enumerate(y))
        cycles += 1


def plain_march(S, x0, nu, cap=MAX_STEPS, tol=TOL):
    x = x0.copy()
    while True:
        Sx = S(x, track=True)
        inc = float(np.max(np.abs((Sx - x)[:nu]))) / max(float(np.max(np.abs(x[:nu]))), 1e-300)
        if not math.isfinite(inc):
            return Sx, "blowup", S.steps, inc
        if inc <= tol:
            return Sx, "ok", S.steps, inc
        if S.steps >= cap:
            return Sx, "max_steps", S.steps, inc
        x = Sx


def run_pcase(key, steady="gmres", want_state=False, converged_momentum=False):
    pset, kind, m, k, ratio, N, mode = key
    os.environ.setdefault("OMP_PROC_BIND", "false")
    import peclet.flow as pf

    t0 = time.time()
    cs = PCase(kind, m, k, ratio, N)
    h = cs.h
    C, rho, rbar, ex, force, lay = cs.build()
    hi_A = cs.muA > cs.muB
    mu_hi, rho_hi = (cs.muA, cs.rhoA) if hi_A else (cs.muB, cs.rhoB)
    dt = D_HI * h * h * rho_hi / mu_hi
    amp = 0.01 * h / (dt * float(np.max(np.abs(ex))))  # colour CFL <= 0.01; the problem is linear
    ex, force = amp * ex, amp * force
    exg = cs.gauge(ex, rho)
    s = pf.Solver((N, N, N), extent=(1.0, 1.0, 1.0))
    s.set_dt(dt)  # first: pins the internal time scale
    s.set_rho(1.0)  # first: rho_ref = 1, so the internal rho field is the physical one
    s.set_mu(1.0)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(np.asfortranarray(C))
    s.set_property_model("mu", "linear", "C", [cs.muB, cs.muA - cs.muB])
    varrho = cs.rhoA != cs.rhoB
    if varrho:
        s.set_property_model("rho", "linear", "C", [cs.rhoB, cs.rhoA - cs.rhoB])
    s.enable_cell_force()
    if kind == "layered":  # through the closures (§11 G-SV1P): force_c = linear(rho, [-rbar g t_c, g t_c])
        for c in range(3):
            s.set_property_model("force_" + "xyz"[c], "linear", "rho",
                                 [-rbar * amp * cs.t1[c], amp * cs.t1[c]])
    else:  # the cell field mu(C_i) F(x_i) (+ the layered part), internal units
        fi = s.unit_scales["force_density_to_internal"]
        for c in range(3):
            s.set_field("force_" + "xyz"[c], np.asfortranarray(np.atleast_1d(fi)[c if np.size(fi) == 3 else 0] * force[c]))
    s.set_advection(False)
    if varrho:  # keep the driver the rho model selected (Chebyshev); set its rtol and a cap that never binds
        s.set_pressure_chebyshev(True, PCAP, PRTOL)
    else:  # uniform rho: the single-rank default driver, MG-PCG
        s.set_pressure_pcg(True, PCAP, PRTOL)
    s.diagnostics.set_vof_viscous_interface(mode)
    if converged_momentum:  # DIAGNOSTIC only: the RB-GS update stop 1e-15 with the residual stop off
        s.diagnostics.set_velocity_solver_params(200000, 1e-15, 2)
        s.set_velocity_residual_tolerance(0.0)
    s.update_properties()
    fsum = max(abs(float(np.sum(np.asarray(s.get_field("force_" + a))))) for a in "xyz")
    fmax = max(float(np.max(np.abs(np.asarray(s.get_field("force_" + a))))) for a in "xyz")
    S = StepMap(s, C, N, rho)
    nu = 3 * N**3
    x0 = np.concatenate([exg[c].ravel(order="F") for c in range(3)] + [np.zeros(N**3)])
    U = float(np.max(np.abs(exg)))
    if steady == "gmres":
        xs, status, steps, inc, cycles = gmres_steady(S, x0, nu, U)
    else:
        xs, status, steps, inc = plain_march(S, x0, nu)
        cycles = 0
    uh = np.stack([xs[c * N**3:(c + 1) * N**3].reshape((N, N, N), order="F") for c in range(3)])
    e = np.abs(uh - exg) / U
    uhg = cs.gauge(uh, rho)  # reported beside: u_h with its own rho_F-weighted mean removed
    err_g = float(np.max(np.abs(uhg - exg))) / U
    am = np.unravel_index(int(np.argmax(e)), e.shape)
    Xf = face_centres(N, am[0])
    rr = float(np.mod(sum(cs.n[q] * Xf[q][am[1:]] for q in range(3)) - cs.s0 - cs.o, cs.P))
    dist = min(rr, abs(rr - cs.a), cs.P - rr) / h  # distance of the max-error face to an interface
    out = {"set": pset, "kind": kind, "m": list(m), "k": k, "ratio": ratio, "N": N, "mode": mode,
           "steady": steady, "status": status, "steps": steps, "cycles": cycles, "inc": inc,
           "err": float(np.max(e)), "err_gauged": err_g, "at_comp": int(am[0]), "at_cell": [int(v) for v in am[1:]],
           "at_xi": float(dist), "mom": S.mom, "force_sum_rel": fsum / max(fmax, 1e-300),
           "max_p_it": S.max_pit, "max_momentum_sweeps": S.max_msw, "secs": round(time.time() - t0, 1),
           "max_momentum_residual": S.max_mres, "converged_momentum": converged_momentum}
    if want_state:
        return out, uh
    return out


def run_wcase(key):
    """G-SV1W (§11): the walled Couette contact check, option-A BCs with wall-face-AVERAGED data
    (each tangential component's Dirichlet value is its average over the h-square of the wall plane
    centred at its node), steady state by WO-6s, error binned by wall distance."""
    pset, kind, beta, gamma, k, ratio, N, mode = key
    os.environ.setdefault("OMP_PROC_BIND", "false")
    import peclet.flow as pf

    t0 = time.time()
    cs = Case(kind, beta, gamma, k, ratio, N)
    h = cs.h
    C = np.zeros((N, N, N))
    ex = np.zeros((3, N, N, N))
    for i, j, kk in itertools.product(range(N), repeat=3):
        xc = np.array([i, j, kk], float) * h
        C[i, j, kk] = cube_frac(cs.n, (cs.s - cs.n @ (xc - cs.c)) / h)
        for comp in range(3):
            x = (np.array([i, j, kk], float) + 0.5) * h
            x[comp] -= 0.5 * h
            ex[comp, i, j, kk] = cs.face_avg(comp, x, tuple(a for a in range(3) if a != comp))
    dt = D_HI * h * h / max(cs.muA, cs.muB)
    amp = 0.01 * h / (dt * float(np.max(np.abs(ex))))
    ex *= amp
    s = pf.Solver((N, N, N), extent=(1.0, 1.0, 1.0))
    s.set_dt(dt)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
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
                    prof[ib, ic, b] = cs.face_avg(b, xb, (b, c))  # wall-face average (G-SV1W)
                    xc = x.copy()
                    xc[c] = ic * h
                    prof[ib, ic, c] = cs.face_avg(c, xc, (b, c))
            s.set_domain_bc_profile(("-" if side == 0 else "+") + "xyz"[a], np.ascontiguousarray(amp * prof))
    s.set_rho(1.0)
    s.set_mu(max(cs.muA, cs.muB))
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(np.asfortranarray(C))
    s.set_property_model("mu", "linear", "C", [cs.muB, cs.muA - cs.muB])
    s.set_advection(False)
    s.set_pressure_pcg(True, PCAP, PRTOL)  # uniform rho: the single-rank default driver
    s.diagnostics.set_vof_viscous_interface(mode)
    S = StepMap(s, C, N)
    nu = 3 * N**3
    x0 = np.concatenate([ex[c].ravel(order="F") for c in range(3)] + [np.zeros(N**3)])
    U = float(np.max(np.abs(ex)))
    xs, status, steps, inc, cycles = gmres_steady(S, x0, nu, U)
    uh = np.stack([xs[c * N**3:(c + 1) * N**3].reshape((N, N, N), order="F") for c in range(3)])
    e = np.abs(uh - ex) / U
    wd = wall_dist(N)
    return {"set": pset, "kind": kind, "beta": beta, "gamma": gamma, "k": k, "ratio": ratio, "N": N,
            "mode": mode, "status": status, "steps": steps, "cycles": cycles, "inc": inc, "err": float(np.max(e)),
            "err_wall1": float(np.max(e[wd <= 1.0])), "err_interior": float(np.max(e[wd > 1.0])),
            "max_p_it": S.max_pit, "secs": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------
def pkey(r):
    if r["set"] == "gsv1w":
        return (r["set"], r["kind"], r["beta"], r["gamma"], r["k"], r["ratio"], r["N"], r["mode"])
    return (r["set"], r["kind"], tuple(r["m"]), r["k"], r["ratio"], r["N"], r["mode"], r.get("steady", "gmres"))


def pload(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    out[pkey(r)] = r
    return out


def pline(r):
    geo = (f"b={r['beta']:g} g={r['gamma']:g}" if r["set"] == "gsv1w" else f"m={''.join(map(str, r['m']))}")
    extra = (f"err_wall1={r['err_wall1']:.3e} err_int={r['err_interior']:.3e}" if r["set"] == "gsv1w" else
             f"mom={r['mom']:.1e} fsum={r['force_sum_rel']:.1e} at_xi/h={r['at_xi']:.2f}")
    return (f"GSV1CASE {r['set']} {r['kind']} {geo} k={r['k']} ratio={r['ratio']:g} N={r['N']} {r['mode']} "
            f"{r.get('steady', 'gmres')} {r['status']} steps={r['steps']} cycles={r['cycles']} inc={r['inc']:.1e} "
            f"err={r['err']:.3e} err_g={r.get('err_gauged', float('nan')):.3e} {extra} max_p_it={r['max_p_it']} max_mres={r.get('max_momentum_residual', -1):.1e} {r['secs']}s")


def _run_any(key):
    if key[0] == "gsv1w":
        return run_wcase(key)
    if key[7] != "march":
        try:
            return run_pcase(key[:7], steady=key[7], converged_momentum=CONV_MOM)
        except RuntimeError as ex:  # a production step refused the iterate (e.g. the colour-CFL guard):
            # the case is NOT CONVERGED (as a capped case), recorded and listed, never judged
            nan = float("nan")
            pset, kind, m, k, ratio, N, mode = key[:7]
            return {"set": pset, "kind": kind, "m": list(m), "k": k, "ratio": ratio, "N": N, "mode": mode,
                    "steady": key[7], "status": "step_error", "error": str(ex)[:200], "steps": -1, "cycles": -1,
                    "inc": nan, "err": nan, "err_gauged": nan, "at_xi": nan, "mom": nan, "force_sum_rel": nan,
                    "max_p_it": -1, "max_momentum_residual": nan, "secs": 0.0, "converged_momentum": CONV_MOM}
    rg, ug = run_pcase(key[:7], steady="gmres", want_state=True, converged_momentum=CONV_MOM)
    rm, um = run_pcase(key[:7], steady="march", want_state=True, converged_momentum=CONV_MOM)
    cs = PCase(*key[1:6])
    C, rho, _, ex, _, _ = cs.build()
    U = float(np.max(np.abs(ug)))
    rm["diff_vs_gmres"] = [float(np.max(np.abs(um - ug))) / U,
                           float(np.max(np.abs(cs.gauge(um, rho) - cs.gauge(ug, rho)))) / U]
    return rm


CONV_MOM = False  # run --converged-momentum: the WO-6p invariance instrument (coordinator F1, 2026-11-01)


def cmd_prun(a):
    global CONV_MOM
    CONV_MOM = bool(a.converged_momentum)
    Ns = [int(v) for v in a.N.split(",")]
    if a.set == "gsv1w":
        keys = [("gsv1w", "couette", 22.5, 22.5, 0, 1e-2, n, md) for n in Ns for md in ("sharp", "averaged")]
    elif a.set == "pilot":
        # WO-6s pilot (DECISION, reported): N 24/48 x both ratios x normals (1,0,0) (2,1,0) (2,1,1),
        # G-SV1P layered, offset k = 3; plain-march agreement on the two N = 24 cases below.
        keys = [("gsv1p", "layered", m, 3, r, n, a.mode, "gmres") for n in (24, 48) for r in (1e-2, 1e2)
                for m in ((1, 0, 0), (2, 1, 0), (2, 1, 1))]
        keys += [("gsv1p", "layered", (2, 1, 0), 3, 1e-2, 24, a.mode, "march"),
                 ("gsv1p", "layered", (2, 1, 1), 3, 1e2, 24, a.mode, "march")]
    else:
        kinds, ratios = PSETS[a.set]
        keys = [(a.set, kd, m, k, r, n, a.mode, "gmres") for n in Ns for r in ratios for kd in kinds
                for m in NORMALS for k in range(8)]
    done = pload(a.out)
    todo = [kk for kk in keys if kk not in done]
    print(f"GSV1RUN set={a.set} N={a.N} total={len(keys)} todo={len(todo)}", flush=True)
    with Pool(a.procs) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(_run_any, todo, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(pline(r), flush=True)
    print("EXIT", flush=True)


def cmd_ptable(a):
    res = pload(a.out)
    # pilot: matvec counts and the march agreement
    gm = [r for r in res.values() if r["set"] == "gsv1p" and r.get("steady") == "gmres"]
    for r in res.values():
        if r.get("steady") == "march":
            g = res.get(pkey(dict(r, steady="gmres")))
            if g is not None:
                d = r.get("diff_vs_gmres")
                print(f"GSV1-PILOT march-agreement m={''.join(map(str, r['m']))} ratio={r['ratio']:g} N={r['N']}: "
                      f"err gmres {g['err']:.6e} march {r['err']:.6e}; max|u_march - u_gmres|/max|u| raw "
                      f"{d[0] if d else float('nan'):.1e} gauged {d[1] if d else float('nan'):.1e} (<= 1e-11) "
                      f"(steps gmres {g['steps']} march {r['steps']} {r['status']})")
    if gm:
        st = sorted(r["steps"] for r in gm)
        print(f"GSV1-STEPS gmres cases={len(gm)} steps min/median/max={st[0]}/{st[len(st) // 2]}/{st[-1]} "
              f"not_ok={sum(1 for r in gm if r['status'] != 'ok')}")
    for name, (kinds, ratios) in PSETS.items():
        for kd in kinds:
            for r in ratios:
                for mode in ("sharp", "averaged"):
                    for m in NORMALS:
                        rows = {k: sorted((n, res[(name, kd, m, k, r, n, mode, "gmres")]) for n in (24, 32, 48, 64)
                                          if (name, kd, m, k, r, n, mode, "gmres") in res) for k in range(8)}
                        if not any(rows.values()):
                            continue
                        Ns = sorted({n for v in rows.values() for n, _ in v})
                        bad = [rr["status"] for v in rows.values() for _, rr in v if rr["status"] != "ok"]
                        full = Ns == [24, 32, 48, 64] and all(len(v) == 4 for v in rows.values())
                        orders = [lsorder([n for n, _ in v], [rr["err"] for _, rr in v]) for v in rows.values()
                                  if len(v) >= 2]
                        env = [max(rr["err"] for v in rows.values() for n2, rr in v if n2 == n) for n in Ns]
                        envo = lsorder(Ns, env) if len(Ns) >= 2 else float("nan")
                        verdict = ("PASS" if envo >= 1.8 else "MISS") if (full and not bad) else "NOT_JUDGED(partial)"
                        if bad:
                            verdict += f" status={','.join(sorted(set(bad)))}"
                        print(f"{name.upper()} {kd} ratio={r:g} m={''.join(map(str, m))} {mode} N={','.join(map(str, Ns))} "
                              f"min_offset_order={min(orders) if orders else float('nan'):.2f} "
                              f"env={','.join(f'{e:.2e}' for e in env)} env_order={envo:.2f} "
                              f"envN2={','.join(f'{e * n * n:.3g}' for e, n in zip(env, Ns))} {verdict}")
    w = {(r["N"], r["mode"]): r for r in res.values() if r["set"] == "gsv1w"}
    if w:
        Ns = sorted({n for n, _ in w})
        ok = all((n, "sharp") in w and (n, "averaged") in w and w[(n, "sharp")]["status"] == "ok"
                 and w[(n, "averaged")]["status"] == "ok" and w[(n, "sharp")]["err_wall1"] <= w[(n, "averaged")]["err_wall1"]
                 for n in Ns)
        print("GSV1W " + " ".join(f"N={n}: sharp {w[(n, 'sharp')]['err_wall1']:.3e} averaged "
                                   f"{w[(n, 'averaged')]['err_wall1']:.3e}" for n in Ns if (n, 'sharp') in w and (n, 'averaged') in w)
              + f" {'PASS' if ok and Ns == [24, 32] else ('MISS' if Ns == [24, 32] else 'NOT_JUDGED(partial)')}")


def cmd_check(a):
    """WO-6p acceptance: the manufactured fields satisfy the continuous equations and the jump
    conditions to <= 1e-12 at 100 random points; the exact face averages against brute-force
    quadrature on random faces; Layered's closure (periodic V, W0)."""
    rng = np.random.default_rng(1)
    worst = {"along_div": 0.0, "along_mom": 0.0, "along_jump": 0.0, "layered_ode": 0.0, "layered_jump": 0.0,
             "layered_closure": 0.0, "face_avg": 0.0}
    eps = 1e-30
    for m in NORMALS:
        for ratio in (1e-2, 1e2, 1e-3, 1e3):
            cs = PCase("along", m, 3, ratio, 24)
            a, b = cs.ka, cs.kb
            sc = b * (a * a + b * b) * max(1.0, ratio)  # the force scale
            for _ in range(100):
                X = rng.random(3)
                J = np.array([np.imag(cs.along_u(X + 1j * eps * np.eye(3)[q])) / eps for q in range(3)]).T  # J[i,q] = du_i/dx_q
                mu = cs.muA if cs.phaseA(X) else cs.muB
                gp = np.array([np.imag(cs.along_p(X + 1j * eps * np.eye(3)[q], mu)) / eps for q in range(3)])
                lap = -(a * a + b * b) * cs.along_u(X)  # each component is a plane-wave sum with |k|^2 = a^2 + b^2
                res = mu * lap - gp + mu * cs.along_F(X)
                worst["along_div"] = max(worst["along_div"], abs(np.trace(J)) / (a * b))
                worst["along_mom"] = max(worst["along_mom"], float(np.max(np.abs(res))) / sc)
                # a random interface point: s' = P/4 + j P/2; traction jump [sigma.n] = 0
                Y = X - (X @ cs.n - cs.s0 - cs.P / 4 - cs.P / 2 * rng.integers(0, 2)) * cs.n
                Jy = np.array([np.imag(cs.along_u(Y + 1j * eps * np.eye(3)[q])) / eps for q in range(3)]).T
                Dy = 0.5 * (Jy + Jy.T)
                tr = [(-cs.along_p(Y, mu_) * cs.n + 2 * mu_ * Dy @ cs.n) for mu_ in (cs.muA, cs.muB)]
                worst["along_jump"] = max(worst["along_jump"], float(np.max(np.abs(tr[0] - tr[1]))) / sc)
            for kd in ("layered", "along+layered"):
                cl = PCase(kd, m, 3, ratio, 24)
                phi = cl.a / cl.P
                rb = phi * cl.rhoA + (1 - phi) * cl.rhoB
                L = Layered(cl.P, cl.a, cl.o, cl.muA, cl.muB, cl.rhoA - rb, cl.rhoB - rb)
                usc = max(float(np.max(np.abs(L.Uf(np.linspace(0, cl.P, 257))))), 1e-300)
                qsc = max(abs(cl.rhoA - rb), abs(cl.rhoB - rb))
                for r_ in rng.random(50) * cl.P:
                    inA = r_ < cl.a
                    U2 = (L.U[0] if inA else L.U[1]).deriv(2)(r_)
                    mu_, q_ = (cl.muA, cl.rhoA - rb) if inA else (cl.muB, cl.rhoB - rb)
                    worst["layered_ode"] = max(worst["layered_ode"], abs(mu_ * U2 + q_) / qsc)
                UA, UB = L.U
                jumps = [abs(UA(cl.a) - UB(cl.a)) / usc, abs(UA(0.0) - UB(cl.P)) / usc,
                         abs(cl.muA * UA.deriv()(cl.a) - cl.muB * UB.deriv()(cl.a)) / (qsc * cl.P),
                         abs(cl.muA * UA.deriv()(0.0) - cl.muB * UB.deriv()(cl.P)) / (qsc * cl.P)]
                worst["layered_jump"] = max(worst["layered_jump"], max(jumps))
                worst["layered_closure"] = max(worst["layered_closure"], L.closure[0] / (usc * cl.P),
                                               L.closure[1] / (usc * cl.P * cl.P))
                # face averages vs brute force (composite Gauss on a split-free fine grid of 4096^2)
                h = cl.h
                for _ in range(3):
                    xi0 = rng.random() * cl.P
                    wa, wb = cl.n[0] * h, cl.n[1] * h
                    ex_ = float(L.avg(np.array(xi0), wa, wb))
                    gx, gw = np.polynomial.legendre.leggauss(8)
                    M = 512
                    yy = ((np.arange(M)[:, None] + 0.5 * (gx[None, :] + 1)) / M - 0.5).ravel()
                    ww = (np.tile(gw, M) / (2 * M))
                    Y_, Z_ = np.meshgrid(yy * wa, yy * wb, indexing="ij")
                    bf = float(np.sum(np.outer(ww, ww) * L.Uf(xi0 + Y_ + Z_)))
                    worst["face_avg"] = max(worst["face_avg"], abs(ex_ - bf) / usc)
    ok = all(v <= 1e-12 for kk, v in worst.items() if kk != "face_avg") and worst["face_avg"] <= 1e-8
    print("GSV1P-CHECK " + " ".join(f"{kk}={v:.1e}" for kk, v in worst.items())
          + f" {'PASS' if ok else 'MISS'} (equations/jumps <= 1e-12; face averages vs brute-force quadrature <= 1e-8,"
            " the quadrature's own error at a kink)")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--set", choices=tuple(SETS) + tuple(PSETS) + ("gsv1w", "pilot"), required=True)
    r.add_argument("--N", default="24,32")
    r.add_argument("--procs", type=int, default=4)
    r.add_argument("--mode", choices=("sharp", "averaged"), default="sharp")
    r.add_argument("--out", required=True)
    # WO-6p's momentum invariant assumes an exact solve: RB-GS update stop 1e-15, residual stop off
    # (coordinator F1). The steady-state gates keep the production default (omit the flag).
    r.add_argument("--converged-momentum", action="store_true")
    t = sp.add_parser("table")
    t.add_argument("--out", required=True)
    sp.add_parser("check")
    o = sp.add_parser("one")
    o.add_argument("case", nargs=6)
    a = p.parse_args()
    if a.cmd == "one":
        kd, b, g, k, r, n = a.case
        print(json.dumps(run_case((kd, float(b), float(g), int(k), float(r), int(n)))), flush=True)
        return
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "run":
        return cmd_run(a) if a.set in SETS else cmd_prun(a)
    res = pload(a.out)
    if any(k[0] in PSETS or k[0] in ("gsv1w", "gsv1p") for k in res):
        cmd_ptable(a)
    else:
        cmd_table(a)


if __name__ == "__main__":
    main()
