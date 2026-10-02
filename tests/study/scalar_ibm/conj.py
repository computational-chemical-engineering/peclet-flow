"""2-D prototype, round 4: CONJUGATE transport with the probe-flux cut-cell FV.

Geometry: solid core r < a (diffusivity Ds, capacity gs), fluid annulus a < r < R (Df = 1, capacity
1), u = 0 at r = R (immersed Dirichlet, probe-flux).  Interface at r = a:
    flux continuity   q = -Ds du_s/dr = -Df du_f/dr   (q: radial, solid -> fluid)
    q = hc (u_s/K - u_f)          (hc = inf  ->  u_s = K u_f: partition coefficient K)
Exact: slowest mode of  g du/dt = div(D grad u)  via a Bessel determinant.

Discretization (two fields on one grid; every cell with fluid fraction > 0 has a fluid unknown,
every cell with solid-core fraction > 0 has a solid unknown):
  faces   : aperture-weighted two-point fluxes, separately per phase (fluid apertures / core apertures)
  facet   : per cut cell, wall area A_w (aperture area vector), centroid x_w (radial projection),
            normal n (into the fluid). Fluid probe x_w + c h n (bilinear over fluid unknowns), solid
            probe x_w - c h n (bilinear over solid unknowns).  Two interface equations
               Df (u_pf - u_Gf)/s = Ds (u_Gs - u_ps)/s          (flux continuity, q = -Df (u_pf-u_Gf)/s)
               q = hc (u_Gs/K - u_Gf)    or   u_Gs = K u_Gf
            are solved for (u_Gf, u_Gs) -> q is a linear combination of probe-stencil unknowns;
            +q A_w enters the fluid cell, -q A_w the solid cell (exactly conservative).
  The fluid facet and the solid facet of a cell are the same facet; cells cut by the core interface
  carry both unknowns.  Outer boundary: Dirichlet probe flux as in jc.py.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

R = 1.0; A = 0.5


def exact(Ds, K, hc, gs, nmax=1):
    def det(mu):
        wf = np.sqrt(mu); ws = np.sqrt(gs * mu / Ds)
        # u_s = P J0(ws r);  u_f = B J0(wf r) + C Y0(wf r)
        M = np.zeros((3, 3))
        M[0] = [0, j0(wf * R), y0(wf * R)]                                  # u_f(R) = 0
        M[1] = [-Ds * ws * j1(ws * A), wf * j1(wf * A), wf * y1(wf * A)]    # Ds u_s' = Df u_f'
        qs = [Ds * ws * j1(ws * A), 0, 0]                                   # q = -Ds u_s'
        if np.isinf(hc):
            M[2] = [j0(ws * A), -K * j0(wf * A), -K * y0(wf * A)]          # u_s = K u_f
        else:
            M[2] = [qs[0] - hc / K * j0(ws * A), hc * j0(wf * A), hc * y0(wf * A)]
        return np.linalg.det(M)
    mus = np.linspace(0.05, 60, 6000); d = [det(m) for m in mus]
    for i in range(len(mus) - 1):
        if d[i] * d[i + 1] < 0:
            return brentq(det, mus[i], mus[i + 1], xtol=1e-13)


class Geo2:
    def __init__(self, ND, off):
        h = 2 * R / ND; n = ND + 6; self.h, self.n = h, n
        x0 = -n * h / 2 + off[0] * h; y0_ = -n * h / 2 + off[1] * h
        self.xe = x0 + h * np.arange(n + 1); self.ye = y0_ + h * np.arange(n + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1]); self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        def ap(edges_fixed, edges_var, rad, inside):
            s = np.sqrt(np.maximum(rad**2 - edges_fixed**2, 0))[:, None]
            lo, hi = edges_var[None, :-1], edges_var[None, 1:]
            return np.clip(np.minimum(hi, s) - np.maximum(lo, -s), 0, h) / h
        # disc apertures (inside rad) on x-faces [i, j] and y-faces [i, j]
        dRx = ap(self.xe, self.ye, R, True); dAx = ap(self.xe, self.ye, A, True)
        dRy = ap(self.ye, self.xe, R, True).T; dAy = ap(self.ye, self.xe, A, True).T
        self.afx, self.afy = dRx - dAx, dRy - dAy        # fluid annulus apertures
        self.asx, self.asy = dAx, dAy                    # solid core apertures
        g, w = np.polynomial.legendre.leggauss(32)
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        r = np.hypot(X, Y)
        self.kf = ((r > A) & (r < R)).astype(float); self.ks = (r < A).astype(float)
        cut = (np.abs(r - A) < 1.5 * h) | (np.abs(r - R) < 1.5 * h)
        for i, j in zip(*np.nonzero(cut)):
            XX, YY = np.meshgrid(self.xc[i] + 0.5 * h * g, self.yc[j] + 0.5 * h * g, indexing="ij")
            rr = np.hypot(XX, YY); W = np.outer(w, w) * 0.25
            self.kf[i, j] = (W * ((rr > A) & (rr < R))).sum(); self.ks[i, j] = (W * (rr < A)).sum()
        # facets: inner (core) and outer (R) wall areas from the aperture area vectors
        def area(ax_, ay_):
            Ax = -(ax_[1:, :] - ax_[:-1, :]) * h; Ay = -(ay_[:, 1:] - ay_[:, :-1]) * h
            return np.hypot(Ax, Ay)
        self.Ain = area(dAx, dAy); self.Aout = area(dRx, dRy)
        self.X, self.Y = X, Y


FALLBACK = [0]


def bilin(G, idx, p):
    h = G.h
    fi = (p[0] - G.xc[0]) / h; fj = (p[1] - G.yc[0]) / h
    i0, j0_ = int(np.floor(fi)), int(np.floor(fj)); tx, ty = fi - i0, fj - j0_
    cand = [(i0, j0_), (i0 + 1, j0_), (i0, j0_ + 1), (i0 + 1, j0_ + 1)]
    ww = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
    ok = [idx[q] >= 0 for q in cand]
    if all(ok):
        return [(idx[q], wv) for q, wv in zip(cand, ww)]
    wsum = sum(wv for o, wv in zip(ok, ww) if o)
    if wsum < 0.5:
        return None
    FALLBACK[0] += 1
    return [(idx[q], wv / wsum) for q, wv, o in zip(cand, ww, ok) if o]


def build(G, Ds, K, hc, gs, c=0.7):
    n, h = G.n, G.h
    vf = G.kf > 1e-12; vs = G.ks > 1e-12
    idf = -np.ones((n, n), int); idf[vf] = np.arange(vf.sum())
    ids = -np.ones((n, n), int); ids[vs] = vf.sum() + np.arange(vs.sum())
    N = vf.sum() + vs.sum()
    rows, cols, vals = [], [], []
    def add(r, cc, v): rows.append(r); cols.append(cc); vals.append(v)
    # faces (two-point, aperture-weighted), K = -(div D grad) * h^2 / h^2
    for (idx, D, ax_, ay_) in ((idf, 1.0, G.afx, G.afy), (ids, Ds, G.asx, G.asy)):
        for (a, P, Q) in ((ax_[1:-1, :], idx[:-1, :], idx[1:, :]), (ay_[:, 1:-1], idx[:, :-1], idx[:, 1:])):
            m = (a > 0) & (P >= 0) & (Q >= 0)
            for aa, p, q in zip(a[m], P[m], Q[m]):
                cf = D * aa / h**2
                add(p, p, cf); add(p, q, -cf); add(q, q, cf); add(q, p, -cf)
    s = c * h
    # inner facets (conjugate)
    for i, j in zip(*np.nonzero(G.Ain > 1e-12 * h)):
        cx, cy = G.X[i, j], G.Y[i, j]
        rr = np.hypot(cx, cy); xw = np.array([cx, cy]) * A / rr; nn = xw / A  # into fluid (outward)
        for sf in (s, 1.0 * h, 1.5 * h):
            pf = bilin(G, idf, xw + sf * nn)
            if pf is not None: break
        for ss in (s, 1.0 * h, 1.5 * h):
            ps = bilin(G, ids, xw - ss * nn)
            if ps is not None: break
        if pf is None or ps is None:
            raise RuntimeError("probe failed")
        gf, gs_ = 1.0 / sf, Ds / ss
        # unknowns uGf, uGs:  gf (uGf - upf) = gs (ups - uGs)      [q = gf (uGf - upf)]
        #   hc=inf: uGs - K uGf = 0 ; else gf (uGf - upf) - hc (uGs/K - uGf) = 0
        Mx = np.array([[gf, gs_], [-K, 1.0]]) if np.isinf(hc) else np.array([[gf, gs_], [gf + hc, -hc / K]])
        # rhs = Bf * upf + Bs * ups
        Bf = np.array([gf, 0.0]) if np.isinf(hc) else np.array([gf, gf])
        Bs = np.array([gs_, 0.0])
        Minv = np.linalg.inv(Mx)
        cf_ = Minv @ Bf; cs_ = Minv @ Bs   # uG = cf_*upf + cs_*ups
        # q = gf (uGf - upf) = gf (cf_[0]-1) upf + gf cs_[0] ups
        qf = gf * (cf_[0] - 1.0); qs = gf * cs_[0]
        Aw = G.Ain[i, j]
        rf = idf[i, j]; rs = ids[i, j]
        for (col, wv) in pf:
            if rf >= 0: add(rf, col, -Aw * qf * wv / h**2)      # fluid gains +q A_w -> K gets -q
            if rs >= 0: add(rs, col, +Aw * qf * wv / h**2)
        for (col, wv) in ps:
            if rf >= 0: add(rf, col, -Aw * qs * wv / h**2)
            if rs >= 0: add(rs, col, +Aw * qs * wv / h**2)
    # outer facets: Dirichlet 0 on the fluid side; flux out of fluid = D (u_p - 0)/s
    for i, j in zip(*np.nonzero(G.Aout > 1e-12 * h)):
        if idf[i, j] < 0: continue
        cx, cy = G.X[i, j], G.Y[i, j]; rr = np.hypot(cx, cy)
        xw = np.array([cx, cy]) * R / rr; nn = -xw / R
        pf = bilin(G, idf, xw + s * nn)
        if pf is None: raise RuntimeError("outer probe failed")
        for col, wv in pf:
            add(idf[i, j], col, G.Aout[i, j] * (1.0 / s) * wv / h**2)
    Kmat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    M = np.concatenate([G.kf[vf], gs * G.ks[vs]])
    return Kmat, M


def slowest(Kmat, M):
    sc = 1 / np.sqrt(M)
    Am = sp.diags(sc) @ Kmat @ sp.diags(sc)
    w = spla.eigs(Am.tocsc(), k=2, sigma=-1e-3, which="LM", return_eigenvectors=False)
    return np.sort(w.real)[0]


if __name__ == "__main__":
    rng = np.random.default_rng(1); offs = rng.random((3, 2))
    NDs = [16, 32, 64, 128]
    cases = [(1.0, 1.0, np.inf, 1.0), (10.0, 1.0, np.inf, 1.0), (0.1, 1.0, np.inf, 1.0),
             (100.0, 1.0, np.inf, 0.5), (1.0, 3.0, np.inf, 1.0), (1.0, 1.0, 5.0, 1.0),
             (10.0, 0.5, 2.0, 2.0)]
    for (Ds, K, hc, gs) in cases:
        ref = exact(Ds, K, hc, gs)
        line = f"Ds={Ds:5g} K={K:3g} hc={hc:4g} gs={gs:3g} mu={ref:8.5f}"; prev = None
        for ND in NDs:
            e = []
            for o in offs:
                G = Geo2(ND, o)
                Km, M = build(G, Ds, K, hc, gs)
                e.append(slowest(Km, M) / ref - 1)
            em = np.sqrt(np.mean(np.square(e)))
            line += f" | {ND:3d}: {np.mean(e):+8.1e}" + (f" p={np.log(prev/em)/np.log(2):4.1f}" if prev else "       ")
            prev = em
        print(line + f'  fallbacks={FALLBACK[0]}', flush=True); FALLBACK[0] = 0
