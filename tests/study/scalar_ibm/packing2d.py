"""2-D prototype, round 8: NEAR-CONTACT conjugate conduction.

Effective conductivity of a periodic square array of cylinders (unit cell, cylinder radius R at the
centre), macroscopic gradient dT/dx = -1, k_f = 1.  Gap g = 1 - 2R from 0.2 down to 0.01.
Unknowns are periodic: T(x) = theta - x.  k_eff = total +x heat flux through a face column.

Methods
  P2F    two-field probe-flux cut-cell FV (conj.py), probe 0.7h, round-4 fallback ladder
         (renormalised valid weights -> 1.0h -> 1.5h).  A probe landing inside ANOTHER solid image
         is not detected (the current design's blind spot).
  P2Fg   P2F + gap-aware fluid probe: s_f = min(0.7h, 0.5 * normal distance to the next solid)
  PH     Peters' hybrid (his 3-D packing code construct_flux_ibm.m, transcribed to 2-D):
           simple crossing face (1 crossing, 2 same-phase cells each side) -> directional
             min-curvature quadratic, aperture-weighted branches;
           close / very close crossing face -> appendix series-parallel
             k_eff = [k_f + (k_s-k_f) a_s (1-n_x^2)] / [1 + (k_f/k_s - 1) l_s n_x^2]
             (l_s = solid fraction of the centre line);
           cut non-crossing face -> k_f + (k_s-k_f) a_s (1-n_x^2) (fluid centres; mirror for solid).
Reference: Richardson extrapolation (p = 2) of the two finest resolutions, compared across methods.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys
from conj_peters import dir_face_both

C0 = np.array([0.5, 0.5])
IMG = [np.array([a, b], float) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def sdf(p, R):  # > 0 in fluid
    return min(np.hypot(*(p - C0 - s)) for s in IMG) - R


def center_of(p):
    return min((C0 + s for s in IMG), key=lambda c: np.hypot(*(p - c)))


class Cell:
    def __init__(self, n, R, off=(0.0, 0.0)):
        self.n, self.R, self.h = n, R, 1.0 / n
        h = self.h
        self.x0 = off[0] * h; self.y0 = off[1] * h
        self.xc = self.x0 + (np.arange(n) + 0.5) * h; self.yc = self.y0 + (np.arange(n) + 0.5) * h
        self.xe = self.x0 + np.arange(n + 1) * h; self.ye = self.y0 + np.arange(n + 1) * h
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij"); self.X, self.Y = X, Y
        self.s = np.array([[sdf(np.array([X[i, j], Y[i, j]]), R) for j in range(n)] for i in range(n)])
        self.solid = self.s < 0
        def solid_len(c, lo, hi, axis):
            tot = 0.0
            for sft in IMG:
                cc = C0 + sft; d = c - cc[axis]
                if abs(d) >= R: continue
                w = np.sqrt(R * R - d * d); m = cc[1 - axis]
                tot += max(0.0, min(hi, m + w) - max(lo, m - w))
            return min(tot, hi - lo)
        self.asx = np.array([[solid_len(self.xe[i], self.ye[j], self.ye[j + 1], 0) / h for j in range(n)] for i in range(n + 1)])
        self.asy = np.array([[solid_len(self.ye[j], self.xe[i], self.xe[i + 1], 1) / h for j in range(n + 1)] for i in range(n)])
        g, w = np.polynomial.legendre.leggauss(12)
        self.ks = self.solid.astype(float)
        for i, j in zip(*np.nonzero(np.abs(self.s) < 0.75 * h)):
            pts = [(X[i, j] + 0.5 * h * a, Y[i, j] + 0.5 * h * b, wa * wb * 0.25) for a, wa in zip(g, w) for b, wb in zip(g, w)]
            self.ks[i, j] = sum(ww for (px, py, ww) in pts if sdf(np.array([px, py]), R) < 0)

    def pos(self, c):  # unwrapped centre of (possibly out-of-range) index c
        return np.array([self.x0 + (c[0] + 0.5) * self.h, self.y0 + (c[1] + 0.5) * self.h])


class System:
    """rows of 'sum outward k dT/dn' with T = theta - x; theta periodic"""
    def __init__(self, N):
        self.r, self.c, self.v = [], [], []; self.rhs = np.zeros(N); self.N = N

    def add(self, row, col, w, xcol):  # contribution w * T(col) = w*theta(col) - w*x(col)
        self.r.append(row); self.c.append(col); self.v.append(w); self.rhs[row] += w * xcol

    def solve(self, pin=0):
        A = sp.csr_matrix((self.v, (self.r, self.c)), shape=(self.N, self.N)).tolil()
        rhs = self.rhs.copy(); A[pin, :] = 0; A[pin, pin] = 1; rhs[pin] = 0
        return spla.spsolve(A.tocsc(), rhs)


def k_PH(G, kf, ks):
    n, h = G.n, G.h
    kc = np.where(G.solid, ks, kf)
    I = lambda c: (c[0] % n) * n + (c[1] % n)
    S = System(n * n); col = []
    for axis in (0, 1):
        sh = np.array((1, 0) if axis == 0 else (0, 1))
        for i in range(n):
            for j in range(n):
                P = (i, j); Q = (i + sh[0], j + sh[1])
                Pw = (P[0] % n, P[1] % n); Qw = (Q[0] % n, Q[1] % n)
                sP, sQ = G.solid[Pw], G.solid[Qw]
                a_s = G.asx[i + 1, j] if axis == 0 else G.asy[i, j + 1]
                pP = G.pos(P); fpt = pP + 0.5 * h * sh
                nrm = fpt - center_of(fpt); nrm /= np.hypot(*nrm); nx2 = nrm[axis] ** 2
                if min(abs(G.s[Pw]), abs(G.s[Qw])) > 1.5 * h:
                    ts = np.array([0.0, 1.0]); sv = np.array([G.s[Pw], G.s[Qw]])
                else:
                    ts = np.linspace(0, 1, 129); sv = np.array([sdf(pP + t * h * sh, G.R) for t in ts])
                cr = np.nonzero(sv[:-1] * sv[1:] < 0)[0]
                if len(cr) == 0:
                    if 0 < a_s < 1:
                        kk = kf + (ks - kf) * a_s * (1 - nx2) if not sP else ks + (kf - ks) * (1 - a_s) * (1 - nx2)
                    else:
                        kk = kc[Pw]
                    st = [(P, -kk), (Q, kk)]
                else:
                    Pm = (P[0] - sh[0], P[1] - sh[1]); Qp = (Q[0] + sh[0], Q[1] + sh[1])
                    simple = (len(cr) == 1 and G.solid[Pm[0] % n, Pm[1] % n] == sP and G.solid[Qp[0] % n, Qp[1] % n] == sQ)
                    if simple:
                        k_ = cr[0]; t0 = ts[k_] + (ts[k_ + 1] - ts[k_]) * sv[k_] / (sv[k_] - sv[k_ + 1])
                        gl, gr = dir_face_both(t0 - 0.5)
                        fP = a_s if sP else 1 - a_s
                        w = kc[Pw] * fP * gl + kc[Qw] * (1 - fP) * gr
                        st = [(Pm, w[0]), (P, w[1]), (Q, w[2]), (Qp, w[3])]
                    else:
                        ls = np.mean(sv < 0)
                        kk = (kf + (ks - kf) * a_s * (1 - nx2)) / (1 + (kf / ks - 1) * ls * nx2)
                        st = [(P, -kk), (Q, kk)]
                for c, wv in st:   # F = sum w T / h ; +F/h to P (outward east), -F/h to Q
                    xc = G.pos(c)[0]
                    S.add(I(P), I(c), wv / h**2, xc); S.add(I(Q), I(c), -wv / h**2, xc)
                if axis == 0 and i == n - 1:
                    col.append(st)
    th = S.solve()
    T = lambda c: th[I(c)] - G.pos(c)[0]
    return sum(-sum(wv * T(c) for c, wv in st) / h * h for st in col)   # sum of -k dT/dx * h


def bilin(G, ids_fn, p, valid):
    n, h = G.n, G.h
    fi = (p[0] - G.x0) / h - 0.5; fj = (p[1] - G.y0) / h - 0.5
    i0, j0 = int(np.floor(fi)), int(np.floor(fj)); tx, ty = fi - i0, fj - j0
    cand = [(i0, j0), (i0 + 1, j0), (i0, j0 + 1), (i0 + 1, j0 + 1)]
    ww = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
    ok = [valid[c[0] % n, c[1] % n] for c in cand]
    ws = sum(w for w, o in zip(ww, ok) if o)
    if ws < 0.5: return None, False
    return [(c, w / ws) for c, w, o in zip(cand, ww, ok) if o], not all(ok)


def k_P2F(G, kf, ks, gapaware=False, c=0.7):
    n, h = G.n, G.h
    kfr = 1 - G.ks
    # a phase DOF exists where the phase has volume OR any open face aperture (consistency with the
    # exact apertures; an independent volume quadrature can miss corner slivers)
    def touched(ax_, ay_):
        t = (ax_[1:, :] > 0) | (ax_[:-1, :] > 0) | (ay_[:, 1:] > 0) | (ay_[:, :-1] > 0)
        return t
    vf = (kfr > 1e-12) | touched(1 - G.asx, 1 - G.asy)
    vs = (G.ks > 1e-12) | touched(G.asx, G.asy)
    idf = -np.ones((n, n), int); idf[vf] = np.arange(vf.sum())
    ids = -np.ones((n, n), int); ids[vs] = vf.sum() + np.arange(vs.sum())
    N = vf.sum() + vs.sum()
    S = System(N)
    If = lambda cc: idf[cc[0] % n, cc[1] % n]; Is = lambda cc: ids[cc[0] % n, cc[1] % n]
    col = []
    for (Ix, D, ax_, ay_, valid) in ((If, kf, 1 - G.asx, 1 - G.asy, vf), (Is, ks, G.asx, G.asy, vs)):
        for axis in (0, 1):
            sh = (1, 0) if axis == 0 else (0, 1)
            for i in range(n):
                for j in range(n):
                    a = ax_[i + 1, j] if axis == 0 else ay_[i, j + 1]
                    P = (i, j); Q = (i + sh[0], j + sh[1])
                    if a <= 0 or Ix(P) < 0 or Ix(Q) < 0: continue
                    cf = D * a / h**2
                    for (cc, wv) in ((P, -cf), (Q, cf)):
                        xc = G.pos(cc)[0]
                        S.add(Ix(P), Ix(cc), wv, xc); S.add(Ix(Q), Ix(cc), -wv, xc)
                    if axis == 0 and i == n - 1:
                        col.append((Ix, D * a, P, Q))
    Ax = -(G.asx[1:, :] - G.asx[:-1, :]) * h; Ay = -(G.asy[:, 1:] - G.asy[:, :-1]) * h
    Aw = np.hypot(Ax, Ay)
    stats = dict(fallback=0, inside_other=0)
    for i, j in zip(*np.nonzero(Aw > 1e-12 * h)):
        if idf[i, j] < 0 or ids[i, j] < 0: continue
        p0 = G.pos((i, j)); cc = center_of(p0); nn = (p0 - cc) / np.hypot(*(p0 - cc)); xw = cc + G.R * nn
        sf_max = c * h
        if gapaware:   # normal distance to the next solid along +n
            ts = np.linspace(1e-4, 3 * h, 300)
            hit = [t for t in ts if sdf(xw + t * nn, G.R) < 0 and np.hypot(*(xw + t * nn - cc)) > G.R + 1e-12]
            if hit: sf_max = min(sf_max, 0.5 * hit[0])
        pf = None
        for sf in (sf_max, 1.0 * h, 1.5 * h):
            pf, fb = bilin(G, If, xw + sf * nn, vf)
            if pf is not None: break
        if sdf(xw + sf * nn, G.R) < 0: stats["inside_other"] += 1
        for ss in (c * h, 1.0 * h, 1.5 * h):
            ps, fb2 = bilin(G, Is, xw - ss * nn, vs)
            if ps is not None: break
        stats["fallback"] += int(fb) + int(fb2)
        gf, gs = kf / sf, ks / ss
        g = gf * gs / (gf + gs)     # q (solid -> fluid) = g (T_ps - T_pf); +q A_w into fluid
        A = Aw[i, j] / h**2
        # rows are sum_faces k grad T . n_out (gradient flux, = div k grad T).  Through the facet the
        # fluid cell's k grad T . n_out = -k dT/dn_f = q (q = heat flux solid -> fluid along n), and the
        # solid cell's is -q.  q = g (T_ps - T_pf).
        for cc_, wv in pf:
            x_ = G.pos(cc_)[0]
            S.add(idf[i, j], If(cc_), -A * g * wv, x_); S.add(ids[i, j], If(cc_), +A * g * wv, x_)
        for cc_, wv in ps:
            x_ = G.pos(cc_)[0]
            S.add(idf[i, j], Is(cc_), +A * g * wv, x_); S.add(ids[i, j], Is(cc_), -A * g * wv, x_)
    th = S.solve(pin=0)
    q = 0
    for (Ix, ka, P, Q) in col:
        TP = th[Ix(P)] - G.pos(P)[0]; TQ = th[Ix(Q)] - G.pos(Q)[0]
        q += -ka * (TQ - TP) / h * h
    return q, stats


if __name__ == "__main__":
    ks = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    gaps = [float(a) for a in sys.argv[2:]] or [0.2, 0.05, 0.02]
    for gap in gaps:
        R = 0.5 * (1 - gap)
        res = {}
        for n in (16, 32, 64, 128):
            G = Cell(n, R, off=(0.0, 0.0))
            res.setdefault("PH", []).append(k_PH(G, 1.0, ks))
            q, st = k_P2F(G, 1.0, ks); res.setdefault("P2F", []).append(q); res.setdefault("P2F_st", []).append(st)
            q, st = k_P2F(G, 1.0, ks, gapaware=True); res.setdefault("P2Fg", []).append(q); res.setdefault("P2Fg_st", []).append(st)
            print(f"gap={gap} n={n:3d} (cells/gap {gap*n:5.2f}) PH={res['PH'][-1]:.6f} P2F={res['P2F'][-1]:.6f} "
                  f"P2Fg={res['P2Fg'][-1]:.6f} | P2F probe-in-other-solid {res['P2F_st'][-1]['inside_other']}", flush=True)
        for m in ("PH", "P2F", "P2Fg"):
            v = res[m]; rich = v[-1] + (v[-1] - v[-2]) / 3
            print(f"   {m:4s} Richardson(64,128) = {rich:.6f}   successive diffs {np.diff(v)}")
