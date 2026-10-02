"""2-D prototype, round 6: explicit advection with kappa storage on cut cells (small-cell problem).

Flow: solid-body rotation psi = r^2/2 in the annulus a < r < R (a = 0.4, R = 1), both circles
immersed. Face fluxes are EXACT integrals of the stream function over the OPEN part of each face,
so the discrete aperture-weighted divergence vanishes in every cut cell (as for flow's projected
MAC flux). A Gaussian blob next to the outer wall is carried once round (T = 2 pi); the exact
answer is the initial field.  dt = CFL * h / max|u| with the FULL-cell CFL 0.5.

Schemes (first-order upwind fluxes throughout, to isolate the storage / small-cell question):
  unit   : (c^{n+1}-c^n) h^2/dt = -sum F c_up     (flow today: conserves sum c, not sum kappa c)
  kappa  : kappa h^2 (c^{n+1}-c^n)/dt = -sum F c_up   (explicit; unstable for small kappa?)
  mb     : kappa storage; faces touching a cell with kappa < 1/2 use IMPLICIT upwind values, all
           other faces explicit (May & Berger-style explicit-implicit split; conservative because
           each face flux is shared by its two cells); one sparse solve per step
  impl   : kappa storage, all faces implicit (reference for stability)
Reported: max/min of c (overshoot = instability or max-principle violation), relative drift of
the physical mass sum kappa c, L1 error vs the exact (initial) field over the fluid volume.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys

R, A = 1.0, 0.4


class Ann:
    def __init__(self, ND, off):
        h = 2 * R / ND; n = ND + 4; self.h, self.n = h, n
        x0 = -n * h / 2 + off[0] * h; y0 = -n * h / 2 + off[1] * h
        self.xe = x0 + h * np.arange(n + 1); self.ye = y0 + h * np.arange(n + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1]); self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        psi = lambda x, y: 0.5 * (x * x + y * y)
        def open_intervals(c, lo, hi):  # fluid set a<r<R on the line (coordinate c fixed), within [lo,hi]
            sR = np.sqrt(max(R**2 - c * c, 0)); sA = np.sqrt(max(A**2 - c * c, 0))
            ivs = [(-sR, -sA), (sA, sR)] if sA > 0 else [(-sR, sR)]
            out = []
            for (p, q) in ivs:
                l, u = max(p, lo), min(q, hi)
                if u > l: out.append((l, u))
            return out
        # Fx[i, j]: flux in +x through x-face at xe[i], row j ; Fy[i, j]: +y through y-face ye[j], column i
        self.Fx = np.zeros((n + 1, n)); self.ax = np.zeros((n + 1, n))
        for i in range(n + 1):
            for j in range(n):
                for (l, u) in open_intervals(self.xe[i], self.ye[j], self.ye[j + 1]):
                    self.Fx[i, j] += psi(self.xe[i], u) - psi(self.xe[i], l)
                    self.ax[i, j] += (u - l) / h
        self.Fy = np.zeros((n, n + 1)); self.ay = np.zeros((n, n + 1))
        for i in range(n):
            for j in range(n + 1):
                for (l, u) in open_intervals(self.ye[j], self.xe[i], self.xe[i + 1]):
                    self.Fy[i, j] += -(psi(u, self.ye[j]) - psi(l, self.ye[j]))
                    self.ay[i, j] += (u - l) / h
        g, w = np.polynomial.legendre.leggauss(24)
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij"); r = np.hypot(X, Y)
        self.kap = ((r > A) & (r < R)).astype(float)
        cut = (np.abs(r - A) < 1.5 * h) | (np.abs(r - R) < 1.5 * h)
        for i, j in zip(*np.nonzero(cut)):
            XX, YY = np.meshgrid(self.xc[i] + 0.5 * h * g, self.yc[j] + 0.5 * h * g, indexing="ij")
            rr = np.hypot(XX, YY)
            self.kap[i, j] = (np.outer(w, w) * 0.25 * ((rr > A) & (rr < R))).sum()
        self.X, self.Y = X, Y
        div = (self.Fx[1:, :] - self.Fx[:-1, :]) + (self.Fy[:, 1:] - self.Fy[:, :-1])
        self.maxdiv = np.abs(div).max()


def upwind_matrix(G, valid, idx, implicit_face):
    """Matrix U with (U c)_i = sum over faces of F_out * c_upwind (net outflow), restricted to faces
    selected by implicit_face(face) True/False; returns (U_sel) as sparse."""
    n = G.n; rows, cols, vals = [], [], []
    for (F, sh) in ((G.Fx, 0), (G.Fy, 1)):
        if sh == 0:
            Fi = F[1:-1, :]; P = idx[:-1, :]; Q = idx[1:, :]; Pm = np.s_[:-1, :]; Qm = np.s_[1:, :]
        else:
            Fi = F[:, 1:-1]; P = idx[:, :-1]; Q = idx[:, 1:]
        m = (P >= 0) & (Q >= 0) & (Fi != 0)
        for f, p, q, sel in zip(Fi[m], P[m], Q[m], implicit_face(sh)[m]):
            if not sel: continue
            up = p if f > 0 else q
            # outflow from p: +f c_up ; inflow to q: -f c_up
            rows += [p, q]; cols += [up, up]; vals += [f, -f]
    N = valid.sum()
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def run(ND, scheme, cfl=0.5, off=(0.37, 0.71)):
    G = Ann(ND, off); h = G.h
    valid = G.kap > 1e-12
    idx = -np.ones((G.n, G.n), int); idx[valid] = np.arange(valid.sum())
    kap = G.kap[valid]
    small = np.zeros((G.n, G.n), bool); small[valid] = kap < 0.5
    def faces_touching_small(sh):
        if sh == 0: return small[:-1, :] | small[1:, :]
        return small[:, :-1] | small[:, 1:]
    allf = lambda sh: np.ones((G.n - 1, G.n) if sh == 0 else (G.n, G.n - 1), bool)
    nof = lambda sh: np.zeros((G.n - 1, G.n) if sh == 0 else (G.n, G.n - 1), bool)
    c0 = np.exp(-((G.X - 0.7) ** 2 + G.Y**2) / (2 * 0.08**2))[valid]
    dt = cfl * h / 1.0
    nsteps = int(np.ceil(2 * np.pi / dt)); dt = 2 * np.pi / nsteps
    M = kap if scheme != "unit" else np.ones_like(kap)
    if scheme in ("unit", "kappa"):
        Uex = upwind_matrix(G, valid, idx, allf); Uim = None
    elif scheme == "mb":
        Uim = upwind_matrix(G, valid, idx, faces_touching_small)
        Uex = upwind_matrix(G, valid, idx, lambda sh: ~faces_touching_small(sh))
    else:
        Uim = upwind_matrix(G, valid, idx, allf); Uex = upwind_matrix(G, valid, idx, nof)
    Mh = M * h * h
    lu = spla.splu((sp.diags(Mh / dt) + Uim).tocsc()) if Uim is not None else None
    c = c0.copy(); mass0 = (kap * c).sum()
    cmax = 1.0
    for _ in range(nsteps):
        rhs = Mh / dt * c - Uex @ c
        c = lu.solve(rhs) if lu is not None else rhs / (Mh / dt)
        cmax = max(cmax, np.abs(c).max())
        if not np.isfinite(cmax) or cmax > 1e6:
            return dict(status="BLOWUP", maxdiv=G.maxdiv, minkap=kap.min())
    err = (kap * np.abs(c - c0)).sum() / (kap * c0).sum()
    return dict(status="ok", cmax=c.max(), cmin=c.min(), overshoot=cmax - 1,
                mass=(kap * c).sum() / mass0 - 1, L1=err, maxdiv=G.maxdiv, minkap=kap.min(), steps=nsteps)


if __name__ == "__main__":
    for ND in (32, 64, 128):
        for scheme in ("unit", "kappa", "mb", "impl"):
            o = run(ND, scheme)
            if o["status"] != "ok":
                print(f"ND={ND:3d} {scheme:6s} BLOW-UP   (min kappa {o['minkap']:.1e}, max|div| {o['maxdiv']:.1e})", flush=True)
                continue
            print(f"ND={ND:3d} {scheme:6s} L1={o['L1']:.3e} mass drift={o['mass']:+.2e} "
                  f"max over run={1+o['overshoot']:.4f} min={o['cmin']:+.1e} | min kappa {o['minkap']:.1e} "
                  f"max|div| {o['maxdiv']:.1e} steps {o['steps']}", flush=True)
