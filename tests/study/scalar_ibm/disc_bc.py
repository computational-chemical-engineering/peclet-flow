"""2-D prototype: immersed-surface scalar BCs on a pipe cross-section (disc of radius R=1 in a
square Cartesian box, fluid INSIDE the disc).  Every gate has an exact answer:

  Neumann  : first non-zero eigenvalue of -lap on the disc      j'_11^2 = 3.389957
             Taylor-Aris closure  -<u'b> = 1/48 (D=U=R=1)       (D_ax/alpha = 1 + Pe^2/48)
  Dirichlet: decay eigenvalue                                    j_01^2  = 5.783186
             Graetz total-flux Nu_F = Pe*lambda*R at Pe = 10, 1000 (1F1 root, Peters eq. 43-44)

Discretizations compared (sparse assembly here; every one is a <=5-point row, i.e. 7-point in 3-D):
  Neumann:
    FV-ap/kappa : aperture two-point fluxes, storage/source weighted by fluid fraction kappa
    FV-ap/1     : same fluxes, unit storage/source (what flow's scalar path does today)
    stair       : binary apertures, centre-in-fluid cells only (pure staircase baseline)
  Dirichlet:
    mask-ap     : flow's per-cell Dirichlet mask (centre-in-solid cells pinned to 0) + apertures
    ghost-lin   : Gibou 2002 linear ghost, symmetric (diagonal += 1/theta)
    ghost-quad  : Robust-Scaled scheme-0 quadratic ghost (flow cut_cell_closure.hpp), non-symmetric
    FV-wall     : cut-cell FV, all kappa>0 cells, wall flux D*A_w*(c_G - c_P)/d(centroid), mass kappa
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.special import hyp1f1, jn_zeros, jnp_zeros

R = 1.0
J01SQ = jn_zeros(0, 1)[0] ** 2
JP11SQ = jnp_zeros(1, 1)[0] ** 2


def graetz_ref(Pe):
    f = lambda s: hyp1f1((1 - s - s**3 / Pe**2) / 2, 1, 2 * s)
    ss = np.linspace(0.05, 3, 600)
    v = [f(s) for s in ss]
    for k in range(len(ss) - 1):
        if v[k] * v[k + 1] < 0:
            s = brentq(f, ss[k], ss[k + 1], xtol=1e-14)
            return 2 * s * s  # Nu_F = Pe*lambda*R = 2 s^2
    raise RuntimeError


class Geo:
    def __init__(self, ND, off):
        h = 2 * R / ND
        n = ND + 6
        self.h, self.n = h, n
        x0 = -n * h / 2 + off[0] * h
        y0 = -n * h / 2 + off[1] * h
        self.xe = x0 + h * np.arange(n + 1)  # cell edges
        self.ye = y0 + h * np.arange(n + 1)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.sdf = R - np.hypot(X, Y)  # >0 fluid
        # x-face apertures ax[i, j]: face at xe[i], cell row j (i = 0..n)
        s = np.sqrt(np.maximum(R**2 - self.xe**2, 0))[:, None]
        lo, hi = self.ye[None, :-1], self.ye[None, 1:]
        self.ax = np.clip(np.minimum(hi, s) - np.maximum(lo, -s), 0, h) / h
        s = np.sqrt(np.maximum(R**2 - self.ye**2, 0))[None, :]
        lo, hi = self.xe[:-1, None], self.xe[1:, None]
        self.ay = np.clip(np.minimum(hi, s) - np.maximum(lo, -s), 0, h) / h
        # fluid fraction, fluid centroid, fluid-average of u=2(1-r^2) by Gauss quadrature
        g, w = np.polynomial.legendre.leggauss(48)
        self.kap = np.zeros((n, n))
        self.cx = X.copy(); self.cy = Y.copy()
        self.ubar = np.zeros((n, n))
        cut = (np.abs(self.sdf) < 1.5 * h)
        full = (self.sdf >= 1.5 * h)
        self.kap[full] = 1.0
        self.ubar[full] = 2 * (1 - X[full] ** 2 - Y[full] ** 2) - (h * h / 3)  # exact cell average
        for i, j in zip(*np.nonzero(cut)):
            xs = self.xc[i] + 0.5 * h * g
            ys = self.yc[j] + 0.5 * h * g
            XX, YY = np.meshgrid(xs, ys, indexing="ij")
            m = (XX**2 + YY**2 < R**2)
            W = np.outer(w, w) * 0.25 * m
            a = W.sum()
            self.kap[i, j] = a
            if a > 0:
                self.cx[i, j] = (W * XX).sum() / a
                self.cy[i, j] = (W * YY).sum() / a
                self.ubar[i, j] = (W * 2 * (1 - XX**2 - YY**2)).sum() / a
        # wall length per cell from the aperture area vector (flow: A = -(o[a+1]-o[a]))
        Awx = -(self.ax[1:, :] - self.ax[:-1, :]) * h
        Awy = -(self.ay[:, 1:] - self.ay[:, :-1]) * h
        self.Aw = np.hypot(Awx, Awy)
        self.dwall = R - np.hypot(self.cx, self.cy)  # centroid-to-wall distance


def assemble_fv(G, ap_x, ap_y, cells, wall_coef=None):
    """-sum_f a_f (cQ-cP)/h^2 on the index set `cells`; optional extra diagonal (wall flux)."""
    n, h = G.n, G.h
    idx = -np.ones((n, n), int)
    idx[cells] = np.arange(cells.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(cells.sum())
    # x faces between (i-1,j) and (i,j): aperture ap_x[i,j], i=1..n-1
    for (A, sh) in ((ap_x, (1, 0)), (ap_y, (0, 1))):
        if sh == (1, 0):
            a = A[1:-1, :]; P = idx[:-1, :]; Q = idx[1:, :]
        else:
            a = A[:, 1:-1]; P = idx[:, :-1]; Q = idx[:, 1:]
        m = (a > 0) & (P >= 0) & (Q >= 0)
        a, P, Q = a[m] / h**2, P[m], Q[m]
        rows += [P, Q]; cols += [Q, P]; vals += [-a, -a]
        np.add.at(diag, P, a); np.add.at(diag, Q, a)
    if wall_coef is not None:
        diag += wall_coef[cells]
    rows.append(np.arange(len(diag))); cols.append(np.arange(len(diag))); vals.append(diag)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(diag),) * 2)
    return K, idx


def assemble_ghost(G, kind, theta_min=1e-3):
    """Cell-centred 5-point -lap on centre-in-fluid cells, Dirichlet 0 at the SDF crossing."""
    n, h = G.n, G.h
    cells = G.sdf > 0
    idx = -np.ones((n, n), int)
    idx[cells] = np.arange(cells.sum())
    N = cells.sum()
    K = sp.lil_matrix((N, N))
    for i, j in zip(*np.nonzero(cells)):
        r = idx[i, j]
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = idx[i + di, j + dj]
            if q >= 0:
                K[r, q] -= 1 / h**2; K[r, r] += 1 / h**2
                continue
            th = G.sdf[i, j] / (G.sdf[i, j] - G.sdf[i + di, j + dj])
            th = max(th, theta_min)
            far = idx[i - di, j - dj]
            if kind == "lin" or far < 0:
                K[r, r] += 1 / (th * h**2)  # ghost = cP + (0-cP)/th
            else:  # quadratic Lagrange {far, near, wall} -> ghost; D*g = Nc*near + Nnb*far
                D = th * (1 + th); Nc = 2 * (th * th - 1); Nnb = th * (1 - th)
                # contribution (g - cP)/h^2 with g = (Nc cP + Nnb cF)/D, sign: K = -lap
                K[r, r] -= (Nc / D - 1) / h**2
                K[r, far] -= (Nnb / D) / h**2
    return K.tocsr(), idx


def smallest_eig(K, Mdiag, k=2):
    """Smallest eigenvalues of K v = mu M v (M diagonal > 0)."""
    s = 1 / np.sqrt(Mdiag)
    A = sp.diags(s) @ K @ sp.diags(s)
    sym = abs(A - A.T).max() < 1e-9 * abs(A).max()
    if sym:
        w = spla.eigsh(A, k=k, sigma=-1e-3, which="LM", return_eigenvectors=False)
    else:
        w = spla.eigs(A, k=k, sigma=-1e-3, which="LM", return_eigenvectors=False).real
    return np.sort(w)


def graetz(K, Mdiag, udiag, Pe):
    """Smallest lambda>0 with  K v = (lambda^2 M + lambda Pe M u) v  (D=1, U=Pe, R=1)."""
    def g(lam):
        B = Mdiag * (lam**2 + lam * Pe * udiag)
        return smallest_eig(K, B, k=1)[0] - 1.0
    lam_hi = 10.0 / Pe + 3.0 / np.sqrt(Pe + 1)
    while g(lam_hi) > 0:
        lam_hi *= 2
    lam = brentq(g, 1e-6, lam_hi, xtol=1e-12)
    return Pe * lam


def run(ND, off, Pes=(10.0, 1000.0)):
    G = Geo(ND, off)
    out = {}
    # ---------------- Neumann ----------------
    cells = G.kap > 1e-12
    K, idx = assemble_fv(G, G.ax, G.ay, cells)
    kap = G.kap[cells]; ub = G.ubar[cells]
    for tag, M in (("FV-ap/kappa", kap), ("FV-ap/1", np.ones_like(kap))):
        w = smallest_eig(K, M, k=2)
        out[("N-eig", tag)] = w[1]
        # Taylor-Aris: lap b = u' (D=1)  ->  K b = -M u';  -<u'b> = 1/48
        U = (M * ub).sum() / M.sum()
        rhs = -M * (ub - U)
        b = spla.spsolve((K + 1e-14 * sp.eye(K.shape[0])).tocsc(), rhs)
        b -= (M * b).sum() / M.sum()
        out[("TA", tag)] = -(M * (ub - U) * b).sum() / M.sum()
    sc = G.sdf > 0
    axb = np.zeros_like(G.ax); ayb = np.zeros_like(G.ay)
    axb[1:-1, :] = (sc[1:, :] & sc[:-1, :]); ayb[:, 1:-1] = (sc[:, 1:] & sc[:, :-1])
    Ks, _ = assemble_fv(G, axb, ayb, sc)
    w = smallest_eig(Ks, np.ones(sc.sum()), k=2)
    out[("N-eig", "stair")] = w[1]
    u_c = 2 * (1 - G.sdf * 0 - (R - G.sdf) ** 2)  # u at cell centre
    U = u_c[sc].mean()
    bb = spla.spsolve((Ks + 1e-14 * sp.eye(Ks.shape[0])).tocsc(), -(u_c[sc] - U))
    bb -= bb.mean()
    out[("TA", "stair")] = -((u_c[sc] - U) * bb).mean()
    # ---------------- Dirichlet ----------------
    # mask-ap: unknowns centre-fluid; masked neighbours (value 0) via the aperture coupling
    # = FV rows with apertures, diag keeps the coupling to the masked neighbour.
    n = G.n
    wallc = np.zeros((n, n))
    for (di, dj, A, ii, jj) in ((1, 0, G.ax, 1, 0), (-1, 0, G.ax, 0, 0), (0, 1, G.ay, 0, 1), (0, -1, G.ay, 0, 0)):
        pass
    # explicit loop for clarity
    for i, j in zip(*np.nonzero(sc)):
        for (di, dj, a) in ((1, 0, G.ax[i + 1, j]), (-1, 0, G.ax[i, j]), (0, 1, G.ay[i, j + 1]), (0, -1, G.ay[i, j])):
            if not sc[i + di, j + dj]:
                wallc[i, j] += a / G.h**2
    Km, _ = assemble_fv(G, G.ax, G.ay, sc, wall_coef=wallc)
    meth = {"mask-ap": (Km, np.ones(sc.sum()), u_c[sc])}
    for kind in ("lin", "quad"):
        Kg, _ = assemble_ghost(G, kind)
        meth["ghost-" + kind] = (Kg, np.ones(sc.sum()), u_c[sc])
    wc = np.where(G.kap > 1e-12, G.Aw / (G.h**2 * np.maximum(G.dwall, 1e-3 * G.h)), 0)
    # wall_coef is per unit cell area (rows are /h^2 already) -> divide A_w*D/d by h^2
    Kf, _ = assemble_fv(G, G.ax, G.ay, cells, wall_coef=wc)
    meth["FV-wall"] = (Kf, kap, ub)
    for tag, (KK, M, u) in meth.items():
        out[("D-eig", tag)] = smallest_eig(KK, M, k=1)[0]
        for Pe in Pes:
            out[(f"NuF{Pe:g}", tag)] = graetz(KK, M, u, Pe)
    return out


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(1)
    offs = rng.random((int(sys.argv[1]) if len(sys.argv) > 1 else 3, 2))
    NDs = [int(a) for a in sys.argv[2:]] or [16, 32, 64, 128]
    ref = {"N-eig": JP11SQ, "TA": 1 / 48, "D-eig": J01SQ,
           "NuF10": graetz_ref(10.0), "NuF1000": graetz_ref(1000.0)}
    print("references:", {k: round(v, 6) for k, v in ref.items()})
    res = {}
    for ND in NDs:
        for o in offs:
            for k, v in run(ND, o).items():
                res.setdefault((k, ND), []).append(v)
    keys = sorted({k for (k, _) in res})
    for (q, tag) in keys:
        line = f"{q:8s} {tag:12s}"
        prev = None
        for ND in NDs:
            e = np.array(res[((q, tag), ND)]) / ref[q] - 1
            em = np.sqrt((e**2).mean())
            line += f" | N={ND:3d} rms {em:9.2e} spread {e.max()-e.min():8.1e}"
            if prev is not None:
                line += f" p={np.log(prev/em)/np.log(2):4.1f}"
            prev = em
        print(line)
