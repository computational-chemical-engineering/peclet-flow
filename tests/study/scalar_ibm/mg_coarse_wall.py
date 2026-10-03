"""2-D prototype (architect, Amendment A1 to doc/scalar_ibm_design.md): which coarse construction of
the lumped-probe surrogate's WALL term makes a sound V-cycle?

Surrogate (steady, Lam = 1): S = aperture 5-point Laplacian + diagonal wall sink W_i = alpha_i/s0_i,
alpha_i = |area vector|/h^2 (aperture-consistent), s0 = 1.1 * 1/2 (|n_x|+|n_y|) h (Dirichlet wall).
Coarse constructions compared (same smoother, transfers and exact bottom, so only A_c differs):
  redisc-sL : faces averaged + wall W_C = <alpha / s_L>, s_L = probe distance at the level's H (design §5.2)
  avg-s0    : faces averaged + wall W_C = <alpha / s0>   (= variational for piecewise-constant P)
  rap       : Galerkin R A P, R = 4-child average, P = bilinear (zero beyond Dirichlet box faces)
Smoother: 4-colour Gauss-Seidel (colour = (i%2, j%2): race-free for 5- and 9-point), 2 pre (colours
0..3) + 2 post (3..0); bilinear prolongation, pinned cells re-zeroed; exact bottom at 4x4.
Measure: V-cycle contraction rho(I - B S) by power iteration (mean projected out when singular).
Results (n = 64 -> 128) are in doc/scalar_ibm_design.md, Amendment A1.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys

SNAP = 1e-3


def edge_frac(pa, pb):
    if pa > 0 and pb > 0:
        return 1.0
    if pa <= 0 and pb <= 0:
        return 0.0
    p, q = (pa, pb) if pa > 0 else (pb, pa)
    return p / (p - q)


def snap(a):
    return 0.0 if a < SNAP else (1.0 if a > 1 - SNAP else a)


class Geo:
    """n x n cells on [0, L]^2; sdf(x, y) > 0 fluid; periodic or Dirichlet box."""
    def __init__(self, n, L, sdf, periodic):
        self.n, self.h, self.periodic = n, L / n, periodic
        h = self.h
        xe = np.arange(n + 1) * h
        # ax[i, j]: x-face at xe[i] (i = 0..n), cell row j ; ay[i, j]: y-face at ye[j]
        self.ax = np.zeros((n + 1, n)); self.ay = np.zeros((n, n + 1))
        for i in range(n + 1):
            for j in range(n):
                self.ax[i, j] = snap(edge_frac(sdf(xe[i], xe[j]), sdf(xe[i], xe[j + 1])))
        for i in range(n):
            for j in range(n + 1):
                self.ay[i, j] = snap(edge_frac(sdf(xe[i], xe[j]), sdf(xe[i + 1], xe[j])))
        if periodic:  # the periodic face is one face
            self.ax[n, :] = self.ax[0, :]; self.ay[:, n] = self.ay[:, 0]
        Ax = (self.ax[1:, :] - self.ax[:-1, :]) * h
        Ay = (self.ay[:, 1:] - self.ay[:, :-1]) * h
        A = np.hypot(Ax, Ay)
        self.unknown = (self.ax[1:, :] + self.ax[:-1, :] + self.ay[:, 1:] + self.ay[:, :-1]) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            nx = np.where(A > 0, Ax / np.maximum(A, 1e-300), 0); ny = np.where(A > 0, Ay / np.maximum(A, 1e-300), 0)
        self.alpha = np.where(self.unknown & (A > 1e-14 * h), A / h**2, 0.0)
        self.supp = 0.5 * (np.abs(nx) + np.abs(ny))  # S(n)/h


def level_op(ax, ay, unknown, h, periodic, W):
    """5-point per-volume operator on an n x n level, identity rows at pinned cells."""
    n = unknown.shape[0]
    idx = lambda i, j: (i % n) * n + (j % n)
    rows, cols, vals = [], [], []
    diag = np.array(W, dtype=float).ravel().copy()
    for i in range(n):
        for j in range(n):
            if not unknown[i, j]:
                continue
            r = idx(i, j)
            for (a, ii, jj) in ((ax[i + 1, j], i + 1, j), (ax[i, j], i - 1, j), (ay[i, j + 1], i, j + 1), (ay[i, j], i, j - 1)):
                if a <= 0:
                    continue
                inside = 0 <= ii < n and 0 <= jj < n
                if inside or periodic:
                    if unknown[ii % n, jj % n]:
                        diag[r] += a / h**2; rows.append(r); cols.append(idx(ii, jj)); vals.append(-a / h**2)
                else:  # Dirichlet box face (value 0 at the face)
                    diag[r] += 2 * a / h**2
    for i in range(n):
        for j in range(n):
            if not unknown[i, j]:
                diag[idx(i, j)] = 1.0
    rows += list(range(n * n)); cols += list(range(n * n)); vals += list(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))


def transfers(n, periodic):
    """R: average of 4 children (nc^2 x n^2); P: cell-centred bilinear (n^2 x nc^2)."""
    nc = n // 2
    R = sp.lil_matrix((nc * nc, n * n)); P = sp.lil_matrix((n * n, nc * nc))
    for I in range(nc):
        for J in range(nc):
            for di in (0, 1):
                for dj in (0, 1):
                    R[I * nc + J, (2 * I + di) * n + 2 * J + dj] = 0.25
    for i in range(n):
        for j in range(n):
            I, J = i // 2, j // 2
            si = 1 if i % 2 else -1; sj = 1 if j % 2 else -1
            for (II, JJ, w) in ((I, J, 9 / 16), (I + si, J, 3 / 16), (I, J + sj, 3 / 16), (I + si, J + sj, 1 / 16)):
                if periodic:
                    P[i * n + j, (II % nc) * nc + JJ % nc] += w
                elif 0 <= II < nc and 0 <= JJ < nc:
                    P[i * n + j, II * nc + JJ] += w
    return R.tocsr(), P.tocsr()


def build_hierarchy(G, mode, nmin=4):
    n, h = G.n, G.h
    levels = []
    ax, ay, unk = G.ax, G.ay, G.unknown
    W0 = G.alpha / np.maximum(1.1 * G.supp * h, 1e-300) * (G.alpha > 0)
    A = level_op(ax, ay, unk, h, G.periodic, W0)
    L = 0
    while True:
        levels.append(dict(A=A, unk=unk.ravel(), n=n))
        if n // 2 < nmin:
            break
        R, P = transfers(n, G.periodic)
        levels[-1].update(R=R, P=P)
        nc, H = n // 2, h * 2 ** (L + 1)
        unk_c = unk.reshape(nc, 2, nc, 2).any(axis=(1, 3))
        if mode == "rap":
            Ac = (R @ A @ P).tolil()
            for k in np.nonzero(~unk_c.ravel())[0]:
                Ac.rows[k] = [k]; Ac.data[k] = [1.0]
            A = Ac.tocsr()
        else:
            axc = 0.5 * (ax[0::2, :][:, 0::2] + ax[0::2, :][:, 1::2])
            ayc = 0.5 * (ay[:, 0::2][0::2, :] + ay[:, 0::2][1::2, :])
            f = 2 ** (L + 1)
            if mode == "redisc-sL":
                s = 1.1 * G.supp * h * f
            else:
                s = 1.1 * G.supp * h
            Wf = np.where(G.alpha > 0, G.alpha / np.maximum(s, 1e-300), 0.0)
            Wc = Wf.reshape(nc, f, nc, f).mean(axis=(1, 3))
            A = level_op(axc, ayc, unk_c, H, G.periodic, Wc)
            ax, ay = axc, ayc
        unk = unk_c; n = nc; L += 1
    return levels


def colour_sets(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return [np.nonzero(((i % 2) * 2 + (j % 2)).ravel() == c)[0] for c in range(4)]


def smooth(A, x, b, sets, order):
    D = A.diagonal()
    for c in order:
        s = sets[c]
        x[s] += (b[s] - A[s] @ x) / D[s]
    return x


def vcycle(levels, l, b, singular):
    lv = levels[l]; A = lv["A"]; n = lv["n"]
    if "R" not in lv:
        if singular:
            x = np.linalg.lstsq(A.toarray(), b, rcond=None)[0]
            x[lv["unk"]] -= x[lv["unk"]].mean(); return x
        return spla.spsolve(A.tocsc(), b)
    sets = lv.setdefault("sets", colour_sets(n))
    x = np.zeros_like(b)
    for _ in range(2):
        x = smooth(A, x, b, sets, (0, 1, 2, 3))
    r = b - A @ x
    rc = lv["R"] @ r
    if singular:
        u = levels[l + 1]["unk"]; rc[u] -= rc[u].mean(); rc[~u] = 0
    ec = vcycle(levels, l + 1, rc, singular)
    x += lv["P"] @ ec
    x[~lv["unk"]] = 0
    for _ in range(2):
        x = smooth(A, x, b, sets, (3, 2, 1, 0))
    return x


def contraction(levels, singular, its=60):
    A = levels[0]["A"]; u = levels[0]["unk"]
    rng = np.random.default_rng(3)
    e = rng.standard_normal(A.shape[0]); e[~u] = 0
    hist = []
    for _ in range(its):
        if singular:
            e[u] -= e[u].mean()
        nrm = np.linalg.norm(e)
        e = e - vcycle(levels, 0, A @ e, singular)
        e[~u] = 0
        if singular:
            e[u] -= e[u].mean()
        hist.append(np.linalg.norm(e) / nrm)
        if not np.isfinite(hist[-1]) or hist[-1] > 1e6:
            return float("inf")
    return float(np.exp(np.mean(np.log(hist[-10:]))))


def bicgstab_its(levels, singular):
    A = levels[0]["A"]; u = levels[0]["unk"]
    rng = np.random.default_rng(5)
    b = np.where(u, 1 + 0.3 * rng.standard_normal(A.shape[0]), 0.0)
    if singular:
        b[u] -= b[u].mean()
    M = spla.LinearOperator(A.shape, matvec=lambda r: vcycle(levels, 0, r.copy(), singular))
    cnt = [0]
    x, info = spla.bicgstab(A, b, M=M, rtol=1e-10, maxiter=300, callback=lambda _: cnt.__setitem__(0, cnt[0] + 1))
    return cnt[0] if info == 0 else f"{cnt[0]}!"


def disc_union(centres, R, L):
    def f(x, y):
        d = np.inf
        for (cx, cy) in centres:
            for ox in (-L, 0, L):
                for oy in (-L, 0, L):
                    d = min(d, np.hypot(x - cx - ox, y - cy - oy) - R)
        return d
    return f


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    L = 1.0
    off = 0.37 / n  # off-lattice placement
    cases = {
        "box+disc R=L/4 (G1-like)": (disc_union([(0.5 + off, 0.5 + off)], 0.25, L), False, False),
        "periodic+disc R=L/8": (disc_union([(0.5 + off, 0.5 + off)], 0.125, L), True, False),
        "periodic+disc R=L/32": (disc_union([(0.5 + off, 0.5 + off)], 1 / 32, L), True, False),
        "periodic 4x4 dense (R=0.4 pitch)": (disc_union([((k + 0.5) / 4 + off, (m + 0.5) / 4 + off) for k in range(4) for m in range(4)], 0.1, L), True, False),
        "box, no solid": (lambda x, y: 1.0, False, False),
    }
    print(f"n = {n}, h = 1/{n}; contraction rho(I - B S) [BiCGStab its to 1e-10]")
    for name, (sdf, per, _) in cases.items():
        G = Geo(n, L, sdf, per)
        line = f"{name:34s}"
        for mode in ("redisc-sL", "avg-s0", "rap"):
            lv = build_hierarchy(G, mode)
            rho = contraction(lv, False)
            its = bicgstab_its(lv, False)
            line += f" | {mode}: {rho:6.3f} [{its}]"
        print(line, flush=True)
