"""2-D prototype, round 2: symmetric compact schemes for Neumann / Robin / Dirichlet on the disc.

Gates (D = R = 1): Robin eigenvalue  x J1(x) = Bi J0(x)  ->  mu = x^2   (Bi=0 Neumann j'11^2 is
the SECOND eigenvalue; Bi=inf is Dirichlet j01^2).

Schemes (all symmetric, <= 5-point rows):
  P    Papac-Gibou-Ratsch 2010: aperture FV, kappa mass, every kappa>0 cell, wall term k*A_w*u_P
  RF   aperture FV, kappa mass, wall flux A_w u_P / (d_centroid + 1/k)       (round 1)
  H*   "aperture + link": unknowns = centre-fluid cells; slivers (kappa>0, centre in solid) merged
       into their fluid-centred face neighbours (mass shared by aperture); fluid-fluid faces
       aperture-weighted; each axis link P->S crossing the wall carries the Gibou distance theta
       and a film over the wall area it represents, A_link = h |n_a|:
          G_link = 1 / (theta/D + 1/(k |n_a|))        (per unit depth, flux = G (u_P - u_s))
     H-k : mass = kappa_P + merged sliver shares
     H-1 : mass = 1 (point values, Gibou)
     Hf-1: same as H-1 but fluid-fluid faces fully open (pure Gibou + Robin links)
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, jnp_zeros
from disc_bc import Geo, assemble_fv, smallest_eig


def mu_ref(Bi):
    if Bi == 0:
        return jnp_zeros(1, 1)[0] ** 2
    if np.isinf(Bi):
        return jn_zeros(0, 1)[0] ** 2
    return brentq(lambda x: x * j1(x) - Bi * j0(x), 1e-9, 2.404825557695773) ** 2


def hybrid(G, k, mass, full_faces=False):
    h, n = G.h, G.n
    sc = G.sdf > 0
    X, Y = np.meshgrid(G.xc, G.yc, indexing="ij")
    rr = np.hypot(X, Y)
    nrm = (X / rr, Y / rr)
    # fluid-fluid faces
    ax = np.zeros_like(G.ax); ay = np.zeros_like(G.ay)
    ax[1:-1, :] = np.where(sc[1:, :] & sc[:-1, :], 1.0 if full_faces else G.ax[1:-1, :], 0)
    ay[:, 1:-1] = np.where(sc[:, 1:] & sc[:, :-1], 1.0 if full_faces else G.ay[:, 1:-1], 0)
    wallc = np.zeros((n, n))
    M = np.where(sc, G.kap, 0.0)
    for i, j in zip(*np.nonzero(sc)):
        for (di, dj, a_face, na) in ((1, 0, G.ax[i + 1, j], nrm[0]), (-1, 0, G.ax[i, j], nrm[0]),
                                     (0, 1, G.ay[i, j + 1], nrm[1]), (0, -1, G.ay[i, j], nrm[1])):
            if sc[i + di, j + dj]:
                continue
            th = max(G.sdf[i, j] / (G.sdf[i, j] - G.sdf[i + di, j + dj]), 1e-3)
            if k > 0:
                film = 0.0 if np.isinf(k) else 1.0 / (k * h * max(abs(na[i, j]), 1e-12))
                wallc[i, j] += 1.0 / (th + film) / h**2
    # merge slivers: each sliver's kappa shared among fluid-centred face neighbours by aperture
    sl = (G.kap > 1e-12) & ~sc
    for i, j in zip(*np.nonzero(sl)):
        nb = [(i + 1, j, G.ax[i + 1, j]), (i - 1, j, G.ax[i, j]),
              (i, j + 1, G.ay[i, j + 1]), (i, j - 1, G.ay[i, j])]
        w = np.array([a if sc[p, q] else 0.0 for p, q, a in nb])
        if w.sum() == 0:  # no fluid-centred face neighbour: give it to the nearest by distance
            w = np.array([1.0 if sc[p, q] else 0.0 for p, q, a in nb])
            if w.sum() == 0:
                continue
        w /= w.sum()
        for (p, q, _), ww in zip(nb, w):
            M[p, q] += ww * G.kap[i, j]
    K, _ = assemble_fv(G, ax, ay, sc, wall_coef=wallc)
    Md = M[sc] if mass == "kappa" else np.ones(sc.sum())
    return K, Md


def run(ND, off, Bi):
    G = Geo(ND, off); h = G.h
    cells = G.kap > 1e-12
    which = 1 if Bi == 0 else 0
    out = {}
    if not np.isinf(Bi):
        K, _ = assemble_fv(G, G.ax, G.ay, cells, wall_coef=np.where(cells, Bi * G.Aw / h**2, 0))
        out["P"] = smallest_eig(K, G.kap[cells], k=2)[which]
        if Bi > 0:
            wc = np.where(cells, G.Aw / (h * h * (np.maximum(G.dwall, 1e-3 * h) + 1 / Bi)), 0)
            K, _ = assemble_fv(G, G.ax, G.ay, cells, wall_coef=wc)
            out["RF"] = smallest_eig(K, G.kap[cells], k=2)[which]
    for tag, mass, full in (("H-k", "kappa", False), ("H-1", "one", False), ("Hf-1", "one", True)):
        K, M = hybrid(G, Bi, mass, full)
        out[tag] = smallest_eig(K, M, k=2)[which]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1); offs = rng.random((3, 2))
    NDs = [16, 32, 64, 128]
    for Bi in (0.0, 0.1, 1.0, 10.0, 100.0, np.inf):
        ref = mu_ref(Bi)
        res = {}
        for ND in NDs:
            for o in offs:
                for t, v in run(ND, o, Bi).items():
                    res.setdefault((t, ND), []).append(v / ref - 1)
        for t in sorted({t for t, _ in res}):
            line = f"Bi={Bi:6g} {t:5s}"; prev = None
            for ND in NDs:
                e = np.array(res[(t, ND)])
                em = np.sqrt(np.mean(e**2))
                line += f" | {ND:3d}: {np.mean(e):+9.2e}" + (f" p={np.log(prev/em)/np.log(2):4.1f}" if prev else "       ")
                prev = em
            print(line, flush=True)
