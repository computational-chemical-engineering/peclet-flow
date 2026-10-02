"""Robin eigenproblem on the disc: -lap c = mu c, D dc/dn + k c = 0 (n out of the fluid), D=R=1.
Exact: sqrt(mu) = x with x J1(x) = Bi J0(x).  Variants:
  RF   : flux form - aperture FV, kappa mass, all kappa>0 cells, wall flux A_w c/(d_centroid + 1/k)
  RFc  : same, d measured from the cell CENTRE (sdf), clamped >= 0.05 h
  RG   : linear ghost on centre-fluid cells, c_Gamma eliminated through the normal Robin relation
         (diag += k|n_a| / (h (1+k d_n)) per crossing link; k->0 = staircase Neumann)
"""
import numpy as np, scipy.sparse as sp, sys
from scipy.optimize import brentq
from scipy.special import j0, j1
from disc_bc import Geo, assemble_fv, smallest_eig

def xref(Bi):
    f = lambda x: x * j1(x) - Bi * j0(x)
    return brentq(f, 1e-9, 2.4048)

def run(ND, off, Bi):
    G = Geo(ND, off); h = G.h; k = Bi
    cells = G.kap > 1e-12
    out = {}
    for tag, d in (("RF", G.dwall), ("RFc", np.maximum(G.sdf, 0.05 * h))):
        wc = np.where(cells, G.Aw / (h * h * (np.maximum(d, 1e-3 * h) + 1 / k)), 0)
        K, _ = assemble_fv(G, G.ax, G.ay, cells, wall_coef=wc)
        out[tag] = smallest_eig(K, G.kap[cells], k=1)[0]
    sc = G.sdf > 0
    n = G.n
    X, Y = np.meshgrid(G.xc, G.yc, indexing="ij")
    rr = np.hypot(X, Y); nx, ny = X / rr, Y / rr
    wallc = np.zeros((n, n))
    for i, j in zip(*np.nonzero(sc)):
        dn = G.sdf[i, j]
        for (di, dj, na) in ((1, 0, nx), (-1, 0, nx), (0, 1, ny), (0, -1, ny)):
            if not sc[i + di, j + dj]:
                th = max(dn / (dn - G.sdf[i + di, j + dj]), 1e-3)
                wallc[i, j] += (k * dn / th / h) / (h * (1 + k * dn))
    axb = np.zeros_like(G.ax); ayb = np.zeros_like(G.ay)
    axb[1:-1, :] = (sc[1:, :] & sc[:-1, :]); ayb[:, 1:-1] = (sc[:, 1:] & sc[:, :-1])
    K, _ = assemble_fv(G, axb, ayb, sc, wall_coef=wallc)
    out["RG"] = smallest_eig(K, np.ones(sc.sum()), k=1)[0]
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(1); offs = rng.random((3, 2))
    NDs = [16, 32, 64, 128]
    for Bi in (0.1, 1.0, 10.0, 100.0):
        ref = xref(Bi) ** 2
        res = {}
        for ND in NDs:
            for o in offs:
                for t, v in run(ND, o, Bi).items():
                    res.setdefault((t, ND), []).append(v / ref - 1)
        for t in ("RF", "RFc", "RG"):
            line = f"Bi={Bi:6g} {t:4s}"; prev = None
            for ND in NDs:
                e = np.sqrt(np.mean(np.square(res[(t, ND)])))
                line += f" | N={ND:3d} {e:8.2e}" + (f" p={np.log(prev/e)/np.log(2):4.1f}" if prev else "")
                prev = e
            print(line)
