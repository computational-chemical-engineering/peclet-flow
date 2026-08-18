"""How many pressure unknowns does the CURRENT centre-based classification throw away, and what
does it do to fluid connectivity? Compares
  CURRENT : cell has a row iff its CENTRE is fluid; face open iff BOTH centres fluid
  FACE    : cell has a row iff >=1 of its 6 faces has a FLUID velocity point; face open iff that
            face point is fluid   (= the paper's rule: pressure lives wherever a fluid face needs
            a gradient)
Face sdf = mean of the two adjacent centres, exactly as the solver classifies it.
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

pack, R = sys.argv[1], float(sys.argv[2])
pk = np.load(pack)
box = np.asarray(pk["box"], float)
G = np.round(box * R).astype(int)
c = np.asarray(pk["centers"]) * R
r = np.asarray(pk["scales"]) * R
S = np.full(tuple(G), 1e30, np.float32)
ax = [np.arange(G[k]) + 0.5 for k in range(3)]
for sh in np.stack(np.meshgrid(*[[-1., 0., 1.]] * 3, indexing="ij"), -1).reshape(-1, 3):
    cs = c + sh * G
    keep = np.all((cs + (r + 3)[:, None] > 0) & (cs - (r + 3)[:, None] < G), axis=1)
    for (cx, cy, cz), rr in zip(cs[keep], r[keep]):
        i0, i1 = np.searchsorted(ax[0], [cx - rr - 3, cx + rr + 3])
        j0, j1 = np.searchsorted(ax[1], [cy - rr - 3, cy + rr + 3])
        k0, k1 = np.searchsorted(ax[2], [cz - rr - 3, cz + rr + 3])
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        d = np.sqrt((ax[0][i0:i1, None, None] - cx) ** 2 + (ax[1][None, j0:j1, None] - cy) ** 2
                    + (ax[2][None, None, k0:k1] - cz) ** 2) - rr
        np.minimum(S[i0:i1, j0:j1, k0:k1], d, out=S[i0:i1, j0:j1, k0:k1])

n = int(np.prod(G))
fluidC = S >= 0                                            # fluid-centred
# minus-face of cell i along axis a = mean(S[i-1], S[i]);  fluid face point <=> that mean >= 0
Fm = [0.5 * (np.roll(S, 1, axis=a) + S) >= 0 for a in range(3)]
nfaces = sum(Fm[a].astype(np.int8) + np.roll(Fm[a], -1, axis=a).astype(np.int8) for a in range(3))

hasface = nfaces > 0
print(f"grid {tuple(G)}  R={R:g}  spheres={len(c)}  cells={n/1e6:.2f}M")
print(f"  fluid-centred cells                     {fluidC.sum():9d}  ({100*fluidC.mean():.2f} %)")
print(f"  cells with >=1 fluid FACE               {hasface.sum():9d}  ({100*hasface.mean():.2f} %)")
print(f"  SOLID-centred but >=1 fluid face        {(hasface & ~fluidC).sum():9d}"
      f"   <- no pressure unknown today")
print(f"  SOLID-centred with >=2 fluid faces      {((nfaces >= 2) & ~fluidC).sum():9d}"
      f"   <- THROATS: real passages with NO continuity equation")
print(f"  fluid-centred with 0 fluid faces        {(fluidC & ~hasface).sum():9d}  (dead rows today)")

def comps(open_faces, active):
    """connected components of the pressure graph over `active` cells."""
    idx = -np.ones(n, np.int64)
    a = active.ravel()
    ii = np.nonzero(a)[0]
    idx[ii] = np.arange(len(ii))
    ID = np.arange(n).reshape(G)
    rows, cols = [], []
    for ax_ in range(3):
        m = open_faces[ax_] & active & np.roll(active, 1, axis=ax_)
        src = ID[m]
        dst = np.roll(ID, 1, axis=ax_)[m]
        rows.append(idx[src]); cols.append(idx[dst])
    rr = np.concatenate(rows); cc = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(ii),) * 2).tocsr()
    ncomp, lab = csg.connected_components(A, directed=False)
    big = np.bincount(lab).max()
    return ncomp, len(ii), len(ii) - big

cur_open = [np.roll(fluidC, 1, axis=a) & fluidC for a in range(3)]   # both centres fluid
nc, tot, orph = comps(cur_open, fluidC)
print(f"  CURRENT graph: {nc:6d} components over {tot} cells, {orph} cells outside the largest")
nc2, tot2, orph2 = comps(Fm, hasface)
print(f"  FACE    graph: {nc2:6d} components over {tot2} cells, {orph2} cells outside the largest")
