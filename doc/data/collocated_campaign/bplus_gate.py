# B+ spectral gate (fluid_only_constraint_plan.md): assemble the ghost operator A on the REAL bed,
# form the B+sym base M = sym(A), and measure (i) definiteness of M, (ii) the size of the skew
# defect, (iii) the extreme eigenvalues of the lagged-defect iteration's driver M^-1 A on the
# mean-free complement. PASS = M PSD (up to nullspace) and Re(spec(M^-1 A)) > 0 with a modest
# radius -- then CG-on-M + lagged skew converges with plain damping.
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, "/home/frankp/Codes/suite/flow/tests/study")
import ghost_projection_apriori as gp

N = int(os.environ.get("N", "48"))
BED = "/home/frankp/Codes/peclet-examples/benchmarks/porous-scaling/results/packings/packing_256x256x256_r16_phi0.60_s3.npz"
pk = np.load(BED)
box = float(np.asarray(pk["box"], float)[0])
C = np.asarray(pk["centers"]) / box - 0.5   # bed -> apriori box coords [-0.5, 0.5)
R = np.asarray(pk["scales"]) / box

def sdf_bed(x, y, z):
    S = np.full_like(x, 1e30)
    for sh in np.stack(np.meshgrid(*[[-1., 0., 1.]] * 3, indexing="ij"), -1).reshape(-1, 3):
        for (cx, cy, cz), rr in zip(C + sh, R):
            if cx < -0.75 or cx > 0.75:  # cheap cull (box is [-.5,.5))
                continue
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - rr
            np.minimum(S, d, out=S)
    return S

geo = gp.build_geo(N, sdf=sdf_bed, mode="center")
rho = gp.row_rescale(geo, order=2)
A = gp.assemble(geo, rho, order=2)
act = geo["active"].ravel()
ia = np.flatnonzero(act)
Aa = A[np.ix_(ia, ia)].tocsr()
# restrict to the largest connected component of the operator graph (pockets are decoupled
# physics with their own constants -- un-deflated they poison the generalized spectrum)
from scipy.sparse.csgraph import connected_components
sym_pattern = ((Aa != 0) + (Aa != 0).T).tocsr()
ncc, lab = connected_components(sym_pattern, directed=False)
big = np.argmax(np.bincount(lab))
keep = np.flatnonzero(lab == big)
Aa = Aa[np.ix_(keep, keep)].tocsr()
ia = ia[keep]
na = len(ia)
print(f"N={N} active rows={na} (components: {ncc}, largest kept)  nnz={Aa.nnz}")

Msym = (0.5 * (Aa + Aa.T)).tocsc()
K = (0.5 * (Aa - Aa.T)).tocsr()
print(f"skew/sym Frobenius: {spla.norm(K):.3e} / {spla.norm(Msym):.3e} "
      f"= {spla.norm(K)/spla.norm(Msym):.3e}; skew rows: "
      f"{int((np.abs(K).sum(axis=1) > 1e-12).sum())}")

# (i) definiteness of Msym: smallest eigenvalues (shift-invert)
lu0 = spla.splu(Msym + 1e-10 * sp.identity(na, format="csc"))
OPinv = spla.LinearOperator((na, na), matvec=lu0.solve)
w, V = spla.eigsh(Msym, k=16, sigma=0, which="LM", OPinv=OPinv)
o = np.argsort(w); w = w[o]; V = V[:, o]
print("smallest eig(sym(A)):", " ".join(f"{v:.3e}" for v in w))
neg = np.flatnonzero(w < -1e-9)
print(f"NEGATIVE eigenvalues: {len(neg)}  (must be 0 for B+sym viability)")
for j in neg:
    v = V[:, j]
    pr = (v**2).sum()**2 / (v**4).sum()
    top = np.argsort(-np.abs(v))[:5]
    print(f"  neg mode {w[j]:.3e}: participation {pr:.1f} cells; top-|v| rows carry "
          f"{(v[top]**2).sum():.2f} of mass")
# low-rank repair via Woodbury: M_fix = M_reg + U diag(tau - w_neg) U^T applied matrix-free
tau = 1e-2
if len(neg):
    U = V[:, neg]
    Dp = tau - w[neg]
    lu_reg = spla.splu(Msym + 1e-8 * sp.identity(na, format="csc"))
    Y = np.column_stack([lu_reg.solve(U[:, j]) for j in range(U.shape[1])])
    Sm = np.diag(1.0 / Dp) + U.T @ Y
    Sinv = np.linalg.inv(Sm)
    def mfix_solve(r):
        t = lu_reg.solve(r)
        return t - Y @ (Sinv @ (U.T @ t))
    lu0 = spla.LinearOperator((na, na), matvec=mfix_solve)
    lu0.solve = mfix_solve
    print(f"low-rank Woodbury fix applied (rank {len(neg)}, tau={tau})")

# (iii) spectrum of M^-1 A on the (roughly) mean-free complement
ones = np.ones(na) / np.sqrt(na)
def P(v):
    return v - ones * (ones @ v)
Lop = spla.LinearOperator((na, na), matvec=lambda v: P(lu0.solve(Aa @ P(v))))
for which, lab in (("LR", "max Re"), ("SR", "min Re"), ("LM", "max |.|")):
    try:
        ev = spla.eigs(Lop, k=6, which=which, maxiter=5000, tol=1e-6,
                       return_eigenvectors=False)
        ev = ev[np.argsort(-ev.real if which == "LR" else ev.real if which == "SR"
                           else -np.abs(ev))]
        print(f"{lab}(M^-1 A): " + " ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in ev[:6]))
    except Exception as e:
        print(f"{lab}: eigs failed: {e}")

# ---- B+kron gate: base = aperture star operator (SPD by construction) on the SAME DOF set ----
# apertures modelled from the face-point sdf (linear area-fraction model, adequate for spectra)
Sc = geo["Sc"]; Sf = geo["Sf"]; h = geo["h"]
fluid = Sc >= 0.0
alpha = [np.clip(0.5 + Sf[a] / h, 0.0, 1.0) for a in range(3)]  # alpha[a][i]: MINUS face of cell i
IDX = np.arange(N**3).reshape(N, N, N)
rows2, cols2, vals2 = [], [], []
def addm(r, c, v):
    rows2.append(r); cols2.append(c); vals2.append(v)
# filtered 7-point part: faces with both centers fluid
for a in range(3):
    am = np.roll(fluid, 1, axis=a)          # neighbor i-1 fluid?
    af = alpha[a]
    m = (fluid & am & (af > 0)).ravel()     # minus-face of cell i open in the filtered graph
    r = IDX.ravel()[m]; c = np.roll(IDX, 1, axis=a).ravel()[m]; w = af.ravel()[m]
    addm(r, r, w); addm(r, c, -w); addm(c, c, w); addm(c, r, -w)
# star part: eliminate each solid-centered cell with >=1 open fluid face
sol = ~fluid
nbr_idx = []
nbr_a = []
for a in range(3):
    ap = np.roll(alpha[a], -1, axis=a)       # PLUS face of cell i (minus face of i+1)
    fp = np.roll(fluid, -1, axis=a)          # +neighbor fluid?
    fm = np.roll(fluid, 1, axis=a)
    nbr_idx += [np.roll(IDX, -1, axis=a), np.roll(IDX, 1, axis=a)]
    nbr_a += [np.where(fp, ap, 0.0), np.where(fm, alpha[a], 0.0)]
D = sum(nbr_a)
star = sol & (D > 0)
sc = np.flatnonzero(star.ravel())
Dv = D.ravel()[sc]
for k1 in range(6):
    a1 = nbr_a[k1].ravel()[sc]; i1 = nbr_idx[k1].ravel()[sc]
    addm(i1, i1, a1)                        # + a_si on the diagonal of neighbor i
    for k2 in range(6):
        a2 = nbr_a[k2].ravel()[sc]; i2 = nbr_idx[k2].ravel()[sc]
        w = a1 * a2 / Dv
        m = w > 0
        addm(i1[m], i2[m], -w[m])           # - a_si a_sj / D  (includes k1==k2 self term)
Astar = sp.csr_matrix((np.concatenate(vals2),
                       (np.concatenate(rows2).astype(np.int64),
                        np.concatenate(cols2).astype(np.int64))), shape=(N**3, N**3))
Astar.sum_duplicates()
Ast = Astar[np.ix_(ia, ia)].tocsc()
print(f"\nB+kron base: nnz={Ast.nnz}  sym-check {spla.norm(Ast - Ast.T):.2e}")
w2 = spla.eigsh(Ast + 1e-10 * sp.identity(na, format='csc'), k=6, sigma=0, which='LM',
                OPinv=spla.LinearOperator((na, na), matvec=spla.splu(
                    Ast + 1e-8 * sp.identity(na, format='csc')).solve),
                return_eigenvectors=False)
print('smallest eig(A_star):', ' '.join(f'{v:.3e}' for v in np.sort(w2)))
lu2 = spla.splu(Ast + 1e-8 * sp.identity(na, format='csc'))
L2 = spla.LinearOperator((na, na), matvec=lambda v: P(lu2.solve(Aa @ P(v))))
for which, lab in (('LR', 'max Re'), ('SR', 'min Re'), ('LM', 'max |.|')):
    try:
        ev = spla.eigs(L2, k=6, which=which, maxiter=5000, tol=1e-6, return_eigenvectors=False)
        print(f'{lab}(Astar^-1 A): ' + ' '.join(f'{v.real:+.4f}{v.imag:+.4f}j' for v in ev))
    except Exception as e:
        print(f'{lab}: eigs failed: {e}')
