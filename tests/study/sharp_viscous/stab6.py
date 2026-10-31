import mac2d as M, numpy as np, sys
from stab import blocks, rho_face
from multiprocessing import Pool
N = int(sys.argv[1])
def one(args):
    muA, rA, ang, off = args
    case = M.Case(N, ang, off, muA, 1.0, "couette")
    rf = rho_face(case, *blocks(case, "facef", "arith", None, None)[3:], rA, 1.0)
    M.CELL_VOL = True; K = blocks(case, "facef", "arith", None, {"xx", "yy", "yu", "xv"})[0]; M.CELL_VOL = False
    At, G, D, uidx, m = blocks(case, "sharp", "facef", "fit", None)
    out = []
    for Dn in (0.3, 3.0, 100.0, 1e5):
        dt = Dn * case.h**2 * (1.0 if muA < 1 else rA / muA); Mm = np.diag(rf / dt); Mi = np.diag(dt / rf)
        P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(D @ Mi @ G) @ D
        out.append(np.max(np.abs(np.linalg.eigvals(P @ np.linalg.solve(Mm + K, Mm + K - At)))))
    return args, out
jobs = [(mu, r, a, o) for (mu, r) in ((1e-3, 1e-4), (1e-2, 1e-3), (1e2, 1e3), (1e3, 1e4)) for (a, o) in ((0, 0.6), (11.25, 0.1), (22.5, 0.3), (33.75, 0.85), (45, 0.5))]
with Pool(20) as p: res = p.map(one, jobs)
import collections; w = collections.defaultdict(lambda: np.zeros(4))
for a, o in res: w[(a[0], a[1])] = np.maximum(w[(a[0], a[1])], o)
for k, v in w.items(): print("N=%d muA/muB=%g rhoA/rhoB=%g worst rho(G) at D_hi=0.3,3,100,1e5: %s" % (N, k[0], k[1], " ".join("%.4f" % x for x in v)))
