"""G-SV9 (a) (doc/vof_sharp_viscous_stress.md §11): the 2-D harness amplification factor
rho(G), G = P (M+K)^-1 (M+K-A), of the CHOSEN split (K = mu(C) cells + face-fraction edges, A = the
sharp operator with the §5.7 interfacial-mean tau rule, TAU_RULE = "mixed-avg"), over the §6 table:
the six planar geometries of stab5.py, the mu/rho ratio pairs of §6 (plus 1e2/1e3 from the N = 32
check), D_hi = 0.1, 1, 10, 100, 1e4, at N = 16 and 32. PASS: rho(G) <= 1 + 1e-12 everywhere.

  nice -n 10 python gsv9a.py            (12 worker processes; one line per N x ratio, one verdict line)
"""
import sys
from multiprocessing import Pool

import numpy as np

import mac2d as M
from stab import blocks, rho_face

M.TAU_RULE = "mixed-avg"
GEOMS = ((0.0, 0.6), (0.0, 0.2), (22.5, 0.0), (22.5, 0.3), (45.0, 0.3), (67.5, 0.5))
PAIRS = ((1e-3, 1e-4), (1e-2, 1e-3), (1e2, 1e3), (1e3, 1e4))
DS = (0.1, 1.0, 10.0, 100.0, 1e4)


def one(args):
    N, muA, rA, ang, off = args
    case = M.Case(N, ang, off, muA, 1.0, "couette")
    rf = rho_face(case, *blocks(case, "facef", "arith", None, None)[3:], rA, 1.0)
    M.CELL_VOL = True
    K = blocks(case, "facef", "arith", None, {"xx", "yy", "yu", "xv"})[0]
    M.CELL_VOL = False
    At, G, D, uidx, m = blocks(case, "sharp", "facef", "fit", None)
    out = []
    for Dn in DS:
        dt = Dn * case.h**2 * (1.0 if muA < 1 else rA / muA)
        Mm = np.diag(rf / dt)
        Mi = np.diag(dt / rf)
        P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(D @ Mi @ G) @ D
        out.append(np.max(np.abs(np.linalg.eigvals(P @ np.linalg.solve(Mm + K, Mm + K - At)))))
    return args, out


def main():
    jobs = [(N, mu, r, a, o) for N in (16, 32) for (mu, r) in PAIRS for (a, o) in GEOMS]
    with Pool(12) as p:
        res = p.map(one, jobs)
    worst = {}
    for a, o in res:
        k = a[:3]
        worst[k] = np.maximum(worst.get(k, np.zeros(len(DS))), o)
    top = 0.0
    for k, v in sorted(worst.items()):
        top = max(top, float(np.max(v)))
        print("GSV9a N=%d muA/muB=%g rhoA/rhoB=%g worst rho(G) at D_hi=%s: %s"
              % (k[0], k[1], k[2], ",".join("%g" % d for d in DS), " ".join("%.4f" % x for x in v)))
    ok = top <= 1.0 + 1e-12
    print("GSV9a %s worst rho(G)=%.6f over %d geometries x %d ratio pairs x %d D_hi x N=16,32"
          % ("PASS" if ok else "MISS", top, len(GEOMS), len(PAIRS), len(DS)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
