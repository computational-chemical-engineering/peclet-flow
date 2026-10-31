"""G-SV9 (a) (doc/vof_sharp_viscous_stress.md §11, amended §6.1 / WO-5b): the 2-D harness
amplification factor rho(G), G = P (M+K)^-1 (M+K-A), of the PRODUCTION split Kb (2 mu(C) m_cc
implicit on interfacial cells, mu(C) m_cc implicit + mu(C) m_cc explicit elsewhere, face-fraction
edges; built by gsv9_split.harness_case), A = the sharp operator with the §5.7 interfacial-mean tau
rule, over the six planar geometries of stab5.py, the four mu/rho ratio pairs, D_hi = 0.1, 1, 10,
100, 1e4, at N = 16 and 32. PASS: rho(G) <= 1 + 1e-12 everywhere for Kb. Control: K1 (mu(C) split
evenly on every cell, the pre-WO-5b code) is printed and must exceed 1 at D_hi = 1e4 for every
N x ratio pair, proving the harness models what it claims.

  nice -n 10 python gsv9a.py            (12 worker processes; one line per N x ratio, one verdict line)
"""
import sys
from multiprocessing import Pool

import numpy as np

from gsv9_split import harness_case

GEOMS = ((0.0, 0.6), (0.0, 0.2), (22.5, 0.0), (22.5, 0.3), (45.0, 0.3), (67.5, 0.5))
PAIRS = ((1e-3, 1e-4), (1e-2, 1e-3), (1e2, 1e3), (1e3, 1e4))
DS = (0.1, 1.0, 10.0, 100.0, 1e4)


def one(args):
    N, muA, rA, ang, off = args
    out = harness_case(N, ang, off, muA, rA, DS, "sharp")
    return args, {nm: np.array([out[(nm, d)] for d in DS]) for nm in ("Kb", "K1")}


def main():
    jobs = [(N, mu, r, a, o) for N in (16, 32) for (mu, r) in PAIRS for (a, o) in GEOMS]
    with Pool(12) as p:
        res = p.map(one, jobs)
    worst = {}
    for a, o in res:
        for nm in ("Kb", "K1"):
            k = a[:3] + (nm,)
            worst[k] = np.maximum(worst.get(k, np.zeros(len(DS))), o[nm])
    top, ctl = 0.0, True
    for k, v in sorted(worst.items()):
        if k[3] == "Kb":
            top = max(top, float(np.max(v)))
        else:
            ctl = ctl and v[-1] > 1.0
        print("GSV9a N=%d muA/muB=%g rhoA/rhoB=%g %s worst rho(G) at D_hi=%s: %s"
              % (k[0], k[1], k[2], k[3], ",".join("%g" % d for d in DS), " ".join("%.4f" % x for x in v)))
    ok = top <= 1.0 + 1e-12
    print("GSV9a %s worst Kb rho(G)=%.6f over %d geometries x %d ratio pairs x %d D_hi x N=16,32; "
          "control K1 > 1 at D_hi=1e4 for every N x pair: %s"
          % ("PASS" if ok else "MISS", top, len(GEOMS), len(PAIRS), len(DS), "YES" if ctl else "NO"))
    return 0 if (ok and ctl) else 1


if __name__ == "__main__":
    sys.exit(main())
