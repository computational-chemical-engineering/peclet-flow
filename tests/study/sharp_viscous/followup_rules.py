"""Architect follow-up (2026-10-31, note section 5.7): tau-averaging rules, fit normal on a curve, offset envelope.

  python followup_rules.py E1|E2|E3   -> res_followup_<tag>.json, then a summary line per group
  E1  planar Poiseuille, 11.25/22.5/33.75 deg, 8 offsets, mu ratio 1e-2/1e3, N 16-128; rules band-avg/mixed-avg/owner
  E2  circle (two-layer azimuthal Couette and body-force 'Poiseuille'), 4 radius offsets, mu ratio 1e-2/1e2/1e3,
      N 16-128; band-avg with fit normal nearest/mean/oracle, mixed-avg, owner
  E3  planar Poiseuille 33.75 deg, 16 offsets, mu ratio 1e2/1e3, N 16-256, band-avg (offset envelope)
'envelope' = max over offsets at each N; 'LS' = least-squares order over the N set.  NPROC (default 10) processes.
"""
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import mac2dc as X

M = X.M0
M.SHARP_BASE, M.FCORR = "facef", "fit"


def run(job):
    tag, kind, p1, p2, ratio, N, rule, fitrule = job
    t = time.time()
    if tag == "E2":
        c = X.CircleCase(N, -0.31, -0.27, 0.9 + p2 / N, ratio, 1.0, kind)
        g = X.CircleGeo(c, fitrule)
    else:
        c = M.Case(N, p1, p2, ratio, 1.0, kind)
        g = X.PlanarGeo(c)
    return list(job), list(X.solve_err(c, g, rule)), time.time() - t


def jobs(tag):
    out = []
    if tag == "E1":
        for rule in ("band-avg", "mixed-avg", "owner"):
            for ang in (11.25, 22.5, 33.75):
                for k in range(8):
                    for ratio in (0.01, 1000.0):
                        for N in (16, 32, 64, 128):
                            out.append(("E1", "poiseuille", ang, k / 8, ratio, N, rule, "-"))
    elif tag == "E2":
        rules = (("band-avg", "nearest"), ("band-avg", "mean"), ("band-avg", "oracle"), ("mixed-avg", "-"), ("owner", "-"))
        for kind in ("couette", "poiseuille"):
            for ratio in (0.01, 100.0, 1000.0):
                for k in range(4):
                    for N in (16, 32, 64, 128):
                        for rule, fr in rules:
                            out.append(("E2", kind, 0.0, k / 4, ratio, N, rule, fr))
    else:
        for ratio in (1000.0, 100.0):
            for k in range(16):
                for N in (16, 32, 64, 128, 256):
                    out.append(("E3", "poiseuille", 33.75, k / 16, ratio, N, "band-avg", "-"))
    return sorted(out, key=lambda j: -j[5])


def summary(res):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (tag, kind, p1, p2, ratio, N, rule, fr), e, _ in res:
        d[(kind, ratio, rule, fr)][(p1, p2)][N] = e[0]
    for grp in sorted(d):
        per = d[grp]
        Ns = sorted(next(iter(per.values())))
        env = [max(per[c][N] for c in per) for N in Ns]
        ls = lambda es: -np.polyfit(np.log(Ns), np.log(es), 1)[0]
        lso = [ls([s[N] for N in Ns]) for s in per.values() if max(s.values()) > 1e-10]
        print("%-40s per-offset LS min %.2f (%d/%d < 1.8) | e(Nmax) %.2e | envelope LS %.2f, e*N^2 %s" % (
            grp, min(lso), sum(o < 1.8 for o in lso), len(lso), env[-1], ls(env), " ".join("%.3g" % (e * N * N) for e, N in zip(env, Ns))))


if __name__ == "__main__":
    tag = sys.argv[1]
    with Pool(int(os.environ.get("NPROC", "10"))) as p:
        res = p.map(run, jobs(tag), chunksize=1)
    json.dump(res, open("res_followup_%s.json" % tag, "w"))
    summary(res)
