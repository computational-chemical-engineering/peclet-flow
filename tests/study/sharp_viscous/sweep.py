import mac2d as M, numpy as np, json, sys, itertools
from multiprocessing import Pool
def run(args):
    kind, ang, off, ratio, sch, base, N = args
    M.SHARP_BASE = base.split("+")[0]
    M.FCORR = "fit" if base.endswith("+J") else ("oracle" if base.endswith("+Jo") else None)
    c = M.Case(N, ang, off, ratio, 1.0, kind)
    try: e = M.solve_err(c, sch, "fit")
    except Exception as ex: e = (np.nan, np.nan)
    return args, e
if __name__ == "__main__":
    tag = sys.argv[1]
    kinds = sys.argv[2].split(","); angs = [float(a) for a in sys.argv[3].split(",")]
    offs = [float(o) for o in sys.argv[4].split(",")]; ratios = [float(r) for r in sys.argv[5].split(",")]
    schs = sys.argv[6].split(","); Ns = [int(n) for n in sys.argv[7].split(",")]
    jobs = []
    for kind, ang, off, ratio, s in itertools.product(kinds, angs, offs, ratios, schs):
        sch, base = (s.split(":") + ["arith"])[:2]
        for N in Ns: jobs.append((kind, ang, off, ratio, sch, base, N))
    jobs.sort(key=lambda j: -j[-1])
    with Pool(int(__import__("os").environ.get("NPROC", "40"))) as p: res = p.map(run, jobs, chunksize=1)
    json.dump([[list(a), list(map(float, e))] for a, e in res], open("res_%s.json" % tag, "w"))
    print("done", len(res))
