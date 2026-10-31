import json, sys, numpy as np, collections
res = json.load(open(sys.argv[1])); by = collections.defaultdict(dict)
for a, e in res: by[tuple(a[:6])][a[6]] = e
rows = collections.defaultdict(list)
for k, d in sorted(by.items()):
    Ns = sorted(d); emax = np.array([d[n][0] for n in Ns]); emean = np.array([d[n][1] for n in Ns])
    pm = np.log(emax[:-1] / emax[1:]) / np.log(2); pl = np.log(emean[:-1] / emean[1:]) / np.log(2)
    rows[(k[0], k[3], k[4], k[5])].append((k[1], k[2], emax, pm, emean, pl))
for g, lst in rows.items():
    exact = all(r[2][-1] < 1e-11 for r in lst)
    pmin = min(np.min(r[3]) for r in lst); pmin_last = min(r[3][-1] for r in lst)
    plmin = min(r[5][-1] for r in lst); emx = max(r[2][-1] for r in lst)
    print("%-10s r=%-6g %-6s %-6s | maxerr(Nmax) worst %.2e | L_inf order last-pair min %5.2f all-pairs min %5.2f | L1 order last min %5.2f%s" % (g[0], g[1], g[2], g[3], emx, pmin_last, pmin, plmin, "  EXACT" if exact else ""))
