"""G-SV9(f), the onset (doc/vof_sharp_viscous_stress.md §6.5, WO-6f; reported, not gated).

The (b') periodic twin (m = (2,1,1), two plane families per period, N = 16) at
D_hi in {0.1, 0.3, 1, 3, 10, 30, 100}, both modes, the four G-SV9 ratio pairs, layer fraction 0.5 and
0.75, measured two ways:
  march   : gsv9.py --geom pplane (the production default solve, 2000 steps), judged on E' (the
            energy with the three translations removed, WO-6a') over windows with E_n >= 1e-20 E_0
            (USER Q15); a run is unstable iff that verdict is MISS (blow-up included).
  arnoldi : gsv9_split.py prod --geom pplane with the WO-6a' translation deflation, the E1/E5 setting
            of §6.5 (production u-only map, non-incremental, RB-GS update stop 1e-12 with the
            residual stop off, Chebyshev 1e-13, k 6, ncv 40, maxiter 1000); unstable iff
            rho > 1 + 1e-9 (the G-SV9(e) threshold). Linearity and ARPACK convergence are printed;
            lin > 1e-10 or conv = 0 is flagged.
Table: the smallest unstable D_hi per mode x pair x frac, for each measure. STOP rule (§6.5):
'sharp' unstable at D_hi <= 3 on any case where 'averaged' is stable -> architect.

  nice -n 10 python gsv9f.py arnoldi --out DIR [--procs 3]        (resumable: one file per case)
  python gsv9f.py table --march DIR --arnoldi DIR                 (march files: DIR/march_<mode>_f<frac>.out)
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PAIRS = ((1e-2, 1e-3), (1e2, 1e3), (1e-3, 1e-4), (1e3, 1e4))  # gsv9.PAIRS: (mu_A/mu_B, rho_A/rho_B)
FRACS = (0.5, 0.75)
MODES = ("sharp", "averaged")
ARN = ["prod", "--geom", "pplane", "--N", "16", "--pdriver", "cheb", "--mom-tol", "1e-12", "--k", "6",
       "--ncv", "40", "--maxiter", "1000"]


def lines(path):
    with open(path) as f:
        return f.readlines()


def cname(mode, pair, D, frac):
    return f"arn_{mode}_mu{pair[0]:g}_rho{pair[1]:g}_D{D:g}_f{frac:g}.out"


def one(job):
    out, mode, pair, D, frac = job
    path = os.path.join(out, cname(mode, pair, D, frac))
    if os.path.exists(path) and any(ln.startswith("PROD") for ln in lines(path)):
        return path, "done"
    cmd = [sys.executable, os.path.join(HERE, "gsv9_split.py")] + ARN + [
        "--mode", mode, "--pair", f"{pair[0]:g},{pair[1]:g}", "--D", f"{D:g}", "--frac", f"{frac:g}"]
    with open(path, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    return path, f"rc={rc}"


def cmd_arnoldi(a):
    os.makedirs(a.out, exist_ok=True)
    # cheapest first: small D_hi converges in fewer sweeps
    jobs = [(a.out, m, p, d, f) for d in DS for f in FRACS for m in MODES for p in PAIRS]
    with ThreadPoolExecutor(a.procs) as ex:
        for path, st in ex.map(one, jobs):
            ln = [x for x in lines(path) if x.startswith("PROD")]
            print(f"GSV9F-ARN {os.path.basename(path)} {st} " + (ln[0].split("|")[1].strip() if ln else "NO PROD LINE"),
                  flush=True)
    print("EXIT", flush=True)


def parse_march(d):
    res = {}
    for path in glob.glob(os.path.join(d, "march_*_f*.out")):
        mode, frac = re.match(r"march_(\w+?)_f([0-9.]+)\.out", os.path.basename(path)).groups()
        for ln in lines(path):
            m = re.match(r"GSV9b'(-AVG)? (\S+) geom=pplane muA/muB=(\S+) rhoA/rhoB=(\S+) D_hi=(\S+)", ln)
            if not m:
                continue
            v = m.group(2)
            fl = v.split("/")[1] if "/" in v else v  # the floor-excluded (judged, Q15) verdict
            res[(mode, (float(m.group(3)), float(m.group(4))), float(m.group(5)), float(frac))] = fl
    return res


def norm_pair(muR, rhoR):
    """gsv9.py prints mu_A/mu_B of its own normalisation; map to the PAIRS entry."""
    for p in PAIRS:
        if abs(p[0] - muR) <= 1e-9 * p[0] and abs(p[1] - rhoR) <= 1e-9 * p[1]:
            return p
    return (muR, rhoR)


def parse_arn(d):
    res = {}
    for mode in MODES:
        for p in PAIRS:
            for D in DS:
                for f in FRACS:
                    path = os.path.join(d, cname(mode, p, D, f))
                    if not os.path.exists(path):
                        continue
                    for ln in lines(path):
                        if ln.startswith("PROD"):
                            rho = float(re.search(r"\| rho=(\S+)", ln).group(1))
                            lin = float(re.search(r"lin=(\S+)", ln).group(1))
                            conv = int(re.search(r"conv=(\d)", ln).group(1))
                            res[(mode, p, D, f)] = (rho, lin, conv)
    return res


def cmd_table(a):
    mr = {(k[0], norm_pair(*k[1]), k[2], k[3]): v for k, v in parse_march(a.march).items()}
    ar = parse_arn(a.arnoldi)
    for k, v in sorted(ar.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        print(f"GSV9F-CASE arnoldi {k[0]} mu={k[1][0]:g} rho={k[1][1]:g} frac={k[3]:g} D_hi={k[2]:g} "
              f"rho={v[0]:.6f} lin={v[1]:.1e} conv={v[2]} march={mr.get(k, '-')}")

    def onset(get, mode, p, f):
        miss = [D for D in DS if get(mode, p, D, f) is None]
        uns = [D for D in DS if get(mode, p, D, f) is True]
        return (min(uns) if uns else None), miss

    gm = lambda mo, p, D, f: (None if (mo, p, D, f) not in mr else mr[(mo, p, D, f)] != "PASS")
    ga = lambda mo, p, D, f: (None if (mo, p, D, f) not in ar else ar[(mo, p, D, f)][0] > 1.0 + 1e-9)
    stop = []
    print("GSV9F-TABLE smallest unstable D_hi (march: E' floor-excluded MISS; arnoldi: rho > 1+1e-9); "
          "'none' = stable at every D_hi run; [missing D]")
    for f in FRACS:
        for p in PAIRS:
            cells = []
            for mo in MODES:
                for nm, g in (("march", gm), ("arnoldi", ga)):
                    o, miss = onset(g, mo, p, f)
                    cells.append(f"{mo}/{nm}={'none' if o is None else f'{o:g}'}" + (f"[{','.join(f'{x:g}' for x in miss)}]" if miss else ""))
            print(f"GSV9F-ONSET frac={f:g} mu={p[0]:g} rho={p[1]:g} " + " ".join(cells))
            for nm, g in (("march", gm), ("arnoldi", ga)):
                for D in DS:
                    if D <= 3.0 and g("sharp", p, D, f) is True and g("averaged", p, D, f) is False:
                        stop.append(f"{nm} frac={f:g} mu={p[0]:g} rho={p[1]:g} D_hi={D:g}")
    full = len(DS) * len(PAIRS) * len(FRACS) * len(MODES)
    print(f"GSV9F-STOPRULE {'STOP (architect): ' + '; '.join(stop) if stop else 'clear'} "
          f"(march {len(mr)}/{full}, arnoldi {len(ar)}/{full} cases)")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("arnoldi")
    r.add_argument("--out", required=True)
    r.add_argument("--procs", type=int, default=3)
    t = sp.add_parser("table")
    t.add_argument("--march", required=True)
    t.add_argument("--arnoldi", required=True)
    a = p.parse_args()
    return cmd_arnoldi(a) if a.cmd == "arnoldi" else cmd_table(a)


if __name__ == "__main__":
    sys.exit(main())
