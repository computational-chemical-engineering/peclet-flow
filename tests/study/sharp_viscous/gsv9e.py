"""G-SV9 (e) (doc/vof_sharp_viscous_stress.md §11, third-pass amendment; WO-6e): the spectral gate.

The Arnoldi spectral radius of the PRODUCTION incremental step map on (u, v, w, p), colour frozen,
mean of p removed, Chebyshev pressure to rtol 1e-13, RB-GS to an update stop of 1e-13 with the
residual stop off (coordinator F1, 2026-11-01: 1e-15 sits below the RB-GS round-off floor at
D_hi 1e4 and hits the sweep cap; the linearity check certifies the stop):
`gsv9_split.py prod --incremental --pdriver cheb --mom-tol 1e-13`, one process per run.
Cases: the (b) plane (n(22.5, 22.5), offset 0.3, walls, exact fractions) at N = 16 and the (d)
film at N = 12, each with the four ratio pairs x D_hi in {0.1, 1, 10, 100, 1e4}, both modes.
PASS: rho <= 1 + 1e-9 for every 'sharp' run; 'averaged' is reported. A run whose linearity check
exceeds 1e-10, or whose solves cap (pressure 20000, momentum 200000 sweeps), is NOT JUDGED.

FIFTH PASS (§6.8 A, WO-A5; USER Q24 pending, default adopted) supersedes the momentum half above:
the momentum runs a FIXED N_A = 2000 RB-GS sweeps per component (`--mom-fixed 2000`, update and
residual stops off), and the momentum cap test is replaced by the doubling certificate printed on the
PROD line (dbl_u, dbl_p <= 1e-12 on x1 and x_s); a failure reruns the case at 2 N_A once, a second
failure is NOT JUDGED. The pressure half stays (max_p_it >= 20000 -> NOT JUDGED). Order: plane, then film.
ARPACK: ncv 40, maxiter 1000; an unconverged run is re-run at ncv 80; a 'sharp' run still
unconverged is decided by its (b)/(d) march verdict and listed (UNCONVERGED).

  nice -n 10 python gsv9e.py [--procs 10] [--build BUILD_DIR] [--pilot]   (one line per run + a verdict)
--pilot runs only the stiffest case (plane, D_hi 1e4, mu 1e-3 / rho 1e-4, sharp); linearity
> 1e-10 there stops the table (coordinator, 2026-11-01).
The momentum cap test is conservative: 'momentum_sweeps' is summed over the three components.
"""
import argparse
import os
import re
import subprocess
import sys
from multiprocessing.pool import ThreadPool

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = ((1e-2, 1e-3), (1e2, 1e3), (1e-3, 1e-4), (1e3, 1e4))
DS = (0.1, 1.0, 10.0, 100.0, 1e4)
GEOMS = (("plane", 16), ("film", 12))
PCAP = 20000  # the pressure half of the cap test (the momentum half: the doubling certificate)
N_A, DBL = 2000, 1e-12  # WO-A5: fixed RB-GS sweeps per component; the doubling certificate's bound
RX = re.compile(r"\| rho=(\S+) \|.*max_pcg_it=(\d+) lin=(\S+) .*conv=(\d) ncv=(\d+) max_momentum_sweeps=(\d+)"
                r" dbl_u=(\S+) dbl_p=(\S+) mom_fixed=(\d+) cpu_per_mv=(\S+)")


def prod(geom, N, mode, mu, rho, D, ncv, env, nfix=N_A):
    cmd = [sys.executable, os.path.join(HERE, "gsv9_split.py"), "prod", "--geom", geom, "--N", str(N),
           "--mode", mode, "--pair", f"{mu:g},{rho:g}", "--D", f"{D:g}", "--incremental", "--pdriver", "cheb",
           "--ncv", str(ncv), "--maxiter", "1000", "--mom-fixed", str(nfix)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = next((x for x in out.stdout.splitlines() if x.startswith("PROD")), None)
    if line is None:
        return None, (out.stderr.strip().splitlines() or ["no output"])[-1][:160]
    m = RX.search(line)
    return dict(rho=float(m.group(1)), pit=int(m.group(2)), lin=float(m.group(3)), conv=int(m.group(4)),
                ncv=int(m.group(5)), msw=int(m.group(6)), du=float(m.group(7)), dp=float(m.group(8)),
                nfix=int(m.group(9)), cpu=float(m.group(10))), line


def run(job):
    geom, N, mode, mu, rho, D, env = job
    nfix = N_A
    r, line = prod(geom, N, mode, mu, rho, D, 40, env, nfix)
    if r is not None and max(r["du"], r["dp"]) > DBL:  # doubling failed: rerun once at 2 N_A
        nfix = 2 * N_A
        r, line = prod(geom, N, mode, mu, rho, D, 40, env, nfix)
    dbl_ok = r is not None and max(r["du"], r["dp"]) <= DBL
    if r is not None and dbl_ok and not r["conv"]:
        r, line = prod(geom, N, mode, mu, rho, D, 80, env, nfix)
    tag = f"GSV9e {mode:8s} geom={geom} N={N} mu={mu:g} rho={rho:g} D_hi={D:g}"
    if r is None:
        return tag, "ERROR", line
    capped = r["pit"] >= PCAP  # the momentum half is the doubling certificate (WO-A5)
    if r["lin"] > 1e-10 or capped or not dbl_ok:
        v = "NOT_JUDGED"
    elif not r["conv"]:
        v = "UNCONVERGED"
    else:
        v = "PASS" if r["rho"] <= 1.0 + 1e-9 else "MISS"
    return tag, v, (f"rho={r['rho']:.6f} lin={r['lin']:.1e} ncv={r['ncv']} conv={r['conv']} "
                    f"max_p_it={r['pit']} max_momentum_sweeps={r['msw']} dbl_u={r['du']:.1e} dbl_p={r['dp']:.1e} "
                    f"N_A={r['nfix']} cpu_per_mv={r['cpu']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=10)
    p.add_argument("--build", default=os.environ.get("PYTHONPATH", ""))
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    env = dict(os.environ, PYTHONPATH=a.build, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    jobs = [(g, N, m, mu, r, D, env) for (g, N) in GEOMS for m in ("sharp", "averaged")
            for (mu, r) in PAIRS for D in DS]
    if a.pilot:
        jobs = [j for j in jobs if j[:6] == ("plane", 16, "sharp", 1e-3, 1e-4, 1e4)]
    res = []
    with ThreadPool(a.procs) as pool:
        for tag, v, info in pool.imap_unordered(run, jobs):
            print(f"{tag} {v} {info}", flush=True)
            res.append((tag, v))
    sh = [v for t, v in res if "sharp" in t]
    av = [v for t, v in res if "averaged" in t]
    cnt = lambda L, k: sum(1 for v in L if v == k)
    if cnt(sh, "MISS"):
        verdict = "MISS"
    elif cnt(sh, "PASS") == len(sh):
        verdict = "PASS"
    else:
        verdict = "NOT_JUDGED/UNCONVERGED-listed"
    print(f"GSV9e {verdict} sharp: {cnt(sh, 'PASS')} PASS, {cnt(sh, 'MISS')} MISS, {cnt(sh, 'NOT_JUDGED')} NOT_JUDGED, "
          f"{cnt(sh, 'UNCONVERGED')} UNCONVERGED, {cnt(sh, 'ERROR')} ERROR of {len(sh)}; averaged (reported): "
          f"{cnt(av, 'PASS')} <=1+1e-9, {cnt(av, 'MISS')} above, {len(av) - cnt(av, 'PASS') - cnt(av, 'MISS')} other")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
