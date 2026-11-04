"""G-SV2 -- Hadamard-Rybczynski drag of a buoyant drop in a periodic box (doc/vof_sharp_viscous_stress.md
§6.7, fourth pass, which binds; §11 G-SV2; WO-G2).

Box [0, L]^3, L = 8R (here L = 1, R = 1/8), triply periodic, isotropic h; d = 2R, d/h in
{8, 12, 16, 24, 32}, N = 4 d/h. Phase A (C = 1) is the sphere |x - x_c| < R, x_c = (L/2, L/2, L/2), a
cell vertex at every N. C_i is the exact z-chord average over 32 x 32 midpoint sub-columns of cell i;
cells with all 8 corners inside (outside) the closed (open) ball are exactly 1 (0) -- with the centre
on a vertex and R/h an integer this is exact integer arithmetic. The colour is frozen. sigma = 0;
rho = 1 in both phases; mu = LinearMix(C), mu_out = mu_B = 1, mu_A = kappa, kappa in {1e-2, 1, 1e2}
(kappa = 1 is the uniform-mu control of the forcing and the base scheme: 'sharp' is bitwise skipped
there, so it is run once). Force (Boussinesq, Hasimoto): the cell field f_z = drg (C_i - phi_h),
phi_h = mean C_i, written with set_field("force_z") in internal units; F = drg V_h, V_h = h^3 sum C_i.
drg = 1e-3 (h/dt) 9 mu_out K_HR / (2 R^2) (colour CFL ~1e-3 at the unbounded HR speed).
Steady Stokes (advection off) by gsv1.py's WO-6s GMRES(60) under the §6.6 rule (converged momentum on
every step, RB-GS update stop 1e-15 = gsv1.MOM_UTOL, coordinator decision 2026-11-01; rows taken at a
looser stop are void and rerun), from u = 0, p = 0; D_hi = mu_hi dt / (rho h^2) = 100, mu_hi = max(kappa, 1) mu_out;
certificate 1e-12 max|u|; cap 3000 matvecs (at the cap: NOT CONVERGED, not judged).
U_d = sum_i C_i wbar_i / sum_i C_i - <w>, wbar_i the mean of w on the -z and +z faces of cell i, <w> the
mean over all z-faces. K = F / (6 pi mu_out U_d R_h), R_h = (3 V_h / 4 pi)^(1/3).
PASS ('sharp', kappa = 1e-2 and 1e2): the unweighted nonlinear LS fit K(h) = K_inf + C h^p over the
five levels, p in [0.5, 4], gives p >= 1.8 and |K_inf - K_per| / K_per <= 3 %,
K_per = K_HR / (1 - 1.7601 K_HR phi^(1/3)), K_HR = (2 + 3 kappa) / (3 (1 + kappa)), phi = pi/384.
kappa = 1 must pass the same criteria, else G-SV2 is NOT JUDGED (instrument or base scheme).
Reported: K per level, the four-finest fit (12-32), 'averaged''s p and K_inf, matvecs per case.
Acceptance before the gate (WO-G2): kappa = 1, d/h 8 and 12: sum force = 0 to 1e-14 relative,
momentum invariant <= 1e-13, CFL <= 1.1e-3.

  nice -n 10 python gsv2.py run --doh 8,12,16 [--kappa 1e-2,1,1e2] [--modes sharp,averaged] --procs 3 --out F.jsonl
  python gsv2.py accept --out F.jsonl
  python gsv2.py table --out F.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsv1  # the WO-6s GMRES driver and the §6.6 momentum rule

L = 1.0
R = L / 8.0
DOH = (8, 12, 16, 24, 32)
KAPPAS = (1e-2, 1.0, 1e2)
SUB = 32
PHI = math.pi / 384.0


def k_hr(kappa):
    return (2.0 + 3.0 * kappa) / (3.0 * (1.0 + kappa))


def k_per(kappa):
    kh = k_hr(kappa)
    return kh / (1.0 - 1.7601 * kh * PHI ** (1.0 / 3.0))


def colour(N):
    """Exact z-chord average over SUB x SUB midpoint sub-columns; exact 0/1 by the corner test."""
    h = L / N
    xc = 0.5 * L
    off = (np.arange(SUB) + 0.5) / SUB * h
    g = (np.arange(N)[:, None] * h + off[None, :]).ravel()  # sub-column midpoints, cell-major
    X, Y = np.meshgrid(g, g, indexing="ij")
    s = np.sqrt(np.maximum(R * R - (X - xc) ** 2 - (Y - xc) ** 2, 0.0))
    lo, hi = xc - s, xc + s
    C = np.zeros((N, N, N))
    for k in range(N):
        zk = k * h
        ln = np.clip(np.minimum(zk + h, hi) - np.maximum(zk, lo), 0.0, h) / h
        C[:, :, k] = ln.reshape(N, SUB, N, SUB).mean(axis=(1, 3))
    # corner test in integer units of h: the centre is the vertex N/2, R/h = N/8 an integer
    r2 = (N // 8) ** 2
    q = np.arange(N + 1) - N // 2  # vertex offsets from the centre
    d2 = q[:, None, None] ** 2 + q[None, :, None] ** 2 + q[None, None, :] ** 2
    cmax = np.zeros((N, N, N), dtype=np.int64)
    cmin = np.full((N, N, N), np.iinfo(np.int64).max, dtype=np.int64)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                v = d2[a:a + N, b:b + N, c:c + N]
                cmax = np.maximum(cmax, v)
                cmin = np.minimum(cmin, v)
    C[cmax <= r2] = 1.0  # all 8 corners in the closed ball: the cell lies in it
    C[cmin >= r2] = 0.0  # nearest point of a cell to a vertex is a corner: no overlap with the open ball
    return np.asfortranarray(C)


def run_case(key):
    kappa, doh, mode = key
    os.environ.setdefault("OMP_PROC_BIND", "false")
    import peclet.flow as pf

    t0 = time.time()
    N = 4 * doh
    h = L / N
    mu_out = 1.0
    mu_hi = max(kappa, 1.0) * mu_out
    dt = gsv1.D_HI * h * h * 1.0 / mu_hi
    C = colour(N)
    phi_h = float(np.mean(C))
    Vh = h**3 * float(np.sum(C))
    Rh = (3.0 * Vh / (4.0 * math.pi)) ** (1.0 / 3.0)
    U_hr = 1e-3 * h / dt
    drg = U_hr * 9.0 * mu_out * k_hr(kappa) / (2.0 * R * R)
    F = drg * Vh
    s = pf.Solver((N, N, N), extent=(L, L, L))
    s.set_dt(dt)  # first: pins the internal time scale
    s.set_rho(1.0)
    s.set_mu(mu_out)
    s.set_pressure_geometry(np.full((N, N, N), 1.0, order="F"))
    s.enable_vof()
    s.set_vof(C)
    s.set_property_model("mu", "linear", "C", [mu_out, kappa * mu_out - mu_out])
    s.enable_cell_force()
    fi = np.atleast_1d(s.unit_scales["force_density_to_internal"])
    fz = drg * (C - phi_h)
    for c in range(3):
        sc = float(fi[c if fi.size == 3 else 0])
        s.set_field("force_" + "xyz"[c], np.asfortranarray(sc * fz if c == 2 else np.zeros((N, N, N))))
    s.set_advection(False)
    s.set_pressure_pcg(True, gsv1.PCAP, gsv1.PRTOL)  # uniform rho: the single-rank default driver
    s.diagnostics.set_vof_viscous_interface(mode)
    gsv1.converge_momentum(s)  # §6.6 rule
    s.update_properties()
    fr = np.asarray(s.get_field("force_z"))
    fsum_abs = abs(float(np.sum(fr))) / max(float(np.sum(np.abs(fr))), 1e-300)
    fsum_max = abs(float(np.sum(fr))) / max(float(np.max(np.abs(fr))), 1e-300)
    rho = np.ones((N, N, N))
    S = gsv1.StepMap(s, C, N, rho)
    nu = 3 * N**3
    x0 = np.zeros(4 * N**3)
    xs, status, steps, inc, cycles = gsv1.gmres_steady(S, x0, nu, U_hr)
    vsolver = s.diagnostics.velocity_solver()
    inc_def = gsv1.default_step_increment(S, xs, nu)
    w = xs[2 * N**3:3 * N**3].reshape((N, N, N), order="F")
    wbar = 0.5 * (w + np.roll(w, -1, axis=2))  # -z face of cell i is w[i]; +z face is w[i+1]
    Ud = float(np.sum(C * wbar)) / float(np.sum(C)) - float(np.mean(w))
    K = F / (6.0 * math.pi * mu_out * Ud * Rh) if Ud != 0.0 else float("nan")
    umax = float(np.max(np.abs(xs[:nu])))
    return {"kappa": kappa, "doh": doh, "N": N, "mode": mode, "status": status, "steps": steps, "cycles": cycles,
            "inc": inc, "K": K, "Ud": Ud, "U_hr": U_hr, "F": F, "Vh": Vh, "Rh": Rh, "phi_h": phi_h,
            "mom": S.mom, "fsum_rel_abs": fsum_abs, "fsum_rel_max": fsum_max, "cfl": umax * dt / h,
            "inc_default_step": inc_def, "velocity_solver": vsolver, "mom_utol": gsv1.MOM_UTOL,
            "max_p_it": S.max_pit, "max_momentum_sweeps": S.max_msw, "secs": round(time.time() - t0, 1),
            "rescaled": S.rescaled, "max_gain": S.max_gain, **({"mom_fixed": gsv1.MOM_FIXED} if gsv1.MOM_FIXED else {})}


def _run(key):
    try:
        return run_case(key)
    except RuntimeError as ex:  # a refused production step: NOT CONVERGED, listed, never judged
        kappa, doh, mode = key
        return {"kappa": kappa, "doh": doh, "N": 4 * doh, "mode": mode, "status": "step_error",
                "error": str(ex)[:200], "steps": -1, "K": float("nan"), "secs": 0.0}


def line(r):
    if r["status"] == "step_error":
        return f"GSV2CASE kappa={r['kappa']:g} d/h={r['doh']} {r['mode']} step_error {r.get('error', '')[:100]}"
    return (f"GSV2CASE kappa={r['kappa']:g} d/h={r['doh']} N={r['N']} {r['mode']} {r['status']} steps={r['steps']} "
            f"cycles={r['cycles']} inc={r['inc']:.1e} K={r['K']:.6f} Ud/U_hr={r['Ud'] / r['U_hr']:.6f} "
            f"mom={r['mom']:.1e} fsum(abs)={r['fsum_rel_abs']:.1e} fsum(max)={r['fsum_rel_max']:.1e} "
            f"cfl={r['cfl']:.2e} inc_def={r['inc_default_step']:.1e} vsolver={r['velocity_solver']} "
            f"max_p_it={r['max_p_it']} max_msw={r['max_momentum_sweeps']} {r['secs']}s")


def load(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            rows = f.readlines()
        for ln in rows:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                if r.get("mom_utol", 1.0) > gsv1.MOM_UTOL:  # taken at a looser update stop: void, rerun
                    continue
                out[(r["kappa"], r["doh"], r["mode"])] = r
    return out


def cmd_run(a):
    dohs = [int(v) for v in a.doh.split(",")]
    kap = [float(v) for v in a.kappa.split(",")]
    modes = a.modes.split(",")
    keys = []
    for k in kap:
        for d in dohs:
            for m in (("sharp",) if k == 1.0 else modes):  # kappa = 1: uniform mu, one control run
                keys.append((k, d, m))
    done = load(a.out)
    todo = [k for k in keys if k not in done]
    print(f"GSV2RUN total={len(keys)} todo={len(todo)} mom_utol={gsv1.MOM_UTOL:g}", flush=True)
    with Pool(a.procs, initializer=gsv1._set_mom_utol, initargs=(gsv1.MOM_UTOL, a.mom_fixed)) as pool, \
            open(a.out, "a") as f:
        for r in pool.imap_unordered(_run, todo, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(line(r), flush=True)
    print("EXIT", flush=True)


def fit(hs, Ks):
    from scipy.optimize import least_squares
    hs, Ks = np.asarray(hs, float), np.asarray(Ks, float)
    best = None
    for p0 in (1.0, 2.0, 3.0):
        x0 = [Ks[-1], (Ks[0] - Ks[-1]) / max(hs[0] ** p0 - hs[-1] ** p0, 1e-300), p0]
        r = least_squares(lambda x: x[0] + x[1] * hs ** x[2] - Ks, x0,
                          bounds=([-np.inf, -np.inf, 0.5], [np.inf, np.inf, 4.0]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.x[1]), float(best.x[2])


def cmd_accept(a):
    res = load(a.out)
    rows = [res.get((1.0, d, "sharp")) for d in (8, 12)]
    if any(r is None or r["status"] != "ok" for r in rows):
        print("GSV2-ACCEPT NOT_JUDGED (kappa = 1, d/h 8 and 12 not both converged)")
        return 1
    fa = max(r["fsum_rel_abs"] for r in rows)
    fm = max(r["fsum_rel_max"] for r in rows)
    mm = max(r["mom"] for r in rows)
    cf = max(r["cfl"] for r in rows)
    ok = fa <= 1e-14 and mm <= 1e-13 and cf <= 1.1e-3
    print(f"GSV2-ACCEPT kappa=1 d/h 8,12: sum f / sum|f| = {fa:.1e} (sum f / max|f| = {fm:.1e}) (<= 1e-14), "
          f"momentum invariant {mm:.1e} (<= 1e-13), CFL {cf:.2e} (<= 1.1e-3) {'PASS' if ok else 'MISS'}"
          + ("" if fm <= 1e-14 else " [the max-normalised force sum exceeds 1e-14]"))
    return 0 if ok else 1


def cmd_table(a):
    res = load(a.out)
    verdict = {}
    for kappa in KAPPAS:
        for mode in (("sharp",) if kappa == 1.0 else ("sharp", "averaged")):
            rows = [res.get((kappa, d, mode)) for d in DOH]
            ks = " ".join(f"{d}:{r['K']:.6f}" if r and r["status"] == "ok" else f"{d}:{r['status'] if r else '-'}"
                          for d, r in zip(DOH, rows))
            mv = " ".join(f"{r['steps']}" if r else "-" for r in rows)
            # WO-A5b: rows with a momentum component at the 200000-sweep cap, reported, NOT voiding (§6.8 A)
            mv += f" | momcap {sum(1 for r in rows if r and r.get('max_momentum_sweeps', 0) >= gsv1.MOM_CAP)}/{len(rows)}"
            kp = k_per(kappa)
            if all(r and r["status"] == "ok" for r in rows):
                hs = [1.0 / d for d in DOH]
                Ki, _, p = fit(hs, [r["K"] for r in rows])
                Ki4, _, p4 = fit(hs[1:], [r["K"] for r in rows[1:]])
                ok = p >= 1.8 and abs(Ki - kp) / kp <= 0.03
                ok4 = p4 >= 1.8 and abs(Ki4 - kp) / kp <= 0.03
                v = "PASS" if ok else ("MISS (four-level pass: Q10, the user rules)" if ok4 else "MISS")
                verdict[(kappa, mode)] = ok
                print(f"GSV2 kappa={kappa:g} {mode} K[d/h]={ks} | fit5: K_inf={Ki:.5f} p={p:.3f} "
                      f"dev={abs(Ki - kp) / kp * 100:.2f}% | fit4(12-32): K_inf={Ki4:.5f} p={p4:.3f} "
                      f"dev={abs(Ki4 - kp) / kp * 100:.2f}% | K_per={kp:.4f} | matvecs {mv} | "
                      f"{v if mode == 'sharp' else 'reported'}")
            else:
                verdict[(kappa, mode)] = None
                print(f"GSV2 kappa={kappa:g} {mode} K[d/h]={ks} | matvecs {mv} | NOT_JUDGED (levels missing or not converged)")
    c = verdict.get((1.0, "sharp"))
    g = [verdict.get((k, "sharp")) for k in (1e-2, 1e2)]
    if c is None or any(v is None for v in g):
        out = "NOT_JUDGED (levels missing or not converged)"
    elif not c:
        out = "NOT_JUDGED (kappa = 1 control misses: instrument or base scheme -> architect)"
    else:
        out = "PASS" if all(g) else "MISS"
    print(f"GSV2 VERDICT {out}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--doh", default="8,12,16")
    r.add_argument("--kappa", default="1e-2,1,1e2")
    r.add_argument("--modes", default="sharp,averaged")
    r.add_argument("--procs", type=int, default=3)
    r.add_argument("--out", required=True)
    r.add_argument("--mom-fixed", type=int, default=0)  # WO-A5b: fixed RB-GS sweeps per component (0 = off)
    for nm in ("accept", "table"):
        q = sp.add_parser(nm)
        q.add_argument("--out", required=True)
    a = p.parse_args()
    return {"run": cmd_run, "accept": cmd_accept, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
