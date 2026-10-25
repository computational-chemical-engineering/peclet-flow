#!/usr/bin/env python3
"""Case A Q6 diagnosis, lead L2: the density the vof_momentum predictor puts on the advected momentum.

Under enable_vof_momentum the momentum rho^e u_e is carried on the half-shifted control volume (CV)
with its own colour C^e, and the advected velocity is uAdv = (rho^e u)/rho^e. The predictor then
builds its time term as rhoF * uAdv with rhoF = 0.5 (rho(C_i) + rho(C_i-1)), the arithmetic face
mean of the CELL densities at n+1 (flow_ibm_project.hpp buildRhsVarMom), not rho^e. Per step the
momentum the projection conserves, P_f = sum rhoF u, therefore changes by two reinterpretations:
  SEED  sum rho(C^{e,n}) u^n - sum rhoF^n u^n       (advection start: C^e clipped from the C^n planes)
  PRED  sum (rhoF^{n+1} - rho^{e,n+1}) uAdv         (predictor: the brief's L2 term)
and the advection itself conserves sum rho^e u exactly (periodic box). Both are reported as a mean
force density in units of <rho> g (sum / (n^3 dt <rho> g)), next to the closure's a/g, which must
cancel every net momentum source at steady state (G5 probe_budget.py; a/g = s / dt / g).

The march is the case-A driver's (run_case_a.build, full stress, single colour, vof_momentum).
PRED and SEED need only Python accessors (vof_advected_velocity, vof_momentum_diagnostics).
The FACE-WISE mismatch |rhoF - rho^e| / rho_l needs C^e, which is not exposed: each probe step is
replayed by vofmom_probe (C++, the same MomentumConsistentAdvector on the dumped C^n, u^n), and the
replay is accepted only if its uAdv and C^{n+1} match the solver's.

Usage (host):
  cmake -S tests/study/case_a/vofmom_probe -B build_vofmom_probe \\
        -DCMAKE_PREFIX_PATH=$PWD/../extern/install/host-openmp && cmake --build build_vofmom_probe
  OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build_host \\
      python tests/study/case_a/probe_vofmom_rho.py --d-over-h 20 --t-probe 12 --out DIR
Prints one summary line; DIR/steps.csv (one row per probe step), DIR/summary.txt.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_case_a as ca  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FLOW = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def rho_of(C):
    return ca.RHO_L + (ca.RHO_G - ca.RHO_L) * C


def face_rho(C, c):
    """The predictor's rhoF: inner u_c[i] is the low face of cell i, pairs cells i-1 and i."""
    r = rho_of(C)
    return 0.5 * (r + np.roll(r, 1, axis=c))


def put(path, a):
    np.asarray(a, dtype=np.float64).ravel(order="F").tofile(path)


def get(path, n):
    return np.fromfile(path, dtype=np.float64).reshape((n, n, n), order="F")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-over-h", type=int, default=20)
    p.add_argument("--t-probe", type=float, default=12.0, help="march length in sqrt(d/g)")
    p.add_argument("--n-probe", type=int, default=24, help="probe steps (a multiple of 6 sweeps)")
    p.add_argument("--muscl", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--probe-bin", default=os.path.join(FLOW, "build_vofmom_probe", "vofmom_probe"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = round(ca.L_OVER_D * a.d_over_h)
    s, h, alpha = ca.build(n, ca.MU_RATIO * ca.MU_L, "arithmetic", "full", single=True,
                           vof_mom=True, advection=True, vof_mom_muscl=a.muscl)
    rho_av = ca.RHO_L + (ca.RHO_G - ca.RHO_L) * alpha
    weps = s.diagnostics.vof_wisp_eps()
    t, step, t0 = 0.0, 0, time.perf_counter()

    def dt_rule():
        L_ = s.vof_step_limits()
        return 0.25 * min(L_["cfl_dt"], L_["capillary_dt"])

    t_end = a.t_probe * ca.T_REF
    while t < t_end:
        dt = dt_rule()
        s.set_dt(dt)
        s.step()
        t += dt
        step += 1
    t_march = time.perf_counter() - t0

    rows = []
    for k in range(a.n_probe):
        C0 = np.array(s.get_vof())
        U0 = [np.array(g()) for g in (s.get_u, s.get_v, s.get_w)]
        parity = s.diagnostics.vof_step_parity()
        dt = dt_rule()
        s.set_dt(dt)
        s.step()
        t += dt
        step += 1
        C1 = np.array(s.get_vof())
        # Both accessors hand out INTERNAL units (the raw uAdv_ field, the advector's census).
        us = s.unit_scales
        vti, dti = us["velocity_to_internal"], us["density_to_internal"]
        UA = [np.array(s.diagnostics.vof_advected_velocity(c)) / vti[c] for c in range(3)]
        sumM = [m_ / (dti * vti[c]) for c, m_ in
                enumerate(s.diagnostics.vof_momentum_diagnostics()["sum_momentum"])]
        shift = s.diagnostics.last_superficial_velocity_shift()
        unit = n**3 * dt * rho_av * ca.G           # sum of momentum density -> mean force / (<rho> g)
        Pf0 = float(np.sum(face_rho(C0, 0) * U0[0]))
        pred = float(np.sum(face_rho(C1, 0) * UA[0])) - sumM[0]
        seed = sumM[0] - Pf0

        # Replay for the face-wise C^e.
        d = os.path.join(a.out, "replay")
        os.makedirs(d, exist_ok=True)
        # the advector's (rho_c0, rho_c1) = rho(C=0), rho(C=1); C = 1 is the gas here
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{n} {dt / h!r} {parity} {ca.RHO_L!r} {ca.RHO_G!r} {weps!r} 0.25 {int(a.muscl)}\n")
        put(os.path.join(d, "C.bin"), C0)
        for c, nm in enumerate("uvw"):
            put(os.path.join(d, f"{nm}.bin"), U0[c])
        r = subprocess.run([a.probe_bin, d], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="8", OMP_PROC_BIND="false"))
        if r.returncode != 0:
            sys.exit(f"vofmom_probe failed: {r.stderr[-400:]}")
        Cn = get(os.path.join(d, "Cnew.bin"), n)
        ua = [get(os.path.join(d, f"ua{c}.bin"), n) for c in range(3)]
        Ce = [get(os.path.join(d, f"Ce{c}.bin"), n) for c in range(3)]
        Cs = [get(os.path.join(d, f"Ce{c}_seed.bin"), n) for c in range(3)]
        uscale = max(float(np.max(np.abs(UA[c]))) for c in range(3))
        err_ua = max(float(np.max(np.abs(ua[c] - UA[c]))) for c in range(3)) / uscale
        err_c = float(np.max(np.abs(Cn - C1)))
        row = dict(k=k, step=step, parity=parity, dt=dt, err_ua=err_ua, err_c=err_c,
                   pred=pred / unit, seed=seed / unit, a_g=shift / dt / ca.G)
        for c in range(3):
            m = (face_rho(C1, c) - rho_of(Ce[c])) / ca.RHO_L       # PRED mismatch, per face
            ms = (face_rho(C0, c) - rho_of(Cs[c])) / ca.RHO_L      # SEED mismatch, per face
            nz = np.abs(m) > 1e-12
            row[f"max_m{c}"] = float(np.max(np.abs(m)))
            row[f"mean_m{c}"] = float(np.mean(np.abs(m[nz]))) if nz.any() else 0.0
            row[f"nface{c}"] = int(nz.sum())
            row[f"max_ms{c}"] = float(np.max(np.abs(ms)))
            # the replay's own PRED / SEED sums (cross-check of the Python-only ones)
            row[f"pred_r{c}"] = float(np.sum(m * ca.RHO_L * UA[c])) / unit
            row[f"seed_r{c}"] = float(np.sum(-ms * ca.RHO_L * U0[c])) / unit
        # Buoyancy projection (as probe_budget.py): the step's net reinterpretation force field
        # f = [SEED + PRED] per face / dt against the buoyancy -(rho_f - <rho>) g, both weighted
        # by the face gas pattern (C_f - alpha): the fraction by which it adds to the buoyancy.
        Cf = 0.5 * (C1 + np.roll(C1, 1, axis=0))
        f_x = ((face_rho(C1, 0) - rho_of(Ce[0])) * UA[0] + (rho_of(Cs[0]) - face_rho(C0, 0)) * U0[0]) / dt
        b_x = -(face_rho(C1, 0) - rho_av) * ca.G
        row["buoy_frac"] = float(np.sum(f_x * (Cf - alpha)) / np.sum(b_x * (Cf - alpha)))
        rows.append(row)

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "steps.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r_ in rows:
            f.write(",".join(repr(r_[k_]) for k_ in keys) + "\n")
    M = {k_: np.array([r_[k_] for r_ in rows]) for k_ in keys}
    line = (f"VOFMOM_RHO d/h={a.d_over_h} n={n} muscl={a.muscl} t*={t / ca.T_REF:.3f} steps={step} "
            f"probe={a.n_probe} replay_err ua={M['err_ua'].max():.1e} C={M['err_c'].max():.1e} | "
            f"PRED={M['pred'].mean():+.4e} (replay {M['pred_r0'].mean():+.4e}) "
            f"SEED={M['seed'].mean():+.4e} (replay {M['seed_r0'].mean():+.4e}) "
            f"PRED+SEED={(M['pred'] + M['seed']).mean():+.4e} a/g={M['a_g'].mean():+.4e} "
            f"buoy_frac={M['buoy_frac'].mean():+.3e} (sd {M['buoy_frac'].std():.1e}) "
            f"[<rho>g] | face |rhoF-rho^e|/rho_l x: max {M['max_m0'].max():.3e} "
            f"mean(nonzero) {M['mean_m0'].mean():.3e} faces {M['nface0'].mean():.0f}/{n**3} "
            f"seed max {M['max_ms0'].max():.3e}; y,z PRED {M['pred_r1'].mean():+.1e},"
            f"{M['pred_r2'].mean():+.1e} | march {t_march:.0f}s")
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
