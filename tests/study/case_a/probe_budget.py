#!/usr/bin/env python3
"""Case A G5 diagnosis, step 1: the discrete momentum budget behind the closure acceleration.

set_superficial_velocity applies, at the end of every step, one uniform shift s of u_x on every
face, i.e. a uniform ACCELERATION a = s/dt, which in the velocity form is the force density
rho_f * a. In the continuum the steady (translating) state has a = 0: the total momentum
P = sum_f rho_f u_f is constant and every other force sums to zero. Discretely

  P^{n+1} - P^n = sum rho^n_f (u* - u^n)            [predictor; the projection sums to zero]
                + sum (rho^{n+1}_f - rho^n_f) u^{n+1,pre}   [the density change, VoF]
                + sum rho^{n+1}_f s                 [the closure]

and the predictor part is dt * sum(-rho_f A(u) + f_csf + T_fullstress + f_grav). Per term, as a
mean force density in units of <rho> g (so that, at steady state, the terms sum to +a/g... with
the sign below: closure = -(sum of the others)):

  ADV   -sum rho_f A(u)        from a twin solver stepped once with advection toggled
                               (the solver default and the case-A runs: advection OFF; the twin
                               then measures what the missing term would contribute)
  VISC  sum T (transpose part)  from a twin stepped once with set_viscous_stress('laplacian')
  CSF   sum f_csf               diagnostics.vof_block_force (the field the RHS adds)
  GRAV  sum -(rho - <rho>) g    from C^n (zero by construction)
  DRHO  sum (rho^{n+1}-rho^n) u^{n+1,pre} / dt
  PRED  sum rho^n_f (u^{n+1,pre} - u^n) / dt   (= ADV + VISC + CSF + GRAV + solver residual)

The twins are bitwise copies of the main solver up to the probe step (checked). After the probe
the main solver alone runs on over --avg-crossings box crossings, recording PRED, CSF, DRHO,
GRAV and the closure every step; with --advection, ADV over the window = PRED - CSF - VISC - GRAV.

Also: the projection of each recorded force field on the buoyancy pattern (C - alpha), relative
to the buoyancy's own, i.e. the fraction by which the term adds to the effective buoyancy.

Usage (host, <= 4 threads):
  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=$PWD/build_host \\
      python tests/study/case_a/probe_budget.py --d-over-h 20 --t-probe 10 --out DIR
Prints one summary line; DIR/budget.csv per step, DIR/summary.txt.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_case_a as ca


def fields(s):
    return [np.array(g()) for g in (s.get_u, s.get_v, s.get_w)], np.array(s.get_vof())


SHIFT = [
    1
]  # inner u[i] pairs with cells i-1, i (rhoF = 0.5 (rf(i) + rf(i - strd)), buildRhsVar)


def face_rho(C, c):
    rho = ca.RHO_L + (ca.RHO_G - ca.RHO_L) * C
    return 0.5 * (rho + np.roll(rho, SHIFT[0], axis=c))


def face_avg(F, c):
    return 0.5 * (F + np.roll(F, 1, axis=c))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-over-h", type=int, default=20)
    p.add_argument("--t-probe", type=float, default=10.0)
    p.add_argument("--avg-crossings", type=float, default=2.0)
    p.add_argument("--stress", default="full")
    p.add_argument("--advection", action="store_true")
    p.add_argument(
        "--twin-tol",
        type=float,
        default=None,
        help="instead of toggling advection, the twin B runs the probe step at this "
        "momentum residual tolerance; TOL = PRED_A - PRED_B is the solve residual",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = round(ca.L_OVER_D * a.d_over_h)
    mk = lambda: ca.build(n, ca.MU_G, "arithmetic", a.stress, advection=a.advection)
    (A, _, alpha), (B, _, _), (Cs, _, _) = mk(), mk(), mk()
    rho_av = ca.RHO_L + (ca.RHO_G - ca.RHO_L) * alpha
    unit = rho_av * ca.G  # N/m^3: force densities reported in <rho> g
    buoy_c = lambda C: (
        (ca.RHO_L - ca.RHO_G) * ca.G * face_avg(C - alpha, 0)
    )  # +x on gas
    t, t_end = 0.0, a.t_probe * ca.T_REF
    step = 0
    while t < t_end:
        L_ = A.vof_step_limits()
        dt = 0.25 * min(L_["cfl_dt"], L_["capillary_dt"])
        for s in (A, B, Cs):
            s.set_dt(dt)
            s.step()
        t += dt
        step += 1
    # the internal force unit is fixed by the FIRST set_dt (tRef): read it only after stepping
    fi = A.unit_scales["force_density_to_internal"]
    uA, CA = fields(A)
    same = all(
        np.array_equal(x, y)
        for s in (B, Cs)
        for x, y in zip(uA + [CA], fields(s)[0] + [fields(s)[1]])
    )

    def probe(s, un, Cn, dt):
        s.set_dt(dt)
        s.step()
        u1, C1 = fields(s)
        sh = s.diagnostics.last_superficial_velocity_shift()
        u1[0] = u1[0] - sh
        pred = [np.sum(face_rho(Cn, c) * (u1[c] - un[c])) / dt for c in range(3)]
        drho = [
            np.sum((face_rho(C1, c) - face_rho(Cn, c)) * u1[c]) / dt for c in range(3)
        ]
        SHIFT[0] = -SHIFT[0]
        alt = [np.sum(face_rho(Cn, c) * (u1[c] - un[c])) / dt for c in range(3)]
        SHIFT[0] = -SHIFT[0]
        probe.alt = alt
        return pred, drho, u1, C1, sh

    L_ = A.vof_step_limits()
    dt = 0.25 * min(L_["cfl_dt"], L_["capillary_dt"])
    if a.twin_tol is None:
        B.set_advection(
            not a.advection
        )  # the twin toggles advection; ADV = +-(PRED_B - PRED_A)
    else:
        B.set_velocity_residual_tolerance(a.twin_tol)
    Cs.set_viscous_stress("laplacian")
    N = float(n**3)
    predB = probe(B, uA, CA, dt)[0]
    predB_alt = probe.alt
    csfB = [
        float(np.sum(np.array(B.diagnostics.vof_block_force(c)) / fi[c]))
        for c in range(3)
    ]
    resB_pair = [
        [(pb[c] - csfB[c]) / N / unit for c in range(3)] for pb in (predB, predB_alt)
    ]
    predC = probe(Cs, uA, CA, dt)[0]
    rows = []
    un, Cn = uA, CA
    n_avg = None
    k = 0
    P_hist = []
    while True:
        pred, drho, u1, C1, sh = probe(A, un, Cn, dt)
        csf_f = [np.array(A.diagnostics.vof_block_force(c)) / fi[c] for c in range(3)]
        csf = [float(np.sum(f)) for f in csf_f]
        grav = float(
            np.sum(-(ca.RHO_L - rho_av) * ca.G - (ca.RHO_G - ca.RHO_L) * ca.G * Cn)
        )
        P = float(np.sum(face_rho(Cn, 0) * un[0]))
        P_hist.append((k, P))
        # projections on the buoyancy pattern (x component): fraction of the buoyancy drive
        bx = buoy_c(Cn)
        nb = float(np.sum(bx * face_avg(Cn - alpha, 0)))
        wb = face_avg(Cn - alpha, 0)

        def proj(F, wb=wb, nb=nb):
            return float(np.sum(F * wb)) / nb

        clos_field = face_rho(C1, 0) * sh / dt
        rows.append(
            [k, dt, sh / dt / ca.G]
            + [x / N / unit for x in pred]
            + [x / N / unit for x in drho]
            + [x / N / unit for x in csf]
            + [
                grav / N / unit,
                proj(csf_f[0]),
                proj(clos_field),
                float(np.sum(face_rho(C1, 0) * sh)) / dt / N / unit,
            ]
        )
        if k == 0:
            sg = (1.0 if a.advection else -1.0) if a.twin_tol is None else 1.0
            adv0 = [sg * (pred[c] - predB[c]) / N / unit for c in range(3)]
            visc0 = [(pred[c] - predC[c]) / N / unit for c in range(3)]
            resB = [
                (predB[c] - csf[c] - grav * (c == 0) - (pred[c] - predC[c])) / N / unit
                for c in range(3)
            ]
            ub = float(np.sum(C1 * face_avg(u1[0], 0)) / np.sum(C1))
            n_avg = int(a.avg_crossings * ca.L_OVER_D * ca.D / max(ub, 1e-9) / dt)
        un = [x.copy() for x in u1]
        un[0] = un[0] + sh
        Cn = C1
        k += 1
        if k > n_avg:
            break
        L_ = A.vof_step_limits()
        dt = 0.25 * min(L_["cfl_dt"], L_["capillary_dt"])
    R = np.array(rows)
    hdr = (
        "k,dt,a_g,pred_x,pred_y,pred_z,drho_x,drho_y,drho_z,csf_x,csf_y,csf_z,grav,"
        "csf_proj_buoy,closure_proj_buoy,closure_force"
    )
    np.savetxt(os.path.join(a.out, "budget.csv"), R, delimiter=",", header=hdr)
    m = R.mean(axis=0)
    adv_mean = (m[3] - m[9] - visc0[0] - m[12]) if a.advection else 0.0
    P0, P1 = P_hist[0][1], P_hist[-1][1]
    dPdt = (P1 - P0) / (R[:, 1].sum()) / N / unit
    line = (
        f"BUDGET d/h={a.d_over_h} stress={a.stress} adv={a.advection} twin={'adv' if a.twin_tol is None else f'tol {a.twin_tol:g} (ADV(step0) := TOL)'} t_probe*={a.t_probe} twins_bitwise={same} "
        f"steps_avg={len(R)} [units <rho>g, x] closure a/g={m[2]:+.5e} "
        f"closure_force={m[15]:+.5e} PRED={m[3]:+.5e} DRHO={m[6]:+.5e} CSF={m[9]:+.5e} "
        f"GRAV={m[12]:+.2e} ADV(step0)={adv0[0]:+.5e} VISC(step0)={visc0[0]:+.3e} "
        f"ADV(win)={adv_mean:+.5e} residB(x,y,z)=({resB[0]:+.1e},{resB[1]:+.1e},{resB[2]:+.1e}) "
        f"predB-csfB(pair i-1,i)={resB_pair[0]} (pair i,i+1)={resB_pair[1]} "
        f"dP/dt={dPdt:+.2e} sum(PRED+DRHO+clos)={m[3] + m[6] + m[15]:+.2e} "
        f"pred_yz=({m[4]:+.1e},{m[5]:+.1e}) csf_yz=({m[10]:+.1e},{m[11]:+.1e}) "
        f"proj_buoy: CSF={m[13]:+.5f} closure={m[14]:+.5f} "
        f"step0: PRED={rows[0][3]:+.5e} CSF={rows[0][9]:+.5e} DRHO={rows[0][6]:+.5e}"
    )
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
