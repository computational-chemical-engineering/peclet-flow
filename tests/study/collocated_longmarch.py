#!/usr/bin/env python
"""Long-march probe of the STALL hypothesis (2026-08-20): the collocated march quasi-freezes on a
slow u-P manifold far from the true fixed point (measured: |uf - halfavg(u)| rms 4.5e-2 <u> at the
stopping point, while the ONLY true fixed point of the map has phi = 0, uf == halfavg(u), and a
dt-free steady system).  If that is the plateau mechanism, marching far past the <u> criterion
must show k creeping toward the staggered answer as the reconciliation gap decays (however
slowly); if k and the gap are EXACTLY stationary, the implemented map has a non-closing loop that
differs from the model and the discrepancy must be found in code.

Runs the phi=0.60 bed, collocated gauge-exact, for STEPS steps regardless of any criterion,
printing every EVERY steps: k (cell + flux estimators), m1 = rms|uf - halfavg(u)|/<|u|>,
m2 = rms alpha-div(halfavg(u))/<|u|>, and the drift of each since the last print.

  SDFLOW_BUILD=build_ge BED=...npz N=192 STEPS=20000 EVERY=500 DT=60 python collocated_longmarch.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402

BED = os.environ["BED"]
N = int(os.environ.get("N", "192"))
STEPS = int(os.environ.get("STEPS", "20000"))
EVERY = int(os.environ.get("EVERY", "500"))
DT = float(os.environ.get("DT", "60.0"))
KIND = os.environ.get("KIND", "gauge-exact")   # gauge-exact | gauge-2a | plain | stag
ROT = int(os.environ.get("ROT", "1"))          # 0 = PM I ablation (set_rotational_pressure(False))
ROTF = int(os.environ.get("ROTF", "0"))        # 1 = filtered rotational (set_rotational_filter)
ROTW = float(os.environ.get("ROTW", "1"))      # rotational under-relaxation w (set_rotational_weight)
WALLW = float(os.environ.get("WALLW", "0"))    # wall-banded blend w0 (set_rotational_wall_weight)
# DTSWITCH: comma list of "step:dt" pairs, e.g. "8000:600,14000:6" -- at the given step the
# solver's dt is changed in place (Frank's discriminator: a TRUE fixed point is dt-free, so any
# k motion after a switch proves the state was a stalled trajectory, not the fixed point).
DTSWITCH = dict((int(a), float(b)) for a, b in
                (kv.split(":") for kv in os.environ.get("DTSWITCH", "").split(",") if kv))
MU, F0 = 0.1, 1e-3


SHIFT = np.array([float(v) for v in os.environ.get("SHIFT", "0,0,0").split(",")])  # cells


def bed_sdf(N, npz):
    pk = np.load(npz)
    box = np.asarray(pk["box"], float)
    Rc = N / box[0]
    c = np.asarray(pk["centers"]) * Rc + SHIFT   # sub-cell translation (periodic; incidence probe)
    r = np.asarray(pk["scales"]) * Rc
    ax = np.arange(N) + 0.5
    S = np.full((N, N, N), 1e30)
    for sh in np.stack(np.meshgrid(*[[-1., 0., 1.]] * 3, indexing="ij"), -1).reshape(-1, 3):
        cs = c + sh * N
        keep = np.all((cs + (r + 3)[:, None] > 0) & (cs - (r + 3)[:, None] < N), axis=1)
        for (cx, cy, cz), rr in zip(cs[keep], r[keep]):
            i0, i1 = np.searchsorted(ax, [cx - rr - 3, cx + rr + 3])
            j0, j1 = np.searchsorted(ax, [cy - rr - 3, cy + rr + 3])
            k0, k1 = np.searchsorted(ax, [cz - rr - 3, cz + rr + 3])
            if i0 >= i1 or j0 >= j1 or k0 >= k1:
                continue
            d = np.sqrt((ax[i0:i1, None, None] - cx) ** 2 + (ax[None, j0:j1, None] - cy) ** 2
                        + (ax[None, None, k0:k1] - cz) ** 2) - rr
            np.minimum(S[i0:i1, j0:j1, k0:k1], d, out=S[i0:i1, j0:j1, k0:k1])
    return np.asfortranarray(np.clip(S, -1e3, 1e3)), Rc


sdf, R = bed_sdf(N, BED)
sv = flow.Solver if KIND == "stag" else flow.SolverColocated
s = sv(N, N, N)
s.set_rho(1.0); s.set_mu(MU); s.set_dt(DT)
s.set_body_force(F0, 0, 0); s.set_advection(False)
s.set_velocity_solver_params(int(os.environ.get("VIT", "150")))
if int(os.environ.get("APORDER", "1")) != 1:
    s.set_aperture_order(int(os.environ["APORDER"]))   # 2 = marching-squares apertures
s.set_pressure_multigrid(True, int(os.environ.get("MGL", "0")) or max(2, int(np.log2(N)) - 2))
s.set_pressure_pcg(True, 300, 1e-8)
if KIND != "stag":
    if KIND == "default":
        pass                                 # AUTO: whatever the shipped default resolves to
    elif KIND == "ghost":
        s.set_ghost_projection(True)         # fluid-only constraint + directional closures (route 2)
    elif KIND == "fluidonly":
        s.set_fluid_only_constraint(1)       # Design A: fluid-only openness filter + gauge-exact G
    elif KIND.startswith("fluidonly2"):
        s.set_fluid_only_constraint(2)       # Design B: SPD Kron star elimination + gauge-exact G
        if "_m" in KIND:                     # e.g. fluidonly2_m13: pair with another cell gradient
            s.set_face_interp(int(KIND.split("_m")[1]))
    elif KIND.startswith("mode"):
        s.set_face_interp(int(KIND[4:]))     # numbered ablations (e.g. mode3 = adjoint (T,T^T) pair)
    elif hasattr(s, "set_collocated_scheme"):
        s.set_collocated_scheme(KIND)
    else:
        s.set_face_interp({"gauge-exact": 9, "plain": 0}[KIND])
if not ROT:
    s.set_rotational_pressure(False)
if ROTF:
    s.set_rotational_filter(True)
if ROTW != 1.0:
    s.set_rotational_weight(ROTW)
if WALLW > 0:
    s.set_rotational_wall_weight(WALLW)
s.set_solid(sdf, cutcell_pressure=True, pressure_coarse="rediscretized")
fluid = sdf >= 0.0

print(f"# bed {os.path.basename(BED)} N={N} R={R:.1f} kind={KIND} rot={ROT} rotf={ROTF} rotw={ROTW} wallw={WALLW} dt={DT} steps={STEPS}", flush=True)
print(f"{'step':>7} {'k_cell/R2':>13} {'k_face/R2':>13} {'m1_rms':>10} {'m2_rms':>10} "
      f"{'dk_cell':>10} {'dm1':>10}", flush=True)
pk = pm = None
Pprev = None
for it in range(1, STEPS + 1):
    if it in DTSWITCH:
        s.set_dt(DTSWITCH[it])
        print(f"# --- dt -> {DTSWITCH[it]} at step {it} ---", flush=True)
    s.step()
    if it % EVERY == 0 or it == 50:
        U = [np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())]
        us = float(np.abs(U[0][fluid]).mean()) + 1e-300
        kc = float(U[0].mean()) * MU / F0 / R ** 2
        if KIND == "stag":
            m1 = 0.0
            OX = [np.asarray(s.get_ox()), np.asarray(s.get_oy()), np.asarray(s.get_oz())]
            kf = float((OX[0] * U[0]).mean()) * MU / F0 / R ** 2
            div = np.zeros_like(U[0])
            for a in range(3):
                flx = OX[a] * U[a]
                div += np.roll(flx, -1, axis=a) - flx
            m2 = float(np.sqrt((div[fluid] ** 2).mean())) / us
        else:
            UF = [np.asarray(s.get_uf()), np.asarray(s.get_vf()), np.asarray(s.get_wf())]
            OX = [np.asarray(s.get_ox()), np.asarray(s.get_oy()), np.asarray(s.get_oz())]
            kf = float((OX[0] * UF[0]).mean()) * MU / F0 / R ** 2
            m1sq = cnt = 0.0
            div = np.zeros_like(U[0])
            for a in range(3):
                half = 0.5 * (U[a] + np.roll(U[a], 1, axis=a))
                op = OX[a] > 0
                m1sq += float(((UF[a] - half)[op] ** 2).sum()); cnt += int(op.sum())
                flx = OX[a] * half
                div += np.roll(flx, -1, axis=a) - flx
            m1 = np.sqrt(m1sq / cnt) / us
            m2 = float(np.sqrt((div[fluid] ** 2).mean())) / us
        dk = "" if pk is None else f"{kc - pk:+.2e}"
        dm = "" if pm is None else f"{m1 - pm:+.2e}"
        Pf = np.asarray(s.get_p())
        dP = "" if Pprev is None else f"{np.abs(Pf - Pprev).max():.3e}"
        Pprev = Pf.copy()
        print(f"{it:>7} {kc:>13.7e} {kf:>13.7e} {m1:>10.3e} {m2:>10.3e} {dk:>10} {dm:>10} "
              f"dP={dP:>10} |P|={np.abs(Pf).max():.3e}", flush=True)
        pk, pm = kc, m1
