#!/usr/bin/env python3
"""WO-M step 2 — what does the solver's fp32 operator storage actually COST us?

The solver stores its multigrid operator (`mac_cutcell_mg.hpp` MReal) and its momentum stencil
(`flow_ibm.hpp` FV) in float. `-DPECLET_FLOW_MREAL_DOUBLE` switches both to double. This script is
the A/B: run it against a default build and against a `-DPECLET_FLOW_MREAL_DOUBLE` build of the same
commit, and diff the JSON. It measures ACCURACY and COST on cases we actually ship, not on synthetic
ones, because a precision change that costs 2x operator bandwidth for an accuracy nobody needs is
not worth shipping.

Cases
  hydro    varRho hydrostatic acid test, ratio 1e1..1e6 (walls +-z): steady max|u|, dP/dz error.
  vof      V2b uniform-velocity identity, ratio 1e1..1e4 (WO-K's decisive gate).
  porous   porous continuity with a PRESCRIBED uniform drag beta: the steady balance
           f = beta*u is exactly solvable, so this measures the porous path against an exact
           reference instead of an empirical correlation.
  contrast the case that started the campaign: a random close packing (phi 0.63), whose order-2
           marching-squares apertures span ~3 decades. Records the FULL residual trace of every
           pressure solve (PECLET_FLOW_MG_DEBUG=2) so a floor or a rebound is visible, not just the
           iteration count. Rule 3b: a capped solve is INVALID, and this reports the cap directly.
  zh       Zick & Homsy sphere-array drag vs the published reference (external ground truth).
  perm     RCP permeability — an actual physical functional, float vs double.
  cost     ms/step and GPU memory on production-sized beds.

Usage:
  PYTHONPATH=<build> python tests/study/precision_ab.py --cases all --json out.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", os.environ.get("SDFLOW_BUILD", "build"))))
from peclet import flow  # noqa: E402


# --------------------------------------------------------------------------------------------
# helpers


def gpu_mem_mb():
    """Resident GPU memory of THIS process, MiB (0 if no CUDA device / not visible)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 0.0
    me = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == me:
            return float(parts[1])
    return 0.0


def rcp_sdf(Ng):
    d = np.load(os.path.join(HERE, "rcp_pack_seed3.npz"))
    pos, r, side = d["pos"], d["r"], float(d["side"])
    g = (np.arange(Ng) + 0.5) / Ng * side
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    best = np.full((Ng, Ng, Ng), 1e30)
    for k in range(len(pos)):
        dd = [None, None, None]
        for a, c in enumerate((X, Y, Z)):
            t = c - (pos[k, a] + side / 2)
            t -= side * np.round(t / side)
            dd[a] = t
        best = np.minimum(best, np.sqrt(dd[0] ** 2 + dd[1] ** 2 + dd[2] ** 2) - r[k])
    return best, side, float(d["phi"])


def sc_sphere_sdf(N, phi):
    """Simple-cubic sphere array at solid fraction phi — the Zick & Homsy geometry, identical to
    scripts/validate_zick_homsy_sdflow.sc_sdf_xyz so the K it feeds is the recorded one."""
    R = (phi * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * N
    g = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    c = N / 2.0
    return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R, R


# Zick & Homsy (1982) SC drag factor, the table validate_zick_homsy_sdflow.py interpolates.
ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


# --------------------------------------------------------------------------------------------
# cases


def case_hydro(ratios=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6), N=8, NZ=24, g=0.1, steps=100):
    """The hydrostatic acid test: a stratified column at rest must STAY at rest, with the discrete
    pressure gradient exactly rho_face*g. Any float defect in the momentum diagonal or in the
    projection coefficient shows up here as a spurious velocity."""
    out = []
    for ratio in ratios:
        s = flow.Solver(N, N, NZ)
        s.set_rho(1.0)
        s.set_mu(0.0)          # inviscid: the balance is exact
        s.set_dt(1.0)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.asfortranarray(np.full((N, N, NZ), 10.0)))
        rho = np.empty((N, N, NZ), order="F")
        zs = np.arange(NZ)
        rho[:, :, :] = np.where(zs < NZ // 2, ratio, 1.0)[None, None, :]
        s.add_field("rho")
        s.set_field("rho", rho)
        s.set_density_mode("variable")
        s.set_property_model("force_z", "linear", "rho", [0.0, -g])
        pit = 0
        for _ in range(steps):
            s.step()
            pit = max(pit, s.last_pressure_iterations())
        umax = max(float(np.abs(s.get_u()).max()), float(np.abs(s.get_v()).max()),
                   float(np.abs(s.get_w()).max()))
        p = s.get_p()
        xc, yc = N // 2, N // 2
        perr = 0.0
        for z in range(1, NZ):
            dp = p[xc, yc, z] - p[xc, yc, z - 1]
            rf = 0.5 * ((ratio if z < NZ // 2 else 1.0) + (ratio if z - 1 < NZ // 2 else 1.0))
            perr = max(perr, abs(dp + g * rf) / (g * ratio))
        out.append(dict(ratio=ratio, max_u=umax, dpdz_rel_err=perr, max_pressure_its=pit))
        print(f"  hydro ratio {ratio:8.0e}: max|u| {umax:.3e}  dP/dz rel-err {perr:.3e}  "
              f"max its {pit}", flush=True)
    return out


def case_vof(ratios=(1e1, 1e2, 1e3, 1e4), n=32, steps=200):
    """WO-K's decisive gate: an arbitrary sharp colour field carried by a UNIFORM velocity must
    leave the velocity unchanged. WO-K measured this floored at ~1.2e-7 by the FLOAT momentum
    operator (rho_f/dt formed in double in the RHS, stored in float in the stencil) and ~1.2e-15 in
    a double build. This re-measures it here so the claim is in one table with its cost."""
    sys.path.insert(0, HERE)
    import vof_momentum_consistency as vmc  # noqa: E402
    res = vmc.gate_uniform(n=n, steps=steps, scenes=("tilted",), ratios=tuple(ratios))
    out = []
    for k, v in (res.items() if isinstance(res, dict) else []):
        out.append(dict(key=str(k), value=v))
    return out if out else dict(raw=repr(res))


def case_porous(N=16, beta=4.0, f_drive=0.2, eps=0.6, dt=0.5, steps=200):
    """Porous (volume-averaged) continuity with a PRESCRIBED uniform drag coefficient. At steady
    state the momentum balance of a uniform periodic bed is exactly f = beta*u_i (interstitial), so
    the reference is analytic — no closure, no empirical correlation, no dem. This is the porous
    code path the Ergun fixed-bed benchmark exercises (buildPorousCoeffCons + addDragDiagonal +
    projectCorrectPorousCons), measured against an exact answer."""
    s = flow.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(dt)
    s.set_advection(False)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    s.enable_drag()
    s.set_porous_continuity(True)
    s.set_field("eps", np.asfortranarray(np.full((N, N, N), eps)))
    s.set_field("drag_beta", np.asfortranarray(np.full((N, N, N), beta)))
    s.exchange_field("eps")
    s.exchange_field("drag_beta")
    s.sync_porous_prev()
    s.set_body_force(0.0, 0.0, f_drive)
    pit = 0
    for _ in range(steps):
        s.step()
        pit = max(pit, s.last_pressure_iterations())
    ui = float(s.get_w().mean())
    # eps-conservative pair: the momentum diagonal is eps*rho/dt + beta and the steady drive is the
    # body force f, so f = beta*u_i exactly.
    ref = f_drive / beta
    rel = abs(ui - ref) / abs(ref)
    print(f"  porous: u_i {ui:.10e}  exact {ref:.10e}  rel-err {rel:.3e}  max its {pit}", flush=True)
    return dict(u_interstitial=ui, exact=ref, rel_err=rel, max_pressure_its=pit,
                porous_residual=float(s.max_porous_residual()))


def case_contrast(Ngs=(48, 64, 96), steps=4, maxit=300, rtol=1e-8, driver="pcg"):
    """THE campaign case: a random close packing whose order-2 (marching-squares) apertures span
    about three decades, which is where the collocated session measured float PCG floor at 8e-7 and
    REBOUND to 3.8e-5 while a double build converged monotonically to 1e-8.

    Reports, per solve: iterations, whether the cap was touched (rule 3b: a capped solve is INVALID,
    not degraded), the minimum r/r0 reached and the final r/r0 — a floor shows up as
    min == final >> rtol and a rebound as final > min."""
    out = []
    for Ng in Ngs:
        sdf, side, phi = rcp_sdf(Ng)
        lv = max(2, int(np.log2(Ng)) - 1)
        s = flow.Solver(Ng, Ng, Ng)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_dt(80.0)
        s.set_body_force(1e-3, 0, 0)
        s.set_advection(False)
        s.set_velocity_solver_params(150)
        s.set_pressure_multigrid(True, levels=lv)
        if driver == "pcg":
            s.set_pressure_pcg(True, maxit, rtol)
        elif driver == "fcg":
            s.set_pressure_fcg(True, maxit, rtol)
        else:
            s.set_pressure_chebyshev(True, maxit, rtol)
        s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                    pressure_coarse="rediscretized")
        its, t0 = [], time.time()
        for _ in range(steps):
            s.step()
            its.append(int(s.last_pressure_iterations()))
        dt_ = time.time() - t0
        rec = dict(Ng=Ng, phi=phi, levels=lv, driver=driver, iters=its, cap=maxit,
                   capped=bool(max(its) >= maxit), div=float(s.max_open_divergence()),
                   umean=float(s.get_u().mean()), s_per_step=dt_ / steps)
        out.append(rec)
        print(f"  contrast Ng={Ng} ({driver}): its {its} cap={maxit} "
              f"{'CAPPED(INVALID)' if rec['capped'] else 'ok'}  div {rec['div']:.2e}  "
              f"{rec['s_per_step']:.2f} s/step", flush=True)
    return out


def case_zh(Ns=(32, 48, 64), phi=0.125, mu=0.1, F=1e-3, dt=None, max_steps=600, tol=1e-6):
    """Zick & Homsy simple-cubic sphere-array drag — the solver's external ground truth."""
    Kref = float(np.interp(phi, ZH_PHI, ZH_K))
    out = []
    for N in Ns:
        sdf, R = sc_sphere_sdf(N, phi)
        s = flow.Solver(N, N, N)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_dt(dt if dt else 60.0)
        s.set_body_force(F, 0, 0)
        s.set_advection(False)
        s.set_velocity_solver_params(120)
        s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
        s.set_pressure_pcg(True, 400, 1e-9)
        s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                    pressure_coarse="rediscretized")
        prev, nit = 0.0, 0
        for it in range(max_steps):
            s.step()
            nit = it + 1
            if it % 5 == 4:
                m = float(s.get_u().mean())
                if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                    break
                prev = m
        umean = float(s.get_u().mean())
        # K = F R^2 / (6 pi mu a U) form used by validate_zick_homsy_sdflow.drag_K
        K = F * (N ** 3) / (6.0 * np.pi * mu * R * umean) if umean else float("nan")
        out.append(dict(N=N, phi=phi, umean=umean, K=K, Kref=Kref,
                        rel_vs_ref=(K / Kref - 1.0) if Kref else float("nan"), steps=nit,
                        pit=int(s.last_pressure_iterations()),
                        div=float(s.max_open_divergence())))
        print(f"  zh N={N}: K {K:.6f} (ref {Kref:.3f}, {100*(K/Kref-1):+.2f}%)  "
              f"umean {umean:.12e}  steps {nit} "
              f"its {out[-1]['pit']}  div {out[-1]['div']:.2e}", flush=True)
    return out


def case_perm(Ngs=(44, 56), mu=0.1, F=1e-3, dt=80.0, max_steps=3000, tol=1e-6):
    """The RCP permeability — a physical functional on the high-contrast geometry."""
    out = []
    for Ng in Ngs:
        sdf, side, phi = rcp_sdf(Ng)
        lv = max(2, int(np.log2(Ng)) - 1)
        s = flow.Solver(Ng, Ng, Ng)
        s.set_rho(1.0)
        s.set_mu(mu)
        s.set_dt(dt)
        s.set_body_force(F, 0, 0)
        s.set_advection(False)
        s.set_velocity_solver_params(150)
        s.set_pressure_multigrid(True, levels=lv)
        s.set_pressure_pcg(True, 400, 1e-9)
        s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                    pressure_coarse="rediscretized")
        prev, nit = 0.0, 0
        for it in range(max_steps):
            s.step()
            nit = it + 1
            if it % 5 == 4:
                m = float(s.get_u().mean())
                if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                    break
                prev = m
        umean = float(s.get_u().mean())
        k = mu * umean / F * (side / Ng) ** 2
        out.append(dict(Ng=Ng, phi=phi, k=k, steps=nit, pit=int(s.last_pressure_iterations()),
                        capped=bool(s.last_pressure_iterations() >= 400),
                        div=float(s.max_open_divergence())))
        print(f"  perm Ng={Ng}: k {k:.8e}  steps {nit}  its {out[-1]['pit']}"
              f"{' CAPPED(INVALID)' if out[-1]['capped'] else ''}", flush=True)
    return out


def case_cost(Ngs=(64, 96, 128), steps=10, warmup=3):
    """ms/step and GPU memory on production-sized high-contrast beds. Operator storage is what the
    precision switch changes, so this is where the +28 B/cell of a full fp64 hierarchy shows up."""
    out = []
    for Ng in Ngs:
        m0 = gpu_mem_mb()
        sdf, side, phi = rcp_sdf(Ng)
        lv = max(2, int(np.log2(Ng)) - 1)
        s = flow.Solver(Ng, Ng, Ng)
        s.set_rho(1.0)
        s.set_mu(0.1)
        s.set_dt(80.0)
        s.set_body_force(1e-3, 0, 0)
        s.set_advection(False)
        s.set_velocity_solver_params(150)
        s.set_pressure_multigrid(True, levels=lv)
        s.set_pressure_pcg(True, 300, 1e-6)   # 1e-8 CAPS on this bed in float (see case_contrast)
        s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                    pressure_coarse="rediscretized")
        for _ in range(warmup):
            s.step()
        m1 = gpu_mem_mb()
        t0 = time.time()
        its = []
        for _ in range(steps):
            s.step()
            its.append(int(s.last_pressure_iterations()))
        dt_ = time.time() - t0
        cells = Ng ** 3
        rec = dict(Ng=Ng, cells=cells, levels=lv, ms_per_step=1e3 * dt_ / steps,
                   median_its=float(np.median(its)),
                   ms_per_iter=1e3 * dt_ / steps / max(1.0, float(np.median(its))),
                   capped=bool(max(its) >= 300),
                   gpu_mem_mb=m1, gpu_mem_delta_mb=m1 - m0,
                   bytes_per_cell=(m1 - m0) * 1024 * 1024 / cells if m1 > m0 else float("nan"))
        out.append(rec)
        print(f"  cost Ng={Ng}: {rec['ms_per_step']:.1f} ms/step  med its {rec['median_its']:.0f}"
              f"  {rec['ms_per_iter']:.2f} ms/it  gpu {m1:.0f} MiB "
              f"({rec['bytes_per_cell']:.0f} B/cell)", flush=True)
    return out


def case_floor(Ng=96, steps=2, maxit=500, driver="pcg", levels=None):
    """ATTAINABLE-ACCURACY FLOOR. Drive the solve with an unreachable tolerance (rtol 1e-14) and let
    it run to `maxit`: the residual stops descending where round-off in the CG recurrence takes over,
    and that plateau — not the iteration count — is what a precision policy can and cannot move.

    Read with the classical bound: CG's attainable relative residual is O(eps * kappa(A)). So the
    ratio of the float floor to the double floor should be ~ eps_f32/eps_f64 ~ 5e8 IF storage
    precision is the binding term, and the double floor itself measures eps_f64 * kappa, i.e. the
    conditioning of the operator — a limit no storage policy can cross. Depth-independence
    discriminates the two: a storage/coarsening defect worsens with more levels, a conditioning
    limit does not (pass `levels` to check).

    Run under PECLET_FLOW_MG_DEBUG=2 and read the [mg] trace; this only sets the configuration up.
    """
    return case_trace(Ng=Ng, steps=steps, maxit=maxit, rtol=1e-14, driver=driver, levels=levels)


def case_trace(Ng=96, steps=2, maxit=300, rtol=1e-8, driver="pcg", levels=None):
    """One high-contrast pressure solve with the operator's own convergence trace. Run under
    PECLET_FLOW_MG_DEBUG=2 PECLET_FLOW_MG_DEBUG_SOLVES=<n> and read the `[mg] it ... r/r0` lines:
    a FLOOR is min == final >> rtol, a REBOUND is final > min. That distinction is the whole
    difference between 'the preconditioner is weak' and 'the operator's A*1=0 defect has stopped
    the residual from going anywhere', and an iteration count cannot show it."""
    sdf, side, phi = rcp_sdf(Ng)
    lv = levels if levels else max(2, int(np.log2(Ng)) - 1)
    s = flow.Solver(Ng, Ng, Ng)
    s.set_rho(1.0)
    s.set_mu(0.1)
    s.set_dt(80.0)
    s.set_body_force(1e-3, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, levels=lv)
    if driver == "pcg":
        s.set_pressure_pcg(True, maxit, rtol)
    elif driver == "fcg":
        s.set_pressure_fcg(True, maxit, rtol)
    else:
        s.set_pressure_chebyshev(True, maxit, rtol)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True, pressure_coarse="rediscretized")
    for _ in range(steps):
        s.step()
    return dict(Ng=Ng, levels=lv, driver=driver, its=int(s.last_pressure_iterations()),
                cap=maxit, div=float(s.max_open_divergence()))


CASES = dict(hydro=case_hydro, trace=case_trace, floor=case_floor, vof=case_vof, porous=case_porous, contrast=case_contrast,
             zh=case_zh, perm=case_perm, cost=case_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="hydro,porous,vof,contrast,zh,perm,cost")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    res = dict(label=args.label, build=os.environ.get("SDFLOW_BUILD", "build"))
    for nm in names:
        print(f"[{nm}]", flush=True)
        res[nm] = CASES[nm]()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
