#!/usr/bin/env python3
"""VoF rung V8 (WO-T) — the staggered/collocated physics battery, both columns.

The collocated (`SolverColocated`) path is the **ABC approximate projection**: average the cell
velocities onto a MAC face field, project THAT exactly, correct the cell field.  Rung V8 adds

  * variable density — the face coefficient ``c_f = o_f rho0/rho_f`` and ``projectCorrectVar`` on the
    FACE field, with the CELL correction the AVERAGE OF THE TWO FACE CORRECTIONS of each axis;
  * forces as FACE accelerations ``a_f = dt (f_f - (P(i)-P(i-s)))/rho_f`` added after
    ``centerToFace``, the cell taking the average of the two faces' total increment (Basilisk
    ``centered.h``; Popinet JCP 2009 §3);
  * colour advection from ``uf_/vf_/wf_``, the field the approximate projection makes exactly
    divergence-free.

What this script measures, staggered vs collocated:

  G3  static droplet with the COMPUTED curvature — the spurious capillary number
      ``Ca = mu max|u| / sigma`` at D/dx = 16 (and 8, 24).
  G4  Hysing case 1 (density ratio 10) — max rise velocity and y_c(3).  The collocated column runs
      WITHOUT momentum consistency (`enable_vof_momentum` is staggered-only at this rung — the
      collocated construction needs Favre face states), so the honest staggered comparison is the
      one with it OFF; both staggered columns are printed.
  G5  capillary wave at 32 cells/lambda — the frequency against the exact viscous two-fluid mode.

Every gate records the pressure iteration count against its cap (preamble rule 3b: a capped solve
makes the run INVALID) and, on the collocated grid, BOTH the cell max|u| and the FACE max|uf|: the
cell field carries the approximate projection's invisible checkerboard, which `centerToFace`
annihilates and the projection therefore cannot remove.

Usage:  PYTHONPATH=<build> python tests/study/vof_collocated.py [static|hysing1|wave] [--quick]
"""
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vof_capillary_references import wave_mode  # noqa: E402
from vof_surface_tension import (HYSING, Scale, Solve, cylinder_fractions,  # noqa: E402
                                 maxvel, sphere_fractions, zero_crossings)

QUICK = "--quick" in sys.argv
GATES = [a for a in sys.argv[1:] if not a.startswith("--")]

GRIDS = (("staggered ", pf.Solver), ("collocated", pf.SolverColocated))


def maxfacevel(s):
    return max(np.abs(s.get_uf()).max(), np.abs(s.get_vf()).max(), np.abs(s.get_wf()).max())


# --------------------------------------------------------------------------- G3 static droplet
def gate_static():
    print("\n" + "=" * 100)
    print("G3  STATIC DROPLET, computed curvature — Ca = mu max|u| / sigma, both grids")
    print("=" * 100)
    print("  The balanced-force identity is exact on both grids (the ctest's T2: the FACE field")
    print("  stays at machine zero with a constant kappa). What is left is the curvature error.")
    print("  On the collocated grid the CELL number additionally carries the invisible")
    print("  checkerboard, so both are printed.\n")
    rungs = [(16, 4.0), (32, 8.0)] if QUICK else [(16, 4.0), (32, 8.0), (48, 12.0)]
    for n, R in rungs:
        row = {}
        for label, cls in GRIDS:
            s = cls(n, n, n)
            s.set_rho(1.0)
            s.set_mu(0.1)
            s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
            s.set_pressure_chebyshev(True, 500, 1e-14)
            s.enable_vof()
            s.set_vof(sphere_fractions((n, n, n), R, (n / 2 + 0.13, n / 2 + 0.27, n / 2 + 0.11)))
            s.set_property_model("rho", "linear", "C", [1.0, 0.0])
            s.set_surface_tension(1.0)
            s.set_dt(0.5 * s.capillary_dt())
            h = Solve(500)
            for _ in range(60):
                s.step()
                h.sample(s)
            ca = 0.1 * maxvel(s) / 1.0
            caf = 0.1 * maxfacevel(s) / 1.0
            row[label] = (ca, caf, h)
            print(f"  D/dx = {2*R:5.1f}  {label}  Ca(cell) = {ca:.3e}   Ca(face) = {caf:.3e}")
            print(f"                          {h}")
        st, co = row["staggered "][0], row["collocated"][0]
        print(f"  D/dx = {2*R:5.1f}  collocated / staggered (cell Ca) = {co/st:.2f} x\n")


# --------------------------------------------------------------------------- G5 capillary wave
def gate_wave():
    print("\n" + "=" * 100)
    print("G5  CAPILLARY WAVE, 32 cells/lambda — omega against the EXACT viscous two-fluid mode")
    print("=" * 100)
    rungs = [(32, 0.005)] if QUICK else [(32, 0.005), (64, 0.005)]
    for nx, nu in rungs:
        nz, ny, rho, sigma = nx, 4, 1.0, 1.0
        lam = float(nx)
        k = 2 * math.pi / lam
        a0 = lam / 100.0
        omega_th = math.sqrt(sigma * k ** 3 / (2 * rho))
        s_ex, _ = wave_mode(k, nu, sigma, rho, H=nz / 2.0)
        omega_ex, gamma_ex = s_ex.imag, -s_ex.real
        for label, cls in GRIDS:
            s = cls(nx, ny, nz)
            s.set_rho(rho)
            s.set_mu(nu * rho)
            s.set_domain_bc(4, 1, 0, 0, 0)
            s.set_domain_bc(5, 1, 0, 0, 0)
            s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
            s.set_pressure_chebyshev(True, 500, 1e-11)
            s.enable_vof()
            xe = np.arange(nx + 1)
            zi = nz / 2.0 + a0 * (np.sin(k * xe[1:]) - np.sin(k * xe[:-1])) / (k * 1.0)
            C = np.zeros((nx, ny, nz))
            for kk in range(nz):
                C[:, :, kk] = np.clip(zi - kk, 0.0, 1.0)[:, None]
            s.set_vof(np.asfortranarray(C))
            s.set_property_model("rho", "linear", "C", [rho, 0.0])
            s.set_surface_tension(sigma)
            dt = 0.5 * s.capillary_dt()
            s.set_dt(dt)
            nsteps = int(2.5 * (2 * math.pi / omega_th) / dt)
            h = Solve(500)
            t, amp = [], []
            for i in range(nsteps):
                s.step()
                h.sample(s)
                col = s.get_vof().sum(axis=2)[:, 0]
                t.append((i + 1) * dt)
                amp.append(float(np.dot(col - nz / 2.0,
                                        np.cos(k * (np.arange(nx) + 0.5))) * 2.0 / nx))
            t, amp = np.array(t), np.array(amp)
            zc = zero_crossings(t, amp)
            omega_num = math.pi / np.mean(np.diff(zc)) if len(zc) >= 3 else float("nan")
            pk = [i for i in range(1, len(amp) - 1)
                  if abs(amp[i]) > abs(amp[i - 1]) and abs(amp[i]) > abs(amp[i + 1])]
            gamma_num = (-math.log(abs(amp[pk[-1]]) / abs(amp[pk[0]])) / (t[pk[-1]] - t[pk[0]])
                         if len(pk) >= 2 else float("nan"))
            print(f"  nx = {nx}  {label}  omega {omega_num:.5f}  vs inviscid {omega_th:.5f} "
                  f"({100*(omega_num/omega_th-1):+.2f} %)  vs EXACT viscous {omega_ex:.5f} "
                  f"({100*(omega_num/omega_ex-1):+.2f} %)")
            print(f"                    gamma {gamma_num:.3e} vs exact {gamma_ex:.3e} "
                  f"({100*(gamma_num/gamma_ex-1):+.1f} %);  {h}")


# --------------------------------------------------------------------------- G4 Hysing case 1
def gate_hysing1(nx=64):
    p = HYSING[1]
    print("\n" + "=" * 100)
    print(f"G4  HYSING RISING BUBBLE case 1 (ratio 10), nx = {nx} — both grids")
    print("=" * 100)
    print("  `enable_vof_momentum` is STAGGERED-ONLY at this rung (the collocated construction")
    print("  needs Favre face states, AMR-Wind's pattern), so the collocated column runs without")
    print("  it and the honest staggered comparison is the momentum-consistency-OFF one. Both")
    print("  staggered columns are printed; V2b is worth ~14 % here.\n")
    Lx = 1.0
    nz, ny = 2 * nx, 4
    sc = Scale(nx / Lx)
    R = sc.len_to_cells(0.25)
    runs = [("staggered  (V2b ON )", pf.Solver, True),
            ("staggered  (V2b OFF)", pf.Solver, False),
            ("COLLOCATED (V2b n/a)", pf.SolverColocated, False)]
    out = {}
    for label, cls, vmom in runs:
        s = cls(nx, ny, nz)
        s.set_rho(p["rho1"])
        s.set_mu(sc.mu(p["mu1"]))
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 600, 1e-12)
        s.enable_vof()
        C = 1.0 - cylinder_fractions((nx, ny, nz), R, nx / 2.0, sc.len_to_cells(0.5))
        s.set_vof(np.asfortranarray(C))
        s.set_property_model("rho", "linear", "C", [p["rho2"], p["rho1"] - p["rho2"]])
        s.set_property_model("mu", "linear", "C",
                             [sc.mu(p["mu2"]), sc.mu(p["mu1"] - p["mu2"])])
        s.set_surface_tension(sc.sigma(p["sigma"]))
        if vmom:
            s.enable_vof_momentum(p["rho2"], p["rho1"])
        s.set_property_model("force_z", "linear", "C",
                             [-sc.bodyforce(p["rho2"] * p["g"]),
                              -sc.bodyforce((p["rho1"] - p["rho2"]) * p["g"])])
        T = 3.0
        dt = 0.5 * s.capillary_dt()
        s.set_dt(dt)
        zs = (np.arange(nz) + 0.5) / sc.s
        h = Solve(600)
        vmax, tvmax, yc, t, i = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()
        while t < T:
            if i % 10 == 0:
                L = s.vof_step_limits()
                dt = 0.4 * min(L["cfl_dt"], L["capillary_dt"])
                s.set_dt(dt)
            t += dt
            i += 1
            s.step()
            h.sample(s)
            B = 1.0 - s.get_vof()
            vb = B.sum()
            yc = float((B.sum(axis=(0, 1)) * zs).sum() / vb)
            vrise = float((B * s.get_w()).sum() / vb) / sc.s
            if vrise > vmax:
                vmax, tvmax = vrise, t
        out[label] = (vmax, tvmax, yc)
        print(f"  {label}  v_rise max {vmax:.4f} at t = {tvmax:.3f}   y_c(3) = {yc:.4f}   "
              f"({i} steps, {time.time()-t0:.0f} s)")
        print(f"                        {h}")
    print(f"\n  published reference (Hysing 2009): v_rise max {p['ref_vmax']:.4f} at "
          f"t = {p['ref_tvmax']:.4f},  y_c(3) = {p['ref_yc']:.4f}")
    a = out["staggered  (V2b OFF)"]
    b = out["COLLOCATED (V2b n/a)"]
    print(f"  collocated vs staggered (both V2b off): v_rise {100*(b[0]/a[0]-1):+.2f} %, "
          f"y_c(3) {100*(b[2]/a[2]-1):+.2f} %")


ALL = {"static": gate_static, "wave": gate_wave,
       "hysing1": lambda: gate_hysing1(32 if QUICK else 64)}

if __name__ == "__main__":
    run = GATES or list(ALL)
    for g in run:
        if g not in ALL:
            print(f"unknown gate {g!r}; known: {', '.join(ALL)}")
            sys.exit(2)
    for g in run:
        ALL[g]()
    print("\ndone.")
