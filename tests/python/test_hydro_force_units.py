#!/usr/bin/env python
"""The hydrodynamic force and torque getters are unit-covariant, the public one is accurate, and
the traction diagnostic reads what it is documented to read.

The public getter is `hydro_force_torque_reaction()` (the discrete reaction). The traction
integral is the diagnostic `diagnostics.hydro_force_torque_traction()`; its old name
`hydro_force_torque()` is deprecated and must return the same array (2026-09-30, register
coupling.md "The public force API returns the reaction").

One sphere in a periodic cube (phi = 0.216, the regression's Zick & Homsy case) is solved in THREE
unit systems that pose the same index-space problem (the same viscous number beta = nu dt / h^2 = 6
and, for the spinning sphere, the same Omega dt):

  * "cell" -- no extent (spacing 1), rho 1, mu 0.1, dt 60, f 1e-3: the historical cell-unit API;
  * "unit" -- extent L = 1, rho = mu = f = 1;
  * "odd"  -- extent L = 7.3, rho 850, mu 0.037, f 2.5: no reference scale is 1.

Every DIMENSIONLESS number formed from a getter's output must then agree across the three systems
to round-off (the index-space arithmetic differs only by the scaling of its inputs). That is the gate that failed before 2026-09-30: `hydro_force_torque()`
applied the index -> physical conversion TWICE (force x forceTotalToPhys^2, torque x
torqueToPhys^2), which is the identity in cell units -- so every cell-unit gate passed -- and under
a physical domain read 0.0194 of the reaction at L = 1 and 3e-8 of it at L = 7.3.

Checks, translating sphere (body force along +x, marched to steady Stokes):
  1. K = f L^3 / (3 pi mu D U) within 1.5 % of Zick & Homsy's 7.442 (N = 24: -0.7 % measured);
  2. the reaction identity: sum F_reaction = f * V_fluid within 1 % (geometric V_fluid);
  3. traction / reaction in [0.60, 0.80] -- the documented resolution-independent ~29 % under-read
     (0.698 at N = 24 and 32);
  4. K, F_reaction / (f V_fluid), traction / reaction and the pressure / viscous split of the
     traction identical across the three systems to 1e-10 relative.
Spinning sphere (Omega about z, fixed step count -- the index-space states are proportional, not
steady):
  5. T_z / (mu a^3 Omega) from the reaction AND from the traction identical across the systems to
     1e-10 relative, and nonzero.
The names:
  6. the deprecated hydro_force_torque() returns what diagnostics.hydro_force_torque_traction()
     returns (shape (4, n, 3), equal to the atomics' round-off).
Torque accuracy (unit system only, marched to steady Stokes):
  7. translating sphere on the vertex: the reaction torque is zero (|T| / (|F| R) < 1e-10);
  8. steadily spinning sphere, phi 0.027, N 32 (R/h 6): the reaction torque within (1.00, 1.04)
     of 8 pi mu a^3 Omega / (1 - phi) (1.0275 measured), the traction's in (0.50, 0.70) (0.593);
  9. translating sphere OFF the vertex (true torque zero, no discrete symmetry): the reaction's
     spurious |T| / (|F| R) below 3e-3 (1.1e-3 measured) and below the traction's (4.4e-3).

Run: OMP_NUM_THREADS=8 PYTHONPATH=<build> python tests/python/test_hydro_force_units.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import math
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

N = 24
PHI = 0.216
K_ZH = 7.442
BETA = 6.0  # nu dt / h^2 -- the cell-unit dt 60 at nu 0.1
OMEGA_DT = 0.06  # the spinning sphere's rotation per step (cell units: Omega 1e-3 at dt 60)
SYSTEMS = {  # name: (L or None for cell units, rho, mu, f)
    "cell": (None, 1.0, 0.1, 1e-3),
    "unit": (1.0, 1.0, 1.0, 1.0),
    "odd": (7.3, 850.0, 0.037, 2.5),
}
TOL_X = 1e-10  # cross-system agreement of a dimensionless number (measured <= 3.1e-14)
TOL_ATOMIC = 1e-12  # two calls of one getter: atomics are tolerance-reproducible, not bitwise

failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


def build(name, spin, n=N, phi=PHI, beta=BETA, shift=(0.0, 0.0, 0.0)):
    L, rho, mu, f = SYSTEMS[name]
    if L is None:
        L = float(n)
        s = pf.Solver((n, n, n))
    else:
        s = pf.Solver((n, n, n), extent=(L, L, L))
    h = L / n
    dt = beta * h * h * rho / mu
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force((0.0 if spin else f, 0.0, 0.0))
    s.set_advection(False)
    s.set_pressure_multigrid(True, levels=3)
    s.set_pressure_pcg(True, 200, 1e-12)
    s.diagnostics.set_velocity_solver_params(400, 1e-13)
    R = (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0) * L
    # The sphere centre on a cell VERTEX, the same relative placement in every system. The
    # cell-unit scene puts cell centre (i, j, k) at (i, j, k); a physical one at (i + 1/2) h.
    # `shift` (in cells) moves it off the vertex, breaking the discrete mirror symmetries.
    c0 = 0.5 * L - (0.5 if s.unit_scales["identity"] else 0.0)
    c = [c0 + shift[k] * h for k in range(3)]
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(16)
    nr[0] = R
    nr[14] = 1.0
    nr[15] = 1.0
    ii = np.array([0, -1], dtype=np.int32)
    ir = np.zeros(17)
    ir[0:3] = c
    ir[6] = 1.0
    ir[7] = 1.0
    s.set_scene(ni, nr, ii, ir, periodic=True)
    omega = OMEGA_DT / dt
    if spin:
        s.set_instance_motion(0, ang_vel=[0.0, 0.0, omega], center=c)
    s.set_solid_from_scene(cutcell_pressure=True)
    return s, dict(L=L, rho=rho, mu=mu, f=f, R=R, h=h, dt=dt, omega=omega)


def translating(name):
    s, p = build(name, spin=False)
    prev = 0.0
    for it in range(600):
        s.step()
        um = float(np.mean(s.get_u()))
        if it > 10 and abs(um - prev) < 1e-9 * abs(um):
            break
        prev = um
    U = float(np.mean(s.get_u()))  # superficial: solid cells hold zero
    rea = np.asarray(s.hydro_force_torque_reaction())
    tra = np.asarray(s.diagnostics.hydro_force_torque_traction())
    old = np.asarray(s.hydro_force_torque())  # deprecated name of the same traction
    D = 2.0 * p["R"]
    fV = p["f"] * (p["L"] ** 3 - math.pi / 6.0 * D ** 3)
    r = dict(steps=it + 1,
             K=p["f"] * p["L"] ** 3 / (3.0 * math.pi * p["mu"] * D * U),
             rea_over_fV=rea[0, 0, 0] / fV,
             tra_over_rea=tra[0, 0, 0] / rea[0, 0, 0],
             tra_p_over_rea=tra[2, 0, 0] / rea[0, 0, 0],
             tra_v_over_rea=tra[3, 0, 0] / rea[0, 0, 0],
             old_vs_tra=float(np.max(np.abs(old - tra)) / np.max(np.abs(tra))),
             old_shape=old.shape == tra.shape == (4, 1, 3),
             T_rea_rel=float(np.linalg.norm(rea[1, 0]) / (np.linalg.norm(rea[0, 0]) * p["R"])))
    print("  %-4s steps %3d  K %.9f  F_rea/(f V_fl) %.9f  traction/reaction %.9f "
          "(pressure %.6f + viscous %.6f)" % (name, r["steps"], r["K"], r["rea_over_fV"],
                                              r["tra_over_rea"], r["tra_p_over_rea"],
                                              r["tra_v_over_rea"]))
    return r


def spinning(name, steps=30):
    s, p = build(name, spin=True)
    for _ in range(steps):
        s.step()
    T0 = p["mu"] * p["R"] ** 3 * p["omega"]
    r = dict(T_rea=np.asarray(s.hydro_force_torque_reaction())[1, 0, 2] / T0,
             T_tra=np.asarray(s.diagnostics.hydro_force_torque_traction())[1, 0, 2] / T0)
    print("  %-4s T_z/(mu a^3 Omega): reaction %.9f  traction %.9f" % (name, r["T_rea"],
                                                                        r["T_tra"]))
    return r


def steady(s, probe, maxit=3000, tol=1e-9):
    """March to steady Stokes: stop when probe() changes by < tol relative over 10 steps."""
    prev = None
    for it in range(maxit):
        s.step()
        if it % 10 == 9:
            v = probe()
            if prev is not None and abs(v - prev) < tol * abs(v):
                return it + 1
            prev = v
    return maxit


def spinning_steady(n, phi):
    """Steady spinning sphere: T_z against the periodic-array value 8 pi mu a^3 Omega / (1 - phi)
    (a rigid rotation sets the cell's mean vorticity, so the fluid counter-rotates at -phi Omega
    to leading order; phi 0.008 and 0.027 at equal R/h agree to 3e-4 under this reference)."""
    s, p = build("unit", True, n=n, phi=phi, beta=BETA * (n / N) ** 2)
    steps = steady(s, lambda: float(np.asarray(s.hydro_force_torque_reaction())[1, 0, 2]))
    T0 = 8.0 * math.pi * p["mu"] * p["R"] ** 3 * p["omega"] / (1.0 - phi)
    rea = np.asarray(s.hydro_force_torque_reaction())
    tra = np.asarray(s.diagnostics.hydro_force_torque_traction())
    r = dict(T_rea=-rea[1, 0, 2] / T0, T_tra=-tra[1, 0, 2] / T0,
             off_axis=float(np.linalg.norm(rea[1, 0, :2]) / abs(rea[1, 0, 2])))
    print("  unit N %d phi %.3f R/h %.2f steps %d: -T_z (1 - phi) / (8 pi mu a^3 Omega) reaction "
          "%.6f  traction %.6f" % (n, phi, p["R"] / p["h"], steps, r["T_rea"], r["T_tra"]))
    return r


def translating_offset(n, phi, shift):
    """Steady translating sphere OFF the cell vertex: the true torque is zero, and no discrete
    symmetry makes it so -- what is left is each getter's own torque error."""
    s, p = build("unit", False, n=n, phi=phi, shift=shift)
    steps = steady(s, lambda: float(np.mean(s.get_u())))
    rea = np.asarray(s.hydro_force_torque_reaction())
    tra = np.asarray(s.diagnostics.hydro_force_torque_traction())
    r = dict(rea=float(np.linalg.norm(rea[1, 0]) / (np.linalg.norm(rea[0, 0]) * p["R"])),
             tra=float(np.linalg.norm(tra[1, 0]) / (np.linalg.norm(tra[0, 0]) * p["R"])))
    print("  unit N %d phi %.3f R/h %.2f centre + %s h, steps %d: |T| / (|F| R) reaction %.2e  "
          "traction %.2e" % (n, phi, p["R"] / p["h"], shift, steps, r["rea"], r["tra"]))
    return r


def agree(rows, key):
    v = np.array([rows[n][key] for n in SYSTEMS])
    return float(np.max(np.abs(v - v[0])) / abs(v[0]))


print("translating sphere, phi %.3f, N %d, D/h %.1f" % (PHI, N,
                                                       2 * (3 * PHI / (4 * math.pi)) ** (1 / 3) * N))
tr = {n: translating(n) for n in SYSTEMS}
for n, r in tr.items():
    check(abs(r["K"] / K_ZH - 1.0) < 0.015, "%s: K %.5f within 1.5 %% of Zick-Homsy %.3f" %
          (n, r["K"], K_ZH))
    check(abs(r["rea_over_fV"] - 1.0) < 0.01, "%s: reaction = f V_fluid (%.5f)" %
          (n, r["rea_over_fV"]))
    check(0.60 <= r["tra_over_rea"] <= 0.80, "%s: traction / reaction %.5f in [0.60, 0.80]" %
          (n, r["tra_over_rea"]))
    check(r["old_shape"] and r["old_vs_tra"] < TOL_ATOMIC,
          "%s: deprecated hydro_force_torque() == diagnostics.hydro_force_torque_traction() "
          "(max rel diff %.1e)" % (n, r["old_vs_tra"]))
for key in ("K", "rea_over_fV", "tra_over_rea", "tra_p_over_rea", "tra_v_over_rea"):
    d = agree(tr, key)
    check(d < TOL_X, "%s identical across unit systems (max rel diff %.2e)" % (key, d))

print("spinning sphere, Omega dt %.3f, 30 steps" % OMEGA_DT)
sp = {n: spinning(n) for n in SYSTEMS}
check(abs(sp["cell"]["T_rea"]) > 1.0 and abs(sp["cell"]["T_tra"]) > 1.0,
      "the torques are not trivially zero")
for key in ("T_rea", "T_tra"):
    d = agree(sp, key)
    check(d < TOL_X, "%s identical across unit systems (max rel diff %.2e)" % (key, d))

print("torque accuracy (the public getter hydro_force_torque_reaction vs the traction diagnostic)")
for n, r in tr.items():
    check(r["T_rea_rel"] < 1e-10, "%s: translating sphere on the vertex, reaction torque zero "
          "(|T| / (|F| R) = %.1e)" % (n, r["T_rea_rel"]))
ss = spinning_steady(32, 0.027)
# Measured 2026-09-30 (flow force-api): reaction 1.0275 / 1.0252 / 1.0193 at R/h 6 / 9 / 12 and
# phi 0.027 (1.0270 / 1.0248 at phi 0.008); traction 0.593 / 0.653 / 0.723.
check(1.0 < ss["T_rea"] < 1.04, "steady spin: reaction torque %.4f of 8 pi mu a^3 Omega / (1 - phi) "
      "in (1.00, 1.04)" % ss["T_rea"])
check(0.50 < ss["T_tra"] < 0.70, "steady spin: traction torque %.4f in (0.50, 0.70) -- the "
      "documented under-read" % ss["T_tra"])
check(ss["off_axis"] < 1e-10, "steady spin: reaction torque about z only (%.1e)" % ss["off_axis"])
to = translating_offset(32, 0.027, (0.31, 0.17, 0.43))
# Measured: reaction 1.1e-3, traction 4.4e-3 at this case; 4.9e-5 against 2.8e-3 at R/h 18.
check(to["rea"] < 3e-3, "off-vertex translating sphere: spurious reaction torque %.1e < 3e-3"
      % to["rea"])
check(to["rea"] < to["tra"], "off-vertex translating sphere: reaction torque error below the "
      "traction's (%.1e vs %.1e)" % (to["rea"], to["tra"]))

if failures:
    print("FAIL: %d check(s)" % len(failures))
    sys.exit(1)
print("PASS")
