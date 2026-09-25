#!/usr/bin/env python
"""Guard: every COLLOCATED density path is stable at large mu*dt and has a dt-independent steady
state (doc/collocated_varrho_forces.md §7.6; suite-wide register "Collocated forces stay in the
implicit predictor").

Why a gate and not a grep. The defect this guards against is a property of the discrete STEP: where
the lagged pressure enters relative to the implicit momentum operator. Added as a face acceleration
AFTER the viscous solve (Basilisk centered.h; flow's rung V8/WO-T until 2026-09-25), the lagged
pressure lies exactly in the range the projection removes, so the velocity update is
non-incremental: with the rotational update the pressure obeys P^{n+1} = -4 kappa dt S P^n / rho,
growing ~12 kappa dt/(rho h^2) per step at (pi, pi, pi) once mu dt/(rho h^2) > 1/12 (G2); with
kappa = 0 the steady state depends on dt (G3). The form can come back under any name -- a
centered.h port, an "explicit pressure correction after the viscous solve", a defect-correction
stage that moves -G P^n to a later explicit update -- and none share a token; both signatures are
unconditional.

Paths (a path whose configuration the solver refuses is REPORTED, not failed):
  constant rho: AUTO (ghost), 'gauge-exact', 'plain', 'embed';
  V8 variable rho: a uniform rho field and a frozen ratio-1000 slab, balanced-force projection
  OFF and ON; V8 constant-rho CSF (enable_vof, sigma > 0, C re-set before every step = frozen,
  constant kappa), OFF and ON. The ON variants are reported as unavailable on a build without
  set_balanced_force_projection.

Protocol, Stokes (advection off, so the step is affine and a perturbation evolves exactly by the
step operator); mu = 1, rho_min = 1, h = 1; a small Taylor-Green cell force keeps the twin non-trivial.
  G2 growth: seed P with eps (-1)^(x+y+z) + eps * U(-1, 1) and run 100 steps at dt in
     {0.1, 1, 10, 100} beside the unseeded twin. Measured: the (pi,pi,pi) FFT amplitude of P - P_twin
     and max|u - u_twin|. Pass iff at step 100 both are <= 2x their reference and <= (1 + 1e-6)x
     their step-50 value. Reference: the seeded amplitude for P; for u (zero before the first step)
     its value after step 1. Values below a round-off floor FLOOR*eps count as the floor (a decayed
     transient is noise, not growth).
  G3 dt-independence: walled at +-y, the Taylor-Green force, dt in {100, 10, 1}. The dt = 100 run
     marches from rest to max|du| <= 1e-12 max|u|; the dt = 10 and dt = 1 runs are WARM-STARTED
     from the dt = 100 state (u, v, w, P, and P_b when the option is on) and march to the same criterion. A dt-independent fixed
     point is then already converged (one or two steps); a dt-dependent one drifts. Pass iff
     max|u(dt_i) - u(dt_j)| <= 1e-9 max|u|. (Warm starting is what makes the ratio-1000 slab
     affordable: from rest its heavy layer relaxes at rate mu k^2/rho ~ 1e-4 per unit time.) Every
     march must converge within G3_MAX_STEPS or the path fails.

Run:  OMP_NUM_THREADS=4 OMP_PROC_BIND=false PYTHONPATH=<build> \
          python tests/python/test_collocated_stability_guard.py
Exit 0 pass, 1 fail, 77 skipped (no module).
"""
import sys

import numpy as np

try:
    import peclet.flow as pf
except ImportError:
    print("SKIP: peclet.flow not importable")
    sys.exit(77)

N, NZ = 16, 4
MU = 1.0
EPS = 1e-3          # seeding amplitude (the step is affine: the amplitude only sets the floor)
FLOOR = 1e-9        # round-off floor, relative to EPS
A_TG = 1e-4         # Taylor-Green cell-force amplitude (keeps the VoF CFL << 0.25 at dt = 100)
SIGMA = 1e-6        # CSF path: capillary dt = sqrt(2/(4 pi sigma)) ~ 400 > the largest dt
KAPPA = 0.25
G2_DTS = (0.1, 1.0, 10.0, 100.0)
G3_DTS = (100.0, 10.0, 1.0)
G2_STEPS = 100
G3_MAX_STEPS = 3000
G3_TOL = 1e-12
G3_GATE = 1e-9


def F(a):
    return np.asfortranarray(np.ascontiguousarray(a, dtype=np.float64))


def tg_force():
    k = 2.0 * np.pi / N
    xc = np.arange(N) + 0.5
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    rep = lambda a: np.repeat(a[:, :, None], NZ, axis=2)  # noqa: E731
    return rep(A_TG * np.sin(k * X) * np.cos(k * Y)), rep(-A_TG * np.cos(k * X) * np.sin(k * Y))


def slab_rho(ratio):
    y = np.arange(N)
    r = np.where((y >= N // 4) & (y < 3 * N // 4), ratio, 1.0)
    return np.repeat(np.repeat(r[None, :, None], N, 0), NZ, 2)


def drop_colour():
    """Volume fractions of a sphere, R = 4, 4x4x4 sub-sampling (frozen interface for the CSF)."""
    sub = 4
    off = (np.arange(sub) + 0.5) / sub
    c = np.array([N / 2 + 0.13, N / 2 + 0.27, NZ / 2 + 0.11])
    out = np.zeros((N, N, NZ))
    for a in off:
        for b in off:
            for d in off:
                X, Y, Z = np.meshgrid(np.arange(N) + a, np.arange(N) + b, np.arange(NZ) + d,
                                      indexing="ij")
                out += ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < 16.0)
    return out / sub ** 3


class Refused(Exception):
    """The configuration was refused before the first step completed (reported, not failed)."""


class Path:
    """One collocated density path: builds a configured solver, and runs the per-step hook."""

    def __init__(self, name, scheme=None, rho=None, csf=False, bfp=None):
        self.name, self.scheme, self.rho, self.csf, self.bfp = name, scheme, rho, csf, bfp
        self.c0 = drop_colour() if csf else None
        self.stepped = set()  # id() of the solvers that completed a step

    def build(self, dt, walls):
        s = pf.SolverColocated(N, N, NZ)
        s.set_rho(1.0)
        s.set_mu(MU)
        s.set_dt(dt)
        s.set_advection(False)
        s.diagnostics.set_velocity_solver_params(400, 1e-13, 2)
        if walls:
            s.set_domain_bc("-y", "wall")
            s.set_domain_bc("+y", "wall")
        if self.scheme is not None:
            s.set_collocated_scheme(self.scheme)
        s.set_pressure_geometry(F(np.full((N, N, NZ), 10.0)))
        s.enable_cell_force()
        fx, fy = tg_force()
        for name, a in (("force_x", fx), ("force_y", fy), ("force_z", np.zeros_like(fx))):
            s.set_field(name, F(a))
            s.diagnostics.exchange_field(name)
        if self.rho is not None:
            s.add_field("rho")
            s.set_field("rho", F(self.rho))
            s.diagnostics.exchange_field("rho")
            s.diagnostics.set_density_mode("variable")
        if self.csf:
            s.enable_vof()
            s.set_vof(F(self.c0))
            s.set_surface_tension(SIGMA)
            s.diagnostics.set_vof_kappa_constant(KAPPA)
        if self.bfp is not None:
            if hasattr(s, "set_balanced_force_projection"):
                s.set_balanced_force_projection(self.bfp)
            elif self.bfp:  # a build without the option runs OFF by construction
                raise NotImplementedError("set_balanced_force_projection not in this build")
        # the driver LAST (set_density_mode re-selects Chebyshev)
        if self.rho is not None:
            s.set_pressure_chebyshev(True, 2000, 1e-14)
        else:
            s.set_pressure_pcg(True, 400, 1e-14)
        return s

    def step(self, s):
        if self.csf:
            s.set_vof(F(self.c0))  # frozen interface: the step stays affine in (u, P)
        try:
            s.step()
        except RuntimeError as exc:  # a refusal at the first step is a scope statement; later a
            if id(s) not in self.stepped:  # throw is a failure of the path
                raise Refused(str(exc)) from exc
            raise
        self.stepped.add(id(s))


def vel(s):
    return np.stack([np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w())])


def cb_amp(d):
    """(pi, pi, pi) Fourier amplitude of a cell field."""
    return abs(np.fft.fftn(d)[N // 2, N // 2, NZ // 2]) / d.size


def g2(path, dt):
    """Returns (ok, text)."""
    a, b = path.build(dt, False), path.build(dt, False)
    rng = np.random.default_rng(12345)
    x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(NZ), indexing="ij")
    seed = EPS * (1.0 - 2.0 * ((x + y + z) % 2)) + EPS * rng.uniform(-1.0, 1.0, (N, N, NZ))
    b.set_field("p", F(np.asarray(b.get_field("p")) + seed))
    b.diagnostics.exchange_field("p")
    p_ref = cb_amp(seed)
    u_ref = p50 = u50 = None
    fl = FLOOR * EPS
    for k in range(1, G2_STEPS + 1):
        path.step(a)
        path.step(b)
        if k in (1, 50, G2_STEPS):
            dp = cb_amp(np.asarray(b.get_p()) - np.asarray(a.get_p()))
            du = float(np.max(np.abs(vel(b) - vel(a))))
            if not (np.isfinite(dp) and np.isfinite(du)):
                return False, f"dt {dt:<5g} NON-FINITE at step {k}"
            if k == 1:
                u_ref = du
            if k == 50:
                p50, u50 = dp, du
    p100, u100 = dp, du
    m = lambda v: max(v, fl)  # noqa: E731
    ok = (m(p100) <= 2.0 * m(p_ref) and m(p100) <= (1 + 1e-6) * m(p50) and
          m(u100) <= 2.0 * m(u_ref) and m(u100) <= (1 + 1e-6) * m(u50))
    return ok, (f"dt {dt:<5g} P_cb {p_ref:.2e} -> {p50:.2e} -> {p100:.2e}   "
                f"|du| {u_ref:.2e} -> {u50:.2e} -> {u100:.2e}  {'ok' if ok else 'FAIL'}")


def march(path, s):
    prev = vel(s)
    for k in range(1, G3_MAX_STEPS + 1):
        path.step(s)
        cur = vel(s)
        if not np.all(np.isfinite(cur)):
            return cur, k, False
        if np.max(np.abs(cur - prev)) <= G3_TOL * max(np.max(np.abs(cur)), 1e-300):
            return cur, k, True
        prev = cur
    return cur, G3_MAX_STEPS, False


def g3(path):
    sols, info = {}, []
    state = None
    for dt in G3_DTS:
        s = path.build(dt, True)
        if state is not None:
            for name, a in state.items():
                if name == "p_balanced":
                    s.add_field(name)
                s.set_field(name, F(a))
                s.diagnostics.exchange_field(name)
        u, k, conv = march(path, s)
        if state is None:
            # the WHOLE state: with the balanced-force projection on, P's split P_b is state too
            # (a fresh P_b = 0 under a transferred total P would add the balanced part twice)
            state = {n: np.asarray(s.get_field(n)) for n in ("u", "v", "w", "p")}
            try:
                state["p_balanced"] = np.asarray(s.get_field("p_balanced"))
            except IndexError:  # the option never ran: no split to carry
                pass
        sols[dt] = u
        info.append(f"dt {dt:g}: {k} steps{'' if conv else ' (NOT converged)'}")
    scale = max(np.max(np.abs(sols[G3_DTS[0]])), 1e-300)
    worst = 0.0
    for i, di in enumerate(G3_DTS):
        for dj in G3_DTS[i + 1:]:
            d = float(np.max(np.abs(sols[di] - sols[dj])))
            worst = max(worst, d if np.isfinite(d) else np.inf)
    # every march must reach the criterion: an unconverged run measures the march, not the fixed point
    ok = all("NOT" not in t for t in info) and worst <= G3_GATE * scale
    return ok, f"max|u(dt_i) - u(dt_j)|/max|u| = {worst / scale:.2e}  [{'; '.join(info)}]  " + \
        ("ok" if ok else "FAIL")


def paths():
    uni = np.ones((N, N, NZ))
    out = [Path("const-rho AUTO (ghost)"), Path("const-rho gauge-exact", "gauge-exact"),
           Path("const-rho plain", "plain"), Path("const-rho embed", "embed")]
    for bfp in (False, True):
        tag = "ON " if bfp else "OFF"
        out += [Path(f"V8 var-rho uniform       {tag}", rho=uni, bfp=bfp),
                Path(f"V8 var-rho slab 1000     {tag}", rho=slab_rho(1000.0), bfp=bfp),
                Path(f"V8 const-rho CSF         {tag}", csf=True, bfp=bfp)]
    return out


def main(argv):
    only = argv[1] if len(argv) > 1 else None
    ok = True
    for path in paths():
        if only and only not in path.name:
            continue
        print(f"--- {path.name}")
        try:
            path.build(1.0, False)
        except NotImplementedError as exc:
            print(f"    unavailable: {exc}")
            continue
        except RuntimeError as exc:
            print(f"    refused: {exc}")
            continue
        for dt in G2_DTS:
            try:
                good, text = g2(path, dt)
            except Refused as exc:
                good, text = True, f"dt {dt:<5g} refused: {exc}"
            except RuntimeError as exc:
                good, text = False, f"dt {dt:<5g} THREW mid-run: {exc}  FAIL"
            ok &= good
            print(f"    G2 {text}")
        try:
            good, text = g3(path)
        except Refused as exc:
            good, text = True, f"refused: {exc}"
        except RuntimeError as exc:
            good, text = False, f"THREW mid-run: {exc}  FAIL"
        ok &= good
        print(f"    G3 {text}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_collocated_stability_guard():
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
