#!/usr/bin/env python3
"""Gate instrument for the steady-march Anderson acceleration (doc/steady_acceleration.md, rev 2,
work orders WO-4, WO-5, WO-7 and WO-8). An INSTRUMENT, not a ctest: it prints one summary line per run and,
with --json, appends one JSON record per run. The numbers go to doc/steady_acceleration_log.md.

Subcommands (all on the module on PYTHONPATH; bound the OpenMP pool):

  oracle   WO-4: the C++ accelerator against the NumPy oracle (tests/study/anderson_oracle.py,
           rev 2) — the per-step relative velocity residual of unconditional step(True) calls on
           two identical solvers, max relative difference over the first --steps (30) steps.
  run      G1 / G2 / G6 / G7: march_to_steady per window (0 = accelerate=False), recording steps,
           accelerated steps, converged / reason, K, wall time, pressure iterations per step by
           phase (mixed vs plain), accelerator seconds, memory; with window 0 among the windows,
           the G7c line per case: plain converged vs accelerated converged per window. The tight
           staggered bed also prints the G1 bed line: |K_acc/K_inf - 1| <= |K_plain/K_inf - 1| +
           1e-8, K_plain from window 0 of the same run or --k-plain (the logged certificate).
  g3       G3: --steps (400) unconditional acc.step(True) per dt (--betas), tight; status,
           restarts, running-min residual, final residual and K. With --extend-to N (WO-8): past
           --steps, keep stepping until the running-min residual reaches the depth bar (sphere
           1e-9) or N steps in total; every criterion is evaluated at the end of the run. The bed
           has no depth bar: run it with --steps 3000 (a fixed 3000 calls per dt). K is sampled
           after the last call that did not restart (review R3).
  g5       G5: interrupt an accelerated march at step --at (25), checkpoint get_field u,v,w,p (or
           set_state, velocity only), restore into a fresh solver, march again; against the
           uninterrupted accelerated march.
  g8       G8: accelerator overhead (acc.seconds over --steps accelerated steps) against the mean
           plain step (last_step_timers()['step'], after --warmup plain steps, so both are timed
           late in the march), and memory_bytes against the §6.3 formula.
  mpi      G4: tight march_to_steady under mpirun (flow.mpi_block + init_mpi), global <u_x>
           monitor; and --hash: the state hash after --steps accelerated steps (np = 1 against a
           serial build).

Cases (as the oracle builds them):
  sphere  §11 of doc/collocated_invisible_subspace.md: one sphere at the centre of the periodic unit
          cube, phi (0.125), rho = F = 1, mu (1), nu dt / h^2 = beta (6) unless --dt; Stokes unless
          --advection. K = F L^3 / (6 pi mu R <u_x>) (Zick & Homsy normalisation). The Z&H simple
          cubic array is the same case at --phi 0.343 / 0.45.
  bed     a random arrangement (.npz: centres, radii, L), A1's K = F L^3 / (3 pi mu D <u_x> N_p).
          --force sets the body force (finite Re), --implicit-advection A1's implicit advection.

Settings: production = PCG(200, 1e-8), 200 velocity sweeps, rtol 1e-4, max_steps 5000; tight =
PCG(400, 1e-12), velocity residual tolerance 1e-12, rtol 1e-10, max_steps 20000.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def settings_of(name):
    if name == "production":
        return dict(pcg=(200, 1e-8), vel_rtol=None, rtol=1e-4, max_steps=5000)
    if name == "tight":
        return dict(pcg=(400, 1e-12), vel_rtol=1e-12, rtol=1e-10, max_steps=20000)
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------- cases
class Case:
    """Geometry + physics of one case; build() returns a configured solver (global, single rank)."""

    def __init__(self, a):
        self.a = a
        self.rho = 1.0
        self.mu = a.mu
        if a.case == "sphere":
            self.L = 1.0
            self.R = (3.0 * a.phi / (4.0 * math.pi)) ** (1.0 / 3.0) * self.L
            self.F = 1.0 if a.force is None else a.force
        else:
            z = np.load(a.arrangement)
            self.cen = np.asarray(z["centres"], float)
            self.rad = np.asarray(z["radii"], float)
            self.L = float(z["L"])
            self.Dm = 2.0 * float(self.rad.mean())
            self.npart = len(self.rad)
            self.F = 1.0 if a.force is None else a.force
        self.N = a.N
        self.h = self.L / self.N

    def dt(self, beta=None):
        if self.a.dt is not None and beta is None:
            return self.a.dt
        b = self.a.beta if beta is None else beta
        return b * self.rho * self.h * self.h / self.mu

    def sdf_global(self, cx, cy, cz):
        X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        if self.a.case == "sphere":
            c = 0.5 * self.L
            return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - self.R
        from scipy.spatial import cKDTree
        pts = np.stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")], axis=1)
        tree = cKDTree(np.mod(self.cen, self.L), boxsize=self.L)
        k = min(4, len(self.cen))
        d, idx = tree.query(np.mod(pts, self.L), k=k)
        return (d - self.rad[idx]).min(axis=1).reshape(X.shape, order="C")

    def K(self, u):
        if not (math.isfinite(u) and u != 0.0):
            return math.nan
        if self.a.case == "sphere":
            return self.F * self.L ** 3 / (6.0 * math.pi * self.mu * self.R * u)
        return self.F * self.L ** 3 / (3.0 * math.pi * self.mu * self.Dm * u * self.npart)

    def configure(self, s, settings, beta=None):
        a = self.a
        s.set_rho(self.rho)
        s.set_mu(self.mu)
        s.set_dt(self.dt(beta))
        s.set_body_force((self.F, 0.0, 0.0))
        if a.advection:
            s.set_advection(True)
            if a.implicit_advection:
                s.set_implicit_advection(True)
            s.set_advection_scheme(a.advection)
        else:
            s.set_advection(False)
        s.diagnostics.set_velocity_solver_params(200)
        s.set_pressure_multigrid(True, levels=max(2, int(np.log2(self.N)) - 1))
        s.set_pressure_pcg(True, max_iter=settings["pcg"][0], rtol=settings["pcg"][1])
        if settings["vel_rtol"] is not None:
            s.set_velocity_residual_tolerance(settings["vel_rtol"])

    def build(self, flow, scheme, settings, beta=None):
        Cls = flow.Solver if scheme == "staggered" else flow.SolverColocated
        N, L = self.N, self.L
        s = Cls((N, N, N), extent=(L, L, L))
        self.configure(s, settings, beta)
        cx, cy, cz = s.cell_centers()
        s.set_solid(np.asfortranarray(self.sdf_global(cx, cy, cz)), cutcell_pressure=True)
        return s


class Proxy:
    """The solver as march_to_steady sees it, recording the accelerator the driver creates."""

    def __init__(self, s, holder):
        self.step = s.step
        self.diagnostics = _DiagProxy(s.diagnostics, holder)


class _DiagProxy:
    def __init__(self, d, holder):
        self._d = d
        self._h = holder

    def anderson_accelerator(self, **kw):
        acc = self._d.anderson_accelerator(**kw)
        self._h["acc"] = acc
        return acc


def umean(s):
    return float(s.get_u().mean())


def emit(a, rec):
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(rec) + "\n")


def common_rec(a, case):
    return dict(label=a.label, case=a.case, scheme=a.scheme, N=a.N, phi=a.phi, mu=a.mu,
                dt=case.dt(), F=case.F, advection=a.advection, settings=a.settings,
                threads=os.environ.get("OMP_NUM_THREADS"))


# ---------------------------------------------------------------------------------------------- run
def cmd_run(flow, a):
    st = settings_of(a.settings)
    if a.max_steps:
        st["max_steps"] = a.max_steps
    case = Case(a)
    conv, Ks = {}, {}
    for w in a.window:
        s = case.build(flow, a.scheme, st)
        holder = {}
        trace = []

        def cb(steps, phase, s=s, holder=holder, trace=trace):
            acc = holder.get("acc")
            trace.append((phase, int(s.diagnostics.last_pressure_iterations()),
                          acc.residual if acc is not None else None,
                          acc.status if acc is not None else None))

        t0 = time.perf_counter()
        res = flow.march_to_steady(Proxy(s, holder), lambda s=s: umean(s), rtol=st["rtol"],
                                   max_steps=st["max_steps"], accelerate=(w > 0),
                                   window=max(w, 1), callback=cb)
        wall = time.perf_counter() - t0
        acc = holder.get("acc")
        piters = {}
        for ph in ("accelerate", "certify", "plain"):
            its = [t[1] for t in trace if t[0] == ph]
            piters[ph] = (sum(its), len(its))
        phases = []
        for t in trace:
            if phases and phases[-1][0] == t[0]:
                phases[-1][1] += 1
            else:
                phases.append([t[0], 1])
        rec = common_rec(a, case)
        rec.update(window=w, converged=res.converged, reason=res.reason, steps=res.steps,
                   accelerated_steps=res.accelerated_steps, num_restarts=res.num_restarts,
                   monitor=res.monitor, K=case.K(res.monitor), wall_s=wall,
                   pressure_iterations=sum(v[0] for v in piters.values()),
                   piters_by_phase=piters,
                   status=acc.status if acc else None, acc_reason=acc.reason if acc else None,
                   num_resets=acc.num_resets if acc else None,
                   acc_seconds=acc.seconds if acc else None,
                   memory_bytes=acc.memory_bytes if acc else None,
                   phases="".join(f"{p[0][0]}{p[1]} " for p in phases).strip())
        if a.trace:
            rec["trace"] = trace
        pi = {k: (f"{v[0] / v[1]:.2f}" if v[1] else "-") for k, v in piters.items()}
        print(f"{a.label:>8s} {a.case} {a.scheme:10s} N={a.N} {a.settings:10s} m={w} "
              f"conv={res.converged} {res.reason} steps={res.steps} acc={res.accelerated_steps} "
              f"K={rec['K']:.10f} wall={wall:.2f}s piters={rec['pressure_iterations']} "
              f"pit/step a/c/p={pi['accelerate']}/{pi['certify']}/{pi['plain']} "
              f"rst={res.num_restarts} "
              f"status={rec['status']} accsec={rec['acc_seconds']} [{rec['phases']}]",
              flush=True)
        emit(a, rec)
        conv[w] = res.converged
        Ks[w] = rec["K"]
        del s, acc
        holder.clear()
    g1_bed(a, case, conv, Ks)
    if 0 in conv and len(conv) > 1:  # G7c (rev 2): plain converged vs accelerated converged
        accw = {w: c for w, c in conv.items() if w > 0}
        ok = (not conv[0]) or all(accw.values())
        print(f"G7c {a.case} {a.scheme} N={a.N} {a.settings}: plain converged={conv[0]}; "
              + " ".join(f"m={w} converged={c}" for w, c in accw.items())
              + f" -> {'PASS' if ok else 'FAIL'}", flush=True)
        rec = common_rec(a, case)
        rec.update(gate="g7c", plain_converged=conv[0],
                   accelerated_converged={str(w): c for w, c in accw.items()}, ok=ok)
        emit(a, rec)


# G1 on the dense bed (orchestrator decision, 2026-10-03): at tight rtol the stop instrument
# certifies early on this bed for BOTH paths (its tail is ~0.9999, the instrument assumes 0.997), so
# the accelerated K is compared with K_inf, the extrapolation of the 60 000-step plain march (log
# WO-8), allowing the plain march's own certificate error at the SAME dt:
#     pass iff |K_acc / K_inf - 1| <= |K_plain / K_inf - 1| + 1e-8   at both ends of K_inf.
# K_plain is the plain tight certificate (window 0 in the same run, or --k-plain from the log).
BED_K_INF = (98.9156994, 98.9156995)


def g1_bed(a, case, conv, Ks):
    if a.case != "bed" or a.scheme != "staggered" or a.settings != "tight":
        return
    kp = Ks.get(0, a.k_plain)
    if kp is None or not (0 in conv or a.k_plain is not None):
        return
    # Extension (orchestrator 2026-10-03): when the plain tight march does not certify within
    # max_steps, the accelerated K must be closer to K_inf than the plain K at max_steps.
    uncert = (0 in conv and not conv[0]) or a.plain_uncertified
    for w, K in Ks.items():
        if w == 0:
            continue
        ends = [(abs(K / ki - 1.0), abs(kp / ki - 1.0)) for ki in BED_K_INF]
        if uncert:
            ok = conv[w] and all(ea < ep for ea, ep in ends)
            rule = "acc closer than the uncertified plain K at max_steps"
        else:
            ok = conv[w] and all(ea <= ep + 1e-8 for ea, ep in ends)
            rule = "acc <= plain + 1e-8"
        print(f"G1 bed staggered dt={case.dt():.6g} m={w}: |K_acc/K_inf-1| "
              f"{ends[0][0]:.2e}..{ends[1][0]:.2e}, |K_plain/K_inf-1| {ends[0][1]:.2e}.."
              f"{ends[1][1]:.2e} (K_plain {kp!r}) -> {'PASS' if ok else 'FAIL'} ({rule})",
              flush=True)
        rec = common_rec(a, case)
        rec.update(gate="g1_bed", window=w, K_acc=K, K_plain=kp, K_inf=list(BED_K_INF),
                   err_acc=[e[0] for e in ends], err_plain=[e[1] for e in ends], ok=ok)
        emit(a, rec)


# ---------------------------------------------------------------------------------------------- oracle
def cmd_oracle(flow, a):
    sys.path.insert(0, HERE)
    import anderson_oracle as O
    st = settings_of(a.settings)
    case = Case(a)
    m = a.window[0]
    # C++
    s1 = case.build(flow, a.scheme, st)
    acc = s1.diagnostics.anderson_accelerator(window=m)
    rc = []
    for _ in range(a.steps):
        acc.step(True)
        rc.append(acc.residual)
    # oracle (rev 2: velocity metric, no instability guard — the oracle's radius has no consequence)
    s2 = case.build(flow, a.scheme, st)
    us = s2.unit_scales
    tau = st["vel_rtol"] if st["vel_rtol"] else st["pcg"][1]
    okw = dict(signature=(case.dt() * us["time_to_internal"], case.rho * us["density_to_internal"],
                          case.mu * us["viscosity_to_internal"],
                          case.F * us["force_density_to_internal"][0], 0.0, 0.0),
               c_p=1.0, gauged=True, collocated_advection=False, cells=(a.N,) * 3, metric="V",
               ritz_scope="mixed", inner_tolerance=tau, ritz_action="none")
    orc = O.AndersonOracle(s2, window=m, **okw)
    ro = []
    for _ in range(a.steps):
        orc.step(True)
        ro.append(orc.residual)
    rel = [abs(x - y) / abs(y) if y != 0 else abs(x - y) for x, y in zip(rc, ro)]
    worst = max(rel)
    print(f"oracle {a.case} {a.scheme} N={a.N} m={m} {a.settings}: max rel diff of the residual "
          f"over {a.steps} steps = {worst:.3e} (step {rel.index(worst) + 1}); C++ "
          f"{rc[0]:.6e} .. {rc[-1]:.6e}, oracle {ro[0]:.6e} .. {ro[-1]:.6e}", flush=True)
    for k, (x, y, r) in enumerate(zip(rc, ro, rel)):
        if a.trace:
            print(f"  {k + 1:3d} {x:.15e} {y:.15e} {r:.2e}")
    rec = common_rec(a, case)
    rec.update(gate="oracle", window=m, steps=a.steps, max_rel=worst, cpp=rc, oracle=ro)
    emit(a, rec)


# ---------------------------------------------------------------------------------------------- g3
# §8 G3: the running-min residual must reach this (sphere). The dense bed has no depth bar
# (orchestrator decision 2026-10-03): it runs a fixed 3000 accelerated calls per dt (--steps 3000),
# because no residual depth pins its K to 1e-8 at every dt (measured K-error / residual ratio
# 37 - 830, log "Review fixes").
G3_DEPTH = {"sphere": 1e-9, "bed": None}


def cmd_g3(flow, a):
    st = settings_of("tight")
    case = Case(a)
    depth = G3_DEPTH[a.case]
    Ks = {}
    for beta in a.betas:
        s = case.build(flow, a.scheme, st, beta=beta)
        acc = s.diagnostics.anderson_accelerator(window=a.window[0])
        res, rmin, rst_steps = [], math.inf, []
        u, u_call = math.nan, 0  # <u_x> sampled after the last NON-restart call (review R3)
        t0 = time.perf_counter()
        for k in range(max(a.steps, a.extend_to)):
            if k >= a.steps and depth is not None and rmin <= depth:
                break  # --extend-to: the depth bar is reached
            r0 = acc.num_restarts
            acc.step(True)
            if acc.num_restarts > r0:
                rst_steps.append(k + 1)
            else:
                u, u_call = umean(s), k + 1
            res.append(acc.residual)
            rmin = min(rmin, acc.residual)
            if acc.status != "active":
                break
        wall = time.perf_counter() - t0
        K = case.K(u)
        Ks[beta] = K
        per100 = max([sum(1 for q in rst_steps if lo < q <= lo + 100)
                      for lo in range(0, len(res), 100)] or [0])
        deep = depth is None or rmin <= depth
        ok = (acc.status == "active" and per100 <= 1 and deep and res[-1] <= 10 * rmin)
        print(f"g3 {a.case} {a.scheme} N={a.N} beta={beta:g} m={a.window[0]}: status={acc.status} "
              f"({acc.reason}) steps={len(res)} restarts={acc.num_restarts} (max/100 {per100}) "
              f"min_res={rmin:.3e} (bar {depth if depth is not None else 'none'}) "
              f"final_res={res[-1]:.3e} K={K:.12f} "
              f"(at call {u_call}) wall={wall:.1f}s -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        rec = common_rec(a, case)
        rec.update(gate="g3", beta=beta, window=a.window[0], status=acc.status,
                   reason=acc.reason, steps=len(res), restarts=acc.num_restarts,
                   restart_calls=rst_steps, restarts_max_per_100=per100, min_res=rmin,
                   depth=depth, final_res=res[-1], K=K, u=u, K_call=u_call, ok=ok, wall_s=wall)
        emit(a, rec)
        del acc, s
    if len(Ks) > 1:
        vals = [v for v in Ks.values() if math.isfinite(v)]
        spread = (max(vals) - min(vals)) / abs(min(vals)) if vals else math.nan
        print(f"g3 {a.case} {a.scheme} N={a.N}: K over betas {sorted(Ks)} spread {spread:.3e} "
              f"-> {'PASS' if spread <= 1e-8 else 'FAIL'} (<= 1e-8)", flush=True)
        rec = common_rec(a, case)
        rec.update(gate="g3_spread", betas=list(Ks), Ks=list(Ks.values()), spread=spread)
        emit(a, rec)


# ---------------------------------------------------------------------------------------------- g5
def cmd_g5(flow, a):
    st = settings_of(a.settings)
    case = Case(a)
    m = a.window[0]
    s0 = case.build(flow, a.scheme, st)
    full = flow.march_to_steady(s0, lambda: umean(s0), rtol=st["rtol"],
                                max_steps=st["max_steps"], accelerate=True, window=m)
    K0 = case.K(full.monitor)
    for mode in ("field", "state"):
        s1 = case.build(flow, a.scheme, st)
        acc = s1.diagnostics.anderson_accelerator(window=m)
        for _ in range(a.at):  # the first a.at steps of phase A (residual far above the target)
            acc.step(True)
        ck = {f: np.array(s1.get_field(f)) for f in ("u", "v", "w", "p")}
        vel = [np.array(s1.get_u()), np.array(s1.get_v()), np.array(s1.get_w())]
        del acc, s1
        s2 = case.build(flow, a.scheme, st)
        if mode == "field":
            for f, v in ck.items():
                s2.set_field(f, np.asfortranarray(v))
        else:
            s2.set_state(*[np.asfortranarray(v) for v in vel])
        res = flow.march_to_steady(s2, lambda s2=s2: umean(s2), rtol=st["rtol"],
                                   max_steps=st["max_steps"], accelerate=True, window=m)
        K = case.K(res.monitor)
        rel = abs(K / K0 - 1.0)
        total = a.at + res.steps
        print(f"g5 {a.case} {a.scheme} N={a.N} {a.settings} m={m} restore={mode}: uninterrupted "
              f"{full.steps} steps K={K0:.12f}; interrupted at {a.at} + {res.steps} = {total} "
              f"steps K={K:.12f} |dK|/K={rel:.2e} extra={total - full.steps} "
              f"-> {'PASS' if rel <= 1e-8 and res.converged else 'FAIL'}", flush=True)
        rec = common_rec(a, case)
        rec.update(gate="g5", restore=mode, window=m, at=a.at, full_steps=full.steps,
                   K_full=K0, steps_after=res.steps, total=total, K=K, rel=rel,
                   converged=res.converged)
        emit(a, rec)


# ---------------------------------------------------------------------------------------------- g8
def cmd_g8(flow, a):
    st = settings_of(a.settings)
    case = Case(a)
    m = a.window[0]
    s = case.build(flow, a.scheme, st)
    for _ in range(a.warmup):  # warm-up: time plain steps late in the march, not in the transient
        s.step()
    plain = []
    for _ in range(a.steps):
        s.step()
        plain.append(s.diagnostics.last_step_timers()["step"])
    acc = s.diagnostics.anderson_accelerator(window=m)
    for _ in range(10):  # history fill + engagement
        acc.step(True)
    t_acc0 = acc.seconds
    steps = []
    for _ in range(a.steps):
        acc.step(True)
        steps.append(s.diagnostics.last_step_timers()["step"])
    over = (acc.seconds - t_acc0) / a.steps
    mp = float(np.mean(plain))
    ms = float(np.mean(steps))
    ns = 7 if (a.scheme == "collocated" and a.advection) else 4
    npad = (a.N + 4) ** 3
    formula = (2 * m + 3) * ns * 8 * npad
    print(f"g8 {a.case} {a.scheme} N={a.N} m={m}: mean plain step {mp * 1e3:.3f} ms, mean step "
          f"under acceleration {ms * 1e3:.3f} ms, accelerator {over * 1e3:.3f} ms/step = "
          f"{100 * over / mp:.2f} % of the plain step ({100 * over / ms:.2f} % of the step under "
          f"acceleration; {a.warmup} plain warm-up steps); memory_bytes {acc.memory_bytes} vs formula "
          f"{formula} ({'equal' if acc.memory_bytes == formula else 'DIFFERENT'}); "
          f"status {acc.status} engaged columns {acc.num_columns}", flush=True)
    rec = common_rec(a, case)
    rec.update(gate="g8", window=m, steps=a.steps, plain_ms=mp * 1e3, acc_step_ms=ms * 1e3,
               overhead_ms=over * 1e3, overhead_pct=100 * over / mp, warmup=a.warmup,
               overhead_pct_of_acc_step=100 * over / ms,
               memory_bytes=acc.memory_bytes, formula=formula)
    emit(a, rec)


# ---------------------------------------------------------------------------------------------- mpi
def cmd_mpi(flow, a):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    st = settings_of(a.settings)
    case = Case(a)
    N, L = a.N, case.L
    Cls = flow.Solver if a.scheme == "staggered" else flow.SolverColocated
    if a.serial:  # the serial build / the single-rank path: the global solver, no init_mpi
        s = case.build(flow, a.scheme, st)
    else:
        origin, bsize = flow.mpi_block(N, N, N)
        s = Cls(tuple(bsize), extent=(L, L, L), global_cells=(N, N, N),
                origin=(0.0, 0.0, 0.0))
        s.init_mpi(N, N, N)
        case.configure(s, st)
        g = (np.arange(N) + 0.5) * (L / N)
        gsdf = case.sdf_global(g, g, g)
        ox, oy, oz = origin
        lx, ly, lz = bsize
        s.set_solid(np.asfortranarray(gsdf[ox:ox + lx, oy:oy + ly, oz:oz + lz]),
                    cutcell_pressure=True)

    def gmean():
        u = np.asarray(s.get_u())
        tot = comm.allreduce(float(u.sum()), op=MPI.SUM)
        return tot / float(N * N * N)

    if a.hash:
        acc = s.diagnostics.anderson_accelerator(window=a.window[0])
        for _ in range(a.steps):
            acc.step(True)
        loc = {f: np.ascontiguousarray(getattr(s, "get_" + f)(), dtype=np.float64)
               for f in ("u", "v", "w", "p")}
        allf = comm.gather(loc, root=0)
        if rank == 0:
            hs = {f: hashlib.sha256(np.concatenate([np.asarray(r[f]).ravel() for r in allf])
                                    .tobytes()).hexdigest() for f in loc}
            tot = hashlib.sha256("".join(hs[f] for f in sorted(hs)).encode()).hexdigest()
            print(f"mpi-hash {a.scheme} N={N} np={size} serial={a.serial} m={a.window[0]} "
                  f"{a.steps} accelerated steps: {tot} status={acc.status} "
                  f"res={acc.residual:.6e}", flush=True)
            rec = common_rec(a, case)
            rec.update(gate="g4_hash", np=size, serial=a.serial, steps=a.steps, hash=tot)
            emit(a, rec)
        return
    for w in a.window:
        if w != a.window[0]:
            raise SystemExit("mpi: one window per process")
        t0 = time.perf_counter()
        res = flow.march_to_steady(s, gmean, rtol=st["rtol"], max_steps=st["max_steps"],
                                   accelerate=(w > 0), window=max(w, 1))
        wall = time.perf_counter() - t0
        if rank == 0:
            K = case.K(res.monitor)
            print(f"mpi {a.case} {a.scheme} N={N} np={size} {a.settings} m={w}: "
                  f"conv={res.converged} {res.reason} steps={res.steps} "
                  f"acc={res.accelerated_steps} K={K:.14f} wall={wall:.1f}s", flush=True)
            rec = common_rec(a, case)
            rec.update(gate="g4", np=size, window=w, converged=res.converged, reason=res.reason,
                       steps=res.steps, accelerated_steps=res.accelerated_steps, K=K,
                       monitor=res.monitor, wall_s=wall)
            emit(a, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "oracle", "g3", "g5", "g8", "mpi"))
    ap.add_argument("case", choices=("sphere", "bed"))
    ap.add_argument("--scheme", choices=("staggered", "collocated"), required=True)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--phi", type=float, default=0.125)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=6.0, help="nu dt / h^2 (ignored with --dt)")
    ap.add_argument("--betas", type=float, nargs="+", default=[6.0, 60.0, 600.0, 1e4])
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--force", type=float, default=None)
    ap.add_argument("--advection", default=None)
    ap.add_argument("--implicit-advection", action="store_true")
    ap.add_argument("--arrangement", default=None)
    ap.add_argument("--settings", choices=("production", "tight"), default="production")
    ap.add_argument("--window", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--k-plain", type=float, default=None,
                    help="run, tight staggered bed: the plain tight certificate's K at this dt "
                         "(from the log) for the G1 bed criterion, when window 0 is not run")
    ap.add_argument("--plain-uncertified", action="store_true",
                    help="run, G1 bed: --k-plain is the plain march's K at max_steps (it did not "
                         "certify); the accelerated K must then be closer to K_inf")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--at", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=100, help="g8: plain steps before timing")
    ap.add_argument("--extend-to", type=int, default=0,
                    help="g3: past --steps, continue until the running-min residual reaches the "
                         "depth bar (sphere 1e-9; the bed has none) "
                         "or this many steps in total (0 = no extension)")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()
    {"run": cmd_run, "oracle": cmd_oracle, "g3": cmd_g3, "g5": cmd_g5, "g8": cmd_g8,
     "mpi": cmd_mpi}[a.cmd](flow, a)


if __name__ == "__main__":
    main()
