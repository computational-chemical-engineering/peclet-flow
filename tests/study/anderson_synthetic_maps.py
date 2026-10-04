#!/usr/bin/env python3
"""Synthetic linear maps through the Anderson oracle (doc/steady_acceleration.md, Revision 2).

An INSTRUMENT, not a gate. It drives `anderson_oracle.AndersonOracle` and the oracle's
`march_to_steady` with a fake solver whose `step()` is a linear map on the padded `u` buffer, so the
design's algorithm runs unchanged on maps whose spectrum is known exactly. No flow build is needed.

The maps are core U4's (core-anderson tests/test_anderson.cpp `runU4`): Box(25, 20, 20, g = 2);
inner entries k < nBulk = 9997 get lam = 0.9 k / (nBulk - 1); entry nBulk gets the outlier; entries
nBulk + 1, nBulk + 2 carry a 2x2 block B (their diagonal lam is 0); c = 1 everywhere (ghosts: lam 0,
so they stay 1). Optional evaluation noise: a fresh seeded random vector of norm sigma ||g|| per
evaluation (core U8's noise model). The oracle's other fields v, w, p stay 0.

    B = rotation(rot, 0.5)        core U4 (outlier 1.02, rot 1.01: unstable) and its stable twin
    B = [[lam0, kappa], [0, lam0]]  a STABLE non-normal block: rho(J) = lam0 < 1 for every kappa,
                                    numerical range radius lam0 + |kappa| / 2

Sections (all printed; `--json` writes one record per run):
  raw      unconditional `acc.step(True)` like core U4, under rev 1 (does the guard fire?) and rev 2
           (calls to residual 1e-8 / 1e-10 / 1e-12)
  driver   the rev-1 and rev-2 `march_to_steady` (rtol 1e-4) on U4's two maps, noise 0 / 1e-12 / 1e-8

    python tests/study/anderson_synthetic_maps.py [--json out.jsonl]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anderson_oracle as O  # noqa: E402

N = (25, 20, 20)
G = 2
E = tuple(n + 2 * G for n in N)


class _Diag:
    def __init__(self, s):
        self.s = s

    def field_view(self, name):
        return self.s.f[name]

    def last_pressure_iterations(self):
        return 0

    def pressure_solve_failed(self):
        return False


class LinearMapSolver:
    """x <- lam x + c on u, plus the 2x2 block B on two inner entries; optional noise."""

    def __init__(self, outlier, block, sigma=0.0, seed=20261002):
        self.f = {k: np.zeros(E, order="F") for k in ("u", "v", "w", "p")}
        self.f["sdf"] = np.ones(E, order="F")
        self.diagnostics = _Diag(self)
        X, Y, Z = np.meshgrid(*[np.arange(e) for e in E], indexing="ij")
        inner = (X >= G) & (X < E[0] - G) & (Y >= G) & (Y < E[1] - G) & (Z >= G) & (Z < E[2] - G)
        self.inner = np.flatnonzero(inner.reshape(-1, order="F"))
        nb = self.inner.size - 3
        self.lam = np.zeros(int(np.prod(E)))
        self.lam[self.inner[:nb]] = 0.9 * np.arange(nb) / (nb - 1)
        self.lam[self.inner[nb]] = outlier
        self.c = np.ones_like(self.lam)
        self.i0, self.i1 = self.inner[nb + 1], self.inner[nb + 2]
        self.B = np.asarray(block, float)
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.u = self.f["u"].reshape(-1, order="F")
        assert np.shares_memory(self.u, self.f["u"])
        xs = self.c / (1.0 - self.lam)
        xs[[self.i0, self.i1]] = np.linalg.solve(np.eye(2) - self.B, [1.0, 1.0])
        self.xstar = xs

    def step(self):
        t = self.u.copy()
        g = self.lam * t + self.c
        ab = np.array([t[self.i0], t[self.i1]])
        g[[self.i0, self.i1]] += self.B @ ab
        if self.sigma > 0.0:
            n = np.zeros_like(g)
            n[self.inner] = self.rng.standard_normal(self.inner.size)
            g += n * (self.sigma * np.linalg.norm(g[self.inner]) / np.linalg.norm(n))
        self.u[:] = g

    def monitor(self):
        return float(self.u[self.inner].mean())

    def monitor_exact(self):
        return float(self.xstar[self.inner].mean())


def rotation(rho, theta=0.5):
    return [[rho * math.cos(theta), -rho * math.sin(theta)],
            [rho * math.sin(theta), rho * math.cos(theta)]]


def jordan(lam0, kappa):
    return [[lam0, kappa], [0.0, lam0]]


def oracle_kw(rev):
    return dict(signature=(1.0,), c_p=1.0, gauged=False, collocated_advection=False, cells=N,
                metric="V", ritz_scope="mixed", inner_tolerance=0.0,
                ritz_action="none" if rev == 2 else "stop")


MAPS = {
    "U4 unstable (1.02, rotation 1.01)": (1.02, rotation(1.01)),
    "U4 stable twin (0.996, rotation 0.95)": (0.996, rotation(0.95)),
    **{f"stable non-normal (lam0 {l0}, kappa {k})": (0.5, jordan(l0, k))
       for l0 in (0.99, 0.996) for k in (0.1, 1.0, 10.0)},
}


def raw(name, rev, calls=400):
    outlier, block = MAPS[name]
    s = LinearMapSolver(outlier, block)
    acc = O.AndersonOracle(s, window=5, **oracle_kw(rev))
    readings, hit, engaged = [], {}, None
    for k in range(1, calls + 1):
        acc.step(True)
        if engaged is None and acc.engaged:
            engaged = k
        if math.isfinite(acc.ritzRadius):
            readings.append((k, acc.residual, acc.ritzRadius))
        for t in (1e-8, 1e-10, 1e-12):
            if t not in hit and acc.residual <= t:
                hit[t] = k
        if acc.status != "active" or acc.residual <= 1e-13:
            break
    err = float(np.max(np.abs(s.u[s.inner] - s.xstar[s.inner])) / np.max(np.abs(s.xstar[s.inner])))
    return dict(section="raw", map=name, rev=rev, engaged=engaged, status=acc.status, last_call=k,
                residual=acc.residual, restarts=acc.numRestarts,
                calls_to=[hit.get(1e-8), hit.get(1e-10), hit.get(1e-12)], max_rel_err=err,
                max_reading=max((r[2] for r in readings), default=None),
                readings_tail=readings[-3:], would_fire=list(acc.ritz_would_fire))


def driver(name, rev, sigma):
    outlier, block = MAPS[name]
    s = LinearMapSolver(outlier, block, sigma)
    hold, phases = {}, []

    def cb(steps, phase):
        if phases and phases[-1][0] == phase:
            phases[-1][1] += 1
        else:
            phases.append([phase, 1])
    res = O.march_to_steady(s, s.monitor, rtol=1e-4, max_steps=5000, accelerate=True, window=5,
                            callback=cb, oracle_kw=oracle_kw(rev), holder=hold, rev=rev)
    m = res["monitor"]
    return dict(section="driver", map=name, rev=rev, sigma=sigma, converged=res["converged"],
                reason=res["reason"], steps=res["steps"],
                phases=" ".join(f"{p[0][0]}{p[1]}" for p in phases),
                monitor_rel_err=abs(m / s.monitor_exact() - 1.0) if math.isfinite(m) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = []
    for name in MAPS:
        for rev in (1, 2):
            r = raw(name, rev)
            recs.append(r)
            fired = f"'{r['status']}' at call {r['last_call']}" if r["status"] != "active" else "active"
            print(f"raw  rev{rev} {name}: engaged {r['engaged']}, {fired}, calls to residual "
                  f"1e-8/1e-10/1e-12 {r['calls_to']}, restarts {r['restarts']}, max Ritz "
                  f"{r['max_reading']:.5f}, last readings "
                  f"{[(k, f'{x:.2e}', round(z, 5)) for k, x, z in r['readings_tail']]}", flush=True)
    for name in ("U4 unstable (1.02, rotation 1.01)", "U4 stable twin (0.996, rotation 0.95)"):
        for rev in (1, 2):
            for sigma in (0.0, 1e-12, 1e-8):
                r = driver(name, rev, sigma)
                recs.append(r)
                print(f"drv  rev{rev} {name} sigma {sigma:g}: converged={r['converged']} "
                      f"{r['reason']} steps={r['steps']} ({r['phases']}), monitor vs exact fixed "
                      f"point {r['monitor_rel_err']}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
