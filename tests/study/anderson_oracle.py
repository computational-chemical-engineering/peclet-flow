#!/usr/bin/env python3
"""Python oracle for Anderson acceleration of steady marches (doc/steady_acceleration.md, WO-1).

An INSTRUMENT, not a gate and not production code. It implements the design note exactly, in NumPy
over the zero-copy host buffers of `diagnostics.field_view`, on a host build of peclet.flow:

  * §3   state vector (full padded buffers of u, v, w, p; roles Velocity / Pressure) and the
         W-metric (velocity at unit weight + c_P^2 x the fluid-centred, gauge-centred pressure),
         c_P = 1 / (mu_int + rho_int / dt_int) in the solver's internal units;
  * §4   type-II Anderson, one `step(accelerate)` = §4.3 steps 0-7 verbatim: change detection, the
         lazy MIX, the map, the provisional column, the reductions, the host decision (non-finite /
         restart / engagement), the commit and the next coefficients; §4.1 constants; §4.4
         solveTruncated (Jacobi-scaled cyclic-Jacobi eigendecomposition, oldest-column drop while
         the scaled condition < kCondMin); §4.6 the Ritz guard (Gelfand radius of I + XX^+ XR);
  * §7   the driver `march_to_steady` (phase A accelerate to (1 - slow_rate) rtol, phase B certify
         with the UNCHANGED §3.2 instrument on consecutive plain steps within num_passes + 3 blocks,
         target x 0.1 and resume on "budget", plain fallback on "growth" / stagnation / a status
         other than active). accelerate=False is the study's march() verbatim.

What the oracle cannot do, by construction (design note D1): the collocated face field uf_ is not in
the field registry, so a collocated run WITH advection (n_s = 7, §3.1 row 3) is refused here. The
parameter signature (internal dt, rho, mu, body force) has no Python getters; the oracle takes it
from the case that built the solver, which never changes it mid-march, so §4.3 step 0's signature
half can never fire in these runs (the inner-entry state check is implemented and live).

Cases (§11 of doc/collocated_invisible_subspace.md, as tests/study/study_avg_velocity_spheres.py
builds them: one sphere phi = 0.125 at the centre of a periodic unit cube, rho = mu = F = 1,
nu dt / h^2 = 6, Stokes; staggered Solver or SolverColocated with its AUTO = 'ghost' scheme):

    sphere   --scheme staggered|collocated --N 16 [--mu 1 --beta 6 | --dt ...] [--advection koren]
    bed      --scheme ...  the A1 dense random bed (design note §10 Q2), from an arrangement .npz

Settings (§8): --settings production (study: PCG(200, 1e-8), 200 velocity sweeps, rtol 1e-4) or
tight (PCG(400, 1e-12), velocity residual tolerance 1e-12, rtol 1e-10, max_steps 20000).

Run (from the flow tree, OpenMP pool bounded):

    OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<flow host build> \\
        python tests/study/anderson_oracle.py sphere --scheme collocated --N 16 \\
        --settings production --window 0 3 5 8 [--json results.jsonl]

`--window 0` is the plain march (accelerate=False). Each run prints one summary line and, with
--json, appends one JSON record (steps, converged, reason, K, restarts, max Ritz radius over the
active range, max sum|alpha|, wall time, the residual trace).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------------------------
# §4.1 fixed constants (C++ constexpr in the production design; not user-settable)
K_MAX_WINDOW = 8
K_ENGAGE_DECREASES = 2
K_RESTART_GROWTH = 4.0
K_MAX_RESTARTS = 5
K_NOISE_FLOOR = 1e-10
K_COND_MIN = 1e-12
K_RITZ_DELTA = 1e-3
K_RITZ_HI = 1e-2
K_RITZ_CONSECUTIVE = 3
K_GELFAND_SQUARINGS = 20

VELOCITY, PRESSURE, CARRIED = 0, 1, 2


# ---------------------------------------------------------------------------------------------------
# §4.4 / §4.6 host linear algebra (deterministic)
def jacobi_eigh(A, tol=1e-15, max_sweeps=50):
    """Cyclic Jacobi eigendecomposition of a symmetric matrix, fixed sweep order (p < q, row-major).

    Stops when the off-diagonal Frobenius norm is <= tol * ||A||_F, or after max_sweeps sweeps.
    Returns (lam, V) with A = V diag(lam) V^T (columns of V are the eigenvectors)."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    V = np.eye(n)
    normF = math.sqrt(float(np.sum(A * A)))
    for _ in range(max_sweeps):
        off = math.sqrt(max(float(np.sum(A * A) - np.sum(np.diag(A) ** 2)), 0.0))
        if off <= tol * normF:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if apq == 0.0:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                if abs(theta) > 1e150:  # theta^2 would overflow; the same t to round-off
                    t = 0.5 / theta
                else:
                    t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                # A <- J^T A J with J the (p, q) rotation [[c, s], [-s, c]]
                ap = A[:, p].copy()
                aq = A[:, q].copy()
                A[:, p] = c * ap - s * aq
                A[:, q] = s * ap + c * aq
                ap = A[p, :].copy()
                aq = A[q, :].copy()
                A[p, :] = c * ap - s * aq
                A[q, :] = s * ap + c * aq
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
    return np.diag(A).copy(), V


def scaled_eig(RR):
    """Jacobi scaling of §4.4 steps 1-3: D = diag(RR)^1/2, A = D^-1 RR D^-1. Returns (D, lam, V)."""
    D = np.sqrt(np.diag(RR))
    A = RR / np.outer(D, D)
    lam, V = jacobi_eigh(A)
    return D, lam, V


def cond_scaled(lam):
    lmax = float(np.max(lam))
    return float(np.min(lam)) / lmax if lmax > 0.0 else 0.0


def solve_truncated(D, lam, V, b):
    """§4.4 step 5: y = sum_i v_i (v_i^T b~) / lam_i, gamma = D^-1 y, with b~ = D^-1 b."""
    bt = b / D
    y = V @ ((V.T @ bt) / lam)
    return y / D


def pinv_truncated(XX):
    """§4.6: the truncated pseudo-inverse of §4.4 (Jacobi-scaled), eigenvalues below
    kCondMin * lam_max discarded, no column dropping. A zero diagonal entry's row/column is the
    pseudo-inverse's null part (it carries no direction)."""
    n = XX.shape[0]
    d = np.diag(XX)
    live = np.where(d > 0.0)[0]
    P = np.zeros((n, n))
    if live.size == 0:
        return P
    Xl = XX[np.ix_(live, live)]
    D, lam, V = scaled_eig(Xl)
    lmax = float(np.max(lam))
    keep = lam >= K_COND_MIN * lmax if lmax > 0.0 else np.zeros_like(lam, dtype=bool)
    Ainv = (V[:, keep] / lam[keep]) @ V[:, keep].T
    P[np.ix_(live, live)] = Ainv / np.outer(D, D)
    return P


def gelfand_radius(T):
    """§4.6 step 3: rho(T) ~ ||T^(2^20)||_max^(2^-20), renormalised every squaring."""
    s = float(np.max(np.abs(T)))
    if s == 0.0 or not math.isfinite(s):
        return 0.0 if s == 0.0 else float("inf")
    A = T / s
    L = math.log(s)
    for _ in range(K_GELFAND_SQUARINGS):
        A = A @ A
        s = float(np.max(np.abs(A)))
        if s == 0.0:
            return 0.0
        A = A / s
        L = 2.0 * L + math.log(s)
    return math.exp(L / 2.0 ** K_GELFAND_SQUARINGS)


# ---------------------------------------------------------------------------------------------------
class AndersonOracle:
    """§4 AndersonCore + the flow adapter, on host NumPy views of the solver's padded buffers."""

    def __init__(self, solver, *, window=5, mixing=1.0, signature, c_p, gauged, collocated_advection,
                 cells):
        if not 1 <= window <= K_MAX_WINDOW:
            raise ValueError(f"window must be in [1, {K_MAX_WINDOW}]")
        if not 0.0 < mixing <= 1.0:
            raise ValueError("mixing must be in (0, 1]")
        if collocated_advection:
            raise ValueError("collocated + advection needs uf_/vf_/wf_ in the state (§3.1 row 3); "
                             "they are not in the field registry, so the oracle cannot carry them")
        self.s = solver
        self.m = window
        self.beta = mixing
        self.cP = float(c_p)
        self.gauged = bool(gauged)
        self.sig = tuple(signature)
        self.sig_source = tuple(signature)  # the case never changes it (see module docstring)
        names = ["u", "v", "w", "p"]
        self.roles = [VELOCITY, VELOCITY, VELOCITY, PRESSURE]
        G = (solver.diagnostics.field_view("u").shape[0] - int(cells[0])) // 2
        assert G == 2, f"ghost width {G}, the design note's G = 2"
        self.views = []
        for nm in names:
            v = solver.diagnostics.field_view(nm)
            assert v.flags.f_contiguous and v.dtype == np.float64
            self.views.append(v.reshape(-1, order="F"))  # storage order, a view (no copy)
            assert np.shares_memory(self.views[-1], v)
        shape = solver.diagnostics.field_view("u").shape
        self.shape = shape
        ex, ey, ez = shape
        x = np.arange(ex)
        y = np.arange(ey)
        z = np.arange(ez)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        inner = (X >= G) & (X < ex - G) & (Y >= G) & (Y < ey - G) & (Z >= G) & (Z < ez - G)
        self.inner = np.flatnonzero(inner.reshape(-1, order="F"))
        sdf = np.asarray(solver.diagnostics.field_view("sdf")).reshape(-1, order="F")
        self.fluid = self.inner[sdf[self.inner] > 0.0]  # fluid-centred inner cells (sign only)
        self.NF = int(self.fluid.size)
        self.ns = len(names)
        self.npad = self.views[0].size
        self.vel = [f for f in range(self.ns) if self.roles[f] == VELOCITY]
        self.pf = [f for f in range(self.ns) if self.roles[f] == PRESSURE]
        # device-side data of §4.2 (here: host arrays)
        z = lambda: np.zeros((self.ns, self.npad))  # noqa: E731
        self.X, self.Rprev, self.Gprev = z(), z(), z()
        self.dR = [z() for _ in range(self.m)]
        self.dG = [z() for _ in range(self.m)]
        # host-side data
        self.RR = np.zeros((self.m, self.m))
        self.GG = np.zeros((self.m, self.m))
        self.GR = np.zeros((self.m, self.m))
        self.mR = np.zeros(self.m)
        self.mG = np.zeros(self.m)
        self.order = []  # slot indices, oldest -> newest; mk = len(order)
        self.havePrev = False
        self.pending = False
        self.gamma = np.zeros(0)
        self.engaged = False
        self.decCount = 0
        self.rhoPrev = math.inf
        self.rhoMin = math.inf
        self.numRestarts = 0
        self.numResets = 0
        self.ritzCount = 0
        self.status = "active"
        self.reason = ""
        self.residual = math.inf
        self.ritzRadius = math.nan
        # instrumentation (not part of the design's state)
        self.zero_diag_drops = 0
        self.cond_drops = 0
        self.ritz_max = 0.0          # max ritz radius over the active range (G7c)
        self.sum_abs_alpha_max = 0.0  # §4.5: logged, never tested
        self.mixed_steps = 0
        self.ritz_debug = None  # set to [] to record the Ritz cross-checks

    # ----------------------------------------------------------------------------------- helpers
    @property
    def mk(self):
        return len(self.order)

    def read(self):
        return np.stack([v for v in self.views])  # copy of the padded buffers

    def write(self, a):
        for f in range(self.ns):
            self.views[f][:] = a[f]

    def pmean(self, a):
        if not self.gauged or self.NF == 0:
            return 0.0
        return float(sum(a[f, self.fluid].sum() for f in self.pf)) / self.NF

    def wdot(self, a, b, ma, mb):
        """§3.2 <a,b>_W: velocity at unit weight over inner entries + c_P^2 x the centred
        fluid-cell pressure product."""
        acc = 0.0
        for f in self.vel:
            acc += float(np.dot(a[f, self.inner], b[f, self.inner]))
        pp = 0.0
        for f in self.pf:
            pp += float(np.dot(a[f, self.fluid] - ma, b[f, self.fluid] - mb))
        return acc + self.cP * self.cP * pp

    def usq(self, g):
        return float(sum(np.dot(g[f, self.inner], g[f, self.inner]) for f in self.vel))

    def reset(self):
        """§4.2 reset(): clears history + counters, keeps havePrev."""
        self.order = []
        self.pending = False
        self.engaged = False
        self.decCount = 0
        self.ritzCount = 0
        self.rhoMin = math.inf
        self.rhoPrev = math.inf

    def disable(self, reason="disabled by the caller"):
        self.status = "disabled"
        self.reason = reason
        self.pending = False

    # ----------------------------------------------------------------------------------- §4.3
    def step(self, accelerate):
        if self.status != "active":  # plain step, no history maintenance, residual kept
            self.s.step()
            return
        # --- 0. change detection
        if self.sig_source != self.sig:
            self.reset()
            self.havePrev = False
            self.numResets += 1
            self.sig = self.sig_source
        buf = self.read()
        if self.havePrev:
            n_changed = int(sum(np.count_nonzero(buf[f, self.inner] != self.Gprev[f, self.inner])
                                for f in range(self.ns)))
            if n_changed > 0:
                self.reset()
                self.havePrev = False
                self.numResets += 1
        # --- 1. lazy mix
        mixed = bool(accelerate and self.pending and self.status == "active")
        if mixed:
            c = np.zeros_like(buf)
            d = np.zeros_like(buf)
            for j, slot in enumerate(self.order):
                c += self.gamma[j] * self.dG[slot]
                d += self.gamma[j] * self.dR[slot]
            x = buf - c - (1.0 - self.beta) * (self.Rprev - d)
            self.write(x)
            self.X = x.copy()
            self.mixed_steps += 1
        else:
            self.X = buf.copy()
        self.pending = False
        # --- 2. the map
        try:
            self.s.step()
        except Exception as e:  # noqa: BLE001 -- the design's catch-all
            if mixed:
                self.write(self.Gprev)
                self.status = "disabled"
                self.reason = f"step failed at an accelerated iterate: {e}"
                self.reset()
                print(f"[anderson] {self.reason}", file=sys.stderr)
                return
            raise
        buf = self.read()
        # --- 3. provisional new column
        s = None
        if self.havePrev:
            if self.mk < self.m:
                used = set(self.order)
                s = next(k for k in range(self.m) if k not in used)
            else:
                s = self.order[0]
            r = buf - self.X
            self.dR[s] = r - self.Rprev
            self.dG[s] = buf - self.Gprev
            self.X = r
        else:
            self.X = buf - self.X
        # --- 4. reductions
        mX = self.pmean(self.X)
        rho = math.sqrt(max(self.wdot(self.X, self.X, mX, mX), 0.0)) \
            if np.all(np.isfinite(self.X)) else math.nan
        U2 = self.usq(buf)
        if s is not None:
            mRs, mGs = self.pmean(self.dR[s]), self.pmean(self.dG[s])
            cols = [j for j in self.order if j != s] + [s]
            row = {}
            for j in cols:
                mRj = mRs if j == s else self.mR[j]
                mGj = mGs if j == s else self.mG[j]
                row[j] = (self.wdot(self.dR[s], self.dR[j], mRs, mRj),
                          self.wdot(self.dG[s], self.dG[j], mGs, mGj),
                          self.wdot(self.dG[s], self.dR[j], mGs, mRj),
                          self.wdot(self.dG[j], self.dR[s], mGj, mRs),
                          self.wdot(self.dR[j], self.X, mRj, mX))
        # --- 5. host decision
        if rho == 0.0 and U2 == 0.0:
            residual = 0.0
        elif U2 == 0.0:
            residual = math.inf
        else:
            residual = rho / math.sqrt(U2)
        self.residual = residual
        failed = mixed and bool(self.s.diagnostics.pressure_solve_failed())
        if not math.isfinite(rho) or failed:
            if mixed:
                self.write(self.Gprev)
                self.reason = "non-finite residual / failed pressure solve at an accelerated iterate"
            else:
                self.reason = "non-finite residual on a plain step"
            self.status = "disabled"
            self.reset()
            self.havePrev = False
            return
        b = np.zeros(self.m)
        if s is not None:  # accept column s
            for j, (rr, gg, gr_sj, gr_js, bj) in row.items():
                self.RR[s, j] = self.RR[j, s] = rr
                self.GG[s, j] = self.GG[j, s] = gg
                self.GR[s, j] = gr_sj
                self.GR[j, s] = gr_js
                b[j] = bj
            self.mR[s], self.mG[s] = mRs, mGs
            if s in self.order:
                self.order.remove(s)
            self.order.append(s)
        if mixed and residual >= K_NOISE_FLOOR and rho > K_RESTART_GROWTH * self.rhoMin:
            self.reset()
            self.numRestarts += 1
            self.rhoMin = rho
            if self.numRestarts >= K_MAX_RESTARTS:
                self.status = "disabled"
                self.reason = "too many restarts"
        self.decCount = self.decCount + 1 if rho < self.rhoPrev else 0
        if self.decCount >= K_ENGAGE_DECREASES:
            self.engaged = True
        self.rhoPrev = rho
        self.rhoMin = min(self.rhoMin, rho)
        # --- 6. commit
        self.Rprev = self.X
        self.X = np.zeros_like(self.X)
        self.Gprev = buf.copy()
        self.havePrev = True
        # --- 7. next coefficients
        self.ritzRadius = math.nan
        if self.status == "active" and self.engaged and self.mk >= 1:
            # §4.4 step 1: a column with a zero diagonal is dropped first
            for j in [j for j in self.order if not self.RR[j, j] > 0.0]:
                self.order.remove(j)
                self.zero_diag_drops += 1
            if self.mk == 0:
                return
            while True:
                idx = self.order
                D, lam, V = scaled_eig(self.RR[np.ix_(idx, idx)])
                if self.mk > 1 and cond_scaled(lam) < K_COND_MIN:
                    self.order.pop(0)
                    self.cond_drops += 1
                    continue
                break
            self.gamma = solve_truncated(D, lam, V, b[self.order])
            self.pending = True
            g = self.gamma
            alpha = np.concatenate([[g[0]], np.diff(g), [1.0 - g[-1]]])
            self.sum_abs_alpha_max = max(self.sum_abs_alpha_max, float(np.abs(alpha).sum()))
            if self.mk >= 2 and K_NOISE_FLOOR <= residual <= K_RITZ_HI:
                idx = self.order
                RR = self.RR[np.ix_(idx, idx)]
                GG = self.GG[np.ix_(idx, idx)]
                GR = self.GR[np.ix_(idx, idx)]
                XX = GG - GR - GR.T + RR
                XR = GR - RR
                T = np.eye(self.mk) + pinv_truncated(XX) @ XR
                self.ritzRadius = gelfand_radius(T)
                if self.ritz_debug is not None:  # instrument: independent cross-checks
                    _, lamx, _ = scaled_eig(XX)
                    eigT = np.linalg.eigvals(T)
                    # M from a direct W-metric least squares on the stored vectors (no Gram cache)
                    dX = [self.dG[j] - self.dR[j] for j in idx]
                    mx = [self.pmean(v) for v in dX]
                    XXd = np.array([[self.wdot(dX[i], dX[k], mx[i], mx[k]) for k in range(len(idx))]
                                    for i in range(len(idx))])
                    self.ritz_debug.append(dict(
                        residual=residual, ritz=self.ritzRadius, eig_abs_max=float(np.max(np.abs(eigT))),
                        cond_XX_scaled=cond_scaled(lamx), cond_RR_scaled=cond_scaled(lam),
                        XX_cache_vs_direct=float(np.max(np.abs(XXd - XX)) / np.max(np.abs(XX))),
                        mixed=mixed))
                self.ritz_max = max(self.ritz_max, self.ritzRadius)
                self.ritzCount = self.ritzCount + 1 if self.ritzRadius > 1.0 + K_RITZ_DELTA else 0
                if self.ritzCount >= K_RITZ_CONSECUTIVE:
                    self.status = "unstable"
                    self.reason = (f"the plain map is locally unstable at this dt "
                                   f"(Ritz radius {self.ritzRadius:.6f})")
                    self.pending = False
                    self.reset()


# ---------------------------------------------------------------------------------------------------
# §7 driver
class MarchResult(dict):
    pass


def march_to_steady(solver, monitor, *, rtol=1e-4, max_steps=5000, accelerate=True, window=5,
                    check_every=5, num_passes=3, slow_rate=0.997, roundoff=1e-11, callback=None,
                    oracle_kw=None, holder=None):
    st = dict(steps=0, acc_steps=0)
    cb = callback or (lambda steps, phase: None)
    acc = None

    def result(converged, reason):
        return MarchResult(converged=converged, steps=st["steps"], accelerated_steps=st["acc_steps"],
                           reason=reason, num_restarts=acc.numRestarts if acc else 0,
                           monitor=float(monitor()), acc=acc)

    def certify(step_fn, budget, phase, growth_ref=None):
        """The §3.2 instrument with a fresh block state; returns pass/budget/max/growth/diverged."""
        prev = dprev = None
        passes = 0
        blocks = 0
        it = 0
        while True:
            if st["steps"] >= max_steps:
                return "max"
            step_fn()
            st["steps"] += 1
            cb(st["steps"], phase)
            if growth_ref is not None and (not math.isfinite(acc.residual)
                                           or acc.residual > 10.0 * growth_ref):
                return "growth"
            local = it
            it += 1
            if local % check_every != check_every - 1:
                continue
            m = monitor()
            if not math.isfinite(m):
                if growth_ref is not None:
                    return "growth"
                if budget is None:
                    return "diverged"
            if prev is not None:
                d = m - prev
                ok = abs(d) <= roundoff * abs(m)
                if not ok and dprev:
                    R = d / dprev
                    ok = 0.0 < R < 1.0 and \
                        abs(d) / (1.0 - max(R, slow_rate ** check_every)) < rtol * abs(m)
                passes = passes + 1 if ok else 0
                if passes >= num_passes:
                    return "pass"
                dprev = d
            prev = m
            blocks += 1
            if budget is not None and blocks >= budget:
                return "budget"

    def plain_tail(phase):
        out = certify(solver.step, None, phase)
        if out == "pass":
            return result(True, "certified")
        if out == "diverged":
            return result(False, "diverged")
        return result(False, "max_steps")

    if not accelerate:
        return plain_tail("plain")
    acc = AndersonOracle(solver, window=window, **(oracle_kw or {}))
    if holder is not None:
        holder["acc"] = acc
    target = (1.0 - slow_rate) * rtol
    while st["steps"] < max_steps:
        best, since, stagnated = math.inf, 0, False
        while st["steps"] < max_steps and acc.status == "active" and acc.residual > target:
            acc.step(True)
            st["steps"] += 1
            st["acc_steps"] += 1
            cb(st["steps"], "accelerate")
            if acc.status == "unstable":
                return result(False, "unstable")
            if acc.residual <= 0.5 * best:
                best, since = acc.residual, 0
            else:
                since += 1
            if since >= 10 * window:
                stagnated = True
                break
        if acc.status != "active":
            return plain_tail("plain")
        out = certify(lambda: acc.step(False), num_passes + 3, "certify", growth_ref=acc.residual)
        if out == "pass":
            return result(True, "certified")
        if out == "max":
            return result(False, "max_steps")
        if out == "growth" or stagnated:
            acc.disable()
            return plain_tail("plain")
        target *= 0.1
    return result(False, "max_steps")


# ---------------------------------------------------------------------------------------------------
# cases
L1 = 1.0
ZH_PHI = [0.000125, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.45, 0.5236]
ZH_K = [1.096, 1.212, 1.525, 2.008, 2.810, 4.292, 7.442, 15.4, 28.1, 42.1]


def settings_of(name):
    if name == "production":
        return dict(pcg=(200, 1e-8), vel_rtol=None, rtol=1e-4, max_steps=5000)
    if name == "tight":
        return dict(pcg=(400, 1e-12), vel_rtol=1e-12, rtol=1e-10, max_steps=20000)
    raise ValueError(name)


def build_solver(flow, scheme, N, sdf_fn, L, *, rho, mu, F, dt, advection, settings):
    Cls = flow.Solver if scheme == "staggered" else flow.SolverColocated
    s = Cls((N, N, N), extent=(L, L, L))
    s.set_rho(rho)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force((F, 0.0, 0.0))
    if advection:
        s.set_advection(True)
        s.set_advection_scheme(advection)
    else:
        s.set_advection(False)
    s.diagnostics.set_velocity_solver_params(200)
    s.set_pressure_multigrid(True, levels=max(2, int(np.log2(N)) - 1))
    s.set_pressure_pcg(True, max_iter=settings["pcg"][0], rtol=settings["pcg"][1])
    if settings["vel_rtol"] is not None:
        s.set_velocity_residual_tolerance(settings["vel_rtol"])
    cx, cy, cz = s.cell_centers()
    s.set_solid(np.asfortranarray(sdf_fn(cx, cy, cz)), cutcell_pressure=True)
    us = s.unit_scales
    rho_i = rho * us["density_to_internal"]
    mu_i = mu * us["viscosity_to_internal"]
    dt_i = dt * us["time_to_internal"]
    f_i = F * us["force_density_to_internal"][0]
    c_p = 1.0 / (mu_i + rho_i / dt_i)  # §3.2, h_int = 1 (cubic cells)
    oracle_kw = dict(signature=(dt_i, rho_i, mu_i, f_i, 0.0, 0.0), c_p=c_p, gauged=True,
                     collocated_advection=(scheme == "collocated" and bool(advection)),
                     cells=(N, N, N))
    return s, oracle_kw


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()

    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("sphere", "bed"))
    ap.add_argument("--scheme", choices=("staggered", "collocated"), required=True)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--phi", type=float, default=0.125)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=6.0, help="nu dt / h^2 (ignored with --dt)")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--advection", default=None, help="advection scheme name (default: Stokes)")
    ap.add_argument("--arrangement", default=None, help="bed: .npz with centres, radii, L")
    ap.add_argument("--settings", choices=("production", "tight"), default="production")
    ap.add_argument("--window", type=int, nargs="+", default=[0, 3, 5, 8],
                    help="0 = plain march (accelerate=False)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--trace", action="store_true", help="print the per-step residual")
    a = ap.parse_args()
    st = settings_of(a.settings)
    if a.max_steps is not None:
        st["max_steps"] = a.max_steps
    rho = F = 1.0
    if a.case == "sphere":
        L = L1
        R = (3.0 * a.phi / (4.0 * math.pi)) ** (1.0 / 3.0) * L
        c = 0.5 * L

        def sdf_fn(cx, cy, cz):
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
            return np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - R

        def drag_K(u):
            return F * L ** 3 / (6.0 * math.pi * a.mu * R * u)
    else:
        z = np.load(a.arrangement)
        cen, rad, L = np.asarray(z["centres"], float), np.asarray(z["radii"], float), float(z["L"])
        Dm = 2.0 * float(rad.mean())
        npart = len(rad)

        def sdf_fn(cx, cy, cz):
            from scipy.spatial import cKDTree
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
            pts = np.stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")], axis=1)
            tree = cKDTree(np.mod(cen, L), boxsize=L)
            k = min(4, len(cen))
            d, idx = tree.query(np.mod(pts, L), k=k)
            return (d - rad[idx]).min(axis=1).reshape(X.shape, order="C")

        def drag_K(u):  # A1 common.drag_K: f V / (3 pi mu D U_s N_p)
            return F * L ** 3 / (3.0 * math.pi * a.mu * Dm * u * npart)
    h = L / a.N
    dt = a.dt if a.dt is not None else a.beta * rho * h * h / a.mu
    for w in a.window:
        s, okw = build_solver(flow, a.scheme, a.N, sdf_fn, L, rho=rho, mu=a.mu, F=F, dt=dt,
                              advection=a.advection, settings=st)
        trace = []
        accref = {}

        def cb(steps, phase, trace=trace, accref=accref):
            acc = accref.get("acc")
            trace.append((steps, phase, acc.residual if acc is not None else None,
                          acc.ritzRadius if acc is not None else None,
                          acc.status if acc is not None else None))
            if a.trace and acc is not None:
                print(f"  {steps:5d} {phase:10s} res={acc.residual:.3e} ritz={acc.ritzRadius:.5f} "
                      f"mk={acc.mk} st={acc.status} rst={acc.numRestarts}", flush=True)

        def monitor(s=s):
            return float(s.get_u().mean())
        t0 = time.perf_counter()
        res = march_to_steady(s, monitor, rtol=st["rtol"], max_steps=st["max_steps"],
                              accelerate=(w > 0), window=max(w, 1), callback=cb, oracle_kw=okw,
                              holder=accref)
        wall = time.perf_counter() - t0
        acc = res["acc"]
        um = res["monitor"]
        K = drag_K(um) if math.isfinite(um) and um != 0 else math.nan
        rec = dict(label=a.label, case=a.case, scheme=a.scheme, N=a.N, phi=a.phi, mu=a.mu, dt=dt,
                   advection=a.advection, settings=a.settings, window=w,
                   converged=res["converged"], reason=res["reason"], steps=res["steps"],
                   accelerated_steps=res["accelerated_steps"], num_restarts=res["num_restarts"],
                   monitor=um, K=K, wall_s=wall, c_p=okw["c_p"],
                   status=acc.status if acc else None, acc_reason=acc.reason if acc else None,
                   num_resets=acc.numResets if acc else None,
                   ritz_max=acc.ritz_max if acc else None,
                   sum_abs_alpha_max=acc.sum_abs_alpha_max if acc else None,
                   cond_drops=acc.cond_drops if acc else None,
                   zero_diag_drops=acc.zero_diag_drops if acc else None,
                   mixed_steps=acc.mixed_steps if acc else None,
                   trace=[list(t) for t in trace] if acc else None)
        if acc:
            def rmax(ph, trace=trace):
                v = [t[3] for t in trace if t[1] == ph and t[3] is not None and math.isfinite(t[3])]
                return max(v) if v else None
            rec["ritz_max_accelerate"] = rmax("accelerate")
            rec["ritz_max_certify"] = rmax("certify")
            rec["ritz_over_1p001"] = sum(1 for t in trace if t[3] is not None and math.isfinite(t[3])
                                         and t[3] > 1.001 and t[4] == "active")
            first = next((t for t in trace if t[4] == "unstable"), None)
            rec["unstable_step"] = first[0] if first else None
            rec["unstable_phase"] = first[1] if first else None
        print(f"{a.label:>10s} {a.case} {a.scheme:10s} N={a.N} {a.settings:10s} m={w} "
              f"conv={res['converged']} reason={res['reason']} steps={res['steps']} "
              f"acc={res['accelerated_steps']} K={K:.10f} restarts={res['num_restarts']} "
              f"ritzA={rec.get('ritz_max_accelerate')} ritzC={rec.get('ritz_max_certify')} "
              f"unstable@{rec.get('unstable_step')}/{rec.get('unstable_phase')} "
              f"sum|a|max={rec['sum_abs_alpha_max']} "
              f"drops={rec['cond_drops']}/{rec['zero_diag_drops']} wall={wall:.2f}s", flush=True)
        if a.json:
            with open(a.json, "a") as fjs:
                fjs.write(json.dumps(rec) + "\n")
        s = None  # release the solver before the next window's


if __name__ == "__main__":
    main()
