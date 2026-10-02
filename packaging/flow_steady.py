"""Steady marches: :func:`march_to_steady` and :class:`MarchResult`.

March a :class:`peclet.flow.Solver` / :class:`peclet.flow.SolverColocated` to its steady state and
certify it with the study's stop instrument (a geometric-remainder bound on the change of a scalar
``monitor()``), optionally accelerated by type-II Anderson mixing of the march state
(``doc/steady_acceleration.md``, revision 2).

The accelerated march has two phases. Phase A applies Anderson mixing until the relative velocity
residual reaches ``(1 - slow_rate) * rtol``. Phase B certifies with the unchanged instrument on
consecutive PLAIN steps, so the reported state is a plain-march state that passed the same
acceptance test as the unaccelerated march. Phase B has a budget of ``2 * (num_passes + 3)`` blocks
and an early exit when a block fails on the remainder bound alone with a ratio R in
``[slow_rate**check_every, 1)``. On either, the target is tightened tenfold and acceleration
resumes with the history intact. If the plain steps grow, acceleration is disabled and the march
finishes as the plain march would. The same happens after a failed certification when phase A
stagnated: its residual did not fall by ``slow_rate ** (10 * window)`` (the plain march's assumed
rate over those calls) within ``10 * window`` calls. ``Solver.step()`` itself is never changed.

``accelerate=False`` is the plain march with the same instrument, step for step.

Scope (the accelerator refuses everything else with a named error): the staggered ``Solver``, and
``SolverColocated`` with the fluid-only ``'ghost'`` scheme; Stokes or finite Re; periodic or domain
boundary conditions; static geometry. Refused: the other collocated schemes, VoF, phase change,
transported scalars, porous continuity, variable density or viscosity, drag, cell forces, moving
scenes, a superficial-velocity target, the pressure warm start and the balanced-force projection.

Memory: the history holds ``(2 window + 3)`` copies of the march state in double precision,
``(2 window + 3) * n_fields * 8`` bytes per padded cell (``n_fields`` = 4, or 7 for the collocated
solver with projected-face advection).

Checkpoints: the Anderson history is not saved. Checkpoint with ``get_field`` / ``set_field`` of
``u, v, w, p`` (not ``set_state``, which restores the velocity only); a restarted accelerated march
rebuilds its history in a few steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["march_to_steady", "MarchResult"]


@dataclass(frozen=True)
class MarchResult:
    """The outcome of :func:`march_to_steady`.

    ``converged`` is True only for ``reason == "certified"``. ``reason`` is one of ``"certified"``,
    ``"max_steps"`` or ``"diverged"`` (a non-finite monitor on the plain march). ``steps`` counts
    every solver step, ``accelerated_steps`` those taken in the accelerated phase; ``monitor`` is
    the last value of ``monitor()``.

    converged=True certifies that the state passed the stop test on consecutive plain steps at this
    dt (stationarity); it does not certify that a plain march from the initial state would reach
    it.
    """

    converged: bool
    steps: int
    accelerated_steps: int
    reason: str
    num_restarts: int
    monitor: float


class _Counter:
    __slots__ = ("steps", "accelerated_steps")

    def __init__(self):
        self.steps = 0
        self.accelerated_steps = 0


def _certify(step_fn, monitor, *, rtol, check_every, num_passes, slow_rate, roundoff, max_steps,
             counter, callback, phase, budget=None, growth_residual=None, growth_ref=None,
             slow_exit=False):
    """The stop instrument on consecutive steps of ``step_fn``, from a fresh block state.

    Every ``check_every`` steps, d = the change of ``monitor()`` over the block and R = d / (the
    previous block's d). A block passes when 0 < R < 1 and |d| / (1 - max(R,
    slow_rate**check_every)) < rtol |m|, or when |d| <= roundoff |m|; ``num_passes`` consecutive
    passes stop. Returns "pass", "max" (``counter.steps`` reached ``max_steps``), "budget" (after
    ``budget`` blocks, never when ``budget`` is None), "diverged" (a non-finite monitor with no
    budget), "growth" (with ``growth_ref``: ``growth_residual()`` above 10 ``growth_ref`` or
    non-finite, or a non-finite monitor) and, with ``slow_exit``, "slow" at the first block that
    fails while slow_rate**check_every <= R < 1 (a clean geometric tail, failing on its remainder
    bound alone). The last two are the driver's policy, not part of the instrument.
    """
    slow_block = slow_rate ** check_every
    prev = dprev = None
    passes = 0
    blocks = 0
    it = 0
    while True:
        if counter.steps >= max_steps:
            return "max"
        step_fn()
        counter.steps += 1
        if callback is not None:
            callback(counter.steps, phase)
        if growth_ref is not None:
            r = growth_residual()
            if not math.isfinite(r) or r > 10.0 * growth_ref:
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
            R = None
            if not ok and dprev:
                R = d / dprev
                ok = 0.0 < R < 1.0 and abs(d) / (1.0 - max(R, slow_block)) < rtol * abs(m)
            passes = passes + 1 if ok else 0
            if slow_exit and not ok and R is not None and slow_block <= R < 1.0:
                return "slow"
            if passes >= num_passes:
                return "pass"
            dprev = d
        prev = m
        blocks += 1
        if budget is not None and blocks >= budget:
            return "budget"


def march_to_steady(solver, monitor, rtol=1e-4, max_steps=5000, accelerate=True, window=5,
                    check_every=5, num_passes=3, slow_rate=0.997, roundoff=1e-11, callback=None):
    """March ``solver`` to its steady state; return a :class:`MarchResult`.

    Parameters
    ----------
    solver : peclet.flow.Solver or peclet.flow.SolverColocated
        A configured solver (geometry, properties, dt, forcing). Its current state is the start.
    monitor : callable
        ``monitor() -> float``: the scalar the stop instrument watches, e.g. the mean velocity along
        the forcing. Under MPI it must return the same value on every rank (a global reduction).
    rtol : float
        Relative bound on the change of ``monitor()`` still to come.
    max_steps : int
        Cap on the total number of solver steps.
    accelerate : bool
        Anderson acceleration (``True``) or the plain march (``False``). Raises on a configuration
        the accelerator refuses, with the reason and the hint to pass ``accelerate=False``.
    window : int
        Anderson window, 1 to 8.
    check_every, num_passes, slow_rate, roundoff
        The stop instrument: block length in steps, consecutive passing blocks required, the
        slowest per-step contraction assumed, and the relative block change accepted as round-off.
    callback : callable, optional
        ``callback(steps, phase)`` after every step, ``phase`` in ``{"accelerate", "certify",
        "plain"}`` (logging, checkpoints).
    """
    counter = _Counter()
    acc = None
    instrument = dict(rtol=rtol, check_every=check_every, num_passes=num_passes,
                      slow_rate=slow_rate, roundoff=roundoff, max_steps=max_steps,
                      counter=counter, callback=callback)

    def result(converged, reason):
        return MarchResult(converged=converged, steps=counter.steps,
                           accelerated_steps=counter.accelerated_steps, reason=reason,
                           num_restarts=acc.num_restarts if acc is not None else 0,
                           monitor=float(monitor()))

    def plain_tail():
        out = _certify(solver.step, monitor, phase="plain", **instrument)
        if out == "pass":
            return result(True, "certified")
        if out == "diverged":
            return result(False, "diverged")
        return result(False, "max_steps")

    if not accelerate:
        return plain_tail()
    acc = solver.diagnostics.anderson_accelerator(window=window)
    target = (1.0 - slow_rate) * rtol  # on the relative VELOCITY residual
    budget = 2 * (num_passes + 3)
    while counter.steps < max_steps:
        # phase A: accelerate
        best, since, stagnated = math.inf, 0, False
        while counter.steps < max_steps and acc.status == "active" and acc.residual > target:
            acc.step(True)
            counter.steps += 1
            counter.accelerated_steps += 1
            if callback is not None:
                callback(counter.steps, "accelerate")
            # Progress is measured against the plain march's rate the instrument already assumes
            # (slow_rate per step): stagnation means Anderson is not beating the plain march over
            # 10 * window calls, which is what justifies handing over to it (review R1; a halving
            # per 10 * window calls stopped acceleration on the dense bed while it was still
            # winning).
            if acc.residual <= slow_rate ** (10 * window) * best:
                best, since = acc.residual, 0
            else:
                since += 1
            if since >= 10 * window:
                stagnated = True
                break
        if acc.status != "active":
            return plain_tail()
        # phase B: certify on consecutive plain steps
        out = _certify(lambda: acc.step(False), monitor, phase="certify", budget=budget,
                       growth_residual=lambda: acc.residual, growth_ref=acc.residual,
                       slow_exit=True, **instrument)
        if out == "pass":
            return result(True, "certified")
        if out == "max":
            return result(False, "max_steps")
        if out == "growth" or stagnated:
            acc.disable()
            return plain_tail()
        target *= 0.1  # "budget" or "slow": resume acceleration with the history intact
    return result(False, "max_steps")
