"""Contract: the public setters refuse a NaN/Inf or physically inadmissible value with ValueError.

Validation lives in the binding (src/flow_bindings_checks.hpp, used from src/flow_bindings.cpp);
the message names the setter, the parameter, the admissible range with its dimension and the
offending value. Each validated argument is covered on both solvers twice: every bad value raises
ValueError naming the parameter, and one valid value is accepted (where a call needs a fixture
this file does not build -- a scene, a wetting wall -- the valid value is shown to pass validation
by the call reaching the C++ check behind it, which raises RuntimeError instead).

Where a sign or a 0 is a switch the check is "finite" only (set_velocity_residual_tolerance: < 0
follows the pressure rtol; set_contact_angle_dynamic sigma: <= 0 takes set_surface_tension's;
set_vof_block_assign every: 0 never re-assigns; set_mass_flux_uniform: the sign is the direction).

Run: PYTHONPATH=<build> python -m pytest -q tests/python/test_input_validation.py
"""
import math
import re

import numpy as np
import pytest

from peclet import flow

NAN, INF = math.nan, math.inf
N = 8
CLASSES = [flow.Solver, flow.SolverColocated]


def _box(cls):
    return cls((N, N, N), extent=(1.0, 1.0, 1.0))


def _fluid(cls, **bc):
    """A physical all-fluid box with properties and a geometry (what add_scalar / VoF need)."""
    s = _box(cls)
    s.set_rho(1.0)
    s.set_mu(0.1)
    s.set_dt(0.01)
    for face, (kind, vel) in bc.items():
        s.set_domain_bc(face, kind, vel)
    s.set_pressure_geometry(np.full((N, N, N), 10.0, order="F"))
    return s


def _scalar(cls):
    s = _fluid(cls)
    s.add_scalar("c", 0.1)
    return s


def _vof(cls):
    s = _fluid(cls)
    s.enable_vof()
    return s


def _vof_open(cls):
    """VoF with an inflow -x face and an outflow +x face."""
    s = _fluid(cls, **{"-x": ("inflow", (0.1, 0.0, 0.0)), "+x": ("outflow", (0.0, 0.0, 0.0))})
    s.enable_vof()
    return s


def _phase_change(cls):
    """Phase change is staggered-only: SolverColocated gets VoF alone (its setter then refuses)."""
    s = _vof(cls)
    if cls is flow.Solver:
        s.enable_phase_change(1.0, 10.0)
    return s


def _blocks(cls):
    s = _vof(cls)
    s.enable_vof_blocks([(4.0, 4.0, 4.0, 2.0)])
    return s


def _staggered_only(match):
    """Accepted on Solver; on SolverColocated past validation into the solver's refusal."""
    return lambda cls: None if cls is flow.Solver else (RuntimeError, match)


_NO_SCENE = (RuntimeError, "set_scene first")  # past validation, refused by the solver
_NO_WALL = (RuntimeError, "wetting wall")

# (id, call(cls, v), bad values, good value, parameter as the message names it, good outcome)
# `good outcome` None = the call returns; (exc, regex) = it gets past validation into a C++ check;
# a callable maps the class to one of those.
CASES = [
    ("Solver.nx", lambda c, v: c(v, N, N), [-1, 0], N, "nx", None),
    ("Solver.nz", lambda c, v: c(N, N, v), [0], N, "nz", None),
    ("Solver.origin", lambda c, v: c((N, N, N), extent=(1.0, 1.0, 1.0), origin=(0.0, v, 0.0)),
     [NAN, INF], -0.5, "origin[1]", None),
    ("Solver.global_cells",
     lambda c, v: c((N, N, N), extent=(1.0, 1.0, 1.0), global_cells=(N, N, v)), [-1, 0], N,
     "global_cells[2]", None),
    ("set_rho", lambda c, v: _box(c).set_rho(v), [NAN, INF, 0.0, -1.0], 1000.0, "rho", None),
    ("set_mu", lambda c, v: _box(c).set_mu(v), [NAN, INF, -1e-3], 0.0, "mu", None),
    ("set_dt", lambda c, v: _box(c).set_dt(v), [NAN, INF, 0.0, -1e-3], 1e-3, "dt", None),
    ("set_body_force", lambda c, v: _box(c).set_body_force((0.0, v, 0.0)), [NAN, INF], -9.81,
     "force[1]", None),
    ("set_superficial_velocity", lambda c, v: _box(c).set_superficial_velocity(True, "x", v),
     [NAN, INF], 0.5, "velocity", _staggered_only("staggered Solver only")),
    ("set_velocity_residual_tolerance", lambda c, v: _box(c).set_velocity_residual_tolerance(v),
     [NAN, INF], -1.0, "rtol", None),
    ("set_backflow_stabilization", lambda c, v: _box(c).set_backflow_stabilization(v),
     [NAN, -0.1, 1.5], 0.2, "beta", None),
    ("set_pressure_solver_params", lambda c, v: _box(c).set_pressure_solver_params(v), [-1, 0],
     20, "iters", None),
    ("set_pressure_multigrid", lambda c, v: _box(c).set_pressure_multigrid(True, v), [-1], 0,
     "levels", None),
    ("set_pressure_chebyshev.max_iter", lambda c, v: _box(c).set_pressure_chebyshev(True, v),
     [-1, 0], 120, "max_iter", None),
    ("set_pressure_chebyshev.rtol", lambda c, v: _box(c).set_pressure_chebyshev(True, 120, v),
     [NAN, -1e-9], 0.0, "rtol", None),
    ("set_pressure_pcg.max_iter", lambda c, v: _box(c).set_pressure_pcg(True, v), [-1, 0], 200,
     "max_iter", None),
    ("set_pressure_pcg.rtol", lambda c, v: _box(c).set_pressure_pcg(True, 200, v), [NAN, INF],
     1e-8, "rtol", None),
    ("set_pressure_fcg.max_iter", lambda c, v: _box(c).set_pressure_fcg(True, v), [-1, 0], 200,
     "max_iter", None),
    ("set_pressure_fcg.rtol", lambda c, v: _box(c).set_pressure_fcg(True, 200, v), [NAN, -1.0],
     1e-8, "rtol", None),
    ("set_velocity_multigrid.levels", lambda c, v: _box(c).set_velocity_multigrid(True, v),
     [-1, 0], 4, "levels", None),
    ("set_velocity_multigrid.vcycles", lambda c, v: _box(c).set_velocity_multigrid(True, 4, v),
     [-1, 0], 8, "vcycles", None),
    ("set_domain_bc", lambda c, v: _box(c).set_domain_bc("-x", "inflow", (v, 0.0, 0.0)),
     [NAN, INF], 0.1, "velocity[0]", None),
    ("set_instance_motion.lin_vel", lambda c, v: _box(c).set_instance_motion(0, (v, 0.0, 0.0)),
     [NAN, INF], 0.1, "lin_vel[0]", _NO_SCENE),
    ("set_instance_motion.ang_vel",
     lambda c, v: _box(c).set_instance_motion(0, ang_vel=(0.0, 0.0, v)), [NAN], 1.0,
     "ang_vel[2]", _NO_SCENE),
    ("set_instance_motion.center",
     lambda c, v: _box(c).set_instance_motion(0, center=(v, 0.5, 0.5)), [INF], 0.5, "center[0]",
     _NO_SCENE),
    ("set_instance_transform.translation",
     lambda c, v: _box(c).set_instance_transform(0, (0.0, v, 0.0)), [NAN], 0.1, "translation[1]",
     _NO_SCENE),
    ("set_instance_transform.quat",
     lambda c, v: _box(c).set_instance_transform(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, v)), [NAN],
     1.0, "quat[3]", _NO_SCENE),
    ("add_scalar.diffusivity", lambda c, v: _fluid(c).add_scalar("c", v), [NAN, -1.0], 0.0,
     "diffusivity", None),
    ("add_scalar.iters", lambda c, v: _fluid(c).add_scalar("c", 0.1, iters=v), [-1, 0], 1,
     "iters", None),
    ("set_scalar_bc", lambda c, v: _scalar(c).set_scalar_bc("c", "-x", "dirichlet", v),
     [NAN, INF], 1.0, "value", None),
    ("set_vof_cfl_limit", lambda c, v: _box(c).set_vof_cfl_limit(v), [NAN, 0.0, -0.25], 0.25,
     "value", None),
    ("set_capillary_cfl", lambda c, v: _vof(c).set_capillary_cfl(v), [NAN, 0.0, -1.0], 0.5,
     "factor", None),
    ("set_mass_flux_uniform", lambda c, v: _phase_change(c).set_mass_flux_uniform(v), [NAN, INF],
     -0.1, "mdot", _staggered_only("phase change is not enabled")),
    ("set_vof_inflow", lambda c, v: _vof_open(c).set_vof_inflow("-x", v), [NAN, -0.5, 1.5], 0.5,
     "value", None),
    ("set_vof_backflow", lambda c, v: _vof_open(c).set_vof_backflow("+x", v), [NAN, 1.5], 1.0,
     "value", None),
    ("enable_vof_blocks.center", lambda c, v: _vof(c).enable_vof_blocks([(v, 4.0, 4.0, 2.0)]),
     [NAN], 4.0, "seeds[0][0]", None),
    ("enable_vof_blocks.radius", lambda c, v: _vof(c).enable_vof_blocks([(4.0, 4.0, 4.0, v)]),
     [NAN, 0.0, -1.0], 2.0, "seeds[0][3]", None),
    ("set_vof_block_assign", lambda c, v: _blocks(c).set_vof_block_assign("lpt", v), [-1], 0,
     "every", None),
    ("set_contact_angle_dynamic.sigma",
     lambda c, v: _vof(c).set_contact_angle_dynamic(True, 90.0, 0.05, 1.0, v), [NAN, INF], 0.0,
     "sigma", _NO_WALL),
    ("set_property_model", lambda c, v: _vof(c).set_property_model("mu", "linear", "C", (v, 1.0)),
     [NAN, INF], 0.1, "params[0]", None),
    ("set_property_table.x",
     lambda c, v: _vof(c).set_property_table("mu", "C", (0.0, v), (0.1, 1.0)), [NAN], 1.0, "x[1]",
     None),
    ("set_property_table.y",
     lambda c, v: _vof(c).set_property_table("mu", "C", (0.0, 1.0), (v, 1.0)), [INF], 0.1, "y[0]",
     None),
]
if hasattr(flow.Solver, "init_mpi"):  # bound under PECLET_FLOW_MPI; a valid call would MPI_Init
    CASES.append(("init_mpi", lambda c, v: _box(c).init_mpi(N, v, N), [-1, 0], None, "gny", None))

_BAD = [(cls, c[0], c[1], b, c[4]) for cls in CLASSES for c in CASES for b in c[2]]
_GOOD = [(cls, c[0], c[1], c[3], c[5]) for cls in CLASSES for c in CASES if c[3] is not None]


@pytest.mark.parametrize("cls,case,call,value,param", _BAD,
                         ids=[f"{c[0].__name__}.{c[1]}={c[3]}" for c in _BAD])
def test_bad_value_raises_value_error_naming_the_parameter(cls, case, call, value, param):
    with pytest.raises(ValueError) as err:
        call(cls, value)
    msg = str(err.value)
    assert re.search(rf"{re.escape(param)} must be ", msg), msg
    assert "got" in msg, msg


@pytest.mark.parametrize("cls,case,call,value,outcome", _GOOD,
                         ids=[f"{c[0].__name__}.{c[1]}" for c in _GOOD])
def test_valid_value_is_accepted(cls, case, call, value, outcome):
    if callable(outcome):
        outcome = outcome(cls)
    if outcome is None:
        call(cls, value)
    else:
        exc, match = outcome
        with pytest.raises(exc, match=match):
            call(cls, value)


def test_message_states_range_unit_and_value():
    s = _box(flow.Solver)
    with pytest.raises(ValueError, match=r"^set_rho: rho must be finite and > 0 \(density\), "
                                         r"got -1$"):
        s.set_rho(-1.0)
    with pytest.raises(ValueError, match=r"^set_body_force: force\[2\] must be finite "
                                         r"\(force/volume\), got nan$"):
        s.set_body_force((0.0, 0.0, NAN))
    with pytest.raises(ValueError, match=r"^set_backflow_stabilization: beta must be finite and "
                                         r"in \[0, 1\] \(dimensionless\), got 1.5$"):
        s.set_backflow_stabilization(1.5)
    with pytest.raises(ValueError, match=r"^set_pressure_pcg: max_iter must be an integer >= 1, "
                                         r"got 0$"):
        s.set_pressure_pcg(True, 0)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_refused_value_leaves_the_setting_unchanged(cls):
    s = _box(cls)
    s.set_vof_cfl_limit(0.3)
    s.set_velocity_residual_tolerance(1e-6)
    with pytest.raises(ValueError):
        s.set_vof_cfl_limit(NAN)
    with pytest.raises(ValueError):
        s.set_velocity_residual_tolerance(INF)
    assert s.vof_cfl_limit() == 0.3
    assert s.velocity_residual_tolerance() == 1e-6


def test_existing_cell_count_check_is_unchanged():
    """The cells check predates this file: RuntimeError, message unchanged."""
    with pytest.raises(RuntimeError, match="every cell count must be positive"):
        flow.Solver((0, N, N), extent=(1.0, 1.0, 1.0))
