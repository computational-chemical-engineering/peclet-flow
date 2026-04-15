"""Helpers for continuation sweeps and cross-resolution solver initialization."""

import numpy as np


def fit_force_law(u_mean_samples, force_samples):
    """Fit ``force ~= a*u_mean + b*u_mean**2`` in a least-squares sense."""
    u_mean_samples = np.asarray(u_mean_samples, dtype=np.float64)
    force_samples = np.asarray(force_samples, dtype=np.float64)
    design = np.column_stack([u_mean_samples, u_mean_samples ** 2])
    coeffs, *_ = np.linalg.lstsq(design, force_samples, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def predict_u_mean(force, linear_coeff, quadratic_coeff):
    """Invert ``force = a*u_mean + b*u_mean**2`` for the positive root."""
    force = float(force)
    linear_coeff = float(linear_coeff)
    quadratic_coeff = float(quadratic_coeff)
    if abs(quadratic_coeff) < 1e-14:
        if abs(linear_coeff) < 1e-14:
            raise ValueError("force law is degenerate")
        return force / linear_coeff
    disc = linear_coeff ** 2 + 4.0 * quadratic_coeff * force
    if disc < 0.0:
        raise ValueError("force law has no real solution for the requested force")
    return (-linear_coeff + np.sqrt(disc)) / (2.0 * quadratic_coeff)


def scale_solver_state(solver, previous_u_mean, target_u_mean, previous_force, target_force):
    """Scale an already converged solver state in place for continuation."""
    if abs(previous_u_mean) < 1e-14:
        raise ValueError("previous_u_mean must be non-zero")
    if abs(previous_force) < 1e-14:
        raise ValueError("previous_force must be non-zero")
    solver.scale_state(
        float(target_u_mean) / float(previous_u_mean),
        float(target_force) / float(previous_force),
    )


def extract_solver_state(solver):
    """Copy ``u``, ``v``, ``w`` and ``p`` from a solver into NumPy arrays."""
    return {
        "u": np.asarray(solver.get_u(), dtype=np.float64),
        "v": np.asarray(solver.get_v(), dtype=np.float64),
        "w": np.asarray(solver.get_w(), dtype=np.float64),
        "p": np.asarray(solver.get_p(), dtype=np.float64),
    }


def load_solver_state(solver, state):
    """Load a previously extracted state into a solver."""
    solver.set_state(
        np.asarray(state["u"], dtype=np.float64),
        np.asarray(state["v"], dtype=np.float64),
        np.asarray(state["w"], dtype=np.float64),
        np.asarray(state["p"], dtype=np.float64),
    )


def _resample_axis(field, target_coords, axis):
    """Periodically resample one axis of a structured field."""
    source_coords = np.linspace(0.0, 1.0, field.shape[axis], endpoint=False)
    field = np.moveaxis(field, axis, 0)
    resampled = np.empty((target_coords.size,) + field.shape[1:], dtype=np.float64)
    for index in np.ndindex(field.shape[1:]):
        resampled[(slice(None),) + index] = np.interp(
            target_coords,
            source_coords,
            field[(slice(None),) + index],
            period=1.0,
        )
    return np.moveaxis(resampled, 0, axis)


def resample_field_linear(field, target_shape):
    """Periodically resample a scalar field to a new ``(nz, ny, nx)`` shape."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("field must have shape (nz, ny, nx)")
    target_shape = tuple(int(v) for v in target_shape)
    if len(target_shape) != 3:
        raise ValueError("target_shape must have length 3")

    target_z = np.linspace(0.0, 1.0, target_shape[0], endpoint=False)
    target_y = np.linspace(0.0, 1.0, target_shape[1], endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_shape[2], endpoint=False)

    resampled = _resample_axis(field, target_x, axis=2)
    resampled = _resample_axis(resampled, target_y, axis=1)
    resampled = _resample_axis(resampled, target_z, axis=0)
    return resampled


def resample_state_linear(state, target_shape):
    """Resample all entries of a state dictionary with linear interpolation."""
    return {
        name: resample_field_linear(field, target_shape)
        for name, field in state.items()
    }
