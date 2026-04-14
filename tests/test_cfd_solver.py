import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

import pnm_backend


def make_sphere_sdf_zyx(n, radius):
    dx = 1.0 / n
    coords = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    sdf_xyz = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) - radius
    return np.transpose(sdf_xyz, (2, 1, 0)).astype(np.float32), dx


def test_single_sphere_large_dt_continuation():
    n = 24
    sdf_zyx, dx = make_sphere_sdf_zyx(n, radius=0.18)
    spacing = [dx, dx, dx]
    solver = pnm_backend.CFDSolver([n, n, n], spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)
    solver.set_rho(1.0)
    solver.set_mu(1.0)
    solver.set_body_force(pnm_backend.float3(10.0, 0.0, 0.0))
    solver.set_pressure_solver_params(200)
    solver.set_velocity_solver_params(30)
    solver.set_pressure_multigrid_enabled(True)
    solver.set_pressure_multigrid_params(4, 2, 2, 32, 2)
    solver.set_velocity_multigrid_enabled(False)
    solver.set_outer_iterations(100)
    solver.set_outer_tolerance(1e-4)

    residuals = []
    for _ in range(3):
        solver.step(1.0)
        u = np.array(solver.get_u(), copy=False)
        assert np.isfinite(u).all()
        residuals.append(solver.get_momentum_residual_max(True))

    assert residuals[-1] < residuals[0]


if __name__ == "__main__":
    test_single_sphere_large_dt_continuation()
