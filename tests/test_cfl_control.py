"""Test adaptive CFL / time-step control."""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

import pnm_backend


def test_auto_dt_smoke():
    n = 24
    dx = 1.0 / n
    spacing = [dx, dx, dx]

    coords = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    sdf_xyz = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) - 0.18
    sdf_zyx = np.transpose(sdf_xyz, (2, 1, 0)).astype(np.float32)

    solver = pnm_backend.CFDSolver([n, n, n], spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    solver.set_rho(1.0)
    solver.set_mu(0.01)
    solver.set_pressure_solver_params(100)
    solver.set_velocity_solver_params(10)
    solver.set_pressure_multigrid_enabled(True)
    solver.set_pressure_multigrid_params(4, 2, 2, 32, 2)

    solver.step(-1.0)
    solver.step(-1.0)

    u = np.array(solver.get_u(), copy=False)
    assert np.isfinite(u).all()
    assert np.max(np.abs(u)) > 0.0


if __name__ == "__main__":
    test_auto_dt_smoke()
