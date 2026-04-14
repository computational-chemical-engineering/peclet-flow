import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

import pnm_backend


def test_ibm_sphere_flow():
    nx = ny = nz = 32
    spacing = [1.0, 1.0, 1.0]

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = nx / 2, ny / 2, nz / 2
    radius = 8.0
    sdf_xyz = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) - radius
    sdf_zyx = np.transpose(sdf_xyz, (2, 1, 0)).astype(np.float32)

    solver = pnm_backend.CFDSolver([nx, ny, nz], spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)
    solver.set_body_force(pnm_backend.float3(1.0, 0.0, 0.0))
    solver.set_rho(1.0)
    solver.set_mu(0.01)
    solver.set_pressure_solver_params(50)
    solver.set_velocity_solver_params(20)

    for _ in range(10):
        solver.step(0.05)

    u = np.array(solver.get_u(), copy=False)
    assert np.isfinite(u).all()
    assert abs(u[nz // 2, ny // 2, nx // 2]) < 1e-5


if __name__ == "__main__":
    test_ibm_sphere_flow()
