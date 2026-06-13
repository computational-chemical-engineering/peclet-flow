"""Test the cut-cell fluid-fraction (face openness) computation."""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))

import pnm_backend


def test_fluid_fraction():
    nx = ny = nz = 10
    spacing = [1.0, 1.0, 1.0]

    sdf_zyx = np.zeros((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                sdf_zyx[k, j, i] = float(i) - 5.5

    solver = pnm_backend.CFDSolver([nx, ny, nz], spacing)
    solver.initialize(sdf_zyx, [0.0, 0.0, 0.0], spacing)

    vf = np.array(solver.get_fluid_fraction(0, pnm_backend.float3(0, 0, 0)))
    af_u = np.array(solver.get_fluid_fraction(1, pnm_backend.float3(-0.5, 0, 0)))

    assert np.isclose(vf[5], 0.0)
    assert np.isclose(vf[6], 1.0)
    assert np.isclose(af_u[5], 0.0)
    assert np.isclose(af_u[6], 0.5)
    assert np.isclose(af_u[7], 1.0)


if __name__ == "__main__":
    test_fluid_fraction()
