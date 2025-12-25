import sys
import os

# Add build to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'build')))
import pnm_backend
import numpy as np

def test_repro():
    res = pnm_backend.int3(32, 32, 32)
    spacing = pnm_backend.float3(1.0, 1.0, 1.0)
    solver = pnm_backend.CFDSolver(res, spacing)
    
    # Init
    origin = pnm_backend.float3(0,0,0)
    sdf_vals = np.ones(32*32*32, dtype=np.float32)
    sdf_data = pnm_backend.SDFData(sdf_vals.tolist(), res, origin, spacing)
    solver.initialize(sdf_data)

    print("Setting CFL=0.5")
    solver.set_cfl(0.5)
    
    dt_initial = solver.get_dt()
    print(f"DT immediately after set_cfl: {dt_initial}")
    
    print("Running step() with auto dt...")
    solver.step() # defaults to dt=-1.0
    
    dt_after = solver.get_dt()
    print(f"DT after step 1: {dt_after}")

if __name__ == "__main__":
    test_repro()
