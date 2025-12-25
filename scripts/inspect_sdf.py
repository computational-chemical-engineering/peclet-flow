import sys
import os
import numpy as np

# Add build dir to path so we can import pnm_backend
sys.path.append(os.path.join(os.getcwd(), 'build'))
import pnm_backend

res = (16, 64, 16)
# Generate SDF as in the notebook
# The notebook used: pnm_backend.generate_slab_sdf(res, 0.2)
sdf_flat = pnm_backend.generate_slab_sdf(pnm_backend.int3(*res), 0.2)
# sdf_flat might be a list or a vector, check type or assume list
sdf = np.array(sdf_flat).reshape(res, order='F')

# Extract profile along Y at center X, Z
ix = res[0] // 2
iz = res[2] // 2
sdf_y = sdf[ix, :, iz]

print("Y Index | SDF Value | State")
dy = 1.0 / res[1]
fluid_indices = []
for iy, val in enumerate(sdf_y):
    y_pos = (iy + 0.5) * dy
    st = "FLUID" if val < 0 else "SOLID"
    print(f"{iy:3d} | {y_pos:.4f} | {val:.4f} | {st}")
    if val < 0:
        fluid_indices.append(y_pos)

if len(fluid_indices) > 0:
    # Width is number of fluid cells * dy
    # Or span.
    h_fluid = len(fluid_indices) * dy
    print(f"\nEstimated Fluid Width: {h_fluid:.4f}")
    y_start = fluid_indices[0] - 0.5*dy
    y_end = fluid_indices[-1] + 0.5*dy
    print(f"Fluid Start: {y_start:.4f}")
    print(f"Fluid End:   {y_end:.4f}")
else:
    print("\nNo fluid cells found.")
