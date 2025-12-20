import sys
import os

# Add build directory to path to find the module
# Adjust this path based on where cmake outputs the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import pnm_backend
import numpy as np

def test_read():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "test_sdf.vti"
    print(f"Reading {filename}...")
    
    sdf_data = pnm_backend.SDFReader.read_vti(filename)
    
    print("Read successful.")
    print(f"Resolution: {sdf_data.resolution}")
    print(f"Origin: {sdf_data.origin}")
    print(f"Spacing: {sdf_data.spacing}")
    
    values = sdf_data.sdf_values
    print(f"Data shape: {values.shape}")
    print(f"Sample data: {values[0,0,0]}")
    
    # Basic check
    expected_shape = (10, 10, 10)
    # Note: Our reader returns (Z, Y, X) order in numpy array because we passed:
    # {d.resolution[2], d.resolution[1], d.resolution[0]}
    # And we know the data in memory is X-fastest.
    # So if we access arr[k, j, i], we access z[k], y[j], x[i].
    
    # Original generation:
    # shape (10, 10, 10) -> (nx, ny, nz)
    # flattened column-major (F) -> x changes fastest.
    # So memory is: x0y0z0, x1y0z0, ...
    
    # Numpy array from buffer with shape (Nz, Ny, Nx):
    # Default numpy is row-major (C-style).
    # So arr[0,0,1] -> offset 1.
    # If we map shape (Nz, Ny, Nx) to memory x0y0z0...
    # Then the LAST index (Nx) corresponds to fastest moving memory.
    # So arr[k, j, i] accesses offset ... + i.
    # This matches.
    
    if list(sdf_data.resolution) == [10, 10, 10]:
        print("Resolution matches.")
    else:
        print("Resolution MISMATCH!")
        exit(1)

if __name__ == "__main__":
    test_read()
