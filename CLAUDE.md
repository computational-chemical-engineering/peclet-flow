# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated incompressible Navier-Stokes CFD solver for porous media simulations. Uses staggered MAC grid with Immersed Boundary Method (IBM) to handle complex geometries defined by Signed Distance Functions (SDF). The solver implements Newton-Raphson iteration with pressure projection.

## Build Commands

```bash
# Build the Python extension module
mkdir -p build && cd build && cmake .. && cmake --build .

# Output: build/pnm_backend.so
```

**Requirements:** CUDA Toolkit, CMake 3.18+, Python 3.10+, C++17

## Running Tests and Verification

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run unit tests
python tests/test_cfd_solver.py
python tests/test_implicit_poiseuille.py
python tests/test_fluid_fractions.py

# Run verification scripts
python scripts/verify_poiseuille.py      # Plane Poiseuille (analytical solution)
python scripts/verify_periodic_spheres.py # Flow around sphere packing
python scripts/verify_divergence.py       # Check incompressibility
```

## Architecture

### Core Components

- **`pnm_backend`** - Python extension module (built via pybind11)
- **`CFDSolver`** - Main solver class with GPU-accelerated kernels
- **`MacGrid`** - Staggered grid data structure (SoA layout for GPU coalescing)
- **`SDFData`/`SDFReader`** - Geometry loading from VTI files

### Memory Layout

- Linear indexing: `I = x + y*nx + z*nx*ny` (x is fastest)
- Python arrays: Fortran order `order='F'` with shape `(nx, ny, nz)`
- Periodic boundaries with wrapping: `(x % res.x + res.x) % res.x`

### Numerical Method

1. **Advection**: TVD scheme with Koren limiter
2. **Diffusion**: Crank-Nicolson (θ configurable 0.5-1.0)
3. **Momentum Solve**: Implicit Jacobian with Red-Black Gauss-Seidel
4. **Pressure Projection**: Poisson solve for auxiliary scalar φ
5. **IBM**: Robust Scaled method with D_rescale for near-wall handling

### Key Source Files

- `src/cfd_solver.cuh` - Data structures (MacGrid, IBM_Data) and public API
- `src/cfd_solver.cu` - Main solver implementation
- `src/cfd_solver_ibm.cu` - Immersed Boundary Method kernels
- `src/bindings.cpp` - Python/C++ interop via pybind11
- `doc/implementation_plan.md` - Detailed architecture documentation

## Python API Usage

```python
import pnm_backend

# Create solver
res = pnm_backend.int3(32, 32, 32)
spacing = pnm_backend.float3(dx, dx, dx)
solver = pnm_backend.CFDSolver(res, spacing)

# Initialize geometry from SDF
sdf_data = pnm_backend.SDFData(values, res, origin, spacing)
solver.initialize(sdf_data)

# Configure solver
solver.set_rho(1.0)
solver.set_mu(0.01)
solver.set_body_force(pnm_backend.float3(1e-2, 0, 0))
solver.set_cfl(0.5)
solver.set_pressure_solver_params(max_iter=500, tol=1e-5)
solver.set_velocity_solver_params(max_iter=50, tol=1e-5)
solver.set_diffusion_theta(1.0)  # 0.5=CN, 1.0=Implicit

# Time step
solver.step(dt)

# Get results (returns flat array, reshape with order='F')
u = np.array(solver.get_u()).reshape((nx, ny, nz), order='F')
```

## Conventions

- **SDF sign**: Negative inside solid, positive in fluid
- **CUDA kernels**: Named with `_kernel` suffix
- **Staggered grid**: u at (i+1/2,j,k), v at (i,j+1/2,k), w at (i,j,k+1/2), p at cell centers
