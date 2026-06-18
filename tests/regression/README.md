# sdflow single-GPU regression suite

`sdflow_regression.py` is an accuracy **and** efficiency regression suite for the `sdflow` cut-cell IBM
Stokes solver. It runs three creeping-flow grid-convergence studies and compares the results against a
saved baseline (`perf_baseline.json`), so a code change that degrades either accuracy or solver efficiency
is caught.

## Cases (all single-GPU, Stokes / creeping flow)

| case | geometry | accuracy metric | reference |
|------|----------|-----------------|-----------|
| `zh_sphere` | one simple-cubic sphere (φ=0.216) | drag factor `K` | Zick & Homsy (1982), `K≈7.442` |
| `random_spheres` | small packed bed of 8 jittered spheres (fixed seed) | dimensionless permeability `k* = k/L² = μ⟨u⟩/(F·N²)` | Richardson extrapolation |
| `hollow_rings` | small packed bed of 3 Raschig rings (hollow cylinders) | `k*` | Richardson extrapolation |

The geometry is self-similar in `N` (the same relative shape resolved on a finer grid), so the
**dimensionless** metric (`K`, `k*`) converges as `N→∞` and a true order of convergence can be fit.

## What is recorded (per case, per grid N)

- **Accuracy:** the metric (`K`/`k*`), the fitted order of convergence `p` (`f(N)=f_inf + C·N^-p`), the
  Richardson-extrapolated value `f_inf`, and (for `zh_sphere`) the error vs the Zick & Homsy reference.
- **Efficiency:** total pressure-solver (MG-PCG) iterations, per-step pressure iterations (median),
  Picard outer iterations, the number of steps to steady state, and the wall-clock time.
- **Correctness:** the cut-cell flux divergence and the max velocity in the deep solid (no-slip).

Typical baseline orders: `zh_sphere` p≈2.4, `random_spheres` p≈2.2, `hollow_rings` p≈1.3 (the thin ring
walls converge geometrically slower). Pressure iters are ~6–10/step and grid-independent (the MG-PCG
signature).

## Usage

Build the single-GPU module first (`cmake -S . -B build && cmake --build build -j`). Then, from the repo
root with the venv active:

```bash
PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py            # run + check (exit 0/1)
PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py --update   # (re)write the baseline
PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py --quick    # fast smoke (coarse grids)
PYTHONPATH=$PWD/build python tests/regression/sdflow_regression.py --cases zh_sphere
```

The full suite runs in ~20 s on an RTX 5080. **Re-record the baseline (`--update`) only when a change is
expected to move the numbers** (and review the diff); otherwise a baseline change masks a regression.

## Tolerances (in `sdflow_regression.py`, `TOL`)

metric ±1.5% per grid · order ±0.4 · extrapolated ±2% · total pressure iters ±25% · per-step pressure
iters ±2 · divergence ≤ max(1e-7, 3×baseline). Wall-clock time is recorded and printed but not asserted
(machine-dependent).
