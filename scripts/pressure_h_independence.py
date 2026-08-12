#!/usr/bin/env python
"""Is the cut-cell pressure multigrid resolution-independent on a wall-bounded domain?

STATUS: NOT YET TRUSTWORTHY -- do not quote numbers from this script for the wall-bounded case.
It works as intended with periodic boundaries and for modes that vary only in the periodic
directions, but ANY wall-normal variation in the prescribed field makes the projection stall (no
residual reduction at all, even at rtol = 0.1), so no h-independence conclusion can be drawn yet.
See "What is known" at the bottom of this docstring before using it.

A weak-scaling ladder that refines a DNS confounds two things: the discrete operator getting harder
to solve, and the turbulence itself changing as it becomes better resolved. This isolates the first.

The method is the textbook h-independence test. Fix a CONTINUOUS problem -- one velocity field
defined as a function of the physical coordinates, whose divergence is therefore a fixed function --
and sample it on grids of increasing resolution over the SAME physical box. Solve the pressure
Poisson equation once at each resolution, from a zero initial guess, to a fixed relative tolerance.
A multigrid whose convergence rate is resolution-independent takes the same number of iterations at
every resolution; a rising count is the solver, not the physics.

The solve is driven through the ordinary solver path with advection off and mu = 0, so the predictor
leaves the prescribed field untouched (u* = u) and the projection sees exactly the divergence we
prescribed.

    python scripts/pressure_h_independence.py --n 32,48,64,96
    python scripts/pressure_h_independence.py --n 32,48,64,96 --rhs broadband
    python scripts/pressure_h_independence.py --n 32,48,64 --bottom smoother   # contrast

Options worth sweeping: --bottom (auto/smoother) shows whether the coarse level is implicated,
--levels caps the hierarchy, --periodic-y removes the walls to separate the wall treatment from the
rest of the operator.

Needs PYTHONPATH pointing at a flow build; single rank, no MPI required.

What is known, all measured at 192x32x64 with mu = 0, advection off, cold start:

  phi = cos(2pi x)                      no walls or walls      7 iterations, |u| after ~ 6e-11
  phi = cos(2pi x) cos(2pi z)           walls                 39 iterations
  phi = cos(2pi x) cos(2pi y)           PERIODIC y             5 iterations
  phi = cos(2pi x) cos(2pi y)           WALLS                500 (the cap; no progress at rtol 0.1)
  random noise                          walls                 10 iterations
  Reichardt-like channel profile        walls                  5 iterations

So the harness itself is sound -- it projects a pure gradient field to 1e-11 -- and the solver is
plainly fine on real channel states (the production DNS converges in 4-6 iterations and reproduces
the MKM statistics). The stall is specific to a prescribed field that varies in the wall-normal
direction, and the most likely explanation is that the test field is not an admissible state rather
than a solver defect: a curl-free (pure gradient) field cannot satisfy both no-penetration and the
wall conditions simultaneously, so forcing its wall-normal component to zero at the wall face leaves
a component the Neumann projection cannot remove.

NEXT STEP (untried): drop the pure-gradient construction, which is what forces that conflict. Use a
general field that is NOT curl-free -- wall-normal component vanishing at both walls, tangential
components arbitrary -- and check it converges before trusting any refinement sweep. Alternatively,
sidestep prescribed fields entirely: take one converged DNS field, interpolate it to several
resolutions, and solve once at each. That uses only the validated path, at the cost of a
band-limited field on the finer grids.
"""
import argparse

import numpy as np


def velocity(nx, ny, nz, kind):
    """A prescribed field built as the DISCRETE gradient of a potential.

    Taking finite differences of a sampled potential -- rather than sampling an analytic velocity --
    makes the discrete divergence exactly the discrete Laplacian of that potential, so the singular
    all-Neumann pressure problem is compatible to machine precision by construction. Sampling an
    analytic velocity instead leaves a small incompatible component, and the solve then stalls at a
    residual floor and never reaches its tolerance (it looks like divergence, and is not).

    The potential uses cos(m*pi*y), whose y-derivative vanishes at both walls, so the wall-normal
    velocity is zero there as the wall boundary condition requires. Wavelengths are fixed fractions
    of the box, so every resolution samples the SAME continuous problem -- the point of the test.
    """
    tp = 2.0 * np.pi
    x = (np.arange(nx) + 0.5)[:, None, None] / nx
    y = (np.arange(ny) + 0.5)[None, :, None] / ny
    z = (np.arange(nz) + 0.5)[None, None, :] / nz
    if kind == "smooth":
        modes = [(1, 1, 1)]
    elif kind == "broadband":
        # fixed physical wavelengths down to 1/16 of the box, amplitude ~1/k: the rougher spectrum a
        # turbulent divergence field carries, still one resolution-independent continuous function.
        modes = [(m, m, m) for m in (1, 2, 4, 8, 16)]
    else:
        raise SystemExit(f"unknown --rhs {kind!r} (smooth|broadband)")
    phi = np.zeros((nx, ny, nz))
    for kx, my, kz in modes:
        phi += (1.0 / kx) * np.cos(tp * kx * x) * np.cos(my * np.pi * y) * np.cos(tp * kz * z)

    u = np.roll(phi, -1, axis=0) - phi                     # periodic x
    w = np.roll(phi, -1, axis=2) - phi                     # periodic z
    v = np.zeros_like(phi)
    v[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]          # interior y faces
    v[:, -1, :] = 0.0                                      # +y wall face; -y wall is a ghost, also 0
    return (np.asfortranarray(u), np.asfortranarray(v), np.asfortranarray(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="32,48,64", help="wall-normal cell counts (the box is refined)")
    ap.add_argument("--aspect", default="6,1,2", help="box shape as x,y,z multiples of the y extent")
    ap.add_argument("--rhs", default="smooth", help="smooth | broadband")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--bottom", default="auto", help="auto | smoother | agglomerated")
    ap.add_argument("--rtol", type=float, default=1e-10, help="tight, so counts resolve clearly")
    ap.add_argument("--periodic-y", action="store_true", help="drop the walls (contrast case)")
    args = ap.parse_args()
    ax, ay, az = (float(v) for v in args.aspect.split(","))

    from peclet import flow

    print(f"box {ax:g}:{ay:g}:{az:g}   rhs={args.rhs}   bottom={args.bottom}   "
          f"levels<={args.levels}   rtol={args.rtol:g}   "
          f"{'PERIODIC y (no walls)' if args.periodic_y else 'walls on -y/+y'}")
    print(f"{'grid':>18} {'Mcells':>8} {'h (rel)':>8} {'pressure iterations':>20}")
    first = None
    for n in (int(v) for v in args.n.split(",")):
        nx, ny, nz = int(round(ax / ay * n)), n, int(round(az / ay * n))
        s = flow.Solver(nx, ny, nz)
        s.set_rho(1.0)
        s.set_mu(0.0)          # predictor becomes (rho/dt) I  =>  u* = u exactly
        s.set_dt(1.0)
        s.set_advection(False)
        s.set_incremental_pressure(False)
        s.set_pressure_warmstart(False)          # cold start: the count is not an initial-guess artefact
        s.set_pressure_multigrid(True, args.levels)
        s.set_pressure_pcg(True, 500, args.rtol)
        s.set_pressure_bottom(args.bottom)
        if not args.periodic_y:
            s.set_domain_bc(2, 1)
            s.set_domain_bc(3, 1)
        s.set_pressure_geometry(np.asfortranarray(np.full((nx, ny, nz), 1e30)))
        s.set_state(*velocity(nx, ny, nz, args.rhs))
        s.step()
        it = s.last_pressure_iterations()
        first = first or n
        print(f"{f'{nx}x{ny}x{nz}':>18} {nx * ny * nz / 1e6:8.2f} {first / n:8.3f} {it:20.1f}")
        del s
    print("\nA resolution-independent multigrid holds the count flat as h shrinks.")


if __name__ == "__main__":
    main()
