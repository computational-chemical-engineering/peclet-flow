"""peclet.flow — the Eulerian incompressible Navier–Stokes solver.

A Kokkos cut-cell Immersed-Boundary-Method solver on a staggered MAC grid (grid-agnostic by design:
Cartesian cut-cell today, able to consume an unstructured Voronoi grid from :mod:`peclet.voro`). The
compiled backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.flow.execution_space``
reports which one this build has.

* :class:`peclet.flow.Solver` — the staggered MAC solver.
* :class:`peclet.flow.SolverColocated` — the collocated/cell-centered variant.
* :mod:`peclet.flow.pnm` — pore-network extraction from SDF pore geometry.

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._flow import *  # noqa: F401,F403  (Solver, SolverColocated, execution_space, ...)

__version__ = "0.1.0"
