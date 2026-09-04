"""peclet.flow — the Eulerian incompressible Navier–Stokes solver.

A Kokkos cut-cell Immersed-Boundary-Method solver on a staggered MAC grid (grid-agnostic by design:
Cartesian cut-cell today, able to consume an unstructured Voronoi grid from :mod:`peclet.voro`). The
compiled backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.flow.execution_space``
reports which one this build has.

* :class:`peclet.flow.Solver` — the staggered MAC solver.
* :class:`peclet.flow.SolverColocated` — the collocated/cell-centered variant.

Pore-network extraction lives in the companion :mod:`peclet.pnm` package (peclet-pnm; it was
``peclet.flow.pnm`` before 2026-07).

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._flow import *  # noqa: F401,F403  (Solver, SolverColocated, execution_space, ...)

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown". This replaces a
# hand-maintained literal that had drifted behind pyproject.toml in every package at 0.6.0.
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("peclet-flow")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
