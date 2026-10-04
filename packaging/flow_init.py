"""peclet.flow — the Eulerian incompressible Navier–Stokes solver.

A Kokkos cut-cell Immersed-Boundary-Method solver on a staggered MAC grid (grid-agnostic by design:
Cartesian cut-cell today, able to consume an unstructured Voronoi grid from :mod:`peclet.voro`). The
compiled backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.flow.execution_space``
reports which one this build has.

* :class:`peclet.flow.Solver` — the staggered MAC solver.
* :class:`peclet.flow.SolverColocated` — the collocated/cell-centered variant.
* :func:`peclet.flow.march_to_steady` — march a solver to its certified steady state, optionally
  Anderson-accelerated (returns a :class:`peclet.flow.MarchResult`).

Pore-network extraction lives in the companion :mod:`peclet.pnm` package (peclet-pnm; it was
``peclet.flow.pnm`` before 2026-07).

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._flow import *  # noqa: F401,F403  (Solver, SolverColocated, execution_space, ...)
from .steady import MarchResult, march_to_steady  # noqa: F401  (steady marches)

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown". This replaces a
# hand-maintained literal that had drifted behind pyproject.toml in every package at 0.6.0.
try:
    from importlib.metadata import PackageNotFoundError as _PNF, version as _dist_version
    try:
        __version__ = _dist_version("peclet-flow")
    except _PNF:  # the CUDA wheel installs the same module under the -cu13 distribution name
        __version__ = _dist_version("peclet-flow-cu13")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
