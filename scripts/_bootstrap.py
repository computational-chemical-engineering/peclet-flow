"""Resolve `peclet.flow` for the scripts in this directory (and tests/study, tests/regression, one
level down) -- one place instead of a copy-pasted sys.path.insert at the top of every script.

Precedence, and why:

  1. If `import peclet.flow` already succeeds with `sys.path` exactly as the interpreter set it up
     (PYTHONPATH, an editable install, a wheel, ...) -- use it, and insert nothing. Whatever the
     caller already arranged is what gets tested; a script must never second-guess it.
  2. Otherwise, if PECLET_FLOW_BUILD is set (or an explicit build dir is passed in, e.g. from a
     script's own --build flag), insert that directory at the FRONT of sys.path and import from
     there.
  3. Otherwise, fail loudly. There is no default build directory.

A silent fallback to some `build*/` guess is exactly the defect this module exists to close: every
`flow/scripts/*.py` used to `sys.path.insert(0, ...SDFLOW_BUILD default build_mpi...)` UNCONDITIONALLY,
ahead of anything PYTHONPATH set up. `PYTHONPATH=$PWD/build python scripts/verify_poiseuille_flow.py`
therefore silently tested `build_mpi` instead -- ten days and ~50 commits stale on 2026-09-11 -- and
kept printing PASS the whole time (docs/RELEASE_PREP.md #1.3). An explicit failure that names both
ways to fix it costs one line; a wrong PASS costs a release. `SDFLOW_BUILD` itself is retired (the
suite-wide sdflow -> flow rename, QUALITY_PLAN D1: no compatibility aliases) in favour of
PECLET_FLOW_BUILD.

Every caller prints, to stderr, exactly which `peclet.flow` it ended up with -- so the reader of a
PASS/FAIL knows which binary produced it.
"""
import os
import sys


def ensure_flow(explicit_build=None):
    """Import and return the `peclet.flow` module, resolved per the module docstring's precedence.

    `explicit_build`, if given, is a directory to try (e.g. from a script's own --build CLI flag)
    with the same standing as PECLET_FLOW_BUILD: consulted only when `peclet.flow` is not already
    importable, and only if PECLET_FLOW_BUILD itself is unset.

    Always prints one line to stderr naming the resolved module file or inserted directory. Raises
    SystemExit with a corrective message if `peclet.flow` cannot be found at all.
    """
    try:
        import peclet.flow as flow
    except ImportError:
        build_dir = os.environ.get("PECLET_FLOW_BUILD") or explicit_build
        if not build_dir:
            raise SystemExit(
                "peclet.flow not importable: set PYTHONPATH=<build tree> "
                "(e.g. PYTHONPATH=$PWD/build) or PECLET_FLOW_BUILD=<build tree> "
                "(e.g. PECLET_FLOW_BUILD=$PWD/build_mpi) and rerun."
            )
        build_dir = os.path.abspath(build_dir)
        sys.path.insert(0, build_dir)
        try:
            import peclet.flow as flow
        except ImportError as exc:
            raise SystemExit(
                f"peclet.flow not importable from build dir {build_dir!r}: {exc}\n"
                "Fix: point PECLET_FLOW_BUILD (or PYTHONPATH) at a directory containing the "
                "built peclet/flow/_flow*.so."
            ) from exc
        print(f"peclet.flow: using build dir {build_dir} -> {flow.__file__}", file=sys.stderr)
        return flow
    print(f"peclet.flow: already importable -> {flow.__file__}", file=sys.stderr)
    return flow
