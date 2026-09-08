#!/usr/bin/env python3
"""QUALITY_PLAN decision D3 guard: no environment variable in `src/` may change a result.

Package E (2026-09-08) turned every numerics-changing / algorithm-selecting `PECLET_FLOW_*` read
into a solver setter or deleted it with the ablation it served. What is still allowed to be read
from the environment is instrumentation only -- `*_DEBUG`, `*_VERBOSE`, `*_PROFILE*`, `*_TIMEOUT`,
`GPU_AWARE_MPI` -- because none of it touches an arithmetic path. This test fails if a new
`getenv` appears in `src/` outside that allow-list, which is the whole point: the next knob has to
be a setter.
"""
import pathlib
import re
import sys

ALLOWED = re.compile(r"_DEBUG(_[A-Z0-9_]+)?$|_VERBOSE$|_PROFILE|_TIMEOUT$|^GPU_AWARE_MPI$")
GETENV = re.compile(r'getenv\(\s*"([A-Za-z0-9_]+)"')

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def test_no_numerics_env_knobs():
    offenders = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in (".hpp", ".cpp", ".h"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for name in GETENV.findall(line):
                if not ALLOWED.search(name):
                    offenders.append(f"{path.relative_to(SRC.parent)}:{lineno}: {name}")
    assert not offenders, (
        "environment variables that can change a result found in src/ (QUALITY_PLAN D3 -- make it "
        "a setter):\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    try:
        test_no_numerics_env_knobs()
    except AssertionError as e:
        print(e)
        sys.exit(1)
    print("no numerics-changing environment reads in src/  [OK]")
