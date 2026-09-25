#!/usr/bin/env python3
"""Iteration-order guard: every multidimensional kernel iterates x fastest, on every backend.

Fields are x-fastest (I = x + y*nx + z*nx*ny; suite/docs/CONVENTIONS.md). Kokkos' default
MDRange iteration is Iterate::Left (first index fastest) on CUDA/HIP but Iterate::Right (LAST index
fastest) on the host backends, so a bare `Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>>` strides
against memory on OpenMP/Serial -- measured 2026-09-25 on the bubble column. The policies are the
aliases `MDRange2` / `MDRange3` of `src/policy.hpp`, which pin Iterate::Left explicitly; this test
fails if `Kokkos::Rank<` or `MDRangePolicy<` appears in `src/` or `tests/` anywhere else.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALIAS_HEADER = ROOT / "src" / "policy.hpp"
BARE = re.compile(r"Kokkos::Rank\s*<|MDRangePolicy\s*<")


def test_iteration_order_follows_storage():
    offenders = []
    for top in ("src", "tests"):
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in (".hpp", ".cpp", ".h") or path == ALIAS_HEADER:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if BARE.search(line.split("//", 1)[0]):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "bare MDRange policies (backend-default iteration order) found -- use MDRange2/MDRange3 "
        "from src/policy.hpp:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    try:
        test_iteration_order_follows_storage()
    except AssertionError as e:
        print(e)
        sys.exit(1)
    print("every MDRange kernel in src/ and tests/ uses the x-fastest aliases  [OK]")
