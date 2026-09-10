#!/usr/bin/env python3
"""QUALITY_PLAN G.6 guard: no operator-storage view narrows to `float` outside the allow-list.

`MReal` (`mac_cutcell_mg.hpp`) is the operator storage type -- float unless the build defines
`-DPECLET_FLOW_OPERATOR_DOUBLE` (an `option()` since G.6, was a raw, untested `-DCMAKE_CXX_FLAGS`).
G.6 templated `IbmOverlayT`/`ibmFillEntry`/`ibmModifyStencil` (cut_cell_ibm.hpp) and the geometry
build that feeds them (mac_ibm.hpp, flow_ibm.hpp) on `Real`/`mreal` so a double build carries the
cut-cell overlay in double end to end; this test fails if a NEW hard `(float)` cast or `float`-typed
operator view creeps back in outside the allow-list below, so the next one has to be typed on
`mreal`/`MReal` (or exempted with a stated reason) instead.

A hit is IGNORED when: the line is a pure comment (only prose, no code -- these mention "(float)" in
plain English and are not casts); the whole FILE is allow-listed (an opt-in debug/forensics file
whose float precision is unrelated to operator correctness, or a file whose ghost-projection overlay
is a stated, deferred gap -- see the reasons below); or the line carries a `PRECISION-EXEMPT:`
marker (a deliberate, reasoned exception -- `git grep PRECISION-EXEMPT src/` to read them).
"""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

CAST = re.compile(r"\(\s*float\s*\)|static_cast\s*<\s*float\s*>|View<\s*float\s*\*")

# Whole files exempt from this test, with the reason (also stated in the file's own header
# comment as PRECISION-EXEMPT so a reader of the file sees it too, not just this list).
FILE_EXEMPT = {
    "ghost_projection_debug.hpp": (
        "opt-in forensics (PECLET_FLOW_GP_DEBUG); never feeds a solve, precision is unrelated to "
        "correctness"
    ),
    "ghost_projection.hpp": (
        "GpOverlay + its SDF/theta sampling stay float -- deferred: the same Real-templating "
        "IbmOverlayT (cut_cell_ibm.hpp) received in G.6, not reached this pass; a real exposure "
        "(the AUTO default collocated scheme), not dead code -- see the file's own header comment"
    ),
}


def is_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or s.startswith("/*") or s.startswith("*")


def main() -> int:
    offenders = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in (".hpp", ".cpp", ".h"):
            continue
        if path.name in FILE_EXEMPT:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not CAST.search(line):
                continue
            if is_comment_line(line):
                continue
            if "PRECISION-EXEMPT" in line:
                continue
            offenders.append(f"{path.relative_to(SRC.parent)}:{lineno}: {line.strip()}")
    if offenders:
        print(
            "hard (float) casts / float-typed operator views found in src/ outside the "
            "allow-list (QUALITY_PLAN G.6 -- type on mreal/MReal, or mark "
            "`// PRECISION-EXEMPT: <reason>` if it is genuinely precision-independent):"
        )
        for o in offenders:
            print("  " + o)
        return 1
    print("no un-exempted float operator casts in src/  [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
