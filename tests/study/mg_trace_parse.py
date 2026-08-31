#!/usr/bin/env python3
"""Summarise a PECLET_FLOW_MG_DEBUG=2 residual trace: per solve, the iteration count, the minimum
r/r0 reached (the ATTAINABLE FLOOR when the run is driven with an unreachable rtol), the final
r/r0, and the rebound factor final/min.

The distinction this exists for: an iteration count cannot tell a FLOOR (min == final, both far
above rtol — round-off has taken over) from a REBOUND (final > min — the CG recurrence has lost
conjugacy and is walking back up) from honest slow convergence. WO-M's whole float-vs-double
comparison turns on it, and so does reading the classical attainable-accuracy bound
||r||/||r0|| ~ O(eps * kappa(A)) off a measured plateau.

Usage: <run emitting [mg] lines> | python mg_trace_parse.py [label]
"""
import re
import sys


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    solves, cur = [], None
    for line in sys.stdin:
        if line.startswith("[mg] solve"):
            cur = []
            solves.append(cur)
        elif "r/r0" in line and cur is not None:
            cur.append(float(re.search(r"r/r0=([0-9.eE+-]+)", line).group(1)))
    print(f"--- {label}")
    for i, v in enumerate(solves):
        if not v:
            continue
        mn = min(v)
        k = v.index(mn)
        print(f"  solve {i}: {len(v):4d} its   floor(min r/r0) {mn:.3e} at it {k + 1:4d}   "
              f"final {v[-1]:.3e}   rebound x{v[-1] / mn:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
