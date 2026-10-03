#!/usr/bin/env python
"""WO-9b instrument: wall time of advance_scalars on G-perf's 128^3 bed (dt D/h^2 = 1, koren,
Dirichlet spheres) after 3 flow steps; NS timed advances (env N = cells per axis, NS). The CUDA
profiler range wraps the timed advances (nsys --capture-range=cudaProfilerApi). On host, run with
KOKKOS_TOOLS_LIBS = the Kokkos simple kernel timer at NS=0 and NS=k and difference the two tables:

  NS=3 OMP_NUM_THREADS=1 KOKKOS_TOOLS_LIBS=<kp_kernel_timer.so> PYTHONPATH=build_dev \
      python advance_profile.py > kt3.txt          (and NS=0 > kt0.txt)
  python advance_profile.py diff kt0.txt kt3.txt 3
"""
import sys
if len(sys.argv) > 1 and sys.argv[1] == "diff":
    import re
    def parse(p):
        d = {}; name = None; on = False
        for line in open(p):
            if line.startswith("Kernels:"): on = True; continue
            if not on: continue
            if line.startswith("- "): name = line[2:].strip(); continue
            m = re.match(r"\s*\((\w+)\)\s+([\d.]+)\s+(\d+)", line)
            if m and name: d[name] = (float(m.group(2)), int(m.group(3))); name = None
            if line.startswith("Summary"): break
        return d
    A, B = parse(sys.argv[2]), parse(sys.argv[3]); n = float(sys.argv[4])
    rows = []
    for k in B:
        t = B[k][0] - A.get(k, (0, 0))[0]; c = B[k][1] - A.get(k, (0, 0))[1]
        rows.append((t / n * 1e3, c / n, k))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"per advance: {tot:.1f} ms in kernels, {sum(r[1] for r in rows):.0f} launches")
    for t, c, k in rows[:int(sys.argv[5]) if len(sys.argv) > 5 else 25]:
        print(f"  {t:8.2f} ms {c:8.0f}  {k}")
    sys.exit(0)
import os, sys, time, math
sys.path.insert(0, "/home/frankp/Codes/suite/flow-scalar-ibm/tests/study/scalar_ibm")
sys.path.insert(0, "/home/frankp/Codes/suite/flow-scalar-ibm/tests/python")
import numpy as np
import peclet.flow as pf
import test_scalar_cutcell_gates as g
n = int(os.environ.get("N", "128"))
s = pf.Solver((n, n, n), extent=(1.0, 1.0, 1.0))
s.set_rho(1.0); s.set_mu(1.0); dt = 0.01; s.set_dt(dt)
R = (0.3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
X, Y, Z = g.grid(s)
s.set_solid(np.asfortranarray(np.sqrt((X - 0.513) ** 2 + (Y - 0.479) ** 2 + (Z - 0.507) ** 2) - R), cutcell_pressure=True)
s.set_body_force((30.0, 9.0, 0.0))
for _ in range(3): s.step()
h = 1.0 / n
s.add_scalar("c", diffusivity=h * h / dt, scheme="koren", cutcell=True)
s.set_scalar_wall("c", "dirichlet", 0.0)
geo = s.diagnostics.scalar_geometry("c")
s.set_field("c", np.asfortranarray(np.where(geo["unknown"] > 0.5, 1.0, 0.0)))
s.step()
s.advance_scalars(); s.diagnostics.scalar_budget("c")
import ctypes
try:
    cudart = ctypes.CDLL("libcudart.so")
except OSError:
    cudart = None
if cudart: cudart.cudaProfilerStart()
t0 = time.perf_counter()
NS = int(os.environ.get("NS", "3"))
for _ in range(NS):
    s.advance_scalars(); s.diagnostics.scalar_budget("c")
t = (time.perf_counter() - t0) / max(NS, 1)
if cudart: cudart.cudaProfilerStop()
print(f"advance_scalars {t*1e3:.1f} ms/step, its {s.diagnostics.scalar_census('c')['krylov_iterations']}")
