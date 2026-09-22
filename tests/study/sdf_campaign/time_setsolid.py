"""Time AmrFlow set_solid on the RCP graded bed — the Layer-2-for-core acceptance driver.
Reuses the study script's own mesh builder so the mesh is exactly the campaign's."""
import importlib.util, sys, time, os
spec = importlib.util.spec_from_file_location(
    "bed", "/home/frankp/Codes/suite/core/tests/study/amr_bed_graded.py")
bed = importlib.util.module_from_spec(spec)
sys.modules["bed"] = bed
spec.loader.exec_module(bed)
import numpy as np
amr = bed.amr

depth = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = 2 ** depth
t0 = time.time()
gapfn, _ = bed.make_gap_lookup(N)
tree = bed.build("g", N, gapfn)
n = tree.num_leaves
print(f"depth={depth}  N={N}  leaves={n}  (mesh build {time.time()-t0:.1f}s)")
fl = amr.Flow(tree, 1.0, bed.MU, 60.0)
fl.set_body_force(bed.FX, 0.0, 0.0)
fl.set_advection(False)
fl.set_ghost_sampled(True)
fl.set_cf_scheme(1)
C, R = bed.load_pack(N)
t0 = time.time()
fl.set_solid_spheres(C, np.array([R]), True)
dt = time.time() - t0
print(f"set_solid: {dt:.2f} s  = {1e6*dt/n:.1f} us/leaf")
# classification fingerprint for the parity gate (before vs after must be bit-identical)
m = np.asarray(fl.is_fluid())
print(f"fluid mask: {int(m.sum())}/{n} fluid  hash={hash(m.tobytes())}")
np.save(sys.argv[2], m) if len(sys.argv) > 2 else None
