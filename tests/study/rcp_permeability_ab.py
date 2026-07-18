"""RCP permeability A/B: {staggered, collocated} x {cutcell, ghost} on the SAME random close
packing (the peclet-examples random-packed-bed geometry: dem pack_bed, N=180, phi~0.63, seed 3;
cached in rcp_pack_seed3.npz — regenerate with the example's pack_bed if absent).

Context: the staggered examples-check found the ghost projection inflates the RCP permeability
(+21/+15/+12% at Ng=32/44/56 vs the cutcell reference) because point-based closure faces cannot
throttle sub-cell throats the way apertures do, and the binary COUPLED graph fragments (guard
decouples the pockets). This study measures whether the collocated ghost inherits the same
defect, with the collocated cutcell (mode 0) and both staggered variants as anchors. Protocol =
the example's steady loop with a tighter stop (tol 1e-6) so the scheme, not the stop criterion,
is compared."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.environ.get("SDFLOW_BUILD", "build_cuda2")))
import numpy as np
from peclet import flow

HERE = os.path.dirname(os.path.abspath(__file__))


def sdf_from_pack(Ng, pos, r, side):
    g = (np.arange(Ng) + 0.5) / Ng * side
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    best = np.full((Ng, Ng, Ng), 1e30)
    for k in range(len(pos)):
        dx = X - (pos[k, 0] + side / 2)
        dx -= side * np.round(dx / side)
        dy = Y - (pos[k, 1] + side / 2)
        dy -= side * np.round(dy / side)
        dz = Z - (pos[k, 2] + side / 2)
        dz -= side * np.round(dz / side)
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz) - r[k])
    return best


def permeability(Ng, sdf, side, colloc, ghost, mode=0, mu=0.1, F=1e-3, dt=80.0, max_steps=3000,
                 tol=1e-6):
    lv = max(2, int(np.log2(Ng)) - 1)
    s = (flow.SolverColocated if colloc else flow.Solver)(Ng, Ng, Ng)
    s.set_rho(1.0)
    s.set_mu(mu)
    s.set_dt(dt)
    s.set_body_force(F, 0, 0)
    s.set_advection(False)
    s.set_velocity_solver_params(150)
    s.set_pressure_multigrid(True, levels=lv)
    s.set_pressure_pcg(True, 400, 1e-9)
    if ghost:
        s.set_ghost_projection(True, 1, 2)
    if mode:
        s.set_face_interp(mode)
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=True,
                pressure_coarse="rediscretized")
    prev = 0.0
    for it in range(max_steps):
        s.step()
        if it % 5 == 4:
            m = float(s.get_u().mean())
            if it > 10 and abs(m - prev) < tol * (abs(m) + 1e-30):
                break
            prev = m
    umean = float(s.get_u().mean())
    return dict(k=mu * umean / F * (side / Ng) ** 2, steps=it + 1,
                pit=s.last_pressure_iterations(), div=s.max_open_divergence())


if __name__ == "__main__":
    d = np.load(os.path.join(HERE, "rcp_pack_seed3.npz"))
    pos, r, side, phi = d["pos"], d["r"], float(d["side"]), float(d["phi"])
    print(f"RCP: N={len(pos)} phi={phi:.4f} side={side:.3f}", flush=True)
    # mode 10 (open-centroid quadrature) is deliberately absent: it DIVERGES on the RCP slivers
    # (k -> 1e18, non-finite MG) — the mode-3a non-telescoping row-sum runaway; measured 2026-07-18.
    variants = [("stag cutcell", False, False, 0), ("stag ghost", False, True, 0),
                ("col cutcell", True, False, 0), ("col ghost", True, True, 0),
                ("col hyb9", True, False, 9)]
    print("columns: " + " | ".join(n for n, *_ in variants)
          + "   (k, % vs stag-cutcell, PCG/BiCGStab iters)", flush=True)
    for Ng in (32, 44, 56):
        sdf = sdf_from_pack(Ng, pos, r, side)
        row = f"{Ng:>4} |"
        kref = None
        for name, colloc, ghost, mode in variants:
            t0 = time.time()
            out = permeability(Ng, sdf, side, colloc, ghost, mode)
            if kref is None:
                kref = out["k"]
            row += (f" {out['k']:.4e} {100 * (out['k'] / kref - 1):+6.1f}%"
                    f" i{out['pit']:>3} |")
        print(row, flush=True)
