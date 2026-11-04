"""G-SV5 / G-SV6 instrument for the sharp viscous interface (doc/vof_sharp_viscous_stress.md §11,
WO-5): a fixed number of production steps, the final fields written to one .npz, one summary line.

  python gsv56.py --case p1 --dh 16 --steps 20 --mode sharp --out p1_sharp_np1.npz
  mpirun --bind-to none -np 4 python gsv56.py ...      (distributed; rank 0 writes the gathered fields)

Cases:
  p1    the P1 bubble (Loisy E1: Ar 29.9, Bo 2, rho ratio 1e-3, mu ratio 1e-2, triply periodic box
        L = 4d, from rest, C = 1 is the gas) as validation/highratio/pec/run_hr_scheme.py sets it up
        (VoF momentum, upwind scheme, CSF, PCG 1e-8), dt = 0.25 min(cfl_dt, capillary_dt) per step.
  wall  the G-SV1 geometry class: unit box, no-slip walls on all six faces, a planar interface with
        normal n(22.5, 22.5 deg) through the centre offset 0.3 h along n, mu ratio 1e-2 (C = 1 the
        low phase), uniform rho, Stokes (advection off), a body force 0.1 along the tangent t1,
        dt at D_hi = mu_hi dt/(rho h^2) = 100. --n cells per axis (default 24).
--mu-uniform: mu = LinearMix(C) with equal phase viscosities (the closure stays; G-SV5 uniform mu).
--sigma S (wall): surface tension S; any S > 0 sets the planes' interfacial threshold to the CSF
wisp guard (1e-8) instead of 0 (the sigma = 0 default of VofCurvature::interfaceEps).
Compare two .npz with  python gsv56.py --compare A.npz B.npz  (prints max|diff| per field and
BITWISE / DIFFER).
"""
import argparse
import hashlib
import math
import sys

import numpy as np


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("p1", "wall"))
    p.add_argument("--dh", type=int, default=16)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--mode", choices=("sharp", "averaged"), default="averaged")
    p.add_argument("--mu-uniform", action="store_true")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="wall case: surface tension (a tiny value puts the planes on the CSF wisp threshold)")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2)
    return p.parse_args()


def compare(a, b):
    A, B = np.load(a), np.load(b)
    worst, same = {}, True
    for k in ("C", "u", "v", "w", "p"):
        d = float(np.max(np.abs(A[k] - B[k])))
        worst[k] = d
        same = same and np.array_equal(A[k], B[k])
    print(("BITWISE " if same else "DIFFER ") + " ".join(f"{k}={v:.3e}" for k, v in worst.items()))
    return 0 if same else 1


def sphere_fractions(n, c, r, sub=8):
    off = (np.arange(sub) + 0.5) / sub
    g = (np.arange(n)[:, None] + off[None, :]).ravel()
    out = np.zeros((n, n, n))
    lo, hi = int(math.floor(c - r)) - 1, int(math.ceil(c + r)) + 1
    for k in range(max(lo, 0), min(hi, n)):
        zs = k + off
        X, Y, Z = np.meshgrid(g, g, zs, indexing="ij")
        ins = ((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2 < r * r).astype(np.float64)
        out[:, :, k] = ins.reshape(n, sub, n, sub, sub).mean(axis=(1, 3, 4))
    return np.asfortranarray(out)


def plane_fractions(n, nrm, x0, sub=8):
    """Cell fractions of the half-space nrm.(x - x0) < 0, x in cells, sub^3 samples."""
    off = (np.arange(sub) + 0.5) / sub
    g = (np.arange(n)[:, None] + off[None, :]).ravel()
    out = np.zeros((n, n, n))
    for k in range(n):
        X, Y, Z = np.meshgrid(g, g, k + off, indexing="ij")
        ins = (nrm[0] * (X - x0[0]) + nrm[1] * (Y - x0[1]) + nrm[2] * (Z - x0[2]) < 0).astype(np.float64)
        out[:, :, k] = ins.reshape(n, sub, n, sub, sub).mean(axis=(1, 3, 4))
    return np.asfortranarray(out)


def main():
    a = parse()
    if a.compare:
        sys.exit(compare(*a.compare))
    import peclet.flow as pf
    comm = None
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            comm = MPI.COMM_WORLD
    except ImportError:
        pass
    n = round(4 * a.dh) if a.case == "p1" else a.n
    if comm is None:
        org, siz = (0, 0, 0), (n, n, n)
    else:
        org, siz = pf.mpi_block(n, n, n)
    sl = tuple(slice(o, o + m) for o, m in zip(org, siz))

    if a.case == "p1":
        AR, BO, RR, MR = 29.9, 2.0, 1e-3, 1e-2
        D, RHO_L, G = 1.0e-3, 1000.0, 9.81
        RHO_G = RR * RHO_L
        MU_L = math.sqrt(RHO_L * (RHO_L - RHO_G) * G * D ** 3) / AR
        MU_G = MU_L if a.mu_uniform else MR * MU_L
        SIGMA = (RHO_L - RHO_G) * G * D ** 2 / BO
        L = 4 * D
        h = L / n
        s = pf.Solver(siz, extent=(L, L, L)) if comm is None else \
            pf.Solver(siz, extent=(L, L, L), global_cells=(n, n, n))
        if comm is not None:
            s.init_mpi(n, n, n)
        s.set_rho(RHO_L)
        s.set_mu(MU_L)
        s.set_pressure_geometry(np.full(siz, 1.0, order="F"))
        s.enable_vof()
        s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
        s.set_property_model("mu", "linear", "C", [MU_L, MU_G - MU_L])
        s.set_advection(True)
        s.set_surface_tension(SIGMA)
        s.set_pressure_pcg(True, 800, 1e-8)
        Cg = sphere_fractions(n, 0.5 * L / h, 0.5 * D / h)
        s.set_vof(np.asfortranarray(Cg[sl]))
        s.enable_vof_momentum(RHO_L, RHO_G)
        alpha = float(np.mean(Cg))
        rho_av = RHO_L + (RHO_G - RHO_L) * alpha
        s.set_property_model("force_x", "linear", "C", [-(RHO_L - rho_av) * G, -(RHO_G - RHO_L) * G])
        s.set_superficial_velocity(True, "x", 0.0)
    else:
        L, h = 1.0, 1.0 / n
        MU_HI, MU_LO = 1.0, (1.0 if a.mu_uniform else 1e-2)
        s = pf.Solver(siz, extent=(L, L, L)) if comm is None else \
            pf.Solver(siz, extent=(L, L, L), global_cells=(n, n, n))
        if comm is not None:
            s.init_mpi(n, n, n)
        for f in ("-x", "+x", "-y", "+y", "-z", "+z"):
            s.set_domain_bc(f, "wall")
        s.set_rho(1.0)
        s.set_mu(MU_HI)
        s.set_pressure_geometry(np.full(siz, 1.0, order="F"))
        s.enable_vof()
        b, g = math.radians(22.5), math.radians(22.5)
        nrm = (math.cos(b) * math.cos(g), math.cos(b) * math.sin(g), math.sin(b))
        x0 = [0.5 * n + 0.3 * c for c in nrm]
        s.set_vof(np.asfortranarray(plane_fractions(n, nrm, x0)[sl]))
        s.set_property_model("mu", "linear", "C", [MU_HI, MU_LO - MU_HI])
        s.set_advection(False)
        s.set_body_force((-0.1 * math.sin(g), 0.1 * math.cos(g), 0.0))
        s.set_pressure_pcg(True, 800, 1e-10)
        if a.sigma > 0:
            s.set_surface_tension(a.sigma)
    if a.mode == "sharp" or hasattr(s, "set_vof_viscous_interface"):  # absent on a pre-sharp build
        s.set_vof_viscous_interface(a.mode)  # explicit both ways: the default 'auto' is sharp here
    cen = {"band_cells": 0, "virtual_cells": 0, "film_cells": 0, "fit_fallbacks": 0}
    for _ in range(a.steps):
        if a.case == "p1":
            lm = s.vof_step_limits()
            s.set_dt(0.25 * min(lm["cfl_dt"], lm["capillary_dt"]))
        else:
            s.set_dt(100.0 * h * h / MU_HI)
        s.step()
        if hasattr(s.diagnostics, "vof_viscous_interface_census"):  # absent on a pre-sharp build
            c = s.diagnostics.vof_viscous_interface_census()
            for k in cen:
                cen[k] += c[k]
    if comm is not None:
        for k in cen:
            cen[k] = comm.allreduce(cen[k])
    loc = {"C": s.get_vof(), "u": s.get_u(), "v": s.get_v(), "w": s.get_w(), "p": s.get_field("p")}
    if comm is not None:
        parts = comm.gather((sl, loc), root=0)
        if comm.Get_rank() != 0:
            return
        glob = {k: np.zeros((n, n, n), order="F") for k in loc}
        for psl, pl in parts:
            for k in loc:
                glob[k][psl] = pl[k]
        loc = glob
    np.savez(a.out, **loc)
    hs = hashlib.sha256(b"".join(np.ascontiguousarray(loc[k]).tobytes() for k in sorted(loc))).hexdigest()[:16]
    npr = 1 if comm is None else comm.Get_size()
    print(f"GSV56 case={a.case} mode={a.mode} mu_uniform={a.mu_uniform} np={npr} steps={a.steps} "
          f"census/step band={cen['band_cells'] / a.steps:.1f} virt={cen['virtual_cells'] / a.steps:.1f} "
          f"film={cen['film_cells'] / a.steps:.1f} fallbacks={cen['fit_fallbacks']} "
          f"max|u|={float(np.max(np.abs(loc['u']))):.6e} sha={hs}")


if __name__ == "__main__":
    main()
