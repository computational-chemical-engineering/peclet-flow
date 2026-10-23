#!/usr/bin/env python3
"""Case A of the bubble-swarm validation: the Esmaeeli-Tryggvason (1999) ordered bubble array.

One bubble in a triply periodic cube is an infinite simple-cubic array. Parameters as rerun by
Innocenti, Jaccod, Popinet & Chibbaro (2021, JFM 918:A23, arXiv 2002.02209, sec. 3 and Table 3)
and Loisy, Naso & Spelt (2017, JFM 816:94, case E1, fig. 1a):

    Ar = sqrt(rho_l |rho_b - rho_l| g d^3) / mu_l = 29.9,   Bo = |rho_b - rho_l| g d^2 / sigma = 2,
    rho_b/rho_l = mu_b/mu_l = 0.1,   L = 1.6 d (Table 3: 16^3 cells at d/h = 10),  phi = 12.8 %.

Re = rho_l U d / mu_l with U the DRIFT velocity  U = <u>_d - <u>  (Loisy 2017 eq. 2.3: the volume
average over the gas minus the volume average over the whole cell). Targets (Table 3): Basilisk
23.05/22.01/21.275/21.0 at d/h 10/20/40/80, Loisy 19.05/20.22/20.58 at 10/20/40, E&T 20.49 at 40.

Set-up: the bubble-column driver's (peclet-examples benchmarks/bubble-column/scripts/run_peclet.py)
multiple-marker VoF with its solver settings, in a PHYSICAL domain (metres, SI properties).
Gravity points along -x, so the bubble rises along +x:
  * buoyancy with the mixture weight removed, f_x = -(rho(C) - <rho>) g, <rho> from the discrete
    gas volume (the -<rho>g body force of Innocenti sec. 3 / Loisy eq. 3.1);
  * set_superficial_velocity(True, 'x', 0): the volume mean <u_x> is held at 0, so the computational
    frame is Loisy's zero-mixture-velocity frame and the drift velocity is the gas-averaged u_x.
    The lateral means <u_y>, <u_z> are free (zero by symmetry) and are written out as a check.

Usage:
    PYTHONPATH=<flow build> OMP_NUM_THREADS=8 OMP_PROC_BIND=false \\
        python tests/study/case_a/run_case_a.py --d-over-h 10 --backend host --t-end 20 --out runA10

  --d-over-h {10,20,40,80}   cells per bubble diameter; box = (1.6 d/h)^3 cells
  --backend {host,cuda}      checked against the imported module's Kokkos execution space (the
                             backend is chosen by PYTHONPATH, i.e. by which build is imported)
  --t-end T                  end time in units of sqrt(d/g) (default 20)
  --out DIR                  writes DIR/case_a.csv (one row per sample) and DIR/summary.txt;
                             a_closure_g = the superficial-velocity closure's acceleration
                             (last_superficial_velocity_shift / dt) in units of g
  --dt-out T                 sample interval in sqrt(d/g) (default 0.05)
  --max-steps N, --wall S    stop early (timing / smoke runs)
  --dt-safety F              fraction of the CFL and capillary limits (default 0.25, the driver's)
  --mu-ratio R               ablation: mu_b/mu_l (default 0.1, the case); Ar and Bo are unchanged
                             (both use mu_l and the density difference)
  --face-mu {arithmetic,harmonic}  ablation: face-viscosity mean (default arithmetic, the driver's)
  --viscous-stress {laplacian,full}  the viscous stress form (set_viscous_stress; default laplacian,
                             the solver's default; doc/full_viscous_stress.md)
  --single-vof               ablation: the structured VoF + CSF without the block container (the
                             bubble seeded as 8x8x8 sub-sampled sphere fractions)
  --vof-momentum             ablation: enable_vof_momentum(rho_g, rho_l) (implies --single-vof:
                             the momentum-consistent transport is refused with the blocks)
  --mpi                      distributed run under mpirun (mpi4py; needs a PECLET_FLOW_MPI build):
                             mpi_block + init_mpi before the geometry; the fields are gathered for
                             the samples, rank 0 writes. At one rank the single-rank solver runs (no
                             mpi_block, no init_mpi), so `--mpi` at np 1 equals a run without it.
  --dump FILE.npz            write the final global u, v, w, p, C (decomposition-invariance checks,
                             doc/full_viscous_stress.md G4)

Prints one summary line.
"""
import argparse
import math
import os
import resource
import sys
import time

import numpy as np
import peclet.flow as pf

# --- physics: dimensionless groups (verified against the sources, see docstring) -------------
AR, BO, RHO_RATIO, MU_RATIO, L_OVER_D = 29.9, 2.0, 0.1, 0.1, 1.6
# --- dimensional realisation (SI): a 1 mm bubble; mu_l and sigma follow from Ar and Bo ---------
D = 1.0e-3                        # m, volume-equivalent diameter
RHO_L = 1000.0                    # kg/m^3
RHO_G = RHO_RATIO * RHO_L
G = 9.81                          # m/s^2, along -x
MU_L = math.sqrt(RHO_L * abs(RHO_G - RHO_L) * G * D ** 3) / AR    # 3.1426e-3 Pa s
MU_G = MU_RATIO * MU_L
SIGMA = abs(RHO_G - RHO_L) * G * D ** 2 / BO                       # 4.4145e-3 N/m
T_REF = math.sqrt(D / G)          # time unit of the literature plots (Loisy fig. 1a)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--d-over-h", type=int, required=True, choices=(10, 20, 40, 80))
    p.add_argument("--backend", required=True, choices=("host", "cuda"))
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--out", required=True)
    p.add_argument("--dt-out", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--wall", type=float, default=float("inf"))
    p.add_argument("--dt-safety", type=float, default=0.25)
    p.add_argument("--mu-ratio", type=float, default=MU_RATIO)
    p.add_argument("--face-mu", choices=("arithmetic", "harmonic"), default="arithmetic")
    p.add_argument("--viscous-stress", choices=("laplacian", "full"), default="laplacian")
    p.add_argument("--single-vof", action="store_true")
    p.add_argument("--vof-momentum", action="store_true")
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--dump", default=None)
    return p.parse_args()


def check_backend(want):
    space = getattr(pf, "execution_space", "unknown")
    got = "cuda" if "cuda" in str(space).lower() else "host"
    if got != want:
        sys.exit(f"--backend {want} but the imported peclet.flow runs on '{space}' "
                 f"({pf.__file__}); point PYTHONPATH at the matching build")
    return str(space)


def sphere_fractions(n, c, r, sub=8):
    """Cell volume fractions of a sphere (centre c, radius r, both in cells), sub^3 samples."""
    off = (np.arange(sub) + 0.5) / sub
    g = (np.arange(n)[:, None] + off[None, :]).ravel()
    out = np.zeros((n, n, n))
    for k in range(n):  # z slabs keep the memory at n^2 sub^3
        zs = k + off
        X, Y, Z = np.meshgrid(g, g, zs, indexing="ij")
        ins = ((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2 < r * r).astype(np.float64)
        out[:, :, k] = ins.reshape(n, sub, n, sub, sub).mean(axis=(1, 3, 4))
    return np.asfortranarray(out)


class Dist:
    """The rank's block of the global n^3 box; single-rank when comm is None."""

    def __init__(self, n, comm=None):
        self.n, self.comm = n, comm
        if comm is None:
            self.org, self.siz = (0, 0, 0), (n, n, n)
        else:
            self.org, self.siz = pf.mpi_block(n, n, n)
        self.sl = tuple(slice(o, o + m) for o, m in zip(self.org, self.siz))

    def gather(self, a):
        """The global array from every rank's local block (identity single-rank)."""
        if self.comm is None:
            return a
        out = np.zeros((self.n,) * 3, order="F")
        for sl, blk in self.comm.allgather((self.sl, np.asarray(a))):
            out[sl] = blk
        return out

    def any(self, flag):
        return flag if self.comm is None else bool(self.comm.allreduce(bool(flag)))


def build(n, mu_g=MU_G, face_mu="arithmetic", stress="laplacian", single=False, vof_mom=False,
          dist=None):
    L = L_OVER_D * D
    dist = dist or Dist(n)
    if dist.comm is None:
        s = pf.Solver((n, n, n), extent=(L, L, L))
    else:
        s = pf.Solver(dist.siz, extent=(L, L, L), global_cells=(n, n, n))
        s.init_mpi(n, n, n)                   # before the geometry
    s.set_rho(RHO_L)
    s.set_mu(MU_L)
    s.set_pressure_geometry(np.full(dist.siz, 1.0, order="F"))      # all fluid (SDF > 0)
    s.enable_vof()
    s.set_vof(np.zeros(dist.siz, order="F"))
    # C = 1 is GAS (the block container's union starts empty)
    s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
    s.set_property_model("mu", "linear", "C", [MU_L, mu_g - MU_L])
    if face_mu == "harmonic":
        s.diagnostics.set_property_mode("variable", True)
    s.set_viscous_stress(stress)
    s.set_surface_tension(SIGMA)
    s.set_pressure_pcg(True, 800, 1e-8)       # last; the bubble-column driver's setting
    # One marker at the box centre. enable_vof_blocks takes GRID coordinates (cells) even under a
    # physical domain -- derived here from the physical centre and radius, never user input.
    h = L / n
    if single or vof_mom:
        s.set_vof(np.asfortranarray(sphere_fractions(n, 0.5 * L / h, 0.5 * D / h)[dist.sl]))
        if vof_mom:
            s.enable_vof_momentum(RHO_G, RHO_L)
    else:
        s.enable_vof_blocks([(0.5 * L / h, 0.5 * L / h, 0.5 * L / h, 0.5 * D / h)])
        s.enable_vof_block_csf()
    alpha = float(np.mean(dist.gather(s.get_vof())))  # discrete gas fraction (exactly conserved)
    rho_av = RHO_L + (RHO_G - RHO_L) * alpha
    s.set_property_model("force_x", "linear", "C",
                         [-(RHO_L - rho_av) * G, -(RHO_G - RHO_L) * G])
    s.set_superficial_velocity(True, "x", 0.0)
    s.update_properties()                     # so the first capillary_dt sees rho(C)
    return s, h, alpha


def sample(s, h, x_prev, t_prev, t, dist, with_blocks=True):
    C = dist.gather(s.get_vof())
    u, v, w = dist.gather(s.get_u()), dist.gather(s.get_v()), dist.gather(s.get_w())
    # staggered: u[i] sits at x_{i+1/2}; the cell-centre value averages the two faces (periodic)
    uc = 0.5 * (u + np.roll(u, 1, axis=0))
    rho = RHO_L + (RHO_G - RHO_L) * C
    vg = C.sum()
    ub_abs = float((C * uc).sum() / vg)                       # <u_x>_d, computational frame
    u_mix = float(uc.mean())                                  # <u_x> (held 0)
    u_liq = float(((1.0 - C) * uc).sum() / (1.0 - C).sum())   # <u_x>_l
    u_zm = float((rho * uc).sum() / rho.sum())                # mass-mean (zero-momentum frame)
    # gas centroid along x, unwrapped: minimal image about the previous (continuous) position
    n = C.shape[0]
    L = n * h
    dxi = (np.arange(n) + 0.5) * h - x_prev
    dxi -= L * np.round(dxi / L)
    x = x_prev + float((C.sum(axis=(1, 2)) * dxi).sum() / vg)
    u_cent = (x - x_prev) / (t - t_prev) if t > t_prev else float("nan")
    blocks = s.diagnostics.vof_block_stats() if with_blocks else []  # none under --single-vof
    if dist.comm is not None:  # each block's stats live on its master rank
        blocks = [b for lst in dist.comm.allgather(blocks) for b in lst]
    return {
        "u_abs": ub_abs, "u_mix": u_mix, "u_liq": u_liq, "u_zm": u_zm,
        "u_drift": ub_abs - u_mix, "u_slip": ub_abs - u_liq, "u_rel_zm": ub_abs - u_zm,
        "u_cent": u_cent, "v_mean": float(v.mean()), "w_mean": float(w.mean()),
        "gas_vol": float(vg) * h ** 3,
        "block_vol": float(blocks[0]["volume"]) if blocks else float("nan"), "x_cent": x}


def main():
    a = parse()
    space = check_backend(a.backend)
    n = round(L_OVER_D * a.d_over_h)
    comm = None
    if a.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
    dist = Dist(n, comm)
    root = comm is None or comm.Get_rank() == 0
    nranks = 1 if comm is None else comm.Get_size()
    if root:
        os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    s, h, alpha = build(n, a.mu_ratio * MU_L, a.face_mu, a.viscous_stress, a.single_vof,
                        a.vof_momentum, dist)
    t_setup = time.perf_counter() - t0
    re_of = lambda U: RHO_L * U * D / MU_L
    cols = ["t", "t_star", "step", "dt", "binding", "u_abs", "u_mix", "u_liq", "u_zm", "u_drift",
            "u_slip", "u_rel_zm", "u_cent", "Re_drift", "Re_abs", "Re_slip", "Re_rel_zm",
            "v_mean", "w_mean", "gas_vol", "vol_err", "block_vol", "x_cent", "a_closure_g"]
    f = open(os.path.join(a.out, "case_a.csv") if root else os.devnull, "w")  # noqa: SIM115
    f.write(f"# case A d/h={a.d_over_h} n={n} L={L_OVER_D * D} d={D} rho_l={RHO_L} "
            f"rho_g={RHO_G} mu_l={MU_L:.6e} mu_g={a.mu_ratio * MU_L:.6e} sigma={SIGMA:.6e} g={G} "
            f"alpha={alpha:.6f} space={space} t_ref={T_REF:.6e} dt_safety={a.dt_safety} "
            f"face_mu={a.face_mu} viscous_stress={a.viscous_stress} single_vof={a.single_vof} "
            f"vof_momentum={a.vof_momentum}" + (f" np={nranks}" if comm is not None else "") + "\n")
    f.write(",".join(cols) + "\n")

    t, step, binding, dt = 0.0, 0, "-", 0.0
    with_blocks = not (a.single_vof or a.vof_momentum)
    r0 = sample(s, h, 0.5 * L_OVER_D * D, 0.0, 0.0, dist, with_blocks)
    vol0 = r0["gas_vol"]
    x_prev, t_prev = r0["x_cent"], 0.0
    t_end = a.t_end * T_REF
    next_out = a.dt_out * T_REF
    last = None
    hist = []                                  # (t_star, Re_drift, Re_rel_zm, x_cent, a_closure_g)
    t_loop = time.perf_counter()
    stop = "t_end"
    while t < t_end * (1 - 1e-12):
        L_ = s.vof_step_limits()
        dt = a.dt_safety * min(L_["cfl_dt"], L_["capillary_dt"])
        binding = "cap" if L_["capillary_dt"] < L_["cfl_dt"] else "cfl"
        s.set_dt(dt)
        s.step()
        t += dt
        step += 1
        done = t >= t_end * (1 - 1e-12)
        if a.max_steps is not None and step >= a.max_steps:
            done, stop = True, "max_steps"
        if dist.any(time.perf_counter() - t0 > a.wall):
            done, stop = True, "wall"
        if t >= next_out or done:
            r = sample(s, h, x_prev, t_prev, t, dist, with_blocks)
            x_prev, t_prev = r["x_cent"], t
            r.update(t=t, t_star=t / T_REF, step=step, dt=dt, binding=binding,
                     Re_drift=re_of(r["u_drift"]), Re_abs=re_of(r["u_abs"]),
                     Re_slip=re_of(r["u_slip"]), Re_rel_zm=re_of(r["u_rel_zm"]),
                     vol_err=r["gas_vol"] / vol0 - 1.0,
                     a_closure_g=s.diagnostics.last_superficial_velocity_shift() / dt / G)
            f.write(",".join(str(r[c]) for c in cols) + "\n")
            f.flush()
            hist.append((r["t_star"], r["Re_drift"], r["Re_rel_zm"], r["x_cent"], r["a_closure_g"]))
            last = r
            while next_out <= t:
                next_out += a.dt_out * T_REF
        if done:
            break
    wall_loop = time.perf_counter() - t_loop
    f.close()
    if a.dump:
        flds = {k: dist.gather(g()) for k, g in
                (("u", s.get_u), ("v", s.get_v), ("w", s.get_w), ("p", s.get_p), ("C", s.get_vof))}
        if root:
            np.savez(a.dump, step=step, t=t, **flds)

    # Steady state. Re(t) carries a component periodic in the bubble's position in the box (period
    # L/U, measured at d/h 10), so the averaging windows are whole box crossings: A = the last two
    # crossings (gas displacement 2L), B = the two before. steady <=> |<Re>_A - <Re>_B| < 1e-3 <Re>_A.
    H = np.array(hist)
    Lb = L_OVER_D * D

    def window_mean(col, x_hi, x_lo):
        m = (H[:, 3] > x_lo) & (H[:, 3] <= x_hi)
        if m.sum() < 2:
            return float("nan")
        return float(np.trapezoid(H[m, col], H[m, 0]) / (H[m, 0][-1] - H[m, 0][0]))

    xe = H[-1, 3]
    re_a = window_mean(1, xe, xe - 2 * Lb) if xe - H[0, 3] >= 2 * Lb else float("nan")
    re_b = window_mean(1, xe - 2 * Lb, xe - 4 * Lb) if xe - H[0, 3] >= 4 * Lb else float("nan")
    zm_a = window_mean(2, xe, xe - 2 * Lb) if xe - H[0, 3] >= 2 * Lb else float("nan")
    acc_a = window_mean(4, xe, xe - 2 * Lb) if xe - H[0, 3] >= 2 * Lb else float("nan")
    wa = (H[:, 3] > xe - 2 * Lb)
    re_lo, re_hi = float(H[wa, 1].min()), float(H[wa, 1].max())
    drel = abs(re_a - re_b) / abs(re_a)
    steady = bool(drel < 1e-3)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    line = (f"CASE_A d/h={a.d_over_h} n={n} mu_ratio={a.mu_ratio:g} face_mu={a.face_mu} "
            f"stress={a.viscous_stress} vof={'mom' if a.vof_momentum else ('single' if a.single_vof else 'blocks')} "
            f"dt_safety={a.dt_safety:g} space={space} stop={stop} t*={last['t_star']:.3f} "
            f"steps={step} Re_drift(2L-mean)={re_a:.4f} [min {re_lo:.3f} max {re_hi:.3f}] "
            f"Re_rel_zm(2L-mean)={zm_a:.4f} dRe/Re(2L vs prev 2L)={drel:.2e} steady={steady} "
            f"vol_err={last['vol_err']:.2e} <u>={last['u_mix']:.1e} <v>={last['v_mean']:.1e} "
            f"<w>={last['w_mean']:.1e} u_cent/u_abs={last['u_cent'] / last['u_abs']:.4f} "
            f"setup={t_setup:.1f}s ms/step={1e3 * wall_loop / max(step, 1):.2f} "
            f"maxrss={rss:.0f}MB a_closure/g(2L-mean)={acc_a:.4e}"
            + (f" np={nranks}" if comm is not None else ""))
    if root:
        with open(os.path.join(a.out, "summary.txt"), "w") as g:
            g.write(line + "\n")
        print(line)


if __name__ == "__main__":
    main()
