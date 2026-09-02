#!/usr/bin/env python3
"""VoF Part III rung W2 gate 4 (WO-W12) — TBFsolver's `channel_18` on the block container.

TBFsolver (Cifani et al., *Computers & Fluids* 2018; `~/Codes/TBFsolver`) ships one case:
a MINIMAL turbulent channel at Re_tau = 127.3 loaded with 18 deformable bubbles.  This is the
transcription, read off `channel_18/specs/*` and the source that consumes them.

THE CASE (every number verified against the Fortran, not the file names):

    geometry   Lx = pi (streamwise, periodic), Ly = 2 (wall-normal, WALLS), Lz = pi/2 (spanwise,
               periodic); half-height h = 1.  TBFsolver's grid is 192 x 160 x 96 (anisotropic:
               dx+ = 2.08, dy+ = 1.59).  flow's cells are cubic, so the transcription is an
               ISOTROPIC grid and the box aspect is rounded to it (below).
    fluids     rho_l = 1, rho_g = 0.1 (ratio 10); mu_l = mu_g = 3.3333e-4 (ratio 1); sigma = 5e-3.
    driving    `flowCtrl 1` + `Ret 127.3`.  In `auxiliaryRoutines::setPressGrad` this is
                   tau_w = (Ret nu_l / h)^2 / nu_l^2 ... = (Ret/Re)^2 with Re = 1/nu_l = 3000,
                   i.e. tau_w = 1.800588e-3, and the applied mean pressure gradient is
                   fs = tau_w - <rho> gCH.  The x-momentum source is then
                       fs + rho gCH = tau_w + (rho - <rho>) gCH,
               so **gCH = 0.1 is a streamwise GRAVITY that contributes only buoyancy** and does
               NOT enter Re_tau: u_tau = Re_tau nu_l / h = 0.0424333 exactly, by construction.
               (`g 0.d0` in the specs is the WALL-NORMAL gravity, and it is off.)  The bubbles are
               therefore driven AGAINST the mean flow: this is a downflow channel.
    bubbles    `initBubbles method 2, nbx 6, nby 1, nbz 3, R 0.125, random_distr .FALSE.` ->
               `vofBlocks::readBubblesArray` puts 18 bubbles on the CENTRELINE y = 1 at
               x0 = (i-1/2) Lx/6, z0 = (k-1/2) Lz/3.  Void fraction 1.49 %.  D/h = 0.25.
    BCs        no-slip on both walls; zero normal gradient for the colour and the pressure.
    IC         `channel_18/0/{ux,uy,uz}` is a converged SINGLE-PHASE turbulent channel snapshot
               (raw Fortran STREAM: one int32 count then the internal field, x fastest, no ghosts,
               staggered sizes 193x160x96 / 192x161x96 / 192x160x97; a 96-byte trailer of stale
               patch records that the reader ignores).  psi and phi0* are identically zero.
               This script INTERPOLATES that snapshot onto its own grid when it is present -- the
               nearest thing to the same initial condition that a different grid admits -- and
               otherwise starts from a perturbed laminar profile and spins up.

DIMENSIONLESS GROUPS: Eo = rho_l gCH D^2 / sigma = 1.25, Mo = gCH mu_l^4/(rho_l^2 sigma^3) =
9.88e-9, density ratio 10, viscosity ratio 1, D+ = 31.8, void fraction 1.49 %.

REFERENCE DATA: **the repository ships none** -- 60 tracked files, no statistics, no profiles, no
post-processing.  The only published figure is qualitative (a streamwise-velocity contour in
`user_guide.pdf`).  So the profiles below are reported as OUR first datum, not as a comparison.

Usage:
  PYTHONPATH=<build> python tests/study/vof_channel_18.py [--ny 80] [--steps N] [--single]
                                                          [--tbf DIR] [--out FILE]
"""
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf


def arg(name, default, cast=float):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


TBF = str(arg("--tbf", "/home/frankp/Codes/TBFsolver/channel_18/0", str))
NY = int(arg("--ny", 80, int))
NSTEP = int(arg("--steps", 4000, int))
SINGLE = "--single" in sys.argv
OUT = str(arg("--out", "", str))

# ---------------------------------------------------------------- the physical case
LX, LY, LZ = math.pi, 2.0, math.pi / 2      # h = LY/2 = 1
RHO_L, RHO_G = 1.0, 0.1
MU = 3.33333333333333e-4
SIGMA = 5.0e-3
GCH = 0.1
RET = 127.3
RE = 1.0 / MU                                # rho_l h / mu_l
TAUW = (RET / RE) ** 2                       # = 1.800588e-3, the applied wall stress
UTAU = RET * MU                              # = 0.0424333
RBUB = 0.125
NBX, NBY, NBZ = 6, 1, 3

# ---------------------------------------------------------------- the isotropic grid
# flow's cells are cubic, so the box aspect must be an integer ratio.  ny cells across Ly = 2
# fixes the cell size; nx and nz are the nearest choices with plenty of factors of two (the
# pressure multigrid's depth is decided by those -- suite/docs/DECOMPOSITION_AND_MULTIGRID.md).
S = NY / LY                                  # cells per unit length; half-height h = NY/2 cells
NX = int(round(LX * S / 16)) * 16
NZ = int(round(LZ * S / 16)) * 16
LXC, LZC = NX / S, NZ / S                    # the box we actually get


def tbf_read(path, shape):
    """One `channel_18/0` binary: int32 count, then count float64s, x fastest, no ghosts."""
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        if n != shape[0] * shape[1] * shape[2]:
            raise ValueError(f"{path}: count {n} != {shape}")
        a = np.frombuffer(f.read(8 * n), dtype="<f8")
    return np.asfortranarray(a.reshape(shape, order="F"))


def interp_component(src, src_pos, dst_pos):
    """Trilinear resample of one staggered component.

    `src_pos[d]` and `dst_pos[d]` are the PHYSICAL coordinates of the sample points along axis d,
    normalised to [0,1) of the respective box.  x and z wrap (periodic), y clamps (walls).
    """
    out = src
    for d, wrap in enumerate((True, False, True)):
        sp, dp = src_pos[d], dst_pos[d]
        n = len(sp)
        if wrap:
            spx = np.concatenate([sp - 1.0, sp, sp + 1.0])
            idx = np.searchsorted(spx, dp) - 1
            idx = np.clip(idx, 0, len(spx) - 2)
            w = (dp - spx[idx]) / (spx[idx + 1] - spx[idx])
            i0, i1 = idx % n, (idx + 1) % n
        else:
            idx = np.clip(np.searchsorted(sp, dp) - 1, 0, n - 2)
            w = np.clip((dp - sp[idx]) / (sp[idx + 1] - sp[idx]), 0.0, 1.0)
            i0, i1 = idx, idx + 1
        sl = [slice(None)] * 3
        sl0, sl1 = list(sl), list(sl)
        sl0[d], sl1[d] = i0, i1
        shape = [1, 1, 1]
        shape[d] = len(dp)
        out = out[tuple(sl0)] * (1.0 - w.reshape(shape)) + out[tuple(sl1)] * w.reshape(shape)
    return np.asfortranarray(out)


def load_tbf_ic():
    """TBFsolver's turbulent snapshot, resampled onto our grid, in CELL units (u' = S u)."""
    nxt, nyt, nzt = 192, 160, 96
    ux = tbf_read(os.path.join(TBF, "ux"), (nxt + 1, nyt, nzt))
    uy = tbf_read(os.path.join(TBF, "uy"), (nxt, nyt + 1, nzt))
    uz = tbf_read(os.path.join(TBF, "uz"), (nxt, nyt, nzt + 1))
    # source sample positions, as fractions of the box
    fx = np.arange(nxt + 1) / nxt          # ux on x-faces (0 .. 1 inclusive)
    cx = (np.arange(nxt) + 0.5) / nxt
    fy = np.arange(nyt + 1) / nyt
    cy = (np.arange(nyt) + 0.5) / nyt
    fz = np.arange(nzt + 1) / nzt
    cz = (np.arange(nzt) + 0.5) / nzt
    # destination: flow's u(i) is the LOW x-face of cell i
    dfx = np.arange(NX) / NX
    dcx = (np.arange(NX) + 0.5) / NX
    dfy = np.arange(NY) / NY
    dcy = (np.arange(NY) + 0.5) / NY
    dfz = np.arange(NZ) / NZ
    dcz = (np.arange(NZ) + 0.5) / NZ
    u = interp_component(ux[:nxt], (fx[:nxt], cy, cz), (dfx, dcy, dcz))
    v = interp_component(uy[:, :nyt], (cx, fy[:nyt], cz), (dcx, dfy, dcz))
    w = interp_component(uz[:, :, :nzt], (cx, cy, fz[:nzt]), (dcx, dcy, dfz))
    return (np.asfortranarray(u * S), np.asfortranarray(v * S), np.asfortranarray(w * S))


def laminar_ic():
    """Perturbed parabolic fallback (mean flow set to the shipped snapshot's bulk, U_b = 0.6113)."""
    y = (np.arange(NY) + 0.5) / NY * LY
    prof = 1.5 * 0.6113 * (1.0 - (y - 1.0) ** 2)
    rng = np.random.default_rng(7)
    u = np.broadcast_to(prof[None, :, None], (NX, NY, NZ)).copy()
    u += 0.15 * 0.6113 * rng.standard_normal((NX, NY, NZ))
    v = 0.05 * 0.6113 * rng.standard_normal((NX, NY, NZ))
    w = 0.05 * 0.6113 * rng.standard_normal((NX, NY, NZ))
    for a in (u, v, w):
        a[:, 0, :] *= 0.0
        a[:, -1, :] *= 0.0
    return (np.asfortranarray(u * S), np.asfortranarray(v * S), np.asfortranarray(w * S))


def sphere_frac(shape, R, c, sub=16):
    nx, ny, nz = shape
    ax = (np.arange(nx)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    ay = (np.arange(ny)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub).ravel()
    X, Y = ax[:, None], ay[None, :]
    half = np.sqrt(np.maximum(R * R - (X - c[0]) ** 2 - (Y - c[1]) ** 2, 0.0))
    z0, z1 = c[2] - half, c[2] + half
    C = np.zeros((nx, ny, nz))
    for k in range(nz):
        seg = np.maximum(np.minimum(z1, k + 1) - np.maximum(z0, k), 0.0)
        C[:, :, k] = seg.reshape(nx, sub, ny, sub).mean(axis=(1, 3))
    return np.asfortranarray(C)


def main():
    Rc = RBUB * S
    print("=" * 100)
    print("TBFsolver channel_18 on the peclet BLOCK VoF container (WO-W12 gate 4)")
    print("=" * 100)
    print(f"  grid {NX} x {NY} x {NZ} isotropic (TBFsolver: 192 x 160 x 96 anisotropic)")
    print(f"  box  {LXC:.4f} x {LY:.4f} x {LZC:.4f} h   (case: {LX:.4f} x {LY:.4f} x {LZ:.4f} h; "
          f"the cubic-cell rounding costs {100*(LXC/LX-1):+.1f} % in x and "
          f"{100*(LZC/LZ-1):+.1f} % in z)")
    print(f"  h = {NY/2:.0f} cells, delta_nu = h/Re_tau = {NY/2/RET:.4f} cells -> "
          f"Delta+ = {RET/(NY/2):.2f} (TBFsolver: 2.08 / 1.59 / 2.08)")
    print(f"  D = {2*Rc:.1f} cells (D/Delta = {2*Rc:.1f}, D+ = {2*RBUB*RET:.1f}); 18 bubbles")
    print(f"  u_tau = {UTAU:.6f}, tau_w = {TAUW:.6e}, Re_tau = {RET}")

    s = pf.Solver(NX, NY, NZ)
    s.set_rho(RHO_L)
    s.set_mu(S * S * MU)
    s.set_domain_bc(2, 1, 0, 0, 0)     # -y wall
    s.set_domain_bc(3, 1, 0, 0, 0)     # +y wall
    s.set_pressure_geometry(np.full((NX, NY, NZ), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-10)

    if os.path.isdir(TBF):
        u, v, w = load_tbf_ic()
        src = f"TBFsolver's own snapshot, resampled ({TBF})"
    else:
        u, v, w = laminar_ic()
        src = "a perturbed parabolic profile (TBFsolver's snapshot not found)"
    for c, a in enumerate((u, v, w)):
        s.set_velocity(c, a)
    print(f"  initial condition: {src}")
    ub = float(u.mean()) / S
    print(f"  bulk velocity of the IC: U_b = {ub:.4f} (Re_b = {ub/MU:.0f}, U_b+ = {ub/UTAU:.2f})")

    if not SINGLE:
        s.enable_vof()
        C = np.zeros((NX, NY, NZ), order="F")
        boxes = []
        for k in range(NBZ):
            for i in range(NBX):
                cx = (i + 0.5) * NX / NBX
                cy = NY / 2.0
                cz = (k + 0.5) * NZ / NBZ
                B = sphere_frac((NX, NY, NZ), Rc, (cx, cy, cz))
                C = np.maximum(C, B)
                lo = [int(math.floor(q - Rc)) - 1 for q in (cx, cy, cz)]
                hi = [int(math.ceil(q + Rc)) + 1 for q in (cx, cy, cz)]
                boxes.append([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
        s.set_vof(np.asfortranarray(C))
        alpha = float(C.sum()) / (NX * NY * NZ)
        rho_av = RHO_L + (RHO_G - RHO_L) * alpha
        print(f"  void fraction {100*alpha:.3f} % (case: 1.492 %), <rho> = {rho_av:.6f}")
        # x-momentum source per unit volume, in cell units:  tau_w + (rho(C) - <rho>) gCH
        f0 = S * (TAUW + (RHO_L - rho_av) * GCH)
        f1 = S * (RHO_G - RHO_L) * GCH
        s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
        s.set_property_model("mu", "linear", "C", [S * S * MU, 0.0])
        s.set_property_model("force_x", "linear", "C", [f0, f1])
        s.set_surface_tension(S ** 3 * SIGMA)
        s.enable_vof_blocks_from_field(boxes)
        s.enable_vof_block_csf()
        print(f"  markers seeded: {len(s.vof_block_stats())}; force_x = {f0:.6e} {f1:+.6e} C")
    else:
        s.set_property_model("force_x", "const", "C", [S * TAUW])
        print(f"  SINGLE PHASE: force_x = {S*TAUW:.6e}")

    # ---- march
    yq = (np.arange(NY) + 0.5) / S           # wall-normal coordinate, in h
    accU = np.zeros(NY)
    accC = np.zeros(NY)
    nacc = 0
    t, wall0 = 0.0, time.time()
    maxit, maxdiv = 0, 0.0
    for i in range(NSTEP):
        if i % 10 == 0:
            L = s.vof_step_limits() if not SINGLE else None
            dt = 0.2 / max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(),
                           np.abs(s.get_w()).max(), 1e-30)
            if L is not None:
                dt = min(dt, 0.4 * L["cfl_dt"], 0.4 * L["capillary_dt"])
            s.set_dt(dt)
        t += s.dt()
        s.step()
        maxit = max(maxit, s.last_pressure_iterations())
        maxdiv = max(maxdiv, s.max_open_divergence())
        if i >= NSTEP // 2:                   # average over the second half
            accU += s.get_u().mean(axis=(0, 2))
            if not SINGLE:
                accC += s.get_vof().mean(axis=(0, 2))
            nacc += 1
        if i % 100 == 0:
            um = s.get_u().mean() / S
            print(f"    step {i:6d}  t = {t/S:8.3f} h/u  dt = {s.dt():.3e}  U_b = {um:.4f}  "
                  f"press {s.last_pressure_iterations():3d}/800  ({time.time()-wall0:.0f} s)")
            if not SINGLE:
                st = s.vof_block_stats()
                v = [b["volume"] for b in st]
                print(f"           markers {len(st)}: V min {min(v):.2f} max {max(v):.2f} "
                      f"(seed {4/3*math.pi*Rc**3:.2f}), total area "
                      f"{sum(b['area'] for b in st):.0f} cells^2")
    U = accU / max(nacc, 1) / S
    Cp = accC / max(nacc, 1)
    print(f"\n  {NSTEP} steps in {time.time()-wall0:.0f} s; pressure {maxit}/800 "
          f"{'OK' if maxit < 800 else '*** CAPPED -> RUN INVALID ***'}, "
          f"max|div(open u)| {maxdiv:.2e}")
    utau_m = math.sqrt(MU * abs(U[0]) / (yq[0]))
    print(f"  wall-gradient u_tau (first cell) {utau_m:.6f} vs the imposed {UTAU:.6f} "
          f"({100*(utau_m/UTAU-1):+.1f} %)")
    print("\n  y/h      y+        <u>/u_tau     void fraction")
    for j in range(0, NY // 2):
        print(f"  {yq[j]:7.4f}  {yq[j]/ (MU/UTAU):8.2f}  {U[j]/UTAU:12.4f}  {Cp[j]:14.5f}")
    if OUT:
        np.savetxt(OUT, np.c_[yq, U / UTAU, Cp],
                   header="y/h  <u>/u_tau  void_fraction   (peclet block VoF, channel_18)")
        print(f"\n  written {OUT}")


if __name__ == "__main__":
    main()
