#!/usr/bin/env python3
"""VoF Part III rung W3 (WO-W3) — TBFsolver's `channel_18` to a statistically steady state.

WO-W12 gate 4 transcribed the case and ran it for 0.84 eddy turnovers on one shared workstation
GPU (`tests/study/vof_channel_18.py`).  W3 is the same case run to ~20 eddy turnovers on Snellius
H100s, which no single job's wall clock can hold, so this driver differs from that one in exactly
three ways and in nothing else:

  1. **it checkpoints and restarts** — velocity + every marker's OWN colour (`vof_block_colour` /
     `enable_vof_blocks_from_colours`, added for this rung: re-seeding from the union field is
     NOT exact once two markers touch, WO-W12 open item 5),
  2. **it accumulates the statistics across chunks** (running sums live in the checkpoint), with an
     explicit transient window that is discarded, and
  3. **it stops on a wall-clock budget** as well as on a step/turnover target, so a chunk always
     ends with a checkpoint on disk.

The physical case is WO-W12's transcription verbatim — see `tests/study/vof_channel_18.py` for the
derivation of every number from the Fortran that consumes `channel_18/specs`:

    geometry   Lx = pi (streamwise, periodic), Ly = 2 (wall-normal, WALLS), Lz = pi/2 (spanwise,
               periodic), h = 1;  mapped onto an ISOTROPIC ny=80 grid -> 128 x 80 x 64.
    fluids     rho_l = 1, rho_g = 0.1;  mu_l = mu_g = 3.3333e-4;  sigma = 5e-3.
    driving    x-momentum source  tau_w + (rho - <rho>) gCH,  tau_w = (Re_tau/Re)^2 = 1.800588e-3,
               gCH = 0.1 a STREAMWISE GRAVITY (buoyancy only) -> u_tau = 0.0424333 by construction.
               The bubbles are driven against the mean flow: a DOWNFLOW channel.
    bubbles    18 spheres of R = 0.125 h on the centreline, 6 x 3 in (x, z); void fraction 1.49 %.
    IC         `channel_18/0/{ux,uy,uz}`, TBFsolver's own converged single-phase snapshot,
               trilinearly resampled onto our grid (U_b = 0.6113).

One eddy turnover is h/u_tau; the driver reports `t u_tau / h` everywhere.

STATISTICS (plane averages over x and z, per wall-normal row; sampled every `--sample` steps and
accumulated only after `t u_tau/h >= --stats-start`):

    <C>(y)                     the void fraction (C is the GAS indicator in this transcription:
                               rho = rho_l + (rho_g - rho_l) C)
    <u>(y), <v>(y), <w>(y)     cell-centred velocity means
    <u'u'>, <v'v'>, <w'w'>     rms components (from the accumulated squares)
    <u'v'>                     the Reynolds shear stress
    the same six conditioned on the LIQUID with the weight (1 - C)  ->  <u>_l etc.

    u_tau is reported three ways: the imposed 0.0424333, the wall gradient of <u>_l at the first
    cell, and the wall gradient at both walls separately (their spread is the convergence measure).

Usage (one chunk):
  PYTHONPATH=<build> python run_channel_18.py --ckpt ck.npz --steps 20000 --wall 39000 \
      --turnovers 20 --stats-start 4 --out results.npz
Re-running the same command resumes from `ck.npz` and stops when the turnover target is reached.
"""
import json
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
NSTEP = int(arg("--steps", 20000, int))
CKPT = str(arg("--ckpt", "channel_18_ckpt.npz", str))
OUT = str(arg("--out", "channel_18_stats.npz", str))
LOGJ = str(arg("--chunklog", "channel_18_chunks.jsonl", str))
WALL = float(arg("--wall", 1e18))
TARGET = float(arg("--turnovers", 20.0))
STATS_START = float(arg("--stats-start", 4.0))
SAMPLE = int(arg("--sample", 10, int))
CKEVERY = int(arg("--ckpt-every", 2000, int))
DTEVERY = int(arg("--dt-every", 1, int))
DTSAFE = float(arg("--dt-safety", 0.25))
QUICK = "--quick" in sys.argv

if QUICK:                                    # build/plumbing validation, not physics
    NY = int(arg("--ny", 32, int))
    NSTEP = int(arg("--steps", 40, int))
    TARGET = float(arg("--turnovers", 1e9))
    STATS_START = float(arg("--stats-start", 0.0))
    SAMPLE = int(arg("--sample", 2, int))
    CKEVERY = int(arg("--ckpt-every", 20, int))

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
S = NY / LY                                  # cells per unit length; half-height h = NY/2 cells
NX = int(round(LX * S / 16)) * 16
NZ = int(round(LZ * S / 16)) * 16
LXC, LZC = NX / S, NZ / S


# ---------------------------------------------------------------- TBFsolver's own IC
def tbf_read(path, shape):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        if n != shape[0] * shape[1] * shape[2]:
            raise ValueError(f"{path}: count {n} != {shape}")
        a = np.frombuffer(f.read(8 * n), dtype="<f8")
    return np.asfortranarray(a.reshape(shape, order="F"))


def interp_component(src, src_pos, dst_pos):
    out = src
    for d, wrap in enumerate((True, False, True)):
        sp, dp = src_pos[d], dst_pos[d]
        n = len(sp)
        if wrap:
            spx = np.concatenate([sp - 1.0, sp, sp + 1.0])
            idx = np.clip(np.searchsorted(spx, dp) - 1, 0, len(spx) - 2)
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
    nxt, nyt, nzt = 192, 160, 96
    ux = tbf_read(os.path.join(TBF, "ux"), (nxt + 1, nyt, nzt))
    uy = tbf_read(os.path.join(TBF, "uy"), (nxt, nyt + 1, nzt))
    uz = tbf_read(os.path.join(TBF, "uz"), (nxt, nyt, nzt + 1))
    fx = np.arange(nxt + 1) / nxt
    cx = (np.arange(nxt) + 0.5) / nxt
    fy = np.arange(nyt + 1) / nyt
    cy = (np.arange(nyt) + 0.5) / nyt
    fz = np.arange(nzt + 1) / nzt
    cz = (np.arange(nzt) + 0.5) / nzt
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


# ---------------------------------------------------------------- the accumulator
KEYS = ("C", "u", "v", "w", "uu", "vv", "ww", "uv",
        "L", "Lu", "Lv", "Lw", "Luu", "Lvv", "Lww", "Luv")


def new_acc():
    a = {k: np.zeros(NY) for k in KEYS}
    a["_n"] = 0.0
    a["_t0"] = -1.0
    a["_t1"] = -1.0
    return a


def sample(acc, s, t):
    """One plane-averaged sample. Velocities are interpolated to CELL CENTRES first.

    flow's `u` is the LOW-x face of each cell and x is periodic, `v` the low-y face (the +y face of
    the last row is the wall, v = 0), `w` the low-z face with z periodic — so the centred value is
    the mean of the cell's two faces on each axis.
    """
    u = s.get_u()
    v = s.get_v()
    w = s.get_w()
    uc = 0.5 * (u + np.roll(u, -1, axis=0))
    vc = np.empty_like(v)
    vc[:, :-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    vc[:, -1, :] = 0.5 * v[:, -1, :]              # +y face of the last row is the wall (v = 0)
    wc = 0.5 * (w + np.roll(w, -1, axis=2))
    uc /= S
    vc /= S
    wc /= S
    C = s.get_vof() if s.vof_blocks_enabled() else np.zeros_like(uc)
    L = 1.0 - C
    ax = (0, 2)
    acc["C"] += C.mean(axis=ax)
    acc["u"] += uc.mean(axis=ax)
    acc["v"] += vc.mean(axis=ax)
    acc["w"] += wc.mean(axis=ax)
    acc["uu"] += (uc * uc).mean(axis=ax)
    acc["vv"] += (vc * vc).mean(axis=ax)
    acc["ww"] += (wc * wc).mean(axis=ax)
    acc["uv"] += (uc * vc).mean(axis=ax)
    acc["L"] += L.mean(axis=ax)
    acc["Lu"] += (L * uc).mean(axis=ax)
    acc["Lv"] += (L * vc).mean(axis=ax)
    acc["Lw"] += (L * wc).mean(axis=ax)
    acc["Luu"] += (L * uc * uc).mean(axis=ax)
    acc["Lvv"] += (L * vc * vc).mean(axis=ax)
    acc["Lww"] += (L * wc * wc).mean(axis=ax)
    acc["Luv"] += (L * uc * vc).mean(axis=ax)
    acc["_n"] += 1.0
    if acc["_t0"] < 0.0:
        acc["_t0"] = t
    acc["_t1"] = t


def profiles(acc):
    """Turn the running sums into the reported profiles (dimensional, then in wall units)."""
    n = max(acc["_n"], 1.0)
    m = {k: acc[k] / n for k in KEYS}
    out = {}
    yq = (np.arange(NY) + 0.5) / S                     # wall-normal coordinate in h
    out["y"] = yq
    out["yplus"] = yq / (MU / UTAU)
    out["alpha"] = m["C"]                              # void fraction
    out["u"] = m["u"]
    out["v"] = m["v"]
    out["w"] = m["w"]
    out["urms"] = np.sqrt(np.maximum(m["uu"] - m["u"] ** 2, 0.0))
    out["vrms"] = np.sqrt(np.maximum(m["vv"] - m["v"] ** 2, 0.0))
    out["wrms"] = np.sqrt(np.maximum(m["ww"] - m["w"] ** 2, 0.0))
    out["uv"] = m["uv"] - m["u"] * m["v"]
    Lw = np.maximum(m["L"], 1e-30)
    ul = m["Lu"] / Lw
    vl = m["Lv"] / Lw
    out["u_liq"] = ul
    out["v_liq"] = vl
    out["w_liq"] = m["Lw"] / Lw
    out["urms_liq"] = np.sqrt(np.maximum(m["Luu"] / Lw - ul ** 2, 0.0))
    out["vrms_liq"] = np.sqrt(np.maximum(m["Lvv"] / Lw - vl ** 2, 0.0))
    out["wrms_liq"] = np.sqrt(np.maximum(m["Lww"] / Lw - out["w_liq"] ** 2, 0.0))
    out["uv_liq"] = m["Lvv"] * 0.0 + (m["Luv"] / Lw - ul * vl)
    out["liquid_fraction"] = m["L"]
    # u_tau from the wall gradient of the liquid mean, each wall separately
    dy = 1.0 / S
    utau_lo = math.sqrt(MU * abs(ul[0]) / (0.5 * dy))
    utau_hi = math.sqrt(MU * abs(ul[-1]) / (0.5 * dy))
    out["utau_lo"] = utau_lo
    out["utau_hi"] = utau_hi
    out["utau_wall"] = 0.5 * (utau_lo + utau_hi)
    out["utau_imposed"] = UTAU
    out["samples"] = acc["_n"]
    out["window_turnovers"] = np.array([acc["_t0"] * UTAU, acc["_t1"] * UTAU])
    return out


# ---------------------------------------------------------------- checkpoint I/O
def save_ckpt(path, s, step, t, acc, blocks, alpha0):
    d = {"step": np.int64(step), "t": np.float64(t), "ny": np.int64(NY),
         "alpha0": np.float64(alpha0), "nblocks": np.int64(len(blocks))}
    d["u"] = s.get_u()
    d["v"] = s.get_v()
    d["w"] = s.get_w()
    # The ACCUMULATED physical pressure is state, not a diagnostic: the incremental-rotational
    # scheme carries P from step to step and the momentum RHS reads -(P(i) - P(i-s)).  Dropping it
    # at a chunk boundary costs a restart transient (measured: 1.6e-3 relative in u after 20 steps
    # of a quick run, against 3e-14 with it restored).  "p" is the registered name of that field.
    d["p"] = s.get_field("p")
    for k in KEYS:
        d["acc_" + k] = acc[k]
    d["acc_n"] = np.float64(acc["_n"])
    d["acc_t0"] = np.float64(acc["_t0"])
    d["acc_t1"] = np.float64(acc["_t1"])
    for i, (box, col) in enumerate(blocks):
        d[f"box{i}"] = np.asarray(box, dtype=np.int64)
        d[f"col{i}"] = np.asarray(col, dtype=np.float64)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)                       # atomic: a killed job never leaves a torn file


def load_ckpt(path):
    z = np.load(path)
    if int(z["ny"]) != NY:
        raise SystemExit(f"checkpoint {path} is ny={int(z['ny'])}, this run is ny={NY}")
    acc = new_acc()
    for k in KEYS:
        acc[k] = np.array(z["acc_" + k])
    acc["_n"] = float(z["acc_n"])
    acc["_t0"] = float(z["acc_t0"])
    acc["_t1"] = float(z["acc_t1"])
    nb = int(z["nblocks"])
    blocks = [(np.array(z[f"box{i}"]).tolist(), np.asfortranarray(z[f"col{i}"]))
              for i in range(nb)]
    pres = np.asfortranarray(z["p"]) if "p" in z.files else None
    return int(z["step"]), float(z["t"]), acc, blocks, \
        (np.asfortranarray(z["u"]), np.asfortranarray(z["v"]), np.asfortranarray(z["w"])), \
        float(z["alpha0"]), pres


def block_state(s):
    """{box, own colour} per marker — the complete state of the block container."""
    out = []
    for b in s.vof_block_stats():
        lo, hi = b["lo"], b["hi"]
        c = s.vof_block_colour(b["id"])
        out.append(([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]], np.asfortranarray(c)))
    return out


# ---------------------------------------------------------------- the run
def main():
    Rc = RBUB * S
    resume = os.path.exists(CKPT)
    print("=" * 100)
    print("TBFsolver channel_18 on the peclet BLOCK VoF container — WO-W3 (statistically steady)")
    print("=" * 100)
    print(f"  grid {NX} x {NY} x {NZ} isotropic (TBFsolver: 192 x 160 x 96 anisotropic)")
    print(f"  box  {LXC:.4f} x {LY:.4f} x {LZC:.4f} h   (case: {LX:.4f} x {LY:.4f} x {LZ:.4f} h)")
    print(f"  Delta+ = {RET/(NY/2):.2f}, D = {2*Rc:.1f} cells, 18 bubbles, "
          f"u_tau = {UTAU:.6f}, tau_w = {TAUW:.6e}")
    print(f"  target {TARGET} eddy turnovers; statistics start at t u_tau/h = {STATS_START}; "
          f"sample every {SAMPLE} steps; wall budget {WALL:.0f} s")
    print(f"  checkpoint {CKPT} ({'RESUME' if resume else 'fresh start'}), out {OUT}")

    s = pf.Solver(NX, NY, NZ)
    s.set_rho(RHO_L)
    s.set_mu(S * S * MU)
    s.set_domain_bc(2, 1, 0, 0, 0)     # -y wall
    s.set_domain_bc(3, 1, 0, 0, 0)     # +y wall
    s.set_pressure_geometry(np.full((NX, NY, NZ), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-10)

    step0, t, acc = 0, 0.0, new_acc()
    alpha0, pres = None, None
    if resume:
        step0, t, acc, blocks, (u, v, w), alpha0, pres = load_ckpt(CKPT)
        print(f"  resumed at step {step0}, t u_tau/h = {t*UTAU:.4f}, "
              f"{int(acc['_n'])} samples already accumulated, {len(blocks)} markers")
    else:
        if os.path.isdir(TBF):
            u, v, w = load_tbf_ic()
            print(f"  initial condition: TBFsolver's own snapshot, resampled ({TBF})")
        else:
            u, v, w = laminar_ic()
            print("  initial condition: a perturbed parabolic profile (snapshot not found)")
        blocks = None
    for c, a in enumerate((u, v, w)):
        s.set_velocity(c, a)
    print(f"  bulk velocity: U_b = {float(u.mean())/S:.4f}")

    s.enable_vof()
    if blocks is None:
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
        alpha0 = float(C.sum()) / (NX * NY * NZ)
    else:
        C = np.zeros((NX, NY, NZ), order="F")
        s.set_vof(C)                       # the union is rebuilt by the seeding scatter
    rho_av = RHO_L + (RHO_G - RHO_L) * alpha0
    f0 = S * (TAUW + (RHO_L - rho_av) * GCH)
    f1 = S * (RHO_G - RHO_L) * GCH
    s.set_property_model("rho", "linear", "C", [RHO_L, RHO_G - RHO_L])
    s.set_property_model("mu", "linear", "C", [S * S * MU, 0.0])
    s.set_property_model("force_x", "linear", "C", [f0, f1])
    s.set_surface_tension(S ** 3 * SIGMA)
    if blocks is None:
        s.enable_vof_blocks_from_field(boxes)
    else:
        s.enable_vof_blocks_from_colours([b for b, _ in blocks], [c for _, c in blocks])
    s.enable_vof_block_csf()
    st0 = s.vof_block_stats()
    vol0 = [b["volume"] for b in st0]
    print(f"  void fraction {100*alpha0:.3f} % (case: 1.492 %), <rho> = {rho_av:.6f}; "
          f"{len(st0)} markers, V in [{min(vol0):.3f}, {max(vol0):.3f}] "
          f"(sphere seed {4/3*math.pi*Rc**3:.3f})")

    # AFTER the geometry (set_pressure_geometry zeroes P and phi when it rebuilds the operator).
    if pres is not None:
        s.set_field("p", pres)
        print("  accumulated pressure restored from the checkpoint")
    # The Weymouth-Yue SWEEP PERMUTATION is state too: the order is kWySweepPerm[n % 6] and the
    # block container's counter runs from 0, one increment per block advection — i.e. it equals the
    # step index.  Resuming with it reset changes the colour at the splitting error (measured
    # 6.2e-4 after ONE step, off a bitwise-identical velocity).  set_vof_step_parity sets the
    # solver's counter AND the block container's.
    s.set_vof_step_parity(step0)

    seedV = 4.0 / 3.0 * math.pi * Rc ** 3
    if NSTEP == 0:      # state round-trip diagnostic: checkpoint the restored state and stop
        save_ckpt(CKPT + ".roundtrip.npz", s, step0, t, acc, block_state(s), alpha0)
        print(f"  --steps 0: wrote {CKPT}.roundtrip.npz and stopped")
        return
    maxit, maxdiv, dt = 0, 0.0, 0.0
    wall0 = time.time()
    stepped = 0
    stop = "steps"
    i = step0
    while stepped < NSTEP:
        if i % DTEVERY == 0:   # ABSOLUTE cadence: a restart must not shift it
            L = s.vof_step_limits()
            dt = 0.2 / max(np.abs(s.get_u()).max(), np.abs(s.get_v()).max(),
                           np.abs(s.get_w()).max(), 1e-30)
            # A NON-FINITE limit is a state failure, not a large dt, and Python's min() would let
            # it through silently (min(x, nan) == x): the capillary limit is
            # sqrt((rho_min + rho_max) h^3 / 4 pi sigma), so it goes -nan the moment the colour
            # leaves [0,1] far enough to make a density negative.  Refuse to step on it.
            for nm in ("cfl_dt", "capillary_dt"):
                if not (L[nm] > 0.0 and math.isfinite(L[nm])):
                    with open(CKPT + ".failed", "w") as f:
                        f.write(json.dumps({"step": i, "t": t, "turnovers": t * UTAU,
                                            "limit": nm, "value": L[nm],
                                            "vof": s.vof_diagnostics()}, default=float) + "\n")
                    raise SystemExit(
                        f"  *** step {i}: {nm} = {L[nm]} at t u_tau/h = {t*UTAU:.4f} — the state "
                        f"is already broken (the colour has left [0,1]); the last checkpoint is "
                        f"untouched.")
            dt = min(dt, DTSAFE * L["cfl_dt"], DTSAFE * L["capillary_dt"])
            s.set_dt(dt)
        t += dt
        try:
            s.step()
        except Exception as exc:               # a Weymouth-Yue boundedness throw, or worse
            # step() is NOT atomic across that throw (VOF_PLAN 13.2 item 9: the momentum half has
            # already advanced by the rejected dt), so catch-and-halve would desynchronise colour
            # and momentum.  Leave the LAST GOOD checkpoint untouched, record what happened, and
            # fail the chunk so `--dependency=afterok` stops the chain instead of marching on.
            with open(CKPT + ".failed", "w") as f:
                f.write(json.dumps({"step": i, "t": t, "turnovers": t * UTAU, "dt": dt,
                                    "error": repr(exc)}) + "\n")
            print(f"\n  *** step {i} FAILED at t u_tau/h = {t*UTAU:.4f}, dt = {dt:.3e}: {exc}")
            print(f"  the last checkpoint ({CKPT}) is untouched; restart from it with a smaller "
                  f"dt safety factor.  Chain stopped.")
            raise
        i += 1
        stepped += 1
        it = s.last_pressure_iterations()
        maxit = max(maxit, it)
        maxdiv = max(maxdiv, s.max_open_divergence())
        if t * UTAU >= STATS_START and (i % SAMPLE == 0):
            sample(acc, s, t)
        if i % 200 == 0:
            el = time.time() - wall0
            D = s.vof_diagnostics()
            cmx = max(float(D.get("max", 0.0)), float(D.get("max_fluid", 0.0)))
            print(f"    step {i:7d}  t u_tau/h = {t*UTAU:8.4f}  dt = {dt:.3e}  "
                  f"U_b = {s.get_u().mean()/S:.4f}  press {it:3d}/800  maxC {cmx:.6f}  "
                  f"{1000*el/max(stepped,1):.0f} ms/step  ({el:.0f} s)", flush=True)
            # The colour leaving [0,1] is the Weymouth-Yue boundedness cap being exceeded, and it
            # is QUIET: volume still telescopes exactly, so nothing else notices until a density
            # goes negative and the capillary limit returns -nan.  Stop while the checkpoint is
            # still good.
            if not (cmx <= 1.0 + 1e-6):
                with open(CKPT + ".failed", "w") as f:
                    f.write(json.dumps({"step": i, "t": t, "turnovers": t * UTAU,
                                        "max_C": cmx, "dt": dt}, default=float) + "\n")
                raise SystemExit(f"  *** step {i}: max C = {cmx} > 1 — Weymouth-Yue boundedness "
                                 f"lost.  Reduce --dt-safety and restart from the checkpoint.")
        done_t = t * UTAU >= TARGET
        done_w = (time.time() - wall0) > WALL
        if done_t or done_w or (i % CKEVERY == 0) or stepped >= NSTEP:
            save_ckpt(CKPT, s, i, t, acc, block_state(s), alpha0)
            if done_t or done_w:
                stop = "turnovers" if done_t else "wall"
                break

    el = time.time() - wall0
    st = s.vof_block_stats()
    vols = [b["volume"] for b in st]
    capped = maxit >= 800
    rec = {
        "stop": stop, "step": i, "steps_this_chunk": stepped, "t": t,
        "turnovers": t * UTAU, "wall_s": el, "ms_per_step": 1000 * el / max(stepped, 1),
        "max_pressure_iterations": maxit, "pressure_cap": 800, "capped": bool(capped),
        "max_open_divergence": maxdiv, "markers": len(st),
        "volume_min": min(vols), "volume_max": max(vols), "volume_seed": seedV,
        "volume_rel_err": max(abs(v / seedV - 1.0) for v in vols),
        "area_total": sum(b["area"] for b in st),
        "samples": acc["_n"], "grid": [NX, NY, NZ],
    }
    print("\n  " + json.dumps(rec))
    with open(LOGJ, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if capped:
        print("  *** PRESSURE CAPPED -> THIS CHUNK IS INVALID (rule 3b) ***")

    if acc["_n"] > 0:
        pr = profiles(acc)
        np.savez(OUT, **{k: np.asarray(v) for k, v in pr.items()},
                 turnovers=np.float64(t * UTAU), step=np.int64(i),
                 grid=np.asarray([NX, NY, NZ]), utau=np.float64(UTAU))
        print(f"  profiles written to {OUT} ({int(acc['_n'])} samples over "
              f"t u_tau/h {pr['window_turnovers'][0]:.3f} .. {pr['window_turnovers'][1]:.3f})")
        print(f"  u_tau: imposed {UTAU:.6f}, wall gradient {pr['utau_wall']:.6f} "
              f"({100*(pr['utau_wall']/UTAU-1):+.1f} %; walls {pr['utau_lo']:.6f} / "
              f"{pr['utau_hi']:.6f})")
        print("\n   y/h      y+       <u>/u_tau  <u>_liq/u_tau  alpha     u'/u_tau  v'/u_tau  "
              "w'/u_tau  -<u'v'>/u_tau^2")
        for j in range(NY):
            print(f"  {pr['y'][j]:7.4f} {pr['yplus'][j]:8.2f} {pr['u'][j]/UTAU:11.4f} "
                  f"{pr['u_liq'][j]/UTAU:13.4f} {pr['alpha'][j]:9.5f} "
                  f"{pr['urms'][j]/UTAU:9.4f} {pr['vrms'][j]/UTAU:9.4f} "
                  f"{pr['wrms'][j]/UTAU:9.4f} {-pr['uv'][j]/UTAU**2:15.5f}")
    else:
        print("  no statistics yet (still inside the discarded transient)")
    print(f"  chunk ended on '{stop}' at step {i}, t u_tau/h = {t*UTAU:.4f}")
    if stop == "turnovers":
        # the marker a chained job looks for: the target is reached, later chunks do nothing
        with open(CKPT + ".done", "w") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  TARGET REACHED -> {CKPT}.done written")


if __name__ == "__main__":
    main()
