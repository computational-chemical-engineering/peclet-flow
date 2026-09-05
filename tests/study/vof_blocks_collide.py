#!/usr/bin/env python3
"""VoF Part III rung W4 (WO-W4) — overlapping markers: one force from the union, and collision,
coalescence and breakup as EXPLICIT models.

Rung W2 formed the surface-tension force on each marker's own block and scattered the three face
forces into the global RHS with UNPACK_SUM.  That is right where a face belongs to ONE marker and
wrong where two markers overlap: the projection sees a single union colour there, so two forces act
against one colour jump and the balanced-force pairing that makes V4 exact — the force being the
discrete gradient of `sigma kappa C` under the SAME difference operator the projection inverts —
does not hold.  WO-W3 measured what that costs: `channel_18` blows up inside ONE step at ~1.5 eddy
turnovers, at a time step 4x below the one that fails, the moment two markers interpenetrate.

W4's rule: each marker still computes its own curvature on its own colour, but it contributes
`(A, |dC|, kappa|dC|, 1)` per face instead of a force, and the owner forms ONE force from the
UNION's own face difference with the |dC|-weighted mean curvature.  A face with a single marker
takes that marker's own force bit for bit.

    bitwise   G1: the new rule is BITWISE rung W2 wherever no two markers claim the same face
    pair      G3: two equal bubbles in line, coalescence 'never' — the film drains, they do not
                  merge, both volumes hold and the union colour never exceeds 1
    merge     G4: the same pair with the 'weber' model — one marker, volume = the sum
    split     G5: an elongated marker breaks; the children's volumes sum to the parent's exactly

Usage:  PYTHONPATH=<build> python tests/study/vof_blocks_collide.py [gate ...] [--quick]
"""
import math
import os
import sys
import time

import numpy as np

import peclet.flow as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vof_surface_tension import Scale, Solve, sphere_fractions  # noqa: E402
from vof_blocks_ns import HYSING1, build_hysing, grace_terminal, marker_box  # noqa: E402

QUICK = "--quick" in sys.argv
GATES = [a for a in sys.argv[1:] if not a.startswith("--")]


def field_digest(s):
    """Everything a step can change, as raw bytes: colour, the three velocities, the pressure."""
    return [np.ascontiguousarray(a) for a in
            (s.get_vof(), s.get_u(), s.get_v(), s.get_w(), s.get_field("p"))]


def bitdiff(a, b):
    """Largest absolute difference, and whether every array is bit-identical."""
    m = 0.0
    same = True
    for x, y in zip(a, b):
        if x.tobytes() != y.tobytes():
            same = False
        m = max(m, float(np.abs(x - y).max()))
    return m, same


# ------------------------------------------------------------------ G1: bitwise where no overlap
def gate_bitwise():
    print("\n" + "=" * 96)
    print("GATE G1 — the union force rule is BITWISE rung W2 wherever no two markers share a face")
    print("=" * 96)
    print("  Two runs of the same scene that differ ONLY in set_block_csf_union(True/False).  The")
    print("  gate is bit-identity of the colour, the three velocity components and the accumulated")
    print("  pressure at EVERY step, for as long as vof_block_overlap_faces() stays (0, 0, 0).")
    nsteps = 60 if QUICK else 300

    # --- scene A: one marker (Hysing case 1 through the block path) -- can never overlap
    print("\n  -- scene A: Hysing case 1, ONE marker (overlap is impossible)")
    nx = 32 if QUICK else 64
    runs = []
    for uni in (True, False):
        s, sc, nz, ny = build_hysing_union(nx, uni)
        h = Solve(600)
        dt = 0.4 * min(s.vof_step_limits()["cfl_dt"], s.capillary_dt())
        s.set_dt(dt)
        digest = []
        ov = 0
        for i in range(nsteps):
            if i % 10 == 0:
                L = s.vof_step_limits()
                s.set_dt(0.4 * min(L["cfl_dt"], L["capillary_dt"]))
            s.step()
            h.sample(s)
            ov = max(ov, max(s.vof_block_overlap_faces()))
            digest.append(field_digest(s))
        runs.append((digest, ov, h))
    dmax, same = 0.0, True
    for a, b in zip(runs[0][0], runs[1][0]):
        m, sm = bitdiff(a, b)
        dmax = max(dmax, m)
        same = same and sm
    print(f"     {nsteps} steps, {runs[0][2]};  overlap faces seen: {runs[0][1]}")
    print(f"     union vs W2: max|diff| {dmax:.3e}, bit-identical {same}  ->  "
          f"{'PASS' if same else 'FAIL'}")
    okA = same and runs[0][1] == 0

    # --- scene B: the W2 pair scene, two markers approaching -- bitwise UNTIL they overlap
    print("\n  -- scene B: two markers in line (the W2 pair scene), until their bands first meet")
    runs = []
    for uni in (True, False):
        s, v0 = build_pair(union=uni, gap=1.0, coalescence=None)
        h = Solve(800)
        digest, firstov = [], None
        for i in range(nsteps):
            if i % 5 == 0:
                L = s.vof_step_limits()
                s.set_dt(0.2 * min(L["cfl_dt"], L["capillary_dt"]))
            s.step()
            h.sample(s)
            if firstov is None and max(s.vof_block_overlap_faces()) > 0:
                firstov = i
            digest.append(field_digest(s))
        runs.append((digest, firstov, h))
    n_clean = runs[0][1] if runs[0][1] is not None else nsteps
    dmax, same = 0.0, True
    for k in range(n_clean):
        m, sm = bitdiff(runs[0][0][k], runs[1][0][k])
        dmax = max(dmax, m)
        same = same and sm
    dafter = 0.0
    if runs[0][1] is not None:
        dafter = max(bitdiff(runs[0][0][k], runs[1][0][k])[0] for k in range(n_clean, nsteps))
    print(f"     {nsteps} steps, {runs[0][2]};  first face claimed by two markers at step "
          f"{runs[0][1]}")
    print(f"     steps 0..{n_clean - 1}: max|diff| {dmax:.3e}, bit-identical {same}")
    print(f"     after it the two rules DIFFER by design: max|diff| {dafter:.3e}")
    print(f"\n  G1 -> {'PASS' if (okA and same) else 'FAIL'}")


def build_hysing_union(nx, union):
    """`build_hysing` with the block path, with the union rule switched on or off."""
    p = HYSING1
    sc = Scale(nx / 1.0)
    nz, ny = 2 * nx, 4
    s = pf.Solver(nx, ny, nz)
    s.set_rho(p["rho1"])
    s.set_mu(sc.mu(p["mu1"]))
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 600, 1e-12)
    s.enable_vof()
    from vof_surface_tension import cylinder_fractions
    R = sc.len_to_cells(0.25)
    B = cylinder_fractions((nx, ny, nz), R, nx / 2.0, sc.len_to_cells(0.5))
    s.set_vof(np.asfortranarray(B))
    s.set_property_model("rho", "linear", "C", [p["rho1"], p["rho2"] - p["rho1"]])
    s.set_property_model("mu", "linear", "C", [sc.mu(p["mu1"]), sc.mu(p["mu2"] - p["mu1"])])
    s.set_surface_tension(sc.sigma(p["sigma"]))
    s.set_property_model("force_z", "linear", "C",
                         [-sc.bodyforce(p["rho1"] * p["g"]),
                          -sc.bodyforce((p["rho2"] - p["rho1"]) * p["g"])])
    s.enable_vof_blocks_from_field([marker_box(B)])
    s.set_block_csf_union(union)
    s.enable_vof_block_csf()
    return s, sc, nz, ny


# ------------------------------------------------------------------------ the two-marker scene
EO, MO, RATIO = 10.0, 1e-3, 100.0


def pair_params(D):
    rho_l, sigma = 1.0, 1.0
    g = EO * sigma / (rho_l * D * D)
    mu_l = (MO * rho_l ** 2 * sigma ** 3 / g) ** 0.25
    return rho_l, sigma, g, mu_l


def build_pair(union=True, gap=2.0, coalescence=None, D=None, equal=True):
    """Two bubbles in line, seeded `gap` cells apart, each as its OWN marker (sphere seeding, so
    neither adopts a slice of the other and the 1e-12 volume gate means something)."""
    D = D or (10.0 if QUICK else 12.0)
    R0 = R1 = 0.5 * D
    if not equal:
        R0, R1 = 0.60 * D, 0.40 * D
    rho_l, sigma, g, mu_l = pair_params(D)
    rho_g, mu_g = rho_l / RATIO, mu_l / RATIO
    nx = ny = int(round(4 * D))
    nz = int(round(8 * D))
    zc0 = 1.8 * D
    zc1 = zc0 + R0 + R1 + gap
    s = pf.Solver(nx, ny, nz)
    s.set_rho(rho_l)
    s.set_mu(mu_l)
    s.set_domain_bc(4, 1, 0, 0, 0)
    s.set_domain_bc(5, 1, 0, 0, 0)
    s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
    s.set_pressure_chebyshev(True, 800, 1e-12)
    s.enable_vof()
    B0 = sphere_fractions((nx, ny, nz), R0, (nx / 2.0, ny / 2.0, zc0))
    B1 = sphere_fractions((nx, ny, nz), R1, (nx / 2.0, ny / 2.0, zc1))
    s.set_vof(np.asfortranarray(np.maximum(B0, B1)))
    s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
    s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
    s.set_surface_tension(sigma)
    s.set_property_model("force_z", "linear", "C", [-rho_l * g, -(rho_g - rho_l) * g])
    s.enable_vof_blocks([(nx / 2.0, ny / 2.0, zc0, R0), (nx / 2.0, ny / 2.0, zc1, R1)])
    s.set_block_csf_union(union)
    s.set_block_liquid(rho_l, mu_l)
    if coalescence:
        s.set_block_coalescence(**coalescence)
    s.enable_vof_block_csf()
    return s, (4 / 3 * math.pi * R0 ** 3, 4 / 3 * math.pi * R1 ** 3)


def run_pair(s, steps, label, every=50):
    h = Solve(800)
    film, maxc, t = [], 0.0, 0.0
    t0 = time.time()
    dt = 0.0
    for i in range(steps):
        if i % 5 == 0:
            L = s.vof_step_limits()
            dt = 0.2 * min(L["cfl_dt"], L["capillary_dt"])
            s.set_dt(dt)
        t += dt
        s.step()
        h.sample(s)
        pr = s.vof_block_pairs()
        if pr:
            film.append((t, pr[0]["film"], pr[0]["approach"], pr[0]["weber"]))
        maxc = max(maxc, float(s.get_vof().max()))
        if (i + 1) % every == 0:
            st = s.vof_block_stats()
            f = pr[0]["film"] if pr else float("nan")
            print(f"    step {i+1:5d}  t = {t:8.3f}  markers {len(st)}  film {f:7.3f}  "
                  f"maxC {maxc:.6f}  press {s.last_pressure_iterations():3d}/800  "
                  f"({time.time()-t0:.0f} s)")
    return dict(film=film, maxc=maxc, solve=h, t=t, wall=time.time() - t0)


# ------------------------------------------------------------------------------ G3: no merge
def gate_pair():
    print("\n" + "=" * 96)
    print("GATE G3 — two equal bubbles in line at Eo = 10, coalescence 'never'")
    print("=" * 96)
    steps = 200 if QUICK else 800
    s, v0 = build_pair(union=True, gap=2.0, coalescence=dict(model="never"))
    r = run_pair(s, steps, "never")
    st = s.vof_block_stats()
    vols = [b["volume"] for b in st]
    drift = [abs(v / q - 1.0) for v, q in zip(vols, v0)]
    fmin = min(f[1] for f in r["film"]) if r["film"] else float("nan")
    print(f"\n  markers at the end: {len(st)}  volumes {['%.6f' % v for v in vols]} "
          f"(seed {['%.6f' % v for v in v0]})")
    print(f"  volume drift {['%.2e' % d for d in drift]}   (gate 1e-12)")
    print(f"  film: start {r['film'][0][1]:.3f} -> min {fmin:.3f} -> end "
          f"{r['film'][-1][1]:.3f} cells;  max union C {r['maxc']:.12f}")
    print(f"  overlap faces at the end: {s.vof_block_overlap_faces()}")
    print(f"  events: {[ (e['type'], e['step']) for e in s.vof_block_events() ]}")
    ok = (len(st) == 2) and max(drift) <= 1e-12 and r["maxc"] <= 1.0 + 1e-12 and fmin <= 1.0
    print(f"\n  G3 -> {'PASS' if ok else 'FAIL'}  (two markers, volumes to 1e-12, union C <= 1, "
          f"film reaches one cell)")


# --------------------------------------------------------------------------------- G4: merge
def gate_merge():
    print("\n" + "=" * 96)
    print("GATE G4 — the same pair with the 'weber' model at a permissive We_crit")
    print("=" * 96)
    steps = 200 if QUICK else 800
    D = 10.0 if QUICK else 12.0
    s, v0 = build_pair(union=True, gap=2.0,
                       coalescence=dict(model="weber", we_crit=1e9, contact_film=1.0))
    r = run_pair(s, steps, "weber")
    st = s.vof_block_stats()
    ev = [e for e in s.vof_block_events() if e["type"] == "merge"]
    print(f"\n  markers at the end: {len(st)} (was 2);  merges {len(ev)}")
    ok = False
    if ev:
        e = ev[0]
        vsum = e["volume_a"] + e["volume_b"]
        print(f"  merge at step {e['step']}, t = {e['time']:.4f}: film {e['film']:.3f} cells, "
              f"We {e['weber']:.4f}, D_eq {e['d_eq']:.3f}")
        shared = s.vof_block_topology()["shared_liquid"]
        print(f"  volumes {e['volume_a']:.6f} + {e['volume_b']:.6f} = {vsum:.6f} -> merged "
              f"{e['volume_new']:.6f};  shared liquid the union max did not carry {shared:.6e}")
        print(f"  volume balance |V_a + V_b - shared - V_merged| = "
              f"{abs(vsum - shared - e['volume_new']):.3e}   (gate: round-off)")
        # the merged marker's rise velocity against Grace for the DOUBLED volume
        rho_l, sigma, g, mu_l = pair_params(D)
        Dm = 2.0 * (3.0 * e["volume_new"] / (4.0 * math.pi)) ** (1.0 / 3.0)
        Ut = grace_terminal(EO * (Dm / D) ** 2, MO, Dm, rho_l, mu_l, sigma, g)
        w = st[0]["velocity"][2] if st else float("nan")
        print(f"  merged D_eq {Dm:.3f} cells;  rise velocity {w:.5f} vs Grace {Ut:.5f} "
              f"({100*(w/Ut-1):+.1f} %)")
        ok = len(st) == 1 and abs(vsum - shared - e["volume_new"]) <= 1e-9
    print(f"\n  G4 -> {'PASS' if ok else 'FAIL'}  (one marker, merged volume = the sum minus the "
          f"shared liquid, to round-off)")


# --------------------------------------------------------------------------------- G5: breakup
def gate_split():
    print("\n" + "=" * 96)
    print("GATE G5 — an elongated marker (capsule, aspect 4) at Eo = 40 breaks up")
    print("=" * 96)
    D = 10.0 if QUICK else 12.0
    Eo = 40.0
    rho_l, sigma = 1.0, 1.0
    g = Eo * sigma / (rho_l * D * D)
    mu_l = (MO * rho_l ** 2 * sigma ** 3 / g) ** 0.25
    rho_g, mu_g = rho_l / RATIO, mu_l / RATIO
    R = 0.25 * D                          # capsule radius; length 4 * (2R) -> aspect 4
    half = 4.0 * R
    nx = ny = int(round(5 * D))
    nz = int(round(8 * D))
    steps = 150 if QUICK else 600
    zc = 3.0 * D

    def capsule(shape):
        C = np.zeros(shape)
        for zz in np.linspace(-half, half, 25):
            C = np.maximum(C, sphere_fractions(shape, R, (nx / 2.0, ny / 2.0, zc + zz)))
        return np.asfortranarray(C)

    out = {}
    for name, blk in (("blocks", True), ("single field (control)", False)):
        s = pf.Solver(nx, ny, nz)
        s.set_rho(rho_l)
        s.set_mu(mu_l)
        s.set_domain_bc(4, 1, 0, 0, 0)
        s.set_domain_bc(5, 1, 0, 0, 0)
        s.set_pressure_geometry(np.full((nx, ny, nz), 10.0, order="F"))
        s.set_pressure_chebyshev(True, 800, 1e-12)
        s.enable_vof()
        B = capsule((nx, ny, nz))
        s.set_vof(B)
        s.set_property_model("rho", "linear", "C", [rho_l, rho_g - rho_l])
        s.set_property_model("mu", "linear", "C", [mu_l, mu_g - mu_l])
        s.set_surface_tension(sigma)
        s.set_property_model("force_z", "linear", "C", [-rho_l * g, -(rho_g - rho_l) * g])
        if blk:
            s.enable_vof_blocks_from_field([marker_box(B)])
            s.set_block_liquid(rho_l, mu_l)
            s.set_block_breakup(True, n_split=3)
            s.enable_vof_block_csf()
        h = Solve(800)
        t, dt, t0 = 0.0, 0.0, time.time()
        v0 = float(B.sum())
        for i in range(steps):
            if i % 5 == 0:
                L = s.vof_step_limits()
                dt = 0.2 * min(L["cfl_dt"], L["capillary_dt"])
                s.set_dt(dt)
            t += dt
            s.step()
            h.sample(s)
            if (i + 1) % 50 == 0:
                extra = ""
                if blk:
                    extra = f"  markers {len(s.vof_block_stats())}"
                print(f"    step {i+1:5d}  t = {t:8.3f}  blobs {axis_blobs(s.get_vof())}{extra}  "
                      f"({time.time()-t0:.0f} s)")
        C = s.get_vof()
        info = dict(blobs=axis_blobs(C), union=float(C.sum()), v0=v0, wall=time.time() - t0)
        if blk:
            st = s.vof_block_stats()
            info["vols"] = sorted([b["volume"] for b in st], reverse=True)
            info["events"] = [e for e in s.vof_block_events() if e["type"] == "split"]
            info["topology"] = s.vof_block_topology()
        else:
            info["vols"] = blob_volumes(C)
        out[name] = info
        print(f"\n  -- {name}: {info['blobs']} blob(s) on the axis, union {info['union']:.4f} "
              f"(seed {v0:.4f});  volumes {['%.4f' % v for v in info['vols']]}  "
              f"({info['wall']:.0f} s)")
        if blk:
            print(f"     splits: {info['events']};  topology {info['topology']}")
    b, c = out["blocks"], out["single field (control)"]
    ok = False
    if b["events"]:
        vsum = sum(b["vols"])
        parent = b["events"][0]["volume_a"]
        print(f"\n  children sum {vsum:.8f} against the parent at the split {parent:.8f} "
              f"(relative {abs(vsum/parent - 1):.3e})")
        if len(b["vols"]) == len(c["vols"]) and c["vols"]:
            d = [abs(x / y - 1) for x, y in zip(b["vols"], sorted(c["vols"], reverse=True))]
            print(f"  child volumes vs the single-field control: {['%.2e' % q for q in d]}")
        ok = abs(vsum / parent - 1) <= 1e-12
    print(f"\n  G5 -> {'PASS' if ok else 'FAIL'}  (the split is reported and the children's "
          f"volumes sum to the parent's to 1e-12)")


def axis_blobs(C):
    col = C[C.shape[0] // 2, C.shape[1] // 2, :] > 0.5
    n, prev = 0, False
    for v in col:
        if v and not prev:
            n += 1
        prev = v
    return n


def blob_volumes(C):
    """Connected components of C > 0.5 in the single-field control (6-connectivity, host)."""
    try:
        from scipy import ndimage
    except ImportError:
        return []
    lab, n = ndimage.label(C > 0.5)
    return sorted([float(C[lab == k].sum()) for k in range(1, n + 1)], reverse=True)


ALL = {"bitwise": gate_bitwise, "pair": gate_pair, "merge": gate_merge, "split": gate_split}

if __name__ == "__main__":
    for g in (GATES or list(ALL)):
        if g not in ALL:
            raise SystemExit(f"unknown gate {g!r}; choose from {list(ALL)}")
        ALL[g]()
