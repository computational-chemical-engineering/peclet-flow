#!/usr/bin/env python3
"""Does the pressure in SEALED fluid pockets of a dense bed change any physical output? A study
instrument (not a ctest): it prints one line per (output, perturbation).

A sealed pocket is a fluid component of the projection's face-aperture graph other than the main
pore space -- nearly always one cell at a sphere near-contact, every face aperture exactly zero. Its
pressure is a null vector of the pressure operator, uncoupled from the velocity constraint, so its
value is whatever the projection writes into the decoupled row.

Method: build the bed as an analytic SCENE (one instance per sphere, so the per-body reaction and
traction loads exist), configure it exactly as tests/study/steady_acceleration_gates.py does
(Case.configure, production settings), march to steady, snapshot u, v, w, p (the G5 'field'
restore; the run is Stokes, so that is the whole state), then for each perturbation
p -> p + c * [pocket mask] (one pocket, all pockets; c = 0 is the baseline, run twice for the
noise floor):
  instant  -- outputs recomputed with no step (the reaction reads the stashed u* of the last step);
  stepped  -- the same after ONE step from the perturbed state.
The outputs: per-body reaction force / torque (staggered only), per-body traction force / torque /
pressure part, the bed totals and K from <u_x>, the velocity fields, and the pressure a user reads
(the fluid mean, x-plane means, main-space values, the pockets themselves).

  PYTHONPATH=build_cuda OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
      python tests/study/pocket_pressure_probe.py --arrangement bed.npz --scheme staggered

Face convention (checked by the instrument itself): get_ox[i] is the face between cells i-1 and i,
as the binding says. anderson_rev1_probes.py reads it as the face (i, i+1); with that reading this
bed shows 1,085 pockets / 1,168 cells, with the right one 205 / 217 (sdf > 0 nodes) or 115 / 126
(--graph dof, the staggered cut-cell operator's own graph, which also joins through solid-CENTRED
cut cells).

The raw field registry (get_field / set_field / diagnostics.field_view) is used here as a study
instrument only; production code must not. The perturbation c is therefore in the registry's
internal pressure units; the reported pressure changes are scaled by the physical (get_p) range.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steady_acceleration_gates as SAG  # noqa: E402  (Case, settings_of: reused, not re-derived)


def case_args(a):
    """The namespace steady_acceleration_gates.Case and Case.configure read."""
    return argparse.Namespace(case="bed", arrangement=a.arrangement, mu=1.0, N=a.N, phi=0.6,
                              force=None, dt=None, beta=a.beta, advection=None,
                              implicit_advection=False)


def set_bed_scene(s, case):
    """One sphere node + one instance per body, periodic min-image, physical coordinates."""
    n = case.npart
    ni = np.tile(np.array([1, -1, -1], dtype=np.int32), n)
    nr = np.zeros((n, 16))
    nr[:, 0] = case.rad
    nr[:, 14] = 1.0  # rotation w
    nr[:, 15] = 1.0  # scale
    ii = np.stack([np.arange(n, dtype=np.int32), -np.ones(n, dtype=np.int32)], axis=1).ravel()
    ir = np.zeros((n, 17))
    ir[:, 0:3] = np.mod(case.cen, case.L)
    ir[:, 6] = 1.0  # rotation w
    ir[:, 7] = 1.0  # scale
    s.set_scene(ni, nr.ravel(), np.ascontiguousarray(ii), ir.ravel(), periodic=True)
    s.set_solid_from_scene(cutcell_pressure=True)


def build(flow, case, scheme, st, scene=True):
    Cls = flow.Solver if scheme == "staggered" else flow.SolverColocated
    N, L = case.N, case.L
    s = Cls((N, N, N), extent=(L, L, L))
    case.configure(s, st)
    if scene:
        set_bed_scene(s, case)
    else:
        cx, cy, cz = s.cell_centers()
        s.set_solid(np.asfortranarray(case.sdf_global(cx, cy, cz)), cutcell_pressure=True)
    return s


def inner(v, G=2):
    return np.array(v[G:-G, G:-G, G:-G])


def components(s, fluid, proj=True, graph="sdf"):
    """Components of the face-aperture graph (ox at index i couples cells i-1 and i; as in
    anderson_rev1_probes.py, rolled so that index i couples i and i+1).

    graph='sdf' -- anderson_rev1_probes.py's definition: nodes are the sdf > 0 cells, an edge needs
                   BOTH cells sdf > 0. A solid-CENTRED cut cell (sdf <= 0, open faces) is not a
                   node, so a fluid cell joined to the main space only through one looks sealed.
    graph='dof'  -- the pressure operator's own graph: every open face is an edge whatever the sdf
                   sign of its cells; nodes are the cells with an open face plus the sdf > 0 cells.
    Returns (labels over the grid, main label, component sizes, apertures, node mask)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = fluid.shape
    idx = np.arange(fluid.size).reshape(n, order="F")
    get = (s.get_ox_proj, s.get_oy_proj, s.get_oz_proj) if proj else (s.get_ox, s.get_oy, s.get_oz)
    opn = [np.roll(np.asarray(g()).reshape(n, order="F"), -1, axis=ax) for ax, g in enumerate(get)]
    # opn[ax][i] is now the +ax face of cell i, i.e. the face between i and i+1
    if graph == "sdf":
        node = fluid.copy()
    else:
        node = fluid.copy()
        for ax in range(3):
            o = opn[ax] > 0.0
            node |= o | np.roll(o, 1, axis=ax)
    Ii, Jj = [], []
    for ax in range(3):
        m = (opn[ax] > 0.0) & node & np.roll(node, -1, axis=ax)
        Ii.append(idx[m])
        Jj.append(np.roll(idx, -1, axis=ax)[m])
    Ii, Jj = np.concatenate(Ii), np.concatenate(Jj)
    A = coo_matrix((np.ones(Ii.size), (Ii, Jj)), shape=(fluid.size, fluid.size))
    _, lab = connected_components(A, directed=False)
    lab = lab.reshape(n, order="F")
    sizes = np.bincount(lab[node], minlength=lab.max() + 1)
    main = int(np.argmax(sizes))
    return lab, main, sizes, opn, node


def outputs(s, case, scheme, fluid):
    o = {}
    if scheme == "staggered":
        r = np.array(s.hydro_force_torque_reaction())
        o["reaction F"] = r[0]
        o["reaction T"] = r[1]
        o["reaction sum Fx"] = np.array([r[0][:, 0].sum()])
    t = np.array(s.diagnostics.hydro_force_torque_traction())
    o["traction F"] = t[0]
    o["traction T"] = t[1]
    o["traction F_p"] = t[2]
    o["traction sum Fx"] = np.array([t[0][:, 0].sum()])
    o["u"] = np.array(s.get_u())
    o["v"] = np.array(s.get_v())
    o["w"] = np.array(s.get_w())
    um = float(o["u"].mean())
    o["<u_x>"] = np.array([um])
    o["K"] = np.array([case.K(um)])
    p = np.array(s.get_p())
    o["p"] = p
    o["<p> fluid"] = np.array([p[fluid].mean()])
    # x-plane fluid means (a user's pressure profile / drop estimate)
    cnt = fluid.sum(axis=(1, 2))
    o["<p>_yz(x)"] = (np.where(fluid, p, 0.0).sum(axis=(1, 2)) / np.maximum(cnt, 1))
    return o


def rel(a, b, scale):
    return float(np.max(np.abs(a - b))) / scale if scale > 0 else float(np.max(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrangement", required=True)
    ap.add_argument("--scheme", choices=("staggered", "collocated"), default="staggered")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--settings", choices=("production", "tight"), default="production")
    ap.add_argument("--window", type=int, default=5, help="march_to_steady window (0 = plain)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=200,
                    help="record the pocket-pressure spread every this many march steps")
    ap.add_argument("--graph", choices=("sdf", "dof"), default="dof",
                    help="pocket definition (see components()); 'sdf' = anderson_rev1_probes.py's")
    ap.add_argument("--per-pocket", action="store_true",
                    help="perturb every pocket on its own (c = 1000 x main range), one step each")
    ap.add_argument("--check-sdf", action="store_true",
                    help="also build the harness's set_solid(array) bed and compare apertures")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    from _bootstrap import ensure_flow
    flow = ensure_flow()

    st = SAG.settings_of(a.settings)
    if a.max_steps:
        st["max_steps"] = a.max_steps
    case = SAG.Case(case_args(a))
    s = build(flow, case, a.scheme, st)
    fluid = np.array(s.get_field("sdf")) > 0.0
    lab, main, sizes, opn, node = components(s, fluid, proj=True, graph=a.graph)
    labg, maing, sizesg, opng, nodeg = components(s, fluid, proj=False, graph=a.graph)
    pk = node & (lab != main)
    ids = np.unique(lab[pk])
    print(f"[{a.scheme}] N={a.N} bodies={case.npart} L={case.L:.6f} graph={a.graph}: fluid (sdf>0) "
          f"{fluid.sum()} cells, graph nodes {node.sum()}; projection-aperture graph "
          f"{len(np.unique(lab[node]))} components, main {sizes[main]}, {len(ids)} pockets holding "
          f"{pk.sum()} cells (of which sdf>0: {(pk & fluid).sum()}; pocket sizes "
          f"{dict((int(k), int(v)) for k, v in zip(*np.unique(sizes[ids], return_counts=True)))}); "
          f"geometric-aperture graph {len(np.unique(labg[nodeg]))} components; "
          f"max|o_proj - o_geom| {max(float(np.abs(x - y).max()) for x, y in zip(opn, opng)):.3e}",
          flush=True)
    # the other definition, for the record: how many of its pockets are pockets here
    lab2, main2, _, _, node2 = components(s, fluid, proj=True,
                                          graph=("dof" if a.graph == "sdf" else "sdf"))
    pk2 = node2 & (lab2 != main2)
    print(f"  other graph ({'dof' if a.graph == 'sdf' else 'sdf'}): {len(np.unique(lab2[pk2]))} "
          f"pockets, {pk2.sum()} cells; cells pocket in both {(pk & pk2).sum()}", flush=True)
    # face convention check: an open face between two cells that are both deeper than one cell
    # inside a solid is impossible, so count them under each reading of get_ox[i]
    sdfv = np.array(s.get_field("sdf"))
    hh = float(np.abs(np.diff(sdfv, axis=0)).max())  # max |sdf step| along x = one cell (|grad|=1)
    # a CUT face (0 < aperture < 1) has the wall crossing it, so the sdf at the face midpoint,
    # ~ the mean of its two cells' sdf, is within ~0.71 h of zero; read with the wrong neighbour it
    # is up to ~1.7 h away
    raw = [np.asarray(g()) for g in (s.get_ox_proj, s.get_oy_proj, s.get_oz_proj)]
    for nm, sh in (("(i-1, i)", 1), ("(i, i+1)", -1)):
        cut = sum(int(((r > 0) & (r < 1)).sum()) for r in raw)
        if cut == 0:
            print("  face convention: no cut faces (binary openness)", flush=True)
            break
        mids = np.concatenate([np.abs(0.5 * (sdfv + np.roll(sdfv, sh, axis=ax)))[(r > 0) & (r < 1)]
                               for ax, r in enumerate(raw)]) / hh
        print(f"  face convention: get_ox[i] read as the face {nm}: |mid-face sdf| / h over cut "
              f"faces: p99 {np.percentile(mids, 99):.3f}, max {mids.max():.3f}", flush=True)
    anyo = np.zeros_like(fluid)
    for ax in range(3):
        o = opn[ax] > 0.0
        anyo |= o | np.roll(o, 1, axis=ax)
    print(f"  solid-centred cells (sdf<=0) with an open projection face: {(anyo & ~fluid).sum()}; "
          f"sdf>0 cells with all six faces closed: {(fluid & ~anyo).sum()}", flush=True)
    if a.check_sdf:
        s2 = build(flow, case, a.scheme, st, scene=False)
        g2 = [np.roll(np.asarray(g()), -1, axis=ax)
              for ax, g in enumerate((s2.get_ox_proj, s2.get_oy_proj, s2.get_oz_proj))]
        f2 = np.array(s2.get_field("sdf")) > 0.0
        print(f"  scene vs harness set_solid(array): max|d o_proj| "
              f"{max(float(np.abs(x - y).max()) for x, y in zip(opn, g2)):.3e}, fluid-mask "
              f"mismatches {(f2 != fluid).sum()}", flush=True)
        del s2

    # ---- march, sampling the pocket pressure spread
    trace = []

    def cb(steps, phase):
        if steps % a.sample_every == 0:
            p = np.array(s.get_p())
            pm = p[fluid & ~pk]
            pp = p[pk]
            trace.append((steps, phase, float(pm.min()), float(pm.max()), float(pp.min()),
                          float(pp.max()), float(pp.std())))

    t0 = time.perf_counter()
    res = flow.march_to_steady(s, lambda: float(s.get_u().mean()), rtol=st["rtol"],
                               max_steps=st["max_steps"], accelerate=(a.window > 0),
                               window=max(a.window, 1), callback=cb)
    print(f"  march: converged={res.converged} {res.reason} steps={res.steps} "
          f"acc={res.accelerated_steps} K={case.K(res.monitor):.8f} "
          f"wall={time.perf_counter() - t0:.1f}s", flush=True)
    for t in trace:
        print(f"    step {t[0]:5d} {t[1]:10s} main p [{t[2]:+.4e}, {t[3]:+.4e}]  pockets "
              f"[{t[4]:+.4e}, {t[5]:+.4e}] std {t[6]:.4e}", flush=True)

    S0 = {f: np.asfortranarray(np.array(s.get_field(f))) for f in ("u", "v", "w", "p")}
    p0 = S0["p"]
    pm = p0[fluid & ~pk]
    dmain = float(pm.max() - pm.min())  # INTERNAL units (the raw registry): the perturbation scale
    pphys = np.array(s.get_p())[fluid & ~pk]
    dmain_phys = float(pphys.max() - pphys.min())  # physical (get_p): the scale of the p outputs
    pp = p0[pk]
    print(f"  converged pressure: main space range {dmain:.6e} (mean {pm.mean():+.4e}); pockets "
          f"min {pp.min():+.4e} max {pp.max():+.4e} mean {pp.mean():+.4e} std {pp.std():.4e} "
          f"-> pocket spread / main range = {(pp.max() - pp.min()) / dmain:.3e}; "
          f"max|p_pocket - <p_main>| / main range = {np.abs(pp - pm.mean()).max() / dmain:.3e}",
          flush=True)
    # solid cells' pressure too (also null rows)
    ps = p0[~fluid & ~anyo]
    print(f"  solid (closed) cells: p min {ps.min():+.4e} max {ps.max():+.4e}", flush=True)

    # the single pocket to perturb: the largest pocket (multi-cell, if any) and one single cell
    big = int(ids[np.argmax(sizes[ids])])
    single = int(ids[np.argmin(sizes[ids])])
    masks = {"one pocket (largest, %d cells)" % sizes[big]: lab == big,
             "one pocket (single cell)": lab == single,
             "all pockets": pk}
    # owners of the perturbed pocket cells (nearest sphere centre, min-image)
    for name, m in masks.items():
        if "all" in name:
            continue
        cx, cy, cz = s.cell_centers()
        X = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), axis=-1)[m]
        d = X[:, None, :] - np.mod(case.cen, case.L)[None]
        d -= case.L * np.round(d / case.L)
        r = np.linalg.norm(d, axis=-1) - case.rad[None]
        near = np.argsort(r, axis=1)[:, :3]
        print(f"  {name}: cells {np.argwhere(m).tolist()}, nearest bodies "
              f"{near.tolist()} gaps {np.take_along_axis(r, near, 1).round(4).tolist()}", flush=True)

    def restore(c=0.0, mask=None):
        for f in ("u", "v", "w", "p"):
            s.set_field(f, S0[f] if (f != "p" or c == 0.0) else np.asfortranarray(p0 + c * mask))
            s.diagnostics.exchange_field(f)

    def run(mask, c):
        """(instant change against the unperturbed state under the SAME stash, outputs after one
        step from the perturbed state)."""
        restore()
        o0 = outputs(s, case, a.scheme, fluid)
        restore(c, mask)
        oi = outputs(s, case, a.scheme, fluid)
        s.step()
        return o0, oi, outputs(s, case, a.scheme, fluid)

    _, _, base_s = run(pk, 0.0)
    _, _, base2_s = run(pk, 0.0)
    keys = list(base_s.keys())

    def scale_of(k, o):
        if k in ("u", "v", "w"):
            return float(max(np.abs(o["u"]).max(), np.abs(o["v"]).max(), np.abs(o["w"]).max()))
        if k in ("p", "<p> fluid", "<p>_yz(x)"):
            return dmain_phys
        return float(np.max(np.abs(o[k])))

    def report(o0, oi, os_, mask, cfac=0.0):
        for k in keys:
            ri = rel(oi[k], o0[k], scale_of(k, o0)) if o0 is not None else 0.0
            rs = rel(os_[k], base_s[k], scale_of(k, base_s))
            print(f"    {k:18s} instant {ri:.3e}   after 1 step {rs:.3e}", flush=True)
        # the pressure split: main space vs pockets, and what survives the step in the pockets
        for tag, o, b in (("instant", oi, o0), ("step", os_, base_s)):
            if b is None:
                continue
            d = o["p"] - b["p"]
            print(f"    p main space {tag:7s}: max|dp| / main range "
                  f"{np.abs(d[fluid & ~pk]).max() / dmain_phys:.3e} (mean shift "
                  f"{d[fluid & ~pk].mean() / dmain_phys:+.3e}); perturbed pocket(s) mean dp / c "
                  f"{d[mask].mean() / dmain_phys / (cfac if cfac else 1.0):+.4e}", flush=True)

    print("  noise floor (baseline step twice):", flush=True)
    report(None, base2_s, base2_s, pk)
    for name, m in masks.items():
        for c in (dmain, 1e3 * dmain):
            print(f"  perturbation: {name}, c = {c:.4e} ({c / dmain:g} x main range)", flush=True)
            o0, oi, os_ = run(m.astype(float), c)
            report(o0, oi, os_, m, c / dmain)
    if a.per_pocket:
        c = 1e3 * dmain
        rows = []
        for pid in ids:
            m = lab == pid
            o0, oi, os_ = run(m.astype(float), c)
            du = [rel(os_[k], base_s[k], scale_of(k, base_s)) for k in ("u", "v", "w")]
            rr = (rel(os_["reaction F"], base_s["reaction F"], scale_of("reaction F", base_s))
                  if "reaction F" in os_ else float("nan"))
            rows.append((max(du), int(pid), rr, rel(os_["K"], base_s["K"], scale_of("K", base_s)),
                         rel(oi["traction F"], o0["traction F"], scale_of("traction F", o0))))
        rows.sort(reverse=True)
        dus = np.array([r[0] for r in rows])
        floor = float(np.median(dus))
        print(f"  per pocket (c = 1000 x main range, one step): median max|du|/max|u| {floor:.3e}; "
              f"pockets above 10 x median: {int((dus > 10 * floor).sum())} of {len(rows)}; "
              f"instant traction change > 1e-12 in {sum(r[4] > 1e-12 for r in rows)} pockets",
              flush=True)
        uu = [S0[f] for f in ("u", "v", "w")]
        for du, pid, rr, dk, dtr in rows[:8]:
            cells = np.argwhere(lab == pid)
            print(f"    pocket {pid}: max|du| {du:.3e}  reaction F {rr:.3e}  K {dk:.3e}  "
                  f"instant traction F {dtr:.3e}  cells {cells.tolist()}", flush=True)
            for cell in cells:
                i = tuple(cell)
                for ax in range(3):
                    hi = list(i)
                    hi[ax] = (hi[ax] + 1) % case.N
                    lo2 = list(i)
                    lo2[ax] = (lo2[ax] - 1) % case.N
                    print(f"      cell {i} sdf {sdfv[i] / hh:+.3f} h  axis {'xyz'[ax]}: "
                          f"-face aperture {opn[ax][tuple(lo2)]:.3g} (nbr sdf "
                          f"{sdfv[tuple(lo2)] / hh:+.3f} h, u_face {uu[ax][i]:+.3e}); "
                          f"+face aperture {opn[ax][i]:.3g} (nbr sdf {sdfv[tuple(hi)] / hh:+.3f} h, "
                          f"u_face {uu[ax][tuple(hi)]:+.3e})", flush=True)
    restore()


if __name__ == "__main__":
    main()
