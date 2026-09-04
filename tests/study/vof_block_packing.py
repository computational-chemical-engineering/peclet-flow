#!/usr/bin/env python3
"""WO-V9 item 4 (second half) — the block gather/scatter packing, device vs host, on a QUIET GPU.

WO-W12 measured 2.731 ms (device) vs 4.626 ms (host) = 1.69x on a comparatively quiet GPU and
79.4 vs 63.4 ms = 0.80x with five other jobs on it, and asked for the number to be re-taken on an
idle machine.  This is that measurement: the 64-marker swarm of the W12 `swarm` gate, kinematic
block advection (`advect_vof_blocks`), device staging on and off, alternated so a drift in machine
state shows up as scatter rather than as a result.

    PYTHONPATH=$PWD/build_cuda python tests/study/vof_block_packing.py
"""
import argparse
import time

import numpy as np

import peclet.flow as flow


def build(n, seeds, device, pool):
    s = flow.Solver(n, n, n)
    s.set_rho(1.0)
    s.set_mu(0.01)
    s.set_pressure_geometry(np.full((n, n, n), 10.0, order="F"))
    s.enable_vof()
    s.set_vof(np.zeros((n, n, n), order="F"))
    # a solenoidal cellular field, so the kinematic advection is admissible
    s.set_dt(0.05)
    s.enable_vof_blocks(seeds)
    s.set_vof_block_device_staging(device)
    s.set_vof_block_pool(pool)
    return s


def seeds_64(n):
    out, k = [], 0
    for i in range(4):
        for j in range(4):
            for m in range(4):
                out.append(((i + 0.5) * n / 4, (j + 0.5) * n / 4, (m + 0.5) * n / 4,
                            2.0 + 7.0 * ((k % 8) / 7.0)))
                k += 1
    return out


def run(n, seeds, device, pool, steps, warm):
    s = build(n, seeds, device, pool)
    for _ in range(warm):
        s.advect_vof_blocks(0.0)
    t0 = time.time()
    for _ in range(steps):
        s.advect_vof_blocks(0.0)
    dt = (time.time() - t0) / steps
    st = s.vof_block_stats()
    vol = sum(b["volume"] for b in st)
    del s
    return dt, vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    seeds = seeds_64(a.n)
    print(f"64-marker swarm, {a.n}^3, kinematic block advection (dt = 0, so the timing is the "
          f"GATHER/SCATTER + the sweeps and nothing else)")
    rows = {True: [], False: []}
    vols = {}
    for r in range(a.reps):
        for dev in (True, False):
            t, v = run(a.n, seeds, dev, True, a.steps, a.warm)
            rows[dev].append(t)
            vols[dev] = v
    for dev in (True, False):
        ts = np.array(rows[dev]) * 1e3
        print(f"  {'device' if dev else 'host  '} staging: "
              f"{ts.min():8.3f} ms/step (min of {a.reps})   {ts.mean():8.3f} +- {ts.std():.3f}")
    d, h = min(rows[True]), min(rows[False])
    print(f"  device/host speedup: {h/d:.3f}x")
    print(f"  total marker volume device {vols[True]:.6f}  host {vols[False]:.6f}  "
          f"|d| {abs(vols[True]-vols[False]):.3e}")


if __name__ == "__main__":
    main()
