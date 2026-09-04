#!/usr/bin/env python3
"""Convert one TBFsolver output folder's statistics into the same NPZ `run_channel_18.py` writes.

  python tbf_profiles.py <TBFsolver output folder> [--ny 160] [--out tbf_profiles.npz]

TBFsolver's statistics (`src/statistics/statistics.f90`) are TIME averages of PLANE integrals,
divided by the slice volume — so for the region `R` (`_W` whole, `_G` gas, `_L` liquid, with the
weight `cv` = 1, `c`, `1 − c` and `c` the GAS fraction, the same convention as our transcription):

    stats_c          <c>(y)                                       1 column
    stats_u<R>       <cv ux>, <cv uy>, <cv uz>,                    9 columns
                     <cv uxux>, <cv uxuy>, <cv uxuz>,
                     <cv uyuy>, <cv uyuz>, <cv uzuz>
    tauw             the time-averaged wall shear stress

They are NOT conditioned averages: the liquid mean velocity is `<(1−c)u>/<1−c>`, which is what this
script forms, exactly as `run_channel_18.py` does with its own `L`-weighted sums.  Both codes'
NPZ therefore carry the same quantities under the same names and the plotting script can overlay
them without further arithmetic.
"""
import os
import sys

import numpy as np

MU = 3.33333333333333e-4
RET = 127.3
UTAU = RET * MU
LY = 2.0


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def col(path, ncol):
    a = np.loadtxt(path)
    a = a.reshape(-1, ncol) if a.ndim == 1 else a
    return a


def main():
    d = sys.argv[1]
    ny = int(arg("--ny", 160))
    out = arg("--out", "tbf_profiles.npz")
    y = (np.arange(ny) + 0.5) * LY / ny

    c = col(os.path.join(d, "stats_c"), 1)[:, 0]
    uW = col(os.path.join(d, "stats_u_W"), 9)
    uL = col(os.path.join(d, "stats_u_L"), 9)

    liq = np.maximum(1.0 - c, 1e-30)
    r = {}
    r["y"] = y
    r["yplus"] = y / (MU / UTAU)
    r["alpha"] = c
    r["liquid_fraction"] = 1.0 - c
    r["u"], r["v"], r["w"] = uW[:, 0], uW[:, 1], uW[:, 2]
    r["uu"], r["uv"] = uW[:, 3], uW[:, 4]
    r["urms"] = np.sqrt(np.maximum(uW[:, 3] - uW[:, 0] ** 2, 0.0))
    r["vrms"] = np.sqrt(np.maximum(uW[:, 6] - uW[:, 1] ** 2, 0.0))
    r["wrms"] = np.sqrt(np.maximum(uW[:, 8] - uW[:, 2] ** 2, 0.0))
    ul, vl, wl = uL[:, 0] / liq, uL[:, 1] / liq, uL[:, 2] / liq
    r["u_liq"], r["v_liq"], r["w_liq"] = ul, vl, wl
    r["urms_liq"] = np.sqrt(np.maximum(uL[:, 3] / liq - ul ** 2, 0.0))
    r["vrms_liq"] = np.sqrt(np.maximum(uL[:, 6] / liq - vl ** 2, 0.0))
    r["wrms_liq"] = np.sqrt(np.maximum(uL[:, 8] / liq - wl ** 2, 0.0))
    r["uv_liq"] = uL[:, 4] / liq - ul * vl
    dy = LY / ny
    r["utau_lo"] = float(np.sqrt(MU * abs(ul[0]) / (0.5 * dy)))
    r["utau_hi"] = float(np.sqrt(MU * abs(ul[-1]) / (0.5 * dy)))
    r["utau_wall"] = 0.5 * (r["utau_lo"] + r["utau_hi"])
    r["utau"] = UTAU
    r["utau_imposed"] = UTAU
    tw = os.path.join(d, "tauw")
    if os.path.exists(tw):
        r["tauw_tav"] = float(np.loadtxt(tw))
    r["grid"] = np.asarray([192, ny, 96])
    np.savez(out, **{k: np.asarray(v) for k, v in r.items()})
    print(f"wrote {out} from {d}")
    print(f"  <alpha> centreline {c[ny//2]:.5f}, bulk {c.mean():.5f}")
    print(f"  <u>_liq/u_tau peak {np.max(ul)/UTAU:.3f} at y/h = {y[int(np.argmax(ul))]:.3f}, "
          f"centreline {ul[ny//2]/UTAU:.3f}")
    print(f"  u_tau wall gradient {r['utau_wall']:.6f} vs imposed {UTAU:.6f} "
          f"({100*(r['utau_wall']/UTAU-1):+.1f} %)")


if __name__ == "__main__":
    main()
