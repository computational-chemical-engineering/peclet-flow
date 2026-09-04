#!/usr/bin/env python3
"""Plot the WO-W3 `channel_18` profiles, and overlay TBFsolver's if a run of it exists.

  python plot_channel_18.py runs/prod/stats.npz [--tbf tbf_profiles.npz] [--out channel_18.png]

`stats.npz` is what `run_channel_18.py` writes.  A `--tbf` file, if given, is expected to carry the
same key names (`y`, `alpha`, `u_liq`, `urms_liq`, ...) so the two codes can be drawn on one axis;
it is written by `tbf_profiles.py` when TBFsolver has been run.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/prod/stats.npz"
    ref = arg("--tbf", None)
    out = arg("--out", "channel_18.png")
    z = np.load(src)
    utau = float(z["utau"])
    y = z["y"]
    ny = len(y)
    half = ny // 2
    # fold the two halves onto y/h in [0, 1] (the case is statistically symmetric)
    def fold(a, even=True):
        b = a[:half]
        c = a[::-1][:half]
        return 0.5 * (b + c) if even else 0.5 * (b - c)

    yf = y[:half]
    r = np.load(ref) if ref else None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(fold(z["u_liq"]) / utau, yf, "-o", ms=3, label="peclet (liquid)")
    ax[0].plot(fold(z["u"]) / utau, yf, "--", lw=1, label="peclet (all)")
    if r is not None:
        ax[0].plot(r["u_liq"][:half] / float(r["utau"]), r["y"][:half], "k-", label="TBFsolver")
    ax[0].set_xlabel(r"$\langle u\rangle/u_\tau$")
    ax[0].set_ylabel("$y/h$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(fold(z["alpha"]), yf, "-o", ms=3, label="peclet")
    if r is not None:
        ax[1].plot(r["alpha"][:half], r["y"][:half], "k-", label="TBFsolver")
    ax[1].set_xlabel(r"void fraction $\langle\alpha\rangle$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for k, lab in (("urms_liq", "$u'$"), ("vrms_liq", "$v'$"), ("wrms_liq", "$w'$")):
        ax[2].plot(fold(z[k]) / utau, yf, "-", label=lab + " peclet")
    ax[2].plot(-fold(z["uv_liq"], even=False) / utau ** 2, yf, ":", label=r"$-\langle u'v'\rangle$")
    ax[2].set_xlabel("wall units")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle(f"channel_18, block VoF — {float(z['turnovers']):.1f} eddy turnovers, "
                 f"{int(z['samples'])} samples, grid {tuple(z['grid'])}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
