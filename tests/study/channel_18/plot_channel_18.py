#!/usr/bin/env python3
"""Plot the WO-W3 `channel_18` profiles, and overlay TBFsolver's when a run of it exists.

  python plot_channel_18.py runs/prod/stats.npz [--tbf tbf_profiles.npz] [--out channel_18.png]

`stats.npz` is what `run_channel_18.py` writes; a `--tbf` file is what `tbf_profiles.py` writes
from a TBFsolver output folder — the same key names, so the two overlay directly.  The two codes
have DIFFERENT wall-normal resolutions (80 vs 160 rows across the channel), so each set is folded
about the centreline with its own row count, never with the other's.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def fold(z, key, even=True):
    """Average the two channel halves of one profile onto y/h in [0, 1], with z's own ny."""
    a = np.asarray(z[key])
    n = len(a) // 2
    b, c = a[:n], a[::-1][:n]
    return (0.5 * (b + c)) if even else (0.5 * (b - c)), np.asarray(z["y"])[:n]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/prod/stats.npz"
    ref = arg("--tbf", None)
    out = arg("--out", "channel_18.png")
    z = np.load(src)
    ut = float(z["utau"])
    r = np.load(ref) if ref else None
    rut = float(r["utau"]) if r is not None else 1.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    v, y = fold(z, "u_liq")
    ax[0].plot(v / ut, y, "-o", ms=3, label="peclet (liquid)")
    v, y = fold(z, "u")
    ax[0].plot(v / ut, y, "--", lw=1, label="peclet (all)")
    if r is not None:
        v, y2 = fold(r, "u_liq")
        ax[0].plot(v / rut, y2, "k-", label="TBFsolver (liquid)")
    ax[0].set_xlabel(r"$\langle u\rangle/u_\tau$")
    ax[0].set_ylabel("$y/h$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    v, y = fold(z, "alpha")
    ax[1].plot(v, y, "-o", ms=3, label="peclet")
    if r is not None:
        v, y2 = fold(r, "alpha")
        ax[1].plot(v, y2, "k-", label="TBFsolver")
    ax[1].set_xlabel(r"void fraction $\langle\alpha\rangle$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for k, lab, col in (("urms_liq", "$u'$", "C0"), ("vrms_liq", "$v'$", "C1"),
                        ("wrms_liq", "$w'$", "C2")):
        v, y = fold(z, k)
        ax[2].plot(v / ut, y, "-", color=col, label=lab + " peclet")
        if r is not None:
            v, y2 = fold(r, k)
            ax[2].plot(v / rut, y2, "--", color=col, label=lab + " TBF")
    ax[2].set_xlabel(r"rms$/u_\tau$ (liquid)")
    ax[2].legend(fontsize=7, ncol=2)
    ax[2].grid(alpha=0.3)

    g = tuple(int(q) for q in np.asarray(z["grid"]))
    w = np.asarray(z["window_turnovers"]) if "window_turnovers" in z.files else None
    win = f"window $t u_\\tau/h$ {w[0]:.2f}–{w[1]:.2f}, " if w is not None else ""
    fig.suptitle(f"channel_18 — peclet block VoF {g[0]}x{g[1]}x{g[2]}"
                 + (" vs TBFsolver 192x160x96" if r is not None else "")
                 + f" — {win}{int(z['samples'])} samples")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
