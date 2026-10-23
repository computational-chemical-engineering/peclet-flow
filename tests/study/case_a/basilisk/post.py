"""Post-process a Basilisk case-A run (array.c) with run_case_a.py's definitions.

usage: python post.py RUN_DIR [L=1.6]
log.dat columns: t ub umix uzm xcent vg step (units d = g = rho_l = 1; mu_l from err.log).
Re_drift = (ub - umix)/mu_l (Loisy eq. 2.3; run_case_a holds umix = 0, Basilisk does not),
Re_zm = (ub - uzm)/mu_l, Re_abs = ub/mu_l (computational frame). Windows as run_case_a.py: A = the
last two box crossings of the gas centroid measured in the <u> = 0 frame (x_rel = xcent - int umix dt),
B = the two before; trapezoidal time means. Re_cent = x_rel displacement over window A / time.
Prints one summary line."""
import re
import sys

import numpy as np

d = sys.argv[1]
L = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
hdr = open(f"{d}/err.log").readline()
mu_l = float(re.search(r"mu2 (\S+)", hdr).group(1))
N = int(re.search(r"# N (\d+)", hdr).group(1))
H = np.loadtxt(f"{d}/log.dat")
t, ub, um, uz, xc = H[:, 0], H[:, 1], H[:, 2], H[:, 3], H[:, 4]
cum = np.concatenate(([0.0], np.cumsum(0.5 * (um[1:] + um[:-1]) * np.diff(t))))
xr = xc - cum
cols = {"drift": (ub - um) / mu_l, "zm": (ub - uz) / mu_l, "abs": ub / mu_l, "umix": um / mu_l}


def wmean(y, hi, lo):
    m = (xr > lo) & (xr <= hi)
    return float(np.trapezoid(y[m], t[m]) / (t[m][-1] - t[m][0])) if m.sum() >= 2 else float("nan")


xe = xr[-1]
A = {k: wmean(v, xe, xe - 2 * L) for k, v in cols.items()}
B = wmean(cols["drift"], xe - 2 * L, xe - 4 * L)
m = xr > xe - 2 * L
ta = t[m][0]
rc = (xe - xr[m][0]) / (t[-1] - ta) / mu_l
print(f"BASILISK N={N} d/h={N / L:g} t_end={t[-1]:.3f} steps={int(H[-1, 6])} Re_drift(2L)={A['drift']:.4f} "
      f"[min {cols['drift'][m].min():.3f} max {cols['drift'][m].max():.3f}] Re_cent={rc:.4f} "
      f"Re_zm={A['zm']:.4f} Re_abs={A['abs']:.4f} <u>/(nu/d)={A['umix']:.4f} "
      f"dRe/Re(2L vs prev 2L)={abs(A['drift'] - B) / A['drift']:.2e} window t*=[{ta:.2f},{t[-1]:.2f}] "
      f"vol_err={H[-1, 5] / H[0, 5] - 1:.1e}")
