#!/usr/bin/env python3
"""Turn a `run_channel_18.py` CHECKPOINT into the profiles NPZ, without waiting for a chunk to end.

The running sums live in the checkpoint (`acc_*`), so the campaign can be read at any moment —
which is the point: `stats.npz` is only written when a chunk finishes, and a chunk is hours.

  python profiles_from_ckpt.py runs/prod/ckpt.npz [--out stats_now.npz]

It reuses `run_channel_18.py`'s own `profiles()` so there is one definition of what the profiles
are; that import needs `peclet.flow` on the path (any build — the transformation touches no solver
state).
"""
import os
import sys

import numpy as np

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/prod/ckpt.npz"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "stats_now.npz"

z = np.load(CK)
ny = int(z["ny"])
sys.argv = [sys.argv[0], "--ny", str(ny)]          # the driver reads its constants from argv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_channel_18 as R                          # noqa: E402

acc = R.new_acc()
for k in R.KEYS:
    acc[k] = np.array(z["acc_" + k])
acc["_n"] = float(z["acc_n"])
acc["_t0"] = float(z["acc_t0"])
acc["_t1"] = float(z["acc_t1"])
if acc["_n"] <= 0:
    raise SystemExit(f"{CK}: no samples accumulated yet (still inside the discarded transient)")

pr = R.profiles(acc)
t = float(z["t"])
np.savez(OUT, **{k: np.asarray(v) for k, v in pr.items()},
         turnovers=np.float64(t * R.UTAU), step=np.int64(z["step"]),
         grid=np.asarray([R.NX, R.NY, R.NZ]), utau=np.float64(R.UTAU))
print(f"wrote {OUT}: {int(acc['_n'])} samples over t u_tau/h "
      f"{pr['window_turnovers'][0]:.3f} .. {pr['window_turnovers'][1]:.3f}, "
      f"run at {t*R.UTAU:.3f} turnovers (step {int(z['step'])})")
print(f"  u_tau: imposed {R.UTAU:.6f}, wall gradient {pr['utau_wall']:.6f} "
      f"({100*(pr['utau_wall']/R.UTAU-1):+.1f} %; walls {pr['utau_lo']:.6f} / {pr['utau_hi']:.6f})")
j = R.NY // 2
print(f"  centreline: <alpha> {pr['alpha'][j]:.5f}, <u>_liq/u_tau {pr['u_liq'][j]/R.UTAU:.3f}")
k = int(np.argmax(pr['u_liq']))
print(f"  peak <u>_liq/u_tau {pr['u_liq'][k]/R.UTAU:.3f} at y/h = {pr['y'][k]:.3f}")
