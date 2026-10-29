"""A-priori operator test for doc/vof_momentum_second_order.md (numpy, host, seconds).

One directional sweep of flow's mass-momentum-consistent update (src/vof/momentum_advect.hpp, the
deviation form `u_new = u_old + dev/rho_new`) transporting a TRANSVERSE velocity v(x) through a sharp
liquid slab at uniform Courant number a, periodic. In 1D the PLIC flux of a step is exact, so C is
exact and v has the exact solution v0(x - T): every error printed is the momentum reconstruction's.

Schemes: upwind (flow default) | muscl_mixed (flow's opt-in MinMod of the mixed CV velocity) |
mc_mixed (MC of the mixed velocity, no cap) | muscl_mixed_capped (muscl_mixed + the mass-aware cap) |
muscl_phase (the design: per-phase interface-biased MC slopes + the cap) | phase_central (Basilisk's
interface-biased rule with an UNLIMITED central slope, no cap).

Run:  python -I apriori1d.py        (prints the table quoted in the design note, section 2)
"""
import numpy as np

A = 0.2


def mm(p, q):
    return np.where(p * q > 0, np.where(np.abs(p) < np.abs(q), p, q), 0.0)


def mc(p, q):
    s = np.where(p * q > 0, np.sign(p), 0.0)
    return s * np.minimum(np.minimum(2 * np.abs(p), 2 * np.abs(q)), 0.5 * np.abs(p + q))


def phase_est(dm, dp, f):
    """Interface-biased slope estimate for the phase whose fraction is f (>= 1/2 = holds it)."""
    fm, fp = np.roll(f, 1) >= 0.5, np.roll(f, -1) >= 0.5
    est = np.where(fm & fp, 0.5 * (dm + dp), np.where(fm, dm, np.where(fp, dp, 0.0)))
    return np.where(f >= 0.5, est, 0.0)


def tvd(dm, dp, est):
    s = np.where(dm * dp > 0, np.sign(dm), 0.0)
    return s * np.minimum(np.minimum(2 * np.abs(dm), 2 * np.abs(dp)), np.abs(est))


def run(N, ratio, scheme, v0, steps=None, slab=(0.3, 0.7), wisp=0.0):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    xl = np.arange(N) * h
    C = np.clip((np.minimum(xl + h, slab[1]) - np.maximum(xl, slab[0])) / h, 0, 1)
    rg, rl = 1.0, float(ratio)
    v = v0(x) if callable(v0) else np.array(v0, dtype=float)
    nst = steps if steps is not None else int(round(1.0 / (A * h)))
    bites = 0
    for _ in range(nst):
        right = np.roll(C, -1) >= np.roll(C, 1)  # liquid sits on the side of the fuller neighbour
        F = np.where(right, np.minimum(A, C), np.maximum(0.0, C - (1 - A)))  # + face, donor = cell
        F = np.where((C <= wisp) | (C >= 1 - wisp), C * A, F)
        dm, dp = v - np.roll(v, 1), np.roll(v, -1) - v
        if scheme == 'upwind':
            sl = sg = np.zeros_like(v)
        elif scheme in ('muscl_mixed', 'muscl_mixed_capped'):
            sl = sg = mm(dm, dp)
        elif scheme == 'mc_mixed':
            sl = sg = mc(dm, dp)
        elif scheme == 'muscl_phase':
            sl, sg = tvd(dm, dp, phase_est(dm, dp, C)), tvd(dm, dp, phase_est(dm, dp, 1 - C))
        elif scheme == 'phase_central':
            sl, sg = phase_est(dm, dp, C), phase_est(dm, dp, 1 - C)
        if scheme.endswith('_capped') or scheme == 'muscl_phase':
            # the cap of the note's section 4.3, 1D uniform a: the - face is inflow, no dilation
            R = np.maximum(rg * (1 - C) + rl * C - (rg * (A - F) + rl * F), 0.0)
            m = np.minimum(np.abs(dm), np.abs(dp))
            ext = np.abs(0.5 * (1 - A) * (rg * (A - F) * sg + rl * F * sl))
            th = np.where(ext > R * m, R * m / np.where(ext > 0, ext, 1.0), 1.0)
            bites += int((th < 1).sum())
            sl, sg = th * sl, th * sg
        ul, ug = v + 0.5 * (1 - A) * sl, v + 0.5 * (1 - A) * sg
        FM = np.roll(F, 1)
        Cn = C + FM - F
        rn = rg * (1 - Cn) + rl * Cn
        dev = rg * ((A - FM) * (np.roll(ug, 1) - v) - (A - F) * (ug - v)) \
            + rl * (FM * (np.roll(ul, 1) - v) - F * (ul - v))
        v = v + dev / rn
        C = Cn
    return x, C, v, nst, bites


def v0(s):
    return np.sin(2 * np.pi * s) + 0.5 * np.cos(6 * np.pi * s)


if __name__ == '__main__':
    Ns = [32, 64, 128, 256, 512]
    schemes = ['upwind', 'muscl_mixed', 'mc_mixed', 'muscl_mixed_capped', 'muscl_phase',
               'phase_central']
    print("L1 error of v after one period, a = 0.2; order = log2 of successive ratios")
    for ratio in [1.0, 1e3, 1e4]:
        for sch in schemes:
            l1, bt = [], []
            for N in Ns:
                x, C, v, nst, b = run(N, ratio, sch, v0)
                l1.append(np.abs(v - v0(x - nst * A / N)).mean())
                bt.append(b)
            print(f"ratio {ratio:5.0e} {sch:19s} " + " ".join(f"{q:.2e}" for q in l1) + " | ord "
                  + " ".join(f"{np.log2(l1[k] / l1[k + 1]):5.2f}" for k in range(len(Ns) - 1))
                  + (f" | cap bites {bt}" if sch.endswith('capped') or sch == 'muscl_phase' else ""))
    print("noise: v = 1 + 1e-14 n(x), N = 64, 200 steps; identity: v = 1 exactly")
    noise = 1e-14 * np.random.default_rng(7).standard_normal(64)
    for ratio in [1e3, 1e4]:
        for sch in schemes:
            _, _, v, _, _ = run(64, ratio, sch, 1.0 + noise, steps=200)
            _, _, w, _, _ = run(64, ratio, sch, np.ones(64), steps=200)
            print(f"ratio {ratio:5.0e} {sch:19s} max|v-1| {np.abs(noise).max():.2e} -> "
                  f"{np.abs(v - 1).max():.2e}   identity {np.abs(w - 1).max():.1e}")
