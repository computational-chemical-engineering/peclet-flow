"""smooth startup (forcing ramped by (1-cos)/2 over R steps): split vs implicit, new sharp and current arith."""
import mac2d as M, numpy as np
from stab import blocks, rho_face
N = 24; R = 40
def ops(case, target, kcellvol, kscheme):
    if target == "sharp": M.SHARP_BASE, M.FCORR = "facef", "fit"; sch = "sharp"
    else: M.SHARP_BASE, M.FCORR = "arith", None; sch = "arith"
    Af, b, uidx, xall, m = M.build(case, sch, "fit"); M.FCORR = None
    nv = len(uidx); Af = Af.toarray()
    M.CELL_VOL = kcellvol; K = blocks(case, kscheme, "arith", None, {"xx", "yy", "yu", "xv"})[0]; M.CELL_VOL = False
    return -Af[:nv, :nv], Af[:nv, nv:nv + N * N], Af[nv:nv + N * N, :nv], -b[:nv], K, uidx, m
for (muA, rA) in ((1e-2, 1e-3), (1e-3, 1e-4), (1e2, 1e3)):
  for (ang, off) in ((22.5, 0.3),):
    case = M.Case(N, ang, off, muA, 1.0, "general")
    for target, kcv, ks in (("sharp", True, "facef"), ("arith", False, "arith")):
        At, G, D, F, K, uidx, m = ops(case, target, kcv, ks)
        rf = rho_face(case, uidx, m, rA, 1.0)
        out = []
        for Dn in (0.1, 0.3, 1.0):
            dt = Dn * case.h**2 * (1.0 if muA < 1 else rA / muA)
            Mm = np.diag(rf / dt); Mi = np.diag(dt / rf)
            P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(D @ Mi @ G) @ D
            Si = np.linalg.inv(Mm + At); Ss = np.linalg.inv(Mm + K)
            ui = np.zeros(len(rf)); us = np.zeros(len(rf)); d = []
            for n in range(1, 3 * R + 1):
                Fn = F * 0.5 * (1 - np.cos(np.pi * min(n, R) / R))
                ui = P @ (Si @ (Mm @ ui + Fn)); us = P @ (Ss @ (Mm @ us + (K - At) @ us + Fn))
                d.append(np.max(np.abs(us - ui)) / np.max(np.abs(ui)))
            out.append("D=%.1f: max %.1e, end %.1e" % (Dn, max(d), d[-1]))
        print("muA/muB=%-5g %-5s split-vs-implicit:" % (muA, target), " | ".join(out))
