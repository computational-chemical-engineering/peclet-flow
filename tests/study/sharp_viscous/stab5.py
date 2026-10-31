import mac2d as M, numpy as np, sys
from stab import blocks, rho_face
N = 16
for (muA, rA) in ((1e-3, 1e-4), (1e-2, 1e-3), (1e3, 1e4)):
  for kname, (ksch, cv) in (("K=facef", ("facef", False)), ("K=cellvol+facef", ("facef", True)), ("K=arith(current)", ("arith", False))):
    worst = np.zeros(5)
    for (ang, off) in ((0.0, 0.6), (0.0, 0.2), (22.5, 0.0), (22.5, 0.3), (45.0, 0.3), (67.5, 0.5)):
        case = M.Case(N, ang, off, muA, 1.0, "couette")
        rf = rho_face(case, *blocks(case, "facef", "arith", None, None)[3:], rA, 1.0)
        M.CELL_VOL = cv
        K, _, _, _, _ = blocks(case, ksch, "arith", None, {"xx", "yy", "yu", "xv"})
        M.CELL_VOL = False
        At, G, D, uidx, m = blocks(case, "sharp", "facef", "fit", None)
        for q, Dn in enumerate((0.1, 1.0, 10.0, 100.0, 1e4)):
            dt = Dn * case.h**2 * (1.0 if muA < 1 else rA / muA); Mm = np.diag(rf / dt); Mi = np.diag(dt / rf)
            P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(D @ Mi @ G) @ D
            worst[q] = max(worst[q], np.max(np.abs(np.linalg.eigvals(P @ np.linalg.solve(Mm + K, Mm + K - At)))))
    print("muA/muB=%-6g %-17s worst rho(G) at D_hi=0.1,1,10,100,1e4:" % (muA, kname), " ".join("%.3f" % x for x in worst))
