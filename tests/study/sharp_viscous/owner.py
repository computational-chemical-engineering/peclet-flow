import mac2d as M, numpy as np
M.SHARP_BASE, M.FCORR, M.TAU_RULE = "facef", "fit", "owner"
for kind in ("couette", "poiseuille", "general"):
  for r in (0.01, 100.0, 0.001):
    worst_o, worst_e = 9.0, 0.0
    for ang in (0.0, 22.5, 33.75):
      for off in (0.0, 0.375, 0.75):
        e = [M.solve_err(M.Case(N, ang, off, r, 1.0, kind), "sharp")[0] for N in (16, 32, 64)]
        worst_e = max(worst_e, e[-1])
        if e[-1] > 1e-10: worst_o = min(worst_o, np.log(e[1] / e[2]) / np.log(2))
    print("owner-rule %-10s r=%-6g worst err N64 %.2e, worst order 32->64 %s" % (kind, r, worst_e, "exact" if worst_o == 9.0 else "%.2f" % worst_o))
