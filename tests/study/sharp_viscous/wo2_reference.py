"""WO-2 reference dump (doc/vof_sharp_viscous_stress.md §10 WO-2): the 2-D harness values the 3-D
extrusion test (`tests/kokkos/test_sharp_viscous.cpp`) must reproduce to 1e-12.

The oracle is mac2d with the production configuration and the §5.7 rule: scheme 'sharp',
SHARP_BASE 'facef', FCORR 'fit', TAU_RULE 'mixed-avg'. Per case the file lists the plane, the mixed
cells (C, both midsection fractions, the fitted tau and D_nn evaluated on the exact face averages,
whether mac2d fitted the cell, i.e. it is in-domain), every node with a mixed cell among its four (the two face
fractions and the two link fractions), and the exact face-averaged velocities within 3 cells of a
mixed cell.

  python wo2_reference.py ../../kokkos/sharp_viscous_mac2d.txt
"""
import sys

import numpy as np

import mac2d as M

M.SHARP_BASE, M.FCORR, M.TAU_RULE = "facef", "fit", "mixed-avg"

N = 16
CASES = [(kind, ang, off, ratio) for kind in ("poiseuille", "general") for ang in (11.25, 22.5, 33.75)
         for off, ratio in ((0.37, 0.01), (0.37, 1000.0))] + [("poiseuille", 45.0, 0.0, 0.01)]


def dump(out):
    lines = ["NCASES %d" % len(CASES)]
    for kind, ang, off, ratio in CASES:
        c = M.Case(N, ang, off, ratio, 1.0, kind)
        M.build(c, "sharp", "fit")
        L = M.build.last
        C, mixed, te, de, xall, m = L["C"], L["mixed"], L["tauexpr"], L["dnnexpr"], L["xall"], L["mac"]
        h = c.h
        lines.append("CASE %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d" % (
            N, c.n[0], c.n[1], c.t[0], c.t[1], c.c[0], c.c[1], c.s, c.muA, c.muB, kind == "general"))
        cells = [(i, j) for i in range(-1, N + 1) for j in range(-1, N + 1) if mixed[i + 1, j + 1]]
        rows = []
        for (i, j) in cells:
            xc, yc = (i + 0.5) * h, (j + 0.5) * h
            cbu = c.frac_line(xc, yc, np.array([0.0, 1.0]), "box")
            cbv = c.frac_line(xc, yc, np.array([1.0, 0.0]), "box")
            tau = sum(v * xall[k] for k, v in te[(i, j)].items()) if (i, j) in te else 0.0
            dnn = sum(v * xall[k] for k, v in de[(i, j)].items()) if (i, j) in de else 0.0
            rows.append("%d %d %.17g %.17g %.17g %d %.17g %.17g" % (i, j, C[i + 1, j + 1], cbu, cbv,
                                                                    (i, j) in te, tau, dnn))
        lines.append("NCELL %d" % len(rows))
        lines += rows
        mx = lambda i, j: 0 <= i <= N - 1 and 0 <= j <= N - 1 and mixed[i + 1, j + 1]
        rows = []
        for i in range(0, N + 1):
            for j in range(0, N + 1):
                if not any(mx(i - di, j - dj) for di in (0, 1) for dj in (0, 1)):
                    continue
                fu = c.frac_line(i * h, j * h, np.array([1.0, 0.0]), "box")
                fv = c.frac_line(i * h, j * h, np.array([0.0, 1.0]), "box")
                thy = c.frac_line(i * h, j * h, np.array([0.0, 1.0]), "hat")
                thx = c.frac_line(i * h, j * h, np.array([1.0, 0.0]), "hat")
                rows.append("%d %d %.17g %.17g %.17g %.17g" % (i, j, fu, fv, thy, thx))
        lines.append("NNODE %d" % len(rows))
        lines += rows
        near = lambda i, j: any(abs(i - p) <= 3 and abs(j - q) <= 3 for (p, q) in cells)
        ru = ["%d %d %.17g" % (i, j, xall[m.iu(i, j)]) for i in range(N + 1) for j in range(-1, N + 1)
              if near(i, j)]
        rv = ["%d %d %.17g" % (i, j, xall[m.iv(i, j)]) for i in range(-1, N + 1) for j in range(N + 1)
              if near(i, j)]
        lines.append("NU %d" % len(ru))
        lines += ru
        lines.append("NV %d" % len(rv))
        lines += rv
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote %d cases to %s" % (len(CASES), out))


if __name__ == "__main__":
    dump(sys.argv[1])
