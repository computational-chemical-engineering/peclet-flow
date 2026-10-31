"""spectral radius of u <- P (M+K)^-1 (M + K - A) u  (homogeneous Dirichlet), A = sharp target, K = implicit part."""
import mac2d as M, numpy as np, sys
def blocks(case, scheme, base, fcorr, own):
    M.SHARP_BASE, M.FCORR, M.OWN_ONLY = base, fcorr, own
    A, b, uidx, xall, m = M.build(case, scheme, "fit")
    M.OWN_ONLY = None
    nv = len(uidx); A = A.toarray()
    return -A[:nv, :nv], A[:nv, nv:nv + case.N**2], A[nv:nv + case.N**2, :nv], uidx, m
def rho_face(case, uidx, m, rA, rB):
    N, h = case.N, case.h; rf = np.zeros(len(uidx))
    C = lambda i, j: case.frac_cell(i * h, j * h) if True else 0
    for i in range(N + 1):
        for j in range(-1, N + 1):
            k = m.iu(i, j)
            if k in uidx:
                c1, c2 = C(i - 1, j), C(i, j); rf[uidx[k]] = 0.5 * ((c1 + c2) * rA + (2 - c1 - c2) * rB)
    for i in range(-1, N + 1):
        for j in range(N + 1):
            k = m.iv(i, j)
            if k in uidx:
                c1, c2 = C(i, j - 1), C(i, j); rf[uidx[k]] = 0.5 * ((c1 + c2) * rA + (2 - c1 - c2) * rB)
    return rf
def main():
  N = int(sys.argv[1]); ang = float(sys.argv[2])
  for (muA, rA) in ((0.01, 1e-3), (100.0, 1e3), (1e-3, 1e-4)):
      case = M.Case(N, ang, 0.3, muA, 1.0, "couette")
      At, G, D, uidx, m = blocks(case, "sharp", "facef", "fit", None)
      rf = rho_face(case, uidx, m, rA, 1.0)
      for kname, ksch in (("K=arith", "arith"), ("K=facef", "facef")):
          K, _, _, _, _ = blocks(case, ksch, "arith", None, {"xx", "yy", "yu", "xv"})
          out = []
          for Dn in (0.03, 0.1, 0.3, 1.0, 10.0, 1e3, 1e6):
              dt = Dn * case.h**2 * 1.0 / 1.0  # D of phase B (mu=1, rho=1)
              Mm = np.diag(rf / dt); Mi = np.diag(dt / rf)
              S = D @ Mi @ G; P = np.eye(len(rf)) - Mi @ G @ np.linalg.pinv(S) @ D
              Amp = P @ np.linalg.solve(Mm + K, Mm + K - At)
              out.append(np.max(np.abs(np.linalg.eigvals(Amp))))
          print("muA/muB=%g rhoA/rhoB=%g %s  rho(G) at D_B=0.03..1e6:" % (muA, rA, kname), " ".join("%.4f" % x for x in out))

if __name__ == "__main__":
    main()
