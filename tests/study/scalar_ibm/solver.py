"""2-D prototype, round 5: can peclet's solvers handle the probe-flux operator?

A = probe-flux cut-cell FV (jc.py, probe 0.7h) on the disc, Dirichlet (worst case) and Robin.
Systems: steady (A u = b) and one backward-Euler step (kappa/dt + A) u = b at diffusion number
dt*D/h^2 = 1, 10.
Tested:
  GS     : lexicographic Gauss-Seidel directly on A (stands in for RB-GS): converges?
  DC     : defect correction  u += S^-1 (b - A u)  with S = symmetric compact surrogate
           (aperture FV + diagonal wall term A_w/(d + 1/k), d = centroid distance; SPD) -> rho(I - S^-1 A)
  BiCGStab / GMRES(20) preconditioned by S (exact S^-1 here = an ideal MG V-cycle): iterations to 1e-10
Also: does S itself suit MG (it is the existing operator family + a diagonal) - assumed, not tested here.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from disc_bc import Geo, assemble_fv
from jc import build


SURR = "centroid"


def surrogate(G, k):
    cells = G.kap > 1e-12
    h = G.h
    d = np.maximum(G.dwall, 1e-3 * h) if SURR == "centroid" else 0.7 * h  # lumped probe: d = probe distance
    wc = np.where(cells, G.Aw / (h * h * (d + (0 if np.isinf(k) else 1 / k))), 0)
    S, _ = assemble_fv(G, G.ax, G.ay, cells, wall_coef=wc)
    return S


def gs_rate(A, b, n=200):
    L = sp.tril(A, 0).tocsc(); U = A - sp.tril(A, 0)
    u = np.zeros_like(b); r0 = np.linalg.norm(b)
    hist = []
    for it in range(n):
        u = spla.spsolve_triangular(L, b - U @ u, lower=True)
        hist.append(np.linalg.norm(b - A @ u) / r0)
        if not np.isfinite(hist[-1]) or hist[-1] > 1e8:
            return "DIVERGES", it
    return hist[-1], n


def run(ND, k, dtn):
    G = Geo(ND, (0.31, 0.62))
    A, M = build(G, k, fc=False, wall="probe0.7")
    S = surrogate(G, k)
    h2 = G.h**2
    if dtn is not None:
        idt = 1.0 / (dtn * h2)  # D = 1
        A = A + sp.diags(M * idt); S = S + sp.diags(M * idt)
    A = A.tocsr(); S = S.tocsc()
    rng = np.random.default_rng(0)
    b = M * (1 + 0.1 * rng.standard_normal(A.shape[0]))
    out = {}
    out["GS"] = gs_rate(A, b)
    lu = spla.splu(S)
    P = spla.LinearOperator(A.shape, matvec=lu.solve)
    # defect-correction spectral radius by power iteration on E = I - S^-1 A
    v = rng.standard_normal(A.shape[0])
    for _ in range(300):
        w = v - lu.solve(A @ v); nv = np.linalg.norm(w); v = w / nv
    out["rho_DC"] = nv
    for name, fn in (("BiCGStab", spla.bicgstab), ("GMRES20", lambda A_, b_, **kw: spla.gmres(A_, b_, restart=20, callback_type='legacy', **kw))):
        cnt = [0]
        def cb(_):
            cnt[0] += 1
        x, info = fn(A, b, M=P, rtol=1e-10, maxiter=500, callback=cb)
        out[name] = (cnt[0], info, np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    # small-cell census
    cells = G.kap > 1e-12
    out["min_kappa"] = G.kap[cells].min()
    return out


if __name__ == "__main__":
    import sys
    SURR = sys.argv[1] if len(sys.argv) > 1 else "centroid"
    for k in (np.inf, 10.0, 1.0):
        for dtn in (None, 1.0):
            for ND in (32, 64, 128):
                o = run(ND, k, dtn)
                gs = o["GS"]
                print(f"Bi={k:5g} dtD/h2={str(dtn):5s} ND={ND:3d} | GS: {gs[0] if isinstance(gs[0], str) else f'{gs[0]:.1e}'} "
                      f"(it {gs[1]}) | rho(DC)={o['rho_DC']:.3f} | BiCGStab it={o['BiCGStab'][0]} | "
                      f"GMRES20 it={o['GMRES20'][0]} res={o['GMRES20'][2]:.1e} | min kappa={o['min_kappa']:.1e}", flush=True)
