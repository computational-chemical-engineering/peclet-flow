"""Layer 3 rungs 2-3 gate: ROTLET. A single sphere spinning in Stokes flow.

Exact exterior solution for a sphere of radius R spinning at omega in UNBOUNDED Stokes flow:
    u(r) = (R/|r|)^3 * (omega x r)
a pure rotational (couplet) field: zero net force, decaying as 1/|r|^2.

The solver's box is PERIODIC, so the true solution is the lattice of images, not the unbounded
rotlet. That difference is physics, not discretisation error, and it does not shrink under
refinement -- so the error is measured against BOTH references and the lattice sum's own
convergence (9^3 vs 13^3 images) is printed, so a reader can see whether the reference is
trustworthy before trusting an order taken against it.

SAMPLING. get_u/get_v/get_w return MAC FACE values -- u at (i-1/2,j,k), v at (i,j-1/2,k), w at
(i,j,k-1/2). Comparing them against the analytic field sampled at CELL CENTRES injects an O(h)
offset that caps the measured order at 1 however accurate the solver is. Each component is
therefore compared at its own staggered point.

Two shells: the near band [1.15R, 2R], a few cells off a curved cut wall, and a far band
[2R, 3.5R]. Order is the claim, not the absolute.
"""
import os
import sys
import numpy as np
from peclet import flow as sdflow

RF = float(os.environ.get("ROTLET_RF", "0.10"))   # R / N
OMEGA = 0.02                                      # rad per unit time, about z
MU, RHO, DT, STEPS = 0.1, 1.0, 200.0, 400
NEAR = (1.15, 2.0)
FAR = (2.0, 3.5)
KN_R, KI_I, KI_R = 16, 2, 17


def rotlet(px, py, pz, R, w):
    """Unbounded rotlet, omega = (0,0,w)."""
    r = np.sqrt(px*px + py*py + pz*pz)
    amp = np.where(r > 0, (R / np.where(r > 0, r, 1.0))**3, 0.0)
    return -amp * w * py, amp * w * px, np.zeros_like(px)


def rotlet_lattice(px, py, pz, R, w, L, nimg):
    """Rotlet summed over a (2*nimg+1)^3 block of periodic images."""
    ux = np.zeros_like(px); uy = np.zeros_like(px); uz = np.zeros_like(px)
    for ix in range(-nimg, nimg + 1):
        for iy in range(-nimg, nimg + 1):
            for iz in range(-nimg, nimg + 1):
                a, b, c = rotlet(px - ix*L, py - iy*L, pz - iz*L, R, w)
                ux += a; uy += b; uz += c
    return ux, uy, uz


def run(N, crossings=False):
    R = RF * N
    C = (0.5 * N, 0.5 * N, 0.5 * N)
    node_ints = np.array([1, -1, -1], dtype=np.int32)
    node_reals = np.zeros(KN_R); node_reals[0] = R
    node_reals[14] = 1.0; node_reals[15] = 1.0
    ii = np.zeros((1, KI_I), dtype=np.int32); ii[0] = (0, -1)
    ir = np.zeros((1, KI_R)); ir[0, 0:3] = C; ir[0, 6] = 1.0; ir[0, 7] = 1.0

    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(DT)
    s.set_advection(False)
    s.set_velocity_solver_params(120)
    s.set_pressure_solver_params(30)
    # MG depth 4, not 1. levels=1 leaves the "coarse" solve on the FULL grid, which on CUDA costs
    # 920 ms/step at N=64 against 19.5 ms at depth 4 (measured). Depth changes the solver path, not
    # the converged answer.
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(node_ints, node_reals, ii.ravel(), ir.ravel(), periodic=True)
    s.set_instance_motion(0, lin_vel=[0.0, 0.0, 0.0], ang_vel=[0.0, 0.0, OMEGA])
    if crossings:
        s.set_exact_crossings_from_scene()      # analytic wall crossings (Layer 2)
    s.set_solid_from_scene(True)
    prev = 0.0
    for it in range(STEPS):
        s.step()
        m = float(np.abs(np.asarray(s.get_v())).max())
        if it > 15 and abs(m - prev) < 1e-9 * (abs(m) + 1e-300):
            break
        prev = m
    return (N, R, C, np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w()),
            s.max_open_divergence(), s.wall_flux_imbalance(), it + 1)


OFFS = ((-0.5, 0.0, 0.0), (0.0, -0.5, 0.0), (0.0, 0.0, -0.5))


def err(res, shell, nimg=None):
    """Relative L2 error over `shell`; lattice reference when nimg is given, unbounded otherwise."""
    N, R, C, u, v, w, _, _, _ = res
    g = np.arange(N).astype(float)
    num = den = 0.0
    npts = 0
    for comp, fld in enumerate((u, v, w)):
        off = OFFS[comp]
        X, Y, Z = np.meshgrid(g + off[0], g + off[1], g + off[2], indexing="ij")
        px = X - C[0]; px -= N * np.round(px / N)
        py = Y - C[1]; py -= N * np.round(py / N)
        pz = Z - C[2]; pz -= N * np.round(pz / N)
        r = np.sqrt(px*px + py*py + pz*pz)
        m = (r > shell[0] * R) & (r < shell[1] * R)
        qx, qy, qz = px[m], py[m], pz[m]
        # References are evaluated ONLY on the shell: the lattice sum is O(images x points), so
        # summing over the whole N^3 grid costs minutes of numpy for a few percent of the cells.
        ref = (rotlet_lattice(qx, qy, qz, R, OMEGA, float(N), nimg) if nimg is not None
               else rotlet(qx, qy, qz, R, OMEGA))
        num += ((fld[m] - ref[comp])**2).sum()
        den += (ref[comp]**2).sum()
        npts += int(m.sum())
    return np.sqrt(num / max(den, 1e-300)), npts


def order_study():
    Ns = [int(x) for x in (sys.argv[1:] or ["48", "96", "144"])]
    nimg = int(os.environ.get("ROTLET_IMAGES", "4"))
    crossings = os.environ.get("ROTLET_CROSSINGS", "0") == "1"
    print("rotlet: R/N=%.3f  omega=%g  near shell r/R %s  far shell %s  images=%d^3  "
          "exact crossings=%s" % (RF, OMEGA, NEAR, FAR, 2*nimg+1, crossings))
    rows = []
    for N in Ns:
        res = run(N, crossings)
        e_near, n_near = err(res, NEAR, nimg)
        e_far, n_far = err(res, FAR, nimg)
        e_unb, _ = err(res, NEAR)
        e_ref2, _ = err(res, NEAR, nimg + 2)     # is the lattice reference itself converged?
        rows.append((N, e_near, e_far))
        print("  N=%3d  R=%.1f cells  steps=%3d  near/far probes %6d/%7d  max|div|=%.2e"
              % (N, RF * N, res[8], n_near, n_far, res[6]))
        print("         L2 rel err  near %.4e  far %.4e   (unbounded ref %.4e; reference moves "
              "%.1e going to %d^3 images)"
              % (e_near, e_far, e_unb, abs(e_near - e_ref2), 2*(nimg+2)+1))
    print("  orders (against the %d^3-image lattice):" % (2*nimg+1))
    for a, b in zip(rows, rows[1:]):
        pn = np.log(a[1] / b[1]) / np.log(b[0] / a[0])
        pf = np.log(a[2] / b[2]) / np.log(b[0] / a[0])
        print("    N %3d -> %3d :  near shell %.2f    far shell %.2f" % (a[0], b[0], pn, pf))


def reference_probe():
    """IS THE PLATEAU THE SOLVER OR THE REFERENCE?

    Decisive experiment: hold R fixed IN CELLS (so the discretisation error is unchanged) and
    change only the box, i.e. only R/L. The naive periodic-image correction to an unbounded rotlet
    scales like (r/L)^3 at fixed r/R, so halving R/L must drop it ~8x; the discretisation error
    cannot move at all. Whatever the near-shell error does is therefore attributable.
    """
    Rcells = float(os.environ.get("ROTLET_RCELLS", "7.2"))
    global RF
    print("reference probe: R fixed at %.1f cells; only the box (and so R/L) changes" % Rcells)
    prev = None
    for N in [int(x) for x in (sys.argv[1:] or ["72", "144"])]:
        RF = Rcells / N
        res = run(N, os.environ.get("ROTLET_CROSSINGS", "0") == "1")
        e_near, n = err(res, NEAR, int(os.environ.get("ROTLET_IMAGES", "4")))
        line = ("  N=%3d  R/L=%.4f  R=%.1f cells  near-shell L2 rel err %.4e"
                % (N, Rcells / N, Rcells, e_near))
        if prev is not None:
            line += "   (dropped %.1fx; the image correction predicts ~%.1fx)" % (
                prev[1] / e_near, (prev[0] / (Rcells / N))**3 * 0 + (N / prev[2])**3)
        print(line)
        prev = (Rcells / N, e_near, N)


if __name__ == "__main__":
    if os.environ.get("ROTLET_MODE", "order") == "refprobe":
        reference_probe()
    else:
        order_study()
