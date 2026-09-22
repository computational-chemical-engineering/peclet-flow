"""R2 gate: dem's external-TORQUE entry point (set_external_torques).

Two claims, both at OMP_NUM_THREADS=1 (dem's multithreaded step is nondeterministic -- §6.0b).
One free body, no gravity, no contacts, no walls: the predictor's rotational update is the whole
physics, so a miss is the torque term and nothing else.

  A. PRINCIPAL AXIS -- EXACT. A constant world torque about a principal axis of a body whose
     angular velocity is already along that axis keeps w x (I w) identically zero, so the discrete
     update degenerates to w_{n+1} = w_n + (tau/I) dt and the trajectory is the closed form
     w(t) = tau t / I with NO time-stepping error at all. Anything but round-off here is a wiring
     bug (wrong inertia component, body/world frame confusion, torque dropped).

  B. NON-PRINCIPAL AXIS -- vs a scipy reference. A constant WORLD torque about a tilted axis on a
     body with three distinct principal moments: the body frame tumbles under it, so tau_body(t)
     is time dependent and the trajectory is genuinely Euler's equations. Reference: solve_ivp
     (DOP853, rtol/atol 1e-12) on the SAME system in dem's own convention --
         q = (x,y,z,w), R(q) maps BODY -> WORLD,  dq/dt = 0.5 (0, w_world) (x) q
         dw_b/dt = invI (R^T tau_world - w_b x I w_b)
     dem's integrator is semi-implicit Euler, so the claim is a dt-CONVERGENCE ladder (first
     order), taken at COARSE dt where truncation dominates -- below dt ~ 1e-3 the float32 state
     floor (gate A) is already larger than the truncation error and the ladder inverts, which is
     itself reported.

  C. Angular-momentum bookkeeping, free: dL_world/dt = tau_world exactly in the continuum, so
     |L(t) - L(0) - tau t| is a second, convention-independent read of the same error.
"""
import os
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1 (dem determinism)"

from peclet import dem as pdem
from scipy.integrate import solve_ivp

# Three DISTINCT principal moments (unit mass), so no accidental symmetry can hide a frame error.
I = np.array([0.40, 0.65, 0.90])
INV_I = 1.0 / I


def qmul(a, b):  # (x,y,z,w) convention, same algebra as dem_portable.hpp quatMult
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def rotmat(q):  # body -> world
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def make_sim(w0, tau, dt):
    """One free unit-mass body with the principal inertia I, spinning at w0 under world torque."""
    s = pdem.Simulation(64)
    s.set_gravity(0.0, 0.0, 0.0)
    s.set_dt(dt)
    s.set_positions(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    s.set_inv_mass(np.array([1.0], dtype=np.float32))
    s.set_inv_inertia(np.array([INV_I], dtype=np.float32))
    s.set_quaternions(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    s.set_angular_velocities(np.array([w0], dtype=np.float32))
    s.set_external_torques(np.array([tau], dtype=np.float32))
    return s


def run(s, dt, nsteps):
    for _ in range(nsteps):
        s.step(dt)
    q = np.asarray(s.get_quaternions())[0].astype(np.float64)
    w = np.asarray(s.get_angular_velocities())[0].astype(np.float64)
    return q / np.linalg.norm(q), w


def reference(w0_world, q0, tau, T):
    def rhs(t, y):
        q = y[0:4] / np.linalg.norm(y[0:4])
        wb = y[4:7]
        R = rotmat(q)
        ww = R @ wb
        dq = 0.5 * qmul(np.array([ww[0], ww[1], ww[2], 0.0]), q)
        tb = R.T @ tau
        dwb = INV_I * (tb - np.cross(wb, I * wb))
        return np.concatenate([dq, dwb])
    wb0 = rotmat(q0).T @ w0_world
    y0 = np.concatenate([q0, wb0])
    sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
    y = sol.y[:, -1]
    q = y[0:4] / np.linalg.norm(y[0:4])
    return q, rotmat(q) @ y[4:7]


def gate_a():
    print("A. torque about a principal axis (exact closed form  w(t) = tau t / I)")
    ok = True
    for axis in range(3):
        tau = np.zeros(3); tau[axis] = 0.25
        dt, n = 1e-3, 4000
        s = make_sim(np.zeros(3), tau, dt)
        _, w = run(s, dt, n)
        exact = np.zeros(3); exact[axis] = tau[axis] * INV_I[axis] * dt * n
        err = np.linalg.norm(w - exact) / np.linalg.norm(exact)
        ok = ok and err < 1e-4
        print("   axis %d: I=%.2f  w_sim=%s  w_exact=%s  rel err %.2e"
              % (axis, I[axis], np.round(w, 8), np.round(exact, 8), err))
    # The residual is float32 accumulation, not truncation: at fixed end time it must GROW with
    # the step count. Exhibit that rather than asserting it.
    tau = np.array([0.25, 0.0, 0.0])
    print("   float32 floor, fixed T=4: error vs step count (truncation would be flat at zero)")
    for dt in (4e-3, 1e-3, 2.5e-4):
        n = int(round(4.0 / dt))
        s = make_sim(np.zeros(3), tau, dt)
        _, w = run(s, dt, n)
        ex = tau[0] * INV_I[0] * 4.0
        print("      dt=%.1e  %6d steps  rel err %.2e   (%.1f x float32 eps per 1000 steps)"
              % (dt, n, abs(w[0] - ex) / ex, abs(w[0] - ex) / ex / (n * 1.19e-7) * 1000))
    print("   GATE A %s  (relative error < 1e-4 = dem's float32 accumulation floor at 4000 steps;"
          " the closed form is exact by construction)" % ("PASS" if ok else "FAIL"))
    return ok


def gate_b():
    print("B. torque about a NON-principal axis vs scipy Euler-equation reference")
    tau = np.array([0.20, -0.13, 0.31])
    w0 = np.array([0.9, 0.4, -0.6])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    T = 2.0
    qr, wr = reference(w0, q0, tau, T)
    rows, prev = [], None
    for dt in (8e-3, 4e-3, 2e-3, 1e-3):
        n = int(round(T / dt))
        s = make_sim(w0, tau, dt)
        q, w = run(s, dt, n)
        if np.dot(q, qr) < 0:
            q = -q
        ew = np.linalg.norm(w - wr) / np.linalg.norm(wr)
        eq = np.linalg.norm(q - qr)
        rate = np.log2(prev / ew) if prev else float("nan")
        rows.append((dt, ew, eq, rate))
        prev = ew
        print("   dt=%.1e  |dw|/|w| = %.3e   |dq| = %.3e   order %s"
              % (dt, ew, eq, "  --" if np.isnan(rate) else "%.2f" % rate))
    # first order in dt, and the finest step within 0.5%
    orders = [r[3] for r in rows[1:]]
    ok = rows[-1][1] < 1e-3 and min(orders) > 0.75
    print("   w_ref = %s   w_sim(dt=1e-3) = %s" % (np.round(wr, 6), np.round(w, 6)))
    print("   GATE B %s  (finest-dt error < 0.1%%, convergence order > 0.75 -- semi-implicit Euler)"
          % ("PASS" if ok else "FAIL"))
    # Below the truncation/round-off crossover the ladder inverts: report it, do not hide it.
    print("   crossover: the same ladder continued past the float32 floor")
    for dt in (2e-4, 5e-5):
        s = make_sim(w0, tau, dt)
        q, w = run(s, dt, int(round(T / dt)))
        print("      dt=%.1e  |dw|/|w| = %.3e  (float32 state, %d steps)"
              % (dt, np.linalg.norm(w - wr) / np.linalg.norm(wr), int(round(T / dt))))
    return ok


def gate_c():
    print("C. angular-momentum bookkeeping: L_world(t) - L_world(0) = tau t")
    tau = np.array([0.20, -0.13, 0.31])
    w0 = np.array([0.9, 0.4, -0.6])
    T = 2.0
    L0 = rotmat(np.array([0.0, 0.0, 0.0, 1.0])) @ (I * w0)
    ok = True
    for dt in (4e-4, 1e-4, 5e-5):
        s = make_sim(w0, tau, dt)
        q, w = run(s, dt, int(round(T / dt)))
        R = rotmat(q)
        L = R @ (I * (R.T @ w))
        err = np.linalg.norm(L - L0 - tau * T) / np.linalg.norm(L0 + tau * T)
        ok = ok and err < 2e-2
        print("   dt=%.1e  |L - L0 - tau t| / |L| = %.3e" % (dt, err))
    print("   GATE C %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    a, b, c = gate_a(), gate_b(), gate_c()
    print("R2 GATE %s" % ("PASS" if (a and b and c) else "FAIL"))
