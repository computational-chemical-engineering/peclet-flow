"""Gate for flow's refresh_wall_velocity() (the cheap moving-BC entry point).

The claim is exactness, not approximation: for a driver that changes an instance's VELOCITY every
step while its TRANSFORM stays put, refresh_wall_velocity() must reproduce rebuild_geometry()
BITWISE -- same velocity field, same pressure, same reaction force, after many steps of a
time-varying wall velocity. Anything less and it is a different solver, not a faster path to the
same one.

Three claims:
  A. BITWISE over a 60-step oscillating-wall run (u, v, w, P, and the per-instance reaction force).
  B. The cost actually falls (that is the whole point).
  C. Stopping the motion clears the wall-velocity fields. wallVelView() keys off the field's
     EXTENT, not hasMotion_, so a previously-built uBc_ used to keep being folded into the momentum
     operator after the caller set the velocity back to zero -- a real bug on the rebuild path too.
     The check: run with a wall velocity, then set it to zero and step; the result must equal a run
     that never moved at all.
"""
import os, time
import numpy as np
from peclet import flow as sdflow

N, RF = 48, 0.18
R = RF * N
RHO, MU, U0 = 1.0, 0.1, 0.02
KN_R, KI_I, KI_R = 16, 2, 17
STEPS = int(os.environ.get("STEPS", "60"))
OM = 0.01


def make(dt, u_init):
    ni = np.array([1, -1, -1], dtype=np.int32)
    nr = np.zeros(KN_R); nr[0] = R; nr[14] = 1.0; nr[15] = 1.0
    ii = np.zeros((1, KI_I), dtype=np.int32); ii[0] = (0, -1)
    ir = np.zeros((1, KI_R)); ir[0, 0:3] = (0.5 * N,) * 3; ir[0, 6] = 1.0; ir[0, 7] = 1.0
    s = sdflow.Solver(N, N, N)
    s.set_rho(RHO); s.set_mu(MU); s.set_dt(dt); s.set_advection(False)
    s.set_velocity_solver_params(60); s.set_pressure_solver_params(20)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(ni, nr, ii.ravel(), ir.ravel(), periodic=True)
    s.set_instance_motion(0, lin_vel=[u_init, 0.0, 0.0])
    s.set_solid_from_scene(True)
    return s


def run(mode, dt=5.0, steps=STEPS, stop_after=None):
    s = make(dt, U0)
    t0 = time.time()
    for it in range(steps):
        t = (it + 1) * dt
        u = 0.0 if (stop_after is not None and it >= stop_after) else U0 * np.cos(OM * t)
        s.set_instance_motion(0, lin_vel=[u, 0.0, 0.0])
        if mode == "rebuild":
            s.rebuild_geometry()
        elif mode == "refresh":
            s.refresh_wall_velocity()
        s.step()
    wall = time.time() - t0
    F = np.asarray(s.hydro_force_torque_reaction())[0][0]
    return (np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w()),
            np.asarray(s.get_p()), F, wall)


def bits_equal(a, b):
    return np.array_equal(a.view(np.int64), b.view(np.int64))


print("A. refresh_wall_velocity vs rebuild_geometry, %d steps of a time-varying wall velocity"
      % STEPS)
rb = run("rebuild")
rf = run("refresh")
names = ["u", "v", "w", "P"]
ok_a = True
for k in range(4):
    eq = bits_equal(rb[k], rf[k])
    d = float(np.abs(rb[k] - rf[k]).max())
    ok_a = ok_a and eq
    print("   %-2s  bitwise %-5s  max|diff| %.3e" % (names[k], str(eq), d))
# The FORCE is assembled by atomics over an unordered traversal, so it is documented as
# tolerance-reproducible, not bitwise -- compare it as such, and only the fields bitwise.
dF = float(np.abs(rb[4] - rf[4]).max()) / max(float(np.abs(rb[4]).max()), 1e-300)
ok_a = ok_a and dF < 1e-12
print("   reaction force  relative |diff| %.2e (atomics: tolerance-reproducible, not bitwise)"
      % dF)
print("     rebuild %s\n     refresh %s"
      % (np.array2string(rb[4], precision=10), np.array2string(rf[4], precision=10)))
print("   GATE A %s  [the SOLUTION is bitwise; the force agrees to the atomics floor]"
      % ("PASS" if ok_a else "FAIL"))

print("B. cost -- the CALL itself, timed in isolation, and the whole step for context")


def time_call(kind, reps=20, dt=5.0):
    s = make(dt, U0)
    s.step()
    for _ in range(3):                       # warm up
        s.set_instance_motion(0, lin_vel=[U0, 0.0, 0.0])
        getattr(s, kind)()
    t0 = time.time()
    for i in range(reps):
        s.set_instance_motion(0, lin_vel=[U0 * (1 + 1e-6 * i), 0.0, 0.0])
        getattr(s, kind)()
    return 1000 * (time.time() - t0) / reps


t_rb = time_call("rebuild_geometry")
t_rf = time_call("refresh_wall_velocity")
t_step = (1000 * rb[5] / STEPS) - t_rb
print("   rebuild_geometry            %7.1f ms" % t_rb)
print("   refresh_wall_velocity       %7.1f ms   -> the CALL is %.2fx cheaper" % (t_rf, t_rb / t_rf))
print("   a bare step() for scale      %7.1f ms   -> the driver's per-step total falls "
      "%.2fx (%.1f -> %.1f ms)"
      % (t_step, (t_step + t_rb) / (t_step + t_rf), t_step + t_rb, t_step + t_rf))
print("   (the refresh still rebuilds the momentum stencils, which is the bulk of a geometry "
      "rebuild; what it skips is re-sampling the SDF, the apertures and the pressure operator)")
ok_b = t_rf < t_rb
print("   GATE B %s" % ("PASS" if ok_b else "FAIL"))

def run_stop(refresh, dt=5.0, steps=STEPS):
    """Build the scene WITH a wall velocity (so uBc_ is really built), then stop it."""
    s = make(dt, U0)
    s.set_instance_motion(0, lin_vel=[0.0, 0.0, 0.0])
    if refresh:
        s.refresh_wall_velocity()
    for _ in range(steps):
        s.step()
    return (np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w()),
            np.asarray(s.get_p()))


def run_never(dt=5.0, steps=STEPS):
    s = make(dt, 0.0)                    # never any motion at all
    for _ in range(steps):
        s.step()
    return (np.asarray(s.get_u()), np.asarray(s.get_v()), np.asarray(s.get_w()),
            np.asarray(s.get_p()))


print("C. stopping the motion must clear the wall-velocity fields")
ref = run_never()
fixed = run_stop(refresh=True)
stale = run_stop(refresh=False)      # exhibits the failure rather than asserting it
ok_c = True
for k in range(4):
    eq = bits_equal(fixed[k], ref[k])
    ok_c = ok_c and eq
    print("   %-2s  stopped+refresh vs never-moved  bitwise %-5s  max|diff| %.3e   "
          "|  stopped WITHOUT refresh differs by %.3e"
          % (names[k], str(eq), float(np.abs(fixed[k] - ref[k]).max()),
             float(np.abs(stale[k] - ref[k]).max())))
exhibits = float(np.abs(stale[0] - ref[0]).max()) > 1e-6
ok_c = ok_c and exhibits
print("   GATE C %s  [zeroing the fields makes a stopped body identical to one that never moved; "
      "without it the baked-in wall velocity survives, which the third column exhibits]"
      % ("PASS" if ok_c else "FAIL"))
print("REFRESH-WALL GATE %s" % ("PASS" if (ok_a and ok_b and ok_c) else "FAIL"))
