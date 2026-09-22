"""FRESH-CELL GATE: does a body that physically MOVES through the grid spike the drag?

A sphere oscillating at vanishing amplitude has an exact answer (Stokes 1851) and, in this solver,
a second reference that is exact by construction: the LINEARISED run, where the boundary condition
oscillates but the geometry never moves. The difference between the two is everything the motion
itself introduces -- which in a sharp-interface IBM is the textbook source of spurious force
oscillations (Lee, Kim, Choi & Yang 2011; Seo & Mittal 2011): cells emerging from the body inherit
a solid state rather than a fluid one, and the stencil changes abruptly as the interface crosses a
face.

TWO METRICS, deliberately:
  * RMS |F^{n+1} - 2F^n + F^{n-1}| / |F1| -- the Seo & Mittal (2011) Eq. (12) second-difference
    measure, the one the literature actually uses (Martins et al. 2017 Eq. 54 adopts it verbatim).
    Comparable in FORM to published numbers; the absolute value is not, since we normalise by the
    fundamental amplitude rather than their C_PD.
  * the residual after fitting 5 harmonics of the driving frequency -- flat spectral gain, no
    explicit dt dependence of its own, and physically interpretable as "the part of the force that
    is not at the driving frequency". Kept because the 2-delta operator's own (omega*dt)^2 rolloff
    makes its temporal exponent metric-dependent rather than physical.

THREE LEGS: refine dt at fixed grid (the artefact should NOT improve -- it is spatial); refine h at
fixed physics; sweep the amplitude (hence the number of cell crossings per cycle). Each run is
repeated with set_fresh_cell_seed off and on.
"""

import os, time, numpy as np
from peclet import flow as sdflow
RHO, MU = 1.0, 0.1; NU = MU/RHO
DR, NPER, FITP, NH = 1.0, 3, 2, 5
KN_R, KI_I, KI_R = 16, 2, 17

def run(N, R, SPP, AR, seed, moving=True):
    delta=DR*R; om=2*NU/delta**2; A=AR*R; U0=A*om
    T=2*np.pi/om; dt=T/SPP; x0=0.5*N
    ni=np.array([1,-1,-1],dtype=np.int32); nr=np.zeros(KN_R); nr[0]=R; nr[14]=1.0; nr[15]=1.0
    ii=np.zeros((1,KI_I),dtype=np.int32); ii[0]=(0,-1)
    ir=np.zeros((1,KI_R)); ir[0,0:3]=(x0,0.5*N,0.5*N); ir[0,6]=1.0; ir[0,7]=1.0
    s=sdflow.Solver(N,N,N); s.set_rho(RHO); s.set_mu(MU); s.set_dt(dt); s.set_advection(False)
    s.set_velocity_solver_params(100); s.set_pressure_solver_params(25)
    s.set_pressure_multigrid(True, levels=4)
    s.set_scene(ni,nr,ii.ravel(),ir.ravel(),periodic=True)
    s.set_fresh_cell_seed(bool(seed))
    s.set_instance_motion(0, lin_vel=[U0,0.0,0.0]); s.set_solid_from_scene(True)
    ts,Fs=[],[]; t0=time.time()
    for it in range(SPP*NPER):
        t=(it+1)*dt; u=U0*np.cos(om*t)
        if moving:
            s.set_instance_transform(0,[x0+A*np.sin(om*t),0.5*N,0.5*N])
            s.set_instance_motion(0, lin_vel=[u,0.0,0.0]); s.rebuild_geometry()
        else:
            s.set_instance_motion(0, lin_vel=[u,0.0,0.0]); s.refresh_wall_velocity()
        s.step(); ts.append(t); Fs.append(float(np.asarray(s.hydro_force_torque_reaction())[0][0][0]))
    ts=np.array(ts); Fs=np.array(Fs); m=ts>=(NPER-FITP)*T
    t,F=ts[m],Fs[m]
    cols=[np.ones_like(t)]
    for k in range(1,NH+1): cols += [np.cos(k*om*t), np.sin(k*om*t)]
    Ab=np.column_stack(cols); c,*_=np.linalg.lstsq(Ab,F,rcond=None)
    res=F-Ab@c; F1=np.hypot(c[1],c[2])
    # Seo & Mittal (2011) Eq. 12 metric: RMS of |F^{n+1} - 2F^n + F^{n-1}|, normalised the same
    # way the force is (here by the fundamental amplitude, so it is dimensionless and comparable
    # ACROSS OUR OWN runs; the absolute value is not comparable to their C_PD normalisation).
    d2 = np.abs(F[2:] - 2*F[1:-1] + F[:-2])
    return dict(F1=F1, p2p=(res.max()-res.min())/F1, rms=res.std()/F1,
                d2=float(np.sqrt((d2**2).mean()))/F1, wall=time.time()-t0, A=A, U0=U0)

def line(tag, r, ref=None):
    e = "" if ref is None else "  dF1 %+6.2f%%" % (100*(r['F1']/ref-1))
    print("   %-26s |F1|=%.5e  p2p=%.3e  rms=%.3e  RMS(F2d)/|F1|=%.3e  (%3.0fs)%s"
          % (tag, r['F1'], r['p2p'], r['rms'], r['d2'], r['wall'], e))

which = os.environ.get("LEG", "dt")
if which == "dt":
    print("TIME-STEP leg: N=96, R=9.6 h, A/R=0.25 (A=2.40 h)  -- grid crossings FIXED")
    st = run(96, 9.6, 200, 0.25, 0, moving=False)
    line("static (linearised) ref", st)
    for spp in (100, 200, 400):
        for sd in (0, 1):
            line("SPP=%3d seed=%d" % (spp, sd), run(96, 9.6, spp, 0.25, sd), st['F1'])
elif which == "h":
    print("GRID leg: R/L=0.1, delta/R=1, A/R=0.25, SPP=200")
    for N, R in ((64, 6.4), (96, 9.6), (128, 12.8)):
        st = run(N, R, 200, 0.25, 0, moving=False)
        line("N=%3d static ref" % N, st)
        for sd in (0, 1):
            line("N=%3d moving seed=%d" % (N, sd), run(N, R, 200, 0.25, sd), st['F1'])
else:
    print("AMPLITUDE leg: N=96, R=9.6 h, SPP=200")
    for AR in (0.05, 0.10, 0.25, 0.50, 1.00):
        st = run(96, 9.6, 200, AR, 0, moving=False)
        for sd in (0, 1):
            line("A/R=%.2f seed=%d" % (AR, sd), run(96, 9.6, 200, AR, sd), st['F1'])
