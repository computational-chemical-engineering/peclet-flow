"""2-D prototype: dispersion estimators from Brownian tracer trajectories, against an exact
Eulerian reference (Brenner closure solved spectrally).

Flow: periodic cellular flow with mean through-flow on [0, 2pi)^2,
    psi = U0*y + A*sin(x)*sin(y)   ->  u = U0 + A sin x cos y,  v = -A cos x sin y
A > U0 gives closed recirculation cells (an analogue of hold-up regions in a bed).

Reference: B periodic, mean zero,  u.grad B - D lap B = <u> - u   (one per direction)
           D*_aa = D <|e_a + grad B_a|^2>   (no walls -> <grad B> = 0)

Estimators compared on the same trajectories:
  MSD   : late-time slope 1/2 dVar(x)/dt over [t1, t2], error by batch means over particles
  GK    : D + int_0^T <u'(0) u'(tau)> dtau  (velocity autocorrelation only)  -- misses the
          velocity-noise cross-covariance when u_x depends on x
  FPT   : first passage over n periods: D = L^2 Var(tau) / (2 <tau>^3)  (inverse Gaussian)
"""
import numpy as np
import scipy.sparse.linalg as spla
import sys, time

TWO_PI = 2 * np.pi


def closure_reference(U0, A, D, N=96):
    k = np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    x = TWO_PI * np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = U0 + A * np.sin(X) * np.cos(Y)
    v = -A * np.cos(X) * np.sin(Y)

    def L(bvec):
        b = bvec.reshape(N, N)
        bh = np.fft.fft2(b)
        bx = np.real(np.fft.ifft2(1j * KX * bh)); by = np.real(np.fft.ifft2(1j * KY * bh))
        lap = np.real(np.fft.ifft2(-K2 * bh))
        r = u * bx + v * by - D * lap
        return (r - r.mean()).ravel() + b.mean()  # pin the mean

    def Pinv(r):
        rh = np.fft.fft2(r.reshape(N, N))
        with np.errstate(divide="ignore", invalid="ignore"):
            bh = np.where(K2 > 0, rh / (D * K2), rh)
        return np.real(np.fft.ifft2(bh)).ravel()

    op = spla.LinearOperator((N * N, N * N), matvec=L)
    P = spla.LinearOperator((N * N, N * N), matvec=Pinv)
    out = []
    for comp, rhs in (("xx", u.mean() - u), ("yy", v.mean() - v)):
        b, info = spla.gmres(op, rhs.ravel(), M=P, rtol=1e-12, restart=200, maxiter=50)
        assert info == 0
        bh = np.fft.fft2(b.reshape(N, N))
        bx = np.real(np.fft.ifft2(1j * KX * bh)); by = np.real(np.fft.ifft2(1j * KY * bh))
        if comp == "xx":
            out.append(D * np.mean((1 + bx) ** 2 + by**2))
        else:
            out.append(D * np.mean(bx**2 + (1 + by) ** 2))
    return out, u.mean()


def vel(x, y, U0, A):
    return U0 + A * np.sin(x) * np.cos(y), -A * np.cos(x) * np.sin(y)


def simulate(U0, A, D, Np, dt, T, nper_fpt=(1, 4, 16), seed=0, nsave=400):
    rng = np.random.default_rng(seed)
    x = rng.random(Np) * TWO_PI; y = rng.random(Np) * TWO_PI  # uniform = invariant measure
    x0 = x.copy()
    nsteps = int(round(T / dt))
    every = max(1, nsteps // nsave)
    ts, X, Y = [], [], []
    U = []  # u' samples for GK (saved at the same cadence)
    fpt = {n: np.full(Np, np.nan) for n in nper_fpt}
    s = np.sqrt(2 * D * dt)
    for it in range(nsteps):
        if it % every == 0:
            ts.append(it * dt); X.append(x - x0); Y.append(y.copy())
            U.append(vel(x, y, U0, A)[0])
        u1, v1 = vel(x, y, U0, A)
        u2, v2 = vel(x + u1 * dt, y + v1 * dt, U0, A)  # Heun drift
        x = x + 0.5 * (u1 + u2) * dt + s * rng.standard_normal(Np)
        y = y + 0.5 * (v1 + v2) * dt + s * rng.standard_normal(Np)
        t = (it + 1) * dt
        for n in nper_fpt:
            hit = np.isnan(fpt[n]) & (x - x0 >= n * TWO_PI)
            fpt[n][hit] = t
    return np.array(ts), np.array(X), np.array(U), fpt


def est_msd(ts, X, t1, t2, nb=20):
    i1, i2 = np.searchsorted(ts, t1), np.searchsorted(ts, t2)
    def one(Xs):
        v = Xs.var(axis=1)
        # least-squares slope of Var over [t1, t2]
        tt = ts[i1:i2 + 1]; vv = v[i1:i2 + 1]
        return 0.5 * np.polyfit(tt, vv, 1)[0]
    full = one(X)
    b = [one(X[:, k::nb]) for k in range(nb)]
    return full, np.std(b, ddof=1) / np.sqrt(nb)


def est_gk(ts, U, D, Umean, tmax, nb=20):
    up = U - Umean
    dt = ts[1] - ts[0]
    nl = int(tmax / dt)
    def one(u):
        # time-origin averaged autocorrelation (stationary: uniform start = invariant measure)
        C = np.array([np.mean(u[: len(u) - l] * u[l:]) for l in range(nl)])
        return D + dt * (C.sum() - 0.5 * C[0])
    full = one(up)
    b = [one(up[:, k::nb]) for k in range(nb)]
    return full, np.std(b, ddof=1) / np.sqrt(nb)


def est_fpt(tau, L, nb=20):
    def one(t):
        t = t[~np.isnan(t)]
        return L**2 * t.var() / (2 * t.mean() ** 3), np.isnan(t).mean()
    full, _ = one(tau)
    miss = np.isnan(tau).mean()
    b = [one(tau[k::nb])[0] for k in range(nb)]
    return full, np.std(b, ddof=1) / np.sqrt(nb), miss


if __name__ == "__main__":
    U0, A = 1.0, float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    Np = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0
    dt = 0.01
    (Dxx, Dyy), Um = closure_reference(U0, A, D)
    tD = TWO_PI**2 / D
    print(f"U0={U0} A={A} D={D}  Pe=U0*L/D={U0*TWO_PI/D:.0f}  t_D=L^2/D={tD:.0f}  T={T}")
    print(f"reference (spectral closure): D*_xx = {Dxx:.6f}  D*_yy = {Dyy:.6f}  <u> = {Um:.6f}")
    t0 = time.time()
    ts, X, U, fpt = simulate(U0, A, D, Np, dt, T)
    print(f"simulated {Np} tracers x {int(T/dt)} steps in {time.time()-t0:.0f} s")
    # running estimates: apparent D(t) = Var/2t and the local slope -> preasymptotic diagnostics
    v = X.var(axis=1)
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        i = min(len(ts) - 1, int(frac * (len(ts) - 1)))
        j = max(1, i // 2)
        print(f"  t={ts[i]:7.1f} ({ts[i]/tD:5.2f} t_D): Var/2t = {v[i]/(2*ts[i]):.4f}   "
              f"slope[t/2,t] = {0.5*(v[i]-v[j])/(ts[i]-ts[j]):.4f}")
    for (t1, t2) in ((0.25 * T, 0.5 * T), (0.5 * T, T)):
        m, e = est_msd(ts, X, t1, t2)
        print(f"MSD slope [{t1:.0f},{t2:.0f}] : {m:.4f} +- {e:.4f}   rel.err {m/Dxx-1:+.3%}")
    for tm in (0.05 * T, 0.2 * T):
        g, e = est_gk(ts, U, D, Um, tm)
        print(f"GK  (int to {tm:.0f})      : {g:.4f} +- {e:.4f}   rel.err {g/Dxx-1:+.3%}")
    for n, tau in fpt.items():
        f, e, miss = est_fpt(tau, n * TWO_PI)
        print(f"FPT over {n:2d} periods     : {f:.4f} +- {e:.4f}   rel.err {f/Dxx-1:+.3%}  (unarrived {miss:.1%})")
