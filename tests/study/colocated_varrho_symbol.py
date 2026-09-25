#!/usr/bin/env python
"""Step-operator model of the collocated variable-density step (doc/collocated_varrho_forces.md,
Appendix A) -- an INSTRUMENT, not a gate.

A dense, exact-solve, 2-D, h = 1 model of one step (rho0 = 1), unknowns (u, v, P). The step matrix
T maps (u^n, v^n, P^n) -> (u^{n+1}, v^{n+1}, P^{n+1}) exactly (dense inverse for the implicit
momentum operator, pseudo-inverse for the pressure Poisson), so its spectral radius is the per-step
growth factor of the scheme. Three schemes:

  "V8"  the retired WO-T step: the pressure and every force as a face acceleration added AFTER the
        implicit viscous solve (Basilisk centered.h). Non-incremental; with the rotational update
        (kappa = mu) its pressure obeys P^{n+1} = -4 kappa dt S P^n / rho -> radius 8 mu dt (2-D).
  "D2"  the predictor form with the ARITHMETIC centre->face map: not an adjoint pair, radius > 1
        at high contrast.
  "D3"  the chosen scheme: the predictor form with the MOMENTUM-weighted centre->face map
        (rho_L u_L + rho_R u_R)/(rho_L + rho_R), so M Gamma = -C^T and the radius is exactly 1.

Run: OMP_NUM_THREADS=8 python tests/study/colocated_varrho_symbol.py
Expected (the design note): V8 at dt 0.2: 1.6 = 8 mu dt; D2 at ratio 1e4 slightly above 1 at
dt 3 and 10 (the note quotes 1.0136 / 1.0145 for its shape); D3: 1.00000 everywhere.
"""
import itertools

import numpy as np


# ---------------------------------------------------------------- Appendix A, verbatim
def model(N, rho, mu, walls=False):
    n, I = N * N, np.eye(N * N)

    def sh(dx, dy):
        S = np.zeros((n, n))
        for x, y in itertools.product(range(N), repeat=2):
            S[x + N * y, (x + dx) % N + N * ((y + dy) % N)] = 1
        return S
    Sm, Sp = [sh(-1, 0), sh(0, -1)], [sh(1, 0), sh(0, 1)]
    o = [np.ones(n), np.ones(n)]
    if walls:
        o[1][:N] = 0.0                                          # closed y-faces at y = 0
    L = np.zeros((n, n))                                        # no-slip reflection at closed faces
    for a in range(2):
        for i in range(n):
            for S, op in ((Sm[a], o[a][i]), (Sp[a], (Sp[a] @ o[a])[i])):
                j = int(np.argmax(S[i]))
                if op:
                    L[i, j] += mu
                    L[i, i] -= mu
                else:
                    L[i, i] -= 2 * mu
    rf = [0.5 * (rho + Sm[a] @ rho) for a in range(2)]
    G = [I - Sm[a] for a in range(2)]
    D = [Sp[a] - I for a in range(2)]
    Pi = [0.5 * (I + Sm[a]) for a in range(2)]
    R = [0.5 * (I + Sp[a]) for a in range(2)]
    Lpi = np.linalg.pinv(sum(D[a] @ np.diag(o[a] / rf[a]) @ G[a] for a in range(2)), rcond=1e-12)
    return locals()


def step_matrix(m, dt, kap, scheme):                            # "V8" | "D2" | "D3" (chosen)
    n, I, rho, o, rf, G, D, Pi, R, L, Lpi = (m[k] for k in "n I rho o rf G D Pi R L Lpi".split())
    Ai, Z, T = np.linalg.inv(np.diag(rho / dt) - L), np.zeros((n, n)), np.zeros((3 * n, 3 * n))
    Fp = [np.diag(o[a] / rf[a]) @ G[a] for a in range(2)]      # face pressure accel per unit P
    us, uf = [], []
    for a in range(2):
        blk = [Z, Z, Z if scheme == "V8" else -Ai @ np.diag(rho) @ R[a] @ Fp[a]]
        blk[a] = Ai @ np.diag(rho / dt)
        us.append(blk)
        C2F = np.diag(1 / rf[a]) @ Pi[a] @ np.diag(rho) if scheme == "D3" else Pi[a]
        f = [np.diag(o[a]) @ C2F @ b for b in blk]
        if scheme == "V8":
            f[2] = f[2] - dt * Fp[a]
        uf.append(f)
    div = [D[0] @ uf[0][k] + D[1] @ uf[1][k] for k in range(3)]
    phi = [Lpi @ d for d in div]
    for a in range(2):
        for k in range(3):
            acc = -Fp[a] @ phi[k] + (-dt * Fp[a] if (scheme == "V8" and k == 2) else 0)
            T[a * n:(a + 1) * n, k * n:(k + 1) * n] = (us[a][k] + R[a] @ acc if scheme == "V8"
                                                       else us[a][k] - R[a] @ Fp[a] @ phi[k])
    for k in range(3):
        T[2 * n:, k * n:(k + 1) * n] = phi[k] / dt - kap * div[k] + (I if k == 2 else 0)
    return T


def radius(T):
    return np.max(np.abs(np.linalg.eigvals(T)))


# ---------------------------------------------------------------- drivers
def density(N, kind, ratio, seed=1):
    """Cell density field (x fastest, index x + N*y)."""
    x, y = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5, indexing="xy")
    if kind == "uniform":
        r = np.ones((N, N))
    elif kind == "slab":
        r = np.where((y >= N / 4) & (y < 3 * N / 4), ratio, 1.0)
    elif kind == "disk":
        r = np.where((x - 0.47 * N) ** 2 + (y - 0.53 * N) ** 2 < (0.3 * N) ** 2, ratio, 1.0)
    else:  # "random": a random two-phase blob field, smoothed then thresholded
        rng = np.random.default_rng(seed)
        f = rng.standard_normal((N, N))
        for _ in range(2):
            f = 0.2 * (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) +
                       np.roll(f, -1, 1))
        r = np.where(f > 0.0, ratio, 1.0)
    return r.reshape(-1).astype(np.float64)


def main():
    np.set_printoptions(precision=6)
    N, mu = 12, 1.0
    print("1. V8 (retired), uniform rho, periodic, kappa = mu: radius vs 8 mu dt")
    m = model(N, density(N, "uniform", 1.0), mu)
    for dt in (0.01, 0.1, 0.2, 1.0):
        print(f"   dt {dt:<5g} radius {radius(step_matrix(m, dt, mu, 'V8')):.6f}   "
              f"(8 mu dt = {8 * mu * dt:.4f})")
    print("2. D2 (arithmetic centre->face, predictor form), ratio 1e4, kappa = mu")
    for kind in ("disk", "random"):
        for walls in (False, True):
            m = model(N, density(N, kind, 1e4), mu, walls)
            rr = [radius(step_matrix(m, dt, mu, "D2")) for dt in (3.0, 10.0)]
            print(f"   {kind:6s} walls={walls!s:5s}  dt 3: {rr[0]:.6f}   dt 10: {rr[1]:.6f}")
    print("3. D3 (chosen: momentum-weighted centre->face), kappa = mu and 0")
    worst = 0.0
    for kind in ("uniform", "slab", "disk", "random"):
        for walls in (False, True):
            for ratio in (10.0, 1e3, 1e6):
                m = model(N, density(N, kind, ratio), mu, walls)
                for dt in (0.01, 1.0, 100.0, 1e4):
                    for kap in (mu, 0.0):
                        worst = max(worst, radius(step_matrix(m, dt, kap, "D3")))
    print(f"   worst radius over uniform/slab/disk/random x periodic/walled x ratio 10..1e6 x "
          f"dt 0.01..1e4 x kappa {{mu, 0}}: {worst:.9f}")


if __name__ == "__main__":
    main()
