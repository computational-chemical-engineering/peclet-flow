# A4 exploration — prototype results (2026-10-02)

Run with the suite venv, `OMP_NUM_THREADS=4`. These are 2-D NumPy prototypes and decide
questions; they are not production code.

## 1. Immersed-surface scalar BCs on a pipe cross-section (`disc_bc.py`, `robin.py`)

**Set-up.**
- A disc of radius R = 1 (fluid inside) sits in a Cartesian box.
- Three random grid offsets per resolution.
- Errors are relative and RMS over the offsets. ND = cells per diameter.
- Exact references:
  - Neumann eigenvalue j'₁₁² = 3.389957;
  - Taylor–Aris −⟨u′b⟩ = 1/48;
  - Dirichlet decay j₀₁² = 5.783186;
  - Graetz Nu_F = Pe·λR (1F1 root): 3.576741 at Pe = 10, 3.656785 at Pe = 1000;
  - Robin x J₁(x) = Bi J₀(x).

| gate | scheme | ND=16 | 32 | 64 | 128 | order |
|---|---|---|---|---|---|---|
| Neumann eig | aperture FV, κ storage | 3.9e-3 | 9.9e-4 | 2.4e-4 | 6.2e-5 | 2.0 |
| Neumann eig | aperture FV, unit storage (**flow today**) | 2.2e-1 | 1.1e-1 | 8.4e-2 | 2.8e-2 | ~1, erratic |
| Neumann eig | staircase | 5.0e-2 | 2.2e-2 | 9.7e-3 | 4.3e-3 | 1.2 |
| Taylor–Aris 1/48 | aperture FV, κ storage/source | 4.7e-3 | 1.2e-3 | 3.2e-4 | 8.1e-5 | 2.0 |
| Taylor–Aris 1/48 | aperture FV, unit storage | 5.0e-1 | 2.4e-1 | 1.2e-1 | 5.8e-2 | 1.0 |
| Dirichlet decay | per-cell mask + apertures (**flow today**) | 2.6e-1 | 1.6e-1 | 8.2e-2 | 4.2e-2 | 0.7–1 |
| Dirichlet decay | linear ghost (Gibou 2002, symmetric) | 3.7e-3 | 9.3e-4 | 2.3e-4 | 5.8e-5 | 2.0 |
| Dirichlet decay | quadratic ghost (Robust-Scaled scheme 0) | 4.4e-3 | 1.2e-3 | 3.0e-4 | 7.7e-5 | 2.0 |
| Dirichlet decay | cut-cell FV, wall flux to centroid | 5.9e-2 | 2.9e-2 | 1.4e-2 | 7.1e-3 | 1.0 |
| Graetz Pe=1000 | linear ghost | 4.5e-3 | 1.1e-3 | 2.8e-4 | 7.0e-5 | 2.0 |
| Graetz Pe=1000 | quadratic ghost | 5.4e-3 | 1.4e-3 | 3.4e-4 | 8.6e-5 | 2.0 |
| Graetz Pe=1000 | mask + apertures | 1.9e-1 | 1.2e-1 | 6.1e-2 | 3.1e-2 | 0.7–1 |

Graetz at Pe = 10 behaves the same.

**Robin (Bi = 0.1, 1, 10, 100).**
- Cut-cell FV with the series wall resistance (d/D + 1/k) is **first order at every Bi**: 3e-3 to
  7.5e-3 at ND = 128.
- The linear ghost with c_Γ eliminated through the normal Robin relation is erratic (p from −1 to 3.7).
  Its Neumann limit is the first-order staircase.
- No simple compact scheme found is second order at all Bi.

## 2. Tracer estimators (`tracers.py 2.0 0.1 20000 1500`)

**Flow and reference.**
- Cellular flow with mean through-flow: ψ = y + 2 sin x sin y. It has closed recirculation cells.
- D_m = 0.1, so Pe = U·L/D_m = 63 and t_D = L²/D_m = 395.
- Reference from the spectral Brenner closure: D*_xx = 1.761961, D*_yy = 0.226360.
- 2·10⁴ tracers, Heun drift plus Euler–Maruyama noise, dt = 0.01, T = 1500.

| estimator | value | rel. err |
|---|---|---|
| MSD slope over [375, 750] | 1.799 ± 0.032 | +2.1 % (pre-asymptotic, 1σ) |
| MSD slope over [750, 1500] | 1.759 ± 0.032 | −0.2 % |
| Green–Kubo, velocity autocorrelation only | 2.70 ± 0.02 | **+53 %** (misses the u′–noise cross-covariance) |
| first passage over 1 / 4 / 16 periods | 1.378 / 1.663 / 1.738 | −22 % / −5.6 % / −1.4 % (pre-asymptotic, ~1/n) |

The running local slope ½ dVar/dt reaches its plateau only after ~2 t_D.
