# Variable density in `peclet.flow`: momentum, projection scaling, and the pressure driver

Status: implemented + validated (2026-07-04, host-openmp backend). Multiphysics Phase 5 — see
`../../docs/MULTIPHYSICS_PLAN.md`; companion note: `variable_viscosity_projection.md` (Phase 4,
shares the FaceProps/rotational machinery).

## 1. The scheme

Enabled by `set_density_mode("variable")` (or automatically by a closure targeting `"rho"`).
The scalar `rho_` (`set_rho`) becomes the **reference density ρ₀**; the registered cell field
`"rho"` carries ρ(x). Staggered grid only (v1).

**Face density.** One arithmetic face mean is used *everywhere*:
`ρ_f(i) = ½(ρ(i) + ρ(i−s_c))` at the staggered face of velocity component c. This three-way
consistency — the momentum time term, the body force, and the projection — is what makes discrete
hydrostatic balance exact (§3). (Arithmetic is the physically right mean for the inertia of the
staggered control volume: mass is volume-additive. Contrast the *viscosity* face mean, where
harmonic is right for stress continuity.)

**Momentum** (`VarFaceProps` in `face_props.hpp`, `buildRhsVar` / `buildAdvStencilVar` siblings in
`flow_ibm.hpp` — the validated constant kernels are never edited):

```
idiag(i)   = ρ_f(i)/dt                      (time term, diffusion stencil diagonal + RHS)
advection  = ρ_f(i) · adv(u)                (explicit weight and the implicit-FOU operator weight)
body force = ½(f(i) + f(i−s_c))             (cell-force field face-interpolated → a ρg cell field
                                             becomes ρ_f·g at the velocity location)
```

**Projection.** The exact projection solves `∇·(open·(dt/ρ_f)∇δp) = ∇·(open·u*)`. Substituting
`δp = (ρ₀/dt)·φ` keeps the solver's φ-scaling and yields:

```
operator face coefficient   c_f = open_f · ρ₀/ρ_f      (buildRhoCoeff, mac_pressure.hpp)
velocity correction         u_f −= (ρ₀/ρ_f) ∂_f φ      (projectCorrectVar)
pressure update             P += (ρ₀/dt)·φ − μ_rot·∇·u*   (unchanged form; ct = rho_/dt IS ρ₀/dt)
divergence                  unchanged (flux openness only)
```

With ρ ≡ ρ₀ every factor is exactly 1.0 in floating point → the constant-density scheme is
recovered identically (validated: relative du = 2e-14 vs the constant solver on an immersed-
cylinder flow).

**Plumbing** (in `project()`): ρ ghosts filled (`fillPropGhosts` — periodic/halo + zero-gradient on
domain-BC faces), ρ bridged to the g=1 MG block *including its ghost ring* (`copyBlockShifted`,
offset G−1 — the face means at the first inner cell need a valid neighbour), coefficients formed on
the inner cells, and handed to `CutcellMG::setOpenness`, whose per-level ghost fill + boundary
re-imposition + rediscretized averaging treat them exactly like openness ("the coefficient rides
the openness rails" — zero CutcellMG changes). Rebuilt every step (ρ may be closure/transport
driven); `chebBoundsSet_` invalidated on every rebuild.

## 2. The pressure driver: Chebyshev, not PCG (an empirical finding)

**MG-PCG stalls on the ρ-scaled coefficient operator.** Observed on the hydrostatic test at
density ratio 3: PCG hits 5000 iterations without converging (residual plateau, velocity error
~4e-3), while the **Chebyshev driver converges in ~20 iterations on the identical system** — and
the coefficient fields were verified bit-identical to the openness in the uniform-ρ control (where
PCG also stalls the moment the *solve path* is the per-step rebuild... no: uniform-ρ with layered
*force* converges; the stall correlates with **layered coefficients**, not with the rebuild).

Interpretation: the V-cycle preconditioner (transfers built/validated for *geometric openness*)
loses the SPD-preserving structure conjugate gradients requires when the level fields are ρ-scaled
coefficients; Chebyshev only needs real spectrum bounds (re-estimated after every coefficient
rebuild) and is immune. Consequence, implemented in `setDensityMode`:

- **Chebyshev is the default pressure driver under variable density** (an explicit
  `set_pressure_pcg` / `set_pressure_chebyshev` after `set_density_mode` still wins).
- Practical accuracy: the first solve on a fresh field leaves a transient velocity residual
  ~1e-6·(g·dt) (the driver's stopping estimate); the incremental scheme absorbs it within a few
  steps and the steady state is machine-exact (§3).
- **Follow-up for the MG (deferred):** make the transfer pair provably symmetric for arbitrary
  positive coefficient fields (or add a coefficient-aware Galerkin option) so PCG becomes usable;
  until then Chebyshev iteration counts stay flat (≤32 observed at ratio 10³) so nothing is lost.

## 3. Validation (host-openmp; `tests/kokkos/test_vardensity_projection.cpp` +
`tests/study/rayleigh_taylor.py`)

| test | result |
|---|---|
| **Hydrostatic acid test** (two-layer ρ at rest + gravity closure `force_z = −g·ρ`; walls ±z): ratio **3** and ratio **1000**, inviscid and μ=0.01 | steady max\|u\| **~1e-16** (machine zero), discrete ∂P/∂z = −ρ_f·g to **~4e-16**; Chebyshev ≤ 32 its at ratio 10³; transient ≤ 9e-6 (first-solve accuracy, absorbed) |
| **Uniform-ρ reduction** (ρ ≡ ρ₀, immersed cylinder, body-force Stokes) | rel du = **2e-14**, dp = 4e-12 vs the constant solver |
| **Rayleigh–Taylor** (ratio 3, Atwood 0.5; transported phase fraction → ρ closure → gravity closure — the full two-phase chain) | amplitude 1.5 → 19.5 cells (**13×**), monotone; early growth ≈ 0.74·√(Agk) (viscous + finite-interface damping) |
| **Single-phase regression** | bit-exact (+0.00%, identical iteration counts) — `varRho_=false` executes the original kernels |

Why the hydrostatic test is exact (and what it guards): from rest, `w* = −g·dt` uniformly (the
face force −g·ρ_f divided by the face inertia ρ_f/dt — same ρ_f), so the interior divergence
vanishes and the wall-column divergence is exactly projectable; the correction returns u = 0 and
`P` accumulates `∂P = −ρ_f·g` with the projection's ρ_f. Any mean mismatch (e.g. harmonic
projection ρ vs arithmetic momentum ρ, or a cell-centred instead of face-interpolated force)
breaks the telescoping and leaves a permanent spurious velocity — this test fails loudly.

## 4. Limitations / deferred

- **Staggered only** (collocated `set_density_mode` throws): the collocated correction path
  (wall-aware transpose maps) needs its own 1/ρ treatment.
- **Outflow + varRho**: `bcCorrectOutflow` corrects the outflow face without the 1/ρ_f factor —
  fine when the outflow region has ρ ≈ uniform; revisit with a two-phase outflow case.
- **Boussinesq vs varRho**: for small Δρ/ρ prefer the Phase-3 Boussinesq closure (cheaper: no
  per-step operator rebuild).
- The rotational-term policy under simultaneous variable μ follows Phase 4
  (`set_variable_rotational`; constant-μ default term is valid for variable ρ — the μ-part of the
  stress is what the rotational correction concerns, cf. Guermond & Salgado [2] using exactly this
  constant-coefficient philosophy for variable density).
- MPI/CUDA validation deferred with the plan phases (the ρ bridge and ghost fills use the halo
  paths, and `minMuInner` already allreduces, so the structure is MPI-ready).
- PCG-on-coefficients MG follow-up (§2).

## References

[1] J.-L. Guermond, P. Minev, J. Shen, *An overview of projection methods for incompressible
flows*, Comput. Methods Appl. Mech. Engrg. 195 (2006) 6011–6045.

[2] J.-L. Guermond, A. Salgado, *A splitting method for incompressible flows with variable density
based on a pressure Poisson equation*, J. Comput. Phys. 228 (2009) 2834–2846.

[3] Phase-4 companion: `variable_viscosity_projection.md` (Deteix–Yakoubi rotational-term
references and the FaceProps machinery).
