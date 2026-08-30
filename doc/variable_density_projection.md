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

## 3. Validation (`tests/kokkos/test_vardensity_projection.cpp` + `tests/study/rayleigh_taylor.py`;
host-openmp **and nvidia-cuda** since 2026-08-30 — see §3.1 for the multi-rank gates)

| test | result |
|---|---|
| **Hydrostatic acid test** (two-layer ρ at rest + gravity closure `force_z = −g·ρ`; walls ±z): ratio **3** and ratio **1000**, inviscid and μ=0.01 | steady max\|u\| **~1e-16** (machine zero), discrete ∂P/∂z = −ρ_f·g to **~4e-16**; Chebyshev ≤ 32 its at ratio 10³; transient ≤ 9e-6 (first-solve accuracy, absorbed) |
| **Uniform-ρ reduction** (ρ ≡ ρ₀, immersed cylinder, body-force Stokes) | rel du = **2e-14**, dp = 4e-12 vs the constant solver |
| **Rayleigh–Taylor** (ratio 3, Atwood 0.5; transported phase fraction → ρ closure → gravity closure — the full two-phase chain) | amplitude 1.5 → 19.5 cells (**13×**), monotone; early growth ≈ 0.74·√(Agk) (viscous + finite-interface damping) |
| **Single-phase regression** | bit-exact (+0.00%, identical iteration counts) — `varRho_=false` executes the original kernels |

**CUDA (2026-08-30).** The same two files on the `nvidia-cuda` backend: hydrostatic steady
max\|u\| **3.99e-17** (ratio 3) and **2.75e-17** (ratio 1000), ∂P/∂z error 7.40e-16 / 3.41e-16;
uniform-ρ reduction rel du 5.01e-14; and the Rayleigh–Taylor amplitude history is identical to the
host-openmp run to every printed digit — `1.50 → 1.87 → 3.13 → 5.54 → 9.30 → 14.16 → 19.52` (13.0×),
monotone, on both backends.

Why the hydrostatic test is exact (and what it guards): from rest, `w* = −g·dt` uniformly (the
face force −g·ρ_f divided by the face inertia ρ_f/dt — same ρ_f), so the interior divergence
vanishes and the wall-column divergence is exactly projectable; the correction returns u = 0 and
`P` accumulates `∂P = −ρ_f·g` with the projection's ρ_f. Any mean mismatch (e.g. harmonic
projection ρ vs arithmetic momentum ρ, or a cell-centred instead of face-interpolated force)
breaks the telescoping and leaves a permanent spurious velocity — this test fails loudly.

## 3.1 Multi-rank (VoF rung V-1 / WO-A, 2026-08-30)

`tests/kokkos_mpi/test_vardensity_mpi.cpp` (+ `test_varmu_mpi.cpp` for the Phase-4 companion), np =
1, 2, 4 on host-openmp **and** nvidia-cuda. Each run compares the distributed solver against a
full-grid single-rank reference built on rank 0 with the identical configuration.

| configuration | measured |
|---|---|
| **`walls-z`** — the hydrostatic acid test, ratio 1000, walls ±z, 32×16×16 | max\|u\| **2.9–3.1e-17** and ∂P/∂z error **5.7e-16** at every np; max\|u_dist − u_ref\| ≤ **6.0e-17**, max\|P_dist − P_ref\| ≤ **2.8e-14** (P itself is O(800)) |
| **`jump-x`** — a SHARP ratio-1000 ρ jump on the *cut* axis (so the jump sits exactly on a rank boundary and the face coefficient ρ₀/ρ_f there is assembled from an exchanged ghost), periodic, body-force driven | max\|u_dist − u_ref\| **2.5e-19 … 3.3e-19**, ΔP ≤ **1.4e-17**; **Chebyshev V-cycle count identical at np = 1, 2, 4 for all 20 steps** |
| **`couette-y`** — 10× μ jumps, harmonic face mean, walls + moving lid | analytic error **0.0129 %** at every np; rel du **0** (np=1) / **3.5e-16** (np=2) / **5.9e-16** (np=4) |
| **`per-x`** — 10× μ jump on the cut axis, periodic | rel du **1.2e-15** (np=2) / **2.0e-15** (np=4) |

**The Chebyshev bounds path is decomposition-independent** (the WO-A gate). On `jump-x` — a
non-degenerate solve — the per-step V-cycle count is *identical* across np = 1, 2, 4 and across
OpenMP thread counts 1, 2, 4, 8. The ±1–2 scatter one sees on `walls-z` from step ~7 onward is not
an MPI effect: once the hydrostatic state has reached machine zero, the driver's own `r0` is
round-off noise and `maxabs(r) < rtol·r0` is a knife edge — at *fixed* np = 1 the sequence already
changes with the thread count alone (steps 7.. read `15,13,13,…` at 1 thread, `16,13,14,…` at 2,
`16,14,14,13,…` at 8). The MPI ctest therefore gates the count over the non-degenerate steps only.

**Bit-exactness, precisely.** np = 1 is bitwise identical (0.000e+00 on every field). np > 1 cannot
be bitwise by construction: Chebyshev's bound estimation (`CutcellMG::dot`) and `removeMean` both go
through an `MPI_SUM` allreduce whose summation order is a function of the rank count. The measured
np>1 residues above (1e-19…1e-16 on u, ≤3e-14 on P) *are* that floor — and on CUDA at np = 2 the
walls-z case happened to come out bitwise identical anyway.

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
- ~~MPI/CUDA validation deferred~~ — **done, §3 + §3.1** (VoF rung V-1 / WO-A). Two limitations were
  found while gating it, both in the *domain-BC* machinery rather than in the variable-ρ/μ path
  itself, and both are OPEN:
  - **Per-face domain BCs are not rank-aware.** `applyVelocityBcCompTo` (and the pressure-openness
    BC) impose the BC on *every* rank's own block faces — there is no `touchesGlobalFace` ownership
    test, unlike the transported-scalar BCs (`applyScalarBc`). If the decomposition cuts a walled
    axis, the domain silently splits into independent sub-domains. Measured on the hydrostatic
    column at 16×16×32 with z cut, np = 2: the velocity canary still reads 4.5e-17 (each sub-column
    is separately hydrostatic!) while max\|P_dist − P_ref\| = **4.0e+02** and ∂P/∂z is off by 8×g·ρ.
    Until this is fixed, a multi-rank run with domain BCs is only correct if the partition does not
    cut a non-periodic axis; the two MPI ctests assert that.
  - **`fillPropGhosts` / `fillPorousEpsGhosts` skip the property BC under MPI.** Both apply the
    zero-gradient (resp. mirror-about-1) override on domain-BC faces only `if (!distributed_)`, so a
    distributed run leaves the μ / ρ / ε ghost on a walled face at its *periodic wrap* value.
    Measured: a two-layer Couette with a monotone μ stack (μ(0) ≠ μ(N−1)) differs from the
    single-rank reference by **2.7e-2 relative already at np = 1**. `test_varmu_mpi` sidesteps it
    with a symmetric μ stack (wrap value == zero-gradient value) and says so.
- PCG-on-coefficients MG follow-up (§2).

## References

[1] J.-L. Guermond, P. Minev, J. Shen, *An overview of projection methods for incompressible
flows*, Comput. Methods Appl. Mech. Engrg. 195 (2006) 6011–6045.

[2] J.-L. Guermond, A. Salgado, *A splitting method for incompressible flows with variable density
based on a pressure Poisson equation*, J. Comput. Phys. 228 (2009) 2834–2846.

[3] Phase-4 companion: `variable_viscosity_projection.md` (Deteix–Yakoubi rotational-term
references and the FaceProps machinery).
