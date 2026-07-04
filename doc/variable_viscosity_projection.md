# Variable viscosity in `peclet.flow`: operator assembly and the rotational-incremental projection

Status: implemented + validated (2026-07-04, host-openmp backend), flow commits `e3cb678`
(variable-coefficient momentum) and `54e25b1` (rotational-pressure treatment).
Part of the multiphysics framework, Phase 4 — see `../../docs/MULTIPHYSICS_PLAN.md`.

This note documents **exactly** what is implemented: the variable-viscosity momentum operator, why
the constant-viscosity rotational pressure update cannot be reused verbatim (with literature), the
scheme we adopted instead, its stability rationale, the validation, and the documented limitations
with the upgrade path.

---

## 1. Background: the constant-viscosity scheme being generalized

The solver advances the momentum equation in the *divided-by-dt* convention (operator
`(ρ/dt)·I − μ∇²`, well conditioned at large dt; see `CLAUDE.md`), then projects:

1. **Predictor** (per velocity component, backward-Euler diffusion, implicit or explicit advection):

   ```
   (ρ/dt)(u* − uⁿ) + ρ·adv(u^k) = −∇Pⁿ + μ ∇²u* + f
   ```

   The 7-band float stencil is assembled by `ibmBuildDiffusion` (`src/cut_cell_ibm.hpp`):
   `A_C = (float)(ρ/dt + 6μ)`, off-diagonals `= (float)(−μ)`, then the Robust-Scaled cut-cell IBM
   overlay is baked on top (`ibmModifyStencil`). The `−∇Pⁿ` **incremental predictor** in the RHS
   (`buildRhs`) is what makes very large time steps / steady-Stokes stepping work.

2. **Projection**: solve the cut-cell Poisson `A φ = −∇·(open·u*)`, correct `u = u* − ∇φ`.

3. **Rotational pressure update** (Timmermans et al. [1], as ported from the CUDA solver):

   ```
   P ← P + (ρ/dt)·φ − μ·∇·u*          (kernel "press", src/flow_ibm.hpp)
   ```

   The `−μ∇·u*` term removes the artificial homogeneous-Neumann boundary layer that the plain
   incremental scheme imposes on the pressure, giving a consistent pressure boundary condition and
   the scheme its accuracy at large dt [1, 2, 3].

## 2. The variable-viscosity momentum operator (what changed in the predictor)

Enabled by `set_property_mode("variable", harmonic=…)` on the Python side (or automatically when a
property closure targets `"mu"`); internal flag `varProps_`.

- **Viscosity field**: a registered cell-centred field `"mu"` (`muField_`), set from Python
  (`set_field("mu", arr)`) or by a closure (e.g. Arrhenius in temperature) each step.
- **Face viscosity**: the stencil needs μ *on faces between cells*. `FieldFaceProps`
  (`src/face_props.hpp`) supplies

  ```
  arithmetic:  μ_f = ½(μ_i + μ_j)                 (default)
  harmonic:    μ_f = 2 μ_i μ_j / (μ_i + μ_j)      (harmonic=True)
  ```

  The harmonic mean is the correct choice across a viscosity **jump**: it is the face conductance
  for which the shear stress `μ ∂u/∂n` is continuous across the material interface (series
  resistance), and it is what reproduces the analytic two-layer Couette profile (§5).
- **Assembly**: `ibmBuildDiffusionVar<FaceProps>` (`src/cut_cell_ibm.hpp`) is a *sibling* of the
  validated constant kernel — never an edit of it (bit-exactness policy):

  ```
  A_off(i) = (float)(−μ_f),   A_C(i) = (float)(ρ/dt + Σ_faces μ_f)
  ```

  Face means are computed in double and cast to float once, mirroring the constant path's
  `(float)(idiag + 6.0*beta)`. `UniformFaceProps` (two constants) exists for the kernel-level
  equivalence test.
- **μ ghosts** (`fillMuGhosts`, `src/flow_ibm.hpp`): periodic/halo fill, then **zero-gradient
  (copy) override on domain-BC faces** — a periodic wrap at a wall would bring the *opposite
  layer's* viscosity to the wall face, which destabilizes the harmonic mean in particular.
- **Solve routing**: under `varProps_` the momentum solve always uses the stencil smoother
  (`bcStencilPath()` includes `varProps_`); the velocity multigrid is disabled
  (`useVelocityMg_ = false`) because `VelocityMG` takes a scalar μ (variable-coefficient
  velocity-MG is deferred). Stencils are rebuilt each step (`rebuildStencils` in `step()`), or per
  Picard iteration on the implicit-FOU path (`buildAdvStencil`).
- **Rotational update sibling**: the pressure-update kernel gets varProps-gated siblings
  (`press_var_full` / `press_var_min`); the validated constant kernel `press` is untouched.

## 3. Why the rotational term cannot keep the local μ: the variable-viscosity problem

For **constant** μ the identity `∇²u = ∇(∇·u) − ∇×∇×u` splits the viscous term into a gradient
part `μ∇(∇·u*)` (absorbed into the pressure — this is where `−μ∇·u*` comes from) and a solenoidal
part. For **variable** μ(x),

```
∇·(2μ D(u))  ≠  ∇(μ ∇·u) + solenoidal remainder,
```

so the pointwise update `P += … − μ(x)∇·u*` is **no longer the gradient part of the viscous
stress**: it injects a spurious non-gradient contribution into the accumulated pressure every
step. Deteix & Yakoubi state it directly: the rotational scheme "is only valid for homogeneous
viscosity" [4, 5].

We observed exactly this failure mode empirically before the fix: a two-layer Couette flow with a
10× viscosity jump and harmonic face means, run with the incremental scheme and the pointwise
rotational term, diverges (NaN) after ~330 steps — a slow secular accumulation, not a CFL-type
blow-up. With a *uniform* μ field the same code reproduces the constant-μ solver exactly, which
isolates the inconsistency to the rotational term at μ-contrast.

### The consistent fix in the literature (not implemented — deferred)

The **shear-rate projection** (SRP) of Deteix & Yakoubi [4, 5] replaces `−μ∇·u*` by a correction
ψ solved from an *additional Poisson problem*. In the algorithmic form of [5, eq. (19)–(20)]
(BDF2, FEM setting): after the standard projection step for φ and the velocity correction,

```
Δψ = ∇·∇·( 2ν(u^{n+1}) D(u^{n+1}) − 2ν* D(ũ) ),      ∇ψ·n given by the same flux on ∂Ω,
p^{n+1} = pⁿ + φ + ψ.
```

For homogeneous ν this reduces to the Timmermans term. It is the fully consistent treatment (their
convergence rates match the rotational scheme's), at the cost of one extra Poisson solve per step
and the assembly of `∇·∇·(2νD(·))` — on our MAC cut-cell grid with the IBM overlay that is a
substantial second-derivative-of-stress discretization, so it is documented here as the **upgrade
path**, not implemented.

## 4. What is implemented: mode-selectable rotational coefficient, incremental predictor kept

The decisive observation: the **large-dt / steady-Stokes capability does not come from the
rotational term** — it comes from the incremental predictor `−∇Pⁿ` and the pressure accumulation.
The rotational term only sharpens the pressure boundary consistency. So under `varProps_` we keep
the incremental scheme fully active and make the rotational *coefficient* safe:

```
P ← P + (ρ/dt)·φ − μ_rot(i)·(∇·u*)(i)
```

selected by `set_variable_rotational(mode, chi)` (`Solver::setVariableRotational`,
`src/flow_ibm.hpp`):

| mode | μ_rot | properties |
|---|---|---|
| `"min"` (default) | `χ · min_Ω μ` (constant) | Stable at **any** contrast: since `μ_min ≤ μ(x)` everywhere, the added term is identical to the homogeneous-case rotational term of a fluid with viscosity μ_min, and the true dissipation dominates it pointwise — the constant-viscosity stability analysis [3] carries over. **Reduces exactly to the validated constant-μ scheme when μ is uniform.** |
| `"full"` | `χ · μ(i)` (pointwise) | Better pressure consistency at **mild** contrast only; demonstrably diverges at 10× (kept as an expert option, matches the constant path bit-for-bit for uniform μ). |
| `"off"` | `0` | Plain incremental projection: unconditionally stable, retains the artificial pressure Neumann layer of the non-rotational scheme [2]. |

Defaults: `mode="min"`, `χ=1`. `μ_min` is a per-step device min-reduction over the inner cells
(`minMuInner()`), with an `MPI_Allreduce(MIN)` under the distributed path so all ranks use the
global minimum. The classical (non-incremental) projection remains available via the pre-existing
`set_incremental_pressure(False)` but is **no longer forced**.

The constant-coefficient idea has precedent: it is the same philosophy as Guermond & Salgado's
variable-density projection, where the Poisson/rotational machinery is run with a constant
coefficient bounded by the minimum density to preserve the constant-coefficient stability and
solver structure [7]. Phase 5 (variable-density projection) will use the analogous construction.

### Python API

```python
s = peclet.flow.Solver(nx, ny, nz)
s.add_field("mu"); s.set_field("mu", mu_array)        # or a closure targeting "mu"
s.set_property_mode("variable", harmonic=True)         # variable-coefficient momentum
s.set_variable_rotational("min", chi=1.0)              # default — shown for completeness
s.step()                                               # incremental-rotational, large dt OK
```

## 5. Validation (all in `tests/study/two_layer_couette.py` + `tests/kokkos/test_variable_mu`)

Two-layer plane Couette, μ₁/μ₂ = 1.0/0.1 (10×), bottom wall fixed, top wall moving at U, N=32,
analytic steady profile piecewise linear with interface velocity `u_i/U = μ₂/(μ₁+μ₂)`:

| configuration | result |
|---|---|
| incremental + rotational(`min`), harmonic, dt=20 | max error **0.0006 %** vs analytic, converged @ 500 steps, stable |
| incremental + rotational(`min`), harmonic, **dt=100** | max error **0.0002 %**, converged @ **200 steps** — large-dt capability retained at 10× contrast |
| incremental + rotational(`min`), arithmetic | 1.9 % (harmonic face mean matters at a jump) |
| incremental + rotational(`full`), 10× | **diverges @ step 331** — the documented homogeneous-only failure [4] |
| uniform μ through the varProps path vs the constant solver | pressure agrees to 2.4e-18; velocity to 2.6e-6 (float stencil-band quantization of the two solve paths — the varProps path uses the float-band stencil smoother, the constant all-fluid domain-BC path the double-precision fold smoother; same physics, both pre-existing discretizations) |
| kernel level (`test_variable_mu`) | `UniformFaceProps` bands ≡ constant kernel; arithmetic/harmonic face bands vs oracle exact |
| single-phase regression (`tests/regression/sdflow_regression.py`) | **bit-exact (+0.00 %, identical iteration counts)** — `varProps_=false` executes the original kernels character-for-character |

## 6. Limitations and deferred work

1. **Laplacian stress form.** The momentum operator is `∇·(μ∇u)`, not the full deviatoric stress
   `∇·(μ(∇u + ∇uᵀ))`. For incompressible flow the omitted term is
   `(∇·(μ∇uᵀ))_i = (∂_i u_j)(∂_j μ)` — it vanishes for layered configurations (Couette/Poiseuille
   with μ = μ(y), which is why §5 converges to the exact profile) but is nonzero at general
   μ-gradients. Adding the transpose term pairs naturally with the SRP upgrade (both come from the
   full-stress formulation of [5]).
2. **Shear-rate projection (SRP)** [4, 5]: the fully consistent pressure correction (extra ψ
   Poisson, §3) — the upgrade if maximal pressure accuracy at strong contrast or with
   open/natural boundaries [6] is needed.
3. **Velocity multigrid** takes scalar μ — forced off under `varProps_`; variable-coefficient
   velocity-MG deferred.
4. Validated on the host-openmp backend; CUDA-backend and multi-rank MPI validation of the
   varProps path are deferred with the corresponding plan phases (the `μ_min` MPI allreduce is
   already in place).
5. `μ_rot` uses the **global** minimum; for extreme, highly-localized contrast this makes the
   rotational correction weak far from the interface (accuracy, not stability). `χ` and `"full"`
   give manual control; SRP is the principled answer.

## References

[1] L.J.P. Timmermans, P.D. Minev, F.N. van de Vosse, *An approximate projection scheme for
incompressible flow using spectral elements*, Int. J. Numer. Methods Fluids 22 (1996) 673–688.

[2] J.-L. Guermond, P. Minev, J. Shen, *An overview of projection methods for incompressible
flows*, Comput. Methods Appl. Mech. Engrg. 195 (2006) 6011–6045.

[3] J.-L. Guermond, J. Shen, *On the error estimates for the rotational pressure-correction
projection methods*, Math. Comp. 73 (2004) 1719–1737.

[4] J. Deteix, D. Yakoubi, *Improving the pressure accuracy in a projection scheme for
incompressible fluids with variable viscosity*, Appl. Math. Lett. 79 (2018) 111–117.
https://doi.org/10.1016/j.aml.2017.12.007

[5] J. Deteix, D. Yakoubi, *Shear rate projection schemes for non-Newtonian fluids*, Comput.
Methods Appl. Mech. Engrg. (2019). arXiv:1902.05643. (Statement "only valid for homogeneous
viscosity" and the SRP algorithm, eqs. (9)–(13), (18)–(20).)

[6] L. Plasman, J. Deteix, D. Yakoubi, *A projection scheme for Navier–Stokes with variable
viscosity and natural boundary condition*, Int. J. Numer. Methods Fluids (2020).
https://doi.org/10.1002/fld.4851

[7] J.-L. Guermond, A. Salgado, *A splitting method for incompressible flows with variable
density based on a pressure Poisson equation*, J. Comput. Phys. 228 (2009) 2834–2846. (The
constant-coefficient-bounded-by-the-minimum philosophy, used here for μ and planned for the
Phase-5 variable-density projection.)
