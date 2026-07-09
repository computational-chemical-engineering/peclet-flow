# The volume-averaged (porous) CFD-DEM fluid scheme

How `peclet.flow` solves the gas phase of an unresolved point-particle CFD-DEM (`peclet.coupling`,
`porous=True`). Staggered MAC grid, divided-by-dt convention, constant gas density ρ. Companion:
`variable_density_projection.md` (the same face-consistent coefficient machinery for variable ρ);
consumer: `coupling/README.md`.

## 1. Equations (Model B)

```
∂ε/∂t + ∇·(εu) = 0                                                (volume-averaged continuity)
ρ( ∂u/∂t + u·∇u ) = −∇p + μ∇²u − β(u − u_p) + f                   (gas momentum)
```

`ε` is the void fraction deposited from the particles, `u` the **interstitial** gas velocity,
`β` the interphase drag coefficient per unit volume, `u_p` the particle velocity, `f` body forces.
**Model B**: the gas feels the full `−∇p` (no `−ε∇p` split); the particles feel drag + gravity only
(no explicit `−V_p∇p`; for gas–solid density ratios the buoyancy this absorbs is negligible). The
drag closures are the literature Model-A forms converted once, in the coupling kernel:
`β_B = β_A/ε` (per particle: `F /= ε`). Model A vs B differ *only* in this pressure-force split —
both use the full continuity above. (The `porous=False` coupling mode is a separate *dilute
simplification* — `div u = 0`, ε only inside the drag law — not Model B.)

## 2. Discretization

Face means at the staggered face `f` of component `c` (cells `i−s_c, i`): `ε_f = ½(ε_i+ε_{i−s_c})`,
`β_f = ½(β_i+β_{i−s_c})`; `idt ≡ ρ/dt`; cut-cell face openness `open_f`.

**Momentum predictor** (backward Euler, incremental — carries the old pressure gradient):

```
A_{p,f} u*_f + Conv(u*) − μ(∇²u*)_f = idt·u^n_f + β_f u_{p,f} + f_f − (P^n_i − P^n_{i−1})
A_{p,f} = idt + β_f
```

- The drag is **face-averaged on the diagonal** — the same `β_f` as the projection below. This
  three-way consistency (diagonal ↔ operator ↔ correction) makes the incremental predictor cancel a
  pressure perturbation exactly in one step; with any mismatch the pressure loop has gain
  `|1 − (idt+β_f)/A_p^{momentum}|` and a fixed bed diverges whenever `β > ρ/dt`.
- **Convection**: fully-implicit first-order upwind (FOU) in the stencil + explicit
  deferred-correction TVD (Koren) — unconditionally stable at the large coupled dt. Enabled by the
  coupling by default.
- **Viscous**: implicit, Laplacian form `μ∇²u` (constant μ).
- Under domain BCs the solve always uses the assembled stencil (`bcStencilPath()` includes
  `hasDrag_`); ghosts are re-imposed per smoother colour.

**Projection** (SIMPLE-like coefficient; the pressure unknown is φ with `δp = idt·φ`):

```
∇·(C_f ∂φ) = ∇·(open·ε_f·u*) + ∂ε/∂t          C_f = open_f · ε_f · w_f,   w_f = idt / (idt + β_f)
u^{n+1}_f  = u*_f − w_f ∂_f φ
P^{n+1}    = P^n + idt·φ − μ·r                 (rotational term; r = the porous residual above)
```

`w_f = idt/A_{p,f}` is the SIMPLE `d_e` built from the drag-loaded momentum diagonal: stiff drag →
`w_f → 0` → the pressure barely moves the velocity (the drag holds it) — unconditionally stable for
any β. `ε ≡ 1, β ≡ 0` reduces every factor to 1.0 in floating point: the single-phase solver is
recovered identically. `∂ε/∂t = (ε^{n+1}−ε^n)/dt` from the deposited fields (`sync_porous_prev()`
seeds `ε^n` after the first deposition).

**Boundary-face ε (the distributor contract).** All porous consumers — the RHS divergence, the
Poisson coefficients, and `max_porous_residual` — read the eps field's ghosts through ONE fill
(`fillPorousEpsGhosts`): periodic/halo base, zero-gradient at walls, and **mirror-around-1 at
inflow/outflow faces so the arithmetic face mean is exactly 1** — the boundary is pure gas (below
the distributor plate / in the freeboard), so a prescribed inflow velocity is the **superficial**
gas velocity and the inlet face flux is `open_f·1·u_in` (the Kuipers/MFIX convention). Before this
policy the RHS read the coupling deposit's ghost leakage while the residual read zero-gradient
ghosts: two different constraints, leaving an irreducible residual `(ε_f^{rhs} − ε_f^{resid})·u_in`
pinned at the distributor row and feeding the bed only `ε_f·U` instead of `U` (a bed under-fed by
30–60%, with the target flux flickering with the bottom grains' deposit each step).

## 3. Pressure solver

Geometric multigrid (`CutcellMG`): **red-black Gauss–Seidel** smoother, **rediscretized** coarse
operator; the `ε_f·w_f` coefficients ride the openness rails (per-level ghost fill + boundary
re-imposition + rediscretized coarsening, zero MG-code changes). Driver: **MG-PCG** (default) or the
standalone V-cycle — verified equivalent on drag-loaded beds (porous residual → 1e-11). Chebyshev is
disabled for porous+drag (diverges on the high-`w_f`-contrast operator). The **GraphAMG bottom**
(agglomerated, decomposition-agnostic coarse solve) is the porous+drag default and is domain-BC
aware: the assembled bottom matrix does not wrap across non-periodic faces (the Dirichlet outflow
anchor stays diagonal-only) and the constant-mode projection is applied only when the operator is
singular (no outflow). `set_pressure_graph_amg` propagates to the MG immediately.

## 4. Defaults and knobs

| what | default | knob |
|---|---|---|
| convection (implicit FOU + deferred TVD) | on (set by the coupling) | `CfdDem(advection=…)` |
| incremental-rotational pressure | on | `set_incremental_pressure` |
| pressure driver | MG-PCG | `set_pressure_pcg` / `set_pressure_multigrid` |
| GraphAMG bottom | on for porous+drag | `set_pressure_graph_amg` |
| d(ε)/dt source in the projection | on | `set_porous_deps_dt` |
| pressure under-relaxation | off (ω=1) | `set_pressure_underrelax` |
| void-fraction floor | 0.4 (≈ random-close-packing voidage) | `CfdDem(eps_min=…)` |

## 5. Validation

- **Ergun ΔP, porous path** (`coupling/tests/test_fixed_bed_ergun_porous.py`): uniform fixed bed,
  periodic + body force, Gidaspow closure — (f_drive, U=ε·u_i) lands on the Ergun curve to ~3%
  across the viscous → inertial regimes, no fitted factors.
- **Ergun ΔP, incompressible path** (`test_fixed_bed_ergun.py`): 0.0% (unchanged).
- Synthetic fixed bed (hand-set ε, β): 300 steps, porous residual → 1e-11, ΔP = Σβ_f·w_f·h.
- Real frozen beds (deposited ε, Gidaspow β up to ~10³ at the packing floor): steady, square and
  cylindrical vessels, CUDA + OpenMP.
- Moving fluidized bed (`coupling/examples/fluidized_bed.py`): fluidizes, finite, 200+ steps, both
  backends. Single-phase regression suite: byte-identical.

## 6. Known limitations / roadmap

- Inertia is `ρ∂u/∂t + ρu·∇u`, not `ρε(…)` — the `ρε_f` convection/time weight (consistent
  volume-averaged inertia) is a planned accuracy upgrade; it changes `w_f` to `idt·ε_f/(idt·ε_f+β_f)`.
- Viscous stress is `μ∇²u`; the volume-averaged form `∇·[εμ(∇u+∇uᵀ)]` (ε and the transpose inside)
  is planned; the normal part slots into the existing `FaceProps` variable-coefficient band.
- The particle side of the drag is explicit (`u_p` frozen over the DEM sub-steps). For very stiff
  drag with *moving* particles (`m_p/β < Δt`) a per-substep implicit particle-drag update
  (PEA-style) is the planned upgrade; the fluid side is already implicit.
