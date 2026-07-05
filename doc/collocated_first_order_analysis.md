# Why the collocated solver is first-order at curved immersed boundaries

**Status:** analysis + design note (2026-07-04). Motivated by a staggered-vs-collocated
grid-convergence study on the Zick–Homsy sphere array. No code changed by this note; it
documents *why* `SolverColocated` converges one order slower than `Solver` at curved
cut-cell boundaries, and what a Basilisk-style embedded-boundary treatment would change to
recover second order.

## TL;DR

- On a **flat, grid-aligned** wall both grids are **pointwise exact** (Poiseuille reproduced
  to solver tolerance). The cut-cell machinery is not "wrong" on either grid.
- On a **curved** wall (a periodic sphere array, Stokes drag vs Zick & Homsy):
  - `Solver` (**staggered MAC**) converges at **second order** and lands on the benchmark
    (−0.02% at N=128, at the solver-tolerance floor).
  - `SolverColocated` (**cell-centred + ABC approximate projection**) converges at **first
    order** to the *same* limit (+0.60% → +0.40% → +0.30% at N=64/96/128; local order
    p≈1.0). It is **consistent**, not biased — the error is reducible by refinement, just at
    O(h) instead of O(h²).
- The `~1%` drag error people see at practical resolution (N≈32) is therefore **first-order
  boundary error**, not an intrinsic velocity-placement floor. Doubling N halves it.
- The root cause is a **loss of consistency at the immersed boundary that is specific to
  cell-centred velocity placement**: the momentum no-slip and the mass-conservation
  constraint are reconstructed on *two different sub-cell representations of the same curved
  surface*, and the projection that couples them is only *approximate*. The staggered grid
  avoids this because its velocity unknowns are co-located with the mass-flux fractions and
  its projection is exact.
- Basilisk's `embed.h` fixes exactly this by driving **every** operator (viscous *and*
  pressure) from **one** geometric description of the cut cell — volume fraction `cs`, face
  fractions `fs`, boundary **centroid** and **normal** — and reconstructing the wall flux
  along the **true normal**. That restores second order on a collocated grid.

## The evidence

Periodic simple-cubic sphere array, φ=0.125, Stokes (advection off), body-force driven;
drag factor `K = F L³ / (6πμR⟨u⟩)` vs the Zick & Homsy (1982) value 4.292. Error `(K−K_ZH)/K_ZH`:

| N   | staggered | collocated | collocated local order |
|----:|----------:|-----------:|-----------------------:|
| 16  | −1.74%    | +0.78%     | —                      |
| 32  | −0.31%    | +0.99%     | —  (under-resolved)    |
| 64  | −0.018%   | +0.598%    | —                      |
| 96  | +0.009%   | +0.397%    | **+1.01**              |
| 128 | +0.013%   | +0.299%    | **+0.99**              |

Staggered: local order 2.2–2.9 in the resolved range, then it hits the ~0.01% solver-tolerance
floor. Collocated: once the coarse-grid transient clears (N≥64) the error falls at a clean
**first order** — |error| runs parallel to O(h¹) on a log-log plot while staggered runs
parallel to O(h²). Both extrapolate to Zick & Homsy (collocated ≈0.15% at N=256, ≈0.075% at
N=512).

Two controls localise the cause:
- **Flat grid-aligned wall → both exact.** So neither the bulk viscous operator nor the
  projection is at fault; the defect needs *curvature* (a boundary whose normal is not
  axis-aligned and whose cut fractions are non-trivial).
- **Measurement is not the artefact.** The physically-conserved superficial velocity is the
  open-weighted flux `Σ o_f u_f` (machine-constant across planes on both grids). Reporting it
  instead of the raw cell mean changes the collocated number by <0.05% and does not help —
  the error is in the converged field, not the diagnostic.

## The current cut-cell IBM (what both grids share)

The immersed no-slip is imposed by a **Robust-Scaled cut-cell overlay** on the backward-Euler
velocity-diffusion stencil (`src/cut_cell_ibm.hpp`, `ibmFillEntry`). For each cut cell and
each of the **six grid-axis directions** `k` it:

1. Detects a solid neighbour (`sdf_n[k] < 0`) and computes the **axis intercept**
   `theta_k = sdf_c / (sdf_c − sdf_n[k])` — the fractional distance, *along that grid axis*,
   from the velocity node to where the wall crosses the link.
2. Builds a **one-dimensional quadratic image** through {wall value (0), the node, the interior
   neighbour} so that the Dirichlet condition holds at `theta_k`. The coefficients are the
   `poly_D / poly_Nc / poly_N_nb` polynomials (and their `_avg`, −1/12 moment-corrected
   variants for the cell-average scheme). The modified diagonal/off-diagonals and the
   inhomogeneous wall term (`K/M/X/Nbc`) encode that image.
3. Applies a **row rescaling** `D_rescale` (the smallest-magnitude boundary polynomial on the
   cell) so a cell that is only slivered-open does not blow up the condition number.

The pressure Poisson uses the **face openness** `o_f` — the fluid *area fraction* of each face,
`buildOpenness → ccFaceOpen` (`src/mac_cutcell.hpp`). The operator is `A = −div(o·grad)`, the
right-hand side is the open-weighted divergence `div(o·u*)`, and the correction is the raw
face gradient re-weighted by `o_f` inside the divergence (`src/mac_pressure.hpp`:
`buildCutcellOp` / `divergOpen` / `projectCorrect`).

Note the two **different geometric descriptions** already in play:
- the viscous no-slip uses **axis intercepts** `theta_k` (six 1-D link distances from the
  velocity node);
- the mass constraint uses **face-area fractions** `o_f` (Basilisk's `fs`).

On the staggered grid these describe the boundary *consistently* (see below). On the collocated
grid they do not.

### The two projections

- **Staggered (`Solver`)** stores the velocity *on the faces*. The projection is the classical
  **exact MAC projection**: a discrete Helmholtz decomposition whose divergence, gradient, and
  Laplacian are mutually adjoint on the same face field. The corrected face field is divergence
  free to round-off.
- **Collocated (`SolverColocated`)** stores all three components at the cell centre and closes
  incompressibility with an **Almgren–Bell–Colella approximate projection**
  (`src/mac_approx_projection.hpp`): average the cell velocities onto faces (`centerToFace`,
  `uf = ½(U_i + U_{i−1})`), make *that* face field divergence-free with the same cut-cell
  Poisson machinery, then correct the cell velocities with a **cell-centred gradient**
  (`projectCorrectCenter`, the average of the two adjacent face gradients, with closed faces
  zeroed). The *face* field is exactly divergence-free; the *cell* field is only approximately
  so — hence "approximate projection".

## Why the staggered grid is second order

Two structural properties conspire, and both are consequences of **co-location**:

1. **The momentum no-slip and the mass flux are imposed at the same place.** Component `u`
   lives on the x-face; its cut-cell overlay is built from the SDF sampled *at that face*
   (`Grid::offset(0) = {−½,0,0}`), and the x-face openness `o_x` that weights `u` in the
   divergence is the fluid fraction of *that same face*. So the discrete "where is the wall for
   `u`" used by the viscous operator and the one used by the incompressibility constraint are
   the **same sub-cell surface**. The velocity field is asked to satisfy a *consistent* pair of
   boundary relations.
2. **The projection is exact.** Because the MAC divergence/gradient/Laplacian are adjoint on
   the face field, the discrete pressure force and the discrete incompressibility are
   consistent to round-off; no boundary inconsistency is injected by the pressure step.

Under these two properties the drag — a *boundary functional* of the solution, i.e. the net
viscous+pressure force the fluid exerts on the sphere — inherits the second-order accuracy of
the underlying cut-cell reconstruction. Empirically we measure p≈2.

On a flat grid-aligned wall the same reasoning trivially holds on *both* grids (the axis
intercept and the face fraction describe the identical 1-D wall, and the quadratic image is
exact for the quadratic Poiseuille profile), which is why both are pointwise exact there.

## Why the collocated grid drops to first order

Curvature breaks the consistency that co-location gave the staggered grid, in two coupled ways.

**(a) The momentum BC and the mass constraint see two different sub-cell boundaries.**
The viscous no-slip is reconstructed at the **cell centre** from the six **axis intercepts**
`theta_k`. The mass flux uses the **face-area fractions** `o_f` a half-cell away. For a curved
surface whose normal is not axis-aligned, these two descriptions of "the wall" differ by O(h):
the set of six axis intercepts from a cell centre is *not* the same surface as the six
surrounding face fractions. On the staggered grid they were forced to coincide by placing the
velocity on the face; on the collocated grid there is a genuine half-cell mismatch that scales
with the boundary curvature and does **not** cancel. This is a first-order boundary error in
the near-wall momentum balance.

**(b) The approximate projection reconstructs the pressure force at cut cells only to first
order.** `projectCorrectCenter` corrects the cell velocity with the average of the two adjacent
face gradients, *zeroing the gradient across any closed (solid) face*. In the interior this is
the second-order central difference; at a cut cell adjacent to solid it degenerates to a
one-sided, openness-blind gradient. The cell velocity's pressure force at the immersed boundary
is therefore only first-order accurate, whereas the exact MAC projection on the staggered grid
carries the pressure force at the boundary consistently.

Neither defect is a bug in the axis-by-axis reconstruction itself (it is second order on the
staggered grid). Both are **consistency defects specific to cell-centred placement on a curved
boundary**: the geometry used by the momentum operator, the geometry used by the mass
constraint, and the pressure coupling between them no longer agree to second order. The
solution field can remain close to second order in the bulk, but the **drag functional**, which
is dominated by the near-wall layer, converges at first order — exactly what the data show.

The sign is consistent with this picture: collocated `K` is *high* (⟨u⟩ low ⇒ slightly too much
resistance), i.e. the cell-centred reconstruction presents a marginally "larger/rougher"
effective obstacle than the face-consistent staggered one, an O(h) effect that shrinks as the
mesh refines.

## The Basilisk solution, in detail

Basilisk's `embed.h` is a **cell-centred** (collocated) embedded-boundary method that is
*second order* for both Poisson/diffusion and the Navier–Stokes projection. It achieves this by
describing each cut cell with **one** consistent set of geometric moments and using it in
**every** operator:

- **`cs`** — the fluid **volume fraction** of the cell (0 solid … 1 full fluid).
- **`fs.x/fs.y/fs.z`** — the fluid **area fraction of each face** (this is exactly our
  openness `o_f`; we already compute it).
- From `cs` and `fs`, per cut cell: the boundary-fragment **area**, its **outward unit normal
  `n`** (`facet_normal`, from the gradient of the fractions), and its **barycentre `b`** (the
  centroid of the wetted part of the embedded interface within the cell) via `embed_geometry`.

The two ingredients that give second order and that the current cut-cell IBM lacks:

1. **Normal-gradient reconstruction at the boundary centroid (`dirichlet_gradient`).** To impose
   a Dirichlet value `u_b` (here 0) Basilisk does *not* use axis intercepts. It builds an **image
   point** a distance `d` from the boundary **centroid `b` along the true normal `n`**,
   **interpolates the field there** from the surrounding cell-centred values with a
   multi-dimensional (bi/tri-linear) stencil (`embed_interpolate`), and forms the wall-normal
   gradient
   `∂u/∂n |_b ≈ (u_interp − u_b)/d` (a two-point, second-order formula; it falls back to a
   one-point formula only when the stencil is too occluded). Because the reconstruction is
   along the **actual normal** and uses a **multi-dimensional** interpolation, the wall shear
   stress is second-order for a curved surface — not tied to the six grid axes.

2. **A single embedded-boundary flux (`embed_flux`) added consistently to *both* operators.**
   The wall contributes a flux `area · μ · ∂u/∂n|_b` to the viscous Helmholtz operator, and the
   pressure Poisson is assembled with **face coefficients `α = fs/ρ`** plus the matching
   embedded-boundary term. The projection then makes the **`fs`-weighted face flux** divergence
   free *with the same boundary geometry the viscous step used*.

So in Basilisk the momentum boundary condition, the viscous flux, the pressure operator, and
the incompressibility constraint are **all** expressed through `(cs, fs, b, n)` — one boundary,
described once, to second order.

## Why the Basilisk approach will succeed where ours degrades

The collocated first-order loss came from (a) two inconsistent sub-cell boundaries and (b) an
approximate pressure coupling at cut cells. Basilisk removes both:

- **One geometry for everything.** The viscous no-slip and the mass constraint are built from
  the *same* `cs`/`fs`/centroid/normal, so the half-cell mismatch of defect (a) cannot arise:
  there is no separate "axis-intercept surface" and "face-fraction surface", only the single
  embedded interface. This is the collocated analogue of the staggered grid's co-location — but
  achieved through shared geometry rather than shared location, so it works with cell-centred
  velocities.
- **The wall flux is reconstructed along the true normal, multi-dimensionally.** The drag is a
  wall-normal-gradient functional; reconstructing that gradient at the boundary centroid along
  `n` (rather than as a by-product of six independent axis quadratics) is what makes the
  boundary functional — not merely the bulk field — second order on a curved surface.
- **Consistent pressure coupling.** With `α = fs/ρ` and the embedded-boundary flux in the
  Poisson operator, the pressure force at the cut cell is represented to the same order as the
  viscous flux, curing defect (b). (This is a genuine improvement over the ABC
  `projectCorrectCenter` cell-gradient, which is only first-order at cut cells.)

Net effect: the near-wall momentum balance becomes second-order consistent on the collocated
grid, so the drag converges at O(h²) — matching the staggered solver — instead of O(h).

## What this would mean for `flow` (implementation sketch)

We already have `fs` (the face openness). A Basilisk-style upgrade of the collocated path would
add, on the collocated grid only:

1. **Cut-cell geometry**: per cut cell compute `cs` (cell volume fraction), the boundary
   **normal** `n` (from ∇ of the fractions / the SDF gradient, which we have) and **centroid**
   `b`. The SDF gives `n` directly; `cs` and `b` come from the same marching-cube/fraction
   computation that already produces `fs`.
2. **`dirichlet_gradient`-style no-slip**: replace the axis-by-axis `ibmFillEntry` overlay *for
   the collocated momentum operator* with an image-point normal-gradient reconstruction at `b`
   along `n`, using a multi-dimensional interpolation of the cell-centred velocities. This is
   the core change.
3. **Consistent embedded flux in the projection**: add the matching embedded-boundary term to
   the collocated pressure operator so the `fs`-weighted projection and the viscous no-slip
   share the geometry (the operator already uses `fs`; the correction would move from the ABC
   `projectCorrectCenter` cell-gradient toward a flux consistent with the embedded boundary).

The staggered path is left untouched — it is already second order for the reason above, and it
remains *the* flow solver for drag/permeability. The value of the change is to make the
collocated variant second-order too (dropping the N=128 sphere-drag error from ~0.30% to the
~0.01% floor), which matters for the collocated grid's intended uses (coupling, single-grid
convenience) where curved immersed boundaries are present.

## Repair attempts within the unidirectional IBM (2026-07-04, experimental)

Directed follow-up: keep the axis-by-axis IBM and repair the *collocated projection*
(`set_face_interp(mode)` runtime toggle; mode 0 = unchanged default; staggered untouched;
all modes pointwise-exact on the flat-wall Poiseuille guard). Chronology of the ablation,
each step falsifying or sharpening the previous hypothesis (Z&H φ=0.125 sphere, GPU):

| mode | scheme | drag err N=32/48 | verdict |
|---|---|---|---|
| 0 | plain ½/½ + central `grad P` (the implicit adjoint pair of the wrong, wall-at-neighbour-centre geometry) | +0.99 / +0.68% | baseline, O(h) |
| 1 | wall-anchored weighted-LSQ quadratic `T_w` (face-centre abscissa; sliver cell shed via `w_c=θ²/(θ²+0.15²)`) | +2.30 / +1.55% | **worse**, still O(h) |
| 2 | mode 1 + exact transpose `Tᵀ_w` as predictor `-grad P` + cell correction | +2.43 / +1.59% | pairing alone is not the lever |
| 3a | mode 2 evaluated at the **open-face-centroid** wall distance (static geometry from trilinear SDF) | −5.9% / diverges | **unstable** (see below) |
| 3b | mode 3a with the **o-weighted exact adjoint** `Cᵀ = Tᵀ·diag(o)·Dᵀ` | +2.57 / +2.03% | stable — but not more accurate |

Findings, in order of importance:

1. **The flux metric is the right a-priori design metric.** The projection's flux model is
   `o_f·uf`, so the correct `uf` is the **open-area mean** of the velocity, not the face-centre
   point value: the open area lies on the fluid side, farther from the wall, so accurate point
   values *under-count* the flux. Measured against the true open-area flux (manufactured Stokes
   solution), all axis-only reconstructions are O(h)-biased low — plain −3.1e-3, `T_w`(face-centre)
   −5.7e-3 — and this ranking **predicts the solver ranking** (mode 2 worse than mode 0). Evaluating
   the same wall-anchored parabola at the open-centroid wall distance kills the a-priori bias
   (→ +3.2e-5 at N=128, ~O(h²)).
2. **Unweighted transpose force = spurious net force = runaway.** `T_w` deliberately has row sums
   ≠ 1 at cut faces (the no-slip anchor), so the force `Tᵀ_w·G·P` does not telescope to zero over a
   periodic box: the fluid feels a net spurious pressure force ∝ near-wall gradients → positive
   feedback → mode 3a's divergence (abscissa capping does not help; the weights were never the
   problem). The **o-weighted exact adjoint** `Cᵀ` of the constraint `C = D·diag(o)·T` makes the
   pressure do zero work on the constraint manifold and cures the instability outright (mode 3b) —
   the energy argument validated experimentally.
3. **But adjoint-consistency ≠ physical consistency of the force.** Mode 3b is stable yet still
   first-order with a *worse* constant than the plain pair. The residual defect is structural: the
   unidirectional IBM momentum row is a **finite-difference** operator (every term a point value at
   the cell centre — the pressure force wants the point gradient ∇p), while the cut-cell constraint
   is **finite-volume** (o-weighted fluxes — adjointness wants an o-weighted scatter). At cut cells
   these two demands conflict, and no axis-only `(T, force)` pair we constructed satisfies both.
   Basilisk resolves exactly this by making the momentum equation finite-volume too (`cs`-weighted,
   with an embedded-boundary flux) so that consistency and adjointness coincide.
4. **The plain pair (mode 0) is the best of the axis-only family.** Its wall-at-neighbour-centre
   geometry is wrong, but it is a *self-consistent* discretization of that wrong geometry, and its
   flux over-count partially cancels the open-area under-count. Every "more accurate" ingredient
   inserted alone breaks a compensation and degrades the drag.

**Conclusion of the repair attempt:** within the strictly unidirectional toolkit (axis intercepts
+ face openness + SDF-derived per-face centroids), the collocated projection appears pinned at
first order with mode 0 as the best constant. A second-order collocated scheme needs the momentum
and constraint to share one finite-volume cut-cell geometry — the embed route — which this ablation
now justifies with data rather than conjecture. The mode-1/2/3 code paths are kept (default off) as
reproducible ablations.

## The fully-FV route: a-priori validation of the wall viscous flux (2026-07-04)

Question: can the IBM become fully finite-volume, keep the 7-point stencil, and reach second
order on the collocated grid? Two observations make the design cheap, and an a-priori test
validates its one uncertain ingredient.

**Observation 1 — the wall-fragment normal is free.** For the fluid part of a cut cell the
divergence theorem gives `(∫_wall n dA)_a = h²(o_{a−} − o_{a+})` — the fragment's area-weighted
fluid-outward normal is already encoded in the face openness, per axis, exactly (verified:
Σ|W| reproduces the sphere area to 5 digits at N=128). No surface reconstruction, no stored
normals, no new geometry beyond the cell volume fraction.

**Observation 2 — the axis decomposition is exact at the wall.** No-slip kills the tangential
derivative, so `∇u = n·(∂u/∂n)` exactly on the surface: the per-axis 1-D wall-anchored profiles
estimate the *complete* gradient; nothing multi-dimensional is structurally missing.

**The test** (manufactured Stokes sphere; total wall viscous flux `μ∮∇u_x·n dA` vs analytic):

| N | axis-intercept anchor | **fragment-centroid anchor** | central naive |
|---:|---:|---:|---:|
| 32 | +7.87% | **+0.121%** | +36% |
| 64 | +3.88% (order 1.0) | **+0.026% (order 2.2)** | +33% |
| 128 | +2.03% (order 0.9) | **+0.0058% (order 2.2)** | +31% |
| 192 | +1.37% (order 1.0) | **+0.0025% (order 2.1)** | +31% |

Anchoring each 1-D profile at the cell's own **SDF closest-point projection**
`p* = x − sdf(x)·∇sdf(x)` (the fragment-centroid proxy) is the whole difference between first
and second order. The axis line through `p*` meets the wall *at* `p*`, so the wall-anchored
one-sided quadratic needs only two samples per axis at `p* ± σh, ±2σh` (σ = the fluid side,
`sign(n_a)`), which are provably on the fluid side; a tangent axis returns a zero gradient
automatically (the anchored quadratic cancels the quadratic growth term). No solid-cell
borrowing: every fragment cell projects its own anchor — even solid-centred cut cells.

**Keeping the 7-point matrix.** The two off-line samples per axis come from trilinear
interpolation of cell values (a compact but >7-point coupling). Standard remedy, already
idiomatic in this codebase: **deferred correction** — keep the 7-point axis-intercept flux
(or the existing IBM row) in the matrix, put the (centroid − intercept) wall-flux difference
in the lagged RHS; the steady state sees the second-order operator, the linear solves stay
7-point.

**Resulting design for a second-order collocated solver** (all ingredients now individually
validated a priori): FV momentum at cut cells (face fluxes `o_f·two-point`, wall flux as above,
`cs` time term with `D_rescale`-style small-cell handling), FV pressure force with the
homogeneous-Neumann wall closure (= the o-weighted exact adjoint of the constraint *plus* the
wall term generated by the same divergence-theorem identity — consistency and adjointness
coincide by construction), and the open-centroid constraint quadrature (mode-3 machinery, kept).
Remaining untested pieces: the trilinear-sampling accuracy near walls (occlusion fallback), and
the coupled-solve behaviour — the next implementation step. Reproduce the a-priori table with
`tests/study/fv_wallflux_apriori.py` (pure NumPy, no build needed); the Z&H drag A/B harness used
for the mode-0..3 solver numbers is `tests/study/collocated_zh_ab.py` (GPU build, edit the mode).

## Implementation of the fully-FV route (mode 4, 2026-07-05)

`set_face_interp(4)` implements the fully-FV design as a **defect correction** on the collocated
momentum, reusing the mode-3 centroid projection (constraint + o-weighted-adjoint force):

    solve   M·u^{k+1} = M·u^k − ω·rs·( L_FV(u^k) − b_FV )

with `M` the existing (stable, small-cell-safe) Robust-Scaled IBM matrix as preconditioner and
`L_FV` the finite-volume operator — `idt·cs·U + μ[ Σ_f o_f(U_i−U_nbr) − Σ_a W_a g_a^centroid ]`,
`b_FV = idt·cs·u^n + cs·(f − ∇p)`. The fixed point satisfies `L_FV·u* = b_FV` exactly, independent
of `M` and of the under-relaxation `ω`. New pieces (`mac_approx_projection.hpp`): `buildCellFraction`
(`cs`, subsampled), `fvViscousApply` (`L_FV`), `stencilMatvec` (`M·u`); wired into `buildRhs` for
`faceInterp==4`. Interior cells have `M = L_FV`, so the defect vanishes and they stay byte-identical
to mode 0.

**Two lessons on the way to a running solve:**
- **Sign.** The FV wall term is `−μ Σ W_a g_a` (from `∫_CV −μ∇²u = −μ[flux_out − flux_in]`); a `+`
  makes the wall anti-dissipative and the solve blows up.
- **Kokkos capture.** A `KOKKOS_LAMBDA` must not read a solver *member* (`fvRelax_`): the device
  dereferences the host `this` pointer → `cudaErrorIllegalAddress`. Copy members to locals first.

**Result — the fixed point is NOT second order.** Once stable, mode 4 converges to an ω-independent
fixed point, but the Z&H sphere drag is `+0.81% / +0.99% / +0.93%` at N=32/48/64 — comparable to mode 0
(`+1.00% / +0.68% / +0.60%`) and **non-convergent** (flat/worsening at finer N), not the a-priori-promised O(h²).

**Where the O(h) is — the viscous operator is NOT it.** An a-priori truncation test of the FV
viscous operator itself (`tests/study/fv_operator_truncation.py`: manufactured field `u = sdf`, so
`∇u = n̂`, `∇²u = 2/r`, and the wall gradient `g_a = n_a` is *exact* — isolating the `o_f` face
fluxes) gives the discrete operator vs `∫_fluidCV(−μ∇²u)dV` at **clean O(h²)** (order 1.98–2.00,
N=32→128). So the face-centre two-point flux `μ o_f(U_i−U_nbr)` is second-order-consistent — Basilisk
was right, and my earlier "face-flux placement" hypothesis is **refuted**. The remaining O(h) is in
the **pressure coupling**, not the viscous operator.

**What was tried on the pressure side, and failed.** The spec's flagged missing piece — a wall
pressure term `−p_i·W_c` added to the momentum force (the mode-3 transpose has only the o-weighted
*face* pressure force, not the wall term of `−∮ p·n dA`) — was implemented and A/B'd. With the
physically-dissipative sign it is stable but makes the drag *worse* (N=32 +0.81%→+1.04%); the opposite
sign diverges. So the naive wall pressure term is not the fix either.

**Conclusion of the mode-4 arc:** the fully-FV *operator* is second-order (proven), but second-order
*drag* in the coupled solve needs the momentum operator, the incompressibility constraint (the
`divergOpen`/`centerToFaceWallAware` divergence), and the pressure force to be **mutually FV-consistent
and adjoint-paired simultaneously** — sharing one embedded-boundary geometry across all three. My
incremental patches each fix one operator against a projection built for a different geometry, so the
coupling stays inconsistent (mode 3: FV projection + FD momentum; mode 4: FV momentum + mode-3
projection; +wall-pressure: still a non-adjoint force). That simultaneous consistency is the Basilisk
`embed.h` reformulation — a from-scratch rebuild of the collocated cut-cell operator set, not an
increment on mode 0. Until it is done, **mode 0 remains the best of the collocated family and
`Solver` (staggered) the production choice.** Mode 4 is kept (default off) as the validated FV
machinery (cs, FV operator, defect correction) for that rebuild to reuse.

## The Basilisk `embed.h` port: true-normal wall drag (modes 5/6/7, 2026-07-05)

Following `doc/collocated_embed_port_plan.md`, the mode-4 axis-by-axis wall reconstruction was replaced
with the Basilisk `embed.h` **true-normal image-point gradient** (`embedDirichletGradient` in
`src/mac_approx_projection.hpp`): along the dominant-|n̂| axis, two image points 1/2 cells into the
fluid, transverse bi-quadratic interpolation of the cell-centred values, a quadratic fit for the O(h²)
wall-normal derivative, with a biased-linear `embed_interpolate` fallback anchored at the fluid home
cell for the near-tangent band and a 1-point degenerate fallback for slivers.

**Rung 0 (a-priori, `tests/study/fv_wallflux_apriori.py` variant C):** the true-normal reconstruction is
clean O(h²) on the Stokes sphere wall drag (order 1.8→2.2, +2.24%→+0.065% at N=32→192), **20× more
accurate than the axis-intercept** (variant A, O(h)) and with **0% truly degenerate** cells — the
biased-linear fallback keeps every fragment reaching the fluid instead of collapsing to the zero
solid-cell value.

**Modes** (`setFaceInterp`, collocated only; staggered + modes 0–4 stay byte-identical, regression
suite PASS): **5** = embed momentum (`embedViscousApply`) + the mode-3 wall-aware projection;
**6** = embed momentum + the plain (mode-0) face map + an **openness-weighted cell correction**
(`centerGradOpen`/`projectCorrectCenterOpen` = Basilisk `centered_gradient`: at a cut cell with one
closed face it applies the *open*-face pressure gradient at **full** weight, not the plain
`projectCorrectCenter` ½ — the O(h) under-correction that is analysis defect (b)); **7** = mode 6 with
the wall-aware flux constraint restored.

**Measured Z&H drag** (φ=0.125 sphere, robust steady protocol: incremental-rotational pressure — Chorin
is inaccurate at steady state — `dt=400`+warm-start, run to |ΔK|<5e-5 over 200 steps; the warm-detector
protocol fires on false plateaus and must not be used):

| N | mode 0 | mode 5 | mode 6 | mode 7 |
|---:|---:|---:|---:|---:|
| 32 | +1.00% | +0.11% | −1.94% | −0.75% |
| 48 | +0.69% | +0.68% | −0.95% | — |
| 64 | +0.59% | +0.77% | −0.27% | +0.47% |
| 96 | +0.40% | — | −0.29% | — |
| 128 | +0.31% | — | −0.21% | — |
| order | ~1.0 | plateaus | ~1.6 | crosses 0 |

Findings, in order of importance:

1. **The embed momentum is the right fix and it works** — the a-priori O(h²) wall gradient, wired as a
   defect correction (`embedViscousApply`, `Lu += μ·area·dudn`, area = |o_{a−}−o_{a+}|), improves the drag
   at every fixed N over the axis reconstruction.
2. **The projection is the remaining lever, and plain vs wall-aware BRACKET the truth.** Mode 5
   (wall-aware, mode-3) *over*-drags and plateaus (does not converge); mode 6 (plain + openness-weighted
   correction) *under*-drags but converges from below at **order ~1.6, below mode 0 for N≥64**; mode 7
   (wall-aware constraint + weighted correction) *over*-corrects and crosses zero. **Mode 6 is the
   cleanest** and the recommended embed configuration, but the tail stalls near −0.2% (order ~1.6, not a
   clean 2) — a residual constraint/pressure inconsistency the bracketing localises but does not close.
3. **The solid-cut-cell mask was the flat-wall bug (Rung 4 core, now fixed).** A grid-aligned channel
   wall at a fractional position gives a thin fluid sliver (cs≈0.26 at N=32) and a +7% centreline error.
   Root cause (profile dump): `ibmSolidMask`/`maskVelocity` zero **every** cell with sdf(centre)<0 —
   including *solid-centred cut cells* (cs>0), which for the embed FV scheme must hold their reconstructed
   near-wall velocity (the below-wall extrapolation U=−0.152). Masking them to 0 leaves the adjacent fluid
   cell under-constrained → a uniform +0.15 shift of the whole channel. **Fix** (`setSolid`, gated
   `faceInterp_>=5`): re-mask from `cs`, pinning only fully-solid cells (cs<1e-6). Flat-wall N=32 → the
   analytic profile to 0.04%; mode-6 flat Poiseuille is now ~2nd order (+0.67→+0.23→+0.07% at N=32/48/64,
   below mode 0 by N=64). Correct for mode 0's IBM (no-slip on the fluid rows), the mask was wrong only
   for the embed momentum.
4. **With the momentum correct, defect (a) — the cell→face flux map — is the sole remaining barrier.** The
   sliver fix flips Z&H mode 6 from an under-drag that fortuitously *cancelled* the constraint error
   (pre-fix −1.94→−0.21%, apparent order ~1.6) to a clean *over*-drag that grows (−0.05, +0.29, +0.42% at
   N=32/48/64) — the plain ½/½ `centerToFace` **over-counts** the curved-cut-face flux; the wall-aware
   centroid map (mode 7) *over-corrects* worse (+2.2%). Even the exact cell values give ½(U_i+U_j)=0.149
   vs the true face flux 0.161 (a 7% linear-interp deficit of a curved profile). This is the approximate
   projection reconstructing `uf` **from cell values** — the O(h) flux error the analysis diagnosed as
   defect (a). Basilisk avoids it by evolving `uf` as a **face-primary** variable, not interpolating it;
   closing p=2 here needs that face-centred constraint, not another cell→face reconstruction.

**Status:** the true-normal embed momentum (Rungs 0–2), the openness-weighted pressure correction (Rung 3,
defect (b)), and the solid-cut-cell sliver mask (Rung 4 core) are all in and each **individually correct**
— mode 6 is now pointwise-accurate on the flat wall. The **one** remaining barrier to 2nd-order *curved*-
wall drag is defect (a): the approximate projection's cell→face flux interpolation, which fundamentally
needs a face-primary `uf` (Basilisk centered.h) rather than any cell reconstruction — a larger change than
the operator swaps done here. Modes 5/6/7 kept default-off; **mode 0 / `Solver` remain the production
defaults** (regression suite byte-identical). Repro: `tests/study/collocated_zh_embed6.py` (mode 6 vs 0),
`tests/study/collocated_zh_embed.py` (mode 5), `tests/study/embed_flatwall_guard.py` (flat-wall sliver).

## Caveats (what is proven vs argued)

- **Proven (measured):** flat-wall exactness on both grids; staggered second-order and
  collocated first-order convergence of the sphere drag to Zick & Homsy; that it is not a
  measurement/averaging artefact; cross-checked on OpenMP (CPU) and CUDA (RTX 5080).
- **Argued (mechanism):** attributing the first-order loss to the boundary/geometry
  inconsistency + approximate projection is the standard understanding of embedded-boundary
  *drag* accuracy and is consistent with every control we ran, but we did not separately isolate
  the viscous-reconstruction contribution from the pressure-correction contribution. A clean way
  to do so before/while implementing: build the collocated momentum operator on the face-openness
  geometry and re-measure the order; and/or compare `dirichlet_gradient` vs axis-by-axis on a
  fixed field.

## References

- A. A. Zick & G. M. Homsy, *Stokes flow through periodic arrays of spheres*, J. Fluid Mech.
  115 (1982) 13–26 — the drag ground truth used here.
- S. Popinet, Basilisk `embed.h` (embedded boundaries): `cs`/`fs` fractions, `embed_geometry`,
  `dirichlet_gradient`, `embed_flux`; and the centered Navier–Stokes solver `navier-stokes/centered.h`.
- A. Almgren, J. Bell, W. Crutchfield et al. — approximate projection methods for the
  incompressible Euler/Navier–Stokes equations (the ABC projection used by `SolverColocated`).
- `src/cut_cell_ibm.hpp` (`ibmFillEntry`), `src/mac_cutcell.hpp` (`buildOpenness`/`ccFaceOpen`),
  `src/mac_pressure.hpp` (`buildCutcellOp`/`divergOpen`/`projectCorrect`),
  `src/mac_approx_projection.hpp` (`centerToFace`/`projectCorrectCenter`),
  `src/grid_layout.hpp` (`Staggered`/`Colocated` policies) — the code this note refers to.
```
