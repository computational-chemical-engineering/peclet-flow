# Plan: port Basilisk `embed.h` to the collocated cut-cell solver (2nd-order drag)

**Status:** implementation plan (2026-07-05), for a fresh session. Prereqs read (in order):
`doc/collocated_first_order_analysis.md` (why collocated is 1st-order), then
`doc/collocated_second_order_literature.md` (why Basilisk `embed.h` is the chosen scheme, and the
measured dead-ends). This doc is the spec.

## The one thing to fix (established, not re-open)

Across this project's two prior sessions everything *except* the momentum no-slip reconstruction was
tried and measured:
- FV viscous **operator** is 2nd-order **when fed a true-normal wall gradient** (proven a-priori,
  `tests/study/fv_operator_truncation.py`, order 1.98–2.00).
- FV-momentum defect correction (mode 4): stable but 1st-order; +wall-pressure term: worse;
  Seo–Mittal pressure-only split (modes 1–3): **all 1st-order, all worse than mode 0** — because the
  mass-conserving cut-cell pressure Poisson is *already* in mode 0.
- **The ceiling is the momentum no-slip: the Robust-Scaled overlay is AXIS-BY-AXIS (six 1-D link
  intercepts θ_k) = O(h) at a curved wall.** A first-order no-slip caps the drag order regardless of
  the projection.

**So the single change that matters:** replace the axis-by-axis intercept wall gradient with Basilisk's
**true-normal image-point gradient** (`dirichlet_gradient`), and drive the viscous flux, the pressure
Poisson, and the projection reconciliation **all from one geometry** (`cs`, `fs`, boundary centroid
`b`, normal `n̂`). That single-geometry consistency is what mode 4 lacked (it paired a true-ish wall
flux with a mode-3 projection built on different geometry).

## What Basilisk embed does (the algorithm to copy)

Sources to read first: `basilisk.fr/src/embed.h`, `basilisk.fr/src/navier-stokes/centered.h`, and
Ghigo–Popinet–Wachs `hal.science/hal-03948786` (§ geometry + `dirichlet_gradient`). Core pieces:

1. **One geometry per cut cell** (`embed_geometry`): volume fraction `cs∈[0,1]`, face fractions
   `fs` (= our face openness `o_f`), boundary-fragment area, barycenter `b` (relative to cell centre),
   inward unit normal `n̂` (from the reconstructed interface / ∇sdf).

2. **`dirichlet_gradient(u, u_D=0, n̂, b)`** — the load-bearing function. Along the largest-|component|
   axis of `n̂`, pick **one or two image points** at `d1,d2` into the fluid; get `u(image)` by
   transverse (bi)linear interpolation from the neighbour cells; fit a quadratic through
   `{u_D at wall, u(d1), u(d2)}` → the O(h²) wall-normal derivative. **Falls back to the 1-point
   (linear, O(h)) estimate at degenerate slivers** — this is Basilisk's small-cell robustness (no
   redistribution, no cell-merging; global 2nd order preserved, only pathological cells drop to 1st).

3. **Embed Laplacian**: `∇²u|_cell = (1/(cs·Δ²)) [ Σ_f fs·(u_nbr − u_cell) + embed_flux ]`, where
   `embed_flux = boundary_area · μ · dirichlet_gradient` (the wall contribution). Face fractions weight
   the axis fluxes; the boundary flux is the wall.

4. **Pressure Poisson (projection)**: the **same** embed operator with a **homogeneous-Neumann** wall
   (∂p/∂n=0 → no embed wall flux, but `fs`-weighting + `cs`-normalisation + same `b`,`n̂`). This is
   close to our current `cutcell_pressure`, but it must share the exact same `cs`/`fs`/`b`/`n̂` as the
   momentum operator — that shared geometry is the point.

5. **Projection reconciliation** (centered.h): project the **face** velocity `u_f` (fs-weighted,
   approximate projection — we already do this), then correct the cell velocity with the embed-aware
   centered pressure gradient. Keep the ABC structure; only the cut-cell weights/geometry change.

## Implementation ladder (each rung validated before the next)

**Rung 0 — a-priori de-risk `dirichlet_gradient` (numpy, no build).** Extend
`tests/study/fv_wallflux_apriori.py` with a **true-normal image-point** variant (vs the existing
axis-anchored foot-point). On `u = sdf` (∇u=n̂) and a smooth manufactured field, confirm the
true-normal gradient is clean O(h²) *and* better-conditioned near slivers (with the 1-point fallback).
Gate: don't touch C++ until this reproduces O(h²). *(Cheap; mirrors what de-risked every prior step.)*

**Rung 1 — geometry.** Build/verify the single geometry as device fields: `cs` (improve on mode-4's
4³-subsampled `buildCellFraction` if the truncation test shows it limits order — consider an analytic
plane-cut fraction from the SDF), `fs` (reuse `ox_/oy_/oz_`), boundary barycenter `b` and normal `n̂`
(from ∇sdf, already computed in `fvViscousApply`). One kernel, one-time at `setSolid`.

**Rung 2 — viscous no-slip as the embed Laplacian (new `set_face_interp` mode, e.g. 5 = "embed").**
Keep the 7-point implicit matrix + GS smoother; put the `fs`-weighted axis Laplacian, `cs` time term,
and the **normal-aligned diagonal part** of the wall flux **in the matrix**; **deferred-correct the
transverse image-interpolation part** of `dirichlet_gradient` (small, O(h)) in the RHS — reuse the
mode-4 defect plumbing (`stencilMatvec`, the `bb = fvM − ω·rs·(fvL − b_FV)` RHS, `set_fv_relax`). The
two mode-4 bugs are already documented and fixed-in-principle: wall term sign is `−μΣW·g`; never read a
solver member inside a `KOKKOS_LAMBDA`. Gate: flat-wall Poiseuille stays exact/2nd-order; mode 0
byte-identical; Z&H error *drops* vs mode 0 at fixed N.

**Rung 3 — same geometry into the pressure Poisson + projection.** Route `cutcell_pressure` and the
projection reconciliation through the identical `cs`/`fs`/`b`/`n̂`. This is the consistency step that
should convert Rung 2's "better but maybe still <2" into clean order → 2. Gate: **Z&H observed order →
~2 with error below mode 0** at N=32/48/64/96/128 (mode-0 baselines +1.00/+0.68/+0.60/+0.40/+0.30%).

**Rung 4 — small cells.** If slivers misbehave, adopt Basilisk's remedy (dirichlet_gradient 1-point
fallback + `cs` limiting) rather than leaning on `D_rescale`; decide whether the embed operator
replaces the Robust-Scaled overlay at cut cells or coexists.

## Guards (hard, every rung)

- Staggered `Solver` **untouched / byte-identical** (mode is collocated-only).
- Mode 0 collocated **byte-identical** (embed is opt-in behind the new mode).
- Flat grid-aligned wall: Poiseuille pointwise-exact in modes 0–3, 2nd-order in the embed mode.
- Regression suite green (`tests/regression/sdflow_regression.py`).

## Reusable machinery already in the tree

`src/mac_approx_projection.hpp`: `buildCellFraction` (cs), `stencilMatvec` (M·u), `fvViscousApply`
(o_f faces + wall drag — swap its axis foot-point gradient for the true-normal `dirichlet_gradient`),
`ccSampleExt` (trilinear sampler for image points), `buildFaceCentroidDist` (centroid geometry),
`centerToFaceWallAware`/`transposeGradWallAware` (adjoint projection map). `src/flow_ibm.hpp`: the
mode-4 defect-correction RHS branch, `set_fv_relax`, the `cs_`/`fvM_`/`fvL_` fields, the ghost-fill
gate. Validation: `tests/study/collocated_zh_ab.py` (A/B harness), `tests/study/fv_wallflux_apriori.py`,
`tests/study/fv_operator_truncation.py`. CUDA build: `build_cuda2` (`export
PATH=/usr/local/cuda-13.2/bin:$PATH`; ~2–3 min/build). Commit at each validated rung; don't push.

## Build/run quick reference

```bash
cd flow && export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake --build build_cuda2 -j                       # or build_mpi for OpenMP
PYTHONPATH=$PWD/build_cuda2 .venv/bin/python tests/study/collocated_zh_ab.py   # A/B vs mode 0
```
Z&H A/B success = collocated observed order → ~2, error below mode 0's +1.00/+0.68/+0.60/+0.40/+0.30%
at N=32/48/64/96/128.
