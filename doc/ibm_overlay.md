# The IBM as a sparse overlay (octree/AMR forward-compatible)

How the velocity (momentum) cut-cell IBM is structured in flow, and exactly what an octree/AMR port has
to replace. Companion to the face-based pressure operator (`doc/flow_multigrid_plan.md`,
"Forward-compatibility" section). Code: `src/mac_ibm.hpp`, `src/cut_cell_ibm.hpp`, `src/flow_ibm.hpp`.

## Principle: draw the symmetry boundary at the projection
Not every operator carries the same obligations:
- **Projection (pressure)** must be face-based, α-weighted, and **symmetric** — that's what makes the
  projection orthogonal, the Poisson matrix SPD, and multigrid grid-independent. It already is
  (`mg_build_op_k`: `A = −div(open·grad)`, the openness α_f is the per-face factor; no row scaling; solid
  faces decouple via α_f=0). **Nothing to change here.**
- **Momentum (velocity)** only ever meets a Gauss-Seidel smoother. It is the **Robust-Scaled, row-based**
  cut-cell IBM, and it should **stay** row-based:
  - Row scaling by `D_rescale` is **invariant under Gauss-Seidel** (the factor cancels between the scaled
    diagonal and the row sum), so the smoother neither knows nor cares.
  - Non-symmetry / non-conservation across an immersed boundary is **physically legitimate** for momentum
    — the wall exerts a force; momentum isn't conserved across the interface.
  - The `X` cross-multiplier (folding one neighbour's coefficient into the *opposite* neighbour) is
    irreducibly **row-structured**; forcing it into face form would be wrong, not just hard.
  - Corollary we verified the hard way: **never multigrid the momentum operator** — row scaling is
    GS-invariant but *not* MG-invariant (residual restriction sums differently-scaled rows; the
    coarse-grid correction goes blind at the boundary skin). Our Phase-2 rediscretized velocity-MG
    diverged for exactly this reason; the fix is "use RB-GS for velocity," which is the default.

So the momentum IBM is not a competitor to the face architecture — it is a **sparse overlay on top of
it**. That keeps the row-based scheme *and* the forward-compatibility.

## The structure: a base face operator + a sparse overlay
```
  momentum operator  =  BASE (mesh-agnostic face loop)  ⊕  IBM OVERLAY (sparse, row edits)
```

| layer | what | code |
|---|---|---|
| **Base operator** | `A = I − βL` (β per face = νΔt·gf); a face loop, like the pressure operator | `ibm_build_diffusion_k` |
| **Overlay (data)** | a sparse SoA of cut cells: per cell a `cell_index` (handle), per-face `dir_code` (neighbour/direction hook) + coefficients `R/K/M/X/Nbc`, and `D_rescale` | `IbmOverlay` (`src/cut_cell_ibm.hpp`) |
| **Overlay apply** | loop the overlay, modify each cut cell's **own** row (diagonal ×`D_rescale`, off-diagonals via `K/M/X`, the Dirichlet `inhom`) | `ibm_modify_stencil_k` |

The build/apply split already exists: `IbmSolver::setSolid` builds the overlay once from the
geometry, and the per-step velocity solve applies it onto the base stencil.

## The two provider boundaries — *all* an octree port replaces
Everything mesh-specific is isolated behind two providers; the numerics (base, overlay coefficients, the
apply math) are mesh-agnostic.

1. **GeometryProvider** — geometry → overlay entries.
   - Cartesian: `build_ibm_overlay` (`src/flow_ibm.hpp` / `src/cut_cell_ibm.hpp`) = count cut cells (`ibm_count_ext_k`), allocate
     the SoA (`ibm_alloc`), fill it (`ibm_geometry_ext_k` → `ibmFillEntry`). `ibmFillEntry` is already
     **indexing-agnostic** — it consumes the 7 SDF samples + a cell handle and emits the Robust-Scaled
     polynomials; it does not care about the layout.
   - Octree: a tree-walk that gathers per-cell SDF and calls the **same** `ibmFillEntry`. One call site
     (`build_ibm_overlay`) changes.

2. **Connectivity** — per cut-cell face: (base-stencil slot, opposite face for the `X` cross-term).
   - Cartesian: the implicit 7-point — slot order is the fixed `{A_E,A_W,A_N,A_S,A_T,A_B}` in
     `ibm_modify_stencil_k`, and `OPP[6]={1,0,3,2,5,4}` (= `k^1`) is the opposite-face map.
   - Octree: supplied per overlay entry via `dir_code` (neighbour handles) and a tree-defined "opposite";
     the apply math is unchanged.

## Octree extension steps (deferred until octrees are committed)
Not built now — like the agglomerated coarse solve and the at-scale Chebyshev benchmark, this is forward
compatibility, and the abstraction should be hardened against a *real* second mesh, not in the abstract.
When octrees land:
1. **Variable face count.** `IbmOverlay` is fixed-6 (Cartesian). Introduce a variable-arity overlay entry
   (offset+count into a flat per-face array), or a separate variable-arity overlay type (leave the
   fixed-6 `IbmOverlay` intact for the Cartesian path). The apply loops `[0, num_faces)` instead of `[0,6)`.
2. **Cell handles, not linear indices.** `cell_index` and the neighbour references become tree cell IDs;
   the apply addresses rows/neighbours through the tree's accessor instead of `i±stride`.
3. **Connectivity for the `X` opposite term.** Define "opposite face" in the tree (a coarse cell faces
   several fine cells on one side — pick the aligned coarse-direction partner, or split `X` across the
   fine partners). This is the one place the row-structured cross-term needs a tree-specific rule.
4. **Same multi-GPU discipline.** Overlay entries whose stencils reach into a halo region must be
   generated on **both** owning ranks (the halo-width-2 rule already used by the TVD limiter); on the
   octree the overlay lists are rebuilt at adaptation alongside the face connectivity, by the same
   rebuild-phase machinery.

The base operator and `ibmFillEntry` need no octree-specific changes — only the two providers above.

## Verification
The Cartesian implementation is **bit-identical** to before (the connectivity refactor uses `OPP[k]==k^1`
and the same slot order). The `tests/kokkos_mpi` IBM stencil tests compare the baked stencil cell-for-cell against
a serial reference using the same kernel; together with `ns_solid`/`stokes_solid`/`ibm_poiseuille` and the
Zick & Homsy validation, it certifies the overlay refactor changed nothing numerically.
