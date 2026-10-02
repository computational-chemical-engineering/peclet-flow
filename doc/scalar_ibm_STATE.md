# Scalar transport with immersed solids — campaign STATE

*Rewritten in place, never appended. History, every number and every rejected variant go in
`scalar_ibm_log.md`.*

## Objective (Frank, 2026-10-02)

Design scalar transport in the presence of solids properly, as a peclet suite feature. It must
cover:

- Dirichlet;
- Neumann, homogeneous and prescribed flux;
- conjugate transport (partition coefficient K, contact resistance);
- Robin.

It must generalize to AMR and to VoF / level-set interfaces. The method comes from the literature,
adapted to peclet: GPU, MPI blocks, matrix-free compact stencils, MG/Krylov, SDF plus apertures,
and a divergence-free MAC face flux. Then a plan, and a step-by-step implementation.

## Where we are

- **Code.** Branch `scalar-ibm` in worktree `suite/flow-scalar-ibm`, off `origin/main` 39680a4.
  Nothing is pushed.
- **Prototypes.** In `tests/study/scalar_ibm/` (2-D NumPy, disc and composite disc, exact
  answers). Run them with the suite venv.
- **Lead candidate: "probe-flux cut-cell FV".**
  - Every cell with fluid fraction κ > 0 is an unknown, with κ in storage and sources.
  - Faces carry the plain aperture two-point flux. The face-centroid correction is not needed.
  - Each cut cell has one wall facet: area vector from the apertures, centroid, normal.
  - The wall gradient is (u(probe) − u_Γ)/s, with the probe at s = 0.7 h along the normal and
    u(probe) interpolated bilinearly (trilinear in 3-D). The reach is 2 cells.
  - The BC is eliminated per facet. Robin: u_Γ = D S/(k + D β). Conjugate: a 2×2 system per
    facet over the fluid probe and the solid probe, which covers K and the contact resistance.
  - **Measured:** second order at every Bi from 0 to ∞ (Dirichlet 3.5e-5 at 128 cells/diameter)
    and for conjugate D_s/D_f from 0.1 to 100, K = 3 and finite h_c (all ≤ 2e-4 at 128).
- **Solver.**
  - BiCGStab, preconditioned by MG on the SPD "lumped-probe" surrogate: aperture FV + κ/dt +
    diagonal wall term A_w/(h²(0.7h + 1/k)).
  - Measured with an exact preconditioner: 8–9 iterations for Dirichlet and 2–5 for Robin,
    independent of resolution. Plain Gauss–Seidel on the probe operator also converges.
- **Rejected:**
  - flow's per-cell Dirichlet mask (order 0.7–1);
  - unit storage (order 1);
  - Gibou/Papac/hybrid "aperture + link" (no single symmetric scheme is 2nd order at all Bi);
  - centroid two-point and series-resistance wall fluxes (order 1);
  - the quadratic normal probe (erratic).
- **Literature** (L1 cut-cell/EB, L2 interface/conjugate/VoF species, L3 particle-resolved and
  benchmarks) is running. Digests go to the scratchpad `lit/`. Copy them into
  `doc/scalar_ibm_literature/` when they land.

## Next action

1. Fold the literature into the design brief. Check the probe-flux FV against AMReX EB / Schwartz
   et al. 2006, Mittal image points and Bochkov–Gibou, and pick the small-cell advection treatment
   (SRD vs May–Berger explicit–implicit vs implicit upwind in cut cells). Prototype advection with
   κ storage if the literature does not settle it.
2. Write the architect brief (`doc/scalar_ibm_brief.md`) → architect design note
   (`doc/scalar_ibm_design.md`) with work orders and gates.
3. Implement in stages, each gated:
   - geometry + κ;
   - Neumann + closure;
   - Dirichlet/Robin probe;
   - solver;
   - conjugate;
   - advection small cells;
   - MPI;
   - GPU.

## Open questions (defaults)

- How κ and the facet centroid are obtained in 3-D. Default: plane-cube (PLIC) volume from the
  core VoF functions.
- The probe fallback when the stencil hits invalid cells (contacts, thin gaps). Default:
  renormalize over valid cells → longer probe → two-point with own value.
- Whether existing scalar paths stay bit-identical. Default: yes; the new path is opt-in per scalar.
