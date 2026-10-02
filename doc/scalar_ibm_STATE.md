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
- **Literature** is done: L1, L2, L3 plus the building-block recon R, in `doc/scalar_ibm_literature/`.
  It corroborates the method. AMReX EB Dirichlet uses the same probe but has no EB Robin.
  Crockett–Colella–Graves 2011 is the conjugate 2×2 form. Zhao 2026 uses the PLIC interface as an
  EB.
- **Architect brief** `doc/scalar_ibm_brief.md` (eb1d9b7). The architect is writing
  `doc/scalar_ibm_design.md`.
- **Build tree** `build_dev` (host-openmp, MPI, tests, ccache, -march=native). The baseline is
  61/61 single-rank ctests passing; the 1-thread state hashes are in
  `doc/scalar_ibm_baseline_hashes.txt` (77fc506).

## Next action

The design note `doc/scalar_ibm_design.md` (95e4a55) is the contract. Execute its work orders WO-1 …
WO-10 in order through the opus-implementer agent. Each WO must pass G12 (the state hashes) and its
own gates.

| WO | status |
|---|---|
| WO-1 core kernels (core worktree `suite/core-scalar-ibm`, branch `scalar-ibm`) | IN PROGRESS |
| WO-2 flow geometry record | next |
| WO-3 operator + Krylov | |
| WO-4 ScalarMG | |
| WO-5 advection/small cells | |
| WO-6 closures | |
| WO-7 conjugate | |
| WO-8 contacts | |
| WO-9 backends/performance | |
| WO-10 docs/register | |

**Side study running:** `tests/study/scalar_ibm/packing2d.py`, near-contact conjugate (square
cylinder array, gap 0.2→0.01, k_s = 100 and 0.01): P2F vs Peters' hybrid ladder. It feeds WO-8 /
Q4. Results go in `pack_ks*.out` → the log.

## Open decisions (defaults; DEFAULT-PENDING-USER in design §13)

- **Q5** API spelling `add_scalar(..., cutcell=True)`.
- **Q6 — DECIDED by Frank (2026-10-02): FIX IT.** Scope (orchestrator): the whole scalar/energy API
  surface (add_scalar D, the phase-change thermal properties, …), converted at the API boundary at
  use time; cell units stay bit-identical. Work is on branch `scalar-units`, worktree
  `suite/flow-scalar-units`, off origin/main, run by an opus-engineer.
  **MERGE ORDER:** `scalar-units` → `scalar-ibm` BEFORE WO-3. WO-3 must reuse its UnitScales helpers
  instead of adding its own `diffToInt`. Design §1.2 / §13 Q6 are to be amended in WO-10.
- **Q7** Cut-cell stays opt-in until G1–G13 pass and one release has shipped.
