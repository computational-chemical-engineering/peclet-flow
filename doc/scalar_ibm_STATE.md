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

**CODE COMPLETE (43ccd78). Waiting for Frank on:**
1. **Landing.** scalar-ibm (flow) + scalar-ibm (core: a031c6f, b1fcb6a). Core goes first and is
   tagged before the flow pointer. Push / main is Frank's call.
2. **Conjugate contact model** (§13 Q4; G13 conjugate row OPEN). Recommendation: a body-aware geometry
   record plus a contact conductance, designed by the architect with Frank's input.
3. **Naming:** set_scalar_tolerance → set_scalar_residual_tolerance; maxit → max_iter; the
   `cutcell=` spelling (Q5); when cut-cell becomes the default (Q7). See doc/scalar_ibm_naming_rows.md.
4. **Register entries:** doc/scalar_ibm_register_entries.md + doc/scalar_units_register_entry.md, to
   be placed in the umbrella at landing.
5. **Battery cadence:** per work order, or only at milestones.

**Follow-up work orders (not started):**
- an agglomerated exact bottom for ScalarMG (review finding 4);
- the outflow-plane-preserving velocity fill in the cut-cell advance (depends on uf-outlet-diag);
- the held snapped-normal probe option;
- WO-11 (steady deferred-correction Koren);
- collocated open faces;
- the tracer route (A4).

## Side thread awaiting Frank

Branch `uf-outlet-diag` (worktree `suite/flow-uf-outlet`) fixes the legacy scalar open-boundary bugs
(log 2026-10-03). Three questions are open:
1. c002c8e: high-side inflow face plane; changes collocated momentum there.
2. af88101: ghost = v at the Dirichlet inflow.
3. Its two register entries.
If it lands, the G12 `scalar` hash becomes 438db94e.

## Open decisions (defaults; DEFAULT-PENDING-USER in design §13)

- **Q5** API spelling `add_scalar(..., cutcell=True)`.
- **Q6 — DECIDED by Frank (2026-10-02): FIX IT.** Scope (orchestrator): the whole scalar/energy API
  surface (add_scalar D, the phase-change thermal properties, …), converted at the API boundary at
  use time; cell units stay bit-identical. Work is on branch `scalar-units`, worktree
  `suite/flow-scalar-units`, off origin/main, run by an opus-engineer.
  **DONE:** 6 commits db28d0c…f9a3d27, merged into scalar-ibm as 184ae9f.
  - Gates: scale invariance ≤ 1e-13, sine decay order 1.98, G12 12/12, battery 190/191 (one
    unrelated timeout, which passes on rerun).
  - Sub-decision (accepted): operator-mdot without set_phase_change_energy RAISES under an extent
    (an implicit internal ρc_p = 1).
  - Also fixed: `vof_interface_area` in plic mode on box cells.
  - Open: the q-kernels carry no w_a (pre-existing; box cells only).
  - Register entry text: `doc/scalar_units_register_entry.md`. It goes into
    ../docs/decisions/flow.md + DECISIONS.md at landing (umbrella shared checkout; not now).
  - The branch could land on flow main independently. Needs Frank's OK.
  Formerly: **MERGE ORDER:** `scalar-units` → `scalar-ibm` BEFORE WO-3. WO-3 must reuse its UnitScales helpers
  instead of adding its own `diffToInt`. Design §1.2 / §13 Q6 are to be amended in WO-10.
- **Q7** Cut-cell stays opt-in until G1–G13 pass and one release has shipped.
