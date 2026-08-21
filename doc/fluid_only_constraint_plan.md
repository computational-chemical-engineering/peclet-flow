# The production fluid-only constraint: design plan (route 2b)

*2026-08-21. Companion to `collocated_invisible_subspace.md` (mechanism) and
`collocated_paper_plan.md` (results tracker). Status: Design A implemented (mode 14a);
B/C pending the ghost R=24/32 order verdict.*

## Why

The campaign established (note §4-6, tracker rows 23-38): a collocated aperture projection
whose constraint couples solid-centered pressure DOFs possesses an attractor family living
ENTIRELY in the solid-row flux-balance equations (fluid-side kernel of the gauge-exact
gradient = constants, measured); gradients that read those DOFs are O(h) (multiplier values,
three variants measured); the wall-banded blend is a stopgap whose margin dies at
(R>=16, dt>=600). The ghost projection -- a fluid-only constraint with directional closures --
is family-free, needs no stabilizer through dt=1e20, and its clean gaps
(-1.43 / -0.265 / +0.077 % at R=8/12/16) are an order of magnitude below every alternative.
But it is nonsymmetric (BiCGStab), 2.3-2.7x the cost, single-rank-guarded (collocated), with
an open np>=16 instability and a march-unstable (1,2) mixed mode. Goal: a SECOND realization
of the fluid-only property engineered for production -- symmetric (CG + existing CutcellMG),
MPI-clean, with the ghost implementation as the correctness oracle.

## The design family (cheap -> deep)

**A. Neumann-closed openness filter (mode 14a, IMPLEMENTED, `set_fluid_only_constraint`).**
Zero every openness face with a solid-centered side before the pressure stack consumes it.
One kernel; operator/divergence/correction/MG all stay consistent automatically; SPD, 7-point,
pointwise-local (MPI-trivial). Closure quality: Neumann-zero at closed faces -- throat flux
through solid-centered cells is not counted by the constraint. Expected O(h) with a modest
constant; measured, not assumed (R=8/12 running). Role: family baseline + the cheapest
uniqueness-restoring production candidate if its constant is small.

**B. Kron (star-mesh) elimination with aperture weights.** Eliminate each solid-centered node s
from the aperture graph by conductance-weighted averaging: new fluid-fluid couplings
g_ij += a_si a_sj / D_s (D_s = sum of s's apertures); face correction at a fluid-solid face
uses phi_bar_s = sum_j (a_sj/D_s) phi_j. SPD by construction (graph Laplacian with nonneg
conductances), fluid rows exactly divergence-free INCLUDING throat flux (operator == correction
by construction). Cost: couplings between fluid cells sharing a solid neighbor are distance-2
and DIAGONAL -> breaks the 7-point stencil -> rides the gp row machinery (its MG surrogate
hierarchy pattern), or needs a CutcellMG stencil generalization. Closure value error O(h) at
the averaged faces but with throat conservation -- expected between A and C.

**C. Directional consistent closures = the ghost projection.** O(h^2)-consistent values,
nonsymmetric. Already exists; the reference. If the clean ladder confirms 2nd order and B
disappoints, the fallback is hardening C: (i) re-diagnose the (1,2) mixed-mode and np>=16
instabilities with this session's gain-bound method (all three instabilities found this
session had the same lagged-stiff-term shape); (ii) lift the collocated guard.

**Symmetric consistent variant (kept open): CutFEM-style ghost penalty.** Keep solid DOFs,
pin them with an SPD penalty on normal second differences (consistent, symmetric); costs a
consistent near-wall mass-balance perturbation + non-7-point couplings. Revisit only if A/B/C
all disappoint on accuracy or cost.

## Decision tree (keyed to pending measurements)

1. Ghost R=24/32 clean rungs (job 25937956): if the +0.077% at R=16 DECAYS -> the fluid-only
   property is 2nd order; proceed. If it FREEZES -> the residual is closure-level (note Q3);
   quantify before choosing B vs C.
2. Mode-14a R=8/12 (running): constant + order of the crude end. If |gap| at R=12 < ~1% the
   7-point path stays alive; else B/C required.
3. Then: implement B on the gp rails; C2 battery + both-bed ladder for the chosen candidate;
   MPI bring-up (A is trivial; B needs ghost-width-2 coverage of the star couplings -- verify
   the a_si a_sj/D_s stencil fits G=2); at-scale bed (s116-class) march for robustness.

## Constraints carried from the suite

Defaults byte-identical (all behind flags; verified digit-for-digit after each edit); staggered
untouched; fixed-protocol comparisons (dt=600 collocated ladders, dt=60 where blend-limited);
never read order near the R~12-16 zero crossing without R>=24 rungs.
