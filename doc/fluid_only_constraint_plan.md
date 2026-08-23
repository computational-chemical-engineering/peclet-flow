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

## The pairing lesson (2026-08-22, added after Design B's failure)

Design B is implemented and its solve is verified consistent (SPD operator == correction,
clean PCG convergence) -- and the march still diverges violently, PM I INCLUDED, at every dt.
Since removing the pressure update does not help, the instability lives in the projection loop
itself: the gauge-exact directional cell gradient is structurally mismatched to the
star/aperture constraint rows (D Pi G_ge disagrees with A_B at O(1) on wall-band modes -> the
dt-free approximate-projection loop amplifies). Together with the session's other measurements
this crystallizes into a design principle:

  * STABILITY wants the (cell gradient, constraint) pair structurally MATCHED near the wall
    (mode 11's exact adjoint pair, and the ghost's shared directional closures, are the two
    stable examples; gauge-exact-vs-aperture, normalized-embed-vs-aperture, and
    gauge-exact-vs-star are the three measured unstable mismatches).
  * ACCURACY wants both sides built from O(h^2)-consistent VALUES (multiplier or averaged
    values give O(h): modes 11/12/13, and Design A/B's phibar).
  * The only architecture measured to satisfy both is the ghost's: constraint AND gradient
    derived from the same directional, O(h^2)-consistent closure family.

Consequence: the (b) endgame is not "a symmetric alternative to the ghost" -- it is a
production-hardened ghost-architecture scheme. Symmetry is bounded by this principle: the
directional closure rows are inherently one-sided, so exact SPD appears incompatible with the
stable+accurate corner (CutFEM ghost-penalty remains the one untested possible exception).
Pending confirmation: the fluidonly2_m13 pairing discriminator (star constraint + matched
aperture gradient must be STABLE, at mode-11-class accuracy).

## Decision tree (keyed to pending measurements)

1. Ghost R=24/32 clean rungs (job 25937956): if the +0.077% at R=16 DECAYS -> the fluid-only
   property is 2nd order; proceed. If it FREEZES -> the residual is closure-level (note Q3);
   quantify before choosing B vs C.
2. Mode-14a R=8/12 (running): constant + order of the crude end. If |gap| at R=12 < ~1% the
   7-point path stays alive; else B/C required.
3. Then: implement B on the gp rails; C2 battery + both-bed ladder for the chosen candidate;
   MPI bring-up (A is trivial; B needs ghost-width-2 coverage of the star couplings -- verify
   the a_si a_sj/D_s stencil fits G=2); at-scale bed (s116-class) march for robustness.

## B+ efficiency track (user-approved 2026-08-22): two deferred-correction variants

Both target the ghost FIXED POINT (accuracy identical to the ghost scheme by construction);
the win is replacing BiCGStab by CG/Chebyshev-able SPD solves (~2x on the pressure stage +
the comm-avoiding MPI lever). Retain BOTH if both pass, one as default one optional:

  * B+kron: base = the Design-B star operator (SPD, implemented), defect = A_ghost - A_star
    lagged in the RHS. Note B's SCHEME instability is irrelevant here -- the fixed point
    enforces the ghost constraint (the matched pair); only the defect-iteration spectrum
    spec(A_star^-1 A_ghost) matters.
  * B+sym: base = sym(A_ghost) (the binary 7-point part + symmetrized overlay -- rides the
    solvePCG SPD-overlay machinery built for Design B), defect = the skew part, lagged.
    Smaller defect, likely friendlier spectrum.

GATE RESULTS (bplus_gate.py, real bed, N=32 AND N=48, percolating component):
  * B+sym: FAILED. sym(A_ghost) is INDEFINITE at the closure rows -- a persistent family of
    small negative eigenvalues (8+ at both N, ~-1e-3..-2.5e-3), and even after low-rank
    Woodbury repair the preconditioned spectrum keeps real parts in [-4.5, +6.5] and a
    1 +/- 73j pair (damping bound w < ~4e-4 -> thousands of iterations). Dead.
  * B+kron: PASSED. A_star exactly SPD; spec(A_star^-1 A_ghost) ENTIRELY REAL POSITIVE:
    [~0.02, 2.22] at N=32, [~0.02, 2.97] at N=48; small isolated cluster at 0.003..0.05
    (through-wall star links the ghost lacks), tight bulk ~[0.9, 3].
  * CONSEQUENCE (sharper than the original B+ framing): the win would be PRECONDITIONING,
    not a stationary defect loop (the small-lambda cluster prices the latter out).
  * REVISED by the three-base comparison (2026-08-23, same gate): exact-base extreme spectra
    vs A_ghost -- binary (current surrogate) [0, 2.29], star [0, 2.22], filtered-aperture
    [0, 4.19]. The star base's lambda_max advantage over the current binary surrogate is
    MARGINAL: the fine-base mismatch is NOT what costs the ghost its 13.5 iters/decade
    (vs cut-cell 2.3). The efficiency study must look at the interior eigenvalue
    DISTRIBUTION and the V-cycle hierarchy quality instead -- a deeper diagnosis, queued
    behind correctness hardening. (The pairing-lesson preconditioning corollary still
    stands -- the star base is as good as binary -- it is just not BETTER.)

Priority: after C-hardening correctness (np>=16 root cause, guard lift); this is the
first efficiency milestone and it reuses tonight's code.

## AMR port plan (core::amr::AmrFlow -- the production branch's phase 5)

AmrFlow IS the collocated aperture scheme on the octree, so it inherits the attractor family
and the Layer-1 instability wholesale; the campaign's cure ports as follows.

1. Structural precondition (already true): cut cells live in the FINEST band by refinement
   policy, so the ghost closures never cross a level boundary -- the closure machinery is
   uniform-grid code on the finest level, and the shared peclet::core::scheme::ghost_closure
   kernels (lifted from flow in the July port) are already in core.
2. Port order: (a) the binary-openness surrogate + closure overlay on the finest level, with
   the level-boundary faces treated as COUPLED (they are fluid-fluid by (1)); (b) the
   gauge-exact gradient is already AmrFlow's gradient family -- verify its band variant reads
   only fluid-centered cells (the kernel-test methodology of kernel_study.py applies verbatim
   on the octree's flattened band); (c) BiCGStab on the AMR BiCGStab rails (they exist -- the
   ghost note's "(1,2) ports on the existing openness-MG rails" -- but port (2,2) ONLY);
   (d) the fragmentation guard on the octree band (component labeling on the leaf graph).
3. Validation ladder mirrors flow's: np1 bit-exactness vs the uniform-grid flow ghost on a
   uniform octree; family collapse (m1 -> 0) on the Z&H sphere with a refined band; then the
   bed ladder. The C2 dt-battery is the acceptance gate (AmrFlow's current scheme fails it
   by construction).
4. Do NOT port: the (1,2) mixed mode (march-unstable), the wall-blend (superseded by the
   scheme change), modes 11-14 (mechanism ablations, flow-only).

## Constraints carried from the suite

Defaults byte-identical (all behind flags; verified digit-for-digit after each edit); staggered
untouched; fixed-protocol comparisons (dt=600 collocated ladders, dt=60 where blend-limited);
never read order near the R~12-16 zero crossing without R>=24 rungs.
