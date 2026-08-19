# Removing the collocated accuracy ceiling — analysis and plan

*2026-08-19. Supersedes the "second order" reading in `collocated_second_order_open_problem.md`'s
status banner. Written after the R=32 ladders measured a non-vanishing bias, and after re-reading
the Basilisk `embed.h` port history, which had already named the same defect.*

## The problem

Refinement ladders to R=32 on two periodic sphere beds show the collocated projection converging
to a **fixed offset** from the staggered/continuum answer rather than to it:

| err % vs staggered k∞ | R=8 | R=16 | R=24 | R=32 | Δ(24→32) |
|---|---|---|---|---|---|
| φ=0.50 gauge-exact | −0.773 | +0.169 | +0.276 | **+0.288** | +0.012 |
| φ=0.50 ghost | −0.344 | +0.087 | +0.132 | **+0.142** | +0.010 |
| φ=0.60 gauge-exact | −1.387 | −0.026 | +0.306 | **+0.385** | +0.079 |
| φ=0.60 ghost | −0.562 | −0.022 | +0.167 | **+0.218** | +0.051 |
| staggered (reference) | +0.7/+0.9 | ~0 | ~0 | **−0.009 / 0.000** | — |

The march protocol was ruled out first: a 100× tighter tolerance moves k by nothing to six digits
(Snellius job 25805513), with ⟨u⟩ flat to seven digits and `maxdiv` at 3.9e-09.

**Why it matters:** the whole point of AMR is to drive error down by local refinement. A ceiling
that refinement cannot cross defeats that, so this blocks the AMR programme — it is not a
tuning issue to be lived with.

## Diagnosis — two independent lines, same defect

The Basilisk `embed.h` port (commits `db5b4aa`/`f5fde8c`/`6d412ec`/`03a71c6`) got the momentum side
to second order (true-normal `dirichlet_gradient` measured O(h²), 20× better than axis-by-axis) and
stopped at one barrier, recorded as **defect (a)**: *the approximate projection reconstructs the
face flux `uf` FROM cell values, so the ½/½ `centerToFace` over-counts the curved-cut-face flux.*
That is the same thing the R=32 ladders now measure from the outside.

Sorting every scheme we have on two axes makes the pattern obvious:

| scheme | wall no-slip reconstruction | constraint face flux | outcome |
|---|---|---|---|
| mode 0 / plain | axis-by-axis, O(h) | interpolated ½/½ | 1st order |
| gauge-exact (9) | axis-by-axis | interpolated ½/½ | fast → **ceiling +0.29/+0.39 %** |
| ghost projection | directional closures | interpolated | fast → **ceiling +0.14/+0.22 %** |
| modes 1–3, Seo–Mittal split | varies | interpolated | 1st order, *worse* than mode 0 |
| mode 10 (better quadrature) | axis-by-axis | interpolated, centroid quadrature | O(h), divergent on slivers |
| **Basilisk embed** | **true-normal centroid** | **face-primary** | 2nd order in the literature |

Everything in the "interpolated" column plateaus. Sophistication in the *closure* (ghost) halves the
bias but cannot remove it, because the flux is still reconstructed from cell data. Mode 10 shows the
repair is not free: improving the quadrature broke the telescoping row-sum that keeps D and G
adjoint, and losing conservation cost more than the quadrature gained. **Conservation, not accuracy,
is the binding constraint on any fix.**

## Design constraints (Frank, 2026-08-19)

- **C1 — variable placement.** Cell centres carry *conserved quantities* (density, momentum density
  expressed as velocity, energy). Faces carry the *volume flux*. Every flux is then
  `face volume flux × conserved quantity interpolated to the face`. This is the flux-form FV shape.
- **C2 — dt-independent steady states.** A face flux that depends on dt is unacceptable. This
  **disqualifies Basilisk's acceleration-event form as written**: `uf = fs·(face_avg(u) + dt·a)`
  followed by `uf −= dt·∇p/ρ` leaves `uf = face_avg(u) + dt·(a − ∇p/ρ)` at steady state, and since
  the viscous term lives in the *cell* equation the bracket does not vanish. Our current scheme is
  measured dt-independent (embed-port notes: "Z&H drag steady state is dt-independent") and that
  property must be preserved, not traded.
- **C3 — must generalise to AMR**, i.e. survive 2:1 coarse–fine faces. A face-primary flux is
  actually *favourable* here: refluxing across a C/F face is the natural operation on face fluxes.
- **C4 — robustness on tight throats.** No return to a fragmenting pressure graph needing a
  connectivity guard; keep the operator symmetric (CG) if at all possible.

## Candidate schemes for Step 2

**(A) Momentum-interpolated face flux with a dt-free coefficient.** The collocated-CFD answer to
exactly this problem: build `uf` from a face-local momentum balance rather than by averaging cells.
The textbook Rhie–Chow form uses `d_f = V/a_P`, and with `a_P ⊃ ρ/dt` that is dt-dependent — the
known weakness, and the reason the earlier survey excluded Rhie–Chow by fiat. But under C1 the
survey's exclusion deserves re-examination, and note our divided convention makes `a_P → μ`-dominated
at the large dt used for steady marches, so a **steady (dt-free) coefficient** is the natural
variant. Cheapest to test; needs a literature check on the dt-independent formulations before
building.

**(B) Flux-form conservative EB finite volume (EBChombo / Trebotich–Graves line).** Matches C1
exactly, is AMR-native (C3), and reaches second order on curved embedded walls in the literature.
Costs: a genuine rewrite of the momentum assembly, and it brings small-cell *redistribution*, which
is a robustness question against C4.

**(C) Basilisk acceleration-event face flux.** Ruled out as written by C2. The port's own notes
record the failure mode (−75 % drag at N=32) and two remedies — switch the momentum step to the
undivided convention solver-wide, or add an implicit face-flux body-force source — but both leave dt
inside `uf`, so neither satisfies C2. **Deprioritised.**

## Plan

**Step 0 — is the ceiling in the constraint or in the diagnostic? (running)**
The permeability has always been measured as the *cell* mean ⟨u⟩. The physically cleaner estimator
is the conserved face flux `mean(α_f·uf)`, which the divergence operator drives to be identical
through every x-plane. `spheres_bench.py` now reports both (`k_over_R2` and `k_over_R2_face`).
First signal at R=8/φ=0.50: the two estimators **bracket** k∞ (staggered +0.70 % cell, −0.33 %
flux), so the choice is worth ~1 % there and could change the whole comparison. If the bias
collapses in the flux metric, the ceiling is in the cell reconstruction and the fix is small; if it
persists, defect (a) is confirmed as a genuine constraint error.

**Step 1 — re-baseline the embed line (running).** Modes 5/6/7 are re-instated (they are the embed
port, not ablations — retiring them was an error) and added to the ladder as `col_embed6` /
`col_embed7`. Every previous embed conclusion was drawn at Z&H N≤128 or R≤16, i.e. inside the range
where the signed error crosses zero and nothing can be concluded. Ladders now run to **R=32** on both
beds, jobs `25817340`–`25817349`.

**Step 2 — build the face-primary flux (after 0/1 report).** Shape chosen against C1–C4; (A) as the
cheap probe, (B) as the principled target. Decide with Frank; do not start before Step 0 says where
the error lives.

**Step 3 — validate and propagate.** R=32 ladders on both beds plus Z&H, then AMR, where the same
face-primary structure must survive C/F faces.

## Open questions

- Does the flux estimator remove, reduce or preserve the bias? (Step 0 answers this.)
- Do the embed modes already beat gauge-exact at R=32, and do *they* plateau? (Step 1.)
- Is there a dt-independent momentum-interpolation formulation with a published curved-wall order?
  This needs a literature check before committing to (A).
- Does (B)'s small-cell redistribution survive contact-tight throats (φ=0.60, gaps of 0.0002 R)?
