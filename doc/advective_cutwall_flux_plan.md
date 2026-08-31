# Advective cut-wall flux — campaign plan (scoped 2026-08-31, awaiting budget)

**Status: SCOPED, NOT STARTED.** Execute in a fresh session. Read this file plus
`suite/docs/ANALYTIC_SDF_GEOMETRY.md` §7 item 8 (with the 2026-08-31 addendum) and the
`peclet-examples` page `ten-cate-sphere` + its ISSUES.md entry before writing code.

## The problem, with its evidence

The momentum advection operators are **geometry-blind**. `sadv::advect / advect_sou / advect_fou`
(`src/staggered_advection.hpp`) and the implicit `fou_operator` compute conservative flux-form
divergences with **unit face areas** over the full grid; `buildRhs*` (`src/flow_ibm.hpp:3205`)
evaluates them at every momentum row and multiplies by the rescale. Two distinct defects follow:

- **D1 (O(1), moving bodies only): masked rows hold ZERO, not the wall velocity.**
  `maskVelocity` (`flow_ibm.hpp:4643`) pins solid u-DOFs to 0.0 after every solve. `adv_vel`
  averages those zeros into the advecting face velocity, and the SOU/Koren `PHI` stencils read
  them up to 2 cells into the solid. For a static wall, 0 *is* the wall velocity, so only D2
  remains — which is why the static 4-sphere bed measured a benign `sum A` of −0.4…−1%
  (§7 item 8's original table). For a body moving at `u_wall`, the advective term near the body
  is wrong by O(u_wall) — an O(1) local error in exactly the term that produces the finite-Re
  screening of wall corrections. This is the same defect family the fresh-cell fix
  (`set_fresh_cell_seed`, flow `1a01769`) closed for the TIME term (u^n of uncovered cells);
  the advective term never got the analogous treatment.

- **D2 (O(h), all cut walls): no apertures, no wall-flux closure.** Faces cut by the wall carry
  full-area fluxes; the wall segment inside a cut cell carries none of the (continuum-zero, but
  discretely non-cancelling) closure. This is the leak §7 item 8 measured and deliberately
  deferred ("report, don't fix") at percent scale.

**Measured consequences** (ten-cate campaign, all probes reproducible from
`peclet-examples/examples/ten-cate-sphere/` + its ISSUES.md entry; probe scripts were session
scratchpad, the recipes are in the ISSUES text):

| observation | value | reading |
|---|---|---|
| E1 (Re 1.5) settling plateau, d/h = 8/12/16 | u/u∞ = 0.781/0.777/0.749 vs measured 0.947 | resolution-flat at the **creeping** confined value (K≈1.67 vs 1.38) — screening never develops |
| invariances | dt×½, sweeps 60→200, SOU→Koren, advection off (0.794) | not a discretization-tuning issue |
| E3/E4 (Re 11.6/31.9) coupled falls | peak 2.16/1.87 × u∞ | **unphysical** — no drag law permits exceeding u∞ |
| Newton audit, towed sphere at E4 speed | F_sphere + F_tank = **+0.32 W** (adv on) vs −0.001 W (adv off, E1) | advective budget leaks at the moving cut wall |
| unconfined control (periodic box + back-pressure) | settles at 1.03 of screened expectation | solver healthy without the moving-wall/confinement interaction |
| tow ↔ free-fall cross-check (E1) | agree to 2% | coupling loop sound; the drag itself is wrong |
| static 4-sphere bed (§7 item 8) | sum A = −0.965%/−0.369% (N=32/48), O(h^2.4) | D2 alone is percent-class and converging |

**Working hypothesis H1:** D1 dominates every moving-body symptom; D2 is the static-class
remainder. H1 predicts the whole table above, including the static/moving asymmetry. Rung A0
tests H1 before any deep FV work is committed.

## Design

### Rung A0 — wall velocity into the advection inputs (H1 test; small, high expected yield)

Do NOT change the global mask convention: the masked zeros are load-bearing elsewhere
(e.g. `flow_ibm.hpp:4213` — "faces whose two adjacent centers are fluid, so the masked
solid-cell zeros never enter"; the viscous/pressure operators and the budget's telescoping
assume them). Instead give the advection evaluation its own input:

- Before the RHS build (once per step, after instance motion is set — NOT per Picard iteration:
  solid rows depend on wall motion only), fill the solid-masked rows of a scratch copy of
  `C[c].u` with the local rigid-body wall velocity `u_wall(x) = v_inst + ω_inst × r` — the
  machinery `buildWallVelocity`/`uBc_` already computes exactly this (reuse it; do not
  re-derive). Fill depth: the kernels reach ±2 cells, so 2 cells of solid suffice; filling every
  masked row is simpler and equally correct.
- Pass the scratch views as U/V/W **and** PHI to `advect_sou/advect/advect_fou` and to the
  implicit path's `fou_operator` (`buildAdvStencil*` — the implicit FOU sees the same zeros; both
  paths or the deferred correction `rho*(aF − aK)` becomes inconsistent).
- MPI: ghost solid rows also hold zeros. The scene is analytic — `u_wall` is computable pointwise
  from instance state with no exchange. Fill after the halo exchange, in the extended block.
- The advRhs_ stash (`ensureAdvStash`, reaction budget R0) then carries the corrected term
  automatically — budget and operator stay one object. `reaction_budget_terms()` needs no change.

Cost estimate: ~3 scratch CCFields (or one fused fill/restore pass), one kernel, plumbing in 4
RHS builders + 2 stencil builders. Byte-identical when no instance moves and every wall velocity
is zero — **make that a stated gate** (static scenes must not change at all; that keeps every
Stokes-era validation untouched by construction).

### Rung A1 — aperture-weighted cut-face fluxes + impermeable wall closure (D2; only if needed)

The FV form over a cut momentum cell: sum face fluxes weighted by openness `o_f`, wall segment
contributes ρ u_w ((u_w − w)·n) dA = 0 for a rigid impermeable no-slip wall — i.e.
aperture-weight the six fluxes and add nothing. The subtlety is the **volume**: the flux-form RHS
is per unit cell volume; aperture-weighted fluxes over a θ-volume cell need the 1/θ that raises
the classic small-cell stability problem for the explicit term. Options, in preference order:
1. θ-floored scaling (cap 1/θ at 1/θ_min, consistent with how the cut-cell viscous bake handles
   small cells — read `ibmModifyStencil` first);
2. flux redistribution to neighbours;
3. leave D2 unfixed if A0 closes the benchmark to the few-% level — D2 is O(h^2.4)-converging
   and already documented as a reported budget term.
Decide AFTER A0's measurements. Do not start here.

### Explicitly out of scope
- Collocated path (`cadv::`), porous `divAdv_` compensation, VoF coupling — the reaction budget
  is staggered-only (R0) and the benchmark is staggered. Do not touch shared pressure-driver
  code (concurrent VoF work lives there).
- Changing `maskVelocity` semantics globally.

## Gates (every rung; numbers, not adjectives)

1. **Bit-identity for static scenes**: any pre-existing gate with no moving instance must be
   byte-identical (the A0 fill writes only masked rows, and static walls have u_wall = 0 — zero
   fill = today's zeros). Includes the 18 `tests/kokkos_mpi` ctests np=1,2,4 and the regression
   suite (`tests/regression/sdflow_regression.py`, perf within baseline).
2. **Newton audit** (towed E4 sphere, d/h=8): F_sphere + F_tank from +0.32 W → target |<0.02 W|.
3. **Tow drag E1** (U = 0.947 u*): F/W from 1.244 → toward ~1.0 (the experiment's operating point).
4. **ten-cate ladder** (THE regression page — re-render `examples/ten-cate-sphere`, zero page
   edits): E1 peaks climb from 0.78 toward 0.947 (few-% at d/h=16); E3/E4 peaks drop below u∞.
5. **No regression on the validated moving-geometry set**: oscillating-sphere (Stokes 1851
   in/out-of-phase), moving-sphere-drag (SFO metric ≤ current 9.55e-04; Blackburn peaks within
   current ±0.5%), rotating-sphere torque gate (+2.4%), jeffery-orbit half-period (−3.4/−5.0% —
   expect same or better), spin-decay 1.0389.
6. **Static-bed sum A** re-measure (§7 item 8 table): unchanged after A0 (it is D2); reduced and
   still O(h)-converging after A1 if A1 is taken.
7. Distributed: np=1 bit-exact vs single-rank, np=2/4 within the established 3e-7/5e-12-class
   tolerances of the existing MPI gates.

## Traps (learned this campaign — do not relearn)

- `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` for batteries; dem numeric claims at 1 thread.
- MG-hostile grids: tank probes must keep multiples of 8 (NX=62 was 10× slower per step).
- Quarto exits 0 on a failed figure cell — grep rendered html for `cell-output-error`.
- Explicit resolved coupling at ρp/ρf ≈ 1.15 needs the virtual-mass stabilizer already in the
  ten-cate page (ma = 2ρ_f V_p, lagged compensation); its post-peak stop arms only past 0.6 u*.
- Periodic free-fall probes need the back-pressure body force or momentum accumulates.
- Off-lattice walls (the 0.3-cell shift) — grid-plane-aligned moving faces produce zero cut
  cells and go silently inert.
- Commit flow first, umbrella pointer LAST; stage named paths only.

## Budget & sequencing

- A0 implement + static-identity gate: ~1 session-day equivalent; probes (Newton, tow, ladder
  d/h=8) are minutes each on the RTX 5080; the full ten-cate re-render ~30 min.
- Decision point after A0: if gates 2–4 land, A1 is optional polish (schedule separately);
  if not, H1 is falsified — return to the probe ladder with the A0 machinery as instrumentation
  (it lets you difference "zeros vs wall-velocity in the advective term" directly).
- Follow-up once green: un-defer `drafting-kissing-tumbling` (E3b) as the second regression page.
