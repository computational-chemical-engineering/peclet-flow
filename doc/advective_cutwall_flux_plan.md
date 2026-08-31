# Advective cut-wall flux — campaign plan (scoped 2026-08-31)

**Status: RUNG A0 EXECUTED 2026-08-31 (flow `fb1a1a7`). H1 is HALF right — see
"A0 results" below before doing anything else.** A0 closed the moving-wall momentum LEAK and
removed a 10% finite-Re moving-body drag error against Blackburn, but it did NOT recover the ten
Cate benchmark: E1's confined plateau moved 0.781 -> 0.803 against a target of 0.947, and E3/E4
still exceed u_inf. The E1/E3/E4 failures therefore have a cause that is NOT D1, and the ladder
below (A1) is no longer obviously the next rung. Read this file plus
`suite/docs/ANALYTIC_SDF_GEOMETRY.md` §7 item 8 (with the 2026-08-31 addenda) and the
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

---

## A0 results (2026-08-31, flow `fb1a1a7`)

**What shipped.** `buildAdvInputs()` fills three scratch fields (`uwAdv_[3]`) with `C[c].u` and
overwrites the solid-masked rows with `uBc_` — the local rigid-body wall velocity
`v_inst + omega_inst x r` that `buildWallVelocity` already evaluates at component c's staggered
points over the whole extended block. `advVelView(c)` hands those to the three explicit RHS
builders (`buildRhs` / `buildRhsForced` / `buildRhsVar`), to both implicit-FOU stencil builders
(`buildAdvStencil` / `buildAdvStencilVar`) and to `VelocityMG::restrictAdvVelocities`. Filled once
per Picard iteration (the fluid rows are `u^k`; the wall rows depend only on instance motion).
`maskVelocity`'s global convention is untouched. Gated on `hasMotion_`; **`PECLET_FLOW_ADV_WALLVEL=0`
is the ablation** and reproduces every pre-A0 baseline to 3-4 significant figures, which is how the
before-columns below were measured in the same binary.

### Gate 1 — static byte-identity: **PASS**
`tests/kokkos_mpi` 60/60 (np = 1, 2, 4, CUDA); `tests/regression/sdflow_regression.py` PASS with
every metric at +0.00 % and every pressure-iteration count identical.

### Gates 2-4 — the moving-body probes (d/h = 8 unless noted)

| probe | before (A0 off) | after (A0 on) | target | verdict |
|---|---|---|---|---|
| Newton audit, towed E4 (Re 32), `sum F/W` | **+0.3217** | **−0.0329** | \|<0.02\| | essentially closed |
| Newton audit, towed E1, adv on | +0.0346 | −0.0014 | — | closed |
| Newton audit, towed E1, adv OFF (control) | −0.0011 | −0.0011 | unchanged | held |
| Newton audit, towed E4, adv OFF | −0.0695 | −0.0695 | — | the probe's own non-advective floor |
| Tow drag E4, `F/W` at U = 0.955 u* | **1.5421** | **1.0866** | ~1.0 | large move |
| Tow drag E1, `F/W` at U = 0.947 u* | 1.2447 | 1.2046 | ~1.0 | small move |
| ten-Cate E1 free fall, peak u/u_inf, d/h = 8/12/16 | 0.781 / 0.777 / 0.749 | **0.803 / 0.797 / 0.766** | 0.947 | **NOT reached** |
| ten-Cate E3 / E4 coupled peaks (× u_inf) | 2.16 / 1.87 | **2.03 / 1.77** | < 1 | **still unphysical** |
| periodic unconfined control, plateau | 1.0322 | **1.0843** | ≈1.03 | moved 5 % |

The E4 Newton residual that motivated the whole rung falls by an order of magnitude and is now
*smaller than the same probe's advection-OFF residual* (−0.0695), i.e. what is left there is not
advective. Note the E4 tow drag says the drag at 0.955 u* now EXCEEDS the weight (1.087), so the
towed configuration predicts a terminal velocity slightly below u_inf — while the coupled E4 fall
peaks at 1.77 u_inf. **That tow ↔ free-fall inconsistency at high Re is new information and is the
sharpest remaining thread** (at E1 the two agreed to 2 %, which is what made the old diagnosis look
complete).

### Gate 5 — validated moving-geometry set

- `galilean_gate.py`: 7.99e-07 with the rung-3 term ON, rung-3-OFF contrast still 1.24e+06x —
  **unchanged** (advection off in that gate, so A0 is a strict no-op there).
- `rotation_gate.py` A: +3.47 / +2.42 / +2.89 %, net force 5.3e-13 — **unchanged** (advection off).
- `fresh_cell_gate.py`: SFO `RMS(F_2delta)/|F1|` = 9.553e-04 at SPP = 200 seed = 1, dF1 −0.04 % —
  **unchanged** (advection off).
- `jeffery-orbit`, `oscillating-sphere`, `rotating-sphere-torque` pages all call
  `set_advection(False)`, so A0 cannot reach them: byte-identical by construction.
- **`moving-sphere-drag`, Blackburn (2002) finite-Re peak `Cd` — the one advection-ON page, and
  the strongest external evidence FOR A0:**

  | L/D | R/h | δ/h | peak Cd before | before vs Blackburn | peak Cd after | after vs Blackburn |
  |---|---|---|---|---|---|---|
  | 7.5 | 6.4 | 4.05 | 4.7259 | **+10.16 %** | 4.2572 | **−0.76 %** |
  | 10.0 | 6.4 | 4.05 | 4.7306 | **+10.27 %** | 4.2617 | **−0.66 %** |
  | 10.0 | 8.0 | 5.06 | 4.3802 | +2.10 % | 4.1987 | −2.13 % |
  | 10.0 | 10.0 | 6.32 | 4.2720 | −0.42 % | 4.1971 | −2.17 % |
  | 10.0 | 10.0 | 4.47 (A/D = 0.5) | 6.0824 | +6.71 % | 5.6689 | **−0.54 %** |

  Worst-case error against the spectral reference drops from **10.3 % to 2.2 %**, and the coarse
  rungs improve by an order of magnitude. What the old ladder read as "convergence with δ/h"
  (+10.2 → −0.4 %) was substantially the D1 error decaying with resolution; with it removed the
  ladder is flat at −0.7…−2.2 %, i.e. there is now a small resolution-INDEPENDENT deficit where
  there used to be a large resolution-dependent excess. (Measured with a standalone copy of the
  page's own `blackburn` / `fre` cells; the page itself was not edited.)

### Gate 6 — static-bed `sum A`
Not re-measurable: `force_gate.py FORCE_ADVECT=1` returns NaN at N = 32/48/64/128 with
`CutcellMG::solvePCG: preconditioner produced non-finite z`. **This is PRE-EXISTING and unrelated
to A0** — the pre-A0 module `build_wop_cuda` (built 12:44 the same day) NaNs identically, the
ablation `PECLET_FLOW_ADV_WALLVEL=0` reproduces it bit for bit, and A0 is structurally inert on a
static scene (`hasMotion_` is false, so `advVelView` returns `CCConst(C[c].u)`). `FORCE_ADVECT=0`
passes at 2.2e-15 / 3.3e-15. **Someone should bisect the advective leg of `force_gate.py` against
the WO-M/WO-O/WO-P landings of 2026-08-31.**

### Gate 7 — distributed
The 60 MPI ctests are static/advection-off, and `mpi_scene_gate.py` sets `set_advection(False)`, so
**there is no existing harness for a moving scene WITH advection under MPI**. A0 needs none by
construction — `uBc_` is built pointwise from the analytic scene over the extended block (ghosts
included) by the same code the rung-2 momentum operator already trusts under MPI — but that is an
argument, not a measurement. Building such a gate is worth doing before A1.

## Where this leaves the diagnosis

H1 said "D1 dominates every moving-body symptom". The measurements split it:

- **D1 confirmed as the momentum-conservation defect and as the finite-Re moving-body drag error.**
  The E4 budget leak and the 10 % Blackburn excess are both D1, and both are gone.
- **D1 is NOT the cause of the ten Cate failures.** E1 (Re 1.5) moved +2.2 % where 21 % was needed
  and stays resolution-flat; E3/E4 still blow through u_inf. At Re 1.5 the advective term is small
  by construction, so this was always the weakest leg of H1 — the tow-drag and free-fall numbers now
  make it explicit.
- **A new fact to explain:** at E4 the towed sphere and the free-falling sphere no longer agree
  (drag 1.087 W at 0.955 u* vs a fall peaking at 1.77 u_inf). Either the coupled loop or the
  transient is wrong at high Re, independently of the wall flux.
- **A regression to explain:** the unconfined periodic control moved 1.0322 → 1.0843. A0 reduces
  drag by ~5 % on a body that is nowhere near a confining wall, which is a bulk effect of the
  extension velocity, not a wall-flux effect.

**Do not start A1 on this evidence.** A1 (aperture-weighted cut-face fluxes) addresses D2, which
these measurements do not implicate in any of the remaining symptoms. The next work is diagnostic:
the tow ↔ free-fall split at E3/E4, the unconfined control's 5 % shift, and the −2 % Blackburn
floor.
