# Advective cut-wall flux — campaign plan (scoped 2026-08-31; ten Cate CLOSED 2026-09-02)

**RESOLUTION (2026-09-02): the remaining ten Cate deficit was not a solver defect at all.** The
tank was built from a slab wider than the periodic box; the scene evaluates the UNION of an
instance's periodic images, so the slab's images refilled the cavity and the tank ran **30 %
narrow** (38 cells instead of 53 at d/h = 8, d/W = 0.21 instead of 0.15). With the slab fixed:
E1 peak u/u∞ = **0.922 vs 0.947** (−2.6 %), E4 = **0.972 vs 0.955** (+1.8 %), both at d/h = 8 and
both physical. The static duct twin gives K(Re 1.5) = 1.09 (creeping 1.11 relative to the
periodic box; duct Cd/Abraham 1.30 → 1.08) — the solver screens confinement as the physics
requires. A0 stands as a real fix (Blackburn 10.3 → 2.2 %, Newton leak closed). The three
"live threads" below are closed by the same finding: (1) tow/free-fall disagreement at E4 was the
narrow tank's return flow; (2) the unconfined control's +5 % shift is the d/h = 8 resolution
class (the fixed sphere sits at 0.95–0.99 of Abraham there); (3) the −2 % Blackburn floor is the
same resolution class. A1 (aperture weighting) remains NOT indicated. See
`suite/docs/ANALYTIC_SDF_GEOMETRY.md` §7 item 11 for the trap, the detector now in
`set_solid_from_scene`, and the finite-Re Galilean gate (towed vs fixed sphere agree to 0.03 %).


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

### Gate 6 — static-bed `sum A`  (RESOLVED 2026-09-02: the gate was mis-parameterised, not broken)
Was: not re-measurable — `force_gate.py FORCE_ADVECT=1` returned NaN at N = 32/48/64/128 with
`CutcellMG::solvePCG: preconditioner produced non-finite z`. **PRE-EXISTING and unrelated to A0**
— the pre-A0 module `build_wop_cuda` (built 12:44 the same day) NaNs identically, the ablation
`PECLET_FLOW_ADV_WALLVEL=0` reproduces it bit for bit, and A0 is structurally inert on a static
scene (`hasMotion_` is false, so `advVelView` returns `CCConst(C[c].u)`). `FORCE_ADVECT=0` passes
at 2.2e-15 / 3.3e-15.

**The bisect this section asked for was run, and it is NEGATIVE.** It is not the WO-M/WO-O/WO-P
landings of 2026-08-31 and it is not a regression at all. `0706196` (2026-08-29, *scene layer
retrofitted onto core SceneQueryDevice + native periodicity*) is the OLDEST commit whose bindings
accept the probe's `set_scene(..., periodic=True)` call, and it goes non-finite at N=24 on **the
very same step 78** as current main — as does `f61dfaa`, which predates every WO-M/O/P commit.
Iteration counts are irrelevant (`set_pressure_solver_params(25 -> 400)` and
`set_velocity_solver_params(100 -> 1000)` leave the blow-up on the identical step at N=32), so it
is not a solver-convergence artefact either.

**Mechanism — the semi-implicit stability limit, hit by the probe's own parameters.**
`set_advection()` leaves the advective term EXPLICIT: it is assembled into the RHS lagged at the
Picard iterate (`buildRhs`, `src/flow_ibm.hpp:3600-3620`, `const bool adv = advect_ && !pureFou`;
`implicitAdv()` at `src/flow_ibm.hpp:2138` is false unless `set_implicit_advection` is called, and
`hydro_force_torque_reaction` **refuses** that path at `src/flow_ibm.hpp:2998-3005`). The step is
therefore the textbook semi-implicit one — implicit viscous, explicit advection — whose linear
amplification for a mode `k` is `g = (1/dt - i u.k)/(1/dt + nu k^2)`, i.e. stable iff
`|u.k|^2 <= 2 nu k^2/dt + (nu k^2)^2`; at large `dt` that is just the CELL REYNOLDS NUMBER
`Re_h = rho |u| h / mu <~ pi`.

The probe held `f = 1e-3`, `mu = 0.1`, `dt = 60` fixed while sweeping N — but this lattice's
permeability in *cell* units grows like `N^2`, so `|u|` and `Re_h` grow like `N^2` with it. Measured
Stokes `max|u|` (~ `0.079 (N/16)^2` to within 4 %): 0.0812 / 0.178 / 0.315 / 0.706 / 1.254 at
N = 16/24/32/48/64, i.e. `Re_h` = **0.81 / 1.78 / 3.15 / 7.06 / 12.54**. N=16 sits a factor 4 below
the limit (and is stable at every `dt` tried, up to 240); every N >= 24 is at or past it. The
divergence is a bulk-pore mode, not a cut-cell one — at N=24 the fastest-growing cell sits at
signed distance +5 h from the nearest grain — and the criterion is quantitatively confirmed by the
`dt` scan: N=32 (`Re_h = 3.15 ~ pi`) is unstable at `dt` = 8/15/60 and stable at `dt` <= 1, exactly
the small-`dt` branch `dt <= 2 nu k^2 / |u.k|^2 ~ 2` predicts; N=16 (`Re_h = 0.81 < pi`) is stable
at every `dt`. Raising the Picard count makes it worse, not better (`set_outer_iterations(10)` at
N=32/dt=15 moves the blow-up from step 67 to step 18) — the signature of a diverging lagged-
advection fixed point. `set_implicit_advection(True)` removes the instability outright at
N=24/32/64, but the reaction budget refuses that path, so it is not available to this gate.

**No solver change is indicated** — the limit is a property of the scheme. The fix is in the probe:
`force_gate.py` now scales the driving force as `f = FORCE_F * (16/N)^2` **on the advective leg
only** (`FORCE_F_SCALE=0` restores the old, unstable f; `FORCE_ADVECT=0` is byte-unchanged, `f`
stays `FORCE_F`). That pins `Re_h` at the N=16 value for every rung and lets `Re_d` grow like N, so
the default ladder now lands near the plan's "Re ~ 30" target instead of diverging. The log line
carries `f` and `Re_h` so the margin is visible.

Gate 6 result with that scaling, `FORCE_ADVECT=1`, RTX 5080 / nvidia-cuda, flow `41997eb`:

| N | f | Re_h | Re_d | reaction identity `ratio-1` |
|---|---|---|---|---|
| 16 | 1.0000e-03 | 0.80 | 3.01 | **2.220e-16** |
| 24 | 4.4444e-04 | 0.78 | 4.29 | **1.332e-15** |
| 32 | 2.5000e-04 | 0.77 | 5.58 | **8.882e-16** |
| 64 | 6.2500e-05 | 0.74 | 10.31 | **6.661e-15** |
| 128 | 1.5625e-05 | 0.69 | 18.61 | **9.770e-15** |

`GATE PASS [reaction identity to solver residual at every N]` — the same 1e-15 class as the
`FORCE_ADVECT=0` leg, so the R0 advective term IS in the budget and A0's `sum A` is now measurable
on a static bed. (`tests/regression/sdflow_regression.py --build build_nan`: **PASS**, +0.00 % on
every metric — no flow source was touched, only this doc.)

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

## Solver-campaign cross-check (2026-09-02) — the deficit is orthogonal to all of it

Between A0 (`fb1a1a7`) and 2026-09-02 the suite landed the VoF V5a/V5b rungs, P1/P2 exact
residuals (opt-in), CutcellMG telescoping (opt-in), and — on the `telescope` branch, not yet in
main — the momentum residual stop (default 1e-5) with velocity MG under MPI. Every ten-cate probe
was re-run on three fresh builds against the A0 baselines (same machine, same probe scripts,
`.sdf-campaign-probes/cutwall_*.py`):

| probe (d/h=8) | A0 | main `a89417c` | main + `MREAL_DOUBLE` | telescope `496ec13` (res-stop 1e-5) |
|---|---|---|---|---|
| E1 free-fall peak u/u∞ | 0.8030 | 0.8030 | 0.8030 | 0.8030 |
| Newton audit, towed E4, ΣF/W | −0.0329 | −0.0329 | −0.0329 | −0.0329 |
| tow drag E4 / E1, F/W | 1.0866 / 1.2046 | 1.0866 / 1.2046 | 1.0866 / 1.2046 | 1.0866 / 1.2047 |
| unconfined control | 1.0843 | 1.0843 | 1.0843 | 1.0844 |

Readings: (1) nothing merged into main since A0 touches moving geometry (bit-identical); (2)
**double pressure-operator storage changes nothing** — the float `MReal` floor (SCALING_ISSUES
#1) is not the confined-drag cause (the momentum IBM overlay `K_val/M_val/…` is hard-float
regardless of `MReal`, so that half is still untested — a separate build flag would be needed);
(3) the residual-stop default is physically inert here (4th digit).

**One real impact of the telescope default, on a different gate:** `rotation_gate.py` (Stokes,
100 fixed sweeps on main) FAILS on telescope — torque within 0.3% (1.02118 vs 1.02424 at N=96) but
the spinning sphere's net force rises from round-off (5e-13) to **1e-6…1e-5**. Telescope with
`set_velocity_residual_tolerance(0)` reproduces main **bit-for-bit**; with 1e-10 the torque
returns to 5 digits and the net force to 1e-9…1e-8. So the gate's "net force at round-off"
criterion is satisfiable only by the fixed-count loop; the physics is unaffected. The gate needs an
explicit tolerance policy before the telescope default reaches main. Also: the telescope branch
does not build without `PECLET_FLOW_MPI=ON` (`MPI_Comm`/`comm_` undefined in
`mac_velocity_mg.hpp:672`, `flow_ibm.hpp:4075`).

**Consequence for the three live threads:** precision and momentum convergence are now excluded
as causes along with dt, sweeps, limiter and advection scheme. What remains is structural in the
moving cut-cell discretisation itself — D2 (apertures/wall closure at the moving wall), the
continuity/pressure treatment of fresh and dying cells, or the coupling time-lag at high Re
(thread 1). Builds kept for the next session: `flow/build_l3_cuda_head`, `flow/build_l3_cuda_dbl`,
`tel/flow/build_l3_cuda_tel`.

---

## Gate 7 (distributed) — MEASURED 2026-09-02, and it FAILS: the A0 fill is not decomposition-independent

Gate 7 above asked for "np=1 bit-exact, np=2/4 within the established 3e-7/5e-12-class tolerances
of the existing MPI gates". Until now nothing measured it: **every ctest in `tests/kokkos_mpi` runs
`setAdvection(false)` and none of them moves a scene instance**, so the A0 rung's MPI claim —
"the scene is analytic, so ghost solid rows are computable pointwise and no extra exchange is
needed" — was an argument, never a measurement. `tests/kokkos_mpi/test_movingscene_advect_mpi.cpp`
(ctest `movingscene_advect_mpi_np{1,2,4}`) is the measurement.

**The case.** Periodic 48^3 box, one analytic sphere d = 8 (off-lattice centre, +0.3), towed
diagonally at U = 0.05 cells/time (Re = U d / nu = 20, nu = 0.02, dt = 4 => CFL 0.2), explicit SOU
advection ON, `set_instance_transform` + `set_instance_motion` + `rebuild_geometry()` every step,
60 steps. The path crosses the ORB cut planes (x = 32 at np = 2; x = 32 and y = 32 at np = 4).
Compared against the full-grid single-rank reference built in the same executable: u, v, w and P
cell by cell, plus the per-instance `hydroForceTorqueReaction()`.

| np | max abs du (max abs u = 4.0334e-02) | max abs dP (max abs P = 2.1442e-03) | dF (abs F = 1.1367e-01) | dT | verdict |
|---|---|---|---|---|---|
| 1 | **0.000e+00** | **0.000e+00** | 1.332e-15 | 2.479e-15 | PASS (fields bit-exact) |
| 2 | 1.452e-07 | 1.854e-08 | 9.999e-08 | 1.450e-07 | **FAIL** (tol 1.21e-08) |
| 4 | 1.136e-05 | 2.839e-07 | 9.249e-06 | 2.476e-05 | **FAIL** (tol 1.21e-08) |

The reaction force is held to 1e-12 relative at np = 1 rather than to zero, and not because of
MPI: `hydroForceTorqueReaction` accumulates with `Kokkos::atomic_add` over an unordered device
traversal, so it is tolerance-reproducible, not bitwise, even between two runs whose fields agree
bit for bit (the same caveat `mpi_scene_gate.py` states).

**The A0 fill is the cause, isolated by a 2x2 ablation at np = 2** (max abs du, same case):

| configuration | max abs du |
|---|---|
| moving + advection ON (the shipped case) | **1.45e-07** |
| moving + advection OFF (`GATE7_ADV=0`) | 5.99e-16 |
| static + advection ON (`GATE7_MOVE=0`, body-force driven) | 3.47e-17 |
| moving + advection ON, A0 fill disabled (`PECLET_FLOW_ADV_WALLVEL=0`) | 1.28e-16 |

Neither the moving-geometry machinery nor the advection is decomposition-dependent on its own, and
with the A0 fill off the moving + advective march is bit-clean across ranks. Note what the third
row also says: this configuration carries **no measurable MG-PCG reduction-order floor at all**, so
the 3e-7 tolerance is generous and the failure is unambiguous.

**It is LOCAL to the rank boundary.** The identical case translated so the body never comes within
the ghost ring of a cut plane (`GATE7_SHIFT=-8`) reads **2.57e-15** at np = 2.

**Likely mechanism** (not yet fixed — this session's brief was to measure, not to repair).
`buildWallVelocity` fills `uBc_` over the extended block from a CENTRED difference of the sampled
SDF, `ccSampleExt(sd, e, sx +/- 1, ...)`. `ccSampleExt` **clamps** its indices to `[0, ext-1]`, so
at the OUTERMOST ghost plane there is no neighbour to difference against and the gradient — hence
`wallPoint`, hence `uBc_` — is wrong there; and `uBc_` is never halo-exchanged. Under MPI those
planes sit at interior rank boundaries, at different global points than the single-rank run's, and
`buildAdvInputs` writes them into the advection's scratch inputs where the SOU stencil (reach 2)
carries them inward. The plan's claim is right about the SCENE and wrong about this fill, which
reads the sampled SDF, not the scene. Two candidate repairs, in preference order: (1) evaluate the
wall velocity from the scene analytically (`q.owner(p)` + `instanceVelocity` at the point, no SDF
gradient) at the outermost planes, or (2) halo-exchange `uBc_` (three extra field exchanges per
geometry/motion update — it is rebuilt once per step on this path, not per Picard iteration).

**Magnitude depends on where the body is when the fields are compared.** An earlier variant of this
case whose body ended with its wall band sitting IN rank 0's outer ghost planes (centre stopping at
x = 27.54 against a cut at x = 32) measured **1.385e-03 at np = 2 and 1.703e-03 at np = 4** — 3.5 %
of max abs u, with the reaction force off by 0.69 % (np = 2) and 0.44 % (np = 4). A body parked on
a rank boundary is the worst case, and it is an entirely ordinary situation in a real run.

**Reproduce** (nvidia-cuda prefix, `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`):
```bash
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH=$PWD/../extern/install/nvidia-cuda \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_kmpi --target test_movingscene_advect_mpi -j
mpirun -np 2 ./build_kmpi/test_movingscene_advect_mpi          # and -np 1, -np 4
PECLET_FLOW_ADV_WALLVEL=0 mpirun -np 2 ./build_kmpi/test_movingscene_advect_mpi   # ablation
```
Runtimes on one RTX 5080 (each run also builds the full-grid reference on rank 0): np = 1 95 s,
np = 2 212 s, np = 4 332 s.

**Also fixed on the way in:** `tests/kokkos_mpi/CMakeLists.txt` did not configure at all. Commit
`86192ad` (V-BC, 2026-09-02) appended a duplicate tail to the gated `foreach` list, leaving an
orphan line after the closing paren — a CMake parse error, and `vof_bc_mpi` was never registered.
The list is now one list and carries `vof_bc_mpi`; the tree configures 78 tests.

## Gate 7 CLOSED (2026-09-02) — the ghost-plane MASK, not the wall velocity

The agent's diagnosis (uBc_ from the clamping sampler on the outermost ghost plane) was half
right: `ibmSolidMask` samples the sdf at the staggered offset through the SAME clamping
`ccSampleExt`, so the ghost-plane MASK can disagree with the neighbour's interior mask, and it is
the mask that decides which ghost rows of the advection scratch receive the wall velocity.
Ablations at np=2 (`test_movingscene_advect_mpi`, body towed across the ORB cut), max|du|:

| change | np=2 max\|du\| |
|---|---|
| shipped A0 | 1.452e-07 |
| + uBc_/uwCell_ ghost exchange (`exchangeExtRaw`) | 1.452e-07 (unchanged) |
| fill inner rows only / ghost rows only | 1.44e-03 / 1.38e-03 (both must be filled) |
| A0 off | 2.2e-15 |
| **+ mask ghost exchange under motion** | **4.1e-16** |

Fix: `if (hasMotion_) exchangeExtRaw(C[c].mask)` after `ibmSolidMask` (static scenes never
consume ghost masks — byte-identical by construction), plus the uBc_/uwCell_ exchange (correct on
its own terms: those planes were wrong too, they just were not what the SOU read first). np=1
stays bit-exact. The `PECLET_FLOW_UBC_EXCHANGE` / `PECLET_FLOW_ADV_FILL_MODE` ablation knobs are
left in as documented instrumentation.
