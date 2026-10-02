# Anderson acceleration of steady marches — design note

*2026-10-02, architect pass on `doc/steady_acceleration_brief.md`. Status: **proposed**, not
implemented. The decisions in §1 go to `../docs/decisions/flow.md` once approved. An implementer
should not need to choose anything this note leaves open. Where a choice depends on a measurement,
§10 states the default to use and the experiment that would change it.*

***Revision 1** (2026-10-02, architect pass on `doc/steady_acceleration_brief2.md`, after WO-1
falsified three premises of revision 0). The section below lists every change and its oracle
evidence; the body of the note is the revision-1 design. Superseded readings are kept as one-line
"superseded by rev 1" entries. Numbers and commands: `doc/steady_acceleration_log.md`, entry
"Revision 1".*

---

## Revision 1 — what changed and why

WO-1's oracle (an exact NumPy implementation of revision 0) found three defects: (1) the Ritz guard
declares "unstable" on plain certification steps (5 of 12 §11 production runs); (2) certification
rarely passes inside its 6-block budget (9 of 24 runs); (3) on the dense random bed phase A cannot
reach its target, so the bed gained only 1.07–1.26× in steps. Revision 1 changes five things, each
proved on the oracle before it was written here (`tests/study/anderson_oracle.py`, whose default is
now `--rev 1`; `--rev 0` reproduces WO-1 bit for bit — collocated N = 16, m = 5: 112 steps).

**What the slow bed mode is (measured, §1.2).** Pressure in fluid cells at sphere near-contacts. On
the A1 bed (φ 0.6, N = 64, staggered) 1,168 fluid cells sit in 1,085 pockets sealed by zero face
apertures (almost all single cells), beside the main pore space of 103,767 cells. In the plain
march, from step 300 to 1500, 99.97–99.996 % of the W-residual energy is pressure, and 90 % of that
pressure energy sits in 41–105 cells. Sealed pockets carry 58–64 % of it, and cells behind faces of
aperture ≤ 0.2 carry another 10–25 %. A sealed cell's pressure is a gauge mode with no coupling to
any velocity. Behind a small aperture the pressure equilibrates through the tiny face flux (a
lubrication-film mode, rate 1 − O(aperture)). Together they give the continuum of Ritz values
0.9998–0.99999 and the power-law tail of the W-residual. Neither moves ⟨u_x⟩: the plain march
certifies at step 325 with its W-residual at 7.1e-5 and its velocity residual at 1.1e-6. The
collocated bed shows the same structure.

**R1 — the metric is velocity only (D3 superseded).** The pressure P, and the collocated face field,
stay in the state. They are mixed, stored and differenced, but not measured (role Carried). The
residual becomes the relative velocity residual ‖g_u − x_u‖/‖g_u‖.
- *Why:* a pressure error that matters shows up in the velocity it drives at the next step;
  measured that way it is weighted by its effect. The c_P weight measured it by its size, and on a
  dense bed that size is dominated by near-contact pockets that do not move ⟨u_x⟩.
- *Evidence (unconditional phase A, m = 5, steps to the monitor staying within 1e-5 / to a velocity
  residual of 3e-7):*

  | case | velocity metric | W metric (rev 0) |
  |---|---|---|
  | §11 coll N16 | 36 / 39 | 43 / 48 |
  | §11 stag N16 | 15 / 21 | 14 / 22 |
  | §11 coll N24 | 14 / 23 | 15 / 24 |
  | §11 stag N24 | 23 / 25 | 24 / 26 |
  | Re ≈ 10 | 102 / 106 | 85 / 89 |
  | bed stag | 76 / 68 | 52 / 104 |
  | bed coll | 36 / 58 | 33 / 74 |

- *Reading:* the velocity metric captures the collocated checkerboard as well as the c_P weight
  did, so revision 0's reason for the weight does not hold. It reaches the handover 35 % / 22 %
  sooner on the two beds, the use case that decides D11. Re ≈ 10 is the one case where W is faster
  (89 vs 106). It also removes c_P, the fluid mask, the gauge pass and one collective per step.

**R2 — phase A hands over on the velocity residual.** The number is unchanged:
(1 − slow_rate)·rtol = 3e-7. Revision 0's premise ("the residual at which the slowest mode can
carry at most rtol of remaining change") holds for the velocity residual. At the handover the
monitor error is 2e-8 to 1.4e-5 (≤ 0.14·rtol) on all seven cases. It failed for the W-residual
because the bed's slow pressure modes carry no monitor change.

**R3 — the Ritz guard runs only where its estimate means something (D5 amended).** It runs only on
an accelerated (mixed) call, only on a window whose every column was formed at a mixed call, and
only while the residual is at least the Ritz floor max(1e-10, 1000·τ). Here τ is the inner-solve
tolerance: the momentum residual tolerance in force, production 1e-8, so the floor is 1e-5. The
consecutive count restarts at every call that is not eligible.
- *Why:* on plain calls the window holds near-collinear plain-march differences (scaled
  cond(XX) 1.6e-11 – 4e-11), so M = XX⁺XR amplifies inner-solve noise; radii of 12.6–56 were
  measured. The same happens inside phase A near the map's own noise floor: the production velocity
  residual stalls at ≈ 2e-8 = 2τ. In 42 unconditional phase-A runs, 15 had readings > 1.001. All
  of them lie at residual ≤ 5.5e-8, apart from one isolated 1.006 at 1.4e-6 = 140τ.
- *Evidence (revision-1 matrix):* 0 false "unstable". Every eligible reading is ≤ 0.9974, except
  the true positive on G7a at m = 8: 1.124 at residual 9.4e-3, three in a row, so "unstable" at
  step 18 on a map whose plain march diverges.

**R4 — certification runs up to 2·(num_passes + 3) = 12 blocks, with an early "slow" exit (D6, §7
amended).**
- *Why:* Anderson's least squares makes the next residual small by cancelling slow-mode and
  medium-mode contributions against each other. So after phase A the monitor's block changes
  start near zero, then change sign or grow (R > 1) for up to five blocks before the slow tail
  emerges. Measured |d| at those blocks is 1e-8 – 1e-10 of |m|, four orders below rtol: the state
  is converged, but the unchanged instrument passes only on a geometric tail.
- *Revision 0's cost:* its 6-block budget resumed phase A, which re-created the same transient.
  An ablation kept every other revision-1 change and only the 6-block budget. Staggered N = 24,
  m = 8 then took 214 steps against 135 plain, with 5 resumes; with 12 blocks it took 74 steps.
- *The early exit:* a block that fails on the remainder bound with R ∈ [slow_rate^check_every, 1)
  resumes acceleration at once. Such a tail is clean, as slow as the instrument assumes, and too
  large. The exit keeps revision 0's reason for having a budget. It never fired in the
  revision-1 matrix; it is insurance against a premature handover.

**R5 — the window stays 5.** The pre-registered rule (m = 3 within 10 % of m = 5 on every case)
was re-applied to the revision-1 numbers and is not met: collocated N = 24 69 vs 48; bed
collocated 163 vs 83.

**Revised step table** (steps to `converged=True`, production settings; plain / m = 3 / m = 5 /
m = 8; ratio and pressure-iteration ratio at m = 5). Revision 0's m = 5 is in brackets.

| case | plain | m = 3 | m = 5 | m = 8 | ratio m = 5 (rev 0) | pressure-iteration ratio m = 5 |
|---|---|---|---|---|---|---|
| §11 collocated N16 | 395 | 65 | 89 | 73 | **4.4×** (3.5×) | 4.4× |
| §11 collocated N24 | 90 | 69 | 48 | 58 | 1.9× (0.89×) | 2.0× |
| §11 staggered N16 | 75 | 62 | 51 | 51 | 1.47× (0.97×) | 1.49× |
| §11 staggered N24 | 135 | 53 | 50 | 74 | 2.7× (2.3×) | 2.6× |
| Re ≈ 10 (staggered) | 345 | 161 | 131 | 103 | 2.6× (3.0×) | 2.65× |
| **dense bed φ 0.6, staggered** | 325 | 111 | **93** | 90 | **3.5×** (1.26×) | 3.5× |
| **dense bed φ 0.6, collocated** | 190 | 163 | **83** | 82 | **2.3×** (1.07×) | 2.4× |
| G7a (plain diverges) | div. | div. | div. | unstable @18 | `converged=False` everywhere | — |
| G1 tight, collocated N16 | 3520 | 149 | 141 | 118 | \|K_acc/K_plain − 1\| ≤ 5.1e-10 | — |
| G1 tight, staggered N16 | 295 | 85 | 79 | 72 | ≤ 1.9e-11 | — |

Every bar of the brief holds: steps_acc ≤ steps_plain on every case and window; no false "unstable";
4.4× on collocated N = 16; G1 ≤ 1e-8. **The dense bed clears 1.5× in steps on both schemes (3.5×,
2.3×).** The pressure-solve iterations per step are the same at mixed and plain iterates
(staggered bed 11.5 vs 11.5, collocated 31.2 vs 33.3), so the iteration ratio tracks the step ratio.
Wall time is measured at G2 (D11).

**Core delta (WO-3b, §9).** Part 1 is required. `AndersonState` gains `innerTolerance`. The Ritz
floor gets a constant `kRitzFloorFactor = 1000`. Each window slot gets a "formed at a mixed call"
flag, and step 7 gets the eligibility rule of R3. Part 2 is a separate commit (Q14, default yes):
delete the Pressure role, `sdf`, `cP`, `gauged` and pass 1, which no consumer uses after R1.

**Not changed:** the state (D2); type-II Anderson, its least squares and the column-drop rule (D4);
the restart rule and its 1e-10 floor; the instrument and its constants (Q11); D7–D13 apart from
dropping one collective; the memory formula (§6.3).

---

## 1. Decision summary (register style)

Each entry gives the decision, the alternative rejected and the reason. Evidence is in §1.1
(revision 0) and §1.2 plus the "Revision 1" section (revision 1).

**D1. Placement: the data path is C++/Kokkos (the grid-agnostic `AndersonCore` in `core`, the
adapter in flow); the control path is a small pure-Python driver.** The data path covers the state,
the history, the reductions and the safeguards, in `peclet::core::solver::AndersonCore` (see D13) +
flow's `AndersonAccelerator<Grid>`. The control path covers the phases and the §3.2 stop
instrument, in `peclet.flow.march_to_steady`.
- *Rejected:* a pure-Python driver over `diagnostics.field_view`. Its device path would be
  CuPy-only, so it would not run on the HIP or OpenMP backends that the Kokkos source supports. It
  cannot see the collocated face field `uf_` (that field is not in the registry). It cannot enforce
  refusals from the solver's own configuration flags. Every layout fact would be hard-coded twice.
- *Rejected:* a C++ march loop that includes the stop instrument. The instrument would then exist
  twice (C++ and the study scripts), and per-step Python hooks (logging, checkpoints) would be
  lost.
- *Reason:* everything that touches fields stays on the device and runs on every backend and under
  MPI. Everything that is policy stays readable and user-adjustable.

**D2. The state vector is exactly what `step()` reads across steps.** It is the full padded buffers
of the velocity (`C[0..2].u`) and the accumulated pressure `P_`. For `SolverColocated` with
projected-face advection on, it also includes the face field `uf_/vf_/wf_`.
- *Rejected:* a velocity-only state. Measured, it stalls at 1.9e-4 error, because the slow
  checkerboard lives in P (§1.1).
- *Rejected:* adding derived fields (u*, φ). They are recomputed every step, so they are not state.
- *Reason:* Anderson accelerates a map x ↦ g(x). If g reads anything outside x, the map Anderson
  sees depends on history, and its secant model is wrong.

**D3 (rev 1). The metric is the velocity at unit weight. The pressure and the face field are
carried: mixed, stored and differenced, never measured.** The residual is the relative velocity
residual (§3.2).
- *Superseded by rev 1:* "velocity at unit weight plus c_P² × the fluid-centred, gauge-centred
  pressure, c_P = h/(μ + ρh²/Δt)" (revision 0).
- *Rejected:* the c_P-weighted pressure term (revision 0). On the dense bed 99.97 % of its residual
  energy is pressure in near-contact cells that does not move the answer (§1.2). It made the
  velocity-residual handover 53 % later on the staggered bed and 28 % later on the collocated bed
  (68 vs 104 steps, 58 vs 74). The checkerboard argument made for it in revision 0 does not hold:
  the velocity metric reaches the collocated N = 16 handover sooner, 39 steps against 48.
- *Rejected:* gauge removal per fluid component, i.e. keeping c_P and removing each sealed pocket's
  constant. It removes only the sealed pockets (58–64 % of the late pressure energy), not the cells
  behind small apertures, and it needs a distributed connected-component labelling.
- *Rejected:* the unweighted Euclidean norm of (u, P), which depends on the unit system; and a metric
  on the face gradient of P, which needs ghost-consistent history and depends on the partition.
- *Reason:* the velocity residual measures a pressure error by the velocity it drives at the next
  step, that is, by its effect on the answer. A pressure error that drives no velocity (a sealed
  pocket's constant) cannot change a velocity monitor. Every entry is a velocity, so the metric is
  unit-free, with no c_P, no fluid mask and no gauge pass.

**D4. Algorithm: type-II Anderson.** Window m = 5 (compile-time cap 8) and mixing β = 1. The least
squares is solved by normal equations through a Jacobi-scaled eigendecomposition. The oldest column
is dropped while the scaled Gram condition exceeds 1e12, i.e. κ(ΔR) > 1e6.
- *Rejected:* QR or TSQR. It costs m sequential reductions or a TSQR tree, and at κ ≤ 1e6 it buys
  nothing because γ only steers the path, not the fixed point.
- *Rejected:* m = 2, which was measured insufficient on the collocated case (§1.1).
- *Rev 1:* WO-1's pre-registered window rule (m = 3 within 10 % of m = 5 in steps on every case)
  was re-applied to the revision-1 numbers and is not met (collocated N = 24 69 vs 48; collocated
  bed 163 vs 83). m stays 5.
- *Rejected:* β < 1 as the default. It slows the fast modes, and the plain step is already the
  scheme's own stable preconditioner.

**D5. Safeguards.**
- Mixing engages after two consecutive residual decreases.
- History restarts when the residual exceeds 4× the minimum since the last restart (applied only
  above the noise floor 1e-10; unchanged in rev 1, 0 restarts in every measured run).
- Acceleration is disabled after 5 restarts.
- A non-finite residual or a failed pressure solve at a mixed iterate restores the last plain-map
  output and disables acceleration.
- **(rev 1)** An instability guard watches the spectral radius of the window's Ritz matrix.
  - It is evaluated only on an **eligible** call. The call's input must be a mixed iterate, every
    window column must have been formed at a mixed call, and the residual must lie in
    [kRitzFloor, 1e-2].
  - kRitzFloor = max(1e-10, 1000·τ), where τ is the inner-solve tolerance (§4.1). That is 1e-5 at
    production settings and 1e-9 at tight ones.
  - If the radius exceeds 1 + 1e-3 on 3 consecutive calls, all of them eligible, the status
    becomes "unstable" and the march returns not-converged. An ineligible call resets the count.
- *Superseded by rev 1:* "evaluated on every active call while the residual is in
  [1e-10, 1e-2]".
- *Rejected (rev 1):* evaluating on plain calls. Their window holds near-collinear plain-march
  differences (scaled cond(XX) 1.6e-11 – 4e-11), and the estimate there is inner-solve noise: radii
  of 12.6–56 gave a false "unstable" in 5 of 12 WO-1 production runs. Growth on plain steps is
  already caught by the certification's growth exit and by the instrument itself.
- *Rejected (rev 1):* a stricter pseudo-inverse truncation in place of the eligibility rule.
  Truncation is a threshold tuned between 1e-11 and 1e-5 on this data, and it does not cover the
  noise floor. Readings > 1.001 occur in phase A too, at residual ≤ 5.5e-8 ≈ 5τ.
- *Rejected:* a tight bound on the mixing coefficients Σ|α|. Extrapolating the 0.996/step
  checkerboard needs Σ|α| ≈ 2/(1−λ) ≈ 500; a tight bound forbids the acceleration being sought.
- *Rejected:* no instability guard. Measured: Anderson keeps marching where the plain march at the
  same Δt diverges. Anderson, like GMRES, can converge to fixed points that the plain march cannot
  reach (§1.1, §2.4).

**D6 (rev 1). Stop interplay: two phases.**
- **Phase A** accelerates until the relative *velocity* residual reaches (1 − slow_rate)·rtol.
- **Phase B** certifies with the **unchanged §3.2 instrument on consecutive plain steps**, within a
  budget of 2·(num_passes + 3) blocks.
- **Early "slow" exit.** Phase B ends at once when a block fails on the remainder bound with a
  ratio R ∈ [slow_rate^check_every, 1).
- **Budget exhausted or slow exit:** tighten the target ×0.1 and resume acceleration with the
  history intact.
- **Growth:** if the plain steps grow, disable acceleration and finish as the plain march would.
- *Superseded by rev 1:* "the relative W-residual" and "a budget of num_passes + 3 blocks" with no
  early exit.
- *Rejected (rev 1):* keeping the 6-block budget. After an Anderson iterate the monitor's first
  block changes start near zero, then change sign or grow (R > 1) for up to five blocks before the
  slow tail emerges, even though |d| is 1e-8 – 1e-10 of |m|. Resuming acceleration then re-creates
  the same transient. In the budget ablation (all other rev-1 changes in place), staggered N = 24,
  m = 8 took 214 steps against 135 plain at 6 blocks, and 74 at 12 blocks.
- *Rejected (rev 1):* no budget at all. A handover that leaves a clean but slow tail (R close to 1,
  remainder above rtol) would then decay at the plain rate instead of being re-accelerated. The
  early exit handles exactly that case and nothing else.
- *Rejected (rev 1):* handing over on the monitor's own changes along the accelerated sequence.
  - *Gain:* with the velocity metric the monitor stays within 1e-5 from step 66–76 (staggered bed)
    and 32–36 (collocated bed), against the velocity target at 65–86 and 57–74. That is nothing on
    the bed that decides D11, and about 20 steps on the other.
  - *Cost:* Anderson's monitor sequence is not monotone (the W-metric runs jitter by ±3e-6 for 200
    steps). A stagnating sequence would trigger false handovers, each costing ≥ 25 plain steps.
    And phase A would depend on the caller's monitor.
- *Rejected:* applying the §3.2 rule to the accelerated sequence. Anderson iterates are not plain
  steps. A stagnating Anderson sequence has small ⟨u_x⟩ changes far from the fixed point, which
  means false stops.
- *Rejected:* a residual-norm-only stop. Its error bound would need a non-normal bound on
  (I − J)⁻¹ that nobody has verified, and it abandons the instrument the user chose.

**D7. MPI: bit-exact rank-count independence is not required.**
- np = 1 must be bit-identical to the serial build. np > 1 must agree to tolerance.
- Each step does one `MPI_Allreduce` of the sum packet (≤ 5m+2 doubles), and the inner state
  check adds a 1-double count. (Rev 1 removed revision 0's 3-double gauged-pressure pre-pass with
  the pressure metric.)
- Rank 0 broadcasts the decision packet. **Identical γ on every rank is a correctness requirement:**
  ghosts are mixed locally, so they stay consistent only if all ranks use the same coefficients.
- *Rejected:* reproducible (binned or integer) summation for rank-count bit-exactness. It needs a
  custom exact accumulator, and the existing Krylov path is already only tolerance-exact at np > 1
  (`tests/kokkos_mpi/test_sdflow_mpi.cpp:125`: "np=1 bit-exact; np>1 the MG-PCG reduction-order
  floor").

**D8. Memory: double precision, full padded buffers, 2m+3 state vectors** (history 2m, plus X, R_prev
and G_prev).
- *Rejected:* float history. Halving memory is not needed below about 7.5 M cells per 16 GB card,
  and the suite's precision policy argues against it (open question Q8).
- *Rejected:* storing inner entries only. That loses state held in ghost indices (the high-side
  outflow face) and needs extra ghost-refresh logic.
- *Reason:* a linear combination of padded buffers is exactly the linear combination of the states.

**D9. Checkpoint: the Anderson history is not checkpointed.** A restart rebuilds it in m + 3 steps.
The checkpoint is `get_field`/`set_field` of `u, v, w, p`, not `set_state`, which restores velocity
only and loses P.
- *Rejected:* checkpointing the history. It makes a checkpoint about 14× larger (2m + 3 = 13 extra
  state vectors on top of the one state) and adds nothing to correctness.

**D10. Scope.**
- Supported: staggered `Solver`, and `SolverColocated` with the fluid-only `"ghost"` scheme. Stokes
  or finite Re, periodic or domain BCs.
- Refused, with a named error: gauge-exact, plain and embed collocated schemes; VoF; phase change;
  transported scalars; porous continuity; variable ρ/μ closures; `drag_beta`; cell forces (CFD-DEM);
  moving scene instances; superficial velocity; pressure warm start; the balanced-force projection.
- *Rejected:* allowing gauge-exact. Its fixed points form an attractor family
  (`collocated_invisible_subspace.md` §4), and Anderson may select a different member than the
  march would, with a spread of 5e-4 in k.

**D11. API.**
- Public: `peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, …)`, returning a `MarchResult`.
- Developer tier: `s.diagnostics.anderson_accelerator(window=5, mixing=1.0)`.
- `Solver.step()` is untouched, so the unaccelerated path stays bit-identical.
- The default of `accelerate` (new function only) is set by a **pre-registered rule** (user,
  2026-10-02, resolving Q1): measured at G2 on the dense random bed (the G1 dense-bed case,
  production build, wall time including accelerator overhead), **`True` if the accelerated march is
  ≥ 1.5× faster in wall time, otherwise `False`** (acceleration then opt-in, documented as for
  symmetric/collocated cases). WO-1's step ratio on the dense bed, if host-feasible, is the early
  indication only. The outcome and its numbers go to the log; nothing else may change here.
- *Early indication (rev 1, oracle, m = 5):*
  - staggered bed (decides, Q13): 3.5× in steps, 3.5× in pressure iterations, 2.7–3.2× in host
    map time;
  - collocated bed: 2.3× in steps, 2.4× in pressure iterations, 2.2× in host map time.

  Rev 0 measured 1.26× and 1.07× in steps.
- *Rejected:* an `enable_*` switch that changes what `step()` does, which would change an existing
  entry point.
- *Rejected:* an environment variable (QUALITY_PLAN D3).

**D12. Finite Re: same design, with smaller gains.**
- Anderson stays appropriate while the plain march contracts. With explicit advection, Δt is
  CFL-bound and the slow spectrum is a continuum near 1, not a few outliers, so the gain falls to
  about 2.5× (measured).
- Unsteady or march-unstable regimes are handled by the D5 guard plus the certification-growth
  fallback (§7).


**D13. `AndersonCore` lives in `core` (`peclet::core::solver`) from the start** (user decision,
2026-10-02, resolving Q9). Flow keeps only the adapter `AndersonAccelerator<Grid>`, `marchState()`
and the Python driver.
- *Rejected:* flow-local first, promoted when amr adopts it. A later move is a second port with its
  own bit-identity proof; the core is grid-agnostic already and sits naturally beside core's
  `solver/` layer (BiCGStab, GraphAMG); amr's lmax = 0 march has the same (u, P) structure.
- *Cost:* core must be tagged and published before flow ships WO-4 (suite release order).

### 1.1 Evidence: a throwaway prototype (2026-10-02, measured)

*Superseded as the reference numbers by §1.3, the measured C++ build (WO-5); kept as the record of
what the design was decided on.*

**What was run:**
- A NumPy prototype of §4 without the safeguards, run on host build
  `../flow-vof-b1/build_omp` (OpenMP, 6–8 threads, an existing VoF-branch build; the single-phase
  step is assumed to equal main's). Script: scratchpad `aa_proto.py`, which is ephemeral; WO-1
  rewrites it.
- Case: §11 of `collocated_invisible_subspace.md`. One sphere per periodic unit cell, φ = 0.125,
  ρ = μ = F = 1, νΔt/h² = 6, N = 16, pressure PCG rtol 1e-10, 200 velocity sweeps.
- State: (u, v, w, p) through `diagnostics.field_view` padded buffers. Metric as in §3, with
  c_P = 1/(1+D) = 1/7.
- "Steps to tol" is the first step after which |⟨u_x⟩/U∞ − 1| stays below tol. U∞ comes from a
  long plain march.

**Steps to tolerance:**

| case | plain 1e-4 / 1e-6 / 1e-8 | Anderson m=5 | m=3 | m=8 | m=2 |
|---|---|---|---|---|---|
| collocated ghost, Stokes | 315 / 1522 / 2652 | **27 / 55 / 77** | 30 / 58 / 96 | 28 / 43 / 78 | 56 / >120 / — |
| staggered, Stokes | 22 / 77 / 165 | **11 / 20 / 32** | 12 / 20 / 35 | — | — |
| staggered, Re ≈ 10 (μ = 0.05, Koren, dt = 3.906e-2, D = 0.5) | 262 / 392 / 524 | 105 / 181 / >200 | — | — | — |
| collocated, Stokes, **velocity-only state** | 315 / … | stalls: error 1.9e-4 at 150 steps | | | |

**Fixed point:**
- Staggered: after 120 Anderson steps, the error against the 2000-step plain reference is 3.3e-16.
- Collocated: 3.6e-9 against a 3000-step plain reference. That reference is itself about 2e-9
  short: its last-step change is 9.2e-12 at a rate of 0.996, so about 2.3e-9 remains.

**Ritz estimate ρ(I+M) on the window (§4.6):**
- Collocated: converges to **0.99616**, which matches the measured checkerboard tail rate 0.9962 at
  N = 16. The window captures the slow mode.
- Staggered: ≤ 0.954, except one spike of 1.09 at relative residual < 5e-12. That is round-off,
  hence the noise floor in D5.
- Re ≈ 10: ≤ 0.966.

**A march-unstable case** (staggered, μ = 0.0158, dt = 1.234e-2, which is CFL 0.5 on the Stokes
velocity estimate):
- The **plain march diverges** to NaN at step about 435; max|u| was 3.57 at step 400.
- Anderson is still bounded at step 900 (⟨u⟩ = 2.018, residual 1.45e-4, slowly converging). Its Ritz
  radius stayed at ≈ 0.992, with one non-consecutive spike of 1.013.
- So Anderson carries the march where the plain map at that Δt cannot go. §2.4 and §7 handle this
  case.

**Cost per step** (prototype, collocated N = 16, 6 threads): plain 17.2 ms/step over its first 315
steps, with 9.2 pressure iterations per step. The prototype runs 25.7 ms/step, but that includes
host copies and Python-loop Gram products, which the C++ design does not have (§6.4). Even so, wall
time to 1e-4 error is 5.4 s plain against 0.69 s accelerated.

### 1.2 Evidence for revision 1: the slow mode of the dense bed (2026-10-02, measured)

**Setup.** Plain march (`march_to_steady(accelerate=False)` settings, production) on the A1 bed:
φ 0.6, 64 spheres, seed 0, N = 64, Δt = 6h², PCG(200, 1e-8). It ran for 1500 steps, recording
the per-field residual every step and the full residual field at steps 300, 600, 1000 and 1500.
Fluid connectivity is taken from the face apertures `get_ox/oy/oz` (two fluid-centred cells are
connected when the shared face's aperture is > threshold). Script and commands: log, entry
"Revision 1".

**Staggered bed:**

| step | W-residual | velocity residual | pressure share of W² | cells holding 90 % of P-energy | in sealed pockets | behind apertures ≤ 0.2 | plain ⟨u_x⟩ error |
|---|---|---|---|---|---|---|---|
| 300 | 7.9e-5 | 1.3e-6 | 0.9997 | 105 | 58 % | 69 % | 5.6e-5 |
| 600 | 3.2e-5 | 3.1e-7 | 0.9999 | 66 | 64 % | 77 % | 1.7e-5 |
| 1000 | 1.6e-5 | 1.2e-7 | 0.99995 | 50 | 63 % | 83 % | 7.3e-6 |
| 1500 | 9.4e-6 | 6.0e-8 | 0.99996 | 41 | 59 % | 85 % | 3.7e-6 |

- **Connectivity:** 1,086 fluid components. One main component of 103,767 cells; 1,085 pockets
  holding 1,168 cells, nearly all single cells at sphere near-contacts, sealed by zero apertures.
- **Decay:** the W-residual's per-step rate creeps from 0.991 (steps 100–200) to 0.999 (steps
  1200–1500), a power-law tail rather than one eigenvalue. Successive late residuals are
  only partly aligned (cos 0.87 → 0.96), and the velocity residual is equally localized (90 % of
  its energy in 37–119 entries after step 1000).
- **Collocated bed:** the same structure. The ghost scheme already decouples 216 cells in 205
  components from its projection. The late pressure-residual energy is 74–99.9 % behind apertures
  ≤ 0.2, and the pressure share of W² is 0.9998.

**Physical reading.**
- A sealed cell's pressure is a gauge mode with no coupling to any velocity. Each pocket adds a
  null vector of the pressure operator, while the metric removes only the global constant, and the
  value drifts with whatever the projection solve writes into the decoupled row.
- A cell behind a face of small aperture a equilibrates through that face's flux at a rate
  1 − O(a): a lubrication-film pressure.
- Neither moves ⟨u_x⟩, and both dominate a pressure-weighted residual. That is why revision 0's
  W-residual target was unreachable on the bed while ⟨u_x⟩ had already converged: Anderson brought
  the monitor within 3e-5 by step 31–43, against 431 for the plain march.
- The observation is not specific to Anderson: the plain march leaves the same pocket pressures
  unconverged when it certifies.

### 1.3 Measured: the C++ build (WO-5, 2026-10-02)

Host = host-openmp, 8 threads, on a shared 48-core box (load 40–120); CUDA = RTX 5080, shared.
Every command and every number: `doc/steady_acceleration_log.md`, entry "WO-5". Step counts are
identical on host and CUDA and equal the revision-1 oracle (log R-6) wherever both exist.

**Production, window 5** (G2; plain / accelerated steps, ratios plain ÷ accelerated):

| case | steps | steps ratio | wall ratio host / CUDA | pressure-iteration ratio |
|---|---|---|---|---|
| §11 collocated N16 | 395 / 89 | 4.44 | 4.93 / 4.57 | 4.37 |
| §11 collocated N24 | 90 / 48 | 1.88 | 2.75 / 2.98 | 2.03 |
| §11 staggered N16 | 75 / 51 | 1.47 | 1.41 / 0.89 | 1.49 |
| §11 staggered N24 | 135 / 50 | 2.70 | 2.68 / 2.74 | 2.56 |
| Z&H 0.343 / 0.45, staggered N32 | 105 / 70, 115 / 54 | 1.50, 2.13 | 1.41, 2.03 / 1.57, 2.31 | 1.50, 2.16 |
| **dense bed φ 0.6, staggered** | 325 / 93 | **3.49** | **3.38–3.45 / 3.55–3.87** | 3.50 |
| **dense bed φ 0.6, collocated** | 190 / 83 | 2.29 | 2.48–2.53 / 2.38–2.73 | 2.44 |

N14/18/20 and m = 3/8: log. **D11 resolved by its rule: `accelerate=True`** (staggered bed ≥ 1.5×
in wall time on both backends). Pressure iterations per step at mixed iterates equal the plain
march's to ±3.5 % (Q12).

**Tight** (G1, |K_acc/K_plain − 1|, CUDA): ≤ 1.75e-9 on 11 of 14 cases; MPI np 2/4 ≤ 2.6e-12 of np 1,
np 1 = serial bit for bit (G4); restart ≤ 1.7e-11 (G5); finite Re ≤ 3.3e-9, 2.4× – 17.9× (G6,
except 1.06× at Re ≈ 100 on the A1 array). **Overhead** (G8, 64³): 0.77–0.85 ms per step on
CUDA = 5.3 % staggered / 2.0 % collocated of the plain step; `memory_bytes` equals §6.3 exactly.

**Open after WO-5 (gates failed; not resolved here):**
- *The Ritz guard at tight settings (G1, G7c).* False "unstable" on §11 collocated N14 and
  staggered N20: readings 1.001–1.005 at residual 5–7e-9, i.e. at 5× the floor 1000·τ = 1e-9,
  where the tight map's residual stalls; isolated readings up to 1.012 at residual 1e-6 – 1.5e-5.
  The oracle reproduces them. Production: 0 of 946 eligible readings above 1.0005.
- *The plain certificate on the staggered bed at rtol 1e-10 (G1).* The plain march certifies
  1.4e-8 below the fixed point (its tail runs at ~0.9999 per step against the instrument's
  slow_rate 0.997); the accelerated K is within ~5e-10 of a 60 000-step plain march.
- *G3 on the staggered bed:* 400 accelerated steps reach residual 1–2e-9, not 1e-9; K spread
  over dt 3.8e-7 (no instability: active, no restarts, residual still falling).
- *G8 on CUDA, staggered:* 5.3 % > 5 %, latency-bound (~26 synchronising calls per step); the
  §6.2 remedy (fuse pass 2's column reductions) is a core change.

---

## 2. Fixed-point argument

### 2.1 What the accelerator can and cannot change

Let g be the step map the solver actually computes, including its inexact inner solves (momentum
residual stop, MG-PCG or BiCGStab to rtol). Its state is x = (u, P [, u_f]) as in D2.

The accelerator never modifies g. It only chooses the input to the next evaluation:

    x_{k+1} = Σ_i α_i [ β g(x_i) + (1−β) x_i ],      Σ_i α_i = 1.

Discrete operators, coefficients, defaults and inner tolerances are untouched. `step()` is called
unchanged.

### 2.2 The reported state is a plain-march state, certified by the plain-march instrument

The driver reports the state after a **run of consecutive plain steps** x_{n+1} = g(x_n). The run
starts from the accelerated iterate and passes the §3.2 instrument with its constants unchanged. So
the accelerated driver reports exactly what the unaccelerated driver reports: a state of the plain
march that passed the same acceptance test. The only difference is where that plain run starts.

By design, the accelerator applies its mix lazily, at the start of the *next* call (§4.2). Between
calls the solver therefore always holds a genuine map output, with every derived quantity
consistent: `uStar_` for the reaction force, the face field, and the `last_*` diagnostics.

### 2.3 Uniqueness: both marches go to the same point

For the supported configurations (D10), g has a unique fixed point modulo the pressure constant.

- **Exact solves.** By Prop. 1 of `collocated_invisible_subspace.md`, P-stationarity gives
  ((ρ/Δt)I + μA_p)φ = 0, hence φ = 0, D_αΠu = 0, and the Δt-free steady system −μLu = F − GP. For
  the staggered scheme and the collocated ghost scheme that system has a unique solution modulo
  gauge (C2; the ghost scheme drops the solid rows that create §4's family; staggered has
  ker G = constants).
- **Inexact solves.** Let the pressure solve return φ with |A_p φ + DΠu*| ≤ τ|DΠu*| for relative
  tolerance τ. Write A_pφ = −DΠu* + e with |e| ≤ τ|DΠu*|. P-stationarity gives
  (ρ/Δt)φ = μDΠu* = −μ(A_pφ − e), i.e. ((ρ/Δt)I + μA_p)φ = μe.
  Then |DΠu*| ≤ |A_pφ| + |e| ≤ C τ |DΠu*| with C = O(1 + ‖μA_p((ρ/Δt)I + μA_p)⁻¹‖) ≤ 2, so
  DΠu* = 0 and φ = 0 whenever τ < 1/2.
- **The momentum solve.** It is warm-started from u^n and always runs at least one sweep. At an
  exact steady state its initial residual is zero, and a stationary sweep leaves an exact solution
  unchanged (`collocated_invisible_subspace.md` §5).

So the inexact map's fixed point is the exact one. Every certified state (§2.2) lies within the
instrument's remainder bound of it, whichever path led there.

**Consequence.** |K_acc − K_plain| ≤ |K_acc − K*| + |K* − K_plain| ≤ 2·rtol·|K| at production
settings, and → 0 at a tight stop. Gate G1 tests the tight form at ≤ 1e-8.

**Gauge.** In a periodic (or wall-only) box, g(x + c·1_P) = g(x) + c·1_P. The constant P mode is
neutral: it affects neither u nor K nor the certificate. The same holds per sealed fluid pocket
(§1.2). Since rev 1 the metric does not measure P at all (§3.2), so none of these modes needs
special treatment. They are carried along by the mix and otherwise left alone.

### 2.4 What the argument does not cover (stability), and how the design handles it

Type-II Anderson on a linear map is GMRES-like on (I − J)x = c (Walker & Ni). It converges whenever
I − J is nonsingular, **including when ρ(J) > 1**. So Anderson can reach a fixed point that the
plain march at the same Δt never reaches. Two kinds exist:

- **Numerical:** explicit advection beyond its CFL limit. Measured in §1.1. The fixed point is the
  Δt-independent discrete steady state, so a plain march at a stable Δt would reach the same point.
- **Physical:** an unstable steady branch of an unsteady flow. The plain march never converges.

Three defences, in the order they act:

1. **The Ritz guard (D5).** Anderson can suppress a growing mode only if that mode is in its
   window, because the mode's growth dominates the secant differences. The Galerkin projection of
   J − I onto the window then has spectral radius ρ(I+M) > 1. Three consecutive detections end the
   march as not-converged ("unstable").
   - Since rev 1 the guard reads only windows built by Anderson itself (mixed calls), and only
     while the residual is at least 1000× the inner-solve noise.
   - The one measured true positive (G7a, m = 8) fired at residual 9.4e-3, three decades above
     the production floor.
   - A growing mode that becomes visible only below the floor is left to defences 2 and 3, and to
     the instrument, which cannot pass a tail with R ≥ 1.
2. **Certification growth (§7).** If the plain steps grow from the accelerated state (residual
   ×10, or non-finite), acceleration is disabled and the plain march continues on its own terms.
   The outcome is then the unaccelerated outcome from that state.
3. **The residual risk is shared with the plain march.** An unstable mode that the transient never
   excites (it sits at round-off, for example by symmetry) is invisible to Anderson and to the
   plain march alike. The §3.2 instrument can stop the plain march on such a state too, so the
   answers agree.

**Out of scope, refused (D10):**
- Configurations whose fixed points are not unique, because Anderson would select among them
  differently from the march: gauge-exact, plain and embed collocated schemes.
- Configurations whose map is time-dependent and has no fixed point: moving geometry, VoF, phase
  change, CFD-DEM.

### 2.5 Mixing preserves the affine constraints

With β = 1, x_{k+1} lies in the affine hull of the g outputs, because Σα = 1. Every linear
constraint that all outputs satisfy, homogeneous or affine, is therefore satisfied by the mix:

- the discrete divergence of the staggered faces or of the collocated u_f (to solver tolerance);
- the solid masks;
- Dirichlet ghost values (inflow, walls).

With β < 1 the hull also contains earlier inputs; for those, the constraints hold to the extent that
the inputs satisfied them. Correctness does not need this, since the plain march starts from any
state. It explains why mixed inputs behave like plain-march inputs, for example why the projection's
right-hand side does not jump.

---

## 3. State vector and scaling

### 3.1 State (per grid; full padded buffers, `n_pad = e.x·e.y·e.z`, G = 2)

| configuration | fields (role) | n_s |
|---|---|---|
| `Solver` (staggered), any advection | `C[0].u, C[1].u, C[2].u` (Velocity, on faces), `P_` (Carried) | 4 |
| `SolverColocated`, ghost scheme, advection off **or** `set_uf_advection(False)` | `C[0..2].u` (Velocity, cell), `P_` (Carried) | 4 |
| `SolverColocated`, ghost scheme, advection on with projected-face advection (`advect_ && ufAdvect_`, the default) | the above + `uf_, vf_, wf_` (Carried) | 7 |

How to read the table:
- **"Carried"** fields are mixed, stored and differenced like the others, but excluded from the
  metric.
  - For the face field: it is a deterministic function of the previous step's (u*, φ), so its
    residual adds no independent information.
  - For P (rev 1; revision 0 measured it with role Pressure): a pressure error is measured through
    the velocity it drives at the next step (§3.2, D3).
- Decide the collocated row from the *configuration* (`advect_ && ufAdvect_`), not from
  `faceFieldValid_`.
- Collocated Stokes: `uf_` is output-only, since every projection overwrites it and nothing reads it
  before then. It is not state.

### 3.2 Metric (inner product) — rev 1: velocity only

For two state vectors a and b on one rank, with inner entries I (G ≤ x < e.x−G, and likewise for y
and z):

    ⟨a,b⟩ = Σ_{f ∈ Velocity} Σ_{i ∈ I} a_f(i) b_f(i)

Definitions:
- Sums are global, over ranks. Inner regions partition the global grid, so each entry counts once.
- Velocity entries count at every inner position, including masked solid ones, whose residual is 0.
- Carried fields (P, the face field) do not enter.

*Superseded by rev 1:* the W-metric of revision 0, ⟨a,b⟩_W = ⟨a,b⟩ + c_P² Σ_{fluid cells}
(a_P − ā_P)(b_P − b̄_P) with c_P = 1/(μ_int + ρ_int/Δt_int) and the global fluid mean removed when
gauged. It was rejected on the dense-bed measurement of §1.2 (D3). The oracle keeps it as
`--rev 0` / `--metric W` for ablation; the production code does not.

**Why velocity only.** The plain step maps a pressure error e_P to a velocity change at the next
step through the predictor's −G e_P term and the projection. So the velocity residual
already contains every pressure error that can ever change a velocity monitor, each weighted by its
effect.
- A pressure error with no velocity effect (a gauge constant, a sealed pocket) cannot change such
  a monitor, and is not measured.
- One with a small effect (a pocket behind a small aperture) is measured small.
- The collocated (π,0,0) checkerboard, the slow mode that motivated the c_P weight, is not
  invisible. It is what makes the plain ⟨u_x⟩ slow (0.996/step), so its velocity footprint is in
  the velocity residual. The velocity metric reaches the collocated N = 16 handover in 39 steps
  against the W-metric's 48 ("Revision 1", R1 table).

**Velocity scale and relative residual:**

    U² = Σ_{f ∈ Velocity} Σ_{i ∈ I} g_f(i)²          (g = the step output just computed)
    residual = sqrt(⟨r,r⟩ / U²)                      (r = g(x) − x; = 0 if r = 0 and U = 0; +inf if U = 0 ≠ r)

This is the relative *velocity* residual. It is what phase A's target, the restart test, the
engagement rule and the Ritz floor compare against.

**The units trap.** Everything here is internal. No physical conversion happens anywhere in the
accelerator, and none is needed: mixing is linear, and the metric is a ratio of velocities, so it is
unit-free.

---

## 4. Algorithm

### 4.1 Fixed constants (C++ `constexpr`, not user-settable)

| name | value | meaning |
|---|---|---|
| `kMaxWindow` | 8 | compile-time cap on m |
| `kEngageDecreases` | 2 | consecutive residual decreases before mixing engages |
| `kRestartGrowth` | 4.0 | restart if ρ_k > 4·ρ_min (since the last restart), at a mixed iterate |
| `kMaxRestarts` | 5 | disable acceleration at the 5th restart |
| `kNoiseFloor` | 1e-10 | below this relative residual, no restart test; also the lower bound of the Ritz floor |
| `kRitzFloorFactor` | 1000 | **(rev 1)** Ritz floor = max(kNoiseFloor, kRitzFloorFactor·τ), τ = the descriptor's `innerTolerance` |
| `kCondMin` | 1e-12 | minimum eigenvalue ratio of the Jacobi-scaled Gram (κ(ΔR) ≤ 1e6) |
| `kRitzDelta` | 1e-3 | instability if ρ(I+M) > 1 + 1e-3 |
| `kRitzHi` | 1e-2 | Ritz test only while residual ≤ 1e-2 (outside the strongly nonlinear initial transient) |
| `kRitzConsecutive` | 3 | consecutive detections that declare "unstable" |
| `kGelfandSquarings` | 20 | ρ(T) ≈ ‖T^(2^20)‖^(2^-20) |

User-settable: `window` m ∈ [1, 8] (default 5), and `mixing` β ∈ (0, 1] (default 1.0).

**τ, the inner-solve tolerance (rev 1).** The caller reports it in the descriptor as
`innerTolerance`.
- *Value:* flow sets it to `velocityResidualTolerance()` when that is > 0. Otherwise (the legacy
  update criterion, `set_velocity_residual_tolerance(0)`) it uses the active pressure driver's
  rtol: `chebRtol_` under Chebyshev, else `pcgRtol_`.
- *Why this tolerance:* the momentum solve's stop is relative to max|b|, an absolute scale, so it
  sets the map's noise floor. Measured: the production velocity residual stalls at ≈ 2e-8 = 2τ
  with τ = 1e-8.
- *Resulting floor:* 1e-5 at production settings, 1e-9 at tight ones. `innerTolerance = 0` means
  unknown and gives the floor 1e-10 (revision 0's behaviour).
- *Why 1000:* readings > 1 + kRitzDelta occur only at residual ≤ 5.5e-8 ≈ 5τ, apart from one
  isolated 1.006 at 140τ. The true positive measured on G7a fired at 9.4e-3, about 1e6·τ.

### 4.2 Data held by `AndersonCore`

- **Device:** `X`, `Rprev`, `Gprev`, `dR[0..m−1]` and `dG[0..m−1]`. Each is a set of n_s padded
  `Kokkos::View<double*>`, one per state field. That makes 2m + 3 state vectors; the history slots
  are used circularly.
- **Host:**
  - window bookkeeping: `mk` (columns in use); slot indices ordered oldest→newest; and **(rev 1)**
    `mixedCol[slot]`, whether the column in that slot was formed at a mixed call;
  - Gram blocks: the m×m blocks `RR`, `GG` and `GR`, where GR(i,j) = ⟨Δg_i, Δr_j⟩;
  - state flags and counters: `havePrev`, `pending`, `gamma[m]`, `engaged`, `decCount`, `rhoPrev`,
    `rhoMin`, `numRestarts`, `numResets`, `ritzCount`;
  - status and reporting: `status` ∈ {active, disabled, unstable}, `reason`, `residual`, and
    `ritzRadius` (last value, NaN if not computed).
  - *Superseded by rev 1:* the per-column fluid P-means `mR[j]`, `mG[j]`; with no pressure in the
    metric they do not exist.
- **Signature**, for change detection: internal Δt, ρ, μ and the body force (3 components) at the
  last call.

`reset()` clears the history and the counters: mk = 0; pending = engaged = false;
decCount = ritzCount = 0; rhoMin = rhoPrev = +inf. It keeps `havePrev`.

### 4.3 One call: `step(accelerate)` (the adapter wraps the core around `solver.step()`)

    step(accelerate):
      # --- 0. change detection -------------------------------------------------------------
      if signature(solver) != stored: reset(); havePrev = false; numResets++; store signature
      if havePrev:
          nChanged = Σ_{inner entries, all fields} [buf ≠ Gprev]           # device count, then MPI_SUM
          if nChanged > 0: reset(); havePrev = false; numResets++          # state written externally
      # --- 1. lazy mix (prepare) -------------------------------------------------------------
      mixed = accelerate and pending and status == active
      if mixed:                                    # buf holds g_{k-1} (== Gprev, just checked)
          for every padded entry e of every field (kernel MIX):
              c = Σ_j gamma[j]·dG_j(e);   d = Σ_j gamma[j]·dR_j(e)
              x = buf(e) − c − (1−β)·(Rprev(e) − d)
              buf(e) = x;  X(e) = x
      else:
          X = buf                                  # copy (kernel COPY)
      pending = false
      # --- 2. the map --------------------------------------------------------------------------
      try:   solver.step()
      except e:
          if mixed: buf = Gprev; status = disabled; reason = "step failed at an accelerated iterate: " + e
                    reset(); notice once (rank 0); return            # do NOT rethrow
          else:     rethrow                                            # the plain march's own failure
      # --- 3. provisional new column (does not touch Rprev / Gprev) ----------------------------
      if havePrev:
          s = (mk < m) ? next free slot : oldest slot
          kernel DIFF over padded entries: r = buf − X;  dR_s = r − Rprev;  dG_s = buf − Gprev;  X = r
      else:
          kernel R: X = buf − X                     # X now holds r_k
      # --- 4. reductions (§6.1; rev 1: velocity fields only, no pass 1) ------------------------
      ⟨X,X⟩, U², and for every window column j (plus s):
              RR(s,j), GG(s,j), GR(s,j), GR(j,s), b_j = ⟨dR_j, X⟩   -> Allreduce
      # --- 5. host decision (all ranks compute; rank 0's packet is broadcast and wins) ---------
      rho = sqrt(⟨X,X⟩);  residual = rho / sqrt(U²)
      if not finite(rho) or (mixed and solver.pressureSolveFailed()):
          if mixed: buf = Gprev;  reason = "non-finite residual / failed pressure solve at an accelerated iterate"
          else:     reason = "non-finite residual on a plain step"
          status = disabled; reset(); havePrev = false; return
      if havePrev: accept column s: copy the new RR/GG/GR row and column; mixedCol[s] = mixed; mk = min(mk+1, m)
      if mixed and residual ≥ kNoiseFloor and rho > kRestartGrowth·rhoMin:
          reset(); numRestarts++; rhoMin = rho
          if numRestarts ≥ kMaxRestarts: status = disabled; reason = "too many restarts"
      decCount = (rho < rhoPrev) ? decCount+1 : 0;  if decCount ≥ kEngageDecreases: engaged = true
      rhoPrev = rho;  rhoMin = min(rhoMin, rho)
      # --- 6. commit on the device --------------------------------------------------------------
      swap(Rprev, X)                                # Rprev := r_k (handle swap; X becomes scratch)
      Gprev = buf                                   # copy (kernel COPY)
      havePrev = true
      # --- 7. next coefficients ------------------------------------------------------------------
      if status == active and engaged and mk ≥ 1:
          while mk > 1 and condScaled(RR) < kCondMin: drop the oldest column (mk--)
          gamma = solveTruncated(RR, b)             # §4.4
          pending = true
          eligible = mixed and mk ≥ 2 and all(mixedCol[j] for j in window)        # rev 1
                     and ritzFloor ≤ residual ≤ kRitzHi                          # ritzFloor: §4.1
          if eligible:
              ritzRadius = gelfand(I + pinvTruncated(XX)·XR)   # §4.6
              ritzCount = (ritzRadius > 1 + kRitzDelta) ? ritzCount+1 : 0
              if ritzCount ≥ kRitzConsecutive:
                  status = unstable; reason = "the plain map is locally unstable at this dt (Ritz radius …)"
                  pending = false; reset()
          else:
              ritzCount = 0                                                        # rev 1
      broadcast (rank 0 → all): status, reason code, mk and slot order, gamma[0..mk-1], pending, residual,
                                ritzRadius, numRestarts

**Notes that are not left to the implementer:**
- The **first call** has `havePrev = false`. It evaluates one plain step, records `Rprev`/`Gprev`,
  and forms no column. The residual of that first evaluation is defined (x = the initial state)
  and counts for engagement. With β = 1, the initial state never enters a mix (§2.5).
- **A plain call** (`accelerate = False`) records history exactly like an accelerated one. A plain
  step is an Anderson step with γ = 0. The pending mix is discarded, not deferred. Its column is
  flagged `mixedCol = false`, so the Ritz guard does not read a window containing it (rev 1).
  The least squares does use it: plain-march columns capture the slowest mode well.
- **`mixed` is rank-consistent** (`accelerate` is the same on every rank, `pending` is broadcast),
  so `mixedCol` and the eligibility rule need no extra broadcast.
- **The column-drop rule (cond < kCondMin) and the zero-norm drop** are unchanged. With the velocity
  metric, a column whose velocity differences are exactly zero (only Carried fields moved) has
  D_j = 0 and is dropped first.
- **Restart** keeps `Rprev`/`Gprev` (step 6 still commits). The next call therefore forms a column
  at once, and mixing resumes after re-engagement (two decreases).
- **After `status ≠ active`,** `step()` keeps working as a plain step with no history maintenance.
  It skips the reductions too; residual stays at its last value. The driver uses `solver.step()`
  directly then anyway.
- **The state check (step 0) compares inner entries only.** Getters and ghost exchanges
  (`exchange_field`, `max_open_divergence`, the force getters) may refresh ghosts between calls;
  ghosts refreshed from unchanged inner values are not a state change. The mix acts on the full
  padded buffer as found.

### 4.4 The least-squares solve (`solveTruncated`, host, deterministic)

The goal is min_γ ‖r_k − ΔR γ‖ in the metric of §3.2 (rev 1: velocity only), i.e. RR γ = b, with
mk ≤ 8.

1. D = diag(RR)^{1/2}. If any D_j = 0, drop that column first.
2. A = D⁻¹ RR D⁻¹ (unit diagonal); b̃ = D⁻¹ b.
3. Eigendecompose A by cyclic Jacobi rotations, in fixed sweep order (p < q, row-major). Stop when
   the off-diagonal Frobenius norm is ≤ 1e-15·‖A‖_F, or after 50 sweeps.
4. `condScaled` = λ_min/λ_max. The caller drops the oldest column while it is < `kCondMin`.
5. y = Σ_i v_i (v_iᵀ b̃)/λ_i; γ = D⁻¹ y.

### 4.5 Coefficient identity used by MIX

x_{k+1} = g_k − ΔG γ − (1−β)(r_k − ΔR γ), where ΔR = [Δr_j] and ΔG = [Δg_j] over the window, in
slot order. This equals Σα_i[βg_i + (1−β)x_i] with Σα = 1, and α never needs to be formed. For
diagnostics, Σ|α| is computable from γ (α_oldest = γ_oldest, …, α_newest = 1 − γ_newest after
differencing). Log it; do not test it (D5).

### 4.6 The Ritz guard (host)

Run only on an eligible call (§4.3 step 7, rev 1). From the cached blocks (rev 1: velocity inner
products, §3.2):
- XX = GG − GR − GRᵀ + RR, i.e. ⟨Δx_i, Δx_j⟩ with Δx = Δg − Δr.
- XR = GR − RR, i.e. ⟨Δx_i, Δr_j⟩.

Under the velocity metric the estimate is the window's action on the velocity components. A growing
mode that Anderson suppresses must show there to matter to a velocity monitor; one with no velocity
component at all can only blow up P, which ends in the non-finite-residual exit.

Then:
1. M = XX⁺·XR, where XX⁺ is the truncated pseudo-inverse from §4.4 with eigenvalues below
   `kCondMin`·λ_max discarded and no column dropping.
2. T = I + M.
3. ρ(T) by Gelfand: A ← T/‖T‖_max, L ← log‖T‖_max. Then 20 times: A ← A·A; s ← ‖A‖_max;
   A ← A/s; L ← 2L + log s. Result: ρ = exp(L/2^20), with ρ = 0 if A vanishes.

The bias is ≤ ln(C)/2^20 ≈ 1e-5 for C ≤ 1e6, well below `kRitzDelta`.

---

## 5. Placement and API

### 5.1 Files

| file | content |
|---|---|
| **core:** `include/peclet/core/solver/anderson.hpp` | `peclet::core::solver::AndersonCore` (D13) — grid-agnostic: takes a core-level descriptor `AndersonState`, which since rev 1 holds the state views, the roles Velocity/Carried, the ghost width `G`, `innerTolerance` (§4.1) and the collectives `AndersonComm`; the rev-0 fields fluid mask, `cP` and `gauged` and the Pressure role are deleted by WO-3b part 2. It implements §4 (kernels MIX/COPY/DIFF/R/COUNT, the reductions, host LS, guards, MPI packets). Header-only, `inline` members, beside `csr_bicgstab.hpp`/`vector_ops.hpp`. No flow dependency — unit-testable with synthetic maps. The parameter signature (dt, ρ, μ, F) is NOT in the core descriptor; the flow adapter checks it. |
| `src/anderson_accelerator.hpp` | `template <class Grid> class AndersonAccelerator` — holds `Solver<Grid>&` + an `AndersonCore`; `step(bool accelerate)` = §4.3 around `solver.step()`; signature read; status accessors. Explicitly instantiated in `src/flow_solver_staggered.cpp` / `src/flow_solver_colocated.cpp` with a matching `extern template` at the end of the header (the G.8 pattern; link via `peclet_flow_solver`). |
| `src/flow_ibm.hpp` | declare `struct MarchState` (= a `core::solver::AndersonState` + the flow signature) and `MarchState marchState();` (public C++, **not bound**). |
| `src/flow_ibm_diagnostics.hpp` | define `marchState()`: builds the field list and roles of §3.1 (velocity components Velocity, `P_` and the face field Carried), `e_`, `G`, `innerTolerance` (§4.1: `velocityResidualTolerance()` if > 0, else `useChebyshev_ ? chebRtol_ : pcgRtol_`), the signature values, and (MPI) `comm_`/`distributed_`; throws `std::runtime_error` naming the first refusal of §5.3. |
| `src/flow_bindings.cpp` | bind the two adapters as private classes `_AndersonAccelerator` / `_AndersonAcceleratorColocated`; add the factory `diagnostics.anderson_accelerator(window=5, mixing=1.0)` (returns a new accelerator; `nb::keep_alive` on the solver). |
| `packaging/flow_steady.py` → installed as `peclet/flow/steady.py` | `march_to_steady` and `MarchResult` (pure Python, §7); `flow_init.py` adds `from .steady import march_to_steady, MarchResult`; `CMakeLists.txt` gets the `configure_file` (build tree) and `install` lines mirroring those for `flow_init.py` (lines 194, 353). |

`MarchState` fields (rev 1):
- `std::vector<CCField> fields`;
- `std::vector<Role> roles` (Velocity / Carried);
- `C3 e`; `int G`;
- `double innerTolerance`;
- `std::array<double,6> signature` (dt, rho, mu, Fx, Fy, Fz, all internal);
- under `PECLET_FLOW_MPI`, `MPI_Comm comm` and `bool distributed`.

*Superseded by rev 1:* the fields `CCConst sdf`, `double cP`, `bool gauged` and the role Pressure.
If WO-3b part 2 is declined (Q14), the core keeps them: `marchState()` then leaves `sdf` empty,
`cP` at 1 and `gauged` false, and gives `P_` the role Carried, so the core never reads them.

### 5.2 Python surface

**Public tier** (module level; NAMING: verbs for actions, `num_*` counts, `rtol` like
`set_pressure_pcg`):

```python
peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, max_steps=5000, accelerate=True,
                            window=5, check_every=5, num_passes=3, slow_rate=0.997,
                            roundoff=1e-11, callback=None) -> MarchResult
```

- `monitor()` returns the scalar the §3.2 instrument watches, for example the mean of the velocity
  along the force. **Under MPI it must return the same value on every rank** (a global reduction);
  every stop decision is a pure function of it and of `acc.residual`, which is rank-consistent by
  construction.
- `callback(steps, phase)` is optional and called after every step, with phase ∈ {"accelerate",
  "certify", "plain"}. Use it for logging and checkpoints.
- `MarchResult` is a frozen dataclass: `converged: bool`, `steps: int`, `accelerated_steps: int`,
  `reason: str` ∈ {"certified", "max_steps", "unstable", "diverged"}, `num_restarts: int`,
  `monitor: float` (the last value).
- `accelerate=True` on an unsupported configuration **raises** with the refusal reason and the hint
  "pass accelerate=False". An allocation that does not fit raises with the byte count (§6.3).
- No parameter is a cell-unit quantity. `rtol`, `slow_rate` and `roundoff` are dimensionless;
  `max_steps`, `check_every`, `num_passes` and `window` are iteration counts.

**Developer tier** (`s.diagnostics`):

```python
acc = s.diagnostics.anderson_accelerator(window=5, mixing=1.0)
acc.step(accelerate=True)   # one map evaluation (+ the lazy mix when accelerate and engaged)
acc.residual                # property: relative velocity residual of the last evaluation (inf before the first)
acc.status                  # "active" | "disabled" | "unstable";   acc.reason: str
acc.num_restarts, acc.num_resets, acc.num_columns, acc.ritz_radius, acc.window, acc.mixing,
acc.memory_bytes            # properties
acc.reset(); acc.disable()  # methods
```

**Opt-in vs default.** `Solver.step()` is unchanged, and a hand-written `for … s.step()` loop is
bit-identical to today. Acceleration is the default *of the new function* `march_to_steady`, which
ships only after the gates (§8) pass. `accelerate=False` reproduces the study's `march()` bit for
bit (G0).

### 5.3 Refusals (`marchState()` throws)

The check runs at construction. It runs again at every `step` call: this is a re-evaluation of the
host flags only, so a feature enabled after construction is refused at the next call.

Each refusal throws with its own message:
1. `SolverColocated` whose active scheme is not the fluid-only ghost projection (gauge-exact, plain,
   embed, `set_face_interp` overrides). Message: "the fixed point is not unique / the march is
   unstable for this scheme".
2. `enable_vof`, `enable_vof_momentum` or VoF blocks.
3. `enable_phase_change`.
4. Any transported scalar (`add_scalar`).
5. Porous continuity.
6. Variable ρ or μ (any property closure or `rho`/`mu` field).
7. A `drag_beta` field.
8. Cell force fields (`force_*`).
9. Moving scene instances.
10. `set_superficial_velocity`.
11. Pressure warm start on (register: "set_pressure_warmstart(True) diverges on the steady Stokes
    march").
12. The balanced-force projection active.

Allowed:
- domain BCs of every type (inflow, wall, slip, outflow, periodic);
- static scenes;
- outer Picard iterations;
- any pressure driver except with warm start;
- velocity MG on or off;
- the wall-banded blend and rotational filter on staggered, since they are part of the map.

---

## 6. MPI and GPU design

### 6.1 Reductions, collectives and determinism

**Kernels:**
- Elementwise kernels (COUNT, MIX, COPY, DIFF, R) are a 1-D `Kokkos::RangePolicy` over the padded
  linear index, i.e. storage order, which satisfies the `iteration_order` rule.
- Reductions over inner entries use `MDRange3<CCExec>` from `policy.hpp`, x fastest.

**Pass 2 packet layout:**
- One fused reduction per window column j, each a 5-double `Kokkos::Sum` on a small struct:
  RR(s,j), GG(s,j), GR(s,j), GR(j,s), b_j.
- One reduction for ⟨X,X⟩ and U².
- (rev 1) Every reduction reads the Velocity fields only; Carried fields are never read by a
  reduction.
- Each reduction reads at most 5 vectors (dR_s, dG_s, dR_j, dG_j, X), so its value type stays small
  enough for GPU shared memory. A single ~45-double reducer is rejected for that reason.
- The partial results are packed into one host array of 5·mk + 2 doubles, followed by **one**
  `MPI_Allreduce(SUM)` on the solver's communicator.

**Collectives per accelerated step** (all latency-bound, ≤ 42 doubles):
1. COUNT → `MPI_Allreduce(SUM)`, 1 double.
2. Pass 2 → `MPI_Allreduce(SUM)`, 5·mk + 2 doubles.
3. The decision packet → `MPI_Bcast` from rank 0, ≤ 16 doubles plus ints.

*Superseded by rev 1:* revision 0's pass 1, a 3-double `MPI_Allreduce` of the gauged fluid pressure
sums, which went with the pressure metric.

A step already performs dozens of collectives (MG-PCG two per iteration, the momentum stop, and so
on), so these three add < 0.1 % at 1536 ranks. On a distributed run with ~30 µs collectives that is
about 90 µs against a step of ~0.8 s.

**Determinism contract:**

| comparison | guarantee |
|---|---|
| np = 1 (MPI build) vs the serial build | **bit-identical** (Allreduce and Bcast on one rank are identities) |
| same np, same backend, same build | **bit-identical run to run** (deterministic Kokkos reductions; the suite's `state_hash` gates rely on the same) |
| np = 2, 4, … vs np = 1 | **tolerance**: iterates differ at reduction-order round-off amplified by κ(ΔR) ≤ 1e6; the converged K agrees to the fixed-point tolerance (G4) |
| across ranks within one run | **identical γ, status, mk and slot order** (rank 0 broadcast) — a correctness requirement: each rank mixes its own ghosts, and they stay equal to the neighbour's mixed inner values only if all ranks use the same coefficients |

### 6.2 Memory traffic per accelerated step (double, n_s = 4, m = 5; units: state vectors)

| kernel | vectors read + written |
|---|---|
| COUNT (buf vs Gprev, inner) | 2 |
| MIX (buf, Rprev, 2m history → buf, X) | 2m + 4 = 14 |
| DIFF (buf, X, Rprev, Gprev → dR_s, dG_s, X) | 7 |
| pass 2 (m+1 reductions × ≤ 5 reads, velocity fields only = ¾ of a vector) | ≈ 22.5 |
| commit (Gprev ← buf) | 2 |
| **total** | **≈ 48 vectors ≈ 1.5 KB per cell** (rev 0: 56 vectors, 1.8 KB, with pass 1 and P in pass 2) |

A plain step moves the equivalent of ≈ 48 KB per cell (4.7–6.2e-8 s/cell-step on an RTX 5080 at
~960 GB/s), so the overhead is ≈ 3 % of a step. Gate G8 bounds it at 5 % on CUDA. If the gate
fails, the first optimisation is to fuse pass 2's per-column reductions in pairs; numerics are
unchanged.

### 6.3 Memory (the binding constraint)

    bytes = (2m + 3) · n_s · 8 · n_pad          (n_pad = Π (n_axis + 4) for a block of n_axis inner cells)

| configuration | m = 3 | **m = 5 (default)** | m = 8 |
|---|---|---|---|
| n_s = 4 (staggered; collocated Stokes), per inner cell, unpadded | 288 B | **416 B** | 608 B |
| … with the G = 2 padding of a 212³ block (×1.058) | 305 B | **440 B** | 643 B |
| … of a 32³ MPI block (×1.42) | 409 B | 591 B | 863 B |
| n_s = 7 (collocated + projected-face advection), unpadded | 504 B | 728 B | 1064 B |

**On the 16 GB card** (the brief's 9.5 M-cell limit is 1.65 KB/cell, i.e. 15.7 GB usable):
- n_s = 4, m = 5: the largest grid is 15.7e9 / (1650 + 440) ≈ **7.5 M cells (≈ 196³)**.
- m = 3: ≈ 8.0 M.
- m = 2 (7 vectors, 237 B): ≈ 8.3 M.

**At 9.5 M cells no window ≥ 2 fits.** The constructor's allocation raises with the message:

> "AndersonAccelerator: window m needs B bytes (formula); window 2 needs B2; run on more ranks or pass accelerate=False"

The run then either uses two GPUs (the accelerator is distributed) or the plain march. On an 80 GB
H100 the history is not binding.

---

## 7. Interaction with the stop criterion (rev 1)

`march_to_steady` keeps the §3.2 instrument **verbatim** as the acceptance test. The instrument is
`certify()` below:
- d is the change of `monitor()` over a block of `check_every` steps;
- a block passes if 0 < R < 1 and |d|/(1 − max(R, slow_rate^check_every)) < rtol·|m|, or if
  |d| ≤ roundoff·|m|;
- a stop needs `num_passes` consecutive passes.

It is applied only to runs of **consecutive plain steps**. Revision 1 changes the driver around the
instrument (target residual, budget, early exit), never the instrument.

    march_to_steady(solver, monitor, rtol, max_steps, accelerate, window, ...):
        steps = 0
        if not accelerate:
            return certify(solver.step, budget=None)                 # == study march(), bit for bit
        acc = solver.diagnostics.anderson_accelerator(window)        # raises on refusal / allocation
        target = (1 - slow_rate) * rtol                              # on the VELOCITY residual (rev 1)
        budget = 2 * (num_passes + 3)                                # rev 1 (rev 0: num_passes + 3)
        while steps < max_steps:
            # ---- phase A: accelerate -----------------------------------------------------------
            best, since, stagnated = inf, 0, False
            while steps < max_steps and acc.status == "active" and acc.residual > target:
                acc.step(True); steps += 1; callback(steps, "accelerate")
                if acc.status == "unstable": return MarchResult(False, steps, "unstable", ...)
                if acc.residual <= 0.5 * best: best, since = acc.residual, 0
                else: since += 1
                if since >= 10 * window: stagnated = True; break
            if acc.status != "active":
                return certify(solver.step, budget=None)             # plain march from here
            # ---- phase B: certify on consecutive plain steps ----------------------------------
            out = certify(lambda: acc.step(False), budget=budget,
                          growth_ref=acc.residual, slow_exit=True)   # see below
            if out == "pass": return MarchResult(True, steps, "certified", ...)
            if out == "max":  return MarchResult(False, steps, "max_steps", ...)
            if out == "growth" or stagnated:
                acc.disable(); return certify(solver.step, budget=None)
            target *= 0.1                                            # "budget" or "slow": resume A

The details below are fixed.

**`certify(step_fn, budget, growth_ref=None, slow_exit=False)`** runs the §3.2 loop with a *fresh*
block state (`prev = dprev = None`, `passes = 0`). Its step counter starts at 0, so `monitor()` is
sampled at local steps 4, 9, 14, … exactly as the study does. It returns:
- "pass" at a stop;
- "budget" after `budget` blocks without a stop (never when `budget is None`);
- "max" at `max_steps`;
- with a `growth_ref`, "growth" as soon as `acc.residual > 10·growth_ref`, or as soon as the residual
  or the monitor is non-finite;
- **(rev 1)** with `slow_exit`, "slow" at the first block that fails while R is defined and
  slow_rate^check_every ≤ R < 1. Such a block failed on the remainder bound alone: its tail is
  geometric, at least as slow as the instrument assumes, and too large. The test reads the same d,
  R and bound the instrument just computed. It is the driver's policy, not part of the instrument.
- A non-finite `monitor()` in a `budget=None` run returns `MarchResult(False, …, "diverged")`. This
  is the one behavioural addition to the plain path; it acts only on runs that are already
  diverging, and G0 tests converging cases.

**The target** is (1 − slow_rate)·rtol on the relative velocity residual, i.e. 3e-7 at the
defaults. It is the residual at which the slowest mode the instrument assumes (rate `slow_rate`) can
carry at most rtol of remaining change, since its error is at most residual/(1 − slow_rate).
- Measured at the handover (revision-1 oracle, all seven cases): monitor error 2e-8 – 1.4e-5, i.e.
  ≤ 0.14·rtol.
- *Superseded by rev 1:* the same number on the W-residual. On the dense bed that residual is
  dominated by pressure modes that carry no monitor change (§1.2), and the target was not reached
  in 120 accelerated steps.

**Why the budget is 2·(num_passes + 3) blocks.**
- *The transient.* An Anderson iterate's next residual is small because the slow-mode and
  medium-mode contributions cancel, so the monitor's first block changes after phase A start near
  zero, then change sign or grow (R > 1). The instrument needs a geometric tail, which emerges
  only once the medium modes have decayed.
- *Measured* (traces in the log): R > 1 or R < 0 for up to 5 blocks at |d| = 1e-8 – 1e-10 of |m|.
  Every revision-1 production run then passed within 5–10 blocks (25–50 plain steps), except the
  collocated bed at m = 3 (12 blocks, then one resume).
- *Revision 0's budget* of num_passes + 3 = 6 blocks allows one failed block before a resume. The
  resume re-creates the transient. In the ablation with every other revision-1 change in place, it
  cost 6–140 extra steps on 6 of 21 runs and gained nothing on the others (log, budget table:
  6 / 9 / 12 blocks).

**Why the early "slow" exit, and why the resume.**
- If the target was not tight enough for a mode slower than `slow_rate`, the plain tail decays at
  that mode's rate. Resuming Anderson with the history intact deflates it in a few steps, and the
  plain columns just recorded by `acc.step(False)` (§4.3) carry exactly that mode.
- The early exit takes that path as soon as the instrument shows a clean, slow, too-large tail,
  instead of after the whole budget.
- It never fired in the revision-1 matrix; it is the insurance that keeps the larger budget from
  costing 60 plain steps on such a case.

**Why a growth exit returns to the plain march** rather than failing: growth means the plain map at
this Δt departs from the accelerated state. The honest outcome is whatever the unaccelerated march
does from there: converge elsewhere, oscillate until `max_steps`, or diverge.

**Measured cost** (revision-1 oracle, production, m = 5): the full table is in "Revision 1".
- §11 collocated N = 16: 39 accelerated + 50 certification = 89 steps against 395, **4.4×**.
- Staggered N = 16: 21 + 30 = 51 against 75, **1.47×**. The instrument's fixed cost of at least
  25 plain steps dominates.
- Re ≈ 10: 106 + 25 against 345, **2.6×**.
- Dense bed: 68 + 25 against 325 staggered, **3.5×**; 58 + 25 against 190 collocated, **2.3×**.

---

## 8. Verification gates

**Common settings:**
- "Tight" means: `set_pressure_pcg(True, 400, 1e-12)` (the ghost scheme's BiCGStab takes the same
  rtol); `set_velocity_residual_tolerance(1e-12)`; `march_to_steady(rtol=1e-10, max_steps=20000)`.
- "Production" means: the study settings (`set_pressure_pcg(True, 200, 1e-8)`, 200 velocity
  sweeps); `rtol=1e-4`; window 5.
- K is the Zick–Homsy-normalised drag of `tests/study/study_avg_velocity_spheres.py`
  (`drag_K(⟨u_x⟩)`), or the case's own K for beds.
- Host means `host-openmp` with `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`. CUDA means the RTX 5080
  `nvidia-cuda` prefix.
- **A twice-failed gate stops the work order.** Numerics and constants are never tuned to pass; this
  is the register's escalation rule.

| gate | what, measured how | configurations | pass threshold |
|---|---|---|---|
| **G0** inertness | (a) `tests/regression/state_hash.py` before/after; (b) `march_to_steady(accelerate=False)` vs the study's `march()`; (c) full ctest battery | (a) all entry paths; (b) §11 φ=0.125 N = 16, 24, collocated + staggered, production; (c) host tree, `-LE bench` | (a) identical hashes; (b) **identical step count and bit-identical ⟨u_x⟩**; (c) 188 existing + new tests all pass |
| **G1** fixed point | K accelerated vs K plain, both `converged=True`, tight | §11 N = 14, 16, 18, 20, 24 (collocated ghost + staggered); Z&H SC array φ = 0.343 and 0.45 at N = 32 (staggered); dense random bed φ ≈ 0.6 (§10 Q2), staggered + collocated ghost | **\|K_acc/K_plain − 1\| ≤ 1e-8** every case |
| **G2** speed-up | steps to `converged=True`, pressure iterations and wall time incl. accelerator overhead, production | all G1 cases, host and CUDA | §11 collocated N = 16: **steps ratio ≥ 3.0 and wall ratio ≥ 2.7**; every case: steps_acc ≤ steps_plain **and** \|K_acc/K_G1 − 1\| ≤ 1e-4; **dense bed: the D11 rule (wall ≥ 1.5×, staggered decides, Q13)**; report all ratios (rev-1 oracle at m = 5: 4.4× collocated N16, 1.47–2.7× staggered §11, 2.6× Re ≈ 10, 3.5× / 2.3× staggered / collocated bed) |
| **G3** no new instability | 400 unconditional `acc.step(True)` (no stop), tight inner solves | §11 N = 16 collocated + staggered at νΔt/h² ∈ {6, 60, 600, 1e4}; dense bed at Δt = 60 and 600 (cell units, as in `collocated_invisible_subspace.md` §3) and at νΔt/h² = 1e4 | status stays "active"; ≤ 1 restart per 100 steps; running-min residual reaches ≤ 1e-9; residual at step 400 ≤ 10 × its running min; the four-Δt K agree to 1e-8 (C2 under acceleration) |
| **G4** MPI | tight `march_to_steady`, kokkos_mpi tree, `mpirun --bind-to none` | §11 N = 32 collocated ghost + staggered, np = 1, 2, 4 | \|K_np/K_1 − 1\| ≤ 1e-9; np=1 (MPI build) vs serial build: identical state hash after 60 accelerated steps; step counts within ±10 % across np (report) |
| **G4c** MPI ctest | `test_anderson_mpi` (C++, `tests/kokkos_mpi`) | staggered N = 16 sphere, 40 `acc.step(True)`, np = 1, 2, 4; plus U7 | max\|u_np − u_1\| ≤ 1e-8·max\|u_1\|; γ bitwise equal on all ranks of a run (assert after the Bcast) |
| **G5** restart | interrupt at step 25: `get_field` u,v,w,p → fresh solver, same setup → `set_field` → `march_to_steady` | §11 N = 16 collocated + staggered, tight | K vs uninterrupted accelerated K ≤ 1e-8; total steps ≤ uninterrupted + 15 (report); variant with `set_state` (velocity only): K ≤ 1e-8, extra steps reported |
| **G6** finite Re | tight `march_to_steady`, accelerated vs plain | staggered §11 sphere N = 16 at μ = 0.05, dt = 3.906e-2 (Re ≈ 10, measured stable); collocated ghost same case (state n_s = 7); a random array at Re ≈ 10 and ≈ 100 (§10 Q3) | \|K_acc/K_plain − 1\| ≤ 1e-8; steps_acc ≤ steps_plain; report ratios (prototype: 2.5× to 1e-4 error at Re ≈ 10) |
| **G7** unstable / unsteady | (a) both drivers on a case whose plain march diverges; (b) synthetic unstable maps; (c) false-alarm census | (a) staggered §11 sphere N = 16, μ = 0.0158, dt = 1.234e-2 (plain NaN at step ≈ 435, measured); (b) core unit test U4; (c) every G1–G6 run | (a) `converged=False` for accelerate True **and** False, no K reported; (b) status "unstable" within 3m steps of engagement; (c) status never "unstable", and every **eligible** Ritz reading (§4.3 step 7, rev 1) ≤ 1.0005 (rev-1 oracle: max 0.9974 over 21 runs) |
| **G8** performance + memory | accelerator time (inside the adapter, excluding `solver.step()`), mean over 50 steps; `memory_bytes` | 64³ §11-type case, staggered + collocated, CUDA and host | overhead ≤ 5 % (CUDA), ≤ 8 % (host) of the mean plain step; `memory_bytes` = formula §6.3 exactly |

**Core unit tests** (in **core**: `tests/test_anderson.cpp`, Kokkos-gated like `test_graph_amg_device.cpp`; U7 in core's MPI tests; synthetic maps on Views, host and CUDA):
- **U1 linear contraction**, as re-specified in core-anderson `2defc9c` (U1a: fast bulk + one
  isolated mode 0.996 converges in 32 steps; U1b: the continuum [0, 0.996] stated as a limit of a
  window of 5; U1c: three isolated modes). *Superseded:* this note's original U1 (a continuum in
  ≤ 80 steps), which a window of 5 cannot meet.
- **U2 affine hull:** every g output satisfies Σx_i = S. Mixed states satisfy it to 1e-14·|S|.
- **U3 restore:** a map that returns NaN at the 3rd mixed iterate. Afterwards buf == Gprev
  bitwise, status "disabled", and the next `step(false)` works.
- **U4 instability:** J with eigenvalues {0…0.9, 1.02} and a 2×2 rotation block of modulus 1.01.
  Status becomes "unstable". A stable twin (max 0.996) never does, and its ritz_radius converges
  within 1e-3 of 0.996.
- **U5 conditioning:** a map whose differences are rank-1. Columns drop, γ stays finite, no NaN.
- **U6 change detection:** writing an inner entry between calls resets the history (`num_resets`
  increments, no mix applied). Writing only ghosts does not.
- **U7 (MPI):** U1 distributed over np = 1, 2, 4. γ is equal on all ranks, and iterates are within
  1e-13 of np = 1 after 20 steps.
- **U8 (rev 1) — no Ritz on plain windows.**
  - *Setup:* x ← Jx + c + η, with J diagonal in [0, 0.999] and η a deterministic pseudo-random
    perturbation of 1e-8·‖x‖ per evaluation (the inner-solve noise).
  - *Plain phase:* 200 `step(false)` calls. Assert `ritzRadius` is NaN after every call and the
    status stays "active".
  - *Accelerated phase:* 50 `step(true)` calls with `innerTolerance = 1e-8`. Assert status never
    "unstable".
  - *Control:* the same accelerated sequence with eligibility forced to the rev-0 rule (a test-only
    hook, or a recorded pre-delta build) is allowed to fail. Record it; do not assert it.
- **U9 (rev 1) — the Ritz floor.** U4's stable twin run to a residual of 1e-12.
  - With `innerTolerance = 1e-8`: assert `ritzRadius` is NaN on every call whose residual is
    < 1e-5, and finite on at least one eligible call above it.
  - With `innerTolerance = 0`: assert it is evaluated down to 1e-10.
  - U4 itself must still report "unstable" within 3m steps of engagement (it runs only mixed
    calls).
- **U10 (rev 1, WO-3b part 2) — Carried fields.** A two-field state (Velocity + Carried) whose map
  moves only the Carried field after step 10. Assert:
  - the residual is 0 from then on;
  - a Carried field that satisfies Σx_i = S on every output satisfies it after a mix to
    1e-14·|S| (U2 for Carried).

---

## 9. Work orders (dependency order; each ends with its gate)

All work happens in a worktree `../flow-anderson` on branch `anderson`. Commit named paths only, and
run the blocking clang-format 18.1.8 check on `src/` and `tests/`. Record numbers in
`doc/steady_acceleration_log.md`, which is append-only.

**WO-1 — Python oracle and parameter confirmation (host only; no production code). DONE.**
- *Revision 0 run:* `tests/study/anderson_oracle.py` (`16393b9`) ran rev 0 and STOPPED on its own
  rule, a false "unstable" (log, entry 2026-10-02 WO-1). That stop produced revision 1.
- *Revision 1 re-run:* done with the oracle at `--rev 1`, its default since this revision (log,
  entry "Revision 1"). Results:
  - G1 tight: ≤ 5.1e-10 collocated, ≤ 1.9e-11 staggered.
  - G2: 4.4× on collocated N = 16; steps_acc ≤ steps_plain on every case and window.
  - G7a: `converged=False` for m = 3, 5 and 8 and for the plain march. At m = 8 this comes from the
    guard ("unstable" at step 18, a true positive); otherwise from the plain tail diverging.
  - No false "unstable"; 0 restarts.
  - Window rule re-applied: not met, m stays 5.
- The oracle stays the reference for WO-4's comparison. `--rev 0` reproduces WO-1.

**WO-2 — `Solver::marchState()` and refusals (rev 1).**
- Declaration in `flow_ibm.hpp`, definition in `flow_ibm_diagnostics.hpp` (§5.1, §5.3). The rev-1
  descriptor holds roles Velocity/Carried (P and the face field Carried) and `innerTolerance`
  (§4.1), and no `sdf`, `cP` or `gauged`.
- `tests/kokkos/test_march_state.cpp` checks:
  - the field list and roles for the three §3.1 rows;
  - `innerTolerance` in three configurations: the default (= the PCG rtol); an explicit
    `set_velocity_residual_tolerance(1e-9)` (= 1e-9); tolerance 0 under Chebyshev (= the Chebyshev
    rtol);
  - that each §5.3 refusal throws with its own message.
- *Gate:* G0(a) and G0(c).

**WO-3 — `AndersonCore` in core (`core/include/peclet/core/solver/anderson.hpp`, D13). DONE** in
`../core-anderson`, branch `anderson`: `7212582` (the descriptor), `e300e29` (the core), `2defc9c`
(tests; U1 re-specified). U1a–U7 pass on host and CUDA, and at MPI np = 1, 2, 4. Core is tagged and
published (`PECLET_CORE_TAG` bumped) before WO-4 lands on flow main, per the suite release order.
Until then flow builds against the sibling core.

**WO-3b — core delta for revision 1 (in `../core-anderson`, branch `anderson`; before WO-4).** Two
commits, so that part 2 reverts on its own.
- **Part 1 (required): the guard's scope.**
  - `AndersonState::innerTolerance`: a `double`, default 0. `validate()` requires it finite and ≥ 0.
    Doc: "relative tolerance of the caller's inner solves; sets the Ritz floor; 0 = unknown".
  - `static constexpr double kRitzFloorFactor = 1000.0`, and
    `double ritzFloor() const { return std::max(kNoiseFloor, kRitzFloorFactor * st_.innerTolerance); }`.
  - `bool mixedCol_[kMaxWindow]`, indexed by slot. At column acceptance (step 5, beside the Gram
    copy): `mixedCol_[s] = mixed_`. `dropColumn` and `reset` need no change, since a slot's flag is
    rewritten whenever the slot is reused.
  - Step 7: replace `if (mk_ >= 2 && kNoiseFloor <= residual_ && residual_ <= kRitzHi)` with
    `eligible = mixed_ && mk_ >= 2 && allMixed() && ritzFloor() <= residual_ && residual_ <= kRitzHi`,
    and add `else ritzCount_ = 0;` to the `if (eligible)`.
  - The restart test is unchanged (it keeps `kNoiseFloor`).
  - No packet change, since `mixed_` is rank-consistent.
  - Doc comments of §4.1/§4.3 updated to match.
- **Part 2 (simplification; Q14, default yes): delete the pressure metric.**
  - `AndersonRole` keeps `Velocity` and `Carried`. Delete `Pressure`; keep the integer values of the
    two that remain.
  - Delete the descriptor fields `sdf`, `cP` and `gauged`, and from the core: pass 1 (the
    fluid-sum reduction and its `sumAll`), `countFluid`, `numFluid_`, `pressure_`, `mR_`/`mG_`, the
    mean subtraction in pass 2, and the Pressure branches of `validate()`.
  - Convert or remove every test that uses the Pressure role.
  - *Why:* after rev 1 no consumer uses it. The branch is unreleased, so this is the cheapest moment
    (after the tag it is a breaking change). It removes one collective per step and the sdf
    dependency. It also keeps a metric that rev 1 rejected on evidence from being re-enabled by
    accident.
- *Gate:*
  - U1a–U7 give bit-identical iterates to the pre-delta build on every test that does not trip the
    guard. Neither part changes an iterate: part 1 only changes whether the guard is evaluated;
    part 2 only removes the pressure term, which these tests do not exercise or which is
    converted. Compare γ sequences and final states with the pre-delta build.
  - U4 still reports "unstable" within 3m steps of engagement.
  - U8, U9, U10 pass, on host and CUDA; U7 at np = 1, 2, 4.
  - The flow oracle needs no change: it already implements both parts at `--rev 1`.

**WO-4 — Adapter, bindings, Python driver, packaging (rev 1).**
- `src/anderson_accelerator.hpp`, the explicit instantiations, the bindings (§5.1–5.2),
  `packaging/flow_steady.py` plus the `flow_init.py` import and the CMake lines. The driver is §7
  revision 1: velocity-residual target, budget 2·(num_passes + 3), the early "slow" exit.
- `tests/python/test_march_to_steady.py`, registered as ctest `march_to_steady`, contains:
  - G0(b) at N = 16;
  - a small G1 (§11 N = 12, collocated and staggered, tight, ≤ 1e-8);
  - G7a;
  - **three pure-Python tests of the driver's `certify` on scripted monitor sequences.** (i) Four
    blocks with R > 1 and then a geometric tail inside the budget → "pass". (ii) A clean tail with
    R = 0.99 that fails the bound → "slow" at the first such block. (iii) A sequence that never
    passes → "budget" after exactly 2·(num_passes + 3) blocks.
- *Gate:*
  - G0(b).
  - C++ vs oracle (`--rev 1`) on host, §11 N = 16 collocated: the per-step velocity `residual`
    sequences agree to 1e-6 relative over the first 30 steps (reduction order differs, so bit
    equality is not expected).
  - The new ctest passes.

**WO-5 — Gate campaign.**
- Write `tests/study/steady_acceleration_gates.py` (an instrument, not a ctest).
- Run G1–G8 on host and CUDA, and G4 with the kokkos_mpi tree. Log every number, including
  pressure iterations per step at mixed and plain iterates.
- **D11 is decided here** on the staggered dense bed (Q13): wall ratio at G2 ≥ 1.5 → default
  `accelerate=True`. The collocated bed is reported beside it.
- *Gate:* all of §8.

**WO-6 — Documentation.**
- Add a CLAUDE.md subsection "Steady marches" covering usage, scope, refusals, the memory formula,
  and "checkpoint with get_field/set_field".
- Add a "Measured" table to this note, replacing the §1.1 estimates.
- Draft the register entries for `../docs/decisions/flow.md` (and core.md for WO-3b); the caller
  commits them once approved. Rev 1 changes two entries already in the register, each as a new
  recorded decision that names what changed:
  - "Steady marches are accelerated by type-II Anderson on the full march state…": its quoted
    metric (c_P-weighted, gauge removed) is superseded by D3 rev 1.
  - "Steady state is certified by the unchanged stop instrument on PLAIN steps…": its residual
    becomes the velocity residual, its budget 2·(num_passes + 3) with the early slow exit (D6 rev 1).
- Add a register entry for the guard's scope (D5 rev 1).
- *Gate:* the docs build (`docs.yml`) is clean.

---

## 10. Risks and open questions

Each item is labelled **[fact]** (knowable by a measurement) or **[preference]** (the user's call),
and has a default that work proceeds with.

| # | question | kind | default (proceed with this) | settles it |
|---|---|---|---|---|
| Q1 | Should `march_to_steady` default to `accelerate=True`? | preference | **DECIDED by the user 2026-10-02: pre-registered rule** — `True` iff the dense bed gains ≥ 1.5× in wall time at G2 (D11) | — |
| Q2 | Which dense random bed (φ ≈ 0.6) and resolution is the G1/G3 reference? | fact | the smallest dense-bed configuration in `~/Codes/peclet-study-A1-drag-audit/scripts/`, at its lowest resolution | A1 owner names the file |
| Q3 | Which random array and Δt rule for G6 at Re ≈ 10, 100? | fact | the A1 finite-Re configuration at its lowest resolution with A1's own Δt rule; if none exists, the §11 sphere at Re ≈ 10 only (measured stable), Re ≈ 100 reported as "not run" | A1 owner |
| Q4 | Do the §4.1 safeguard constants cause false restarts/"unstable" on production beds? | fact | **ANSWERED by WO-1:** restarts never (0 in every run); false "unstable" yes, on plain windows (rev 0). Rev 1 fixes the guard's *scope* (D5); no constant was tuned | WO-5's G7c re-checks |
| Q5 | Should A1 use an interim driver on the released wheel 1.2.0? | preference | **no**; A1 adopts on the flow release carrying this (the oracle is host-only and copies) | user (A1 budget vs effort) |
| Q6 | Largest per-GPU grid in A1 production? Runs > 7.5 M cells on a 16 GB card do not fit at m = 5 | fact | such runs pass `accelerate=False` or go to two GPUs; the constructor's error names the bytes | A1 owner |
| Q7 | Should the collocated face field be registered (`uf`, `vf`, `wf` in the field registry) so a collocated-advection checkpoint round-trips exactly? It would also make `redistribute` carry it, fixing CLAUDE.md's open item but changing a rebalanced run's numerics | preference (changes an existing path) | **not done here**; the collocated-advection restart re-seeds u_f (a transient, same fixed point) | user, as its own recorded decision |
| Q8 | Is float history storage acceptable (memory ÷ 2 for the history)? | fact + preference | **double** | an oracle run with float32 history: if steps-to-stop are unchanged on all WO-1 cases, the user decides |
| Q9 | Promote `AndersonCore` to `core` for `peclet.amr`? | preference | **DECIDED by the user 2026-10-02: in `core` from the start** (D13) | — |
| Q10 | Public names `march_to_steady`, `MarchResult`, `diagnostics.anderson_accelerator` and the keyword names of §5.2 | preference | as stated (checked against NAMING §1: verbs, `num_*`, `rtol`, no cell units) | user |
| Q11 | Is the §3.2 instrument library API from now on (`slow_rate=0.997`, `roundoff=1e-11`, `check_every=5`, `num_passes=3` as library defaults)? | preference | yes, the study's constants; the study script switches to `march_to_steady(accelerate=False)` in WO-4 (G0(b) proves equivalence) | user |

| Q12 | Does a mixed iterate cost more inner work than a plain one (velocity sweeps; the oracle cannot count them)? | fact | proceed. Pressure iterations per step are equal (staggered bed 11.47 vs 11.49, collocated 31.2 vs 33.3). Host map time per step on a shared, loaded 48-core box: +3 % collocated bed, +10–28 % staggered bed (two repeats); map-time ratio 2.7–3.2× staggered, 2.2× collocated | G2's wall ratio on the C++ build (WO-5), host and CUDA |
| Q13 | Which dense bed decides D11's 1.5× rule: staggered, collocated, or both? | preference | **staggered decides**: A1's solver is `peclet.flow.Solver` only (`common.make_solver`); the collocated bed is reported beside it. Both clear 1.5× in steps (3.5×, 2.3×) | user |
| Q14 | Delete the core's Pressure role, `sdf`, `cP`, `gauged` and pass 1 (WO-3b part 2)? | preference (a tested core interface) | **yes**, as a separate commit, before the core tag. It has no consumer after rev 1, and removing it after the tag would be a breaking change | user / core owner |
| Q15 | Is `kRitzFloorFactor = 1000` right on other configurations (GPU, MPI, large D)? | fact | 1000. Evidence: false readings only at ≤ 5τ, plus one isolated reading at 140τ; the true positive at ≈ 1e6τ | G7c census over WO-5 (every eligible reading ≤ 1.0005) |
| Q16 | Is the certification budget 2·(num_passes + 3) enough at production resolution (N = 128 beds, CUDA)? | fact | as stated. Measured passes within 5–10 blocks; one run (collocated bed, m = 3) used 12 and resumed once | G2 at WO-5; if a case needs > 12 blocks routinely, report the d/R traces to the architect, do not raise the budget |
| Q17 | The 1,168 sealed single-cell pockets on the A1 bed: their pressure is never converged by either march (§1.2) and `get_p()` returns arbitrary values there. Act on it? | preference (outside this campaign) | **no action here**; note it for the A1 owner. It touches neither velocity nor K. A pressure post-processing user should mask pockets | user |

**User ruling 2026-10-02:** the stated defaults of Q2–Q8, Q10 and Q11 are accepted; Q9 is
decided the other way (D13); Q1 is decided by a pre-registered measurement rule (D11). Q12–Q17 are
new in rev 1 and run on their defaults until the user rules.

**Risks stated plainly:**
- **Staggered §11 gains are modest (1.47× at N = 16, 2.7× at N = 24).** The instrument's fixed
  certification cost (≥ 25 plain steps, typically 25–50 after an Anderson iterate, §7) dominates a
  75-step march. A shorter certificate would mean changing the instrument; that is not proposed.
- **Step counts vary ±40 % with the window** (collocated N = 24: 69 / 48 / 58 for m = 3 / 5 / 8)
  because the certification transient varies. Phase A itself is stable across m (e.g. 40 / 39 / 38
  on collocated N = 16). G2's bars are per case and per window, not on the trend.
- **Finite-Re gains are modest (2.6× at Re ≈ 10, m = 5).** CFL-bound Δt gives a continuum of slow
  modes; windows ≤ 8 cannot deflate a continuum.
- **Anderson can reach states the plain march at that Δt cannot** (§2.4, measured). The Ritz guard
  and the certification-growth fallback cover the cases where the plain march would visibly diverge
  or Anderson actively suppresses growth. A weakly unstable steady state can still be certified, but
  then the unaccelerated instrument would certify it too.
- **Inner-solve cost at mixed iterates** may differ from plain iterates (Q12). Pressure
  iterations do not differ; host map time differed by +3 % to +28 % on a loaded machine. G2's
  wall-time ratio decides.
- **The guard is blind below its floor** (1e-5 at production settings, rev 1). A growing mode
  that Anderson suppresses only once the residual is below the floor is left to the certification:
  growth exit, R ≥ 1 never passes, budget, then resume, then stagnation, then plain. That path
  ends not-converged or diverged, never certified on a growing tail.
- **Monitor consistency under MPI** is the caller's responsibility (documented). A monitor that
  differs across ranks can deadlock the driver, as it would the study's `march()`.

**Transfer to other codes.** To AMR: yes. The core is View-based and grid-agnostic; an amr adapter
supplies leaf u and P and its own refusals. To VoF: no, it is time-accurate. Not designed here.
