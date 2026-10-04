# Cut-cell scalar transport — independent review (branch `scalar-ibm`)

*2026-10-04, reviewer agent. Reviewed against `doc/scalar_ibm_design.md` with Amendments A1–A3 and
the brief `doc/scalar_ibm_review_brief.md`. The ranges were flow `39680a4..d2118c5` (src, tests,
CMakeLists) and core `6859d4e..HEAD`. I read the design (§1–§6, A1, A2, A3) and the log rulings
D-WO2 to D-WO9. I read in full the driver (`flow_ibm_scalars_cutcell.hpp`), the operator kernels,
`scalar_krylov.hpp`, the ScalarMG build, V-cycle, smoothers and fills, the geometry kernels, core's
`probe_flux.hpp`, and the small edits in `flow_ibm_project/scalars/mpi`. I also read the
`scalar-units` diff of `flow_ibm_closures.hpp`. Two probes were run on `build_dev` (host OpenMP, 4
threads); their scripts are in the session scratchpad and are reproduced inline where they matter.*

**Path.** Both `doc/scalar_ibm_review.md` and the fallback `doc/scalar_ibm_code_review.md` match
`.gitignore:71` (`doc/*_review*.md`). This file is therefore committed with `git add -f` on its
single named path.

## Verdict

**Ready to land behind the opt-in once findings 1 and 2 are fixed.** Both fixes are small guards
and change no numerics. Finding 3 is a one-function test change and should come with them. Finding
4 needs a decision before steady solves go into production at scale. Nothing found is a defect in
the discretisation:
- conservation, the probe closure, A1, A2 and the conjugate ψ form are all sound as implemented;
- the parallel contract holds by construction wherever I could trace it.

## Findings, most severe first

### 1. Collocated: the scalar advects with `uf_` without checking that it was ever built — **should fix** — VERIFIED (probe)

- **Where.** `flow_ibm_scalars_cutcell.hpp:651` reads `uf_/vf_/wf_` unconditionally. The only
  "moving fluid" guards are at `:758` (no cut-cell projection) and the open-face check after it.
- **The gap.** `set_solid` zeroes `uf_` and sets `faceFieldValid_ = false`
  (`flow_ibm_geometry.hpp:86-90`). A raw `set_field('u', …)` does not seed the face field. So on
  `SolverColocated`, `advance_scalars()` or `solve_scalar_steady()` without an intervening `step()`
  sees max|uf| = 0. It sets `st.advecting = false` and silently returns the pure-diffusion solution.
- **Probe.** `SolverColocated` 16³, 'gauge-exact', sphere R = 0.2, `set_field('u', 0.5)`, a
  cut-cell scalar with a Dirichlet wall, `solve_scalar_steady`. Result: max|u_cell| = 3.125
  (physical), census `max_cell_peclet = 0`, `num_flux_faces = 0`, no warning. The same case seeded
  through `set_velocity` advects (Pe_h 0.5, 3968 flux faces).
- **Precedent.** This is the exact trap flow already closed for VoF: `advect_vof` refuses on
  `!faceFieldValid_` (`flow_ibm_vof.hpp:1328`, "ISSUES sweep item 5"). The pre-existing open item
  "`uf_` is not carried through `redistribute`" reaches the same path: a steady solve after a
  rebalance reads a stale field.
- **Minimal fix.** In `scalarCutAdvection`, when `Grid::collocated && !faceFieldValid_` and the cell
  velocity is non-zero, raise with `advect_vof`'s remedy text.
- **Related, both grids (documentation or census, not a blocker).** Nothing checks that the
  advecting flux is discretely divergence-free. A user-written staggered `u`, or a `set_velocity`
  seed on collocated, which is a cell→face average that is not projected, is accepted silently.
  The cell divergence max_i |Σ_f F_out| / max Out costs one more entry in the existing MAX
  reduction and would expose it.

### 2. Steady advection with reversed flow on an outflow face is silent and can get the wrong compatibility projection — **should fix** — not run

- **Where.**
  - `scalar_cutcell_operator.hpp:979`: `AC += Fout` with `Fout < 0` on backflow.
  - `:712` / `flow_ibm_scalars_cutcell.hpp:1046`: `omegaOpen` is clamped silently.
  - `flow_ibm_scalars_cutcell.hpp:1116-1125`: `singular` looks only at `cw` and Dirichlet faces.
- **What A2 asked for.** A2 required that an implicit-backflow diagonal be reported. The code
  clamps the coarse mass and says nothing, and the log lists backflow as "ungated".
- **Two consequences.**
  - **(a) Inflow plus an outflow with partial backflow** (a wake reaching the outlet) stays
    non-singular. The backflow cells lose column dominance, so S_adv is no longer an M-matrix
    there (see 3). This is mostly a question of whether the zero-gradient backflow BC is
    well-posed (§6.5), but it is currently invisible.
  - **(b) Open faces but no inflow face.** For example, both ends 'outflow' with a body-force
    drive, or one outflow face with walls elsewhere. These necessarily backflow. The steady
    operator is then singular: zero-gradient everywhere, so constants are in the right null
    space. But its left null vector is not 1: the backflow columns sum to −|F|, the outflow
    columns to +F. The code nonetheless flags `singular` and projects b on the uniform mean. That
    is not the compatibility projection, so BiCGStab cannot converge. The result is maxit 200, one
    stderr line, and a meaningless iterate.
- **Minimal fix.** Count the open-face rows with `Fout < 0` (one more census entry; for steady,
  the count over the implicit open-face rows). Warn once, or refuse steady mode when the count is
  above 0. Also treat "any open face" as non-singular, or refuse case (b) outright.

### 3. Brief question 2 (A2's M-matrix property): it is structural and holds at any pressure tolerance and Pe_h. The test checks the wrong invariant — **should fix (verification)** — derivation, not run

- **Why it holds.**
  - On every level the advective couplings enter as one number per face: Q⁺/Q⁻ on the upwind
    diagonal and the same number, negated, as the downwind off-diagonal.
  - At level 0, `max(F_out,0)` of one cell is exactly `−min(F_out,0)` of its neighbour.
  - At the coarse levels `assembleBands` uses `qm(i)+qp(i+e)` on the diagonal and `−qp(i+e)`,
    `−qm(i)` in the neighbouring rows.
  - So the diffusion and advection parts have **zero column sums up to rounding, independently of
    the field's divergence**. The diagonal additions (mass, W, Dirichlet folds, ω_open) only add
    dominance.
  - S_adv is therefore a column-diagonally dominant Z-matrix: an M-matrix by columns, and RB-GS is
    a convergent regular splitting.
  - The left null vector of the singular closure problem is exactly 1 on every level, so the
    per-level mean removal is the right compatibility projection even with a loose pressure.
  - Divergence only perturbs the **row** sums, and with them the right null vector.
- **The u3 reading.** The −5.8e-13 "margin" in u3 is a row margin. It reads the residual of the
  pressure solve, not a defect.
- **The only regime that breaks the property is finding 2** (backflow).
- **Where the test goes wrong.** `tests/kokkos/test_scalar_mg.cpp:557-558` (`mMatrixDefects`)
  asserts **row** dominance to 1e-13·AC. That is why the C4/u3 harness had to run the pressure PCG
  at 1e-14.
- **Fix.** Assert column sums ≥ −1e-13·max AC per level, and positive off-diagonals = 0. That
  tests the invariant the argument rests on, at the default pressure tolerance.

### 4. D-WO4-1 hides a scalability limit, not a correctness defect — **needs a decision**

- **Where.** `scalar_mg.hpp:972-982`: the bottom is `kBottom = 16` RB-GS sweeps on whatever level
  the table stops at. There is no agglomerated or exact coarsest solve and no telescoping (§13 Q3).
- **What gating on factor-of-two boxes hides.** An axis that cannot halve stops the table: odd or
  2-poor global sizes, or under MPI any rank's odd block or origin.
- **Measured.** n = 77 → 78 iterations against 6 at n = 80 (CLAUDE.md trap). Steady solves always
  use the full table, so the closure/dispersion use case (paper A4) is the one exposed.
  Production grids sized by the physics (e.g. 3·2^k) and ORB blocks at np ≫ 1 will hit it.
- **My recommendation.** Before steady production at scale, either:
  - give ScalarMG the pressure's 'auto' answer: agglomerate the coarsest level and solve it
    exactly once it exceeds a few cells per axis; or
  - at minimum, warn when the coarsest level is larger than `set_pressure_bottom_extent`'s 4 per
    axis.

  Transient runs below κ_A = 25 use level 0 only and are unaffected.

### 5. The singular-solve gauge shift is not reflected in the reported residual — **note** — derivation

- **Where.** `flow_ibm_scalars_cutcell.hpp:1314` (and the two-field twin) shifts c by δ after the
  solve. `st.residual`/`converged` (`:1454`) are taken before the shift.
- **Effect.** The true residual grows by δ·(row sum) = δ·div_i. With the pressure at its default
  tolerance this is negligible.
- **Scenario.** A closure solve on a flow projected at rtol 1e-6 to 1e-8 (the bubble-column ruling
  uses 1e-8). θ then satisfies Aθ = b + δ·div, while the census reports 1e-10. This is the same
  order as the advection error a non-solenoidal field causes anyway, as §5.1 says.
- **Fix.** Recompute the true residual after the shift, or report it.

### 6. Rank-consistency of the small flags at an open face rests on flow internals — **note** (fragility) — gated by G9c np = 2/4

- **How it works today.** The captured outflow plane's transverse ghost rows
  (`flow_ibm_scalars_cutcell.hpp:694`) are read **before** any post-projection exchange. They equal
  the neighbour's values only because:
  - `bcCorrectOutflow` runs over the full transverse extent with exchanged φ ghosts;
  - the pre-divergence `fillVelGhosts` applies the outflow BC after its exchange.
- **The risk.** A change to either flow kernel's range would let a ghost-layer-1 small flag differ
  from the owner's. One face would then be implicit on one rank and explicit on the other: a
  conservation leak at the rank boundary. The per-rank budget identity would not show it.
- **Cheap hardening.** One G = 2 `fillGhosts(st.small)` per advance instead of computing ghost
  layer 1 locally (`:808`). That makes the agreement independent of how the ghost flux was formed.

### 7. Smaller notes

- **Koren above bulk Courant ½ (D-WO5-1).** Koren is the `add_scalar` default and overshoots there
  with no warning: legacy-identical, but the census already holds `bulk_courant`. A one-time
  warning for koren with C_bulk > ½ is free.
- **Probe normal (D-WO2-2).** The probe direction is the **snapped** area vector. On corner slivers
  it can be O(1) off the PL normal: the anisotropic R1b mechanism in the log. The held option,
  probing along the unsnapped PL normal while keeping the snapped area for the flux, was never
  applied. The impact is small (tiny facets).
- **Warnings latch.** `scalarCutWarnings` latches once per scalar (`:1478`) and is never reset by a
  geometry rebuild or a later switch to conjugate. A second `set_solid`'s resolution warnings, and
  the solid R2 warning, are suppressed.
- **Per-advance device allocations.**
  - `WallTable` (`:917`), the `MaterialTable`, and one `gv` per Dirichlet face (`:974`) are
    allocated every advance.
  - The census adds about 15 reductions with a host sync per advance.
  - On CUDA each View free synchronises.
  - The measured GPU ratio (0.96–0.98) says this is not dominant today. Hoisting the tables would
    remove it.
- **Velocity ghost fills on the live flow state.**
  - A run with only cut-cell scalars still executes `advanceScalars`' plain `fillGhosts(Uf/Vf/Wf)`.
  - `solve_scalar_steady` does the same to the flow's live velocity (`:727`).
  - That is legacy bug A's footprint: it wraps the high-side open plane. The cut-cell path itself
    is immune because it uses the captured flux.
  - Whatever `uf-outlet-diag` decides applies to cut-cell runs unchanged.

## The eight areas of the brief

1. **Gate restatements.**
   - D-WO2-1/2, D-WO4-2/3, D-WO5c-1/D-WO6-3, D-WO6-1, D-WO7-2/5, D-WO8-2, D-WO9-1 and D-WO9-3 each
     fix a badly posed gate. None hides a defect:
     - conservation is structural (one number per face);
     - MPI parity at rtol 1e-13 is the meaningful comparison;
     - Koren noise at 1e-9 is reduction-order noise.
   - D-WO4-1 hides a real limit: finding 4.
   - D-WO5-1 is right, but it is worth a warning (7).
2. **A2.** Sound; see 3. The exception is backflow (2).
3. **Small-cell split.**
   - Flags are computed from bitwise-identical ghost data, and `thr` comes from a global MAX, so
     classification agrees across ranks (caveat 6).
   - Δt changes and the steady/transient switch rebuild everything per call; I found no stale state
     (`st.small` is only read when `advecting`).
   - Positivity: explicit non-small cut cells satisfy Δt·Out ≤ thr·κV with thr ≤ 1 when the bulk is
     stable.
   - The probe overlay makes the implicit matrix a non-Z-matrix, so near Dirichlet walls the
     cut-cell centre values can legitimately go slightly below g. That is expected for a
     centre-value representation, but users of reactive kinetics should know.
4. **Unknown sets.** Consistent:
   - the facet coefficients zero `cw/rw/cc` on non-unknown cells;
   - V1 keeps probe weights > 1e-12 on unknowns;
   - the band guards and `faceFluxes` drop every coupling to a non-unknown, symmetrically;
   - the coarse pins are `restrictMax`, W is gathered from unknown children only, and pinned
     coarse rows are identity rows with W = 0;
   - the flags are exchanged and cleared beyond non-periodic faces.

   Probe on `num_guarded_flux_faces` (fluxes the guard drops): 0 at aperture order 1 and 2 on the
   G9b geometry. The pressure's MS openness is ≤ the scalar's PL aperture by construction (same fan,
   plus a centre gate).
5. **Units.** The cut-cell path converts at advance time from stored physical values. The legacy
   path re-derives `sc.D` in `refreshUnitDerived`. Both read `add_scalar(D)` as D·tRef/hRef². The
   R_c, k, q, S and G factors match §1.2. The closure refresh (LinearMix 3 params; Boussinesq ρ₀,
   with g kept physical; Arrhenius μ_ref) is dimensionally right.
6. **Conjugate.**
   - ψ storage C_sK κ_s ψ_s = C_s κ_s c_s, G_c = 1/(s_f/Λ_f + R_c + s_s/Λ_s), and
     `get_scalar_solid` = K ψ_s.
   - The fluid row carries +cc(u_pf − u_ps) and the solid row its negative: conservative.
   - The mean-gradient shift matches the corrected §1.5 (b_f += cc(GxS − GxF)).
   - In the surrogate, SAC_f, SAC_s ≥ W, so det ≥ 0; the fallback point update is only reachable in
     degenerate steady isolated pairs.
   - Pinned fluid coarse cells keep W = 0, because `c.W` is zeroed before `assembleCoarse`.
7. **Open-face capture (staggered).** Free of the legacy stale-`cOld` pattern: `cOld` is copied
   after the zeroing and its ghosts are filled before the explicit fluxes. It is also free of the
   inflow-ghost pattern: the inflow is F_in·g directly. Caveats: 6, and the inherited plain fill
   (7).
8. **GPU path.**
   - The resident BiCGStab is statement for statement the host pass, including every breakdown
     exit and the `it` count.
   - The slots and the one-thread kernels are ordered on one execution-space instance, and the
     packet `deep_copy` fences.
   - `fillShell` reads only wrapped **inner** cells, so it has no race.
   - The wrapped reads in the smoother are enabled only when every inner dimension is even, which
     prevents same-colour neighbours across the seam.
   - The wall-reuse cache is keyed on bitwise `cw/cc` content plus the identity of the geometry's
     unknown view, the level count and the two-phase mode. A new geometry builds a new ScalarMG, so
     its invalidation is complete for the inputs W depends on.

## Not examined

- The internals of core's `cut_cell_geometry.hpp` (tet fractions, two-facet grouping) beyond the
  interface, and the core tests.
- `test_units.cpp`, `flow_ibm_phase_change.hpp`'s units diff in detail, and the bindings beyond the
  solid-field docstrings.
- The CUDA build and G11.
- The G6/G13 gate code.
- The 2-D study scripts.
- The `bench` instrument.

I did not rerun the battery or G12.
