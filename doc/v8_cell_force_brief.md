# Architect brief — V8 (collocated variable-ρ) under a cell force: root cause measured, design wanted

Written 2026-09-25 by the orchestrating session. It adds to `suite/docs/HANDOFF_COLLOCATED_VARRHO_CELL_FORCE.md`,
which it summarises here: you do not need to read that file. Worktree:
`/home/frankp/Codes/suite/flow-v8-cellforce` (branch `v8-cellforce` = flow `1fdee46` + the test SKIP removed).
Builds: `build_omp/` (Python module, host-openmp), `build_ktest_omp/` (tests/kokkos).

## 1. The question

What should the collocated variable-density path (rung V8, `SolverColocated` with
`effVarRho() || csfActive()`) do instead, so that:
- it is stable at every dt;
- its steady state does not depend on dt;
- a volumetric cell force is placed at the cell value, per the user rule in §5;
- the balanced-force properties for pressure, gravity and CSF survive.

Everything outside V8 must stay byte-identical. Deliver a design note with work orders and gates (§8).

## 2. Why this needs the architect

V8 is a discretization architecture, not a local bug. The fix has to trade off three things:
- the ABC approximate projection's invisible subspace;
- the Timmermans rotational pressure update;
- the balanced-force (well-balanced hydrostatic / V4 CSF) property that V8 was built for.

The obvious one-line fixes are wrong in ways shown below (§6).

## 3. Current design (anchors in the worktree)

The V8 step is as follows. `buildRhsColoFF` (`src/flow_ibm_project.hpp:596-631`) builds the
momentum RHS with no pressure gradient and no force:

    bb = rs*(rho_c/dt * u^n - rho_c*adv) + bc

1. **Predictor.** The implicit viscous (+ drag diagonal) solve gives u*.
2. **Face field.** `centerToFace(u*)` gives uf*. Then `applyFaceAcceleration` (`:633-653`,
   kernels in `src/collocated_varrho.hpp:70-89`) adds, on every open face:

       a_f = dt*( f_const + ½(fb(i)+fb(i-s)) - w_a (P(i)-P(i-s)) ) / rho_f   (+ CSF, addFaceAccelCsf)
       uf* += a_f

3. **Projection.** `div_ = D(uf*)`, then solve for φ with face coefficient o_f ρ0/ρ_f, then
   `uf -= (ρ0/ρ_f) G φ`, which is exactly divergence-free.
4. **Cell correction** (`applyCellFaceAverageCorrection`, `:655-666`):

       u(i) += ½(ã(i) + ã(i+s)),   ã = a_f - (ρ0/ρ_f) Gφ

   A closed face contributes 0. This is the average of the two faces' total increments.
5. **Pressure** (`projectPressureUpdate`, `:1170ff`). This is the rotational incremental (Timmermans) update:

       P += (ρ0/dt) φ - κ div_,   κ = rotWeight·μ   (varProps: χ·μ_min)

   `div_` is the divergence of uf* *after* the face acceleration was added. Defaults are
   `rotationalP_ = true`, `rotWeight_ = 1`, `incremental_ = true`.

The constant-ρ collocated path works differently. Its predictor carries the force at the cell
value and the cell pressure gradient ½(P(i+s)−P(i−s)) inside the implicit solve (`buildRhs*`,
`:560-594`, `Grid::atVelocity`). The cell correction there is `gpCenterGrad`, the exact transpose
of `centerToFace`. The same rotational update is used. It is validated: order 1.99, cell value to
5e-14.

## 4. Measured (2026-09-25, this worktree, host-openmp)

**Repro.** `tests/python/test_cell_force_placement.py` with the SKIP removed. The collocated
var-ρ runs give placement NaN, TG convergence NaN and drag NaN. The other three grid/density
combinations pass: order 1.99–2.00, drag ≤ 1.5e-12.

**Root cause A: instability.** The rotational term acts on an unfiltered face pressure acceleration.
- **Linear analysis.** Take constant ρ = 1, a periodic box, Stokes. In the V8 step, the P^n
  inside a_f is removed exactly by the φ/dt term: for a mode, φ = −dt·P + (u, f terms). So P
  survives only through −κ·div_, and div_ contains D(−dt G P) = +4 dt S P, with
  S = Σ_a sin²(θ_a/2). For the full checkerboard θ = (π,π,π) every cos(θ_a/2) vanishes, so
  centerToFace and the face→cell average both annihilate it. The mode never reaches u and
  decouples:

      P^{n+1} = −4 κ dt S P^n  →  multiplier −12 μ dt/(ρ h²)   (−8 in 2D)

  The stability bound is therefore μ·dt/(ρh²) < 1/12. This is the explicit-diffusion limit,
  reintroduced into an implicit scheme.
- **Symbol scan.** A von Neumann scan of the full (u, P) step, z-uniform, 64² modes (script
  `scratchpad/symbol.py`, reproduced in the appendix) gives these spectral radii:

  | configuration | spectral radius | dt range |
  |---|---|---|
  | V8, κ = μ | 8μdt at (π,π) | dt ≥ 0.125 |
  | V8, κ = 0 | ≤ 1 | every dt up to 32.4 |
  | constant-ρ collocated, κ = μ or 0 | = 1 | every dt |

  The constant-ρ collocated radius of exactly 1 is the known neutral checkerboard family (see
  `doc/collocated_invisible_subspace.md`). There the rotational term acts on
  D·centerToFace·(viscously filtered u*), which cannot see the checkerboard.
- **In the code.** The growing mode, found by 3D FFT of P after 80 steps of the TG case, is
  θ = (π,π,π). Its per-step multiplier matches the prediction to every printed digit:

  | μdt | measured | predicted |
  |---|---|---|
  | 1 | **−12.0000** | −12 |
  | 0.1 | **−1.2000** | −1.2 |

  The f_x(x) case grows through the mode (0.875π, π, π) at −11.8478. The prediction is
  4(sin²(0.4375π) + 2) = 11.8478. The growth depends on μ·dt only: (dt 1, μ 0.1) gives
  1.172 per step and (dt 0.1, μ 1) gives 1.180.
- **Ablation.** `diagnostics.set_rotational_pressure(False)` makes V8 stable at dt = 1 and at
  dt = 32.4. The f_x(x) case then settles to u = 4e-19 with P balancing the force.
- **The handoff's leading suspicion is refuted.** It blamed the face-average wide stencil in the
  force's own direction. The force plays no part in the growth.

  The apparent "along-axis vs across-axis" selectivity is only seeding. For f_x(y) the face
  divergence is *bitwise* zero, so P stays 0.000 and there is nothing to amplify. Every other
  force, including the analytically solenoidal Taylor–Green, seeds the mode with round-off at
  about 1e-19. It then reaches 1e42–1e74 within 60–100 steps.

  The invisible subspace is involved, but only as the reason nothing damps the mode.
- **Prior sighting.** The V8 gate `tests/kokkos/test_vof_collocated.cpp` T2 (`:376-398`, the
  droplet at ratio 1000) already recorded "unstable, ~4x per step" at μdt/(ρ_min h²) = 0.45. It
  attributed this to the explicit face acceleration and used it to **rate V8 to density ratio
  ~100**. The prediction at 0.45 is between 8× and 12× that number, i.e. 3.6–5.4 per step.
  Ablation of T2 with the rotational term off. This was a temporary `setRotationalPressure(false)`
  in `makeDroplet`, since reverted; the logs are `v8gate_rot.log` and `v8gate_norot.log` in the
  worktree root.

  | ratio-1000 droplet | rotational on | rotational off |
  |---|---|---|
  | base μ of the sweep | THREW (WY CFL 0.64) | face **4.3e-18** |
  | μ = 0.1 | THREW | face **5.9e-18** |
  | μ = 0 and 0.01 | about 7e-18 | unchanged |

  The T1 hydrostatic column is also unchanged. So the ratio-~100 rating comes from cause A, not
  from the face-acceleration construction itself.

**Root cause B: dt-dependent steady state.** Measured with the rotational term off. Forces and
∇P are applied after the implicit viscous solve.
- **Mechanism.** At a steady state φ = 0, and the balance becomes μ L u* = A(f_f − G P)/…, with
  u = (I + dt μ L_c/ρ) u*. The steady velocity therefore carries a factor (1 + dt μ Λ). This is an
  O(dt) splitting error that does not vanish at steady state. At the test's dt = 10/(μλ) the
  factor is about 11.
- **Numbers with the rotational term off (N = 16):**
  - Placement misfit is 9.46 against the cell value and 9.53 against the face mean. The
    prediction (1+dtμΛ)·cos²(k/2) − 1 = 9.53.
  - TG error is 9.2, 9.8 and 10.6 at N = 16/32/64. The order is −0.1: no convergence.
  - Drag (dt = 1) is 2.48 against round-off on the other paths.
- **Placement is also wrong.** Even without the factor, the cell sees A·C2F f = ¼(f(i−1) + 2f(i) + f(i+1)) along the
  force's own axis. That is neither the cell value, which the user rule demands, nor the face mean.

## 5. Settled — do not relitigate

- **ABC, not Rhie–Chow.** The collocated coupling is the Almgren–Bell–Colella approximate
  projection. Never Rhie–Chow, and not as an "upgrade" either (`docs/decisions/flow.md:494`,
  `:2053`, suite-wide).
- **The Basilisk face-acceleration form was rejected** as the collocated projection's form
  (`docs/decisions/suite-wide.md:300-312`, 2026-09-03). The quote:

  > flow uses the incremental predictor with the cell pressure gradient = the EXACT TRANSPOSE of
  > the centre→face constraint (gauge-exact `gpCenterGrad`), used in both the predictor and the
  > correction; the Basilisk face-acceleration form was considered in flow's collocated
  > discussions and REJECTED — do not reintroduce it.

  The reason: "adjoint pairing = stability, invisible subspaces, rotational-update instability".
  V8 (WO-T) nevertheless implemented Basilisk `centered.h`. Your note must say explicitly how the
  chosen design stands against this entry. If it keeps a face-acceleration form, it needs a
  **new recorded decision** that says what changed.
- **USER rule 2026-09-25** (`docs/decisions/flow.md:3162`).
  - Volumetric forces go where the velocity is defined; on the collocated grid that is the
    **cell value**. They are force_x/y/z, Boussinesq, ρg, and drag β in both the RHS target and
    the diagonal (`Grid::atVelocity`, `src/grid_layout.hpp:43-51,88-92`).
  - Surface forces (pressure, viscous stress, surface tension) are finite-volume face integrals
    of the velocity control volume. They are never a cell-centred divergence that is then
    interpolated.
  - The placement follows the grid, never the density model.
- **Balanced-force CSF on V8** (the V4 pairing) must keep annihilating a constant-κ surface
  tension (T2 ratio 1 and ratio 1000).
- **The hydrostatic column must stay balanced** at ratio 1000. In T1 the FACE field must stay at
  machine zero. The cell field is reported, per T1b.
- **Never change numerics while restructuring.** The following must stay byte-identical:
  - the staggered paths (both ρ models, including porous CFD-DEM);
  - collocated constant ρ (all schemes);
  - the `tests/regression/state_hash.py` cases outside V8.
- **flow is the method reference.** Read `doc/collocated_invisible_subspace.md` (the neutral
  checkerboard family, the wall-blend and the filtered rotational update) and
  `doc/variable_density_projection.md` (the three-way ρ_f consistency) before designing.
- Out of scope: the staggered paths, CFD-DEM coupling, gamma calibration, dem.

## 6. Genuinely open — what to decide

1. **Stability.** What replaces the rotational term's action on V8? Some candidates, from a
   quick look only:
   - **κ = 0 on V8.** The symbol analysis says this is stable. It gives up the rotational
     accuracy gain (the pressure boundary layer at walls).
   - **Rotational term on D(centerToFace u*) only, i.e. before a_f.** A pre-check suggests this
     is wrong. At a steady state D(C2F u*) = −D a_f ≠ 0 whenever the force has a gradient part,
     so P would drift linearly and never reach a steady state.
   - **Put the pressure (and forces) inside the implicit solve,** so the viscous filter applies
     as it does on the other paths.
   - **The filtered rotational update** (`setRotationalFilter`).
   - **Other options.**

   Decide with the symbol analysis, including variable ρ at ratio 1000 across an interface.
2. **Placement and splitting.** Which terms stay face accelerations, if any? Candidates are
   pressure, CSF and hydrostatic ρg. Which become cell sources inside the implicit momentum solve?
   How does the steady state become dt-independent?
3. **What that does to V8's balanced-force properties.** Cover the hydrostatic column at ratio
   1000, the constant-κ droplet at ratio 1 and 1000, and the cell-level O(1) imbalance at
   interface cells that motivated V8 (`collocated_varrho.hpp:19-24`). Address the conflict with
   the §5 user rule directly. Gravity ρg is both a volumetric force and a quantity whose balance
   against ∇P defines hydrostatics.
4. **Whether the density-ratio-~100 rating of V8 lifts** once cause A is removed.

## 7. Verification (how the answer will be judged)

- **`tests/python/test_cell_force_placement.py` with the SKIP removed.** On collocated var-ρ:
  placement at the cell value (< 1e-6), TG order about 2, and drag at round-off (< 1e-8).
  Run with `OMP_NUM_THREADS=4 PYTHONPATH=build_omp python tests/python/test_cell_force_placement.py`.
- **Stability.** Run at every dt 0.1–100 and μ·dt/(ρ_min h²) up to at least 10. Add a new gate that
  runs the checkerboard growth probe: 3D FFT, and a multiplier of at most 1 on (π,π,π).
- **`tests/kokkos/test_vof_collocated.cpp` T1–T5** (`build_ktest_omp/test_vof_collocated`),
  including T2's ratio-1000 μ sweep.
- **`tests/kokkos_mpi/test_vof_collocated_mpi.cpp`**, and the gates in
  `suite/docs/wo_vof_mpi_parity_gates.md`. np = 1 stays exactly decomposition-independent.
- **The full flow battery, 167/167.** Use `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` and
  `--bind-to none`, and run np8 last.
- **state_hash byte-identity outside V8.**

## 8. Deliverable

Write **one** design note at `/home/frankp/Codes/suite/flow-v8-cellforce/doc/collocated_varrho_forces.md`.
It must contain:
- (a) the root cause, confirming or correcting §4 A and B with your own analysis;
- (b) the chosen design, with the equations of the step;
- (c) the rejected alternatives and why;
- (d) what changes for the balance of pressure, gravity and surface tension;
- (e) the work orders for `opus-implementer`, each with its gate;
- (f) the byte-identity argument for everything outside V8;
- (g) a register entry, drafted in the `docs/decisions/flow.md` format.

Also:
- Write no production code and do not commit; the orchestrator commits the note.
- Analysis scripts may go in `/tmp/claude-1003/-home-frankp-Codes-suite/d534b3c2-1bb3-4eb0-9125-33b85b8791a1/scratchpad/`.
- If the choice turns on a user value judgement, say so in the note and give a default rather
  than stopping.

## Appendix — symbol model used for §4 (constant ρ = 1, h = 1, per-axis c = cos(θ/2), g = 2i sin(θ/2))

    V8:    u* = m u,  m = 1/(1+4 dt μ S)
           div = Σ g c u* + 4 dt S P                 (f = 0)
           φ = −div/(4S)
           u' = u* + c(−dt g P − g φ)
           P' = P + φ/dt − κ div
    CONST: u* = m(u − dt g c P)
           div = Σ g c u*
           φ = −div/(4S)
           u' = u* − c g φ
           P' = P + φ/dt − κ div
