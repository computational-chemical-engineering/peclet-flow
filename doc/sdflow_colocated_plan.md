# sdflow collocated grid: design & plan

Goal: add a **cell-centered (collocated) velocity** variant of the sdflow solver *alongside* the existing
staggered MAC solver, sharing as much code as possible. The pressure coupling uses the
**MAC / approximate projection** of Almgren–Bell–Colella (the scheme Basilisk uses — **not** Rhie–Chow):
cell velocities are averaged to faces, the *face* field is made divergence-free by the **already-implemented**
cut-cell projection, and the cell velocities are corrected with the central-difference pressure gradient.
The **rotational (incremental) pressure** update and the **SDF-based IBM** are reused unchanged; only the
**grid positions** of the velocity unknowns differ.

> **STATUS (2026-06-22): phases 1–4 done.** Phase 1 — `SdflowSolver<GridLayout>` extracted, `Staggered`
> bit-identical (regression +0.00%, 18/18 MPI ctests). Phase 2 — `Colocated` policy + cell-centered
> advection (`colocated_advection.hpp`); `sdflow.SolverColocated` exposed; Poiseuille matches staggered to
> machine zero. Phase 3 — approximate (MAC) projection (`mac_approx_projection.hpp`): Taylor–Green vortex
> validated (face velocities divergence-free to ~1e-15; L2 error 11.7× down 32→64; energy decay tracks
> analytic). Phase 4 — cut-cell IBM: periodic sphere-packing Stokes permeability incompressible (face div
> ~1e-10), exact no-slip, grid-converges and approaches the (Z&H-validated) staggered solver monotonically
> (2.61%→1.12%, N=32→64). **§4 decision: Option A passes the gate; B / openness-aware cell gradient not
> needed for converged integral quantities.** Phase 5a — collocated domain BCs (no-slip walls + Dirichlet
> lid): cell-centered reflection ghosts (every component reflects about the boundary face; no fold),
> explicit-reflection diffusion smoother, Neumann phi wall ghost. Lid-driven cavity Re=100 N=128 vs Ghia:
> u_rms 0.0071 / v_rms 0.0075 (< 0.02), face div 1.2e-15 — matches the staggered solver. Remaining:
> **phase 5b** (inflow/outflow: developing channel + backward-facing step — needs the collocated outflow
> velocity ghost + mass-conserving outflow face correction, on top of the shared α/β openness split) and
> **phase 5c** (collocated multi-rank: cell-velocity halo already wired via fillGhosts→exchange; needs a
> kokkos_mpi test).

## 0. Where we are (grounded in the code)

sdflow is already organized as **grid-agnostic free-function operator headers** + a **thin orchestrator
class** (`sdflow::SdflowIbm`, `src/sdflow_ibm.hpp`, 598 lines). Most of the machinery is already
face-based and therefore reusable by a collocated solver with no change.

**Already grid-agnostic (reused verbatim):**
- `src/mac_cutcell.hpp` — face openness `ox/oy/oz` sampled at face centers (`buildOpenness`, line 83:
  `lx-0.5, ly, lz`). Where the *velocity* sample sits does not move the faces, so this is **identical**.
- `src/mac_pressure.hpp` — `buildCutcellOp`, `applyCutcellOp`, `cutcellSmoothColor`, `divergOpen`,
  `projectCorrect` all act on **faces** (the pressure node is the cell center in both schemes).
- `src/mac_cutcell_mg.hpp` (`CutcellMG`) — the entire rotational pressure solve (V-cycle / MG-PCG /
  Chebyshev, MPI-folded) is reused unchanged. This is the "same rotational method".
- `src/cut_cell_ibm.hpp` + the diffusion stencil build in `src/mac_ibm.hpp` — the Robust-Scaled IBM
  overlay is **already parameterized by a velocity offset** `Off3 off` (`sdflow_ibm.hpp:191`).

**Staggered-specific (needs a collocated counterpart):**
1. **Component offsets** — `offs[3]` (`sdflow_ibm.hpp:191`, `:424`). Staggered passes
   `{-0.5,0,0},{0,-0.5,0},{0,0,-0.5}`; collocated passes `{0,0,0}` for all three.
2. **Advection** — `src/staggered_advection.hpp` `adv_vel()` (line 43) interpolates advecting velocities
   onto the faces of *staggered* control volumes. The one piece of real math that must be rewritten for a
   cell-centered control volume.
3. **Velocity ↔ face coupling in projection** — staggered stores the face velocity *as* `u(i)`, so
   `divergOpen`/`projectCorrect` read/write it directly. Collocated must **average** cell → face, project
   the face field, then **correct centers** separately.
4. **Velocity ghost / domain-BC fills** (`src/mac_bc.hpp`, `fillVelGhosts`) — staggered-face conventions;
   collocated needs cell-centered reflection ghosts.

## 1. The collocated pressure coupling (approximate / MAC projection, à la Basilisk)

Per step, after the cell-centered predictor `u*` (existing IBM implicit diffusion + collocated advection):

1. **Average to faces** (no pressure term): `u_f* = ½(u_c(L) + u_c(R))`, similarly `v_f*`, `w_f*`. This is
   a genuine MAC field.
2. **Solve** `A·phi = div_open(u_f*)` with the **existing `CutcellMG`** — identical operator, identical
   openness, fed the reconstructed face field.
3. **Correct the faces**: `u_f = u_f* − grad_f(phi)` — the existing `projectCorrect`. The *face* field is
   now discretely divergence-free (the approximate-projection guarantee). These `u_f` become the
   **advecting** velocities for the next step's advection.
4. **Correct the centers**: `u_c −= grad_c(phi)`, with the central-difference cell gradient
   `grad_c phi|_x = ½(phi(i+1) − phi(i−1))` (= the average of the two adjacent face gradients).
5. **Rotational pressure update**: `P += (rho/dt)·phi − mu·div(u_f*)` — unchanged
   (`sdflow_ibm.hpp:533`).

The face field is *exactly* divergence-free; the cell field is only approximately so — this is precisely
the Almgren–Bell–Colella approximate projection. No Rhie–Chow term, hence no `D_f` interpolation through
the cut-cell `D_rescale` diagonal (which was the main numerical risk of the Rhie–Chow route — now gone).

## 2. Architecture — policy-templated orchestrator (decided)

Three layers; the existing operator headers stay as free functions:

- **Operator headers (unchanged):** `mac_cutcell.hpp`, `mac_pressure.hpp`, `mac_cutcell_mg.hpp`,
  `mac_ibm.hpp`, `cut_cell_ibm.hpp`, `mac_bc.hpp`, `mac_velocity_mg.hpp`.
- **New headers:**
  - `src/colocated_advection.hpp` — cell-centered Koren/TVD (Godunov-style) advection with **face-normal
    advecting velocities** (the projected `u_f` from the previous step).
  - `src/mac_approx_projection.hpp` — center→face averaging (`centerToFace`), the cut-face reconstruction
    (§4), and the central-difference cell correction (`projectCorrectCenter`).
- **`GridLayout` policy** — two small traits, `Staggered` / `Colocated`, supplying: the component offsets,
  the advection call, the face-averaging + reconstruction call, and the correction call.
- **One orchestrator** `SdflowSolver<GridLayout>` — produced by refactoring today's `SdflowIbm`. The step
  loop, field allocation, ghost handling, implicit diffusion, MG bridge (g=2↔g=1), and MPI are **shared**;
  the ~4 grid-specific spots dispatch through the policy. `SdflowIbm` becomes
  `using SdflowIbm = SdflowSolver<Staggered>` (bit-identical); collocated is `SdflowSolver<Colocated>`.

Rationale: maximum reuse, branch-free hot path, future capabilities written once. Cost: a one-time
templating refactor of the orchestrator, guarded bit-identical by the regression suite + ctests.

The collocated solver needs one extra persistent field set vs staggered: the **face-velocity MAC field**
`uf/vf/wf` (transient within a step, but kept across steps to serve as the next step's advecting
velocity — the divergence-free face field is the natural advecting field, exactly as in Basilisk).

## 3. How the IBM influences face velocities near a solid

Two **separate** mechanisms, mapping onto the two physical constraints:

1. **Momentum (no-slip) → cell-center velocities.** The Robust-Scaled cut-cell overlay (offset `{0,0,0}`)
   drives the *cell-center* velocities toward the wall value in the implicit diffusion solve; solid-interior
   cells are masked to 0. So the two cell values feeding a face average already "know" the wall.
2. **Continuity (no-flux) → faces, via openness.** The face openness `o_f` (`buildOpenness`,
   `mac_cutcell.hpp:74`) is the **fluid area fraction** of the face. The projection only ever sees
   `o_f · u_f`: `divergOpen` weights the flux (`mac_pressure.hpp:51`), `buildCutcellOp` weights the Poisson
   stencil identically. Hence a fully-solid face (`o_f=0`) carries no flux even if its averaged `u_f*` is
   garbage; a cut face carries only its open-area fraction.

So the IBM's effect on faces is **indirect and consistent**: momentum forces the cell velocities to no-slip;
continuity area-weights each face. No special wall-enforcing face interpolation is required — the openness
*is* the area weight, exactly as in the staggered solver.

**Consistency:** the momentum overlay (no-slip at cell center) and the continuity openness (no-flux on face)
reference slightly different geometric loci, but both sample the **same SDF** with the same `ccSampleExt`,
so they agree under refinement.

## 4. The one open numerical decision: the cut-face velocity value

Plain midpoint averaging samples `u_f*` at the geometric face center, which may sit in the *solid* part of a
cut face, so it can mis-estimate the flux through the *wetted* (open) sub-area. Options, increasing fidelity:

| opt | reconstruction | order | code | verdict |
|---|---|---|---|---|
| **A** | plain area-weighted average `u_f*=½(u_L+u_R)`, flux `o_f·u_f*` | 1st–2nd | reuses everything | **start here** (Basilisk baseline) |
| **B** | evaluate fluid-side velocity at the **centroid of the open part** of the face (SDF normal+fraction already in `ccFractionCore`, `mac_cutcell.hpp:31-42`) | 2nd | modest, reuses sampler | escalate if A's near-wall flux is short |
| C | ghost-fluid: blend `u_f*` toward the wall velocity by the solid fraction | 2nd | more | for moving boundaries later |

**Plan: build A, gate on Zick–Homsy + periodic-sphere permeability vs the staggered solver; escalate to B
only if accuracy is insufficient.** The projection is approximate anyway — getting the *area weighting*
right (A already does) is what protects mass conservation; the centroid correction is a 2nd-order refinement.

**OUTCOME (phase 4): Option A passes.** Periodic 2×2×2 sphere-packing Stokes permeability (Option A
averaging + plain central-difference cell correction) is incompressible (projected face div ~1e-10), exact
no-slip in the deep solid, grid-converges, and approaches the Z&H-validated staggered permeability
monotonically with resolution (rel. diff 2.61%→1.53%→1.12% at N=32/48/64). So neither the open-centroid
reconstruction (B) nor the openness-aware cell gradient is needed for converged integral quantities; both
remain available refinements if a future case needs accurate near-wall *cell* velocities or wall stress.
See `scripts/verify_colocated_spheres.py`.

**Two smaller IBM-entry points to handle in the collocated path:**
- **Cell-correction gradient at cut cells.** `½(phi(i+1)−phi(i−1))` reaches neighbors that may be solid
  (phi decoupled/≈0); near the wall it should become a one-sided / openness-weighted difference, mirroring
  how staggered `projectCorrect` only touches open faces.
- Both IBM mechanisms read the same SDF (above) — keep that the single source of geometric truth.

## 5. Phasing & validation

1. **Refactor, no behavior change.** Extract `SdflowSolver<GridLayout>`; `Staggered` is the policy. Prove
   bit-identical via `tests/regression/sdflow_regression.py` + the 18 `tests/kokkos_mpi` ctests + the Python
   verify scripts. Commit.
2. **Collocated advection + diffusion, no pressure.** Add `colocated_advection.hpp`, offsets `{0,0,0}`;
   validate body-force Poiseuille (analytic parabola), `cutcellPressure=False`.
3. **Collocated approximate projection.** Add `mac_approx_projection.hpp`; wire §1 steps 1–5 with cut-face
   Option A. Validate: divergence-free faces, Taylor–Green decay, lid cavity vs Ghia.
4. **IBM validation.** Zick–Homsy sphere-array drag + periodic-sphere permeability vs the staggered solver
   and external ground truth. Decide A vs B here.
5. **Domain BCs + MPI** for collocated (channel, BFS; bit-exact multi-rank), mirroring the staggered
   milestones.

## 6. Risks
- **Cut-face accuracy (§4)** — the main numerical unknown; gated at milestone 4 with a defined escalation
  (A→B).
- **Refactor regressions** — mitigated by the bit-identical guard in phase 1.
- **Cell-correction gradient at cut cells** — needs the openness-aware one-sided difference (§4), else
  near-wall cell velocities drift.
