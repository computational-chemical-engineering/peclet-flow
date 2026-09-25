# Collocated multiphase with immersed solids — plan for the next package

*Companion plan to [`collocated_varrho_forces.md`](collocated_varrho_forces.md) (the "parent
note"), 2026-09-25, architect pass. This is a plan, not a finished design. Phase S2 in particular
needs its own architect pass if it is triggered. It records the carrier decision and the choices the
parent package makes for its sake (parent §4.9, L1–L5).*

## 1. Goal and scope

**Goal.** Variable density, geometric VoF with balanced-force CSF, and contact angles on
`SolverColocated` **with an immersed SDF solid**, at density ratios up to 1000.

**Two constraints on the design:**

- It must transfer directly to amr (parent §13). amr + solids + multiphase is the end goal.
- The user's immediate production target (two-phase flow in porous media at low capillary number)
  already runs on the **staggered** grid. The parent package gives it an exact static balance there
  (the balanced-force projection, parent §4.7). This plan is therefore about the collocated/amr
  road, not an emergency.

**Today:**

- V8 refuses an immersed solid (`requireCollocatedFaceForceScope`; `test_vof_collocated` T5).
- The ghost projection v1 supports neither variable ρ nor V8, so AUTO falls back to gauge-exact.

## 2. The cut-cell trade-off (why the carrier is not obvious)

Multiphase with solids needs three things at cut cells:

- **(S)** stability at every dt, with no stabiliser;
- **(C)** a finite-volume flux constraint, so that Weymouth–Yue transport is exactly conservative
  in cut cells;
- **(A)** O(h²) accuracy of the wall pressure force.

On flow's measured record (constant ρ, `collocated_invisible_subspace.md`), each candidate gives up
at least one:

| scheme | (S) | (C) | (A) | evidence |
|---|---|---|---|---|
| gauge-exact aperture (`faceInterp_` 9) | **no**: attractor family, and a rotational instability at cut cells that needs the wall-banded blend, whose margin depends on resolution | yes | ~yes | the invisible-subspace note: a 5e-4 spread across dt; R = 16 diverges at dt = 600 with w₀ = 0.5 |
| ghost projection (production, single phase) | yes (family-free, empirically) | **no**: a point-based constraint whose closure faces carry a ghost velocity *inside* the solid, not a flux over the cut cell's fluid volume | yes | `ghost_projection.hpp`; BiCGStab costs 2.3–2.7× the pressure stage |
| embed (Basilisk `embed.h` port) | **no**: unstable on beds (non-adjoint normalised pair, ×10–15 per 250 steps, immune to the blend) | yes | open | the invisible-subspace note, §6 S1.1 |
| **aperture-adjoint ρ-pair** (the parent's operators with $o=\alpha$) | **yes, proved** (parent §4.3 holds for fractional $o$). [MODEL]: radius exactly 1 with an immersed disk at ratios 1–10⁶ and dt 0.01–10⁴ | **yes**: the same aperture constraint the staggered cut-cell VoF already conserves on | **no**: O(h). The constant-ρ analogue (modes 11–13, deleted at 1.0.0) measured −11.1 % / −8.1 % / −8.6 % at R = 8 and −5.4 % at R = 12 against staggered on a dense bed | the invisible-subspace note, §6 S1.2–3 |

## 3. Decision

**S1 carrier: the aperture-adjoint ρ-pair.** V8 with $o=\alpha$. It is the only candidate that
meets the two hard requirements of multiphase, (S) and (C). It also:

- is exact under the balanced-force projection ([MODEL]: a ratio-1000 column with a disk crossing
  the interface gives 1.5e-13 ON and 1.4e-2 OFF after 30 steps);
- adds **no new operator family**: it is the parent package's code with fractional openness, which
  parent L1 makes a data change rather than a code change;
- keeps an SPD pressure operator (PCG/Chebyshev, no BiCGStab);
- matches amr's aperture-projection family.

Its weakness, the O(h) wall pressure force, is an accuracy deficit rather than a correctness one.
It is measured and bounded, and it is decided at gate D-S2 below.

**S2 (conditional accuracy upgrade).** A "ghost v2" momentum and pressure, plus a separate aperture
*transport* projection that produces WY's advecting field. The cost is a second solve per step. It
starts only if D-S2 says S1's gap is unacceptable, and it gets its own architect pass.

- The parent's weight-sum rule (L1) already makes gravity balance under the ghost one-sided
  reconstruction: the weights (1.5, −0.5) applied to face accelerations sum to 1.
- The option's RHS rule (L2) already absorbs `gpDivergDelta`.
- S2 therefore inherits both.

**Register consequence.** Reintroducing an aperture-adjoint family takes a new recorded decision.
The constant-ρ modes 11–13 were deleted as inferior for single-phase beds. The entry must say:

- the aperture-adjoint pair is the carrier for **multiphase with solids** because (C) is
  non-negotiable there;
- single-phase with solids keeps ghost.

Q11 of the parent note flags the two-family consequence for amr (a preference for the user).

## 4. S1 operators at cut and closed faces

- **Openness.** $o=\alpha$ is the cut-cell flux aperture (`ox_`/`oy_`/`oz_`, the field the
  staggered cut-cell projection uses). It is multiplicative in $\Pi_\rho$'s divergence, Φ, W, k and
  β (parent L1). Closed faces have α = 0.
- **The weight-sum rule.** At a cut cell, $W_c=\tfrac12(\alpha_-+\alpha_+)$. Hydrostatic balance is
  exact under it ([MODEL] disk: 1.5e-13).
- **Solid-centred cells:**
  - their momentum rows are pinned by the IBM, and the velocity is masked after the correction
    (the existing `maskVelocity`);
  - their pressure rows **stay coupled** wherever α > 0. That coupling is what makes the pair
    adjoint and family-free, and also what costs the O(h).
- **Density at cut faces** (parent L4). The operator's $\rho_f$, β's face mean and $\Pi_\rho$'s
  weights use the **same** cell ρ values, which are the ones the staggered cut-cell variable-ρ path
  already uses (`fillPropGhosts` + closure). Reuse them; do not invent a solid-cell ρ.
- **Momentum operator.** The collocated cut-cell IBM (Robust-Scaled overlay) with the $\rho_c/\Delta t$
  time term (`VarFaceProps` with sc = 0).
- **Wall velocity.** S1 covers static walls only (u = 0 in solid-centred cells). Moving scenes are
  a later rung.
- **Pockets.** Per-pocket constants, as the main projection already handles them. The option's RHS
  is compatible.

## 5. VoF through solids and contact angles

- **Transport, through the T4 bridge.** The collocated `uf_` sits on the low faces, exactly the
  staggered `u` positions. S1's projected `uf_` is α-divergence-free exactly as the staggered
  cut-cell projection's field is. So the staggered cut-cell WY kernel (`core::vof::cutcell` + flow's
  driver) runs **verbatim** on `uf_`.
- **What S1b must add:**
  - route `advectVof`'s cut-cell branch for the collocated grid;
  - lift the V8 + solid refusal.
- **Contact angles.** `core::vof::wetting` and flow's static and dynamic contact-angle machinery
  consume cell fields (C, SDF, interface normals), so they are grid-agnostic and reused.
- **CSF at cut faces.** The V4 face form times α. For constant κ it is exactly a face gradient of
  σκC, so it is balanced under the option at any α.

## 6. Phases, work orders and gates

**S1a — variable ρ + solids, single phase.**

- Changes:
  - V8 accepts `set_solid`;
  - $o=\alpha$ from the cut-cell geometry;
  - the IBM momentum rows;
  - masking.
- Gates:
  - the stability guard (parent §7.6) extended to a sphere-bed configuration, OFF and ON: G2/G3 at
    dt 0.1–100, ratio 1 and 1000;
  - hydrostatic, ratio 1000, with a solid sphere crossing the interface: with the option ON, face
    < 1e-12, cell < 1e-10 and dP/dz < 1e-10 from step 1; with it OFF, decaying;
  - Zick & Homsy drag at uniform ρ (variable-ρ mode), R = 8 and 12: record the gap against
    staggered, and require first-order convergence toward staggered (observed order ≥ 0.8);
  - MPI np = 1 exact, np = 2 and 4 within the parent's G5 tolerances.

**S1b — VoF + CSF + solids** (θ = 90° or drops away from walls).

- Changes: cut-cell WY on `uf_`.
- Gates:
  - exact conservation: Σ C·V_fluid constant to 1e-13 over 200 steps of a drop translating past a
    sphere at ratio 1000;
  - 0 ≤ C ≤ 1 (with the staggered path's wisp guard);
  - a constant-κ drop near a sphere with the option ON: face < 1e-13;
  - T4-style bridge identity: the collocated colour field equals a staggered run fed the same
    `uf_`, bitwise.

**S1c — contact angles.**

- Gates:
  - sessile caps at θ = 60°, 90°, 120° on an immersed plane: equilibrium height and base radius
    against the spherical cap within the staggered path's error; option-ON spurious currents
    < 1e-10·σ/μ;
  - capillary rise in an immersed tube (Jurin height) at two resolutions, collocated against
    staggered.

**S1d — MPI.** np = 1, 2, 4 for S1a–c; np = 1 exact.

**Decision gate D-S2 (user value judgement plus a measurement).**

- Measure: the S1 and staggered errors on (i) Jurin height and (ii) the entry capillary pressure of
  a pore throat (Young–Laplace threshold), each at two resolutions.
- **Default rule:** S1 is production for collocated multiphase with solids if its error is ≤ 2× the
  staggered error at the finer resolution on both. Otherwise S2 starts.
- The user sets the tolerance.

**S2 (conditional).** Ghost v2 + an aperture transport projection. Architect pass first.

**amr track** (after S1a, parallel to S1b–d; follows parent §13):

- **A0.** Promote parent L5 and the CSF pair to `core::scheme` / `core::vof` (core minor release;
  flow byte-identity as the gate).
- **A1.** Establish whether amr's uniform-ρ pair is transpose-exact at coarse–fine faces (a fact).
  Fix it or record it. The guard already runs (parent WO-G1).
- **A2.** amr variable ρ, single phase, on its aperture family with the C/F weights of parent §13.2.
  Also the pressure-driver decision at high contrast (parent §13.6(i)).
- **A3.** amr VoF + solids through its scalar-transport path (its own plan).

## 7. Risks and open questions (defaults)

| # | question | needs | default |
|---|---|---|---|
| P1 | Is S1's O(h) wall force acceptable for the target? | fact (D-S2) + **user preference** (tolerance) | 2× the staggered error, as above |
| P2 | Two collocated families (ghost single-phase, aperture multiphase+solids) in flow and amr | **user preference** | Accept for S1; revisit at S2 |
| P3 | Moving solids (scene) with V8 | scope | After S1c; wall-velocity faces in $\Pi_\rho$ are the one new rule |
| P4 | Option with inflow/outflow (drainage experiments) | **user preference** (scope) | Parent Q7: next package if the drainage target needs it |
| P5 | Momentum-consistent VoF on the collocated grid (motion above ratio ~100) | fact (parent WO-V3) | Out of this plan until WO-V3 reports |
