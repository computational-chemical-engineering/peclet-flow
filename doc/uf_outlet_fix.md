# The high-side face plane and the scalar inflow (branch `uf-outlet-diag`)

Status: implemented on `uf-outlet-diag`, NOT merged. Landing is Frank's decision, because it
changes shipped scalar results at open boundaries. Two independent packages:

- **A**: the high-side domain face of a MAC face field survives the ghost fills. Commits `4f9c284`,
  `c002c8e`, `0420c93`, and the `5f7015b` test.
- **B**: a scalar enters through a Dirichlet inflow face with the prescribed flux. Commits
  `e56dc77` and `af88101`; the second carries the test.

Numbers below are host-openmp at cell units. Probes are in `tests/study/uf_outlet_*.py`. The
reference channel is 48×16×4: inflow U = 1 at −x, outflow at +x, walls at ±y, μ = 0.1, dt = 0.5,
scheme gauge-exact unless stated.

## A. The face plane wrap

### Mechanism

Along each axis the LOW domain face is an inner index and the HIGH one is the first ghost index
(`CLAUDE.md`, "the two ends of an axis are asymmetric"). `fillGhosts` wraps every axis
periodically, both in the single-rank `fillAxis` and in the distributed exchange. A MAC face
field that is ghost-filled therefore comes back with its high boundary face carrying the low
boundary's face. Three sites did this.

1. **`projectCorrectVelocities`, collocated.** The order was: `fillGhosts(uf_/vf_/wf_)`, then
   `bcCorrectOutflow`. After that, `uf_` on the outlet held U_in − Δφ. It should hold
   u*_out − Δφ, the flux the projection solved for, where `centerToFace` had written
   u*_out = u*_last.
2. **`advanceScalars`, both grids.** It runs its own `fillGhosts` on the face velocities. These
   are collocated `uf_`, or staggered `C[a].u` in place. The scalar's outflow flux was therefore
   the inlet plane. On the staggered grid the fill also overwrote the projection-corrected
   `C[0].u` outlet face.
3. **`bridgeVelocityToVof`, collocated.** It fills `uf_` in place before the copy. The VoF
   advector does not read that index (register, SCALING_ISSUES #8: colo-jet identity 1.1e-16).
   But the fill undid site 1 for everything that runs later in the step.

### Who read the wrong plane

- **Collocated momentum at an outflow face: no.** Advection reads `openFaceView()`, whose outflow
  faces `buildOpenFaceField` replaces with the zero-gradient extrapolation
  (`doc/uf_advection.md`). The `uf_` field is rebuilt from cells every step. The backflow
  stabilisation reads the cell velocity.
- **Staggered momentum: no.** The velocity is refilled before the predictor reads it.
  - Measured on a staggered channel with a passive scalar, 100 steps: u/v/w/p are bit-identical
    with and without the fix.
  - With a Boussinesq scalar they differ only through the scalar: max|Δu| 2.8e-4 (relative
    2e-4), max|Δp| 3.8e-4.
- **Scalar transport: yes, both grids.** The scalar overshoots in the outlet column:
  - c ∈ [0, 1] physically;
  - after 100 steps, max c = 1.350 (staggered) and 1.355 (collocated) before the fix, 1.0000
    after;
  - c ≡ 1 moves to 0.607…1.184 in one step at steady state.
  - This affects every configuration that registers a scalar with `add_scalar` and has a
    high-side open face: a passive scalar, a Boussinesq temperature, and the phase-change
    temperature.
  - The VoF `energy` scalar is transported in `advectVof`, not here.
- **`diagnostics.outflow_backflow()` and the β = 0 reversal warning, collocated: yes.** They read
  the raw `uf_` outlet, which is the inlet plane.
  - Reversed channel, census max_reverse at steps 1 / 2 / 200: 2.000 / 0.637 / 1.008.
  - The divergence-consistent values are 1.000 / 1.019 / 1.128.
  - Staggered was already exact.
- **Staggered `max_open_divergence_projected` after a scalar step: yes.** It read 0.785
  (= 1 − u_last at the wall row) instead of 2e-16.
- **Collocated momentum at a high-side INFLOW face (+x inflow, −x outflow; a top inlet): yes.**
  `openFaceView` replaces outflow faces only, so the advective flux into the inlet column was the
  outlet's plane. Measured at 100 steps, centreline:

| | inlet cell u | x=1 u | mid u |
|---|---|---|---|
| collocated, before | −1.03823 | −1.34924 | −1.24521 |
| collocated, after | −1.00036 | −1.35191 | −1.24938 |
| staggered (unchanged) | −1.00208 | −1.35115 | −1.25167 |

### Fix

`Solver::fillFaceGhostsKeepBoundary(f, a)` (`src/flow_ibm_bc.hpp`) wraps `fillGhosts(f)` in a
save and restore of the high-side face plane of normal axis `a`:

- The save and restore cover the INNER transverse range only (`bcSave/RestoreHighFacePlaneInner`).
  The transverse ghost rows are the neighbour's, which is the SCALING_ISSUES #8 rule and its
  rejected alternative.
- It acts only on a non-periodic axis whose high global face this rank owns. Otherwise it is
  plain `fillGhosts`.
- It is used at all three sites. `buildOpenFaceField` is untouched.

Commit `4f9c284` restricts it to outflow faces. Commit `c002c8e` extends it to every
non-periodic type, which fixes the high-side inflow case. Wall faces carry 0 on both ends, so
their restore is value-neutral. Reverting `c002c8e` alone gives back the outflow-only behaviour.

## B. The scalar inflow

### Defect 1: stale `cOld` ghosts (`e56dc77`)

`advanceScalars` copied c into `cOld` and filled c's ghosts afterwards. The explicit advection
reads `cOld`'s ghosts, so the first step after an inner-only write upwinded from stale values:

- the writes are `set_field`, the coupling drivers, and the VoF energy transport;
- on a fresh scalar the stale ghosts are zeros;
- c ≡ 1 with a Dirichlet-1 inlet fell to 0.5 in the inlet column, on both grids.

This is the same defect as WO-P23's `pcBuildInterface` fix. The fix fills the ghosts before the
copy, which is idempotent wherever they were already valid.

### Defect 2: the reflection ghost upwinded (`af88101`)

The Dirichlet ghost is 2v − c_inner. That is right for the diffusion row, where linear
interpolation gives the face value v. As an UPWIND state it carried u_n (2v − c_inner):

- 1-D plug, 64×4×4, U = 1, 40 steps: the inventory came out at U t + 0.500 exactly. That is the
  first step's u (2 − 0) dt.
- `scalarInflowGhosts` sets both ghost layers of `cOld` to v at a scalar-Dirichlet face that is
  also a velocity inflow face. This is the same open-face rule as the flux openness in
  `setSolid`: type 2 with a normal velocity or a profile.
- Koren then returns v exactly (r = 0). Walls keep the reflection.
- After the fix: inventory U t to 6e-14 (FOU and Koren, both grids), and the c = ½ crossing at
  x = U t.

## Gates

| gate | result |
|---|---|
| 12 state hashes vs `flow-scalar-ibm/doc/scalar_ibm_baseline_hashes.txt` | A: 12/12 byte-identical. B: 11/12. `scalar` 387d9bac → 438db94e, because of defect 1: that case calls `set_field('T')` before step 1. Max\|ΔT\| 5.7e-4 of 0.97 and max\|Δu\| 1.4e-5 of 0.060 at step 5. Defect 2 does not touch it (walls). |
| ctest `open_face_plane` (new, both grids) | pass. Census − required 1.5e-12 / 4.6e-12; outlet \|c−1\| 4e-16; staggered div 4e-16; high-inflow \|u_colo − u_stag\| 1.7e-3. On the pre-fix module it FAILS at the first check (1.000). |
| ctest `scalar_inflow` (new, both grids, FOU + Koren) | pass. Inlet \|c−1\| 3e-14; plug inventory − U t ≤ 6e-14; front at U t. Pre-fix: FAIL (0.5). With defect 1 fixed alone: FAIL (inventory +0.500). |
| full non-bench battery, OMP 4, -j4, MPI tree (`--bind-to none`) | 190/190 pass. 182 passed in the first run. On a host at load ~87 the outer 7000 s timeout stopped the last 8 (`vof_advect/blocks/redistribute_mpi`); rerun at -j2, they passed. |

## Proposed register entry (`../docs/decisions/flow.md`; not edited from this branch)

```
### A MAC face field's HIGH-side domain face is kept across every ghost fill
- decision: every ghost fill of a face field whose consumer reads the boundary face (the
  collocated projection before bcCorrectOutflow, advanceScalars on both grids, the collocated VoF
  bridge) goes through fillFaceGhostsKeepBoundary: save/restore the high-side face plane over the
  INNER transverse range on every non-periodic axis whose high global face the rank owns.
- rejected: (a) reordering the fill and bcCorrectOutflow -- the fill overwrites the plane whatever
  ran first; (b) restoring outflow faces only -- a high-side INFLOW face is wrapped the same way
  and collocated momentum reads it (openFaceView replaces outflow faces only); (c) restoring the
  full transverse extent (SCALING_ISSUES #8).
- why: the plain fill wrapped the low boundary's plane onto the high face. It handed scalars the
  inlet plane as outflow flux (c overshoot to 1.35 in [0,1]), blinded the collocated backflow
  census, overwrote the staggered corrected outlet in place (max_open_divergence_projected 0.785),
  and fed collocated momentum the outlet plane at a high-side inlet (inlet-cell u 3.8 % off).
- evidence: flow doc/uf_outlet_fix.md; ctest open_face_plane.
- consequence: momentum at an outflow face is unchanged by construction (openFaceView); results
  change for scalars at any high-side open face and for collocated runs with a high-side inflow.

### A scalar's advective inflow state is the prescribed value, and cOld's ghosts are filled first
- decision: advanceScalars fills c's ghosts BEFORE freezing c^n into cOld, and at a scalar-Dirichlet
  face that is a velocity inflow face sets cOld's ghosts to the prescribed value v.
- rejected: upwinding the Dirichlet reflection 2v - c_inner (right for the diffusion row, wrong as
  an upwind state: inflow u_n (2v - c_inner)); changing c's own ghost (would break the diffusion
  row); applying the constant ghost at Dirichlet WALLS (no flux there; it only moves the limiter).
- why: stale ghosts gave zero influx on the first step after set_field (c = 1 -> 0.5); the
  reflection gave a plug +0.5 U dt of spurious inventory.
- evidence: flow doc/uf_outlet_fix.md section B; ctest scalar_inflow.
```
