# VoF work orders — the remainder of the ladder (V6, V7, V9, P0–P3, W0–W2)

Written 2026-09-02 (Fable) after `suite/docs/VOF_PLAN.md` §13. The **shared preamble of
`vof_workorders_v5.md` applies verbatim** (hard rules 1–8, build recipes, conventions, the
worktree rule, `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`, rule 3b on capped solves, rule 4 on
twice-failed gates, never `git add -A`). Read it first. Findings go in the log at the bottom of
this file, newest first.

Facts landed today that these WOs rely on (all in `flow/CLAUDE.md`'s VoF section and the
`vof_workorders_v5.md` findings): cut-cell transport (`set_solid(..., cutcell_pressure=True)`,
`advect_vof`, `eps_eff`, the flux clamp), the static θ-fill (`set_contact_angle`,
`set_contact_angle_field`, `contact_angle_diagnostics`; θ is a per-cell FIELD already; flat SDF
walls must not sit at half-integer coordinates; domain-BC walls get no θ fill), open boundaries
(`set_vof_inflow`, `set_vof_backflow`, `max_open_divergence_projected`; the varRho outflow
operator is being fixed by WO-R2), the collocated all-fluid path, `PECLET_FLOW_EXACT_RESIDUAL=1`
(7.5 orders of flux divergence at ratio 1000; WO-R2 makes `enable_vof` set it).

---

## WO-V6 — dynamic contact angle and hysteresis  [Fable derivation → OPUS]

**Goal.** Replace the static θ imposed by the V5b fill with the grid-scale apparent angle of a
moving contact line, with an explicit slip length, and advancing/receding hysteresis. Nothing in
the fill changes; only the value of θ per contact cell does. Never report a dynamic-wetting
result without the slip parameter stated (VOF_PLAN §6).

**Model (state it on every page and docstring).** Afkhami, Zaleski & Bussmann, *JCP* 228:5370
(2009): the angle to impose at the grid scale `Δ` for a contact line moving at speed `U_cl` is
```
θ_Δ³ = θ_e³ + 9 Ca_cl ln(Δ/λ)          Ca_cl = μ_l U_cl / σ,   advancing: Ca_cl > 0, receding < 0
```
(Cox–Voinov with the outer scale set to the cell size, so the *numerical* slip `∝ Δ` is replaced by
the explicit `λ`). Clamp `θ_Δ` to `[1°, 179°]`. Hysteresis (the Huang-2026-style pair the plan
names, in its simplest consistent form — Fang et al., Dussan): with an advancing angle `θ_a` and a
receding angle `θ_r`, measure the current apparent angle `θ_app` of the contact cell (from the
fluid-side PLIC normal: `cos θ_app = m_f · n_w`, the same quantity the fill rotates); if
`θ_r ≤ θ_app ≤ θ_a` the contact line is **pinned** — impose `θ = θ_app` (the fill then reproduces
the current interface, so the contact line does not move); if `θ_app > θ_a` impose `θ_a`
(+ the Cox–Voinov correction with `U_cl`), if `θ_app < θ_r` impose `θ_r` (with the receding
correction). `U_cl`: the contact cell's fluid velocity (cell-centre mean of the faces) projected
on the wall-tangential direction of `m_f`'s in-wall component `t̂` (the direction the contact line
moves in), sign positive when the liquid advances (velocity along `−t̂`, since `m` points into the
gas). Smooth `U_cl` over the contact cell's in-wall neighbours (3-point mean) to kill the
cell-to-cell noise of a MAC velocity next to a wall.

**API.** `set_contact_angle_dynamic(theta_e_deg, slip_length_cells, mu_liquid)` (Cox–Voinov on a
static base); `set_contact_angle_hysteresis(theta_a_deg, theta_r_deg)` (+ the dynamic
correction if set); `set_contact_angle_dynamic_off()`; `contact_angle_diagnostics()` gains
`mean_imposed_theta`, `mean_apparent_theta`, `max_Ca_cl`, `pinned_cells`. The base
`set_contact_angle` (static) is byte-identical when neither is set.

**Gates.**
- **G1 the model itself, kinematically.** Prescribe a uniform wall-tangential velocity field
  (kinematic, `advect_vof`) over a flat SDF wall (quarter-integer placement) with a contact line:
  the imposed θ equals the Cox–Voinov value from the prescribed `U_cl` to 1e-10 (a pure kernel
  check of the sign convention: advancing raises θ, receding lowers it).
- **G2 spreading drop vs Cox–Voinov.** A drop released as a hemisphere (θ_app = 90°) on a wall
  with `θ_e = 30°`, Oh ~ 0.1, `λ = 0.1 Δ`: record `(θ_app, Ca_cl)` of the contact line over the
  spreading; gate: the points fall on `θ_app³ − θ_e³ = 9 Ca_cl ln(R/λ)` for the macroscopic
  `R ≈` the contact radius within the scatter of the measurement, i.e. the fitted slope of
  `θ_app³ − θ_e³` against `Ca_cl` within 25 % of `9 ln(a/λ)` over the window `Ca_cl ∈ [1e-3, 1e-1]`
  (Legendre & Maglio 2015 report the same test at that accuracy). Also: the spreading radius
  `a(t)` compared with Tanner's law exponent 1/10 at late times (report the fitted exponent).
- **G3 capillary rise vs Gründing et al. 2020** (`AMM` 86:142, the benchmark the plan names):
  their case with two plates, the rise height `h(t)` vs their reference curve for one slip
  length (they give the sensitivity); gate 10 % on the final height and the right qualitative
  overshoot class. The V5b Jurin scene was inconclusive (4-cell plates → overlapping bands,
  sharp plate ends): use 8-cell-thick plates with rounded ends (SDF union of a slab and two
  cylinders), quarter-integer placement, and read the meniscus curvature locally
  (`R = w/(2 cos θ)`) as well as the level difference. Do the STATIC Jurin check first
  (θ = 30°, 60°): level difference within 5 % — that is the V5b G4 gate re-run on the fixed scene
  and it must pass before the dynamics mean anything.
- **G4 hysteresis: drop on an incline.** A drop on a tilted flat wall (gravity with a
  tangential component), `θ_a = 70°, θ_r = 50°`: below the critical Bond number
  `Bo_c ≈ (cos θ_r − cos θ_a)` (in the `ρ g V^{2/3} sin α / σ` form; ElSherbini & Jacobi 2006
  give the retention relation) the drop must stay pinned (contact cells report `pinned`, centroid
  velocity → 0); above it, it slides with the advancing front at ≈ θ_a and the rear at ≈ θ_r
  (report both measured angles). Gate: pinned below `0.7 Bo_c`, sliding above `1.5 Bo_c`.
- **G5 MPI** np 1/2/4 bitwise on G1, reduction floor on G2, with the decomposition cutting the
  contact line.
- **G6 inertness**: no dynamic/hysteresis call ⇒ byte-identical to V5b (all VoF ctests).

**Deliverables.** `src/vof/wetting_dynamic.hpp` (container-free: Cox–Voinov, hysteresis
selector, `U_cl` projection), the plumbing where the θ field is filled, bindings,
`tests/kokkos/test_vof_wetting_dynamic.cpp` (G1, G4 small), MPI twin, `tests/study/vof_wetting_dynamic.py`
(G2, G3, G4), findings, CLAUDE.md.

---

## WO-W0 — Part III, the block container (single rank + distributed)  [Fable exchange design → OPUS]

**Goal.** A per-bubble VoF block (the TBFsolver `vofBlock` pattern, VOF_PLAN §10) as a THIRD
container over the same kernels: each block owns its own `WyAdvector` (+ later `VofCurvature`),
runs the V0/V1 kernels on a small dense box, and scatters its colour into the global union field
that the closures see. No numerical coalescence by construction. This WO stops at kinematic
transport (W0); WO-W12 adds NS coupling.

**Design.**
1. *Block.* Global index box `[lo, hi)` per bubble, = bubble extent + 3 cells on every side
   (the halo the kernels need — the same number TBFsolver uses); `WyAdvector` initialised on
   the box with `g = 3` (its extended block therefore reaches 6 cells beyond the bubble).
   Block table (id, lo, hi, master rank) replicated on every rank (an `MPI_Allgather` of
   `N_bub × 8` ints per step — trivial). Blocks are re-centred when the interface comes within
   one cell of the inner box (recompute the extent from the block's own colour every step; on
   re-centring, copy the colour into the new box: exact, no interpolation).
2. *Gather.* Each rank overlapping a block's EXTENDED box copies its (inner + halo-filled) face
   velocities restricted to the overlap into a contiguous buffer, in global-index order, and
   sends it to the master. Counts are a pure function of the replicated table and the
   decomposition, so every rank knows every message size: plain `MPI_Isend/Irecv` with
   precomputed counts, no NBX handshake needed (core's `NbxEngine` stays available for the
   dynamic case; do not use it here). The master unpacks into the block's face-velocity arrays
   in a FIXED order (by global index), so the assembled field is independent of arrival order —
   the bitwise gate depends on this. Single rank: the "gather" is a strided copy.
3. *Advect.* The block's colour is the block's own state (the master keeps it between steps —
   the global union is derived, never the source). Ghost policy: the block's extended-box cells
   outside the domain follow the global colour BC (periodic wrap / clamp / datum), cells inside
   the domain but outside the block's inner box are the block's own colour (pure gas for an
   isolated bubble), exactly as the standalone advector treats its ghosts.
4. *Scatter.* The block's inner colour goes back to the owning ranks (same messages reversed);
   each rank forms the union `C_global = max_blocks C_block` (UNPACK_MAX) into the registered
   `"C"` field (inner cells), then the ordinary `fillPropGhosts`. Per-bubble statistics on the
   master: volume, centroid, velocity (from the centroid), the second moments (deformation).
5. *Load balance.* Master assignment: round-robin by block id at W0; the weighted-ORB
   redistribution (`BlockDecomposer::init(…, weights)`) is W1. Record the imbalance you measure.
6. *The L1 promotion.* Move the container-free kernels (`plic.hpp`, `curvature.hpp`, the
   rules in `cutcell.hpp` and `wetting.hpp` that take only scalars/arrays) to
   `core/include/peclet/core/vof/` under `peclet::core::vof`, with `flow`'s headers becoming
   thin includes + `using` declarations, so the block container, the structured solver and
   the future AMR path share one copy. This touches `core` (a second repo): commit there first,
   bump the umbrella pointer LAST, and gate on **every VoF ctest bit-identical** — it is a file
   move. Do it as the FIRST commit of this WO, alone, so it can be reverted independently.

**Gates.**
- **G1 one bubble, bitwise.** The V1 LeVeque scene (32³, T = 3 reversal) with one sphere as a
  block: the block's colour equals the global-field `WyAdvector` result **bitwise** at every step
  (both use the same face field; the block's ghosts are the same values the global sweep reads).
- **G2 two bubbles, no coalescence.** Two spheres advected into each other by a prescribed
  solenoidal field: the union field shows overlap (`C_global = max`), the two blocks keep their
  volumes to 1e-14 each, and separating the flow (reverse) recovers two intact bubbles; the
  single-field control merges them irreversibly. This is the *raison d'être* gate.
- **G3 re-centring.** A sphere translated by 20 cells: the block moves, volume exact, the colour
  bitwise equal to a block that was large enough not to move.
- **G4 MPI** np 1/2/4 bitwise on G1–G3 with the decomposition cutting the bubble; master ranks
  ≠ owning ranks by construction (assign masters round-robin so at least one block's master owns
  none of its cells).
- **G5** L1 promotion: all VoF ctests bit-identical before and after the move, both backends.
- **G6** every existing VoF ctest bit-identical (the container is additive).

**Deliverables.** `src/vof/block_container.hpp` (`VofBlock`, `VofBlockSet`), the exchange
(`src/vof/block_exchange.hpp`, MPI-guarded), solver plumbing (`enable_vof_blocks(seeds)`,
`vof_block_stats()`), bindings, `tests/kokkos/test_vof_blocks.cpp`, `tests/kokkos_mpi/test_vof_blocks_mpi.cpp`,
the core move + its ctest run, findings, CLAUDE.md (flow and core).

---

## WO-P01 — Part II, rungs P0 (fixed-flux interface) and P1 (Stefan problem)  [Fable spec → OPUS]

**Goal.** The phase-change kernel set of VOF_PLAN §9 in its planar form: interface regression by
PLIC plane shift with exact clip-and-redistribute, the divergence source shifted into pure gas
cells, and a temperature field with the interface at `T_sat` giving `ṁ` from one-sided pure-cell
gradients. 1-D/planar gates only; P2/P3 follow.

**Design (Boyd & Ling 2023 / Malan 2021 pattern, as the plan says — do not invent a smeared
source).**
1. *Mass flux field* `mdot` (kg m⁻² s⁻¹, in solver units) on interfacial cells: P0 prescribes it
   (uniform); P1 computes it as `ṁ = (k_l ∇T_l·n − k_g ∇T_g·n)/h_lv` with each one-sided gradient
   from the PURE cells on that side: take the interfacial cell's PLIC normal `n = m/|m|₂` and
   plane point, sample `T` at the pure cells within a 5³ stencil on that side weighted by the
   collinearity `ξ = (d·n)²/|d|²` and inverse distance (Malan's weighting), fit `∂T/∂n` by
   weighted least squares of `T − T_sat` against the normal distance from the plane. Interfacial
   cells hold `T = T_sat` (Dirichlet) in the energy solve; IHTR (`T_Γ = T_sat + ṁ R_int`) as an
   option with `R_int` default 0 at this rung.
2. *Interface regression.* Per interfacial cell, shift the PLIC plane: the liquid volume to
   remove is `ΔV = ṁ A_Γ Δt/ρ_l` with `A_Γ` the PLIC polygon area (`plic.hpp` has the volume;
   area = |∂V/∂α| by the finite difference of `plicVolume` in α, which is exact to round-off for
   a plane — or add a `plicArea` kernel); new `C = plicVolume(m, α')` with `α'` solved by
   `plicAlpha` for the target volume `V − ΔV`; if `V − ΔV < 0`, set `C = 0` and push the deficit
   into the liquid-side neighbours along `−n` (clip-and-redistribute, conservative to
   round-off; count and report). Never a volume source in the C equation.
3. *Divergence source.* `S = ṁ A_Γ (1/ρ_g − 1/ρ_l)/V_cell` from each interfacial cell is
   deposited into the nearest PURE GAS cell along `+n` (one layer behind the interface — the
   plan's "compact pure-gas layer"; if that cell is itself interfacial, walk one more), added to
   the Poisson RHS as `div u = S` (an extra compatible RHS array through the existing deflated
   pressure solve; global `Σ S` must be balanced by the outflow — closed domains need an outflow
   face, which WO-R provides, or the equal-density trick `ρ_g = ρ_l` for P0's regression-only
   check).
4. *Transport.* Because the source sits in pure gas cells, the projected face velocity at the
   interfacial cells is the liquid velocity; WY advects C with it unchanged. (The band-extended
   velocity of the plan is the P3 upgrade if the Scriven gate needs it.)
5. *Energy.* `T` as a registered scalar through `add_scalar` (Koren TVD + implicit diffusion,
   `scalar_transport.hpp`) with per-phase `ρ c_p` and `k` from the colour (closures), the
   Dirichlet `T_sat` in interfacial cells imposed as fixed cells in the scalar operator (a
   per-cell Dirichlet mask — add it to `ScalarField` as an optional mask; inert when empty),
   and the advective term using the same face velocity. The consistent `ρ c_p T` geometric
   transport (plan item 6) is the P3 upgrade.

**Gates.**
- **P0a regression-only** (ρ_g = ρ_l = 1, no source): planar interface, uniform `ṁ`, 1000
  steps: interface position `x_Γ(t) = x_0 − ṁ t/ρ_l` to 1e-12 (exact by construction), liquid
  volume to round-off, `C ∈ [0, 1]`.
- **P0b with the source** (ratio 100, outflow at the gas end via WO-R's outflow, or a periodic
  box with the source balanced by a prescribed sink): gas velocity `u_g = ṁ (1/ρ_g − 1/ρ_l)`
  away from the interface to 1e-10 relative (the 1-D exact solution), `max|div − S|` at the
  projection floor, interfacial-cell velocity = the liquid velocity (0 in the liquid frame).
- **P1 Stefan** (1-D, liquid at `T_sat` heated from a wall at `T_w > T_sat` through a vapour
  layer — the classic Stefan problem with the vapour growing; or the equivalent solidification
  form): `x_Γ(t) = 2 λ √(α_g t)` with λ from `λ e^{λ²} erf(λ) = St/√π`, `St = c_p (T_w −
  T_sat)/h_lv`; gate: interface position within 0.5 % at N = 256 cells across the layer at
  `t` where the layer is 64 cells thick (Malan reports 0.23 %); order of convergence over
  64/128/256 reported (expect ~1.5–2).
- **MPI** np 1/2/4: P0a bitwise; P1 at the reduction floor, with the decomposition cutting the
  interface.
- **Inert**: no phase-change call ⇒ every VoF ctest byte-identical.

**Deliverables.** `src/vof/phase_change.hpp` (container-free: the plane shift, the one-sided
gradient fit, the source rule), solver plumbing (`enable_phase_change(rho_g, rho_l, h_lv, ...)`,
`set_mass_flux(field)` for P0, `phase_change_diagnostics()`), the scalar Dirichlet mask,
bindings, `tests/kokkos/test_vof_phase_change.cpp`, MPI twin, `tests/study/vof_stefan.py`,
findings, CLAUDE.md.

---

## WO-V7 — the pore-scale campaign (after WO-R2)  [OPUS runs, Fable/user interpret]

Three cases, each a script under `tests/study/pore_scale/` and together one gallery page
(`examples/pore-scale-imbibition`): (1) the pore doublet (two channels of different width from a
common inlet, drainage and imbibition at three capillary numbers, θ = 45° and 135°): which
branch fills first vs the classical Chatzis–Dullien criterion; (2) imbibition into the E6
sphere packing at θ = 30°/60° and Ca = 1e-4/1e-3 (the capillary dt binds: budget the wall clock
from the measured ms/step and pick the largest Ca that answers the question — VOF_PLAN §4 V7
preamble); (3) a Zhao-2019-style 2-D micromodel (quasi-2D, a random cylinder array) at
θ = 45°/90°/135°: the invasion pattern's fractal dimension / saturation at breakthrough vs their
published trend with wettability. Every run: exact residual on, `max_open_divergence_projected`,
pressure iterations vs cap; a capped run is not a data point. Report ms/step and the dt census
(capillary vs CFL) for the WO-V9 profile.

---

## WO-V9 — performance: profile first, then the levers  [OPUS]

1. Per-kernel timers around the VoF pipeline (reconstruct, fluxes, sweeps, fills, curvature,
   CSF, the bridges) exposed through `vof_timing()`; measure on the gallery cases (E4 Hysing
   64×128×4, E6/E7 packing 64×64×128, a 128³ droplet) on a QUIET GPU: fraction of the step in
   VoF vs pressure vs momentum. 2. The worklist compaction's actual gain (`useWorklist` on/off).
   3. The halo cost of the g = 3 exchange under MPI np 2/4 (the PARIS partial-column-sum trick
   is only worth it if this shows). 4. Interface-weighted ORB rebalancing: weights = mixed-cell
   counts + a base cost, through the existing `redistribute`; measure the imbalance before/after
   on a bubble swarm (W1 provides the swarm; until then, the E7 packing case at np 4). Report;
   implement only the lever the numbers justify.

---

# Findings log (v6 work orders)

(append per WO, newest first)
