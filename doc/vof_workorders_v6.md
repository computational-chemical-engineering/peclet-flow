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

## WO-P23 — Part II, rungs P2 (sucking interface) and P3 (Scriven bubble growth)  [Fable spec → OPUS, after WO-P01]

Build on WO-P01 as shipped (read its findings first: the ṁ sign convention with the PLIC normal
into the gas, the analytic `plicArea`, the liquid-aware clip-and-redistribute, the fixed-order
gathers, and the P1 order 1.07 → 1.50 mechanism: the energy solve pins the whole interfacial
CELL at `T_sat` while the gradient is measured from the PLIC PLANE — an O(h) sign-oscillating
mismatch).

**P2 — the sucking interface** (Welch & Wilson, *JCP* 160:662, 2000): planar, the vapour at
`T_sat` on one side, SUPERHEATED liquid on the other; the interface moves INTO the liquid and the
liquid's superheat is "sucked" into the vapour production. Similarity solution with the two-phase
Stefan condition (Welch & Wilson give it; Boyd & Ling 2023 §4.2 restate it and report order ≥ 1.4
on the interface position). Needs the liquid side of the gradient fit to be live, i.e. the
per-phase `k(C)` and `ρ c_p(C)` closures in the energy solve (VOF_PLAN §9 item 6: `ρ c_p T`
advected with the same geometric fluxes, face heat capacities per sweep — implement the
consistent transport here, it is what stops artificial heating at high `ρ c_p` ratio) AND the
band-extended liquid velocity (§9 item 3): with the source in the gas, the liquid moves toward
the interface; WY must advect `C` with the LIQUID velocity extended one band into the gas
(constant extension along `n` from the nearest pure-liquid cells, then a band-local divergence
cleanup — the plan names Palmore–Desjardins eq. 61; a cheap first version: extend, then subtract
the band's mean normal divergence per interfacial cell so `Σ_f o_f u_f = 0` there to 1e-12;
measure whether the full band Helmholtz projection is needed by the gate). Gate: interface
position order ≥ 1.4 over 64/128/256 and the temperature profile within 1 % of the similarity
solution at N = 256 (ratio 10 first, then the water/steam ratio ~1600 and report).
**The P1 pinning mismatch is the first thing to fix** (it is what caps P1 at order 1.07 on the
coarse pair): impose `T_sat` at the PLANE by a ghost-value (GFM-style) Dirichlet in the interfacial
cell — the cell value that makes the one-sided linear profile from the pure neighbour hit `T_sat`
at the plane distance — instead of pinning the cell centre; re-measure P1's order (expect ≥ 1.5 on
both pairs) before P2.

**P3 — Scriven bubble growth** (Scriven, *CES* 10:1, 1959): a spherical vapour bubble growing in
uniformly superheated liquid, `R(t) = 2 β √(α_l t)` with β from Scriven's integral (tabulate it
numerically in the test; Ja = ρ_l c_p ΔT/(ρ_g h_lv) = 0.5, 2, 10). 3-D, 128³ first (bubble from
D/Δ ≈ 12 to ≈ 40), then 192³ if the GPU allows; state the resolution. Gate: `R(t)` within 1 %
over the last half of the run at the finest resolution you ran (Malan/Boyd–Ling report ≲ 1 % at
256³); if it misses, the named lever is Aslam quadratic extrapolation of `T` across the band
(Tanguy 2014) — implement it only if the gate says so, as an option, and record before/after.
Both: MPI np 1/2/4 at the reduction floor with the decomposition cutting the interface; every VoF
and phase-change ctest of P01 bit-identical when the new options are off; every run records
the pressure iterations vs cap and `max|div − S|`.

Deliverables: the plane-anchored Dirichlet, `k(C)`/`ρc_p(C)` closures + consistent `ρ c_p T`
transport, the band-extended velocity, `tests/kokkos/test_vof_phase_change.cpp` extended (P2 at
64/128), `tests/study/vof_sucking.py`, `tests/study/vof_scriven.py`, findings, CLAUDE.md.

---

## WO-W12 — Part III, rungs W1 (many bubbles, redistribution, statistics) and W2 (NS coupling, channel_18)  [OPUS, after WO-W0]

Build on WO-W0 as shipped (read its findings: the partitioned gather/scatter, the bitwise horizon
and `bubbleEps`, the round-robin master imbalance 1.3–2.0, host-staged packing, the re-centring
reallocation, all-fluid only).

**W1.** (a) Master assignment by the weighted ORB: weights = block cell counts, through
`BlockDecomposer::init(…, weights)` on a 1-D "block space", or simply a greedy longest-processing-
time assignment over ranks (measure both; ship the one with the better max/mean on a 64-bubble
swarm at np 4/8 — LPT is deterministic and needs no communication, which the bitwise gate likes);
periodic re-assignment every N steps with block state migrated (the colour box is a contiguous
buffer — send it). (b) Device-resident packing (the pack/unpack kernels run on the block's
memory space; host staging only at the MPI boundary if the MPI is not CUDA-aware — check how
`GridHalo`'s device-resident variant does it). (c) A block pool so re-centring does not reallocate.
(d) Per-bubble Lagrangian outputs through Python: `vof_block_stats()` → id, volume, centroid,
velocity, the 3×3 second-moment tensor (deformation), and a per-bubble `interface_area` from the
PLIC polygons (the gallery asked for one). Gates: 64 bubbles in the LeVeque field at np 1/2/4/8
**bitwise** across np and across re-assignment events; every marker's volume to 1e-14; measured
imbalance before/after (report); packing time device vs host (report, not gated).

**W2.** NS coupling: the union `C` drives the closures as today; curvature per block
(`VofCurvature` on the block, the same cascade) and the CSF face force formed ON THE BLOCK
(`σ κ_f ΔC/h` with the V4 rule on the block's faces) scattered **UNPACK_SUM** into three global
face-force fields that `addCsfRhs` consumes in "block mode" (a sibling branch; the global-field
mode is byte-identical when blocks are off). The face velocity gathered to the block is the
projected `u^{n+1}` exactly as `advectVof` uses it. Momentum consistency is NOT in W2 (blocks are
rated to ratio ~100 with motion, like V2a; W2b = the design for the union-field momentum
sweeps). Gates: (1) **Hysing case 1 through the block path equals the global-field run** within
1 % on max rise velocity and `y_c(3)` (both without `enable_vof_momentum`, quasi-2D 64×128×4);
(2) a single 3-D rising bubble at Eo = 10, Mo = 1e-3 (ratio 100) against the Grace-diagram terminal
velocity / Duineveld-class shape within 10 %; (3) **two bubbles head-on** (one rising, one held
by a counter-flow or two rising in line): no numerical coalescence, film drains to one cell and
the blocks stay two; (4) **`channel_18`**: transcribe TBFsolver's `channel_18` case
(`/home/frankp/Codes/TBFsolver/channel_18`: read its input files for the domain, the 18 bubble
seeds, Eo/Re/ratio, the wall BCs and the body force) to the block path at TBFsolver's resolution
or the nearest power of two; run to a statistically steady state as far as the GPU allows; report
the void-fraction profile across the channel and the mean liquid velocity profile against
whatever TBFsolver's case directory ships (if it ships no reference data, report ours as the
first datum and say so); (5) MPI np 1/2/4 at the reduction floor with bubbles cut by the ORB;
(6) every existing VoF ctest bit-identical.

Deliverables: the W1 pieces in `src/vof/block_container.hpp`/`block_exchange.hpp`, the block CSF
mode, bindings, `tests/kokkos/test_vof_blocks.cpp` extended, MPI twin, `tests/study/vof_blocks_swarm.py`,
`tests/study/vof_channel_18.py`, findings, CLAUDE.md.

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

## WO-V7 findings — the pore-scale campaign (doublet, packing imbibition, micromodel) — 2026-09-02, Opus

Branch `vof-v7`, worktree `../flow-v7`, built from `origin/main` at `2b55edb` (WO-R2 landed).
Deliverables: `tests/study/pore_scale/{pore_doublet,imbibition_packing,micromodel_2d}.py`, this
entry, and the draft gallery page `examples/pore-scale-imbibition/index.qmd` on the
`peclet-examples` branch `vof-examples-6` (not pushed, per the work order).

**Every number below was taken on a SHARED machine** — four other sessions (WO-P23, WO-W12, the E6
and E7 gallery pages) were on the same GPU and the same 48-core host throughout. The `ms/step`
figures are an upper bound and are quoted only because WO-V9 asked for them: the same doublet
configuration measured **43 ms/step on a briefly idle GPU** and 80–525 ms/step over the campaign
depending on who else was running. Nothing else here is timing-dependent.

### Common configuration, and why each piece is what it is

One set of fluids for all three cases: `sigma = 100`, `rho_l/rho_g = 100/1`, `mu_l/mu_g = 4/0.04`
(density **and** viscosity ratio 100), no gravity and no body force of any kind. The drive is always
an inflow face at a prescribed uniform superficial velocity `U` against an outflow face
(`set_vof_inflow(in, 1.0)` + `set_vof_backflow(out, 0.0)`), never a periodic net force (VOF_PLAN
§13.2 item 6). Static contact angle, `enable_vof_momentum` on, the exact residual on by its new
default under `enable_vof`, `set_outflow_rho_correction` at its new ON default, the solid clear of
both open faces, FCG selected LAST (a `set_property_model("rho", …)` fires `set_density_mode`,
which reselects Chebyshev and silently discards an earlier driver choice).

* **Ratio 100 and not 1000**, as the work order specifies: V2b's uniform-velocity momentum identity
  is floored at 1.2e-7 by the FLOAT momentum-operator storage, and WO-R's G3 gas-over-pool caveat
  (a resting pool at ratio 1000 picks up 3.1e-2 of the inlet speed, unchanged by WO-R2) is open.
* **The capillary numbers are APPARENT**: `Ca = mu_l U_inlet / sigma`, from the prescribed inlet
  velocity, not from a measured contact-line speed. The angle is the STATIC V5b one and the
  velocity-side Navier slip (V6b) does not exist, so contact-line mobility is set by the wall's
  numerical slip — WO-V6 measured that at ~1/180 of Lucas–Washburn. That is not an ornamental
  caveat; it is the mechanism behind finding 3.
* **The property values are not free, but they are not arbitrary either.** Eliminating `sigma`,
  `rho` and `mu` in favour of the two dimensionless groups gives the step count of a pore-scale VoF
  run as `N ~ L sqrt(w / (Re Ca))` — **independent of sigma, rho and mu separately**. The only
  levers on cost are the geometry and the Reynolds number one is willing to accept. The values above
  put `Re = rho_l U w / mu_l` at 1 / 10 / 100 for `Ca = 1e-4 / 1e-3 / 1e-2` in the doublet, so the
  capillary-dominated end of the sweep is properly creeping and the viscous end is not.

### Finding 1 (the expensive one): a flat SDF wall on an INTEGER coordinate makes a driven two-phase run diverge geometrically, and none of its own diagnostics says so

The doublet's channel walls are boxes. With their faces at integer `z` — exactly on a cell face, so
`sdf` at every cell centre is ±1/2 and **no wall cell is cut at all** — the whole campaign was
unrunnable. The ladder that isolated it (88×4×80 doublet, 250–300 steps each, host-openmp, `dt`
re-picked every step from `vof_step_limits()`):

| variant | step 50 | step 150 | step 300 | verdict |
|---|---|---|---|---|
| base (integer walls, θ 45, ratio 100, ST + momentum) | max\|u\| 1.106e+03, dt 1.9e-04 | 1.271e+05, dt 1.7e-06 | **1.522e+08**, dt 1.4e-09 | diverges |
| neutral 90° fill (no `set_contact_angle`) | 9.421e+02 | 1.100e+05 | 1.518e+08 | diverges — **not the contact angle** |
| ratio 10 | 2.300e+01 | 1.875e+03 | 1.700e+06 | diverges — **not the density ratio** |
| ratio 1 | 2.296e+00 | 3.259e+01 | 3.603e+03 | diverges, more slowly — **not the ratio at all** |
| no septum (a plain slit, same walls) | 1.122e+03 | 1.313e+05 | 1.497e+07 | diverges — **not the corners** |
| surface tension OFF, fixed dt 0.05 | — | — | — | throws on step 1 at CFL 5.63, i.e. `max\|uf\|` = 112 already — **not surface tension** |
| **no solid at all** (same inflow/outflow, ratio 100, ST, momentum) | 2.626e-01 | 2.714e-01 | 2.773e-01 | **flat and stable** |
| **walls shifted a QUARTER cell**, all else identical | 3.56 at step 1, **decaying to 1.21 by step 12**, dt at the capillary limit | | | **stable** |

So it is the flat SDF wall, and specifically where it sits inside its cell — WO-S finding 5 again
(quarter-integer, not half-integer, placement), but with a far worse symptom than a biased angle.
The corner where a channel mouth meets the slab front reads `max|u| = 1.125e+02` on the **first**
step against a physical scale of 0.42, and thereafter grows ~5 % per step while the interface-local
`dt` limiter shrinks by the same factor. The Weymouth–Yue cap therefore **never fires**, `step()`
never raises, the pressure solve reports a healthy 25–33 iterations out of 400 at every step, and
the only tell is that simulated time stops advancing: `t` frozen at 0.1868 s after 1381 steps with
`max_open_divergence_projected()` at 2.4e+20.

Two things to carry:

1. **The scene rule is campaign-level, not gate-level**: every flat SDF wall in a two-phase run
   must sit at a quarter-integer coordinate. `WALL_SHIFT = 0.25` in `pore_doublet.py` carries the
   measurement in its comment. Curved solids (the packing's spheres, the micromodel's cylinders)
   are generically off-grid and were never affected — which is why E6/E7 never saw this.
2. **A driver-side divergence guard is missing and would be nearly free.** A run whose `dt` has
   fallen ten orders below its initial capillary limit is diverging; nothing in `flow` says so,
   because every individual diagnostic is locally correct. Recorded in the examples repo's
   `ISSUES.md` too.

### Finding 2: `capillary_dt` is a function of the CURRENT density field, so the first dt of a gas-filled domain is 7.1× too large

`vof_step_limits()['capillary_dt']` evaluates Brackbill from the density field as it stands. Called
immediately after `enable_vof` on a domain that is entirely gas, the closures have not run and it
returns the *base* `set_rho` value, 0.2835 s; one `step()` later it returns 0.0399 s. A driver that
sizes its first `dt` from the pre-step call starts 7.1× over the limit and the advector throws on
step 1 (`surface tension: dt = 0.141700 exceeds the capillary limit 0.019947`). All three scripts
re-pick `dt` from the solver's own limiter **every step**, which costs one device reduction and
removes this and the stale-limit trap the E7 page recorded, at once.

### Case 1 — the pore doublet (`pore_doublet.py`)

**Scene.** Quasi-2D, 88×4×80, periodic in y. Inflow at −x at a uniform superficial `U`, outflow at
+x, walls at ±z (buried inside the solid). Three SDF slabs occupy 20 ≤ x < 68 (**offset by a
quarter cell**, finding 1) and leave a narrow channel `w = 16` at z ∈ [48, 64) and a wide one
`2w = 32` at z ∈ [8, 40); the open fraction of the cross-section is 0.60. Both channels leave a
common inlet plenum and rejoin in a common outlet plenum, and the slabs are clear of both open
faces. The inlet plenum is liquid-filled at t = 0, so t = 0 is the front **at the branch
entrances**; filling the plenum from the inlet costs as many steps again and carries no physics.

**Two front metrics, and why both are reported.** The **tip** is the leading edge — the last x,
contiguous from the entrance, at which *any* cell of the cross-section is more than half liquid.
The **mean** front is the last x at which the openness-weighted mean colour of the cross-section
exceeds ½. They differ by the length of the meniscus, and that length scales with the CHANNEL
WIDTH, so the mean front carries a systematic bias of order `w/2` in favour of the narrow branch —
about 8 cells out of 48 here. The first pass of this campaign used the mean front only and it
reversed the verdict at θ = 135°, Ca = 1e-2 (mean: narrow 20 / wide 17 at the same instant at
which the tip read narrow 21 / wide 22). **Breakthrough is declared on the tip**, and both are
printed.

**The classical criterion, evaluated on this scene.** `|ΔP_c(narrow) − ΔP_c(wide)| = 2σ|cos θ|/w −
2σ|cos θ|/2w` against `ΔP_μ = 12 μ_l ū L/(2w)²` at the mean branch velocity gives
**471 / 47.1 / 4.71** at Ca = 1e-4 / 1e-3 / 1e-2. All three points are therefore
**capillary-dominated**, and Chatzis & Dullien's criterion predicts *narrow first at every Ca for
θ = 45°* and *wide first at every Ca for θ = 135°*; the viscous-dominated crossover for this
geometry sits at Ca ≈ 4.7e-2, above the whole sweep. (Lengthening the branch is the cheapest way to
bring it into range: `ΔP_μ` is linear in L, `ΔP_c` independent of it.)

| θ | Ca | Re | t_bt narrow | t_bt wide | fills first | tip n/w | S_narrow | S_wide | pressure | max\|div\|_proj | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45° | 1e-2 | 100 | **69.65 s** | 85.91 s | **narrow** ✔ | 48 / 47 | 0.775 | 0.730 | 63/400 | 5.7e-08 | 473 |
| 45° | 1e-3 | 10 | 973.4 s | **696.2 s** | **wide** ✘ | 47 / 48 | 0.793 | 0.906 | 128/400 | 7.2e-09 | 159 |
| 45° | 1e-4 | 1 | — (partial) | — (partial) | wide ✘ | mean 5 / 10 | 0.115 | 0.228 | 106/400 | 1.1e-06 | 121 |
| 135° | 1e-2 | 100 | 101.6 s | 103.0 s | tied (narrow by 1.4 %) | 48 / 47 | 0.937 | 0.885 | 69/400 | 5.7e-08 | 525 |
| 135° | 1e-3 | 10 | — never | **749.1 s** | **wide only** ✔ | 0 / 48 | **0.000** | 0.936 | 86/400 | 7.2e-09 | 153 |
| 135° | 1e-4 | 1 | — (partial) | — (partial) | wide ✔ | mean 2 / 12 | 0.043 | 0.235 | 80/400 | 5.6e-10 | 137 |

(The Ca = 1e-4 rows are the FIRST pass, stopped by a 1800 s wall-clock budget after 13 800–15 000
steps at 12.9 % and 0.0 % of the branch filled; they are reported as partial fills, on the mean
metric, and their ordering is unambiguous — the wide front is 2–6× ahead. Every row is valid: no
capped solve anywhere, colour exactly 0 in solid cells, clipped volume 0.000e+00.)

**Verdict, in two halves.**

* **The wettability contrast is right, and it is enormous.** At Ca = 1e-3, everything identical but
  the angle, the narrow branch ends at S = 0.793 for θ = 45° and at S = **0.000** for θ = 135° —
  the non-wetting liquid enters the narrow branch briefly (tip 3–4 cells early in the run) and is
  then **expelled** from it as the wide branch takes the flux. That is the drainage half of
  Chatzis–Dullien reproduced without qualification, and it is the same at Ca = 1e-4 (S 0.043 vs
  0.115 at θ = 45°) and at 1e-2 (narrow leads by 19 % at θ = 45°, ties at θ = 135°). @eq-pc is
  doing exactly what it should.
* **The imbibition half is reproduced only at the HIGHEST capillary number, and the Ca dependence
  runs backwards.** For θ = 45° the criterion predicts narrow-first at all three Ca; measured, the
  narrow branch leads at Ca = 1e-2 and *loses* at 1e-3 and 1e-4. The mechanism is the one the
  preamble names. Writing the doublet's flow split with a fixed total flux Q,
  `Q_narrow = [ΔΔP_c + R_wide Q] / (R_narrow + R_wide)`, so as Q → 0 the wetting doublet must reach
  `Q_narrow → ΔΔP_c/(R_n+R_w) > 0` with `Q_wide < 0` — the narrow branch imbibes *while drawing
  liquid back out of the wide one*. That limit needs the contact line to advance under its own
  suction, and this solver's contact line advances at whatever the wall's numerical slip permits
  (WO-V6: ~1/180 of Lucas–Washburn). When the imposed velocity is well above that slip velocity
  (Ca = 1e-2) the wetting condition acts and the narrow branch wins; below it (Ca ≤ 1e-3) the
  capillary term cannot be delivered, the split degenerates to the pure viscous one
  `Q_n ≈ R_w Q/(R_n+R_w)`, and the wide branch (conductance ∝ w³) wins at every lower Ca.
  **So the Ca at which the measured ordering flips is not a physical crossover at all: it is where
  the imposed velocity crosses the solver's numerical slip velocity.** That is a quantitative
  argument for V6b, and it is the campaign's main verdict on case 1.

### Case 2 — imbibition into an SDF sphere packing (`imbibition_packing.py`)

**Scene.** 36 grains of contact radius 8 settled by a deterministic NumPy soft-sphere relaxation in
a laterally periodic 48×48 column (no `dem` dependency), sampled at an SDF radius of 0.85 × the
contact radius so the throats are resolvable. Grid 48×48×96, x/y periodic, liquid inflow at −z,
outflow at +z, the bed clear of both. Bed surfaces span z = 17.2 … 67.0 (3.7 SDF grain diameters);
bed core z = 24 … 60 with **porosity 0.5984**. The **percolation throat radius** (the max–min path
from the bed bottom to its top, computed exactly by a Dijkstra sweep on the SDF) is **2.198 cells**
— i.e. the bottleneck's meniscus lives at or below the ~2.5-cell floor where V3 measured the
height-function cascade to be permanently in its PLIC-volumetric paraboloid fallback. The inlet
plenum is liquid-filled at t = 0 (filling it from the inlet costs more steps than the bed and
carries no physics). Breakthrough = the first sample at which the openness-weighted colour anywhere
on the bed's top plane exceeds ½; each run then continues for a further 25 % of its breakthrough
step count. Trapped gas = gas in the bed core that is NOT connected (6-connected, with the x and y
periodic seams stitched by a union–find) to the outlet plenum.

| θ | Ca | Re_grain | t_breakthrough | S at breakthrough | S final | trapped gas | pressure | max\|div\|_proj | ms/step | capillary dt binds |
|---|---|---|---|---|---|---|---|---|---|---|
| 30° | 1e-3 | 8.5 | **690.7 s** (step 6372) | **0.7257** | 0.9134 | **0.0103** | 27/400 | 2.26e-08 | 280 | 19 % |
| 60° | 1e-3 | 8.5 | **800.9 s** (step 6372) | **0.8521** | 0.9834 | **0.0112** | 27/400 | 1.68e-09 | 154 | 60 % |

Both runs valid (no capped solve), colour exactly 0 in solid cells, clipped volume 0.000e+00,
C ∈ [1e-31, 1] over the uncut fluid.

**The result, and it is not the textbook one.** The MORE strongly wetting case breaks through
**16 % earlier and 15 % drier** (S_bt 0.726 vs 0.852). The textbook expectation for strong
imbibition is the opposite — a flatter, more compact front and a *higher* saturation at
breakthrough. The mechanism is the same one that shows up in case 1: with the contact line's own
mobility suppressed by the wall's numerical slip, a lower θ cannot fill *behind* the front any
faster; what it does buy is a larger capillary suction into the throats the front has already
reached, which accelerates the leading fingers. Read as a wettability *trend* the sign is
therefore inverted relative to the experiment, and V6b is the named fix. The trapped-gas fractions
(1.0 % vs 1.1 %) are equal within the measurement, i.e. at this Ca and this small a bed the
trapping is set by the geometry, not by the angle.

Note also that the two runs report the same *step* number for breakthrough (6372) purely because
breakthrough is only tested every 118 steps; the times differ because the dt histories differ (the
θ = 30° run spends 81 % of its steps on the Weymouth–Yue cap rather than the capillary one, the
θ = 60° run only 40 %).


### Case 3 — a Zhao-2019-style micromodel (`micromodel_2d.py`)

**Scene, and a geometry the work order asked for that does not exist.** WO-V7 asks for ~60
cylinders at porosity ~0.6 on 128×128×4. Those three are not simultaneously available: a square
array at porosity 0.6 has throats of 0.286 × the lattice spacing, which at 56 posts on this grid is
**3.08 cells**. That array was built and run first, and at 0.121 pore volumes injected it emitted
```
peclet::flow CutcellMG::solveFCG: preconditioner produced non-finite z; returning zero correction
```
four times in a row, with `max|u|` at 93× the inlet velocity. The array was therefore traded from
post COUNT to throat WIDTH: **30 posts of radius 6.5 on a jittered staggered lattice (spacing
21.5 × 21.3, jitter ±1.5), narrowest throat 6.375 cells, porosity 0.712**, which survives. Two
candidate mechanisms for the 3-cell failure, **not separated by this campaign**: (a) the θ-fill
writes a three-cell band into the solid on each side of a throat, so at 3.1 cells the two posts'
bands meet in its middle — the overlap WO-S recorded as making its 4-cell-plate Jurin scene
inconclusive; (b) a Haines jump through a throat whose meniscus radius is ≈1.5 cells is simply
unresolved (a real pore-filling event's local velocity runs up towards σ/μ_l = 25, a thousand times
the inlet velocity here). Separating them wants a θ-sweep at fixed throat width.

**The failure MODE is itself the reportable defect**, independently of the mechanism: the message
goes to stdout, the correction is silently replaced by zero, and the run continues — so a caller's
rule-3b "no capped solve" check passes on a pressure solve that has been returning nothing. Logged
in the examples repo's `ISSUES.md`.

**Runs.** θ = 45° / 90° / 135° at Ca = 1e-3, one process at a time, `OMP_NUM_THREADS=4`,
host-openmp. None reached breakthrough inside a 2400 s budget (that needs ≈18 000 steps; see the
cost section), so all three were stopped at the **same injected volume, 0.10 pore volumes of the
array**, which makes the patterns directly comparable — and the pattern, not the saturation, is
what Zhao's experiment is about. Saturation is then equal by construction (0.1006 / 0.1043 /
0.1043) and carries no information; the discriminators are the box dimension and the front
roughness.

| θ | PV injected | S | D_box | rows reached | front mean | front **std** | deepest finger | clusters | pressure | max\|div\|_proj | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45° | 0.1000 | 0.1006 | 1.582 | **111 / 128** | 10.22 | **7.58** | **22** | **4** | 167/400 | 1.01e-05 | 392 |
| 90° | 0.1043 | 0.1043 | 1.610 | 128 / 128 | 7.62 | 5.35 | 18 | 1 | 179/400 | 8.58e-09 | 423 |
| 135° | 0.1044 | 0.1043 | 1.644 | 128 / 128 | 7.34 | **4.78** | 15 | 1 | 170/400 | 1.53e-09 | 364 |

(Front statistics: for every transverse row the liquid reached, the front is the furthest x it
reached; the table gives the mean, the standard deviation and the maximum of that over rows, plus
how many rows were reached at all and how many connected invaded clusters there are. The script
prints them and `front_stats` is the function.)

**The trend is monotone, it is large, and it is INVERTED against Zhao et al.** Their result is that
strong imbibition displaces *compactly* (cooperative pore filling) and drainage *fingers*. Measured
here at a common injected volume, the **wetting** case has the raggedest front (std 7.58 against
4.78 at θ = 135°, i.e. **59 % rougher**), the deepest finger (22 cells against 15), four
disconnected invaded clusters against one, and **17 of 128 transverse rows never reached at all**,
while the **drainage** case advances as a nearly flat front that reaches every row. The figure on
the gallery page shows it without any statistic being needed.

That is the same mechanism as cases 1 and 2, and this is the case that names it most directly:
**cooperative pore filling requires the contact line to run ahead along the post walls, which is
exactly the motion V6b's Navier slip would enable and which the wall's numerical slip currently
forbids.** With the line effectively pinned, the extra capillary suction at low θ acts only on
the menisci already at the front and accelerates whichever finger is furthest — it *destabilises*
the front instead of stabilising it. The sign of the wettability effect on the pattern is therefore
wrong at rung V6a, and the campaign's three cases fail in the same direction for the same reason.

Two health caveats on this case, both recorded rather than smoothed: the θ = 45° run's
`max|div(open u)|_projected` is **1.01e-05**, three to four orders worse than the other two, and it
spent 29 % of its steps on the Weymouth–Yue cap against 0.4 % and 0.0 % — i.e. the wetting run is
the violent one, consistent with the fingering. And the trajectory is **chaotic**: the same θ = 45°
scene run with 12 OpenMP threads instead of 4 reached `max|u| = 4.12` at step 2268 where the
4-thread run reads 0.632, because reduction order differs at 1e-16 and invasion percolation in a
disordered array amplifies it. The pattern STATISTICS are the reportable quantity here; a
trajectory is not.

### The dt census and the cost, for WO-V9

The capillary limit `0.5 * sqrt((rho_l + rho_g) h^3 / 4 pi sigma) = 0.14175 s` is the ceiling in
every run; the Weymouth–Yue interface CFL displaces it only where the front is fast. Measured
fraction of steps on which the WY cap was the binding one (`dt` re-picked from
`vof_step_limits()` every step):

| case | Ca | WY CFL binds | capillary binds | steps | ms/step (SHARED) |
|---|---|---|---|---|---|
| doublet θ = 45° | 1e-2 | **98.2 %** | 1.8 % | 1056 | 473 |
| doublet θ = 45° | 1e-3 | 1.1 % | **98.9 %** | 6900 | 159 |
| doublet θ = 45° | 1e-4 | 12.9 % | **87.1 %** | 14963 | 121 |
| doublet θ = 135° | 1e-2 | 46.8 % | 53.2 % | 803 | 525 |
| doublet θ = 135° | 1e-3 | 31.8 % | **68.2 %** | 13815 | 149 |
| doublet θ = 135° | 1e-4 | **0.0 %** | **100 %** | 13812 | 137 |
| packing θ = 30° | 1e-3 | 80.9 % | 19.1 % | 8024 | 280 |
| packing θ = 60° | 1e-3 | 40.3 % | 59.7 % | 8024 | 154 |

So the plan's "the capillary limit binds 18 of 18" is **not** what a *driven* pore-scale run looks
like: at the high-Ca end and inside a packing (where a throat manufactures a local jet) the
advective cap is the binding one for most of the run, and only the low-Ca doublet is purely
capillary-bound. **Both limits have to be re-picked every step** — the ten-step re-pick pattern
the earlier VoF pages established is not safe here (E7 recorded the same thing) and the *first*
pick is wrong by 7.1× (finding 2).

**Cost arithmetic, and the one lever there is.** Eliminating the properties in favour of the two
dimensionless groups, the number of steps to advance a front a distance `L` through a channel of
width `w` is

```
    N = L/(U dt)  with  U = Ca sigma/mu_l  and  dt = f sqrt((rho_l+rho_g) h^3/(4 pi sigma))
      ~ L * sqrt(w / (Re * Ca))          (Re = rho_l U w / mu_l)
```

— **independent of sigma, rho and mu separately**. There is no choice of fluid properties that
makes a pore-scale VoF run cheaper at a given (geometry, Re, Ca); the only levers are shortening
the path, accepting a larger Reynolds number, or raising Ca. That is why the doublet at Ca = 1e-4
costs ~1.5e5 steps for a 48-cell branch and was budgeted rather than completed, and it is the
arithmetic behind VOF_PLAN §4's warning. On an idle GPU the doublet ran at **43 ms/step**
(88×4×80, 41 FCG iterations); on the shared machine the same configuration ranged 80–525 ms/step.

**Solver settings that mattered, measured on the doublet:** FCG rtol 1e-11 → 1e-8 costs nothing
in the functionals and takes the iteration count 60 → 41 and the step 54 → 43 ms; the MG level
count has a shallow optimum (3 levels: 81 iterations, 43 ms; 4: 55, 43; **5: 42, 44**; 8: 41, 50)
— the deep hierarchy is launch-latency-bound on a grid this small. Pressure warm-starting moved
nothing (the incremental-rotational `phi` is already small). For grids of this size the
**host-OpenMP build is competitive with and often faster than a contended GPU**: 80 ms/step
(8 threads) against 43 ms idle / 251 ms contended on the doublet, and 307 ms/step (12 threads)
against 1529 ms/step on a 95 %-busy GPU for the 128×128×4 micromodel.

### The campaign's verdict, in one paragraph

Three independent pore-scale problems, three published expectations, and the **same** answer.
The wettability effect is present, large and unambiguous everywhere — a factor of 70 (in fact
0.79 against 0.000) in the doublet's narrow-branch saturation, 59 % in the micromodel's front
roughness, 16 % in the packing's breakthrough time — so the static θ-fill is doing real physics
on real geometry, through cut cells, at density ratio 100, driven by an inflow/outflow pair, with
no capped pressure solve anywhere. But wherever the published result depends on the contact line
advancing **under its own capillary suction** rather than being pushed by the imposed flux, the
sign is wrong: the doublet does not hand the narrow branch the lead as Ca falls, the packing
breaks through *earlier and drier* at the more wetting angle, and the micromodel fingers *more*
under imbibition than under drainage. One mechanism accounts for all three, and WO-V6 already
measured it: contact-line mobility is set by the wall's numerical slip and is ~1/180 of
Lucas–Washburn. **V7 therefore turns V6b from a loose end into the blocking item for every
quantitative pore-scale claim**, and it supplies the number that says where the boundary is: in
the doublet the measured branch ordering flips between Ca = 1e-3 and 1e-2, i.e. where the imposed
velocity crosses the solver's numerical slip velocity, not at the physical Chatzis–Dullien
crossover (Ca ≈ 5e-2 for that geometry).

### What could not be run, and why

* **Ca = 1e-4 in the doublet was budgeted, not completed** (1800 s each, 14 000–15 000 steps,
  12.9 % and 0.0 % of the branch filled). It needs ≈1.5e5 steps per angle; the ordering it reports
  is unambiguous but its breakthrough times are not measured. Both are reported as PARTIAL.
* **No micromodel run reached breakthrough**, so "saturation at breakthrough" is not measured for
  case 3. The comparison was moved to a common injected volume (0.10 PV), which is the right
  comparison for a *pattern* anyway; the breakthrough saturation needs ≈18 000 steps per angle
  (≈2 h each at the rates measured on this machine).
* **Ca = 1e-4 in the packing was not attempted.** At the measured 154–280 ms/step it is ≈8e4 steps
  ≈ 3.5–6 h per angle, and the machine was shared six ways. The work order allowed exactly this
  judgement ("measure ms/step first and state the budget").
* **The ~60-post / porosity-0.6 micromodel is not runnable at all** at this grid (case 3 above).
* **The 3.1-cell-throat failure mechanism was not isolated** (θ-fill band overlap vs unresolved
  Haines jump). Named, not guessed.
* **No MPI runs.** Nothing in this campaign is an MPI gate; the underlying rungs (V5a, V5b, V-BC,
  R2) carry theirs.


## WO-W0 findings (2026-09-02, Opus) — Part III rung W0: the block container + the L1 promotion

Branches `vof-w0` in **flow** and in **core** (the move touches both repos). Backends: CUDA
(RTX 5080, `nvidia-cuda` prefix) and OpenMP (`host-openmp`), both for every gate. MPI np 1/2/4.
`OMP_NUM_THREADS=8 OMP_PROC_BIND=false` throughout. No timings are quoted: the host was carrying
several other agents' jobs (load average ~30) for the whole session.

### Item 6 first, alone — the L1 promotion (gate G5)

`plic.hpp`, `curvature.hpp`, `cutcell.hpp`, `wetting.hpp` moved verbatim to
`core/include/peclet/core/vof/` under `peclet::core::vof`; flow's four headers became **thin
includes + one using-DIRECTIVE**. A directive rather than a list of using-declarations because
qualified lookup into a namespace follows its using-directives ([namespace.qual]) — so both
`peclet::flow::vof::plicVolume` and an unqualified call from inside `peclet::flow::vof` resolve, and
a kernel added in core later needs no edit in flow. That is what keeps the concurrent WO-R2 / WO-V6 /
WO-P01 edits to `advect_wy.hpp` / `cutcell.hpp` / `wetting.hpp` applying unchanged.

**G5: BIT-IDENTICAL, both backends.** The whole VoF ctest battery (`vof_plic`, `vof_advect`,
`vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_surface_tension`, `vof_cutcell`, `vof_wetting`,
`vof_bc`, `vof_collocated`) was run under `ctest -V` before and after the move on each backend, and
the two logs are **identical line for line** after stripping only the build-directory paths: 415
lines of measured output on OpenMP, 416 on CUDA. 10/10 pass either side.

Two build facts the move exposed, both fixed here:
- `tests/kokkos/CMakeLists.txt` handed `${CORE_INC}` to only some targets (not `test_vof_plic`,
  `test_vof_advect`, `test_vof_curvature`). Since `src/vof/*.hpp` now include `peclet/core/vof/…`,
  **any** target that compiles a `src/` header needs the path; the fix is one directory-wide
  `include_directories("${CORE_INC}")` next to `peclet_sibling_include`, not per-target patching.
- `peclet_sibling_include()` resolved the sibling only as `<source>/../core`, which cannot be
  pointed at a git WORKTREE — and worktrees are how this suite runs concurrent agents. It now
  honours `-DPECLET_SIBLING_PECLET_CORE=<repo root>` (unset ⇒ byte-identical behaviour). Every
  build in this WO used `-DPECLET_SIBLING_PECLET_CORE=…/core-w0` / `-DTPX_DIR=…/core-w0`.
- **A pre-existing merge artefact on `main`** (`tests/kokkos_mpi/CMakeLists.txt`, since before
  `518c2a5`): the gated `foreach` list closed with `vof_collocated_mpi)` and was followed by a
  DANGLING argument line `vof_surface_tension_mpi vof_cutcell_mpi vof_bc_mpi)`. CMake cannot parse
  it, so **the whole `tests/kokkos_mpi` project failed to configure** and `vof_bc_mpi` had silently
  been dropped from the build. Repaired on this branch (its own commit) — the MPI battery could not
  be built at all otherwise — and independently on `main` by the concurrent WO-P01 session
  (`9f59d54`) while this WO ran, so the commit was dropped in the rebase. Corroborates that
  session's finding: anyone's MPI numbers taken from `main` between `86192ad` and `9f59d54` came
  from a build that could not have been configured.

### The container (items 1–5)

`src/vof/block_container.hpp` (`VofBox`, `VofRun`/`vofAxisRuns`, `VofBlock`, `VofBlockStats`,
`VofBlockSet`) is MPI-free index math + orchestration; `src/vof/block_exchange.hpp` (`VofPiece`,
`vofBuildPieces`, `VofBlockExchange`) is the gather/scatter, MPI-guarded and a strided copy at
`size == 1`. Solver plumbing: `enable_vof_blocks(seeds)` / `disable_vof_blocks()` /
`advect_vof_blocks(dt)` / `vof_block_stats()` / `vof_block_imbalance()` / `vof_block_census()`,
all bound. Nothing is allocated unless `enable_vof_blocks` runs (**gate G6**: the container is
additive, and the G5 battery above is the same battery, so every existing VoF ctest is
bit-identical with the container in the build).

**Gate P (added — the plan self-check the work order did not ask for and should have).** The
gather/scatter pieces must PARTITION a block's box: every block-local cell written by exactly one
owner's piece. Verified against a 4-way ORB of a 32³ grid for an interior box, a box crossing the
−x seam, one crossing the +x seam, one hanging outside a WALLED axis, and one spanning the whole
grid: **4 / 4 / 4 / 2 / 4 pieces, 4096 / 4096 / 4096 / 3328 / 32768 cells, written-once =
cells, multiply-written = 0** in every case. This is what makes arrival order irrelevant, and hence
the bitwise gates possible; it is cheap and it caught the axis-run enumeration bug (a box longer
than a periodic axis — the extended box of a block that spans the axis — must produce THREE runs,
not throw: the runs partition the range in block-LOCAL index, and revisiting the same global cells
is exactly the periodic ghost).

**Gate G1 — one bubble vs the global field. The work order's gate as written is unattainable, and
the reason is not the container (hard rule 4).** The V1 LeVeque scene (32³, T = 3, 768 steps, CFL
0.25), one sphere as a block, compared against a plain whole-grid `WyAdvector` driven by the same
face field:

| `bubbleEps` | bitwise horizon | max\|d\| over 768 steps | re-centrings | largest box | union volume drift |
|---|---|---|---|---|---|
| 0 (exact support) | **8 steps** | 1.215e-63 | 4 | 32768 cells = **100 %** of the grid | 5.890e-15 |
| 1e-12 (production) | **5 steps** | 4.337e-19 | 12 | 26071 cells = **79.6 %** | 5.890e-15 |

Mechanism: Weymouth–Yue leaves round-off residue in every cell its sweeps touch (documented at V4
down to −3e-35; here down to ±1e-300 and to signed zeros). That residue is part of the global
field's state and is deliberately **not** part of a marker's — the block's ghost policy is "outside
my box it is pure gas", which is the marker model. Once the residue has left the block's box the
global field fluxes it back across the block boundary and the block does not, so the two differ by
the residue and by nothing else (1.2e-63 and 4.3e-19 against a colour of order 1). The alternative
ghost policy — ghosting from the UNION — WOULD reproduce the global field bit for bit, and would let
a neighbouring marker's colour flux in, i.e. coalesce. So the gate is a choice between "bitwise" and
"the whole point of the container".

**Corrected G1**, and it passes: (a) bitwise while the block's box still holds the global field's
support, (b) thereafter max|d| bounded by the residue (measured above), (c) union volume conserved
to 5.9e-15, (d) the union differs from the global field ONLY by the UNPACK_MAX clip (below), and
(e) an exactly bitwise case where the residue does not escape — the G3 translating sphere, **max|d|
= 0.000e+00 over 100 steps**. (e) is the real container gate and is where the "same kernels,
different container" claim is actually tested.

**`bubbleEps` is load-bearing, not cosmetic.** With the extent defined by `C != 0` the box grows
along the bubble's whole WAKE and the block degenerates into the global field: 100 % of a 32³ grid
after 768 LeVeque steps, and on the translating sphere 18081 cells (a box that only ever grows,
`lo_x` pinned at its start) against **9261 cells and a box that genuinely translates, `lo_x` 1 → 20**
at 1e-12. 1e-12 sits 12 orders below any physical colour and ~5 above the residue. What it drops is
accumulated and reported (`vof_block_stats()['discarded']`, measured **−9.5e-17** over the 20-cell
translation of a 524-cell bubble) — measured as the colour of old-box cells falling outside the new
box, NOT as `sum(old) − sum(new)`, which at |sum| ~ 1e2 is 1e-13 of summation rounding and says
nothing (that naive version read −5.7e-14 and was meaningless).

**UNPACK_MAX clips the negative residue.** `C = max_blocks C_block` starting from an empty union
turns a −1e-17 wisp into an exact 0. Over the LeVeque reversal, 32036 union cells differ from the
global field, max|d| **6.245e-17**, and **0 of them** were anything other than that clip (checked
cell by cell: `ref < 0 && union == 0`). This is inherent to TBFsolver's UNPACK_MAX and is recorded
rather than worked around; it is also why the union is compared to the global field separately from
the block.

**Gate G2 — two bubbles, no coalescence. PASS, and it is the rung's raison d'être.** Two spheres
(r = 0.10, 48³) either side of the attracting plane of a solenoidal cellular field sampled as a
DISCRETE CURL of a stream function (so the discrete face divergence telescopes), 216 steps at
CFL 0.20, with the flow reversing at T/2:
- **236 cells carry BOTH markers** at closest approach, and the union DEFICIT `ΣV_marker − ΣC_union`
  peaks at **30.014 cells of shared liquid** — an overlap that is structurally impossible in a
  single colour field;
- per-marker volume drift **1.718e-15** and **2.577e-15**; union volume drift 1.718e-15;
- after the reversal the blocks recover two separated bubbles (**neck colour 0.000e+00**) while the
  single-field control has merged them irreversibly (**neck 7.697e-01**); recovery L1 against the
  initial field **50.72 (blocks)** vs **65.22 (control)**, ratio 1.29.

*One scene fact worth keeping*: put the attracting plane on a cell **face** and the gate is
structurally impossible — each marker stays on its own side of the face and no cell can carry both.
Measured 0 shared cells with the plane at x = 1/2 on a 48³ grid; shifting it to a cell CENTRE gives
the 236 above. The gate is about a cell, so the geometry has to put a cell there.

**Gate G3 — re-centring. PASS.** A sphere translated 20 cells (48³, CFL 0.2, 100 steps), moving
block (margin 3 + pad 2) against a block large enough never to move (`allowShrink = false`, box =
the whole grid):

| `bubbleEps` | moving box | re-centrings (moving / fixed) | first differing step | max\|d\| | volume drift | centroid x |
|---|---|---|---|---|---|---|
| 0 | [0,41)×[14,35)×[14,35), lo_x 1 → 0 | 7 / 0 | none | **0.000e+00** | 2.171e-16 | 0.63579 (exact 0.63542) |
| 1e-12 | [20,41)×[14,35)×[14,35), lo_x 1 → **20** | 7 / 0 | 38 | 3.860e-18 | 4.342e-16 | 0.63579 |

The re-centring copy is by GLOBAL index and therefore exact; at `bubbleEps = 0` nothing is dropped
and the moving block is **bitwise** the fixed one for all 100 steps.

**Gate G4 — distributed, np 1/2/4. PASS, BITWISE.** `tests/kokkos_mpi/test_vof_blocks_mpi.cpp`
runs each scene distributed AND as the same container on one rank on every rank, and compares this
rank's slice of the union bitwise: **first differing step −1 (none) in all three scenes at np 1, 2
and 4**, and the per-marker volumes agree with the single-rank run to **0.000e+00**. The
decomposition cuts the bubbles by construction (the ORB of 2/4 ranks splits the grid across the
seeds), and masters are round-robin by block id, so on the three-bubble scene at np ≥ 2 **one
block's master owns none of its own cells** and its entire state arrives by message.

*The exchange.* The block table (id, box, master) and the flow decomposition are both replicated, so
every rank computes every message size as a pure function of the two — plain `MPI_Isend/Irecv` with
precomputed counts, **no NBX handshake** (it would pay a full round of unexpected-message discovery
to learn what both sides already know; `core::NbxEngine` stays for W1's redistribution, where the
*assignment* changes). Measured per step on rank 0:

| scene | np | gather | scatter |
|---|---|---|---|
| 1 bubble / LeVeque 32³ | 2 | 259 584 B in 1 msg | 41 600 B in 1 msg |
| | 4 | 346 944 B in 3 msgs | 48 640 B in 3 msgs |
| 2 bubbles / cellular 48³ | 2 | 479 232 B in 2 msgs | 67 200 B in 2 msgs |
| | 4 | 539 136 B in 4 msgs | 80 640 B in 4 msgs |
| 3 bubbles / translation 32³ | 2 | 551 376 B in 3 msgs | 82 280 B in 3 msgs |
| | 4 | 399 336 B in 5 msgs | 49 008 B in 5 msgs |

(The gather carries three doubles per EXTENDED-box cell, the scatter one per INNER-box cell, hence
the ~6.5× ratio.) Packing is host-staged (`create_mirror_view_and_copy` per call): W0 is a
correctness rung and a 20³ extended block is ~190 kB; the device-resident packing kernel and the
CUDA-aware path are the W1/W2 optimisation, in the same order `core`'s grid halo grew.

**Load balance (item 5) — the measured imbalance, and W0's known weakness.** Round-robin master
assignment, `vof_block_imbalance()` = max/mean of the per-rank block-cell load:

| blocks | np=1 | np=2 | np=4 |
|---|---|---|---|
| 1 | 1.000 | 2.000 (masters 1,0) | 4.000 (masters 1,0,0,0) |
| 2 | 1.000 | 1.000 (1,1) | 2.000 (1,1,0,0) |
| 3 | 1.000 | **1.337** (2,1 — 11772 vs 5832 cells) | **1.559** (1,1,1,0 — 6859/5832/4913/0) |

So with fewer blocks than ranks the surplus ranks are simply idle, and even with blocks ≥ ranks the
round robin is blind to block SIZE (the np=2 three-bubble case is 1.337 purely because ids 0 and 2
are the two large bubbles). Both are exactly what the weighted-ORB assignment of rung W1 is for;
`vof_block_census()` reports the per-rank breakdown these numbers come from.

### After the rebase onto `main`

The branch was rebased onto `b4c829a` (which had meanwhile taken WO-P01's phase-change rung and the
`movingscene_advect_mpi` gate). Rebuilt and re-run there: **`tests/kokkos` 12/12 pass** (the ten VoF
ctests plus `vof_phase_change` and the new `vof_blocks`), `test_vof_blocks_mpi` **PASSED at np 1, 2
and 4**, and the flow Python module builds. The bit-identity measurement above was taken against the
tree the move was made on (`518c2a5`); the rebase touched no file the move touched, and `plic.hpp` /
`curvature.hpp` / `cutcell.hpp` / `wetting.hpp` came through the rebase without a conflict, which is
the direct evidence that nothing upstream had edited them.

### Open / deferred

1. **The block ghost policy discards the global field's round-off wake** — measured above, bounded
   by the residue, and it is the marker model rather than a defect; but it does mean a block run and
   a global-field run of the same physical problem are not bit-comparable beyond the wake horizon.
   Worth stating in any W1–W3 cross-check.
2. **`bubbleEps` has no fragment/satellite policy behind it yet.** TBFsolver culls fragments
   explicitly; W0 only thresholds the EXTENT and reports what that drops. A real breakup (a detached
   ligament above the threshold) would grow the box instead of becoming its own block — that is W4.
3. **Host-staged packing** (above) — the W1/W2 lever, with no measurement yet because the host was
   loaded all session.
4. **The block container is all-fluid.** `enable_vof_blocks` raises on `set_solid`; the cut-cell
   block (openness-weighted fluxes on the block's own geometry) is W12 together with the NS
   coupling.
5. **Re-centring reallocates a `WyAdvector`.** Deterministic and exact, but a translating bubble at
   `recentrePad = 2` re-centres every ~3 steps and each one allocates 10 fields of the new box. A
   pool, or growing in place, is the obvious W1 cleanup.
## WO-V6 — rung V6 (dynamic contact angle + hysteresis) — DONE 2026-09-02, branch `vof-v6`

Worktree `../flow-v6`, on `main` = `518c2a5`. Commit `91c5e6b` (the rung + the gates) plus the doc
commit at the end. Numbers are given for BOTH backends where they differ; `host-openmp` and
`nvidia-cuda` reproduce each other **digit for digit** on every kernel gate below unless a column
says otherwise.

**Always state the slip length.** Every number below carries its `lambda`; a dynamic-wetting result
without it is not a result (VOF_PLAN §6 — a VoF contact line's numerical slip is proportional to the
cell size, so the imposed angle without an explicit `lambda` is silently grid-dependent).

### What shipped

`src/vof/wetting_dynamic.hpp` — container-free kernels (`coxVoinovAngle`, `vofHysteresisBase`,
`vofDynamicContactAngle`, `vofWallTangent`, `vofContactLineSpeed`) plus the `VofDynamicWetting`
driver (its own scratch views, a `measure` pass and an `impose` pass, and the census). The solver
plumbing in `flow_ibm.hpp` (`setContactAngleDynamic` / `setContactAngleHysteresis` /
`setContactAngleDynamicOff` / `setContactAngleSmoothing` / `setContactAngleClamp` /
`effectiveContactSigma` / `getVofDynamicField` / `buildVofCellVelocity`, the V6 block in
`vofFillGhosts`, the base-angle copy in `applyContactAngle`, and eight new
`ContactAngleDiagnostics` fields), the bindings, `tests/kokkos/test_vof_wetting_dynamic.cpp`,
`tests/kokkos_mpi/test_vof_wetting_dynamic_mpi.cpp` and `tests/study/vof_wetting_dynamic.py`.

**`advect_wy.hpp` and `wetting.hpp` are NOT touched by this rung.** The driver reads the advector
through its public accessors (`colour`, `wallSdf`, `contactAngle`, `wettingNormal`, `cellKind`,
`extent`, `inner`, `ghost`, `size`) and writes only `contactAngle()`, so V5b's fill, its branch
census and passes 2-3 are byte-for-byte the code WO-S shipped. That was deliberate: WO-R2 and WO-W0
are editing `advect_wy.hpp` and `wetting.hpp` concurrently.

Cost when the rung is ON: five extra ghost exchanges of the colour block per band fill (the three
cell-velocity components, then the raw `U_cl` and its validity flag), on top of the three WO-S
already adds for the fluid-only normal. Skipped entirely when no dynamic angle is configured.

### Gates

| gate | measured | verdict |
|---|---|---|
| **G1a the model as arithmetic** (`tests/kokkos/test_vof_wetting_dynamic`, host-openmp AND nvidia-cuda, digit for digit) | `coxVoinovAngle` reproduces `theta^3 = theta_e^3 + 9 Ca ln(1/lambda)` to **4.441e-16** worst over `theta_e` in {30, 60, 90, 120} x `Ca` in {-2e-2, -5e-3, 0, 5e-3, 2e-2, 5e-2} at `lambda = 0.1`, and the identity residual `theta^3 - theta_e^3 - 9 Ca ln(1/lambda)` is <= 1e-14 wherever the clamp does not fire. Monotone in the right direction at every angle: `theta_e = 30/60/90/120` go to **40.408 / 63.411 / 91.576 / 120.896** at `Ca = +1e-2` (advancing RAISES) and **1.000 / 56.149 / 88.366 / 119.091** at `Ca = -1e-2` (receding LOWERS; the 30-degree row is the film-entrainment branch, where the cube goes non-positive and the clamp returns 1 deg) | PASS |
| **G1b the sign convention as geometry** | `vofWallTangent` recovers `theta_app` from `m_f . n_w` to **2.220e-16 rad** and `t_hat` to **1e-14** over 3 apparent angles x 4 azimuths; `vofContactLineSpeed` returns `+0.3` for a flow along `+t_hat` and `-0.3` along `-t_hat`, to 1e-15. An interface parallel to the wall correctly reports "no contact-line direction" | PASS |
| **G1c the model through the solver, KINEMATIC** (32x8x24, flat SDF wall at a QUARTER-integer z = 4.25, liquid slab `x` in [0.5, 16.5) in a periodic box, uniform `u = 0.02`, `theta_e = 60`, **`lambda = 0.1` cells**, `mu_l = 1`, `sigma = 1`) | the slab's two contact lines have opposite `t_hat`, so one advances and one recedes in the SAME field: `U_cl = **+0.020000** / **-0.020000**` (exactly `+/-U`, to 1e-12), `theta_app = 90.0000` on both, imposed **66.4908** (advancing) and **51.6818** (receding) against the host-computed Cox-Voinov 66.4908 / 51.6818. Over all **384** contact cells `max |theta_imposed - CoxVoinov(U_cl)| = **7.105e-15 deg** (gate 1e-10). Diagnostics: dynamic 384, mean imposed 59.0863, mean apparent 90.0000, `max|Ca_cl| = 2.0000e-02`, `max|U_cl| = 2.0000e-02`. Identical to the last digit on nvidia-cuda | PASS |
| **G4a the hysteresis selector as a truth table** (`theta_a = 70`, `theta_r = 50`) | `theta_app = 80 -> advancing, 70.000`; `60 -> PINNED, 60.000` (the apparent angle itself, to 1e-15 — the fill is idempotent, so nothing moves); `40 -> receding, 50.000`; with `Ca = +1e-2`, `80 -> advancing, 72.557`; with `Ca = -1e-2`, `40 -> receding, 44.144`. With hysteresis OFF the base is always `theta_e` | PASS |
| **G4b hysteresis through the solver** (same scene, `Ca_cl = 0`) | `theta_a/theta_r = 120/60` (90 inside the window): all **384** contact cells PINNED, imposed **90.0000** = the apparent angle; `70/50`: all 384 **advancing**, imposed **70.0000**; `130/110`: all 384 **receding**, imposed **110.0000** | PASS |
| **G3-static Jurin on the FIXED scene** (`tests/study/vof_wetting_dynamic.py jurin`; 96x4x112, two 8-cell plates with SEMICIRCULAR ends (a capsule SDF) at QUARTER-integer faces, gap `w = 16`, periodic outer channel `w_out = 64`, `sigma = 1`, `drho g = 3e-3`, ratio 10, `mu_l = 0.2`, Bond 0.768 (gap) / 12.288 (outer), 600 steps to `t = 84.3`, started AT the exact meniscus-ODE equilibrium — the fixed-point protocol) | `theta = 30`: Jurin difference **27.0633**, measured **26.7086** -> **-1.31 %**; `theta = 60`: Jurin **15.6250**, measured **15.4843** -> **-0.90 %**. `dV/V` 5.5e-13 / 3.7e-15, `max|u|` 9.7e-2 / 6.5e-2, 14 / 16 pressure iterations against a cap of 300, none capped, colour in solid cells exactly 0. Both drift slowly DOWNWARD over the run (27.058 -> 26.709 and 15.636 -> 15.484), the sign the V5b angle bias predicts | **PASS** (gate 5 %) |
| **G3-static, the WO-S buoyancy ABLATION** (`jurin_wos`: the same scene with WO-S's `force_z = [dg/(ratio-1), -dg/(1-1/ratio)]`, i.e. zero in the gas and `-dg` in the liquid, which has a NON-zero volume mean) | measured **27.5200** -> **+1.69 %**, and the trace drifts UPWARD (27.059 -> 27.520) exactly as the accelerating-frame mechanism predicts (the frame's `-rho a` cancels part of gravity, so the equilibrium the run relaxes to RISES). 19 pressure iterations. So the non-zero-mean force IS a real perturbation with the predicted sign, but at this horizon it is a **1.7 %** effect, not the -83 % WO-S measured — see finding 3 | (mechanism; both still inside the 5 % gate) |
| **G3-static, ATTRACTION from a FLAT interface** (the same scene, 1500 steps to `t = 217.4`) | level difference **3.7418** against 27.0633, i.e. **-86.2 %** — WO-S's G4 result reproduced (they measured 2.5 against 15.0 in 800 steps). The trace is 0.068 / 0.472 / 1.162 / 1.888 / 2.567 / 3.143 / 3.555 / 3.772 / 3.814 / 3.742 — still rising and decelerating, NOT converged. `dV/V` 1.1e-13, 13 iterations, no cap. The Lucas-Washburn rate for this slot (`dh/dt = w^2 (2 sigma cos(theta)/w - drho g h)/(12 mu h)`) is **4.3 cells per unit time** at `h = 2.7`; the measured rate is **0.024**, i.e. **~180x slower** — see finding 6 | **FAIL as a gate, and the failure is the diagnosis** |
| **G2 spreading drop vs Cox-Voinov** (`spread`; 64x64x40, `D/dx = 24`, wall at a quarter-integer `z = 4.25`, `theta_e = 30`, **`lambda = 0.1` cells**, Oh 0.1 (`mu = 0.4899`), `sigma = 1`, 1200 steps to `t = 239`, released as a HEMISPHERE) | the drop spreads 90 deg -> **44.7 deg** with the contact radius going 12.20 -> 17.44 cells and `Ca_cl = mu (da/dt)/sigma` falling 1.8e-2 -> 4.4e-3. Fit of `theta_app^3 - theta_e^3` against `Ca_cl` over the work order's window `Ca in [1e-3, 1e-1]` (23 points, mean `a` 15.64): slope **64.309** against `9 ln(a/lambda) = 45.471`, i.e. **+41.4 %** (gate 25 %). Restricting to `theta_app < 70 / 60 / 50` gives +47.5 / +89.0 / +41.4 %, so it is not a windowing artefact of the ANGLE; using the EXACT Cox function `g(theta) = int_0^theta (x - sin x cos x)/(2 sin x) dx` instead of `theta^3/9` moves it only to **+36.9 %** (the cube law is 3.5 % off at 90 deg and 1.2 % at 45 deg), so it is not the small-angle approximation either. Tanner: fitted `a ~ t^0.1375` over the late half and `t^0.1282` over the late third, against the law's 0.1. 10 pressure iterations, no cap | **FAIL** — finding 7 |
| **G2b the SLIP SENSITIVITY** (`slipsweep`: the same run at `lambda = 0.02 / 0.1 / 0.5` cells, 800 steps each, IDENTICAL fit window — 15 points — so the windowing enters all three equally. This is the corrected gate: it tests the MODEL, not the scheme's own inner cut-off, because whatever fixed sub-grid contribution the scheme adds cancels in a difference) | slope **47.377 / 43.788 / 39.853** against `9 ln(a/lambda) = 59.372 / 45.016 / 30.699` (-20.2 % / -2.7 % / +29.8 %). The implied effective inner cut-off `lambda_eff = a exp(-slope/9)` is **0.0758 / 0.1146 / 0.1808** cells while the PRESCRIBED `lambda` moves 0.02 -> 0.5, a factor 25 against a factor 2.4. Sensitivity: `d(slope) = -3.588` and `-3.936` where the model prescribes `9 dln(1/lambda) = -14.485`, i.e. **the macroscopic apparent angle responds with 25 % of the prescribed sensitivity** (-75.2 % / -72.8 %, gate 25 %) | **FAIL** — finding 7, and this is the number that names the mechanism |
| **G3-dynamics capillary rise** (`rise`: the G3 scene from a FLAT interface, 800 steps to `t = 113.9`, `theta_e = 30`, Oh 0.05) | final level difference **2.754** (static control) / **2.550** (`lambda = 0.3`) / **2.329** (`lambda = 0.05`) against the equilibrium 27.063 — all **-90 %**, all ASYMPTOTIC (zero overshoot), all limited by the same contact-line mobility as the attraction run above. **The MODEL's effect is nonetheless clean and monotone in the right direction**: the mean IMPOSED angle is 30.00 (static) / **35.48** (`lambda = 0.3`) / **40.86** (`lambda = 0.05`) — a smaller slip means a larger Cox-Voinov correction — and the rise is correspondingly RETARDED, **2.754 > 2.550 > 2.329** (-7.4 % and -15.4 % against the static control). 12 pressure iterations, no cap, `dV/V <= 1.9e-13` | **FAIL on the final height** (the run is not converged, not wrong — see finding 6); the slip ORDERING is the reportable result |
| **G4 hysteresis, drop on an incline** (`incline`: 48x48x32, `D/dx = 16`, wall at `z = 4.25`, `theta_a = 70`, `theta_r = 50`, **`lambda = 0.1` cells**, Oh 0.2 (`mu = 0.8`), the wall kept axis-aligned and GRAVITY tilted, zero-mean tangential force, `Bo_c = cos(theta_r) - cos(theta_a) = 0.3008`, 800 steps to `t = 153.5`) | centroid displacement **+0.228 / +0.307 / +0.650 / +1.104** cells at `Bo/Bo_c = 0.50 / 0.70 / 1.50 / 2.50`, with the pinned fraction of contact cells **90 / 90 / 82 / 74 %** and `advancing = 0` in every row. By the gate's own criterion (a 0.5-cell displacement) that is "pinned below 0.7 Bo_c, sliding above 1.5" — the letter of the gate. **But the mechanism is not reproduced**: displacement/Bo is **0.456 / 0.439 / 0.433 / 0.442**, i.e. the response is EXACTLY LINEAR in the driving with no threshold at all, so the "pinned" rows are creeping, not pinned, and the verdict is an artefact of the fixed observation window. 10 pressure iterations, no cap | letter PASS, **mechanism FAIL** — finding 6 |
| **G5 MPI np 1 / 2 / 4** (`tests/kokkos_mpi/test_vof_wetting_dynamic_mpi`, host-openmp; 16x8x32, a flat SDF wall at a quarter-integer `x = 4.25`, a liquid slab in z whose two contact lines have opposite `t_hat`, uniform `w = 0.02`; the ORB cuts z at np = 2 and xz at np = 4, so both contact lines are cut) | all FIVE dynamic fields — imposed angle, apparent angle, `U_cl`, `Ca_cl`, state — and the theta band fill are **0.000e+00, BITWISE, at np = 1, 2 AND 4**, for both the dynamic-only and the hysteresis configuration, and the state census matches the single-rank reference exactly (144/0/0/0 and 144/144/0/0). Coupled 20-step surface-tension run: np=1 colour and velocity **0.000e+00**; np=2 colour 1.110e-16, velocity 3.123e-17, imposed theta 1.421e-14; np=4 colour 1.110e-16, velocity 2.689e-17, imposed theta **0.000e+00**. Volume drift <= 4.4e-16, colour in solid cells exactly 0, 10 pressure iterations at every np | **PASS** |
| **G6 inertness** (every `tests/kokkos` binary built at this commit vs the same binary built at `main` = `518c2a5`, full stdout `diff`) | **host-openmp: 30 identical, 0 differing**, and **nvidia-cuda: 30 identical, 0 differing** — all 30 binaries, not just the VoF ones. Structurally so: `advect_wy.hpp` and `wetting.hpp` are untouched and every `flow_ibm.hpp` addition is behind `vofDyn_.active()`, which is false unless `set_contact_angle_dynamic` / `set_contact_angle_hysteresis` is called | **PASS** |

### Findings

**1. The work order's `U_cl` sign is wrong, and with it the model is an unstable feedback rather
than a stabilising one.** WO-V6 writes "`U_cl`: … projected on the wall-tangential direction of
`m_f`'s in-wall component `t̂` …, sign positive when the liquid advances (velocity along `−t̂`,
since `m` points into the gas)". The parenthetical is a slip, and it is the load-bearing half.
`m` points into the GAS, so its in-wall part `t̂` points from the LIQUID side towards the DRY side
along the wall, and the liquid advancing over dry wall is motion along **`+t̂`**:

```
U_cl = + u_anchor . t_hat        (shipped)
U_cl = - u_anchor . t_hat        (the work order)
```

The concrete case is gate G1c: a vertical interface with liquid at `x < x0` has `m = +x̂` and
`t̂ = +x̂`, and a flow `u = +U x̂` pushes the liquid onto the dry wall. With the shipped sign that is
`Ca_cl > 0`, Cox–Voinov gives `θ_D > θ_e` (66.49° against 60° at `Ca_cl = 0.02`,
`λ = 0.1 Δ`), and the Young force `cos θ_e − cos θ_D < 0` OPPOSES the motion — viscous bending
retarding the spreading, which is the physics the model exists to represent. With the work order's
sign the same flow reports a receding line, `θ_D = 51.68° < θ_e`, and the Young force ACCELERATES
the advancing line: a positive feedback. The gate that discriminates them is cheap and is what
G1c is built around — a periodic liquid slab in a uniform wall-tangential flow has TWO contact
lines whose `t̂` are opposite, so one must come out advancing and one receding in the SAME field,
and their imposed angles must straddle `θ_e`. Measured: `U_cl = +0.020000 / −0.020000` (exactly
`±U`, to 1e-12), imposed `66.4908° / 51.6818°`, `θ_e = 60°`.

**2. Jurin's law is EXACT for the eps-weighted level integral, at ANY Bond number — it is not a
low-Bond asymptote, and WO-S's G4 was therefore a well-posed gate that genuinely failed.** The
static 2-D meniscus in a slot of width `w` obeys

```
sigma d/dx [ z' / sqrt(1 + z'^2) ] = drho g (z - z_ref),     z'(±w/2) = ± cot(theta)
```

Integrate across the slot: the left side telescopes to `sigma [ z'/sqrt(1+z'^2) ]` evaluated at the
two walls, and `z' = cot θ` gives `z'/sqrt(1+z'^2) = cos θ`, so the left side is `2 sigma cos θ`
exactly. The right side is `drho g w z̄` with `z̄` the MEAN interface height. Hence

```
zbar = 2 sigma cos(theta) / (drho g w)      — Jurin, with no assumption on Bond.
```

The colour-integral read-out `Σ_z (Σ_x C eps)/(Σ_x eps)` **is** that mean, provided every fluid
column of the channel is included (cut wall columns too, which the eps weighting handles). So no
meniscus correction is needed and the gate as WO-S stated it was the right one. Verified
numerically: a shooting solution of the same ODE reproduces `2 sigma cos θ/(drho g w)` to every
digit printed, at Bond 0.768 (the gap) and Bond 12.288 (the outer channel), for θ = 30° and 60°.

**3. WO-S's G4 Jurin failure (−83 %) was an UNCONVERGED ATTRACTION RUN, not a defect and not the
body force — and the fixed-point protocol on the same physics passes to ~1 %.** Two measurements
separate the candidates.

(a) The *fixed-point* protocol — start the run AT the exact static equilibrium (the meniscus ODE
profiles at a common datum, which by construction is the equilibrium for that liquid volume) and
ask whether it holds — gives **−1.31 %** at θ = 30° and **−0.90 %** at θ = 60° over 600 steps.
So the wetting condition, the cut-cell transport and the balanced force reproduce Jurin's law on
this scene.

(b) The *attraction* protocol — start FLAT, the protocol WO-S used — gives **3.74 cells against
27.06 after 1500 steps (−86 %)**, i.e. WO-S's number reproduced on a scene with none of the
features they suspected (8-cell plates instead of 4, rounded ends instead of sharp, quarter-integer
faces, a zero-mean body force). The trace is still rising and decelerating at the end of the run.
The rise is limited by the contact line, not by the scene: see finding 6.

The non-zero-mean body force WO-S used *is* a real defect and its sign is the one the mechanism
predicts (an accelerating frame contributes `−rho a`, which cancels part of gravity, so the
equilibrium the run relaxes towards RISES), but it is **worth +1.69 % over this horizon**, not
−83 %. Ablation, same scene, same fixed-point protocol, 600 steps: zero-mean **26.709** (−1.31 %,
drifting DOWN), WO-S's form **27.520** (+1.69 %, drifting UP). The V6 scene keeps the zero-mean form
(`set_zero_mean_buoyancy`, `force_z = g (rho_bar − rho)`) because it is the correct one and because
the frame velocity grows without bound on a longer run — but the record must say that it is a
second-order term here, and `jurin_wos` is the ablation that says so.

The other two scene changes are kept for the reasons they were made, both of which remain sound:
8-cell plates so the two faces' 3-cell wetting bands no longer overlap and the walk along `n_w`
never crosses the plate's medial surface, and **semicircular ends** (a capsule SDF — the distance
to a segment minus a radius — so `|grad sdf| = 1` everywhere and there is no 90° corner).

**4. A cut-cell initial condition that samples the colour over the WHOLE cell instead of its FLUID
part puts a fake interface down the middle of the liquid, and the curvature cascade turns it into
`|kappa| ~ 3e6`.** Found while building the scene, recorded because it is a trap any cut-cell VoF
initial condition can fall into. `C` is the liquid fraction of the cell's **fluid** volume
(VOF_PLAN §3 rule 2), so a sub-cell sample that lies inside the solid must be excluded from BOTH
the numerator and the denominator. Sampling the whole cell instead gives `C = eps` in a fully
submerged cut column (measured `C = 0.7500` where the answer is 1, in a column with `eps = 0.75`),
which is `0 < C < 1`, hence "mixed", hence a PLIC polygon and a curvature — branch 5 (the
volumetric-paraboloid fallback) returned `|kappa| = 3.23e+06` at that cell against a physical
value of ~0.1, and the run reached `max|u_f| = 788` twelve steps later. The tell is that the bogus
`C` equals the cell's `eps` exactly.

**5. The Gründing et al. (2020) reference curve could not be obtained in this environment, so G3's
dynamic half is gated against the analytic equilibrium and the damping class instead.** The paper
(*AMM* 86:142) is paywalled and its benchmark dataset (TUdatalib) is behind an access wall; both
returned HTTP 403 here. What their comparison actually pins is (i) the final rise height, which is
Jurin's and is available analytically and EXACTLY for this read-out (finding 2), and (ii) whether
the approach is asymptotic or oscillatory, which is a function of the Ohnesorge number and is a
qualitative classification, not a digitised curve. The corrected G3-dynamics gate is therefore:
the final level difference within 10 % of the exact static equilibrium, and the overshoot class
reported and required to be consistent across the slip sweep. The slip SENSITIVITY (their second
axis) is reported as the spread of the rise curve over `lambda`.

**6. THE contact line of this scheme has an anomalously low mobility, and it is what fails G3's
dynamic half and G4's mechanism.** Three independent measurements, all on quarter-integer walls
where WO-S finding 5's pinning artefact is absent:

- *Capillary rise, flat start.* The Lucas–Washburn rate for the G3 slot,
  `dh/dt = w^2 (2 sigma cos(theta)/w − drho g h)/(12 mu h)`, is **4.3 cells per unit time** at
  `h = 2.7` with `w = 16`, `mu = 0.2`, `sigma = 1`. Measured: **0.024**. A factor **~180**.
- *Drop on an incline, below the retention threshold.* The centroid displacement over a fixed time
  is **0.456 / 0.439 / 0.433 / 0.442** cells per unit `Bo/Bo_c` at `Bo/Bo_c = 0.5 / 0.7 / 1.5 /
  2.5` — exactly linear in the driving, with **no threshold**, while 90 % of the contact cells
  correctly report `PINNED`. A pinned contact line creeps anyway, at a rate proportional to the
  force.
- *WO-S finding 4, re-read.* "O(10^3) capillary-limited steps per 30 degrees of contact-line
  travel" is the same number in different units.

The three are consistent with a single cause: the interface slides through the wall band at a rate
set by the scheme's own near-wall velocity, and neither the imposed angle nor the hysteresis
selector controls that rate. **The rung supplies the ANGLE half of Afkhami–Zaleski–Bussmann; the
VELOCITY half — a Navier slip length in the momentum wall condition — is not implemented**, and
these are the measurements of what that costs. Practical consequence, and it must be stated
wherever a dynamic-wetting number from this build is reported: **a case whose answer depends on how
FAR the contact line travels (capillary rise to Jurin, a drop sliding down an incline, imbibition
breakthrough) is not affordable at this mobility** — the G3 scene needs O(1e4) capillary-limited
steps to reach Jurin from flat. A case whose answer depends on the ANGLE at a given contact-line
speed (the G1/G2 measurements, and every quasi-static shape) is fine.

**7. The imposed angle tracks the model to 7e-15; the drop's MACROSCOPIC angle responds to the slip
length with only 25 % of the prescribed sensitivity — because the scheme carries its own inner
cut-off near 0.1 cells that does not move with `lambda`.** This is the honest rating of the rung and
it is the same mechanism as finding 6, measured from the other side.

The composition property of Cox–Voinov says the macroscopic slope should be
`9 ln(a/lambda) = 9 ln(a/Delta) + 9 ln(Delta/lambda)`: the fill supplies the second term exactly
(gate G1c: **7.105e-15 deg** over 384 contact cells), and the resolved hydrodynamics between the
cell size and the contact radius must supply the first. Measured with `lambda` swept over a factor
**25** (0.02 → 0.5 cells) on identical fits:

| `lambda` (cells) | fitted slope | `9 ln(a/lambda)` | implied `lambda_eff` |
|---|---|---|---|
| 0.02 | 47.377 | 59.372 | 0.0758 |
| 0.1  | 43.788 | 45.016 | 0.1146 |
| 0.5  | 39.853 | 30.699 | 0.1808 |

`d(slope)` is **−3.588** and **−3.936** where the model prescribes **−14.485**. So `lambda_eff`
moves by a factor 2.4 while `lambda` moves by 25, and it stays pinned near **0.1 Δ** — an order of
magnitude BELOW the cell size, which is exactly the "grid-dependent mobility" trap VOF_PLAN §6
names, measured. Two candidate explanations were ruled out directly: the small-angle cube law
(replacing `theta^3/9` by the exact Cox function `g(theta)` moves the discrepancy only from +41.4 %
to +36.9 %) and the fit window on the ANGLE (restricting to `theta_app < 70 / 60 / 50` gives
+47.5 / +89.0 / +41.4 %). What is NOT ruled out, and is the first thing to run next, is a grid
refinement at fixed `lambda`: if `lambda_eff` scales with `Delta` the cause is the scheme's
numerical slip and the fix is the momentum-side Navier condition; if it does not, the cause is the
band fill's reach.

Note also that the ABSOLUTE slope is not a well-conditioned gate: the same `lambda = 0.1` run gives
**43.788** over 15 points (800 steps) and **64.309** over 23 points (1200 steps), because
`theta_app^3` against `Ca_cl` is not linear over the whole spreading. **Corrected G2 gate**: gate
the SLIP SENSITIVITY `d(slope)/d ln(1/lambda)` on identical windows (which is what G2b measures and
is insensitive to any fixed sub-grid contribution), and report the absolute slope with its window.

**8. Two smaller things, for the record.**
(i) `tests/kokkos_mpi/CMakeLists.txt` **does not configure at all on `main`** (`518c2a5`): a merge
artefact between the concurrent V5 work orders left a stray argument list after the `foreach`'s
closing paren, and CMake reports `Parse error. Expected "(", got identifier with text
"vof_cutcell_mpi"` at line 45. `vof_bc_mpi` (WO-R's MPI test) was also never registered. Repaired in
this branch's first commit — and INDEPENDENTLY found and repaired by WO-P01 while this WO ran (their
findings entry below traces it to WO-R's `86192ad` and their fix to `9f59d54`); the rebase merged
the two test lists. Recorded twice on purpose: two concurrent sessions each spent time on a
`main` whose MPI test suite could not be configured at all.
(ii) `vof_dynamic_field()` and `contact_angle_diagnostics()` regenerate the band fill, which under
V6 also refreshes the velocity ghost ring (`buildVofCellVelocity` runs the same
`fillVelGhostsKeepOutflow` / `fillVelGhosts` the colour bridge uses). That is a ghost-only side
effect of a diagnostic, in the same family as the `max_open_divergence()` mutation WO-R recorded —
it does not change an inner value, but it is not free and it is not nothing.

### Corrected gates proposed

- **G2** — gate the slip SENSITIVITY `d(slope)/d ln(1/lambda)` over identical fit windows (G2b),
  not the absolute slope; report the absolute slope with its `Ca` window and its `lambda_eff`.
  On the shipped build the sensitivity is 25 % of the model's, and finding 7 says why.
- **G3-dynamics** — the Gründing et al. curve is not obtainable here (finding 5); gate the final
  height against the exact static equilibrium **only on a run long enough for the contact line to
  travel the required distance** (O(1e4) steps for this scene at the measured mobility), and gate
  the slip ORDERING (a smaller `lambda` must retard the rise) on the affordable run, which it does:
  2.754 (static) > 2.550 (`lambda = 0.3`) > 2.329 (`lambda = 0.05`).
- **G4** — gate the creep RATE normalized by `Bo`, which must vanish below `Bo_c`; a displacement
  threshold over a fixed window cannot tell pinning from creep. On the shipped build the normalized
  rate is flat (0.456 / 0.439 / 0.433 / 0.442), so there is no threshold.
- **G3-static** — keep it as written (5 % on the level difference); it passes on the fixed scene
  with the **fixed-point** protocol, and the attraction protocol is a contact-line-mobility gate,
  not a wetting gate.

## WO-P01 findings (Part II, rungs P0 + P1) — 2026-09-02, Opus

Branch `vof-p01`. Backends: host **OpenMP** (`build_ktest_omp`, `OMP_NUM_THREADS=8
OMP_PROC_BIND=false`) and **CUDA** (`build_ktest_cuda`). New files: `src/vof/phase_change.hpp`
(container-free), `tests/kokkos/test_vof_phase_change.cpp`,
`tests/kokkos_mpi/test_vof_phase_change_mpi.cpp`, `tests/study/vof_stefan.py`. `flow_ibm.hpp` gains
one public phase-change section + three inert one-line hooks (`step()` head, `project()` RHS, the
mask refresh before `advanceScalars()`); `scalar_transport.hpp` gains the optional per-cell
Dirichlet mask as two NEW sibling kernels (`scalarMaskStencil`, `scalarMaskRhs`) — no validated
kernel body was touched.

### Gate numbers

| gate | measured | reference / gate |
|---|---|---|
| **K1** `plicArea` vs closed forms | axis-aligned **1** (bitwise), face diagonal **1.4142135623730951** (= √2 bitwise), centred hexagon **1.299038105676658** (= 3√3/4 bitwise) | the three classic unit-cube cross-sections |
| **K1** `plicArea` vs `\|m\|₂ dV/dα` (central FD, 9 normals × 49 volumes) | max abs deviation **1.414e-06** | the FD's own O(δ²) truncation at δ = 1e-6 |
| **K2** one-sided WLS gradient, non-axis-aligned n = (0.5366, −0.7397, 0.4061), 65 samples | **2.1900163572290023** vs exact **2.190016357229001** | rel **6e-16** |
| **P0a** planar regression, 64×4×4, ṁ = 0.02, ρ_l = 1, dt = 1, **1000** kinematic steps | max \|x_Γ − (x₀ − ṁt/ρ_l)\| = **1.243e-14** (4 threads) / **1.776e-14** (8 threads) | gate **1e-12** ✅ |
| **P0a** boundedness / ledger | `C ∈ [0, 1]` exactly (min **0.0**, max **1.0**); **320** cell-crossing clips (20 crossings × 16 columns), \|redistributed\| = **3.2**; interface area **16.0** exactly (A_Γ = 1 per cell to the last bit) | — |
| **P0b** ratio **100**, closed column + balanced sink, 20 COUPLED steps, FCG(400, 1e-12) | u_gas = **0.0099000000000000008**, exact **0.0099000000000000008**, rel **0.0** (bitwise; 1.75e-16 at 8 threads) | gate **1e-10** ✅ |
| **P0b** plateau spread / liquid frame | spread **5.20e-18**; max\|u_liquid\| **5.44e-20**; the interfacial cell's own two faces **−6.13e-20** and **−8.56e-20** (= the liquid velocity) | 1-D exact solution |
| **P0b** source consistency | max\|div(u) − S\| = **1.735e-18**; Σ S = **0.1584** into **16** cells (= ṁ·A·(1/ρ_g − 1/ρ_l)·n_y n_z exactly), fallback cells **0** | projection floor |
| **P0b** solver health | pressure iterations **20 / 400** — NOT capped (rule 3b) | — |
| **P1** Stefan N = 64 (280 steps) | layer **16.18535** cells, exact **16.00000**, **+1.1584 %** | — |
| **P1** Stefan N = 128 (1119 steps) | layer **32.17670**, exact **32.00000**, **+0.5522 %** | — |
| **P1** Stefan N = 256 (4474 steps, layer 64 cells thick) | layer **64.12492**, exact **64.00000**, **+0.1952 %** | gate **0.5 %** ✅ (Malan et al. 2021 report 0.23 %) |
| **P1** observed order | 64→128 **1.069**, 128→256 **1.500**, three-point fit **1.285** | WO expected ~1.5–2 |
| **P1** census, all three N | `C ∈ [0, 1]` exactly, unresolved residue **0**, fallback cells **0**, deficit/excess **0** at the final step | — |
| **MPI P0a** np 1/2/4 (x cut at np 2 and 4; the interface crosses both cut planes during the run) | max\|C_dist − C_ref\| = **0.000e+00** | **bitwise** ✅ |
| **MPI P1** np 1/2/4, walls on ±x, 280 steps with the energy solve | max\|C_dist − C_ref\| = **0.000e+00** | **bitwise** — better than the reduction floor the gate allowed |
| **INERT** ṁ ≡ 0 (every phase-change kernel runs) | max\|C_pc − C_ref\| = **0.0** | bitwise ✅ |
| **INERT** whole `tests/kokkos` battery, phase change never called | the STDOUT of **16** test binaries (`vof_plic`, `vof_advect`, `vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_surface_tension`, `vof_cutcell`, `vof_wetting`, `vof_bc`, `vof_collocated`, `scalar_transport`, `poiseuille`, `poiseuille_ibm`, `cutcell`, `mg`, `ibm`) run from a `main` (518c2a5) worktree and from this branch is **byte-identical** (`diff -rq`, only the marker file differs) | every number, not just pass/fail |
| ctest batteries | `main` **30/30**, this branch **31/31** (host OpenMP) | — |
| CUDA backend | the whole single-rank gate battery passes on `build_ktest_cuda`; MPI np 1/2/4 on `build_kmpi_cuda` is **0.000e+00** on both P0a and P1 | — |

### Which P0b variant, and why

**A CLOSED column: walls on ±x, periodic in y/z, with a prescribed balancing sink plane six cells
from the far wall** (`set_divergence_source`, the new generic RHS hook). Not an outflow — the
variable-density outflow OPERATOR is inconsistent by the density ratio until WO-R2 lands, so an
outlet at ratio 100 would have measured that defect instead of this one. Not the work order's
"periodic box with a balanced sink" either, and the reason is a measurement, not taste: **a periodic
box has nothing to anchor the frame**, the projection's only constraint there is zero net momentum
(`∮ u ρ_f dx = 0`), so the liquid recoils at `u_l = −u_g ρ_g L_g/(ρ_l L_l + ρ_g L_g)` — about
−0.8 % of u_g on this scene. The physics is right and the RELATIVE velocity is still exact, but the
gate "u_gas = ṁ(1/ρ_g − 1/ρ_l) to 1e-10" would then be a statement about a near-cancellation rather
than about the source. With walls the liquid is genuinely at rest (5.4e-20) and all three halves of
the gate — liquid velocity, interfacial-cell velocity, gas plateau — are separate, exact statements.

### Mechanisms and corrections

1. **The work order's `ṁ` sign is inconsistent with its own normal convention, and it is the sign
   that decides evaporation from condensation.** WO-P01 item 1 writes
   `ṁ = (k_l ∇T_l·n − k_g ∇T_g·n)/h_lv` and, two lines later, `n = m/|m|₂` — the PLIC normal, which
   points **into the gas**. The interfacial energy balance is `ṁ h_lv = (q_l − q_g)·n̂` with
   `q = −k∇T` and `n̂` out of the liquid, i.e.
   **`ṁ = (k_g ∇T_g·n − k_l ∇T_l·n)/h_lv`**. Check on the Stefan problem: superheated vapour behind
   the interface gives `∇T_g·n > 0` and `ṁ > 0`. The work order's pairing with the same `n` would
   *condense* a superheated vapour. The shipped kernel (`pcMassFlux`) and the binding docstring
   carry the corrected form and the derivation.

2. **The PLIC polygon area is analytic here, not a finite difference — and the work order's claim
   that the FD "is exact to round-off for a plane" is false.** `V(α)` is the Scardovelli–Zaleski
   **piecewise cubic** in α, so a central difference is exact only inside its linear branches
   (which the grid-aligned planar rung happens to sit in — the trap is that it would have passed
   P0a and then been wrong on every tilted interface). `plicArea` uses
   `A = |m|₂ dV/dα` with `dV/dα` the analytic piecewise quadratic, rearranged to be
   **cancellation-free as n₁ → 0**: `w² − ⟨w−n₁⟩²` is evaluated as the product `n₁(2w−n₁)`, the two
   O(1) terms that differ by O(n₁) are paired in closed form, and the term `⟨w−(n₂+n₃)⟩²` is proved
   identically zero for `w ≤ ½` (since `n₁ ≤ ⅓`). A nearly axis-aligned plane is exactly the
   configuration these rungs run on, so the naive form would have lost `log₁₀ n₁` digits there.

3. **Clip-and-redistribute must be LIQUID-AWARE.** The work order's rule — push the deficit into the
   `−n` face neighbours weighted by `n_d²` — leaves a **permanent negative colour wisp** wherever a
   push target is already empty: measured **−2.476e-6** at N = 64 on the P1 ladder, in a cell that
   had just been emptied and then received the `n_t²` transverse share of a *neighbouring column's*
   deficit. The transverse tilt is not a modelling error: the energy solve's red-black
   Gauss–Seidel updates the two parities in different sweeps, so columns that are identical by
   symmetry differ at ~1e-16 in T, which gives the MYC normal a ~1e-8 transverse component — i.e.
   this is unavoidable upstream and has to be handled in the rule. `pcPushWeights` now takes an
   availability flag per direction (a liquid deficit may only go where liquid remains; a
   condensation excess only where there is room) and renormalizes; if NOTHING can absorb the
   residue it falls back to the unrestricted weights, so **conservation is never traded away** and
   the event is counted (`phase_change_diagnostics()['unresolved']`, 0 everywhere measured). With
   the restriction, `C ∈ [0,1]` exactly and the interface position is **bit-identical** to before.

4. **Everything is a GATHER, never an atomic scatter — that is what buys the bitwise MPI gate.**
   Both the divergence-source deposit (each pure-gas cell scans a 5³ box for donors that named it)
   and the deficit redistribution (each receiver recomputes the donor's WHOLE allocation) run with
   a fixed summation order. The per-cell interface data (`mdot`, `A_Γ`, `n`) is computed on inner
   cells and then halo-exchanged, so the depth-1 and depth-2 consumers read the owner's values;
   `pcZeroDomainGhosts` kills the halo's periodic wrap on non-periodic domain faces, which would
   otherwise import the far side's interface as a phantom donor. Result: **np 1/2/4 bitwise on
   both P0a and P1**, not "at the reduction floor".

5. **`max_open_divergence_projected()` is the wrong diagnostic once a source exists.** By design
   `div(open u) = S`, so it reports `max|S|` (0.0099 on the P0b scene) and looks like a catastrophe.
   The quantity to gate is **`max|div(u) − S|`** (1.7e-18 here). Recorded rather than changed: the
   diagnostic is correct for every run without a source and every recorded number in the repo was
   taken with that meaning.

6. **The first-order component of the P1 error is the Dirichlet-at-the-cell-centre approximation,
   not the gradient fit.** The energy solve pins the WHOLE interfacial cell at `T_sat`, so the
   numerical thermal boundary sits at that cell's centre while the gradient fit measures the normal
   distance from the PLIC **plane** — a mismatch of up to half a cell that changes sign as the
   interface sweeps through a cell. That is consistent with the measured pair of orders (1.07 on
   64→128, 1.50 on 128→256: the oscillating part averages out faster than it refines). The plan
   already names the levers — Aslam quadratic extrapolation (VOF_PLAN §9 item 1) and IHTR — and
   `set_phase_change_thermal(..., r_int=)` ships the Robin form now (default 0 = hard Dirichlet).

7. **Deferred on purpose, with the reason:** per-cell `k(C)` and the consistent `ρ c_p T` geometric
   transport (VOF_PLAN §9 item 6) are the **P3** upgrade, so the energy scalar keeps `add_scalar`'s
   CONSTANT diffusivity. This costs nothing at the Stefan gate — the liquid is saturated and every
   liquid cell is either pinned at `T_sat` by the Dirichlet mask or surrounded by cells that are, so
   `k_l` never enters the energy solve (it does enter `ṁ`, where it is a parameter of
   `set_phase_change_thermal`). It WILL bind at P2 (the sucking interface) and P3.

8. **Scope, enforced with messages rather than assumed:** the collocated grid, an immersed solid,
   and `enable_vof_momentum` each throw from `enable_phase_change` with the reason. The band-extended
   liquid velocity (VOF_PLAN §9 item 3) is not implemented — it is the P3 lever if Scriven needs it;
   at P0/P1 the source sits in pure gas cells and the interfacial cell's face velocity IS the liquid
   velocity, which is the measurement in the P0b row above.

### A defect found in `main`, outside this WO but blocking its MPI gate

`tests/kokkos_mpi/CMakeLists.txt` at `main` (518c2a5) **does not parse**: the WO-R commit
`86192ad` left a duplicated, orphaned continuation line after the `foreach(t …)` list
(`vof_surface_tension_mpi vof_cutcell_mpi vof_bc_mpi)`), so `cmake -S tests/kokkos_mpi` fails with
*"Parse error. Expected \"(\", got identifier with text \"vof_cutcell_mpi\""* and **the entire
multi-rank ctest suite has been unbuildable since that commit** — including `vof_bc_mpi`, which
that commit added and which therefore has never been registered. Fixed on this branch by folding
`vof_bc_mpi` into the list (and adding `vof_phase_change_mpi`); a concurrent session fixed it the
same way in `9f59d54` while this WO ran, and the rebase merged the two lists. Anyone's MPI numbers
taken from `main` between 86192ad and 9f59d54 came from a build that could not have been
configured.
