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

## WO-V6b — the velocity half of the dynamic contact line: Navier slip in the cut-cell wall closure  [Fable design → OPUS; COORDINATE with the velocity-solve session first]

**Why it is now on the critical path.** WO-V6 shipped the angle half of Afkhami–Zaleski–Bussmann
and measured the contact line moving ~180× slower than Lucas–Washburn; WO-V7 then found every
imbibition-side result inverted (pore doublet: narrow branch fills first only at Ca = 1e-2;
packing: the more wetting case breaks through earlier and drier; micromodel: wetting gives the
raggedest front) because cooperative pore filling needs the contact line to run ahead along the
wall — which the no-slip IBM closure forbids except through its ~0.1 Δ numerical slip. Drainage
results are right; imbibition results are qualitative until this lands.

**Design.** In the RS cut-cell IBM closure (`cut_cell_ibm.hpp`: the Dirichlet wall value that
`ibmModifyStencil`/`ibmBuildDiffusion` bake into the momentum operator), replace the no-slip
Dirichlet on the TANGENTIAL velocity by a Navier condition with slip length λ:
`u_t(wall) = λ ∂u_t/∂n`, i.e. with the first fluid value `u_t(d)` at wall distance `d` the wall
value is `u_t(0) = λ u_t(d)/(d + λ)` — a Robin closure that reduces to no-slip at λ = 0 (bit-
identical) and to free-slip as λ → ∞. The wall-NORMAL component stays Dirichlet (impermeable,
moving-body velocity if any). `λ` is a solver parameter (`set_wall_slip_length(lambda_cells)`,
default 0), the same λ the angle model uses (`set_contact_angle_dynamic` takes it — unify: one λ).
Where the IBM closure is applied at the staggered face DOFs, decompose per component: the
component parallel to the wall normal is normal, the other two are tangential (use `∇sdf` per DOF;
for a DOF whose normal is oblique, project).

**Gates.** (1) Couette flow over a flat SDF wall with λ: the analytic slip profile
`u(y) = U (y + λ)/(H + 2λ)` to 1e-10 at every N (Poiseuille-type exactness on a quadratic-free
profile); (2) λ = 0 bit-identical to today's single-phase regression, the Z&H drag and every VoF
ctest; (3) WO-V6's G2 slope test: the macroscopic Cox–Voinov slope now responds to λ with the
model's sensitivity (d(slope)/d ln λ within 25 % of −9); (4) capillary rise dynamics: the
Lucas–Washburn early-time law `h² = (σ w cos θ/(3μ)) t` (slot of width w) within 20 % on the
repaired WO-V6 plate scene with λ = 0.05–0.3 Δ (state the λ); (5) WO-V7's pore doublet at
Ca = 1e-3, θ = 45°: the narrow branch fills first (the Chatzis–Dullien verdict flips back); (6) MPI
np 1/2/4 bitwise on (1), floor on (4).

**Coordination.** `cut_cell_ibm.hpp` and the momentum operator assembly are being edited by the
velocity-solve session (momentum residual stop, velocity MG, telescoping — all on main since
2026-09-02 evening). Before starting: `git fetch`, read their CLAUDE.md paragraphs, and keep the
change to the closure VALUE (the Dirichlet datum) rather than the stencil structure so it merges.

---

## WO-P3g — a second-order interfacial energy operator (the P3 closure)  [Fable design → OPUS]

Six attempts at the Scriven 1 % gate (WO-P23 … P3f) each retired one candidate; P3f's verdict
is that the remaining 1.0–1.5 % is the RESIDUE OF A CANCELLATION between three first-order
errors — (F1) the GFM Dirichlet row is a two-point flux (first order in the row; −17 % at a
2.4-cell thermal layer, −5 % at 8 cells on a FLAT interface), (F2) the one-sided fit measures
distance to the tangent PLANE, not the sphere (+6 % at R = 20, order 0.98), and (F3) the
Dirichlet overwrite destroys −0.7 % (Ja 0.5) / −4.3 % (Ja 2) of the latent heat, one-signed —
so fixing any one alone makes the gate worse (measured), and the mesh ladder is anti-convergent
(the compensating term is the O(h/R) one). This rung fixes all three together, in the ENERGY
OPERATOR, gated by P3f's own instruments.

**Design.**
1. *ṁ from the operator's own flux, not from a separate fit.* Define
   `ṁ A_Γ h_lv = Q_liq − Q_gas`, where `Q_liq` is the discrete heat flux the energy operator
   actually transfers from the liquid cells into the interfacial cell's Dirichlet row (the sum
   of the row's off-diagonal contributions, evaluated with the converged T of the step) and
   `Q_gas` likewise. Energy removed and mass produced are then the SAME discrete quantity, so
   F3 disappears by construction (the enthalpy budget's residual is zero to round-off). The
   quadratic-fit ṁ stays as a diagnostic (`phase_change_diagnostics()['mdot_fit']`).
2. *A second-order GFM row.* Replace the two-point ghost-fluid row (cell value + `T_sat` at
   distance `θ h`) by the Gibou–Fedkiw (JCP 176:205, 2002) quadratic form: for a liquid cell `i`
   whose neighbour `i+1` across the face is interfacial, use `T_{i-1}`, `T_i` and `T_sat` at
   `θ h` to build a second-order one-sided Laplacian row (the coefficients are the standard
   `2/((1+θ)θ h²)`-type family; write them out and gate with a 1-D quadratic profile that must
   be reproduced EXACTLY). The row is non-symmetric — the scalar solver's RB-GS smoother does
   not need symmetry; verify convergence on the planar rungs. Where `θ` is tiny (< 0.1), fall
   back to the neighbour-behind (`i-1`) row so the coefficients stay bounded (Gibou's rule).
3. *Curvature-consistent distances.* The distance from a cell centre to the interface along a
   grid axis is the plane distance corrected by the interface curvature at the lateral offset:
   `θ_c h = θ h + κ ρ²/2` (sign with `κ` positive for a gas bubble seen from the liquid), with
   `κ` from the V3 cascade (`kappa` field; PV fallback where HF fails), `ρ` the lateral offset of
   the axis line from the cell's interface centroid. Use it in the GFM row (item 2) and in the
   diagnostic fit (P3f's `fit_curvature` option, now fed by the cascade instead of a prescribed
   κ).
4. *Interfacial cells that are almost pure.* Keep P01's per-face GFM formulation for which
   faces get a row; a cell with `C` within `pcEffPureEps` of 0/1 is pure for the operator, and
   its enthalpy is carried (P3f's `carry_conserve` deposit, which is bitwise conservative on the
   planar scene) — with item 1 in place the carry no longer double-counts, which is what made it
   worse in P3f.

**Gates (P3f's instruments, in this order).** (a) 1-D: a linear and a quadratic temperature
profile against a fixed plane at every `θ ∈ {0.05…0.95}` — the new row reproduces the
quadratic EXACTLY (round-off) and the flux `Q_liq` equals the analytic one to 1e-12; (b) the
2×2 probe of P3f (`{sphere, plane} × {Scriven, linear}`), 128³, R = 6/10/14/20: the area-averaged
ṁ from item 1 within 0.5 % at R ≥ 10 and **order ≥ 1.8 in h/R** (P3f: +19…+6 %, order 0.9), and
`−q_gfm/E_lat` within 0.5 % at an 8-cell thermal layer (P3f: −5.1 %); (c) the enthalpy budget on
Scriven: residual ≤ 1e-10 of `E_lat` per step, both Ja; (d) **Scriven 128³ Ja 0.5 and 2**:
`max|ΔR|/R` < 1 % over the last half AND `|β_eff/β − 1|` < 1 %, then Ja 10 once, then the
96³/128³/192³ ladder at fixed R/L must now CONVERGE (order ≥ 1); (e) planar rungs: P1 and P1′ at
the noise floor, P2 order ≥ 2 retained (record every digit that moves; P0a/P0b are
ṁ-prescribed and must be byte-identical); (f) MPI np 1/2/4 bitwise on P0a/P1; `tests/kokkos`
green; every VoF ctest bit-identical with phase change off; the new operator behind an option
that becomes the default on a passed (d).

**Deliverables.** The row family + flux bookkeeping in `src/vof/phase_change.hpp` /
`scalar_transport.hpp` siblings, the κ plumbing, `tests/kokkos/test_vof_phase_change.cpp` gates
(a)/(b)-planar, `tests/study/vof_scriven.py` (b)/(c)/(d), findings, CLAUDE.md.

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

## WO-V9 findings — the VoF performance profile, and the one lever the numbers justify — 2026-09-04, Opus

Branch `vof-v9`, worktree `../flow-v9`, from `origin/main` at `40fc1b7`.

**Machine state: IDLE.** GPU 0 % utilisation, 15 MiB of 16303 MiB in use, no other compute process,
load average 0.5 at the start of the session; every build finished before the first timing run and
none ran during one. This is the measurement window WO-W12 and WO-V7 both asked for, and it
**inverts two of their conclusions** (below). Backend rows marked *CUDA* are the RTX 5080
(`nvidia-cuda`), rows marked *host* are `host-openmp`. Timing runs used the full
`OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads` as the work order specifies; the MPI
runs used **one pinned thread per rank** (`OMP_NUM_THREADS=1`, `mpirun --bind-to core`) for the
reason in item 3. Every case records its max pressure iterations against its cap; **no run in this
entry touched a cap** (rule 3b).

### What shipped

* **`set_vof_timing(True)` / `vof_timing()` / `reset_vof_timing()`** — per-stage and per-kernel
  timers around the whole VoF pipeline (`src/vof/advect_wy.hpp` `WyAdvector::Timing`,
  `src/vof/curvature_field.hpp` `VofCurvature::Timing`, `src/flow_ibm.hpp`
  `IbmSolver::VofTiming`). Every stage boundary calls `Kokkos::fence()` **only when armed** — the
  rule `phaseTick()` already applies to the step's three coarse phases, because on a device backend
  a boundary that does not fence bills queued work to whichever stage next reads the clock. Off,
  the cost is one predictable branch per stage and no fence.
* **`set_vof_worklist(bool)`** — the advector's reconstruction-pass compaction, which had no
  binding at all (item 2 could not otherwise be measured from Python).
* **`set_vof_curvature_worklist(bool)`, default ON** — the lever (see item 5): the V3 cascade over
  a compacted list of the interfacial cells instead of over the whole inner region. The per-cell
  bodies of tier 1/2 and tier 3 were lifted VERBATIM out of their kernels into
  `curvHeightCell` / `curvFallbackCell` (`src/vof/curvature_field.hpp`) so the dense and the
  compacted kernel share one body; the only substitution is `i` as a parameter instead of
  `L3(x,y,z,e)`, and tier 3's `L3(x+ox,y+oy,z+oz,e)` → the identical `i + ox + oy*sy + oz*sz`.
* **`tests/kokkos/test_vof_timing.cpp`** (ctest `vof_timing`, both backends) — gates that the
  instrument and both compactions are inert: timers OFF vs ON and worklists ON vs OFF are
  `max|d| = 0.000e+00` on C, u, v, w and P over 30 steps of a surface-tension + momentum-consistent
  run, on the all-fluid path AND on the cut-cell path.
* Study scripts `tests/study/vof_profile.py` (the five-case profile, both compaction ablations),
  `tests/study/vof_profile_mpi.py` (item 3), `tests/study/vof_orb_weights.py` (item 4),
  `tests/study/vof_block_packing.py` (item 4b).

### Item 1 — the profile. The answer is that VoF is not where the time goes; the PRESSURE SOLVE is.

Percentages are of the timed step. 20 timed steps after 5 warm on CUDA; 10–15 on host. The packing
scenes reuse the gallery drivers' physics, grids, closures and solver settings verbatim; their beds
are a deterministic RSA packing of the same grain radius and count rather than the pages' DEM
deposit, because importing `dem` and `flow` into one interpreter mixes two Kokkos backends
(`suite/CLAUDE.md`) — the profile depends on the cut-cell count, not on which loose packing it is.

**CUDA (idle GPU):**

| case | grid | ms/step | press | **VoF total** | colour adv | mom adv | curvature | CSF | phase ch. | recon | fluxes | sweeps | **g=3 fill** | predictor | mom solve | **projection** | remainder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hysing 1 | 64×4×128 | **31.2** | 15/600 | **4.67 %** | – | 4.07 | 0.54 | 0.07 | – | 0.45 | 0.14 | 0.07 | 1.55 | 0.23 | 6.22 | **88.20** | 0.67 |
| E6 trickle | 48×48×96 | **34.3** | 20/400 | **18.45 %** | – | 13.91 | 4.48 | 0.07 | – | 0.53 | 0.35 | 0.25 | **6.24** | 0.32 | 8.98 | **71.22** | 1.03 |
| E7 packed | 64×64×160 | **98.7** | 18/600 | **13.13 %** | – | 7.91 | 5.16 | 0.06 | – | 0.30 | 0.19 | 0.17 | 3.05 | 0.23 | 5.25 | **80.91** | 0.49 |
| droplet | 128³ | **212.9** | 13/500 | **8.68 %** | 0.78 | – | **7.82** | 0.09 | – | 0.30 | 0.15 | 0.05 | 0.04 | 0.30 | 4.67 | **85.99** | 0.36 |
| Scriven | 96³ | **69.7** | 15/600 | **8.86 %** | 3.79 | – | 0.00 | 0.00 | 5.07 | 1.54 | 1.23 | 0.09 | 0.15 | 0.35 | 5.73 | 43.48 | **41.58** |

**host-openmp (8 threads):**

| case | ms/step | **VoF total** | colour/mom adv | curvature | phase ch. | recon | fluxes | sweeps | **g=3 fill** | predictor | mom solve | **projection** | remainder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hysing 1 | **86.4** | **8.16 %** | 6.99 | 0.93 | – | 0.40 | 0.32 | 0.35 | 1.14 | 0.78 | 7.87 | **80.83** | 2.36 |
| E6 trickle | **168.6** | **40.19 %** | 33.87 | 6.23 | – | 0.75 | 0.55 | 0.33 | **15.91** | 0.72 | 10.13 | 47.25 | 1.71 |
| E7 packed | **694.9** | **26.48 %** | 22.09 | 4.32 | – | 0.54 | 0.39 | 0.39 | **10.48** | 0.46 | 6.40 | **65.81** | 0.86 |
| droplet | **1501.6** | **3.78 %** | 2.14 | 1.52 | – | 0.60 | 0.33 | 0.25 | 0.05 | 0.76 | 11.57 | **83.23** | 0.65 |
| Scriven | **526.4** | **12.52 %** | 8.59 | 0.00 | 3.93 | 3.77 | 1.51 | 0.35 | 0.47 | 0.89 | 6.37 | 35.44 | **44.79** |

Five things this table says, in order of how much they change what to do next:

1. **The pressure projection is 43–88 % of every step, and VoF is 4.7–18.5 % (CUDA).** Every
   candidate VoF lever in the work order is therefore competing for at most a fifth of the step,
   and on the two flagship free-surface cases (Hysing, droplet) for less than a tenth. *The
   optimisation target of the VoF campaign is the pressure solve, not the VoF kernels.* On Hysing 1
   the projection is 27.4 ms of a 31.2 ms step on a **32 768-cell grid** — that is 14 Chebyshev
   iterations' worth of launch latency, not arithmetic, and it is the same launch-latency floor
   WO-V7 found in its MG-depth sweep.
2. **Inside VoF, the two big items are the curvature cascade and the g = 3 FILL — never the
   advection arithmetic.** Reconstruct + fluxes + sweeps together are 0.3–2.9 % of the step in
   every case. The curvature is up to 7.8 % (droplet) and the fill up to 6.2 % (trickle, CUDA) /
   15.9 % (trickle, host).
3. **What `k_exchange` measures on a packing is not communication — it is the WETTING BAND FILL.**
   `vofFillGhosts` on the colour field is one raw exchange, then (with a solid) the fluid-only
   Youngs normals + three more raw exchanges + the θ pass + the three-pass solid-band fill + a
   second raw exchange — five raw exchanges and two kernel passes, three times a step (once per
   sweep), plus the momentum advector's 18 raw sibling exchanges. That is why the number is 0.04 %
   on the all-fluid droplet and 6.2 % on the trickle bed at a THIRD of the grid size.
4. **The Scriven step is 42–45 % "remainder", and that is the energy scalar** (`add_scalar("T")` at
   200 sweeps in `advanceScalars`), not the phase-change kernels, which are 5.1 % (CUDA). Anyone
   optimising the phase-change ladder should start at the temperature solve.
5. **The dt census is not the plan's.** Every surface-tension case here is capillary-bound in
   **100 %** of its steps (Hysing, trickle, packed) and the Scriven case is WY-CFL-bound in 100 %
   of its steps. WO-V7's driven pore-scale runs were 0–98 % WY-bound depending on Ca; VOF_PLAN §4's
   "capillary binds 18 of 18" holds for the *undriven* cases and the WY cap takes over as soon as
   the run manufactures a local jet (a throat, or a phase-change interface velocity).

**The host-vs-CUDA verdict INVERTS WO-V7's.** WO-V7 recorded "host-OpenMP is competitive with and
often faster than a contended GPU" (80 ms/step at 8 threads against 43 idle / 251 contended). On an
idle GPU there is no contest at any of these sizes:

| case | CUDA ms/step | host-openmp (8 thr) ms/step | CUDA speedup |
|---|---|---|---|
| Hysing 1 64×4×128 | 31.2 | 86.4 | **2.8×** |
| E6 trickle 48×48×96 | 34.3 | 168.6 | **4.9×** |
| E7 packed 64×64×160 | 98.7 | 694.9 | **7.0×** |
| droplet 128³ | 212.9 | 1501.6 | **7.1×** |
| Scriven 96³ | 69.7 | 526.4 | **7.6×** |

WO-V7's statement should be read as what it was: a measurement of a GPU carrying five other jobs.
The smallest, most launch-latency-bound case (Hysing, 32 k cells) is where the host comes closest,
which is the expected direction.

### Item 2 — `useWorklist` on the ADVECTOR is bit-neutral, and it is a PESSIMISATION on 9 of the 10 (case, backend) pairs measured

Bit-neutral: gated in `test_vof_timing` (`max|d| = 0.000e+00`, all-fluid and cut-cell). The gain
(`k_reconstruct`, ms/step):

| case | CUDA ON | CUDA OFF | CUDA gain | host ON | host OFF | host gain |
|---|---|---|---|---|---|---|
| Hysing 1 | 0.139 | **0.091** | −53 % | 0.342 | **0.269** | −27 % |
| E6 trickle | 0.177 | **0.075** | −136 % | 1.195 | **0.664** | −80 % |
| E7 packed | 0.292 | **0.166** | −76 % | 3.612 | **2.097** | −72 % |
| droplet 128³ | 0.647 | **0.589** | −10 % | 8.996 | **5.625** | −60 % |
| Scriven 96³ | **1.074** | 2.181 | **+103 %** | 19.821 | **16.395** | −21 % |

(negative = the compaction is slower than the guarded dense pass). The one win is Scriven on CUDA,
and it is the one case whose mixed set is *scattered*: `enable_phase_change` turns the wisp guard
OFF, so `wyIsMixed(c, 0)` accepts the round-off residue the deposit leaves through the whole
domain, and a dense warp then carries one or two diverging lanes. Everywhere else the mixed cells
lie in coherent runs on a surface, the guarded dense pass is already coherent, and the compaction
only adds a `parallel_scan` + a host-visible fence per sweep.

**Whole-step effect: ≤ 1.6 % anywhere** (Scriven 70.7 → 69.4 ms with it on; everything else inside
the case's own repeat spread). **The default was left ON**, because flipping it would change nothing
measurable on four cases and cost 1.6 % on the fifth; the switch now exists and the table above is
the reason to use it. The finding worth carrying is the *mechanism*: compaction pays for warp
coherence, not for skipped cells, so the predictor is how SCATTERED the interfacial set is, and
neither the mixed-cell count nor the mixed fraction predicts it (Scriven and the droplet have the
same order of both and behave oppositely).

### Item 3 — the g = 3 halo under MPI: its share of the step is FLAT in np, and the PARIS trick is not indicated

**First, the trap that makes any np > 1 number on this machine meaningless unless it is avoided.**
A single-GPU node cannot measure MPI scaling: at np = 2 and 4 on CUDA *every* timer scales up by
the same factor (Hysing 36.5 → 516 → 1035 ms/step; the exchange share 0.89 → 1.88 → 1.89 %), which
is the signature of the CUDA contexts time-slicing one device, not of communication. And on
host-openmp, `mpirun --bind-to none` with `OMP_PROC_BIND=spread` makes np = 2 **twelve times
SLOWER** than np = 1 (packed 1.42 → 17.3 s/step at identical pressure iterations), because both
ranks' OpenMP pools spread over the same 48 cores. **One pinned thread per rank
(`OMP_NUM_THREADS=1 mpirun --bind-to core`) is the only configuration on this host that measures
anything**, and it scales properly:

| case | np | ms/step | speedup | **g=3 fill ms/step** | **share of step** | pressure |
|---|---|---|---|---|---|---|
| Hysing 1 64×4×128 | 1 | 131.9 | 1.00 | 2.343 | 1.78 % | 15/600 |
| | 2 | 74.1 | 1.78 | 1.468 | 1.98 % | 15/600 |
| | 4 | 46.7 | 2.83 | 1.035 | 2.22 % | 15/600 |
| E7 packed 64×64×160 | 1 | 4043 | 1.00 | 522.9 | **12.93 %** | 17/600 |
| | 2 | 2161 | 1.87 | 277.8 | **12.86 %** | 17/600 |
| | 4 | 1374 | 2.94 | 169.5 | **12.34 %** | 17/600 |

**The verdict is unambiguous and it is a NEGATIVE result for the PARIS trick.** On the case where
the g = 3 stage is expensive (the packing, 12.9 % of the step) its share does not move at all from
np = 1 to np = 4 — it *falls* slightly. A stage whose cost is the same fraction at np = 1, where
there is no message at all, is not paying for message depth: it is paying for the per-cell FILL
work (item 1 point 3 — the wetting normals, the θ pass and the three-pass band fill, run three
times a step). PARIS's partial-column-sum halo-2 construction halves the DEPTH of the exchanged
band; here that would attack a term that is at most the 1.8–2.2 % the all-fluid case shows, and
would leave the 12.9 % untouched. **Not worth it.** The cheap lever on that 12.9 %, if anyone wants
it, is to notice that the wall geometry (`n_w`, the θ field, the fluid-only Youngs normals of the
*wall* rows) is a property of the SOLID and not of C, and is being rebuilt three times per step for
the three sweeps.

### Item 4 — the interface-weighted ORB. The W12 swarm has nothing to rebalance, and `rebalance_by_weights` CRASHES on the one that does

(a) **The W12 64-bubble swarm is a 4×4×4 lattice**, i.e. exactly the configuration a cell-count ORB
already balances perfectly. Measured at np = 4, 64³, that swarm's interfacial load is
`[4574, 4574, 4574, 4574]`, **imbalance 1.0000 before and 1.0000 after** the interface-weighted
redistribution, and ms/step 377.6 → 380.7 (+0.8 %, i.e. the migration's cost and nothing else). *The
lever cannot be measured on the scene the work order names.* The scene it is for is a swarm that has
not spread — 64 markers in the bottom half (`--layout plume` in `tests/study/vof_orb_weights.py`),
where the plain ORB reads:

| scene (np = 4) | mixed cells per rank | imbalance (mixed) | imbalance (cells) |
|---|---|---|---|
| W12 lattice, 64³ | 4574 / 4574 / 4574 / 4574 | **1.0000** | 1.0000 |
| plume, 64³ | 2652 / 2785 / 3162 / 3106 | **1.0806** | 1.0000 |
| plume, 48³ | 3788 / 1718 / 2204 / 773 | **1.7862** | **1.7778** |

(the 48³ row is also a reminder that the *cell-count* ORB is itself 1.78 out of balance on a grid
whose axis is 16·3 — the aligned-ORB power-of-two snapping of `docs/DECOMPOSITION_AND_MULTIGRID.md`
§2, not a VoF matter.)

(b) **OPEN DEFECT, and it is why there is no "after" column: `rebalance_by_weights` corrupts the
heap when the weighted ORB actually MOVES the partition on a VoF run.** Reproducer:

```
PYTHONPATH=<build_mpi_omp> OMP_NUM_THREADS=1 mpirun -np 4 --bind-to core \
  python tests/study/vof_orb_weights.py --n 48 --steps 8 --warm 3 --w 8 --layout plume
```

fails immediately after the `rebalance_by_weights` call with `free(): invalid pointer` /
`malloc(): unsorted double linked list corrupted` inside `_flow…so`, or SIGSEGV at address 0x20,
on 2 of 4 ranks. It is **pre-existing**, not a WO-V9 regression: it reproduces identically on a
binary built before this branch's curvature change. It is conditional on the partition MOVING —
the same call with the same weights on the *lattice* swarm (whose weighted and unweighted
partitions coincide) completes, and a minimal 32³ VoF + surface-tension reproducer with a
half-domain or a spherical-shell weight field completes too. Suspects, in the order worth checking:
`IbmSolver::redistribute`'s `scatterPadded` memcpy is sized from `newDec.block(rank)` while the
field it writes was reallocated by `allocateBlock`, and step 5 re-runs `setSolid` (which rebuilds
the VoF block) *after* the fields have already been scattered once.

(c) **The block gather/scatter packing, device vs host, on the QUIET machine — W12's open number.**
64-marker swarm, 48³, kinematic `advect_vof_blocks`, min of 3 alternated repetitions:

| staging | ms/step (min of 3) | mean ± sd |
|---|---|---|
| **device** | **20.205** | 20.379 ± 0.173 |
| host | 37.351 | 37.518 ± 0.207 |

**1.849×**, with the marker volumes agreeing to `|d| = 0.000e+00`. This **confirms W12's quiet 1.69×
and refutes its contended 0.80×**: the device-resident packing is the right design and the
inversion W12 saw was the contention it said it was.

### Item 5 — the lever that IS justified, and what it measured

Of the work order's four candidates, the profile rules out three: batching the per-piece block
kernels (the block path is not in any of the five cases, and its packing is already 1.85× ahead —
item 4c), a fused reconstruct+flux kernel (reconstruct + fluxes together are 0.3–2.9 % of the
step), and the PARIS halo trick (item 3). The fourth — **the curvature cascade over a worklist
instead of the band** — is the one the numbers justify, and it is what shipped.

The instrument first: the cascade's own passes, CUDA, ms/step.

| case | pass | dense | compacted |
|---|---|---|---|
| droplet 128³ | compaction scans | – | 0.323 |
| | PLIC planes | 0.230 | 0.149 |
| | tier 1/2 height functions | 0.384 | 0.164 |
| | **tier 3 PV paraboloid** | **15.588** | **7.800** |
| | branch census | 0.036 | 0.037 |
| | **curvature total** | **16.440** | **8.659** |

**The whole curvature stage is the tier-3 PLIC-volumetric fallback** (95 % of it on the droplet,
80 % on the packed column, 96 % on the ctest's 24³ blob) — not the height functions, not the plane
pass. That is the V3 rung's own documented behaviour (the fallback fires on 19 % of interfacial
cells at D/Δ ≈ 40 and on 100 % below 5 cells/diameter) meeting the fact that the fallback is a 5³
Wendland-weighted 6-parameter fit and everything else is a column sum.

Measured gain, CUDA, `set_vof_curvature_worklist(True)` (the new default) against `False`:

| case | curvature dense | curvature compacted | **curvature gain** | step OFF | step ON | **step gain** |
|---|---|---|---|---|---|---|
| droplet 48³ | 7.353 | 5.300 | **1.39×** | 38.15 | 36.14 | **−5.3 %** |
| droplet 64³ | 7.160 | 5.355 | **1.34×** | 48.16 | 46.33 | **−3.8 %** |
| droplet 80³ | 9.788 | 5.495 | **1.78×** | 74.35 | 69.75 | **−6.2 %** |
| droplet 96³ | 12.788 | 5.579 | **2.29×** | 101.29 | 93.96 | **−7.2 %** |
| droplet 112³ | 13.164 | 8.581 | **1.53×** | 152.57 | 148.04 | −3.0 % |
| droplet 128³ | 16.424 | 8.648 | **1.90×** | 212.60 | 204.78 | **−3.7 %** |
| E7 packed 64×64×160 | 5.023 | 5.937 | 0.85× | 98.95 | 99.92 | +1.0 % |
| E6 trickle 48×48×96 | 1.510 | 1.819 | 0.83× | 34.37 | 34.69 | +0.9 % |

and on **host-openmp** it is a small loss everywhere (droplet 128³ curvature 33.4 → 40.7 ms, of
which 5.4 ms is the two scans), invisible at the step level (1845.9 vs 1800.2 ms, inside that
case's own spread).

**Bit-identity**: `max|d| = 0.000e+00` on C, u, v, w, P over 30 steps, all-fluid and cut-cell,
both backends (ctest `vof_timing` T3). It is a re-ordering of independent per-cell work and nothing
else — which is exactly why the two kernels were made to share one lifted body rather than being
written twice.

**Why the default is ON despite two losing rows.** The wins are 1.3–2.3× on the curvature and
−3.0 to −7.2 % on the STEP across six resolutions of the free-surface family, reproducible and
monotone in the amount of fallback work. The losses are +0.9 % and +1.0 % of the step on the two
packings — at or inside those cases' own repeat spread (the packed case's five independent 20-step
repeats in this session span 98.7…99.9 ms, 1.2 %), while the *curvature-stage* loss (15–18 %) is
real but is 15–18 % of a 4.5–6 % item. A packing run that wants that 1 % back has
`set_vof_curvature_worklist(False)`.

**What decides the sign, measured but not turned into an AUTO rule.** The compacted cascade has a
floor of ~5.1–5.4 ms/step on this GPU that is independent of grid size from 24³ to 96³, i.e. it is
launch-and-occupancy bound on the few hundred to few thousand cells the fallback actually serves;
the dense cascade instead grows with the region. Compaction therefore wins wherever the dense
region sweep would cost more than that floor. Neither the interfacial-cell count nor the
interfacial fraction predicts the crossover on its own — droplet 40³ wins at 1879 interfacial cells
while the packed column loses at 1817, and the packed column has 2.7× the interfacial fraction of
the trickle bed and loses by the same margin — so an AUTO rule would have been fitted to two points
and was not written. The honest statement is the table.

### What could NOT be measured, and why

* **Real MPI scaling of the g = 3 halo.** One node, one GPU. Every CUDA np > 1 number in this entry
  is device time-slicing (item 3); the host-openmp pinned ladder is a genuine measurement of the
  *fraction* but of a CPU code, and it goes only to np = 4 on a single socket. The conclusion that
  the PARIS trick is not indicated rests on the *np-independence of the fraction*, which is robust
  to that (a stage that is 12.9 % of the step at np = 1, where no message exists, cannot be
  message-bound), not on the absolute numbers.
* **The interface-weighted ORB's "after".** Blocked by the `rebalance_by_weights` heap corruption
  (item 4b). The before-imbalances are recorded so the fix has a target.
* **The np = 4 host-openmp packed run at `--bind-to none`** was abandoned after np = 2 measured
  17.3 s/step; it would have taken the whole budget to measure a binding artefact.
* **Whether the ~5 ms compacted-cascade floor is the fallback kernel's occupancy or a fence.** The
  breakdown attributes it to `vof::curv::pv_list`, but no profiler run (nsys/ncu) was made.

---
## WO-P3g findings — the second-order interfacial energy operator: gates (a) and (c) PASS at round-off, gate (d) CLOSES at Ja 0.5 (0.027 %) and the mesh ladder stops being anti-convergent, gate (b) and Ja 2 do not — 2026-09-04, Opus

Branch `vof-p3g`, worktree `../flow-p3g`, from `origin/main` at `23900eb`. Backend **nvidia-cuda**
(`build_cuda`, `build_ktest_cuda`, `build_kmpi_cuda`), `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, one
solver process at a time. No run printed `preconditioner produced non-finite z`; every Scriven run
below is clean under rule 3b (max pressure iterations **22–44** against the 600 cap, none capped).

**Verdict up front, in six statements.**

1. **Gate (a) PASSES exactly.** The Gibou–Fedkiw row family is
   `a_Gamma = 2/((1+theta) theta)`, `a_behind = 2/(1+theta)` against the shipped `(1/theta, 1)`.
   On a 1-D quadratic against a fixed plane it reproduces `T''` to **1.11e-15** and the
   interfacial flux to **9.99e-16** at every `theta in {0.05 … 0.95}`; the shipped two-point row's
   flux error reaches **0.589** of an exact 0.85 at `theta = 0.05`. It reduces to the interior row
   BITWISE at `theta = 1` (`2/2 = 1` both).
2. **Gate (c) PASSES to round-off, and it is by construction.** With `mdot` defined as the
   operator's own flux, `|E_lat − q_op|/E_lat` is **3.6e-16** (Ja 0.5) / **2.9e-16** (Ja 2) on
   Scriven, and the LAGGED half — the heat the previous energy solve actually removed against the
   latent heat this step books — is **2.6e-16** / **3.8e-16**. F3 is gone.
3. **The flux the operator conserves is the flux through the CELL FACE, not the flux at the
   interface**, and that is a structural first-order error the second-order row cannot remove.
   Derived and measured: for `T = a + bx + cx²` with the interface at `theta`, the second-order
   row's total interfacial transfer is exactly `−(b + c) = −T'(1/2)` — the face flux — at every
   `theta`, while the interfacial flux is `−(b + 2c theta)`. On the a-priori 2×2 probe the
   `plane × scriven` column therefore stays at `−18.7 / −11.6 / −8.3 / −5.7 %` (order 1.02),
   essentially the shipped row's `−17.0 / −10.4 / −7.4 / −5.1 %`. **Conservation and consistency
   are in direct opposition here, and item 1 chose conservation.**
4. **Two defects the operator flux exposed, both fixed here, both real:** an interfacial cell at a
   NON-PERIODIC domain face drew heat from a ghost that carries no row (+5.8 % on those cells'
   `mdot`), and **27–29 % of the operator's interfacial heat sits on interfacial cells the joined
   marching-tetrahedra sheet gives ZERO area** — which under the shipped `mdot` produced no mass
   either, and under the operator flux would simply destroy that heat.
5. **Gate (b) improves and still does not converge.** `sphere × scriven` goes
   `+8.0/+7.8/+6.6/+5.2 %` (shipped) → `+(-0.3)/+1.6/+3.0/+3.4 %` at R = 6/10/14/20, but the order
   in `h/R` is **negative** — the error grows with R, because the operator's own `sphere × linear`
   geometric bias is **+27.4/+16.8/+13.8/+11.0 %**, roughly TWICE the fit's, and first order.
6. **Gate (d) CLOSES at Ja 0.5 for the first time in seven work orders — 0.027 % and
   beta_eff +0.027 % against the shipped 1.036 % / −1.655 % — and the MESH LADDER stops being
   anti-convergent** (`beta_eff/beta − 1` = −0.015 % at 96³ and +0.027 % at 128³, sign-changing at
   the noise floor, against WO-P3f's −1.073 → −1.655 → −2.557 %). **It does not close at Ja 2**
   (3.644 % / −1.321 % against 1.486 % / −1.475 %), whose error is bought in the first ~15 steps of
   a scene with a 2.8-cell thermal layer and then recovers. **The default therefore stays
   `energy_order = 1`**, per the work order's rule.
7. **The single most consequential finding is not in the operator at all**: it is that item 1 makes
   the DIVERGENCE-SOURCE DEPOSIT load-bearing, and a new one-line VOLUME AUDIT
   (`d(gas)/d(gas booked)`, 1.000000 for a healthy run, **0.935** here) found it in one run after
   six work orders had been arguing about the flux. With `set_phase_change_deposit_fallback(True)`
   Ja 0.5 goes from 3.489 % to 0.027 %; the fallback ALONE changes nothing (order 1 + fallback
   reproduces 1.036 % / −1.655 % to the digit).

### What shipped (all four pieces are OPTIONS; the default is unchanged)

* `set_phase_change_energy_order(order)` — the package. `order = 2` turns on all four items
  below; `order = 1` is `origin/main`, bitwise.
* `set_phase_change_mdot_operator(bool)` — item 1. `mdot = q/(h_lv A)` with `q` the sum, over the
  interfacial cell's PURE face neighbours, of `a_Gamma k_p o_p (T_p − T_Gamma)` PLUS the
  second-order row's one-sided rescaling of the band behind, `(a_behind − 1) k_b o_b (T_p − T_b)`
  — the whole heat the operator transfers, not just the Dirichlet coupling. `vof::pcOperatorMassFlux`
  closes the Schrage/IHTR relation `mdot (h_lv A + R_int sum a_Gamma k o) = sum a_Gamma k o (T_p − T_sat)`
  in one step. The least-squares estimator stays as `phase_change_diagnostics()['mdot_fit']`.
* `set_phase_change_gfm_order(1|2)` — item 2, `vof::pcGfmRow` + `scalarMaskGfm2` (a SIBLING kernel;
  `scalarMaskGfm` is untouched and still what the default runs).
* `set_phase_change_curvature_distance(bool)` — item 3, `vof::pcGfmThetaK` + `pcUpdateCurvature`
  (the V3 cascade's `kappa` on the G = 2 block, exchanged at depth 1). It feeds the row's `theta`
  AND the one-sided fits, replacing WO-P3f's prescribed `set_phase_change_fit_curvature`.
* `set_phase_change_deposit_fallback(bool)` — WO-P3f open item 6, promoted from the
  `PECLET_PC_DEPOSIT_FALLBACK` environment variable to an entry point, because item 1 makes it
  load-bearing (below).
* Diagnostics: `phase_change_diagnostics()` gains `mdot_fit`, `q_operator`, `q_orphan`;
  `phase_change_budget()` gains `q_behind`. `tests/kokkos/test_vof_phase_change.cpp` gains **K5**
  (gate (a)); both test binaries gain the `PECLET_P3G_*` hooks; `tests/study/vof_scriven.py` gains
  `--energy-order/--op-mdot/--gfm-order/--curv-dist/--carry-opt/--deposit-fallback`, the gate-(c)
  identity read-out and a VOLUME AUDIT (`d(gas)/d(gas booked)`).

### Gate (a) — the coefficient family, and the exactness proof (`test_vof_phase_change` K5)

Geometry in the row's own terms: the pure cell at `x = 0`, the interfacial cell at `x = +1`, the
interface plane at `x = theta`, the third point (the cell BEHIND) at `x = -1`. The non-uniform
three-point second difference through `(T_behind, T_0, T_Gamma)` at spacings `(1, theta)` is

    d2T/dx2 = 2/(h_L + h_R) [ (T_R - T_0)/h_R - (T_0 - T_L)/h_L ]
            = 2/((1 + theta) theta) (T_Gamma - T_0)  +  2/(1 + theta) (T_behind - T_0)

so **`a_Gamma = 2/((1+theta) theta)`, `a_behind = 2/(1+theta)`**, against the shipped
`(1/theta, 1)`. Bounded over the shipped clamp `theta in [0.1, 1.9]`:
`a_Gamma in [0.363, 18.18]`, `a_behind in [0.690, 1.818]`, so no additional `theta < 0.1` fallback
is needed and none is used — the WO's "fall back to the neighbour-behind row" is implemented as the
geometric fallback instead (no third point => the shipped two-point row, bitwise). The row is
non-symmetric but strictly diagonally dominant, and the RB-GS smoother converged on every scene
below (pressure and energy residuals unchanged; P2 reaches its noise floor as before).

`T = 0.83 - 1.47 x + 0.62 x^2`, exact `T'' = 1.24`, exact face flux `-(b+c) = 0.85`:

| theta | a_Gamma | a_behind | row's `T''` | err | row's flux `Q` | err | SHIPPED row's `Q` err |
|---|---|---|---|---|---|---|---|
| 0.05 | 38.0952 | 1.9048 | 1.2400000000000011 | 1.11e-15 | 0.84999999999999898 | 9.99e-16 | **+5.890e-01** |
| 0.25 | 6.4000 | 1.6000 | 1.2399999999999998 | 2.22e-16 | 0.85000000000000009 | 1.11e-16 | +4.650e-01 |
| 0.45 | 3.0651 | 1.3793 | 1.2400000000000002 | 2.22e-16 | 0.84999999999999964 | 3.33e-16 | +3.410e-01 |
| 0.65 | 1.8648 | 1.2121 | 1.24 | 0.00e+00 | 0.84999999999999987 | 1.11e-16 | +2.170e-01 |
| 0.85 | 1.2719 | 1.0811 | 1.2399999999999998 | 2.22e-16 | 0.85000000000000009 | 1.11e-16 | +9.300e-02 |

worst over `theta in [0.05, 0.95]`: `|T''_row - T''|` **1.110e-15**, `|Q - Q_exact|` **9.992e-16**
(and **4.441e-16** on a linear profile, where both orders are exact). **The gate passes.**

**The flux `Q` has to carry BOTH halves of the row**, and this is where the work order's text needed
a correction. `Q = a_Gamma (T_0 - T_Gamma) + (a_behind - 1)(T_0 - T_behind)`: the Dirichlet coupling
PLUS the one-sided rescaling of the band behind, which cell `-1`'s own row does not mirror and which
is therefore part of what the interface removes from the solved set. With the Dirichlet coupling
alone `Q` is off by a factor `2/(1+theta)` — a factor 2 as `theta -> 0`.

**And this is where the second-order row's ceiling is.** `Q` evaluates to `-(b + c) = -T'(1/2)`
identically in `theta`: **the exact conductive flux through the CELL FACE.** The flux `mdot` needs is
`-T'(theta)`, at the interface. The two differ by `T'' (1/2 - theta)`, i.e. by `O(h) T''` — first
order, and exactly the sensible heat of the material between the face and the interface, which the
scheme does not carry because the interfacial cell is an identity row with no energy equation. A
conservative `mdot` (item 1) is therefore a FACE flux and cannot be second-order consistent as an
interfacial flux. That is not a defect of the row; it is the price of item 1, and it is why gate (b)'s
`plane x scriven` column barely moves.

### Gate (b) — the a-priori 2x2 probe, 128^3, `sub = 16`, area mode 6, Ja 0.5

`--mdot-probe 6,10,14,20 --mdot-geom sphere,plane --mdot-prof linear,scriven --energy-order 2`.
Area-weighted `mdot` against the analytic `mdot = rho_v beta sqrt(alpha/t)` (WO-P3f's shipped
column is quoted from its findings, same scene, same seeds):

| R | BL | plane x linear | **sphere x linear** | order | plane x scriven | **sphere x scriven** | SHIPPED sphere x scriven |
|---|---|---|---|---|---|---|---|
| 6 | 2.44 | +0.418 % | **+27.356 %** | — | −18.714 % | **−0.268 %** | +8.002 % |
| 10 | 4.06 | +0.418 % | **+16.758 %** | 0.959 | −11.551 % (0.945) | **+1.609 %** | +7.820 % |
| 14 | 5.69 | +0.418 % | **+13.837 %** | 0.569 | −8.280 % (0.990) | **+2.996 %** | +6.647 % |
| 20 | 8.12 | +0.418 % | **+11.014 %** | 0.640 | −5.747 % (1.024) | **+3.417 %** | +5.160 % |

`(E_lat − q_op)/E_lat` is **0 to round-off on every row** (max 8.6e-16), and `−q_gfm/E_lat`
(the budget instrument's own pairing, which counts (inner unmasked cell, any masked neighbour) where
the operator counts (inner masked cell, any neighbour that carries a row)) reads **1.0008 … 1.0020**.
`plane x linear` reads +0.418 % at every radius — the probe's own sub-sampling quantum (WO-P3f), a
constant that cancels out of every comparison.

**The gate is MISSED on both halves.** `sphere x scriven` is inside 0.5 % only at R = 6 and grows to
+3.4 % at R = 20 — better than the shipped +5.2 % but with a NEGATIVE order, i.e. anti-convergent
again. The reason is the `sphere x linear` column: the operator's own geometric bias is
**+11.0 % at R = 20 against the fit's +6.2 %**, first order (0.57–0.96), and it is the largest single
error in the rung now. Its mechanism is the row's `theta` CLAMP: `theta = |phi_i|/|n_d|` diverges on
a face whose axis is nearly TANGENT to the interface, `pcGfmTheta` clamps it at `thetaMax = 1.9`, and
the clamp therefore over-draws on exactly the near-tangential faces, which on a sphere are numerous.
Neither the second-order row nor the curvature-consistent distance touches it (ablation below).

Ablation of the three items on `sphere`, R = 10 / 20 (Ja 0.5, 128^3, mode 6):

| configuration | linear R 10 | linear R 20 | scriven R 10 | scriven R 20 |
|---|---|---|---|---|
| shipped (least-squares fit) | +12.06 % | +6.23 % | +7.82 % | +5.16 % |
| item 1 only (op flux, row 1, no kappa) | +17.89 % | +12.31 % | +5.92 % | +6.35 % |
| item 1 + item 3 (kappa from the cascade) | +16.35 % | +11.69 % | +4.53 % | +5.76 % |
| item 1 + item 2 (second-order row) | +17.98 % | +11.52 % | +2.84 % | +3.93 % |
| **items 1+2+3 (`--energy-order 2`)** | **+16.76 %** | **+11.01 %** | **+1.61 %** | **+3.42 %** |

Read it as: item 2 buys the profile half (`scriven` improves by ~2.4 points), item 3 buys ~1.4
points of the geometry half, and neither dents the `linear` column's +11–12 %.

### Gate (c) — the enthalpy budget, on the planar rungs first and then on Scriven

The gate has two halves and both are new instruments in `vof_scriven.py`:

* the BY-CONSTRUCTION half, `|E_lat − q_op| / E_lat`, where `E_lat = rho_l removed_volume h_lv/dt`
  is the latent heat the regression books and `q_op = phase_change_diagnostics()['q_operator']` is
  the heat the operator's rows draw on the fields the build read;
* the LAGGED half, `|q_op(n) + q_solve(n−1)| / q_op(n)`, where `q_solve = q_gfm + q_behind` from
  `phase_change_budget()` is the heat the PREVIOUS energy solve actually removed. `mdot` is
  evaluated at the head of step `n` from the converged `T` of step `n−1`, so this is the identity
  that says the energy the solve took out and the mass the next step makes are one number.

Planar rung first (`--carry-probe`, 64^3, a planar interface on the sub-sampling grid, an exactly
linear superheat, `mdot` fixed by construction at 2.000e-3, ratio 100, `--energy-order 2`):

| step | `E_lat` (W) | `q_op` (W) | `(E_lat − q_op)/E_lat` | `q_solve(n−1)` (W) | `(q_op + q_solve(n−1))/q_op` |
|---|---|---|---|---|---|
| 1 | 2.04800e+02 | 2.04800e+02 | **−1.388e-16** | — | — |
| 2 | 2.05055e+02 | 2.05055e+02 | **+0.000e+00** | −2.05055e+02 | **+0.000e+00** |
| 3 | 2.05247e+02 | 2.05247e+02 | **+1.385e-16** | −2.05247e+02 | **+0.000e+00** |
| 4 | 2.05406e+02 | 2.05406e+02 | **+1.384e-16** | −2.05406e+02 | **−1.384e-16** |
| 5 | 2.05541e+02 | 2.05541e+02 | **+1.383e-16** | −2.05541e+02 | **+0.000e+00** |

`E_lat` at step 1 is 204.800 W, i.e. exactly `2.000e-3 x 4096 x 25`. Scriven 128^3, mode 6, MUSCL,
similarity start, `R 6 -> 20`: worst over the run

| | by construction | lagged |
|---|---|---|
| Ja 0.5 | **3.606e-16** | **2.634e-16** |
| Ja 2 | **2.858e-16** | **3.787e-16** |

**Gate (c) passes at round-off, on both rungs and both Ja.** F3 is gone by construction: the
Dirichlet overwrite can no longer change the mass balance, because the mass balance is now the
energy balance.

**A defect the gate found on the way, and it is not cosmetic.** An interfacial cell at a
NON-PERIODIC domain face has ghost neighbours that `pcZeroDomainGhosts` marks unmasked — they look
like pure cells — but no row is ever built there, so gathering their Dirichlet coupling invents
heat. On the planar probe (whose interface spans the whole y–z cross-section) the edge cells' `mdot`
read **2.115e-3 against the exact 2.000e-3, +5.8 %**, and the lagged identity sat at 5.3e-4 instead
of 0. `pcBuildInDomain` separates "a ghost owned by another rank" (a row exists there) from "outside
the global domain" (none does) with the same fill-then-zero construction the rest of the rung uses,
so it is decomposition-independent by the same argument. With it the lagged identity is **bitwise**.

### Gate (d) — Scriven, 128^3, ratio 100, similarity start, MUSCL, `R 6 -> 20`, area mode 6

**Attempt 1, the package as the work order specifies it.** The baseline row is this session's own
re-run and it reproduces WO-P3e/P3f to the digit, so the harness is faithful.

| configuration | Ja 0.5 max\|dR\|/R | beta_eff/beta − 1 | Ja 2 max\|dR\|/R | beta_eff/beta − 1 | band_div | fallback | VOLUME AUDIT |
|---|---|---|---|---|---|---|---|
| **shipped** (order 1) | **1.036 %** | **−1.655 %** | 1.486 % | −1.475 % | 6.0e-12 | 0 | **1.000000** |
| `--energy-order 2` | **3.489 %** | **−5.444 %** | **5.836 %** | **−6.725 %** | 1.0e-01 | 208 | — |
| item 1 ALONE (`--op-mdot 1`, row 1, no kappa, no carry) | **3.517 %** | **−5.316 %** | — | — | 1.0e-01 | 176 | **0.934650** |

**Item 1 alone reproduces the whole failure** (3.517 % against the package's 3.489 %), so items 2, 3
and 4 are together worth 0.03 pp in the coupled run and the mechanism is item 1's.

**The mechanism, and the instrument that names it.** The new VOLUME AUDIT prints
`d(gas volume) / d(gas volume the regression BOOKED)` per step — the gas volume can only change by
the regression plus the net liquid flux through the outflow faces, so it separates "the books are
wrong" from "the books are right and the colour field does not follow". The shipped scheme reads
**1.000000 (min 1.000000, max 1.000000)** over the last half. Item 1 reads **0.934650
(0.896570 … 0.981961)**: **6.5 % of the vapour the regression books never materialises.**

And `fallback` says why. Item 1 makes the interfacial AREA cancel out of the mass balance, so the
27 % of the operator's heat that sits on ZERO-AREA interfacial cells has to evaporate somewhere —
`Aeff = 1` gives those cells `dV = q dt/(h_lv rho_l)` and a divergence source
`S = q (1/rho_g − 1/rho_l)/h_lv`. But those are exactly the cells DEEP in the band, whose
along-the-normal deposit walk (`round(k n)`, k = 1, 2) finds no pure gas cell: **`fallback` goes
0 -> 176…208 and `band_div` 6.0e-12 -> 1.0e-01 (Ja 0.5) / 9.7e-01 (Ja 2)**, i.e. `div(open u) = S`
on those cells' OWN faces and Weymouth–Yue advects the colour with a field that is not the liquid
velocity. `mdot` is right (area-averaged +0.30 % over the last half, against the shipped −0.93 %) and
the bubble still grows 5 % too slowly, which is exactly what an audit of 0.935 with a correct flux
means. **WO-P3f's open item 6 was not a loose end; item 1 makes it load-bearing.**

**Attempt 2 — the same package with the deposit fallback the mechanism demands.** One change:
`--deposit-fallback 1`, i.e. an interfacial cell whose two along-the-normal candidates are both
still interfacial takes the best cell of the `+n` half of the 5^3 box (WO-P23's rule, which has
always existed as a fallback and has always been default-OFF).

| configuration | Ja 0.5 max\|dR\|/R | beta_eff/beta − 1 | Ja 2 max\|dR\|/R | beta_eff/beta − 1 |
|---|---|---|---|---|
| shipped (order 1) | 1.036 % | −1.655 % | 1.486 % | −1.475 % |
| `--energy-order 2` (attempt 1) | 3.489 % | −5.444 % | 5.836 % | −6.725 % |
| **`--energy-order 2 --deposit-fallback 1`** | **0.027 %** | **+0.027 %** | 3.644 % | **−1.321 %** |
| order 1 + `--deposit-fallback 1` (the control) | **1.036 %** | **−1.655 %** | — | — |

**The control matters: the deposit fallback ALONE changes nothing.** On the shipped scheme
`fallback` is already 0 (WO-P3e), so turning the rule on reproduces 1.036 % / −1.655 % to the digit.
The improvement is the P3g operator; the fallback is what the P3g operator NEEDS.

At **Ja 0.5 the P3 gate CLOSES on both halves for the first time in seven work orders**: 0.027 % and
+0.027 %, a 40x improvement on the shipped 1.036 % / −1.655 %, with `band_div` **9.5e-13**,
`fallback` 0, `unresolved` 0, `C in [-7.6e-17, 1]`, max pressure iterations 30/600 and the volume
audit at **1.000000**.

**Ja 2 does not**, and its trace says where: the error is acquired in the first ~15 steps and then
RECOVERS monotonically — `R` rel error −0.24 → −3.04 → −4.00 → −3.98 → −3.68 → −3.30 → −2.93 →
−2.60 → **−2.32 %** at the end, and `beta_eff` is −1.321 % against the shipped −1.475 %. That is a
START-UP transient, not a rate error: at Ja = 2, 128^3, the thermal boundary layer at `t0` is
**2.82 cells (99 % of dT)** and `R0/(2 beta^2) = 0.54 cells`, i.e. the scene is under-resolved where
it begins. The `max|dR|/R` half of the gate measures the LAST half of the run, but the offset it
carries was bought at the start. This is what the mesh ladder is for.

**The mesh ladder at fixed `R/L` (Ja 0.5, mode 6), which is what WO-P3f's anti-convergence claim
rests on:**

| grid | R (cells) | `L/R_end` | shipped `beta_eff/beta − 1` | **`--energy-order 2 --deposit-fallback 1`** | max\|dR\|/R | band_div | fallback |
|---|---|---|---|---|---|---|---|
| 96³ | 4.5 → 15 | 6.4 | −1.073 % | **−0.015 %** | 0.155 % | 7.0e-12 | 0 |
| 128³ | 6 → 20 | 6.4 | −1.655 % | **+0.027 %** | 0.027 % | 9.5e-13 | 0 |
| 192³ | 9 → 30 | 6.4 | −2.557 % | **DIVERGES** (see below) | — | — | — |

**The 192³ `R 9 → 30` rung DIVERGES with the new operator** at the study's default `cfl = 0.2`:
the Weymouth–Yue Courant number runs away and the dt-collapse guard trips at **step 199 of ~250**
(`t = 299.94` of 365.58, last CFL **0.9079** at `dt = 9.17e-05` against an initial 5.91e-01). That
is the one row of WO-P3f's own ladder that was already NOT at the deposit floor (`band_div`
1.6e-02, `fallback` 24) — at `R/h = 30` the interfacial band is thick enough that the `+n` walk
fails on a real population of cells, and item 1 hands every one of them a source. A `cfl = 0.1`
re-run was queued and not completed within this session; **the third rung is owed and the divergence
is the rung's sharpest open item** (WO-P3f open item 6 again, at the resolution where it bites).

**96³ → 128³ the error changes SIGN and stays under 0.03 %** — that is a noise floor, not an order,
and it is the direct retirement of WO-P3f's `−1.073 → −1.655 → −2.557 %` (which grew like
`(R/h)^1.1` because the cancelling term was the `O(h/R)` one). There is no cancellation left to
break.

**Ja 10, 128³, `R 6 → 20`** (run for information; the work order asks for it only once both other
Ja pass): `max|dR|/R` **34.174 %**, `beta_eff` **−34.952 %**, against WO-P23's shipped **~40 %**.
The volume audit is 1.000000 and `band_div` 8.6e-11, so the scheme is healthy and the scene is not:
at Ja 10 the thermal layer at `t0` is a fraction of a cell.

### Gate (e) — the planar rungs, and inertness

**Byte-identity against `origin/main` (`23900eb`), run.** `test_vof_phase_change` built from this
worktree and from a separate `origin/main` checkout (`../flow-p3g-ref`), 4 threads, nvidia-cuda:
the whole stdout is **IDENTICAL apart from the new K5 block** (`diff` empty after removing K5).
For the record: P0a `1.776e-14`, P0b `u_gas` exact / `max|div − S| 3.469e-18`, P1 `+1.3099 %`,
P1' `−0.0139 %`, ENERGY identity `0.000e+00`, P2 `+0.1929 %`, INERT `0.000e+00`.

**What the OPTIONS do to the planar rungs** (they are OFF by default, so this is a measurement, not
a regression; the binary reports the two CHECKs that fail their own scenes' tolerances):

| | P0a | P0b | P1 | P1' | ENERGY | P2 | INERT |
|---|---|---|---|---|---|---|---|
| shipped | 1.776e-14 | exact / 3.469e-18 | +1.3099 % | **−0.0139 %** | 0.000e+00 | **+0.1929 %** | 0.000e+00 |
| `PECLET_P3G_ORDER=2` | 1.776e-14 | identical | +1.2631 % | **+0.2074 %** | 0.000e+00 | **+1.1160 %** | 0.000e+00 |
| `PECLET_P3G_ORDER=2` + `PECLET_PC_DEPOSIT_FALLBACK=1` | 1.776e-14 | identical | +1.2631 % | +0.2074 % | 0.000e+00 | +1.1160 % | 0.000e+00 |
| `PECLET_P3G_OPMDOT=1` alone | 1.776e-14 | identical | **+0.1520 %** | +1.0965 % | 0.000e+00 | −0.4192 % | 0.000e+00 |

**P0a and P0b are `mdot`-PRESCRIBED and are byte-identical at every configuration**, as the work
order requires; so are the ENERGY uniform-`T` identity (`0.000e+00` at `rho c_p` ratio 1e4) and the
INERT gate. The deposit fallback is exactly inert on the planar rungs (`fallback` is 0 there).

`P1'`'s −0.0139 % was never a converged number: WO-P23's four-way ablation table shows it is the
noise floor of a CANCELLATION between the plane-anchored rows and the quadratic fit (the error
changes sign between N = 128 and 256). Item 1 moves P1 from +1.3099 % to **+0.1520 %** — a factor
8.6 — and P1' to +1.0965 %, i.e. it replaces that cancellation with a single, one-signed error.
`P2` moves +0.1929 % → +1.1160 % (package) / −0.4192 % (item 1 alone); its `band_div` stays at
2.3e-13 … 1.8e-12 and no solve is capped. **The P2 order ladder at `--energy-order 2` was NOT run
(64/128/256 at Fo = 0.5 is ~4500 steps at N = 256) and is the one gate-(e) digit this WO owes.**

### Gate (f) — MPI, and the batteries

* **`tests/kokkos_mpi`, `vof_phase_change_mpi` np 1/2/4** (64x4x4, the ORB cutting x so the interface
  crosses a rank boundary during every run), nvidia-cuda, `OMP_NUM_THREADS=4`:

  | configuration | np = 1 | np = 2 | np = 4 |
  |---|---|---|---|
  | the DEFAULTS | **Passed** (53.3 s) | **Passed** (186.3 s) | **Passed** (363.2 s) |
  | `PECLET_P3G_ORDER=2` + `PECLET_PC_DEPOSIT_FALLBACK=1` | **Passed** (23.1 s) | **Passed** (176.5 s) | **Passed** (342.9 s) |

  **3/3 and 3/3**, i.e. the whole new operator — the operator-flux `mdot` (a fixed-order gather over
  an interfacial cell's face neighbours), the second-order row (which rescales a band the neighbour
  cell's own row does not mirror), the per-cell cascade `kappa` (exchanged at depth 1), the
  `pcBuildInDomain` mask and the 5^3 deposit fallback — is decomposition-independent at the same
  floor the rung has always held. The test asserts np-to-np agreement itself; its per-case digits
  are below.

  Run directly (`mpirun -np N ./build_kmpi_cuda/test_vof_phase_change_mpi`) with
  `PECLET_P3G_ORDER=2 PECLET_PC_DEPOSIT_FALLBACK=1`, i.e. with EVERY piece of the new operator on,
  the ORB cutting x at np 2/4:

  | case (the whole P3g operator ON) | np = 1 | np = 2 | np = 4 |
  |---|---|---|---|
  | **P0a** 1000 kinematic steps, `max\|C_dist − C_ref\|` | **0.000e+00** | **0.000e+00** | **0.000e+00** |
  | **P1** Stefan, 280 coupled steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
  | P2 sucking, 55 coupled steps (interface position, rel) | 7.873e-16 | 2.362e-15 | 2.362e-15 |
  | P2 pointwise `max\|C_dist − C_ref\|` | 1.199e-14 | 1.199e-14 | 1.532e-14 |
  | P2 pointwise `max\|T_dist − T_ref\|` | 1.735e-15 | 1.735e-15 | 3.303e-15 |

  **P0a and P1 are BITWISE at every rank count with the new operator**, and P2 is at the same
  RB-GS reduction floor the rung has held since WO-P01 — the identical table WO-P3e and WO-P3f
  printed for the shipped scheme. That scene's own answer moves with the option (P1 layer
  +0.2074 % against the shipped −0.0139 %, P2 +0.7301 %), which is the measurement; the gate is the
  three columns being equal.

* **`tests/kokkos`, the FULL battery at the shipped defaults: 33/33 passed** (nvidia-cuda,
  `OMP_NUM_THREADS=4`). Every VoF ctest is green with phase change off, and
  `test_vof_phase_change`'s own stdout is identical to `origin/main`'s (gate (e)).
* **`tests/kokkos_mpi`, the whole VoF subset at the shipped defaults (`ctest -R vof_`):
  40/40 passed.**


### Gate (d), continued — the Ja 2 start-up transient, isolated

The Ja 2 gate fails on `max|dR|/R` from `R 6 -> 20` and its trace recovers monotonically, so the
error is bought where the scene is under-resolved. The direct probe is the SAME grid and the same
`R/L` from a LATER start, `R 10 -> 20` (the thermal layer at `t0` scales with `R`, so it is 1.7x
thicker there):

| Ja 2, 128^3 | max\|dR\|/R | beta_eff/beta − 1 | area-avg mdot, last half | band_div | fallback |
|---|---|---|---|---|---|
| `R 6 -> 20`, shipped | 1.486 % | −1.475 % | — | 2.5e-11 | 0 |
| `R 6 -> 20`, `--energy-order 2 --deposit-fallback 1` | 3.644 % | −1.321 % | +1.78 % | 5.1e-11 | 0 |
| **`R 10 -> 20`, `--energy-order 2 --deposit-fallback 1`** | **1.078 %** | **−0.809 %** | +0.88 % | 5.5e-11 | 0 |
| **`R 10 -> 20`, shipped (the control)** | **0.626 %** | **−0.889 %** | — | 1.6e-11 | 0 |

From the later start the growth-rate half of the gate PASSES (**−0.809 %**) and the `max|dR|/R`
half misses by 0.078 pp, against 3.644 % from the early start — and the `R` trace is flat
(−0.076 % at step 1, −1.078 % at the half point, −0.967 % at the end) rather than diverging. **The
Ja 2 residual is largely a start-up transient of an under-resolved scene, measured.**

**But the control is the honest half of that statement: the SHIPPED scheme passes the later-start
Ja 2 scene too** (0.626 % / −0.889 %), and passes `max|dR|/R` more comfortably than the new operator
does. So at Ja = 2 the new operator buys a slightly better growth RATE (−0.809 against −0.889 %) and
pays for it in the offset. **The rung's case rests on Ja 0.5 and on the ladder, not on Ja 2.** (The
shipped control's own volume audit is 0.999934 with a minimum of 0.999248 — it is not exactly at the
floor on this scene either, which is worth knowing before anything reads that audit as a pass/fail.)

### The mechanism, stated once

WO-P3f's three first-order errors and what this rung did to each:

* **F3 (the Dirichlet overwrite destroys enthalpy) is GONE BY CONSTRUCTION.** Defining ṁ as the
  operator's own flux makes the mass balance the energy balance; the identity holds at round-off
  (gate (c)) whatever the overwrite does. `carry_conserve` is still on in the package, but with
  item 1 in place it is worth 0.03 pp coupled (the item-1-alone ablation).
* **F2 (the two-point row's `O(h/delta_T)` flux deficit) is HALF gone.** The row is now exact on a
  quadratic (gate (a)), which is what buys `sphere x scriven` its 2.4 points in gate (b). What is
  NOT gone, and cannot be while item 1 holds, is that a conserved flux is a CELL-FACE flux: the
  second-order row transfers exactly `−T'(1/2)` where ṁ needs `−T'(theta)`, and the difference is
  the sensible heat of the material between the face and the interface, which the scheme has no
  equation for. `plane x scriven` therefore stays at −5.7 % at an 8-cell layer.
* **F1 (the fit's `O(h/R)` interface-curvature bias) is REPLACED, not removed.** The operator's own
  geometric bias on `sphere x linear` is **+11.0 % at R = 20** against the fit's +6.2 % — larger,
  and still first order. Its mechanism is the row's `theta` CLAMP: on a face whose axis is nearly
  TANGENT to the interface `theta = |phi_i|/|n_d|` diverges, `thetaMax = 1.9` truncates it, and the
  row therefore over-draws on exactly the faces a sphere has most of. The curvature-consistent
  distance (item 3) is worth 1.4 of those 11 points.

And one error nobody had booked, which item 1 turned from harmless into fatal and then into the
gate's key: **the divergence-source deposit.** Under the shipped ṁ, an interfacial cell the joined
sheet gives no area produces no mass and needs no deposit. Under the operator flux the area cancels,
so those cells — **27–29 % of the total interfacial heat** — must evaporate, and they are precisely
the cells deep in the band whose along-the-normal deposit walk fails. Leaving the source in place
puts `div(open u) = S` on their own faces and Weymouth–Yue then advects the colour with a field that
is not the liquid velocity: the VOLUME AUDIT reads **0.935** instead of 1.000000 and the bubble grows
5 % too slowly with a correct ṁ. With `set_phase_change_deposit_fallback(True)` the audit is
1.000000, `band_div` is back at 9.5e-13 and Ja 0.5 closes at 0.027 %.

### Open, and the corrected gates this WO proposes

1. **P3 is CLOSED at Ja 0.5 (0.027 % / +0.027 %) and OPEN at Ja 2 (3.644 % / −1.321 %)**, and the
   Ja 2 residual is a START-UP transient of an under-resolved scene (thermal layer 2.82 cells at
   `t0`, `R0/(2 beta^2) = 0.54 cells`), not a rate error — the trace recovers monotonically from
   −4.00 % to −2.32 % and `beta_eff` is already better than the shipped scheme's. The clean next
   probe is a LATER start (`--r0 10` at Ja 2 keeps `R/L` and doubles the initial layer) rather than
   another operator change.
2. **A conserved `mdot` is a CELL-FACE flux.** The interfacial cell is a Dirichlet identity row, so
   it has no `rho c_p dT/dt` and the heat stored between the cell face and the interface is
   unbooked. That is the last first-order term in the rung and no row order removes it. The design
   that does: give the interfacial cell its OWN energy equation — a Robin/mixed row carrying its
   heat capacity, with the interface condition as a source — instead of an identity row. Then the
   operator's flux IS the interfacial flux and item 1 and item 2 stop pulling against each other.
3. **The `theta` clamp is now the largest single a-priori error** (`sphere x linear` +11.0 % at
   R = 20, first order). `thetaMax = 1.9` was chosen for a plane-anchored row on a PLANE; on a
   sphere it over-draws on every near-tangential face. Raising it is not obviously right (a
   near-tangential face's pure cell is one cell from the interface even though the interface is far
   along that grid line) and the WO-P01 continuity requirement constrains any change. **Sweep
   `thetaMax` on the `sphere x linear` probe before touching anything else** — it is one parameter
   and one probe, and no work order has ever varied it.
4. **The 192³ `R 9 → 30` rung DIVERGES** (dt-collapse guard at step 199, CFL 0.9079). It is the
   rung's sharpest open item and it is WO-P3f's open item 6 at the resolution where it bites: at
   `R/h = 30` the `+n` deposit walk fails on a real population of cells and item 1 gives every one
   of them a source. The 5³ fallback fixes it at `R/h ≤ 20`; whether it is enough at 30, or whether
   the deposit needs a genuine band-extended velocity (VOF_PLAN §9 item 3, still not implemented),
   is the question a P3h has to answer, and it is the same question a bubble swarm will ask.
5. **`set_phase_change_deposit_fallback` should probably become the default**, independently of
   this rung: it is provably inert wherever the along-the-normal rule succeeds (`fallback` 0 =>
   byte-identical, measured), and it is the difference between 0.027 % and 3.489 % here. WO-P3f's
   open item 6 asked for exactly this and it is now settled on the Scriven scene; what it still
   needs is the planar and MPI batteries at `deposit_fallback = 1` (this WO ran them at the
   shipped default).
6. **`-q_gfm/E_lat` should be read from `q_operator`, not from the budget.** The budget instrument
   pairs (inner unmasked cell, any masked neighbour) while the operator pairs (inner masked cell,
   any neighbour that carries a row); on a curved interface those two sets differ by 0.08–0.2 % and
   on a plane that cuts the domain boundary by 5e-4. `phase_change_diagnostics()['q_operator']` is
   the one that is exactly the mass balance.
7. **The VOLUME AUDIT belongs in every P-rung gate list.** `d(gas)/d(gas booked)` is one line of
   Python, it reads 1.000000 for a healthy run, and it is what separated "the flux is wrong" from
   "the flux is right and the colour field does not follow" in twenty minutes after six work orders
   had been arguing about the flux.

### Which default shipped, and why

**`set_phase_change_energy_order(1)` — the shipped scheme — remains the default, and
`origin/main`'s behaviour is unchanged** (`test_vof_phase_change` is identical to `origin/main`
apart from the new K5 block; the whole `tests/kokkos` battery and the MPI battery are green at the
defaults). The work order's rule is explicit — the new operator becomes the default only on a
PASSED gate (d) — and gate (d) is passed at Ja 0.5 and not at Ja 2.

That is a deliberately conservative call, and the coordinator should read it against what the rung
actually established, because this is the first time in seven work orders that the P3 gate has
closed at all:

* Ja 0.5, 128^3: **0.027 % / +0.027 %** against the shipped **1.036 % / −1.655 %** — a factor 40,
  with the volume audit at 1.000000 and `band_div` at 9.5e-13;
* the mesh ladder at fixed `R/L` is no longer anti-convergent (below), which retires the single
  strongest statement WO-P3f made;
* the enthalpy books close at round-off, which no previous rung could say;
* Ja 2's residual is a start-up transient of a scene whose thermal layer is 2.8 cells where it
  begins, and its `beta_eff` is already better than the shipped scheme's.

**What would settle it in one run each:** Ja 2 from a later start (`--r0 10`, same `R/L`), and the
P2 order ladder at `--energy-order 2`. If both hold, flipping the default is a one-line change in
`enablePhaseChange` and the four options collapse into it.

---

## WO-P23 findings (Part II, rungs P2 + P3) — 2026-09-02/03, Opus

Branch `vof-p23`, worktree `../flow-p23`, from `origin/main` at `b4c829a`. Backends: host
**OpenMP** (`build_omp` / `build_ktest_omp` / `build_kmpi_omp`, `OMP_PROC_BIND=false`) and **CUDA**
(`build_cuda` / `build_ktest_cuda` / `build_kmpi_cuda`). New files: `src/vof/energy_advect.hpp`,
`tests/study/vof_sucking.py`, `tests/study/vof_scriven.py`; extended:
`src/vof/phase_change.hpp`, `src/scalar_transport.hpp`, the phase-change section of
`src/flow_ibm.hpp`, `tests/kokkos/test_vof_phase_change.cpp`,
`tests/kokkos_mpi/test_vof_phase_change_mpi.cpp`. `src/vof/advect_wy.hpp` gains ONE accessor
(`faceFlux()`); no validated kernel body was touched.

### What shipped, and what the work order asked for that did not survive contact

| WO-P23 item | shipped as | verdict |
|---|---|---|
| plane-anchored Dirichlet, "the cell value that makes the one-sided linear profile from the pure neighbour hit `T_sat` at the plane distance" | **a PER-FACE ghost-fluid row** (`pcGfmTheta` + `scalarMaskGfm`), plus the work order's per-cell value as the CARRIED value of the interfacial cell (`pcCarriedValue`), which no neighbour reads | the literal per-cell BOUNDARY condition is **refuted, measured** — see mechanisms 1 and 2 |
| `k(C)` / `rho c_p(C)` closures + consistent `rho c_p T` transport | `src/vof/energy_advect.hpp` + `scalarBuildDiffusionVarK` / `scalarBuildRhsHeat`; `set_phase_change_energy(rho_cp_gas, rho_cp_liquid)` | shipped; uniform-T identity **bitwise** at ratio 1e4 — but see mechanism 4 |
| band-extended liquid velocity (VOF_PLAN §9 item 3) | **NOT implemented; the DEPOSIT was fixed instead** — `phase_change_diagnostics()['band_div']` is the direct read-out of the property it exists to guarantee | see mechanism 5 |
| "Aslam quadratic extrapolation of `T` across the band … only if the gate says so" | `set_phase_change_quadratic_fit`, **ON by default** — least-squares rather than PDE form; the gate said so twice over | mdot order 1.1 -> 2.0 |
| P2 order >= 1.4 over 64/128/256 | 64->128 order **2.52** at ratio 10 | PASS on the pair measured |
| P3 `R(t)` within 1 % | **FAILS at every Ja** (2.2 / 6.0 / 40 % at Ja 0.5 / 2 / 10, 128^3) | see the P3 table and open question 1 |

### Gate numbers

All P1/P2 numbers are host-OpenMP; the CUDA column agrees to the digits printed unless stated.

#### The P1 Stefan ladder — the four-way ablation that decides the two new defaults

Same problem as WO-P01 (St = 1, `rho_g = rho_l`, Fo = 0.5 so `dt ~ h^2`, 280 / 1119 / 4474 steps),
`tests/study/vof_stefan.py`-equivalent driver, relative error of the vapour-layer thickness:

| plane-anchored | quadratic fit | N = 64 | N = 128 | N = 256 | order 64->128 | 128->256 |
|---|---|---|---|---|---|---|
| off | off (**= rung P0/P1**) | +1.3099 % | +0.5943 % | +0.1952 % | 1.140 | 1.606 |
| **on** | off | +1.7044 % | +0.8375 % | +0.4191 % | 1.025 | 0.999 |
| off | **on** | -2.5284 % | -2.8994 % | -2.7924 % | -0.197 | 0.054 |
| **on** | **on** (**shipped default**) | **-0.0139 %** | **-0.0023 %** | **+0.0031 %** | (noise floor: the error changes sign between 128 and 256) |
| per-CELL value (the WO's literal reading) | off | +6.2030 % | +5.6164 % | +5.4165 % | 0.143 | 0.052 |

Read the table as one statement: **each half alone is worse than neither, and the pair is two orders
of magnitude better than both.** The plane-anchored rows alone remove the sign-oscillating component
(1.14/1.61, a non-monotone pair, becomes a clean 1.03/1.00) and expose the fit's curvature bias; the
quadratic fit alone corrects a gradient that is then imposed in the wrong place. `C in [0, 1]`
exactly, `unresolved = 0`, `fallback = 0` at every N and every variant.

#### The `mdot` kernel itself (the decisive, time-stepping-free probe)

An EXACT analytic sucking-interface state (ratio 10, Ja 1) is imposed — the colour by exact planar
fractions, the temperature by the similarity solution — and `apply_phase_change(0.0)` is called once,
so the number below is the one-sided fit plus the interfacial Dirichlet and nothing else. Interface
at `0.25 N + 0.37` cells (the sub-cell offset is swept over 0.13 / 0.37 / 0.50 / 0.87; the orders
below vary by less than 0.02 across it):

| fit | N = 64 | N = 128 | N = 256 | N = 512 | orders |
|---|---|---|---|---|---|
| linear (rung P0/P1) | -0.6079 % | -0.2638 % | -0.1214 % | -0.0580 % | 1.204 / 1.119 / 1.065 |
| **quadratic** | +0.1066 % | +0.0275 % | +0.0070 % | +0.0018 % | **1.956 / 1.979 / 1.989** |

The linear fit is a straight line through the interface value fitted to samples that start ~1 cell
from the plane and reach ~2.5, so a curved profile tilts it by `O(T'' h)` — a clean FIRST-order
error in `mdot`. One more basis function on the same samples makes it second order.

#### P2 — the Welch & Wilson sucking interface (`tests/study/vof_sucking.py`)

Ratio 10, Ja = 1, wall at the vapour end, OUTFLOW at the liquid end, `Fo = 0.5`, the far-field
Dirichlet refreshed to the exact similarity value every step:

| N | steps | layer | exact | rel | `\|T - T_exact\|_inf` (liquid) | pressure iters | `band_div` |
|---|---|---|---|---|---|---|---|
| 64 | 296 | 16.03090 | 16.00000 | **+0.1931 %** | 6.34e-03 (0.634 % of dT) | 297 / 4000 | 3.80e-04 |
| 128 | 1183 | 32.01079 | 32.00000 | **+0.0337 %** | 3.06e-03 (0.306 % of dT) | 1003 / 4000 | 1.16e-04 |

**Observed order 2.52** on the interface position (the gate asked for >= 1.4; Boyd & Ling 2023 §4.2
report >= 1.4) and 1.05 on the temperature profile, which is within 0.31 % of the similarity solution
at N = 128 (the gate asked for 1 %). No capped solve at either resolution (rule 3b).

#### P3 — Scriven bubble growth (`tests/study/vof_scriven.py`), and why it misses

3-D, density ratio 100 (Ja is bounded above by `rho_l/rho_v`, so Ja = 10 does not exist at ratio 10),
outflow on all six faces with `set_rho(rho_l)` so the boundary coefficient IS the varRho one, `T_inf`
Dirichlet on all six, `sigma = 0`, `R(t)` read from the LIQUID VOLUME DEFICIT
`R = (3 sum(1-C)/4pi)^(1/3)`. Gate: `max |R_num - R_exact| / R_exact` over the LAST HALF of the run.

| Ja | beta | `delta_T` at R = 6 / 20 (cells) | 128^3, R 6->20 | + MUSCL | 192^3, R 9->30 (1.5x resolution) | 192^3, R 6->20 (1.5x clearance) |
|---|---|---|---|---|---|---|
| 0.5 | 0.7845 | 8.8 / 29.5 | **2.235 %** | **2.002 %** | 2.589 % | **2.001 %** |
| 2 | 2.3574 | 2.8 / 9.5 | **5.958 %** | **2.636 %** | 4.713 % | — |
| 10 | 10.8587 | 0.63 / 2.09 | **36.62 %** | — | — | — |

**Neither refinement closes it, and the two say different things.** 192^3 with the same bubble in
CELLS (R 6 -> 20) is the SAME grid resolution with 1.5x the clearance: **2.001 % against 2.002 %**,
so domain confinement is excluded to three digits. 192^3 with the same PHYSICAL bubble (R 9 -> 30)
is 1.5x the resolution at fixed clearance: **2.589 % against 2.235 %**, i.e. no convergence either.
The error is therefore neither the box nor the mesh spacing on its own. The `mdot` column locates
it in time rather than in space: `mdot` is **+9.5 %** at the first sample, crosses zero, and ends
**-2.7 %**, so the deficit is accumulated in the first few steps out of a sub-sampled sphere and
never repaid. Every run is clean (`C in [0,1]`, `unresolved = 0`, 30-45 pressure iterations against
a cap of 600, none capped).

Ablations at 128^3, Ja = 0.5: **quadratic fit OFF: 12.893 %** (and `mdot` -25.7 % at the first
sample — on a CURVED interface the linear fit is not a small correction, it is the answer);
**consistent energy transport OFF: 1.455 %** (better than with it — mechanism 4).

`delta_T` is the radius at which the exact profile reaches 99 % of `T_inf`, minus `R`. It is the
number that explains the table: at Ja = 10 the thermal boundary layer is **sub-cell** for the whole
run at 128^3, which no interfacial gradient fit can survive; at Ja = 2 it is 2.8 cells at the START,
and the error accumulated there is what the last-half gate then measures (the Ja = 2 growth RATE
recovers — `mdot` is within 0.2-1.7 % over the second half with MUSCL while `R` stays 2.6 % behind).

### Mechanisms and corrections

**1. A per-CELL plane-anchored value is not a boundary condition — it is right on one side and wrong
on the other, and the P1 ladder says so.** The work order asks for "the cell value that makes the
one-sided linear profile from the pure neighbour hit `T_sat` at the plane distance". Implemented
literally (`T_cell = T_G + G phi_c`, `G` the one-sided fit of the side the CENTRE lies on), it reads
**+6.20 / +5.62 / +5.42 %** at N = 64/128/256 with observed order **0.10**: it does not converge.
The mechanism is not subtle once seen — the interfacial cell has BOTH neighbours, and the value that
is correct for the side the fit came from is an over-heat for the other. On the Stefan ladder that
other side is the saturated liquid, so the interfacial cell drives a spurious heat flux into it, the
liquid acquires a gradient, and `mdot = (k_g G_g - k_l G_l)/h_lv` picks it straight back up: a
positive feedback with a biased fixed point, which is exactly what a saturating, non-converging error
looks like. **One cell-centred value cannot serve two faces.** The shipped form is therefore PER
FACE: the pure cell's own row carries `k open (T_i - T_G)/theta` with `theta` the distance in cells
from ITS centre to the neighbour's PLIC plane along that face's axis, and the interfacial cell's
value is never read across a face. The liquid-side face then reads `T_G` at its own `theta` — exactly
zero flux for a saturated liquid, as it must be.

**2. …and then the per-cell value IS needed, for a different job.** With the plane-anchored rows the
interfacial cell is decoupled from the solve, so what it holds only matters when the interface sweeps
past and it becomes PURE — at which point the energy operator and the gradient fit both read it.
Leaving it at `T_G` makes every newly exposed cell start at the saturation temperature, i.e. a cold
spot the diffusion then has to remove: measured **-1.31 / -0.72 / -0.36 %**, a clean order 0.93, with
the sign of "the interface is held back". Giving it the one-sided extrapolation (`pcCarriedValue`)
removes it. So the work order's expression is right and its PLACE was wrong.

**3. The energy scalar's ghost band was never filled before the gradient fit read it — a 56 % error
in `mdot` that did not converge.** `pcBuildInterface` samples `T` over a 5^3 stencil, i.e. at ±2,
and `set_field` / the coupling drivers write inner cells only while `advanceScalars` fills the ghosts
at its END. The FIRST `apply_phase_change` (or `step`) of every run therefore fitted its one-sided
gradients partly against a band of zeros. It is invisible on the P0/P1 gates because their interface
is far from every boundary in x and the y/z ghosts are refilled after step 1 — but on the exact-state
kernel probe, which is quasi-2D so the y/z ghosts sit INSIDE the stencil, `mdot` came out **8.16
against the exact 18.48** and stayed there under refinement (the ghost samples are counted as pure
phase at `T = 0` and pull the fit towards zero). One line in `pcBuildInterface`. It moves WO-P01's
recorded P1 numbers: **+1.158 -> +1.310 %** at N = 64 and **+0.552 -> +0.594 %** at N = 128
(N = 256 is unchanged at +0.195 %, and the recorded orders 1.069/1.500 become 1.140/1.606).

**4. The consistent `rho c_p T` transport is a correctness statement, and at moderate Jakob number it
COSTS accuracy unless the donor is reconstructed.** The identity it buys is exact and bitwise: a
uniform temperature survives an arbitrary sharp colour advected by a uniform velocity at `rho c_p`
ratio 1e4 with `max|T - T_0| = 0.0` over 20 kinematic steps (the energy twin of WO-K's
uniform-velocity gate; it is bitwise rather than "small" because the update is evolved in the
DEVIATION form `T + [Phi_-(That_- - T) - Phi_+(That_+ - T)]/rcp(C^{new})`, in which the three terms
that must cancel do so in exact arithmetic). But the geometric flux carries a plain donor-cell
temperature, whose first-order numerical diffusion `|u| h (1 - CFL)/2` thickens the thermal boundary
layer and therefore LOWERS the interfacial gradient — which is what `mdot` is. Measured on Scriven at
128^3: **Ja = 0.5, 2.235 % with the consistent transport against 1.455 % with the scalar module's
(inconsistent) Koren TVD**. `set_phase_change_energy_muscl(True)` adds a MinMod-limited donor
reconstruction and buys it back: **Ja = 0.5 -> 2.002 %, Ja = 2 -> 2.636 % from 5.958 %**. It ships
OFF by default, as the energy twin of `set_vof_momentum_muscl`, and every P3 number above says which
setting it was taken with.

**5b. WO-R2's WISP GUARD AND PHASE CHANGE ARE INCOMPATIBLE ON A CURVED INTERFACE, and the rebase is
where it showed.** This rung was developed against `origin/main` at `b4c829a` and rebased onto
`2b55edb` at the end. After the rebase the P3 Scriven bubble — which had been running clean —
DIVERGED: `R(t)` error **48.0 %**, 34 steps, `C` up to 1.0018, `T` in [-9.4, +1.3], and the study's
dt-collapse guard tripping. Bisected against the three defaults the rebase brought in
(`set_outflow_rho_correction`, `set_pressure_exact_residual`, `set_vof_wisp_eps`), it is the wisp
guard and only the wisp guard: `set_vof_wisp_eps(0)` restored **2.002 %** and 80 clean steps, the
pre-rebase number to four digits, while ablating the other two changed nothing (48.1 % / 50.7 %).

The mechanism is a disagreement about what an interface IS. WO-R2 item 4 makes `enable_vof` set
`WyAdvector::wispEps = 1e-8`, so the ADVECTOR treats a cell with `C <= 1e-8` as a pure phase for
reconstruction and flux. Phase change used its own `1e-12`, so over the band `1e-12 < C < 1e-8` the
driver reconstructed a plane, gave the cell an area, an `mdot`, a plane-anchored Dirichlet row and a
divergence-source deposit, while Weymouth-Yue moved that cell's colour algebraically as a pure
phase. The planar P0/P1/P2 gates never see it — their interface has no wisps — which is exactly why
it had to be found on the curved case.

Two changes, both shipped. (a) The tolerances are now READ from the advector:
`pcEffInterfaceEps() = max(pcInterfaceEps_, wispEps)` and `pcEffPureEps() = max(pcPureEps_,
wispEps)`. Both halves are needed and the second is not cosmetic — with only the interfacial
tolerance raised, the deposit's "find a pure gas cell" walk rejects exactly the cells the colour
field has already emptied, the source is left in interfacial cells, and `band_div` on the P2 gate
reads **2.2e+02**. (b) `enable_phase_change` then sets `wispEps = 0` outright. (a) alone is not
enough: with the tolerances shared and the guard at 1e-8 the Scriven run is MARGINAL — it completed
once at 1.973 % and diverged once at t = 126 of 162 with the Courant number pinned at its cap — so
the rung takes the deterministic option and `set_vof_wisp_eps` after `enable_phase_change` is the
deliberate override (`PECLET_P23_WISP=1e-8` in `tests/study/vof_scriven.py` reproduces the row).

**5. The band-extended liquid velocity (VOF_PLAN §9 item 3) is not needed in 1-D and IS the residual
in 3-D — but the cure is the DEPOSIT, not an extension.** What §9 item 3 exists to guarantee is that
Weymouth-Yue advects the colour with the LIQUID velocity at interfacial cells, and the direct
read-out of that is `max |div(open u)|` over the interfacial cells — it is zero iff no deposit sits
on a face WY reads. That is `phase_change_diagnostics()['band_div']`, added here. On the planar P2
scene it is **3.8e-4 / 1.2e-4** at N = 64/128 against liquid velocities of order 200-400 cells/s, i.e.
a relative 1e-6: no extension needed, exactly as WO-P01's P0b row predicted. On the CURVED Scriven
bubble it is **2e-3 … 2e-1** and it correlates one-for-one with `fallback`, the census of interfacial
cells whose deposit found no pure gas cell. The rung P0/P1 deposit rule tried exactly two candidates
(`round(k n)`, k = 1, 2) and left the source IN the interfacial cell otherwise — 48 to 262 cells on
these runs. A 5^3 best-by-collinearity search fills those holes, but **the ORDER matters and the
measurement says so**: making the search the PRIMARY rule diverges the Scriven bubble (`max|uf|`
runs away, the dt-collapse guard trips at step 316 of the Ja = 0.5 run with the Weymouth-Yue Courant
number pinned at 0.38 however small dt is), because it re-targets deposits the along-the-normal rule
was placing correctly. It therefore ships as a FALLBACK behind the P0/P1 candidates and OFF by
default (`PECLET_PC_DEPOSIT_FALLBACK=1`), so the validated behaviour is unmoved wherever it existed.
With the wisp-tolerance fix of 5b the fallback changes the Scriven result by 0.8 percentage points
(48.048 -> 48.828 % on the broken configuration; both are diverged runs), so it is recorded as a
measured option rather than promoted.

#### MPI, and a round-off sensitivity that is NOT a distribution defect

`tests/kokkos_mpi/test_vof_phase_change_mpi`, 64x4x4, the ORB cutting x at np 2 and 4 so the
interface crosses a rank boundary during every run:

| case | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| **P0a** planar regression, 1000 kinematic steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| **P1** Stefan, 280 steps with the energy solve | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| **P2** sucking, 55 COUPLED steps — interface position | 2.3e-05 | 7.3e-05 | 1.1e-04 |
| **P2** — pointwise `max\|C_dist - C_ref\|` | 3.6e-04 | 8.3e-04 | 1.1e-03 |

P0a and P1 are **bitwise**, as WO-P01 left them: the exchange, the gather-based source deposit and
the gather-based deficit redistribution are all exact, and the plane-anchored rows' new depth-1 data
(`n`, `phi_c`, `T_G`, the mask) is exchanged and domain-zeroed on the same footing. On
**nvidia-cuda** the P2 row reads `layer` **1.516e-05** and pointwise **4.729e-04** at np = 1 AND at
np = 2 — the two are identical to every digit, which is the decomposition statement.

The P2 row is a property of the COUPLED scene, not of the decomposition, and the np = 1 column is
where that is visible: there the distributed and reference solvers run the SAME block and differ
only in the arithmetic path `initMpi` selects. The same np = 1 host run reads 0.0 at 1 step,
3.3e-16 at 3, 2.4e-4 at 12 and then PLATEAUS. Bisected with `PECLET_P23_OFF` at 12 steps: all three
WO-P23 options off gives **4.1e-14**, and ANY ONE of them on gives **1.2e-4 … 2.8e-4**. The
amplifier is the interface CROSSING a cell boundary — the classification threshold
`pcIsInterfacial` switches a pure cell's whole energy row on and off, and the sharper interfacial
treatment makes that switch bigger. The INTEGRAL (the interface position) moves by 1.5e-5 … 1.1e-4,
which is what the gate is on. **Open**: a smooth blend of the interfacial row over the last decade
of `C` would remove the switch; it is a change to the classification, not to this rung, and it
should be measured against the P0a/P1 bitwise gates before it ships.

#### Inertness and byte-identity

- **INERT** (`mdot == 0`, every phase-change kernel runs): `max |C_pc - C_ref| = 0.000e+00`.
- **Byte-identity against `main` (b4c829a).** Every `tests/kokkos` binary built from both trees and
  run at 4 threads: **30 of 31 byte-identical** (`diff` of the full stdout). The one that differs is
  `test_vof_phase_change` — this WO's own file, extended with K3/K4/P1'/ENERGY/P2, and whose P1 row
  moves because of the ghost-fill defect (mechanism 3). Its P1 gate reads `+1.3099 %` against
  `+1.158 %` at `main`, still inside its own 2 % tolerance; `stefanRun` now pins
  `set_phase_change_plane_dirichlet(false)` / `set_phase_change_quadratic_fit(false)` explicitly so
  it stays the rung P0/P1 ablation next to `stefanRunP23`.
- **CUDA**: the whole single-rank battery reproduces the host numbers to the digits printed
  (P1 `+1.3099 %`, P1' `-0.0139 %`, ENERGY identity `0.000e+00`, INERT `0.000e+00`, P0b `u_gas` exact;
  P2 reads `+0.1791 %` there against `+0.1706 %` on host, inside that scene's own sensitivity).
- **The whole `tests/kokkos` battery on the rebased tree: 33/33 passed** (host-openmp).

### Open, and what a follow-on should measure

1. **P3 misses the 1 % gate at every Jakob number and grid refinement does not close it.** Both
   refinements were run: 192^3 with the same bubble in CELLS (R 6 -> 20, so 1.5x the clearance at
   FIXED resolution) gives **2.001 %** against 128^3's **2.002 %** at Ja = 0.5 — domain confinement
   is excluded to three digits — and 192^3 with the same PHYSICAL bubble (R 9 -> 30, so 1.5x the
   resolution at fixed clearance) gives **2.589 %** against **2.235 %**, i.e. no convergence either.
   The error is therefore neither the box nor the mesh spacing on its own. The `mdot` column says
   where it sits: `mdot` is +9.5 % at the first sample, crosses zero, and ends -2.7 %, so the
   deficit is accumulated early and never repaid. The two levers this WO added move it in the right
   direction (MUSCL: Ja = 2 from 5.958 to 2.636 %) and the deposit search removes its most likely
   remaining source, but the rung is NOT closed and should not be reported as validated.
2. **The `pcIsInterfacial` switch** (see the MPI section).
3. **The P2 ladder's N = 256 point is limited by the PRESSURE SOLVE, not by the scheme.** Pressure
   iterations grow 297 -> 1003 -> 3645 on 64/128/256 of a 256x4x4 grid — the transverse extent
   cannot coarsen, so the multigrid degenerates (`suite/docs/DECOMPOSITION_AND_MULTIGRID.md`), and
   at N = 256 the interface error jumps from +0.034 % to +1.590 % with the first colour wisp of the
   ladder (-2e-11). The 64 -> 128 order is **2.518**; the 128 -> 256 point should be re-taken on a
   grid whose transverse extent the multigrid can coarsen before it is quoted.

---

## WO-P3b findings (follow-up of WO-P23 item 1) — is the 2 % Scriven deficit the INITIALISATION? — 2026-09-03, Opus

Branch `vof-p3b`, worktree `../flow-p3b`, from `origin/main` at `94d73bc`. Backend **nvidia-cuda**
(`build_cuda`), `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, one solver process at a time on a shared
GPU. Only `tests/study/vof_scriven.py` was touched — **no kernel, no solver file**. Every run below
is clean under rule 3b (max pressure iterations 23–45 against the 600 cap, **none capped**) and no
run printed `preconditioner produced non-finite z`.

**Verdict up front: the hypothesis is REFUTED, and it was refuted before the first run.**
`VOF_PLAN.md` §13 item 8 asks for T(r, t₀) to be initialised from Scriven's similarity profile with
`R₀ = 2β√(α_l t₀)` instead of a uniform superheat. **`tests/study/vof_scriven.py` has done exactly
that since WO-P23 shipped it** — `scriven_T(r, r0, beta, rr, dT)` evaluated on the cell-centre radius,
with `t0 = (r0/(2β))²/α_l` by construction, i.e. R₀ and t₀ are consistent by definition and the
boundary layer is present at t = t₀. So the 2.002 / 2.636 / 36.6 % of WO-P23 were *already* the
similarity-start numbers. This WO makes that explicit (`--init`), adds the missing control
(`--init uniform`) and the finite-volume variant (`--init cellavg`), and then measures what the
deficit actually is.

### What shipped (study script only)

`--init {similarity | cellavg | uniform}` (default `similarity`, byte-identical to the shipped
expression), `--cfl`, `--area-probe`, and four read-outs the verdict needs:

* the **thermal boundary layer** at t₀ (radius where the exact profile reaches 99 % of ΔT) and
  `R₀/(2β²)` beside it;
* **R₀ as the sub-sampled colour field actually carries it** — the gate reads R from the liquid
  volume deficit, so that is the R₀ that has to be consistent (measured **−0.029 %** at R₀ = 6 on
  128³, **−0.139 %** at R₀ = 3 on 64³; 8³/16³ subsampling moves it to +0.011 / +0.002 %, i.e. it is
  three decades below the gate and is NOT the story);
* the **early ṁ transient**, steps 1–6;
* **the growth RATE against an early OFFSET.** `R = 2β√(α t)` ⇔ `R² = 4β²α t`, a straight line
  through the origin, so fitting `R_num²` against `t` over the last half gives `β_eff` (the rate —
  the method) and an intercept (a deficit acquired in the first steps and then carried — the start).
  **This is the discriminator the hypothesis needed and WO-P23 did not have.**
* the **area-averaged ṁ** (`removed_volume·ρ_l/(dt·A_Γ)`) and **`A_Γ/(4πR²)`**. `mdot_mean` is an
  unweighted mean over interfacial CELLS and does not measure the integrated flux — which is why
  WO-P23 could read ṁ +9.5 % while R fell behind.

### The gate, both ways (128³, ratio 100, MUSCL on, R 6 → 20, FCG 600/1e-10)

| Ja | `--init` | last-half max \|ΔR\|/R | **β_eff/β − 1** | R offset at the end | ṁ rel, steps 1–3 | area-avg ṁ, last half | `A_Γ/4πR² − 1` |
|---|---|---|---|---|---|---|---|
| 0.5 | **similarity** (shipped) | **2.002 %** | **−2.517 %** | +0.103 cells | +9.5, +6.1, +4.2 % | −0.450 % | −3.380 % |
| 0.5 | cellavg | 2.008 % | −2.513 % | +0.101 cells | +9.2, +5.8, +4.0 % | — | — |
| 0.5 | **uniform** (the trap) | 4.735 % | −1.700 % | +0.492 cells | **+205, +140, +123 %** | −2.629 % | −3.316 % |
| 2 | **similarity** | **2.636 %** | **−2.568 %** | −0.009 cells | −0.6, −3.9, −5.0 % | +0.310 % | −3.375 % |
| 2 | cellavg | 2.807 % | −2.613 % | −0.019 cells | −2.3, −5.2, −6.3 % | — | — |
| 2 | **uniform** | 1.499 % | −2.150 % | +0.130 cells | +35.9, +17.9, +15.8 % | — | — |

Read it in three statements.

1. **The similarity start does everything item 8 said it would do to the transient, and nothing to
   the gate.** It removes 95 % of the initial ṁ error (+205 % → +9.5 % at Ja 0.5; +36 % → −0.6 % at
   Ja 2) and it removes the early radius offset (+1.14 % → −0.06 % at step 1). The last-half error
   is unchanged, because the last-half error is not an offset.
2. **The uniform control can look BETTER, and that is a cancellation, not accuracy** (Ja = 2:
   1.499 % against the similarity start's 2.636 %). The uniform start over-evaporates the superheat
   sitting at the interface, which buys a POSITIVE radius offset (+0.49 cells at Ja 0.5, +0.13 at
   Ja 2) that then partly cancels a negative growth-rate deficit as the run proceeds; its R error
   crosses zero mid-run. Reading only `max |ΔR|/R` hides this — **`β_eff` is the invariant**, and it
   is −2.5 % under BOTH starts.
3. **Cell-averaging the profile instead of point-sampling it changes the fourth digit** (2.002 →
   2.008 %, 2.636 → 2.807 %), so the O(h²T'') half of the initialisation question is settled too.

### The residual is a growth-RATE deficit, and it survives everything except the mesh — barely

Mesh ladder under the similarity start, same clearance at every rung (R₀ = n/21.3, R₁ = n/6.4):

| grid | R | Ja = 0.5 last half | β_eff/β − 1 | Ja = 2 last half | β_eff/β − 1 |
|---|---|---|---|---|---|
| 64³ | 3 → 10 | 4.188 % | −2.683 % | 15.781 % | −10.432 % |
| 128³ | 6 → 20 | **2.002 %** | **−2.517 %** | **2.636 %** | **−2.568 %** |
| 192³ | 9 → 30 | 2.495 % | −3.212 % | 2.509 % | −3.665 % |

`β_eff` is flat at −2.5 … −3.2 % across a factor of three in mesh at Ja = 0.5 — **it does not
converge**, which is WO-P23's conclusion re-derived on a quantity that separates rate from offset.
(The Ja = 2 64³ row is a different failure: its thermal boundary layer is 1.4 cells at t₀.)

Ablations at 128³, Ja = 0.5, similarity start — each one kills a candidate:

| ablation | last half | β_eff/β − 1 | what it excludes |
|---|---|---|---|
| baseline | 2.002 % | −2.517 % | `band_div` 2.4e-03, fallback 48 |
| `--cfl 0.1` (half the time step) | 1.930 % | −2.458 % | **the temporal splitting** |
| `--sweeps 800` (energy solve, vs 200) | 2.002 % | −2.517 % | **the energy solve** — every digit identical |
| start at R 12 → 20 instead of 6 → 20 | 1.255 % | −2.371 % | **the start state** (a 2× better-resolved interface at t₀) |
| `PECLET_PC_DEPOSIT_FALLBACK=1` | 1.978 % | −2.485 % | **the deposit / band divergence**: `band_div` 2.4e-03 → **4.5e-12**, fallback 48 → **0**, and β_eff moves 0.03 pp |

The last row matters beyond this WO: WO-P23 mechanism 5 named the deposit search as "the most likely
remaining source" of the P3 miss. It is not. Turning it on removes the band divergence *entirely*
(eleven orders) and leaves the gate where it was. Also note 64³ Ja = 0.5 runs with `band_div`
**3.6e-12** and `fallback = 0` unaided, and carries the same −2.7 % rate deficit.

### Where the deficit IS: the summed PLIC interface area of a SPHERE is 5–9 % low, and it does not converge

The area-averaged ṁ column above is the tell: over the last half the flux **per unit area** is
within **±0.5 %** of Scriven's exact ṁ (Ja 0.5: −0.450 %; Ja 2: +0.310 %) — the thermal half of the
rung (the one-sided quadratic fit, the plane-anchored Dirichlet, the initialisation) is right — while
`A_Γ/(4πR²) − 1` sits at **−3.4 %** and the bubble grows as `∫ ṁ dA`. Multiplying the two reproduces
the growth deficit: at 64³ Ja 0.5 `(1+0.0348)(1−0.0605) = −2.8 %` against a measured β_eff of
**−2.68 %**; at 128³ Ja 2 `(1+0.0031)(1−0.0338) = −3.1 %` against **−2.57 %**.

`--area-probe` isolates it with **no time stepping at all**: exact sphere fractions in, one
`apply_phase_change(0.0)`, `phase_change_diagnostics()['interface_area']` out.

| R (cells), 128³ | interfacial cells | Σ A_PLIC | 4πR² | marching cubes | Σ A_PLIC rel |
|---|---|---|---|---|---|
| 4 | 248 | 190.82 | 201.83 | — | **−5.458 %** |
| 6 | 464 | 410.07 | 452.13 | 449.86 (−0.50 %) | **−9.304 %** |
| 8 | 848 | 735.43 | 804.71 | — | **−8.609 %** |
| 12 | 2088 | 1711.35 | 1809.96 | 1810.74 (+0.04 %) | **−5.448 %** |
| 16 | 3728 | 3022.57 | 3219.08 | — | **−6.105 %** |
| 20 | 5576 | 4700.16 | 5026.19 | 5044.64 (+0.37 %) | **−6.487 %** |
| 28 | 11072 | 9252.98 | 9852.29 | 9891.41 (+0.40 %) | **−6.083 %** |

Three controls, so that this is not read as a classification or a reference artefact:

* **No interfacial cell is lost.** The census (464 / 2088 / 5576 / 11072) equals the exact count of
  cells with `0 < C < 1` in the same field, computed independently in numpy — and equals it again
  with a `1e-12` guard, so `pcIsInterfacial` is not dropping slivers.
* **The reference is not the suspect.** Marching cubes on the SAME colour field returns the sphere
  area to **−0.50 / +0.04 / +0.37 / +0.40 %** of 4πR² at R = 6/12/20/28.
* **`plicArea` itself is exact on a PLANE.** `--area-probe=-1,-2,-3,-4` puts an exact half-space at
  a 0.37-cell offset (a plane exactly on a cell face has no mixed cell at all) into a 96³ box:
  the axis-aligned row returns Σ A_PLIC = **9216.0000**, which is 96² to ten digits, i.e. the exact
  plane area — while marching cubes reads 95² there, the edge convention. The tilted planes
  (1,1,0) / (1,1,1) / (1,2,3) read +1.5 / +1.2 / +0.6 % against that same edge-affected MC
  reference. So the area kernel is right, and only a CURVED interface loses 6 %.

Which points at the one quantity the campaign has already measured as **non-convergent**: the MYC
normal (V0: normal error order **0.83**, reconstruction error order 1.98 — the reason V3 takes
curvature from column sums and never from ∇C). `plicArea` returns `|m|₂ dV/dα`, which is **linear**
in that normal, so a normal error that does not converge produces an area error that does not
converge. Every earlier phase-change gate is planar (P0a, P0b, P1 Stefan, P2 sucking) — exactly the
case where the MYC normal is exact — which is why P0–P2 are second order and P3 is not.

### What a follow-on should do (P3c)

1. **Measure the per-cell area error against the exact sphere normal**, i.e. re-evaluate
   `plicArea` with `n = (x − c)/|x − c|` and the cell's own α, and confirm the 6 % is the normal.
   The instrument is small and needs no time stepping (`--area-probe` is the harness).
2. **If it is:** the fix is an area that does not go through the MYC normal — the interface area
   from the same height-function/paraboloid cascade V3 already runs for curvature (it is the
   consistent choice: κ and A_Γ are the same geometry), or a Youngs-normal-free
   `Σ |∇C|` -type estimator gated on the sphere probe. Gate it on the probe table above (target
   ≤ 1 % at R ≥ 8, converging), then re-run P3; P0a/P1/P2 must stay bit-identical, and they will,
   because on a plane the MYC normal and the fix agree.
3. **Do not re-run P3 as a mesh study before that.** The rate deficit is flat at −2.5…−3.2 % over
   64/128/192³ and the gate will not move.
4. The **`--init uniform`** row should stay in the driver as the control, and any P3 number must be
   quoted with `β_eff` beside it: `max |ΔR|/R` alone rewarded the WRONG initialisation at Ja = 2.

---
## WO-P3c findings (follow-up of WO-P3b) — the interfacial AREA: what the deficit is, and what no per-cell construction can do about it — 2026-09-03, Opus

Branch `vof-p3c`, worktree `../flow-p3c`, from `origin/main` at `4d4c95d`. Backend **nvidia-cuda**
(`build_cuda`, `build_ktest_cuda`, `build_kmpi_cuda`), `OMP_NUM_THREADS=4/8 OMP_PROC_BIND=false`,
one solver process at a time on the shared GPU. Every run below is clean under rule 3b (Scriven:
max pressure iterations 18–34 against the 600 cap, **none capped**; P2: 297/4000).

**Verdict up front, in three statements.**

1. **WO-P3b's headline is an artefact of its own probe.** The "summed PLIC area of a sphere is
   5.5–9.3 % low with no convergence" is a property of the **4³ sub-sampled colour field the probe
   builds**, not of `plicArea` and not of the MYC normal. Sub-sampling quantizes `C` to multiples
   of 1/64, so every cell whose true liquid fraction is below 1/128 is rounded to exactly 0 or 1
   and **drops out of the interface** — a quarter of the interfacial cells of a sphere, carrying
   6 % of its area, while the VOLUME moves by 1e-4 % (which is why WO-P3b's `R₀` control passed).
   Refine the probe's colour field and the deficit collapses: at R = 20, `Σ A` goes
   **−6.49 → −1.77 → −0.44 %** at sub = 4/8/16 with the shipped kernel.
2. **The first P3c task — re-evaluate the area with the EXACT sphere normal — is done and it
   REFUTES the normal hypothesis.** On the same sub = 4 field, the exact radial normal gives
   −5.57 / −9.31 / −8.51 / −5.42 / −6.58 % at R = 4/6/8/12/20 against MYC's
   −5.46 / −9.30 / −8.61 / −5.45 / −6.49 %. The MYC normal is not the culprit; nothing about the
   estimator is.
3. **The residual, once the colour field is resolved, is FIRST ORDER in `h/R` and is common to
   every per-cell construction** — PLIC plane, height-function metric, height-function footprint,
   paraboloid normal. Two analytic controls with no code in them (below) put it at
   −2.9 / −1.5 / −1.4 / −0.8 % at R = 8/12/20/28 and order ≈ 1.0 over R = 20 → 40. **It is not the
   normal, not the fractions, not the metric and not the footprint: it is that per-cell surface
   pieces do not JOIN across cells.** The WO's design (a) is implemented and gated, it is the best
   of the four variants, and it does not reach the 0.5 %-and-converging gate because no per-cell
   area can.

### What shipped

`set_phase_change_area(mode)` (default **0**, the rung P0/P1 behaviour), `phase_change_area()`,
`vof_interface_area()`, the diagnostics `area_hf_cells` / `area_pv_cells` /
`area_no_cascade_cells`. New files `src/vof/interface_area.hpp` (container-free) and
`src/vof/interface_area_field.hpp` (the g = 3 block driver — a SIBLING of `VofCurvature`, not an
edit to its validated passes, hard rule 1; same tiers, same tolerances, same reach, no new halo,
no reduction but the census). `pcBuildInterface` gains one branch on a bool. Study driver:
`--sub`, `--area-sub` (chunked, so sub = 128 is affordable), `--area-mode`, `--area-shape
cylinder`, and the interfacial-cell density per unit area. Test: **K5** and `PECLET_P3C_AREA`.

| mode | construction | where the FOOTPRINT comes from |
|---|---|---|
| **0 `kAreaPlic`** | `plicArea = \|m\|₂ dV/dα` on the MYC normal (rungs P0/P1) | the cell's own PLIC polygon |
| **1 `kAreaMetric`** | the PLIC footprint × the cascade's slope `√(1+h_x²+h_y²)` (HF) or the paraboloid's gradient (PV) | the cell's own PLIC polygon |
| **2 `kAreaNormal`** | `plicArea(n*, plicAlpha(n*, C))` — the plane rebuilt on the cascade normal | the rebuilt plane |
| **3 `kAreaFootprint`** | the height function's own footprint × its own metric | the linearized height patch — **the only one that TILES** |

Mode 3 is the WO's design (a) taken literally ("the area element integrated over the cell's
footprint in the column direction"). Its footprint is `|R| = F(+½) − F(−½)` with `F(t)` the 2-D
PLIC fraction of the linearized graph (`plicVolume(h_x, h_y, 0, ·)`), and the cells of one column
partition the transverse square **exactly**, because they share `h_x, h_y` (the neighbouring
columns' heights shift with the cell) and their `h₀` differ by exactly 1. The PV branch has no
column, so mode 3 falls back to mode 2 there and the census says how much that is (17.5 % of the
cells of an R = 20 sphere; **100 %** on a (1,1,1) plane, where no height column closes).

### Gate (b) — the a-priori probe, corrected

**The sphere, `--area-probe … --area-sub S`, 128³, exact fractions, no time stepping.** `Σ A/4πR² − 1`:

| R | cells (S=4 / S=16) | **mode 0**, S=4 (= WO-P3b) | mode 0, S=8 | **mode 0, S=16** | mode 1, S=16 | mode 2, S=16 | **mode 3, S=16** |
|---|---|---|---|---|---|---|---|
| 4 | 248 / 272 | −5.458 % | −1.200 % | −0.730 % | −1.700 % | −0.559 % | **−0.512 %** |
| 6 | 464 / 632 | −9.304 % | −1.673 % | −0.800 % | −1.336 % | −0.712 % | **−0.613 %** |
| 8 | 848 / 1160 | −8.609 % | −1.871 % | −0.224 % | −0.619 % | −0.142 % | **−0.082 %** |
| 12 | 2088 / 2552 | −5.448 % | −1.659 % | −0.223 % | −0.401 % | −0.218 % | **−0.006 %** |
| 16 | 3728 / 4592 | −6.105 % | — | −0.439 % | −0.423 % | −0.459 % | **−0.175 %** |
| 20 | 5576 / 7184 | −6.487 % | −1.771 % | −0.443 % | −0.404 % | −0.471 % | **−0.199 %** |
| 28 | 11072 / 14048 | −6.083 % | — | −0.484 % | −0.376 % | −0.514 % | **−0.214 %** |

Read the **cells** column first: the S = 4 field has a QUARTER fewer interfacial cells than the
S = 16 one at every radius. That is the whole of WO-P3b's table. The independent numpy ladder with
the exact radial normal continues it to S = 32: at R = 20, **−6.575 / −1.819 / −0.451 / −0.111 %**
for S = 4/8/16/32, with the volume moving by −0.0036 / +0.0027 / +0.0006 / −0.0001 %.

**Gate verdict.** `≤ 0.5 % at R ≥ 8`: mode 0 **passes** (−0.22…−0.48 %), mode 3 passes with room
(−0.006…−0.21 %), modes 1 and 2 pass at R ≥ 12. `CONVERGING with order ≥ 1.5`: **nothing passes**,
and §"where the residual is" says why that half of the gate cannot be met by a per-cell area.

**The plane (96³, `--area-probe=-1,-2,-3,-4`).** The axis-aligned row returns
`Σ A = 9216.0000` — 96² to ten digits — in **every mode** (K5 proves the bitwise statement at
kernel level). The tilted (1,1,0) plane has an analytic answer too, `√2 × 95.477 × 96 = 12962.5`:
mode 3 reads **12965.4 (+0.02 %)** while modes 0/1/2 read 12877.1 / 12878.1 / 12866.3
(**−0.66 / −0.65 / −0.74 %**) — on an EXACT PLANE, where the geometry is exact by construction.
The difference is the same sliver quantization (`plane_colour` sub-samples at 8³): a cell the
colour field has rounded to pure loses its PLIC footprint, while mode 3's footprint comes from the
column and its neighbour's footprint simply extends to cover it. **Mode 3 is structurally immune
to the defect that produced WO-P3b's table**; modes 0/1/2 are not.

**The cylinder (`--area-shape cylinder`) — the clean curved control**, exact in z, so its
reference `2πR n_z` carries no z discretisation and its interfacial cells are the same 148 per
layer that an analytic circle crosses:

| R | S = 16 | S = 32 | S = 64 | **S = 128**, mode 0 | mode 1 | mode 2 | **mode 3** | marching cubes |
|---|---|---|---|---|---|---|---|---|
| 12 | −5.004 % | −2.533 % | −1.833 % | **−1.587 %** | −1.684 % | −1.584 % | **−1.648 %** | −0.590 % |
| 20 | −6.992 % | −1.899 % | −1.659 % | **−1.415 %** | −1.404 % | −1.420 % | **−1.179 %** | −0.493 % |

The sub-ladder is the quantization again; the S = 128 column is the residual, and **the four
constructions agree to 0.24 pp**.

### Where the residual IS: a per-cell area is FIRST order on a curved interface

Two controls, both analytic, neither of which contains any of the code under test.

**(a) The PLIC chord of an exact circle.** Exact cell fractions (adaptive quadrature), the exact
radial normal, `α` from the fraction and the analytic 2-D chord `|m|₂ dV/dα`:

| R | 8 | 12 | 20 | 28 |
|---|---|---|---|---|
| `Σ chord / 2πR − 1` | **−2.861 %** | **−1.474 %** | **−1.424 %** | **−0.843 %** |

It reproduces the code's cylinder row (−1.415 % at R = 20) to two digits **with no reconstruction
error of any kind in it**. So even a perfect PLIC — exact normal, exact volume — is a few percent
low on a curved interface.

**(b) The mode-3 footprint construction with EXACT heights** (same circle, the cascade's per-cell
column direction, the linearized graph, the exact partition):

| R | 8 | 12 | 20 | 28 | 40 |
|---|---|---|---|---|---|
| `Σ (footprint × metric)/2πR − 1` | −3.497 % | −2.423 % | −1.655 % | −1.105 % | −0.813 % |

**Observed order 1.03 over R = 20 → 40.** So the construction the work order specifies is
first order, and its coefficient is the same as PLIC's. The mechanism both share: a cell's piece
of the surface is chosen INDEPENDENTLY of its neighbours' (its own plane; its own column
direction, which switches around |slope| = 1), so the pieces overlap and gap at every cell face
instead of joining, and the mismatch is `O(h²κ)` per face over `O(L/h)` faces — a relative
`O(hκ) = O(h/R)`. Marching cubes, whose triangles join by construction, is 3–10× closer on the
identical fields (−0.59 / −0.49 % on the cylinder rows above).

That number is not academic: the Scriven bubble runs at `R = 6 → 20`, i.e. `h/R = 1/6 … 1/20`,
which is exactly where this term is 2–4 %.

### Gate (c) — Scriven, 128³, ratio 100, similarity start, MUSCL, `R 6 → 20`

`max |ΔR|/R` over the last half **and** `β_eff/β − 1` (the WO-P3b discriminator), Ja 0.5 and 2:

| Ja | area mode | last-half max \|ΔR\|/R | **β_eff/β − 1** | `A/(4πR²) − 1`, last half | area-avg ṁ, last half |
|---|---|---|---|---|---|
| 0.5 | **0** (= WO-P3b) | **2.002 %** | **−2.517 %** | −3.380 % | −0.450 % |
| 0.5 | 1 | 2.035 % | −2.515 % | −3.356 % | −0.419 % |
| 0.5 | **3** | **1.307 %** | **−1.863 %** | −2.498 % | −0.694 % |
| 0.5 | 0, initial colour at sub = 16³ | 2.005 % | −2.540 % | −3.333 % | −0.470 % |
| 2 | **0** (= WO-P3b) | **2.636 %** | **−2.568 %** | −3.375 % | +0.310 % |
| 2 | **3** | **1.830 %** | **−1.766 %** | −2.496 % | +0.251 % |

Both mode-0 rows reproduce WO-P3b **to the digit** (2.002 / −2.517 / −3.380 and 2.636 / −2.568 /
−3.375), so the harness is faithful.

Three things to read out of it.

* **Mode 3 buys a third of the gap, and it buys it exactly where the theory says.** The area
  deficit goes −3.38 → −2.50 % and `β_eff` goes −2.52 → −1.86 % (Ja 0.5), −2.57 → −1.77 % (Ja 2).
  A uniform flux deficit `ε` gives `R ∝ √(1−ε)` for a thermally controlled bubble
  (`3R²Ṙ = (1−ε) C R` ⇒ `R² = 2(1−ε)Ct/3`), i.e. `Δβ_eff ≈ ε/2`: predicted +0.44 pp for the
  measured +0.88 pp of area, observed +0.65 / +0.80 pp. **The area deficit IS the growth deficit**,
  which is WO-P3b's inference confirmed — with a different cause than it named.
* **The gate is still missed** (1.31 % and 1.83 % against 1 %), because 2.5 % of area is still
  missing and no per-cell construction removes it. **Rule 4: this is the third failure of the P3
  1 % gate (WO-P23, WO-P3b, WO-P3c) and the run stops here with the mechanism.** Ja = 10 was NOT
  run: it is indicated only after Ja 0.5 and 2 pass, and its thermal boundary layer is sub-cell at
  128³ anyway (WO-P23).
* **The run's colour field is NOT the initialisation.** Starting from a 16³ sub-sampled sphere
  (area −0.80 % at t₀ instead of −9.30 %) changes the last half by 0.003 pp. The interfacial-cell
  density confirms it: the run starts at **1.000 cells per h²** (the sub = 4 field) and is at
  **1.435 within ten steps** and **1.476** at the end, against the ideal `⟨|n|₁⟩ = 1.5` — Weymouth–Yue
  re-creates the sliver cells within a few steps. What it cannot re-create is a joined surface.

### Inertness, the planar rungs and MPI

`PECLET_P3C_AREA=<mode>` re-runs every scene of both phase-change binaries on a non-default area.

| gate | mode 0 | mode 1 | mode 2 | mode 3 |
|---|---|---|---|---|
| K1, K2, K3, K4 | (kernels, area-independent) | identical | identical | identical |
| **P0a** planar regression, 1000 steps | 1.776e-14 | **byte-identical** | **byte-identical** | **byte-identical** |
| **P0b** ratio 100, closed column | u_gas rel 0.000e+00 | **byte-identical** | **byte-identical** | **byte-identical** |
| **P1 / P1'** Stefan N = 64 | +1.3099 % / −0.0139 % | **byte-identical** | **byte-identical** | **byte-identical** |
| **ENERGY** uniform-T identity at rcp ratio 1e4 | 0.000e+00 | **byte-identical** | **byte-identical** | **byte-identical** |
| **INERT** ṁ ≡ 0 | 0.000e+00 | **byte-identical** | **byte-identical** | **byte-identical** |
| **P2** sucking N = 64 | +0.1791 % | +0.2099 % | +0.1803 % | +0.1929 % |

Only P2 moves, and it is the one planar scene whose interface is not exactly grid-aligned in the
arithmetic: the energy solve's red–black parity asymmetry gives the MYC normal a ~1e-8 transverse
component (WO-P01 finding 3), so the height patch and the PLIC polygon differ in the last digits.
The 0.03 pp is smaller than that scene's own host-vs-CUDA spread (+0.1706 vs +0.1791 %).

**K5** (new, container-free): on a plane the metric `√(1+h_x²+h_y²)` equals `|m|₂/|m_d|` to
**0.0**, `hfSurfaceNormal` returns ±n to **2.2e-16**, mode 1 and mode 2 agree with mode 0 to
**2.2e-16 / 1.1e-15** relative, and the **38 axis-aligned rows are BITWISE** equal.

**MPI** (`test_vof_phase_change_mpi`, 64×4×4, the ORB cutting x so the interface crosses a rank
boundary during every run):

| case | mode 0, np 1/2/4 | mode 1, np 1/2/4 |
|---|---|---|
| **P0a** 1000 kinematic steps | **0 / 0 / 0** | **0 / 0 / 0** |
| **P1** Stefan, 280 coupled steps | **0 / 0 / 0** | **0 / 0 / 0** |
| P2 sucking, 55 coupled steps (interface position) | 1.5e-05 / 1.5e-05 / 5.6e-05 | 1.0e-04 / 1.0e-04 / 8.1e-05 |

P0a and P1 are **bitwise at both area modes** — the area driver is a pure local stencil on the
colour field's own g = 3 block with no reduction in it, so it is decomposition-independent by
construction. The P2 row is that scene's known coupled sensitivity (WO-P23), not a distribution
defect: np 1 and np 2 are identical to every digit in both modes.

### `vof_interface_area()` and W12's `vof_block_stats()['area']`

`vof_interface_area()` is added (the E7 gallery's request): the same sum, over the inner region,
MPI-reduced, **in whichever geometry `set_phase_change_area` selects**, so a page and the phase
change quote one number. It needs `enable_vof` only.

W12's `VofBlockSet::interfaceArea` does use `mycNormal → plicAlpha → plicPolygon →
polygonAreaCentroid`, which is `plicArea` by another route, i.e. **mode 0**. It has NOT been
switched, and that is a measurement rather than an omission: the premise for switching it was a
MYC bias, and there is none (§verdict 2). What its number carries is the first-order term of
§"where the residual is" — on a resolved sphere it reads about **0.2–0.5 % low** at R = 8…28 (mode
0, S = 16 column above), and a caller who wants the better number can set mode 3 and read
`vof_interface_area()`. Switching W12's kernel would have moved a shipped, gated number by 0.3 pp
for no defensible gain.

### Corrected gates proposed

1. **Any a-priori area probe must state, and show converged, the resolution of its OWN colour
   field.** `Σ A` is dominated by the sliver cells that a sub-sampled initialisation deletes;
   `Σ (1−C)` is not. Quote the interfacial-cell count beside the area (it is the tell: 464 vs 632
   at R = 6), and run the sub ladder. WO-P3b's table, and the "5.5–9.3 %, non-convergent" claim in
   `VOF_PLAN.md` §13 item 8 that it produced, should be read with this correction.
2. **Do not gate a per-cell interfacial area on second-order convergence.** It is first order in
   `h/R` for every construction measured, with the analytic controls above as the reference. The
   honest gate is a stated tolerance at a stated `h/R`, plus the requirement that the estimator not
   ADD to the floor (mode 3: −0.006…−0.21 % on the sphere at S = 16, i.e. at the floor).
3. **If a rung needs an area whose SUM converges, the lever is a JOINED surface, not a better
   per-cell normal.** Candidates, in order: the marching-cubes area of the `C = ½` level set
   (already 3–10× better on every field measured here, and cheap), or a partition-of-unity
   paraboloid area over the band. Both are new geometry, i.e. a rung of their own — call it P3d —
   and both should be gated on the corrected probe of item 1 and on the cylinder, whose reference
   is exact.
4. **Quote `β_eff` and `A/(4πR²)` beside every P3 number** (WO-P3b item 4 stands), and now also the
   **area mode**: the two Scriven tables above differ by 0.7 pp of `β_eff` on nothing else.

### Open

* **P3 remains NOT closed** at 1.31 % (Ja 0.5) / 1.83 % (Ja 2), mode 3, 128³. The remaining
  deficit is `−2.5 %` of area and it is quantitatively the first-order term; a P3d joined-surface
  area is the named lever and it is a Fable decision whether the rung is worth it.
* **Mode 3 is not the default.** It is better on every gate that moves and byte-identical on every
  gate that does not, but it fails the "converging" half of the WO's a-priori gate — which nothing
  can — and the campaign's rule is that a default changes on a passed gate, not on a better number.
  The recommendation is to make it the default TOGETHER with the corrected gate of item 2, in the
  same commit that records the P3d verdict.
* **The tilted-plane rows of the probe** are still quoted against marching cubes, which carries a
  domain-edge convention (it reads 95² where the exact answer is 96²). The (1,1,0) row now has an
  analytic reference in this entry; the (1,1,1) and (1,2,3) rows do not, and should get one before
  anyone reads their percentages as accuracy.

---
## WO-P3d findings (follow-up of WO-P3c) — the JOINED interfacial area: marching tetrahedra on the cell-centre lattice — 2026-09-03, Opus

Branch `vof-p3d`, worktree `../flow-p3d`, from `origin/main` at `59e9afa`. Backend **nvidia-cuda**
(`build_cuda`, `build_ktest_cuda`, `build_kmpi_cuda`), `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, one
solver process at a time on the shared GPU. No run printed `preconditioner produced non-finite z`.

**Verdict up front, in four statements.**

1. **The joined surface closes the area question.** `Sigma A / 4 pi R^2 - 1` on the corrected
   (sub = 16) sphere probe is **+0.011 % at R = 8** and **+0.008 % at R = 28**, against mode 0's
   −0.22 … −0.48 % and mode 3's −0.08 … −0.21 %, and on the CYLINDER — the control whose reference
   `2 pi R n_z` is exact — **−0.007 / −0.003 %** at R = 12/20 against **−1.59 / −1.42 %** (mode 0)
   and **−1.65 / −1.18 %** (mode 3), observed order **2.04**. WO-P3c's first-order per-cell floor
   is gone, which is what "the pieces JOIN" means quantitatively.
2. **The work order's primary design is REFUTED and its named follow-on is what works.** Marching
   tetrahedra on the raw `C = 1/2` level set (`set_phase_change_area(4)`) reads **+4.2 … +5.8 %** on
   the sphere with no convergence and **+20.7 %** on a tilted plane. The same sheet built on the
   zero of the PLIC-reconstructed signed distance (mode 6) reads +0.011 % and +0.0002 %. Mechanism
   in §"why the colour source fails".
3. **How the two endpoint planes are combined is worth three decades**, and it is a second thing the
   work order could not have known: averaging the two ROOTS gives +0.504 % at R = 8, blending the
   two signed-distance FUNCTIONS gives **+0.011 %**. The blend ships.
4. **P3 is now the ENERGY rung's problem, not the area's.** (Scriven table below.)

### What shipped

`src/vof/marching_cubes.hpp` (container-free) and `src/vof/marching_cubes_field.hpp` (the g = 3
block driver — a SIBLING of `VofInterfaceArea`, not an edit to WO-P3c's gated passes, hard rule 1);
four new `set_phase_change_area` modes; `phase_change_diagnostics()['area_orphan']`; the study
driver's `--area-probe=-5..-8` (the PERIODIC plane family) and `--area-advect`; the ctest gate
**K6**. `pcAreaCascadeCompute` dispatches on the mode; nothing else in `pcBuildInterface` moved.

| mode | sheet | edge crossing | deposit |
|---|---|---|---|
| **4 `kAreaMcColour`** | marching tets, `C = 1/2` | linear in `psi = 1/2 - C` | whole triangle to the centroid's cell |
| **5 `kAreaMcColourSplit`** | the same | the same | clipped to each cell's cube |
| **6 `kAreaMcPlic`** | marching tets, PLIC signed distance | the blend of the two cells' own planes | centroid |
| **7 `kAreaMcPlicSplit`** | the same | the same | clipped |

Three design points, each of which is the answer to a measurement:

* **Marching TETRAHEDRA (Kuhn's 6-tet split of the dual cube), not the 256-case cube table.** Four
  corner signs give one of three topologies, so the case analysis is a `popcount` and the kernel is
  table-free; and the Kuhn decomposition is **translation invariant**, so two neighbouring dual
  cubes split their shared face on the SAME diagonal and the sheet is watertight with no
  ambiguous-face rule. The dual cube's 8 corners are 8 CELL CENTRES, so the g = 3 colour block
  carries the whole stencil (`+-1` for the cubes, `+-2` counting the MYC normals) and there is **no
  new halo**.
* **The deposit is a GATHER.** A triangle born inside a dual cube can land in any of that cube's 8
  corner cells, which as a scatter means atomics and an arrival-order-dependent sum. So a cell walks
  the **8 cubes it is a corner of**, re-derives every triangle of each, and keeps its own share. The
  8x redundancy buys bitwise decomposition-independence by construction — the WO-P01 lesson applied
  to a quantity that is naturally a scatter.
* **A piece landing in a cell the wisp predicate calls PURE is RETARGETED**, not dropped. Measured:
  exactly the **12 axis-tangent pole cells** of a sphere, where the cell outside the interface is
  exactly `C = 1` while the `C = 1/2` sheet still crosses its cube — 1.51 h^2, i.e. **0.19 % of the
  area at R = 8**. The phase-change consumer reads `A` only on interfacial cells, so that area
  would have been silently dropped from `int mdot dA`; worse, the drop CANCELS against the sheet's
  own error (the usable sum reads +0.32 % where the sheet reads +0.50 % — flattering by
  coincidence). The retarget is a permutation of the booking: K6 gates that the 8 corners' total is
  unchanged by it, to 1e-14.

### Gate (a) — the a-priori probe

**The sphere** (`--area-probe … --area-sub 16`, 128^3, exact fractions, no time stepping),
`Sigma A/4 pi R^2 - 1`:

| R | 4 | 6 | **8** | 12 | 16 | 20 | 28 |
|---|---|---|---|---|---|---|---|
| mode 0 (P3c) | −0.730 | −0.800 | **−0.224** | −0.223 | −0.439 | −0.443 | −0.484 % |
| mode 3 (P3c) | −0.512 | −0.613 | **−0.082** | −0.006 | −0.175 | −0.199 | −0.214 % |
| **mode 4/5** | +4.180 | +5.197 | **+4.588** | +5.432 | +5.607 | +5.452 | +5.783 % |
| **mode 6/7** | +0.217 | +0.051 | **+0.011** | +0.011 | +0.022 | +0.009 | +0.008 % |

Modes 4 and 5 (and 6 and 7) are equal to the last digit, as they must be: the 8 corner octants
partition the dual cube, so the two deposit rules differ only in the per-cell DISTRIBUTION.
Mode 6/7 converges at order **4.3 over R = 4 -> 8** and then sits on a **0.01 % floor** — it is
already below the probe's own reference uncertainty (the reference is `4 pi R^2` with R from the
sampled volume). The honest statement of the convergence half of the gate is therefore the CYLINDER,
whose reference carries no z discretisation: **order 2.04** over R = 12 -> 20.

**The planes.** WO-P3c's open item 3 asked for an analytic reference for the tilted rows. There is
one, and it needed a different SCENE rather than a better reference: a single half-space in a
PERIODIC box is not a single plane — the wrap turns the domain faces into a second interface, which
a joined reconstruction reports and a per-cell one misses (the old `--area-probe=-2` row reads
+5.2 % for mode 6 for that reason alone). With an INTEGER normal, the level sets of
`f = n.x (mod L)` are closed flat surfaces of the torus and the co-area formula gives their total
area in closed form: `integral |grad f| dV = |n|_2 L^3 = L * A` hence `A = |n|_2 L^2` per level, so
the two levels bounding `f < L/2` carry **exactly `2 |n|_2 L^2`**. No edge convention, no seam, no
reference uncertainty. `--area-probe=-5,-6,-7,-8` (96^3, sub = 16):

| normal | exact | mode 0 | mode 3 | mode 4 | **mode 6/7** |
|---|---|---|---|---|---|
| (0,0,1) | 18432.0000 | **0.000 %** | **0.000 %** | **0.000 %** | **0.000 %** (18432.0000) |
| (1,1,0) | 26066.7844 | −0.217 % | −0.000 % | +20.677 % | **+0.0002 %** |
| (1,1,1) | 31925.1605 | −0.267 % | −0.267 % | +21.884 % | **+0.001 %** |

The axis-aligned row is 96^2 to ten digits in every mode. The tilted rows are the gate the work
order asked for (0.1 %) and mode 6 passes it by a factor of 400. **K6** proves the same statement at
kernel level, where it needs no box and no ghost policy: given one dual cube the exact signed
distance of a plane, the sheet inside it is that plane's cross-section of the cube (whose area is
`plicArea` of the same plane on the unit cube) to **< 1e-14** over 10 normals x 25 offsets, both
deposits agreeing to 1e-14 and the retarget changing the total by 1e-14.

**The cylinder** (`--area-shape cylinder`, exact in z, reference `2 pi R n_z`):

| R | sub | mode 0 | mode 3 | mode 4 | **mode 6/7** |
|---|---|---|---|---|---|
| 12 | 128 | −1.587 % | −1.648 % | +3.198 % | **−0.007 %** |
| 20 | 128 | −1.415 % | −1.179 % | +3.374 % | **−0.003 %** |

and the sub ladder for mode 6 at R = 20: −3.180 / −0.002 / −0.002 / −0.003 % at S = 16/32/64/128.
(The S = 16 row is WO-P3c's corrected-gate item 1 biting the joined area too, though at R = 12 it is
already at +0.025 %: a probe still has to show its own colour field converged.)

### Why the colour source fails, and why that is not "marching cubes fails"

`C(d)` — the liquid fraction of a cell as a function of its centre's distance from the plane — is
the Scardovelli-Zaleski piecewise CUBIC, so linear interpolation of it locates the crossing well
only over a step short compared with a cell. Kuhn's tets interpolate along the cube's 12 unit edges
but also along 6 `sqrt(2)` face diagonals and the `sqrt(3)` body diagonal. Every vertex on a long
edge is misplaced ALONG THE NORMAL, the sheet wrinkles, and a wrinkle only ever ADDS area — hence a
one-signed +5 % that does not converge, and +21 % on a 45-degree plane where the cubic is at its
most curved. Marching CUBES, which interpolates only the unit edges, reads −0.2 … +0.4 % on the very
same fields (the probe prints `skimage.measure.marching_cubes` as the external cross-check). So the
defect is the COMBINATION, and the fix is to interpolate a quantity that is linear over a
`sqrt(3)`-cell step: the signed distance. That is mode 6, and it recovers the exactness on the plane
that the tets otherwise lose.

The same reasoning explains the blend. With `phi_a(s) = d_a + s (n_a . dv)` the crossing each
endpoint cell's own plane predicts is exact on a plane but extrapolates a TANGENT plane up to
`sqrt(3)/2` cells on a curved interface, where it is wrong by `s^2/(2R)`. Averaging the two roots
leaves that error; blending the two distance FUNCTIONS,
`Phi(s) = (1-s) phi_a(s) + s phi_b(s)` (a quadratic in `s`, solved exactly), weights each plane by
how NEAR the crossing is to it. Measured on the sphere ladder, `Sigma A/4 pi R^2 - 1`:

| R | 4 | 6 | 8 | 12 | 16 | 20 | 28 |
|---|---|---|---|---|---|---|---|
| average of the two roots | +0.892 | +1.162 | +0.504 | +0.421 | +0.163 | +0.138 | +0.079 % |
| **blend of the two distances** | +0.217 | +0.051 | **+0.011** | +0.011 | +0.022 | +0.009 | +0.008 % |

Both are exact on a plane, so no planar gate could have chosen between them; the sphere probe is the
only instrument that separates them, and it separates them by three decades.

### Gate (b) — under ADVECTION: does the area drift with the wisp population?

`--area-advect`: a sphere carried 100 Weymouth-Yue steps (`advect_vof`) through a field that is
EXACTLY discretely divergence-free — two Taylor-Green cellular pairs plus a uniform drift, sampled
on the staggered faces, where the identity
`sin(k(x+h)) - sin(k x) = 2 sin(kh/2) cos(k(x+h/2))` makes the two axes' differences cancel term by
term because `x + h/2` of a face IS the cell centre the transverse factor is evaluated at. Measured
`max|div(open u)|` **0.0** (pure drift) and **1.9e-15** (with the cellular field), against
`advect_vof`'s own 1e-10 refusal threshold. Every area mode is read on the SAME field at the same
step. 128^3, sphere R = 16, sub = 16, cfl 0.2:

| scene | wisp cells, 0 -> 100 steps | mode 0 span | mode 3 span | mode 4 span | **mode 6/7 span** |
|---|---|---|---|---|---|
| pure translation | 0 -> **19 634** | **0.452 pp** (−0.439 -> +0.013 %) | 0.280 pp | 0.150 pp | **0.008 pp** (+0.022 -> +0.020 %) |
| + cellular deformation | 0 -> **738 439** (57 % of the domain) | **0.388 pp** (−0.439 -> −0.052 %) | 0.261 pp | 0.035 pp | **0.012 pp** (+0.022 -> +0.029 %) |

Read the first column first: Weymouth-Yue re-creates round-off colour residue in every cell its
sweeps touch, and after 100 steps of a cellular field **738 439 cells** satisfy the interfacial
predicate while the liquid volume has moved by 8e-14. **The per-cell PLIC area drifts by half a
percentage point with that population** — it is summing over those cells — and the joined sheet,
which is built from the `C = 1/2` crossing and never enumerates them, moves by 0.01 pp. That is
gate (b), and it also says something about mode 0 that no static probe could: 0.45 pp of its number
is wisps.

### Gate (c) — Scriven, 128^3, ratio 100, similarity start, MUSCL, `R 6 -> 20`

Both mode-0 rows reproduce WO-P3b/WO-P3c **to the digit** (2.002 / −2.517 / −3.380 and
2.636 / −2.568 / −3.375), so the harness is faithful. No run capped (30–34 of 600).

| Ja | area mode | last-half max \|ΔR\|/R | **β_eff/β − 1** | `A/(4πR²) − 1`, last half | area-avg ṁ | `band_div` | deposit fallback | iters |
|---|---|---|---|---|---|---|---|---|
| 0.5 | **0** | 2.002 % | −2.517 % | −3.380 % | −0.450 % | 2.40e-03 | 48 | 34/600 |
| 0.5 | 3 (P3c) | 1.307 % | −1.863 % | −2.498 % | −0.694 % | — | — | — |
| 0.5 | **6** | **1.036 %** | **−1.655 %** | −2.151 % | −0.926 % | **6.01e-12** | **0** | 30/600 |
| 0.5 | 7 | 1.083 % | −1.621 % | −2.206 % | −0.754 % | 4.27e-03 | 48 | 34/600 |
| 2 | **0** | 2.636 % | −2.568 % | −3.375 % | +0.310 % | 1.06e-02 | 48 | 31/600 |
| 2 | 3 (P3c) | 1.830 % | −1.766 % | −2.496 % | +0.251 % | — | — | — |
| 2 | **6** | **1.486 %** | **−1.475 %** | −2.190 % | +0.145 % | **1.08e-11** | **0** | 30/600 |
| 2 | 7 | 1.500 % | −1.482 % | −2.218 % | +0.214 % | 5.67e-03 | 48 | 31/600 |

**The gate is MISSED** (1.036 / 1.486 % against 1 %; β_eff 1.66 / 1.48 % against 1 %), and mode 6 is
nevertheless the best P3 the campaign has produced — it halves mode 0's radius error at both Ja and
takes `β_eff` from −2.5 % to −1.5 %. **Rule 4: this is the FOURTH failure of the P3 1 % gate
(WO-P23, WO-P3b, WO-P3c, WO-P3d) and the run stops here with the mechanism.** Ja = 10 was NOT run:
it is indicated only after Ja 0.5 and 2 pass, and its thermal boundary layer is sub-cell at 128^3
(WO-P23).

Three things to read out of it.

* **The mechanism has MOVED, and that is the finding.** The area estimator is no longer the
  limiter: on a resolved sphere it is +0.011 %, on the cylinder −0.007 %, on a tilted plane
  +0.0002 %, and under 100 WY steps it moves by 0.01 pp. Yet **the RUN's area is still −2.15 %**
  below `4πR²`. Whatever is missing is now in the COLOUR FIELD the coupled phase-change run
  carries, not in the geometry read off it — the deficit survives an estimator that measures an
  exact sphere to one part in 10⁴. `Δβ_eff ≈ ε/2` still holds across the change: −1.23 pp of area
  (−3.38 → −2.15) predicts +0.62 pp of β_eff and +0.86 pp is observed.
* **It is not the initialisation.** On the sub = 4 sphere the run starts from — the field whose
  quantization produced WO-P3b's whole table — the joined sheet reads **+0.507 / −0.223 / +0.006 /
  −0.245 %** at R = 6/10/14/20 while `Σ A_PLIC` reads **−9.304 / −6.369 / −4.739 / −6.487 %**. The
  sheet is essentially IMMUNE to the sliver quantization: it is built from the `C = ½` crossing and
  a cell that rounds to pure simply has no crossing in it, while its neighbours' crossings still
  bound the same surface. So WO-P3c's corrected-gate item 1 ("any area probe must show its own
  colour field converged") applies to a per-cell area and NOT to this one, and the `--sub 16` run
  ablation is the direct confirmation (below).
* **Mode 6 removes the deposit failure as a side effect, and mode 7 does not.** `band_div` goes
  2.4e-03 → **6.0e-12** and the deposit fallback 48 → **0** under the CENTROID deposit, because a
  cell the sheet books nothing to has `A = 0`, hence `S = 0`, hence never needs a pure-gas donor at
  all — precisely the 48 cells on a curved interface whose `+n` walk fails (WO-P23 mechanism 5).
  The SPLIT deposit gives every touched cell some area and keeps the 48. That, plus the marginally
  better gate, is why **mode 6 is the recommended one of the four** even though 6 and 7 carry
  identical total area.

### Where the remaining 2 % of area is, and where it is NOT

Two ablations, both on the Ja = 0.5 / mode 6 run.

**D1 — the sub = 4 sphere the run starts from** (`--area-probe 6,10,14,20 --area-sub 4`):

| R | 6 | 10 | 14 | 20 |
|---|---|---|---|---|
| mode 0 `Σ A_PLIC` | −9.304 % | −6.369 % | −4.739 % | −6.487 % |
| **mode 6, the joined sheet** | **+0.507 %** | **−0.223 %** | **+0.006 %** | **−0.245 %** |

The joined sheet is **essentially immune to the sliver quantization** that produced WO-P3b's entire
table and that WO-P3c had to fix by refining the probe. The reason is structural: the sheet is built
from the `C = ½` crossing, and a cell whose true fraction rounds to pure simply has no crossing in
it while its neighbours' crossings still bound the same surface — where a per-cell area, which
enumerates mixed cells, loses that cell's whole polygon. **WO-P3c's corrected-gate item 1 therefore
applies to a per-cell area and NOT to this one.**

**D2 — the same run started from a sub = 16 colour field** (P3c's ablation, repeated on mode 6):
`max |ΔR|/R` **1.036 → 1.124 %**, `β_eff` −1.655 → −1.630 %, `A/(4πR²)` −2.151 → −2.192 %, `R₀` from
the colour field +0.0017 % instead of −0.0287 %. **The initialisation moves nothing** (as it did not
for mode 0).

So the −2.15 % is a property of the colour field a COUPLED phase-change run carries, and neither the
estimator nor the start. What distinguishes that field from the advection gate's (where the same
estimator holds +0.02 % over 100 WY steps with 738 k wisp cells) is the **plane-shift regression
with clip-and-redistribute**: every step moves each interfacial cell's plane by `mdot A dt/ρ_l` and
pushes the part that will not fit into neighbours, which is a colour update no reconstruction
generated. That is the next instrument, and it is a rung of its own — the same shape of finding
WO-P3c made one level up.

### Gate (d) — the planar rungs, and gate (e) — inertness

`PECLET_P3C_AREA=<mode>` re-runs every scene of both phase-change binaries on a non-default area.
Printed values, `tests/kokkos/test_vof_phase_change`:

| gate | mode 0 (default) | mode 3 | mode 4 | **mode 6** | **mode 7** |
|---|---|---|---|---|---|
| **K5** (the P3c kernel gate) | metric 0.0, normal 2.2e-16, m1 2.2e-16, m2 1.1e-15, 38 rows bitwise | identical | identical | identical | identical |
| **P0a** planar regression, 1000 steps | 1.776e-14 | **byte-identical** | **byte-identical** | **byte-identical** | **byte-identical** |
| **P0b** ratio 100, closed column | u_gas rel **0.000e+00** | 0.000e+00 | −1.752e-16 | **0.000e+00** | **0.000e+00** |
| **P1 / P1'** Stefan N = 64 | +1.3099 % / −0.0139 % | identical | identical | **identical** | **identical** |
| **ENERGY** uniform-T at rcp ratio 1e4 | 0.000e+00 | identical | identical | **identical** | **identical** |
| **INERT** ṁ ≡ 0 | 0.000e+00 | identical | identical | **identical** | **identical** |
| **P2** sucking N = 64 | +0.1791 % | +0.1929 % | +0.1929 % | +0.1929 % | +0.1929 % |

**P0a, P0b, P1, P1', ENERGY and INERT are BYTE-IDENTICAL on the joined sheet**, which is a stronger
statement than "digit-level" and it is not an accident: on an axis-aligned plane the sheet's
cross-section of every dual cube is an exact unit square and the crossing lands in exactly the cell
whose colour is mixed, so `A = 1.0` on exactly the cells `plicArea` gives 1.0 to. K6's
`|colour-source axis − exact|` is **0.000e+00** — bitwise at kernel level, over 22 axis-aligned rows.
The only scene that moves is P2 (+0.0138 pp), which is the same 0.0138 pp modes 1/3/4 move it by and
is the scene's own red–black parity asymmetry (WO-P01 finding 3), smaller than its host-vs-CUDA
spread.

**K6** (new): 199 rows over 10 normals x 25 offsets — `|PLIC-source sum − exact|` **1.68e-14**,
`|split − centroid|` **1.59e-14**, `|retargeted − plain|` **1.59e-14**, `|colour-source axis −
exact|` **0.000e+00**. The 1.7e-14 is ~75 eps on a sum of a dozen square-rooted cross products, i.e.
the round-off of the SUM.

`tests/kokkos` at the shipped default: **33/33**.

### Gate — MPI, np 1/2/4 (`test_vof_phase_change_mpi`, 64x4x4, the ORB cutting the interface)

| case | mode 0 | **mode 6** | **mode 7** |
|---|---|---|---|
| **P0a** 1000 kinematic steps | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** |
| **P1** Stefan, 280 coupled steps | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** |
| P2 sucking, 55 coupled steps (interface position) | 1.5e-05 / 1.5e-05 / 5.6e-05 | **0.0 / 7.9e-16 / 7.9e-16** | **0.0 / 0.0 / 0.0** |

P0a and P1 are **bitwise at every area mode** — the sheet is a pure local stencil on the colour
block with a fixed-order gather and no atomic scatter, so it is decomposition-independent by
construction. (The wider `tests/kokkos_mpi` regression battery at the shipped default was still
running when this entry was written; nothing in it can reach the new code, which is unreachable
below `set_phase_change_area(4)`.) **The P2 row improves by eleven orders**, which is a second thing worth recording: that
scene's known distributed sensitivity comes from the red–black energy solve giving the MYC normal a
~1e-8 transverse component, and a per-cell area is LINEAR in that normal while the joined sheet —
built from `C = ½` crossings — is not. So mode 6/7 makes the sucking interface bitwise across np as
well.

### The verdict, and what shipped as the default

Gate (a) **PASSES** with room: 0.011 % on a sub = 16 sphere at R = 8 (gate 0.5 %), 0.0002 % on the
tilted plane (gate 0.1 %), 0.007 % on the cylinder (gate 0.5 %), bitwise on an axis-aligned plane,
converging (cylinder order 2.04; sphere order 4.3 from R = 4 to 8 and then a 0.01 % floor). Gate (b)
**PASSES** (0.008–0.012 pp over 100 WY steps against the per-cell area's 0.39–0.45 pp). Gate (d)
**PASSES** byte-identically and the MPI gate is bitwise. Gate (c) **FAILS** at 1.036 % / 1.486 %.

The work order's rule is explicit: *"If (a) and (c) pass, make it the default … otherwise leave
mode 0 and ship it as an instrument with the mechanism recorded."* **(c) does not pass, so the
default stays mode 0** and modes 4–7 ship as instruments. This is also the campaign's standing rule
(a default changes on a passed gate, not on a better number) and the same call WO-P3c made for
mode 3.

**But the recommendation attached to it is different from WO-P3c's**, and it should be recorded as
such: mode 3's case for the default rested on being better on the gates that moved; **mode 6's rests
on the gate having been re-derived**. It is the only construction measured in this campaign that is
at the floor on every a-priori geometry (sphere, cylinder, axis-aligned plane, tilted plane), the
only one that does not drift with the wisp population, the only one that makes P2 bitwise across np,
and it removes the deposit fallback entirely. What it does not do is close P3 — because P3 is no
longer area-limited. **If Fable wants a default change, the honest gate to change it on is gate (a)
plus (b) plus (d), all of which it passes, with P3 quoted as the open rung it no longer explains.**

### Open

* **P3 remains NOT closed** at 1.036 % (Ja 0.5) / 1.486 % (Ja 2), mode 6, 128^3 — the best the
  campaign has produced and still outside 1 %. **Rule 4: fourth failure, stopped.** Ja = 10 not run.
* **The next instrument is the PLANE-SHIFT REGRESSION, not the area.** The area estimator measures
  an exact sphere to 1e-4 and an advected one to 1e-4, and the run's interface still carries 2.15 %
  more `4πR²` than area. The one operation in the coupled loop that no reconstruction generated is
  `C ← C − ṁ A dt/ρ_l` with clip-and-redistribute. The a-priori instrument is the analogue of
  `--area-advect`: apply `apply_phase_change(dt)` with a UNIFORM ṁ to an exact sphere with no energy
  solve and no velocity, and watch `Σ A` against `4πR²(t)` — the exact answer is known
  (`R(t) = R₀ − ṁ t/ρ_l`), and any drift is the regression's.
* **Mode 6 vs mode 7.** Identical totals by construction; mode 6 is marginally better on both
  Scriven rows and removes the deposit fallback, mode 7 spreads the area over every touched cell.
  Mode 6 is the recommended one; mode 7 is kept because the split IS the physically-distributed
  answer and the difference between them is a measurement of how much the per-cell DISTRIBUTION
  matters (here: 0.05 pp of `max|ΔR|/R`, and the whole of the `band_div` improvement).
* **The tilted-plane rows −2/−3/−4 of the old probe** should be read with the periodic-seam
  correction above; the new `-5..-8` rows are the ones with an exact reference. WO-P3c's open
  item 3 is thereby closed for (1,1,0) and (1,1,1) and superseded for the rest.
* `W12`'s `vof_block_stats()['area']` still uses the PLIC polygon (mode 0). It is now measurably
  0.2–0.5 % low on a resolved sphere and drifts ~0.4 pp with the wisp population; switching it is a
  one-line change to a shipped, gated number and is left to whoever re-validates W12.

---
## WO-P3e findings (follow-up of WO-P3d) — the interface REGRESSION: an a-priori probe, and the defect that produced the number the last two work orders chased — 2026-09-03, Opus

Branch `vof-p3e`, worktree `../flow-p3e`, from `origin/main` at `ee7e7e6` (the commit that made
`set_phase_change_area(6)` the default). Backend **nvidia-cuda** (`build_cuda`, `build_ktest_cuda`,
`build_kmpi_cuda`), `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, one solver process at a time on the
shared GPU. No run printed `preconditioner produced non-finite z`. Every Scriven run below is clean
under rule 3b (max pressure iterations **30** against the 600 cap, none capped).

**Verdict up front, in four statements.**

1. **The `-2.15 %` "run area deficit" that WO-P3d handed to this work order does not exist.** It is
   the study driver comparing `phase_change_diagnostics()['interface_area']`, which
   `pcBuildInterface` measures at the **head** of the step, with `R` read from `get_vof()` at its
   **end** — one whole `dR` apart. Recomputed on the same field at the same time
   (`vof_interface_area()` after the step), the run's sheet reads **+0.043 %** (Ja 0.5) and
   **+0.041 %** (Ja 2) of `4 pi R^2`, not −2.15 %. The tell was already in WO-P3d's own table and
   nobody read it: the ratio is proportional to `dt`, so the LAST row of a run — whose `dt` is the
   leftover `te - t` — reads **−0.03 %** at Ja 0.5 (`dt` 0.107 against a typical 2.4) and −0.66 %
   at Ja 2 (`dt` about a third of typical).
2. **The regression's plane shift is not the mechanism either, by four orders of magnitude.** The
   probe measures its error as *exactly* `delta/R` — the linearization `dV = mdot A dt/rho_l` of
   the swept volume `int_0^delta A(s) ds` — reproduced to two digits at every radius and every
   step size measured. And the Scriven run's own regression step is
   **`delta = mdot dt/rho_l = 0.7 ... 1.9e-3` cells**, i.e. `delta/R ~ 1e-4`: at density ratio 100
   the regression supplies only `rho_v/rho_l` of the interface motion and Weymouth-Yue advection
   by the liquid velocity supplies the other 99 %, while the time step is set by the CFL on the
   latter. Measured directly on an advection-realistic field at the run's own `delta`: removed
   volume **+0.0016 %**, new radius **+0.00000 %**.
3. **The clip-and-redistribute is quiet at the run's step and is what limits the probe at large
   `delta`** — 0 to 12 cells clipped per step in the run against ~7000 interfacial cells, residue
   moved `<= 4e-3` of a removed volume of 5.8 (7e-4 relative), `unresolved = 0` everywhere. On the
   probe it is the *only* error left once the shift is exact, and it is what the `delta >= 0.2`
   rows measure.
4. **The gate is still missed and the mechanism has moved again — to the FLUX.** 1.036 % (Ja 0.5)
   and 1.486 % (Ja 2), `beta_eff` −1.655 / −1.475 %, with the area now measured right, the
   regression exonerated and the redistribute quiet. What is left is `mdot` from the energy solve
   on a moving, curved interface: the area-weighted `mdot` drifts from **+10.4 %** at the first
   sample to **−2.7 %** at the last (Ja 0.5) and from **+1.3 %** to **−1.8 %** over the last half
   (Ja 2). **Rule 4: this is the FIFTH failure of the P3 1 % gate (WO-P23, P3b, P3c, P3d, P3e) and
   the run stops here with the mechanism.** Ja = 10 was NOT run (indicated only after 0.5 and 2
   pass; its thermal boundary layer is sub-cell at 128^3).

### What shipped

* `tests/study/vof_scriven.py --regress-probe R1,R2,... [--regress-delta ...] [--regress-modes ...]
  [--regress-advect N]` — the a-priori regression instrument (below), and two
  read-outs added to the coupled run: **`delta` per step** (`removed_volume/area`, with `delta/R`)
  and **`A_end`** (`vof_interface_area()` recomputed after the step) beside the stale
  `interface_area`, plus the radius each of them implies.
* one one-line defect fix in the same driver (`--help` crashed; see below).

**`src/` is UNTOUCHED by this work order.** `git diff origin/main -- src tests/kokkos
tests/kokkos_mpi` is empty: only `tests/study/vof_scriven.py`, this file and `CLAUDE.md` change. A
corrected plane shift WAS implemented, measured and then removed — see "the corrected shift, and why
it is not shipped" below.

### Gate (i)/(ii)/(iii) — the EXACT sphere, no time stepping

128^3, exact fractions at `sub = 16` (WO-P3c's corrected probe resolution), uniform prescribed
`mdot = 1` with `rho_l = 1` so the normal displacement is exactly `delta = dt`, ONE
`apply_phase_change(delta)`. The bubble is the GAS, so evaporation grows it: the exact liquid
volume removed is the shell `4 pi R^2 delta (1 + delta/R + delta^2/3R^2)` and the exact new radius
from the volume deficit is `R + delta`.

`dV/dV_exact - 1` and `(R1 - (R0+delta))/(R0+delta)`, for the SHIPPED (linearized) shift and for
the corrected swept-volume shift that was implemented and then removed (below):

| R | mode | delta | **dV shipped** | dV corrected | **R1 shipped** | R1 corrected | `delta/R` |
|---|---|---|---|---|---|---|---|
| 8 | 0 | 0.05 | -0.8452 % | -0.8971 % | -0.0052 % | -0.0055 % | 0.625 % |
| 8 | 0 | 0.10 | -1.4611 % | -1.9306 % | -0.0178 % | -0.0236 % | 1.250 % |
| 8 | 0 | 0.20 | -2.6776 % | -2.3419 % | -0.0638 % | -0.0558 % | 2.500 % |
| 8 | 0 | 0.50 | -6.2084 % | -6.0648 % | -0.3453 % | -0.3373 % | 6.250 % |
| 8 | **6** | 0.05 | **-0.6115 %** | -0.7908 % | -0.0038 % | -0.0049 % | 0.625 % |
| 8 | **6** | 0.10 | **-1.2288 %** | -2.1595 % | -0.0150 % | -0.0263 % | 1.250 % |
| 8 | **6** | 0.20 | **-2.4482 %** | -3.6993 % | -0.0583 % | -0.0881 % | 2.500 % |
| 8 | **6** | 0.50 | **-5.9873 %** | -10.7928 % | -0.3330 % | -0.6019 % | 6.250 % |
| 12 | 0 | 0.05 | -0.6376 % | -0.7537 % | -0.0026 % | -0.0031 % | 0.417 % |
| 12 | 0 | 0.10 | -1.0499 % | -0.9314 % | -0.0086 % | -0.0076 % | 0.833 % |
| 12 | 0 | 0.20 | -1.8676 % | -1.8423 % | -0.0301 % | -0.0297 % | 1.667 % |
| 12 | 0 | 0.50 | -4.2672 % | -6.4484 % | -0.1642 % | -0.2484 % | 4.167 % |
| 12 | **6** | 0.05 | **-0.4047 %** | -0.5939 % | -0.0017 % | -0.0024 % | 0.417 % |
| 12 | **6** | 0.10 | **-0.8180 %** | -1.0669 % | -0.0067 % | -0.0088 % | 0.833 % |
| 12 | **6** | 0.20 | **-1.6376 %** | -2.5846 % | -0.0264 % | -0.0417 % | 1.667 % |
| 12 | **6** | 0.50 | **-4.0428 %** | -9.3714 % | -0.1556 % | -0.3614 % | 4.167 % |
| 20 | 0 | 0.05 | -0.6917 % | -0.6398 % | -0.0017 % | -0.0016 % | 0.250 % |
| 20 | 0 | 0.10 | -0.9393 % | -0.9224 % | -0.0046 % | -0.0046 % | 0.500 % |
| 20 | 0 | 0.20 | -1.4322 % | -1.7675 % | -0.0140 % | -0.0173 % | 1.000 % |
| 20 | 0 | 0.50 | -2.8912 % | -7.1195 % | -0.0689 % | -0.1697 % | 2.500 % |
| 20 | **6** | 0.05 | **-0.2405 %** | -0.4465 % | -0.0006 % | -0.0011 % | 0.250 % |
| 20 | **6** | 0.10 | **-0.4893 %** | -1.1455 % | -0.0024 % | -0.0057 % | 0.500 % |
| 20 | **6** | 0.20 | **-0.9843 %** | -2.8083 % | -0.0097 % | -0.0275 % | 1.000 % |
| 20 | **6** | 0.50 | **-2.4499 %** | -9.3318 % | -0.0583 % | -0.2226 % | 2.500 % |

Read the mode-6 rows against the last column: **the shipped regression's volume error IS
`-delta/R`, to two digits, at every radius and every step size.** That is the linearization and
nothing else — `A delta` is the first term of `int_0^delta A(s) ds`, and for a sphere
`sum A(s) = 4 pi (R+s)^2`, so the missing term is exactly `delta/R`. (The mode-0 rows carry that
term PLUS mode 0's own `-0.22 ... -0.48 %` per-cell area deficit, WO-P3c, which is why they do not
read the last column.)

### The corrected shift, and why it is not shipped

The work order's first candidate — "the shift should conserve the NORMAL displacement `delta` per
cell" — was implemented (`vof::pcSweptFactor`: `f = (V(alpha) - V(alpha - delta |m|_2))/(delta
A_plic)` from the cell's own PLIC plane, continued linearly with `A_plic` once the plane leaves the
cell, applied as a correction on whichever area `set_phase_change_area` selects), measured in two
bookings, and then REMOVED. The measurements are why.

* **Booking A** — the correction `A_plic (f - 1)` added to EVERY interfacial cell. This is the only
  variant that reaches the work order's 1e-3 gate on (i)/(ii): R = 20, mode 6, `dV` **−0.2405 →
  +0.0628 %** at `delta` = 0.05 and **−0.4893 → −0.0832 %** at 0.10, with `R1` at +2e-6 and −4e-6.
  It reaches it by giving area to cells the mode-6 sheet booked NONE to — and those are exactly the
  cells whose `+n` walk finds no pure gas cell, so the deposit fallback comes back:
  `fallback` **0 → 48** and `band_div` **1.1e-11 → 1.6e-04** on the Ja = 2 Scriven run, undoing
  WO-P3d's own side benefit.
* **Booking B** — the same correction, applied only where the area mode booked surface (`A > 0` is
  mode 6's "no surface here" sentinel). No side effect, and it is the column in the table above:
  on mode 0, where the area IS the cell's own polygon, it removes 20 % of the `delta/R` term at
  `delta` = 0.05 and is WORSE beyond `delta` = 0.2; on mode 6 it is worse than doing nothing at
  every `delta`, because the PLIC polygon's per-cell curvature factor and the sheet's per-cell share
  are different distributions over the same cells.
* **Both move the Scriven gate by ≤ 0.02 pp** (booking A: 1.036 → 1.025 % at Ja 0.5 and
  1.486 → 1.470 % at Ja 2; booking B: 1.034 % and 1.481 %), which is what `delta/R ~ 1e-4` requires.

Two things this closes. **What limits a corrected shift is not the shift**: once any cell empties in
a step, the continuation of the swept region into the neighbour is carried by the
clip-and-redistribute — a `n_d^2`-weighted transport of the residue along `-n`, not a geometric
continuation — and it is what the `delta >= 0.2` rows measure (at `delta` = 0.5 on R = 20 it moves
731 of 2513 units of removed volume and takes `min C` to −0.196). **And a SHEET-consistent
correction does not go through the cell's own polygon at all**: the exact statement for a surface
moving along its normal is `A(s) = A(0)(1 + kappa s + K s^2)`, so the factor is
`1 + kappa delta/2 + K delta^2/3` with `kappa` the mean curvature the V3 cascade already computes —
low-noise, sheet-consistent, and a rung of its own. Since the term it corrects is 1e-4 of the
Scriven gate, that rung is not indicated by anything measured here.

**So nothing was shipped into `src/`**, and the campaign rule holds: a default (or an option) changes
on a passed gate, not on a better number in one corner of a ladder.

**(iii) the area, before and after.** Mode 6 reads `+0.011 %` of `4 pi R_0^2` before the step at
every radius (WO-P3d's floor, reproduced). After it, against `4 pi (R_0+delta)^2`:

| R = 20, mode 6 | delta 0.05 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|
| shipped (linearized) shift | +0.067 % | +0.470 % | +3.303 % | +11.981 % |
| corrected shift (booking B) | +0.060 % | +0.361 % | +2.058 % | +16.227 % |

**A linearized plane shift ROUGHENS the surface**, monotonically in `delta` — each cell's plane
moves by `dV/A(0)` rather than by `delta`, and the discrepancy is per-cell. At the run's
`delta = 1.8e-3` it is unmeasurable (below), which is why this shows up only on the ladder.

### Gate (iv) — isotropy: is the shift faceting the bubble?

Two read-outs on the same fields: the removed volume in bins of the cubic invariant
`s = u_x^4 + u_y^4 + u_z^4` (1/3 on the body diagonal, 1/2 on a face diagonal, 1 on an axis)
divided by the EXACT removal in the same bins; and the `l = 2` / `l = 4` real spherical-harmonic
moments of the removal density, normalised by its total. A cubic lattice can only excite `l = 4`
(the three mirror symmetries kill `l = 2`), so the `l = 2` row is the control.

| field | mode | delta | bins (body-diag → axis) | `l4/l0` actual | `l4/l0` of the EXACT removal |
|---|---|---|---|---|---|
| exact sphere R = 12 | 0 | 0.10 | +0.36 / −0.03 / −0.16 / −5.96 % | 3.3e-03 | 7.2e-04 |
| exact sphere R = 12 | **6** | 0.10 | +0.21 / −2.42 / −3.10 / +3.57 % | 3.5e-03 | 6.6e-04 |
| exact sphere R = 20 | **6** | 0.05 | +1.96 / +0.34 / −2.02 / +0.01 % | 7.0e-04 | 7.1e-04 |
| **advected R = 16** | 0 | 0.0018 | +0.52 / −1.05 / −0.85 / −0.50 % | 2.1e-04 | 2.2e-03 |
| **advected R = 16** | **6** | 0.0018 | +0.88 / −0.51 / −0.52 / −0.23 % | 4.8e-04 | 2.2e-03 |
| **advected R = 16** | **6** | 0.05 | +0.09 / +0.19 / −0.05 / −0.03 % | 2.6e-04 | 2.4e-04 |

`l = 2` is at round-off on every exact-sphere row (1e-17) and at 1e-4 on the advected ones, i.e. the
field's own asymmetry, not the shift's. **On the advection-realistic field the removal is isotropic
within the fractions' own noise** — the `l = 4` moment of the actual removal is *below* that of the
exact removal computed on the same sub-sampled fields. The few-percent bin structure on the pristine
exact sphere is the sliver-quantization pattern of the sub-sampled initialisation (WO-P3c's
corrected-gate item 1), not a directional bias of the shift: it does not survive one hundred
Weymouth-Yue steps, and a directional bias would.

### The ADVECTED sphere — the row that decides the work order

128^3, R = 16, `sub = 16`, then 100 Weymouth-Yue steps of the pure-translation solenoidal field of
WO-P3d gate (b) (`max|div(open u)| = 0.0`), so the fractions are the ones a running solver carries:
**24 458 mixed cells against the exact sphere's 7 184**, `|C - C_exact|_1 = 5.25`, volume moved by
`0.0004 %`. Then one regression step.

| mode | delta | dV (shipped shift) | R1 (shipped shift) | clipped at 0 | residue moved | pure-LIQUID cells touched |
|---|---|---|---|---|---|---|
| 0 | **0.0018 (the RUN's own)** | **+0.0016 %** | **+0.00000 %** | 1059 | 3.6e-06 | 0 |
| 0 | 0.05 | −0.2990 % | −0.0009 % | 1461 | 4.3e-01 | 0 |
| 0 | 0.10 | −0.6096 % | −0.0038 % | 1818 | 4.6e+00 | 1 |
| 0 | 0.20 | −1.2269 % | −0.0150 % | 2438 | 3.9e+01 | 20 |
| **6** | **0.0018** | **+0.4008 %** | **+0.00005 %** | **1** | **1.6e-05** | 0 |
| **6** | 0.05 | +0.0990 % | +0.0003 % | 146 | 9.7e-01 | 0 |
| **6** | 0.10 | −0.2129 % | −0.0013 % | 368 | 7.0e+00 | 1 |
| **6** | 0.20 | −0.8326 % | −0.0102 % | 922 | 4.6e+01 | 20 |

**At the run's own step size the regression is exact to 1e-5 in the radius and to 1e-4 … 4e-3 in the
volume, on an advection-realistic field.** (The mode-6 `+0.40 %` is not the shift: it is that mode's
AREA on this particular field — see the next section — and it enters `dV` because `dV = mdot A dt`.
The RADIUS, which is what the gate reads, is right at 5e-7.)

### A correction to WO-P3d's gate (b): the joined sheet's wisp immunity is the WISP GUARD's

On the *identical* 100-step field, the mode-6 sheet reads **+0.020 %** of `4 pi R^2` when the solver
has only `enable_vof` (WO-P3d gate (b), reproduced here to the digit) and **+0.412 %** when it has
`enable_phase_change`. The difference is one number: `enable_phase_change` sets
`set_vof_wisp_eps(0)` (WO-P23 mechanism 5b — the guard and phase change are incompatible on a curved
interface), so the interfacial predicate on the phase-change path is `pcInterfaceEps_ = 1e-12`
instead of the advector's `1e-8`, so the round-off wisp cells between those two thresholds get a PLIC plane
and contribute a spurious crossing to the sheet. Mode 0 reads `+0.013 %` either way, because a wisp
cell's PLIC polygon has essentially zero area while its *plane* is what the sheet interpolates
between. **WO-P3d's gate (b) is therefore not a statement about the configuration the phase change
runs in**, and the honest version of "the joined sheet does not drift with the wisp population" is
"…at `wispEps = 1e-8`". In the coupled Scriven run the wisp population is far smaller
(1.48 interfacial cells per `h^2`, against 7.6 on the 100-step translation) and the run's sheet does
read `+0.04 %` — so this does not move P3, but it is a live trap for any future consumer that reads
`vof_interface_area()` on the phase-change path after a long advection.

### The stale-area defect, and what it cost

`phase_change_diagnostics()['interface_area']` is filled by `pcBuildInterface`, which runs at the
**head** of `step()`; `R` is read from `get_vof()` after the step. The two are one `dR` apart, so
`A/(4 pi R^2)` is low by `2 dR/R`, and this scene runs at `dR ~ 0.18` cells per step against
`R = 6 ... 20` — i.e. **2.2 %**, which is the number WO-P3c and WO-P3d built a mechanism on ("the
run's interfacial area is still −2.15 % below `4 pi R^2` … whatever is missing is now in the COLOUR
FIELD the coupled phase-change run carries"). Recomputing on the current field:

| Ja | area mode | stale `A/(4 pi R^2) - 1` | **`A_end/(4 pi R^2) - 1`** |
|---|---|---|---|
| 0.5 | 6 | −2.151 % | **+0.043 %** |
| 2 | 6 | −2.190 % | **+0.041 %** |

and the tell that was already in WO-P3d's printed table: the ratio is **proportional to `dt`**, so
the LAST row of a run — whose `dt` is the leftover `te - t` — reads **−0.03 %** at Ja 0.5 (`dt`
0.107 against a typical 2.4) and −0.66 % at Ja 2 (`dt` about a third of typical). A quantity that
collapses by two orders when one step is made short is a per-step bookkeeping artefact, not a
property of the field.

Two consequences beyond this WO. `R_area = sqrt(A/4pi)` (added by this WO alongside) reads
−1.5 … −2.0 % against the exact radius when built on the stale area and **tracks `R_num` to
0.02 pp** when built on `A_end` — so the *sheet* and the *liquid-volume deficit* agree on the
bubble's size, and there is no "the field is smeared" residual to explain. And the WO-P3d inference
`Delta beta_eff ~ epsilon/2` applied to the stale number was coincidence: the area is right and
`beta_eff` is still −1.65 %.

### Gate (c) — Scriven, 128^3, ratio 100, similarity start, MUSCL, `R 6 -> 20`, area mode 6

| Ja | shift | **max \|dR\|/R** | **beta_eff/beta − 1** | `A_end/(4 pi R^2)` | area-avg `mdot`, last half | `band_div` | fallback | iters |
|---|---|---|---|---|---|---|---|---|
| 0.5 | **shipped** | **1.036 %** | **−1.655 %** | +0.043 % | −0.926 % | 6.0e-12 | 0 | 30/600 |
| 0.5 | corrected (A) | 1.025 % | −1.641 % | +0.043 % | −0.931 % | 3.8e-12 | 0 | 30/600 |
| 0.5 | corrected (B) | 1.034 % | −1.652 % | +0.043 % | −0.929 % | 5.6e-12 | 0 | 30/600 |
| 2 | **shipped** | **1.486 %** | **−1.475 %** | +0.041 % | +0.145 % | 1.1e-11 | 0 | 30/600 |
| 2 | corrected (A) | 1.470 % | −1.455 % | +0.041 % | +0.146 % | 1.6e-04 | **48** | 30/600 |
| 2 | corrected (B) | 1.481 % | −1.469 % | +0.041 % | +0.146 % | 1.8e-11 | 0 | 30/600 |

The shipped rows reproduce WO-P3d to the digit (1.036 / −1.655 and 1.486 / −1.475), so the harness
is faithful. **A corrected plane shift moves the gate by 0.002 … 0.016 pp** — which is what
`delta/R ~ 1e-4` predicts, and it is the quantitative close of the work order's first named
candidate. (The Ja = 2 `fallback 48` / `band_div 1.6e-04` row is booking A's side effect, described
above; it is the reason neither booking shipped.)

The regression's own census in the coupled run, printed per step: `delta` **8.8e-04 … 1.9e-03**
cells (Ja 0.5) and **6.8e-04 … 1.9e-03** (Ja 2), `delta/R` **9.6e-05 … 2.2e-04**; **0 to 12 cells
clipped** per step out of ~7000 interfacial; residue moved `<= 3.9e-03` against a removed volume of
~5.8 per step; `unresolved = 0.0` in every run.

### Where the deficit is now

The area is right (+0.04 %), the shift is right (1e-4 of `delta`), the redistribute is quiet
(7e-4 of the removed volume) and the deposit is at the floor (`band_div` 6e-12, `fallback` 0). What
remains is the **flux**, and its signature is a drift in time rather than an offset:

| Ja | `mdot_area` rel. error: first sample | mid | last | R rel. error: step 1 | half | end |
|---|---|---|---|---|---|---|
| 0.5 | **+10.4 %** | +0.5 % (t = 64) | **−2.7 %** | +0.086 % | −0.166 % | −1.036 % |
| 2 | +1.3 % (first of the last half) | — | **−1.8 %** | −0.034 % | −1.465 % | −1.486 % |

Ja = 2 acquires its whole deficit in the first ~40 steps (its thermal boundary layer is 2.8 cells at
`t_0`) and then carries it at a constant *relative* size — the fitted radius offset at the end is
−0.003 cells, i.e. it is a pure scale error and `beta_eff` reads it as a rate. Ja = 0.5 starts
correct and loses the radius steadily as `mdot_area` crosses from +10 % to −2.7 %. Neither is an
initialisation offset (WO-P3b) and neither is the area (this WO).

**What a P3f should measure, in this order.** (a) An ENERGY BUDGET: a liquid cell that becomes
interfacial has its superheat replaced by the plane-anchored `T_sat` condition, and the enthalpy
that disappears is enthalpy that should have evaporated liquid — on a growing bubble that is a
one-signed sink whose size scales with the number of cells the interface sweeps per step, which is
exactly a `mdot` deficit that grows with `R^2 dR/dt`. Instrument: `sum rho c_p T` plus
`h_lv sum(removed volume) rho_l` against the boundary heat flux, per step. (b) The `mdot`
one-sided fit's stencil on a CURVED interface: the 5^3 quadratic fit is taken along the PLIC normal
of the *cell*, and its samples straddle a curved isotherm, so the fitted `dT/dn` carries an
`O(h/R)` curvature bias of the same sign and size as what is measured — the a-priori instrument is
WO-P23's exact-analytic-state probe (`mdot` on an imposed Scriven profile at fixed `R`), run at
`R = 6, 10, 14, 20` instead of on a plane, which no work order in this campaign has done.
(c) Confinement, now that the area is not masking it: 192^3 at `R 6 -> 20` (1.5x clearance) and
`R 9 -> 30` (1.5x resolution) on **mode 6** — WO-P3b's rows were mode 0, where the −3.4 % area
dominated both.

### Gate (d) — the planar rungs, inertness, and MPI

**`src/`, `tests/kokkos` and `tests/kokkos_mpi` are byte-for-byte `origin/main`** (`git diff
origin/main --` over those three trees is empty), so every gate below is `main`'s own number, taken
here as the check that this worktree is measuring what it thinks it is.

* **Byte-identity, run rather than argued.** `test_vof_phase_change` built from this worktree and
  from a separate `origin/main` (`ee7e7e6`) checkout and run at 4 threads on nvidia-cuda: the full
  stdout — K1…K6, **P0a, P0b, P1, P1', ENERGY, INERT, P2** — is **BYTE-IDENTICAL** (`diff` empty).
  This was taken while the corrected shift WAS in the tree (as a default-off option) and is the
  measurement that it was inert; it is trivially still true now that it is gone.
* `tests/kokkos` at the shipped default: **33/33 passed** (nvidia-cuda, `OMP_NUM_THREADS=4`,
  1735 s).
* **MPI, np 1/2/4** (`test_vof_phase_change_mpi`, 64x4x4, the ORB cutting x so the interface crosses
  a rank boundary during every run), nvidia-cuda — `ctest -R vof_phase_change_mpi` **3/3 passed**:

| case | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| **P0a** 1000 kinematic steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| **P1** Stefan, 280 coupled steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| P2 sucking, 55 coupled steps (interface position) | 0.0 | 7.901e-16 | 7.901e-16 |
| P2 pointwise `max\|C_dist - C_ref\|` | 1.299e-14 | 1.299e-14 | 1.331e-14 |

For the record, the corrected shift was ALSO taken through this gate while it existed
(`PECLET_P3E_SWEPT=1`, the hook `PECLET_P3C_AREA` uses): **P0a and P1 bitwise at np 1/2/4** — the
swept factor is a per-cell function of that cell's own plane, with no reduction in it — while P2
moved to 4.573e-06 / 4.573e-06 / 1.294e-04, np-independent at np 1 and 2 and one decade larger at
np 4, the same pattern WO-P23 recorded for that scene with its own options (2.3e-05 / 7.3e-05 /
1.1e-04). On the single-rank battery with the option on, K1…K6 / P0a / P1 / P1' / ENERGY / INERT
were **byte-identical** (on an axis-aligned plane `V(alpha)` is linear, so the correction is
algebraically zero), P0b moved by 3.3e-14 relative and P2 by 0.035 pp (+0.1929 → +0.1579 %) — that
scene's documented round-off amplifier, and 2.5x what the AREA MODES move it by.

### One-line defect fixed on the way

`tests/study/vof_scriven.py --help` CRASHED (`TypeError: %o format: an integer is required`) since
WO-P3c: argparse `%`-expands every help string, and `--area-sub`'s says "6 % of the area". Escaped.

### Open, and the corrected gates this WO proposes

1. **P3 remains NOT closed** at 1.036 % (Ja 0.5) / 1.481 % (Ja 2), mode 6, 128^3 — the FIFTH
   failure, stopped under rule 4. Ja = 10 not run.
2. **A diagnostic filled at the HEAD of a step must be quoted against a state from the head of that
   step.** `phase_change_diagnostics()` is a census of the *last* `pcBuildInterface`, and this
   campaign built two work orders on a ratio of one of its entries to a quantity read after the
   step. The driver now prints `A_end` beside it and labels the old one `[STALE]`; the same caution
   applies to `removed_volume`, `interface_cells` and `mdot_mean`, all of which are pre-step
   quantities. **WO-P3c's and WO-P3d's `A/(4 pi R^2) = -2.15 %` rows should be read with this
   correction**, and with them the inference "the deficit is in the colour field the coupled run
   carries".
3. **WO-P3d's gate (b) is a statement at `wispEps = 1e-8`, not at the phase change's own 1e-12.**
   Either re-take it inside `enable_phase_change`, or state the guard beside the number.
4. **The regression is closed as a mechanism at any `delta/R <= 1e-3`**, which covers every
   density-ratio >> 1 boiling case, because there `delta = (rho_v/rho_l) x (interface motion)`. It
   is NOT closed for a ratio near 1 (P1's own Stefan regime, and any bubble in a liquid-liquid
   system), where `delta` IS the whole motion — the probe table above IS the gate for that case, and
   at `delta/R = 6 %` the shipped shift is 6 % low on the removed volume. Nothing in this campaign
   runs that case on a CURVED interface yet; when something does, the correction to reach for is the
   SHEET-consistent `A(1 + kappa delta/2 + K delta^2/3)` with `kappa` from the V3 cascade, not the
   per-cell PLIC sweep measured here.
5. **A P3f should not repeat the mesh study** until the flux is instrumented: WO-P3b's 64/128/192
   ladder and WO-P3c's confinement rows were both taken on mode 0, whose area error dominated them.
   Re-taking them on mode 6 is item (c) above — cheap — but the two a-priori instruments named in
   (a) and (b) are what decide it.
6. **What limits the regression at large `delta` is the clip-and-redistribute**, not the shift: at
   `delta = 0.5` on an R = 20 sphere it moves 731 of 2513 units of removed volume and takes `min C`
   to −0.196, and a corrected shift is then WORSE than the linear one. If a future rung ever needs a
   half-cell regression step, that is the piece to redesign — the push along `-n` with `n_d^2`
   weights is a first-order transport of the residue, not a geometric continuation of the swept
   region into the neighbour cell.

---

## WO-P3f findings (follow-up of WO-P3e) — the FLUX: two first-order errors of opposite sign, measured a priori, and why P3's 1 % is their cancellation — 2026-09-03, Opus

Branch `vof-p3f`, worktree `../flow-p3f`, from `origin/main` at `22b4f62`. Backend **nvidia-cuda**
(`build_cuda`, `build_ktest_cuda`, `build_kmpi_cuda`), `OMP_NUM_THREADS=4 OMP_PROC_BIND=false`, one
solver process at a time. No run printed `preconditioner produced non-finite z`. Every Scriven run
below is clean under rule 3b (max pressure iterations **27–44** against the 600 cap, none capped).

**Verdict up front, in five statements.**

1. **The shipped `mdot` estimator is +5 … +8 % HIGH a priori** on the run's own configuration —
   exact sphere, exact Scriven similarity profile, one evaluation, no time stepping. Split by the
   new 2×2 probe: the INTERFACE's curvature contributes **+19.2 / +12.1 / +8.8 / +6.2 %** at
   R = 6/10/14/20 with observed order **0.91 / 0.93 / 0.98** in `h/R` — a clean first-order bias,
   and the largest single error in the rung. Nobody had ever run this estimator off a plane.
2. **The energy solve's own interfacial flux is a DIFFERENT number.** `q_gfm`, the heat the
   plane-anchored (GFM) rows actually draw, is not `mdot h_lv A_Gamma`: in the coupled run
   `-q_gfm/E_lat` runs **0.985 → 1.039** (Ja 0.5) and **0.947 → 1.009** (Ja 2). That mismatch is a
   genuine non-conservation — heat that leaves the liquid without evaporating anything — and so is
   the per-cell Dirichlet **overwrite**, at **−0.5 … −2.2 %** of `E_lat` (Ja 0.5) and
   **−3.7 … −29.5 %** (Ja 2).
3. **The work order's candidate (a) as stated is half right and the instrument says which half.**
   A cell CHANGING CLASS is not an energy sink: it does not move and its temperature is still in
   the field, so `e_enter` (measured at +3.5 … +8 % of `E_lat` at Ja 0.5 and +37 … +95 % at Ja 2)
   is a flux between two BOOKS, not a leak. The leak is the OVERWRITE, and conserving it is a
   real repair whose a-priori gate passes bitwise.
4. **Every repair that improves an a-priori gate makes the coupled gate WORSE.** Correcting the
   fit's curvature bias (a-priori bias `+6.2 % → −0.74 %` at R = 20, order `0.98 → 1.33`) takes
   Scriven from **1.036 % to 6.977 %** at Ja 0.5. Conserving the overwrite (a-priori identity
   residual **0.000e+00**, bitwise, every step) takes Ja 2 from **1.486 % to 3.183 %** and flips
   `beta_eff` from −1.475 % to **+2.975 %**.
5. **P3's 1–1.5 % is therefore not a residual to be closed; it is what is left when a `+5 %`
   estimator bias cancels a `≈ −5 %` energy leak.** The mesh ladder confirms it independently:
   refining at FIXED `R/L` makes the answer worse, **−1.073 → −1.655 → −2.557 %** in `beta_eff` at
   96³/128³/192³, because the cancelling term is the one that is O(h/R). **Rule 4: this is the
   SIXTH failure of the P3 1 % gate and the run stops here with the mechanism.** Ja = 10 was NOT
   run. Both repairs ship as options, both **default OFF**.

### What shipped

* `set_phase_change_budget(bool)` + `phase_change_budget()` — instrument (a), the energy budget of
  the energy solve (below). Two reductions and one extra cell field per solve; no kernel and no
  allocation when off. `apply_phase_change` also evaluates its `q_gfm` half, so an exact-state
  probe can ask the same question with no time stepping.
* `vof_scriven.py --mdot-probe R1,R2,… [--mdot-prof scriven,linear] [--mdot-geom sphere,plane]
  [--fit-curvature]` — instrument (b), the a-priori mass-flux probe.
* `set_phase_change_carry_conserve(bool)` — repair 1 (the overwrite's enthalpy, returned), and
  `phase_change_carry_ledger()`; `vof_scriven.py --carry`, `--carry-probe N` (its a-priori gate).
* `set_phase_change_fit_curvature(kappa)` + `vof::pcCurvedDistance` — repair 2 (the one-sided fits
  measure the distance to the CURVED interface), with `kappa = div(n)` PRESCRIBED; `--fit-curvature`.
* `vof_scriven.py --budget N` prints the budget every N steps of the coupled run.

**Byte-identity, run rather than argued.** `test_vof_phase_change` built from this worktree and
from a separate `origin/main` (`22b4f62`) checkout, run at 4 threads on nvidia-cuda: the full
stdout — K1…K6, **P0a, P0b, P1, P1', ENERGY, P2, INERT** — is **BYTE-IDENTICAL** (`diff` empty).
Both defaults are inert inside the kernels (`kappa == 0.0` skips the correction; the carry deposit
allocates nothing and launches nothing).

### Instrument (b) — the a-priori mass-flux probe, and the 2×2 that decides it

128³, exact fractions at `sub = 16`, one `apply_phase_change` at the run's own regression step
(`delta ≈ 1e-3` cells), energy solve OFF, area mode 6, quadratic fit, plane-anchored Dirichlet.
`geom` ∈ {exact sphere, flat interface}; `prof` ∈ {Scriven's similarity profile at the matching
time, an exactly LINEAR profile with the same interfacial slope}. The flat rows carry the profile as
a function of the signed distance, so sphere and plane have IDENTICAL `T'`, `T''` at the interface
and differ only in the interface's curvature. `plane × linear` is the exactness control.

Area-weighted `mdot` against the analytic `mdot = rho_v beta sqrt(alpha/t)`, **Ja = 0.5**:

| R | BL (cells) | plane × linear | **sphere × linear** | order in h/R | plane × scriven | **sphere × scriven** |
|---|---|---|---|---|---|---|
| 6 | 2.44 | +0.680 % | **+19.160 %** | — | −2.385 % | **+8.002 %** |
| 10 | 4.06 | +0.680 % | **+12.062 %** | 0.906 | −0.462 % | **+7.820 %** |
| 14 | 5.69 | +0.680 % | **+8.822 %** | 0.930 | +0.083 % | **+6.647 %** |
| 20 | 8.12 | +0.680 % | **+6.231 %** | 0.975 | +0.376 % | **+5.160 %** |

and **Ja = 2** (whose thermal boundary layer is 1.1 → 3.6 cells at 128³):

| R | BL | plane × linear | sphere × linear | plane × scriven | **sphere × scriven** |
|---|---|---|---|---|---|
| 6 | 1.08 | +0.680 % | +19.160 % | −1.974 % | **−1.661 %** |
| 10 | 1.80 | +0.680 % | +12.062 % | +3.024 % | **+8.615 %** |
| 14 | 2.52 | +0.680 % | +8.822 % | +2.921 % | **+9.051 %** |
| 20 | 3.60 | +0.680 % | +6.231 % | +2.205 % | **+7.297 %** |

Read the `sphere × linear` column against its order: **the estimator carries a clean FIRST-ORDER
`h/R` bias of +6 % at R = 20 and +19 % at R = 6, and it is positive.** The mechanism is geometric
and needs no numerics: the fit models `T` as a function of the distance to the interfacial cell's
tangent PLANE, and a sample at lateral offset `rho` sits `rho^2/(2R)` further from a sphere than
from its tangent plane, so every off-axis sample is hotter than the model expects and the fitted
`dT/dn` comes out high. The `plane × scriven` column is the profile's own contribution (the 5³
stencil straddles a boundary layer 1–8 cells thick) and it is the one that CONVERGES.

`plane × linear` reads `+0.680 %` at every radius and with zero spread over 16384 cells: that is
the probe's own sub-sampling quantum, not the solver. With `sub = 16` a flat interface at
`shift = 0.37` is represented as `C = 0.625`, i.e. a plane at `0.375`, and the analytic profile is
anchored at `0.370` — a 0.005-cell offset against a ~1.1-cell nearest sample. It is a constant
that all four columns carry and it cancels out of every comparison; the `--carry-probe` scene
places the plane on the sub-sampling grid and is exact.

**The same probe reads the ENERGY side.** `-q_gfm`, the heat the plane-anchored rows draw on the
same fields, against the exact `mdot_ex h_lv A_exact` (Ja = 0.5):

| R | plane × linear | sphere × linear | plane × scriven | sphere × scriven | `-q_gfm/E_lat` (sphere × scriven) |
|---|---|---|---|---|---|
| 6 | +0.534 % | +28.094 % | **−17.009 %** | +6.507 % | 0.98566 |
| 10 | +0.534 % | +18.036 % | −10.363 % | +6.048 % | 0.98325 |
| 14 | +0.534 % | +15.129 % | −7.361 % | +6.575 % | 0.99907 |
| 20 | +0.534 % | +12.406 % | **−5.050 %** | +6.440 % | 1.01209 |

Two facts nobody had measured. **The GFM row's flux is FIRST ORDER and under-draws by O(h/delta_T)**
— `−17 %` at a 2.4-cell boundary layer, `−5 %` at 8.1 cells, on a FLAT interface where the fit is
right to 0.4 % — because it is a two-point difference `k open (T_i - T_Gamma)/theta` over
`theta ≈ 1.4` cells. And **it carries the interface-curvature bias too, about twice as large as the
fit's** (`+12.4 %` against `+6.2 %` at R = 20 on `sphere × linear`). The two cancel on
`sphere × scriven`, which is why `-q_gfm/E_lat` sits within 1.5 % of 1 there — an accident of the
scene, not a property of the scheme.

### Instrument (a) — the energy budget of the energy solve

Interfacial cells are Dirichlet rows, i.e. they are OUTSIDE the energy solve, so the set over which
the energy equation conserves enthalpy changes membership every step. The instrument books, per
solve: the enthalpy of the unmasked / pure-liquid / masked sets; what the identity rows INJECT when
they overwrite the transported temperature (`d_overwrite`, and its part on cells that were not
masked last step); the enthalpy the class changes move between the two books (`e_enter`/`e_leave`);
`q_gfm`; and the class-change census. The discrete balance closes:
`sum rho c_p (T^{n+1} - T*)/dt` equals `q_gfm` to **0.35 %** at step 80 (the difference is the
domain-boundary flux plus the RB-GS residual), which is the check that the instrument measures what
it claims.

Scriven 128³, mode 6, MUSCL, similarity start, `R 6 → 20`, `E_lat = sum mdot A_Gamma h_lv`:

| step | Ja 0.5 `E_lat` (W) | `-q_gfm/E_lat` | `d_overwrite` (% of `E_lat`) | `e_enter` (J) | L→I | I→G | masked |
|---|---|---|---|---|---|---|---|
| 1 | 2.032e2 | 0.98451 | +7.52 % | 0 | 0 | 0 | 658 |
| 10 | 2.271e2 | 1.04424 | −0.51 % | 46.0 | 232 | 144 | 1056 |
| 20 | 2.738e2 | 1.01569 | −2.19 % | 22.9 | 152 | 96 | 1456 |
| 30 | 3.220e2 | 1.01301 | −2.02 % | 22.5 | 168 | 120 | 2048 |
| 40 | 3.764e2 | 1.02279 | −0.94 % | 53.1 | 472 | 290 | 2856 |
| 50 | 4.237e2 | 1.02778 | −1.29 % | 35.8 | 376 | 368 | 3684 |
| 60 | 4.760e2 | 1.03573 | −0.78 % | 61.5 | 686 | 456 | 4798 |
| 70 | 5.259e2 | 1.03851 | −0.89 % | 57.3 | 696 | 618 | 5830 |
| 80 | 5.773e2 | **1.03903** | **−0.74 %** | 62.0 | 862 | 652 | 7228 |

| step | Ja 2 `E_lat` (W) | `-q_gfm/E_lat` | `d_overwrite` (% of `E_lat`) | `e_enter` (J) | L→I | I→G | masked |
|---|---|---|---|---|---|---|---|
| 1 | 4.197e2 | 0.94747 | −19.98 % | 0 | 0 | 0 | 680 |
| 10 | 4.719e2 | 0.96127 | **−29.45 %** | 43.8 | 104 | 24 | 1008 |
| 20 | 5.834e2 | 0.95874 | −12.81 % | 71.6 | 192 | 144 | 1448 |
| 30 | 7.047e2 | 0.96589 | −8.52 % | 84.4 | 296 | 200 | 2048 |
| 40 | 8.311e2 | 0.95809 | −8.39 % | 92.1 | 396 | 320 | 2728 |
| 50 | 9.492e2 | 0.98724 | −3.69 % | 130.2 | 584 | 416 | 3728 |
| 60 | 1.069e3 | 0.98806 | −3.91 % | 146.3 | 730 | 612 | 4606 |
| 70 | 1.184e3 | 0.99834 | −5.21 % | 147.0 | 834 | 756 | 5784 |
| 80 | 1.296e3 | 1.00858 | **−4.30 %** | 164.9 | 1060 | 938 | 7064 |

`e_leave` is **0.000e+00** on every row of both runs, and that is right: the cells that leave the
masked set leave as VAPOUR at `T_sat`, so they carry no superheat. `I→L` is 0 throughout — a growing
bubble only sweeps liquid → interfacial → vapour.

**The correction to the work order's hypothesis (a), and it matters.** `e_enter` is large and
one-signed and scales exactly as the work order predicted (with the cells swept per step) — and it
is NOT an energy sink. A cell that changes class does not move; its temperature is still in the
field, and the geometric `rho c_p T` transport carries it out of the cell conservatively as the
colour drains. Booking `e_enter` as "energy that disappears" would have been the sixth gate in this
campaign to measure the wrong quantity. What DOES disappear is (i) `d_overwrite`, the identity rows
replacing the transported temperature by `pcCarriedValue` — one-signed, and 6× larger at Ja 2 than
at Ja 0.5, which is the Jakob number's definition — and (ii) the `q_gfm` / `E_lat` mismatch. Their
sum at step 80 is **4.6 %** of the latent heat at Ja 0.5 and **5.2 %** at Ja 2: the same size as
the estimator's positive bias, and of the opposite sign.

### Repair 1 — conserve the overwrite (`set_phase_change_carry_conserve`), and its a-priori gate

Before the overwrite, `rho c_p(C_j) (T_j - dval_j)` is handed to cell `j`'s face neighbours that are
still IN the solve, weighted by `n_d^2` (the clip-and-redistribute allocation) as a fixed-order
GATHER — each receiving cell recomputes its donors' decision — so it is decomposition-independent
with no reverse-add halo. Which SIDE receives is decided per axis and locally: of the two neighbours
along that axis, the one whose deviation from `T_Gamma` has the same sign as the interfacial cell's
own, i.e. the phase the enthalpy came from (the superheated liquid on an evaporating bubble, the
superheated vapour on the Stefan problem) with no scene-specific rule.

**A-priori gate** (`--carry-probe`): 64³, a PLANAR interface on the sub-sampling grid, an exactly
LINEAR superheat, `h_lv` chosen so the fitted `mdot` is exactly 2e-3 — the one configuration in
which the one-sided fit is exact, so nothing but the book-keeping is under test.

| step | `d_overwrite` (J) | deposited | lost | **identity residual** | `dH` OFF | `dH` ON |
|---|---|---|---|---|---|---|
| 2 | −0.240431 | +0.224108 | 1.97e-10 | **0.000e+00** | −2.20404 | −2.20404 |
| 3 | −0.246880 | +0.238940 | 1.89e-10 | **0.000e+00** | −2.26648 | −2.26648 |
| 4 | −0.250161 | +0.245062 | 1.74e-10 | **0.000e+00** | −2.34112 | −2.34112 |
| 5 | −0.251748 | +0.248771 | 2.16e-10 | **0.000e+00** | −2.42572 | −2.42572 |
| 6 | −0.252552 | +0.250918 | 3.27e-10 | **0.000e+00** | −2.51887 | −2.51887 |

(The `d_overwrite` column is the OFF run's; with the option on the two states diverge, so the
identity is checked against the ON run's own overwrite.) `deposited + lost + d_overwrite` is
**0.000e+00 bitwise at every step**, and `lost` — the interfacial cells whose whole `n_d^2`
allocation lands on cells that are themselves identity rows — is **4e-10 of the deposit** on a
planar interface. **The gate passes.** On the CURVED Scriven interface `lost` is **18 … 25 %** of
the total, because the interfacial band there is 1.5 cells per `h^2` thick and an interfacial cell's
liquid-side neighbour is often interfacial itself; the identity residual is still exactly
`0.000e+00`, i.e. the option is exactly conservative over what it can place and reports the rest.

### Repair 2 — the curvature-corrected fit distance (`set_phase_change_fit_curvature`)

`vof::pcCurvedDistance(phi, d, n, kappa) = phi + kappa |d_perp|^2 / 4` with `kappa = div(n)`
(`-2/R` for a gas sphere). The curvature is PRESCRIBED, from the known geometry — an instrument, not
an estimator; building an estimator is only worth doing if the coupled gate ever wants it.

**A-priori gate** (128³, `sub = 16`, exact sphere), `mdot_area` relative bias, before → after:

| R | `sphere × linear` before | after | order in h/R (after) | `sphere × scriven` before | after |
|---|---|---|---|---|---|
| 6 | +19.160 % | **−3.396 %** | — | +8.002 % | −7.882 % |
| 10 | +12.062 % | **−1.834 %** | 1.206 | +7.820 % | −3.427 % |
| 14 | +8.822 % | **−1.193 %** | 1.276 | +6.647 % | −1.978 % |
| 20 | +6.231 % | **−0.744 %** | 1.325 | +5.160 % | −1.124 % |

**The gate passes**: an order-1 bias becomes an order-1.3 one an order of magnitude smaller, and the
residual `sphere × scriven` error is the profile's own (which the `plane × scriven` column shows).
`q_gfm` is unchanged to four digits, as it must be — the correction is in the fit, not in the GFM
row's `theta`, and that asymmetry is why `-q_gfm/E_lat` moves to **1.076 … 1.155** with it on.

### Gate (c) — confinement and mesh, re-taken on mode 6

Ja = 0.5, ratio 100, MUSCL, similarity start, area mode 6, `sub = 4`:

| grid | R (cells) | `L/R_end` | **max \|dR\|/R** | **beta_eff/beta − 1** | area-avg `mdot`, last half | band_div | fallback | iters |
|---|---|---|---|---|---|---|---|---|
| 128³ | 6 → 20 | 6.4 | **1.036 %** | **−1.655 %** | −0.926 % | 6.0e-12 | 0 | 30/600 |
| 192³ | 6 → 20 | 9.6 | **1.034 %** | **−1.652 %** | −0.921 % | 8.1e-12 | 0 | 39/600 |
| 128³ | 4 → 13.33 | 9.6 | 0.669 % | −0.810 % | +1.306 % | 5.9e-12 | 0 | 27/600 |
| 96³ | 4.5 → 15 | 6.4 | 0.805 % | −1.073 % | +0.623 % | 9.1e-13 | 0 | 28/600 |
| 192³ | 9 → 30 | 6.4 | 1.825 % | −2.557 % | −2.506 % | 1.6e-02 | 24 | 44/600 |

**Confinement is excluded to three digits**: the same bubble in cells at 1.5× the clearance reads
−1.652 % against −1.655 %. **The mesh ladder is ANTI-CONVERGENT**: at fixed `R/L` and fixed physics,
96³ → 128³ → 192³ gives −1.073 → −1.655 → −2.557 %, i.e. the error grows like `(R/h)^1.1` (the
consecutive ratios are 1.54 and 1.545 against grid ratios 1.333 and 1.5). At FIXED `h` the same
scaling holds: `R_end = 13.3` reads −0.810 % and `R_end = 20` reads −1.655 %. That is exactly what a
cancellation predicts — the compensating term is the `O(h/R)` curvature bias, and refining removes
it — and it closes the question WO-P23 item 1 and WO-P3b left open ("grid refinement does not close
it"): refinement CANNOT close it, it opens it.

(The 192³ `R 9 → 30` row carries `band_div 1.6e-02` and `fallback 24`, i.e. that scene's deposit
walk fails on 24 interfacial cells; it is the only row of the ladder that is not at the deposit
floor and its number should be read with that caveat. The two clean 192³/128³ rows carry the
confinement statement.)

### The gate — Scriven, 128³, ratio 100, similarity start, MUSCL, `R 6 → 20`, area mode 6

| configuration | Ja 0.5 **max \|dR\|/R** | **beta_eff/beta − 1** | Ja 2 **max \|dR\|/R** | **beta_eff/beta − 1** |
|---|---|---|---|---|
| **shipped** (both options off) | **1.036 %** | **−1.655 %** | **1.486 %** | **−1.475 %** |
| + `carry_conserve` | **0.486 %** | −1.132 % | **3.183 %** | **+2.975 %** |
| + `fit_curvature` | 6.977 % | −7.435 % | 7.903 % | −8.295 % |
| + both | 6.264 % | −6.757 % | 3.058 % | −3.249 % |

Every row reproduces the shipped baseline to the digit (1.036 / −1.655 and 1.486 / −1.475, WO-P3e),
so the harness is faithful. `A_end/(4 pi R^2)` is +0.04 … +0.05 % on every row, `band_div` ≤ 2.5e-11
except the `both` Ja 2 row (1.1e-01, fallback 6), `unresolved` 0 everywhere, no capped solve.

**The gate is missed on every row and the two repairs pull in opposite directions.** `carry_conserve`
alone would pass the `max|dR|/R` half at Ja 0.5 (0.486 %) and fails `beta_eff` (−1.132 %); it is
2.1× WORSE at Ja 2, with the sign of the error flipped, which is the direct measurement that Ja 2's
whole deficit WAS this leak cancelling the estimator's bias. `fit_curvature` — the repair with the
cleanest a-priori gate in this work order — is 5–7× worse on both. **Under rule 4 the run stops
here; neither option becomes a default, and `origin/main`'s behaviour is unchanged.**

### The mechanism, stated once

Three first-order errors, all measured, two of them for the first time:

* **F1** the one-sided `mdot` fit's `O(h/R)` interface-curvature bias, **+6 % at R = 20**, positive
  (it over-predicts evaporation);
* **F2** the plane-anchored Dirichlet row's `O(h/delta_T)` two-point flux, **−5 % at a 8-cell
  boundary layer** and **−17 % at 2.4 cells** on a flat interface, negative, PLUS its own
  curvature bias of about `+2 x F1` — so on a sphere the two partly cancel and `q_gfm` lands within
  a few percent of `mdot h_lv A`;
* **F3** the enthalpy the per-cell Dirichlet overwrite destroys, **−0.7 % of the latent heat at
  Ja 0.5 and −4.3 % at Ja 2**, one-signed.

The coupled system is a feedback loop: `R` integrates the fit's `mdot`, while the thermal boundary
layer thickens at the rate the SINK drains it. To leading order `dR/R ≈ F1 − (F2 + F3)/2`, which is
`+5 % − 5 %` at Ja 0.5 — and the measurement is −1.0 %. Removing any one term breaks the balance by
its own size, which is exactly what the four-row gate table shows. **Closing P3 needs F1 and F2
repaired TOGETHER**, i.e. the curvature-corrected distance applied to the GFM row's `theta` as well
as to the fit, and the GFM row raised to second order (a three-point ghost-fluid row, or the
`mdot` taken from the discretely conservative interfacial balance rather than from a separate
least-squares fit). Both are changes to the energy OPERATOR, not to the phase-change driver, and
neither is a tuning knob; that is the rung this campaign has to build next, and this work order's
four options and two probes are the instruments to gate it with.

### Gate (d) — the planar rungs, inertness, and MPI

* **Byte-identity against `origin/main` (`22b4f62`), run:** `test_vof_phase_change` from both trees,
  4 threads, nvidia-cuda — the whole stdout is **BYTE-IDENTICAL** (`diff` empty). For the record the
  planar gates it prints: P0a `1.776e-14`, P0b `u_gas` exact / `max|div - S| 3.469e-18`,
  P1 `+1.3099 %`, P1' `−0.0139 %`, ENERGY identity `0.000e+00`, P2 `+0.1929 %`,
  INERT `0.000e+00`.
* `tests/kokkos` at the shipped defaults: **33/33 passed** (nvidia-cuda, `OMP_NUM_THREADS=4`,
  311 s).
* **MPI** (`tests/kokkos_mpi`, nvidia-cuda, `OMP_NUM_THREADS=4`): the whole VoF subset
  **40/40 passed** (`ctest -R vof_`), and the full 100-test battery ran 89 of 100 with **0
  failures** before the session's shell was killed by an unrelated agent on the shared box; the 11
  not reached are the non-VoF tail (`wall_slip_mpi_np4` onward). `vof_phase_change_mpi` np 1/2/4
  (64x4x4, the ORB cutting x so the interface crosses a rank boundary during every run), verbose:

| case | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| **P0a** 1000 kinematic steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| **P1** Stefan, 280 coupled steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| P2 sucking, 55 coupled steps (interface position) | 0.000e+00 | 7.901e-16 | 7.901e-16 |
| P2 pointwise `max\|C_dist - C_ref\|` | 1.299e-14 | 1.299e-14 | 1.331e-14 |
| P2 pointwise `max\|T_dist - T_ref\|` | 8.882e-16 | 8.882e-16 | 9.992e-16 |

  — identical to WO-P3e's table to every digit, which is the statement that this work order's
  `src/` changes are inert at their defaults on the distributed path too.

* **The OPTIONS through the same gate.** Both test files gained `PECLET_P3F_CARRY` /
  `PECLET_P3F_KAPPA` (the hook `PECLET_P3C_AREA` uses), applied to BOTH the reference and the
  distributed solver. `PECLET_P3F_CARRY=1`, np 1/2/4 — **3/3 passed**, and the enthalpy gather is
  decomposition-independent at the same floor as everything else in the rung:

| case (carry ON) | np = 1 | np = 2 | np = 4 |
|---|---|---|---|
| **P0a** 1000 kinematic steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| **P1** Stefan, 280 coupled steps | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| P2 sucking (interface position) | 1.560e-15 | 7.798e-16 | 1.560e-15 |
| P2 pointwise `max\|C_dist - C_ref\|` | 2.776e-14 | 2.776e-14 | 3.175e-14 |

  (that scene's own answer moves with the option — P1 layer −1.4167 %, P2 +1.6885 % — which is the
  measurement, not the gate; the gate is the three columns being equal). The curvature correction
  needs no such gate argued separately — it is a per-cell function of the existing 5^3 stencil with
  no reduction and no new neighbour dependency — but the hook is there to take it.

* **What the options do to the PLANAR rungs, single rank** (they are OFF by default, so this is a
  measurement, not a regression): `PECLET_P3F_CARRY=1` moves **P1' −0.0139 % -> −1.4167 %** and
  **P2 +0.1929 % -> +2.7033 %** (both outside those scenes' own tolerances, so the binary reports 2
  failed CHECKs); `PECLET_P3F_KAPPA=0.05` — a deliberately FICTITIOUS curvature on a flat interface,
  i.e. a decomposition/inertness probe rather than physics — moves P1 +1.3099 -> +0.5693 %,
  P1' −0.0139 -> −0.9704 % and P2 +0.1929 -> +2.6686 %. K1…K6, P0a, P0b, ENERGY and INERT are
  unmoved by both. **So the overwrite leak is compensating on the PLANAR rungs too**: P1' and P2 are
  as accurate as they are partly because of it, which is the same statement the Scriven table makes
  and it is why neither option can become a default on anything measured here.

### Open, and the corrected gates this WO proposes

1. **P3 remains NOT closed** at 1.036 % (Ja 0.5) / 1.486 % (Ja 2), mode 6, 128³ — the SIXTH failure,
   stopped under rule 4. Ja = 10 not run.
2. **A class change is not an energy transfer.** `e_enter` / `e_leave` are a flux between the masked
   and unmasked BOOKS; the cell does not move. Quote `d_overwrite` (and the `q_gfm` / `E_lat`
   mismatch) when asking where enthalpy goes, never the class-change columns.
3. **`-q_gfm/E_lat` should be a standing gate on every phase-change rung.** A scheme whose energy
   sink and whose mass source are two different discretizations of the same flux is non-conservative
   by construction, and the planar rungs P0/P1/P2 cannot see it (there both are accurate). It is one
   line of the budget; put it in the P-rung gate list.
4. **The GFM row is FIRST ORDER** and no work order had measured it. Its `−17 %` at a 2.4-cell
   boundary layer is the reason Ja = 2 (BL 1.1 → 3.6 cells at 128³) is the harder case, and it caps
   any repair to the fit alone. Repairing it is the next rung's item 1.
5. **Refinement makes P3 worse, measured** (−1.073 → −1.655 → −2.557 % at 96³/128³/192³, fixed
   `R/L`). WO-P23 item 1 and WO-P3b's ladder should be read with this: the rung is not
   under-resolved, it is cancelling.
6. **The 192³ `R 9 → 30` deposit fails on 24 cells** (`band_div` 1.6e-02) and the `fit_curvature +
   carry` Ja 2 run on 6 (`band_div` 1.1e-01). `PECLET_PC_DEPOSIT_FALLBACK=1` exists for exactly
   this and is still OFF by default (WO-P23 mechanism 5); a rung that runs bubbles at `R/h ≥ 30`
   has to settle it.
7. **`set_phase_change_fit_curvature` takes a PRESCRIBED curvature.** If a future rung wants it as
   a default it needs an estimator; the V3 cascade already computes a low-noise mean curvature for
   surface tension and is the obvious source, but nothing here measured that path.

---

## WO-V6b findings — the velocity half of the dynamic contact line (Navier slip in the cut-cell wall closure), plus the integer-coordinate wall defect — 2026-09-03, Opus

Branch `vof-v6b`, worktree `../flow-v6b`, from `origin/main` at `9ad0646`. Commits: `fa1e346`
(part A's tie-break + the Navier closure + the bindings), `ef6c58e` (the gate scripts + the MPI
ctest), plus this entry. Backend **host-openmp** throughout (`build_omp`, `OMP_NUM_THREADS=4`,
`OMP_PROC_BIND=false`); the machine was shared with three other sessions, so no number here is a
timing.

Everything ran with the velocity-solve session's new defaults in force (momentum residual stop at
the pressure driver's rtol, velocity-MG AUTO, telescoping ON) — they are on `origin/main` at the
branch point.

---

### PART A — a velocity DOF with `sdf` EXACTLY 0 was classified as fluid by ONE consumer and as solid by every other

**The survey.** Every classification of the same SDF field in the cut-cell/IBM/projection stack,
and what each one does at exact zero (worktree line numbers):

| site | predicate | "fluid" means | a point with `sdf == 0` is |
|---|---|---|---|
| `mac_ibm.hpp:31` `ibmIsCut` (centre) | `sc <= 0` ⇒ NOT a cut cell | `sdf > 0` | **non-fluid** — no wall closure is built |
| `mac_ibm.hpp:34` `ibmIsCut` (neighbour) | `sn[k] < 0` ⇒ solid neighbour | `sdf >= 0` | fluid |
| `cut_cell_ibm.hpp:134` `ibmFillEntry` | `sdf_n[k] < 0` ⇒ ghost/solid | `sdf >= 0` | fluid |
| `mac_ibm.hpp:195` `ibmCleanFluidMask` | `sc <= 0` ⇒ solid | `sdf > 0` | **non-fluid** |
| `mac_ibm.hpp:177` `ibmSolidMask` | **was** `sd < 0` ⇒ pinned | `sdf >= 0` | **fluid — the odd one out** |
| `mac_cutcell.hpp:79` `ccFractionCore` | `sd <= 0` ⇒ area fraction 0 | `sdf > 0` | **closed** |
| `mac_cutcell.hpp:124` `ccFaceOpen` | `sd <= 0` ⇒ face closed | `sdf > 0` | **closed** |
| `mac_cutcell.hpp:138` `ccTriFrac` | `a >= 0` ⇒ fluid vertex | `sdf >= 0` | fluid |
| `mac_approx_projection.hpp:98/104` ghost-projection donors | `sdf >= 0` | `sdf >= 0` | fluid |
| `mac_approx_projection.hpp:174/292` sub-sample | `> 0.0` | `sdf > 0` | solid |
| `mac_approx_projection.hpp:278/283` `buildCellFraction` centre | `sc >= 0` | `sdf >= 0` | fluid |
| `mac_approx_projection.hpp:423` | `sdf >= 0` | `sdf >= 0` | fluid |
| `star_elimination.hpp:81/89` star overlay | `sdf >= 0` ⇒ not eliminated | `sdf >= 0` | fluid |

The file owner's lead was right in substance and inverted in direction: `star_elimination.hpp`'s
`>= 0` is one of NINE cell-centred sites that agree with each other, and the cell-centred sites are
not where the damage is — a cell CENTRE lands on `sdf == 0` only for a wall on a cell-centre plane.
The single genuine disagreement is at the **velocity DOF**: `ibmSolidMask` (the mask that PINS a
DOF to the wall datum, `maskVelocity`) called an exactly-on-wall DOF FLUID, while `ibmIsCut` (which
decides whether a Robust-Scaled wall closure is built for it), `ibmCleanFluidMask` and the face
openness all called it non-fluid. So such a DOF was

* **not pinned** (`ibmSolidMask` → 0), and
* **not closed** (`ibmIsCut` returns false at `sc <= 0`, so no overlay row, so no Dirichlet datum),
* while its FACE was **closed** to the projection (`ccFaceOpen` → 0, so it never enters the
  divergence and the pressure solve never sees it),
* yet it **is** read by the FOU advection operator of its fluid neighbours (`advVelView`) and by
  the diffusion stencil of the first fluid DOF.

An unconstrained velocity unknown sitting on the wall, invisible to every diagnostic. That is the
whole of WO-V7 finding 1.

**When does `sdf == 0` happen at a DOF?** Exactly when a flat wall sits on a grid plane. The
staggered sample is `ccSampleExt` at the component's `-1/2` offset, i.e. the MEAN of the two
adjacent cell-centre values, so a wall on an INTEGER coordinate (a cell face) puts the NORMAL
component's DOF exactly on it (centres at `±1/2` → mean `0.0`, exactly, in IEEE), and a wall on a
cell-CENTRE plane (a half-integer, `zw = k + 1/2`) puts the two TANGENTIAL components' DOFs exactly
on it. Both occur in the shipped tree: the WO-V7 doublet is the first, `tests/kokkos`'s
`test_vof_cutcell` G5 cap is the second.

**The fix.** `ibmSolidMask` now uses `sd <= 0`, which is the convention the other four DOF-level
consumers already use. For a DOF lying exactly ON the wall the pin is not an approximation: it IS
the Dirichlet datum (0 for a static wall), and the neighbouring fluid DOF's plain interior stencil
then becomes a wall-resolved discretization. One character plus its reason, in `src/mac_ibm.hpp`.

**The reproducer, before and after.** WO-V7's "no septum (a plain slit, same walls)" ladder row,
reproduced here as a standalone script (88×4×80, walls on INTEGER `z`, θ = 45°, Ca = 1e-2, ratio
100, surface tension + momentum consistency on, `dt` re-picked from `vof_step_limits()` every step):

| step | `t` | `dt` | max\|u\| BEFORE | max\|u\| AFTER |
|---|---|---|---|---|
| 1 | 0.1418 | 1.4e-01 | 1.2835e+02 | **1.6865e+00** |
| 10 | 0.1577 / 1.3884 | 1.5e-03 / 1.4e-01 | 1.3924e+02 | **9.5137e-01** |
| 50 | 0.1825 / 7.0584 | 2.0e-04 / 1.4e-01 | 1.0408e+03 | **7.7146e-01** |
| 150 | 0.1867 / 21.234 | 2.0e-06 / 1.4e-01 | 1.0721e+05 | **6.7951e-01** |
| 300 | 0.1867 / 42.496 | 2.0e-09 / 1.4e-01 | **1.0347e+08** | **6.7767e-01** |

Before: the geometric divergence, with `t` frozen at 0.1867 s and `dt` chased down nine orders
while the pressure solve reported a healthy 21/400 iterations at every step. After: `dt` never
leaves the capillary limit 0.1418, `t` reaches 42.5 s in the same 300 steps, `max|u|` DECAYS to
0.68 and `max|div(open u)|_projected` stays at 1e-10. The exact-zero DOF census for that scene is
**256 u-DOFs** (the slabs' vertical faces at integer `x`) and **384 w-DOFs** (the horizontal faces
at integer `z`).

**Independent confirmation from a SHIPPED test.** `tests/kokkos/test_vof_cutcell` is the one
binary of 33 whose output changes, and it changes because its G5 scene ("a liquid cap on a flat SDF
wall at a HALF-INTEGER `z`", `zw = 3.5`, so `sdf` at the cell centres is `…,-1,0,+1,…` and the u/v
DOFs at `z = 3` sit exactly on the wall) was carrying the defect all along. Same binary, same
scene, only the tie-break:

| G5 metric | before | after |
|---|---|---|
| max\|u\| including the wall band | 7.879e-01 | **5.053e-03** (156× smaller) |
| max\|u\| over the open fluid | 9.821e-03 | **5.053e-03** |
| trace of max\|u\| over the run | 5.26e-01 → 7.88e-01 (growing) | 2.87e-02 → 5.05e-03 (decaying) |
| wall-band max\|kappa\| | 0.2222 | 0.1722 |
| apparent θ (target 90°) | 89.935 | 90.068 |
| G8b packing draining, max\|u\| | 2.151e-02 | **7.119e-06** |
| G8b `max\|div(open u)\|` | 4.497e-17 | **2.534e-18** |
| colour drift / volume drift | 5.95e-16 | **0.000e+00** |

The test still prints "all vof cut-cell checks passed" and exits 0. Its own comment — "so the cells
straddling it are genuinely cut (eps = 1/2) — a wall on a cell face would be a degenerate case" — is
right about the CELL fraction and wrong about the velocity DOF, which is the case it was trying to
avoid.

**Verdict: FIXED, not escalated.** It is a one-convention tie-break, the intent is unambiguous (four
of the five DOF-level consumers already say `<= 0`), the change is inert wherever no DOF sample is
exactly zero, and the only shipped test it touches gets strictly better on every metric it prints.
The escalation clause did not trigger: there are not two consumers disagreeing about semantics,
there is one consumer disagreeing with four.

**Two things it does NOT fix, and one it costs.**
1. A wall on a cell-centre plane pins the TANGENTIAL DOFs that lie on it, so no wall model —
   including the Navier slip below — can act there: the slip velocity at such a wall is
   structurally 0. **Quarter-integer placement (WO-S finding 5, WO-V7 finding 1) remains the
   scene rule**, and it is now also the rule that makes part B expressible.
2. The driver-side divergence guard WO-V7 asked for (a run whose `dt` has fallen orders below its
   initial capillary limit is diverging) is still missing. This fix removes the cause it found, not
   the class.

---

### PART B — WO-V6b: the Navier condition in the Robust-Scaled cut-cell closure

**What shipped.** `set_wall_slip_length(lambda_cells)` (+ `wall_slip_length()`,
`wall_slip_sandwich_cells()`). The tangential wall Dirichlet datum of the RS closure becomes
`u_t(wall) - u_body = lambda du_t/dn`. Files: `src/cut_cell_ibm.hpp` (three new polynomials + two
new optional arguments to `ibmFillEntry`), `src/mac_ibm.hpp` (the per-cut-cell normal and the
per-axis slip length in `buildIbmOverlay`, plus part A's tie-break), `src/flow_ibm.hpp`
(`buildVelocityOverlays` extracted verbatim from `setSolidDevice` so a closure change can rebuild
without re-running the geometry setup, `setWallSlipLength`, the shared lambda), the bindings, the
study `tests/study/wall_slip.py`, the MPI ctest `tests/kokkos_mpi/test_wall_slip_mpi.cpp`, and the
`wall_slip` / `slipsweep_v6b` / `lw` options in `tests/study/vof_wetting_dynamic.py` and the
`--slip` option in `tests/study/pore_scale/pore_doublet.py`.

**The design is NOT a modified datum — it is a one-parameter family of the SAME polynomials, and
that is what makes lambda = 0 free.** The shipped closure fits the quadratic `p` through
(`u_m` at −1, `u_c` at 0, `u_g` at +1) and imposes `p(theta) = u_b` at the wall crossing; the Robin
condition along the axis (solid on the +x side, so the inward fluid normal is −x) is
`p(theta) + lam p'(theta) = u_b`. Solving for the ghost with `P = theta + lam`,
`Q = theta^2 + 2 lam theta` gives `(P+Q) u_g = 2 u_b - 2 u_c (1-Q) + u_m (P-Q)`, i.e.

```
D   = theta(1+theta) + lam(1 + 2 theta)        (poly_D   + lam(1+2theta))
X   = theta(1-theta) + lam(1 - 2 theta)        (poly_N_nb + lam(1-2theta))
K   = 2(theta^2 - 1) + 4 lam theta             (poly_Nc  + 4 lam theta)
Nbc = 2                                        (UNCHANGED)
```

so `lam = 0` is literally the shipped polynomial and `lam -> infinity` is the free-slip
(zero-normal-derivative) closure — checked by hand at `theta = 1/2`, where it gives `u_g = u_c`,
which is what `p'(1/2) = 0` requires. Consequences worth recording: the stencil STRUCTURE is
untouched (same 7-point row, same overlay layout, same `ibmModifyStencil`, which was not edited at
all), so the change merges with anything the velocity-solve session does to the operator assembly;
the closure stays fully IMPLICIT (no lagged slip term); and `D` is now bounded below by `lam`, so
the row scaling `D_rescale` is BETTER conditioned at small `theta`, not worse.

**Per-DOF geometry.** At each cut cell the unit normal comes from the central difference of the
same seven SDF samples the closure already gathers (`grad sdf` points into the fluid). The
component's tangential weight is `s = 1 - n_c^2` — 0 for the component parallel to the wall normal,
which therefore keeps a pure impermeability Dirichlet, 1 for a purely tangential one — and the slip
length measured along axis `a` is `lam_a = s lambda / |n_a|`, because the wall distance along that
axis is the normal distance divided by `|n_a|`. That division is not a fudge: the closure depends
only on `lam_a/theta_a = s lambda / d` with `d` the NORMAL wall distance, so `|n_a|` cancels.
`|n_a|` is floored at 1e-3 (a wall nearly parallel to the axis).

**Two approximations, both stated because they are invisible from the call.**
1. *The tangential projector is taken DIAGONAL*: `u_t,c ~ (1 - n_c^2) u_c`, the cross terms
   `-n_c n_j u_j (j != c)` are dropped. They are EXACTLY zero for an axis-aligned wall — every gate
   scene here, the doublet's slabs, the Jurin plates, the spreading drop's floor — and O(n_c n_j)
   on a curved solid (a packing). Carrying them would couple the three segregated component solves,
   which is a different piece of work.
2. *A SANDWICHED axis (both neighbours solid — a one-cell fluid gap) keeps the no-slip closure.* A
   slip length is a sub-cell wall model and a gap a single cell spans does not resolve one. It is
   COUNTED, not silent: `wall_slip_sandwich_cells()` returns the per-component count, and it is
   `(0, 0, 0)` on every scene in this entry.

**One lambda, two switches.** `set_wall_slip_length` and `set_contact_angle_dynamic` write the same
stored value (last call wins), so the Cox–Voinov inner cut-off and the momentum closure can never
disagree — that is the unification the work order asked for. Only `set_wall_slip_length` switches
the MOMENTUM half on. **This is a deviation from the literal reading and it is deliberate**: making
`set_contact_angle_dynamic` turn the momentum closure on as well would have changed every number
`tests/kokkos/test_vof_wetting_dynamic` and `test_vof_wetting_dynamic_mpi` record (they configure
`lambda = 0.1`), i.e. it would have re-baselined a validated ctest, which rule 1 forbids. The two
switches keep WO-V6's angle-half results exactly what they were and make the velocity half an
explicit opt-in.

Order-independent: `set_wall_slip_length` before or after `set_solid` gives a **bitwise identical**
field (measured, 0.0). That is worth stating because the domain BCs are NOT (see the defects below).

#### The six gates at a glance

| gate | verdict | the number |
|---|---|---|
| 1 analytic slip profile | **PASS at the float storage floor** (the work order's 1e-10 is unreachable — the `lambda = 0` no-slip parabola already misses by 1.15e-06) | slip increment 1.2000033e-03 vs the exact 1.2e-03, **+2.8e-06 relative**; host-openmp and nvidia-cuda digit for digit |
| 2 `lambda = 0` bit-identical | **PASS** | 32 of 33 `tests/kokkos` binaries byte-identical to `origin/main`; the 33rd is part A's fix and improves every metric it prints; the whole single-phase regression (Z&H drag, both permeabilities, iteration and step counts, divergence) identical to the last digit |
| 3 Cox–Voinov slip sensitivity | **PASS** on the swept range (per-interval reported, not uniform) | `d(slope)/d ln(1/lambda)` = **6.98** vs the model's 9 (**−22.4 %**, gate 25 %), against WO-V6's 2.34 (−74 %) |
| 4 Lucas–Washburn | **FAIL** — first attempt, four configurations, nothing tuned | `d(h²)/dt` 0.1322 vs 23.09 (−99.4 %); the slip is worth **+26 %** of the rate at `lambda = 0.3`; the gap-width probe shows the limiter is band-local, not the wall closure |
| 5 pore doublet, Ca = 1e-3, theta = 45° | **FAIL** — the verdict does not flip back | wide first at `lambda` = 0, 0.1 and 0.5; the narrow branch's breakthrough moves 1.8 % over that range |
| 6 MPI np 1/2/4 | **PASS** | np = 1 **bitwise** (0.000e+00); np = 2/4 ~1e-27 against a 3.3e-13 tolerance, with BOTH walls' Robin closures straddling a rank boundary |

**One-line verdict.** The velocity half is implemented, exact where it can be checked exactly,
free when off, and distributed — and it is NOT the dominant term in this scheme's contact-line
mobility. It buys a factor 3 of Cox–Voinov sensitivity on a FREE interface and ~26 % of the speed
of a CONFINED one; the remaining factor ~175 is a resistance local to the wetting band, which gate
4's gap-width probe isolates and which is the next rung.


#### Gate 1 — the analytic slip profile

The work order's gate is a Couette profile. **It could not be driven, and that is itself a
measurement** (three obstructions, all recorded in `tests/study/wall_slip.py`):

* a type-1 domain-BC WALL ignores its tangential velocity — a plain channel (8×32×8, no solid,
  Stokes, `dt = 100`) with `set_domain_bc(2, 1, 0,0,0)` / `set_domain_bc(3, 1, U,0,0)` stays
  identically 0 after 400 steps;
* a type-2 INFLOW face given a purely tangential velocity DOES drive it (that is what
  `tests/kokkos_mpi/test_varmu_mpi.cpp` uses, and it reproduces `u(y) = U y/H` exactly) — **but not
  when an immersed solid is present**: the same scene with `set_solid(..., cutcell_pressure=True)`
  leaves the field at exactly 0 with 0 pressure iterations, at any BC ordering;
* and domain BCs set AFTER the geometry are silently ignored altogether (the all-fluid channel
  drives at 10 pressure iterations with the BCs set first and returns exactly 0 with them set last).

The last two are defects outside this WO; the second is what blocks the literal gate. So gate 1 is
taken on a **slip POISEUILLE**, which is strictly stronger: exact on a QUADRATIC (hence on any
linear profile), and it exercises the slip at TWO walls with different crossing fractions at once.
Slit of width `H` between two flat SDF walls at QUARTER-INTEGER `z`, uniform body force `G`:
`u(z) = (G/2mu)((z-z0)(z1-z) + lambda H)` satisfies the Robin condition exactly, so the closure
must reproduce it.

8×8×40, walls at `z = 8.25 / 32.25` (`H = 24`, `theta = 0.25` at the low wall and `0.75` at the
high one), `mu = 1`, `G = 1e-3`, Stokes, run to a relative change < 1e-13 per step:

| lambda (cells) | steps | max\|u − u_exact\| | relative | u at z = 20.5 |
|---|---|---|---|---|
| 0 | 94 | 8.2722e-08 | 1.1494e-06 | 7.19688327e-02 |
| 0.02 | 96 | 8.2172e-08 | 1.1380e-06 | 7.22088322e-02 |
| 0.05 | 97 | 8.4181e-08 | 1.1600e-06 | 7.25688342e-02 |
| 0.1 | 100 | 8.5989e-08 | 1.1752e-06 | 7.31688360e-02 |
| 0.3 | 111 | 9.1259e-08 | 1.2076e-06 | 7.55688413e-02 |
| 0.5 | 115 | 9.9019e-08 | 1.2700e-06 | 7.79688490e-02 |
| 1.0 | 124 | 1.1135e-07 | 1.3261e-06 | 8.39688613e-02 |

`wall_slip_sandwich_cells() = (0,0,0)`, `max|div(open u)| = 1.21e-17`, 8 pressure iterations of 400,
never capped. **`nvidia-cuda` reproduces every column of that table digit for digit** (the only
differences in the whole printout are `max|div|` 1.24e-17 against 1.21e-17 and the last digit of two
of the convergence deltas), so the closure arithmetic is backend-independent.

**The work order's 1e-10 is not reachable and the reason is not the slip.** The `lambda = 0` row —
the SHIPPED no-slip closure on the exact no-slip parabola — already misses by **1.15e-06 relative**,
and that is the FLOAT momentum-operator storage (`IbmOverlay` rows are `float`, `D_axis` and
`R = D_rescale/D_axis` are `float`). The slip adds essentially nothing to it: the error grows only
from 1.1494e-06 to 1.3261e-06 while `lambda` goes 0 → 1 cell. Read as an increment instead, the
measured slip is exact: at `lambda = 0.1` the profile rises by **1.2000033e-03** against the
analytic `G lambda H / 2mu = 1.2e-03`, i.e. **+2.8e-06 relative**. **PASS at the storage floor**;
the corrected gate is "the slip INCREMENT to the float floor", and the honest constant is 1.2e-06,
not 1e-10.

#### The float floor of `lambda` (the work order asked for it, and it is where predicted)

The closure is stored in float, so `lambda` disappears from `D = theta(1+theta) + lambda(1+2 theta)`
once `lambda(1+2 theta) < eps_f32 * D`. At the low wall (`theta = 0.25`, `D = 0.3125`) that predicts
`lambda_floor ~ eps_f32 * 0.3125 / 1.5 = 2.5e-08`. Measured on the same scene as
`max |u(lambda) - u(0)|` against the analytic increment `G lambda H / 2mu`:

| lambda | measured Δu | expected | ratio |
|---|---|---|---|
| 1e-2 | 1.200012e-04 | 1.2e-04 | 1.0000 |
| 1e-3 | 1.199963e-05 | 1.2e-05 | 1.0000 |
| 1e-4 | 1.199773e-06 | 1.2e-06 | 0.9998 |
| 1e-5 | 1.200428e-07 | 1.2e-07 | 1.0004 |
| 1e-6 | 1.199916e-08 | 1.2e-08 | 0.9999 |
| 1e-7 | 2.409558e-09 | 1.2e-09 | 2.008 |
| 1e-8 | 1.594867e-10 | 1.2e-10 | 1.329 |
| **1e-9** | **0.000000e+00** | 1.2e-11 | **0** |
| 1e-10 | 0.000000e+00 | 1.2e-12 | 0 |

So the closure is faithful to four digits down to `lambda = 1e-6` cells, quantized between 1e-7 and
1e-8, and **exactly indistinguishable from no-slip at and below 1e-9 cells** — within a factor 25
of the predicted 2.5e-08. Every physically interesting `lambda` (1e-2 … 1) is six to eight orders
above the floor, so float storage is not a constraint on this rung. The `lambda = 0` early-out is a
guarded branch, not a reliance on that floor.

#### Gate 6 — MPI np 1 / 2 / 4 (`tests/kokkos_mpi/test_wall_slip_mpi`)

The same slip Poiseuille, 8×8×40, walls moved to `z = 16.25 / 32.25` so that **both** Robin closures
straddle a rank boundary at np = 4: the ORB gives z-blocks `[0,16) [16,24) [24,32) [32,40)`, the low
wall's cut cell `k = 16` (theta = 0.25) is the first cell of block 1 with its solid neighbour
`k = 15` in block 0, and the high wall's cut cell `k = 31` (theta = 0.75) is the last cell of block 2
with its solid neighbour `k = 32` in block 3. The test asserts that z is cut and prints the blocks.
Pointwise against a full-grid single-rank reference on rank 0, 200 steps, at BOTH `lambda = 0` (the
inertness control under MPI) and `lambda = 0.1`:

| np | lambda = 0: max\|du\| | lambda = 0.1: max\|du\| | analytic rel err (0 / 0.1) |
|---|---|---|---|
| 1 | **0.000e+00** (bitwise) | **0.000e+00** (bitwise) | 5.154e-07 / 5.379e-07 |
| 2 | 5.017e-28 | 1.668e-27 | 5.154e-07 / 5.379e-07 |
| 4 | 8.463e-28 | 1.922e-27 | 5.154e-07 / 5.379e-07 |

(`|u| = 3.2e-02`, so np > 1 agrees to ~1e-26 absolute against a 3.3e-13 tolerance — well below the
reduction-order floor, and the analytic error is decomposition-INDEPENDENT to all printed digits.)
**PASS.**

#### Gate 3 — the Cox–Voinov slip sensitivity (WO-V6's corrected G2b, re-run with the momentum half ON)

`tests/study/vof_wetting_dynamic.py slipsweep_v6b`: the spreading drop (64×64×40, `D/dx = 24`, wall
at a quarter-integer `z = 4.25`, `theta_e = 30`, Oh 0.1, `sigma = 1`), 800 steps, identical 15-point
fit windows, `lambda` swept over a factor 25 — with `set_wall_slip_length(lambda)` on.

| lambda | fitted slope | 9 ln(a/lambda) | mean a | implied lambda_eff | WO-V6 slope (angle half only) |
|---|---|---|---|---|---|
| 0.02 | **46.194** | 59.416 | 14.73 | 0.0869 | 47.377 |
| 0.10 | **39.445** | 45.206 | 15.19 | 0.1897 | 43.788 |
| 0.50 | **23.712** | 31.369 | 16.32 | 1.1708 | 39.853 |

| interval | d(slope) measured | model `9 dln(1/lambda)` | this rung | WO-V6 |
|---|---|---|---|---|
| 0.02 → 0.1 | −6.749 | −14.485 | −53.4 % | −75.2 % |
| 0.1 → 0.5 | **−15.733** | −14.485 | **+8.6 %** | −72.8 % |
| **0.02 → 0.5 (the full sweep)** | **−22.482** | −28.970 | **−22.4 %** | −74.0 % |

i.e. `d(slope)/d ln(1/lambda) = 22.482/ln 25 = **6.98** against the model's **9**` — **−22.4 %,
inside the 25 % gate**, where WO-V6 measured 2.34 (−74 %). **PASS on the quantity the gate names**
(the sensitivity over the swept range), with the per-interval breakdown reported because it is not
uniform: the large-`lambda` interval is within 9 % of the model and the small-`lambda` one is still
only half of it. `lambda_eff` now moves by a factor **13.5** (0.0869 → 1.1708) while `lambda` moves
by 25 — against WO-V6's factor **2.4**, i.e. the pinning near 0.1 cells that finding 7 named is
largely gone, and what is left of it sits at the small-`lambda` end, where the prescribed slip is
below the scheme's own residual numerical slip and cannot dominate it. The absolute slopes are
reported with their window, per WO-V6's corrected gate; Tanner exponents 0.1451 / 0.1441 / 0.1207
against the law's 0.1. 10 pressure iterations, never capped.

#### Gate 2 — inertness at `lambda = 0`

Every `tests/kokkos` binary built at this branch, run against the SAME binary built from
`origin/main` = `9ad0646` in a second worktree, full stdout `diff`, host-openmp,
`OMP_NUM_THREADS=8`:

**32 of 33 identical, 1 differing, every exit code 0.** The one that differs is
`test_vof_cutcell`, and it differs because of PART A, not because of the slip: its G5 scene puts a
wall on a cell-centre plane, so its tangential velocity DOFs sit exactly on the wall. The table in
part A is that diff; the test still prints "all vof cut-cell checks passed". Nothing that does not
contain an exactly-zero DOF sample moved by a bit — including all seven other VoF binaries,
`test_vof_wetting_dynamic` (which configures `lambda = 0.1` for the ANGLE half, and is unchanged
precisely because the momentum half is a separate switch), `test_ibm`, `test_ibm_overlay`,
`test_ibm_apply`, `test_cutcell`, `test_poiseuille_ibm`, `test_mg` and `test_sdflow_tg`.

Structurally so: `ibmModifyStencil` was not edited at all, the `lambda > 0` polynomial branch is
guarded by `lamAxis != nullptr` which `buildIbmOverlay` passes only when
`slipLambda > 0 && comp >= 0 && SCHEME == 0`, and `wallSlip_` is false unless
`set_wall_slip_length` is called.

#### Gate 4 — Lucas–Washburn on the repaired WO-V6 plate scene — **FAIL, and the failure names the NEXT limiter**

`tests/study/vof_wetting_dynamic.py lw`: the WO-V6 G3 capillary-rise scene (96×4×112, gap `w = 16`,
outer channel 64, capsule plates at quarter-integer faces, `theta_e = 30`, `sigma = 1`,
`mu_l = 0.2`, `drho g = 3e-3`, ratio 10, zero-mean buoyancy), FLAT start, 1200 steps to `t = 173`,
the trace probed every 20 steps. The early-time law `h^2 = (sigma w cos(theta)/(3 mu)) t` gives
`d(h^2)/dt = 23.09`; the fit window is `2 <= h <= 0.4 * Jurin` (so the hydrostatic term is at most
40 % of the capillary drive) — 25–30 points per run.

| configuration | d(h²)/dt | vs 23.09 | final h at t = 173 | mean apparent theta |
|---|---|---|---|---|
| static angle, no momentum slip (control) | 0.1263 | −99.45 % | 3.772 | — |
| dynamic angle, `lambda = 0.05`, momentum slip OFF | 0.08233 | −99.64 % | 3.143 | 75.60 → n/a |
| dynamic angle, `lambda = 0.30`, momentum slip OFF | 0.1049 | −99.55 % | 3.465 | 73.64 |
| dynamic angle, `lambda = 0.05`, **momentum slip ON** | 0.08717 | −99.62 % | 3.209 | 74.29 |
| dynamic angle, `lambda = 0.30`, **momentum slip ON** | **0.1322** | −99.43 % | **3.858** | **70.76** |

All runs asymptotic (0.0 % overshoot), `dV/V <= 2.8e-14`, 12 pressure iterations of 300, none
capped, no non-finite-z notice on any stdout.

**The momentum Navier slip helps, and by nothing like enough**: at `lambda = 0.3` it lifts the
Lucas–Washburn rate by **+26.0 %** (0.1049 → 0.1322) and the height reached in a fixed time by
**+11.3 %** (3.465 → 3.858); at `lambda = 0.05` by +5.9 % and +2.1 %. Monotone in `lambda` and in
the right direction — but the rate is still **175× below Lucas–Washburn**, against WO-V6's 180×.
Gate 4 FAILS on its first attempt in all four configurations, and per rule 4 nothing was tuned.

**The mechanism, from the numbers already taken.** The census says the imposed angle is 35–39°
(`theta_e = 30` plus the Cox–Voinov correction) while the interface's own **apparent** angle stays
at **70.8–75.6°**. The interface is not adopting the angle it is being given, so only
`cos(71°)/cos(35°) ~ 0.40` of the intended Young force is ever delivered — and adding wall slip
moves the apparent angle by only 75.6 → 70.8, i.e. it lets the near-wall fluid MOVE without making
the interface BEND. That is a different limiter from the one V6b addresses, and it is consistent
with gate 3, where the same closure produced a 3× improvement: the spreading drop's interface is
free to deform over a 15-cell contact radius, whereas the meniscus in a 16-cell slot is a 3-cell
wetting band at each wall plus ten flat cells in the middle, so its curvature — the thing that
delivers the capillary pressure — is set by the band's reach, not by the wall's velocity condition.

**The gap-width probe settles what the resistance is, and it is not the slot.** The same scene with
the gap DOUBLED (`w = 32`, `w_out = 48`) and `drho g` retuned to `6.667e-4` so the Jurin equilibrium
is the SAME 27.06 cells — i.e. only the gap moves — over the same 1200 steps to `t = 173`:

| gap `w` | LW coefficient `sigma w cos(theta)/(3 mu)` | h reached at t = 173 (static / `lambda = 0.3` + slip) | mean apparent theta |
|---|---|---|---|
| 16 | 23.09 | 3.772 / 3.858 | 73.6–75.6 |
| 32 | 46.19 (**2× larger**) | **1.520 / 1.558** (**2.48× SMALLER**) | 70.66 |

Lucas–Washburn requires `dh/dt ∝ w`: doubling the gap must make the rise TWICE as fast. Measured, it
makes it **2.5× slower** — the front speed goes as **1/w**, which is the capillary driving
`2 sigma cos(theta)/w` divided by a resistance that **does not scale with the gap at all**. A
Poiseuille-limited rise would give `dh/dt ∝ w^2 Δp ∝ w`; a rise limited by a resistance local to the
wall band gives exactly the measured `1/w`. And the apparent-angle deficit is the same at both gaps
(70.7 vs 70.8–75.6), i.e. it is a band-local property, not a confinement one.

**So the honest verdict on the campaign question.** V6b supplies the velocity half and it is
measurably worth ~26 % of the contact-line speed at `lambda = 0.3` on a confined meniscus and a
factor ~3 of the Cox–Voinov sensitivity on a free one. It does NOT recover Lucas–Washburn, and the
remaining factor is **not** in the momentum wall condition: the rise rate is set by a resistance
LOCAL TO THE WETTING BAND (the colour's motion through the near-wall cells and the curvature that
band produces), which is independent of the gap and which neither the imposed angle nor the wall
velocity closure controls. That is the next rung, and the two numbers that define it are the `1/w`
scaling above and the 70°-vs-37° apparent-versus-imposed angle.

#### Gate 5 — WO-V7's pore doublet at Ca = 1e-3, theta = 45° — **FAIL: the verdict does NOT flip back**

`tests/study/pore_scale/pore_doublet.py --theta 45 --ca 1e-3 --slip <lambda>`, the WO-V7 scene
unchanged (88×4×80, quarter-integer walls, `sigma = 100`, ratio 100/100, inflow/outflow drive,
static angle, momentum consistency on, FCG selected last, `dt` re-picked every step). The no-slip
CONTROL was re-run on THIS build and backend rather than compared against WO-V7's numbers, because
the two differ materially (see below).

| lambda (cells) | t_bt narrow | t_bt wide | fills first | tip n/w | S_narrow | S_wide | pressure | max\|div\|_proj | steps |
|---|---|---|---|---|---|---|---|---|---|
| 0 (control) | 929.18 | **456.44** | **wide** ✘ | 47 / 48 | 0.7971 | 0.8487 | 116/400 | 1.54e-09 | 6555 |
| 0.1 | 929.18 | **456.44** | **wide** ✘ | 47 / 48 | 0.7932 | 0.8582 | 142/400 | 2.32e-09 | 6555 |
| 0.5 | **912.88** | **456.44** | **wide** ✘ | 47 / 47 | 0.7932 | 0.8193 | 130/400 | 1.94e-09 | 6440 |

All runs valid: no capped solve, no `non-finite z` notice on any stdout, colour exactly 0 in solid
cells, clipped volume 0. The breakthrough times are quantised by the 115-step sampling interval, so identical entries mean
"within one sample" — the runs DO differ in detail (the saturations, the pressure iteration counts
and `max|u|` in the gas all move; `wall_slip_length()` reads back 0.1 / 0.5 and
`wall_slip_sandwich_cells()` is `(0,0,0)` in both), they simply do not differ in the answer. The
narrow branch's breakthrough moves by **1.8 %** (929.2 → 912.9) as `lambda` goes 0 → 0.5, the wide
branch's not at all, and the wide branch wins by a factor 2.0 in every row. Two `lambda` were tried
and nothing else was; per rule 4 the gate stops here.

**So the Chatzis–Dullien verdict does not flip, and gate 4 already says why.** The doublet's
imbibition ordering is set by whether the capillary term can be DELIVERED, and gate 4's gap-width
probe shows that what limits delivery is a resistance local to the wetting band, independent of the
channel width and unaffected by the wall velocity condition. WO-V7's diagnosis — "the Ca at which
the measured ordering flips is where the imposed velocity crosses the solver's numerical slip
velocity" — is right that the contact-line mobility is the mechanism and, on this evidence, wrong
about which term of the mobility is the bottleneck: it is not the momentum wall closure.

**A separate observation, for whoever re-reads WO-V7's table.** This branch's no-slip control gives
`t_bt` wide **456.4** where WO-V7 recorded **696.2** on the same scene (narrow 929.2 vs 973.4);
the verdict is the same but the numbers are not. WO-V7 built from `2b55edb`, i.e. BEFORE the
velocity-solve session's momentum-residual-stop default landed on `main`, and ran the CUDA backend;
this control is host-openmp at `9ad0646`. Either difference is enough to explain it, and neither is
this WO's, but the WO-V7 table should not be compared digit-for-digit against anything built after
2026-09-02.

### Defects found on the way, outside this WO

1. **A type-1 domain-BC wall ignores its tangential velocity.** `set_domain_bc(face, 1, U, 0, 0)`
   on a plain channel (no solid, Stokes, 400 steps) leaves the field identically 0. There is no
   moving no-slip wall on the domain-BC path; the only way to drive a Couette is a type-2 INFLOW
   face whose velocity is purely tangential.
2. **That inflow workaround stops working as soon as an immersed solid is present.** The same
   channel with `set_solid(..., cutcell_pressure=True)` and a tangential type-2 inflow face returns
   exactly 0 with **0 pressure iterations** — the forcing never enters `u*` at all. This is what
   makes the work order's literal Couette gate unrunnable, and it means **no wall-driven shear case
   can currently be built over an immersed body**.
3. **Domain BCs set AFTER the geometry are silently ignored.** All-fluid channel, BCs first:
   drives, 10 pressure iterations, exact linear profile. Same calls, BCs after
   `set_pressure_geometry` / `set_solid`: exactly 0, 0 iterations, no warning. (`set_wall_slip_length`
   is order-INDEPENDENT by construction — measured bitwise — but the BC ordering trap is real and
   undocumented.)
4. **The driver-side divergence guard is still missing** (WO-V7's second carry): a run whose `dt`
   has fallen orders below its initial capillary limit while `t` stops advancing is diverging, and
   nothing says so. Part A removes the cause WO-V7 found, not the class.

### Open / deferred

* **The cross terms of the tangential projector** (`-n_c n_j u_j`, `j != c`). Zero on every
  axis-aligned wall, O(n_c n_j) on a packing. Needs the other two components interpolated at this
  DOF and therefore couples the segregated component solves — a separate rung. Until then a
  curved-solid slip result carries this approximation; say so when reporting one.
* **A sandwiched axis keeps no-slip** (counted, `(0,0,0)` everywhere here). The Robin closure
  generalizes to the sandwich case in closed form — with `P_a = theta_a + lam`,
  `Q_a = theta_a^2 + 2 lam theta_a` the determinant is `P_p Q_m + P_m Q_p` and the `lambda = 0`
  limit reproduces `poly_D_sandwich * (theta_m + theta_p)` exactly (verified algebraically) — but it
  was not implemented, because a one-cell fluid gap does not resolve a slip length and no gate
  exercises it.
* **SCHEME 1 (cell-average) has no slip branch.** `flow` builds `buildIbmOverlay<0>` everywhere, so
  this is unreachable today; the `lamAxis` argument is ignored for SCHEME != 0 rather than silently
  mixing point-value slip terms into cell-average polynomials.
* **The float storage floor is 1.15e-06 relative on the wall rows**, which is what caps gate 1. If a
  sharper wall gate is ever wanted, it is the `MReal`/overlay precision that has to move, not the
  closure (and see the memory note that `MREAL_DOUBLE=ON` silently builds FLOAT).

## WO-W12 findings (2026-09-03, Opus) — Part III rungs W1 + W2

Branch `vof-w12` in **flow** (nothing in `core` needed changing). Backend: CUDA (RTX 5080,
`nvidia-cuda`), MPI np 1/2/4/8. `OMP_NUM_THREADS` 2–8, `OMP_PROC_BIND=false` throughout. **The
host and the GPU carried five other agents' jobs for the whole session** (GPU utilisation pinned at
95 %, six compute processes), so the only timing quoted as a finding is the device-vs-host packing
ratio, and even that is qualified below.

### W1 (a) — master assignment: LPT wins, and the ORB's contiguity constraint is why

Three modes, all pure functions of the REPLICATED block table (so every rank computes the same
assignment with no communication — which is what lets a re-assignment happen mid-run without
breaking a bitwise gate): `RoundRobin` (W0's), `Lpt` (greedy longest-processing-time on the block
cell counts) and `WeightedOrb` (core's `BlockDecomposer<1>` over a 1-D block space, the work
order's route). Measured on a 64-bubble lattice with radii 2 … 9 cells, i.e. block cell counts
spanning **10.2×** (1728 … 17576, total 515 584) — the regime a round robin by block id is blind to:

| np | round robin | LPT | weighted ORB |
|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 2 | **1.1420** | **1.0000** | **1.0000** |
| 4 | **1.4528** | **1.0000** | **1.0000** |
| 8 | **2.1817** | **1.0000** | **1.0000** |

Both replacements are perfect on this swarm, so the tie-break is structural, and it goes to LPT:
**the ORB's blocks must be CONTIGUOUS in block id.** On a lattice ordered by id the sizes happen to
be well mixed; order the same 64 blocks so the big ones are adjacent and the ORB cannot do better
than the coarsest contiguous cut, while LPT is unaffected. LPT is also 4/3-competitive by Graham's
bound (asserted in the ctest) and needs no dependency. **`Lpt` is what `set_vof_block_assign(1, N)`
ships**; `RoundRobin` stays the container default so every W0 number reproduces.

**Re-assignment is exact.** Only two things in a block are state — the colour and the previous
centroid — so a master change is one contiguous message plus four doubles (`VofBlock::serializeAux`;
without the centroid a migrated marker's reported *velocity* would blank for one step). Gated in
`test_vof_blocks_mpi` by a 64-bubble LeVeque scene with `assignMode = Lpt, reassignEvery = 8`
compared against the SAME container on one rank (which has one rank and therefore never migrates):
**first differing step −1 (none) at np 1, 2, 4 and 8**, per-marker volumes agreeing with the
single-rank run to **0.000e+00**, with **48 / 98 / 120 master changes actually migrated** at
np 2 / 4 / 8 (3.7 MB in 44 msgs on rank 0 at the last np=2 event). Dynamic imbalance under LPT on
that scene: **1.002 / 1.003 / 1.012** at np 2 / 4 / 8, against the round robin's 1.337 / 1.559 /
3.117 on W0's three-bubble scene. At np ≥ 2 **16 / 27 / 34 of the 64 blocks have a master that owns
none of their cells**, so the whole state arrives by message.

### W1 (b) — device-resident packing

The four block transfers are now ONE templated pattern with four instantiations (gather face
velocity: extended box, 3 components; gather colour: inner box, 1; scatter colour MAX: inner box, 1;
scatter force SUM: inner box, 3), whose pack/unpack kernels run in the block's own memory space.
Host traffic is one staging copy **per MPI message** instead of one mirror of the whole local patch
per step, and **none at all** for the master's own cells (a device-to-device kernel). That is the
order `core`'s grid halo grew in, and the staging survives because the MPI here is not
CUDA-aware.

**Bitwise inert** (it is a copy of a double at every step), gated four ways on an 8-bubble LeVeque
scene — (host, no pool), (device, no pool), (host, pool), (device, pool) — **0 cells differ,
max|dV| = 0.000e+00** in all three comparisons.

**The ratio, with its caveat.** On a comparatively quiet GPU the whole block step read
**2.731 ms (device) vs 4.626 ms (host) = 1.69×**. Re-measured later the same session with the GPU
saturated by five other agents: **79.4 vs 63.4 ms = 0.80×**, i.e. the ordering INVERTS under
contention — the device path issues ~4 small kernels per piece where the host path issues three
big mirror copies, and kernel-launch latency is what a contended GPU serialises. Both numbers are
reported because the second is not evidence against the design, it is evidence that this
measurement needs a quiet machine (WO-V9's remit) and that the per-piece kernels want batching.

### W1 (c) — the block pool

Advectors retired by a re-centring are recycled by EXACT extent and handed back with their colour
and three face-velocity fields zeroed, which is the state a freshly `init`ed one is in — so the
pool is bitwise inert, and the ctest gates that rather than asserting it. Hit rate is entirely a
property of the scene: on the strongly deforming LeVeque field the boxes change size almost every
re-centring (**4 hits / 56 misses** on 8 bubbles over 360 steps), while a translating bubble keeps
its box SIZE and only moves its origin, which is the case the pool was written for.
`vof_block_pool_stats()` reports both.

### W1 (d) — per-bubble Lagrangian outputs

`vof_block_stats()` now also carries **`area`**, the marker's interface area in cell units squared:
the sum over the inner box of the PLIC polygon area on the MYC normal (`mycNormal` → `plicAlpha` →
`plicPolygon` → `polygonAreaCentroid`, the same planes the curvature cascade reconstructs).
Measured on a seeded sphere of R = 9.00 cells: **1016.36 against 4πR² = 1017.88, −0.15 %**.
`id`, `master` and the box are now filled for EVERY block from the replicated table (they were only
being written on the master, so a rank mastering nothing reported a zero box — a W0 reporting bug
the docstring already contradicted).

**A wisp guard is required here too, and finding out why was the session's second bug.** A
Weymouth–Yue round-off wisp satisfies `0 < C < 1`; its MYC normal is degenerate (an all-zero
stencil returns `(1,0,0)`) and `plicAlpha(1,0,0,1e-15)` puts the plane just inside the face, so the
**polygon is the FULL unit square and the cell contributes an area of 1**. Three such cells made a
marker's reported area differ by **3.0 cells²** between np = 1 and np = 4 off a colour that agreed
to 1e-14. With `areaEps = 1e-8` the same comparison reads **1.7e-13**.

### W2 — the NS coupling

Each master block runs its OWN `VofCurvature` cascade on its own dense box and forms the V4
balanced-force face force there (`σ κ_f (C(i) − C(i−s_c))/h`, the same `csfFaceCurvature` +
`csfFaceForce` pair `addCsfRhs` applies to the global field); the three face fields are scattered
**UNPACK_SUM** into the RHS through a sibling branch, `addCsfRhsBlocks`. **Why the force and not
the curvature is scattered:** κ is not additive and the union colour is a `max`, so a face between
two OVERLAPPING markers has no single (κ, ΔC) pair to build a force from. The force is the additive
quantity, and forming it where each marker's own colour still exists is the only place the
balanced-force pairing — the same face difference the projection's gradient uses — is available per
marker. This is TBFsolver's `VOF.f90::computeSurfaceTension` structure (block `stx/sty/stz` →
`boxes_2_grid_vf(…, UNPACK_SUM)`) on the suite's own kernels.

Inertness: `vofBlockCsf()` is false whenever `vofBlocks_` is null, so `csfActive()`, the `step()`
CSF dispatch and the `advectVof()` dispatch all reduce to their W0 text character for character.

**A defect the gate found, and it is the one worth remembering.** `set_surface_tension` sets the
STRUCTURED cascade's wisp guard `interfaceEps = 1e-8` (V4/WO-P: a wisp's zero-area PLIC polygon
returns |κ| up to 1e8 and the face between it and a real interfacial cell then carries a force
eight orders too large), but each block allocated its own `VofCurvature` with the V3 default of 0.
Measured consequence: the distributed run's scattered CSF face force was **bitwise identical to the
single-rank one after ONE step and differed by 6.7e-3 after TWO** — a 13-order amplification of a
3e-16 colour difference through a flipped cascade branch — and the coupled state then diverged to
du 3.3e-3, dP 3.7e-3 at 12 steps. `VofBlockSet::curvProto` now carries the cascade's tunables and
the solver sets them from its own. **A per-container copy of a shared estimator must copy its
configuration, not just its code**; the general rule this instance teaches is that any container
that instantiates `VofCurvature` itself has to be handed `interfaceEps`.

**Also fixed here (not a defect, a scope correction):** the kinematic entry point
`advect_vof_blocks(dt)` refuses a face field whose discrete divergence exceeds 1e-10, which is
right for a prescribed velocity but wrong inside `step()`, where the advecting field is the
projection's own output and its residual divergence IS the conservation floor — exactly as for the
structured `advectVof()`, which never carried such a check. Measured projected residual at ratio 10
without `PECLET_FLOW_EXACT_RESIDUAL`: 1.8e-7 … 1.1e-5, i.e. the check would refuse every coupled
step. `advectVofBlocks(dt, requireSolenoidal)` — true from Python, false from `step()`.

#### Gate 1 — Hysing case 1, block path vs global-field path. **PASS**

Same physics, same discretisation, same adaptive-dt schedule, both **without**
`enable_vof_momentum`; the only difference is which container carries the colour and where the CSF
face force is formed. Quasi-2-D 64 × 4 × 128, T = 3, 2032 steps each:

| | v_rise max | at t | y_c(3) | volume drift | pressure | max\|div(open u)\| |
|---|---|---|---|---|---|---|
| global colour field | 0.2827 | 1.050 | 1.2086 | −3.84e-12 | 23/600 | 9.08e-06 |
| **blocks** | **0.2827** | **1.050** | **1.2086** | **−2.78e-12** | 23/600 | 9.08e-06 |
| block − global | **−0.00 %** | | **−0.00 %** | | | |

(Gate: both within 1 %.) Neither run touched the pressure cap (rule 3b). Both sit +17.0 % / +11.8 %
off Hysing's published 0.2417 / 1.081 — that is the documented cost of running WITHOUT momentum
consistency at ratio 10 (WO-P measured 15 % on the peak rise velocity for exactly this case) and is
identical in the two containers, which is what the gate is about.

**The colour convention flips, and it does not matter.** With blocks the union `C` is the DISPERSED
phase (UNPACK_MAX starts from an empty union, so a cell no marker covers must read 0), whereas the
structured script has `C = 1` in the heavy liquid. The property models are written mirrored. The CSF
is invariant under the flip — κ(1−C) = −κ(C) and ∇(1−C) = −∇C, so σκ∇C is unchanged — which is what
makes the comparison meaningful rather than a coincidence.

#### Gate 2 — a 3-D bubble at Eo = 10, Mo = 1e-3, ratio 100 vs the Grace diagram. **PASS (−6.2 %)**

D = 16 cells, box 64 × 64 × 144 (periodic laterally, walls top and bottom), 1018 steps, pressure
21/800, max|div| 7.07e-6. Measured terminal velocity from a linear fit of the marker centroid over
the last third of the trajectory: **0.5439 cells/s** (instantaneous plateau 0.5526 from t ≈ 37 on),
Re = 21.8, aspect ratio √(m_xx/m_zz) = **2.256** (ellipsoidal, as the Grace diagram requires at this
(Eo, Re)), volume drift **+6.5e-11**, interface area 928.7 against the seed sphere's 804.2 cells².

**The gate as the work order states it measures an ill-posed reference, and this is a rule-4
correction rather than a failure.** Clift, Grace & Weber's correlation contains a **dimensional**
factor `(μ_l/μ_water)^−0.14` that a purely dimensionless (Eo, Mo, ratio) specification does not
determine:

  * silently setting it to 1 gives U_T = **0.9587** cells/s, and our result would read −43.3 %;
  * but **Mo = 1e-3 at water-like ρ and σ forces μ_l ≈ 0.0727 Pa·s ≈ 81 × water** (water itself is
    Mo ≈ 2.5e-11, twelve orders away), so a factor of 1 describes no liquid at all. The physical
    system this (Eo, Mo) implies at ρ = 1000 kg/m³, σ = 0.065 N/m, g = 9.81 m/s² is a
    **d = 8.14 mm bubble in a 72.7 cP glycerol/water mixture** — the standard Mo = 1e-3 fluid —
    giving H = 20.18, J = 9.140 and **U_T = 0.5796 cells/s** (Re 23.2, Fr 0.733).

Against that: **−6.2 %** (trajectory fit) / −4.7 % (plateau), inside the 10 % gate.
`grace_terminal()` now derives the factor and prints both readings, with the argument in its
docstring, so nobody repeats the mistake.

#### Gate 3 — two bubbles in line. **INCONCLUSIVE as written; PASS on the corrected gate (3b)**

*As written* (a large bubble, D = 14.4 cells, rising in line behind a small one, D = 9.6, seeded
with a 2-cell film, Eo = 10, ratio 100, box 60 × 60 × 132, 1120 steps each):
**the film never drains below 2 cells and NEITHER container merges.** Blocks: minimum axial gap
**2 cells**, peak shared liquid **0.000**, marker volumes drifting +5.5e-6 / −3.8e-6 relative (the
coupled-run conservation floor set by the projection's own 1.1e-5 divergence residual, not by the
container). Control: minimum gap **2 cells**, never merged. So the gate does not reach the regime in
which the two containers can differ — at D/Δ ≈ 10–14 and Eo = 10 the lubrication + capillary
resistance holds the film at the grid scale for as long as we can afford to run, and the gate's
discriminating question is never asked. **Recorded, not tuned.**

*The corrected gate*, **3b**, asks it directly and passes. Seed the two markers ALREADY TOUCHING
(centre distance = R_s + R_l − 1, so their bands share cells from step 0) and run 400 real
two-phase steps at Eo = 10, ratio 100, box 48 × 48 × 84:

| | markers | volumes | drift vs the exact spheres | union | shared liquid | blobs on the axis |
|---|---|---|---|---|---|---|
| **blocks** | **2** | 1563.487 / 463.266 | **+1.9e-5 / +4.2e-5** | 2016.309 | **10.444 cells** | 1 |
| control (one field) | 1 | — | — | 2017.956 | 0 by construction | 1 |

pressure 28/800 and 20/800 (uncapped), max\|div(open u)\| 2.8e-6 in both. **As a colour field the
two states are indistinguishable — one blob, the same 2016/2018 cells of liquid — and only the
block container still has two markers, each carrying its own whole bubble.** That is the rung's
raison d'être through a real NS step rather than a kinematic one.

*One seeding fact this gate paid for.* `enable_vof_blocks_from_field` gathers each marker's colour
out of the global union restricted to its box, and when two markers OVERLAP at seed time there is
no way to split that union between them: clipping the boxes at the midplane gave the two markers
**−2.7 % and +7.1 %** of their intended volumes (each adopting a slice of the other). For markers in
contact the sphere seeder (`enable_vof_blocks`, which paints each block from the exact analytic
fraction) is the only correct entry point — and the general lesson for W4's coalescence/breakup work
is that a per-marker colour SOURCE, not a union plus a box, is what a marker needs at birth.

#### Gate 5 — MPI. **PASS at np 1 / 2 / 4**

`tests/kokkos_mpi/test_vof_blocks_ns_mpi.cpp`: 24 × 24 × 48, two markers stacked along the CUT
axis, walls on ±z, ratio 10, gravity, 12 steps of a real `IbmSolver::step()`, run twice — with the
block CSF and with surface tension off — against a full-grid single-rank reference gathered on
rank 0.

| config | np | du | dv | dw | dP | dC | CSF face force |
|---|---|---|---|---|---|---|---|
| blocks, σ = 0 | 1 | 0 | 0 | 0 | 0 | 0 | — |
| | 2 | 8.4e-19 | 5.8e-19 | 2.2e-18 | 5.6e-17 | 1.1e-16 | — |
| | 4 | 5.5e-19 | 5.7e-19 | 2.6e-18 | 5.6e-17 | 1.1e-16 | — |
| blocks + block CSF | 1 | **0** | **0** | **0** | **0** | **0** | **0** |
| | 2 | 1.3e-15 | 1.4e-15 | 1.1e-15 | 3.6e-15 | 1.1e-14 | 1.4e-14 |
| | 4 | 9.4e-16 | 1.1e-15 | 8.6e-16 | 3.2e-15 | 1.1e-14 | 1.3e-14 |

np = 1 is **bitwise**; np > 1 sits at the pressure driver's documented reduction-order floor (the
same tolerances `test_vof_surface_tension_mpi` uses: u 1e-12, P 1e-9, C 1e-12). Per-marker volumes
agree with the single-rank run to **0.000e+00** and the areas to 1.7e-13. The two markers carry
**0.175 cells of SHARED liquid** — a state a single colour field cannot represent — through a
distributed NS step.

#### Gate 4 — TBFsolver's `channel_18`. **TRANSCRIBED AND RUN; no reference exists to compare to**

`tests/study/vof_channel_18.py`. Every number was read off the Fortran that consumes the specs, not
off the file names, and three of them are not what the spec file suggests:

* **`gCH = 0.1` is a streamwise GRAVITY, not the driving pressure gradient.** With `flowCtrl 1` the
  applied mean gradient is `fs = tau_w − <rho> gCH` and the x-momentum source is
  `fs + rho gCH = tau_w + (rho − <rho>) gCH`, so `gCH` contributes only BUOYANCY and drops out of
  the mean balance. `tau_w = (Re_tau/Re)^2 = 1.800588e-3` with `Re = 1/nu_l = 3000`, hence
  `u_tau = Re_tau nu_l/h = 0.0424333` **exactly, by construction** — Re_tau is imposed, not
  measured. (`g 0.d0` in the specs is the WALL-NORMAL gravity and is off.)
* The bubbles are therefore driven AGAINST the mean flow: **this is a downflow channel**, and the
  gas phase carries a negative streamwise force.
* `initBubbles method 2, nbx 6, nby 1, nbz 3` puts the 18 bubbles on the **centreline** y = h at
  `x0 = (i−½)Lx/6`, `z0 = (k−½)Lz/3`; x is streamwise (periodic), **y wall-normal (walls)**,
  z spanwise (periodic). Groups: ratio 10, viscosity ratio 1, D/h = 0.25, void fraction 1.49 %,
  Eo = rho gCH D²/sigma = 1.25, Mo = 9.9e-9, D+ = 31.8.

**The transcription and what it costs.** flow's cells are cubic and TBFsolver's grid is not
(dx+ 2.08 / dy+ 1.59 / dz+ 2.08 on 192 × 160 × 96), so the case is mapped onto an ISOTROPIC
**128 × 80 × 64** grid: h = 40 cells, **Delta+ = 3.18** (coarser than the reference in every
direction), D = 10 cells, and the box aspect rounds to 3.20 × 2 × 1.60 h against 3.1416 × 2 ×
1.5708 (**+1.9 % in x and z**; ny = 80 was chosen over the exact 81.5 because the pressure
multigrid's depth lives on the factors of two). Measured void fraction **1.438 %** against the
case's 1.492 %, the same +1.9 % × +1.9 % of box.

**The initial condition is TBFsolver's own.** `channel_18/0/{ux,uy,uz}` is a converged single-phase
turbulent snapshot in raw Fortran STREAM (one `int32` count, then the internal field x-fastest with
no ghosts, staggered sizes 193×160×96 / 192×161×96 / 192×160×97, plus a 96-byte trailer the reader
ignores; `psi` and `phi0*` are identically zero). The script reads it and trilinearly resamples each
staggered component onto our grid — the nearest thing to the same start a different grid admits.
Verified: the resampled bulk velocity is **U_b = 0.6113**, i.e. TBFsolver's own snapshot mean to
four figures (Re_b = 1834, U_b+ = 14.41).

**Run.** 6000 steps to **t u_tau/h = 0.841** in 5272 s on the shared GPU; **pressure 13/800
(uncapped, rule 3b satisfied)**, max|div(open u)| **1.15e-6**; all 18 markers held their volume to
the printed 523.60/523.61 against a seed of 523.60 for the whole run, total interface area drifting
5649 → 5758 cells² as the bubbles deform. Profiles averaged over the second half of the window
(`--out` writes y/h, <u>/u_tau, void fraction).

| | measured |
|---|---|
| peak `<u>/u_tau` | **17.58 at y/h = 0.74** (not at the centreline) |
| `<u>/u_tau` at the centreline | **11.99** |
| void fraction at the centreline | **0.181** |
| void fraction at y/h < 0.81 | **0** (the bubbles have not left the core) |
| wall-gradient u_tau (first cell) | **0.03730**, −12.1 % of the imposed 0.042433 |

Both features are the expected signature of the case rather than convergence: the bubbles sit in the
core carrying a NEGATIVE streamwise force, so they flatten and then invert the centreline velocity
(the peak moves off-axis), and the fixed pressure gradient meets the extra drag by decelerating —
`u_tau` measured at the wall falls below the imposed one while the flow is still adjusting.

**What this is NOT.** One eddy turnover is `h/u_tau = 23.6 s = 7.7 k steps` at the capillary/CFL dt
this case admits, so 6000 steps is **0.84 turnovers**: a transient from TBFsolver's single-phase
snapshot with bubbles inserted, not a statistically steady state (which needs ~20 turnovers ≈ 44 h
of exclusive GPU). The void-fraction profile in particular is still essentially the initial
condition — the bubbles have migrated by less than one diameter.

**And there is nothing to compare it to.** The repository ships **no reference statistics for
`channel_18`**: 60 tracked files, no profiles, no post-processing, no data directory; the only
published artefact is a qualitative streamwise-velocity contour in `user_guide.pdf`. So the numbers
above are reported as **our first datum**, exactly as the work order instructed for that case, and
the cross-code comparison the rung was aiming at needs TBFsolver to be BUILT AND RUN (it is a
Fortran+MPI code with no shipped output). That is the honest scope of gate 4 as delivered:
the case is transcribed, the physics runs, the solver is healthy, and the comparison is open.

#### Gate 6 — inertness. **PASS**

`tests/kokkos` on the nvidia-cuda backend at the final commit: **100 % tests passed, 0 failed out
of 32** (2486 s), the ten VoF ctests plus `vof_phase_change` and `vof_blocks` among them. The block
container, the block CSF and the W1 machinery allocate nothing and change no branch unless
`enable_vof_blocks` runs: `vofBlockCsf()` is false whenever `vofBlocks_` is null, so `csfActive()`,
the `step()` CSF dispatch and the `advectVof()` dispatch reduce to their previous text exactly.

### Summary of the gates

| gate | verdict | headline number |
|---|---|---|
| W1a assignment | PASS | round robin 1.142/1.453/2.182 → **LPT 1.0000** at np 2/4/8 (10.2× size spread) |
| W1a re-assignment | PASS | **bitwise at np 1/2/4/8**, 48/98/120 master changes migrated |
| W1b device packing | PASS (bitwise) | **1.69× quiet / 0.80× contended** — see the caveat |
| W1c block pool | PASS (bitwise) | 4/60 hits on LeVeque, ~100 % on translation |
| W1d per-bubble area | PASS | **−0.15 %** vs 4πR² at R = 9 cells |
| W2-1 Hysing block vs global | **PASS** | **−0.00 % / −0.00 %** on v_rise max and y_c(3) |
| W2-2 Grace | **PASS** | **−6.2 %** vs the correctly-conditioned correlation (gate rewritten) |
| W2-3 two bubbles in line | **INCONCLUSIVE** | film holds at 2 cells; neither container merges |
| W2-3b markers in contact | **PASS** | **10.4 cells of shared liquid**, volumes to 4e-5 |
| W2-4 channel_18 | transcribed, run, **no reference exists** | 0.84 eddy turnovers, uncapped |
| W2-5 MPI | **PASS** | bitwise at np 1; 1.3e-15 / 3.6e-15 / 1.1e-14 at np 2/4 |
| W2-6 inertness | **PASS** | 32/32 ctests |

### After the rebase onto `origin/main`

Rebased onto `9ad0646` (which had meanwhile taken WO-V6's dynamic contact angle, WO-V6b's design
and WO-V7's pore-scale campaign). Two textual conflicts, both "two sessions appended a section at
the same anchor": `tests/kokkos_mpi/CMakeLists.txt`'s gated target list (`vof_wetting_dynamic_mpi`
against `vof_blocks_ns_mpi` — both kept) and the findings log's newest-first head (WO-V7's entry
against this one — both kept, this one first by date). Nothing in `src/vof/block_*.hpp` conflicted,
which is the direct evidence that no concurrent session touched the block files.

Rebuilt and re-run at the rebased head: **`tests/kokkos` 33/33 pass** (1175 s; 33 now, the extra one
being WO-V6's `vof_wetting_dynamic`), **`test_vof_blocks_mpi` PASSED at np 1/2/4/8** with the same
48/98/120 master migrations, and **`test_vof_blocks_ns_mpi` PASSED at np 1/2/4** — bitwise at np 1,
du 8.0e-16 … 1.3e-15, dP 4.0e-15 … 8.0e-15, dC 7.1e-15 … 1.0e-14 at np 2/4 (the last bits of the
reduction floor move with the upstream solver changes, as they must).

### Open / deferred

1. **The block container is still ALL-FLUID.** `enable_vof_blocks` raises on `set_solid`; the
   cut-cell block (openness-weighted fluxes on the block's own geometry) was in W0's deferred list
   and is still there — W2 was NS coupling, not geometry.
2. **No momentum consistency on the block path** (deliberate, per the work order), so the rung is
   rated to ratio ≈ 100 with motion, exactly like V2a. W2b (union-field momentum sweeps) is
   unwritten.
3. **The film between two markers is only ever resolved to the grid.** Gate 3 measured 2 cells at
   D/Δ ≈ 10 and neither container merged; nothing here says what happens at D/Δ = 30, and the
   sub-grid drainage model that would make the answer physical is rung W4's.
4. **Per-piece pack kernels are launch-latency bound.** The device path issues ~4 small kernels per
   (block, rank) piece; on a saturated GPU that loses to three big mirror copies. Batching the
   pieces of one message into one kernel (a small index View) is the obvious fix and belongs with
   WO-V9's profile.
5. **`enable_vof_blocks_from_field` cannot seed OVERLAPPING markers.** It gathers each marker's
   colour from the union restricted to its box, so two markers in contact each adopt a slice of the
   other (−2.7 % / +7.1 % measured). Markers at birth need a per-marker colour SOURCE; the sphere
   seeder has one, an arbitrary shape does not. W4 (breakup → split a block) will hit this first.
6. **channel_18 has no cross-code comparison yet** (item above): it needs TBFsolver built and run,
   and a statistically steady window (~20 eddy turnovers ≈ 44 h of exclusive GPU at this grid).

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
