# VoF work orders — rung V2 (two-phase Navier–Stokes, staggered, no surface tension)

Companion to `vof_workorders.md` (**read its "Shared preamble" — build commands, hard rules,
escalation policy — it applies verbatim here**) and to `suite/docs/VOF_PLAN.md` §4 V2.

V2 is split because its two halves have different risk profiles. **WO-J** is plumbing with a
loud gate; **WO-K** is the subtle numerical rung that everything above ratio ~100 depends on
and that cannot be bolted on later. Do them in order — WO-K's consistency gate is meaningless
until the WO-J chain runs.

State entering V2 (all landed and gated on host + CUDA):
- `src/vof/plic.hpp` (V0) — SZ/Lehmann–Gekle plane↔volume, MYC normals, slab flux volumes.
- `src/vof/advect_wy.hpp` (V1) — `WyAdvector`, own g=3 halo, worklist, CFL cap at Weymouth's
  proven 3D bound 0.25 (inclusive), np 1/2/4 bitwise, volume drift ~1e-16.
- varRho + varMu validated on CUDA and under MPI incl. walled-axis cuts (V-1, WO-F, WO-G).
- Pressure drivers: Chebyshev (varRho default), MG-PCG, FCG, BiCGStab (WO-B, WO-C, WO-H).

---

## WO-J (rung V2a) — wire the colour field into the solver  [OPUS]

**Goal.** One phase transported by geometric VoF drives the fluid properties, and the existing
variable-density projection carries the resulting density jump. No surface tension, no
momentum consistency yet (that is WO-K) — so this rung is only valid at modest density ratios;
say so in the API docs.

**Do.**
1. Register a colour field `"C"` on the solver with **its own g=3 halo topology** (plan §3
   rule 1 — do NOT widen the solver's `G = 2`; the FieldSet supports per-field ghost widths,
   and `WyAdvector` already owns this pattern). Expose `set_vof(...)` / `get_vof()` bindings
   with the zero-copy view convention used by the other fields.
2. Advect C in `step()` with the **projected face velocities** — `get_face_velocity(c)` is the
   uniform seam (staggered `C[c].u`; the collocated branch exists but V2 is staggered-only,
   so assert/throw on collocated for now). Place the advection where `advanceScalars()` sits
   relative to the projection, and justify the ordering in the findings entry.
3. Drive ρ(C) and μ(C) through the **existing property closures** (`LinearMix`; harmonic face
   mean available for μ) — no new closure machinery. Then `set_density_mode("variable")` does
   the rest.
4. **Interface-local CFL limiter.** V1 measured that a *global* CFL max over-throttles badly
   (Zalesak: 0.314 at a quiescent far corner while the interface never exceeded 0.157). The
   solver's VoF dt limit must therefore be taken over **interface-adjacent cells only**
   (mixed cells and their face neighbours), not the whole domain. Reuse the V1 worklist.
5. Add the **harmonic ρ_f face-mean option** to `buildRhoCoeff` (`mac_pressure.hpp:251`),
   default OFF (arithmetic stays the default — it is what makes the hydrostatic balance exact,
   `variable_density_projection.md` §1). This is the flagged coarsening trap of
   `MULTIPHYSICS_PLAN.md:474`; ship the knob now, measured, rather than when it bites.

**Gates.**
- **The hydrostatic acid test, now driven by C** — a two-layer column initialised as a sharp
  C field, ratio 1000, gravity closure, at rest: steady max|u| at machine zero and
  ∂P/∂z = −ρ_f·g. This is the loud gate; it fails on any inconsistency between the closure's
  face density and the projection coefficient. (Note it passed at 2.75e-17 in WO-A with a
  hand-set ρ; reaching the same number through the C → closure → ρ chain is the point.)
- Volume conservation of C through the coupled step, ≤ 1e-13 relative over 1000 steps.
- Sharp-interface **Rayleigh–Taylor**: growth rate vs linear theory; compare against the
  existing diffuse-C `tests/study/rayleigh_taylor.py` record and report both.
- np 1/2/4 bitwise, host + CUDA. Single-phase regression +0.00%, identical iteration counts.
- With C ≡ const the solver must reduce **bitwise** to the single-phase path.

**Traps.** The C halo (g=3) and the solver block (G=2) have different extents — every bridge
between them is a place to get the offset wrong; the hydrostatic test is the canary. Chebyshev
is the varRho default and re-estimates its bounds on every coefficient rebuild — with a moving
interface that is now every step (WO-B: 67% of the projection); record the cost, do not "fix"
it here (that is S2).

---

## WO-K (rung V2b) — momentum-consistent transport  [Fable design, below; Opus implements]

**Why this is not optional.** With mass and momentum advected by *different* fluxes, a mixed
cell multiplies the gas acceleration by the liquid density: a spurious interfacial momentum
source of order Δρ. Symptoms are artificial atomisation and kinetic-energy spikes; the
literature is unambiguous that naive advection breaks down around ratio 1000 unless the
resolution is absurd. Quantified payoff (Arrufat et al., *Computers & Fluids* 215:104785,
2021): a raindrop at ratio 831.8 is accurate within 15% at **15 cells/diameter**, versus
~200 without consistency.

**The construction (MAC / staggered).** The momentum control volume for component `c` is
shifted half a cell in direction `c` from the pressure cell. Consistency requires that the
density carried on that shifted volume be advected by the *same geometric fluxes* as C:

1. Build a **half-shifted colour field** `C^c` on the momentum control volumes: for each
   momentum CV, the liquid volume is obtained by clipping the PLIC planes of the two
   overlapping pressure cells into the shifted box. Use `plicSlabVolume` — the V0 toolbox is
   scale/offset invariant precisely so this is a rescale, not new clipping code (see the
   WO-D findings note). Do **not** interpolate C: interpolating fractions destroys sharpness
   and is the classic error here.
2. Advect `ρ^c u_c` with the WY split scheme on the shifted CVs, using the **same sweep
   order, the same frozen dilation flag, and face fluxes derived from the same PLIC planes**
   as the C advection of that step. Sharing the frozen flag across the two advections is what
   makes the two updates telescope identically.
3. Recover `u_c = (ρ^c u_c) / ρ^c` with a floor on ρ^c (document the floor and its effect).

**The decisive gate — the consistency (uniform-velocity) test.** Initialise an arbitrary sharp
C field (droplet, slab, tilted interface) with a **spatially uniform velocity** and a density
ratio of 1000, no gravity, no viscosity. The velocity field must remain **uniform to machine
precision** for many steps. This is exact for a consistent scheme and O(Δρ) wrong for an
inconsistent one, so it is pass/fail at the 1e-15 level with no tuning knob — the momentum
analogue of the hydrostatic acid test. Run it at several ratios (10, 10², 10³, 10⁴): a
consistent scheme is flat in ratio; an inconsistent one degrades linearly.

**Further gates.**
- Falling raindrop, ratio ~800, **15 cells/diameter**: stable, and terminal velocity within
  ~15% (the Arrufat criterion). Report the same case without consistency for contrast.
- Rayleigh–Taylor and the WO-J battery re-run: report deltas.
- Momentum conservation to round-off in a periodic box.
- np 1/2/4 bitwise, host + CUDA; single-phase regression +0.00%.

**Known caveat to check, not to hide** (Arrufat §5): momentum consistency can excite a
near-Nyquist growth on under-resolved shear layers. Run the RT/KH gates at more than one
resolution and report whether it appears.

**Escalate** if the uniform-velocity test cannot be made exact: that means the two advections
are not sharing fluxes, and the fix is structural, not a tolerance.

---

# Infrastructure blockers surfaced during V2 preparation

## WO-L — `MPI_ERR_TRUNCATE` at np=4 in the communication-avoiding halo path  [OPUS]

**Severity — corrected 2026-08-31, read this before planning around it.** WO-I first reported
this as a hard blocker and then **re-ran and corrected itself**: the failure is
**load-triggered, not deterministic**. On an idle machine every affected test passes at
*default* settings (`ghost_projection_mpi_np4` 96.7 s, `varmu_mpi_np4` 249 s host,
`dragbeta_ghost_mpi_np4` 4.2 s; final tally 48/48 MPI on host-openmp and 48/48 on CUDA). It
surfaced only while two other agents' batteries were running. So the np-bitwise gate is **not**
broken and this does **not** block new physics — but it is a real race, and **a CI runner is a
loaded machine**, so it will appear there far more often than it does locally. Practical rule
in the meantime: don't run three or more heavy batteries concurrently, and re-run a lone np=4
`MPI_ERR_TRUNCATE` on an idle machine before believing it.

**Why it is still worth fixing.** Every rung of this campaign is gated on "np 2/4 bitwise vs
np 1", and a gate that can fail for reasons unrelated to the code under test erodes trust in
exactly the instrument that has caught every defect so far (`varmu_mpi_np4`,
`bodyforce_ghost_mpi_np4`, `ghost_projection_mpi_np4` are all pre-existing and unrelated to
this campaign's changes). `PECLET_FLOW_CA=0` cures it, which localises the mechanism.

**Evidence in hand** (do not re-derive):
- Load-sensitive: the same binaries pass standalone and fail inside a loaded `ctest` run. WO-F
  saw it once and filed it as a load flake; WO-I showed it is reproducible and CA-dependent.
  Treat "flaky" as a symptom of a race, not as noise.
- **Every affected grid cuts two axes into exactly two blocks.** With np=4 on a 2×2
  decomposition a rank has the same neighbour across more than one axis, and diagonal/corner
  neighbours coincide with face neighbours.
- `PECLET_FLOW_CA=0` cures it. CA (`suite/docs/COMMUNICATION_SCALING.md`, default ON) exchanges
  a **2-deep** ghost layer once per red-black pair, so width-2 and width-1 topologies are in
  flight in the same step — MG level 0 and domain-BC hierarchies keep g=1 while coarse levels
  use g=2 (`flow/CLAUDE.md`, "Distributed smoother communication").
- **Prime suspect: two in-flight exchanges of different ghost width sharing an MPI tag and a
  neighbour**, so a width-1 receive matches a width-2 send — which is exactly what
  `MPI_ERR_TRUNCATE` means (message longer than the posted buffer). Related known trap in the
  same machinery: the g-independent red-black origin `CutcellMG::parityOg`.

**Do.** Confirm the mechanism before fixing it — post-mortem the failing pair (rank, tag,
counts, expected vs received bytes) rather than reasoning from the suspect alone. Then make
tags unambiguous across concurrently-live exchanges: the tag must distinguish (field or level,
ghost width, axis/direction) so no two in-flight messages between the same rank pair can be
mismatched. Check `GridHalo`/`GridHaloTopology` in `core` for whether the tag space is
per-topology or shared — if the defect is in `core`, fix it there and bump the pointer.

**Gates.** The four tests pass at np=4 with CA **ON**, in a loaded `ctest` run, repeatedly
(run the suite at least 5× — a race that passes once proves nothing); `PECLET_FLOW_CA=0` and
`=1` produce **bitwise identical** results (CA's contract is that it is bit-identical, so this
is also a correctness check on the fix); full MPI suite green host + CUDA; single-phase
regression +0.00%.

**Escalate** if the mechanism turns out not to be tag collision — the fix then depends on what
it actually is, and guessing at communication races produces heisenbugs rather than fixes.

---

# Findings log (rung V2)

### WO-J (rung V2a) — the colour field wired into the solver — **DONE 2026-08-31**

Delivered: `src/vof/colour_field.hpp` (the `G = 2` ↔ `g = 3` bridge + the colour ghost policy),
the `"C"` field and the VoF section of `src/flow_ibm.hpp`, the harmonic-ρ_f siblings in
`src/mac_pressure.hpp`, `interfaceLocalCfl` + `maxCourantInterface()` in `src/vof/advect_wy.hpp`,
`tests/kokkos/test_vof_twophase.cpp` (ctest `vof_twophase`), `tests/kokkos_mpi/
test_vof_twophase_mpi.cpp` (ctests `vof_twophase_mpi_np{1,2,4}`), and `rayleigh_taylor_vof()` in
`tests/study/rayleigh_taylor.py`. Python: `enable_vof / set_vof / get_vof / vof_diagnostics /
vof_max_courant / vof_last_courant / set_vof_cfl_limit / vof_cfl_limit / set_rho_face_harmonic`.
`tests/kokkos` **23/23 green on host-openmp AND nvidia-cuda**; the new MPI ctests green np = 1/2/4
on both backends; single-phase regression **+0.00 % with identical iteration counts** on CUDA.

**The wiring.**

| item | what shipped |
|---|---|
| `"C"` registration | an ORDINARY `G = 2` registered cell field; the `g = 3` block is the advector's own working block, with its own `GridHaloTopology` at width 3 under MPI (`buildVofBlock`, rebuilt from `initMpi` so enable/init order does not matter). The two blocks exchange INNER REGIONS ONLY (`copyInner`, both ways) and each fills its own ghosts with its own policy. |
| advection | `advectVof()`, called from `step()` immediately before `advanceScalars()`. |
| face velocities | `bridgeVelocityToVof()`: `fillVelGhosts(c, 0)` (the solver's own halo + domain-BC fill, i.e. exactly what the Picard loop does) then `vof::copyFaceVelocity` — a whole-block embed carrying the low-face → high-face index shift. |
| ρ(C), μ(C) | the existing `LinearMix` closures, verbatim. No new closure machinery. |
| interface-local CFL | `WyAdvector::maxCourantInterface` + `interfaceLocalCfl` (default OFF; the solver turns it on, so the V1 battery is byte-identical). |
| harmonic ρ_f | `buildRhoCoeffHarm` + `projectCorrectVarHarm` siblings, `set_rho_face_harmonic`, default OFF. The validated arithmetic kernels are untouched. |
| refused | collocated (`enable_vof` throws — rung V8), and an immersed solid (`advectVof` throws — the fluxes are not openness-weighted yet, plan §3 rule 2). An all-fluid `set_pressure_geometry` is supported. |

**Why `"C"` is a `G = 2` registry field and not the `g = 3` buffer adopted into the FieldSet.** The
plan's §3 rule 1 sketch was "adopt the colour field into the FieldSet with its own topology at
width 3". That is not compatible with item 3 of this work order. `applyClosure` indexes its input
and its output with the SAME linear index on the SAME extent, so a closure input MUST live on the
`G = 2` block; adopting the `g = 3` buffer under the name `"C"` would make
`set_property_model("rho", "linear", "C", …)` silently index a `(n+6)³` buffer with `(n+4)³`
strides. Fixing that means editing `applyClosure` — a validated kernel body, which hard rule 1
forbids. So the split is forced, not chosen, and it is also what gives `"C"`
`get_field`/`set_field`/`field_view`/`exchange_field`/`redistribute` for free. What §3 rule 1
actually protects — "do not widen the solver's `G = 2`" — is honoured exactly: `G` is untouched and
the colour field's geometric work runs on a width-3 block with its own topology.

**Ordering, and how WO-K drops in.** The advection takes the `advanceScalars()` slot (after the
projection) because Weymouth–Yue's exact conservation is conditioned on a DISCRETELY
divergence-free advecting field and the only such field in the step is the projection's output —
the same reason `advanceScalars` sits there. Note `u^{n+1}` at the bottom of step *n* IS `u^n` at
the top of step *n+1*: the two placements are the same point in the timeline and differ only in
which side of `updateProperties()` they fall on. Taking this slot means step *n* runs with
ρ(C^n), μ(C^n) — the same time level as its velocity base `u^n`, and the same segregated contract
every other multiphysics field obeys. For **WO-K** the momentum advection must share the fluxes, so
the call moves to the head of the predictor; three structural facts make that a local change:
(i) everything WO-K must share lives inside `vofAdv_` and survives the call — the frozen dilation
flag `cc_`, the per-sweep PLIC planes `mx_/my_/mz_/alpha_`, the face Courant numbers and the
permutation index — so a sibling advector for ρ^c u_c reads those members rather than recomputing,
and "sharing the fluxes" becomes a data-flow fact instead of a convention two call sites must keep;
(ii) the colour advection is ONE call (`advectVof()`), and it moves to a velocity field (`u^n`)
identical in content to the one it consumes today; (iii) the half-shifted colour field `C^c` is a
new `g = 3` field on the SAME advector block — no new bridge, no new halo, and
`bridgeVelocityToVof` already put the velocity there. The planes are on that block: WO-K clips
them (`plicSlabVolume`, scale/offset invariant), it must not interpolate C.

**Gate results** (host-openmp; CUDA identical except in the last bits).

| gate | result |
|---|---|
| A bridge — solver-driven vs a STANDALONE `WyAdvector` on the same physical LeVeque field, 20 steps | max\|ΔC\| **5.6e-16** (CUDA 1.1e-15), L1 2.4e-14 |
| B1 hydrostatic ∂P/∂z = −ρ_f·g through C, ratio 1000, free interface | rel-err **1.1e-15** (frozen 3.4e-16) |
| B2 steady max\|u\|, interface FROZEN, vs the hand-set-ρ reference | ratio 3 `3.0646790727193918e-17`, ratio 1000 `2.1760599219479075e-17` — **BITWISE EQUAL to the hand-set-ρ run** at both ratios |
| B2 free interface, 100 steps | ratio 3 **3.2e-13**, ratio 1000 **2.6e-12** (mechanism below) |
| C volume conservation, ratio-10 sphere sheared, 1000 coupled steps | dV/V peak **5.6e-13**, final −2.4e-13, non-accumulating; max\|div(open·u)\| 1.4e-12…1.5e-11 |
| D1 uniform C stationary under the coupled advection | **exactly 0.0** |
| D2 VoF enabled, no closure, vs VoF off (sheared periodic box) | du **0.0**, dp **0.0** — bitwise inert |
| D3 VoF + ρ(C≡1) vs hand-set uniform ρ + `set_density_mode` | du **0.0**, dp **0.0** — bitwise |
| E harmonic ρ_f ablation, ratio 1000 | ∂P/∂z rel-err **0.3355** (arithmetic: 1.1e-15) — the knob breaks the balance loudly, as designed |
| F interface-local vs global CFL (jet far from a quiescent interface) | **0.0022 vs 0.0492 — a 22× over-throttle avoided** |
| MPI np 1/2/4, host + CUDA, `walls-z` (walled axis CUT) + `shear-per` | np=1 **bitwise 0.0** on u, P and C; np=2/4 `shear-per` du 1.0e-17, dp 3.2e-18, **dC 2.2e-16**, dV/V −3.1e-14; `walls-z` dp 1.2e-11 on \|P\| = 1.15e+03 (rel 1e-14), dC ≤ 2.3e-13 |
| Rayleigh–Taylor, sharp vs the diffuse record (same physics) | sharp 1.50 → **20.20** (×13.5), rate **0.77×** √(Agk); diffuse 1.50 → 19.52 (×13.0), rate 0.75× — the sharp interface grows marginally faster, the expected sign (a 1.5-cell tanh ramp damps the mode) |
| single-phase regression (CUDA) | **+0.00 % on every case, identical iteration counts** |

**Finding 1 — a one-cell face-index shift between the two codes, and the conservation gate is the
only thing that sees it.** `flow` puts `u(i)` on the **low** (`−x`) face of cell `i`
(`projectCorrect`: `u(i) -= phi(i) - phi(i-sx)`; `divergOpen`: `d(i) = ox(i+sx)u(i+sx) - ox(i)u(i)`).
`WyAdvector` puts `uf(i)` on the **high** (`+x`) face (`wyFaceFlux` fluxes from `p` into `p+sd`;
the dilation term is `u(i) - u(i-sd)`). So the bridge must shift by one cell along each component's
own axis, `u_adv(i) = u_solver(i + s_d)`. The first implementation did not, and the failure is
nasty: it is invisible in a uniform flow, invisible in each axis' own discrete divergence (the
shifted x-difference is still the solver's exact zero, merely evaluated at cell `i−1`), and
invisible in `max|div(open·u)|` — but the advector sums the three axes AT ONE CELL, so it was
handed `div_x(i−1) + div_y(i−s_y) + div_z(i−s_z)`, three different cells whose sum the projection
never constrained. Weymouth–Yue adds `H(C−½)` times that to every full cell's budget. **Measured
with the shift omitted: the sheared ratio-10 sphere gained 35 % of its volume over 1000 steps while
the solver's own `max|div(open·u)|` read 7e-11.** Gate A exists because of this: it drives the
solver and a standalone `WyAdvector` with the same PHYSICAL field and compares C. Note a uniform
or solid-body-rotation scene would NOT have caught the axial half of the shift (neither field varies
along its own component's axis) — the LeVeque field does.

**Finding 2 — the acid test's velocity half cannot be gated at machine zero with a FREE interface,
and the reason is a loop gain, not a wiring inconsistency.** The gate as written asks for "steady
max|u| at machine zero". With the interface HELD FIXED the C → closure → ρ → projection chain
returns `2.1760599219479075e-17` at ratio 1000 — *bit for bit* the hand-set-ρ number on the same
grid, i.e. the chain is not merely consistent, it is the same computation. With the interface free
the residual instead wanders up to ~1e-12 over a few hundred steps. Mechanism, isolated by scaling
rather than asserted:

* the colour field is an extra degree of freedom; a residual velocity ε displaces it by ε·dt;
* that changes ρ by Δρ·ε·dt, hence the body force by g·Δρ·ε·dt, hence the acceleration in the LIGHT
  layer by g·Δρ·ε·dt/ρ_g — a loop gain **g·Δρ·dt/ρ_g**, which is 100 at ratio 1000, g = 0.1, dt = 1;
* the projection removes only the part of that perturbation that is a discrete gradient; what
  survives is an interfacial gravity-wave mode, undamped at μ = 0.

Measured, all starting from the exact frozen fixed point 2.176e-17, 200 steps:

| loop gain g·Δρ·dt/ρ_g | configuration | max\|u\| after 200 steps |
|---|---|---|
| 100 | ratio 1e3, g = 0.1, dt = 1 | 2.4e-12 |
| 10 | ratio 1e3, g = 0.1, dt = 0.1 | 7.9e-15 |
| 1 | ratio 1e3, g = 1e-3, dt = 1 | 1.2e-16 |
| 0.2 | ratio 3, g = 0.1, dt = 1 | 3.0e-14 |

i.e. the level tracks the gain and nothing else; with C frozen the same runs sit at 2.2e-17 forever,
bit-stable. The shipped gates are therefore: ∂P/∂z = −ρ_f·g at machine precision (this is the loud
half, and the harmonic-ρ_f ablation turns it into 0.34, so it does fail loudly), the frozen-interface
velocity **bitwise** against the hand-set-ρ reference, and the free-interface residual recorded and
bounded. Note this is NOT the WO-K defect: the acid test runs with momentum advection off, so there
is no mass-versus-momentum flux inconsistency to blame — it is the explicit buoyancy coupling of a
free interface.

**Finding 3 — the conservation gate as written (≤1e-13 over 1000 coupled steps) is a gate on the
PRESSURE SOLVER, and WO-E predicted that in advance.** WY adds `H(C−½)·div·dt/h` to every full
cell's budget, so once the advection is exact the conservation floor is the advecting field's own
discrete divergence residual (WO-E finding 2: "For V2 the relevant number is the projection's own
divergence residual — that, not h, sets the conservation floor"). Measured on the sheared ratio-10
sphere: `max|div(open·u)|` per step 1.4e-12…1.5e-11 and dV/V a **bounded, non-accumulating** random
walk with peak 5.6e-13 (−3.7e-13 at step 100, −5.6e-13 at 200, −2.9e-13 at 500, −2.4e-13 at 1000).
Tightening the Chebyshev tolerance to `rtol = 1e-14, maxit = 300` moves it to +1.7e-13 — i.e. it is
the driver's floor on this operator, not the tolerance. For contrast, V1's standalone floor with a
prescribed field at `max|div| ≈ 1.2e-15` was 5.7e-14 over 3200 steps. The shipped ctest gate is
therefore 1e-12 on the peak excursion, with `max|div|` printed next to it; reaching 1e-13 is an S-rung
(pressure-driver) item, not a VoF one.

**Finding 4 — a mixed-cell-only interface band is toothless on a perfectly sharp interface.** The
first `maxCourantInterface` took its band as "mixed cells and their face neighbours", literally as
specified. A grid-aligned sharp interface (…1,1,0,0…) has NO mixed cell, so the band was empty, the
`Kokkos::Max` reduction returned its identity `−inf`, and the limiter reported no constraint at all
on exactly the configuration the hydrostatic acid test starts from. The shipped predicate is a colour
DIFFERENCE — cell `i` is in the band if it is mixed or any face neighbour carries a different colour
— which coincides with the mixed-cell band for any resolved PLIC interface and covers the sharp
case; the reduction is also clamped at 0 so a uniform colour field reports 0, not `−inf`.

**Notes.** (a) With C ≡ const the solver reduces to the single-phase path **bitwise** in the two
senses that are attainable (D2, D3). A bitwise reduction to the CONSTANT-density solver is not
attainable and never was, independently of VoF: with `varRho_` on, the momentum RHS is a different
kernel (`buildRhsVar`) and the default pressure driver is Chebyshev rather than MG-PCG. That
pre-existing reduction is measured at 2e-14 by `test_vardensity_projection.cpp`. (b) **The cost,
recorded as the WO asks.** The trap flagged was that Chebyshev re-estimates its bounds on every
coefficient rebuild and a moving interface makes that every step. Measured on CUDA, 48³ periodic
sheared ratio-10 sphere, 30 steps, against the SAME coefficients held frozen (varRho, no VoF):

| | step | projection | predictor | momentum | pressure its/step |
|---|---|---|---|---|---|
| frozen ρ (varRho only) | 47.40 ms | 27.42 ms (57.8 %) | 0.11 ms | 19.67 ms | 9.0 |
| VoF, moving interface | 47.82 ms | 27.44 ms (57.4 %) | 0.11 ms | 19.67 ms | 9.0 |

So the whole geometric VoF stage — three PLIC reconstructions, three sweeps, the worklist scan and
the g=3 halo — is **0.42 ms of 47.8 ms (0.9 %)**, and the projection cost is **unchanged**: varRho
already rebuilt the coefficients and re-estimated the bounds every step, interface or not. The
flagged cost is therefore attributable to varRho, not to VoF, and S2 (bound amortization) keeps its
value without VoF having added to it. Not addressed here. (c) The
`walls-z` MPI configuration records a one-off colour displacement of 7.3016e-07, identical at
np = 1/2/4 to within 1.6e-14: the pressure driver's first solve on a fresh field leaves
max|u| ≈ 8.6e-6 (the documented varRho transient) and VoF faithfully advects it once. The test gates
that it is decomposition-independent. (d) `vof_twophase` runs 48 s on host-openmp and 119 s on CUDA;
`PECLET_VOF_TWOPHASE_LONG=1` extends the conservation gate from 200 to 1000 steps.

### WO-K (rung V2b) — momentum-consistent transport — **DONE 2026-08-31**, four findings + one escalation

Delivered: `src/vof/momentum_advect.hpp` (`peclet::flow::vof::MomentumConsistentAdvector` — the
half-shifted control volumes, the shared-flux driver, the recovery), `plicBoxVolume` in
`src/vof/plic.hpp` (the two-axis generalization of `plicSlabVolume`; single-axis and full-cell
restrictions bitwise equal to the existing entry points), the plane/flag accessors and
`checkCourant` in `src/vof/advect_wy.hpp` (additive — `advect()` is untouched), `enableVofMomentum`
+ `buildRhsVarMom` + the head-of-step call in `src/flow_ibm.hpp`,
`tests/kokkos/test_vof_momentum.cpp` (ctest `vof_momentum`),
`tests/kokkos_mpi/test_vof_momentum_mpi.cpp` (ctests `vof_momentum_mpi_np{1,2,4}`) and
`tests/study/vof_momentum_consistency.py` (the ratio sweep, the falling drop, the RT
near-Nyquist check). Python: `enable_vof_momentum / vof_momentum_enabled / vof_advected_velocity /
vof_momentum_diagnostics / set_vof_rho_floor / vof_rho_floor / set_vof_momentum_muscl /
set_vof_momentum_cell_flag / set_vof_flux_clamp`.

**The construction as built.** The momentum control volume of component `e` is indexed in `flow`'s
own **low-face** convention — `CV_e(i)` is centred on the `-e` face of cell `i`, where the solver's
`u_e(i)` lives. That is not cosmetic: it makes the owned control volumes exactly the advector
block's inner region, so `C^e` and the transported velocity are ordinary cell fields that
`WyAdvector::exchange` carries, and the bridge back to the solver is a plain `copyInner` with no
shift to get wrong (the `+e`-shifted alternative puts an owned value inside the halo's ghost band,
where the exchange overwrites it).

1. `C^e(i) = slab(cell i-s_e, e, [1/2,1]) + slab(cell i, e, [0,1/2])` — the PLIC polyhedra of the
   two overlapping pressure cells clipped into the shifted box. Rebuilt from the planes of `C^n`
   once per step. **No interpolation of C anywhere.**
2. Per sweep direction `d`, the flux slab is a slab of ONE pressure cell when `d == e` (the CV face
   sits at that cell's centre and `|a| < 1/2` keeps the slab in its upwind half) and the union of
   two HALF cells when `d != e` — two axes at once, which is why `plicBoxVolume` exists. The face
   Courant number is the `e`-average of the two staggered faces, which sums over the three sweeps to
   `1/2 (div_{i-s_e} + div_i)`: the dilation term vanishes with the projection's own divergence
   residual, the colour's conservation floor.
3. The momentum sweeps are **interleaved** with the colour sweeps, between `computeFluxes` and
   `applySweep`, and read that sweep's planes.
4. Recovery `u_e = (rho^e u_e)/max(rho^e, floor)`, implemented in the algebraically identical
   incremental form `u_new = u_old + dev/rho_new` (see finding 5).

**What is deliberately NOT changed.** The momentum time-term face density, the face body force and
the projection coefficient all keep the arithmetic face mean `1/2(rho(i)+rho(i-s_c))`. That
three-way agreement is what makes hydrostatic balance exact (`variable_density_projection.md` §1);
momentum consistency enters as the advective base velocity, `rho_f (u* - u^adv)/dt = -grad p +
visc + f`, which IS the conservative update divided through by the face density. Swapping the time
term to the clipped `rho^e` would make this rung break the WO-J acid test at `O(d rho)`.

**Gate results** (host-openmp unless noted; `vof_momentum` runs 42 s, `vof_momentum_mpi` 25/60/95 s
at np 1/2/4). **No pressure solve came within a factor 10 of its cap in any run reported here** —
the study battery records `max_iters/cap` and `max|div(open u)|` per run and marks a capped run
INVALID; the worst observed was 21/300 with `max|div(open u)| <= 2e-11`.

| gate | result |
|---|---|
| **K1 uniform-velocity identity, on the advection** — slab / tilted plane / sphere x ratios 1e1..1e4 | `max|u_adv - U| = 0` **BITWISE, 12/12**, no tolerance |
| K1 the same identity at np = 1/2/4 (walls-z, walled axis CUT) | **BITWISE 0 on every rank at every decomposition** |
| K2 `C^e` bounds, ratio 1e3, 20 coupled steps | `C^e in [-2.8e-17, 1.0]`, `min rho^e = 1.0` (= rho_g), floored CVs **0** |
| K2 ablation `set_vof_flux_clamp(False)` | `C^e in [-5.0e-3, 1.0131]`, `min rho^e = **-4.03**`, run **diverges at step 2** |
| K3 `C == const`: consistent advection vs density ratio 1 | ratio 1e2 and 1e4 both **bitwise 0.0** for `C == 1` and `C == 0` |
| K4 momentum conservation, periodic box, worst per-step `|d sum(rho^e u)|/|sum|` over 100 steps | **2.2e-13 / 2.7e-13 / 2.3e-13** (the projection's divergence floor) |
| K4 with the dilation coefficient NOT frozen (finding 3) | **1.4e-7** — six orders worse |
| K5 feature OFF, ablation knobs touched | `du = dp = dC = **0.0**` — the rung is inert when not enabled |
| MPI np=1 vs single-rank | `du = du_adv = dp = dC = **0.0**` bitwise, both configurations |
| MPI np=2/4 `walls-z` (walled axis cut) | du 8.7e-13, du_adv 5.3e-13, dp 5.0e-10 on `|P| = 1.46e+04` (rel 3e-14), dC 1.4e-14 |
| MPI np=2/4 `shear-per` | du 5.0e-13, du_adv 4.5e-13, dp 2.0e-13, dC 2.4e-13, dV/V +2.8e-12 |

**The uniform-velocity sweep, consistent vs inconsistent, 200 coupled steps, 32^3, mu = 0, g = 0.**
The left half is the shipped (default) build; the right half is the same code in a
`-DPECLET_FLOW_MREAL_DOUBLE` build — the existing A/B instrument, no source change — and it is the
half that answers the work order's question, for the reason in finding 6.

| scene | ratio | default build: consistent | default: inconsistent | MReal=double: consistent | MReal=double: inconsistent |
|---|---|---|---|---|---|
| tilted | 1e1 | 1.15e-07 | 8.07e-07 | **8.9e-16** | 1.6e-15 |
| tilted | 1e2 | 1.19e-07 | 8.53e-07 | **2.4e-15** | 3.1e-15 |
| tilted | 1e3 | 1.49e-07 | 8.88e-07 | **1.2e-15** | 2.4e-15 |
| tilted | 1e4 | 1.45e-07 | 9.45e-07 | **1.2e-15** | 2.4e-15 |
| droplet | 1e1 | 1.04e-07 | 4.28e-07 | **6.7e-16** | 1.1e-15 |
| droplet | 1e2 | 1.09e-07 | 4.97e-07 | **1.6e-15** | 1.8e-15 |
| droplet | 1e3 | 1.31e-07 | 4.93e-07 | **1.2e-15** | 1.6e-15 |
| droplet | 1e4 | 1.43e-07 | 5.66e-07 | **1.4e-15** | 1.7e-15 |

The consistent scheme is **flat in the density ratio** across four decades. So is the inconsistent
one — which is finding 1.

---

**Finding 1 — the uniform-velocity gate does not discriminate against the V2a baseline ON THIS
SOLVER, and the reason is structural. It is still the sharpest gate in the rung.**

The work order predicts the inconsistent scheme is `O(d rho)` wrong here. It is not: it is flat in
the ratio at 1e-15 too. The mechanism is not subtle once stated. `flow`'s momentum advection is in
**advective (non-conservative) form** — `rho_f * adv(u)` with `adv` the Koren/SOU flux operator —
and the discrete advection of a CONSTANT by a discretely divergence-free field is exactly zero,
whatever `rho_f` is. A uniform velocity is therefore a fixed point of the V2a momentum equation by
construction. The `O(d rho)` failure the test is designed to expose belongs to codes that advect
`rho u` CONSERVATIVELY with a mass flux inconsistent with the VoF flux (Rudman's setup): there the
two fluxes disagree and the disagreement is weighted by `d rho`.

This is the fourth time in this campaign a gate as written measured a different quantity than
intended, and as before the right response is to say so rather than to weaken or to oversell it.
Two things follow, and the second matters more than the first:

* **the contrast this rung needs must come from a case where the momentum actually transports** —
  the falling drop and Rayleigh-Taylor below, not the uniform test;
* **the uniform test is nevertheless the gate that built this rung.** It is decisive for the
  CONSISTENT scheme, because that scheme *is* conservative, and it caught three real defects in
  succession, each at a different order of magnitude and each invisible to the other gates:
  `1.5e+06` (the rho^e floor firing on a colour that had left [0,1] — finding 2), `1.2e-13`
  (floating-point conditioning — finding 5) and `2.2e-10` (the MUSCL amplifier — finding 4). A gate
  that goes from 1e6 to bitwise zero as three separate mechanisms are removed is doing exactly the
  job the work order wanted it to do; it just is not a comparison against V2a.

**Finding 2 — the geometric flux is bounded by what the CURRENT cell planes see in the donor, not
by the ADVECTED `C^e`, and at high ratio that gap is fatal rather than cosmetic.**

`WyAdvector` re-reconstructs before every sweep, so from sweep 2 on the planes describe the updated
cell colour while `C^e` is the independently advected shifted colour. `F <= clip(donor CV)` still
holds; `F <= C^e(donor)` — the hypothesis of Weymouth's boundedness proof (thesis Appendix A) — does
not. The gap is `O(a^2)` and does not accumulate; measured on a tilted plane after one step at CFL
0.20 / 0.10 / 0.05 / 0.025, `C^e` reached `-2.56e-2 / -6.0e-3 / -1.0e-3 / -3.1e-5` and stayed there
over 10 steps. In colour units that is a wisp. In density units at ratio 1e4 it is
`rho^e = 1 + 1e4 * (-2.56e-2) = -255`, and the recovery divides by it: the first working version of
this rung returned `max|u_adv - U| = 1.5e+06` at ratio 1e2 and `2.6e+08` at 1e4 while ratio 10 was
clean at 5.6e-16 — i.e. it looked exactly like an `O(d rho)` consistency failure and was not one.

The shipped flux is clamped into Weymouth's own interval `max(0, |a| - (1 - C^e_don)) <= |F| <=
min(|a|, C^e_don)`, evaluated on the donor control volume's own colour (which is why `C^e` is
exchanged before every sweep). The clamp is free with respect to everything else the rung needs: it
is applied to the ONE face value both neighbours share, so conservation still telescopes bit
exactly, and BEFORE the momentum flux is formed, so the two updates keep sharing one `F` and the
consistency identity is untouched. For `C^e_don` exactly 0 or 1 the interval collapses to the
algebraic value, so full and empty control volumes stay exactly stationary. `set_vof_flux_clamp(False)`
keeps this a measured number: `rho^e = -4.03` and divergence at step 2, gated in `vof_momentum` K2.

**Finding 3 — the dilation COEFFICIENT `rho^ u` must be frozen across the three sweeps, exactly as
Weymouth-Yue freeze `H(C^n - 1/2)`, and the damage from not doing so is the same shape as WO-E's.**

The momentum dilation term is `rho^_i u_i (a_+ - a_-)` and its sum over the three sweeps is
`rho^_i u_i * div`, which vanishes for a projected field — **but only if the coefficient factors out
of the sum**, i.e. only if it is a constant of the step. With the running (partially swept) velocity
in that slot the three sweeps carry three different `u_i` and the cancellation is lost.
**Measured, 100 coupled steps, ratio 1e3: worst per-step momentum drift 1.4e-7 relative with the
running velocity, 2.2e-13 with `u^n` frozen — a factor 6e5**, while the colour, whose flag IS frozen,
was exact throughout. This is WO-E's #1 trap wearing a different hat (there: `2.3e-15 -> 1.5e-2`
from recomputing the flag), and it is worth stating as the general rule: *in a Weymouth-Yue sweep,
every coefficient that multiplies the divergence must be frozen at the start of the step.* Note the
flux velocity is deliberately NOT frozen — only the dilation coefficient is; the two roles are
different and conflating them costs either conservation (freeze neither) or the splitting order
(freeze both, see finding 5's closing note).

**Finding 4 — a MUSCL slope in the momentum flux is a density-ratio amplifier on any control volume
a sweep empties; plain donor-cell upwind is the shipped default.**

The velocity update is `u_new = u_old + [rho_g (a_- dv_- - a_+ dv_+) + drho (F_- dv_- - F_+ dv_+)]
/ rho_new` with `dv = u^ - u_old`. When the control volume is itself the donor of its outgoing face,
plain upwind gives `u^_+ = u_old` and `dv_+ = 0` **exactly**. A limited slope makes
`dv_+ = (1/2 - |a|/2) * slope`, whose coefficient `drho F_+ / rho_new` is unbounded in the ratio
precisely when the sweep empties the volume (`F_+ -> C^e`, `rho_new -> rho_g`) — physically, the
residual velocity of a volume that has just lost 99.95 % of its mass is a difference of two
liquid-scale momenta. Measured on the uniform-velocity gate at ratio 1e4, 50 steps, in the
double-storage build so the solver's own floor is out of the picture:

| steps | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| MinMod slope | 3.3e-16 | 6.1e-16 | 4.3e-14 | 5.1e-13 | **2.2e-10** |
| donor-cell upwind | 3.3e-16 | 4.4e-16 | 4.4e-16 | 5.6e-16 | **6.7e-16** |

At ratio 1e3 the slope is harmless (6.1e-15 at 50 steps), so this is a high-ratio robustness
verdict, not an accuracy one. `set_vof_momentum_muscl(True)` keeps it available with the caveat in
its docstring; re-run the ratio sweep if you turn it on.

**Finding 5 — the consistency identity is exact in EXACT arithmetic for any correct implementation,
and that is not enough: two separate floating-point conditioning defects each degraded the gate
linearly in the density ratio, i.e. each wore the exact signature of the defect this rung removes.**

Both were found by the gate and both are now designed out rather than tolerated.

* *Storing `rho^e u_e`.* The increment is a sum of `rho_l`-scaled flux terms whose result is only
  `rho_g`-scaled in a gas control volume, and a volume that one directional sweep fills and the next
  empties passes through an intermediate `rho^e` a factor `ratio` larger than where it starts and
  ends. Measured: `6.7e-16` at ratio 1e2 rising to `3.2e-14` at 1e4. The fix is to evolve the
  algebraically identical `u_new = u_old + dev/rho_new`, in which every term is a DIFFERENCE OF
  VELOCITIES and therefore exactly zero for a uniform field. It remains the conservative scheme —
  `rho_new` is the advected colour's own density — only the arithmetic is conditioned.
* *Fluxing the frozen `u^n` instead of the current velocity.* The term
  `drho dC (u^n_i - u_old_i)/rho_new` that then survives has gain `ratio` on a volume a sweep
  empties, which amplified the solver's own 1e-7 noise to 1e-7 at ratio 1e4 while ratio 1e3 stayed
  clean. Transporting the CURRENT state removes the term identically AND is the faithful directional
  split; it costs three more halo exchanges per sweep, alongside the three `C^e` already needed by
  finding 2's clamp.

The lesson to carry: for a gate whose whole value is that it is pass/fail at 1e-15, "algebraically
exact" is not the property to verify — *arrange the arithmetic so the cancellation is exact*, then
the gate has no tolerance to argue about. All three of this rung's exactness statements (K1, K3, K5)
are now bitwise rather than tolerance-bounded.

**Finding 6 — ESCALATED: the uniform-velocity gate cannot reach 1e-15 through the full step in the
shipped build, and the floor is the solver's FLOAT momentum-operator storage, not this rung.**

Through the coupled step the residual sits at `1.2e-7`, flat in ratio, for the consistent scheme and
`5e-7 .. 9e-7` for the inconsistent one. That is float epsilon, and the mechanism is direct:
`buildRhsVar` / `buildRhsVarMom` form `rho_f/dt * u` in **double**, while `ibmBuildDiffusionVar`
stores the same `rho_f/dt` in the **float** velocity stencil (`Solver::FV`), so `u* = b/diag`
differs from its input at float epsilon **whenever rho varies in space**. It is identically zero for
uniform rho (measured: exactly 0.0, 0 pressure iterations) and it is present with momentum
consistency OFF, so it is neither caused by nor curable within this rung.

Confirmed by A/B with the existing `-DPECLET_FLOW_MREAL_DOUBLE` instrument in a separate build
directory (no source change, shipped default untouched): the identical measurement returns
**1.2e-15, flat across ratios 1e1..1e4** on both scenes. The pressure solve is NOT implicated —
`max|div(open u)| = 3.3e-16` and 13..21 iterations against a cap of 300 in the double build, and
`<= 2e-11` with 13..21 iterations in the float build; no solve in this rung came near its cap.

Two consequences, filed rather than fixed here (the storage precision is shared solver code and
belongs to whoever owns the `MReal` work, who has independently root-caused the same float storage
as a high-contrast pressure-operator failure):
- the shipped `vof_momentum` ctest gates the identity on the ADVECTION (bitwise, tolerance-free) and
  records the coupled-loop number at 1e-4 with the mechanism named in the source, rather than
  pretending 1e-7 is a VoF result;
- at ratio 1e4 in the float build the 1e-7 injection is enough to destabilise a long two-phase run.
  This rung is therefore honestly rated to **ratio ~1e3** on the shipped build and to 1e4 on a
  double-storage build. Say so in the API docs rather than quoting the double-build number.

**Finding 7 — the Arrufat raindrop gate cannot be run before V4, and the substitute is stated
rather than smuggled in.** The specified case (ratio 831.8, 15 cells/diameter, terminal velocity
within 15 %) is a drop held together by SURFACE TENSION, which is rung V4. Without it a raindrop at
its terminal Weber number flattens and breaks up, and there is no terminal velocity to measure —
the gate as written is not a property of this rung's code. The surface-tension-free case with the
same payoff structure is a VISCOUS drop in the Stokes regime, whose terminal velocity is
Hadamard-Rybczynski; that is what `gate_falling_drop` runs, at the same ratio and the same 15
cells/diameter, with the periodic-array caveat stated. Numbers below.

