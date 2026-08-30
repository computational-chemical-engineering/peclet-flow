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

**Why this outranks new physics.** Every rung of this campaign is gated on "np 2/4 bitwise vs
np 1". WO-I established that four MPI tests — three of them **pre-existing and unrelated to any
of our work** (`varmu_mpi_np4`, `bodyforce_ghost_mpi_np4`, `ghost_projection_mpi_np4`) — abort
with `MPI_ERR_TRUNCATE` at np=4, that it **reproduces on a pristine tree**, and that
`PECLET_FLOW_CA=0` cures all of them. A gate that fails intermittently for reasons unrelated to
the code under test is not a gate. Fix this before trusting any further np=4 result.

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
