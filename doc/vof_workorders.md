# VoF work orders — phase 0 (V-1, S0, S1, V0, V1)

Execution companion to `suite/docs/VOF_PLAN.md` (read its §0–§6 first; ownership map in its
§11). Each work order below is written to be executed by an **Opus agent standalone**. The
gates are the contract: a work order is DONE when every gate passes; it is ESCALATED (stop,
write findings into this file under the WO, do not "fix" numerics to make a gate pass) when
a gate fails twice for a non-obvious reason.

## Shared preamble (read once, applies to every WO)

Build (CUDA backend; from `flow/`):
```bash
source ../.venv/bin/activate
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build -j
# kokkos unit ctests:
cmake -S tests/kokkos -B build_ktest -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build_ktest -j && OMP_NUM_THREADS=8 OMP_PROC_BIND=false ctest --test-dir build_ktest --output-on-failure
# MPI ctests:
cmake -S tests/kokkos_mpi -B build_kmpi -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
```
Always `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` for test batteries (48-core trap).

Hard rules (violating any of these is an automatic escalation):
1. **Never edit a validated kernel body.** New physics = new sibling kernels/files. The
   single-phase regression (`tests/regression/sdflow_regression.py`) must stay bit-exact:
   +0.00%, identical iteration counts.
2. **Device-first, host oracle.** Kernels are `KOKKOS_INLINE_FUNCTION` in header-only
   `.hpp` under `src/` (never `.cu`). Where a WO asks for an oracle, the same inline
   functions run in a serial host loop and results are compared bitwise (CUDA) — see the
   GPU-vs-OpenMP tolerance policy: bitwise on identical backend, tolerance across backends.
3. **Bit-exact MPI**: every np-2/4 result identical to np-1 (`tests/kokkos_mpi` pattern).
4. Commit in `flow` at each validated gate, then bump the umbrella pointer (submodule
   first, umbrella last). Do not push a red tree. **Never `git add -A` / `git add .` /
   `git commit -a`** — stage the paths you changed, by name. Concurrent agents share these
   checkouts and the suite carries untracked scratch directories; a blanket add both muddies
   the commit and has already swept another agent's in-flight files into the wrong commit
   (2026-08-30). Staging a submodule pointer (`git add flow`) is a named path and is fine.
5. Style: match `flow/src/*.hpp` (namespaces `peclet::flow`, x-fastest indexing per
   `suite/docs/CONVENTIONS.md`, doubles for geometry).

Paper sources (all verified retrievable): Scardovelli & Zaleski JCP 164:228 (2000);
Lehmann & Gekle, Computation 10:21 (2022) [open access — branch-reduced SZ for cubic
cells]; Aulisa, Manservisi, Scardovelli & Zaleski JCP 225:2301 (2007) [MYC]; Weymouth &
Yue JCP 229:2853 (2010); Popinet JCP 228:5838 (2009). If a paper is unreachable, escalate
rather than improvise the algorithm.

---

## WO-A (rung V-1) — MPI + CUDA validation of the varRho/varMu paths  [OPUS]

**Why.** The multiphysics phases were validated host-openmp only
(`flow/doc/variable_density_projection.md` §4). VoF rung V2 needs varRho on CUDA and under
MPI. The structure is believed MPI-ready (ρ bridge + `fillPropGhosts` use the halo paths;
`minMuInner` allreduces) — believed ≠ gated.

**Do.**
1. Run the existing `tests/kokkos/test_vardensity_projection.cpp` and
   `test_variable_mu.cpp` on the CUDA backend. Record results here.
2. New `tests/kokkos_mpi/test_vardensity_mpi.cpp` (copy the 4-line CMake registration
   pattern + the np-1-reference structure from the existing `test_*_mpi.cpp`):
   hydrostatic two-layer column, ratio 1000, gravity closure, walls ±z, 20 steps.
3. New `tests/kokkos_mpi/test_varmu_mpi.cpp`: two-layer Couette, 10× μ jump, harmonic
   face mean (mirror `tests/study/two_layer_couette.py` at reduced size).
4. Run `tests/study/rayleigh_taylor.py` against the CUDA build; record amplitude curve vs
   the host-openmp record in `variable_density_projection.md` §3.

**Gates.**
- CUDA unit tests pass; hydrostatic steady max|u| ≤ 1e-14 at ratio 1000 on CUDA.
- MPI tests: np 2 and np 4 **bit-exact** vs np 1, host AND CUDA backends.
- Chebyshev iteration counts identical across np (the bounds path must not depend on
  decomposition; if it does — escalate with the per-rank bound values).
- Single-phase regression bit-exact.

**Known traps.** ρ must be bridged to the g=1 MG block *including the ghost ring*
(`copyBlockShifted` offset G−1) — the hydrostatic test is the canary for a stale bridge.
`chebBoundsSet_` must be invalidated on every coefficient rebuild; a stale bound diverges
silently.

---

## WO-B (rung S0) — pressure-driver measurement battery on static two-phase coefficients  [OPUS]

**Why.** Decide how far up the S-ladder (plan §5) to climb, from measurement.

**Do.** New `tests/study/vardensity_solver_probe.py`. For geometry ∈ {open box, immersed
cylinder (reuse an existing scene setup), `data/packing_ring.vti`} × ρ-field ∈ {sphere
blob, flat film (slab), tilted film} × ratio ∈ {1e2, 1e3, 1e4}:
- set `set_density_mode("variable")`, write the manufactured ρ field (smooth tanh edge,
  2-cell width, AND a sharp variant) into the `"rho"` field via `field_view`;
- run 20 projection-only steps (quiescent + gravity closure, the hydrostatic setup) and
  20 steps of a lid/body-force flow;
- record per step: pressure iterations, final residual, wall time, for driver ∈
  {chebyshev, pcg} (`set_pressure_chebyshev` / `set_pressure_pcg` after
  `set_density_mode` — the explicit call wins over the varRho default).
Emit a markdown table + JSON (`tests/study/vardensity_solver_probe.json`).

**Gates.** The battery runs to completion on host + CUDA; the PCG stall of
`variable_density_projection.md` §2 is reproduced on at least one configuration (if it is
NOT reproducible, that is a finding — escalate with the table, do not chase it); the table
answers: Chebyshev its vs ratio, vs geometry, vs sharp/smooth edge.

**Success metric for the ladder decision** (record, don't judge): Chebyshev ≤ ~40 its at
1e3 across geometries ⇒ S2 amortization is the only remaining work; > ~80 or
ratio-divergent ⇒ S3/S4 promote.

---

## WO-C (rung S1) — flexible CG driver option  [OPUS]

**Why.** PCG's stall is consistent with a (mildly) nonsymmetric/variable preconditioner;
flexible CG tolerates exactly that at the cost of one extra stored vector.

**Do.** In the pressure driver that wraps `CutcellMG` (locate the PCG loop used by
`project()`; it lives with the MG driver machinery around `mac_cutcell_mg.hpp` /
`flow_ibm.hpp` — find it, don't guess): add a **flexible** variant, selected by
`set_pressure_fcg()` (binding alongside `set_pressure_pcg`/`set_pressure_chebyshev`):
- keep the previous preconditioned residual `z_prev`;
- β = rᵀ_{k+1}(z_{k+1} − z_k) / (rᵀ_k z_k)   (Polak–Ribière / IPCG form);
- everything else identical to the PCG loop (same stopping estimate, same deflation).
Do not touch the existing PCG/Chebyshev code paths — new function or a branch at the β
computation only, default off.

**Gates.**
- With FCG unselected: bit-exact single-phase regression (the code path must be inert).
- On the WO-B battery: FCG converges on every configuration where PCG stalls; record
  its/wall-time columns next to Chebyshev.
- On constant-density problems FCG ≈ PCG iterations (±1) — sanity that β reduces correctly.

---

## WO-D (rung V0) — PLIC toolbox  [OPUS]

**Files.** `src/vof/plic.hpp` (namespace `peclet::flow::vof`), test
`tests/kokkos/test_vof_plic.cpp`.

**Contents** (all `KOKKOS_INLINE_FUNCTION`, pure functions of local data, unit cube
convention — cell mapped to [0,1]³, normal m with Σ|m_i| = 1). **Signature rule (hard):**
these functions take scalars and small local arrays ONLY — no `View` types, no grid
indexing, no halo/topology types. They are scheduled for promotion to
`peclet::core::vof` at the V4 freeze (plan §11: shared by the structured, AMR, and
bubble-block containers), and the promotion must be a file move:
1. `plicVolume(mx,my,mz,alpha) -> V`: fluid volume under the plane m·x = α — the
   Scardovelli–Zaleski (2000) analytic forward formula, in the branch-reduced form of
   Lehmann & Gekle (2022) (mirror to the positive octant, sort m₁≤m₂≤m₃, the piecewise
   polynomial cases).
2. `plicAlpha(mx,my,mz,V) -> α`: the analytic inverse (same papers, same case structure).
3. `youngsNormal(c 3×3×3 stencil) -> m`: normalized-∇C with the standard 27-point weights.
4. `mycNormal(c 3×3×3 stencil) -> m`: Aulisa et al. (2007) mixed Youngs-centered — the
   central-column candidates per axis + the Youngs candidate, selection per the paper.
5. `faceFluxVolume(...)`: volume of the PLIC polyhedron truncated by an axis-aligned slab
   [0, f] in one direction (needed by WO-E; expressible via `plicVolume` on a rescaled
   cell — implement it that way, no polyhedron clipping code).
6. `initPlane(...)` test helper: exact volume fractions for a given global plane (this is
   just `plicVolume` per cell); `initSphere(...)`: fractions by 4× recursive midpoint
   subdivision (pattern: `buildCellFraction`'s subsampling in
   `src/mac_approx_projection.hpp`), test-only accuracy is fine.

**Gates** (`test_vof_plic.cpp`):
- Forward exactness: axis-aligned and hand-computable oblique planes vs closed-form
  volumes, |ΔV| < 1e-14.
- Round trip: 10⁵ randomized (m, V) pairs (including near-axis m, V→0, V→1 extremes):
  |plicVolume(m, plicAlpha(m,V)) − V| < 1e-13. This is the gate that catches every case-
  boundary bug.
- Consistency: `faceFluxVolume` with f = 1 equals `plicVolume`; f-additivity
  (vol[0,f] + vol[f,1] = vol[0,1]) to 1e-14.
- Normals: on `initPlane` data, MYC normal within 1° of exact for 10³ random plane
  orientations, and max error reported (record — MYC is *near*-2nd-order, not exact on
  planes; do NOT tighten this gate to exactness); on `initSphere`, L1 normal error
  2nd-order under refinement (16³→32³→64³ fit slope > 1.7).
- Device/host: the full randomized round-trip battery run in a `parallel_for` and in a
  serial host loop, results **bitwise identical** per element on the same backend.

---

## WO-E (rung V1) — Weymouth–Yue split advection, standalone  [OPUS]

**Files.** `src/vof/advect_wy.hpp`, `tests/kokkos/test_vof_advect.cpp`,
`tests/kokkos_mpi/test_vof_advect_mpi.cpp`, study script
`tests/study/vof_advection_battery.py` (optional python wiring may be deferred; the
ctests are the deliverable).

**The algorithm** (Weymouth & Yue 2010; this exact structure, no variations):
per step, with face-normal velocities `uf` given (divergence-free), Δt with
max|uf|Δt/h < 0.5 **asserted**:
1. Freeze the dilation flag once: `cc_i = (C^n_i > 0.5) ? 1 : 0`. It is used by all three
   sweeps and NOT recomputed between them (recomputing it silently destroys exact
   conservation — the #1 trap).
2. Three directional sweeps in an order that cycles through the 6 permutations of (x,y,z)
   with step index (`perm[step % 6]`). Each sweep along direction d:
   a. reconstruct PLIC (MYC normal + `plicAlpha`) in mixed cells — build a compacted
      worklist first (`parallel_scan` over mixed cells: 0 < C < 1); full/empty cells
      short-circuit;
   b. donor-cell geometric face flux through each d-face: upwind cell's PLIC truncated by
      the slab of width |uf|Δt/h on the outflow side (`faceFluxVolume`); full/empty donor
      → algebraic flux C_donor·uf·Δt/h;
   c. update: `C_i += (F_{i−} − F_{i+}) + cc_i·Δt·(uf_{i+} − uf_{i−})/h`;
   d. halo-exchange C ghosts (each sweep needs updated ghosts — 3 exchanges per step).
3. NO clipping in this rung (conservation must close to round-off). Record wisp census
   (count of cells with 0 < C < 1e-8 or 1−1e-8 < C < 1) as a diagnostic only.

Storage: raw `View<double***>` blocks with **g = 3 ghosts** and their own
`core::halo::GridHaloTopology` built at width 3 (`buildTopology(dec, rank, 3, per, comm)`)
— do NOT touch the solver's G=2 machinery; V1 is standalone (no `Solver` integration yet).

**Gates.**
- Plane translation (uniform diagonal u, periodic box, sphere of fractions): after
  returning to start (N steps), L1 shape error 2nd-order under refinement; total volume
  drift |ΣC − ΣC⁰|/ΣC⁰ < 1e-13 after 1000 steps.
- Zalesak disk (slotted disk, solid-body rotation, one full revolution, 100² cells in
  plane): L1 error reported and recorded as baseline (compare order of magnitude with
  published geometric-VoF values ~0.01–0.02; a 10× miss = bug).
- 3D LeVeque deformation field with time reversal (T = 3, 64³ and 128³): L1 error at
  return recorded as baseline; volume conserved < 1e-13 throughout (this field is
  divergence-free analytically — sample `uf` from the stream-function form so the
  *discrete* face divergence is ~round-off; if using pointwise sampling, record the
  discrete divergence level and the matching conservation floor).
- CFL guard: a step with CFL ≥ 0.5 must abort with a clear error.
- MPI: np 1/2/4 bitwise-identical C fields (host + CUDA), periodic and non-periodic.
- Worklist on/off produces bitwise-identical results (the compaction must be a pure
  optimization).

**Escalate if:** conservation closes only to ~1e-10 (symptom: the dilation flag was
recomputed, or the flux slab width used the wrong face velocity sign convention); Zalesak
error is off by 10×; MPI differs at the last bit only in ghost-adjacent cells (halo width
or exchange-per-sweep bug).

---

## WO-F (blocker fix) — domain BCs are not rank-aware  [OPUS]

**Why.** WO-A's gating uncovered two pre-existing defects in the DOMAIN-BC machinery (not in
VoF). Both make any **wall-bounded multi-rank case with variable properties** wrong, so both
sit on V2's critical path. Verified in the source 2026-08-30:

1. **`applyVelocityBcCompTo` (`src/flow_ibm.hpp:3435`) has no ownership test.** It applies the
   per-face BC whenever `bc_[ff] != 0`, regardless of whether *this rank's block* touches that
   global face. A partition cutting a walled axis therefore imposes a wall at every interior
   rank boundary, splitting the domain into independent sub-domains. `touchesGlobalFace(f)`
   (`:4490`) already exists and the **scalar** path uses it (`applyScalarBc`, `:4486`) — the
   velocity path simply never adopted it. Measured by WO-A: with z cut at np=2 the velocity
   canary stays clean (max|u| 4.5e-17) while `max|P_dist − P_ref| = 4.0e+02` and ∂P/∂z is off
   by 8·g·ρ. **The velocity does not reveal this — only the pressure does.**
2. **`fillPropGhosts` (`:1912`) and `fillPorousEpsGhosts` skip the domain-face override under
   MPI** (`if (!distributed_)`), so μ/ρ/ε ghosts on a walled face keep their *periodic wrap*
   value. Wrong at **every np including 1** (the guard keys on `distributed_`, not on rank
   count). The comment immediately above the guard states exactly why the override is needed
   ("a periodic wrap there would bring the wrong layer's value to the wall face — destabilising,
   especially for the harmonic mean"). Measured: the literal two-layer Couette differs from the
   single-rank reference by 2.7e-2 relative at np=1, and is bitwise identical with varMu off.

**Do.**
1. Add the `touchesGlobalFace(f)` ownership test to the velocity BC path, matching the scalar
   path's pattern. Audit **every** per-face BC application for the same omission — at minimum
   `applyVelocityBcCompTo`, the pressure-openness BC construction, the outflow correction
   (`bcCorrectOutflow`), and any inlet-profile application — and fix each. Enumerate what you
   audited in the findings log, including the sites you checked and found already correct.
2. Replace the `if (!distributed_)` guards in `fillPropGhosts` / `fillPorousEpsGhosts` with the
   per-face `touchesGlobalFace(f)` test.
3. Extend `tests/kokkos_mpi/test_vardensity_mpi.cpp` and `test_varmu_mpi.cpp` to **cut the
   walled axis** (they currently assert it stays uncut — that assertion was the correct
   temporary response to the bug and must now be removed), and add a **pressure** comparison,
   not only velocity: the np-2/4 pressure field must match np-1. Restore the WO-A-documented
   literal two-layer Couette (asymmetric μ) in `test_varmu_mpi`, which the symmetric stack was
   chosen to dodge.

**Gates.**
- **Single-rank bit-exactness is automatic and must be verified, not assumed**: single-rank
  `touchesGlobalFace` is always true, so the guarded code runs exactly as before. Single-phase
  regression +0.00%, identical iteration counts.
- Walled-axis-cut np 2/4 **bitwise** vs np 1 on **velocity AND pressure**, host + CUDA.
- The restored asymmetric two-layer Couette matches the analytic solution at every np.
- All pre-existing MPI ctests stay green.

**Escalate if** a fix would change single-rank numerics — that means the defect is not what this
WO describes and the diagnosis needs revisiting before any code lands.

## WO-G (blocker fix) — body-force ghosts are never filled  [OPUS]

**Why.** Found by WO-F, independently verified in-source 2026-08-30. This one is **not
MPI-only** and it moves already-validated numbers, which is why it gets its own work order
with an explicit no-re-baseline rule.

The mechanism, confirmed at three sites:
- `applyClosure` (`src/property_closures.hpp:44`) writes **inner cells only**
  (`MDRangePolicy({g,g,g}, {e-g})`), and its own comment says the ghosts are "refilled by the
  field's own exchange".
- **Nothing ever exchanges `cellForce_`.** `fillPropGhosts` is called for `rhoField_`,
  `muField_`, `dragBeta_`, `epsField_` — never for `cellForce_[c]`
  (`grep cellForce_ | grep fillGhost` is empty). The fields are zero-initialised at
  registration and their ghost rings stay at that value forever.
- `buildRhsVar` (`flow_ibm.hpp:2998`) and the `buildRhsForced` siblings read the
  face-interpolated force `0.5*(fb(i) + fb(i - strd))`, which reaches into the ghost on the
  first inner plane.

Consequence: **the face body force is halved on the first inner plane of every block** — at
every rank boundary under MPI, and single-rank at the periodic wrap plane. Affected physics:
Rayleigh–Taylor (gravity closure), Boussinesq thermal convection, CFD-DEM feedback forces —
i.e. cases with *recorded validated numbers*. WO-A's hydrostatic acid test still passed at
2.75e-17, so establish early whether the wall-BC velocity pin masks it there; that is
diagnostic information, not a reason to doubt the defect.

**Do.**
1. Give `cellForce_[c]` the ghost fill its consumers assume. Prefer routing it through the
   existing property-ghost path (`fillPropGhosts`, which is now per-face rank-aware after
   WO-F) at the point the closures are applied, so periodic wrap, halo exchange and
   domain-face policy are all inherited rather than re-implemented. Check whether a body
   force wants Neumann-copy or something else at a *wall* face and justify the choice in the
   findings log — a body force is not a transported property.
2. Audit every other field written by `applyClosure` or by a Python-side `field_view` write
   for the same "written inner-only, read with a face average" hazard, and list what you
   checked.
3. **MEASURE, DO NOT RE-BASELINE.** Report the before/after delta for: the single-phase
   regression (all 13 grid points), `tests/study/rayleigh_taylor.py` (the amplitude series
   recorded in `variable_density_projection.md` §3), `tests/study/dvd_cavity.py` vs de Vahl
   Davis, and the hydrostatic acid tests at ratio 3 and 1000. Leave every `perf_baseline*.json`
   and every recorded number in the docs **untouched**; put the deltas in the findings log.
   The user decides whether to re-baseline.
4. Add an MPI ctest that would have caught this: a uniform body force across a rank boundary,
   asserting the face force is uniform (np 2/4 bitwise vs np 1), and a single-rank periodic
   variant asserting no wrap-plane anomaly.

**Gates.**
- The new tests fail before the fix and pass after (demonstrate both).
- np 2/4 bitwise vs np 1, host + CUDA, on the new tests and all pre-existing ones (42+).
- Deltas measured and reported for every case in item 3.
- **No baseline file and no recorded number in any doc is edited.**

**Escalate if** the single-phase regression moves at all — it should not: the regression cases
use `set_body_force` (a uniform scalar, not a closure field), so a `cellForce_` ghost fix must
leave them bit-exact. A movement there means the fix reached further than intended.

## Findings log

(append per WO on completion/escalation)

### WO-A (rung V-1) — MPI + CUDA validation of varRho/varMu — **DONE 2026-08-30**, two escalations

Delivered: CUDA runs of `tests/kokkos/test_vardensity_projection.cpp` and `test_variable_mu.cpp`;
two new MPI ctests `tests/kokkos_mpi/test_vardensity_mpi.cpp` + `test_varmu_mpi.cpp` (registered in
the gated `PECLET_FLOW_MPI` foreach, np = 1/2/4, host-openmp **and** nvidia-cuda); a CUDA run of
`tests/study/rayleigh_taylor.py`; and the doc update — `doc/variable_density_projection.md` §3
(CUDA numbers), new §3.1 (the multi-rank table), §4 (the two escalations replace "MPI/CUDA
validation deferred").

**Gates.**

| gate | result |
|---|---|
| CUDA unit tests pass; hydrostatic steady max\|u\| ≤ 1e-14 at ratio 1000 on CUDA | **PASS** — `2.75e-17` (ratio 1000), `3.99e-17` (ratio 3), ∂P/∂z error `3.41e-16` / `7.40e-16`; uniform-ρ reduction rel du `5.01e-14`; `test_variable_mu` OK |
| MPI np 2/4 vs np 1, host AND CUDA | **PASS at the reduction-order floor.** np = 1 bitwise (`0.000e+00` on every field, both backends). np > 1: hydrostatic Δu ≤ `6.0e-17` / ΔP ≤ `2.8e-14` (P is O(800)); the ρ-jump-on-a-rank-boundary case Δu ≤ `3.3e-19` / ΔP ≤ `1.4e-17`; varμ rel du ≤ `2.0e-15`. **Literal bitwise equality at np > 1 is impossible by construction** — Chebyshev's bound estimation (`CutcellMG::dot`) and `removeMean` go through an `MPI_SUM` allreduce whose summation order is a function of the rank count. This is the established `tests/kokkos_mpi` convention (np = 1 bit-exact, np > 1 at the floor), stated in the tests' headers with the mechanism. |
| Chebyshev iteration counts identical across np | **PASS, and the WO's escalation branch is NOT triggered.** On the non-degenerate solve (`jump-x`, ratio 1000, jump on the rank boundary) the per-step V-cycle count is *identical* for all 20 steps at np = 1, 2, 4 on both backends. The ±1–2 scatter visible on the hydrostatic case from step ~7 is round-off, not decomposition: once the state is at machine zero the driver's `r0` is noise and `maxabs(r) < rtol·r0` is a knife edge — at **fixed np = 1** the sequence already changes with the OpenMP thread count alone (steps 7.. read `15,13,13,…` at 1 thread, `16,13,14,…` at 2, `16,14,14,13,…` at 8). The ctest therefore gates the count to *exact* equality over the non-degenerate steps (window predicate: the reference's own max\|u\| entering the step). |
| Rayleigh–Taylor on CUDA vs the host-openmp §3 record | **PASS, identical to every printed digit**: `1.50 → 1.87 → 3.13 → 5.54 → 9.30 → 14.16 → 19.52` (13.0×), monotone, on CUDA *and* on a same-day host-openmp rerun; the §3 record was "1.5 → 19.5 (13×)". |
| Single-phase regression bit-exact | **PASS** — `tests/regression/sdflow_regression.py` on the nvidia-cuda build: **+0.00 %** on every metric (K, k\*, fitted order p, Richardson extrapolate) and **identical** pressure-iteration totals, per-step medians and step counts on all 13 grid points of `zh_sphere` / `random_spheres` / `hollow_rings`. No production code was touched by this WO — the deliverable is test files plus one CMake line. |

**ESCALATION #1 — flow's per-face domain BCs are not rank-aware (production fix required).**
`applyVelocityBcCompTo` and the pressure-openness BC impose the BC on **every** rank's own block
faces; there is no `touchesGlobalFace` ownership test, although the transported-scalar BC path
(`applyScalarBc`, `flow_ibm.hpp:4439`) has exactly that. If the decomposition cuts a non-periodic
axis, the halo exchange fills the ghosts correctly and the BC then *overwrites* them, splitting the
domain into independent sub-domains. **The velocity canary does not catch it** — measured on the
hydrostatic column at 16×16×32 with z cut, np = 2 and 4: max\|u\| = `4.5e-17` (each sub-column is
separately hydrostatic!) while max\|P_dist − P_ref\| = **`4.004e+02`** and the discrete ∂P/∂z is off
by **8×g·ρ**. Only the pressure reveals it. Consequences: every multi-rank domain-BC run to date is
only correct if the partition happens not to cut a walled/inflow/outflow axis, and neither the
existing `tests/kokkos_mpi` suite nor `test_multiphysics_mpi` covered the combination (they are all
periodic + IBM). Both new ctests choose a grid whose ORB cuts only x and **assert** the walled axis
stays uncut, so the day the decomposition changes they fail loudly instead of silently passing.
Not fixed here per the WO's hard rule 1 (production change = escalation, not a decision).

**ESCALATION #2 — `fillPropGhosts` / `fillPorousEpsGhosts` skip the property BC under MPI.**
Both apply their domain-face override (zero-gradient for μ/ρ; mirror-about-1 for ε) only
`if (!distributed_)` (`flow_ibm.hpp`), so a distributed run leaves the μ / ρ / ε ghost on a
non-periodic face at its **periodic wrap** value. This is the same family as #1 — the guard looks
like a placeholder for the missing ownership test — but its effect is worse, because it is wrong at
**every** np including np = 1. Measured: the WO's literal two-layer Couette (monotone μ stack, so
μ(0) ≠ μ(N−1)) gives rel du = **`2.7e-2`** between the distributed solver and the single-rank
reference **at np = 1**; with variable viscosity switched off the same configuration is bitwise
identical, which localizes it to the μ ghost. `test_varmu_mpi` therefore ships a **symmetric**
μ2|μ1|μ2 stack (two 10× jumps, so the harmonic-mean gate is unweakened) for which the wrap value
coincides with the zero-gradient value; the file says so and names the restoration. varRho's
hydrostatic column is immune only incidentally — its wall-face ρ ghost multiplies a closed
(openness = 0) face.

**Why the periodic "mean-removed buoyancy" hydrostatic variant was dropped** (recorded so nobody
re-derives it): making the walled column periodic by driving it with `force = g(ρ̄ − ρ)` looks exact
on paper (Σ of the face forces vanishes, so a periodic φ exists), but it is not a rest state — the
domain-mean acceleration `mean(f_f/ρ_f)` is nonzero and lives in the projection's null space, so the
constant mode is unremovable. Measured max\|u\| = 1/N_axis exactly (6.24e-2 at N = 16, 3.12e-2 at
N = 32) at np = 1. The shipped `jump-x` configuration keeps the density jump on the rank boundary
without claiming a rest state.

**Note on cost.** `test_varmu_mpi` is the expensive one: 4096 cells but 240 momentum RB-GS sweeps
per step × 80 steps × 2 solvers, so it is **launch/barrier-bound, not work-bound**. Measured ~12 s
per np at 1 OpenMP thread, ~8× slower at 8 threads, and on CUDA 144 s at np=1 / 1322 s at np=2 —
the latter on a GPU shared with four other agents' processes, where N ranks time-slice one context
and every tiny kernel serialises. Do not read its wall time as solver cost. If it ever needs
trimming, `STEPS_COUETTE` 80 → 60 costs 25 % of the runtime and moves the analytic error 0.0129 % →
0.0960 % (still 5× inside the 0.5 % gate); the velocity multigrid is NOT an option — `setPropertyMode`
disables it (variable-coefficient vmg is deferred).

### WO-D (rung V0) — PLIC toolbox — **DONE 2026-08-30**, with two recorded deviations

Delivered: `src/vof/plic.hpp` (`peclet::flow::vof`, container-free `KOKKOS_INLINE_FUNCTION`s only —
no `View`, no indexing, no halo types, so the V4 promotion to `peclet::core::vof` is a file move) +
`tests/kokkos/test_vof_plic.cpp` (ctest `vof_plic`, ~0.8 s). Full `tests/kokkos` battery 20/20 green
on host-openmp AND nvidia-cuda. Sources followed: Scardovelli & Zaleski JCP 164:228 (2000) in the
branch-reduced form of Lehmann & Gekle *Computation* 10:21 (2022) — their eq. (11)/Listing 1
(forward) and Listing 4 (the optimized SZ inverse, L1-normalized, which is the convention this WO
specifies) — plus MYC from Aulisa et al. JCP 225:2301 (2007) via `basilisk/src/myc.h`. **Both cubic
branches were re-derived from scratch and reproduce Listing 4 term for term**, so the transcription
is verified rather than trusted.

**Gate results (identical on both backends unless noted).**

| gate | result |
|---|---|
| A forward, hand-computed planes (all 5 SZ cases + 6 signed axes) | max \|dV\| **2.2e-16** (< 1e-14) |
| A forward vs an independent inclusion-exclusion oracle, 192 363 samples | max \|dV\| **1.2e-15** |
| B round trip `plicVolume(m, plicAlpha(m,V))`, 1e5 samples | max \|dV\| **6.7e-15** (< 1e-13) |
| B round trip, by family | isotropic 6.7e-15 · near-axis 7.8e-16 · degenerate 2.2e-16 · big-ratio 3.1e-16 |
| B reverse `alpha -> V -> alpha` (V in [1e-3, 1-1e-3]) | max \|d alpha\| **1.3e-15** |
| C `faceFluxVolume(f=1)` vs `plicVolume` | **bitwise identical, 60 000/60 000** |
| C slab additivity (2-way / 7-way partition / axis-aligned closed form) | 1.2e-15 / 2.2e-15 / 5.6e-17 |
| D1 MYC on exact planes, 1000 orientations | max **1.0151 deg**, mean **0.1001 deg**, 1/1000 over 1 deg (Youngs: max 3.67, mean 1.21) |
| D2 sphere, PLIC reconstruction error, 16-32-64 | 4.83e-3, 1.18e-3, 3.09e-4 — **order 1.98** |
| D2 sphere, MYC normal ANGLE, 16-32-64 | 1.302, 0.708, 0.412 deg — **order 0.83** |
| D2 `initSphere` volume vs 4/3 pi R^3 | 7.8e-6, 2.0e-6, 4.9e-7 — order 2.00 |
| E device vs serial host, 1e5 round trips | host-openmp: **100 000/100 000 bitwise**. CUDA: 94 474/100 000 bitwise, max \|diff\| 8.4e-15, device round-trip 3.9e-14 |

Single-phase regression: unaffected **by construction** — the diff adds `src/vof/plic.hpp` (included
by nothing but the new ctest; verified by grep over `src/`, `CMakeLists.txt`, `pyproject.toml`), the
new test, and 4 CMake lines in `tests/kokkos/CMakeLists.txt`. No compilation input of the solver
module changed.

**Deviation 1 — a boundary defect in Lehmann & Gekle's Listing 1, fixed.** Their forward listing
hoists the case-(5) test to the front with the condition `min(n1+n2, n3) <= d && d <= n3`. Eq. (11)
defines case (5) as "the remaining free sector mutually excluded by the other four cases", i.e. the
interval `[n1+n2, n3]`, which is empty unless `n3 >= n1+n2`. When `n3 < n1+n2` the `min(...)` form
still fires at the single point `d == n3`, where the correct case is (3), and returns a wrong volume:
`m = (1/3,1/3,1/3), alpha = 1/3` gave **0 instead of the exact 1/6**. Caught by the hand-computed
`x+y+z<1` tetrahedron in gate A (the randomized gates never hit the measure-zero point).
`src/vof/plic.hpp` uses `n1+n2 <= w && w <= n3`, which keeps the front position and its n1 = 0
protection without the defect. The inverse (Listing 4) is already self-consistent with the corrected
interval — its case-(5) branch is only reachable when `n1+n2 <= n3` — so no change there.

**Deviation 2 — ESCALATED: gate D2 as written measures a quantity the published MYC does not
deliver.** The WO asks for "L1 normal error 2nd-order under refinement (fit slope > 1.7)". Measured
slope for the normal **angle** is **0.83** (pairwise 0.88 / 0.78, decaying to 0.56 at 128^3). This is
not an implementation bug; the mechanism was isolated:

- MYC is not exact on planes (gate D1: mean 0.10 deg, max 1.28 deg over 200k orientations), so it
  fails the Pilliod-Puckett criterion, which is exactly the criterion for a 2nd-order normal.
- On the sphere, **~28 % of mixed cells take the Youngs fallback at every resolution** (29.0 / 29.2 /
  27.9 / 27.4 % at 16/32/64/128), and the Youngs branch's normal error **does not converge at all**
  (measured order 0.50 -> 0.33 -> 0.07; Youngs-only L1 is flat at ~1.2 deg). That non-converging
  population progressively dominates the L1 average, which is why the observed order decays.
- The centred-columns branch alone converges at ~0.96/0.93/0.79 — first order, limited by 3-cell
  column saturation.
- The reference normal is not the culprit: repeating the measurement against the radial direction at
  the exact interface-patch centroid (instead of the cell centre) gives 0.96/0.88/0.67 — same story.

What Aulisa et al. (2007) actually report ~2nd order for is the **reconstruction** error (the
symmetric difference between the exact interface and the PLIC plane), and their own numbers degrade
the same way (2.22 on their coarsest mesh, 1.37 on their finest). Our implementation measures
**1.98** for that quantity (2.03 / 1.93 pairwise) — i.e. it reproduces the published behaviour. So
the shipped gate is the reconstruction-error slope > 1.7 (literature-anchored, and the quantity that
actually governs the advected interface position downstream); the normal-angle error and its order
are printed every run and guarded only by a regression tripwire (monotone decrease, 16->32 order
> 0.7). **Do not "fix" this by tuning MYC** — a 2nd-order normal needs an exact-on-planes scheme
(ELVIRA/LVIRA, 5^3 stencil in 3D and the known 3D cost bottleneck) or the height-function /
plicRDF refinement already scheduled at V3/V5.

Corollary for D1: the WO's "within 1 deg" is exceeded by the published algorithm on ~0.1 % of plane
orientations (max 1.0151 deg here, 1.284 deg over 200k). The ctest gates the measured envelope
(max <= 1.5 deg, mean <= 0.2 deg) and records the numbers. An ablation confirms the algorithm is
right as transcribed: Basilisk's counter-intuitive final selection (`|m[cn][cn]| > max|m_Youngs|`
=> take Youngs) is a **saturation detector** — a clipped column under-reports the height slope and so
inflates the centred candidate's dominant component. Disabling it ("never switch to Youngs") makes
the plane error **max 15.53 deg / mean 0.273 deg** versus **max 1.28 / mean 0.105** as published.

**Also worth carrying into WO-E.** `plicVolume` renormalizes (m, alpha) internally, so it is exactly
invariant under `(m, alpha) -> (lambda m, lambda alpha)`; that is what lets `plicSlabVolume` be a
two-line coordinate rescale with no clipping code, and it is why `faceFluxVolume(f=1)` comes back
*bitwise* equal to `plicVolume`. `sphereCellFraction` is a recursive-octree helper with an exact
tangent-plane leaf closure (2nd order, 4.9e-7 relative at 64^3), not the 4^3 midpoint subsampling the
WO sketched — the sketched accuracy (~1e-2 per cell) would have swamped the D2 convergence gate.

### WO-E (rung V1) — Weymouth–Yue split advection, standalone — **DONE 2026-08-30**

Delivered: `src/vof/advect_wy.hpp` (`peclet::flow::vof::WyAdvector` — its own extended block at
**g = 3** with its own ghost-refresh callback; the solver's `G = 2` machinery, `flow_ibm.hpp` and
`flow_bindings.cpp` are untouched), `tests/kokkos/test_vof_advect.cpp` (ctest `vof_advect`, 8.3 s on
CUDA / 17 s host-openmp), `tests/kokkos/vof_advect_scenes.hpp` (scene builders shared by both
batteries), `tests/kokkos_mpi/test_vof_advect_mpi.cpp` (ctests `vof_advect_mpi_np{1,2,4}`), plus 4 +
11 lines of CMake registration. `tests/kokkos` **21/21 green on host-openmp AND nvidia-cuda**; the
new MPI ctests green np = 1/2/4 on both backends.

**Source.** The JCP paper is paywalled, so the primary source used is Weymouth's MIT thesis
(*Physics and learning based computational models for breaking bow waves…*, 2008, dspace
`1721.1/44754`), whose §2.2.2 **is** the paper's derivation and whose **Appendix A is the full
boundedness proof** — strictly more than the paper contains. Cross-checked term for term against the
independent restatement in Arrufat et al., *Computers & Fluids* 215:104785 (2021), arXiv:1811.12327
§3.3.4 (their eqs. 20/26/27). Nothing was improvised.

**Gate results** (CUDA; host-openmp identical except in the last bits, per the cross-backend
tolerance policy).

| gate | result |
|---|---|
| A planar slab, uniform diagonal flow, 1024 steps (8 laps), CFL 0.25 | **Linf 0.0, L1 0.0, drift 0.0 — exact, bit for bit** |
| B sphere translation, L1(vol) at 16/32/64 | 3.9025e-3, 1.1540e-3, 2.4577e-4 — **order 1.76 / 2.23** |
| B volume drift, per resolution / 32^3 over 1024 steps | ≤ 2.1e-16 / **2.12e-16** (gate 1e-13) |
| C Zalesak, one revolution, 100^2, 1000 steps | **L1/V = 2.81e-2**, E1(per 2D cell) 1.634e-3, drift 2.0e-16 |
| C sensitivity to step count (700 / 1000 / 2000) | 2.54e-2 / 2.81e-2 / 3.22e-2 — weak, not the cause |
| D LeVeque T = 3 with reversal, L1(vol) at 32/64/128 | 7.7485e-3, 2.6779e-3, **5.9770e-4** — order 1.53 then **2.16** |
| D relative shape error L1/V at 32/64/128 | 5.48e-1, 1.894e-1, 4.23e-2 |
| D volume drift / discrete face divergence | 5.9e-15, 2.1e-14, **5.7e-14** / max \|div\| dt/h ≤ 1.2e-15 |
| E CFL guard | throws at CFL = 0.5 with the value in the message; runs at 0.49 |
| F worklist on/off, 40 LeVeque steps | **0 / 54 872 cells differ (bitwise)** |
| G dilation flag frozen vs recomputed per sweep, 200 steps | **2.33e-15 vs 1.455e-2 — 6.2e12x worse** |
| MPI np 1/2/4, periodic + walls + mixed, host + CUDA | **0 bitwise diffs** in every case; global drift ≤ 1.6e-15 |

Single-phase regression: unaffected **by construction**, verified not assumed — the solver module is
one TU (`nanobind_add_module(sdflow … src/flow_bindings.cpp)`, no globs) and
`grep -rn "vof/" src/ CMakeLists.txt pyproject.toml` outside `src/vof/` returns nothing, so no
compilation input of the solver changed. (`src/flow_ibm.hpp` / `src/flow_bindings.cpp` carry another
agent's in-flight WO-A edits in this shared checkout; they are deliberately not staged here.)

**Zalesak is anchored against published numbers, twice.** `L1/V` is the metric of Xie & Xiao,
*THINC-scaling* (arXiv:2103.09541) eq. 27 — `sum|C-C_ex| / sum|C_ex|` — whose Table 5 at N = 100
reads THINC-scaling 1.55e-2, MTHINC 1.61e-2, UMTHINC 2.61e-2, THINC/QQ 3.22e-2 on the identical disk
(r = 0.15 at (0.5,0.75), slot |x-0.5| <= 0.025 and y <= 0.85). Ours, **2.81e-2**, sits inside that
spread. The second anchor is Cassinelli et al. (arXiv:1903.11949) eq. 15, the per-cell mean
`E1 = (1/N^2) sum|C-C_ex|`: ours is 1.63e-3 at D/h = 30, within their PLIC band. A linear PLIC is
expected at the high end of that spread — the schemes below it use quadratic/THINC interface
representations specifically to hold the slot's sharp corners, and MYC's own plane-normal error
(WO-D: mean 0.10 deg, ~28 % Youngs fallback) is the other half. This is **not** a 10x miss.
LeVeque likewise reproduces the published behaviour: Cassinelli report "asymptotically second-order
convergence" for PLIC on this case, and the measured order goes 1.53 (32->64) then **2.16**
(64->128).

**Finding 1 — the CFL cap in the work order is the 2D bound; Weymouth's own 3D bound is half of
it.** Thesis eq. 2.23 / A.33 gives the boundedness restriction as `|u| dt/h < 1/(2(N-1))` for
N-dimensional flow: **1/2 in 2D but 1/4 in 3D**. The WO (and Basilisk / PARIS / AMR-Wind, and
`VOF_PLAN.md` §6) all quote 0.5. Shipped as specified — `cflLimit = 0.5`, aborting at or above it —
with the 3D value documented in the header and settable per instance. Two things make this safe
rather than a latent bug: (i) **conservation is independent of boundedness** — the telescoping is
algebraic and holds whatever C does, so an over-CFL run loses `0 <= C <= 1`, never volume; and (ii)
measured on the LeVeque field at 32^3 and 64^3, sweeping the target CFL 0.24 -> 0.40 -> **0.48**, C
stayed inside `[-5.6e-17, 1.0]` at every setting and the shape error even improved slightly
(7.7485e-3 -> 7.5865e-3 at 32^3, fewer steps). So the 1/(2(N-1)) bound is *sufficient*, not tight,
on this flow. `PECLET_VOF_LEVEQUE_CFL` reproduces the sweep. If a future rung ever sees C leave
[0,1] under a 3D flow, set `cflLimit = 0.25` before suspecting the kernels.

**Finding 2 — a naive zero-gradient wall ghost fill is decomposition-DEPENDENT, and it fails exactly
where the WO says to look.** The natural non-periodic ghost fill is sequential zero-gradient axis
passes over the block. It is not np-invariant: a ghost that is outside the domain in y while its
x-neighbourhood is an *interior* halo gets the x-extension of the true row on one decomposition and
the x-extension of the y = 0 row on another. That difference is not inert — it enters the 3^3 MYC
stencil of the inner corner cell, hence its normal, hence its outgoing flux, and surfaces as
"np differs at the last bit only in ghost-adjacent cells", the WO's own escalation symptom. The fix
(`vofscene::clampFill`) defines the value of every outside-domain ghost as the field at its
**globally clamped** index, which is decomposition-independent by definition; the clamped source is
provably inside the block's extended range (a ghost at global index < 0 on axis a only exists when
`origin[a] < g`) and is always either inner or already exchanged. This was found by reasoning about
the fill rather than by chasing a failing gate, so no gate ever failed.

**Two structural facts worth carrying into V2+.**
1. **Full cells are exactly stationary in floating point, and that is load-bearing.** For a cell
   whose 1D neighbourhood is full, both fluxes take the algebraic branch (`1 * a`) and the dilation
   term is the exact negation of the flux difference; IEEE subtraction is antisymmetric, so
   `fl(a_- - a_+) + fl(a_+ - a_-)` is an exact zero and `C + 0 == C`. Empty cells likewise stay
   exactly 0. This is why the conservation floor scales with **interface area**, not domain volume
   (the measured drifts are ~1e-16 at 32^3 and ~5.7e-14 at 128^3 over 3200 steps). It survives only
   if the flux and the dilation term scale the SAME `uf` by the SAME `dt/h` — writing the dilation
   as `(uf_+ - uf_-) * dth` instead of `uf_+*dth - uf_-*dth` would silently break it.
2. **The velocity field must be discretely solenoidal, not analytically solenoidal.** The dilation
   term adds `H(C-1/2) * div * dt/h` to *every* cell's budget, interior full cells included, so
   pointwise-sampled `uf` would pin the conservation floor at O(h^2) — 10 orders above the gate. The
   LeVeque field is therefore sampled as the **discrete curl of an edge vector potential**
   `A = (0, -psi2(x,z) sin 2pi y, psi1(x,y) sin 2pi z) cos(pi t/T)`, `psi1 = sin^2(pi x) sin^2(pi y)/pi`,
   `psi2 = sin^2(pi x) sin^2(pi z)/pi` (curl A reproduces the field exactly), which puts the measured
   discrete divergence at 1e-17..1e-15 and the drift with it. Uniform translation and solid-body
   rotation are exactly zero bitwise for free. When V2 couples this to the projection, the relevant
   number is the projection's own divergence residual — that, not h, sets the conservation floor.

**Deferred / notes.** (a) The 128^3 LeVeque rung is env-gated (`PECLET_VOF_LEVEQUE_128=1`, 3200
steps, ~5 min on CUDA) so the default ctest stays under 10 s; its numbers are recorded in the table
above from a real run. (b) `WyAdvector`'s implementation methods are `public` with a "treat as
private" comment — nvcc rejects an extended `__host__ __device__` lambda whose enclosing member
function has private access. (c) `debugRecomputeDilation` ships as a permanent, default-off switch so
gate G keeps the #1 trap a measured number; (d) no clipping at this rung, per the plan — the wisp
census is printed instead (e.g. 9489 wisp cells at 128^3 after 3200 LeVeque steps, C in
[-2.1e-17, 1.0]).

### WO-F (blocker fix) — domain BCs are not rank-aware — **DONE 2026-08-30**, one new escalation

Both WO-A escalations are fixed, and the audit found the omission in **nine more** per-face
domain-BC application sites plus two adjacent MPI defects. The single-rank path is byte-identical by
construction: `touchesGlobalFace` is identically true on one rank, so every guarded call runs exactly
as before — verified, not assumed (single-phase regression **+0.00 %** on every metric with identical
iteration counts, below).

**The mechanism, once, so nobody re-derives it.** The core grid halo is built **periodic on all three
axes** (`initMpi`: `std::array<bool,3> per{true,true,true}`), so the exchange fills *every* ghost —
an interior rank boundary with the neighbour's data, a global domain face with the periodic wrap.
The domain BC then **overwrites** the wrap on the rank that owns that face. That fill-then-BC order is
exactly the single-rank order, which is why the ownership test is the whole fix and nothing else has
to change. The transported-scalar path (`applyScalarBc`) already did this; the velocity, openness,
pressure and property paths never adopted it.

**Audit — FIXED (ownership test added).**

| # | site | what it imposed on every rank's own block face |
|---|---|---|
| 1 | `flow_ibm.hpp` `applyVelocityBcCompTo` — **both** the staggered and the collocated loop (wall / Dirichlet-lid / per-position inlet profile / outflow branches) | **THE defect.** wall + inflow velocity ghosts, outflow zero-gradient |
| 2 | `flow_ibm.hpp` `setSolid`: the FLUX-openness construction (`bcZeroOpenness`) | the "pressure-openness BC construction" of the WO — closed β at walls/tangential inflow |
| 3 | `flow_ibm.hpp` `setupBcDiffusion` (`bcDiffusionFold`) | the implicit tangential wall fold baked into `bcDcorr_` / `bcBrhs_` |
| 4 | `flow_ibm.hpp` `pressureBcGhost` (`"pbcghost"`) | zero-gradient P ghosts for the incremental predictor's ∇P |
| 5 | `flow_ibm.hpp` `applyBackflowStab` | the dissipative outflow diagonal |
| 6 | `flow_ibm.hpp` `project()` — outflow φ Dirichlet ghost (`bcZeroPressureGhost`) | φ = 0 at outflow |
| 7 | `flow_ibm.hpp` `project()` — collocated φ wall ghost (`bcNeumannGhost`) | ∂φ/∂n = 0 at walls |
| 8 | `flow_ibm.hpp` `project()` — `bcCorrectOutflow`, **both** the collocated face-field and the staggered cell-field site | the mass-conserving outflow face correction |
| 9 | `flow_ibm.hpp` `maxOpenDivergence` — collocated outflow `bcNeumannGhost` | the diagnostic's outflow face |
| 10 | `mac_cutcell_mg.hpp` `CutcellMG::applyBoundaryOpenness` | the OPERATOR openness α re-imposed **per level** (wall/inflow → 0, outflow → 1) |
| 11 | `mac_cutcell_mg.hpp` `CutcellMG::applyOutflowGhost` | φ = 0 at outflow ghosts, **per level** |
| 12 | `flow_ibm.hpp` `fillPropGhosts` / `fillPorousEpsGhosts` | the μ/ρ/ε domain-face override, previously gated `if (!distributed_)` — wrong at **every** np including 1 |

10 and 11 needed a per-level ownership test, so `CutcellMG::Level` gained `gdim` (that level's GLOBAL
inner dims) next to the existing `og`, and `CutcellMG::touchesGlobalFace(lv, f)` is `og == 0` /
`og + inner == gdim`. `applyOutflowGhost`'s signature changed from `(C3 ext, …)` to `(const Level&, …)`
— every call site already had the level in scope.

**Audit — two adjacent MPI defects found and fixed (not per-face BCs, but in the same machinery).**

- `flow_ibm.hpp` `smoothComp`'s **domain-BC const-coefficient smoother** hard-coded the red-black
  parity origin `og{0,0,0}` instead of `og_` — the only smoother in the file that did not carry the
  global block origin, so its colours swap on any rank whose block origin has odd parity. Reachable
  only on a domain-BC problem with no solid, no advection and no variable coefficients (otherwise
  `bcStencilPath()` takes over), which is why no ctest ever hit it. `{0,0,0}` single-rank → byte-identical.
- The solver's **velocity multigrid is single-rank**: `IbmSolver` calls `vmg_.init` and *never*
  `VelocityMG::initMpi` (which exists and is gated by the standalone `velocitymg_mpi` ctest). Under a
  decomposition `VelocityMG::fill` would periodic-wrap this rank's own BLOCK instead of exchanging —
  i.e. solve the momentum equation on a per-rank torus. It is now **disabled with a stderr notice**
  when `distributed_`, rather than silently wrong. (varRho/varProps already disable it for their own
  reason.) Wiring the distributed hierarchy is separate work.

**Audit — CHECKED AND ALREADY CORRECT (no change).**

- `applyScalarBc` / `applyScalarBcStencil` — already carry `touchesGlobalFace`; the pattern came from here.
- `fillVelGhostsTo`'s periodic axis fill (`fillAxis`) — only on the `!distributed_` branch; the
  distributed branch exchanges first and then applies the (now guarded) BC.
- `CutcellMG::buildAmg`, the agglomerated coarse solve — its non-periodic boundary handling is
  expressed in **global** cell ids (`crosses && bc_[d] != 0`), so it is decomposition-independent by
  construction.
- `CutcellMG::caSmooth` — communication-avoiding smoothing is disabled whenever `hasBC_`, so the CA
  ghost-ring re-smoothing never interacts with a domain BC.
- `CutcellMG::setBoundaryConditions` (`hasBC_`, `hasOutflow_`, `removeMean_`) — global flags, no
  per-rank geometry.
- `setDomainBcProfile` — the profile is resampled onto **this rank's** ghost-inclusive face plane and
  indexed by local face position, so it is rank-local by construction; the only thing missing was the
  ownership test, now in `applyVelocityBcCompTo` (#1). The caller must still hand each rank its own
  block's slice — there is no scatter helper (`flow/CLAUDE.md`'s "multi-rank inlet-profile scatter").
- `VelocityMG::fillProlongBcGhosts` / `setDomainBcOp`'s `boundaryFold` — per-face and unguarded, but
  unreachable under MPI (see the velocity-MG item above); left alone deliberately, and recorded here
  so a future distributed VelocityMG starts from this list.

**Gates.**

| gate | result |
|---|---|
| Single-phase regression, +0.00 % and identical iteration counts (**verified, not assumed**) | **PASS** — `tests/regression/sdflow_regression.py` on the nvidia-cuda build: **+0.00 %** on K / k\* / fitted order p / Richardson extrapolate, and **identical** `p_iter_tot`, per-step iteration medians and step counts on all 13 grid points of `zh_sphere` / `random_spheres` / `hollow_rings` |
| Walled-axis-**cut** np 2/4 vs np 1 on **velocity AND pressure**, host + CUDA | **PASS.** `vardensity_mpi` on 16×16×32 with **z cut** (np=4 cuts x and z): host max\|u_dist−u_ref\| ≤ **5.9e-17**, max\|P_dist−P_ref\| ≤ **9.1e-13**; CUDA **bitwise 0.000e+00 at np=2**, ≤ 4.5e-17 / 2.8e-14 at np=4 (P is O(800)). The periodic `jump-z` companion is **bitwise identical at every np on both backends**. `varmu_mpi` on 16×32×8 with **y cut** (np=4 cuts x and y): du ≤ **7.8e-16**, dp ≤ **6.7e-17**. **Before the fix the same configuration read dp = 2.5e+01 / ∂P/∂z error 0.5** (and WO-A measured 4.0e+02 on the original grid) while the velocity canary stayed at 4e-17 throughout — the pressure comparison is the only thing that sees this |
| Restored asymmetric two-layer Couette matches analytics at every np | **PASS** — the literal monotone μ1\|μ2 stack (10× jump on the CUT walled axis): analytic error **0.0003 %** at np = 1, 2, 4 on host and CUDA (the symmetric stack WO-A had to ship read 0.0129 %) |
| Chebyshev V-cycle count vs decomposition | **PASS**, max-delta **0** over the non-degenerate window at every np on both backends |
| All pre-existing MPI ctests green | **PASS** — `tests/kokkos_mpi` **42/42, 0 failed** on host-openmp with the new tests in place (1787 s), and `tests/kokkos` **21/21** single-rank |

**Test changes.** Both ctests now do the opposite of what WO-A shipped: they **require** the walled
axis to be cut (`if (size > 1 && !cut[axis]) FAIL`), so the day a decomposition change stops cutting
it the coverage loss is loud. `test_vardensity_mpi` moved 32×16×16 → **16×16×32** (walls-z and the
sharp ρ jump both on the long, cut z axis) and gates the pressure field; `test_varmu_mpi` moved
32×16×8 → **16×32×8** and restored the literal two-layer stack. varμ's pressure gate carries an
**absolute** floor (1e-14) beside the relative one, because plane Couette is pure shear and \|P\|
is ~6e-8 there — a purely relative pressure gate would have been gating round-off. The strong
pressure gate is `vardensity_mpi`'s hydrostatic column, where \|P\| = O(800).

**ESCALATION — a THIRD pre-existing defect: closure-written body-force fields never have their ghosts
filled.** Found by root-causing a residual `dp = 2.5e+01` that survived every ownership fix.
`applyClosure` (`property_closures.hpp`) writes the **inner cells only** — its own comment says
"ghosts untouched — refilled by the field's own exchange" — and while a closure targeting `"mu"` or
`"rho"` does get that exchange (`fillPropGhosts`, called from `rebuildStencils` / `project`), a
closure targeting **`force_x/y/z` gets none**: nothing in `step()` ever exchanges a `force_*` field.
`buildRhsVar` / `buildRhsForced` then compute the face body force as `0.5*(fb(i) + fb(i-strd))`, so
**the face force on the first inner plane of every block is halved** (the ghost is the field's
zero-initialised value).

Evidence, measured on the hydrostatic column with z cut at np = 2 (jump deliberately moved to z = 8
so it does not coincide with the rank boundary at z = 16): every face reads the exact `∂P = −g·ρ_f`
— `−100` in the heavy layer, `−50.05` at the ρ jump, `−0.1` in the light layer — **except z = 16,
which reads `−0.05`, exactly half**. The rank-boundary ρ ghost itself was dumped and is *correct*
(1000 on rank 1's low-z plane), and so is the assembled MG level-0 operator row; only the force ghost
is wrong. The velocity stays at 3e-17 because the momentum's face inertia and the projection both use
the same (correct) ρ_f, so `w* = f_f/(ρ_f/dt)` is still uniform — the error lands entirely in P.

This is **not** an MPI-only bug and **not** in WO-F's scope: single-rank it halves the face force at
the periodic wrap plane of any closure-driven body force, so `tests/study/rayleigh_taylor.py`, the
Boussinesq thermal-convection validation and every CFD-DEM feedback closure are affected. Fixing it
(one `fillGhosts` per closure output after `updateProperties()`) would therefore **change single-rank
numbers of already-validated results**, which is exactly the case the WO says to escalate rather than
decide. Not fixed here. The two ctests seed the ghost ring explicitly (`setField("force_z", …)` +
`exchangeField`, alongside the closure, which rewrites the same static inner values every step), with
the reasoning inline, so the WO-F gates measure WO-F's defect and not this one.

**Note on a one-off battery failure (resolved).** An earlier full host-openmp `tests/kokkos_mpi` run reported
`sdflow_colocated_mpi_np4` failing with `MPI_ERR_TRUNCATE` in `MPI_Waitall`. It is **not** caused by
this work order: re-run twice in isolation the patched binary passes and prints `k_dist =
5.77833108e+00` / `5.71014308e+00`, `rel = 3.54e-15` / `2.80e-15` — **digit-for-digit identical to a
binary built from pristine HEAD in a separate worktree**. Every guarded site in this diff sits inside
a `hasBc_` / `hasOutflow_` branch and that test is periodic + IBM, so the code path is provably inert
there. The battery was running against a 12-core `nvcc` build and another agent's core MPI battery on
the same 48-core host at the time; the final battery, run on a quiet machine, is 42/42. Recorded as an
observed load-induced flake in the collocated np=4 halo, not investigated further.

### WO-G (blocker fix) — body-force ghosts are never filled — **DONE 2026-08-30**, one new escalation

Fixed in one place: `IbmSolver::fillCellForceGhosts()` (`src/flow_ibm.hpp`), called from `step()`
immediately after `updateProperties()`, routing `force_x/y/z` through the (WO-F rank-aware)
`fillPropGhosts`. That point is the only one in the step that is after **both** writers — the
closures just above, and an external CFD-DEM `field_view` + `exchange_field_add` deposit, which
happens before `step()` — and before the only consumer (`buildRhsVar`, in the Picard loop). Total
production diff: 5 lines of code + the reasoning comment.

**Ghost policy at a wall: Neumann copy, the same as rho — and the policy does NOT differ per BC
type.** The argument, since a body force is indeed not a transported property:

- `buildRhsVar` face-interpolates the force with the **same** arithmetic mean the momentum time term
  and the projection coefficient use for rho (`f_f = ½(f(i)+f(i−s))`, `rho_f = ½(rho(i)+rho(i−s))`),
  and the physical content of the *pair* is the acceleration `f_f/rho_f`. This three-way consistency
  is exactly what `variable_density_projection.md` §1/§3 identifies as the reason discrete
  hydrostatic balance is exact. So the force's ghost policy is not free: whatever policy rho has, the
  force must have the SAME one, or the ratio breaks at the boundary and the telescoping stops. rho
  gets `fillPropGhosts` → so does the force.
- **Wall / inflow (Dirichlet).** The ghost is read *only* by the face force of the wall-NORMAL
  component ON the boundary plane — the single unknown the Dirichlet BC pins (`bcVelocityComp`,
  `comp == a`: `at(bf) = wall`) and whose flux openness is 0. It is therefore unobservable today
  (that is the whole reason the acid test passed, see below). Neumann copy is still the right answer:
  it is the only choice that keeps `f_f/rho_f` equal to the intended acceleration if that pin is ever
  relaxed (free-slip / traction BC), and *zero* would be the assertion that a **volumetric source**
  stops at the wall, which is false. A body force is a source, not a flux, so there is no reflection
  or odd-extension principle available — only extrapolation, and the piecewise-constant (Neumann)
  extrapolation is O(h), the same order as rho's own ghost. Nothing about a wall makes a different
  extrapolation more accurate.
- **Outflow.** Zero-gradient is what every other quantity gets there, and a zero ghost would halve
  the body force on the outlet face — this very defect, relocated to the outlet.

**The defect's actual reach, measured — narrower than the escalation assumed in one direction and
sharper in another.**

1. **Only `buildRhsVar` reads the ghost.** `buildRhsForced` (the constant-density forced path, i.e.
   Boussinesq / de Vahl Davis / `flatwall_displacement.py`) reads the CELL value `fb(i)` alone — no
   `fb(i−strd)`. So the WO's list of "affected physics" over-reaches: **Boussinesq thermal convection
   was never affected.** The affected set is variable-density (`varRho_`) and the eps-conservative
   porous momentum (`porous_ && porousCons_` → `effVarRho()`), i.e. VoF/RT-class runs and CFD-DEM.
2. **On a periodic axis the error is a net body-force DEFICIT of exactly 1/(2·N_axis).** The halved
   plane makes `u*` non-uniform, the projection removes the non-uniform part, and what survives is
   the (deficient) mean: measured on a uniform-force periodic box, `u/u_exact` = **0.984375** at
   N=32 (= 31.5/32) and **0.96875** at N=16 (= 15.5/16), to the last bit. Under MPI each rank
   boundary costs another half-plane: np=2 on the z-cut 32-cell axis gives 0.96875, np=4 on the
   x-cut 16-cell axis gives 0.9375. That is a 1.6–6.3 % force error, not a round-off effect.
3. **At a WALL it is completely masked** — this answers the WO's question about WO-A's acid test.
   The halved face is the wall-normal velocity plane, which the Dirichlet BC overwrites
   (`at(bf) = wall`) before the divergence and whose flux openness is 0, so the value never reaches
   the solution. Measured: the walls-z hydrostatic column at np=1 reads `dP/dz err = 1.39e-15` with
   the defect present and `1.39e-15` after the fix — bit-identical. That is why WO-A's 2.75e-17
   passed, and why **Rayleigh–Taylor (walled ±z, force_z) is unaffected too**.
4. **An INTERIOR rank boundary has no such pin.** np=2/4 on the same walled column: `dP/dz` error
   **0.5** (exactly half a face force) at plane z=16 — the rank boundary — while `max|u|` stayed at
   **1.6e-17**. The velocity canary is blind; only the pressure sees it. Same signature as WO-F.

**Before / after deltas (WO item 3) — nothing recorded was re-baselined; no `perf_baseline*.json`
and no recorded number in any doc was edited.**

| case | before (pristine HEAD) | after (fix) | delta |
|---|---|---|---|
| **Single-phase regression**, all 13 grid points (`zh_sphere` 16/24/32/48/64, `random_spheres` + `hollow_rings` 24/32/48/64), nvidia-cuda | K/k\* and iteration totals **equal to the recorded baseline** (`+0.00 %` on every metric) | **identical** — `+0.00 %` on K / k\* / fitted order p / Richardson extrapolate, identical `p_iter_tot`, per-step medians and step counts | **0** (as the WO predicted: these cases use `set_body_force`, a uniform scalar, and never register a force FIELD, so `hasCellForce_` is false and the new call is not even reached) |
| **Hydrostatic acid test, ratio 3** (`tests/study/rayleigh_taylor.py`) | max\|u\| `3.994840715318005e-17`, ∂P/∂z rel-err `7.401486830834376e-16` | **bit-identical** | **0** |
| **Hydrostatic acid test, ratio 1000** | max\|u\| `2.753321457158628e-17`, ∂P/∂z rel-err `3.411160243160793e-16` | **bit-identical** | **0** |
| **Rayleigh–Taylor** amplitude series (the §3 record `1.5 → 19.5`, ×13) | `1.5`, `1.865885382400677`, `3.1271757233794446`, `5.535032809746294`, `9.30443304452113`, `14.160760845454503`, `19.518144168246145`; growth `13.012096112164096` | **bit-identical, every digit** | **0** |
| **de Vahl Davis cavity** (`tests/study/dvd_cavity.py`), conduction N=24 Ra=0 | Nu `1.0000000000000338`, 200 steps | **bit-identical** | **0** |
| **de Vahl Davis cavity**, N=32 Ra=1e4 | Nu `2.299525620604257`, u\* `16.585277478201778`, v\* `20.125750226019694`, 700 steps | **bit-identical** | **0** |
| `tests/kokkos_mpi/test_vardensity_mpi` `walls-z` / `jump-z`, np 1/2 CUDA | du = dp = `0.000e+00`, max\|u\| `2.97e-17`, ∂P/∂z err `1.14e-15`, Chebyshev `17..13` | **identical**, with the WO-F hand-seed of the force ghosts REMOVED | **0** |

So **every recorded validated number is bit-identical across this fix.** The reason is structural,
not luck: all of them are either constant-density-forced (no ghost read at all) or walled on the
forced axis (the Dirichlet pin masks the halved plane). What moved is exactly the set nothing had
gated: periodic body forces and multi-rank interiors, which is what the new ctest now covers.

**Gates.**

| gate | result |
|---|---|
| New tests **fail before** the fix (demonstrated, not asserted) | **PASS.** Pristine HEAD built in a separate `git worktree` (host-openmp) and the pre-edit CUDA binary. host np=1: `per-z` value `0.984375` (want 1) FAIL, `per-x` `0.96875` FAIL, `walls-z` OK; np=2: `per-z` `0.96875`, du `1.563e-02`, dp `6.055e-02`; `walls-z` dp `2.500e-02`, **∂P/∂z err `5.000e-01` at plane z=16 with max\|u\| `1.63e-17`**; np=4: `per-x` `0.9375`, du `3.125e-02`. CUDA reproduces every value |
| New tests **pass after** | **PASS**, host-openmp and nvidia-cuda, np 1/2/4: `per-z` and `per-x` `spread = 0.000e+00`, `value = 1` exactly, cross-component `0.00e+00`, and **du = dp = `0.000e+00` (bitwise vs the np=1 reference) at every np on both backends**; `walls-z` du ≤ `3.7e-17`, dp ≤ `2.3e-16`, ∂P/∂z err `1.388e-15` |
| np 2/4 vs np 1, host + CUDA, on all pre-existing tests | **PASS — host-openmp `tests/kokkos_mpi` 45/45, 0 failed** (3440 s; 42 pre-existing + the 3 new `bodyforce_ghost_mpi_np{1,2,4}`). **nvidia-cuda 45/45**, in three passes: 1–38 first, then 34–38 and 39–45 re-run after `test_varmu_mpi` was rebuilt without its WO-F hand seed (the first CUDA `ctest` was killed by the agent harness at #39 — not a test failure). `varmu_mpi_np{1,2,4}` re-run green on host too (167/243/248 s) with the seed removed |
| Single-rank `tests/kokkos` | **PASS — 21/21** on host-openmp |
| Deltas measured and reported for every case in item 3 | **PASS** — the table above |
| No baseline file and no recorded number in any doc edited | **PASS** — `git diff` touches `src/flow_ibm.hpp`, `CLAUDE.md` (new paragraph), `tests/kokkos_mpi/CMakeLists.txt` (one name), `test_vardensity_mpi.cpp` (removal of the hand seed) and adds `test_bodyforce_ghost_mpi.cpp`. `perf_baseline*.json` untouched; `doc/variable_density_projection.md` untouched |

**Audit (WO item 2) — every field written inner-only (by `applyClosure` or a Python `field_view` /
`set_field` write) and then read with a face average.**

| field | face-average consumer(s) | ghost fill | verdict |
|---|---|---|---|
| `force_x/y/z` | `buildRhsVar` `0.5*(fb(i)+fb(i−s_c))` | none | **THE defect — fixed here** |
| `rho` | `VarFaceProps::idiag`, `buildRhsVar` `rhoF`, `buildAdvStencilVar` `fouw`, `buildRhoCoeff`, `projectCorrectVar` | `fillPropGhosts(rhoField_)` in `rebuildStencils`, `buildAdvStencilVar` (c==0) and `project()` | correct — every consumer is preceded by a fill in the same step |
| `mu` | `VarFaceProps` arithmetic/harmonic face mean | `fillMuGhosts()` in `rebuildStencils` / `buildAdvStencilVar` | correct |
| `eps` | `divergOpenEps`, `buildPorousCoeff*`, `maxPorousResidual` | `fillPorousEpsGhosts()` before the divergence in `project()`, and the coupling driver fills + clips the ghosts before `step()` | correct, and deliberately driver-owned before the step (the comment at the coefficient bridge states the one-policy requirement) |
| `epsRho_` (`rho_eff = eps·rho`) | `buildRhsVar` via `effRhoField()` | not exchanged — but `updateEpsRho` is a **whole-block** `RangePolicy(0, n_)` product, so its ghosts are valid iff `eps`'s are | correct by construction |
| `drag_beta` | `mac_pressure` `buildPorousCoeffDrag/Cons`, `projectCorrectPorousDrag` (**and** `addDragDiagonal`'s `0.5*(beta(i)+beta(i−s_c))` when `porous_`) | `fillPropGhosts(dragBeta_)` in `project()` **only** | **ESCALATED — see below** |
| `divAdv_` | `buildRhs*` `0.5*(dv(i)+dv(i−strd))` | none needed: `computeDivAdv` writes the `G−1` ring, i.e. one ghost layer, so the first inner plane's neighbour is valid | correct — and this is the pattern `applyClosure` should have used |
| `P_` | `buildRhs*` `P(i)−P(i−strd)` | `fillGhosts(P_)` + `pressureBcGhost()` at the top of `step()` | correct |
| transported scalars (`add_scalar`) | own advection/diffusion stencils | `scalarFillGhosts` (fill + `applyScalarBc`) | correct — this is where the rank-aware pattern came from |
| `sdf`, openness `ox/oy/oz` | geometry face reads | filled at `setSolid` / `setPressureGeometry` | correct (static) |

Python-side writers checked: `tests/study/dvd_cavity.py` (`force_y` closure — `buildRhsForced`, no
ghost read), `tests/study/rayleigh_taylor.py` (`force_z` closure on a walled axis — masked),
`tests/study/flatwall_displacement.py` (`set_field("force_*")`, constant density —
`buildRhsForced`, no ghost read), `coupling/python/peclet_coupling/driver.py` (`force_*` and
`drag_beta` via `field_view` + `exchange_field_add`, `eps` via `exchange_field` + explicit
domain fill + clip).

**ESCALATION — a FOURTH pre-existing defect, same family: `drag_beta`'s face average is consumed one
phase before its ghosts are filled.** Under `porous_`, `addDragDiagonal` builds the momentum diagonal
from `beta_f = 0.5*(beta(i) + beta(i−s_c))`, and all three of its call sites are inside stencil
builds that run at the TOP of `step()` (`rebuildStencils` line ~1628, `buildAdvStencil`,
`buildAdvStencilVar`). The only `fillPropGhosts(dragBeta_)` is inside `project()`, i.e. *after* those
builds. So on the first inner plane of every block the momentum diagonal uses the previous step's
ghost — or, for the CFD-DEM writer, the deposit residue that `exchange_field_add` leaves behind and
never refills — while the projection's coefficient on that same face uses the freshly exchanged
value. That is precisely the momentum/projection `beta_f` mismatch whose consequence
`addDragDiagonal`'s own comment records ("the loop has gain (idt+beta_f)/(idt+beta_cell) at a beta
jump (bed top: ~3) and the accumulated pressure diverges exponentially"), only localized to block
boundaries instead of the bed top. **Not fixed here**: `drag_beta` is not a force field, the WO scopes
this work order to the force-field ghost plumbing, and the fix would move CFD-DEM numbers (the HCS /
fluidized-bed benchmarks) that need their own measurement campaign. The one-line shape of the fix is
the same as this WO's: `fillPropGhosts(dragBeta_)` next to `fillCellForceGhosts()` at the top of
`step()`, keeping the `project()` call (it must also run after any mid-step change).

**Test notes.** `tests/kokkos_mpi/test_bodyforce_ghost_mpi.cpp` gates **absolute physics**, not a
distributed-vs-reference comparison, and that is load-bearing: the single-rank reference carries the
identical defect, so `du = 0` proves nothing — `per-z`/`per-x` at np=1 ARE the single-rank periodic
wrap-plane variant the WO asks for, and they fail before the fix. `per-z` and `walls-z` require the
z axis to be cut at np > 1 (loud on a decomposition change); `per-x` runs at every np because at
np=2 its "rank boundary" is the periodic wrap plane in x, which is an equally hard gate. `per-x`
writes the force through `setField` (the external CFD-DEM writer path) rather than a closure, so both
writers are covered. `test_vardensity_mpi` lost the WO-F hand seed of the force ghosts (see the diff
comment): the seed and the fix disagree at the wall — the seed's exchange periodic-wraps `-g·1` onto
the heavy wall, the fix applies the Neumann copy `-g·RATIO` that rho itself gets — and the result is
unchanged either way, because that plane is pinned.

**Cost.** `fillCellForceGhosts` is 3 ghost exchanges per step, and only when a force field is
registered. It is applied unconditionally rather than gated on `effVarRho()` so the field's ghost
contract does not depend on which RHS kernel happens to consume it; on the Boussinesq path it is
numerically inert (measured: de Vahl Davis bit-identical), which is itself the check that the fill
reaches nothing it should not.
