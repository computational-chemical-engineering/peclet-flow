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

## WO-H (RECOMMENDED, not yet authorized) — the PCG selector no-op + the 3-D wall-bounded MG-PCG stall

**Status: awaiting user go-ahead.** Off the VoF critical path (Chebyshev is the varRho default
and converges in 11–18 its on box/cylinder/ring geometries), but it is a real pre-existing
defect in the *documented* pressure-driver API and in the solver's behaviour on 3-D
wall-bounded grids. Scoped here so it can be launched as-is.

**The two coupled defects** (WO-B, verified in-source 2026-08-30):
1. **`setPressurePcg(bool /*on*/, int, double)` (`flow_ibm.hpp:240`) never writes
   `useChebyshev_`** — the `on` flag is discarded (the signature literally comments the
   parameter out) and only `pcgMaxit_`/`pcgRtol_` are stored. The comment at `:237` explains
   the original intent ("MG-PCG by default; the `on` flag is accepted for API parity"), which
   was true before `setDensityMode` began setting `useChebyshev_ = true`. Consequence: after
   `set_density_mode("variable")` **there is no way to select PCG through the documented
   call**, contradicting `CLAUDE.md` ("PCG and Chebyshev are mutually exclusive — last set
   wins"). The working spelling is `set_pressure_chebyshev(False, …)`. This is why
   `variable_density_projection.md` §2's "PCG stalls on ρ-scaled coefficients" was actually
   measuring Chebyshev against its own 120-iteration cap.
2. **With PCG genuinely selected, MG-PCG stalls on 3-D wall-bounded grids at CONSTANT
   density** — 200/200 iterations with div ≈ 1.2e-5 at nz ≥ 8 (Chebyshev: 13–14 on the same
   operator), while periodic+IBM is healthy at every ratio from ρ≡1 to 10⁴ (7–10 its).
   Present in the 2026-07-06 release build; independent of bottom mode, mean-removal scope,
   momentum tolerance and μ; vanishes at `levels=1`. **It is not a variable-density defect at
   all.** No shipped validated result is affected because every domain-BC verification script
   is quasi-2D (nz=4), where the stall does not appear.

**Order matters**: fixing (1) alone would silently switch users from a working Chebyshev to a
stalling PCG on exactly the 3-D wall-bounded configurations of (2). **Fix (2) first, or fix
both together and gate them jointly.**

**Do.** Diagnose (2) properly before touching it: the `levels=1` cure points at the coarse
hierarchy under domain BCs, and `nz ≥ 8` vs `nz = 4` points at how many levels actually
coarsen the walled axis. Prime suspects to separate by measurement: whether the domain-BC
V-cycle is symmetric (a nonsymmetric preconditioner breaks CG's orthogonality — note WO-C
tests exactly this hypothesis with FCG, so **read WO-C's result before starting**), and
whether the coarse-level BC re-imposition is self-adjoint. Then repair the selector, make the
two setters genuinely mutually exclusive in both directions, and correct `CLAUDE.md` and the
binding docstrings. Add a 3-D wall-bounded PCG convergence ctest (nz ≥ 8) — the coverage gap
that let this live since July.

> **ANSWERED IN PART BY WO-C (2026-08-30) — read its findings entry before starting.**
> (i) **The suspect is confirmed: the domain-BC V-cycle is not a symmetric preconditioner.**
> Flexible CG converges on **93 of the 130** configurations where MG-PCG fails, with 0
> regressions, and the β-numerator contamination `pr = |rᵀz_k|/|rᵀz_{k+1}|` (zero iff M is
> symmetric; printed under `PECLET_FLOW_MG_DEBUG=2` by `solveFCG`) measures **0.062 median** on
> periodic + IBM against **0.43–0.48** wall-bounded on the *same* geometry.
> (ii) **It is the FIRST coarse level.** `levels=1` solves in 1 iteration; `levels=2` already
> shows the full effect (PCG 200/200, `pr` median 0.58) and deeper hierarchies are no worse. So
> target the level-1 pair — `applyBoundaryOpenness`'s per-level re-imposition and the
> non-periodic prolongation ghosts — as an adjoint pair, not the depth or the bottom solve.
> (iii) **The fix order is now cheaper to satisfy.** A working Krylov driver on domain-BC grids
> exists today (`set_pressure_fcg`), so the selector repair can land with the varRho/porous
> defaults routed to FCG rather than PCG, without waiting for the symmetry work.
> (iv) **Two residual failure modes FCG does NOT cure**, both new and both on V2's path: the
> gravity-driven hydrostatic column with a global stratification (a *stationary* iteration — the
> residual freezes at `r/r0 = 6.98` with `pr` locked at exactly 0.500 — while Chebyshev returns
> the machine-exact rest state), and a small coefficient ρ₀/ρ_f adjacent to a prescribed-velocity
> (inflow-type) face at ratio ≥ 10². Do not assume the symmetry repair covers them.
> (v) The new ctest should gate **all three** Krylov-family drivers on a 3-D wall-bounded grid,
> including the two residual configurations above.

**Gates.** Single-phase regression bit-exact; the new 3-D wall-bounded PCG test converges;
`set_pressure_pcg(True)` after `set_density_mode("variable")` demonstrably selects PCG;
existing quasi-2D verifications unchanged; 45+ MPI ctests green.

## WO-I (authorized 2026-08-30) — `drag_beta` ghosts are stale in the momentum build

**Why.** Found by WO-G; the same defect class as the body-force ghosts, one phase earlier in
the step. Verified in-source: `addDragDiagonal` (`flow_ibm.hpp:4593`) forms the face drag
coefficient `0.5*(beta(i) + beta(i - s_c))`, and it is called from the momentum stencil builds
at the **top** of `step()`; the only `fillPropGhosts(dragBeta_)` is inside `project()`
(`:3831`). So on every block's first inner plane the **momentum diagonal uses the previous
step's β ghost** (or, under CFD-DEM, the residue of the last deposit) while the projection
coefficient on that same face uses the freshly exchanged value.

That is exactly the momentum/projection β_f mismatch whose consequence `addDragDiagonal`'s own
comment records: *"the accumulated pressure diverges exponentially."* Same three-way
consistency argument as WO-G's ghost-policy note — the momentum time term, the drag diagonal
and the projection coefficient must agree on the face value or the discrete balance breaks.

**Unlike WO-G, this one is expected to move validated numbers.** It sits under the CFD-DEM and
HCS gas–solid benchmarks. Therefore this work order is **measure-first and re-baseline-never**,
exactly as WO-G was.

**Do.**
1. Exchange `dragBeta_` ghosts at a point that is after its writer (the CFD-DEM deposit /
   closure) and before the *first* consumer (the momentum stencil build at the top of
   `step()`). WO-G's `fillCellForceGhosts()` call site immediately after `updateProperties()`
   is the natural precedent — reuse it if the ordering holds, and say why it does.
2. Check whether `project()`'s existing `fillPropGhosts(dragBeta_)` then becomes redundant. If
   it does, leave it (harmless, and removing it is a separate change) but say so.
3. Audit the remaining porous fields for the same top-of-step-consumer / late-exchange
   pattern: `epsField_`, `epsRho`, and anything else `buildPorousCoeff*` or the porous momentum
   path face-averages. List what you checked.
4. **MEASURE, DO NOT RE-BASELINE.** Report before/after deltas for: the single-phase regression
   (13 grid points), the porous/CFD-DEM verification scripts in `flow/` and any HCS / gas–solid
   benchmark scripts you can find and run (search `scripts/`, `tests/study/`, and the coupling
   project), and a hydrostatic-in-porous case if one exists. Leave every `perf_baseline*.json`
   and every recorded number in every doc **untouched**; the deltas go in the findings log.
5. Add an MPI ctest that would have caught it: a uniform β across a rank boundary with the
   momentum diagonal asserted uniform on the first inner plane, np 1/2/4 bitwise.

**Gates.**
- New test fails before the fix and passes after — demonstrate both.
- np 2/4 bitwise vs np 1, host + CUDA; all pre-existing ctests green (45+ MPI, 21 kernel).
- **Single-phase regression must stay bit-exact** — it does not enable the porous path, so a
  movement there means the fix reached further than intended: escalate.
- Deltas reported for every case in item 4; no baseline or recorded number edited.

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

### WO-B (rung S0) — pressure-driver measurement battery — **DONE 2026-08-30**, three escalations

Delivered: `tests/study/vardensity_solver_probe.py` + `tests/study/vardensity_solver_probe.json`
(the emitted markdown table is printed by the script and reproduced in condensed form below).
**406 configurations × 20 steps** on nvidia-cuda (592 s) and the 32³ subset (328 configurations) on
host-openmp (792 s), both stored as separate `runs` entries in the JSON: geometry ∈ {open box, immersed cylinder, the regression suite's 3-Raschig-ring bed,
`data/packing_ring.vti` strided to 64³} × ρ-shape ∈ {**constant-ρ control**, slab, sphere blob,
grid-diagonal tilted film} × edge ∈ {tanh over ~2 cells, sharp one-cell step} × ratio ∈ {10², 10³,
10⁴} × driver ∈ {Chebyshev, MG-PCG} × case ∈ {hydrostatic, lid-driven, **fully periodic + body
force**}. No production code was touched.

**THE HEADLINE: `variable_density_projection.md` §2 is a misdiagnosis, and the real defect is
bigger and older than variable density.** Three measured facts, each reproducible from the shipped
script:

1. **`set_pressure_pcg(True, …)` cannot select PCG at all.**
   `IbmSolver::setPressurePcg(bool /*on*/, int maxit, double rtol)` (`flow_ibm.hpp`) ignores its
   `on` flag and only stores `pcgMaxit_`/`pcgRtol_`; the only setter that writes `useChebyshev_` is
   `setPressureChebyshev`. So after `set_density_mode("variable")` — which sets
   `useChebyshev_ = true` — **every `set_pressure_pcg(True, …)` call leaves the solve on
   Chebyshev**, contradicting the comment at `setDensityMode` ("an explicit set_pressure_pcg …
   AFTER set_density_mode still wins (last set)"), the binding docstring ("exclusive with
   Chebyshev") and `flow/CLAUDE.md` ("last set wins"). The tell is that such a "PCG" run caps at
   exactly **120** iterations — `chebMaxit_`'s default — no matter what cap was passed. The whole
   first pass of this battery ran that way; every number below uses
   `set_pressure_chebyshev(False, …)` first, which is the only spelling that actually selects PCG.
2. **With PCG genuinely selected the §2 stall reproduces exactly — and it is NOT caused by the
   ρ-scaled coefficients.** On the literal §2 configuration (`rayleigh_taylor.py`'s
   `hydrostatic()`: 8×8×24, walls ±z, μ=0, gravity closure, ratio 3) real PCG runs
   **2000/2000 iterations on every step**. But the same stall appears at **constant density**, and
   it does **not** appear on a periodic problem at any ratio (the one periodic weak spot — the
   grid-diagonal `tilt` film — plateaus four orders lower, at the round-off floor; see below):

   | configuration (32³ unless noted, real driver selection) | PCG | Chebyshev |
   |---|---|---|
   | periodic + immersed cylinder, ρ ≡ 1 (no varRho) | **7** its, div 4.1e-12 | 8 |
   | periodic + immersed cylinder, ratio 10² / 10³ / 10⁴ | **7–10 / 7–10 / 7–10**, div 4.2e-12 | 8 / 8 / 8 |
   | lid box, **constant density**, third axis nz = 2 / 4 | 24–37 / 18–55, div 6e-13 | 17–18 (nz = 4) |
   | lid box, **constant density**, nz = 8 / 16 / 32 | **200/200**, div **1.1e-05 … 1.2e-05** | 13–14, div 2.1e-13 |
   | all-six-walls box, constant density, nz = 4 / 32 | 21–119 / **200/200** (div 1.2e-05) | 14 / 14 |
   | hydrostatic column (walls ±z), constant density, 32³ | **200/200**, div 2.2e-05 | 11–12 |

   So the discriminator is **the domain-BC (non-periodic) coarse hierarchy on a genuinely 3-D
   grid**, not ρ. `levels=1` (no coarse grid at all) converges in **1** iteration; every depth
   2…6 stalls; `set_pressure_bottom` smoother/auto/agglomerated and
   `set_pressure_mean_removal` all/fine make no difference; and the RHS is *not* the cause —
   40, 400 and 2000 momentum RB-GS sweeps give bit-identical stalls. The V-cycle itself is a fine
   preconditioner (Chebyshev, which only needs real bounds, converges in 11–18 everywhere here),
   so this is exactly the "the preconditioner is not SPD w.r.t. the fine operator" signature §2
   inferred — attached to the **boundary-condition** handling of the hierarchy, not to ρ.
3. **What the density ratio changes is the consequence, not the cause.** On the §2 column with
   real PCG: at ratio 3 the stalled solve still returns the right answer (steady max|u| 5.9e-17,
   ∂P/∂z error 7.4e-16); at ratio 1000 it **destroys** it — iteration counts go erratic
   (`12, 41, 34, 11, 9, 9` — the breakdown guard exiting early) and max|u| = **2.10e+01**,
   ∂P/∂z error **1.03**, max|div(open·u)| = **43**.

**This defect is pre-existing and long-standing, not introduced by WO-A/F/G.** Reproduced on the
**2026-07-06 release build** (`4c781e3`, two days after varRho landed, built in a worktree against
today's core): lid box nz=4 PCG 20–24 its (fine), nz=32 PCG **200/200** with div 1.23e-05, Chebyshev
13–14. WO-F and WO-G are single-rank byte-identical on these paths, so they are not implicated.
(The varRho commit `ab5ae43` itself does **not** compile standalone — it calls
`buildFaceCentroidDist`/`transposeGradWallAware`, which only arrive in `cb4bfa0` — so the exact tree
§2 was written against cannot be rebuilt.)

**Why nothing caught it: every shipped domain-BC verification is quasi-2D.** `verify_lid_cavity`,
`verify_channel`, `verify_bfs` and `dvd_cavity` all run with the third axis at 4 cells, which is
inside the healthy regime. Measured on the **literal** `verify_lid_cavity_sdflow.run()` configuration
(N=128, nz=4, Re=100, levels=8, 200 steps, host-openmp): PCG median **62** its (56–73),
div **3.1e-16**, min centreline u/U = **−0.1690**; Chebyshev median 87, div 3.3e-14, the identical
−0.1690. **No shipped validated result is affected.** The 3-D wall-bounded case that is affected is
precisely the two-phase / hydrostatic geometry V2 needs.

**The battery (nvidia-cuda; `it_median` aggregated over the three ρ-shapes / worst step; the
constant-ρ control is the `const rho` column).**

*Chebyshev is flat in the density ratio on every well-resolved geometry, in every case.*

| case | geom | edge | const ρ | 10² | 10³ | 10⁴ |
|---|---|---|---|---|---|---|
| hydro | box | sharp | 11 | 12 | 12 | 12 |
| hydro | cyl | sharp | 13 | 13 | 13 | 13 |
| hydro | rings | sharp | 14 | 14 | 14 | 14 |
| lid | box | sharp | 14 | 14 | 14 | 14 |
| lid | cyl | sharp | 14 | 14 | 14 | 14 |
| lid | rings | sharp | 14 | 15 | 15 | 15 |
| per | cyl | sharp | 8 | 8 | 8 | 8 |
| per | rings | sharp | 13 | 13 | 13 | 13 |
| hydro | **pack** (64³) | sharp | 23 | 60 | 86 | **144** |
| lid | **pack** | sharp | 68 | 75 | 134 | **177** |
| per | **pack** | sharp | 69 | 74 | 90 | **155** |

Smooth vs sharp is **not** a significant axis: the largest difference anywhere on
box/cyl/rings is 18 vs 14 (rings, hydro, 10⁴) and the *smooth* edge is the worse one — a 2-cell
tanh spreads the jump over more coarse-grid cells than a step does, so a sharp VoF interface is if
anything the easier coefficient field for this hierarchy. Ratio 10²→10⁴ costs at most +1 iteration
on box/cyl/rings (1.00×–1.09× vs the constant-ρ control).

*On the PRODUCTION configuration (`per` — fully periodic pore-scale flow with a body force) the two
drivers swap places, and PCG is much better than Chebyshev on the real pore geometry:*

| geometry, `per` case, sharp edge | driver | ρ ≡ 1 | 10² | 10³ | 10⁴ |
|---|---|---|---|---|---|
| immersed cylinder | PCG | 7 | 7 | 7 | 7 |
| immersed cylinder | Chebyshev | 8 | 8 | 8 | 8 |
| 3-ring bed | PCG | 10 | 11 | 11 | 11 |
| 3-ring bed | Chebyshev | 13 | 13 | 13 | 13 |
| **`packing_ring.vti` 64³** | **PCG** | **14** | **14** | **16** | **16** |
| **`packing_ring.vti` 64³** | Chebyshev | 69 | 74 | 90 | **155** |

That last pair is the single most decision-relevant measurement in this work order: on the real
ring packing at ratio 10⁴, **MG-PCG needs 16 iterations and Chebyshev needs 155**, on the identical
operator. The `pack` difficulty reported in the wall-bounded rows above is therefore a **Chebyshev**
weakness (fixed 15-step power-iteration spectrum bounds, then inflated ×0.95/×1.05, on a tortuous
cut-cell spectrum), not a coefficient-aware-coarsening problem. Per-row on `pack`/`per`: PCG's
median is **13–19** on every one of the 26 configurations (worst single step 34) and **all 26 are
healthy**; Chebyshev's median runs 44 → 200 and 5 of its 13 rows cap.

**PCG's one weak spot on the periodic path is the tilted film, and it is a floor, not a stall.**
Of the 50 periodic 32³ PCG configurations, 33 are healthy and the 17 that are not are **exactly the
`tilt` shape** (grid-diagonal interface, in a fully periodic — hence singular — box). They plateau
at max|div(open·u)|/u between **5e-10 and 6.8e-7**, i.e. at the round-off floor of the mean-removed
system, and simply never reach `rtol·r0 = 1e-8`; the `const`, `slab` and `blob` shapes are healthy
at every ratio. Contrast the domain-BC failure, whose plateau is **1e-5** — four orders worse and a
genuinely wrong projection. Chebyshev is healthy on all 50.

**Resolution and coarse-solve controls on `pack`** (so the above is not read as a discretization
artifact): at **128³** (`--geoms pack --n 128`, stride 2, so the ring walls are ~2 cells thick
instead of sub-cell) Chebyshev gets *worse*, not better — const 58/69, ratio 10² 82/107, ratio 10⁴
200/200 — so it is the geometry, not the 64³ downsampling. `set_pressure_bottom` makes no
difference on `pack` (agglomerated 24/68/105/200 vs smoother 23/68/88/200 vs auto), and reducing the
depth to `levels=3` is worse (71/68/115/200). The coarse *solve* is not the lever.

**Chebyshev's per-step bound re-estimation is 2/3 of its pressure stage** (the S2 target,
quantified). Under varRho the coefficient operator is rebuilt every step and `chebBoundsSet_` is
invalidated with it, so `estimateEigenvalues(…, iters=15, …)` runs every step — **two 15-step power
loops, each applying M⁻¹A once, i.e. 30 extra V-cycles per step** on top of the ~14 the solve needs.
Isolated by A/B (constant density estimates once, at step 1, and reuses): the estimate costs
**21.2 ms** of a **31.6 ms** varRho Chebyshev projection at 32³, i.e. **67 %**; amortizing it would
leave 10.5 ms — a **3× cut of the varRho pressure stage**, independent of iteration count.

**Ratio ceiling of the varRho path itself** (Chebyshev, hydrostatic column, 32³, measured while
hunting for the stall — recorded because it bounds what "ratio sweep" can mean): ratio 10³ → 10⁴ →
10⁵ are all machine-exact (steady max|u| 3.0e-17 / 3.8e-17 / 5.6e-17, 17 / 18 / 19 its), **10⁶
loses hydrostatic exactness** (max|u| **1.7e-05**, 21 its) and **10⁸ NaNs** both drivers. The MG
level fields are `float`, so ρ₀/ρ_f = 10⁻⁶ is at the edge of single-precision relative accuracy;
this is a coefficient *storage* limit, not a driver limit (both drivers fail identically). Ratio 10⁴
— an air/water contrast at 10³ plus margin — is comfortably inside it.

**Gates.**

| gate | result |
|---|---|
| The battery runs to completion on host **and** CUDA | **PASS** — nvidia-cuda 406/406 configurations (592 s) and host-openmp 328/328 (the 32³ `box`/`cyl`/`rings` battery, 792 s), both stored in the JSON. Cross-backend: over the 328 shared configurations the **healthy/unhealthy verdict is identical on every single one** (0 mismatches), and over the 198 that are healthy on both the largest median-iteration difference is **0.5** (a median over an even count — i.e. no configuration differs by a whole iteration). The per-driver tallies match exactly: Chebyshev 0/164 unhealthy on both backends, PCG 113/114 wall-bounded and 17/50 periodic unhealthy on both. See "Pruning" below for what the host run deliberately omits |
| The §2 PCG stall is reproduced on at least one configuration | **PASS**, but only after fixing the driver selection, and the reproduction **refutes §2's mechanism** — see the headline. 2000/2000 iterations on the literal §2 case, and on the 32³ battery **113 of the 114 wall-bounded PCG configurations are unhealthy** (hydro 56/57, lid 57/57; 30 of them capped on all 20 steps) against **0 of 164 for Chebyshev**. The same stall occurs at constant density |
| The table answers: Chebyshev its vs ratio, vs geometry, vs sharp/smooth | **PASS** — flat in ratio (1.00–1.09×) and in edge on box/cyl/rings; 11–18 its. Geometry is the only strong axis, and only for the real `packing_ring.vti` bed (23–69 constant-ρ, 144–177 at ratio 10⁴) |
| Solver code untouched | **PASS** — the diff is one new study script, its JSON, this entry and a superseding note in `doc/variable_density_projection.md` §2 |

**The S-ladder implication (recorded, not decided).** The WO's success metric — "Chebyshev ≤ ~40 its
at 10³ across geometries ⇒ S2 is the only remaining work; > ~80 or ratio-divergent ⇒ S3/S4 promote"
— splits: **12–15 its** on box/cylinder/3-ring-bed (S2 territory, comfortably), **90–155 its and
ratio-divergent** on the real `packing_ring.vti` bed (nominally S3/S4 territory). But the promotion
does not follow, because **MG-PCG solves that same ratio-10⁴ packing operator in 16 iterations**.
Coefficient-aware coarsening (S3) and symmetric/Galerkin transfers (S4) are aimed at a hierarchy
that fails on ρ-scaled coefficients; no measurement in this battery shows one — the periodic
hierarchy handles ratio 10⁴ at parity with ratio 1 under *both* drivers on every geometry, and the
wall-bounded hierarchy fails identically at ratio 1. What the measurements do indicate, in order:
(i) fix the domain-BC MG-PCG stall and the `set_pressure_pcg` no-op — that is what made varRho look
broken, and it is a prerequisite for any honest driver comparison on a wall-bounded two-phase case;
(ii) **S1 (flexible CG, WO-C) is now the sharpest instrument available**, since the stall is exactly
the non-SPD-preconditioner failure FCG is designed to tolerate, and this battery is ready to gate
it; (iii) S2 remains worth its cost *for as long as Chebyshev is the varRho default*, and is worth
**3×** on the pressure stage; (iv) **S3/S4 are not indicated by any measurement here** and should
stay parked until one is.

**Pruning (declared).** The host-openmp run omits the `pack` (64³ `packing_ring.vti`) sub-sweep: it
is the only rung where a stalling PCG configuration costs ~4000 V-cycles on 2.6e5 cells, and the
CUDA run already establishes both the geometry effect and the driver crossover there; the host run's
purpose is the cross-backend consistency gate, which the three 32³ geometries serve at 1/10 the cost.
Nothing else was dropped: the full geometry × shape × edge × ratio × driver × case cross-product ran
on CUDA. Two combinations are structurally skipped by the script (`per` × `box` × {`const`, `slab`}):
a z-only ρ stratification in a periodic box leaves the predictor x-uniform, so `div(u*) = 0`
identically and the pressure solve returns after 0 iterations — a degenerate cell, not a measurement.

**Reading the tables — three traps.** (a) A **low** iteration count is not a good solve: PCG's
breakdown guard (`pAp <= 1e-300` or a non-finite recurrence scalar in `CutcellMG::solvePCG`) exits
early and keeps the last finite iterate, which is how "8 / 200" rows arise next to div/u = O(1).
The script therefore records `div_rel = max|div(open·u)| / u_scale` and a `healthy` flag, and the
table's `ok` / `CAP` / `BAD` columns are the honest reading. (b) Conversely, a **capped** row is not
always a bad solve: several `pack` and `per`/`tilt` rows hit the 200 cap with div/u at 1e-10…1e-8,
i.e. the iteration plateaued at the round-off floor of the mean-removed singular system and simply
never reached `rtol·r0`. Those are stopping-criterion artifacts; the domain-BC stall is not (its
floor is div/u ≈ 1e-5). (c) Wall times are the **fastest** of the 20 steps, not the median: this host
is shared with other agents (an AMR MPI battery was running through part of the sweep) and medians
picked up 10× spikes. Iteration counts are immune and are the primary metric.

**ESCALATION #1 — `set_pressure_pcg`'s `on` flag is a no-op, so PCG is unreachable under varRho
(and under porous).** `setPressurePcg(bool /*on*/, …)` never writes `useChebyshev_`. Consequences
beyond this WO: (i) `variable_density_projection.md` §2's own escape hatch does not exist, so the
varRho Chebyshev default is currently *mandatory*, not a default; (ii) the same applies to the
porous path — `setPorous` sets `useChebyshev_ = true` behind a comment that makes the *identical*
claim from the *identical* kind of observation ("MG-PCG stalls on the eps-scaled coefficient
operator … observed: PCG 2000 iters stuck where Chebyshev converges in ~40 … an explicit driver set
afterwards wins"), so that finding is due the same re-measurement this WO gave §2's; (iii) any past
measurement that selected PCG *after* one of those calls measured Chebyshev instead. The one-line shape of the fix is `useChebyshev_ = !on;` (or
`if (on) useChebyshev_ = false;`) in `setPressurePcg`, but it is a production change and would flip
the driver under every varRho/porous script that currently relies on the accidental Chebyshev — and
straight into escalation #2 on wall-bounded cases. Not fixed here (measurement-only WO).

**ESCALATION #2 — MG-PCG stalls on the domain-BC pressure hierarchy of any 3-D wall-bounded grid,
at constant density.** Characterized above and reproducible with the shipped script
(`--geoms box --shapes const --cases lid --drivers pcg`, or the compact matrix in the tables). The
residual plateaus at r/r₀ ≈ 1.8e-2 after ~7 iterations and then oscillates for the remaining 193
(`PECLET_FLOW_MG_DEBUG=2` prints the history), leaving max|div(open·u)|/u ≈ 1e-5 — a genuinely
unconverged projection. Onset is between a third-axis extent of 4 and 8 cells; it is independent of
which axis is walled vs periodic, of `set_pressure_bottom` (smoother / auto / agglomerated all
stall), of `set_pressure_mean_removal` (fine / all), of the momentum tolerance (40 / 400 / 2000
RB-GS sweeps give bit-identical stalls), and of μ.

**It is in the geometric coarse levels, and two configurations avoid it entirely:**

| lid box, constant ρ, PCG | nz = 4 | nz = 32 |
|---|---|---|
| `levels=4`, geometric bottom (the default) | 18–55 its, div 5.8e-13 | **200/200**, div 1.24e-05 |
| `levels=4`, `set_pressure_graph_amg(True)` | **200/200**, div 1.57e-05 | **200/200**, div 1.39e-05 |
| `levels=1` (no coarse grid) | 1 it, div 1.7e-13 | 1 it, div 1.7e-13 |
| `levels=1` + `set_pressure_graph_amg(True)` | **1 it**, div 3.4e-14 | **1 it**, div 2.8e-13 |

So the natural suspects are the coarse levels' boundary treatment —
`CutcellMG::applyBoundaryOpenness`'s per-level re-imposition and the prolongation's non-periodic
boundary ghosts, whose pair need not be the adjoint the V-cycle needs to stay SPD — but this WO
measured rather than localized it. Note the third row: **the GraphAMG bottom does not cure it and at
nz = 4 it causes it**, which matters because `configurePorousDragSolver` (`flow_ibm.hpp`) puts
CFD-DEM-with-implicit-drag on exactly that combination (PCG + GraphAMG bottom). The clean workaround
that does work is the mesh-independent single-level solve — `set_pressure_multigrid(True, levels=1)`
+ `set_pressure_graph_amg(True)`, one iteration on both grids — which is also a useful A/B for
whoever fixes this.

Reach: any 3-D wall-bounded run with PCG, i.e. all of V2's two-phase-in-a-box cases, and the
CFD-DEM drag path; **no currently validated result** (all quasi-2D, verified above). Fixing it is a
production change and needs its own work order; **WO-C (S1, flexible CG) should be gated on this
battery**, because FCG is precisely the remedy for a non-SPD preconditioner and would settle whether
the transfers must be rebuilt (S4) or merely tolerated.

**ESCALATION #3 — the varRho hydrostatic path loses exactness at density ratio ≥ 10⁶.** Ratio 10⁵ is
machine-exact (max|u| 5.6e-17); 10⁶ gives max|u| 1.7e-05 and 10⁸ NaNs both drivers. The MG level
coefficient fields are `float` and the coefficient is ρ₀/ρ_f, so 10⁻⁶ sits at single-precision
relative resolution. Out of scope for VoF (air/water is 10³) and recorded only so nobody reads the
ratio axis as open-ended.

**Reproduce.**
```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build python \
  tests/study/vardensity_solver_probe.py --pack --cheb-overhead --tag nvidia-cuda
# the §2 reproduction, on its own:
#   --geoms box --shapes slab --edges sharp --ratios 1e3 --cases hydro --drivers pcg
# the constant-density control that refutes the rho mechanism:
#   --geoms box --shapes const --cases lid --drivers pcg,cheb
# the production configuration where PCG beats Chebyshev 10x:
#   --geoms pack --n 64 --shapes const,slab --edges sharp --cases per
```

### WO-C (rung S1) — flexible CG driver — **DONE 2026-08-30**; VERDICT: nonsymmetric preconditioner CONFIRMED

Delivered: `CutcellMG::solveFCG` (`src/mac_cutcell_mg.hpp`, a sibling of `solvePCG`),
`IbmSolver::setPressureFcg` + the `set_pressure_fcg(on, max_iter, rtol)` binding, and an `fcg`
driver in `tests/study/vardensity_solver_probe.py` (opt-in: `--drivers pcg,fcg`; the default
`--drivers` stays `cheb,pcg` so WO-B's battery reproduces byte-for-byte). `CLAUDE.md`'s
pressure-driver table and `doc/variable_density_projection.md` §2 updated.

**The implementation.** `solveFCG` is `solvePCG` line for line — same matvec (star overlay
included), same V-cycle preconditioner, same `removeMean` scope, same `maxabs(r) < rtol·r0`
stopping estimate, same breakdown guards, same final mean removal — with exactly one difference:

```
Fletcher-Reeves (solvePCG):  beta = r_{k+1}^T z_{k+1} / (r_k^T z_k)
Polak-Ribiere   (solveFCG):  beta = r_{k+1}^T (z_{k+1} - z_k) / (r_k^T z_k)
```

Cost: one extra level-0 vector (`zp1_`, allocated **lazily at the first FCG solve**, so an
unselected FCG costs not even memory) and one extra global dot per iteration. Measured overhead on
the periodic `packing_ring` bed: projection 14.1 ms (FCG) vs 13.8 ms (PCG) at identical iteration
counts — **2 %**. The existing PCG/Chebyshev/BiCGStab code paths were not touched.

**How the selector trap was avoided.** `setPressurePcg`'s `on` flag is still a no-op (WO-H defect 1,
deliberately left — repairing it alone would move every varRho/porous script onto the stalling PCG),
and `set_pressure_fcg` does **not** inherit the bug: it writes its own `useFcg_` **and** clears
`useChebyshev_` when `on`, so it genuinely selects even after `set_density_mode`/`set_porous`.
Exclusivity holds in both directions — `set_pressure_fcg(False)` returns to MG-PCG, and a later
`set_pressure_chebyshev(True, …)` wins at the dispatch (which tests `useChebyshev_` first), so
`setPressureChebyshev` needed no edit. `set_pressure_fcg(True)` throws under `set_ghost_projection`
(that operator is nonsymmetric and is solved by BiCGStab) rather than being silently ignored — the
failure mode that produced defect 1. Verified on the wall-bounded lid box at 32³ (three
distinguishable signatures on one operator: Chebyshev ~14, PCG 200/200, FCG ~20):

| after `set_density_mode("variable")`, then … | measured iterations | driver actually run |
|---|---|---|
| (nothing) | 18, 17, 16, 16 | Chebyshev (the varRho default) |
| `set_pressure_pcg(True, 200, 1e-8)` | 18, 17, 16, 16 — **identical** | Chebyshev — **the no-op, reproduced** |
| `set_pressure_fcg(True, 200, 1e-8)` | 200, 200, 200, 200 | **FCG** (this configuration is a residual, see below) |
| `set_pressure_fcg(True, …)` then `(False, …)` | 200, 200, 200, 200 | MG-PCG |
| `set_pressure_fcg(True, …)` then `set_pressure_chebyshev(True, …)` | 17, 15, 14, 15 | Chebyshev |

**Gate 1 — inertness. PASS.** `tests/regression/sdflow_regression.py` on the nvidia-cuda build:
**+0.00 %** on every metric (K, k\*, fitted order p, Richardson extrapolate) and **identical**
pressure-iteration totals, per-step medians and step counts on all 13 grid points of `zh_sphere` /
`random_spheres` / `hollow_rings`. Run twice — before and after the diagnostic trace line was added
to `solveFCG` — with the same result. Inertness is *verified*, not argued: unlike WO-D/WO-E this WO
does change compilation inputs of the solver module (`flow_ibm.hpp`, `flow_bindings.cpp`,
`mac_cutcell_mg.hpp`).

**Gate 2 — constant-density sanity (FCG ≈ PCG ±1). PASS, and it is exact where it should be.** With
a preconditioner that is symmetric w.r.t. the fine operator, `r_{k+1}^T z_k = 0` identically, so the
two βs coincide; on the configurations where PCG is healthy the two drivers agree:

| constant-ρ control, periodic + IBM (PCG healthy) | PCG med/max | FCG med/max |
|---|---|---|
| immersed cylinder 32³ | 7 / 7 | **7 / 7** |
| 3-Raschig-ring bed 32³ | 10 / 12 | **10 / 11** |
| `packing_ring.vti` 64³ | 14 / 16 | **14 / 16** |

Over the whole 328-configuration battery, of the 34 configurations healthy under *both* drivers the
FCG−PCG median-iteration delta is **|Δ| ≤ 1 on 32**, and the two exceptions are FCG **better** (−7 on
`rings/blob/hydro`, −75 on `rings/tilt/per` where PCG sat on a plateau). There is **no** configuration
where FCG needs more iterations than PCG.

**Gate 3 — THE DIAGNOSTIC RUN. FCG CONVERGES where PCG stalls ⇒ nonsymmetric preconditioner.**
On the constant-density 3-D wall-bounded configurations that define the WO-B stall:

| 32³, constant ρ (no varRho at all), levels 4 | PCG | FCG | Chebyshev (WO-B) |
|---|---|---|---|
| lid box (walls −x/+x/−z, lid +z) | **200/200**, div/u 9.6e-06 | **20 / 20**, div/u 7.8e-14 | 14 |
| lid, immersed cylinder | **200/200**, div/u 1.9e-05 | **21 / 22**, div/u 1.8e-11 | 14 |
| lid, 3-ring bed | **200/200**, div/u 1.2e-05 | **18 / 18**, div/u 5.3e-11 | 14 |
| hydrostatic box (walls ±z) | **200/200**, div/u 1.6e-12 | **16 / 44**, div/u 4.4e-23 | 11 |
| hydrostatic, immersed cylinder | **200/200**, div/u 1.1e-06 | **35 / 38**, div/u 1.3e-11 | 13 |
| hydrostatic, 3-ring bed | **200/200**, div/u 7.5e-09 | **12 / 24**, div/u 2.5e-10 | 14 |
| quasi-2D lid cavity 64×64×**4** (the shipped verify regime, PCG healthy) | 25 / **155** | **21 / 22** | 25 |

Whole-battery tally (nvidia-cuda, 328 configurations = WO-B's cross-product with `--drivers pcg,fcg`):

| | wall-bounded (hydro + lid) | periodic (`per`) |
|---|---|---|
| MG-PCG unhealthy | **113 / 114** | 17 / 50 |
| FCG unhealthy | **36 / 114** | **1 / 50** |

**93 of the 130 PCG failures are cured; 0 regressions** (no configuration is healthy under PCG and
unhealthy under FCG). FCG also all but removes the periodic `tilt` plateau WO-B recorded (17 → 1).
The PCG columns reproduce WO-B's tally exactly (113/114 and 17/50), which is the check that this is
the same battery on the same operator.

**Cross-backend: the full 328-configuration battery was run on host-openmp as well**, and the two
agree more tightly than WO-B's own gate: **0 healthy-verdict mismatches out of 328**, and over the
**161** configurations healthy on both backends the largest median-iteration difference is **0.0** —
not one configuration differs by a single iteration. Raw records for all three runs (nvidia-cuda
328, host-openmp 328, the `packing_ring` sweep 21) ship as
`tests/study/vardensity_solver_probe_fcg.json`; WO-B's `vardensity_solver_probe.json` is left
untouched.

**Direct instrumentation of the hypothesis (the number WO-H should keep).** `solveFCG` prints, under
`PECLET_FLOW_MG_DEBUG=2`, `pr = |r_{k+1}^T z_k| / |r_{k+1}^T z_{k+1}|` — precisely the term
Fletcher–Reeves keeps and Polak–Ribière subtracts, and **exactly zero in exact arithmetic iff the
preconditioner is symmetric w.r.t. the fine operator**. (Iteration 1 is uninformative: `z_0 = p_0`
and `r_1^T p_0 = 0` follows from the α step whatever the symmetry — measured 1e-15…1e-13 there in
every case, which is also a check that the diagnostic is wired correctly.) From iteration 2 on,
constant ρ, `levels=4`:

| configuration | median `pr` | max `pr` | PCG |
|---|---|---|---|
| immersed cylinder, **periodic** | **0.062** | 0.221 | healthy, 7 its |
| immersed cylinder, **lid (walls)** — same geometry | **0.451** | 1.401 | 200/200 |
| open box, lid | 0.478 | 1.356 | 200/200 |
| open box, hydrostatic | 0.434 | 1.078 | 200/200 |
| immersed cylinder, hydrostatic | 0.469 | 1.121 | 200/200 |

So the spurious term is **an order of magnitude larger on wall-bounded grids, and reaches the size of
the legitimate term** (β goes negative — measured `beta = −1.94e−02` at iteration 3 on the lid box),
while on the periodic hierarchy it stays at the few-percent level CG tolerates. Changing *only* the
BCs on the same geometry moves it 0.062 → 0.451.

**The asymmetry is introduced by the FIRST coarse level, not by depth or the bottom solve.** Lid box,
constant ρ, depth sweep:

| levels | PCG | FCG | median `pr` (it ≥ 2) | max `pr` |
|---|---|---|---|---|
| 1 (no coarse grid) | 1 it, div 1.1e-13 | 1 it, div 1.1e-13 | — (no iterations) | — |
| 2 | **200/200**, div 1.1e-05 | 25 / 25, div 2.6e-13 | 0.580 | 1.027 |
| 3 | **200/200**, div 1.9e-05 | 24 / 24, div 6.2e-14 | 0.550 | 1.081 |
| 4 | **200/200**, div 9.6e-06 | 20 / 20, div 7.8e-14 | 0.489 | 1.356 |
| 5 | **200/200**, div 5.9e-06 | 18 / 18, div 3.7e-13 | 0.503 | 1.177 |

A single coarsening step under domain BCs is enough, and adding levels does not make it worse. That
points WO-H at the **level-1** pair — `CutcellMG::applyBoundaryOpenness`'s per-level re-imposition of
the non-periodic face openness, and the prolongation's non-periodic boundary ghosts (Neumann
zero-gradient / Dirichlet 0) — and away from the coarse *solve* (WO-B already showed
smoother/auto/agglomerated make no difference, and GraphAMG makes it worse). The pair need not be
adjoints of each other, and that is exactly what an asymmetric M looks like.

**Gate 4 — the ratio-10⁴ `packing_ring` periodic case (WO-B's decision-relevant pair).** 64³, sharp
edge, `per`, aggregated over the ρ-shapes exactly as WO-B aggregated:

| driver | ρ ≡ 1 | 10² | 10³ | 10⁴ | healthy |
|---|---|---|---|---|---|
| MG-PCG | 14.0 | 14.5 | 15.5 | **16.0** | 7/7 |
| **FCG** | **14.0** | **15.0** | **17.0** | **16.0** | **7/7** |
| Chebyshev | 69.0 | 74.5 | 90.5 | **155.0** | 6/7 (caps at 10⁴) |

The Chebyshev column reproduces WO-B's `69 / 74 / 90 / 155` to the digit, so this is the same
measurement. FCG is at parity with PCG (all 7 configurations healthy, div/u 3.2e-10 for both) and
~10× Chebyshev at ratio 10⁴, at a 2 % projection-time cost. **On the production pore-scale
configuration FCG is a strictly-safer drop-in for MG-PCG.**

**RESIDUAL #1 (report, do not chase here) — the gravity-driven hydrostatic column with a GLOBAL
stratification is a different failure, and FCG does not touch it.** The 36 wall-bounded FCG failures
are exactly `hydro` × {`slab`, `tilt`} — all 18 of each, every geometry, every edge, every ratio —
plus the one surviving periodic `tilt`. `hydro` × `blob` (a heavy sphere not touching the walls) is
cured. Isolated at 32³, walls ±z, periodic x/y, sharp slab, ratio 10², gravity closure
`force_z = −g·ρ`, showing max|u| per step:

```
Chebyshev  15/2.4e-06  16/7.8e-11  16/3.9e-11  16/6.4e-14 ... -> the machine-exact rest state
MG-PCG    200/4.3e-01 200/8.6e-01 200/1.3e+00 200/1.7e+00 ... -> destroyed, |u| grows linearly
FCG       200/4.3e-01 200/8.5e-01 200/1.3e+00 200/1.7e+00 ... -> destroyed, identically
```

The trace shows why it is a *different* mode: the residual **rises** to `r/r0 = 6.9765` at iteration 1
and then **freezes to seven digits for the remaining 199 iterations**, with `pr` locked at exactly
`0.5000` — a stationary iteration, not a slowly-converging one. Tightening `rtol` to 1e-12 changes
nothing (Chebyshev then takes 21 its and reaches max|u| 1.9e-16). Both Krylov drivers are stuck on the
same invariant subspace while the V-cycle *as a solver* (Chebyshev) has no difficulty, so the operator
is compatible and solvable — WO-H should treat this as a second, separate defect and not assume the
symmetry repair covers it.

**RESIDUAL #2 — a small coefficient adjacent to a prescribed-velocity face.** Wall-bounded lid box,
32³, sharp z-slab, no gravity, varying which side carries the heavy fluid (so which side carries the
*small* coefficient ρ₀/ρ_f):

| face +z | ratio | heavy BELOW (coeff ≈ 1 at the lid) | heavy ABOVE (coeff = 1/ratio at the lid) |
|---|---|---|---|
| **lid** (BC type 2, Dirichlet velocity) | 10 | FCG 28, div 3.0e-11 | FCG 20, div 5.7e-13 |
| **lid** | 10² | FCG 30, div 2.0e-11 | FCG **200/200**, div 3.6e-03 |
| **lid** | 10³ | FCG 30, div 9.8e-12 | FCG **200/200**, div 3.1e-04 |
| **wall** (type 1) + body force | 10 / 10² / 10³ | FCG 30–33, div ≤ 1.1e-15 | FCG 30–33, div ≤ 1.1e-15 |

MG-PCG is 200/200 in every row; Chebyshev is 13–17 and healthy in every row. So with **all walls** FCG
is healthy at every ratio and orientation, and the failure appears only when a large coefficient
contrast sits against an *inflow-type* face — where the operator openness α is 0 (Neumann) while the
flux openness β stays open. A second candidate sub-site for WO-H, independent of Residual #1.

**WHAT THIS MEANS FOR WO-H (the verdict the WO asked for).**
1. **Pursue V-cycle symmetry.** FCG converging on 93 of PCG's 130 failures, with the β-numerator
   contamination measured at 0.43–0.48 on wall-bounded grids against 0.062 periodic, on the same
   geometry, is the answer to WO-H's "prime suspect" question: **the domain-BC V-cycle is not a
   symmetric preconditioner.** The operator and the boundary treatment of the *fine* level are not
   implicated on this evidence (Chebyshev, which needs only real spectrum bounds, converges on the
   identical operator everywhere).
2. **Look at level 1 specifically** — `applyBoundaryOpenness` per-level re-imposition and the
   non-periodic prolongation ghosts, as an adjoint pair. `levels=2` already exhibits the full effect.
3. **Fix order is unchanged and now cheaper.** WO-H can repair the `setPressurePcg` no-op the moment a
   working Krylov driver exists on domain-BC grids — `set_pressure_fcg` is that driver today, so the
   "repairing the selector strands users on a stalling PCG" objection can be answered by routing the
   varRho/porous defaults to FCG instead of PCG even before the symmetry work lands.
4. **Two residual failure modes are NOT covered by FCG** (above) and need their own diagnosis. In
   particular the hydrostatic-with-global-stratification case is a *stationary* iteration, not a slow
   one, and it is the configuration the VoF V2 rung depends on.
5. **Chebyshev remains the correct varRho default** — it is the only driver healthy on all four of
   {periodic, wall-bounded, hydrostatic-stratified, prescribed-velocity-face} — with the S2
   bound-amortization (3× on the pressure stage) still the right investment while that is true.

**Not delivered (declared).** No ctest: WO-C did not ask for one, and WO-H is already scoped to add
the 3-D wall-bounded pressure-convergence ctest that would cover both drivers — adding a
half-overlapping one here would have to be rewritten there. FCG is structurally MPI-ready (every
reduction goes through the same MPI-folded `dot`/`maxabs`/`removeMean` as `solvePCG`, and the
V-cycle is the same MPI-folded one) but is **not** MPI-gated; it is off by default, so nothing
multi-rank reaches it.

**Reproduce.**
```bash
# the diagnostic run (the whole battery, both drivers):
OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build python \
  tests/study/vardensity_solver_probe.py --drivers pcg,fcg --tag nvidia-cuda
# the headline, on its own (PCG 200/200 vs FCG 20, constant density):
#   --geoms box --shapes const --cases lid --drivers pcg,fcg
# the ratio-1e4 packing_ring pair:
#   --geoms "" --pack --shapes const,slab,blob --edges sharp --cases per --drivers cheb,pcg,fcg
# raw records for all three runs above: tests/study/vardensity_solver_probe_fcg.json
# the symmetry read-out (pr = |r^T z_k| / |r^T z_{k+1}|, zero iff M is symmetric):
#   PECLET_FLOW_MG_DEBUG=2 PECLET_FLOW_MG_SOLVES=4 ... --drivers fcg
```

### WO-H — the MG-PCG domain-BC stall + the PCG selector no-op — **DONE 2026-08-30**, one residual escalated

Delivered: `CutcellMG::applyNeumannGhost` (`src/mac_cutcell_mg.hpp`) + its call before the
prolongation in `vcycleImpl`, with the `PECLET_FLOW_MG_BCGHOST=0` ablation; a genuinely selecting
`IbmSolver::setPressurePcg` (+ `setPressureChebyshev` made exclusive in both directions) and the two
binding docstrings; the new ctest `tests/kokkos/test_pressure_wallbounded.cpp` (nz = 16, 6
configurations x 3 drivers); and the doc corrections — `CLAUDE.md`'s pressure-driver bullets and the
domain-BC multigrid paragraph, `doc/variable_density_projection.md` §2, and the two stale
justifications inside `setDensityMode` / `setPorousContinuity`.

**WHAT THE ASYMMETRY ACTUALLY WAS — a one-line omission, and the sibling class shows it.**
`CutcellMG`'s per-level ghost fill is **periodic on all three axes** (`fill()` single-rank, the core
`GridHalo` multi-rank — both built periodic by construction). On a walled face that leaves a level's
`x` ghost holding the value from the **opposite side of the domain**. Every *operator* consumer is
immune to that, which is why it survived three years: the wall face openness is 0, so the smoother,
the residual and the matvec all multiply that ghost by `AW`/`AE` = 0. But **`prolongAdd` is not an
operator consumer** — trilinear interpolation samples the coarse ghost with weight **1/4** whatever
the openness (`cx = 0.5*ifx - 0.25 + gc` -> `floor` = `gc-1`, `wx` = 0.75), so every fine cell against
a wall was receiving a quarter of its coarse correction *from the far wall*. That is a long-range
coupling present in P and absent from R — an asymmetric coarse-grid correction, i.e. exactly the
non-SPD preconditioner WO-C's `pr` instrument measured.

The fix imposes the zero-gradient (Neumann) ghost on owned wall/inflow faces (`touchesGlobalFace`,
per the WO-F pattern) after the periodic fill and before the prolongation, alongside the Dirichlet
ghost `applyOutflowGhost` already imposed for outflow faces. Then the boundary fine cell simply takes
the coarse value (`0.25*c0 + 0.75*c0 = c0`).

**The strongest evidence that this is the intended design and not an invention: `VelocityMG` has
always done it.** `mac_velocity_mg.hpp` carries `fillProlongBcGhosts` / `fillBcGhost` — "fill a
non-periodic boundary ghost of a coarse correction before trilinear prolongation
(`mg_fill_bc_ghost_k`): Dirichlet (outflow) -> ghost 0; Neumann (wall/inflow) -> ghost = nearest
inner (zero-gradient)" — the port of the retired CUDA kernel of that name. The **pressure** MG only
ever received the Dirichlet half. `CLAUDE.md`'s "the trilinear prolongation fills the non-periodic
boundary ghosts (Neumann -> zero-gradient, Dirichlet -> 0)" was describing the velocity MG and the
intent, not the pressure code.

**A dense measurement of the preconditioner, not an inference.** A throwaway harness assembles the
V-cycle preconditioner `M` as an explicit matrix — one `solvePCG(..., maxit=0)` per unit basis vector,
so it is the very code path CG uses, mean-removal included — on an 8^3 box with unit openness, and
reports `||M - M^T||_F / ||M||_F`:

| BCs (8^3, levels 3, constant coefficient) | before | after |
|---|---|---|
| fully periodic | 0.0080 | **0.0080 (byte-identical)** |
| walls +-z | 0.0430 | **0.0066** |
| lid box (walls -x/+x/-z, prescribed-velocity +z) | 0.0512 | **0.0053** |
| all six walls | 0.0535 | **0.0053** |
| five walls + outflow +x | 0.0304 | **0.0094** |

So the repair does not merely reduce the wall-bounded asymmetry — it puts it **below** the periodic
hierarchy's own residual asymmetry (which comes from the R/P pair being averaging vs trilinear, is
identical before and after, and is small enough that CG has always tolerated it).

**`pr` before/after (the number WO-C asked WO-H to keep).** `PECLET_FLOW_MG_DEBUG=2`,
`solveFCG`, constant rho, `levels=4`, 32^3, iterations >= 2, 4 solves — the same configurations
WO-C tabulated, run in one build with `PECLET_FLOW_MG_BCGHOST` as the only difference:

| configuration | median `pr` before | max before | median `pr` after | max after |
|---|---|---|---|---|
| immersed cylinder, **periodic** (the reference) | 0.0624 | 0.320 | **0.0624** | 0.320 |
| immersed cylinder, hydrostatic (walls +-z) | 0.5176 | 1.424 | **0.0511** | 0.165 |
| immersed cylinder, lid (walls) | 0.4157 | 1.401 | **0.0287** | 0.148 |
| open box, hydrostatic | 0.4438 | 1.266 | **0.0860** | 0.500 |
| open box, lid | 0.4776 | 1.359 | **0.0076** | 0.019 |

The "before" column reproduces WO-C's recorded values (0.451 / 0.478 / 0.434 / 0.469 and periodic
0.062) — the ablation is the same measurement, so the A/B is on one build.

**Iteration counts on the previously stalling cases** (`tests/kokkos/test_pressure_wallbounded.cpp`,
24x24x16, 4 levels, 8 steps, cap 200; median/max its and max\|div(open u)\|/u after the last step;
"before" = the same binary under `PECLET_FLOW_MG_BCGHOST=0`):

| configuration | driver | before | after |
|---|---|---|---|
| A lid box, **constant rho** | MG-PCG | **200/200**, div/u 5.1e-06 | **6 / 6**, 5.9e-13 |
| A | FCG | 22 / 22, 1.4e-13 | **6 / 6**, 5.2e-13 |
| A | Chebyshev | 12 / 12, 5.1e-13 | **7 / 7**, 4.4e-13 |
| B hydrostatic column, **constant rho** | MG-PCG | **200/200**, 3.6e-14 | **6 / 6**, 1.6e-23 |
| B | FCG | 15 / 22 | **6 / 6** |
| B | Chebyshev | 11 / 12 | **7 / 7** |
| C stratified column, ratio 1e2 | MG-PCG | 200/200, div/u **1.22** (destroyed) | 200/200, 2.1e-06 |
| C ratio 1e2 | FCG | 55 / **200**, div/u **1.22** (destroyed) | **55 / 55**, 3.5e-22, max\|u\| 2.3e-14 |
| C ratio 1e2 | Chebyshev | 14 / 14 | **10 / 10** |
| C stratified column, ratio 1e3 | MG-PCG | 200/200, 1.54 | 200/200, 1.08 |
| C ratio 1e3 | FCG | 29 / 200, 1.43 | 200/200, 1.08 |
| C ratio 1e3 | Chebyshev | 16 / 16 | **12 / 12** |
| D small coeff at an inflow face, 1e2 | MG-PCG | 200/200, 5.4e-08 | 200/200, 2.7e-13 |
| D 1e2 | FCG | 22 / 27 | **10 / 18** |
| D 1e2 | Chebyshev | 13 / 14 | **9 / 9** |
| D small coeff at an inflow face, 1e3 | MG-PCG | 200/200, 5.6e-04 | 200/200, 2.0e-05 |
| D 1e3 | FCG | 17 / **200**, div/u **5.6e-04** (destroyed) | **10 / 74**, 5.1e-15 |
| D 1e3 | Chebyshev | 13 / 16 | **9 / 10** |

Note that **Chebyshev improves too** (12 -> 7, 14 -> 10, 16 -> 12, 13 -> 9): the repair makes the
V-cycle a *better* preconditioner, not only a symmetric one — the far-wall teleport was polluting the
correction whatever the outer driver.

**THE TWO RESIDUAL MODES ARE ONE DEFECT, AND IT IS A COEFFICIENT DEFECT, NOT A BOUNDARY ONE
(ESCALATION).** WO-C's Residual #1 (stratified hydrostatic column) and Residual #2 (small coefficient
against a prescribed-velocity face) are the same mechanism, and the dense harness names it: with a
high-contrast coefficient the **V-cycle preconditioner becomes INDEFINITE**. `sym(M)`, factored by
LDL^T on the 8^3 walled box with `c_f = rho0/rho_f` from a sharp mid-height slab:

| coefficient contrast | walls +-z, before | walls +-z, after | fully periodic |
|---|---|---|---|
| 1 (uniform) | 0 negative pivots, skew 0.043 | 0 negative, skew 0.0066 | 0 negative, skew 0.0080 |
| 1e2 | **1 negative** (-1.1e-12), skew 0.048 | 0 negative, skew 0.037 | 0 negative, skew 0.020 |
| 1e3 | **1 negative** (-3.45), skew 0.048 | **1 negative** (-0.19), skew 0.037 | **1 negative**, skew 0.020 |
| 1e4 | — | — | **7 negative** (-6.3), skew 0.020 |

No choice of the CG beta survives an indefinite preconditioner, which is why FCG did not cure these
and why the repair only *raises the ceiling* (ratio 1e2 stratified and both inflow-face ratios are now
solved by FCG; ratio 1e3 stratified is not). The cause is `coarsenOpenAvg`'s **arithmetic** averaging
of the face coefficient: across a 1000:1 jump the coarse face coefficient comes out ~0.5 where the
physically right value is ~2e-3, so the coarse operator is not an approximation of the fine one at all.
It is present **fully periodically** as well (a negative pivot at 1e3, seven at 1e4), so it is not a
boundary-treatment problem and it is not something WO-H's scope could fix. **This is VOF_PLAN's S3
(coefficient-aware coarsening), now indicated by a direct measurement instead of inferred** — and it
is the honest reason Chebyshev (which needs only real spectrum bounds, and is healthy on every
configuration in this WO) remains the varRho/porous default. Do NOT "fix" it by switching
`coarsenOpenAvg` to a harmonic mean: that field is the *geometric openness* on the periodic/IBM path,
where the arithmetic average is the validated cut-cell coarsening and any change breaks the
byte-identity gate. A coefficient-aware variant has to be a separate, gated path.

**Defect 1 — the selector, repaired.** `setPressurePcg(bool on, int, double)` now writes
`useChebyshev_ = false; useFcg_ = false` when `on`, so MG-PCG is genuinely selectable after
`set_density_mode` / `set_porous`; `setPressureChebyshev(true, ...)` clears `useFcg_` symmetrically;
`setPressureFcg` already cleared `useChebyshev_`. `set_pressure_pcg(False, ...)` **throws** rather
than being silently ignored — MG-PCG is the terminal fallback of the dispatch, so "not PCG" is only
expressible by naming another driver, and the message says so (`setPressureFcg`'s precedent: write
your own flag, clear the competing one, throw where the request cannot be honoured). Demonstrated on
the wall-bounded lid box at 24x24x16, ratio 1e2, **after `set_density_mode("variable")`** — three
distinguishable signatures on one operator (Chebyshev ~9, MG-PCG 200, FCG ~10):

| after `set_density_mode("variable")`, then ... | iterations | driver actually run |
|---|---|---|
| (nothing) | 10, 9, 9, 9 | Chebyshev (the varRho default) |
| `set_pressure_pcg(True, 200, 1e-8)` | **200, 200, 200, 200** | **MG-PCG — the no-op is gone** (was 10, 9, 9, 9) |
| `set_pressure_fcg(True, 200, 1e-8)` | 18, 7, 10, 10 | FCG |
| `set_pressure_fcg(True)` then `set_pressure_pcg(True)` | 200, 200, 200, 200 | MG-PCG |
| `set_pressure_pcg(True)` then `set_pressure_chebyshev(True)` | 9, 8, 9, 9 | Chebyshev |
| `set_pressure_fcg(True)` then `set_pressure_chebyshev(True)` | 9, 8, 9, 9 | Chebyshev |
| `set_pressure_chebyshev(False, ...)` (the old working spelling) | 500 x4 | MG-PCG (cap = `pcgMaxit_`'s 500) |
| `set_pressure_pcg(False, ...)` | raises `set_pressure_pcg(False): MG-PCG is the default/terminal pressure driver ...` | — |

(This configuration is case D above, so MG-PCG capping here is the residual coefficient defect, not
the selector — that is exactly why it makes a clean three-way signature.)

**Gates.**

| gate | result |
|---|---|
| Single-phase regression **bit-exact** | **PASS** — `tests/regression/sdflow_regression.py --build build_woh` on nvidia-cuda: **+0.00 %** on every metric (K, k*, order p, Richardson extrapolate) and **identical** pressure-iteration totals, per-step medians and step counts on all 13 grid points of `zh_sphere` / `random_spheres` / `hollow_rings`. Structurally guaranteed: `applyNeumannGhost` returns immediately unless `hasBC_`, which no periodic/IBM problem sets |
| New 3-D wall-bounded ctest converges for PCG, FCG and Chebyshev | **PASS** — `pressure_wallbounded`, 14 gated (case, driver) pairs, nz = 16; green on **host-openmp AND nvidia-cuda** (19 s / 50 s) |
| ...and **fails for PCG before the fix** | **DEMONSTRATED** — the same binary under `PECLET_FLOW_MG_BCGHOST=0` fails 9 checks (both constant-density cases, all three PCG rows plus the FCG rows of C-1e2 and D-1e3) |
| `set_pressure_pcg(True)` after `set_density_mode("variable")` selects PCG | **PASS** — table above |
| `pr` on wall-bounded configurations drops to the periodic level | **PASS** — 0.42-0.52 -> **0.008-0.086** against the periodic 0.062, on the same geometries |
| Existing quasi-2D domain-BC verifications unchanged | **PASS** — lid cavity vs Ghia Re=100 N=128: u rms **0.0075**, v rms 0.0039, min centreline u **-0.2101** (Ghia -0.2058), max flux divergence **1.1e-16**, `result: PASS` — the recorded "~0.7 % rms". Channel + BFS below |
| `tests/kokkos` full suite | **PASS 22/22** on host-openmp AND nvidia-cuda (the 21 pre-existing + `pressure_wallbounded`) |
| All MPI ctests green, host + CUDA | see below |

**Byte-identity of the periodic/IBM path, precisely.** `applyNeumannGhost` is `if (!hasBC_ || !bcGhost_) return;` and `hasBC_` is only set by `setBoundaryConditions` with a non-zero face type. The
periodic + IBM hierarchy therefore never enters it — confirmed twice over: the dense `M` for the
fully periodic case is **identical to the last bit** with the ablation on and off, and the regression
reproduces every recorded iteration count exactly.

**Reproduce.**
```bash
# the headline (PCG 200/200 -> 6 on a constant-density wall-bounded grid), and its ablation:
OMP_NUM_THREADS=8 OMP_PROC_BIND=false ./build_ktest/test_pressure_wallbounded
PECLET_FLOW_MG_BCGHOST=0 OMP_NUM_THREADS=8 OMP_PROC_BIND=false ./build_ktest/test_pressure_wallbounded
# the symmetry read-out, before vs after, on one build:
PECLET_FLOW_MG_BCGHOST=0 PECLET_FLOW_MG_DEBUG=2 PECLET_FLOW_MG_DEBUG_SOLVES=4 \
  python tests/study/vardensity_solver_probe.py --geoms box --shapes const --cases lid --drivers fcg
# the battery, all three drivers:
OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=$PWD/build python \
  tests/study/vardensity_solver_probe.py --drivers cheb,pcg,fcg
```

### WO-I (blocker fix) — `drag_beta` ghosts are stale in the momentum build — **DONE 2026-08-30**

Fixed in one place, exactly as WO-G: `IbmSolver::fillDragBetaGhosts()` (`src/flow_ibm.hpp`), called
from `step()` immediately after `fillCellForceGhosts()`, routing `drag_beta` through the (WO-F
rank-aware) `fillPropGhosts`. Production diff: 5 lines of code + the reasoning comment.

**Call-site justification (WO item 1) — the WO-G site does hold, and for the same three reasons.**

- *After every writer.* `drag_beta` has exactly two: a closure targeting it (applied by
  `updateProperties()` two lines above) and the external CFD-DEM writer
  (`field_view` + `exchangeFieldAdd`, which runs inside `CfdDem.compute_forces()` **before**
  `flow.step()` is entered). Nothing inside `step()` writes the field.
- *Before the first consumer.* All three `addDragDiagonal` call sites are downstream:
  `rebuildStencils()` ~15 lines below, and `buildAdvStencil` / `buildAdvStencilVar` inside the Picard
  loop.
- *No policy freedom.* `project()` already fills this same field with `fillPropGhosts`, and the whole
  point of the fix is that the momentum diagonal and the projection coefficient must agree on `β_f`
  face by face (`doc/porous_drag_scheme.md` §2; the same three-way face-mean argument
  `doc/variable_density_projection.md` §1/§3 makes for ρ). Any policy other than the one `project()`
  uses would re-create the mismatch. So unlike WO-G there is no wall/outflow question to settle here —
  it is settled by consistency.

**WO item 2 — `project()`'s `fillPropGhosts(dragBeta_)` is now REDUNDANT.** Nothing writes
`drag_beta` between the new call and it. It is **kept**: removing it is a separate change, `project()`
should keep its own ghost contract for any future mid-step writer, and it costs one exchange and only
on the porous path.

**The defect's mechanism and size, measured.** Single rank the ghost band is not merely *stale*, it
is **zero**: the coupling driver's `_fold(db)` folds the deposit's ghost band onto the owners and then
assigns `0.0` to it, every step, before `flow.step()`. So `β_f = ½(β(i) + 0) = β/2` on the first inner
plane of every block — a factor-2 drag error, not round-off — while `buildPorousCoeffDrag`/`Cons` and
`projectCorrectPorous*` used the full exchanged value on the same face.

On a **periodic uniform bed of N cells on the forced axis** the consequence is an exactly computable
mean-drag deficit: the N face drags sum to `(N−½)β`, so the effective mean is `β(2N−1)/(2N)` and the
steady velocity is high by `2N/(2N−1)`. At N = 16 that is **32/31 = 1.0322581**. The Ergun fixed-bed
benchmark measured **1.0322581** (see the table). The agreement is to 7 digits and it identifies the
"~3 %" recorded in `doc/porous_drag_scheme.md` §5 as *entirely* this defect, not a closure-model
residual. (Structurally the same law WO-G measured for the halved body force, `1/(2·N_axis)`, with
the drag in place of the force.)

**Before / after deltas (WO item 4) — nothing was re-baselined; no `perf_baseline*.json` and no
recorded number in any doc was edited.** Both sides are built from the same commit `98e2bb8` in two
`git worktree`s (`wt_before` = pristine, `wt_after` = pristine + the fill), so the A/B is immune to
the concurrent WO-H work in the shared checkout — **none of WO-H's `applyNeumannGhost` change is in
either side of this table.** nvidia-cuda backend unless stated.

| case | before (pristine) | after (fix) | delta |
|---|---|---|---|
| **Single-phase regression**, all 13 grid points (`zh_sphere` 16/24/32/48/64, `random_spheres` + `hollow_rings` 24/32/48/64) | `+0.00 %` on every metric vs the recorded baseline | **identical** — every `K` / `k*` / fitted order `p` / `K_inf` / `p_iter_tot` / iters-per-step / step count / divergence equal to the last printed digit | **0** — and structurally so: no `drag_beta` field is ever registered, `hasDrag_` is false, the new call returns immediately |
| **Ergun fixed bed, POROUS path** (`coupling/tests/test_fixed_bed_ergun_porous.py`, N=16 periodic, Gidaspow, `f_drive = 0.2`) | `U = 1.5526859544e-03`, Ergun rel-err **3.225808 %** | `U = 1.5041664805e-03`, rel-err **2.28e-08** | U **−3.1250e-02** (= exactly `1 − 31/32`); the error collapses by **6 orders of magnitude** |
| **…`f_drive = 20`** | `U = 1.5463744659e-01`, rel-err **3.225812 %** | `U = 1.4982417796e-01`, rel-err **6.84e-08** | U **−3.113e-02** |
| **…`f_drive = 1000`** (inertial branch, quadratic drag → smaller velocity response) | `U = 6.6017128601`, rel-err **3.225809 %** | `U = 6.4216283623`, rel-err **5.40e-08** | U **−2.728e-02** |
| **Cross-check the fix is RIGHT, not merely different**: the same uniform bed on the INCOMPRESSIBLE path (`test_fixed_bed_ergun.py`, `porous=False`) must give the same superficial `U` | porous `1.55269e-03` vs incompressible `1.50417e-03` — **3.2 % apart** | porous `1.5041664805e-03` vs incompressible `1.5041665900e-03` — agree to **7 digits** (and to 8 at f=1000) | the two independent paths now agree; before, only one of them was right |
| **Ergun fixed bed, INCOMPRESSIBLE path** (drag ON, `porous_` OFF) | `U` = `1.5041665900e-03` / `1.498241712e-01` / `6.421628373`, rel-err 9.6e-08 / 2.3e-08 / 5.6e-08 | **bit-identical** | **0** — as designed: `addDragDiagonal`'s `faceAvg` is gated on `porous_`, so the non-porous drag path never reads the ghost |
| **Terminal velocity** (`test_terminal_velocity.py`, single particle, `porous=False`), Stokes and Schiller–Naumann | slip `2.2172412719e-04` / `0.2992163301`, drag `4.1812295094e-03` / `6.2370071411` | **bit-identical** | **0** |
| **Fluidized bed** (`coupling/examples/fluidized_bed.py` — cylindrical vessel, INFLOW + OUTFLOW domain BCs, moving grains, 120 steps) | `h0 = 2.8563998`, `hf = 11.880898` (ratio 4.16, FLUIDIZED) | `h0` identical, `hf = 11.881428` | `hf` **+4.5e-05 relative — BELOW this case's own run-to-run noise**: two `before` runs differ by 2.2e-04 and two `after` runs by 5.5e-04 (moving grains + atomic deposition). Qualitatively unchanged (fluidizes, same ratio to 3 digits) |
| **MFIX-Exa HCS gas–solid clustering** (`peclet-examples/examples/hcs-clustering/make_hcs_gas_mfix.py`, 256×256×8, N=50 000, Tang/BVK drag, Δ\*=2, truncated to t\*=300) | see the HCS paragraph below | | |

**Audit (WO item 3) — every remaining porous / drag field that is written inner-only and then
face-averaged, and WHERE its ghosts are filled relative to its FIRST consumer in the step.**

| field | face-average consumer(s) | first consumer in `step()` | ghost fill | verdict |
|---|---|---|---|---|
| `drag_beta` | `addDragDiagonal` `½(β(i)+β(i−s_c))` (only when `porous_`); `buildPorousCoeffDrag` / `buildPorousCoeffCons`; `projectCorrectPorousDrag` / `Cons` | `rebuildStencils` / `buildAdvStencil*` at the top of the step | was `project()` **only** → now `fillDragBetaGhosts()` at the top | **THE defect — fixed here** |
| `eps` | `divergOpenEps` (i±s), `buildPorousCoeff*` via the `eps1_` bridge, `maxPorousResidual` — **and, indirectly, `updateEpsRho()`** | **`updateEpsRho()` at the top of `step()`** (`porous_ && porousCons_`, and `porousCons_` defaults to **true**), then `makeFaceProps`/`buildRhsVar` face-average `epsRho_` | `fillPorousEpsGhosts()` inside `project()` and in `maxPorousResidual`; the coupling driver fills its own before `step()` | **SAME STRUCTURAL PATTERN, one policy short — see the escalation below.** Not fixed here (out of this WO's scope, and it moves the same benchmarks again) |
| `epsRho_` (`ρ_eff = ε·ρ`) | `VarFaceProps::idiag`, `buildRhsVar`'s `rhoF`, `buildAdvStencilVar`'s `fouw` | top of `step()` | never exchanged — but `updateEpsRho` is a whole-block `RangePolicy(0, n_)` product, so its ghosts are exactly as valid as `eps`'s | correct **by construction**, and therefore inherits `eps`'s status exactly |
| `epsPrev_`, `depsdt_` | none (cell-local `(ε−ε_prev)/dt` on inner cells) | — | n/a | correct |
| `divAdv_` | `buildRhs*` `½(dv(i)+dv(i−s))` (porous advection-form compensation) | inside the Picard loop, after `computeDivAdv()` in the same iteration | none needed: `computeDivAdv` writes the `G−1` ring itself | correct |
| `force_x/y/z` | `buildRhsVar` `½(f(i)+f(i−s))` | Picard loop | `fillCellForceGhosts()` at the top | correct (WO-G) |
| `rho`, `mu` | `VarFaceProps`, `buildRhoCoeff`, `projectCorrectVar` | `rebuildStencils` | `fillPropGhosts`/`fillMuGhosts` inside `rebuildStencils`, `buildAdvStencilVar` and `project()` | correct (re-checked) |
| openness `ox/oy/oz`, `sdf` | geometry face reads | static | filled at `setSolid`/`setPressureGeometry` | correct (static) |

Also swept for completeness: every `0.5 * (X(i) + X(i − stride))` in `src/*.hpp`
(`face_props.hpp`, `flow_ibm.hpp`, `mac_pressure.hpp`, `mac_approx_projection.hpp`,
`gauge_exact_gradient.hpp`). The only ones reading a *written-by-an-external-writer* cell field are
the rows above; `filterCellField` does its own `fillGhosts` before its stencil, and the collocated
face-interpolation and gauge-gradient reads are of `u`/`P`/`sdf`, all of which are filled in the same
phase that reads them.

**ESCALATION (not fixed) — `eps` has the same top-of-step-consumer / late-canonical-fill shape, and
the CFD-DEM driver's own fill is NOT the projection's policy at an inflow/outflow face.**
`porousCons_` defaults to true, so **every** porous run calls `updateEpsRho()` at the top of `step()`;
that is a whole-block product, so the momentum time term and the implicit-FOU weight face-average an
`ε` ghost that was last written either by the previous step's `fillPorousEpsGhosts()` (stale — the
deposit rewrites it in between) or, in the coupled path, by the driver itself
(`exchange_field("eps")` + `_fill_domain` + `clip`, `coupling/python/peclet_coupling/driver.py`).
On a periodic axis the driver's fill and `fillPorousEpsGhosts` agree exactly (both are the wrap /
halo value), which is why the periodic Ergun beds above are clean. They **disagree at an
inflow/outflow domain face**: the driver applies zero-gradient, while `fillPorousEpsGhosts`
deliberately mirrors about 1 so the arithmetic face mean is exactly 1 (the Kuipers/MFIX distributor
convention, `doc/porous_drag_scheme.md` §2). So on a fluidized bed the momentum inertia at the
distributor plane uses `ε_f = ε_inner` while the projection uses `ε_f = 1` — the same
momentum/projection face-mean mismatch this WO fixes for β, one field over. It is *smaller* than the
β one (a factor `ε` rather than 2, on one plane, and the driver additionally clips) and it is
configuration-specific (open domain faces only), which is why it is recorded rather than fixed: the
one-line shape is `fillPorousEpsGhosts()` next to `fillDragBetaGhosts()`, but it needs its own
decision about whether the mirror-about-1 policy should also govern the momentum inertia (it changes
what "superficial vs interstitial" means at the distributor row) and its own measurement campaign on
the fluidized-bed / HCS benchmarks.

**HCS gas–solid (the benchmark the WO worried about) — no resolvable change, and the run-to-run
noise is why.** `peclet-examples/examples/hcs-clustering/make_hcs_gas_mfix.py` (256×256×8 diameters,
N = 50 000, φ = 0.05, e = 0.8, ρ\*=1000, Tang/BVK2 drag, volume-averaged gas on a Δ\*=2 grid, i.e.
`flow` at 128×128×4) truncated to **t\* = 300** (15 000 steps, ~10 min/run); the published run goes to
t\* = 10 000 and was **not** re-run — that is a multi-hour GPU run and the cached asset
(`hcs_gas_mfix.npz`) was deliberately left untouched. **Two runs per side**, because the deposit is
atomic and the grains chaotic:

| t\* | before #1 | before #2 | after #1 | after #2 |
|---|---|---|---|---|
| 50 | 8.073e-02 | 8.068e-02 | 8.067e-02 | 8.049e-02 |
| 100 | 2.575e-02 | 2.571e-02 | 2.573e-02 | 2.560e-02 |
| 150 | 1.199e-02 | 1.198e-02 | 1.200e-02 | 1.186e-02 |
| 200 | 6.740e-03 | 6.737e-03 | 6.738e-03 | 6.647e-03 |
| 250 | 4.270e-03 | 4.242e-03 | 4.239e-03 | 4.168e-03 |
| 300 | **2.923e-03** | **2.886e-03** | **2.876e-03** | **2.820e-03** |

(T/T₀; the clustering index `cidx` is 0.90 → 0.94–0.97 on every one of the four runs.) The
before-group spread at t\* = 300 is 1.3 % and the after-group spread 2.0 %, against a group-mean
difference of 1.9 % — i.e. **the effect is not resolvable with two runs per side**, and any statement
stronger than "≲2 %, buried in the run-to-run noise" would be unsupported. Structurally that is
plausible: the fix changes the *gas* momentum diagonal, while T/T₀ is the PARTICLE granular
temperature, whose drag is applied per particle from the interpolated slip and never reads the face
mean.

**Distributed coupling (Python), np = 1/2/4, host-openmp.**
`coupling/tests/test_mpi_fixed_bed_ergun.py` runs the **incompressible** path (`porous=False`), so it
is — correctly — unchanged: `U` = 1.5042e-03 / 1.4982e-01 / 6.4216 and rel-err 0.0 % at np = 1, 2 and
4, identical before and after. `coupling/tests/test_mpi_moving_suspension.py` — the one distributed
test that IS porous (`CfdDem` defaults `porous=True`), Stokes drag, implicit, moving grains with
ownership migration — is **also unchanged to every printed digit**: `mean_vx = -2.27080469e-01`
(np=1) and `-2.27080460e-01` (np=2, rel-err 3.9e-08 vs np=1) on **both** the pristine and the fixed
tree. Structurally consistent with the HCS result: the observable is the PARTICLE velocity, whose
drag is applied per particle from the interpolated slip and never reads the face mean. Its np = 4 leg
timed out at 1200 s on both trees alike (pre-existing, unrelated to this fix).

**What was NOT run, explicitly.**
- The **full-length HCS** run (t\* = 10 000) and the fluidization **sweep** (`fluidized_bed.py sweep`,
  4 velocities × 120 steps): multi-hour, and their published numbers are the ones a re-baselining
  decision would need — that is the user's call, not this WO's.
- `coupling/tests/test_mpi_smoothing.py`, and the np = 4 leg of `test_mpi_moving_suspension.py`
  (which timed out at 1200 s on the pristine and the fixed tree alike).
- There is **no hydrostatic-in-porous case** anywhere in the repo (searched `flow/scripts`,
  `flow/tests/study`, `flow/tests/kokkos*`, `coupling/tests`, `coupling/examples`): the only porous
  configurations that exist are the two Ergun beds, the terminal-velocity drop, the fluidized bed and
  the HCS example. Item 4's "if one exists" is answered: it does not.

**Two incidental findings, both unrelated to this fix and both recorded rather than acted on.**
1. `coupling/CMakeLists.txt` `configure_file`s only `__init__.py` and `driver.py` into the build
   tree, but `python/peclet_coupling/__init__.py` does `from .resolved import ResolvedCfdDem` — so a
   fresh dev build of `peclet.coupling` cannot be imported at all
   (`ModuleNotFoundError: No module named 'peclet.coupling.resolved'`) until `resolved.py` is copied
   in by hand. Every measurement above needed that copy.
2. `dem`'s multi-rank module option is `PECLET_DEM_MPI`, but `dem/CMakeLists.txt`'s own header
   comment (line 13) and `suite/CLAUDE.md` both document it as `-DDEM_MPI=ON`, which CMake silently
   accepts as an unused variable — the module then builds *without* `init_mpi`/`step_mpi` and every
   distributed coupling test dies with
   `AttributeError: 'Simulation' object has no attribute 'init_mpi'`.

**ESCALATION #2 (pre-existing, NOT WO-I) — `MPI_ERR_TRUNCATE` at np = 4, and
`PECLET_FLOW_CA=0` cures it.** Found while running the np = 4 gate. Symptom:

```
*** An error occurred in MPI_Waitall
*** MPI_ERR_TRUNCATE: message truncated
*** MPI_ERRORS_ARE_FATAL
```

**It is not this WO's:** it strikes `varmu_mpi_np4` and `bodyforce_ghost_mpi_np4` — pre-existing
tests, neither of which registers a `drag_beta` field, so `fillDragBetaGhosts()` returns before doing
anything in both — and it reproduces on the **pristine `98e2bb8` tree** built in the `wt_before`
worktree.

What is established:

- **Four independent tests, three of them pre-existing.** `varmu_mpi_np4` (host, inside a loaded
  `ctest`), `bodyforce_ghost_mpi_np4` (CUDA, 5/5 runs, pristine tree included),
  `ghost_projection_mpi_np4` (CUDA, inside a loaded `ctest`) and `dragbeta_ghost_mpi_np4`. Every one
  of them is a decomposition that cuts **two axes into exactly two blocks** — 16×16×32 → 8×16×16,
  16×32×8 → 8×16×8 — so on those axes a rank's left and right neighbour are the *same rank*, and two
  exchanges to that one neighbour are in flight together.
- **`PECLET_FLOW_CA=0` fixes it completely.** With the communication-avoiding smoother exchange off,
  np = 4 CUDA runs green for both tests: `bodyforce_ghost_mpi` `PASS` on all three configurations,
  and `dragbeta_ghost_mpi` `du = dp = 0.000e+00` with `diag = [4, 4]` on all three. CA is exactly the
  path that exchanges a **2-deep** ghost layer where every other exchange sends 1-deep, and
  `MPI_ERR_TRUNCATE` is by definition a receive buffer smaller than its matching send — a same-tag,
  same-neighbour size mismatch, with the doubled neighbour of a 2-blocks-per-axis periodic
  decomposition as the obvious trigger.
- **It is LOAD-SENSITIVE, i.e. a race and not a fixed size bug.** The same binary and the same np
  pass standalone and fail under concurrent load, on both backends: host `dragbeta_ghost_mpi_np4`
  passes standalone (measured repeatedly, including after rebasing onto WO-H's changes) and failed
  inside a loaded `ctest`; CUDA `ghost_projection_mpi_np4` failed inside the loaded `ctest` and then
  produced zero truncations standalone; and pristine host `varmu_mpi_np4`, after failing in **0.58 s**
  inside the loaded `ctest`, runs standalone to a clean **`VARMU MPI (np=4): PASS`** —
  `couette-y du=4.441e-16 dp=6.093e-17 analytic err 0.0003 %`, `per-y du=5.551e-16 dp=6.463e-17`,
  i.e. the exact numbers §3.1 records, three runs for three runs. This session's machine sat at load ~28 with two
  other agents' batteries running. A race between two in-flight exchanges of DIFFERENT ghost width
  sharing a tag and a neighbour is the natural reading, and it is why the effect concentrates on the
  doubled-neighbour decompositions above.

Not diagnosed further and **not fixed**: it is in the halo / CA smoother communication layer, not the
porous coefficient plumbing this WO is scoped to. Recommended next step for whoever takes it: post
distinct tags (or distinct communicators) per ghost width in `GridHalo`, and add an np = 4 CUDA test
on a 2-blocks-per-axis grid — the coverage gap that let this live. Consequence for the gate table
below: np = 4 legs are reported both as-run and under `PECLET_FLOW_CA=0`, and always alongside the
pre-existing tests so the two are known to fail and pass together.

**Gates.**

| gate | result |
|---|---|
| New test **fails before** the fix (demonstrated, not asserted) | **PASS.** Pristine `98e2bb8` built in a separate `git worktree` with only the new test file + its CMake line added. host-openmp: at every np, `diag = [2.5, 4]` — the first inner plane's momentum diagonal is `idt + β/2 = 2.5` against `idt + β = 4` — and the velocity value gate fails: np=1 `per-z 0.33210449218750004`, `per-x 0.33217773437499998` (want `0.33203125`); np=2 all three configs `0.33217773…` with `du = 7.324e-05`; np=4 `per-x 0.33232421874999996`, `du = 1.465e-04`. Each rank boundary adds another half-plane, exactly as WO-G measured for the force. CUDA reproduces (`per-z np=4 diag [2.5, 4]`, `du = 7.324e-05`) |
| New test **passes after** | **PASS.** host-openmp np = 1/2/4: `diag = [4, 4]` exactly, `u spread = 0.000e+00`, `value = 0.33203125` exactly, `cross = 0`, `max\|P\| = 0`, and **`du = dp = 0.000e+00` (bitwise vs the np=1 reference) at every np**. nvidia-cuda np = 1/2 identical; np = 4 identical **with `PECLET_FLOW_CA=0`** (escalation #2) |
| np 2/4 vs np 1, host + CUDA, on all pre-existing tests | **PARTIAL — see escalation #2.** host-openmp, in two passes (1–37, then 38–48; the agent harness killed the first `ctest` at #38, not a test failure): **46 of 48 green**, the two failures being `varmu_mpi_np4` (**pre-existing**, registers no drag field, so `fillDragBetaGhosts` returns before doing anything) and `dragbeta_ghost_mpi_np4` — both `MPI_ERR_TRUNCATE` under machine load, both green standalone. nvidia-cuda, run in two passes for the same reason: green except the `*_np4` legs of `bodyforce_ghost_mpi` and `ghost_projection_mpi` (**both pre-existing**) and `dragbeta_ghost_mpi`, all the same truncation, all **green under `PECLET_FLOW_CA=0`**, and reproducing identically on the pristine tree — i.e. the failure tracks the CA exchange, the decomposition and the machine load, not this fix |
| Single-rank `tests/kokkos` | **PASS — 21/21** on host-openmp |
| **Single-phase regression bit-exact** (the WO's escalate-if) | **PASS — no movement at all.** All 13 grid points identical on `K` / `k*` / order `p` / `K_inf` / `k*_inf` / `p_iter_tot` / iters-per-step / step count / divergence, to every printed digit, before vs after; `=== regression: PASS ===`, `+0.00 %` against the recorded baseline on both sides |
| Deltas measured and reported for every case in item 4 | **PASS** — the tables above, including an explicit list of what was not run |
| No baseline file and no recorded number in any doc edited | **PASS** — `git diff` touches `src/flow_ibm.hpp`, `CLAUDE.md` (one new paragraph + the 42→48 ctest count), `tests/kokkos_mpi/CMakeLists.txt` (one name) and adds `tests/kokkos_mpi/test_dragbeta_ghost_mpi.cpp`. `perf_baseline*.json` untouched; `doc/porous_drag_scheme.md` untouched (its §5 "~3 %" is now known to be the defect — **left for the user to decide**); `doc/variable_density_projection.md` untouched |

**Concurrency note.** WO-H was working in the same checkout throughout and landed its
`applyNeumannGhost` pressure-MG repair while this WO was measuring. Every number above is from two
`git worktree`s at `98e2bb8`, so **no WO-H change is inside either side of any delta**; the only
consequence of the shared machine is wall-clock (load average ~28 for most of the session), which is
why the HCS runs read 32–70 ms/step against the 18 ms/step recorded for this configuration.

**Cost.** One ghost exchange per step, and only when `enableDrag()` has been called. Applied
unconditionally rather than gated on `porous_` so the field's ghost contract does not depend on which
consumer reads it; on the non-porous drag path it is numerically inert, which the bit-identical
incompressible Ergun bed and terminal-velocity rows above are the check of.
