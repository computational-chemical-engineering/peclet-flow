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
   first, umbrella last). Do not push a red tree.
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

## Findings log

(append per WO on completion/escalation)
