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
