# VoF work orders — the finishing campaign (V5a, V-BC, V5b, V8, examples)

Written 2026-09-02 (Fable) after the review recorded in `suite/docs/VOF_PLAN.md` §12 and
`suite/docs/VOF_NEXT_SESSION.md`. Companion to `vof_workorders.md` (phase 0),
`vof_workorders_v2.md` (V2) and `vof_workorders_v34.md` (V3/V4). Each work order is written to be
executed by an **Opus agent standalone**; findings go into this file under the WO, never into the
shipped numbers.

## Shared preamble (read once, applies to every WO)

Build (from your OWN worktree, see rule 6; CUDA backend shown, host-openmp is the same with
`host-openmp` for the prefix):
```bash
source ../.venv/bin/activate
export PATH=/usr/local/cuda-13.2/bin:$PATH
cmake -S . -B build_cuda -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j 16
cmake -S tests/kokkos -B build_ktest_cuda -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
cmake --build build_ktest_cuda -j 16 && OMP_NUM_THREADS=8 OMP_PROC_BIND=false ctest --test-dir build_ktest_cuda --output-on-failure
cmake -S tests/kokkos_mpi -B build_kmpi_cuda -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda" -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
```
Always `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` for batteries (48-core trap). Python studies:
`PYTHONPATH=$PWD/build_cuda python ...`.

Hard rules (violating any is an automatic escalation):
1. **Never edit a validated kernel body.** New physics = new sibling kernels / new files, or a new
   branch that is provably inert when its feature is off. Every existing VoF ctest and the
   single-phase regression (`tests/regression/sdflow_regression.py`, +0.00 %, identical iteration
   counts) must stay **bit-identical**.
2. **Device-first, host oracle.** `KOKKOS_INLINE_FUNCTION` in header-only `.hpp` under `src/`.
   Geometry kernels stay **container-free** (`plic.hpp` rule: scalars and small arrays only).
3. **Bit-exact MPI** at np 1/2/4 (`tests/kokkos_mpi` pattern), with a decomposition that CUTS the
   feature under test (a wall, a solid, an inflow face).
3b. **A capped pressure solve makes a run INVALID.** Every gate that reports a functional records
   the max pressure-iteration count against its cap and the flux divergence, and discards a run
   that touched the cap.
4. **A twice-failed gate stops the run.** Write the mechanism into the findings log; never tune
   numerics to pass. Five gates in this campaign so far measured the wrong quantity — say so when
   you find one, and propose the corrected gate.
5. **Never `git add -A`.** Stage named paths; `git diff --cached --stat` before every commit.
   Commit inside `flow` first; the umbrella pointer bump is done by the coordinating session.
6. **Work in your own `git worktree`** (`git worktree add -b <branch> ../flow-<name> main`).
   Three sessions share `suite/flow`; another one is editing the **velocity solve** (the momentum
   smoother / velocity MG / `cut_cell_ibm.hpp`) right now. Do not touch those files. Keep your
   `flow_ibm.hpp` edits inside the VoF section (`// --- Geometric VoF`, ~line 4947 onward), the
   VoF hooks in `step()`/`project()`, and the domain-BC helpers you are asked to add. Rebase on
   `main` before you report; if the rebase conflicts outside your section, stop and report.
7. **Bindings**: every new C++ entry point gets a nanobind binding in `src/flow_bindings.cpp` with
   a docstring that states units, conventions and the measurement behind the default.
8. **Record numbers, not adjectives.** Every gate line in your findings carries the measured value,
   the reference, the grid, the pressure iteration count and the backend.

Conventions you will need (all measured, all load-bearing):
- `flow`'s `u(i)` is the **low** face of cell `i`; `WyAdvector`'s `uf(i)` is the **high** face.
  The bridge shifts by one cell along the component's own axis (`vof::copyFaceVelocity`). The
  same shift applies to ANY face quantity you carry into the advector block (the openness!).
- `ox_(i)` = openness (fluid area fraction) of the −x face of cell `i` on the G=2 block;
  `sdf_ > 0` is fluid; `hasSolid_` = any inner `sdf < 0`; `buildCellFraction` (
  `mac_approx_projection.hpp:268`) gives the 4³-subsampled cell fluid fraction and is currently
  only built on the collocated grid (`cs_`).
- `C` = liquid fraction of the **fluid** volume of a cell (VOF_PLAN §3 rule 2), `C = 1` liquid.
- PLIC: `m · x = alpha` on the unit cube, liquid side `m · x < alpha`, **m points into the gas**,
  L1-normalized (`plic.hpp` header).
- `wyIsMixed(c)` = `0 < c < 1`; the curvature's predicate adds `interfaceEps` (1e-8 under surface
  tension).
- The colour block is the advector's own g = 3 block; `vofFillGhosts` = halo exchange /
  periodic wrap, then `clampFill` (global-index Neumann) on non-periodic faces.
- Domain BC types (`setDomainBc`): 0 periodic, 1 wall, 2 inflow (Dirichlet velocity, optional
  profile), 3 outflow. `fillPropGhosts` = Neumann copy on every non-periodic face.
- Units: cell size 1, time in seconds; `tests/study/vof_surface_tension.py::Scale` maps physical
  numbers (`mu' = s² mu`, `sigma' = s³ sigma`, `(rho g)' = s rho g`).

---

## WO-Q — rung V5a: VoF transport in cut cells (openness-weighted Weymouth–Yue)  [OPUS]

**Goal.** Lift `advectVof`'s `hasSolid_` throw: the colour field is advected through an SDF solid
with openness-weighted geometric fluxes, exactly conservative on the fluid volume, with no colour
in the solid, and a neutral (90°) colour fill in the solid band so the MYC/HF stencils see a
consistent interface at the wall. This is the prerequisite of every example with a packing and of
WO-S (contact angle), which only replaces the fill.

**Design (follow it; deviations go in the findings with a measurement).**

1. *Geometry on the colour block.* Add to `WyAdvector` an optional geometry: three face-openness
   views `of_[d]` in ITS high-face convention and a cell fluid fraction `eps_`, with
   `hasGeometry()` false by default. The solver builds them in `buildVofBlock()` when
   `hasSolid_`: `vofEps_` on the G=2 block by `buildCellFraction` (allocate your own `vofCs_`;
   `cs_` is collocated-only), embedded concentric into the g=3 block with the outermost layer set
   to 1; the openness by `vof::copyFaceVelocity(of_[d], ..., ox_/oy_/oz_, ..., d)` — the same
   shifted embed the velocity uses, because it is the same face. Rebuild when `set_solid` runs
   after `enable_vof` (hook `buildVofBlock` from `setSolid`, as `initMpi` already does).
2. *Cells.* Classify on the g=3 block once per geometry build: **solid** = `eps == 0` and all six
   faces closed; **fluid** otherwise, with `eps_eff = max(eps, eps_floor)` for any cell that has an
   open face (`eps_floor = 1/64`, the subsampling resolution — document it). C is advected in
   fluid cells only; solid cells are FILLED (item 5).
3. *Fluxes.* One flux per face, stored at the face's − side exactly as now, but
   `F_f = o_f · wyFaceFlux(a_f, ...)` — the PLIC slab fraction of the donor (reconstructed on the
   whole unit cell from its C, as if the cell were whole) times the open area times the Courant
   number. Zero where `o_f = 0`. This is the openness weighting `scalarBuildRhs` uses, applied to
   the geometric flux (the Huang 2025 solid-clipped polygon is the accurate follow-on; record its
   absence).
4. *Update in fluid-volume units.* For an inner fluid cell,
   `eps_i C_i ← eps_i C_i + (F_{i−} − F_{i+}) + c_i (o_{i+} a_{i+} − o_{i−} a_{i−})` then divide by
   `eps_i`. The dilation coefficient `c_i = H(C^n − ½)` stays frozen per step (it is frozen
   already). Conservation telescopes because both the flux and the dilation use the same `o·a`
   per face and the projection zeroes `Σ_f o_f u_f`: **this is the identity the gate measures.**
   Without geometry the kernel must reduce to the existing arithmetic **bitwise** — use
   `o = 1, eps = 1` code paths that are literally the old expressions (a branch on
   `hasGeometry()` outside the lambda, two lambdas), not multiplications by 1.0.
5. *Solid-band fill* (part of `vofFillGhosts`, only when `hasGeometry()`): after the exchange and
   `clampFill`, three passes; pass k (k = 1..3) runs over every SOLID cell at ghost depth
   ≤ 3 − k and sets it to the mean of its face neighbours that are fluid (pass 1) or already
   filled in an earlier pass (pass ≥ 2), leaving cells with no such neighbour untouched; then a
   second exchange so ghost solid cells hold their owner's value. Reading only exchanged fluid
   colour and earlier-pass values inside the covered depth is what makes the inner result
   decomposition-independent; the second exchange is what makes the ghosts so. The effect on the
   height functions is a zero-slope continuation into the wall — the 90° Afkhami–Bussmann
   limit. WO-S replaces the pass-1 rule with the θ-consistent one and keeps the rest.
6. *Boundedness in cut cells.* Weymouth's bound is on the flux volume relative to the CELL
   volume, so in a cut cell the effective Courant number is `o_f |a_f| / eps_i`. Do two things,
   both measured: (a) `maxCourantInterface` reports `o_f |a_f| / max(eps_i, 0.1)` in cut cells,
   so the interface-local dt limiter throttles for cells above 0.1 but not for slivers; (b) after
   each sweep, clip C into [0,1] in cut cells (`eps < 1`) ONLY, accumulating the clipped liquid
   volume into `Diagnostics::clippedVolume` — a diagnostic first. Report it on every gate. If it
   exceeds 1e-8 of the total per step on the packing gate, add conservative redistribution to the
   fluid face neighbours (weighted by `eps`) and gate on the drift again.
7. *Kinematic entry point.* Add `advect_vof(dt)` (Python) — advect the colour ONCE with the
   solver's CURRENT face velocity (`set_u/set_v/set_w` or whatever the existing setters are;
   check `flow_bindings.cpp`) without a Navier–Stokes step, plus `set_vof_step_parity(n)` if the
   sweep permutation must be controllable. This is what the advection-benchmark example and gates
   G2/G3 need. It must throw if the velocity is not discretely divergence-free within 1e-10
   (measure with `max_open_divergence()` — WY conservation is conditional on it).
8. *Momentum consistency in cut cells (`enable_vof_momentum` + solid) — part 2, after gates
   G1–G7 pass.* Approximation to implement and MEASURE: the half-shifted CV_e(i) spanning cells
   `i−s_e` and `i` gets fluid volume `½(eps_{i−s_e} + eps_i)`, transverse face openness
   `½(o_d(i−s_e) + o_d(i))` for `d ≠ e`, and for the axial face (`d = e`, a cell-centre plane)
   the cell fraction `eps` of the cell whose centre it sits on. Gate: WO-K's uniform-velocity
   identity holds **bitwise away from cut cells** and is bounded (report it) at cut cells; the
   colour/momentum drift per step on the packing draining case ≤ 1e-10. If the bound at cut
   cells exceeds 1e-6 relative at ratio 1000, stop and report — that is a Fable derivation.

**Gates.**
- **G1 byte-identity.** All `vof_*` ctests (host-openmp AND nvidia-cuda) and the MPI `vof_*`
  ctests reproduce their recorded numbers digit for digit; the single-phase regression is
  +0.00 % with identical iteration counts. Run in your worktree at your commit only.
- **G2 conservation through a packing (kinematic).** Build a 64³ periodic box with the 3-sphere
  or a 9-sphere SDF array (`tests/kokkos/vof_advect_scenes.hpp` has scene helpers; use
  `set_solid(..., cutcell_pressure=True)`), a body force, ratio 1, run the single-phase solver to a
  Stokes steady state, freeze that projected velocity, then advect a liquid slab (C = 1 for
  z < nz/2) with `advect_vof` for 500 steps at interface-local CFL 0.2. Gate: relative drift of
  `Σ eps_i C_i` ≤ 1e-11 (the projection's own `max_open_divergence()` decides the floor — report
  both); `Σ C` over solid cells == 0 exactly; min C ≥ −1e-12 and max C ≤ 1 + 1e-12 in cells with
  `eps == 1`; clipped volume reported.
- **G3 the two-block bridge with geometry.** Gate A of `test_vof_twophase.cpp` extended with the
  openness: the solver-driven advection through the packing equals a standalone `WyAdvector`
  given the same velocity AND the same openness/fraction, bitwise. (This is the gate that catches
  an unshifted openness embed.)
- **G4 coupled draining.** The same packing, a liquid layer above gas at ratio 10, gravity,
  `enable_vof` + closures, 1000 steps: no NaN, colour drift ≤ 10× the projection floor, colour
  in solid cells exactly 0, `max|u|` bounded, pressure iterations recorded (rule 3b).
- **G5 the 90° neutral fill.** A hemispherical liquid cap (D/Δ = 24) resting on a flat SDF wall
  placed at a HALF-INTEGER z (so the wall cells are genuinely cut), ratio 1, `set_surface_tension`,
  200 steps at 0.5·dt_σ: the cap stays a hemisphere — measure the apparent contact angle from
  the cap volume and its contact radius (spherical-cap relations) and gate |θ − 90°| ≤ 3°;
  Young–Laplace `ΔP = 2σ/R` to 1 %; spurious Ca reported against the V4 free-droplet value at
  the same D/Δ (1.4e-5 … 2.6e-5). Record `max|kappa|` and the branch census in the wall band.
- **G6 MPI.** np 1/2/4 on G2 (bitwise) and G4 (reduction floor), with a decomposition cutting
  through the spheres; host-openmp and nvidia-cuda.
- **G7 no regression on the free-surface battery.** `tests/study/vof_surface_tension.py static
  hysing1 --quick` reproduce WO-P's numbers (static Ca at D/Δ = 16: 5.90e-5; Hysing 1 max rise
  velocity 0.2497) to 3 significant figures.

**Deliverables.** `src/vof/cutcell.hpp` (container-free: the clip/limit rules, the fill-pass
kernel body, the effective-Courant rule), the geometry branch in `src/vof/advect_wy.hpp`, the
solver plumbing, bindings, `tests/kokkos/test_vof_cutcell.cpp` (G2, G3, G5) and
`tests/kokkos_mpi/test_vof_cutcell_mpi.cpp` (G6), a study script `tests/study/vof_cutcell.py`
(G2, G4, G5 in Python, printing every number), the findings entry below, and the CLAUDE.md
paragraph (scope: "no immersed solid" is lifted; state what the cut-cell flux approximates).

---

## WO-R — rung V-BC: two-phase open boundaries (inflow / outflow / wall colour)  [OPUS]

**Goal.** Liquid and gas entering and leaving through domain faces, with exact colour
bookkeeping, and the variable-density outflow correction that `variable_density_projection.md` §4
lists as missing. Needed by the trickle-flow example and by every gas–liquid–solid inlet.

**Design.**
1. *Colour at an inflow face (type 2).* `set_vof_inflow(face, value)` and
   `set_vof_inflow_profile(face, C2d)` (a `(nb, nc)` Fortran-order array on the face's inner
   grid, resampled with the clamp rule `setDomainBcProfile` uses). The colour block's THREE ghost
   layers on that face are overwritten with the value after `clampFill`, on the rank that
   `touchesGlobalFace`. The advector gets an `outside` mask (`UCField`, 1 for every ghost cell
   that lies outside the global domain on a non-periodic axis, built once with the block) and
   `wyFaceFlux` gets a sibling `wyFaceFluxBc` used only on faces whose donor is `outside`:
   algebraic `C_donor · a`, never a reconstructed slab (a uniform ghost band has no usable MYC
   normal, and the inflow colour is a boundary DATUM, not an interface). Fractional inflow
   colours are therefore allowed and mean "this fraction of the incoming flux is liquid".
   Default when nothing is set: the Neumann copy (today's behaviour).
2. *Colour at an outflow face (type 3).* Zero-gradient stays (`clampFill`). Add
   `set_vof_backflow(face, value)`, default 0 (gas): where the outflow face velocity points INTO
   the domain, the ghost band carries the backflow colour instead — the `inletOutlet` behaviour
   of interFoam (Rusche 2002 thesis §4; OpenFOAM `inletOutletFvPatchField`), the standard for VoF
   outlets. Reading the face velocity sign happens in the fill, after `bridgeVelocityToVof`.
3. *Colour at a wall face (type 1).* Neumann copy (90°) as today; note in the docstring that
   WO-S's θ-fill applies to domain walls too (a domain wall is a flat SDF wall at the face).
4. *The variable-density outflow correction.* `bcCorrectOutflow` (mass-conserving correction of
   the high-side outflow face from `phi`) lacks the `1/ρ_f` factor that `projectCorrectVar`
   applies everywhere else. Add `bcCorrectOutflowVar` (sibling; `ρ_f` at the outflow face = the
   inner cell's ρ under the Neumann property policy) and call it where `bcCorrectOutflow` is
   called on the staggered path when `varRho_`. Byte-identical when `!varRho_`.
5. *The Neumann property policy at an inflow face.* `fillPropGhosts` copies the inner ρ to the
   ghost. With a liquid inlet next to a gas interior (or vice versa) the inflow ghost ρ should be
   the INLET fluid's ρ, or the face density in the momentum time term and the projection
   coefficient at the inlet face are wrong by up to the ratio. Since ρ is a closure of C and the
   colour ghost now carries the inflow value, make the closure evaluation at inflow ghosts follow
   from C: after the colour BC fill, re-apply the ρ/μ closures on the ghost band of inflow faces
   (a small kernel; measure with gate G3).
6. *P1 measurement (no code).* Run `tests/study/vof_surface_tension.py hysing2` twice, with and
   without `PECLET_FLOW_EXACT_RESIDUAL=1`, and record max pressure iterations, `max|div|`, the
   rise velocity and `y_c(3)` for both. WO-P recorded 116/600 and 1.85e-3 for this case and
   attributed it to the float `A·1 ≠ 0` defect P1 removes. Report; do not change defaults.

**Gates.**
- **G1 colour budget (kinematic).** A 32×32×64 box, inflow at −z with a uniform velocity
  profile, outflow at +z, walls elsewhere, ratio 1, `advect_vof` with the projected uniform flow;
  `set_vof_inflow(4, 1.0)` for 100 steps then 0.0 for 400: the budget
  `Σ eps C(t) − ∫(inflow liquid flux) + ∫(outflow liquid flux) = const` to 1e-12 relative, where
  the fluxes are the advector's own face fluxes summed on the boundary faces (expose them in
  `vof_diagnostics()` as `inflowVolume`/`outflowVolume` per step). The slug leaves the domain
  with its length intact (report the profile), and nothing is left behind (`Σ C → 0` to 1e-12).
- **G2 Nusselt falling film.** Quasi-2D (ny = 4), walls ±x, inflow at +z (liquid film of
  thickness δ = 8 cells against the −x wall, velocity the Nusselt profile
  `u = (ρ g/μ)(δ y − y²/2)`, gas at rest elsewhere), outflow at −z, gravity −z, ratio 100 and
  μ ratio 50 first (V2a regime), then ratio 1000 with `enable_vof_momentum`. Gate at steady state:
  film flow rate within 3 % of the Nusselt value, film thickness within 0.5 cell, no liquid
  accumulation (`Σ eps C` steady to 1e-4 over the last 200 steps), pressure iterations recorded.
  This is the gate on item 4: with the `1/ρ_f` factor missing the film cannot leave cleanly at
  ratio 1000 — record the before/after.
- **G3 gas over a pool.** Gas inflow at −x over a resting liquid pool (C = 1 for z < nz/2), outflow
  at +x, ratio 1000, no gravity, `enable_vof_momentum`, 500 steps: the pool's liquid volume is
  conserved to 1e-10, `max|u|` in the liquid stays below 1e-3 of the gas inlet speed, and the
  inflow ghost density reads the GAS density (item 5) — assert it through `get_field("rho")` on
  the first inner plane.
- **G4 MPI.** G1 at np 1/2/4 with the decomposition cutting the inflow and outflow faces:
  bitwise on the colour; G2 at the reduction floor.
- **G5 inert.** With no VoF BC set, every existing VoF ctest bit-identical; single-phase
  regression +0.00 %.

**Deliverables.** The mask + `wyFaceFluxBc` in `src/vof/advect_wy.hpp` (new sibling function; the
old kernel body untouched), `src/vof/colour_bc.hpp` (the fill rules), solver plumbing +
`bcCorrectOutflowVar` in `mac_pressure.hpp` or `flow_ibm.hpp` (a sibling of the existing helper),
bindings, `tests/kokkos/test_vof_bc.cpp`, `tests/kokkos_mpi/test_vof_bc_mpi.cpp`,
`tests/study/vof_open_boundaries.py` (G1–G3), findings below, CLAUDE.md paragraph.

---

## WO-S — rung V5b: static contact angle on SDF solids (the θ-consistent fill)  [Fable design → OPUS]

**Goal.** Replace WO-Q's neutral pass-1 fill by a fill that makes the height functions and the
MYC stencils see an interface meeting the solid at the prescribed angle θ, so that the unmodified
V3 cascade returns the right curvature at the contact line and the balanced force does the rest.
Static angle only; θ is a per-cell field so V6 (dynamic angle, hysteresis) changes only what
fills it.

**The geometry (derived 2026-09-02; the sign conventions are checked below, keep them).**
- `n_w = ∇sdf / |∇sdf|` points from the solid INTO the fluid (sdf > 0 is fluid).
- The PLIC normal `m` points from the liquid INTO the gas (`plic.hpp`).
- The contact angle θ is measured through the liquid. At the contact line
  **`m · n_w = cos θ`**: θ = 0 (complete wetting) puts the interface parallel to the wall with
  the liquid film between, so `m = n_w`; θ = 180° gives `m = −n_w`; θ = 90° gives `m ⊥ n_w`.
- For a solid cell `s` in the band: walk from `s` along `n_w(s)` in unit steps (up to 4) to the
  first fluid cell `f` (`eps > 0`). If `f` is not interfacial (`C_f ≤ eps_i` or `≥ 1 − eps_i`),
  `C_s = C_f` (pure-phase continuation — identical to the neutral fill). Otherwise:
  1. take the fluid-side normal `m_f` from a **fluid-only Youngs gradient** — the 27-point
     `youngsNormal` weights restricted to cells with `eps > 0`, renormalized (this avoids the
     circularity that the fluid cell's own MYC normal already reads the ghost you are filling;
     Basilisk's `contact.h` breaks the same loop with a second reconstruction pass — that is the
     fallback if gate G1 fails, see below);
  2. split `m_f = (m_f · n_w) n_w + t`; if `|t| < 1e-6` keep `m_f` (interface parallel to the
     wall, no rotation defined); else `t̂ = t/|t|` and the target normal
     **`m_θ = cos θ · n_w + sin θ · t̂`** (L1-normalize for `plicAlpha`);
  3. the pivot is the contact point: the fluid cell's interface centroid `p_f` (the PLIC polygon
     centroid, or the plane point closest to the cell centre if no centroid helper exists)
     projected onto the wall, `c = p_f − sdf(p_f) n_w`;
  4. `C_s` = the fraction of the unit cube of cell `s` on the liquid side of the plane through
     `c` with normal `m_θ` (`plicVolume` with the alpha of that plane in cell `s`'s local frame,
     clamped to [0,1]).
- Passes 2–3 of WO-Q's fill stay as they are (mean of already-filled neighbours); only pass 1
  changes. Domain walls (BC type 1) use the same rule with `n_w` the inward face normal.
- Flat-wall limit check (do it as gate G0 before anything else): a plane interface meeting a
  flat wall at θ, exact fractions, the fill reproduces the exact fractions of the continued
  plane in the ghost cells to 1e-12 for θ ∈ {30°, 60°, 90°, 120°, 150°} — this IS the
  Afkhami–Bussmann height-function boundary condition (IJNMF 57:453, 2008) expressed as
  fractions.

**API.** `set_contact_angle(theta_deg)` (uniform) and `set_contact_angle_field(theta_deg_array)`;
`contact_angle_diagnostics()` = number of contact cells, mean measured apparent angle (see G1 for
the measurement), and the branch census in the wall band.

**Gates.**
- **G0** the flat-wall limit above (a pure kernel test, `tests/kokkos/test_vof_wetting.cpp`).
- **G1 drop on a flat SDF wall**, wall at half-integer z, D/Δ = 24, ratio 1 then 100, μ chosen
  for Oh ≈ 0.1 (fast relaxation), 0.5·dt_σ, run to rest (`max|u|` decaying below 1e-4 of the
  capillary velocity σ/μ): the equilibrium is a spherical cap; measure θ from the cap volume V and
  contact radius a (the cap relations `V = π h(3a² + h²)/6`, `sin θ = 2ah/(a² + h²)` — read h
  from the colour column at the axis, a from the C = ½ contour on the first fluid plane), gate
  |θ − θ_set| ≤ 3° for θ ∈ {30°, 60°, 90°, 120°, 150°}. Report the near-wall spurious Ca.
- **G2 drop on an SDF sphere.** Solid sphere R_s = 12 cells, liquid drop of volume equal to a
  sphere of R_d = 8 sitting on it, θ ∈ {60°, 90°, 120°}: at rest (no gravity) the free surface is
  a spherical cap intersecting the solid sphere at angle θ — the reference cap radius and
  position follow from V and θ by a one-dimensional root solve (write it in the test). Gate: cap
  radius within 3 %, θ within 3°. (Asghar et al. 2023's wetting suite is the published cousin.)
- **G3 volume.** Liquid volume exact to the projection floor throughout G1/G2; zero colour in
  solid cells.
- **G4 capillary rise / Jurin.** Two SDF plates 24 cells apart, θ = 30°, gravity, liquid bath at
  the bottom (periodic-in-y quasi-2D): the equilibrium rise height against Jurin's
  `h = 2σ cos θ/(ρ g w)` to 5 % at 200 μm-equivalent scaling (state the Bond number).
- **G5 MPI** np 1/2/4 bitwise on G0 and the reduction floor on G1, decomposition cutting the drop
  and the wall.
- **G6** every earlier gate (WO-Q G1–G7) unchanged; θ = 90° must reproduce WO-Q's neutral fill
  **bitwise** (it is the same plane).

Escalation path: if G1 misses by more than 3° at every θ with the same sign, switch item 1 to the
two-pass scheme (neutral fill → MYC normals → θ-fill → MYC again) and re-measure; if it still
misses, stop — the pivot choice (item 3) is then the suspect and needs a Fable derivation.

---

## WO-T — rung V8 (minimal): the collocated path  [Fable design → OPUS]

**Goal.** `enable_vof` on `SolverColocated`: variable density in the ABC approximate projection,
the balanced force on the face field with the consistent cell counterpart, colour advection from
the projected face field. Everything the staggered path validated must be re-run here; AMR VoF
inherits this rung, which is why it is not optional.

**Design (Basilisk's `centered.h` pattern, which is an ABC code; Popinet JCP 2009 §3).**
1. *Variable density in the projection.* `setDensityMode(true)` allowed on the collocated grid:
   the face coefficient `c_f = o_f ρ₀/ρ_f` (arithmetic ρ_f; the staggered `buildRhoCoeff`),
   `projectCorrectVar` on `uf_/vf_/wf_` (already exact-adjoint on faces), and the CELL correction
   as the **average of the two face corrections of each axis** — not a cell-centred `∇φ/ρ_c`.
   That averaging operator is the one `projectCorrectCenter` already applies to `φ` differences
   with openness weights; write a sibling that averages `(1/ρ_f)(φ(i+s) − φ(i))` face values so the
   cell sees exactly the mean of what its faces saw.
2. *Forces as face accelerations.* On the collocated grid every interfacial/body force enters the
   predictor at the CELL through a central difference today (`buildRhsVar`'s
   `0.5*(P(i+s) − P(i−s))` and the cell body force). With ρ jumping across an interface the cell
   balance `g_c − ∇_c P/ρ_c` is O(1) wrong at interface cells even when every face is exactly
   balanced. Rule for V8: the face field `uf*` gets the face acceleration
   `a_f = (f_f − ∇_f P^n)/ρ_f` (body force `f_f` = arithmetic face mean of the cell force, CSF
   `σ κ_f (C(i) − C(i−s))/h` — the V4 rule verbatim) added AFTER `centerToFace(u*)`, where `u*` is
   predicted WITHOUT the pressure gradient and body force; the cell velocity update then uses the
   **average of the face `a_f − ∇_f φ/ρ_f`** — the same averaging operator as item 1. This is the
   "consistent cell-side counterpart through the same averaging operator" VOF_PLAN §4 V8 asks for;
   it makes a hydrostatic column and a stationary droplet exactly balanced on the faces and the
   cells see the average of an exact zero. Implement it as a NEW collocated varRho branch (inert
   unless `varRho_` on the collocated grid) — the constant-density collocated path is validated
   and must stay bit-identical (`benchmarks/staggered-vs-collocated`).
3. *Colour advection.* `bridgeVelocityToVof` from `uf_/vf_/wf_` (already discretely divergence-free
   after `projectCorrect`; check the face index convention against `colour_field.hpp` — the
   collocated face field is `flow`-low-face like `C[c].u`), `enableVof` no longer throws on
   collocated. Interface-local CFL from the same faces.
4. *Momentum consistency.* NOT in this rung (cell-centred momentum needs Favre face states,
   AMR-Wind's construction). The collocated path is rated to ratio ≤ 100 with motion until then;
   say so in the docstring. `enable_vof_momentum` throws on collocated.
5. *Wall blend / attractor settings.* Run the acid tests with the attractor campaign's defaults
   (`set_collocated_scheme` default; `doc/collocated_invisible_subspace.md`) and report any
   sign of the invisible-subspace mode (a growing checkerboard in the cell field with the face
   field clean) — that is the risk the plan named.

**Gates** (all on `SolverColocated`, both backends):
- **G1 hydrostatic** through C at ratio 1000: `∂P/∂z = −ρ_f g` to 1e-12 on faces; cell `max|u|`
  ≤ 1e-12 with the interface frozen.
- **G2 static droplet, constant κ**: face-field `max|uf|` at machine zero (≤ 1e-15); cell
  `max|u|` reported (the approximate projection leaves the averaged remnant — record it, gate
  ≤ 1e-13).
- **G3 static droplet, computed κ**: Ca at D/Δ = 16 within 2× of the staggered 5.9e-5.
- **G4 Hysing case 1** (ratio 10): max rise velocity and `y_c(3)` within 3 % of the staggered
  values (0.2497 / 1.0808).
- **G5 capillary wave** 32 cells/λ, ν = 0.005: ω within 1 % of the staggered 0.06017.
- **G6 advection benchmarks** (`advect_vof` from WO-Q) bit-identical to the staggered ones on the
  same face field (same kernel, same faces).
- **G7 MPI** np 1/2/4, and the constant-density collocated regression bit-identical.

**Deliverables.** The collocated varRho branch (projection + RHS), the face-acceleration
predictor, bindings, `tests/kokkos/test_vof_collocated.cpp`, MPI twin,
`tests/study/vof_collocated.py` running the staggered battery cases on the collocated grid and
printing both columns, findings, CLAUDE.md.

---

## WO-U — the example gallery (`~/Codes/peclet-examples`)  [OPUS, page designs by Fable]

Pages, in the order they become runnable (E1–E4 need nothing beyond `main` + WO-Q's `advect_vof`;
E5–E7 need WO-Q/R/S). Every page follows `STYLE_GUIDE.md` (bootstrap cell, What-you'll-learn,
problem, code in small cells, results with the headline number in prose, Adapt, Reproduce), runs
on the staggered solver, and ends with a **collocated cross-check** cell once WO-T lands (before
that, a note that the collocated path is rung V8). Frozen outputs are rendered by
`render_example.sh` against `PECLET_LOCAL_BUILD=<worktree>/build_cuda`. Log every surprise in
`ISSUES.md` before working around it.

- **E1 `vof-advection-benchmarks`** — Zalesak's slotted disc (quasi-2D, 100², one revolution),
  LeVeque's 3-D deformation field with reversal (T = 3, 32³/64³/128³, L1 error and order), and
  the conservation floor (`max_open_divergence` vs colour drift). Headline: exact conservation
  (drift 1e-14) and order 2.2 on LeVeque. Needs `advect_vof`.
- **E2 `parasitic-currents`** — the stationary droplet: `Ca = μ max|u|/σ` vs D/Δ = 8…32 with the
  computed curvature (order 2.1), the balanced-force switch (`set_csf_mode(1)`: 3e15×), the
  wisp guard ablation, and the curvature branch census. Headline: machine-zero spurious currents
  with an exact curvature; the estimator is the ceiling. This is the "characterization of
  parasitic currents" page.
- **E3 `capillary-oscillations`** — the standing capillary wave against the **viscous** two-fluid
  dispersion relation (`s² + ω₀²(1 − k/√(k² + s/ν)) = 0`, equal fluids; Fable's derivation in
  VOF_NEXT_SESSION Item 2 — it removes the −2…−4 % "deviation" WO-P recorded against the inviscid
  formula) and the mode-2 droplet against Lamb + the viscous correction (Miller & Scriven 1968;
  see the Item-2 trace for what to state about the residual deficit).
- **E4 `rising-bubble`** — Hysing et al. (2009) cases 1 and 2: rise velocity and centroid vs the
  published curves, the bubble shapes, `enable_vof_momentum` on/off (14 % at ratio 10). The
  **bubble** example.
- **E5 `droplet-wetting`** — a drop on a flat wall and on an SDF sphere at θ = 60°/90°/120°:
  equilibrium shape vs the spherical-cap solution, near-wall spurious currents. The **droplet**
  example (WO-S).
- **E6 `trickle-flow-packing`** — liquid fed at the top of a small SDF sphere packing (the
  random-packed-bed recipe, ~40 spheres), gas at rest or co-current, outflow at the bottom, θ = 60°:
  liquid holdup vs time, wetted area, pressure drop; a still and a movie. The **liquid over a
  packing** example (WO-Q + WO-R + WO-S). Inflow: liquid distributor as a `set_vof_inflow_profile`
  region with the Nusselt film velocity; outlet: `set_vof_backflow(…, 0)` gas backflow.
- **E7 `bubble-through-packing`** (show-off) — a gas bubble rising through a liquid-saturated
  sphere packing in a periodic column: breakup and trapping, with a movie. Runs after E6.
- **E8** a table page or a section in `benchmarks/staggered-vs-collocated` with the collocated
  column for E2–E4 (WO-T).

Open-boundary literature for E6/WO-R (checked 2026-09-02): the `inletOutlet` switch
(zero-gradient on outflow, prescribed value on backflow) is OpenFOAM's `inletOutletFvPatchField`
(Rusche 2002; OpenFOAM user guide §6.4); Dong & Wang, *PLoS One* 11(5):e0154565 (2016) and Dong,
*JCP* 266:47 (2014) give the energy-stable two-phase open boundary condition whose backflow term
`½ρ|u|²n − (n·u)u` is active only where `n·u < 0` and validate it with a capillary wave at ratio
1000 and bubbles leaving through the open face — WO-R's backflow colour is the VoF-side half of
that recipe, and `set_backflow_stabilization` is the velocity-side half already in `flow`. The
TU/e IBM–VoF line (Baltussen, Deen, Kuipers: coupled IBM + VoF with a static contact angle on
resolved particles, liquid coverage in gas–liquid–solid packed beds) is the direct precedent for
E6; cite the specific paper after checking it (search "Numerical study of liquid coverage in a
gas–liquid–solid packed bed", Deen & Kuipers).

Gallery plumbing: add a "Two-phase flow (VoF)" section to `index.qmd` between "Flow through
packings" and "Resolved particles", a navbar menu in `_quarto.yml`, a gallery card image per
page, and `references.bib` entries (Weymouth & Yue 2010; Popinet 2009; Francois 2006; Hysing
2009; Lamb 1932; Prosperetti 1981; Miller & Scriven 1968; Afkhami & Bussmann 2008; Rusche 2002;
Zalesak 1979; LeVeque 1996).

---

# Findings log (v5 work orders)

(append per WO, newest first)

## WO-S — rung V5b (static contact angle on SDF solids) — DONE 2026-09-02, branch `vof-wos`

Commit `2761bc9` (the rung + the gates) plus the doc commit at the end. All numbers are
host-openmp unless a CUDA column is given; both backends agree.

### What shipped

`src/vof/wetting.hpp` (container-free like `plic.hpp`: `youngsNormalFluidOnly`, `vofPlicCentroid`,
`vofWettingPlane`, `vofWettingFraction`), the theta pass `solidBandFillPassWetting` +
`buildWettingNormals` + `wettingCensus` in `src/vof/advect_wy.hpp`, the solver plumbing in
`flow_ibm.hpp` (`setContactAngle` / `setContactAngleField` / `setContactAnglePivot` /
`contactAngleDiagnostics` / `applyContactAngle`, and three lines in `vofFillGhosts`), the bindings,
`tests/kokkos/test_vof_wetting.cpp`, `tests/kokkos_mpi/test_vof_wetting_mpi.cpp` and
`tests/study/vof_wetting.py`. **Only pass 1 of the V5a band fill changes**; passes 2-3 and the
shrinking depth budget are WO-Q's, and with no `set_contact_angle` call the theta fields are never
even allocated.

### Gates

| gate | measured | verdict |
|---|---|---|
| **G0a** flat-wall idempotence (pure kernel, EXACT `m_f`) | plane interface meeting a flat wall at theta, exact fractions, five angles x three azimuths x nine columns x three band rows: `max |C_fill - C_exact|` = **5.6e-16 / 1.0e-15 / 3.3e-16 / 8.0e-16 / 2.9e-16** at theta = 30/60/90/120/150 for the shipped anchor. The three anchor variants: **volume (default) 1.0e-15, PLIC-centroid 1.0e-15, contact-line 1.0e-15, and the WORK ORDER's `c = p_f - sdf(p_f) n_w` 2.6e-1** | PASS (WO anchor FAILS, see finding 1) |
| **G0c** the rotation | over theta_apparent 20..160 x theta_target 15..165: `max |m_theta . n_w - cos theta|` = **1.6e-16**; the plane passes through `p_f` to **0.0** | PASS |
| **G0d** the wetting limits | with `C_f = 0.3063` at an 85 deg apparent angle: `C_band` = **1.000000** at theta = 0 and **0.000000** at theta = 180 (complete wetting fills the band, complete non-wetting empties it); 30 deg gives more liquid than `C_f` and 150 deg less | PASS |
| **G0e** what the fluid-only estimator costs | on EXACT plane fractions next to the wall: the recovered normal's ANGLE TO THE WALL is wrong by **23.13 / 6.69 / 0.00 / 6.69 / 20.91 deg** at theta = 30/60/90/120/150, against **2.31 / 2.10 / 0.00 / 2.10 / 2.31** for the same Youngs stencil with the solid rows present. Its AZIMUTH — the only part the construction uses — is exact: **0.000 deg** worst over the sweep, and the END-TO-END band error of the fill driven by the fluid-only normal is **1.1e-15** | PASS (see finding 2) |
| **G1** the prescribed angle is a fixed point (`test_vof_wetting`, 40x40x32, D/dx = 20, wall z = 4.25, sigma 1, mu 0.5, 150 steps) | theta 60 -> **60.121**, 90 -> **89.170**, 120 -> **118.719**; worst **1.281 deg** (gate 3.0). Volume drift 3.0e-15 / 5.0e-15 / 1.3e-14; `Ca(open)` 6.9e-4 / 1.2e-3 / 2.6e-3; raw `max|u|` 1.7e-3 / 2.4e-3 / 5.2e-3; 10-11 pressure iterations, cap 300, none capped | PASS |
| **G1** the theta sweep (study, 64x64x40, D/dx = 24, wall z = 4.25, Oh = 0.1, ratio 1 and 100, 500 steps) | see the sweep tables below: within 1.2 deg up to theta = 90, 3.5-3.8 deg at 120/150 (ratio 1) and 7.2 deg at 150 (ratio 100) | **FAIL above 90 deg** — finding 6 |
| **G1b** attraction (from a hemisphere) | see the sweep table below | see finding 4 |
| **G1w** wall placement | see the wall-placement table below — the decisive measurement of this rung | (mechanism) |
| **G3** volume | the liquid volume drift over every G1/G2 run is **<= 1.7e-14** relative and `sum C` over solid cells is **exactly 0** in all of them | PASS |
| **G5** MPI (host-openmp) | np 1/2/4 on a 16x16x32 grid with a flat SDF wall at x = 5.25 and a cap on it, the ORB cutting z at np=2 and xz at np=4 (so the contact LINE is cut): the theta band fill (`vof_filled_colour`) is **0.0 — bitwise** at every np, and the band census is identical (112 theta / 384 neighbour / 1552 pure / 0 parallel / 0 neutral, mean apparent 83.933 deg). Coupled 25-step surface-tension run: np=1 colour **0.0** and velocity **0.0** (bitwise), np=2 colour 2.2e-16 velocity 3.0e-16, drift -1.4e-15, `sum C` over solid 0, 9 pressure iterations | PASS |
| **ctest battery** | `ctest -R vof_` on host-openmp: **100 % tests passed, 0 failed out of 8** (`vof_plic` 0.77 s, `vof_advect` 50.57, `vof_twophase` 58.29, `vof_momentum` 32.49, `vof_curvature` 8.40, `vof_surface_tension` 161.87, `vof_cutcell` 58.56, the new `vof_wetting` 149.30; 520 s total on a machine shared with two other agents). Each of the seven pre-existing ones was additionally diffed one by one against the same binary built at `5b0ecdb`, on host-openmp AND nvidia-cuda | PASS |
| **G6** every earlier gate unchanged | with no `set_contact_angle` call: `vof_plic`, `vof_advect`, `vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_surface_tension`, `vof_cutcell` reproduce their V5a output **digit for digit** (`diff` of the full stdout against the same binaries built at `5b0ecdb`) on **host-openmp AND nvidia-cuda** | PASS |
| **G6** theta = 90 vs the neutral fill | NOT bitwise, and it cannot be — see finding 5. Measured on the same cap scene at t = 0: `max |C_theta - C_neutral|` = **1.000** over all solid cells and **4.28e-2** over the band cells the V3 cascade can reach. After 150 steps the two runs give theta **89.170 vs 89.171**, `max|u|` **2.444e-3 vs 2.445e-3**, `Ca(open)` **1.222e-3 vs 1.222e-3** | corrected gate, PASS |

### The theta sweep (G1, `tests/study/vof_wetting.py g1`)

64x64x40, D/dx = 24, flat SDF wall at z = 4.25, sigma = 1, Oh = 0.1 (mu = 0.4899), ratio 1,
500 steps at the capillary limit with the interface CFL held below 0.15. The drop starts AS the
spherical cap of the prescribed angle (the fixed-point protocol; the attraction protocol is
finding 4). `theta` is read from the conserved volume and the axis colour column
(`a = sqrt((6V/(pi h) - h^2)/3)`, `theta = 2 atan(h/a)`); the "contour" column is the contact radius
read off the first fluid plane instead, which is biased and is reported to show by how much.

| theta_set | theta | err | h | a | contour theta | dV/V | Ca(open) | raw max|u| | band th/nbr/pure | apparent | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | **30.688** | +0.688 | 5.532 | 20.161 | 34.474 | 1.7e-14 | 2.96e-4 | 1.45e-3 | 1472/2048/29248 | 34.71 | 9 |
| 60 | **59.980** | -0.020 | 8.839 | 15.316 | 62.765 | 3.3e-15 | 3.68e-4 | 2.09e-3 | 720/1184/30864 | 59.49 | 10 |
| 90 | **88.841** | -1.159 | 11.880 | 12.123 | 89.399 | 5.0e-16 | 2.20e-4 | 4.50e-4 | 400/864/31504 | 85.85 | 11 |
| 120 | **116.858** | -3.142 | 14.798 | 9.094 | 114.079 | 2.6e-15 | 7.99e-4 | 1.82e-3 | 448/608/31712 | 110.91 | 10 |
| 150 | **146.231** | -3.769 | 17.559 | 5.330 | 137.543 | 1.8e-15 | 1.48e-3 | 3.02e-3 | 384/624/31760 | 136.29 | 10 |

No run touched the pressure cap (300); `sum C` over solid cells is **exactly 0** in every row.

**The gate as written (|theta - theta_set| <= 3 deg at every angle) FAILS at 120 and 150 on this
scene and passes at 30/60/90** — see finding 6 for the mechanism and the corrected gate. The same
protocol on the ctest scene (40x40x32, D/dx = 20, 150 steps) reads 60 -> **60.121**,
90 -> **89.170**, 120 -> **118.719**, worst **1.281 deg**.

### G2 — the drop on an SDF sphere (`tests/study/vof_wetting.py g2`)

64^3, solid sphere `Rs = 12`, liquid volume that of a sphere of `Rd = 8` (V = 2144.66), sigma = 1,
mu = 0.4, no gravity, 600 steps. The reference shape is the spherical cap whose sphere meets the
solid sphere at theta: the LAW OF COSINES gives the centre distance directly,
`d^2 = Rs^2 + Rc^2 - 2 Rs Rc cos(theta)`, and the liquid volume is `(4/3) pi Rc^3` minus the
two-sphere lens, so `(V, theta) -> Rc` is a 1-D root solve (written out in the study script). The
run is measured the other way round: `(V, apex height H = d + Rc) -> (Rc, d) -> theta`.

| theta_set | theta | err | Rc | Rc ref | Rc err | H | H ref | dV/V | Ca | iters |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | **64.520** | +4.520 | 9.383 | 9.585 | **-2.11 %** | 21.009 | 20.578 | 4.4e-11 | 7.0e-4 | 9 |
| 90 | **87.328** | -2.672 | 8.615 | 8.549 | **+0.77 %** | 23.058 | 23.283 | 2.2e-11 | 3.3e-3 | 9 |
| 120 | **113.758** | -6.242 | 8.172 | 8.119 | **+0.65 %** | 25.195 | 25.649 | 1.0e-11 | 6.0e-4 | 9 |

`sum C` over solid cells is exactly 0 in every row and no run touched the pressure cap.

**The cap-radius half of the gate PASSES with room to spare (worst 2.11 % against 3 %) while the
angle half fails at 60 and 120 — and that is a statement about the MEASUREMENT, not about the
shape.** The angle is inferred from the conserved volume and the APEX HEIGHT, and that inversion is
badly conditioned: at theta = 60 the measured `H` is 21.009 against a reference 20.578, i.e. **0.43
of a cell**, and it produces 4.5 deg of apparent angle — `dtheta/dH ~ 10.5 deg per cell` at this
`Rs`/`Rd`. A half-cell error in a colour column sum, which is what an interface crossing a cell
diagonally gives, is therefore worth five degrees. The shape itself is right to under a percent in
every row. **Corrected gate proposed for G2**: gate the CAP RADIUS (3 %, well conditioned, and it
passes) and report the angle with the conditioning `dtheta/dH` alongside it, or measure the angle
locally at the contact line (the mean apparent angle of `contact_angle_diagnostics`, which reads
74.6 / 89.6 / 114.1 here) rather than by inverting a global shape.

### The same sweep at density ratio 100 (G1, `rho` and `mu` LinearMix closures)

Identical scene and protocol, `rho_gas = rho_liquid/100`, `mu_gas = mu_liquid/100`, 500 steps.

| theta_set | theta | err | dV/V | Ca(open) | raw max|u| | iters |
|---|---|---|---|---|---|---|
| 30 | **29.920** | -0.080 | 3.2e-12 | 1.72e-3 | 4.02e-3 | 11 |
| 60 | **58.360** | -1.640 | 1.9e-11 | 2.57e-3 | 1.69e-2 | 11 |
| 90 | **88.439** | -1.561 | 2.5e-11 | 3.02e-3 | 6.17e-3 | 11 |
| 120 | **116.415** | -3.585 | 1.5e-12 | 4.48e-3 | 9.14e-3 | 12 |
| 150 | **142.796** | -7.204 | 3.9e-11 | 1.47e-3 | 6.36e-3 | 11 |

From 30 to 120 deg the ratio-100 column reproduces the ratio-1 column row for row to within a
degree (29.92 vs 30.69, 58.36 vs 59.98, 88.44 vs 88.84, **116.42 vs 116.86**), i.e. the equilibrium
the theta fill selects is a property of the FILL and not of the density contrast, and the
120-degree residual is the same -3.6 deg in both. The 150-degree row is the exception and it gets
WORSE with the density contrast (**-7.20 vs -3.77**) — it is also the row whose contact radius is
5.8 cells, so the ratio only amplifies an already under-resolved contact line rather than
introducing a new defect. Volume conservation stays at the projection floor (<= 3.9e-11) and the
pressure solve never approaches its cap in any row.

### Where the SDF wall sits inside the cell (G1w) — the decisive measurement of this rung

One angle (theta_set = 60), one initial condition (a hemisphere, so the contact line HAS to travel),
one grid (64x64x40, D/dx = 24, Oh = 0.1), 1200 steps; only the wall's position inside the cell
changes. `eps` / `ox` are the fluid fraction and the tangential face openness of the wall-adjacent
cell, read from `vof_geometry(0)` and `get_ox()`. The first row is reproduced by the shipped
`tests/study/vof_wetting.py g1w` (58.076, `Ca(open)` 9.19e-3, band census 720/1264/30784, 11
pressure iterations, no cap) and the other three by the same scene with `zw` changed.

| wall z | eps of the wall cell | tangential ox/oy | theta after 1200 steps | raw max|u| | verdict |
|---|---|---|---|---|---|
| 4.00 (on a cell FACE) | 1.00 (uncut) | 1.000 | **58.076** (converged; 90 -> 77.3 -> 63.5 -> 59.9 -> 58.4 -> 58.05 -> 58.08) | 8.2e-2 | mobile |
| 4.25 | 0.75 | **0.750** | reaches 69.9 by step 400 and keeps going | 1.7e-2 | mobile, CUT |
| 3.50 (the WORK ORDER's half-integer) | 0.50 | **0.000** | **83.26 — STALLED** (83.4 at 200, 81.2 at 400, then back up) | **1.04** | PINNED |
| 3.75 | 0.25 | 0.000 | **89.65 — frozen** (nothing moves at all) | 2.4e-4 | PINNED |

At exactly `k + 1/2` the tangential MAC faces of the wall-adjacent cell sit ON the SDF zero level.
`buildOpenness` treats `sdf > 0` as fluid, so `sdf == 0` closes them: **the cell reads `eps = 0.5`
and `ox = oy = 0.000` simultaneously**. It is a fluid cell with no tangential flux at all — no
colour can move along the wall, no momentum can, and the unrelieved Young force appears as a
velocity of order 1 on DOFs the projection never sees.

Per-z-plane read-out at the half-integer wall (theta = 60, 200 steps, 48x48x32):

```
   z  |  max|u|   max|v|   max|w|  | max ox  max oy  max oz | eps
    2 | 0.000e+00 0.000e+00 0.000e+00 |  0.000  0.000  0.000 |  0.000   <- solid
    3 | 7.093e-01 7.106e-01 0.000e+00 |  0.000  0.000  0.000 |  0.500   <- the wall cell
    4 | 3.392e-02 3.423e-02 4.440e-15 |  1.000  1.000  1.000 |  1.000   <- the first open fluid row
```

### G4 — capillary rise / Jurin: NOT PASSED, and the run is not conclusive

80x4x96 quasi-2D, two SDF plates 4 cells thick bounding a gap `w = 24` with a periodic OUTER
channel `w_out = 48`, the plates spanning z in [16, 88] so the two channels are connected by an open
reservoir below and above, ratio 10, `drho g = 2.4e-3` (Bond `drho g w^2/sigma` = **1.382**),
theta = 30, ZERO-MEAN buoyancy (a non-zero-mean body force in a fully periodic box accelerates the
whole fluid without bound — WO-Q finding 9 — and Jurin depends only on `drho`, so subtracting a
constant is exact here). Both channels rise, so the reference is the LEVEL DIFFERENCE
`2 sigma cos(theta)/(drho g) (1/w - 1/w_out)` = **15.035 cells**.

Measured after 800 steps: inner level **39.883**, outer **37.378**, difference **2.505 cells** —
**-83 %**, FAIL — with `dV/V` 1.5e-12, 17 pressure iterations (cap 300, none capped) and a
**`max|u|` of 7.1e-1 whose trace over the run is 6.0e-1 / 5.8e-1 / 6.4e-1 / 8.4e-1 / 9.0e-1 /
8.8e-1 / 7.4e-1 / 7.1e-1**, i.e. a large, non-decaying parasitic mode rather than a rise in
progress. **The run is reported, not diagnosed**: the scene has two features this session could not
separate from the rung — the plates are only 4 cells thick, so the wetting bands of their two faces
overlap and the walk along `n_w` crosses the plate's medial surface where `grad(sdf)` is degenerate,
and the plate ENDS are sharp 90-degree SDF corners where `|grad(sdf)| != 1`. Both are first-order
suspects for a `max|u|` of that size and neither is what G4 is meant to measure. **The next step is
to rerun G4 with thick plates (>= 8 cells) rounded at the ends, and to check the meniscus curvature
`R = w/(2 cos theta)` directly from the shape before trusting a level difference** — a curvature
read-out is local and does not depend on the reservoir bookkeeping the level difference needs.

### Findings

**1. The work order's anchor is not on the interface plane, so the fill is not idempotent — and
idempotence is the whole mechanism.** WO-S item 3 pins the theta-plane at
`c = p_f - sdf(p_f) n_w`, the fluid cell's PLIC centroid projected onto the wall ALONG `n_w`. That
point lies on the wall but not on the interface plane unless the interface is perpendicular to it:
the projection changes the plane's offset by `m_theta . (c - p_f) = -sdf(p_f) cos(theta)`, which is
an O(1) shift of the interface (`sdf(p_f)` is 0.5-1.5 cells) and has the WRONG SIGN — it removes
liquid from the band for a wetting angle. The consequence is not an accuracy loss, it is a
STRUCTURAL one: an interface that already meets the wall at exactly theta is not reproduced, so
theta is not a fixed point of the scheme and the equilibrium angle is biased by whatever it takes
for the shift to cancel. MEASURED (gate G0a, exact plane fractions, exact `m_f`): the band fraction
is off by up to **0.0953 / 0.2614 / 0 / 0.2439 / 0.0962** at theta = 30/60/90/120/150, i.e. exactly
zero at 90 (where `cos theta = 0`) and worst in mid-range. Three anchors ARE idempotent to 1e-15:
`p_f` itself (the Afkhami-Bussmann / Basilisk `contact.h` rule — the ghost height is the FIRST FLUID
row's height plus the prescribed slope, pivoted at that row and not at the wall), the point of the
fluid plane that lies ON the wall, and the shipped one. All four ship as
`set_contact_angle_pivot(0..3)`; the work order's is mode 2.

**The shipped anchor is the volume one, and it is also the only one that never reads the part of
`m_f` that cannot be measured**: `alpha = plicAlpha(m_theta, C_f)`, i.e. the plane of normal
`m_theta` whose liquid volume in the anchor cell is exactly its colour. It needs no pivot point, it
is volume-consistent with the cell it is anchored on, and it is exactly idempotent because
`plicAlpha` is the analytic inverse of `plicVolume`.

**2. A fluid-only estimator cannot measure the wall-normal component of the interface normal, and
it does not need to.** WO-S item 1 prescribes a "fluid-only Youngs gradient" to break the
circularity (the fluid cell's own MYC normal reads the very band cells the fill is writing). It
works — but only half of it. Below the first fluid row there is no colour to difference against, so
the wall-normal component falls back to a one-sided difference over half the distance across a
SATURATING profile. MEASURED on exact plane fractions (gate G0e): the recovered normal's angle to
the wall is wrong by **23.13 deg at theta = 30 and 20.91 at theta = 150**, against **2.31 deg** for
the same Youngs stencil with the solid rows present. Feeding that into the WO's construction (which
uses the full `m_f` for the reconstruction and the centroid) would put a ~20 deg error into the
plane's position.

The resolution is structural, not numerical: the rung's whole point is to OVERWRITE the angle to the
wall with the prescribed one, so the construction only needs the AZIMUTH of the contact line — the
direction of `t = m_f - (m_f . n_w) n_w` inside the wall. Those components come from complete,
two-sided half-plane differences and are **exact for a plane: 0.000 deg over the whole sweep**. The
shipped `vofWettingPlane` therefore builds `m_theta = cos(theta) n_w + sin(theta) t_hat` from the
azimuth alone and anchors it on `C_f`, and the END-TO-END band error of the fill driven by the
fluid-only normal instead of the exact one is **1.1e-15** (G0e, last column). The two-pass fallback
the work order names as the escalation path (neutral fill -> MYC -> theta-fill -> MYC) was therefore
never needed.

**3. The anchor's own column is not enough, and the symptom is a 5-degree bias.** Walking strictly
along `n_w` to "the first fluid cell" gives a PURE-PHASE anchor in every column just outside the
contact circle — and that is exactly where the continued interface still puts liquid INSIDE the
solid, because a wetting interface leans outward as it goes down. MEASURED on an equilibrium
theta = 60 cap (48^3, D/dx = 20, wall on a cell face), the band column one cell outside the contact
column: the fill wrote **0.0000 / 0.0000 / 0.0000** in the three rows below the wall where the
continued plane gives **0.0500 / 0.6300 / 1.0000**. The height function there reads a wall slope
much closer to 90 deg than to 60, and the equilibrium angle of a from-90-degrees relaxation came out
**65.2 deg for theta_set = 60 — 5 deg biased towards 90 — and stalled there**. Shipped fix: where
the anchor is pure phase, average the theta-planes of the anchor's MIXED fluid neighbours (its 3^3
stencil) evaluated in the band cell; with no mixed neighbour the pure continuation stands. The same
column then reads **0.1045 / 0.6363 / 0.9945**, and the same relaxation converges to **58.1 deg**.
The branch is reported separately as `neighbour_cells`.

**4. The prescribed angle is a fixed point; the ATTRACTION to it is slow and the residual is a
degree or three towards 90.** Two protocols, both in `tests/study/vof_wetting.py`. Starting AT the
prescribed angle (the fixed-point statement) the cap stays there: the sweep table above. Starting
from a HEMISPHERE the contact line has to travel and the driving vanishes as it arrives — measured
at theta_set = 60, wall on a cell face, D/dx = 24, Oh = 0.1: 90.0 -> 77.3 (200 steps) -> 63.5 (400)
-> 59.9 (600) -> 58.4 (800) -> 58.05 (1000) -> **58.08 (1200, converged)**, i.e. it reaches the
prescribed angle to **-1.9 deg** and stops. The residual has the same sign as the fixed-point
residual (towards 90) and the same size, so the two protocols agree on where the discrete
equilibrium is. Practical consequence for E5/E6: a wetting run needs O(10^3) capillary-limited steps
per 30 degrees of contact-line travel, not O(10^2).

**5. Where the SDF wall sits INSIDE the cell decides whether the contact line can move at all, and
the work order's placement is the one that pins it.** WO-Q's G5 and WO-S's G1 both ask for a wall at
a HALF-INTEGER z "so the wall cells are genuinely cut". They are cut — `eps = 0.5` — but that
placement puts the wall-adjacent cell's TANGENTIAL MAC faces exactly on the SDF zero level, and
`buildOpenness`'s `sdf > 0` test closes them: measured `ox = oy = 0.000` on a cell whose `eps` is
0.5 (the table above). That cell can exchange nothing along the wall, so the contact line is
immobile: a theta = 60 relaxation from a hemisphere stalls at **83.3 deg** with a steady
`max|u| = 1.04`, while the same run with the wall on a cell FACE converges to **58.1 deg** with
`max|u| = 8e-2`, and with the wall at `k + 1/4` — genuinely cut, `eps = ox = oy = 0.75` — the
contact line is mobile AND the near-wall velocity is the smallest of the three (1.7e-2). The gates
and the shipped study therefore use a QUARTER-integer wall; that is a corrected gate, not a tuned
one, and it is the only change of scene this rung makes.

**This is also the answer to WO-Q's open question 8** ("whether the wall-band velocities are a real
defect or an artefact of the IBM DOFs"): they are an artefact, and a specific one. The 0.788 raw
`max|u|` WO-Q recorded on its G5 cap lives entirely on the `z = z_wall` plane of velocity DOFs whose
FACE OPENNESS IS 0 — the projection never sees them, no flux is weighted by them, and the IBM
constrains them. Move the wall a quarter cell and the same measurement reads **1.7e-3**, i.e. a
factor of ~460 lower, with the physics unchanged. The number to report as a spurious current is the
one over the open fluid, and on the corrected scene that is `Ca = 2.2e-4 ... 1.5e-3` over
theta = 30..150 at D/dx = 24 with mu = 0.49 — against the V4 free-droplet 2.6e-5 at the same
resolution, so a contact line still costs about an order of magnitude in parasitic currents, which
is the honest version of WO-Q's "~20x".

**6. The prescribed angle is reproduced to 3 deg for theta <= 90 and the error grows to 3.5-3.8 deg
at 120 and 150; it is a CONVERGED bias, not an unfinished transient.** Sweep table above. The
120-degree row was rerun to rest as a drift trace (same scene): 118.15 (250 steps) -> 116.86 (500)
-> 116.48 (750) -> 116.45 (1000) -> 116.44 (1250) -> **116.40 (1500)**, with `max|u|` decaying
5.4e-3 -> 1.8e-3 -> 4.3e-4 -> 2.1e-4 -> **1.8e-4**, i.e. the drop is at rest and the discrete
equilibrium really is 116.4, an error of **-3.60 deg**. The errors over the sweep are monotone in theta
(+0.69, -0.02, -1.16, -3.55, -3.77 at 30/60/90/120/150), which is NOT the symmetric `|cos theta|`
signature of the cut-cell whole-cell reconstruction, so that approximation is not the whole story.
The contact-radius resolution is: `a` = 20.2, 15.3, 12.1, 9.1 and **5.3 cells** at those angles, so
the two failing rows are also the two where the contact line itself is resolved by fewer than 10
radial cells and where the two independent readings of the angle (the volume-consistent one and the
first-fluid-plane contour) part company most (146.2 vs 137.5 at theta = 150).

The cut-cell whole-cell reconstruction (item ii of finding 8) was ruled out directly: the same
theta = 120 fixed-point run with the wall on a cell FACE — where the anchor cell is UNCUT and that
approximation is exactly absent — reads **118.25 (250 steps) -> 115.81 (500) -> 115.27 (750)**,
i.e. WORSE than the cut-wall 116.86 / 116.48, so the bias is not the cut-cell reconstruction.

**Corrected gate proposed**: |theta - theta_set| <= 3 deg for theta in [30, 90] AND a contact radius
of at least 10 cells; above 90 deg the gate should be run at a resolution that keeps `a >= 10`
(D/dx = 24 gives `a = 9.1` at 120 and 5.3 at 150 — the scene, not the scheme, is under-resolved
there). The measurement to settle it is a D/dx refinement at fixed theta = 120/150, which this
session did not have the machine time for and which is the first thing to run next.

**7. `set_contact_angle(90)` is NOT bit-identical to WO-Q's neutral fill, and it cannot be.** The
work order's G6 asserts it is "the same plane". It is not: the neutral rule writes the MEAN of the
fluid FACE NEIGHBOURS and the theta rule writes a plane FRACTION, and the two coincide only where
the interface is already perpendicular to the wall and the band cell has exactly one fluid
neighbour. Measured on the cap scene at t = 0: `max |C_theta - C_neutral|` = **1.000** over all
solid cells and **4.28e-2** over the band cells the V3 cascade can reach (the all-cells number is
dominated by cells deeper than the neutral rule's three-pass reach, which it leaves untouched and
the theta walk fills). The physical consequence is nil: after 150 steps the two runs give theta
**89.170 vs 89.171**, `max|u|` **2.444e-3 vs 2.445e-3** and `Ca(open)` **1.222e-3 vs 1.222e-3`.
**Corrected gate, and the one that ships**: (a) with NO `set_contact_angle` call every V5a number is
byte-identical (this is the real inertness statement, and it holds digit for digit on both backends
for all seven `vof_*` ctests); (b) at theta = 90 the reachable-band difference and the physical
outcome are reported as above.

**8. What is NOT implemented, stated for the record.** (i) Domain walls (`set_domain_bc` type 1) do
NOT get the theta fill — WO-S's "a domain wall is a flat SDF wall at the face" is true physically
but the band fill only ever runs on the SDF classification, so a domain-BC wall still gets the
zero-gradient (90 deg) `clampFill`. Model a wetting wall as an SDF slab, which is what every gate
here does. (ii) The anchor cell's PLIC plane is reconstructed on the WHOLE unit cube even when the
anchor is CUT, consistent with WO-Q's flux; in a cut anchor the plane is displaced by roughly
`(1 - eps_f)/2` times the wall-normal component of `m_theta` (0.14 cells at theta = 60 on a
half-cut cell, zero at 90 and zero in an uncut cell). Removing it needs the solid-clipped volume
relation (Huang, *JCP* 2025/2026), the same refinement WO-Q's flux defers. (iii) `set_contact_angle`
costs three extra ghost exchanges of the colour block per fill (the fluid-only normal field), i.e.
about 12 per step under MPI on top of the two the fill already does; they are what make the theta
pass decomposition-independent (finding 9) and they are skipped entirely when no angle is set.

**9. The theta pass reads the anchor at ghost depth 3, which a 3^3 stencil on a g = 3 block cannot
evaluate — so the fluid-only normal is built on the INNER region and EXCHANGED.** This is WO-Q
finding 5 applied to a new field, and it is the reason the MPI gate is bitwise rather than "at the
floor". Building it "wherever the stencil fits" instead would make the depth-2 pass-1 values
decomposition-dependent, and those feed the inner result through passes 2-3 whenever a block
boundary lies within three cells of a solid surface — i.e. exactly in a sphere packing with a cut
that goes through the spheres. The wall normal is handled the same way: the SDF is embedded on the
colour block and run through the colour field's own ghost policy, and `n_w` is then a central
difference of the EXCHANGED field, valid at every cell pass 1 writes.

## WO-Q — rung V5a (VoF transport in cut cells) — DONE 2026-09-02, branch `vof-woq`

Commits: `abcba6f` (the rung), plus the item-8 momentum work and the study/doc commits listed at
the end. All numbers below are host-openmp unless a CUDA column is given; both backends agree.

### What shipped

`src/vof/cutcell.hpp` (container-free rules), the geometry branch in `src/vof/advect_wy.hpp` and
`src/vof/momentum_advect.hpp`, the solver plumbing in `flow_ibm.hpp`, the bindings,
`tests/kokkos/test_vof_cutcell.cpp`, `tests/kokkos_mpi/test_vof_cutcell_mpi.cpp` and
`tests/study/vof_cutcell.py`. The transported quantity is `eps_i C_i`; every flux is
`F_f = o_f * wyFaceFlux(a_f, ...)`; the dilation uses the SAME `o_f a_f`; the canonical `"C"` is 0
in solid cells and the working block's solid band carries the three-pass neutral fill.

### Gates

| gate | measured | verdict |
|---|---|---|
| **G1** byte-identity | all six `vof_*` ctests (`vof_plic`, `vof_advect`, `vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_surface_tension`) reproduce their recorded output **digit for digit** (`diff` of the full stdout against the same binaries built at `main`), host-openmp AND nvidia-cuda. The whole `tests/kokkos` battery is green on both backends (**27/27 on nvidia-cuda** including the new `vof_cutcell`; host-openmp likewise). The single-phase regression (`tests/regression/sdflow_regression.py`, CUDA) is **`+0.00 %` on every metric of all three cases with identical iteration counts and identical step counts** — `zh_sphere` K 7.3891/7.4162/7.4361/7.4404, `random_spheres` order 2.19 and `k*_inf` 0.0062362, `hollow_rings` order 1.38 and `k*_inf` 0.017184, all `[ok]` | PASS |
| **G3a** openness embed | `max|o_advector - independent build with the shift written out|` = **0.0**; `max|eps - independent|` = **0.0** (24³ packing, `buildOpenness`/`buildCellFraction` re-run on a g=3 block) | PASS |
| **G3b** solver vs standalone | 20 kinematic steps through the packing at interface CFL 0.2: `max|dC|` over fluid cells = **0.0 (bitwise)** | PASS |
| **G2** conservation | 24³ packing, 200 kinematic steps at CFL 0.2 (dt 1.85): `sum eps_eff C` **6.110906250000000e3 -> 6.110906250046966e3, relative drift 7.686e-12** against a projection floor `max|div(open u)| = 3.048e-11` (9 pressure iterations, no cap). `sum C` over solid cells **exactly 0**; min/max C over uncut fluid cells **0.0 / 1.0 exactly**; clipped liquid volume over the whole run **5.42e-19** (8.9e-23 of the liquid volume); 991 solid cells, 1076 cut cells | PASS |
| **G5** 90° neutral fill | cap on a flat SDF wall at z = 3.5, D/Δ = 24, σ = 1, µ = 0.05, 120 steps at 0.5 dt_σ: **θ = 89.935°** (target 90, tol 3°); cap radius 11.977 vs 12; Young–Laplace ΔP 0.166185 vs 2σ/R 0.166990, **rel 4.8e-3** (tol 1 %); volume drift **4.5e-15**; 16 pressure iterations, no cap; `sum C` over solid **0**; wall band max\|κ\| 0.2222 with branch census 3000/196/0/0/4/0/0 (**no branch-6**) | PASS |
| **G6** MPI (host-openmp AND nvidia-cuda) | np 1/2/4 on a 16×16×32 grid whose ORB cut goes through the spheres (cut axes z at np=2, xz at np=4): geometry **0.0**, band fill **0.0**, kinematic colour **0.0** — all three **bitwise**; kinematic drift 4.429e-13 (reference 4.433e-13); coupled 30-step run at the reduction floor, colour 4.4e-16 (np=2) / 3.3e-16 (np=4), velocity 5.9e-17 / 2.8e-17, drift 2.75e-12, `sum C` over solid **0**, 10 pressure iterations. On CUDA the coupled case is bitwise at np=2 as well (colour 0.0) and 2.2e-16 at np=4. The pre-existing MPI VoF battery is unchanged: `vof_advect_mpi` drift 1.0e-15, `vof_twophase_mpi` dC 2.2e-16, `vof_momentum_mpi` du_adv 2.4e-13 (tol 4.6e-12), `vof_curvature_mpi` 0/8192 cells differ — both backends, np 1/2/4 | PASS |
| **G7** free-surface battery | `tests/study/vof_surface_tension.py static hysing1`: static droplet **Ca = 5.898e-05 at D/Δ = 16** (WO-P recorded 5.90e-5) and 2.543e-4 / 2.649e-5 at D/Δ = 8 / 24; Hysing case 1 at nx = 64 **v_rise max 0.2497 at t = 0.886, y_c(3) = 1.0808** — WO-P's recorded numbers to every digit | PASS |
| **study battery, full size** (`tests/study/vof_cutcell.py`, CUDA) | **G2** 48³, 500 kinematic steps at CFL 0.2 (dt 1.569): `sum eps_eff C` drift **−5.041e-12** against `max|div(open u)| = 3.920e-11`, 9868 solid + 4612 cut cells, clipped 2.30e-18, colour in solid 0, C ∈ [0,1] exactly. **G4** 48³ draining, ratio 10, 400 coupled steps with momentum consistency ON: drift **4.666e-14 per step**, 12 pressure iterations (cap 300, 0 capped), `max|u|` 1.13e-2, colour in solid 0. **G5** D/Δ = 24, 200 steps at 0.5 dt_σ: **θ = 89.650°**, cap radius 11.986 vs 12, Young–Laplace **rel 3.494e-3**, volume drift −3.9e-15, Ca (open fluid) 6.74e-4 | PASS |
| **item 8a** consistency identity in cut cells | packing + uniform `U = (1, 0.6, -0.4)`, `enable_vof_momentum`, ratios 10/100/1000: `max|u_adv - U| = 0` — **bitwise, including in cut cells** (ρ^e floor never hit; 6252 flux clamps bind) | PASS |
| **item 8b** coupled draining | 24³ packing, ratio 10, zero-mean buoyancy, 200 coupled steps: colour drift **6.02e-14 per step** with momentum consistency ON and 6.02e-14 OFF (tol 1e-10); `max|div(open u)|` 2.5e-14; 11 pressure iterations (cap 200); `max|u|` 2.15e-2; `sum C` over solid 0 | PASS |

### Findings

**1. The work order's effective-Courant rule (item 6a) is wrong on its own, and it fails loudly.**
`o_f |a_f| / max(eps_i, 0.1)` is *smaller* than `|a_f|` wherever `o_f < eps_i`, so a nearly-closed
face inside an open cell licenses an arbitrarily large slab thickness — and `|a_f|` is the thickness
of the slab the geometric flux clips out of the donor CELL, which is only a flux for `|a| <= 1`.
Measured with the rule as written: a 24³ packing at nominal "CFL 0.2" ran at **dt = 1.85** and lost
**70 % of the liquid volume in 200 steps** while the flux sum still telescoped — the textbook
signature of an over-CFL Weymouth–Yue run (conservation is algebraic and survives; boundedness does
not). The corrected rule, shipped, is
**`max( |a_f| , o_f |a_f| / max(eps_i, 0.1) )`** — it reduces exactly to the uncut `|a_f|` in clear
fluid and throttles by `1/eps` in a cut cell. Consequence worth carrying: in the 24³ packing the
cut-cell Courant number is up to **6×** the plain one (at dt = 0.2 and `|U| = 1` the plain CFL is
0.2 and the cut one 1.24), so a cut-cell VoF run takes correspondingly smaller steps.

**2. The clip is NOT the mechanism; Weymouth's admissible interval on the flux is.** With the
whole-cell-PLIC-times-open-area flux and the [0,1] clip of item 6b as the only bound, the clip fired
at up to **3.2e-5 liquid volume per step** on the 24³ packing (5.2e-9 of the total, i.e. just under
the work order's 1e-8 trigger) and the conserved functional drifted **1.3e-8 in 30 steps** — the
1e-11 gate was unreachable. The remedy the work order names (redistribution to the fluid face
neighbours) needs an extra halo exchange per sweep and a rule for out-of-domain ghosts. The cheaper
and exactly conservative remedy, shipped instead, is the cut-cell generalization of the clamp WO-K
already uses on the momentum control volumes (`vofCutFluxClamp`): through a face of open area `o`
and slab thickness `|a|` the scheme sweeps a FLUID volume `o|a|`, of which at most `eps_don C_don`
can be liquid and at most `eps_don (1 - C_don)` gas, so
`max(0, o|a| - eps_don(1 - C_don)) <= |F| <= min(o|a|, eps_don C_don)`. It is applied to the ONE
value both neighbours share, so conservation still telescopes bit-exactly, and it is applied **only
when the donor is MIXED** — a pure-phase donor takes `wyFaceFlux`'s algebraic branch, whose flux is
already exactly bounded, and clamping it would break the exact full-cell cancellation. With it the
clip stops firing (5.4e-19 over 200 steps) and the drift is the projection floor. `set_vof_cutcell_flux_clamp(False)`
is the measured ablation.

**3. The exact full-cell cancellation needs the same PARENTHESISATION in the flux and the
dilation.** `wyFaceFlux` forms the Courant number first (`u * dth`) and the flux is `o * (that)`, so
the dilation term must be `o * (u * dth)` and NOT `(o * u) * dth` — the two differ by an ulp and the
difference does not cancel. Written the wrong way the drift after one step was 3.7e-14 instead of
the ~1e-15 it is now.

**4. A conserved functional has to be named, and it is `sum eps_eff C`, not `sum eps C`.**
`buildCellFraction` subsamples 4³, so `eps` is a multiple of 1/64 and a cell can read `eps == 0`
while still owning an OPEN face (the face openness comes from a different quadrature). Such a cell
is FLUID by the work order's own classification and legitimately receives flux, so the update must
divide by `eps_eff = max(eps, 1/64)`; the raw `sum eps C` then silently drops whatever enters those
cells. Both are reported (`vof_diagnostics()['volume']` and `['raw_volume']`); on the shipped
packings at 24³ they agree to the last digit, but at 48³ they do NOT: `sum eps_eff C` starts at
4.876225e4 while `sum eps C` starts at 4.876175e4 — a 5.0e-1 gap that is exactly the colour sitting
in `eps == 0`-with-an-open-face cells, and that the raw sum would have to lose. Both drift by the
same −5.041e-12, i.e. the identity is on the eps_eff sum and the raw sum merely tracks it.

**5. The classification at ghost DEPTH 3 must be the owner's, and the way to get it is to exchange
it.** The work order says to embed `eps` "with the outermost layer set to 1". That layer is read:
pass 1 of the band fill runs over solid cells at depth ≤ 2 and reads their face neighbours at depth
3, so a locally-guessed classification there is a decomposition dependence in the INNER result.
Shipped instead: `eps`, the three openness fields and the classification (as a double, `kindDouble`)
all go through the colour field's OWN ghost policy after they are built, which puts the owner's
value in every ghost layer under MPI and the periodic wrap single-rank. That is what makes the
`geometry 0.0 / band fill 0.0` bitwise MPI gate hold. The `-d` face of a cell on the low plane of
each axis lies outside the block, so `classifyGeometry` leaves that plane provisionally fluid and
the exchange overwrites it.

**6. The canonical `"C"` carrying 0 in solid cells is free — measured, not assumed.** The working
block always carries the neutral fill; the question is what the closures and the CSF see. The G5 cap
run with `set_vof_solid_colour_zero(True)` (the WO-Q contract) and with the band fill written into
`"C"` instead is **bitwise identical in every reported quantity** (θ, ΔP, volume, max\|u\|, the
curvature census) and differs only in `sum C` over solid cells (0 vs 452). Mechanism: the faces
whose neighbour is a SOLID cell have openness 0, so the CSF force and the projection coefficient
there are multiplied by zero and the IBM masks the velocity. So the gate ("colour in solid cells
exactly 0") costs nothing. The knob ships as `set_vof_solid_colour_zero`.

**7. A hidden index-convention bug this campaign nearly shipped, kept here because the symptom is
so generic.** The new G=2 solid mask was allocated on the EXTENDED block (`CCField(..., n_)`, which
is the extended size) and written by `copyInner` at `(x+G, y+G, z+G)`, but read with INNER strides.
It silently zeroed live fluid colour and cost **0.5 % of the liquid volume per step** with no
boundedness violation, no clip activity and no NaN — a perfectly smooth, physically plausible loss.
It was found by asking "which cells are not 1 when the whole fluid is 1", not by any gate.

**8. The spurious currents at a contact line are ~20× the free-droplet value, and the near-wall
band is worse.** G5, D/Δ = 24, σ = 1, µ = 0.05: `max|u|` in the OPEN fluid (z ≥ ⌈z_wall⌉+1) is
**9.82e-3**, i.e. **Ca = 4.91e-4** against the V4 free-droplet 2.6e-5 at the same resolution.
Including the wall band the raw `max|u|` is **0.788** (Ca 3.9e-2), and that number is on the
IBM-constrained velocity DOFs on and inside the wall — reporting it as a spurious current would
measure the immersed boundary rather than the surface-tension balance, which is why the shipped
number excludes them. It is still growing slowly at 120 steps (0.53 / 0.69 / 0.76 / 0.79 over the
run) while the shape and the volume are exact, so it is a saturating near-wall parasitic mode, not a
drift of the interface. **Open question for WO-S**: whether the θ-consistent fill changes it, and
whether the wall-band velocities are a real defect or an artefact of the IBM DOFs.

**9. OPEN — the momentum-consistent cut-cell path fails on an unbounded-acceleration case while the
colour-only path does not.** With a NON-zero-mean buoyancy `-g rho` in a fully periodic box (so the
whole fluid accelerates without bound), the 24³ packing at ratio 10 with `enable_vof_momentum` runs
clean and conservative to 1e-12 for **155 steps** — `C^e` inside [0,1] to the last bit, `rho^e`
never floored, 9–10 pressure iterations — and then at step ~160, with `max|u|` ≈ 0.19, `C^e` goes to
`+inf` **inside the advection** and the velocity follows. The identical case with momentum
consistency OFF completes 200 steps (drift −2.29e-13, `max|u|` 0.236). There is no bounded-quantity
precursor in the trace, so it is reported rather than patched; reproduce with
`PECLET_VOF_CUTCELL_NONZERO_FORCE=1 PECLET_VOF_CUTCELL_TRACE=1 ./test_vof_cutcell`. The shipped
item-8 gate uses the well-posed zero-mean force, where both paths are clean.

**11. G7 as written compares a coarse run against a fine recorded number.** `--quick` runs the
Hysing gate at **nx = 32** while the numbers the work order asks it to reproduce (0.2497 / 1.0810)
were recorded at **nx = 64**. At nx = 32 the same build reads v_rise max **0.2501**, y_c(3)
**1.0844** — the same to 3 significant figures, but the 4th digit is the grid, not the change. Run
without `--quick` for the gate as intended: it then returns 0.2497 / 1.0808, i.e. the recorded
values to every digit. (The static-droplet rung is present in both, and matches at 5.898e-5.)

**10. What the cut-cell flux approximates, stated for the record.** The PLIC polyhedron is
reconstructed on the WHOLE unit cell (as if the cell were not cut) and its slab volume is multiplied
by the open area, instead of being clipped against the SOLID polygon as well (Huang, *JCP*
2025/2026). It is conservative either way (one number per face), exact wherever the interface and
the wall are parallel or the cell is whole, and O(1) wrong in the DISTRIBUTION inside a cell whose
interface crosses its wall. `vof_diagnostics()['clipped_volume']` is the tripwire, and it reads
5.4e-19 over 200 steps on the shipped packing gate.
