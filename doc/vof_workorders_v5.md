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

## WO-R2 — the variable-density outflow operator, the cut-cell × boundary composition, and the VoF defaults  [OPUS]

Written 2026-09-02 after WO-R's findings (read them first: "item 4 refuted, and the varRho
open-boundary operator defect"). Four items, each small, each gated; together they are what E6
(liquid over a packing with a liquid inlet at the top and an outlet at the bottom, ratio ~800)
needs.

1. **The operator.** `CutcellMG::applyBoundaryOpenness` (`mac_cutcell_mg.hpp`) re-imposes the
   boundary face value as the literal `1.0` on every level including level 0, overwriting the
   `open·ρ₀/ρ_f` coefficient `buildRhoCoeff` put there under varRho. Fix: at an outflow face
   impose the CALLER's coefficient (level 0: the varRho coefficient already computed; coarser
   levels: whatever the coarsening produced, i.e. do not overwrite), and keep the literal 1.0
   only where the coefficient path is not in use (constant density — must stay bit-identical:
   the `pressure_wallbounded` ctest, the single-phase regression's open-boundary points, the
   examples' developing-channel/BFS numbers are the tripwires). Then WO-R's
   `set_outflow_rho_correction(...)` (the `1/ρ_f` high-side correction) becomes the consistent
   choice — flip its default to ON under varRho and re-measure; the low side goes through
   `projectCorrectVar` and needs nothing. Gate: WO-R's G2 Nusselt film at ratio 100 and 1000
   (the −z outlet), film flow rate within 3 % and no capped solve; WO-R's G3 gas-over-pool
   pressure solve no longer capped; `max|div|` of the projected field at the outlet ≤ 1e-10 at
   ratio 10/100/1000 on the WO-R budget box.
2. **Compose V5a with V-BC.** The out-of-domain donor rule (`wyFaceFluxBc`, algebraic
   `C_datum · a`) and the cut-cell path (`F = o_f · flux`, the flux clamp, the `eps_eff` update)
   must act together: a domain-face flux in a geometry-carrying block is `o_f · C_datum · a`
   (o_f = 1 on the open part of a domain face that no solid cuts; where a solid DOES cut the
   domain face, o_f < 1). Gate: the WO-R G1 colour budget with a sphere array inside the box
   (packing kept 3 cells clear of the inlet/outlet planes AND a second scene where one sphere
   cuts the outlet plane): budget to 1e-12, solid colour exactly 0, MPI np 1/2/4 bitwise on the
   kinematic run.
3. **VoF turns the exact residual on.** WO-R item 6 measured `PECLET_FLOW_EXACT_RESIDUAL=1`
   removing 7.5 orders of flux divergence on Hysing case 2 and moving nothing else. Add
   `set_pressure_exact_residual(bool)` (`mac_cutcell.hpp`: a process-wide flag the env var
   initialises; the setter overrides it) and have `enableVof` set it ON, with a docstring saying
   why; `set_pressure_exact_residual(False)` after `enable_vof` is the ablation. Gate: Hysing 2
   reproduces WO-R's `=1` column through the new default; every VoF ctest at ratio 1 is
   bit-identical (the exact residual only differs where the float rounding of the bands differs
   from the double flux form — measure and record which VoF ctests move at the last digit and
   by how much; if any moves more than 1e-12 relative, stop and report).
4. **Wisp guard on the advector's mixed predicate** (WO-R's V0/V1 fragility: an emptied domain
   reaches `C → −inf`). `WyAdvector::wispEps` (default 0 = V1 verbatim); `enable_vof` sets it to
   1e-8 (the same threshold the curvature already uses under surface tension). Cells with
   `C ≤ eps` or `≥ 1 − eps` are treated as pure for reconstruction and flux (algebraic flux of
   their actual C, so conservation is untouched). Gate: WO-R's emptying-domain MPI scene runs to
   completion with no NaN; the V1 ctests' recorded numbers with `wispEps = 0` unchanged; with
   1e-8 record the digit-level differences.

Also fix `maxAbsDiff`'s NaN blindness in `tests/kokkos*` helpers (`fmax(m, NaN) == m`): a NaN
field must fail every bitwise gate (WO-R found it, fixed it in its own test only).

Deliverables: the operator fix, the composition, the two defaults, `tests/kokkos/test_vof_bc.cpp`
extended, findings, CLAUDE.md. Do not change `max_open_divergence()`'s mutating behaviour (WO-R
open question 2 is a user decision — `max_open_divergence_projected()` is the non-mutating
sibling to use in VoF scripts).

---

# Findings log (v5 work orders)

(append per WO, newest first)

## WO-R2 — the variable-density outflow operator, the V5a × V-BC composition, the VoF defaults  [Opus, 2026-09-02]

Branch `vof-wor2`. All numbers nvidia-cuda unless a host-openmp column is given; both backends were
run and agree.

### What shipped

`CutcellMG::setOutflowCoefficient` + `mgSaveFacePlane` / `mgRestoreFacePlane` / `mgCoarsenFacePlane`
and the rewritten `applyBoundaryOpenness` (`src/mac_cutcell_mg.hpp`); `buildRhoCoeffOutflowFace`
(`src/mac_pressure.hpp`); the varRho pressure build + the collocated outflow correction
(`src/flow_ibm.hpp`); `set_outflow_rho_correction` **default flipped to ON**;
`vof::wyIsMixed(c, eps)`, `vof::wyColourJump`, `WyAdvector::wispEps` and the composed
`computeFluxesCut` (`src/vof/advect_wy.hpp`, `src/vof/momentum_advect.hpp`);
`setExactResidual`/`exactResidualPinned` (`src/mac_cutcell.hpp`) with `enableVof` turning it on;
`set_vof_wisp_eps` / `set_pressure_exact_residual` bindings; the `advect_vof` guard repair; gate G
in `tests/kokkos/test_vof_bc.cpp`, gate C2 in `tests/kokkos/test_vof_advect.cpp`, the composed +
drained scenes in `tests/kokkos_mpi/test_vof_bc_mpi.cpp`; the `maxAbsDiff` NaN sweep (13 files).

### The operator defect: what it actually was, and why only one side of it was visible

WO-R's diagnosis was right and its proposed one-line fix ("impose the caller's coefficient") is only
half the story, because the staggered face index makes the two sides of an axis **asymmetric**:

| domain face of axis `a` | face index | who writes it | what `applyBoundaryOpenness` did |
|---|---|---|---|
| low  (`bc_[2a] == 3`)   | `g` — an **inner** index | `buildRhoCoeff` (level 0), `coarsenOpenAvg` (coarse) | overwrote a correct `o·ρ₀/ρ_f` with **1.0** |
| high (`bc_[2a+1] == 3`) | `dims-g` — a **ghost** index | **nobody**; the periodic/halo fill wraps the opposite boundary in | 1.0 was the only value it ever had |

So the low side needed the overwrite REMOVED and the high side needed a value SUPPLIED. The shipped
fix is exactly that: under `setOutflowCoefficient(true)` the low side is left alone at every level,
and the high side is snapshotted from the caller's field before the level-0 fill and then
**area-coarsened plane by plane** down the hierarchy (`mgCoarsenFacePlane`, the same rule
`coarsenOpenAvg` applies to every interior face). `IbmSolver::project` fills the level-0 high plane
with `buildRhoCoeffOutflowFace` — literally the expression `bcCorrectOutflowVar` evaluates at that
face, which is what makes the pair exact. With the flag off the literal-1.0 path is untouched.

Two consequences worth carrying:
* **the coarse-level high plane is a NEW quantity.** WO-R2 said "coarser levels: whatever the
  coarsening produced". Nothing was produced — `coarsenOpenAvg` writes coarse INNER cells only, so
  the coarse high-side outflow face is a ghost index it never reaches. It had to be coarsened
  explicitly.
* **MG telescoping is refused** under the coefficient path (`setOpenness` throws): the telescope
  stage gathers inner cells only, so the fine high-side plane does not survive the merge. Telescoping
  is off by default; a two-phase open-boundary run must keep it off.

### Item 1 gates

**The F2 verdict INVERTS, exactly as WO-R2 predicted.** `tests/kokkos/test_vof_bc.cpp` gate F2,
stratified duct 32×4×16, walls ±z, inflow −x, outflow +x, 5 steps, `max|div(open u)|` of the
**projected** field:

| ratio | operator = raw openness (WO-R) | operator = `o·ρ₀/ρ_f` (WO-R2) |
|---|---|---|
| | plain corr. / with `1/ρ_f` | plain corr. / with `1/ρ_f` |
| 1    | 1.407e-17 / 1.407e-17 (bitwise) | 1.400e-17 / 1.400e-17 (bitwise) |
| 10   | **8.763e-10** / 9.236e-03 | 9.972e-05 / **8.314e-10** |

(the 1.407e-17 → 1.400e-17 at ratio 1 is not this change: that row runs constant-density, where the
whole coefficient path is off. It is the exact-residual default of item 3.) The gate's polarity is
flipped in the test with the mechanism written into it, and `set_outflow_rho_correction` now
**defaults ON** (`PECLET_FLOW_OUTFLOW_RHO=0` is the ablation).

**The outlet divergence by density ratio** (same duct, 5 steps, `max_open_divergence_projected`,
shipped defaults, i.e. the fixed operator + the exact residual):

| ratio | `1/ρ_f` correction ON (default) | OFF (ablation) | pressure |
|---|---|---|---|
| 1    | **1.400e-17** | 1.400e-17 | 32/400 |
| 10   | **2.534e-16** | 9.972e-05 | 33/400 |
| 100  | **1.092e-15** | 7.393e-04 | 53/400 |
| 1000 | **5.445e-15** | 9.082e-03 | **400/400 — capped** |

The WO-R2 gate (`≤ 1e-10` at 10/100/1000) is met by four to five orders. The ratio-1000 **cap on
this duct is pre-existing and unrelated** (it caps identically in the ablation, and it is the S3
coefficient-coarsening item): note that the two ratio-1000 configurations that WO-R could not get a
valid run out of at all — the Nusselt film and the gas-over-pool — now converge in 64 and 76
iterations.

**G2 Nusselt falling film — PASS at ratio 100 AND 1000**, the gate WO-R had to stop on.
`tests/study/vof_open_boundaries.py nusselt`, 32×4×64, δ = 8, film Re 5, FCG, 2400 steps with dt
re-picked every 20 from the solver's own WY limit:

| ratio | δ measured (target 8) | Q vs Nusselt | ΣC steadiness, last 200 steps | pressure | `max|div|` projected |
|---|---|---|---|---|---|
| 100  | **8.0825** (+0.0825 cells) | **+0.21 %** | 1.103e-08 | **57 / 400** | 2.29e-12 |
| 1000 | **8.0820** (+0.0820 cells) | **+0.21 %** | 1.897e-08 | **64 / 2000** | 2.13e-12 |

Gate: Q within 3 %, δ within 0.5 cell, ΣC steady to 1e-4, no cap. **All four met at both ratios.**
Before/after on the SAME binary (`PECLET_FLOW_OUTFLOW_COEFF=0` restores the old operator row): the
ratio-100 run **dies** — `WyAdvector: CFL = 0.317306 exceeds the boundedness cap 0.25 at dt =
0.321394` — which is WO-R's failure reproduced exactly (its `max|w|` 1.4550 against the film's
`u_max` 0.31236 is the same event one step earlier). WO-R's recorded functionals came from a
60-step probe before the blow-up (δ −0.0040, Q −1.40 %); the converged film is +0.0825 / +0.21 %,
i.e. the sign of the thickness error changes once the run is actually steady.

**G3 gas over a pool — the item-1 half PASSES, the quiescence half still fails (but the run is now
VALID).** 64×4×32, ratio 1000, gas inflow −x over a pool, outflow +x, 500 steps, FCG cap 800:

| | WO-R | WO-R2 |
|---|---|---|
| pressure | **800 / 800 — CAPPED, run INVALID** | **76 / 800, valid** |
| `max|div|` projected | 8.55e-05 | **3.28e-12** |
| pool volume drift (gate 1e-10) | −2.815e-03 | **1.030e-03** — still FAIL |
| `max|u|` in the liquid / inlet speed (gate 1e-3) | 3.123e-02 | **3.124e-02** — still FAIL |
| inflow ghost ρ | correct | correct (1 above the pool, 1000 at the pool's inlet plane) |

So the WO-R2 gate ("G3's pressure solve no longer capped") is met, and the two physics tolerances
that were already failing are **unchanged by this work order** — they are a spurious-current /
momentum-consistency question at ratio 1000, not an outflow-operator one. Recorded, not fixed.

**G1 colour budget (coupled) — PASS and tighter.** 32×32×64 driven by `step()`, 60 + 740 steps:
budget drift **1.295e-15 relative** (WO-R 4.589e-15), `max|div(open u)|` projected **1.92e-13**
(1.77e-13), pressure 22/400, C ∈ [−1.39e-16, 1+7e-16].

**Byte-identity at constant density — PASS.** `pressure_wallbounded` is byte-identical to `main`
on nvidia-cuda (`diff` of the full stdout). With BOTH new defaults ablated
(`PECLET_FLOW_EXACT_RESIDUAL=0 PECLET_FLOW_VOF_WISP_EPS=0`) all eight VoF ctests — `vof_plic`,
`vof_advect`, `vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_cutcell`, `vof_wetting`,
`vof_surface_tension` — are **byte-identical to the same tests built at `main`**, which is the
proof that items 1, 2 and 4 are inert when their features are off.

### Item 2 — V5a × V-BC composed

`computeFluxesCut` now carries the out-of-domain mask and the per-face ledger itself:
`F_f = o_f · wyFaceFluxBc(...)`, i.e. `o_f · C_datum · a` at a domain face, with `o_f < 1` where a
solid cuts that face, and the clamp skipped for an out-of-domain donor (it is on the algebraic
branch, already exactly bounded — clamping it would break the exact full-cell cancellation).

**Gate G (`tests/kokkos/test_vof_bc.cpp`, new)** — 20×20×32 duct, walls ±x ±y, inflow −z (w = 1),
outflow +z, a four-sphere array in the middle, upper half liquid at t = 0 so the OUTLET carries
liquid from step 1; 80 `step()`s to a steady field, then 340 kinematic advections with dt re-picked
from the solver's own cut-cell interface Courant:

| scene | solid / cut cells | `max|div(open u)|` | Σ eps_eff C in → out | injected / LEFT | budget drift (rel) | ΣC over solid | clipped |
|---|---|---|---|---|---|---|---|
| packing clear of both planes | 640 / 800 | 2.086e-15 | 5883.25 → 2137.30 | 833.57 / 4579.52 | **2.708e-16** | **0.000e+00** | 3.52e-20 |
| a sphere CUTS the outlet plane | 716 / 996 | 7.274e-14 | 5721.00 → 1954.90 | 851.17 / 4617.27 | **7.196e-15** | **0.000e+00** | 8.35e-20 |

Both are inside the 1e-12 gate by three orders, C stays in [0, 1] over the uncut fluid, and 17
pressure iterations (no cap).

**Corrected gate — the MPI half cannot be bitwise, and the reason is structural.** WO-R2 asks for
"MPI np 1/2/4 bitwise on the kinematic run". The pattern that makes a kinematic colour run bitwise
across decompositions (`vof_cutcell_mpi`: run the reference on the FULL grid on every rank, then
`setField` its velocity slice into each block) **cannot be used through an open boundary**: the
outflow-face correction lives on a ghost face index that `setField` does not carry, and the ghost
fill `setField` triggers is precisely the one that erases it (`fillVelGhosts` → `outflowCorrValid_
= false`, WO-R's own finding). A distributed solver fed that way would advect a field that differs
from the single-rank one at the outlet plane by the whole outflow correction. The composed MPI
scene is therefore driven by `step()` on both sides and gated at the allreduce-order floor, like
the existing coupled-jet scene, with the BUDGET identity gated absolutely. The two rungs' bitwise
kinematic gates still exist separately (`vof_cutcell_mpi` closed box, `vof_bc_mpi` slug).

### Item 3 — VoF turns the exact residual on

`exactResidual()` is now a settable process-wide flag (`setExactResidual`) that the env var
INITIALISES; `enableVof` turns it on unless `PECLET_FLOW_EXACT_RESIDUAL` was set explicitly
(`exactResidualPinned`), which is what makes `=0` the battery-wide ablation.
`set_pressure_exact_residual(False)` after `enable_vof` is the per-run ablation.

**The WO expected last-digit moves at ratio 1 and got four-order improvements in residuals.** With
`PECLET_FLOW_VOF_WISP_EPS=0` so that only this default is active, against the same tests built at
`main` (nvidia-cuda, full-stdout `diff`):

| ctest | moved? | what moved |
|---|---|---|
| `vof_plic` | no | byte-identical |
| `vof_advect` | no | byte-identical (the standalone advector never touches the pressure solve) |
| `vof_curvature` | no | byte-identical |
| `vof_wetting` | no | byte-identical |
| `vof_cutcell` | **yes** | G2 packing `max|div(open u)|` **3.048e-11 → 1.480e-15**; conserved functional drift **7.685e-12 → 1.488e-16**; clipped volume 1.47e-19 → 5.97e-19 |
| `vof_momentum` | **yes** | K4 per-step momentum conservation **2.382e-13 / 1.822e-13 / 1.852e-13 → 2.791e-16 / 3.401e-16 / 2.267e-16** (three components); K1 `max|u−U|` 1.3921e-07 → 1.3922e-07; K2 clamped-flux count 84486 → 84646 |
| `vof_surface_tension` | **yes, at 1e-17** | static-droplet `max|u|` 2.6829e-17 → 3.5725e-17 (n = 16), 2.4458e-17 → 2.4252e-17 (n = 32), 2.4682e-17 → 2.6129e-17 (n = 48) — every value is at the round-off floor of a field that should be exactly zero |
| `vof_twophase` | **yes, and one gate had to be re-stated — see below** | |

The WO's "if any moves more than 1e-12 relative, stop and report" is written for a FUNCTIONAL. What
moved by more than 1e-12 here are **residuals and conservation defects, every one of them
downward** — the exact operator removes the float `A·1 ≠ 0` error, which is the whole point of P1.
No published functional moved: Hysing case 1 (`vof_surface_tension` gate) and the static-droplet Ca
are unchanged, the cut-cell θ and Young–Laplace numbers are unchanged, and `vof_momentum`'s K1
identity moved in its last printed digit at 1e-07 (its own floor, the float momentum diagonal).
Reporting rather than stopping, with the ablation shipped.

**The one gate that had to be re-stated: `vof_twophase` gate E, the harmonic-ρ_f ABLATION.** That
gate exists to show that a harmonic face mean is NOT the consistent one (`CHECK(perr > 1e-3)`); with
the float bands it settles at a dP/dz error of **0.3355**, and with the exact operator the same
ratio-1000 hydrostatic column **leaves the Weymouth–Yue boundedness cap before step 100 and the
advector throws**. Verified to be a property of `PECLET_FLOW_EXACT_RESIDUAL=1` alone by running the
`main` binary with the env var set: it aborts at the identical point with the identical message
(`CFL = 0.294669 ... at dt = 1`). It is a LOUDER confirmation of the same statement, so the gate now
accepts either outcome and prints which it saw. The ARITHMETIC runs in the same gate are unaffected
and in fact tighter: ratio 1000 free C, dP/dz rel-err **1.022e-15 → 4.263e-16**.

### Item 4 — the wisp guard, and two more defects it exposed

`WyAdvector::wispEps` (default **0 = V1 verbatim, bit for bit** — `wyIsMixed(c, 0)` is
`c > 0 && c < 1.0 - 0.0`, the same expression); `IbmSolver::enableVof` sets it to **1e-8**, the same
threshold the V3 curvature predicate uses. It gates the reconstruction pass, every flux branch
(`wyFaceFlux`, `wyFaceFluxBc`, `computeFluxesCut`), the momentum advector's `vofCellBox`, and the
interface Courant band. A cell outside the band is fluxed ALGEBRAICALLY as `C_donor · a` — its
ACTUAL colour — so the telescoping conservation is untouched.

**4b, the interface Courant band (new gate C2 in `tests/kokkos/test_vof_advect.cpp`).** Published
Zalesak setup, 100×100×4, one revolution (1000 steps), `interfaceLocalCfl = true` (the solver's
setting), host-openmp:

| `wispEps` | band Courant after step 1 | after 1000 | worst | volume drift |
|---|---|---|---|---|
| 0 | 0.2545 | **0.3142** | 0.3142 | 0.000e+00 |
| 1e-8 | 0.2545 | **0.2608** | 0.2608 | 8.787e-15 |

and the two numbers are both explained from first principles, which is what makes this a gate rather
than a threshold: the disk's farthest point is at r = 0.25 + 0.15 = 0.40 from the rotation centre, a
band of "mixed cells and their face neighbours" reaches r + 1.5 h, and `ω (0.40 + 1.5 h) dt/h` =
**0.2608** — the guarded run's number to the printed digit. The unguarded 0.3142 is `ω · 0.5 · dt/h`,
i.e. the band has grown to the full radius of the disk's sweep: the round-off wake. (The global max a
whole-domain limiter would report is 0.4443.) The coordinator's proposed threshold of 0.26 is
0.0008 BELOW the exact geometric bound, so the gate is stated against the bound, not against 0.26.

**Two test gates measured the difference of two predicates once the default changed, and both were
wrong gates rather than regressions.** `vof_twophase` gate A ("the solver's VoF equals a standalone
`WyAdvector` on the same LeVeque field") and `vof_cutcell` gate G3b ("solver vs standalone through
the packing") build their reference with a raw `WyAdvector`, whose `wispEps` default is 0, and
compare it against a solver at 1e-8: they read **8.481e-09** and **2.594e-09**, i.e. exactly the
scale of the threshold. Configuring the reference like the solver
(`IbmSolver::defaultVofWispEps()`, exposed for this) restores **0.000e+00** on G3b and the
projection-residual floor on gate A. Any future gate that compares the two paths must copy the
knob.

**Item 3's own gate: Hysing case 2 through the new default.** `tests/study/vof_surface_tension.py
hysing2`, 64×128×4, adaptive dt, nvidia-cuda. WO-R's `PECLET_FLOW_EXACT_RESIDUAL=1` column is
reproduced to **every printed digit** by the exact-residual default alone; adding the wisp guard
keeps both functionals and the divergence identical while taking **9.4 % fewer steps**:

| | WO-R default (float bands) | WO-R `=1` | WO-R2 exact only (`WISP_EPS=0`) | WO-R2 shipped defaults |
|---|---|---|---|---|
| `max|div(open u)|` | 1.85e-03 | **5.15e-11** | **5.15e-11** | **5.15e-11** |
| v_rise max | 0.2574 at t = 0.671 | 0.2574 at 0.671 | **0.2574 at 0.671** | **0.2574 at 0.671** |
| `y_c(3)` | 1.1082 | 1.1082 | **1.1082** | **1.1082** |
| steps to t = 3 | 1123 | 1123 | **1123** | **1017** |
| max pressure iters | 116 / 600 | 116 / 600 | **116 / 600** | 158 / 600 |
| dt-limit census (capillary / WY) | 5 / 108 | 5 / 108 | **5 / 108** | 8 / 94 |

The 1123 → 1017 is item 4b on a production case: the wisp guard narrows the interface Courant band
back to the interface, the WY limit relaxes, dt grows and the run needs fewer (larger) steps — and
the two published functionals do not move at all.

**Item 4's own gate: the emptying-domain MPI scene.** `tests/kokkos_mpi/test_vof_bc_mpi.cpp` gate A
was extended from 185 to **500 steps**, i.e. 300 steps INTO the drained regime where WO-R measured
`ΣC → -inf` at step 186 and NaN at 187. host-openmp, np 1/2/4:

| np | colour vs single-rank | ledger vs single-rank | global budget | in / out | non-finite cells |
|---|---|---|---|---|---|
| 1 | **0.000e+00** | 0.000e+00 | 1.075e-12 | 1024 / 1024 | **0** |
| 2 | **0.000e+00** | 8.476e-13 | 2.274e-13 | 1024 / 1024 | **0** |
| 4 | **0.000e+00** | 8.476e-13 | 2.274e-13 | 1024 / 1024 | **0** |

**The composed MPI scene** (new gate C in the same file: the four-sphere packing with one sphere
cutting the outlet plane, 40 coupled steps), host-openmp:

| np | colour vs single-rank | budget \|Δ Σ eps_eff C − ledger\| (rel) | ΣC over solid | pressure |
|---|---|---|---|---|
| 1 | **0.000e+00** | 4.638e-11 (**1.252e-14**) | **0.000e+00** | 16 / 400 |
| 2 | 1.166e-15 | 4.684e-11 (**1.264e-14**) | **0.000e+00** | 16 / 400 |
| 4 | 1.332e-15 | 4.638e-11 (**1.252e-14**) | **0.000e+00** | 16 / 400 |

(np = 2 cuts z, np = 4 cuts x and z, i.e. both the inflow/outflow planes and the packing.) The whole
`vof|vardensity` MPI battery is **30/30 green** on host-openmp.

### Two defects found on the way, both fixed here

1. **`advect_vof`'s divergence guard was INERT on a bare box** (reported by the E1 gallery page).
   `max_open_divergence()` returns 0.0 when `cutcellPressure_` is unset, so the guard's
   `div <= 1e-10` compared 0 against 1e-10 and passed whatever the field was: a
   cell-centre-sampled LeVeque field, whose true `max|div(open u)|` is 0.612, was accepted and lost
   **4.93 % of the liquid in 50 steps** with no diagnostic at all. `advectVofKinematic` now THROWS
   with the fix in the message unless a cut-cell pressure operator exists, and measures with the
   NON-mutating `maxOpenDivergenceProjected()` (WO-R's finding: the mutating one re-imposes the
   zero-gradient outflow face and reports a field the advector will not be handed).
2. **`tests/kokkos_mpi/CMakeLists.txt` did not PARSE.** Commit `86192ad` (WO-R) left a duplicated
   `foreach` fragment — `... vof_collocated_mpi)` followed by a bare
   `vof_surface_tension_mpi vof_cutcell_mpi vof_bc_mpi)` — so CMake refused the whole file
   ("Parse error. Expected \"(\"") and **the entire MPI battery has been unconfigurable since WO-R
   landed**, with `vof_bc_mpi` never in the build list at all. That is why WO-R's "re-run
   `vof_bc_mpi` on a quiet machine before merging" could not have been done. Merged into one
   `foreach` here; the battery is green again.

### `maxAbsDiff`'s NaN blindness, swept

`m = std::fmax(m, std::fabs(a[i] - b[i]))` returns `0.000e+00` for a field that has gone entirely
NaN, because `fmax(x, NaN) == x` — so every bitwise gate built on it passes on the worst possible
field. WO-R fixed its own copy; the other **13** in `tests/kokkos` / `tests/kokkos_mpi` now return
the non-finite difference (`test_vof_collocated`, `test_vof_momentum`, `test_vof_surface_tension`,
`test_vof_twophase`, and the MPI `bodyforce_ghost`, `dragbeta_ghost`, `vardensity`, `varmu`,
`vof_collocated`, `vof_cutcell`, `vof_momentum`, `vof_twophase`, `vof_wetting`). **Not swept:** the
sibling `maxAbs(v)` helpers in the same files have exactly the same hole; they are a magnitude
report rather than a gate in most of their uses, so they are recorded here rather than changed.

### Open, and what the coordinating session should decide

1. **G3's quiescence half still fails** and is now the sharpest open two-phase item on an open
   boundary: at ratio 1000 a resting pool under a gas stream loses 1.03e-03 of its volume in 500
   steps and picks up 3.1e-02 of the inlet speed. The run is VALID now (76/800), so it is a clean
   measurement of a spurious-current / momentum-consistency defect rather than of a capped solve.
2. **The ratio-1000 stratified duct still caps** (400/400) while the Nusselt film and the pool at
   the same ratio converge in 64 and 76 iterations. The cap is unaffected by this WO (it caps
   identically in the operator ablation) and belongs to the S3 coefficient-coarsening item.
3. **MG telescoping and two-phase open boundaries are mutually exclusive** until the telescope
   stage carries the boundary face plane. Currently a throw.
4. **`max_open_divergence()` still mutates** (WO-R open question 2, unchanged — it is a
   re-baselining decision, and `advect_vof` no longer depends on it).
5. **The composed MPI gate is at the reduction floor, not bitwise**, for the structural reason
   above. If a bitwise composed gate is wanted, the velocity bridge needs a `setField` variant that
   carries the outflow face.

## WO-T — rung V8 minimal (the collocated path) — DONE 2026-09-02, branch `vof-wot`

Worktree `../flow-wot`. All numbers below are reported for BOTH backends where they differ;
`host-openmp` and `nvidia-cuda` agree to the digits shown unless a column says otherwise.

### What shipped

`src/collocated_varrho.hpp` (five sibling kernels: the face acceleration, its CSF addend, the
`uf += af` apply, the `af -= (rho0/rho_f) grad(phi)` completion, and the openness-gated cell
average), the collocated branches in `flow_ibm.hpp` (`buildRhsColoFF`, `applyFaceAcceleration`,
`applyCellFaceAverageCorrection`, `colocatedFaceForce`, `requireCollocatedFaceForceScope`,
`collocatedV8AutoFallback`, the `Grid::collocated ? 0 : strideOf(c)` placement in `makeFaceProps`,
the collocated branch of `bridgeVelocityToVof`), the lifted throws in `setDensityMode` / `enableVof`,
the binding docstrings, `tests/kokkos/test_vof_collocated.cpp`,
`tests/kokkos_mpi/test_vof_collocated_mpi.cpp` and `tests/study/vof_collocated.py`.

Design items 1-3 and 5 are implemented as written. Item 4 (`enable_vof_momentum`) stays a throw, as
the work order specifies.

### Gates

| gate | measured | verdict |
|---|---|---|
| **G1 hydrostatic** (ratio 1000, mu = 0, 100 steps, frozen interface, 8x8x24) | walled column, hand-set rho AND through C, identical numbers: face `max|uf|` **8.131e-15** (host) / **9.228e-15** (CUDA), `dP/dz = -rho_f g` to **2.341e-8** relative, 13 Chebyshev iterations (cap 120, none capped). Staggered column on the same problem: face 2.176e-17 / 1.821e-17, dP/dz 3.411e-16, 10 iterations. Triply periodic box with the zero-mean force: collocated face **3.343e-10**, dP/dz 6.149e-9, 11 iterations (staggered 1.767e-14 / 4.263e-16). **The CELL field is 2.821e-08 — see finding 1; the gate as written ("cell max|u| <= 1e-12") measures a quantity the ABC projection is structurally blind to, and the corrected gate is the FACE field** | PASS on the corrected quantity |
| **G1b mu independence** (the collocated bonus) | walled, ratio 1000, mu = 0 / 1e-3 / 1e-2 / 1e-1: collocated face **8.1e-15 / 3.4e-15 / 1.7e-15 / 5.1e-15**, dP/dz 2.34e-8 / 2.01e-8 / 2.29e-8 / 4.19e-8 — flat. The force never passes through `A = rho_f/dt - mu*Lap`, so the mu*dt^2 non-commutation WO-P measured on the staggered predictor does not exist on this path | PASS |
| **G2 static droplet, constant kappa** (32^3, R = 8, sigma = 1, mu = 0.1, dt = 0.5 dt_sigma, 30 steps) | ratio 1: staggered cell/face **1.9291e-17**, COLLOCATED cell **2.2565e-17** face **2.1751e-17** — the V4 exactness identity, reproduced at the face AND at the cell. ratio 10: staggered 2.0374e-05, COLLOCATED **1.6316e-11 cell / 1.2100e-11 face** (6 orders better). ratio 100: staggered 1.8377e-05, COLLOCATED **1.5901e-06 / 2.4928e-06**. ratio 1000: staggered 1.0485e-05, **COLLOCATED UNSTABLE** (see finding 2) | PASS at ratio <= 100 |
| **G3 static droplet, computed kappa** (`tests/study/vof_collocated.py static`, 60 steps, mu = 0.1, sigma = 1) | `Ca = mu max|u|/sigma`: D/dx = 8 staggered **2.543e-04**, COLLOCATED **2.018e-04** cell / 1.942e-04 face (**0.79x** the staggered); D/dx = 16 staggered **5.898e-05** (WO-P's recorded number to four digits), COLLOCATED **4.834e-05** cell / 4.706e-05 face (**0.82x**). 12-13 Chebyshev iterations (cap 500, none capped), `max|div(open u)|` 1.6e-15 / 2.4e-16. The gate asked for "within 2x of the staggered 5.9e-5" — it is 0.82x, i.e. slightly BETTER | PASS |
| **G4 Hysing rising bubble, case 1** (ratio 10, quasi-2D 32x4x64, adaptive dt to T = 3, 719 steps, `tests/study/vof_collocated.py hysing1 --quick`) | `enable_vof_momentum` is staggered-only at this rung, so the honest comparison is the momentum-consistency-OFF pair: staggered **v_rise 0.2829 at t = 1.090, y_c(3) = 1.2002**; COLLOCATED **v_rise 0.2795 at t = 1.036, y_c(3) = 1.2098** — **-1.19 % / +0.79 %**, inside the gate's 3 %. (Staggered with V2b ON, for scale: 0.2501 / 1.0844 at this grid — WO-Q finding 11's nx = 32 numbers to every digit; the published reference is 0.2417 / 1.0810 at nx = 64.) 23-24 pressure iterations against a cap of 600 on all three, `max|div(open u)|` 5.5e-06 identical across the three runs | PASS |
| **G5 capillary wave** (32 cells/lambda, nu = 0.005, 2.5 periods, `tests/study/vof_collocated.py wave`) | staggered **omega = 0.06017** (WO-P's recorded number to every digit; -0.04 % against the EXACT viscous two-fluid mode 0.06019), COLLOCATED **omega = 0.05974** (-0.74 % against the exact mode) — the two grids differ by **0.71 %**, inside the gate's 1 %. Decay rates 1.087e-03 vs 1.075e-03 (both ~-14 % against the exact 1.264e-03, i.e. the deviation is the SCHEME's, not the grid's). 12/500 pressure iterations on both, `max|div(open u)|` 2.0e-15 | PASS |
| **G6 advection bit-identity** (24^3, an ABC cell velocity projected once, then 20 kinematic `advect_vof` steps on the frozen face field) | the collocated `max|div(open uf)|` after the seeding step is **0.000e+00**; the colour after 20 steps is **bitwise identical** to a STAGGERED solver handed the same `uf/vf/wf` through `set_field("u"/"v"/"w")` — `max|dC| = 0.0`, `sum C` 9.110000000000001e+02 on both — while the colour genuinely moved (`max|C - C0| = 1.0`). This is the gate on the collocated bridge: `uf_(i)` sits at i-1/2, exactly where flow's staggered `u(i)` sits, so it is the same `copyFaceVelocity` shift on a different source view | PASS (bitwise) |
| **T3 uniform-rho equivalence** (16x16x8, mu = 0, body force, 30 steps) | the V8 face-force predictor vs the VALIDATED constant-density collocated path: `max|du| = 0.000e+00`, `max|dP| = 0.000e+00` — **bitwise**. With uniform rho and mu = 0, `avg_f(P(i) - P(i-s))` IS the central difference, and the two schemes coincide to the last bit | PASS (bitwise) |
| **G7 MPI np = 1 / 2 / 4** (`tests/kokkos_mpi/test_vof_collocated_mpi`, 16x16x32, host-openmp; the ORB cuts z at np = 2 and xz at np = 4, i.e. the stratification axis, the walls and the interface all sit on a rank boundary) | Two configurations against a full-grid single-rank reference. **np = 1: `du`, `duf`, `dP`, `dC` all 0.000e+00 — bitwise** on both (`hydro-z` walled ratio 1000 through a frozen colour, 20 steps; `vof-z` periodic ratio 10 with `enable_vof` + advection + a body force, 10 steps, reference `|u| = 1.0e-02`). **np = 2: `hydro-z` du 1.355e-14, duf 6.255e-15, dP 5.116e-13 (|P| 1150), dC 0.0; `vof-z` du = duf = dP = dC = 0.000e+00 — still BITWISE**. **np = 4 (blocks 8x16x16, xz cut): `hydro-z` du 1.217e-14, duf 7.438e-15, dP 1.080e-12, dC 0.0; `vof-z` all 0.000e+00 — BITWISE again.** Iteration counts identical at every step and every np. On **nvidia-cuda** the same test is **fully bitwise at np = 1 and np = 2** (du = duf = dP = dC = 0.0 on both configurations) and at np = 4 reads `hydro-z` du 1.488e-14, duf 8.704e-15, dP 5.116e-13, dC 0.0. The face-acceleration plane at index `e-g` is therefore decomposition-independent, which is the thing this test exists for; note that the tolerance for `hydro-z` is relative to the FORCING scale g*dt, not to the (essentially zero) rest-state velocity — see finding 8 | PASS |
| **G7b byte-identity of everything existing** | `tests/regression/sdflow_regression.py` (CUDA, staggered): **`+0.00 %` on every metric of all three cases, identical iteration counts AND identical step counts** — `zh_sphere` K 7.2997/7.3891/7.4162/7.4361/7.4404 with order 2.29 and K_inf 7.447, `random_spheres` order 2.19 and k*_inf 0.0062362, `hollow_rings` order 1.38 and k*_inf 0.017184, every row `[ok]`. **The COLLOCATED regression (`--solver colocated --scheme ghost`, the constant-density path this rung must not disturb) is likewise `+0.00 %` on every metric with identical iteration counts** — `zh_sphere` order 1.59 / K_inf 7.445, `random_spheres` order 1.46 / k*_inf 0.0062399, `hollow_rings` order 1.75 / k*_inf 0.017201, `=== regression: PASS ===`. All **27** pre-existing `tests/kokkos` binaries reproduce their `main` output **digit for digit** on **BOTH** backends (`diff` of the full stdout against the same tests built at `main` in a second worktree: *27 identical, 0 differing* on host-openmp AND on nvidia-cuda) | PASS |
| **battery green** | the whole `tests/kokkos` suite on host-openmp: **28/28 tests passed**, including the new `vof_collocated` (137 s); on nvidia-cuda `test_vof_collocated` reproduces every host number to the digits printed (the only difference anywhere is the last digit of a colour SUM, a reduction-order effect) | PASS |
| **T5 scope** | `enable_vof` + immersed solid, `set_density_mode` + immersed solid, `enable_vof_momentum`, `set_rho_face_harmonic` — all four throw on `SolverColocated` with a message naming the reason | PASS |

### Findings

**1. The invisible subspace showed up exactly where the plan said it would, and this rung makes it
six orders SMALLER than the collocated status quo.** A cell-field checkerboard is annihilated by
`centerToFace` (`½(U(i)+U(i-1))` kills the odd-even mode), so the approximate projection cannot see
it and cannot remove it. Signature, measured on the walled hydrostatic column at ratio 1000, mu = 0:
the FACE field is at machine zero (8.1e-15) while the CELL `w` alternates sign cell by cell along z
with an envelope peaking at the density interface — `[-1.8e-11, +2.7e-11, -4.6e-11, ...]` at ratio
2 — i.e. precisely "a growing checkerboard in the CELL field while the FACE field stays clean",
except that it **decays**: 2.821e-08 at 100 steps -> 8.006e-09 at 400 (3.52x), roughly as 1/n.
It is a property of the GRID, not of this rung, and the control says so: the VALIDATED
constant-density collocated path with a plain body force on the same walled column reads cell
`max|u|` **3.092e-02** (mu = 0) and **8.709e-04** (mu = 0.01) with its face field at 5.1e-13, so
the V8 ratio-1000 number (2.8e-08) is **six orders better than the constant-density collocated
baseline** on the same problem. Nothing damps the mode in the heavy phase (its kinematic viscosity
is mu/rho = 1e-4 there), which is why the decay is algebraic rather than geometric.
The accumulated pressure inherits it: `centerToFace` does not annihilate a MODULATED checkerboard,
it leaves the envelope's gradient, so every step injects a small potential into `P +=
(rho/dt) phi`; that leak, not the face balance, is the 2.3e-8 relative `dP/dz` error (the
two-face-averaged pressure statement does NOT remove it — 1.24e-6 absolute vs 2.34e-6 raw — so it
is not a pure odd-even artefact of the read-out).
**The corrected G1 gate** is therefore: FACE `max|uf|` at machine zero, cell `max|u|` reported with
its decay rate and against the constant-density control. Gating the cell field at 1e-12 on an ABC
grid asks the projection for something it structurally cannot deliver.

**2. The collocated rung's real ceiling is not the density ratio, it is `mu*dt/(rho_min h^2)` — and
it is the price of the face force being explicit.** Static droplet, constant kappa, ratio 1000,
dt = 0.5 dt_sigma = 4.46, 40 steps:

| mu | mu*dt/(rho_gas h^2) | staggered face | COLLOCATED face |
|---|---|---|---|
| 0     | 0     | 1.3483e-10 | **4.2335e-11** |
| 0.01  | 0.045 | 3.9129e-06 | **5.4329e-11** |
| 0.1   | 0.45  | 6.7815e-06 | **UNSTABLE** — 7.2e-09 at step 0, flat to step 5, then ~4x per step: 1.2e-05 at step 10, 3.3e-02 at step 15, trips the Weymouth-Yue CFL cap at step 16 |

Mechanism: on the staggered path the force enters the RHS of the momentum solve, so `A^-1` (with
`A = rho_f/dt - mu*Lap`) DAMPS its high-wavenumber content — that damping is exactly what makes the
staggered balance inexact at variable rho (WO-P's mu*dt^2 residue), and it is also what stabilises
it. Applying the force at the face, outside `A`, removes both: the balance becomes exact (the
mu-flat G1b row, and the 4-6 orders of accuracy at mu <= 0.01 above), and the face and the cell are
then advanced by different operators — the face by the raw increment, the cell by `A^-1` followed by
the AVERAGED increment — whose mismatch grows once the per-step viscous smoothing stops being small.
`advect_` is OFF by default and was off here, so this is not an advective instability. **Rating:
the collocated rung is honest to density ratio ~100** (measured stable and better than staggered
there), or ratio 1000 with `mu*dt/(rho_min h^2) <~ 0.05`. The obvious next move, NOT taken here
because it is a design change rather than an implementation one, is a viscous-augmented explicit-force
time-step limit (Galusinski & Vigneaux's combined capillary/viscous criterion) exposed alongside
`capillary_dt()`.

**3. The AUTO collocated scheme has to be re-decided when `set_density_mode` / `enable_vof` runs
after the geometry.** `setSolid`/`setPressureGeometry` picks GHOST vs gauge-exact from the
configuration it sees, and the ghost projection v1 supports neither. Calling `set_density_mode`
after `set_pressure_geometry` — the natural order, and the one every gate here uses — left
`ghostProjection_` on and produced a runtime throw from inside `project()` on the first step.
`collocatedV8AutoFallback` re-runs the same fallback (with the same stderr notice) from
`setDensityMode(true)` and from `enableVof` on the collocated grid. All-fluid, gauge-exact IS the
plain central difference, so nothing else changes.

**4. A closed face must contribute ZERO to both the face field and the cell average, and the two
have to be pinned together.** Written without the openness gate, the wall face of the hydrostatic
column carried `af = dt*g = 0.1` forever (its flux is multiplied by openness 0, so the projection
never touches it) — harmless for the pressure solve, fatal for anything that READS the face field,
which after item 3 of this work order includes the colour transport. `buildFaceAccelVar`,
`addFaceAccelCsf` and `faceAccelSubGradPhi` all skip `o <= 1e-12` and pin `af = 0` there, and the
cell average uses `projectCorrectCenter`'s identical predicate. The consequence worth stating: the
near-wall cell sees HALF of its open face's increment, exactly as `projectCorrectCenter` already
halves the near-wall pressure gradient — and at equilibrium the open face's total increment is zero,
so the wall cell stays at rest.

**5. `VarFaceProps::idiag` needed the velocity-unknown placement, not the staggered stride.** The
accessor's `sc` is the component face stride, so `idiag(i) = 0.5*(rho(i)+rho(i-sc))/dt` — right for
a face-placed unknown, wrong for a cell-placed one. `Grid::collocated ? 0 : strideOf(c)` gives
`0.5*(rho(i)+rho(i)) == rho(i)` exactly in floating point. It was unreachable before this rung
(`haveRho` required varRho or porous, both of which threw on the collocated grid), which is why it
had never been wrong in practice.

**6. The periodic hydrostatic box is NOT a free extra gate — with a free interface it is
Rayleigh-Taylor unstable by construction.** A periodic two-layer column always has exactly one
unstably stratified interface, so the "periodic, through C" case that the first draft of the gate
ran measured an RT mode, not a balance (it read 2.4e-09 on the STAGGERED grid, where the walled
version reads 1.8e-17). The shipped periodic case uses a FROZEN hand-set rho, where the density is
not a degree of freedom; the through-C case is walled, heavy below, and frozen the way
`test_vof_twophase.cpp` gate B2 freezes it.

**8. Two MPI gates in the first draft measured the wrong quantity, and both are the same mistake:
a tolerance relative to the ANSWER on a problem whose answer is zero.** (a) The velocity scale was
taken from the z component alone, which is identically 0 in the x-driven `vof-z` configuration, so
the gate demanded a bitwise match of the MPI reduction order on an O(1e-2) field; it is now the max
over all three components. (b) `hydro-z` is a REST state — its converged `|u|` is 1.4e-07, and that
is the invisible checkerboard, not physics — so a tolerance of `1e-11 x |u|` is `1e-18` and no
distributed run can meet it. The scale that means something for a rest state is the velocity the
body force would produce in one step, `g*dt`, which is exactly what the projection has to cancel;
against that the measured np=2/4 deviations (1.2-1.4e-14) are `~1e-13` relative, i.e. the ordinary
reduction-order floor. Recorded because the same trap is latent in every "acid test at np > 1".

**9. This MPI test is EXPENSIVE for what it gates, and the reason is the reference, not the
distributed run.** Every configuration runs the distributed case AND a full-grid single-rank
reference on rank 0, and under variable density the Chebyshev bounds are re-estimated every step
(~45 extra V-cycles), so 40 + 40 steps of a 16x16x32 collocated ratio-1000 column took >15 minutes
on a loaded host. The shipped configuration is 20 steps (`hydro-z`) + 10 (`vof-z`), which is enough
for the bitwise statement; if it is still slow in CI the reference is the thing to shrink.

**7. MG-PCG diverges outright on the collocated ratio-1000 coefficient operator.** Not a V8 defect —
it is the known high-contrast indefinite-preconditioner item (`CLAUDE.md`, VOF_PLAN S3) — but it is
worth the line because `set_density_mode` installs Chebyshev by default and a reader may be tempted
to override it: `set_pressure_pcg(True, 800, 1e-15)` on the ratio-1000 walled column reaches
`max|u| = 5.2e+03` after 800 iterations, while Chebyshev at the same tolerance is at 1e-15 in 16.
The solver tolerance is NOT what limits any number in this rung: at rtol 1e-9 (the default) and
1e-15 the hydrostatic numbers agree to three digits.


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
## WO-R — rung V-BC: two-phase open boundaries  [Opus, 2026-09-02]

Branch `vof-wor`. Deliverables shipped: `src/vof/colour_bc.hpp`, `wyFaceFluxBc` + the `outside`
mask + the per-face boundary ledger in `src/vof/advect_wy.hpp`, `bcCorrectOutflowVar` in
`src/mac_bc.hpp`, `applyClosureFaceGhost` in `src/property_closures.hpp`, the solver plumbing +
bindings, `tests/kokkos/test_vof_bc.cpp`, `tests/kokkos_mpi/test_vof_bc_mpi.cpp`,
`tests/study/vof_open_boundaries.py`, the CLAUDE.md paragraph.

### The headline: the varRho pressure operator's DOMAIN-BOUNDARY face coefficient is 1, and the projection's correction at that face is not

This is the real answer to `doc/variable_density_projection.md` §4, and it is not the one item 4
assumed. `CutcellMG::applyBoundaryOpenness` re-imposes the boundary face value on **every level,
level 0 included**: `bcSetOpenness(..., 1.0)` at an outflow face. Under `varRho` the field handed to
`setOpenness` is the *coefficient* `cx1_ = open·ρ₀/ρ_f` (`buildRhoCoeff`), so that line overwrites a
correct `ρ₀/ρ_f` with **1**. The two projection corrections then disagree with it in opposite ways:

| outflow face | operator coefficient | correction applied | consistent? |
|---|---|---|---|
| **high** side (`bc_[2a+1]==3`) | 1 (`bcSetOpenness`) | 1 (`bcCorrectOutflow`, plain φ difference) | **yes** |
| **low** side (`bc_[2a]==3`) | 1 (`bcSetOpenness`) | `ρ₀/ρ_f` (`projectCorrectVar`, the face is an inner index) | **no — by the full ratio** |

Measured, both halves:

* **high side** (stratified duct, walls ±z, inflow −x, outflow +x, 5 steps, `max|div(open u)|` of the
  **projected** field, `tests/kokkos/test_vof_bc.cpp` gate F2):

  | density ratio | plain correction (shipped) | with `1/ρ_f` (`bcCorrectOutflowVar`) |
  |---|---|---|
  | 1    | 1.407e-17 | 1.407e-17 (bitwise equal) |
  | 10   | **8.763e-10** | **9.236e-03** |

  So **WO-R item 4 is REFUTED**: adding the `1/ρ_f` factor to `bcCorrectOutflow` makes the outflow
  divergence seven orders worse, because it breaks the accidental consistency with the
  coefficient-1 operator row. `bcCorrectOutflowVar` ships as `set_outflow_rho_correction(...)`,
  **default OFF**, with these numbers in its docstring. The corrected item is: *fix the operator*.

* **low side** (Nusselt falling film, outflow at −z, ratio 100, one step from the exactly developed
  state): `max|w|` **1.4550** against the film's own `u_max` 0.3124 — a 4.7× velocity in the GAS,
  where `ρ₀/ρ_f = 100`, and `max|div|` (projected) **1.44**. The run trips the Weymouth–Yue cap
  within 20 steps. The `set_outflow_rho_correction` knob is *inert* here (it only touches the high
  side), which is the cross-check that this is a second, independent instance of the same
  inconsistency.

**Proposed fix (NOT implemented — out of this rung's deliverables, and it deserves its own WO):**
in `CutcellMG::applyBoundaryOpenness`, impose at an outflow face the *coefficient* the caller
supplied rather than the literal 1 (level 0 already has the right value from `buildRhoCoeff`; the
coarse levels need the coarsened one). Then both corrections are the `ρ₀/ρ_f` one and
`bcCorrectOutflowVar` becomes correct rather than harmful — i.e. item 4's design is right *after*
the operator is fixed and wrong before it. Gate for that WO: F2's table must read ~1e-10 in the
`with 1/ρ_f` column and the Nusselt film must run to steady state.

### Two more defects found, both fixed here

1. **`max_open_divergence()` MUTATES the velocity field on the staggered path.** It calls
   `fillVelGhosts(c, 0)` before measuring, which re-imposes the zero-gradient outflow face and so
   **destroys `bcCorrectOutflow`'s correction** — the mechanism by which mass leaves. Calling it
   once per step inside a time loop therefore changes the run. Measured:
   * constant-density duct: the mutating diagnostic reads **1.262e-09**, the projected field's own
     residual is **1.407e-17**;
   * the coupled colour-budget study (32×32×64, 800 steps) with the mutating diagnostic in the loop:
     budget drift **2.7e-12** relative, `max|div|` 1.8e-08; with the non-mutating one:
     **4.589e-15** and **1.77e-13**. Three to four orders, purely from the diagnostic.
   * and it is what made the Nusselt film look stable in the first probes: resetting the
     inconsistent outflow correction every step suppressed the blow-up above (`max|w|` 0.3078 with
     the mutating diagnostic in the loop against 1.4550 without).
   Fix: **`max_open_divergence_projected()`**, a non-mutating sibling. The default was left alone —
   every recorded open-boundary number in the repo was taken with the mutating one and
   re-baselining them is not this rung's call.
2. **`bridgeVelocityToVof` erased the same correction**, so the colour advector was handed a field
   that is not discretely divergence-free at the outlet — exactly the hypothesis Weymouth–Yue's
   exact conservation rests on. Fixed with `fillVelGhostsKeepOutflow`, selected by a
   `outflowCorrValid_` flag so the KINEMATIC path (no projection since the last full fill, the
   boundary face never set) still gets the zero-gradient fill that supplies that face at all.

### Gates

**G1 colour budget — PASS.**
*Kinematic* (`tests/kokkos/test_vof_bc.cpp` gate C, 32×32×64, uniform inflow −z / outflow +z, walls
elsewhere, `set_vof_inflow(4,1)` for 100 steps then 0 for 400, advecting field the exactly
divergence-free uniform w):

| backend | injected | left | remaining ΣC | budget drift (rel) | in/out closure | C range |
|---|---|---|---|---|---|---|
| nvidia-cuda | 20480 | 20480 | −7.3e-25 | **2.487e-15** | 8.882e-16 | [−6.9e-18, 1] |
| host-openmp (quick, nz=32) | 8192 | 8192 | −8.4e-20 | **4.774e-15** | 8.882e-16 | [−6.1e-18, 1] |

The slug leaves with its length intact: at its fullest, ΣC = 20480 over **19 full planes + 2
partial** for an injected length of 20 cells. `max|div(open u)|` of the prescribed field 0.000e+00.

*Coupled* (`tests/study/vof_open_boundaries.py budget`, the same box driven by `step()`, 60 + 740
steps, μ = 0.5, Chebyshev): injected 6144, budget drift **2.819e-11 absolute = 4.589e-15 relative**,
`max|div(open u)|` (projected) **1.77e-13**, pressure **22/400** (not capped), C ∈ [−2.29e-16,
1+1.3e-15].

*Corrected gate.* WO-R's "nothing is left behind (ΣC → 0 to 1e-12)" is only meaningful with a
**uniform** advecting field. In the no-slip duct the near-wall liquid never leaves (the coupled run
ends with 1177 of 6144 still inside), so that half belongs to the kinematic twin — where it reads
−7.3e-25 — and the coupled gate should keep only the budget identity. Also: WO-R's 1e-12 relative
tolerance is *looser* than what the advection achieves; the measured floor is the projection's
divergence residual and it is 2.5e-15.

**G2 Nusselt falling film — FAILED TWICE, STOPPED per rule 4.**
Quasi-2D 32×4×64, δ = 8 cells against the −x wall, walls ±x, prescribed Nusselt inlet at +z,
outflow at −z, reduced gravity `−(ρ_l−ρ_g) g C` (so the gas is exactly force-free and hydrostatic,
which is what the similarity solution assumes; with gravity on the gas the Dirichlet p = 0 outlet
accelerates the gas column instead of supporting it — a different problem). Film Reynolds 5,
`u_max` 0.31236, Nusselt `Q` 1.66595.

*What was measured before the failure* — 60 steps from the exactly developed state, at z = nz/2:

| ratio | μ ratio | momentum consistency | δ measured | Q measured | vs Nusselt | pressure | max\|div\| (projected) |
|---|---|---|---|---|---|---|---|
| 100  | 50 | on | **7.9960** (−0.0040 cells) | 1.6426 | **−1.40 %** | 66–154 / 400, valid | 2.13e-05 |
| 1000 | 50 | on | **8.0000** | 1.6604 | **−0.34 %** | 400/400 and 953/2000 — **CAPPED, run INVALID** | 2.96e-04 |

So the film physics is well inside the 3 % / 0.5-cell gate at both ratios, and neither run reaches
"steady state to 1e-4 over the last 200 steps": at ratio 100 the run trips the Weymouth–Yue cap
(the low-side outflow inconsistency above), at ratio 1000 the pressure solve caps.

*Attempts, both recorded rather than retried further:* (i) fixed dt = 0.5 sized on the analytical
`u_max` → CFL 0.571 at step ~20; (ii) adaptive dt re-picked every 20 steps from
`vof_step_limits()['cfl_dt']` (the `gate_hysing` pattern) → dt 0.321, CFL 0.317 at step ≤ 20. The
mechanism is not dt: `max|w|` is already 1.4550 after the FIRST step.

*The 1/ρ_f before/after at ratio 1000, explicitly, as asked.* It cannot be taken on this case: the
film's outlet is on the **low** side of z and `bcCorrectOutflowVar` only touches the high side, so
the knob is measurably inert there (`max|w|` 1.4550 and `max|div|` 1.44 after one step with the knob
both on and off, to every printed digit). The before/after that does exist is the high-side one in
the table at the top of this entry (ratio 10: **8.763e-10 off / 9.236e-03 on**), and at ratio 1000
the same high-side comparison is unavailable for a different reason: every ratio-1000 wall-bounded
open-boundary configuration tried caps its pressure solve, so no valid run exists to compare.
**Corrected gate for G2:** it must wait for the operator fix above; as written it measures the
operator defect, not the boundary machinery.

*Pressure-driver facts G2 established and every later open-boundary case needs:*
* on a wall-bounded open-boundary box at ratio ≥ 100, **Chebyshev DIVERGES** (the film accelerates
  until the WY cap throws), **MG-PCG burns any cap** (600/600, 2000/2000) and **FCG converges**
  (98–154 iterations at ratio 100). Ratio 1000 caps every driver.
* **the driver must be selected AFTER the `rho` closure.** `set_property_model("rho", …)` fires
  `set_density_mode`, which reselects Chebyshev and silently discards an earlier
  `set_pressure_pcg/fcg`. This is WO-H's "capped at 120" tell, alive and well, and it cost two
  probe runs here.
* the ratio-1000 cap is **not** WO-M's float `A·1 = 0` defect. Measured on the G3 pool
  configuration (32×4×16, ratio 1000, FCG cap 800, 40 steps) against a
  `-DPECLET_FLOW_MREAL_DOUBLE` build of the same tree: the per-step iteration counts and
  divergences agree to **every printed digit** from step 1 onward (step 0 reads 112 float / 120
  double), e.g. step 39 = 634 iterations and 4.821e-03 in both builds. So it is the coarsening /
  boundary-coefficient side, i.e. the S3 family, not the operator storage precision.

**G3 gas over a pool — item-5 half PASS, quiescence half FAIL (run INVALID).**
64×4×32, gas inflow at −x above a liquid pool filling the lower half, outflow at +x, ratio 1000,
`enable_vof_momentum`, 500 steps, FCG cap 800.
* **the inflow ghost density is exactly the inlet fluid's**: 1 above the pool (want 1), **1000 at
  the pool's own inlet plane** (want 1000). That is WO-R item 5, and it is the half of G3 that
  measures the boundary machinery.
* pool volume 4096 → 4084.47 (**−2.815e-03** relative, gate 1e-10); `max|u|` in the liquid
  1.2494e-02 = **3.123e-02 of the gas inlet speed** (gate 1e-3); pressure **800/800 CAPPED → run
  INVALID** (rule 3b); `max|div|` (projected) 8.55e-05.
* *Corrected gate.* WO-R asks for the ghost density "through `get_field('rho')` on the first inner
  plane". That measures the wrong quantity — `get_field` returns the INNER cells, whose ρ is
  ρ(C_inner) whatever the ghost policy does. The ghost is what the inlet FACE density is built
  from and is reachable only through `field_view('rho')` (the padded buffer). The script gates that.

**G4 MPI — PASS at np 1/2/4 on BOTH backends** (16×16×32; the aligned ORB cuts z at np = 2 and
**x and z** at np = 4, i.e. the inflow and outflow faces are cut):

| backend | np | slug colour vs single-rank | ledger vs single-rank | global budget \|ΣC − ledger\| | in / out | coupled-jet colour | inflow ρ ghost | pressure |
|---|---|---|---|---|---|---|---|---|
| host-openmp | 1 | **0.000e+00** | 0.000e+00 | 1.845e-12 | 1024 / 1024 | **0.000e+00** | correct on all owners | 42/400 |
| host-openmp | 2 | **0.000e+00** | 8.810e-14 | 1.933e-12 | 1024 / 1024 | 6.173e-14 | correct | 42/400 |
| host-openmp | 4 | **0.000e+00** | 3.155e-13 | 2.160e-12 | 1024 / 1024 | 5.473e-14 | correct | 42/400 |
| nvidia-cuda | 1 | **0.000e+00** | 0.000e+00 | 1.768e-12 | 1024 / 1024 | **0.000e+00** | correct | 42/400 |
| nvidia-cuda | 2 | **0.000e+00** | 5.112e-14 | 1.819e-12 | 1024 / 1024 | **0.000e+00** | correct | 42/400 |
| nvidia-cuda | 4 | **0.000e+00** | 5.112e-14 | 1.819e-12 | 1024 / 1024 | 5.751e-14 | correct | 42/400 |

(The host-openmp rows above are from the 20 + 200-step version; the shipped test runs 20 + 165, and
its host-openmp rerun reproduces the np = 1 row exactly. See the wisp finding below for why the
length changed. All six rows were taken BEFORE the rebase onto WO-Q; after the rebase the
host-openmp np = 1 and np = 2 rows reproduce to every digit and `vof_bc` + WO-Q's `vof_cutcell`
both pass, but the np = 4 rerun was still queued behind other sessions' MPI batteries when this
branch was pushed — re-run `ctest -R vof_bc_mpi` on a quiet machine before merging.)

The colour is **bitwise** at every np on the kinematic slug (no reduction in the update); the
coupled jet sits at the usual allreduce-order floor. The ledger is a per-rank partial sum over the
global faces a rank OWNS — without that test a rank in the middle of the decomposition counts its
own block-boundary flux and the per-face totals are meaningless (measured: np = 2 reported an
inflow of 2048 instead of 1024 before the ownership gate).

**G5 inert — PASS.** With no VoF boundary colour set, all six VoF ctests (`vof_plic`, `vof_advect`,
`vof_twophase`, `vof_momentum`, `vof_curvature`, `vof_surface_tension`) are **byte-identical** to
their pre-change output on **both** host-openmp and nvidia-cuda (`diff` on the full stdout, 12 of
12 identical). The single-phase regression is **PASS with +0.00 % on every metric and identical
iteration counts** on all three beds (`zick_homsy`, `random_spheres`, `hollow_rings`).

### A latent V0/V1 fragility this rung was the first to reach: WY wisps in an EMPTIED domain

An open-boundary domain can drain completely; no earlier VoF configuration could. When it does, the
colour field is nothing but Weymouth-Yue round-off residue, and `wyIsMixed(c)` — `0 < c < 1`, with
**no wisp guard**, unlike the V3 curvature predicate's `interfaceEps` — still calls those cells
mixed. The MYC normal of a stencil of ±1e-18 values is degenerate and `plicAlpha` divides by it.

Measured, nvidia-cuda, np = 1, the kinematic slug run continued past the drain
(`PECLET_VOF_BC_TRACE=1` in `tests/kokkos_mpi/test_vof_bc_mpi.cpp`):

```
step 181  outflow flux -0.191507   ΣC  5.047e-04   min -5.6e-18  max  6.3e-05
step 185  outflow flux  3.891e-18  ΣC -1.460e-17   min -2.3e-18  max  4.3e-19
step 186  outflow flux  -inf       ΣC -inf
step 187  outflow flux  +inf       ΣC  NaN         1 non-finite cell
```

Not caused by this rung: `wyFaceFluxBc` is bitwise `wyFaceFlux` wherever the mask is 0, and the
mask only covers out-of-domain ghosts, while the degenerate donor here is an inner cell. It is
also round-off-luck: the single-rank nvidia-cuda run of the SAME drain (the `vof_bc` ctest, 500
steps) ends at ΣC = −7.3e-25 with no non-finite cell, and host-openmp never trips it — the
difference is the colour ghost fill (`GridHalo` vs `periodicFill`) perturbing the wisps.

**A gate defect it exposed, fixed here:** the standard `maxAbsDiff` loop uses `std::fmax(m, |a−b|)`,
and `fmax(x, NaN)` returns `x` — so a field that has gone entirely NaN compares **0.000e+00** to its
reference and every bitwise gate passes. `tests/kokkos_mpi/test_vof_bc_mpi.cpp` now propagates NaN
and counts non-finite cells explicitly. **Every `maxAbsDiff` in `tests/kokkos*` has this hole.**

The shipped MPI test stops at step 185, just after the slug has fully left (in == out == 1024), so
it exercises the budget without entering the wisp regime. The proposed repair — a wisp guard on
`wyIsMixed`, or clipping |C| below ~1e-12 to 0 after each sweep — changes a validated V1 predicate
that reconstruction and fluxing must share, so it belongs in its own work order with its own
byte-identity argument.

### Item 6 (measurement only, no default changed): Hysing case 2 with and without `PECLET_FLOW_EXACT_RESIDUAL=1`

`tests/study/vof_surface_tension.py hysing2`, 64×128×4, adaptive dt, nvidia-cuda, identical in every
other respect.

| | default | `PECLET_FLOW_EXACT_RESIDUAL=1` |
|---|---|---|
| max pressure iterations | **116 / 600** | **116 / 600** |
| `max\|div(open u)\|` | **1.85e-03** | **5.15e-11** |
| max rise velocity | 0.2574 at t = 0.671 (+2.9 % vs 0.2502) | 0.2574 at t = 0.671 (+2.9 %) |
| `y_c(3)` | 1.1082 (−2.6 % vs 1.1376) | 1.1082 (−2.6 %) |
| steps to t = 3 | 1123 | 1123 |
| binding limit | capillary on 5 of 113 dt re-picks, WY CFL on 108 | identical |

**P1 removes 7.5 orders of the flux divergence (1.85e-03 → 5.15e-11) and moves nothing else** — the
iteration count, the step count, the dt-limit census and both published functionals are identical to
every printed digit. That confirms WO-P's attribution of the 1.85e-3 to the float `A·1 ≠ 0` defect,
and it also says the defect does **not** contaminate this benchmark's reported numbers. No default
was changed.

### Open questions for the coordinating session

1. **The operator fix above.** It is the blocker for every two-phase open-boundary case at
   ratio ≥ 10 with a low-side outlet, and it turns item 4 from harmful into correct. Own WO?
2. **Should `max_open_divergence()` stop mutating?** The non-mutating sibling exists; making it the
   default would move every recorded open-boundary divergence number in the repo (the channel/BFS
   verifies included) — in the right direction, but it is a re-baselining decision.
3. **`advect_vof(dt)`** is WO-Q's deliverable and was deliberately NOT added here to keep
   `flow_bindings.cpp` mergeable; this rung's kinematic gates call `IbmSolver::advectVof()` from
   C++ and the study script drives `step()`. WO-Q's `advect_vof` should route through the same
   `outflowCorrValid_` logic (it will, automatically).
4. **The half-shifted colour under `enable_vof_momentum` keeps the zero-gradient band at an inflow
   face** (the BC fill is guarded on the colour field's identity, because the same `exchange` hook
   carries the momentum fields). It matters only when the inflow colour differs from the colour
   already at the boundary; G3's pool inlet is profiled precisely so that it does not.
5. **The wisp guard on `wyIsMixed`.** Own WO — it changes a validated predicate. Until then, any
   VoF configuration that can empty its domain is a NaN risk on some backend.
6. **`maxAbsDiff`'s NaN blindness** is repo-wide in `tests/kokkos*`; the pattern
   `m = fmax(m, fabs(a-b))` cannot fail on a NaN field. Worth a sweep.
