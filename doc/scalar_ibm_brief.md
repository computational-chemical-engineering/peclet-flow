# Architect brief: scalar transport with immersed surfaces in peclet

*Author: the orchestrating session, 2026-10-02. Branch `scalar-ibm`, worktree
`suite/flow-scalar-ibm`. Every number below comes from a committed prototype in
`tests/study/scalar_ibm/`; the full record is `doc/scalar_ibm_log.md`. The literature digests are in
`doc/scalar_ibm_literature/` (L1 cut-cell/EB, L2 interface/conjugate/VoF, L3 particle-resolved and
benchmarks, R building blocks). Read this brief first; open those only for a specific question.*

## 1. The question

**Design the scalar-transport-with-immersed-surfaces subsystem for peclet.** It ships first in
`peclet.flow`. It must be built so that the same container-free kernels later serve `peclet.amr`
(block octree) and fluid–fluid VoF interfaces (species transfer with a partition coefficient).

It must cover:
- Neumann: homogeneous, and prescribed flux;
- Dirichlet;
- Robin (linear);
- conjugate transport: diffusion inside the solids, a partition coefficient K and a contact
  resistance R_c;
- advection by the solver's discretely divergence-free face flux.

**What must come back:** a design note `doc/scalar_ibm_design.md` that an Opus implementer can
execute without making decisions. It needs data structures, kernels, solver composition, API,
staged work orders and gates (§8).

## 2. Why this needs the architect

**Leverage.** The kernels must be reusable across three containers: structured blocks, octree
leaves and PLIC interface cells. A wrong choice of geometric record or storage layout now gets
copied into amr and VoF.

**Risk.** Three things are at stake:
- a new non-symmetric operator inside a code whose solvers are SPD-first;
- a small-cell treatment that touches conservation;
- a change to what κ (fluid volume fraction) means in the scalar storage term.

The register forbids changing existing numerics, so the new path must sit beside the current one,
bit-identical when off.

## 3. Current state of the code (with anchors)

**Scalar transport today** (`src/scalar_transport.hpp`, `src/flow_ibm_scalars.hpp`).

```
(1/dt)(c^{n+1} - c^n) + div(open u c) = div(open D grad c) + S
AC(i) = idt + D*wx*ox(i+sx) + D*wx*ox(i) + ... ;  AW(i) = -D*wx*ox(i)  ...   (scalar_transport.hpp:325-354)
```

- Unit storage: there is **no fluid fraction κ anywhere in flow**.
- Explicit conservative Koren/SOU/FOU advection on MAC face velocities × openness
  (`:356-394`).
- Red-black GS, 50 sweeps by default. `ScalarField` holds c, cOld, b, a 7-band
  AC/AW/AE/AS/AN/AB/AT, D, the scheme, `bc[6]`, `bcVal[6]`, `dmask`/`dval` (an identity-row
  per-cell Dirichlet mask), and the energy fields `kcell`/`rcp`/`gfmB`.
- `advanceScalars()` order (`flow_ibm_scalars.hpp:53-148`):
  1. build the diffusion operator;
  2. domain BC stencil;
  3. GFM plane-anchored Dirichlet (VoF phase change, `scalarMaskGfm`/`scalarMaskGfm2`,
     coefficient `k*w*openness/theta`);
  4. mask stencil;
  5. copy cOld;
  6. energy carry;
  7. ghost fill and advection RHS;
  8. GFM RHS;
  9. mask RHS;
  10. RB-GS sweeps with a ghost fill per colour.
- Python: `add_scalar(name, diffusivity, scheme, iters)`, `set_scalar_bc(name, face, type, value)`,
  `advance_scalars()` (also called inside `step()`). Energy / phase change:
  `set_phase_change_thermal(enabled, tname, Tsat, kg, kl, Rint)` with a Robin interfacial resistance.
- Tests: `tests/kokkos/test_scalar_transport.cpp`, `test_vof_phase_change.cpp`,
  `tests/kokkos_mpi/test_vof_phase_change_mpi.cpp`.

**Geometry available.**
- `sdf_` sits on the device over the whole extended block, with ghost width G = 2
  (`flow_ibm.hpp:75`, `flow_ibm_core.hpp:209-213`). It is filled by periodic wrap (single rank) or
  a GridHalo exchange, then `extendSdfDomainGhosts` for non-periodic faces.
  *Verify that both ghost layers are filled on every path.*
- Face openness ox/oy/oz per cell (the −face of cell i), from `ccFaceOpenMS`
  (`mac_cutcell.hpp:174`): marching squares over 5 trilinear SDF samples per face, floor 1e-3,
  aperture only, no centroid. It is the default (register: "Marching-squares (order-2) apertures
  are the shipped default in both flow and AMR").
- core `vof/plic.hpp` (container-free, device): `plicVolume(mx, my, mz, alpha)` (:221),
  `plicAlpha` (:303), `plicSlabVolume` (:339), `plicBoxVolume`.
- The discrete divergence `d(i) = Σ_a [o_a(i+s_a) u_a(i+s_a) − o_a(i) u_a(i)]`
  (`mac_pressure.hpp:300-310`) is zero after projection (staggered). Collocated has the projected
  face field `uf_/vf_/wf_`.

**Solver building blocks** (digest R).
- flow `CutcellMG::solveBiCGStab` (`mac_cutcell_mg.hpp:1801`): matrix-free BiCGStab with a
  V-cycle preconditioner, MPI staging through a g = 2 halo, used by the collocated "ghost" scheme.
- core `MomentumSolver::solveBiCGStab` (`core/include/peclet/core/solver/csr_bicgstab.hpp:122`):
  functor-based, with a pluggable preconditioner and injected `refresh`/`dotReduce`.
- `CutcellMG`: rediscretized coarse operators from averaged openness; RB-GS smoother.
- `VelocityMG`: a Helmholtz operator with a per-cell diagonal array.
- The overlay pattern `IbmOverlayT` (`cut_cell_ibm.hpp:142`): a device-compacted list of cut cells
  with SoA per-face factors; `ibmModifyStencil` rewrites the 7-point row in place.

**amr.** Per-leaf `FaceGeometry {rawArea, distance, alpha}` (`amr/include/peclet/amr/face_geom.hpp`).
Coarse–fine openness is evaluated at the finer neighbour's actual lower corner (register). There is
no scalar transport in amr yet.

## 4. Constraints and invariants

- **Conventions:** SDF < 0 in solid. x-fastest indexing. Python arrays `order='F'`. Physical units
  at the API (the solver computes on the unit lattice; `UnitScales` folds the metric in). No
  cell-unit API surface. Anisotropic cells are supported (axis weights w_a).
- **Bit-identity:** every existing scalar, energy and phase-change path stays bit-identical when
  the new option is off. `tests/regression/state_hash.py` is the byte gate.
- **Settings:** no environment variable may change a result; every switch is a per-solver setter.
  Strings, not integer codes. Two API tiers: public surface vs `s.diagnostics`. Naming canon
  `../docs/NAMING.md`.
- **Device-first:** everything on device. Host code only as oracle/tests. MPI-distributable on the
  core ORB with GridHalo exchange (G = 2).
- **Kokkos:** `.hpp` only, iteration order through `MDRange3` from `policy.hpp` (ctest
  `iteration_order`). Double operator storage is the default; float casts are policed (ctest
  `no_float_operator_casts`).
- **Shared code:** container-free kernels belong in `core` (the precedent is
  `core/scheme/cut_cell_closure.hpp` and `core/vof/`), so amr can include them.
- **Performance yardstick:** SOTA massively parallel (user directive), setup included.

## 5. Already decided (not open)

These come from Frank's directives and from measured evidence.

1. **Scope:** Dirichlet, Neumann, Robin and conjugate in one framework. AMR and VoF reuse in the
   design. Periodic domains first.
2. **κ in the storage and source terms** (cut-cell FV). Without it, aperture-FV Neumann is first
   order: the Neumann eigenvalue error is 11 % at 32 cells per diameter, and Taylor–Aris 1/48 is
   24 % off. Under advection, unit storage loses 2–12 % of the physical mass Σκc.
3. **Every cell with κ > 0 is an unknown.** Merging slivers into fluid-centred neighbours makes
   Neumann first order (log, round 2).
4. **Faces carry the plain aperture two-point flux.** Face-centroid interpolation
   (Johansen–Colella) changed nothing measurable (log, round 3), so it is not needed for solution
   accuracy.
5. **Wall flux: "probe-flux".** Each cut cell carries one facet (area vector, centroid,
   normal). A probe point sits at s = c·h along the normal from the facet centroid, and
   u(probe) is a (bi/tri)linear interpolant over κ > 0 cells.
   - Gradient: du/ds ≈ (u_p − u_Γ)/s.
   - The BC is eliminated per facet:
     - Robin D du/ds = k (u_Γ − g) gives u_Γ in closed form;
     - Dirichlet is k → ∞;
     - Neumann is k = 0 (or a prescribed flux);
     - conjugate is a 2×2 system per facet over a fluid probe and a solid probe (Crockett,
       Colella & Graves 2011 structure, L2), which covers K and R_c.
   - This is AMReX `MLEBABecLap`'s Dirichlet stencil (L1 §3), extended to Robin and conjugate.
   - **Measured (2-D disc, exact eigenvalues):**
     - order 2.0 at Bi = 0, 0.1, 1, 10, 100, ∞;
     - 3.5e-5 relative error at 128 cells per diameter for Dirichlet (Gibou's symmetric ghost:
       5.8e-5);
     - conjugate composite disc order ~2 for D_s/D_f = 0.1–100, K = 3, h_c = 5, capacity ratio 2;
     - ≤ 2e-4 at 128 cells per diameter.
6. **Solver: Krylov (BiCGStab or GMRES) preconditioned by MG on an SPD "lumped-probe" surrogate.**
   - The surrogate is aperture FV + κ/dt + the diagonal wall term A_w/(h²(c·h + D/k)), i.e. the
     probe weights summed onto the diagonal.
   - Measured with an exact surrogate solve: 8–9 BiCGStab iterations for Dirichlet, 2–5 for
     Robin; independent of ND from 32 to 128; steady and dt·D/h² = 1.
   - A centroid-distance surrogate failed for Dirichlet: 60–90 iterations, ρ = 0.995.
   - Plain GS directly on the probe operator also converges (it does not diverge despite the
     positive off-diagonals).
7. **Small cells under advection need a treatment.** Explicit κ-storage advection blows up at the
   full-cell CFL (min κ 4e-5 to 2e-4).
8. **Rejected (measured, do not re-propose):**

| rejected scheme | measured order / failure |
|---|---|
| the per-cell Dirichlet mask (flow today) | 0.7–1, with 3–16 % errors |
| unit storage | 1 |
| Gibou linear ghost alone (Dirichlet only, other unknown set) | Neumann limit breaks |
| Papac/Gibou symmetric Robin | 1.4–1.7 at Bi ≥ 1; Dirichlet limit is a mask |
| the "aperture + link" hybrid | 1 in the Neumann limit |
| centroid two-point / series-resistance wall flux (= Liu–Fedkiw–Kang GFM) | 1 |
| quadratic normal probe | erratic |
| AMReX's short κ-dependent probe (≈0.3 h) with renormalized fallback | erratic, 0.4–1.2 |

## 6. Genuinely open — decide these

1. **Geometric record per cut cell.** It must be source-agnostic (SDF solid, PLIC interface, amr
   leaf). What is stored, and where is it computed?
   - **κ:** a plane-cube `plicVolume` from (normal, SDF at the centre), subsampling, or a
     divergence-theorem formula consistent with the marching-squares apertures?
   - **Facet area vector:** −Σ a_f A_f e_f, exactly consistent with the apertures, which the
     closure problem and conservation rely on.
   - **Facet centroid:** projection of the cell centre onto the plane, or the PLIC polygon
     centroid?
   - Probe stencil indices and weights, and storage precision.
2. **Probe distance in 3-D, and the fallback.** 2-D data: 0.7 h clean with no fallback; 1.0 h is
   4× worse; short probes need fallbacks and degrade. In 3-D, c ≥ √3/2·h guarantees that every
   trilinear neighbour of a planar wall is at least cut.
   - Choose c.
   - Specify the fallback ladder for thin gaps, particle contacts and concave corners, where a
     probe can enter another body or a covered cell. L3: contact regions dominate packed-bed Nu
     error (Claassen 2024; Das 2017 needs 80 cells per diameter for 1 % at Re = 50).
3. **Conjugate storage.**
   - Two fields on one grid (a fluid DOF where κ_f > 0, a solid DOF where κ_s > 0), or one field
     with a second DOF array only for cut cells?
   - Per-body solid properties (D_s, ρc_p or capacity, K, R_c) through a material id from the
     scene?
   - Use the ψ = c/K substitution (L2) so the surrogate stays SPD?
4. **Solver composition.**
   - Which Krylov (core functor BiCGStab vs flow's pressure BiCGStab vs FGMRES)?
   - The MG for the surrogate: reuse CutcellMG or VelocityMG with a diagonal entry point, or a new
     light hierarchy? Rediscretized vs Galerkin coarse operators. The register records that
     rediscretized coarse momentum operators failed in amr and that Galerkin is the robust
     velocity-MG default, so weigh that.
   - For conjugate, the surrogate couples fluid and solid DOFs of the same cell (2×2 block):
     block smoother?
   - The MPI dot products / halo injection.
   - How it replaces the fixed 50 RB-GS sweeps (a residual-based stop) without touching the
     legacy path.
5. **Advection small cells.**
   - The prototype measured a May–Berger-style explicit–implicit split: implicit FOU on faces
     touching κ < ½, explicit elsewhere. It is exactly conservative, positive and stable at the
     full-cell CFL, at no extra cost, since the implicit solve already exists.
   - The literature (L1) favours weighted state redistribution (Berger–Giuliani 2021/2022): 2nd
     order and AMReX-proven, but it needs a ~3-cell halo or a second exchange.
   - Choose, considering moving geometry later (DEM particles; VoF interfaces with fresh and dead
     cells). How does Koren TVD on full cells combine with either?
6. **Container-free split.**
   - Which kernels go to `core` (facet geometry, probe interpolation weights, per-facet BC
     elimination, small-cell split) and which stay in flow (driver, storage, solver wiring)?
   - The amr hooks: coarse–fine probes and per-leaf facets.
   - The VoF hooks: a PLIC facet as the record source, and two-phase species with K.
7. **API (physical units).**
   - Per scalar, per body or material: wall condition type and value (Dirichlet value, flux, Robin
     k and g).
   - Conjugate material properties.
   - Opt-in switch (e.g. `add_scalar(..., cut_cell=True)` or a separate entry point) and
     interaction with the existing `set_scalar_bc` domain faces and `dmask`.
   - Diagnostics: per-facet wall flux (local Nu/Sh), per-body integrated flux, a conservation
     budget.
   - Source terms: needed for the Brenner closure problem, where the macroscopic gradient enters
     as a source.
8. **Relation to the existing VoF phase-change energy path** (GFM plane-anchored Dirichlet, Robin
   IHTR): leave it alone now, but say how it would migrate onto this framework later.

## 7. Verification (make concrete in the design)

The 2-D prototypes are the oracle for structure. In 3-D, the gates (L3 §4; exact unless noted) are:

| gate | case | reference / check |
|---|---|---|
| G-geom | κ, apertures, area vector | Σ_f a_f A_f e_f + A_w n = 0 to round-off; κ vs exact sphere volume |
| G1 | sphere conduction, Dirichlet | Nu = 2, in a large box or with a far-field correction |
| G2 | concentric shells | Neumann / flux |
| G3 | Robin sphere, Da sweep | Sh = 2 Da/(1 + Da); transient eigenvalues λ cot λ = 1 − Bi |
| G4 | Maxwell conjugate sphere in a uniform gradient, κ ratio 1e-2…1e3 | interior gradient 3G/(κ + 2) |
| G5 | periodic simple-cubic array effective conductivity | Sangani–Acrivos: insulating = pure Neumann; conducting / finite contrast = conjugate |
| G6 | transient composite sphere (spherical Bessel) | with K, R_c and a capacity ratio |
| G7 | pipe as an SDF in a periodic box | Graetz Nu_T 3.6568 (Frank's eigenproblem form, `peclet-papers/work/A4/nusselt_periodic_Peters.pdf`); decay j₀₁² = 5.78319 |
| G8 | pipe | Taylor–Aris 1/48 via the steady Brenner closure (an inhomogeneous-Neumann source problem) |
| G9 | 3-D annulus/torus rotating-flow advection | conservation 1e-14, positivity, stability at the full-cell CFL |
| G10 | MPI | np = 1, 2, 4 parity (bit-exact or to Krylov tolerance, stated) |
| G11 | backends | GPU vs OpenMP |
| G12 | legacy paths | state_hash bit-identical |

Each gate is a ctest with a convergence-order assertion, at three resolutions where meaningful.

## 8. Deliverable

Write `doc/scalar_ibm_design.md` with these sections:

1. formulation (continuous + discrete, all four BC types + conjugate, with signs);
2. geometric record and its construction (3-D formulas);
3. probe stencil, fallback ladder, precision;
4. operator application (the overlay matvec) and the SPD surrogate;
5. solver composition, MPI and stopping criterion;
6. advection and small cells;
7. core/flow split and the amr/VoF hooks;
8. Python API (names checked against NAMING.md);
9. diagnostics;
10. **work orders** in implementation order, each with files, functions, the gate it must pass,
    and what it must not touch;
11. **gates** with numbers and tolerances;
12. register entries to add (each departure from a textbook alternative, with the rejected
    alternative);
13. risks and what would make you revisit the design.

Also return a summary of no more than 30 lines naming every decision you took.

## 9. Out of scope

- Implementing anything. No production code; prototypes are allowed if they settle a design
  question you cannot settle otherwise, but say so.
- Moving geometry and VoF interfaces as *implementations*: design the hooks only.
- Nonlinear surface reactions. amr implementation.
- The Lagrangian tracer route (a separate later package; see `tests/study/scalar_ibm/tracers.py`
  for the estimator findings).
- Higher-order wall gradients for local Nusselt maps: mention them as an optional post-processor
  (a quadratic normal stencil, Schwartz 2006) only.
