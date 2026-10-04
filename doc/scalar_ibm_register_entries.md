# Proposed register entries: cut-cell scalar transport (WO-10, 2026-10-04)

This is PROPOSED text for the orchestrator to place in the umbrella when branch `scalar-ibm` lands:
`suite/docs/decisions/flow.md` (and one entry in `suite/docs/decisions/core.md`), plus the one-line
index in `suite/docs/DECISIONS.md`. The umbrella is a shared checkout on `main`, so it is not edited
from this branch.

This file is not itself part of the register. Delete it once the entries are placed, and regenerate
the index and its counts with `docs/decisions/build_index.py`.

- **Source of truth.** `doc/scalar_ibm_design.md` (architect note 95e4a55, Amendments A1 03435df,
  A2 a61f88e and A3 2ae5ff9) and `doc/scalar_ibm_log.md` (every number and every orchestrator ruling
  D-WO*).
- **Quotes** are verbatim from those two files.
- **The scalar-units entry** (Frank's Q6, "fix it") is already proposed in
  [`scalar_units_register_entry.md`](scalar_units_register_entry.md). It belongs to the same landing
  and is included here **by reference**, not repeated.

Status line used below: `settled (design + orchestrator rulings, branch scalar-ibm; in force when it
lands on main)`. The API spelling `add_scalar(..., cutcell=True)` (§13 Q5) and whether cut-cell
becomes the default (Q7) are still pending Frank, so neither is an entry.

## For `docs/decisions/flow.md` (append)

```markdown
---

### Cut-cell scalar transport stores κ (the fluid fraction) in storage and sources
- area: flow
- source: flow doc/scalar_ibm_design.md §1.3–§1.4, D1, §12 entry 1; doc/scalar_ibm_log.md round 1, round 6
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    Cut-cell FV with κ (fluid fraction) in storage and sources; every κ > 0 cell with an open face
    is an unknown
    Neumann, aperture FV: 2nd order with κ in storage/source; 1st order without it.
    unit storage (flow today) | stable, but the physical mass Σκc drifts −12 %, −5.5 %, −2.1 %
- rejected: unit storage, i.e. the legacy scalar's (first order; the Neumann eigenvalue 11 % off at 32 cells per diameter; 2–12 % mass loss under advection)
- why: κ in storage is what makes the Neumann limit second order and the advective mass exact (1e-16 in the 2-D annulus); the small cells it creates are handled by the dynamic implicit split, not by giving up κ

---

### Every κ > 0 scalar cell with an open face is its own unknown; slivers are never merged or linked
- area: flow
- source: flow doc/scalar_ibm_design.md §1.3, §2.4, §12 entry 2; doc/scalar_ibm_log.md round 2
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    Merging slivers costs the Neumann limit; every κ > 0 cell must be its own unknown.
- rejected: cell merging or linking of slivers (the H-k / Hf-1 link schemes: Neumann limit −4.2e-3 at order 1.0, against −6.2e-5 at order 2.0 for κ-FV with every cell its own unknown)
- why: one unknown per κ > 0 cell keeps every Robin number second order; the conditioning cost of tiny κ is carried by the probe closure (which never uses a cut cell's own tiny distance) and by the small-cell split

---

### The scalar unknown sets follow the snapped apertures; κ = 0 ⇒ all apertures 0 by the strict-sign rule
- area: flow
- source: flow doc/scalar_ibm_design.md §2.2 ("Fluid" means φ > 0 strictly), §2.4, §2.6; ruling D-WO7-1 (doc/scalar_ibm_log.md, WO-7 rulings; commit bb039d1)
- decided: 2026-10-02 (fluid), 2026-10-03 (solid, D-WO7-1)
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    **Fluid unknown:** κ > 1e-14 **and** Σ_f a^snap_f > 0.
    **"Fluid" means φ > 0 strictly.** A vertex with φ = 0 is solid. This makes a wall lying exactly on
    a face close that face, so κ = 0 never coexists with an open face.
    D-WO7-1 (Q-L): option (b). A solid unknown exists iff κ_s > 0 ∧ (Σ a_s > 0 ∨ the cell has a
    conjugate facet) — the solid mirror of the fluid's sealed rule.
- rejected: unknown sets from a κ that is not the apertures' own volume (plane-cube `plicVolume`: κ = 0 next to an open face, κ > 0 behind all-closed faces, i.e. division by κ and orphan rows; the 4³ `cs_` subsampling: slivers read κ = 0 with open faces); the design's first solid rule "κ_s > 0 and conjugate" (it admitted 95–192 uncoupled solid unknowns per rung whose transient rows m(ψ − ψⁿ) = 0 froze them, so G6's total-mass error stalled at 0.44); the strict mirror "κ_s > 0 ∧ Σ a_s > 0" (it would drop solid cells coupled only through a conjugate facet)
- why: a cell is an unknown exactly when the operator couples it, so there are no orphan or frozen rows; with D-WO7-1, G6 is second order on the total mass (1.91–2.00) and the uncoupled count is 0 on every gate run

---

### Sealed scalar cells (fluid or solid) are not unknowns; they are gated on their VOLUME, not their count
- area: flow
- source: flow doc/scalar_ibm_design.md §2.4, §9; rulings D-WO2-1 and D-WO7-1 (doc/scalar_ibm_log.md WO-2, WO-7 rulings)
- decided: 2026-10-03
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    D-WO2-1 — sealed cells. Gate (e)'s "0 sealed" is restated as: sealed volume Σκ·V over sealed cells
    ≤ 1e-6 of the fluid volume (the §9 threshold; measured ≤ 1.9e-9). The sealed count stays in the
    census. The sealed definition (§2.4) and the snapping rule (§2.3) are NOT changed.
    The excluded volume is in the census (`sealed_solid_volume`, with `solid_volume` = the conjugate
    solid κ_s V, physical) and gated ≤ 1e-6 of the solid volume (as D-WO2-1).
- rejected: a "0 sealed" count gate (impossible: the corner slivers whose faces all snap to 0 at the 1e-3 floor grow like (R/h)², 0–11 / 38–46 / 154–159 at R/h 8 / 16 / 32); making sealed cells unknowns (empty rows); changing the 1e-3 snap floor (it is the pressure's measured conditioning constant, and linear exactness needs both-ends snapping)
- why: the sealed volume is ≤ 1.9e-9 of the fluid volume and ≤ 1.6e-8 of the solid volume on every gate geometry; a fluid one above 1e-6 prints a resolution warning

---

### Scalar faces carry the plain aperture two-point flux
- area: flow
- source: flow doc/scalar_ibm_design.md D2, §1.4, §12 entry 3; doc/scalar_ibm_log.md round 3
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    The face-centroid flux interpolation makes no difference (JC vs JC-nofc agree to 3 digits), so it
    is not needed.
- rejected: Johansen–Colella face-centroid interpolation (no measurable change; costs symmetry and compactness)
- why: the 7-point part stays symmetric and compact, and the surrogate built on it stays SPD

---

### Immersed scalar walls use probe-flux: one probe on the facet normal, the BC eliminated per facet
- area: flow
- source: flow doc/scalar_ibm_design.md D3, §1.4, §3, §12 entry 4; doc/scalar_ibm_log.md rounds 1–3; gates G1–G3, G7 (log WO-3/WO-4)
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    **Conclusion.** The decisive ingredient is a wall gradient over a probe distance ≥ ~0.7 h,
    interpolated from neighbours. It never uses the cut cell's own value over its (possibly tiny)
    distance. One unknown set and one formula cover Neumann → Robin → Dirichlet at 2nd order, and the
    3-D stencil is compact: 27-point plus a reach of 2.
- rejected: flow's per-cell Dirichlet mask (order 0.7–1); the Gibou linear ghost alone; Papac/Gibou symmetric Robin (order 1.4–1.7, its Dirichlet limit a mask); the aperture + link hybrid (no symmetric compact scheme is second order at all Bi); the centroid two-point / series-resistance wall flux (order 1; its own entry); the quadratic normal probe (smallest error but erratic: signs and orders jump)
- why: second order at every Bi from 0 to ∞ in 2-D and in 3-D: G1 Nu 1.93/1.97, G3a Da 0.1–100 1.93–2.00, G7 Graetz 2.01–2.02, G2 1.89/1.94

---

### The scalar probe distance is s = 1.1·½Σ|n_a|h_a (0.55–0.95 h), the shortest that never needs the fallback
- area: flow
- source: flow doc/scalar_ibm_design.md D6, §3.1–§3.3, §12 entry 4; doc/scalar_ibm_log.md "probe-distance sweep" (2026-10-02); `jc.py probeN1.1`
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    The fallback (renormalization = local constant extrapolation) is what degrades short probes.
    Choose the probe so that the stencil never needs it.
    | AMReX κ rule | 762 | 7.3e-5 | 9.3e-5 | 0.7–1.2, erratic |
    | 0.3 h | 762 | 1.1e-4 | 1.5e-4 | 0.4–0.9, erratic |
    | 0.7 h | 0 | 8.4e-6 | 3.5e-5 | 2.0, reach 2 |
- rejected: AMReX's κ-dependent short probe dx_eb = max(0.3, (κ² − ¼)/(2κ)) h with renormalized fallback (762 fallbacks, order 0.7–1.2, erratic); a 0.3 h probe (order 0.4–0.9, erratic); a constant √3/2·h probe (the worst case of S(n), needlessly long: error grows with the distance, 0.7 h 3.5e-5, 1.0 h 1.5e-4, 1.5 h 4.5e-4)
- why: s ≥ S(n) guarantees every trilinear stencil cell of a planar wall is at least cut, so R0 holds on every smooth geometry (100 % on the isotropic sphere and pipe sets) and the reach stays within the G = 2 halo; the normal-dependent distance matches the constant 0.7 h at order 2.0 (probeN1.1)

---

### The immersed scalar wall flux is never a two-point / series-resistance flux over the cut cell's centroid distance
- area: flow
- source: flow doc/scalar_ibm_design.md §12 entry 4; doc/scalar_ibm_log.md round 1, round 5, round 7
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    Dirichlet: Gibou linear ghost 2nd order; mask 0.7–1; cut-cell FV with a centroid wall flux 1st.
    Robin, series-resistance FV: 1st order.
    | GFM series (= Liu–Fedkiw–Kang) | 1.2e-3 / 2.4e-2 | 7.5e-5 / 5.1e-3 | 4.5e-5 / 2.6e-3 | 2.0e-5 / 2.6e-4 |
- rejected: the centroid two-point wall flux and the series-resistance (GFM, Liu–Fedkiw–Kang) wall and interface flux (first order; at the Maxwell disc, d/h = 128, k₂ = 0.01, GFM L∞ 2.4e-2 against probe-flux 1.3e-4); a surrogate built on the centroid distance (ρ(I − S⁻¹A) = 0.995, BiCGStab 59–87 iterations against 8–9 with the probe distance)
- why: a distance that shrinks with the cut makes the flux depend on the cell's own (possibly tiny) value; the probe distance keeps both the accuracy and the conditioning

---

### Conjugate scalar transport is two fields on one grid in ψ = c/K, with the series-resistance 2×2 elimination at the fluid and solid probes (P2F)
- area: flow
- source: flow doc/scalar_ibm_design.md D8, §1.1, §1.3–§1.4, §12 entry 5; doc/scalar_ibm_log.md round 4, round 7 (`conj_peters.py`), WO-7 (G4, G6)
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    The one-field directional scheme imposes the jump of k ∂T/∂x along grid lines, not along the
    interface normal. On tilted interfaces its L∞ stays first order. P2F, with the true normal,
    probes and per-facet elimination, is 2–50× better in L∞ at every contrast and 3–10× better in
    L1, except at k₂ = 100, where DIR (tex reading) equals it.
    **Decision.** P2F remains the conjugate method. Peters' directional scheme is recorded as a measured
    alternative.
- rejected: Peters' directional one-field scheme (min-curvature T_ξ; Maxwell disc at d/h = 128, L1/L∞: DIRc 6.7e-5/6.0e-3, 5.7e-6/4.0e-4, 7.9e-6/4.3e-4, 1.9e-4/8.4e-3 against P2F 9.6e-6/1.3e-4, 4.1e-6/6.4e-5, 5.5e-6/9.2e-5, 2.0e-5/2.3e-4 at k₂ = 0.01 / 0.5 / 2 / 100; L∞ order ~1; no contact-resistance path; first-order Neumann limit); one field plus a side array; a mixture one-field surrogate (arithmetic coarsening across the jump); Das-style lagged partitioned coupling
- why: 3-D G4 orders 1.84–1.94 at Λ_s/Λ_f = 1e-2 … 1e3, exact (≤ 3e-11) at ratio 1, K exact, and G6 1.91–2.00 for all six (Λ_s, C_sK, K, R_c) cases; ψ = c/K keeps the coupling symmetric and the surrogate SPD; the two fields are full because the solid is 50–60 % of a packed bed

---

### Scalar cut-cell geometry is the fan-tetrahedron PL model on the marching-squares samples; the scalar apertures are ungated and snapped at both ends
- area: flow
- source: flow doc/scalar_ibm_design.md D4, D5, §2.2–§2.3, §2.6, §12 entry 6; log WO-1, WO-2
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    The fan-tet κ is the divergence-theorem volume of the same PL surface that defines the apertures.
    It is exactly consistent, bounded in [0, 1], and κ > 0 whenever any aperture is.
    So the scalar keeps its own fields `sax/say/saz`, and `ox_` is never touched.
- rejected: plane-cube `plicVolume(n, φ_c)` for SDF sources (inconsistent with the apertures on curved walls, edges and gaps); 4³ subsampling (`cs_`, quantized to 1/64); the cell-centre projection as facet centroid (O(h) off the facet in corner-cut cells, hence first order); reusing the gated pressure openness `ox_` (up to ~0.5 wrong at every centre-solid face, an O(1) flux error on a surface set)
- why: κ, apertures, facet area vector and centroid come from one polyhedron, so the PL divergence identity holds to round-off (5.6e-16) and a_f + a_s = 1 after snapping gives linear exactness; volume and area converge at order 2.00 (G-geom e, g)

---

### ScalarMG coarsens the surrogate with rediscretized coarse FACES and variational coarse WALL terms: the plain average of the level-0 terms at the FINE probe distance (Amendment A1)
- area: flow
- source: flow doc/scalar_ibm_design.md §5.2 Amendment A1, §5.3, §12 entry 7; doc/scalar_ibm_log.md WO-4, WO-4 completion; commits c1e312e (RAP evidence), 03435df, 35e183d; `tests/study/scalar_ibm/mg_coarse_wall.py`
- decided: 2026-10-03
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    Every lumped wall or interface term on a coarse level is the **plain average of its level-0
    value**, evaluated at the **fine** probe distance
    Rescaling s by the level size divides that energy by ~2^L. The coarse problem becomes softer than
    any prolongated function, by 2^L, exactly where the coarse cell cannot resolve the near-wall dip.
    The correction then overshoots by that factor.
- rejected: wall terms at the level's own probe distance s_L, the original design (stand-alone contraction 0.79 at the finest G1 and 3.1, divergent, on a periodic box with a Dirichlet sphere; 2-D 1.5–2.7; its rationale "keep the wall-to-face ratio level-independent" is refuted); Galerkin RAP (27-point coarse operators and an 8-colour sweep, no better overall: 0.28–0.36 in 3-D, 0.33–0.42 without a solid); extending CutcellMG; VelocityMG's staircase coarse operator (Dirichlet-like coarse walls, wrong for Neumann and singular closures)
- why: after A1 every gate geometry contracts at 0.17–0.32 (box, Neumann, no solid) and 0.50–0.52 (periodic isolated sink), G1 iterations 12/14/15 → 10/10/11, MPI parity at the default rtol (np 4) 7.3e-8 → 1.5e-13; this CONFIRMS for the wall term the amr entry "Rediscretized coarse momentum operators fail" — the face part stays rediscretized because the surrogate is in the rediscretizable family

---

### The scalar linear solve is BiCGStab + one ScalarMG V-cycle on the lumped-probe surrogate, with a max-norm relative stop (default 1e-10)
- area: flow
- source: flow doc/scalar_ibm_design.md D9, D11, §4.3, §5.1, §12 entry 8; doc/scalar_ibm_log.md round 5
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    **Conclusion.** Use BiCGStab (or GMRES/FGMRES), preconditioned by an MG V-cycle on the SPD
    lumped-probe surrogate.
    The cycle is a fixed linear operator, so plain BiCGStab is correct (no FGMRES).
- rejected: fixed RB-GS sweeps (the legacy 50-sweep path); (F)GMRES (restart storage and k reductions per iteration; FGMRES unnecessary for a fixed cycle); the centroid-distance surrogate (ρ = 0.995)
- why: 8–9 BiCGStab iterations for Dirichlet and 2–5 for Robin with an exact surrogate solve, mesh-independent; in 3-D G1 10/10/11 and G-iter met on every row

---

### Scalar small cells under advection: a dynamic implicit-FOU split relative to the bulk Courant number
- area: flow
- source: flow doc/scalar_ibm_design.md D12, §6.3–§6.6, §12 entry 9; doc/scalar_ibm_log.md round 6, WO-5 (G9)
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    small_i  ⟺  κ_i < 1  ∧  Δt·Out_i > max(½, C_bulk) · κ_i V
    A cut cell stays explicit exactly when its own Courant number is no worse than the bulk's.
- rejected (for now): weighted state redistribution (needs a ~3-cell halo; its gain sits where no-slip velocities make the flux O(h); recorded for VoF and slip walls, §13 Q8); explicit κ storage (blows up at min κ 4e-5); fully implicit FOU (more diffusive, L1 1.63 against 1.50)
- why: stable at full-cell CFL, mass to 1e-16, positive; in 3-D (G9) the cut band carries ≤ 4.5 % of the FOU L1 error at R_o/h 64, so Q8 is not triggered

---

### Scalar advection uses the projection's own constrained face flux; the scalar apertures weight diffusion only
- area: flow
- source: flow doc/scalar_ibm_design.md §6.1, §12 entry 10; doc/scalar_ibm_log.md WO-5 (G9b)
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    **One predicate** `Solver::scalarFaceFlux(a)` returns it. It must read the same openness as the
    divergence kernel of the projection in force.
- rejected: the scalar apertures for advection (not the projection's constraint, so Σ F ≠ 0 per cell and constants are not preserved)
- why: G9b: a constant stays exactly 1 (max|c − 1| = 0) under staggered, 'gauge-exact', 'plain' and 'embed'; wrong openness gives ~1e-2

---

### Cut-cell scalars refuse the ghost projection (collocated 'ghost', staggered set_ghost_projection) and, for now, collocated open faces
- area: flow
- source: rulings D-WO5-4 (§13 Q13) and D-WO5b-1 (doc/scalar_ibm_log.md WO-5, WO-5b); `Solver::scalarCutRefusals` (src/flow_ibm_scalars_cutcell.hpp)
- decided: 2026-10-03
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main); the open-face refusal is reversible once the collocated `uf_` outlet issue is settled
- quote: |
    **D-WO5-4.** The collocated 'ghost' scheme is refused for cut-cell scalars (Q13): its face field
    is not divergence-free under any openness (2–4.5e-2). Collocated users select gauge-exact, plain
    or embed.
    **D-WO5b-1 (Q-I).** Collocated open faces stay REFUSED for cut-cell scalars (option c).
- rejected: advecting with the ghost projection's face field under its binary openness `oxb_` (divergence 4.3e-2 of max|F|) or the geometric `ox_` (2.0e-2): a constant drifts 4.1e-2 / 0.80 / 1.28 at bulk Courant 0.016 / 0.47 / 0.86; capturing the collocated high-side boundary flux after `project()` (the plane holds the opposite face's values: a constant drops 0.48 in the outlet column in one step)
- why: the ghost constraint is the binary divergence PLUS the closure delta (`gpDivergDelta`), so no openness-weighted flux of it is divergence-free; this is design revisit item (iii) for 'ghost', the AUTO default of SolverColocated, so a cut-cell scalar there needs an explicit `set_collocated_scheme('gauge-exact' | 'plain' | 'embed')`

---

### Scalar conservation is exact in the discretization; the solver residual is reported as a defect, never fixed up
- area: flow
- source: flow doc/scalar_ibm_design.md D13, §6.4, §9, §12 entry 11; gates G3b, G9, G9c, G-adv (v) (log WO-3 … WO-5c)
- decided: 2026-10-02
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    Conservation is exact for the discretization; the budget reports the linear-solver defect; **no
    mass fix-up**
- rejected: a post-solve uniform mass correction
- why: the budget identity closes to round-off (≤ 2.8e-14 of d_mass per step, ≤ 1e-11 of the gross budget for steady solves, D-WO5c-1); a large defect means a loose tolerance and must stay visible

---

### Steady scalar advection is preconditioned by a V-cycle on the ADVECTIVE surrogate (Amendment A2)
- area: flow
- source: flow doc/scalar_ibm_design.md §6.7 Amendment A2, §12 entry 13; doc/scalar_ibm_log.md WO-5 (Q-H), WO-5c; `tests/study/scalar_ibm/mg_advection.py`; commits a61f88e, e1127ed, 7a3e5fb
- decided: 2026-10-03
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    In **steady mode with advection** the V-cycle runs on the **advective surrogate**
    S_adv = S + the implicit-FOU couplings, on every level.
    **Coarse advection: the summed positive parts of the sub-face fluxes.** This is the
    piecewise-constant Galerkin value.
    **Transient mode, and steady mode at rest, are unchanged and stay bitwise.**
- rejected: the symmetric lumped surrogate for steady advection (preconditioned spread β ≈ Pe_h N/2π; 3-D 13 → 40 → 89 iterations at Pe_h 0 / 0.1 / 0.3 and no convergence at 1); advection at level 0 only (2-D 35 / 119 / 222 at Pe_h 0.1 / 1 / 10 — the failing modes are the low ones); a symmetric surrogate with the upwind diffusion sym(A) (β → N/π); pseudo-transient continuation (the same spread on every step, or O(N²) steps); GraphAMG on the assembled FOU operator (SPD design with a Chebyshev smoother, no MPI path); defect correction around the same preconditioner; VelocityMG's restricted-velocity coarse upwinding (not divergence-free on the coarse grid); downstream or line smoothers (no order on a periodic torus; not parallel on a GPU block or across MPI blocks); the net (signed-sum) coarse flux (measured identical, but drops the coarse exchange diffusion); F- or W-cycles as the default (they halve the count but change the cycle; recorded escalation §13 Q18)
- why: 3-D G-adv converges everywhere to Pe_h 10: (a) 8/13/23 and 10/17/33 iterations, closures 5/8/12 and 7/10/17, growth ≤ 1.43× per doubling, C4 contraction ≤ 0.764, the solution unchanged (9.8e-12); the transient and at-rest paths are bitwise unchanged (170/170 arrays)

---

### The transient ScalarMG level rule switches to the full table at κ_A = 1 + 4 dt'D'Σw_a ≥ 25 (Amendment A3, ruling D-WO9-3)
- area: flow
- source: flow doc/scalar_ibm_design.md §5.2 Amendment A3 + its D-WO9-3 paragraph; doc/scalar_ibm_log.md D-WO9-2 and the WO-10 entry; `ScalarMG::kFullTableKappa` (src/scalar_mg.hpp); commits bdc957c (13, named), b9c8ca0 (25)
- decided: 2026-10-04
- status: settled (design + orchestrator rulings, branch scalar-ibm; in force when it lands on main)
- quote: |
    *D-WO9-3 (2026-10-04) — the switch moves to κ_A = 25.* The orchestrator relaxed G-iter's bound on
    the **cold first step** to ≤ 12; warm steps stay ≤ 10.
    κ_A = 25 is the largest measured value at which level 0 is measurably cheaper (table above); at
    49 the two tie.
- rejected: κ_A = 13, the design's value, kept by D-WO9-2 only because level 0 there took 11 iterations on one cold first step against G-iter's ≤ 10; κ_A ≈ 49 (cost alone: a tie, 15 against 10 iterations, not measurably cheaper)
- why: per advance on the 128³ bed level 0 is 1.05–1.24 vs 1.93 s at κ_A 7, 1.23 vs 1.69 s at 13, 1.35–1.55 vs 1.86–2.04 s at 25; host advance ÷ projection at G-perf's condition 1.50 → 1.05; bitwise unchanged outside 13 ≤ κ_A < 25
```

## For `docs/decisions/core.md` (append)

```markdown
---

### Cut-cell geometry and probe kernels are container-free core headers
- area: core
- source: flow doc/scalar_ibm_design.md D14, §7.1, §12 entry 12; core commits a031c6f, b1fcb6a (branch scalar-ibm)
- decided: 2026-10-02
- status: settled (design, core branch scalar-ibm; in force when it lands on main)
- quote: |
    Container-free kernels in **core** (`scheme/cut_cell_geometry.hpp`, `scheme/probe_flux.hpp`);
    driver, storage, operators, ScalarMG and Krylov in flow
    Lattice offsets are returned, not DOF indices. flow converts them to linear block offsets; amr maps
    them to leaf DOFs.
- rejected: flow-private copies (amr and VoF would fork them); a core Krylov (core's MomentumSolver binds the operator to a CSR type and the vectors to a [0, n)+ghost-tail layout, and flow's blocks interleave ghosts)
- why: amr (per-leaf SDF samples) and the VoF species hook (plane source) reuse the same κ / aperture / facet / probe kernels, so the geometry stays consistent by construction across codes
```

## For `docs/DECISIONS.md` (one line each)

In the flow section ("In force"):

```markdown
- **Cut-cell scalar transport stores κ (the fluid fraction) in storage and sources**. **Rejected:** unit storage (first order; 2–12 % mass loss under advection)  <sub>flow</sub>
- **Every κ > 0 scalar cell with an open face is its own unknown; slivers are never merged or linked**. **Rejected:** merging/linking slivers (Neumann limit first order)  <sub>flow</sub>
- **The scalar unknown sets follow the snapped apertures; κ = 0 ⇒ all apertures 0 by the strict-sign rule (solid: κ_s > 0 ∧ (Σ a_s > 0 ∨ a conjugate facet), D-WO7-1)**. **Rejected:** plicVolume / cs_ κ for SDF sources (orphan rows); the solid rule "κ_s > 0 and conjugate" (frozen uncoupled unknowns)  <sub>flow</sub>
- **Sealed scalar cells (fluid or solid) are not unknowns; they are gated on their volume (≤ 1e-6), not their count**. **Rejected:** a "0 sealed" count gate (grows like (R/h)²); changing the 1e-3 snap floor  <sub>flow</sub>
- **Scalar faces carry the plain aperture two-point flux**. **Rejected:** Johansen–Colella face-centroid interpolation (no measurable change)  <sub>flow</sub>
- **Immersed scalar walls use probe-flux: one probe on the facet normal, the BC eliminated per facet**. **Rejected:** the per-cell Dirichlet mask (order 0.7–1); Papac/Gibou symmetric Robin; the aperture + link hybrid; the quadratic normal probe (erratic)  <sub>flow</sub>
- **The scalar probe distance is s = 1.1·½Σ|n_a|h_a (0.55–0.95 h), the shortest that never needs the fallback**. **Rejected:** AMReX's κ-dependent short probe (order 0.7–1.2, erratic); 0.3 h (0.4–0.9); a constant √3/2·h  <sub>flow</sub>
- **The immersed scalar wall flux is never a two-point / series-resistance flux over the cut cell's centroid distance**. **Rejected:** centroid two-point and series-resistance / GFM (Liu–Fedkiw–Kang) wall fluxes (first order); the centroid-distance surrogate (ρ = 0.995)  <sub>flow</sub>
- **Conjugate scalar transport is two fields on one grid in ψ = c/K, with the series-resistance 2×2 elimination at the fluid and solid probes (P2F)**. **Rejected:** Peters' directional one-field scheme (L∞ first order, 2–50× worse); one field + side array; a mixture one-field surrogate; lagged partitioned coupling  <sub>flow</sub>
- **Scalar cut-cell geometry is the fan-tetrahedron PL model on the marching-squares samples; the scalar apertures are ungated and snapped at both ends**. **Rejected:** plicVolume for SDF sources; 4³ subsampling; the cell-centre projection as facet centroid; reusing the gated pressure openness  <sub>flow</sub>
- **ScalarMG coarsens the surrogate with rediscretized coarse faces and variational coarse wall terms: the plain average of the level-0 terms at the FINE probe distance (A1)**. **Rejected:** wall terms at the level's own probe distance (contraction 3.1, divergent); Galerkin RAP (27-point, no better); extending CutcellMG; VelocityMG's staircase  <sub>flow</sub>
- **The scalar linear solve is BiCGStab + one ScalarMG V-cycle on the lumped-probe surrogate, max-norm relative stop (default 1e-10)**. **Rejected:** fixed RB-GS sweeps; (F)GMRES; the centroid-distance surrogate  <sub>flow</sub>
- **Scalar small cells under advection: a dynamic implicit-FOU split relative to the bulk Courant number**. **Rejected (for now):** weighted state redistribution (revisit for VoF/slip walls); explicit κ storage (unstable); fully implicit FOU (diffusive)  <sub>flow</sub>
- **Scalar advection uses the projection's own constrained face flux; the scalar apertures weight diffusion only**. **Rejected:** the scalar apertures for advection (constants not preserved)  <sub>flow</sub>
- **Cut-cell scalars refuse the ghost projection (collocated 'ghost', staggered set_ghost_projection) and, for now, collocated open faces**. **Rejected:** advecting with the ghost field under any openness (a constant drifts 0.80 in 50 steps); the collocated high-side boundary plane after project()  <sub>flow</sub>
- **Scalar conservation is exact in the discretization; the solver residual is reported as a defect, never fixed up**. **Rejected:** a post-solve uniform mass correction  <sub>flow</sub>
- **Steady scalar advection is preconditioned by a V-cycle on the ADVECTIVE surrogate (summed positive sub-face fluxes on the coarse levels; A2)**. **Rejected:** the symmetric surrogate (no convergence at Pe_h 1); level-0-only advection; pseudo-transient continuation; GraphAMG; defect correction; VelocityMG coarse upwinding; line/downstream smoothers; the net coarse flux; F/W-cycles as default  <sub>flow</sub>
- **The transient ScalarMG level rule switches to the full table at κ_A ≥ 25 (A3, D-WO9-3)**. **Rejected:** κ_A = 13 (level 0 cheaper up to 25); κ_A ≈ 49 (a tie, not cheaper)  <sub>flow</sub>
```

In the core section ("In force"):

```markdown
- **Cut-cell geometry and probe kernels are container-free core headers (scheme/cut_cell_geometry.hpp, scheme/probe_flux.hpp)**. **Rejected:** flow-private copies (amr and VoF would fork them); a core Krylov  <sub>core</sub>
```

Plus the scalar-units line from [`scalar_units_register_entry.md`](scalar_units_register_entry.md).
