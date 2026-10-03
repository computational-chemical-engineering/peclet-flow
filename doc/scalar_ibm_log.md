# Scalar transport with immersed solids — append-only log

## 2026-10-02 — first study and round-2 prototypes (2-D disc, `tests/study/scalar_ibm/`)

**Round 1** (`disc_bc.py`, `robin.py`; table in `RESULTS.md`):
- Neumann, aperture FV: 2nd order with κ in storage/source; 1st order without it.
- Dirichlet: Gibou linear ghost 2nd order; mask 0.7–1; cut-cell FV with a centroid wall flux 1st.
- Robin, series-resistance FV: 1st order.

**Round 2** (`hybrid.py`; signed mean relative error of the Robin eigenvalue at ND = 128; order in
brackets):

| Bi | P (Papac: κ-FV + k A_w u_P) | H-k (centre-fluid, slivers merged, aperture faces, links) | Hf-1 (Gibou + Robin links) |
|---|---|---|---|
| 0 (Neumann) | −6.2e-5 (2.0) | −4.2e-3 (1.0) | −4.3e-3 (1.2) |
| 0.1 | −1.1e-5 (1.4–1.8) | −3.4e-3 (1.0) | −3.2e-3 (1.1) |
| 1 | +2.5e-5 (1.5) | −2.8e-3 (1.0) | −2.6e-3 (1.1) |
| 10 | +3.4e-4 (1.7) | −8.3e-4 (1.1) | −7.0e-4 (1.1) |
| 100 | +2.2e-3 (1.5) | −2.3e-4 (1.4) | −1.3e-4 (1.5) |
| ∞ (Dirichlet) | — (degenerates to a mask) | −1.5e-4 (1.6) | −5.8e-5 (2.0) |

**Conclusions.**
- Merging slivers costs the Neumann limit; every κ > 0 cell must be its own unknown.
- Aperture-weighted fluid–fluid faces cost the Dirichlet limit in the link scheme (H-1 at Bi = ∞
  is order 1.5; full faces give 2.0).
- No symmetric compact scheme found is 2nd order at all Bi.
- Papac is the best compact Robin, but its Dirichlet limit is a mask.

## 2026-10-02 — round 3: Johansen–Colella / probe-flux FV (`jc.py`)

**Set-up.**
- Every κ > 0 cell is an unknown; κ mass; aperture two-point faces.
- Wall flux from a probe point on the wall normal, with Robin eliminated per facet:
  - D du/ds = S − β u_G;
  - u_G = D S / (k + D β);
  - flux into the cell = −k A_w u_G.

**Variants** (relative error of the Robin eigenvalue at ND = 128; order ~2.0 at every Bi unless
noted):

| variant | Bi = 0.1 | 1 | 10 | 100 | ∞ |
|---|---|---|---|---|---|
| JC quadratic normal (2 column-intersection points) | 1.2e-5 | 2.8e-6 | 4.1e-5 | 7.1e-5 | 7.5e-5 |
| JC-lin (1 point on the next column line, 3-pt quadratic transverse) | 2.8e-5 | 7.9e-5 | 1.1e-4 | 2.1e-4 | 2.1e-4 |
| probe c = 0.7 h, bilinear (4 cells, reach 2) | 2.2e-5 | 5.6e-5 | 8.4e-6 | 3.0e-5 | 3.5e-5 |
| probe c = 1.0 h | | | | | 1.5e-4 |
| probe c = 1.5 h | | | | | 4.5e-4 |

- The quadratic-normal variant has the smallest error, but it is erratic: signs and orders jump.
- The face-centroid flux interpolation makes no difference (JC vs JC-nofc agree to 3 digits), so it
  is not needed.

**Conclusion.** The decisive ingredient is a wall gradient over a probe distance ≥ ~0.7 h,
interpolated from neighbours. It never uses the cut cell's own value over its (possibly tiny)
distance. One unknown set and one formula cover Neumann → Robin → Dirichlet at 2nd order, and the
3-D stencil is compact: 27-point plus a reach of 2.

## 2026-10-02 — round 4: conjugate probe-flux FV (`conj.py`)

**Set-up.**
- Solid core r < 0.5 (D_s, capacity g_s, partition K, contact h_c), fluid annulus, u = 0 at
  R = 1 (immersed, probe Dirichlet).
- Two fields on one grid; per facet a fluid probe and a solid probe; the 2×2 interface elimination
  (pymrm `ibm_coupling` structure, in flux form) is exactly conservative.
- Exact reference: a Bessel determinant.

Error at ND = 128:

| case | error | order |
|---|---|---|
| D_s = 10 | 1.7e-4 | 2.0 |
| D_s = 0.1 | 9.2e-6 | 3.0 |
| D_s = 100, g_s = 0.5 | 2.0e-4 | 2.0 |
| K = 3 | 8.6e-5 | 2.0 |
| h_c = 5 | 4.1e-5 | 2.1 |
| D_s = 10, K = 0.5, h_c = 2, g_s = 2 | 7.0e-7 | (~2.5 to ND = 64) |

The probe fallback (renormalized valid weights, then a longer probe) fired once per case, at
ND = 16.

## 2026-10-02 — round 5: solver fit (`solver.py`)

**Set-up.**
- A = the probe-flux operator (c = 0.7 h) on the disc.
- Systems: steady, and backward-Euler with dt·D/h² = 1 or 10.
- Preconditioner: an exact solve of a symmetric 5-point surrogate S, standing in for an ideal MG
  V-cycle.

**Lexicographic Gauss–Seidel directly on A converges.** It is as slow as GS always is on the
steady problem (residual 0.7 after 200 sweeps at ND = 128) and reaches 5e-16 at dt·D/h² = 1. So
the positive off-diagonal probe weights do NOT break GS.

**Surrogate S = aperture FV + κ + diagonal wall term A_w/(h²(d + 1/k)).**

| d in the surrogate | Bi = ∞ | Bi = 10 | Bi = 1 |
|---|---|---|---|
| centroid distance (= RF) | ρ(I − S⁻¹A) = 0.995; BiCGStab 59–87 it, GMRES(20) 150–368 it | BiCGStab 3–6 | BiCGStab 2–3 |
| probe distance 0.7 h ("lumped probe": off-diagonal probe weights summed onto the diagonal) | ρ = 0.55; BiCGStab 8–9 it, GMRES(20) 15–17, mesh-independent ND 32→128 | ρ ≤ 0.27; BiCGStab 3–5 | ρ ≤ 0.05; BiCGStab 2–3 |

**Conclusion.** Use BiCGStab (or GMRES/FGMRES), preconditioned by an MG V-cycle on the SPD
lumped-probe surrogate. The surrogate is the existing aperture-weighted 7-point family plus a κ/dt
diagonal plus a diagonal wall term, so CutcellMG-style machinery applies. The probe operator itself
is applied only as a sparse cut-cell overlay matvec.

## 2026-10-02 — round 6: small cells under explicit advection (`advect.py`)

**Set-up.**
- Solid-body rotation in an annulus 0.4 < r < 1, both walls immersed.
- Exact open-face stream-function fluxes, so the aperture divergence is ≤ 2e-16.
- A Gaussian blob carried once round; full-cell CFL 0.5; FOU fluxes everywhere.

| scheme | result at ND = 32 / 64 / 128 |
|---|---|
| explicit, κ storage | blows up at all three resolutions (min κ 2e-4, 1.3e-4, 3.8e-5) |
| unit storage (flow today) | stable, but the physical mass Σκc drifts −12 %, −5.5 %, −2.1 % |
| May–Berger-style split (implicit upwind on faces touching κ < ½, explicit elsewhere, κ storage) | stable at the full-cell CFL; mass to 1e-16; positive; L1 error equal to unit's (1.50 / 1.36 / 1.14; FOU diffusion dominates) |
| fully implicit FOU | stable and conservative, but more diffusive (L1 1.63 / 1.51 / 1.33) |

**Conclusion.** κ storage forces a small-cell treatment. The explicit–implicit split is
conservative, stable and positive, and costs nothing extra because diffusion is already an implicit
(BiCGStab) solve per step. State redistribution (Berger–Giuliani) remains the alternative to
compare, and the literature (L1) decides between them.

## 2026-10-02 — probe-distance sweep (`jc.py probeK probe0.3 probe0.5 probe0.7`)

**Context.**
- AMReX `MLEBABecLap` (L1 digest §3) uses a two-point Dirichlet probe at
  dx_eb = max(0.3, (κ² − ¼)/(2κ)) h along the normal from the boundary centroid, with trilinear
  interpolation: the same family as ours.
- With bilinear weights renormalized over valid (κ > 0) cells when the stencil touches a covered
  cell, the results are as follows (error at ND = 128):

| probe | fallbacks over the sweep | Bi = 10 | Bi = ∞ | order |
|---|---|---|---|---|
| AMReX κ rule | 762 | 7.3e-5 | 9.3e-5 | 0.7–1.2, erratic |
| 0.3 h | 762 | 1.1e-4 | 1.5e-4 | 0.4–0.9, erratic |
| 0.5 h | 301 | 3.1e-5 | 1.6e-5 | 2.0–2.2, reach 1 |
| 0.7 h | 0 | 8.4e-6 | 3.5e-5 | 2.0, reach 2 |

**Conclusions.**
- The fallback (renormalization = local constant extrapolation) is what degrades short probes.
- Choose the probe so that the stencil never needs it. In 2-D, 0.7 h > √2/2·h.
- In 3-D, the guarantee that every trilinear neighbour is at least a cut cell for a planar wall
  needs c ≥ √3/2·h ≈ 0.87 h. This is a design input.

## 2026-10-02 — round 7: Frank Peters' directional conjugate IBM (`conj_peters.py`)

**Sources.**
- `~/Codes/conjugate/IBM_conjugated_transport.tex` (draft, unpublished).
- Code: gitlab.tue.nl SMM/research_projects/eajfpeters/ibm-conjugate-transport (MATLAB, Feb 2023),
  `conjugate_heat_conduction_cylinder_flux_ibm_new.m`.

**Method.** One field, one unknown per cell; a cell belongs to the phase of its centre.
- On a grid line that crosses the interface at ξ, T is a quadratic on each side through 2 cells +
  T_ξ.
- T_ξ minimizes T''(ξ⁻)² + T''(ξ⁺)²; it is k-independent and bounded as ξ → ±½.
- Code version, faithfully ported as "DIRc":
  - every face has k = (1 − a_s) k₁ + a_s k₂;
  - on crossing faces the flux is k₁ (1 − a_s) T'_fluid(0) + k₂ a_s T'_solid(0), with each one-sided
    quadratic evaluated or extrapolated to the face.
- The tex reading ("DIR": k of the side containing the face) differs from the code.
- The tex closed-form weights equal a generic Lagrange derivation to 1e-10.

**Test** (the tex's own): disc R = 1 with k₂ in a side-5 box with k₁ = 1, exact Maxwell solution,
Dirichlet box. Errors at cell centres.

**Reproduction.** Centred grid at d/Δx = 128:

| k₂ | metric | Frank's figure | DIRc |
|---|---|---|---|
| 100 | L1 | ≈1e-4 | 9.3e-5 |
| 100 | L∞ | ≈5e-3 | 4.7e-3 |
| 0.5 | L1 | 3–8e-6 | 9.0e-6 |
| 0.5 | L∞ | ≈3e-4 | 3.6e-4 |

The port reproduces the figures.

**Comparison** (2 random offsets, d/Δx = 128, L1 / L∞):

| k₂ | 0.01 | 0.5 | 2 | 100 |
|---|---|---|---|---|
| OF-a (aperture-mean one field) | 4.4e-4 / 1.5e-2 | 6.0e-5 / 2.4e-3 | 7.7e-5 / 2.4e-3 | 1.3e-3 / 1.5e-2 |
| GFM series (= Liu–Fedkiw–Kang) | 1.2e-3 / 2.4e-2 | 7.5e-5 / 5.1e-3 | 4.5e-5 / 2.6e-3 | 2.0e-5 / 2.6e-4 |
| DIRc (Peters, code) | 6.7e-5 / 6.0e-3 | 5.7e-6 / 4.0e-4 | 7.9e-6 / 4.3e-4 | 1.9e-4 / 8.4e-3 |
| DIR (Peters, tex reading) | 4.8e-4 / 2.4e-2 | 4.9e-5 / 5.4e-3 | 2.8e-5 / 2.7e-3 | 8.8e-6 / 2.6e-4 |
| DIRq (T_ξ from flux continuity) | 1.5e-3 / 9.3e-2 | 6.1e-5 / 2.4e-3 | 3.6e-5 / 1.1e-3 | 3.2e-4 / 1.1e-2 |
| **P2F (two-field probe-flux FV)** | **9.6e-6 / 1.3e-4** | **4.1e-6 / 6.4e-5** | **5.5e-6 / 9.2e-5** | 2.0e-5 / 2.3e-4 |

Orders:
- P2F: 2.0 in L1 and 1.5–2 in L∞ at every k₂.
- DIRc: 1.0–1.8 in L1, scattered with placement, and ~1 in L∞.
- Every one-field scheme: ~1 in L∞.

**Conclusions.**
1. Within the directional framework, the min-curvature T_ξ beats the "obvious" flux-continuity
   T_ξ by 2–20× in L1. It is a real element of the scheme.
2. The one-field directional scheme imposes the jump of k ∂T/∂x along grid lines, not along the
   interface normal. On tilted interfaces its L∞ stays first order. P2F, with the true normal,
   probes and per-facet elimination, is 2–50× better in L∞ at every contrast and 3–10× better in
   L1, except at k₂ = 100, where DIR (tex reading) equals it.
3. What DIRc offers: one unknown per cell, no κ, no facet geometry, a compact 4-point line
   stencil.
4. What it lacks: a path to a contact resistance (T_ξ does not see the flux), a 2nd-order Neumann
   limit (k₂ → 0 is 1st order) and Robin. A partition coefficient could enter as
   T_ξ⁺ = K T_ξ⁻ in the min-curvature problem (not tested).

**Decision.** P2F remains the conjugate method. Peters' directional scheme is recorded as a measured
alternative.

## 2026-10-02 — WO-1: core kernels (core branch `scalar-ibm`, commit `a031c6f`, not pushed)

**Built** (core worktree `suite/core-scalar-ibm`, new files only; no existing core header touched):
`include/peclet/core/scheme/cut_cell_geometry.hpp` (`triPositiveFraction`, `tetPositiveFraction`,
`CutCellGeometry`, `cutCellGeometryFanTet`, `snapAperture`, `facetAreaVector`; plane source
reserved), `include/peclet/core/scheme/probe_flux.hpp` (`probeSupport`, `trilinearStencil`,
`ProbeResult`, `buildProbe`, `wallConductance`, `interfaceConductance`), ctests `cut_cell_geometry`
(Kokkos build: the vof oracles are Kokkos headers) and `probe_flux` (host-only, plain-inline path).

**Gates** (host-openmp, Release):

| gate | measured | bound |
|---|---|---|
| (a) 1000 planar cuts, isotropic: κ / apertures / area vector ÷ h² / centroid ÷ h | 4.4e-16 / 8.9e-16 / 8.9e-16 / 1.8e-15 | 1e-14 / 1e-14 / 1e-13 / 1e-13 |
| (a) same, random anisotropic h ∈ [0.5, 2]³ | 5.6e-16 / 3.7e-15 / 2.0e-15 / 1.7e-15 | same |
| (b) 20000 random samples (¼ with exact zeros): PL closure ÷ max A | 5.6e-16; κ ∈ [0, 1]; κ = 0 ⇒ apertures 0 | 1e-14 |
| (c) gap 0.3h / slab 0.3h (x-normal, centred; also oblique, off-centre) | `gap` / `thinSolid`, 2 facets, cos(n₀, n₁) = −1 (oblique −0.99934) | — |
| (d) ladder | 1000/1000 random planar facets R0 (s ∈ [0.575, 0.953]h); covered cell → R1a (s = 0.75); wall at 0.8 s₀ → R1b (s = 0.22, 4 cells); no valid cell → R2 | — |
| (d) linear reproduction | probe weights 4.9e-16, stencil 1.3e-15 | 1e-14 |

Also: a shared face's aperture is bitwise equal from both cells (2000 random lattices, §2.2 corner
sum); F(v) + F(−v) = 1 to 6.7e-16; a V3 (reach) failure on h = (1, 1, 10) falls to R2 without
querying the Lookup. Core battery: plain 77/77, Kokkos host-openmp 93/93 (`-LE bench`). flow
`build_dev` reconfigured with `-DPECLET_SIBLING_PECLET_CORE=…/core-scalar-ibm` and built clean.

**Readings of the note made while implementing** (none changes a gate):
- corner index c = bx + 2by + 4bz (x fastest; "lexicographic in (x, y, z) bits" read with the
  suite's axis order);
- the face fan is `ccFaceOpenMS`'s exactly: tangents t1 < t2, loop c00, c10, c11, c01, summed left
  to right — the same bits from both cells;
- V3 is checked on all 8 stencil cells before the Lookup is called, so nothing beyond ±2 is ever
  read; R1b needs φ(p₀), so it is tried only when R0's stencil passed V3; R1a only when R0 failed
  V1 alone; a piece with n·n_ref = 0 joins the reference group A (facet 0);
- face area A_a = product of the other two spacings.

## 2026-10-02 — WO-2: flow geometry record (flow branch `scalar-ibm`; core `b1fcb6a` adds `fanFaceAperture`)

**Built.** `src/scalar_cutcell_geometry.hpp` (block kernels: apertures incl. the high face at
ext − G, cells, scan-compacted facet overlay + cut-cell CSR, fluid probe ladder with the block
`Lookup`, ghost-slab flag clearing, census), `src/flow_ibm_scalars_cutcell.hpp`
(`ensureScalarCutGeometry` lazy build, `invalidateScalarCutGeometry`, getters), `flow_ibm.hpp`
(`scg_`, `scgVersion_`, declarations, include), one invalidation call in `setSolidDevice` (every
geometry path — set_solid, set_pressure_geometry, set_solid_from_scene, redistribute — lands
there), bindings `diagnostics.scalar_geometry(name)` and the geometry part of
`diagnostics.scalar_census(name)`, ctests `scalar_cutcell_geometry` and
`scalar_cutcell_geometry_mpi_np{1,2,4}`. Core gained `fanFaceAperture` (the face fan as one
function, used by `cutCellGeometryFanTet` itself — bitwise the old expression), so the face kernel
and the cell kernel cannot drift.

**Gates** (host-openmp; spheres SOLID, measured volume Σ(1 − κ)V; RMS over 3 offsets):

| | R/h = 8 | 16 | 32 | order | bound |
|---|---|---|---|---|---|
| (e) volume, iso | −9.77e-3 | −2.44e-3 | −6.10e-4 | 1.997, 2.002 | 2(h/R)², ≥ 1.8 |
| (e) area Σ\|A_φ\|, iso | −9.03e-3 | −2.25e-3 | −5.61e-4 | 2.002, 2.008 | 3(h/R)², ≥ 1.8 |
| (g) volume, h' = (1,1,2) | −4.89e-3 | −1.22e-3 | −3.05e-4 | 1.999, 2.000 | same |
| (g) area | −4.40e-3 | −1.10e-3 | −2.72e-4 | 2.002, 2.016 | same |

- every cut cell `single`, min ρ 0.9927 (R/h = 8) → 0.9996 (32), iso and aniso;
- (f) max |A^snap − areaPL| / max A = 1.77e-3 (bound 2e-3);
- pipe (G7, R/h 16/32/64): R0 100 %, 0 sealed;
- MPI np = 1, 2, 4 (periodic sphere; sphere cut by a −x wall with ±y slip): every cell field,
  every high face and every facet (alpha, normal, centroid, body, s, rung, weights, offsets)
  BITWISE equal to the single-rank build by global index; census equal;
- §2.5 prerequisite: sdf_ ghost layers 1–2 exact on every face slab — single-rank periodic, wall
  extension, slip mirror; MPI exchange + extension at np 1, 2, 4;
- G12: 12/12 state hashes identical (OMP_NUM_THREADS=1);
- single-rank non-bench battery 61/62: the one failure is `scalar_cutcell_geometry`, on exactly
  the two clauses below (17 sealed checks, 6 aniso R0 checks); `_np{1,2,4}` of the new MPI test pass;
- device: the block kernels compiled with nvcc (nvidia-cuda prefix) in a standalone TU and run on
  the RTX 5080: facets 2657 / R0 2657 / unknowns 58051 / sealed 25, solid volume equal to the
  OpenMP build to 10 digits.

**Found while gating — two clauses of the note did not hold as written:**
1. (e) "0 sealed" fails at every resolution: 0–11 / 38–46 / 154–159 sealed cells (iso), up to 407
   (aniso). They are genuine §2.4 sealed cells — corner slivers of fluid, κ ≤ 1.3e-5, whose three
   faces all snap to 0 at the 1e-3 floor. Their total volume is ≤ 1.9e-9 of the fluid volume, far
   under the §9 warning threshold (1e-6). The count grows like (R/h)², so no resolution reaches 0.
2. R0 = 100 % holds on the isotropic spheres and the pipe but not on h' = (1,1,2): 1–11 facets per
   run (≤ 2.2e-4 of them) take R1b, none R2. Mechanism: on a tiny corner facet the x/y faces snapped
   to 1 while the z face (half the area) did not, so the SNAPPED area vector points exactly along
   ±z, and the probe along it enters the sphere.

**DECISIONS (orchestrator ruling, 2026-10-03):**
- D-WO2-1 — sealed cells. Gate (e)'s "0 sealed" is restated as: sealed volume Σκ·V over sealed cells
  ≤ 1e-6 of the fluid volume (the §9 threshold; measured ≤ 1.9e-9). The sealed count stays in the
  census. The sealed definition (§2.4) and the snapping rule (§2.3) are NOT changed.
- D-WO2-2 — R0 on anisotropic grids. R0 = 100 % stays required on the isotropic sphere set and the
  pipe set; on h' = (1,1,2) the gate is R0 ≥ 99.9 % of facets and ZERO R2 (measured: min 99.97 %,
  R2 = 0). Held option for WO-8: probe along the UNSNAPPED PL normal while keeping the snapped area
  for the flux magnitude.
- The five WO-2 readings are accepted as made: `scalar_geometry` / `scalar_census` take any
  registered scalar name until WO-3 adds the cut-cell flag, `probe_rungs = {'fluid': (R0, R1a,
  R1b, R2)}`; facets for every cell whose PL record has them (no unknown filter); `num_cut_cells` =
  cells carrying ≥ 1 facet; one invalidation call in `setSolidDevice`; the §9 warnings deferred to
  WO-3/WO-8.

With the restated asserts `scalar_cutcell_geometry` passes, as do `_mpi_np{1,2,4}`.

## 2026-10-03 — round 8: near-contact conjugate (`tests/study/scalar_ibm/packing2d.py`)

**Set-up.**
- Periodic square array, one cylinder per unit cell, gap g = 1 − 2R, k_f = 1, macroscopic gradient.
- k_eff from the face-column flux.
- Methods: PH = Peters' hybrid ladder (his `construct_flux_ibm.m` transcribed to 2-D: directional
  quadratic where there is room, appendix series-parallel k_eff on close / very-close faces);
  P2F = two-field probe FV, which does NOT detect a probe inside another solid; P2Fg = P2F with
  s_f = min(0.7 h, ½·normal gap).
- Check: k_s = k_f gives k_eff = 1 to 1e-6 (n = 16) … 1e-8 (n = 64). This needed two fixes in the
  port: DOF validity from the apertures, and the facet sign.
- Reference: Richardson (64, 128). All three agree to 0.05–0.2 %.

**k_s = 100.** The "%" is the error vs the reference.

| case | PH | P2F | P2Fg |
|---|---|---|---|
| g = 0.05, n = 32 | −0.2 % | −1.1 % | −1.1 % |
| g = 0.02, n = 32 | −0.7 % | −2.2 % | −1.2 % |
| g = 0.02, n = 16 | −1.8 % | −12 % | −3 % |
| g = 0.01, n = 32 | −1.2 % | −7 % | −1.2 % |
| g = 0.01, n = 16 | −2.2 % | −29 % | −3.5 % |

**k_s = 0.01.**
- PH is non-monotone in small gaps:
  - g = 0.02: −23 % at n = 16 and n = 32;
  - g = 0.01: 0.0645 / 0.0577 / 0.0580 / 0.0644.
- P2F and P2Fg are smooth and monotone, ≤ 0.8 % at n = 16 in every case.

**Conclusions.**
1. Probes landing in the neighbouring body are THE near-contact failure of the probe FV for
   conducting contacts. A gap cap at ½ the normal gap removes most of it. This supports the design's
   ladder (φ(p0) < 0 → R1b gap-midpoint); WO-8 must gate exactly this case.
2. Peters' series-parallel contact formula is the most robust at coarse resolution for conducting
   contacts (≈1–2 % at 0.3 cells per gap), but fails for insulating gaps (−10 to −23 %).
3. Possible WO-8 refinement, held: for conducting contacts with gap < ~h, a film-conduction coupling
   of the two solid probes through the gap, k_f/δ_n in series; equivalent in spirit to Peters'
   very-close formula. Only if the G13 gate demands it.

## 2026-10-03 — WO-3 start: orchestrator rulings

The handoff is in `doc/scalar_ibm_wo3_handoff.md` (untracked by repo convention).

**Merged tree verified** (184ae9f): build clean; `units_|scalar_cutcell_geometry` 14/14; G12 12/12.

**Units (design §1.2 amended by Q6).**
- Reuse `diffToInt()` (D), `divToInt()` (S), `volToPhys()·divToPhys()` (body flux) and
  `areaToPhys()`.
- Append only `speedToInt()` = tRef/hRef, for Robin k and Neumann q.
- `resistToInt` → WO-7.

**Rulings.**
- **D-WO3-1.** Steady gates G1/G2/G3a/G7 run on the two coarsest rungs with maxit 3000 in WO-3; the
  full ladder reruns with WO-4.
- **D-WO3-2.** `scalar_geometry`/`scalar_census` reject non-cut-cell names.
- **D-WO3-3.** Rebuild the operator every advance; caching → WO-9.
- **D-WO3-4.** A no-op guard for faces toward non-unknown cells is allowed.

## 2026-10-03 — WO-3: single-phase operator, Krylov, steady + transient diffusion

**Built.**
- `src/scalar_cutcell_operator.hpp` (new): `ScalarCutState` (settings verbatim/physical, the
  operator of the last advance, Krylov scratch + statistics) and the block kernels — the 7 bands
  with the D-WO3-4 guard and identity rows, the per-facet `cw/rw` from a resolved per-body wall
  table (`wallConductance`), the CSR overlay matvec and rhs, the lumped surrogate diagonal (§4.3),
  the Dirichlet domain-face fold (constant or profile), per-facet probe value + influx, the
  inner-cell reductions.
- `src/scalar_krylov.hpp` (new): `ScalarVec {f, s, solid}`, `ScalarKrylovOps`, `scalarBiCGStab` —
  `CutcellMG::solveBiCGStab`'s recurrence, breakdown and stagnation guards and singular mean
  projection, with the §5.1 stop (ref = max(max|b|, max|A x0|), max-norm, both half-steps), one
  true-residual check + one restart, non-finite raises.
- `flow_ibm_scalars_cutcell.hpp`: the setters, refusals, `scalarCutAssembleSolve` (zero
  non-unknowns, c^n, bands, wall table, rhs, domain folds, surrogate, singular detection +
  projection + gauge, level-0 preconditioner = 2 + 2 RB-GS sweeps by global parity on the
  surrogate, exchange before each colour), `scalarWallFlux`, `scalarBudget`, `scalarFacets`.
- `flow_ibm_scalars.hpp`: the one-line dispatch (§5.5); `addScalar(..., cutcell)`; `setScalarBc`
  clears a profile; `scalarDirichletMask` refuses a cut-cell scalar. `flow_ibm_phase_change.hpp`:
  `set_phase_change_thermal` naming a cut-cell scalar is a ValueError. `flow_ibm.hpp`: declarations,
  the includes, `UnitScales::speedToInt()` appended. `scalar_transport.hpp`: `ScalarField` gains
  `cutcell` + `cut` (struct members only; no kernel touched).
- Bindings: `add_scalar(iters=None, cutcell=False)`, `set_scalar_bc` profile overload,
  `set_scalar_wall`, `set_scalar_source` (float / array), `set_scalar_tolerance`,
  `solve_scalar_steady`, `scalar_wall_flux`; diagnostics `scalar_budget`, `scalar_facets`,
  `set_scalar_max_iterations`, the solve part of `scalar_census`; `scalar_geometry` /
  `scalar_census` reject non-cut-cell names (D-WO3-2).
- Tests: `tests/python/test_scalar_cutcell_gates.py` (ctests `scalar_cutcell_{api,g1,g2,g3a,g3b,g7}`),
  `tests/kokkos/test_scalar_cutcell_operator.cpp`, `tests/kokkos_mpi/test_scalar_cutcell_solve_mpi.cpp`
  (np 1, 2, 4).

**Gates** (host-openmp, OMP_NUM_THREADS=4; RMS over 3 offsets; every case in physical units;
steady gates on the two coarsest rungs, D-WO3-1):

| gate | coarse | fine | order | bound | iterations |
|---|---|---|---|---|---|
| G1 Nu/2 − 1, R/h 8 → 16 | −2.488e-2 | −6.510e-3 | 1.93 | ≥ 1.7 | 20–21 / 40–43 |
| G1 field L1 / L∞ | 3.12e-3 / 3.75e-2 | 8.11e-4 / 9.62e-3 | 1.94 / 1.96 | ≥ 1.8 / 1.5 | |
| G1 aniso (1,1,2), R/h_min 16 → 32 | −1.185e-2 | −3.056e-3 | 1.95 | ≥ 1.7 | 35–37 / 69–76 |
| G2 c_Γ rel, R_i/h 6 → 12 | +2.315e-2 | +6.266e-3 | 1.89 | ≥ 1.7 | 19–20 / 35–37 |
| G2 field L1 | 1.54e-3 | 3.79e-4 | 2.02 | ≥ 1.8 | |
| G3a Sh rel, Da 0.1 / 1 / 10 / 100 | −9.90e-3 / −1.48e-2 / −2.26e-2 / −2.46e-2 | −2.48e-3 / −3.80e-3 / −5.88e-3 / −6.44e-3 | 2.00 / 1.97 / 1.94 / 1.93 | ≥ 1.7 | 19–23 / 37–54 |
| G3b μ rel, Bi 0.1, R/h 8/16/32 | 2.43e-4, 7.16e-5 | 2.06e-5 | 1.76, 1.80 | ≥ 1.7; ≤ 2e-3 | 20–22 / 33–39 / 58–70 per step |
| G3b Bi 1 | −1.25e-3, −2.48e-4 | −5.29e-5 | 2.34, 2.23 | | 12–14 / 24–26 / 47–56 |
| G3b Bi 10 | 1.46e-2, 3.73e-3 | 9.51e-4 | 1.97, 1.97 | | 9–10 / 16–19 / 30–35 |
| G3b Bi 100 | 2.27e-2, 5.60e-3 | 1.41e-3 | 2.02, 1.99 | | 9–10 / 16–18 / 27–34 |
| G3b Bi ∞ | 2.36e-2, 5.81e-3 | 1.46e-3 | 2.02, 1.99 | | 10–12 / 16–18 / 28–33 |
| G7a Dirichlet j01², R/h 16 → 32 | 1.640e-3 | 4.19e-4 | 1.97 | ≥ 1.8; ≤ 5e-4 | 19–20 / 35–38 per step |
| G7a Neumann j′11² | −2.05e-4 | −5.17e-5 | 1.99 | ≥ 1.8; ≤ 5e-4 | 32–36 / 59–72 |
| G7b Graetz Nu_T | 8.83e-4 | 2.18e-4 | 2.02 | ≥ 1.7; ≤ 1e-3 | 21–26 / 39–44 per solve |

- Budget: G1 |wall − box − defect| ≤ 2.2e-15 |wall|, |defect| ≤ 3.2e-10 |wall|; G3b identity
  ≤ 2.8e-14 |d_mass| on every step of every case (bound 1e-13). G2: the inner wall's flux equals
  q × its discrete facet area to 4e-15. R0 = 100 % on every isotropic sphere and pipe; aniso R1b
  on ≤ 4 of 12 650 facets, no R2 (D-WO2-2).
- Operator test: guard fires on 0 interior faces; 7-point part bitwise symmetric; identity rows
  exact; steady Neumann row sums ≤ 2e-16 AC; Dirichlet g = 1 → max|c − 1| = 7.6e-12; insulating
  transient c = 5 exact; singular steady (flux + source, periodic box): incompatibility 1.0, 24
  iterations, gauge Σκc = 4e-13; budget identity (Robin + source + Dirichlet faces) 4e-14 / 5e-14
  relative, transient / steady.
- MPI (`scalar_cutcell_solve_mpi`): np = 1 bitwise (field, iterations 30/11/11/11, wall flux);
  np = 2, 4: max|c − c₁| = 1.6e-15 (max c 1.46), identical iteration counts.
- G12: 12/12 hashes equal at OMP_NUM_THREADS=1. Battery: 205/205 (`-LE bench`, 628 s).
- Iterations: the level-0 preconditioner needs about 2.3 × R/h per steady solve, not the O(N)
  feared in D-WO3-1 — maxit 3000 is never approached.

**Readings / decisions (none changes a gate):**
- `ScalarField` is defined in `scalar_transport.hpp`, so its two new members and a forward
  declaration of `ScalarCutState` went there (the note says `flow_ibm.hpp`); `ScalarCutState`
  lives in `scalar_cutcell_operator.hpp`.
- D-WO3-1's maxit is raised through a diagnostics-tier setter, `diagnostics.set_scalar_max_
  iterations(name, maxit)` (default 200) — a new name for WO-10's NAMING rows.
- The restart runs within the same iteration budget; `krylov_converged` = true residual
  ≤ 10 rtol ref; `krylov_residual` = true residual / ref.
- `scalar_budget` after a steady solve is the rate balance: d_mass = 0, defect = Σ V r,
  identity_error = defect − (wall_in + boundary_in + source_in).
- The budget evaluates r = b − A c in FLUX form (Σ t (c_i − c_nb), the same t as the bands): exactly
  the band-form residual in exact arithmetic, but the band form's round-off (ε t |c|) gave 2–9e-13
  |d_mass| at Bi = 0.1 (dt D/h² ≈ 3500), over the 1e-13 gate; the flux form gives ≤ 2.8e-14. The
  solve itself uses the band form.
- G7a Neumann: the dipole has zero mass, so the BE ratio is taken on the first moment Σ κ V x c.
  G7a and G7b both ran on {16, 32} (D-WO3-1 lists G7 among the steady gates).
- `scalar_facets` centroids follow `cell_centers()` (origin + (i + ½) h, also in cell units).
- The per-cell source refuses a block that changed size (redistribute) rather than dropping it.
- Not in WO-3: the advection census keys (`num_small_cells`, `num_implicit_faces`,
  `bulk_courant`, WO-5), the §9 resolution warnings (WO-8), dirty-flag caching (D-WO3-3, WO-9).

## 2026-10-03 — WO-4: ScalarMG, single phase — G-iter and G10 NOT met (stopped on two questions)

**Built.**
- `src/scalar_mg.hpp` (new): `ScalarMG` per §5.2. Level table = VelocityMG's rule verbatim (single
  rank and distributed in place, no depth cap), level 0 the scalar's G = 2 block (the solver's
  exchange injected), coarse levels g = 1 with their own `GridHalo<double>`. Surrogate in face form,
  double. Level 0: AC = WO-3's SAC, faces -Lam w a with sco::buildBands' guard (so the level-0 sweep
  is bitwise the band sweep). Coarse (rediscretized, D10): face <Lam a> by `coarsenOpenAvgCell` on
  the guarded products, w_a(L) = w_a/cfac^2; mass by `restrictAvg`; wall W_C = (1/N_L) sum alpha
  G(s_L) gathered from the level-0 facets at s_L = 1.1 * 1/2 sum |n_a| H'_a(L); pins = max of the
  children; Dirichlet domain faces 2 w_a(L) <Lam a_bf> from a per-level boundary plane. RB-GS by
  global parity, exchange per colour; V-cycle pre 2 (R->B), residual after a fresh exchange,
  `restrictAvg`, singular: coarse rhs mean removed, `prolongAdd`, pinned re-zeroed, post 2 (B->R);
  bottom 16 sweeps (8 R->B + 8 B->R) + mean removal. Level rule kappa_A < 13 -> level 0 alone.
  `contraction()`: power estimate of rho(I - M^-1 S), deterministic global-index seed.
  Q2's fallback (Galerkin RAP, restrict = average, prolong = trilinear incl. the zero ghosts and
  the pinned re-zero) is implemented for EVALUATION behind the C++-only `setGalerkin` (default
  off); its coarse operator is 27-point and is swept with 8 per-axis-parity colours (see Q-B).
- Wiring in `flow_ibm_scalars_cutcell.hpp` (level table per geometry version/block, coefficients
  every build, `precond` = one V-cycle); `ScalarCutState` gains `mg`, `mgVersion`, `mgN`,
  `mgLevels`; census `mg_levels` is now the levels used.
- Tests: `tests/kokkos/test_scalar_mg.cpp` (ctest `scalar_mg`: level table == VelocityMG on 8
  grids incl. odd, short and h' = (1,1,2), (1,1.5,3), (2,1,1); coarse identity rows / M-matrix /
  face rule; level rule; contraction at the finest G1 and on G5b's geometry);
  `test_scalar_cutcell_solve_mpi.cpp` (G10 problems `mixed`, `g1` R/h = 16, `singular`, and the
  distributed level table == VelocityMG::initMpi); `test_scalar_cutcell_gates.py`: the full ladder
  for g1/g2/g3a/g7, every successive order, the provisional absolute bounds, G-iter rows on
  g1/g2/g3a(Da = 1) and a new `giter` gate (transient dt D/h^2 = 1; singular steady on G5b's
  geometry); maxit back to the default 200 (D-WO3-1's 3000 removed).

**Inert proof.** Transient runs with dt D/h^2 in {0.3, 0.5, 0.9} (level 0 alone): fields and
iteration counts `np.array_equal` to the WO-3 build (f0fc80a/9736d6f) — 6/6 arrays. G12: 12/12
hashes equal at OMP_NUM_THREADS=1.

**Accuracy, full ladder** (RMS over 3 offsets; preconditioner-independent at rtol 1e-10):

| gate | rungs | errors | orders | bound at finest gated rung |
|---|---|---|---|---|
| G1 Nu/2 - 1 | 8/16/32 | 2.488e-2, 6.510e-3, 1.664e-3 | 1.93, 1.97 | 1.66e-3 <= 2e-3 |
| G1 field L1 / Linf | 8/16/32 | 3.12e-3/3.75e-2, 8.11e-4/9.62e-3, 2.07e-4/2.40e-3 | 1.94/1.96, 1.97/2.01 | |
| G1 aniso (1,1,2) | 16/32 | 1.185e-2, 3.056e-3 | 1.95 | |
| G2 c_Gamma | 6/12/24 | 2.315e-2, 6.266e-3, 1.637e-3 | 1.89, 1.94 | 1.64e-3 <= 2e-3 |
| G2 field L1 | 6/12/24 | 1.54e-3, 3.79e-4, 9.42e-5 | 2.02, 2.01 | |
| G3a Sh, Da 0.1 | 8/16/32 | 9.90e-3, 2.48e-3, 6.20e-4 | 2.00, 2.00 | 6.2e-4 <= 3e-3 |
| G3a Da 1 | | 1.48e-2, 3.80e-3, 9.60e-4 | 1.97, 1.98 | 9.6e-4 |
| G3a Da 10 | | 2.26e-2, 5.88e-3, 1.50e-3 | 1.94, 1.97 | 1.5e-3 |
| G3a Da 100 | | 2.46e-2, 6.44e-3, 1.65e-3 | 1.93, 1.97 | 1.65e-3 |
| G3b mu (Bi 0.1/1/10/100/inf) | 8/16/32 | unchanged from WO-3 to 3 digits | unchanged | |
| G7a Dirichlet j01^2 | 16/32/64 | 1.640e-3, 4.193e-4, 1.062e-4 | 1.97, 1.98 | 4.19e-4 <= 5e-4 |
| G7a Neumann j'11^2 | 16/32/64 | 2.056e-4, 5.176e-5, 1.295e-5 | 1.99, 2.00 | 5.18e-5 <= 5e-4 |
| G7b Graetz Nu_T | 16/32/64 | 8.827e-4, 2.176e-4, 5.403e-5 | 2.02, 2.01 | 2.18e-4 <= 1e-3 |

Budget: G1 |wall - box - defect| <= 3.8e-14 |wall|, |defect| <= 3.7e-11 |wall|; G3b identity
<= 3.0e-14 |d_mass|. R0 = 100 % on every isotropic rung.

**Iterations, WO-3 (level 0) -> WO-4 (ScalarMG)** (per steady solve, or per step):
G1 8/16/32: 20-21 / 40-43 / (not run) -> 10-12 / 13-14 / 15. G1 aniso 16/32: 35-37 / 69-76 ->
11-13 / 14-15. G2 6/12/24: 19-20 / 35-37 / - -> 12-13 / 11 / 16-17. G3a 8/16/32: 19-23 /
37-54 / - -> 4-9 / 4-10 / 5-11. G3b per step at 32: 27-70 -> 7-16. G7a Dirichlet 16/32/64:
19-20 / 35-38 / - -> 10-12 / 11-14 / 18-20; Neumann: 32-36 / 59-72 / - -> 8-9 / 14-15 / 28-31.
G7b: 21-26 / 39-44 / - -> 10-13 / 13-14 / 21-22. Singular, G5b geometry n = 20/40/80: 5 / 5 / 5-6.
A 128^3 steady G1 solve takes ~2 s at 4 threads.

**G-iter (single phase): NOT met.**
- G1 <= 20 and growth <= 3: pass (12, 14, 15). G3a Da 1: pass (4, 4, 5). Singular (G5b
  geometry): pass (5, 5, 6; <= 30, growth <= 5).
- G2 growth: FAIL (13, 11, 17: +6). The G2 box n = ceil(2 Ro/h) + 6 = 36/66/126 gives a 3/2/2-level
  table (66 -> 33, 126 -> 63: a 63^3 bottom under 16 sweeps). On boxes rounded up to a multiple of
  16 (48/80/128, 5/5/7 levels; the cells added are solid) the same solves take 12-13 / 15 / 17
  (+2, +2, pass). The gate's box is the test's choice, not the note's (Q-C).
- Transient dt D/h^2 = 1 (<= 8 per step): FAIL by one or two — 10 on the cold first step, 8-9 after,
  at every rung 8/16/32 (5/6/7 levels).
- Contraction <= 0.3 (power estimate, geometric mean of the last 10 of 30): FAIL at the finest G1.

| case | rediscretized (D10) | Galerkin RAP, 8-colour GS (Q2, eval.) | averaged W, s = s0 (R7-rejected; probe only) |
|---|---|---|---|
| G1 R/h = 32, 128^3, Dirichlet sphere + box | **0.788** (15 it) | **0.358** (11 it) | 0.274 |
| G1-like 64^3, R = 16 | 0.470 (13 it) | 0.303 (10 it) | 0.222 |
| periodic box + Dirichlet sphere, 64^3 | **3.105** (15 it) | 0.284 (14 it) | 0.503 |
| Dirichlet box, no solid, 64^3 | 0.175 | 0.334 | — |
| periodic, no solid (singular) | 0.179 | 0.167 | — |
| Neumann sphere + Dirichlet box | 0.185 | 0.336 | — |
| Neumann sphere, periodic (singular) | 0.193 | 0.167 | — |
| G5b geometry n = 80 (ND 66.4, singular) | 0.193 (6 it) | 0.173 (6 it) | — |
| G5b geometry n = 77 (ND 64.0) | 0.968 (1 level: 77 is odd; 78 it) | same | — |

Reading: the plumbing is not the cause (Neumann and no-solid cases contract at 0.18-0.19; the V-cycle
is bitwise decomposition-independent, below). The rediscretized wall term W_C at s_L ~ 2^L s_0
under-represents an immersed DIRICHLET wall once the coarse cell approaches the body size (R/h =
32 has 7 levels, bottom cell 64h; s_L > R from level 5): the coarse correction of the smooth mode
overshoots — divergent as a stationary iteration when the sphere is the only Dirichlet condition.
Averaging W with the fine s (the variant §12 R7 rejects) contracts at 0.27; Galerkin RAP at 0.36.
§5.2's argument (rescaling keeps the wall/face ratio level-independent) holds per cell but not for
the coarse correction of the low modes. This is the note's "revisit" (ii) territory only in part:
RAP misses 0.3 even WITHOUT a solid (0.334, a property of the average/trilinear pair), so the 0.3
bound and the RAP smoother are both open (Q-A, Q-B).

**G10: np = 1 bitwise; np > 1 NOT met on G1.**
- np = 1: field, iterations and wall flux bitwise on `mixed`, `g1`, `singular`; the distributed
  level table equals VelocityMG::initMpi's.
- np = 2 / 4: `mixed` 1.56e-12 / 5.52e-12 (iterations equal); `singular` 3.55e-15 / 7.11e-15; `g1` (R/h = 16, walls on
  all faces) **7.85e-9 / 7.26e-8** of max|c_1| = 1.05, iterations 13 / **15** vs 13 — over the
  1e-9 bound and the +-1 rule.
- The V-cycle is not the cause: z = M^-1 r for a fixed global-index r is BITWISE equal np = 2 vs
  single on `mixed` and `g1` (8.9e-16 on `singular`, its mean sums). At rtol 1e-13 the `g1` fields
  agree to 6.45e-12 / 3.58e-11 with equal iterations (16): the gap is BiCGStab's sensitivity to the
  dot-product order, amplified to the stopping error (lambda_min of S ~ 5e-3 for the 64^3
  Dirichlet box: |c - c*| up to ~ 2e-8 at rtol 1e-10). WO-3's level-0 preconditioner was never run
  on G1 under MPI. (Q-D)

**Questions (stopped; nothing chosen):**
- Q-A (Q2 / revisit (ii)): rediscretized ScalarMG misses the 0.3 contraction at the finest G1
  (0.788) although its iteration counts pass; RAP (0.358) also misses, and so does RAP on a plain
  Dirichlet box (0.334). Which coarse construction ships, and is 0.3 the right bound for it?
- Q-B: Galerkin RAP's coarse operator is 27-point, so red-black is not a Gauss-Seidel colouring
  (same-colour edge neighbours race). Its smoother — 8-colour GS (measured above, 8 exchanges per
  sweep) or a colour-Jacobi RB, or something else — is not fixed by the note.
- Q-C: G2's box (ceil(2 Ro/h) + 6) and G5b's ND = 64 (n = 77, odd: a one-level table) make G-iter
  measure the level table. Round the boxes to MG-friendly sizes in the gates (cells added are
  solid / the array period changes by < 4 %), or keep them and accept?
- Q-D: G10's 1e-9 / +-1 on G1 at the default rtol 1e-10 is not reachable by any decomposition-
  independent preconditioner under BiCGStab (gap = stopping error). Gate at rtol 1e-13 (passes:
  3.4e-11 rel), or restate the bound relative to rtol / lambda_min?
- Transient G-iter misses <= 8 by 1-2 iterations (10 cold, 8-9 warm): bound or cycle?

**Battery** (`-LE bench`, 207 tests, OMP 4, -j2, 1155 s): 202 pass; the 5 failures are exactly the
gate rows above — `scalar_cutcell_g2` (G2 growth), `scalar_cutcell_giter` (transient <= 8),
`scalar_mg` (contraction), `scalar_cutcell_solve_mpi_np2/_np4` (G10 on `g1`). Committed with those
gates left failing as evidence (not loosened), pending the answers.

**Not changed:** CutcellMG / VelocityMG sources; the legacy scalar path; the probe operator.

## 2026-10-03 — WO-4 rulings (orchestrator) and their reruns

**Decisions.**
- **D-WO4-1 (Q-C).** G-iter and its growth are gated on multigrid-friendly boxes: each box rounded
  UP to a multiple of 2^(levels+1) per axis (the register's factors-of-two rule for benchmark grids,
  docs/DECOMPOSITION_AND_MULTIGRID.md §3.1). A box with few factors of two stays gated for
  correctness only. Applied to G2 and to G5b's geometry. *Reading (implementer's, reversible in
  `mg_box()` of the gates test):* "levels" = the number of ladder rungs, 3 -> multiples of 16 —
  the rounding the WO-4 evidence used. G2: 36/66/126 -> 48/80/128 (the added cells lie outside
  R_o: solid; c_Gamma errors identical to 4 digits). G5b geometry: 20/40/80 -> 32/48/80 (the box is
  the array period, so ND becomes 26.6/39.9/66.4; the rungs are no longer doublings).
- **D-WO4-2 (Q-D).** G10 is gated at rtol 1e-13: np = 1 bitwise; np = 2, 4 fields within 1e-10 of
  the single-rank run (read as <= 1e-10 max|c_1|; every case also passes it absolute), wall flux
  to the same bound, iterations +-1 (the vardensity_mpi_np4 precedent). At the default rtol the
  np > 1 gap is printed as information only.
- **D-WO4-3.** Transient G-iter at dt D/h^2 = 1: the provisional <= 8 is restated as <= 10 per
  step, the cold first step included (§13 Q1: about 2x measured).
- **D-WO4-4 (Q-A / Q-B held).** The shipped coarse construction stays the note's rediscretized
  levels (c1e312e). The contraction <= 0.3 row stays FAILING (ctest `scalar_mg`) pending the
  architect; no switch to RAP or to the R7 (averaged-W) variant. The RAP evaluation path stays
  C++-only and off.

**Reruns** (OMP 4).
- G2 G-iter on the mg boxes: 12-13 / 15 / 17 (growth +2, +2) — pass; native boxes 13 / 11 / 17,
  correctness only.
- G5b geometry, singular, n = 32/48/80: 5 / 5 / 5-6 — pass.
- Transient: 10 (cold) then 8-9, every rung R/h 8/16/32 — pass at <= 10.
- G10 at rtol 1e-13 (np = 1 / 2 / 4):
  - mixed: bitwise / 5.33e-15 / 5.55e-15; iterations 15 7 7 7 vs 14 7 7 7 at np 2, 4 (+-1).
  - g1: bitwise / 6.45e-12 / 3.58e-11 of max 1.053; iterations 16 = 16.
  - singular: bitwise / 3.55e-15 / 7.11e-15; iterations 6 = 6.
  - Information, g1 at rtol 1e-10: 7.85e-9 / 7.26e-8, iterations 13 / 15 vs 13.
- G12: 12/12 hashes equal at OMP_NUM_THREADS=1.
- ctest: `scalar_cutcell_g2`, `scalar_cutcell_giter` and `scalar_cutcell_solve_mpi_np{1,2,4}`
  pass; `scalar_mg` fails only on the held contraction row (0.788).

**Extra measurement (requested).** Periodic box 4R with a Dirichlet sphere (c = 1) and a source,
steady — the case where the stand-alone rediscretized V-cycle diverges (rho = 3.1 at 64^3). With
the V-cycle as the BiCGStab preconditioner it converges at every offset: R/h = 16 (64^3, 6 levels)
13 / 14 / 14 iterations, true residual 3e-11 .. 9e-11 of the reference; R/h = 32 (128^3, 7 levels)
15 / 16 / 16, 1e-11 .. 6e-11. The diverging mode is few-dimensional and the Krylov method removes it.

## 2026-10-03 — WO-4 completion: Amendment A1 (coarse wall term at the fine probe distance)

**Change** (design Amendment A1, 03435df). The coarse gather averages the level-0 wall terms:
W_C = (1/N_L) Σ cw(φ), with cw = α G(s_φ) — the fine probe distance (rung included), the same
gather and summation order. Robin is covered by cw; WO-7's conjugate coupling goes the same way.
Coarse faces stay rediscretized: 7-point, RB-GS. RAP was removed (`setGalerkin`, the assembly and
the 8-colour sweep); its evidence stays in c1e312e and in the WO-4 entry. `ScalarMG::Inputs` takes
`facetW` (= cw) instead of the per-body wall table.

**Contraction guard, restated per A1** (`scalar_mg`; power estimate, geometric mean of the last 10
of 30; C1 < 1 everywhere, C2 <= 0.35, C3 <= 0.75):

| row | before A1 (s_L) | after A1 (s_φ) | BiCGStab iterations after |
|---|---|---|---|
| C2 G1 R/h 8 / 16 / 32 | – / 0.470 / 0.788 | 0.186 / 0.222 / 0.274 | 9 / 10 / 11 |
| C2 G2 Ri/h 6 / 12 / 24 (mg boxes 48/80/128) | – | 0.280 / 0.272 / 0.323 | 10 / 10 / 11 |
| C2 G3a Da 1, R/h 8 / 16 / 32 | – | 0.172 / 0.177 / 0.180 | 4 / 5 / 5 |
| C2 Neumann sphere + Dirichlet box 64^3 | 0.185 | 0.185 | 5 |
| C2 no solid, Dirichlet box 64^3 | 0.175 | 0.175 | 5 |
| C2 G5b geometry n 32 / 48 / 80 (singular) | – / – / 0.193 | 0.190 / 0.190 / 0.193 | 5 / 5 / 6 |
| C1 G5b geometry n = 77 (odd: one level) | 0.968 | 0.968 | 78 |
| C3 periodic + Dirichlet sphere R/h 16 / 32 | 3.105 / – | 0.504 / 0.523 | 13 / 14 |

Every row passes C1-C3. (Neumann, no-solid and singular rows are unchanged, as they must be: they
carry no wall conductance.)

**Reruns.**
- Inert: the level-0-only runs remain `np.array_equal` to WO-3 (6/6). G12: 12/12 hashes equal.
- Accuracy ladder: every order and bound line is identical to the pre-A1 run (the preconditioner
  moves the solution only within rtol).
- G-iter (stop rule: no row worse by more than 2 — none is; most improve):
  - G1 12/14/15 -> 10/10/11;
  - G2 (mg box) 13/15/17 -> 10/11/12;
  - G3a Da 1 4/4/5 -> 4/5/5;
  - transient max 10 -> 10;
  - singular 5/5/6 -> 5/5/6;
  - G3b per step at R/h 32: Bi 1 11-13 -> 11-12, Bi 10 7-8 -> 7, Bi 100 9-10 -> 9, Bi inf 12 -> 11-12;
  - G7: unchanged within 1 (Dirichlet 64: 18-20; Neumann 64: 28-31; Graetz 64: 21-22).
- G10 at rtol 1e-13, np = 1 / 2 / 4:
  - mixed: bitwise / 1.33e-15 / 1.33e-15, iterations 14 7 7 7 at every np;
  - g1: bitwise / 1.13e-13 / 2.13e-14, iterations 14 = 14;
  - singular: bitwise / 3.55e-15 / 7.11e-15, iterations 6 = 6.
  - Information, g1 at the default rtol: 8.19e-13 / 1.50e-13, iterations 10 = 10 (before A1:
    7.85e-9 / 7.26e-8 and 13 / 15 vs 13). The pre-A1 gap was the indefinite preconditioner
    amplifying the reduction-order noise, A1's reading (c), not BiCGStab alone.
- Battery (`-LE bench`, OMP 4, -j4): 207/207 pass, 717 s. **WO-4 done.**

## 2026-10-03 — WO-5: advection and small cells (§6, D12) — G9b decided, two gates held, three questions

**Built.**
- `Solver::scalarFaceFlux(a)` (flow_ibm_scalars_cutcell.hpp): the one predicate (§6.1). It returns
  the projection's face field and the openness ITS divergence kernel weights it with
  (`getOpennessProj`'s rule: `oxb_` under the ghost projection, `ox_` otherwise); staggered
  `C[a].u`, collocated the projected `uf_/vf_/wf_`. F/V = open * vel (index velocity: A_a/V = 1/h'_a
  is inside it). Read-only; no projection/advection code or `ufAdvVelocity` touched.
- scalar_cutcell_operator.hpp: face fluxes guarded toward non-unknowns (D-WO3-4's analogue; it also
  zeroes wall/slip domain faces through the cleared ghost flags), C_bulk (global MAX over full cells),
  the small flags over inner + ghost layer 1 (no exchange), the implicit FOU bands (outflow per cell
  kept for the surrogate), the explicit faces (koren/sou/fou on c^n, FOU when the 2-up/1-down
  stencil leaves the unknowns), the explicit rhs, and the budget's implicit-flux residual.
  Steady: implicit FOU on every face (§6.7). Everything is skipped when no face carries flux.
- scalar_mg.hpp: the lumped outflow rides on the restricted mass (m = κ idt + ω), i.e. §5.2's
  "restrictAvg of the level-0 per-cell field"; `hasOutflow` false is the WO-4 expression verbatim.
- Census keys `num_small_cells`, `num_implicit_faces`, `bulk_courant`, plus two new diagnostics names
  for WO-10's NAMING rows: `num_flux_faces` (the implicit fraction's denominator) and
  `num_guarded_flux_faces` (projection flux toward a non-unknown, dropped; 0 on every gate case).
- Refusals added (RuntimeError at the advance): the ghost projection (Q13, below); a flow
  inflow/outflow face (Q-E); a moving fluid without the cut-cell projection (Q-F).
- Tests: gates `scalar_cutcell_g9`, `scalar_cutcell_g9b` (ctests), the api refusals,
  `scalar_cutcell_solve_mpi` problem `g9` (G10), `scalar_mg` rows "C3 + advection".

**Inert proof.** Zero-velocity runs (G1-like steady, transient Robin + source at level 0, transient
full table, singular steady, collocated gauge-exact stepping): 41/41 arrays (fields, iterations, MG
levels, residuals, wall flux, budget) `np.array_equal` to the pre-WO-5 build (fddf042). G12: 12/12
hashes equal at OMP_NUM_THREADS=1.

**G9** (annulus 0.4 < r < 1, nz 4, solid-body rotation, Gaussian σ 0.08 at r 0.7, one revolution,
D = 0 as round 6). The face field is node-potential differences of the clamped stream function, set
to the wall value on both nodes of every face the projection closes (`ox_` is gated, so the exact
open-face fluxes of round 6 put 46 % of max|F| on closed faces); its divergence in the predicate's
openness is round-off. One offset per rung with min κ < 1e-2.

| R_o/h | scheme, C | steps | small cells | implicit faces | BiCGStab | (i) identity / M0 | (ii) |ΔM+Σdefect| / M0 | min c | L1 (cut cells / band share) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | fou 0.5 / 0.9 | 333 / 185 | 28 | 1.06 % | ≤ 6 / 7 | 5.0e-15 / 4.1e-15 | 1.5e-16 / 1.2e-16 | 0 / 0 | 1.536 / 1.482 (8 % / 20 %) |
| 16 | koren 0.5 / 0.9 | 333 / 185 | 28 | 1.06 % | ≤ 6 / 7 | 5.7e-15 / 8.0e-15 | 2.0e-16 / 1.4e-16 | −2.7e-10 / **−0.20** | 0.756 / 1.006 |
| 32 | fou 0.5 / 0.9 | 700 / 389 | 64 | 0.60 % | ≤ 7 / 8 | 1.1e-14 / 1.1e-14 | 2.0e-16 / 9.7e-17 | 0 / 0 | 1.405 / 1.350 (4 % / 11 %) |
| 32 | koren 0.5 / 0.9 | 700 / 389 | 64 | 0.60 % | ≤ 7 / 8 | 1.3e-14 / 9.5e-15 | 4.0e-16 / 1.8e-16 | −4.1e-10 / **−0.23** | 0.561 / 0.811 |
| 64 | fou 0.5 / 0.9 | 1427 / 793 | 64 | 0.15 % | ≤ 6 / 8 | 2.2e-14 / 2.5e-14 | 2.5e-16 / 6.5e-17 | 0 / 0 | 1.201 / 1.141 (1.4 % / 4.5 %) |
| 64 | koren 0.5 / 0.9 | 1427 / 793 | 64 | 0.15 % | ≤ 7 / 8 | 2.5e-14 / 3.1e-14 | 1.3e-16 / 2.6e-16 | −4.7e-10 / **−40** | 0.635 / 2.50 |

- min κ over unknowns 3.0e-5 / 3.4e-5 / 1.0e-5 (gate (v)); finite throughout; max c ≤ max c0 for
  fou and koren 0.5. (ii) sanity at rtol 1e-13 (fou 0.9): raw drift 7.6e-15 / 2.1e-15 / 8.5e-15 M0.
  C_bulk measured 0.499–0.500 / 0.899–0.900. ΔM = −defect per step is scalar_budget's sign.
- **Held failing: koren at bulk Courant 0.9, gate (iii)** (Q-G). It is the bulk scheme, not the cut
  cells: forward Euler + the legacy Koren reconstruction is TVD only to C ≤ 1/2 (C_i = 1 + ψ/2r −
  ψ_{i−1}/2 ≤ 2). Control with NO solid (periodic box, uniform translation): min c = −2.4e-10 at
  C_bulk 0.5, **−21** at 0.9; the annulus at R_o/h 16: −2.7e-10 (0.5), −2.6e-10 (0.6), −9.9e-3 (0.7),
  −0.20 (0.9), every minimum in a full cell. sou (unlimited) at C 0.25: −0.20 … −0.47 likewise.
- Q8: the cut band carries 1.4 % (cut cells) / 4.5 % (± one cell) of the fou L1 at R_o/h = 64 and
  0.0 % / 0.3 % of koren's — not dominated; WSRD not triggered. L1 is FOU/time-error dominated:
  fou 1.54 → 1.41 → 1.20 (round 6: 1.50 / 1.36 / 1.14); koren at C 0.25: 0.85 → 0.36 → 0.32; koren
  at C 0.5 0.76 → 0.56 → 0.63 (the limiter at its FE bound; cut share 0 % at 64).

**G9b** (Stokes through an SC array c = 0.3, 32^3, body force (30, 9, 0), 50 steps, c = 1, D = 0,
C_bulk 0.47): staggered, collocated 'gauge-exact', 'plain', 'embed': max|c − 1| = 0 exactly
(the predicate's divergence ≤ 2e-13 max|F|). **'ghost' fails:** max|c − 1| = 4.1e-2 (C 0.016),
0.80 (C 0.47), 1.28 (C 0.86); its face field has max|Σ o u| / max|F| = 4.3e-2 with the binary
openness `oxb_` and 2.0e-2 with the geometric `ox_` — the constraint is the binary divergence PLUS
the closure delta (`gpDivergDelta`), so no openness-weighted face flux of it is divergence-free. The
staggered `diagnostics.set_ghost_projection` path is the same kernel pair: 4.5e-2. Per §13 Q13 both
are **refused** for a cut-cell scalar (also at rest, as the rule reads). This is revisit item (iii)
for 'ghost' — the AUTO default of SolverColocated — so a cut-cell scalar there needs an explicit
`set_collocated_scheme('gauge-exact'|'plain'|'embed')`.

**G10 on G9** (36^3 z-invariant annulus R_o 16 cells, koren, 50 steps, C_bulk 0.60, rtol 1e-13,
cut-cell projection): np = 1 bitwise (field, iterations, counts); np = 2 / 4: max|c − c1| =
2.78e-16 of max 0.617, iterations identical (1–3 per step), small cells 180 and implicit faces
360 of 46 944 identical on every decomposition; budget identity ≤ 2.4e-14 of the mass. The WO-4
problems unchanged (mixed 1.11e-15, g1 6.04e-14, singular 5.33e-15 at np 4).

**Iterations with implicit upwind bands** (the brief's check).
- WO-4 rows: no advection → bitwise unchanged (inert proof); `scalar_mg` C1–C3 rows identical.
- Transient, advecting: G9 (D = 0, level 0) ≤ 8 per step; annulus R_o/h 32 with D at dt D/h² = 3
  and 30 (full table, 3 levels), koren C 0.5: 5 and 6 per step advecting = 5 and 6 at rest. Not
  degraded.
- **Steady, advecting: degraded** (Q-H). The C3 problem (64^3 periodic, Dirichlet sphere R 16,
  source) with a projected Stokes field rescaled to peak cell Péclet Pe_h = max|u| h/D:
  13 iterations at rest → 19 (Pe_h 0.03), 40 (0.1), 89 (0.3), **200 not converged** (1; residual
  1.7e-4), 127 / 200 with residual ≥ 1 (3, 10). The A1 guard on the surrogate passes (0.135 at
  Pe_h 1, 0.071 at 10, vs 0.504 at rest) — it measures the symmetric surrogate, which carries no
  advective coupling, so it cannot see this. Without the lumped outflow (an experiment, reverted):
  158 / 130 iterations at Pe_h 1 / 10 — the diffusion-only surrogate is no better.

**Questions (stopped; nothing chosen beyond a refusal):**
- **Q-E** (§6.5): the advective rows of an inflow/outflow domain face need the projection's flux
  through the HIGH boundary face. advanceScalars' plain `fillGhosts(Uf/Vf/Wf)` (legacy, before the
  dispatch) overwrites that first-ghost-index plane with the opposite face's value, so the predicate
  cannot read it. Recover it how — save the plane before the fill (edits advanceScalars), the
  boundary cell's mass balance, or the BC's own velocity? Refused until decided.
- **Q-F** (§6.1): with `set_solid(..., cutcell_pressure=False)` (the default) no projection runs and
  `ox_` is never built, so a moving fluid would be silently unadvected. Refused when the fluid moves;
  a fluid at rest keeps WO-3/4's behaviour. Keep, or refuse cut-cell scalars without the cut-cell
  operator altogether?
- **Q-G** (§11 G9 (iii)): koren at bulk Courant 0.9 cannot be bounded by FE + the legacy Koren
  reconstruction (TVD to 1/2, measured above without any solid). Gate koren at C ≤ 1/2, or change
  the time integration of the explicit part (an SSP-RK2 would double the explicit work)? The check
  is left FAILING in `scalar_cutcell_g9` as evidence, not loosened.
- **Q-H** (§4.3, §6.7, revisit (ii)): the lumped symmetric surrogate does not precondition steady
  advection beyond Pe_h ≈ 0.1 (40 iterations; 89 at 0.3). WO-6's G8 Taylor–Aris at Pe = UR/D = 10
  and R/h = 32 has a peak Pe_h of 0.3–0.6, against G-iter's singular-steady ≤ 30. Transient is unaffected. Needs a design decision (an advection-aware coarse operator
  / smoother, or a different steady driver); not tuned.

**Battery** (`-LE bench`, OMP 4, -j4, 765 s): 208/209 pass; the one failure is
`scalar_cutcell_g9`, exactly its three held koren-C-0.9 (iii) rows (Q-G).

**Not changed:** projection/advection code, `ufAdvVelocity`, the legacy scalar kernels and
advanceScalars, CutcellMG/VelocityMG.

## 2026-10-03 — WO-5 orchestrator rulings

- **D-WO5-1 (Q-G).** Koren is gated at bulk Courant ≤ ½. Forward Euler with the Koren limiter is TVD
  only to ½; with no solid it reaches min −21 at C = 0.9. This is a bulk-scheme limit,
  legacy-identical.
- **D-WO5-2 (Q-F).** Keep refusing cut-cell scalars in a moving fluid without `cutcell_pressure`. A
  fluid at rest is allowed.
- **D-WO5-3 (Q-E).** Inflow/outflow faces take the boundary-face flux the projection constrained,
  captured right after the projection. New gate G9c: a channel with a solid and inflow/outflow, with
  the budget closing including the boundary fluxes.
- **D-WO5-4.** The collocated 'ghost' scheme is refused for cut-cell scalars (Q13): its face field
  is not divergence-free under any openness (2–4.5e-2). Collocated users select gauge-exact, plain
  or embed.
- **Q-H → architect (brief `doc/scalar_ibm_brief_A2.md`).** Steady advection-diffusion is not
  preconditioned by the symmetric surrogate: 13 → 40 → 89 → no convergence at Pe_h 0 / 0.1 / 0.3 / 1.

## 2026-10-03 — WO-5b: the follow-ups to WO-5 (rulings D-WO5-1..4) — staggered open faces done, collocated held (one question)

**D-WO5-1.** G9 (iii) for koren is gated at bulk Courant ≤ ½; the koren C = 0.9 rows print (iii) as
`INFO` (min −0.20 / −0.23 / −40 at R_o/h 16 / 32 / 64, as WO-5); their (i), (ii), (iv), (v) stay
gated and pass. Every G9 number is unchanged from WO-5 (periodic: the inert proof below).
**D-WO5-2.** Kept; the message already names `set_solid(..., cutcell_pressure=True)` and
`set_pressure_geometry`; the api gate now checks both are named. **D-WO5-4.** Kept; the message lists
'gauge-exact', 'plain', 'embed' and "the staggered Solver without set_ghost_projection" (api gate).

**D-WO5-3, built (staggered).**
- `Solver::scalarCaptureOpenFaceFlux` (flow_ibm_scalars_cutcell.hpp), called by `step()` right after
  `project()` and the domain-BC re-imposition (keeps outflow), inside the Picard loop (the last
  iteration wins), before any ghost fill: per inflow/outflow global face this rank touches, the plane
  `open * vel` at the boundary face (low: index G; high: the first ghost index) read through
  `scalarFaceFlux`, over the FULL transverse extent (the transverse ghost rows feed ghost layer 1's
  small flags, §6.3). Taken whether or not a cut-cell scalar exists (a plane copy; inert for the
  flow), so a scalar added to a developed flow can be solved at once; cleared by every geometry
  build.
- scalar_cutcell_operator.hpp: `openFaceFluxes` writes the plane into `phi_a` at the boundary face
  (where the inner cell is an unknown), so Out_i, C_bulk and the small flags see it. §1.4's rows in
  `openFaceAdvection`: inflow b += F_in g_bf (the scalar's Dirichlet value, never implicit); outflow
  F_out c_i (zero-gradient exit), explicit at c^n, implicit (AC and the lumped `outflow` += F_out)
  for a small inner cell or steady. `openFaceResidual` / `openFaceInflux` carry them into the budget:
  `boundary_in` now = the Dirichlet faces' diffusive flux + the open faces' advective flux (§1.6.1's
  "domain terms"; docstring updated). Census: the open faces are counted in `num_flux_faces` /
  `num_implicit_faces` (`openFaceCountsLocal`).
- The interior kernels (`advectionBands`, `explicitFaceFluxes`, `advectionResidual`,
  `countImplicitLocal`) now skip a face toward a non-unknown — inert elsewhere, where the guard
  already makes that flux 0.
- Refusals: the WO-5 open-face refusal is gone on the staggered grid. NEW (not in the rulings; a
  refusal, reversible): an open face with a moving fluid and no projection yet (e.g.
  `advance_scalars` after `set_field` only) raises, naming `step()`.

**Inert proof** (vs the pre-WO-5b build, 26e917f): 198/198 arrays `np.array_equal` — G1-like
steady with Dirichlet faces, transient Robin + source at level 0 and on the full table (3 steps
each), the periodic annulus (fou C 0.9, koren C 0.5, 25 steps: fields, budgets, census), G9b
staggered + collocated gauge-exact (8 steps), and a periodic projected-Stokes steady + transient
advecting solve. **G12:** 12/12 hashes = `doc/scalar_ibm_baseline_hashes.txt` at OMP_NUM_THREADS=1
(the `channel` and `colocated_advect_bc` cases run the capture).

**G9c** (ctest `scalar_cutcell_g9c`): channel 2 × 1 × 1 (48 × 24 × 24), inflow U = 1 / no-slip y, z /
outflow, μ 0.05, a sphere R 0.23 + a cap (R 0.2, centred outside the outlet, cutting it); D = 0.01,
c0 = 0.2, Dirichlet 1 at the inlet, Neumann exit; one flow-through (t ≈ 2). The per-step identity
is measured with the mass change summed exactly (`math.fsum` of κV(c − c^n)): the budget's own
`d_mass` differences two O(M) reductions, whose round-off (≤ 1e-13 M) is ~1e-12 of one step's change
in a long channel. `identity_error / M` is gated at ≤ 1e-12 as a sanity check.

| row | steps | C_bulk | small / impl. faces | identity / \|dM\| | identity_error / M | run cons. / M0 | c − [0.2, 1] |
|---|---|---|---|---|---|---|---|
| koren | 286 | 0.461 | 0 / 0 | 4.6e-14 | 7.0e-14 | 5.8e-14 | [−1.4e-12, +2.0e-11] |
| fou | 148 | 0.888 | 27 / 79 | 3.4e-14 | 6.3e-14 | 4.9e-14 | [−3.5e-12, −1.6e-10] |
| koren, Dirichlet outlet 0.5 | 40 | 0.413 | 0 / 0 | 1.4e-14 | 8.1e-14 | 7.7e-15 | — |

Outlet mean c at the end 0.60 (the front crossed the outflow face). Constant c = 1 with inflow 1:
max|c − 1| = 0 (20 steps). Steady (implicit FOU, D = 0.5, Pe_h 0.083): 17 iterations, rate balance
1.6e-14 of Q c_in. fou is bounded to the solve's rtol (1e-10, implicit diffusion), not 1e-12.

**G10 row g9c** (`scalar_cutcell_solve_mpi`): 32³ channel along z, inflow −z / walls x, y / outflow
+z, a sphere and a cap cutting the outlet across the np = 2 (x) and np = 4 (y) rank boundaries, fou
at C_bulk ~1.1, 30 NS steps. np 1 bitwise; np 2 / 4: max|c − c₁| = 7.8e-16 / 6.7e-16, iterations
identical (5, 3, …), small cells (≤ 75; 10 in the outlet layer of the single-rank run, chosen by a
scan of the cap) and implicit faces (≤ 253) equal per step on every decomposition; identity ≤ 1.8e-13
M. G9b unchanged (0 on all four projections). **Battery** (`-LE bench`, OMP 4, -j4, 809 s): 209/209,
plus `scalar_cutcell_g9c` registered after it and passing.

**Question (stopped; the collocated grid keeps a refusal):**
- **Q-I** (§6.5 on `SolverColocated`): the projected face field does NOT hold the flux the
  projection constrained on a high-side open face. `projectCorrectVelocities` runs a plain
  `fillGhosts(uf_/vf_/wf_)` (which wraps that first-ghost plane from the opposite face) and then adds
  `bcCorrectOutflow`'s correction to the wrapped value. Measured: a constant c = 1 drops by 0.48 in
  the outlet column in one step (interior ≤ 2e-11). There is nothing correct to capture after
  `project()`. Options: (a) capture inside `projectCorrectVelocities`, before that fill, as
  uf*(bf) − w(φ(bf) − φ(bf−1)) (projection code); (b) fix the flow side by saving and restoring the
  plane around that fill, as `fillVelGhostsTo(..., doOutflow=false)` does for the staggered grid.
  This may move the collocated outflow numerics; `colocated_advect_bc` is a G12 hash. (c) keep the
  refusal. Refused for now with a named error (api gate + G9c).
- Noted, not decided: backflow on an outflow face uses §1.4's signed F_out c_i literally. Implicit,
  this gives AC += F_out < 0. No gate exercises it.

**Not changed:** the projection, ufAdvVelocity, the legacy scalar path, ScalarMG / the
preconditioner, steady mode (Q-H / A2), the design note.

## 2026-10-03 — WO-5b orchestrator rulings

- **D-WO5b-1 (Q-I).** Collocated open faces stay REFUSED for cut-cell scalars (option c).
  - **FOUND (a possible flow issue beyond scalars):** after the collocated projection, the outlet
    plane of `uf_` holds the inlet's values. `projectCorrectVelocities` fills ghosts before the
    outflow correction.
  - Collocated momentum advection reads `uf_` (register: "COLLOCATED momentum advection uses the
    PROJECTED face field"), so it may be affected at open boundaries.
  - To be investigated separately; it touches projection code and would move the
    `colocated_advect_bc` hash. Not fixed inside this campaign.
- **D-WO5b-2.** The G9c identity is gated against the exactly summed (fsum) mass change, plus
  `identity_error` ≤ 1e-12 of M as a sanity check. Accepted.
- **D-WO5b-3.** Accepted: the new refusal (open face + moving fluid before the first projection)
  and the inert boundary capture.
- **Known limitation:** outflow backflow uses the signed F_out·c_i (zero-gradient). It is ungated.

## 2026-10-03 — WO-5c: the steady advective surrogate (Amendment A2) — built, every iteration and contraction bound met; stopped on G-adv (v)

**Built** (per A2, §6.7; nothing tuned).
- `scalar_mg.hpp`: `Inputs::advective` (+ the operator's bands AW…AT, `phi[3]`, `omegaOpen`). On the
  advective path level 0 is the operator's 7 bands with SAC as the diagonal (views, no storage);
  each coarse level gets lazily allocated Qp/Qm (6) and bands (6). Q coarsening: level 1 from the
  guarded positive parts of the level-0 flux (`smg::GuardedPart`, advectionBands' steady coupling
  predicate), deeper levels from the finer Q, by `coarsenOpenAvgCell`'s sum and division, then
  ×1/r_a; `fill` of the 6 arrays, the low-face plane of every non-periodic axis zeroed. Assembly:
  AC_L = [AC as today, mass = κ·idt + ω_open] + ((Qm_x(i) + Qp_x(i+e_x)) + (Qm_y(i) + Qp_y(i+e_y)))
  + (Qm_z(i) + Qp_z(i+e_z)) (evaluated left to right, as written), AW = AFX(i) − Qp_x(i),
  AE = AFX(i+e_x) − Qm_x(i+e_x), …; pinned rows identity. Band-form colour sweep (unknown cells of
  the colour), band-form residual, both in A2's summation order; `applySurrogate` on the advective
  path is the operator's band matvec `applyCutcellOp(SAC, AW…AT)` (u4's reference); `contraction()`
  follows. Cycle, transfers, colours, exchanges, bottom and level rule untouched.
- `scalar_cutcell_operator.hpp`: `ScalarCutState::omegaOpen` (steady advecting solves only: the
  open-face rows' implicit outflow, written by `openFaceAdvection`'s new optional output, clamped
  at 0 afterwards — read as max(Σ_open faces F_out, 0)); `maxCellPecletLocal` (interior faces = both
  cells unknown; λ = 0 with flux gives +inf).
- `flow_ibm_scalars_cutcell.hpp`: `in.advective = steady && st.advecting` with the views; the
  transient advecting path keeps `outflow` on the mass verbatim; `max_cell_peclet` as the third
  entry of the advance's MAX reduction. `flow_bindings.cpp`: census key `max_cell_peclet`.
- Tests: `scalar_mg` rows u1–u6 (+ `test_scalar_mg advective` runs them alone); MPI problem
  `steady_adv` with the z = M⁻¹r check; gate `gadv` (ctest `scalar_cutcell_gadv`, CMakeLists).

**Inert proof** (vs the unmodified 9ac02d3 build, saved before any change): 170/170 arrays
`np.array_equal` — at rest: G1-like steady (Dirichlet faces), transient Robin + source at level 0
and on the full table (3 steps each), singular steady (G5b geometry); transient advecting: G9
annulus R_o/h 16 fou and koren at C 0.5 (50 steps, every 5th: field, census, budget), the annulus
R_o/h 32 with D at dt·D/h² = 3 and 30 (8 steps each), the G9c channel koren C ≤ ½ and fou C ~0.9
(15 steps of `step()`, fields incl. u). `scalar_mg` C1–C3 rows unchanged (0.186 … 0.523, the same
digits as WO-4/A1). **G12:** 12/12 hashes = `doc/scalar_ibm_baseline_hashes.txt` at
OMP_NUM_THREADS=1.

**G-adv (vi)** (one-time): (a) R/h 16, Pe_h 0.1, rtol 1e-13, the same face field (scale computed in
Python, independent of the build): max|c_A2 − c_WO5| = 9.8e-12 max|c| (≤ 1e-9); iterations 43
(WO-5) → 13 (A2).

**`scalar_mg` A2 rows** (cell units).
- (u1) uniform flow 0.75 along x, y, z on 32³ and 32×32×4 (5 levels; (2,2,1) levels): 0 of 14 040
  / 2 040 entries differ from Qp_a = φ/cf_a, Qm_a = 0.
- (u2) divergence-free random field, 32³: max|out − in| / max Q = 4.7e-16, 5.9e-16, 1.0e-15,
  7.5e-16, 8.6e-16 (levels 0–4); column sums ≤ 3.6e-16 max AC.
- (u3) C3 at Pe_h 10: min (AC − Σ|b|)/AC = −4.5e-16 … −5.5e-16 on levels 0–4, +3.7e-2 at the
  bottom; no positive band. **Needs a divergence-free field:** with the default pressure rtol 1e-10
  the margin is −5.8e-13 (level 0), −3.3e-13, −1.5e-13 (levels 1, 2) — A2's row sums are exact only
  up to the field's divergence. The C4/u3 harness therefore runs the pressure PCG to 1e-14 (test
  only).
- (u4) level-0 advective applySurrogate vs the band matvec with SAC: 0 of 262 144 rows differ.
- (u5) channel 64×32×32, inflow −x / outflow +x, walls y, z, a sphere: 3 424 domain-face entries
  over 5 coarse levels, 0 nonzero; steady 8 iterations, Pe_h 0.26.
- (u6) **C4** (C3 at R/h 16, census Pe_h): 0.566 / 0.680 / 0.764 at Pe_h 0.1 / 1 / 10 (11 / 16 / 24
  iterations) — C1 and ≤ 0.90 (prov.) pass.

**G-adv** (physical units, OMP 4; iterations at Pe_h 0.1 / 1 / 10; all converged to rtol 1e-10 —
(i) passes everywhere):

| row | 32³ or R/h 16 | 64³ or R/h 32 | bound (prov.) | growth per doubling |
|---|---|---|---|---|
| (a) C3 problem | 8 / 13 / 23 | 10 / 17 / 33 | 20/25/40, 25/35/55 | 1.25 / 1.31 / 1.43 |
| (b) closure, Neumann (singular) | 5 / 8 / 12 | 7 / 10 / 17 | 15 / 20 / 30 | 1.40 / 1.25 / 1.42 |
| (b′) Dirichlet spheres | 12 / 10 / 15 | 12 / 13 / 18 | 15 / 20 / 30 | 1.00 / 1.30 / 1.20 |
| (c) G9c channel, Pe_h 1 / 10 | 8 / 9 | — | 40 | — |

Q20's tightened bounds (2× measured) would be (a) 16/26/46 and 20/34/66 — not applied (the note
says "may"; the orchestrator's call). Census Pe_h equals the target to ≤ 1e-9 relative on every
row. (c) sets Pe_h through D (the captured open-face flux cannot be rescaled by `set_field`).
(b)'s source: u_x = the cell average of its two x-face velocities (`get_u`), mean κ-weighted over
the fluid unknowns — a harness reading of "u_x − ⟨u_x⟩".

**G-adv (v): FAILS — the normalization of "relative" is open (stopped; nothing chosen).**
- (a): |identity_error| / max(|wall_in|, |source_in|, |boundary_in|) = 1.1e-12 at R/h 16 and
  1.9e-12 at R/h 32, the same at every Pe_h — **and the same at rest** (Pe_h 0: 1.15e-12 / 1.88e-12)
  and with the WO-5 build (1.14e-12 at Pe_h 0.1). It is the round-off of the budget's N-cell sums on
  this problem (a uniform source balanced by one wall), independent of A2 and of advection; it
  grows ~√N.
- (b), (b′): the source is mean-free, so wall_in, source_in and boundary_in are all round-off
  (≤ 1e-13); relative to them the identity is O(1) (4.7e-1 … 1.1). Relative to the gross source
  V Σκ|s| it is 1e-17 … 2.1e-15 (printed as INFO by the gate).
- (c): 1.5e-14 of the inflow rate Q c_in (G9c's normalization) — passes.
- The same quantity fails the existing `compare()` identity check (≤ 1e-12 of |wall_in|) on
  `steady_adv` in `scalar_cutcell_solve_mpi`: 4.7e-13 at OMP 2 but 1.1e-12 at OMP 4 on the
  SINGLE-RANK reference itself — the thread count changes the reduction order, i.e. round-off.
- Left FAILING as evidence (`scalar_cutcell_gadv`, `scalar_cutcell_solve_mpi_np{1,2,4}`), not
  loosened.

**G10 row `steady_adv`** (64³ periodic, Dirichlet sphere R 16, Pe_h 1 from a single-rank Stokes
field set on every block, rtol 1e-13): np 1 bitwise (field, 22 iterations, wall flux); np 2 / 4:
max|c − c₁| = 4.7e-13 / 3.1e-12 of max|c₁| 182.4 (≤ 1e-10 relative), iterations 22 = 22; the
advective V-cycle's z = M⁻¹r for a fixed global-index r: **0 of 262 144 cells differ at np 1, 2,
4**. Other problems unchanged (mixed 1.33e-15, g1 2.13e-14, singular 7.11e-15, g9 6.9e-17, g9c
6.7e-16 at np 4).

**G-perf.** V-cycle at 64³ (6 levels, C3): advective 3.2–3.6 ms vs symmetric 2.9–3.2 ms on the same
block, ratio 1.00–1.23 over two runs on a loaded host (expected ≈ 1.5; red flag > 2: none).
Steady G-adv(b) at 64³ (OMP 4): 0.16 s (Pe_h 1, 10 iterations), 0.25 s (Pe_h 10, 17 iterations).

**Battery** (`-LE bench`, 211 tests, OMP 4; run in three pieces on a host at load 70–100 — the
first two were cut by the 30-min background limit and by a restart with OMP_WAIT_POLICY=passive /
OMPI_MCA_mpi_yield_when_idle=1, scheduling only): 207 pass; the 4 failures are exactly the (v)
identity rows — `scalar_cutcell_gadv` (18 FAIL lines, every one a (v) row) and
`scalar_cutcell_solve_mpi_np1/np2/np4` (only the identity CHECK on `steady_adv`).

**Not changed:** the face-form path (transient, steady at rest, `hasOutflow` on the mass), A1's
wall gather, the probe operator, the Krylov driver, CutcellMG / VelocityMG, the discretization.

**Question (stopped):**
- **Q-J (G-adv (v)).** "≤ 1e-12 relative" against what? Against the net terms it is round-off-
  limited on (a) at rest and in the WO-5 build already (1.1–1.9e-12, ~√N), and ill-posed for the
  mean-free closure source of (b)/(b′). Options: the gross source V Σκ|s| (or a gross flux scale);
  a bound that scales with N·ε; or compensated summation in the budget (changes no solve, but
  `scalar_budget` output digits). The same choice decides `compare()`'s identity check for
  `steady_adv`.
