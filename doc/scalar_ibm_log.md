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
