# Sharp viscous stress at a VoF interface on the staggered MAC grid — design note

Status: DESIGN (architect, 2026-10-31). Branch `sharpvisc` (on `mom2`). USER 2026-10-31: "Yes give the
sharp interface stress for Peclet priority".
Brief: `bubble_column_perf/validation/highratio/BRIEF_ARCH_SHARPVISC.md`; recon digest `RECON_VISCOUS.md`.
A-priori harness (2-D MAC Stokes, committed with this note): `tests/study/sharp_viscous/`.
Builds on `doc/full_viscous_stress.md` (cited as FVS) and `doc/variable_viscosity_projection.md`.

---

## 0. Summary

- **Diagnosis (measured, §3).** Near a planar interface the exact flow has two local modes.
  - The *across* (shear) mode has a kink in ∂ₙu_t. Its traction is continuous, so it needs the series
    (harmonic) mean along each link.
  - The *along* (stretching) mode has continuous strain and a discontinuous stress. It needs the
    parallel (arithmetic) mean over each face, plus a pressure jump 2[μ]D_nn.
  - On an oblique interface both modes project onto every MAC stress sample. One scalar μ per sample
    therefore cannot be exact for both, and every average is first order. Measured on rotated two-layer
    Couette/Poiseuille, N = 16…128:
    - arithmetic (flow today): L∞ order 0.90–0.96, L1 1.08;
    - PLIC-exact face fractions: L∞ 0.73–0.94;
    - symmetric 4-cell harmonic: exact at 0°, first order at 22.5°/45°;
    - per-link harmonic (GFM, Liu–Fedkiw–Kang): exact on shear, but first order on stretching.
  - The along mode has a second, independent defect. With a cell-centred pressure, **no** per-sample
    viscosity makes the stretching-mode viscous force a discrete gradient. The balance condition reduces
    to a discrete wave equation that a step does not satisfy except at 0° and 45°. Every scalar scheme
    measures first order at 22.5° in both L∞ and L1, for example arithmetic 1.42e-1 → 1.86e-2.
- **Decision (§4, §5).** A conservative sharp operator with three ingredients:
  1. On every stress sample in the interface band, μ̄ is the arithmetic mean over **that row's own MAC
     face**, with the area fraction taken from the PLIC plane. This makes the along mode exact.
  2. A per-link interface-traction correction, `T_ℓ(τ̂)(1 − μ̄ R̄_ℓ)`.
     - R̄_ℓ is the series resistance of link ℓ, weighted by the link's own fraction.
     - τ̂ is the tangential interface traction, from a small kinked least-squares fit per interfacial cell.
     - This makes the across mode exact.
  3. A cell-centred **midsection correction** 2(μ_A−μ_B)(C̄⁽ᶜ⁾−C)(∂_cu_c − D̂_nn). It is the
     finite-volume-exact pressure force of the viscous normal-stress jump 2[μ]D_nn (the answer to "must
     the normal-stress jump move into the pressure?": yes, in this form; §5.6).
- **A-priori result (2-D, §3.4).**
  - Couette and pure stretching: exact to round-off at every angle (0–45°), every sub-cell offset and
    every μ ratio from 1e-3 to 1e3.
  - Poiseuille and Poiseuille+stretching: L∞ order 1.93–1.95 on the finest pair, L1 order 2.00.
  - Full 8-offset sweep: §3.4.
- **Time treatment (§6).** The implicit per-component band changes in one place: the edge β becomes
  the PLIC face-fraction mean. The cell μ(C) is unchanged.
  - Everything else is explicit on uⁿ, as the transpose term is today: the transpose, the τ-terms and
    the midsection term.
  - Measured amplification ρ(G) of predictor + projection, 2-D, 6 geometries, D_hi = 0.1 … 1e4: ≤ 0.95
    at μ ratio 1e-3 / ρ 1e-4, ≤ 0.88 at 1e-2 / 1e-3, ≤ 0.99 at 1e3. The alternative split (midsection
    term with a face-fraction cell β) is unstable, ρ up to 19.7, and is rejected.
- **Register impact (§13).**
  - Kept: register 272 (full stress, explicit conservative transpose on uⁿ).
  - Reversed, inside the VoF band only: the premise that T is symmetric. Its proof (FVS §6) is replaced
    by a measured spectral gate.
  - Extended: register 212 ("μ on the CV's own faces"), now with exact PLIC face fractions.

---

## 1. Problem and scope

**Built.** On the staggered `Solver`, when VoF is on, μ = LinearMix(C), full stress is active
(`fullStressActive()`) and μ is not globally uniform: a second-order discretisation of ∇·(2μD) across the
interface. The interface is not grid-aligned. The operator stays implicit-solvable, on-device and
MPI-distributable (np-bitwise).

**Out of scope.**
- Momentum advection (muscl_phase).
- Curvature/CSF beyond the normal-stress-jump term this note forces.
- AMR, Snellius.
- `SolverColocated`: plan in §8, not built here.
- Immersed solids: AUTO is already 'laplacian' with solids (FVS A4); sharp is inactive there.
- Porous momentum, moving scenes, phase change ([u] ≠ 0 breaks the model), contact lines touching walls
  (the fit may degrade there; census, §12 Q6).
- μ closures other than LinearMix in C (fallback to 'averaged' with a notice; §12 Q7).

---

## 2. Constraints and invariants

- **Sides.** A = the C = 1 phase (liquid by flow convention), B = C = 0.
  - μ_A = p0 + p1 and μ_B = p0 of the LinearMix closure; μ(f) = fμ_A + (1−f)μ_B.
  - R(f) = f/μ_A + (1−f)/μ_B, the series resistance per unit length.
  - "lo" and "hi" are the phases of smaller and larger μ; r = μ_lo/μ_hi.
- **Unknowns are face averages.** The MAC unknown u_c is the flux average over its face, because
  continuity is imposed on face fluxes. On a cut face the face average and the face-centre point value
  differ by O(h·[∂u]). **Every error in this note and in G-SV1 is therefore measured against exact face
  averages.** A point-value reference would make second order impossible for any consistent scheme. This
  clarifies the gate; it does not loosen it.
- **Link kernels** follow from that.
  - Normal link (∂_c u_c, cell i): the difference of two face averages is the cell average of ∂_c u_c,
    so its side-A fraction is exactly C_i.
  - Shear link ∂_a u_c at edge E: the two u_c faces are adjacent in a, in the c-face plane through E. Its
    kernel is a hat of half-width 1 cell in a times a box of 1 cell in b.
- **Signs, indexing, precision.**
  - Signs and indexing as the code: u_c(i) on the −c face of cell i; x-fastest; `st = {1, e.x, e.x·e.y}`.
  - Fractions are computed in index space. They are affine-invariant, so they are metric-free, like
    `plicVolume`.
  - Normals, strains and stresses are physical; `vofPhysNormal` gives n.
  - FP64 throughout.
- **Bitwise invariants.**
  - Uniform μ: unchanged; the skip rule of FVS §5 also skips every term here.
  - Non-VoF and variable-μ single-phase (`variable_mu`, `varmu_mpi`, `full_stress_operator`): untouched
    code path.
  - Away from the interface band, every operator entry and RHS term is bitwise equal to today's.
  - np 1/2/4 bitwise (local stencils, fixed summation order, no new reductions).
- **Implicit solve.** Each per-component 7-band stays symmetric and an M-matrix (RB-GS). Velocity MG
  stays refused under variable μ (register 212).
- **No new global synchronisation.** One extra neighbour halo exchange per step (§7).
- **Settled and kept.** Full deviatoric stress; explicit conservative MAC transpose on uⁿ (never product
  form, never on the Picard iterate); incremental rotational correction (281); ABC/MAC projection; no
  Rhie–Chow; no face-acceleration kick; mass–momentum-consistent VoF transport; no env vars; physical
  inputs only.

---

## 3. Diagnosis: why every average is first order

### 3.1 The local solution at a planar interface

Take a plane with normal n, continuous u, and L = ∇u (L_pq = ∂_p u_q). The jump conditions without
surface tension are:
- [u] = 0;
- [P σ n] = 0, where P = I − nn and σ = μ(L + Lᵀ);
- [−p + 2μD_nn] = 0.

Since u is continuous, all tangential derivatives are continuous, and so is D_nn = −∇_t·u_t. The only
jumping derivative is ∂ₙu_t:

  L_B = L_A + n⊗a,  a = (μ_A/μ_B − 1) P S_A n,  S = L + Lᵀ.

The local field splits into two modes.
- **Across (shear) mode.** The kink n⊗a carries the tangential traction τ = μ_k P S_k n, which is
  continuous. The stress tensor is continuous; for Couette u = U(ξ)t, σ = τ(n⊗t + t⊗n) on both sides.
- **Along (stretching) mode.** D_nn and D_tt' are continuous, the stress 2μ_kD jumps, and
  [p] = 2[μ]D_nn. The δ-part of ∇·σ is purely normal, 2[μ]D_nn n δ_I, and is balanced by the pressure.

### 3.2 Error mechanism

The momentum rows are finite-volume balances of face fluxes. A flux error δF on a face enters the two
CVs that share it with opposite signs, so it acts as a dipole of strength δF·h.
- A band of O(1) flux errors (any average on a cut face) is a dipole layer of density O(h). The velocity
  error is then O(h).
- Fluxes exact for the local piecewise-linear field leave O(h) flux errors and an O(h²) velocity error.

So second order needs **conservative fluxes that are exact for kinked-linear fields, to O(h)**.

A non-conservative ghost-fluid row, in which each row sees only its own side, is worse. Its O(h) ghost
errors do not cancel between the two rows of a face and act as monopoles. Measured: Poiseuille order
0.74–0.80, §3.4, row "gfm".

### 3.3 Why a scalar per sample cannot do it

Shear sample at edge E(a,c). The link differences of the exact kinked field, with T_ac = n_a τ_c and
"cont" the continuous part, are:

  m_ac = cont_ac + R̄_ac T_ac,  m_ca = cont_ca + R̄_ca T_ca.

The exact face-averaged stress is μ̄_face (cont_ac + cont_ca) + T_ac + T_ca. A scalar sample
μ_s (m_ac + m_ca) is therefore exact only if both of the following hold:
- μ_s = μ̄_face, which is the arithmetic mean over the face (along mode);
- μ_s R̄_ac = μ_s R̄_ca = 1, which is the harmonic mean over each link (across mode).

These coincide only when the interface is grid-aligned and the two link fractions are equal. The cell
normal samples have the same conflict between the arithmetic and the harmonic mean of C. Every average
therefore leaves an O(1) flux error on cut faces.

**The along mode and the pressure.** For u linear, any scheme gives F_c = Σ_a 2D_ac δ_a μ⁽ᵃᶜ⁾. For F to be
balanced by the cell pressure, F must be a discrete gradient. In 2-D that requires
(δ_x² − δ_y²) μ^node = 2 cot 2α δ_xδ_y μ^cell. A sampled step satisfies this only at α = 0° and 45°;
its Fourier symbol fails at k ~ π/h. No choice of per-sample viscosities balances it, including the
exact face fractions. A finite-volume-exact viscous flux needs the finite-volume-exact pressure force,
and the cell pressure does not provide it. §5.4(c) provides it.

### 3.4 Measured (2-D harness, `tests/study/sharp_viscous/`)

Setup:
- Steady two-phase Stokes on [0,1]²: MAC, direct solve (the steady state of predictor + projection).
- Planar interface at angle α and offset s·h.
- Exact face-averaged Dirichlet data on a ghost layer; error measured against exact face averages.
- N = 16, 32, 64, 128. Angles 0, 22.5, 45°; offsets 0, 3/8, 3/4; μ_A/μ_B = 1e-2 and 1e2.
- Cases: Couette (τ = 1); Poiseuille (body force t, τ₀ = 1); extension (u = ε(ξn − ηt) with the pressure
  jump); general (Poiseuille + extension).

Reading the table: "exact" means ≤ 2e-11 relative at every N. The orders are the minimum over angles and
offsets of the finest-pair L∞ order (the all-pairs minimum in brackets). L∞ errors are relative, at N = 128.

| scheme | Couette | Poiseuille | extension (22.5°) |
|---|---|---|---|
| arith (flow today: μ(C) cells, 4-cell mean edges) | 1.8e-2, order 0.96 (0.82) | 2.2e-2, 0.93 (0.75) | 1.9e-2, order 1.0, L1 1.1 |
| harm4 (symmetric 4-cell harmonic, FVS Q4 option) | 1.0e-2; exact at 0°, first order oblique | 1.2e-2, 0.94 (0.75) | 1.3e-2, order 1.0 |
| facef (PLIC-exact face fractions, arithmetic) | 7.6e-3, 0.92 (0.73) | 9.3e-3, 0.92 (0.73) | 5.4e-3, order 1.0 |
| linkh (per-link harmonic = GFM series) | exact | 2.6e-5, 1.89 (1.68) | 3.5e-2, order 1.0 |
| sharp τ-terms on arith base, no midsection | exact | 2.0e-4, 1.88 (1.65) | 3.4e-2, order 1.0 |
| gfm (τ-terms, row sees its own side's μ) | exact | 5.9e-3, **0.74 (0.46)** | exact |
| facef + midsection, no τ-terms | first order (0.92) | first order (0.92) | exact |
| **chosen: facef + τ-terms + midsection** | **exact (≤ 1.3e-11)** | **1.0e-4, 1.93 (1.71); L1 2.00** | **exact (≤ 9e-10)** |

The chosen scheme on the general case: 4.4e-5, order 1.93 (1.71), L1 2.00. At 0° and 45° the extension
case is exact for every scheme, by symmetry.

**OPEN — the full G-SV1-style 2-D sweep of the chosen scheme** (angles 0, 11.25, 22.5, 33.75, 45°;
8 offsets; μ ratios 1e-2, 1e2, 1e-3, 1e3; N = 16…128). It was running when this note was committed.
Results go to `bubble_column_perf/validation/highratio/apriori_sv/res_gsv1.json`; summarise with
`python summ.py res_gsv1.json`. Default: none needed, because 3-D G-SV1 is the gate. If any
Poiseuille least-squares order over 16–128 is below 1.8, return to the architect before WO-6.

---

## 4. Decision, and what was rejected

**Chosen:** the conservative sharp operator of §5. It has PLIC face-fraction μ̄ per row face, a
per-link traction correction with the link's own series resistance, and the midsection correction. Two
interface quantities feed it, τ̂ and D̂_nn, from one local least-squares fit per interfacial cell.
Everything beyond the implicit band is explicit on uⁿ.

Rejected:
- **Any single average** (arithmetic, harmonic, geometric, 4-cell, PLIC sub-cell μ). These are first
  order by §3.3 and measured so (§3.4). Basilisk's three averages agree with this: they spread by 1.56
  per halving, order 0.64. Sub-cell PLIC fractions are the right μ̄ for the along mode but not for the
  across mode.
- **A tensorial laminate viscosity** (harmonic across and arithmetic along, from the PLIC normal), whether
  as a cell tensor or as a symmetric "dissipation deficit".
  - On MAC, the samples of different components sit at different places and see different link
    fractions. A cell-centred tensor therefore has to interpolate strains across the kink, which gives
    O(1) errors.
  - The condensed energy form is SPD but its stationarity is not the traction condition unless every
    link of a sample shares one fraction. It is exact only on average.
  - It would have kept the FVS §6 energy proof, but it cannot be exact per sample, which G-SV1 needs.
- **Ghost-fluid with jump conditions** (each row sees its own side's μ; Kang–Fedkiw–Liu style). It is
  exact on both linear modes, but non-conservative: O(h) ghost errors become monopoles, giving first
  order (Poiseuille 0.74). Second order would need O(h²) ghost stresses, i.e. a second-order traction
  estimate with curvature terms. That is fragile at μ ratio ≥ 1e2, because the correction is amplified
  by μ_hi/μ_lo.
- **Per-link harmonic (GFM series, Liu–Fedkiw–Kang).** It is exact on shear and wrong on stretching.
  This is consistent with flow's scalar register entry, "never a two-point / series-resistance flux".
- **Two velocity fields in the band** (P2F-style, as for conjugate scalars). It would change projection,
  advection and halo, at a large cost for what a one-field correction achieves here.
- **Moving the normal-stress jump into a GFM pressure jump in the projection.** It would make the
  projection operator jump-dependent and touch the balanced-force CSF and MG. The midsection term
  (§5.4c) is the same finite-volume pressure force, written as a conservative stress correction.
- **The midsection term with a face-fraction cell β in the implicit band.** It is unstable: ρ(G) = 2.4–19.7
  at μ ratio 1e-3 for D ≥ 10, and 1.9 at 1e-2 (§6). The cell β stays μ(C).
- **The midsection term with D̂_nn weighted towards the high-μ side.** It did not cure that instability
  (ρ 7.9 at D = 100).
- **Keeping today's 4-cell edge β implicit with the new target.** It is mildly unstable, ρ(G) = 1.007 at
  1e-3 / 1e-4.

---

## 5. The operator

### 5.1 Samples, faces, links

At the edge E(a,c) (a ≠ c, b the third axis) of the u_c CV of cell i — the edge shared by cells
i, i−s_a, i−s_c, i−s_a−s_c (`face_props.hpp:99`) — there are four regions, in global index
coordinates with E at (a, c) = (e_a, e_c):

| region | plane | extent | used as |
|---|---|---|---|
| F_c(E): the u_c row's a-face | a = e_a | c ∈ [e_c−½, e_c+½] box, b ∈ [e_b, e_b+1] box | μ̄_c,E (implicit u_c band and its flux) |
| F_a(E): the u_a row's c-face | c = e_c | a ∈ [e_a−½, e_a+½] box, b box | μ̄_a,E |
| L_ac(E): link ∂_a u_c | c = e_c | a ∈ [e_a−1, e_a+1] **hat**, b box | θ̄_ac, R̄_ac = R(θ̄_ac) |
| L_ca(E): link ∂_c u_a | a = e_a | c ∈ [e_c−1, e_c+1] hat, b box | θ̄_ca, R̄_ca |

At cell i, for component c:
- the c-midsection fraction C̄⁽ᶜ⁾_i: plane c = i_c + ½, box × box;
- the normal link, with θ̄ = C_i.

### 5.2 Geometry: owner plane and fractions

- **Interfacial cell.** A cell is interfacial if the colour reconstruction gives it a plane, i.e. the same
  predicate `VofCurvature::reconstructPlanes` uses (`interfaceEps`).
  - Virtual plane: a pure cell with a face-neighbour of the opposite pure phase is also interfacial.
    Its plane is that face: m = ±e_axis, α at the face. When there are several such faces, take the
    first axis in the order x, y, z.
  - Without the virtual plane, a grid-aligned interface lying on a face would fall back to the 4-cell
    mean and fail G-SV1 at offset 0.
- **Owner of edge E.** Among the four cells of E that are interfacial, the owner is the one with the
  smallest |m·ξ_E − α|/|m|₂. Here ξ_E is the edge midpoint in that cell's local coordinates. Ties go to
  the lowest linear index.
  - E is a **band edge** iff it has an owner.
  - **τ̂ at an edge sample.** The validated rule is the arithmetic mean of τ̂ over the band cells among
    E's four cells, taken in fixed index order. Band cells are the interfacial cells and their
    3×3×3 neighbours, and the fit (§5.3) runs on all of them. This is the default.
  - **OPEN — owner-only rule.** The alternative is to take τ̂ from E's owner alone, which fits only
    interfacial cells and is cheaper. Its 2-D check (`owner.py`, N ≤ 64) was running at commit time.
    Adopt it only if that check shows Couette exact and Poiseuille order ≥ 1.85 at every ratio;
    otherwise keep the band mean.
  - A non-band edge keeps today's β and gets no correction, so it stays bitwise.
  - A cell sample is in the band iff its cell is interfacial; it is its own owner.
- **Fractions** come from the owner plane (m, α), shifted to the owner cell's local coordinates
  (`plicVolume` measures the side m·ξ < α). In each formula below, q is the in-plane hat or box axis, b
  the other in-plane axis, p the plane-normal axis with ξ_p its fixed coordinate, and q0, b0 the low
  ends of the region:
  - box × box: f = plicVolume(m_q, m_b, 0, α − m_p ξ_p − m_q q0 − m_b b0);
  - hat × box on q ∈ [q_c−1, q_c+1]: f = plicVolume(m_q, m_q, m_b, α − m_p ξ_p − m_q(q_c−1) − m_b b0).
    The hat is the convolution of two unit boxes, so it is a 3-D box with the q-component duplicated.
  - Midsection: f = plicVolume(m_a, m_b, 0, α − m_c(ξ_c of the midplane)).
  - These are thin compositions of `core::vof::plicVolume` and are exact for a plane. WO-1 checks them
    against brute-force quadrature.
- **Physical normal.** n = `vofPhysNormal(m)` of the owner, used for the kink model and the T-terms.

### 5.3 Interface state: τ̂ and D̂_nn (one least-squares fit per interfacial cell K)

- **Unknowns.** L ∈ ℝ^{3×3}, the gradient on the **low-μ side** at the interface near K.
- **Model, one row per link ℓ** (direction a_ℓ, component c_ℓ, low-side kernel fraction θ̄ˡᵒ_ℓ):

    m_ℓ = L_{a c} + (1 − θ̄ˡᵒ_ℓ) n_a (r − 1) (P e_c)·(S n),  S = L + Lᵀ,

  This is exact for any kinked-linear field, with n = n_K. It is linear in L: the coefficient of L_pq
  is δ_{ap}δ_{cq} + (1−θ̄ˡᵒ)(r−1) n_a [(Pe_c)_p n_q + (Pe_c)_q n_p].
- **Links used.**
  - All normal links of the 3×3×3 block around K: 81, with θ̄ = C_j.
  - Both shear links of every edge of every block cell: 3 orientations × 3·4·4 = 144 edges, 288 links.
    Their θ̄ come from each edge's own owner plane, as in §5.2.
  - Links with a node outside the physical domain are left out; wall and inflow data are used as data.
- **Solve.** Unit weights. Accumulate the 9×9 normal matrix AᵀA and Aᵀm in fixed link order, then
  Cholesky.
  - If the smallest pivot is below 1e-10·trace/9, the fit falls back: τ̂ = 0 and D̂_nn = 0, which locally
    reverts to "facef" (first order), and the census counts the cell.
- **Outputs.** τ̂_K = μ_lo P(S n) (tangential, 3 components) and D̂_nn,K = n·L n (continuous across the
  interface). Stored per interfacial cell; zero elsewhere.
- **Why the low side.** τ = μ_k P S_k n holds on both sides, but the correction multiplies τ by
  |1 − μ̄R̄| ~ θ(1−θ)(√(μ_hi/μ_lo) − √(μ_lo/μ_hi))² (≈ 20 at ratio 100). The error on side k is μ_k·O(h);
  only the low side keeps the product at the natural scale.
- **Required accuracy.** τ̂ and D̂_nn need only O(h) (first-order) accuracy: they enter conservative
  fluxes (§3.2). This is why a 9-unknown linear fit suffices and the scheme is robust at ratio 1e3.

### 5.4 Sample stresses (exact formulas)

All quantities below are physical. Link differences: m_ac = (u_c(·+s_a) − u_c(·))/h_a at the sample.

**(a) Edge E(a,c), for the u_c row** (swap the roles of μ̄ for the u_a row):

  σ⁽ᶜ⁾_E = μ̄_c,E (m_ac + m_ca) + n_a τ̂_c (1 − μ̄_c,E R̄_ac) + n_c τ̂_a (1 − μ̄_c,E R̄_ca)
  σ⁽ᵃ⁾_E = μ̄_a,E (m_ac + m_ca) + n_a τ̂_c (1 − μ̄_a,E R̄_ac) + n_c τ̂_a (1 − μ̄_a,E R̄_ca)

- μ̄_c,E = μ(f(F_c(E))) and μ̄_a,E = μ(f(F_a(E))); (n, τ̂) are those of the owner of E.
- **Implicit:** μ̄_c,E · m_ac, the u_c band coupling u_c across E. It is symmetric because both rows on
  the face use μ̄_c,E.
- **Explicit:** the rest — μ̄_c,E·m_ca (the transpose element, with this β) and both T-terms.
- Each value is shared by the two CVs on its face, so the scheme is conservative.

**(b) Cell i, for the u_c rows** at faces i and i+s_c:

  σ⁽ᶜ⁾_i = 2μ(C_i) m_cc + 2(μ_A − μ_B)(C̄⁽ᶜ⁾_i − C_i)(m_cc − D̂_nn,i) + 2 n_c τ̂_c,i (1 − μ(C̄⁽ᶜ⁾_i) R(C_i))

- **Implicit:** μ(C)·m_cc, today's band. **Explicit:** μ(C)·m_cc (today's transpose normal element) plus
  the last two terms.
- For pure cells, C̄ = C ∈ {0,1} and μ(C)R(C) = 1, so the correction vanishes exactly.
- Algebraically, the first two terms equal 2μ(C̄⁽ᶜ⁾)m_cc + 2(μ_B−μ_A)(C̄⁽ᶜ⁾−C)D̂_nn: the face-fraction
  normal stress plus the midsection force of the pressure jump J = 2(μ_B − μ_A)D̂_nn.

**(c) Why the midsection term.** Write the exact pressure as p = p_B − J H_A. Its finite-volume force on
the u_c CV uses the c-midsection averages, p̄⁽ᶜ⁾_i = p_i − J(C̄⁽ᶜ⁾_i − C_i), where p_i is the cell
average. With face-fraction stresses, the viscous flux of the along mode is finite-volume exact. Adding
J δ_c(C̄⁽ᶜ⁾ − C)/h turns the residual force into a discrete gradient of the cell field J·C. The pressure
then balances it exactly: measured extension error ≤ 9e-10 at all angles. The term is in divergence form
(a cell-centred stress), so an O(h) error in D̂_nn costs only O(h²) (§3.2).

### 5.5 Jump conditions enforced, and the order argument

- **Patch test.** For any planar interface and any piecewise-linear velocity satisfying [u] = 0,
  [Pσn] = 0 and [p] = 2[μ]D_nn:
  - the fit returns the exact τ and D_nn (the model is exact);
  - every sample stress equals the exact face-averaged traction;
  - the midsection term makes the pressure force finite-volume exact.
  So the exact face averages plus the cell-averaged pressure solve the discrete system. Measured: Couette
  and extension exact to round-off at all 12 angle/offset/ratio settings.
- **Smooth interface and flow.** The deviation from kinked-linear is O(h) in the gradients. The sources
  are the second derivatives, PLIC position O(h²), PLIC normal O(h) and curvature O(κh) in the fit. Each
  sample flux error is therefore O(h).
- Since the fluxes are conservative, the errors form a dipole layer of density O(h²), giving an O(h²)
  velocity error in the bulk and on the band. Measured: L∞ 1.93–1.95, L1 2.00.
- **Pressure.** The pressure carries the full jump, σκ + 2[μ]D_nn, smeared over one cell in balanced
  form. It is O(1) inside the band and second order outside; velocity accuracy does not depend on it.

### 5.6 Answers to the brief's three questions

1. **Must 2[μ]∂ₙuₙ move into the pressure jump?** Yes, in the discrete sense. No per-sample viscosity can
   make its force pressure-compatible on an oblique interface (§3.3). The midsection term is exactly the
   finite-volume pressure force of that jump. It is a stress correction, not an entry in the CSF array.
   The CSF surface tension is unchanged: it is already a cell-field gradient.
2. **What changes in the explicit transpose term?**
   - Same kernel, same placement, still once per step on uⁿ.
   - In the band, β is the component's own face fraction μ̄_c,E. Today it is the 4-cell mean, the same
     for all components. T is therefore no longer symmetric there.
   - The T-terms and the midsection term are added in the same pass.
   - The FVS §6 energy proof no longer covers the band; G-SV9 replaces it.
3. **Collocated:** §8.

---

## 6. Time treatment: the split and its stability

The predictor is (M + K)u* = (M + K − A)uⁿ + rhs, where:
- M = ρ_F/dt, with ρ_F as today (`face_props.hpp:70`);
- K is the implicit per-component band (cells μ(C), band edges μ̄_c,E, non-band edges unchanged);
- A is the full sharp operator.

Projection and rotational correction are unchanged. The steady state is A's, independent of dt.

Measured in 2-D (`stab.py`, `stab5.py`; dense spectrum of G = P(M+K)⁻¹(M+K−A), the M-orthogonal MAC
projection P, homogeneous Dirichlet, N = 16, planar interfaces (0°, 0.6), (0°, 0.2), (22.5°, 0.0),
(22.5°, 0.3), (45°, 0.3), (67.5°, 0.5)). Each cell is the worst ρ(G) over the six geometries.

| μ_A/μ_B, ρ_A/ρ_B | K = μ(C) cells + face-fraction edges (chosen) | K = face fractions everywhere | K = today's arith |
|---|---|---|---|
| 1e-3, 1e-4 | 0.987 / 0.950 / 0.939 / 0.937 / 0.937 | 0.986 / 0.883 / **2.45 / 12.0 / 19.7** | 1.000 / **1.007 / 1.007 / 1.006 / 1.005** |
| 1e-2, 1e-3 | 0.986 / 0.882 / 0.707 / 0.703 / 0.703 | 0.986 / 0.882 / **1.005 / 1.77 / 1.91** | 0.986 / 0.903 / 0.898 / 0.897 / 0.897 |
| 1e3, 1e4 | 0.995 / 0.989 / 0.987 / 0.986 / 0.986 | 0.994 / 0.950 / **2.15 / 7.39 / 9.81** | 0.999 / **1.002 / 1.003 / 1.002 / 1.002** |

Columns are D_hi = μ_hi dt/(ρ_hi h²) = 0.1 / 1 / 10 / 100 / 1e4. At N = 32 the chosen split was run on five geometries: (0°, 0.6), (11.25°, 0.1), (22.5°, 0.3),
(33.75°, 0.85), (45°, 0.5). The worst ρ(G) at D_hi = 0.3 / 3 / 100 / 1e5 was:
- 1e-3 / 1e-4: 0.989 / 0.975 / 0.974 / 0.974;
- 1e-2 / 1e-3: 0.989 / 0.904 / 0.817 / 0.817;
- 1e2 / 1e3: 0.990 / 0.910 / 0.802 / 0.802;
- 1e3 / 1e4: 0.990 / 0.954 / 0.950 / 0.950.

Stable everywhere, and the margin does not shrink with N in a way that suggests a grid-scale mode. The
values at D ≈ 0.3 are the slowest physical decay.

- **The instability is the midsection term.** The τ-terms alone are stable at every D (ρ ≤ 0.56).
  - With a face-fraction cell β, the D̂_nn force reaches the gas rows explicitly, at liquid-viscosity
    strength.
  - With μ(C) in the cell band, the explicit remainder is 2(μ_A−μ_B)(C̄−C)(m_cc − D̂_nn). It is a strain
    *difference* and vanishes for an aligned interface.
- **Transient accuracy.** Ramped start-up, 2-D, N = 24, 22.5°: the deviation between split and fully
  implicit is the same class as today's explicit transpose (`transient2.py`, D_hi = 0.1 / 0.3 / 1):
  - ratio 1e-2: sharp 2.5e-2 / 5.4e-2 / 1.2e-1, against today's arith 2.6e-2 / 7.3e-2 / 1.8e-1;
  - ratio 1e-3: sharp 2.5e-2 / 5.9e-2 / 1.4e-1, against 5.3e-2 / 1.5e-1 / 3.7e-1;
  - ratio 1e2: sharp 1.5e-1 / 2.3e-1 / 2.9e-1, against 1.1e-1 / 2.2e-1 / 3.6e-1.
  P1 runs at D_l ≈ 0.10 (d/h 16) to 0.32 (d/h 48).
- **No time-step limit is introduced.** If the 3-D G-SV9 finds one, stop and return to the architect.
  Do not add a cap.

---

## 7. MPI, halo, device, cost

- **Planes.**
  - `VofCurvature::reconstructPlanes` writes planes of the current C on the colour block's inner region
    grown by `kPvHalf = 2`, from C ghosts of depth 3. Edge owners at distance ≤ 2 from an owned cell are
    therefore available without communication. They are bitwise equal across ranks because they come
    from identical ghost data.
  - Today these planes exist only when `csfActive()` (`flow_ibm_vof.hpp:2193`). With σ = 0 and sharp
    active, the step must call the plane reconstruction itself, once per step after `updateProperties`.
  - The solver kernels read planes on the colour block (g = 3). They use the same index mapping the CSF
    face force uses.
- **Fit.**
  - It runs on owned interfacial cells. Its reach is 2 cells for C, planes and velocities: 3×3×3 block
    links plus edge owners. That fits velocity G = 2 with face edges, which the transpose already fills
    (`fillVelGhostsTo(…, faceEdges=true)`).
  - **One new halo exchange per step:** τ̂ (3) and D̂_nn (1), depth 1. An edge sample on the rank
    boundary may have its owner in the first ghost layer. Computing the fit there redundantly would need
    velocity depth 3.
  - No reductions, no global synchronisation. The fixed accumulation order keeps it np-bitwise.
- **Kernels.**
  - All use `ccFor3` / `MDRange3<CCExec>`, x-fastest, with an early-out on non-band cells. The curvature
    worklist (`listI_`) may be reused for the fit; the default is the dense predicate.
  - The 9×9 Cholesky is in registers; spills are acceptable on a band of O(N²) cells.
  - Fractions are recomputed on the fly (a few `plicVolume` calls each) rather than stored.
- **Storage.** τ̂ and D̂_nn as four `SField`s on the solver block: 32 B/cell, 28 MB at d/h 24 and 226 MB
  at d/h 48. §12 Q2 covers the fallback to worklist storage.
- **Cost estimate.**
  - Per interfacial cell, about 370 rows × 55 FMA ≈ 2e4 FMA.
  - At d/h 24 there are ≈ 3e3 interfacial cells per bubble, so ≈ 6e7 FMA per step: under 1 ms on GPU,
    a few ms on 16 host threads.
  - Plus 1 exchange of 4 fields, and the band terms in `addFullStressRhs` and `rebuildStencils`.
  - Expected well inside G-SV7 (≤ 1.2×).

---

## 8. Collocated carry-over (the user's next air–water target)

The principles carry over:
- face-fraction μ̄ for the along mode;
- per-link τ-correction with the link's own series resistance for the across mode;
- the finite-volume-exact pressure force of 2[μ]D_nn.

**The formulas do not carry over.** Do not port them mechanically.
1. **Kernels.** Collocated unknowns are cell averages. A face link between two cell averages has kernel
   (box∗box)⊗box⊗box, i.e. a hat in a times boxes in b and c, so θ̄ is a "4-D" PLIC integral. It is
   computable (the integral of `plicVolume` over one shift, or duplicating a component twice), but it is
   new.
2. **No full stress yet.** Collocated has no full stress at all: FVS §9 deferred it, and its stability
   pairing does not hold there. The sharp layer must come after it.
3. **Pressure.** The collocated pressure acts through the ABC projection on the face field and the
   cell-force channel (`doc/collocated_varrho_forces.md`). The midsection argument must be redone for
   that pressure gradient. Corrections must stay inside the predictor; the "kick" is forbidden.

Recommended order: collocated full stress (FVS §9); then a collocated variant of this 2-D harness; then
the design. Default until then: collocated stays averaged.

---

## 9. API and defaults

- **Setter (tier and name are the user's call, §12 Q3).**
  - Proposed: `s.diagnostics.set_vof_viscous_interface("sharp" | "averaged")` while it is being built.
  - On promotion: AUTO resolves to "sharp" when VoF is on, μ = LinearMix(C), full stress is active and
    neither phase change nor porous momentum is on.
  - "averaged" reproduces today's numbers bitwise.
- **Refusals and notices.**
  - harmonic=True with sharp: raises, since sharp replaces it.
  - Non-LinearMix μ, or phase change: AUTO falls back to averaged with a one-time stderr notice; an
    explicit "sharp" raises.
  - Solids: full stress is inactive under AUTO, so sharp is too.
- **Census.** `diagnostics.vof_viscous_interface_census()` returns the band-cell count, virtual-plane
  cells and fit fallbacks for the last step.

---

## 10. Work orders (for an opus-implementer; in dependency order)

**WO-0 (done with this note).** The 2-D harness under `tests/study/sharp_viscous/`.
- Acceptance: `python sweep.py …` reproduces §3.4.

**WO-1. Region fractions** (container-free; in `core`).
- `core/include/peclet/core/vof/plic.hpp`, or a sibling `plic_regions.hpp`, per that file's own rule
  ("new container-free VoF math belongs in core"):
  - `plicFaceFraction(m, α, axis, ξ_p, q0, b0)`, box × box;
  - `plicHatBoxFraction(m, α, plane_axis, ξ_p, hat_axis, q_c, b0)`;
  - `plicMidsectionFraction(m, α, c)`.
- flow builds against the sibling core headers on the branch. At landing, core is tagged first (suite
  rule).
- Acceptance:
  - each fraction matches brute-force quadrature (2e6 samples, |Δ| ≤ 3e-3 statistical, and exact
    identities below) over 10⁴ random planes;
  - box × box reproduces `plicSlabVolume` thin-slab limits;
  - hat × box with m_b = 0 reproduces the 1-D hat CDF to 1e-14;
  - `plicVolume(m_K, α_K) = C_K` (round-off) on reconstructed planes.

**WO-2. Band geometry and fit** (flow `src/vof/sharp_viscous.hpp`, new).
- Interfacial predicate, virtual planes, edge owner, per-sample fractions, the τ̂/D̂_nn fit, the census.
- Host oracle in the test.
- Acceptance:
  - 3-D extrusion of the 2-D harness cases (n_z = 0, z-independent fields, exact planes injected).
    τ̂, D̂_nn and every fraction equal the harness values to 1e-12 relative.
  - Exact τ and D_nn for kinked-linear fields at 9 orientations, to ≤ 1e-11.
  - Fallback count 0 on all of them.

**WO-3. Implicit band.** `FaceProps::beta` / `ibmBuildDiffusionVar` take μ̄_c,E on band edges under
sharp mode; cells unchanged.
- Acceptance:
  - every non-band entry bitwise equal to 'averaged';
  - band entries equal to the oracle;
  - per-component symmetry and the M-matrix property asserted.

**WO-4. Explicit kernel.** Extend `addFullStressRhs` (or a fused sibling): band transpose with μ̄_c,E,
T-terms, midsection term, all on `old_`.
- Express each explicit σ as an element in the kernel's existing metric convention (`wc`, `rs`).
- Acceptance:
  - host-oracle match ≤ 1e-13;
  - extended `full_stress_operator` subtests pass unchanged on non-VoF μ;
  - new subtest: the operator applied to exact face averages of 3-D kinked-linear fields (Couette and
    extension at 9 orientations) gives a residual ≤ 1e-11 relative to the band stress scale.

**WO-5. Step integration.**
- Plane reconstruction when σ = 0; ordering after `updateProperties` and before `rebuildStencils`.
- The τ̂/D̂ exchange; skip rules; the setter, refusals and census.
- Acceptance:
  - G-SV5 (uniform μ bitwise; 'averaged' bitwise; non-VoF digests unchanged);
  - G-SV6 np 1/2/4 bitwise, host and CUDA.

**WO-6. Operator gates in 3-D.** G-SV1, G-SV1c, G-SV1d, G-SV9 (§11) as ctests or `tests/study` scripts
with one-line summaries.

**WO-7. Physics gates.** G-SV2 (Hadamard–Rybczynski), G-SV3 (P1), G-SV4 (P2), G-SV7 (cost), G-SV8
(robustness), G-SV10 (census). Local GPU only under
`flock /home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock`.

**WO-8. Docs.**
- flow `CLAUDE.md` velocity-solve paragraph and register entry (§13), plus CHANGELOG.
- AUTO flip as its own commit after the gates, on the user's recorded OK (§12 Q4), so it reverts
  cleanly.

Escalate to the architect, rather than tune, if any of these happens:
- G-SV9 finds ρ > 1;
- G-SV1 fails only near curvature or at virtual planes;
- G-SV3 orders fall between 1 and 1.5.

---

## 11. Verification gates

The implementer judges PASS / MISS / NOT JUDGED and never retunes. "Order" means the least-squares
slope of log e against log h over the stated resolutions.

**Pre-registered (brief §8), with measurement definitions; none loosened.**

- **G-SV1 — rotated Couette and Poiseuille, 3-D production operator.**
  - Normal n = (cos β cos γ, cos β sin γ, sin β), β, γ ∈ {0, 22.5°, 45°}, giving 9 orientations.
  - Offsets k/8 cell along n, k = 0…7. μ ratio 1e-2 and 1e2.
  - Box [0,1]³ with exact face-averaged data on the ghost layer. Steady state by the production stepper
    at D_hi = 100, to a velocity-increment ≤ 1e-12·max|u|.
  - N = 16, 24, 32, 48. Error = max |u_h − ū_face| / max|ū_face|.
  - PASS: order ≥ 1.8 for every orientation × offset × ratio. Couette passes as "exact" if
    e ≤ 1e-10 at every N.
  - The 2-D a-priori version is in §3.4.
- **G-SV2 — Hadamard–Rybczynski, κ = 1e-2, 1, 1e2.**
  - Steady Stokes: advection off, colour frozen. Periodic box L = 8R; drop driven by its buoyancy Δρg.
  - K = F/(6πμ_out U_d R), where U_d is the drop's mean velocity relative to the box mean.
  - d/h = 8, 12, 16, 24, 32. "Drag error order" is the self-convergence order: fit K(h) = K∞ + C h^p
    over all five levels and require p ≥ 1.8.
  - Sanity check (added, not a substitute): K∞ within 3 % of the HR value
    (2+3κ)/(3(1+κ)) / (1 − 1.7601·(2+3κ)/(3(1+κ))·φ^{1/3}). This is the leading periodic correction with
    the Stokeslet weight; its O(φ) remainder is about 1 % at L = 8R.
- **G-SV3 — P1 from rest to t = 60, d/h 16/24/32/48.** Observed order ≥ 1.5, and
  |Re(32) − Re_ext|/Re_ext ≤ 0.5 % (Re_ext from the three finest).
- **G-SV4 — P2, d/h 16–48.** χ trends towards Basilisk L8 0.716, not away. |ΔRe| against 'averaged' is
  reported.
- **G-SV5.** Uniform μ bitwise; non-VoF `state_hash` digests unchanged; full battery green on host and
  CUDA. 'averaged' bitwise equal to today.
- **G-SV6.** np 1/2/4 bitwise, host and CUDA. Sharp on, P1 d/h 16 for 20 steps, plus the G-SV1
  configuration at N = 24.
- **G-SV7.** P1 d/h 24 ms/step ≤ 1.2 × 'averaged' (same build, same GPU).
- **G-SV8.** P1 at ρ ratio 1e-4, μ ratio 1e-3 runs 10 t-units without blow-up.

**Added.**

- **G-SV1c (along mode).** Same configurations as G-SV1 with u = ε(ξn − ηt₁) and the exact pressure jump.
  Error ≤ 1e-9 at every N. Poiseuille + extension: order ≥ 1.8.
- **G-SV1d.** G-SV1 at μ ratio 1e-3 and 1e3 (the G-SV8 contrast). Same thresholds.
- **G-SV9 (stability; replaces the FVS §6 proof in the band).**
  - (a) 2-D harness: ρ(G) ≤ 1 + 1e-12 for the table in §6 at N = 16 and 32.
  - (b) 3-D homogeneous march, N = 16.
    - Random solenoidal initial field; plane at n(22.5°, 22.5°), offset 0.3.
    - μ ratio 1e-2, 1e2, 1e-3, 1e3 with ρ ratio 1e-3, 1e3, 1e-4, 1e4.
    - D_hi ∈ {0.1, 1, 10, 100, 1e4}; 2000 steps.
    - PASS: kinetic energy ≤ initial, and the growth ratio over any 100 steps after step 200 is
      ≤ 1 + 1e-10.
  - (c) No blow-up in G-SV3 or G-SV8.
- **G-SV10 (census).** P1 d/h 24: fit fallbacks ≤ 0.1 % of interfacial cells per step, averaged over the
  run. Virtual-plane cells are reported.
- **G-SV11 (single phase stays put).** `variable_mu`, `varmu_mpi`, `full_stress_operator{,_np2,_np4}`,
  `collocated_stability_guard` and `amr_stability_guard` pass bitwise.

---

## 12. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1 | 3-D stability margin (2-D measured; 3-D adds a third shear plane) | **fact** (G-SV9b) | Proceed. If ρ > 1 anywhere, stop and return to the architect; do not add a dt cap. |
| Q2 | Memory for 4 extra fields at d/h ≥ 48 on 16 GB | **fact** (measure free memory at P1 d/h 48) | Dense fields. Switch to worklist storage if they exceed 10 % of free device memory. |
| Q3 | Name and tier of the setter | **user preference** (docs/NAMING.md) | `diagnostics.set_vof_viscous_interface("sharp"\|"averaged")` until the gates pass. |
| Q4 | AUTO = sharp after the gates | **user preference** | Yes: the user named it priority. Make it its own commit and register entry. |
| Q5 | G-SV2 reference in a periodic box | **fact** | Gate on the self-convergence order. The HR periodic-corrected value is a 3 % sanity check. |
| Q6 | Interfaces touching walls or contact lines (fit links cut by the wall) | **fact** (census near walls) | Drop links outside the domain. Fallback to 'facef' if the fit is singular. Report in the census. |
| Q7 | Other μ closures (Table1D in C, temperature-dependent) | **user preference** (scope) | Not in v1. AUTO falls back to averaged with a notice. |
| Q8 | Thin films, under 1 cell of one phase | **fact** (census) | The fit is well-posed while every link type appears. Otherwise fallback; τ̂ = 0 is first order locally. |
| Q9 | Plane source when σ = 0 | **fact** (does VofCurvature exist with σ = 0?) | Construct and reconstruct it in the step when sharp is active. |
| Q10 | Curved interface: one plane per fit block (O(κh) model error) | **fact** (G-SV2 order) | Accept. If G-SV2 misses only at d/h 8–12, report it; do not add curvature terms without the architect. |
| Q12 | τ̂ at edges: band mean (validated) or owner only (cheaper) | **fact** (`owner.py`) | Band mean. The fit runs on interfacial cells and their 3×3×3 neighbours. |
| Q11 | Collocated | **user preference** (sequencing) | §8 order: full stress on collocated first; this design is not ported until then. |

---

## 13. Register entry (text for `docs/DECISIONS.md` and `docs/decisions/flow.md`)

**Decision.** The VoF interface viscous stress on the staggered MAC grid is the *sharp conservative*
operator. It applies when VoF is on, μ = LinearMix(C) and full stress is active. It has three parts:
- μ̄ on each band sample from the PLIC area fraction of that row's own face;
- a per-link interface-traction correction n_a τ̂_c(1 − μ̄R̄_ac), with the link's own series resistance;
- the midsection correction 2(μ_A−μ_B)(C̄⁽ᶜ⁾−C)(∂_cu_c − D̂_nn), which is the FV-exact pressure force of
  the viscous normal-stress jump.

τ̂ and D̂_nn come from one 9-unknown kinked LS fit per interfacial cell, on the low-μ side. The implicit
band changes only on band edges (face-fraction β); cells keep μ(C). Everything else is explicit on uⁿ.

**Rejected:**
- every single average: arithmetic, harmonic, geometric, 4-cell, PLIC sub-cell. First order on oblique
  interfaces: L∞ 0.73–0.96 measured.
- per-link harmonic / GFM series (Liu–Fedkiw–Kang): exact on shear, first order on stretching.
- non-conservative ghost-fluid with own-side μ: Poiseuille order 0.74.
- tensorial laminate viscosity: exact only on average.
- a GFM pressure jump in the projection.
- the midsection term with a face-fraction cell β: ρ(G) up to 19.7.
- today's 4-cell edge β with the new target: ρ(G) 1.007.

**Reverses, in the VoF band only:** the premise of register 272 (FVS §6) that T is symmetric because β is
component-independent. β is now component-dependent there and T is not symmetric. Stability is
established by the measured spectral gate G-SV9, not by the energy proof. Register 272's choices (full
stress, explicit conservative transpose on uⁿ, no product form, no Picard placement) stand.

**Why:** 2-D a-priori, N = 16–128, 0–45°, all offsets, μ ratio 1e-3–1e3:
- Couette and extension exact to round-off;
- Poiseuille and general L∞ order 1.93–1.95, L1 2.00;
- ρ(G) ≤ 0.99 at every D.

Date 2026-10-31. Evidence: `doc/vof_sharp_viscous_stress.md` §3.4 and §6; `tests/study/sharp_viscous/`.

---

## Appendix A — the a-priori harness

`tests/study/sharp_viscous/` (pure Python + SciPy, host):
- `mac2d.py`: the 2-D MAC Stokes builder; every scheme of §3.4; exact face-averaged references.
- `sweep.py` + `summ.py`: the convergence sweeps.
- `stab.py`, `stab5.py`, `stab6.py`: the amplification spectra.
- `transient2.py`: the split-versus-implicit start-up.

To rerun, activate the suite venv, then:

```
python sweep.py <tag> couette,poiseuille,extension,general 0,22.5,45 0,0.375,0.75 0.01,100 \
       sharp:facef+J,arith 16,32,64,128 && python summ.py res_<tag>.json
```

The harness uses exact geometry (analytic planes). The production operator replaces it with PLIC planes
(§5.2); WO-2 ties the two together to 1e-12.
