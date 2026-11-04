# Sharp viscous stress at a VoF interface on the staggered MAC grid — design note

Status: DESIGN (architect, 2026-10-31; amended the same day by a second pass, §5.7; third pass §6.1–§6.4; fourth pass 2026-11-01, §6.5–§6.7). Branch `sharpvisc` (on `mom2`). USER 2026-10-31: "Yes give the
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
- **Follow-up rulings (§5.7, second architect pass).**
  - τ̂ at an edge is the mean over the edge's **interfacial** cells, and the fit runs on interfacial cells
    only. Measured like for like, the averaging rule moves the constant by ≤ 16 % and never the order.
  - The virtual-plane tie goes to the low face. A 1-cell film gets planes, not a fallback.
  - The 33.75° shortfall is pre-asymptotic. WO-6 proceeds with G-SV1 at N = 24–64. Whether G-SV1 is
    judged per offset or on the offset envelope is the user's call (§12 Q13).
- **Third-pass rulings (§6.1–§6.4, WO-5b and WO-6s, §11, §12 Q14–Q19).**
  - *The G-SV9(b) blow-up is not a 3-D mode.* §6's spectra kept the whole cell normal stress 2μ(C)m_cc
    implicit. The code runs today's split, μ(C) implicit and μ(C) explicit. On that split the 2-D harness is
    unstable too (worst ρ 1.135), and the production step agrees with it to ≤ 0.2 % at N = 16.
  - *Fix (WO-5b).* Interfacial cells take their normal stress fully implicit; A is unchanged.
    Harness worst ρ 0.9859.
  - *The film blow-ups (G-SV9(d)) were the pressure solve.* The gate forced MG-PCG at density ratio ≥ 1e3.
    With the Chebyshev driver, both paths are stable on the film (ρ ≤ 0.72).
  - *G-SV1 goes triply periodic (USER flag).* The wall-contact error is out of scope, conditional on
    G-SV1W.
  - *Steady states by GMRES on the production step (WO-6s).*
- **Seventh-pass ruling (§5.8, WO-9, G-SV12, §12 Q31–Q34).** Edge-owner ties are resolved by averaging
  the edge outputs over the *owner set*: every interfacial cell within τ_own = 1e-6 cells of the
  closest plane. No single-owner tie-break can be reflection-equivariant at a symmetric colour field,
  and round-off decided the near ties: `mirror_symmetry` 3e-5–8e-5, now required ≤ 1e-12. Tie-free
  edges are bitwise unchanged.
- **Eighth-pass ruling (§5.9).** There is no void between tie spread and generic gaps, so a sharp set
  edge at τ_own can itself be switched by round-off. The weights become a plateau (w = 1 for g ≤ 1e-6)
  followed by a C¹ smoothstep ramp to 0 at 1e-5, which makes the operator continuous in the distances.
  G-SV12 study rows are judged on bands only; the census keys are INFO.

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

**Full G-SV1-style 2-D sweep of the chosen scheme** (angles 0, 11.25, 22.5, 33.75, 45°; 8 offsets;
μ ratios 1e-2, 1e2, 1e-3, 1e3; N = 16…128; `res_gsv1.json`):
- Couette and extension: exact at every setting (≤ 8e-9).
- Poiseuille and general, least-squares L∞ order over 16–128: worst 1.79 at ratio 1e-2, 1.76 at 1e2,
  1.74 at 1e-3 and 1.69 at 1e3. Seven of 160 Poiseuille cases fall below 1.8, all at 33.75°. Errors at
  N = 128 are ≤ 7.4e-5. L1 order is 1.75–1.92.

**CLOSED 2026-10-31 — the 33.75° shortfall (§5.7 Q-C).** It was either pre-asymptotic oscillation with
the offset or an O(h) leak.
- The pre-registered test: N = 256, with "128→256 order ≥ 1.9 ⇒ pre-asymptotic, G-SV1 then uses
  N = 24, 32, 48, 64". One case reached only 1.86, so the question came back to the architect.
- N = 512 settles it. The worst case's 256→512 order is 1.99, and e·N² saturates.
- Ruling: pre-asymptotic. The registered consequence applies.
- The earlier remark that owner-only τ̂ (`owner.py`, 32→64, 1.70–1.82) is worse than the band mean was
  a confounded comparison. Like for like, the two rules have the same order (§5.7 Q-A).

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
    first axis in the order x, y, z and, on that axis, the −a face before the +a face. §5.7 Q-B gives
    the exact rule, which also covers the 1-cell film.
  - Without the virtual plane, a grid-aligned interface lying on a face would fall back to the 4-cell
    mean and fail G-SV1 at offset 0.
- **Owner set of edge E (seventh pass, §5.8).** For each of the four cells of E that is interfacial,
  d = |m·ξ_E − α|/|m|₂, with ξ_E the edge midpoint in that cell's local coordinates. The owner set
  T_E is every interfacial cell of E with d ≤ d_min + τ_own, τ_own = 1e-6 (index units, absolute).
  Every per-edge output below (face fractions, link fractions, the kink normal inside the T-term) is
  evaluated with each member of T_E as "the owner" and **averaged over T_E** (§5.8 gives the exact
  formulas; weighted since the eighth pass, plateau 1e-6 then a C¹ ramp to 1e-5, §5.9). Wherever §5 says "the owner", read "each member of T_E, then the mean". A tie-free edge
  (|T_E| = 1) is the single-owner operator bit for bit.
  - Superseded (seventh pass): "the owner is the one with the smallest distance; ties go to the
    first cell in the fixed offset order". No single-owner tie-break can be reflection-equivariant at
    a symmetric colour field, and round-off decided the near ties (§5.8, `mirror_symmetry` 3e-5–8e-5).
  - E is a **band edge** iff it has an owner, i.e. iff T_E is non-empty (some cell of E is
    interfacial). The predicate is unchanged.
  - **τ̂ at an edge sample (§5.7 Q-A).** τ̂_E is the arithmetic mean of τ̂_K over the **interfacial**
    cells K among E's four cells, summed in the fixed offset order. The set is never empty, because E's
    owner is in it.
    - A fallback cell (§5.3, τ̂_K = 0) stays in the set (as implemented, review F4, 2026-11-04): an
      edge whose interfacial cells are {fallback, fitted} runs its τ-term at the fitted cell's
      weight only, between 'sharp' and "facef". Excluding fallbacks would need the fallback flag in
      the halo; not done while the census reads 0 fallbacks on every gated case (G-SV10).
    - The fit (§5.3) runs on interfacial cells only, with n = n_K.
    - Superseded: the mean over all band cells (interfacial cells plus their 3×3×3 neighbours). It needs
      fits on cells that have no plane and gives the same order.
  - Not used: owner only (τ̂ from E's owner alone). Like for like it has the same order and a constant up
    to 16 % larger (§5.7).
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
  - **A region lying IN its owner's plane** (a face F_c(E) coincident with a virtual plane, α′ = 0
    exactly) is a tie in `plicVolume` (degenerate normal: `alpha > 0 ? 1 : 0`), so it reads
    f = 0 and μ̄ = μ_B = μ(C = 0), whichever phase is low (review F5, 2026-11-04). This is the rule,
    not an accident: σ does not depend on μ̄ for a field the kink model represents exactly (the
    offset-0 patch test passes both labellings), and with C = 1 the low phase the implicit flux and
    the explicit T-term on that face, each ~μ_hi/(2μ_lo) times the traction, cancel; G-SV9(d)/(e)
    run the film at both ratio signs.
- **Physical normal.** n = `vofPhysNormal(m)` of the owner, used for the kink model and the T-terms.
- At an edge whose owner set has several members, the fractions and the normal above are formed
  per member and averaged as §5.8 specifies (the T-term as a per-member product, never from averaged
  fractions).

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
    An edge lying ON a wall is kept (`edgeInDomain` is inclusive at k_a = dlo/dhi, review F7,
    2026-11-04): its link ∂_a u_c then has one node in the ghost beyond the wall (the BC mirror, a
    half-cell one-sided gradient), which the fit takes as a hat average over [−1, 1] with θ̄ from
    the owner plane extended past the wall. Exact for a linear field; it matters only at a contact
    line, which is out of scope (Q3/Q6, G-SV1W passes).
- **Solve.** Unit weights. Accumulate the 9×9 normal matrix AᵀA and Aᵀm in fixed link order, then
  Cholesky.
  - If the smallest pivot is below 1e-10·trace/9, or the system is not finite, the fit falls back:
    τ̂ = 0 and D̂_nn = 0, which locally moves toward "facef" (first order; τ̂ = 0 still enters the
    edge mean, §5.2), and the census counts the cell. Pinned by `sharp_viscous` testFallback.
- **Outputs.** τ̂_K = μ_lo P(S n) (tangential, 3 components) and D̂_nn,K = n·L n (continuous across the
  interface). Stored per interfacial cell; zero elsewhere.
- **Why the low side.** τ = μ_k P S_k n holds on both sides, but the correction multiplies τ by
  |1 − μ̄R̄| ~ θ(1−θ)(√(μ_hi/μ_lo) − √(μ_lo/μ_hi))² (≈ 20 at ratio 100). The error on side k is μ_k·O(h);
  only the low side keeps the product at the natural scale.
- **Required accuracy.** τ̂ and D̂_nn need only O(h) (first-order) accuracy: they enter conservative
  fluxes (§3.2). This is why a 9-unknown linear fit suffices and the scheme is robust at ratio 1e3.
  - This was measured, not only argued (§5.7 Q-A). Three averaging rules at the edge (owner only,
    interfacial mean, band mean) and three fit normals on a curved interface all give the same order.
  - The constants differ by ≤ 16 %. No centring of the O(h) part is needed.

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

- **Implicit (third pass, §6.1):** on an interfacial cell, the whole 2μ(C)·m_cc. **Explicit** there: the last
  two terms only. Every other cell keeps today's split: μ(C)·m_cc implicit, μ(C)·m_cc explicit (the
  transpose normal element). The operator is the same either way; only the split moves.
  - Superseded: today's split on band cells too, which is what WO-1…5 implemented. G-SV9(b) found it
    unstable at D_hi ≥ 100 (§6.1).
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

### 5.7 Follow-up rulings (architect, second pass, 2026-10-31)

**Evidence.** Two study scripts back these rulings.
- `tests/study/sharp_viscous/mac2dc.py` is mac2d with one PLIC plane per mixed cell, edge owners as
  §5.2, and a selectable fit normal. Its planar path reproduces mac2d bitwise.
- `followup_rules.py` runs experiments E1–E3. Results are in `bubble_column_perf/validation/highratio/
  apriori_sv/res_followup_{E1,arch2}.json`.

The quantities reported are:
- **per-offset LS**: the G-SV1-style least-squares L∞ order of each case;
- **envelope**: the maximum error over the offsets at each N, and its LS order.

**Fixed offset order.** Every summation order in §5.2 and here enumerates cells by their
offset from the edge or cell, lexicographic in (dz, dy, dx), x fastest. Never use the global linear
index.
- **Amended (seventh pass, §5.8): no owner tie is broken by this order any more.** An owner tie is
  resolved by the mean over the owner set; the offset order only fixes the order of that sum (and of
  the τ̂ mean). For two members the sum is order-free (IEEE addition commutes); for three or four
  the order moves the result at round-off only. The virtual-plane face choice (Q-B) is a separate
  rule and is unchanged.
- Superseded: "every tie … enumerates cells by their offset". A fixed tie order is translation-
  invariant (which is why it was chosen, for np-independence) but not reflection-equivariant.
- Away from a periodic seam the offset order is the lowest-linear-index order.
- Across a seam it stays translation-invariant, and the wrapped global index does not. Serial and
  np > 1 then agree wherever a ghost image takes part.

**Q-A — the fit normal on non-interfacial band cells.** Decided: there are no such fits.
- **Rule.** τ̂ at a band edge is the mean over the interfacial cells among the edge's four cells (the
  brief's candidate B).
- **Why.** The rule removes the question, and it is what §5.3 ("stored per interfacial cell") and §7
  ("runs on owned interfacial cells") already describe.
- **Settled content kept.** The settled point "a mean, not the owner alone" stands. Only the set
  averaged changes.

*E1, planar, like for like.* Poiseuille at 11.25, 22.5 and 33.75°, 8 offsets, N = 16…128. Each entry
gives μ ratio 1e-2 / 1e3.

| τ̂ at E | e(128), worst | per-offset LS min | envelope LS |
|---|---|---|---|
| band mean (old §5.2 rule) | 2.99e-5 / 7.55e-5 | 1.79 / 1.69 | 1.98 / 1.93 |
| **interfacial mean (chosen)** | 3.30e-5 / 7.43e-5 | 1.80 / 1.69 | 1.97 / 1.93 |
| owner only | 3.48e-5 / 7.44e-5 | 1.81 / 1.69 | 1.99 / 1.93 |

*E2, curved.* The setup:
- a circle of radius ≈ 0.9 (κ = 1.11) centred outside the box, so the normals sweep 17–70° across it,
  with 4 radius offsets h/4 apart;
- two-layer azimuthal Couette (τ = 1 at r0, exact), and the same flow with an azimuthal body force
  (curved Poiseuille);
- μ ratios 1e-2, 1e2 and 1e3, N = 16…128.

What it shows:
- Fit normal on non-interfacial cells under the band mean: the brief's candidate A (the nearest
  interfacial cell by plane distance), the normalised mean of the neighbouring normals, and the exact
  normal give e·N² equal within 4 % in all 6 groups. The normal rule is immaterial.
- Interfacial mean against band mean: e(128) within −1…+3 %, envelope LS 1.87–2.12 against 1.86–2.14.
- Owner only: up to +10 % (Couette).
- Every rule, including the exact normal, has per-offset LS minima of 1.61–1.84. That is fill-in (Q-C),
  not a property of the rule.

*Conclusions from E1 and E2:*
- **The brief's centring hypothesis is refuted.** The old gap (owner 1.70–1.82 against band 1.93–1.95)
  was confounded:
  - `owner.py` included 33.75° and used the 32→64 pair;
  - the band figure excluded 33.75° and used 64→128;
  - the band mean's own like-for-like 32→64 minimum is 1.69 / 1.55.
- The averaging moves the constant, not the order, as §3.2's dipole argument says: any O(h) flux error,
  smooth or jagged, gives O(h²).

*Stability.* `stab6` geometries, N = 32. The worst ρ(G) under the interfacial mean equals the §6 values
within 3e-4 at every D_hi and ratio; the largest is 0.990.

*Rejected:*
- (A), keep the band and take the nearest interfacial normal: 2–3× more fits, plus a normal rule and a
  tie rule, for no measurable gain.
- (C), one fit per band edge with the owner's n: about 4× the fit cost, and the effect it targets is
  ≤ 4 %.
- Owner only: not reinstated. It is the weakest constant on the curve.
- If the band set is ever reinstated, use (A) with the metric |m_J·ξ_K − α_J|/|m_J|₂ and ties in
  offset order.

**Q-B — the virtual-plane tie.**
- **Phase threshold.** "Pure" uses the interfacial predicate's own threshold ε (`interfaceEps`):
  C ≤ ε is phase 0 and C ≥ 1 − ε is phase 1.
- **Qualifying.** A pure cell qualifies on axis a if a face-neighbour across an a-face is pure of the
  other phase.
  - Only interior faces count, and periodic faces, whose ghost is the image.
  - A non-periodic physical-domain face never qualifies.
- **Which face.** The plane goes on the first qualifying axis in the order x, y, z. On that axis it goes
  on the −a face if the −a neighbour qualifies, else on the +a face.
- **The plane.** In local coordinates it is ξ_a = f, with f = 0 (−a) or 1 (+a), m = σe_a and α = σf.
  σ = +1 if the C = 1 phase lies at ξ_a < f, else −1.
- **Determinism.** The rule reads C on a 7-point stencil inside the depth-3 colour ghosts and uses no
  reduction, so it is bitwise np-independent.

**The 1-cell film gets planes; it does not fall back.**
- *The geometry is exact anyway.*
  - The far face is owned at distance 0 by the opposite-phase neighbour across it, whose own virtual
    plane lies on that face; the film's plane is at distance 1.
  - On the near face, two coincident planes tie at distance 0.
  - In an alternating stack (ABAB… along a), every cell's plane is on its −a face and every face is still
    owned at distance 0.
- *The sign of n does not matter.* The kink model and the T-terms are quadratic in n: n_a(Pe_c)·(Sn), and
  n_a τ̂_c with τ̂ ∝ PSn. The film cell's fit is therefore the same whichever face is chosen.
- *A fallback buys nothing.* Withholding the plane gives the same owners and adds a third cell class.
  - What a film really lacks is a one-kink fit block. The fallback would not repair that, because the
    neighbours' fits see the same two interfaces.
  - That is Q8: first order locally, by design.
- *Coverage.* G-SV10 counts film cells: virtual-plane cells whose two a-neighbours are both opposite.
  G-SV9(d) covers stability on this geometry.

**Q-C — the 2-D order shortfall: pre-asymptotic, and WO-6 proceeds.**
- **Reading.** The brief's reading is confirmed, with one refinement: the constant does not oscillate,
  it rises monotonically to a bound.
  - Worst case: e·N² = 0.70, 0.73, 0.78, 1.13, 1.245, 1.258 for N = 16…512. The last doubling adds 1 %.
  - An O(h) leak would double e·N² at every doubling.
- **Mechanism.** The L∞ error is h² times the largest local constant on the interface, and that constant
  depends on each cut cell's sub-cell configuration.
  - At an irrational slope, the N cut cells sample those configurations ever more densely.
  - The sampled maximum therefore climbs towards its supremum, and a slope fitted while it is still
    climbing reads low.
- **E3 (33.75°, 16 offsets, N = 16…256).** The envelope samples 16N configurations.

| μ ratio | envelope e·N², N = 16 … 256 | envelope LS | per-offset LS min |
|---|---|---|---|
| 1e3 | 1.07, 1.14, 1.20, 1.24, 1.25 | 1.94 | 1.72 (3/16 below 1.8) |
| 1e2 | 0.83, 0.87, 0.91, 0.94, 0.96 | 1.95 | 1.79 (1/16 below 1.8) |

- **Ruling.**
  - The N = 256 test was there to tell fill-in from a leak. Two facts answer that: the 256→512 orders of
    the three worst cases (1.986–1.990) and the saturating envelope. The shortfall is pre-asymptotic.
  - §3.4's pre-registered consequence applies: G-SV1 runs at N = 24, 32, 48, 64 (§11).
  - WO-6 proceeds.
- **Measurement definition — not mine to rule (§12 Q13, user).** The per-offset LS slope is biased low
  for any scheme whose L∞ constant is a sampled supremum.
  - In 2-D it reads 1.69 (planar) and 1.61 (curved, even with the exact normal), for schemes whose
    envelope order is 1.93 and ≥ 1.86.
  - In 3-D a plane has O(N²) cut cells, so the bias is presumably smaller. That is unmeasured.
  - **Recommended.** Per orientation × ratio × case, take e_env(N) as the maximum over the 8 offsets and
    the order as its LS slope over the N set. PASS at ≥ 1.8.
  - This is still the worst-case error and drops no configuration. An O(h) term at any offset still
    dominates it asymptotically.
  - **Default until the user rules.** G-SV1 as written binds, and WO-6 reports the envelope beside it.

### 5.8 Seventh pass (architect, 2026-11-04): edge-owner ties — the owner set and its mean

**Amended by §5.9 (eighth pass).** The separation premise (spread ≪ τ_own ≪ gap) is false. The set
keeps its plateau at τ_own and gains a C¹ ramp to 1e-5, so the mean below is a *weighted* mean. The
escalation trigger "spread > 1e-8" and the census-zero bitwise class are withdrawn.

**Evidence.** `bubble_column_perf/validation/highratio/land/MIRROR_DIAG.md` (WO-M; DIAG patch
`land/mirror_diag.patch`, driver `land/mirror_diag.py`).
- The sharp default fails `mirror_symmetry`. Velocity defect, A against mirrored B: 3.08e-5
  (structured VoF, cy = 3.0), 7.87e-5 (block VoF, cy = 3.0), 3.76e-5 (cy = 6.0, clear of the walls).
- The whole defect is the owner tie in `SharpGeom::edgeOwner` (`src/vof/sharp_viscous.hpp`, the strict
  `dist < best`). Two mechanisms, each necessary:
  - *exact ties* (bit-equal d) when the blob's own symmetry plane lies on a grid face, broken towards
    the −y cell in A and in B, which are not mirror images;
  - *near ties* (0 < |Δd| ≤ 1e-12): the same geometric ties perturbed by arithmetic that is not
    reflection-equivariant (ξ = δ against 1 − δ, the octant undo in `plicAlpha`, MYC sum order).
- Near ties arise without a y-symmetry plane on a face as well: S = 2, cy = 3.3 has 12 exact and 14
  near ties on xz edges (x/z symmetries of the sampled colour) and a defect of 5.06e-5.
- The defect does not shrink with h (S = 1 → 2: 6.68e-5 → 5.91e-5).
- At a tied edge the candidates' fractions or normals differ by up to 0.335 (N = 24): an O(1) local
  change of the operator.
- "Near ties as ties" combined with B's tie order mirrored: 1.8e-15 – 2.9e-15 in every row. That is two
  rules, one in A and one in B, not one rule.
- Ruled out as causes: the virtual-plane tie (0 virtual cells in the test) and the τ̂ summation order
  (inert at the printed digits).

**Finding 1 — no single-owner rule can pass.**
- Let M′ be the reflection about the blob's own symmetry plane (a grid face), M the test's domain
  mirror, and T = M∘M′ the translation between them.
- The test checks Op(M C) = M Op(C) at C = C_A. Since M C_A = T C_A and Op is translation-equivariant
  (offset order), this says Op(C_A) = M′ Op(C_A): at an M′-symmetric colour the operator must be
  M′-symmetric.
- At an edge E on the symmetry plane, the tied candidates are M′-images of each other. M′ swaps them, so a
  rule that returns one of them returns the other for the mirrored input. No tie-break satisfies the
  invariant: not a fixed order, not a mirrored order, not any secondary key.
- **A symmetric combination of the tied candidates is necessary.**
- Example: for a mirror pair across the plane y = e_y through E, the u_c row's T-term at an a = y edge is
  ±n_y τ̂_c (1 − μ̄R̄_ac), with the sign decided by the tie. The exact value is 0:
  - σ_yc is odd under the mirror;
  - the interface normal at E has n_y = 0, while each cell's plane carries n_y = ±O(κh).
  The mean of the pair gives 0, so the mean is also the more accurate value there.

**Finding 2 — a tie must be recognised with a tolerance.** Ties that are exact in exact arithmetic arrive
with round-off spread:
- measured ≤ 1e-12 at step 0;
- bounded worst case about 1e-8, from `plicAlpha`'s asin branch near |arg| = 1 (√ε_mach loss at a case
  boundary);
- and dα/dC = 1/A_cut ≈ 1e5 in a corner-clipped cell at C ≈ 1e-8, acting on colour asymmetry that
  round-off grows over a persistently symmetric run.

**Finding 3 — the remaining discontinuity is acceptable once it is moved off zero.**
- Any rule built on a minimum is discontinuous in the colour. The settled operator already jumps by the
  candidates' difference wherever the argmin switches; the gates have run with that.
- The current rule fails because of *where* its jump sits: at gap 0, which is exactly where every
  symmetric and every grid-aligned configuration lives, so round-off decides.
- A tolerance band moves the jumps to gap = ±τ, each half the size. Structured inputs do not sit there.
- Round-off can decide membership only when a generic gap lies within the round-off spread of τ: about
  1e-15 per edge evaluation, 1e-8 in the asin worst case.

**The rule.** For the edge E_ac(k), take the candidates t = 0…3 in `edgeCells` offset order and let K_E be
the interfacial ones.
1. For t ∈ K_E: (m_t, α_t) = `plane(q_t, class)` and d_t = |m_t·ξ_t − α_t|/|m_t|₂. This is today's
   expression, bit for bit; store the four d_t (pass 1).
2. d_min = min_{t∈K_E} d_t. The owner set is T_E = {t ∈ K_E : d_t ≤ d_min + τ_own}, with τ_own = 1e-6.
   Compare exactly this way, using the stored d_t (never recomputed). n_E = |T_E|.
3. E is a band edge iff n_E ≥ 1, i.e. iff K_E ≠ ∅. This is unchanged, and the non-band paths
   (`edgePhase`, the four-class fast path in `SvBandOp::beta`) are unchanged.
4. Pass 2 walks T_E in offset order. It accumulates each output from 0.0 as acc += G_t and divides once
   at the end, G_E = acc / n_E. With F_row(E; t) and L_ac(E; t) the settled single-owner fractions
   (`rowFaceFraction`, `linkFraction` with owner q_t) and n_t = `vofPhysNormal(m_t)`:
   - face fraction (implicit band and explicit transpose): F̄_row(E) = Σ_t F_row(E; t) / n_E, and
     β = μ(F̄_row). Here row ∈ {a, c}.
   - fit shear-link rows (§5.3): θ̄_ac(E) = Σ_t L_ac(E; t) / n_E, θ̄_ca likewise. Each link keeps one LS
     row with its datum.
   - the T-term (§5.4 a), per member then mean:
     T_row(E) = Σ_t [ n_a,t τ̂_c (1 − μ(F_row,t) R(L_ac,t)) + n_c,t τ̂_a (1 − μ(F_row,t) R(L_ca,t)) ] / n_E,
     with τ̂ the interfacial mean (§5.7 Q-A, unchanged).
5. **Equivalent statement.** μ and R are affine in the fraction. So the edge's sample stress
   μ(F̄)·Δu/h + T is, in exact arithmetic, the arithmetic mean of the sample stresses that the settled
   single-owner operator gives with each member of T_E as owner. The fit row is likewise the mean of the
   members' rows, because `fitCoef` is affine in θ and the datum is member-independent.
6. **API shape (recommended, not binding).** Replace `edgeOwner` by
   `template <class F> int forEachOwner(k, a, c, F&& f) const`. It runs pass 1, then calls
   f(o, m, α) for each member in offset order, and returns n_E (0 = not a band edge). There are three
   call sites: `SvBandOp::beta`, `edgeTraction`, and the fit's shear-link item. No caller keeps a
   single owner.

**Why the T-term is averaged per member, not from averaged fractions.**
- μ(F̄)R(L̄) ≠ mean(μR). Take two members that place E's regions in opposite pure phases: each has
  μR = 1 and a T-term of 0, while the averaged fractions give μ̄R̄ = (2 + r + 1/r)/4 ≈ r/4. That is a
  spurious explicit stress of about (r/4)·τ̂.
- A mean normal can also vanish (antiparallel members across a film).
- β and θ̄ are affine in the fraction, so for them the mean fraction *is* the per-member mean.

**Properties (what the implementer may rely on, and what the gates check).**
- *Tie-free edges are unchanged.* For n_E = 1, (0 + G)/1 = G exactly, so the host build is bitwise the
  f01a251 operator at every tie-free edge. On CUDA the restructured inline code may contract FMAs
  differently, so tie-free edges agree to round-off there, not bitwise.
- *Reflection-equivariant.*
  - Membership depends on distances, which are reflection-invariant in exact arithmetic, through a
    tolerance ≥ 60× their round-off spread.
  - The mean is exactly order-free for n_E = 2, and order-dependent only at round-off for n_E = 3, 4.
  - The rule itself is permutation-equivariant too. End to end it is not: see Q32.
- *np-bitwise and translation-invariant.* The rule reads the same four cells (inside the depth-3 colour
  ghosts) and sums in the fixed offset order. It uses a compile-time constant and no reduction. The §7
  argument stands unchanged, including across a periodic seam.
- *Accuracy.*
  - Every member is an interfacial cell whose plane passes within τ_own of the closest, so every
    single-owner output is a valid sample of the settled operator. The mean of O(h²)-consistent samples
    is O(h²)-consistent.
  - On a plane, all members are exact, so the mean is exact.
- *Stability.* At a tie edge the implicit coefficient is the members' mean and the explicit term the
  members' mean. By the mediant inequality, the explicit/implicit ratio is at most the worst member's,
  and the settled rule could already have picked that member. G-SV9's per-edge premise is unchanged.
- *Films and virtual planes.*
  - Coincident planes (Q-B: "on the near face, two coincident planes tie at distance 0") give identical
    member outputs. Since (G + G)/2 = G exactly, the film and ABAB cases are bitwise unchanged at
    n_E = 2, and within 1 ulp at n_E = 4 (the film layer's a-parallel edges).
  - Q-B's −a-before-+a choice is output-inert for a film under the mean. The film's own plane is at
    d = 1 from the far face, and coincident with its neighbour's plane on the near face.
  - Q-B's axis order (x, y, z) is not permutation-equivariant at a pure cell that qualifies on two
    axes (a staircase corner). This is out of scope here (Q32).
- *F5 (a region lying in its owner's plane reads μ_B)* is phase-labelled, not orientation-labelled, so
  it is reflection-equivariant and is untouched.
- *Cost.*
  - Per edge: one 4-entry array of d_t plus a compare. Member planes are re-read in pass 2 (four loads).
  - Fractions are evaluated n_E times, on tie edges only.
  - A persistently symmetric run ties on O(N) edges (along the symmetry planes) out of O(N²) band edges.
- *Host ↔ CUDA.* Owner decisions agree across backends unless a generic gap lies within the backend
  round-off of τ_own. Field-level host ↔ CUDA agreement remains ordinary round-off (amplified by the
  dynamics and by the device FP32 V-cycle), and no longer hinges on ties.

**τ_own = 1e-6 (index units, absolute): why this value.**
- *Lower bound.* τ_own must exceed the round-off spread of exact ties: ≤ 1e-12 measured, about 1e-8
  bounded worst case (Finding 2). 1e-6 leaves ≥ 60× margin.
- *Upper bound.* Accuracy sets none, since any member within τ is a valid owner. τ only sets how often
  generic near-ties are averaged: a fraction of about 2τ per candidate pair, a handful of edges in a
  long run, harmless.
- τ must also stay far from structural gaps (0, ½, 1, √2/2, …) where grid-aligned configurations sit.
  1e-6 does.
- *Absolute, not relative.* d ∈ [0, √3] and its round-off is absolute. d = 0 at virtual planes, where a
  relative tolerance degenerates.
- *Not tied to `interfaceEps`.* That is a volume, user-set, and may be 0.
- *Not a setter.* A compile-time constant `kSvOwnerTieTol` in `sharp_viscous.hpp`. Nothing physical
  would tune it, and QUALITY_PLAN D3 forbids result-changing knobs without a reason.

**The census (diagnostic, not on the step path).** `diagnostics.vof_viscous_interface_census()` gains
three keys. They are computed from the same last-step band state and over the same owned range
[lo, hi) as the existing keys, over every edge E_ac(k) with k in that range, (a, c) ∈ {(0,1), (0,2),
(1,2)} and `edgeInDomain`. They are LOCAL to the rank: sum or max/min over ranks yourself.
- `'tied_edges'`: band edges with n_E ≥ 2.
- `'max_tie_spread'`: the maximum over tied edges of (max_{t∈T_E} d_t − d_min), in index units; 0.0 if
  there are none.
- `'min_untied_gap'`: the minimum, over band edges with |K_E| ≥ 2 and K_E ≠ T_E, of
  (min_{t∈K_E∖T_E} d_t − d_min); +inf if there are none.
- These read the void the rule relies on: spread ≪ τ_own ≪ gap. Edges on a non-periodic high domain
  face (k_a = dhi) are not counted; that is documented, not a defect.

**Rejected (one line each):**
- *A tie order, fixed or mirrored, for exact ties only.* Measured: it removes mechanism 1 only (cy = 6:
  3.76e-5 → 1.17e-5). Finding 1 also excludes any order.
- *A tolerance plus a single-candidate tie-break by an "equivariant" key.* Impossible by Finding 1: the
  tied candidates are swapped by the symmetry, and no single choice is a fixed point.
- *A continuous blend w_t = φ((d_t − d_min)/δ).*
  - It is equivariant and Lipschitz, but it amplifies round-off by about 0.3/δ, so passing at 1e-12
    needs δ ≳ 1e-2.
  - At that width it changes the operator at a finite fraction of all band edges (every gate
    re-validated) and blends planes up to δ cells farther than the closest.
  - It buys continuity that no invariant asks for. Not now (Q33).
- *Exactly equivariant arithmetic for d.* Out of reach upstream (MYC sum order, `plicAlpha`), and
  insufficient by Finding 1.
- *Mean over all interfacial cells of E, no distance.* It breaks the film's "every face owned at
  distance 0" exactness (§5.7 Q-B) and moves every edge's constant.
- *Mean of fractions and normal, evaluated once.* It gives the spurious ≈ (r/4)·τ̂ explicit stress above.
- *Relative tolerance (the DIAG's 1e-12·max(1, d)).* It degenerates at d = 0. With d ≤ √3 the factor is
  ≤ 1.73 anyway.
- *τ = 1e-12 (the DIAG value).* It passed a 3-step test but has no margin against the 1e-8 worst-case
  spread.
- *Scoping `mirror_symmetry` to 'averaged'.* Rejected by the coordinator (2026-11-04): asymmetry seeds
  path instability in symmetric rise.

**Value judgement.** None of this choice needs the user. The coordinator's invariant (mirror-symmetric to
round-off) forces a symmetric combination (Finding 1). The tolerance is a fact with margin (Q31). The
gate bands (§11 G-SV12, WO-9.4) are the coordinator's to apply.

### 5.9 Eighth pass (architect, 2026-11-04): the void is absent — a C¹ ramp after the plateau; G-SV12 restated

**What §5.8 got wrong.** §5.8 rested on a separation: round-off spread ≪ τ_own ≪ generic gap. The census
(WO-9.4 step 1, svl-owner `b4a4a31`; evidence `bubble_column_perf/validation/highratio/HANDOFF_WO9b.md`,
raw data `sv_impl/w9/`) refutes it. There is no void:
- *Frozen G-SV1 colours, 506 cases.*
  - The axis and two-axis normals (100/110/101/210/012) are clean: spread ≤ 4e-12, no untied gap.
  - The (2,1,1) family at k = 3 has spread 5.0e-7 and gap 4.8e-6 at every N. These are structural
    near-ties, not round-off.
  - gsv1w (22.5°/22.5°) has spread 3.4–4.2e-7 and gap 1.0e-6.
  - (1,1,1) at k = 5 has spread 2.3e-10, 4.6e-10, 1.1e-8, 1.4e-8 at N 24/32/48/64.
- *Evolving runs.* P1 d/h 16 at step 1 has 654 tied edges, spread 3.3e-16, gap 2.2e-5. From step 2 on,
  the spread is 9.7e-7 and the gap 1.2e-6. The walled N24 case has ties by step 5.
- *Consequence.* No G-SV1 row is tie-free.
- *What held.* Mirror ≤ 2.3e-15 on host and CUDA, with equal tie counts; MPI bitwise; tie-free identity
  digest = f01a251; smoke reruns against 56d115f bitwise, 2e-15 and 1.3e-7 relative.
- *Diagnosis.* The density of near-equidistance gaps is finite at 0. Neighbouring planes on a smooth or
  layered interface are nearly equidistant from shared edges, and there are O(10³–10⁴) band edges per
  step. Gaps therefore fill every neighbourhood of τ_own.
- *The defect.* The set switch at gap = τ_own is a finite operator jump. Round-off decides it whenever a
  gap lies within its own round-off δ of τ_own. The probability is about 2δρ per step, with ρ the gap
  density at τ (≈ 10⁶–10⁷ per unit gap per step, from the P1 numbers).
  - It is negligible for ordinary δ ≈ 1e-16 (about 1e-9 per step).
  - It is likely over a long run in cells where `plicAlpha` loses half its digits (δ ≈ 1e-8; see (c)
    below): of order 1e-2 per step. Such a switch injects an O(0.15) local asymmetry.
  - That is the gap-0 defect, moved to gap τ and made rarer, but not removed.

**(a) Decision — keep the plateau, add a C¹ ramp; the weights become continuous.**
- With g_t = d_t − d_min, each candidate t ∈ K_E gets the weight
  w(g) = 1 for g ≤ τ₀; w = s²(3 − 2s) with s = (τ₁ − g)/(τ₁ − τ₀) for τ₀ < g < τ₁; w = 0 otherwise
  (and for NaN).
  - τ₀ = 1e-6 (`kSvOwnerTieTol`, unchanged); τ₁ = 1e-5 (`kSvOwnerRampTol`, new).
  - The closest candidate (strict <, first in offset order) has w = 1 by construction, even when its
    distance is NaN. This is the as-built rule of WO-9.2, endorsed here. It keeps the F3 fallback test.
- Each edge output becomes the weighted mean G_E = Σ_t w_t G_t / Σ_t w_t, summed in offset order. §5.8
  items 4–5 hold with weights:
  - F̄ = Σ wF / Σ w, and β = μ(F̄);
  - θ̄ = Σ wθ / Σ w;
  - T_row = Σ w_t[per-member product] / Σ w.
  The edge stress is still a convex combination of the single-owner stresses, so §5.8's accuracy and
  mediant-stability arguments carry over unchanged.
- *What it buys.* The operator is now a continuous, piecewise-C¹ function of the distances. No gap value
  exists at which round-off moves it by a finite amount. A round-off δ in any gap moves an edge output by
  at most K·δ·max|G_t − G_E|, with K = 1.5/(τ₁ − τ₀) = 1.67e5:
  - ordinary δ ≤ 1e-14 gives ≤ 2e-9 local, invisible at the 1e-12 mirror gate;
  - a √ε cell (δ ≈ 1e-8) gives ≤ 5e-4 local, deterministically, instead of a 0.15 jump with probability
    2δρ.
- *What it keeps.*
  - Genuine ties (spread ≤ 1.4e-8 measured, ≤ ~5e-8 bounded) sit 20× inside the plateau. Their weights
    are exactly 1, so the mirror-pair mean is exact, as in §5.8.
  - A genuine tie beyond τ₀ would now degrade smoothly (w = 0.97 at g = 2e-6), not by a cliff.
- *Inert where it should be.*
  - Where no candidate lies in (τ₀, τ₁), every weight is exactly 1.0 or 0.0. 1.0·G = G, and Σ w is the
    integer n_E, so the result is bitwise the as-built 582ac73 operator.
  - That covers the frozen axis and two-axis colours (no untied gap), and every tie-free edge (bitwise
    f01a251).
- *What it moves.* Candidates with gaps in (1e-6, 1e-5) now take partial weights: (2,1,1) k3's
  structural 4.8e-6 gap gets w = 0.616. On a smooth interface, near-equidistant planes are nearly the
  same plane locally, so ΔG is small there. The smoke data (1.3e-7 relative for the plateau) bound the
  expected movement well inside the G-SV12 bands.
- *Why τ₁ = 1e-5.* One decade of ramp keeps K small enough that ordinary round-off is invisible, while
  the number of partially weighted edges per step stays near ρ·τ₁ ≈ 10–100. τ₁ = 1e-4 would cut the √ε
  case another 10× at about 10× more blended edges. Not needed (Q35 removes the √ε case at its source).

*Rejected for (a):*
- *Keep the sharp set at 1e-6.* Round-off-decided O(0.15) switches, likely in √ε cells over a long
  symmetric run. That is exactly the seed the coordinator excluded.
- *Scale τ relative to N, κh or the spread.* No scale separates genuine from generic ties: the gap
  density is finite at 0, and structural gaps (5e-7, 4.8e-6) exist at every N. Any threshold has
  neighbours.
- *A structural tie test (bitwise mirror-equal colour stencils).* Exact only at step 0. In an evolving
  symmetric run the colour symmetry is round-off-exact, not bitwise, so the test fails exactly where it
  is needed (persistent symmetric rise).
- *Ties defined at step 1 only.* Not a function of the state: it breaks restart, checkpoint and
  `redistribute` invariance. A symmetric run needs ties at every step.
- *A wider ramp alone, without the plateau (§5.8's rejected blend).* A genuine tie with spread δ would
  get w = 1 − O((δ/W)²) instead of exactly 1, which is asymmetric at the 1e-4 level for δ = 1e-8. The
  plateau is what keeps mirror pairs exact.

**(b) G-SV12, restated** (§11 carries the text):
- *Binding pass/fail:*
  - mirror ≤ 1e-12, every row, host and CUDA;
  - the `sharp_viscous` reflection case ≤ 1e-13;
  - `sharp_viscous_mpi_np{1,2,4}` bitwise;
  - the tie-free identity digest (= f01a251);
  - the new `ownerWeight` unit test;
  - the pinned plateau `tied_edges` equal on both backends;
  - frozen axis and two-axis G-SV1 colours bitwise equal to 582ac73 (the ramp is inert there).
- *Study gates: band-only*, against the recorded single-owner baselines. The verdict must not change.
  The bands are those of §11 G-SV12. The census-zero bitwise class is withdrawn, because it is empty.
- *Census keys are INFO, never triggers.* `max_tie_spread` and `min_untied_gap` mix genuine ties with
  generic near-ties, so neither identifies a defect.
- *Escalation triggers* (they replace "spread > 1e-8"):
  1. any binding invariant fails;
  2. a verdict flips: STOP;
  3. a band breach with an unchanged verdict: INFO to the architect;
  4. `blended_edges` + `tied_edges` exceeds 5 % of band edges at any step of P1 d/h 16–48 or G-SV2: INFO.
     It would mean the near-equidistance density is higher than assumed, so the rule alters more of the
     operator than designed;
  5. the frozen (1,1,1) k5 spread at N = 96 or 128 exceeds 1e-7: escalate (Q35).

**(c) The (1,1,1) N-growth.**
- *What it is.* |m| ∝ (1,1,1) sits on `plicAlpha`'s n1 = n2 = n3 degeneracy, where case (2) has zero
  width. The cubic branch's asin argument approaches ±1 there, and the inversion loses about half the
  digits. More cut cells (∝ N²) sample closer to the singular point, so the max spread climbs toward
  √ε_mach·O(1) ≈ 2–5e-8 and saturates.
- *For the owner rule it does not matter.* It sits 20× inside the plateau, and with the ramp an
  overshoot degrades smoothly.
- *It is an upstream fact.* The stored α carries ~1e-8 error in degenerate-normal cells. That perturbs
  fractions at the same level whatever the owner rule (Q35).
- *Falsifier.* Frozen (1,1,1) k5 at N = 96 and 128 should give spread ≤ 5e-8.

**Implementation delta (from 582ac73; WO-9.2b; one commit).** In `src/vof/sharp_viscous.hpp`:
1. Add `constexpr double kSvOwnerRampTol = 1e-5;` beside `kSvOwnerTieTol`.
2. Add `KOKKOS_INLINE_FUNCTION static double ownerWeight(double g)`:
   `if (g <= kSvOwnerTieTol) return 1.0; if (!(g < kSvOwnerRampTol)) return 0.0;`
   `const double s = (kSvOwnerRampTol - g) / (kSvOwnerRampTol - kSvOwnerTieTol);`
   `return s * s * (3.0 - 2.0 * s);`.
3. `edgeOwnerSet` also fills `double w[4]`. For t ∈ K_E, w[t] = (t == best) ? 1.0 :
   ownerWeight(dist[t] − dmin); otherwise 0. The member mask = {w > 0}.
4. `forEachOwner` calls `f(o, m, α, w_t)` for each member and returns `double W` = Σ w_t, summed in
   offset order; W = 0.0 means not a band edge. The three callers do `acc += w * G` and `G_E = acc / W`,
   and test `W == 0.0` where they tested `n == 0`.
5. Census: `edgeTieKeys` keeps the plateau semantics:
   - n_E counts members with g ≤ τ₀;
   - spread is the max g over those members;
   - gap is the min g over candidates with g > τ₀.
   Add the key `'blended_edges'`: edges with a candidate in τ₀ < g < τ₁. Update the binding docstring.
6. Update the header comment and the docstrings: "weighted mean; plateau τ₀, C¹ ramp to τ₁".
7. Tests:
   - `ownerWeight` unit test: w(0) = w(τ₀) = 1 exactly; w(τ₁) = w(2τ₁) = w(NaN) = 0; non-increasing on a
     1e5-point sweep of [0, 2τ₁]; |Δw/Δg| ≤ K(1 + 1e-9); |1 − w(τ₀ + h)| and |w(τ₁ − h)| ≤ 3(h/(τ₁−τ₀))²
     at h = 1e-9.
   - The frozen axis and two-axis colours are bitwise equal to 582ac73.
   - The existing WO-9.2/9.3 gates still pass.

---

## 6. Time treatment: the split and its stability

The predictor is (M + K)u* = (M + K − A)uⁿ + rhs, where:
- M = ρ_F/dt, with ρ_F as today (`face_props.hpp:70`);
- K is the implicit per-component band: band edges μ̄_c,E; non-band edges unchanged; cells 2μ(C) on
  interfacial cells and μ(C) elsewhere (§6.1). **Third-pass correction:** the spectra below were measured with
  2μ(C) on every cell (the harness's K). The code first ran μ(C) on every cell, and that split is unstable
  (§6.1);
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

### 6.1 Third pass (architect, 2026-10-31): the G-SV9(b) blow-up is the cell split, not a 3-D mode

**Question.** G-SV9(b) (3-D march, plane n(22.5°, 22.5°), N = 16) blows up under 'sharp' at D_hi = 100
and 1e4 for all four ratio pairs; 'averaged' does not. The 2-D harness above had measured ρ(G) ≤ 0.99.

**Evidence.** Script `tests/study/sharp_viscous/gsv9_split.py`; logs in
`bubble_column_perf/validation/highratio/sv_impl/arch3/` (`harness16.out`, `prod3d*.out`).

- **The departure.** The harness above (`stab.py`, `gsv9a.py`) builds K with
  `blocks(…, {"xx","yy","yu","xv"})` under `CELL_VOL`. Its cell links carry the stress factor 2, so K holds
  the whole cell normal stress 2μ(C)m_cc. The code holds half of it:
  - the Laplacian band (`ibmBuildDiffusionVar`, cell β = μ(C)) is implicit;
  - the normal element N_c of `src/full_stress.hpp:12` carries the other μ(C)m_cc, explicit on uⁿ;
  - `SharpFaceProps::beta` (`src/vof/sharp_viscous.hpp:415-418`) returns `base.beta` for the cell link,
    to both.

  §5.4(b) as first written prescribes exactly this split. So the implementation follows the note, and the
  note's stability evidence was measured on a different split.
- **2-D harness, N = 16.** The six geometries above × four ratio pairs. Worst ρ(G) at
  D_hi = 0.1 / 1 / 10 / 100 / 1e4:

| A | K: the cell normal stress | worst ρ(G) |
|---|---|---|
| sharp | K2: 2μ(C) implicit on every cell (what this section measured) | 0.9955 / 0.9886 / 0.9867 / 0.9864 / 0.9864 |
| sharp | **K1: μ(C) implicit + μ(C) explicit (the code)** | 0.9947 / 0.9813 / **1.0381 / 1.1120 / 1.1353** |
| sharp | **Kb: 2μ(C) implicit on interfacial cells, K1 elsewhere (chosen)** | 0.9954 / 0.9883 / 0.9862 / 0.9859 / 0.9859 |
| averaged | K1 (today) | 0.9870 / 0.8859 / 0.9307 / 0.9742 / 0.9882 |
| averaged | Kb (not proposed) | 0.9870 / 0.8861 / 0.5793 / 0.7111 / 0.7629 |

  At D_hi = 1e4, K1 exceeds 1 in 18 of the 24 cases: every oblique one, and (0°, 0.6) at ratio 1e2 and 1e3.
  At (0°, 0.6) with ratio 1e3 / 1e4 it already does so at D_hi = 10 (1.038). The aligned (0°, 0.2) stays
  ≤ 0.95.
- **The production 3-D step (the discriminating test).**
  - *Method.* Arnoldi (scipy `eigs`) on u ↦ step(u), colour frozen, converged solves (RB-GS to 1e-15, the
    Chebyshev pressure driver to 1e-13). The projection is non-incremental, so the map is the harness's G.
    Linearity holds to ≤ 1e-11.
  - *Case.* The harness case (22.5°, 0.3), extruded in z: Nz = 4, z periodic, walls on x and y.
    D_hi = 1e4.

| N | μ_A/μ_B | prod. sharp | harness K1 | harness K2 | prod. averaged | harness averaged K1 |
|---|---|---|---|---|---|---|
| 8 | 1e-3 | 1.0004 | 1.0218 | 0.8524 | 0.8869 | 0.8909 |
| 8 | 1e3 | 1.0963 | 1.0912 | 0.9253 | 0.9244 | 0.9321 |
| 16 | 1e-3 | 1.0890 | 1.0905 | 0.9339 | 0.9867 | 0.9874 |
| 16 | 1e3 | 1.0932 | 1.0936 | 0.9288 | 0.9865 | 0.9870 |

  - The leading eigenvector is z-independent with u_z = 0. The k_z = 0 subspace is invariant to 3e-12 and
    carries the same ρ.
  - Production matches K1 to ≤ 2 % (≤ 0.2 % at N = 16), and not K2.
  - **Verdict: no genuine 3-D mode. The departure is the measured split.**
- **The G-SV9(b) plane itself.** n(22.5°, 22.5°), walls on all faces, exact fractions, N = 16, the same
  Arnoldi (pressure converged in ≤ 19 iterations).
  - Sharp: ρ = 1.042 / 1.067 at D_hi = 100 / 1e4 (μ ratio 1e-3); 1.042 / 1.069 (1e3); 1.007 / 1.033 (1e-2);
    1.008 / 1.035 (1e2). The leading eigenvalue is ≈ −1.04 ± 0.03i: sign-alternating.
  - ρ² is 1.01–1.14 per step, the march's measured energy growth of 1.04–1.13.
  - Averaged: ≤ 0.983. At D_hi = 1 and 0.1, both modes give 0.838 and 0.980.
  - N = 12 gives the same pattern, and is stable at D_hi = 10 (0.927). `gsv9.py`'s 8³-sub-sampled colour
    gives ρ within 1e-3 of the exact-fraction colour, so the sub-sampling is immaterial.
- **Mechanism.**
  - With K1, the cell normal element is split evenly: K_cell = T_cell = μ(C).
  - The symmetric part keeps the bracket K ± T ≥ 0, and at large D it reaches λ = −1 exactly on
    irrotational fields. So on solenoidal fields 'averaged' sits just inside −1 at large D_hi (0.983–0.988).
  - The sharp cell terms, 2Δμ(C̄−C)(m_cc − D̂_nn) and the cell τ-term, act explicitly on the same normal
    strain, at liquid-viscosity strength. They push that cluster past −1.
  - With the normal element implicit on a cell, K′ − (T − N) = K − T + 2N. Reaching −1 now also needs
    N v = 0, so the cluster moves away from −1 (to ≤ 0.76 for 'averaged' under Kb). That leaves room for the
    sharp terms.

**Decision: the fix is the split, not a dt cap (WO-5b).**
- **The rule.** Under 'sharp', on every interfacial cell (`cellClass ≠ kSvNone`, virtual-plane cells
  included), the u_c row's cell link along c carries:
  - 2μ(C_i) in the implicit band;
  - 0 in the explicit normal element N_c.

  The midsection and τ-terms stay explicit, as in §5.4(b). Every other cell keeps today's split.
- **What does not change.**
  - A itself, its steady state and every G-SV1 result.
  - The operator-level exactness (7.6e-12) and the G-SV6 operator ctest.
  - 'averaged', and every u_c row not adjacent along c to an interfacial cell: bitwise.
  - Only the transient on band rows moves.
- **Why virtual-plane cells are included.** Their cell extras vanish (C̄ = C, μ(C̄)R(C) = 1), so including
  them only moves a normal element into the implicit band. That is the K2 direction, which is stable.
- **Rejected.**
  - *A dt cap.* The instability starts at D_hi ≈ 10. G-SV1 and G-SV2 run steady Stokes at D_hi = 100, by
    design: that is the incremental predictor's large-dt capability. A cap would hide a split defect, not
    remove it.
  - *K2 on every cell under 'sharp'* (the literal split measured above). The margin is the same (0.9864
    against 0.9859). But it moves the transient of every bulk row under 'sharp': in the bulk, the explicit
    normal element is part of a transpose that sums to zero on a solenoidal uⁿ, and moving it into the
    implicit band changes u* by O(dt) everywhere. It would also break the §5.2 property that non-band rows
    equal 'averaged' bitwise.
  - *Changing 'averaged'.* Kb would also widen its margin (0.763 against 0.988). But 'averaged' is stable
    and must stay bitwise today's (G-SV5).
- **Not settled by this pass.** Kb is measured in 2-D only; the 3-D Kb waits for WO-5b. If G-SV9(b) or (e)
  still finds ρ > 1 after WO-5b, stop and return to the architect. Do not tune, and do not cap dt.

### 6.2 Third pass: the film, the pressure driver, the growth criterion (Q2)

**G-SV9(d) failed on the pressure solve, not on the split.**
- `gsv9.py` calls `set_pressure_pcg(True, 800, 1e-12)` after the ρ property model. That overrides the
  Chebyshev driver which `set_property_model("rho", …)` selects.
- Both CG drivers fail above density ratio ~1e3 (flow `CLAUDE.md`, "Pressure solve"). In every blow-up the
  PCG capped from step 1, and an unconverged projection is not a projection.
- Production step, film geometry, N = 12, Arnoldi with converged solves (Chebyshev), D_hi = 10 / 1e4:

| μ_A/μ_B (ρ_A/ρ_B) | averaged | sharp |
|---|---|---|
| 1e-3 (1e-4) | 0.554 / 0.495 | 0.575 / 0.523 |
| 1e3 (1e4) | 0.554 / 0.495 | 0.576 / 0.722 |
| 1e-2 (1e-3) | 0.535 / 0.468 | 0.555 / 0.495 |
| 1e2 (1e3), FCG | 0.535 / 0.468 | 0.556 / 0.711 |

  Both paths are stable on the film at every ratio.
- **The CG drivers fail silently at these contrasts.** At density ratio 1e4, FCG either capped at 20 000
  iterations or stopped on its tolerance with a wrong field: on the extruded case it gave ρ = 1.20
  ('averaged') and 1.87 ('sharp'), where Chebyshev gives 0.987 and 1.089.

**Rulings.**
- G-SV9(d) is well-posed once the gate runs the production pressure driver. It stays a gate for 'sharp',
  with the PASS of (b), and 'averaged' runs beside it as the control. The MISS recorded for both paths on
  the PCG run is void, being an instrument fault. Re-judge after WO-5b.
- `gsv9.py` must not choose the pressure driver. Delete the `set_pressure_pcg` call, keep the Chebyshev
  driver that the ρ model selects, and give it rtol 1e-12 and a cap that never binds. Each run line reports
  the driver and the maximum iterations, and a capped step makes the run NOT JUDGED.
- **Production meaning.**
  - On this evidence today's split is stable on a 1-cell film at these ratios; the blow-up was a test
    artifact.
  - A 1-cell film in production is a question of solver health and of accuracy (Q8: first order locally),
    not of stability. The variable-ρ default is already Chebyshev.
- **Package scope.**
  - The film belongs to this package as a gate.
  - The CG-at-high-contrast failure is documented behaviour and not this package's to fix.
  - Recorded for the coordinator, as a separate item outside this package: a forced CG driver at density
    ratio ≥ 1e3 returns wrong fields silently, not just slowly. A one-time warning would be cheap.

**The growth criterion at the floor (USER: any change after the data is the user's call).**
- **What the data say.** The plane D_hi = 1 MISSes are not eigen-growth.
  - The non-incremental Arnoldi gives ρ = 0.837–0.839 for both modes and both ratio pairs.
  - The march's window ratios above 1 (1.0019 averaged, 1.000038 sharp) occur only after E/E0 has fallen
    below ~1e-16. Both runs end at E/E0 ≈ 1e-69. Below that level the linear map is no longer what the
    solvers apply: absolute floors in their stop tests take over. The incremental (u, p) spectrum is in
    §6.4. On that production map the D_hi = 1 cases give ρ = 0.976 / 0.913 'averaged' and 0.921 / 0.913
    'sharp' (μ 1e-3 / 1e-2): stable, with margin.
- **Proposed (USER).**
  - Judge E(n+100)/E(n) ≤ 1 + 1e-10 only over windows with E_n ≥ 1e-20·E_0. Report the ratio below that
    level, without judging it.
  - Add the spectral gate G-SV9(e), which has no floor.
  - Reason: a random initial field projects O(1/n) ≈ 1e-4 of its energy onto every eigenvector. Any mode
    with |λ| > 1 therefore dominates long before E falls to 1e-20·E_0, so the floor rule cannot hide an
    instability. Below 1e-20 the measurement is of the solver's stop tests, not of the scheme.
- **Default until the user rules.** Report both verdicts. The floor-excluded one plus G-SV9(e) decides
  sequencing; the as-written one is recorded beside it.

### 6.3 Third pass: G-SV1 without walls, and the wall-contact error (Q3)

**Ruling: (P), a triply periodic layered test.** Specified in §11 as G-SV1P / G-SV1cP / G-SV1dP.
**USER**: this replaces the pre-registered walled box. Default: run (P), and keep the walled numbers as a
reported diagnostic.

- **Why not the ghost-data hook (B).** "Exact face averages on the ghost layer" cannot be fed to the
  production stencil through a hook.
  - The wall rows fold the ghost into the matrix as 2u_w − u_in.
  - The colour ghosts, the planes and τ̂ in the ghost layer would all have to be exact too.
  - A periodic topology cannot carry a non-periodic field, because the faces at x = 0 and x = 1 are one
    unknown.

  A hook would therefore be a new BC type, touching production BC code, used by nothing in production.
- **Why not fixing the walls (W).**
  - *The defect.* At a Dirichlet wall edge, the wall-normal shear link is the half-link [wall, first node]
    with a box kernel, but R̄ is taken from the hat over [e−1, e+1], which reaches into the ghost.
  - *The data in run A.* Point values were used where the half-link needs the wall-face average.
  - *The fix.* It is well defined: a half-box R̄ and θ̄, and owners among in-domain cells only.
  - *Why not now.* Box edges, corners and fit rank loss make the exactness tail long, and production
    contact lines sit on no-slip walls. There the continuum solution is singular (moving contact line) or
    a corner flow, and in neither case kinked-linear. Second order at a wall contact line is not a
    deliverable of any viscous operator.
- **What (P) loses, and where it goes.**
  - No periodic flow is kinked-linear: periodicity forces τ·Σd_k/μ_k = 0, hence τ = 0.
  - So Couette exactness and extension exactness move to operator level, as G-SV1x: the WO-4 subtest with
    the exact pressure included.
  - The along mode's second-order content moves to a manufactured periodic flow, G-SV1cP.
  - Irrational normals are replaced by rational ones. Each offset then samples a finite set of sub-cell
    configurations that does not change with N, so the per-offset slope carries no fill-in bias (§5.7 Q-C).
  - The dense sampling of irrational slopes is still covered by the 2-D sweep (§3.4) and the 3-D extrusion
    identity (WO-2).
- **The wall-contact error.** Out of scope for this package: a known limitation, recorded in Q6.
  - It stands on one condition, which is a fact, checked by G-SV1W: at a wall contact, 'sharp' must be no
    worse than 'averaged'.
  - If G-SV1W misses, WO-W (below) comes into scope.
  - If G-SV9(b) still misses after WO-5b while its periodic twin G-SV9(b′) passes, the wall treatment is a
    stability defect. Return to the architect.

### 6.4 Third pass: the steady state for G-SV1 (Q4)

**Ruling: GMRES on the production step's fixed-point residual (WO-6s). No production change.**
- **The equation.** With the colour frozen, the step is an affine map S on the full state x = (u, v, w, p),
  written S(x) = Gx + b. The steady state solves (I − G)x = b.
  - The matvec is v ↦ v − S(v) + S(0): one production step, with S(0) computed once.
  - p goes in and out via `get_field("p")` / `set_field("p")`, internal units both ways. The flow
    `CLAUDE.md` names these as the checkpoint path; `get_p` is physical and must not be mixed in.
  - The mean of p is removed on input and output.
- **Why the fixed point is the production steady state.**
  - At a fixed point of the incremental-rotational step, φ = χμ_min·D u* = χμ_min·L φ. Since L ≤ 0, φ = 0,
    and then A u + G p = f.
  - So the result is dt-independent and equal to the plain march's limit.
  - The non-incremental projection is not equivalent. Its fixed point carries Chorin's term K M⁻¹ G φ, so it
    must not be used for G-SV1.
- **Settings.**
  - GMRES(60), started from the exact face averages with p = 0. D_hi = 100, as pre-registered.
  - Certification is the pre-registered criterion, on a plain production step: after each restart cycle,
    the step increment must satisfy ‖S(x) − x‖_∞ ≤ 1e-12·max|u|.
  - Cap: 3000 matvecs. A case at the cap is NOT CONVERGED and is not judged.
  - Periodic gauge: per component, the discrete momentum Σ_faces ρ_F u_c is invariant, being 0 from rest.
    Compare u_h with ū − ⟨ū⟩_ρF, removing the ρ_F-weighted mean of the exact face averages.
- **Cost.**
  - Expected 100–300 matvecs a case, against 1.7–2.4e3 plain steps. Measure the count on a 10-case pilot
    first.
  - The −1 cluster that makes the plain march slow (|λ| 0.98–0.99) maps to ≈ 2 in I − G, which GMRES
    handles easily.
- **Rejected.**
  - *Lifting the `march_to_steady` refusal for frozen colour.* It adds a test-only state to the production
    API, and the type-II Anderson it would run equals GMRES on a linear map.
  - *A directly assembled steady operator.* It needs a new "apply A" surface and a saddle-point solver
    beside the time split. It would be faithful only by a further proof.
  - *A plain march at larger dt.* The steady state is dt-independent, but the −1 cluster slows the plain
    march at every large D.
- **The incremental (u, p) spectrum** (`gsv9_split.py prod --incremental`, G-SV9(b) plane, N = 16,
  Chebyshev).
  - It carries slow real modes that the u-only map lacks: 'averaged' 0.9946 / 0.9981 at D_hi = 100 / 1e4
    (μ 1e-2), and 0.9997 at 1e4 (μ 1e-3). ARPACK did not converge for μ 1e-3 at D_hi = 100: a cluster
    at 1.
  - This is the pressure increment relaxing under the rotational coefficient χμ_min. It is why the plain
    march needs ~2e3 steps.
  - For GMRES the eigenvalues of I − G sit at 3e-4 … 5e-3. A few isolated ones are deflated within a few
    iterations; a dense cluster would not be. The pilot decides.
  - *Fallback if the pilot needs > 1000 matvecs.* The fixed point does not depend on χ: (I − χμ_min L)φ = 0
    gives φ = 0 for every χ ≥ 0. So `diagnostics.set_variable_rotational` may be changed for the steady
    solve only. Accept this only if the fixed point agrees with the default setting's to ≤ 1e-11·max|u|
    on two pilot cases.
  - Sharp plane before WO-5b: 1.046 / 1.072 (μ 1e-3) and 1.017 / 1.045 (μ 1e-2), the same mode as the
    u-only map (≈ −1.04).

---

### 6.5 Fourth pass (architect, 2026-11-01): the periodic twin G-SV9(b′) (Q1)

**Data.** (b′) = the (b) march on a triply periodic box, m = (2,1,1), two plane families per period
(layer fraction 0.5), N = 16 (`R/gsv9b/pplane_*.out`, R = `validation/highratio/sv_impl/rb7`).
- 'sharp': every D_hi ≥ 100 blows up within 15–24 steps. Production Arnoldi (converged solves,
  incremental, μ 1e-2 / ρ 1e-3, D_hi 100): ρ = 1.4868, real negative, linearity 8.2e-12.
- 'averaged': the μ 1e-3/ρ 1e-4 pair and its inverse grow ~1.06/step at D_hi 10…1e4; the other
  pairs plateau at window ratios 1.00000–1.00025.

**Instrument findings (settled).**
1. *The plateau is the translation subspace, not a mode of the scheme.* In a periodic box the three
   uniform fields are exact eigenvectors with λ = 1 (zero stress, zero divergence, zero fit). Σρ_F u_c
   is conserved only to the momentum-solve residual, and the march runs the production default solve
   (§6.6), which exits above tolerance at D ≥ 10: each step deposits momentum, the undamped translation
   modes accumulate it, and E plateaus or creeps (1.00005–1.00025 per 100 steps). **Fix (WO-6a′):** on a
   periodic box `gsv9.py` judges E′ = E − Σ_c (Σ_F ρ_F u_c)² / (2 Σ_F ρ_F), the energy of the
   M-orthogonal complement of the translations (an invariant subspace of any momentum-conserving step);
   E is still printed. `gsv9_split.py prod` on a periodic box deflates the three translations
   (project them out of every Arnoldi vector, M-orthogonally) so that ρ is not pinned at 1.
2. *The geometry is outside the operator's envelope as specified.* Each layer is P/2 = 1/(2√6) = 3.27
   cells thick at N = 16, and the band is half the box (2048 of 4096 cells). The fit block of every
   interfacial cell reaches ±2.45 h along n (§5.3), so it contains the other interface. The model is
   still exact for parallel layers (each link's θ̄ comes from its own owner), but §6 and §6.1 never
   measured stability with two interfaces inside one fit/stencil reach.

**Discriminators (architect, host, N = 16 unless stated; production u-only map, non-incremental,
converged solves: RB-GS update stop 1e-12, residual stop off; Chebyshev 1e-13; D_hi = 100;
`gsv9_split.py prod --geom pplane [--roll|--frac|--m]`; logs `sv_impl/arch4/disc_*.out`).**

| case | mode, μ_A/μ_B, ρ_A/ρ_B | gas layer / liquid layer (cells) | ρ | leading λ |
|---|---|---|---|---|
| E1 (b′) | sharp, 1e-2, 1e-3 | 3.27 / 3.27 | 1.4558 | −1.4558 (double) |
| E2 E1 rolled (5,3,7) cells | sharp, 1e-2, 1e-3 | 3.27 / 3.27 | **1.455803 (identical)** | identical |
| E7 frac 0.75 | sharp, 1e-2, 1e-3 | 4.90 / 1.63 | 1.2057 | −1.2057 (double) |
| E8 frac 0.25 | sharp, 1e-2, 1e-3 | 1.63 / 4.90 | 1.0000 (translations), next 0.969 | stable |
| E4 N = 24 | sharp, 1e-2, 1e-3 | 4.90 / 4.90 | 1.2019 | −1.2019 (double) |
| E5 (b′) | **averaged**, 1e-3, 1e-4 | 3.27 / 3.27 | **1.0652** | +1.0652 ± 0.0014i, −1.018 ± 0.042i |
| E6 (b′), incremental | **averaged**, 1e-3, 1e-4 | 3.27 / 3.27 | 1.0212 | −1.0206 ± 0.037i |
| E3 walled, one plane m = (2,1,1) | sharp, 1e-2, 1e-3 | half box | running (`disc_E3_wall211.out`) | ≤ 1: two interfaces in reach are the trigger; > 1: one oblique interface is enough at this normal, so (b)'s pass is geometry-dependent. The rulings below hold either way. |

**Verdict: a genuine mode of the composite split, not an implementation departure.**
- *Not the seam.* A whole-cell translation leaves the spectrum identical to 6 digits (E2). Every seam
  hypothesis (SvRole, explicit element, SharpFaceProps, τ̂ wrap, halo) would break that invariance.
- *Not only the instrument.* The plateau is (finding 1), the blow-up is not: E1/E5 are converged,
  linear (lin ≤ 3e-9), and above 1 by 6–46 %.
- *Not specific to 'sharp'.* Today's production operator ('averaged') has a growing mode on the same
  geometry at the extreme contrast (E5, E6), and the (b′) march shows it from D_hi = 1 (window ratio
  1.31–1.71 per 100 steps at μ 1e3 / ρ 1e4).
- *The argument that failed.* FVS §6 proves only the predictor: X = (M+K)⁻¹(M−T) is self-adjoint and a
  contraction in the (M+K)-norm. The projection P is a contraction in the M-norm. PX is bounded by
  neither when K/M varies by orders of magnitude between coupled unknowns. 2×2 example: M = I,
  K = diag(100, 0.01), T = [[0,1],[1,0]] (K ± T ≥ 0), ‖X‖_M = 1.40. FVS §6 itself lists the composite as
  unproven and left it to large-dt gates. §6 and §6.1 measured walled geometries with one interface, where
  it happened to hold (≤ 0.99). A layered periodic box puts every gas face within a few cells of
  liquid-strength edge coupling (interface edges carry μ_hi-scale β against ρ_lo/dt), and there it does not
  hold. The sharp explicit terms (§6.1 mechanism) add to the same non-normality. That is why 'sharp' goes
  unstable at μ 1e2 contrast where 'averaged' does not.
- *Production regime.* On (b′) 'sharp' passes the march at D_hi = 0.1, 1 and 10 at every pair. That
  includes μ 1e3 / ρ 1e4, where 'averaged' grows from D_hi = 1. P1/P2 run at D_l 0.10–0.32. So in the
  production range 'sharp' is no worse than today, and better at the extreme pair. The defect bites the
  large-D_hi capability that §6 claimed, which was proved for neither mode.

**Rulings.**
- **No dt cap, no tuning** (the §6 rule stands; a cap would need a USER ruling and is not proposed).
- **G-SV9(b′) stays reported-only**, as pre-registered. The STOP is resolved: (b) and (d), the binding
  march gates, pass after WO-5b. Their pass is no longer read as "unconditionally stable": the note's
  claim is narrowed to "stable on the gate geometries; the composite is unproven for both modes"
  (register text §13 to say so).
- **WO-6 proceeds** (GMRES needs only I − G nonsingular; λ = −1.46 gives 2.46). **WO-7 proceeds** at
  production dt.
- **New reported gate G-SV9(f), the onset (WO-6f).** The (b′) march and the E1/E5 Arnoldi, with the
  WO-6a′ deflation, at D_hi ∈ {0.1, 0.3, 1, 3, 10, 30, 100}, both modes, the four pairs, frac 0.5 and 0.75,
  N = 16. Report the smallest unstable D_hi per mode and pair. **STOP → architect if 'sharp' is unstable at
  D_hi ≤ 3 on any case where 'averaged' is stable.** That is the production-relevant claim.
- **The composite instability is a separate package (Q23), not this one.** It is in today's default
  ('averaged', register 272's split), which must stay bitwise here (G-SV5). It needs its own design pass:
  a periodic layered 2-D harness first (the harness is walled today); then a split whose composite is
  provably contractive. Candidates, none decided: the transpose implicit on band rows only (a coupled
  band solve), or an M-norm-contractive predictor. Production relevance is a USER priority call: thin
  gas/liquid layers of 2–5 cells at μ ≥ 1e3, ρ ≥ 1e4 contrasts and D ≥ 1. Default: the coordinator logs it,
  and this package does not wait for it.

### 6.6 Fourth pass: the momentum solve inside the steady gates (Q2)

**Facts.** Under VoF with variable ρ/μ the velocity V-cycle is refused (register 212); RB-GS runs with
the residual stop that follows the pressure rtol, and a stagnation exit. At D_hi = 100 it cannot meet
that stop: the smooth and translation modes of M + K contract by ≈ 1 − 1/(1 + 6D) per sweep, ~2e4 sweeps
per 1e-13 (measured 19 500). It exits at 1.7× its scale. The step map is then neither affine nor
momentum-conserving, and GMRES, which needs an affine map, diverged on G-SV1P (1,1,1) k5/k6 (iterate
×200, CFL refusal). Converged momentum: the same cases in 63 matvecs, one cycle.

**Why converging is not a selective rescue.** The steady state is A u + G p = f (§6.4); the momentum
solver is not in it. It only sets the path, and the converged solve makes the map affine so that GMRES
can find that state. A certified fixed point of the default map is the same state in exact arithmetic
(at a fixed point the warm-start residual is zero), but its certificate is weak: an inexact step barely
moves the smooth modes, so a small increment does not bound their residual. The pilot's ~4-digit
default-vs-converged agreement is that weakness, measured.

**Rule (binding).**
- Every steady gate — G-SV1P/1cP/1dP, G-SV1W, the walled G-SV1 diagnostic, G-SV2 — and the WO-6p
  invariance check run **every** production step of the steady solve (S(0), every matvec, every
  certificate step) with `gsv1.py --converged-momentum`: RB-GS, residual stop off
  (`set_velocity_residual_tolerance(0)`), update stop 1e-13, cap never binding; pressure by the driver
  the ρ model selects, rtol 1e-13, cap never binding. Same setting for both modes, every case, every N.
- **Coordinator decision 2026-11-01 (supersedes the 1e-13 update stop above):** every steady gate runs the
  update stop **1e-15** (residual stop off, pressure rtol 1e-13; a sweep-cap hit is NOT CONVERGED, listed);
  the invariance threshold ≤ 1e-13 is unchanged. At 1e-13 the translation mode leaves ~(1+6D)×stop
  (2.35e-13, gsv1p m100 k3 1e-2 N24); at 1e-15 the pilot gives ≤ 3e-14.
- **Certificate:** on that converged step, ‖S(x) − x‖∞ ≤ 1e-12·max|u| (unchanged).
- **Reported, not judged:** the increment of one production-default step from the certified state.
  Expected ≤ 1e-11·max|u|; a larger value is listed (it would mean the default exit does not even hold a
  steady state).
- Marches (G-SV9(b)(b′)(d), G-SV3, G-SV8) keep the production default: they judge the production path.
  G-SV9(e) keeps its converged solves (1e-13).
- **Reruns.** Every G-SV1P/1dP/1cP result from `R/host2.sh` and `R/gsv1/cuda.sh` taken with the default
  solve is void for judging (kept as a diagnostic); rerun the full sets under the rule. The WO-6p
  invariance pilot (`pilot_conv`) already runs converged and stands.
- **Cost, and the only allowed lever (Q19).** ~2e4 sweeps per matvec at D_hi = 100. D_hi may drop to 10
  for ALL cases of a gate only if a 4-case pilot (both ratios, N = 48) shows ≥ 2× less total sweep count
  and fixed points agreeing with D_hi = 100 to ≤ 1e-11·max|u|. Default: D_hi = 100.

**The stagnation exit in production (coordinator Q20): not this package.** It is documented behaviour
above D ≈ 4 (flow `CLAUDE.md`, velocity solve). P1/P2 run at D_l 0.10–0.32, gas D ≈ 1.5–5 at air–water
contrasts. WO-7's logging is released: per step, the momentum exit residual over its tolerance. If more
than 1 % of P1/P2 steps exit above tolerance, Q20 becomes its own package (a variable-μ velocity solver;
register 212 is the architect's to revisit). Default: proceed. §6.5 adds what the periodic twin says
about it.

### 6.7 Fourth pass: G-SV2 fully specified (Q3)

Replaces the G-SV2 bullet of §11 (the pre-registered parts — κ set, L = 8R, d/h set, the
self-convergence order ≥ 1.8, the 3 % sanity — are unchanged; the rest was unstated).

- **Box and grid.** [0, L]³, L = 8R, triply periodic, isotropic h. d = 2R, d/h ∈ {8, 12, 16, 24, 32},
  so N = 4·d/h ∈ {32, 48, 64, 96, 128}.
- **Drop.** Phase A (C = 1) is the sphere |x − x_c| < R, x_c = (L/2, L/2, L/2): a cell vertex at every N,
  so each level samples the same symmetry class. Rejected: a per-N or random offset, which adds an
  O(h²) configuration noise that varies non-smoothly with N and spoils a 3-parameter fit on 5 points.
- **Colour.** C_i is the exact z-chord average over 32×32 midpoint sub-columns of cell i: the length of
  [z_c − s, z_c + s] ∩ [z_i, z_i + h], s = √max(R² − (x−x_c)² − (y−y_c)², 0), over h. Cells with all 8
  corners inside (outside) are exactly 1 (0). Frozen: reset after every step.
- **σ = 0.** The frozen colour holds the shape; HR is shape-exact for any σ (its normal-stress jump is a
  constant, absorbed by the pressure as a step), and CSF would add spurious currents to U_d.
- **Properties.** ρ = 1 in both phases. Stokes has no inertia and the steady state is ρ-independent
  (§6.4); a density contrast adds a variable-ρ projection that changes nothing at steady state and makes
  the gauge ρ_F-weighted. μ = LinearMix(C): μ_B = μ_out = 1, μ_A = κ μ_out, κ ∈ {1e-2, 1, 1e2}. κ = 1 is
  uniform μ, so 'sharp' is skipped bitwise there: it is the control of the forcing and the base scheme.
- **Force (Boussinesq buoyancy, Hasimoto convention).** Cell field f_z = Δρg (C_i − φ_h), φ_h the mean of
  C_i; f_x = f_y = 0. Write it with `set_field("force_z")` in internal units (`s.unit_scales`); the face
  force is the face interpolation, Σ = 0 to round-off. This is a force ΔρgV_h on the drop,
  V_h = h³ΣC_i, plus a uniform −Δρgφ_h over the whole box: exactly Hasimoto's periodic-array
  distribution, so **F = Δρg V_h**. Rejected: F = (ρ_d − ρ̄)gV = ΔρgV(1 − φ), the net body force on the
  drop region. It differs by O(φ) = 0.8 % and is not the convention of the periodic correction below.
- **Amplitude.** Δρg = 1e-3·(h/dt)·9μ_out K_HR/(2R²): colour CFL ≈ 1e-3 at the unbounded HR speed (the
  problem is linear; the CFL refusal must never fire).
- **Steady solve.** WO-6s GMRES(60) under the §6.6 rule, from u = 0, p = 0. D_hi = μ_hi dt/(ρh²) = 100,
  μ_hi = max(κ, 1)·μ_out. Certificate 1e-12·max|u|; cap 3000 matvecs (at the cap: NOT CONVERGED, not
  judged). Advection off. Modes: 'sharp' (gated) and 'averaged' (reported).
- **U_d.** U_d = Σ_i C_i w̄_i / Σ_i C_i − ⟨w⟩, with w̄_i = ½(w on the −z and +z faces of cell i) and ⟨w⟩
  the mean over all z-faces (zero by momentum invariance at uniform ρ; subtracted anyway). For a drop,
  (1/V)∫_drop u dV is the translation velocity exactly: in the drop frame ∫u dV = ∮x(u·n)dS = 0. The
  discrete functional errs by O(h²) (O(h⁴|∇u|) per cut cell, O(R²/h²) cut cells).
- **K** = F/(6π μ_out U_d R_h), R_h = (3V_h/4π)^{1/3}.
- **PASS ('sharp', κ = 1e-2 and 1e2).** Fit K(h) = K∞ + C h^p by unweighted nonlinear least squares over
  all five levels, p bounded to [0.5, 4]: p ≥ 1.8, and |K∞ − K_per|/K_per ≤ 3 % with
  K_per = K_HR/(1 − 1.7601 K_HR φ^{1/3}), K_HR = (2 + 3κ)/(3(1 + κ)), φ = π/384 (φ^{1/3} = 0.20149).
  K_per = 0.8788 (κ = 1e-2), 1.1829 (κ = 1), 1.5416 (κ = 1e2).
- **κ = 1** must pass the same criteria. If it misses, G-SV2 is NOT JUDGED (instrument or base scheme,
  not the sharp operator) and returns to the architect.
- **Reported, not judged:** K per level; the same fit over the four finest levels (12–32); 'averaged''s
  p and K∞; matvecs per case. A five-level MISS with a four-level pass is Q10 (curvature pre-asymptotic at
  d/h 8): reported, and the user rules; it is not a PASS.
- **Where.** d/h 8–16 host; d/h 24, 32 CUDA under `heavy.lock`. Script `tests/study/sharp_viscous/gsv2.py`,
  one line per case plus a verdict line, re-using `gsv1.py`'s GMRES driver.

### 6.8 Fifth pass (architect, 2026-11-04): three instruments that could not deliver a verdict

Brief: `validation/highratio/BRIEF_ARCH_SHARPVISC5.md`. Discriminating runs on `build_host_rb` (no rebuild):
scripts and logs in `validation/highratio/sv_impl/arch5/` (`exp_a.py`, `exp_b.py`, `exp_c*.py`, `*.log`,
`b*.jsonl`). **Verdict.**
- **A and C are instrument faults, settled by measurement.** Each gets a corrected instrument below. A
  also undoes a coordinator rule that the steady gates' FINAL verdicts silently depend on.
- **B: the instrument is faulty (measured), but whether the operator also contributes is OPEN.** It
  waits on two N48 runs that were running at commit (`arch5/b.jsonl`, `b2.jsonl`). The default is
  stated at the end of B.
- The pass was stopped at the coordinator's context limit.

#### A. G-SV9(e): the momentum stop, not the momentum solve

**Facts (EXP-A, `arch5/a_d100.log`, `a_d10000.log`).** The pilot case (walled plane, N16, sharp,
μ 1e-3 / ρ 1e-4, incremental, Chebyshev 1e-13), one matvec. The inputs are the pilot's first random
vector x1 and x_s, which is 30 power steps from x1 and stands in for the Arnoldi's converged vectors.
The reference is a FIXED 64000 sweeps per component, which repeats to 9.7e-14 (u) and 1e-16 (p).
Relative max-norm difference to the reference, D_hi = 1e4 (D_hi = 1e2 gives the same picture):

| momentum setting | sweeps (3 comps) | CPU s | x1: u / p | x_s: u / p |
|---|---|---|---|---|
| fixed 500 / component | 1500 | 0.07 | 3.6e-10 / 3.0e-11 | 1.3e-11 / 2.4e-14 |
| fixed 1000 | 3000 | 0.12 | 8.5e-14 / 1.2e-16 | 3.1e-14 / 7e-20 |
| fixed 2000 … 32000 | 6000 … 96000 | 0.22 … 3.3 | 9.7e-14 / 2e-16 (= the floor) | 3e-14 / 6e-19 |
| update stop 1e-13, cap 2e5 (**the pilot**) | 400585 | 14.3 | **4.2e-12 / 4.7e-13** | 1.6e-12 / 1.7e-15 |
| update stop 1e-13, cap 2e4 | 40585 | 1.46 | 4.2e-12 / 4.7e-13 (identical) | 1.6e-12 |
| update stop 1e-11 / 1e-9, cap 2e5 | 400462 / 400339 | 14.3 | 4.2e-10 / 4.2e-8 | 1.5e-10 / 1.5e-8 |

**Diagnosis.**
1. *Every component reaches its round-off floor within ~1000 sweeps, at D_hi 1e2 and at 1e4.* The (e)
   box is walled on all faces, so the slowest mode of M + K is Dirichlet-bounded. Once D(πh)² ≫ 1, the
   Gauss–Seidel contraction per sweep is 1 − O((πh)²), whatever D is. §6.6's (1+6D) estimate applies to
   the translation modes of a PERIODIC box. It does not apply to (e), so D-independence is expected.
2. *The update stop is not a convergence criterion* (`flow_ibm_core.hpp:1576–1590`: stop when
   `du ≤ velTol·du0`, where du is the max increment of the colour-1 half-sweep and du0 is its value on
   the first sweep). The level it can reach is floor/du0, and that level is a property of the warm start,
   not of the solution.
   - Two components never meet 1e-13·du0. They sit at the floor from about sweep 1000 and then run the
     other ~199 000 sweeps to the cap: the cap is 98 % of the cost (caps of 2e4 and 2e5 give identical
     outputs). That is the "D-independent 14.3 s per matvec".
   - The third component meets the stop after ~585 sweeps, before it has converged. The increment
     under-reads the slow-mode error by about 40×, so its matvec error is 40 × tol (4.2e-12, 4.2e-10,
     4.2e-8 at tol 1e-13, 1e-11, 1e-9). That is 40× above the floor, and it is set by the stop itself.
3. *Consequences for the pilot.* Its map was accurate to ~4e-12 (u) and ~5e-13 (p), and linear to
   2.8e-13. The cap verdict NOT_JUDGED is a false alarm: the components that capped are the converged
   ones. No property of the operator is involved.

**Is ρ = 0.999726 (lin 2.8e-13, conv 1) enough on its own?** The evidence is strong. The margin is
2.7e-4 and the map perturbation is ≤ 4e-12, so crossing 1 would need an eigenvalue condition number
above ~1e7. But the run was taken under a stop the instrument violates by construction, and the rerun
under the corrected instrument costs ~3 h. **Rule: judge from the rerun, not from this argument.** The
old pilot line stays as a diagnostic.

**Corrected (e) instrument (binding once USER Q24 accepts it; default: accept).**
- *Momentum:* RB-GS with a FIXED count, N_A = 2000 sweeps per component: `set_velocity_solver_params(2000,
  0.0, 2)`, residual stop off. This is twice the measured convergence count. A fixed-count Gauss–Seidel
  is an exactly linear map, with no stopping decision, so the linearity check measures round-off and
  nothing else.
- *Certificate per run, before ARPACK.* It replaces the momentum half of "whose solves cap":
  - (i) linearity ≤ 1e-10 (unchanged);
  - (ii) doubling: for x1 and x_s (30 power steps), ‖y(2N_A) − y(N_A)‖∞ ≤ 1e-12·‖y(N_A)‖∞, separately for
    u and for p.
  - If (ii) fails, double N_A once and repeat. A second failure is NOT JUDGED (listed). Print both
    doubling numbers on the PROD line.
- *Unchanged:*
  - Chebyshev 20000 / 1e-13. A pressure-cap hit is still NOT JUDGED; the pilot measured max_p_it 16.
  - ARPACK ncv 40, maxiter 1000, re-run at ncv 80.
  - PASS ρ ≤ 1 + 1e-9 for 'sharp', with 'averaged' reported.
- *Cost:* 6000 sweeps (0.22 s) plus the Chebyshev solve, about 0.3 s per matvec at N16, against 14.3 s
  now. The pilot (~3.6e4 matvecs) takes ~3 h. The 80-run table runs on 8 host processes in ≲ 1.5 days.
  Order: the pilot first; then the plane cases, then the film cases.

**The same rule in the steady gates (affects FINAL verdicts: USER Q25).** The coordinator rule of
2026-11-01, "a sweep-cap hit → NOT CONVERGED, listed", is implemented neither by `gsv1.py table` nor by
`w6judge.py`. These judged rows have a component at the 200000 cap:

| gate | capped rows |
|---|---|
| G-SV1P | 71 / 432 |
| G-SV1dP | 38 / 400 |
| G-SV1cP | 59 / 510 |
| G-SV1W | 4 / 4 |
| G-SV2 | 15 / 25 |

As written, the rule would void G-SV1W and most of G-SV2, both FINAL PASS in the interim verdict. EXP-A
shows that the rule is the thing at fault, not the table:
- A capped component is at its floor. The cap costs time, not accuracy.
- A steady certificate is a plain step warm-started at the iterate. Its momentum solve therefore starts
  within its own residual of the answer and cannot make an unconverged state look converged.
- What remains of the stop's error is ~40 × 1e-15 relative on the component that stops early. That is
  ten orders below the errors being judged.

**Ruling:** retire the cap rule for the steady gates. The pre-registered certificate governs,
unchanged. The momentum cap count stays a reported column. WO-A5b confirms the ruling on periodic
geometry, where the translation modes are slower (§6.6).

#### B. G-SV1cP: the forcing's representation, not the operator

**The manufactured problem is consistent.**
- div u = 0, and μ_kΔu − ∇p_k + μ_kF = 0 in each phase.
- On Γ (cos bs′ = 0): S_nt = 0 and −p + 2μD_nn = 0 on both sides (`gsv1.py check` agrees).
- The net force is exactly zero, in the continuum and on the lattice.
  - μ(C_i)·cos(bs′_i) is a function of m·i mod N, and sin(aη_i) is a wavenumber-1 sinusoid in t·i mod N.
  - The map i ↦ (m·i, t·i) mod N is equidistributed up to characters whose order divides gcd(m×t) ≤ 2.
    No such character cancels a wavenumber-1 sinusoid for N > 2.
- The measured 4.4e-9 (`force_sum_rel`, max over cells at N48 (1,1,1)) is the coherent round-off of the
  inclusion–exclusion `cube_frac`. That function returns fractions 1 − 6e-12 for full cubes (measured),
  with the same sign across a band of ~5e3 cells, weighted by [μ]F. It contributes ≲ 1e-12 per step to
  the invariance column. It is not the O(1) error.

**What is wrong (fact).** f = μ_kF jumps by [μ]F_t at Γ, where F_t = ±b(3a² + b²)cos(aη) is at its maximum.
- The instrument writes the cell field μ(C_i)F(x_i). The solver puts it on a face as ½(f_{i−1} + f_i),
  so μ is weighted by the 2h-wide mean (C_{i−1} + C_i)/2.
- The face's control volume holds μ at its own fraction C_CV. On band faces the two differ by
  O([μ]F): measured at 25 % (m (1,1,0)) to 84 % (m (2,1,1), N8) of max|f|.
- Along a normal line the error has zero sum, so it is a force-dipole layer of moment O(h²[μ]|F|). That
  gives a velocity error of O(h²|F|/k): relative ~4(kh)² with |F| ≈ k²|u|·(3a² + b²)/(a² + b²). It is
  second order with an O(1)–O(4) constant, 0.96 at N24 for (1,1,0).
- It depends on how the sub-cell configuration cuts the cells, hence the 20× spread over the eight
  offsets.

**"Identical at 1e-2 and 1e2" is a symmetry of the instrument, not evidence.**
- With φ_A = ½ and f ∝ μ, swapping the phases is a half-period shift along n. That shift equals a
  whole-cell translation by (N/2)e_q for any q with m_q odd, together with u → −u.
- Ratio r → 1/r is that swap combined with a constant rescaling of μ, which leaves u unchanged.
- So the 1e-2 and 1e2 rows are one problem. They agree to every printed digit except at k = 0, where
  faces lie on Γ and the tie is broken differently (0.0453 vs 0.0383 for (1,0,0)).

**Discriminator (EXP-B, `arch5/b.jsonl`, `b2.jsonl`).** m (1,1,0), k = 3, ratio 1e2, the worst row.
- The consistent force is the face-control-volume average of μ(x)F(x) over L³ sub-cubes per CV, with
  exact sub-cube fractions from a residue table. It is turned into cell fields by inverting the solver's
  face mean along each line.

| variant | N24 | N48 |
|---|---|---|
| instrument as is | 0.956 | 0.302 |
| uniform μ, no jump at all (gauged error) | 0.035 | not run |
| consistent force, L = 2 / 4 / 8 | 0.1509 / 0.1527 / 0.1532 | L = 4: running (`b.jsonl` line 6) |
| consistent force L = 4, mode 'averaged' | 0.349 | — |
| m (2,1,1) k 3: as is / uniform μ / consistent L = 4 (stiff placement; uniform spread 0.238) | 0.121 / 0.059 / **0.230** | consistent: running (`b2.jsonl`) |
| m (2,1,0) k 3: as is / consistent L = 4 | 0.113 / **0.270** | — |

**Reading (B is OPEN).**
- *(1,1,0).* The consistent force removes 84 % of the error (0.956 → 0.153), and the result is
  converged in L.
- *(2,1,1) and (2,1,0).* It roughly doubles the error (0.12 → 0.23, 0.11 → 0.27). So the error left is
  NOT the smooth-truncation level (ctrl 0.035 / 0.059) on any of the three normals.
- *Two explanations survive:*
  - (a) The operator's own error at the D_nn jump. Its sign relative to the old forcing error happened
    to cancel on (2,1,x).
  - (b) The "consistent" target is not the force the sharp operator's control volume actually balances,
    for example if its CV moment or its pressure sampling differs from the plain staggered CV.
- 'averaged' with the same force gives 0.349 on (1,1,0), against 0.153 for 'sharp'. Sharp is better, but
  the gap does not separate (a) from (b).
- **The run that settles it** is the N48 pair now running: consistent L = 4 at (1,1,0) and (2,1,1), k 3.
  An observed order (N24 → 48) ≥ 1.8 on both means (a) is a pre-asymptotic constant: adopt WO-B5 and
  judge. An order ≤ 1.5 on either means (a) or (b) with first-order content: STOP → architect, with the
  operator's per-face residual of the exact along mode as the next instrument.
- **Default meanwhile:** G-SV1cP stays NOT JUDGED and its N64 stays HELD. The facts below stand either
  way: the forcing error, the r ↔ 1/r symmetry, the net force, the invariance.

**Corrected G-SV1cP instrument (WO-B5; conditional on the N48 pair above; USER Q27, a post-data
change of the force input; default: accept if the N48 pair shows order ≥ 1.8).** The target is a face force consistent with the face control volume, for both kinds.
- Fractions: from the residue table of the sub-grid. It is exact; equal residues get bit-equal values,
  so the lattice cancellation of the net force holds exactly.
- *Target face force,* per component c and face F:
  - g_F = (1/L³)Σ over the L³ sub-cubes of the CV of [μ(C_s)F(x_s) + (ρ(C_s) − ρ̄)g t1_c], with L = 4.
  - The CV of face i along c covers sub-indices [L·i − L/2, L·i + L/2) along c, and [L·j, L·j + L) across.
  - The layered term is present only in 'along+layered'. ρ̄ is the exact volume mean.
  - Measured L-dependence: 0.3 % in err between L = 4 and 8.
- *Inversion* of the solver's face mean ½(f_{i−1} + f_i) along every line (periodic, N even):
  - The map is singular on the alternating mode, so each line's alternating sum S = Σ_F(−1)^F g_F is
    first removed from the target, as r_F = α(−1)^F w_F + β w_F.
  - w_F = 1 on faces whose CV lies fully in the high-μ phase (C_CV ≥ 1 − 1e-9, or ≤ 1e-9 when μ_B is
    the high one); α and β are fixed by Σ(−1)^F r_F = S and Σr_F = 0.
  - Then f = IFFT(FFT(g − r)/(½(1 + e^{−2πik/N}))) along the line, with the Nyquist coefficient zero.
- *Why stiff faces:* the residual (measured 0.5–1.3 % of max|f| at N24) then acts on the stiff phase,
  where its response is μ_lo/μ_hi smaller.
  - Spreading it uniformly instead made (2,1,1) worse (0.238 at N24): the residual then drives the weak
    phase.
  - Every line has fully stiff faces for the nine normals at N ≥ 24 (checked at N24 for four of them).
  - A line without any is an error, not a fallback.
- *Reported per row:* max|r|/max|g|, and max over components of |Σ_F f_F|/max|f| (expected ≤ 1e-13).
- *Invariance (WO-6p column) for 1cP:* judged on the certifying step's output, |Σρ_F u_c|/(Σρ_F max|u|) ≤
  1e-13. The max over all restart steps is reported.
  - Reason: from x0 (p = 0, far from the along-mode pressure) the first restart's du0 is O(U). Its
    update stop then leaves a translation error of ~(1+6D)·40·1e-15·du0 ~ 1e-12. That measures the stop,
    not the step's invariant; it is the 201/510 MISS.
- Exact solution, gauge, N set, offsets, normals, PASS: unchanged. All old 1cP rows are void for judging
  (kept as diagnostic). G-SV1cP N64 stays HELD until WO-B5's host rows are in.

#### C. G-SV1dP μ 1e3: the refusal is the step's OUTPUT, advected at the tail

**Facts (EXP-C, `arch5/c3.log`, m (2,1,0), k 2, N24).**
- The head pre-check sees the input. Every Krylov input has CFL 0.0088, the exact solution's (the
  Krylov scaling bounds max|u_in| by U).
- The refusal comes at Krylov vector j = 19 of cycle 0, from `WyAdvector::enforceCfl` inside
  `advectVof()` at the tail of the step, i.e. on the projected output. The message lacks the head
  check's "Rejected at the HEAD"; the solver velocity after the throw is max 274 = 65 U.
- The step amplifies the Krylov vectors' pressure part (internal units) into velocity: max|y_u|/U is
  5–14 for j = 15–18, and 65 at j = 19 with max|p_in|/U = 0.76.
- The colour advected at the tail is discarded by `set_vof(C0)`. **The refusal carries no information
  about the operator or the fixed point.**

**Rule (one for all rows; an instrument repair, no gate change).**
- Every evaluation of S inside the GMRES driver (S(0), Krylov matvecs, restart steps) that the
  colour-CFL guard refuses is re-evaluated by affinity at the reduced amplitude σ′ = σ·0.01/CFL_out.
  CFL_out is `s.vof_max_courant()` read right after the throw, which then reflects the refused step's
  output velocity.
- The step is set entirely by the (u, p, C) the StepMap writes, plus warm starts converged to 1e-13, so
  the retry is exact to round-off.
- The **certifying step stays a plain, unscaled production step**. If it refuses, the row is NOT
  CONVERGED (listed): the fixed point itself would then be over the cap.
- The rule is a bitwise no-op for every evaluation that did not refuse. The 400 converged G-SV1dP rows
  and every G-SV1P/1cP row are unchanged. Only the 32 refused rows (and any N64 row that refuses) are
  rerun.
- *Rejected:*
  - Freezing the colour: it already is frozen. The guard protects an advection whose result the
    instrument throws away, and there is no production lever short of a production change.
  - Raising `set_vof_cfl_limit`: a production setting, and the momentum-consistent advector refuses
    ≥ 0.5.
  - Lowering the amplitude for all rows: not bitwise, so every row would be rerun.
  - Equilibrating p in the Krylov basis (Q30): it would also cure the 65× gain, but it changes every row.

**Evidence that the rule suffices (EXP-C2, `arch5/c_scaled*.log`).** Two rows, run with the rule:
- (1,0,0) k 3, 1e3, N24 (refused in the gate): two Krylov evaluations rescaled (CFL_out 0.276, 0.259).
  It converged in one cycle, inc 3.2e-14, err 0.0138, in line with the non-refused offsets (≤ 0.0357).
- (2,1,0) k 2: no further refusal. Restarted GMRES is converging slowly, inc 1.5e-8 → 3.7e-11 over
  cycles 1–3, and was still running at commit (`c_scaled.log`).
- If it reaches the 3000 cap, it is NOT CONVERGED under the rule, and Q30 (p-scaling) is the lever for
  the whole set.
- **C is settled.** The refusal is the tail advection of a non-physical output, and the retry rule
  removes it.

#### D. Sixth pass (architect, 2026-11-04): Q26, the along-mode order — PARTIAL, verdict OPEN

**Instrument (new, cheap, production path).** `arch6/resid.py` (dir `bubble_column_perf/validation/highratio/sv_impl/arch6/`): ONE production step from u^n = ū (exact face
averages), p^n = 0, the consistent force (L = 4, stiff), at D = 1e-3. Then r = ρ(u^{n+1} − ū)/dt = P_h[(A+T)ū + f̄]
+ O(D). This is exact bookkeeping: the steady velocity error is −L⁻¹r, and −G p̄ drops out (P_h G = 0), so r is the
unique, gauge-free source of the error. Band faces (|ξ| ≤ 3h) are reduced to Fourier-weighted moments per unit
interface area: M0 = Σ r w h³/A (net force), M1 = Σ r ξ w h³/A (dipole), w = 2cos(aη) or 2sin(aη), in (n, t).
Scales: μ_hi U a and μ_hi U. A conservative band flux error δσ gives M0, M1 ∝ δσ h, so O(h²) velocity needs both at
O(h²). Checks: D-linear to 0.3 % (1e-3 against 1e-4), force residual 2e-16, fit fallbacks 0 at every N.

**Measured, m (1,1,0), k 3, ratio 1e2, N 16/24/32/48/64** (`arch6/resid.jsonl`, `summ_resid.py`):

| quantity | sharp | averaged | uniform μ |
|---|---|---|---|
| bulk max r / (μ_hi U a²), N24→64 | 5.1e-2 → 7.3e-3, order 1.96–1.99 | 5.9e-2 → 8.3e-3, 1.98–1.99 | order 1.99 |
| M0_t (cos), N64; orders | 4.7e-3; 2.2–2.8 | 4.2e-3; 2.2–2.9 | 1.9e-2 (N48); 2.2–2.8 |
| M1_t (cos), N64; orders | 7.4e-3; 2.1–2.4 | 2.0e-2; 2.0–2.1 | ≤ 5e-4 |
| M1_n (sin), N64; orders | 5e-6–1.4e-4; ≥ 4.7 | 2.4e-4; 4.8–5.0 | order 4.8 |
| **M0_n (sin), N24/32/48/64** | **1.0e-2 / 4.7e-3 / 6.3e-3 / 4.6e-3; orders 2.6, −0.7, 1.1** | 3.9e-2 / 3.8e-2 / 2.3e-2 / 1.5e-2; orders 0.1, 1.2, 1.6 | 9.8e-4 / 3.5e-4 / 7.4e-5; 3.6–3.8 |
| band max r / (μ_hi U a²) | 0.51 flat (O(1); divergence form, expected) | 0.65 flat | 2nd order |

**Reading.**
- *(i) is excluded* on (1,1,0): force exact, no fallbacks, step linear in D, and the bulk residual is second order.
- *Shear part sound.* Every tangential moment and the normal dipole converge at order ≥ 2. So the face-fraction
  shear samples and the τ̂ T-terms behave as §5.5 claims.
- *One anomaly, and it is the only candidate for (iii).* Sharp's net normal force per interface area, modulated
  like D_nn (sin aη), stops converging at ≈ 5e-3 μ_hi U a for N 32–64. The uniform-μ control has 7e-5 at order 3.8;
  'averaged' has 1.5e-2, still falling at order ≈ 1.6. This is a modulated normal force sheet, and a sheet is not
  a gradient, so it drives velocity at O(h⁰·F/(μa)) asymptotically. It belongs to the normal-stress path: the
  cell sample's 2μ(C̄)m_cc, the midsection term with D̂_nn, or the normal component of the edge samples at an
  oblique plane. Its size (5e-3 against err 0.05 at N48) is too small to explain the N24→48 order 1.57 alone, so
  (ii) and (iii) both survive on (1,1,0).
- *Not yet in:* the (2,1,1) residual rows (running, `resid.jsonl`) and the steady N16/32/64 sharp rows plus the
  uniform-μ N32/48 and 'averaged' N24/48 rows (`arch6/steady.jsonl`, via `arch5/exp_b.py`; N64 ≈ 2 h).

**Decision rule for the coordinator, applied to those files (no new runs needed):**
1. Steady order, sharp N32→64, both normals ≥ 1.8 → **(ii)**. Adopt WO-B5 and judge G-SV1cP on the N32/48/64
   envelope (N24 is pre-asymptotic at kh 0.37/0.64). Q26 closes.
2. Steady order N32→64 < 1.5 on either normal AND sharp's M0_n flat (|order| < 0.5 over N32–64) on that normal
   → **(iii)**, in the normal-stress path. The next instrument is a term split of r: a study-only switch replacing
   D̂_nn by the exact D_nn(x_i) in the midsection term, and one dropping it. That is a study hook in production
   code, so it is the coordinator's call. Until then it is NOT a redesign: the shear path is verified, and a defect
   confined to one cell-centred term is a local fix.
3. Otherwise (1.5 to 1.8) → NOT JUDGED. Report the four-point orders. The default below stands.

**Default meanwhile (recommendation): ship 'sharp' as the staggered AUTO default now, with this mode's order
documented as "1.4–1.6 measured at N24→48, under study".** Reasons:
- On the same instrument sharp beats 'averaged' everywhere: steady err 0.153 against 0.349 at N24; normal sheet
  5e-3 against 1.5e-2 at N64; dipole M1_t 2.7× smaller.
- 'averaged' carries a first-order normal sheet of its own.
- Every other gate is second order.

So there is no measurement on which 'averaged' is the safer default for this mode.

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

- **Setter (§12 Q3; LANDED 2026-11-04, USER: sharp is the default).**
  - `s.set_vof_viscous_interface("auto" | "sharp" | "averaged")` on the public tier, default "auto";
    getter `s.vof_viscous_interface()`. The build-time `s.diagnostics.set_vof_viscous_interface`
    never shipped and is gone (no alias, QUALITY_PLAN D2).
  - AUTO resolves at each step head to "sharp" when VoF is on, μ = LinearMix(C) varies, full stress
    is active and neither phase change, porous momentum, an immersed solid nor moving scene instances
    are present; else to "averaged" with a one-time stderr notice. `SolverColocated` resolves AUTO to
    "averaged" (§8).
  - "averaged" reproduces the pre-sharp numbers bitwise.
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
    - The oracle is `mac2d.py` with `TAU_RULE = "mixed-avg"`, the §5.7 rule. `mac2dc.py`'s planar path
      gives the same values bitwise.
  - Exact τ and D_nn for kinked-linear fields at 9 orientations, to ≤ 1e-11.
  - Fallback count 0 on all of them.
  - Virtual planes (§5.7 Q-B):
    - A 1-cell film along each axis, and an ABAB stack: every edge on a film face is owned at distance 0.
    - The plane (m, α) follows the low-face rule.
  - Ties in offset order with a periodic seam inside the band: owners and τ̂ are bitwise equal at np 1
    and np 2.

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

**WO-5b. The split on interfacial cells (third pass, §6.1). Do it before any further WO-6 run.**
- **Change.** `SharpFaceProps` (`src/vof/sharp_viscous.hpp:405-430`) gets a role, implicit or explicit, set
  by `makeSharpFaceProps(c, role)` (`src/flow_ibm_closures.hpp:766`). In `beta(i, j)`, on the cell link
  (`ad == geo.st[c]`), let q be the cell between the two u_c faces, the lower of i and j. If
  `geo.cellClass(q) ≠ kSvNone`:
  - the implicit role returns 2·`base.beta(i, j)`;
  - the explicit role returns 0.

  Every other link and every non-interfacial cell returns what it returns today.
- **Call sites.**
  - Implicit role: the two band builds, `src/flow_ibm_core.hpp:1053` and `:1434`. If the velocity
    V-cycle derives coarse operators from these props, use the implicit role there too. The coarse operator
    is a preconditioner, so this moves iteration counts only.
  - Explicit role: `addFullStressRhs<vof::SharpFaceProps, false>` at `src/flow_ibm_closures.hpp:263-264`.
  - `addSharpViscousRhs` is unchanged: the midsection and τ-terms stay explicit.
- **Invariants.**
  - A is unchanged: implicit + explicit = 2μ(C) m_cc on every cell, exactly as before.
  - `beta` stays symmetric under the (i, j) swap.
  - The classes are already filled over the whole block, ghosts included (`sharpClassify`), so np 1/2/4
    stay bitwise.
  - 'averaged' never constructs `SharpFaceProps`, so it is unaffected.
- **Acceptance.**
  1. `gsv9a.py` builds K = Kb, via `gsv9_split.py`'s `harness_case`. G-SV9(a): worst ρ ≤ 1 + 1e-12 at
     N = 16 and 32 (N = 16 measured 0.9859). The K1 row is printed as a control and must stay > 1 at
     D_hi = 1e4, proving the harness models what it claims.
  2. Production, `gsv9_split.py prod --geom ext --N 16 --Nz 4 --pdriver cheb --D 1e4`, pairs 1e-3 / 1e-4 and
     1e3 / 1e4. Sharp ρ must be ≤ 1 and within 0.01 of harness Kb for (22.5°, 0.3): 0.9347 and 0.9307.
  3. G-SV9(e) PASS (§11).
  4. The operator is unchanged:
     - WO-4's exactness subtest stays at its value (7.6e-12);
     - one G-SV1P pilot steady state, before against after, agrees to ≤ 1e-11·max|u|. The fixed point is
       the same; only the path to it changes.
  5. G-SV5 and G-SV11 bitwise rows pass; the `sharp_viscous_mpi_np{1,2,4}` ctest stays bitwise.
  6. If G-SV5's "uniform μ bitwise" row breaks, gate the new branch on μ_A ≠ μ_B. Never relax the row.

**WO-6. Operator gates in 3-D.** G-SV1P, G-SV1cP, G-SV1dP, G-SV1x, G-SV1W and G-SV9 (§11), as ctests or
`tests/study` scripts with one-line summaries. Third-pass parts, in order:
- **WO-6a (`gsv9.py`).**
  - Delete the `set_pressure_pcg(True, 800, 1e-12)` override, so that the Chebyshev driver chosen by the ρ
    property model runs, with rtol 1e-12 and a cap that never binds.
  - Each run line reports the driver and the maximum iterations. A capped step makes the run NOT JUDGED.
  - Print both growth verdicts: as written, and with the floor excluded (§6.2).
  - Add the periodic twin (b′): normal (2,1,1), offset 0.3 cell, reported only.
- **WO-6e (G-SV9(e)).** A one-line-per-run wrapper over `gsv9_split.py prod --incremental --pdriver cheb`,
  with the case table of §11.
- **WO-6s (steady driver, §6.4).**
  - In `gsv1.py`, replace the plain march by GMRES(60) on the fixed-point residual of the production step
    over (u, v, w, p).
  - p goes through `get_field` / `set_field("p")`, with its mean removed.
  - Start from the exact face averages with p = 0. Certify on a plain step: increment ≤ 1e-12·max|u|.
    Cap 3000 matvecs.
  - First a 10-case pilot (N = 24 and 48, both ratios, 3 normals). Record the matvec counts and the march
    agreement: on 2 of the cases, the GMRES fixed point against a plain march to the same criterion must
    agree to ≤ 1e-11·max|u|.
- **WO-6p (periodic drivers).** `gsv1.py --set gsv1p | gsv1cp | gsv1dp`, as specified under G-SV1P (§11).
  - Acceptance: the manufactured fields satisfy the continuous equations and the jump conditions to
    ≤ 1e-12 at 100 random points.
  - Σ_faces ρ_F u_c stays ≤ 1e-13·Σρ_F·max|u| over a run.
  - The layered profile reproduces the analytic U to round-off.
- **WO-6x.** Confirm that WO-4's subtest checks extension together with the exact cell-averaged pressure:
  A ū + G p̄ = 0 on rows ≥ 3 cells from every wall. If it does not, extend it (G-SV1x).
- **WO-6w (G-SV1W).** The walled Couette check of §11, both modes, N = 24 and 32. Steady states by WO-6s.

**Fourth-pass work orders (§6.5–§6.7), in order.**
- **WO-6q (rerun under the §6.6 rule).** `gsv1.py`'s GMRES driver takes `--converged-momentum` for
  S(0), every matvec and the certificate step, unconditionally in every steady set (remove the option's
  default-off path for gated sets). Add the reported default-step increment. Rerun G-SV1P/1dP/1cP
  (N 24/32 host, 48/64 CUDA) and G-SV1W. Acceptance: the WO-6p invariance Σρ_F u_c ≤ 1e-13·Σρ_F·max|u|
  on every case; no case judged from a default-solve run. **Amended 2026-11-02 (USER, post-data):**
  rows the GMRES driver hands to the march fallback are EXCLUDED from the 1e-13 criterion and listed
  instead (the march, stop 1e-15 over ~1500 steps, accumulates the translation-mode drift ~(1+6D)×stop
  per step: an instrument property). The GMRES rows judge the operator: WO-6p PASS 12/12
  (≤ 3.0e-14); the two march rows read 1.0e-11 and 4.5e-12.
- **WO-6a′ (periodic energy).** `gsv9.py` on a periodic box judges E′ (§6.5 finding 1) and prints E;
  `gsv9_split.py prod` on a periodic box M-orthogonally projects the three translations out of every
  Arnoldi vector. Acceptance: E8 (`--frac 0.25`) then reports ρ = 0.969 ± 1e-3 instead of 1.0000;
  the (b′) D_hi = 10 'sharp' lines no longer plateau.
- **WO-6f (G-SV9(f), the onset; reported).** As specified in §6.5. One line per case, plus a table of
  the smallest unstable D_hi per mode × pair × frac.
- **WO-G2 (G-SV2).** `tests/study/sharp_viscous/gsv2.py`, as §6.7; re-uses `gsv1.py`'s GMRES driver.
  Acceptance before the gate: at κ = 1, d/h 8 and 12, Σ force = 0 to 1e-14 relative, momentum
  invariant ≤ 1e-13, CFL ≤ 1.1e-3.

**Fifth-pass work orders (§6.8; study scripts only, no production code; in this order).**

- **WO-A5 (G-SV9(e), fixed-count momentum).** Files: `gsv9_split.py prod` and `gsv9e.py`.
  - *Option.* `gsv9_split.py prod --mom-fixed N` sets `set_velocity_solver_params(N, 0.0, 2)` with the
    residual stop off. `gsv9e.py` passes `--mom-fixed 2000` and drops `--mom-tol`.
  - *Doubling certificate (§6.8 A (ii)).* Run it before `eigs`, on x1 and x_s (30 power steps), at N and
    2N. Print `dbl_u=… dbl_p=…` on the PROD line.
  - *Verdict logic.* Retire the momentum half of the cap test; keep the pressure half (`pit ≥ 20000` →
    NOT JUDGED). Doubling > 1e-12 → rerun at 2N once, then NOT JUDGED.
  - *Acceptance:*
    - On the pilot case: dbl_u, dbl_p ≤ 1e-12 (EXP-A: ≤ 9.7e-14), and per-matvec CPU ≤ 0.5 s.
    - Linearity ≤ 1e-10.
    - `exp_a.py`'s fixed-2000 row is reproduced to 2 digits.
  - *Then:* the pilot (`gsv9e.py --pilot`), then the full table. The old pilot line is a diagnostic.
- **WO-A5b (steady gates: the cap rule, spot check).**
  - *Table.* `gsv1.py table`, `gsv2.py table` and `w6judge.py` print a momentum-cap column: the count of
    rows with `max_momentum_sweeps ≥ 200000`, reported and NOT voiding.
  - *Spot check.* `gsv1.py`/`gsv2.py` take `--mom-fixed N` (same meaning as WO-A5). Rerun four capped
    rows with `--mom-fixed 20000`; on a periodic box the translation modes need ~1.1e4 sweeps at D_hi 100
    (§6.6):
    - G-SV2 κ 1e2, d/h 8;
    - G-SV1P (1,1,0) 1e-2, N24, one capped k;
    - G-SV1dP (1,1,0) 1e-3, N24 (406671 sweeps);
    - G-SV1W N24 'sharp'.
  - *Acceptance:*
    - |Δerr|/err ≤ 1e-8 (G-SV2: |ΔK|/K ≤ 1e-8) and the certificate ≤ 1e-12 on all four.
    - A larger difference is a STOP → architect: the caps would then matter, and Q25's ruling falls.
- **WO-B5 (G-SV1cP consistent force; only after the §6.8 B N48 pair shows order ≥ 1.8).** File: `gsv1.py`'s `PCase.build` for 'along' and
  'along+layered', as §6.8 B. `arch5/exp_b.py` (`consistent_force`, `fine_fractions`, the `stiff` branch)
  is the reference implementation; port it, do not re-derive it.
  - L = 4, and the per-row reports listed in §6.8 B.
  - *Invariance:* judged on the certifying step, with the max over restarts reported.
  - *Acceptance:*
    - `gsv1.py check` grows a `cvforce` line. It requires:
      - the L = 4 vs L = 8 face force ≤ 2e-2·max|f| at N24 for (2,1,1) k 3 (the band term is O(h/L²));
      - Σ_F f_F ≤ 1e-13·max|f|·N³ per component;
      - every line has fully stiff faces, at all nine normals × N ∈ {24, 32, 48, 64}.
    - EXP-B's (1,1,0) k 3 N24 r 1e2 row is reproduced: err 0.1527 ± 0.001.
  - *Then:* rerun G-SV1cP N24/32/48 on host (both kinds, both ratios, the eight offsets), and N64 on CUDA
    under `heavy.lock`. Judge as pre-registered (envelope LS order ≥ 1.8).
- **WO-C5 (GMRES refusal retry).** File: `gsv1.py` `gmres_steady` (and the copy inside `gsv2.py`, if it
  has one): the §6.8 C rule.
  - Per row, report `rescaled` (count) and `max_gain` (max CFL_out/CFL_in over evaluations).
  - *Acceptance:*
    - (a) A row that never refused — G-SV1dP (1,0,0) k 0, 1e3, N24 — reproduces its jsonl err, inc and
      steps bitwise.
    - (b) On a refused row, the rescaled evaluation matches the same scaled input evaluated with no
      preceding refusal to ≤ 1e-13 relative.
  - *Then:* rerun the 32 refused G-SV1dP rows (N24–48 host). N64: apply the rule to the in-flight CUDA set
    only for rows that refuse.
  - A row that reaches the 3000-step cap is NOT CONVERGED (listed); its normal × ratio is NOT JUDGED at
    that N, as now.

**WO-W (parked, not scheduled; it comes into scope only if G-SV1W misses).** At a non-periodic Dirichlet
face:
- the wall-normal shear link of a wall edge is the half-link [wall plane, first node], so θ̄ and R̄ take the
  half-box kernel;
- the fit uses that link as data, with that kernel;
- owners are drawn from in-domain cells only.

A design pass comes first, because box edges and corners need their own rule.

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

**WO-9. Owner set and mean (seventh pass, §5.8).** Requested as "WO-8"; renumbered because WO-8 above is
Docs. Branch `sharpvisc-land`; host tree first, CUDA under the GPU lock. Commit-sized steps in order:
- **WO-9.1 Census (inert).** Add the three census keys of §5.8 (`SharpCensus`, the binding, its
  docstring). Nothing on the step path changes.
  - *Accept:* `state_hash` = main.
  - *Accept:* `sharp_viscous`, `sharp_viscous_mpi_np{1,2,4}` and `vof_viscous_interface_auto` pass
    unchanged.
  - *Accept:* on the two sharp `mirror_symmetry` rows (run A, read after the first `step()`),
    `tied_edges` equals MIRROR_DIAG's exact + near count: 80 structured (46 + 34), 94 block A (13 + 81).
    A difference is acceptable only if it is explained by the two enumerations (e.g. edges on the
    high domain face). `max_tie_spread` ≤ 1e-12. Record `min_untied_gap`.
  - A count mismatch means a gap in (1e-12, 1e-6] exists, which is a fact for Q31: report it, do not
    tune.
- **WO-9.2 The rule.** Replace `edgeOwner` by the owner-set visitor at its three call sites
  (`SvBandOp::beta`, `edgeTraction`, the fit's shear-link item), exactly as §5.8 "The rule" items 1–5.
  Add `kSvOwnerTieTol = 1e-6`. Rewrite the header comment "Fixed offset order (§5.7)" and the
  `edgeOwner`/`edgeTraction` docstrings to say: owner set, mean, offset order for sums only. Tests in
  the same commit:
  - `sharp_viscous`, a **reflection self-symmetry case.** A sphere of radius 5.3 cells centred on a
    grid vertex of a 24³ periodic box, μ ratio 1e3, σ = 0, velocity the pure strain
    u = (X, Y, −2Z) about the centre (definite parity under each reflection; its wrap jump lies
    ≥ 4 cells outside every fit block).
    - For each axis reflection R, and over all band edges, max |G(E) − s·G(R E)| ≤ 1e-13·max|G| for
      G ∈ {F̄_row, θ̄_ac, θ̄_ca, T_row}.
    - s = −1 for T_row iff exactly one of a, c is the reflected axis; else s = +1.
    - The case must report `tied_edges` > 0, so that it exercises ties.
  - `sharp_viscous`, a **tie-free identity.** On an existing case whose census reads `tied_edges` = 0,
    the band-op outputs are bitwise equal to f01a251 (host). Record the values from an f01a251 build
    once, in the test.
  - `sharp_viscous_mpi_np{1,2,4}`: add the vertex-centred sphere with a rank boundary on a symmetry
    plane and a periodic seam through tied edges. F̄, θ̄, T and τ̂ must be bitwise equal at np 1/2/4.
  - *Accept:* the three ctests above, plus `vof_viscous_interface_auto`, on host.
- **WO-9.2b The ramp (eighth pass, §5.9).** Apply §5.9's implementation delta (items 1–7) on top of
  582ac73. It is one commit. *Accept:* the §5.9 (b) binding list, host first, then CUDA. Then resume
  WO-9.4 with the ramp in place.
- **WO-9.3 The mirror gate.** In `tests/python/test_mirror_symmetry.py`:
  - tighten the velocity threshold to 1e-12 for every row (pressure stays 1e-8, the PCG tolerance);
  - after the first `step()` of run A in the two sharp rows, assert `tied_edges` equals the WO-9.1
    value and `max_tie_spread` ≤ 1e-9.
  - Re-run the MIRROR_DIAG harness rows on the new build, with the DIAG patch not applied: VoF S = 1,
    cy ∈ {3.0, 3.3, 6.0, 6.3}; blocks S = 1, cy ∈ {3.0, 6.0}; VoF S = 2, cy ∈ {3.0, 3.3}.
  - *Accept:* every row ≤ 1e-12, host and CUDA (expected ~1e-15, as the DIAG "both" rows).
  - If a non-sharp row exceeds 1e-12 on CUDA, it keeps 1e-10 and is reported. That would be a
    pre-existing condition, not WO-9's.
- **WO-9.4 Batteries and study gates (no code).** Host battery `-LE bench` (286), the CUDA battery,
  `state_hash` = main, G-SV6 (np 2/4, host + CUDA), G-SV7, and the study gates by the census rule of
  G-SV12. Log each as one summary line in the landing log.
- **WO-9.5 Docs.** One sentence in the flow `CLAUDE.md` sharp paragraph ("owner ties: mean over the
  owner set within 1e-6 cells; offset order for sums only"); the §13 amendment into
  `docs/decisions/flow.md` and `docs/DECISIONS.md` (umbrella, coordinator); CHANGELOG.

Escalate WO-9 to the architect, do not tune, if any of these happens:
- the triggers of §5.9 (b), which replace this list's earlier "`max_tie_spread` > 1e-8" trigger.
  That trigger is withdrawn: the census spread mixes genuine ties with generic near-ties.

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
  - N = 24, 32, 48, 64: the branch §3.4 pre-registered, ruled in §5.7 Q-C. It was 16, 24, 32, 48.
    Error = max |u_h − ū_face| / max|ū_face|.
  - **PASS (USER 2026-10-31, §12 Q13 decided):** per orientation × ratio × case, the offset envelope
    e_env(N) = max over the 8 offsets of the max-norm error; PASS iff the LS order of e_env over
    N = 24, 32, 48, 64 is ≥ 1.8. Couette passes as "exact" if e ≤ 1e-10 at every N (unchanged).
  - Reported alongside, not gating: e_env·N², and the per-offset LS orders (their minimum). The
    earlier reading "order ≥ 1.8 for every orientation × offset × ratio" is superseded.
  - The 2-D a-priori version is in §3.4.
- **Third pass (§6.3): G-SV1, G-SV1c and G-SV1d move to a triply periodic box. DECIDED USER 2026-11-01
  (§12 Q16).**
  - The periodic gates below (G-SV1P/1cP/1dP, envelope order per Q13) plus G-SV1x bind. The walled box
    with option-A data is reported as a diagnostic and is not judged.
  - The thresholds and the envelope rule are unchanged.
- **G-SV1P — periodic layered flow (replaces G-SV1's order gate).**
  - **Box.** [0,1]³, triply periodic, N ∈ {24, 32, 48, 64}.
  - **Normals.** m ∈ (1,0,0), (2,1,0), (1,1,0), (0,1,2), (1,0,1), (2,1,1), (1,2,1), (1,1,2), (1,1,1).
    - n = m/|m|; the period along n is P = 1/|m|.
    - They cover aligned, the in-plane obliques in all three coordinate planes, the 3-D obliques with each
      axis dominant, and both diagonals.
    - |m|² ≤ 6, so every layer is ≥ 3.7 cells thick at N = 24.
  - **Layers.** Phase A where frac((n·x − s0)/P) < 3/8, else B, with s0 = n·(½,½,½) + (k/8)h, k = 0…7.
    - 3N/8 is an integer for every N above, so each offset's sub-cell configuration set is N-independent
      and the per-offset slope has no fill-in bias.
    - C is the exact cell volume fraction (inclusion–exclusion, as in `gsv1.py`), summed over the plane
      family.
  - **Ratios.** μ_A/μ_B ∈ {1e-2, 1e2}; ρ_A/ρ_B = 1/2.
  - **Force.** f = (ρ(C) − ρ̄) g t1, with t1 the first of `gsv1.py`'s `tangents(n)` and ρ̄ the mean cell
    density.
    - Apply it through the closures `force_c = linear(rho, [−ρ̄ g t1_c, g t1_c])`, c = x, y, z. The fallback
      is `set_field("force_c")` in internal units via `s.unit_scales`.
    - The face force is the face interpolation of the cell field. Σ_faces = 0 to round-off.
  - **Exact solution.** u = U(n·x) t1, with U periodic and piecewise quadratic: μ_k U″ = −(ρ_k − ρ̄) g, and
    U and μU′ continuous.
    - ū is the exact face average of U (the face kernel, as `gsv1.py`'s `avgU`).
    - Gauge: per component, subtract the ρ_F-weighted mean of ū. From rest the discrete Σρ_F u_c stays 0.
  - **Steady state.** WO-6s at D_hi = 100, to a step increment ≤ 1e-12·max|u|.
  - **Error.** e = max|u_h − ū| / max|ū|.
  - **PASS.** Per normal × ratio, the LS order of the offset envelope e_env(N) over the four N is ≥ 1.8.
    Reported: e_env·N² and the per-offset LS orders.
- **G-SV1dP.** G-SV1P at μ ratio 1e-3 and 1e3. Same PASS. Fifth pass (§6.8 C): a step refused by the colour-CFL guard inside
  the GMRES driver is re-evaluated by affinity (WO-C5). The certifying step stays plain.
- **G-SV1cP — periodic along mode (replaces G-SV1c's order rows).**
  - **Geometry.** Same box, normals, N and offsets as G-SV1P. φ_A = 1/2: phase A where s′ mod P ∈ (P/4, 3P/4),
    s′ = n·x − s0, so the interfaces lie where cos(b s′) = 0 with b = 2π|m|. ρ = 1.
  - **Tangent.** t1 = t/|t|, with t the integer vector ⊥ m, a = 2π|t|, η = t1·x:

| m | (1,0,0) | (2,1,0) | (1,1,0) | (0,1,2) | (1,0,1) | (2,1,1) | (1,2,1) | (1,1,2) | (1,1,1) |
|---|---|---|---|---|---|---|---|---|---|
| t | (0,1,0) | (−1,2,0) | (−1,1,0) | (0,−2,1) | (−1,0,1) | (−1,2,0) | (2,−1,0) | (1,−1,0) | (1,−1,0) |

  - **Exact fields.**
    - Velocity: u = −a sin(aη) cos(bs′) n + b cos(aη) sin(bs′) t1. It is smooth and solenoidal, with
      S_nt = 0 on the interfaces: a pure along mode.
    - Pressure: p_k = 2μ_k ab sin(aη) sin(bs′), which carries the jump 2[μ]D_nn.
    - Force: f_k = μ_k [a(b² − a²) sin(aη) cos(bs′) n + b(3a² + b²) cos(aη) sin(bs′) t1].
  - **Force input.** The cell field μ(C_i)·F(x_i), written to `force_c` in internal units.
    **Fifth pass (§6.8 B):** this input carries an O([μ]F) band error (measured 25–84 % of max|f|). The
    face-CV-consistent force (WO-B5) replaces it if the §6.8 B N48 pair passes (Q26/Q27). Until then G-SV1cP is NOT JUDGED.
    **Verdict 2026-11-04: NOT JUDGED under the §6.8 D rule** — sharp N32→64 order 1.68 (1,1,0) / 1.55
    (2,1,1), local orders rising; averaged 1.27 / 0.45 (N24→48). The AUTO default ships with this
    documented (USER 2026-11-04); the term split (§6.8 D) is a study-only follow-up (Q26).
  - **Flows.**
    - (i) This field alone.
    - (ii) This field plus G-SV1P's layered flow on the same interfaces (ρ_A/ρ_B = 1/2 for that part).
  - **Exact face averages.** 4×4 Gauss–Legendre per face for the smooth part; the layered part as in
    G-SV1P. Same gauge.
  - **PASS.** Envelope LS order ≥ 1.8 for (i) and (ii), at μ ratios 1e-2 and 1e2.
- **G-SV1x — exactness at operator level (replaces the steady "Couette exact" and "extension ≤ 1e-9"
  rows).**
  - WO-4's subtest: the 9 (β, γ) orientations of G-SV1, Couette and extension.
  - Residual ≤ 1e-11 relative to the band stress scale, on every row ≥ 3 cells from a wall.
  - For extension the residual is A ū + G p̄, with p̄ the exact cell-averaged pressure, jump included
    (WO-6x).
  - Measured 7.6e-12, pending the pressure check.
- **G-SV1W — the wall contact (reported; decides Q6 / WO-W).**
  - Walled box, Couette 22.5° / 22.5°, offset 0, μ ratio 1e-2, N = 24 and 32. Option-A BCs, but with
    wall-face-AVERAGED Dirichlet data.
  - Steady state by WO-6s, both modes.
  - PASS iff, at both N, 'sharp''s maximum error over faces within one cell of a wall ≤ 'averaged''s.
  - First data (option A, point values, sharp): 7.18e-3 / 6.16e-3, max at the contact line.
- **G-SV2 — Hadamard–Rybczynski, κ = 1e-2, 1, 1e2. Fully specified in §6.7 (fourth pass), which
  binds; the summary below is kept for the record.**
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
- **G-SV6.** Sharp on, P1 d/h 16 for 20 steps, plus the G-SV1 configuration at N = 24 (walled,
  σ = 0). **USER 2026-10-31** (replaces "np 1/2/4 bitwise" at solver level, which today's
  'averaged' path already misses at 1.1e-15): PASS iff max|u_np − u_np1| / max|u| ≤
  max(1e-12, 1.1 × the 'averaged' path's deviation on the same case), for np = 2 and 4, host and
  CUDA. The operator-level ctest `sharp_viscous_mpi_np{1,2,4}` stays bitwise.
- **G-SV7.** P1 d/h 24 ms/step ≤ 1.2 × 'averaged' (same build, same GPU).
- **G-SV8.** P1 at ρ ratio 1e-4, μ ratio 1e-3 runs 10 t-units without blow-up.

**Added.**

- **G-SV1c (along mode).** Same configurations as G-SV1 with u = ε(ξn − ηt₁) and the exact pressure jump.
  Error ≤ 1e-9 at every N. Poiseuille + extension: order ≥ 1.8 (the envelope order, USER 2026-10-31).
- **G-SV1d.** G-SV1 at μ ratio 1e-3 and 1e3 (the G-SV8 contrast). Same thresholds (envelope order,
  USER 2026-10-31).
- **G-SV9 (stability; replaces the FVS §6 proof in the band).**
  - (a) 2-D harness: ρ(G) ≤ 1 + 1e-12 for the table in §6 at N = 16 and 32.
  - (b) 3-D homogeneous march, N = 16.
    - Random solenoidal initial field; plane at n(22.5°, 22.5°), offset 0.3.
    - μ ratio 1e-2, 1e2, 1e-3, 1e3 with ρ ratio 1e-3, 1e3, 1e-4, 1e4.
    - D_hi ∈ {0.1, 1, 10, 100, 1e4}; 2000 steps.
    - PASS: kinetic energy ≤ initial, and the growth ratio over any 100 steps after step 200 is
      ≤ 1 + 1e-10.
  - (c) No blow-up in G-SV3 or G-SV8.
  - (d) Film geometry (§5.7 Q-B). As (b), with a grid-aligned 1-cell slab of the low-μ phase in place of
    the oblique plane.
    - The slab is normal to x, and both of its faces carry virtual planes.
    - Same ratio pairs, D_hi values and PASS as (b).
  - **Third-pass amendments (§6.1, §6.2).**
    - **(a)** uses the production split, Kb, with K1 printed as a control (WO-5b).
    - **(b)(d)** run the production pressure driver, never a forced CG driver (WO-6a). A capped step makes
      the run NOT JUDGED.
      - The first MISS records (PCG at 800 iterations) are void for (d), being an instrument fault. They
        stand for (b), which shows the split instability.
      - **DECIDED USER 2026-11-01 (§12 Q15)**: the growth ratio is judged only over windows with
        E_n ≥ 1e-20·E_0. The as-written ratio is still reported, not judged.
    - **(b′), reported only.** The (b) march on a triply periodic box with the rational normal (2,1,1),
      offset 0.3 cell. If (b) misses while (b′) passes, the cause is the walls (§6.3).
    - **(e), spectral (added, not a loosening).**
      - *Measure.* The Arnoldi ρ of the PRODUCTION incremental step map on (u, v, w, p), colour frozen,
        mean of p removed. Chebyshev pressure to rtol 1e-13, RB-GS to rtol 1e-15:
        `gsv9_split.py prod --incremental --pdriver cheb`.
      - *Cases.* The (b) plane at N = 16 and the (d) film at N = 12, each with the four ratio pairs ×
        D_hi ∈ {0.1, 1, 10, 100, 1e4}, both modes.
      - *PASS.* ρ ≤ 1 + 1e-9 for 'sharp', with 'averaged' reported. A run whose linearity check exceeds
        1e-10, or whose solves cap, is NOT JUDGED.
      - **Fifth pass (§6.8 A, USER Q24 pending; default adopted):** momentum fixed at 2000 RB-GS sweeps per
        component, doubling certificate in place of the momentum cap test (WO-A5).
      - *ARPACK.* Use ncv 40 and maxiter 1000. An unconverged run is re-run at ncv 80. A 'sharp' case still
        unconverged is decided by its (b)/(d) march verdict, and is listed.
      - *Before WO-5b.* Sharp plane ρ = 1.046 / 1.072 at D_hi = 100 / 1e4 (μ 1e-3) and 1.017 / 1.045
        (1e-2). 'averaged' gives ≤ 0.9997, with one case (μ 1e-3, D_hi = 100) unconverged in ARPACK: a cluster
        at 1. The film gives 0.9975–0.9982 for both modes.
- **G-SV10 (census).** P1 d/h 24: fit fallbacks ≤ 0.1 % of interfacial cells per step, averaged over the
  run. Virtual-plane cells are reported, and film cells among them.
- **G-SV11 (single phase stays put).** `variable_mu`, `varmu_mpi`, `full_stress_operator{,_np2,_np4}`,
  `collocated_stability_guard` and `amr_stability_guard` pass bitwise.
- **G-SV12 (mirror equivariance and owner ties; seventh pass §5.8, restated by the eighth pass §5.9,
  WO-9).**
  - *Mirror.* `mirror_symmetry`: every row velocity ≤ 1e-12 and pressure ≤ 1e-8, host and CUDA. The
    `tied_edges` pinned after the first `step()` is equal on both backends, and
    `max_tie_spread` ≤ 1e-9 (on this case only: it pins step-1 symmetric ties). The MIRROR_DIAG
    harness rows (WO-9.3) are ≤ 1e-12. The `sharp_viscous` reflection case is ≤ 1e-13 relative.
  - *Ramp (eighth pass).*
    - The `ownerWeight` unit test passes.
    - The frozen axis and two-axis G-SV1 colours are bitwise equal to 582ac73.
    - The tie-free identity digest equals f01a251.
  - *np.* `sharp_viscous_mpi_np{1,2,4}`, with the tie case, is bitwise.
  - *Host ↔ CUDA.* There is no field-level gate (FMA contraction and the device FP32 V-cycle make it a
    tolerance question with no owner-specific content). Cross-backend agreement of the owner
    *decisions* is the pinned census integer, which each backend's own battery asserts.
  - *Which study gates re-run, and the acceptance band.* All re-run in full (compute is free), each
    run reading the census after every step:
    - G-SV1x, G-SV1P, G-SV1dP, G-SV1W, G-SV1cP (still NOT JUDGED; INFO), G-SV2, G-SV3 (P1 ladder,
      d/h 16–48), G-SV9 (a)–(e), G-SV10;
    - G-SV4 and G-SV8 if they have recorded numbers to compare against.
  - *Withdrawn (eighth pass): the census-zero bitwise class.* It is empty on every G-SV1 row. Every
    study row is judged **band-only** against its recorded single-owner baseline. Its verdict must
    not change, and it must stay inside the movement band:
    - G-SV1-family e(N) within ±10 % of recorded per row; orders ±0.05; G-SV1x exactness unchanged;
    - G-SV2 gated quantity within ±1e-3 relative; order ±0.05;
    - P1 Re within ±1e-3 relative per d/h;
    - G-SV9 ρ ≤ recorded + 1e-3, and ≤ 1 where it was;
    - G-SV10 fallbacks ≤ 0.1 %.
  - The bands come from §5.8's argument that a tie's effect on an even quantity is second order in
    the asymmetry the old rule injected. Near-equidistant members are nearly the same plane, so the
    ramp's partial weights move results by little (smoke: ≤ 1.3e-7 relative).
  - Escalation follows §5.9 (b):
    - a band breach with an unchanged verdict is INFO;
    - a verdict flip is a STOP;
    - `blended_edges` + `tied_edges` above 5 % of band edges is INFO;
    - a frozen (1,1,1) spread above 1e-7 at N 96/128 is an escalation.
  - The census keys `max_tie_spread` and `min_untied_gap` are INFO and never trigger anything.
  - *Cost.* Sharp ms/step on P1 d/h 24, new against f01a251 on the same build type: ≤ 1.02 on host
    and on the GPU. G-SV7 (≤ 1.2 × 'averaged') still holds.

---

## 12. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1 | 3-D stability margin (2-D measured; 3-D adds a third shear plane) | **fact** — ANSWERED (third pass, §6.1) | Not a 3-D mode: the cell split. WO-5b; the 3-D margin after it is Q14. |
| Q2 | Memory for 4 extra fields at d/h ≥ 48 on 16 GB | **fact** (measure free memory at P1 d/h 48) | Dense fields. Switch to worklist storage if they exceed 10 % of free device memory. |
| Q3 | Name and tier of the setter | **user preference** (docs/NAMING.md) | `diagnostics.set_vof_viscous_interface("sharp"\|"averaged")` until the gates pass. DECIDED 2026-11-04 (USER): public `set_vof_viscous_interface("auto"\|"sharp"\|"averaged")`, default "auto" (§9). |
| Q4 | AUTO = sharp after the gates | **user preference** | Yes: the user named it priority. Make it its own commit and register entry. |
| Q5 | G-SV2 reference in a periodic box | **fact** | Gate on the self-convergence order. The HR periodic-corrected value is a 3 % sanity check. |
| Q6 | Interfaces touching walls or contact lines (fit links cut by the wall) | **fact** (census near walls; G-SV1W) | Drop links outside the domain. Fallback to 'facef' if the fit is singular. Report in the census. Third pass: the contact-line error (hat R̄ on a half-link) is out of scope while G-SV1W passes; WO-W parked (§6.3). |
| Q7 | Other μ closures (Table1D in C, temperature-dependent) | **user preference** (scope) | Not in v1. AUTO falls back to averaged with a notice. |
| Q8 | Thin films, under 1 cell of one phase | **fact** (census) | The fit is well-posed while every link type appears. Otherwise fallback; τ̂ = 0 is first order locally. |
| Q9 | Plane source when σ = 0 | **fact** (does VofCurvature exist with σ = 0?) | Construct and reconstruct it in the step when sharp is active. |
| Q10 | Curved interface: one plane per fit block (O(κh) model error) | **fact** (G-SV2 order) | Accept. If G-SV2 misses only at d/h 8–12, report it; do not add curvature terms without the architect. |
| Q12 | τ̂ at edges: band mean, owner only, or interfacial mean | **fact** — settled by E1/E2 (§5.7 Q-A) | Interfacial mean. Fits run on interfacial cells only; all three rules have the same order. |
| Q13 | G-SV1's measurement: per-offset LS order or the offset envelope (§5.7 Q-C) | **user preference** — **DECIDED USER 2026-10-31** | The envelope order binds (§11 G-SV1); per-offset orders reported, not gating. |
| Q11 | Collocated | **user preference** (sequencing) | §8 order: full stress on collocated first; this design is not ported until then. |
| Q14 | 3-D margin of the Kb split (2-D measured 0.9859) | **fact** (G-SV9(e) and (b) after WO-5b) | Proceed. If ρ > 1, stop and return to the architect; no tuning, no dt cap. |
| Q15 | G-SV9 growth ratio at the energy floor (§6.2) | **user preference** — **DECIDED USER 2026-11-01** | Judge the growth ratio only over windows with E_n ≥ 1e-20·E_0; the as-written ratio is reported. |
| Q16 | G-SV1/1c/1d: the walled box replaced by the periodic G-SV1P/1cP/1dP, exactness at operator level (G-SV1x) | **user preference** — **DECIDED USER 2026-11-01** | The periodic gates (envelope order, Q13) + G-SV1x bind; the walled option-A numbers are a diagnostic only. |
| Q17 | Wall-contact accuracy (Q6) in this package? | **fact** (G-SV1W), then **user preference** (scope) | Out of scope; WO-W parked. A G-SV1W MISS brings it in, design pass first. |
| Q18 | A forced MG-PCG/FCG driver at density ratio ≥ 1e3 returns wrong fields silently (§6.2) | **user preference** (scope; outside this package) | Hand it to the coordinator as a separate item. Nothing here beyond WO-6a. |
| Q19 | WO-6s solver parameters (restart, cap, D_hi) | **fact** (the WO-6s pilot) | GMRES(60), cap 3000, D_hi = 100. The steady state is dt-independent (§6.4), so D_hi may move only on pilot evidence, with the march-agreement check. |
| Q20 | The RB-GS stagnation exit in production (§6.6) | **fact** (WO-7 exit-residual log) | Not this package. Opens its own package if > 1 % of P1/P2 steps exit above tolerance. |
| Q21 | Steady gates: one momentum-solve setting (§6.6) | settled (fourth pass) | Converged momentum (update stop 1e-15 per the coordinator decision of 2026-11-01, superseding 1e-13; residual stop off) for every step of every steady gate; default-solve G-SV1P/1dP/1cP results void, rerun. |
| Q22 | G-SV2 parameters (§6.7) | settled (fourth pass) | ρ uniform, σ = 0, Boussinesq force, F = ΔρgV_h, vertex-centred drop, U_d = C-weighted drop mean. |
| Q23 | Composite (predictor × M-orthogonal projection) instability at high contrast, layered geometry, D ≥ 1 — in 'averaged' too (§6.5) | **user preference** (priority; separate package) | Logged for the coordinator; own architect pass (periodic layered harness first). This package proceeds; G-SV9(f) reports the onset, and 'sharp' worse than 'averaged' at D_hi ≤ 3 is a STOP. |
| Q24 | G-SV9(e): fixed-count momentum (N_A = 2000) and the doubling certificate replace "RB-GS to 1e-13/1e-15 … or whose solves cap ⇒ NOT JUDGED" (momentum half; §6.8 A) | **user preference** (a post-data change to a pre-registered instrument) | Accept. It is stricter: the measured map error falls from 4e-12 to ≤ 1e-13, and the solve is certified directly. Judge (e) from the WO-A5 rerun, not from the old pilot. |
| Q25 | Steady gates: retire the coordinator rule "a sweep-cap hit → NOT CONVERGED" (never implemented by the tables; 71/38/59/4/15 capped rows in G-SV1P/1dP/1cP/1W/G-SV2) | **user preference** (it touches FINAL verdicts), then **fact** (WO-A5b) | Retire it. G-SV1W and G-SV2 stay FINAL unless WO-A5b's spot check differs by more than 1e-8 (then STOP → architect). **DECIDED 2026-11-04 (USER): the sweep-cap → NOT CONVERGED rule is retired** (evidence §6.8 A + WO-A5b 4/4, rel. diff ≤ 2.3e-13); the cap counts stay a reported column. |
| Q26 | G-SV1cP: is the along-mode order shortfall the operator? (§6.8 D) | **fact**: `arch6/steady.jsonl` (sharp N16/32/64, controls) + `arch6/resid.jsonl` (2,1,1) rows, then §6.8 D's rule | Sixth pass: (i) excluded on (1,1,0); shear path second order; one candidate (iii) term, a non-converging D_nn-modulated normal force sheet (≈5e-3 μ_hi U a). Apply §6.8 D rule 1/2/3 when the rows land. Meanwhile ship sharp as AUTO with this mode documented "1.4–1.6, under study"; G-SV1cP NOT JUDGED. **Rule applied 2026-11-04 (`arch6/steady.jsonl`, ratio 100, k3):** sharp along-mode order N32→64 1.68 (1,1,0) / 1.55 (2,1,1), local orders rising (1.27/1.50/1.63/1.74 and 0.69/1.20/1.47/1.64); averaged N24→48 1.27 / 0.45; the M0_n sheet stays flat (3.99e-3 → 3.61e-3 on (2,1,1)). Neither (ii) nor (iii) → **G-SV1cP NOT JUDGED**; the default ships with this documented (USER 2026-11-04); the term split is a study-only follow-up. |
| Q27 | G-SV1cP force input: the face-CV-consistent force (§6.8 B) replaces "the cell field μ(C_i)F(x_i)"; invariance judged on the certifying step | **user preference** (a post-data change to an instrument) | Accept. The old input measures its own forcing: 84 % of the worst row's error goes when the force is made consistent, with the operator unchanged. |
| Q28 | Uniform-μ controls end with a mean velocity of 1.4–1.7 % of max\|u\| (EXP-B ctrl: err vs err_gauged; `mom` 1.4e-2, 1.7e-2). A momentum-conserving step started from a gauged x0 should not produce that | **fact** (one instrumented ctrl run, `mom` per restart step) | Not blocking: no gate judges a uniform-μ ctrl row, and G-SV2 κ = 1 subtracts ⟨w⟩. Investigate before any uniform-μ control is cited as evidence of momentum conservation. |
| Q29 | Steady gates on fixed-count momentum for cost (the certifying step runs to the 2e5 cap when its warm start is the answer)? | **fact** (time saved at N64) + **user preference** | No, for the sets in flight (one setting within a gate). Reconsider for the next campaign. |
| Q30 | GMRES basis scaling: the step turns the Krylov vectors' p part (internal units) into velocity with gain up to 65 (§6.8 C), so restarted GMRES needs 5–8 cycles on (2,1,0), (1,2,1), (2,1,1) | **fact** | No change: it is not needed for a verdict. If WO-C5 leaves NOT CONVERGED rows, scale p by μ_hi,int/h_int in the unknowns for ALL rows of the affected set, as its own instrument change. |
| Q31 | Placement of τ_own = 1e-6 (§5.8): is the void between tie spread and untied gaps real in production? | **fact** — ANSWERED (eighth pass, §5.9): no void | Plateau τ₀ = 1e-6 kept, plus a C¹ ramp to τ₁ = 1e-5; the census keys are INFO. |
| Q32 | Permutation (axis-swap) equivariance end to end | **fact** (scope) | Not gated. The owner rule is permutation-equivariant; upstream it is not: MYC's axis choice, Q-B's x, y, z order at pure cells qualifying on two axes, and the WY split order. Reflections are gated (G-SV12). |
| Q33 | A continuous owner blend (§5.8 rejected list) | **fact** (is owner switching a measurable source of temporal noise, e.g. parasitic currents at a static drop?) | No. Revisit only on such a measurement. |
| Q34 | A field-level host ↔ CUDA gate | **fact** (host ↔ CUDA drift on P1 with `'fp64'` V-cycle) | No. Each backend runs the mirror gate; the owner decisions are pinned by the census integer (G-SV12). |
| Q35 | `plicAlpha` loses about half its digits at normals with equal \|components\| (n1 = n2, (1,1,1)/(2,1,1); §5.9 (c)): the stored α carries ~1e-8 error there | **fact** (frozen (1,1,1) spread at N 96/128; α error against a Newton-polished α) | Not in this landing. Candidate fix in `core`: one Newton step on `plicVolume` after the analytic inversion. It moves every VoF hash, so it needs its own decision and gates. |

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
established by the measured spectral gate G-SV9, not by the energy proof. The composite of predictor
and projection is unproven for both modes, and a layered periodic case is unstable at D_hi ≥ 100
('sharp') and ≥ 1 ('averaged', μ 1e3 / ρ 1e4); see §6.5 and Q23. Register 272's choices (full
stress, explicit conservative transpose on uⁿ, no product form, no Picard placement) stand.

**Why:** 2-D a-priori, N = 16–128, 0–45°, all offsets, μ ratio 1e-3–1e3:
- Couette and extension exact to round-off;
- Poiseuille and general L∞ order 1.93–1.95, L1 2.00;
- ρ(G) ≤ 0.99 at every D.

Date 2026-10-31. Evidence: `doc/vof_sharp_viscous_stress.md` §3.4 and §6; `tests/study/sharp_viscous/`.

**Amendment (seventh pass, 2026-11-04): edge-owner ties.**
- *Decision.* An edge's owner set is every interfacial cell of the edge within τ_own = 1e-6 index
  cells of the closest plane, |m·ξ_E − α|/|m|₂, weighted 1, then a C¹ smoothstep ramp to 0 at
  1e-5 (eighth pass, §5.9). Each per-edge output is the weighted mean of the single-owner outputs
  over that set: the face fraction (hence β), the link fractions, and the T-term
  as a per-member product. The fixed offset order fixes summation order only.
- *Rejected.*
  - A single owner with ties to the first cell in offset order, or any other single-candidate
    tie-break. At a reflection-symmetric colour field the tied candidates are swapped by the symmetry,
    so no single choice is equivariant, and round-off decided the near ties (`mirror_symmetry`
    3.1e-5–7.9e-5).
  - A continuous distance-weighted blend: it needs a width that changes the operator at a finite
    fraction of all edges.
  - The T-term from averaged fractions: a spurious ≈ (r/4)·τ̂ explicit stress.
  - A sharp set boundary (§5.8 as first written): with no void between tie spread and generic gaps,
    round-off decided the switch at τ.
  - A structural (bitwise-colour) tie test, and ties fixed at step 1: neither survives an evolving
    symmetric run.
- *Evidence.* §5.8; `bubble_column_perf/validation/highratio/land/MIRROR_DIAG.md`.

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
