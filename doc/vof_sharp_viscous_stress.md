# Sharp viscous stress at a VoF interface on the staggered MAC grid — design note

Status: DESIGN (architect, 2026-10-31; amended the same day by a second pass, §5.7). Branch `sharpvisc` (on `mom2`). USER 2026-10-31: "Yes give the
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
- **Owner of edge E.** Among the four cells of E that are interfacial, the owner is the one with the
  smallest |m·ξ_E − α|/|m|₂. Here ξ_E is the edge midpoint in that cell's local coordinates. Ties go to
  the first cell in the fixed offset order (§5.7). That is the lowest linear index everywhere except
  across a periodic seam.
  - E is a **band edge** iff it has an owner.
  - **τ̂ at an edge sample (§5.7 Q-A).** τ̂_E is the arithmetic mean of τ̂_K over the **interfacial**
    cells K among E's four cells, summed in the fixed offset order. The set is never empty, because E's
    owner is in it.
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

**Fixed offset order.** Every tie and every summation order in §5.2 and here enumerates cells by their
offset from the edge or cell, lexicographic in (dz, dy, dx), x fastest. Never use the global linear
index.
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
- **Third pass (§6.3): G-SV1, G-SV1c and G-SV1d move to a triply periodic box. USER flag: a gate change
  after data.**
  - Default until the user rules: run the periodic gates below. The walled box with option-A data is
    reported as a diagnostic and is not judged.
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
- **G-SV1dP.** G-SV1P at μ ratio 1e-3 and 1e3. Same PASS.
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
      - **USER**: judge the growth ratio only over windows with E_n ≥ 1e-20·E_0. Until the user rules, both
        verdicts are reported; the floor-excluded one and (e) decide sequencing.
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
      - *ARPACK.* Use ncv 40 and maxiter 1000. An unconverged run is re-run at ncv 80. A 'sharp' case still
        unconverged is decided by its (b)/(d) march verdict, and is listed.
      - *Before WO-5b.* Sharp plane ρ = 1.046 / 1.072 at D_hi = 100 / 1e4 (μ 1e-3) and 1.017 / 1.045
        (1e-2). 'averaged' gives ≤ 0.9997, with one case (μ 1e-3, D_hi = 100) unconverged in ARPACK: a cluster
        at 1. The film gives 0.9975–0.9982 for both modes.
- **G-SV10 (census).** P1 d/h 24: fit fallbacks ≤ 0.1 % of interfacial cells per step, averaged over the
  run. Virtual-plane cells are reported, and film cells among them.
- **G-SV11 (single phase stays put).** `variable_mu`, `varmu_mpi`, `full_stress_operator{,_np2,_np4}`,
  `collocated_stability_guard` and `amr_stability_guard` pass bitwise.

---

## 12. Risks and open questions (each with a default)

| # | question | needs | default |
|---|---|---|---|
| Q1 | 3-D stability margin (2-D measured; 3-D adds a third shear plane) | **fact** — ANSWERED (third pass, §6.1) | Not a 3-D mode: the cell split. WO-5b; the 3-D margin after it is Q14. |
| Q2 | Memory for 4 extra fields at d/h ≥ 48 on 16 GB | **fact** (measure free memory at P1 d/h 48) | Dense fields. Switch to worklist storage if they exceed 10 % of free device memory. |
| Q3 | Name and tier of the setter | **user preference** (docs/NAMING.md) | `diagnostics.set_vof_viscous_interface("sharp"\|"averaged")` until the gates pass. |
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
| Q15 | G-SV9 growth ratio at the energy floor (§6.2) | **user preference** (a criterion change after data) | Report both verdicts; the floor-excluded one (E_n ≥ 1e-20·E_0) plus G-SV9(e) decide sequencing. |
| Q16 | G-SV1/1c/1d: the walled box replaced by the periodic G-SV1P/1cP/1dP, exactness at operator level (G-SV1x) | **user preference** (a gate change after data) | Run the periodic gates; report the walled option-A numbers as a diagnostic. |
| Q17 | Wall-contact accuracy (Q6) in this package? | **fact** (G-SV1W), then **user preference** (scope) | Out of scope; WO-W parked. A G-SV1W MISS brings it in, design pass first. |
| Q18 | A forced MG-PCG/FCG driver at density ratio ≥ 1e3 returns wrong fields silently (§6.2) | **user preference** (scope; outside this package) | Hand it to the coordinator as a separate item. Nothing here beyond WO-6a. |
| Q19 | WO-6s solver parameters (restart, cap, D_hi) | **fact** (the WO-6s pilot) | GMRES(60), cap 3000, D_hi = 100. The steady state is dt-independent (§6.4), so D_hi may move only on pilot evidence, with the march-agreement check. |

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
