# Scalar transport with immersed surfaces — design

*Architect design note, 2026-10-02. Branch `scalar-ibm` (flow worktree `suite/flow-scalar-ibm`).
Inputs: `doc/scalar_ibm_brief.md` (the question), `doc/scalar_ibm_log.md` (every measured number,
rounds 1–7), `doc/scalar_ibm_literature/` (L1, L2, L3, R), `tests/study/scalar_ibm/*.py` (the 2-D
oracles). This note stands alone: an implementer who has seen none of those can execute §10 against
§11. Every choice left open is in §13 with a default.*

**Status of the evidence.** All accuracy claims below come from the 2-D prototypes. The 3-D numbers
in §11 are provisional bounds derived from them; the **convergence-order assertions are the real
gates**. One new prototype run was made for this note. It is `jc.py probeN1.1` (§3.1): a probe distance
that depends on the normal. It is committed with this note.

---

## 0. Decision index

| # | Decision | Rejected alternative(s) | Section |
|---|---|---|---|
| D1 | Cut-cell FV with κ (fluid fraction) in storage and sources; every κ > 0 cell with an open face is an unknown | unit storage; cell merging/linking | §1, §2.4 |
| D2 | Faces carry the plain aperture two-point flux | Johansen–Colella face-centroid interpolation | §1.4 |
| D3 | Wall flux = **probe-flux**: one probe on the facet normal, per-facet elimination of Dirichlet/Neumann/Robin/conjugate | the list in brief §5.8, plus Peters' directional one-field conjugate (log round 7) | §1.4, §3 |
| D4 | Geometry = **fan-tetrahedron piecewise-linear (PL) model** on the shipped marching-squares samples: κ, apertures, facet area vector and facet centroid all come from ONE consistent polyhedral model | plane-cube `plicVolume(n, φ_c)`; 4³ subsampling (`cs_`); centroid = projection of the cell centre | §2 |
| D5 | The scalar has its OWN apertures (marching-squares, **ungated**, snapped at both ends at 1e-3). Advection uses the **projection's** face flux | reusing the gated pressure openness for diffusion; scalar apertures for advection | §2.3, §6.1 |
| D6 | Probe distance `s = 1.1 · ½ Σ_a |n_a| h_a` (normal-dependent); four-rung fallback ladder | constant √3/2·h; AMReX κ-dependent short probe | §3 |
| D7 | Double precision everywhere; no float variant | a templated `Real` | §3.5 |
| D8 | Conjugate: **two fields on one grid** (fluid c, solid c_s), ψ = c/K inside the solver, series-resistance 2×2 elimination at the probes | one field + side array; mixture one-field surrogate; Peters directional one-field; Das-style lagged coupling | §1.3, §1.4 |
| D9 | Solver: **BiCGStab** on the true probe operator, preconditioned by one V-cycle of a new **ScalarMG** on the SPD lumped-probe surrogate | fixed RB-GS sweeps; FGMRES/GMRES; centroid-distance surrogate | §5 |
| D10 | ScalarMG coarse **faces** are rediscretized from the coarsened geometry. Coarse **wall/interface terms** are the plain average of the level-0 terms at the **fine** probe distance (Amendment A1). Coupled two-field levels with a 2×2 block RB-GS for conjugate | wall terms at the level's probe distance s_L (overshoots; measured divergent, A1); Galerkin RAP (27-point, no better, A1); extending CutcellMG; VelocityMG's staircase; block-Jacobi per phase | §5.2 (A1)–5.3 |
| D11 | Stop on the max-norm relative residual (the velocity solver's form); the legacy 50-sweep path is untouched | fixed iteration count | §5.1 |
| D12 | Small cells under advection: **dynamic split** — implicit FOU on faces touching a cut cell whose explicit outflow would exceed what bulk cells see this step; explicit Koren/SOU/FOU elsewhere | weighted state redistribution (now; revisit for VoF/slip); explicit κ storage; unit storage; fully implicit FOU | §6 |
| D13 | Conservation is exact for the discretization; the budget reports the linear-solver defect; **no mass fix-up** | a uniform mass correction after the solve | §6.4, §9 |
| D14 | Container-free kernels in **core** (`scheme/cut_cell_geometry.hpp`, `scheme/probe_flux.hpp`); driver, storage, operators, ScalarMG and Krylov in flow | everything in flow; a core Krylov | §7 |
| D15 | Closure problems by a **mean-gradient mode** (c = G·x + θ, θ periodic, moving frame) and a steady solve | user-assembled sources | §1.5, §8 |
| D16 | Opt-in per scalar: `add_scalar(..., cutcell=True)`; physical units on the new path (DEFAULT-PENDING-USER: spelling) | a second entry point; switching the legacy path | §8 |
| D17 | Steady mode with advection: the V-cycle runs on the **advective surrogate**: the level-0 bands (FOU couplings included) with the lumped wall, and coarse advection = the summed positive parts of the sub-face fluxes (piecewise-constant Galerkin), band-form RB-GS. Transient mode and steady mode at rest keep the symmetric surrogate, bitwise (Amendment A2) | the symmetric surrogate with lumped outflow (fails beyond Pe_h ≈ 0.3); advection at level 0 only; pseudo-transient continuation; GraphAMG; defect correction; VelocityMG's restricted-velocity coarse upwinding; downstream/line smoothers; F/W-cycles as the default | §6.7 (A2) |

---

## 1. Formulation

### 1.1 Continuous problem and sign conventions

The scalar domain is the fluid region Ω_f, where the SDF φ > 0. Its walls are the immersed surfaces Γ (φ = 0).

- **n** is the unit normal on Γ pointing **into the fluid**.
- **q_in** is the flux per unit area **into the fluid** across Γ. Positive q_in means scalar enters the fluid.
- With flux vector **J** = −D∇c + **u**c and **u**·n = 0 on Γ, q_in = −D ∂c/∂n.

Fluid equation:

    ∂c/∂t + ∇·(u c) = ∇·(D ∇c) + S                 in Ω_f

Wall conditions on Γ, chosen per body:

| type | condition | parameters |
|---|---|---|
| Neumann | q_in = q | q (0 = insulating, the default) |
| Dirichlet | c_Γ = g | g |
| Robin | q_in = k (g − c_Γ) | k ≥ 0 (transfer coefficient, length/time), g |
| conjugate | solid interior simulated (below) | solid material |

- A first-order surface reaction consuming at rate k_r c is Robin with k = k_r, g = 0.
- Dirichlet is the limit k → ∞; Neumann-zero is k = 0.

**Conjugate solid** Ω_s (φ < 0, bodies flagged conjugate):

    C_s ∂c_s/∂t = ∇·(C_s D_s ∇c_s)                  in Ω_s

- C_s is the capacity ratio and D_s the solid's own diffusivity.
  - Heat: C_s = (ρc_p)_s/(ρc_p)_f and D_s = k_s/(ρc_p)_s, so the flux is −(k_s/(ρc_p)_f)∇T.
  - Porous particle: C_s = ε_p and D_s = the pore diffusivity.
- Interface:
  - flux continuity, −D ∂c/∂n = −C_s D_s ∂c_s/∂n;
  - partition and contact: q = (c_s/K − c)/R_c, where q is the flux from solid to fluid.
  - R_c = 0 means c_s = K c.

**ψ form (used inside the solver).**
- ψ = c in the fluid and ψ_s = c_s/K in the solid.
- Per phase p: capacity C_p and conductivity Λ_p.
  - Fluid: C_f = 1, Λ_f = D.
  - Solid: C_s K and Λ_s = C_s D_s K.
- The interface conditions become flux continuity Λ ∂ψ/∂n equal on both sides, and
  q = (ψ_s − ψ_f)/R_c.
- This is the ordinary conjugate problem. It is symmetric, which keeps the surrogate SPD.
- The solver stores ψ_s. `get_scalar_solid` returns c_s = K ψ_s.

### 1.2 Units

The API takes physical values. The solver computes on the unit lattice, folding the metric in through
`Solver::UnitScales` (hRef = min h, tRef = first `set_dt`). Internal per-axis cell size is
h'_a = `hp[a]`, and the other internal quantities are:
- cell volume V = h'_x h'_y h'_z;
- a-face area A_a = V/h'_a;
- w_a = 1/h'_a².

Conversions:
- **Append** them to `UnitScales` (never reorder; each is exactly 1.0 with no extent armed).
- Apply them **at advance time** from the stored physical values, never at set time. tRef may not
  exist yet when a setter runs.

| quantity | physical | internal |
|---|---|---|
| D, D_s | L²/T | × `diffToInt()` = tRef/hRef² |
| k (Robin) | L/T | × `speedToInt()` = tRef/hRef |
| q (wall flux per area) | c·L/T | × `speedToInt()` |
| R_c | T/L | × `resistToInt()` = hRef/tRef |
| S (volumetric source) | c/T | × tRef (= 1/`timeToInt()`) |
| G (mean gradient) | c/L | × hRef (= `lenToPhys()`) |
| c, g, K, C_s | — | unchanged |
| outputs: body flux (c·L³/T), facet flux (c·L/T), mean flux (c·L/T) | | × hRef³/tRef, × 1/`speedToInt()`, × 1/`speedToInt()` |

The three new members `diffToInt`, `speedToInt` and `resistToInt` are appended to `UnitScales`.

- With `extent=None`, every factor is 1, so the new path is in cell units, exactly like the legacy
  path.
- With an extent armed, the new path is physical. The legacy `add_scalar` diffusivity stays in
  internal units, which is a pre-existing gap (§13, Q6).

### 1.3 Unknowns and storage

**Fluid unknown.** Cell i is a fluid unknown iff κ_i > 0 and at least one of its six snapped apertures
is > 0 (§2.4).
- The registered field `<name>` holds c (or θ in mean-gradient mode).
- Non-unknown cells hold exactly 0. They are zeroed when the geometry is built, kept there by
  identity rows, and **re-zeroed at the start of every advance or steady solve**, because a user
  `set_field` may have written them.

**Solid unknown** (conjugate scalars only). Cell i is a solid unknown iff κ_s,i = 1 − κ_i > 0 and the
cell's solid belongs to a conjugate body (§2.7).
- It is stored in a second registered field `<name>_solid` (ψ_s).
- That field is allocated on the first `set_scalar_solid` call.
- Two full fields are chosen over a compact side array. In a packed bed the solid interior is 50–60 %
  of the cells, so a "side array" would be a full field anyway, and every stencil near an interface
  would pay an indirection (D8).

### 1.4 Discrete balance (per unit **full-cell** volume V, backward Euler)

Notation for fluid unknown i:
- a_a^±(i) are the snapped scalar apertures of its high/low a-faces. In flow's face-field
  convention, a_a^−(i) = `sa_a(i)` and a_a^+(i) = `sa_a(i+e_a)`.
- φ runs over the facets of cell i (≤ 2).
- α_φ = |A⃗_φ|/V is the facet area per cell volume.
- u_p,φ = Σ_k w_k c_{i+o_k} is the probe interpolation (offsets o_k, weights w_k, Σ w_k = 1).
- s_φ is the probe distance.

The fluid row is

    (κ_i/Δt)(c_i − c_i^n)
      + Σ_a Σ_± Λ_f w_a a_a^± (c_i − c_{i±e_a})                      [faces; 0 where a = 0]
      + Σ_φ α_φ Q_φ                                                  [walls: flux OUT of the fluid]
      + (1/V) Σ_{f ∈ impl(i)} F_f^out c_up(f)                        [implicit FOU faces, §6]
      + B_i                                                          [domain faces, below]
    = κ_i S_i − (1/V) Σ_{f ∈ expl(i)} F_f^out c*_f(c^n)               [explicit faces, §6]

Here F_f^out is the volume flux through face f out of cell i, and c_up is the upwind value.

**Wall closure Q_φ** (eliminated per facet, probe gradient ∂c/∂n ≈ (u_p − c_Γ)/s):

| type | Q_φ (flux out of fluid, per area) | matrix part | RHS part |
|---|---|---|---|
| Neumann | −q_φ | none | + α_φ q_φ |
| Robin | G_φ (u_p − g_φ), G_φ = 1/(s_φ/Λ_f + 1/k) | + α_φ G_φ w_k at (i, i+o_k) | + α_φ G_φ g_φ |
| Dirichlet | same with G_φ = Λ_f/s_φ | same | same |
| conjugate (fluid row) | G_c (u_pf − u_ps), G_c = 1/(s_f/Λ_f + R_c + s_s/Λ_s) | + α G_c w_k (fluid stencil), − α G_c w_k (solid stencil) | — |

Derivation of the Robin line:
- D(u_p − c_Γ)/s = k(c_Γ − g) gives c_Γ = (D u_p + k s g)/(D + k s).
- So q_in = k(g − c_Γ) = (g − u_p)/(s/D + 1/k).

This is the prototype's elimination (`jc.py`), verified to order 2.0 at Bi = 0…∞ (log round 3).

Conjugate:
- Fluid-side flux: q = Λ_f(ψ_Γf − u_pf)/s_f.
- Solid-side flux: q = Λ_s(u_ps − ψ_Γs)/s_s, with the solid probe at x_φ − s_s n.
- Contact: q = (ψ_Γs − ψ_Γf)/R_c.
- Eliminating the two wall values gives q = (u_ps − u_pf)/(s_f/Λ_f + R_c + s_s/Λ_s).

The **solid row** of the same cell is

    (C_s K κ_s,i/Δt)(ψ_s,i − ψ_s,i^n) + Σ_a Σ_± Λ_s,face w_a a_s^± (ψ_s,i − ψ_s,i±e_a)
      + Σ_φ α_φ G_c (u_ps − u_pf) = 0

It carries the same q with the opposite sign, so the coupling is exactly conservative.

Notes on the solid row:
- Solid apertures are a_s = 1 − a^snap. A face carries solid flux only if both cells are solid
  unknowns.
- Λ_s,face is the harmonic mean of the two cells' Λ_s when their materials differ.
- A contact between a conjugate and a non-conjugate solid is adiabatic (documented approximation).
- Validated in 2-D (`conj.py`, log round 4): order 2 for D_s/D_f = 0.1–100, K = 3, h_c = 5 and
  capacity 2.
- Against Peters' directional one-field scheme (log round 7, Maxwell disc at d/h = 128), the
  two-field probe-flux scheme is 2–50× better in L∞ at every contrast and 2nd order in L∞.

**Domain faces** (B_i). The rows below apply to the inner cell next to a non-periodic global face,
with boundary-face aperture a_bf:

| type | effect |
|---|---|
| `'dirichlet'` value g_bf (scalar or per-face profile) | LHS + 2Λ w_a a_bf c_i, RHS + 2Λ w_a a_bf g_bf |
| `'neumann'` | nothing (the band toward the ghost is 0) |
| flow inflow face | requires scalar `'dirichlet'`; advective RHS + F_in g_bf/V |
| flow outflow face | advective outflow F_out c_i/V: explicit, or implicit by the small rule |

- A face that is periodic for the scalar but non-periodic for the flow is refused, as is the reverse.
- The solid field uses zero flux at non-periodic domain faces.

**Steady mode** (`solve_scalar_steady`).
- Drop κ/Δt and the c^n term.
- Advection is implicit FOU on every face (first order, v1; §6.7).

### 1.5 Mean-gradient (closure) mode

`set_scalar_mean_gradient(name, G)` changes the unknown to θ, periodic, with c = G·x + θ, written in
the frame moving with the fluid's mean velocity Ū. **Rule: wherever the discretization evaluates c at
a point x, it uses θ + G·x, and the G·x part goes to the RHS.** Concretely:
- **faces:** for each face of cell i that carries diffusion (both cells unknowns of the phase), the
  G-part Λ_face w_a a (G·(x_i − x_j)) moves to the RHS, giving RHS_i += Σ_a Λ_face w_a h'_a G_a
  (a_a^+ − a_a^−) over the coupled faces.
  - For the fluid, every face with a > 0 is coupled, so this equals (Λ_f/V) G·A⃗^snap.
  - For the solid it uses a_s and only the solid-coupled faces.
  - x_j − x_i is always the true displacement ±h'_a e_a, never a wrapped index difference.
- **Dirichlet/Robin probes:** RHS += α G_φ (g_φ − Σ_k w_k G·x_k), where x_k are the global centres
  of the stencil cells, unwrapped relative to cell i.
- **Conjugate:** RHS ∓= α G_c (Σ_s w G·x − Σ_f w G·x) in the fluid/solid rows.
- **Neumann:** nothing extra; the face term already produces n·∇θ = −n·G.
- **advection:** θ is advected; the linear part enters exactly through the source −(1/V) G·(U_i − κ_i V Ū).
  - U_i = Σ_{f∈i} F_f^out (x_f − x_i) is the volume-integrated cell velocity.
  - Ū = Σ U_i / Σ κ_i V.
  - The source sums to zero exactly, which makes steady problems compatible.
- **domain faces:** a non-periodic face on an axis with G_a ≠ 0 is refused. A Dirichlet face on other
  axes uses θ_bf = g − G·x_bf.

### 1.6 Properties the discretization must have (each is gated in §11)

1. **Conservation.**
   - Every face flux and every facet flux is one number, entering two rows with opposite signs.
   - Hence Σ_i V·row_i = ΔM − Δt·(wall + domain + source terms) exactly; the budget identity holds to
     round-off.
   - The left null vector of the steady pure-Neumann operator is **1** over the rows.
2. **Constant preservation.** c ≡ const stays constant under Neumann walls and a face flux F that is
   discretely divergence-free.
3. **Linear exactness (conjugate, Λ_s = Λ_f, K = 1, R_c = 0).**
   - ψ = G·x is reproduced to round-off:
     - two-point faces are exact on linear data;
     - trilinear probes are exact;
     - the elimination returns −Λ G·n;
     - the snapped area-vector identity closes each cell.
   - This needs a_f + a_s = 1 **after** snapping, hence the both-ends snap (§2.3).
4. **Symmetry structure.**
   - The 7-point part without advection is symmetric.
   - The probe couplings are not symmetric and have positive off-diagonals.
   - The surrogate (§4.3) is SPD.

---

## 2. Geometric record and its construction

### 2.1 The source-agnostic contract

A **source** gives the core kernel one cell's local geometry and gets back a `CutCellGeometry`.

- **SDF source** (flow now, amr later): 15 PL samples of φ — 8 corners, 6 face centres and the centre
  of the cell. Container-free; values in internal length units.
- **Plane source** (VoF PLIC later): a plane (m, α) per cell. Signature reserved, not implemented
  now (§7.4).

`CutCellGeometry` holds (double, cell-centred frame, internal lengths):
- `kappa` ∈ [0, 1];
- `aperture[6]` (−x,+x,−y,+y,−z,+z; **unsnapped**);
- `areaPL[3]` (net PL facet area vector, into the fluid);
- `areaSum` (Σ |pieces|);
- `nFacet` ∈ {0, 1, 2};
- `facetArea[2][3]`, `facetCentroid[2][3]`;
- `kind` ∈ {none, single, gap, thinSolid}.

### 2.2 The fan-tetrahedron PL model (D4)

**Samples.** The sample values are built as follows:
- corner (i±½, j±½, k±½): the mean of the 8 surrounding cell-centred SDF values, summed in a FIXED
  order keyed to the corner's lowest cell, `((s000+s100)+(s010+s110)) + ((s001+s101)+(s011+s111))`,
  then × 0.125;
- face centre: 0.5·(s_lo + s_hi);
- cell centre: s_i.

These equal the trilinear values the shipped `ccFaceOpenMS` samples. The fixed summation order makes
a shared corner or face bitwise identical from both cells and on every rank.

**Tetrahedra.**
- Each face is split into 4 triangles (corner, next corner, face centre), the same triangle fan as
  `ccFaceOpenMS`.
- Each triangle is coned to the cell centre, giving 24 tetrahedra of equal volume V/24.
- φ is linear on each tet. The model is continuous, and conforming across cells because shared faces
  have identical fans.
- **"Fluid" means φ > 0 strictly.** A vertex with φ = 0 is solid. This makes a wall lying exactly on
  a face close that face, so κ = 0 never coexists with an open face.

**Positive fractions** (robust case formulas; no subtraction of close numbers):
- triangle with values (a, b₁, b₂), a > 0 ≥ b: F = a²/((a−b₁)(a−b₂)).
  - Two positive vertices: F = 1 − b²/((b−a₁)(b−a₂)).
- tet, one positive vertex a: F = a³/((a−b₁)(a−b₂)(a−b₃)).
- tet, three positive vertices: F = 1 − b³/((b−a₁)(b−a₂)(b−a₃)), where b ≤ 0 is the negative vertex.
- tet, two positive (a₁, a₂ > 0 ≥ b₁, b₂). All terms are non-negative:

      F = [a₁²a₂² − (a₁+a₂)(b₁+b₂)a₁a₂ + b₁b₂(a₁²+a₁a₂+a₂²)] / [(a₁−b₁)(a₁−b₂)(a₂−b₁)(a₂−b₂)]

  Check: (1, 1, −1, −1) gives 8/16 = ½.

**Outputs.**
- κ = (1/24) Σ_tets F_tet.
- aperture_f = ¼ Σ_{4 triangles of f} F_tri.
- **Pieces.** In each tet with 0 < n₊ < 4, the zero set is a triangle or (n₊ = 2) a planar quad.
  - Vertices are edge crossings x = x_p + t(x_n − x_p) with t = φ_p/(φ_p − φ_n) ∈ (0, 1].
  - The quad vertices are in the order X₁₁, X₁₂, X₂₂, X₂₁.
  - Area vector: triangle ½(X₁−X₀)×(X₂−X₀); quad ½(X₂−X₀)×(X₃−X₁).
  - Each area vector is oriented toward a positive vertex.
  - Centroid: the triangle mean, or the area-weighted mean of the quad's two triangles.
  - Pieces with area < 1e-14·min A_a are dropped.
- areaPL = Σ piece vectors; areaSum = Σ |piece|; ρ = |areaPL|/areaSum.
- **Exact identity** (the PL divergence theorem): Σ_a A_a (aperture_{a+} − aperture_{a−}) e_a = areaPL.
  It holds to round-off (unit gate).

**Facet classification.**
- ρ ≥ 0.5: `single`. One facet; centroid = areaSum-weighted piece centroid.
- ρ < 0.5: split the pieces into two groups by sign(n_piece · n_ref), where n_ref is the normal of
  the largest piece. Each group gives one facet (area vector = its sum, centroid = its weighted mean).
  If a group is empty, use `single`.
  - `gap` if (x̄_B − x̄_A)·n_A > 0 (fluid between two walls).
  - `thinSolid` otherwise (solid between two fluids). The single fluid DOF spans both sides; this is
    a resolution error that is counted and warned about (§9).
- Smooth single bodies at R ≥ 8h give ρ ≥ 0.9 everywhere. Edges and corners of CSG solids give
  0.5 ≤ ρ < 0.9 and stay single-facet with the averaged normal.

### 2.3 Snapping and the operator's area vector (D5)

**Snapping.** Each face aperture is snapped once, by a face kernel (one value per face, shared by
both cells): a < 1e-3 → 0, and a > 1 − 1e-3 → 1.
- The 1e-3 floor is the shipped pressure constant; the measured conditioning lesson in
  `mac_cutcell.hpp:164` applies equally.
- Snapping at **both** ends makes a_s = 1 − a^snap exact. Linear exactness (§1.6.3) needs this.

**No centre gate.** The pressure openness `ox_` closes every face whose centre sample is ≤ 0. Its
reason is the masked staggered velocity DOF there (register: "ungated aperture opens masked-DOF
faces"). That reason does not exist for a cell-centred scalar.
- The gate would make an aperture up to ~0.5 wrong at every centre-solid face, an O(1) flux error on
  a surface set.
- So the scalar keeps its own fields `sax/say/saz`, and `ox_` is never touched.

**Operator area vector.** For a single facet,
`A⃗_φ := Σ_a A_a (a^snap_{a+} − a^snap_{a−}) e_a`, so the cell's discrete divergence identity is
exact by definition.
- n_φ = A⃗_φ/|A⃗_φ| and α_φ = |A⃗_φ|/V.
- If |A⃗_φ| < 1e-14·max A_a, there is no facet.
- For two facets, use the two PL group vectors and add the snap residual
  ΔA = A⃗^snap − areaPL to the larger facet, so the two still sum to A⃗^snap.
- The **centroid** is always the PL centroid. It is the true centroid of the discrete facet, accurate
  to O(h²) on curved walls.

### 2.4 Unknown sets

- **Fluid unknown:** κ > 1e-14 **and** Σ_f a^snap_f > 0.
- **Sealed:** κ > 0 with all snapped apertures 0. Not an unknown; its κ and count go to the census.
- **κ = 0 ⇒ all apertures 0.** This holds by the strict-sign rule; the kernel asserts it in debug
  builds.
- **Ghosts.** After the build, κ and the flags are exchanged with the ordinary G = 2 field halo
  (`fillGhosts`). Flags in ghost cells beyond a **non-periodic global face** are then zeroed (rank
  rule `touchesGlobalFace`), because the halo wraps periodically.
- **Apertures are not exchanged.** The face kernel computes every face of every inner cell from the
  SDF, including the high-side face at ghost index `ext−G`. This avoids the high-face wrap trap
  (SCALING_ISSUES #3) and is bitwise equal to the neighbouring rank's value.

### 2.5 Where and when it is built

`Solver::ensureScalarCutGeometry()` builds lazily, on the first cut-cell advance or steady solve
after the geometry exists. It is invalidated by `set_solid`, `set_solid_from_scene`,
`set_pressure_geometry` and `redistribute`. Kernel sequence:
1. `scgApertures` (MDRange3 over faces).
2. `scgCells` (MDRange3 over inner cells). Cells whose 15 samples are all of one sign are skipped:
   κ = 1 or 0, no facet.
3. Count + `parallel_scan` → compact facet list in x-fastest cell order, plus the cell→facet CSR.
   This order is deterministic.
4. Halo κ/flags.
5. Fluid probe ladder per facet (§3).

Prerequisite (WO-2, gated): `sdf_` must carry **both** ghost layers on every path:
- single-rank periodic wrap;
- MPI `GridHalo` exchange;
- `extendSdfDomainGhosts`, including the `'slip'` mirror.

The 15 samples of an inner cell read sdf at ±1, and the probe clearance test reads sdf at ±2.

### 2.6 Why not the alternatives (κ, centroid)

- **`plicVolume(n, φ_c)`.**
  - Exact for planes, but not consistent with the marching-squares apertures on curved walls, edges or
    gaps.
  - It can give κ = 0 next to an open face, and κ > 0 behind all-closed faces. Those cells break
    explicit advection (division by κ) or leave orphan rows.
  - The fan-tet κ is the divergence-theorem volume of the same PL surface that defines the apertures.
    It is exactly consistent, bounded in [0, 1], and κ > 0 whenever any aperture is.
  - Plane sources (PLIC) still use the plane, because their κ is the colour.
- **4³ subsampling (`cs_`).** Quantized to 1/64, so slivers read κ = 0 with open faces (the
  `vof/cutcell.hpp` rule-1 trap). Not consistent with the apertures.
- **Centroid = projection of the cell centre on the plane.** Up to O(h) off the actual facet in
  corner-cut cells. A Robin/Dirichlet flux evaluated O(h) tangentially away is an O(1) truncation in
  those cells. The Robin solution gains only one order from that, so the result would be first
  order. The PL centroid is O(h²).

### 2.7 Body id per facet and material per solid cell

- **Facet body id:** the scene owner at the facet centroid (`q.owner(p)`). This follows flow's
  hydro-force precedent: query at the point, not from the cell-centred `cutOwner_`. With raw-SDF
  geometry, the id is 0.
- **Solid DOF material:**
  - interior solid cells: `cutOwner_` (inner grid; 0 without a scene);
  - cut cells: the body of the cell's conjugate facet, or the larger-area one if both facets are
    conjugate.
- A cell where two conjugate materials share one solid DOF takes that material. This is a resolution
  limitation, counted in the census.

---

## 3. Probe stencil, fallback ladder, precision

### 3.1 Probe distance (D6)

`S(n) = ½ Σ_a |n_a| h'_a` is the support half-width of a cell along n. A cell whose centre is at
signed distance ≥ −S(n) from a plane intersects the plane's positive half-space.

Every trilinear stencil cell of a point at distance s from a planar wall lies within one cell
extent, so its signed distance is ≥ s − 2S(n). That cell is therefore at least cut iff s ≥ S(n).

**Choice: s = σ S(n) with σ = 1.1.**
- Isotropic range: s ∈ [0.55h, 0.953h]. The brief's constant √3/2·h is the worst case of S(n).
- The prototypes favour short probes: 0.7h gave 3.5e-5, 1.0h 1.5e-4, 1.5h 4.5e-4.
- So the minimal guaranteed distance is used, normal by normal.

**New evidence (this note, `jc.py probeN1.1`, 2-D disc, ND = 128).** The normal-dependent
s = 1.1·½(|n_x|+|n_y|)h matches the constant 0.7h:

| Bi | probeN1.1 | probe 0.7 |
|---|---|---|
| 0 | −6.19e-5 | −6.19e-5 |
| 1 | −5.59e-5 | −5.60e-5 |
| 10 | −7.42e-6 | −8.42e-6 |
| ∞ | +3.62e-5 | +3.48e-5 |

Both have order 2.0 (1.5–2.5 at small Bi, identical for both) and zero fallbacks. A probe distance
that varies along the wall does not cost order.

**Reach.** With σ ≤ 1.5 and isotropic cells, |p − x_i|_∞ < 2h, so the trilinear stencil stays within
±2 cells, which is the G = 2 halo. The kernel still checks reach explicitly (anisotropic cells) and
fails the rung if it exceeds 2.

### 3.2 Stencil and validity

- **Trilinear stencil.** p is expressed in index space relative to the cell centre, ξ_a = p_a/h'_a.
  The stencil is the 8 cells around ξ (base = floor(ξ)), with trilinear weights. It reproduces linear
  fields exactly.
- **Validity of a candidate stencil**, for the phase being probed:
  - **(V1)** every cell with weight > 1e-12 is an unknown of that phase;
  - **(V2) clearance:** φ_phase(p) ≥ 0.5 s, with φ_phase = +φ (fluid) or −φ (solid), trilinear in sdf.
    V2 rejects probes entering another body, crossing into a gap's far wall, or crossing a concave
    corner.
  - **(V3)** reach ≤ 2 on every axis.

### 3.3 Ladder (fluid side; the solid side is identical with −n and the solid flags)

| rung | probe | stencil | accept if |
|---|---|---|---|
| **R0** | s₀ = 1.1 S(n) | trilinear, all 8 | V1 ∧ V2 ∧ V3 |
| **R1a** (longer probe) | s = 1.5 S(n) | trilinear | R0 failed V1 only; V1 ∧ V2 ∧ V3 |
| **R1b** (gap midpoint) | w = s₀ + φ_phase(p₀) (estimated gap); if w > 0.2 S(n): s = ½w | trilinear with invalid cells dropped, weights renormalized | Σ valid weights ≥ 0.5 ∧ V3 |
| **R2** (last resort, 1st order) | s = 0.5 S(n) | the cell itself, weight 1 | always |

- The rung is recorded per facet.
- **Gate:** on every smooth single-body geometry (G1–G9) every facet is R0, at every resolution and
  offset.
- R1b and R2 exist for contacts, thin gaps and under-resolved corners (G13). The 2-D data show that
  renormalization degrades accuracy, and that is acceptable only there.

### 3.4 Solid probe

The solid probe exists for conjugate facets only and is per scalar, because which bodies are conjugate
is per scalar.
- Same ladder with −n, the solid flags and −φ.
- The fluid probes are geometry-only and shared by all cut-cell scalars.

### 3.5 Storage and precision (D7)

The facet overlay `ScalarFacetOverlay` is SoA in device Views. Index f runs fastest
(`LayoutLeft` for the [nF][8] arrays), so a warp reads consecutive facets.

| member | type | per facet |
|---|---|---|
| `cell` | int64 | linear index on the G = 2 block |
| `alpha` | double | α_φ |
| `normal[3]`, `centroid[3]` | double | n_φ (into fluid), x_φ relative to the cell centre |
| `body` | int32 | instance id |
| `sF`, `offF[8]`, `wF[8]`, `rungF` | double, int32, double, uint8 | fluid probe; offsets are linear offsets on the block |
| `cellFacetStart` | int32 | CSR over the cut cells that carry facets (≤ 2 facets/cell) |

Per conjugate scalar, a parallel `sS, offS[8], wS[8], rungS`.

- Memory: about 175 B per facet, +105 B for conjugate.
- **Everything is double.** The scalar operator is cheap. Float storage silently breaks A·1 = 0 at
  contrast (register), so a float variant would buy nothing and cost a policed exception.

---

## 4. Operator application and the SPD surrogate

### 4.1 Stored true operator

**Bands.** Per phase, 7 bands AC, AW, AE, AS, AN, AB, AT in double, the legacy layout. The fluid
phase reuses the `ScalarField` bands; the solid phase has its own. They hold:
- κ/Δt (or C_s K κ_s/Δt);
- the face diffusion Λ w a;
- the implicit-FOU face terms;
- the domain-face folds.

Non-unknown rows are identity rows (AC = 1, all bands 0, RHS 0). Faces toward non-unknowns already
have a = 0.

**Overlay coefficients**, computed per advance from the per-body tables:
- `cw(f) = α_φ G_φ` for Dirichlet/Robin facets (0 for Neumann);
- `cc(f) = α_φ G_c` for conjugate facets;
- `rw(f)`, the RHS contribution (α G g, α q, or the mean-gradient shifts).

**Matvec y = A x** (residual analogous):
1. Exchange the ghosts of x: one `fillGhosts` per phase field (G = 2). One exchange covers both the
   7-point part and the ±2-reach overlay.
2. Band kernel: MDRange3 over inner cells, `y = Σ bands·x`.
3. Overlay kernel: RangePolicy over the CSR rows. One thread per cut cell loops over that cell's
   facets, so there are no atomics and the result is deterministic:
   - `y_f(i) += cw·Σ w_k x_f(i+o_k)`;
   - conjugate: `y_f(i) += cc·(Σ wF x_f − Σ wS x_s)` and `y_s(i) += cc·(Σ wS x_s − Σ wF x_f)`.

### 4.2 Per-advance tables

The physical → internal conversions of §1.2 are done here:
- per-body arrays (device, size = num bodies): type, k, g, q;
- per-material arrays: Λ_s, C_s K, R_c;
- then per-facet `cw/cc/rw`.

**Dirty flags.**
- The geometry version, Δt, any wall/solid/bc/tolerance setter, and a mean-gradient change mark
  the operator dirty. Dirty means: rebuild the bands and coefficients, and rebuild the ScalarMG
  coefficients without rebuilding its level table.
- Every step, the advection-dependent terms are recomputed: implicit-face bands and the lumped
  outflow diagonal.

### 4.3 The surrogate S (lumped probe; log round 5)

Level 0, per phase:

    S_ii = κ_i/Δt (C K κ_s/Δt)  +  Σ_faces Λ w a  +  Σ_φ α_φ G_φ   [Dirichlet/Robin, Σ w_k = 1 lumped]
           +  Σ_φ α_φ G_c   [conjugate]  +  (1/V) Σ_{impl f} max(F_f^out, 0)   [lumped outflow]
           +  2Λ w a_bf   [Dirichlet domain faces]
    S_ij  = −Λ w a   (face neighbours)
    S_{i_f, i_s} = −Σ_φ α_φ G_c   (same-cell fluid–solid coupling, conjugate)

- S is symmetric positive definite: an M-matrix, and the 2×2 coupling blocks are PSD.
- It is singular only in the steady pure-Neumann case (§5.1).
- *Amendment A2:* S, with its lumped outflow, is the preconditioner in transient mode and in steady
  mode at rest. In **steady mode with advection** the surrogate also keeps the FOU off-diagonals
  (S_adv: the true bands with SAC as the diagonal; §6.7, A2).
- Measured (2-D, exact surrogate solve): BiCGStab takes 8–9 iterations for Dirichlet and 2–5 for
  Robin, mesh-independent from ND 32 to 128. The centroid-distance surrogate needed 59–87 (ρ = 0.995).
  That is why the lumped wall term uses the **probe** distance.

---

## 5. Solver composition, MPI, stopping

### 5.1 Krylov driver (D9, D11)

**`scalarBiCGStab`** is a flow-local template in `src/scalar_krylov.hpp`. It is modelled statement
for statement on `CutcellMG::solveBiCGStab` (breakdown guards, stagnation guard, mean projection),
with three injected pieces:
- the matvec;
- the preconditioner `z = M⁻¹ r` (one V-cycle);
- `dot`/`maxabs`, which are `ccReduce3` over inner cells plus `MPI_Allreduce`.

Vectors are `ScalarVec {CCField f, s; bool solid;}`.
- Vector operations run over inner cells.
- Identity rows keep r, p, v, … exactly 0 at non-unknowns, so dots need no mask.
- Core's `MomentumSolver` is not used. It binds the operator to a CSR type and the vectors to a
  [0, n)+ghost-tail layout, and flow's blocks interleave ghosts.

**Stopping.**
- ref = max(max|b|, max|A x₀|). A x₀ = b − r₀ comes free.
- Stop when max|r_k| ≤ rtol·ref. The recurrence residual is checked each half-step.
- At exit, recompute the true residual once. If it is > 10·rtol·ref, restart once from it.
- The max-norm stop decision is reduction-order independent, so at np = 1 it is bitwise.
- Default rtol = 1e-10 (`set_scalar_tolerance`).
- maxit = 200. On non-convergence: keep the iterate, set the census flag, print a one-time stderr
  warning. On non-finite: raise.

**Initial guess.** c^n for transient; the current field for steady.

**Singular systems.** The singular case is steady, with no facet having G > 0, no Dirichlet domain
face, and periodic/Neumann boundaries.
- Project b ← b − mean_unknowns(b) over both phases' unknown rows.
- Remove the mean in the V-cycle as `CutcellMG::removeMean` does: rhs at each level, and the final
  correction.
- After the solve, fix the gauge so that Σ κV c (+ Σ C_s K κ_s V ψ_s) equals its pre-solve value,
  which is 0 for mean-gradient closures.
- The census reports the incompatibility |Σ b|/Σ|b| before projection.
- Under advection the right null vector is constant only up to the projection's divergence residual.
  The gauge shift then perturbs the solution by O(div): documented, negligible.

**Reductions per iteration:** ≤ 5 `MPI_Allreduce`; t·s and t·t are fused through the existing
`allreduceSum2` pattern. Host-resident Krylov scalars in v1. A device-resident variant (flow's PCG
precedent) is a later optimization, triggered by §11 G-perf.

### 5.2 ScalarMG (D10)

A new class in `src/scalar_mg.hpp`. It is used only by the cut-cell scalar path, one instance per
cut-cell scalar.

**Level table.** Reproduces `VelocityMG::init/initMpi`'s rule, verified identical by a unit test:
- level 0 is the scalar's own G = 2 block;
- in-place coarsening per axis while every rank's block is even on that axis and ≥ 4;
- the aspect rule with θ = 2 (`setMetric(h')`);
- coarse levels have g = 1;
- no telescoping in v1 (§13 Q3).

**Per level, per phase p:**
- x[p], rhs[p], res[p];
- face form AC[p], AFX[p], AFY[p], AFZ[p] (double);
- pin[p] (uint8);
- W (the conjugate coupling), when there are two phases;
- one `GridHalo<double>` per level.

**Coarse construction (rediscretized; level L, coarse cell C).**
- **Face coefficient:** t_a,C = w_a(L) · ⟨Λ a⟩, the plain average over the fine sub-faces of the
  product Λ·a. It is coarsened recursively with `coarsenOpenAvg` applied to the product field.
  For one phase with constant Λ, this is exactly CutcellMG's rule.
- **Mass:** m_C = average of the children's m (κ/Δt, or C K κ_s/Δt), with `restrictAvg`. Mass is
  extensive, so averaging is both the Galerkin and the rediscretized value.
- **Wall/interface term — SUPERSEDED by Amendment A1 below: use the FINE probe distance s_φ, not
  s_L.** The original text follows for the record.

  Direct gather from level-0 facets; no atomics:

      W_C^(L) = (1/N_L) Σ_{fine facets φ under C} α_φ · G_φ(s_L(φ)),
      s_L(φ) = 1.1 · ½ Σ_a |n_a(φ)| H'_a(L)

  - N_L is the number of fine cells per coarse cell.
  - G_φ(s) is the same closure as level 0 but with the level's probe distance: Λ/s,
    1/(s/Λ + 1/k), or 1/(s/Λ_f + R_c + s/Λ_s).
  - One kernel per level over coarse cells loops over the descendant fine cells in fixed order, and
    over their facets via the CSR. Cost O(N₀) per level, paid only when the operator is dirty.
- **Why the probe distance is rescaled** (*refuted by measurement; see A1*).
  - Face terms scale as Λ/H². An averaged wall term scales as (area/volume)·(Λ/s₀) ~ Λ/(H·h).
  - Plain averaging would therefore grow the wall-to-face ratio by 2× per level, 2^L at depth L:
    the coarse problem drifts toward a stiff Dirichlet wall that the fine problem does not have.
  - Rediscretizing s keeps the ratio level-independent. That is the consistent coarse surrogate.
- **Lumped outflow:** `restrictAvg` of the level-0 per-cell field, every step (one kernel per level).
  *Amendment A2:* this holds in transient mode only. In steady mode with advection the coarse levels
  carry the summed sub-face fluxes as band couplings, and only the open-face outflow ω_open rides on
  the mass (§6.7, A2).
- **Pins:** a coarse cell is pinned for phase p iff all its children are pinned (restrict flags with
  max). Pinned rows are identity rows with zero correction.
- **Dirichlet domain faces on level L:** 2Λ w_a(L) a_bf,C, using `touchesGlobalFace` on the level.

**Smoother.**
- Red-black Gauss–Seidel, colour = (gx+gy+gz) mod 2 from **global** indices, so the colours do not
  depend on the decomposition.
- Exchange before each colour.
- In cells where both phases are unknown and W > 0, solve the 2×2 block
  `[S_ff, −W; −W, S_ss][x_f; x_s] = …` (guard: det ≤ 1e-300·S_ff·S_ss → point update).

**V-cycle.**
- Symmetric: pre = 2 sweeps in R→B order, post = 2 in B→R.
- Restriction is `restrictAvg` (residual). Prolongation is `prolongAdd` (trilinear) followed by
  re-zeroing pinned cells.
- Bottom: 16 sweeps, plus mean removal if singular.
- The cycle is a fixed linear operator, so plain BiCGStab is correct (no FGMRES).

**Level rule.** It is the physics rule, as for momentum:
- Transient: let κ_A = 1 + 4 Δt' max(D', D_s') Σ_a w_a, where Λ_s/(C_s K) = D_s. If κ_A < 13, use
  level 0 only, i.e. 2 + 2 RB-GS sweeps as the preconditioner. Otherwise use the full table.
- Steady: always the full table.

*Amendment A3 (D-WO9-2, 2026-10-04) — the switch stays at κ_A = 13, now measured.* The value is
the named constant `ScalarMG::kFullTableKappa`. Level 0 alone against the full table on G-perf's
128³ bed (koren, Dirichlet spheres, host OpenMP at 2 threads, interleaved A/B, load 69–105), per
advance, with the BiCGStab iterations per step:

| κ_A (dt D/h²) | level 0 | full table |
|---|---|---|
| 7 (0.5) | 1.05–1.24 s, 9–10 it | 1.93 s, 9–11 it |
| 13 (1) | 1.23–1.27 s, 9–10 it | 1.67–1.70 s, 9 it |
| 25 (2) | 1.35–1.55 s, 10–12 it | 1.86–2.04 s, 9–10 it |
| 49 (4) | 1.51–1.93 s, 14–15 it | 1.84–1.98 s, 10–11 it |
| 97 (8) | 2.52–2.60 s, 19–21 it | 2.05–2.08 s, 10–12 it |

On cost alone the switch would sit near κ_A ≈ 49. It cannot: at κ_A = 13, level 0 fails G-iter.
R/h 16 takes 11 iterations on one of nine cold first steps, against the ≤ 10 bound, at 1, 2 and 4
threads. G-iter and G-perf both sit at κ_A = 13 exactly, so 13 is the largest switch that keeps
every gate. Below 13, level 0 is already the cheaper choice.

#### Amendment A1 (WO-4, 2026-10-03) — the coarse wall term uses the fine probe distance

**Decision (Q-A).**
- Coarse faces stay rediscretized: averaged Λ·a times w_a(L), 7-point.
- Every lumped wall or interface term on a coarse level is the **plain average of its level-0
  value**, evaluated at the **fine** probe distance:

      W_C^(L) = (1/N_L) Σ_{fine facets φ under C} α_φ G_φ(s_φ)

  - This is the gather of §5.2 with s_L(φ) replaced by s_φ. Keep the gather and its fixed summation
    order.
  - The same rule covers Robin (G at s_φ) and, in WO-7, the conjugate coupling: G_c with the fine
    s_f, s_s.
- **Galerkin RAP does not ship.**
  - Delete `setGalerkin`, the RAP assembly and its 8-colour sweep from `scalar_mg.hpp`, together with
    their test rows. The evidence stays in the log and in commit `c1e312e`.
  - **Q-B is therefore moot:** red-black Gauss–Seidel on 7-point coarse levels stays, and so does the
    2×2 block variant for conjugate.

**Why (first principles).**
- A coarse-grid correction never overshoots if the coarse operator is at least as stiff, in energy,
  as the fine operator applied to prolongated coarse functions, i.e. as the Galerkin energy
  e_Cᵀ Pᵀ S P e_C.
- Rediscretized faces satisfy this to leading order for trilinear P. CutcellMG relies on it, and
  here the no-solid and Neumann cases contract at 0.18–0.25.
- A prolongated coarse function is smooth across the fine cut cells beneath it. Its sink energy is
  therefore ≈ Σ W_i e_C², with W_i at the fine s_φ, **whatever H is**.
- Rescaling s by the level size divides that energy by ~2^L. The coarse problem becomes softer than
  any prolongated function, by 2^L, exactly where the coarse cell cannot resolve the near-wall dip.
  The correction then overshoots by that factor.
- It diverges once an under-resolved Dirichlet body is the dominant sink.
- The averaged term is the variational value for piecewise-constant P, and close to Pᵀ W P for
  trilinear P.
- §5.2's argument (keep the wall-to-face ratio level-independent) treated a coarse level as a
  rediscretization of the continuous problem. A coarse correction needs the fine operator restricted
  to the coarse space, and the two differ precisely where the geometry is under-resolved.
- So the amr register lesson ("coarsen the exact operator") **does** apply to the wall term. §12
  entry 7 and §5.3 are amended accordingly.

**Evidence.**

3-D, log WO-4 — stand-alone V-cycle contraction ρ(I − M⁻¹S):

| case | rediscretized s_L | averaged at s₀ | RAP |
|---|---|---|---|
| G1 128³ | 0.788 | 0.274 | 0.358 |
| G1-like 64³ | 0.470 | 0.222 | 0.303 |
| periodic box + Dirichlet sphere, 64³ | 3.105 | 0.503 | 0.284 |
| Dirichlet box, no solid | 0.175 | (= rediscretized) | 0.334 |

2-D prototype `tests/study/scalar_ibm/mg_coarse_wall.py`. It uses the same 4-colour Gauss–Seidel,
average restriction, bilinear prolongation and exact 4×4 bottom for all three columns, so only the
coarse operator differs. Entries are contraction [BiCGStab iterations to 1e-10]; n = 64 → n = 128:

| case | rediscretized s_L | averaged at s₀ | RAP (avg R, bilinear P) |
|---|---|---|---|
| box + disc R = L/4 (G1-like) | 0.444 → 0.682 [7, 8] | 0.190 → 0.229 [6, 7] | 0.295 → 0.360 [6, 7] |
| periodic + disc R = L/8 | **1.477 → 1.975** [8, 10] | 0.324 → 0.355 [7, 7] | 0.290 → 0.301 [7, 7] |
| periodic + disc R = L/32 | **2.078 → 2.653** [8, 10] | 0.658 → 0.705 [7, 8] | 0.609 → 0.648 [7, 7] |
| periodic 4×4 dense Dirichlet array | 0.356 → 0.586 [5, 7] | 0.183 → 0.256 [4, 6] | 0.116 → 0.155 [4, 5] |
| box, no solid | 0.233 → 0.253 [6, 6] | 0.233 → 0.253 [6, 6] | 0.361 → 0.417 [7, 7] |

Readings:
1. s_L overshoots, and worsens with every added level, both in 2-D and in 3-D (0.47 → 0.79 for
   6 → 7 levels).
2. Averaging at s₀ never diverges and is nearly depth-stable.
3. RAP is not better overall:
   - it wins on the dense array;
   - it loses on boxes without a solid, through its non-symmetric average/bilinear pair;
   - it costs 27-point coarse operators and an 8-colour sweep.
4. The tiny isolated sink (R = L/32) sits at 0.65–0.70 for **both** variational constructions. That
   is the limit of a coarse space that cannot represent the dip, not a defect of either.
5. BiCGStab hides all of it at these sizes. The 3-D runs agree: 13–16 iterations on the diverging
   periodic case (log D-WO4).

**Why not keep s_L and gate only on Krylov iterations** (the orchestrator's alternative)? The defect
is structural, and it grows where the tests do not look:
- (a) The overshoot factor ~2^L grows by one factor of 2 per added level. 256³ and 512³ runs get
  divergent modes even on G1-like geometry.
- (b) There is one bad mode per under-resolved Dirichlet body. In a dilute suspension of reacting
  particles, the Krylov count therefore grows with the number of bodies.
- (c) A contraction of 3.1 means an eigenvalue of M⁻¹S below 0 or above 2. The preconditioned
  operator is indefinite: a breakdown risk for BiCGStab, not merely a slower solve.
- (d) The cure is a one-line change that makes the construction safe by principle.

Krylov counts stay the **primary** gate. A contraction guard stays as well, because it is the only
gate that sees (a)–(c) at test sizes.

**Gate restatement** (replaces the §11 G-iter contraction line):
- **Primary:** BiCGStab iteration counts, G-iter as ruled in D-WO4-1 and D-WO4-3. Unchanged.
- **Contraction guard**, measured by the implemented power estimate:
  - **(C1)** ρ < 1 on every gate geometry. A stand-alone V-cycle that diverges fails, whatever the
    iteration count.
  - **(C2)** ρ ≤ 0.35 on Dirichlet/Robin box geometries (G1, G2, G3a), Neumann geometries and
    no-solid geometries, at every rung. Measured: 0.22–0.27 in 3-D, 0.19–0.26 in 2-D.
  - **(C3)** ρ ≤ 0.75 on periodic geometries whose only sink is one Dirichlet body. Add the periodic
    box + Dirichlet sphere case of D-WO4 as a `scalar_mg` row. Measured: 0.503 in 3-D, 0.32–0.71 in
    2-D.
- The old "≤ 0.3" was set without data. With red-black or colour Gauss–Seidel and average/trilinear
  transfers, a no-solid box already sits at 0.18–0.25 and RAP at 0.33–0.42, so that bound mostly
  measured the smoother/transfer pair.

**Implementation order (WO-4 continuation).**
1. s_L → s_φ in the coarse gather.
2. Remove the RAP path.
3. Rerun `scalar_mg` (C1–C3), G-iter, G10 at rtol 1e-13, and G12. Log the numbers.
4. If any G-iter row gets worse by more than 2 iterations, stop and report; do not tune.

**Left open (§13 Q2′, a fact, with a default).** The tiny-sink limit (C3) is inherent to geometric
coarse spaces. Default: accept it. If a dilute-suspension run (many bodies smaller than the coarse
cells) shows iteration counts growing with the number of bodies, the next step is operator-dependent
prolongation (BoxMG-type, so that coarse basis functions follow the dips). That is a new design,
not a knob.

### 5.3 Why this, and the register

- **Not CutcellMG.**
  - It is the pressure's single-field class (3.7 kLOC).
  - Adding a diagonal and a second phase would put the pressure path's bit-identity at risk.
  - Its operator is built from openness alone, with no diagonal entry point.
- **Not VelocityMG's coarse operator.** The staircase coarse levels pin the cut band, which amounts
  to Dirichlet-like coarse walls. That is wrong for Neumann walls and for the singular steady closure
  problems.
- **Rediscretized, not Galerkin.** The register holds two settled entries:
  - "Rediscretized coarse momentum operators fail — must coarsen the exact assembled operator
    (Galerkin)" (amr);
  - "Velocity-diffusion MG rediscretization diverges" (flow).

  In both, the fine level was an operator **outside** the rediscretizable family: the Robust-Scaled
  IBM stencil with its D_rescale row scaling, against a Neumann/openness coarse operator. Here the MG
  operates on the **surrogate**, which is in the family by construction: the same formula at every
  level. The non-family part (the probe off-diagonals) is handled by the outer Krylov on the true
  operator. This departure is recorded as a new register entry (§12, R7), with the trigger for
  switching to Galerkin RAP (§13 Q2).
  - **Amended by A1:** this holds for the face part only. The wall term must take its variational
    (fine-s) value. Being "in the family" per cell does not make a rediscretized coarse sink match
    the fine operator on the coarse space.
- **Coupled two-field levels rather than block-Jacobi per phase.** Per-body mean offsets, at high
  contrast and in steady periodic conjugate problems, are slow modes of any per-phase preconditioner,
  and their number grows with the number of particles. The coupled coarse problem carries them.

### 5.4 Parallel contract

- **Ownership.** A facet belongs to the rank owning its cell. The record is built from the
  exchanged SDF, so it is bitwise equal across decompositions.
- **Exchanges.**
  - Build: κ and the flags, once.
  - Per matvec: one G = 2 exchange per phase field.
  - Per MG sweep colour: one per level.
  - Per step: the face-flux ghosts. These are already filled by `advanceScalars` today.
- **Reductions.**
  - Per step: one MAX for C_bulk (§6.3).
  - Per BiCGStab iteration: ≤ 5.
  - MG mean removal: sums, singular case only.
- **Contract.**
  - np = 1 in an MPI build is bit-identical to the single-rank build.
  - np > 1 agrees to the Krylov reduction-order floor: ≤ 10·rtol relative in the max-norm, and
    iteration counts within ±1 per solve. This is flow's existing contract (`test_sdflow_mpi`).
  - The V-cycle itself is decomposition-independent except for the singular-case mean sums.
- **Device residency.** Everything runs on the device. The host only sees Krylov scalars and
  diagnostics.

### 5.5 Legacy untouched

The dispatch is the first statement in the per-scalar loop of `advanceScalars`:
`if (sc.cutcell) { advanceScalarCutCell(sc); continue; }`.
- No legacy kernel, field or call is modified.
- `ScalarField` gains `bool cutcell` and `std::shared_ptr<ScalarCutState> cut`. Members do not change
  arithmetic.
- G12 (state hashes) is the byte gate after every work order.

---

## 6. Advection and small cells

### 6.1 The face flux (D5)

F_a(i) is the internal volume flux through the −a face of cell i, taken from the field that **the
active projection made discretely divergence-free**, with **its** openness:
- staggered: `ox_·C[a].u·A_a`;
- collocated: the projected `uf_/vf_/wf_`, times the constraint openness of the active collocated
  scheme.

**One predicate** `Solver::scalarFaceFlux(a)` returns it. It must read the same openness as the
divergence kernel of the projection in force.
- Rationale: constant preservation needs Σ_f F = 0 per cell. The scalar's own apertures are not that
  constraint.
- The scalar apertures weight **diffusion only**.
- G9b is the falsifiable check, on every grid and collocated scheme.

### 6.2 Explicit faces

- Reconstructions:
  - `koren`: `sadv::tvd` on the 4-cell axis stencil;
  - `sou`: `sadv::sou`;
  - `fou`: `sadv::fou_flux`.
  This is the legacy reconstruction applied to κ-storage values.
- **A face uses explicit FOU** if any cell of its reconstruction stencil (two upwind and one
  downwind) is not a fluid unknown. That includes ghosts beyond non-periodic faces.

### 6.3 Small-cell classification (dynamic, per step; D12)

Out_i = Σ_f max(F_f^out, 0), and C_bulk = global MAX over **full** cells (κ = 1, all six a = 1) of
Δt·Out/V.

    small_i  ⟺  κ_i < 1  ∧  Δt·Out_i > max(½, C_bulk) · κ_i V
    face f implicit (FOU, at c^{n+1})  ⟺  either adjacent cell is small

- A cut cell stays explicit exactly when its own Courant number is no worse than the bulk's. So
  its explicit stability is the bulk's, which is the legacy assumption.
- Full cells are never reclassified.
- Flags are computed over inner cells and ghost layer 1 from F and κ, which already have ghosts.
  That needs no extra exchange and is deterministic.
- Implicit faces add the FOU terms to the bands of both cells, and their outflow to the surrogate's
  lumped diagonal.

### 6.4 Properties

- **Conservation.** Each face's advective flux is one number, explicit or implicit, used by both
  cells.
  - The per-step budget identity holds to round-off.
  - The physical drift equals the accounted solver defect Δt Σ r V. No fix-up (D13).
- **Constant preservation.** Holds whenever Σ F = 0 (§6.1).
- **Positivity.** The implicit part is an M-matrix. Explicit FOU on non-small cells exports ≤ the
  cell content. Koren keeps the legacy bulk behaviour.
- **Cost.** No extra exchange (FOU couplings are nearest-neighbour) and no extra solve: the implicit
  solve exists anyway.
- **Measured (2-D annulus, `advect.py`).**
  - Explicit κ storage blows up (min κ 4e-5…2e-4).
  - The split is stable at full-cell CFL, with mass to 1e-16 and positive.
  - Its L1 error equals unit storage's, which loses 2–12 % of the mass.
- At **no-slip** walls the face velocities of cut cells are O(h), so FOU there costs no order. The
  first-order local error matters only at slip walls and moving interfaces (§13 Q8).

### 6.5 Domain faces

These are §1.4's rows.
- An inflow face requires a scalar Dirichlet value (raise otherwise).
- Outflow is upwind.

### 6.6 Why not weighted state redistribution now; moving-geometry hooks

**WSRD** (Berger–Giuliani; AMReX) is 2nd order at cut cells, but:
- it needs a ~3-cell grown box, i.e. a second exchange or a G = 3 scalar block;
- it adds a neighbourhood/weight/limited-gradient pass;
- its accuracy gain sits where no-slip velocities make the advective flux O(h).

It is the recorded alternative for VoF interfaces and slip walls (trigger in §13 Q8).

**Moving geometry (hook only).** The record is a pure function of the SDF, so a per-step rebuild of
the moving bodies' band is possible. Two rules would then be needed:
- **fresh cells** (κ^n = 0 → κ^{n+1} > 0) are seeded with their fluid probe interpolant;
- **dead cells** (κ^{n+1} = 0) hand their mass κ^n V c^n to their probe stencil in proportion to the
  weights, which conserves it.

The geometric conservation law needs the wall-velocity flux that flow already has for moving bodies.
Not designed further here.

### 6.7 Steady mode with advection (v1; preconditioner superseded by Amendment A2)

- Implicit FOU on all faces: first order in advection, documented. The discretization stands.
- Deferred-correction Koren (flow's momentum precedent: implicit FOU plus lagged (Koren − FOU)) is
  WO-11, optional.
- *Refuted by WO-5 (log, Q-H):* "the outflow lumped onto the surrogate suffices, because G7b and G8
  involve no transport across the gradient". The Krylov method meets every error component, not
  only the solution's, so the preconditioner must carry advection wherever the operator does.
  Amendment A2 replaces the steady preconditioner.

#### Amendment A2 (WO-5 Q-H, 2026-10-03) — steady advection: the upwind couplings enter the surrogate

**Problem.** WO-5 measured the steady solve on the C3 problem (64³ periodic box, Dirichlet sphere
R = 16h, a source, the projected Stokes field rescaled to peak cell Péclet Pe_h = |u|h/D):

| Pe_h | 0 | 0.1 | 0.3 | 1 | 3, 10 |
|---|---|---|---|---|---|
| BiCGStab iterations | 13 | 40 | 89 | 200, not converged (1.7e-4) | 127, 200 (residual ≥ 1) |

Transient mode is unaffected: 5–6 per step advecting, the same as at rest. The first scientific use
of this code (dispersion closure problems in periodic beds, paper A4) needs steady solves up to
Pe_h ≈ 10.

**Cause (first principles).**
- Split A = S + K′, where S is the symmetric surrogate and K′ is essentially the skew (central)
  part of the FOU operator. For a Fourier mode of wavenumber k along the flow, S⁻¹K′ ≈ iUk/(Dk²).
- The preconditioned spectrum therefore lies on a vertical segment 1 ± iβ with
  β ≈ U/(D k_min) = Pe_h N/(2π), where N is the number of cells across the period.
- An optimal Krylov polynomial on such a segment contracts by σ = β/(1 + √(1 + β²)) per degree:
  about 23β degrees for 1e-10.
  - A BiCGStab iteration adds about one useful degree here. Its real-ω minimal-residual half-step
    barely damps eigenvalues near 1 ± iβ: |1 − ω(1 + iμ)|² ≥ μ²/(1 + μ²).
  - At 64³, β = 1.0 / 3.1 / 10 for Pe_h 0.1 / 0.3 / 1. Predicted ≈ 25 / 70 / 230 iterations;
    measured 40 / 89 / > 200.
  - The 2-D prototype shows the predicted β ∝ Pe_h·N directly: 18 / 36 / 82 iterations at Pe_h 0.1
    for N = 64 / 128 / 256 (table below).
- **The modes at fault are the low modes** (wavelengths of the period). Only the coarse levels see
  them, so advection in the level-0 smoother alone cannot help (measured: 35 / 119 / 222 iterations
  at N = 128). The coarse operators must carry it.
- **No symmetric surrogate can carry it.** Even with the upwind scheme's numerical diffusion added
  (S = sym(A)), β ≈ Pe_h N/(2π(1 + Pe_h/2)) → N/π at large Pe_h, which is 20 at N = 64.
- **Transient mode is genuinely different.** There, only the small-cell faces are implicit (≤ 1 %
  of the flux faces); every other face's advection is on the right-hand side. The advective
  coupling inside A is local, and nothing needs fixing.

**Decision.**
1. In **steady mode with advection** the V-cycle runs on the **advective surrogate**
   S_adv = S + the implicit-FOU couplings, on every level.
   - **Level 0** is exactly the true operator's 7 bands, with the diagonal replaced by SAC (the
     bands' diagonal plus the lumped wall term, §4.3). There is no new storage. The probe overlay
     stays lumped, as it is at rest.
   - **Coarse levels** keep everything of §5.2 and A1: rediscretized faces, the averaged wall term,
     the Dirichlet planes and the pins. They add the coarse advective couplings defined below, and
     the level is assembled into 7 bands.
   - **The path flag is global:** `advective = steady ∧ st.advecting`. `st.advecting` is already
     an all-rank MAX. Every rank takes the same path, so the extra exchanges below cannot deadlock.
2. **Transient mode, and steady mode at rest, are unchanged and stay bitwise.** They keep the WO-4
   face-form path; in transient mode WO-5's lumped outflow still rides on the coarse mass.
3. **Unchanged in every mode:**
   - BiCGStab on the true operator, the stop rule, maxit 200;
   - the level table and the level rule (steady = the full table);
   - RB-GS with global-parity colours in the symmetric order: pre 2 sweeps R→B, post 2 B→R, bottom
     8 R→B + 8 B→R;
   - `restrictAvg`, trilinear `prolongAdd` with pinned cells re-zeroed, the singular mean removal;
   - one V-cycle per preconditioner application.

**Coarse advection: the summed positive parts of the sub-face fluxes.** This is the
piecewise-constant Galerkin value.

The level-0 input is φ_a(i) = F/V through the LOW a-face of cell i (§6.1, `st.phi[a]`; signed,
positive along +a, internal 1/T). Per level L and axis a, two non-negative arrays live on the LOW
a-face of each cell, both per unit level-L cell volume:
- Qp_a(i): the flow toward +a, i.e. from cell i − e_a into i;
- Qm_a(i): the flow toward −a, i.e. from i into i − e_a.

Level 1 from level 0:

    Qp_a(C) = (1/r_a) · avg over the sub-faces f of C's low a-face of P₊(f)
    P₊(i)   = max(+φ_a(i), 0)   if cell i and cell i − e_a are both fluid unknowns, else 0
    (Qm_a and P₋ the same with −φ_a)

- "avg over the sub-faces" is exactly `coarsenOpenAvgCell`'s sum and division, in its fixed order,
  applied to the functor P (the `GuardedProduct` pattern). It is followed by one multiplication by
  1/r_a, with r_a ∈ {1, 2} the level's ratio on the face's normal axis.
  - Every division is by a power of two, hence exact. The only rounding is the fixed-order sum.
  - Together: Q_C = Σ_sub max(±φ, 0) · V/V_C, the summed sub-face flux per unit coarse volume. On a
    (2,2,1) level this is Σ over 2 sub-faces / 4 on x and y, and Σ over 4 / 4 on z.
- The guard is `advectionBands`' implicit-coupling predicate, so level-1 Q coarsens exactly the
  couplings present in the level-0 bands.
- Q is 0 on every non-periodic global face, because the ghost beyond it is not an unknown. That
  includes open (inflow/outflow) faces; their outflow is carried by the mass (below).

Level L ≥ 2 from level L − 1: the same, with P replaced by level L − 1's stored Qp/Qm.

After coarsening a level:
- `fill` each of the 6 arrays (exchange, or the single-rank periodic wrap);
- zero the low-face plane of every non-periodic axis, exactly as the face products are
  (`zeroPlanes(…, 0, g + 1)`).

Why this value:
- **It is variational.** It equals R A P on the FOU part for piecewise-constant P and
  R = Pᵀ/N_L (`restrictAvg`).
  - A sub-face interior to a coarse cell cancels exactly: it appears once as outflow of its upwind
    child and once as inflow of its downwind child.
  - Each boundary sub-face contributes its own upwind direction.
  - So the coarse level keeps the fine operator's symmetric part (the numerical diffusion Σ|F|/2
    per coarse face) in full. A1's failure mode was a coarse operator softer than the fine one on
    the coarse space, and this construction cannot have it. A1's wall term is the same kind of
    value.
- **Conservation and divergence.**
  - Every level is a conservative M-matrix with zero column sums: each coarse face flux enters the
    upwind diagonal and the downwind off-diagonal with the same value.
  - Row sums are zero up to the children's summed divergence, since the coarse divergence is
    exactly the sum of the children's.
  - Constants stay in the right null space, and ones in the left. So in the singular closure
    problem the mean removal on each level remains the exact compatibility projection.
- **It is also the rediscretized FOU, with the coarse face velocity taken as the sub-face average.**
  For a uniform flow, Qp_a = φ_a / cf_a exactly (a unit-test identity).

**Open domain faces (ruling D-WO5-3; the inflow/outflow rows of §1.4).**
- Outflow through an open global face is a diagonal-only term of the boundary cells. Inflow is
  right-hand side only.
- The operator keeps the outflow as its own level-0 cell field **ω_open**: the implicit outflow
  coefficient that the open-face rows add to AC, summed over the cell's open faces, and 0 elsewhere.
  - It is kept alongside `outflow`, which stays the transient path's input.
  - If an open-face row ever adds a negative diagonal (implicit backflow), ω_open takes max(·, 0).
    Report that case; it is a §6.5 question, not A2's.
- On the coarse levels ω_open rides on the mass, m = κ·idt + ω_open (idt = 0 when steady), and is
  `restrictAvg`ed per level exactly as `MassField` does today.
- This is exact. ω_open is nonzero only on the children next to the face, so the child average
  equals Σ_sub F_out · V/V_C.
  - The transient `outflow` average is not exact: it counts the faces interior to the coarse cell.
  - The transient path keeps it anyway: it is measured adequate there, and it is bitwise frozen.

**Assembly (coarse level L).** Pinned rows stay identity rows (AC = 1, bands 0). For an unknown
cell i:
- **Diagonal:**

      AC_L(i) = [AC as built today: mass (incl. ω_open) + Σ faces + wall + Dirichlet folds]
              + ((Qm_x(i) + Qp_x(i+e_x)) + (Qm_y(i) + Qp_y(i+e_y))) + (Qm_z(i) + Qp_z(i+e_z))

  - The face-form AC of the advective path must NOT contain the interior lumped outflow; Q carries
    it.
- **Off-diagonals:**
  - AW(i) = AFX(i) − Qp_x(i) and AE(i) = AFX(i+e_x) − Qm_x(i+e_x);
  - AS/AN use AFY, Qp_y, Qm_y in the same way, and AB/AT use the z arrays.
  - AF is the stored negative face coefficient −w_a(L)⟨Λa⟩ of §5.2.
- **Band convention** (the operator's): row(i) = AC x_i + AW x_{i−e_x} + AE x_{i+e_x} + AS x_{i−e_y}
  + AN x_{i+e_y} + AB x_{i−e_z} + AT x_{i+e_z}.

**Kernels (advective path, every level; AC = SAC at level 0).**
- **Colour sweep**, on the unknown cells of the colour:

      x_i ← (rhs_i − (((AW x_W + AE x_E) + (AS x_S + AN x_N)) + (AB x_B + AT x_T))) / AC_i

- **Residual:** r_i = rhs_i − (AC_i x_i + (((AW x_W + AE x_E) + (AS x_S + AN x_N)) + (AB x_B +
  AT x_T))).
- `applySurrogate` and `contraction()` use the band form, so the contraction instrument measures
  ρ(I − M⁻¹S_adv). WO-5's `scalar_mg` rows "C3 + advection" become the C4 rows of §11.
- Exchanges: one x exchange per colour and per residual on every level, as today.

**Storage and cost.**
- **Level 0:** nothing new in ScalarMG; one per-cell field ω_open in the operator.
- **Each coarse level:** Qp/Qm (6 arrays) and the bands AW…AT (6). They are allocated on the first
  advective build and kept: ≈ (12/7)·N doubles ≈ 14 B/cell over the hierarchy.
- **Per sweep:** 7 coefficient arrays instead of 4 (plus x and rhs). That is ≈ 1.5× the level-0
  sweep traffic, and ≈ 1.4× per BiCGStab iteration once the matvecs are counted.
- **Build:** per steady solve, one coarsening and 6 exchanges per coarse level.

**Envelope.**
- The gated range is Pe_h ≤ 10 (G-adv).
- Beyond it the solve degrades gracefully: in 2-D at n = 128, Pe_h 30 takes 12–24 iterations and
  Pe_h 100 takes 18–35.
- There is no refusal and no warning; non-convergence is already flagged (§5.1). FOU accuracy at
  high Pe_h is Q11's business.
- The census reports **`max_cell_peclet`**, the maximum over interior flux faces and axes of
  |φ_a| / (D′ w_a), where D′ is the fluid's internal Λ (`Inputs::lam`) and w_a = 1/h′_a² the
  metric weight of §5.2's level rule. It is the aperture-weighted face Péclet number |u_a| a h_a/D.
  - It costs one more entry in the existing MAX reduction of the advance.
  - G-adv rows are parametrized by it.

**Evidence: 2-D prototype `tests/study/scalar_ibm/mg_advection.py`.**
- Set-up:
  - A = aperture FV diffusion + the lumped Dirichlet wall (A1's level-0 term) + conservative FOU;
  - a discretely divergence-free node-stream-function flow, at angle atan 0.3 to the grid, no-slip
    at the discs;
  - periodic box; BiCGStab to 1e-10, random right-hand side;
  - the 3-D code's cycle: RB-GS 2 + 2 from global parity, average restriction, bilinear
    prolongation, A1 coarse wall; exact 4×4 bottom.
- A = S here. The probe off-diagonals are the same advection-independent gap as at rest; that is
  why the 3-D C3 count at rest is 13 against the prototype's 5.
- Entries: iterations at Pe_h 0 / 0.1 / 1 / 10, with [ρ(I − M⁻¹A)] at Pe_h 10; n.c. = not
  converged in 300.

| case | n | WO-5 symmetric surrogate | A2 advective surrogate |
|---|---|---|---|
| P1 periodic + Dirichlet disc R = L/4 (the C3 analogue) | 64 | 5 / 18 / 81 / n.c. | 5 / 6 / 8 / 13 [0.59] |
| | 128 | 6 / 36 / 194 / n.c. | 6 / 7 / 11 / 17 [0.74] |
| | 256 | 6 / 82 / n.c. / n.c. | 6 / 8 / 13 / 26 [0.84] |
| P2 Neumann 2×2 disc array, ε = 0.50 (closure, singular) | 64 | 4 / 19 / 68 / n.c. | 4 / 5 / 7 / 8 [0.29] |
| | 128 | 4 / 34 / 145 / n.c. | 4 / 6 / 7 / 9 [0.41] |
| | 256 | 5 / 74 / n.c. / n.c. | 5 / 7 / 8 / 13 [0.53] |
| P3 the same array, Dirichlet (reactive bed) | 64 | 5 / 7 / 23 / 151 | 5 / 5 / 6 / 7 [0.32] |
| | 128 | 5 / 10 / 52 / n.c. | 5 / 5 / 7 / 11 [0.49] |
| | 256 | 6 / 18 / 130 / n.c. | 6 / 6 / 10 / 14 [0.64] |
| P4 closed-streamline eddies (5× the mean flow), no solid, singular | 64 | 4 / 21 / 104 / n.c. | 4 / 5 / 7 / 13 [0.65] |
| | 128 | 4 / 46 / 260 / n.c. | 4 / 6 / 9 / 18 [0.78] |
| | 256 | 4 / 114 / n.c. / n.c. | 4 / 6 / 11 / 26 [0.87] |

Variants and probes. Entries are iterations at n = 256, Pe_h = 10, for P1 / P2 / P3 / P4, unless
the row says otherwise.

| variant | iterations |
|---|---|
| **A2 as specified** (summed positive parts, trilinear, 2 + 2) | 26 / 13 / 14 / 26 |
| coarse flux = summed **signed** flux, then upwinded ("net") | 26 / 13 / 14 / 26 (ρ within 0.01 everywhere) |
| piecewise-constant prolongation | 24 / 11 / 13 / 23 |
| 3 + 3 sweeps | 21 / 10 / 11 / 20 |
| F-cycle | 13 / – / – / 14 |
| W-cycle | 12 / – / – / 10 |
| advection at level 0 only (coarse = the WO-5 levels); n = 128, Pe_h 0.1 / 1 / 10 | P1 35 / 119 / 222; P2 33 / 91 / 176 |
| 16-sweep bottom on a 16×16 bottom grid; n = 128, Pe_h 0 / 1 / 10 | P1 6 / 10 / 16, P2 6 / 8 / 11, P4 4 / 9 / 16 (exact 4×4 bottom: 6 / 11 / 17, 4 / 7 / 9, 4 / 9 / 18) |
| beyond the envelope; n = 128, Pe_h 30 / 100 | P1 19 / 26, P2 12 / 18, P4 24 / 35 |
| fixed physics: P1 with U L/D = 640, i.e. Pe_h 10 / 5 / 2.5 at n = 64 / 128 / 256 | 13 / 15 / 18 |
| full GMRES (matvecs); P1 and P2, Pe_h 1 / 10 | P1 21 / 38, P2 16 / 23 (BiCGStab: 26 / 52 and 16 / 26 matvecs) |

Readings:
1. The advective surrogate removes the failure. Counts grow mildly with Pe_h and with depth at
   fixed Pe_h (×1.1–1.6 per doubling), and only ×1.15–1.2 per doubling at fixed physics.
2. Closed streamlines (P4) are no worse than open ones.
3. The coarse advection is what matters (the level-0-only row). How it is coarsened does not (the
   net row).
4. F- and W-cycles halve the count at depth. They are the recorded escalation (§13 Q18), not the
   default.
5. A 16-sweep bottom on a large bottom grid, as at np > 1, costs nothing measurable.

**Alternatives rejected, and why.**
- **(b) Pseudo-transient continuation.**
  - Each pseudo-step is a transient solve, whose preconditioned spread is β_t = Pe_h √Fo / 2
    (Fo = DΔt/h²).
  - Reaching the steady state needs Δt·λ_min ≫ 1, where λ_min ≈ D(2π/L)² (the slowest periodic
    mode). That means Fo ≳ (N/2π)², and at that Fo, β_t ≈ Pe_h N/(4π): the steady solve's own
    ill-conditioning, now paid on every step.
  - A smaller Δt trades it for O(N²) steps. Either way it is strictly dominated.
- **(c) GraphAMG on the assembled FOU operator.**
  - It is built for SPD operators: smoothed-aggregation prolongation (energy minimization) and a
    4th-kind Chebyshev smoother (register, voro). Chebyshev needs a real spectrum, which FOU at
    Pe_h ~ 10 does not have.
  - It has no MPI path, and it is a second, assembled operator representation rebuilt per solve.
  - A nonsymmetric AMG (AIR-type) is the route only if Pe_h ≫ 10 becomes a requirement (§13 Q19).
- **(d) Defect correction around a stronger solver.** This is a stationary iteration around the
  same inner preconditioner, and Krylov acceleration of that preconditioner is never worse (the
  register also rejects undamped DC as a fallback). The only stronger solver available is this
  advective MG, so (d) is A2 with a worse outer loop.
- **(e) Restricting Pe_h.** Kept only as the documented, gated envelope; not a refusal.
- **Advection at level 0 only.** It fails (35 / 119 / 222): the failing modes are the low modes.
- **A symmetric surrogate with the upwind numerical diffusion, sym(A).** β → N/π (theory above).
- **VelocityMG's construction** (`buildAdvCoarse`: restricted cell velocities, upwinded on the
  coarse grid).
  - It is conservative but not divergence-free on the coarse grid: its row sums are nonzero, which
    act as spurious coarse sources.
  - It needs a cell-velocity field the scalar path does not have.
  - The summed flux is free, exact, and built from the field the operator already holds.
- **Net (signed-sum) coarse flux.** Measured identical, so rejected on principle: it drops the
  exchange part Σ|F| − |ΣF| of the coarse symmetric part, which is A1's softer-than-fine failure
  mode. The positive-part construction costs only 3 more arrays per coarse level.
- **Downstream-ordered or line smoothers.**
  - A periodic torus with wrapping or closed streamlines has no downstream order.
  - Neither smoother is parallel inside a GPU block, and a line solve crosses MPI blocks.
  - They are not needed: point RB-GS degrades gracefully to Pe_h 100.
- **F-/W-cycles, 3 + 3 sweeps, piecewise-constant prolongation.**
  - Each lowers the count (table above), but each changes the cycle.
  - The V-cycle meets the requirement, keeps one cycle structure, and keeps the communication
    profile at scale: an F-cycle visits level L L+1 times, a W-cycle 2^L times.
- **GMRES.** It needs 12–27 % fewer matvecs at Pe_h 10, but costs restart storage and k reductions
  per iteration. BiCGStab stays (D9).

**Interactions.**
- **A1.** Unchanged: the wall gather is the same and enters AC as before. The C1–C3 rows carry no
  advection, so they are unchanged.
- **WO-6.** G8 (Taylor–Aris at Pe = 10, R/h = 32, peak Pe_h 0.3–0.6) is a steady, advecting,
  singular solve. WO-6 therefore depends on WO-5c, and its G-iter row is the singular-steady ≤ 30.
- **WO-7 (conjugate).** The solid never advects, so the solid phase keeps the face form.
  - In two-phase cells, the 2×2 block update takes the fluid neighbour sum from the bands on the
    advective path, and from the faces otherwise. That is one template parameter of the block
    kernel, and its face-form instantiation is the bitwise at-rest/transient path.
  - The W coarsening is A1's, unchanged.
- **WO-11 (deferred-correction Koren).** Every outer iteration solves the FOU system with this
  preconditioner. The outer rate is set by the Koren − FOU difference, not by A2.
- **MPI.** The V-cycle stays decomposition-independent: in-place coarsening keeps children
  rank-local, the bands are pointwise, and the colours come from global parity. G10 therefore holds
  by construction. The build adds 6 exchanges per coarse level per steady solve.

**Work order:** WO-5c (§10). **Gates:** G-adv, C4 and a G10 row (§11). **Open:** §13 Q18–Q21.

---

## 7. core/flow split; amr and VoF hooks (D14)

### 7.1 core (container-free, `PECLET_CORE_CC_HD`, double, namespace `peclet::core::scheme`)

`include/peclet/core/scheme/cut_cell_geometry.hpp`:
- `triPositiveFraction(a, b, c)`, `tetPositiveFraction(v0, v1, v2, v3)`;
- `struct CutCellGeometry` (§2.1);
- `CutCellGeometry cutCellGeometryFanTet(const double corner[8], const double face[6], double centre, const double h[3])`.
  Corner order is lexicographic in (x, y, z) bits; face order is −x, +x, −y, +y, −z, +z.
- `double snapAperture(double a)` (both ends, 1e-3);
- `void facetAreaVector(const double ap[6], const double h[3], double out[3])`;
- *(reserved, not implemented now)* `CutCellGeometry cutCellGeometryFromPlane(const double m[3], double alpha, const double h[3])`.

`include/peclet/core/scheme/probe_flux.hpp`:
- `double probeSupport(const double n[3], const double h[3])`;
- `void trilinearStencil(const double xi[3], int base[3], double w[8])`;
- `struct ProbeResult {int rung; double s; int n; int off[8][3]; double w[8];}`;
- `template <class Lookup> ProbeResult buildProbe(const double xw[3], const double n[3], const double h[3], const Lookup&)`.
  `Lookup` provides `bool unknown(int dx, int dy, int dz) const` and `double phi(const double xi[3]) const`,
  the latter phase-signed and in index space relative to the cell.
- `double wallConductance(double s, double L, double k)`, with k = 0 for Neumann and k = +inf for
  Dirichlet;
- `double interfaceConductance(double sf, double Lf, double Rc, double ss, double Ls)`.

Lattice offsets are returned, not DOF indices. flow converts them to linear block offsets; amr maps
them to leaf DOFs.

These are new headers only; no existing core header changes. Precedent: `scheme/cut_cell_closure.hpp`.

### 7.2 flow (new files; existing ones only as listed in §10)

- `src/scalar_cutcell_geometry.hpp` (record build kernels and the block `Lookup`);
- `src/scalar_cutcell_operator.hpp` (bands, overlay, matvec/residual, surrogate level 0, RHS,
  advection, small cells, mean-gradient terms);
- `src/scalar_krylov.hpp`;
- `src/scalar_mg.hpp`;
- `src/flow_ibm_scalars_cutcell.hpp` (domain header: `ensureScalarCutGeometry`,
  `advanceScalarCutCell`, `solveScalarSteady`, setters, diagnostics).
  - It is included only in the two instantiation TUs (G.8 rule).
  - A member template called from the bindings must be defined in `flow_ibm.hpp`.

### 7.3 amr hook

- Per leaf, amr samples its SDF at the leaf's 15 points and calls `cutCellGeometryFanTet` with the
  leaf's h. That gives κ, apertures and facets consistent by construction.
- amr's own `FaceGeometry.alpha` should then come from the same kernel. A coarse–fine face takes its
  aperture at the finer neighbour's lower corner (the amr register), i.e. it is computed from the
  fine side's samples.
- The probe ladder runs with an amr `Lookup`.
  - The 8 trilinear corners are fine-lattice offsets. Where a corner falls in a coarser leaf, amr
    substitutes that leaf's value through its own C/F interpolant (default cf = 1), which is amr's
    decision.
  - The facet stores DOF ids (int64) instead of block offsets.
- The coarse-level W gather (§5.2) and the 2×2 elimination are unchanged.

### 7.4 VoF hook (species with a partition coefficient)

The two-phase species problem is the **conjugate machinery with the solid replaced by the second
fluid**:
- κ_l = C and κ_g = 1 − C (times the solid fraction where a wall cuts the cell);
- apertures per phase from the PLIC plane (`cutCellGeometryFromPlane`);
- a PLIC facet;
- ψ continuity with K = Henry's coefficient and R_c = the interfacial resistance;
- each phase's species advected with the colour-consistent geometric (Weymouth–Yue) fluxes
  (precedent: `vof/energy_advect.hpp`).

The record is rebuilt per step for interface cells, with the moving-geometry fresh/dead rules. Two
things are decided **in the VoF package**:
- the rule for a face shared by two interface cells whose planes give different apertures;
- the aperture source in cells cut by both a wall and the interface.

### 7.5 Migration of the existing phase-change energy path (not now)

| today | under this framework |
|---|---|
| the per-cell Dirichlet mask, and the GFM plane-anchored rows (`scalarMaskGfm`/`Gfm2`) | PLIC-sourced facets with Dirichlet T_sat |
| the Robin interfacial resistance R_int | Robin with k = 1/R_int, g = T_sat |
| consistent ρc_p(C)T transport | per-phase advection with the colour fluxes |

The temperature becomes a two-field (liquid/vapour) problem with K = 1.
- The migration gate is the existing phase-change tests (`test_vof_phase_change{,_mpi}`) at equal or
  better accuracy.
- It is a separate package after the VoF species hook.
- Until then a cut-cell scalar refuses `set_phase_change_thermal`/`_energy` and the per-cell mask.

---

## 8. Python API (D15, D16)

Both `Solver` and `SolverColocated`. Names were checked against `../docs/NAMING.md`:
- strings for enumerations;
- `set_` setters;
- computed reductions as bare-name methods;
- array copies as `get_`;
- `instance=` as in `set_instance_motion`;
- American spelling;
- physical units.

### 8.1 Public tier

| member | semantics |
|---|---|
| `add_scalar(name, diffusivity=0.0, scheme='koren', iters=None, cutcell=False)` | `cutcell=True` selects this design. `iters` must be None then (ValueError otherwise; the legacy default stays 50). The diffusivity is **physical** on the new path. `cutcell` spelling follows `set_solid(..., cutcell_pressure=)`. **DEFAULT-PENDING-USER** (Q5). |
| `set_scalar_bc(name, face, type, value=0.0)` | Unchanged for legacy scalars. For a cutcell scalar, `value` may also be a per-face profile, shape (N_t1, N_t2) in the face's tangential axes in x, y, z order, this rank's slice under MPI (as `set_domain_bc_profile`). Consistency with the flow's domain BCs is checked at the first advance (§1.4). |
| `set_scalar_wall(name, type, value=0.0, coefficient=0.0, instance=None)` | `type` is `'neumann'` (value = flux into the fluid, c·L/T), `'dirichlet'` (value = wall c) or `'robin'` (coefficient = k in L/T, value = g; flux into fluid = k(g − c_wall)). `instance=None` means all bodies; an int needs a scene (ValueError otherwise). Default for every body: `'neumann'`, 0. |
| `set_scalar_solid(name, diffusivity, capacity=1.0, partition=1.0, contact_resistance=0.0, instance=None)` | Makes the body/bodies conjugate. The arguments are D_s (L²/T, the solid's own diffusivity), C_s, K (= c_s/c at equilibrium) and R_c (T/L, on ψ). Registers `<name>_solid`. Calling `set_scalar_wall` for the same instance afterwards turns it back into a wall (last call wins). Validation: diffusivity ≥ 0, capacity > 0, partition > 0, contact_resistance ≥ 0. |
| `set_scalar_source(name, source)` | Volumetric source S, c/T: a float or an (nx, ny, nz) F-order array on this rank's block. Weighted by κ (fluid only, v1). |
| `set_scalar_mean_gradient(name, gradient)` | Closure mode (§1.5). The registered field then holds θ. `(0, 0, 0)` turns it off. |
| `set_scalar_tolerance(name, rtol)` | Max-norm relative residual tolerance (default 1e-10). The same public tier as `set_velocity_residual_tolerance`. |
| `solve_scalar_steady(name)` | Steady solve now (§1.4, §6.7). Statistics go to diagnostics. |
| `get_scalar_solid(name)` | Copy of c_s = Kψ_s, (nx, ny, nz) F-order, NaN outside solid unknowns. |
| `scalar_wall_flux(name)` | Integrated flux **into the fluid** per body over the last advance/solve, (num_bodies,) in c·L³/T. Computed by a deterministic host-side sum in facet order plus MPI_Allreduce. |
| `scalar_mean_flux(name)` | Box-averaged total flux vector (3,), in c·L/T. The diffusive part is the face sum of −Λ a A (c_j − c_i) with c = θ + G·x, over both phases. The advective part is (1/V_box) Σ_i (U_i − κ_iVŪ)θ_i. Exact as a plane average for steady fields without wall sources. k_eff = −J·Ĝ/(D|G|); dispersion D* = −J/(φ G) (φ = discrete porosity). |

**Refusals** (RuntimeError at the first advance or solve, naming the reason):
- porous continuity on;
- moving scene instances;
- `set_phase_change_thermal`/`_energy` naming a cutcell scalar (ValueError at that call);
- the per-cell Dirichlet mask;
- the mean gradient with a non-periodic axis where G_a ≠ 0;
- periodic/non-periodic mismatch with the flow's domain faces;
- an inflow face without a scalar Dirichlet value.

**Call order.** All setters may be called at any time.
- Changes take effect at the next advance or steady solve (dirty flags, §4.2).
- The geometry is built lazily after `init_mpi` and the geometry setters (whose order is already
  enforced).
- `add_scalar` may precede or follow `set_solid`.

### 8.2 Diagnostics tier (`s.diagnostics`)

| member | returns |
|---|---|
| `scalar_census(name)` | dict: `num_unknowns`, `num_solid_unknowns`, `num_cut_cells`, `num_facets`, `num_two_sided`, `num_thin_solid`, `num_sealed`, `sealed_volume`, `probe_rungs` (fluid/solid R0, R1a, R1b, R2), `num_small_cells`, `num_implicit_faces`, `bulk_courant`, `krylov_iterations`, `krylov_residual`, `krylov_converged`, `mg_levels`, `steady_incompatibility` |
| `scalar_budget(name)` | dict over the last advance: `mass`, `mass_solid`, `d_mass`, `wall_in`, `boundary_in`, `source_in`, `defect` (= Δt Σ r V) and `identity_error` (the round-off residual of the identity in §1.6.1). It is computed on request from the stored state, so it costs nothing per step. |
| `scalar_facets(name)` | dict of arrays per facet: `centroid` (physical coordinates), `normal`, `area`, `instance`, `wall_value`, `flux` (q_in per area, physical), `probe_rung`. `wall_value` is c_Γ from the elimination: Robin (D u_p + k s g)/(D + k s); Dirichlet g; Neumann u_p + q s/D; conjugate ψ_Γf. These are local Nu/Sh maps (1st-order L∞ in the gradient, §13 Q10). |
| `scalar_geometry(name)` | dict: `kappa`, `aperture_x/y/z` (snapped), `unknown`, plus `kappa_solid` and `solid_unknown` if conjugate. F-order arrays. |

---

## 9. Diagnostics: what they are for and what they must show

- **Local Nu/Sh:** `scalar_facets()['flux']`. The wall gradient is first order in L∞ (the
  error-gain literature, L1 §5.5). Area-integrated fluxes are 2nd order.
- **Per-body Nu/Sh:** `scalar_wall_flux()`.
- **Conservation:**
  - `scalar_budget()` closes `d_mass − Δt·(wall_in + boundary_in + source_in) + defect` to round-off
    (`identity_error`);
  - a non-round-off `identity_error` is a bug;
  - a large `defect` is a loose tolerance.
- **Resolution warnings** (printed once per scalar, stderr, at the first build):
  - `num_thin_solid > 0` (a solid thinner than about a cell; one fluid DOF spans both sides);
  - R2 rungs > 1 % of facets;
  - `num_sealed > 0` with sealed_volume > 1e-6 of the fluid volume.
- **Optional post-processor (not in this package):** a 2nd-order local wall gradient from a quadratic
  normal stencil (Schwartz 2006, the J–C normal-line quadratic). The 2-D prototype found it erratic
  as an *operator*. As a diagnostic it is harmless.

---

## 10. Work orders (dependency order)

General rules for every WO:
- **G12** (all 12 state hashes = `doc/scalar_ibm_baseline_hashes.txt`) and the full non-bench battery
  must pass before its commit.
- `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`.
- Stage named paths only.
- No env knobs; MDRange3/`ccFor3` for 3-D loops; no float.

**WO-1 — core kernels.**
- Where: core worktree `suite/core-scalar-ibm`, branch `scalar-ibm`. Build flow against it with
  `-DPECLET_SIBLING_PECLET_CORE=/home/frankp/Codes/suite/core-scalar-ibm`.
- Files: new `include/peclet/core/scheme/cut_cell_geometry.hpp`, `.../probe_flux.hpp`,
  `tests/test_cut_cell_geometry.cpp`, `tests/test_probe_flux.cpp`; CMake registration.
- Functions: §7.1, except the plane source.
- Gate: G-geom (a)–(d) of §11, plus the probe unit cases.
- Must not touch: any existing core header.

**WO-2 — flow geometry record.**
- Files: `src/scalar_cutcell_geometry.hpp` (new), `src/flow_ibm_scalars_cutcell.hpp` (new),
  `src/flow_ibm.hpp` (members `scg_`, `scgVersion_`; declarations; the include at the bottom),
  invalidation calls in `flow_ibm_geometry.hpp`/`flow_ibm_scene.hpp`/`flow_ibm_mpi.hpp`,
  `src/flow_bindings.cpp` (`diagnostics.scalar_geometry`, partial `scalar_census`),
  `tests/kokkos/test_scalar_cutcell_geometry.cpp`, `tests/kokkos_mpi/test_scalar_cutcell_geometry_mpi.cpp`.
- Steps:
  - the ghost-layer verification of §2.5 (a test that fails if layer 2 is stale on any path);
  - the kernels of §2.5;
  - the fluid ladder (§3.3) with the block `Lookup`;
  - the census.
- Gate:
  - G-geom (e)–(g);
  - R0 = 100 % on the sphere and pipe sets;
  - MPI np = 1, 2, 4: the geometry arrays, gathered by global index, are bitwise equal;
  - G12.
- Must not touch: `ox_/oy_/oz_`, `buildOpenness*`, `ccFaceOpenMS`, `cs_`, `cutOwner_` semantics, any
  pressure/momentum/VoF code.

**WO-3 — single-phase operator, Krylov, steady + transient diffusion.**
- Files:
  - `src/scalar_cutcell_operator.hpp` (new);
  - `src/scalar_krylov.hpp` (new);
  - `flow_ibm_scalars_cutcell.hpp`;
  - `flow_ibm_scalars.hpp` (the one-line dispatch of §5.5);
  - `flow_ibm.hpp` (`ScalarField` members, `ScalarCutState`, the `UnitScales` appends);
  - bindings for `add_scalar(cutcell)`, `set_scalar_wall`, `set_scalar_source`,
    `set_scalar_tolerance`, `solve_scalar_steady`, `scalar_wall_flux`, the `set_scalar_bc` profile,
    `diagnostics.scalar_budget/census/facets`;
  - tests under `tests/python/` (ctests) for G1, G2, G3, G7.
- In this WO the preconditioner is ScalarMG level 0 only (2 + 2 RB-GS). The face flux is taken as
  zero; advection arrives in WO-5. The branch is not merged before WO-5.
- Gate:
  - G1, G2, G3, G7a, G7b;
  - the budget identity on G3b;
  - G12.
  - Iteration counts are recorded, not gated.
- Must not touch: the legacy kernels in `scalar_transport.hpp`, `applyScalarBc*`, CutcellMG,
  VelocityMG.

**WO-4 — ScalarMG (one phase).**
- Files: `src/scalar_mg.hpp` (new; reuses `coarsenOpenAvg`, `restrictAvg`, `prolongAdd` as free
  functions), the wiring in `flow_ibm_scalars_cutcell.hpp`, a level-table unit test against
  VelocityMG.
- Gate:
  - G-iter (single-phase rows);
  - V-cycle contraction ≤ 0.3;
  - G10 on G1 and on the singular case;
  - G12.
- Must not touch: the CutcellMG/VelocityMG sources.

**WO-5 — advection and small cells.**
- Files: `scalar_cutcell_operator.hpp` (explicit RHS, implicit bands, classification, C_bulk),
  `flow_ibm_scalars_cutcell.hpp` (the `scalarFaceFlux` predicate).
- Gate: G9, G9b, G12; G10 on G9.
- Must not touch: projection/advection code, `ufAdvVelocity`.

**WO-5c — steady advective surrogate (Amendment A2, §6.7).**
- Depends on WO-5 and on the open-face follow-up of ruling D-WO5-3 (called `WO-5b` in the code:
  the open-face rows and G9c). It comes before WO-6.
- Files:
  - `src/scalar_mg.hpp`: the advective path — `Inputs` gains `advective`, the level-0 band views
    AW…AT, `phi[3]` and ω_open; lazy coarse Qp/Qm and band arrays; the Q coarsening; the band
    assembly; the band-form sweep, residual, `applySurrogate` and `contraction`;
  - `src/scalar_cutcell_operator.hpp`: the ω_open field, written by the open-face rows; the local
    max of |φ_a|/(D′ w_a);
  - `src/flow_ibm_scalars_cutcell.hpp`: wiring (`in.advective = steady && st.advecting`, the views,
    ω_open on the mass in place of `outflow`); `max_cell_peclet` through the existing MAX
    reduction;
  - `src/flow_bindings.cpp`: the census key `max_cell_peclet`;
  - tests: `tests/kokkos/test_scalar_mg.cpp` (rows u1–u6 below);
    `tests/kokkos_mpi/test_scalar_cutcell_solve_mpi.cpp` (problem `steady_adv`, plus the z = M⁻¹r
    bitwise check); `tests/python/test_scalar_cutcell_gates.py` (G-adv, as ctest
    `scalar_cutcell_gadv`).
- Steps:
  1. **Before any change**, save the references from the current build:
     - G-adv(a) at R/h 16, Pe_h 0.1, rtol 1e-13 (the steady field);
     - WO-5's zero-velocity inert arrays;
     - the transient advecting arrays: G9 at R_o/h 16, `fou` and `koren` at C 0.5, 50 steps; the
       annulus rows with D at dt·D/h² = 3 and 30.
  2. Operator: ω_open and the Péclet maximum.
  3. ScalarMG advective path, exactly per A2: the formulas, the summation orders, pins and ghosts.
  4. `scalar_mg` unit rows:
     - (u1) uniform flow along each axis, periodic box without a solid, including an anisotropic
       (2,2,1) level: Qp_a = φ_a/cf_a exactly and Qm_a = 0 on every level;
     - (u2) a divergence-free random field (node stream-function differences): on every level the
       advective row sums satisfy |Σ out − Σ in| ≤ 1e-13·max Q, and the column sums are 0 to
       round-off;
     - (u3) M-matrix: bands ≤ 0 and AC ≥ Σ|bands| − 1e-13·AC on every unknown row of every level
       (steady, Dirichlet sphere);
     - (u4) the level-0 advective `applySurrogate` of a random x equals the operator's band matvec
       with SAC as the diagonal, bitwise;
     - (u5) Q = 0 on both domain-face planes of a non-periodic axis on every level, open faces
       included;
     - (u6) the C4 contraction rows (WO-5's "C3 + advection" rows, restated).
  5. Run G-adv, the G10 row, the inert proof, G12, and the battery. Log every number.
- Gate:
  - G-adv (i)–(vi);
  - G10 (the G-adv row);
  - **inert, bitwise against step 1:** the zero-velocity arrays, and the transient advecting
    arrays (fields, iteration counts, census);
  - `scalar_mg` C1–C3 unchanged;
  - G12; the battery.
- Must not touch:
  - ScalarMG's transient and at-rest steady paths (the face form, `hasOutflow` on the mass);
  - CutcellMG / VelocityMG; A1's wall gather; the probe operator; the Krylov driver; the
    discretization.
- **Stop rules.** Stop and report with the numbers if:
  - any G-adv (i) row fails;
  - any inert array differs;
  - any provisional bound is exceeded.

  Do not tune: no extra sweeps, no cycle change, no Pe_h-dependent switch. The orchestrator's
  default next step is §13 Q18.

**WO-6 — mean gradient and closures.**
- Files: the operator/RHS θ terms, the steady moving frame, `scalar_mean_flux`, the binding
  `set_scalar_mean_gradient`.
- Gate: G8, G5b, the no-solid identity (k* = 1 to 1e-12); G10 on G8.
- *A2:* depends on WO-5c. G8 is a steady advecting singular solve, and its iteration count falls
  under G-iter's singular-steady row (≤ 30). Re-run G-adv(b) with the mean-gradient mode (θ,
  G = e_x) in place of the u_x − ⟨u_x⟩ source, with the same bounds.

**WO-7 — conjugate.**
- Files:
  - core: no change beyond WO-1;
  - flow: the solid field and its flags/apertures (per scalar), materials, ψ-form rows, the overlay
    coupling, the solid ladder, ScalarMG with two phases (W, 2×2 smoother), `set_scalar_solid`,
    `get_scalar_solid`, the census/budget solid terms.
- Gate: G4, G5a, G6; G10 on G6; G-iter (conjugate rows); G12.
- *A2:* on the advective path the fluid phase is in band form and the solid stays in face form.
  - The 2×2 block update takes the fluid neighbour sum from the bands. Make that one template
    parameter of the block kernel; its face-form instantiation is the at-rest/transient path and
    stays bitwise.
  - Extra gate row: G-adv(b) with conducting spheres (Λ_s/Λ_f = 10, K = 1) at Pe_h 1 and 10. It
    must converge, in ≤ 1.5× the insulating (b) count.

**WO-8 — contacts and thin features.**
- Gate: G13; the warnings of §9 fire on the thin-plate case.
- No new data structure: two-facet cells exist since WO-2. Fixes only.

**WO-9 — backends and performance.**
- Gate: G11; G-perf recorded in `doc/scalar_ibm_log.md`. Red flags go to §13 triggers.

**WO-10 — documentation.**
- `CLAUDE.md` (a scalar section: the opt-in, refusals, traps), the register entries of §12,
  `../docs/NAMING.md` rows for the new names (including A2's diagnostics key `max_cell_peclet`),
  the flow `doc/README.md` index.
- Gate: review only.

**WO-11 (optional, later) — steady deferred-correction Koren.**
- Gate: a 2-D-like periodic cylinder-array closure at Pe = 10, 100. Self-convergence order ≥ 1.7
  against FOU's ~1.
- *A2:* every outer correction solves the FOU system with the A2 preconditioner; nothing in ScalarMG
  changes. Record the outer iteration count; its rate is the Koren − FOU contrast, not A2's.

---

## 11. Gates

Rules:
- "Order" is the RMS over **3 random grid offsets** (prototype practice), between successive
  resolutions.
- An absolute bound marked *(prov.)* is provisional. The first passing run records the measured
  value in the log. The bound may then be tightened to 2× the measured value, but **never loosened**
  without an architect consult.
- Default solver rtol is 1e-10, except where stated.

**G-geom.**
- (a) 1000 random planar cuts:
  - |κ − `plicVolume`| ≤ 1e-14;
  - apertures within 1e-14 of the exact plane–square fractions;
  - the facet area vector and centroid within 1e-13 (relative to h², h) of `plicPolygon` +
    `polygonAreaCentroid` (core `vof/curvature.hpp`).
- (b) Random (non-planar) samples:
  - PL closure |Σ ± a A e − areaPL| ≤ 1e-14·max A;
  - κ ∈ [0, 1];
  - κ = 0 ⇒ all apertures 0.
- (c) Synthetic gap (two planes 0.3h apart) → `gap`, 2 facets, opposite normals. Slab 0.3h →
  `thinSolid`.
- (d) Probe ladder unit cases:
  - all valid → R0;
  - one stencil cell covered → R1a;
  - opposite wall at 0.8 s₀ → R1b;
  - no valid cell → R2;
  - the weights reproduce linear fields to 1e-14.
- (e) Spheres R/h ∈ {8, 16, 32} in flow:
  - |ΣκV/V_exact − 1| ≤ 2(h/R)², order ≥ 1.8;
  - |Σ|A⃗_φ|/(4πR²) − 1| ≤ 3(h/R)², order ≥ 1.8;
  - every cut cell `single` with ρ ≥ 0.9;
  - 0 sealed.
- (f) |A⃗^snap − areaPL| ≤ 2e-3·max A per cell.
- (g) Anisotropic h' = (1, 1, 2): (e) with h = max h'.

**G1 Dirichlet sphere (steady).**
- Setup: R/h ∈ {8, 16, 32}, box 4R, the exact field R/r imposed as a box Dirichlet profile, wall c = 1.
- Nu = (wall flux)/(2πRD) → 2.
- Gate:
  - order ≥ 1.7;
  - |Nu/2 − 1| ≤ 2e-3 at R/h = 32 *(prov.)*;
  - field L1 order ≥ 1.8, L∞ order ≥ 1.5;
  - the budget identity closes:
    - |wall flux − box flux − defect| ≤ 1e-12·|wall flux|;
    - |defect| ≤ 1e-6·|wall flux|;
  - R0 = 100 %.
- Anisotropic h' = (1, 1, 2), R/h_min ∈ {16, 32}: order ≥ 1.7.

**G2 concentric shells.**
- Setup: inner sphere R_i, uniform flux q (Neumann); immersed Dirichlet outer sphere R_o = 2.5R_i;
  R_i/h ∈ {6, 12, 24}.
- Exact: c = c_o + (qR_i²/D)(1/r − 1/R_o).
- Gate: order ≥ 1.7 of the area-mean c_Γ (from `scalar_facets`) on the inner wall; field L1
  order ≥ 1.8; |error| ≤ 2e-3 at 24 *(prov.)*.

**G3 Robin.**
- (a) Steady external sphere:
  - Da = kR/D ∈ {0.1, 1, 10, 100}, g = 0, box profile c = 1 − A R/r with A = Da/(1+Da);
  - Sh = flux/(2πRD) = 2A.
  - Gate: order ≥ 1.7 each; ≤ 3e-3 at R/h = 32 *(prov.)*.
- (b) Transient interior sphere:
  - the fluid is inside, Bi ∈ {0.1, 1, 10, 100, ∞};
  - μ = Dλ²/R² with λ cot λ = 1 − Bi (smallest root);
  - measured by the **BE late-time ratio** μ_h = (c^n/c^{n+1} − 1)/Δt on the total mass, with
    Δt·μ ≈ 1, after ≥ 30 steps. This inversion is exact for BE, so the result is the spatial
    eigenvalue.
  - Gate: order ≥ 1.7 each; ≤ 2e-3 at R/h = 32 *(prov.)*; budget identity ≤ 1e-13.

**G4 Maxwell conjugate sphere.**
- Setup: box 6R, R/h ∈ {6, 12, 24}, the exact exterior field as the box profile,
  Λ_s/Λ_f ∈ {1e-2, 1, 10, 1e3}.
- (a) At ratio 1, with rtol = 1e-12: max|ψ − G·x| ≤ 1e-10·|G|L.
- (b) Interior gradient, by a least-squares fit over full solid cells, against 3G/(ratio + 2):
  order ≥ 1.7, ≤ 5e-3 at 24 *(prov.)*.
- (c) A K = 3 variant with the same Λ_s exercises K. The interior is identical in ψ and
  c_s = 3ψ_s.
- (d) Iterations at ratios 1e-2 and 1e3 ≤ 2 × (ratio-1 count) + 5.

**G5 periodic simple-cubic array, c = 0.3, ND (cells per diameter) ∈ {16, 32, 64}.**
- (a) Conducting, Λ_s/Λ_f = 1e4:
  - k* = −J_x/(D G_x) via `scalar_mean_flux`;
  - the reference is Andrianov–Topol eq. 157 for SC (a₁..a₆ in L3 §2 B6), evaluated to 5 digits and
    recorded in the test (≈ 2.333; the finite-contrast offset is ≈ 2e-4 relative);
  - gate: ≤ 3e-3 at ND = 32 *(prov.)*; self-convergence order ≥ 1.5.
- (b) Insulating (Neumann only):
  - self-convergence order ≥ 1.7;
  - k* ≤ the Hashin–Shtrikman upper bound (1−c)/(1+c/2) = 0.6087;
  - the no-solid run gives k* = 1 to 1e-12.

**G6 transient composite sphere (conjugate).**
- Setup: core R, immersed Dirichlet outer sphere 2R, R/h ∈ {8, 16, 32}. Six cases, in ψ-form
  (Λ_s, C_sK, K, R_c), with Λ_f = C_f = 1:
  - (10, 1, 1, 0)
  - (0.1, 1, 1, 0)
  - (100, 0.5, 1, 0)
  - (3, 3, 3, 0)
  - (1, 1, 1, 0.2)
  - (5, 1, 0.5, 0.5)
- These are `conj.py`'s cases mapped by Λ_s = Ds·K, C_sK = gs·K, R_c = 1/hc. In API terms that is
  `diffusivity` = Λ_s/(C_s K), `capacity` = C_s, `partition` = K, `contact_resistance` = R_c.
- Reference: the l = 0 spherical analogue of `conj.py exact()` (j₀/y₀ radial functions,
  3×3 determinant). Measured by the BE late-time ratio.
- Gate: order ≥ 1.7 each; ≤ 2e-3 at R/h = 16 *(prov.)*.

**G7 pipe, an SDF along z, nz = 4 periodic, R/h ∈ {16, 32, 64}.**
- (a) Dirichlet decay j₀₁² = 5.783186 (BE ratio, no flow): order ≥ 1.8; ≤ 5e-4 at 32 (2-D: 1.35e-4).
  Neumann j′₁₁² = 3.389957 (dipole initial condition, mean removed): order ≥ 1.8; ≤ 5e-4 at 32.
- (b) Graetz Nu_T = 3.656793 (Pe → ∞ form of Frank's eigenproblem):
  - smallest β with K v = β M u_z v, by inverse iteration with `solve_scalar_steady`;
  - source S = u_z(x_c)·v_prev, with the analytic Poiseuille u_z at cell centres clipped ≥ 0 (an
    O(h³) effect);
  - Nu_T = βUR²/D;
  - gate: order ≥ 1.7; ≤ 1e-3 at 32 *(prov.)*.
- (c) Optional: the axial-conduction QEP at Pe = 10 → 3.576741 (`disc_bc.graetz` form, root-finding
  in λ).

**G8 Taylor–Aris.**
- Setup: G7's pipe; frozen Poiseuille face velocities written with `set_field('w', …)` (staggered,
  internal units, face centres; divergence-free because z-invariant); mean gradient e_z; steady.
- Measure: −⟨u′θ⟩ (interstitial) = Pe²/48 · D, with Pe = UR/D.
  - The test computes it from fields: U_i from F = `get_oz`·w·A, and κ from `scalar_geometry`.
  - It also checks agreement with `scalar_mean_flux` to 1e-10.
- Gate: |48·value/(Pe²D) − 1| ≤ 1e-3 at R/h = 32; order ≥ 1.8 (2-D: 3.2e-4 at ND 64).

**G9 advection in an annulus.**
- Setup:
  - coaxial cylinders R_i = 0.4, R_o = 1, nz = 4, R_o/h ∈ {16, 32, 64};
  - solid-body rotation face velocities via `set_field`;
  - a Gaussian blob carried one revolution;
  - schemes `fou` and `koren`;
  - bulk Courant 0.5 and 0.9.
- Gate:
  - (i) per-step budget identity ≤ 1e-13 relative;
  - (ii) |M_end − M₀ − Σ defects| ≤ 1e-12 M₀ (the conservation check proper); with rtol = 1e-13,
    the raw drift |ΔM|/M₀ ≤ 1e-8 (sanity);
  - (iii) positivity:
    - `fou`: min c ≥ −1e-12 max c₀;
    - `koren`: min c ≥ −1e-3 max c₀ and max c ≤ (1+1e-3) max c₀;
  - (iv) finite throughout;
  - (v) the test really exercises slivers: min κ over unknowns < 1e-2 at every resolution.
- The census implicit-face fraction is recorded.

**G9b constant preservation under flow's own projection.**
- Setup:
  - Stokes flow through an SC sphere array by `step()`;
  - grids: staggered, and collocated `'ghost'`, `'gauge-exact'`, `'plain'`, `'embed'`;
  - c ≡ 1;
  - 50 steps.
- Gate: max|c − 1| ≤ 1e-6 (wrong openness gives about 1e-2).

**G-adv steady advection** (Amendment A2; WO-5c).
- Pe_h is the census `max_cell_peclet` of the solve. Each row rescales its face velocity field so
  that Pe_h = 0.1, 1 and 10.
- Setups:
  - **(a)** WO-5's C3 problem: a periodic box 4R, a Dirichlet sphere (c = 1) with a source, and the
    projected Stokes field of that geometry (WO-5's harness); R/h = 16 (64³) and 32 (128³).
  - **(b)** Closure:
    - the periodic simple-cubic sphere array, c = 0.3 (G9b's geometry), with the Stokes field from
      `step()` (G9b's body force);
    - Neumann spheres; steady and singular;
    - per-cell source s = u_x − ⟨u_x⟩ over the fluid, which is the B-field source for G = e_x
      until WO-6's mean-gradient mode replaces it;
    - 32³ and 64³ cells per period.
  - **(b′)** The same with Dirichlet spheres (a reactive bed).
  - **(c)** G9c's channel (open inflow/outflow faces plus a solid), steady, Pe_h 1 and 10.
- Gate:
  - **(i)** Every row converges: rtol 1e-10 within maxit 200, with no census non-convergence flag.
    This is the falsifiable statement that Q-H is fixed.
  - **(ii)** Iterations *(prov.)*, at Pe_h 0.1 / 1 / 10:
    - (a) ≤ 20 / 25 / 40 at R/h 16, and ≤ 25 / 35 / 55 at R/h 32;
    - (b), (b′) ≤ 15 / 20 / 30 at both resolutions;
    - (c) ≤ 40.
  - **(iii)** Growth per doubling at fixed Pe_h ≤ 1.7× (2-D: ≤ 1.6×).
  - **(iv) C4**, the stand-alone contraction ρ(I − M⁻¹S_adv) on (a) at R/h 16: < 1 at every Pe_h
    (C1), and ≤ 0.90 at Pe_h ≤ 10 *(prov.)*. In 2-D it is 0.34–0.59 at n = 64 and up to 0.87 at
    n = 256.
  - **(v)** The steady budget identity holds to ≤ 1e-12 relative; for (c) it includes the open
    faces' advective fluxes.
  - **(vi)** The discrete solution is unchanged. At (a), R/h 16, Pe_h 0.1, where the WO-5
    preconditioner also converges: max|c_A2 − c_WO5| ≤ 1e-9·max|c|, both at rtol 1e-13, against
    WO-5c's saved reference. This is a one-time check, logged.

**G10 MPI.**
- Configurations: G1 (R/h = 16), G3b (Bi = 10, 20 steps), G6 case 1 (20 steps), G9 (`koren`,
  50 steps), G8, at np ∈ {1, 2, 4}.
- *A2 row:* G-adv(a) at R/h 16, Pe_h 1, with D-WO4-2's form.
  - np = 1 bitwise; np = 2, 4 within 1e-10·max|c₁| at rtol 1e-13; iterations ±1.
  - The advective V-cycle's z = M⁻¹r, for a fixed global-index r, is bitwise equal at
    np = 1, 2, 4.
- Gate:
  - np = 1 bitwise against the single-rank build;
  - np > 1: max|c − c₁| ≤ 1e-9·max|c₁|, iterations within ±1 per solve;
  - geometry arrays bitwise.

**G11 backends** (CUDA against OpenMP; G1, G6 case 1, G9).
- Geometry fields ≤ 1e-12 relative.
- Probe rungs identical.
- Solutions ≤ 1e-9 relative.
- Iterations within ±1.
- *Ruling D-WO9-1 (WO-9b, 2026-10-04):* the G9 **koren** rows gate the solution at ≤ 1e-7·max|c|,
  plus G9's integral quantities (i), (ii) and (iv), and (iii) where G9 gates it, met on **both**
  backends at their existing bounds. Reason (log, WO-9b Q-N): the Krylov dots' reduction order
  (thread count, backend) moves each converged iterate inside the stopping tolerance; FOU damps
  these differences, while the Koren update carries them, so no backend tolerance holds koren at 1e-9.
  The Koren stencil never reads a non-unknown (§6.2's guard).

**G12 legacy.**
- `tests/regression/state_hash.py`: all 12 hashes equal `doc/scalar_ibm_baseline_hashes.txt`.
- The non-bench battery passes.
- After **every** work order.

**G13 contacts** (WO-8).
- Setup: two Dirichlet spheres in a box with far field 0, R/h ∈ {8, 16, 32}:
  - gap R/32;
  - contact (overlap R/64);
  - and a conjugate pair (ratio 100) in contact.
- Gate:
  - runs without failure;
  - census two-sided > 0;
  - R2 ≤ 1 % of facets;
  - total-flux self-convergence order ≥ 1.0;
  - ≤ 50 iterations.
- Thin plate 0.4h: `num_thin_solid > 0` and the warning is printed.

**G-iter.**
- Steady G1, G2 and G3a (Da = 1): ≤ 20 BiCGStab iterations at every resolution, growth ≤ 3 per
  doubling.
- Singular steady (G5b, G8): ≤ 30, growth ≤ 5.
- Transient at Δt·D/h² = 1: ≤ 8 per step (warm start).
- Conjugate G4/G6: ≤ 30.
- V-cycle contraction on the surrogate: *restated by Amendment A1* as C1 (< 1 everywhere), C2
  (≤ 0.35 on box, Neumann and no-solid geometries) and C3 (≤ 0.75 on periodic isolated-sink
  geometries). Krylov counts are the primary gate.
- Steady with advection: G-adv (Amendment A2), including its C4 contraction row. The transient
  advecting counts of WO-5 are frozen bitwise by WO-5c's inert gate.

**G-perf** (recorded; red flags rather than failures).
- Matvec effective bandwidth ≥ 50 % of measured STREAM.
- Memory ≤ 300 B/cell per single-phase scalar.
- Scalar advance ≤ the pressure projection time of the same step on a 128³ bed at Δt·D/h² ≤ 1.
- *A2:* the advective V-cycle's time against the symmetric one on the same block (expected ≈ 1.5×;
  red flag above 2×), and the steady G-adv(b) solve time at 64³.
- Any red flag opens §13 Q9.

---

## 12. Register entries to add (flow area unless noted; with WO-10)

1. **Scalar transport stores κ (fluid fraction) in storage and sources.** Rejected: unit storage
   (first order; Neumann eigenvalue 11 % off at 32/D; mass loss 2–12 % under advection).
2. **Every κ > 0 cell with an open face is a scalar unknown.** Rejected: merging/linking slivers
   (Neumann limit becomes first order, log round 2).
3. **Scalar faces carry the plain aperture two-point flux.** Rejected: Johansen–Colella face-centroid
   interpolation (no measurable change, log round 3; costs symmetry and compactness).
4. **Immersed scalar walls use probe-flux with per-facet elimination of the BC; probe distance
   1.1·½Σ|n_a|h_a; ladder R0/R1a/R1b/R2.** Rejected:
   - the per-cell Dirichlet mask (order 0.7–1);
   - the Gibou linear ghost alone;
   - Papac/Gibou symmetric Robin (1.4–1.7);
   - the aperture + link hybrid;
   - the centroid two-point / series-resistance flux (= Liu–Fedkiw–Kang GFM, order 1);
   - the quadratic normal probe (erratic);
   - AMReX's κ-dependent short probe with renormalized fallback (0.4–1.2);
   - a constant √3/2·h probe (needlessly long).
5. **Conjugate scalar transport is two fields on one grid in ψ = c/K, with the series-resistance
   2×2 elimination at fluid/solid probes (P2F).** Rejected:
   - Peters' directional one-field scheme (min-curvature T_ξ). Log round 7, Maxwell disc at
     d/h = 128: L∞ first order at every contrast, 2–50× worse in L∞ than P2F; no contact-resistance
     path; first-order Neumann limit.
   - one field + a side array;
   - a mixture one-field surrogate (arithmetic coarsening across the jump);
   - Das-style lagged partitioned coupling.
6. **Scalar cut-cell geometry is the fan-tetrahedron PL model on the marching-squares samples
   (κ, apertures, facet area vector and centroid from one polyhedron); the scalar apertures are
   ungated and snapped at both ends.** Rejected:
   - plane-cube `plicVolume` from (n, φ_c) for SDF sources;
   - 4³ subsampling;
   - the cell-centre projection as facet centroid;
   - reusing the gated pressure openness.
7. *(as amended by A1, 2026-10-03)* **ScalarMG preconditions the SPD lumped-probe surrogate with
   rediscretized coarse FACES and variational coarse WALL/interface terms: the plain average of the
   level-0 terms at the fine probe distance.** Rejected:
   - wall terms at the level's own probe distance s_L. This was the original design and is
     measured to overshoot: stand-alone contraction 0.79 at the finest G1 and 3.1 (divergent) on a
     periodic box with a Dirichlet sphere; 2-D 1.5–2.7. Its rationale (keep the wall-to-face ratio
     level-independent) is refuted.
   - Galerkin RAP: 27-point, a new smoother, and no better overall (0.28–0.36 in 3-D; 0.33–0.42
     without a solid);
   - extending CutcellMG;
   - VelocityMG's staircase coarse operator.
   - This **confirms**, for the wall term, the amr entry "Rediscretized coarse momentum operators
     fail": a coarse sink softer than the fine operator on the coarse space overshoots. The face part
     stays rediscretized.
8. **The scalar linear solve is BiCGStab + one ScalarMG V-cycle with a max-norm relative stop
   (default 1e-10).** Rejected: fixed RB-GS sweeps; (F)GMRES; the centroid-distance surrogate
   (ρ = 0.995).
9. **Scalar small cells under advection: dynamic implicit-FOU split relative to the bulk Courant
   number.** Rejected for now: WSRD (3-cell halo; revisit for VoF/slip walls); explicit κ storage
   (unstable); fully implicit FOU (diffusive).
10. **Scalar advection uses the projection's own constrained face flux; diffusion uses the scalar
    apertures.** Rejected: scalar apertures for advection (not divergence-free → constants not
    preserved).
11. **Scalar conservation is exact in the discretization; the solver residual is reported as a
    defect, never fixed up.** Rejected: a post-solve uniform mass correction.
12. *(core)* **Cut-cell geometry and probe kernels are container-free core headers
    (`scheme/cut_cell_geometry.hpp`, `scheme/probe_flux.hpp`).** Rejected: flow-private copies
    (amr and VoF would fork them).
13. *(A2, 2026-10-03)* **Steady advection is preconditioned by a V-cycle on the ADVECTIVE
    surrogate.** That is the level-0 bands, FOU couplings included, with the lumped wall; coarse
    advection is the summed positive parts of the sub-face fluxes (the piecewise-constant Galerkin
    value), swept by band-form RB-GS. Transient mode keeps the symmetric surrogate. Rejected:
    - the symmetric lumped surrogate (with lumped outflow) for steady advection. Its preconditioned
      spread is β ≈ Pe_h N/2π; in 3-D it took 13 → 40 → 89 iterations at Pe_h 0 / 0.1 / 0.3 and
      did not converge at 1 (log WO-5, Q-H);
    - advection at level 0 only: 2-D 35 / 119 / 222 iterations at Pe_h 0.1 / 1 / 10. The modes at
      fault are the low ones;
    - pseudo-transient continuation: the same spread on every step, or O(N²) steps;
    - GraphAMG on the assembled operator: an SPD design (Chebyshev smoother), and no MPI path;
    - defect correction around the same preconditioner;
    - VelocityMG's restricted-velocity coarse upwinding: not divergence-free on the coarse grid;
    - downstream or line smoothers: no order exists on a periodic torus, and they are not parallel
      on a GPU block or across MPI blocks;
    - the net (signed-sum) coarse flux: measured identical, but it drops the coarse exchange
      diffusion;
    - F- or W-cycles as the default: they halve the count but change the cycle. They are the
      recorded escalation (§13 Q18).

---

## 13. Risks and open questions (each with a default)

| # | Question | Needs | Default (work proceeds on it) | Revisit trigger / experiment |
|---|---|---|---|---|
| Q1 | 3-D absolute accuracy bounds | **fact** | the *(prov.)* bounds of §11 | first passing run; tighten to 2× measured |
| Q2 | Is the rediscretized ScalarMG good enough? | **fact** — **answered by A1 (2026-10-03)** | rediscretized faces + wall terms averaged at the fine probe distance | Q2′: if dilute suspensions (many bodies smaller than the coarse cells) show iterations growing with the number of bodies, design operator-dependent (BoxMG-type) prolongation; RAP is not the remedy (A1) |
| Q3 | Depth at scale without telescoping | **fact** | in-place depth only | if steady singular at np ≥ 8 needs > 1.5× the np = 1 iterations, add core stage telescoping (`chooseStageTarget`/`RedistributeTopology`, as CutcellMG) |
| Q4 | Contact-region accuracy for packed-bed Nu | **fact** | ladder of §3.3, one solid DOF per cell | G13 self-convergence < 1 or R2 > 1 % → design a contact model (Claassen 2024 line) |
| Q5 | API spelling of the opt-in | **user preference** | `add_scalar(..., cutcell=True)` — **DEFAULT-PENDING-USER** | rename before the first release that ships it (no alias needed while unreleased) |
| Q6 | The legacy `add_scalar` diffusivity is in internal units under an armed extent | **user preference** (scope) | leave legacy untouched; document — **DEFAULT-PENDING-USER** | fix at the next breaking release, or make cutcell the default (Q7) |
| Q7 | Should cut-cell become the default scalar discretization? | **user preference** | opt-in until G1–G13 pass and one release has shipped — **DEFAULT-PENDING-USER** | a recorded decision after that release |
| Q8 | First-order FOU at small-cell faces on slip walls / interfaces | **fact** | the dynamic split (§6.3) | if the G9 L1 error at the finest grid is dominated by the cut band (> 50 % of the total), or at VoF species time: implement WSRD with a G = 3 scalar block |
| Q9 | Memory and time per scalar | **fact** | separate level-0 surrogate storage, host Krylov scalars | G-perf red flag → derive the level-0 surrogate from the bands on the fly; device-resident Krylov scalars; fused two-field exchange |
| Q10 | Local wall gradients are first order in L∞ | **user preference** (scope) | integrated fluxes only are guaranteed 2nd order | if local Nu maps become a deliverable: the quadratic-normal post-processor (§9) |
| Q11 | Steady advection is first order (FOU) in v1 | **user preference** (scope) | v1 FOU; WO-11 optional | first porous-media dispersion study |
| Q12 | Default rtol 1e-10 | **user preference** | 1e-10, max-norm relative | — |
| Q13 | Which openness each collocated scheme constrains | **fact** | the projection's divergence-kernel openness via one predicate | G9b decides per scheme; a scheme that fails is refused for cut-cell scalars |
| Q14 | Two different conjugate materials sharing one solid DOF | **fact** | the larger-area facet's material | G13 conjugate pair |
| Q15 | Thin solids (< ~2h) leak through one fluid DOF | **fact** (resolution) | census + warning | — (a user resolution requirement) |
| Q16 | Moving geometry (DEM) and VoF species | **user preference** (scope) | hooks only (§6.6, §7.4) | their own packages |
| Q17 | Solid volumetric sources (reaction heat in particles) | **user preference** (scope) | fluid sources only in v1 | add `set_scalar_source(..., phase='solid')` when asked |
| Q18 | Is the V-cycle enough for steady advection in 3-D, at depth and at scale? (A2) | **fact** | the V-cycle of A2 | Any of: a G-adv bound exceeded; growth > 1.7× per doubling; a production closure at Pe_h ≤ 10 needing > 60 iterations. Then switch the steady advective path to the **F-cycle**: F(L) = pre-smooth, restrict, F(L+1) from 0, then one V(L+1) on the coarse defect, prolong, post-smooth; the bottom as now; mean removal on both coarse right-hand sides when singular. In 2-D at n = 256, Pe_h 10, it halves the count (26 → 13, 26 → 14). Level L is visited L + 1 times. This is a recorded decision, not a setter. |
| Q19 | Steady advection at Pe_h ≫ 10 | **user preference** (scope) | ungated and graceful (2-D: Pe_h 100 → ≤ 35 iterations at n = 128); the accuracy of FOU there is Q11's question | A study that needs Pe_h ≥ 30 routinely: first measure G-adv at 30 and 100. If the counts exceed 2× the Pe_h 10 row, a nonsymmetric AMG (AIR-type) is a new design. |
| Q20 | 3-D iteration and contraction bounds of G-adv (A2) | **fact** | the *(prov.)* numbers of §11 | first passing run; tighten to 2× measured |
| Q21 | Should transient mode use the advective surrogate too? (A2) | **fact** | No. Its FOU is local (small-cell faces only), and measured at 5–6 per step it equals the count at rest. | a transient row whose iterations grow with Pe_h (e.g. many small cells with a large implicit outflow) |

**What would make me revisit the design itself:**
- **(i)** G1 or G3 order < 1.7 in 3-D while the 2-D oracle stays at 2. Suspect the PL centroid or the
  probe support bound in 3-D; rerun `jc.py`-style ablations in a 3-D prototype before changing
  anything.
- **(ii)** G-iter failing with Galerkin RAP too. The lumped surrogate would then be inadequate in 3-D,
  and the surrogate itself would need revisiting (e.g. keeping the probe's same-cell weight
  off-lumped).
- **(iii)** G9b failing on a collocated scheme whose face field is documented as projected. The
  premise that a discretely divergence-free face flux exists would be false for that scheme.
