# The collocated cut-cell 2nd-order-drag problem — a first-principles statement

> **✅ RESOLVED (2026-07-17) — see §9.** The directional ghost-cell projection
> (`set_ghost_projection`, the staggered scheme's closures on the face-averaged field) plus a
> directional one-sided cell gradient for `-grad(P)` closes the problem: Z&H drag converges
> monotonically to the benchmark (−0.175% → −0.018% at N=32→128, vs mode-0's +1.00% → +0.30%,
> O(h)), with the a-priori tests of §5 finally built and identifying the defects by measurement.
> The sections below are kept as the (historically accurate) problem statement.

**Status:** problem statement (2026-07-05), written to *stop* the pattern of porting a
literature scheme, constructing a convincing rationale, implementing, and finding it fails. The
companion notes (`collocated_first_order_analysis.md`, `collocated_second_order_literature.md`,
`collocated_embed_port_plan.md`) contain that accumulated narrative; **this document deliberately
discards the narrative** and states only (i) what has been *measured*, (ii) what is *argued* (and may
be wrong), and (iii) the precise open question plus the cheap experiment that would settle it by
measurement instead of theory.

Read this one first. Treat every "why" in the other notes as a hypothesis, not a finding.

---

## 1. The problem, in one sentence

`SolverColocated` (cell-centred velocity + Almgren–Bell–Colella *approximate* projection) computes the
Stokes drag of a periodic sphere array to **first order** in the mesh; the staggered `Solver` gets
**second order** on the identical geometry. We want the collocated variant second order too, keeping the
approximate projection (no exact-MAC, no Rhie–Chow). We have not achieved it.

---

## 2. What is MEASURED (ground truth — trust these)

M1. **Staggered MAC → 2nd-order drag; collocated ABC → 1st-order drag**, both converging to the same
   Zick–Homsy limit. (`collocated_first_order_analysis.md` table; reproduced many times.)

M2. **On a flat, grid-aligned wall the collocated velocity field is essentially exact** (matches the
   analytic Poiseuille profile to ~0.04%, ~2nd order), *once* the momentum uses the true-normal embed
   wall gradient and solid-centred cut cells are left live (not masked to zero). Profile dump in
   `tests/study/embed_flatwall_guard.py` + the mode-6 velocity dump.

M3. **A fully-developed channel's projection is inactive** (the flow is divergence-free by construction:
   ∂u/∂x = 0, v = w = 0, so ∇·u = 0 pointwise and the pressure solve returns ~0). Therefore **M2 tests the
   momentum + no-slip only, and says nothing about the incompressibility constraint.** This is the single
   most important and most easily-forgotten fact.

M4. **The momentum wall drag is 2nd order a-priori, even for the curved field.** A manufactured Stokes
   field around a sphere, fed to the true-normal wall-gradient reconstruction, gives the wall viscous flux
   to O(h²) (`tests/study/fv_wallflux_apriori.py` variant C: +2.24%→+0.065%, order→2.2). So the momentum
   operator is not the first-order culprit on curved walls either.

M5. **The curved-wall drag error is a real error in the converged field, not a diagnostic artifact.**
   Measuring the drag from the cs-weighted superficial velocity instead of the raw cell mean changes it
   by <0.05% and does not remove the ~0.2–0.4% error (measured, this session).

M6. **Every cell-to-face flux reconstruction tried leaves a first-order residual, and they bracket the
   truth.** With the embed momentum + sliver fix held fixed, only the projection's face map varies:

   | face flux map (constraint) | cell correction | Z&H drag N=32/48/64 | behaviour |
   |---|---|---|---|
   | plain ½(Uᵢ+Uⱼ) | openness-weighted grad | −0.05 / +0.29 / +0.42% | **over-drags, grows** |
   | wall-aware open-centroid | openness-weighted grad | +2.15 / — / +1.82% | over-drags worse |
   | plain ½(Uᵢ+Uⱼ) | plain ½(g⁻+g⁺) | (pre-sliver-fix) 1st order | best axis-only |

   None converges at 2nd order. Plain under/over-shoots small; wall-aware overshoots large.

M7. **The Basilisk face-primary fix is numerically incompatible with this solver's time convention.**
   Adding the body force at the face (`uf = fs·(face_avg(u) + dt·a)`) plus Basilisk's explicit cell
   correction `u += dt·(a − ∇p)` diverges here (−75% drag at N=32), because this solver integrates
   momentum in the **divided convention** `(ρ/dt·cs − μ∇²) u = (ρ/dt·cs) uⁿ + f − ∇p` (well-conditioned at
   large dt), in which the implicit viscous term barely dissipates at large dt, so an *explicit* `dt·a`
   forcing is unbalanced. Basilisk uses the **undivided** convention `(I − dt·μ∇²) u = uⁿ`, where the
   explicit correction is consistent. (Measured this session; trial code reverted.)

---

## 3. What is ARGUED (hypotheses — may be the "fabulation"; do not trust)

A1. *"The defect is the open-area-mean vs face-centre mismatch at partially-open cut faces."* Plausible
   and consistent with M6, but **the fix it predicts (evaluate the face velocity at the open-area
   centroid) measured WORSE (M6, wall-aware)**, which the hypothesis does not explain. So either the
   hypothesis is incomplete, or the open-centroid reconstruction was implemented inaccurately, or it broke
   something else (see A2). **Unresolved.**

A2. *"The wall-aware constraint fails because it is not the exact adjoint of the pressure force, so the
   pressure does spurious work."* Has partial empirical support (an earlier unweighted transpose diverged;
   the o-weighted adjoint was stable). But whether a *correctly adjoint-paired* open-centroid scheme would
   be 2nd order was **never tested** — the stable version measured no better, but it may not have been the
   right pairing. **Unresolved.**

A3. *"You must keep a face-primary velocity uf (Basilisk/Trebotich); a purely cell-primary approximate
   projection cannot be 2nd order at curved walls."* This is a **literature inference**, not a proof, and
   not something we measured. M7 only shows the *specific* Basilisk port is convention-incompatible; it
   does not show cell-primary is impossible.

**The pattern to avoid:** each of A1–A3 came from a paper, was dressed as the explanation, and drove an
implementation that failed. The failures were real; the explanations were guesses.

---

## 4. The precise open question (stated structurally, not by analogy to any scheme)

A projection method needs three discrete operators on the cut-cell grid:

- **C** : cell velocity **u** → the per-cell incompressibility residual (a discrete `∮_∂(fluid cell) u·n dA`).
- **G** : cell pressure **p** → the cell-velocity pressure force (a discrete `∇p`).
- **A** : the pressure Poisson operator actually solved, `A p = C u*`, after which `u = u* − G p`.

On the **staggered** grid these satisfy, exactly and at 2nd order:
  (S1) C is the exact midpoint flux `Σ_a (o_{a+} u_{a+} − o_{a−} u_{a−})` with u living at the face centre
       = the open-area centroid to O(h²);
  (S2) **A = −C G exactly** (the discrete Laplacian is the exact composition), so after the solve
       `C u = C u* − C G p = C u* − A p ≡ 0` — the field is discretely divergence-free.

On the **collocated** grid, u is not on the faces, so C must *reconstruct* a face flux from cell values,
and the solved A is a *chosen* nice Poisson operator (`−div(o·grad)`), **not** the exact product C G.
That inexact composition is precisely why the projection is "approximate": after the solve,
`C u = C u* − C G p ≠ 0` in general — there is a leftover divergence.

**The question.** Find C, G (and the paired A) on the collocated cut-cell grid such that simultaneously:
  (Q1) C is a **2nd-order-accurate** discretisation of `∮ u·n dA` over the fluid cell boundary
       (i.e. it uses the *open-area-mean* normal velocity on each partial face, not the face-centre value);
  (Q2) the leftover divergence `C u* − C G p` (with `A p = C u*`) is **2nd-order small at cut cells**,
       not O(h) — equivalently, A approximates C G to 2nd order at cut cells;
  (Q3) G is consistent with C so the pressure does no spurious work (the projection is a genuine
       approximate L² projection);
  and the resulting **steady drag is 2nd order**.

Mode 0 satisfies (Q3) and is self-consistent but fails (Q1) (face-centre flux, O(h)). Every attempt to
fix (Q1) has either not actually delivered (Q1), or violated (Q2)/(Q3). **We do not know which of (Q1),
(Q2), (Q3) is the binding constraint** — that is the crux, and it has never been isolated.

---

## 5. The decisive experiment that is MISSING (and would replace theory with measurement)

The momentum side was de-risked by an **a-priori truncation test** (`fv_wallflux_apriori.py`): feed a
known analytic field to the discrete wall-flux operator and measure its order *before* any coupled solve.
That test is why we *know* (M4) the momentum is 2nd order. **There is no equivalent a-priori test for the
constraint**, and its absence is why Section 3 is all hypotheses. Build it:

Take a **manufactured divergence-free Stokes velocity field** `u_exact(x)` around one sphere (analytic, or
the Sampson/Zick series; must satisfy `∇·u = 0` and `u=0` on the sphere). Then, with **no solver run**:

- **T1 (does C see the true flux?).** Sample `u_exact` at the cell centres. Compute the discrete
  constraint residual `C u_exact` per cut cell for each candidate face-flux map. Since `∇·u_exact = 0` and
  `u_exact·n = 0` on the wall, the *exact* `∮ u·n dA = 0`, so `C u_exact` is pure truncation error.
  Measure its order in h. **This settles (Q1) directly:** plain ½/½ should be O(h) at partial faces; the
  open-area-centroid map should be O(h²). If the "better" map is *not* O(h²) here, the reconstruction
  itself is wrong (kills A1/A2 cleanly). If it *is* O(h²) here but the coupled drag still failed, the
  defect is (Q2)/(Q3), not (Q1) — a completely different fix.

- **T2 (does the approximate projection leave O(h) divergence?).** For the same field, form `A` and
  `G`, solve `A p = C u_exact`, correct `u = u_exact − G p`, and measure `C u` (the leftover divergence)
  and its order. This isolates (Q2) — the A-vs-CG inconsistency — from the reconstruction accuracy.

- **T3 (adjointness).** Check `⟨C u, p⟩ = ⟨u, Cᵀ p⟩` numerically for the chosen C and whether `G = Cᵀ`.
  A nonzero `⟨u, (G − Cᵀ) p⟩` on a divergence-free u is the spurious pressure work of A2, now *measured*
  rather than argued.

T1/T2/T3 are pure NumPy, no build, hours not sessions — the same discipline that made the momentum side
trustworthy. **Until they exist, any statement about *why* the drag is first order (including everything in
Section 3 and in the companion notes) is conjecture.** The correct next action is to write T1–T3, read off
which of (Q1)/(Q2)/(Q3) is violated and at what order, and only then choose a fix.

---

## 6. Constraints the solution must respect (so a fix isn't dead on arrival)

- **Divided time convention.** Momentum is `(ρ/dt·cs − μ∇²_embed) u = (ρ/dt·cs) uⁿ + f − ∇p`. Any
  face/flux treatment must be **implicit-compatible** with this (M7): an explicit `dt·a`-style correction
  is unstable at the large dt used to reach steady state. A fix that needs the undivided convention is a
  *solver-wide* change and must be scoped as such.
- **Incremental-rotational pressure is required for accuracy.** Non-incremental Chorin gives the wrong
  steady drag here (−40%, measured) — the classic splitting error does not vanish at steady state. So the
  pressure treatment must stay incremental/rotational.
- **Keep the approximate projection** (project the face field, reconcile the cell velocity) — exact-MAC or
  Rhie–Chow are explicitly out of scope (they change the solver's character).
- **Opt-in only.** Mode 0 and the staggered `Solver` are production and must stay byte-identical
  (regression suite). Any new scheme is a new `set_face_interp` mode, default off.

---

## 7. What is already built and correct (reuse, don't re-derive)

- True-normal embed wall gradient `embedDirichletGradient` + `embedViscousApply` (momentum; a-priori
  O(h²), M4) — `src/mac_approx_projection.hpp`.
- Openness-weighted cell pressure gradient `centerGradOpen` / `projectCorrectCenterOpen` (Basilisk
  `centered_gradient`) — a candidate G.
- Solid-cut-cell **sliver mask** fix (leave cs>0 cut cells live in the embed solve) — required for M2.
- `cs` (`buildCellFraction`), openness `o_f` (`buildOpenness`), open-face centroid distance
  (`buildFaceCentroidDist`) — the geometry T1–T3 need.
- Study harnesses: `collocated_zh_embed6.py` (mode 6 vs 0, robust steady protocol: dt=400 + warm-start,
  drift<5e-5 over 200 steps — the warm-*detector* protocol fires on false plateaus, do not use it),
  `fv_wallflux_apriori.py` (the momentum a-priori to mirror for the constraint).

## 8. Modes currently in the tree (all default off; mode 0 / Solver = production)

5 = embed momentum + old wall-aware projection (over-drags, plateaus);
6 = embed momentum + plain face map + openness-weighted correction (best; flat-wall exact; curved-wall
1st order, over-drags ~0.2–0.4%);
7 = mode 6 + wall-aware flux constraint (over-drags worse).
Mode 8 (Basilisk face-primary) was tried and reverted (M7).

**Bottom line for whoever takes this next:** the momentum is solved (M2, M4). The unsolved object is the
**incompressibility constraint operator C (and its pair G, A) at partially-open cut cells** (Section 4).
Do **not** port another scheme first. Write the a-priori constraint-truncation tests T1–T3 (Section 5),
find out by *measurement* which of (Q1)/(Q2)/(Q3) is actually violated and at what order, and let that —
not a paper — dictate the fix. Respect Section 6 or the fix will not run.

---

## 9. RESOLUTION (2026-07-17): the directional ghost-cell projection

The Section-5 discipline was followed (`tests/study/ghost_collocated_apriori.py`, extending the
staggered `ghost_projection_apriori.py`), and the measurements changed the diagnosis:

**What the a-priori tests measured (all at 8/8 gates, N up to 128):**

- **[C2] The dominant defect was never on the list: the `-grad(P)` operator itself.** The mode-0
  predictor is the central difference `½(P_{i+1} − P_{i−1})`, and at a cut cell it reads P at the
  **solid-centered neighbour, whose row is decoupled and holds 0** (RB-GS skips zero-diagonal rows;
  mean-removal is gated on the active diagonal; `P += (ρ/dt)φ` keeps it 0). Reading that 0 is a
  **gauge-dependent O(1) error in the gradient — O(1/h) in physical units** (measured: 11 → 92
  physical-units error at N=16→128, order −1.0). The o-weighted kernels read the same 0 through
  partially-open faces (`open_real`, same order −1.0); `projectCorrectCenter` is O(1) (its ½-weight
  one-sided form); `centerGradOpen` with *binary* openness is the best prior art at O(h). A
  **directional one-sided 2nd-order gradient** — central where both axis-neighbours are
  fluid-centered, `(−3P_i + 4P_{i+1} − P_{i+2})/2` toward the fluid otherwise, never reading a
  decoupled cell — is O(h²) (order 1.97–2.00) and exactly gauge-invariant (`gpCenterGrad`).
  This error is forced on an O(h)-measure cut-cell layer → elliptic damping → O(h) drag: it alone
  reproduces the observed first order, and it survived **every** prior constraint-side fix, which
  is why modes 1–8 all stayed O(h).
- **[C1] (= T1) The mode-0 constraint is worse than the O(h) the doc conjectured: it is O(1) at
  cut cells** (openness divergence of the face-averaged exact solenoidal field: 0.44 → 0.53,
  order ≈ 0 at N=16→128). The ghost-closed divergence of the same face-averaged field is the
  staggered structure: O(h) localized at the IB (order → 0.99), O(h²) in the bulk (fixed-shell
  order → 1.99).
- **[C3] (= T2) The full manufactured chain** (perturb the cell field through the gradient
  operator, face-average, ghost-divergence, singular solve, face + cell correction) delivers the
  corrected **cell velocity at order 2.2–2.6** (L2 2.6–2.8) with the diagnostic ≡ residual
  identity to round-off. The cell-correction ladder inside the chain reproduces the ranking
  (ghost ≥ open ≫ pcc).

**The scheme** (`set_ghost_projection(True, matrix_order, rhs_order)` on `SolverColocated`,
`src/ghost_projection.hpp`): the staggered directional ghost projection carries over **verbatim**
— the collocated projection already projects a MAC face field (`uf = ½(U_i+U_{i−1})`), the face
correction `uf −= ∇φ` is the identical substitution, so the φ matrix, binary-openness MG
surrogate, BiCGStab driver, and fragmentation guard are shared unchanged. New pieces: the RHS/
diagnostic divergence closes the **face-averaged** field (`gpDivergDelta` on `uf_`), and both the
incremental `-grad(Pⁿ)` predictor and the cell correction use `gpCenterGrad` (reading the
projection's fragmentation-guarded sdf). Only `face_interp` mode 0 composes with it. At the
steady fixed point φ→0, so the drag is set by (momentum + gpCenterGrad(P)) × (ghost constraint)
— the (1,2) mixed form's decoupling, which is why the matrix could stay the staggered one.

**Measured Z&H drag** (φ=0.125 SC sphere, K_ref=4.2920, warm-detector + tail protocol of
`tests/study/collocated_zh_ghostproj.py`; ghost = (1,2) mixed):

| N | mode 0 | ghost (1,2) | iters 0 / g |
|---:|---:|---:|---:|
| 32 | +1.004% | **−0.175%** | 10 / 6 |
| 48 | +0.685% | **−0.084%** | 9 / 6 |
| 64 | +0.598% | **−0.056%** | 10 / 7 |
| 96 | +0.397% | **−0.029%** | 11 / 7 |
| 128 | +0.299% | **−0.018%** | 10 / 7 |

Monotone from below to the same limit (the staggered ghost's signature), 16× more accurate than
mode 0 at N=128 and **below the Z&H table's own ~0.05% precision from N=96** (tail Richardson
orders are not meaningful there — measured pairwise 1.8/1.45/1.6/1.6 over the resolved range,
same caveat as the staggered study). BiCGStab holds flat 6–7 iterations vs PCG's 9–11. The
flat-wall offset-slab Poiseuille stays pointwise exact (4e-6, ghost divergence 0). Both
production defaults (staggered + collocated mode 0) are byte-identical (regression suite +0.00%).

**Answering Section 4 by measurement:** the binding violations were (Q1) — but at O(1), in both
C *and* G, not the conjectured O(h) in C alone — and the constraint that fixes them is not
adjointness (Q3): the ghost pair (C, G) is deliberately **non-adjoint** (a nonsymmetric 13/7-point
C with a one-sided G), yet stable under the MG-surrogate BiCGStab and convergent. A1's open-area
flux hypothesis is moot — the ghost C is a *point* divergence, no apertures at all; the sub-cell
throat caveat of the staggered study (RCP tight throats) applies to the collocated variant
identically. Constraints of Section 6 respected: divided convention untouched (the ghost pieces
are all in the projection/predictor), incremental-rotational pressure kept, still an approximate
projection of the averaged face field, opt-in with byte-identical defaults.

### 9.1 Tight-throat (RCP) behaviour and the mode-9 cutcell-ghost hybrid (2026-07-18)

Measured on the peclet-examples random-close-packing geometry (dem `pack_bed`, N=180, φ=0.630,
cached `tests/study/rcp_pack_seed3.npz`; harness `tests/study/rcp_permeability_ab.py`;
permeability k vs the **staggered cutcell** reference on the same sampled SDF):

| Ng | stag ghost | col mode 0 | col ghost | col mode 9 |
|---:|---:|---:|---:|---:|
| 32 | +27.2% | −19.6% | −10.1% | −13.0% |
| 44 | +18.3% | −13.3% | −4.3% | −8.6% |
| 56 | +14.2% | +13.6%* | −1.2% | −6.2% |

(*) mode 0 at Ng=56 is genuinely erratic — the sign flip is real (re-verified at tol 1e-8) and
it needed 7725 steps to settle vs ~650 for its neighbours: the O(1/h) gradient defect both
scatters k and cripples the transient on under-resolved throats.

- The **collocated ghost inherits the staggered ghost's throat defect** (binary graph fragments
  identically — 99 pockets at Ng=56 — and the point closures over-carry the throats, sitting
  ~5–7% above the aperture family), but its raw numbers look best here only because that
  inflation *cancels* against the collocated under-shoot — two wrongs, not a scheme to trust on
  tight throats.
- **Mode 9** (`set_face_interp(9)`) is the throat-safe collocated cutcell: mode-0's aperture
  projection verbatim (throttling, symmetric MG-PCG at 15–31 iters, no guard needed) with only
  the two measured-O(1) operators — the `-grad(P)` predictor and the cell correction — replaced
  by `gpCenterGrad`. It converges monotonically toward the staggered reference and is stable
  where mode 0 is erratic. On Z&H it holds a −0.04..−0.10% error band at N=32..128 — **not**
  clean 2nd order (the aperture constraint's truncation on the *pinned* ½/½ face average floors
  it; the staggered scheme escapes only because its face values are free DOFs) but 7–20× below
  mode 0 and comparable to the ghost at practical N.
- **Mode 10** (mode 9 + the open-centroid wall-aware quadrature) is a dead ablation: O(h) with a
  worse constant on Z&H and divergent on RCP slivers — the mode-3a non-telescoping row-sum
  runaway, *not* cured by the telescoping force. This finally closes A1 in the coupled setting:
  the constraint quadrature is not the lever, on either geometry.

**Practical guidance (mirrors the staggered situation exactly):** ghost projection for
resolved/smooth immersed geometry (2nd order, fewer iterations); aperture cutcell — mode 9
rather than mode 0 — for under-resolved tight-throat porous media.
