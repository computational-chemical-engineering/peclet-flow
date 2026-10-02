# L1 — Cut-cell / embedded-boundary FV for diffusion (Dirichlet/Neumann/Robin/conjugate) + small-cell advection

Literature digest, 2026-10-02. For: peclet scalar transport with immersed solids (uniform MAC grid + block-octree AMR,
SDF at cell centres, apertures from marching squares, matrix-free 7-point / RB-GS / GMG / MG-PCG / BiCGStab, 1–2 cell halos).

Verification legend: **[read]** = method section read in full text (PDF/source); **[abs]** = abstract/metadata only;
**[src]** = read in source code; **UNVERIFIED** = citation or claim not checked against a primary source.
Local copies of the full texts read are next to this file (`*.txt`, `amrex/`).

---

## 1. The Colella/LBNL EB family (Johansen–Colella → McCorquodale → Schwartz → Crockett → Devendran)

### 1.1 Johansen & Colella 1998 — the base scheme (2-D)
H. Johansen, P. Colella, "A Cartesian grid embedded boundary method for Poisson's equation on irregular domains",
J. Comput. Phys. 147(1):60–85 (1998). DOI 10.1006/jcph.1998.5965 (OSTI https://www.osti.gov/biblio/320965). **[abs; method as restated in 1.2/1.3, read]**
- Unknowns at **Cartesian cell centres** (even if the centre is covered: values there are treated as a smooth extension).
- Cut-face flux = centred difference **linearly interpolated to the face centroid** using the neighbouring full face.
- Dirichlet wall flux: ∂φ/∂n from **quadratic fit along the normal through φ_B and two interpolated values φ_I1, φ_I2**
  (each interpolated in a grid line/plane crossed by the normal).
- Neumann: wall flux given directly.
- Truncation O(h²) in regular cells, **O(h)/O(1/κ)-type in cut cells**, yet solution O(h²) — reconciled by a modified-equation argument.
- Matrix **non-symmetric**; amenable to geometric multigrid; AMR in the paper.

### 1.2 McCorquodale, Colella, Johansen 2001 — heat equation
P. McCorquodale, P. Colella, H. Johansen, "A Cartesian grid embedded boundary method for the heat equation on irregular domains",
J. Comput. Phys. 173(2):620–635 (2001). DOI 10.1006/jcph.2001.6900. Preprint read: https://www.osti.gov/servlets/purl/835804 **[read]**
- Same spatial operator (eq. 6: F_{i+½,j} = η(U_{i+1,j}−U_{ij})/h + (1−η)(U_{i+1,j±1}−U_{i,j±1})/h, η from face-centroid offset).
- **Crank–Nicolson is unsafe**: for moving Dirichlet boundaries "oscillatory behavior … attributed to … neutral stability of
  Crank–Nicolson at high wave numbers and the presence of eigenvalues of L_H with nontrivial imaginary parts" (i.e. the EB
  operator is non-normal/non-symmetric). Uses L0-stable **TGA** (Twizell–Gumel–Arigu) Runge–Kutta. Backward Euler would also be L0-stable.
- Fixed boundary: CN 2nd order in L1 for Neumann but **zeroth order in max norm and diverging for Dirichlet** on moving boundaries; TGA 2nd order.

### 1.3 Schwartz, Barad, Colella, Ligocki 2006 — 3-D, the reference description
P. Schwartz, M. Barad, P. Colella, T. Ligocki, "A Cartesian grid embedded boundary method for the heat equation and Poisson's
equation in three dimensions", J. Comput. Phys. 211(2):531–550 (2006). DOI 10.1016/j.jcp.2005.06.010.
Preprint read: https://www.osti.gov/servlets/purl/878684 **[read]** (page range from memory: UNVERIFIED; volume/issue/DOI verified)
- Operator: (Δ^hφ)_i = (1/(κ_i h)) [Σ ±α_{face} F_face + α_B F_B]; **solved in κ-weighted form κ_iΔ^hφ = κ_iρ_i** "to avoid
  problems arising from arbitrarily small values of κ_i in the denominator". (This is exactly the "κ in storage/source" the
  peclet prototype found necessary.)
- Cut-face flux: **bilinear interpolation of centred differences to the face centroid** (4 parallel faces). Quote: "we also
  tried using simple linear interpolation based on three of the faces … such a method is unstable for some configurations
  of adjacent small control volumes, in the sense that point Jacobi fails to converge for any value of the relaxation
  parameter." Fallback: piecewise-constant (no centroid correction) when the 4 faces are unavailable.
- Dirichlet wall gradient, primary: eq. (8) ∂φ/∂n ≈ [d₂/d₁(φ_B−φ_I1) − d₁/d₂(φ_B−φ_I2)]/(d₂−d₁), φ_I1, φ_I2 by **biquadratic
  interpolation in two grid planes** crossed by the normal (18 cells). Fallback (under-resolved): **least-squares gradient**
  from ≤7 cell centres in the normal octant, O(h) flux.
- **Error hierarchy (verbatim logic):** regular cells τ = O(h²); cut cell with 2nd-order wall flux τ = O(h/κ); with the
  1st-order least-squares flux τ = O(1/κ). "Both methods … lead to a second order solution error. This is because, for
  Dirichlet boundary conditions, solution error is two orders of accuracy more than the truncation error on the boundary."
  But "it is necessary to use (8) to obtain second-order accurate values for ∇φ at the boundary"; with the LS stencil
  ∇φ error is O(h).
- Condition number "bounded independent of κ, and comparable to that of the uniform grid algorithm"; geometric MG V/W cycles.
  Heat equation via TGA. AMR-compatible (Chombo).

### 1.4 Crockett, Colella, Graves 2011 — conjugate / discontinuous coefficients (two-material EB)
R.K. Crockett, P. Colella, D.T. Graves, "A Cartesian grid embedded boundary method for solving the Poisson and heat equations
with discontinuous coefficients in three dimensions", J. Comput. Phys. 230(7):2451–2469 (2011). DOI 10.1016/j.jcp.2010.12.017.
Preprint read: https://www.osti.gov/servlets/purl/980773 **[read]**
- Each material has its own cut-cell FV (Johansen–Colella). Interface state φ_B and flux unknown; each side's normal
  derivative is written **linear in the unknown interface value**: ∂φ^{B,p}/∂n = w_B^p φ_B^p + Σ w_i φ_i + O(h^q)
  (q = 2: J–C quadratic-along-normal stencil; q = 1: least squares when under-resolved).
- Jump conditions (value jump + β⁺∂φ⁺/∂n − β⁻∂φ⁻/∂n = g_N) → a **local 2×2 solve per interface face** for the two normal
  derivatives; boundary fluxes "recalculated after every step in relaxing" (i.e. inside Gauss–Seidel).
- 2nd-order overall; stable/efficient with geometric MG for coefficient contrasts up to 10⁶; AMR tested.
- **Relevance:** the same "gradient linear in φ_B, eliminate φ_B locally" closure gives a Robin BC (a 1×1 elimination of
  αφ_B + β∂φ/∂n = g). That Robin specialisation is my inference, not a paper's claim.

### 1.5 Higher order (for completeness)
- D. Devendran, D. Graves, H. Johansen, T. Ligocki, "A fourth-order Cartesian grid embedded boundary method for Poisson's
  equation", Comm. Appl. Math. Comput. Sci. 12(1):51– (2017). DOI 10.2140/camcos.2017.12.51 **[abs]**. Weighted
  least-squares polynomial stencils for face fluxes; stability checked through eigenvalues; non-symmetric, wide stencils.
- Y. Qian, W. Li, Y. Tan, Q. Zhang, "A fourth-order, multigrid cut-cell method for solving Poisson's equation in
  three-dimensional irregular domains", arXiv:2410.05865 (2024) **[abs]**. PLG stencil search + modified MG.
- Overton-Katz et al., "A high order Cartesian grid, finite volume method for elliptic interface problems", arXiv:2302.09161
  (seen in search; authors/journal UNVERIFIED).
Not a fit for 1–2-cell halos and 7-point matrix-free MG.

---

## 2. The Gibou/Min family (symmetric, level-set, quad/octree)

### 2.1 Gibou, Fedkiw, Cheng, Kang 2002 — symmetric Dirichlet ghost
F. Gibou, R.P. Fedkiw, L.-T. Cheng, M. Kang, "A second-order-accurate symmetric discretization of the Poisson equation on
irregular domains", J. Comput. Phys. 176:205–227 (2002). https://www.sciencedirect.com/science/article/pii/S0021999101969773 **[abs]**
- Node/cell-centred FD; ghost across the interface by **linear extrapolation** → only the diagonal changes, matrix SPD.
- Analysis: G. Yoon, C. Min, "Analyses on the finite difference method by Gibou et al. for Poisson equation", J. Comput. Phys.
  280:184–194 (2015), https://www.sciencedirect.com/science/article/abs/pii/S0021999114006457 (authors verified in JCP 2015 TOC)
  **[abs]**: proves **2nd-order solution**; **gradients 1st order**; unpreconditioned condition number O(1/(h·h_min))
  (h_min = smallest wall distance), Jacobi/SGS/ILU-preconditioned O(h⁻²), MILU O(h⁻¹).
- Contrast: G. Yoon, C. Min, "Convergence analysis of the standard central finite difference method for Poisson equation",
  J. Sci. Comput. 67:602–617 (2016), DOI 10.1007/s10915-015-0096-2 **[abs]**: for the *standard* (Shortley–Weller-type,
  non-symmetric, quadratic-consistent) scheme the numerical **gradient is proven 2nd order** (super-convergence) in arbitrary smooth domains.
- Y.-T. Ng, H. Chen, C. Min, F. Gibou, "Guidelines for Poisson solvers on irregular domains with Dirichlet boundary conditions
  using the ghost fluid method", J. Sci. Comput. 41:300–320 (2009), DOI 10.1007/s10915-009-9299-8 (citation from Gibou's
  publication list; content UNVERIFIED — paywalled).

### 2.2 Ng, Min, Gibou 2009 — aperture-weighted Neumann FV (SPD)
Y.-T. Ng, C. Min, F. Gibou, "An efficient fluid–solid coupling algorithm for single-phase flows", J. Comput. Phys.
228(23):8807–8829 (2009). DOI 10.1016/j.jcp.2009.08.032. Preprint read: http://math.ewha.ac.kr/~chohong/publications/article_12_copy.pdf **[read]**
- Neumann Poisson: Σ L_face (p_nb − p)/Δx with **face length fractions L (apertures)**, no face-centroid correction; SPD;
  "second-order accurate in both the L1 and L∞ norms".
- **Pitfall evidence:** replacing apertures by face *volume* fractions ("masses", Batty–Bertails–Bridson 2007) is "only first
  order accurate in the average L1-norm and is not convergent in the L∞-norm", error maximal next to the solid.

### 2.3 Papac, Gibou, Ratsch 2010 — symmetric Robin on uniform grids
J. Papac, F. Gibou, C. Ratsch, "Efficient symmetric discretization for the Poisson, heat and Stefan-type problems with Robin
boundary conditions", J. Comput. Phys. 229:875–889 (2010). DOI 10.1016/j.jcp.2009.10.017.
Full text read: https://www.math.ucla.edu/~cratsch/Publications/Publications2010/sdarticle.pdf **[read]**
- Robin ∂q/∂n + a q = f. Integrate over C_ij ∩ Ω: storage ∫q dA = q_ij·**Area(C∩Ω)** (κ in storage — yes), source likewise.
- Cut-face fluxes: plain 2-point differences × **face length in Ω** (aperture; no centroid correction).
- Wall term: ∫_{C∩Γ}(f − a q) dl = **−a q_ij L_Γ** + ∫f dl — i.e. **wall value := cell-centre value, no distance/resistance
  correction**; goes on the diagonal only → **SPD** ("terms that arise from the boundary … appear only in the diagonal
  coefficient and in the right hand side"). Geometric quantities by Min–Gibou simplex integration (2nd order).
- Reported: **2nd order in L1 and L∞** for Poisson and heat (fit slopes 2.04/1.92, 1.74/1.88 on a flower/star domain);
  Stefan 1st order. CG + incomplete Cholesky.

### 2.4 Papac, Helgadottir, Ratsch, Gibou 2013 — Robin on quad/octrees
J. Papac, A. Helgadottir, C. Ratsch, F. Gibou, "A level set approach for diffusion and Stefan-type problems with Robin boundary
conditions on quadtree/octree adaptive Cartesian grids", J. Comput. Phys. 233:241–261 (2013). DOI 10.1016/j.jcp.2012.08.038.
Full text read: https://www.math.ucla.edu/~cratsch/Publications/Publications2013/Papac_JCP233_241_2013.pdf **[read, skimmed]**
- Hybrid: Min–Gibou 2007 non-graded octree FD in the bulk, Papac-2010 FV in a **uniform band of finest cells around the
  interface** (no T-junctions at the interface). T-junction ghost values make the system **non-symmetric** (BiCGSTAB + SGS);
  M-matrix. 2nd-order solution for diffusion; Stefan 1st order.

### 2.5 Bochkov & Gibou 2019; Arias, Bochkov & Gibou 2018 — Robin, solution vs gradient order
- D. Bochkov, F. Gibou, "Solving Poisson-type equations with Robin boundary conditions on piecewise smooth interfaces",
  J. Comput. Phys. 376:1156–1198 (2019). https://www.sciencedirect.com/science/article/pii/S0021999118306831 **[abs]**:
  two FV schemes — **(i) symmetric: 2nd-order solution, 1st-order gradients (L∞); (ii) non-symmetric: 2nd-order solution
  and 2nd-order gradients**; plus hierarchical geometric reconstruction for piecewise-smooth (kinked) boundaries.
- V. Arias, D. Bochkov, F. Gibou, "Poisson equations in irregular domains with Robin boundary conditions — Solver with
  second-order accurate gradients", J. Comput. Phys. 365:1–6 (2018). DOI 10.1016/j.jcp.2018.03.022 (from search summary;
  method content UNVERIFIED — paywalled).
- What makes gradients 2nd order, stated by the same group (Bochkov & Gibou 2020, arXiv:1905.08718 **[read]**, §2 remarks):
  their compact FV has "truncation error … O(h²) for grid points away from the immersed interface and O(1) for cells crossed
  by the interface" (after scaling by cell volume) → "second-order accurate numerical solutions with first-order accurate
  gradients". To reach O(h) truncation and 2nd-order gradients: "1) Estimate fluxes between cells at the **centroids of cell
  faces** using linear interpolation as done, for example, in [Johansen–Colella; Bochkov–Gibou 2019]; 2) Retain the
  quadratic term in Taylor expansion; 3) Use a quadratic interpolant … to approximate ∂_n u at projection points."

### 2.6 Interfaces (conjugate) in this family
- D. Bochkov, F. Gibou, "Solving elliptic interface problems with jump conditions on Cartesian grids", J. Comput. Phys.
  407:109269 (2020), DOI 10.1016/j.jcp.2020.109269, arXiv:1905.08718 **[read, partly]**: compact FV, Taylor expansion along
  the normal + one-sided linear least squares relate the two sides; 2nd-order solution, 1st-order gradients; **non-symmetric
  when μ⁺≠μ⁻**; stencil = 5-point + diagonal neighbours; condition number bounded independent of μ⁺/μ⁻.
- A. Guittet, M. Lepilliez, S. Tanguy, F. Gibou, "Solving elliptic problems with discontinuities on irregular domains – the
  Voronoi Interface Method", J. Comput. Phys. 298:747–765 (2015). https://www.sciencedirect.com/science/article/pii/S0021999115004234
  **[abs + as summarised in Bochkov 2020]**: SPD, 2nd-order solution, 1st-order fluxes; needs a local Voronoi mesh.

---

## 3. Small cells for explicit advection

| Method | Ref | Order at cut cells | Conservative | Notes |
|---|---|---|---|---|
| Flux redistribution | Chern & Colella 1987 LLNL report (UNVERIFIED); R.B. Pember, J.B. Bell, P. Colella, W.Y. Crutchfield, M.L. Welcome, J. Comput. Phys. 120(2):278–304 (1995), DOI 10.1006/jcph.1995.1165 **[abs]**; P. Colella, D.T. Graves, B.J. Keen, D. Modiano, J. Comput. Phys. 211(1):347–366 (2006) **[abs; DOI not seen]** | **1st** (Berger–Giuliani: "only first order accurate at the cut cells") | yes | cut cell keeps κ·(update), rest (1−κ)δM given to neighbours, volume-weighted; reach 1 cell; trivially GPU-parallel (atomic or gather) |
| Cell merging / linking | refs in Berger–Giuliani 2021 | 2nd possible | yes | non-overlapping merges are hard to make robust in 3-D ("not aware of any production codes that implement this in a fully general, robust manner") |
| h-box | M.J. Berger, C. Helzel, R.J. LeVeque, SIAM J. Numer. Anal. 41(3):893–918 (2003), https://faculty.washington.edu/rjl/pubs/hbox/40539.pdf **[abs]** | 2nd | yes | "not been extended to three dimensions due to its complexity" (Berger–Giuliani) |
| State redistribution (SRD) | M. Berger, A. Giuliani, "A state redistribution algorithm for finite volume schemes on cut cell meshes", J. Comput. Phys. 428:109820 (2021), DOI 10.1016/j.jcp.2020.109820, arXiv:2005.05734 **[read]** | 2nd (preserves linears; does not reduce base-scheme accuracy) | yes | post-process on the *state*: each cell gets a (possibly overlapping) merging neighbourhood with Σ V ≥ ½ h^d (normal or centred within the 3×3(×3) tile); neighbourhood average weighted by inverse overlap count; LS gradient on neighbourhood + Barth–Jespersen limiter; shuts off for cells ≥ target |
| Weighted SRD (WSRD) | A. Giuliani, A.S. Almgren, J.B. Bell, M.J. Berger, M.T. Henry de Frahan, D. Rangarajan, J. Comput. Phys. 464:111305 (2022), DOI 10.1016/j.jcp.2022.111305 **[abs]** | 2nd | yes | 3-D, less dissipative weighting; used for advective **and diffusive** updates in AMReX codes, MPI + hybrid parallel |
| Provably stable WSRD | M. Berger, A. Giuliani, arXiv:2308.16332 (2023/24) **[abs]** | 2nd | yes | monotone, TVD, GKS-stable in many cases; pre-merging step |
| WSRD + AMR | I. Barrio Sanchez, A.S. Almgren, J.B. Bell, M.T. Henry de Frahan, W. Zhang, arXiv:2309.06372 (2023) **[abs]** | 2nd | yes (across levels) | "re-redistribution" when cut cells sit at/near coarse–fine interfaces |
| WSRD on staggered mesh | S. Kang, A.S. Almgren, et al., arXiv:2604.11959 (2026), ERF **[abs]** | — | yes | WSRD extended to face-staggered velocities, separate EB data per staggering |
| Thresholds | J.E. Karell, "Spectral analysis and redistribution thresholds for cut-cell finite-volume methods", arXiv:2607.28808 (2026) **[abs]** | — | — | small cells give O(α⁻¹) coefficients; blending threshold s₀ = (L−2)/(L−1) suffices, full merging unnecessary |

Halo/stencil reach: AMReX's `StateRedistribute` kernels work on boxes grown by up to 3 cells
(`amrex::grow(bx,3)`, Src/EB/AMReX_EB_StateRedistribute.cpp **[src]**; default `target_volfrac = 0.5`), on top of the
halo the MUSCL flux needs. Flux redistribution needs only a 1-cell reach.
Basilisk `embed.h` (`update_tracer`) **[src via basilisk.fr]**: fluxes that would "overflow" a small cell beyond
Δt_max = c_s Δ/max(f_i|u_i|) are stored and redistributed to neighbours weighted by c_s (a flux-redistribution variant).

---

## 4. Production codes — what they actually do on the EB

**AMReX `MLEBABecLap`** (cell-centred EB Helmholtz; GPU via AMReX kernels; MLMG on AMR) — docs
https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html and source Src/LinearSolvers/MLMG/AMReX_MLEBABecLap_3D_K.H **[src]**:
- EB BCs: **homogeneous Neumann (default), homogeneous Dirichlet, inhomogeneous Dirichlet** (`setEBDirichlet(lev, phi_on_eb, beta)`).
  **No Robin on the EB**; Robin (a φ + b ∂φ/∂n = f) exists only on *domain* faces (`LinOpBCType::Robin`). There is no
  inhomogeneous-Neumann EB setter (prescribed flux enters through the RHS).
- Cut-face flux (cell-centred φ): **bilinear interpolation to the face centroid** from 4 parallel faces (fracy, fracz =
  |face-centroid offset|), only where the transverse cells exist — the Schwartz 2006 scheme. Operator divides by κ (1/kappa).
- **Dirichlet wall gradient: two-point, along the normal**: point g at distance dg along −n from the *boundary centroid*,
  `dx_eb = max(0.3, (κ²−0.25)/(2κ))` (in cell units, `get_dx_eb`, AMReX_MLLinOp_K.H), φ_g by **trilinear interpolation**
  from the 8 cells in the normal octant; ∂φ/∂n = (φ_B − φ_g)/dg. This is O(h) flux → per Schwartz 2nd-order solution,
  1st-order wall gradient. Optional `setPhiOnCentroid()` path uses least-squares gradients (`grad_eb_of_phi_on_centroids_extdir`).
- Smoother: the Dirichlet term enters both the diagonal (`gamma`) and the off-diagonals → **non-symmetric** operator;
  geometric MG with EB coarsened by the factory.
**incflo** (AMReX-Fluids): tracers/temperature use `MLEBABecLap`; EB Dirichlet via `tracer_eb`/`temperature_eb`, default
homogeneous Neumann; redistribution `StateRedist` (issue https://github.com/AMReX-Fluids/incflo/issues/241, 2026-09-16 **[read]**,
which also documents that an EB Dirichlet set on a shared op cannot be reverted to Neumann).
**MFIX-Exa** (https://mfix.netl.doe.gov/doc/mfix-exa/guide/latest/user_guide/inputs/boundary_conditions.html **[read]**):
EB temperature = adiabatic (default) | constant (Dirichlet) | "CHT-1D" (1-D conjugate wall model). No Robin/flux EB option documented.
**PeleLMeX**: isothermal EB (Dirichlet) via MLEBABecLap; EB conduction tracked by integrating the EB Fourier flux
(from search snippets of PeleLMeX PR #707 / tutorials — **UNVERIFIED** in docs; tutorial page 404'd).
**ERF**: EB on staggered mesh with WSRD (arXiv:2604.11959, above).
**Chombo EB** (EBAMRPoissonOp; `DirichletPoissonEBBC` with `setOrder` → first/second-order stencils; `NeumannPoissonEBBC`)
— class reference https://davis.lbl.gov/Manuals/CHOMBO-RELEASE-3.1/DirichletPoissonEBBC_8H_source.html **[src header]**;
the algorithms are those of §1.3–1.4. No EB Robin class found.
**Basilisk `embed.h`** (http://basilisk.fr/src/embed.h **[src]**): face gradients by J–C 1998 face-centroid interpolation
(eq. 16, "figure 3"); Dirichlet wall gradient from 1 or 2 points along the normal with quadratic interpolation in grid
lines (code comments call the 2-point version "third-order"; 5×5 stencil in 2-D), falling back to 1 point; Neumann = given
flux; **no Robin**. Papers: A.R. Ghigo, S. Popinet, A. Wachs, "A conservative finite volume cut-cell method on an adaptive
Cartesian tree grid for moving rigid bodies in incompressible flows", HAL hal-03948786 (2023; journal UNVERIFIED);
Limare, Popinet, Josserand, Xue, Ghigo, "A hybrid level-set/embedded boundary method applied to solidification-melt
problems", J. Comput. Phys. 474:111829 (2023) (**[abs via search]**, DOI not seen).
**Environmental scalar transport with EB on AMR**: M. Barad, P. Colella, S.G. Schladow, "An adaptive cut-cell method for
environmental fluid mechanics", Int. J. Numer. Meth. Fluids (2009), DOI 10.1002/fld.1893 (**[abs via search]**).

---

## 5. Accuracy pitfalls — what the literature says (and where it is my inference)

1. **κ-weighting is the standard fix for small-cell conditioning and for consistency.** Schwartz 2006 solves κΔ^hφ = κρ;
   Papac 2010 puts Area(C∩Ω) in storage and source. With unit storage the cut-cell balance is inconsistent by O(1)
   relative → matches the peclet prototype (Neumann 1st order without κ).
2. **Error-gain rule (Schwartz 2006, J–C 1998 modified-equation analysis):** Dirichlet solution error is *two* orders
   above the boundary truncation error, so an O(1/κ) cut-cell truncation (O(h) wall flux) still gives O(h²) solutions;
   **but O(h²) wall gradients require the O(h²) normal-line quadratic stencil.** For Neumann/Robin the gain is
   (inference, consistent with Bochkov–Gibou 2020's statement "O(1) truncation in cut cells → 2nd-order solution,
   1st-order gradients") one order less — hence the need for face-centroid fluxes there to raise gradient order.
3. **Why a two-point wall flux to the cell centre/centroid is 1st order (inference):** (φ_B − φ_c)/|x_B − x_c| approximates
   the derivative along x_B − x_c, not along n; the tangential part is an **O(1) flux error** whenever x_c is not on the
   normal through x_B (generic), i.e. inconsistent, not merely low order → O(1/(κh)) truncation → O(h) solution even with
   the two-order Dirichlet gain. Every 2nd-order scheme found evaluates the gradient **on the normal line through the
   boundary centroid** (J–C/Schwartz quadratic, AMReX two-point at dg ≥ 0.3h with trilinear φ_g, Basilisk) or by a
   consistent least-squares fit (Schwartz fallback, Crockett). The same mechanism explains the 1st-order Robin with a
   series resistance d/D if d is the centre-to-wall distance along a non-normal line (inference — compare with Papac 2010,
   which drops the resistance entirely, q_wall := q_cell, and still reports 2nd-order solutions).
4. **Aperture-only cut faces (no face-centroid correction):** sufficient for 2nd-order *solutions* with Neumann (Ng 2009,
   Papac 2010, Bochkov 2019 symmetric scheme, peclet prototype) and Robin (Papac); gives **1st-order gradients**.
   Face-centroid (bi)linear interpolation is what the non-symmetric 2nd-order-gradient schemes add (Bochkov–Gibou 2020
   remark; Schwartz 2006; AMReX). It breaks symmetry. Using face *volume* fractions instead of apertures is worse:
   1st order L1, non-convergent L∞ (Ng 2009). Schwartz: the 3-face (linear, not bilinear) interpolation can make
   point-Jacobi diverge for adjacent small cells.
5. **Gradient vs solution (Nusselt/Sherwood):** for linear-ghost symmetric Dirichlet (Gibou 2002) gradients are 1st order
   (Yoon–Min 2015); for the standard quadratic-consistent central scheme gradients super-converge to 2nd order
   (Yoon–Min 2016). For cut-cell FV, 2nd-order wall gradient needs the normal-line quadratic stencil (Schwartz 2006).
   Note "1st-order L∞ gradient" errors are localised at the wall — exactly where Nu is evaluated; L1 surface-averaged Nu
   often converges faster (not checked in a source).
6. **Time integration:** EB operators are non-normal (complex eigenvalues); Crank–Nicolson can be oscillatory/zeroth
   order in max norm for Dirichlet EBs (McCorquodale 2001). Use L0-stable schemes (backward Euler, TGA, SDIRK).
7. **Conditioning:** symmetric linear-ghost Dirichlet (Gibou 2002) has cond ~ 1/(h·h_min) unpreconditioned — the
   diagonal grows as the wall approaches a node; Jacobi/SGS-type smoothing restores O(h⁻²) (Yoon–Min 2015). Cut-cell FV
   in κ-weighted form keeps cond bounded independent of κ (Schwartz 2006; Bochkov–Gibou 2020).

---

## 6. Summary table

| Method | BCs on EB | Solution order | Wall-flux / gradient order | Symmetric? | Stencil | Small-cell treatment (elliptic) | AMR | GPU / matrix-free fit | Codes |
|---|---|---|---|---|---|---|---|---|---|
| Johansen–Colella 1998 / Schwartz 2006 | D, N (inhom.) | 2 | 2 with normal-line quadratic (18 pts 3-D); 1 with LS fallback | no | 3×3×3 + 2 planes along n (≈2-cell reach) | κ-weighted equation; none needed for stability (implicit) | yes (Chombo) | good (local, fixed stencils; non-sym → BiCGStab or MG-as-solver) | Chombo, AMReX (variant), Basilisk (variant) |
| AMReX MLEBABecLap | N (hom.), D (hom./inhom.) | 2 | 1 (two-point along n, trilinear φ_g) | no | 27-pt (bilinear face + trilinear wall) | divide by κ / κ-weighted smoother | yes | proven GPU, MLMG | AMReX, incflo, PeleLMeX, MFIX-Exa, ERF |
| Crockett 2011 (two-material EB) | conjugate (jumps) | 2 | 2 (1 under-resolved) | no | as J–C, both sides | κ-weighted; flux recomputed per GS sweep | yes | good; local 2×2 per interface face | Chombo |
| Gibou 2002 linear ghost | D | 2 | 1 | **yes (SPD)** | 7-pt, diagonal-only change | none; cond ~1/(h·h_min) | yes (Min–Gibou octrees) | excellent (diag only) | Gibou group codes (CASL) |
| Ng–Min–Gibou 2009 aperture FV | N | 2 | 1 | **yes** | 7-pt × apertures | κ in storage | yes | excellent | — |
| Papac 2010 | Robin (q_wall = q_cell) | 2 (reported L1, L∞) | 1 (per Bochkov 2019 symmetric class) | **yes (SPD)** | 7-pt × apertures, diag Robin term a·L_Γ | κ (area) in storage | octree 2013 (non-sym at T-junctions) | excellent | Gibou group |
| Bochkov–Gibou 2019 (non-sym) / Arias 2018 | Robin (piecewise-smooth Γ) | 2 | 2 | no | compact + face-centroid interp. | κ-scaled | quadtree shown | good | Gibou group |
| Bochkov–Gibou 2020 | conjugate (jumps) | 2 | 1 | no (μ⁺≠μ⁻) | 5/7-pt + diagonals | κ-scaled, cond bounded | yes | good | — |
| Voronoi Interface Method 2015 | conjugate | 2 | 1 | yes | local Voronoi mesh | — | yes | poor (unstructured patch) | — |
| Devendran 2017 / Qian 2024 | D, N | 4 | 3–4 | no | wide LS stencils | LS / PLG | partial | poor for 1–2-cell halos | Chombo |
| Flux redistribution | (advection) | 1 at cut cells | — | — | 1-cell reach | redistribute (1−κ)δM | yes | excellent | Chombo, AMReX (FluxRedist), Basilisk variant |
| (W)SRD | (advection; also diffusive updates) | 2 | — | — | neighbourhood in 3×3×3; kernels need ~3-cell grown box | temporary overlapping merging to V ≥ ½h^d | yes (re-redistribution) | proven GPU (AMReX) | AMReX: incflo, PeleLMeX, ERF, MFIX-Exa |

---

## 7. ASSESSMENT — best fits for peclet (my judgement, not literature claims)

**A. Neumann / prescribed flux / Robin: keep the symmetric aperture FV with κ in storage and add Robin Papac-style as a
diagonal term (Papac 2010; Ng 2009; Bochkov–Gibou 2019 symmetric scheme).** It is exactly peclet's current operator plus
κ, stays SPD → MG-PCG and RB-GS unchanged, 7-point, 1-cell halo, AMR-ready (Papac 2013 octrees). Literature: 2nd-order
solution, 1st-order wall gradient. Action implied: re-test Robin with **q_wall := q_cell** (no d/D series resistance) and κ
storage; the literature reports 2nd order for that form, while the prototype's resistance form was 1st order — the
non-normal centre-to-wall distance (pitfall 3) is a likely cause (hypothesis to verify with the eigenvalue gate). If 2nd-order
Nusselt/Sherwood is needed, the upgrade path is the Bochkov–Gibou non-symmetric scheme (face-centroid interpolation;
BiCGStab, which peclet already has) — accept the loss of symmetry only where gradients are the deliverable.

**B. Dirichlet: Gibou 2002 symmetric linear ghost now; Schwartz/AMReX normal-line wall flux as the conservative-FV option.**
Gibou's ghost is diagonal-only, SPD, 2nd-order solution (proven), 1st-order gradient; mind cond ~1/(h·h_min) (RB-GS/MG
handle it). If the scalar must be discretely conservative inside cut cells (coupled to κ-storage advection and
redistribution), use the cut-cell FV with the wall gradient formed **on the normal through the boundary centroid**: AMReX's
two-point form (φ_g by trilinear interpolation at dg = max(0.3, (κ²−¼)/(2κ))·h; 8-cell octant; 1-cell halo; O(h) flux →
2nd-order solution) or the Schwartz normal-line quadratic (2nd-order gradient, 2-cell reach). Both are non-symmetric →
BiCGStab with the existing MG as preconditioner. Never the two-point flux to the cell centroid.

**C. Advection small cells: weighted state redistribution (Giuliani et al. 2022), with flux redistribution as a cheap
first step.** WSRD is 2nd order, conservative, production-proven on GPUs/AMR in AMReX (incflo, PeleLMeX, ERF — incl. a
staggered-mesh variant), works as a post-processing pass after the existing Koren-TVD update, and has an AMR coarse–fine
extension. Cost: ~3-cell grown-box reach in AMReX's implementation → peclet's 1–2-cell halos would need widening (or a
second exchange) for the redistribution pass. Flux redistribution needs only 1 cell but is 1st order at cut cells.

Conjugate (solid-side diffusion): Crockett–Colella–Graves 2011 (local 2×2 interface solve inside each GS sweep, κ-weighted FV on both sides,
2nd order, MG up to 10⁶ contrast) is the closest fit to a matrix-free MG design; Bochkov–Gibou 2020 is the compact
alternative (2nd-order solution, 1st-order flux, non-symmetric).
