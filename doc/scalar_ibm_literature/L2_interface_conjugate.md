# L2 — Interface / conjugate scalar transport on Cartesian grids: literature digest

Compiled 2026-10-02 for peclet scalar transport design: conjugate solid diffusion, partition coefficient
c_s = K c_f, contact resistance, and later Henry's-law jumps at VoF interfaces.

**Verification tags.** Every bibliographic line below (authors, year, journal, volume, pages, DOI) was
confirmed against Crossref metadata during this search. The content tags say how much of each paper was read:

- **[full]**: method sections were read from an open PDF.
- **[abs]**: only the abstract or the publisher/search summary was read.
- **[meta]**: only the bibliography was confirmed; the content description comes from how other papers I read
  characterise it.
- **UNVERIFIED**: a claim I could not check against a source.

---

## 0. Two structural facts that drive everything below (my analysis, not from a single paper)

1. **A partition coefficient does not have to break symmetry.** Write c = K_p ψ in each phase p, with
   K_f = 1 and K_s = K, so that ψ is continuous across the interface. The conservation law becomes
   K_p ∂ψ/∂t + … = ∇·(D_p K_p ∇ψ). This is the ordinary conjugate problem with conductivity Λ = D K and
   capacity K: symmetric, with mass matrix K·V.
   - Contact resistance enters as q = (ψ_a − ψ_b)/R. That is a symmetric two-point coupling.
   - Thermal conjugate transfer is the special case K = ρc_p.
   - The single-field CST of Haroun, Marschall and Farsoiya works in the untransformed c. That is what
     produces its non-symmetric drift term (§3).
2. **A two-point series-resistance face/interface flux is exactly the Liu–Fedkiw–Kang (2000) GFM effective
   coefficient.** For conjugate problems:
   - β̂ = β⁺β⁻/(β⁺θ + β⁻(1−θ)).
   - Adding the resistance: 1/β̂ = θh/β⁻ + R + (1−θ)h/β⁺.
   - The GFM is first-order in multi-D. The solution converges (Liu & Sideris 2003), but the flux does not
     converge pointwise because the tangential jump is smeared (Egan & Gibou 2020).
   - This matches the prototype's finding that the series-resistance flux is 1st order.
   - xGFM (§1.2) restores convergent gradients **without changing the SPD matrix**.

---

## 1. Elliptic/parabolic interface problems with jumps on Cartesian grids

### 1.1 Ghost Fluid Method / boundary-condition capturing (GFM)
- **Liu, X.-D., Fedkiw, R., Kang, M. (2000).** *A boundary condition capturing method for Poisson's equation
  on irregular domains.* J. Comput. Phys. 160, 151–178. doi:10.1006/jcph.2000.6444 [abs via search summary]
  - Handles [u] and [βu_n] jumps dimension by dimension.
  - The matrix is "the standard symmetric matrix for the variable coefficient Poisson equation", so
    black-box solvers apply.
  - Coco & Russo (2018, intro, [full]) describe it as first-order accurate.
- **Liu, X.-D., Sideris, T. (2003).** *Convergence of the ghost fluid method for elliptic equations with
  interfaces.* Math. Comp. 72, 1731–1746. doi:10.1090/s0025-5718-03-01525-4 [meta]
  - Convergence proof.
- **Kang, M., Fedkiw, R., Liu, X.-D. (2000).** *A boundary condition capturing method for multiphase
  incompressible flow.* J. Sci. Comput. 15, 323–360. doi:10.1023/A:1011178417620 [meta]
- **Hong, J.-M., Shinar, T., Kang, M., Fedkiw, R. (2007).** *On boundary condition capturing for multiphase
  interfaces.* J. Sci. Comput. 31, 99–125. doi:10.1007/s10915-006-9120-x [meta]
- **Gibou, F., Fedkiw, R., Cheng, L.-T., Kang, M. (2002).** *A second-order-accurate symmetric discretization
  of the Poisson equation on irregular domains.* J. Comput. Phys. 176, 205–227. doi:10.1006/jcph.2001.6977 [meta]
  - Dirichlet only. Symmetric linear ghost, 2nd order. This is the prototype's "symmetric linear ghost".
- **Ng, Y.-T., Chen, H., Min, C., Gibou, F. (2009).** *Guidelines for Poisson solvers on irregular domains with
  Dirichlet boundary conditions using the ghost fluid method.* J. Sci. Comput. 41, 300–320.
  doi:10.1007/s10915-009-9299-8 [meta]
- **Ianniello, S. (2024).** *GFMxP: a novel Ghost Fluid Method to solve the variable coefficient Poisson
  equation.* Int. J. Multiphase Flow 170, 104634. doi:10.1016/j.ijmultiphaseflow.2023.104634 [meta]
  - Content UNVERIFIED.

**Fit to peclet.**
- Stencil: compact 7-point, one unknown per cell (the phase is set by the sign of the SDF at the centre).
- Matrix: SPD. Multigrid works as for any variable-coefficient Poisson.
- Partition coefficient: via ψ (§0).
- Contact resistance: add R to the series sum.
- Accuracy: 1st order.
- Volume fractions and apertures are not used.

### 1.2 xGFM (extended GFM)
- **Egan, R., Gibou, F. (2020).** *xGFM: Recovering convergence of fluxes in the ghost fluid method.* J. Comput.
  Phys. 409, 109351. doi:10.1016/j.jcp.2020.109351 [abs, verbatim via Semantic Scholar]
  - The abstract says: "pointwise convergence of the gradient is recovered by ensuring consistent jump
    conditions with an iterative process. Compared to the GFM, the discretization stencil and the symmetric
    positive-definiteness of the resulting linear system are not modified; only the right-hand side of the
    linear system is affected."
  - GFM is shown to be consistent with the Gibou 2002 symmetric Dirichlet scheme in each subdomain, but its
    matching conditions are inconsistent.
  - Tested in 2-D and 3-D.
  - Search-engine summaries say it is demonstrated on distributed quad/octree grids and is "second order for
    solution and gradients". The abstract only says "convergent gradients", so the **exact gradient order is
    UNVERIFIED**.

**Fit to peclet.**
- It is a sequence of SPD solves with the same operator: an outer fixed-point or defect-correction loop on
  the RHS around the existing MG-PCG.
- It extends directly from the prototype's successful symmetric Dirichlet ghost.
- How partition coefficients and contact resistance enter the xGFM jump-extension step is not covered by the
  paper (UNVERIFIED). With ψ, a partition coefficient becomes [ψ] = 0. A resistance gives
  [ψ] = R Λ ∂_nψ, a flux-dependent jump that would be updated in the same iteration.

### 1.3 Immersed Interface Method (IIM)
- **LeVeque, R.J., Li, Z. (1994).** *The immersed interface method for elliptic equations with discontinuous
  coefficients and singular sources.* SIAM J. Numer. Anal. 31, 1019–1044. doi:10.1137/0731054 [meta]
- **Li, Z., Ito, K. (2006).** *The Immersed Interface Method.* SIAM. doi:10.1137/1.9780898717464 [meta]
- **Adams, L., Li, Z. (2002).** *The immersed interface/multigrid methods for interface problems.* SIAM J. Sci.
  Comput. 24, 463–479. doi:10.1137/s1064827501389849 [meta]
- **Adams, L., Chartier, T. (2005).** *A comparison of algebraic multigrid and geometric immersed interface
  multigrid methods for interface problems.* SIAM J. Sci. Comput. 26, 762–784. doi:10.1137/s1064827503425262 [meta]

Characterisation, per Coco–Russo and Bochkov–Gibou intros [full]:
- Taylor expansions modify a 6-point (2-D) stencil near the interface.
- It needs high-order jump conditions and surface derivatives.
- It is non-symmetric. Crockett et al. note "loss of symmetry".
- Multigrid needs custom interpolation (Adams & Li) or AMG.
- Poor fit for a matrix-free SPD stack.

### 1.4 Voronoi Interface Method (VIM)
- **Guittet, A., Lepilliez, M., Tanguy, S., Gibou, F. (2015).** *Solving elliptic problems with discontinuities
  on irregular domains – the Voronoi Interface Method.* J. Comput. Phys. 298, 747–765.
  doi:10.1016/j.jcp.2015.06.026 [abs via search]
  - Accuracy: 2nd-order solution and 1st-order gradients in L∞, even at large diffusivity ratios.
  - Matrix: **SPD, and only the RHS carries the jumps.**
  - Requires a local Voronoi mesh near the interface, plus interpolation back to the Cartesian grid. Bochkov
    & Gibou note "challenges, especially in 3-D".
- **Guittet, A., Poignard, C., Gibou, F. (2017).** J. Comput. Phys. 332, 143–159. doi:10.1016/j.jcp.2016.11.048 [meta]
- **Mistani, P., Guittet, A., Poignard, C., Gibou, F. (2019).** *A parallel Voronoi-based approach for
  mesoscale simulations of cell aggregate electropermeabilization.* J. Comput. Phys. 380, 48–64.
  doi:10.1016/j.jcp.2018.12.009 [meta]
  - Parallel octree VIM.

**Fit to peclet:** poor. Connectivity near the interface is irregular, which does not suit compact
matrix-free GPU stencils or GMG coarsening.

### 1.5 Bochkov & Gibou (2020): finite volume with jumps, one unknown per cell
- **Bochkov, D., Gibou, F. (2020).** *Solving elliptic interface problems with jump conditions on Cartesian
  grids.* J. Comput. Phys. 407, 109269. doi:10.1016/j.jcp.2020.109269 [full, arXiv:1905.08718]
- Discretisation:
  - Finite volume over the cell, split by phase.
  - Each grid point carries one unknown, the value on its own side.
  - The other-side value comes from a normal Taylor expansion plus one-sided linear least-squares.
  - No augmented variables.
- Accuracy: 2nd-order solution, 1st-order gradients in L∞. Retaining quadratic terms would give 2nd-order
  gradients; the authors leave that as future work.
- Matrix: "In general the case where µ⁺ ≠ µ⁻, the resulting linear system is **nonsymmetric**."
- Stencil: in the worst case the 5-point stencil plus diagonal neighbours (9-point in 2-D, 27-point in 3-D).
- Condition number: bounded as µ⁺/µ⁻ → 0 or ∞ with the "Bias Slow" variant; it scales as h⁻².
- Solver: BiCGStab with hypre.
- AMR: demonstrated on quad/octree grids that are uniform near the interface, with a superconvergent scheme
  elsewhere.
- **Bochkov, D., Gibou, F. (2019).** *Solving Poisson-type equations with Robin boundary conditions on piecewise
  smooth interfaces.* J. Comput. Phys. 376, 1156–1198. doi:10.1016/j.jcp.2018.10.020 [abs]
  - Two finite-volume schemes:
    - (i) **Symmetric**: 2nd-order solution, 1st-order gradients.
    - (ii) Non-symmetric: 2nd-order solution and 2nd-order gradients.
  - In 2-D and 3-D.
  - **This is the closest literature analogue to a symmetric cut-cell Robin operator.**
- **Papac, J., Gibou, F., Ratsch, C. (2010).** *Efficient symmetric discretization for the Poisson, heat and
  Stefan-type problems with Robin boundary conditions.* J. Comput. Phys. 229, 875–889.
  doi:10.1016/j.jcp.2009.10.017 [meta]
- **Papac, J., Helgadottir, A., Ratsch, C., Gibou, F. (2013).** Same topic on quadtree/octree grids. J. Comput.
  Phys. 233, 241–261. doi:10.1016/j.jcp.2012.08.038 [meta]
- **Arias, V., Bochkov, D., Gibou, F. (2018).** Robin solver with 2nd-order gradients. J. Comput. Phys. 365,
  1–6. doi:10.1016/j.jcp.2018.03.022 [meta]
- **Ng, Y.-T., Min, C., Gibou, F. (2009).** *An efficient fluid–solid coupling algorithm for single-phase
  flows.* J. Comput. Phys. 228, 8807–8829. doi:10.1016/j.jcp.2009.08.032 [abs]
  - Cut-cell finite volume, SPD for both the projection and implicit viscosity.

### 1.6 Embedded-boundary (cut-cell) finite volume with discontinuous coefficients (Colella group)
- **Crockett, R.K., Colella, P., Graves, D.T. (2011).** *A Cartesian grid embedded boundary method for solving
  the Poisson and heat equations with discontinuous coefficients in three dimensions.* J. Comput. Phys. 230,
  2451–2469. doi:10.1016/j.jcp.2010.12.017 [full, LBNL eScholarship preprint]
- Discretisation:
  - Cut-cell finite volume after Johansen–Colella, with a separate control volume ("VoF") per material in
    each cut cell.
  - Each side writes its normal derivative as ∂ϕ/∂n|_B,p = w_B ϕ_B,p + Σ w_i ϕ_i. The second-order version
    interpolates quadratically at 2 points along the normal; a least-squares fallback is used otherwise.
  - With the two matching conditions this gives a **local 2×2 (4×4) linear system for ϕ_B and the flux**.
  - The resulting boundary fluxes enter Gauss–Seidel relaxation in each phase, and "boundary fluxes are
    recalculated after every step in relaxing".
- Solver and robustness:
  - Geometric multigrid. Coarsening has special cases: least-squares stencil failure, and coarse VoFs that
    lack a stencil.
  - Stable and efficient for **coefficient contrasts up to 10⁶**.
- Accuracy: overall 2nd order.
- AMR: tested (Chombo).
- Heat equation: also covered.
- Symmetry: the 2-probe interface flux makes the operator non-symmetric. The paper says the extended
  least-squares fallback "is useful in preserving symmetry in cases where the boundary normal is along a
  cardinal direction" only.
- **Partition coefficient and contact resistance**: both are linear and local, so they enter the same 2×2
  elimination (ϕ_B⁺ = Kϕ_B⁻; flux = (ϕ_B⁺ − ϕ_B⁻)/R). This is my analysis.
- **Johansen, H., Colella, P. (1998).** J. Comput. Phys. 147, 60–85. doi:10.1006/jcph.1998.5965 [meta]
- **Schwartz, P., Barad, M., Colella, P., Ligocki, T. (2006).** EB heat/Poisson in 3-D. J. Comput. Phys. 211,
  531–550. doi:10.1016/j.jcp.2005.06.010 [meta]
- **McCorquodale, P., Colella, P., Johansen, H. (2001).** EB heat equation. J. Comput. Phys. 173, 620–635.
  doi:10.1006/jcph.2001.6900 [meta]

### 1.7 Oevermann & Klein cut-cell finite volume
- **Oevermann, M., Klein, R. (2006).** *A Cartesian grid finite volume method for elliptic equations with
  variable coefficients and embedded interfaces.* J. Comput. Phys. 219, 749–769. doi:10.1016/j.jcp.2006.04.010 [abs via search]
  - Finite volume with a dual bilinear (finite-element-like) reconstruction in cut cells.
  - Level-set interface; 2nd order in L∞ and L².
  - Small-cell singularities are removed asymptotically.
  - Symmetry UNVERIFIED (probably non-symmetric).
- **Oevermann, M., Scharfenberg, C., Klein, R. (2009).** *A sharp interface finite volume method for elliptic
  equations on Cartesian grids.* J. Comput. Phys. 228, 5184–5206. doi:10.1016/j.jcp.2009.04.018 [meta]

### 1.8 Cisternino & Weynans
- **Cisternino, M., Weynans, L. (2012).** *A parallel second order Cartesian method for elliptic interface
  problems.* Commun. Comput. Phys. 12, 1562–1587. doi:10.4208/cicp.160311.090112a [abs]
  - Finite differences with **additional interface unknowns** that express the transmission conditions.
  - 2nd order everywhere.
  - Parallelised with PETSc.
  - Symmetry: a summary asserted it, but I could not confirm. **UNVERIFIED**; it is likely non-symmetric
    because of the interface rows.
- **Weynans, L. (2026).** *Convergence of a Cartesian method for interface elliptic problems.* ICIAM 2023
  proceedings, Springer, pp. 161–189. doi:10.1007/978-981-95-6107-0_8 [meta]

### 1.9 Coco & Russo: ghost-point multigrid (designs the multigrid around the interface)
- **Coco, A., Russo, G. (2018).** *Second order finite-difference ghost-point multigrid methods for elliptic
  problems with discontinuous coefficients on an arbitrary interface.* J. Comput. Phys. 361, 299–330.
  doi:10.1016/j.jcp.2018.01.016 [full, Oxford Brookes preprint]
- Unknowns: grid points near the interface carry a real value and a ghost value (extrapolated from the other
  side through the jump conditions).
- Coefficients: matrix-valued coefficients are also handled.
- Accuracy: **2nd order in the solution and the gradient**, with monotone error decay even at large jumps.
- Matrix: **non-symmetric** ("The resulting linear system is not symmetric, and a proper multigrid solver is
  proposed").
- Multigrid design:
  - The ghost-point equations get their own relaxation.
  - Extra relaxation sweeps are done near the interface and boundary.
  - GS-LEX, W-cycle, ν₁ = 2, ν₂ = 1.
  - The convergence factor approaches the Local Fourier Analysis (LFA) value for rectangles and is
    **independent of the coefficient jump and of h**.
- **Coco, A., Russo, G. (2013).** *Finite-difference ghost-point multigrid methods on Cartesian grids for elliptic
  problems in arbitrary domains.* J. Comput. Phys. 241, 464–501. doi:10.1016/j.jcp.2012.11.047 [meta]
  - Mixed Dirichlet/Neumann.
- **Coco, A., Russo, G. (2012).** 1-D interface multigrid. Numer. Math. Theor. Meth. Appl. 5, 19–42.
  doi:10.4208/nmtma.2011.m12si02 [abs, arXiv:1111.1167]

### 1.10 Other symmetric interface schemes
- **Qin, Z., Riaz, A., Balaras, E. (2020).** *A locally second order symmetric method for discontinuous solution
  of Poisson's equation on uniform Cartesian grids.* Comput. Fluids 198, 104397.
  doi:10.1016/j.compfluid.2019.104397 [abs]
  - The composite solution is a weighted average of two fictitious fields in each interfacial cell.
  - The Poisson coefficient is smoothed in a narrow band.
  - Uses an ordinary SDF; the tangential gradient jump is not needed.
  - **The linear system is symmetric**, and solutions are 2nd order at points adjacent to the interface.
  - A candidate worth reading in full; details are UNVERIFIED.
- **Cao, F., Yuan, D., Jia, D., Yuan, G. (2024).** *A finite difference method for two dimensional elliptic
  interface problems with imperfect contact.* J. Comput. Math. 42(5), 1328–1355. doi:10.4208/jcm.2302-m2022-0111 [abs]
  - **Contact resistance** with a flux-dependent jump.
  - About 2nd order; monotonicity proved in special cases.
  - Symmetry not stated.

---

## 2. Conjugate heat transfer with IB, ghost-cell and cut-cell methods

- **Das, S., Panda, A., Deen, N.G., Kuipers, J.A.M. (2018).** *A sharp-interface immersed boundary method to
  simulate convective and conjugate heat transfer through highly complex periodic porous structures.* Chem.
  Eng. Sci. 191, 1–18. doi:10.1016/j.ces.2018.04.061 [full, TU/e Pure]
  - Uniform Cartesian grid with ghost cells on both sides.
  - The interface temperature comes from **4 probes**, 2 per side at distances Δn and 2Δn along the normal:
    T_S = [k_f(4T₁−T₂) + k_s(4T₃−T₄)] / 3(k_f+k_s). The 2-probe fallback is (k_fT₁+k_sT₃)/(k_f+k_s).
  - T_S is then imposed as a Dirichlet value on each side, which **decouples** the two solves. T_S is taken
    from stored values (lagged), so the coupling is partitioned rather than monolithic. The paper names a
    fully implicit CHT only for its 1-D reference.
  - Solver: B-ICCG.
  - Verification:
    - Stagnant k_eff of a model porous medium at k_s/k_f = 10, 100, 1000 converges at 2nd order, but
      absolute accuracy drops as the ratio grows.
    - Transient hot sphere in an infinite medium.
    - Open-cell foam.
  - It is the same Eindhoven multiphase group lineage.
- **Nagendra, K., Tafti, D.K., Viswanath, K. (2014).** *A new approach for conjugate heat transfer problems
  using immersed boundary method for curvilinear grid based solvers.* J. Comput. Phys. 267, 225–246.
  doi:10.1016/j.jcp.2014.02.045 [abs via search]
  - CHT condition via mirror ghost nodes in the fluid and probe points in the solid.
  - "No complex interpolations".
  - Benchmarks UNVERIFIED.
- **Crocker, R., Dubief, Y., Desjardins, O. (2014).** *A second order thermal and momentum immersed boundary
  method for conjugate heat transfer in a Cartesian finite volume solver.* arXiv:1411.1004 [abs]
  - Level-set-defined interface.
  - Benchmark: **co-annular rotating cylinders (Re = 50), an analytic solution**, 2nd order in L₂ and L∞
    for k_s/k_f = 9 to 100.
  - Also a heated turbulent channel at Re_τ = 150 with k_s/k_f = 4.
  - Journal version UNVERIFIED.
- **Narváez, G.F., Lamballais, E., Schettini, E.B.C. (2021).** *Simulation of turbulent flow subjected to
  conjugate heat transfer via a dual immersed boundary method.* Comput. Fluids 229, 105101.
  doi:10.1016/j.compfluid.2021.105101 [abs]
  - Two temperature fields, each smoothly reconstructed inside its own immersed zone.
  - Weakly coupled at the interface.
  - 2nd order against analytic solutions: channel, pipe, and a **composite wall with variable conductivity**.
- **Kumar, M., Natarajan, G. (2019).** *Diffuse-interface immersed-boundary framework for conjugate-heat-transfer
  problems.* Phys. Rev. E 99, 053304. doi:10.1103/PhysRevE.99.053304 [abs]
  - A monolithic, unified energy equation with volume-fraction-blended conductivity.
  - Diffuse interface, not sharp.
- **Xia, J., Luo, K., Fan, J. (2014).** *A ghost-cell based high-order immersed boundary method for inter-phase
  heat transfer simulation.* Int. J. Heat Mass Transf. 75, 302–312. doi:10.1016/j.ijheatmasstransfer.2014.03.048 [abs via search]
  - Heated cylinder at Re = 50, 100, 200 with Dirichlet and Neumann conditions.
  - Whether it handles conjugate transfer is UNVERIFIED.
- **Pan, D. (2010).** *A simple and accurate ghost cell method for the computation of incompressible flows over
  immersed bodies with heat transfer.* Numer. Heat Transf. B 58, 17–39. doi:10.1080/10407790.2010.504697 [meta]
  - Conjugate coverage UNVERIFIED.
- **Kang, S., Iaccarino, G., Ham, F. (2008).** *DNS of a buoyancy-dominated turbulent flow using an immersed
  boundary method.* CTR Annual Research Briefs 2008, p. 231.
  https://web.stanford.edu/group/ctr/ResBriefs08/18_SKang.pdf [abs via search]
  - Turbulent conjugate heat transfer with IB.
- **Liu, M., Hasegawa, Y. (2026).** *A volume penalization method for solving conjugate scalar transport with
  interfacial flux jump conditions.* Comput. Fluids 318, 107217. doi:10.1016/j.compfluid.2026.107217
  (arXiv:2601.10134) [abs]
  - Handles flux **and scalar jumps** (partition-type) by penalisation.
  - About 3% deviation against body-fitted solutions; no formal convergence order stated.
- **Orova, M., Yiantsios, S.G. (2024).** IB for mass transport with discontinuous species concentration via
  source dipoles. J. Eng. Math. 145. [abs via search]
  - DOI not confirmed: Crossref only returned an SSRN preprint, 10.2139/ssrn.4441823. **Treat the journal
    citation as UNVERIFIED.**
- **Not found:** the "Xu, … 'A sharp interface method for conjugate heat transfer'" reference in the brief is
  **UNVERIFIED**.
- **GPU:** I located **no** GPU implementation of a sharp conjugate or jump-condition Cartesian scalar solver.
  This gap is UNVERIFIED, not proven.

**Benchmarks worth adopting.** Each has an analytic or semi-analytic solution.
- Concentric cylinders or spheres, steady radial conduction. Add K and R and the solution stays analytic.
- Co-annular rotating cylinders (Crocker).
- Composite wall with variable conductivity (Narváez).
- Transient sphere in an infinite medium (Das).
- Stagnant k_eff of a periodic array checked against the parallel/series bounds (Das).
- Heated cylinder in cross-flow (Xia; Nusselt numbers against the literature).

---

## 3. Two-phase species transfer with a Henry's-law jump (VoF / level set)

### 3.1 Single-field CST (continuous species transfer)
- **Haroun, Y., Legendre, D., Raynal, L. (2010).** *Volume of fluid method for interfacial reactive mass
  transfer: application to stable liquid film.* Chem. Eng. Sci. 65, 2896–2909. doi:10.1016/j.ces.2010.01.012 [meta]
- **Haroun, Y., Legendre, D., Raynal, L. (2010).** Chem. Eng. Sci. 65, 351–356. doi:10.1016/j.ces.2009.07.018 [meta]
- **Marschall, H., Hinterberger, K., Schüler, C., Habla, F., Hinrichsen, O. (2012).** *Numerical simulation of
  species transfer across fluid interfaces in free-surface flows using OpenFOAM.* Chem. Eng. Sci. 78, 111–127.
  doi:10.1016/j.ces.2012.02.034 [meta]
- **Deising, D., Marschall, H., Bothe, D. (2016).** *A unified single-field model framework for VOF simulations
  of interfacial species transfer applied to bubbly flows.* Chem. Eng. Sci. 139, 173–195.
  doi:10.1016/j.ces.2015.06.021 [meta]
  - Per Farsoiya et al. 2021, it verified the **harmonic-mean D** closure.
- **Maes, J., Soulaine, C. (2018).** *A new compressive scheme to simulate species transfer across fluid
  interfaces using the VOF method.* Chem. Eng. Sci. 190, 405–418. doi:10.1016/j.ces.2018.06.026 [abs]
  - Standard CST is strongly diffusive at high Pe; C-CST adds a compressive term.
  - Verified against a 1-D tube analytic solution.
- **Maes, J., Soulaine, C. (2020).** *A unified single-field VOF-based formulation for multi-component
  interfacial transfer with local volume changes.* J. Comput. Phys. 402, 109024.
  doi:10.1016/j.jcp.2019.109024 [meta]

**Structure of CST.** I read it in Farsoiya 2021 [full]:
- c = T c_l + (1−T) c_g.
- The interfacial D is harmonic.
- The Henry jump enters as a drift term −∇·(D c (α−1)/(αT+1−T) ∇T).
- This is **non-symmetric**, and the interface is smeared over 1–2 cells.
- It is only loosely related to the cut-cell solid operator. CST is the diffuse ("volume-of-solid") analogue,
  not the sharp one.

### 3.2 Two-field, geometric VoF, and subgrid models
- **Bothe, D., Fleckenstein, S. (2013).** *A Volume-of-Fluid-based method for mass transfer processes at fluid
  particles.* Chem. Eng. Sci. 101, 283–302. doi:10.1016/j.ces.2013.05.029 [meta]
  - Two-field geometric VoF with a subgrid-scale model, as Farsoiya and Zhao characterise it.
- **Fleckenstein, S., Bothe, D. (2015).** J. Comput. Phys. 301, 35–58. doi:10.1016/j.jcp.2015.08.011 [meta]
  - Multicomponent, with local volume change.
- **Weiner, A., Bothe, D. (2017).** *Advanced subgrid-scale modeling for convection-dominated species transport
  at fluid interfaces with application to mass transfer from rising bubbles.* J. Comput. Phys. 347, 261–289.
  doi:10.1016/j.jcp.2017.06.040 [meta]
- **Davidson, M.R., Rudman, M. (2002).** Numer. Heat Transf. B 41, 291–308. doi:10.1080/104077902753541023 [meta]
  - Continuous concentration, no jump.

### 3.3 Basilisk
- **Farsoiya, P.K., Popinet, S., Deike, L. (2021).** *Bubble-mediated transfer of dilute gas in turbulence.*
  J. Fluid Mech. 920, A34. doi:10.1017/jfm.2021.447 [full]
  - One-fluid CST following Haroun, with harmonic D.
  - **Advection is made consistent by two VoF-coupled tracer fields** (φ_l = c_l T, φ_g), following
    Bothe & Fleckenstein.
  - Implicit Euler diffusion with the drift folded into a Helmholtz-type operator, solved by Basilisk's
    quadtree **multigrid**.
  - Tested on a static bubble (inverse Laplace) and a rising bubble (Levich).
- **Farsoiya, P.K., Magdelaine, Q., Antkowiak, A., Popinet, S., Deike, L. (2023).** *Direct numerical
  simulations of bubble-mediated gas transfer and dissolution in quiescent and turbulent flows.* J. Fluid Mech.
  954, A29. [meta via ADS/Princeton]
- **Gennari, G., Jefferson-Loveday, R., Pickering, S.J. (2022).** *A phase-change model for diffusion-driven
  mass transfer problems in incompressible two-phase flows.* Chem. Eng. Sci. 259, 117791.
  doi:10.1016/j.ces.2022.117791 [abs]
  - A two-scalar species method on geometric VoF.
  - How the jump is imposed is UNVERIFIED.
- **Cipriano, E., Frassoldati, A., Faravelli, T., Popinet, S., Cuoci, A. (2024).** *Multicomponent droplet
  evaporation in a geometric VOF framework.* J. Comput. Phys., 112955. [meta via the Zhao 2026 reference list]
  - Interfacial fluxes are lagged (explicit), per Zhao.
- **Limare, A., Popinet, S., Josserand, C., Xue, Z., Ghigo, A. (2023).** *A hybrid level-set / embedded boundary
  method applied to solidification–melt problems.* J. Comput. Phys. 474, 111829. doi:10.1016/j.jcp.2022.111829 [abs]
  - **Two-fluid embedded boundaries in Basilisk**: conservative, 2nd order, Stefan problem.
  - This is the existing two-sided version of `embed.h`.

### 3.4 Sharp: embedded boundary on PLIC, and ghost fluid
- **Zhao, S., Zhang, J., Ni, M.-J. (2022).** *Boiling and evaporation model for liquid–gas flows: a sharp and
  conservative method based on the geometrical VOF approach.* J. Comput. Phys. 452, 110908.
  doi:10.1016/j.jcp.2021.110908 [meta]
- **Zhao, S., Zhang, J., Ni, M.-J. (2026, preprint).** *A sharp and conservative VOF method for multicomponent
  liquid–gas mass transfer.* arXiv:2608.19254 [full]
  - **Species are solved separately per phase.**
  - The **PLIC interface is treated as an embedded boundary**.
  - Species use a one-sided Robin condition. Temperature uses a two-sided flux-jump condition, built from
    2-point normal probes on each side with the interface value eliminated simultaneously.
  - Fluxes are real fluxes through Γ, so the scheme is conservative.
  - Basilisk with quad/octree AMR and multigrid.
  - "Second-order accuracy".
  - The paper does not say whether the matrix is symmetric.
- **Salimi, S.Z., Scapin, N., Popescu, E.-R., Costa, P., Brandt, L. (2024).** *A VoF method for multicomponent
  droplet evaporation with Robin boundary conditions.* J. Comput. Phys. 514, 113211. [abs via OpenAlex]
  - Robin coupling between gas and liquid species.
  - Ghost values via normal-derivative reconstruction plus PDE extrapolation, after Chai et al. 2021:
    **Chai, M., Luo, K., Wang, H., Zheng, S., Fan, J. (2021).** Comput. Fluids 214, 104772.
    doi:10.1016/j.compfluid.2020.104772 [meta]
  - 2nd order for Poisson–Robin, 1st order for the Stefan problem.
  - DOI for Salimi et al. not seen in Crossref (UNVERIFIED); the SSRN record exists.
- **Scapin, N., Costa, P., Brandt, L. (2020).** *A volume-of-fluid method for interface-resolved simulations of
  phase-changing two-fluid flows.* J. Comput. Phys. 407, 109251. doi:10.1016/j.jcp.2020.109251 [meta]
- **Tanguy, S., Sagan, M., Lalanne, B., Couderc, F., Colin, C. (2014).** *Benchmarks and numerical methods for
  the simulation of boiling flows.* J. Comput. Phys. 264, 1–22. doi:10.1016/j.jcp.2014.01.014 [meta]
  - Ghost fluid plus level set.

### 3.5 Which approaches share structure with a cut-cell solid treatment

**Yes (one operator family):**
- Crockett–Colella–Graves two-sided embedded boundary.
- Limare et al. two-fluid embed (Basilisk).
- Zhao et al. 2022/2026: the PLIC face is an embedded boundary with apertures and Γ-area, and both
  Robin and flux-jump conditions are closed with normal probes.
- Bochkov & Gibou 2019/2020: finite volume with Robin and jump conditions in one family.
- Das et al. 4-probe CHT, which is the same algebra used as a lagged Dirichlet value.

**Partly:** GFM/xGFM. The same jump algebra applies to solid and fluid interfaces, but the method uses no
apertures.

**No:** CST and C-CST, which are diffuse single-field. They are the counterpart of a volume-penalisation solid,
not of a cut cell.

---

## 4. Unified Dirichlet / Neumann / Robin / jump formulations, and symmetry

**The common pattern.**
- Per cut cell and per side p, approximate ∂_nφ_p|_Γ = w_B,p φ_Γ,p + Σ w_i φ_i, which is linear in the
  interface trace.
- Every interface condition of the general form a⁺q⁺ + a⁻q⁻ + b⁺φ⁺ + b⁻φ⁻ = d (two such rows for a two-sided
  interface) is then a **local 2×2 linear solve** for (φ_Γ⁺, φ_Γ⁻).
- Substituting back gives interface fluxes that are linear in the cell unknowns.
- Dirichlet, Neumann and Robin are the one-sided degenerate cases.
- This is the Crockett (2011), Das (2018) and Zhao (2026) mechanism. It covers the pymrm-style general
  condition directly; that is my analysis.

**How symmetry is kept or lost:**
- **Kept** when each side's normal derivative uses only the **own cell's** value (two-point:
  w_i = −w_B on the cell itself). The eliminated interface flux is then (φ_a − φ_b)/(d_a/Λ_a + R + d_b/Λ_b),
  a symmetric M-matrix coupling. This is GFM, the symmetric Robin schemes of Papac 2010 and Bochkov–Gibou 2019,
  and Ng–Min–Gibou.
  - Cost: 1st-order flux. The solution is still 2nd order for the one-sided Robin case through
    supra-convergence (Bochkov–Gibou 2019, scheme i).
- **Lost** as soon as probes or least-squares reach neighbouring cells to get a 2nd-order normal derivative.
  This happens in Crockett, Bochkov–Gibou 2020, Coco–Russo, IIM and Zhao.
- **Recovered with the operator unchanged** by moving the high-order part to the RHS:
  - xGFM iterates the jump consistency.
  - Generically: defect correction, an SPD low-order solve on A_L with residual from the high-order A_H.
  - Or VIM, a geometric construction that is SPD by design.
- **Qin–Riaz–Balaras 2020** claim a symmetric system with locally 2nd-order solutions, using two blended
  fictitious fields.

---

## 5. Summary table

| method | conditions handled | solution order | interface-flux order | symmetric? | stencil | MG/CG-friendly | AMR | used for VoF/LS? |
|---|---|---|---|---|---|---|---|---|
| GFM (Liu–Fedkiw–Kang 2000) | [u], [βu_n] jumps; R and K via series coefficient (my analysis) | 1 | not pointwise convergent (Egan–Gibou) | **yes (SPD)** | 5/7-pt | **yes**, plain GMG/PCG | yes (octree in Gibou group) | yes, LS (pressure, boiling) |
| xGFM (Egan–Gibou 2020) | same as GFM | 2 (per search summary) | convergent (order UNVERIFIED) | **yes, RHS-only corrections** | 5/7-pt | **yes**, repeated SPD solves | quad/octree (per search summary) | LS |
| Gibou 2002 symmetric ghost | Dirichlet only | 2 | 1 | yes | 5/7-pt | yes | yes | LS |
| Symmetric Robin FV (Papac 2010; B–G 2019 i) | Robin/Neumann, one-sided | 2 | 1 | **yes** | compact | yes | quad/octree (Papac 2013) | LS |
| B–G 2019 ii | Robin | 2 | 2 | no | compact+ | BiCGStab | — | LS |
| IIM (LeVeque–Li 1994) | general jumps | 2 | ~1–2 | no | 6/9-pt modified | custom MG / AMG | limited | LS |
| VIM (Guittet 2015) | jumps | 2 | 1 | **yes (SPD)** | local Voronoi (irregular) | yes, but not structured | parallel octree (Mistani 2019) | LS |
| Bochkov–Gibou 2020 | jumps | 2 | 1 | **no** | up to 9/27-pt | BiCGStab+hypre; bounded κ | quad/octree (uniform near Γ) | LS |
| Crockett–Colella–Graves 2011 | jumps; K and R slot in (my analysis) | 2 | ~1–2 (UNVERIFIED) | no | cut-cell + 2-probe normals | **GMG, contrast up to 1e6** | **yes (Chombo)** | EB (not VoF) |
| Oevermann–Klein 2006/09 | jumps | 2 | UNVERIFIED | UNVERIFIED (likely no) | cut-cell bilinear | UNVERIFIED | — | LS |
| Cisternino–Weynans 2012 | jumps, extra interface unknowns | 2 | ~1–2 | UNVERIFIED | FD + interface DOFs | PETSc | — | LS |
| Coco–Russo 2018 | jumps (incl. matrix coefficients) | 2 | **2** | no | FD + ghost DOFs | **custom GMG, jump-independent rate** | — | LS |
| Qin–Riaz–Balaras 2020 | jumps | 2 (local) | UNVERIFIED | **yes** | compact (band) | yes (presumably) | — | LS |
| Das 2018 CHT (4 probes) | conjugate (continuity) | 2 (global k_eff) | — | each side SPD; coupling lagged | 7-pt + probes | ICCG | — | — |
| Kumar–Natarajan 2019 | conjugate (diffuse) | ~2 overall (claimed) | smeared | yes | 7-pt | yes | — | (diffuse) |
| CST / C-CST (Haroun; Marschall; Deising; Maes–Soulaine) | Henry jump (diffuse) | ≤1 near Γ (smeared) | smeared | no (drift term) | 7-pt | Basilisk MG (Farsoiya) | quadtree (Basilisk) | **VoF** |
| Two-field + SGS (Bothe–Fleckenstein; Weiner–Bothe) | Henry, flux continuity | — | subgrid model | UNVERIFIED | — | — | — | VoF (geometric) |
| EB-on-PLIC (Zhao 2022/2026) | Robin per phase, flux jump, equilibrium | 2 | 2 (claimed "accurate") | not stated (probes → likely no) | cut-cell + 2-probe | Basilisk MG | **quad/octree** | **VoF (PLIC)** |
| Two-fluid embed (Limare 2023) | Stefan / two-fluid | 2 | 2 (claimed) | UNVERIFIED | cut-cell | Basilisk MG | quad/octree | LS+EB |
| Robin ghost (Salimi 2024 / Chai 2021) | Robin coupling of phases | 2 (Poisson) | — | UNVERIFIED | ghost + PDE extrapolation | — | — | VoF |

---

## 6. ASSESSMENT — which 2–3 approaches fit peclet best (my opinion, not from the literature)

**1. Default: a symmetric GFM/series-resistance operator, upgraded with xGFM-style RHS correction (or generic
defect correction).**
- It keeps the existing matrix-free 7-point SPD stack unchanged: RB-GS, GMG and MG-PCG. Only the face
  coefficients change, through a series sum that already includes K (via ψ) and R.
- It extends the prototype's successful symmetric Dirichlet ghost.
- Accuracy is recovered by an outer loop that changes only the RHS.
- Risks:
  - The number of outer iterations is unknown.
  - It ignores apertures, so cut-cell volumes and conservation are not geometric.
  - The xGFM gradient order and any octree specifics need the full paper (UNVERIFIED here).

**2. Second-order, aperture-aware, and the long-term unifier: a two-sided cut-cell finite volume
(Crockett–Colella–Graves style).**
- Two unknowns per cut cell (one per phase, each with its own volume fraction and apertures), with
  2-probe normal derivatives.
- A local 2×2 elimination at Γ absorbs K, R, Robin, Dirichlet and Neumann alike.
- It is the same operator family that Basilisk now uses for:
  - solid walls (`embed.h`, already ported into flow);
  - two-fluid embedded boundaries (Limare 2023);
  - PLIC species and heat jumps (Zhao 2026).
- So **one operator serves solid walls and VoF interfaces**. That answers question 3 of the brief.
- Cost: the operator is non-symmetric, so use BiCGStab with GMG, or GMG with GS relaxation that recomputes
  the interface fluxes, as Crockett does (robust to 1e6 contrast).
- Alternatively, use the symmetric two-point version as the MG/PCG preconditioner and the 2-probe version
  as the defect-correction operator. This combines with option 1.
- Coco–Russo is the reference for making geometric multigrid converge at a jump-independent rate on such
  ghost or interface rows.

**3. For VoF species, not CST.** CST is non-symmetric, smeared, and loses accuracy at high Sc/Pe (Maes–Soulaine
2018; Weiner–Bothe 2017).
- Prefer the ψ = c/K transform for the diffusion solve, which is symmetric.
- Use two VoF-consistent tracers for advection, as in Bothe–Fleckenstein and Farsoiya.
- Put the interface coupling in the operator family of option 2.

**Open question for the prototype.** The symmetric Robin schemes in the literature have 1st-order flux
truncation in cut cells, yet still give a 2nd-order solution through supra-convergence (Bochkov–Gibou 2019,
scheme i; Papac 2010). The prototype's two-point series flux is only 1st order. It is worth checking:
- whether the distance d is measured from the cell centre or from the partial-volume centroid, and along the
  normal or along the grid line;
- whether the 1st order shows up only in L∞ at tiny cut cells.
