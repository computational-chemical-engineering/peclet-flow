# L3 — Particle-resolved scalar transport: methods, accuracy, benchmarks, verification gates

Literature digest for peclet scalar (heat/mass) transport with immersed SDF solids
(Dirichlet / Neumann / Robin / conjugate). Compiled 2026-10-02.

**How this was checked.** Every bibliographic entry below was checked against Crossref metadata (DOI, authors,
journal, volume, pages). "Read" means I read the full text (open PDF, TU/e repository or Europe PMC).
"Abstract" means I read only the abstract, from OpenAlex or Semantic Scholar. **UNVERIFIED** marks anything that
comes from my own recollection, or from a search-engine snippet I could not confirm against the source. Elsevier
and Springer full texts were blocked (403 / captcha), so method details for those papers are limited to what the
abstracts and secondary sources say.

---

## 1. Particle-resolved DNS of heat/mass transfer: methods, boundary treatment, accuracy

### 1a. Eindhoven (Kuipers / Deen / Peters) sharp-interface line — most relevant to peclet

| Ref | BC method | Order / accuracy evidence | Resolution finding |
|---|---|---|---|
| **Deen, Kriebitzsch, van der Hoef, Kuipers (2012)**, *Chem. Eng. Sci.* 81, 329–344. [doi:10.1016/j.ces.2012.06.055](https://doi.org/10.1016/j.ces.2012.06.055). Metadata only; full text blocked. | **Implicit** incorporation of the Dirichlet temperature into the discrete fluid equations. Uses a **directional quadratic** fit along each grid line through the wall point, the IB-fluid cell and the next fluid cell. The coefficients are reproduced in Das 2018 (below). | 2nd order (as cited by Das 2018). Verification against the Ranz–Marshall single-sphere correlation. | d/h = 20 for a single sphere (secondary source: Thiam et al. 2019 snippet). |
| **Deen & Kuipers (2013)**, *Ind. Eng. Chem. Res.* 52, 11266–11274. [doi:10.1021/ie303411k](https://doi.org/10.1021/ie303411k). Abstract read. | The same implicit second-order BC, applied to species (Dirichlet: infinitely fast reaction). | Verified against "well-known empirical expressions for the Sherwood number". | — |
| **Deen & Kuipers (2014)**, *Chem. Eng. Sci.* 116, 645–656. [doi:10.1016/j.ces.2014.05.036](https://doi.org/10.1016/j.ces.2014.05.036). Metadata only. | Coupled heat + mass (infinitely fast surface reaction). Lu et al. 2018 describe it as using the "directional quadratic interpolation scheme". | — | — Proposed a modified Gunn fit; the coefficients are **UNVERIFIED**. |
| **Deen, Peters, Padding, Kuipers (2014)**, review, *Chem. Eng. Sci.* 116, 710–724. [doi:10.1016/j.ces.2014.05.039](https://doi.org/10.1016/j.ces.2014.05.039) | Review of DNS of mass, momentum and heat transfer in dense gas–solid flows. | — | — |
| **Lu, Das, Peters, Kuipers (2018)**, *Chem. Eng. Sci.* 176, 1–18. [doi:10.1016/j.ces.2017.10.018](https://doi.org/10.1016/j.ces.2017.10.018), CC-BY. Abstract read. | **Ghost-cell** IBM with **second-order quadratic** reconstruction. Handles **mixed / Robin** BCs (surface reaction vs diffusion) "at the exact position of the particle surface", incorporated implicitly. | Verified first in the limit of **unsteady molecular diffusion without convection**. Then a single sphere in flow and hundreds of spheres, swept over a range of Damköhler numbers. | — |
| **Lu, Tan, Peters, Kuipers (2018)**, *Ind. Eng. Chem. Res.* 57, 15565–15578. [doi:10.1021/acs.iecr.8b03158](https://doi.org/10.1021/acs.iecr.8b03158). **Read** (Europe PMC PMC6251562). | Ghost-cell with second-order quadratic interpolation. Robin BC for species. For heat, the particle temperature is a dynamic Dirichlet value taken from the particle energy balance. Exothermic 1st-order surface reaction. | Unsteady diffusion to a sphere vs a high-resolution 1-D "exact" solution. ΔT(particle − bulk) at 3 s, "exact" vs DNS: Da 0.01: 0.66 / 0.67; Da 0.1: 6.19 / 6.25; Da 1: 36.89 / 37.75; Da 10: 72.55 / 73.78; Da 100: 80.21 / 81.16 (error 1–2.3 %). | Grid size 2.5e-4 m for d = 5 mm, i.e. **N = d/h = 20**. In the dense array, N = 20 gives **4.14 %** deviation in the total heat-transfer rate (mesh test on a sub-array). |
| **Lu, Peters, Kuipers (2019)**, *Chem. Eng. Sci.* 204, 203–219. [doi:10.1016/j.ces.2019.02.043](https://doi.org/10.1016/j.ces.2019.02.043). Metadata only. | Dependently coupled heat and mass transfer. | — | — |
| **Lu, Peters, Kuipers (2020)**, *AIChE J.* 66 (online 2019). [doi:10.1002/aic.16786](https://doi.org/10.1002/aic.16786). Metadata only. | Mass transfer in bidisperse sphere arrays. | — | — |
| **Das, Deen, Kuipers (2017)**, *Chem. Eng. Sci.* 160, 1–19. [doi:10.1016/j.ces.2016.11.008](https://doi.org/10.1016/j.ces.2016.11.008). **Read** (TU/e repository). | Second-order sharp implicit IBM. Particles **and** the cylindrical column wall are both immersed. Conjugate wall heat transfer. | **Graetz check** in an immersed circular tube (Re_D = 20, Pr = 10, 10/20/40/60 cells per D): converges to f_D·Re = 64 and **Nu = 3.6568**. | **Key finding for dense packings with contacts**, grid study on a bed section (Table 6). **Re = 50:** Nu error vs extrapolated value is 28.4 % at G20, 7.4 % at G40, **1.0 % at G80**. **Re = 500:** Nu error is ≈23 % at G20, 11 % at G40, 5.6 % at G80 (computed from the table values 16.256 / 14.726 / 13.973 vs 13.228). Production runs used G40. "For creeping flow G20 is sufficient." |
| **Das, Panda, Deen, Kuipers (2018)**, *Chem. Eng. Sci.* 191, 1–18. [doi:10.1016/j.ces.2018.04.061](https://doi.org/10.1016/j.ces.2018.04.061). **Read** (TU/e repository). | Sharp-interface IBM on a staggered FV grid; geometry is an STL surface; periodic BCs. **Dirichlet:** the ghost value comes from a quadratic fit S–C–X_p along the grid line: φ_Xn = a φ_C + b φ_Xp + c, with a = −2n_S/(1−n_S), b = n_S/(2−n_S), c = 2φ_S/((1−n_S)(2−n_S)), where n_S is the wall distance in cell units. It switches to linear when n_S ≥ 0.999; the coefficients are folded into the matrix implicitly. **Neumann:** a normal probe plus trilinear interpolation, with the probe distance increased until every stencil cell is fluid. **Conjugate:** solid energy equation solved too, with continuity of T and of the normal flux. The surface temperature lags by one time step. | **2nd-order global** convergence for Dirichlet, Neumann and CHT. Error is measured against the finest grid (d/Δx up to 160). Momentum and energy *balance* errors converge only 1st order for Dirichlet, but 2nd order for Neumann. | **Periodic cylinder array** (pitch 3d, ε = 0.913, Pr = 1; Re ≈ 50, garbled in the extracted text): f = 3.238, **Nu_CWT = 2.542**, **Nu_CWF = 3.763**. **10 cells/d gives Nu within 1 % (CWT) and 2 % (CWF)**; drag is within 3 %. CHT model porous medium: accuracy decreases as k_s/k_f grows (10 → 1000), but stays 2nd order. Transient CHT sphere vs a 1-D 2000-point reference: d/Δx = 10 is acceptable even at k ratio 1000, provided Δt ≲ 10·Δx²/(2α). |
| **Buist, Backx, Deen, Kuipers (2017)**, *Chem. Eng. Sci.* 169, 310–320. [doi:10.1016/j.ces.2016.04.022](https://doi.org/10.1016/j.ces.2016.04.022). **Read** (TU/e repository). | **Direct-forcing** IBM (Lagrangian markers, heat source q) for a Dirichlet sphere, as in Tavassoli 2013. Compared with experiments on BCC arrays. | Single sphere: 20 cells/d is "nearly grid independent; deviation of 2 %". DNS is 7–20 % off Ranz–Marshall. | Dense arrays: 20 cells/d, but 40 at Re = 600. They suspect resolution underestimates Nu at solids fraction 0.5. Notes that Tavassoli 2013 found a **structural underestimation of Gunn**. |
| **Tavassoli, Kriebitzsch, van der Hoef, Peters, Kuipers (2013)**, *Int. J. Multiphase Flow* 57, 29–37. [doi:10.1016/j.ijmultiphaseflow.2013.06.009](https://doi.org/10.1016/j.ijmultiphaseflow.2013.06.009). Metadata, plus the description in Buist 2017. | Direct-forcing IBM extended to heat. Lagrangian surface markers; the heat source enforces Dirichlet T. | Nu for random arrays is in "reasonable agreement" with correlations, but systematically below Gunn (per Buist 2017). | **UNVERIFIED** |
| **Tavassoli, Peters, Kuipers (2015)**, *Chem. Eng. Sci.* 129, 42–48. [doi:10.1016/j.ces.2015.02.024](https://doi.org/10.1016/j.ces.2015.02.024) | The same IBM applied to fixed random arrays of **non-spherical** particles (spherocylinders). | — | — |
| **Tavassoli, Peters, Kuipers (2017)**, *Powder Technol.* 314, 291–298. [doi:10.1016/j.powtec.2016.09.088](https://doi.org/10.1016/j.powtec.2016.09.088) | Bidisperse arrays, Re 30–100. Monodisperse correlations work if Re and Nu are based on the Sauter diameter. | — | — |
| **Claassen, Baltussen, Peters, Kuipers (2024)**, *Chem. Eng. Sci.* 291, 119936. [doi:10.1016/j.ces.2024.119936](https://doi.org/10.1016/j.ces.2024.119936), CC-BY. Abstract read. | Ghost-cell IBM for **conjugate** heat/mass in packed beds. Compares a traditional IBM with two new variants: IB surface values, and an **effective flux**. | The traditional IBM is fine only when fluid and solid properties are comparable. It **fails near contact points at the usual 20 cells per radius**. Both new variants are accurate for conduction and for Re 10–200. | **Directly relevant to peclet's conjugate design: contact-point treatment, not bulk resolution, is the bottleneck.** |

### 1b. Other PR-DNS heat-transfer codes

- **Tenneti, Sun, Garg, Subramaniam (2013)**, *Int. J. Heat Mass Transfer* 58, 471–479.
  [doi:10.1016/j.ijheatmasstransfer.2012.11.006](https://doi.org/10.1016/j.ijheatmasstransfer.2012.11.006).
  PUReIBM uses direct forcing on a regular grid with isothermal particles. The temperature solver is pseudo-spectral.
  A **thermal self-similarity condition** (scaled temperature θ) allows thermally fully developed heat transfer in
  **periodic boxes**. The fluid heats up along the bed, so a plain periodic T is wrong. **peclet needs the same device
  for periodic Nu.** The duct verification is a square duct, with **Nu = 2.976** from Shah & London. PUReIBM gives
  3.013 / 3.029 / 3.033 at Re 20 / 50 / 100 (Sun PhD thesis, Iowa State; read).
- **Sun, Tenneti, Subramaniam (2015)**, *Int. J. Heat Mass Transfer* 86, 898–913.
  [doi:10.1016/j.ijheatmasstransfer.2015.03.046](https://doi.org/10.1016/j.ijheatmasstransfer.2015.03.046). The
  correlation (formula confirmed in Sun's thesis, Eq. at line 4367) is
  **Nu = (−0.46 + 1.77ε + 0.69ε²)/ε³ + (1.37 − 2.4ε + 1.2ε²) Re^0.7 Pr^(1/3)**, with ε the fluid fraction.
  Resolution: at εs = 0.4, Re = 20, Nu changes by **11 % between Dm = d/Δx = 20 and 70** (asymptote 7.8). The
  error model gives a worst-case 18.5 % at Dm = 20, of which 12 % is discretization error. 5 realizations give a
  95 % confidence interval of about 15 %.
- **Tenneti & Subramaniam (2014)**, *Annu. Rev. Fluid Mech.* 46, 199–230.
  [doi:10.1146/annurev-fluid-010313-141344](https://doi.org/10.1146/annurev-fluid-010313-141344). Review.
- **Thiam, Masi, Climent, Simonin, Vincent (2019)**, *Acta Mech.* 230, 541–567.
  [doi:10.1007/s00707-018-2346-5](https://doi.org/10.1007/s00707-018-2346-5). Lagrangian VOF / fictitious-domain
  penalty method on random fixed arrays. Compares Nu based on fluid-mean temperature with Nu based on the bulk
  (cup-mixing) temperature. Method description is from a search snippet: **UNVERIFIED**.
- **Municchi & Radl (2017)**, *Int. J. Heat Mass Transfer* 111, 171–190.
  [doi:10.1016/j.ijheatmasstransfer.2017.03.122](https://doi.org/10.1016/j.ijheatmasstransfer.2017.03.122). PR-DNS
  closures for bidisperse suspensions. Also **Municchi & Radl (2018)**, *Int. J. Heat Mass Transfer* 120, 1146–1161,
  [doi:10.1016/j.ijheatmasstransfer.2017.12.105](https://doi.org/10.1016/j.ijheatmasstransfer.2017.12.105), on bounded
  dense systems. The mesh/BC method is **UNVERIFIED**; I believe it is body-fitted OpenFOAM but did not confirm it.
- **Kravets & Kruggel-Emden (2017)**, *Powder Technol.* 318, 293–305.
  [doi:10.1016/j.powtec.2017.05.039](https://doi.org/10.1016/j.powtec.2017.05.039). Fully resolved **LBM**, local
  heat transfer in random packings. The BC details are **UNVERIFIED**.
- **Chadil, Vincent, Estivalèzes (2021)**, *Fluids* 7, 15. [doi:10.3390/fluids7010015](https://doi.org/10.3390/fluids7010015).
  Open access; it reviews PR-DNS heat-transfer computation. I did not read it.

### 1c. Body-fitted packed-bed CFD (Wehinger / Dixon / Jurtz): conjugate, reacting, shaped particles

- **Jurtz, Kraume, Wehinger (2019)**, "Advances in fixed-bed reactor modeling using particle-resolved CFD", *Rev.
  Chem. Eng.* 35, 139–190. [doi:10.1515/revce-2017-0059](https://doi.org/10.1515/revce-2017-0059). Abstract read; the
  full text could not be fetched. Covers packing generation, meshing, contacts, microkinetics coupling, dispersion,
  heat and mass transfer.
- **Dixon & Partopour (2020)**, *Annu. Rev. Chem. Biomol. Eng.* 11, 109–130.
  [doi:10.1146/annurev-chembioeng-092319-075328](https://doi.org/10.1146/annurev-chembioeng-092319-075328). Abstract
  read. Meshing of non-sphere packings is an open problem.
- **Dixon, Nijemeisland, Stitt (2013)**, *Comput. Chem. Eng.* 48, 135–153.
  [doi:10.1016/j.compchemeng.2012.08.011](https://doi.org/10.1016/j.compchemeng.2012.08.011). Contact-point study
  (caps / bridges / gaps). A search snippet says Dixon et al. derived an effective bridge conductivity, and that caps
  and bridges differ little at high flow where convection dominates. The bridge size and the numbers are
  **UNVERIFIED**.
- **Dixon et al. (2011)**, single-sphere mesh study, *Comput. Chem. Eng.* 35, 1171–1185.
  [doi:10.1016/j.compchemeng.2010.12.006](https://doi.org/10.1016/j.compchemeng.2010.12.006)
- **Dixon & Nijemeisland (2001)**, *Ind. Eng. Chem. Res.* 40, 5246–5254.
  [doi:10.1021/ie001035a](https://doi.org/10.1021/ie001035a)
- **Wehinger, Fütterer, Kraume (2017)**, contact modifications for **cylinders**, *Ind. Eng. Chem. Res.* 56, 87–99.
  [doi:10.1021/acs.iecr.6b03596](https://doi.org/10.1021/acs.iecr.6b03596)
- **Wehinger, Eppinger, Kraume (2015)**, *Chem. Eng. Sci.* 122, 197–209.
  [doi:10.1016/j.ces.2014.09.007](https://doi.org/10.1016/j.ces.2014.09.007). Dry reforming; particle-resolved CFD.
- **Wehinger, Kraume et al. (2016)**, *AIChE J.* 62, 4436–4452. [doi:10.1002/aic.15520](https://doi.org/10.1002/aic.15520).
  Conjugate heat transfer, surface-to-surface radiation and microkinetics vs spatial reactor profiles. Excellent
  agreement for the heat-transfer-only case (search snippet).
- **Wehinger, Klippel, Kraume (2017)**, pore processes, *Comput. Chem. Eng.* 101, 11–22.
  [doi:10.1016/j.compchemeng.2017.02.029](https://doi.org/10.1016/j.compchemeng.2017.02.029)
- **Partopour & Dixon (2016)**, microkinetics in resolved-particle CFD, *Comput. Chem. Eng.* 88, 126–134.
  [doi:10.1016/j.compchemeng.2016.02.015](https://doi.org/10.1016/j.compchemeng.2016.02.015). Also **Partopour & Dixon
  (2019)**, *Chem. Eng. J.* 377, 119738. [doi:10.1016/j.cej.2018.08.124](https://doi.org/10.1016/j.cej.2018.08.124).

### 1d. Engineering correlations (reference targets, not gates)

- **Gunn (1978)**, *Int. J. Heat Mass Transfer* 21, 467–476.
  [doi:10.1016/0017-9310(78)90080-7](https://doi.org/10.1016/0017-9310(78)90080-7). Formula as quoted in Buist 2017
  (read):
  **Nu = (7 − 10ε + 5ε²)(1 + 0.7 Re^0.2 Pr^(1/3)) + (1.33 − 2.4ε + 1.2ε²) Re^0.7 Pr^(1/3)**, valid for 0.35 < ε < 1.
- **Wakao & Funazkri (1978)**, *Chem. Eng. Sci.* 33, 1375–1384.
  [doi:10.1016/0009-2509(78)85120-3](https://doi.org/10.1016/0009-2509(78)85120-3). Heat analogue: **Wakao, Kaguei,
  Funazkri (1979)**, *Chem. Eng. Sci.* 34, 325–336.
  [doi:10.1016/0009-2509(79)85064-2](https://doi.org/10.1016/0009-2509(79)85064-2). The usual form is
  Sh = 2 + 1.1 Sc^(1/3) Re^0.6 (3 < Re < 10⁴). One snippet gave the mass form without the "2 +", so the exact form
  is **UNVERIFIED**.
- **Ranz–Marshall:** Nu = 2 + 0.6 Re^0.5 Pr^(1/3) (as quoted in Buist 2017). The original (Chem. Eng. Prog. 1952)
  citation is **UNVERIFIED** (no DOI).
- **Whitaker (1972)**, *AIChE J.* 18, 361–371. [doi:10.1002/aic.690180219](https://doi.org/10.1002/aic.690180219)
- **Richter & Nikrityuk (2012)**, non-spherical particles (cube, ellipsoid), Nu and drag for Re 10–250. *Int. J. Heat
  Mass Transfer* 55, 1343–1354. [doi:10.1016/j.ijheatmasstransfer.2011.09.005](https://doi.org/10.1016/j.ijheatmasstransfer.2011.09.005)

**Resolution synthesis (item 1).** To get Nu within 1–2 %, the requirement depends on the geometry:

- **Isolated particles or dilute arrays:** a 2nd-order sharp, implicit method needs only **~10–20 cells/d**.
- **Random arrays at εs ≈ 0.4:** direct-forcing PUReIBM was still 11 % off at 20 cells/d.
- **Dense packings with contacts at Re ≥ 50:** a 2nd-order sharp IBM needed **~80 cells/d** (Das 2017). This is
  because the near-contact gap boundary layers dominate.
- **Creeping flow:** about 20 cells/d suffices.
- **Conjugate transport at contacts:** this fails at 40 cells/d unless the contact flux is treated specially
  (Claassen 2024).

---

## 2. Analytic and high-accuracy benchmarks for scalar boundary conditions

Notation: a = radius, d = 2a, Pe_d = U d/D, Pe_a = U a/D, Nu = Nu_d = h d/k unless stated otherwise.

**(B1) Isolated sphere, pure conduction or diffusion (Dirichlet).** T = T∞ + (T_s − T∞) a/r, so **Nu_d = 2**
exactly. This needs an unbounded domain. In a box, impose the exact solution on the outer boundary. This is the
classical result in every textbook.

**(B2) Robin sphere, steady, first-order surface reaction** −D ∂c/∂n = k_r c. Define Da = k_r a/D. The solution is
c = c∞(1 − A a/r) with A = Da/(1 + Da), so **Sh_d = 2 Da/(1 + Da)**, the overall rate referred to c∞. This is a
two-line derivation, not a literature value. Lu et al. 2018 sweep Da the same way. The **transient**
coupled heat + mass version, with the Lu-IECR table values quoted in §1a, is a published code-to-1D reference.

**(B3) Robin / Biot transient sphere.** Interior conduction with external h. The eigenvalues satisfy
**λ_n cot λ_n = 1 − Bi**, with Bi = h a/k_s. This is the textbook result (Carslaw & Jaeger, *Conduction of Heat in
Solids*, Oxford 1959; book, no DOI). It tests Robin coupled to the interior solve.

**(B4) Conjugate sphere in a uniform far-field gradient G** (Maxwell's problem, conductivity ratio κ = k_s/k_f):

- inside: T = **3/(κ + 2)** G·x;
- outside: T = G·x [1 − (κ − 1)/(κ + 2) (a/r)³].

This is exact for all κ (textbook electrostatics and heat conduction). It tests continuity of T and of k∂T/∂n. With
the exact solution imposed on the box boundary there is no truncation error. **Ideal conjugate gate; sweep
κ = 10⁻², 1, 10, 10³.**

**(B5) Concentric spheres or cylinders, conjugate or Neumann.**

- Spheres: series resistances, T(r) = A + B/r in each shell, with T and k∂T/∂r continuous. An inner flux q with an
  outer Dirichlet wall gives T(r) = T_o + (q a²/k)(1/r − 1/R).
- Cylinders: T(r) uses ln r.

These are exact (textbook). They test Neumann, and CHT with a curved interface.

**(B6) Effective conductivity of periodic arrays (conjugate, finite contrast).**

- **Sangani & Acrivos (1983)**, *Proc. R. Soc. Lond. A* 386, 263–275.
  [doi:10.1098/rspa.1983.0036](https://doi.org/10.1098/rspa.1983.0036). Abstract read. Gives k* to O(c⁹) for SC, BCC
  and FCC, and numerical values "over the whole range of α and c".
- For **perfectly conducting** spheres (α → ∞), the expansion as reproduced by Andrianov & Topol (arXiv:1106.1783,
  Eq. 157; read) is:

  **k* = 1 − 3c / ( −1 + c + a₁c^(10/3)(1 + a₂c^(11/3))/(1 − a₃c^(7/3)) + a₄c^(14/3) + a₅c⁶ + a₆c^(22/3) + O(c^(25/3)) )**

  | Array | a₁ | a₂ | a₃ | a₄ | a₅ | a₆ |
  |---|---|---|---|---|---|---|
  | SC | 1.305 | 0.231 | 0.405 | 0.0723 | 0.153 | 0.0105 |
  | BCC | 0.129 | −0.413 | 0.764 | 0.257 | 0.0113 | 0.00562 |
  | FCC | 0.0753 | 0.697 | "−07.41" as printed (typo?) | 0.0420 | 0.0231 | 9.14·10⁻⁷ |

  My own evaluation for SC at c = 0.3 gives k* ≈ 2.333; Maxwell–Garnett gives 2.286.
- **For finite κ**, the leading Rayleigh form is k* = 1 − 3c/(−1/β + c − …) with β = (κ − 1)/(κ + 2). That reduces
  to Maxwell–Garnett (1 + 2βc)/(1 − βc) at O(c). The octupole term ≈ 1.305 β′c^(10/3) with β′ = (κ − 1)/(κ + 4/3)
  is **UNVERIFIED** (recalled). Take finite-κ numbers from the SA 1983 tables or from McPhedran & McKenzie.
- **McPhedran & McKenzie (1978)**, SC lattice, *Proc. R. Soc. Lond. A* 359, 45–63.
  [doi:10.1098/rspa.1978.0031](https://doi.org/10.1098/rspa.1978.0031). Explicit formula with poles up to order 27;
  experiments on lossy spheres.
- **McKenzie, McPhedran, Derrick (1978)**, BCC/FCC, *Proc. R. Soc. Lond. A* 362, 211–232.
  [doi:10.1098/rspa.1978.0129](https://doi.org/10.1098/rspa.1978.0129)
- **Zuzovsky & Brenner (1977)**, *ZAMP* 28, 979–992. [doi:10.1007/bf01601666](https://doi.org/10.1007/bf01601666)
- **2-D cylinders: Perrins, McKenzie, McPhedran (1979)**, *Proc. R. Soc. Lond. A* 369, 207–225.
  [doi:10.1098/rspa.1979.0160](https://doi.org/10.1098/rspa.1979.0160), square and hexagonal arrays. The square-array
  formula k* = 1 − 2f/(T + f − 0.305827 f⁴T/(T² − 1.402958 f⁸) − 0.013362 f⁸), with T = (1 + κ)/(1 − κ), is
  **UNVERIFIED** in its exact arrangement. The coefficients were confirmed to exist in the literature via a search
  hit. Check against the paper, or compute it with a multipole code, before using it as a gate.
- **Godin (2012)**, arXiv:1201.1419, *J. Math. Phys.* 53, 063703. Arbitrary lattices of circular inclusions.
- **Bounds and limits:**
  - **Hashin & Shtrikman (1962)**, *J. Appl. Phys.* 33, 3125–3131. [doi:10.1063/1.1728579](https://doi.org/10.1063/1.1728579)
  - Near-touching spheres, perfect conductors: k* ~ −K₁ ln(1 − χ). **Batchelor & O'Brien (1977)**, *Proc. R. Soc. Lond.
    A* 355, 313–333. [doi:10.1098/rspa.1977.0100](https://doi.org/10.1098/rspa.1977.0100). The contact-region
    asymptotics are relevant to packed-bed CHT.
  - Packed-bed stagnant conductivity: **Zehner & Schlünder (1970)**, *Chem. Ing. Tech.* 42, 933–941.
    [doi:10.1002/cite.330421408](https://doi.org/10.1002/cite.330421408)
- Das 2018 (read) used a 2-D CHT model porous medium, with porosity 0.196 and k_s/k_f = 10/100/1000. They report
  k_eff = 7.23 / 67.70 / 672.12 against arithmetic and harmonic bounds. This is a code-to-code value only; finest-grid
  extrapolation.

**(B7) Sphere in Stokes flow, Nu(Pe), Re → 0** (Dirichlet + advection–diffusion boundary layer):

- **Acrivos & Taylor (1962)**, *Phys. Fluids* 5, 387–394. [doi:10.1063/1.1706630](https://doi.org/10.1063/1.1706630).
  Low Pe: Nu_d = 2 + Pe_d/2 + O(Pe² ln Pe). The 2 + Pe_d/2 terms are unambiguous. The higher terms quoted by El
  Hasadi & Padding (arXiv:2007.10214) are 2 + Pe/2 + ¼Pe² ln Pe + 0.034Pe² + (1/16)Pe³ ln Pe. Whether the log
  argument is the radius- or diameter-based Pe is **UNVERIFIED**, so use only the first two terms.
- **Acrivos & Goddard (1965)**, *J. Fluid Mech.* 23, 273–291.
  [doi:10.1017/s0022112065001350](https://doi.org/10.1017/s0022112065001350). High Pe, with Pe and Nu radius-based:
  Nu_a = Pe_a^(1/3)[0.6245 + 0.461 Pe_a^(−1/3)] (search-confirmed). Equivalently
  **Nu_d = 0.922 + 0.991 Pe_d^(1/3)**.
- Rimmer (1968), next term: *J. Fluid Mech.* 32, 1–7. [doi:10.1017/s0022112068000546](https://doi.org/10.1017/s0022112068000546)
- All-Pe creeping-flow fit (Clift, Grace & Weber, *Bubbles, Drops and Particles*, Academic Press 1978; book):
  **Sh = 1 + (1 + Pe_d)^(1/3)**.
- Finite Re: Feng & Michaelides (2000), *Int. J. Heat Mass Transfer* 43, 219–229.
  [doi:10.1016/s0017-9310(99)00133-7](https://doi.org/10.1016/s0017-9310(99)00133-7). Tabulated numerics. The table
  values are **UNVERIFIED**.

**(B8) Graetz / fully developed duct convection** (immersed SDF tube; Dirichlet or Neumann on a curved wall):

| Duct | Nu_T (Dirichlet) | Nu_H (uniform flux) |
|---|---|---|
| Circular tube | **3.6568** (used and recovered by Das 2017) | **48/11 = 4.3636** |
| Parallel plates (D_h = 2·gap) | 7.541 | 140/17 = 8.235 |
| Square duct | **2.976** (PUReIBM gate, Sun thesis) | 3.608 (**UNVERIFIED**) |

Source: Shah & London, *Laminar Flow Forced Convection in Ducts*, Academic Press 1978 (book).

**(B9) Taylor–Aris dispersion** (no-flux Neumann wall + advection + long-time moments):

- Tube: **K = D(1 + Pe_a²/48)**, with Pe_a = Ū a/D and Ū the mean velocity.
- Plates: **K = D(1 + (2/105) Pe_h²)**, with h the half-gap.
- Valid for t ≫ a²/D.
- **Taylor (1953)**, *Proc. R. Soc. Lond. A* 219, 186–203. [doi:10.1098/rspa.1953.0139](https://doi.org/10.1098/rspa.1953.0139)
- **Aris (1956)**, *Proc. R. Soc. Lond. A* 235, 67–77. [doi:10.1098/rspa.1956.0065](https://doi.org/10.1098/rspa.1956.0065)
- Rectangular channels: **Chatwin & Sullivan (1982)**, *J. Fluid Mech.* 120, 347–358.
  [doi:10.1017/s0022112082002791](https://doi.org/10.1017/s0022112082002791). The square-duct coefficient is
  **UNVERIFIED**. Mostaghimi et al. 2012 used the square capillary as their validation case.

**(B10) Periodic cylinder array, CWT and CWF** (Das 2018, code-to-code, 2-D, cheap): see §1a. The quoted values are
f = 3.238, Nu_CWT = 2.542 and Nu_CWF = 3.763. Use the log-mean ΔT definition with the periodic thermal treatment.

---

## 3. Dispersion in periodic arrays and packed beds (validation data)

- **Brenner (1980)**, *Phil. Trans. R. Soc. A* 297, 81–133. [doi:10.1098/rsta.1980.0205](https://doi.org/10.1098/rsta.1980.0205)
  Macrotransport theory: the B-field closure gives D* in periodic media. **Brenner & Adler (1982)**, Part II
  (surface and intraparticle transport): [doi:10.1098/rsta.1982.0108](https://doi.org/10.1098/rsta.1982.0108).
  This is the right formalism for peclet's periodic boxes: solve the periodic B-equation, or the moments.
- **Salles, Thovert, Delannay, Prevors, Auriault, Adler (1993)**, *Phys. Fluids A* 5, 2348–2376.
  [doi:10.1063/1.858751](https://doi.org/10.1063/1.858751). Abstract read. D*(Pe) by the B-equation **and** by random
  walks, for deterministic (cubic), random and reconstructed unit cells, with systematic comparison to data. **Best
  code-to-code dispersion reference for periodic arrays.**
- **Edwards, Shapiro, Brenner, Shapira (1991)**, *Transp. Porous Media* 6.
  [doi:10.1007/bf00136346](https://doi.org/10.1007/bf00136346). Dispersion in 2-D spatially periodic cylinder arrays.
- **Eidsath, Carbonell, Whitaker, Herrmann (1983)**, *Chem. Eng. Sci.* 38, 1803–1816.
  [doi:10.1016/0009-2509(83)85037-4](https://doi.org/10.1016/0009-2509(83)85037-4). Volume-averaging closure solved
  in periodic unit cells and compared with packed-bed experiments. The content is from my recollection; the abstract
  was unavailable, so this is **partly UNVERIFIED**.
- **Koch & Brady (1985)**, *J. Fluid Mech.* 154, 399–427. [doi:10.1017/s0022112085001598](https://doi.org/10.1017/s0022112085001598).
  Abstract read; Pe = U a/D_f. The mechanisms and their high-Pe scalings are:
  - mechanical dispersion ∝ Pe;
  - boundary-layer dispersion ∝ Pe ln Pe;
  - closed-streamline / holdup dispersion ∝ Pe².

  Results are tabulated in their §6, with good agreement with dense-bed experiments. **A grid-resolved code
  must capture the Pe ln Pe term.** It comes from the no-slip diffusive boundary layer and is a resolution-sensitive
  diagnostic.
- **Maier, Kroll, Kutsovsky, Davis, Bernard (1998)**, LBM flow in bead packs, *Phys. Fluids* 10, 60–74.
  [doi:10.1063/1.869550](https://doi.org/10.1063/1.869550). The velocity PDF is resolution-sensitive and converges to
  a sharp peak near zero.
- **Maier, Kroll, Bernard, Howington, Peters, Davis (2000)**, "Pore-scale simulation of dispersion", *Phys. Fluids*
  12, 2065–2079. [doi:10.1063/1.870452](https://doi.org/10.1063/1.870452). LBM + random-walk particle tracking in
  regular **and** random sphere packings. Asymptotic D_L and D_T vs Pe agree with NMR. Longitudinal dispersion is
  lower than older experimental literature suggests. NMR comparison:
  [doi:10.1016/s0730-725x(01)00331-9](https://doi.org/10.1016/s0730-725x(01)00331-9). Diameter-dependent dispersion in
  cylindrical packs: Vandre et al. (2008), *AIChE J.* 54, 2024–2028.
  [doi:10.1002/aic.11529](https://doi.org/10.1002/aic.11529)
- **Mostaghimi, Bijeljic, Blunt (2012)**, *SPE J.* 17, 1131–1141. [doi:10.2118/135261-pa](https://doi.org/10.2118/135261-pa).
  Finite-difference Stokes solve plus AMG on micro-CT voxels. Transport uses a **Pollock-type streamline tracer with
  a semi-analytic near-wall velocity treatment** (sub-grid no-slip) plus a random walk. Validated on Taylor–Aris in a
  square capillary; predicts the D_L data for Pe = 10⁻² to 10⁵.
- **Pollock (1988)**, *Groundwater* 26, 743–750. [doi:10.1111/j.1745-6584.1988.tb00425.x](https://doi.org/10.1111/j.1745-6584.1988.tb00425.x)
- **Bijeljic, Muggeridge, Blunt (2004)**, *Water Resour. Res.* 40. [doi:10.1029/2004wr003567](https://doi.org/10.1029/2004wr003567).
  Network model. D_L ~ Pe^1.19 in the transition, ∝ Pe for Pe > 400; first advective effects at Pe ~ 0.1. Transverse:
  **Bijeljic & Blunt (2007)**, *Water Resour. Res.* 43. [doi:10.1029/2006wr005700](https://doi.org/10.1029/2006wr005700)
- **Delgado (2006)**, "A critical review of dispersion in packed beds", *Heat Mass Transfer* 42, 279–310 (online 2005).
  [doi:10.1007/s00231-005-0019-0](https://doi.org/10.1007/s00231-005-0019-0). Also **Delgado (2007)**, *Chem. Eng. Res.
  Des.* 85, 1245–1252. [doi:10.1205/cherd07017](https://doi.org/10.1205/cherd07017). These are data compilations and
  correlations. The rule-of-thumb asymptotes Pe_L = u d/D_L ≈ 2 and Pe_T ≈ 10 for liquids and gases at high Re are
  my recollection: **UNVERIFIED** against these papers.
- **Gunn (1987)**, axial and radial dispersion in fixed beds, *Chem. Eng. Sci.* 42, 363–373.
  [doi:10.1016/0009-2509(87)85066-2](https://doi.org/10.1016/0009-2509(87)85066-2)
- **Claassen, Fathiganjehlou, Peters, Buist, Baltussen, Kuipers (2025)**, *Int. J. Heat Mass Transfer* 240, 126630.
  [doi:10.1016/j.ijheatmasstransfer.2024.126630](https://doi.org/10.1016/j.ijheatmasstransfer.2024.126630).
  Mechanical dispersion in slender beds (D/d = 4.2, 5.25, 7) by PR-CFD vs a pore-network model. The pore-network
  model underpredicts because it misses Taylor and boundary-layer dispersion.
- **Eghbalmanesh et al. (2024)**, *AIChE J.* [doi:10.1002/aic.18322](https://doi.org/10.1002/aic.18322). PR-CFD vs MRI
  velocity in a reconstructed slender bed; open access. A good flow-field gate before dispersion.

---

## 4. Recommended verification gates for peclet scalar transport

Order: cheapest and most diagnostic first. Each gate needs a **grid-convergence order** (target 2, or ≥ 1.5 for
fluxes) **plus** an absolute tolerance at the production resolution.

| # | Name | Geometry / condition | Reference value (source) | Tests |
|---|---|---|---|---|
| G1 | `sphere_conduction_dirichlet` | Isolated sphere, T_s fixed. Box boundary set to the exact a/r field. | **Nu = Sh = 2**, exact field (§2 B1) | Dirichlet cut-cell closure, surface-flux integration, 2nd-order field and flux convergence vs d/h = 8…64 |
| G2 | `concentric_shells_neumann` | Sphere (and 2-D cylinder) with prescribed flux q, inside a Dirichlet outer shell | T(r) = T_o + (q a²/k)(1/r − 1/R) (B5) | Neumann on curved SDF surfaces; flux-conservation error |
| G3 | `sphere_robin_reaction` | Isolated sphere, −D∂c/∂n = k_r c; sweep Da = 0.01 … 100 | **Sh = 2Da/(1 + Da)** (B2, derived). Transient coupled heat+mass vs Lu et al. 2018 IECR table (ΔT at 3 s: 0.66, 6.19, 36.89, 72.55, 80.21 K) | Robin closure across the Dirichlet-like (Da → ∞) and Neumann-like (Da → 0) limits |
| G4 | `maxwell_conjugate_sphere` | Sphere with k_s/k_f = κ in a far-field gradient G; box boundary set to the exact solution; κ ∈ {10⁻², 1, 10, 10³} | **Interior gradient = 3G/(κ + 2)**, exterior dipole (B4) | Conjugate T and flux continuity, high-contrast robustness; κ = 1 must reproduce a uniform gradient exactly |
| G5 | `periodic_array_keff` | Fully periodic SC (and BCC) sphere array under a mean gradient; first κ → ∞ (perfect conductor), then finite κ | **SA 1983 expansion** (e.g. SC c = 0.3 perfect conductor ≈ 2.333 from Eq. 157); finite-κ values from SA 1983 / McPhedran tables; 2-D square cylinder array from Perrins 1979; Hashin–Shtrikman bounds as sanity | Conjugate transport in **periodic** boxes (peclet's native setting), multigrid with jump coefficients |
| G6 | `transient_conjugate_sphere` | Hot sphere cooling in a quiescent medium (κ = 10, 100, 1000); plus a lumped-Robin variant | 1-D spherical high-res reference (Das 2018 §5.4 method); Robin eigenvalues λ cot λ = 1 − Bi (B3) | Time accuracy of the interface coupling (Das found a lag error once Δt > 10 Δt_F) |
| G7 | `graetz_sdf_tube` | Circular (and square) duct immersed as an SDF, fully developed, periodic with the thermal similarity (scaled-T) condition | **Nu_T = 3.6568, Nu_H = 48/11 = 4.3636** (tube); **Nu_T = 2.976** (square) (B8) | Advection-diffusion + Dirichlet/Neumann on curved walls + the **periodic thermally-fully-developed formulation** needed for bed Nu |
| G8 | `taylor_aris` | Poiseuille in an SDF tube and between plates; tracer moments at t ≫ a²/D | **K/D = 1 + Pe_a²/48**; plates **1 + 2Pe_h²/105** (B9) | No-flux Neumann + advection; scheme diffusion (spurious D_num shows up directly) |
| G9 | `sphere_stokes_nu_pe` | Single sphere, Re ≪ 1, Pe_d = 0.1 … 10³ | Low Pe: 2 + Pe_d/2; high Pe: **0.922 + 0.991 Pe_d^(1/3)** (Acrivos & Goddard); bridge with Clift 1 + (1 + Pe)^(1/3) (B7) | Resolving thin concentration boundary layers; tells you d/h vs Pe |
| G10 | `periodic_cylinder_array_cwt_cwf` | 2-D cylinder in a periodic cell, pitch 3d, ε = 0.913, Pr = 1, Re ≈ 50 (confirm Re in Das 2018) | **Nu_CWT = 2.542, Nu_CWF = 3.763, f = 3.238**; ≤1 % / 2 % at 10 cells/d (Das 2018) | Code-to-code against the closest published method (directional quadratic sharp IBM) |
| G11 | `dense_bed_nu_resolution` | Random sphere bed section with contacts (εs ≈ 0.4–0.6), Re 50 and 500; d/h = 20, 40, 80 | Das 2017 Table 6 convergence pattern (Nu error at Re = 50: 28 % / 7.4 % / 1.0 % for G20/G40/G80); Sun 2015 and Gunn correlations as bands | **Contact-region resolution**: the honest cost of 1–2 % Nu in packed beds; ties into the Claassen 2024 contact fix |
| G12 | `periodic_array_dispersion` | Periodic SC array and a random packing, Stokes flow; D_L and D_T vs Pe by B-equation or moments | Salles et al. 1993 (periodic cells, random walk + B-eq.); Maier et al. 2000 (random packings, NMR-validated); Koch & Brady 1985 scalings (Pe, Pe ln Pe) | Dispersion and macrotransport: the end-to-end application gate |

**Notes for the designer.**

1. The closest published analogue to an SDF cut-cell code is the Eindhoven line:
   - Deen 2012 and Das 2018 use a directional quadratic extrapolation along grid lines with per-direction wall
     distances n_S, folded implicitly into the matrix.
   - Lu 2018 uses a ghost-cell quadratic with a Robin BC.
   - Claassen 2024 adds a conjugate contact-point fix.

   An SDF gives n_S per direction directly.
2. Balance errors: Das 2018 measured global energy-balance errors converging only 1st order for Dirichlet, despite a
   2nd-order Nu. A **conservative** cut-cell flux form would avoid this. Worth a gate assertion, since balance errors
   matter for packed-bed reactor yields.
3. Periodic Nu needs the Tenneti 2013 thermal similarity condition, or Das 2018's "temperature-periodic" condition.
   A plain periodic T decays to T_s.
4. For dispersion at high Pe, scheme diffusion competes with the Pe ln Pe boundary-layer mechanism. G8 is the
   cheapest place to quantify numerical diffusion.
