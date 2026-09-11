/// @file
/// @brief flow — IbmSolver transported scalars: addScalar/setScalarBc/advanceScalars and their BC
/// application.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_SCALARS_HPP
#define PECLET_FLOW_FLOW_IBM_SCALARS_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::addScalar(const std::string& name, double D, int scheme, int iters) {
  ScalarField sc;
  sc.name = name;
  sc.c = addField(name);  // registered, zero-initialised, on the G=2 block
  sc.cOld = CCField(name + "_old", n_);
  sc.b = CCField(name + "_b", n_);
  sc.AC = CCField(name + "_AC", n_);
  sc.AW = CCField(name + "_AW", n_);
  sc.AE = CCField(name + "_AE", n_);
  sc.AS = CCField(name + "_AS", n_);
  sc.AN = CCField(name + "_AN", n_);
  sc.AB = CCField(name + "_AB", n_);
  sc.AT = CCField(name + "_AT", n_);
  sc.D = D;
  sc.scheme = scheme;
  sc.iters = iters < 1 ? 1 : iters;
  scalars_.push_back(sc);
}

template <class Grid>
bool Solver<Grid>::hasScalar(const std::string& name) const {
  for (const auto& sc : scalars_)
    if (sc.name == name)
      return true;
  return false;
}

template <class Grid>
void Solver<Grid>::setScalarBc(const std::string& name, int face, int type, double value) {
  for (auto& sc : scalars_)
    if (sc.name == name) {
      sc.bc[face] = type;
      sc.bcVal[face] = value;
      return;
    }
  throw std::runtime_error("set_scalar_bc: no scalar named '" + name + "'");
}

template <class Grid>
void Solver<Grid>::advanceScalars() {
  if (scalars_.empty())
    return;
  const double idt = 1.0 / dt_;
  CCField Uf, Vf, Wf;
  if constexpr (Grid::collocated) {
    Uf = uf_;
    Vf = vf_;
    Wf = wf_;
  } else {
    Uf = C[0].u;
    Vf = C[1].u;
    Wf = C[2].u;
  }
  fillGhosts(Uf);
  fillGhosts(Vf);
  fillGhosts(Wf);  // face velocities need the ±2 advection reach
  // Phase 3 (V5.4): the cell metric the scalar/energy GFM rows pull back through. The per-axis
  // Laplacian weights are `u_.w[a]` — Phase 2's, passed at each call site in its own spelling.
  const vof::VofMetric gmS = u_.vofMetric();
  for (auto& sc : scalars_) {
    // WO-P23: the CONSISTENT-ENERGY branch — per-cell k(C) in the bands, a rho c_p(C) time term,
    // and NO advective term (the transport was already done geometrically with the colour's own
    // fluxes in advectVof). Taken OUTSIDE the kernels, so a scalar without `energy` executes the
    // validated constant-D bodies verbatim.
    const bool energy = sc.energy && sc.kcell.extent(0) == n_;
    const bool hasMask = sc.dmask.extent(0) == n_;
    if (energy) {
      scalarBuildDiffusionVarK(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(ox_),
                               CCConst(oy_), CCConst(oz_), CCConst(sc.kcell), CCConst(sc.rcp),
                               CCConst(sc.dmask), idt, e_, G, u_.w[0], u_.w[1], u_.w[2]);
      applyScalarBcStencilVar(sc);
    } else {
      scalarBuildDiffusionOpen(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(ox_),
                               CCConst(oy_), CCConst(oz_), sc.D, idt, e_, G, u_.w[0], u_.w[1],
                               u_.w[2]);
      applyScalarBcStencil(sc);  // re-open Dirichlet domain faces (set_domain_bc closes openness)
    }
    // WO-P23: the PLANE-ANCHORED (ghost-fluid) form of that Dirichlet set — the pure cells'
    // rows carry the condition at the PLIC plane and never read the interfacial cell's value.
    const bool gfm = hasMask && pcPlaneDir_ && sc.gfmB.extent(0) == n_ && pcGphi_.extent(0) == n_;
    // WO-P3g items 2 + 3: the second-order (Gibou-Fedkiw) row and/or the curvature-consistent
    // `theta`. The branch is OUTSIDE the kernel, so the shipped configuration executes the
    // validated `scalarMaskGfm` body verbatim.
    const bool gfm2 = gfm && (pcGfmOrder_ >= 2 || (pcCurvDist_ && pcKappa_.extent(0) == n_));
    if (gfm2)
      scalarMaskGfm2(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, sc.gfmB, CCConst(ox_),
                     CCConst(oy_), CCConst(oz_), CCConst(sc.dmask), CCConst(pcTgam_),
                     CCConst(pcGn_[0]), CCConst(pcGn_[1]), CCConst(pcGn_[2]), CCConst(pcGphi_),
                     CCConst(pcCurvDist_ && pcKappa_.extent(0) == n_ ? pcKappa_ : pcGphi_),
                     CCConst(sc.kcell), sc.D, energy, pcGfmThMin_, pcGfmThMax_, pcGfmOrder_,
                     pcCurvDist_ && pcKappa_.extent(0) == n_, e_, G, u_.w[0], u_.w[1], u_.w[2],
                     gmS);
    else if (gfm)
      scalarMaskGfm(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, sc.gfmB, CCConst(ox_),
                    CCConst(oy_), CCConst(oz_), CCConst(sc.dmask), CCConst(pcTgam_),
                    CCConst(pcGn_[0]), CCConst(pcGn_[1]), CCConst(pcGn_[2]), CCConst(pcGphi_),
                    CCConst(sc.kcell), sc.D, energy, pcGfmThMin_, pcGfmThMax_, e_, G, u_.w[0],
                    u_.w[1], u_.w[2], gmS);
    // WO-P01: the optional PER-CELL Dirichlet set (interfacial cells at T_sat). Inert — and the
    // operator therefore bit-identical — until a caller allocates the mask.
    if (hasMask)
      scalarMaskStencil(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(sc.dmask), e_, G);
    Kokkos::deep_copy(sc.cOld, sc.c);
    // WO-P3f: add the enthalpy the interfacial cells' overwrite gave back (option, inert off).
    if (hasMask && pcCarryConserve_ && pcCarrySrc_.extent(0) == n_)
      pcCarryApply(sc);
    scalarFillGhosts(sc);
    if (energy)
      scalarBuildRhsHeat(sc.b, CCConst(sc.cOld), CCConst(sc.rcp), idt, e_, G);
    else
      scalarBuildRhs(sc.b, CCConst(sc.cOld), CCConst(Uf), CCConst(Vf), CCConst(Wf), CCConst(ox_),
                     CCConst(oy_), CCConst(oz_), idt, sc.scheme, e_, G);
    if (gfm)
      scalarAddGfmRhs(sc.b, CCConst(sc.gfmB), CCConst(sc.dmask), e_, G);
    // WO-P3f: the energy budget's PRE snapshot — taken here because `scalarMaskRhs` on the
    // next line is where a newly interfacial cell's transported temperature is discarded.
    // Inert (no kernel, no allocation) unless `set_phase_change_budget(true)` ran.
    if (hasMask && pcBudgetOn_ && pcClsPrev_.extent(0) == n_)
      pcBudgetPre(sc);
    if (hasMask)
      scalarMaskRhs(sc.b, sc.c, CCConst(sc.dmask), CCConst(sc.dval), e_, G);
    // implicit diffusion: red-black Gauss-Seidel with a ghost fill before each color sweep.
    for (int it = 0; it < sc.iters; ++it) {
      scalarFillGhosts(sc);
      cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                         og_, G, 0);
      scalarFillGhosts(sc);
      cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                         og_, G, 1);
    }
    scalarFillGhosts(sc);
    if (hasMask && pcBudgetOn_ && pcClsPrev_.extent(0) == n_)
      pcBudgetPost(sc);  // WO-P3f, inert when the instrument is off
  }
}

template <class Grid>
ScalarField& Solver<Grid>::scalarField(const std::string& name) {
  for (auto& sc : scalars_)
    if (sc.name == name)
      return sc;
  throw std::runtime_error("no scalar named '" + name + "'");
}

template <class Grid>
void Solver<Grid>::scalarDirichletMask(const std::string& name) {
  ScalarField& sc = scalarField(name);
  if (sc.dmask.extent(0) != n_) {
    sc.dmask = CCField(name + "_dmask", n_);
    sc.dval = CCField(name + "_dval", n_);
    sc.gfmB = CCField(name + "_gfmb", n_);
  }
  if (pcGphi_.extent(0) != n_) {
    pcTgam_ = CCField("pc_tgam", n_);
    pcGn_[0] = CCField("pc_gnx", n_);
    pcGn_[1] = CCField("pc_gny", n_);
    pcGn_[2] = CCField("pc_gnz", n_);
    pcGphi_ = CCField("pc_gphi", n_);
  }
}

template <class Grid>
void Solver<Grid>::scalarFillGhosts(ScalarField& sc) {
  fillGhosts(sc.c);
  applyScalarBc(sc);
}

template <class Grid>
void Solver<Grid>::applyScalarBc(ScalarField& sc) {
  for (int f = 0; f < 6; ++f)
    if (sc.bc[f] != 0 && touchesGlobalFace(f))
      applyScalarBcFace(sc.c, f / 2, f % 2, sc.bc[f], sc.bcVal[f]);
}

template <class Grid>
bool Solver<Grid>::touchesGlobalFace(int f) const {
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    const int a = f / 2;
    const int o = (a == 0) ? og_.x : (a == 1) ? og_.y : og_.z;
    const int n = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const int gn = (a == 0) ? gnx_ : (a == 1) ? gny_ : gnz_;
    return (f % 2 == 0) ? (o == 0) : (o + n == gn);
  }
#endif
  (void)f;
  return true;
}

template <class Grid>
void Solver<Grid>::applyScalarBcStencilVar(ScalarField& sc) {
  for (int f = 0; f < 6; ++f) {
    if (sc.bc[f] != 2 || !touchesGlobalFace(f))
      continue;
    const int a = f / 2, side = f % 2;
    CCField band = (a == 0)   ? (side == 0 ? sc.AW : sc.AE)
                   : (a == 1) ? (side == 0 ? sc.AS : sc.AN)
                              : (side == 0 ? sc.AB : sc.AT);
    patchScalarDirichletFaceVar(sc.AC, band, sc.kcell, a, side);
  }
}

template <class Grid>
void Solver<Grid>::applyScalarBcStencil(ScalarField& sc) {
  for (int f = 0; f < 6; ++f) {
    if (sc.bc[f] != 2 || !touchesGlobalFace(f))
      continue;  // only Dirichlet reopens; Neumann/periodic leave the (closed/interior) band
    const int a = f / 2, side = f % 2;
    CCField band = (a == 0)   ? (side == 0 ? sc.AW : sc.AE)
                   : (a == 1) ? (side == 0 ? sc.AS : sc.AN)
                              : (side == 0 ? sc.AB : sc.AT);
    patchScalarDirichletFace(sc.AC, band, sc.D, a, side);
  }
}

template <class Grid>
void Solver<Grid>::patchScalarDirichletFaceVar(CCField AC, CCField band, CCField kc, int a,
                                               int side) {
  const double wa = u_.w[a];  // V5.4: this axis's Laplacian weight (exactly 1.0 isotropic)
  const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
  const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
  const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
  const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
  const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
  const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
  const int aInner = (side == 0) ? G : (G + na - 1);
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::scalar_bc_stencil_var",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
        const double D = kc(i) * wa;  // V5.4: this axis's Laplacian weight (1.0 isotropic)
        AC(i) += D + band(i);
        band(i) = -D;
      });
}

template <class Grid>
void Solver<Grid>::patchScalarDirichletFace(CCField AC, CCField band, double Din, int a, int side) {
  const double D = Din * u_.w[a];  // V5.4: this axis's Laplacian weight (exactly 1.0 isotropic)
  const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
  const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
  const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
  const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
  const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
  const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
  const int aInner = (side == 0) ? G : (G + na - 1);
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::scalar_bc_stencil",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
        // base build put band(i) = -D*open_face and A_C += D*open_face; force the face fully open
        // (band -> -D, A_C gains D*(1-open)) without double-counting when it was already open.
        AC(i) += D + band(i);
        band(i) = -D;
      });
}

template <class Grid>
void Solver<Grid>::applyScalarBcFace(CCField c, int a, int side, int type, double val) {
  const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
  const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
  const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
  const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
  const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
  const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
  const int aInner = (side == 0) ? G : (G + na - 1);  // inner boundary cell a-index
  const int dir = (side == 0) ? -1 : +1;              // toward the ghost
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::scalar_bc_face",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long base = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
        for (int L = 1; L <= 2; ++L) {
          const long gcell = base + (long)dir * L * sa;
          const long icell = base - (long)dir * (L - 1) * sa;
          c(gcell) = (type == 2) ? (2.0 * val - c(icell)) : c(icell);
        }
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_SCALARS_HPP
