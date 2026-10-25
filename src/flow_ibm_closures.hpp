/// @file
/// @brief flow — IbmSolver property closures: property/density modes, porous continuity, drag, and
/// the property/eps ghost fills.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_CLOSURES_HPP
#define PECLET_FLOW_FLOW_IBM_CLOSURES_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::fillPropGhosts(CCField f, bool edges) {
  fillGhosts(f);
  for (int face = 0; face < 6; ++face)
    if (bc_[face] != 0 && touchesGlobalFace(face))
      applyScalarBcFace(f, face / 2, face % 2, 1, 0.0, edges);  // type 1 = Neumann copy
  vofBcPropGhosts(f);  // WO-R item 5; a no-op unless a VoF inflow colour is set
}

template <class Grid>
void Solver<Grid>::fillMuGhosts() {
  // edges: the staggered cross-face viscosity reads mu at i - s_c +- s_a, a diagonal ghost at a
  // wall row -- which must be the zero-gradient copy, not the opposite wall's periodic wrap.
  fillPropGhosts(muField_, /*edges=*/true);
}

template <class Grid>
void Solver<Grid>::fillCellForceGhosts() {
  if (!hasCellForce_)
    return;
  for (int c = 0; c < 3; ++c)
    fillPropGhosts(cellForce_[c]);
}

template <class Grid>
void Solver<Grid>::fillDragBetaGhosts() {
  if (!hasDrag_)
    return;
  fillPropGhosts(dragBeta_);
}

template <class Grid>
void Solver<Grid>::fillPorousEpsGhosts() {
  fillGhosts(epsField_);
  for (int face = 0; face < 6; ++face) {
    const int t = bc_[face];
    if (t == 0 || !touchesGlobalFace(face))
      continue;  // rank-owned faces only (see fillPropGhosts)
    if (t == 2 || t == 3)
      applyScalarBcFace(epsField_, face / 2, face % 2, 2, 1.0);  // open face: face eps == 1
    else
      applyScalarBcFace(epsField_, face / 2, face % 2, 1, 0.0);  // wall: zero-gradient
  }
}

template <class Grid>
bool Solver<Grid>::effVarRho() const {
  return varRho_ || (porous_ && porousCons_);
}

template <class Grid>
CCField Solver<Grid>::effRhoField() {
  return varRho_ ? rhoField_ : epsRho_;
}

template <class Grid>
void Solver<Grid>::updateEpsRho() {
  CCExec space;
  CCField er = epsRho_;
  CCConst ep = CCConst(epsField_);
  const double rho = rho_;
  Kokkos::parallel_for(
      "peclet::flow::eps_rho", Kokkos::RangePolicy<CCExec>(space, 0, n_),
      KOKKOS_LAMBDA(std::size_t i) { er(i) = ep(i) * rho; });
}

template <class Grid>
VarFaceProps Solver<Grid>::makeFaceProps(int c) {
  VarFaceProps fp;
  fp.haveMu = varProps_;
  if (varProps_) {
    fp.mu = CCConst(muField_);
    fp.msc = Grid::collocated ? 0 : strideOf(c);  // mu on the staggered unknown's own faces
  } else
    fp.muC = mu_;
  fp.harmMu = harmonicMu_;
  fp.haveRho = effVarRho();
  if (effVarRho()) {
    fp.rho = CCConst(effRhoField());
    fp.idt = 1.0 / dt_;
    // Placement of the velocity unknown: the staggered unknown sits on the -c FACE, so its time
    // diagonal is the arithmetic face mean of rho; the COLLOCATED unknown sits at the cell CENTRE
    // (Grid::offset == 0), so it is rho(i) itself — which is what stride 0 gives,
    // 0.5*(rho(i)+rho(i)) == rho(i) exactly in floating point. Inert until rung V8: `haveRho` was
    // unreachable on the collocated grid (set_density_mode and set_porous_continuity both threw).
    fp.sc = Grid::collocated ? 0 : strideOf(c);
  } else
    fp.rhoIdtC = rho_ / dt_;
  return fp;
}

template <class Grid>
void Solver<Grid>::setViscousStress(const std::string& form) {
  if (form == "auto") {
    viscousStressAuto_ = true;
    fullStress_ = false;
    return;
  }
  if (form == "laplacian") {
    viscousStressAuto_ = false;
    fullStress_ = false;
    fullStressAuto_ = false;
    return;
  }
  if (form != "full")
    throw std::invalid_argument(
        "set_viscous_stress: form must be 'auto', 'full' or 'laplacian', got '" + form + "'");
  if constexpr (Grid::collocated)
    throw std::runtime_error(
        "SolverColocated.set_viscous_stress('full'): the full viscous stress is staggered-only in "
        "v1 (doc/full_viscous_stress.md §9); use peclet.flow.Solver");
  viscousStressAuto_ = false;
  fullStressAuto_ = false;
  fullStress_ = true;
}

template <class Grid>
bool Solver<Grid>::fullStressActive() const {
  return (fullStress_ || fullStressAuto_) && varProps_ && !Grid::collocated;
}

template <class Grid>
void Solver<Grid>::resolveViscousStressAuto() {
  // WO-7 (doc/full_viscous_stress.md §12): AUTO = full on the staggered grid when mu is variable,
  // arithmetic, not porous, not moving and free of immersed solids (review B1: the regular-node
  // rule drops the transpose flux across the STAIRCASE face, not the wall -- an O(gamma h)
  // near-wall error on oblique walls that does not scale with the mu variation; the repair is
  // open); otherwise laplacian with a one-time stderr notice. A constant mu has nothing to resolve
  // (the two forms coincide). Re-decided every step: the flags it reads are final at the step head,
  // as for the explicit "full" refusals.
  fullStressAuto_ = false;
  if (!varProps_)
    return;
  const char* why = nullptr;
  if constexpr (Grid::collocated)
    why = "the collocated grid (staggered-only in v1, §9)";
  else if (harmonicMu_)
    why = "the harmonic face viscosity";
  else if (porous_)
    why = "porous (eps-weighted) momentum";
  else if (hasMotion_)
    why = "moving scene instances";
  else if (hasSolid_)
    why =
        "an immersed solid (the staircase drops the transpose flux, an O(gamma h) near-wall error "
        "on oblique walls; repair open)";
  if (why == nullptr) {
    fullStressAuto_ = true;
    return;
  }
  if (!fullStressAutoNotice_) {
    fullStressAutoNotice_ = true;
    fprintf(stderr,
            "peclet::flow %s: AUTO viscous stress fell back to the Laplacian form div(mu grad u) "
            "(%s is outside the full stress's v1 scope, doc/full_viscous_stress.md §5.6). Select "
            "explicitly with set_viscous_stress to silence this notice.\n",
            Grid::collocated ? "SolverColocated" : "Solver", why);
  }
}

template <class Grid>
void Solver<Grid>::requireFullStressScope(const char* who) const {
  const auto refuse = [&](const char* why) {
    throw std::runtime_error(std::string(who) + ": set_viscous_stress('full') " + why +
                             " (doc/full_viscous_stress.md §5.6); call "
                             "set_viscous_stress('laplacian') for this configuration");
  };
  if (harmonicMu_)
    refuse(
        "does not support the harmonic face viscosity (the series direction is "
        "component-dependent and breaks the symmetry of the transpose term)");
  if (porous_)
    refuse(
        "does not support porous (eps-weighted) momentum: the eps-weighted stress is a "
        "different equation");
  if (hasMotion_)
    refuse(
        "does not support moving scene instances (the regular-node masks would have to follow "
        "them)");
}

template <class Grid>
void Solver<Grid>::fullStressPrelude() {
  // 1. mu ghosts: rebuildStencils() filled them unless this is the implicit-FOU path (which fills
  //    them later, in buildAdvStencilVar -- after the uniformity test below would read them).
  if (implicitAdv())
    fillMuGhosts();
  // 2. The exact GLOBAL uniformity test over the inner cells plus the ring T reads: T is either the
  //    complete conservative term or nothing (§4.4 (C)). One reduction + host read per step.
  double lo = 0.0, hi = 0.0;
  muRangeInner(CCConst(muField_), e_, G, lo, hi);
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g[2] = {lo, -hi}, r[2];
    MPI_Allreduce(g, r, 2, MPI_DOUBLE, MPI_MIN, comm_);
    lo = r[0];
    hi = -r[1];
  }
#endif
  muUniform_ = (lo == hi);
  if (muUniform_)
    return;
  // 3. The u^n ghosts incl. the (a, c) edges, written into old_ only (C[c].u is untouched);
  //    doOutflow = false keeps the projected outflow face, faceEdges its transverse ghost ring
  //    (the edge T reads at the last inner row; distributed only, review S1).
  for (int c = 0; c < 3; ++c)
    fillVelGhostsTo(old_[c], c, 0, /*doOutflow=*/false, /*faceEdges=*/true);
  // 4. The regular-node masks of a static solid geometry, rebuilt after every geometry build.
  if (hasSolid_ && regDirty_)
    buildFullStressRegMasks();
}

template <class Grid>
void Solver<Grid>::buildFullStressRegMasks() {
  CCExec space;
  const std::size_t n = (std::size_t)n_;
  // A transient double carrier for the ghost exchange (the halo is GridHalo<double>); freed on
  // return, so the persistent cost is the three uint8 masks.
  CCField tmp(
      Kokkos::view_alloc(space, "peclet::flow::full_stress_reg_tmp", Kokkos::WithoutInitializing),
      n);
  for (int c = 0; c < 3; ++c) {
    if (fsReg_[c].extent(0) != n)
      fsReg_[c] = RegMask(
          Kokkos::view_alloc(space, "peclet::flow::full_stress_reg", Kokkos::WithoutInitializing),
          n);
    CCConst m = CCConst(C[c].mask);
    Kokkos::parallel_for(
        "peclet::flow::full_stress_reg_mask", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { tmp(i) = (m(i) == 0.0) ? 1.0 : 0.0; });
    const auto ov = C[c].ov;
    Kokkos::parallel_for(
        "peclet::flow::full_stress_reg_cut", Kokkos::RangePolicy<CCExec>(space, 0, C[c].nCut),
        KOKKOS_LAMBDA(int k) { tmp(ov.cell_index(k)) = 0.0; });
    fillPropGhosts(tmp, /*edges=*/true);
    RegMask r = fsReg_[c];
    Kokkos::parallel_for(
        "peclet::flow::full_stress_reg_pack", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { r(i) = (tmp(i) != 0.0) ? 1 : 0; });
  }
  space.fence();  // tmp is released on return
  regDirty_ = false;
}

template <class Grid>
void Solver<Grid>::addFullStressRhsComp(int c) {
  // §5.5: the low-face node of component c on a non-periodic global c-face is BC-imposed data.
  const bool skipLo = bc_[2 * c] != 0 && touchesGlobalFace(2 * c);
  const CCConst v[3] = {CCConst(old_[0]), CCConst(old_[1]), CCConst(old_[2])};
  if (hasSolid_) {
    const RegConst reg[3] = {fsReg_[0], fsReg_[1], fsReg_[2]};
    addFullStressRhs<VarFaceProps, true>(c, C[c].b, CCConst(C[c].rscale), v, makeFaceProps(c), e_,
                                         G, u_.w[c], skipLo, reg);
  } else {
    addFullStressRhs<VarFaceProps, false>(c, C[c].b, CCConst(C[c].rscale), v, makeFaceProps(c), e_,
                                          G, u_.w[c], skipLo);
  }
}

template <class Grid>
void Solver<Grid>::computeDivAdv() {
  CCExec space;
  C3 e = e_;
  CCField dv = divAdv_;
  CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::div_adv",
      MD(space, {G - 1, G - 1, G - 1}, {e.x - G + 1, e.y - G + 1, e.z - G + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        dv(i) = (U(i + sx) - U(i)) + (V(i + sy) - V(i)) + (W(i + sz) - W(i));
      });
}

template <class Grid>
void Solver<Grid>::setPropertyModel(const std::string& target, ClosureKind kind,
                                    const std::string& in0, const std::string& in1,
                                    const std::vector<double>& params) {
  if (target == "mu" && vmgExplicit_ && useVelocityMg_)  // before anything is registered
    throwVelocityMgVariableMu("set_property_model('mu', ...)");
  Closure cl;
  cl.kind = kind;
  cl.out = ensureTarget(target);
  cl.in0 = CCConst(fields_.at(in0).data);
  if (!in1.empty())
    cl.in1 = CCConst(fields_.at(in1).data);
  cl.outName = target;  // kept so a redistribute can re-resolve the handles (rebindFieldAliases)
  cl.in0Name = in0;
  cl.in1Name = in1;
  // UNITS. A closure writes a registered field, and registered fields carry the solver's
  // INTERNAL units. Its inputs are UNSCALED fields (a phase fraction, a temperature, a
  // concentration — the solver never rescales a transported scalar), so the output's dimension
  // sits in the parameters, and on a target of known dimension the ones carrying it convert by
  // that target's factor: all of LinearMix's (`out = p0 + p1*in0 + p2*in1`, which is what makes
  // set_property_model("rho", "linear", "C", [rho_gas, rho_liquid - rho_gas]) take PHYSICAL
  // densities), Boussinesq's rho0 (`out = rho0*g*beta*(T - T0)` into force_a), Arrhenius' mu_ref.
  // The physical values are kept, so a scale pinned later re-derives them. Exactly the identity
  // in cell units.
  for (int k = 0; k < 4 && k < (int)params.size(); ++k)
    cl.pPhys[k] = params[k];
  closureUnitsNotice(target);
  closureRefreshUnits(cl);
  closures_.push_back(cl);
  if (target == "mu")  // a closure driving mu turns on variable viscosity
    setPropertyMode(true, harmonicMu_);
  if (target == "rho")  // a closure driving rho turns on the variable-density path
    setDensityMode(true);
}

template <class Grid>
void Solver<Grid>::setDensityMode(bool variable) {
  if (variable)
    collocatedV8AutoFallback("variable density on the collocated grid");
  varRho_ = variable;
  if (variable) {
    if (fields_.has("rho"))
      rhoField_ = fields_.at("rho").data;
    else {
      rhoField_ = addField("rho");
      Kokkos::deep_copy(rhoField_, rho_);
    }
    if (rho1_.extent(0) == 0) {  // g=1 MG-block scratch for the projection coefficients
      rho1_ = CCField("rho1", n1_);
      cx1_ = CCField("cx1", n1_);
      cy1_ = CCField("cy1", n1_);
      cz1_ = CCField("cz1", n1_);
    }
    ensureCellForceAll();    // buildRhsVar reads the per-cell force (zero until a closure sets it)
    useVelocityMg_ = false;  // scalar-coefficient velocity MG (variable-coeff deferred)
    // Pressure driver: CHEBYSHEV by default under variable density — but NOT for the reason this
    // comment used to give ("MG-PCG stalls on the rho-scaled coefficient operator"), which WO-B
    // refuted: the stall it described is a DOMAIN-BC defect at constant density, repaired by
    // WO-H (CutcellMG::applyNeumannGhost), and on the periodic pore-scale operator MG-PCG beats
    // Chebyshev ~10x at density ratio 1e4. The reason that survives measurement is narrower and
    // real: at a high density CONTRAST the arithmetic coarsening of the face coefficient makes
    // the V-cycle preconditioner INDEFINITE (measured on a dense sym(M): a negative pivot from
    // ratio ~1e3), and no Krylov CG survives that, while Chebyshev — which needs only real
    // spectrum bounds, re-estimated on every coefficient rebuild — is healthy on every
    // configuration measured. Coefficient-aware coarsening (VOF_PLAN S3) is what would lift it.
    // An explicit set_pressure_pcg/_fcg/_chebyshev AFTER set_density_mode still wins (last set),
    // and since WO-H set_pressure_pcg's `on` flag genuinely honours that promise.
    useChebyshev_ = true;
    chebBoundsSet_ = false;
    chebWarmOk_ = false;
    bfpChebSet_ = false;
  }
  registerBalancedForceState();  // V8 turns the balanced-force projection on by default
}

template <class Grid>
bool Solver<Grid>::hasCutcellPressure() const {
  return cutcellPressure_;
}

template <class Grid>
void Solver<Grid>::setPorousContinuity(bool on) {
  if constexpr (Grid::collocated) {
    if (on)
      throw std::runtime_error("set_porous_continuity: staggered-only (v1)");
  }
  porous_ = on;
  if (on) {
    if (fields_.has("eps"))
      epsField_ = fields_.at("eps").data;
    else {
      epsField_ = addField("eps");
      Kokkos::deep_copy(epsField_, 1.0);  // no particles -> eps=1 -> reduces to div(u)=0
    }
    if (epsPrev_.extent(0) == 0) {
      epsPrev_ = CCField("epsPrev", n_);
      depsdt_ = CCField("depsdt", n_);
    }
    if (divAdv_.extent(0) == 0)
      divAdv_ = CCField("divAdv", n_);  // cell div(u) for the porous advection-form compensation
    if (epsRho_.extent(0) == 0)
      epsRho_ = CCField("epsRho", n_);  // rho_eff = eps*rho (eps-conservative momentum)
    // The eps-conservative momentum path (porousCons_) routes through buildRhsVar, which reads
    // the per-cell force unconditionally — allocate it (zero) like setDensityMode does. Without
    // this a porous run with NO drag/closure (never the coupled case, which enables drag and
    // thereby the force fields) dereferences an empty device View.
    ensureCellForceAll();
    Kokkos::deep_copy(epsPrev_, epsField_);  // d(eps)/dt=0 on the first step
    if (eps1_.extent(0) == 0)
      eps1_ = CCField("eps1", n1_);
    if (beta1_.extent(0) == 0)
      beta1_ = CCField("beta1", n1_);
    if (rho1_.extent(0) == 0) {  // share the g=1 coefficient scratch with the varRho path
      rho1_ = CCField("rho1", n1_);
      cx1_ = CCField("cx1", n1_);
      cy1_ = CCField("cy1", n1_);
      cz1_ = CCField("cz1", n1_);
    }
    // CHEBYSHEV by default (as for variable density). CAVEAT on the original justification
    // ("MG-PCG stalls on the eps-scaled coefficient operator"): it rests on the same kind of
    // observation WO-B refuted for varRho, and it was recorded through the setter whose `on` flag
    // was a no-op until WO-H — so a "PCG" run made that way actually measured Chebyshev. The
    // default is kept because it is safe (Chebyshev needs only real spectrum bounds and is the
    // one driver healthy on every high-contrast coefficient configuration measured), NOT because
    // the eps-scaled PCG stall has been re-measured; that re-measurement is still owed
    // (doc/vof_workorders.md, WO-B escalation #1). Bounds re-estimated on every coefficient
    // rebuild (chebBoundsSet_ invalidation in project()). An explicit driver set afterwards wins.
    useChebyshev_ = true;
    chebBoundsSet_ = false;
    chebWarmOk_ = false;
    bfpChebSet_ = false;
    configurePorousDragSolver();  // if drag already on, switch to GraphAMG+PCG (Chebyshev
                                  // diverges)
  }
}

template <class Grid>
void Solver<Grid>::syncPorousPrev() {
  if (porous_)
    Kokkos::deep_copy(epsPrev_, epsField_);
}

template <class Grid>
void Solver<Grid>::setPorousDepsDt(bool on) {
  porousDepsDt_ = on;
}

template <class Grid>
void Solver<Grid>::setPorousConservative(bool on) {
  porousCons_ = on;
}

template <class Grid>
void Solver<Grid>::setPressureUnderRelax(double w) {
  pressUnderRelax_ = w;
}

template <class Grid>
void Solver<Grid>::setPropertyMode(bool variable, bool harmonic) {
  // An EXPLICIT velocity multigrid is refused in either order; the AUTO rule is simply re-decided
  // (setSolidVelocityMgAuto excludes varProps_), so only an explicit request can conflict.
  if (variable && vmgExplicit_ && useVelocityMg_)
    throwVelocityMgVariableMu("set_property_mode('variable')");
  varProps_ = variable;
  harmonicMu_ = harmonic;
  if (variable) {
    if (fields_.has("mu"))
      muField_ = fields_.at("mu").data;
    else {
      muField_ = addField("mu");
      Kokkos::deep_copy(muField_,
                        mu_);  // default to the scalar mu until a closure/set_field sets it
    }
    useVelocityMg_ =
        false;  // the velocity multigrid takes a scalar mu (variable-coeff vmg deferred)
  }
}

template <class Grid>
void Solver<Grid>::setVariableRotational(int mode, double chi) {
  varRotMode_ = mode < 0 ? 0 : (mode > 2 ? 2 : mode);
  varRotChi_ = chi < 0.0 ? 0.0 : chi;
}

template <class Grid>
void Solver<Grid>::setPropertyTable(const std::string& target, const std::string& in0,
                                    const std::vector<double>& xs, const std::vector<double>& ys) {
  Closure cl;
  cl.kind = ClosureKind::Table1D;
  cl.out = ensureTarget(target);
  cl.in0 = CCConst(fields_.at(in0).data);
  cl.outName = target;  // see setPropertyModel
  cl.in0Name = in0;
  cl.nTab = (int)std::min(xs.size(), ys.size());
  cl.tabX = CCField(target + "_tabx", cl.nTab);
  cl.tabY = CCField(target + "_taby", cl.nTab);
  auto hx = Kokkos::create_mirror_view(cl.tabX);
  auto hy = Kokkos::create_mirror_view(cl.tabY);
  for (int k = 0; k < cl.nTab; ++k) {
    hx(k) = xs[k];
    hy(k) = ys[k];
  }
  Kokkos::deep_copy(cl.tabX, hx);
  Kokkos::deep_copy(cl.tabY, hy);
  cl.tabYPhys.assign(ys.begin(), ys.begin() + cl.nTab);
  closureUnitsNotice(target);
  closureRefreshUnits(cl);  // ys are in the target's physical units (identity in cell units)
  closures_.push_back(cl);
}

template <class Grid>
double Solver<Grid>::closureTargetToInt(const std::string& target) const {
  if (target == "rho")
    return u_.rhoToInt();
  if (target == "mu")
    return u_.muToInt();
  if (target == "force_x" || target == "force_y" || target == "force_z")
    return u_.forceToInt(target[6] - 'x');
  return 0.0;
}

template <class Grid>
void Solver<Grid>::closureUnitsNotice(const std::string& target) const {
  if (u_.physical && closureTargetToInt(target) == 0.0)
    std::fprintf(stderr,
                 "peclet.flow set_property_model NOTICE: a physical domain is armed and target "
                 "'%s' is not rho/mu/force_x/y/z, so this closure's parameters are taken in "
                 "that field's INTERNAL units (see the `unit_scales` property).\n",
                 target.c_str());
}

template <class Grid>
void Solver<Grid>::closureRefreshUnits(Closure& cl) {
  double kp = closureTargetToInt(cl.outName);
  if (kp == 0.0)
    kp = 1.0;  // a field of unknown dimension: the caller's numbers are internal
  if (cl.kind == ClosureKind::Table1D) {
    if (!u_.physical)
      return;  // cell units: the table was uploaded verbatim at registration
    auto hy = Kokkos::create_mirror_view(cl.tabY);
    for (int k = 0; k < cl.nTab; ++k)
      hy(k) = cl.tabYPhys[(std::size_t)k] * kp;
    Kokkos::deep_copy(cl.tabY, hy);
    return;
  }
  // the parameters that carry the output's dimension; the others are kept as given
  const int nScaled = (cl.kind == ClosureKind::LinearMix) ? 3 : 1;
  for (int k = 0; k < 4; ++k)
    cl.p[k] = (k < nScaled) ? cl.pPhys[k] * kp : cl.pPhys[k];
}

template <class Grid>
void Solver<Grid>::updateProperties() {
  for (auto& cl : closures_)
    applyClosure(cl, e_, G);
}

template <class Grid>
void Solver<Grid>::enableCellForce() {
  ensureCellForceAll();
}

template <class Grid>
void Solver<Grid>::enableDrag() {
  if (!fields_.has("drag_beta"))
    dragBeta_ = addField("drag_beta");
  else
    dragBeta_ = fields_.at("drag_beta").data;
  ensureCellForceAll();  // force_* carries beta*u_p (the implicit-drag RHS target)
  hasDrag_ = true;
  configurePorousDragSolver();
}

template <class Grid>
void Solver<Grid>::configurePorousDragSolver() {
  if (!(porous_ && hasDrag_))
    return;
  pressGraphAmg_ = true;  // GraphAMG bottom (domain-BC operators: buildAmg skips
                          // the wrap across non-periodic faces and pcgAmg keeps the
                          // mean only when the operator is singular)
  if (cutcellPressure_)   // MG already built (set_solid ran) -> apply now
    mg_.setAgglomerationMode(1);
  useChebyshev_ = false;  // PCG, not Chebyshev (diverges on the high w_f ratio)
  chebBoundsSet_ = false;
  chebWarmOk_ = false;
  bfpChebSet_ = false;
}

template <class Grid>
void Solver<Grid>::addDragDiagonal(int c) {
  CCExec space;
  C3 e = e_;
  FV AC = C[c].AC;
  CCConst beta = CCConst(dragBeta_);
  const long sc = strideOf(c);
  // Porous continuity: the projection's operator/correction carry the FACE drag relaxation
  // w_f = idt/(idt + beta_f), beta_f = 1/2(beta(i)+beta(i-sc)) (buildPorousCoeffDrag /
  // projectCorrectPorousDrag). The staggered momentum diagonal of u_c(i) — the face between cells
  // i-sc and i — must carry the SAME beta_f: then a pressure perturbation deltaP produces
  // du* = -grad(deltaP)/(idt+beta_f) and the projection returns phi = -deltaP/idt exactly (same
  // operator), so the incremental predictor cancels pressure errors in one step. With the cell
  // value beta(i) the loop has gain (idt+beta_f)/(idt+beta_cell) at a beta jump (bed top: ~3) and
  // the accumulated pressure diverges exponentially.
  // The drag is VOLUMETRIC, so beta is placed where the unknown lives by the GRID, not by the
  // continuity model: the face mean on the MAC grid (porous or not -- the non-porous path used the
  // cell value, half a cell off the face, while its RHS target beta*u_p in force_* is placed by the
  // same Grid::atVelocity), the cell value collocated. Target and diagonal must agree, or a uniform
  // particle velocity does not give a uniform steady gas velocity.
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::add_drag_diag", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double bd = Grid::atVelocity(beta, i, sc);
        AC(i) = (MReal)((double)AC(i) + bd);
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_CLOSURES_HPP
