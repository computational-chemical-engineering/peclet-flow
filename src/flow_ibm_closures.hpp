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
void Solver<Grid>::fillPropGhosts(CCField f) {
  fillGhosts(f);
  for (int face = 0; face < 6; ++face)
    if (bc_[face] != 0 && touchesGlobalFace(face))
      applyScalarBcFace(f, face / 2, face % 2, 1, 0.0);  // type 1 = Neumann copy
  vofBcPropGhosts(f);  // WO-R item 5; a no-op unless a VoF inflow colour is set
}

template <class Grid>
void Solver<Grid>::fillMuGhosts() {
  fillPropGhosts(muField_);
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
  if (varProps_)
    fp.mu = CCConst(muField_);
  else
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
void Solver<Grid>::computeDivAdv() {
  CCExec space;
  C3 e = e_;
  CCField dv = divAdv_;
  CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
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
  // INTERNAL units. LinearMix is `out = p0 + p1*in0 + p2*in1` with a dimensionless input (a
  // phase fraction, a normalised scalar), so a target of "rho" or "mu" scales every parameter by
  // that property's own factor — which is what makes the documented two-phase spelling
  // set_property_model("rho", "linear", "C", [rho_gas, rho_liquid - rho_gas]) take the caller's
  // PHYSICAL densities. Exactly the identity in cell units. The non-linear kinds mix
  // dimensionally different parameters in one array and are NOT converted; they say so.
  double kp = 1.0;
  if (target == "rho")
    kp = u_.rhoToInt();
  else if (target == "mu")
    kp = u_.muToInt();
  if (kp != 1.0 && kind != ClosureKind::LinearMix) {
    std::fprintf(stderr,
                 "peclet.flow set_property_model NOTICE: a physical domain is armed, but only "
                 "the 'linear' closure on 'rho'/'mu' converts its parameters. This closure's "
                 "parameters are in the solver's INTERNAL units (see the `unit_scales` "
                 "property); target '%s'.\n",
                 target.c_str());
    kp = 1.0;
  }
  for (int k = 0; k < 4 && k < (int)params.size(); ++k)
    cl.p[k] = params[k] * kp;
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
  }
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
  closures_.push_back(cl);
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
  // the accumulated pressure diverges exponentially. Non-porous (incompressible drag, w==1 path)
  // keeps the validated cell-beta form.
  const bool faceAvg = porous_;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::add_drag_diag", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double bd =
            faceAvg ? 0.5 * ((double)beta(i) + (double)beta(i - sc)) : (double)beta(i);
        AC(i) = (MReal)((double)AC(i) + bd);
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_CLOSURES_HPP
