/// @file
/// @brief flow — IbmSolver core: allocation, physical-domain scales, rho/mu/dt and
/// pressure/velocity driver setters, stencils/ghosts/advection-input builders, the momentum-solve
/// smoother (smoothComp) family, reductions, gather/scatter and the field registry.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_CORE_HPP
#define PECLET_FLOW_FLOW_IBM_CORE_HPP

namespace peclet::flow {

template <class Grid>
Solver<Grid>::Solver(int nx, int ny, int nz) {
  allocateBlock(nx, ny, nz);
}

template <class Grid>
typename Solver<Grid>::SceneMap Solver<Grid>::sceneMap() const {
  SceneMap m;
  if (!u_.physical)
    return m;
  for (int a = 0; a < 3; ++a) {
    m.a[a] = u_.org[a] + 0.5 * u_.h[a];
    m.b[a] = u_.h[a];
    m.velToInt[a] = u_.velToInt(a);
  }
  m.dToInt = u_.lenToInt();
  m.angToInt = u_.angVelToInt();
  return m;
}

template <class Grid>
Solver<Grid>::Solver(int nx, int ny, int nz, const std::array<double, 3>& extent,
                     const std::array<double, 3>& origin, const std::array<long, 3>& globalCells) {
  allocateBlock(nx, ny, nz);
  setPhysicalDomain(extent, origin, globalCells);
}

template <class Grid>
void Solver<Grid>::setPhysicalDomain(const std::array<double, 3>& extent,
                                     const std::array<double, 3>& origin,
                                     const std::array<long, 3>& globalCells) {
  const long c[3] = {globalCells[0] > 0 ? globalCells[0] : nx_,
                     globalCells[1] > 0 ? globalCells[1] : ny_,
                     globalCells[2] > 0 ? globalCells[2] : nz_};
  for (int a = 0; a < 3; ++a) {
    if (!(extent[a] > 0.0))
      throw std::runtime_error("physical domain: every extent component must be > 0");
    if (c[a] <= 0)
      throw std::runtime_error("physical domain: every cell count must be > 0");
  }
  double hh[3];
  for (int a = 0; a < 3; ++a)
    hh[a] = extent[a] / (double)c[a];
  // Phase 2 C2 — THE SNAP (doc/anisotropic_metric.md §1.4).  Phase 1 refused any extent whose
  // three spacings differed; this replaces the throw.  If the three agree to 1e-12 relative the
  // spacings are SNAPPED to h_0 on every axis, so the metric is EXACTLY the identity
  // (hp = w = (1,1,1), vol = 1, aniso = false) — that exactness is what keeps every operator
  // fold of C1/C2 bit-identical and what keeps `units_scale_invariance` on literally Phase 1's
  // arithmetic, whose per-axis extents N_a*H can differ by an ulp across axes (trap 2).
  // Otherwise the cells really are anisotropic: hRef = min_a h_a, and h_a' comes from the STORED
  // doubles (trap 1 — never extent/(cells*hRef); h_a' is exactly 1.0 only on the finest axis).
  // What that admits is §7 / `requireIsotropic` below; Phase 3 then admitted the VoF
  // entry points too, so nothing in the solver refuses an anisotropic domain today.
  bool iso = true;
  for (int a = 1; a < 3; ++a)
    if (std::fabs(hh[a] - hh[0]) > 1e-12 * std::fabs(hh[0]))
      iso = false;
  if (iso)
    for (int a = 1; a < 3; ++a)
      hh[a] = hh[0];
  u_.physical = true;
  for (int a = 0; a < 3; ++a) {
    u_.h[a] = hh[a];
    u_.ext[a] = extent[a];
    u_.org[a] = origin[a];
    u_.cells[a] = c[a];
  }
  u_.hRef = std::min(hh[0], std::min(hh[1], hh[2]));
  for (int a = 0; a < 3; ++a) {
    u_.hp[a] = iso ? 1.0 : u_.h[a] / u_.hRef;
    u_.w[a] = iso ? 1.0 : 1.0 / (u_.hp[a] * u_.hp[a]);
  }
  u_.vol = iso ? 1.0 : u_.hp[0] * u_.hp[1] * u_.hp[2];
  u_.aniso = !iso;
  // rhoRef / tRef are pinned by the first set_rho / set_dt; whatever was set BEFORE the domain
  // (nothing, in the documented order) is re-derived here.
  refreshUnitDerived();
  // Phase 2 commit C4 LIFTED the collocated refusal that stood here: the embedded closures
  // (`fvViscousApply`, `embedViscousApply` / `embedDirichletGradient`) now march along the
  // anisotropic index-space normal `m` of doc/anisotropic_metric.md §6.1 and carry the per-axis
  // `w_a` on their face and wall terms, `starCorrectFaces` carries the same `w_a` its
  // `projectCorrect` fix-up needs, and the V8 face acceleration weights its pressure difference.
  // `enable_vof` kept its own refusal here until PHASE 3 lifted it (see enableVof).
}

template <class Grid>
void Solver<Grid>::requireIsotropic(const char* what) const {
  if (!u_.aniso)
    return;
  char msg[520];
  std::snprintf(msg, sizeof msg,
                "%s: this entry point requires ISOTROPIC cells, but extent/cells gives "
                "dx=%.17g dy=%.17g dz=%.17g (hRef=%.17g, h'=(%.17g, %.17g, %.17g)).",
                what, u_.h[0], u_.h[1], u_.h[2], u_.hRef, u_.hp[0], u_.hp[1], u_.hp[2]);
  throw std::runtime_error(msg);
}

template <class Grid>
bool Solver<Grid>::hasPhysicalDomain() const {
  return u_.physical;
}

template <class Grid>
const typename Solver<Grid>::UnitScales& Solver<Grid>::unitScales() const {
  return u_;
}

template <class Grid>
std::array<double, 3> Solver<Grid>::spacing() const {
  return {u_.h[0], u_.h[1], u_.h[2]};
}

template <class Grid>
std::array<double, 3> Solver<Grid>::domainOrigin() const {
  return {u_.org[0], u_.org[1], u_.org[2]};
}

template <class Grid>
std::array<double, 3> Solver<Grid>::domainExtent() const {
  if (u_.physical)
    return {u_.ext[0], u_.ext[1], u_.ext[2]};
  std::array<long, 3> gc = globalCells();
  return {(double)gc[0], (double)gc[1], (double)gc[2]};
}

template <class Grid>
std::array<long, 3> Solver<Grid>::globalCells() const {
  if (u_.physical && u_.cells[0] > 0)
    return {u_.cells[0], u_.cells[1], u_.cells[2]};
#ifdef PECLET_FLOW_MPI
  if (gnx_ > 0)
    return {gnx_, gny_, gnz_};
#endif
  return {nx_, ny_, nz_};
}

template <class Grid>
std::vector<double> Solver<Grid>::cellCentres(int axis) const {
  if (axis < 0 || axis > 2)
    throw std::runtime_error("cell_centers: axis must be 0, 1 or 2");
  const int n[3] = {nx_, ny_, nz_};
  const int og[3] = {og_.x, og_.y, og_.z};
  std::vector<double> c((std::size_t)n[axis]);
  for (int i = 0; i < n[axis]; ++i)
    c[(std::size_t)i] = u_.org[axis] + ((double)(og[axis] + i) + 0.5) * u_.h[axis];
  return c;
}

template <class Grid>
void Solver<Grid>::refreshUnitDerived() {
  // Phase 3: `hp`, `w`, `vol` and `aniso` are set by `setPhysicalDomain` — including its SNAP,
  // which forces them to the EXACT identity when the three spacings agree to 1e-12.  Recomputing
  // them from `h[a]/hRef` here would undo that snap the moment the extents differ by an ulp
  // across axes, so this only derives the one member Phase 3 added and hands the metric to the
  // VoF drivers.
  u_.hpMax = std::max(u_.hp[0], std::max(u_.hp[1], u_.hp[2]));
  pushVofMetric();
  rho_ = rhoPhys_ * u_.rhoToInt();
  mu_ = muPhys_ * u_.muToInt();
  const double dtInt = dtPhys_ * u_.timeToInt();
  if (dtInt != dt_)
    dt_ = dtInt;
  for (int a = 0; a < 3; ++a)
    f_[a] = fPhys_[a] * u_.forceToInt(a);
  sigmaCsf_ = sigmaPhys_ * u_.sigmaToInt();
  slipLambda_ = slipPhys_ * u_.lenToInt();
  for (int face = 0; face < 6; ++face) {
    for (int a = 0; a < 3; ++a)
      bcVel_[face][a] = bcVelPhys_[face][a] * u_.velToInt(a);
    if (!bcProfRaw_[face].empty())
      resampleBcProfile(face);
  }
  // rho/mu/dt all feed the momentum diagonal; a scale that moved after the operator was built
  // must rebuild it. (Only reachable on the physical path — the cell-unit path never calls
  // this, so no existing run gains a rebuild.)
  dtDirty_ = true;
}

template <class Grid>
void Solver<Grid>::pushVofMetric() {
  const vof::VofMetric g = u_.vofMetric();
  vofAdv_.metric = g;
  vofCurv_.metric = g;
  vofDyn_.metric = g;
  pcAreaC_.metric = g;
  pcAreaMc_.metric = g;
  if (vofBlocks_)
    vofBlocks_->setMetric(g);
}

template <class Grid>
void Solver<Grid>::allocateBlock(int nx, int ny, int nz) {
  nx_ = nx;
  ny_ = ny;
  nz_ = nz;
  e_ = C3{nx + 2 * G, ny + 2 * G, nz + 2 * G};
  n_ = (std::size_t)e_.x * e_.y * e_.z;
  e1_ = C3{nx + 2, ny + 2, nz + 2};  // g=1 block for the cut-cell pressure MG
  n1_ = (std::size_t)e1_.x * e1_.y * e1_.z;
  sdf_ = CCField("sdf", n_);
  ox_ = CCField("ox", n_);
  oy_ = CCField("oy", n_);
  oz_ = CCField("oz", n_);
  phi_ = CCField("phi", n_);
  div_ = CCField("div", n_);
  P_ = CCField("P", n_);
  // g=1 scratch for the MG bridge (openness + rhs/phi + PCG vectors)
  ox1_ = CCField("ox1", n1_);
  oy1_ = CCField("oy1", n1_);
  oz1_ = CCField("oz1", n1_);
  rhs1_ = CCField("rhs1", n1_);
  phi1_ = CCField("phi1", n1_);
  r_ = CCField("r", n1_);
  z_ = CCField("z", n1_);
  pp_ = CCField("pp", n1_);
  Ap_ = CCField("Ap", n1_);
  for (int c = 0; c < 3; ++c) {
    C[c].u = CCField("u", n_);
    C[c].b = CCField("b", n_);
    C[c].AC = FV("AC", n_);
    C[c].AW = FV("AW", n_);
    C[c].AE = FV("AE", n_);
    C[c].AS = FV("AS", n_);
    C[c].AN = FV("AN", n_);
    C[c].AB = FV("AB", n_);
    C[c].AT = FV("AT", n_);
    C[c].inhom = CCField("inhom", n_);
    C[c].rscale = CCField("rscale", n_);
    C[c].mask = CCField("mask", n_);
    bcDcorr_[c] = CCField("dcorr", n_);
    bcBrhs_[c] = CCField("brhs", n_);
    const int maxCut = nx * ny * nz;
    // G.6: IbmOverlay row data follows MReal (was hardcoded float -- FV32) so a
    // -DPECLET_FLOW_OPERATOR_DOUBLE build carries the overlay in double end to end.
    C[c].ov = IbmOverlay{Kokkos::View<int*, CCMem>("ci", maxCut),
                         Kokkos::View<int*, CCMem>("nb", maxCut),
                         FV("dr", maxCut),
                         Kokkos::View<int*, CCMem>("dc", (std::size_t)maxCut * 6),
                         FV("K", (std::size_t)maxCut * 6),
                         FV("M", (std::size_t)maxCut * 6),
                         FV("X", (std::size_t)maxCut * 6),
                         FV("Nbc", (std::size_t)maxCut * 6),
                         FV("R", (std::size_t)maxCut * 6)};
    C[c].idMap = Kokkos::View<int*, CCMem>("idMap", n_);
    C[c].counter = Kokkos::View<int, CCMem>("cnt");
    old_[c] = CCField("uOld", n_);    // u^n time base (fixed over the step's Picard sweeps)
    prev_[c] = CCField("uPrev", n_);  // previous Picard iterate (outer-tolerance check)
  }
  if constexpr (Grid::collocated) {  // transient face (MAC) field for the approximate projection
    uf_ = CCField("uf", n_);
    vf_ = CCField("vf", n_);
    wf_ = CCField("wf", n_);
    tgp_ = CCField("tgp", n_);  // scratch: the collocated cell pressure gradient
    fvM_ = CCField("fvM", n_);  // scratch: M·u^k (embed defect matvec)
    fvL_ = CCField("fvL", n_);  // scratch: L_FV(u^k) (embed operator apply)
    cs_ = CCField("cs", n_);    // static cell fluid fraction (embed)
    xcx_ = CCField("xcx", n_);  // static per-face open-centroid wall distance (wall-aware map)
    xcy_ = CCField("xcy", n_);
    xcz_ = CCField("xcz", n_);
  }
  // Register the pre-existing solver fields in the named directory so the multiphysics machinery
  // (scalar transport, property closures) and load-balance redistribution can enumerate the whole
  // set uniformly. adopt() aliases the members (no reallocation, no ownership); all live on the
  // G=2 velocity block and share velHalo_ under MPI.
  fields_.adopt("u", C[0].u, G, peclet::core::Centering::FaceX);
  fields_.adopt("v", C[1].u, G, peclet::core::Centering::FaceY);
  fields_.adopt("w", C[2].u, G, peclet::core::Centering::FaceZ);
  fields_.adopt("p", P_, G, peclet::core::Centering::Cell);
  fields_.adopt("sdf", sdf_, G, peclet::core::Centering::Cell);
}

template <class Grid>
void Solver<Grid>::setRho(double r) {
  rhoPhys_ = r;
  if (u_.physical && !u_.rhoRefSet) {
    u_.rhoRef = r;
    u_.rhoRefSet = true;
    refreshUnitDerived();
    return;
  }
  const double ri = r * u_.rhoToInt();
  if (ri != rho_) {
    rho_ = ri;
    dtDirty_ = true;  // the momentum stencil bakes rho/dt in its diagonal (rebuildStencils)
  }
}

template <class Grid>
void Solver<Grid>::setMu(double m) {
  muPhys_ = m;
  const double mi = m * u_.muToInt();
  if (mi != mu_) {
    mu_ = mi;
    dtDirty_ = true;  // the momentum stencil bakes mu in its off-diagonals (rebuildStencils)
  }
}

template <class Grid>
void Solver<Grid>::setDt(double d) {
  dtPhys_ = d;
  if (u_.physical && !u_.tRefSet) {
    u_.tRef = d;
    u_.tRefSet = true;
    const double before = dt_;
    refreshUnitDerived();  // pins tRef, so mu'/p'/F'/v all move with it
    if (dt_ != before)
      dtDirty_ = true;
    return;
  }
  const double di = d * u_.timeToInt();
  if (di != dt_) {
    dt_ = di;
    dtDirty_ = true;  // the momentum stencil bakes rho/dt in its diagonal (rebuildStencils);
                      // a mid-run dt change must rebuild it or the operator and RHS disagree
  }
}

template <class Grid>
void Solver<Grid>::setBodyForce(double fx, double fy, double fz) {
  fPhys_ = {fx, fy, fz};
  for (int a = 0; a < 3; ++a)
    f_[a] = fPhys_[a] * u_.forceToInt(a);
}

template <class Grid>
void Solver<Grid>::setVelocityIterations(int it) {
  velIters_ = it;
}

template <class Grid>
void Solver<Grid>::setVelocityTolerance(double rtol, int minIters) {
  velTol_ = rtol > 0.0 ? rtol : 0.0;
  velMinIters_ = minIters < 1 ? 1 : minIters;
}

template <class Grid>
long Solver<Grid>::lastMomentumSweeps() const {
  return lastMomentumSweeps_;
}

template <class Grid>
void Solver<Grid>::setVelocityResidualTolerance(double rtol) {
  velResTol_ = rtol;
}

template <class Grid>
void Solver<Grid>::setVelocityMultigridAuto(long cellsPerRank, long minGlobalCells) {
  vmgAutoCells_ = cellsPerRank;
  if (minGlobalCells >= 0)
    vmgAutoMinGlobal_ = minGlobalCells;
}

template <class Grid>
double Solver<Grid>::velocityResidualTolerance() const {
  if (velResTol_ >= 0.0)
    return velResTol_;
  return useChebyshev_ ? chebRtol_ : pcgRtol_;  // FCG shares pcgRtol_; the plain V-cycle driver
                                                // has no tolerance and takes the PCG default
}

template <class Grid>
double Solver<Grid>::lastMomentumResidual() const {
  return lastMomentumResid_;
}

template <class Grid>
void Solver<Grid>::setPressureMeanRemoval(bool all) {
  mg_.setMeanRemovalScope(all);
}

template <class Grid>
void Solver<Grid>::setPressureIterations(int it) {
  presIters_ = it;
}

template <class Grid>
void Solver<Grid>::setAdvection(bool on) {
  advect_ = on;
}

template <class Grid>
void Solver<Grid>::setAdvectionScheme(int s) {
  advScheme_ = s;
}

template <class Grid>
void Solver<Grid>::setImplicitAdvection(bool on) {
  implicitFou_ = on;
}

template <class Grid>
void Solver<Grid>::setOuterIterations(int iters) {
  outerIters_ = iters < 1 ? 1 : iters;
}

template <class Grid>
void Solver<Grid>::setOuterTolerance(double tol) {
  outerTol_ = tol;
}

template <class Grid>
long Solver<Grid>::lastOuterIterations() const {
  return lastOuterIters_;
}

template <class Grid>
void Solver<Grid>::setVelocityMultigrid(bool on, int levels, int vcycles) {
  useVelocityMg_ = on;
  vmgExplicit_ = true;  // an explicit choice disables the AUTO rule
  vmgLevels_ = levels < 1 ? 1 : levels;
  vmgVcycles_ = vcycles < 1 ? 1 : vcycles;
}

template <class Grid>
bool Solver<Grid>::velocityMultigridActive() const {
  return useVelocityMg_;
}

template <class Grid>
void Solver<Grid>::setPressureBottomMode(int mode) {
  pressAgglomMode_ = mode;
  if (cutcellPressure_)
    mg_.setAgglomerationMode(mode);
}

template <class Grid>
void Solver<Grid>::setPressureTelescope(bool on) {
  mg_.setTelescope(on);
}

template <class Grid>
bool Solver<Grid>::pressureTelescope() const {
  return mg_.telescope();
}

template <class Grid>
void Solver<Grid>::setPressureTelescopeForceLevel(int level) {
  mg_.setTelescopeForceLevel(level);
}

template <class Grid>
int Solver<Grid>::pressureTelescopeCount() const {
  return mg_.telescopeCount();
}

template <class Grid>
void Solver<Grid>::setPressureGraphAmg(bool on) {
  pressGraphAmg_ = on;
  if (cutcellPressure_)
    mg_.setGraphAmgBottom(on);  // propagate live (previously only applied at the next set_solid,
                                // so toggling after geometry silently had no effect)
}

template <class Grid>
void Solver<Grid>::setPressureLevels(int levels) {
  nLevels_ = levels < 1 ? 1 : levels;
}

template <class Grid>
void Solver<Grid>::setBackflowStab(double beta) {
  backflowBeta_ = beta < 0.0 ? 0.0 : beta;
}

template <class Grid>
void Solver<Grid>::setDeferredCorrection(bool on) {
  deferredCorr_ = on;
}

template <class Grid>
void Solver<Grid>::setPressureChebyshev(bool on, int maxit, double rtol) {
  useChebyshev_ = on;
  if (on)
    useFcg_ = false;
  chebMaxit_ = maxit;
  chebRtol_ = rtol;
  chebBoundsSet_ = false;
}

template <class Grid>
void Solver<Grid>::setPressurePcg(bool on, int maxit, double rtol) {
  if (!on)
    throw std::runtime_error(
        "set_pressure_pcg(False): MG-PCG is the default/terminal pressure driver, so it cannot be "
        "deselected on its own — select the driver you want instead "
        "(set_pressure_chebyshev(True, ...) or set_pressure_fcg(True, ...)).");
  useChebyshev_ = false;  // genuine selection: the three drivers are mutually exclusive
  useFcg_ = false;
  chebBoundsSet_ = false;
  pcgMaxit_ = maxit;
  pcgRtol_ = rtol;
}

template <class Grid>
void Solver<Grid>::setPressureFcg(bool on, int maxit, double rtol) {
  if (on && ghostProjection_)
    throw std::runtime_error(
        "set_pressure_fcg: the ghost-projection operator is nonsymmetric and is solved by "
        "BiCGStab; FCG does not apply");
  useFcg_ = on;
  if (on)
    useChebyshev_ = false;  // genuine selection (the pair is exclusive)
  pcgMaxit_ = maxit;
  pcgRtol_ = rtol;
}

template <class Grid>
void Solver<Grid>::setGhostProjection(bool on, int matrixOrder, int rhsOrder) {
  if constexpr (Grid::collocated) {
    // Collocated ghost mode: the SAME phi matrix/closures on the 1/2-1/2 face-averaged field
    // (the face correction uf -= grad(phi) is the identical substitution), plus the directional
    // gpCenterGrad cell gradient for the predictor -grad(P^n) and the cell correction. Only the
    // plain (mode-0) face map applies — the wall-aware/FV/embed face-interp modes replace the
    // very operators this scheme owns.
    if (on && faceInterp_ != 0)
      faceInterp_ = 0;  // QUARANTINED verification path: it owns the operators the gauge-exact
                        // scheme replaces, so it selects the plain face map itself rather than
                        // throwing on the (now default) gauge-exact scheme.
  }
  if (on && geometryBuilt_)
    throw std::runtime_error(
        "set_ghost_projection / set_collocated_scheme('ghost'): call BEFORE set_solid -- the "
        "ghost overlay is built with the geometry");
  if (on && (porous_ || varRho_ || hasBc_ || useChebyshev_))
    throw std::runtime_error(
        "set_ghost_projection: incompatible with porous/variable-rho/domain-BC/Chebyshev (v1)");
  if (matrixOrder < 1 || matrixOrder > 2 || rhsOrder < 1 || rhsOrder > 2)
    throw std::runtime_error("set_ghost_projection: matrix_order/rhs_order must be 1 or 2");
  if (on && distributed_ && (hasExactCross_ || hasOpenOverride_))
    throw std::runtime_error(
        "set_ghost_projection: exact-crossings/openness-override are single-rank only");
  ghostProjection_ = on;
  colSchemeAuto_ = false;  // explicit selection disables the AUTO default
  gpMatrixOrder_ = matrixOrder;
  gpRhsOrder_ = rhsOrder;
  gpNRows_ = -1;  // takes effect at the next set_solid
}

template <class Grid>
void Solver<Grid>::setExactCrossings(const std::vector<double>& t) {
  requireNoGeometry("set_exact_crossings");
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  if (t.empty()) {
    hasExactCross_ = false;
    return;
  }
  if (t.size() != 9 * n)
    throw std::runtime_error("set_exact_crossings: expected 9*nx*ny*nz values");
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    throw std::runtime_error("set_exact_crossings: single-rank only");
#endif
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < 3; ++k) {
      tEx_[c][k] = CCField("tEx", n);
      Kokkos::deep_copy(
          tEx_[c][k],
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              t.data() + ((std::size_t)c * 3 + k) * n, n));
    }
  hasExactCross_ = true;
}

template <class Grid>
void Solver<Grid>::setOpennessOverride(const std::vector<double>& ox, const std::vector<double>& oy,
                                       const std::vector<double>& oz) {
  requireNoGeometry("set_openness_override");
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  if (ox.empty()) {
    hasOpenOverride_ = false;
    return;
  }
  if (ox.size() != n || oy.size() != n || oz.size() != n)
    throw std::runtime_error("set_openness_override: expected nx*ny*nz values per field");
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    throw std::runtime_error("set_openness_override: single-rank only");
#endif
  oxOverride_ = ox;
  oyOverride_ = oy;
  ozOverride_ = oz;
  hasOpenOverride_ = true;
}

template <class Grid>
void Solver<Grid>::setIncrementalPressure(bool on) {
  incremental_ = on;
}

template <class Grid>
void Solver<Grid>::setPressureWarmstart(bool on) {
  pwarm_ = on;
}

template <class Grid>
void Solver<Grid>::setFaceInterp(int mode) {
  if (mode != 0 && mode != 9 && (mode < 5 || mode > 7))
    throw std::runtime_error(
        "set_face_interp: mode " + std::to_string(mode) +
        " is not one of 0 (plain), 5/6/7 (embed rungs) or 9 (gauge-exact); use "
        "set_collocated_scheme('plain' | 'gauge-exact' | 'embed' | 'ghost')");
  if (ghostProjection_ && mode != 0)
    throw std::runtime_error(
        "set_face_interp: incompatible with the ghost projection (set_ghost_projection(False) "
        "first, or use set_collocated_scheme which handles the transition)");
  faceInterp_ = mode;
  colSchemeAuto_ = false;  // explicit selection disables the AUTO default
}

template <class Grid>
int Solver<Grid>::faceInterp() const {
  return faceInterp_;
}

template <class Grid>
void Solver<Grid>::setCollocatedScheme(const std::string& name) {
  if (name != "ghost" && ghostProjection_) {
    ghostProjection_ = false;  // scheme transition: drop the ghost before selecting a face mode
    gpNRows_ = -1;
  }
  if (name == "gauge-exact") {
    setFaceInterp(9);
  } else if (name == "plain") {
    setFaceInterp(0);
  } else if (name == "embed") {
    setFaceInterp(7);
  } else if (name == "ghost") {
    // The fluid-only constraint scheme (route 2b, 2026-08): binary-openness divergence +
    // directional closures + gauge-exact gradient. Clean-protocol record: family-free
    // (m1 -> 1e-5 monotone, |P| frozen), NO Layer-1 instability, C2 across dt = 60..1e20 with
    // no stabilizer, both-bed ladders -1.4% -> +0.22% (R=8..24; its own small plateau), Z&H
    // anchor -0.018% at N=128. Costs: BiCGStab (~2.3-2.7x pressure stage; star preconditioner
    // planned), ~1.6 KB/cell overlay (caps single-GPU size), fragmentation guard. The (1,2)
    // mixed mode stays quarantined (march-unstable on >2000-sphere beds).
    setGhostProjection(true, 2, 2);
  } else
    throw std::runtime_error(
        "set_collocated_scheme: expected 'ghost', 'gauge-exact', 'plain' or 'embed', got '" + name +
        "'");
}

template <class Grid>
void Solver<Grid>::setRotationalPressure(bool on) {
  rotationalP_ = on;
}

template <class Grid>
void Solver<Grid>::setRotationalWeight(double w) {
  rotWeight_ = w;
}

template <class Grid>
void Solver<Grid>::setRotationalWallWeight(double w0) {
  rotWallW_ = w0;
}

template <class Grid>
void Solver<Grid>::setApertureOrder(int order) {
  requireNoGeometry("set_aperture_order");
  if (order < 1 || order > 2)
    throw std::runtime_error("set_aperture_order: order must be 1 or 2");
  apertureOrder_ = order;
}

template <class Grid>
int Solver<Grid>::apertureOrder() const {
  return apertureOrder_;
}

template <class Grid>
void Solver<Grid>::setAdvectionWallVelocity(bool on) {
  advWallVel_ = on;
}

template <class Grid>
bool Solver<Grid>::advectionWallVelocity() const {
  return advWallVel_;
}

template <class Grid>
void Solver<Grid>::setCommAvoiding(int mask) {
  if (distributed_)
    throw std::runtime_error(
        "set_comm_avoiding: call BEFORE init_mpi -- the momentum half is latched with the halo "
        "topology");
  caMode_ = mask;
  mg_.setCommAvoiding(mask);
}

template <class Grid>
int Solver<Grid>::commAvoiding() const {
  return caMode_;
}

template <class Grid>
void Solver<Grid>::setMultigridAspectThreshold(double theta) {
  if (!(theta > 1.0))
    throw std::runtime_error("set_multigrid_aspect_threshold: theta must be > 1");
  aspectTheta_ = theta;
  mg_.setAspectThreshold(theta);
  vmg_.setAspectThreshold(theta);
}

template <class Grid>
double Solver<Grid>::multigridAspectThreshold() const {
  return aspectTheta_;
}

template <class Grid>
void Solver<Grid>::setPressureStrict(bool on) {
  mg_.setStrictPressure(on);
}

template <class Grid>
bool Solver<Grid>::pressureStrict() const {
  return mg_.strictPressure();
}

template <class Grid>
void Solver<Grid>::setPressureCoarseGhost(bool on) {
  mg_.setCoarseGhost(on);
}

template <class Grid>
bool Solver<Grid>::pressureCoarseGhost() const {
  return mg_.coarseGhost();
}

template <class Grid>
void Solver<Grid>::setPressureBottomExtent(int cells) {
  mg_.setAgglomerationExtent(cells);
}

template <class Grid>
int Solver<Grid>::pressureBottomExtent() const {
  return mg_.agglomerationExtent();
}

template <class Grid>
void Solver<Grid>::setDecomposition(int levels, double maxImbalance) {
  if (distributed_)
    throw std::runtime_error(
        "set_decomposition: call BEFORE init_mpi -- the decomposition is built there");
  decompLevels_ = levels;
  decompMaxImbalance_ = maxImbalance;
}

template <class Grid>
int Solver<Grid>::decompositionLevels() const {
  return decompLevels_;
}

template <class Grid>
double Solver<Grid>::decompositionMaxImbalance() const {
  return decompMaxImbalance_;
}

template <class Grid>
void Solver<Grid>::setFluidOnlyConstraint(int mode) {
  requireNoGeometry("set_fluid_only_constraint");
  if (mode < 0 || mode > 2)
    throw std::runtime_error("set_fluid_only_constraint: mode must be 0, 1 or 2");
  fluidOnlyMode_ = mode;
  if (mode != 0)
    colSchemeAuto_ = false;  // mechanism instruments run on the aperture rails, not AUTO-ghost
}

template <class Grid>
void Solver<Grid>::setRotationalFilter(bool on, double eps) {
  rotFilter_ = on;
  rotFilterEps_ = eps;
}

template <class Grid>
void Solver<Grid>::seedFaceFieldFromCells() {
  if constexpr (Grid::collocated) {
    for (int c = 0; c < 3; ++c)
      fillVelGhosts(c, 0);
    if (faceInterp_ == 5 || faceInterp_ == 7)
      centerToFaceWallAware(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u),
                            CCConst(sdf_), CCConst(xcx_), CCConst(xcy_), CCConst(xcz_), true, e_,
                            G);
    else
      centerToFace(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), e_, G);
    faceFieldValid_ = true;
  }
}

template <class Grid>
void Solver<Grid>::uploadVelocity(const std::vector<double>& uu, const std::vector<double>& vv,
                                  const std::vector<double>& ww) {
  const std::vector<double>* src[3] = {&uu, &vv, &ww};
  std::vector<double> scaled[3];
  for (int c = 0; c < 3; ++c) {
    const double k = u_.velToInt(c);
    if (k == 1.0)
      continue;
    scaled[c].resize(src[c]->size());
    for (std::size_t i = 0; i < src[c]->size(); ++i)
      scaled[c][i] = (*src[c])[i] * k;
    src[c] = &scaled[c];
  }
  CCExec space;
  const int ex = e_.x, ey = e_.y, nx = nx_, ny = ny_, nz = nz_, g = G;
  for (int c = 0; c < 3; ++c) {
    // Upload the inner field once, write it into the inner cells on device, then refresh the
    // periodic ghosts (G4) — the old path mirrored the field down, looped on host, and copied
    // back up.
    CCField din("peclet::flow::vel_in_d", static_cast<std::size_t>(nx_) * ny_ * nz_);
    Kokkos::deep_copy(
        din,
        Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            src[c]->data(), src[c]->size()));
    CCField u = C[c].u;
    Kokkos::parallel_for(
        "peclet::flow::upload_velocity",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          u((long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey) =
              din((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
        });
    fillGhosts(C[c].u);
  }
  seedFaceFieldFromCells();  // ISSUES sweep item 5 (collocated only; a no-op staggered)
}

template <class Grid>
int Solver<Grid>::nx() const {
  return nx_;
}

template <class Grid>
int Solver<Grid>::ny() const {
  return ny_;
}

template <class Grid>
int Solver<Grid>::nz() const {
  return nz_;
}

template <class Grid>
bool Solver<Grid>::implicitAdv() const {
  return advect_ && (implicitFou_ || (hasBc_ && (!useVelocityMg_ || hasSolid_)));
}

template <class Grid>
bool Solver<Grid>::bcStencilPath() const {
  return hasBc_ &&
         (hasSolid_ || (!useVelocityMg_ && (implicitAdv() || varProps_ || varRho_ || hasDrag_)));
}

template <class Grid>
bool Solver<Grid>::mixedVelocityMg() const {
  return hasBc_ && useVelocityMg_ && hasSolid_ && !varProps_ && !varRho_ && !hasDrag_;
}

template <class Grid>
long Solver<Grid>::strideOf(int c) const {
  return (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
}

template <class Grid>
void Solver<Grid>::rebuildStencils() {
  // Per-axis viscous coefficient b_a = mu' * w_a (doc/anisotropic_metric.md §2); w == 1.0 on
  // the isotropic path, so this is literally `beta = mu_` there.
  const double idiag = rho_ / dt_, bx = mu_ * u_.w[0], by = mu_ * u_.w[1], bz = mu_ * u_.w[2];
  if (varProps_)
    fillMuGhosts();  // face means read mu at i +- stride (boundary inner cells -> ghosts)
  if (varRho_)
    fillPropGhosts(rhoField_);
  for (int c = 0; c < 3; ++c) {
    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    if (varProps_ || effVarRho())
      ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x,
                           e_.y, e_.z, G, makeFaceProps(c), u_.w[0], u_.w[1], u_.w[2]);
    else
      ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x, e_.y,
                        e_.z, bx, by, bz, idiag);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
    if (hasDrag_)
      addDragDiagonal(c);
  }
}

template <class Grid>
void Solver<Grid>::copyInner(CCField dst, C3 de, int dg, CCConst src, C3 se, int sg) {
  CCExec space;
  const int NX = nx_, NY = ny_;
  Kokkos::parallel_for(
      "peclet::flow::copyInner", Kokkos::RangePolicy<CCExec>(space, 0, (long)nx_ * ny_ * nz_),
      KOKKOS_LAMBDA(long c) {
        const int ix = (int)(c % NX), iy = (int)((c / NX) % NY), iz = (int)(c / ((long)NX * NY));
        const long di =
            (long)(ix + dg) + (long)(iy + dg) * de.x + (long)(iz + dg) * (long)de.x * de.y;
        const long si =
            (long)(ix + sg) + (long)(iy + sg) * se.x + (long)(iz + sg) * (long)se.x * se.y;
        dst(di) = src(si);
      });
}

template <class Grid>
void Solver<Grid>::copyBlockShifted(CCField dst, C3 de, CCConst src, C3 se, int off) {
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::copyBlockShifted",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {de.x, de.y, de.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long di = (long)x + (long)y * de.x + (long)z * (long)de.x * de.y;
        const long si =
            (long)(x + off) + (long)(y + off) * se.x + (long)(z + off) * (long)se.x * se.y;
        dst(di) = src(si);
      });
}

template <class Grid>
void Solver<Grid>::fillGhosts(CCField f) {
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    velDev_->exchange(f);
    return;
  }
#endif
  fillAxis(f, 0);
  fillAxis(f, 1);
  fillAxis(f, 2);
}

template <class Grid>
void Solver<Grid>::fillGhostsFaces(CCField f) {
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    velDev_->exchange(f);
    return;
  }  // halo gives all ghosts; the 7-pt smoother uses the faces
#endif
  CCExec space;
  C3 e = e_;
  const int Nx = nx_, Ny = ny_, Nz = nz_;
  const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
  CCField ff = f;
  Kokkos::parallel_for(
      "peclet::flow::ibm_facefill", Kokkos::RangePolicy<CCExec>(space, 0, (long)nx_ * ny_ * nz_),
      KOKKOS_LAMBDA(long n) {
        const int ix = (int)(n % Nx), iy = (int)((n / Nx) % Ny), iz = (int)(n / ((long)Nx * Ny));
        const long i = (long)(ix + G) * sx + (long)(iy + G) * sy + (long)(iz + G) * sz;
        if (ix < G)
          ff(i + (long)Nx * sx) = ff(i);
        else if (ix >= Nx - G)
          ff(i - (long)Nx * sx) = ff(i);
        if (iy < G)
          ff(i + (long)Ny * sy) = ff(i);
        else if (iy >= Ny - G)
          ff(i - (long)Ny * sy) = ff(i);
        if (iz < G)
          ff(i + (long)Nz * sz) = ff(i);
        else if (iz >= Nz - G)
          ff(i - (long)Nz * sz) = ff(i);
      });
}

template <class Grid>
void Solver<Grid>::fillAxis(CCField f, int axis) {
  CCExec space;
  C3 e = e_;
  int N3[3] = {nx_, ny_, nz_};
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, e.x, (long)e.x * e.y};
  const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int N = N3[a];
  CCField ff = f;
  Kokkos::parallel_for(
      "peclet::flow::ibm_pfill",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) {
          ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
          ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
        }
      });
}

template <class Grid>
bool Solver<Grid>::advWallInputs() const {
  return advWallVel_ && !Grid::collocated && hasScene_ && hasMotion_ && advect_ &&
         uBc_[0].extent(0) == n_ && C[0].mask.extent(0) == n_;
}

template <class Grid>
void Solver<Grid>::buildAdvInputs() {
  if (!advWallInputs())
    return;
  CCExec space;
  for (int c = 0; c < 3; ++c) {
    if (uwAdv_[c].extent(0) != n_)
      uwAdv_[c] = CCField("uwAdv", n_);
    Kokkos::deep_copy(uwAdv_[c], C[c].u);
    CCField a = uwAdv_[c];
    CCConst m = CCConst(C[c].mask), w = CCConst(uBc_[c]);
    Kokkos::parallel_for(
        "peclet::flow::adv_wall_inputs", Kokkos::RangePolicy<CCExec>(space, 0, (long)n_),
        KOKKOS_LAMBDA(long i) {
          if (m(i) <= 0.5)
            return;
          a(i) = w(i);
        });
  }
  space.fence();
}

template <class Grid>
CCConst Solver<Grid>::advVelView(int c) const {
  return advWallInputs() ? CCConst(uwAdv_[c]) : CCConst(C[c].u);
}

template <class Grid>
bool Solver<Grid>::ensureAdvStash(int c, bool adv) {
  const bool want = !Grid::collocated && hasScene_ && adv;
  if (want && advRhs_[c].extent(0) != n_)
    advRhs_[c] = CCField("advRhs", n_);
  haveAdvRhs_ = want;
  return want;
}

template <class Grid>
void Solver<Grid>::buildAdvStencil(int c) {
  // b_a = mu' * w_a (doc/anisotropic_metric.md §2); the FOU part is index-native and unchanged.
  const double idiag = rho_ / dt_, fouw = rho_, bx = mu_ * u_.w[0], by = mu_ * u_.w[1],
               bz = mu_ * u_.w[2];
  C3 e = e_;
  ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y, e.z,
                    bx, by, bz, idiag);
  CCExec space;
  FV AC = C[c].AC, AW = C[c].AW, AE = C[c].AE, AS = C[c].AS, AN = C[c].AN, AB = C[c].AB,
     AT = C[c].AT;
  CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2);  // A0: wall-aware inputs
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "advstencil", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
               czp = AT(i);
        sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
        Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
        AC(i) = (MReal)cC;
        AW(i) = (MReal)cxm;
        AE(i) = (MReal)cxp;
        AS(i) = (MReal)cym;
        AN(i) = (MReal)cyp;
        AB(i) = (MReal)czm;
        AT(i) = (MReal)czp;
      });

  Kokkos::deep_copy(C[c].rscale, 1.0);
  Kokkos::deep_copy(C[c].inhom, 0.0);
  ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                   C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
  if (hasDrag_)
    addDragDiagonal(c);
}

template <class Grid>
void Solver<Grid>::buildAdvStencilVar(int c) {
  C3 e = e_;
  if (c == 0) {
    if (varProps_)
      fillMuGhosts();
    if (varRho_)
      fillPropGhosts(rhoField_);
    if (!varRho_ && porous_ && porousCons_)
      updateEpsRho();  // eps ghosts are driver-filled; whole-block product has valid ghosts
  }
  ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y, e.z,
                       G, makeFaceProps(c), u_.w[0], u_.w[1], u_.w[2]);
  CCExec space;
  FV AC = C[c].AC, AW = C[c].AW, AE = C[c].AE, AS = C[c].AS, AN = C[c].AN, AB = C[c].AB,
     AT = C[c].AT;
  CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2);  // A0: wall-aware inputs
  const bool vr = effVarRho();
  const double rhoC = rho_;
  CCConst rf = vr ? CCConst(effRhoField()) : CCConst();
  const long sc = strideOf(c);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "advstencil_var", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
               czp = AT(i);
        sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
        const double fouw = vr ? 0.5 * (rf(i) + rf(i - sc)) : rhoC;
        Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
        AC(i) = (MReal)cC;
        AW(i) = (MReal)cxm;
        AE(i) = (MReal)cxp;
        AS(i) = (MReal)cym;
        AN(i) = (MReal)cyp;
        AB(i) = (MReal)czm;
        AT(i) = (MReal)czp;
      });
  Kokkos::deep_copy(C[c].rscale, 1.0);
  Kokkos::deep_copy(C[c].inhom, 0.0);
  ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                   C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
  if (hasDrag_)
    addDragDiagonal(c);
}

template <class Grid>
double Solver<Grid>::maxAbsDiffInner(CCConst a, CCConst b) {
  CCExec space;
  C3 e = e_;
  double m = 0;
  Kokkos::parallel_reduce(
      "maxdiff",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double d = Kokkos::fabs(a(i) - b(i));
        if (d > acc)
          acc = d;
      },
      Kokkos::Max<double>(m));
  return m;
}

template <class Grid>
template <class Fill, class Color, class ColorDu>
void Solver<Grid>::velSweepLoop(Fill&& fill, Color&& sweepColor, ColorDu&& sweepColorDu,
                                std::function<double()> resid, double bnorm) {
  double du0 = 0.0;
  int used = velIters_;
  const double vtol = velocityResidualTolerance();
  const bool useRes = vtol > 0.0 && resid;
  auto gmax = [&](double v) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = 0.0;
      MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
      return g;
    }
#endif
    return v;
  };
  double scale = 0.0, rPrev = -1.0;
  if (useRes) {
    // The convergence scale from the initial residual: forcing may enter through a Dirichlet
    // ghost (inflow), not b, hence max(|b|, |A u|). NO early return on a converged warm start:
    // the projection needs u* to carry the O(rtol) response of the momentum equation (the
    // hydrostatic acid test drifts by 1e-8 in dP/dz if the solve is skipped), so at least one
    // sweep always runs -- a negligible cost against the 8-9 the solve typically takes.
    fill();
    (void)gmax(resid());
    scale = std::max(gmax(bnorm), gmax(lastAxNorm_));
  }
  for (int it = 0; it < velIters_; ++it) {
    fill();
    sweepColor(0);
    fill();
    if (useRes) {
      sweepColor(1);
      // check on the first sweep, then every 4th (a residual costs about half a sweep)
      if (it == 0 || (it + 1) % 4 == 0 || it + 1 == velIters_) {
        fill();
        const double r = gmax(resid());
        const double ratio = scale > 0 ? r / scale : 0.0;
        // Round-off floor / stagnation guard: a residual at ~1e-14 of the scale, or one that no
        // longer decreases between checks, cannot be improved by more sweeps -- stop rather than
        // chase noise to the cap (an exact case such as the hydrostatic column lands here).
        const bool floor = r <= 1e-14 * scale || (rPrev >= 0.0 && r >= rPrev);
        rPrev = r;
        if (r <= vtol * scale || floor || it + 1 == velIters_) {
          lastMomentumResid_ = std::max(lastMomentumResid_, ratio);  // EXIT ratio per component
          used = it + 1;
          break;
        }
      }
    } else if (velTol_ > 0.0) {
      double du = sweepColorDu(1);
#ifdef PECLET_FLOW_MPI
      if (distributed_) {
        double gd = 0.0;
        MPI_Allreduce(&du, &gd, 1, MPI_DOUBLE, MPI_MAX, comm_);
        du = gd;
      }
#endif
      if (it == 0)
        du0 = du;
      if (it + 1 >= velMinIters_ && du <= velTol_ * du0) {
        used = it + 1;
        break;
      }
    } else {
      sweepColor(1);
    }
  }
  lastMomentumSweeps_ += used;
}

template <class Grid>
VelocityMG::Comm Solver<Grid>::vmgComm() const {
#ifdef PECLET_FLOW_MPI
  return comm_;
#else
  return nullptr;
#endif
}

template <class Grid>
double Solver<Grid>::finishResidual(int c) {
  if (hasBc_) {
    const int t = bc_[2 * c];
    if ((t == 1 || t == 2 || t == 4) && touchesGlobalFace(2 * c))
      zeroPlane(velRes_, e_, c, G);
  }
  lastAxNorm_ = peclet::flow::maxAbsDiffInner(CCConst(C[c].b), CCConst(velRes_), e_, G);
  return maxAbsInner(CCConst(velRes_), e_, G);
}

template <class Grid>
std::function<double()> Solver<Grid>::stencilResidual(int c, bool exchange) {
  if (velocityResidualTolerance() <= 0.0)
    return nullptr;
  if (velRes_.extent(0) != n_)
    velRes_ = CCField("velRes", n_);
  return [this, c, exchange]() {
#ifdef PECLET_FLOW_MPI
    if (exchange && distributed_)
      velDev_->exchange(C[c].u);
#else
    (void)exchange;
#endif
    residualVarPin(velRes_, CCConst(C[c].u), CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                   FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                   CCConst(C[c].mask), e_, G);
    return finishResidual(c);
  };
}

template <class Grid>
std::function<double()> Solver<Grid>::constCoeffResidual(int c, double bx, double by, double bz,
                                                         double Ac) {
  if (velocityResidualTolerance() <= 0.0)
    return nullptr;
  if (velRes_.extent(0) != n_)
    velRes_ = CCField("velRes", n_);
  const bool an = u_.aniso;
  return [this, c, bx, by, bz, Ac, an]() {
    const I3 e{e_.x, e_.y, e_.z};
    diffResidual(velRes_, CCConst(C[c].u), CCConst(C[c].b), e, G, bx, by, bz, Ac,
                 CCConst(bcDcorr_[c]), an);
    return finishResidual(c);
  };
}

template <class Grid>
double Solver<Grid>::stencilBnorm(int c) {
  return velocityResidualTolerance() > 0.0 ? maxAbsInner(CCConst(C[c].b), e_, G) : 0.0;
}

template <class Grid>
void Solver<Grid>::smoothComp(int c) {
  if constexpr (Grid::collocated) {
    if (hasBc_) {  // collocated domain BC: the (all-fluid) IBM diffusion stencil + cell-centered
                   // wall
      // reflection ghosts refreshed each colour (explicit no-slip; no fold). Converges to the
      // wall value.
      velSweepLoop([&] { fillVelGhostsTo(C[c].u, c, 0); },
                   [&](int col) {
                     ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                         FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                         FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
                   },
                   [&](int col) {
                     return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC),
                                                  FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS),
                                                  FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                                                  CCConst(C[c].mask), e_, og_, G, col);
                   },
                   stencilResidual(c, /*exchange=*/false), stencilBnorm(c));
      return;
    }
  }
  if (mixedVelocityMg()) {
    // MIXED: immersed solid + domain BCs (the packed bed with an inlet/outlet). Fine = the sharp
    // cut-cell stencil, solid pin, clean-fluid exclude + held-face exclude; coarse = staircase
    // Helmholtz + domain-face folds (VelocityMG::setStaircaseBc). The BC hook re-imposes the
    // level-0 velocity BC after every ghost fill, exactly as the RB-GS path's fillVelGhosts(c,1).
    const Off3 off = Grid::offset(c);
    ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, off, u_.aniso, u_.hp[0], u_.hp[1], u_.hp[2], u_.w[0],
               u_.w[1], u_.w[2]);
    ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, off);
    vmg_.setFineStencil(FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                        FPC(C[c].AB), FPC(C[c].AT));
    // coarse = staircase Helmholtz (+ FOU from the restricted advecting velocity when the fine
    // stencil carries implicit advection) + domain-face folds
    vmg_.setStaircaseBc(c, CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_,
                        rho_ / dt_, 0.5, /*upwind=*/implicitAdv(), rho_);
    // fold=0: level 0 is the UNFOLDED cut-cell stencil, so its wall ghosts are reflections
    // (exactly the bcStencilPath RB-GS convention); the folded coarse levels hold theirs at 0.
    fillVelGhosts(c, 0);
    vmg_.setBcApplyL0([this, c](CCField x) { applyVelocityBcCompTo(x, c, 0, true); });
    lastMomentumSweeps_ += vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_,
                                      vmgComm(), velocityResidualTolerance());
    lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
    maskVelocity(c);
    return;
  }
  if (bcStencilPath()) {
    // Domain BCs solved with the Robust-Scaled cut-cell / FOU stencil (built by setSolid /
    // buildAdvStencil) while refreshing the domain-BC ghosts each colour -- explicit walls/inflow
    // (reflection, fold=0) + outflow zero-gradient. Mirrors the collocated path above. Used for
    // an immersed solid (cut-cell no-slip in the operator) and/or implicit advection (FOU upwind
    // in the stencil -> stable at large dt). The const-coeff fold smoothers below are all-fluid,
    // diffusion-only: they ignore the solid AND run advection explicitly (CFL-limited).
    velSweepLoop([&] { fillVelGhostsTo(C[c].u, c, 0); },
                 [&](int col) {
                   ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                       FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                       FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
                 },
                 [&](int col) {
                   return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                                FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                                FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_,
                                                og_, G, col);
                 },
                 stencilResidual(c), stencilBnorm(c));
    return;
  }
  if (hasBc_ &&
      useVelocityMg_) {  // domain-BC velocity multigrid: const-coeff aniso op + no-slip/inflow/
    // outflow boundary fold on every level (CUDA setDiffusionConstAllLevels +
    // setDiffusionBoundaryFold).
    vmg_.setDomainBcOp(c, mu_, rho_ / dt_);  // per component (the fold is component-dependent)
    fillVelGhosts(
        c, 1);  // set the level-0 boundary ghosts (wall fold=0, inflow value, outflow zero-grad)
    // Re-impose the velocity BC on the vel-MG's level-0 iterate each colour/residual (the
    // const-coeff smoother updates the held Dirichlet faces) -> the vel-MG converges to the RB-GS
    // fixed point (not the ~2% drift CUDA's vmg leaves at the boundary corners).
    // (the hook applies the BC only: VelocityMG::fill owns the periodic wrap / halo exchange)
    vmg_.setBcApplyL0([this, c](CCField x) { applyVelocityBcCompTo(x, c, 1, true); });
    lastMomentumSweeps_ += vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_,
                                      vmgComm(), velocityResidualTolerance());
    lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
    return;
  }
  if (hasBc_) {  // domain-BC (no immersed solid): CUDA's double const-coeff diff_k + dcorr fold
    // og_ is the GLOBAL block origin (red-black parity); {0,0,0} single-rank, so byte-identical
    // there. It was hard-coded {0,0,0} here, which swaps the colours on any rank whose block
    // origin has odd parity — the only smoother in the file that did not carry og_.
    const I3 e{e_.x, e_.y, e_.z}, og{og_.x, og_.y, og_.z};
    // b_a = mu' * w_a; Ac = rho/dt + 2*((bx+by)+bz) in EXACTLY that association order, which is
    // fl(6*mu_) at bx==by==bz (doc/anisotropic_metric.md §2, trap 3).
    const double bx = mu_ * u_.w[0], by = mu_ * u_.w[1], bz = mu_ * u_.w[2];
    const double Ac = rho_ / dt_ + 2.0 * ((bx + by) + bz);
    const bool an = u_.aniso;
    velSweepLoop([&] { fillVelGhosts(c, 1); },  // re-impose wall faces (fold) before each color
                 [&](int col) {
                   diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, bx, by, bz, Ac, col,
                                   CCConst(bcDcorr_[c]), an);
                 },
                 [&](int col) {
                   return diffSmoothColorDu(C[c].u, CCConst(C[c].b), e, og, G, bx, by, bz, Ac, col,
                                            CCConst(bcDcorr_[c]), an);
                 },
                 constCoeffResidual(c, bx, by, bz, Ac), stencilBnorm(c));
    return;
  }
  if (useVelocityMg_) {  // IBM velocity multigrid: fine = sharp As_[c]; coarse op depends on the
                         // regime.
    vmg_.setFineStencil(FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                        FPC(C[c].AB), FPC(C[c].AT));
    if (implicitFou_ && advect_) {
      // UPWIND-CONVECTIVE coarse op (advection-dominated): aniso const-coeff diffusion + dt*FOU
      // from the restricted advecting velocity (restrictAdvVelocities ran once in step()). No pin
      // / no exclude mask.
      vmg_.buildUpwindCoarse(c, mu_, rho_ / dt_, rho_);
    } else {
      // STAIRCASE coarse op (diffusion-only): theta classification + clean-fluid exclude (exact
      // == RB-GS).
      const Off3 off =
          Grid::offset(c);  // velocity-unknown placement (staggered: -1/2 face; collocated: 0)
      ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, off, u_.aniso, u_.hp[0], u_.hp[1], u_.hp[2], u_.w[0],
                 u_.w[1], u_.w[2]);
      ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, off);
      vmg_.setStaircase(CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_, rho_ / dt_,
                        0.5);
    }
    lastMomentumSweeps_ += vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_,
                                      vmgComm(), velocityResidualTolerance());
    lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
    maskVelocity(c);  // re-impose no-slip at solid (the masked solve leaves them at the pin value)
    return;
  }
  // IBM / periodic: Robust-Scaled cut-cell stencil (float). The 7-point smoother reads faces
  // only -> the fused 1-kernel face fill suffices.
#ifdef PECLET_FLOW_MPI
  if (distributed_ && caMomentum_) {
    // Communication-avoiding pair (the momentum counterpart of CutcellMG::smooth's CA path):
    // ONE 2-deep exchange per red-black pair instead of one per colour — the velocity block is
    // g=2 already. Colour 0 overlaps the exchange with the interior sweep, then sweeps the
    // boundary shell PLUS the 1-deep ghost ring, redundantly recomputing the neighbour's
    // boundary cells from the same operands the neighbour uses (2-deep u ghosts; the ring rows
    // of the stencil/mask/rhs are exchanged below, so they are the owner's bit-exact values).
    // Colour 1 then sweeps with NO exchange: its boundary cells read only colour-0 ring cells,
    // which equal what a fresh exchange would have delivered — bit-identical at half the halo
    // events. The tolerance stop's colour-1 kernel is the ORIGINAL full-inner fused reduction
    // (host pencil form intact), so du matches the blocking path exactly.
    // Stencil + mask ring exchange: once per (re)build. The per-step machinery (implicit-FOU
    // Picard rebuilds, variable properties, implicit drag, eps-conservative porous) rewrites the
    // stencil every solve, so those paths re-exchange every solve — mirrors the step() rebuild
    // gates; a false positive costs 8 extra exchanges, a false negative would break the np>1
    // bit-exactness (the ring rows would read a stale operator).
    const bool perStepStencil = implicitAdv() || varProps_ || varRho_ || effVarRho() || hasDrag_;
    if (momStencilDirty_[c] || perStepStencil) {
      for (FV* a : {&C[c].AC, &C[c].AW, &C[c].AE, &C[c].AS, &C[c].AN, &C[c].AB, &C[c].AT})
        velDevF_->exchange(*a);
      velDev_->exchange(C[c].mask);
      momStencilDirty_[c] = false;
    }
    velDev_->exchange(C[c].b);  // rhs ring (owner's inner values); fixed over the sweeps
    const C3 lo{G + 1, G + 1, G + 1}, hi{e_.x - G - 1, e_.y - G - 1, e_.z - G - 1};
    const C3 rlo{G - 1, G - 1, G - 1}, rhi{e_.x - G + 1, e_.y - G + 1, e_.z - G + 1};
    const C3 z0{0, 0, 0};
    velSweepLoop(
        [] {},
        [&](int col) {
          if (col == 0) {
            velDev_->exchangeBegin(C[c].u);
            ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                   FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                   FPC(C[c].AT), CCConst(C[c].mask), e_, og_, col, lo, hi, z0, z0);
            velDev_->exchangeEnd(C[c].u);
            ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                   FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                   FPC(C[c].AT), CCConst(C[c].mask), e_, og_, col, rlo, rhi, lo,
                                   hi);
          } else {
            ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                                FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                                CCConst(C[c].mask), e_, og_, G, col);
          }
        },
        [&](int col) {
          return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                       FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                       FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
        },
        stencilResidual(c, /*exchange=*/true), stencilBnorm(c));
    return;
  }
  if (distributed_) {
    // Overlap the per-colour halo with the interior sweep (the momentum counterpart of the MG
    // smoothers' split, 3ace962): post the exchange, sweep the interior cells — whose 7-point
    // stencil reads no ghost — while the messages fly, complete it, then sweep the boundary
    // shell. A colour's cells never read same-colour cells, so interior-then-shell is
    // bit-identical to the blocking exchange + full sweep; the tolerance stop's max-increment
    // combines the two passes by max (order-independent). Only this periodic/IBM path overlaps:
    // the domain-BC paths re-impose ghost BCs each colour and keep the blocking order (the same
    // decision as VelocityMG's overlap). The exchange is posted INSIDE the colour lambda (fill
    // is a no-op) so the packed send values are exactly the blocking call's.
    const C3 ilo{G, G, G}, ihi{e_.x - G, e_.y - G, e_.z - G};
    const C3 lo{G + 1, G + 1, G + 1}, hi{e_.x - G - 1, e_.y - G - 1, e_.z - G - 1};
    const C3 z0{0, 0, 0};
    velSweepLoop(
        [] {},
        [&](int col) {
          velDev_->exchangeBegin(C[c].u);
          ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                                 FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                                 CCConst(C[c].mask), e_, og_, col, lo, hi, z0, z0);
          velDev_->exchangeEnd(C[c].u);
          ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                                 FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                                 CCConst(C[c].mask), e_, og_, col, ilo, ihi, lo, hi);
        },
        [&](int col) {
          velDev_->exchangeBegin(C[c].u);
          const double di = ibmRbgsStencilColorDuBox(
              C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS),
              FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_, col, lo, hi,
              z0, z0);
          velDev_->exchangeEnd(C[c].u);
          const double ds = ibmRbgsStencilColorDuBox(
              C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS),
              FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_, col, ilo, ihi,
              lo, hi);
          return di > ds ? di : ds;
        },
        stencilResidual(c, /*exchange=*/true), stencilBnorm(c));
    return;
  }
#endif
  velSweepLoop([&] { fillGhostsFaces(C[c].u); },
               [&](int col) {
                 ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                     FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                     FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
               },
               [&](int col) {
                 return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                              FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                              FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_,
                                              og_, G, col);
               },
               stencilResidual(c, /*exchange=*/false), stencilBnorm(c));
}

template <class Grid>
void Solver<Grid>::maskVelocity(int c) {
  CCExec space;
  CCField u = C[c].u, m = C[c].mask;
  Kokkos::parallel_for(
      "vmask", Kokkos::RangePolicy<CCExec>(space, 0, n_), KOKKOS_LAMBDA(std::size_t i) {
        if (m(i) > 0.5)
          u(i) = 0.0;
      });
}

template <class Grid>
double Solver<Grid>::minMuInner() {
  CCExec space;
  C3 e = e_;
  CCConst f = CCConst(muField_);
  double m = 1e300;
  Kokkos::parallel_reduce(
      "minmu",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (f(i) < acc)
          acc = f(i);
      },
      Kokkos::Min<double>(m));
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = m;
    MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MIN, comm_);
    m = g;
  }
#endif
  return m;
}

template <class Grid>
double Solver<Grid>::reduceMaxAbsInner(CCConst f) {
  CCExec space;
  C3 e = e_;
  double m = 0;
  Kokkos::parallel_reduce(
      "maxabs",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double a = Kokkos::fabs(f(i));
        if (a > acc)
          acc = a;
      },
      Kokkos::Max<double>(m));
  return m;
}

template <class Grid>
std::vector<double> Solver<Grid>::gatherInner(CCField fld) {
  auto h = Kokkos::create_mirror_view(fld);
  Kokkos::deep_copy(h, fld);
  std::vector<double> out((std::size_t)nx_ * ny_ * nz_);
  for (int z = 0; z < nz_; ++z)
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x)
        out[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_] =
            h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y);
  return out;
}

template <class Grid>
void Solver<Grid>::scatterInner(CCField fld, const std::vector<double>& in) {
  if (in.size() != (std::size_t)nx_ * ny_ * nz_)
    throw std::runtime_error("flow::setField: array size does not match the inner grid");
  auto h = Kokkos::create_mirror_view(fld);
  Kokkos::deep_copy(h, fld);  // preserve existing ghosts
  for (int z = 0; z < nz_; ++z)
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x)
        h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
            in[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_];
  Kokkos::deep_copy(fld, h);
}

template <class Grid>
CCField Solver<Grid>::addField(const std::string& name) {
  if (fields_.has(name))
    return fields_.at(name).data;
  return fields_.add(name, n_, G, peclet::core::Centering::Cell).data;
}

template <class Grid>
bool Solver<Grid>::hasField(const std::string& name) const {
  return fields_.has(name);
}

template <class Grid>
CCField Solver<Grid>::fieldView(const std::string& name) {
  return fields_.at(name).data;
}

template <class Grid>
std::vector<std::string> Solver<Grid>::fieldNames() const {
  return fields_.names();
}

template <class Grid>
void Solver<Grid>::exchangeField(const std::string& name) {
  fillGhosts(fields_.at(name).data);
}

template <class Grid>
void Solver<Grid>::exchangeFieldAdd(const std::string& name) {
#ifdef PECLET_FLOW_MPI
  if (distributed_ && velHalo_) {
    CCField f = fields_.at(name).data;
    auto h = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h, f);
    peclet::core::halo::GridFieldView<double> view{h.data()};
    velHalo_->reverseAdd(view);
    Kokkos::deep_copy(f, h);
  }
#else
  (void)name;
#endif
}

template <class Grid>
std::vector<double> Solver<Grid>::getField(const std::string& name) {
  return gatherInner(fields_.at(name).data);
}

template <class Grid>
void Solver<Grid>::setField(const std::string& name, const std::vector<double>& v) {
  scatterInner(fields_.at(name).data, v);
}

template <class Grid>
std::array<int, 3> Solver<Grid>::blockShape() const {
  return {e_.x, e_.y, e_.z};
}

template <class Grid>
int Solver<Grid>::ghostWidth() const {
  return G;
}

template <class Grid>
std::array<int, 3> Solver<Grid>::globalResolution() const {
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    return {gnx_, gny_, gnz_};
#endif
  return {nx_, ny_, nz_};
}

template <class Grid>
std::array<int, 3> Solver<Grid>::blockOrigin() const {
  return {og_.x, og_.y, og_.z};
}

template <class Grid>
void Solver<Grid>::setRhoFaceHarmonic(bool on) {
  rhoFaceHarmonic_ = on;
}

template <class Grid>
void Solver<Grid>::setPressureExactResidual(bool on) {
  exactResidual_ = on;
  mg_.setExactResidual(on);
}

template <class Grid>
void Solver<Grid>::setOutflowOperatorCoefficient(bool on) {
  outflowOpCoeff_ = on;
}

template <class Grid>
bool Solver<Grid>::outflowOperatorCoefficient() const {
  return outflowOpCoeff_;
}

template <class Grid>
bool Solver<Grid>::pressureExactResidual() const {
  return exactResidual_;
}

template <class Grid>
bool Solver<Grid>::rhoFaceHarmonic() const {
  return rhoFaceHarmonic_;
}

template <class Grid>
void Solver<Grid>::setOutflowRhoCorrection(bool on) {
  outflowRhoCorr_ = on;
}

template <class Grid>
bool Solver<Grid>::outflowRhoCorrection() const {
  return outflowRhoCorr_;
}

template <class Grid>
CCField Solver<Grid>::ensureTarget(const std::string& name) {
  if (name == "force_x" || name == "force_y" || name == "force_z")
    ensureCellForceAll();
  return addField(name);  // idempotent; returns the (now-existing) buffer
}

template <class Grid>
void Solver<Grid>::ensureCellForceAll() {
  static const char* fn[3] = {"force_x", "force_y", "force_z"};
  for (int c = 0; c < 3; ++c)
    cellForce_[c] = addField(fn[c]);  // zero-initialised, registered
  hasCellForce_ = true;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_CORE_HPP
