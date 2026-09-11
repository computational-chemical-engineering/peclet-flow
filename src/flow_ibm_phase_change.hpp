/// @file
/// @brief flow — IbmSolver phase change: interface area, mass flux, energy transport, budget
/// accounting, the pc* interface-tracking and thermal-mask machinery.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_PHASE_CHANGE_HPP
#define PECLET_FLOW_FLOW_IBM_PHASE_CHANGE_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::enablePhaseChange(double rhoG, double rhoL, double hlv) {
  if constexpr (Grid::collocated)
    throw std::runtime_error(
        "enable_phase_change: rungs P0/P1 are STAGGERED-ONLY (the collocated grid carries every "
        "force as a face acceleration and the source deposit has not been composed with it).");
  enableVof();
  if (hasSolid_)
    throw std::runtime_error(
        "enable_phase_change: an immersed solid is out of scope at rungs P0/P1 (the source "
        "deposit and the regression would need the solid-clipped flux polygons of rung V5a's "
        "follow-on). Use an all-fluid set_pressure_geometry.");
  if (vofMomEnabled_)
    throw std::runtime_error(
        "enable_phase_change is not composable with enable_vof_momentum at this rung: both own "
        "the head of the step, and momentum consistency would have to carry the interfacial mass "
        "transfer in its own fluxes.");
  if (!(rhoG > 0.0) || !(rhoL > 0.0))
    throw std::runtime_error("enable_phase_change: both phase densities must be > 0");
  if (!(hlv > 0.0))
    throw std::runtime_error("enable_phase_change: the latent heat h_lv must be > 0");
  pcRhoG_ = rhoG;
  pcRhoL_ = rhoL;
  pcHlv_ = hlv;
  // WO-P23: phase change and WO-R2 item 4's wisp guard are NOT compatible, and the measurement
  // is in the findings. `enable_vof` sets `WyAdvector::wispEps = 1e-8`, which makes the advector
  // treat a cell with `C <= 1e-8` as a PURE phase for reconstruction and flux. The phase-change
  // driver needs the colour it reconstructs a plane from, deposits a source behind and pins a
  // Dirichlet row in to be the colour that is actually advected; over the band
  // `1e-12 < C < 1e-8` the two disagree, and on a CURVED interface that diverges the run — the
  // P3 Scriven bubble at Ja = 0.5 reads `R(t)` error 48 % and trips the study's dt-collapse guard
  // at step 34, against 2.002 % and 80 clean steps with the guard off. Sharing the tolerance
  // (`pcEffInterfaceEps` / `pcEffPureEps`) is necessary but not sufficient: it fixes the planar
  // gates and moves the Scriven blow-up from step 34 to step ~200 of 80-worth of physical time,
  // and only `wispEps = 0` removes it. So phase change turns the guard off, loudly, and
  // `set_vof_wisp_eps` after `enable_phase_change` is the deliberate override.
  setVofWispEps(0.0);
  pcMdot_ = addField("mdot");
  pcSrc_ = addField("pc_source");
  if (pcArea_.extent(0) != n_) {
    pcArea_ = CCField("pc_area", n_);
    pcNrm_[0] = CCField("pc_nx", n_);
    pcNrm_[1] = CCField("pc_ny", n_);
    pcNrm_[2] = CCField("pc_nz", n_);
    pcDep_ = CCField("pc_dep", n_);
    pcTgt_ = CCField("pc_tgt", n_);
    pcCnew_ = CCField("pc_cnew", n_);
    pcDefic_ = CCField("pc_defic", n_);
  }
  pcEnabled_ = true;
}

template <class Grid>
bool Solver<Grid>::phaseChangeEnabled() const {
  return pcEnabled_;
}

template <class Grid>
void Solver<Grid>::setMassFluxUniform(double v) {
  requirePhaseChange("set_mass_flux_uniform");
  Kokkos::deep_copy(pcMdot_, v);
  pcThermal_ = false;
}

template <class Grid>
void Solver<Grid>::setMassFlux(const std::vector<double>& v) {
  requirePhaseChange("set_mass_flux");
  scatterInner(pcMdot_, v);
  fillPropGhosts(pcMdot_);
  pcThermal_ = false;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeThermal(const std::string& tname, double Tsat, double kg,
                                         double kl, double Rint) {
  requirePhaseChange("set_phase_change_thermal");
  if (!hasScalar(tname))
    throw std::runtime_error("set_phase_change_thermal: no scalar named '" + tname +
                             "' (call add_scalar first)");
  pcTName_ = tname;
  pcTsat_ = Tsat;
  pcKg_ = kg;
  pcKl_ = kl;
  pcRint_ = Rint;
  pcThermal_ = true;
  scalarDirichletMask(tname);  // allocate the per-cell Dirichlet mask + value fields
  pcUpdateThermalMask();
}

template <class Grid>
void Solver<Grid>::setPhaseChangeThermalOff() {
  pcThermal_ = false;
}

template <class Grid>
void Solver<Grid>::setPhaseChangePlaneDirichlet(bool on) {
  pcPlaneDir_ = on;
}

template <class Grid>
bool Solver<Grid>::phaseChangePlaneDirichlet() const {
  return pcPlaneDir_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeQuadraticFit(bool on) {
  pcQuadFit_ = on;
}

template <class Grid>
bool Solver<Grid>::phaseChangeQuadraticFit() const {
  return pcQuadFit_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeArea(int mode) {
  if (mode < 0 || mode > 7)
    throw std::runtime_error(
        "set_phase_change_area: mode must be 0 (PLIC/MYC), 1 (cascade metric), 2 (cascade "
        "normal), 3 (cascade footprint) or 4-7 (the joined marching-tetrahedra sheet: 4 colour/"
        "centroid, 5 colour/split, 6 PLIC-distance/centroid, 7 PLIC-distance/split)");
  pcAreaMode_ = mode;
}

template <class Grid>
int Solver<Grid>::phaseChangeArea() const {
  return pcAreaMode_;
}

template <class Grid>
double Solver<Grid>::vofInterfaceArea() {
  if (!vofEnabled_)
    throw std::runtime_error("vof_interface_area: VoF is not enabled (call enable_vof first)");
  bridgeColourToVof();
  double a = 0.0;
  if (pcAreaMode_ == vof::kAreaPlic) {
    const I3 e3 = I3{e3_.x, e3_.y, e3_.z};
    const int g = kVofG;
    const long sy = e3.x, sz = (long)e3.x * e3.y;
    CCConst c = CCConst(vofAdv_.colour());
    const double eps = pcEffInterfaceEps();
    Kokkos::parallel_reduce(
        "peclet::flow::vof_area_plic",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {g, g, g},
                                                       {g + nx_, g + ny_, g + nz_}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e3.x + (long)z * sz;
          if (!vof::pcIsInterfacial(c(i), eps))
            return;
          double st[27];
          for (int kk = -1; kk <= 1; ++kk)
            for (int jj = -1; jj <= 1; ++jj)
              for (int ii = -1; ii <= 1; ++ii)
                st[vof::plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
          double m[3];
          vof::mycNormal(st, m);
          acc += vof::plicArea(m[0], m[1], m[2], vof::plicAlpha(m[0], m[1], m[2], c(i)));
        },
        a);
    Kokkos::fence();
  } else {
    a = pcAreaCascadeCompute().area;
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double gsum = 0.0;
    MPI_Allreduce(&a, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm_);
    a = gsum;
  }
#endif
  return a;
}

template <class Grid>
vof::VofInterfaceArea::Stats Solver<Grid>::pcAreaCascadeCompute() {
  if (pcAreaMode_ >= vof::kAreaMcColour) {
    if (!pcAreaMc_.ready())
      pcAreaMc_.init(nx_, ny_, nz_, kVofG);
    pcAreaMc_.interfaceEps = pcEffInterfaceEps();
    const auto m = pcAreaMc_.compute(vofAdv_.colour(), pcAreaMode_);
    vof::VofInterfaceArea::Stats s;
    s.interfacial = m.cells;
    s.hf = m.cells - m.orphanCells;  // the cells the flux integral can actually use
    s.hfMixed = 0;
    s.pv = 0;
    s.noEstimate = m.orphanCells;
    s.area = m.area;
    pcMcOrphanArea_ = m.orphanArea;
    return s;
  }
  pcMcOrphanArea_ = 0.0;
  if (!pcAreaC_.ready())
    pcAreaC_.init(nx_, ny_, nz_, kVofG);
  pcAreaC_.interfaceEps = pcEffInterfaceEps();
  return pcAreaC_.compute(vofAdv_.colour(), pcAreaMode_);
}

template <class Grid>
SField Solver<Grid>::pcAreaCascadeField() const {
  return (pcAreaMode_ >= vof::kAreaMcColour) ? pcAreaMc_.area() : pcAreaC_.area();
}

template <class Grid>
void Solver<Grid>::setPhaseChangeEnergy(double rcpGas, double rcpLiquid) {
  requirePhaseChange("set_phase_change_energy");
  if (!pcThermal_)
    throw std::runtime_error(
        "set_phase_change_energy: call set_phase_change_thermal first (the consistent transport "
        "is for the energy scalar it names)");
  if (!(rcpGas > 0.0) || !(rcpLiquid > 0.0))
    throw std::runtime_error("set_phase_change_energy: both phase rho*c_p must be > 0");
  pcRcpG_ = rcpGas;
  pcRcpL_ = rcpLiquid;
  if (pcKcell_.extent(0) != n_) {
    pcKcell_ = CCField("pc_k", n_);
    pcRcp_ = CCField("pc_rcp", n_);
  }
  ScalarField& sc = scalarField(pcTName_);
  sc.energy = true;
  sc.kcell = pcKcell_;
  sc.rcp = pcRcp_;
  if (!vofEnergy_.initialized())
    vofEnergy_.init(vofAdv_, pcRcpG_, pcRcpL_);
  else
    vofEnergy_.setPhaseRcp(pcRcpG_, pcRcpL_);
  vofEnergy_.exchange = [this](CCField f) { this->vofExchangeScalar(f); };
  pcEnergy_ = true;
  pcUpdateEnergyProps();
}

template <class Grid>
void Solver<Grid>::setPhaseChangeEnergyMuscl(bool on) {
  vofEnergy_.energyMuscl = on;
}

template <class Grid>
bool Solver<Grid>::phaseChangeEnergyMuscl() const {
  return vofEnergy_.energyMuscl;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeEnergyOff() {
  pcEnergy_ = false;
  if (!pcTName_.empty() && hasScalar(pcTName_))
    scalarField(pcTName_).energy = false;
}

template <class Grid>
bool Solver<Grid>::phaseChangeEnergy() const {
  return pcEnergy_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeBudget(bool on) {
  requirePhaseChange("set_phase_change_budget");
  pcBudgetOn_ = on;
  if (on && pcClsPrev_.extent(0) != n_) {
    pcClsPrev_ = CCField("pc_cls_prev", n_);
    Kokkos::deep_copy(pcClsPrev_, -1.0);  // "no previous step": the first call counts no changes
  }
}

template <class Grid>
bool Solver<Grid>::phaseChangeBudget() const {
  return pcBudgetOn_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeCarryConserve(bool on) {
  requirePhaseChange("set_phase_change_carry_conserve");
  pcCarryConserve_ = on;
  if (on && pcCarrySrc_.extent(0) != n_) {
    pcCarrySrc_ = CCField("pc_carry_src", n_);
    Kokkos::deep_copy(pcCarrySrc_, 0.0);
  }
}

template <class Grid>
bool Solver<Grid>::phaseChangeCarryConserve() const {
  return pcCarryConserve_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeFitCurvature(double kappa) {
  requirePhaseChange("set_phase_change_fit_curvature");
  pcFitKappa_ = kappa / u_.curvToPhys();
}

template <class Grid>
double Solver<Grid>::phaseChangeFitCurvature() const {
  return pcFitKappa_ * u_.curvToPhys();
}

template <class Grid>
void Solver<Grid>::setPhaseChangeEnergyOrder(int order) {
  requirePhaseChange("set_phase_change_energy_order");
  if (order != 1 && order != 2)
    throw std::runtime_error("set_phase_change_energy_order: order must be 1 or 2");
  setPhaseChangeMdotOperator(order == 2);
  setPhaseChangeGfmOrder(order);
  setPhaseChangeCurvatureDistance(order == 2);
  setPhaseChangeCarryConserve(order == 2);
}

template <class Grid>
int Solver<Grid>::phaseChangeEnergyOrder() const {
  return (pcMdotOperator_ && pcGfmOrder_ == 2 && pcCurvDist_ && pcCarryConserve_) ? 2 : 1;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeMdotOperator(bool on) {
  requirePhaseChange("set_phase_change_mdot_operator");
  if (on && pcRint_ != 0.0 && !pcThermal_)
    throw std::runtime_error("set_phase_change_mdot_operator: needs the thermal mass flux");
  pcMdotOperator_ = on;
  if (on && pcMdotFit_.extent(0) != n_)
    pcMdotFit_ = CCField("pc_mdot_fit", n_);
}

template <class Grid>
bool Solver<Grid>::phaseChangeMdotOperator() const {
  return pcMdotOperator_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeGfmOrder(int order) {
  requirePhaseChange("set_phase_change_gfm_order");
  if (order != 1 && order != 2)
    throw std::runtime_error("set_phase_change_gfm_order: order must be 1 or 2");
  pcGfmOrder_ = order;
}

template <class Grid>
int Solver<Grid>::phaseChangeGfmOrder() const {
  return pcGfmOrder_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeCurvatureDistance(bool on) {
  requirePhaseChange("set_phase_change_curvature_distance");
  pcCurvDist_ = on;
  if (on && pcKappa_.extent(0) != n_)
    pcKappa_ = CCField("pc_kappa", n_);
  pcMaskFresh_ = false;  // the row geometry has to be rebuilt with (or without) the curvature
}

template <class Grid>
bool Solver<Grid>::phaseChangeCurvatureDistance() const {
  return pcCurvDist_;
}

template <class Grid>
void Solver<Grid>::setPhaseChangeDepositFallback(bool on) {
  requirePhaseChange("set_phase_change_deposit_fallback");
  pcDepositFallback_ = on;
}

template <class Grid>
bool Solver<Grid>::phaseChangeDepositFallback() const {
  return pcDepositFallback_;
}

template <class Grid>
double Solver<Grid>::phaseChangeQOperator() const {
  return pcQOperator_;
}

template <class Grid>
double Solver<Grid>::phaseChangeQOrphan() const {
  return pcQOrphan_;
}

template <class Grid>
double Solver<Grid>::phaseChangeCarryDeposited() const {
  return pcCarryDeposited_;
}

template <class Grid>
double Solver<Grid>::phaseChangeCarryLost() const {
  return pcCarryLost_;
}

template <class Grid>
typename Solver<Grid>::PhaseChangeBudget Solver<Grid>::phaseChangeBudgetValues() const {
  return pcBudget_;
}

template <class Grid>
double Solver<Grid>::pcBandDivergence() {
  const C3 e = e_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  CCConst c = CCConst(cField_), U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
  CCConst ox = CCConst(ox_), oy = CCConst(oy_), oz = CCConst(oz_);
  const double eps = pcEffInterfaceEps();
  double m = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_band_div",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!vof::pcIsInterfacial(c(i), eps))
          return;
        const double d = (ox(i + sx) * U(i + sx) - ox(i) * U(i)) +
                         (oy(i + sy) * V(i + sy) - oy(i) * V(i)) +
                         (oz(i + sz) * W(i + sz) - oz(i) * W(i));
        acc = Kokkos::fmax(acc, Kokkos::fabs(d));
      },
      Kokkos::Max<double>(m));
  Kokkos::fence();
  return Kokkos::fmax(m, 0.0);
}

template <class Grid>
void Solver<Grid>::pcUpdateEnergyProps() {
  if (!pcEnergy_)
    return;
  const C3 e = e_;
  CCField kf = pcKcell_, rf = pcRcp_;
  CCConst c = CCConst(cField_);
  const double kg = pcKg_, kl = pcKl_, rg = pcRcpG_, rl = pcRcpL_;
  Kokkos::parallel_for(
      "peclet::flow::pc_energy_props",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double cl = Kokkos::fmin(Kokkos::fmax(c(i), 0.0), 1.0);
        kf(i) = vof::pcPhaseMix(kg, kl, cl);
        rf(i) = vof::pcPhaseMix(rg, rl, cl);
      });
  Kokkos::fence();
  fillPropGhosts(pcKcell_);
  fillPropGhosts(pcRcp_);
}

template <class Grid>
void Solver<Grid>::setDivergenceSource(const std::vector<double>& v) {
  if (pcUser_.extent(0) != n_)
    pcUser_ = addField("div_source");
  scatterInner(pcUser_, v);
  fillPropGhosts(pcUser_);
  pcHasUser_ = true;
}

template <class Grid>
void Solver<Grid>::clearDivergenceSource() {
  pcHasUser_ = false;
}

template <class Grid>
void Solver<Grid>::setDivergenceSink(const std::vector<double>& w) {
  requirePhaseChange("set_divergence_sink");
  if (pcSink_.extent(0) != n_)
    pcSink_ = addField("div_sink");
  scatterInner(pcSink_, w);
  fillPropGhosts(pcSink_);
  const C3 e = e_;
  CCConst wv = CCConst(pcSink_);
  double acc = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_sink_weight",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a) {
        a += wv((long)x + (long)y * e.x + (long)z * (long)e.x * e.y);
      },
      acc);
  Kokkos::fence();
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = 0;
    MPI_Allreduce(&acc, &g, 1, MPI_DOUBLE, MPI_SUM, comm_);
    acc = g;
  }
#endif
  if (!(acc > 0.0))
    throw std::runtime_error("set_divergence_sink: the weights must have a positive global sum");
  pcSinkW_ = acc;
  pcHasSink_ = true;
}

template <class Grid>
void Solver<Grid>::clearDivergenceSink() {
  pcHasSink_ = false;
}

template <class Grid>
void Solver<Grid>::applyPhaseChange(double dtPhysArg) {
  const double dt = dtPhysArg * u_.timeToInt();
  requirePhaseChange("apply_phase_change");
  pcBuildInterface();
  pcScatterSource();
  pcRegress(dt);
  pcUpdateEnergyProps();
  pcUpdateThermalMask();
  // WO-P3f: the a-priori half of the energy-budget instrument. With no energy solve to bracket,
  // `pcBudgetPost` on the CURRENT temperature and the just-rebuilt plane geometry answers the
  // one question an exact-state probe can ask of the energy side: how much heat do the
  // plane-anchored rows draw across this interface, against the `mdot h_lv A_Gamma` the SAME
  // fields make the regression book? Inert unless `set_phase_change_budget(true)` ran.
  if (pcBudgetOn_ && pcThermal_ && pcClsPrev_.extent(0) == n_) {
    ScalarField& sc = scalarField(pcTName_);
    if (sc.dmask.extent(0) == n_)
      pcBudgetPost(sc);
  }
}

template <class Grid>
void Solver<Grid>::phaseChangeStep() {
  if (!pcEnabled_)
    return;
  pcBuildInterface();
  pcScatterSource();
  pcRegress(dt_);
}

template <class Grid>
typename Solver<Grid>::PhaseChangeDiagnostics Solver<Grid>::phaseChangeDiagnostics() {
  requirePhaseChange("phase_change_diagnostics");
  PhaseChangeDiagnostics d = pcDiag_;
  // colour extrema of the CURRENT field (the regression's own boundedness read-out)
  CCConst c = CCConst(cField_);
  const C3 e = e_;
  double mn = 1e300, mx = -1e300;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_extrema",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& lo, double& hi) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        lo = Kokkos::fmin(lo, c(i));
        hi = Kokkos::fmax(hi, c(i));
      },
      Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
  Kokkos::fence();
  d.minC = mn;
  d.maxC = mx;
  d.bandDiv = pcBandDivergence();
  if (pcEnergy_ && vofEnergy_.initialized())
    vofEnergy_.extrema(d.Tmin, d.Tmax);
  return d;
}

template <class Grid>
void Solver<Grid>::pcBuildInterface() {
  // WO-P23 (a defect found in the P0/P1 code): the gradient fit reads the temperature at +-2, so
  // the energy scalar's ghost band has to be VALID here. `set_field` and the coupling drivers
  // write inner cells only, and `advanceScalars` fills the ghosts at its END — so the FIRST
  // `apply_phase_change` / `step` of every run fitted its one-sided gradients against a band of
  // zeros. Measured on the P2 sucking-interface kernel probe (an exact analytic state, no time
  // stepping, quasi-2D so the y/z ghosts are inside the 5^3 stencil): mdot came out
  // **8.16 against the exact 18.48**, a 56 % error that did NOT converge under refinement,
  // because the transverse ghost samples are counted as pure liquid at T = 0 and pull the fit
  // towards zero. One line, and it is a correctness fix, not a tolerance.
  if (pcThermal_)
    scalarFillGhosts(scalarField(pcTName_));
  // WO-P3g item 1: the operator-flux `mdot` reads the very rows the energy solve uses, so the
  // Dirichlet mask, the plane geometry and (item 3) the curvature must belong to the colour this
  // build reconstructs from. In the COUPLED step they already do — `pcUpdateThermalMask` runs at
  // the bottom of `step()` on exactly this colour, and nothing has touched it since — so this
  // costs nothing there. After `set_vof` / `set_field`, i.e. in the one-shot `apply_phase_change`
  // probe, they do not, and rebuilding here is what makes the probe measure the same operator the
  // run does.
  if (pcMdotOperator_ && pcThermal_ && !pcMaskFresh_) {
    pcUpdateEnergyProps();
    pcUpdateThermalMask();
  }
  // WO-P3c: the cascade-consistent interfacial area, computed on the colour field's OWN g = 3
  // block (the height columns reach +-3, which the G = 2 phase-change block does not carry) and
  // copied back onto the inner region. Inert at `pcAreaMode_ == kAreaPlic`: the kernel below then
  // takes the branch it always took, and `pcAreaCg2_` is never even allocated.
  const bool cascadeArea = (pcAreaMode_ != vof::kAreaPlic);
  if (cascadeArea) {
    if (pcAreaCg2_.extent(0) != n_)
      pcAreaCg2_ = CCField("pc_area_cascade", n_);
    bridgeColourToVof();
    const auto as = pcAreaCascadeCompute();
    pcDiag_.areaHf = as.hf + as.hfMixed;
    pcDiag_.areaPv = as.pv;
    pcDiag_.areaNone = as.noEstimate;
    pcDiag_.areaOrphan = pcMcOrphanArea_;
    copyInner(pcAreaCg2_, e_, G, CCConst(pcAreaCascadeField()), e3_, kVofG);
  }
  CCConst areaCasc = CCConst(cascadeArea ? pcAreaCg2_ : pcMdot_);
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCField mdot = pcMdot_, area = pcArea_, dep = pcDep_, tgt = pcTgt_;
  CCField nx = pcNrm_[0], ny = pcNrm_[1], nz = pcNrm_[2];
  CCConst c = CCConst(cField_);
  const bool thermal = pcThermal_;
  CCConst T = thermal ? CCConst(scalarField(pcTName_).c) : CCConst(cField_);
  const double eps = pcEffInterfaceEps(), pureEps = pcEffPureEps();
  const bool quad = pcQuadFit_;
  const bool depFallback = pcDepositFallback_;
  const double Tsat = pcTsat_, kg = pcKg_, kl = pcKl_, hlv = pcHlv_, Rint = pcRint_;
  const double kapPresc = pcFitKappa_;  // WO-P3f, 0 = the shipped (tangent-plane) distance
  const double rhoG = pcRhoG_, rhoL = pcRhoL_;
  // WO-P3g. `curvDist` (item 3) replaces the prescribed curvature by the V3 cascade's per-cell
  // one; `opMode` (item 1) replaces the least-squares fit by the energy operator's own flux, for
  // which the whole Dirichlet row set has to be readable inside the kernel.
  const bool curvDist = pcCurvDist_ && pcKappa_.extent(0) == n_;
  CCConst kapF = CCConst(curvDist ? pcKappa_ : pcMdot_);
  ScalarField* scT = thermal ? &scalarField(pcTName_) : nullptr;
  const bool opMode =
      pcMdotOperator_ && thermal && pcGphi_.extent(0) == n_ && scT->dmask.extent(0) == n_;
  const bool opEnergy = opMode && scT->energy && scT->kcell.extent(0) == n_;
  if (opMode)
    pcBuildInDomain();
  CCConst mkF = CCConst(opMode ? scT->dmask : pcMdot_);
  CCConst inD = CCConst(opMode ? pcInDomain_ : pcMdot_);
  CCConst kcF = CCConst(opEnergy ? scT->kcell : pcMdot_);
  CCConst oxF = CCConst(ox_), oyF = CCConst(oy_), ozF = CCConst(oz_);
  const double Dconst = thermal ? scT->D : 0.0, thMin = pcGfmThMin_, thMax = pcGfmThMax_;
  const int gfmOrder = pcGfmOrder_;
  // Phase 3 (V5, `flow/doc/anisotropic_vof.md` §7): the phase-change layer is where lengths,
  // areas and the cell VOLUME all enter at once.  `gme` is the per-axis metric; `vCell` the cell
  // volume in hRef^3.  Both are exactly 1 on every isotropic run.
  const vof::VofMetric gme = u_.vofMetric();
  const double vCell = u_.vol;
  CCField mdotFit = opMode ? pcMdotFit_ : pcMdot_;
  long nIface = 0, nFallback = 0;
  double sumArea = 0.0, sumMdot = 0.0, sumQ = 0.0, sumQorph = 0.0, sumMdotFitA = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_build",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, long& nif, long& nfb, double& aacc, double& macc,
                    double& qacc, double& qorph, double& mfacc) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        area(i) = 0.0;
        dep(i) = 0.0;
        tgt(i) = 0.0;
        nx(i) = 0.0;
        ny(i) = 0.0;
        nz(i) = 0.0;
        if (!vof::pcIsInterfacial(c(i), eps)) {
          if (thermal)
            mdot(i) = 0.0;
          return;
        }
        double st[27];
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              st[vof::plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
        double m[3];
        vof::mycNormal(st, m);
        const double al = vof::plicAlpha(m[0], m[1], m[2], c(i));
        // PHYSICAL area (V5.1) — `areaCasc` already carries the metric from its own driver.
        const double A = cascadeArea ? areaCasc(i) : vof::plicAreaMetric(m[0], m[1], m[2], al, gme);
        // The PHYSICAL unit normal and the PHYSICAL centre distance (V5.2): `s(m)` is the shape
        // factor of §2, exactly 1.0 at equal spacings, so both are today's values there.
        double n[3] = {1.0, 0.0, 0.0};
        const double sMet = vof::vofPhysNormal(m, gme, n);
        if (!(sMet > 0.0))
          return;
        const double phic = vof::pcCentreDistance(m[0], m[1], m[2], al) / sMet;
        // **WO-P3g item 1 — the area is a UNIT CONVERSION, not a term in the mass balance.**
        // The regression removes `dV = mdot A dt/rho_l` and the source deposits
        // `S = mdot A (1/rho_g - 1/rho_l)`, and with `mdot = q/(h_lv A)` BOTH are `A`-free:
        // `dV = q dt/(h_lv rho_l)`, `S = q (1/rho_g - 1/rho_l)/h_lv`. So a cell the AREA
        // ESTIMATOR gave nothing still has to evaporate the heat its Dirichlet rows draw, or
        // that heat is simply destroyed. Measured on the a-priori probe (exact sphere, R = 20,
        // Ja 0.5, area mode 6): **27.6 % of the operator's total interfacial heat sits on
        // interfacial cells with A = 0** -- the joined marching-tetrahedra sheet deposits each
        // triangle to the cell holding its centroid, so a band cell can carry an interface and
        // no area. `Aeff` is the unit conversion those cells use; the DIAGNOSTIC area
        // (`interface_area`) stays the geometric one, so `mdot_area` is unaffected.
        double Aeff = A;
        double md = mdot(i);
        if (thermal) {
          const double Tg = vof::pcInterfaceTemperature(Tsat, md, Rint);
          // WO-P3g item 3: the sample distance is measured to the CURVED interface, with the
          // curvature taken per cell from the V3 cascade (`curvDist`) or, as WO-P3f shipped it,
          // prescribed for the whole field. Bitwise unchanged when both are 0.
          const double kapFit = curvDist ? kapF(i) : kapPresc;
          vof::PcGradFit fg, fl;
          for (int dz = -2; dz <= 2; ++dz)
            for (int dy = -2; dy <= 2; ++dy)
              for (int dx = -2; dx <= 2; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                // V5.2: the sample offset is PHYSICAL — `delta = H d` — because the fit models
                // T against the physical normal distance and returns a physical dT/dn.  With
                // index offsets the fitted gradient would be off by the shape factor `s(m)`.
                // `g.h = {1,1,1}` multiplies each component by 1.0.
                const double di[3] = {(double)dx, (double)dy, (double)dz};
                double dp[3];
                gme.toPhys(di, dp);
                const double w = vof::pcGradWeight(dp[0], dp[1], dp[2], n);
                if (!(w > 0.0))
                  continue;
                const long j = i + dx + dy * sy + dz * sz;
                const double cj = c(j);
                double phi = vof::pcOffsetDistance(phic, n, dp[0], dp[1], dp[2]);
                // WO-P3f: bitwise unchanged at kappaFit == 0 (the shipped default)
                if (kapFit != 0.0)
                  phi = vof::pcCurvedDistance(phi, dp[0], dp[1], dp[2], n, kapFit);
                if (cj <= pureEps && phi > 0.0)
                  vof::pcGradAdd(fg, w, phi, T(j), Tg);
                else if (cj >= 1.0 - pureEps && phi < 0.0)
                  vof::pcGradAdd(fl, w, phi, T(j), Tg);
              }
          md = quad ? vof::pcMassFlux(kg, vof::pcGradSolve2(fg), kl, vof::pcGradSolve2(fl), hlv)
                    : vof::pcMassFlux(kg, vof::pcGradSolve(fg), kl, vof::pcGradSolve(fl), hlv);
          // **WO-P3g item 1 — `mdot` from the ENERGY OPERATOR's own interfacial flux.**
          //
          // The rows the energy solve runs are `scalarMaskGfm2`'s, one per (pure cell, masked
          // neighbour) face; this gathers the SAME rows from the masked side and asks how much
          // heat they draw. Every ingredient is read from the state that solve used — the
          // Dirichlet mask, the face openness, `k(C)` and the plane geometry — and the plane this
          // kernel just reconstructed is bitwise the plane `pcUpdateThermalMask` stored (same
          // colour, same `mycNormal`/`plicAlpha`), so `theta` is identical to the row's own.
          //
          // What it buys: `mdot h_lv A = q` makes the heat the energy equation loses and the mass
          // the regression produces ONE discrete quantity (WO-P3f measured the shipped pair
          // disagreeing by 0.95…1.04), and `A` then cancels out of both the plane shift
          // `dV = mdot A dt/rho_l` and the divergence source — the interfacial area stops being a
          // term in the mass balance at all.
          if (opMode) {
            const long stq[3] = {1, sy, sz};
            double qsum = 0.0, csum = 0.0;
            for (int d = 0; d < 3; ++d)
              for (int sg = -1; sg <= 1; sg += 2) {
                const long pc = i + (long)sg * stq[d];
                if (mkF(pc) > 0.5)
                  continue;  // an identity row: it carries no Dirichlet coupling
                if (!(inD(pc) > 0.5))
                  continue;  // outside the global domain: no row exists there at all
                const double of = (d == 0)   ? ((sg > 0) ? oxF(i + 1) : oxF(i))
                                  : (d == 1) ? ((sg > 0) ? oyF(i + sy) : oyF(i))
                                             : ((sg > 0) ? ozF(i + sz) : ozF(i));
                if (!(of > 0.0))
                  continue;
                const double ofB = (d == 0)   ? ((sg > 0) ? oxF(pc + 1) : oxF(pc))
                                   : (d == 1) ? ((sg > 0) ? oyF(pc + sy) : oyF(pc))
                                              : ((sg > 0) ? ozF(pc + sz) : ozF(pc));
                const long jb = pc + (long)sg * stq[d];
                const bool behind = !(mkF(jb) > 0.5) && ofB > 0.0;
                // the step from the PURE cell `pc` to this (masked) cell is -sg
                // V5.3: `theta` is a distance ALONG a grid line, i.e. an axis ratio — it takes
                // the INDEX centre distance and the INDEX normal component, both of which the
                // metric-aware overload reconstructs from the physical pair.
                const double th = vof::pcGfmThetaKAniso(
                    phic, n, d, (double)(-sg), curvDist ? kapF(i) : 0.0, thMin, thMax, gme);
                const vof::PcGfmRow row = vof::pcGfmRow(th, behind, gfmOrder);
                const double cf = (opEnergy ? kcF(pc) : Dconst) * of * row.aGamma;
                qsum += cf * (T(pc) - Tsat);
                csum += cf;
                // The three-point row also RESCALES `pc`'s band toward the cell BEHIND it, and
                // that rescaling is one-sided (cell `jb`'s own row does not mirror it), so it is
                // part of the heat the interface removes from the solved set. Booking it here is
                // what makes `sum_j Q_j` the EXACT total the operator transfers -- and, on a
                // 1-D quadratic, the exact conductive flux through the interfacial face at every
                // theta (gate (a); with the Dirichlet coupling alone it is a factor 2/(1+theta)
                // off, which is the whole point of the second-order row).
                if (behind && row.aBehind != 1.0) {
                  const double kb = opEnergy ? 0.5 * (kcF(pc) + kcF(jb)) : Dconst;
                  qsum += (row.aBehind - 1.0) * kb * ofB * (T(pc) - T(jb));
                }
              }
            mdotFit(i) = md;  // the least-squares estimator, kept as a diagnostic
            Aeff = (A > 0.0) ? A : 1.0;
            md = vof::pcOperatorMassFlux(qsum, csum, Aeff, hlv, Rint);
            qacc += qsum;
            if (!(A > 0.0))
              qorph += qsum;  // reported, but no longer dropped
            mfacc += mdotFit(i) * A;
          }
          mdot(i) = md;
        }
        area(i) = Aeff;
        nx(i) = n[0];
        ny(i) = n[1];
        nz(i) = n[2];
        ++nif;
        aacc += A;  // the GEOMETRIC area, for the diagnostics; `Aeff` is the unit conversion
        macc += md;
        // the divergence source and the pure-gas cell that will carry it
        // V5.5: `S` is a volumetric source, so the interfacial flux is divided by the cell
        // VOLUME `V' = hp_x hp_y hp_z` (exactly 1.0 isotropic: `x / 1.0 == x`).
        const double S = vof::pcDivSource(md, Aeff, rhoG, rhoL) / vCell;
        if (S != 0.0) {
          // WO-P23: the receiving PURE GAS cell is the BEST cell of the 5^3 neighbourhood on
          // the `+n` side, scored by Malan's own collinearity weight `(d.n)^2/|d|^3` — closest
          // and most nearly along the normal wins, so it returns the `d = round(n)` cell whenever
          // that one is pure and degrades gracefully when it is not. The rung P0/P1 rule tried
          // exactly two candidates (`round(k n)`, k = 1, 2) and left the source IN the
          // interfacial cell when both were still interfacial — which is what happens on a
          // CURVED interface: measured 48 … 262 such cells on the P3 Scriven bubble, and each of
          // them then carries `div(open u) = S` on its OWN faces, i.e. Weymouth-Yue advects it
          // with a field that is not the liquid velocity (that is exactly what
          // `phase_change_diagnostics()['band_div']` reports). The search order is fixed and the
          // comparison strict, so the choice is deterministic and decomposition-independent.
          int tx = 0, ty = 0, tz = 0;
          bool found = false;
          for (int k = 1; k <= 2 && !found; ++k) {  // the rung P0/P1 rule, FIRST and unchanged
            const int ox = (int)Kokkos::round(k * n[0]);
            const int oy = (int)Kokkos::round(k * n[1]);
            const int oz = (int)Kokkos::round(k * n[2]);
            if (ox == 0 && oy == 0 && oz == 0)
              continue;
            if (c(i + ox + oy * sy + oz * sz) <= pureEps) {
              tx = ox;
              ty = oy;
              tz = oz;
              found = true;
            }
          }
          if (!found && depFallback) {
            // Only where that rule FAILS does the deposit fall back to the best cell of the `+n`
            // half of the 5^3 box, scored by Malan's collinearity weight. Those are the cells the
            // P0/P1 code left the source in (48 … 262 of them on the Scriven bubble), and each
            // one then carries `div(open u) = S` on its OWN faces, i.e. Weymouth-Yue advecting
            // the colour with a field that is not the liquid velocity.
            //
            // The order matters and the measurement says so: making the search the PRIMARY rule
            // (best-`w` over the whole box) DIVERGES the Scriven bubble — `max|uf|` runs away and
            // the study's dt-collapse guard trips at step 316 of the Ja = 0.5 run with the
            // Weymouth-Yue Courant number pinned at 0.38 however small dt is. The planar gates
            // (P0a/P0b/P1/P2) never saw it, because there the two rules choose the same cell.
            // Preferring the along-the-normal candidates keeps the validated behaviour wherever
            // it existed and only fills the holes.
            double best = 0.0;
            for (int dz = -2; dz <= 2; ++dz)
              for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                  if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                  // V5.5: the candidate is scored on its PHYSICAL offset, so on a stretched
                  // grid the search still prefers the cell that is nearest AND most along the
                  // normal. `g.h = {1,1,1}` multiplies each component by 1.0.
                  const double di[3] = {(double)dx, (double)dy, (double)dz};
                  double dp[3];
                  gme.toPhys(di, dp);
                  if (!(dp[0] * n[0] + dp[1] * n[1] + dp[2] * n[2] > 0.0))
                    continue;  // the deposit goes BEHIND the interface, into the gas
                  const double w = vof::pcGradWeight(dp[0], dp[1], dp[2], n);
                  if (!(w > best))
                    continue;
                  if (c(i + dx + dy * sy + dz * sz) > pureEps)
                    continue;
                  best = w;
                  tx = dx;
                  ty = dy;
                  tz = dz;
                }
            if (!(best > 0.0))
              ++nfb;  // no pure gas cell in the 5^3 box on the +n side: the source stays put
          } else if (!found) {
            ++nfb;  // the default: no pure gas cell within two cells, the source stays put
          }
          dep(i) = S;
          tgt(i) = (double)((tx + 2) + 5 * (ty + 2) + 25 * (tz + 2));
        }
      },
      nIface, nFallback, sumArea, sumMdot, sumQ, sumQorph, sumMdotFitA);
  Kokkos::fence();
  pcQOperator_ = sumQ;
  pcQOrphan_ = sumQorph;
  pcMdotFitMean_ = sumArea > 0.0 ? sumMdotFitA / sumArea : 0.0;
  pcDiag_.qOperator = sumQ;
  pcDiag_.qOrphan = sumQorph;
  pcDiag_.mdotFit = pcMdotFitMean_;
  pcDiag_.interfaceCells = nIface;
  pcDiag_.fallbackCells = nFallback;
  pcDiag_.area = sumArea;
  pcDiag_.mdotMean = nIface > 0 ? sumMdot / (double)nIface : 0.0;
  // extrema of mdot over interfacial cells
  double mn = 0.0, mx = 0.0;
  if (nIface > 0) {
    mn = 1e300;
    mx = -1e300;
    CCConst md = CCConst(pcMdot_), ar = CCConst(pcArea_);
    Kokkos::parallel_reduce(
        "peclet::flow::pc_mdot_extrema",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& lo, double& hi) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          if (ar(i) > 0.0) {
            lo = Kokkos::fmin(lo, md(i));
            hi = Kokkos::fmax(hi, md(i));
          }
        },
        Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
    Kokkos::fence();
  }
  pcDiag_.mdotMin = mn;
  pcDiag_.mdotMax = mx;
  // The exchange that makes the two consumers decomposition-independent. `fillGhosts` is the
  // halo/periodic base; on a NON-periodic domain face the periodic wrap would import the far
  // side's interface as a phantom source/deficit donor, so those ghosts are zeroed.
  fillGhosts(pcMdot_);
  fillGhosts(pcArea_);
  fillGhosts(pcDep_);
  fillGhosts(pcTgt_);
  for (int d = 0; d < 3; ++d)
    fillGhosts(pcNrm_[d]);
  pcZeroDomainGhosts(pcMdot_);
  pcZeroDomainGhosts(pcArea_);
  pcZeroDomainGhosts(pcDep_);
  pcZeroDomainGhosts(pcTgt_);
  for (int d = 0; d < 3; ++d)
    pcZeroDomainGhosts(pcNrm_[d]);
}

template <class Grid>
void Solver<Grid>::pcScatterSource() {
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCField src = pcSrc_;
  CCConst dep = CCConst(pcDep_), tgt = CCConst(pcTgt_);
  double sum = 0.0;
  long ncell = 0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_source_gather",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc, long& nc) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        double s = 0.0;
        for (int dz = -2; dz <= 2; ++dz)
          for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
              const long j = i + dx + dy * sy + dz * sz;
              const double dj = dep(j);
              if (dj == 0.0)
                continue;
              const int code = (int)tgt(j);
              const int tx = code % 5 - 2, ty = (code / 5) % 5 - 2, tz = code / 25 - 2;
              if (tx + dx == 0 && ty + dy == 0 && tz + dz == 0)
                s += dj;
            }
        src(i) = s;
        acc += s;
        if (s != 0.0)
          ++nc;
      },
      sum, ncell);
  Kokkos::fence();
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = 0;
    MPI_Allreduce(&sum, &g, 1, MPI_DOUBLE, MPI_SUM, comm_);
    sum = g;
  }
#endif
  // WO-P23: the auto-balanced sink. Subtract the GLOBAL deposited source, spread over the sink
  // weights, so a closed domain's Poisson RHS is compatible by construction.
  if (pcHasSink_) {
    const double f = sum / pcSinkW_;
    CCConst wv = CCConst(pcSink_);
    Kokkos::parallel_for(
        "peclet::flow::pc_sink_apply",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          src(i) -= f * wv(i);
        });
    Kokkos::fence();
  }
  pcDiag_.sourceSum = sum;
  pcDiag_.sourceCells = ncell;
  fillPropGhosts(pcSrc_);
}

template <class Grid>
void Solver<Grid>::pcRegress(double dt) {
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCField Cf = cField_, cnew = pcCnew_, defic = pcDefic_;
  CCConst md = CCConst(pcMdot_), ar = CCConst(pcArea_);
  const double rhoL = pcRhoL_;
  // V5.5: `mdot A dt/rho_l` is a liquid VOLUME; the colour it removes is that volume divided by
  // the CELL volume `V' = hp_x hp_y hp_z` (exactly 1.0 on every isotropic run).
  const double vCell = u_.vol;
  double removed = 0.0;
  long ndef = 0, nexc = 0;
  double redist = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_regress_raw",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z, double& rem, long& nd, long& ne, double& rd) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        const double A = ar(i);
        if (!(A > 0.0)) {
          cnew(i) = Cf(i);
          defic(i) = 0.0;
          return;
        }
        const double dV = vof::pcRegressVolume(md(i), A, dt, rhoL) / vCell;
        const double raw = Cf(i) - dV;
        const double cl = Kokkos::fmin(Kokkos::fmax(raw, 0.0), 1.0);
        cnew(i) = cl;
        defic(i) = raw - cl;
        const bool inner =
            (x >= G && x < e.x - G && y >= G && y < e.y - G && z >= G && z < e.z - G);
        if (inner) {
          rem += dV;
          if (raw < 0.0)
            ++nd;
          if (raw > 1.0)
            ++ne;
          rd += Kokkos::fabs(raw - cl);
        }
      },
      removed, ndef, nexc, redist);
  Kokkos::fence();
  CCConst cn = CCConst(pcCnew_), df = CCConst(pcDefic_);
  CCConst nxv = CCConst(pcNrm_[0]), nyv = CCConst(pcNrm_[1]), nzv = CCConst(pcNrm_[2]);
  Kokkos::parallel_for(
      "peclet::flow::pc_regress_apply",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        double v = cn(i);
        const long st[3] = {1, sy, sz};
        for (int d = 0; d < 3; ++d)
          for (int s = -1; s <= 1; s += 2) {
            const long j = i + (long)s * st[d];  // the neighbour that might push into i
            const double dj = df(j);
            if (dj == 0.0)
              continue;
            // Recompute j's WHOLE allocation here (not just i's share): every receiver runs the
            // identical arithmetic on the identical inputs, so the sum each cell forms has a
            // fixed order and is bitwise independent of the decomposition.
            const double sgn = dj < 0.0 ? -1.0 : 1.0;
            double n[3] = {nxv(j), nyv(j), nzv(j)};
            int step[3];
            double w[3];
            bool avail[3];
            for (int q = 0; q < 3; ++q) {
              const double p = sgn * n[q];
              const int sq = (p > 0.0) ? 1 : ((p < 0.0) ? -1 : 0);
              const double ct = sq == 0 ? 0.0 : cn(j + (long)sq * st[q]);
              avail[q] = sq != 0 && (dj < 0.0 ? (ct > 0.0) : (ct < 1.0));
            }
            vof::pcPushWeights(n, sgn, avail, step, w);
            // j pushes into j + step[d]*e_d; that is i iff step[d] == -s
            if (step[d] == -s)
              v += dj * w[d];
          }
        Cf(i) = v;
      });
  Kokkos::fence();
  pcDiag_.removedVolume = removed;
  pcDiag_.deficitCells = ndef;
  pcDiag_.excessCells = nexc;
  pcDiag_.redistributed = redist;
  // How much residue found no neighbour able to absorb it (pushed anyway, on the unrestricted
  // weights, so conservation holds and the colour goes slightly out of [0,1] instead).
  double unres = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_unresolved",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        const double dj = df(i);
        if (dj == 0.0)
          return;
        const double sgn = dj < 0.0 ? -1.0 : 1.0;
        const long st[3] = {1, sy, sz};
        double n[3] = {nxv(i), nyv(i), nzv(i)};
        bool avail[3];
        for (int q = 0; q < 3; ++q) {
          const double p = sgn * n[q];
          const int sq = (p > 0.0) ? 1 : ((p < 0.0) ? -1 : 0);
          const double ct = sq == 0 ? 0.0 : cn(i + (long)sq * st[q]);
          avail[q] = sq != 0 && (dj < 0.0 ? (ct > 0.0) : (ct < 1.0));
        }
        int step[3];
        double w[3];
        if (!vof::pcPushWeights(n, sgn, avail, step, w))
          acc += Kokkos::fabs(dj);
      },
      unres);
  Kokkos::fence();
  pcDiag_.unresolved = unres;
  fillPropGhosts(cField_);
}

template <class Grid>
void Solver<Grid>::pcUpdateCurvature() {
  if (pcKappa_.extent(0) != n_)
    pcKappa_ = CCField("pc_kappa", n_);
  bridgeColourToVof();
  vofCurv_.compute(vofAdv_.colour());
  copyInner(pcKappa_, e_, G, CCConst(vofCurv_.kappa()), e3_, kVofG);
  // read at depth 1 (the GFM row asks for the INTERFACIAL neighbour's curvature), so exchange;
  // a periodic wrap on a non-periodic domain face would import a phantom curvature, so zero it.
  fillGhosts(pcKappa_);
  pcZeroDomainGhosts(pcKappa_);
}

template <class Grid>
void Solver<Grid>::pcBuildInDomain() {
  if (pcInDomain_.extent(0) == n_)
    return;
  pcInDomain_ = CCField("pc_indomain", n_);
  Kokkos::deep_copy(pcInDomain_, 0.0);
  const C3 e = e_;
  CCField f = pcInDomain_;
  Kokkos::parallel_for(
      "peclet::flow::pc_indomain",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        f((long)x + (long)y * e.x + (long)z * (long)e.x * e.y) = 1.0;
      });
  Kokkos::fence();
  fillGhosts(pcInDomain_);
  pcZeroDomainGhosts(pcInDomain_);
}

template <class Grid>
void Solver<Grid>::pcUpdateThermalMask() {
  if (!pcThermal_)
    return;
  if (pcCurvDist_)
    pcUpdateCurvature();  // WO-P3g item 3: kappa for the row's theta and for the carried-value fit
  ScalarField& sc = scalarField(pcTName_);
  scalarFillGhosts(sc);  // the carried-value refit reads T at +-2 (see pcBuildInterface)
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCField mk = sc.dmask, dv = sc.dval, tg = pcTgam_;
  CCField gn0 = pcGn_[0], gn1 = pcGn_[1], gn2 = pcGn_[2], gph = pcGphi_;
  CCConst c = CCConst(cField_), md = CCConst(pcMdot_), T = CCConst(sc.c);
  const double eps = pcEffInterfaceEps(), pureEps = pcEffPureEps(), Tsat = pcTsat_, Rint = pcRint_;
  const double kapPresc = pcFitKappa_;                            // WO-P3f
  const vof::VofMetric gme = u_.vofMetric();                      // Phase 3 (V5.2/V5.3)
  const bool curvDist = pcCurvDist_ && pcKappa_.extent(0) == n_;  // WO-P3g item 3
  CCConst kapF = CCConst(curvDist ? pcKappa_ : pcMdot_);
  const bool carry = pcPlaneDir_, quad = pcQuadFit_;
  const long syl = sy, szl = sz;
  Kokkos::parallel_for(
      "peclet::flow::pc_thermal_mask",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * sz;
        const bool on = vof::pcIsInterfacial(c(i), eps);
        const double Tgam = vof::pcInterfaceTemperature(Tsat, md(i), Rint);
        mk(i) = on ? 1.0 : 0.0;
        tg(i) = Tgam;
        dv(i) = Tgam;
        gn0(i) = 0.0;
        gn1(i) = 0.0;
        gn2(i) = 0.0;
        gph(i) = 0.0;
        if (!on)
          return;
        // WO-P23: the plane geometry the PLANE-ANCHORED (ghost-fluid) Dirichlet rows need — the
        // unit normal and the signed centre distance of THIS cell's PLIC plane, rebuilt from the
        // colour the energy solve is about to run with (the head-of-step values belong to C^n and
        // the interface has moved since).
        double st[27];
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              st[vof::plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
        double m[3];
        vof::mycNormal(st, m);
        // V5.2/V5.3: the stored normal and centre distance are PHYSICAL — the plane-anchored
        // rows and the one-sided fit both measure in physical lengths, and the axis pullback
        // happens in `pcGfmThetaKAniso`. Identity at the unit metric.
        double n[3] = {0.0, 0.0, 0.0};
        const double sMet = vof::vofPhysNormal(m, gme, n);
        if (!(sMet > 0.0))
          return;
        const double al = vof::plicAlpha(m[0], m[1], m[2], c(i));
        gn0(i) = n[0];
        gn1(i) = n[1];
        gn2(i) = n[2];
        const double phic = vof::pcCentreDistance(m[0], m[1], m[2], al) / sMet;
        gph(i) = phic;
        if (!carry)
          return;
        // The value this cell CARRIES until the interface sweeps past it and it becomes pure.
        // Nothing reads it across a face (the plane-anchored rows use `tg`), so it is free to be
        // the one-sided extrapolation of the profile on the side the cell CENTRE lies on.
        const bool gasSide = phic > 0.0;
        const double kapFit = curvDist ? kapF(i) : kapPresc;  // WO-P3g item 3
        vof::PcGradFit f;
        for (int dz = -2; dz <= 2; ++dz)
          for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0)
                continue;
              const double di[3] = {(double)dx, (double)dy, (double)dz};
              double dp[3];
              gme.toPhys(di, dp);  // V5.2: PHYSICAL sample offsets
              const double w = vof::pcGradWeight(dp[0], dp[1], dp[2], n);
              if (!(w > 0.0))
                continue;
              const long j = i + dx + dy * syl + dz * szl;
              const double cj = c(j);
              double phi = vof::pcOffsetDistance(phic, n, dp[0], dp[1], dp[2]);
              if (kapFit != 0.0)  // WO-P3f, bitwise unchanged at 0
                phi = vof::pcCurvedDistance(phi, dp[0], dp[1], dp[2], n, kapFit);
              if (gasSide) {
                if (cj <= pureEps && phi > 0.0)
                  vof::pcGradAdd(f, w, phi, T(j), Tgam);
              } else if (cj >= 1.0 - pureEps && phi < 0.0) {
                vof::pcGradAdd(f, w, phi, T(j), Tgam);
              }
            }
        dv(i) = vof::pcCarriedValue(Tgam, quad ? vof::pcGradSolve2(f) : vof::pcGradSolve(f), phic);
      });
  Kokkos::fence();
  // The GFM rows read mask / dval / n / phi of a FACE NEIGHBOUR, i.e. at depth 1: exchange, then
  // kill the periodic wrap on non-periodic domain faces (a wrapped "interfacial" ghost would turn
  // a domain-boundary face into a phantom Dirichlet).
  fillGhosts(sc.dmask);
  fillGhosts(sc.dval);
  fillGhosts(pcTgam_);
  fillGhosts(pcGn_[0]);
  fillGhosts(pcGn_[1]);
  fillGhosts(pcGn_[2]);
  fillGhosts(pcGphi_);
  pcZeroDomainGhosts(sc.dmask);
  pcZeroDomainGhosts(sc.dval);
  pcZeroDomainGhosts(pcTgam_);
  pcZeroDomainGhosts(pcGn_[0]);
  pcZeroDomainGhosts(pcGn_[1]);
  pcZeroDomainGhosts(pcGn_[2]);
  pcZeroDomainGhosts(pcGphi_);
  // WO-P3f: the enthalpy the overwrite is about to destroy, returned to the phase it came from.
  // Runs AFTER the exchanges above (it reads the donor's mask/normal/T_Gamma at depth 1 and the
  // donor's own neighbours at depth 2). Inert -- no kernel, no allocation -- when the option
  // is off.
  if (pcCarryConserve_ && pcCarrySrc_.extent(0) == n_)
    pcCarryDeposit(sc);
  // WO-P3g item 1: the Dirichlet rows now describe THIS colour field, which is what the
  // operator-flux `mdot` needs before it may read them (see the note in `pcBuildInterface`).
  pcMaskFresh_ = true;
}

template <class Grid>
void Solver<Grid>::pcZeroDomainGhosts(CCField f) {
  for (int face = 0; face < 6; ++face) {
    if (bc_[face] == 0 || !touchesGlobalFace(face))
      continue;
    const int a = face / 2, side = face % 2;
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
    const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
    const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
    const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
    const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
    const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
    const int aInner = (side == 0) ? G : (G + na - 1);
    const int dir = (side == 0) ? -1 : +1;
    Kokkos::parallel_for(
        "peclet::flow::pc_zero_ghosts",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(CCExec(), {G, G}, {G + nt1, G + nt2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long base = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
          for (int L = 1; L <= 2; ++L)
            f(base + (long)dir * L * sa) = 0.0;
        });
  }
  Kokkos::fence();
}

template <class Grid>
void Solver<Grid>::pcCarryDeposit(ScalarField& sc) {
  const C3 e = e_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  CCConst c = CCConst(cField_), T = CCConst(sc.c), dv = CCConst(sc.dval), mk = CCConst(sc.dmask),
          tg = CCConst(pcTgam_), gn0 = CCConst(pcGn_[0]), gn1 = CCConst(pcGn_[1]),
          gn2 = CCConst(pcGn_[2]);
  CCField srcF = pcCarrySrc_;
  const bool useRcp = pcEnergy_;
  const double rcpG = pcRcpG_, rcpL = pcRcpL_;
  double dep = 0.0, lost = 0.0;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_carry_deposit", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long st[3] = {sx, sy, sz};
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (mk(i) > 0.5)
          return;  // masked cells are the identity rows; they receive nothing
        double add = 0.0;
        for (int d = 0; d < 3; ++d)
          for (int sg = -1; sg <= 1; sg += 2) {
            const long j = i + (long)sg * st[d];
            if (!(mk(j) > 0.5))
              continue;
            const double rj = useRcp ? vof::pcPhaseMix(rcpG, rcpL, c(j)) : 1.0;
            const double dE = rj * (T(j) - dv(j));
            if (!(dE != 0.0))
              continue;
            const double nn[3] = {gn0(j), gn1(j), gn2(j)};
            const double devj = T(j) - tg(j);
            int chosen[3] = {0, 0, 0};
            double w[3] = {0.0, 0.0, 0.0}, wsum = 0.0;
            for (int dd = 0; dd < 3; ++dd) {
              if (nn[dd] == 0.0)
                continue;
              const int sL = (nn[dd] > 0.0) ? -1 : +1;  // n points into the GAS
              const double dl = T(j + (long)sL * st[dd]) - tg(j);
              const double dg = T(j - (long)sL * st[dd]) - tg(j);
              const int s = (dl * devj >= dg * devj) ? sL : -sL;
              if (mk(j + (long)s * st[dd]) > 0.5)
                continue;  // that neighbour is itself an identity row: it cannot absorb
              chosen[dd] = s;
              w[dd] = nn[dd] * nn[dd];
              wsum += w[dd];
            }
            if (!(wsum > 0.0))
              continue;  // no neighbour left in the solve; the residual is booked below
            if (chosen[d] != -sg)
              continue;
            add += (w[d] / wsum) * dE;
          }
        if (add == 0.0)
          return;
        const double ri = useRcp ? vof::pcPhaseMix(rcpG, rcpL, c(i)) : 1.0;
        srcF(i) += add / ri;  // stored as a TEMPERATURE increment for cell i
        acc += add;
      },
      dep);
  // The donor total, so `lost` is the EXACT unplaced remainder rather than a census of one
  // failure mode: an interfacial cell whose whole `n_d^2` allocation lands on cells that are
  // themselves identity rows keeps its discarded enthalpy, and on a curved interface the band is
  // thicker than one cell, so that is not a rare event. Report it, never hide it.
  Kokkos::parallel_reduce(
      "peclet::flow::pc_carry_total", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& tot) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(mk(i) > 0.5))
          return;
        const double ri = useRcp ? vof::pcPhaseMix(rcpG, rcpL, c(i)) : 1.0;
        tot += ri * (T(i) - dv(i));
      },
      lost);
  Kokkos::fence();
  pcCarryDeposited_ = dep;
  pcCarryLost_ = lost - dep;
}

template <class Grid>
void Solver<Grid>::pcCarryApply(ScalarField& sc) {
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCField cOld = sc.cOld, srcF = pcCarrySrc_;
  CCConst mk = CCConst(sc.dmask);
  Kokkos::parallel_for(
      "peclet::flow::pc_carry_apply",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(mk(i) > 0.5))
          cOld(i) += srcF(i);
        srcF(i) = 0.0;
      });
  Kokkos::fence();
}

template <class Grid>
void Solver<Grid>::pcBudgetPre(ScalarField& sc) {
  const C3 e = e_;
  const long sy = e_.x, sz = (long)e_.x * e_.y;
  CCConst c = CCConst(cField_), T = CCConst(sc.cOld), dvl = CCConst(sc.dval),
          mk = CCConst(sc.dmask), cls = CCConst(pcClsPrev_);
  const bool useRcp = sc.energy && sc.rcp.extent(0) == n_;
  CCConst rcp = useRcp ? CCConst(sc.rcp) : CCConst(sc.c);
  const double Tsat = pcTsat_;
  double hOpen = 0, hLiq = 0, hMask = 0, dEo = 0, dEoN = 0, eEnt = 0, eLev = 0;
  long nEL = 0, nEG = 0, nLL = 0, nLG = 0, nM = 0;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_budget_pre", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a1, double& a2, double& a3, double& a4, double& a5,
                    double& a6, double& a7) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double r = useRcp ? rcp(i) : 1.0;
        const double h = r * (T(i) - Tsat);
        const bool msk = mk(i) > 0.5;
        if (msk) {
          a3 += h;
          a4 += r * (dvl(i) - T(i));
        } else {
          a1 += h;
          if (c(i) >= 0.5)
            a2 += h;
        }
        const double cp = cls(i);
        if (!(cp >= 0.0))
          return;  // the first instrumented step has no previous class
        const bool wasMsk = (cp > 0.5 && cp < 1.5);
        if (msk && !wasMsk) {
          a6 += h;
          a5 += r * (dvl(i) - T(i));
        } else if (!msk && wasMsk) {
          a7 += h;
        }
      },
      hOpen, hLiq, hMask, dEo, dEoN, eEnt, eLev);
  Kokkos::parallel_reduce(
      "peclet::flow::pc_budget_counts", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, long& c1, long& c2, long& c3, long& c4, long& c5) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const bool msk = mk(i) > 0.5;
        if (msk)
          c5 += 1;
        const double cp = cls(i);
        if (!(cp >= 0.0))
          return;
        const bool wasMsk = (cp > 0.5 && cp < 1.5);
        if (msk && !wasMsk) {
          if (cp > 1.5)
            c1 += 1;
          else
            c2 += 1;
        } else if (!msk && wasMsk) {
          if (c(i) >= 0.5)
            c3 += 1;
          else
            c4 += 1;
        }
      },
      nEL, nEG, nLL, nLG, nM);
  Kokkos::fence();
  pcBudget_.hOpen = hOpen;
  pcBudget_.hLiquid = hLiq;
  pcBudget_.hMasked = hMask;
  pcBudget_.dEoverwrite = dEo;
  pcBudget_.dEoverwriteNew = dEoN;
  pcBudget_.eEnter = eEnt;
  pcBudget_.eLeave = eLev;
  pcBudget_.nEnterLiquid = nEL;
  pcBudget_.nEnterGas = nEG;
  pcBudget_.nLeaveLiquid = nLL;
  pcBudget_.nLeaveGas = nLG;
  pcBudget_.nMasked = nM;
}

template <class Grid>
void Solver<Grid>::pcBudgetPost(ScalarField& sc) {
  const C3 e = e_;
  const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
  CCConst c = CCConst(cField_), T = CCConst(sc.c), mk = CCConst(sc.dmask), tg = CCConst(pcTgam_),
          gnx = CCConst(pcGn_[0]), gny = CCConst(pcGn_[1]), gnz = CCConst(pcGn_[2]),
          gph = CCConst(pcGphi_);
  CCConst ox = CCConst(ox_), oy = CCConst(oy_), oz = CCConst(oz_);
  const bool useRcp = sc.energy && sc.rcp.extent(0) == n_;
  const bool useK = sc.energy && sc.kcell.extent(0) == n_;
  CCConst rcp = useRcp ? CCConst(sc.rcp) : CCConst(sc.c);
  CCConst kc = useK ? CCConst(sc.kcell) : CCConst(sc.c);
  const double Tsat = pcTsat_, Dc = sc.D, thMin = pcGfmThMin_, thMax = pcGfmThMax_;
  const bool gfm = pcPlaneDir_ && pcGphi_.extent(0) == n_;
  // WO-P3g: the instrument has to mirror the ROW, or it measures a scheme nobody ran.
  const int gfmOrder = pcGfmOrder_;
  const bool curvDist = pcCurvDist_ && pcKappa_.extent(0) == n_;
  CCConst kapF = CCConst(curvDist ? pcKappa_ : pcGphi_);
  const vof::VofMetric gme = u_.vofMetric();  // Phase 3 (V5.3)
  CCField clsOut = pcClsPrev_;
  double hOpen = 0, q = 0, qb = 0;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "peclet::flow::pc_budget_post", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a1, double& a2, double& a3) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double r = useRcp ? rcp(i) : 1.0;
        const bool msk = mk(i) > 0.5;
        clsOut(i) = msk ? 1.0 : (c(i) >= 0.5 ? 2.0 : 0.0);
        if (msk)
          return;
        a1 += r * (T(i) - Tsat);
        if (!gfm)
          return;
        const long st[3] = {sx, sy, sz};
        const double kd = useK ? kc(i) : Dc;
        for (int d = 0; d < 3; ++d)
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            const long j = i + (long)sgn * st[d];
            if (!(mk(j) > 0.5))
              continue;
            const double of = (d == 0)   ? ((sgn < 0) ? ox(i) : ox(i + sx))
                              : (d == 1) ? ((sgn < 0) ? oy(i) : oy(i + sy))
                                         : ((sgn < 0) ? oz(i) : oz(i + sz));
            const double nvec[3] = {gnx(j), gny(j), gnz(j)};
            const double th = vof::pcGfmThetaKAniso(gph(j), nvec, d, (double)sgn,
                                                    curvDist ? kapF(j) : 0.0, thMin, thMax, gme);
            const long jb = i - (long)sgn * st[d];
            const double ofB = (d == 0)   ? ((sgn < 0) ? ox(i + sx) : ox(i))
                               : (d == 1) ? ((sgn < 0) ? oy(i + sy) : oy(i))
                                          : ((sgn < 0) ? oz(i + sz) : oz(i));
            const bool behind = !(mk(jb) > 0.5) && ofB > 0.0;
            const vof::PcGfmRow row = vof::pcGfmRow(th, behind, gfmOrder);
            a2 += kd * of * row.aGamma * (tg(j) - T(i));
            if (behind && row.aBehind != 1.0) {
              const double kb = useK ? 0.5 * (kc(i) + kc(jb)) : Dc;
              a3 += (row.aBehind - 1.0) * kb * ofB * (T(jb) - T(i));
            }
          }
      },
      hOpen, q, qb);
  Kokkos::fence();
  pcBudget_.hOpenNew = hOpen;
  pcBudget_.qGfm = q;
  pcBudget_.qBehind = qb;
  pcBudget_.calls += 1;
}

template <class Grid>
void Solver<Grid>::pcApplyDivergenceSource(CCField div) {
  if (!pcEnabled_ && !pcHasUser_)
    return;
  const C3 e = e_;
  const bool hasPc = pcEnabled_, hasUser = pcHasUser_;
  CCConst sp = hasPc ? CCConst(pcSrc_) : CCConst(div);
  CCConst su = hasUser ? CCConst(pcUser_) : CCConst(div);
  Kokkos::parallel_for(
      "peclet::flow::pc_div_source",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                     {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double s = 0.0;
        if (hasPc)
          s += sp(i);
        if (hasUser)
          s += su(i);
        div(i) -= s;
      });
}

template <class Grid>
double Solver<Grid>::pcEffInterfaceEps() const {
  return Kokkos::fmax(pcInterfaceEps_, vofAdv_.wispEps);
}

template <class Grid>
double Solver<Grid>::pcEffPureEps() const {
  return Kokkos::fmax(pcPureEps_, vofAdv_.wispEps);
}

template <class Grid>
void Solver<Grid>::requirePhaseChange(const char* who) const {
  if (!pcEnabled_)
    throw std::runtime_error(std::string(who) +
                             ": phase change is not enabled (call "
                             "enable_phase_change first)");
}

template <class Grid>
double Solver<Grid>::phaseTick() {
  Kokkos::fence();
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_PHASE_CHANGE_HPP
