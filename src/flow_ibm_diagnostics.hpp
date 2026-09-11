/// @file
/// @brief flow — IbmSolver diagnostics: state getters, divergence probes, timers and the
/// outflow/backflow census.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_DIAGNOSTICS_HPP
#define PECLET_FLOW_FLOW_IBM_DIAGNOSTICS_HPP

namespace peclet::flow {

template <class Grid>
typename Solver<Grid>::OutflowBackflow Solver<Grid>::outflowBackflow() {
  OutflowBackflow ob;
  if (!hasOutflow_)
    return ob;
  CCExec space;
  const C3 e = e_;
  const int dims[3] = {e.x, e.y, e.z};
  const long st[3] = {1, e.x, (long)e.x * e.y};
  // Phase 2 C2 (doc/anisotropic_metric.md §4.2, trap 10): `energyInflux` mixes three velocity
  // COMPONENTS, which on anisotropic cells carry three different velocity scales, so it cannot
  // be converted after the fact -- each component is taken to physical units inside the reduce
  // (k_a = velToPhys(a)) and the density with it.  `maxReverse` is scaled there too, i.e. before
  // the max over face planes whose axes have different velocity scales.  Phase 1 left this
  // diagnostic in index units and the Python binding converts nothing, so in cell units every
  // factor here is exactly 1.0 and the reported numbers do not move.
  const double rho = u_.rhoRef * rho_;
  for (int a = 0; a < 3; ++a)
    for (int s = 0; s < 2; ++s) {
      if (bc_[2 * a + s] != 3 || !touchesGlobalFace(2 * a + s))
        continue;
      const int b = (a + 1) % 3, c = (a + 2) % 3;
      const double ka = u_.velToPhys(a), kb = u_.velToPhys(b), kc = u_.velToPhys(c);
      const long sa = st[a], sb = st[b], sc = st[c];
      const int bf = (s == 0) ? G : (dims[a] - G);       // the boundary normal-velocity plane
      const int bic = (s == 0) ? G : (dims[a] - G - 1);  // the outlet-adjacent inner cell
      const double sgn = (s == 0) ? 1.0 : -1.0;          // u.n < 0 <-> sgn*u > 0
      CCConst un = Grid::collocated ? CCConst(a == 0 ? uf_ : a == 1 ? vf_ : wf_) : CCConst(C[a].u);
      CCConst ub = CCConst(C[b].u), uc = CCConst(C[c].u);
      using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>;
      double mx = 0.0, en = 0.0;
      long nrev = 0;
      Kokkos::parallel_reduce(
          "peclet::flow::backflow_census", MD(space, {G, G}, {dims[b] - G, dims[c] - G}),
          KOKKOS_LAMBDA(int p0, int p1, double& lmx, double& len, long& lnr) {
            const long base = (long)p0 * sb + (long)p1 * sc;
            const double back = sgn * un(base + (long)bf * sa);
            if (back > 0.0) {
              const long ic = base + (long)bic * sa;
              const double tb = kb * ub(ic), tc = kc * uc(ic);
              const double bp = ka * back;
              lmx = Kokkos::fmax(lmx, bp);
              len += rho * bp * 0.5 * (bp * bp + tb * tb + tc * tc);
              lnr += 1;
            }
          },
          Kokkos::Max<double>(mx), Kokkos::Sum<double>(en), Kokkos::Sum<long>(nrev));
      ob.maxReverse = std::max(ob.maxReverse, mx);
      ob.energyInflux += en;
      ob.reversed += nrev;
      ob.total += (long)(dims[b] - 2 * G) * (dims[c] - 2 * G);
    }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double d[2] = {ob.energyInflux, 0.0}, dg[2];
    MPI_Allreduce(&ob.maxReverse, &d[1], 1, MPI_DOUBLE, MPI_MAX, comm_);
    MPI_Allreduce(&d[0], &dg[0], 1, MPI_DOUBLE, MPI_SUM, comm_);
    long l[2] = {ob.reversed, ob.total}, lg[2];
    MPI_Allreduce(l, lg, 2, MPI_LONG, MPI_SUM, comm_);
    ob.maxReverse = d[1];
    ob.energyInflux = dg[0];
    ob.reversed = lg[0];
    ob.total = lg[1];
  }
#endif
  ob.fraction = ob.total > 0 ? (double)ob.reversed / (double)ob.total : 0.0;
  return ob;
}

template <class Grid>
std::vector<double> Solver<Grid>::getVelocity(int c) {
  std::vector<double> out = gatherInner(C[c].u);
  const double k = u_.velToPhys(c);  // exactly 1.0 in cell units
  if (k != 1.0)
    for (double& x : out)
      x *= k;
  return out;
}

template <class Grid>
void Solver<Grid>::setVelocity(int c, const std::vector<double>& v) {
  const double k = u_.velToInt(c);
  if (k != 1.0) {
    std::vector<double> vi(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
      vi[i] = v[i] * k;
    scatterInner(C[c].u, vi);
  } else {
    scatterInner(C[c].u, v);
  }
  maskVelocity(c);
  seedFaceFieldFromCells();  // ISSUES sweep item 5 (collocated only; a no-op staggered)
}

template <class Grid>
std::vector<double> Solver<Grid>::getFaceVelocity(int c) {
  std::vector<double> out;
  if constexpr (Grid::collocated) {
    CCField fa[3] = {uf_, vf_, wf_};
    out = gatherInner(fa[c]);
  } else {
    out = gatherInner(C[c].u);
  }
  const double k = u_.velToPhys(c);  // exactly 1.0 in cell units
  if (k != 1.0)
    for (double& x : out)
      x *= k;
  return out;
}

template <class Grid>
std::vector<double> Solver<Grid>::getOpenness(int c) {
  CCField o[3] = {ox_, oy_, oz_};
  return gatherInner(o[c]);
}

template <class Grid>
std::vector<double> Solver<Grid>::getMomentumDiagonal(int c) {
  auto h = Kokkos::create_mirror_view(C[c].AC);
  Kokkos::deep_copy(h, C[c].AC);
  std::vector<double> out((std::size_t)nx_ * ny_ * nz_);
  for (int z = 0; z < nz_; ++z)
    for (int y = 0; y < ny_; ++y)
      for (int x = 0; x < nx_; ++x)
        out[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_] =
            (double)h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y);
  return out;
}

template <class Grid>
std::vector<double> Solver<Grid>::getOpennessProj(int c) {
  const bool gp = ghostProjection_ && oxb_.extent(0) > 0;
  CCField o[3] = {gp ? oxb_ : ox_, gp ? oyb_ : oy_, gp ? ozb_ : oz_};
  return gatherInner(o[c]);
}

template <class Grid>
std::vector<double> Solver<Grid>::getPressure() {
  // Incremental scheme: P_ accumulates the physical pressure. Classical Chorin (!incremental_):
  // derive it on demand from the last projection potential, p = (rho/dt)*phi (CUDA
  // press_from_phi_k).
  std::vector<double> out = gatherInner(incremental_ ? P_ : phi_);
  // Internal pressure -> the caller's units; the Chorin branch first derives p' = (rho'/dt')*phi.
  const double ct = (incremental_ ? 1.0 : rho_ / dt_) * u_.pToPhys();
  if (ct != 1.0)
    for (double& x : out)
      x *= ct;
  return out;
}

template <class Grid>
double Solver<Grid>::maxOpenDivergenceProjected() {
  return maxOpenDivergenceProjectedInternal() * u_.divToPhys();
}

template <class Grid>
double Solver<Grid>::maxOpenDivergenceProjectedInternal() {
  if (!cutcellPressure_)
    return 0.0;
  if constexpr (Grid::collocated)
    return maxOpenDivergenceInternal();  // the collocated branch already measures the face
                                         // field
  for (int c = 0; c < 3; ++c)
    fillVelGhostsTo(C[c].u, c, 0, false);
  divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
             CCConst(oz_), div_, e_, G);
  addWallFluxDivergence(div_);
  double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = 0;
    MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return g;
  }
#endif
  return m;
}

template <class Grid>
double Solver<Grid>::maxOpenDivergence() {
  return maxOpenDivergenceInternal() * u_.divToPhys();
}

template <class Grid>
double Solver<Grid>::maxOpenDivergenceInternal() {
  if (!cutcellPressure_)
    return 0.0;
  if constexpr (Grid::collocated) {
    // Report the residual of the PROJECTED face field uf_ (made divergence-free by project(),
    // ghosts filled). Re-averaging the central-difference-corrected CELL field would instead show
    // the inherent O(h^2) approximate-projection cell divergence -- a property of the scheme, not
    // the solver residual. At an outflow, re-impose the zero-gradient face (matching the
    // staggered diagnostic, whose fillVelGhosts overwrites the mass-conserving outflow
    // correction): the operator zeroes the alpha-divergence, but the raw beta-divergence at the
    // open-boundary corner is otherwise spurious.
    if (hasOutflow_) {
      B3 e{e_.x, e_.y, e_.z};
      CCField fa[3] = {uf_, vf_, wf_};
      for (int a = 0; a < 3; ++a)
        if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1))
          bcNeumannGhost(fa[a], e, G, a, 1);
    }
    if (ghostProjection_ && gpNRows_ >= 0) {
      // Ghost mode: the closed point divergence of the projected face field (same kernel pair
      // as the RHS) — the mode's true residual.
      divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(oxb_), CCConst(oyb_),
                 CCConst(ozb_), div_, e_, G);
      gpDivergDelta(div_, CCConst(uf_), CCConst(vf_), CCConst(wf_), gpOv_, gpNRows_,
                    C3{nx_, ny_, nz_}, e_, G, distributed_);
    } else
      divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_), CCConst(oz_),
                 div_, e_, G);
  } else {
    for (int c = 0; c < 3; ++c)
      fillVelGhosts(c, 0);  // ghosts incl. outflow zero-gradient before the divergence
    if (ghostProjection_ && gpNRows_ >= 0) {
      // Ghost mode: the closed point divergence (same kernels as the RHS) IS the true residual
      // of the mode. (EXPLICIT sliver faces read the corrected stored value here vs u* in the
      // RHS — the only, and rare, departure from the exact identity.)
      divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(oxb_), CCConst(oyb_),
                 CCConst(ozb_), div_, e_, G);
      gpDivergDelta(div_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), gpOv_, gpNRows_,
                    C3{nx_, ny_, nz_}, e_, G, distributed_);
    } else
      divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                 CCConst(oz_), div_, e_, G);
  }
  // The diagnostic must measure the residual of the constraint the projection actually
  // solved, so it carries the same wall-flux source (rung 3). Without this the moving case
  // would report a "divergence error" that is really the wall flux the solve balances.
  addWallFluxDivergence(div_);
  double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = 0;
    MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return g;
  }
#endif
  return m;
}

template <class Grid>
double Solver<Grid>::maxPorousResidual() {
  if (!porous_ || !cutcellPressure_)
    return 0.0;
  for (int c = 0; c < 3; ++c)
    fillVelGhosts(c, 0);
  fillPorousEpsGhosts();  // the SAME eps ghost policy the projection used (the coupling deposit
                          // rewrites the ghosts between project() and this diagnostic)
  divergOpenEps(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                CCConst(oz_), CCConst(epsField_), div_, e_, G);
  {  // add back the SAME d(eps)/dt source the projection used (depsdt_ from the last project())
    CCExec space;
    C3 e = e_;  // local copy — capturing e_ in the KOKKOS_LAMBDA would read this-> on the device
    CCField d = div_, dd = depsdt_;
    const bool useDt = porousDepsDt_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::porous_resid", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * e.x * e.y;
          if (useDt)
            d(i) += dd(i);  // residual of the SAME constraint the projection solved
        });
  }
  double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g = 0;
    MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return g;
  }
#endif
  return m;
}

template <class Grid>
long Solver<Grid>::lastPressureIterations() const {
  return lastPressureIters_;
}

template <class Grid>
std::vector<std::array<int, 3>> Solver<Grid>::pressureMgLevelRatios() const {
  std::vector<std::array<int, 3>> out;
  for (const C3& r : mg_.levelRatios())
    out.push_back({r.x, r.y, r.z});
  return out;
}

template <class Grid>
bool Solver<Grid>::pressureSolveFailed() const {
  return lastPressureFailed_;
}

template <class Grid>
double Solver<Grid>::lastStepSeconds() const {
  return tStep_;
}

template <class Grid>
double Solver<Grid>::lastPredictorSeconds() const {
  return tPredictor_;
}

template <class Grid>
double Solver<Grid>::lastMomentumSeconds() const {
  return tMomentum_;
}

template <class Grid>
double Solver<Grid>::lastProjectionSeconds() const {
  return tProjection_;
}

template <class Grid>
double Solver<Grid>::lastPressureAllreduceSeconds() const {
  return mg_.allreduceSeconds();
}

template <class Grid>
long Solver<Grid>::lastPressureAllreduceCount() const {
  return mg_.allreduceCount();
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_DIAGNOSTICS_HPP
