/// @file
/// @brief flow — IbmSolver cut-cell scalar transport (doc/scalar_ibm_design.md): the geometry
/// record of WO-2 — the lazy build `ensureScalarCutGeometry` (§2.5), its invalidation and the
/// diagnostics getters — and the single-phase operator, solve and outputs of WO-3 (§1.4, §4, §5.1,
/// §8). The block kernels are `scalar_cutcell_geometry.hpp` and `scalar_cutcell_operator.hpp`, the
/// Krylov driver `scalar_krylov.hpp`; the per-cell geometry and the probe ladder are core's
/// `peclet::core::scheme` (WO-1).
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_SCALARS_CUTCELL_HPP
#define PECLET_FLOW_FLOW_IBM_SCALARS_CUTCELL_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::invalidateScalarCutGeometry() {
  ++scgVersion_;
  scalarOpenValid_ = false;  // WO-5b: the captured open-face planes belong to the old block
}

template <class Grid>
void Solver<Grid>::ensureScalarCutGeometry() {
  if (!geometryBuilt_)
    throw std::runtime_error(
        "scalar cut-cell geometry: no geometry yet -- call set_solid, set_solid_from_scene or "
        "set_pressure_geometry first");
  if (scg_.version == scgVersion_)
    return;
  const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
  scg::ScalarCutGeometry r;
  r.kappa = CCField("peclet::flow::scg_kappa", n_);
  r.sax = CCField("peclet::flow::scg_sax", n_);
  r.say = CCField("peclet::flow::scg_say", n_);
  r.saz = CCField("peclet::flow::scg_saz", n_);
  r.unknown = CCField("peclet::flow::scg_unknown", n_);
  const CCConst sd(sdf_);
  // 1. apertures (every face of every inner cell + the high face at ext - G; not exchanged)
  scg::buildApertures(r.sax, r.say, r.saz, sd, e_, G);
  // 2. kappa, unknown flags, facet counts
  scg::CellPass cp{Kokkos::View<std::int8_t*, CCMem>("peclet::flow::scg_nfac", nInner),
                   Kokkos::View<std::int8_t*, CCMem>("peclet::flow::scg_kind", nInner)};
  scg::buildCells(r.kappa, r.unknown, cp, sd, r.sax, r.say, r.saz, e_, G, u_.hp);
  // 3. compact facet list in x-fastest cell order + the cut-cell CSR
  scg::buildFacets(r.fac, cp, sd, r.sax, r.say, r.saz, e_, G, u_.hp);
  // 4. halo kappa / flags (the ordinary G = 2 exchange, which wraps periodically), then clear the
  //    flags beyond a non-periodic global face (rank rule touchesGlobalFace)
  fillGhosts(r.kappa);
  fillGhosts(r.unknown);
  for (int face = 0; face < 6; ++face)
    if (bc_[face] != 0 && touchesGlobalFace(face))
      scg::zeroGhostSlab(r.unknown, e_, G, face);
  // body id per facet: the scene owner AT the facet centroid (the hydro-force precedent: query at
  // the point, never from the cell-centred cutOwner_); 0 with raw-SDF geometry (§2.7)
  if (hasScene_ && sceneQ_ && r.fac.n > 0) {
    const auto q = sceneQ_->view();
    const SceneMap sm = sceneMap();
    const C3 og = og_, e = e_;
    const int g = G;
    const double h0 = u_.hp[0], h1 = u_.hp[1], h2 = u_.hp[2];
    auto fcell = r.fac.cell;
    auto fcen = r.fac.centroid;
    auto fbody = r.fac.body;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scg_body", Kokkos::RangePolicy<CCExec>(space, 0, r.fac.n),
        KOKKOS_LAMBDA(const long f) {
          const long i = fcell(f);
          const long x = i % e.x, y = (i / e.x) % e.y, z = i / ((long)e.x * e.y);
          const peclet::core::Vec3<double> p{
              sm.a[0] + sm.b[0] * ((double)(x - g + og.x) + fcen(f, 0) / h0),
              sm.a[1] + sm.b[1] * ((double)(y - g + og.y) + fcen(f, 1) / h1),
              sm.a[2] + sm.b[2] * ((double)(z - g + og.z) + fcen(f, 2) / h2)};
          fbody(f) = q.owner(p);
        });
    space.fence();
  }
  // 5. the fluid probe ladder per facet (§3.3)
  scg::buildFluidProbes(r.fac, sd, CCConst(r.unknown), e_, u_.hp);
  // census, summed over ranks
  scg::ScalarCutCensus c =
      scg::localCensus(r.fac, cp, CCConst(r.kappa), CCConst(r.unknown), CCConst(r.sax),
                       CCConst(r.say), CCConst(r.saz), e_, G, u_.vol);
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    long cnt[10] = {c.numUnknowns, c.numCutCells, c.numFacets, c.numTwoSided, c.numThinSolid,
                    c.numSealed,   c.rungs[0],    c.rungs[1],  c.rungs[2],    c.rungs[3]};
    long sum[10];
    MPI_Allreduce(cnt, sum, 10, MPI_LONG, MPI_SUM, comm_);
    double v = c.sealedVolume, vs = 0.0;
    MPI_Allreduce(&v, &vs, 1, MPI_DOUBLE, MPI_SUM, comm_);
    c.numUnknowns = sum[0];
    c.numCutCells = sum[1];
    c.numFacets = sum[2];
    c.numTwoSided = sum[3];
    c.numThinSolid = sum[4];
    c.numSealed = sum[5];
    for (int k = 0; k < 4; ++k)
      c.rungs[k] = sum[6 + k];
    c.sealedVolume = vs;
  }
#endif
  const double l = u_.lenToPhys();
  c.sealedVolume *= l * l * l;
  r.census = c;
  r.version = scgVersion_;
  scg_ = r;
}

template <class Grid>
const scg::ScalarCutGeometry& Solver<Grid>::scalarCutGeometry() {
  ensureScalarCutGeometry();
  return scg_;
}

template <class Grid>
std::vector<double> Solver<Grid>::scalarGeometryField(int which) {
  ensureScalarCutGeometry();
  const CCField f[5] = {scg_.kappa, scg_.sax, scg_.say, scg_.saz, scg_.unknown};
  if (which < 0 || which > 4)
    throw std::runtime_error("scalarGeometryField: which must be 0..4");
  return gatherInner(f[which]);
}

template <class Grid>
scg::ScalarCutCensus Solver<Grid>::scalarCutCensus() {
  ensureScalarCutGeometry();
  return scg_.census;
}

// ================================================================================================
// WO-3: the single-phase cut-cell operator, the Krylov solve, steady + transient diffusion.
// ================================================================================================

template <class Grid>
bool Solver<Grid>::isCutcellScalar(const std::string& name) const {
  for (const auto& sc : scalars_)
    if (sc.name == name)
      return sc.cutcell;
  return false;
}

template <class Grid>
ScalarField& Solver<Grid>::cutcellScalar(const std::string& name, const char* who) {
  for (auto& sc : scalars_)
    if (sc.name == name) {
      if (!sc.cutcell)
        throw std::invalid_argument(std::string(who) + ": '" + name +
                                    "' is not a cut-cell scalar (add_scalar(..., cutcell=True))");
      return sc;
    }
  throw std::invalid_argument(std::string(who) + ": no scalar named '" + name + "'");
}

template <class Grid>
int Solver<Grid>::scalarNumBodies() const {
  return (hasScene_ && nInst_ > 0) ? nInst_ : 1;
}

template <class Grid>
void Solver<Grid>::setScalarWall(const std::string& name, int type, double value,
                                 double coefficient, int instance) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_wall").cut;
  if (type < 0 || type > 2)
    throw std::invalid_argument("set_scalar_wall: type must be 'neumann', 'dirichlet' or 'robin'");
  if (!std::isfinite(value))
    throw std::invalid_argument("set_scalar_wall: value must be finite");
  if (type == 2 && !(coefficient >= 0.0))
    throw std::invalid_argument("set_scalar_wall: the robin coefficient k must be >= 0");
  const ScalarWallSpec w{type, value, type == 2 ? coefficient : 0.0};
  if (instance < 0) {  // every body: the default, and no override survives
    st.wallDefault = w;
    st.wallInstance.clear();
    return;
  }
  if (!hasScene_)
    throw std::invalid_argument(
        "set_scalar_wall: instance= needs a scene (set_scene + set_solid_from_scene); raw-SDF "
        "geometry has one body, set with instance=None");
  if (instance >= nInst_)
    throw std::invalid_argument("set_scalar_wall: instance " + std::to_string(instance) +
                                " out of range (the scene has " + std::to_string(nInst_) + ")");
  for (auto it = st.wallInstance.begin(); it != st.wallInstance.end(); ++it)
    if (it->first == instance) {
      st.wallInstance.erase(it);
      break;
    }
  st.wallInstance.emplace_back(instance, w);
}

template <class Grid>
void Solver<Grid>::setScalarSource(const std::string& name, double S) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_source").cut;
  if (!std::isfinite(S))
    throw std::invalid_argument("set_scalar_source: the source must be finite");
  st.sourceConst = S;
  st.sourceField = CCField();
  st.sourceIsField = false;
}

template <class Grid>
void Solver<Grid>::setScalarSourceField(const std::string& name, const std::vector<double>& S) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_source").cut;
  if (S.size() != (std::size_t)nx_ * ny_ * nz_)
    throw std::invalid_argument(
        "set_scalar_source: the array must have this rank's (nx, ny, nz) "
        "shape");
  if (st.sourceField.extent(0) != n_)
    st.sourceField = CCField(name + "_cc_source", n_);
  scatterInner(st.sourceField, S);
  st.sourceConst = 0.0;
  st.sourceIsField = true;
}

template <class Grid>
void Solver<Grid>::setScalarTolerance(const std::string& name, double rtol) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_tolerance").cut;
  if (!(rtol > 0.0) || !std::isfinite(rtol))
    throw std::invalid_argument("set_scalar_tolerance: rtol must be finite and > 0");
  st.rtol = rtol;
}

template <class Grid>
void Solver<Grid>::setScalarMaxIterations(const std::string& name, int maxit) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_max_iterations").cut;
  if (maxit < 1)
    throw std::invalid_argument("set_scalar_max_iterations: maxit must be >= 1");
  st.maxit = maxit;
}

template <class Grid>
void Solver<Grid>::setScalarBcProfile(const std::string& name, int face,
                                      const std::vector<double>& prof, int n1, int n2) {
  ScalarField& sc = cutcellScalar(name, "set_scalar_bc");
  if (face < 0 || face > 5)
    throw std::invalid_argument("set_scalar_bc: face out of range");
  int t1, t2;
  sco::faceTangents(face / 2, t1, t2);
  const int n[3] = {nx_, ny_, nz_};
  if (n1 != n[t1] || n2 != n[t2] || prof.size() != (std::size_t)n1 * n2)
    throw std::invalid_argument("set_scalar_bc: a profile must have shape (N_t1, N_t2) = (" +
                                std::to_string(n[t1]) + ", " + std::to_string(n[t2]) +
                                ") -- this rank's face cells in the tangential axes in x, y, z "
                                "order");
  for (double v : prof)
    if (!std::isfinite(v))
      throw std::invalid_argument("set_scalar_bc: the profile must be finite");
  sc.bc[face] = 2;  // a profile is a Dirichlet value
  sc.bcVal[face] = 0.0;
  sc.cut->hasProfile[face] = true;
  sc.cut->profile[face] = prof;
  sc.cut->profileN[face][0] = n1;
  sc.cut->profileN[face][1] = n2;
}

template <class Grid>
void Solver<Grid>::scalarCutRefusals(const ScalarField& sc) const {
  const std::string who = "cut-cell scalar '" + sc.name + "': ";
  if (porous_)
    throw std::runtime_error(who + "porous (volume-averaged) continuity is not supported");
  if (hasMovingInstance())
    throw std::runtime_error(who + "moving scene instances are not supported (static geometry)");
  if (sc.dmask.extent(0) == n_)
    throw std::runtime_error(who + "the per-cell Dirichlet mask is not supported");
  // §13 Q13, decided by G9b (WO-5): the ghost projection's constraint is the binary-openness
  // divergence PLUS wall-anchored closure terms (gpDivergDelta), so no openness-weighted face flux
  // of its field is discretely divergence-free — a constant concentration drifted by 0.80 in 50
  // Stokes steps at bulk Courant 0.47 (staggered and every other collocated scheme: 0 exactly).
  // Covers the collocated 'ghost' scheme (the AUTO default) and the staggered verification path.
  if (ghostProjection_)
    throw std::runtime_error(
        who + "the ghost projection ('ghost' collocated scheme, or diagnostics." +
        "set_ghost_projection) leaves no discretely divergence-free face flux to advect with " +
        "(doc/scalar_ibm_design.md §6.1, gate G9b); the accepted projections are the collocated " +
        "schemes 'gauge-exact', 'plain' and 'embed' (set_collocated_scheme, before set_solid) "
        "and " +
        "the staggered Solver without set_ghost_projection");
  static const char* kFace[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};
  for (int f = 0; f < 6; ++f) {
    const bool flowPeriodic = bc_[f] == 0, scalarPeriodic = sc.bc[f] == 0;
    if (flowPeriodic != scalarPeriodic)
      throw std::runtime_error(
          who + "face " + kFace[f] + " is " + (scalarPeriodic ? "periodic" : "non-periodic") +
          " for the scalar but " + (flowPeriodic ? "periodic" : "non-periodic") +
          " for the flow (set_scalar_bc / set_domain_bc must agree)");
    if (bc_[f] == 2 && sc.bc[f] != 2)
      throw std::runtime_error(who + "the inflow face " + kFace[f] +
                               " needs a scalar 'dirichlet' value (set_scalar_bc)");
    // WO-5b (open, reported): on the collocated grid the projected face field does not keep the
    // flux the projection constrained on a high-side open face -- projectCorrectVelocities' plain
    // ghost fill of uf_/vf_/wf_ wraps that plane from the opposite face before the outflow
    // correction is added to it -- so there is no divergence-free boundary flux to capture (a
    // constant drifted by 0.48 in the outlet column in one step). Refused until decided.
    if constexpr (Grid::collocated)
      if (bc_[f] == 2 || bc_[f] == 3)
        throw std::runtime_error(
            who + "face " + kFace[f] + " is a flow " + (bc_[f] == 2 ? "inflow" : "outflow") +
            " face: on the collocated grid the projected face field does not keep the boundary "
            "flux the projection constrained, so open domain faces are not supported for a "
            "cut-cell scalar there yet -- use the staggered Solver (doc/scalar_ibm_design.md "
            "§6.5)");
  }
}

template <class Grid>
void Solver<Grid>::scalarCutMatvec(ScalarField& sc, CCField y, CCField x) {
  ScalarCutState& st = *sc.cut;
  fillGhosts(x);  // one G = 2 exchange covers the 7-point part and the +-2-reach probes
  applyCutcellOp(y, CCConst(x), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_, G);
  sco::overlayApply(y, CCConst(x), scg_.fac, st.cw);
}

template <class Grid>
typename Solver<Grid>::ScalarFaceFlux Solver<Grid>::scalarFaceFlux(int a) const {
  // §6.1 (D5): constant preservation needs sum_f F = 0 per cell, and only the projection's own
  // divergence kernel states which openness that sum carries: `divergOpen(f, oxb_)` (+ the
  // closure delta) in ghost-projection mode, `constraintDivergence` = `divergOpen(f, ox_)` in every
  // other non-porous branch of both grids (porous continuity is refused for a cut-cell scalar).
  const bool gp = ghostProjection_ && oxb_.extent(0) > 0;
  const CCField o[3] = {gp ? oxb_ : ox_, gp ? oyb_ : oy_, gp ? ozb_ : oz_};
  ScalarFaceFlux f;
  f.open = o[a];
  if constexpr (Grid::collocated)
    f.vel = a == 0 ? uf_ : (a == 1 ? vf_ : wf_);  // the projected MAC face field
  else
    f.vel = C[a].u;  // the stored staggered velocity IS the projected face velocity
  return f;
}

template <class Grid>
void Solver<Grid>::scalarCaptureOpenFaceFlux() {
  // WO-5b (§6.5, ruling D-WO5-3). Called right after the projection: the high-side boundary face
  // is the first GHOST index, which the next plain ghost fill (advanceScalars' own, the VoF
  // bridge's) wraps from the opposite face, so the flux the projection constrained there is read
  // now. The plane spans the full transverse extent: the transverse ghost rows are what the small
  // flags of ghost layer 1 read (§6.3), with the values the neighbour holds for its own rows.
  if (!hasBc_)
    return;
  // Captured whether or not a cut-cell scalar exists yet (one plane copy per open face), so a
  // scalar added to a developed flow can be solved before the next step.
  bool anyOpen = false;
  for (int f = 0; f < 6; ++f)
    anyOpen = anyOpen || bc_[f] == 2 || bc_[f] == 3;
  if (!anyOpen)
    return;
  const int ext[3] = {e_.x, e_.y, e_.z};
  const long st[3] = {1, (long)e_.x, (long)e_.x * e_.y};
  for (int f = 0; f < 6; ++f) {
    if (!(bc_[f] == 2 || bc_[f] == 3) || !touchesGlobalFace(f))
      continue;
    const int a = f / 2, side = f % 2;
    int t1, t2;
    sco::faceTangents(a, t1, t2);
    const int m1 = ext[t1], m2 = ext[t2];
    if (scalarOpenFlux_[f].extent(0) != (std::size_t)m1 * m2)
      scalarOpenFlux_[f] = CCField("peclet::flow::scalar_open_flux", (std::size_t)m1 * m2);
    const ScalarFaceFlux ff = scalarFaceFlux(a);
    const CCConst open(ff.open), vel(ff.vel);
    CCField pl = scalarOpenFlux_[f];
    const long sa = st[a], s1 = st[t1], s2 = st[t2];
    const long bfa = side == 0 ? G : ext[a] - G;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scalar_capture_open_flux", MDRange2<CCExec>(space, {0, 0}, {m1, m2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long bf = bfa * sa + (long)j1 * s1 + (long)j2 * s2;
          pl((long)j1 + (long)j2 * m1) = open(bf) * vel(bf);
        });
    space.fence();
  }
  scalarOpenValid_ = true;
}

template <class Grid>
void Solver<Grid>::scalarCutAdvection(ScalarField& sc, bool steady) {
  ScalarCutState& st = *sc.cut;
  const scg::ScalarCutGeometry& gm = scg_;
  const CCConst unk(gm.unknown), kap(gm.kappa);
  for (int a = 0; a < 3; ++a)
    if (st.phi[a].extent(0) != n_)
      st.phi[a] = CCField(sc.name + "_cc_phi", n_);
  // the face flux of the projection in force, guarded (sco::faceFluxes). A steady solve is called
  // outside step(), so it fills the face velocity's ghosts itself (advanceScalars did for an
  // advance).
  // WO-5b (§6.5): the open (inflow/outflow) global faces this rank touches take the boundary-face
  // flux the last projection constrained (scalarCaptureOpenFaceFlux), written over the guard's 0.
  bool anyOpen = false, openHere[6];
  for (int f = 0; f < 6; ++f) {
    const bool o = bc_[f] == 2 || bc_[f] == 3;
    anyOpen = anyOpen || o;
    openHere[f] = o && scalarOpenValid_ && touchesGlobalFace(f);
    st.openFace[f] = false;
    st.openInflow[f] = bc_[f] == 2;
  }
  long nFlux = 0, nGuard = 0;
  double vmax = 0.0;
  for (int a = 0; a < 3; ++a) {
    const ScalarFaceFlux ff = scalarFaceFlux(a);
    if (steady)
      fillGhosts(ff.vel);
    vmax = std::fmax(vmax, sco::maxabsLocal(CCConst(ff.vel), e_, G));
    sco::faceFluxes(st.phi[a], CCConst(ff.vel), CCConst(ff.open), unk, a, e_);
    for (int side = 0; side < 2; ++side)
      if (openHere[2 * a + side])
        sco::openFaceFluxes(st.phi[a], CCConst(scalarOpenFlux_[2 * a + side]), unk, a, side, e_, G);
    long n1 = 0, n2 = 0;
    sco::faceFluxCountsLocal(CCConst(st.phi[a]), CCConst(ff.vel), CCConst(ff.open), a, e_, G, n1,
                             n2);
    nFlux += n1;
    nGuard += n2;
  }
  // (the third entry: A2's census max_cell_peclet, max |phi_a| / (Lam' w_a) over interior faces)
  double mx[3] = {
      sco::maxAbsFluxLocal(CCConst(st.phi[0]), CCConst(st.phi[1]), CCConst(st.phi[2]), e_, G), vmax,
      sco::maxCellPecletLocal(CCConst(st.phi[0]), CCConst(st.phi[1]), CCConst(st.phi[2]), unk,
                              st.lam, u_.w, e_, G)};
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double o[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(mx, o, 3, MPI_DOUBLE, MPI_MAX, comm_);
    mx[0] = o[0];
    mx[1] = o[1];
    mx[2] = o[2];
  }
#endif
  st.maxCellPeclet = mx[2];
  // WO-5 (open, reported): without the cut-cell pressure operator no projection runs and no
  // openness exists (set_solid(..., cutcell_pressure=False)), so there is no discretely
  // divergence-free face flux to advect with (§6.1); a moving fluid is refused rather than left
  // silently unadvected. A fluid at rest (diffusion only, WO-3/WO-4) is unaffected.
  if (!cutcellPressure_ && mx[1] > 0.0)
    throw std::runtime_error("cut-cell scalar '" + sc.name +
                             "': the fluid moves but no cut-cell projection exists to advect "
                             "with -- use set_solid(..., cutcell_pressure=True) or "
                             "set_pressure_geometry (doc/scalar_ibm_design.md §6.1)");
  // WO-5b: the flux of an open domain face is the one the projection constrained, captured by
  // step(); before the first projection there is none to advect with.
  if (anyOpen && !scalarOpenValid_ && mx[1] > 0.0)
    throw std::runtime_error("cut-cell scalar '" + sc.name +
                             "': the fluid moves and the domain has an inflow/outflow face, but no "
                             "projection has supplied that face's flux yet -- advance with step() "
                             "(doc/scalar_ibm_design.md §6.5)");
  st.advecting = mx[0] > 0.0;
  st.numSmall = 0;
  st.numImplicitFaces = 0;
  st.bulkCourant = 0.0;
  st.numFluxFaces = 0;
  st.numGuardedFaces = 0;
  if (!st.advecting) {
    long cnt[1] = {nGuard};
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      long o[1];
      MPI_Allreduce(cnt, o, 1, MPI_LONG, MPI_SUM, comm_);
      cnt[0] = o[0];
    }
#endif
    st.numGuardedFaces = cnt[0];
    return;
  }
  if (st.small.extent(0) != n_)
    st.small = CCField(sc.name + "_cc_small", n_);
  if (st.outflow.extent(0) != n_)
    st.outflow = CCField(sc.name + "_cc_outflow", n_);
  const CCConst px(st.phi[0]), py(st.phi[1]), pz(st.phi[2]);
  if (steady) {  // §6.7: implicit FOU on every face; no classification
    Kokkos::deep_copy(CCExec(), st.small, 0.0);
  } else {
    // §6.3: C_bulk = the global MAX over full cells of dt Out / V, then the per-cell flags over the
    // inner cells and ghost layer 1 (no exchange)
    double cb = sco::bulkCourantLocal(px, py, pz, kap, unk, CCConst(gm.sax), CCConst(gm.say),
                                      CCConst(gm.saz), dt_, e_, G);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double o = 0.0;
      MPI_Allreduce(&cb, &o, 1, MPI_DOUBLE, MPI_MAX, comm_);
      cb = o;
    }
#endif
    st.bulkCourant = cb;
    sco::smallCells(st.small, px, py, pz, kap, unk, dt_, std::fmax(0.5, cb), e_, G);
  }
  long cnt[4] = {sco::countSmallLocal(CCConst(st.small), e_, G), 0, nFlux, nGuard};
  for (int a = 0; a < 3; ++a)
    cnt[1] += sco::countImplicitLocal(CCConst(st.phi[a]), CCConst(st.small), unk, a, steady, e_, G);
  for (int f = 0; f < 6; ++f) {
    st.openFace[f] = openHere[f];
    if (!openHere[f])
      continue;
    long n1 = 0, n2 = 0;
    sco::openFaceCountsLocal(CCConst(st.phi[f / 2]), CCConst(st.small), unk, st.openInflow[f],
                             f / 2, f % 2, steady, e_, G, n1, n2);
    cnt[2] += n1;
    cnt[1] += n2;
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    long o[4];
    MPI_Allreduce(cnt, o, 4, MPI_LONG, MPI_SUM, comm_);
    for (int k = 0; k < 4; ++k)
      cnt[k] = o[k];
  }
#endif
  st.numSmall = cnt[0];
  st.numImplicitFaces = cnt[1];
  st.numFluxFaces = cnt[2];
  st.numGuardedFaces = cnt[3];
}

template <class Grid>
void Solver<Grid>::advanceScalarCutCell(ScalarField& sc) {
  scalarCutAssembleSolve(sc, false);
}

template <class Grid>
void Solver<Grid>::solveScalarSteady(const std::string& name) {
  scalarCutAssembleSolve(cutcellScalar(name, "solve_scalar_steady"), true);
}

template <class Grid>
void Solver<Grid>::scalarCutAssembleSolve(ScalarField& sc, bool steady) {
  scalarCutRefusals(sc);
  ensureScalarCutGeometry();
  if (!steady && !(dt_ > 0.0))
    throw std::runtime_error("cut-cell scalar '" + sc.name + "': set_dt first");
  ScalarCutState& st = *sc.cut;
  const scg::ScalarCutGeometry& gm = scg_;
  const CCConst unk(gm.unknown), kap(gm.kappa);
  auto alloc = [&](CCField& f, const char* tag) {
    if (f.extent(0) != n_)
      f = CCField(sc.name + tag, n_);
  };
  alloc(st.SAC, "_cc_sac");
  alloc(st.kr, "_cc_r");
  alloc(st.krh, "_cc_rh");
  alloc(st.kp, "_cc_p");
  alloc(st.kv, "_cc_v");
  alloc(st.kt, "_cc_t");
  alloc(st.kz, "_cc_z");
  alloc(st.kz2, "_cc_z2");
  if (st.cw.extent(0) != (std::size_t)gm.fac.n) {
    st.cw = Kokkos::View<double*, CCMem>(sc.name + "_cc_cw", gm.fac.n);
    st.rw = Kokkos::View<double*, CCMem>(sc.name + "_cc_rw", gm.fac.n);
  }
  // internal coefficients from the stored physical values (§1.2, at advance time)
  const double lam = sc.Dphys * u_.diffToInt();
  const double idt = steady ? 0.0 : 1.0 / dt_;
  const double srcFactor = u_.divToInt();
  st.lam = lam;
  st.idt = idt;
  st.dt = steady ? 0.0 : dt_;
  st.srcFactor = srcFactor;
  st.steady = steady;
  st.numUnknowns = gm.census.numUnknowns;
  // the field: identity-row cells hold exactly 0 (§1.3), the time base is c^n
  sco::zeroNonUnknown(sc.c, unk, e_, G);
  Kokkos::deep_copy(CCExec(), sc.cOld, sc.c);
  // bands
  sco::buildBands(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, kap, unk, CCConst(gm.sax),
                  CCConst(gm.say), CCConst(gm.saz), lam, idt, u_.w, e_, G);
  // the per-body wall table, resolved and converted (§4.2)
  const int nb = scalarNumBodies();
  st.wtType.assign(nb, st.wallDefault.type);
  st.wtK.assign(nb, st.wallDefault.coefficient);
  st.wtG.assign(nb, st.wallDefault.value);
  st.wtQ.assign(nb, st.wallDefault.value);
  for (const auto& [inst, w] : st.wallInstance)
    if (inst >= 0 && inst < nb) {
      st.wtType[inst] = w.type;
      st.wtK[inst] = w.coefficient;
      st.wtG[inst] = w.value;
      st.wtQ[inst] = w.value;
    }
  const double sp = u_.speedToInt();
  for (int b = 0; b < nb; ++b) {
    st.wtK[b] = st.wtType[b] == 2 ? st.wtK[b] * sp : 0.0;
    st.wtQ[b] = st.wtType[b] == 0 ? st.wtQ[b] * sp : 0.0;
    st.wtG[b] = st.wtType[b] == 0 ? 0.0 : st.wtG[b];
  }
  sco::WallTable wt;
  wt.nb = nb;
  wt.type = Kokkos::View<int*, CCMem>("peclet::flow::sco_wt_type", nb);
  wt.k = Kokkos::View<double*, CCMem>("peclet::flow::sco_wt_k", nb);
  wt.gval = Kokkos::View<double*, CCMem>("peclet::flow::sco_wt_g", nb);
  wt.q = Kokkos::View<double*, CCMem>("peclet::flow::sco_wt_q", nb);
  {
    auto ht = Kokkos::create_mirror_view(wt.type);
    auto hk = Kokkos::create_mirror_view(wt.k);
    auto hg = Kokkos::create_mirror_view(wt.gval);
    auto hq = Kokkos::create_mirror_view(wt.q);
    for (int b = 0; b < nb; ++b) {
      ht(b) = st.wtType[b];
      hk(b) = st.wtK[b];
      hg(b) = st.wtG[b];
      hq(b) = st.wtQ[b];
    }
    Kokkos::deep_copy(wt.type, ht);
    Kokkos::deep_copy(wt.k, hk);
    Kokkos::deep_copy(wt.gval, hg);
    Kokkos::deep_copy(wt.q, hq);
  }
  sco::facetCoefficients(st.cw, st.rw, gm.fac, unk, wt, lam);
  // right-hand side: kappa (idt c^n + S') + the walls' rw
  if (st.sourceIsField && st.sourceField.extent(0) != n_)
    throw std::runtime_error("cut-cell scalar '" + sc.name +
                             "': the per-cell source no longer matches this rank's block "
                             "(redistributed?) -- call set_scalar_source again");
  const bool useField = st.sourceIsField;
  sco::buildRhs(sc.b, CCConst(sc.cOld), kap, unk, idt, st.sourceConst,
                CCConst(useField ? st.sourceField : sc.cOld), useField, srcFactor, e_, G);
  sco::overlayRhs(sc.b, gm.fac, st.rw);
  // Dirichlet domain faces (§1.4): fold 2 Lam w_a a_bf into the diagonal and the rhs
  const CCField saAx[3] = {gm.sax, gm.say, gm.saz};
  bool anyDir = false;
  for (int f = 0; f < 6; ++f) {
    st.dirFace[f] = false;
    if (sc.bc[f] != 2 || bc_[f] == 0 || !touchesGlobalFace(f))
      continue;
    const int a = f / 2, side = f % 2;
    int t1, t2;
    sco::faceTangents(a, t1, t2);
    const int n[3] = {nx_, ny_, nz_};
    const long nf = (long)n[t1] * n[t2];
    Kokkos::View<double*, CCMem> gv(sc.name + "_cc_gface", nf);
    if (st.hasProfile[f]) {
      if (st.profileN[f][0] != n[t1] || st.profileN[f][1] != n[t2])
        throw std::runtime_error("cut-cell scalar '" + sc.name +
                                 "': the domain-face profile no longer matches this rank's block");
      auto hv = Kokkos::create_mirror_view(gv);
      for (long k = 0; k < nf; ++k)
        hv(k) = st.profile[f][(std::size_t)k];
      Kokkos::deep_copy(gv, hv);
    } else {
      Kokkos::deep_copy(gv, sc.bcVal[f]);
    }
    st.gFace[f] = gv;
    st.dirFace[f] = true;
    anyDir = true;
    sco::dirichletFaceFold(sc.AC, sc.b, unk, CCConst(saAx[a]), gv, lam * u_.w[a], a, side, e_, G);
  }
  // advection (WO-5, §6): the face flux of the projection, the small-cell split, the implicit FOU
  // bands (their outflow lumps into the surrogate diagonal through AC) and the explicit faces' net
  // outflow at c^n on the rhs. Skipped entirely when no face carries flux (the WO-4 operator).
  scalarCutAdvection(sc, steady);
  if (st.advecting) {
    const CCConst px(st.phi[0]), py(st.phi[1]), pz(st.phi[2]);
    sco::advectionBands(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, st.outflow, px, py, pz,
                        CCConst(st.small), unk, steady, e_, G);
    if (!steady) {
      fillGhosts(sc.cOld);  // c^n's ghosts: the reconstruction reaches two cells past the face
      for (int a = 0; a < 3; ++a) {
        if (st.Phi[a].extent(0) != n_)
          st.Phi[a] = CCField(sc.name + "_cc_Phi", n_);
        sco::explicitFaceFluxes(st.Phi[a], CCConst(st.phi[a]), CCConst(sc.cOld), unk,
                                CCConst(st.small), a, sc.scheme, false, e_, G);
      }
      sco::explicitAdvectionRhs(sc.b, CCConst(st.Phi[0]), CCConst(st.Phi[1]), CCConst(st.Phi[2]),
                                unk, e_, G);
    }
    // WO-5b: the open domain faces' rows (§1.4): inflow F_in g on the rhs, outflow F_out c_i.
    // A2: a steady solve also collects their implicit outflow in omega_open (the advective
    // surrogate's coarse mass), clamped at 0.
    if (steady) {
      if (st.omegaOpen.extent(0) != n_)
        st.omegaOpen = CCField(sc.name + "_cc_omega_open", n_);
      Kokkos::deep_copy(CCExec(), st.omegaOpen, 0.0);
    }
    for (int f = 0; f < 6; ++f)
      if (st.openFace[f])
        sco::openFaceAdvection(sc.AC, sc.b, st.outflow, CCConst(sc.cOld), CCConst(st.phi[f / 2]),
                               CCConst(st.small), unk, st.gFace[f], st.openInflow[f], f / 2, f % 2,
                               steady, e_, G, st.omegaOpen, steady);
    if (steady)
      sco::clampNonNegative(st.omegaOpen, e_, G);
  }
  // the level-0 surrogate diagonal (§4.3)
  sco::surrogateDiagonal(st.SAC, CCConst(sc.AC), gm.fac, st.cw);
  // the singular case (§5.1): steady, no facet with G > 0, no Dirichlet domain face anywhere
  double sing[2] = {sco::maxFacetLocal(st.cw), anyDir ? 1.0 : 0.0};
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double gs[2];
    MPI_Allreduce(sing, gs, 2, MPI_DOUBLE, MPI_MAX, comm_);
    sing[0] = gs[0];
    sing[1] = gs[1];
  }
#endif
  st.singular = steady && !(sing[0] > 0.0) && !(sing[1] > 0.0);
  st.built = true;
  // ScalarMG (§5.2): the level table once per geometry version and block (VelocityMG's rule), the
  // surrogate on every level rebuilt with the operator (ruling D-WO3-3)
  if (!st.mg || st.mgVersion != gm.version || st.mgN != n_) {
    st.mg = std::make_shared<ScalarMG>();
    st.mg->setMetric(u_.w, u_.hp);
    const bool per[3] = {bc_[0] == 0, bc_[2] == 0, bc_[4] == 0};
    st.mg->setPeriodic(per);
    ScalarMG::Fill fill0 = [this](CCField f) { fillGhosts(f); };
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      st.mg->initMpi(*dec_, comm_, og_, fill0);
    else
#endif
      st.mg->init(nx_, ny_, nz_, fill0);
    st.mgVersion = gm.version;
    st.mgN = n_;
  }
  {
    ScalarMG::Inputs in;
    in.lam = lam;
    in.idt = idt;
    // the level rule: transient with kappa_A = 1 + 4 dt' D' sum_a w_a < 13 -> level 0 alone
    const double kA = 1.0 + 4.0 * st.dt * lam * ((u_.w[0] + u_.w[1]) + u_.w[2]);
    in.fullTable = steady || !(kA < 13.0);
    in.singular = st.singular;
    in.SAC = st.SAC;
    in.kappa = kap;
    in.unknown = unk;
    in.sax = CCConst(gm.sax);
    in.say = CCConst(gm.say);
    in.saz = CCConst(gm.saz);
    in.fac = &gm.fac;
    in.facetW = st.cw;
    for (int f = 0; f < 6; ++f)
      in.dirFace[f] = st.dirFace[f];
    if (steady && st.advecting) {
      // A2 (§6.7): the advective surrogate — the operator's bands (SAC the diagonal), the face
      // flux for the coarse advection, omega_open on the coarse mass. `st.advecting` is an
      // all-rank MAX, so every rank takes this path.
      in.advective = true;
      in.AW = sc.AW;
      in.AE = sc.AE;
      in.AS = sc.AS;
      in.AN = sc.AN;
      in.AB = sc.AB;
      in.AT = sc.AT;
      for (int a = 0; a < 3; ++a)
        in.phi[a] = CCConst(st.phi[a]);
      in.omegaOpen = CCConst(st.omegaOpen);
    } else if (st.advecting) {  // the lumped implicit outflow onto the coarse levels (§5.2)
      in.outflow = CCConst(st.outflow);
      in.hasOutflow = true;
    }
    st.mg->build(in);
    st.mgLevels = st.mg->levelsUsed();
  }

  // ---- reductions (+ MPI) ----
  auto allSum = [&](double* v, int k) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::vector<double> o((std::size_t)k);
      MPI_Allreduce(v, o.data(), k, MPI_DOUBLE, MPI_SUM, comm_);
      for (int j = 0; j < k; ++j)
        v[j] = o[(std::size_t)j];
    }
#endif
    (void)v;
    (void)k;
  };
  auto allMax = [&](double v) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double o = 0.0;
      MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_MAX, comm_);
      return o;
    }
#endif
    return v;
  };
  const double nUnk = (double)st.numUnknowns;
  auto removeMean = [&](CCField a) {
    double s[2];
    sco::sumUnknownLocal(CCConst(a), unk, e_, G, s[0], s[1]);
    allSum(s, 1);
    if (nUnk > 0.0)
      sco::addOnUnknown(a, unk, -s[0] / nUnk, e_, G);
  };
  // the right-hand side the Krylov sees: b, or (singular) its projection onto the range
  CCField bK = sc.b;
  st.incompatibility = 0.0;
  if (st.singular) {
    alloc(st.kb, "_cc_bproj");
    alloc(st.kq, "_cc_q");
    Kokkos::deep_copy(CCExec(), st.kb, sc.b);
    double s[2];
    sco::sumUnknownLocal(CCConst(st.kb), unk, e_, G, s[0], s[1]);
    allSum(s, 2);
    st.incompatibility = s[1] > 0.0 ? std::fabs(s[0]) / s[1] : 0.0;
    if (nUnk > 0.0)
      sco::addOnUnknown(st.kb, unk, -s[0] / nUnk, e_, G);
    bK = st.kb;
  }
  // gauge of the singular solve: sum kappa V c is kept at its pre-solve value (§5.1)
  double kc0 = 0.0, k0 = 0.0;
  if (st.singular) {
    double m[2];
    sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, m[0], m[1]);
    allSum(m, 2);
    kc0 = m[0];
    k0 = m[1];
  }

  ScalarKrylovOps ops;
  ops.e = e_;
  ops.g = G;
  ops.matvec = [&](const ScalarVec& y, const ScalarVec& x) { scalarCutMatvec(sc, y.f, x.f); };
  // One ScalarMG V-cycle on the surrogate (§5.2; level 0 alone under the transient level rule).
  ops.precond = [&](const ScalarVec& z, const ScalarVec& r) {
    CCField rr = r.f;
    if (st.singular) {
      Kokkos::deep_copy(CCExec(), st.kq, r.f);
      removeMean(st.kq);
      rr = st.kq;
    }
    st.mg->apply(z.f, rr);
    if (st.singular)
      removeMean(z.f);
  };
  ops.dot = [&](const ScalarVec& a, const ScalarVec& b) {
    double s = sco::dotLocal(CCConst(a.f), CCConst(b.f), e_, G);
    allSum(&s, 1);
    return s;
  };
  ops.dot2 = [&](const ScalarVec& a, const ScalarVec& b, const ScalarVec& c, double& ab,
                 double& cc) {
    double s[2];
    sco::dot2Local(CCConst(a.f), CCConst(b.f), CCConst(c.f), e_, G, s[0], s[1]);
    allSum(s, 2);
    ab = s[0];
    cc = s[1];
  };
  ops.maxabs = [&](const ScalarVec& a) { return allMax(sco::maxabsLocal(CCConst(a.f), e_, G)); };
  ops.removeMean = [&](const ScalarVec& a) { removeMean(a.f); };
  auto vec = [](CCField f) {
    ScalarVec v;
    v.f = f;
    return v;
  };
  const ScalarKrylovResult kr =
      scalarBiCGStab(ops, vec(bK), vec(sc.c), vec(st.kr), vec(st.krh), vec(st.kp), vec(st.kv),
                     vec(st.kt), vec(st.kz), vec(st.kz2), st.maxit, st.rtol, st.singular);
  if (st.singular && k0 > 0.0) {
    double m[2];
    sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, m[0], m[1]);
    allSum(m, 2);
    sco::addOnUnknown(sc.c, unk, (kc0 - m[0]) / k0, e_, G);
  }
  fillGhosts(sc.c);
  st.iterations = kr.iterations;
  st.residual = kr.ref > 0.0 ? kr.trueRes / kr.ref : 0.0;
  st.converged = kr.converged;
  if (!kr.converged && !st.warnedNoConv) {
    st.warnedNoConv = true;
    bool root = true;
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      int r = 0;
      MPI_Comm_rank(comm_, &r);
      root = r == 0;
    }
#endif
    if (root)
      std::fprintf(stderr,
                   "peclet.flow: cut-cell scalar '%s': the solve did not converge in %d "
                   "iterations (residual %.3e of the reference, tolerance %.1e); the iterate is "
                   "kept -- see diagnostics.scalar_census (printed once)\n",
                   sc.name.c_str(), kr.iterations, st.residual, st.rtol);
  }
}

template <class Grid>
std::vector<double> Solver<Grid>::scalarWallFlux(const std::string& name) {
  ScalarField& sc = cutcellScalar(name, "scalar_wall_flux");
  ScalarCutState& st = *sc.cut;
  const int nb = scalarNumBodies();
  std::vector<double> out((std::size_t)nb, 0.0);
  if (!st.built)
    return out;
  const auto& fo = scg_.fac;
  Kokkos::View<double*, CCMem> up("peclet::flow::sco_up", fo.n), qin("peclet::flow::sco_qin", fo.n);
  fillGhosts(sc.c);
  sco::facetFlux(up, qin, CCConst(sc.c), fo, st.cw, st.rw);
  auto hq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qin);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.body);
  for (long f = 0; f < fo.n; ++f) {  // facet order: deterministic
    int b = hb(f);
    if (b < 0 || b >= nb)
      b = 0;
    out[(std::size_t)b] += u_.vol * hq(f);
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    std::vector<double> o(out.size());
    MPI_Allreduce(out.data(), o.data(), nb, MPI_DOUBLE, MPI_SUM, comm_);
    out = o;
  }
#endif
  const double toPhys = u_.volToPhys() * u_.divToPhys();
  for (auto& v : out)
    v *= toPhys;
  return out;
}

template <class Grid>
typename Solver<Grid>::ScalarCutBudget Solver<Grid>::scalarBudget(const std::string& name) {
  ScalarField& sc = cutcellScalar(name, "scalar_budget");
  ScalarCutState& st = *sc.cut;
  ScalarCutBudget out;
  if (!st.built)
    return out;
  const scg::ScalarCutGeometry& gm = scg_;
  const CCConst unk(gm.unknown), kap(gm.kappa);
  const double V = u_.vol;
  // the residual of the stored system at the stored solution, r = b - A c, in flux form
  // (sco::residualFluxForm: the same r in exact arithmetic, round-off at the scale of the fluxes)
  fillGhosts(sc.c);
  sco::residualFluxForm(st.kt, CCConst(sc.c), CCConst(sc.b), kap, unk, CCConst(gm.sax),
                        CCConst(gm.say), CCConst(gm.saz), st.lam, st.idt, u_.w, e_, G);
  sco::overlayApply(st.kt, CCConst(sc.c), gm.fac, st.cw, /*subtract=*/true);
  if (st.advecting)  // the implicit faces' outflow (the explicit faces' sits in b)
    sco::advectionResidual(st.kt, CCConst(sc.c), CCConst(st.phi[0]), CCConst(st.phi[1]),
                           CCConst(st.phi[2]), CCConst(st.small), unk, st.steady, e_, G);
  if (st.advecting)  // WO-5b: the open faces' implicit outflow
    for (int f = 0; f < 6; ++f)
      if (st.openFace[f])
        sco::openFaceResidual(st.kt, CCConst(sc.c), CCConst(st.phi[f / 2]), CCConst(st.small), unk,
                              st.openInflow[f], f / 2, f % 2, st.steady, e_, G);
  {
    const CCField saAx[3] = {gm.sax, gm.say, gm.saz};
    for (int f = 0; f < 6; ++f)
      if (st.dirFace[f])
        sco::dirichletFaceResidual(st.kt, CCConst(sc.c), unk, CCConst(saAx[f / 2]),
                                   st.lam * u_.w[f / 2], f / 2, f % 2, e_, G);
  }
  double v[6] = {0, 0, 0, 0, 0, 0};  // sum r, kappa c, kappa c^n, wall, boundary, source
  {
    double sumAbs = 0.0, kc = 0.0, k = 0.0;
    sco::sumUnknownLocal(CCConst(st.kt), unk, e_, G, v[0], sumAbs);  // identity rows: r = 0
    sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, kc, k);
    v[1] = kc;
    sco::kappaMomentsLocal(CCConst(sc.cOld), kap, unk, e_, G, kc, k);
    v[2] = kc;
  }
  {
    const auto& fo = gm.fac;
    Kokkos::View<double*, CCMem> up("peclet::flow::sco_up", fo.n),
        qin("peclet::flow::sco_qin", fo.n);
    sco::facetFlux(up, qin, CCConst(sc.c), fo, st.cw, st.rw);
    auto hq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qin);
    double w = 0.0;
    for (long f = 0; f < fo.n; ++f)
      w += hq(f);
    v[3] = w;
  }
  {
    const CCField saAx[3] = {gm.sax, gm.say, gm.saz};
    double bsum = 0.0;
    for (int f = 0; f < 6; ++f)
      if (st.dirFace[f])
        bsum += sco::dirichletFaceInflux(CCConst(sc.c), unk, CCConst(saAx[f / 2]), st.gFace[f],
                                         st.lam * u_.w[f / 2], 1.0, f / 2, f % 2, e_, G);
    if (st.advecting)  // WO-5b: the advective flux through the open faces
      for (int f = 0; f < 6; ++f)
        if (st.openFace[f])
          bsum += sco::openFaceInflux(CCConst(sc.c), CCConst(sc.cOld), CCConst(st.phi[f / 2]),
                                      CCConst(st.small), unk, st.gFace[f], st.openInflow[f], 1.0,
                                      f / 2, f % 2, st.steady, e_, G);
    v[4] = bsum;
  }
  {
    const bool useField = st.sourceField.extent(0) == n_;
    v[5] = sco::sourceSumLocal(kap, unk, st.sourceConst, CCConst(useField ? st.sourceField : sc.c),
                               useField, st.srcFactor, e_, G);
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double o[6];
    MPI_Allreduce(v, o, 6, MPI_DOUBLE, MPI_SUM, comm_);
    for (int j = 0; j < 6; ++j)
      v[j] = o[j];
  }
#endif
  // internal: per unit V sums times V
  const double sumR = V * v[0], mass = V * v[1], massOld = V * v[2], wall = V * v[3],
               bnd = V * v[4], src = V * v[5];
  const double toMass = u_.volToPhys(), toRate = u_.volToPhys() * u_.divToPhys();
  out.mass = mass * toMass;
  out.wallIn = wall * toRate;
  out.boundaryIn = bnd * toRate;
  out.sourceIn = src * toRate;
  if (st.steady) {  // a rate balance: no storage term, the defect is sum V r per unit time
    out.dMass = 0.0;
    out.defect = sumR * toRate;
    out.identityError = (-(wall + bnd + src) + sumR) * toRate;
  } else {
    const double dm = mass - massOld;
    out.dMass = dm * toMass;
    out.defect = st.dt * sumR * toMass;
    out.identityError = (dm - st.dt * (wall + bnd + src) + st.dt * sumR) * toMass;
  }
  return out;
}

template <class Grid>
typename Solver<Grid>::ScalarCutFacets Solver<Grid>::scalarFacets(const std::string& name) {
  ScalarField& sc = cutcellScalar(name, "scalar_facets");
  ScalarCutState& st = *sc.cut;
  ensureScalarCutGeometry();
  const auto& fo = scg_.fac;
  ScalarCutFacets out;
  const long n = fo.n;
  out.centroid.assign((std::size_t)n * 3, 0.0);
  out.normal.assign((std::size_t)n * 3, 0.0);
  out.area.assign((std::size_t)n, 0.0);
  out.wallValue.assign((std::size_t)n, std::numeric_limits<double>::quiet_NaN());
  out.flux.assign((std::size_t)n, 0.0);
  out.instance.assign((std::size_t)n, 0);
  out.rung.assign((std::size_t)n, 0);
  auto hcell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.cell);
  auto halpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.alpha);
  auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.normal);
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.centroid);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.body);
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.sF);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.rungF);
  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scg_.unknown);
  Kokkos::View<double*, CCMem> up("peclet::flow::sco_up", n), qin("peclet::flow::sco_qin", n);
  const bool have = st.built && st.cw.extent(0) == (std::size_t)n;
  if (have) {
    fillGhosts(sc.c);
    sco::facetFlux(up, qin, CCConst(sc.c), fo, st.cw, st.rw);
  }
  auto hup = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), up);
  auto hq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qin);
  const int og[3] = {og_.x, og_.y, og_.z};
  const double l = u_.lenToPhys();
  const int nb = scalarNumBodies();
  for (long f = 0; f < n; ++f) {
    const long i = hcell(f);
    const long xyz[3] = {i % e_.x, (i / e_.x) % e_.y, i / ((long)e_.x * e_.y)};
    for (int d = 0; d < 3; ++d) {
      out.centroid[(std::size_t)(3 * f + d)] =
          u_.org[d] + ((double)(xyz[d] - G + og[d]) + 0.5) * u_.h[d] + hc(f, d) * l;
      out.normal[(std::size_t)(3 * f + d)] = hn(f, d);
    }
    out.area[(std::size_t)f] = halpha(f) * u_.vol * u_.areaToPhys();
    out.instance[(std::size_t)f] = hb(f);
    out.rung[(std::size_t)f] = (int)hr(f);
    if (!have || !(hu(i) > 0.5))
      continue;
    int b = hb(f);
    if (b < 0 || b >= nb)
      b = 0;
    const double a = halpha(f), s = hs(f), u = hup(f), lam = st.lam;
    const int t = st.wtType[(std::size_t)b];
    double cG;
    if (t == 1)
      cG = st.wtG[(std::size_t)b];
    else if (t == 2) {
      const double k = st.wtK[(std::size_t)b];
      cG = (lam + k * s) > 0.0 ? (lam * u + k * s * st.wtG[(std::size_t)b]) / (lam + k * s) : u;
    } else
      cG = lam > 0.0 ? u + st.wtQ[(std::size_t)b] * s / lam : u;
    out.wallValue[(std::size_t)f] = cG;
    out.flux[(std::size_t)f] = a > 0.0 ? hq(f) / a / u_.speedToInt() : 0.0;
  }
  return out;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_SCALARS_CUTCELL_HPP
