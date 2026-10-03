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
void Solver<Grid>::setScalarSolid(const std::string& name, double D, double capacity,
                                  double partition, double contactResistance, int instance) {
  ScalarField& sc = cutcellScalar(name, "set_scalar_solid");
  ScalarCutState& st = *sc.cut;
  if (!std::isfinite(D) || !(D >= 0.0))
    throw std::invalid_argument("set_scalar_solid: diffusivity must be finite and >= 0");
  if (!std::isfinite(capacity) || !(capacity > 0.0))
    throw std::invalid_argument("set_scalar_solid: capacity must be finite and > 0");
  if (!std::isfinite(partition) || !(partition > 0.0))
    throw std::invalid_argument("set_scalar_solid: partition must be finite and > 0");
  if (!std::isfinite(contactResistance) || !(contactResistance >= 0.0))
    throw std::invalid_argument("set_scalar_solid: contact_resistance must be finite and >= 0");
  if (instance >= 0 && !hasScene_)
    throw std::invalid_argument(
        "set_scalar_solid: instance= needs a scene (set_scene + set_solid_from_scene); raw-SDF "
        "geometry has one body, set with instance=None");
  if (instance >= 0 && instance >= nInst_)
    throw std::invalid_argument("set_scalar_solid: instance " + std::to_string(instance) +
                                " out of range (the scene has " + std::to_string(nInst_) + ")");
  ScalarWallSpec w;
  w.type = 3;
  w.solidD = D;
  w.solidC = capacity;
  w.solidK = partition;
  w.solidRc = contactResistance;
  if (st.solid.extent(0) != n_)  // §1.3: the solid field, registered, on the first call
    st.solid = addField(name + "_solid");
  if (instance < 0) {
    st.wallDefault = w;
    st.wallInstance.clear();
    return;
  }
  for (auto it = st.wallInstance.begin(); it != st.wallInstance.end(); ++it)
    if (it->first == instance) {
      st.wallInstance.erase(it);
      break;
    }
  st.wallInstance.emplace_back(instance, w);
}

template <class Grid>
sco::MaterialTable Solver<Grid>::scalarMaterialTable(ScalarField& sc, std::vector<char>& conj) {
  ScalarCutState& st = *sc.cut;
  const int nb = scalarNumBodies();
  std::vector<ScalarWallSpec> spec((std::size_t)nb, st.wallDefault);
  for (const auto& [inst, w] : st.wallInstance)
    if (inst >= 0 && inst < nb)
      spec[(std::size_t)inst] = w;
  conj.assign((std::size_t)nb, 0);
  st.mtLam.assign((std::size_t)nb, 0.0);
  st.mtCK.assign((std::size_t)nb, 1.0);
  st.mtK.assign((std::size_t)nb, 1.0);
  st.mtRc.assign((std::size_t)nb, 0.0);
  st.maxSolidD = 0.0;
  const double dI = u_.diffToInt(), rI = u_.resistToInt();
  for (int b = 0; b < nb; ++b) {
    const ScalarWallSpec& w = spec[(std::size_t)b];
    if (w.type != 3)
      continue;
    conj[(std::size_t)b] = 1;
    // psi form (§1.1): Lam_s = C_s D_s K, capacity C_s K; internal by §1.2 at use time
    st.mtLam[(std::size_t)b] = (w.solidC * w.solidD * w.solidK) * dI;
    st.mtCK[(std::size_t)b] = w.solidC * w.solidK;
    st.mtK[(std::size_t)b] = w.solidK;
    st.mtRc[(std::size_t)b] = w.solidRc * rI;
    st.maxSolidD = std::fmax(st.maxSolidD, w.solidD * dI);
  }
  sco::MaterialTable mt;
  mt.nb = nb;
  mt.conj = Kokkos::View<int*, CCMem>("peclet::flow::sco_mt_conj", nb);
  mt.lam = Kokkos::View<double*, CCMem>("peclet::flow::sco_mt_lam", nb);
  mt.ck = Kokkos::View<double*, CCMem>("peclet::flow::sco_mt_ck", nb);
  mt.kp = Kokkos::View<double*, CCMem>("peclet::flow::sco_mt_k", nb);
  mt.rc = Kokkos::View<double*, CCMem>("peclet::flow::sco_mt_rc", nb);
  auto hc = Kokkos::create_mirror_view(mt.conj);
  auto hl = Kokkos::create_mirror_view(mt.lam);
  auto hk = Kokkos::create_mirror_view(mt.ck);
  auto hp = Kokkos::create_mirror_view(mt.kp);
  auto hr = Kokkos::create_mirror_view(mt.rc);
  for (int b = 0; b < nb; ++b) {
    hc(b) = conj[(std::size_t)b];
    hl(b) = st.mtLam[(std::size_t)b];
    hk(b) = st.mtCK[(std::size_t)b];
    hp(b) = st.mtK[(std::size_t)b];
    hr(b) = st.mtRc[(std::size_t)b];
  }
  Kokkos::deep_copy(mt.conj, hc);
  Kokkos::deep_copy(mt.lam, hl);
  Kokkos::deep_copy(mt.ck, hk);
  Kokkos::deep_copy(mt.kp, hp);
  Kokkos::deep_copy(mt.rc, hr);
  return mt;
}

template <class Grid>
void Solver<Grid>::ensureScalarSolidRecord(ScalarField& sc, const std::vector<char>& conj) {
  ensureScalarCutGeometry();
  ScalarCutState& st = *sc.cut;
  if (st.solidVersion == scg_.version && st.solidKey == conj && st.mat.extent(0) == n_)
    return;
  const scg::ScalarCutGeometry& gm = scg_;
  const auto& fo = gm.fac;
  if (st.mat.extent(0) != n_) {
    st.mat = CCField(sc.name + "_cc_mat", n_);
    st.sunk = CCField(sc.name + "_cc_sunk", n_);
  }
  // the material table needs only the conjugate flags here (§2.7)
  sco::MaterialTable mt;
  const int nb = (int)conj.size();
  mt.nb = nb;
  mt.conj = Kokkos::View<int*, CCMem>("peclet::flow::sco_rec_conj", nb);
  {
    auto hc = Kokkos::create_mirror_view(mt.conj);
    for (int b = 0; b < nb; ++b)
      hc(b) = conj[(std::size_t)b];
    Kokkos::deep_copy(mt.conj, hc);
  }
  const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
  const bool hasOwner = hasScene_ && cutOwner_.extent(0) == nInner;
  double svol[2] = {0.0, 0.0};
  sco::buildSolidMaterial(st.mat, st.sunk, CCConst(gm.kappa), CCConst(gm.sax), CCConst(gm.say),
                          CCConst(gm.saz), fo, Kokkos::View<const int*, CCMem>(cutOwner_), hasOwner,
                          mt, e_, G, svol);
  fillGhosts(st.mat);
  fillGhosts(st.sunk);
  for (int face = 0; face < 6; ++face)
    if (bc_[face] != 0 && touchesGlobalFace(face))
      scg::zeroGhostSlab(st.sunk, e_, G, face);
  // the facets of conjugate bodies, and their solid probes (§3.4)
  st.conjF = Kokkos::View<std::uint8_t*, CCMem>(sc.name + "_cc_conjF", fo.n);
  st.sS = Kokkos::View<double*, CCMem>(sc.name + "_cc_sS", fo.n);
  st.offS = decltype(st.offS)(sc.name + "_cc_offS", fo.n);
  st.wS = decltype(st.wS)(sc.name + "_cc_wS", fo.n);
  st.rungS = Kokkos::View<std::uint8_t*, CCMem>(sc.name + "_cc_rungS", fo.n);
  {
    auto cf = st.conjF;
    auto fbody = fo.body;
    auto cj = mt.conj;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::sco_conj_facets", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
        KOKKOS_LAMBDA(const long f) {
          cf(f) = cj(sco::bodyIndex((double)fbody(f), nb)) != 0 ? 1 : 0;
        });
    space.fence();
  }
  sco::buildSolidProbes(st.sS, st.offS, st.wS, st.rungS,
                        Kokkos::View<const std::uint8_t*, CCMem>(st.conjF), fo, CCConst(sdf_),
                        CCConst(st.sunk), e_, u_.hp);
  // census: solid unknowns (inner) and the solid rungs (conjugate facets)
  long cnt[5] = {0, 0, 0, 0, 0};
  {
    CCConst su(st.sunk);
    const C3 e = e_;
    const int g = G;
    double nsu = 0.0;
    ccReduce3(
        "peclet::flow::sco_count_solid", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          acc += su((long)x + (long)y * e.x + (long)z * (long)e.x * e.y) > 0.5 ? 1.0 : 0.0;
        },
        nsu);
    cnt[0] = (long)nsu;
    auto rs = st.rungS;
    auto cf = st.conjF;
    for (int k = 0; k < 4; ++k) {
      long c = 0;
      Kokkos::parallel_reduce(
          "peclet::flow::sco_solid_rungs", Kokkos::RangePolicy<CCExec>(0, fo.n),
          KOKKOS_LAMBDA(const long f, long& acc) {
            acc += (cf(f) != 0 && (int)rs(f) == k) ? 1 : 0;
          },
          c);
      cnt[1 + k] = c;
    }
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    long o[5];
    MPI_Allreduce(cnt, o, 5, MPI_LONG, MPI_SUM, comm_);
    for (int k = 0; k < 5; ++k)
      cnt[k] = o[k];
    double ov[2];
    MPI_Allreduce(svol, ov, 2, MPI_DOUBLE, MPI_SUM, comm_);
    svol[0] = ov[0];
    svol[1] = ov[1];
  }
#endif
  const double lv = u_.lenToPhys();  // D-WO7-1: the census's sealed solid volume, physical
  st.solidVolume = svol[0] * u_.vol * lv * lv * lv;
  st.sealedSolidVolume = svol[1] * u_.vol * lv * lv * lv;
  st.numSolidUnknowns = cnt[0];
  for (int k = 0; k < 4; ++k)
    st.solidRungs[k] = cnt[1 + k];
  st.solidVersion = gm.version;
  st.solidKey = conj;
}

template <class Grid>
std::vector<double> Solver<Grid>::getScalarSolid(const std::string& name) {
  ScalarField& sc = cutcellScalar(name, "get_scalar_solid");
  ScalarCutState& st = *sc.cut;
  const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
  std::vector<double> out(nInner, std::numeric_limits<double>::quiet_NaN());
  if (!st.built || !st.conj || st.solid.extent(0) != n_ || st.sunk.extent(0) != n_)
    return out;
  const auto ps = gatherInner(st.solid);
  const auto su = gatherInner(st.sunk);
  const auto mt = gatherInner(st.mat);
  const int nb = (int)st.mtK.size();
  for (std::size_t i = 0; i < nInner; ++i) {
    if (!(su[i] > 0.5))
      continue;
    int b = (int)mt[i];
    if (b < 0 || b >= nb)
      b = 0;
    out[i] = st.mtK[(std::size_t)b] * ps[i];
  }
  return out;
}

template <class Grid>
std::vector<double> Solver<Grid>::scalarSolidGeometryField(const std::string& name, int which) {
  ScalarField& sc = cutcellScalar(name, "scalar_geometry");
  ScalarCutState& st = *sc.cut;
  std::vector<char> conj;
  (void)scalarMaterialTable(sc, conj);
  ensureScalarSolidRecord(sc, conj);
  if (which == 1)
    return gatherInner(st.sunk);
  std::vector<double> k = gatherInner(scg_.kappa);
  for (auto& v : k)
    v = 1.0 - v;
  return k;
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
void Solver<Grid>::setScalarMeanGradient(const std::string& name, const std::array<double, 3>& G) {
  ScalarCutState& st = *cutcellScalar(name, "set_scalar_mean_gradient").cut;
  for (double v : G)
    if (!std::isfinite(v))
      throw std::invalid_argument("set_scalar_mean_gradient: the gradient must be finite");
  for (int a = 0; a < 3; ++a)
    st.meanGrad[a] = G[(std::size_t)a];
}

template <class Grid>
sco::MeanGradPosition Solver<Grid>::scalarMeanGradPosition(const double Gphys[3]) const {
  sco::MeanGradPosition p;
  const int og[3] = {og_.x, og_.y, og_.z};
  for (int a = 0; a < 3; ++a) {
    p.G[a] = Gphys[a];
    p.org[a] = u_.org[a];
    p.h[a] = u_.h[a];
    p.base[a] = (long)og[a] - G;
  }
  p.sy = e_.x;
  p.sz = (long)e_.x * e_.y;
  return p;
}

template <class Grid>
std::array<double, 3> Solver<Grid>::scalarMeanFlux(const std::string& name) {
  ScalarField& sc = cutcellScalar(name, "scalar_mean_flux");
  ScalarCutState& st = *sc.cut;
  std::array<double, 3> out{0.0, 0.0, 0.0};
  if (!st.built)
    return out;
  const scg::ScalarCutGeometry& gm = scg_;
  fillGhosts(sc.c);
  // the face flux of the last solve (read only when it advected; any field otherwise)
  const bool adv = st.advecting;
  double d[3], v[3];
  sco::meanFluxLocal(CCConst(sc.c), CCConst(gm.kappa), CCConst(gm.unknown), CCConst(gm.sax),
                     CCConst(gm.say), CCConst(gm.saz), CCConst(adv ? st.phi[0] : sc.c),
                     CCConst(adv ? st.phi[1] : sc.c), CCConst(adv ? st.phi[2] : sc.c), adv, st.lam,
                     st.gInt, u_.hp, st.ubar, e_, G, d, v);
  if (st.conj) {  // WO-7: the solid faces' diffusive flux (§8.1: "over both phases")
    double ds[3];
    fillGhosts(st.solid);
    sco::solidMeanFluxLocal(CCConst(st.solid), CCConst(st.sunk), CCConst(st.spx), CCConst(st.spy),
                            CCConst(st.spz), st.gInt, u_.hp, e_, G, ds);
    for (int a = 0; a < 3; ++a)
      d[a] += ds[a];
  }
  double s[6] = {d[0], d[1], d[2], v[0], v[1], v[2]};
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double o[6];
    MPI_Allreduce(s, o, 6, MPI_DOUBLE, MPI_SUM, comm_);
    for (int j = 0; j < 6; ++j)
      s[j] = o[j];
  }
#endif
  // the box average: per unit full-cell volume sums over the GLOBAL cell count, then physical
  const auto gc = globalCells();
  const double nBox = (double)gc[0] * (double)gc[1] * (double)gc[2];
  const double toPhys = 1.0 / u_.speedToInt();
  for (int a = 0; a < 3; ++a)
    out[(std::size_t)a] = (s[a] + s[3 + a]) / nBox * toPhys;
  return out;
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
  // §1.5 / §8.1: theta is periodic along every axis the mean gradient has a component on
  for (int f = 0; f < 6; ++f)
    if (sc.cut->meanGrad[f / 2] != 0.0 && (bc_[f] != 0 || sc.bc[f] != 0))
      throw std::runtime_error(who + "the mean gradient has a component along " +
                               std::string(1, "xyz"[f / 2]) + " but the domain face " + kFace[f] +
                               " is not periodic (set_scalar_mean_gradient needs periodic faces "
                               "on every axis it has a component on)");
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
void Solver<Grid>::scalarCutMatvecTwo(ScalarField& sc, CCField yf, CCField ys, CCField xf,
                                      CCField xs) {
  ScalarCutState& st = *sc.cut;
  fillGhosts(xf);  // one G = 2 exchange per phase field (§4.1)
  fillGhosts(xs);
  applyCutcellOp(yf, CCConst(xf), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_, G);
  applyCutcellOp(ys, CCConst(xs), st.ASC, st.ASW, st.ASE, st.ASS, st.ASN, st.ASB, st.AST, e_, G);
  sco::conjOverlayApply(yf, ys, CCConst(xf), CCConst(xs), scg_.fac, st.cw, st.cc, st.offS, st.wS);
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
  // mean-gradient mode (§1.5): G' = G hRef; every G term below is skipped when G = 0
  st.meanGradOn = st.meanGrad[0] != 0.0 || st.meanGrad[1] != 0.0 || st.meanGrad[2] != 0.0;
  for (int a = 0; a < 3; ++a) {
    st.gPhys[a] = st.meanGrad[a];
    st.gInt[a] = st.meanGrad[a] * u_.lenToPhys();
    st.ubar[a] = 0.0;
  }
  const sco::MeanGradPosition mgPos = scalarMeanGradPosition(st.gPhys);
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
  if (st.meanGradOn)  // §1.5: rw = alpha G_phi (g - sum_k w_k G.x_k) on Dirichlet/Robin facets
    sco::meanGradientProbeShift(st.rw, gm.fac, st.cw, mgPos);
  // WO-7: a conjugate body makes this a two-field solve (§1.1, D8); without one the single-phase
  // path below runs unchanged
  st.conj = false;
  for (int b = 0; b < nb; ++b)
    st.conj = st.conj || st.wtType[(std::size_t)b] == 3;
  st.conj = st.conj && st.solid.extent(0) == n_;
  sco::MaterialTable mt;
  if (st.conj) {
    std::vector<char> conjB;
    mt = scalarMaterialTable(sc, conjB);
    ensureScalarSolidRecord(sc, conjB);
  }
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
    if (st.hasProfile[f] && (st.profileN[f][0] != n[t1] || st.profileN[f][1] != n[t2]))
      throw std::runtime_error("cut-cell scalar '" + sc.name +
                               "': the domain-face profile no longer matches this rank's block");
    if (st.hasProfile[f] || st.meanGradOn) {
      // §1.5: in mean-gradient mode the face holds theta_bf = g - G.x_bf (x_bf the boundary face
      // centre; G_a = 0 on this non-periodic axis by the refusals)
      const int og[3] = {og_.x, og_.y, og_.z};
      const double xa = u_.org[a] + (double)(og[a] + (side == 0 ? 0 : n[a])) * u_.h[a];
      auto hv = Kokkos::create_mirror_view(gv);
      for (int j2 = 0; j2 < n[t2]; ++j2)
        for (int j1 = 0; j1 < n[t1]; ++j1) {
          const long k = (long)j1 + (long)j2 * n[t1];
          double v = st.hasProfile[f] ? st.profile[f][(std::size_t)k] : sc.bcVal[f];
          if (st.meanGradOn) {
            double p[3];
            p[a] = xa;
            p[t1] = u_.org[t1] + ((double)(og[t1] + j1) + 0.5) * u_.h[t1];
            p[t2] = u_.org[t2] + ((double)(og[t2] + j2) + 0.5) * u_.h[t2];
            v -= (st.gPhys[0] * p[0] + st.gPhys[1] * p[1]) + st.gPhys[2] * p[2];
          }
          hv(k) = v;
        }
      Kokkos::deep_copy(gv, hv);
    } else {
      Kokkos::deep_copy(gv, sc.bcVal[f]);
    }
    st.gFace[f] = gv;
    st.dirFace[f] = true;
    anyDir = true;
    sco::dirichletFaceFold(sc.AC, sc.b, unk, CCConst(saAx[a]), gv, lam * u_.w[a], a, side, e_, G);
  }
  if (st.meanGradOn) {  // §1.5 faces: the G.x part of every coupled face term on the rhs
    double gf[3];
    for (int a = 0; a < 3; ++a)
      gf[a] = lam * u_.w[a] * u_.hp[a] * st.gInt[a];
    sco::meanGradientFaceRhs(sc.b, unk, CCConst(gm.sax), CCConst(gm.say), CCConst(gm.saz), gf, e_,
                             G);
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
    if (st.meanGradOn) {
      // §1.5 advection: theta is advected; the linear part enters exactly as the source
      // -G'.(U'_i/V - kappa_i U'bar), U'bar = sum U'_i / sum kappa_i V (the moving frame)
      const double hh[3] = {0.5 * u_.hp[0], 0.5 * u_.hp[1], 0.5 * u_.hp[2]};
      double m[4];
      sco::cellVelocitySumLocal(px, py, pz, unk, hh, e_, G, m);
      double kc = 0.0;
      sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, kc, m[3]);
#ifdef PECLET_FLOW_MPI
      if (distributed_) {
        double o[4];
        MPI_Allreduce(m, o, 4, MPI_DOUBLE, MPI_SUM, comm_);
        for (int j = 0; j < 4; ++j)
          m[j] = o[j];
      }
#endif
      for (int a = 0; a < 3; ++a)
        st.ubar[a] = m[3] > 0.0 ? m[a] / m[3] : 0.0;
      sco::meanGradientAdvectionRhs(sc.b, px, py, pz, kap, unk, st.gInt, hh, st.ubar, e_, G);
    }
  }
  if (st.conj) {
    // WO-7: the solid phase's operator (§1.4 solid row, §4.1): psi_s^n, the face products, the
    // mass C_s K kappa_s idt, the 7 bands, the rhs; the per-facet interface coupling cc; the
    // mean-gradient terms of §1.5 on the solid faces and the conjugate facets
    const CCConst sunk(st.sunk), matv(st.mat);
    alloc(st.spx, "_cc_spx");
    alloc(st.spy, "_cc_spy");
    alloc(st.spz, "_cc_spz");
    alloc(st.ms, "_cc_ms");
    alloc(st.ASC, "_cc_asc");
    alloc(st.ASW, "_cc_asw");
    alloc(st.ASE, "_cc_ase");
    alloc(st.ASS, "_cc_ass");
    alloc(st.ASN, "_cc_asn");
    alloc(st.ASB, "_cc_asb");
    alloc(st.AST, "_cc_ast");
    alloc(st.bS, "_cc_bs");
    alloc(st.sOld, "_cc_sold");
    alloc(st.SACs, "_cc_sacs");
    alloc(st.W0, "_cc_w0");
    if (st.cc.extent(0) != (std::size_t)gm.fac.n)
      st.cc = Kokkos::View<double*, CCMem>(sc.name + "_cc_cc", gm.fac.n);
    sco::zeroNonUnknown(st.solid, sunk, e_, G);
    Kokkos::deep_copy(CCExec(), st.sOld, st.solid);
    sco::solidFaceProducts(st.spx, st.spy, st.spz, sunk, matv, CCConst(gm.sax), CCConst(gm.say),
                           CCConst(gm.saz), mt, e_);
    sco::solidMass(st.ms, kap, sunk, matv, mt, idt, e_, G);
    sco::solidBands(st.ASC, st.ASW, st.ASE, st.ASS, st.ASN, st.ASB, st.AST, CCConst(st.ms), sunk,
                    CCConst(st.spx), CCConst(st.spy), CCConst(st.spz), u_.w, e_, G);
    sco::solidRhs(st.bS, CCConst(st.ms), CCConst(st.sOld), e_, G);
    sco::conjugateCoefficients(st.cc, gm.fac, Kokkos::View<const std::uint8_t*, CCMem>(st.conjF),
                               Kokkos::View<const double*, CCMem>(st.sS), unk, sunk, mt, lam);
    if (st.meanGradOn) {
      double gs[3];
      for (int a = 0; a < 3; ++a)
        gs[a] = u_.w[a] * u_.hp[a] * st.gInt[a];
      sco::solidMeanGradientFaceRhs(st.bS, sunk, CCConst(st.spx), CCConst(st.spy), CCConst(st.spz),
                                    gs, e_, G);
      sco::conjMeanGradientShift(sc.b, st.bS, gm.fac, st.cc, st.offS, st.wS, mgPos);
    }
    // the two-field level-0 surrogate (§4.3)
    sco::conjSurrogate(st.SAC, st.SACs, st.W0, CCConst(sc.AC), CCConst(st.ASC), gm.fac, st.cw,
                       st.cc);
  } else {
    // the level-0 surrogate diagonal (§4.3)
    sco::surrogateDiagonal(st.SAC, CCConst(sc.AC), gm.fac, st.cw);
  }
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
    // (WO-7: max(D', D_s') over the conjugate materials, D_s = Lam_s/(C_s K))
    const double kA = 1.0 + 4.0 * st.dt * (st.conj ? std::fmax(lam, st.maxSolidD) : lam) *
                                ((u_.w[0] + u_.w[1]) + u_.w[2]);
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
    if (st.conj) {  // WO-7: the coupled two-field levels (§5.2)
      in.twoPhase = true;
      in.SACs = st.SACs;
      in.W0 = st.W0;
      in.sunk = CCConst(st.sunk);
      in.spx = CCConst(st.spx);
      in.spy = CCConst(st.spy);
      in.spz = CCConst(st.spz);
      in.ms0 = CCConst(st.ms);
      in.facetC = st.cc;
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
  ScalarKrylovResult kr;
  if (!st.conj) {
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
    // (in mean-gradient mode the target is 0: sum kappa V theta = 0, §5.1)
    double kc0 = 0.0, k0 = 0.0;
    if (st.singular) {
      double m[2];
      sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, m[0], m[1]);
      allSum(m, 2);
      kc0 = st.meanGradOn ? 0.0 : m[0];
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
    kr = scalarBiCGStab(ops, vec(bK), vec(sc.c), vec(st.kr), vec(st.krh), vec(st.kp), vec(st.kv),
                        vec(st.kt), vec(st.kz), vec(st.kz2), st.maxit, st.rtol, st.singular);
    if (st.singular && k0 > 0.0) {
      double m[2];
      sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, m[0], m[1]);
      allSum(m, 2);
      sco::addOnUnknown(sc.c, unk, (kc0 - m[0]) / k0, e_, G);
    }

  } else {
    // WO-7: the two-field solve (§5.1 on both phases): the same BiCGStab, ScalarVec {c, psi_s};
    // dots and maxima over both fields; the singular projection and gauge over both phases'
    // unknowns (the null vector is constant in psi), the gauge weights kappa and C_s K kappa_s
    const CCConst sunk(st.sunk), matv(st.mat);
    alloc(st.krS, "_cc_rs");
    alloc(st.krhS, "_cc_rhs");
    alloc(st.kpS, "_cc_ps");
    alloc(st.kvS, "_cc_vs");
    alloc(st.ktS, "_cc_ts");
    alloc(st.kzS, "_cc_zs");
    alloc(st.kz2S, "_cc_z2s");
    const double nTot = (double)st.numUnknowns + (double)st.numSolidUnknowns;
    auto removeMeanTwo = [&](CCField a, CCField as) {
      double sf[2], ss[2];
      sco::sumUnknownLocal(CCConst(a), unk, e_, G, sf[0], sf[1]);
      sco::sumUnknownLocal(CCConst(as), sunk, e_, G, ss[0], ss[1]);
      double v[1] = {sf[0] + ss[0]};
      allSum(v, 1);
      if (nTot > 0.0) {
        sco::addOnUnknown(a, unk, -v[0] / nTot, e_, G);
        sco::addOnUnknown(as, sunk, -v[0] / nTot, e_, G);
      }
    };
    CCField bKf = sc.b, bKs = st.bS;
    st.incompatibility = 0.0;
    if (st.singular) {
      alloc(st.kb, "_cc_bproj");
      alloc(st.kq, "_cc_q");
      alloc(st.kbS, "_cc_bprojs");
      alloc(st.kqS, "_cc_qs");
      Kokkos::deep_copy(CCExec(), st.kb, sc.b);
      Kokkos::deep_copy(CCExec(), st.kbS, st.bS);
      double sf[2], ss[2];
      sco::sumUnknownLocal(CCConst(st.kb), unk, e_, G, sf[0], sf[1]);
      sco::sumUnknownLocal(CCConst(st.kbS), sunk, e_, G, ss[0], ss[1]);
      double v[2] = {sf[0] + ss[0], sf[1] + ss[1]};
      allSum(v, 2);
      st.incompatibility = v[1] > 0.0 ? std::fabs(v[0]) / v[1] : 0.0;
      if (nTot > 0.0) {
        sco::addOnUnknown(st.kb, unk, -v[0] / nTot, e_, G);
        sco::addOnUnknown(st.kbS, sunk, -v[0] / nTot, e_, G);
      }
      bKf = st.kb;
      bKs = st.kbS;
    }
    // gauge (§5.1): sum kappa V c + sum C_s K kappa_s V psi_s kept (0 in mean-gradient mode)
    auto moments = [&](double m[2]) {
      double a[2], b2[2];
      sco::kappaMomentsLocal(CCConst(sc.c), kap, unk, e_, G, a[0], a[1]);
      sco::solidMomentsLocal(CCConst(st.solid), kap, sunk, matv, mt, e_, G, b2[0], b2[1]);
      m[0] = a[0] + b2[0];
      m[1] = a[1] + b2[1];
      allSum(m, 2);
    };
    double kc0 = 0.0, k0 = 0.0;
    if (st.singular) {
      double m[2];
      moments(m);
      kc0 = st.meanGradOn ? 0.0 : m[0];
      k0 = m[1];
    }
    ScalarKrylovOps ops;
    ops.e = e_;
    ops.g = G;
    ops.matvec = [&](const ScalarVec& y, const ScalarVec& x) {
      scalarCutMatvecTwo(sc, y.f, y.s, x.f, x.s);
    };
    ops.precond = [&](const ScalarVec& z, const ScalarVec& r) {
      CCField rf = r.f, rs = r.s;
      if (st.singular) {
        Kokkos::deep_copy(CCExec(), st.kq, r.f);
        Kokkos::deep_copy(CCExec(), st.kqS, r.s);
        removeMeanTwo(st.kq, st.kqS);
        rf = st.kq;
        rs = st.kqS;
      }
      st.mg->applyTwo(z.f, z.s, rf, rs);
      if (st.singular)
        removeMeanTwo(z.f, z.s);
    };
    ops.dot = [&](const ScalarVec& a, const ScalarVec& b) {
      double v = sco::dotTwoLocal(CCConst(a.f), CCConst(b.f), CCConst(a.s), CCConst(b.s), e_, G);
      allSum(&v, 1);
      return v;
    };
    ops.dot2 = [&](const ScalarVec& a, const ScalarVec& b, const ScalarVec& c, double& ab,
                   double& cc) {
      double v[2];
      sco::dot2TwoLocal(CCConst(a.f), CCConst(b.f), CCConst(c.f), CCConst(a.s), CCConst(b.s),
                        CCConst(c.s), e_, G, v[0], v[1]);
      allSum(v, 2);
      ab = v[0];
      cc = v[1];
    };
    ops.maxabs = [&](const ScalarVec& a) {
      return allMax(
          std::fmax(sco::maxabsLocal(CCConst(a.f), e_, G), sco::maxabsLocal(CCConst(a.s), e_, G)));
    };
    ops.removeMean = [&](const ScalarVec& a) { removeMeanTwo(a.f, a.s); };
    auto vec2 = [](CCField f, CCField s2) {
      ScalarVec v;
      v.f = f;
      v.s = s2;
      v.solid = true;
      return v;
    };
    kr = scalarBiCGStab(ops, vec2(bKf, bKs), vec2(sc.c, st.solid), vec2(st.kr, st.krS),
                        vec2(st.krh, st.krhS), vec2(st.kp, st.kpS), vec2(st.kv, st.kvS),
                        vec2(st.kt, st.ktS), vec2(st.kz, st.kzS), vec2(st.kz2, st.kz2S), st.maxit,
                        st.rtol, st.singular);
    if (st.singular && k0 > 0.0) {
      double m[2];
      moments(m);
      const double d = (kc0 - m[0]) / k0;
      sco::addOnUnknown(sc.c, unk, d, e_, G);
      sco::addOnUnknown(st.solid, sunk, d, e_, G);
    }
    fillGhosts(st.solid);
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
  if (st.conj) {  // WO-7: a conjugate body's flux into the fluid is its interface flux cc (u_ps -
                  // u_pf)
    Kokkos::View<double*, CCMem> us("peclet::flow::sco_us", fo.n);
    fillGhosts(st.solid);
    sco::conjFacetFlux(up, us, qin, CCConst(sc.c), CCConst(st.solid), fo, st.cw, st.rw, st.cc,
                       st.offS, st.wS, scalarMeanGradPosition(st.gPhys), st.meanGradOn);
  } else {
    sco::facetFlux(up, qin, CCConst(sc.c), fo, st.cw, st.rw);
  }
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
  if (st.conj) {  // WO-7: the solid rows (flux form) and the two-field overlay, both subtracted
    fillGhosts(st.solid);
    sco::solidResidualFluxForm(st.ktS, CCConst(st.solid), CCConst(st.bS), CCConst(st.ms),
                               CCConst(st.sunk), CCConst(st.spx), CCConst(st.spy), CCConst(st.spz),
                               u_.w, e_, G);
    sco::conjOverlayApply(st.kt, st.ktS, CCConst(sc.c), CCConst(st.solid), gm.fac, st.cw, st.cc,
                          st.offS, st.wS, /*subtract=*/true);
  } else {
    sco::overlayApply(st.kt, CCConst(sc.c), gm.fac, st.cw, /*subtract=*/true);
  }
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
  double sumR = V * v[0];
  const double mass = V * v[1], massOld = V * v[2], wall = V * v[3], bnd = V * v[4], src = V * v[5];
  // WO-7: the solid rows' residual and the solid mass sum C_s K kappa_s V psi_s (now and at the
  // time base). The conjugate facets' flux is internal (it enters both rows), so `wall_in` keeps
  // the wall facets alone and `d_mass` is the change of the TOTAL (fluid + solid) mass.
  double massS = 0.0, massSOld = 0.0;
  if (st.conj) {
    sco::MaterialTable mt;
    std::vector<char> conjB;
    mt = scalarMaterialTable(sc, conjB);
    double vs[3] = {0.0, 0.0, 0.0}, sAbs = 0.0, w = 0.0;
    sco::sumUnknownLocal(CCConst(st.ktS), CCConst(st.sunk), e_, G, vs[0], sAbs);
    sco::solidMomentsLocal(CCConst(st.solid), kap, CCConst(st.sunk), CCConst(st.mat), mt, e_, G,
                           vs[1], w);
    sco::solidMomentsLocal(CCConst(st.sOld), kap, CCConst(st.sunk), CCConst(st.mat), mt, e_, G,
                           vs[2], w);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double o[3];
      MPI_Allreduce(vs, o, 3, MPI_DOUBLE, MPI_SUM, comm_);
      for (int j = 0; j < 3; ++j)
        vs[j] = o[j];
    }
#endif
    sumR = V * (v[0] + vs[0]);
    massS = V * vs[1];
    massSOld = V * vs[2];
  }
  const double toMass = u_.volToPhys(), toRate = u_.volToPhys() * u_.divToPhys();
  out.mass = mass * toMass;
  out.massSolid = massS * toMass;
  out.wallIn = wall * toRate;
  out.boundaryIn = bnd * toRate;
  out.sourceIn = src * toRate;
  if (st.steady) {  // a rate balance: no storage term, the defect is sum V r per unit time
    out.dMass = 0.0;
    out.defect = sumR * toRate;
    out.identityError = (-(wall + bnd + src) + sumR) * toRate;
  } else {
    const double dm = st.conj ? (mass + massS) - (massOld + massSOld) : mass - massOld;
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
  Kokkos::View<double*, CCMem> usv("peclet::flow::sco_us", n);
  const bool have = st.built && st.cw.extent(0) == (std::size_t)n;
  const bool conj = have && st.conj && st.cc.extent(0) == (std::size_t)n;
  if (have) {
    fillGhosts(sc.c);
    if (conj) {
      fillGhosts(st.solid);
      sco::conjFacetFlux(up, usv, qin, CCConst(sc.c), CCConst(st.solid), fo, st.cw, st.rw, st.cc,
                         st.offS, st.wS, scalarMeanGradPosition(st.gPhys), st.meanGradOn);
    } else {
      sco::facetFlux(up, qin, CCConst(sc.c), fo, st.cw, st.rw);
    }
  }
  auto hup = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), up);
  auto hq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), qin);
  // mean-gradient mode (§1.5): the probe value of c is that of theta plus sum_k w_k G.x_k
  const bool mgOn = have && st.meanGradOn;
  const sco::MeanGradPosition mgPos = scalarMeanGradPosition(st.gPhys);
  auto hoff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.offF);
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.wF);
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
    double u = hup(f);
    if (mgOn)
      for (int k = 0; k < 8; ++k)
        u += hw(f, k) * mgPos(i + (long)hoff(f, k));
    const double a = halpha(f), s = hs(f), lam = st.lam;
    const int t = st.wtType[(std::size_t)b];
    double cG;
    if (t == 3) {  // WO-7, conjugate: psi_Gamma,f = u_pf + q s_f / Lam_f, q = flux into the fluid
      const double q = a > 0.0 ? hq(f) / a : 0.0;
      cG = lam > 0.0 ? u + q * s / lam : u;
    } else if (t == 1)
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
