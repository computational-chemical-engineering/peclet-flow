/// @file
/// @brief flow — IbmSolver cut-cell scalar transport (doc/scalar_ibm_design.md): the geometry
/// record of WO-2 — the lazy build `ensureScalarCutGeometry` (§2.5), its invalidation and the
/// diagnostics getters. The block kernels are `scalar_cutcell_geometry.hpp`; the per-cell geometry
/// and the probe ladder are core's `peclet::core::scheme` (WO-1).
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

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_SCALARS_CUTCELL_HPP
