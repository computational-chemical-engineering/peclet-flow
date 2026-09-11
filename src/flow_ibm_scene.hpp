/// @file
/// @brief flow — IbmSolver scene and moving geometry: scene instances and motion upload, wall
/// velocity, wall-flux divergence, fresh-cell seeding.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_SCENE_HPP
#define PECLET_FLOW_FLOW_IBM_SCENE_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::setScene(const std::vector<int>& nodeInts, const std::vector<double>& nodeReals,
                            const std::vector<int>& instInts, const std::vector<double>& instReals,
                            bool periodic) {
  namespace g = peclet::core::geom;
  g::SceneBuilder<double> b =
      g::SceneBuilder<double>::decode(nodeInts, nodeReals, instInts, instReals, /*grids=*/{},
                                      /*pool=*/{});
  if (b.instances().empty())
    throw std::runtime_error("set_scene: at least one instance required");
  for (const auto& nd : b.nodes())
    if (nd.kind == g::kGrid)
      throw std::runtime_error("set_scene: grid leaves are not supported (analytic scenes only)");
  // Global inner-grid extents: the distributed build carries them (gnx_); single-rank (or a
  // non-MPI build) the local block IS the global grid.
  double GX = nx_, GY = ny_, GZ = nz_;
#ifdef PECLET_FLOW_MPI
  if (gnx_ > 0) {
    GX = gnx_;
    GY = gny_;
    GZ = gnz_;
  }
#endif
  // The scene is in the caller's PHYSICAL coordinates when a domain is armed (so one scene
  // serves flow, dem and voro unchanged), and in the historical cell coordinates otherwise.
  peclet::core::Vec3<double> so{0, 0, 0}, se{GX, GY, GZ};
  if (u_.physical) {
    if (u_.cells[0] != (long)GX || u_.cells[1] != (long)GY || u_.cells[2] != (long)GZ)
      throw std::runtime_error(
          "set_scene: the physical domain was armed for a different global grid than the solver "
          "has (pass global_cells = the grid init_mpi gets)");
    so = peclet::core::Vec3<double>{u_.org[0], u_.org[1], u_.org[2]};
    se = peclet::core::Vec3<double>{u_.ext[0], u_.ext[1], u_.ext[2]};
  }
  g::PeriodicBox<double> box{se.x, se.y, se.z, periodic};
  sceneB_ = std::make_shared<g::SceneBuilder<double>>(std::move(b));
  sceneOrigin_ = so;
  sceneExtent_ = se;
  scenePeriodic_ = periodic;
  buildSceneQuery();
  hasScene_ = true;
  // Moving-geometry state travels WITH the scene: core's Instance already carries linVel/angVel/
  // center, so a caller can encode motion directly and it arrives here. CENTRE OF ROTATION: the
  // encoded `center` when it is nonzero, otherwise the instance's own translation -- which is
  // what a caller who only placed the body means by "spin it". Set `center` explicitly (or pass
  // it to set_instance_motion) to spin about some other point.
  nInst_ = static_cast<int>(sceneB_->instances().size());
  instCen_.assign((std::size_t)nInst_ * 3, 0.0);
  instLin_.assign((std::size_t)nInst_ * 3, 0.0);
  instAng_.assign((std::size_t)nInst_ * 3, 0.0);
  instCenPinned_.assign((std::size_t)nInst_, 0);
  for (int i = 0; i < nInst_; ++i) {
    auto& in = sceneB_->instanceRef(i);
    // CENTRE OF ROTATION (§7 item 3, resolved 2026-09-02): NaN = follows the body (the
    // builder's default); any other finite point = PINNED there, the world origin included.
    // An all-zero centre from a RAW instance array is the legacy 'follows the body' (every
    // producer wrote zeros before NaN existed); a world-origin pin from a raw array goes
    // through set_instance_motion(center=...). The decision is an explicit per-instance flag
    // from here on, never re-inferred from the numbers.
    // The decoder already resolved the record: an 18-real record carries the flag, a legacy
    // 17-real record pins only a finite non-zero centre.
    const bool pinned = in.centerPinned && std::isfinite(in.center.x) &&
                        std::isfinite(in.center.y) && std::isfinite(in.center.z);
    instCenPinned_[(std::size_t)i] = pinned ? 1 : 0;
    const auto c = pinned ? in.center : in.transform.translation;
    in.center = c;  // the resolved centre, so no consumer ever reads the NaN sentinel
    instCen_[3 * (std::size_t)i + 0] = c.x;
    instCen_[3 * (std::size_t)i + 1] = c.y;
    instCen_[3 * (std::size_t)i + 2] = c.z;
    instLin_[3 * (std::size_t)i + 0] = in.linVel.x;
    instLin_[3 * (std::size_t)i + 1] = in.linVel.y;
    instLin_[3 * (std::size_t)i + 2] = in.linVel.z;
    instAng_[3 * (std::size_t)i + 0] = in.angVel.x;
    instAng_[3 * (std::size_t)i + 1] = in.angVel.y;
    instAng_[3 * (std::size_t)i + 2] = in.angVel.z;
  }
  refreshMotionFlag();
  uploadMotion();
}

template <class Grid>
std::array<double, 3> Solver<Grid>::instanceCenter(int i) const {
  if (i < 0 || i >= nInst_)
    throw std::runtime_error("instance_center: instance index out of range");
  return {instCen_[3 * (std::size_t)i], instCen_[3 * (std::size_t)i + 1],
          instCen_[3 * (std::size_t)i + 2]};
}

template <class Grid>
bool Solver<Grid>::instanceCenterPinned(int i) const {
  if (i < 0 || i >= nInst_)
    throw std::runtime_error("instance_center_pinned: instance index out of range");
  return instCenPinned_[(std::size_t)i] != 0;
}

template <class Grid>
void Solver<Grid>::setInstanceMotion(int i, const std::array<double, 3>& lin,
                                     const std::array<double, 3>& ang, const double* center) {
  if (!hasScene_)
    throw std::runtime_error("set_instance_motion: call set_scene first");
  if (i < 0 || i >= nInst_)
    throw std::runtime_error("set_instance_motion: instance index out of range");
  auto& in = sceneB_->instanceRef(i);
  for (int k = 0; k < 3; ++k) {
    instLin_[3 * (std::size_t)i + k] = lin[k];
    instAng_[3 * (std::size_t)i + k] = ang[k];
    if (center)
      instCen_[3 * (std::size_t)i + k] = center[k];
  }
  if (center)
    instCenPinned_[(std::size_t)i] = 1;  // an explicit centre is pinned, whatever its value
  in.linVel = peclet::core::Vec3<double>{lin[0], lin[1], lin[2]};
  in.angVel = peclet::core::Vec3<double>{ang[0], ang[1], ang[2]};
  in.center =
      peclet::core::Vec3<double>{instCen_[3 * (std::size_t)i + 0], instCen_[3 * (std::size_t)i + 1],
                                 instCen_[3 * (std::size_t)i + 2]};
  refreshMotionFlag();
  uploadMotion();
}

template <class Grid>
void Solver<Grid>::setInstanceTransform(int i, const std::array<double, 3>& translation,
                                        const std::array<double, 4>& quat) {
  if (!hasScene_)
    throw std::runtime_error("set_instance_transform: call set_scene first");
  if (i < 0 || i >= nInst_)
    throw std::runtime_error("set_instance_transform: instance index out of range");
  auto& in = sceneB_->instanceRef(i);
  // Follows-the-body vs pinned is the explicit flag, not a coincidence of the numbers (the old
  // float comparison turned a pinned centre into a tracked one whenever the body passed
  // through it).
  const bool centreTracked = instCenPinned_[(std::size_t)i] == 0;
  in.transform.translation =
      peclet::core::Vec3<double>{translation[0], translation[1], translation[2]};
  in.transform.rotation = peclet::core::Quat<double>{quat[0], quat[1], quat[2], quat[3]};
  if (centreTracked) {
    for (int k = 0; k < 3; ++k)
      instCen_[3 * (std::size_t)i + k] = translation[k];
    in.center = in.transform.translation;
  }
  sceneDirty_ = true;
}

template <class Grid>
void Solver<Grid>::rebuildGeometry() {
  if (!hasScene_)
    throw std::runtime_error("rebuild_geometry: call set_scene first");
  CCField uSave[3], mSave[3];
  for (int c = 0; c < 3; ++c) {
    uSave[c] = CCField("uSave", n_);
    Kokkos::deep_copy(uSave[c], C[c].u);
    if (freshSeed_) {  // remember which points were SOLID, to find the ones the body uncovers
      mSave[c] = CCField("mSave", n_);
      Kokkos::deep_copy(mSave[c], C[c].mask);
    }
  }
  CCField pSave("pSave", n_);
  Kokkos::deep_copy(pSave, P_);
  if (sceneDirty_) {
    buildSceneQuery();
    sceneDirty_ = false;
  }
  // Crossings BEFORE the solid: the overlay build consumes tEx_, so deriving them first means
  // ONE geometry rebuild per step rather than two.
  if (sceneCrossings_)
    setExactCrossingsFromScene();
  setSolidFromScene(cutcellPressure_);
  for (int c = 0; c < 3; ++c)
    Kokkos::deep_copy(C[c].u, uSave[c]);
  Kokkos::deep_copy(P_, pSave);
  if (freshSeed_)
    seedFreshCells(mSave);
}

template <class Grid>
void Solver<Grid>::seedFreshCells(CCField mOld[3]) {
  if (!hasMotion_)
    return;
  CCExec space;
  const C3 e = e_;
  for (int c = 0; c < 3; ++c) {
    if (mOld[c].extent(0) != n_ || uBc_[c].extent(0) != n_)
      continue;
    CCField u = C[c].u;
    CCConst mo = CCConst(mOld[c]), mn = CCConst(C[c].mask), w = CCConst(uBc_[c]);
    Kokkos::parallel_for(
        "peclet::flow::seed_fresh", Kokkos::RangePolicy<CCExec>(space, 0, (long)n_),
        KOKKOS_LAMBDA(long i) {
          if (mo(i) > 0.5 && mn(i) <= 0.5)
            u(i) = w(i);
        });
  }
  space.fence();
}

template <class Grid>
void Solver<Grid>::setFreshCellSeed(bool on) {
  freshSeed_ = on;
}

template <class Grid>
bool Solver<Grid>::freshCellSeed() const {
  return freshSeed_;
}

template <class Grid>
void Solver<Grid>::refreshWallVelocity() {
  if (!hasScene_)
    throw std::runtime_error("refresh_wall_velocity: call set_scene first");
  if (sceneDirty_)
    throw std::runtime_error(
        "refresh_wall_velocity: an instance TRANSFORM changed since the last geometry build -- "
        "this call only refreshes the wall VELOCITY, so the run would continue on stale "
        "geometry. Call rebuild_geometry() instead.");
  buildWallVelocity();
  rebuildStencils();
}

template <class Grid>
bool Solver<Grid>::hasMovingInstance() const {
  return hasMotion_;
}

template <class Grid>
int Solver<Grid>::sceneInstanceCount() const {
  return nInst_;
}

template <class Grid>
void Solver<Grid>::setWallFluxDivergence(bool on) {
  wallFluxDiv_ = on;
}

template <class Grid>
bool Solver<Grid>::wallFluxDivergence() const {
  return wallFluxDiv_;
}

template <class Grid>
bool Solver<Grid>::hasScene() const {
  return hasScene_;
}

template <class Grid>
std::vector<int> Solver<Grid>::getCutOwner() const {
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  std::vector<int> out(n, -1);
  if (cutOwner_.extent(0) != n)
    return out;
  using HostV = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(HostV(out.data(), n), cutOwner_);
  return out;
}

template <class Grid>
void Solver<Grid>::setSolidFromScene(bool cutcellPressure) {
  if (!hasScene_)
    throw std::runtime_error("set_solid_from_scene: call set_scene first");
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  CCField din("peclet::flow::sceneSdf", n);
  if (cutOwner_.extent(0) != n)
    cutOwner_ = Kokkos::View<int*, CCMem>("peclet::flow::cutOwner", n);
  auto own = cutOwner_;
  const auto q = sceneQ_->view();
  const int nx = nx_, ny = ny_, nz = nz_;
  const C3 og = og_;
  const SceneMap sm = sceneMap();
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::scene_sample",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * (double)(x + og.x),
                                           sm.a[1] + sm.b[1] * (double)(y + og.y),
                                           sm.a[2] + sm.b[2] * (double)(z + og.z)};
        // evalOwner is ONE traversal returning bitwise eval's value plus the argmin instance, so
        // carrying the ownership field costs nothing over the sample it rides on.
        int oi = -1;
        const std::size_t idx =
            (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
        din(idx) = q.evalOwner(p, oi) * sm.dToInt;
        own(idx) = oi;
      });
  space.fence();
  // PERIODIC IMAGES ARE A UNION. The query takes the minimum over an instance's 27 neighbour
  // images -- right for a body straddling a periodic face, and a silent TRAP for a leaf WIDER
  // than the box: its images overlap, and a cavity carved from it (a container wall built as
  // slab-minus-cavity) is refilled wherever a neighbouring image's slab covers it. Measured: a
  // 0.7 L slab minus a 53-cell cavity gave a 38-cell duct; the ten Cate tank ran 30 % narrow
  // through two campaigns and read as "creeping-valued confinement". Detect it EXACTLY: when
  // some instance's bounding sphere spans more than the box on a periodic axis, sample the
  // primary image alone and count the cells whose solid/fluid sign the images changed.
  imageOverlapCells_ = 0;
  if (scenePeriodic_) {
    namespace g = peclet::core::geom;
    const auto hv = sceneB_->view();
    const double lmin = std::fmin(sceneExtent_.x, std::fmin(sceneExtent_.y, sceneExtent_.z));
    bool wide = false;
    for (int i = 0; i < hv.instanceCount; ++i)
      if (2.0 * g::instanceBound(hv, i).r > lmin)
        wide = true;
    if (wide) {
      g::PeriodicBox<double> nobox{sceneExtent_.x, sceneExtent_.y, sceneExtent_.z, false};
      auto qnp = g::SceneQueryDevice<double, CCMem>::build(*sceneB_, sceneOrigin_, sceneExtent_,
                                                           nobox, /*accelerate=*/false);
      const auto qv = qnp.view();
      long cnt = 0;
      Kokkos::parallel_reduce(
          "peclet::flow::scene_image_overlap",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
          KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
            const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * (double)(x + og.x),
                                               sm.a[1] + sm.b[1] * (double)(y + og.y),
                                               sm.a[2] + sm.b[2] * (double)(z + og.z)};
            int oi = -1;
            const std::size_t idx =
                (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
            const double dp = qv.evalOwner(p, oi);
            if ((dp < 0.0) != (din(idx) < 0.0))
              ++acc;
          },
          cnt);
      space.fence();
      imageOverlapCells_ = cnt;
      if (cnt > 0)
        std::fprintf(stderr,
                     "peclet.flow set_solid_from_scene WARNING: an instance is wider than the "
                     "periodic box, and the UNION of its periodic images changes the solid at "
                     "%ld cells on this rank. If it is a container wall (slab minus cavity), "
                     "keep the slab's half-extent at half the box plus the wall thickness -- "
                     "not more -- or the images refill the cavity. "
                     "periodic_image_overlap_cells() returns this count.\n",
                     cnt);
    }
  }
  setSolidDevice(din, cutcellPressure);
  if (hasMotion_)
    checkMovingInstancesAreCut();
}

template <class Grid>
void Solver<Grid>::checkMovingInstancesAreCut() {
  const int nI = nInst_;
  if (nI <= 0 || cutOwner_.extent(0) != (std::size_t)nx_ * ny_ * nz_)
    return;
  // The datum enters the momentum operator only through the IBM overlay's CUT ROWS (staggered
  // points whose stencil crosses the wall, `idMap >= 0`), not through face apertures: a plane
  // cutting faces parallel to itself leaves every aperture 0 or 1 and still has cut rows.
  Kokkos::View<long*, CCMem> cnt("peclet::flow::movingCut", nI);  // zero-initialised
  auto own = cutOwner_;
  const C3 e = e_;
  const int nx = nx_, ny = ny_, nz = nz_;
  CCExec space;
  for (int c = 0; c < 3; ++c) {
    if (C[c].idMap.extent(0) != n_)
      continue;
    Kokkos::View<const int*, CCMem> idm = C[c].idMap;
    Kokkos::parallel_for(
        "peclet::flow::moving_cut_count",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long ie = (long)(x + G) + (long)(y + G) * e.x + (long)(z + G) * (long)e.x * e.y;
          if (idm(ie) < 0)
            return;
          const int o =
              own((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
          if (o >= 0 && o < nI)
            Kokkos::atomic_add(&cnt(o), (long)1);
        });
  }
  // DEGENERATE points: a staggered point where the sampled sdf is EXACTLY zero is fluid to the
  // mask (strict < 0) and not a ghost to its neighbours' folds (strict < 0 too), so a wall face
  // on a lattice plane never folds its datum -- the body is inert. Count them per owner.
  Kokkos::View<long*, CCMem> deg("peclet::flow::movingDegenerate", nI);
  {
    CCConst sd = CCConst(sdf_);
    for (int c = 0; c < 3; ++c) {
      const auto po = Grid::offset(c);
      const double ox = po.x, oy = po.y, oz = po.z;
      Kokkos::parallel_for(
          "peclet::flow::moving_degenerate_count",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const double sv = ccSampleExt(sd, e, x + G + ox, y + G + oy, z + G + oz);
            if (sv != 0.0)
              return;
            const int o =
                own((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
            if (o >= 0 && o < nI)
              Kokkos::atomic_add(&deg(o), (long)1);
          });
    }
  }
  space.fence();
  auto h = Kokkos::create_mirror_view(cnt);
  Kokkos::deep_copy(h, cnt);
  auto hd = Kokkos::create_mirror_view(deg);
  Kokkos::deep_copy(hd, deg);
  movingCutCells_.assign((std::size_t)nI, 0);
  movingDegenerate_.assign((std::size_t)nI, 0);
  for (int i = 0; i < nI; ++i) {
    movingCutCells_[(std::size_t)i] = h(i);
    movingDegenerate_[(std::size_t)i] = hd(i);
  }
#ifdef PECLET_FLOW_MPI
  // A body lives on SOME rank: the count is global, and only rank 0 speaks.
  if (distributed_) {
    std::vector<long> g((std::size_t)nI, 0);
    MPI_Allreduce(movingCutCells_.data(), g.data(), nI, MPI_LONG, MPI_SUM, comm_);
    movingCutCells_ = g;
    MPI_Allreduce(movingDegenerate_.data(), g.data(), nI, MPI_LONG, MPI_SUM, comm_);
    movingDegenerate_ = g;
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank != 0)
      return;
  }
#endif
  for (int i = 0; i < nI; ++i) {
    const bool moves =
        instLin_[3 * (std::size_t)i] != 0.0 || instLin_[3 * (std::size_t)i + 1] != 0.0 ||
        instLin_[3 * (std::size_t)i + 2] != 0.0 || instAng_[3 * (std::size_t)i] != 0.0 ||
        instAng_[3 * (std::size_t)i + 1] != 0.0 || instAng_[3 * (std::size_t)i + 2] != 0.0;
    if (moves && movingCutCells_[(std::size_t)i] == 0)
      std::fprintf(stderr,
                   "peclet.flow set_solid_from_scene WARNING: instance %d has a velocity but owns "
                   "NO cut row of the momentum operator (sub-cell body?): the wall velocity "
                   "cannot enter and the body behaves as a STATIONARY wall. "
                   "moving_instance_cut_cells() returns the counts.\n",
                   i);
    if (moves && movingDegenerate_[(std::size_t)i] > 0)
      std::fprintf(stderr,
                   "peclet.flow set_solid_from_scene WARNING: instance %d has a velocity and its "
                   "surface passes EXACTLY through %ld staggered velocity points (a face on a "
                   "lattice plane). Such points are fluid to the mask and not ghosts to the "
                   "cut-cell fold, so the wall datum never enters there and the face acts as a "
                   "STATIONARY wall. Shift the body off the lattice (any fractional offset). "
                   "moving_instance_degenerate_points() returns the counts.\n",
                   i, movingDegenerate_[(std::size_t)i]);
  }
}

template <class Grid>
std::vector<long> Solver<Grid>::movingInstanceCutCells() const {
  return movingCutCells_;
}

template <class Grid>
std::vector<long> Solver<Grid>::movingInstanceDegeneratePoints() const {
  return movingDegenerate_;
}

template <class Grid>
long Solver<Grid>::periodicImageOverlapCells() const {
  return imageOverlapCells_;
}

template <class Grid>
void Solver<Grid>::setExactCrossingsFromScene() {
  if (!hasScene_)
    throw std::runtime_error("set_exact_crossings_from_scene: call set_scene first");
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  const auto q = sceneQ_->view();
  const int nx = nx_, ny = ny_, nz = nz_;
  const C3 og = og_;
  const SceneMap sm = sceneMap();
  CCExec space;
  for (int c = 0; c < 3; ++c) {
    // Component c's sample placement comes from the GRID POLICY: staggered puts it on the low
    // face along axis c (offset -1/2 there), collocated at the cell center (offset 0) -- the
    // collocated ghost projection consumes tEx_[c][c] at CENTERS, so hardcoding the staggered
    // offsets here would silently compute crossings from the wrong points on that path.
    const auto po = Grid::offset(c);
    const double offc[3] = {(double)po.x, (double)po.y, (double)po.z};
    for (int a = 0; a < 3; ++a) {
      tEx_[c][a] = CCField("tEx", n);
      CCField t = tEx_[c][a];
      const double ox = offc[0], oy = offc[1], oz = offc[2];
      const int aa = a;
      Kokkos::parallel_for(
          "peclet::flow::scene_crossings",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            // The segment is ONE CELL long along `aa`; `s` stays the dimensionless fraction in
            // (0,1) the consumer expects, so only the endpoints and the step become physical.
            const double px = sm.a[0] + sm.b[0] * ((double)(x + og.x) + ox),
                         py = sm.a[1] + sm.b[1] * ((double)(y + og.y) + oy),
                         pz = sm.a[2] + sm.b[2] * ((double)(z + og.z) + oz);
            const double dx = aa == 0 ? sm.b[0] : 0.0, dy = aa == 1 ? sm.b[1] : 0.0,
                         dz = aa == 2 ? sm.b[2] : 0.0;
            const auto f = [&](double s) {
              return q.eval(peclet::core::Vec3<double>{px + s * dx, py + s * dy, pz + s * dz});
            };
            const double f0 = f(0.0), f1 = f(1.0);
            const std::size_t idx =
                (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
            if ((f0 < 0.0) == (f1 < 0.0)) {  // no sign change -> no crossing on this segment
              t(idx) = Kokkos::Experimental::quiet_NaN_v<double>;
              return;
            }
            double lo = 0.0, hi = 1.0, flo = f0;
            for (int it = 0; it < 52; ++it) {  // bisection to ~1 ulp of the unit interval
              const double mid = 0.5 * (lo + hi);
              const double fm = f(mid);
              if ((fm < 0.0) == (flo < 0.0)) {
                lo = mid;
                flo = fm;
              } else {
                hi = mid;
              }
            }
            t(idx) = 0.5 * (lo + hi);
          });
    }
  }
  space.fence();
  hasExactCross_ = true;
  sceneCrossings_ = true;  // scene-derived: valid on every rank, unlike the host override path
}

template <class Grid>
void Solver<Grid>::buildSceneQuery() {
  namespace g = peclet::core::geom;
  g::PeriodicBox<double> box{sceneExtent_.x, sceneExtent_.y, sceneExtent_.z, scenePeriodic_};
  sceneQ_ = std::make_shared<g::SceneQueryDevice<double, CCMem>>(
      g::SceneQueryDevice<double, CCMem>::build(*sceneB_, sceneOrigin_, sceneExtent_, box));
}

template <class Grid>
void Solver<Grid>::refreshMotionFlag() {
  hasMotion_ = false;
  for (std::size_t k = 0; k < instLin_.size(); ++k)
    if (instLin_[k] != 0.0 || instAng_[k] != 0.0)
      hasMotion_ = true;
}

template <class Grid>
void Solver<Grid>::uploadMotion() {
  const std::size_t m = (std::size_t)nInst_ * 3;
  if (instCenD_.extent(0) != m) {
    instCenD_ = Kokkos::View<double*, CCMem>("instCen", m);
    instLinD_ = Kokkos::View<double*, CCMem>("instLin", m);
    instAngD_ = Kokkos::View<double*, CCMem>("instAng", m);
  }
  if (m == 0)
    return;
  using HostConst =
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(instCenD_, HostConst(instCen_.data(), m));
  Kokkos::deep_copy(instLinD_, HostConst(instLin_.data(), m));
  Kokkos::deep_copy(instAngD_, HostConst(instAng_.data(), m));
}

template <class Grid>
peclet::core::geom::InstanceMotionView<double> Solver<Grid>::motionView() const {
  peclet::core::geom::InstanceMotionView<double> mv;
  mv.cen = instCenD_.data();
  mv.lin = instLinD_.data();
  mv.ang = instAngD_.data();
  mv.n = nInst_;
  return mv;
}

template <class Grid>
void Solver<Grid>::buildWallVelocity() {
  if (!hasScene_ || !hasMotion_) {
    // Never moved: the fields stay EMPTY and every consumer takes its old, bit-identical path.
    // MOVED AND THEN STOPPED is different, and was wrong: wallVelView() keys off the field's
    // extent, not hasMotion_, so a previously-built uBc_ would keep being folded into the
    // momentum operator's inhomogeneity after the caller set the velocity back to zero. Zero
    // them instead of stranding them. Allocation state is unchanged either way, so a run that
    // never moves is bit-identical to before.
    for (int c = 0; c < 3; ++c) {
      if (uBc_[c].extent(0) == n_)
        Kokkos::deep_copy(uBc_[c], 0.0);
      if (uwCell_[c].extent(0) == n_)
        Kokkos::deep_copy(uwCell_[c], 0.0);
    }
    return;
  }
  for (int c = 0; c < 3; ++c) {
    if (uBc_[c].extent(0) != n_)
      uBc_[c] = CCField("uBc", n_);
    if (uwCell_[c].extent(0) != n_)
      uwCell_[c] = CCField("uwCell", n_);
  }
  const auto q = sceneQ_->view();
  const auto mv = motionView();
  CCConst sd = CCConst(sdf_);
  const C3 e = e_, og = og_;
  const SceneMap sm = sceneMap();
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  // rung 2: component c's own staggered point, storing component c
  for (int c = 0; c < 3; ++c) {
    const auto po = Grid::offset(c);
    const double ox = po.x, oy = po.y, oz = po.z;
    CCField out = uBc_[c];
    const int cc = c;
    Kokkos::parallel_for(
        "peclet::flow::wall_velocity_stag", MD(space, {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          const double sx = (double)lx + ox, sy = (double)ly + oy, sz = (double)lz + oz;
          const double s0 = ccSampleExt(sd, e, sx, sy, sz);
          const peclet::core::Vec3<double> grad{
              0.5 * (ccSampleExt(sd, e, sx + 1, sy, sz) - ccSampleExt(sd, e, sx - 1, sy, sz)),
              0.5 * (ccSampleExt(sd, e, sx, sy + 1, sz) - ccSampleExt(sd, e, sx, sy - 1, sz)),
              0.5 * (ccSampleExt(sd, e, sx, sy, sz + 1) - ccSampleExt(sd, e, sx, sy, sz - 1))};
          // wallPoint works on the INDEX level set (sd/grad are index-space); map the point and
          // the wall foot into the scene's coordinates for the instance query, and the scene's
          // physical wall velocity back into the index velocity the momentum datum carries.
          const peclet::core::Vec3<double> pi{sx - G + og.x, sy - G + og.y, sz - G + og.z};
          const peclet::core::Vec3<double> wi = peclet::core::geom::wallPoint(pi, s0, grad);
          const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * pi.x, sm.a[1] + sm.b[1] * pi.y,
                                             sm.a[2] + sm.b[2] * pi.z};
          const peclet::core::Vec3<double> w{sm.a[0] + sm.b[0] * wi.x, sm.a[1] + sm.b[1] * wi.y,
                                             sm.a[2] + sm.b[2] * wi.z};
          const peclet::core::Vec3<double> v =
              peclet::core::geom::instanceVelocity(mv, q.owner(p), w, q.box);
          out(i) = cc == 0 ? v.x * sm.velToInt[0]
                           : (cc == 1 ? v.y * sm.velToInt[1] : v.z * sm.velToInt[2]);
        });
  }
  // rung 3: the whole wall velocity at cell centres (the wall-flux divergence source)
  {
    CCField ux = uwCell_[0], uy = uwCell_[1], uz = uwCell_[2];
    Kokkos::parallel_for(
        "peclet::flow::wall_velocity_cell", MD(space, {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          const double sx = lx, sy = ly, sz = lz;
          const double s0 = ccSampleExt(sd, e, sx, sy, sz);
          const peclet::core::Vec3<double> grad{
              0.5 * (ccSampleExt(sd, e, sx + 1, sy, sz) - ccSampleExt(sd, e, sx - 1, sy, sz)),
              0.5 * (ccSampleExt(sd, e, sx, sy + 1, sz) - ccSampleExt(sd, e, sx, sy - 1, sz)),
              0.5 * (ccSampleExt(sd, e, sx, sy, sz + 1) - ccSampleExt(sd, e, sx, sy, sz - 1))};
          const peclet::core::Vec3<double> pi{sx - G + og.x, sy - G + og.y, sz - G + og.z};
          const peclet::core::Vec3<double> wi = peclet::core::geom::wallPoint(pi, s0, grad);
          const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * pi.x, sm.a[1] + sm.b[1] * pi.y,
                                             sm.a[2] + sm.b[2] * pi.z};
          const peclet::core::Vec3<double> w{sm.a[0] + sm.b[0] * wi.x, sm.a[1] + sm.b[1] * wi.y,
                                             sm.a[2] + sm.b[2] * wi.z};
          const peclet::core::Vec3<double> v =
              peclet::core::geom::instanceVelocity(mv, q.owner(p), w, q.box);
          ux(i) = v.x * sm.velToInt[0];
          uy(i) = v.y * sm.velToInt[1];
          uz(i) = v.z * sm.velToInt[2];
        });
  }
  space.fence();
  // GHOST PLANES (gate 7, 2026-09-02). Both kernels above take a centred difference of the
  // SAMPLED sdf through ccSampleExt, which CLAMPS its indices: on the outermost ghost plane the
  // gradient is wrong, and buildAdvInputs then writes that plane into the advection scratch
  // where the SOU stencil (reach 2) carries it inward. Under MPI that plane is a neighbour's
  // interior, computed there with a full stencil; single-rank periodic it is the wrapped
  // interior plane. Either way the exchange is exact: fill the ghosts from the owner. Measured
  // before the fix: np=2/4 max|du| 1.45e-07 / 1.14e-05 against 3e-7 (bit-exact at np=1), up to
  // 3.5 % of max|u| with a body parked on a rank cut. Static scenes never reach this code.
  for (int c = 0; c < 3; ++c) {
    exchangeExtRaw(uBc_[c]);
    exchangeExtRaw(uwCell_[c]);
  }
}

template <class Grid>
void Solver<Grid>::exchangeExtRaw(CCField f) {
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    velDev_->exchange(f);
    return;
  }
#endif
  for (int a = 0; a < 3; ++a)
    if (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)
      fillAxis(f, a);
}

template <class Grid>
void Solver<Grid>::addWallFluxDivergence(CCField d) {
  if (!hasScene_ || !hasMotion_ || !wallFluxDiv_ || uwCell_[0].extent(0) != n_)
    return;
  CCExec space;
  const C3 e = e_;
  CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
  CCConst wx = CCConst(uwCell_[0]), wy = CCConst(uwCell_[1]), wz = CCConst(uwCell_[2]);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::wall_flux_div", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double ax = -(oxv(i + sx) - oxv(i));
        const double ay = -(oyv(i + sy) - oyv(i));
        const double az = -(ozv(i + sz) - ozv(i));
        d(i) += wx(i) * ax + wy(i) * ay + wz(i) * az;
      });
  space.fence();
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_SCENE_HPP
