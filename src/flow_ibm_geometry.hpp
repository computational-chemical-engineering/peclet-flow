/// @file
/// @brief flow — IbmSolver static solid geometry: setSolid / setSolidDevice and its stages,
/// pressure geometry.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_GEOMETRY_HPP
#define PECLET_FLOW_FLOW_IBM_GEOMETRY_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::setPressureGeometry(const std::vector<double>& sdfInner) {
  setSolid(sdfInner, true);
}

template <class Grid>
void Solver<Grid>::setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
  const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
  if (sdfInner.size() != n)
    throw std::runtime_error("set_solid: expected nx*ny*nz values");
  CCField din("peclet::flow::sdfInner_d", n);
  using HostConst =
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  // The caller's SDF is a PHYSICAL signed distance sampled at cellCentres(); the solver's
  // geometry lives on the unit lattice, where a distance is measured in cells. (Isotropic in
  // Phase 1: one hRef divides all three axes. Anisotropic cells need the metric of Phase 2,
  // because the level set's gradient then stops being the index-space normal.)
  const double k = u_.lenToInt();  // exactly 1.0 in cell units
  if (k != 1.0) {
    std::vector<double> scaled(n);
    for (std::size_t i = 0; i < n; ++i)
      scaled[i] = sdfInner[i] * k;
    Kokkos::deep_copy(din, HostConst(scaled.data(), n));
  } else {
    Kokkos::deep_copy(din, HostConst(sdfInner.data(), n));
  }
  setSolidDevice(din, cutcellPressure);
}

template <class Grid>
void Solver<Grid>::buildVelocityOverlays(bool resetU) {
  const bool useEx = hasExactCross_ && !Grid::collocated;  // exact-theta arrays are for the
                                                           // staggered point placement
  const mreal lam = (mreal)(wallSlip_ ? slipLambda_ : 0.0);
  if (lam > 0.0f && slipSkipDev_.data() == nullptr)
    slipSkipDev_ = Kokkos::View<int, CCMem>("peclet::flow::slipSkip");
  for (int c = 0; c < 3; ++c) {
    const Off3 off =
        Grid::offset(c);  // velocity-unknown placement (staggered: -1/2 face; collocated: 0)
    C[c].nCut = buildIbmOverlay<0>(
        CCConst(sdf_), e_, G, off, /*Dirichlet*/ 0, C[c].ov, C[c].idMap, C[c].counter,
        useEx ? CCConst(tEx_[c][0]) : CCConst(), useEx ? CCConst(tEx_[c][1]) : CCConst(),
        useEx ? CCConst(tEx_[c][2]) : CCConst(), C3{nx_, ny_, nz_}, lam, lam > 0.0f ? c : -1,
        lam > 0.0f ? slipSkipDev_ : Kokkos::View<int, CCMem>(),  // SCHEME 0 = point-value (matches
                                                                 // CUDA ibm_geometry_ext_k<0>)
        (mreal)u_.hp[0], (mreal)u_.hp[1], (mreal)u_.hp[2]);      // §6.4 slip metric (trap 9)
    if (lam > 0.0f) {
      int sk = 0;
      Kokkos::deep_copy(sk, slipSkipDev_);
      slipSandwich_[(std::size_t)c] = sk;
    } else {
      slipSandwich_[(std::size_t)c] = 0;
    }
    ibmSolidMask(C[c].mask, CCConst(sdf_), e_, off);
    // Gate 7: ibmSolidMask samples the sdf at the staggered offset through the CLAMPING
    // sampler, so on the outermost ghost plane the mask can disagree with the neighbour's
    // interior value. The moving-geometry advection fill reads the ghost mask (it decides
    // which ghost rows carry the wall velocity), so take the owner's mask there. Static
    // scenes never consume ghost masks: keep them byte-identical by gating on motion.
    if (hasMotion_)
      exchangeExtRaw(C[c].mask);
    if (resetU)
      Kokkos::deep_copy(C[c].u, 0.0);
  }
}

template <class Grid>
void Solver<Grid>::mirrorSdfSlipFaces(CCField f) {
  if (!hasBc_)
    return;
  B3 e{e_.x, e_.y, e_.z};
  for (int face = 0; face < 6; ++face)
    if (bc_[face] == 4 && touchesGlobalFace(face))
      bcMirrorGhost(f, e, G, face / 2, face % 2);
}

template <class Grid>
void Solver<Grid>::setSolidDevice(CCField din, bool cutcellPressure) {
  cutcellPressure_ = cutcellPressure;
  setSolidSelectScheme();
  setSolidUploadSdf(din);
  setSolidBuildOverlaysAndStencils();
  setSolidVelocityMgAuto();
  setSolidInitVelocityMg();
  if (cutcellPressure_) {
    setSolidBuildOpenness();
    setSolidStarOverlay();
    setSolidGhostProjectionOverlay(din);
    setSolidInitPressureMg();
  }
  // Rung V5a (WO-Q): the colour block's cut-cell geometry is derived from the openness and the
  // SDF that were just rebuilt, so set_solid AFTER enable_vof must rebuild it — the same reason
  // initMpi rebuilds the block. Inert (and byte-identical) when VoF is off.
  if (vofEnabled_)
    buildVofBlock();
  geometryBuilt_ = true;
}

template <class Grid>
void Solver<Grid>::setSolidSelectScheme() {
  if constexpr (Grid::collocated) {
    // DEFAULT SWITCH (2026-08-25, user decision after the attractor campaign): the collocated
    // scheme default is AUTO = the GHOST (fluid-only) projection — family-free, unconditionally
    // stable, protocol-independent (doc/collocated_invisible_subspace.md; clean ladders both
    // beds) — falling back to gauge-exact with a stderr notice on the configurations the ghost
    // v1 does not support (porous / variable-rho / domain-BC / Chebyshev / analytic overrides).
    // Any explicit scheme selection (set_collocated_scheme / set_face_interp /
    // set_ghost_projection / set_fluid_only_constraint) disables AUTO.
    if (colSchemeAuto_) {
      const bool ok = !(porous_ || varRho_ || hasBc_ || useChebyshev_ || hasExactCross_ ||
                        hasOpenOverride_ || fluidOnlyMode_ != 0);
      if (ok) {
        ghostProjection_ = true;
        gpMatrixOrder_ = 2;
        gpRhsOrder_ = 2;
        faceInterp_ = 0;  // the ghost owns the operators the face-interp modes replace
        gpNRows_ = -1;
      } else {
        if (ghostProjection_)
          gpNRows_ = -1;
        ghostProjection_ = false;
        faceInterp_ = 9;
        fprintf(stderr,
                "peclet::flow SolverColocated: AUTO scheme fell back to gauge-exact "
                "(configuration unsupported by the ghost projection v1). Select explicitly "
                "with set_collocated_scheme to silence this notice.\n");
      }
    }
  }
}

template <class Grid>
void Solver<Grid>::setSolidUploadSdf(CCField din) {
#ifdef PECLET_FLOW_MPI
  for (bool& d : momStencilDirty_)  // stencil ring re-exchange for the CA momentum sweeps
    d = true;
#endif
  // Does the geometry actually contain solid? (all-fluid set_pressure_geometry passes sd>0
  // everywhere -> stays false, keeping the channel/BFS path.) Device reduction: din lives on
  // device now, and pulling it back just to scan it would defeat the point.
  {
    const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
    int anySolid = 0;
    CCConst dinC(din);
    Kokkos::parallel_reduce(
        "peclet::flow::has_solid", Kokkos::RangePolicy<CCExec>(0, nInner),
        KOKKOS_LAMBDA(const std::size_t i, int& acc) { acc = acc || (dinC(i) < 0.0); }, anySolid);
    hasSolid_ = anySolid != 0;
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {  // a solid anywhere in the global domain enables the IBM momentum path
    int local = hasSolid_ ? 1 : 0, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm_);
    hasSolid_ = global != 0;
  }
#endif
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    // Multi-rank: din is THIS rank's LOCAL inner block; fill the inner cells ON DEVICE, then
    // halo-exchange the ghosts (cross-rank + periodic) so the overlay/openness read the
    // neighbour's SDF at the block boundary. (Was a host mirror + triple loop + full H2D.)
    CCExec space;
    const int ex = e_.x, ey = e_.y, nx = nx_, ny = ny_, nz = nz_, g = G;
    CCField sdf = sdf_;
    CCConst dinC(din);
    Kokkos::parallel_for(
        "peclet::flow::sdf_fill_inner",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          sdf((long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey) =
              dinC((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
        });
    space.fence();
    velDev_->exchange(sdf_);
  } else
#endif
  {
    // Single-rank: periodic-wrap gather on device (G4) — fills the whole extended block
    // (inner + periodic ghosts) in one kernel. `din` is already device-resident.
    CCExec space;
    const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
    CCField sdf = sdf_;
    Kokkos::parallel_for(
        "peclet::flow::sdf_periodic_wrap",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                    iz = (((z - g) % nz) + nz) % nz;
          sdf((long)x + (long)y * ex + (long)z * (long)ex * ey) =
              din((std::size_t)ix + (std::size_t)iy * nx + (std::size_t)iz * (std::size_t)nx * ny);
        });
    space.fence();
  }
}

template <class Grid>
void Solver<Grid>::setSolidBuildOverlaysAndStencils() {
  // FREE-SLIP / symmetry faces (type 4): the periodic/halo fill above wrapped the OPPOSITE side
  // of the domain into the ghost band, so a wall at the far side becomes a phantom solid ON the
  // symmetry plane (measured: a half Poiseuille channel closed by a type-4 face read u ~ 0
  // there). The symmetric extension is the mirror of the interior about the face, which is
  // exactly what the BC asserts about the geometry. Rank-owned faces only (touchesGlobalFace);
  // the other face types keep the wrap they always had (byte-identical).
  mirrorSdfSlipFaces(sdf_);
  buildVelocityOverlays(/*resetU=*/true);
  // MOVING GEOMETRY (rung 2): the wall-velocity fields must exist BEFORE the momentum operator
  // is assembled -- rebuildStencils folds them into the inhomogeneous term. sdf_ and its ghosts
  // are final at this point, which is what the central-difference normals read.
  buildWallVelocity();
  rebuildStencils();
  // Staggered domain BCs bake an implicit-diffusion wall fold; the collocated grid instead uses
  // explicit reflection ghosts (refreshed each smoother sweep), so it needs no fold.
  if (hasBc_ && !Grid::collocated)
    setupBcDiffusion();
}

template <class Grid>
void Solver<Grid>::setSolidVelocityMgAuto() {
#ifdef PECLET_FLOW_MPI
  // AUTO: pick the V-cycle when the per-rank block is small (see setVelocityMultigridAuto). The
  // decision uses the GLOBAL cells / ranks, so every rank agrees without communication. Only on
  // the validated operator modes: IBM-periodic, all-fluid domain-BC (explicit advection), mixed.
  if (!vmgExplicit_ && distributed_ && vmgAutoCells_ > 0) {
    int np = 1;
    MPI_Comm_size(comm_, &np);
    const double perRank = (double)gnx_ * gny_ * gnz_ / (double)np;
    const bool eligible = !varProps_ && !varRho_ && !hasDrag_ && !porous_ && !Grid::collocated &&
                          (!hasBc_ || hasSolid_ || !implicitFou_);
    // np > 1: a single rank has no halo latency to hide (RB-GS is the cheaper solver there) and
    // a distributed np=1 run must stay bit-identical to the single-rank path. Global size floor:
    // the rule is about latency-bound LARGE runs; a small global problem split across ranks
    // (every ctest, every quick check) keeps RB-GS so distributed == single-rank stays exact.
    const double global = (double)gnx_ * gny_ * gnz_;
    useVelocityMg_ = eligible && np > 1 && global >= (double)vmgAutoMinGlobal_ &&
                     perRank < (double)vmgAutoCells_;
    if (useVelocityMg_) {
      vmgLevels_ = 3;
      vmgVcycles_ = 40;
    }
  }
#endif
}

template <class Grid>
void Solver<Grid>::setSolidInitVelocityMg() {
  if (useVelocityMg_) {  // velocity-MG hierarchy: IBM (staircase/upwind), domain-BC
                         // (const-coeff) or mixed (staircase + folds) mode
    // The per-axis metric BEFORE the hierarchy is built (doc/anisotropic_metric.md trap 5):
    // every level's b_a^L = mu' * w_a / cfac_a^2, and C3's aspect-ratio level rule reads it too.
    vmg_.setMetric(u_.w, u_.hp);
#ifdef PECLET_FLOW_MPI
    // Distributed: level 0 on the solver's own decomposition (the g=2 velocity block), coarse
    // levels coarsened in place with the even-block gate (no telescoping here yet -- measured
    // first, see docs/SCALING_ISSUES.md issue 5).
    if (distributed_)
      vmg_.initMpi(*dec_, vmgLevels_, comm_);
    else
#endif
      vmg_.init(nx_, ny_, nz_, vmgLevels_);
    if (hasBc_)
      vmg_.setBC(bc_);
    if (!hasBc_ || hasSolid_) {  // the staircase (IBM / mixed) paths classify by volume fraction
      vmgTheta_ = CCField("vmgTheta", n_);
      vmgClean_ = CCField("vmgClean", n_);
    }
  }
}

template <class Grid>
void Solver<Grid>::setSolidBuildOpenness() {
  // Phase 2 C2 (doc/anisotropic_metric.md §4.5, trap 7): buildOpenness has taken dx,dy,dz
  // all along and was handed 1.0 — `ccFractionCore` (the order-1 aperture model) forms the
  // gradient (s+ - s-)/(2 dx) and the face extent |n_b| dy + |n_c| dz on the index lattice, so
  // the right metric is h_a' (the marching-squares path, order 2, ignores them).
  buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, u_.hp[0], u_.hp[1], u_.hp[2],
                apertureOrder_);  // on the g=2 velocity block
  if (hasOpenOverride_) {
    // Analytic-SDF exact apertures (setOpennessOverride): overwrite the sampled-SDF openness
    // with the user-provided inner fields + periodic wrap into the ghost ring (single-rank).
    const std::vector<double>* src[3] = {&oxOverride_, &oyOverride_, &ozOverride_};
    CCField dst[3] = {ox_, oy_, oz_};
    for (int f = 0; f < 3; ++f) {
      CCField din("peclet::flow::openOv_d", (std::size_t)nx_ * ny_ * nz_);
      Kokkos::deep_copy(
          din,
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              src[f]->data(), src[f]->size()));
      CCExec space;
      const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
      CCField o = dst[f];
      Kokkos::parallel_for(
          "peclet::flow::open_override_wrap",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                      iz = (((z - g) % nz) + nz) % nz;
            o((long)x + (long)y * ex + (long)z * (long)ex * ey) = din(
                (std::size_t)ix + (std::size_t)iy * nx + (std::size_t)iz * (std::size_t)nx * ny);
          });
      space.fence();
    }
  }
  if constexpr (Grid::collocated) {  // static open-centroid wall distances (wall-aware map)
    buildFaceCentroidDist(xcx_, xcy_, xcz_, CCConst(sdf_), e_);
    buildCellFraction(cs_, CCConst(sdf_), e_, G);  // cell fluid fraction (embed)
    if (faceInterp_ >= 5 && faceInterp_ <= 7) {    // EMBED: a solid-CENTRED cut cell (cs>0) is
                                                   // partially fluid and holds
      // its reconstructed near-wall velocity — masking it to 0 (the sdf<0 IBM mask) drops the
      // near-wall closure and shifts the whole channel. Re-mask from cs: pin ONLY fully-solid
      // cells (cs≈0), keeping every partial-fluid cut cell live in the embed solve +
      // projection.
      CCConst cs = CCConst(cs_);
      const std::size_t nn = n_;
      for (int c = 0; c < 3; ++c) {
        CCField m = C[c].mask;
        Kokkos::parallel_for(
            "peclet::flow::embed_solid_mask", Kokkos::RangePolicy<CCExec>(0, nn),
            KOKKOS_LAMBDA(std::size_t i) { m(i) = cs(i) < 1e-6 ? 1.0 : 0.0; });
      }
    }
  }
  if (fluidOnlyMode_ == 1) {
    // Mode-14a FLUID-ONLY constraint (setFluidOnlyConstraint): close every face with a
    // solid-CENTERED side in the openness the pressure stack consumes. The aperture operator,
    // the divergence, the face correction and the MG rediscretization all read these fields,
    // so one filter makes constraint/operator/correction consistent by construction: pressure
    // DOFs decouple at solid-centered cells (their rows go empty like solid cells), the
    // invisible multiplier subspace of collocated_invisible_subspace.md S4 ceases to exist,
    // and the operator stays SPD + 7-point (CG + CutcellMG untouched). Closure quality is
    // Neumann-zero at the closed faces (the crude end of the fluid-only family -- measured,
    // not assumed); the consistent-closure variants ride on the gp row machinery instead.
    CCConst sd = CCConst(sdf_);
    CCField oa[3] = {ox_, oy_, oz_};
    C3 e = e_;
    for (int a = 0; a < 3; ++a) {
      CCField o = oa[a];
      const long sa = (a == 0) ? 1 : (a == 1) ? (long)e.x : (long)e.x * e.y;
      Kokkos::parallel_for(
          "peclet::flow::fluid_only_openness",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {1, 1, 1}, {e.x, e.y, e.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            if (sd(i) < 0.0 || sd(i - sa) < 0.0)
              o(i) = 0.0;
          });
    }
  }
#ifdef PECLET_FLOW_MPI
  // openness ghosts (the operator + divergence read the +neighbour face) -> exchange across
  // ranks
  if (distributed_) {
    velDev_->exchange(ox_);
    velDev_->exchange(oy_);
    velDev_->exchange(oz_);
  }
#endif
  if (hasBc_) {  // FLUX openness (beta): a face is OPEN only where it carries normal flux --
                 // outflow, or
    B3 e2{e_.x, e_.y, e_.z};
    CCField oa[3] = {ox_, oy_, oz_};  // an inflow with nonzero normal velocity. Walls
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {    // and tangential-only Dirichlet faces (e.g. a
        const int t = bc_[2 * a + s];  // lid: type 2 with zero normal vel) are CLOSED.
        const bool open = (t == 3) || (t == 2 && (bcProf_[2 * a + s].extent(0) > 0 ||
                                                  std::fabs(bcVel_[2 * a + s][a]) > 1e-12));
        if (t != 0 && !open && touchesGlobalFace(2 * a + s))
          bcZeroOpenness(oa[a], e2, G, a, s);  // rank-owned global face only
      }
  }  // the MG re-derives the OPERATOR openness alpha (inflow Neumann -> closed) per level via
     // setBC.
  copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
  copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
  copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
}

template <class Grid>
void Solver<Grid>::setSolidStarOverlay() {
  if (fluidOnlyMode_ == 2) {
    // Design B: the MG hierarchy is built from the FILTERED openness (Design A's operator,
    // the symmetric surrogate preconditioner + the 7-point part of the true operator); the
    // geometric ox_/oy_/oz_ stay ORIGINAL for the divergence and the face correction. Filter
    // the g=1 bridge in place, then build the star overlay from the original apertures.
    if (porous_ || varRho_ || hasBc_ || ghostProjection_ || distributed_ || !Grid::collocated)
      throw std::runtime_error(
          "set_fluid_only_constraint(2): v1 is single-rank periodic collocated only");
    CCExec space;
    CCConst sd = CCConst(sdf_);
    CCField oa1[3] = {ox1_, oy1_, oz1_};
    const C3 e1 = e1_, e2 = e_;
    for (int a = 0; a < 3; ++a) {
      CCField o1 = oa1[a];
      const long sa2 = (a == 0) ? 1 : (a == 1) ? (long)e2.x : (long)e2.x * e2.y;
      Kokkos::parallel_for(
          "peclet::flow::star_filter_bridge",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx_, ny_, nz_}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i2 =
                (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
            if (sd(i2) < 0.0 || sd(i2 - sa2) < 0.0)
              o1((long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y) = 0.0;
          });
    }
    space.fence();
    starCounter_ = Kokkos::View<int, CCMem>("star_counter");
    const C3 nn{nx_, ny_, nz_};
    nStar_ = buildStarOverlay(CCConst(sdf_), CCConst(ox_), CCConst(oy_), CCConst(oz_), e_, G, nn,
                              StarOverlay{}, starCounter_);
    starOv_ = starMakeOverlay(std::max(nStar_, 1));
    buildStarOverlay(CCConst(sdf_), CCConst(ox_), CCConst(oy_), CCConst(oz_), e_, G, nn, starOv_,
                     starCounter_);
  }
}

template <class Grid>
void Solver<Grid>::setSolidGhostProjectionOverlay(CCField din) {
  // A few setup paths below (the ghost-projection pocket decoupling) are host-side
  // connected-component analyses and genuinely need the inner SDF on the host. Materialise it
  // ONCE, lazily, so the common path keeps the device-resident benefit.
  std::vector<double> sdfInnerHost_;
  auto sdfInner_ = [&]() -> const std::vector<double>& {
    if (sdfInnerHost_.empty()) {
      const std::size_t nI = (std::size_t)nx_ * ny_ * nz_;
      sdfInnerHost_.resize(nI);
      Kokkos::deep_copy(
          Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              sdfInnerHost_.data(), nI),
          din);
    }
    return sdfInnerHost_;
  };
  if (ghostProjection_) {
    // Directional ghost-cell projection: build the closure overlay + the binary (COUPLED)
    // openness. The binary field replaces the geometric openness on the MG rails (the MG
    // hierarchy becomes the symmetric surrogate preconditioner; the overlay delta enters only
    // the fine-level BiCGStab matvec). The geometric ox_/oy_/oz_ above stay for diagnostics.
    if (porous_ || varRho_ || hasBc_)
      throw std::runtime_error(
          "ghost projection: incompatible with porous/variable-rho/domain-BC (v1)");
    const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
    gpOv_ = gpMakeOverlay<MReal>((long)nInner);  // worst-case sizing, like the momentum overlay
    gpIdMap_ = Kokkos::View<int*, CCMem>("gp_idmap", nInner);
    gpCounter_ = Kokkos::View<int, CCMem>("gp_counter");
    oxb_ = CCField("oxb", n_);
    oyb_ = CCField("oyb", n_);
    ozb_ = CCField("ozb", n_);
    gpRh_ = CCField("gpRh", n1_);
    gpT_ = CCField("gpT", n1_);
    gpZ2_ = CCField("gpZ2", n1_);
    if (distributed_)
      gpX2_ = CCField("gpXg2", n_);  // g=2 staging block for the distributed BiCGStab matvec
    // Fragmentation guard: the binary COUPLED-face condition is stricter than aperture
    // connectivity, so tight-throat geometries (e.g. a random close packing with touching
    // spheres) fragment the fluid graph into a main component + tiny pockets at the
    // contacts. Each pocket adds its own null vector that the single global mean-removal
    // cannot handle, and BiCGStab breaks down (measured: fields to ~1e152 on the RCP
    // example). Host BFS over the coupled graph of the INNER sdf; fluid cells outside the
    // largest component are treated as SOLID for the PROJECTION ONLY (sdfGp), decoupling
    // their rows; the momentum step keeps the true sdf.
    // Distributed: connectivity is a GLOBAL property (a pocket can span rank boundaries) and
    // every rank must agree on the main component, so allgather the inner sdf, run the
    // deterministic guard on the global grid identically on every rank, and keep this
    // rank's block of the result.
    std::vector<double> sdfGpHost;
    {
      std::vector<double> work;
      int fx = nx_, fy = ny_, fz = nz_;
      bool verbose = true;
#ifdef PECLET_FLOW_MPI
      int myRank = 0;
      if (distributed_) {
        MPI_Comm_rank(comm_, &myRank);
        int nRanks = 1;
        MPI_Comm_size(comm_, &nRanks);
        fx = gnx_;
        fy = gny_;
        fz = gnz_;
        verbose = myRank == 0;
        std::vector<int> cnts(nRanks), disp(nRanks);
        long acc = 0;
        for (int r = 0; r < nRanks; ++r) {
          const auto b = dec_->block(r);
          cnts[r] = (int)(b.size[0] * b.size[1] * b.size[2]);
          disp[r] = (int)acc;
          acc += cnts[r];
        }
        std::vector<double> flat((std::size_t)acc);
        MPI_Allgatherv(sdfInner_().data(), (int)sdfInner_().size(), MPI_DOUBLE, flat.data(),
                       cnts.data(), disp.data(), MPI_DOUBLE, comm_);
        work.assign((std::size_t)fx * fy * fz, 0.0);
        for (int r = 0; r < nRanks; ++r) {
          const auto b = dec_->block(r);
          const double* src = flat.data() + disp[r];
          for (int z = 0; z < (int)b.size[2]; ++z)
            for (int y = 0; y < (int)b.size[1]; ++y)
              for (int x = 0; x < (int)b.size[0]; ++x)
                work[(std::size_t)(x + b.origin[0]) + (std::size_t)(y + b.origin[1]) * fx +
                     (std::size_t)(z + b.origin[2]) * (std::size_t)fx * fy] =
                    src[(std::size_t)x + (std::size_t)y * b.size[0] +
                        (std::size_t)z * (std::size_t)b.size[0] * b.size[1]];
        }
      } else
#endif
        work = sdfInner_();
      const std::size_t nTot = work.size();
      const int nx = fx, ny = fy, nz = fz;
      auto id = [&](int x, int y, int z) {
        return (std::size_t)((x + nx) % nx) + (std::size_t)((y + ny) % ny) * nx +
               (std::size_t)((z + nz) % nz) * (std::size_t)nx * ny;
      };
      std::vector<int> comp(nTot, -1);
      std::vector<std::size_t> stack;
      int ncomp = 0, mainComp = -1;
      std::size_t mainSize = 0, nActive = 0;
      for (std::size_t seed = 0; seed < nTot; ++seed) {
        if (comp[seed] >= 0 || work[seed] < 0.0)
          continue;
        std::size_t size = 0;
        comp[seed] = ncomp;
        stack.assign(1, seed);
        while (!stack.empty()) {
          const std::size_t c = stack.back();
          stack.pop_back();
          ++size;
          const int x = (int)(c % nx), y = (int)((c / nx) % ny),
                    z = (int)(c / ((std::size_t)nx * ny));
          const int nb[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
          for (auto& q : nb) {
            const std::size_t j = id(q[0], q[1], q[2]);
            if (comp[j] >= 0 || work[j] < 0.0)
              continue;
            // COUPLED face: mean-of-centers face sdf fluid AND both centers fluid
            if (0.5 * (work[c] + work[j]) < 0.0)
              continue;
            comp[j] = ncomp;
            stack.push_back(j);
          }
        }
        if (size > mainSize) {
          mainSize = size;
          mainComp = ncomp;
        }
        nActive += size;
        ++ncomp;
      }
      if (ncomp > 1) {
        std::size_t pockets = 0;
        for (std::size_t i = 0; i < nTot; ++i)
          if (work[i] >= 0.0 && comp[i] != mainComp) {
            work[i] = -(std::abs(work[i]) * 1.001 + 1e-30);
            ++pockets;
          }
        if (verbose)
          printf(
              "peclet::flow ghost projection: %d fluid components; decoupled %zu pocket "
              "cells outside the main component (%zu of %zu fluid cells)\n",
              ncomp, pockets, mainSize, nActive);
      }
#ifdef PECLET_FLOW_MPI
      if (distributed_) {
        const auto b = dec_->block(myRank);
        sdfGpHost.resize(sdfInner_().size());
        for (int z = 0; z < nz_; ++z)
          for (int y = 0; y < ny_; ++y)
            for (int x = 0; x < nx_; ++x)
              sdfGpHost[(std::size_t)x + (std::size_t)y * nx_ +
                        (std::size_t)z * (std::size_t)nx_ * ny_] =
                  work[(std::size_t)(x + b.origin[0]) + (std::size_t)(y + b.origin[1]) * fx +
                       (std::size_t)(z + b.origin[2]) * (std::size_t)fx * fy];
      } else
#endif
        sdfGpHost = std::move(work);
    }
    sdfGp_ = CCField("peclet::flow::sdfGp", n_);
    CCField sdfGp = sdfGp_;  // the projection's sdf view (pockets decoupled); persisted for
                             // the collocated gpCenterGrad predictor/correction
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      // local inner block + halo exchange (cross-rank + periodic), same as the sdf_ upload:
      // the overlay build reads sdfGp ghosts up to +/-2 = G across block boundaries.
      auto h = Kokkos::create_mirror_view(sdfGp_);
      Kokkos::deep_copy(h, sdfGp_);
      for (int z = 0; z < nz_; ++z)
        for (int y = 0; y < ny_; ++y)
          for (int x = 0; x < nx_; ++x)
            h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
                sdfGpHost[(std::size_t)x + (std::size_t)y * nx_ +
                          (std::size_t)z * (std::size_t)nx_ * ny_];
      Kokkos::deep_copy(sdfGp_, h);
      velDev_->exchange(sdfGp_);
    } else
#endif
    {  // upload + periodic wrap (same pattern as the sdf upload above)
      CCField din("peclet::flow::sdfGpInner_d", nInner);
      Kokkos::deep_copy(
          din,
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              sdfGpHost.data(), sdfGpHost.size()));
      CCExec space;
      const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
      Kokkos::parallel_for(
          "peclet::flow::sdfgp_wrap",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                      iz = (((z - g) % nz) + nz) % nz;
            sdfGp((long)x + (long)y * ex + (long)z * (long)ex * ey) = din(
                (std::size_t)ix + (std::size_t)iy * nx + (std::size_t)iz * (std::size_t)nx * ny);
          });
      space.fence();
    }
    gpBinaryOpenness(oxb_, oyb_, ozb_, CCConst(sdfGp), e_);
    gpNRows_ = buildGpOverlay(CCConst(sdfGp), e_, G, C3{nx_, ny_, nz_}, gpOv_, gpIdMap_, gpCounter_,
                              gpMatrixOrder_, gpRhsOrder_,
                              hasExactCross_ ? CCConst(tEx_[0][0]) : CCConst(),
                              hasExactCross_ ? CCConst(tEx_[1][1]) : CCConst(),
                              hasExactCross_ ? CCConst(tEx_[2][2]) : CCConst(),
                              /*useGhost=*/distributed_);
    if (gpDebugLevel() > 0) {  // PECLET_FLOW_GP_DEBUG row forensics (analysis only)
      int gpDbgRank = 0;
#ifdef PECLET_FLOW_MPI
      if (distributed_)
        MPI_Comm_rank(comm_, &gpDbgRank);
#endif
      gpDebugReport(gpOv_, gpNRows_, C3{nx_, ny_, nz_}, gpIdMap_, gpDbgRank);
    }
    copyInner(ox1_, e1_, 1, CCConst(oxb_), e_, G);  // MG surrogate = binary openness
    copyInner(oy1_, e1_, 1, CCConst(oyb_), e_, G);
    copyInner(oz1_, e1_, 1, CCConst(ozb_), e_, G);
  }
}

template <class Grid>
void Solver<Grid>::setSolidInitPressureMg() {
  mg_.setBoundaryConditions(bc_);  // per-level wall openness + null-space gating (no-op if
                                   // periodic); BEFORE initMpi — the per-level ghost width
                                   // (CA smoothing) is chosen for the periodic operator only
  // Phase 2 C3 (doc/anisotropic_metric.md §5, trap 5): the per-axis spacings BEFORE the level
  // table is built — the aspect-ratio rule defers an axis that is already theta times coarser
  // than the finest coarsenable one.  Exactly (1,1,1) on the isotropic path, where the rule
  // returns today's decision verbatim.
  mg_.setMetric(u_.hp);
#ifdef PECLET_FLOW_MPI
  if (distributed_)  // share the level-0 decomposition so the MG block matches this rank's
                     // block
    mg_.initMpi(gnx_, gny_, gnz_, nLevels_, comm_, dec_.get());
  else
#endif
    mg_.init(nx_, ny_, nz_,
             nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
  // Phase 2 C2 (doc/anisotropic_metric.md §1.3/§3): the per-axis pressure weight
  // w_a = 1/h_a'^2 IS `setOpenness`'s idx2/idy2/idz2 (the coarse levels already form
  // w_a/cfac_a^2).  Exactly 1.0 on the isotropic path.
  mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), u_.w[0], u_.w[1], u_.w[2]);
  // Coarse-solve policy: an explicit set_pressure_graph_amg(True) forces agglomeration,
  // otherwise the mode set by set_pressure_bottom (default auto) decides.
  mg_.setAgglomerationMode(pressGraphAmg_ ? 1 : pressAgglomMode_);
  Kokkos::deep_copy(phi_, 0.0);
  Kokkos::deep_copy(P_, 0.0);
}

template <class Grid>
bool Solver<Grid>::geometryBuilt() const {
  return geometryBuilt_;
}

template <class Grid>
void Solver<Grid>::requireNoGeometry(const char* who) const {
  if (geometryBuilt_)
    throw std::runtime_error(
        std::string(who) +
        ": call BEFORE the geometry (set_solid / set_pressure_geometry / set_solid_from_scene) "
        "-- this setting is folded into the operators when the geometry is built, so a later "
        "call would be silently ignored");
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_GEOMETRY_HPP
