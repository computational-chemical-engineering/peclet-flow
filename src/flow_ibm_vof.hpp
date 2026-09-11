/// @file
/// @brief flow — IbmSolver Volume-of-Fluid: the colour field, cut-cell VoF geometry, contact angle,
/// VoF blocks, curvature, surface tension / CSF RHS, kinematic and momentum-consistent advection,
/// stepAdaptive.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_VOF_HPP
#define PECLET_FLOW_FLOW_IBM_VOF_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::setVofTiming(bool on) {
  vofTiming_ = on;
  vofAdv_.timingOn = on;
  vofCurv_.timingOn = on;
  resetVofTiming();
}

template <class Grid>
bool Solver<Grid>::vofTiming() const {
  return vofTiming_;
}

template <class Grid>
void Solver<Grid>::resetVofTiming() {
  vt_ = VofTiming();
  vofAdv_.resetTiming();
  vofCurv_.resetTiming();
  tStepSum_ = tPredSum_ = tMomSum_ = tProjSum_ = 0.0;
}

template <class Grid>
const typename Solver<Grid>::VofTiming& Solver<Grid>::vofTimingReport() const {
  return vt_;
}

template <class Grid>
const vof::WyAdvector::Timing& Solver<Grid>::vofKernelTiming() const {
  return vofAdv_.timing();
}

template <class Grid>
const vof::VofCurvature::Timing& Solver<Grid>::vofCurvatureTiming() const {
  return vofCurv_.timing();
}

template <class Grid>
void Solver<Grid>::setVofCurvatureWorklist(bool on) {
  vofCurv_.useWorklist = on;
}

template <class Grid>
bool Solver<Grid>::vofCurvatureWorklist() const {
  return vofCurv_.useWorklist;
}

template <class Grid>
double Solver<Grid>::vofTimingStepSeconds() const {
  return tStepSum_;
}

template <class Grid>
double Solver<Grid>::vofTimingPredictorSeconds() const {
  return tPredSum_;
}

template <class Grid>
double Solver<Grid>::vofTimingMomentumSeconds() const {
  return tMomSum_;
}

template <class Grid>
double Solver<Grid>::vofTimingProjectionSeconds() const {
  return tProjSum_;
}

template <class Grid>
void Solver<Grid>::setVofWorklist(bool on) {
  vofAdv_.useWorklist = on;
}

template <class Grid>
bool Solver<Grid>::vofWorklist() const {
  return vofAdv_.useWorklist;
}

template <class Grid>
void Solver<Grid>::buildVofBlock() {
  vofAdv_.init(nx_, ny_, nz_, 1.0, kVofG);  // h = 1: flow works in cell units
  vofAdv_.cflLimit = vofCflLimit_;
  vofAdv_.interfaceLocalCfl = true;  // WO-J item 4 — see maxCourantInterface()
  const I3 e3 = vofAdv_.extent();
  e3_ = C3{e3.x, e3.y, e3.z};
  vofAdv_.exchange = [this](CCField f) { this->vofFillGhosts(f); };
  vofAdv_.globalMax = nullptr;
#ifdef PECLET_FLOW_MPI
  if (distributed_ && dec_) {
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    // Periodic on all three axes, exactly like the velocity halo: the halo owns every interior
    // ghost and wraps the global boundary, and vofFillGhosts then overwrites the out-of-domain
    // ghosts of any non-periodic axis with the globally-clamped value.
    std::array<bool, 3> per{true, true, true};
    vofHalo_ = std::make_shared<GridHaloTopology<3>>();
    vofHalo_->buildTopology(*dec_, rank, kVofG, per, comm_);
    vofDev_ = std::make_shared<GridHalo<double>>();
    vofDev_->setLabel("vof g3");
    vofDev_->init(*vofHalo_);
    MPI_Comm cm = comm_;
    vofAdv_.globalMax = [cm](double v) {
      double r = v;
      MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, cm);
      return r;
    };
  }
#endif
  buildVofGeometry();  // rung V5a (WO-Q): openness + fluid fraction on the colour block
  // WO-R: the out-of-domain mask and the resampled boundary-colour profiles are properties of
  // THIS block, so they are (re)built with it — and the mask is only INSTALLED on the advector
  // when a VoF boundary colour has actually been set, which is what keeps the V1 flux path
  // bit-identical otherwise (gate G5).
  vofRebuildBcBlock();
  // The half-shifted momentum CVs live on this same block, so they are rebuilt with it.
  vofCurv_.init(nx_, ny_, nz_, kVofG);
  if (vofMomEnabled_) {
    vofMom_.init(vofAdv_, vofRhoG_, vofRhoL_);
    for (int c = 0; c < 3; ++c)
      if (uAdv_[c].extent(0) != n_)
        uAdv_[c] = CCField("uAdv", n_);
  }
  // The other drivers that live on THIS block and are initialised lazily: their own `ready()` /
  // `initialized()` predicates ask "has init run", not "is it the right size", so a
  // re-decomposition has to re-init them here or they keep the previous block's allocations
  // while every kernel indexes them with the new `e3_`.
  const std::size_t blockLen = (std::size_t)e3_.x * e3_.y * e3_.z;
  if (pcAreaC_.ready() && pcAreaC_.area().extent(0) != blockLen)
    pcAreaC_.init(nx_, ny_, nz_, kVofG);
  if (pcAreaMc_.ready() && pcAreaMc_.area().extent(0) != blockLen)
    pcAreaMc_.init(nx_, ny_, nz_, kVofG);
  if (vofEnergy_.initialized() && vofEnergy_.temperature().extent(0) != blockLen)
    vofEnergy_.init(vofAdv_, pcRcpG_, pcRcpL_);
  bindVofBlockPatch();  // rung W0: the block exchange's Views were just reallocated
}

template <class Grid>
int Solver<Grid>::vofWetWallMask() const {
  if (!vofEnabled_ || !contactAngleSet_)
    return 0;
  int m = 0;
  for (int f = 0; f < 6; ++f)
    if (bc_[f] == 1 || bc_[f] == 4)
      m |= (1 << f);
  return m;
}

template <class Grid>
KOKKOS_INLINE_FUNCTION double Solver<Grid>::vofWallPlaneSdf(int gx, int gy, int gz, I3 gs,
                                                            int mask) {
  const int gi[3] = {gx, gy, gz};
  const int q[3] = {gs.x, gs.y, gs.z};
  double d = 1e30;
  for (int a = 0; a < 3; ++a) {
    const double c = (double)gi[a] + 0.5;
    if (mask & (1 << (2 * a)))
      d = Kokkos::fmin(d, c);
    if (mask & (1 << (2 * a + 1)))
      d = Kokkos::fmin(d, (double)q[a] - c);
  }
  return d;
}

template <class Grid>
void Solver<Grid>::buildVofGeometry() {
  if (!vofEnabled_)
    return;
  const int wet = vofWetWallMask();
  if ((!hasSolid_ || !cutcellPressure_) && !wet) {
    vofAdv_.disableGeometry();  // all-fluid: the V1 kernels run byte-identically
    vofAdv_.disableWetting();
    vofSolidG2_ = CCField();
    return;
  }
  if (wet && !cutcellPressure_)
    throw std::runtime_error(
        "set_contact_angle: a wetting DOMAIN wall (bc type 1/4) needs a cut-cell pressure "
        "operator, because the colour transport weights its geometric fluxes with the face "
        "openness. Build one with set_pressure_geometry(all_fluid_sdf) (all-fluid) or "
        "set_solid(sdf, cutcell_pressure=True).");
  vofAdv_.enableGeometry();
  if (hasSolid_ && cutcellPressure_) {
    if (vofCs_.extent(0) != n_)
      vofCs_ = CCField("vofCs", n_);
    buildCellFraction(vofCs_, CCConst(sdf_), e_,
                      G);  // inner region; ghosts come from the exchange
    copyInner(vofAdv_.epsFraction(), e3_, kVofG, CCConst(vofCs_), e_, G);
    vofExchangeRaw(vofAdv_.epsFraction());
    CCField oa[3] = {ox_, oy_, oz_};
    for (int d = 0; d < 3; ++d) {
      vof::copyFaceVelocity(vofAdv_.faceOpenness(d), I3{e3_.x, e3_.y, e3_.z}, kVofG, oa[d],
                            I3{e_.x, e_.y, e_.z}, G, d);
      vofExchangeRaw(vofAdv_.faceOpenness(d));
    }
  } else {
    // Domain walls only: the trivial geometry (every cell whole, every face open) is exactly
    // what the uncut kernels compute, and `applyDomainWallGeometry` then closes the band.
    Kokkos::deep_copy(vofAdv_.epsFraction(), 1.0);
    for (int d = 0; d < 3; ++d)
      Kokkos::deep_copy(vofAdv_.faceOpenness(d), 1.0);
  }
  if (wet)
    applyDomainWallGeometry(wet);
  vofAdv_.classifyGeometry();
  vofExchangeRaw(vofAdv_.kindDouble());  // the owner's classification into every ghost layer
  // ISSUES sweep item 3: that exchange ends in `clampFill`, whose zero-gradient copy would set
  // the out-of-domain band back to the first INNER cell's classification (fluid) -- the band
  // is the one region with no owner, so it has to be re-imposed here. Without this the whole
  // domain-wall fill is silently inert (measured: contact census 0, equilibrium ~141 deg at
  // every prescribed angle, i.e. the perfectly non-wetting empty band).
  if (wet)
    imposeDomainWallKind(wet);
  vofAdv_.finalizeGeometry();
  // The G=2 mirror of the classification: the canonical "C" field reports EXACTLY 0 in solid
  // cells (gate G2), while the g=3 working block carries the neutral band fill that the MYC and
  // height-function stencils need. The fill is regenerated deterministically by every
  // `vofFillGhosts`, so nothing is lost by not persisting it.
  if (vofSolidG2_.extent(0) != n_)
    vofSolidG2_ = CCField("vofSolidG2", n_);
  copyInner(vofSolidG2_, e_, G, CCConst(vofAdv_.kindDouble()), e3_, kVofG);
  applyContactAngle();  // rung V5b (WO-S): re-wire the theta field / wall SDF onto the new block
  zeroSolidColour();
}

template <class Grid>
void Solver<Grid>::applyDomainWallGeometry(int mask) {
  const I3 gs = vofGlobalSize(), org = vofOrigin();
  const C3 e = e3_;
  const int g = kVofG, mk = mask;
  CCField ep = vofAdv_.epsFraction();
  CCField of[3] = {vofAdv_.faceOpenness(0), vofAdv_.faceOpenness(1), vofAdv_.faceOpenness(2)};
  CCField ofx = of[0], ofy = of[1], ofz = of[2];
  Kokkos::parallel_for(
      "peclet::flow::vof_domain_wall_geom",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const int gx = x - g + org.x, gy = y - g + org.y, gz = z - g + org.z;
        auto inside = [&](int ax, int ay, int az) {
          const int gi[3] = {ax, ay, az};
          const int q[3] = {gs.x, gs.y, gs.z};
          for (int a = 0; a < 3; ++a) {
            if ((mk & (1 << (2 * a))) && gi[a] < 0)
              return false;
            if ((mk & (1 << (2 * a + 1))) && gi[a] >= q[a])
              return false;
          }
          return true;
        };
        const bool in0 = inside(gx, gy, gz);
        if (!in0)
          ep(i) = 0.0;
        // the advector's HIGH-face convention: of[d](i) is the +d face of cell i
        if (!in0 || !inside(gx + 1, gy, gz))
          ofx(i) = 0.0;
        if (!in0 || !inside(gx, gy + 1, gz))
          ofy(i) = 0.0;
        if (!in0 || !inside(gx, gy, gz + 1))
          ofz(i) = 0.0;
      });
  CCExec().fence();
}

template <class Grid>
void Solver<Grid>::imposeDomainWallKind(int mask) {
  const I3 gs = vofGlobalSize(), org = vofOrigin();
  const C3 e = e3_;
  const int g = kVofG, mk = mask;
  CCField kd = vofAdv_.kindDouble();
  Kokkos::parallel_for(
      "peclet::flow::vof_domain_wall_kind",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const int gi[3] = {x - g + org.x, y - g + org.y, z - g + org.z};
        const int q[3] = {gs.x, gs.y, gs.z};
        for (int a = 0; a < 3; ++a) {
          if (((mk & (1 << (2 * a))) && gi[a] < 0) ||
              ((mk & (1 << (2 * a + 1))) && gi[a] >= q[a])) {
            kd(i) = 1.0;
            return;
          }
        }
      });
  CCExec().fence();
}

template <class Grid>
void Solver<Grid>::vofExtendWallBand(CCField f, int mask) {
  const I3 gs = vofGlobalSize(), org = vofOrigin();
  const C3 e = e3_;
  const int g = kVofG;
  for (int a = 0; a < 3; ++a)
    for (int sd = 0; sd < 2; ++sd) {
      if (!(mask & (1 << (2 * a + sd))))
        continue;
      const int q = (a == 0) ? gs.x : (a == 1) ? gs.y : gs.z;
      const int o = (a == 0) ? org.x : (a == 1) ? org.y : org.z;
      // block-local index of the depth-3 and depth-2 layers on this face
      const int i3 = (sd == 0) ? (-3 - o + g) : (q - o + g + 2);
      const int i2 = (sd == 0) ? (-2 - o + g) : (q - o + g + 1);
      const int dims[3] = {e.x, e.y, e.z};
      if (i3 < 0 || i3 >= dims[a] || i2 < 0 || i2 >= dims[a])
        continue;  // this rank's block does not own that face
      const long st[3] = {1, (long)e.x, (long)e.x * (long)e.y};
      const int b = (a + 1) % 3, c = (a + 2) % 3;
      const long sa = st[a], sb = st[b], sc = st[c];
      CCField ff = f;
      Kokkos::parallel_for(
          "peclet::flow::vof_extend_wall_band",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(CCExec(), {0, 0}, {dims[b], dims[c]}),
          KOKKOS_LAMBDA(int p0, int p1) {
            const long base = (long)p0 * sb + (long)p1 * sc;
            ff(base + (long)i3 * sa) = ff(base + (long)i2 * sa);
          });
    }
  CCExec().fence();
}

template <class Grid>
void Solver<Grid>::zeroSolidColour() {
  if (!vofEnabled_ || !vofAdv_.hasGeometry() || !vofSolidG2_.extent(0) || !vofSolidZero_)
    return;
  // `vofSolidG2_` lives on the EXTENDED G=2 block (copyInner wrote it at (x+G, y+G, z+G)), so it
  // is indexed exactly like cField_ — indexing it as an inner-sized array reads the wrong cells
  // and silently zeroes live fluid colour (measured: 0.5 % of the liquid volume lost per step).
  CCField c = cField_;
  CCConst sl = CCConst(vofSolidG2_);
  const int ex = e_.x, ey = e_.y, g = G;
  Kokkos::parallel_for(
      "peclet::flow::vof_zero_solid",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {nx_, ny_, nz_}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey;
        if (sl(i) > 0.5)
          c(i) = 0.0;
      });
  CCExec().fence();
}

template <class Grid>
bool Solver<Grid>::vofAxisPeriodic(int a) const {
  return bc_[2 * a] == 0 && bc_[2 * a + 1] == 0;
}

template <class Grid>
void Solver<Grid>::vofFillGhosts(CCField f) {
  vofExchangeRaw(f);
  // Rung V5a (WO-Q): the neutral (90 deg) solid-band fill, for the COLOUR field only. It is a
  // stencil device, not transported data — the MYC 3^3 stencil and the V3 height-function
  // columns of a near-wall cell reach into the solid, and leaving those cells at 0 makes every
  // wall perfectly non-wetting. Three passes with a shrinking depth budget (`vof/cutcell.hpp`),
  // then a SECOND exchange so the outermost ghost layer holds its owner's filled value (the
  // passes only reach ghost depth 2, and the curvature cascade reads depth 3).
  if (vofAdv_.hasGeometry() && f.data() == vofAdv_.colour().data()) {
    // Rung V5b (WO-S): the theta-consistent pass 1 reads the FLUID-ONLY Youngs normal of the
    // anchor fluid cell, which the pass may reach at ghost depth 3 — one deeper than a 3^3
    // stencil can be evaluated on this block. Build it on the INNER region and run it through
    // the block's own ghost policy, exactly as the geometry classification is: every read the
    // theta pass then makes is the OWNER's value, which is what keeps the inner fill
    // decomposition-independent.
    if (vofAdv_.hasWetting()) {
      vofAdv_.buildWettingNormals();
      for (int d = 0; d < 3; ++d)
        vofExchangeRaw(vofAdv_.wettingNormal(d));
      // Rung V6 (WO-V6): produce the theta FIELD the pass is about to read. Pass A measures the
      // apparent angle and the raw contact-line speed at ghost depth <= 2 from already-exchanged
      // data; both are then exchanged so the 3-point in-wall smoothing of pass B, which reaches
      // depth 3, reads the OWNER's values — the same argument that makes the V5b fill bitwise
      // across np. Five extra exchanges per fill (three velocity components + two measurements),
      // skipped entirely when no dynamic angle is configured.
      if (vofDyn_.active() && vofDyn_.allocated()) {
        buildVofCellVelocity();
        vofDyn_.sigma = effectiveContactSigma();
        if (!(vofDyn_.sigma > 0.0) && vofDyn_.dynamic && !contactSigmaWarned_) {
          contactSigmaWarned_ = true;
          std::fprintf(stderr,
                       "peclet.flow: set_contact_angle_dynamic is active but sigma is 0 "
                       "(no set_surface_tension and no explicit sigma) - Ca_cl is 0 and the "
                       "Cox-Voinov correction is inert.\n");
        }
        vofDyn_.measure(vofAdv_);
        vofExchangeRaw(vofDyn_.uclRaw());
        vofExchangeRaw(vofDyn_.valid());
        vofDyn_.impose(vofAdv_);
      }
    }
    vofAdv_.solidBandFill();
    if (const int wet = vofWetWallMask())
      vofExtendWallBand(f, wet);  // ISSUES sweep item 3: the depth-3 layer no pass can write
    vofExchangeRaw(f);
  }
}

template <class Grid>
void Solver<Grid>::vofExchangeRaw(CCField f) {
  const bool px = vofAxisPeriodic(0), py = vofAxisPeriodic(1), pz = vofAxisPeriodic(2);
#ifdef PECLET_FLOW_MPI
  if (distributed_ && vofDev_)
    vofDev_->exchange(f);
  else
#endif
    vof::periodicFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, px, py, pz);
  if (px && py && pz)
    return;
  const I3 gs = vofGlobalSize(), org = vofOrigin();
  // ISSUES sweep item 3: the theta band of a wetting DOMAIN wall lives in exactly the ghost
  // cells `clampFill` would overwrite with the zero-gradient copy, so the colour field skips
  // the clamp there and `vofFillGhosts` owns that band instead. 0 for every other field and
  // for every configuration without a domain-wall contact angle -> byte-identical.
  const int skip = (f.data() == vofAdv_.colour().data()) ? vofWetWallMask() : 0;
  vof::clampFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, org, gs, px, py, pz, skip);
  vofApplyColourBc(f);  // WO-R: inflow / inletOutlet backflow; a no-op unless one is set
}

template <class Grid>
I3 Solver<Grid>::vofGlobalSize() const {
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    return I3{gnx_, gny_, gnz_};
#endif
  return I3{nx_, ny_, nz_};
}

template <class Grid>
I3 Solver<Grid>::vofOrigin() const {
  return I3{og_.x, og_.y, og_.z};
}

template <class Grid>
void Solver<Grid>::bridgeVelocityToVof() {
  // Rung V8 (WO-T): on the collocated grid the divergence-free field is the PROJECTED MAC face
  // field uf_/vf_/wf_, not the cell field — and it is in flow's own low-face convention
  // (`uf(i) = 1/2(U(i)+U(i-1))` sits at i-1/2, the -x face of cell i), the SAME convention
  // `getFaceVelocity` reports and `copyFaceVelocity` shifts. So the bridge is the identical call
  // on a different source view; the cell field never enters the colour transport.
  if constexpr (Grid::collocated) {
    CCField fa[3] = {uf_, vf_, wf_};
    for (int c = 0; c < 3; ++c) {
      fillGhosts(fa[c]);  // the face field's own ghost policy (project() does exactly this)
      vof::copyFaceVelocity(vofAdv_.faceVel(c), I3{e3_.x, e3_.y, e3_.z}, kVofG, fa[c],
                            I3{e_.x, e_.y, e_.z}, G, c);
    }
    return;
  }
  for (int c = 0; c < 3; ++c) {
    // KEEP the projection's outflow-face correction when there IS one (see
    // fillVelGhostsKeepOutflow): the full fill would overwrite it with the zero-gradient copy
    // and hand the advector a field that is not divergence-free at the outlet. When no
    // projection has run since the last full fill — the KINEMATIC path, where the caller
    // prescribes the velocity on the inner cells and the boundary face has never been set —
    // the zero-gradient fill is exactly what supplies that face, so run the full one.
    // With no outflow face at all the two are identical.
    if (outflowCorrValid_)
      fillVelGhostsTo(C[c].u, c, 0, false);
    else
      fillVelGhosts(c, 0);
    // The uniform face-velocity seam (getFaceVelocity): staggered C[c].u already lives on the
    // faces. copyFaceVelocity carries the low-face -> high-face index shift; see
    // colour_field.hpp.
    vof::copyFaceVelocity(vofAdv_.faceVel(c), I3{e3_.x, e3_.y, e3_.z}, kVofG, C[c].u,
                          I3{e_.x, e_.y, e_.z}, G, c);
  }
}

template <class Grid>
std::vector<vof::VofBox> Solver<Grid>::vofBlockRankBoxes(int size) const {
  std::vector<vof::VofBox> rb(static_cast<std::size_t>(size));
  const I3 gs = vofGlobalSize();
#ifdef PECLET_FLOW_MPI
  if (distributed_ && dec_) {
    for (int r = 0; r < size; ++r) {
      const auto b = dec_->block(static_cast<std::size_t>(r));
      for (int d = 0; d < 3; ++d) {
        rb[r].lo[d] = static_cast<int>(b.origin[d]);
        rb[r].hi[d] = static_cast<int>(b.origin[d] + b.size[d]);
      }
    }
    return rb;
  }
#endif
  rb[0].hi[0] = gs.x;
  rb[0].hi[1] = gs.y;
  rb[0].hi[2] = gs.z;
  return rb;
}

template <class Grid>
void Solver<Grid>::bindVofBlockPatch() {
  if (!vofBlockExch_)
    return;
  // The rank table first, for the same reason: `buildVofBlock` runs on every re-decomposition
  // (redistribute -> initMpi), and the pieces are meaningless against the old boxes.
  int size = 1;
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    MPI_Comm_size(comm_, &size);
#endif
  vofBlockExch_->setRankBoxes(vofBlockRankBoxes(size));
  vof::VofBlockExchange::Patch pp;
  pp.e = I3{e3_.x, e3_.y, e3_.z};
  pp.n = I3{nx_, ny_, nz_};
  pp.o = vofOrigin();
  pp.g = kVofG;
  vofBlockExch_->setPatch(pp, vofAdv_.faceVel(0), vofAdv_.faceVel(1), vofAdv_.faceVel(2));
}

template <class Grid>
void Solver<Grid>::harvestVofBlockUnion() {
  copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
  fillPropGhosts(cField_);
}

template <class Grid>
void Solver<Grid>::bridgeColourToVof() {
  copyInner(vofAdv_.colour(), e3_, kVofG, CCConst(cField_), e_, G);
  vofFillGhosts(vofAdv_.colour());
}

template <class Grid>
void Solver<Grid>::addCsfRhs(int c) {
  CCExec space;
  C3 e = e_;
  CCField bb = C[c].b;
  CCConst rs = CCConst(C[c].rscale), cv = CCConst(cField_), kp = CCConst(kappaField_),
          kb = CCConst(kappaBranch_);
  const long strd = strideOf(c);
  // Phase 3 (V3.1): the CSF carries the PRESSURE-GRADIENT WEIGHT of this component's axis,
  // `w_a = 1/h_a'^2` — the same symbol Phase 2 puts on `-(P(i) - P(i - s_a))`. Exactly 1.0 on
  // every isotropic run, so `x / 1.0 == x` keeps this kernel bit-identical.
  const double sig = sigmaCsf_, h = 1.0 / u_.w[c];
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "csf_rhs", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double dC = cv(i) - cv(i - strd);
        if (dC == 0.0)
          return;  // no interface across this face -> no force, and no orphan either
        double kf = 0.0;
        vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
        bb(i) += rs(i) * vof::csfFaceForce(sig, kf, dC, h);
      });
}

template <class Grid>
void Solver<Grid>::addCsfRhsCellInterp(int c) {
  CCExec space;
  C3 e = e_;
  CCField bb = C[c].b;
  CCConst rs = CCConst(C[c].rscale), cv = CCConst(cField_), kp = CCConst(kappaField_),
          kb = CCConst(kappaBranch_);
  const long strd = strideOf(c);
  const double sig = sigmaCsf_, h = 1.0 / u_.w[c];  // the ablation takes the same weight (V3.1)
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "csf_rhs_cellinterp", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double f[2] = {0.0, 0.0};
        for (int q = 0; q < 2; ++q) {  // q = 0 -> cell i-s_c, q = 1 -> cell i
          const long j = i - (1 - q) * strd;
          if (!vof::csfKappaDefined(kb(j)))
            continue;
          f[q] = sig * kp(j) * 0.5 * (cv(j + strd) - cv(j - strd)) / h;
        }
        bb(i) += rs(i) * 0.5 * (f[0] + f[1]);
      });
}

template <class Grid>
void Solver<Grid>::addCsfRhsBlocks(int c) {
  CCExec space;
  C3 e = e_;
  CCField bb = C[c].b;
  CCConst rs = CCConst(C[c].rscale), fb = CCConst(csfBlkF_[c]);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "csf_rhs_blocks", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        bb(i) += rs(i) * fb(i);
      });
}

template <class Grid>
typename Solver<Grid>::CsfDiagnostics Solver<Grid>::csfDiagnostics() {
  if (!vofEnabled_ || !kappaField_.extent(0))
    throw std::runtime_error(
        "csf_diagnostics: needs VoF + a curvature field (call set_surface_tension and step, or "
        "compute_vof_curvature)");
  CsfDiagnostics d;
  C3 e = e_;
  CCConst cv = CCConst(cField_), kp = CCConst(kappaField_), kb = CCConst(kappaBranch_);
  const double sig = sigmaCsf_, eps = csfInterfaceEps_;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  for (int c = 0; c < 3; ++c) {
    const double h = 1.0 / u_.w[c];  // the V3.1 per-axis gradient weight of this component
    const long strd = strideOf(c);
    double mx = 0.0;
    long orph = 0, forced = 0;
    Kokkos::parallel_reduce(
        "csf_diag", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& m, long& o, long& f) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double dC = cv(i) - cv(i - strd);
          if (dC == 0.0)
            return;
          double kf = 0.0;
          const bool ok = vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
          if (ok)
            ++f;
          else if (Kokkos::fabs(dC) > eps)
            ++o;
          m = Kokkos::fmax(m, Kokkos::fabs(vof::csfFaceForce(sig, kf, dC, h)));
        },
        Kokkos::Max<double>(mx), orph, forced);
    d.maxForce[c] = mx;
    d.orphanFaces[c] = orph;
    d.forcedFaces[c] = forced;
  }
  return d;
}

template <class Grid>
void Solver<Grid>::enableVof() {
  // PHASE 3 LIFTED Phase 2's `requireIsotropic` refusal here (flow/doc/anisotropic_vof.md).
  // The colour transport itself needed nothing — a stretched cell IS the unit cube of the index
  // coordinates, so every volume fraction, PLIC plane<->volume relation, slab flux and
  // Weymouth-Yue sweep is metric-free (§3, decision V1). What took the metric is everything
  // that reads a DIRECTION, a LENGTH or an AREA: the height-function curvature over physical
  // column and transverse spacings, the two paraboloid fits in the physical frame (§4), the CSF
  // face force carrying THIS axis's pressure-gradient weight `w_a` so the balanced-force
  // identity survives (§5), the theta rotation of the wetting fill (§6), and the phase-change
  // layer's normals, areas, sample distances and `V_cell` (§7).

  if constexpr (Grid::collocated) {
    // Rung V8 (WO-T): allowed. The colour is advected by the PROJECTED face field uf_/vf_/wf_ —
    // which is what the ABC approximate projection makes exactly divergence-free, i.e. precisely
    // the field Weymouth-Yue's conservation proof needs — and every interfacial force is a face
    // acceleration (collocated_varrho.hpp). ALL-FLUID only at this rung.
    if (hasSolid_)
      throw std::runtime_error(
          "enable_vof: geometric VoF on SolverColocated (rung V8) is ALL-FLUID only — an immersed "
          "solid needs the cut-cell face acceleration and the matching one-sided closures, which "
          "is a later rung. Use the staggered Solver (rung V5a supports cut cells).");
    collocatedV8AutoFallback("geometric VoF on the collocated grid");
  }
  if (vofEnabled_)
    return;
  cField_ = addField("C");  // the G=2 registry mirror (closure input / IO / redistribute)
  vofEnabled_ = true;
  // WO-R2 item 3 — the EXACT (matrix-free, double, flux-form) level-0 residual/matvec becomes
  // the default the moment there is an interface. Measured on Hysing case 2 (WO-R item 6,
  // 64x128x4, adaptive dt): it removes 7.5 orders of the projected flux divergence
  // (1.85e-03 -> 5.15e-11) and moves NOTHING else — 116/600 pressure iterations, 1123 steps, the
  // dt-limit census and both published functionals identical to every printed digit. The defect
  // it removes is the float operator's broken row-sum identity A*1 = 0, and a two-phase
  // coefficient contrast is precisely what amplifies it. `set_pressure_exact_residual(False)`
  // after `enable_vof` is the ablation.
  setPressureExactResidual(true);
  // WO-R2 item 4 — the wisp tolerance. 0 is V1 verbatim (and stays the standalone advector's
  // default); an open-boundary domain that DRAINS reaches C ~ 1e-18 everywhere, where the MYC
  // normal is degenerate and plicAlpha divides by it (measured: sum C -> -inf -> NaN in three
  // steps). Same 1e-8 the curvature predicate uses under surface tension.
  vofAdv_.wispEps = vofWispEps_;
  buildVofBlock();
}

template <class Grid>
void Solver<Grid>::setVofWispEps(double eps) {
  if (!(eps >= 0.0) || eps >= 0.5)
    throw std::runtime_error("set_vof_wisp_eps: the threshold must be in [0, 0.5)");
  vofWispEps_ = eps;
  vofAdv_.wispEps = eps;
}

template <class Grid>
double Solver<Grid>::vofWispEps() const {
  return vofWispEps_;
}

template <class Grid>
constexpr double Solver<Grid>::defaultVofWispEps() {
  return 1e-8;
}

template <class Grid>
bool Solver<Grid>::vofEnabled() const {
  return vofEnabled_;
}

template <class Grid>
void Solver<Grid>::setVof(const std::vector<double>& c) {
  enableVof();
  scatterInner(cField_, c);
  zeroSolidColour();  // rung V5a: solid cells carry no colour (the band fill is regenerated)
  fillPropGhosts(cField_);
  pcMaskFresh_ = false;  // WO-P3g: the Dirichlet/plane geometry no longer describes this colour
}

template <class Grid>
std::vector<double> Solver<Grid>::getVof() {
  return gatherInner(cField_);
}

template <class Grid>
vof::WyAdvector::Diagnostics Solver<Grid>::vofDiagnostics() {
  if (!vofEnabled_)
    throw std::runtime_error("vof_diagnostics: VoF is not enabled (call enable_vof/set_vof)");
  bridgeColourToVof();
  auto d = vofAdv_.diagnostics();
  // `solidSumC` is the census of the CANONICAL field: the working block's solid cells carry the
  // neutral band fill (reported as `solidFillSum`), while "C" itself is 0 there — that is the
  // quantity gate G2 of WO-Q asks for.
  d.solidSumC = vofSolidColourSum();
  return d;
}

template <class Grid>
double Solver<Grid>::vofSolidColourSum() {
  if (!vofEnabled_ || !vofAdv_.hasGeometry() || !vofSolidG2_.extent(0))
    return 0.0;
  CCConst c = CCConst(cField_), sl = CCConst(vofSolidG2_);
  const int ex = e_.x, ey = e_.y, g = G;
  double acc = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::vof_solid_sum",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {nx_, ny_, nz_}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a) {
        const long i = (long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey;
        if (sl(i) > 0.5)
          a += c(i);
      },
      acc);
  Kokkos::fence();
  return acc;
}

template <class Grid>
double Solver<Grid>::vofMaxCourant() {
  if (!vofEnabled_)
    return 0.0;
  bridgeVelocityToVof();
  bridgeColourToVof();
  const double loc = vofAdv_.maxCourantInterfaceAuto(dt_ / vofAdv_.h());
  return vofAdv_.globalMax ? vofAdv_.globalMax(loc) : loc;
}

template <class Grid>
double Solver<Grid>::vofLastCourant() const {
  return vofEnabled_ ? vofAdv_.lastCfl() : 0.0;
}

template <class Grid>
void Solver<Grid>::setVofCflLimit(double v) {
  vofCflLimit_ = v;
  if (vofEnabled_)
    vofAdv_.cflLimit = v;
}

template <class Grid>
double Solver<Grid>::vofCflLimit() const {
  return vofCflLimit_;
}

template <class Grid>
void Solver<Grid>::requireVofGeometry(const char* who) {
  if constexpr (Grid::collocated) {
    // Rung V8 scope, re-checked here because `set_solid` can follow `enable_vof`.
    if (hasSolid_)
      throw std::runtime_error(
          std::string(who) +
          ": geometric VoF on SolverColocated (rung V8) is ALL-FLUID only. The cut-cell colour "
          "transport (rung V5a) is validated on the STAGGERED solver.");
  }
  if (!hasSolid_ || vofAdv_.hasGeometry())
    return;
  std::string m(who);
  m += ": an immersed solid is present but the colour block has no cut-cell geometry. Rung V5a "
       "needs the cut-cell openness, i.e. set_solid(sdf, cutcell_pressure=True) (the staircase "
       "pressure operator has no face openness to weight the geometric fluxes with).";
  throw std::runtime_error(m);
}

template <class Grid>
bool Solver<Grid>::vofHasGeometry() const {
  return vofAdv_.hasGeometry();
}

template <class Grid>
void Solver<Grid>::setVofCutFluxClamp(bool on) {
  vofAdv_.cutFluxClamp = on;
}

template <class Grid>
bool Solver<Grid>::vofCutFluxClamp() const {
  return vofAdv_.cutFluxClamp;
}

template <class Grid>
void Solver<Grid>::setVofSolidColourZero(bool on) {
  vofSolidZero_ = on;
}

template <class Grid>
bool Solver<Grid>::vofSolidColourZero() const {
  return vofSolidZero_;
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofFilledColour() {
  if (!vofEnabled_)
    throw std::runtime_error("vof_filled_color: VoF is not enabled");
  bridgeColourToVof();
  CCField t("vofFilled", n_);
  copyInner(t, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
  return gatherInner(t);
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofGeometry(int which) {
  if (!vofEnabled_)
    throw std::runtime_error("vof_geometry: VoF is not enabled (needs enable_vof)");
  if (which < 0 || which > 4)
    throw std::runtime_error("vof_geometry: `which` must be 0..4");
  // ISSUES sweep item 4. An ALL-FLUID VoF solver has no cut-cell geometry block, but the
  // geometry it would carry is not undefined -- it is the trivial one, and it is exactly what
  // the V1 transport kernels execute: every cell fully fluid (eps = 1), every face fully open
  // (o = 1), no solid (kind = 0). Returning it lets ONE diagnostic --
  // `vof_geometry(0) * (1 - get_vof())` for the gas volume, say -- serve a packed scene and its
  // all-fluid control, instead of forcing the caller to branch on `vof_has_geometry()` and
  // synthesise the ones itself (which is what examples/bubble-through-packing had to do).
  if (!vofAdv_.hasGeometry())
    return std::vector<double>((std::size_t)nx_ * ny_ * nz_, which == 4 ? 0.0 : 1.0);
  CCField t("vofGeom", n_);
  CCConst src;
  if (which == 0)
    src = CCConst(vofAdv_.epsFraction());
  else if (which < 4)
    src = CCConst(vofAdv_.faceOpenness(which - 1));
  else
    src = CCConst(vofAdv_.kindDouble());
  copyInner(t, e_, G, src, e3_, kVofG);
  return gatherInner(t);
}

template <class Grid>
void Solver<Grid>::setContactAngle(double thetaDeg) {
  if (!(thetaDeg >= 0.0 && thetaDeg <= 180.0))
    throw std::runtime_error("set_contact_angle: theta must be in [0, 180] degrees");
  contactAngleDeg_ = thetaDeg;
  contactAngleField_.clear();
  contactAngleSet_ = true;
  requireWettingWall();
  buildVofGeometry();  // item 3: a wetting DOMAIN wall changes the colour block's geometry
  applyContactAngle();
}

template <class Grid>
void Solver<Grid>::requireWettingWall() const {
  if (!vofEnabled_ || hasSolid_ || vofWetWallMask() != 0)
    return;
  throw std::runtime_error(
      "set_contact_angle: this solver has no wetting wall - no immersed SDF solid (set_solid) "
      "and no impermeable domain face (set_domain_bc(face, 1) no-slip or 4 free-slip). The "
      "angle would be imposed nowhere. (Inflow/outflow faces carry no contact line.)");
}

template <class Grid>
void Solver<Grid>::setContactAngleField(const std::vector<double>& thetaDeg) {
  if (thetaDeg.size() != (std::size_t)nx_ * ny_ * nz_)
    throw std::runtime_error("set_contact_angle_field: expected nx*ny*nz values");
  contactAngleField_ = thetaDeg;
  contactAngleSet_ = true;
  requireWettingWall();
  buildVofGeometry();  // item 3
  applyContactAngle();
}

template <class Grid>
bool Solver<Grid>::contactAngleSet() const {
  return contactAngleSet_;
}

template <class Grid>
double Solver<Grid>::contactAngle() const {
  return contactAngleDeg_;
}

template <class Grid>
void Solver<Grid>::setContactAnglePivot(int mode) {
  if (mode < 0 || mode > 3)
    throw std::runtime_error("set_contact_angle_pivot: mode must be 0..3");
  contactPivot_ = mode;
  vofAdv_.wettingPivot = mode;
}

template <class Grid>
int Solver<Grid>::contactAnglePivot() const {
  return contactPivot_;
}

template <class Grid>
void Solver<Grid>::setContactAngleDynamic(double thetaEDeg, double slipCells, double muLiquid,
                                          double sigma) {
  // `slipCells` is a LENGTH in the caller's units (cells when no physical domain is armed);
  // the model needs lambda/Delta, which is what the internal value is.
  const double slipInt = slipCells * u_.lenToInt();
  if (!(slipInt > 0.0 && slipInt < 1.0))
    throw std::runtime_error(
        "set_contact_angle_dynamic: the slip length must lie in (0, 1) cell (lambda < Delta)");
  if (!(muLiquid > 0.0))
    throw std::runtime_error("set_contact_angle_dynamic: mu_liquid must be positive");
  vofDyn_.dynamic = true;
  vofDyn_.slip = slipInt;
  vofDyn_.muLiquid = muLiquid * u_.muToInt();
  contactSigmaOverride_ = sigma;
  // ONE lambda across the two halves of Afkhami-Zaleski-Bussmann (WO-V6b): this call and
  // set_wall_slip_length write the SAME stored value, so the angle model's inner cut-off and
  // the momentum wall closure can never disagree (last call wins). Whether the MOMENTUM half is
  // active is a separate switch, set only by set_wall_slip_length -- so every result WO-V6
  // validated with the angle half alone stays exactly what it was.
  slipPhys_ = slipCells;
  slipLambda_ = slipInt;
  if (wallSlip_ && hasSolid_ && !Grid::collocated) {
    buildVelocityOverlays(/*resetU=*/false);
    rebuildStencils();
  }
  setContactAngle(thetaEDeg);  // sets the static base and (re)wires the theta field
}

template <class Grid>
void Solver<Grid>::setWallSlipLength(double lambdaCells) {
  if (!(lambdaCells >= 0.0))
    throw std::runtime_error("set_wall_slip_length: lambda must be >= 0");
  const double lambdaInt = lambdaCells * u_.lenToInt();
  if (lambdaCells > 0.0 && Grid::collocated)
    throw std::runtime_error(
        "set_wall_slip_length: the Navier wall closure is a staggered-grid feature (the "
        "collocated scheme carries its own wall treatment)");
  slipPhys_ = lambdaCells;
  slipLambda_ = lambdaInt;
  wallSlip_ = lambdaCells > 0.0;
  if (vofDyn_.dynamic && lambdaCells > 0.0)
    vofDyn_.slip = lambdaInt;
  if (hasSolid_) {
    buildVelocityOverlays(/*resetU=*/false);
    rebuildStencils();
  }
}

template <class Grid>
double Solver<Grid>::wallSlipLength() const {
  return wallSlip_ ? slipPhys_ : 0.0;
}

template <class Grid>
std::array<int, 3> Solver<Grid>::wallSlipSandwichCells() const {
  return slipSandwich_;
}

template <class Grid>
void Solver<Grid>::setContactAngleHysteresis(double thetaADeg, double thetaRDeg) {
  if (!(thetaADeg >= thetaRDeg))
    throw std::runtime_error("set_contact_angle_hysteresis: theta_a must be >= theta_r");
  if (!(thetaRDeg >= 0.0 && thetaADeg <= 180.0))
    throw std::runtime_error("set_contact_angle_hysteresis: angles must lie in [0, 180]");
  const double toRad = 3.14159265358979323846 / 180.0;
  vofDyn_.hysteresis = true;
  vofDyn_.thetaA = thetaADeg * toRad;
  vofDyn_.thetaR = thetaRDeg * toRad;
  if (!contactAngleSet_)
    setContactAngle(0.5 * (thetaADeg + thetaRDeg));  // a base is required; the mid angle
  else
    applyContactAngle();
}

template <class Grid>
void Solver<Grid>::setContactAngleDynamicOff() {
  vofDyn_.dynamic = false;
  vofDyn_.hysteresis = false;
  applyContactAngle();
}

template <class Grid>
bool Solver<Grid>::contactAngleDynamic() const {
  return vofDyn_.dynamic;
}

template <class Grid>
bool Solver<Grid>::contactAngleHysteresis() const {
  return vofDyn_.hysteresis;
}

template <class Grid>
double Solver<Grid>::contactAngleSlip() const {
  return vofDyn_.slip;
}

template <class Grid>
void Solver<Grid>::setContactAngleSmoothing(bool on) {
  vofDyn_.smooth = on;
}

template <class Grid>
void Solver<Grid>::setContactAngleClamp(double loDeg, double hiDeg) {
  if (!(loDeg > 0.0 && hiDeg < 180.0 && loDeg < hiDeg))
    throw std::runtime_error("set_contact_angle_clamp: need 0 < lo < hi < 180");
  const double toRad = 3.14159265358979323846 / 180.0;
  vofDyn_.thetaMin = loDeg * toRad;
  vofDyn_.thetaMax = hiDeg * toRad;
}

template <class Grid>
double Solver<Grid>::effectiveContactSigma() const {
  return contactSigmaOverride_ > 0.0 ? contactSigmaOverride_ : sigmaCsf_;
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofDynamicField(int which) {
  if (!vofEnabled_ || !vofDyn_.active() || !vofDyn_.allocated())
    throw std::runtime_error(
        "vof_dynamic_field: no dynamic contact angle (call set_contact_angle_dynamic / "
        "set_contact_angle_hysteresis first)");
  bridgeColourToVof();  // regenerates the fill, hence the dynamic pass
  CCField t("vofDyn", n_);
  const double toDeg = 180.0 / 3.14159265358979323846;
  CCConst src;
  if (which == 0)
    src = CCConst(vofDyn_.imposed());
  else if (which == 1)
    src = CCConst(vofDyn_.apparent());
  else if (which == 2)
    src = CCConst(vofDyn_.speed());
  else if (which == 3)
    src = CCConst(vofDyn_.capillary());
  else
    src = CCConst(vofDyn_.stateField());
  copyInner(t, e_, G, src, e3_, kVofG);
  auto v = gatherInner(t);
  if (which <= 1)
    for (auto& q : v)
      q *= toDeg;
  return v;
}

template <class Grid>
typename Solver<Grid>::ContactAngleDiagnostics Solver<Grid>::contactAngleDiagnostics() {
  if (!vofEnabled_)
    throw std::runtime_error("contact_angle_diagnostics: VoF is not enabled");
  ContactAngleDiagnostics d;
  d.setAngle = contactAngleDeg_;
  if (!vofAdv_.hasWetting())
    return d;
  bridgeColourToVof();  // regenerates the fill, hence the census
  long counts[vof::kVofWetCount];
  long nApp = 0;
  // ISSUES sweep item 3: a wetting DOMAIN wall's band lives in the GHOST layers, so the
  // census has to reach them or `contact_cells` reads 0 -- the very symptom the item was
  // reported for. 0 (the inner region) when no domain wall is wetting.
  vofAdv_.wettingCensusGhost(counts, d.meanApparentAngle, nApp, vofWetWallMask() ? kVofG : 0);
  d.unfilledCells = counts[vof::kVofWetNone];
  d.contactCells = counts[vof::kVofWetTheta];
  d.neighbourCells = counts[vof::kVofWetNeighbour];
  d.pureCells = counts[vof::kVofWetPure];
  d.parallelCells = counts[vof::kVofWetParallel];
  d.neutralCells = counts[vof::kVofWetNeutral];
  if (vofDyn_.active() && vofDyn_.allocated()) {
    const auto cs = vofDyn_.census(vofAdv_);
    d.dynamicCells = cs.contactCells;
    d.pinnedCells = cs.cells[vof::kVofDynPinned];
    d.advancingCells = cs.cells[vof::kVofDynAdvancing];
    d.recedingCells = cs.cells[vof::kVofDynReceding];
    d.meanImposedTheta = cs.meanImposedDeg;
    d.meanApparentTheta = cs.meanApparentDeg;
    d.maxCaCl = cs.maxCa;
    d.maxContactSpeed = cs.maxSpeed;
  }
  return d;
}

template <class Grid>
void Solver<Grid>::applyContactAngle() {
  if (!contactAngleSet_ || !vofEnabled_ || !vofAdv_.hasGeometry())
    return;  // remembered; buildVofGeometry calls back once the geometry exists
  vofAdv_.enableWetting();
  vofAdv_.wettingPivot = contactPivot_;
  // (a) the SDF on the colour block: inner region from the solver's own sdf_, then the colour
  //     field's ghost policy, so the central-difference wall normal at ghost depth <= 2 is the
  //     OWNER's (the WO-Q finding-5 argument, applied to the wall normal).
  if (hasSolid_)
    copyInner(vofAdv_.wallSdf(), e3_, kVofG, CCConst(sdf_), e_, G);
  else
    Kokkos::deep_copy(vofAdv_.wallSdf(), 1e30);
  vofExchangeRaw(vofAdv_.wallSdf());
  // ISSUES sweep item 3: fold the wetting DOMAIN wall planes into the same field, on the WHOLE
  // block (the plane distance is a closed form of the global index, so no exchange is needed and
  // the out-of-domain band gets the value the clamp cannot produce). `min` is the SDF of the
  // union of the solids, so an SDF body and a domain wall compose; and the central difference
  // the theta pass takes of this field is exactly the inward face normal in the band.
  if (const int wet = vofWetWallMask()) {
    const I3 gs = vofGlobalSize(), org = vofOrigin();
    const C3 e = e3_;
    const int g = kVofG;
    CCField sd = vofAdv_.wallSdf();
    Kokkos::parallel_for(
        "peclet::flow::vof_wall_sdf_domain",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double d = vofWallPlaneSdf(x - g + org.x, y - g + org.y, z - g + org.z, gs, wet);
          sd(i) = Kokkos::fmin(sd(i), d);
        });
    CCExec().fence();
  }
  // (b) theta, in radians.
  const double toRad = 3.14159265358979323846 / 180.0;
  if (contactAngleField_.empty()) {
    Kokkos::deep_copy(vofAdv_.contactAngle(), contactAngleDeg_ * toRad);
  } else {
    CCField t("thetaG2", n_);
    std::vector<double> rad(contactAngleField_.size());
    for (std::size_t i = 0; i < rad.size(); ++i)
      rad[i] = contactAngleField_[i] * toRad;
    scatterInner(t, rad);
    copyInner(vofAdv_.contactAngle(), e3_, kVofG, CCConst(t), e_, G);
    vofExchangeRaw(vofAdv_.contactAngle());
  }
  // (c) rung V6: the STATIC base the dynamic selector starts from. The V6 pass OVERWRITES
  //     `contactAngle()` every fill, so it must never read back its own previous output.
  if (vofDyn_.active()) {
    vofDyn_.allocate(vofAdv_.size());
    Kokkos::deep_copy(vofDyn_.base(), vofAdv_.contactAngle());
  }
}

template <class Grid>
void Solver<Grid>::buildVofCellVelocity() {
  if (!vofDyn_.allocated())
    return;
  if constexpr (Grid::collocated)
    throw std::runtime_error("set_contact_angle_dynamic: SolverColocated has no immersed solid");
  const long st[3] = {1, (long)e_.x, (long)e_.x * (long)e_.y};
  for (int c = 0; c < 3; ++c) {
    // The cell-centre mean reads the face ONE cell out on the component's own axis, so the
    // velocity ghost ring has to be valid. Same rule as `bridgeVelocityToVof`: keep the
    // projection's outflow-face correction when there is one, otherwise the full fill (the
    // kinematic path, where the zero-gradient rule is what supplies the boundary face at all).
    if (outflowCorrValid_)
      fillVelGhostsTo(C[c].u, c, 0, false);
    else
      fillVelGhosts(c, 0);
    if (vofDynVel_[c].extent(0) != n_)
      vofDynVel_[c] = CCField("vofDynVel", n_);
    CCField t = vofDynVel_[c];
    CCConst u = CCConst(C[c].u);
    const long sc = st[c];
    const int ex = e_.x, ey = e_.y, g = G;
    Kokkos::parallel_for(
        "peclet::flow::vof_dyn_cellvel",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {nx_, ny_, nz_}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey;
          t(i) = 0.5 * (u(i) + u(i + sc));
        });
    CCExec().fence();
    copyInner(vofDyn_.cellVel(c), e3_, kVofG, CCConst(t), e_, G);
    vofExchangeRaw(vofDyn_.cellVel(c));
  }
}

template <class Grid>
const vof::WyAdvector& Solver<Grid>::vofAdvector() const {
  return vofAdv_;
}

template <class Grid>
void Solver<Grid>::setVofStepParity(long n) {
  vofStep_ = n;
  if (vofBlocks_)
    vofBlocks_->setStep(n);
}

template <class Grid>
long Solver<Grid>::vofStepParity() const {
  return vofStep_;
}

template <class Grid>
void Solver<Grid>::advectVofKinematic(double dtPhysArg) {
  const double dt = dtPhysArg * u_.timeToInt();
  if (!vofEnabled_)
    throw std::runtime_error("advect_vof: VoF is not enabled (call enable_vof / set_vof first)");
  requireVofGeometry("advect_vof");
  // ISSUES sweep item 5: on the collocated grid the advecting field is the MAC FACE field, and
  // a face field that was never built is all zeros — which the divergence guard below happily
  // certifies as solenoidal, so the whole advection became a silent no-op. `set_state` /
  // `set_velocity` now seed it (seedFaceFieldFromCells) and a projection builds it; refuse the
  // one remaining way to reach the trap.
  if constexpr (Grid::collocated) {
    if (!faceFieldValid_)
      throw std::runtime_error(
          "advect_vof: SolverColocated advects the colour with the MAC FACE field uf_/vf_/wf_ "
          "(the only discretely divergence-free field on this grid), and it has never been "
          "built — it is all zeros, which the divergence guard would certify as solenoidal and "
          "the advection would be a silent no-op. Call set_state()/set_velocity() (which now "
          "seed the face field through the same centerToFace map project() uses) or step() "
          "first.");
  }
  // WO-R2 item 4a (found by the E1 gallery page): the divergence guard below was INERT on a bare
  // box. `maxOpenDivergence()` returns 0.0 when there is no cut-cell pressure operator, so a
  // cell-centre-sampled analytic field — LeVeque, whose true max|div(open u)| is 0.612 — sailed
  // through the check and lost 4.93 % of the liquid in 50 steps with no diagnostic at all.
  // Refuse the configuration instead of measuring nothing.
  if (!cutcellPressure_)
    throw std::runtime_error(
        "advect_vof: no cut-cell pressure operator, so the divergence guard cannot measure "
        "anything (max_open_divergence() would return 0 whatever the field is). Build one with "
        "set_pressure_geometry(sdf) on an all-fluid box, or set_solid(sdf, "
        "cutcell_pressure=True) — Weymouth-Yue conservation is conditional on "
        "sum_f o_f u_f = 0 per cell and a cell-centre-sampled analytic field does NOT satisfy "
        "it (measured on LeVeque: max|div| 0.612, 4.93 % of the liquid lost in 50 steps).");
  // and measure the field the caller actually has: max_open_divergence() re-imposes the
  // zero-gradient outflow face before measuring, i.e. it MUTATES u and reports a field the
  // advector will not be handed (WO-R). The projected sibling does neither.
  const double div = maxOpenDivergenceProjectedInternal();  // the 1e-10 gate is index-unit
  if (!(div <= 1e-10)) {
    char msg[320];
    std::snprintf(msg, sizeof(msg),
                  "advect_vof: the current face velocity is not discretely divergence-free "
                  "(max|div(open*u)| = %.6g > 1e-10). Weymouth-Yue conservation is conditional "
                  "on it; project the field first (step() / project()).",
                  div);
    throw std::runtime_error(msg);
  }
  bridgeVelocityToVof();
  bridgeColourToVof();
  if (pcEnergy_) {  // WO-P23: the temperature rides the same sweeps here too
    ScalarField& sc = scalarField(pcTName_);
    copyInner(vofEnergy_.temperature(), e3_, kVofG, CCConst(sc.c), e_, G);
    vofEnergy_.advect(vofAdv_, dt, vofStep_++);
    copyInner(sc.c, e_, G, CCConst(vofEnergy_.temperature()), e3_, kVofG);
    scalarFillGhosts(sc);
  } else {
    vofAdv_.advect(dt, vofStep_++);
  }
  copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
  zeroSolidColour();
  fillPropGhosts(cField_);
  pcUpdateEnergyProps();
}

template <class Grid>
void Solver<Grid>::enableVofBlocks(const std::vector<std::array<double, 4>>& seeds) {
  prepareVofBlocks();
  for (const auto& sd : seeds)
    vofBlocks_->seedSphere(sd[0], sd[1], sd[2], sd[3]);
  vofBlocks_->scatter(vofAdv_.colour());
  finishVofBlocks();
}

template <class Grid>
void Solver<Grid>::prepareVofBlocks() {
  if (!vofEnabled_)
    throw std::runtime_error("enable_vof_blocks: VoF is not enabled (call enable_vof first)");
  if (hasSolid_)
    throw std::runtime_error(
        "enable_vof_blocks: the block container is ALL-FLUID at rung W0 (the cut-cell block is "
        "rung W12). Drop set_solid, or use the structured colour field (advect_vof).");
  int rank = 0, size = 1;
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
  }
#endif
  vofBlocks_ = std::make_shared<vof::VofBlockSet>();
  const std::array<bool, 3> per{vofAxisPeriodic(0), vofAxisPeriodic(1), vofAxisPeriodic(2)};
  vofBlocks_->init(vofGlobalSize(), per, rank, size, 1.0);  // flow works in cell units
  vofBlocks_->cflLimit = vofCflLimit_;
  vofBlockExch_ = std::make_shared<vof::VofBlockExchange>();
  vofBlockExch_->init(vofGlobalSize(), per, vofBlockRankBoxes(size), rank);
#ifdef PECLET_FLOW_MPI
  if (distributed_)
    vofBlockExch_->setComm(comm_);
#endif
  bindVofBlockPatch();
  vofBlocks_->setExchange(vofBlockExch_);
}

template <class Grid>
void Solver<Grid>::finishVofBlocks() {
  vofBlocks_->assignMasters();
  vofBlocks_->scatter(vofAdv_.colour());
  harvestVofBlockUnion();
}

template <class Grid>
bool Solver<Grid>::vofBlocksEnabled() const {
  return static_cast<bool>(vofBlocks_);
}

template <class Grid>
void Solver<Grid>::disableVofBlocks() {
  vofBlocks_.reset();
  vofBlockExch_.reset();
}

template <class Grid>
void Solver<Grid>::advectVofBlocks(double dtPhysArg, bool requireSolenoidal) {
  const double dt = dtPhysArg * u_.timeToInt();
  if (!vofBlocks_)
    throw std::runtime_error("advect_vof_blocks: call enable_vof_blocks first");
  const double div = requireSolenoidal ? maxOpenDivergenceInternal() : 0.0;  // index-unit gate
  if (!(div <= 1e-10)) {
    char msg[320];
    std::snprintf(msg, sizeof(msg),
                  "advect_vof_blocks: the current face velocity is not discretely "
                  "divergence-free (max|div(open*u)| = %.6g > 1e-10). Weymouth-Yue conservation "
                  "is conditional on it; project the field first (step() / project()).",
                  div);
    throw std::runtime_error(msg);
  }
  bridgeVelocityToVof();  // the block exchange reads THESE views (advector high-face convention)
  vofBlocks_->advect(dt, vofAdv_.colour());
  harvestVofBlockUnion();
}

template <class Grid>
std::vector<vof::VofBlockStats> Solver<Grid>::vofBlockStats() const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_stats: call enable_vof_blocks first");
  return vofBlocks_->statsAll();
}

template <class Grid>
double Solver<Grid>::vofBlockImbalance() const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_imbalance: call enable_vof_blocks first");
  return vofBlocks_->cellImbalance();
}

template <class Grid>
void Solver<Grid>::vofBlockCensus(std::vector<long>& masters, std::vector<long>& cells) const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_census: call enable_vof_blocks first");
  vofBlocks_->masterCensus(masters, cells);
}

template <class Grid>
void Solver<Grid>::setVofBlockAssign(int mode, long every) {
  if (!vofBlocks_)
    throw std::runtime_error("set_vof_block_assign: call enable_vof_blocks first");
  vofBlocks_->assignMode = static_cast<vof::VofMasterAssign>(mode);
  vofBlocks_->reassignEvery = every;
  vofBlocks_->assignMasters();
}

template <class Grid>
int Solver<Grid>::vofBlockAssign() const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_assign: call enable_vof_blocks first");
  return static_cast<int>(vofBlocks_->assignMode);
}

template <class Grid>
double Solver<Grid>::vofBlockImbalanceOf(int mode) const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_imbalance_of: call enable_vof_blocks first");
  const auto save = vofBlocks_->assignMode;
  vofBlocks_->assignMode = static_cast<vof::VofMasterAssign>(mode);
  const std::vector<int> m = vofBlocks_->plannedMasters();
  vofBlocks_->assignMode = save;
  const int np = vofBlocks_->size();
  std::vector<long> load(static_cast<std::size_t>(np), 0);
  long tot = 0, mx = 0;
  for (std::size_t i = 0; i < m.size(); ++i) {
    const long w = vofBlocks_->blocks()[i].box.cells();
    load[static_cast<std::size_t>(m[i])] += w;
    tot += w;
  }
  for (long v : load)
    mx = std::max(mx, v);
  return tot == 0 ? 1.0 : static_cast<double>(mx) * np / static_cast<double>(tot);
}

template <class Grid>
void Solver<Grid>::setVofBlockDeviceStaging(bool on) {
  if (!vofBlockExch_)
    throw std::runtime_error("set_vof_block_device_staging: call enable_vof_blocks first");
  vofBlockExch_->deviceStaging = on;
}

template <class Grid>
void Solver<Grid>::setVofBlockPool(bool on) {
  if (!vofBlocks_)
    throw std::runtime_error("set_vof_block_pool: call enable_vof_blocks first");
  vofBlocks_->usePool = on;
  if (!on)
    vofBlocks_->clearPool();
}

template <class Grid>
std::array<long, 2> Solver<Grid>::vofBlockPoolStats() const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_pool_stats: call enable_vof_blocks first");
  return {vofBlocks_->poolHits(), vofBlocks_->poolMisses()};
}

template <class Grid>
void Solver<Grid>::enableVofBlocksFromField(const std::vector<std::array<int, 6>>& boxes) {
  prepareVofBlocks();
  for (const auto& q : boxes) {
    vof::VofBox bb;
    for (int d = 0; d < 3; ++d) {
      bb.lo[d] = q[d];
      bb.hi[d] = q[3 + d];
    }
    vofBlocks_->seedBox(bb);
  }
  bridgeColourToVof();  // the caller's `set_vof` field -> the g=3 patch the gather reads
  vofBlocks_->finishSeeding(vofAdv_.colour());
  finishVofBlocks();
}

template <class Grid>
std::vector<double> Solver<Grid>::vofBlockColour(long id) {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_color: call enable_vof_blocks first");
  if (id < 0 || static_cast<std::size_t>(id) >= vofBlocks_->count())
    throw std::runtime_error("vof_block_color: no such block id");
  return vofBlocks_->blockColourHost(static_cast<std::size_t>(id));
}

template <class Grid>
void Solver<Grid>::enableVofBlocksFromColours(const std::vector<std::array<int, 6>>& boxes,
                                              const std::vector<std::vector<double>>& colours) {
  if (boxes.size() != colours.size())
    throw std::runtime_error("enable_vof_blocks_from_colors: one color array per box is required");
  prepareVofBlocks();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    vof::VofBox bb;
    for (int d = 0; d < 3; ++d) {
      bb.lo[d] = boxes[i][d];
      bb.hi[d] = boxes[i][3 + d];
    }
    vofBlocks_->seedBoxWithColour(bb, colours[i]);
  }
  finishVofBlocks();
}

template <class Grid>
void Solver<Grid>::enableVofBlockCsf() {
  if (!vofBlocks_)
    throw std::runtime_error("enable_vof_block_csf: call enable_vof_blocks first");
  if (!(sigmaCsf_ > 0.0))
    throw std::runtime_error("enable_vof_block_csf: set_surface_tension(sigma) first (sigma > 0)");
  if constexpr (Grid::collocated)
    throw std::runtime_error(
        "enable_vof_block_csf: the block CSF is STAGGERED-only at rung W2 (the collocated face "
        "-acceleration form is V8's, and composing the two is not this rung).");
  // The block cascade must be the SAME estimator as the structured one -- above all the wisp
  // guard: without `interfaceEps` a Weymouth-Yue round-off wisp is an "interfacial" cell whose
  // zero-area PLIC polygon returns |kappa| ~ 1e8, and the face between it and a real interface
  // carries a force eight orders too large.  Measured consequence when it was missing: the
  // distributed run's CSF force differed from the single-rank one by 6.7e-3 after two steps,
  // amplified from a 3e-16 colour difference by a flipped cascade branch.
  vofBlocks_->curvProto.interfaceEps = csfInterfaceEps_;
  vofBlocks_->curvProto.weightWidth = vofCurv_.weightWidth;
  vofBlocks_->curvProto.monoTol = vofCurv_.monoTol;
  vofBlocks_->curvProto.ptWeightWidth = vofCurv_.ptWeightWidth;
  vofBlocks_->curvProto.cosMin = vofCurv_.cosMin;
  vofBlocks_->curvProto.useMixedHeightFit = vofCurv_.useMixedHeightFit;
  vofBlocks_->curvProto.useWorklist = vofCurv_.useWorklist;
  vofBlocks_->enableCsf(sigmaCsf_);
  const long len3 = static_cast<long>(e3_.x) * e3_.y * e3_.z;
  for (int c = 0; c < 3; ++c) {
    vofBlkF_[c] = SField("vof::blockcsf::patch", len3);
    csfBlkF_[c] = CCField("vof::blockcsf", n_);
  }
  computeVofBlockCsf();
}

template <class Grid>
void Solver<Grid>::computeVofBlockCsf() {
  if (!vofBlocks_ || !vofBlocks_->csfEnabled)
    return;
  vofBlocks_->computeCsf(vofBlkF_[0], vofBlkF_[1], vofBlkF_[2]);
  for (int c = 0; c < 3; ++c)
    copyInner(csfBlkF_[c], e_, G, CCConst(vofBlkF_[c]), e3_, kVofG);
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofBlockForce(int c) {
  if (!csfBlkF_[c].extent(0))
    throw std::runtime_error("vof_block_force: call enable_vof_block_csf() first");
  return gatherInner(csfBlkF_[c]);
}

template <class Grid>
vof::VofCurvature::Stats Solver<Grid>::vofBlockCurvatureStats() const {
  if (!vofBlocks_)
    throw std::runtime_error("vof_block_curvature_stats: call enable_vof_blocks first");
  return vofBlocks_->csfCurvatureStats();
}

template <class Grid>
void Solver<Grid>::setVofInflow(int f, double value) {
  checkVofBcFace(f, 2, "set_vof_inflow");
  vofInflowSet_[f] = true;
  vofInflowC_[f] = value;
  vofInflowProfRaw_[f].clear();
  vofBcArm();
}

template <class Grid>
void Solver<Grid>::setVofInflowProfile(int f, const std::vector<double>& prof, int nb, int nc) {
  checkVofBcFace(f, 2, "set_vof_inflow_profile");
  if ((int)prof.size() != nb * nc)
    throw std::runtime_error("set_vof_inflow_profile: profile size != nb*nc");
  vofInflowSet_[f] = true;
  vofInflowProfRaw_[f] = prof;
  vofInflowProfNb_[f] = nb;
  vofInflowProfNc_[f] = nc;
  vofBcArm();
}

template <class Grid>
void Solver<Grid>::setVofBackflow(int f, double value) {
  checkVofBcFace(f, 3, "set_vof_backflow");
  vofBackflowSet_[f] = true;
  vofBackflowC_[f] = value;
  vofBcArm();
}

template <class Grid>
bool Solver<Grid>::vofBcActive() const {
  return vofBcActive_;
}

template <class Grid>
std::vector<double> Solver<Grid>::vofBcVolumes() const {
  return std::vector<double>(vofBcVol_, vofBcVol_ + 6);
}

template <class Grid>
std::vector<double> Solver<Grid>::vofBcVolumesTotal() const {
  return std::vector<double>(vofBcVolTotal_, vofBcVolTotal_ + 6);
}

template <class Grid>
void Solver<Grid>::resetVofBcVolumes() {
  for (int f = 0; f < 6; ++f)
    vofBcVol_[f] = vofBcVolTotal_[f] = 0.0;
}

template <class Grid>
void Solver<Grid>::vofApplyColourBc(CCField f) {
  if (!vofBcActive_ || !vofEnabled_)
    return;
  if (f.data() != vofAdv_.colour().data())
    return;
  const I3 e3{e3_.x, e3_.y, e3_.z};
  for (int face = 0; face < 6; ++face) {
    if (!touchesGlobalFace(face))
      continue;  // rank-owned global faces only (the WO-F rule)
    const int a = face / 2, sd = face % 2;
    if (bc_[face] == 2 && vofInflowSet_[face]) {
      if (vofInflowProf3_[face].extent(0))
        vof::bcColourProfile(f, e3, kVofG, a, sd, vofInflowProf3_[face], vofProf3Nc_[face]);
      else
        vof::bcColourConst(f, e3, kVofG, a, sd, vofInflowC_[face]);
    } else if (bc_[face] == 3 && vofBackflowSet_[face]) {
      // reads the face velocity the advector is about to flux with, so it must run after
      // bridgeVelocityToVof() — which it does: advectVof bridges the velocity first.
      vof::bcColourBackflow(f, e3, kVofG, a, sd, vofAdv_.faceVel(a), vofBackflowC_[face]);
    }
  }
}

template <class Grid>
void Solver<Grid>::vofBcPropGhosts(CCField f) {
  if (!vofBcActive_ || !vofEnabled_)
    return;
  const bool isColour = cField_.extent(0) && f.data() == cField_.data();
  const I3 e2{e_.x, e_.y, e_.z};
  for (int face = 0; face < 6; ++face) {
    if (bc_[face] != 2 || !vofInflowSet_[face] || !touchesGlobalFace(face))
      continue;
    const int a = face / 2, sd = face % 2;
    if (isColour) {
      if (vofInflowProfG2_[face].extent(0))
        vof::bcColourProfile(f, e2, G, a, sd, vofInflowProfG2_[face], vofProfG2Nc_[face]);
      else
        vof::bcColourConst(f, e2, G, a, sd, vofInflowC_[face]);
    } else {
      for (const auto& cl : closures_)
        if (cl.out.data() == f.data())
          applyClosureFaceGhost(cl, e_, G, a, sd);
    }
  }
}

template <class Grid>
void Solver<Grid>::vofRebuildBcBlock() {
  if (!vofEnabled_)
    return;
  const I3 e3{e3_.x, e3_.y, e3_.z};
  const bool px = vofAxisPeriodic(0), py = vofAxisPeriodic(1), pz = vofAxisPeriodic(2);
  if (vofBcActive_) {
    vofOutside_ = vof::UCField("vof::outside", vofAdv_.size());
    vof::buildOutsideMask(vofOutside_, e3, kVofG, vofOrigin(), vofGlobalSize(), px, py, pz);
    vofAdv_.setOutsideMask(vofOutside_);
  } else {
    vofAdv_.setOutsideMask(vof::UCField());
  }
  // the boundary-flux ledger counts only the GLOBAL domain faces this rank owns
  for (int face = 0; face < 6; ++face)
    vofAdv_.setBcFaceOwned(face, vofBcActive_ && bc_[face] != 0 && touchesGlobalFace(face));
  for (int face = 0; face < 6; ++face) {
    vofInflowProf3_[face] = CCField();
    vofInflowProfG2_[face] = CCField();
    if (vofInflowProfRaw_[face].empty())
      continue;
    vofInflowProf3_[face] =
        resampleFaceScalar(vofInflowProfRaw_[face], vofInflowProfNb_[face], vofInflowProfNc_[face],
                           face, e3_, kVofG, vofProf3Nc_[face]);
    vofInflowProfG2_[face] =
        resampleFaceScalar(vofInflowProfRaw_[face], vofInflowProfNb_[face], vofInflowProfNc_[face],
                           face, e_, G, vofProfG2Nc_[face]);
  }
}

template <class Grid>
CCField Solver<Grid>::resampleFaceScalar(const std::vector<double>& prof, int nb, int nc, int face,
                                         C3 ext, int g, int& outNc) {
  const int a = face / 2;
  const int dims[3] = {ext.x, ext.y, ext.z};
  const int bax = (a + 1) % 3, cax = (a + 2) % 3;
  const int Lb = dims[bax], Lc = dims[cax];
  CCField pf("vof::bcprof", (std::size_t)Lb * Lc);
  auto h = Kokkos::create_mirror_view(pf);
  auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  for (int p0 = 0; p0 < Lb; ++p0)
    for (int p1 = 0; p1 < Lc; ++p1)
      h((long)p0 * Lc + p1) = prof[(std::size_t)cl(p0 - g, nb) * nc + cl(p1 - g, nc)];
  Kokkos::deep_copy(pf, h);
  outNc = Lc;
  return pf;
}

template <class Grid>
void Solver<Grid>::checkVofBcFace(int f, int wantType, const char* who) {
  if (f < 0 || f > 5)
    throw std::runtime_error(std::string(who) + ": face must be 0..5 (-x,+x,-y,+y,-z,+z)");
  enableVof();
  if (bc_[f] != wantType)
    throw std::runtime_error(std::string(who) + ": face " + std::to_string(f) +
                             " has domain BC type " + std::to_string(bc_[f]) + ", not " +
                             std::to_string(wantType) + " — call set_domain_bc(face, " +
                             std::to_string(wantType) +
                             ", ...) first (2 = inflow, "
                             "3 = outflow).");
}

template <class Grid>
void Solver<Grid>::vofBcArm() {
  vofBcActive_ = true;
  vofRebuildBcBlock();
  // Refresh C's G=2 ghost band NOW. The property ghosts derive from it (vofBcPropGhosts) and the
  // first step's `project()` fills rho's ghosts BEFORE the colour advection refills C's, so
  // without this the first step would evaluate rho(C) at the inflow ghost on the pre-BC (Neumann
  // copy) colour — i.e. on the interior's phase.
  if (vofEnabled_ && cField_.extent(0))
    fillPropGhosts(cField_);
}

template <class Grid>
void Solver<Grid>::vofHarvestBcVolumes() {
  if (!vofBcActive_)
    return;
  for (int f = 0; f < 6; ++f) {
    vofBcVol_[f] = vofAdv_.bcFaceVolume(f);
    vofBcVolTotal_[f] += vofBcVol_[f];
  }
}

template <class Grid>
void Solver<Grid>::computeVofCurvature() {
  if (!vofEnabled_)
    throw std::runtime_error(
        "compute_vof_curvature: VoF is not enabled (call enable_vof / set_vof first)");
  if (!kappaField_.extent(0)) {
    kappaField_ = addField("kappa");
    kappaBranch_ = addField("kappa_branch");
  }
  bridgeColourToVof();
  vofCurvStats_ = vofCurv_.compute(vofAdv_.colour());
  copyInner(kappaField_, e_, G, CCConst(vofCurv_.kappa()), e3_, kVofG);
  copyInner(kappaBranch_, e_, G, CCConst(vofCurv_.branch()), e3_, kVofG);
  // kappa is face-interpolated by the V4 surface-tension force exactly as the properties are, so
  // it gets the same rank-aware ghost policy they do (WO-G / WO-I).
  fillPropGhosts(kappaField_);
  fillPropGhosts(kappaBranch_);
}

template <class Grid>
vof::VofCurvature::Stats Solver<Grid>::vofCurvatureStats() const {
  return vofCurvStats_;
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofCurvature() {
  if (!kappaField_.extent(0))
    throw std::runtime_error("vof_curvature: call compute_vof_curvature() first");
  std::vector<double> out = gatherInner(kappaField_);
  const double k = u_.curvToPhys();
  if (k != 1.0)
    for (double& x : out)
      x *= k;
  return out;
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofCurvatureBranch() {
  if (!kappaBranch_.extent(0))
    throw std::runtime_error("vof_curvature_branch: call compute_vof_curvature() first");
  return gatherInner(kappaBranch_);
}

template <class Grid>
void Solver<Grid>::setVofCurvatureWeightWidth(double d) {
  vofCurv_.weightWidth = d;
}

template <class Grid>
double Solver<Grid>::vofCurvatureWeightWidth() const {
  return vofCurv_.weightWidth;
}

template <class Grid>
void Solver<Grid>::setVofCurvatureMixedHeightFit(bool on) {
  vofCurv_.useMixedHeightFit = on;
}

template <class Grid>
bool Solver<Grid>::vofCurvatureMixedHeightFit() const {
  return vofCurv_.useMixedHeightFit;
}

template <class Grid>
void Solver<Grid>::setSurfaceTension(double sigma) {
  if (!(sigma >= 0.0))
    throw std::runtime_error("set_surface_tension: sigma must be >= 0");
  if (sigma > 0.0) {
    enableVof();
    if (!kappaField_.extent(0)) {
      kappaField_ = addField("kappa");
      kappaBranch_ = addField("kappa_branch");
    }
    // Wisp guard on the curvature's interfacial predicate. NOT optional once the curvature feeds
    // a force: Weymouth-Yue leaves round-off colour residue (measured down to -3e-35) in every
    // cell its sweeps touch, those cells satisfy `0 < C < 1`, and the cascade returns |kappa| up
    // to 1e8 for them off a zero-area PLIC polygon. A face between one of them and a real
    // interfacial cell then carries a force eight orders too large. See
    // `vof::VofCurvature::interfaceEps` for the measurement; the V3 default (0) is unchanged for
    // anyone calling `compute_vof_curvature()` without surface tension.
    vofCurv_.interfaceEps = csfInterfaceEps_;
  }
  // sigma is PHYSICAL (force per unit length); internally sigma*tRef^2/(rhoRef*hRef^3), which is
  // what makes p' = sigma'*kappa' hold with kappa' = kappa*hRef.
  sigmaPhys_ = sigma;
  sigmaCsf_ = sigma * u_.sigmaToInt();
}

template <class Grid>
double Solver<Grid>::surfaceTension() const {
  return sigmaPhys_;
}

template <class Grid>
void Solver<Grid>::setVofInterfaceEps(double eps) {
  csfInterfaceEps_ = eps;
  if (sigmaCsf_ > 0.0)
    vofCurv_.interfaceEps = eps;
}

template <class Grid>
double Solver<Grid>::vofInterfaceEps() const {
  return csfInterfaceEps_;
}

template <class Grid>
void Solver<Grid>::setCsfMode(int m) {
  csfMode_ = m;
}

template <class Grid>
int Solver<Grid>::csfMode() const {
  return csfMode_;
}

template <class Grid>
bool Solver<Grid>::vofBlockCsf() const {
  return static_cast<bool>(vofBlocks_) && vofBlocks_->csfEnabled;
}

template <class Grid>
bool Solver<Grid>::csfActive() const {
  return vofEnabled_ && sigmaCsf_ > 0.0 && (kappaField_.extent(0) != 0 || vofBlockCsf());
}

template <class Grid>
void Solver<Grid>::setVofKappaFrozen(bool on) {
  kappaFrozen_ = on;
}

template <class Grid>
bool Solver<Grid>::vofKappaFrozen() const {
  return kappaFrozen_;
}

template <class Grid>
void Solver<Grid>::setVofKappaConstant(double kappa) {
  enableVof();
  if (!kappaField_.extent(0)) {
    kappaField_ = addField("kappa");
    kappaBranch_ = addField("kappa_branch");
  }
  Kokkos::deep_copy(kappaField_, kappa / u_.curvToPhys());
  Kokkos::deep_copy(kappaBranch_, (double)vof::kCurvHf);
  kappaFrozen_ = true;
}

template <class Grid>
double Solver<Grid>::capillaryDt() {
  return capillaryDtInternal() * u_.timeToPhys();
}

template <class Grid>
double Solver<Grid>::capillaryDtInternal() {
  if (!(sigmaCsf_ > 0.0))
    return std::numeric_limits<double>::infinity();
  // Phase 3 (V3.2): the SMALLEST spacing sets the shortest resolvable capillary wave. `hpMin`
  // is exactly 1.0 on every isotropic run, so the number is unchanged.
  double hpMin = u_.hp[0];
  for (int a = 1; a < 3; ++a)
    hpMin = hpMin < u_.hp[a] ? hpMin : u_.hp[a];
  return vof::capillaryDt(phaseDensitySum(), vofEnabled_ ? hpMin : 1.0, sigmaCsf_);
}

template <class Grid>
void Solver<Grid>::setCapillaryCfl(double f) {
  capillaryCfl_ = f;
}

template <class Grid>
double Solver<Grid>::capillaryCfl() const {
  return capillaryCfl_;
}

template <class Grid>
typename Solver<Grid>::VofStepLimits Solver<Grid>::vofStepLimits() {
  VofStepLimits L;
  L.courant = vofMaxCourant();
  L.cflDt = (L.courant > 0.0) ? dtPhys_ * vofCflLimit_ / L.courant
                              : std::numeric_limits<double>::infinity();
  L.capillaryDt = capillaryDt();
  const double cap = capillaryCfl_ * L.capillaryDt;
  L.capillaryBinds = cap < L.cflDt;
  L.binding = L.capillaryBinds ? cap : L.cflDt;
  return L;
}

template <class Grid>
double Solver<Grid>::phaseDensitySum() {
  if (vofMomEnabled_)
    return vofRhoG_ + vofRhoL_;
  if (!effVarRho())
    return 2.0 * rho_;
  CCExec space;
  C3 e = e_;
  double lo = 1e300, hi = -1e300;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  // A closure-driven rho field is produced by updateProperties() at the head of the step, so
  // before the FIRST step it is still all zeros and a naive min+max would report 0 — and this is
  // a diagnostic users call while choosing dt, i.e. exactly then. Refresh and retry once in that
  // case; the step's own call site runs after updateProperties() and never takes the branch.
  for (int pass = 0; pass < 2; ++pass) {
    CCConst f = CCConst(effRhoField());
    lo = 1e300;
    hi = -1e300;
    Kokkos::parallel_reduce(
        "rho_minmax", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& mn, double& mx) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          mn = Kokkos::fmin(mn, f(i));
          mx = Kokkos::fmax(mx, f(i));
        },
        Kokkos::Min<double>(lo), Kokkos::Max<double>(hi));
    if (hi > 0.0 || pass == 1)
      break;
    updateProperties();
  }
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    double g[2] = {lo, -hi}, r[2];
    MPI_Allreduce(g, r, 2, MPI_DOUBLE, MPI_MIN, comm_);
    lo = r[0];
    hi = -r[1];
  }
#endif
  return lo + hi;
}

template <class Grid>
double Solver<Grid>::stepAdaptive(double cflTarget, double capillaryCflTarget, double dtMax) {
  if (!vofEnabled_)
    throw std::runtime_error(
        "step_adaptive: needs enable_vof() (it picks dt from the two "
        "explicit two-phase limits vof_step_limits() reports)");
  if (!(cflTarget > 0.0) || !(capillaryCflTarget > 0.0) || !(dtMax > 0.0))
    throw std::runtime_error("step_adaptive: cfl_target, capillary_cfl and dt_max must be > 0");
  const VofStepLimits L = vofStepLimits();
  double dt = cflTarget * L.cflDt;
  const double dc = capillaryCflTarget * L.capillaryDt;
  if (dc < dt)
    dt = dc;
  if (dtMax < dt)
    dt = dtMax;
  if (!(dt > 0.0) || !std::isfinite(dt))
    throw std::runtime_error("step_adaptive: no finite dt (cfl_dt = " + std::to_string(L.cflDt) +
                             ", capillary_dt = " + std::to_string(L.capillaryDt) +
                             ") - pass a finite dt_max when neither limit is active");
  setDt(dt);
  // The limits were just evaluated on this exact state, so `step()`'s own head-of-step
  // pre-check (item 1) would repeat the interface-local reduction and the g=3 fill for nothing.
  vofPrecheckDone_ = true;
  step();
  return dt;
}

template <class Grid>
void Solver<Grid>::vofStepPrecheck() {
  const bool skip = vofPrecheckDone_;
  vofPrecheckDone_ = false;
  if (!vofEnabled_ || skip)
    return;
  if (csfActive()) {
    const double cap = capillaryCfl_ * capillaryDtInternal();
    if (!(dt_ <= cap))
      throw std::runtime_error(capillaryThrowMessage(cap));
  }
  // The block container carries one advector per marker with its own colour support; the union
  // field on `vofAdv_` is a derived quantity there, so the cap is left to the blocks themselves.
  if (vofBlocks_)
    return;
  const double cfl = vofMaxCourant();
  if (!(cfl <= vofCflLimit_))
    throw std::runtime_error(
        "peclet::flow::vof::WyAdvector: CFL = max|uf| dt/h = " + std::to_string(cfl) +
        " exceeds the Weymouth-Yue boundedness cap " + std::to_string(vofCflLimit_) +
        " (dt = " + std::to_string(dt_) + ", h = " + std::to_string(vofAdv_.h()) +
        ") - reduce dt. Rejected at the HEAD of step(): no field has been advanced, so a retry "
        "at a smaller dt is exact (see set_vof_cfl_limit / step_adaptive).");
}

template <class Grid>
std::string Solver<Grid>::capillaryThrowMessage(double cap) const {
  return "surface tension: dt = " + std::to_string(dt_) + " exceeds the capillary limit " +
         std::to_string(cap) + " (Brackbill sqrt((rho_1+rho_2) h^3/(4 pi sigma)) = " +
         std::to_string(cap / (capillaryCfl_ > 0.0 ? capillaryCfl_ : 1.0)) + " x safety factor " +
         std::to_string(capillaryCfl_) +
         "). Surface tension is EXPLICIT: this is a hard stability boundary (Denner & van "
         "Wachem 2015), not a margin. Reduce dt, or raise set_capillary_cfl deliberately.";
}

template <class Grid>
void Solver<Grid>::updateVofCurvature() {
  if (!csfActive())
    return;
  if (vofBlockCsf())
    computeVofBlockCsf();  // rung W2: per-block cascade + the CSF face force, scattered SUM
  else if (!kappaFrozen_)
    computeVofCurvature();
  const double cap = capillaryCfl_ * capillaryDtInternal();
  if (!(dt_ <= cap))
    throw std::runtime_error(
        "surface tension: dt = " + std::to_string(dt_) + " exceeds the capillary limit " +
        std::to_string(cap) + " (Brackbill sqrt((rho_1+rho_2) h^3/(4 pi sigma)) = " +
        std::to_string(capillaryDtInternal()) + " x safety factor " +
        std::to_string(capillaryCfl_) +
        "). Surface tension is EXPLICIT: this is a hard stability boundary (Denner & van Wachem "
        "2015), not a margin. Reduce dt, or raise set_capillary_cfl deliberately.");
}

template <class Grid>
void Solver<Grid>::advectVof() {
  if (!vofEnabled_)
    return;
  // Rung W2: with the block container on, the colour is the BLOCKS' state and the registered
  // "C" is the union derived from it — so the step's colour stage is the block advection, in
  // exactly this slot and for exactly the same reason (the face field has just been projected).
  // `vofBlocks_` is null unless `enable_vof_blocks` ran, so every other path is unchanged.
  if (vofBlocks_) {
    advectVofBlocks(dt_, /*requireSolenoidal=*/false);
    return;
  }
  requireVofGeometry("enable_vof");
  const double _tv0 = vofTick();  // WO-V9: the whole colour stage
  double _tb = 0.0;               // ... of which the G=2 <-> g=3 bridges
  const double _tb0 = vofTick();
  bridgeVelocityToVof();
  bridgeColourToVof();
  vofAdd(_tb, _tb0);
  vofAdv_.resetBcFaceVolume();  // WO-R: the per-step boundary liquid ledger
  if (pcEnergy_) {
    // WO-P23: the CONSISTENT rho c_p T transport. The temperature rides the SAME sweeps, planes
    // and fluxes as the colour (`vof/energy_advect.hpp`), so it has to be bridged onto the g=3
    // block first and taken off it after; the scalar module then does diffusion ONLY (its
    // `energy` flag switches off the Koren advective term and the constant-D operator).
    ScalarField& sc = scalarField(pcTName_);
    copyInner(vofEnergy_.temperature(), e3_, kVofG, CCConst(sc.c), e_, G);
    vofEnergy_.advect(vofAdv_, dt_, vofStep_++);
    copyInner(sc.c, e_, G, CCConst(vofEnergy_.temperature()), e3_, kVofG);
    scalarFillGhosts(sc);
  } else {
    vofAdv_.advect(dt_, vofStep_++);
  }
  vofHarvestBcVolumes();
  // Back to the G=2 registry mirror, then ITS ghost policy: the closures write inner cells only,
  // but the property face means (rho_f in the momentum diagonal, the projection coefficient, the
  // face body force) read the ghost ring, so C's ghosts must be filled with the SAME policy rho
  // uses or the derived rho ghost is inconsistent with the interior at a boundary.
  const double _tb1 = vofTick();
  copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
  zeroSolidColour();  // rung V5a: the canonical field carries 0 in solid cells, not the fill
  fillPropGhosts(cField_);
  vofAdd(_tb, _tb1);
  vofAdd(vt_.advect, _tv0);
  vt_.bridge += _tb;
}

template <class Grid>
void Solver<Grid>::enableVofMomentum(double rhoGasPhys, double rhoLiquidPhys) {
  const double rhoGas = rhoGasPhys * u_.rhoToInt(), rhoLiquid = rhoLiquidPhys * u_.rhoToInt();
  if constexpr (Grid::collocated)
    throw std::runtime_error(
        "enable_vof_momentum: momentum-consistent VoF transport is STAGGERED-ONLY (rung V2b); the "
        "collocated construction is Favre-averaged face states, rung V8.");
  enableVof();
  if (porous_)
    throw std::runtime_error(
        "enable_vof_momentum: the volume-averaged porous momentum and the momentum-consistent VoF "
        "transport both own the momentum time term; they are not composable at this rung.");
  if (!(rhoGas > 0.0) || !(rhoLiquid > 0.0))
    throw std::runtime_error("enable_vof_momentum: both phase densities must be > 0");
  vofRhoG_ = rhoGas;
  vofRhoL_ = rhoLiquid;
  vofMomEnabled_ = true;
  for (int c = 0; c < 3; ++c)
    if (uAdv_[c].extent(0) != n_)
      uAdv_[c] = CCField("uAdv", n_);
  vofMom_.init(vofAdv_, vofRhoG_, vofRhoL_);
}

template <class Grid>
bool Solver<Grid>::vofMomentumEnabled() const {
  return vofMomEnabled_;
}

template <class Grid>
void Solver<Grid>::setVofRhoFloorFrac(double f) {
  vofMom_.rhoFloorFrac = f;
}

template <class Grid>
double Solver<Grid>::vofRhoFloorFrac() const {
  return vofMom_.rhoFloorFrac;
}

template <class Grid>
double Solver<Grid>::vofRhoFloor() const {
  return vofMom_.lastRhoFloor();
}

template <class Grid>
void Solver<Grid>::setVofMomentumMuscl(bool on) {
  vofMom_.momentumMuscl = on;
}

template <class Grid>
void Solver<Grid>::setVofMomentumCellFlag(bool on) {
  vofMom_.useCellDilationFlag = on;
}

template <class Grid>
void Solver<Grid>::setVofFluxClamp(bool on) {
  vofMom_.clampFluxes = on;
}

template <class Grid>
vof::MomentumConsistentAdvector::Diagnostics Solver<Grid>::vofMomentumDiagnostics() {
  if (!vofMomEnabled_)
    throw std::runtime_error("vof_momentum_diagnostics: enable_vof_momentum was never called");
  return vofMom_.diagnostics();
}

template <class Grid>
std::vector<double> Solver<Grid>::getVofAdvectedVelocity(int c) {
  if (!vofMomEnabled_)
    throw std::runtime_error("vof_advected_velocity: enable_vof_momentum was never called");
  return gatherInner(uAdv_[c]);
}

template <class Grid>
void Solver<Grid>::advectVofMomentum() {
  if (!vofMomEnabled_)
    return;
  requireVofGeometry("enable_vof_momentum");
  if (implicitAdv())
    throw std::runtime_error(
        "enable_vof_momentum is incompatible with implicit advection (set_implicit_advection / a "
        "domain-BC stencil path): the momentum advection is already done conservatively by the "
        "VoF fluxes, and the implicit-FOU operator would add a second one.");
  if (!effVarRho())
    throw std::runtime_error(
        "enable_vof_momentum requires the variable-density momentum/projection path: register a "
        "density closure on C (set_property_model('rho','linear','C',[rho_g, rho_l-rho_g])) or "
        "call set_density_mode('variable').");
  if (vofMom_.phaseRhoG() != vofRhoG_ || vofMom_.phaseRhoL() != vofRhoL_)
    vofMom_.setPhaseDensities(vofRhoG_, vofRhoL_);
  const double _tv0 = vofTick();  // WO-V9: the whole colour + momentum stage
  double _tb = 0.0;
  const double _tb0 = vofTick();
  bridgeVelocityToVof();
  bridgeColourToVof();
  vofAdd(_tb, _tb0);
  vofAdv_.resetBcFaceVolume();  // WO-R: the per-step boundary liquid ledger
  vofMom_.advect(vofAdv_, dt_, vofStep_++);
  vofHarvestBcVolumes();
  // Colour back to the G=2 registry mirror (same contract as advectVof), and the advected
  // velocity back onto the solver's velocity index convention.
  const double _tb1 = vofTick();
  copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
  zeroSolidColour();
  fillPropGhosts(cField_);
  // A plain inner-to-inner copy: the momentum control volumes are indexed in the solver's own
  // low-face convention (vof/momentum_advect.hpp "Indexing"), so CV_c(i) IS the solver's u_c(i)
  // and there is no shift here to get wrong.
  for (int c = 0; c < 3; ++c)
    copyInner(uAdv_[c], e_, G, CCConst(vofMom_.advectedVelocity(c)), e3_, kVofG);
  vofAdd(_tb, _tb1);
  vofAdd(vt_.momAdvect, _tv0);
  vt_.momBridge += _tb;
}

template <class Grid>
void Solver<Grid>::vofExchangeScalar(CCField f) {
  const bool px = vofAxisPeriodic(0), py = vofAxisPeriodic(1), pz = vofAxisPeriodic(2);
#ifdef PECLET_FLOW_MPI
  if (distributed_ && vofDev_)
    vofDev_->exchange(f);
  else
#endif
    vof::periodicFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, px, py, pz);
  if (px && py && pz)
    return;
  const I3 gs = vofGlobalSize(), org = vofOrigin();
  vof::clampFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, org, gs, px, py, pz);
}

template <class Grid>
double Solver<Grid>::vofTick() const {
  if (!vofTiming_)
    return 0.0;
  Kokkos::fence();
  return vof::WyAdvector::wallSeconds();
}

template <class Grid>
void Solver<Grid>::vofAdd(double& acc, double t0) {
  if (!vofTiming_)
    return;
  Kokkos::fence();
  acc += vof::WyAdvector::wallSeconds() - t0;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_VOF_HPP
