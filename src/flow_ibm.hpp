/// @file
/// @brief flow — host-facing Kokkos IBM Navier-Stokes solver (drop-in flow-style API).
///
/// Assembles the validated cut-cell IBM operators into a runnable solver on a fully-periodic MAC
/// box with immersed SDF solids: per-component backward-Euler implicit diffusion with the
/// Robust-Scaled cut-cell no-slip stencil (buildIbmOverlay + ibmBuildDiffusion + ibmModifyStencil +
/// ibmSolidMask + ibmRbgsSweep), then a rotational incremental-pressure Chorin projection through
/// the open-face-weighted cut-cell pressure Poisson (buildCutcellOp + divergOpen, solved by CG with
/// the constant null space projected out, then projectCorrect; P += (rho/dt)*phi - mu*div(u*)
/// matching CUDA press_update_k). Schemes are a FAITHFUL port of the CUDA flow (point-value
/// cut-cell IBM = ibm_geometry_ext_k<0>; rotational pressure): the velocity field matches CUDA to
/// ~1e-13 (machine precision). Physical units (rho/mu/dt + body force). std::vector setters/getters
/// so a pybind module can drive it. The verify_poiseuille / verify_periodic_spheres mechanism (k
/// matches CUDA to all printed digits), on any backend. NOTE (faithfulness items, see memory): the
/// CG uses a diagonal preconditioner where CUDA uses RB-GS-preconditioned MG-PCG (same converged
/// solution); the pressure operator is stored double where CUDA uses float mreal -- to reconcile in
/// a later port pass.
#ifndef PECLET_FLOW_SDFLOW_IBM_HPP
#define PECLET_FLOW_SDFLOW_IBM_HPP

#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "grid_layout.hpp"
#include "mac_approx_projection.hpp"
#include "mac_cutcell_mg.hpp"
#include "mac_ibm.hpp"
#include "mac_pressure.hpp"
#include "mac_stencils.hpp"
#include "mac_velocity_mg.hpp"
#include "face_props.hpp"
#include "peclet/core/field/field_set.hpp"
#include "property_closures.hpp"
#include "scalar_transport.hpp"
#include "staggered_advection.hpp"

namespace peclet::flow {

// Templated on a GridLayout policy (grid_layout.hpp) that supplies the grid-position-dependent
// pieces (currently: the per-component velocity sample offset). IbmSolver == Solver<Staggered> (the
// alias below) is bit-identical to the pre-policy solver; the Colocated policy is added in a later
// phase.
template <class Grid>
class Solver {
 public:
  using FV = Kokkos::View<float*, CCMem>;
  static constexpr int G = 2;  // velocity block: Koren advection reach (pressure/MG bridged to g=1)

  Solver(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
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
    }
    // Register the pre-existing solver fields in the named directory so the multiphysics machinery
    // (scalar transport, property closures) and load-balance redistribution can enumerate the whole
    // set uniformly. adopt() aliases the members (no reallocation, no ownership); all live on the G=2
    // velocity block and share velHalo_ under MPI.
    fields_.adopt("u", C[0].u, G, peclet::core::Centering::FaceX);
    fields_.adopt("v", C[1].u, G, peclet::core::Centering::FaceY);
    fields_.adopt("w", C[2].u, G, peclet::core::Centering::FaceZ);
    fields_.adopt("p", P_, G, peclet::core::Centering::Cell);
    fields_.adopt("sdf", sdf_, G, peclet::core::Centering::Cell);
  }

  void setRho(double r) { rho_ = r; }
  void setMu(double m) { mu_ = m; }
  void setDt(double d) { dt_ = d; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setVelocityIterations(int it) { velIters_ = it; }
  void setPressureIterations(int it) { presIters_ = it; }
  void setAdvection(bool on) { advect_ = on; }  // explicit high-order advection (default SOU)
  // High-order advection scheme for the (explicit, or deferred-correction) flux: 0 = second-order
  // upwind (SOU, default — 2nd order at smooth extrema too); 1 = Koren TVD (monotone limiter, the
  // legacy CUDA scheme). Only matters when advection is enabled; FOU stays the deferred-correction
  // base.
  void setAdvectionScheme(int s) { advScheme_ = s; }
  // Implicit-FOU deferred-correction advection (CUDA set_implicit_advection): solve the
  // first-order-upwind part of advection implicitly (in the velocity operator) + keep (Koren-FOU)
  // explicit in the RHS -> unconditionally stable for advection (high Re / large dt). Requires the
  // IBM stencil (rebuilt per Picard iteration with the FOU term); the domain-BC path needs
  // velocity-MG (separate milestone).
  void setImplicitAdvection(bool on) { implicitFou_ = on; }
  // Picard outer iterations over the step (CUDA set_outer_iterations): the advecting velocity is
  // lagged at the current iterate u^k while the time base stays u^n. iters>=1; tol>0 stops early on
  // max|du| < tol.
  void setOuterIterations(int iters) { outerIters_ = iters < 1 ? 1 : iters; }
  void setOuterTolerance(double tol) { outerTol_ = tol; }
  long lastOuterIterations() const { return lastOuterIters_; }
  // Velocity (momentum) multigrid for the IBM diffusion solve (CUDA set_velocity_multigrid): the
  // STAIRCASE coarse operator (exact == RB-GS, stiff-stable at large dt). Call before set_solid;
  // built at geometry time.
  void setVelocityMultigrid(bool on, int levels, int vcycles) {
    useVelocityMg_ = on;
    vmgLevels_ = levels < 1 ? 1 : levels;
    vmgVcycles_ = vcycles < 1 ? 1 : vcycles;
  }
  void setPressureLevels(int levels) {
    nLevels_ = levels < 1 ? 1 : levels;
  }  // MG depth (CUDA default 4)
  // Backflow stabilization at outflow faces (Bazilevs 2009 / Esmaily-Moghadam 2011): beta in [0,1]
  // scales the dissipative outflow term that prevents backflow divergence (0 = off). Default 0.2.
  void setBackflowStab(double beta) { backflowBeta_ = beta < 0.0 ? 0.0 : beta; }
  // Deferred-correction advection: on (default) = implicit FOU operator + explicit (HO - FOU)
  // high-order correction (2nd order; HO = SOU by default, or Koren TVD via set_advection_scheme).
  // off = pure implicit FOU (1st order, more dissipative, unconditionally stable) -- useful for very
  // sharp shear layers where the (unlimited SOU) explicit correction overshoots and destabilizes.
  void setDeferredCorrection(bool on) { deferredCorr_ = on; }
  // Chebyshev pressure driver (CUDA set_pressure_chebyshev): communication-light alternative to
  // MG-PCG -- Chebyshev semi-iteration preconditioned by one symmetric V-cycle, no per-iteration
  // global dot-products. Spectral bounds of M^{-1}A are estimated once (lazily) on the first solve
  // and reused every step.
  void setPressureChebyshev(bool on, int maxit, double rtol) {
    useChebyshev_ = on;
    chebMaxit_ = maxit;
    chebRtol_ = rtol;
    chebBoundsSet_ = false;
  }
  // MG-PCG pressure tolerance/iteration cap (CUDA set_pressure_pcg). The Kokkos cut-cell pressure
  // solve is MG-PCG by default; this just sets its bounds (the `on` flag is accepted for API
  // parity).
  void setPressurePcg(bool /*on*/, int maxit, double rtol) {
    pcgMaxit_ = maxit;
    pcgRtol_ = rtol;
  }
  // Incremental-rotational pressure (CUDA set_incremental_pressure, default ON): the predictor
  // carries -grad(P^n) and the physical pressure is accumulated rotationally P += (rho/dt)*phi -
  // mu*div(u*). OFF => classical non-incremental Chorin (no -grad(P^n) predictor; P derived on
  // demand as (rho/dt)*phi).
  void setIncrementalPressure(bool on) { incremental_ = on; }
  // Pressure warm-start (CUDA set_pressure_warmstart, default OFF): seed each cut-cell pressure
  // solve from the previous step's projection potential (consecutive phi's are similar along a
  // steady march -> a more converged phi per fixed solver budget) instead of zeroing the initial
  // guess.
  void setPressureWarmstart(bool on) { pwarm_ = on; }
  // Collocated cell->face interpolation for the approximate projection: 0 = plain ½/½ averaging
  // (default), 1 = wall-aware (wall-anchored weighted-LSQ quadratic at faces bordering the immersed
  // solid; centerToFaceWallAware in mac_approx_projection.hpp). No effect on the staggered path.
  void setFaceInterp(int mode) { faceInterp_ = mode; }
  // CUDA-only 3-stream concurrent velocity solve (set_velocity_streams): no Kokkos analogue in this
  // port (the default-execution-space kernels are already stream-ordered). Accepted as a no-op for
  // API parity.
  void setVelocityStreams(bool /*on*/) {}
  // Seed/restore the velocity state (CUDA set_state / upload_velocity): u/v/w are inner-cell fields
  // (flat x-fastest, size nx*ny*nz); written into the velocity block + ghosts refreshed (periodic
  // wrap).
  void uploadVelocity(const std::vector<double>& uu, const std::vector<double>& vv,
                      const std::vector<double>& ww) {
    const std::vector<double>* src[3] = {&uu, &vv, &ww};
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
  }
#ifdef PECLET_FLOW_MPI
  // Multi-rank: this rank's IbmSolver is constructed with its LOCAL block dims (= the
  // BlockDecomposer of the GLOBAL grid for this rank); initMpi wires the g=2 velocity-block halo +
  // the global-origin red-black parity, and switches fillGhosts/maxOpenDivergence + the pressure MG
  // (CutcellMG::initMpi) onto their distributed paths. The caller decomposes first (deterministic
  // ORB) to size the constructor; initMpi re-derives it.
  void initMpi(int gnx, int gny, int gnz, MPI_Comm comm) {
    distributed_ = true;
    comm_ = comm;
    gnx_ = gnx;
    gny_ = gny;
    gnz_ = gnz;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::array<bool, 3> per{true, true, true};
    velHalo_ = std::make_shared<GridHaloTopology<3>>();
    peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size),
                                                 peclet::core::IVec<3>{gnx, gny, gnz});
    velHalo_->buildTopology(dec, rank, G, per, comm);
    velDev_ = std::make_shared<GridHalo<double>>();
    velDev_->init(*velHalo_);
    const auto oig = velHalo_->indexer().originInclGhost();
    og_ = {(int)oig[0] + G, (int)oig[1] + G,
           (int)oig[2] + G};  // block inner origin -> global parity
  }
#endif
  // per-face domain BC {face 0..5 = -x,+x,-y,+y,-z,+z}: type 0=periodic,1=no-slip
  // wall,2=Dirichlet/inflow,3=outflow.
  void setDomainBc(int face, int type, double vx, double vy, double vz) {
    bc_[face] = type;
    bcVel_[face][0] = vx;
    bcVel_[face][1] = vy;
    bcVel_[face][2] = vz;
    hasBc_ = false;
    hasOutflow_ = false;
    for (int i = 0; i < 6; ++i) {
      if (bc_[i])
        hasBc_ = true;
      if (bc_[i] == 3)
        hasOutflow_ = true;
    }
  }
  // per-position inlet velocity profile on `face` (CUDA set_domain_bc_profile): prof is (nb,nc,3)
  // on the inner grid of the face's two perpendicular axes; sets the face to inflow (type 2).
  // Resampled (clamp) to the ghost-inclusive face grid so the BC kernel indexes it directly by face
  // position.
  void setDomainBcProfile(int face, const std::vector<double>& prof, int nb, int nc) {
    const int a = face / 2;
    const int dims[3] = {e_.x, e_.y, e_.z};
    const int bax = (a + 1) % 3, cax = (a + 2) % 3;
    const int Lb = dims[bax], Lc = dims[cax];
    CCField pf("bcprof", (std::size_t)Lb * Lc * 3);
    auto h = Kokkos::create_mirror_view(pf);
    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    for (int p0 = 0; p0 < Lb; ++p0)
      for (int p1 = 0; p1 < Lc; ++p1) {
        const int ib = cl(p0 - G, nb), ic = cl(p1 - G, nc);
        for (int k = 0; k < 3; ++k)
          h(((long)p0 * Lc + p1) * 3 + k) = prof[((std::size_t)ib * nc + ic) * 3 + k];
      }
    Kokkos::deep_copy(pf, h);
    bcProf_[face] = pf;
    bcProfNc_[face] = Lc;
    bc_[face] = 2;
    hasBc_ = true;  // a profiled face is an inflow
  }
  // all-fluid + domain-BC pressure (CUDA set_pressure_geometry): same path as set_solid with an
  // open SDF.
  void setPressureGeometry(const std::vector<double>& sdfInner) { setSolid(sdfInner, true); }

  // SDF on the inner cells (flat x-fastest, size nx*ny*nz; <0 solid). cutcellPressure enables the
  // open-face-weighted cut-cell projection (off => velocity-only, e.g. unidirectional body-force
  // flow).
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
    cutcellPressure_ = cutcellPressure;
    hasSolid_ = false;  // does the geometry actually contain solid? (all-fluid set_pressure_geometry
    for (double v : sdfInner)  // passes sd>0 everywhere -> stays false, keeping the channel/BFS path)
      if (v < 0.0) {
        hasSolid_ = true;
        break;
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
      // Multi-rank: sdfInner is THIS rank's LOCAL inner block; fill the inner cells, then
      // halo-exchange the ghosts (cross-rank + periodic) so the overlay/openness read the
      // neighbour's SDF at the block boundary.
      auto h = Kokkos::create_mirror_view(sdf_);
      Kokkos::deep_copy(h, sdf_);
      for (int z = 0; z < nz_; ++z)
        for (int y = 0; y < ny_; ++y)
          for (int x = 0; x < nx_; ++x)
            h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
                sdfInner[(std::size_t)x + (std::size_t)y * nx_ +
                         (std::size_t)z * (std::size_t)nx_ * ny_];
      Kokkos::deep_copy(sdf_, h);
      velDev_->exchange(sdf_);
    } else
#endif
    {
      // Single-rank: upload the inner SDF once and do the periodic-wrap gather on device (G4) —
      // fills the whole extended block (inner + periodic ghosts) in one kernel instead of a host
      // triple loop + a full extended-block H2D.
      CCField din("peclet::flow::sdfInner_d", static_cast<std::size_t>(nx_) * ny_ * nz_);
      Kokkos::deep_copy(
          din,
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              sdfInner.data(), sdfInner.size()));
      CCExec space;
      const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
      CCField sdf = sdf_;
      Kokkos::parallel_for(
          "peclet::flow::sdf_periodic_wrap",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                      iz = (((z - g) % nz) + nz) % nz;
            sdf((long)x + (long)y * ex + (long)z * (long)ex * ey) = din(
                (std::size_t)ix + (std::size_t)iy * nx + (std::size_t)iz * (std::size_t)nx * ny);
          });
      space.fence();
    }
    for (int c = 0; c < 3; ++c) {
      const Off3 off =
          Grid::offset(c);  // velocity-unknown placement (staggered: -1/2 face; collocated: 0)
      C[c].nCut = buildIbmOverlay<0>(
          CCConst(sdf_), e_, G, off, /*Dirichlet*/ 0, C[c].ov, C[c].idMap,
          C[c].counter);  // SCHEME 0 = point-value (matches CUDA ibm_geometry_ext_k<0>)
      ibmSolidMask(C[c].mask, CCConst(sdf_), e_, off);
      Kokkos::deep_copy(C[c].u, 0.0);
    }
    rebuildStencils();
    // Staggered domain BCs bake an implicit-diffusion wall fold; the collocated grid instead uses
    // explicit reflection ghosts (refreshed each smoother sweep), so it needs no fold.
    if (hasBc_ && !Grid::collocated)
      setupBcDiffusion();
    if (useVelocityMg_) {  // velocity-MG hierarchy: IBM (staircase/upwind) or domain-BC
                           // (const-coeff) mode
      vmg_.init(nx_, ny_, nz_, vmgLevels_);
      if (hasBc_)
        vmg_.setBC(bc_);
      else {
        vmgTheta_ = CCField("vmgTheta", n_);
        vmgClean_ = CCField("vmgClean", n_);
      }
    }
    if (cutcellPressure_) {
      buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, 1.0, 1.0, 1.0);  // on the g=2 velocity block
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
            if (t != 0 && !open)
              bcZeroOpenness(oa[a], e2, G, a, s);
          }
      }  // the MG re-derives the OPERATOR openness alpha (inflow Neumann -> closed) per level via
         // setBC.
      copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
      copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
      copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
#ifdef PECLET_FLOW_MPI
      if (distributed_)
        mg_.initMpi(gnx_, gny_, gnz_, nLevels_, comm_);
      else
#endif
        mg_.init(nx_, ny_, nz_,
                 nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
      mg_.setBoundaryConditions(
          bc_);  // per-level wall openness + null-space gating (no-op if periodic)
      mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), 1.0, 1.0, 1.0);
      Kokkos::deep_copy(phi_, 0.0);
      Kokkos::deep_copy(P_, 0.0);
    }
  }

  void step() {
    // Multiphysics: refresh material properties / body forces from the current fields (frozen over
    // the step). No-op (byte-identical) when no closure is registered.
    updateProperties();
    // Variable viscosity: rebuild the diffusion stencil from the current mu field (the implicit-FOU
    // path rebuilds it per Picard in buildAdvStencil, so only the non-advective path needs this).
    if (varProps_ && !implicitAdv())
      rebuildStencils();
    // u^n time base, fixed for the whole step (Picard lags the advecting velocity at u^k, not the
    // base).
    for (int c = 0; c < 3; ++c)
      Kokkos::deep_copy(old_[c], C[c].u);
    if (cutcellPressure_ && incremental_) {
      fillGhosts(P_);
      if (hasBc_)
        pressureBcGhost();
    }  // grad(P^n) for the incremental predictor (once)
    lastOuterIters_ = 0;
    for (int outer = 0; outer < outerIters_; ++outer) {
      lastOuterIters_ = outer + 1;
      if (outerTol_ > 0)
        for (int c = 0; c < 3; ++c)
          Kokkos::deep_copy(prev_[c], C[c].u);
      if (advect_ || hasBc_)
        for (int c = 0; c < 3; ++c)
          fillVelGhosts(c, 0);  // explicit ghosts (periodic + BC) for advect
      for (int c = 0; c < 3; ++c)
        hasCellForce_ ? buildRhsForced(c) : buildRhs(c);  // RHS from u^n base + advection lagged
                                                          // at u^k (+ per-cell body force)
      // Implicit-FOU: rebuild the IBM velocity stencil = backward-Euler diffusion + rho*FOU(u^k),
      // then re-apply the cut-cell bake. Per Picard iteration (advecting velocity changes). Applies to
      // the IBM (periodic/porous) path when the user opts in, AND ALWAYS to the domain-BC stencil path
      // (inflow/outflow) -- implicitAdv() -> fully-implicit upwind advection (stable at large dt). The
      // velocity-MG BC path keeps its own FOU coarse operator.
      if (implicitAdv() && (!hasBc_ || !useVelocityMg_))
        for (int c = 0; c < 3; ++c)
          buildAdvStencil(c);
      // Outflow backflow stabilization: dissipate reverse flow at the outlet in the momentum operator
      // used by the domain-BC stencil smoother (prevents backflow divergence). Inert without reversal.
      if (bcStencilPath() && backflowBeta_ > 0.0 && hasOutflow_)
        for (int c = 0; c < 3; ++c)
          applyBackflowStab(c);
      // upwind-convective velocity-MG: restrict the (frozen u^k) advecting velocity to the coarse
      // levels ONCE, before the per-component solves update it (shared across the 3 momentum
      // components).
      if (useVelocityMg_ && implicitFou_ && advect_ && !hasBc_)
        vmg_.restrictAdvVelocities(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u));
      for (int c = 0; c < 3; ++c)
        smoothComp(c);  // per-component IBM implicit-diffusion solve
      if (cutcellPressure_)
        project();  // cut-cell projection -> incompressible
      if (hasBc_)
        for (int c = 0; c < 3; ++c)
          applyVelocityBcComp(c, 0, false);  // re-impose domain BCs (keep outflow)
      if (outerTol_ > 0) {  // outer convergence: max velocity change over this Picard iteration
        double corr = 0.0;
        for (int c = 0; c < 3; ++c)
          corr = Kokkos::fmax(corr, maxAbsDiffInner(CCConst(C[c].u), CCConst(prev_[c])));
        lastOuterCorr_ = corr;
        if (corr < outerTol_)
          break;
      }
    }
    // Segregated multiphysics: advance any transported scalars with the just-projected
    // divergence-free velocity (properties frozen over the step). No-op (byte-identical) when no
    // scalar is registered.
    advanceScalars();
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) { return gatherInner(C[c].u); }
  // The divergence-free FACE velocity component (collocated: the projected MAC face field
  // uf_/vf_/wf_, exactly div-free; staggered: C[c].u already lives on the faces). For a periodic
  // bed its mean is the momentum-balance superficial velocity, unperturbed by the openness-aware
  // cell gradient correction (projectCorrectCenter) that biases the cell-field mean at cut cells.
  std::vector<double> getFaceVelocity(int c) {
    if constexpr (Grid::collocated) {
      CCField fa[3] = {uf_, vf_, wf_};
      return gatherInner(fa[c]);
    } else {
      return gatherInner(C[c].u);
    }
  }
  // TEMP DIAGNOSTIC: the face openness (fluid area fraction) used by the cut-cell projection.
  // component c: 0 -> ox_ (low -x face of each inner cell), 1 -> oy_, 2 -> oz_. Grid-independent
  // (built once from the SDF). Exposed to compare the open-weighted superficial flux against the
  // raw velocity mean.
  std::vector<double> getOpenness(int c) {
    CCField o[3] = {ox_, oy_, oz_};
    return gatherInner(o[c]);
  }
  std::vector<double> getPressure() {
    // Incremental scheme: P_ accumulates the physical pressure. Classical Chorin (!incremental_):
    // derive it on demand from the last projection potential, p = (rho/dt)*phi (CUDA
    // press_from_phi_k).
    if (incremental_)
      return gatherInner(P_);
    std::vector<double> out = gatherInner(phi_);
    const double ct = rho_ / dt_;
    for (double& x : out)
      x *= ct;
    return out;
  }
  double maxOpenDivergence() {
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
          if (bc_[2 * a + 1] == 3)
            bcNeumannGhost(fa[a], e, G, a, 1);
      }
      divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_), CCConst(oz_),
                 div_, e_, G);
    } else {
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);  // ghosts incl. outflow zero-gradient before the divergence
      divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                 CCConst(oz_), div_, e_, G);
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
  long lastPressureIterations() const { return lastPressureIters_; }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }

 private:
  struct Comp {
    CCField u, b, inhom, rscale, mask;
    FV AC, AW, AE, AS, AN, AB, AT;
    IbmOverlay ov;
    Kokkos::View<int*, CCMem> idMap;
    Kokkos::View<int, CCMem> counter;
    int nCut = 0;
  };

 public:  // nvcc forbids extended __host__ __device__ lambdas inside private/protected members.
  // Advection treated implicitly (implicit-FOU upwind + deferred correction): the user opt-in
  // (set_implicit_advection) on any path, OR the DEFAULT on the domain-BC path (inflow/outflow),
  // where explicit advection is unstable. The velocity-MG BC path carries its own FOU coarse operator,
  // so the default does not apply there (it still honours the explicit opt-in).
  bool implicitAdv() const { return advect_ && (implicitFou_ || (hasBc_ && !useVelocityMg_)); }
  // Domain-BC momentum solved via the Robust-Scaled cut-cell / FOU stencil smoother (ibmRbgsStencilColor
  // + reflection-ghost BCs), not the all-fluid const-coeff fold. Needed when (a) an immersed solid is
  // present (cut-cell no-slip must be in the operator), or (b) advection is implicit (the FOU upwind
  // lives in the stencil -> stable at large dt, the fully-implicit design).
  bool bcStencilPath() const {
    return hasBc_ && !useVelocityMg_ && (hasSolid_ || implicitAdv() || varProps_);
  }
  // Fill the mu-field ghosts for the face means: periodic/halo base, then zero-gradient (copy) on
  // domain-BC (wall/inflow/outflow) faces — a periodic wrap there would bring the wrong layer's mu
  // to the wall face (destabilising, especially for the harmonic mean).
  void fillMuGhosts() {
    fillGhosts(muField_);
    if (!distributed_)
      for (int f = 0; f < 6; ++f)
        if (bc_[f] != 0)
          applyScalarBcFace(muField_, f / 2, f % 2, 1, 0.0);  // type 1 = Neumann copy
  }
  void rebuildStencils() {
    const double idiag = rho_ / dt_, beta = mu_;
    if (varProps_)
      fillMuGhosts();  // face means read mu at i +- stride (boundary inner cells -> ghosts)
    for (int c = 0; c < 3; ++c) {
      Kokkos::deep_copy(C[c].rscale, 1.0);
      Kokkos::deep_copy(C[c].inhom, 0.0);
      if (varProps_)
        ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x,
                             e_.y, e_.z, G,
                             FieldFaceProps{CCConst(muField_), idiag, harmonicMu_});
      else
        ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x, e_.y,
                          e_.z, beta, idiag);
      ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                       C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
    }
  }
  // copy the nx*ny*nz inner cells between two extended blocks of different ghost width (g=2 <-> g=1
  // MG).
  void copyInner(CCField dst, C3 de, int dg, CCConst src, C3 se, int sg) {
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
  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners). Distributed:
  // the velocity-block halo (cross-rank + periodic, all ghosts incl. corners).
  void fillGhosts(CCField f) {
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
  // Fused periodic FACE-ghost fill in ONE kernel (vs 3 fillAxis): each inner boundary cell scatters
  // its periodic image to the opposite face ghost, all 3 axes at once. Valid only for
  // FACE-neighbour (7-point) stencils -- it does NOT fill the corner/edge ghosts (which fillAxis's
  // sequential x->y->z does). The IBM RB-GS smoother reads only the 7-point stencil, so this is
  // exact there and cuts the velocity solve's dominant kernel-launch cost (~7200 -> ~2400 fill
  // launches/step) at low resolution. NOT for the Koren advection RHS (reads diagonals) -- keep the
  // full fillGhosts there.
  void fillGhostsFaces(CCField f) {
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
  void fillAxis(CCField f, int axis) {
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
  void buildRhs(int c) {
    CCExec space;
    const double idiag = rho_ / dt_, fc = f_[c], rho = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u), uu = CCConst(C[c].u),
            un = CCConst(old_[c]);
    const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
    // Pure implicit FOU (no deferred correction): 1st-order upwind carried entirely by the operator,
    // no explicit high-order term in the RHS -- maximally dissipative/stable (diffuses sharp shear
    // layers). Only meaningful on an implicit-advection path.
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();  // fold RHS only on the const-coeff domain-BC path;
    // on the stencil path (solid and/or implicit advection) the walls enter via reflection ghosts
    // (smoothComp) and the RHS carries the IBM inhom (=0 for no-slip) + the deferred correction. incr
    // predictor carries -grad(P^n).
    const bool ifou =
        implicitAdv() && deferredCorr_;  // deferred correction: keep (HO - FOU) explicit in the RHS
                                         // (implicit on the domain-BC path by default, opt-in elsewhere)
    const int sch = advScheme_;   // 0 = SOU (default), 1 = Koren TVD
    // b = descale*(idiag*u^n - rho*Koren(u^k) + rho*FOU(u^k) + f - grad P^n) - inhom  (+ BC fold
    // brhs). The time base is u^n (Picard); the advecting velocity & advected field are the current
    // iterate u^k.
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{uu, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          // incremental predictor's -grad(P^n): central-difference cell gradient on the collocated
          // grid, one-sided face gradient (P at the high cell of the staggered face) on the
          // staggered grid.
          const double gp = !incr              ? 0.0
                            : Grid::collocated ? 0.5 * (P((long)i + strd) - P((long)i - strd))
                                               : (P(i) - P((long)i - strd));
          bb(i) =
              rs(i) * (idiag * un(i) + fc - rho * aK + rho * aF - gp) + (bc ? brhs(i) : -inh(i));
        });  // BC fold (brhs) on the domain-BC path; -inhom on the IBM path (=0 for no-slip)
  }
  // Sibling of buildRhs adding a per-cell body force fb(i) (Boussinesq buoyancy / CFD-DEM feedback):
  // the constant fc becomes fc + fb(i). Kept as a separate kernel so buildRhs stays byte-identical
  // (no codegen drift on the single-phase path). Selected in step() when hasCellForce_.
  void buildRhsForced(int c) {
    CCExec space;
    const double idiag = rho_ / dt_, fc = f_[c], rho = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst fb = CCConst(cellForce_[c]);
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u), uu = CCConst(C[c].u),
            un = CCConst(old_[c]);
    const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();
    const bool ifou = implicitAdv() && deferredCorr_;
    const int sch = advScheme_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs_forced", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{uu, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          const double gp = !incr              ? 0.0
                            : Grid::collocated ? 0.5 * (P((long)i + strd) - P((long)i - strd))
                                               : (P(i) - P((long)i - strd));
          bb(i) = rs(i) * (idiag * un(i) + fc + fb(i) - rho * aK + rho * aF - gp) +
                  (bc ? brhs(i) : -inh(i));
        });
  }
  // Implicit-FOU velocity stencil (CUDA build_adv_stencil_k + ibm_modify_stencil): backward-Euler
  // diffusion (idiag+6beta diag, -beta off) + rho*FOU(u^k) upwind operator (diagonally dominant ->
  // stable at high Re), then the Robust-Scaled cut-cell bake. The advecting velocity u^k = the
  // current C[*].u (ghosts filled).
  void buildAdvStencil(int c) {
    const double idiag = rho_ / dt_, beta = mu_, fouw = rho_;
    C3 e = e_;
    if (varProps_) {
      if (c == 0)
        fillMuGhosts();
      ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y,
                           e.z, G, FieldFaceProps{CCConst(muField_), idiag, harmonicMu_});
    } else
      ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y, e.z,
                        beta, idiag);
    CCExec space;
    FV AC = C[c].AC, AW = C[c].AW, AE = C[c].AE, AS = C[c].AS, AN = C[c].AN, AB = C[c].AB,
       AT = C[c].AT;
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "advstencil", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
                 czp = AT(i);
          sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
          Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
          AC(i) = (float)cC;
          AW(i) = (float)cxm;
          AE(i) = (float)cxp;
          AS(i) = (float)cym;
          AN(i) = (float)cyp;
          AB(i) = (float)czm;
          AT(i) = (float)czp;
        });

    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
  }
  // Backflow stabilization (Bazilevs 2009 / Esmaily-Moghadam 2011) for the NORMAL momentum at outflow
  // faces: add the dissipative diagonal term beta*rho*|min(u.n,0)| where the outflow reverses (fluid
  // re-entering, u.n<0). This removes the spurious kinetic-energy influx that the do-nothing/zero-
  // gradient outflow advects in -- the "backflow divergence" that blows up separated flows (e.g. the
  // BFS recirculation reaching the outlet), worse on finer grids. Purely dissipative (u_ext=0), so it
  // is implicit + unconditionally stable, and INERT where the outlet is outgoing (u.n>=0) -> the
  // channel and any non-reversing outflow stay byte-identical. Applied to C[c].AC after buildAdvStencil
  // (per Picard iteration, lagged at u^k); only the component normal to each outflow face.
  void applyBackflowStab(int c) {
    if (backflowBeta_ <= 0.0 || !hasOutflow_)
      return;
    CCExec space;
    const double beta = backflowBeta_, rho = rho_;
    C3 e = e_;
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    FV AC = C[c].AC;
    CCConst u = CCConst(C[c].u);
    const int a = c;  // the normal component of a face on axis a is component a
    for (int s = 0; s < 2; ++s) {
      if (bc_[2 * a + s] != 3)
        continue;  // outflow faces only
      const long sa = st[a];
      const int na = dims[a];
      const int bic = (s == 0) ? G : (na - G - 1);  // outflow-adjacent inner normal-velocity cell
      const double sgn = (s == 0) ? 1.0 : -1.0;     // reversal (u.n<0): u>0 at -a, u<0 at +a
      const int b = (a + 1) % 3, cc = (a + 2) % 3;
      const long sb = st[b], sc = st[cc];
      using MD2 = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>;
      Kokkos::parallel_for(
          "peclet::flow::backflow", MD2(space, {G, G}, {dims[b] - G, dims[cc] - G}),
          KOKKOS_LAMBDA(int p0, int p1) {
            const long i = (long)p0 * sb + (long)p1 * sc + (long)bic * sa;
            const double back = sgn * u(i);  // > 0 exactly where the outflow reverses (|min(u.n,0)|)
            if (back > 0.0)
              AC(i) += (float)(beta * rho * back);  // dissipative diagonal (u_ext = 0)
          });
    }
  }
  // max|a-b| over inner cells (Picard outer-tolerance check).
  double maxAbsDiffInner(CCConst a, CCConst b) {
    CCExec space;
    C3 e = e_;
    double m = 0;
    Kokkos::parallel_reduce(
        "maxdiff",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double d = Kokkos::fabs(a(i) - b(i));
          if (d > acc)
            acc = d;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  void smoothComp(int c) {
    if constexpr (Grid::collocated) {
      if (hasBc_) {  // collocated domain BC: the (all-fluid) IBM diffusion stencil + cell-centered
                     // wall
        // reflection ghosts refreshed each colour (explicit no-slip; no fold). Converges to the
        // wall value.
        for (int it = 0; it < velIters_; ++it) {
          fillVelGhostsTo(C[c].u, c, 0);
          ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                              MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                              MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
          fillVelGhostsTo(C[c].u, c, 0);
          ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                              MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                              MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
        }
        return;
      }
    }
    if (bcStencilPath()) {
      // Domain BCs solved with the Robust-Scaled cut-cell / FOU stencil (built by setSolid /
      // buildAdvStencil) while refreshing the domain-BC ghosts each colour -- explicit walls/inflow
      // (reflection, fold=0) + outflow zero-gradient. Mirrors the collocated path above. Used for an
      // immersed solid (cut-cell no-slip in the operator) and/or implicit advection (FOU upwind in the
      // stencil -> stable at large dt). The const-coeff fold smoothers below are all-fluid,
      // diffusion-only: they ignore the solid AND run advection explicitly (CFL-limited).
      for (int it = 0; it < velIters_; ++it) {
        fillVelGhostsTo(C[c].u, c, 0);
        ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                            MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                            MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
        fillVelGhostsTo(C[c].u, c, 0);
        ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                            MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                            MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
      }
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
      vmg_.setBcApplyL0([this, c](CCField x) { fillVelGhostsTo(x, c, 1); });
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      return;
    }
    if (hasBc_) {  // domain-BC (no immersed solid): CUDA's double const-coeff diff_k + dcorr fold
      const I3 e{e_.x, e_.y, e_.z}, og{0, 0, 0};
      const double beta = mu_, Ac = rho_ / dt_ + 6.0 * mu_;
      for (int it = 0; it < velIters_; ++it) {
        fillVelGhosts(c, 1);  // re-impose wall faces (fold) before each color
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 0, CCConst(bcDcorr_[c]));
        fillVelGhosts(c, 1);
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 1, CCConst(bcDcorr_[c]));
      }
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
        ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, off);
        ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, off);
        vmg_.setStaircase(CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_,
                          rho_ / dt_, 0.5);
      }
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      maskVelocity(
          c);  // re-impose no-slip at solid (the masked solve leaves them at the pin value)
      return;
    }
    for (int it = 0; it < velIters_;
         ++it) {  // IBM / periodic: Robust-Scaled cut-cell stencil (float)
      fillGhostsFaces(
          C[c].u);  // 7-point smoother reads faces only -> the fused 1-kernel face fill suffices
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                          MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                          MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
      fillGhostsFaces(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                          MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                          MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
    }
  }
  // pressure ghost at domain faces for the incremental predictor's grad(P): zero-gradient (Neumann)
  // at every non-periodic face so grad(P) carries no spurious force there (the periodic fill
  // wrapped the opposite boundary's pressure). Outflow pressure (Dirichlet p=0) is enforced
  // separately in the MG solve.
  void pressureBcGhost() {
    CCExec space;
    C3 e = e_;
    CCField P = P_;
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        if (bc_[2 * a + s] == 0)
          continue;
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        const long sa = st[a], sb = st[b], sc = st[c];
        const int na = dims[a];
        const int bic = (s == 0) ? G : (na - G - 1);
        const int lo = (s == 0) ? 0 : (na - G), hi = (s == 0) ? (G - 1) : (na - 1);
        Kokkos::parallel_for(
            "pbcghost",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
            KOKKOS_LAMBDA(int p0, int p1) {
              const long base = (long)p0 * sb + (long)p1 * sc;
              const double pin = P(base + (long)bic * sa);
              for (int ia = lo; ia <= hi; ++ia)
                P(base + (long)ia * sa) = pin;
            });
      }
  }
  // domain-BC velocity ghosts: periodic-fill periodic axes, then apply per-face BCs (fold=0
  // explicit/1 implicit).
  void fillVelGhosts(int comp, int fold) { fillVelGhostsTo(C[comp].u, comp, fold); }
  void applyVelocityBcComp(int comp, int fold, bool doOutflow) {
    applyVelocityBcCompTo(C[comp].u, comp, fold, doOutflow);
  }
  // Field-parameterized variants (so the velocity-MG can re-impose the BC on its own level-0
  // iterate).
  void fillVelGhostsTo(CCField f, int comp, int fold) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      velDev_->exchange(f);
      applyVelocityBcCompTo(f, comp, fold, true);
      return;
    }
#endif
    for (int a = 0; a < 3; ++a)
      if (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)
        fillAxis(f, a);
    applyVelocityBcCompTo(f, comp, fold, true);
  }
  void applyVelocityBcCompTo(CCField f, int comp, int fold, bool doOutflow) {
    if (!hasBc_)
      return;
    B3 e{e_.x, e_.y, e_.z};
    if constexpr (Grid::collocated) {
      // Cell-centered velocity: reflect this component about each non-periodic boundary face. Walls
      // (type 1, vel 0) and Dirichlet/lid (type 2, prescribed vel) both use the same reflection;
      // outflow (type 3) and per-position inlet profiles are the inflow/outflow milestone (phase
      // 5b).
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
          const int ff = 2 * a + s;
          const int t = bc_[ff];
          if (t == 0)
            continue;
          if (t == 3) {
            if (doOutflow)
              bcNeumannGhost(f, e, G, a, s);
            continue;
          }  // outflow: zero-gradient ghost
          if (bcProf_[ff].extent(0) >
              0)  // per-position inlet profile (e.g. the BFS partial parabola)
            bcVelocityColocated(f, e, G, a, s, 0.0, comp, bcProf_[ff], bcProfNc_[ff]);
          else
            bcVelocityColocated(f, e, G, a, s,
                                bcVel_[ff][comp]);  // wall / inflow / lid (Dirichlet)
        }
      return;
    }
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int ff = 2 * a + s;
        const int t = bc_[ff];
        if (t == 0)
          continue;
        if (t == 3) {
          if (doOutflow)
            bcOutflowComp(f, e, G, a, s, comp, fold);
          continue;
        }
        if (bcProf_[ff].extent(0) > 0)
          bcVelocityComp(f, e, G, a, s, comp, 0.0, fold, bcProf_[ff], bcProfNc_[ff]);
        else
          bcVelocityComp(f, e, G, a, s, comp, bcVel_[ff][comp], fold);
      }
  }
  // implicit-diffusion wall fold (CUDA setup_bc_diffusion): dcorr += (wall:+beta tangential /
  // outflow:-beta), brhs += 2*beta*wall (tangential Dirichlet); bake dcorr into the per-component
  // stencil diagonal.
  void setupBcDiffusion() {
    const double beta = mu_;
    B3 e{e_.x, e_.y, e_.z};
    for (int c = 0; c < 3; ++c) {
      Kokkos::deep_copy(bcDcorr_[c], 0.0);
      Kokkos::deep_copy(bcBrhs_[c], 0.0);
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
          const int t = bc_[2 * a + s];
          double dval, bval;
          if (t == 3) {
            dval = -beta;
            bval = 0.0;
          } else if (t != 0 && c != a) {
            dval = beta;
            bval = 2.0 * beta * bcVel_[2 * a + s][c];
          } else
            continue;  // periodic, or the normal component at a wall (held directly)
          bcDiffusionFold(bcDcorr_[c], bcBrhs_[c], e, G, a, s, dval, bval);
        }
      // dcorr is passed to the (double) const-coeff smoother diffSmoothColor each sweep -- matching
      // CUDA diff_k (Ac + dcorr in double), NOT baked into the float stencil.
    }
  }
  // Incremental (rotational) cut-cell projection: solve A phi = -div_open(u*) (RB-GS,
  // mean-removed), u -= grad phi, then accumulate the physical pressure P += (rho/dt)*phi -
  // mu*div(u*) (Timmermans).
  void project() {
    // ghosts incl. domain BCs (outflow zero-gradient) BEFORE the divergence -- matches CUDA
    // apply_velocity_bc before diverg_open, so div(u*) counts the outflow flux (else the rotational
    // pressure pumps the mis-counted outflow divergence and blows up the outflow-wall corner).
    if constexpr (Grid::collocated) {
      // Approximate (MAC) projection: average the cell velocities onto a face field, then project
      // THAT. Use the BC-aware ghost fill (periodic / cross-rank + domain BCs) so the averaged
      // inflow/outflow faces carry the right value -- at open boundaries the flux is counted
      // (closed walls are openness 0).
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);
      if (faceInterp_ == 1)  // wall-aware reconstruction at solid-bordering faces (setFaceInterp)
        centerToFaceWallAware(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u),
                              CCConst(sdf_), e_, G);
      else
        centerToFace(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), e_, G);
      divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_), CCConst(oz_),
                 div_, e_, G);
    } else {
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);
      divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                 CCConst(oz_), div_, e_, G);
    }
    // bridge -div(u*) (g=2 block) -> the MG rhs (g=1 block); keep div(u*) in div_ for the pressure
    // update
    copyInner(rhs1_, e1_, 1, CCConst(div_), e_, G);
    {
      CCExec space;
      CCField r = rhs1_;
      Kokkos::parallel_for(
          "negdiv", Kokkos::RangePolicy<CCExec>(space, 0, n1_),
          KOKKOS_LAMBDA(std::size_t i) { r(i) = -r(i); });
    }
    // geometric multigrid solve of the cut-cell pressure Poisson A phi = -div(u*) (CUDA
    // mac_multigrid): MG-PCG by default, or the communication-light Chebyshev driver (bounds
    // estimated once, then reused). Warm start (CUDA pwarm_): keep the previous step's phi1_ as the
    // initial guess instead of zeroing.
    if (!pwarm_)
      Kokkos::deep_copy(phi1_, 0.0);
    if (useChebyshev_) {
      if (!chebBoundsSet_) {
        mg_.estimateEigenvalues(CCConst(rhs1_), chebA_, chebB_, 15, 2, 2, 12);
        chebBoundsSet_ = true;
      }
      lastPressureIters_ =
          mg_.solveChebyshev(rhs1_, phi1_, chebMaxit_, chebRtol_, 2, 2, 12, chebA_, chebB_);
    } else {
      lastPressureIters_ =
          mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12);
    }
    copyInner(phi_, e_, G, CCConst(phi1_), e1_, 1);  // bridge phi back g=1 -> g=2
    fillGhosts(phi_);
    if (hasOutflow_) {  // hold phi=0 at the outflow ghost so grad(phi) drives the outflow face
                        // (Dirichlet p=0)
      B3 e{e_.x, e_.y, e_.z};
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s)
          if (bc_[2 * a + s] == 3)
            bcZeroPressureGhost(phi_, e, G, a, s);
    }
    if constexpr (Grid::collocated) {
      // phi: zero-gradient (Neumann) at non-periodic walls so the cell-centered central-difference
      // correction carries no spurious normal acceleration through the wall (the periodic fill
      // wrapped the opposite boundary's phi). Outflow (Dirichlet p=0) is handled by hasOutflow_
      // above (phase 5b).
      if (hasBc_) {
        B3 e{e_.x, e_.y, e_.z};
        for (int a = 0; a < 3; ++a)
          for (int s = 0; s < 2; ++s) {
            const int t = bc_[2 * a + s];
            if (t != 0 && t != 3)
              bcNeumannGhost(phi_, e, G, a, s);
          }
      }
      // Correct the face field (-> discretely divergence-free; transient this step) and the cell
      // field (central-difference cell gradient).
      projectCorrect(uf_, vf_, wf_, CCConst(phi_), e_, G);
      fillGhosts(uf_);
      fillGhosts(vf_);
      fillGhosts(wf_);    // complete the divergence-free face field (boundary faces)
      if (hasOutflow_) {  // correct the high-side outflow face on the face field so mass leaves
                          // (phi=0 there)
        B3 e{e_.x, e_.y, e_.z};
        CCField fa[3] = {uf_, vf_, wf_};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3)
            bcCorrectOutflow(fa[a], phi_, e, G, a);
      }
      projectCorrectCenter(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(ox_), CCConst(oy_),
                           CCConst(oz_), e_, G);
    } else {
      projectCorrect(C[0].u, C[1].u, C[2].u, CCConst(phi_), e_, G);
      if (hasOutflow_) {  // correct the high-side outflow normal face that projectCorrect misses
                          // (mass leaves)
        B3 e{e_.x, e_.y, e_.z};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3)
            bcCorrectOutflow(C[a].u, phi_, e, G, a);
      }
    }
    // the grad(phi) correction also touches solid faces; re-impose no-slip there so the decoupled
    // solid velocity cannot accumulate (matches the CUDA apply_mask/mask_k after correct_k ->
    // stability).
    for (int c = 0; c < 3; ++c)
      maskVelocity(c);
    // Rotational incremental pressure (Timmermans), matching CUDA press_update_k: P += (rho/dt)*phi
    // - mu*div(u*). Classical non-incremental Chorin (!incremental_) skips the accumulation;
    // getPressure() derives p from phi.
    if (incremental_) {
      CCExec space;
      CCField P = P_, ph = phi_, d = div_;
      const double ct = rho_ / dt_, mu = mu_;
      if (varProps_) {
        // Variable viscosity: the pointwise Timmermans term -mu(i)*div(u*) is inconsistent for
        // heterogeneous mu (see setVariableRotational). Default = constant coefficient chi*mu_min
        // (stable by domination, exact fallback to the uniform-mu scheme); "full" = pointwise
        // (mild contrast only); "off" = plain incremental.
        if (varRotMode_ == 1) {
          CCConst mf = CCConst(muField_);
          const double chi = varRotChi_;
          Kokkos::parallel_for(
              "press_var_full", Kokkos::RangePolicy<CCExec>(space, 0, n_),
              KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - chi * mf(i) * d(i); });
        } else {
          const double muRot = (varRotMode_ == 2) ? 0.0 : varRotChi_ * minMuInner();
          Kokkos::parallel_for(
              "press_var_min", Kokkos::RangePolicy<CCExec>(space, 0, n_),
              KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - muRot * d(i); });
        }
      } else {
        Kokkos::parallel_for(
            "press", Kokkos::RangePolicy<CCExec>(space, 0, n_),
            KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - mu * d(i); });
      }
    }
  }
  void maskVelocity(int c) {
    CCExec space;
    CCField u = C[c].u, m = C[c].mask;
    Kokkos::parallel_for(
        "vmask", Kokkos::RangePolicy<CCExec>(space, 0, n_), KOKKOS_LAMBDA(std::size_t i) {
          if (m(i) > 0.5)
            u(i) = 0.0;
        });
  }
  // Minimum viscosity over the (global, under MPI) inner cells — the provably-stable rotational
  // coefficient for variable viscosity (chi*mu_min <= mu(x) everywhere).
  double minMuInner() {
    CCExec space;
    C3 e = e_;
    CCConst f = CCConst(muField_);
    double m = 1e300;
    Kokkos::parallel_reduce(
        "minmu",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
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
  double reduceMaxAbsInner(CCConst f) {
    CCExec space;
    C3 e = e_;
    double m = 0;
    Kokkos::parallel_reduce(
        "maxabs",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double a = Kokkos::fabs(f(i));
          if (a > acc)
            acc = a;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  std::vector<double> gatherInner(CCField fld) {
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
  // Inverse of gatherInner: scatter an x-fastest (nx,ny,nz) inner-region host buffer into the inner
  // cells of a ghosted G=2 field (ghost cells untouched — refill via exchangeField/fillGhosts).
  void scatterInner(CCField fld, const std::vector<double>& in) {
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

  // --- Named field registry (multiphysics field container) ------------------------------------
  // Register a new zero-initialised cell-centred field on the G=2 velocity block and return its
  // buffer. Idempotent: re-adding an existing name returns the existing buffer unchanged.
  CCField addField(const std::string& name) {
    if (fields_.has(name))
      return fields_.at(name).data;
    return fields_.add(name, n_, G, peclet::core::Centering::Cell).data;
  }
  bool hasField(const std::string& name) const { return fields_.has(name); }
  CCField fieldView(const std::string& name) { return fields_.at(name).data; }
  std::vector<std::string> fieldNames() const { return fields_.names(); }
  // Ghost-exchange a registered field (cross-rank + periodic under MPI; periodic-only single-rank).
  void exchangeField(const std::string& name) { fillGhosts(fields_.at(name).data); }
  // Host round-trip: read a registered field's inner region as an x-fastest (nx,ny,nz) buffer, or
  // write one (ghosts left stale until the next exchangeField).
  std::vector<double> getField(const std::string& name) { return gatherInner(fields_.at(name).data); }
  void setField(const std::string& name, const std::vector<double>& v) {
    scatterInner(fields_.at(name).data, v);
  }
  // Padded-block extents + ghost width, so a zero-copy field buffer (size ex*ey*ez, x-fastest) can
  // be reshaped in Python.
  std::array<int, 3> blockShape() const { return {e_.x, e_.y, e_.z}; }
  int ghostWidth() const { return G; }

  // --- Scalar transport (advection-diffusion) -------------------------------------------------
  // Register a transported scalar `name` with constant diffusivity D (grid units). scheme: 0 FOU,
  // 1 Koren TVD (default), 2 SOU. iters = RB-GS sweeps for the implicit diffusion solve. Its field
  // is registered in the directory (get_field/set_field/field_view). Openness (set_solid /
  // set_pressure_geometry) must be established for transport to occur.
  void addScalar(const std::string& name, double D, int scheme, int iters) {
    ScalarField sc;
    sc.name = name;
    sc.c = addField(name);  // registered, zero-initialised, on the G=2 block
    sc.cOld = CCField(name + "_old", n_);
    sc.b = CCField(name + "_b", n_);
    sc.AC = CCField(name + "_AC", n_);
    sc.AW = CCField(name + "_AW", n_);
    sc.AE = CCField(name + "_AE", n_);
    sc.AS = CCField(name + "_AS", n_);
    sc.AN = CCField(name + "_AN", n_);
    sc.AB = CCField(name + "_AB", n_);
    sc.AT = CCField(name + "_AT", n_);
    sc.D = D;
    sc.scheme = scheme;
    sc.iters = iters < 1 ? 1 : iters;
    scalars_.push_back(sc);
  }
  bool hasScalar(const std::string& name) const {
    for (const auto& sc : scalars_)
      if (sc.name == name)
        return true;
    return false;
  }
  // Per-face scalar BC: face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic, 1 Neumann zero-flux
  // (adiabatic), 2 Dirichlet value. Single-rank / non-decomposed domains (distributed BC deferred).
  void setScalarBc(const std::string& name, int face, int type, double value) {
    for (auto& sc : scalars_)
      if (sc.name == name) {
        sc.bc[face] = type;
        sc.bcVal[face] = value;
        return;
      }
    throw std::runtime_error("set_scalar_bc: no scalar named '" + name + "'");
  }
  // Advance all registered scalars one dt with the current divergence-free velocity (also called at
  // the end of step()). Exposed so a test can prescribe a velocity and transport a scalar in
  // isolation.
  void advanceScalars() {
    if (scalars_.empty())
      return;
    const double idt = 1.0 / dt_;
    CCField Uf, Vf, Wf;
    if constexpr (Grid::collocated) {
      Uf = uf_;
      Vf = vf_;
      Wf = wf_;
    } else {
      Uf = C[0].u;
      Vf = C[1].u;
      Wf = C[2].u;
    }
    fillGhosts(Uf);
    fillGhosts(Vf);
    fillGhosts(Wf);  // face velocities need the ±2 advection reach
    for (auto& sc : scalars_) {
      scalarBuildDiffusionOpen(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(ox_),
                               CCConst(oy_), CCConst(oz_), sc.D, idt, e_, G);
      applyScalarBcStencil(sc);  // re-open Dirichlet domain faces (set_domain_bc closes openness)
      Kokkos::deep_copy(sc.cOld, sc.c);
      scalarFillGhosts(sc);
      scalarBuildRhs(sc.b, CCConst(sc.cOld), CCConst(Uf), CCConst(Vf), CCConst(Wf), CCConst(ox_),
                     CCConst(oy_), CCConst(oz_), idt, sc.scheme, e_, G);
      // implicit diffusion: red-black Gauss-Seidel with a ghost fill before each color sweep.
      for (int it = 0; it < sc.iters; ++it) {
        scalarFillGhosts(sc);
        cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                           og_, G, 0);
        scalarFillGhosts(sc);
        cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                           og_, G, 1);
      }
      scalarFillGhosts(sc);
    }
  }

  // --- Property closures + per-cell body force ------------------------------------------------
  // Register a property/force closure. target: a registered field name — a material property
  // ("mu"/"rho"/…) or a body-force component ("force_x"/"force_y"/"force_z"). kind: LinearMix /
  // BoussinesqForce / ArrheniusMu. in0/in1: input field names (in1 "" if unused). params: up to 4
  // doubles (meaning per kind — property_closures.hpp). Applied at the top of step() in registration
  // order. Targeting a force component turns on the per-cell body-force RHS path.
  void setPropertyModel(const std::string& target, ClosureKind kind, const std::string& in0,
                        const std::string& in1, const std::vector<double>& params) {
    Closure cl;
    cl.kind = kind;
    cl.out = ensureTarget(target);
    cl.in0 = CCConst(fields_.at(in0).data);
    if (!in1.empty())
      cl.in1 = CCConst(fields_.at(in1).data);
    for (int k = 0; k < 4 && k < (int)params.size(); ++k)
      cl.p[k] = params[k];
    closures_.push_back(cl);
    if (target == "mu")  // a closure driving mu turns on variable viscosity
      setPropertyMode(true, harmonicMu_);
  }
  // Enable/disable variable-coefficient momentum (variable viscosity). variable=true binds the "mu"
  // field (creating it, seeded with the current scalar mu, if absent) and forces the stencil solve
  // path. harmonic selects the harmonic face mean (continuous shear stress across a viscosity jump)
  // vs arithmetic. Escape hatch: set_field("mu", arr) then set_property_mode(True).
  void setPropertyMode(bool variable, bool harmonic) {
    varProps_ = variable;
    harmonicMu_ = harmonic;
    if (variable) {
      if (fields_.has("mu"))
        muField_ = fields_.at("mu").data;
      else {
        muField_ = addField("mu");
        Kokkos::deep_copy(muField_, mu_);  // default to the scalar mu until a closure/set_field sets it
      }
      useVelocityMg_ = false;  // the velocity multigrid takes a scalar mu (variable-coeff vmg deferred)
    }
  }
  // Rotational-pressure treatment under variable viscosity. The Timmermans rotational term
  // P += (rho/dt)phi - mu*div(u*) is only valid for HOMOGENEOUS viscosity (Deteix & Yakoubi, Appl.
  // Math. Lett. 2018 / arXiv:1902.05643): with spatially varying mu the pointwise term is no longer
  // the gradient part of the viscous stress, and the accumulated inconsistency destabilises the
  // incremental scheme at strong contrast (observed: 10x jump + harmonic faces -> divergence).
  // Modes (the incremental predictor -grad(P^n) and P accumulation are kept in ALL of them — that is
  // what enables large-dt / steady-Stokes stepping):
  //   0 "min"  (default): rotational coefficient chi*mu_min — a CONSTANT dominated by the true local
  //            dissipation everywhere (mu_min <= mu(x)), so the constant-viscosity stability theory
  //            carries over; reduces EXACTLY to the validated scheme when mu is uniform.
  //   1 "full": chi*mu(i) pointwise — better pressure consistency at MILD contrast; not stable at
  //            strong contrast (user's responsibility).
  //   2 "off" : plain incremental (no rotational term) — unconditionally stable, keeps the artificial
  //            pressure Neumann layer of the non-rotational scheme.
  // The fully consistent variable-viscosity correction (shear-rate projection: an extra Poisson
  // solve for psi with rhs div(div(2 nu D(u)))) is deferred.
  void setVariableRotational(int mode, double chi) {
    varRotMode_ = mode < 0 ? 0 : (mode > 2 ? 2 : mode);
    varRotChi_ = chi < 0.0 ? 0.0 : chi;
  }
  // Tabulated property: out = piecewise-linear interp of (xs, ys) at the input field (xs ascending).
  void setPropertyTable(const std::string& target, const std::string& in0,
                        const std::vector<double>& xs, const std::vector<double>& ys) {
    Closure cl;
    cl.kind = ClosureKind::Table1D;
    cl.out = ensureTarget(target);
    cl.in0 = CCConst(fields_.at(in0).data);
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
  // Apply all closures (also called at the top of step()). Exposed for testing.
  void updateProperties() {
    for (auto& cl : closures_)
      applyClosure(cl, e_, G);
  }

 private:
  // Resolve a closure target to a registered buffer. A force component allocates ALL three
  // cellForce_ slots (buildRhsForced reads every component) and enables the body-force RHS path.
  CCField ensureTarget(const std::string& name) {
    if (name == "force_x" || name == "force_y" || name == "force_z")
      ensureCellForceAll();
    return addField(name);  // idempotent; returns the (now-existing) buffer
  }
  void ensureCellForceAll() {
    static const char* fn[3] = {"force_x", "force_y", "force_z"};
    for (int c = 0; c < 3; ++c)
      cellForce_[c] = addField(fn[c]);  // zero-initialised, registered
    hasCellForce_ = true;
  }
  // Ghost fill for a scalar: periodic (single-rank) / MPI halo base, then override any domain
  // Dirichlet/Neumann faces.
  void scalarFillGhosts(ScalarField& sc) {
    fillGhosts(sc.c);
    applyScalarBc(sc);
  }
  // Overwrite the ghost band on each Dirichlet/Neumann domain face (both layers, for the ±2
  // advection reach). Single-rank only — distributed domain-BC scalars are a later phase.
  void applyScalarBc(ScalarField& sc) {
    bool any = false;
    for (int f = 0; f < 6; ++f)
      any = any || (sc.bc[f] != 0);
    if (!any || distributed_)
      return;
    for (int f = 0; f < 6; ++f)
      if (sc.bc[f] != 0)
        applyScalarBcFace(sc.c, f / 2, f % 2, sc.bc[f], sc.bcVal[f]);
  }
  // Re-open the diffusion face at a Dirichlet domain boundary: set_domain_bc closes the boundary
  // openness (ox_=0), which correctly makes Neumann/adiabatic walls zero-flux but would also cut a
  // Dirichlet wall's heat path. For each Dirichlet face, restore the face coefficient (band = -D,
  // A_C += D); the ghost carries 2*value - inner so the row is the standard Dirichlet operator.
  void applyScalarBcStencil(ScalarField& sc) {
    if (distributed_)
      return;
    for (int f = 0; f < 6; ++f) {
      if (sc.bc[f] != 2)
        continue;  // only Dirichlet reopens; Neumann/periodic leave the (closed/interior) band
      const int a = f / 2, side = f % 2;
      CCField band = (a == 0)   ? (side == 0 ? sc.AW : sc.AE)
                     : (a == 1) ? (side == 0 ? sc.AS : sc.AN)
                                : (side == 0 ? sc.AB : sc.AT);
      patchScalarDirichletFace(sc.AC, band, sc.D, a, side);
    }
  }
  // nvcc requires member functions that contain extended (device) lambdas to be PUBLIC — the
  // OpenMP/host build accepts them private, so the breakage only shows on the CUDA backend.
 public:
  void patchScalarDirichletFace(CCField AC, CCField band, double D, int a, int side) {
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
    const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
    const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
    const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
    const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
    const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
    const int aInner = (side == 0) ? G : (G + na - 1);
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scalar_bc_stencil",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long i = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
          // base build put band(i) = -D*open_face and A_C += D*open_face; force the face fully open
          // (band -> -D, A_C gains D*(1-open)) without double-counting when it was already open.
          AC(i) += D + band(i);
          band(i) = -D;
        });
  }
  void applyScalarBcFace(CCField c, int a, int side, int type, double val) {
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
    const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
    const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
    const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
    const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
    const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
    const int aInner = (side == 0) ? G : (G + na - 1);  // inner boundary cell a-index
    const int dir = (side == 0) ? -1 : +1;              // toward the ghost
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scalar_bc_face",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long base = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
          for (int L = 1; L <= 2; ++L) {
            const long gcell = base + (long)dir * L * sa;
            const long icell = base - (long)dir * (L - 1) * sa;
            c(gcell) = (type == 2) ? (2.0 * val - c(icell)) : c(icell);
          }
        });
  }

 private:
  int nx_, ny_, nz_;
  C3 e_, e1_;
  std::size_t n_, n1_;
  double rho_ = 1.0, mu_ = 0.1, dt_ = 50.0;
  std::array<double, 3> f_{{0, 0, 0}};
  int velIters_ = 200, presIters_ = 20;
  int pcgMaxit_ = 500;
  double pcgRtol_ = 1e-10;  // cut-cell pressure MG-PCG
  bool useChebyshev_ = false,
       chebBoundsSet_ = false;  // Chebyshev pressure driver (set_pressure_chebyshev)
  int chebMaxit_ = 120;
  double chebRtol_ = 1e-9, chebA_ = 0.0, chebB_ = 0.0;
  int nLevels_ = 4;  // multigrid depth (CUDA default; set_pressure_multigrid)
  long lastPressureIters_ = 0;
  CutcellMG mg_;
  // --- multi-rank (MPI) state, gated (single-GPU module never links MPI -> byte-identical when
  // off) ---
  bool distributed_ = false;
  C3 og_{0, 0, 0};  // velocity-block inner origin (global red-black parity); {0,0,0} single-rank
#ifdef PECLET_FLOW_MPI
  std::shared_ptr<GridHaloTopology<3>> velHalo_;  // g=2 velocity-block topology
  std::shared_ptr<GridHalo<double>> velDev_;      // g=2 velocity-block ghost exchange
  MPI_Comm comm_ = MPI_COMM_NULL;
  int gnx_ = 0, gny_ = 0, gnz_ = 0;  // communicator + GLOBAL dims
#endif
  int bc_[6] = {0, 0, 0, 0, 0, 0};
  double bcVel_[6][3] = {};
  bool hasBc_ = false, hasOutflow_ = false;  // domain BCs
  bool hasSolid_ = false;  // an immersed solid is present (any inner SDF < 0) -- with domain BCs, the
                           // momentum solve must use the cut-cell IBM stencil, not the all-fluid fold
  double backflowBeta_ = 0.2;  // outflow backflow-stabilization coefficient (0 = off; inert unless the
                               // outflow reverses, so purely-outgoing outlets stay byte-identical)
  CCField bcProf_[6];
  int bcProfNc_[6] = {0, 0, 0, 0, 0, 0};  // per-position inlet profiles (face grid [Lb*Lc*3])
  CCField bcDcorr_[3], bcBrhs_[3];        // implicit-diffusion face fold (per component)
  bool advect_ = false, cutcellPressure_ = false, implicitFou_ = false;
  bool deferredCorr_ = true;  // deferred-correction advection (off = pure implicit FOU, 1st order)
  int advScheme_ = 0;  // high-order advection: 0 = SOU (default), 1 = Koren TVD
  bool incremental_ = true,
       pwarm_ = false;    // incremental-rotational pressure (CUDA default on) + warm-start
  int faceInterp_ = 0;    // collocated cell->face map: 0 = plain average, 1 = wall-aware (opt-in)
  bool useVelocityMg_ = false;
  int vmgLevels_ = 4, vmgVcycles_ = 8;  // IBM velocity multigrid (staircase)
  VelocityMG vmg_;
  CCField vmgTheta_, vmgClean_;
  int outerIters_ = 1;
  double outerTol_ = 0.0;  // Picard outer iteration (CUDA set_outer_iterations)
  long lastOuterIters_ = 0;
  double lastOuterCorr_ = 0.0;
  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  CCField uf_, vf_, wf_;      // collocated: transient face (MAC) field (approx projection)
  CCField old_[3], prev_[3];  // u^n time base + previous Picard iterate
  Comp C[3];
  peclet::core::FieldSet fields_;     // named directory of all cell fields (velocity/p/sdf + user)
  std::vector<ScalarField> scalars_;  // transported scalars (advection-diffusion)
  std::vector<Closure> closures_;     // property/body-force closures (applied at top of step())
  CCField cellForce_[3];              // per-cell momentum body force (Boussinesq / CFD-DEM feedback)
  bool hasCellForce_ = false;
  bool varProps_ = false;             // variable-coefficient momentum (variable viscosity)
  bool harmonicMu_ = false;           // harmonic vs arithmetic face-viscosity mean
  CCField muField_;                   // per-cell dynamic viscosity (when varProps_)
  int varRotMode_ = 0;                // rotational term under varProps: 0 chi*mu_min, 1 chi*mu(i), 2 off
  double varRotChi_ = 1.0;            // rotational coefficient scale chi
};

// The staggered MAC solver — THE flow solver, bit-identical to the pre-policy class. Bindings + the
// kokkos_mpi tests reference this name unchanged.
using IbmSolver = Solver<Staggered>;

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SDFLOW_IBM_HPP
