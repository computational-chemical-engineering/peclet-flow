/// @file
/// @brief sdflow — host-facing Kokkos IBM Navier-Stokes solver (drop-in sdflow-style API).
///
/// Assembles the validated cut-cell IBM operators into a runnable solver on a fully-periodic MAC box with
/// immersed SDF solids: per-component backward-Euler implicit diffusion with the Robust-Scaled cut-cell
/// no-slip stencil (buildIbmOverlay + ibmBuildDiffusion + ibmModifyStencil + ibmSolidMask + ibmRbgsSweep),
/// then a rotational incremental-pressure Chorin projection through the open-face-weighted cut-cell pressure
/// Poisson (buildCutcellOp + divergOpen, solved by CG with the constant null space projected out, then
/// projectCorrect; P += (rho/dt)*phi - mu*div(u*) matching CUDA press_update_k). Schemes are a FAITHFUL port
/// of the CUDA sdflow (point-value cut-cell IBM = ibm_geometry_ext_k<0>; rotational pressure): the velocity
/// field matches CUDA to ~1e-13 (machine precision). Physical units (rho/mu/dt + body force). std::vector
/// setters/getters so a pybind module can drive it. The verify_poiseuille / verify_periodic_spheres mechanism
/// (k matches CUDA to all printed digits), on any backend. NOTE (faithfulness items, see memory): the CG uses
/// a diagonal preconditioner where CUDA uses RB-GS-preconditioned MG-PCG (same converged solution); the
/// pressure operator is stored double where CUDA uses float mreal -- to reconcile in a later port pass.
#ifndef CFD_SDFLOW_IBM_HPP
#define CFD_SDFLOW_IBM_HPP

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <vector>

#include "mac_ibm.hpp"
#include "mac_pressure.hpp"
#include "mac_cutcell_mg.hpp"
#include "mac_velocity_mg.hpp"
#include "mac_stencils.hpp"
#include "staggered_advection.hpp"

namespace sdflow {

class SdflowIbm {
 public:
  using FV = Kokkos::View<float*, CCMem>;
  static constexpr int G = 2;   // velocity block: Koren advection reach (pressure/MG bridged to g=1)

  SdflowIbm(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    e_ = C3{nx + 2 * G, ny + 2 * G, nz + 2 * G};
    n_ = (std::size_t)e_.x * e_.y * e_.z;
    e1_ = C3{nx + 2, ny + 2, nz + 2};                   // g=1 block for the cut-cell pressure MG
    n1_ = (std::size_t)e1_.x * e1_.y * e1_.z;
    sdf_ = CCField("sdf", n_);
    ox_ = CCField("ox", n_); oy_ = CCField("oy", n_); oz_ = CCField("oz", n_);
    phi_ = CCField("phi", n_); div_ = CCField("div", n_); P_ = CCField("P", n_);
    // g=1 scratch for the MG bridge (openness + rhs/phi + PCG vectors)
    ox1_=CCField("ox1",n1_); oy1_=CCField("oy1",n1_); oz1_=CCField("oz1",n1_);
    rhs1_=CCField("rhs1",n1_); phi1_=CCField("phi1",n1_);
    r_ = CCField("r", n1_); z_ = CCField("z", n1_); pp_ = CCField("pp", n1_); Ap_ = CCField("Ap", n1_);
    for (int c = 0; c < 3; ++c) {
      C[c].u = CCField("u", n_); C[c].b = CCField("b", n_);
      C[c].AC=FV("AC",n_);C[c].AW=FV("AW",n_);C[c].AE=FV("AE",n_);C[c].AS=FV("AS",n_);
      C[c].AN=FV("AN",n_);C[c].AB=FV("AB",n_);C[c].AT=FV("AT",n_);
      C[c].inhom=CCField("inhom",n_); C[c].rscale=CCField("rscale",n_); C[c].mask=CCField("mask",n_);
      bcDcorr_[c]=CCField("dcorr",n_); bcBrhs_[c]=CCField("brhs",n_);
      const int maxCut = nx*ny*nz;
      C[c].ov = IbmOverlay{ Kokkos::View<int*,CCMem>("ci",maxCut),Kokkos::View<int*,CCMem>("nb",maxCut),
        FV("dr",maxCut),Kokkos::View<int*,CCMem>("dc",(std::size_t)maxCut*6),FV("K",(std::size_t)maxCut*6),
        FV("M",(std::size_t)maxCut*6),FV("X",(std::size_t)maxCut*6),FV("Nbc",(std::size_t)maxCut*6),FV("R",(std::size_t)maxCut*6)};
      C[c].idMap = Kokkos::View<int*,CCMem>("idMap", n_);
      C[c].counter = Kokkos::View<int,CCMem>("cnt");
      old_[c] = CCField("uOld", n_);   // u^n time base (fixed over the step's Picard sweeps)
      prev_[c] = CCField("uPrev", n_);  // previous Picard iterate (outer-tolerance check)
    }
  }

  void setRho(double r) { rho_ = r; }
  void setMu(double m) { mu_ = m; }
  void setDt(double d) { dt_ = d; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setVelocityIterations(int it) { velIters_ = it; }
  void setPressureIterations(int it) { presIters_ = it; }
  void setAdvection(bool on) { advect_ = on; }       // explicit Koren-TVD advection (matches CUDA to ~1e-13)
  // Implicit-FOU deferred-correction advection (CUDA set_implicit_advection): solve the first-order-upwind
  // part of advection implicitly (in the velocity operator) + keep (Koren-FOU) explicit in the RHS ->
  // unconditionally stable for advection (high Re / large dt). Requires the IBM stencil (rebuilt per Picard
  // iteration with the FOU term); the domain-BC path needs velocity-MG (separate milestone).
  void setImplicitAdvection(bool on) { implicitFou_ = on; }
  // Picard outer iterations over the step (CUDA set_outer_iterations): the advecting velocity is lagged at
  // the current iterate u^k while the time base stays u^n. iters>=1; tol>0 stops early on max|du| < tol.
  void setOuterIterations(int iters) { outerIters_ = iters < 1 ? 1 : iters; }
  void setOuterTolerance(double tol) { outerTol_ = tol; }
  long lastOuterIterations() const { return lastOuterIters_; }
  // Velocity (momentum) multigrid for the IBM diffusion solve (CUDA set_velocity_multigrid): the STAIRCASE
  // coarse operator (exact == RB-GS, stiff-stable at large dt). Call before set_solid; built at geometry time.
  void setVelocityMultigrid(bool on, int levels, int vcycles) {
    useVelocityMg_ = on; vmgLevels_ = levels < 1 ? 1 : levels; vmgVcycles_ = vcycles < 1 ? 1 : vcycles;
  }
  void setPressureLevels(int levels) { nLevels_ = levels < 1 ? 1 : levels; }  // MG depth (CUDA default 4)
  // Chebyshev pressure driver (CUDA set_pressure_chebyshev): communication-light alternative to MG-PCG --
  // Chebyshev semi-iteration preconditioned by one symmetric V-cycle, no per-iteration global dot-products.
  // Spectral bounds of M^{-1}A are estimated once (lazily) on the first solve and reused every step.
  void setPressureChebyshev(bool on, int maxit, double rtol) {
    useChebyshev_ = on; chebMaxit_ = maxit; chebRtol_ = rtol; chebBoundsSet_ = false;
  }
  // MG-PCG pressure tolerance/iteration cap (CUDA set_pressure_pcg). The Kokkos cut-cell pressure solve is
  // MG-PCG by default; this just sets its bounds (the `on` flag is accepted for API parity).
  void setPressurePcg(bool /*on*/, int maxit, double rtol) { pcgMaxit_ = maxit; pcgRtol_ = rtol; }
  // Incremental-rotational pressure (CUDA set_incremental_pressure, default ON): the predictor carries
  // -grad(P^n) and the physical pressure is accumulated rotationally P += (rho/dt)*phi - mu*div(u*). OFF =>
  // classical non-incremental Chorin (no -grad(P^n) predictor; P derived on demand as (rho/dt)*phi).
  void setIncrementalPressure(bool on) { incremental_ = on; }
  // Pressure warm-start (CUDA set_pressure_warmstart, default OFF): seed each cut-cell pressure solve from
  // the previous step's projection potential (consecutive phi's are similar along a steady march -> a more
  // converged phi per fixed solver budget) instead of zeroing the initial guess.
  void setPressureWarmstart(bool on) { pwarm_ = on; }
  // CUDA-only 3-stream concurrent velocity solve (set_velocity_streams): no Kokkos analogue in this port
  // (the default-execution-space kernels are already stream-ordered). Accepted as a no-op for API parity.
  void setVelocityStreams(bool /*on*/) {}
  // Seed/restore the velocity state (CUDA set_state / upload_velocity): u/v/w are inner-cell fields
  // (flat x-fastest, size nx*ny*nz); written into the velocity block + ghosts refreshed (periodic wrap).
  void uploadVelocity(const std::vector<double>& uu, const std::vector<double>& vv,
                      const std::vector<double>& ww) {
    const std::vector<double>* src[3] = {&uu, &vv, &ww};
    for (int c = 0; c < 3; ++c) {
      auto h = Kokkos::create_mirror_view(C[c].u); Kokkos::deep_copy(h, C[c].u);
      for (int z=0;z<nz_;++z) for (int y=0;y<ny_;++y) for (int x=0;x<nx_;++x)
        h((long)(x+G)+(long)(y+G)*e_.x+(long)(z+G)*(long)e_.x*e_.y) =
            (*src[c])[(std::size_t)x+(std::size_t)y*nx_+(std::size_t)z*(std::size_t)nx_*ny_];
      Kokkos::deep_copy(C[c].u, h);
      fillGhosts(C[c].u);
    }
  }
#ifdef CFD_MPI
  // Multi-rank: this rank's SdflowIbm is constructed with its LOCAL block dims (= the BlockDecomposer of the
  // GLOBAL grid for this rank); initMpi wires the g=2 velocity-block halo + the global-origin red-black parity,
  // and switches fillGhosts/maxOpenDivergence + the pressure MG (CutcellMG::initMpi) onto their distributed
  // paths. The caller decomposes first (deterministic ORB) to size the constructor; initMpi re-derives it.
  void initMpi(int gnx, int gny, int gnz, MPI_Comm comm) {
    distributed_ = true; comm_ = comm; gnx_ = gnx; gny_ = gny; gnz_ = gnz;
    int rank = 0, size = 1; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    std::array<bool, 3> per{true, true, true};
    velHalo_ = std::make_shared<GridHalo<3>>();
    tpx::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), tpx::IVec<3>{gnx, gny, gnz});
    velHalo_->buildTopology(dec, rank, G, per, comm);
    velDev_ = std::make_shared<DeviceGridExchangeKokkos<double>>(); velDev_->init(*velHalo_);
    const auto oig = velHalo_->indexer().originInclGhost();
    og_ = {(int)oig[0] + G, (int)oig[1] + G, (int)oig[2] + G};  // block inner origin -> global parity
  }
#endif
  // per-face domain BC {face 0..5 = -x,+x,-y,+y,-z,+z}: type 0=periodic,1=no-slip wall,2=Dirichlet/inflow,3=outflow.
  void setDomainBc(int face, int type, double vx, double vy, double vz) {
    bc_[face]=type; bcVel_[face][0]=vx; bcVel_[face][1]=vy; bcVel_[face][2]=vz;
    hasBc_=false; hasOutflow_=false;
    for (int i=0;i<6;++i) { if (bc_[i]) hasBc_=true; if (bc_[i]==3) hasOutflow_=true; }
  }
  // per-position inlet velocity profile on `face` (CUDA set_domain_bc_profile): prof is (nb,nc,3) on the
  // inner grid of the face's two perpendicular axes; sets the face to inflow (type 2). Resampled (clamp) to
  // the ghost-inclusive face grid so the BC kernel indexes it directly by face position.
  void setDomainBcProfile(int face, const std::vector<double>& prof, int nb, int nc) {
    const int a=face/2; const int dims[3]={e_.x,e_.y,e_.z}; const int bax=(a+1)%3, cax=(a+2)%3;
    const int Lb=dims[bax], Lc=dims[cax];
    CCField pf("bcprof", (std::size_t)Lb*Lc*3); auto h=Kokkos::create_mirror_view(pf);
    auto cl=[](int v,int n){ return v<0?0:(v>=n?n-1:v); };
    for (int p0=0;p0<Lb;++p0) for (int p1=0;p1<Lc;++p1) {
      const int ib=cl(p0-G,nb), ic=cl(p1-G,nc);
      for (int k=0;k<3;++k) h(((long)p0*Lc+p1)*3+k) = prof[((std::size_t)ib*nc+ic)*3+k];
    }
    Kokkos::deep_copy(pf,h); bcProf_[face]=pf; bcProfNc_[face]=Lc;
    bc_[face]=2; hasBc_=true;  // a profiled face is an inflow
  }
  // all-fluid + domain-BC pressure (CUDA set_pressure_geometry): same path as set_solid with an open SDF.
  void setPressureGeometry(const std::vector<double>& sdfInner) { setSolid(sdfInner, true); }

  // SDF on the inner cells (flat x-fastest, size nx*ny*nz; <0 solid). cutcellPressure enables the
  // open-face-weighted cut-cell projection (off => velocity-only, e.g. unidirectional body-force flow).
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
    cutcellPressure_ = cutcellPressure;
    auto h = Kokkos::create_mirror_view(sdf_);
#ifdef CFD_MPI
    if (distributed_) {
      // Multi-rank: sdfInner is THIS rank's LOCAL inner block; fill the inner cells, then halo-exchange the
      // ghosts (cross-rank + periodic) so the overlay/openness read the neighbour's SDF at the block boundary.
      Kokkos::deep_copy(h, sdf_);
      for (int z = 0; z < nz_; ++z) for (int y = 0; y < ny_; ++y) for (int x = 0; x < nx_; ++x)
        h((long)(x+G) + (long)(y+G)*e_.x + (long)(z+G)*(long)e_.x*e_.y) =
            sdfInner[(std::size_t)x + (std::size_t)y*nx_ + (std::size_t)z*(std::size_t)nx_*ny_];
      Kokkos::deep_copy(sdf_, h);
      velDev_->exchange(sdf_);
    } else
#endif
    {
    auto wrap = [](int i, int n) { return (i % n + n) % n; };  // periodic ghosts in all 3 axes
    for (int z = 0; z < e_.z; ++z) for (int y = 0; y < e_.y; ++y) for (int x = 0; x < e_.x; ++x) {
      int ix = wrap(x - G, nx_), iy = wrap(y - G, ny_), iz = wrap(z - G, nz_);
      h((long)x + (long)y*e_.x + (long)z*(long)e_.x*e_.y) =
          sdfInner[(std::size_t)ix + (std::size_t)iy*nx_ + (std::size_t)iz*(std::size_t)nx_*ny_];
    }
    Kokkos::deep_copy(sdf_, h);
    }
    const Off3 offs[3] = {{-0.5f,0,0},{0,-0.5f,0},{0,0,-0.5f}};
    for (int c = 0; c < 3; ++c) {
      C[c].nCut = buildIbmOverlay<0>(CCConst(sdf_), e_, G, offs[c], /*Dirichlet*/ 0, C[c].ov, C[c].idMap, C[c].counter);  // SCHEME 0 = point-value (matches CUDA ibm_geometry_ext_k<0>)
      ibmSolidMask(C[c].mask, CCConst(sdf_), e_, offs[c]);
      Kokkos::deep_copy(C[c].u, 0.0);
    }
    rebuildStencils();
    if (hasBc_) setupBcDiffusion();  // bake the implicit-diffusion wall fold into the per-component stencil
    if (useVelocityMg_) {  // velocity-MG hierarchy: IBM (staircase/upwind) or domain-BC (const-coeff) mode
      vmg_.init(nx_, ny_, nz_, vmgLevels_);
      if (hasBc_) vmg_.setBC(bc_);
      else { vmgTheta_ = CCField("vmgTheta", n_); vmgClean_ = CCField("vmgClean", n_); }
    }
    if (cutcellPressure_) {
      buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, 1.0, 1.0, 1.0);  // on the g=2 velocity block
#ifdef CFD_MPI
      // openness ghosts (the operator + divergence read the +neighbour face) -> exchange across ranks
      if (distributed_) { velDev_->exchange(ox_); velDev_->exchange(oy_); velDev_->exchange(oz_); }
#endif
      if (hasBc_) {  // FLUX openness (beta): a face is OPEN only where it carries normal flux -- outflow, or
        B3 e2{e_.x,e_.y,e_.z}; CCField oa[3]={ox_,oy_,oz_};  // an inflow with nonzero normal velocity. Walls
        for (int a=0;a<3;++a) for (int s=0;s<2;++s) {        // and tangential-only Dirichlet faces (e.g. a
          const int t=bc_[2*a+s];                            // lid: type 2 with zero normal vel) are CLOSED.
          const bool open = (t==3) || (t==2 && (bcProf_[2*a+s].extent(0)>0 || std::fabs(bcVel_[2*a+s][a])>1e-12));
          if (t!=0 && !open) bcZeroOpenness(oa[a],e2,G,a,s);
        }
      }  // the MG re-derives the OPERATOR openness alpha (inflow Neumann -> closed) per level via setBC.
      copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
      copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
      copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
#ifdef CFD_MPI
      if (distributed_) mg_.initMpi(gnx_, gny_, gnz_, nLevels_, comm_); else
#endif
      mg_.init(nx_, ny_, nz_, nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
      mg_.setBoundaryConditions(bc_);     // per-level wall openness + null-space gating (no-op if periodic)
      mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), 1.0, 1.0, 1.0);
      Kokkos::deep_copy(phi_, 0.0); Kokkos::deep_copy(P_, 0.0);
    }
  }

  void step() {
    // u^n time base, fixed for the whole step (Picard lags the advecting velocity at u^k, not the base).
    for (int c = 0; c < 3; ++c) Kokkos::deep_copy(old_[c], C[c].u);
    if (cutcellPressure_ && incremental_) { fillGhosts(P_); if (hasBc_) pressureBcGhost(); }  // grad(P^n) for the incremental predictor (once)
    lastOuterIters_ = 0;
    for (int outer = 0; outer < outerIters_; ++outer) {
      lastOuterIters_ = outer + 1;
      if (outerTol_ > 0) for (int c = 0; c < 3; ++c) Kokkos::deep_copy(prev_[c], C[c].u);
      if (advect_ || hasBc_) for (int c=0;c<3;++c) fillVelGhosts(c, 0);  // explicit ghosts (periodic + BC) for advect
      for (int c = 0; c < 3; ++c) buildRhs(c);         // RHS from u^n base + advection lagged at u^k
      // Implicit-FOU: rebuild the IBM velocity stencil = backward-Euler diffusion + rho*FOU(u^k), then
      // re-apply the cut-cell bake. Per Picard iteration (advecting velocity changes). IBM path only;
      // the domain-BC FOU operator lives in the velocity-MG levels (separate milestone).
      if (implicitFou_ && advect_ && !hasBc_) for (int c = 0; c < 3; ++c) buildAdvStencil(c);
      // upwind-convective velocity-MG: restrict the (frozen u^k) advecting velocity to the coarse levels ONCE,
      // before the per-component solves update it (shared across the 3 momentum components).
      if (useVelocityMg_ && implicitFou_ && advect_ && !hasBc_)
        vmg_.restrictAdvVelocities(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u));
      for (int c = 0; c < 3; ++c) smoothComp(c);       // per-component IBM implicit-diffusion solve
      if (cutcellPressure_) project();                 // cut-cell projection -> incompressible
      if (hasBc_) for (int c=0;c<3;++c) applyVelocityBcComp(c, 0, false);  // re-impose domain BCs (keep outflow)
      if (outerTol_ > 0) {  // outer convergence: max velocity change over this Picard iteration
        double corr = 0.0;
        for (int c = 0; c < 3; ++c) corr = Kokkos::fmax(corr, maxAbsDiffInner(CCConst(C[c].u), CCConst(prev_[c])));
        lastOuterCorr_ = corr;
        if (corr < outerTol_) break;
      }
    }
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) { return gatherInner(C[c].u); }
  std::vector<double> getPressure() {
    // Incremental scheme: P_ accumulates the physical pressure. Classical Chorin (!incremental_): derive it
    // on demand from the last projection potential, p = (rho/dt)*phi (CUDA press_from_phi_k).
    if (incremental_) return gatherInner(P_);
    std::vector<double> out = gatherInner(phi_);
    const double ct = rho_/dt_;
    for (double& x : out) x *= ct;
    return out;
  }
  double maxOpenDivergence() {
    if (!cutcellPressure_) return 0.0;
    for (int c=0;c<3;++c) fillVelGhosts(c, 0);  // ghosts incl. outflow zero-gradient before the divergence
    divergOpen(CCConst(C[0].u),CCConst(C[1].u),CCConst(C[2].u), CCConst(ox_),CCConst(oy_),CCConst(oz_), div_, e_, G);
    double m = reduceMaxAbsInner(CCConst(div_));
#ifdef CFD_MPI
    if (distributed_) { double g = 0; MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_); return g; }
#endif
    return m;
  }
  long lastPressureIterations() const { return lastPressureIters_; }
  int nx() const { return nx_; } int ny() const { return ny_; } int nz() const { return nz_; }

 private:
  struct Comp { CCField u,b,inhom,rscale,mask; FV AC,AW,AE,AS,AN,AB,AT; IbmOverlay ov; Kokkos::View<int*,CCMem> idMap; Kokkos::View<int,CCMem> counter; int nCut=0; };

 public:  // nvcc forbids extended __host__ __device__ lambdas inside private/protected members.
  void rebuildStencils() {
    const double idiag = rho_/dt_, beta = mu_;
    for (int c = 0; c < 3; ++c) {
      Kokkos::deep_copy(C[c].rscale, 1.0); Kokkos::deep_copy(C[c].inhom, 0.0);
      ibmBuildDiffusion(C[c].AC,C[c].AW,C[c].AE,C[c].AS,C[c].AN,C[c].AB,C[c].AT, e_.x,e_.y,e_.z, beta, idiag);
      ibmModifyStencil(C[c].AC,C[c].AW,C[c].AE,C[c].AS,C[c].AN,C[c].AB,C[c].AT, C[c].inhom, C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
    }
  }
  // copy the nx*ny*nz inner cells between two extended blocks of different ghost width (g=2 <-> g=1 MG).
  void copyInner(CCField dst, C3 de, int dg, CCConst src, C3 se, int sg) {
    CCExec space; const int NX=nx_, NY=ny_;
    Kokkos::parallel_for("sdflow::copyInner", Kokkos::RangePolicy<CCExec>(space,0,(long)nx_*ny_*nz_),
      KOKKOS_LAMBDA(long c){ const int ix=(int)(c%NX), iy=(int)((c/NX)%NY), iz=(int)(c/((long)NX*NY));
        const long di=(long)(ix+dg)+(long)(iy+dg)*de.x+(long)(iz+dg)*(long)de.x*de.y;
        const long si=(long)(ix+sg)+(long)(iy+sg)*se.x+(long)(iz+sg)*(long)se.x*se.y;
        dst(di)=src(si); });

  }
  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners). Distributed: the
  // velocity-block halo (cross-rank + periodic, all ghosts incl. corners).
  void fillGhosts(CCField f) {
#ifdef CFD_MPI
    if (distributed_) { velDev_->exchange(f); return; }
#endif
    fillAxis(f,0); fillAxis(f,1); fillAxis(f,2);
  }
  // Fused periodic FACE-ghost fill in ONE kernel (vs 3 fillAxis): each inner boundary cell scatters its
  // periodic image to the opposite face ghost, all 3 axes at once. Valid only for FACE-neighbour (7-point)
  // stencils -- it does NOT fill the corner/edge ghosts (which fillAxis's sequential x->y->z does). The IBM
  // RB-GS smoother reads only the 7-point stencil, so this is exact there and cuts the velocity solve's
  // dominant kernel-launch cost (~7200 -> ~2400 fill launches/step) at low resolution. NOT for the Koren
  // advection RHS (reads diagonals) -- keep the full fillGhosts there.
  void fillGhostsFaces(CCField f) {
#ifdef CFD_MPI
    if (distributed_) { velDev_->exchange(f); return; }  // halo gives all ghosts; the 7-pt smoother uses the faces
#endif
    CCExec space; C3 e=e_; const int Nx=nx_,Ny=ny_,Nz=nz_; const long sx=1,sy=e.x,sz=(long)e.x*e.y; CCField ff=f;
    Kokkos::parallel_for("sdflow::ibm_facefill", Kokkos::RangePolicy<CCExec>(space,0,(long)nx_*ny_*nz_),
      KOKKOS_LAMBDA(long n){
        const int ix=(int)(n%Nx), iy=(int)((n/Nx)%Ny), iz=(int)(n/((long)Nx*Ny));
        const long i=(long)(ix+G)*sx+(long)(iy+G)*sy+(long)(iz+G)*sz;
        if (ix<G) ff(i+(long)Nx*sx)=ff(i); else if (ix>=Nx-G) ff(i-(long)Nx*sx)=ff(i);
        if (iy<G) ff(i+(long)Ny*sy)=ff(i); else if (iy>=Ny-G) ff(i-(long)Ny*sy)=ff(i);
        if (iz<G) ff(i+(long)Nz*sz)=ff(i); else if (iz>=Nz-G) ff(i-(long)Nz*sz)=ff(i); });
  }
  void fillAxis(CCField f, int axis) {
    CCExec space; C3 e=e_; int N3[3]={nx_,ny_,nz_};
    int dims[3]={e.x,e.y,e.z}; long st[3]={1,e.x,(long)e.x*e.y};
    const int a=axis,b=(axis+1)%3,c=(axis+2)%3; const long sa=st[a],sb=st[b],sc=st[c]; const int N=N3[a];
    CCField ff=f;
    Kokkos::parallel_for("sdflow::ibm_pfill", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{dims[b],dims[c]}),
      KOKKOS_LAMBDA(int p0,int p1){ const long base=(long)p0*sb+(long)p1*sc;
        for(int gl=0;gl<G;++gl){ ff(base+(long)gl*sa)=ff(base+(long)(gl+N)*sa); ff(base+(long)(G+N+gl)*sa)=ff(base+(long)(G+gl)*sa);} });

  }
  void buildRhs(int c) {
    CCExec space; const double idiag = rho_/dt_, fc = f_[c], rho = rho_; C3 e = e_;
    CCField bb=C[c].b, rs=C[c].rscale, P=P_, brhs=bcBrhs_[c], inh=C[c].inhom;
    CCConst U=CCConst(C[0].u), V=CCConst(C[1].u), W=CCConst(C[2].u), uu=CCConst(C[c].u), un=CCConst(old_[c]);
    const long strd = (c==0) ? 1 : (c==1) ? e_.x : (long)e_.x*e_.y;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_, bc = hasBc_;  // incremental predictor carries -grad(P^n)
    const bool ifou = implicitFou_ && advect_;  // deferred correction: keep (Koren - FOU) explicit in the RHS
    // b = descale*(idiag*u^n - rho*Koren(u^k) + rho*FOU(u^k) + f - grad P^n) - inhom  (+ BC fold brhs). The
    // time base is u^n (Picard); the advecting velocity & advected field are the current iterate u^k.
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for("rhs", MD(space,{G,G,G},{e.x-G,e.y-G,e.z-G}),
      KOKKOS_LAMBDA(int x,int y,int z){
        const long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y;
        double aK=0.0, aF=0.0;
        if (adv) { sadv::ViewAcc Ua{U,e.x,e.y}, Va{V,e.x,e.y}, Wa{W,e.x,e.y}, Fa{uu,e.x,e.y};
                   aK = sadv::advect(c, x,y,z, Ua,Va,Wa, Fa);
                   if (ifou) aF = sadv::advect_fou(c, x,y,z, Ua,Va,Wa, Fa); }
        const double gp = incr ? (P(i)-P((long)i-strd)) : 0.0;
        bb(i) = rs(i) * (idiag*un(i) + fc - rho*aK + rho*aF - gp)
                + (bc ? brhs(i) : -inh(i)); });  // BC fold (brhs) on the domain-BC path; -inhom on the IBM path (=0 for no-slip)

  }
  // Implicit-FOU velocity stencil (CUDA build_adv_stencil_k + ibm_modify_stencil): backward-Euler diffusion
  // (idiag+6beta diag, -beta off) + rho*FOU(u^k) upwind operator (diagonally dominant -> stable at high Re),
  // then the Robust-Scaled cut-cell bake. The advecting velocity u^k = the current C[*].u (ghosts filled).
  void buildAdvStencil(int c) {
    const double idiag = rho_/dt_, beta = mu_, fouw = rho_; C3 e = e_;
    ibmBuildDiffusion(C[c].AC,C[c].AW,C[c].AE,C[c].AS,C[c].AN,C[c].AB,C[c].AT, e.x,e.y,e.z, beta, idiag);
    CCExec space; FV AC=C[c].AC,AW=C[c].AW,AE=C[c].AE,AS=C[c].AS,AN=C[c].AN,AB=C[c].AB,AT=C[c].AT;
    CCConst U=CCConst(C[0].u), V=CCConst(C[1].u), W=CCConst(C[2].u);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for("advstencil", MD(space,{G,G,G},{e.x-G,e.y-G,e.z-G}),
      KOKKOS_LAMBDA(int x,int y,int z){
        const long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y;
        double cC=AC(i),cxm=AW(i),cxp=AE(i),cym=AS(i),cyp=AN(i),czm=AB(i),czp=AT(i);
        sadv::ViewAcc Ua{U,e.x,e.y}, Va{V,e.x,e.y}, Wa{W,e.x,e.y};
        sadv::fou_operator(c, x,y,z, Ua,Va,Wa, fouw, cC,cxm,cxp,cym,cyp,czm,czp);
        AC(i)=(float)cC; AW(i)=(float)cxm; AE(i)=(float)cxp; AS(i)=(float)cym; AN(i)=(float)cyp; AB(i)=(float)czm; AT(i)=(float)czp; });

    Kokkos::deep_copy(C[c].rscale, 1.0); Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC,C[c].AW,C[c].AE,C[c].AS,C[c].AN,C[c].AB,C[c].AT, C[c].inhom, C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
  }
  // max|a-b| over inner cells (Picard outer-tolerance check).
  double maxAbsDiffInner(CCConst a, CCConst b) {
    CCExec space; C3 e=e_; double m=0;
    Kokkos::parallel_reduce("maxdiff", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<3>>(space,{G,G,G},{e.x-G,e.y-G,e.z-G}),
      KOKKOS_LAMBDA(int x,int y,int z,double& acc){ const long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y;
        const double d=Kokkos::fabs(a(i)-b(i)); if(d>acc) acc=d; }, Kokkos::Max<double>(m));
    return m;
  }
  void smoothComp(int c) {
    if (hasBc_ && useVelocityMg_) {  // domain-BC velocity multigrid: const-coeff aniso op + no-slip/inflow/
      // outflow boundary fold on every level (CUDA setDiffusionConstAllLevels + setDiffusionBoundaryFold).
      vmg_.setDomainBcOp(c, mu_, rho_/dt_);  // per component (the fold is component-dependent)
      fillVelGhosts(c, 1);  // set the level-0 boundary ghosts (wall fold=0, inflow value, outflow zero-grad)
      // Re-impose the velocity BC on the vel-MG's level-0 iterate each colour/residual (the const-coeff smoother
      // updates the held Dirichlet faces) -> the vel-MG converges to the RB-GS fixed point (not the ~2% drift
      // CUDA's vmg leaves at the boundary corners).
      vmg_.setBcApplyL0([this, c](CCField x) { fillVelGhostsTo(x, c, 1); });
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      return;
    }
    if (hasBc_) {  // domain-BC (no immersed solid): CUDA's double const-coeff diff_k + dcorr fold
      const I3 e{e_.x,e_.y,e_.z}, og{0,0,0}; const double beta=mu_, Ac=rho_/dt_ + 6.0*mu_;
      for (int it = 0; it < velIters_; ++it) {
        fillVelGhosts(c, 1);  // re-impose wall faces (fold) before each color
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 0, CCConst(bcDcorr_[c]));
        fillVelGhosts(c, 1);
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 1, CCConst(bcDcorr_[c]));
      }
      return;
    }
    if (useVelocityMg_) {  // IBM velocity multigrid: fine = sharp As_[c]; coarse op depends on the regime.
      vmg_.setFineStencil(FPC(C[c].AC),FPC(C[c].AW),FPC(C[c].AE),FPC(C[c].AS),FPC(C[c].AN),FPC(C[c].AB),FPC(C[c].AT));
      if (implicitFou_ && advect_) {
        // UPWIND-CONVECTIVE coarse op (advection-dominated): aniso const-coeff diffusion + dt*FOU from the
        // restricted advecting velocity (restrictAdvVelocities ran once in step()). No pin / no exclude mask.
        vmg_.buildUpwindCoarse(c, mu_, rho_/dt_, rho_);
      } else {
        // STAIRCASE coarse op (diffusion-only): theta classification + clean-fluid exclude (exact == RB-GS).
        const Off3 offs[3] = {{-0.5f,0,0},{0,-0.5f,0},{0,0,-0.5f}};
        ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, offs[c]);
        ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, offs[c]);
        vmg_.setStaircase(CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_, rho_/dt_, 0.5);
      }
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      maskVelocity(c);  // re-impose no-slip at solid (the masked solve leaves them at the pin value)
      return;
    }
    for (int it = 0; it < velIters_; ++it) {  // IBM / periodic: Robust-Scaled cut-cell stencil (float)
      fillGhostsFaces(C[c].u);  // 7-point smoother reads faces only -> the fused 1-kernel face fill suffices
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
      fillGhostsFaces(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
    }
  }
  // pressure ghost at domain faces for the incremental predictor's grad(P): zero-gradient (Neumann) at
  // every non-periodic face so grad(P) carries no spurious force there (the periodic fill wrapped the
  // opposite boundary's pressure). Outflow pressure (Dirichlet p=0) is enforced separately in the MG solve.
  void pressureBcGhost() {
    CCExec space; C3 e=e_; CCField P=P_;
    int dims[3]={e.x,e.y,e.z}; long st[3]={1,e.x,(long)e.x*e.y};
    for (int a=0;a<3;++a) for (int s=0;s<2;++s) {
      if (bc_[2*a+s]==0) continue;
      const int b=(a+1)%3,c=(a+2)%3; const long sa=st[a],sb=st[b],sc=st[c]; const int na=dims[a];
      const int bic=(s==0)?G:(na-G-1); const int lo=(s==0)?0:(na-G), hi=(s==0)?(G-1):(na-1);
      Kokkos::parallel_for("pbcghost", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{dims[b],dims[c]}),
        KOKKOS_LAMBDA(int p0,int p1){ const long base=(long)p0*sb+(long)p1*sc; const double pin=P(base+(long)bic*sa);
          for(int ia=lo;ia<=hi;++ia) P(base+(long)ia*sa)=pin; });
    }

  }
  // domain-BC velocity ghosts: periodic-fill periodic axes, then apply per-face BCs (fold=0 explicit/1 implicit).
  void fillVelGhosts(int comp, int fold) { fillVelGhostsTo(C[comp].u, comp, fold); }
  void applyVelocityBcComp(int comp, int fold, bool doOutflow) { applyVelocityBcCompTo(C[comp].u, comp, fold, doOutflow); }
  // Field-parameterized variants (so the velocity-MG can re-impose the BC on its own level-0 iterate).
  void fillVelGhostsTo(CCField f, int comp, int fold) {
#ifdef CFD_MPI
    if (distributed_) { velDev_->exchange(f); applyVelocityBcCompTo(f, comp, fold, true); return; }
#endif
    for (int a=0;a<3;++a) if (bc_[2*a]==0 && bc_[2*a+1]==0) fillAxis(f, a);
    applyVelocityBcCompTo(f, comp, fold, true);
  }
  void applyVelocityBcCompTo(CCField f, int comp, int fold, bool doOutflow) {
    if (!hasBc_) return;
    B3 e{e_.x,e_.y,e_.z};
    for (int a=0;a<3;++a) for (int s=0;s<2;++s) {
      const int ff=2*a+s; const int t=bc_[ff]; if (t==0) continue;
      if (t==3) { if (doOutflow) bcOutflowComp(f, e, G, a, s, comp, fold); continue; }
      if (bcProf_[ff].extent(0)>0) bcVelocityComp(f, e, G, a, s, comp, 0.0, fold, bcProf_[ff], bcProfNc_[ff]);
      else bcVelocityComp(f, e, G, a, s, comp, bcVel_[ff][comp], fold);
    }
  }
  // implicit-diffusion wall fold (CUDA setup_bc_diffusion): dcorr += (wall:+beta tangential / outflow:-beta),
  // brhs += 2*beta*wall (tangential Dirichlet); bake dcorr into the per-component stencil diagonal.
  void setupBcDiffusion() {
    const double beta = mu_; B3 e{e_.x,e_.y,e_.z};
    for (int c=0;c<3;++c) {
      Kokkos::deep_copy(bcDcorr_[c], 0.0); Kokkos::deep_copy(bcBrhs_[c], 0.0);
      for (int a=0;a<3;++a) for (int s=0;s<2;++s) {
        const int t=bc_[2*a+s]; double dval,bval;
        if (t==3) { dval=-beta; bval=0.0; }
        else if (t!=0 && c!=a) { dval=beta; bval=2.0*beta*bcVel_[2*a+s][c]; }
        else continue;  // periodic, or the normal component at a wall (held directly)
        bcDiffusionFold(bcDcorr_[c], bcBrhs_[c], e, G, a, s, dval, bval);
      }
      // dcorr is passed to the (double) const-coeff smoother diffSmoothColor each sweep -- matching CUDA
      // diff_k (Ac + dcorr in double), NOT baked into the float stencil.
    }
  }
  // Incremental (rotational) cut-cell projection: solve A phi = -div_open(u*) (RB-GS, mean-removed),
  // u -= grad phi, then accumulate the physical pressure P += (rho/dt)*phi - mu*div(u*) (Timmermans).
  void project() {
    // ghosts incl. domain BCs (outflow zero-gradient) BEFORE the divergence -- matches CUDA apply_velocity_bc
    // before diverg_open, so div(u*) counts the outflow flux (else the rotational pressure pumps the
    // mis-counted outflow divergence and blows up the outflow-wall corner).
    for (int c=0;c<3;++c) fillVelGhosts(c, 0);
    divergOpen(CCConst(C[0].u),CCConst(C[1].u),CCConst(C[2].u), CCConst(ox_),CCConst(oy_),CCConst(oz_), div_, e_, G);
    // bridge -div(u*) (g=2 block) -> the MG rhs (g=1 block); keep div(u*) in div_ for the pressure update
    copyInner(rhs1_, e1_, 1, CCConst(div_), e_, G);
    { CCExec space; CCField r=rhs1_;
      Kokkos::parallel_for("negdiv", Kokkos::RangePolicy<CCExec>(space,0,n1_), KOKKOS_LAMBDA(std::size_t i){ r(i)=-r(i); });
 }
    // geometric multigrid solve of the cut-cell pressure Poisson A phi = -div(u*) (CUDA mac_multigrid):
    // MG-PCG by default, or the communication-light Chebyshev driver (bounds estimated once, then reused).
    // Warm start (CUDA pwarm_): keep the previous step's phi1_ as the initial guess instead of zeroing.
    if (!pwarm_) Kokkos::deep_copy(phi1_, 0.0);
    if (useChebyshev_) {
      if (!chebBoundsSet_) { mg_.estimateEigenvalues(CCConst(rhs1_), chebA_, chebB_, 15, 2, 2, 12); chebBoundsSet_ = true; }
      lastPressureIters_ = mg_.solveChebyshev(rhs1_, phi1_, chebMaxit_, chebRtol_, 2, 2, 12, chebA_, chebB_);
    } else {
      lastPressureIters_ = mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12);
    }
    copyInner(phi_, e_, G, CCConst(phi1_), e1_, 1);  // bridge phi back g=1 -> g=2
    fillGhosts(phi_);
    if (hasOutflow_) {  // hold phi=0 at the outflow ghost so grad(phi) drives the outflow face (Dirichlet p=0)
      B3 e{e_.x,e_.y,e_.z};
      for (int a=0;a<3;++a) for (int s=0;s<2;++s) if (bc_[2*a+s]==3) bcZeroPressureGhost(phi_, e, G, a, s);
    }
    projectCorrect(C[0].u,C[1].u,C[2].u, CCConst(phi_), e_, G);
    if (hasOutflow_) {  // correct the high-side outflow normal face that projectCorrect misses (mass leaves)
      B3 e{e_.x,e_.y,e_.z};
      for (int a=0;a<3;++a) if (bc_[2*a+1]==3) bcCorrectOutflow(C[a].u, phi_, e, G, a);
    }
    // the grad(phi) correction also touches solid faces; re-impose no-slip there so the decoupled solid
    // velocity cannot accumulate (matches the CUDA apply_mask/mask_k after correct_k -> stability).
    for (int c = 0; c < 3; ++c) maskVelocity(c);
    // Rotational incremental pressure (Timmermans), matching CUDA press_update_k: P += (rho/dt)*phi - mu*div(u*).
    // Classical non-incremental Chorin (!incremental_) skips the accumulation; getPressure() derives p from phi.
    if (incremental_) {
      CCExec space; CCField P=P_, ph=phi_, d=div_; const double ct=rho_/dt_, mu=mu_;
      Kokkos::parallel_for("press", Kokkos::RangePolicy<CCExec>(space,0,n_),
        KOKKOS_LAMBDA(std::size_t i){ P(i) += ct*ph(i) - mu*d(i); });
    }
  }
  void maskVelocity(int c) {
    CCExec space; CCField u=C[c].u, m=C[c].mask;
    Kokkos::parallel_for("vmask", Kokkos::RangePolicy<CCExec>(space,0,n_),
      KOKKOS_LAMBDA(std::size_t i){ if (m(i) > 0.5) u(i) = 0.0; });

  }
  double reduceMaxAbsInner(CCConst f) {
    CCExec space; C3 e=e_; double m=0;
    Kokkos::parallel_reduce("maxabs", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<3>>(space,{G,G,G},{e.x-G,e.y-G,e.z-G}),
      KOKKOS_LAMBDA(int x,int y,int z,double& acc){ const long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y;
        const double a=Kokkos::fabs(f(i)); if(a>acc) acc=a; }, Kokkos::Max<double>(m));
    return m;
  }
  std::vector<double> gatherInner(CCField fld) {
    auto h = Kokkos::create_mirror_view(fld); Kokkos::deep_copy(h, fld);
    std::vector<double> out((std::size_t)nx_*ny_*nz_);
    for (int z=0; z<nz_; ++z) for (int y=0; y<ny_; ++y) for (int x=0; x<nx_; ++x)
      out[(std::size_t)x+(std::size_t)y*nx_+(std::size_t)z*(std::size_t)nx_*ny_] =
          h((long)(x+G)+(long)(y+G)*e_.x+(long)(z+G)*(long)e_.x*e_.y);
    return out;
  }

 private:
  int nx_, ny_, nz_; C3 e_, e1_; std::size_t n_, n1_;
  double rho_=1.0, mu_=0.1, dt_=50.0;
  std::array<double,3> f_{{0,0,0}};
  int velIters_ = 200, presIters_ = 20;
  int pcgMaxit_ = 500; double pcgRtol_ = 1e-10;   // cut-cell pressure MG-PCG
  bool useChebyshev_ = false, chebBoundsSet_ = false;  // Chebyshev pressure driver (set_pressure_chebyshev)
  int chebMaxit_ = 120; double chebRtol_ = 1e-9, chebA_ = 0.0, chebB_ = 0.0;
  int nLevels_ = 4;                               // multigrid depth (CUDA default; set_pressure_multigrid)
  long lastPressureIters_ = 0;
  CutcellMG mg_;
  // --- multi-rank (MPI) state, gated (single-GPU module never links MPI -> byte-identical when off) ---
  bool distributed_ = false;
  C3 og_{0,0,0};   // velocity-block inner origin (global red-black parity); {0,0,0} single-rank
#ifdef CFD_MPI
  std::shared_ptr<GridHalo<3>> velHalo_;                          // g=2 velocity-block topology
  std::shared_ptr<DeviceGridExchangeKokkos<double>> velDev_;      // g=2 velocity-block ghost exchange
  MPI_Comm comm_ = MPI_COMM_NULL; int gnx_=0, gny_=0, gnz_=0;     // communicator + GLOBAL dims
#endif
  int bc_[6] = {0,0,0,0,0,0}; double bcVel_[6][3] = {}; bool hasBc_ = false, hasOutflow_ = false;  // domain BCs
  CCField bcProf_[6]; int bcProfNc_[6] = {0,0,0,0,0,0};  // per-position inlet profiles (face grid [Lb*Lc*3])
  CCField bcDcorr_[3], bcBrhs_[3];                // implicit-diffusion face fold (per component)
  bool advect_ = false, cutcellPressure_ = false, implicitFou_ = false;
  bool incremental_ = true, pwarm_ = false;     // incremental-rotational pressure (CUDA default on) + warm-start
  bool useVelocityMg_ = false; int vmgLevels_ = 4, vmgVcycles_ = 8;  // IBM velocity multigrid (staircase)
  VelocityMG vmg_; CCField vmgTheta_, vmgClean_;
  int outerIters_ = 1; double outerTol_ = 0.0;        // Picard outer iteration (CUDA set_outer_iterations)
  long lastOuterIters_ = 0; double lastOuterCorr_ = 0.0;
  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  CCField old_[3], prev_[3];                            // u^n time base + previous Picard iterate
  Comp C[3];
};

}  // namespace sdflow

#endif  // CFD_SDFLOW_IBM_HPP
