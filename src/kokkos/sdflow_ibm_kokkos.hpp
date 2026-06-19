// cfd-gpu — host-facing Kokkos IBM Navier-Stokes solver (drop-in sdflow-style API).
//
// Assembles the validated cut-cell IBM operators into a runnable solver on a fully-periodic MAC box with
// immersed SDF solids: per-component backward-Euler implicit diffusion with the Robust-Scaled cut-cell
// no-slip stencil (buildIbmOverlay + ibmBuildDiffusion + ibmModifyStencil + ibmSolidMask + ibmRbgsSweep),
// then a rotational incremental-pressure Chorin projection through the open-face-weighted cut-cell pressure
// Poisson (buildCutcellOp + divergOpen, solved by CG with the constant null space projected out, then
// projectCorrect; P += (rho/dt)*phi - mu*div(u*) matching CUDA press_update_k). Schemes are a FAITHFUL port
// of the CUDA sdflow (point-value cut-cell IBM = ibm_geometry_ext_k<0>; rotational pressure): the velocity
// field matches CUDA to ~1e-13 (machine precision). Physical units (rho/mu/dt + body force). std::vector
// setters/getters so a pybind module can drive it. The verify_poiseuille / verify_periodic_spheres mechanism
// (k matches CUDA to all printed digits), on any backend. NOTE (faithfulness items, see memory): the CG uses
// a diagonal preconditioner where CUDA uses RB-GS-preconditioned MG-PCG (same converged solution); the
// pressure operator is stored double where CUDA uses float mreal -- to reconcile in a later port pass.
#ifndef CFD_SDFLOW_IBM_KOKKOS_HPP
#define CFD_SDFLOW_IBM_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <vector>

#include "mac_ibm_kokkos.hpp"
#include "mac_pressure_kokkos.hpp"
#include "mac_cutcell_mg_kokkos.hpp"
#include "staggered_advection_kokkos.hpp"

namespace cfdk {

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
      const int maxCut = nx*ny*nz;
      C[c].ov = IbmOverlay{ Kokkos::View<int*,CCMem>("ci",maxCut),Kokkos::View<int*,CCMem>("nb",maxCut),
        FV("dr",maxCut),Kokkos::View<int*,CCMem>("dc",(std::size_t)maxCut*6),FV("K",(std::size_t)maxCut*6),
        FV("M",(std::size_t)maxCut*6),FV("X",(std::size_t)maxCut*6),FV("Nbc",(std::size_t)maxCut*6),FV("R",(std::size_t)maxCut*6)};
      C[c].idMap = Kokkos::View<int*,CCMem>("idMap", n_);
      C[c].counter = Kokkos::View<int,CCMem>("cnt");
    }
  }

  void setRho(double r) { rho_ = r; }
  void setMu(double m) { mu_ = m; }
  void setDt(double d) { dt_ = d; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setVelocityIterations(int it) { velIters_ = it; }
  void setPressureIterations(int it) { presIters_ = it; }
  void setAdvection(bool on) { advect_ = on; }       // explicit Koren-TVD advection (matches CUDA to ~1e-13)
  void setPressureLevels(int levels) { nLevels_ = levels < 1 ? 1 : levels; }  // MG depth (CUDA default 4)

  // SDF on the inner cells (flat x-fastest, size nx*ny*nz; <0 solid). cutcellPressure enables the
  // open-face-weighted cut-cell projection (off => velocity-only, e.g. unidirectional body-force flow).
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
    cutcellPressure_ = cutcellPressure;
    auto h = Kokkos::create_mirror_view(sdf_);
    auto wrap = [](int i, int n) { return (i % n + n) % n; };  // periodic ghosts in all 3 axes
    for (int z = 0; z < e_.z; ++z) for (int y = 0; y < e_.y; ++y) for (int x = 0; x < e_.x; ++x) {
      int ix = wrap(x - G, nx_), iy = wrap(y - G, ny_), iz = wrap(z - G, nz_);
      h((long)x + (long)y*e_.x + (long)z*(long)e_.x*e_.y) =
          sdfInner[(std::size_t)ix + (std::size_t)iy*nx_ + (std::size_t)iz*(std::size_t)nx_*ny_];
    }
    Kokkos::deep_copy(sdf_, h);
    const Off3 offs[3] = {{-0.5f,0,0},{0,-0.5f,0},{0,0,-0.5f}};
    for (int c = 0; c < 3; ++c) {
      C[c].nCut = buildIbmOverlay<0>(CCConst(sdf_), e_, G, offs[c], /*Dirichlet*/ 0, C[c].ov, C[c].idMap, C[c].counter);  // SCHEME 0 = point-value (matches CUDA ibm_geometry_ext_k<0>)
      ibmSolidMask(C[c].mask, CCConst(sdf_), e_, offs[c]);
      Kokkos::deep_copy(C[c].u, 0.0);
    }
    rebuildStencils();
    if (cutcellPressure_) {
      buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, 1.0, 1.0, 1.0);  // on the g=2 velocity block
      copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
      copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
      copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
      mg_.init(nx_, ny_, nz_, nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
      mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), 1.0, 1.0, 1.0);
      Kokkos::deep_copy(phi_, 0.0); Kokkos::deep_copy(P_, 0.0);
    }
  }

  void step() {
    if (cutcellPressure_) fillGhosts(P_);            // grad(P^n) for the incremental predictor
    if (advect_) { fillGhosts(C[0].u); fillGhosts(C[1].u); fillGhosts(C[2].u); }  // u^n ghosts for advect
    for (int c = 0; c < 3; ++c) buildRhs(c);         // all RHS from u^n (advection couples the components)
    for (int c = 0; c < 3; ++c) smoothComp(c);       // per-component IBM implicit-diffusion solve
    if (cutcellPressure_) project();                 // cut-cell projection -> incompressible
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) { return gatherInner(C[c].u); }
  std::vector<double> getPressure() { return gatherInner(P_); }
  double maxOpenDivergence() {
    if (!cutcellPressure_) return 0.0;
    fillGhosts(C[0].u); fillGhosts(C[1].u); fillGhosts(C[2].u);
    divergOpen(CCConst(C[0].u),CCConst(C[1].u),CCConst(C[2].u), CCConst(ox_),CCConst(oy_),CCConst(oz_), div_, e_, G);
    return reduceMaxAbsInner(CCConst(div_));
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
    Kokkos::parallel_for("cfdk::copyInner", Kokkos::RangePolicy<CCExec>(space,0,(long)nx_*ny_*nz_),
      KOKKOS_LAMBDA(long c){ const int ix=(int)(c%NX), iy=(int)((c/NX)%NY), iz=(int)(c/((long)NX*NY));
        const long di=(long)(ix+dg)+(long)(iy+dg)*de.x+(long)(iz+dg)*(long)de.x*de.y;
        const long si=(long)(ix+sg)+(long)(iy+sg)*se.x+(long)(iz+sg)*(long)se.x*se.y;
        dst(di)=src(si); });
    space.fence();
  }
  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners).
  void fillGhosts(CCField f) { fillAxis(f,0); fillAxis(f,1); fillAxis(f,2); }
  void fillAxis(CCField f, int axis) {
    CCExec space; C3 e=e_; int N3[3]={nx_,ny_,nz_};
    int dims[3]={e.x,e.y,e.z}; long st[3]={1,e.x,(long)e.x*e.y};
    const int a=axis,b=(axis+1)%3,c=(axis+2)%3; const long sa=st[a],sb=st[b],sc=st[c]; const int N=N3[a];
    CCField ff=f;
    Kokkos::parallel_for("cfdk::ibm_pfill", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{dims[b],dims[c]}),
      KOKKOS_LAMBDA(int p0,int p1){ const long base=(long)p0*sb+(long)p1*sc;
        for(int gl=0;gl<G;++gl){ ff(base+(long)gl*sa)=ff(base+(long)(gl+N)*sa); ff(base+(long)(G+N+gl)*sa)=ff(base+(long)(G+gl)*sa);} });
    space.fence();
  }
  void buildRhs(int c) {
    CCExec space; const double idiag = rho_/dt_, fc = f_[c], rho = rho_; C3 e = e_;
    CCField bb=C[c].b, rs=C[c].rscale, P=P_;
    CCConst U=CCConst(C[0].u), V=CCConst(C[1].u), W=CCConst(C[2].u), uu=CCConst(C[c].u);
    const long strd = (c==0) ? 1 : (c==1) ? e_.x : (long)e_.x*e_.y;
    const bool incr = cutcellPressure_, adv = advect_;  // incremental predictor carries -grad(P^n)
    // b = rscale*(idiag*u^n - rho*advect(u^n) + f - grad P^n)  (CUDA advect_rhs_k + sub_gradp, then IBM scale)
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for("rhs", MD(space,{G,G,G},{e.x-G,e.y-G,e.z-G}),
      KOKKOS_LAMBDA(int x,int y,int z){
        const long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y;
        double a=0.0;
        if (adv) { sadvk::ViewAcc Ua{U,e.x,e.y}, Va{V,e.x,e.y}, Wa{W,e.x,e.y}, Fa{uu,e.x,e.y};
                   a = sadvk::advect(c, x,y,z, Ua,Va,Wa, Fa); }
        const double gp = incr ? (P(i)-P((long)i-strd)) : 0.0;
        bb(i) = rs(i) * (idiag*uu(i) + fc - rho*a - gp); });
    space.fence();
  }
  void smoothComp(int c) {
    for (int it = 0; it < velIters_; ++it) {
      fillGhosts(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, C3{0,0,0}, G, 0);
      fillGhosts(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, C3{0,0,0}, G, 1);
    }
  }
  // Incremental (rotational) cut-cell projection: solve A phi = -div_open(u*) (RB-GS, mean-removed),
  // u -= grad phi, then accumulate the physical pressure P += (rho/dt)*phi - mu*div(u*) (Timmermans).
  void project() {
    fillGhosts(C[0].u); fillGhosts(C[1].u); fillGhosts(C[2].u);
    divergOpen(CCConst(C[0].u),CCConst(C[1].u),CCConst(C[2].u), CCConst(ox_),CCConst(oy_),CCConst(oz_), div_, e_, G);
    // bridge -div(u*) (g=2 block) -> the MG rhs (g=1 block); keep div(u*) in div_ for the pressure update
    copyInner(rhs1_, e1_, 1, CCConst(div_), e_, G);
    { CCExec space; CCField r=rhs1_;
      Kokkos::parallel_for("negdiv", Kokkos::RangePolicy<CCExec>(space,0,n1_), KOKKOS_LAMBDA(std::size_t i){ r(i)=-r(i); });
      space.fence(); }
    // geometric multigrid MG-PCG solve of the cut-cell pressure Poisson A phi = -div(u*) (CUDA mac_multigrid)
    Kokkos::deep_copy(phi1_, 0.0);
    lastPressureIters_ = mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12);
    copyInner(phi_, e_, G, CCConst(phi1_), e1_, 1);  // bridge phi back g=1 -> g=2
    fillGhosts(phi_);
    projectCorrect(C[0].u,C[1].u,C[2].u, CCConst(phi_), e_, G);
    // the grad(phi) correction also touches solid faces; re-impose no-slip there so the decoupled solid
    // velocity cannot accumulate (matches the CUDA apply_mask/mask_k after correct_k -> stability).
    for (int c = 0; c < 3; ++c) maskVelocity(c);
    // Rotational incremental pressure (Timmermans), matching CUDA press_update_k: P += (rho/dt)*phi - mu*div(u*).
    CCExec space; CCField P=P_, ph=phi_, d=div_; const double ct=rho_/dt_, mu=mu_;
    Kokkos::parallel_for("press", Kokkos::RangePolicy<CCExec>(space,0,n_),
      KOKKOS_LAMBDA(std::size_t i){ P(i) += ct*ph(i) - mu*d(i); });
    space.fence();
  }
  void maskVelocity(int c) {
    CCExec space; CCField u=C[c].u, m=C[c].mask;
    Kokkos::parallel_for("vmask", Kokkos::RangePolicy<CCExec>(space,0,n_),
      KOKKOS_LAMBDA(std::size_t i){ if (m(i) > 0.5) u(i) = 0.0; });
    space.fence();
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
  int nLevels_ = 4;                               // multigrid depth (CUDA default; set_pressure_multigrid)
  long lastPressureIters_ = 0;
  CutcellMG mg_;
  bool advect_ = false, cutcellPressure_ = false;
  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  Comp C[3];
};

}  // namespace cfdk

#endif  // CFD_SDFLOW_IBM_KOKKOS_HPP
