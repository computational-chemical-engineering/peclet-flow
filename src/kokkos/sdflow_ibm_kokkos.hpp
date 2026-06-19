// cfd-gpu — host-facing Kokkos IBM velocity solver (drop-in sdflow-style API).
//
// Wraps the validated assembled IBM velocity solve (buildIbmOverlay + ibmBuildDiffusion +
// ibmModifyStencil + ibmSolidMask + ibmRbgsSweep) for all three MAC velocity components, driven from a
// physical-units API (rho/mu/dt, body force) with an SDF solid (cut-cell no-slip, cell-average
// scheme). Periodic in x,z; immersed solids handle walls. std::vector setters/getters so a pybind
// module can drive it. Pressure projection is a no-op for the divergence-free channel flow it targets
// (verify_poiseuille_sdflow); a full multi-feature solver would add the cut-cell pressure MG.
#ifndef CFD_SDFLOW_IBM_KOKKOS_HPP
#define CFD_SDFLOW_IBM_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <array>
#include <vector>

#include "mac_ibm_kokkos.hpp"

namespace cfdk {

class SdflowIbm {
 public:
  using FV = Kokkos::View<float*, CCMem>;
  static constexpr int G = 1;

  SdflowIbm(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    e_ = C3{nx + 2 * G, ny + 2 * G, nz + 2 * G};
    n_ = (std::size_t)e_.x * e_.y * e_.z;
    sdf_ = CCField("sdf", n_);
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

  // SDF on the inner cells (flat x-fastest, size nx*ny*nz; <0 solid). Builds the per-component overlays.
  void setSolid(const std::vector<double>& sdfInner) {
    auto h = Kokkos::create_mirror_view(sdf_);
    // inner cells from input; ghosts: periodic wrap in x,z, clamp (nearest) in y.
    auto wrap = [](int i, int n) { return (i % n + n) % n; };
    for (int z = 0; z < e_.z; ++z) for (int y = 0; y < e_.y; ++y) for (int x = 0; x < e_.x; ++x) {
      int ix = wrap(x - G, nx_), iz = wrap(z - G, nz_);
      int iy = y - G; if (iy < 0) iy = 0; if (iy >= ny_) iy = ny_ - 1;  // clamp y (walls)
      h((long)x + (long)y*e_.x + (long)z*(long)e_.x*e_.y) =
          sdfInner[(std::size_t)ix + (std::size_t)iy*nx_ + (std::size_t)iz*(std::size_t)nx_*ny_];
    }
    Kokkos::deep_copy(sdf_, h);
    const Off3 offs[3] = {{-0.5f,0,0},{0,-0.5f,0},{0,0,-0.5f}};
    for (int c = 0; c < 3; ++c) {
      C[c].nCut = buildIbmOverlay<1>(CCConst(sdf_), e_, G, offs[c], /*Dirichlet*/ 0, C[c].ov, C[c].idMap, C[c].counter);
      ibmSolidMask(C[c].mask, CCConst(sdf_), e_, offs[c]);
      Kokkos::deep_copy(C[c].u, 0.0);
    }
    rebuildStencils();
  }

  void step() {
    for (int c = 0; c < 3; ++c) solveComponent(c);
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) {
    auto h = Kokkos::create_mirror_view(C[c].u); Kokkos::deep_copy(h, C[c].u);
    std::vector<double> out((std::size_t)nx_*ny_*nz_);
    for (int z=0; z<nz_; ++z) for (int y=0; y<ny_; ++y) for (int x=0; x<nx_; ++x)
      out[(std::size_t)x+(std::size_t)y*nx_+(std::size_t)z*(std::size_t)nx_*ny_] =
          h((long)(x+G)+(long)(y+G)*e_.x+(long)(z+G)*(long)e_.x*e_.y);
    return out;
  }
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
  void fillXZ(CCField f) {
    CCExec space; const int Nx=nx_, Nz=nz_; C3 e=e_; CCField uu=f;
    Kokkos::parallel_for("fx", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{e.y,e.z}),
      KOKKOS_LAMBDA(int y,int z){ long base=(long)y*e.x+(long)z*(long)e.x*e.y;
        for(int gl=0;gl<G;++gl){ uu(base+gl)=uu(base+gl+Nx); uu(base+G+Nx+gl)=uu(base+G+gl);} });
    Kokkos::parallel_for("fz", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{e.x,e.y}),
      KOKKOS_LAMBDA(int x,int y){ long base=(long)x+(long)y*e.x; long sz=(long)e.x*e.y;
        for(int gl=0;gl<G;++gl){ uu(base+(long)gl*sz)=uu(base+(long)(gl+Nz)*sz); uu(base+(long)(G+Nz+gl)*sz)=uu(base+(long)(G+gl)*sz);} });
    space.fence();
  }
  void solveComponent(int c) {
    CCExec space; const double idiag = rho_/dt_, fc = f_[c];
    CCField uu=C[c].u, bb=C[c].b, rs=C[c].rscale;
    Kokkos::parallel_for("rhs", Kokkos::RangePolicy<CCExec>(space,0,n_),
      KOKKOS_LAMBDA(std::size_t i){ bb(i) = rs(i) * (idiag*uu(i) + fc); });
    space.fence();
    for (int it = 0; it < velIters_; ++it) {
      fillXZ(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, C3{0,0,0}, G, 0);
      fillXZ(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC),MConst(C[c].AW),MConst(C[c].AE),MConst(C[c].AS),
                          MConst(C[c].AN),MConst(C[c].AB),MConst(C[c].AT), CCConst(C[c].mask), e_, C3{0,0,0}, G, 1);
    }
  }

 private:
  int nx_, ny_, nz_; C3 e_; std::size_t n_;
  double rho_=1.0, mu_=0.1, dt_=50.0;
  std::array<double,3> f_{{0,0,0}};
  int velIters_ = 200;
  CCField sdf_;
  Comp C[3];
};

}  // namespace cfdk

#endif  // CFD_SDFLOW_IBM_KOKKOS_HPP
