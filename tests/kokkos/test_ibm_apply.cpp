// Correctness of the Kokkos IBM operator assembly: build the base backward-Euler diffusion stencil
// (ibmBuildDiffusion), build the Robust-Scaled overlay (ibmFillEntry), then apply it
// (ibmModifyStencil), and compare the modified stencil + inhomogeneous term + row scaling to a host
// replication. Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cut_cell_ibm.hpp"

using namespace dns;
using DSpace = Kokkos::DefaultExecutionSpace;
using FV = Kokkos::View<float*, IMem>;
using DV = Kokkos::View<double*, IMem>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int ex = 12, ey = 12, ez = 12;
    const std::size_t n = (std::size_t)ex * ey * ez;
    const double beta = 0.7, idiag = 3.0;
    const float u_bc = 0.25f;
    const int M = 64;  // cut cells

    // distinct interior grid indices + random cut-cell SDF configs
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> ic(2, 9);
    std::uniform_real_distribution<float> uc(0.05f, 0.95f), us(-0.95f, -0.05f), uf(0.05f, 0.95f), u01(0, 1);
    std::vector<int> cidx; std::vector<float> sc(M); std::vector<float> sn((std::size_t)M * 6);
    std::vector<char> used(n, 0);
    for (int i = 0; i < M;) {
      int x = ic(rng), y = ic(rng), z = ic(rng);
      std::size_t c = (std::size_t)x + (std::size_t)y * ex + (std::size_t)z * ex * ey;
      if (used[c]) continue; used[c] = 1; cidx.push_back((int)c);
      sc[i] = uc(rng);
      bool sw = u01(rng) < 0.3f;
      for (int k = 0; k < 6; ++k) { bool solid = (sw && k < 2) || (u01(rng) < 0.5f); sn[(std::size_t)i*6+k] = solid?us(rng):uf(rng); }
      ++i;
    }

    auto mkF = [&](const char* nm) { FV v(nm, n); return v; };
    FV AC=mkF("AC"),AW=mkF("AW"),AE=mkF("AE"),AS=mkF("AS"),AN=mkF("AN"),AB=mkF("AB"),AT=mkF("AT");
    DV inhom("inhom", n), rscale("rscale", n);  // zero-init
    ibmBuildDiffusion(AC,AW,AE,AS,AN,AB,AT, ex,ey,ez, beta, idiag);

    // build overlay on device
    IbmOverlay ov{ Kokkos::View<int*,IMem>("ci",M), Kokkos::View<int*,IMem>("nb",M),
                   Kokkos::View<float*,IMem>("dr",M), Kokkos::View<int*,IMem>("dc",(std::size_t)M*6),
                   FV("K",(std::size_t)M*6),FV("Mv",(std::size_t)M*6),FV("X",(std::size_t)M*6),FV("Nbc",(std::size_t)M*6),FV("R",(std::size_t)M*6) };
    Kokkos::View<int*,IMem> dci("dci",M); { auto m=Kokkos::create_mirror_view(dci); for(int i=0;i<M;++i)m(i)=cidx[i]; Kokkos::deep_copy(dci,m); }
    FV dsc("dsc",M); { auto m=Kokkos::create_mirror_view(dsc); for(int i=0;i<M;++i)m(i)=sc[i]; Kokkos::deep_copy(dsc,m); }
    FV dsn("dsn",(std::size_t)M*6); { auto m=Kokkos::create_mirror_view(dsn); for(std::size_t i=0;i<(std::size_t)M*6;++i)m(i)=sn[i]; Kokkos::deep_copy(dsn,m); }
    Kokkos::parallel_for("bo", Kokkos::RangePolicy<DSpace>(0,M), KOKKOS_LAMBDA(int i){
      float s6[6]; for(int k=0;k<6;++k)s6[k]=dsn((std::size_t)i*6+k);
      ibmFillEntry<0>(ov, i, dci(i), dsc(i), s6, 0);
    });
    Kokkos::fence();
    ibmModifyStencil(AC,AW,AE,AS,AN,AB,AT, inhom, rscale, ov, M, u_bc);

    auto gf=[&](FV v){ std::vector<float> o(n); auto m=Kokkos::create_mirror_view(v); Kokkos::deep_copy(m,v); for(std::size_t i=0;i<n;++i)o[i]=m(i); return o; };
    auto gd=[&](DV v){ std::vector<double> o(n); auto m=Kokkos::create_mirror_view(v); Kokkos::deep_copy(m,v); for(std::size_t i=0;i<n;++i)o[i]=m(i); return o; };
    auto GAC=gf(AC),GAW=gf(AW),GAE=gf(AE),GAS=gf(AS),GAN=gf(AN),GAB=gf(AB),GAT=gf(AT); auto GIN=gd(inhom),GRS=gd(rscale);

    // --- host replication ---
    std::vector<float> hAC(n),hAW(n),hAE(n),hAS(n),hAN(n),hAB(n),hAT(n); std::vector<double> hin(n,0),hrs(n,0);
    float nb=(float)(-beta), cc=(float)(idiag+6.0*beta);
    for (std::size_t i=0;i<n;++i){hAC[i]=cc;hAW[i]=nb;hAE[i]=nb;hAS[i]=nb;hAN[i]=nb;hAB[i]=nb;hAT[i]=nb;}
    using HOV = IbmOverlayT<Kokkos::HostSpace>;
    HOV h{ Kokkos::View<int*,Kokkos::HostSpace>("hci",M),Kokkos::View<int*,Kokkos::HostSpace>("hnb",M),
           Kokkos::View<float*,Kokkos::HostSpace>("hdr",M),Kokkos::View<int*,Kokkos::HostSpace>("hdc",(std::size_t)M*6),
           Kokkos::View<float*,Kokkos::HostSpace>("hK",(std::size_t)M*6),Kokkos::View<float*,Kokkos::HostSpace>("hM",(std::size_t)M*6),
           Kokkos::View<float*,Kokkos::HostSpace>("hX",(std::size_t)M*6),Kokkos::View<float*,Kokkos::HostSpace>("hNbc",(std::size_t)M*6),
           Kokkos::View<float*,Kokkos::HostSpace>("hR",(std::size_t)M*6) };
    for (int i=0;i<M;++i){ float s6[6]; for(int k=0;k<6;++k)s6[k]=sn[(std::size_t)i*6+k]; ibmFillEntry<0>(h,i,cidx[i],sc[i],s6,0); }
    const int OPP[6]={1,0,3,2,5,4};
    for (int li=0;li<M;++li){ int c=h.cell_index(li); float descale=h.D_rescale(li); hrs[c]=descale;
      double orig[6]={hAE[c],hAW[c],hAN[c],hAS[c],hAT[c],hAB[c]}; double aC=(double)hAC[c]*(double)descale; double mod[6]={0,0,0,0,0,0}; double inh=0;
      for(int k=0;k<6;++k){ float K=h.K_val(li*6+k),Mv=h.M_val(li*6+k),X=h.X_val(li*6+k),Nbc=h.Nbc_val(li*6+k); double vnb=orig[k];
        aC+=vnb*K; inh+=(double)Nbc*u_bc*vnb; mod[k]+=vnb*((double)descale*Mv-1.0); mod[OPP[k]]+=vnb*X; }
      hAC[c]=(float)aC; hAE[c]=(float)(orig[0]+mod[0]); hAW[c]=(float)(orig[1]+mod[1]); hAN[c]=(float)(orig[2]+mod[2]);
      hAS[c]=(float)(orig[3]+mod[3]); hAT[c]=(float)(orig[4]+mod[4]); hAB[c]=(float)(orig[5]+mod[5]); hin[c]+=inh; }

    // Modified coefficients involve descale*M-1 cancellation + cross-terms; fma contraction differs
    // between nvcc and gcc, so use a relative tolerance appropriate to that (dist/inhom tracked separately).
    auto cf=[&](float a,float b){ return std::fabs((double)a-(double)b)<=2e-3*(1.0+std::fabs((double)b)); };
    auto cd=[&](double a,double b){ return std::fabs(a-b)<=2e-3*(1.0+std::fabs(b)); };
    int bad=0; double maxrel=0;
    auto rel=[&](double a,double b){ double r=std::fabs(a-b)/(1.0+std::fabs(b)); if(r>maxrel)maxrel=r; };
    for (std::size_t i=0;i<n;++i){
      rel(GAC[i],hAC[i]); rel(GAW[i],hAW[i]); rel(GAE[i],hAE[i]); rel(GIN[i],hin[i]);
      if(!cf(GAC[i],hAC[i])||!cf(GAW[i],hAW[i])||!cf(GAE[i],hAE[i])||!cf(GAS[i],hAS[i])||!cf(GAN[i],hAN[i])||!cf(GAB[i],hAB[i])||!cf(GAT[i],hAT[i]))++bad;
      if(!cd(GIN[i],hin[i])||!cd(GRS[i],hrs[i]))++bad;
    }
    if (bad){ std::fprintf(stderr,"FAIL: %d modified-stencil cells differ (max rel diff %.3e)\n",bad,maxrel); status=1; }
    else std::printf("[ibm_apply] PASS: build+overlay+modify stencil (%d cut cells) matches host (exec: %s)\n", M, DSpace::name());
  }
  Kokkos::finalize();
  return status;
}
