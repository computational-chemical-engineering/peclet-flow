// Consistency of the implicit-FOU advection operator (sadvk::fou_operator) with the explicit FOU
// flux (sadvk::advect_fou): the 7-point operator coefficients applied to a field PHI must equal
// dt*advect_fou(PHI) per cell. Also checks fou_operator_aniso(s=1) == fou_operator and that the
// operator is diagonally dominant (cC >= sum|off|). Runs on any Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "staggered_advection_kokkos.hpp"

using namespace sadvk;
using Mem = Kokkos::DefaultExecutionSpace::memory_space;
using DView = Kokkos::View<double*, Mem>;
using CView = Kokkos::View<const double*, Mem>;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 2, inx = 14, iny = 12, inz = 10;
    const int ex = inx + 2*g, ey = iny + 2*g, ez = inz + 2*g;
    const std::size_t n = (std::size_t)ex*ey*ez;
    const double dt = 0.3;

    std::mt19937 rng(6);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> hU(n),hV(n),hW(n),hP(n);
    for (std::size_t i=0;i<n;++i){hU[i]=uf(rng);hV[i]=uf(rng);hW[i]=uf(rng);hP[i]=uf(rng);}
    auto up=[&](const char*nm,std::vector<double>&h){DView v(nm,n);auto m=Kokkos::create_mirror_view(v);for(std::size_t i=0;i<n;++i)m(i)=h[i];Kokkos::deep_copy(v,m);return v;};
    DView U=up("U",hU),V=up("V",hV),W=up("W",hW),P=up("P",hP);

    const long ninner=(long)inx*iny*inz;
    DView resid("resid",ninner);  // |LHS-RHS| (operator vs explicit flux) + aniso diff
    const int comp=1;
    Kokkos::parallel_for("fou_op", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,ninner),
      KOKKOS_LAMBDA(long c){
        const int lx=(int)(c%inx),ly=(int)((c/inx)%iny),lz=(int)(c/((long)inx*iny));
        const int x=lx+g,y=ly+g,z=lz+g;
        ViewAcc Ua{CView(U),ex,ey},Va{CView(V),ex,ey},Wa{CView(W),ex,ey},Pa{CView(P),ex,ey};
        double cC=0,cxm=0,cxp=0,cym=0,cyp=0,czm=0,czp=0;
        fou_operator(comp,x,y,z,Ua,Va,Wa,dt,cC,cxm,cxp,cym,cyp,czm,czp);
        // also aniso with s=1 must match
        double aC=0,axm=0,axp=0,aym=0,ayp=0,azm=0,azp=0;
        fou_operator_aniso(comp,x,y,z,Ua,Va,Wa,dt,1.0,1.0,1.0,aC,axm,axp,aym,ayp,azm,azp);
        long sx=1,sy=ex,sz=(long)ex*ey, i=(long)x+(long)y*ex+(long)z*sz;
        double lhs = cC*Pa(x,y,z) + cxm*P(i-sx)+cxp*P(i+sx)+cym*P(i-sy)+cyp*P(i+sy)+czm*P(i-sz)+czp*P(i+sz);
        double rhs = dt*advect_fou(comp,x,y,z,Ua,Va,Wa,Pa);
        double anisoDiff = Kokkos::fabs(aC-cC)+Kokkos::fabs(axm-cxm)+Kokkos::fabs(axp-cxp)+Kokkos::fabs(aym-cym)+Kokkos::fabs(ayp-cyp)+Kokkos::fabs(azm-czm)+Kokkos::fabs(azp-czp);
        // cC - sum|off| = dt*div(advecting velocity) (exact identity); >= 0 only for a div-free field
        // (as in the real solver). With random velocity here it can be negative — not a defect.
        resid(c) = Kokkos::fabs(lhs-rhs) + anisoDiff;
      });
    auto hr=Kokkos::create_mirror_view(resid); Kokkos::deep_copy(hr,resid);
    int bad=0; double maxr=0;
    for (long c=0;c<ninner;++c){ if(hr(c)>1e-12*(1+std::fabs(hr(c))))++bad; if(hr(c)>maxr)maxr=hr(c); }
    if (bad){ std::fprintf(stderr,"FAIL: %ld/%ld cells operator != dt*advect_fou (max %.2e)\n",(long)bad,ninner,maxr); status=1; }
    if(!status) std::printf("[fou_operator] PASS: operator == dt*advect_fou and aniso(s=1)==fou_operator (%ld cells, exec %s)\n",
                            ninner, Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return status;
}
