// Correctness of the Kokkos IBM pieces (dns): the variable-coefficient RB-GS smoother
// (ibmRbgsSweep) vs host replication + a Laplacian convergence sanity, and the geometric fields
// (ibmVolfrac / ibmSolidMask) property checks (all-fluid -> theta 1 / mask 0, all-solid -> 0 / 1).
// Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mac_ibm.hpp"

using namespace dns;

static long l3(int x, int y, int z, C3 e) { return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y; }

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 1;
    C3 inner{18, 14, 10}, e{inner.x + 2 * g, inner.y + 2 * g, inner.z + 2 * g}, og{0, 0, 0};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;

    std::mt19937 rng(91);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::uniform_real_distribution<float> uac(3.0f, 7.0f), uoff(-1.0f, 1.0f), u01(0.0f, 1.0f);
    std::vector<double> hx(n), hb(n), hmask(n);
    std::vector<float> hAC(n), hAW(n), hAE(n), hAS(n), hAN(n), hAB(n), hAT(n);
    for (std::size_t i = 0; i < n; ++i) {
      hx[i]=uf(rng); hb[i]=uf(rng); hmask[i]=(u01(rng)<0.2f)?1.0:0.0;
      hAC[i]=uac(rng); hAW[i]=uoff(rng); hAE[i]=uoff(rng); hAS[i]=uoff(rng); hAN[i]=uoff(rng); hAB[i]=uoff(rng); hAT[i]=uoff(rng);
    }
    auto upd = [&](const char* nm, std::vector<double>& h) { CCField v(nm, n); auto m=Kokkos::create_mirror_view(v);
      for (std::size_t i=0;i<n;++i) m(i)=h[i]; Kokkos::deep_copy(v,m); return v; };
    auto upf = [&](const char* nm, std::vector<float>& h) { Kokkos::View<float*,CCMem> v(nm,n); auto m=Kokkos::create_mirror_view(v);
      for (std::size_t i=0;i<n;++i) m(i)=h[i]; Kokkos::deep_copy(v,m); return v; };
    auto getd = [&](CCField v){ std::vector<double> o(n); auto m=Kokkos::create_mirror_view(v); Kokkos::deep_copy(m,v); for(std::size_t i=0;i<n;++i)o[i]=m(i); return o; };

    CCField x=upd("x",hx), b=upd("b",hb), mask=upd("mask",hmask);
    auto AC=upf("AC",hAC), AW=upf("AW",hAW), AE=upf("AE",hAE), AS=upf("AS",hAS), AN=upf("AN",hAN), AB=upf("AB",hAB), AT=upf("AT",hAT);

    ibmRbgsSweep(x, b, AC, AW, AE, AS, AN, AB, AT, mask, e, og, g);
    auto gx = getd(x);

    // host RB-GS (same float->double promotion, colour 0 then 1)
    std::vector<double> rx = hx;
    for (int color = 0; color < 2; ++color)
      for (int z=g; z<e.z-g; ++z) for (int y=g; y<e.y-g; ++y) for (int xx=g; xx<e.x-g; ++xx) {
        if (((og.x+xx+og.y+y+og.z+z)&1)!=color) continue;
        long i=l3(xx,y,z,e), sx=1, sy=e.x, sz=(long)e.x*e.y;
        if (hmask[i]>0.5) { rx[i]=0.0; continue; }
        double ac=hAC[i]; if (std::fabs(ac)<1e-30) continue;
        double s=(double)hAE[i]*rx[i+sx]+(double)hAW[i]*rx[i-sx]+(double)hAN[i]*rx[i+sy]+(double)hAS[i]*rx[i-sy]+(double)hAT[i]*rx[i+sz]+(double)hAB[i]*rx[i-sz];
        rx[i]=(hb[i]-s)/ac;
      }
    auto close=[](double a,double b2){ return std::fabs(a-b2)<=1e-9*(1.0+std::fabs(b2)); };
    int bad=0; for (std::size_t i=0;i<n;++i) if(!close(gx[i],rx[i])) ++bad;
    if (bad) { std::fprintf(stderr,"FAIL: %d RB-GS cells differ\n",bad); status=1; }

    // Laplacian convergence: AC=6, A_off=-1, no mask -> 6x - sum_nbr = b ; residual drops.
    std::vector<float> lc(n,6.f), lo(n,-1.f);
    auto LC=upf("LC",lc), LW=upf("LW",lo), LE=upf("LE",lo), LS=upf("LS",lo), LN=upf("LN",lo), LB=upf("LB",lo), LT=upf("LT",lo);
    CCField px=upd("px",hx); CCConst noMask;  // empty mask
    auto resid=[&](){ auto m=getd(px); double r=0;
      for (int z=g; z<e.z-g; ++z) for (int y=g; y<e.y-g; ++y) for (int xx=g; xx<e.x-g; ++xx) {
        long i=l3(xx,y,z,e), sx=1, sy=e.x, sz=(long)e.x*e.y;
        double s=m[i+sx]+m[i-sx]+m[i+sy]+m[i-sy]+m[i+sz]+m[i-sz];
        double res=(6.0*m[i]-s)-hb[i]; r+=res*res; } return std::sqrt(r); };
    double r0=resid();
    for (int it=0;it<100;++it) ibmRbgsSweep(px, b, LC, LW, LE, LS, LN, LB, LT, noMask, e, og, g);
    double r1=resid();
    if (!(r1 < 0.2*r0)) { std::fprintf(stderr,"FAIL: IBM smoother residual %.3e -> %.3e\n",r0,r1); status=1; }

    // volfrac / solid-mask properties
    std::vector<double> allf(n,10.0), alls(n,-10.0);
    CCField sf=upd("sf",allf), so=upd("so",alls), th("th",n), mk("mk",n);
    Off3 off{0,0,0};
    ibmVolfrac(th, sf, e, off); auto gth=getd(th);
    for (std::size_t i=0;i<n;++i) if(!close(gth[i],1.0)) ++bad;
    ibmVolfrac(th, so, e, off); auto gth2=getd(th);
    for (std::size_t i=0;i<n;++i) if(!close(gth2[i],0.0)) ++bad;
    ibmSolidMask(mk, sf, e, off); auto gmk=getd(mk);
    for (std::size_t i=0;i<n;++i) if(!close(gmk[i],0.0)) ++bad;
    ibmSolidMask(mk, so, e, off); auto gmk2=getd(mk);
    for (std::size_t i=0;i<n;++i) if(!close(gmk2[i],1.0)) ++bad;
    if (bad && !status) { std::fprintf(stderr,"FAIL: volfrac/mask property\n"); status=1; }

    if (!status)
      std::printf("[mac_ibm] PASS: variable-coeff RB-GS matches host; Laplacian resid %.2e -> %.2e; volfrac/mask ok (exec: %s)\n",
                  r0, r1, CCExec::name());
  }
  Kokkos::finalize();
  return status;
}
