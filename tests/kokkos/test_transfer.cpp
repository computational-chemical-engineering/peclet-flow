// Correctness of the Kokkos multigrid transfer (sdflow::restrict_ / prolong) + projection correction
// (sdflow::correct) vs host replication. Plus property checks: restriction of a constant is that
// constant; prolongation of a constant adds that constant. Runs on whatever backend Kokkos has.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mac_transfer.hpp"

using namespace sdflow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 1;
    T3 ratio{2, 2, 2};
    T3 cinner{8, 6, 5};
    T3 cext{cinner.x + 2 * g, cinner.y + 2 * g, cinner.z + 2 * g};
    T3 finner{ratio.x * cinner.x, ratio.y * cinner.y, ratio.z * cinner.z};
    T3 fext{finner.x + 2 * g, finner.y + 2 * g, finner.z + 2 * g};
    auto cn = (std::size_t)cext.x * cext.y * cext.z;
    auto fn = (std::size_t)fext.x * fext.y * fext.z;

    std::mt19937 rng(21);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> hfine(fn), hcoarse(cn), hu(fn), hv(fn), hw(fn), hphi(fn);
    for (auto* vec : {&hfine, &hu, &hv, &hw, &hphi}) for (auto& x : *vec) x = uf(rng);
    for (auto& x : hcoarse) x = uf(rng);

    auto upN = [&](const char* nm, std::vector<double>& h, std::size_t n) {
      TField v(nm, n); auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i) m(i) = h[i];
      Kokkos::deep_copy(v, m); return v; };
    auto get = [&](TField v, std::size_t n) {
      std::vector<double> o(n); auto m = Kokkos::create_mirror_view(v); Kokkos::deep_copy(m, v);
      for (std::size_t i = 0; i < n; ++i) o[i] = m(i); return o; };
    auto close = [](double a, double b) { return std::fabs(a - b) <= 1e-10 * (1.0 + std::fabs(b)); };
    long lc = [&](int x,int y,int z){ return (long)x+(long)y*cext.x+(long)z*(long)cext.x*cext.y; }(0,0,0);
    (void)lc;
    auto CI = [&](int x,int y,int z){ return (std::size_t)x+(std::size_t)y*cext.x+(std::size_t)z*(std::size_t)cext.x*cext.y; };
    auto FI = [&](int x,int y,int z){ return (std::size_t)x+(std::size_t)y*fext.x+(std::size_t)z*(std::size_t)fext.x*fext.y; };
    int bad = 0;

    // --- restriction ---
    TField fine = upN("fine", hfine, fn), coarse("coarse", cn);
    restrict_(coarse, fine, cext, fext, g, cinner, ratio);
    auto gco = get(coarse, cn);
    for (int icz=0; icz<cinner.z; ++icz) for (int icy=0; icy<cinner.y; ++icy) for (int icx=0; icx<cinner.x; ++icx) {
      double s=0; for(int dz=0;dz<ratio.z;++dz)for(int dy=0;dy<ratio.y;++dy)for(int dx=0;dx<ratio.x;++dx)
        s += hfine[FI(ratio.x*icx+dx+g, ratio.y*icy+dy+g, ratio.z*icz+dz+g)];
      double r = s/(ratio.x*ratio.y*ratio.z);
      if (!close(gco[CI(icx+g,icy+g,icz+g)], r)) ++bad;
    }

    // --- prolongation (coarse filled incl ghost) ---
    TField coarse2 = upN("coarse2", hcoarse, cn), fine2 = upN("fine2", hfine, fn);
    prolong(fine2, coarse2, fext, cext, g, finner, ratio);
    auto gf = get(fine2, fn);
    auto trilerp_h = [&](double x,double y,double z){
      double fx=std::floor(x),fy=std::floor(y),fz=std::floor(z); double wx=x-fx,wy=y-fy,wz=z-fz;
      int x0=(int)fx,y0=(int)fy,z0=(int)fz;
      auto F=[&](int xx,int yy,int zz){ return hcoarse[CI(xx,yy,zz)]; };
      double c00=F(x0,y0,z0)*(1-wx)+F(x0+1,y0,z0)*wx, c10=F(x0,y0+1,z0)*(1-wx)+F(x0+1,y0+1,z0)*wx;
      double c01=F(x0,y0,z0+1)*(1-wx)+F(x0+1,y0,z0+1)*wx, c11=F(x0,y0+1,z0+1)*(1-wx)+F(x0+1,y0+1,z0+1)*wx;
      double c0=c00*(1-wy)+c10*wy, c1=c01*(1-wy)+c11*wy; return c0*(1-wz)+c1*wz; };
    for (int ifz=0; ifz<finner.z; ++ifz) for (int ify=0; ify<finner.y; ++ify) for (int ifx=0; ifx<finner.x; ++ifx) {
      double cx=ratio.x==2?0.5*ifx-0.25+g:(double)(ifx+g);
      double cy=ratio.y==2?0.5*ify-0.25+g:(double)(ify+g);
      double cz=ratio.z==2?0.5*ifz-0.25+g:(double)(ifz+g);
      double r = hfine[FI(ifx+g,ify+g,ifz+g)] + trilerp_h(cx,cy,cz);
      if (!close(gf[FI(ifx+g,ify+g,ifz+g)], r)) ++bad;
    }

    // --- correction ---
    TField u=upN("u",hu,fn), v=upN("v",hv,fn), w=upN("w",hw,fn), phi=upN("phi",hphi,fn);
    correct(u, v, w, phi, fext, g);
    auto gu=get(u,fn), gvv=get(v,fn), gw=get(w,fn);
    for (int z=g; z<fext.z-g; ++z) for (int y=g; y<fext.y-g; ++y) for (int x=g; x<fext.x-g; ++x) {
      std::size_t i=FI(x,y,z); long sx=1,sy=fext.x,sz=(long)fext.x*fext.y;
      if (!close(gu[i], hu[i]-(hphi[i]-hphi[i-sx]))) ++bad;
      if (!close(gvv[i], hv[i]-(hphi[i]-hphi[i-sy]))) ++bad;
      if (!close(gw[i], hw[i]-(hphi[i]-hphi[i-sz]))) ++bad;
    }
    if (bad) { std::fprintf(stderr, "FAIL: %d transfer/correct cells differ\n", bad); status = 1; }

    // --- property: restrict(const)=const, prolong adds const ---
    std::vector<double> ones(fn, 3.5);
    TField cf = upN("cf", ones, fn), cc("cc", cn);
    restrict_(cc, cf, cext, fext, g, cinner, ratio);
    auto gcc = get(cc, cn);
    for (int icz=0; icz<cinner.z; ++icz) for (int icy=0; icy<cinner.y; ++icy) for (int icx=0; icx<cinner.x; ++icx)
      if (!close(gcc[CI(icx+g,icy+g,icz+g)], 3.5)) ++bad;
    std::vector<double> cones(cn, 2.0), fzero(fn, 0.0);
    TField cc2 = upN("cc2", cones, cn), ff = upN("ff", fzero, fn);
    prolong(ff, cc2, fext, cext, g, finner, ratio);
    auto gff = get(ff, fn);
    for (int ifz=0; ifz<finner.z; ++ifz) for (int ify=0; ify<finner.y; ++ify) for (int ifx=0; ifx<finner.x; ++ifx)
      if (!close(gff[FI(ifx+g,ify+g,ifz+g)], 2.0)) ++bad;
    if (bad && !status) { std::fprintf(stderr, "FAIL: constant-field property\n"); status = 1; }

    if (!status)
      std::printf("[mac_transfer] PASS: restrict/prolong/correct match host + constant properties (exec: %s)\n",
                  TExec::name());
  }
  Kokkos::finalize();
  return status;
}
