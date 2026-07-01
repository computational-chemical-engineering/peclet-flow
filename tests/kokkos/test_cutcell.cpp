// Correctness of the Kokkos cut-cell face openness (peclet::flow::buildOpenness) vs a host replication on a
// sphere SDF, plus properties: an all-fluid SDF gives openness 1, an all-solid SDF gives 0. SDF sign
// convention: negative inside solid, positive in fluid. Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_cutcell.hpp"

using namespace peclet::flow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    C3 ext{16, 16, 16};
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
    auto IDX = [&](int x, int y, int z) { return (std::size_t)x + (std::size_t)y * ext.x + (std::size_t)z * (std::size_t)ext.x * ext.y; };

    // solid sphere radius 5 at (8,8,8): sdf = dist - R  (>0 fluid outside, <0 solid inside)
    std::vector<double> hsdf(n);
    const double cx = 8, cy = 8, cz = 8, R = 5;
    for (int z = 0; z < ext.z; ++z) for (int y = 0; y < ext.y; ++y) for (int x = 0; x < ext.x; ++x)
      hsdf[IDX(x, y, z)] = std::sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)) - R;

    auto upC = [&](const char* nm, std::vector<double>& h) {
      CCField v(nm, n); auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i) m(i) = h[i];
      Kokkos::deep_copy(v, m); return v; };
    auto getC = [&](CCField v) { std::vector<double> o(n); auto m = Kokkos::create_mirror_view(v); Kokkos::deep_copy(m, v); for (std::size_t i=0;i<n;++i) o[i]=m(i); return o; };

    CCField sdf = upC("sdf", hsdf), ox("ox", n), oy("oy", n), oz("oz", n);
    buildOpenness(ox, oy, oz, sdf, ext, dx, dy, dz);
    auto gox = getC(ox), goy = getC(oy), goz = getC(oz);

    // host reference (same functions)
    auto close = [](double a, double b) { return std::fabs(a - b) <= 1e-10 * (1.0 + std::fabs(b)); };
    int bad = 0;
    // Host reference replicates the same math on the host SDF vector.
    auto sampleH = [&](double X, double Y, double Z) {
      double fx=std::floor(X),fy=std::floor(Y),fz=std::floor(Z); double wx=X-fx,wy=Y-fy,wz=Z-fz;
      int x0=(int)fx,y0=(int)fy,z0=(int)fz; auto cl=[&](int v,int nn){return v<0?0:(v>=nn?nn-1:v);};
      int x1=cl(x0+1,ext.x),y1=cl(y0+1,ext.y),z1=cl(z0+1,ext.z); x0=cl(x0,ext.x);y0=cl(y0,ext.y);z0=cl(z0,ext.z);
      auto F=[&](int xx,int yy,int zz){return hsdf[IDX(xx,yy,zz)];};
      double c00=F(x0,y0,z0)*(1-wx)+F(x1,y0,z0)*wx, c10=F(x0,y1,z0)*(1-wx)+F(x1,y1,z0)*wx;
      double c01=F(x0,y0,z1)*(1-wx)+F(x1,y0,z1)*wx, c11=F(x0,y1,z1)*(1-wx)+F(x1,y1,z1)*wx;
      double c0=c00*(1-wy)+c10*wy,c1=c01*(1-wy)+c11*wy; return c0*(1-wz)+c1*wz; };
    auto faceH = [&](double X,double Y,double Z,int type){
      double sd=sampleH(X,Y,Z); if(sd<=0) return 0.0; double e=1.0;
      return ccFractionCore(sd, sampleH(X+e,Y,Z),sampleH(X-e,Y,Z),sampleH(X,Y+e,Z),sampleH(X,Y-e,Z),
                            sampleH(X,Y,Z+e),sampleH(X,Y,Z-e),type,dx,dy,dz); };
    for (int z=0; z<ext.z; ++z) for (int y=0; y<ext.y; ++y) for (int x=0; x<ext.x; ++x) {
      std::size_t i=IDX(x,y,z);
      if (!close(gox[i], faceH(x-0.5,y,z,1))) ++bad;
      if (!close(goy[i], faceH(x,y-0.5,z,2))) ++bad;
      if (!close(goz[i], faceH(x,y,z-0.5,3))) ++bad;
    }
    if (bad) { std::fprintf(stderr, "FAIL: %d openness faces differ\n", bad); status = 1; }

    // count cut faces (0<open<1) — the sphere surface must produce some
    int cut = 0, open1 = 0, closed0 = 0;
    for (std::size_t i = 0; i < n; ++i) { double v=gox[i]; if(v>1e-9&&v<1-1e-9)++cut; else if(v>1-1e-9)++open1; else ++closed0; }

    // properties
    std::vector<double> allf(n, 100.0), alls(n, -100.0);
    CCField sf=upC("sf",allf), so=upC("so",alls), a("a",n), b("b",n), c("c",n);
    buildOpenness(a,b,c, sf, ext, dx,dy,dz); auto ga=getC(a);
    for (std::size_t i=0;i<n;++i) if(!close(ga[i],1.0)) ++bad;
    buildOpenness(a,b,c, so, ext, dx,dy,dz); auto ga2=getC(a);
    for (std::size_t i=0;i<n;++i) if(!close(ga2[i],0.0)) ++bad;
    if (bad && !status) { std::fprintf(stderr, "FAIL: all-fluid/all-solid property\n"); status = 1; }

    if (!status)
      std::printf("[mac_cutcell] PASS: openness matches host (%d cut / %d open / %d closed x-faces) + properties (exec: %s)\n",
                  cut, open1, closed0, CCExec::name());
  }
  Kokkos::finalize();
  return status;
}
