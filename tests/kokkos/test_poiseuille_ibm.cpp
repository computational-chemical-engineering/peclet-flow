// Validation of the assembled IBM velocity solve (the verify_poiseuille_sdflow physics): plane
// Poiseuille through an SDF-defined channel with cut-cell IBM no-slip walls, body-force driven,
// physical units (rho/mu/dt). Solve the backward-Euler x-momentum to steady with the IBM-modified
// stencil and compare to U_max = F*H^2/(8*mu). Exercises buildIbmOverlay + ibmBuildDiffusion +
// ibmModifyStencil + ibmSolidMask + ibmRbgsSweep assembled together. Runs on any Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_ibm.hpp"

using namespace sdflow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 1;
    const int nx = 4, nz = 4, ny = 48;       // periodic x (flow) & z; channel along y
    // Half-integer wall positions so the SDF zero lies BETWEEN cells -> fluid cells with solid
    // neighbours (cut cells) form, exercising the Robust-Scaled cut-cell no-slip (not just masking).
    const double ylo = 6.5, yhi = ny - 6.5;   // immersed walls; H = yhi - ylo
    const double H = yhi - ylo;
    const double rho = 1.0, mu = 0.1, dt = 50.0, F = 0.01;
    const double idiag = rho / dt, beta = mu;
    C3 e{nx + 2 * g, ny + 2 * g, nz + 2 * g};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;
    const Off3 offU{-0.5f, 0.0f, 0.0f};        // u stored at the -x face

    // channel SDF: sdf(iy) = min(iy - ylo, yhi - iy) (>0 in fluid, <0 in the wall) at each ext cell.
    CCField sdf("sdf", n);
    { auto h = Kokkos::create_mirror_view(sdf);
      for (int z=0; z<e.z; ++z) for (int y=0; y<e.y; ++y) for (int x=0; x<e.x; ++x) {
        int iy = y - g; double s = std::min((double)iy - ylo, yhi - (double)iy);
        h((long)x + (long)y*e.x + (long)z*(long)e.x*e.y) = s;
      }
      Kokkos::deep_copy(sdf, h); }

    // build the u-component IBM overlay
    const int maxCut = (int)((long)nx * ny * nz);
    IbmOverlay ov{ Kokkos::View<int*,CCMem>("ci",maxCut), Kokkos::View<int*,CCMem>("nb",maxCut),
                   Kokkos::View<float*,CCMem>("dr",maxCut), Kokkos::View<int*,CCMem>("dc",(std::size_t)maxCut*6),
                   Kokkos::View<float*,CCMem>("K",(std::size_t)maxCut*6),Kokkos::View<float*,CCMem>("M",(std::size_t)maxCut*6),
                   Kokkos::View<float*,CCMem>("X",(std::size_t)maxCut*6),Kokkos::View<float*,CCMem>("Nbc",(std::size_t)maxCut*6),
                   Kokkos::View<float*,CCMem>("R",(std::size_t)maxCut*6) };
    Kokkos::View<int*,CCMem> idMap("idMap", n);
    Kokkos::View<int,CCMem> counter("counter");
    int nCut = buildIbmOverlay<1>(CCConst(sdf), e, g, offU, /*bc_type Dirichlet*/ 0, ov, idMap, counter);

    // base diffusion stencil + apply overlay
    using FV = Kokkos::View<float*, CCMem>;
    FV AC("AC",n),AW("AW",n),AE("AE",n),AS("AS",n),AN("AN",n),AB("AB",n),AT("AT",n);
    Kokkos::View<double*,CCMem> inhom("inhom",n), rscale("rscale",n);
    Kokkos::deep_copy(rscale, 1.0);  // non-cut cells: scale 1
    ibmBuildDiffusion(AC,AW,AE,AS,AN,AB,AT, e.x,e.y,e.z, beta, idiag);
    ibmModifyStencil(AC,AW,AE,AS,AN,AB,AT, inhom, rscale, ov, nCut, /*u_bc*/ 0.0f);

    // solid mask for u
    CCField mask("mask", n);
    ibmSolidMask(mask, CCConst(sdf), e, offU);

    CCField u("u", n), b("b", n);
    Kokkos::deep_copy(u, 0.0);
    CCExec space;

    // periodic x,z ghost fill (y handled by the immersed walls / mask)
    auto fillXZ = [&]() {
      CCField uu = u; const int Nx = nx, Nz = nz;
      Kokkos::parallel_for("fx", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{e.y,e.z}),
        KOKKOS_LAMBDA(int y,int z){ long base=(long)y*e.x+(long)z*(long)e.x*e.y;
          for(int gl=0;gl<g;++gl){ uu(base+gl)=uu(base+gl+Nx); uu(base+g+Nx+gl)=uu(base+g+gl);} });
      Kokkos::parallel_for("fz", Kokkos::MDRangePolicy<CCExec,Kokkos::Rank<2>>(space,{0,0},{e.x,e.y}),
        KOKKOS_LAMBDA(int x,int y){ long base=(long)x+(long)y*e.x; long sz=(long)e.x*e.y;
          for(int gl=0;gl<g;++gl){ uu(base+(long)gl*sz)=uu(base+(long)(gl+Nz)*sz); uu(base+(long)(g+Nz+gl)*sz)=uu(base+(long)(g+gl)*sz);} });
      space.fence();
    };

    // time-step the IBM momentum solve to steady state (slow approach: ~mu*k^2*dt per step)
    for (int step = 0; step < 600; ++step) {
      // RHS b' = rscale * (idiag*u + F)   (Robust-Scaled; inhom=0 for no-slip walls)
      { CCField uu=u, bb=b; auto rs=rscale; double id=idiag, ff=F;
        Kokkos::parallel_for("rhs", Kokkos::RangePolicy<CCExec>(space,0,n),
          KOKKOS_LAMBDA(std::size_t i){ bb(i) = rs(i) * (id*uu(i) + ff); });
        space.fence(); }
      for (int it = 0; it < 200; ++it) {
        fillXZ();
        ibmRbgsStencilColor(u, CCConst(b), MConst(AC),MConst(AW),MConst(AE),MConst(AS),MConst(AN),MConst(AB),MConst(AT),
                            CCConst(mask), e, C3{0,0,0}, g, 0);
        fillXZ();
        ibmRbgsStencilColor(u, CCConst(b), MConst(AC),MConst(AW),MConst(AE),MConst(AS),MConst(AN),MConst(AB),MConst(AT),
                            CCConst(mask), e, C3{0,0,0}, g, 1);
      }
    }

    // extract centerline profile u(y) at x=g,z=g and compare to the parabola
    auto hu = Kokkos::create_mirror_view(u); Kokkos::deep_copy(hu, u);
    const double Uana = F * H * H / (8.0 * mu);
    double umax = 0, l2num = 0, l2den = 0;
    for (int iy = 0; iy < ny; ++iy) {
      long i = (long)g + (long)(iy + g) * e.x + (long)g * (long)e.x * e.y;
      double un = hu(i);
      umax = std::fmax(umax, un);
      if (iy > ylo && iy < yhi) {  // fluid interior
        double yy = iy; double ue = (F / (2.0 * mu)) * (yy - ylo) * (yhi - yy);
        l2num += (un - ue) * (un - ue); l2den += ue * ue;
      }
    }
    const double l2err = std::sqrt(l2num / l2den);
    const double umaxErr = std::fabs(umax - Uana) / Uana;
    std::printf("[poiseuille_ibm] cut cells=%d; U_max=%.5f analytic=%.5f (err %.2e); profile L2 err=%.2e\n",
                nCut, umax, Uana, umaxErr, l2err);
    if (nCut <= 0) { std::fprintf(stderr, "FAIL: no cut cells found\n"); status = 1; }
    if (umaxErr > 3e-2) { std::fprintf(stderr, "FAIL: U_max off analytic\n"); status = 1; }
    if (l2err > 3e-2) { std::fprintf(stderr, "FAIL: profile not parabolic\n"); status = 1; }
    if (!status) std::printf("[poiseuille_ibm] PASS: IBM cut-cell channel reproduces Poiseuille (exec %s)\n",
                             Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return status;
}
