// Domain-BC validation in context: steady plane Poiseuille flow. Periodic in x (flow) and z, no-slip
// walls in y, body force fx. The steady x-momentum reduces to -nu*Lap(u) = fx with u=0 at the walls,
// whose solution is the parabola u(y) = (fx/2nu) y(Ly-y). We solve it with the ported RB-GS diffusion
// smoother (cfdk::diffSmoothColor) using the wall face-fold for the no-slip tangential BC (the mac_bc
// operators), and check u_max/U_mean -> 1.5 and the profile matches the analytic parabola. Validates
// the domain-BC operators assembled with the diffusion operator. Runs on any Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_stencils_kokkos.hpp"

using namespace cfdk;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 24, g = 1;
    const double nu = 1.0, fx = 1.0;            // grid units; Ly = N
    I3 e{N + 2 * g, N + 2 * g, N + 2 * g}, og{0, 0, 0};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;
    const long sy = e.x;
    const double Ac = 6.0 * nu;                  // steady: 6nu*u - nu*sum = fx

    SField u("u", n), b("b", n), dcorr("dcorr", n);
    Kokkos::deep_copy(u, 0.0);
    // RHS = fx at inner cells; dcorr = +nu at the y-wall-adjacent inner planes (face-fold: the dropped
    // no-slip wall face moves its beta=nu onto the diagonal). bval = 2*nu*wall = 0 for no-slip.
    {
      auto hb = Kokkos::create_mirror_view(b);
      auto hd = Kokkos::create_mirror_view(dcorr);
      Kokkos::deep_copy(hb, b); Kokkos::deep_copy(hd, dcorr);
      for (std::size_t i = 0; i < n; ++i) { hb(i) = 0; hd(i) = 0; }
      for (int z = g; z < e.z - g; ++z) for (int y = g; y < e.y - g; ++y) for (int x = g; x < e.x - g; ++x) {
        long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        hb(i) = fx;
        if (y == g || y == e.y - g - 1) hd(i) += nu;  // wall-adjacent (bottom / top)
      }
      Kokkos::deep_copy(b, hb); Kokkos::deep_copy(dcorr, hd);
    }

    // ghost fill: periodic in x and z; no-slip wall (ghost=0) in y.
    SExec space;
    auto fillBC = [&]() {
      SField uu = u; const int Ny = N;
      // x periodic
      Kokkos::parallel_for("fx", Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(space, {0,0}, {e.y, e.z}),
        KOKKOS_LAMBDA(int y, int z){ long base=(long)y*e.x+(long)z*(long)e.x*e.y;
          for(int gl=0;gl<g;++gl){ uu(base+gl)=uu(base+gl+N); uu(base+g+N+gl)=uu(base+g+gl);} });
      // z periodic
      Kokkos::parallel_for("fz", Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(space, {0,0}, {e.x, e.y}),
        KOKKOS_LAMBDA(int x, int y){ long base=(long)x+(long)y*e.x; long sz=(long)e.x*e.y;
          for(int gl=0;gl<g;++gl){ uu(base+(long)gl*sz)=uu(base+(long)(gl+N)*sz); uu(base+(long)(g+N+gl)*sz)=uu(base+(long)(g+gl)*sz);} });
      // y walls: ghost = 0
      Kokkos::parallel_for("fy", Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(space, {0,0}, {e.x, e.z}),
        KOKKOS_LAMBDA(int x, int z){ long base=(long)x+(long)z*(long)e.x*e.y;
          for(int gl=0;gl<g;++gl){ uu(base+(long)gl*sy)=0.0; uu(base+(long)(g+N+gl)*sy)=0.0;} (void)Ny; });
      space.fence();
    };

    for (int it = 0; it < 6000; ++it) {
      fillBC(); diffSmoothColor(u, SConst(b), e, og, g, nu, Ac, 0, SConst(dcorr));
      fillBC(); diffSmoothColor(u, SConst(b), e, og, g, nu, Ac, 1, SConst(dcorr));
    }

    // extract the y-profile (x,z-uniform) and compare to the analytic parabola
    auto hu = Kokkos::create_mirror_view(u); Kokkos::deep_copy(hu, u);
    std::vector<double> prof(N), exact(N);
    double umax = 0, usum = 0;
    for (int yc = 0; yc < N; ++yc) {
      long i = (long)(g) + (long)(yc + g) * e.x + (long)(g) * (long)e.x * e.y;  // x=g,z=g column
      prof[yc] = hu(i);
      double yphys = yc + 0.5;
      exact[yc] = (fx / (2.0 * nu)) * yphys * (N - yphys);
      umax = std::fmax(umax, prof[yc]); usum += prof[yc];
    }
    const double umean = usum / N;
    const double ratio = umax / umean;  // Poiseuille: 1.5
    double num = 0, den = 0;
    for (int yc = 0; yc < N; ++yc) { num += (prof[yc]-exact[yc])*(prof[yc]-exact[yc]); den += exact[yc]*exact[yc]; }
    const double l2err = std::sqrt(num / den);

    std::printf("[poiseuille] u_max/U_mean=%.4f (target 1.5); profile L2 err vs parabola=%.3e  (umax=%.4f)\n",
                ratio, l2err, umax);
    if (std::fabs(ratio - 1.5) > 0.03) { std::fprintf(stderr, "FAIL: u_max/U_mean off 1.5\n"); status = 1; }
    if (l2err > 2e-2) { std::fprintf(stderr, "FAIL: profile not parabolic\n"); status = 1; }
    if (!status) std::printf("[poiseuille] PASS: no-slip-wall diffusion gives the Poiseuille parabola (exec %s)\n",
                             Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return status;
}
