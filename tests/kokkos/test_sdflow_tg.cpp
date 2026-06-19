// Validation of the assembled Kokkos sdflow step against the Taylor-Green vortex (an exact solution
// of incompressible Navier-Stokes). On a periodic staggered grid in grid units (dx=1, domain N
// cells, wavenumber k=2pi/N):
//   STOKES (advection off): the div-free TG velocity decays at the exact discrete backward-Euler
//     rate 1/(1+dt*nu*Lambda) per step, Lambda = 4(1-cos k) (discrete -Laplacian eigenvalue, 2D
//     z-uniform mode). We check the amplitude ratio, that div(u) stays ~0 (projection), and that the
//     spatial profile is preserved (uniform decay, no mode corruption).
//   NS (advection on): TG advection is balanced by the pressure gradient, so the decay stays close.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "sdflow_kokkos.hpp"

using cfdk::SdflowKokkos;

// Initialise the staggered Taylor-Green field (grid units): u at the -x face, v at the -y face.
static void initTG(SdflowKokkos& s, std::vector<double>& u0, std::vector<double>& v0) {
  const int N = s.N(), G = SdflowKokkos::G;
  auto e = s.ext();
  const double k = 2.0 * M_PI / N;
  auto hu = Kokkos::create_mirror_view(s.u());
  auto hv = Kokkos::create_mirror_view(s.v());
  auto hw = Kokkos::create_mirror_view(s.w());
  Kokkos::deep_copy(hu, s.u()); Kokkos::deep_copy(hv, s.v()); Kokkos::deep_copy(hw, s.w());
  u0.assign((size_t)e.x * e.y * e.z, 0.0); v0.assign(u0.size(), 0.0);
  for (int cz = 0; cz < N; ++cz) for (int cy = 0; cy < N; ++cy) for (int cx = 0; cx < N; ++cx) {
    long i = (long)(cx+G) + (long)(cy+G)*e.x + (long)(cz+G)*(long)e.x*e.y;
    double ux = cx, uy = cy + 0.5;          // -x face: x=cx, y=cy+0.5
    double vx = cx + 0.5, vy = cy;          // -y face: x=cx+0.5, y=cy
    hu(i) = std::cos(k*ux) * std::sin(k*uy);
    hv(i) = -std::sin(k*vx) * std::cos(k*vy);
    hw(i) = 0.0;
    u0[i] = hu(i); v0[i] = hv(i);
  }
  Kokkos::deep_copy(s.u(), hu); Kokkos::deep_copy(s.v(), hv); Kokkos::deep_copy(s.w(), hw);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 24;
    const double nu = 0.1, dt = 0.05;
    const int nsteps = 10;
    const double k = 2.0 * M_PI / N;
    const double Lambda = 4.0 * (1.0 - std::cos(k));            // discrete -Laplacian eigenvalue (2D)
    const double fStep = 1.0 / (1.0 + dt * nu * Lambda);        // backward-Euler amplitude factor/step
    const double fExpect = std::pow(fStep, nsteps);

    // --- STOKES (advection off): tight check vs the discrete decay ---
    {
      SdflowKokkos s(N, nu, dt);
      s.setAdvection(false);
      s.setIterations(/*nDiff*/ 300, /*nPois*/ 80);
      std::vector<double> u0, v0;
      initTG(s, u0, v0);
      const double a0 = s.l2(s.u());
      for (int it = 0; it < nsteps; ++it) s.step();
      const double a1 = s.l2(s.u());
      const double ratio = a1 / a0;
      const double divmax = s.maxDivU();

      // profile preservation: u_final / ratio should match u_init
      auto hu = Kokkos::create_mirror_view(s.u()); Kokkos::deep_copy(hu, s.u());
      auto e = s.ext();
      double num = 0, den = 0;
      for (int cz = 0; cz < N; ++cz) for (int cy = 0; cy < N; ++cy) for (int cx = 0; cx < N; ++cx) {
        long i = (long)(cx+2) + (long)(cy+2)*e.x + (long)(cz+2)*(long)e.x*e.y;
        double pred = u0[i] * ratio;
        num += (hu(i) - pred) * (hu(i) - pred);
        den += pred * pred;
      }
      double profErr = std::sqrt(num / den);

      const double rateErr = std::fabs(ratio - fExpect) / fExpect;
      std::printf("[sdflow_tg] STOKES: ratio=%.6f expected=%.6f (rel err %.2e); max|div|=%.2e; profile err=%.2e\n",
                  ratio, fExpect, rateErr, divmax, profErr);
      if (rateErr > 5e-3) { std::fprintf(stderr, "FAIL: decay rate off\n"); status = 1; }
      if (divmax > 1e-9) { std::fprintf(stderr, "FAIL: divergence not zero\n"); status = 1; }
      if (profErr > 5e-3) { std::fprintf(stderr, "FAIL: profile not preserved\n"); status = 1; }
    }

    // --- NS (advection on): decay should stay close to the same diffusion rate ---
    // The post-projection divergence equals the pressure-Poisson residual. With advection, div(u*) is
    // O(advection), so plain RB-GS (a slow pressure solver — this is why sdflow uses multigrid; the
    // transfer operators are ported but not yet wired here) leaves a small residual that shrinks with
    // more sweeps. We confirm the decay is right and that div decreases as the pressure solve does more.
    {
      SdflowKokkos s1(N, nu, dt), s2(N, nu, dt);
      s1.setAdvection(true); s1.setIterations(300, 80);
      s2.setAdvection(true); s2.setIterations(300, 400);
      std::vector<double> u0, v0;
      initTG(s1, u0, v0); initTG(s2, u0, v0);
      const double a0 = s1.l2(s1.u());
      for (int it = 0; it < nsteps; ++it) { s1.step(); s2.step(); }
      const double ratio = s1.l2(s1.u()) / a0;
      const double div80 = s1.maxDivU(), div400 = s2.maxDivU();
      const double rateErr = std::fabs(ratio - fExpect) / fExpect;
      std::printf("[sdflow_tg] NS:     ratio=%.6f expected=%.6f (rel err %.2e); max|div| 80 sweeps=%.2e, 400 sweeps=%.2e\n",
                  ratio, fExpect, rateErr, div80, div400);
      if (rateErr > 5e-2) { std::fprintf(stderr, "FAIL: NS decay too far from TG\n"); status = 1; }
      if (div80 > 1e-2) { std::fprintf(stderr, "FAIL: NS divergence too large\n"); status = 1; }
      if (!(div400 < div80)) { std::fprintf(stderr, "FAIL: more pressure sweeps did not reduce divergence\n"); status = 1; }
    }

    if (!status)
      std::printf("[sdflow_tg] PASS: assembled Kokkos NS step reproduces Taylor-Green (exec: %s)\n",
                  Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return status;
}
