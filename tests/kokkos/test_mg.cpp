// Convergence test of the Kokkos geometric multigrid V-cycle (dns::MgPoisson) for the periodic
// Poisson Lap(phi)=d. Random mean-zero rhs; solve with increasing V-cycle counts and check the
// residual drops by ~an order of magnitude per cycle (vs ~0.99/sweep for plain RB-GS). Runs on any
// Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mac_mg.hpp"

using namespace dns;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 64, g = 1;
    I3 e{N + 2 * g, N + 2 * g, N + 2 * g};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;

    // random mean-zero d on inner cells (periodic solvability)
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> hd(n, 0.0);
    double sum = 0; long cnt = 0;
    for (int z=g; z<e.z-g; ++z) for (int y=g; y<e.y-g; ++y) for (int x=g; x<e.x-g; ++x) {
      long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y; hd[i]=uf(rng); sum+=hd[i]; ++cnt; }
    double mean = sum / cnt;
    for (int z=g; z<e.z-g; ++z) for (int y=g; y<e.y-g; ++y) for (int x=g; x<e.x-g; ++x) {
      long i=(long)x+(long)y*e.x+(long)z*(long)e.x*e.y; hd[i]-=mean; }

    SField d("d", n); { auto m=Kokkos::create_mirror_view(d); for(std::size_t i=0;i<n;++i)m(i)=hd[i]; Kokkos::deep_copy(d,m); }

    MgPoisson mg(N);
    std::printf("[mg] N=%d levels=%d\n", N, mg.numLevels());

    double prev = 0;
    for (int nv = 1; nv <= 6; ++nv) {
      SField phi("phi", n);  // start from zero each time
      mg.solve(phi, SConst(d), nv);
      double res = mg.finestResidualMax();
      double factor = (nv > 1) ? res / prev : 0.0;
      std::printf("[mg] %d V-cycle(s): max|resid|=%.3e  (factor vs prev %.3f)\n", nv, res, factor);
      prev = res;
    }

    // require strong convergence: a few V-cycles drive the residual far below the rhs scale.
    SField phi("phi", n);
    mg.solve(phi, SConst(d), 8);
    double res8 = mg.finestResidualMax();
    if (!(res8 < 1e-6)) { std::fprintf(stderr, "FAIL: 8 V-cycles did not converge (resid %.3e)\n", res8); status = 1; }
    else std::printf("[mg] PASS: 8 V-cycles -> max|resid|=%.3e (exec %s)\n", res8, SExec::name());
  }
  Kokkos::finalize();
  return status;
}
