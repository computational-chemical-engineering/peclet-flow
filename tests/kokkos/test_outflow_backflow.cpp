// Outflow reversal census (IbmSolver::outflowBackflow) -- the detector for the one inflow/outflow
// regime that is only conditionally energy-stable: reversed flow on a zero-gradient outflow face
// with the backflow stabilization off (peclet-examples ISSUES.md "Inflow/outflow diverges to
// NaN"). Plumbing gate on both grids' staggered path: a channel whose outlet is partly reversed
// by the initial field reports the reversed half; a plain developing channel reports nothing.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

void configure(peclet::flow::IbmSolver& s, int nx, int ny, int nz, double U) {
  s.setRho(1.0);
  s.setMu(0.1);
  s.setDt(0.01);  // rho/dt dominates: one step leaves the imposed field nearly unchanged
  s.setAdvection(true);
  s.setDomainBc(0, 2, U, 0.0, 0.0);  // -x inflow
  s.setDomainBc(1, 3, 0.0, 0.0, 0.0);  // +x outflow
  s.setDomainBc(2, 1, 0.0, 0.0, 0.0);
  s.setDomainBc(3, 1, 0.0, 0.0, 0.0);
  s.setPressureLevels(3);
  s.setPressurePcg(true, 100, 1e-8);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1e30));
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int nx = 16, ny = 8, nz = 4;
    const double U = 1.0;
    // (1) partly reversed outlet: u = +3U in the upper half, -U in the lower half (net outgoing
    //     2U*H/2 - U*H/2 > 0 so the projection keeps the lower half reversed), stabilization OFF ->
    //     the census reports the lower half and step() prints its warning once.
    {
      peclet::flow::IbmSolver s(nx, ny, nz);
      configure(s, nx, ny, nz, U);
      s.setBackflowStab(0.0);
      std::vector<double> u((std::size_t)nx * ny * nz);
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            u[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
                (y < ny / 2) ? -U : 3.0 * U;
      s.setVelocity(0, u);
      s.step();
      const auto ob = s.outflowBackflow();
      std::printf("[backflow] reversed outlet: max_reverse %.3f  fraction %.3f  energy_influx %.3e  "
                  "(%ld / %ld faces)\n",
                  ob.maxReverse, ob.fraction, ob.energyInflux, ob.reversed, ob.total);
      CHECK(ob.total == (long)ny * nz);
      CHECK(ob.reversed > 0);
      CHECK(ob.fraction > 0.3 && ob.fraction < 0.7);
      CHECK(ob.maxReverse > 0.5 * U && ob.maxReverse < 1.5 * U);
      CHECK(ob.energyInflux > 0.0);
    }
    // (2) the developing channel (uniform inlet, nothing reversed): all zeros, no warning.
    {
      peclet::flow::IbmSolver s(nx, ny, nz);
      configure(s, nx, ny, nz, U);
      for (int it = 0; it < 5; ++it)
        s.step();
      const auto ob = s.outflowBackflow();
      std::printf("[backflow] developing channel: max_reverse %.3e  fraction %.3f  (%ld / %ld)\n",
                  ob.maxReverse, ob.fraction, ob.reversed, ob.total);
      CHECK(ob.reversed == 0 && ob.maxReverse == 0.0 && ob.energyInflux == 0.0);
      CHECK(ob.total == (long)ny * nz);
    }
    // (3) no outflow face at all (a periodic box): everything zero, total zero.
    {
      peclet::flow::IbmSolver s(nx, ny, nz);
      s.setRho(1.0);
      s.setMu(0.1);
      s.setDt(0.01);
      const auto ob = s.outflowBackflow();
      CHECK(ob.total == 0 && ob.reversed == 0 && ob.maxReverse == 0.0);
    }
    if (failures == 0)
      std::printf("[backflow] PASS (exec %s)\n", Kokkos::DefaultExecutionSpace::name());
    else
      std::fprintf(stderr, "[backflow] FAILED (%d checks)\n", failures);
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
