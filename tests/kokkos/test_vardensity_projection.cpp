// Variable-density projection: (a) the hydrostatic acid test — a stratified two-layer fluid at
// rest under gravity must stay at rest with the discrete pressure gradient exactly rho_face*g
// (any inconsistency between the momentum face density, the force face value, and the projection
// face coefficient shows up as a spurious velocity); (b) uniform-rho reduction — the varRho path
// with rho == rho_ reproduces the constant-density solver.
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

double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}

void hydrostatic(double ratio) {
  const int N = 8, NZ = 24;
  const double g = 0.1;
  peclet::flow::IbmSolver s(N, N, NZ);
  s.setRho(1.0);
  s.setMu(0.0);  // inviscid: the balance is exact (no viscous wall layer in the predictor)
  s.setDt(1.0);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z, periodic x,y
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * NZ, 10.0));
  std::vector<double> rho((std::size_t)N * N * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        rho[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            (z < NZ / 2) ? ratio : 1.0;  // heavy below (stable stratification)
  s.addField("rho");
  s.setField("rho", rho);
  s.setDensityMode(true);
  s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -g});
  double last = 1e300;
  for (int it = 0; it < 100; ++it) {
    s.step();
    last = std::fmax(maxAbs(s.getVelocity(0)),
                     std::fmax(maxAbs(s.getVelocity(1)), maxAbs(s.getVelocity(2))));
    CHECK(!std::isnan(last));
    if (std::isnan(last))
      return;
  }
  // steady state: velocity at machine zero, dP/dz = -g*rho_face to machine precision
  auto p = s.getPressure();
  double perr = 0;
  const int xc = N / 2, yc = N / 2;
  for (int z = 1; z < NZ; ++z) {
    const double dp = p[(std::size_t)xc + (std::size_t)yc * N + (std::size_t)z * N * N] -
                      p[(std::size_t)xc + (std::size_t)yc * N + (std::size_t)(z - 1) * N * N];
    const double rf = 0.5 * (((z < NZ / 2) ? ratio : 1.0) + ((z - 1 < NZ / 2) ? ratio : 1.0));
    perr = std::fmax(perr, std::fabs(dp + g * rf) / (g * ratio));
  }
  std::printf("hydrostatic ratio %g: final max|u| %.2e  P-grad rel-err %.2e\n", ratio, last, perr);
  CHECK(last < 1e-12);
  CHECK(perr < 1e-11);
}

void uniformReduction() {
  // Periodic body-force Stokes flow past an immersed cylinder: varRho(rho==rho_) vs constant.
  const int N = 12, NZ = 6;
  std::vector<std::vector<double>> uu, pp;
  for (int var = 0; var < 2; ++var) {
    peclet::flow::IbmSolver s(N, N, NZ);
    s.setRho(2.0);
    s.setMu(0.1);
    s.setDt(5.0);
    s.setBodyForce(1e-3, 0, 0);
    std::vector<double> sdf((std::size_t)N * N * NZ);
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
              std::sqrt(std::pow(x - N / 2.0, 2) + std::pow(y - N / 2.0, 2)) - N / 5.0;
    s.setSolid(sdf, true);
    if (var)
      s.setDensityMode(true);  // rho field seeded == rho_
    for (int it = 0; it < 40; ++it)
      s.step();
    uu.push_back(s.getVelocity(0));
    pp.push_back(s.getPressure());
  }
  double du = 0, dp = 0, us = 0;
  for (std::size_t i = 0; i < uu[0].size(); ++i) {
    du = std::fmax(du, std::fabs(uu[0][i] - uu[1][i]));
    dp = std::fmax(dp, std::fabs(pp[0][i] - pp[1][i]));
    us = std::fmax(us, std::fabs(uu[0][i]));
  }
  std::printf("uniform-rho reduction: rel du %.2e  dp %.2e\n", du / us, dp);
  CHECK(du / us < 1e-6);  // different pressure driver (Chebyshev vs PCG) -> solver tolerance
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  hydrostatic(3.0);
  hydrostatic(1000.0);
  uniformReduction();
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
