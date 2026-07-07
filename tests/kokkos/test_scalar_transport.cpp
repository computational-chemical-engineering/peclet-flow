// Scalar transport on the flow Solver: openness-weighted implicit diffusion (2nd-order operator via
// the discrete-eigenvalue convergence) + explicit conservative advection (machine-precision
// conservation on a periodic all-fluid box).
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

std::vector<double> allFluid(int nx, int ny, int nz) {
  return std::vector<double>((std::size_t)nx * ny * nz, 10.0);
}

// Discrete Laplacian eigenvalue from one backward-Euler diffusion step of a cosine mode.
double measureEigen(int N) {
  const double D = 1.0, dt = 0.5;
  peclet::flow::IbmSolver s(N, 4, 4);
  s.setDt(dt);
  s.setPressureGeometry(allFluid(N, 4, 4));
  s.addScalar("c", D, 0, 600);
  const double k = 2.0 * M_PI / N;
  std::vector<double> c0((std::size_t)N * 4 * 4);
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < N; ++x)
        c0[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * 4] = std::cos(k * x);
  s.setField("c", c0);
  s.advanceScalars();  // velocity zero -> pure diffusion
  auto c1 = s.getField("c");
  const double peak0 = c0[0], peak1 = c1[0];  // x=y=z=0
  return (peak0 / peak1 - 1.0) / (dt * D);    // mu_h (eigenmode: c1 = c0/(1+dt*D*mu_h))
}

void run() {
  // 2nd-order diffusion operator: |mu_h - k^2| ~ O(k^4) => relative error halves^2 per refinement.
  const double e16 =
      std::fabs(measureEigen(16) - std::pow(2 * M_PI / 16, 2)) / std::pow(2 * M_PI / 16, 2);
  const double e32 =
      std::fabs(measureEigen(32) - std::pow(2 * M_PI / 32, 2)) / std::pow(2 * M_PI / 32, 2);
  const double order = std::log(e16 / e32) / std::log(2.0);
  std::printf("diffusion rel-errs %.3e %.3e  order %.2f\n", e16, e32, order);
  CHECK(order > 1.9);

  // Uniform-velocity advection conserves the scalar to machine precision (periodic, openness 1).
  const int N = 32;
  const double U = 0.4, dt = 0.5;
  peclet::flow::IbmSolver s(N, 4, 4);
  s.setDt(dt);
  s.setPressureGeometry(allFluid(N, 4, 4));
  s.addScalar("c", 0.0, 1, 1);  // pure advection, Koren
  std::vector<double> u((std::size_t)N * 4 * 4, U), zero((std::size_t)N * 4 * 4, 0.0);
  s.uploadVelocity(u, zero, zero);
  std::vector<double> c0((std::size_t)N * 4 * 4);
  double tot0 = 0.0;
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < N; ++x) {
        const double b = std::exp(-std::pow(x - N / 2.0, 2) / (2 * std::pow(N / 12.0, 2)));
        c0[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * 4] = b;
        tot0 += b;
      }
  s.setField("c", c0);
  for (int it = 0; it < 20; ++it)
    s.advanceScalars();
  auto c = s.getField("c");
  double tot = 0.0, mx = -1e30;
  for (double v : c) {
    tot += v;
    mx = std::fmax(mx, v);
  }
  const double cons = std::fabs(tot - tot0) / tot0;
  std::printf("advection conservation rel %.2e  max %.4f (init 1.0)\n", cons, mx);
  CHECK(cons < 1e-10);
  CHECK(mx < 1.0 + 1e-7);  // no overshoot
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
