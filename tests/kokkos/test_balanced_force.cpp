// The balanced-force projection on the STAGGERED grid (doc/collocated_varrho_forces.md §4.6-4.7,
// gate G8).
//
// `set_balanced_force_projection(True)` solves, once per step before the momentum predictor,
// D(O c w G X) = D(O c beta) with the projection's own operator (c = rho0/rho_f, beta = the
// predictor's own face force) and moves X into P. A gradient force is then balanced by the
// pressure BEFORE the implicit momentum operator A = rho_f/dt - mu*Lap acts, so the static
// balances the staggered predictor only APPROACHES at mu > 0 (WO-P's mu*dt^2 residue: A does not
// commute with the discrete gradient at variable rho or next to a wall or a solid) are exact from
// the first step. The option is state-independent, so it changes neither stability nor the
// converged steady state; OFF is byte-identical (gated by the test_vof_* / test_vardensity_*
// outputs and tests/regression/state_hash.py, not here).
//
//   (b) T2 staggered drop, constant kappa, ratio 1 / 10 / 100 / 1000 and the ratio-1000 mu sweep:
//       face < 1e-13 after 30 steps.
//   (c) hydrostatic column, walled, ratio 1000, mu in {0, 1e-3, 1e-2, 1e-1}: dP/dz < 1e-10 after
//       ONE step, face < 1e-12.
//   (d) low-Ca walled drop, 32^3, walls on every face, R = 8, kappa = 0.25, sigma = 0.01, mu = 10,
//       dt = 0.5 capillary_dt: face < 1e-13 ON; OFF reported.
//   (e) immersed solids: (i) a ratio-1000 walled column with a solid sphere (R = 5) crossing the
//       interface, mu = 0.1: face < 1e-12 and dP/dz between fluid cells < 1e-10 after 30 steps;
//       (ii) the drop of (d) cut by an immersed plane 0.5 R below its centre, ratio 1 and 1000,
//       mu = 0.1: face < 1e-12. OFF reported.
//   (f) refusals: porous, ghost, block CSF, inflow/outflow, non-incremental -> named errors; the
//       V8 DEFAULT with an inflow/outflow face resolves OFF (notice) instead of throwing.
#include <cmath>
#include <cstdio>
#include <functional>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::ClosureKind;
using Stag = peclet::flow::Solver<peclet::flow::Staggered>;
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

std::size_t idx(int x, int y, int z, int nx, int ny) {
  return (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
}
double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}
template <class S>
double maxFaceVel(S& s) {
  return std::fmax(std::fmax(maxAbs(s.getFaceVelocity(0)), maxAbs(s.getFaceVelocity(1))),
                   maxAbs(s.getFaceVelocity(2)));
}

// Volume fractions of a sphere (the sampler of test_vof_collocated.cpp).
std::vector<double> sphereC(int n, double R, double cx, double cy, double cz, int sub = 24) {
  std::vector<double> C((std::size_t)n * n * n, 0.0);
  const double w = 1.0 / sub;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int b = 0; b < sub; ++b)
          for (int a = 0; a < sub; ++a) {
            const double px = i + (a + 0.5) * w, py = j + (b + 0.5) * w;
            const double r2 = R * R - (px - cx) * (px - cx) - (py - cy) * (py - cy);
            if (r2 <= 0.0)
              continue;
            const double h = std::sqrt(r2);
            const double lo = std::fmax(cz - h, (double)k), hi = std::fmin(cz + h, (double)k + 1);
            if (hi > lo)
              acc += hi - lo;
          }
        C[idx(i, j, k, n, n)] = acc / (sub * sub);
      }
  return C;
}

// ------------------------------------------------------------------ droplets: (b), (d), (e)(ii)
struct DropCfg {
  double ratio = 1.0, mu = 0.1, sigma = 1.0;
  bool walls = false, plane = false;
  int steps = 30;
};

double runDrop(const DropCfg& c, bool on, long* bfIters = nullptr) {
  const int n = 32;
  const double R = 8.0, cx = n / 2 + 0.13, cy = n / 2 + 0.27, cz = n / 2 + 0.11;
  Stag s(n, n, n);
  s.setRho(c.ratio);
  s.setMu(c.mu);
  s.setDt(1.0);
  s.setVelocityResidualTolerance(0.0);
  if (c.walls)
    for (int f = 0; f < 6; ++f)
      s.setDomainBc(f, 1, 0, 0, 0);
  if (c.plane) {  // an immersed plane 0.5 R below the drop centre: solid below z = cz - R/2
    std::vector<double> sdf((std::size_t)n * n * n);
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
          sdf[idx(x, y, z, n, n)] = (z + 0.5) - (cz - 0.5 * R);
    s.setSolid(sdf, true);
  } else {
    s.setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 10.0));
  }
  s.setPressureChebyshev(true, 500, 1e-14);
  s.enableVof();
  s.setVof(sphereC(n, R, cx, cy, cz));
  s.setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {1.0, c.ratio - 1.0});
  s.setSurfaceTension(c.sigma);
  s.setVofKappaConstant(0.25);
  s.setDt(0.5 * s.capillaryDt());
  if (on)
    s.setBalancedForceProjection(true);
  s.setPressureChebyshev(true, 500, 1e-14);
  long it = 0;
  for (int k = 0; k < c.steps; ++k) {
    s.step();
    it = std::max(it, s.lastBalancedForceIterations());
  }
  if (bfIters)
    *bfIters = it;
  return maxFaceVel(s);
}

// ------------------------------------------------------------------ hydrostatic: (c), (e)(i)
struct HydroResult {
  double faceU = 0, pErr = 0;
};

HydroResult hydrostatic(double ratio, double mu, int steps, bool sphere, bool on) {
  const int N = sphere ? 16 : 8, NZ = sphere ? 32 : 24;
  const double g = 0.1;
  Stag s(N, N, NZ);
  s.setRho(1.0);
  s.setMu(mu);
  s.setDt(1.0);
  s.setVelocityResidualTolerance(0.0);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);
  std::vector<double> sdf((std::size_t)N * N * NZ, 10.0);
  if (sphere)
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const double dx = x + 0.5 - 8.2, dy = y + 0.5 - 7.9, dz = z + 0.5 - 16.3;
          sdf[idx(x, y, z, N, N)] = std::sqrt(dx * dx + dy * dy + dz * dz) - 5.0;
        }
  if (sphere)
    s.setSolid(sdf, true);
  else
    s.setPressureGeometry(sdf);
  auto heavy = [&](int z) { return z < NZ / 2; };
  std::vector<double> fld((std::size_t)N * N * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        fld[idx(x, y, z, N, N)] = heavy(z) ? ratio : 1.0;
  s.addField("rho");
  s.setField("rho", fld);
  s.setDensityMode(true);
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "rho", "", std::vector<double>{0.0, -g});
  if (on)
    s.setBalancedForceProjection(true);
  for (int k = 0; k < steps; ++k)
    s.step();
  HydroResult r;
  r.faceU = maxFaceVel(s);
  const auto p = s.getPressure();
  for (int z = 1; z < NZ; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        if (sdf[idx(x, y, z, N, N)] < 0.0 || sdf[idx(x, y, z - 1, N, N)] < 0.0)
          continue;  // P inside the solid is not a fluid pressure
        const double dp = p[idx(x, y, z, N, N)] - p[idx(x, y, z - 1, N, N)];
        const double rf = 0.5 * ((heavy(z) ? ratio : 1.0) + (heavy(z - 1) ? ratio : 1.0));
        r.pErr = std::fmax(r.pErr, std::fabs(dp + g * rf) / (g * ratio));
      }
  return r;
}

void gateDrop() {
  std::printf("\n=== (b) staggered drop, constant kappa, 30 steps: face |u| OFF -> ON\n");
  for (double ratio : {1.0, 10.0, 100.0, 1000.0}) {
    DropCfg c;
    c.ratio = ratio;
    long it = 0;
    const double off = runDrop(c, false), on = runDrop(c, true, &it);
    std::printf("  ratio %6g  OFF %.3e  ON %.3e   (balanced-force iters max %ld)\n", ratio, off, on,
                it);
    CHECK(on < 1e-13);
  }
  for (double mu : {0.0, 0.01, 0.1}) {
    DropCfg c;
    c.ratio = 1000.0;
    c.mu = mu;
    const double off = runDrop(c, false), on = runDrop(c, true);
    std::printf("  ratio 1000 mu %-5g  OFF %.3e  ON %.3e\n", mu, off, on);
    CHECK(on < 1e-13);
  }
  std::printf("\n=== (d) low-Ca walled drop (sigma 0.01, mu 10, walls on every face), 30 steps\n");
  for (double ratio : {1.0, 1000.0}) {
    DropCfg c;
    c.ratio = ratio;
    c.mu = 10.0;
    c.sigma = 0.01;
    c.walls = true;
    const double off = runDrop(c, false), on = runDrop(c, true);
    std::printf("  ratio %6g  OFF %.3e (reported)  ON %.3e\n", ratio, off, on);
    CHECK(on < 1e-13);
  }
  std::printf(
      "\n=== (e)(ii) constant-kappa sessile cap: the drop cut by an immersed plane, mu 0.1\n");
  for (double ratio : {1.0, 1000.0}) {
    DropCfg c;
    c.ratio = ratio;
    c.plane = true;
    const double off = runDrop(c, false), on = runDrop(c, true);
    std::printf("  ratio %6g  OFF %.3e (reported)  ON %.3e\n", ratio, off, on);
    CHECK(on < 1e-12);
  }
}

void gateHydro() {
  std::printf(
      "\n=== (c) hydrostatic column, walled, ratio 1000: ON after ONE step (OFF 100 steps)\n");
  for (double mu : {0.0, 1e-3, 1e-2, 1e-1}) {
    const auto off = hydrostatic(1000.0, mu, 100, false, false);
    const auto on = hydrostatic(1000.0, mu, 1, false, true);
    std::printf("  mu %-6g  OFF face %.3e dP/dz %.3e   |   ON(1 step) face %.3e dP/dz %.3e\n", mu,
                off.faceU, off.pErr, on.faceU, on.pErr);
    CHECK(on.faceU < 1e-12);
    CHECK(on.pErr < 1e-10);
  }
  std::printf(
      "\n=== (e)(i) ratio-1000 column with a solid sphere (R = 5) across the interface, "
      "mu 0.1, 30 steps\n");
  const auto off = hydrostatic(1000.0, 0.1, 30, true, false);
  const auto on = hydrostatic(1000.0, 0.1, 30, true, true);
  std::printf("  OFF face %.3e dP/dz %.3e (reported)   |   ON face %.3e dP/dz %.3e\n", off.faceU,
              off.pErr, on.faceU, on.pErr);
  CHECK(on.faceU < 1e-12);
  CHECK(on.pErr < 1e-10);
}

void expectThrow(const char* what, const std::function<void()>& f) {
  bool threw = false;
  std::string msg;
  try {
    f();
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  std::printf("  %-34s -> throws: %s%s%.90s\n", what, threw ? "yes" : "NO", threw ? "  " : "",
              msg.c_str());
  CHECK(threw);
}

void gateRefusals() {
  std::printf("\n=== (f) refusals\n");
  const int N = 16;
  const std::vector<double> fluid((std::size_t)N * N * N, 10.0);
  auto base = [&](auto& s) {
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
  };
  expectThrow("porous continuity", [&] {
    Stag s(N, N, N);
    base(s);
    s.setPressureGeometry(fluid);
    s.setPorousContinuity(true);
    s.setBalancedForceProjection(true);
    s.step();
  });
  expectThrow("ghost projection", [&] {
    Stag s(N, N, N);
    base(s);
    s.setGhostProjection(true);
    s.setBalancedForceProjection(true);
    s.setPressureGeometry(fluid);
    s.step();
  });
  expectThrow("block CSF", [&] {
    Stag s(N, N, N);
    base(s);
    s.setPressureGeometry(fluid);
    s.enableVof();
    s.enableVofBlocks({{{N / 2.0, N / 2.0, N / 2.0, 4.0}}});
    s.setSurfaceTension(0.01);
    s.enableVofBlockCsf();
    s.setBalancedForceProjection(true);
    s.step();
  });
  expectThrow("inflow/outflow face", [&] {
    Stag s(N, N, N);
    base(s);
    s.setDomainBc(0, 2, 0.1, 0.0, 0.0);
    s.setDomainBc(1, 3, 0, 0, 0);
    s.setPressureGeometry(fluid);
    s.setBalancedForceProjection(true);
    s.step();
  });
  // The V8 DEFAULT with an inflow/outflow face resolves OFF (a one-time stderr notice), and the
  // step runs; only an EXPLICIT ON with an open face throws the named error.
  {
    Colo s(N, N, N);
    base(s);
    s.setDomainBc(0, 2, 0.1, 0.0, 0.0);
    s.setDomainBc(1, 3, 0, 0, 0);
    s.setPressureGeometry(fluid);
    s.setDensityMode(true);  // V8: the option's default would be ON
    bool threw = false;
    try {
      s.step();
      s.step();
    } catch (const std::exception& e) {
      threw = true;
      std::printf("  V8 default + open face threw: %s\n", e.what());
    }
    std::printf("  V8 default + inflow/outflow: %s, option %s\n", threw ? "THREW" : "ran",
                s.balancedForceProjection() ? "ON" : "OFF");
    CHECK(!threw);
    CHECK(!s.balancedForceProjection());
  }
  expectThrow("V8 + inflow/outflow face, explicit ON", [&] {
    Colo s(N, N, N);
    base(s);
    s.setDomainBc(0, 2, 0.1, 0.0, 0.0);
    s.setDomainBc(1, 3, 0, 0, 0);
    s.setPressureGeometry(fluid);
    s.setDensityMode(true);
    s.setBalancedForceProjection(true);
    s.step();
  });
  expectThrow("non-incremental pressure", [&] {
    Stag s(N, N, N);
    base(s);
    s.setPressureGeometry(fluid);
    s.setIncrementalPressure(false);
    s.setBalancedForceProjection(true);
    s.step();
  });
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateRefusals();
    gateHydro();
    gateDrop();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "\n%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("\nAll staggered balanced-force projection gates passed.\n");
  return 0;
}
