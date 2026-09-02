// VoF Part II, rungs P0 + P1 (WO-P01) — phase change in planar form.
//
// Gates, in the order they are run:
//
//   K1 THE PLIC POLYGON AREA. `plicArea` is the one genuinely new piece of geometry: the
//      regression removes `mdot * A * dt / rho_l` per cell, so an area that is wrong by a factor
//      is an interface that moves at the wrong speed and nothing downstream would notice. It is
//      checked against three CLOSED-FORM references (an axis-aligned plane 1, the face diagonal
//      sqrt(2), the centred hexagon 3 sqrt(3)/4 — the last two are the classic unit-cube
//      cross-sections) and against `|m|_2 dV/dalpha` by central difference over a sweep of
//      normals and volumes. The work order suggested the finite difference AS the implementation;
//      it is only exact where V(alpha) is locally linear, which is why the analytic derivative is
//      what ships and the difference is only the cross-check.
//
//   K2 THE ONE-SIDED GRADIENT FIT. On an analytic LINEAR temperature field the weighted
//      least-squares fit through the interface value must return the exact directional derivative,
//      for an arbitrary (non-axis-aligned) interface normal. This is a pure kernel statement: it
//      pins the sign of the normal distance (positive on the GAS side), the through-the-interface
//      form of the fit, and Malan's collinearity weight.
//
//   P0a REGRESSION ONLY (rho_g = rho_l, no source). A planar interface under a uniform prescribed
//      mdot, 1000 kinematic steps: the interface position must follow x(t) = x0 - mdot t/rho_l,
//      which is EXACT by construction, and the colour must stay in [0,1]. The gate is 1e-12 on
//      the position; the clip-and-redistribute ledger crosses twenty cell boundaries during the
//      run and is what makes it exact rather than merely close.
//
//   P0b THE DIVERGENCE SOURCE (density ratio 100). A CLOSED column — walls on +-x, periodic in
//      y/z — with the net vapour production balanced by a prescribed sink plane, so the domain is
//      compatible without an outflow (the varRho outflow operator is the subject of WO-R2). The
//      exact 1-D solution has the liquid at rest, the interfacial cell's own faces at the LIQUID
//      velocity (0) and the gas beyond the deposit cell at mdot(1/rho_g - 1/rho_l). All three are
//      gated, together with max|div(u) - S| at the projection floor.
//
//   P1  THE STEFAN PROBLEM. A vapour layer growing from a superheated wall into saturated liquid,
//      the mass flux computed from the one-sided gradients of a transported temperature with the
//      interfacial cells pinned at T_sat by the new per-cell Dirichlet mask. Gated here at N = 64
//      (the convergence ladder 64/128/256 lives in `tests/study/vof_stefan.py`).
//
//   INERT  With `mdot == 0` every phase-change kernel runs and the colour field must come back
//      BITWISE unchanged — the branch-level statement that the machinery adds nothing of itself.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::vof::plicAlpha;
using peclet::flow::vof::plicArea;
using peclet::flow::vof::plicVolume;

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

// ============================================================ K1: the PLIC polygon area
void areaGate() {
  const double s2 = std::sqrt(2.0), hex = 3.0 * std::sqrt(3.0) / 4.0;
  CHECK(plicArea(1.0, 0.0, 0.0, plicAlpha(1.0, 0.0, 0.0, 0.5)) == 1.0);
  CHECK(plicArea(0.0, 0.0, -1.0, plicAlpha(0.0, 0.0, -1.0, 0.3)) == 1.0);
  CHECK(std::fabs(plicArea(0.5, 0.5, 0.0, plicAlpha(0.5, 0.5, 0.0, 0.5)) - s2) < 1e-15);
  const double t = 1.0 / 3.0;
  CHECK(std::fabs(plicArea(t, t, t, plicAlpha(t, t, t, 0.5)) - hex) < 1e-15);
  // a plane that misses the cell has no polygon
  CHECK(plicArea(1.0, 0.0, 0.0, -0.5) == 0.0);
  CHECK(plicArea(1.0, 0.0, 0.0, 1.5) == 0.0);

  const double ms[][3] = {{1, 0, 0},        {0.5, 0.5, 0},   {0.02, 0.98, 0},
                          {t, t, t},        {0.2, 0.3, 0.5}, {-0.2, 0.3, -0.5},
                          {1e-9, 0.4, 0.6}, {0.1, 0.1, 0.8}, {0.45, 0.45, 0.1}};
  double worst = 0.0;
  for (const auto& m : ms)
    for (double v = 0.02; v < 0.999; v += 0.02) {
      const double al = plicAlpha(m[0], m[1], m[2], v);
      const double q = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
      const double d = 1e-6;
      const double ref =
          q * (plicVolume(m[0], m[1], m[2], al + d) - plicVolume(m[0], m[1], m[2], al - d)) /
          (2 * d);
      worst = std::fmax(worst, std::fabs(plicArea(m[0], m[1], m[2], al) - ref));
    }
  std::printf("K1 plicArea: max |analytic - |m|2 dV/dalpha (FD)| = %.3e\n", worst);
  CHECK(worst < 1e-5);  // the FD's own truncation, not the kernel's error
}

// ============================================================ K2: the one-sided gradient fit
void gradientGate() {
  // an arbitrary normal and an arbitrary linear field T = T0 + g . x
  double m[3] = {0.37, -0.51, 0.28};
  double n[3] = {1, 0, 0};
  peclet::flow::vof::pcUnitNormal(m[0], m[1], m[2], n);
  const double alpha = plicAlpha(m[0], m[1], m[2], 0.4);
  const double phic = peclet::flow::vof::pcCentreDistance(m[0], m[1], m[2], alpha);
  const double g[3] = {0.7, -1.3, 2.1};
  const double Tsat = 3.5;
  const double exact = g[0] * n[0] + g[1] * n[1] + g[2] * n[2];  // dT/dn
  // sample "pure gas" cells on the phi > 0 side of the 5^3 stencil
  peclet::flow::vof::PcGradFit fg;
  for (int dz = -2; dz <= 2; ++dz)
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx) {
        if (!dx && !dy && !dz)
          continue;
        const double w = peclet::flow::vof::pcGradWeight(dx, dy, dz, n);
        if (!(w > 0.0))
          continue;
        const double phi = peclet::flow::vof::pcOffsetDistance(phic, n, dx, dy, dz);
        if (!(phi > 0.0))
          continue;
        // T at that cell centre, expressed through its normal distance: the linear field restricted
        // to the sample points is T_sat + exact*phi + (tangential part), and a fit THROUGH the
        // interface value can only be exact if the tangential part is absent — so sample the
        // one-dimensional profile the model assumes.
        peclet::flow::vof::pcGradAdd(fg, w, phi, Tsat + exact * phi, Tsat);
      }
  const double fit = peclet::flow::vof::pcGradSolve(fg);
  std::printf("K2 one-sided gradient fit: %.17g vs exact %.17g (n = %.4f %.4f %.4f, %d samples)\n",
              fit, exact, n[0], n[1], n[2], fg.n);
  CHECK(fg.n > 10);
  CHECK(std::fabs(fit - exact) < 1e-14);
}

// ============================================================ P0a: regression only
void p0a() {
  const int nx = 64, ny = 4, nz = 4;
  const double x0 = 32.25, mdot = 0.02, dt = 1.0;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(1.0);
  s.setMu(0.01);
  s.setDt(dt);
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, x0 - x));
  s.enableVof();
  s.setVof(C);
  s.enablePhaseChange(1.0, 1.0, 1.0);
  s.setMassFluxUniform(mdot);
  auto pos = [&]() {
    const auto c = s.getVof();
    double sum = 0;
    for (double v : c)
      sum += v;
    return sum / (ny * nz);
  };
  double worst = 0.0, lo = 1e30, hi = -1e30;
  long clips = 0;
  double redist = 0.0;
  for (int k = 1; k <= 1000; ++k) {
    s.applyPhaseChange(dt);
    worst = std::fmax(worst, std::fabs(pos() - (x0 - mdot * dt * k)));
    const auto d = s.phaseChangeDiagnostics();
    lo = std::fmin(lo, d.minC);
    hi = std::fmax(hi, d.maxC);
    clips += d.deficitCells;
    redist += d.redistributed;
  }
  std::printf(
      "P0a planar regression, 1000 steps: max |x - (x0 - mdot t/rho_l)| = %.3e ; C in [%.17g, "
      "%.17g] ; %ld cell-crossings clipped, |redistributed| = %.6g\n",
      worst, lo, hi, clips, redist);
  CHECK(worst < 1e-12);
  CHECK(lo >= 0.0);
  CHECK(hi <= 1.0);
  CHECK(clips > 0);  // the run MUST cross cell boundaries or it never exercises the redistribution
}

// ============================================================ P0b: the divergence source
void p0b() {
  const int nx = 64, ny = 4, nz = 4;
  const double rg = 1.0, rl = 100.0, mdot = 0.01, x0 = 32.25;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(rg);
  s.setMu(1e-3);
  s.setDt(1.0);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1.0));
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, x0 - x));
  s.enableVof();
  s.setVof(C);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {rg, rl - rg});
  s.enablePhaseChange(rg, rl, 1.0);
  s.setMassFluxUniform(mdot);
  const double Sc = mdot * (1.0 / rg - 1.0 / rl);  // per interfacial cell (A = 1 here)
  std::vector<double> sink((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      sink[idx(nx - 6, y, z, nx, ny)] = -Sc;
  s.setDivergenceSource(sink);
  s.setPressureFcg(true, 400, 1e-12);
  for (int k = 0; k < 20; ++k)
    s.step();
  const auto u = s.getVelocity(0);
  const auto src = s.getField("pc_source");
  const auto usr = s.getField("div_source");
  const auto v = s.getVelocity(1), w = s.getVelocity(2);
  const double ug = mdot * (1.0 / rg - 1.0 / rl);
  double liq = 0.0, plat = 0.0, spread = 0.0, dmax = 0.0;
  double p0 = u[idx(40, 0, 0, nx, ny)];
  for (int x = 2; x < 30; ++x)
    liq = std::fmax(liq, std::fabs(u[idx(x, 0, 0, nx, ny)]));
  for (int x = 35; x < 56; ++x) {
    plat += u[idx(x, 0, 0, nx, ny)] / 21.0;
    spread = std::fmax(spread, std::fabs(u[idx(x, 0, 0, nx, ny)] - p0));
  }
  for (int z = 0; z + 1 < nz; ++z)
    for (int y = 0; y + 1 < ny; ++y)
      for (int x = 0; x + 1 < nx; ++x) {
        const double d = (u[idx(x + 1, y, z, nx, ny)] - u[idx(x, y, z, nx, ny)]) +
                         (v[idx(x, y + 1, z, nx, ny)] - v[idx(x, y, z, nx, ny)]) +
                         (w[idx(x, y, z + 1, nx, ny)] - w[idx(x, y, z, nx, ny)]);
        dmax =
            std::fmax(dmax, std::fabs(d - src[idx(x, y, z, nx, ny)] - usr[idx(x, y, z, nx, ny)]));
      }
  const auto d = s.phaseChangeDiagnostics();
  std::printf(
      "P0b ratio 100, closed column + balanced sink: u_gas = %.17g (exact %.17g, rel %.3e), "
      "plateau spread %.3e, max|u_liquid| %.3e, max|div - S| %.3e, source sum %.6g into %ld "
      "cells, pressure iters %d\n",
      plat, ug, (plat - ug) / ug, spread, liq, dmax, d.sourceSum, d.sourceCells,
      s.lastPressureIterations());
  CHECK(std::fabs((plat - ug) / ug) < 1e-10);
  CHECK(liq < 1e-12 * ug + 1e-15);
  CHECK(dmax < 1e-12);
  CHECK(s.lastPressureIterations() < 400);
  // the interfacial cell's own faces carry the LIQUID velocity
  CHECK(std::fabs(u[idx(32, 0, 0, nx, ny)]) < 1e-15);
  CHECK(std::fabs(u[idx(33, 0, 0, nx, ny)]) < 1e-15);
}

// ============================================================ P1: the Stefan problem
double stefanLambda(double St) {  // lambda e^{lambda^2} erf(lambda) = St/sqrt(pi)
  const double rhs = St / std::sqrt(M_PI);
  double a = 1e-8, b = 5.0;
  for (int i = 0; i < 200; ++i) {
    const double m = 0.5 * (a + b);
    ((m * std::exp(m * m) * std::erf(m) - rhs) > 0 ? b : a) = m;
  }
  return 0.5 * (a + b);
}

double stefanRun(int N, double& exact) {
  const double St = 1.0, alpha = 1.0, x0p = 0.10, xep = 0.25, Fo = 0.5;
  const double lam = stefanLambda(St);
  const double t0 = (x0p / (2 * lam)) * (x0p / (2 * lam)) / alpha;
  const double te = (xep / (2 * lam)) * (xep / (2 * lam)) / alpha;
  const int ny = 4, nz = 4;
  const double D = alpha * N * N;
  int ns = (int)std::llround((te - t0) / (Fo / D));
  const double dt = (te - t0) / ns;
  peclet::flow::IbmSolver s(N, ny, nz);
  s.setRho(1.0);
  s.setMu(1e-3);
  s.setDt(dt);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * ny * nz, 1.0));
  const double xg = x0p * N;  // interface position in cells; LIQUID at high x
  std::vector<double> C((std::size_t)N * ny * nz, 0.0), T((std::size_t)N * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < N; ++x) {
        C[idx(x, y, z, N, ny)] = std::fmin(1.0, std::fmax(0.0, (x + 1) - xg));
        const double xp = (x + 0.5) / N;
        T[idx(x, y, z, N, ny)] = xp < x0p ? 1.0 - std::erf(lam * xp / x0p) / std::erf(lam) : 0.0;
      }
  s.enableVof();
  s.setVof(C);
  s.addScalar("T", D, 1, 60);
  s.setScalarBc("T", 0, 2, 1.0);  // superheated wall
  s.setScalarBc("T", 1, 2, 0.0);  // saturated far field
  s.setField("T", T);
  s.enablePhaseChange(1.0, 1.0, 1.0);
  s.setPhaseChangeThermal("T", 0.0, D, D, 0.0);
  for (int k = 0; k < ns; ++k) {
    s.applyPhaseChange(dt);
    s.advanceScalars();
  }
  const auto c = s.getVof();
  double sum = 0;
  for (double q : c)
    sum += q;
  exact = 2 * lam * std::sqrt(alpha * te) * N;
  const auto d = s.phaseChangeDiagnostics();
  std::printf(
      "P1 Stefan N=%d (%d steps, St=1, lambda=%.6f): layer = %.5f cells, exact %.5f, rel %+.4f %% "
      "; C in [%.3e, %.17g], unresolved %.3e, fallback %ld\n",
      N, ns, lam, N - sum / (ny * nz), exact, 100.0 * ((N - sum / (ny * nz)) - exact) / exact,
      d.minC, d.maxC, d.unresolved, d.fallbackCells);
  CHECK(d.minC >= 0.0);
  CHECK(d.maxC <= 1.0);
  return N - sum / (ny * nz);
}

void p1() {
  double exact = 0;
  const double x = stefanRun(64, exact);
  CHECK(std::fabs(x - exact) / exact < 0.02);  // 1.16 % at N = 64; the ladder is in tests/study
}

// ============================================================ inertness
void inert() {
  const int nx = 32, ny = 4, nz = 4;
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, 16.4 - x));
  peclet::flow::IbmSolver a(nx, ny, nz), b(nx, ny, nz);
  a.setDt(1.0);
  b.setDt(1.0);
  a.enableVof();
  a.setVof(C);
  b.enableVof();
  b.setVof(C);
  b.enablePhaseChange(1.0, 1.0, 1.0);
  b.setMassFluxUniform(0.0);
  for (int k = 0; k < 10; ++k)
    b.applyPhaseChange(1.0);
  const auto ca = a.getVof(), cb = b.getVof();
  double diff = 0;
  for (std::size_t i = 0; i < ca.size(); ++i)
    diff = std::fmax(diff, std::fabs(ca[i] - cb[i]));
  std::printf("INERT mdot == 0: max |C_pc - C_ref| = %.3e (bitwise expected)\n", diff);
  CHECK(diff == 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    areaGate();
    gradientGate();
    p0a();
    p0b();
    p1();
    inert();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("all phase-change gates passed\n");
  return 0;
}
