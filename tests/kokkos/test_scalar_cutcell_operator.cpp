// Cut-cell scalar operator and solve (doc/scalar_ibm_design.md §1.4, §1.6, §4, §5.1; WO-3),
// single rank. Structural properties of the assembled operator, independent of the accuracy gates
// (tests/python/test_scalar_cutcell_gates.py):
//
//   guard      — ruling D-WO3-4: a face with a snapped aperture > 0 between an unknown and an
//                interior neighbour always has an unknown neighbour, so the band guard toward a
//                non-unknown cell is a no-op away from the domain boundary;
//   symmetry   — the 7-point part is symmetric: AE(i) == AW(i + 1) (and N/S, T/B) BITWISE;
//   rows       — identity rows are AC = 1, bands 0, rhs 0; steady Neumann rows sum to 0;
//   constants  — Dirichlet walls at g = 1, steady: c == 1 (the probe weights sum to 1); transient,
//                insulating walls: a constant stays constant (§1.6.2);
//   singular   — steady, insulating walls + flux + source: the incompatibility is reported, the
//                projected problem converges and the gauge keeps sum kappa V c (§5.1);
//   budget     — transient and steady, with Robin walls, a source and Dirichlet domain faces: the
//                identity of §1.6.1 closes to round-off;
//   refusals   — a scalar periodic face against a flow wall, and the reverse.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
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

using IbmSolver = peclet::flow::IbmSolver;
constexpr int G = IbmSolver::G;
constexpr int N = 24;

std::vector<double> sphereSdf(double cx, double cy, double cz, double R, bool solidInside) {
  std::vector<double> s((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        s[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            solidInside ? r - R : R - r;
      }
  return s;
}

using HV = Kokkos::View<double*, Kokkos::HostSpace>;
HV host(const peclet::flow::CCField& f) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
}

// A solver with the off-centre sphere; `walls` puts no-slip walls on +-x (else all periodic).
void setup(IbmSolver& s, bool walls) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(0.8);
  if (walls) {
    s.setDomainBc(0, 1, 0, 0, 0);
    s.setDomainBc(1, 1, 0, 0, 0);
  }
  s.setSolid(sphereSdf(11.3, 12.6, 11.9, 6.4, true), false);
}

void testStructure() {
  IbmSolver s(N, N, N);
  setup(s, false);
  s.addScalar("c", 1.3, 1, 50, true);
  s.solveScalarSteady("c");  // insulating walls, no source: b = 0, assembled, nothing to solve
  const auto& gm = s.scalarCutGeometry();
  auto unk = host(gm.unknown);
  const HV sa[3] = {host(gm.sax), host(gm.say), host(gm.saz)};
  auto& sc = s.scalarField("c");
  auto AC = host(sc.AC), b = host(sc.b);
  const HV lo[3] = {host(sc.AW), host(sc.AS), host(sc.AB)};
  const HV hi[3] = {host(sc.AE), host(sc.AN), host(sc.AT)};
  const int ex = N + 2 * G;
  const long st[3] = {1, ex, (long)ex * ex};
  long guarded = 0, asym = 0, badId = 0, badSum = 0, nUnk = 0;
  double maxRow = 0.0;
  for (int z = G; z < N + G; ++z)
    for (int y = G; y < N + G; ++y)
      for (int x = G; x < N + G; ++x) {
        const long i = (long)x + (long)y * st[1] + (long)z * st[2];
        if (!(unk(i) > 0.5)) {
          bool id = AC(i) == 1.0 && b(i) == 0.0;
          for (int a = 0; a < 3; ++a)
            id = id && lo[a](i) == 0.0 && hi[a](i) == 0.0;
          badId += id ? 0 : 1;
          continue;
        }
        ++nUnk;
        double row = AC(i);
        for (int a = 0; a < 3; ++a) {
          row += lo[a](i) + hi[a](i);
          // the periodic box: every neighbour is an interior (or periodic-image) cell
          if (sa[a](i) > 0.0 && !(unk(i - st[a]) > 0.5))
            ++guarded;
          if (sa[a](i + st[a]) > 0.0 && !(unk(i + st[a]) > 0.5))
            ++guarded;
          const long j = i + st[a];
          const int c[3] = {x, y, z};
          if (c[a] + 1 < N + G && unk(j) > 0.5 && hi[a](i) != lo[a](j))
            ++asym;
        }
        maxRow = std::fmax(maxRow, std::fabs(row) / AC(i));
        if (std::fabs(row) > 1e-14 * AC(i))
          ++badSum;
      }
  std::printf(
      "structure: %ld unknowns, guard fires on %ld interior faces, %ld asymmetric pairs, "
      "%ld bad identity rows, max |row sum|/AC %.1e\n",
      nUnk, guarded, asym, badId, maxRow);
  CHECK(nUnk > 0);
  CHECK(guarded == 0);
  CHECK(asym == 0);
  CHECK(badId == 0);
  CHECK(badSum == 0);
}

double maxDevUnknown(IbmSolver& s, double v) {
  auto unk = host(s.scalarCutGeometry().unknown);
  auto c = host(s.scalarField("c").c);
  const int ex = N + 2 * G;
  double m = 0.0;
  for (int z = G; z < N + G; ++z)
    for (int y = G; y < N + G; ++y)
      for (int x = G; x < N + G; ++x) {
        const long i = (long)x + (long)y * ex + (long)z * ex * ex;
        if (unk(i) > 0.5)
          m = std::fmax(m, std::fabs(c(i) - v));
      }
  return m;
}

void testConstants() {
  {  // Dirichlet walls at g = 1, steady: A 1 = b exactly up to round-off
    IbmSolver s(N, N, N);
    setup(s, false);
    s.addScalar("c", 0.9, 1, 50, true);
    s.setScalarWall("c", 1, 1.0, 0.0, -1);
    s.setScalarTolerance("c", 1e-12);
    s.setScalarMaxIterations("c", 3000);
    s.solveScalarSteady("c");
    const double d = maxDevUnknown(s, 1.0);
    std::printf("constants: steady Dirichlet g = 1 -> max|c - 1| = %.2e (%d iterations)\n", d,
                s.scalarField("c").cut->iterations);
    CHECK(d <= 1e-9);
    CHECK(s.scalarField("c").cut->converged);
  }
  {  // transient, insulating walls: c == 5 stays
    IbmSolver s(N, N, N);
    setup(s, false);
    s.addScalar("c", 0.9, 1, 50, true);
    std::vector<double> five((std::size_t)N * N * N, 5.0);
    s.setField("c", five);
    for (int k = 0; k < 3; ++k)
      s.advanceScalars();
    const double d = maxDevUnknown(s, 5.0);
    std::printf("constants: transient insulating, c = 5 -> max|c - 5| = %.2e\n", d);
    CHECK(d <= 1e-12);
  }
}

void testSingular() {
  IbmSolver s(N, N, N);
  setup(s, false);
  s.addScalar("c", 0.9, 1, 50, true);
  s.setScalarWall("c", 0, 0.1, 0.0, -1);  // a flux into the fluid
  s.setScalarSource("c", 0.05);           // and a source: incompatible in a periodic box
  s.setScalarMaxIterations("c", 3000);
  s.solveScalarSteady("c");
  const auto& st = *s.scalarField("c").cut;
  auto unk = host(s.scalarCutGeometry().unknown);
  auto kap = host(s.scalarCutGeometry().kappa);
  auto c = host(s.scalarField("c").c);
  double kc = 0.0, kac = 0.0;
  const int ex = N + 2 * G;
  for (int z = G; z < N + G; ++z)
    for (int y = G; y < N + G; ++y)
      for (int x = G; x < N + G; ++x) {
        const long i = (long)x + (long)y * ex + (long)z * ex * ex;
        if (unk(i) > 0.5) {
          kc += kap(i) * c(i);
          kac += kap(i) * std::fabs(c(i));
        }
      }
  std::printf(
      "singular: incompatibility %.3e, %d iterations, converged %d, residual %.2e, "
      "sum kappa c = %.2e (sum kappa |c| = %.2e)\n",
      st.incompatibility, st.iterations, (int)st.converged, st.residual, kc, kac);
  CHECK(st.singular);
  CHECK(st.incompatibility > 0.1);
  CHECK(st.converged);
  CHECK(std::fabs(kc) <= 1e-10 * kac);
}

void testBudget() {
  for (int steady = 0; steady < 2; ++steady) {
    IbmSolver s(N, N, N);
    setup(s, true);
    s.addScalar("c", 1.1, 1, 50, true);
    s.setScalarBc("c", 0, 2, 0.25);
    s.setScalarBc("c", 1, 2, 1.5);
    s.setScalarWall("c", 2, 2.0, 0.5, -1);  // robin: k = 0.5, g = 2
    s.setScalarSource("c", 0.3);
    std::vector<double> init((std::size_t)N * N * N);
    for (std::size_t k = 0; k < init.size(); ++k)
      init[k] = 0.5 + 0.25 * std::sin(0.37 * (double)k);
    s.setField("c", init);
    if (steady)
      s.solveScalarSteady("c");
    else
      for (int k = 0; k < 2; ++k)
        s.advanceScalars();
    const auto b = s.scalarBudget("c");
    // ruling D-WO6-3 (the D-WO5c-1 convention): the identity relative to the GROSS budget, the
    // sum of the terms' absolute values (the source is uniform: |source_in|), bound 1e-11 — it is
    // reduction-order round-off and must not depend on the thread count. Cell units: the rates
    // are internal, and a transient budget is in mass units (dt times the rates).
    const double dtB = steady ? 1.0 : s.scalarField("c").cut->dt;
    const double scale =
        std::fabs(b.dMass) +
        dtB * ((std::fabs(b.wallIn) + std::fabs(b.boundaryIn)) + std::fabs(b.sourceIn));
    std::printf(
        "budget (%s): d_mass %.6e wall %.6e boundary %.6e source %.6e defect %.2e "
        "identity %.2e (rel %.1e)\n",
        steady ? "steady" : "transient", b.dMass, b.wallIn, b.boundaryIn, b.sourceIn, b.defect,
        b.identityError, std::fabs(b.identityError) / scale);
    CHECK(std::fabs(b.identityError) <= 1e-11 * scale);
    CHECK(std::fabs(b.wallIn) > 0.0 && std::fabs(b.boundaryIn) > 0.0 && b.sourceIn > 0.0);
    // the per-body wall flux is the budget's wall term
    const auto wf = s.scalarWallFlux("c");
    CHECK(wf.size() == 1 && std::fabs(wf[0] - b.wallIn) <= 1e-13 * std::fabs(b.wallIn));
  }
}

void testRefusals() {
  {  // scalar Dirichlet face against a periodic flow face
    IbmSolver s(N, N, N);
    setup(s, false);
    s.addScalar("c", 1.0, 1, 50, true);
    s.setScalarBc("c", 2, 2, 1.0);
    bool threw = false;
    try {
      s.advanceScalars();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
  }
  {  // flow wall on -x, scalar left periodic
    IbmSolver s(N, N, N);
    setup(s, true);
    s.addScalar("c", 1.0, 1, 50, true);
    bool threw = false;
    try {
      s.solveScalarSteady("c");
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
  }
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    testStructure();
    testConstants();
    testSingular();
    testBudget();
    testRefusals();
  }
  Kokkos::finalize();
  std::printf(failures ? "%d failure(s)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
