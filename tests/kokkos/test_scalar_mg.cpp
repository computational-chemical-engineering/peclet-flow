// ScalarMG, the cut-cell scalar preconditioner (doc/scalar_ibm_design.md §5.2; WO-4), single rank.
//
//   levels       — the level table equals VelocityMG::init's on the same grid and metric (inner
//                  dims and coarsening ratio per level), isotropic and anisotropic, odd and
//                  short axes included (§5.2: "verified identical by a unit test");
//   coarse       — the rediscretized coarse surrogate of a Dirichlet sphere: symmetric by
//                  construction (face form), identity rows exactly at pinned cells, an M-matrix
//                  (positive diagonal, non-positive faces, weakly diagonally dominant on every
//                  unknown row), and the face coefficient of the constant-Lam single-phase case
//                  equal to CutcellMG's rule w_a(L) <Lam a>;
//   level rule   — transient with dt D / h^2 < 1 uses level 0 alone, steady the full table;
//   contraction  — the guard of design Amendment A1 (the power estimate of rho(I - M^-1 S), the
//                  geometric mean of the last 10 of 30): (C1) < 1 on every geometry; (C2) <= 0.35
//                  on box geometries (G1, G2, G3a at every rung), Neumann geometries (G5b's array
//                  on multigrid-friendly n) and no-solid geometries; (C3) <= 0.75 on a periodic
//                  box whose only sink is one Dirichlet sphere. Krylov counts stay the primary
//                  gate (tests/python/test_scalar_cutcell_gates.py). WO-5 adds the C3 problem
//                  advecting (steady implicit FOU, peak cell Peclet 1 and 10), under the same C3.
#include <cmath>
#include <cstdio>
#include <cstring>
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

using IbmSolver = peclet::flow::IbmSolver;
using peclet::flow::C3;
using peclet::flow::CCField;
using peclet::flow::ScalarMG;
using peclet::flow::VelocityMG;
using HV = Kokkos::View<double*, Kokkos::HostSpace>;
HV host(const CCField& f) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
}

void testLevels() {
  struct Case {
    int n[3];
    double hp[3];
  };
  const Case cases[] = {
      {{64, 64, 64}, {1, 1, 1}},   {{48, 40, 12}, {1, 1, 1}}, {{128, 128, 4}, {1, 1, 1}},
      {{30, 64, 7}, {1, 1, 1}},    {{77, 77, 77}, {1, 1, 1}}, {{64, 64, 32}, {1, 1, 2}},
      {{20, 24, 48}, {1, 1.5, 3}}, {{96, 32, 64}, {2, 1, 1}},
  };
  for (const Case& c : cases) {
    const double w[3] = {1.0 / (c.hp[0] * c.hp[0]), 1.0 / (c.hp[1] * c.hp[1]),
                         1.0 / (c.hp[2] * c.hp[2])};
    VelocityMG vmg;
    vmg.setMetric(w, c.hp);
    vmg.init(c.n[0], c.n[1], c.n[2], 64);
    ScalarMG smg;
    smg.setMetric(w, c.hp);
    smg.init(c.n[0], c.n[1], c.n[2], [](CCField) {});
    bool same = vmg.levels() == smg.levels();
    for (int L = 0; same && L < smg.levels(); ++L) {
      const auto& a = vmg.level(L);
      const auto& b = smg.level(L);
      same = a.inner.x == b.inner.x && a.inner.y == b.inner.y && a.inner.z == b.inner.z &&
             a.ratio.x == b.ratio.x && a.ratio.y == b.ratio.y && a.ratio.z == b.ratio.z &&
             a.cfac.x == b.cfac.x && a.cfac.y == b.cfac.y && a.cfac.z == b.cfac.z;
    }
    std::printf("levels %dx%dx%d h' (%g,%g,%g): %d levels, bottom %dx%dx%d  %s\n", c.n[0], c.n[1],
                c.n[2], c.hp[0], c.hp[1], c.hp[2], smg.levels(),
                smg.level(smg.levels() - 1).inner.x, smg.level(smg.levels() - 1).inner.y,
                smg.level(smg.levels() - 1).inner.z, same ? "== VelocityMG" : "DIFFERS");
    CHECK(same);
  }
}

// Sphere of radius R (cells) at c in an n^3 box; solid inside.
std::vector<double> sphereSdf(int n, double cx, double cy, double cz, double R) {
  std::vector<double> s((std::size_t)n * n * n);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - cz;
        s[(std::size_t)x + (std::size_t)y * n + (std::size_t)z * n * n] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return s;
}

// G1's configuration on an n^3 box (cells): Dirichlet sphere R = n/8 ... set by caller.
void g1Setup(IbmSolver& s, int n, double R, double off) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(1.0);
  for (int f = 0; f < 6; ++f)
    s.setDomainBc(f, 1, 0.0, 0.0, 0.0);
  s.setSolid(sphereSdf(n, 0.5 * n + off, 0.5 * n - 0.6 * off, 0.5 * n + 0.3 * off, R), false);
  s.addScalar("c", 0.7, 1, 50, true);
  for (int f = 0; f < 6; ++f)
    s.setScalarBc("c", f, 2, 0.25);
  s.setScalarWall("c", 1, 1.0, 0.0, -1);
}

void testCoarse() {
  const int n = 32;
  IbmSolver s(n, n, n);
  g1Setup(s, n, 8.0, 0.37);
  s.solveScalarSteady("c");
  auto& st = *s.scalarField("c").cut;
  const ScalarMG& mg = *st.mg;
  CHECK(st.mgLevels == mg.levels() && mg.levels() == 5);  // 32 -> 16 -> 8 -> 4 -> 2
  long pinned = 0, rows = 0, badId = 0, badMM = 0, badFace = 0;
  for (int L = 1; L < mg.levels(); ++L) {
    const auto& c = mg.level(L);
    const auto& f = mg.level(L - 1);
    auto AC = host(c.AC), AX = host(c.AFX), AY = host(c.AFY), AZ = host(c.AFZ), U = host(c.unk);
    auto PX = host(c.px);
    const long sy = c.ext.x, sz = (long)c.ext.x * c.ext.y;
    const double wx = 1.0 / ((double)c.cfac.x * c.cfac.x);
    // CutcellMG's coarse face rule on the finer level's products (the recursion, checked once)
    HV FP = L >= 2 ? host(f.px) : HV();
    for (int z = c.g; z < c.ext.z - c.g; ++z)
      for (int y = c.g; y < c.ext.y - c.g; ++y)
        for (int x = c.g; x < c.ext.x - c.g; ++x) {
          const long i = x + y * sy + z * sz;
          ++rows;
          if (!(U(i) > 0.5)) {
            ++pinned;
            if (!(AC(i) == 1.0 && AX(i) == 0.0 && AY(i) == 0.0 && AZ(i) == 0.0))
              ++badId;
            continue;
          }
          const double off = -(AX(i) + AX(i + 1) + AY(i) + AY(i + sy) + AZ(i) + AZ(i + sz));
          const bool nonpos = AX(i) <= 0 && AX(i + 1) <= 0 && AY(i) <= 0 && AY(i + sy) <= 0 &&
                              AZ(i) <= 0 && AZ(i + sz) <= 0;
          if (!(AC(i) > 0.0 && nonpos && AC(i) >= off * (1.0 - 1e-14)))
            ++badMM;
          if (!(AX(i) == -(wx * PX(i))))
            ++badFace;
          if (L >= 2) {
            const long fsy = f.ext.x, fsz = (long)f.ext.x * f.ext.y;
            const int fx = 2 * (x - c.g) + f.g, fy = 2 * (y - c.g) + f.g, fz = 2 * (z - c.g) + f.g;
            double sxs = 0.0;
            for (int a = 0; a < 2; ++a)
              for (int b = 0; b < 2; ++b)
                sxs += FP(fx + (fy + a) * fsy + (fz + b) * fsz);
            if (!(PX(i) == sxs / 4.0))
              ++badFace;
          }
        }
  }
  std::printf(
      "coarse: %ld rows (%ld pinned): identity-row defects %ld, M-matrix defects %ld, "
      "face-rule defects %ld\n",
      rows, pinned, badId, badMM, badFace);
  CHECK(pinned > 0 && badId == 0 && badMM == 0 && badFace == 0);
}

void testLevelRule() {
  const int n = 32;
  IbmSolver s(n, n, n);
  g1Setup(s, n, 8.0, 0.37);
  s.setDt(0.9 / 0.7);  // dt D / h^2 = 0.9 -> kappa_A = 11.8 < 13
  s.advanceScalars();
  const int small = s.scalarField("c").cut->mgLevels;
  s.setDt(1.05 / 0.7);  // 1.05 -> kappa_A = 13.6
  s.advanceScalars();
  const int at = s.scalarField("c").cut->mgLevels;
  s.solveScalarSteady("c");
  const int steady = s.scalarField("c").cut->mgLevels;
  std::printf("level rule: dt D/h^2 = 0.9 -> %d level(s), 1.05 -> %d, steady -> %d\n", small, at,
              steady);
  CHECK(small == 1 && at == 5 && steady == 5);
}

double contraction(IbmSolver& s, const char* tag) {
  auto& st = *s.scalarField("c").cut;
  const std::size_t n = st.SAC.extent(0);
  CCField e("e", n), t("t", n), z("z", n);
  const std::vector<double> r = st.mg->contraction(30, e, t, z);
  double gm = 1.0;
  const int tail = 10;
  for (int k = (int)r.size() - tail; k < (int)r.size(); ++k)
    gm *= r[(std::size_t)k];
  gm = std::pow(gm, 1.0 / tail);
  std::printf(
      "%s: %d levels, ||e_k||/||e_k-1||: first %.3f, k=10 %.3f, k=30 %.3f; "
      "geometric mean of the last %d: %.3f\n",
      tag, st.mgLevels, r[0], r[9], r.back(), tail, gm);
  return gm;
}

// The concentric-shells scene of G2 (cell units): instance 0 a sphere Ri, instance 1 a huge box
// minus a sphere Ro (solid outside Ro), both centred at c.
void shellsScene(IbmSolver& s, double c, double Ri, double Ro) {
  const std::vector<int> ni = {1, -1, -1, 3, -1, -1, 1, -1, -1, 34, 1, 2};
  std::vector<double> nr(4 * 16, 0.0);
  for (int k = 0; k < 4; ++k) {
    nr[16 * k + 14] = 1.0;
    nr[16 * k + 15] = 1.0;
  }
  nr[0] = Ri;
  nr[16 + 0] = nr[16 + 1] = nr[16 + 2] = 1e5;
  nr[32 + 0] = Ro;
  const std::vector<int> ii = {0, -1, 3, -1};
  std::vector<double> ir(2 * 18, 0.0);
  for (int k = 0; k < 2; ++k) {
    ir[18 * k + 0] = ir[18 * k + 1] = ir[18 * k + 2] = c;
    ir[18 * k + 6] = 1.0;
    ir[18 * k + 7] = 1.0;
  }
  s.setScene(ni, nr, ii, ir, false);
  s.setSolidFromScene(false);
}

void walls(IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(1.0);
  for (int f = 0; f < 6; ++f)
    s.setDomainBc(f, 1, 0.0, 0.0, 0.0);
}

// rho on one configured problem; class 2 -> C2 (<= 0.35), 3 -> C3 (<= 0.75), always C1 (< 1).
void guardRow(IbmSolver& s, const char* tag, int cls) {
  s.solveScalarSteady("c");
  char t[160];
  std::snprintf(t, sizeof t, "%-44s %2d it", tag, s.scalarField("c").cut->iterations);
  const double rho = contraction(s, t);
  CHECK(rho < 1.0);
  if (cls == 2)
    CHECK(rho <= 0.35);
  if (cls == 3)
    CHECK(rho <= 0.75);
}

void testContraction() {
  char tag[96];
  for (int n : {32, 64, 128}) {  // G1: Dirichlet sphere R = n/4, Dirichlet box
    IbmSolver s(n, n, n);
    g1Setup(s, n, 0.25 * n, 0.37);
    std::snprintf(tag, sizeof tag, "C2 G1 R/h=%d (%d^3)", n / 4, n);
    guardRow(s, tag, 2);
  }
  const int g2box[3] = {48, 80, 128};
  for (int k = 0; k < 3; ++k) {  // G2 on its multigrid-friendly boxes (ruling D-WO4-1)
    const int Rih = 6 << k, n = g2box[k];
    IbmSolver s(n, n, n);
    walls(s);
    shellsScene(s, 0.5 * n + 0.29, Rih, 2.5 * Rih);
    s.addScalar("c", 0.8, 1, 50, true);
    for (int f = 0; f < 6; ++f)
      s.setScalarBc("c", f, 1, 0.0);  // the box is solid (outside Ro): Neumann
    s.setScalarWall("c", 0, 0.3, 0.0, 0);
    s.setScalarWall("c", 1, 0.0, 0.0, 1);
    std::snprintf(tag, sizeof tag, "C2 G2 Ri/h=%d (%d^3)", Rih, n);
    guardRow(s, tag, 2);
  }
  for (int n : {32, 64, 128}) {  // G3a, Da = 1: Robin sphere, Dirichlet box
    IbmSolver s(n, n, n);
    g1Setup(s, n, 0.25 * n, 0.37);
    s.setScalarWall("c", 2, 0.0, 0.7 / (0.25 * n), -1);  // k = Da D / R
    std::snprintf(tag, sizeof tag, "C2 G3a Da=1 R/h=%d (%d^3)", n / 4, n);
    guardRow(s, tag, 2);
  }
  {  // Neumann sphere, Dirichlet box
    IbmSolver s(64, 64, 64);
    g1Setup(s, 64, 16.0, 0.37);
    s.setScalarWall("c", 0, 0.3, 0.0, -1);
    guardRow(s, "C2 Neumann sphere + Dirichlet box (64^3)", 2);
  }
  {  // no solid, Dirichlet box
    IbmSolver s(64, 64, 64);
    walls(s);
    s.setSolid(std::vector<double>((std::size_t)64 * 64 * 64, 100.0), false);
    s.addScalar("c", 0.7, 1, 50, true);
    for (int f = 0; f < 6; ++f)
      s.setScalarBc("c", f, 2, 0.25);
    s.setScalarSource("c", 0.1);
    guardRow(s, "C2 no solid, Dirichlet box (64^3)", 2);
  }
  for (int n : {32, 48, 80, 77}) {  // G5b's geometry: periodic SC array, c = 0.3, singular
    IbmSolver s(n, n, n);
    s.setRho(1.0);
    s.setMu(1.0);
    s.setDt(1.0);
    const double R = 0.5 * n * std::cbrt(0.3 * 6.0 / M_PI);
    s.setSolid(sphereSdf(n, 0.5 * n + 0.21, 0.5 * n - 0.13, 0.5 * n + 0.07, R), false);
    s.addScalar("c", 0.7, 1, 50, true);
    s.setScalarWall("c", 0, 0.3, 0.0, -1);
    s.setScalarSource("c", 0.1);
    // n = 77 (G5b's literal ND = 64) is odd: one level, correctness-only (C1)
    std::snprintf(tag, sizeof tag, "%s G5b geometry n=%d (ND %.1f, singular)",
                  n == 77 ? "C1" : "C2", n, 2 * R);
    guardRow(s, tag, n == 77 ? 1 : 2);
    CHECK(s.scalarField("c").cut->singular);
  }
  for (int n : {64, 128}) {  // C3: periodic box, the only sink one Dirichlet sphere R = n/4
    IbmSolver s(n, n, n);
    s.setRho(1.0);
    s.setMu(1.0);
    s.setDt(1.0);
    s.setSolid(sphereSdf(n, 0.5 * n + 0.37, 0.5 * n - 0.22, 0.5 * n + 0.11, 0.25 * n), false);
    s.addScalar("c", 0.7, 1, 50, true);
    s.setScalarWall("c", 1, 1.0, 0.0, -1);
    s.setScalarSource("c", -0.3);
    std::snprintf(tag, sizeof tag, "C3 periodic + Dirichlet sphere R/h=%d (%d^3)", n / 4, n);
    guardRow(s, tag, 3);
  }
  // WO-5: the same C3 problem ADVECTING — a projected Stokes field through the periodic array
  // (20 steps under a body force), rescaled to a peak cell Peclet number max|u| h / D of 1 and 10,
  // steady: implicit FOU on every face (§6.7), its outflow lumped into the surrogate diagonal and
  // restricted onto the coarse levels with the mass (§5.2). The A1 guard and the iteration count
  // must not degrade against the row above.
  for (const double pe : {1.0, 10.0}) {
    const int n = 64;
    IbmSolver s(n, n, n);
    s.setRho(1.0);
    s.setMu(1.0);
    s.setDt(1.0);
    s.setSolid(sphereSdf(n, 0.5 * n + 0.37, 0.5 * n - 0.22, 0.5 * n + 0.11, 0.25 * n), true);
    s.setBodyForce(1e-3, 4e-4, 2e-4);
    for (int k = 0; k < 20; ++k)
      s.step();
    double umax = 0.0;
    std::vector<double> u[3] = {s.getField("u"), s.getField("v"), s.getField("w")};
    for (const auto& f : u)
      for (double v : f)
        umax = std::fmax(umax, std::fabs(v));
    const double D = 0.7, scale = pe * D / umax;
    const char* nm[3] = {"u", "v", "w"};
    for (int c = 0; c < 3; ++c) {
      for (double& v : u[c])
        v *= scale;
      s.setField(nm[c], u[c]);
    }
    s.addScalar("c", D, 1, 50, true);
    s.setScalarWall("c", 1, 1.0, 0.0, -1);
    s.setScalarSource("c", -0.3);
    std::snprintf(tag, sizeof tag, "C3 + advection, Pe_h=%g, R/h=%d (%d^3)", pe, n / 4, n);
    guardRow(s, tag, 3);
    const auto& st = *s.scalarField("c").cut;
    std::printf("    advecting: %ld implicit faces of %ld carrying flux\n", st.numImplicitFaces,
                st.numFluxFaces);
    CHECK(st.advecting && st.numImplicitFaces == st.numFluxFaces && st.numFluxFaces > 0);
  }
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    testLevels();
    testCoarse();
    testLevelRule();
    testContraction();
  }
  Kokkos::finalize();
  std::printf(failures ? "%d failure(s)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
