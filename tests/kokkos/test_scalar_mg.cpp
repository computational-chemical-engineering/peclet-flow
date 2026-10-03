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
//   contraction  — §11 G-iter: the power estimate of rho(I - M^-1 S) at the finest G1 grid
//                  (R/h = 32, box 4R, Dirichlet sphere and box faces) and on the G5b geometry
//                  (periodic simple-cubic array, c = 0.3, insulating, steady: singular) is <= 0.3.
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

// Q2's fallback, measured beside the default for the record (no gate): the same problem with the
// coarse levels built by Galerkin RAP (ScalarMG::setGalerkin, C++ only, default off).
void galerkinReference(IbmSolver& s, const char* tag) {
  auto& st = *s.scalarField("c").cut;
  st.mg->setGalerkin(true);
  Kokkos::deep_copy(s.scalarField("c").c, 0.0);
  s.solveScalarSteady("c");
  char t[128];
  std::snprintf(t, sizeof t, "  [Galerkin RAP reference] %s, %d BiCGStab iterations", tag,
                st.iterations);
  contraction(s, t);
  st.mg->setGalerkin(false);
}

void testContraction() {
  {  // the finest G1 rung: R/h = 32, box 4R = 128
    const int n = 128;
    IbmSolver s(n, n, n);
    g1Setup(s, n, 32.0, 0.37);
    s.solveScalarSteady("c");
    std::printf("G1 R/h=32: %d BiCGStab iterations\n", s.scalarField("c").cut->iterations);
    CHECK(contraction(s, "contraction G1 R/h=32 (128^3)") <= 0.3);
    galerkinReference(s, "G1 R/h=32");
  }
  {  // G5b's geometry: periodic SC array, c = 0.3, insulating, steady (singular)
    for (int n : {80, 77}) {
      IbmSolver s(n, n, n);
      s.setRho(1.0);
      s.setMu(1.0);
      s.setDt(1.0);
      const double R = 0.5 * n * std::cbrt(0.3 * 6.0 / M_PI);
      s.setSolid(sphereSdf(n, 0.5 * n + 0.21, 0.5 * n - 0.13, 0.5 * n + 0.07, R), false);
      s.addScalar("c", 0.7, 1, 50, true);
      s.setScalarWall("c", 0, 0.3, 0.0, -1);
      s.setScalarSource("c", 0.1);
      s.solveScalarSteady("c");
      CHECK(s.scalarField("c").cut->singular);
      char tag[96];
      std::snprintf(tag, sizeof tag, "contraction G5b-geometry N=%d (ND=%.1f, singular)", n, 2 * R);
      std::printf("G5b geometry n=%d: %d BiCGStab iterations\n", n,
                  s.scalarField("c").cut->iterations);
      const double rho = contraction(s, tag);
      if (n == 80)
        galerkinReference(s, "G5b geometry n=80");
      if (n == 80)
        CHECK(rho <= 0.3);
    }
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
