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
//                  gate (tests/python/test_scalar_cutcell_gates.py);
//   advective    — design Amendment A2 (WO-5c), the steady advective surrogate, rows u1-u6: (u1)
//                  uniform flow: Qp_a = phi_a / cf_a exactly, Qm_a = 0, (2,2,1) levels included;
//                  (u2) a divergence-free random field: advective row sums <= 1e-13 max Q, column
//                  sums 0 to round-off, every level; (u3) the M-matrix on every level; (u4) the
//                  level-0 advective applySurrogate == the operator's band matvec with SAC,
//                  bitwise; (u5) Q = 0 on the domain-face planes of every non-periodic axis, open
//                  faces included; (u6) C4: the C3 problem advecting at census Pe_h 0.1 / 1 / 10,
//                  contraction < 1 and <= 0.90 (prov.). Prints the G-perf V-cycle time ratio.
#include <chrono>
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
  // dt D / h^2 = F -> kappa_A = 1 + 12 F; F just below / above the switch (D-WO9-3: 25 -> F = 2)
  const double Fsw = (ScalarMG::kFullTableKappa - 1.0) / 12.0;
  s.setDt(0.95 * Fsw / 0.7);  // kappa_A = 23.8 < 25
  s.advanceScalars();
  const int small = s.scalarField("c").cut->mgLevels;
  s.setDt(1.05 * Fsw / 0.7);  // kappa_A = 26.2
  s.advanceScalars();
  const int at = s.scalarField("c").cut->mgLevels;
  s.solveScalarSteady("c");
  const int steady = s.scalarField("c").cut->mgLevels;
  std::printf("level rule: dt D/h^2 = %.3f -> %d level(s), %.3f -> %d, steady -> %d\n", 0.95 * Fsw,
              small, 1.05 * Fsw, at, steady);
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
}

// ---- the advective surrogate (design Amendment A2, §6.7; WO-5c): rows u1-u6
// ----------------------

// Inner linear index of level `lv`.
long inner(const ScalarMG::Level& lv, int x, int y, int z) {
  return (long)(x + lv.g) + (long)(y + lv.g) * lv.ext.x + (long)(z + lv.g) * lv.ext.x * lv.ext.y;
}

// A periodic box without a solid (the cut-cell projection's all-open openness), cell units, a
// cut-cell scalar D = 0.7 with a source; the caller sets the face velocities, then solves steady.
void openBox(IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(1.0);
  s.setSolid(std::vector<double>((std::size_t)s.nx() * s.ny() * s.nz(), 100.0), true);
  s.addScalar("c", 0.7, 1, 50, true);
  s.setScalarSource("c", 0.1);
}

// (u1) uniform flow along each axis, periodic box without a solid, on a grid whose table has
// (2,2,2) and (2,2,1) levels: Qp_a = phi_a / cf_a exactly and Qm_a = 0 on every coarse level (the
// velocity 0.75 makes every sub-face sum exact).
void testAdvUniform() {
  const int nn[2][3] = {{32, 32, 32}, {32, 32, 4}};
  for (const auto& n : nn)
    for (int ax = 0; ax < 3; ++ax) {
      IbmSolver s(n[0], n[1], n[2]);
      openBox(s);
      const std::size_t nc = (std::size_t)n[0] * n[1] * n[2];
      const char* nm[3] = {"u", "v", "w"};
      for (int a = 0; a < 3; ++a)
        s.setField(nm[a], std::vector<double>(nc, a == ax ? 0.75 : 0.0));
      s.solveScalarSteady("c");
      const auto& mg = *s.scalarField("c").cut->mg;
      long bad = 0, cells = 0;
      bool aniso = false;
      for (int L = 1; L < mg.levels(); ++L) {
        const auto& lv = mg.level(L);
        aniso = aniso || (lv.cfac.x != lv.cfac.z);
        const int cf[3] = {lv.cfac.x, lv.cfac.y, lv.cfac.z};
        for (int a = 0; a < 3; ++a) {
          const HV qp = host(lv.qp[a]), qm = host(lv.qm[a]);
          const double want = a == ax ? 0.75 / (double)cf[a] : 0.0;
          for (int z = 0; z < lv.inner.z; ++z)
            for (int y = 0; y < lv.inner.y; ++y)
              for (int x = 0; x < lv.inner.x; ++x) {
                const long i = inner(lv, x, y, z);
                bad += (qp(i) == want && qm(i) == 0.0) ? 0 : 1;
                ++cells;
              }
        }
      }
      std::printf(
          "(u1) uniform flow along %c, %dx%dx%d (%d levels%s): %ld of %ld Qp/Qm entries "
          "differ from phi/cf_a, 0\n",
          "xyz"[ax], n[0], n[1], n[2], mg.levels(), aniso ? ", (2,2,1) levels" : "", bad, cells);
      CHECK(bad == 0 && cells > 0);
      if (n[2] == 4)
        CHECK(aniso);
    }
}

// (u2) a divergence-free random field (node stream-function differences, periodic): on every
// level |sum out - sum in| <= 1e-13 max Q per row (level 0 from phi), and the column sums of the
// band operator vanish to round-off (no solid, steady, no open face: the diffusion columns cancel
// too, so the whole column sum is the advective one).
void testAdvDivergenceFree() {
  const int n = 32;
  IbmSolver s(n, n, n);
  openBox(s);
  std::vector<double> psi[3];
  unsigned long long st = 0x2545F4914F6CDD1Dull;
  auto rnd = [&]() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return (double)(st >> 11) * (1.0 / 9007199254740992.0) - 0.5;
  };
  for (auto& p : psi) {
    p.resize((std::size_t)n * n * n);
    for (double& v : p)
      v = rnd();
  }
  auto P = [&](int c, int x, int y, int z) {
    return psi[c][(std::size_t)((x + n) % n) + (std::size_t)((y + n) % n) * n +
                  (std::size_t)((z + n) % n) * n * n];
  };
  std::vector<double> u((std::size_t)n * n * n), v(u.size()), w(u.size());
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const std::size_t k = (std::size_t)x + (std::size_t)y * n + (std::size_t)z * n * n;
        u[k] = (P(2, x, y + 1, z) - P(2, x, y, z)) - (P(1, x, y, z + 1) - P(1, x, y, z));
        v[k] = (P(0, x, y, z + 1) - P(0, x, y, z)) - (P(2, x + 1, y, z) - P(2, x, y, z));
        w[k] = (P(1, x + 1, y, z) - P(1, x, y, z)) - (P(0, x, y + 1, z) - P(0, x, y, z));
      }
  s.setField("u", u);
  s.setField("v", v);
  s.setField("w", w);
  s.solveScalarSteady("c");
  const auto& sc = s.scalarField("c");
  const auto& mg = *sc.cut->mg;
  CHECK(sc.cut->advecting && sc.cut->steady);
  bool ok = true;
  for (int L = 0; L < mg.levels(); ++L) {
    const auto& lv = mg.level(L);
    const long sx = 1, sy = lv.ext.x, sz = (long)lv.ext.x * lv.ext.y;
    HV q[6];
    if (L == 0) {  // level 0: the positive parts of phi (no solid: every face is a coupling)
      for (int a = 0; a < 3; ++a) {
        const HV ph = host(sc.cut->phi[a]);
        q[a] = HV("qp", ph.extent(0));
        q[3 + a] = HV("qm", ph.extent(0));
        for (std::size_t k = 0; k < ph.extent(0); ++k) {
          q[a](k) = std::fmax(ph(k), 0.0);
          q[3 + a](k) = std::fmax(-ph(k), 0.0);
        }
      }
    } else {
      for (int a = 0; a < 3; ++a) {
        q[a] = host(lv.qp[a]);
        q[3 + a] = host(lv.qm[a]);
      }
    }
    const HV AC = host(lv.AC), AW = host(lv.AW), AE = host(lv.AE), AS = host(lv.AS),
             AN = host(lv.AN), AB = host(lv.AB), AT = host(lv.AT);
    double maxQ = 0.0, maxAC = 0.0, row = 0.0, col = 0.0;
    for (int z = 0; z < lv.inner.z; ++z)
      for (int y = 0; y < lv.inner.y; ++y)
        for (int x = 0; x < lv.inner.x; ++x) {
          const long i = inner(lv, x, y, z);
          for (int k = 0; k < 6; ++k)
            maxQ = std::fmax(maxQ, q[k](i));
          maxAC = std::fmax(maxAC, std::fabs(AC(i)));
          const double out =
              ((q[3](i) + q[0](i + sx)) + (q[4](i) + q[1](i + sy))) + (q[5](i) + q[2](i + sz));
          const double in =
              ((q[0](i) + q[3](i + sx)) + (q[1](i) + q[4](i + sy))) + (q[2](i) + q[5](i + sz));
          row = std::fmax(row, std::fabs(out - in));
          // column i: the diagonal plus each neighbour's coefficient on x_i (periodic: the
          // neighbours of an inner cell are inner or wrapped ghosts; use the inner wrap)
          auto I = [&](int dx, int dy, int dz) {
            return inner(lv, (x + dx + lv.inner.x) % lv.inner.x, (y + dy + lv.inner.y) % lv.inner.y,
                         (z + dz + lv.inner.z) % lv.inner.z);
          };
          const double cs = AC(i) + AW(I(1, 0, 0)) + AE(I(-1, 0, 0)) + AS(I(0, 1, 0)) +
                            AN(I(0, -1, 0)) + AB(I(0, 0, 1)) + AT(I(0, 0, -1));
          col = std::fmax(col, std::fabs(cs));
        }
    std::printf("(u2) level %d: max|out - in| = %.2e max Q, max|column sum| = %.2e max AC\n", L,
                row / maxQ, col / maxAC);
    ok = ok && row <= 1e-13 * maxQ && col <= 1e-13 * maxAC && maxQ > 0.0;
  }
  CHECK(ok);
}

// The C3 problem (periodic box 4R, a Dirichlet sphere R = n/4 with a source) with the projected
// Stokes field of that geometry (20 steps under a body force: WO-5's harness), cell units, the
// pressure at its default tolerance. (u3) checks the COLUMN sums, which are zero by construction
// whatever the field's divergence (review finding 3); the field's divergence moves only the row
// sums, which is why the former row check needed the pressure PCG at 1e-14.
void c3Stokes(IbmSolver& s, int n) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(1.0);
  s.setSolid(sphereSdf(n, 0.5 * n + 0.37, 0.5 * n - 0.22, 0.5 * n + 0.11, 0.25 * n), true);
  s.setBodyForce(1e-3, 4e-4, 2e-4);
  for (int k = 0; k < 20; ++k)
    s.step();
  s.addScalar("c", 0.7, 1, 50, true);
  s.setScalarWall("c", 1, 1.0, 0.0, -1);
  s.setScalarSource("c", -0.3);
}

// Rescale the face velocities so that the census max_cell_peclet of the next steady solve is `pe`
// (G-adv's parametrization): one probe solve at the present field, then the field times
// pe / measured.
void rescalePeclet(IbmSolver& s, double pe) {
  s.solveScalarSteady("c");
  const double k = pe / s.scalarField("c").cut->maxCellPeclet;
  for (const char* nm : {"u", "v", "w"}) {
    std::vector<double> f = s.getField(nm);
    for (double& v : f)
      v *= k;
    s.setField(nm, f);
  }
}

// Seconds per V-cycle (z = M^-1 r) on the current build, r a fixed pseudo-random vector on the
// unknowns (G-perf: the advective cycle against the symmetric one on the same block).
double vcycleSeconds(IbmSolver& s, int reps) {
  auto& st = *s.scalarField("c").cut;
  const std::size_t n = st.SAC.extent(0);
  CCField r("r", n), z("z", n);
  const std::vector<double> unk = s.scalarGeometryField(4);
  std::vector<double> rv(unk.size());
  for (std::size_t k = 0; k < rv.size(); ++k)
    rv[k] = unk[k] > 0.5 ? std::sin(0.37 * (double)k) : 0.0;
  s.setField("c", rv);
  Kokkos::deep_copy(r, s.scalarField("c").c);
  st.mg->apply(z, r);  // warm-up
  Kokkos::fence();
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < reps; ++k)
    st.mg->apply(z, r);
  Kokkos::fence();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// (u3) A2's M-matrix property on every level, by COLUMNS (review finding 3, ruling D-WOR-3): every
// off-diagonal band <= 0, and every column sum AC(j) + sum_k band_k(j's neighbour toward j) >=
// -1e-13 max AC. The advective couplings enter as one number per face (Q on the upwind diagonal,
// -Q as the downwind off-diagonal), so the advection and diffusion parts have zero column sums up to
// round-off INDEPENDENTLY of the field's divergence; the mass, wall, Dirichlet and omega_open terms
// only add to the diagonal. A column-diagonally dominant Z-matrix is an M-matrix (by columns), and
// the left null vector of a singular closure problem is exactly 1 on every level. The C3 box is
// periodic on every axis: the neighbours wrap over the inner cells. Returns the defect count.
long mMatrixDefects(const ScalarMG& mg, const char* tag) {
  long bad = 0;
  for (int L = 0; L < mg.levels(); ++L) {
    const auto& lv = mg.level(L);
    const HV AC = host(lv.AC), AW = host(lv.AW), AE = host(lv.AE), AS = host(lv.AS),
             AN = host(lv.AN), AB = host(lv.AB), AT = host(lv.AT);
    double maxAC = 0.0;
    for (int z = 0; z < lv.inner.z; ++z)
      for (int y = 0; y < lv.inner.y; ++y)
        for (int x = 0; x < lv.inner.x; ++x)
          maxAC = std::fmax(maxAC, std::fabs(AC(inner(lv, x, y, z))));
    double minCol = 1e300;
    long rows = 0, positive = 0, negative = 0;
    for (int z = 0; z < lv.inner.z; ++z)
      for (int y = 0; y < lv.inner.y; ++y)
        for (int x = 0; x < lv.inner.x; ++x) {
          const long i = inner(lv, x, y, z);
          auto I = [&](int dx, int dy, int dz) {
            return inner(lv, (x + dx + lv.inner.x) % lv.inner.x, (y + dy + lv.inner.y) % lv.inner.y,
                         (z + dz + lv.inner.z) % lv.inner.z);
          };
          for (const HV* b : {&AW, &AE, &AS, &AN, &AB, &AT})
            positive += (*b)(i) > 0.0 ? 1 : 0;
          const double cs = AC(i) + AW(I(1, 0, 0)) + AE(I(-1, 0, 0)) + AS(I(0, 1, 0)) +
                            AN(I(0, -1, 0)) + AB(I(0, 0, 1)) + AT(I(0, 0, -1));
          minCol = std::fmin(minCol, cs / maxAC);
          negative += cs < -1e-13 * maxAC ? 1 : 0;
          ++rows;
        }
    std::printf(
        "(u3) %s level %d: %ld columns, min column sum = %.3e max AC, %ld below -1e-13, %ld positive "
        "off-diagonals\n",
        tag, L, rows, minCol, negative, positive);
    bad += positive + negative;
  }
  return bad;
}

// (u6) the C4 contraction rows (WO-5's "C3 + advection" rows, restated by A2): the C3 problem at
// R/h 16 advecting at census Pe_h 0.1, 1, 10 — C1 (< 1) and <= 0.90 (prov.). On the Pe_h 10 row
// also (u3) the M-matrix on every level and (u4) the level-0 advective applySurrogate == the
// operator's band matvec with SAC as the diagonal, bitwise. G-perf: the advective V-cycle's time
// against the symmetric one (the same problem at rest) on the same block.
void testAdvective() {
  const int n = 64;
  double tRest = 0.0;
  {
    IbmSolver s(n, n, n);
    c3Stokes(s, n);
    for (const char* nm : {"u", "v", "w"})
      s.setField(nm, std::vector<double>((std::size_t)n * n * n, 0.0));
    s.solveScalarSteady("c");
    CHECK(!s.scalarField("c").cut->advecting);
    tRest = vcycleSeconds(s, 20);
  }
  for (const double pe : {0.1, 1.0, 10.0}) {
    IbmSolver s(n, n, n);
    c3Stokes(s, n);
    rescalePeclet(s, pe);
    const auto& st = *s.scalarField("c").cut;
    char tag[96];
    std::snprintf(tag, sizeof tag, "C4 C3 + advection, Pe_h=%.3g, R/h=%d (%d^3)", pe, n / 4, n);
    s.solveScalarSteady("c");
    std::printf("    census max_cell_peclet %.6g; %ld implicit faces of %ld carrying flux\n",
                st.maxCellPeclet, st.numImplicitFaces, st.numFluxFaces);
    CHECK(std::fabs(st.maxCellPeclet / pe - 1.0) <= 1e-12);
    CHECK(st.advecting && st.numImplicitFaces == st.numFluxFaces && st.numFluxFaces > 0);
    char t[160];
    std::snprintf(t, sizeof t, "%-44s %2d it", tag, st.iterations);
    const double rho = contraction(s, t);
    CHECK(rho < 1.0);
    CHECK(rho <= 0.90);
    if (pe == 10.0) {
      auto& sc = s.scalarField("c");
      CHECK(mMatrixDefects(*st.mg, "Pe_h 10") == 0);
      // (u4)
      const std::size_t nn = st.SAC.extent(0);
      CCField x("x", nn), y1("y1", nn), y2("y2", nn);
      {
        auto hx = Kokkos::create_mirror_view(x);
        for (std::size_t k = 0; k < nn; ++k)
          hx(k) = std::cos(1.3 * (double)k) * (1.0 + 0.1 * std::sin(0.01 * (double)k));
        Kokkos::deep_copy(x, hx);
      }
      st.mg->applySurrogate(y1, x);  // fills x's ghosts (the solver's exchange)
      peclet::flow::applyCutcellOp(y2, peclet::flow::CCConst(x), st.SAC, sc.AW, sc.AE, sc.AS, sc.AN,
                                   sc.AB, sc.AT, st.mg->level(0).ext, IbmSolver::G);
      const HV h1 = host(y1), h2 = host(y2);
      const auto& l0 = st.mg->level(0);
      long diff = 0;
      for (int z = 0; z < l0.inner.z; ++z)
        for (int yy = 0; yy < l0.inner.y; ++yy)
          for (int xx = 0; xx < l0.inner.x; ++xx) {
            const long i = inner(l0, xx, yy, z);
            diff += std::memcmp(&h1(i), &h2(i), sizeof(double)) == 0 ? 0 : 1;
          }
      std::printf(
          "(u4) level-0 advective applySurrogate vs the band matvec with SAC: %ld of %d "
          "rows not bitwise\n",
          diff, l0.inner.x * l0.inner.y * l0.inner.z);
      CHECK(diff == 0);
      const double tAdv = vcycleSeconds(s, 20);
      std::printf(
          "G-perf: V-cycle %d^3 (%d levels): advective %.2f ms, symmetric (at rest) "
          "%.2f ms, ratio %.2f (expected ~1.5, red flag > 2)\n",
          n, st.mgLevels, 1e3 * tAdv, 1e3 * tRest, tAdv / tRest);
    }
  }
}

// (u5) Q = 0 on both domain-face planes of every non-periodic axis on every coarse level, open
// faces included: a channel along x (inflow -x, outflow +x), no-slip y and z walls, a sphere;
// the flow developed by step() (which captures the open-face flux), then a steady solve.
void testAdvOpenFaces() {
  const int nx = 64, ny = 32, nz = 32;
  IbmSolver s(nx, ny, nz);
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(2.0);
  s.setDomainBc(0, 2, 0.05, 0.0, 0.0);  // -x inflow
  s.setDomainBc(1, 3, 0.0, 0.0, 0.0);   // +x outflow
  for (int f = 2; f < 6; ++f)
    s.setDomainBc(f, 1, 0.0, 0.0, 0.0);
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const double dx = x + 0.5 - 27.3, dy = y + 0.5 - 15.6, dz = z + 0.5 - 16.9;
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - 7.4;
      }
  s.setSolid(sdf, true);
  for (int k = 0; k < 10; ++k)
    s.step();
  s.addScalar("c", 0.4, 0, 50, true);
  s.setScalarBc("c", 0, 2, 1.0);  // dirichlet inflow
  for (int f = 1; f < 6; ++f)
    s.setScalarBc("c", f, 1, 0.0);  // neumann
  s.solveScalarSteady("c");
  const auto& st = *s.scalarField("c").cut;
  const auto& mg = *st.mg;
  CHECK(st.advecting && st.converged);
  long bad = 0, checked = 0;
  for (int L = 1; L < mg.levels(); ++L) {
    const auto& lv = mg.level(L);
    const int ext[3] = {lv.ext.x, lv.ext.y, lv.ext.z};
    const long sa[3] = {1, (long)lv.ext.x, (long)lv.ext.x * lv.ext.y};
    for (int a = 0; a < 3; ++a) {
      const HV qp = host(lv.qp[a]), qm = host(lv.qm[a]);
      const int b = (a + 1) % 3, c = (a + 2) % 3;
      for (const int pl : {lv.g, ext[a] - lv.g})  // the low and the high domain-face plane
        for (int jb = lv.g; jb < ext[b] - lv.g; ++jb)
          for (int jc = lv.g; jc < ext[c] - lv.g; ++jc) {
            const long i = (long)pl * sa[a] + (long)jb * sa[b] + (long)jc * sa[c];
            bad += (qp(i) == 0.0 && qm(i) == 0.0) ? 0 : 1;
            ++checked;
          }
    }
  }
  std::printf(
      "(u5) channel with open x faces, walls y z: %d levels, %ld domain-face entries, %ld "
      "nonzero; steady %d iterations, census max_cell_peclet %.3g\n",
      mg.levels(), checked, bad, st.iterations, st.maxCellPeclet);
  CHECK(bad == 0 && checked > 0);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const bool advOnly = argc > 1 && std::strcmp(argv[argc - 1], "advective") == 0;
    if (!advOnly) {  // (`test_scalar_mg advective` runs the A2 rows alone)
      testLevels();
      testCoarse();
      testLevelRule();
      testContraction();
    }
    testAdvUniform();
    testAdvDivergenceFree();
    testAdvective();
    testAdvOpenFaces();
  }
  Kokkos::finalize();
  std::printf(failures ? "%d failure(s)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
