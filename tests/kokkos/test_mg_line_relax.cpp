/// @file
/// @brief ctest `mg_line_relax`: the zebra line smoother of doc/mg_depth_policy.md Amendment A2.4
/// (gate G-LR (5), A2.5), single rank, host and device backends.
///
/// Level: 32x32x25 (level 0 of a 2-level hierarchy, s = z: the line kernel takes s explicitly; also
/// 32x32x2 and 32x32x3, the shortest columns a triggered level can have),
/// s = z periodic and s = z between walls (x, y periodic). Face openness uniform in [0.05, 1], 5 %
/// of the faces closed, and one SEALED column (every lateral face of the column (5, 7) closed: its
/// column system is the singular 1-D Neumann / periodic Laplacian, which exercises the pivot guard
/// and the Sherman-Morrison denominator guard).
/// Checks, for each colour and both s BCs:
///   FP64: one colour pass of CutcellMG::lineSmoothColor equals x + delta with delta the dense
///         (cyclic) column solve of A_col delta = b - A x on every column of that colour (long
///         double Gaussian elimination, partial pivoting) to 1e-12 relative (max |error| / max
///         |delta| per column); the other colour's cells are bitwise unchanged; every value finite.
///   FP32: the same on the FP32 face weights w_f = fl32(t_f) (lineSmoothColorFp32, in-kernel wrap),
///         the reference built from the FP32 operands (D = fp32Diag, r in the flux form), to 1e-4
///         relative; x = c * 1, b = 0 bitwise unchanged for c in {1, -3.7, 1e5}; all finite.
///   Wall weights: between walls the face coefficient on both wall planes is exactly 0 (FP64 AF_z
///   and FP32 W_z), so the kernel's l_0 = u_{m-1} := 0 drops nothing.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "mg_fp32_problems.hpp"

using namespace peclet::flow;
using namespace peclet_flow_fp32_problems;

namespace {

int fails = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %-78s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  if (!ok)
    ++fails;
}
std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3g", v);
  return b;
}

constexpr int NX = 32, NY = 32;
constexpr int SX = 5, SY = 7;  // the sealed column (0-based inner coordinates)

Problem makeLineProblem(bool walls, int NZ) {
  Problem P;
  P.name = walls ? "z walls" : "z periodic";
  P.nx = NX;
  P.ny = NY;
  P.nz = NZ;
  P.levels = 2;
  if (walls)
    P.bc[4] = P.bc[5] = 1;
  const C3 e = P.ext();
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  P.ox.assign(n, 0.0);
  P.oy.assign(n, 0.0);
  P.oz.assign(n, 0.0);
  std::mt19937 rng(20261026u);
  std::uniform_real_distribution<double> open(0.05, 1.0), u01(0.0, 1.0);
  auto face = [&] { return u01(rng) < 0.05 ? 0.0 : open(rng); };
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        const long i = P.idx(x, y, z);
        P.ox[i] = face();
        P.oy[i] = face();
        P.oz[i] = face();
      }
  for (int z = 0; z < NZ; ++z) {  // seal the column (SX, SY) laterally (x, y periodic)
    P.ox[P.idx(SX, SY, z)] = 0.0;
    P.ox[P.idx((SX + 1) % NX, SY, z)] = 0.0;
    P.oy[P.idx(SX, SY, z)] = 0.0;
    P.oy[P.idx(SX, (SY + 1) % NY, z)] = 0.0;
  }
  return P;
}

// Dense solve of M d = r (m x m, long double, partial pivoting). false if singular.
bool denseSolve(std::vector<long double> M, std::vector<long double> r, int m,
                std::vector<long double>& d) {
  for (int c = 0; c < m; ++c) {
    int p = c;
    for (int k = c + 1; k < m; ++k)
      if (std::fabs(M[k * m + c]) > std::fabs(M[p * m + c]))
        p = k;
    if (M[p * m + c] == 0.0L)
      return false;
    if (p != c) {
      for (int j = 0; j < m; ++j)
        std::swap(M[c * m + j], M[p * m + j]);
      std::swap(r[c], r[p]);
    }
    for (int k = c + 1; k < m; ++k) {
      const long double f = M[k * m + c] / M[c * m + c];
      for (int j = c; j < m; ++j)
        M[k * m + j] -= f * M[c * m + j];
      r[k] -= f * r[c];
    }
  }
  d.assign(m, 0.0L);
  for (int k = m - 1; k >= 0; --k) {
    long double s = r[k];
    for (int j = k + 1; j < m; ++j)
      s -= M[k * m + j] * d[j];
    d[k] = s / M[k * m + k];
  }
  return true;
}

// The host operator of one level: row coefficients of every inner cell, with wrapped reads.
struct HostOp {
  C3 e, n;
  long sy, sz;
  std::vector<long double> AC, AF[3];  // FP64: AC, AF; FP32: D (as fp32Diag), -W
  long idx(int lx, int ly, int lz) const { return (long)lx + (long)ly * sy + (long)lz * sz; }
  bool decoupled(long i, bool fp32) const { return fp32 ? AC[i] == 0.0L : AC[i] < 1e-30L; }
};

// The reference pass: on every column (lx, ly) of `color`, x_k + delta_k with A_col delta = r.
// x, b: host copies (level extent). Returns the max relative error over the columns compared;
// sealed: the sealed column's comparison is skipped (its system is singular) -- finiteness only.
template <class XT>
double comparePass(const HostOp& H, const std::vector<XT>& b, const std::vector<XT>& x0,
                   const std::vector<XT>& x1, bool periodic, bool fp32, int color, bool& otherSame,
                   bool& finite, int& nCols) {
  const C3 e = H.e, n = H.n;
  const int g = 1, m = n.z;
  const long t = H.sz;
  double worst = 0.0;
  otherSame = true;
  finite = true;
  nCols = 0;
  for (std::size_t i = 0; i < x1.size(); ++i)
    finite = finite && std::isfinite((double)x1[i]);
  for (int ly = g; ly < e.y - g; ++ly)
    for (int lx = g; lx < e.x - g; ++lx) {
      if (((lx + ly) & 1) != color) {  // og = {0, 0, 0} single rank; the column colour drops z
        for (int lz = g; lz < e.z - g; ++lz) {
          const long i = H.idx(lx, ly, lz);
          otherSame = otherSame && std::memcmp(&x0[i], &x1[i], sizeof(XT)) == 0;
        }
        continue;
      }
      if (lx - g == SX && ly - g == SY)
        continue;  // the sealed column: singular column system
      std::vector<long double> M((std::size_t)m * m, 0.0L), r(m, 0.0L), d;
      for (int k = 0; k < m; ++k) {
        const int lz = g + k;
        const long i = H.idx(lx, ly, lz);
        if (H.decoupled(i, fp32)) {
          M[k * m + k] = 1.0L;
          continue;
        }
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, H.sy, H.sz);
        long double Ax;
        if (fp32) {  // the flux form: D x_i + sum_f (-w_f) x_j = sum_f w_f (x_i - x_j)
          Ax = H.AC[i] * x0[i] + H.AF[0][i] * x0[w.xm] + H.AF[0][i + 1] * x0[w.xp] +
               H.AF[1][i] * x0[w.ym] + H.AF[1][i + H.sy] * x0[w.yp] + H.AF[2][i] * x0[w.zm] +
               H.AF[2][i + t] * x0[w.zp];
        } else {
          Ax = H.AC[i] * x0[i] + H.AF[0][i + 1] * x0[w.xp] + H.AF[0][i] * x0[w.xm] +
               H.AF[1][i + H.sy] * x0[w.yp] + H.AF[1][i] * x0[w.ym] + H.AF[2][i + t] * x0[w.zp] +
               H.AF[2][i] * x0[w.zm];
        }
        r[k] = (long double)b[i] - Ax;
        M[k * m + k] = H.AC[i];
        const long double lo = H.AF[2][i], up = H.AF[2][i + t];
        if (k > 0)
          M[k * m + k - 1] += lo;
        else if (periodic)
          M[k * m + m - 1] += lo;
        if (k < m - 1)
          M[k * m + k + 1] += up;
        else if (periodic)
          M[k * m + 0] += up;
      }
      if (!denseSolve(M, r, m, d)) {
        worst = 1e300;
        continue;
      }
      long double dmax = 0.0L, emax = 0.0L;
      for (int k = 0; k < m; ++k) {
        const long i = H.idx(lx, ly, g + k);
        const long double got = (long double)x1[i] - (long double)x0[i];
        dmax = std::max(dmax, std::fabs(d[k]));
        emax = std::max(emax, std::fabs(got - d[k]));
      }
      ++nCols;
      if (dmax > 0.0L)
        worst = std::max(worst, (double)(emax / dmax));
    }
  return worst;
}

void runCase(bool walls, int nz) {
  const Problem P = makeLineProblem(walls, nz);
  std::printf("%s (32x32x%d, s = z)\n", P.name.c_str(), nz);
  CutcellMG mg;
  mg.init(P.nx, P.ny, P.nz, P.levels);
  mg.setBoundaryConditions(P.bc);
  mg.setVcyclePrecision(CutcellMG::kVcycleFp64);
  mg.setOpenness(CCConst(toDevice(P.ox, "ox")), CCConst(toDevice(P.oy, "oy")),
                 CCConst(toDevice(P.oz, "oz")), 1.0, 1.0, 1.0);
  CutcellMG::Level& lv = mg.level(0);
  const C3 e = lv.ext;
  const std::size_t n = lv.n;
  const bool periodic = !walls;
  check(mg.linePeriodic(2) == periodic, "linePeriodic(z) follows the domain BC");
  if (lv.rhs.extent(0) != n)
    lv.rhs = CCField("rhs", n);
  lv.lnc = CCField("c", n);
  lv.lny = CCField("y", n);
  lv.lnz = CCField("z", n);

  HostOp H;
  H.e = e;
  H.n = lv.inner;
  H.sy = e.x;
  H.sz = (long)e.x * e.y;
  {
    const auto ac = toHost(lv.AC);
    const auto fx = toHost(lv.AFX), fy = toHost(lv.AFY), fz = toHost(lv.AFZ);
    H.AC.assign(ac.begin(), ac.end());
    H.AF[0].assign(fx.begin(), fx.end());
    H.AF[1].assign(fy.begin(), fy.end());
    H.AF[2].assign(fz.begin(), fz.end());
    if (walls) {
      bool zero = true;
      for (int ly = 1; ly < e.y - 1; ++ly)
        for (int lx = 1; lx < e.x - 1; ++lx)
          zero = zero && fz[H.idx(lx, ly, 1)] == 0 && fz[H.idx(lx, ly, e.z - 1)] == 0;
      check(zero, "FP64 wall faces: AF_z = 0 on both wall planes");
    }
  }

  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  // ---- FP64
  for (int color = 0; color < 2; ++color) {
    std::vector<double> hx(n, 0.0), hb(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      hx[i] = U(rng);
      hb[i] = U(rng);
    }
    Kokkos::deep_copy(lv.x, Kokkos::View<double*, Kokkos::HostSpace>(hx.data(), n));
    Kokkos::deep_copy(lv.rhs, Kokkos::View<double*, Kokkos::HostSpace>(hb.data(), n));
    mg.lineSmoothColor(lv, 2, color);
    Kokkos::fence();
    const auto x1 = toHost(lv.x);
    // the ghosts were refilled by the pass's own fill; compare from the filled pre-pass x
    std::vector<double> x0 = hx;
    for (int lz = 0; lz < e.z; ++lz)
      for (int ly = 0; ly < e.y; ++ly)
        for (int lx = 0; lx < e.x; ++lx) {
          const bool inner =
              lx >= 1 && lx < e.x - 1 && ly >= 1 && ly < e.y - 1 && lz >= 1 && lz < e.z - 1;
          if (!inner)
            x0[H.idx(lx, ly, lz)] = x1[H.idx(lx, ly, lz)];
        }
    bool same = false, finite = false;
    int cols = 0;
    const double err = comparePass(H, hb, x0, x1, periodic, false, color, same, finite, cols);
    check(err <= 1e-12, "FP64 colour " + std::to_string(color) + ": " + std::to_string(cols) +
                            " columns = dense solve, max rel err " + num(err) + " <= 1e-12");
    check(same, "FP64 colour " + std::to_string(color) + ": the other colour bitwise unchanged");
    check(finite, "FP64 colour " + std::to_string(color) + ": all finite (sealed column included)");
  }

  // ---- FP32: w_f = fl32(t_f) = fl32(-AF_f) (AF = -t_f in FP64)
  {
    std::vector<VReal> W[3];
    for (int a = 0; a < 3; ++a) {
      W[a].resize(n);
      for (std::size_t i = 0; i < n; ++i)
        W[a][i] = static_cast<VReal>(-(double)H.AF[a][i]);
    }
    auto toDevF = [&](const std::vector<VReal>& v) {
      VField f("w", v.size());
      auto h = Kokkos::create_mirror_view(f);
      for (std::size_t i = 0; i < v.size(); ++i)
        h(i) = v[i];
      Kokkos::deep_copy(f, h);
      return f;
    };
    lv.WX = toDevF(W[0]);
    lv.WY = toDevF(W[1]);
    lv.WZ = toDevF(W[2]);
    lv.xf = VField("xf", n);
    lv.rhsf = VField("rhsf", n);
    lv.lncf = VField("cf", n);
    lv.lnyf = VField("yf", n);
    lv.lnzf = VField("zf", n);
    if (walls) {
      bool zero = true;
      for (int ly = 1; ly < e.y - 1; ++ly)
        for (int lx = 1; lx < e.x - 1; ++lx)
          zero = zero && W[2][H.idx(lx, ly, 1)] == 0.0f && W[2][H.idx(lx, ly, e.z - 1)] == 0.0f;
      check(zero, "FP32 wall weights: W_z = 0 on both wall planes");
    }
    HostOp H32 = H;
    for (int lz = 0; lz < e.z; ++lz)
      for (int ly = 0; ly < e.y; ++ly)
        for (int lx = 0; lx < e.x; ++lx) {
          const long i = H.idx(lx, ly, lz);
          const bool inner =
              lx >= 1 && lx < e.x - 1 && ly >= 1 && ly < e.y - 1 && lz >= 1 && lz < e.z - 1;
          // fp32Diag's order, in FP32
          const VReal D =
              inner ? ((((W[0][i + 1] + W[0][i]) + W[1][i + H.sy]) + W[1][i]) + W[2][i + H.sz]) +
                          W[2][i]
                    : 0.0f;
          H32.AC[i] = (long double)D;
          for (int a = 0; a < 3; ++a)
            H32.AF[a][i] = -(long double)W[a][i];
        }
    for (int color = 0; color < 2; ++color) {
      std::vector<VReal> hx(n, 0.0f), hb(n, 0.0f);
      for (std::size_t i = 0; i < n; ++i) {
        hx[i] = static_cast<VReal>(U(rng));
        hb[i] = static_cast<VReal>(U(rng));
      }
      Kokkos::deep_copy(lv.xf, Kokkos::View<VReal*, Kokkos::HostSpace>(hx.data(), n));
      Kokkos::deep_copy(lv.rhsf, Kokkos::View<VReal*, Kokkos::HostSpace>(hb.data(), n));
      mg.lineSmoothColorFp32(lv, 2, color);
      Kokkos::fence();
      const auto x1 = toHost(lv.xf);
      bool same = false, finite = false;
      int cols = 0;
      const double err = comparePass(H32, hb, hx, x1, periodic, true, color, same, finite, cols);
      check(err <= 1e-4, "FP32 colour " + std::to_string(color) + ": " + std::to_string(cols) +
                             " columns = dense solve, max rel err " + num(err) + " <= 1e-4");
      check(same, "FP32 colour " + std::to_string(color) + ": the other colour bitwise unchanged");
      check(finite,
            "FP32 colour " + std::to_string(color) + ": all finite (sealed column included)");
    }
    for (const VReal c : {1.0f, -3.7f, 1e5f}) {
      std::vector<VReal> hx(n, c), hb(n, 0.0f);
      Kokkos::deep_copy(lv.xf, Kokkos::View<VReal*, Kokkos::HostSpace>(hx.data(), n));
      Kokkos::deep_copy(lv.rhsf, Kokkos::View<VReal*, Kokkos::HostSpace>(hb.data(), n));
      mg.lineSmoothColorFp32(lv, 2, 0);
      mg.lineSmoothColorFp32(lv, 2, 1);
      Kokkos::fence();
      const auto x1 = toHost(lv.xf);
      const bool same = std::memcmp(x1.data(), hx.data(), n * sizeof(VReal)) == 0;
      check(same, "FP32 x = " + num(c) + " * 1, b = 0: both colours bitwise unchanged");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    for (int nz : {25, 2, 3}) {  // the A2.5 level, and the shortest columns (m = 2, 3)
      runCase(false, nz);
      runCase(true, nz);
    }
  }
  Kokkos::finalize();
  std::printf("mg_line_relax: %s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails,
              fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
