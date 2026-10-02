/// @file
/// @brief ctest `geo_bottom`: the B1 geometric-Krylov bottom (doc/vof_step_performance_design.md
/// §5.7, WO-6) against the per-kernel multigrid code it re-uses.
///
/// The problem: a single-rank 32x24x16 hierarchy of two levels, so the agglomerated bottom is the
/// 16x12x8 level of the bubble column, with three geometric sub-levels below it (8x6x4, 4x3x2,
/// 2x3x2). The face coefficient varies smoothly over a ratio of 50. Two boundary sets: all
/// periodic, and walls on y (which exercises the zero-gradient wall ghost before each
/// prolongation).
///
///   1. M: the team kernel's V-cycle z = M r against the per-kernel V-cycle run over the same
///      levels (a CutcellMG whose levels ARE the bottom plus the sub-levels, with the smoothed
///      bottom: geoBottomReferenceForTest). (a) With the mean removals off on both sides it is
///      bitwise equal -- every cell body, ghost policy and transfer. (b) The full M differs only
///      through the exit fluid-mean reduction (a team reduction against a Kokkos range reduction:
///      a different summation order), bounded here by 4 eps max|z|; strict equality is printed.
///      (The design note's §5.7 unit gate asks for the full M bitwise -- see the WO-6 log entry.)
///   2. The inner FCG converges to tau = 1e-8 (relative, infinity norm, the fluid mean removed)
///      within the cap of 100, raises no flag, and its solution's TRUE residual, recomputed here on
///      the host from the level's operator, is within 10 tau of r0.
/// Runs on every backend: a host backend never selects the engine, so the hooks build the
/// sub-hierarchy themselves there (geoForceSubForTest).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

std::vector<double> toHost(CCField f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}
std::vector<double> toHostOp(FPV f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  std::vector<double> v(h.extent(0));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = (double)h(i);
  return v;
}
CCField toDevice(const std::vector<double>& v, const char* name) {
  CCField f(name, v.size());
  auto h = Kokkos::create_mirror_view(f);
  for (std::size_t i = 0; i < v.size(); ++i)
    h(i) = v[i];
  Kokkos::deep_copy(f, h);
  return f;
}

// A fluid-mean-free random field on the inner cells of a level (zero ghosts).
std::vector<double> randomInner(const CutcellMG::Level& lv, unsigned seed) {
  std::vector<double> v(lv.n, 0.0);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  double sum = 0.0;
  long cnt = 0;
  for (int z = 1; z <= lv.inner.z; ++z)
    for (int y = 1; y <= lv.inner.y; ++y)
      for (int x = 1; x <= lv.inner.x; ++x) {
        const long i = x + (long)y * lv.ext.x + (long)z * lv.ext.x * lv.ext.y;
        v[i] = U(rng);
        sum += v[i];
        ++cnt;
      }
  for (int z = 1; z <= lv.inner.z; ++z)
    for (int y = 1; y <= lv.inner.y; ++y)
      for (int x = 1; x <= lv.inner.x; ++x)
        v[x + (long)y * lv.ext.x + (long)z * lv.ext.x * lv.ext.y] -= sum / (double)cnt;
  return v;
}

int runCase(const char* name, const int bc[6]) {
  const int nx = 32, ny = 24, nz = 16;
  CutcellMG mg;
  mg.setBoundaryConditions(bc);
  mg.init(nx, ny, nz, 2);
  mg.setAgglomerationMode(-1);
  const C3 e{nx + 2, ny + 2, nz + 2};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  std::vector<double> hx(n), hy(n), hz(n);
  const double lr = std::log(50.0), k = 2.0 * M_PI;
  for (int z = 0; z < e.z; ++z)
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x) {
        const std::size_t i = x + (std::size_t)y * e.x + (std::size_t)z * e.x * e.y;
        const double s = 0.5 + 0.5 * std::sin(k * x / nx) * std::cos(k * (y + 0.5 * z) / ny);
        const double t = 0.5 + 0.5 * std::cos(k * (x + z) / nz);
        hx[i] = std::exp(lr * s) / 50.0;  // in [0.02, 1]: ratio 50
        hy[i] = std::exp(lr * t) / 50.0;
        hz[i] = std::exp(lr * (1.0 - s)) / 50.0;
      }
  CCField ox = toDevice(hx, "ox"), oy = toDevice(hy, "oy"), oz = toDevice(hz, "oz");
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
  mg.geoForceSubForTest();
  CutcellMG::Level& bt = mg.level(mg.nLevels() - 1);
  int fails = 0;
  printf("[%s] bottom %dx%dx%d, %d sub-levels, connected=%d, ineligible: %s\n", name, bt.inner.x,
         bt.inner.y, bt.inner.z, mg.geoSubLevels(), mg.geoConnected() ? 1 : 0,
         mg.geoBottomIneligible() ? mg.geoBottomIneligible() : "(eligible)");
  if (mg.geoSubLevels() != 3 || !mg.geoConnected()) {
    printf("[%s] FAIL: expected 3 sub-levels and a connected bottom\n", name);
    return 1;
  }

  // 1. M against the per-kernel V-cycle over the same levels.
  {
    CCField r = toDevice(randomInner(bt, 7), "r");
    auto compare = [&](bool noMean, long& ndiff, double& dmax, double& amax) {
      CCField zt("zt", bt.n), zr("zr", bt.n);
      CutcellMG ref = mg.geoBottomReferenceForTest(noMean);
      ref.precondForTest(zr, r);
      mg.geoBottomPrecondForTest(zt, r, noMean);
      const auto a = toHost(zt), b = toHost(zr);
      ndiff = 0;
      dmax = 0.0;
      amax = 0.0;
      for (int z = 1; z <= bt.inner.z; ++z)
        for (int y = 1; y <= bt.inner.y; ++y)
          for (int x = 1; x <= bt.inner.x; ++x) {
            const long i = x + (long)y * bt.ext.x + (long)z * bt.ext.x * bt.ext.y;
            amax = std::max(amax, std::fabs(b[i]));
            if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
              ++ndiff;
              dmax = std::max(dmax, std::fabs(a[i] - b[i]));
            }
          }
    };
    long nd = 0;
    double dm = 0.0, am = 0.0;
    // (a) every cell-body phase -- smoothing, residual, restriction, ghosts, prolongation --
    //     with the mean removals off on both sides: bitwise.
    compare(true, nd, dm, am);
    printf("[%s] M without its mean removals: %s (max|z| %.3e)\n", name,
           nd == 0 ? "bitwise equal to the per-kernel V-cycle" : "FAIL, differs", am);
    if (nd != 0) {
      printf("[%s]   %ld cells differ, max|dz| %.3e\n", name, nd, dm);
      ++fails;
    }
    // (b) the full M: the exit fluid mean is a TEAM reduction here and a Kokkos range reduction
    //     in the per-kernel code, so its last bits may differ (the subtracted mean differs by
    //     rounding; every cell then moves by at most a few ulp of |z|). Strict equality is
    //     reported; the bound 4 eps max|z| is asserted.
    compare(false, nd, dm, am);
    const bool strict = nd == 0, bounded = dm <= 4.0 * 2.220446049250313e-16 * am;
    printf("[%s] full M: strict bitwise %s (%ld cells differ, max|dz| %.3e = %.2e max|z|) -> %s\n",
           name, strict ? "yes" : "no", nd, dm, am > 0 ? dm / am : 0.0,
           bounded ? "within 4 eps" : "FAIL");
    if (!bounded)
      ++fails;
  }

  // 2. The inner FCG to tau within the cap.
  {
    const auto hb = randomInner(bt, 11);
    Kokkos::deep_copy(bt.rhs, 0.0);
    {
      CCField b = toDevice(hb, "b");
      Kokkos::deep_copy(bt.rhs, b);
    }
    const int it = mg.geoBottomSolveForTest();
    const bool flag = mg.geoFlagForTest();
    const auto x = toHost(bt.x);
    const auto AC = toHostOp(bt.AC), AX = toHostOp(bt.AFX), AY = toHostOp(bt.AFY),
               AZ = toHostOp(bt.AFZ);
    const C3 be = bt.ext, bn = bt.inner;
    auto at = [&](int xx, int yy, int zz) {  // periodic wrap of the inner index
      xx = (xx - 1 + bn.x) % bn.x + 1;
      yy = (yy - 1 + bn.y) % bn.y + 1;
      zz = (zz - 1 + bn.z) % bn.z + 1;
      return x[xx + (long)yy * be.x + (long)zz * be.x * be.y];
    };
    double r0 = 0.0, rn = 0.0;
    for (int z = 1; z <= bn.z; ++z)
      for (int y = 1; y <= bn.y; ++y)
        for (int xx = 1; xx <= bn.x; ++xx) {
          const long i = xx + (long)y * be.x + (long)z * be.x * be.y, sy = be.x,
                     sz = (long)be.x * be.y;
          const double Ax = AC[i] * x[i] + AX[i + 1] * at(xx + 1, y, z) + AX[i] * at(xx - 1, y, z) +
                            AY[i + sy] * at(xx, y + 1, z) + AY[i] * at(xx, y - 1, z) +
                            AZ[i + sz] * at(xx, y, z + 1) + AZ[i] * at(xx, y, z - 1);
          r0 = std::max(r0, std::fabs(hb[i]));
          rn = std::max(rn, std::fabs(hb[i] - Ax));
        }
    const bool ok = it >= 1 && it < kGeoCap && !flag && rn <= 10.0 * kGeoTau * r0;
    printf("[%s] FCG: %d iterations (cap %d), flag %d, true residual %.3e = %.3e r0 -> %s\n", name,
           it, kGeoCap, flag ? 1 : 0, rn, rn / r0, ok ? "ok" : "FAIL");
    if (!ok)
      ++fails;
  }
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    const int periodic[6] = {0, 0, 0, 0, 0, 0};
    const int wallsY[6] = {0, 0, 1, 1, 0, 0};
    fails += runCase("periodic", periodic);
    fails += runCase("walls-y", wallsY);
  }
  Kokkos::finalize();
  printf(fails ? "geo_bottom: FAIL (%d)\n" : "geo_bottom: PASS\n", fails);
  return fails ? 1 : 0;
}
