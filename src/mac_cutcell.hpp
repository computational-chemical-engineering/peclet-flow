/// @file
/// @brief flow — portable (Kokkos) cut-cell pressure-operator face openness from an SDF.
///
/// Kokkos port of mac_cutcell.cuh: the gradient-normalised masked fluid fraction (cc_fraction_core)
/// + trilinear SDF sampling, producing the staggered face openness ox/oy/oz (ox[i] = openness of
/// the -x face of cell i) over the extended (inner+ghost) block. Faithful copy of the fraction
/// math. KOKKOS_INLINE_FUNCTION so it is shared with the host reference. Runs on any Kokkos
/// backend.
#ifndef PECLET_FLOW_MAC_CUTCELL_HPP
#define PECLET_FLOW_MAC_CUTCELL_HPP

#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <type_traits>
#include <utility>

namespace peclet::flow {

using CCExec = Kokkos::DefaultExecutionSpace;
using CCMem = CCExec::memory_space;
using CCField = Kokkos::View<double*, CCMem>;
using CCConst = Kokkos::View<const double*, CCMem>;

struct C3 {
  int x, y, z;
};

// Masked fluid fraction of a face from its SDF samples (centre + 6 axis neighbours). type 1/2/3 =
// x/y/z face. (sd<=0 => closed.) Verbatim from cc_fraction_core.
KOKKOS_INLINE_FUNCTION double ccFractionCore(double sd, double sxp, double sxm, double syp,
                                             double sym, double szp, double szm, int type,
                                             double dx, double dy, double dz) {
  if (sd <= 0.0)
    return 0.0;
  const double gx = (sxp - sxm) / (2.0 * dx), gy = (syp - sym) / (2.0 * dy),
               gz = (szp - szm) / (2.0 * dz);
  double gmag = Kokkos::sqrt(gx * gx + gy * gy + gz * gz);
  if (gmag < 1e-6)
    gmag = 1e-6;
  const double nx = gx / gmag, ny = gy / gmag, nz = gz / gmag;
  double denom = (type == 1)   ? (Kokkos::fabs(ny) * dy + Kokkos::fabs(nz) * dz)
                 : (type == 2) ? (Kokkos::fabs(nx) * dx + Kokkos::fabs(nz) * dz)
                               : (Kokkos::fabs(nx) * dx + Kokkos::fabs(ny) * dy);
  if (denom < 1e-9)
    denom = 1e-9;
  double frac = 0.5 + sd / denom;
  if (frac < 0.0)
    frac = 0.0;
  if (frac > 1.0)
    frac = 1.0;
  return frac;
}

KOKKOS_INLINE_FUNCTION double ccSampleExt(CCConst sdf, C3 ext, double x, double y, double z) {
  const double fx = Kokkos::floor(x), fy = Kokkos::floor(y), fz = Kokkos::floor(z);
  const double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  const int x1 = cl(x0 + 1, ext.x), y1 = cl(y0 + 1, ext.y), z1 = cl(z0 + 1, ext.z);
  x0 = cl(x0, ext.x);
  y0 = cl(y0, ext.y);
  z0 = cl(z0, ext.z);
  const long sy = ext.x, sz = static_cast<long>(ext.x) * ext.y;
  auto F = [&](int xx, int yy, int zz) {
    return sdf(static_cast<long>(xx) + static_cast<long>(yy) * sy + static_cast<long>(zz) * sz);
  };
  const double c00 = F(x0, y0, z0) * (1 - wx) + F(x1, y0, z0) * wx;
  const double c10 = F(x0, y1, z0) * (1 - wx) + F(x1, y1, z0) * wx;
  const double c01 = F(x0, y0, z1) * (1 - wx) + F(x1, y0, z1) * wx;
  const double c11 = F(x0, y1, z1) * (1 - wx) + F(x1, y1, z1) * wx;
  const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

KOKKOS_INLINE_FUNCTION double ccFaceOpen(CCConst sdf, C3 ext, double fx, double fy, double fz,
                                         int type, double dx, double dy, double dz) {
  const double sd = ccSampleExt(sdf, ext, fx, fy, fz);
  if (sd <= 0.0)
    return 0.0;
  const double e = 1.0;
  return ccFractionCore(
      sd, ccSampleExt(sdf, ext, fx + e, fy, fz), ccSampleExt(sdf, ext, fx - e, fy, fz),
      ccSampleExt(sdf, ext, fx, fy + e, fz), ccSampleExt(sdf, ext, fx, fy - e, fz),
      ccSampleExt(sdf, ext, fx, fy, fz + e), ccSampleExt(sdf, ext, fx, fy, fz - e), type, dx, dy,
      dz);
}

// Fill staggered face openness over the whole extended block (ox[i] = -x face of cell i, etc.).
inline void buildOpenness(CCField ox, CCField oy, CCField oz, CCConst sdf, C3 ext, double dx,
                          double dy, double dz) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cc_open", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = static_cast<long>(lx) + static_cast<long>(ly) * ext.x +
                       static_cast<long>(lz) * static_cast<long>(ext.x) * ext.y;
        ox(i) = ccFaceOpen(sdf, ext, lx - 0.5, ly, lz, 1, dx, dy, dz);
        oy(i) = ccFaceOpen(sdf, ext, lx, ly - 0.5, lz, 2, dx, dy, dz);
        oz(i) = ccFaceOpen(sdf, ext, lx, ly, lz - 0.5, 3, dx, dy, dz);
      });
}

// HOST-only serial cutoff: below this many cells an OpenMP launch costs more than the work it does
// (fork/join is ~20-30 us at 24 threads), so run the loop sequentially instead. That is
// BIT-IDENTICAL for elementwise kernels and for colored sweeps (same-color cells are independent,
// so the order within a sweep is irrelevant); reductions are deliberately NOT cut over — that would
// change the FP summation order with the block size, and the multi-rank bit-exactness contract is
// worth more than the microseconds. The device path never sees this (hostRunSerial is false there).
//
// MEASURED (5965WX, 24 threads, per-level V-cycle timer): the win lives almost entirely in the MG's
// BOTTOM level, which fires ~24 trivial launches per V-cycle — 64^3/rank L4(4^3): 0.018 -> 0.007 s
// per 50 V-cycles, ~8% of the whole V-cycle; 128^3/rank ~1.5%; 256^3/rank (the fat-rank target
// size) ~0.2%, i.e. below run-to-run noise. A LARGER cutoff back-fires (131072 serializes levels
// with real work: 64^3 projection 16.6 -> 20.6 ms/step), so keep it small.
inline long hostSerialCellCutoff() {
  static const long n = [] {
    const char* e = std::getenv("PECLET_FLOW_HOST_SERIAL_CELLS");
    return e ? std::atol(e) : 8192L;
  }();
  return n;
}
// True when a host launch of `cells` cells should run sequentially instead (always false on device).
inline bool hostRunSerial(long cells) {
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>)
    return cells > 0 && cells < hostSerialCellCutoff();
  else
    return false;
}

// Launch a 3-D elementwise kernel with a HOST-TUNED MDRange tiling: one full-x row per tile (the
// Kokkos host default tiles are tiny, wrecking streaming locality — measured 5.6x on the RB-GS
// smoother when its loop went x-contiguous). The lambda is passed through UNCHANGED, and the
// device keeps the default MDRange tiling untouched — byte-identical results on both backends
// (elementwise kernels are order-independent).
template <class F>
inline void ccFor3(const char* name, C3 lo, C3 hi, F f) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    if (hostRunSerial((long)(hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z))) {
      for (int lz = lo.z; lz < hi.z; ++lz)  // too small to be worth a fork/join (bit-identical)
        for (int ly = lo.y; ly < hi.y; ++ly)
          for (int lx = lo.x; lx < hi.x; ++lx)
            f(lx, ly, lz);
      return;
    }
    Kokkos::parallel_for(
        name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}, {hi.x - lo.x, 2, 2}), f);
  } else {
    Kokkos::parallel_for(name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}), f);
  }
}

// Reduction sibling of ccFor3 (same host tiling rationale). NOTE: the host tiling changes the
// FP accumulation order of sum-reductions — numerically legitimate (parallel reductions are
// unordered by contract) but not bit-identical to the previous host rounding.
template <class F, class R>
inline void ccReduce3(const char* name, C3 lo, C3 hi, F f, R&& reducer) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    Kokkos::parallel_reduce(
        name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}, {hi.x - lo.x, 2, 2}), f,
        std::forward<R>(reducer));
  } else {
    Kokkos::parallel_reduce(name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}), f,
                            std::forward<R>(reducer));
  }
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CUTCELL_HPP
