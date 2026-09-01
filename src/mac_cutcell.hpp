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

// PECLET_FLOW_EXACT_RESIDUAL=1 — P1 of the defect-correction campaign
// (docs/DEFECT_CORRECTION_PLAN.md). Off by default; byte-identical when off.
//
// The rule: the residual and the Krylov matvec use the EXACT operator in double, in flux form;
// everything below that line is a preconditioner and may stay float. With the gate on, the
// level-0 matvec (matvecOverlap, the choke point for solvePCG and the flexible PCG) applies
// applyCutcellOpExact -- matrix-free from the double face openness Level::ox/oy/oz that
// buildCutcellOp assembles the float bands from -- so the Krylov fixed point becomes A_exact by
// construction and the float hierarchy is demoted to a pure preconditioner, whose errors change
// the convergence RATE and never the fixed point.
//
// Nothing inside vcycle() changes: residualCutcell, the smoother, restriction, the CA ring and
// the AMG bottom all keep the float bands on purpose. This is strictly stronger than the
// double-diagonal ablation (PECLET_FLOW_MG_DIAGRESUM) on the identity both exist to protect:
// the flux form annihilates the constant vector BITWISE, a stored double diagonal only to
// eps_f64. And it costs 0 B/cell where the double diagonal costs +17.
inline bool exactResidual() {
  static const bool v = [] {
    const char* e = std::getenv("PECLET_FLOW_EXACT_RESIDUAL");
    return e && std::atoi(e) != 0;
  }();
  return v;
}

// P2 shares this gate: the star overlay's additive delta (star_elimination.hpp) is part of the
// same level-0 operator, so "the matvec is exact" has to mean the overlay too or the composed
// operator is still the float one.

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

// Fluid fraction (sd >= 0 = fluid) of a triangle with linear vertex values (a, b, c): the exact
// linear-simplex level-set area fraction. Denominators (x-y)(x-z) are strictly positive whenever
// the signs are mixed (x is the odd one out), guarded against exact-zero degeneracies.
KOKKOS_INLINE_FUNCTION double ccTriFrac(double a, double b, double c) {
  const bool pa = a >= 0.0, pb = b >= 0.0, pc = c >= 0.0;
  const int np = (pa ? 1 : 0) + (pb ? 1 : 0) + (pc ? 1 : 0);
  if (np == 3)
    return 1.0;
  if (np == 0)
    return 0.0;
  double x, y, z;
  if (np == 1) {  // rotate the positive vertex into x
    if (pa) { x = a; y = b; z = c; } else if (pb) { x = b; y = c; z = a; } else { x = c; y = a; z = b; }
    const double den = (x - y) * (x - z);
    return den > 1e-300 ? (x * x) / den : 1.0;
  }
  // np == 2: rotate the negative vertex into x
  if (!pa) { x = a; y = b; z = c; } else if (!pb) { x = b; y = c; z = a; } else { x = c; y = a; z = b; }
  const double den = (x - y) * (x - z);
  return 1.0 - (den > 1e-300 ? (x * x) / den : 1.0);
}

// MARCHING-SQUARES face openness (setApertureOrder(2), 2026-08-26): the O(h^2) upgrade of
// ccFaceOpen motivated by the measured convexity bias of the one-sample linear model (the
// tangent-plane estimate over-closes apertures on convex solids by +0.59%/+0.27% in bed
// permeability at R=8/12, decaying ~h^2 -- flow doc/collocated_paper_plan.md row 51). Five
// trilinear samples per face (4 corners + center), triangle-fan decomposition (4 triangles of
// area 1/4 around the center sample -- no marching-squares saddle ambiguity), exact linear
// fraction per triangle. Sub-resolution floor 1e-6 (measured lesson: alpha ~ 1e-12 rows from
// exact geometry destroy the operator conditioning; the crude model's clip was an accidental
// regularizer -- the floor makes the regularization explicit). Floor VALUE 1e-3 (2026-08-26):
// 1e-6 measured insufficient -- alpha in [1e-6, 1e-2] rows drag plain RB-GS (levels=1) to a
// ~1e-4 divergence floor within test budgets (redistribute_mpi_np4); a face open by <0.1% of
// its area carries no resolved flux, so snapping it closed costs nothing measurable.
KOKKOS_INLINE_FUNCTION double ccFaceOpenMS(CCConst sdf, C3 ext, double fx, double fy, double fz,
                                           int type) {
  const double e = 0.5;
  double t1x = 0, t1y = 0, t1z = 0, t2x = 0, t2y = 0, t2z = 0;  // tangent half-offsets
  if (type == 1) {
    t1y = e;
    t2z = e;
  } else if (type == 2) {
    t1x = e;
    t2z = e;
  } else {
    t1x = e;
    t2y = e;
  }
  const double ccg = ccSampleExt(sdf, ext, fx, fy, fz);
  if (ccg <= 0.0)
    return 0.0;  // CENTER GATE (DOF-support consistency, kept from order 1): a face whose
                 // staggered velocity point is solid has a MASKED u DOF -- an alpha > 0 aperture
                 // there is a constraint the projection cannot act on (measured: dropping this
                 // gate leaves an uncorrectable ~1.6e-4 divergence floor under plain RB-GS,
                 // redistribute_mpi_np4). The marching-squares fraction below only refines the
                 // AREA of center-fluid faces -- which is where the convexity bias lived.
  const double c00 = ccSampleExt(sdf, ext, fx - t1x - t2x, fy - t1y - t2y, fz - t1z - t2z);
  const double c10 = ccSampleExt(sdf, ext, fx + t1x - t2x, fy + t1y - t2y, fz + t1z - t2z);
  const double c11 = ccSampleExt(sdf, ext, fx + t1x + t2x, fy + t1y + t2y, fz + t1z + t2z);
  const double c01 = ccSampleExt(sdf, ext, fx - t1x + t2x, fy - t1y + t2y, fz - t1z + t2z);
  const double cc = ccg;
  const double frac = 0.25 * (ccTriFrac(c00, c10, cc) + ccTriFrac(c10, c11, cc) +
                              ccTriFrac(c11, c01, cc) + ccTriFrac(c01, c00, cc));
  if (frac < 1e-3)
    return 0.0;
  if (frac > 1.0 - 1e-12)
    return 1.0;
  return frac;
}

// Fill staggered face openness over the whole extended block (ox[i] = -x face of cell i, etc.).
// order 1 = the shipped one-sample linear model (byte-identical default); order 2 =
// marching-squares (ccFaceOpenMS).
inline void buildOpenness(CCField ox, CCField oy, CCField oz, CCConst sdf, C3 ext, double dx,
                          double dy, double dz, int order = 1) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  if (order >= 2) {
    Kokkos::parallel_for(
        "peclet::flow::cc_open_ms", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long i = static_cast<long>(lx) + static_cast<long>(ly) * ext.x +
                         static_cast<long>(lz) * static_cast<long>(ext.x) * ext.y;
          ox(i) = ccFaceOpenMS(sdf, ext, lx - 0.5, ly, lz, 1);
          oy(i) = ccFaceOpenMS(sdf, ext, lx, ly - 0.5, lz, 2);
          oz(i) = ccFaceOpenMS(sdf, ext, lx, ly, lz - 0.5, 3);
        });
    return;
  }
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
