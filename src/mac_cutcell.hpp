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

#include "policy.hpp"

namespace peclet::flow {

using CCExec = Kokkos::DefaultExecutionSpace;
using CCMem = CCExec::memory_space;
using CCField = Kokkos::View<double*, CCMem>;
using CCConst = Kokkos::View<const double*, CCMem>;

// The EXACT level-0 operator apply — P1 of the defect-correction campaign
// (docs/DEFECT_CORRECTION_PLAN.md). PER SOLVER (`set_pressure_exact_residual`): off at
// construction, byte-identical when off, and turned ON by `enable_vof`.
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
// the AMG bottom all keep the float bands on purpose.
//
// `IbmSolver::enableVof` turns it ON (measured: on Hysing case 2 it removes 7.5 orders of the
// projected flux divergence, 1.85e-03 -> 5.15e-11, and moves the iteration count, the step count,
// the dt-limit census and both published functionals by nothing at all — the float `A*1 != 0`
// defect is exactly what a moving interface's coefficient contrast amplifies).
// `set_pressure_exact_residual(False)` after `enable_vof` is the ablation.

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
    if (pa) {
      x = a;
      y = b;
      z = c;
    } else if (pb) {
      x = b;
      y = c;
      z = a;
    } else {
      x = c;
      y = a;
      z = b;
    }
    const double den = (x - y) * (x - z);
    return den > 1e-300 ? (x * x) / den : 1.0;
  }
  // np == 2: rotate the negative vertex into x
  if (!pa) {
    x = a;
    y = b;
    z = c;
  } else if (!pb) {
    x = b;
    y = c;
    z = a;
  } else {
    x = c;
    y = a;
    z = b;
  }
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
  using MD = MDRange3<CCExec>;
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

// Re-derive ONE staggered face-openness plane: component `a` at the HIGH domain-face index
// (ext.a - g) along axis a. SCALING_ISSUES #3: that index is a GHOST index, so the distributed
// openness halo exchange overwrites it with the periodic wrap of the opposite boundary -- which at
// a non-periodic domain face is another boundary's aperture, not this one's. It is the face the
// divergence weights the outgoing flux by AND (since the Dirichlet row carries the aperture) the
// one the operator reads, so it has to be the geometry's own value. `buildOpenness` had it right
// before the exchange; this re-derives it from the same (already exchanged, already
// domain-extended) SDF rather than keeping a save buffer alive across a per-step set_solid.
inline void buildOpennessHighFace(CCField oa, CCConst sdf, C3 ext, int g, int a, double dx,
                                  double dy, double dz, int order = 1) {
  CCExec space;
  const int dims[3] = {ext.x, ext.y, ext.z};
  const long st[3] = {1, (long)ext.x, (long)ext.x * ext.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int bf = dims[a] - g;
  const int type = a + 1;
  using MD = MDRange2<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_open_high_face", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        double q[3];
        q[a] = (double)bf - 0.5;
        q[b] = (double)p0;
        q[c] = (double)p1;
        const long i = (long)p0 * sb + (long)p1 * sc + (long)bf * sa;
        oa(i) = (order >= 2) ? ccFaceOpenMS(sdf, ext, q[0], q[1], q[2], type)
                             : ccFaceOpen(sdf, ext, q[0], q[1], q[2], type, dx, dy, dz);
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
inline constexpr long kHostSerialCellCutoff = 8192L;
// True when a host launch of `cells` cells should run sequentially instead (always false on
// device).
inline bool hostRunSerial(long cells) {
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>)
    return cells > 0 && cells < kHostSerialCellCutoff;
  else
    return false;
}

// The host launch rule (doc/vof_step_performance_design.md §4.5, "rule H"): a 3-D elementwise
// kernel is ONE cell body `f(x, y, z)`, launched
//   * on a device backend as MDRange3 with the default tiling (unchanged), and
//   * on a host backend as a RangePolicy over the (y, z) ROWS of the box, with the x loop inside
//     (a "pencil"), serially below kHostSerialCellCutoff cells.
// The host MDRange it replaces (even with the {nx,2,2} row tiles this helper used to pass) cost
// 2.4x a flat x loop in the MDRange tile machinery. The rows are statically partitioned over the
// threads, identically for every kernel of the same box, so a thread keeps touching the same rows.
// Elementwise kernels are order-independent, so every launch form is byte-identical.
//
// ccFor3 marks the x loop `omp simd`. CONTRACT (§4.5 item 4): an iteration writes only its own
// cell(s) and reads no cell that another iteration of the same row writes (red-black colour passes
// qualify: same-colour cells are independent). A body that breaks it -- a ghost copy whose source
// and destination rows overlap -- uses ccForRows3, the same launch without the simd mark. With
// -ffp-contract=off (the host flags, A1) vectorizing a non-reduction loop is bit-identical to the
// scalar code.
#if defined(_OPENMP)
#define PECLET_FLOW_OMP_SIMD _Pragma("omp simd")
#else
#define PECLET_FLOW_OMP_SIMD
#endif
namespace ccdetail {
template <bool Simd, class F>
inline void ccRows3(const char* name, C3 lo, C3 hi, F f) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int ny = hi.y - lo.y, nz = hi.z - lo.z;
    if (ny <= 0 || nz <= 0 || hi.x <= lo.x)
      return;
    const long rows = (long)ny * nz;
    auto row = [=](long r) {
      const int y = lo.y + (int)(r % ny), z = lo.z + (int)(r / ny);
      if constexpr (Simd) {
        PECLET_FLOW_OMP_SIMD
        for (int x = lo.x; x < hi.x; ++x)
          f(x, y, z);
      } else {
        for (int x = lo.x; x < hi.x; ++x)
          f(x, y, z);
      }
    };
    if (hostRunSerial(rows * (hi.x - lo.x))) {
      for (long r = 0; r < rows; ++r)  // too small to be worth a fork/join (bit-identical)
        row(r);
      return;
    }
    Kokkos::parallel_for(name, Kokkos::RangePolicy<CCExec>(space, 0, rows), row);
  } else {
    Kokkos::parallel_for(name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}), f);
  }
}
}  // namespace ccdetail

template <class F>
inline void ccFor3(const char* name, C3 lo, C3 hi, F f) {
  ccdetail::ccRows3<true>(name, lo, hi, f);
}
// ccFor3 without the `omp simd` mark on the host x loop (see the contract above).
template <class F>
inline void ccForRows3(const char* name, C3 lo, C3 hi, F f) {
  ccdetail::ccRows3<false>(name, lo, hi, f);
}

// Reduction sibling of ccFor3. On a host backend it is the PENCIL reduction of B2
// (doc/vof_step_performance_design.md §5.8, a recorded order): a RangePolicy over the (y, z) rows
// of the box -- the same static row partition as ccFor3 -- each thread reducing its rows in (y, z)
// order with x ascending inside a row, scalar (no `omp simd` reduction), and the per-thread
// partials combined by Kokkos in thread order. Sum results therefore depend on the thread count
// only, as before; max/min reductions are order-free and unchanged. Unlike ccFor3 a reduction is
// never cut over to a serial loop below kHostSerialCellCutoff (that would make the summation
// order a function of the block size). The device branch is the MDRange reduction, unchanged.
namespace ccdetail {
template <class R, bool = Kokkos::is_reducer_v<std::decay_t<R>>>
struct ReduceValue {
  using type = typename std::decay_t<R>::value_type;
};
template <class R>
struct ReduceValue<R, false> {  // a scalar result, or a View holding it
  using D = std::decay_t<R>;
  template <class T, bool = Kokkos::is_view_v<T>>
  struct Pick {
    using type = T;
  };
  template <class T>
  struct Pick<T, true> {
    using type = typename T::non_const_value_type;
  };
  using type = typename Pick<D>::type;
};
}  // namespace ccdetail
template <class F, class... R>
inline void ccReduce3(const char* name, C3 lo, C3 hi, F f, R&&... reducers) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int ny = hi.y - lo.y > 0 ? hi.y - lo.y : 0, nz = hi.z - lo.z > 0 ? hi.z - lo.z : 0;
    const long rows = hi.x > lo.x ? (long)ny * nz : 0;
    Kokkos::parallel_reduce(
        name, Kokkos::RangePolicy<CCExec>(space, 0, rows),
        [=](long r, typename ccdetail::ReduceValue<R>::type&... acc) {
          const int y = lo.y + (int)(r % ny), z = lo.z + (int)(r / ny);
          for (int x = lo.x; x < hi.x; ++x)
            f(x, y, z, acc...);
        },
        std::forward<R>(reducers)...);
  } else {
    Kokkos::parallel_reduce(name, MD(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}), f,
                            std::forward<R>(reducers)...);
  }
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CUTCELL_HPP
