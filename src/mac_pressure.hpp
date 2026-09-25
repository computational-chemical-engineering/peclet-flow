/// @file
/// @brief flow — portable (Kokkos) cut-cell pressure operator + Chorin projection.
///
/// Kokkos port of the variable-coefficient pressure machinery (mg_build_op_k / mg_smooth_var_k from
/// mac_multigrid.cuh, diverg_open_k / correct_k from distributed_ns.cuh): the open-face-weighted
/// Poisson operator A = -div(open grad) built from the staggered face openness, its red-black
/// Gauss-Seidel smoother (solid cells AC~0 decoupled), the open-weighted flux divergence, and the
/// staggered gradient correction. gf = 1/h^2 per axis (1 in grid units). Runs on any Kokkos
/// backend.
#ifndef PECLET_FLOW_MAC_PRESSURE_HPP
#define PECLET_FLOW_MAC_PRESSURE_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "mac_cutcell.hpp"
#include "policy.hpp"

namespace peclet::flow {

// --- A0: the per-cell bodies (doc/vof_step_performance_design.md §4.5 rule H, §5.2) -------------
// Every launch form of these kernels -- the device MDRange, the host pencil, and the single-team
// kernels to come -- calls the SAME inline body, so the per-cell arithmetic cannot drift between
// them. Each expression is a verbatim copy of the kernel it came from, in the same order: a
// rewritten expression can contract into a different FMA chain on the device (design note R1).
// The neighbour INDICES of the field being smoothed or applied are arguments (xp = i + sx,
// xm = i - sx, ...), so a caller can hand a periodically wrapped index instead of a ghost one; the
// coefficients are always read at their own (unwrapped) indices.

// A = -div(open grad): AC = sum of the 6 face terms (openness*gf), off-diagonal across each face =
// -term. ox[i] is the -x face openness of cell i (== +x face of cell i-1). (mg_build_op_k port.)
template <class OpV, class OV>
KOKKOS_INLINE_FUNCTION void cutcellBuildOpCell(const OpV& AC, const OpV& AW, const OpV& AE,
                                               const OpV& AS, const OpV& AN, const OpV& AB,
                                               const OpV& AT, const OV& ox, const OV& oy,
                                               const OV& oz, long i, long sx, long sy, long sz,
                                               double gfx, double gfy, double gfz) {
  const double tw = ox(i) * gfx, te = ox(i + sx) * gfx;
  const double ts = oy(i) * gfy, tn = oy(i + sy) * gfy;
  const double tb = oz(i) * gfz, tt = oz(i + sz) * gfz;
  AW(i) = -tw;
  AE(i) = -te;
  AS(i) = -ts;
  AN(i) = -tn;
  AB(i) = -tb;
  AT(i) = -tt;
  AC(i) = te + tw + tn + ts + tt + tb;
}
// One red/black Gauss-Seidel update of cell i: phi = (b - offdiag)/AC; AC~0 (fully solid) cells are
// decoupled and keep phi.
template <class PV, class BV, class OpV>
KOKKOS_INLINE_FUNCTION void cutcellSmoothCell(const PV& phi, const BV& b, const OpV& AC,
                                              const OpV& AW, const OpV& AE, const OpV& AS,
                                              const OpV& AN, const OpV& AB, const OpV& AT, long i,
                                              long xp, long xm, long yp, long ym, long zp,
                                              long zm) {
  const double ac = AC(i);
  if (ac < 1e-30)
    return;  // fully closed (solid) cell: decoupled, phi stays 0
  const double s = AE(i) * phi(xp) + AW(i) * phi(xm) + AN(i) * phi(yp) + AS(i) * phi(ym) +
                   AT(i) * phi(zp) + AB(i) * phi(zm);
  phi(i) = (b(i) - s) / ac;
}
// (A x)_i from the bands (applyCutcellOp).
template <class XV, class OpV>
KOKKOS_INLINE_FUNCTION double cutcellApplyCell(const XV& x, const OpV& AC, const OpV& AW,
                                               const OpV& AE, const OpV& AS, const OpV& AN,
                                               const OpV& AB, const OpV& AT, long i, long xp,
                                               long xm, long yp, long ym, long zp, long zm) {
  return AC(i) * x(i) + AE(i) * x(xp) + AW(i) * x(xm) + AN(i) * x(yp) + AS(i) * x(ym) +
         AT(i) * x(zp) + AB(i) * x(zm);
}
// (A x)_i in the exact flux form from the double face openness (applyCutcellOpExact).
template <class XV, class OV>
KOKKOS_INLINE_FUNCTION double cutcellApplyExactCell(const XV& x, const OV& ox, const OV& oy,
                                                    const OV& oz, long i, long sx, long sy, long sz,
                                                    long xp, long xm, long yp, long ym, long zp,
                                                    long zm, double gfx, double gfy, double gfz) {
  const double xi = x(i);
  return ox(i) * gfx * (xi - x(xm)) + ox(i + sx) * gfx * (xi - x(xp)) + oy(i) * gfy * (xi - x(ym)) +
         oy(i + sy) * gfy * (xi - x(yp)) + oz(i) * gfz * (xi - x(zm)) +
         oz(i + sz) * gfz * (xi - x(zp));
}

template <class OpV>
inline void buildCutcellOp(OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB, OpV AT, CCConst ox,
                           CCConst oy, CCConst oz, C3 e, int g, double gfx, double gfy,
                           double gfz) {
  ccFor3(
      "peclet::flow::cc_build_op", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellBuildOpCell(AC, AW, AE, AS, AN, AB, AT, ox, oy, oz, i, sx, sy, sz, gfx, gfy, gfz);
      });
}

// --- A2: the FACE form of the pressure operator (doc/vof_step_performance_design.md §4.2, §5.3) --
// The pressure hierarchy stores the diagonal AC and ONE band coefficient per face instead of seven
// bands: AFX(i) is the off-diagonal coefficient of cell i's LOW x face, -open_x(i) * gfx (MReal) --
// exactly the old AW(i) -- and the same face is cell i-1's high face, so
//   AW(i) = AFX(i)   AE(i) = AFX(i+sx)   AS(i) = AFY(i)   AN(i) = AFY(i+sy)
//   AB(i) = AFZ(i)   AT(i) = AFZ(i+sz).
// Bit-identical to the bands: AFX holds the same MReal-rounded value AW held, and every consumer
// reads it into the identical band expression -- the order AE, AW, AN, AS, AT, AB, each term a
// (coefficient) * (neighbour) product. Storing the band SIGN matters on the device: the first form
// of this change stored +open*gf and read (-TX(i+1))*phi + ..., and nvcc then contracted a
// different product into each FMA than for the band expression -- the bubble column drifted by
// 8.8e-15 (u, relative) in 50 steps; host builds do not contract and were bitwise either way
// (doc/vof_step_performance_design.md §4.2 note, WO-3). AC is built by the identical expression.
// 32 instead of 56 bytes of operator per cell (double), and three arrays per level instead of six.
// The seven-band kernels above stay for the scalar-transport smoother and the standalone kernel
// tests.
//
// AFX/AFY/AFZ are written over the FULL extent [0, ext) -- every index a band consumer ever reads,
// including the i+sx of the last ring cell of a communication-avoiding (g=2) level -- from the
// ghost-filled, boundary-re-imposed openness; AC over the build box [gb, ext-gb) as before.
template <class OpV, class OV>
KOKKOS_INLINE_FUNCTION void cutcellBuildFaceOpCell(const OpV& AC, const OpV& AFX, const OpV& AFY,
                                                   const OpV& AFZ, const OV& ox, const OV& oy,
                                                   const OV& oz, long i, long sx, long sy, long sz,
                                                   double gfx, double gfy, double gfz, bool diag) {
  const double tw = ox(i) * gfx;
  const double ts = oy(i) * gfy;
  const double tb = oz(i) * gfz;
  AFX(i) = -tw;
  AFY(i) = -ts;
  AFZ(i) = -tb;
  if (diag) {
    const double te = ox(i + sx) * gfx;
    const double tn = oy(i + sy) * gfy;
    const double tt = oz(i + sz) * gfz;
    AC(i) = te + tw + tn + ts + tt + tb;
  }
}
template <class PV, class BV, class OpV>
KOKKOS_INLINE_FUNCTION void cutcellSmoothFaceCell(const PV& phi, const BV& b, const OpV& AC,
                                                  const OpV& AFX, const OpV& AFY, const OpV& AFZ,
                                                  long i, long sx, long sy, long sz, long xp,
                                                  long xm, long yp, long ym, long zp, long zm) {
  const double ac = AC(i);
  if (ac < 1e-30)
    return;  // fully closed (solid) cell: decoupled, phi stays 0
  const double s = AFX(i + sx) * phi(xp) + AFX(i) * phi(xm) + AFY(i + sy) * phi(yp) +
                   AFY(i) * phi(ym) + AFZ(i + sz) * phi(zp) + AFZ(i) * phi(zm);
  phi(i) = (b(i) - s) / ac;
}
template <class XV, class OpV>
KOKKOS_INLINE_FUNCTION double cutcellApplyFaceCell(const XV& x, const OpV& AC, const OpV& AFX,
                                                   const OpV& AFY, const OpV& AFZ, long i, long sx,
                                                   long sy, long sz, long xp, long xm, long yp,
                                                   long ym, long zp, long zm) {
  return AC(i) * x(i) + AFX(i + sx) * x(xp) + AFX(i) * x(xm) + AFY(i + sy) * x(yp) +
         AFY(i) * x(ym) + AFZ(i + sz) * x(zp) + AFZ(i) * x(zm);
}
template <class RV, class XV, class BV, class OpV>
KOKKOS_INLINE_FUNCTION void cutcellResidualFaceCell(const RV& r, const XV& x, const BV& b,
                                                    const OpV& AC, const OpV& AFX, const OpV& AFY,
                                                    const OpV& AFZ, long i, long sx, long sy,
                                                    long sz, long xp, long xm, long yp, long ym,
                                                    long zp, long zm) {
  const double Ax = (double)AC(i) * x(i) + (double)AFX(i + sx) * x(xp) + (double)AFX(i) * x(xm) +
                    (double)AFY(i + sy) * x(yp) + (double)AFY(i) * x(ym) +
                    (double)AFZ(i + sz) * x(zp) + (double)AFZ(i) * x(zm);
  r(i) = b(i) - Ax;
}

// Face-form build: AFX/AFY/AFZ over [0, ext), AC over the box [gb, ext-gb) (gb = the level's build
// width, as buildCutcellOp's g).
template <class OpV>
inline void buildCutcellOpFace(OpV AC, OpV AFX, OpV AFY, OpV AFZ, CCConst ox, CCConst oy,
                               CCConst oz, C3 e, int gb, double gfx, double gfy, double gfz) {
  ccFor3(
      "peclet::flow::cc_build_op", C3{0, 0, 0}, C3{e.x, e.y, e.z},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const bool diag =
            lx >= gb && lx < e.x - gb && ly >= gb && ly < e.y - gb && lz >= gb && lz < e.z - gb;
        cutcellBuildFaceOpCell(AC, AFX, AFY, AFZ, ox, oy, oz, i, sx, sy, sz, gfx, gfy, gfz, diag);
      });
}
// Face-form siblings of cutcellSmoothColor / cutcellSmoothColorBox (same launch forms).
template <class OpV>
inline void cutcellSmoothColorFace(CCField phi, CCConst b, OpV AC, OpV AFX, OpV AFY, OpV AFZ, C3 e,
                                   C3 og, int g, int color) {
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    const long cells = (long)nyi * nzi * (e.x - 2 * g);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      PECLET_FLOW_OMP_SIMD  // same-colour cells are independent (ccFor3's contract)
          for (int lx = g + ((P ^ (g & 1)) & 1); lx < e.x - g; lx += 2) {
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                              i - sy, i + sz, i - sz);
      }
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                              i - sy, i + sz, i - sz);
      });
}
template <class OpV>
inline void cutcellSmoothColorBoxFace(CCField phi, CCConst b, OpV AC, OpV AFX, OpV AFY, OpV AFZ,
                                      C3 e, C3 og, int color, C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = rhi.y - rlo.y, nzi = rhi.z - rlo.z;
    const long cells = (long)nyi * nzi * (rhi.x - rlo.x);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = rlo.y + (int)(t % nyi), lz = rlo.z + (int)(t / nyi);
      const bool yzSkip = ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z;
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      for (int lx = rlo.x + ((P ^ (rlo.x & 1)) & 1); lx < rhi.x; lx += 2) {
        if (yzSkip && lx >= slo.x && lx < shi.x)
          continue;  // inside the skip box (already swept by the interior pass)
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                              i - sy, i + sz, i - sz);
      }
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth_box",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth_box", MD(space, {rlo.x, rlo.y, rlo.z}, {rhi.x, rhi.y, rhi.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;  // inside the skip box (already swept by the interior pass)
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                              i - sy, i + sz, i - sz);
      });
}
// Face-form siblings of applyCutcellOp / applyCutcellOpBox.
template <class OpV>
inline void applyCutcellOpFace(CCField y, CCConst x, OpV AC, OpV AFX, OpV AFY, OpV AFZ, C3 e,
                               int g) {
  ccFor3(
      "peclet::flow::cc_apply", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyFaceCell(x, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                    i - sy, i + sz, i - sz);
      });
}
template <class OpV>
inline void applyCutcellOpBoxFace(CCField y, CCConst x, OpV AC, OpV AFX, OpV AFY, OpV AFZ, C3 e,
                                  C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  ccFor3(
      "peclet::flow::cc_apply_box", C3{rlo.x, rlo.y, rlo.z}, C3{rhi.x, rhi.y, rhi.z},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyFaceCell(x, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                    i - sy, i + sz, i - sz);
      });
}

// Open-weighted flux divergence d_i = sum_f signed(o_f * face-velocity), consistent with A
// (diverg_open_k).
inline void divergOpen(CCConst u, CCConst v, CCConst w, CCConst ox, CCConst oy, CCConst oz,
                       CCField d, C3 e, int g) {
  ccFor3(
      "peclet::flow::diverg_open", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        d(i) = (ox(i + sx) * u(i + sx) - ox(i) * u(i)) + (oy(i + sy) * v(i + sy) - oy(i) * v(i)) +
               (oz(i + sz) * w(i + sz) - oz(i) * w(i));
      });
}

// One red/black sweep of the variable operator: phi=(b - offdiag)/AC; AC~0 (fully solid) cells
// decoupled. b carries the negated divergence so the system is A phi = -div(u*) (matches the
// validated const-coeff sign).
template <class OpV>
inline void cutcellSmoothColor(CCField phi, CCConst b, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN,
                               OpV AB, OpV AT, C3 e, C3 og, int g, int color) {
  CCExec space;
  // Host backends: line-sweep form (one (y,z) pencil per task, stride-2 x-loop at the colour's
  // parity) — bit-identical (same-colour cells are independent); device keeps MDRange untouched.
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    const long cells = (long)nyi * nzi * (e.x - 2 * g);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      PECLET_FLOW_OMP_SIMD  // same-colour cells are independent (ccFor3's contract)
          for (int lx = g + ((P ^ (g & 1)) & 1); lx < e.x - g; lx += 2) {
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothCell(phi, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                          i + sz, i - sz);
      }
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothCell(phi, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                          i + sz, i - sz);
      });
}

// Box-restricted red/black sweep for the distributed overlap smoother: sweeps [rlo,rhi) only,
// skipping the box [slo,shi) (pass slo==shi to skip nothing). Used to split one color's sweep into
// an INTERIOR pass (cells whose 7-point stencil reads no ghost cell — runs while the halo exchange
// is in flight) and a boundary-SHELL pass (after the exchange lands). A color's cells never read
// same-color cells, so interior-then-shell ordering is bit-identical to the full blocking sweep
// (and a cell swept twice recomputes the identical value — overlapping shell slabs are safe).
template <class OpV>
inline void cutcellSmoothColorBox(CCField phi, CCConst b, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN,
                                  OpV AB, OpV AT, C3 e, C3 og, int color, C3 rlo, C3 rhi, C3 slo,
                                  C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    // Host line-sweep form of the box sweep (see cutcellSmoothColor); the (ly,lz)-level skip test
    // hoists out of the x-loop, the x-range skip stays per cell.
    const int nyi = rhi.y - rlo.y, nzi = rhi.z - rlo.z;
    const long cells = (long)nyi * nzi * (rhi.x - rlo.x);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = rlo.y + (int)(t % nyi), lz = rlo.z + (int)(t / nyi);
      const bool yzSkip = ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z;
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      for (int lx = rlo.x + ((P ^ (rlo.x & 1)) & 1); lx < rhi.x; lx += 2) {
        if (yzSkip && lx >= slo.x && lx < shi.x)
          continue;  // inside the skip box (already swept by the interior pass)
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothCell(phi, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                          i + sz, i - sz);
      }
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth_box",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth_box", MD(space, {rlo.x, rlo.y, rlo.z}, {rhi.x, rhi.y, rhi.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;  // inside the skip box (already swept by the interior pass)
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellSmoothCell(phi, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                          i + sz, i - sz);
      });
}

// y = A x for the cut-cell operator over inner cells (matvec for PCG; mg_apply_var_k port).
template <class OpV>
inline void applyCutcellOp(CCField y, CCConst x, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB,
                           OpV AT, C3 e, int g) {
  ccFor3(
      "peclet::flow::cc_apply", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyCell(x, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                                i + sz, i - sz);
      });
}

// Box-restricted matvec for the distributed overlap: rows in [rlo,rhi) skipping [slo,shi). Reads
// x, writes y (no aliasing), so the interior rows can apply while x's ghost exchange is in flight
// and the shell rows apply after it lands — trivially bit-identical to the blocking order.
template <class OpV>
inline void applyCutcellOpBox(CCField y, CCConst x, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB,
                              OpV AT, C3 e, C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  ccFor3(
      "peclet::flow::cc_apply_box", C3{rlo.x, rlo.y, rlo.z}, C3{rhi.x, rhi.y, rhi.z},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyCell(x, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                                i + sz, i - sz);
      });
}

// --- Exact (matrix-free, double) level-0 apply — the defect-correction residual/matvec ---------
//
// P1 of the suite-wide defect-correction campaign (docs/DEFECT_CORRECTION_PLAN.md). The band form
// above reads the STORED coefficients, which are float by default (MReal), so the operator the
// Krylov method actually inverts is the fp32-rounded one and its singular row-sum identity
// A*1 = 0 holds only to eps_f32 (WO-M measured the consequence: a resolution-independent residual
// floor at ~5e-9 followed by a 1e4-1e5 rebound). These two kernels apply the SAME operator
// matrix-free from the double face openness that buildCutcellOp assembles the bands from
// (Level::ox/oy/oz, CCField = View<double*>, ghost-filled and boundary-re-imposed by
// setOpenness) — so they need no new storage, and they read 24 B/cell of coefficient instead of
// 28.
//
// Written in FLUX (difference) form, y_i = sum_f t_f (x_i - x_nbr) with t_f = open_f * gf:
//   * the diagonal is never formed, so it can never disagree with the faces;
//   * a constant vector is annihilated BITWISE whatever the precision of t_f, because every
//     difference vanishes identically. A stored double diagonal only gets A*1 = 0 to eps_f64.
//   * a fully-closed (solid) cell has every t_f = 0, so y_i = 0 exactly, as with the bands
//     (AC = 0 there). Where the openness is tiny-but-nonzero the two DISAGREE, and that is the
//     point: the band AC is the float-rounded sum of the six faces, this is the exact one.
// Sign convention matches buildCutcellOp exactly: AC = sum(t_f), A<face> = -t_f, hence
// AC*x_i + sum(-t_f*x_nbr) == sum_f t_f (x_i - x_nbr).
//
// The hierarchy below level 0 is untouched — it is the preconditioner and may stay float.
inline void applyCutcellOpExact(CCField y, CCConst x, CCConst ox, CCConst oy, CCConst oz, C3 e,
                                int g, double gfx, double gfy, double gfz) {
  ccFor3(
      "peclet::flow::cc_apply_exact", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyExactCell(x, ox, oy, oz, i, sx, sy, sz, i + sx, i - sx, i + sy, i - sy,
                                     i + sz, i - sz, gfx, gfy, gfz);
      });
}

// Box-restricted sibling of applyCutcellOpExact, for the distributed overlap matvec: rows in
// [rlo,rhi) skipping [slo,shi). Same skip-box signature as applyCutcellOpBox, and the same
// argument for bit-identity with the blocking form — it reads x and the openness, writes y, no
// aliasing — so interior-then-shell ordering reproduces the blocking apply exactly.
inline void applyCutcellOpExactBox(CCField y, CCConst x, CCConst ox, CCConst oy, CCConst oz, C3 e,
                                   C3 rlo, C3 rhi, C3 slo, C3 shi, double gfx, double gfy,
                                   double gfz) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  ccFor3(
      "peclet::flow::cc_apply_exact_box", C3{rlo.x, rlo.y, rlo.z}, C3{rhi.x, rhi.y, rhi.z},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = cutcellApplyExactCell(x, ox, oy, oz, i, sx, sy, sz, i + sx, i - sx, i + sy, i - sy,
                                     i + sz, i - sz, gfx, gfy, gfz);
      });
}

// PHASE 2 (anisotropic cells), doc/anisotropic_metric.md §1.3/§3: every correction kernel below
// takes the per-axis pressure weight `w_a = 1/h_a'^2` and applies it OUTSIDE the whole existing
// expression (`u -= w_x * (today's term)`) — the same weight `CutcellMG::setOpenness` puts on the
// operator row of that axis, which is what makes the corrected open flux telescope. On the
// isotropic path `w == 1.0` exactly and `1.0 * x == x` in IEEE-754, so these are bit-identical;
// the defaulted arguments keep the kernel-level tests calling the historical signature.
//
// Projection correction u -= grad(phi) on the staggered faces (correct_k port). No openness here —
// the openness lives in the operator + divergence; closed faces carry phi~0 on both sides so stay
// unchanged.
inline void projectCorrect(CCField u, CCField v, CCField w, CCConst phi, C3 e, int g,
                           double wx = 1.0, double wy = 1.0, double wz = 1.0) {
  ccFor3(
      "peclet::flow::correct", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= wx * (phi(i) - phi(i - sx));
        v(i) -= wy * (phi(i) - phi(i - sy));
        w(i) -= wz * (phi(i) - phi(i - sz));
      });
}

// Variable-density projection correction (sibling of projectCorrect): u_f -= (rho0/rho_f) grad(phi)
// with rho_f the arithmetic face mean — the SAME face density that scaled the Poisson coefficient
// c_f = open_f*rho0/rho_f, so the corrected open flux telescopes to A*phi exactly (discrete
// consistency; constant rho == rho0 reduces to projectCorrect identically, ratio 1.0 exact in FP).
inline void projectCorrectVar(CCField u, CCField v, CCField w, CCConst phi, CCConst rho,
                              double rho0, C3 e, int g, double wx = 1.0, double wy = 1.0,
                              double wz = 1.0) {
  ccFor3(
      "peclet::flow::correct_var", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= wx * (rho0 / (0.5 * (rho(i) + rho(i - sx))) * (phi(i) - phi(i - sx)));
        v(i) -= wy * (rho0 / (0.5 * (rho(i) + rho(i - sy))) * (phi(i) - phi(i - sy)));
        w(i) -= wz * (rho0 / (0.5 * (rho(i) + rho(i - sz))) * (phi(i) - phi(i - sz)));
      });
}

// Variable-density Poisson face coefficients on the MG (g=1) block: c_f = open_f * rho0 / rho_f,
// rho_f = arithmetic face mean. Computed over the inner cells only — CutcellMG::setOpenness runs
// its own periodic/halo ghost fill + non-periodic boundary re-imposition on whatever level-0 fields
// it receives, exactly as for the raw openness (the coefficient "rides the openness rails"). The
// rho ghost ring of the g=1 block must be valid (bridged from the filled G=2 field).
inline void buildRhoCoeff(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy, CCConst oz,
                          CCConst rho, double rho0, C3 e, int g) {
  ccFor3(
      "peclet::flow::rho_coeff", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        cx(i) = ox(i) * rho0 / (0.5 * (rho(i) + rho(i - sx)));
        cy(i) = oy(i) * rho0 / (0.5 * (rho(i) + rho(i - sy)));
        cz(i) = oz(i) * rho0 / (0.5 * (rho(i) + rho(i - sz)));
      });
}

// --- Harmonic rho_f face mean (VoF rung V2a / WO-J item 5; the flagged coarsening trap of
// MULTIPHYSICS_PLAN.md:474) -------------------------------------------------------------------
//
// SIBLINGS of buildRhoCoeff / projectCorrectVar with rho_f = 2*rho_a*rho_b/(rho_a+rho_b) instead of
// the arithmetic mean. Default OFF (set_rho_face_harmonic); the validated arithmetic kernels above
// are untouched and stay the default.
//
// Read this before turning it on. The two means are NOT interchangeable, and the arithmetic one is
// the correct fine-level choice for THIS discretization, for two independent reasons:
//   1. The face coefficient is c_f = open_f * rho0/rho_f, i.e. proportional to the mobility
//      beta = 1/rho. The ARITHMETIC mean of rho is exactly the HARMONIC mean of beta — the
//      series-resistance / homogenization-correct choice for a flux crossing two half-cells
//      (VOF_PLAN.md §5). Taking rho harmonic makes beta arithmetic, i.e. the parallel rule, which
//      is the wrong one for a normal flux.
//   2. Discrete hydrostatic balance is EXACT only because the momentum time term, the
//      face-interpolated body force, and the projection coefficient use the SAME rho_f
//      (doc/variable_density_projection.md §1/§3). The momentum side interpolates rho
//      arithmetically (mass is volume-additive on the staggered control volume) and is not part of
//      this switch, so switching only the projection breaks that three-way consistency and leaves
//      a permanent spurious velocity at the interface. This is measured, not asserted — see the
//      WO-J findings entry.
// The knob exists because arithmetic COARSENING of the coefficient is what makes the V-cycle
// preconditioner indefinite past ratio ~1e3 (doc/variable_density_projection.md §2), so the
// question "does a harmonic face mean help the solver" needs to be answerable by measurement
// rather than by argument. It ships measured.
//
// The projection stays an EXACT projection either way: the operator coefficient and the velocity
// correction must use the same rho_f or the corrected open flux no longer telescopes to A*phi, so
// the flag switches BOTH kernels together (IbmSolver::project / projectVelocities).
inline void buildRhoCoeffHarm(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy,
                              CCConst oz, CCConst rho, double rho0, C3 e, int g) {
  ccFor3(
      "peclet::flow::rho_coeff_harm", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        // 2 a b/(a+b) written as rho0*(a+b)/(2ab) for the coefficient rho0/rho_f.
        cx(i) = ox(i) * rho0 * (rho(i) + rho(i - sx)) / (2.0 * rho(i) * rho(i - sx));
        cy(i) = oy(i) * rho0 * (rho(i) + rho(i - sy)) / (2.0 * rho(i) * rho(i - sy));
        cz(i) = oz(i) * rho0 * (rho(i) + rho(i - sz)) / (2.0 * rho(i) * rho(i - sz));
      });
}

// WO-R2 item 1 — the variable-density coefficient of the HIGH-side outflow domain face.
//
// `buildRhoCoeff` runs over the INNER cells, and the staggered face index of a cell is its LOW
// face, so the low domain face of an axis (inner index `g`) is covered and the high one
// (`dims-g`, a ghost index) is not. `CutcellMG::applyBoundaryOpenness` used to fill that gap with
// the literal openness 1.0, which is right for the raw-openness operator and wrong by the density
// ratio for the coefficient one. This writes the missing plane, `open_f * rho0/rho_f` with:
//   * open_f = 1 — the same fully-open value the raw path imposed there (the g=1 coefficient
//     block's ghost ring carries no geometric openness of its own; a solid cutting an OUTFLOW
//     plane is a separate, pre-existing gap that the raw path has too);
//   * rho_f = the arithmetic (or harmonic) face mean of the LAST INNER cell and the domain ghost,
//     literally the expression `bcCorrectOutflowVar` applies at that face — which is what makes
//     the projection there exact. Under the Neumann property ghost (`fillPropGhosts`) both
//     evaluate to the last inner cell's own rho.
// The whole transverse plane is written (ghost ring included) so that a distributed CA level's
// ring rows coarsen from a valid fine plane; `rho`'s ghost ring must be filled first, which is
// what the bridge in IbmSolver::project guarantees.
inline void buildRhoCoeffOutflowFace(CCField ca, CCConst oa, CCConst rho, double rho0, C3 e, int g,
                                     int a, bool harmonic) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int bf = dims[a] - g;
  using MD = MDRange2<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::rho_coeff_outflow", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long i = (long)p0 * sb + (long)p1 * sc + (long)bf * sa;
        const double ra = rho(i), rb = rho(i - sa);
        const double rf = harmonic ? (2.0 * ra * rb / (ra + rb)) : (0.5 * (ra + rb));
        // SCALING_ISSUES #3: `oa(i)` is this face's cut-cell aperture, which every OTHER face of
        // the coefficient carries (buildRhoCoeff multiplies by it) and this one used to drop --
        // invisible while the outlet is clear of solid (aperture 1), an operator that disagrees
        // with the divergence constraint by (1 - aperture) once a solid cuts the outlet.
        ca(i) = oa(i) * rho0 / rf;
      });
}

inline void projectCorrectVarHarm(CCField u, CCField v, CCField w, CCConst phi, CCConst rho,
                                  double rho0, C3 e, int g, double wx = 1.0, double wy = 1.0,
                                  double wz = 1.0) {
  ccFor3(
      "peclet::flow::correct_var_harm", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= wx * (rho0 * (rho(i) + rho(i - sx)) / (2.0 * rho(i) * rho(i - sx)) *
                      (phi(i) - phi(i - sx)));
        v(i) -= wy * (rho0 * (rho(i) + rho(i - sy)) / (2.0 * rho(i) * rho(i - sy)) *
                      (phi(i) - phi(i - sy)));
        w(i) -= wz * (rho0 * (rho(i) + rho(i - sz)) / (2.0 * rho(i) * rho(i - sz)) *
                      (phi(i) - phi(i - sz)));
      });
}

// --- Volume-averaged (porous) continuity for unresolved CFD-DEM
// ----------------------------------- The proper continuity is d(eps)/dt + div(eps u) = 0 (eps =
// void fraction from the particles), so the velocity is NOT solenoidal: div(eps u) = -d(eps)/dt.
// These size the projection to that constraint.

// eps-weighted open-face divergence: d = div(open * eps_f * u), eps_f = arithmetic face mean.
// Reduces to divergOpen when eps == 1 everywhere (no particles).
inline void divergOpenEps(CCConst u, CCConst v, CCConst w, CCConst ox, CCConst oy, CCConst oz,
                          CCConst eps, CCField d, C3 e, int g) {
  ccFor3(
      "peclet::flow::diverg_open_eps", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double exp = 0.5 * (eps(i) + eps(i + sx)), exm = 0.5 * (eps(i) + eps(i - sx));
        const double eyp = 0.5 * (eps(i) + eps(i + sy)), eym = 0.5 * (eps(i) + eps(i - sy));
        const double ezp = 0.5 * (eps(i) + eps(i + sz)), ezm = 0.5 * (eps(i) + eps(i - sz));
        d(i) = (ox(i + sx) * exp * u(i + sx) - ox(i) * exm * u(i)) +
               (oy(i + sy) * eyp * v(i + sy) - oy(i) * eym * v(i)) +
               (oz(i + sz) * ezp * w(i + sz) - oz(i) * ezm * w(i));
      });
}

// Porous Poisson face coefficient c_f = open_f * eps_f (eps_f = arithmetic face mean).
// Constant-density gas: the correction stays u -= grad(phi) (projectCorrect) so the open*eps flux
// telescopes to A*phi. (Combining with variable rho — c_f *= rho0/rho_f, projectCorrectVar — is a
// later composition.)
// SCALING_ISSUES #3: the three porous builders run ONE index past the inner block along each axis,
// so the HIGH domain face of each axis -- a ghost index no other kernel writes, and the face the
// MG's `applyBoundaryOpenness` carries at a Dirichlet outlet -- is built by the same formula as
// every other face, cut-cell aperture `o(i)` included. The extra plane is read-only elsewhere (the
// per-level ghost fill overwrites it) and every operand it reads is a ghost-filled index.
inline void buildPorousCoeff(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy, CCConst oz,
                             CCConst eps, C3 e, int g) {
  ccFor3(
      "peclet::flow::porous_coeff", C3{g, g, g}, C3{e.x - g + 1, e.y - g + 1, e.z - g + 1},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        cx(i) = ox(i) * 0.5 * (eps(i) + eps(i - sx));
        cy(i) = oy(i) * 0.5 * (eps(i) + eps(i - sy));
        cz(i) = oz(i) * 0.5 * (eps(i) + eps(i - sz));
      });
}

// Semi-implicit-drag porous coefficient: c_f = open_f * eps_f * w_f, with the face drag-relaxation
// w_f = idt/(idt + beta_f) (idt = rho/dt, beta = the momentum-diagonal drag coefficient). This
// makes the pressure correction CONSISTENT with the drag-loaded momentum diagonal A_P = idt + beta:
// where the drag is stiff (dense bed) w_f -> 0 and the pressure barely moves the velocity (the drag
// holds it) — the SIMPLE/PISO-with-implicit-drag scheme (OpenFOAM rAU, MFIX). Reduces to
// buildPorousCoeff when beta==0 (w==1). The correction MUST use the same w_f
// (projectCorrectPorousDrag) so the open*eps*w flux telescopes to A*phi.
inline void buildPorousCoeffDrag(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy,
                                 CCConst oz, CCConst eps, CCConst beta, double idt, C3 e, int g) {
  ccFor3(
      "peclet::flow::porous_coeff_drag", C3{g, g, g}, C3{e.x - g + 1, e.y - g + 1, e.z - g + 1},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        cx(i) = ox(i) * 0.5 * (eps(i) + eps(i - sx)) * idt / (idt + 0.5 * (beta(i) + beta(i - sx)));
        cy(i) = oy(i) * 0.5 * (eps(i) + eps(i - sy)) * idt / (idt + 0.5 * (beta(i) + beta(i - sy)));
        cz(i) = oz(i) * 0.5 * (eps(i) + eps(i - sz)) * idt / (idt + 0.5 * (beta(i) + beta(i - sz)));
      });
}

// --- eps-CONSERVATIVE porous projection pair -------------------------------------------------
// For the conservative volume-averaged momentum (time term (eps_f rho/dt) u, Model-B full -grad p)
// the face momentum diagonal is D_f = eps_f*rho*idt + beta_f, the velocity correction is
//   u_f -= (rho*idt / D_f) * grad(phi)          (beta=0, eps=1 -> plain projectCorrect exactly)
// and the flux constraint div(eps u) = rhs makes the Poisson coefficient
//   c_f = open_f * eps_f * (rho*idt/D_f) = open_f * (eps_f*rho*idt) / D_f
// — the eps of the flux cancels against the eps of the inertia, unlike the plain-u pair above
// (buildPorousCoeff*/projectCorrectPorousDrag), which is consistent only for the non-conservative
// momentum form and kinematically drags gas along with the moving porosity (energy injection).
inline void buildPorousCoeffCons(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy,
                                 CCConst oz, CCConst eps, CCConst beta, bool useBeta, double rhoidt,
                                 C3 e, int g) {
  ccFor3(
      "peclet::flow::porous_coeff_cons", C3{g, g, g}, C3{e.x - g + 1, e.y - g + 1, e.z - g + 1},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        auto cf = [&](long s, CCConst o) {
          const double epsF = 0.5 * (eps(i) + eps(i - s));
          const double bF = useBeta ? 0.5 * (beta(i) + beta(i - s)) : 0.0;
          const double inert = epsF * rhoidt;
          return o(i) * inert / (inert + bF) * 1.0;
        };
        cx(i) = cf(sx, ox);
        cy(i) = cf(sy, oy);
        cz(i) = cf(sz, oz);
      });
}
inline void projectCorrectPorousCons(CCField u, CCField v, CCField w, CCConst phi, CCConst eps,
                                     CCConst beta, bool useBeta, double rhoidt, C3 e, int g,
                                     double wx = 1.0, double wy = 1.0, double wz = 1.0) {
  ccFor3(
      "peclet::flow::correct_porous_cons", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        auto wf = [&](long s) {
          const double epsF = 0.5 * (eps(i) + eps(i - s));
          const double bF = useBeta ? 0.5 * (beta(i) + beta(i - s)) : 0.0;
          return rhoidt / (epsF * rhoidt + bF);
        };
        u(i) -= wx * (wf(sx) * (phi(i) - phi(i - sx)));
        v(i) -= wy * (wf(sy) * (phi(i) - phi(i - sy)));
        w(i) -= wz * (wf(sz) * (phi(i) - phi(i - sz)));
      });
}

// Drag-relaxed velocity correction (sibling of projectCorrect): u_f -= w_f * grad(phi), w_f the
// SAME face drag relaxation as buildPorousCoeffDrag. beta==0 -> w==1 -> projectCorrect exactly.
inline void projectCorrectPorousDrag(CCField u, CCField v, CCField w, CCConst phi, CCConst beta,
                                     double idt, C3 e, int g, double wx = 1.0, double wy = 1.0,
                                     double wz = 1.0) {
  ccFor3(
      "peclet::flow::correct_porous_drag", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= wx * (idt / (idt + 0.5 * (beta(i) + beta(i - sx))) * (phi(i) - phi(i - sx)));
        v(i) -= wy * (idt / (idt + 0.5 * (beta(i) + beta(i - sy))) * (phi(i) - phi(i - sy)));
        w(i) -= wz * (idt / (idt + 0.5 * (beta(i) + beta(i - sz))) * (phi(i) - phi(i - sz)));
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_PRESSURE_HPP
