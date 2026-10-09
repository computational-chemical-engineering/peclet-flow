// D -- the FP32 V-cycle preconditioner's kernels (doc/vof_projection_cost_design.md §4.4.3,
// §4.4.4 items 1-4). Included by mac_cutcell_mg.hpp after VReal, ccWrapNbrs, ccWrapRowPeeled and
// the FP64 transfer kernels; CutcellMG drives them (precondVcycleFp32). The FP64 bodies are not
// touched: every FP32 body is new and is used only on an eligible solve (§4.4.1).
//
// Storage per level: the FP32 face weights WX, WY, WZ (w_f = fl32(t_f) > 0, each face stored once
// and read by both of its cells) and the FP32 iterates xf, rhsf, resf. The diagonal is never
// stored in any precision: D_i is summed from the six weights the update reads anyway, and the
// arithmetic is the flux/correction form, so x = c * 1 gives r = 0 exactly (§4.2 P1, P5). Every
// literal in an FP32 body carries `f`; no double variable enters an FP32 body.
#pragma once

namespace peclet::flow {

using VField = Kokkos::View<VReal*, CCMem>;
using VConst = Kokkos::View<const VReal*, CCMem>;

// §4.4.3: D_i = ((((WX(i+sx) + WX(i)) + WY(i+sy)) + WY(i)) + WZ(i+sz)) + WZ(i), FP32, this order.
template <class WV>
KOKKOS_INLINE_FUNCTION VReal fp32Diag(const WV& WX, const WV& WY, const WV& WZ, long i, long sx,
                                      long sy, long sz) {
  return ((((WX(i + sx) + WX(i)) + WY(i + sy)) + WY(i)) + WZ(i + sz)) + WZ(i);
}
// §4.4.3: q_i = sum_f w_f (x_i - x_j(f)), left to right in the stated face order.
template <class XV, class WV>
KOKKOS_INLINE_FUNCTION VReal fp32Flux(const XV& x, const WV& WX, const WV& WY, const WV& WZ, long i,
                                      long sx, long sy, long sz, long xp, long xm, long yp, long ym,
                                      long zp, long zm) {
  const VReal xi = x(i);
  return WX(i) * (xi - x(xm)) + WX(i + sx) * (xi - x(xp)) + WY(i) * (xi - x(ym)) +
         WY(i + sy) * (xi - x(yp)) + WZ(i) * (xi - x(zm)) + WZ(i + sz) * (xi - x(zp));
}
// §4.4.3 smoother cell: x_i = x_i + (b_i - q_i) / D_i on a cell with D_i != 0 (else skipped).
template <class XV, class BV, class WV>
KOKKOS_INLINE_FUNCTION void fp32SmoothCell(const XV& x, const BV& b, const WV& WX, const WV& WY,
                                           const WV& WZ, long i, long sx, long sy, long sz, long xp,
                                           long xm, long yp, long ym, long zp, long zm) {
  const VReal d = fp32Diag(WX, WY, WZ, i, sx, sy, sz);
  if (d == 0.0f)
    return;  // decoupled (the FP64 AC < 1e-30 set, §4.4.1 (5)): x stays as it is
  const VReal q = fp32Flux(x, WX, WY, WZ, i, sx, sy, sz, xp, xm, yp, ym, zp, zm);
  x(i) = x(i) + (b(i) - q) / d;
}
// §4.4.3 residual cell: r_i = b_i - q_i.
template <class RV, class XV, class BV, class WV>
KOKKOS_INLINE_FUNCTION void fp32ResidualCell(const RV& r, const XV& x, const BV& b, const WV& WX,
                                             const WV& WY, const WV& WZ, long i, long sx, long sy,
                                             long sz, long xp, long xm, long yp, long ym, long zp,
                                             long zm) {
  r(i) = b(i) - fp32Flux(x, WX, WY, WZ, i, sx, sy, sz, xp, xm, yp, ym, zp, zm);
}

// One red-black colour pass of the FP32 smoother with the A3 wrapped neighbour reads (every inner
// dimension even; cutcellSmoothColorFaceWrap's launch forms: host pencils with the row ends
// peeled, device MDRange).
inline void cutcellSmoothColorFp32Wrap(VField x, VConst b, VConst WX, VConst WY, VConst WZ, C3 e,
                                       C3 n, C3 og, int g, int color) {
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    const long cells = (long)nyi * nzi * (e.x - 2 * g);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      ccWrapRowPeeled(ly, lz, e, n, g, g + ((P ^ (g & 1)) & 1), 2,
                      [&](long i, long xp, long xm, long yp, long ym, long zp, long zm) {
                        fp32SmoothCell(x, b, WX, WY, WZ, i, sx, sy, sz, xp, xm, yp, ym, zp, zm);
                      });
    };
    if (hostRunSerial(cells)) {
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth_f32",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth_f32", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        fp32SmoothCell(x, b, WX, WY, WZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym, w.zp, w.zm);
      });
}
// The FP32 residual r = b - W x over the inner cells with the wrapped reads of x.
inline void residualFp32Wrap(VField r, VConst x, VConst b, VConst WX, VConst WY, VConst WZ, C3 e,
                             C3 n, int g) {
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
    ccWrapForPeeled("peclet::flow::cc_residual_f32", e, n, g,
                    [=](long i, long xp, long xm, long yp, long ym, long zp, long zm) {
                      fp32ResidualCell(r, x, b, WX, WY, WZ, i, sx, sy, sz, xp, xm, yp, ym, zp, zm);
                    });
    return;
  }
  ccFor3(
      "peclet::flow::cc_residual_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        fp32ResidualCell(r, x, b, WX, WY, WZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym, w.zp, w.zm);
      });
}

// §4.4.4 item 1, the level-0 ENTRY (replaces the z = 0 memset and the first pre-smooth colour
// pass): rhsf = fl32(rr * sigma) on every inner cell (sigma = 2^-e: exact); on the cells of
// `color` with D_i != 0 (when `smoothFirst`) xf = rhsf / D_i, on every other inner cell xf = 0.
inline void entryFp32(VField xf, VField rhsf, CCConst rr, VConst WX, VConst WY, VConst WZ,
                      double sigma, C3 e, C3 og, int g, int color, bool smoothFirst) {
  ccFor3(
      "peclet::flow::mg_entry_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const VReal b = (VReal)(rr(i) * sigma);
        rhsf(i) = b;
        VReal x0 = 0.0f;
        if (smoothFirst && ((og.x + lx + og.y + ly + og.z + lz) & 1) == color) {
          const VReal d = fp32Diag(WX, WY, WZ, i, sx, sy, sz);
          if (d != 0.0f)
            x0 = b / d;
        }
        xf(i) = x0;
      });
}

// §4.4.4 item 2, restriction FP32 -> FP32 that zeroes the coarse iterate on the same cells (the
// FP32 restrictAvgZeroX): the children summed in FP32 in restrictAvgCell's order, times the
// inverse child count (a power of two: 0.125f at ratio 2 on every axis).
inline void restrictFp32ZeroX(VField coarse, VField coarseX, VConst fine, C3 cext, C3 fext, int gc,
                              int gf, C3 cinner, C3 ratio) {
  const VReal inv = (VReal)(1.0 / (double)(ratio.x * ratio.y * ratio.z));
  ccFor3(
      "peclet::flow::restrict_f32", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        VReal s = 0.0f;
        for (int dz = 0; dz < ratio.z; ++dz)
          for (int dy = 0; dy < ratio.y; ++dy)
            for (int dx = 0; dx < ratio.x; ++dx) {
              const int fx = ratio.x * icx + dx + gf, fy = ratio.y * icy + dy + gf,
                        fz = ratio.z * icz + dz + gf;
              s += fine((long)fx + (long)fy * fsy + (long)fz * fsz);
            }
        const long ci =
            (long)(icx + gc) + (long)(icy + gc) * cext.x + (long)(icz + gc) * (long)cext.x * cext.y;
        coarse(ci) = s * inv;
        coarseX(ci) = 0.0f;
      });
}
// §4.4.4 item 3, the bottom interface down: restriction FP32 -> FP64, the children summed in
// double from (double)fine, times the double inverse count, into the bottom's rhs; zeroes the
// bottom's x on the same cells.
inline void restrictFp32ToDZeroX(CCField coarse, CCField coarseX, VConst fine, C3 cext, C3 fext,
                                 int gc, int gf, C3 cinner, C3 ratio) {
  const double inv = 1.0 / (double)(ratio.x * ratio.y * ratio.z);
  ccFor3(
      "peclet::flow::restrict_f32d", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        double s = 0.0;
        for (int dz = 0; dz < ratio.z; ++dz)
          for (int dy = 0; dy < ratio.y; ++dy)
            for (int dx = 0; dx < ratio.x; ++dx) {
              const int fx = ratio.x * icx + dx + gf, fy = ratio.y * icy + dy + gf,
                        fz = ratio.z * icz + dz + gf;
              s += (double)fine((long)fx + (long)fy * fsy + (long)fz * fsz);
            }
        const long ci =
            (long)(icx + gc) + (long)(icy + gc) * cext.x + (long)(icz + gc) * (long)cext.x * cext.y;
        coarse(ci) = s * inv;
        coarseX(ci) = 0.0;
      });
}

// The coarse sample of prolongAddCell (§14 H-2's integer form): lower coarse index and weight.
KOKKOS_INLINE_FUNCTION int fp32ProlongLow(int ratio, int i, int gc) {
  return (ratio == 2) ? (i >> 1) + gc - 1 + (i & 1) : i + gc;
}
// §4.4.4 item 2, prolongation FP32 <- FP32 (added to fine): prolongAddCell's trilinear form with
// the weights 0.25f / 0.75f and (1.0f - w) written as a subtraction.
inline void prolongFp32(VField fine, VConst coarse, C3 fext, C3 cext, int gf, int gc, C3 finner,
                        C3 ratio) {
  ccFor3(
      "peclet::flow::prolong_f32", C3{0, 0, 0}, C3{finner.x, finner.y, finner.z},
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        const int x0 = fp32ProlongLow(ratio.x, ifx, gc), y0 = fp32ProlongLow(ratio.y, ify, gc),
                  z0 = fp32ProlongLow(ratio.z, ifz, gc);
        const VReal wx = (ratio.x == 2) ? ((ifx & 1) ? 0.25f : 0.75f) : 0.0f;
        const VReal wy = (ratio.y == 2) ? ((ify & 1) ? 0.25f : 0.75f) : 0.0f;
        const VReal wz = (ratio.z == 2) ? ((ifz & 1) ? 0.25f : 0.75f) : 0.0f;
        const long sy = cext.x, sz = (long)cext.x * cext.y;
        auto C = [&](int xx, int yy, int zz) {
          return coarse((long)xx + (long)yy * sy + (long)zz * sz);
        };
        const VReal c00 = C(x0, y0, z0) * (1.0f - wx) + C(x0 + 1, y0, z0) * wx;
        const VReal c10 = C(x0, y0 + 1, z0) * (1.0f - wx) + C(x0 + 1, y0 + 1, z0) * wx;
        const VReal c01 = C(x0, y0, z0 + 1) * (1.0f - wx) + C(x0 + 1, y0, z0 + 1) * wx;
        const VReal c11 = C(x0, y0 + 1, z0 + 1) * (1.0f - wx) + C(x0 + 1, y0 + 1, z0 + 1) * wx;
        const VReal c0 = c00 * (1.0f - wy) + c10 * wy, c1 = c01 * (1.0f - wy) + c11 * wy;
        const long fi =
            (long)(ifx + gf) + (long)(ify + gf) * fext.x + (long)(ifz + gf) * (long)fext.x * fext.y;
        fine(fi) = fine(fi) + (c0 * (1.0f - wz) + c1 * wz);
      });
}
// §4.4.4 item 3, the bottom interface up: prolongation FP32 <- FP64, the interpolation in double
// (prolongAddCell's expression), xf = fl32((double)xf + interp).
inline void prolongFp32FromD(VField fine, CCConst coarse, C3 fext, C3 cext, int gf, int gc,
                             C3 finner, C3 ratio) {
  ccFor3(
      "peclet::flow::prolong_f32d", C3{0, 0, 0}, C3{finner.x, finner.y, finner.z},
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        const int x0 = fp32ProlongLow(ratio.x, ifx, gc), y0 = fp32ProlongLow(ratio.y, ify, gc),
                  z0 = fp32ProlongLow(ratio.z, ifz, gc);
        const double wx = (ratio.x == 2) ? ((ifx & 1) ? 0.25 : 0.75) : 0.0;
        const double wy = (ratio.y == 2) ? ((ify & 1) ? 0.25 : 0.75) : 0.0;
        const double wz = (ratio.z == 2) ? ((ifz & 1) ? 0.25 : 0.75) : 0.0;
        const long sy = cext.x, sz = (long)cext.x * cext.y;
        auto C = [&](int xx, int yy, int zz) {
          return coarse((long)xx + (long)yy * sy + (long)zz * sz);
        };
        const double c00 = C(x0, y0, z0) * (1 - wx) + C(x0 + 1, y0, z0) * wx;
        const double c10 = C(x0, y0 + 1, z0) * (1 - wx) + C(x0 + 1, y0 + 1, z0) * wx;
        const double c01 = C(x0, y0, z0 + 1) * (1 - wx) + C(x0 + 1, y0, z0 + 1) * wx;
        const double c11 = C(x0, y0 + 1, z0 + 1) * (1 - wx) + C(x0 + 1, y0 + 1, z0 + 1) * wx;
        const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
        const long fi =
            (long)(ifx + gf) + (long)(ify + gf) * fext.x + (long)(ifz + gf) * (long)fext.x * fext.y;
        fine(fi) = (VReal)((double)fine(fi) + (c0 * (1 - wz) + c1 * wz));
      });
}

}  // namespace peclet::flow
