/// @file
/// @brief flow — portable (Kokkos) IBM geometric fields + variable-coefficient RB-GS smoother.
///
/// Kokkos port of the self-contained pieces of mac_ibm.cuh: cut-cell detection (ibm_is_cut), the
/// staggered SDF gather, volume fraction (ibm_volfrac_k), solid + clean-fluid masks, and the
/// variable-coefficient Red-Black Gauss-Seidel smoother (ibm_rbgs_stencil_k) for the IBM-modified
/// momentum operator (mixed precision: float matrix coefficients, double iterate). The
/// Robust-Scaled overlay BUILD (ibm_geometry/modify_stencil with the IBM_Data SoA + poly_*) is the
/// heavier piece, left for a dedicated follow-up. Reuses the cut-cell SDF sampler. Runs on any
/// Kokkos backend.
#ifndef PECLET_FLOW_MAC_IBM_HPP
#define PECLET_FLOW_MAC_IBM_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "cut_cell_ibm.hpp"  // IbmOverlay, ibmFillEntry
#include "mac_cutcell.hpp"   // peclet::flow::C3, peclet::flow::ccSampleExt, CCConst

namespace peclet::flow {

using mreal = float;  // matrix coefficient type (matches cfd's mreal)
using MConst = Kokkos::View<const float*, CCMem>;

struct Off3 {
  float x, y, z;
};

// Cut cell = fluid centre with at least one solid axis neighbour.
KOKKOS_INLINE_FUNCTION bool ibmIsCut(float sc, const float sn[6]) {
  if (sc <= 0.0f)
    return false;
  for (int k = 0; k < 6; ++k)
    if (sn[k] < 0.0f)
      return true;
  return false;
}

// Find cut cells over the inner block and build the Robust-Scaled overlay (port of ibm_count_ext_k
// + ibm_geometry_ext_k): per inner cell, gather the 7 staggered SDF samples; if cut, atomically
// claim a slot, set idMap[cell]=slot, and fill the overlay. counter/idMap are reset here. Returns
// the cut count (overlay arrays must be sized >= number of inner cells). bc_type 0=Dirichlet,
// 1=Neumann.
template <int SCHEME>
inline int buildIbmOverlay(CCConst sdf, C3 ext, int g, Off3 off, int bc_type, const IbmOverlay& ov,
                           Kokkos::View<int*, CCMem> idMap, Kokkos::View<int, CCMem> counter,
                           CCConst tx = CCConst(), CCConst ty = CCConst(), CCConst tz = CCConst(),
                           C3 nn = C3{0, 0, 0}) {
  CCExec space;
  Kokkos::deep_copy(space, counter, 0);
  Kokkos::deep_copy(space, idMap, -1);
  const bool hasEx = tx.size() > 0;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_build_overlay", MD(space, {g, g, g}, {ext.x - g, ext.y - g, ext.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long idx = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
        const float sc = (float)ccSampleExt(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
        const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        float sn[6];
        for (int k = 0; k < 6; ++k)
          sn[k] = (float)ccSampleExt(sdf, ext, lx + d[k][0] + off.x, ly + d[k][1] + off.y,
                                     lz + d[k][2] + off.z);
        if (!ibmIsCut(sc, sn))
          return;
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        idMap(idx) = slot;
        // Optional analytic-SDF exact crossings (setExactCrossings): tx/ty/tz are INNER-grid
        // arrays, t_a(i) = exact crossing fraction from this component's staggered point i toward
        // i + e_a (NaN = no crossing). +a dir: theta = t_a(i); -a dir: theta = 1 - t_a(i - e_a),
        // periodic wrap on the inner grid.
        float thEx[6];
        if (hasEx) {
          const int ix = lx - g, iy = ly - g, iz = lz - g;
          auto wrap = [](int v, int n) { v %= n; return v < 0 ? v + n : v; };
          const CCConst* ta[3] = {&tx, &ty, &tz};
          for (int a = 0; a < 3; ++a) {
            const long ip = (long)ix + (long)iy * nn.x + (long)iz * (long)nn.x * nn.y;
            const int mx = a == 0 ? wrap(ix - 1, nn.x) : ix;
            const int my = a == 1 ? wrap(iy - 1, nn.y) : iy;
            const int mz = a == 2 ? wrap(iz - 1, nn.z) : iz;
            const long im = (long)mx + (long)my * nn.x + (long)mz * (long)nn.x * nn.y;
            thEx[2 * a] = (float)(*ta[a])(ip);
            thEx[2 * a + 1] = 1.0f - (float)(*ta[a])(im);  // NaN propagates -> fallback
          }
        }
        ibmFillEntry<SCHEME>(ov, slot, (int)idx, sc, sn, bc_type, hasEx ? thEx : nullptr);
      });

  int cnt = 0;
  Kokkos::deep_copy(cnt, counter);
  return cnt;
}

// Volume fraction theta = clamp(0.5 + sdf_sample, 0, 1) at the staggered point (lx+off, ...).
inline void ibmVolfrac(CCField theta, CCConst sdf, C3 ext, Off3 off) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_volfrac", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
        const double sd = ccSampleExt(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
        const double t = 0.5 + sd;
        theta(i) = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
      });
}

// Solid mask: 1 where the staggered SDF point is inside the solid (sd<0), else 0.
inline void ibmSolidMask(CCField mask, CCConst sdf, C3 ext, Off3 off) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_solid", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
        const double sd = ccSampleExt(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
        mask(i) = (sd < 0.0) ? 1.0 : 0.0;
      });
}

// Clean-fluid-interior mask: 1 only at fluid cells with no solid neighbour (not cut, not solid).
inline void ibmCleanFluidMask(CCField m, CCConst sdf, C3 ext, Off3 off) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_clean", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
        const float sc = (float)ccSampleExt(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
        const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        float sn[6];
        for (int k = 0; k < 6; ++k)
          sn[k] = (float)ccSampleExt(sdf, ext, lx + d[k][0] + off.x, ly + d[k][1] + off.y,
                                     lz + d[k][2] + off.z);
        const bool solid = (sc <= 0.0f);
        m(i) = (solid || ibmIsCut(sc, sn)) ? 0.0 : 1.0;
      });
}

// One Red-Black sweep of the variable-coefficient stencil: x[i] = (b[i] - sum(A_off*x_nbr)) /
// A_C[i]. float matrix coeffs promote to double; solid cells pinned to 0. Call colour 0 then 1.
// Host backends take a line-sweep form: one task per (y,z) pencil, contiguous stride-2 x-loop
// starting at the colour's parity — no per-cell parity branch, unit-stride reads. Bit-identical to
// the MDRange form (same-colour cells are independent, so update order within a colour cannot
// change the result); the device keeps the MDRange form untouched.
inline void ibmRbgsStencilColor(CCField x, CCConst b, MConst AC, MConst AW, MConst AE, MConst AS,
                                MConst AN, MConst AB, MConst AT, CCConst solidmask, C3 ext, C3 og,
                                int g, int color) {
  CCExec space;
  const bool hasMask = (solidmask.extent(0) != 0);
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = ext.y - 2 * g, nzi = ext.z - 2 * g;
    Kokkos::parallel_for(
        "peclet::flow::ibm_rbgs", Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi),
        KOKKOS_LAMBDA(long t) {
          const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
          const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
          const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
          for (int lx = g + ((P ^ (g & 1)) & 1); lx < ext.x - g; lx += 2) {
            const long i = (long)lx + (long)ly * sy + (long)lz * sz;
            if (hasMask && solidmask(i) > 0.5) {
              x(i) = 0.0;
              continue;
            }
            const double ac = AC(i);
            if (Kokkos::fabs(ac) < 1e-30)
              continue;
            const double s = (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                             (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) +
                             (double)AT(i) * x(i + sz) + (double)AB(i) * x(i - sz);
            x(i) = (b(i) - s) / ac;
          }
        });
    return;
  }
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_rbgs", MD(space, {g, g, g}, {ext.x - g, ext.y - g, ext.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
        const long i = (long)lx + (long)ly * ext.x + (long)lz * sz;
        if (hasMask && solidmask(i) > 0.5) {
          x(i) = 0.0;
          return;
        }
        const double ac = AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double s = (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                         (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) +
                         (double)AT(i) * x(i + sz) + (double)AB(i) * x(i - sz);
        x(i) = (b(i) - s) / ac;
      });
}

// ibmRbgsStencilColor + a fused max|Δx| reduction over the swept colour (values already in
// registers — no extra memory pass). The momentum tolerance stop runs this as the SECOND colour of
// a sweep: one colour's max increment is the convergence proxy. Same math as the plain sweep.
inline double ibmRbgsStencilColorDu(CCField x, CCConst b, MConst AC, MConst AW, MConst AE,
                                    MConst AS, MConst AN, MConst AB, MConst AT, CCConst solidmask,
                                    C3 ext, C3 og, int g, int color) {
  CCExec space;
  const bool hasMask = (solidmask.extent(0) != 0);
  double du = 0.0;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = ext.y - 2 * g, nzi = ext.z - 2 * g;
    Kokkos::parallel_reduce(
        "peclet::flow::ibm_rbgs_du", Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi),
        KOKKOS_LAMBDA(long t, double& m) {
          const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
          const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
          const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
          for (int lx = g + ((P ^ (g & 1)) & 1); lx < ext.x - g; lx += 2) {
            const long i = (long)lx + (long)ly * sy + (long)lz * sz;
            if (hasMask && solidmask(i) > 0.5) {
              x(i) = 0.0;
              continue;
            }
            const double ac = AC(i);
            if (Kokkos::fabs(ac) < 1e-30)
              continue;
            const double s = (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                             (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) +
                             (double)AT(i) * x(i + sz) + (double)AB(i) * x(i - sz);
            const double xn = (b(i) - s) / ac;
            const double d = Kokkos::fabs(xn - x(i));
            if (d > m)
              m = d;
            x(i) = xn;
          }
        },
        Kokkos::Max<double>(du));
    return du;
  }
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "peclet::flow::ibm_rbgs_du", MD(space, {g, g, g}, {ext.x - g, ext.y - g, ext.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz, double& m) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
        const long i = (long)lx + (long)ly * ext.x + (long)lz * sz;
        if (hasMask && solidmask(i) > 0.5) {
          x(i) = 0.0;
          return;
        }
        const double ac = AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double s = (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                         (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) +
                         (double)AT(i) * x(i + sz) + (double)AB(i) * x(i - sz);
        const double xn = (b(i) - s) / ac;
        const double d = Kokkos::fabs(xn - x(i));
        if (d > m)
          m = d;
        x(i) = xn;
      },
      Kokkos::Max<double>(du));
  return du;
}

// Box-restricted variant for the distributed overlap smoother: sweeps [rlo,rhi) only, skipping
// [slo,shi) (pass slo==shi to skip nothing). Interior-then-shell split of one colour's sweep is
// bit-identical to the full sweep — a colour's cells never read same-colour cells (see
// cutcellSmoothColorBox in mac_pressure.hpp).
inline void ibmRbgsStencilColorBox(CCField x, CCConst b, MConst AC, MConst AW, MConst AE,
                                   MConst AS, MConst AN, MConst AB, MConst AT, CCConst solidmask,
                                   C3 ext, C3 og, int color, C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  CCExec space;
  const bool hasMask = (solidmask.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_rbgs_box", MD(space, {rlo.x, rlo.y, rlo.z}, {rhi.x, rhi.y, rhi.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
        const long i = (long)lx + (long)ly * ext.x + (long)lz * sz;
        if (hasMask && solidmask(i) > 0.5) {
          x(i) = 0.0;
          return;
        }
        const double ac = AC(i);
        if (Kokkos::fabs(ac) < 1e-30)
          return;
        const double s = (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                         (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) +
                         (double)AT(i) * x(i + sz) + (double)AB(i) * x(i - sz);
        x(i) = (b(i) - s) / ac;
      });
}

inline void ibmRbgsSweep(CCField x, CCConst b, MConst AC, MConst AW, MConst AE, MConst AS,
                         MConst AN, MConst AB, MConst AT, CCConst solidmask, C3 ext, C3 og, int g) {
  ibmRbgsStencilColor(x, b, AC, AW, AE, AS, AN, AB, AT, solidmask, ext, og, g, 0);
  ibmRbgsStencilColor(x, b, AC, AW, AE, AS, AN, AB, AT, solidmask, ext, og, g, 1);
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_IBM_HPP
