// cfd-gpu — portable (Kokkos) cut-cell pressure-operator face openness from an SDF.
//
// Kokkos port of mac_cutcell.cuh: the gradient-normalised masked fluid fraction (cc_fraction_core)
// + trilinear SDF sampling, producing the staggered face openness ox/oy/oz (ox[i] = openness of the
// -x face of cell i) over the extended (inner+ghost) block. Faithful copy of the fraction math.
// KOKKOS_INLINE_FUNCTION so it is shared with the host reference. Runs on any Kokkos backend.
#ifndef CFD_MAC_CUTCELL_KOKKOS_HPP
#define CFD_MAC_CUTCELL_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace cfdk {

using CCExec = Kokkos::DefaultExecutionSpace;
using CCMem = CCExec::memory_space;
using CCField = Kokkos::View<double*, CCMem>;
using CCConst = Kokkos::View<const double*, CCMem>;

struct C3 {
  int x, y, z;
};

// Masked fluid fraction of a face from its SDF samples (centre + 6 axis neighbours). type 1/2/3 =
// x/y/z face. (sd<=0 => closed.) Verbatim from cc_fraction_core.
KOKKOS_INLINE_FUNCTION double ccFractionCore(double sd, double sxp, double sxm, double syp, double sym,
                                             double szp, double szm, int type, double dx, double dy,
                                             double dz) {
  if (sd <= 0.0) return 0.0;
  const double gx = (sxp - sxm) / (2.0 * dx), gy = (syp - sym) / (2.0 * dy), gz = (szp - szm) / (2.0 * dz);
  double gmag = Kokkos::sqrt(gx * gx + gy * gy + gz * gz);
  if (gmag < 1e-6) gmag = 1e-6;
  const double nx = gx / gmag, ny = gy / gmag, nz = gz / gmag;
  double denom = (type == 1) ? (Kokkos::fabs(ny) * dy + Kokkos::fabs(nz) * dz)
                : (type == 2) ? (Kokkos::fabs(nx) * dx + Kokkos::fabs(nz) * dz)
                              : (Kokkos::fabs(nx) * dx + Kokkos::fabs(ny) * dy);
  if (denom < 1e-9) denom = 1e-9;
  double frac = 0.5 + sd / denom;
  if (frac < 0.0) frac = 0.0;
  if (frac > 1.0) frac = 1.0;
  return frac;
}

KOKKOS_INLINE_FUNCTION double ccSampleExt(CCConst sdf, C3 ext, double x, double y, double z) {
  const double fx = Kokkos::floor(x), fy = Kokkos::floor(y), fz = Kokkos::floor(z);
  const double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  const int x1 = cl(x0 + 1, ext.x), y1 = cl(y0 + 1, ext.y), z1 = cl(z0 + 1, ext.z);
  x0 = cl(x0, ext.x); y0 = cl(y0, ext.y); z0 = cl(z0, ext.z);
  const long sy = ext.x, sz = static_cast<long>(ext.x) * ext.y;
  auto F = [&](int xx, int yy, int zz) { return sdf(static_cast<long>(xx) + static_cast<long>(yy) * sy + static_cast<long>(zz) * sz); };
  const double c00 = F(x0, y0, z0) * (1 - wx) + F(x1, y0, z0) * wx;
  const double c10 = F(x0, y1, z0) * (1 - wx) + F(x1, y1, z0) * wx;
  const double c01 = F(x0, y0, z1) * (1 - wx) + F(x1, y0, z1) * wx;
  const double c11 = F(x0, y1, z1) * (1 - wx) + F(x1, y1, z1) * wx;
  const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

KOKKOS_INLINE_FUNCTION double ccFaceOpen(CCConst sdf, C3 ext, double fx, double fy, double fz, int type,
                                         double dx, double dy, double dz) {
  const double sd = ccSampleExt(sdf, ext, fx, fy, fz);
  if (sd <= 0.0) return 0.0;
  const double e = 1.0;
  return ccFractionCore(sd, ccSampleExt(sdf, ext, fx + e, fy, fz), ccSampleExt(sdf, ext, fx - e, fy, fz),
                        ccSampleExt(sdf, ext, fx, fy + e, fz), ccSampleExt(sdf, ext, fx, fy - e, fz),
                        ccSampleExt(sdf, ext, fx, fy, fz + e), ccSampleExt(sdf, ext, fx, fy, fz - e),
                        type, dx, dy, dz);
}

// Fill staggered face openness over the whole extended block (ox[i] = -x face of cell i, etc.).
inline void buildOpenness(CCField ox, CCField oy, CCField oz, CCConst sdf, C3 ext, double dx, double dy,
                          double dz) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "cfdk::cc_open", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = static_cast<long>(lx) + static_cast<long>(ly) * ext.x +
                       static_cast<long>(lz) * static_cast<long>(ext.x) * ext.y;
        ox(i) = ccFaceOpen(sdf, ext, lx - 0.5, ly, lz, 1, dx, dy, dz);
        oy(i) = ccFaceOpen(sdf, ext, lx, ly - 0.5, lz, 2, dx, dy, dz);
        oz(i) = ccFaceOpen(sdf, ext, lx, ly, lz - 0.5, 3, dx, dy, dz);
      });

}

}  // namespace cfdk

#endif  // CFD_MAC_CUTCELL_KOKKOS_HPP
