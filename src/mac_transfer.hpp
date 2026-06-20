// cfd-gpu — portable (Kokkos) multigrid transfer operators + projection velocity correction.
//
// Kokkos port of mg_restrict_k / mg_prolong_k (mac_multigrid.cuh) and correct_k (distributed_ns.cuh):
// per-axis-ratio averaging restriction, trilinear prolongation (added to the fine solution), and the
// staggered pressure-gradient velocity correction. Faithful copies; per-axis ratio (1 or 2) supports
// semi-coarsening. Runs on any Kokkos backend.
#ifndef CFD_MAC_TRANSFER_HPP
#define CFD_MAC_TRANSFER_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace dns {

using TExec = Kokkos::DefaultExecutionSpace;
using TMem = TExec::memory_space;
using TField = Kokkos::View<double*, TMem>;
using TConst = Kokkos::View<const double*, TMem>;

struct T3 {
  int x, y, z;
};

// Averaging restriction: coarse[ic] = mean of the ratio^3 fine cells it covers.
inline void restrict_(TField coarse, TConst fine, T3 cext, T3 fext, int g, T3 cinner, T3 ratio) {
  TExec space;
  using MD = Kokkos::MDRangePolicy<TExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "dns::restrict", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = static_cast<long>(fext.x) * fext.y;
        double sum = 0.0;
        for (int dz = 0; dz < ratio.z; ++dz)
          for (int dy = 0; dy < ratio.y; ++dy)
            for (int dx = 0; dx < ratio.x; ++dx) {
              const int fx = ratio.x * icx + dx + g, fy = ratio.y * icy + dy + g, fz = ratio.z * icz + dz + g;
              sum += fine(static_cast<long>(fx) + static_cast<long>(fy) * fsy + static_cast<long>(fz) * fsz);
            }
        const long ci = static_cast<long>(icx + g) + static_cast<long>(icy + g) * cext.x +
                        static_cast<long>(icz + g) * static_cast<long>(cext.x) * cext.y;
        coarse(ci) = sum / static_cast<double>(ratio.x * ratio.y * ratio.z);
      });

}

KOKKOS_INLINE_FUNCTION double trilerp(TConst c, double x, double y, double z, T3 cext) {
  const double fx = Kokkos::floor(x), fy = Kokkos::floor(y), fz = Kokkos::floor(z);
  const double wx = x - fx, wy = y - fy, wz = z - fz;
  const int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  const long sy = cext.x, sz = static_cast<long>(cext.x) * cext.y;
  auto F = [&](int xx, int yy, int zz) {
    return c(static_cast<long>(xx) + static_cast<long>(yy) * sy + static_cast<long>(zz) * sz);
  };
  const double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  const double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  const double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  const double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  const double c0 = c00 * (1 - wy) + c10 * wy;
  const double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

// Trilinear prolongation of the coarse correction, ADDED to the fine field (coarse ghost pre-filled).
inline void prolong(TField fine, TConst coarse, T3 fext, T3 cext, int g, T3 finner, T3 ratio) {
  TExec space;
  using MD = Kokkos::MDRangePolicy<TExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "dns::prolong", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        const double cx = ratio.x == 2 ? 0.5 * ifx - 0.25 + g : (double)(ifx + g);
        const double cy = ratio.y == 2 ? 0.5 * ify - 0.25 + g : (double)(ify + g);
        const double cz = ratio.z == 2 ? 0.5 * ifz - 0.25 + g : (double)(ifz + g);
        const long fi = static_cast<long>(ifx + g) + static_cast<long>(ify + g) * fext.x +
                        static_cast<long>(ifz + g) * static_cast<long>(fext.x) * fext.y;
        fine(fi) += trilerp(coarse, cx, cy, cz, cext);
      });

}

// Projection velocity correction: u -= grad(phi) on the staggered faces, over inner cells.
inline void correct(TField u, TField v, TField w, TConst phi, T3 e, int g) {
  TExec space;
  using MD = Kokkos::MDRangePolicy<TExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "dns::correct", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = static_cast<long>(x) + static_cast<long>(y) * e.x +
                       static_cast<long>(z) * static_cast<long>(e.x) * e.y;
        const long sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        u(i) -= phi(i) - phi(i - sx);
        v(i) -= phi(i) - phi(i - sy);
        w(i) -= phi(i) - phi(i - sz);
      });

}

}  // namespace dns

#endif  // CFD_MAC_TRANSFER_HPP
