// cfd-gpu — portable (Kokkos) MAC stencil operators: Red-Black Gauss-Seidel smoothers + divergence.
//
// Kokkos port of the core stencil kernels in distributed_ns.cuh (diff_k, pois_k, diverg_k): per inner
// cell, a 7-point stencil on the extended (inner+ghost) block, x-fastest. Red-Black Gauss-Seidel is
// expressed as two parallel_for passes (color 0 then 1) with the global-parity filter
// ((x+ogx)+(y+ogy)+(z+ogz))&1 == color — identical to the CUDA RB-GS (within a color the updates are
// independent, so no data race). Faithful copy of the math. Runs on any Kokkos backend.
#ifndef CFD_MAC_STENCILS_HPP
#define CFD_MAC_STENCILS_HPP

#include <Kokkos_Core.hpp>

namespace sdflow {

using SExec = Kokkos::DefaultExecutionSpace;
using SMem = SExec::memory_space;
using SField = Kokkos::View<double*, SMem>;
using SConst = Kokkos::View<const double*, SMem>;

struct I3 {
  int x, y, z;
};

KOKKOS_INLINE_FUNCTION long L3(int x, int y, int z, I3 e) {
  return static_cast<long>(x) + static_cast<long>(y) * e.x + static_cast<long>(z) * static_cast<long>(e.x) * e.y;
}

// One Red-Black sweep colour of the implicit-diffusion smoother:
//   c[i] = (b[i] + beta*sum_neighbours) / (Ac + dcorr[i]).  Call for colour 0 then 1.
inline void diffSmoothColor(SField c, SConst b, I3 e, I3 og, int g, double beta, double Ac, int color,
                            SConst dcorr) {
  SExec space;
  const bool hasD = (dcorr.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "sdflow::diff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color) return;
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double s = c(i + sx) + c(i - sx) + c(i + sy) + c(i - sy) + c(i + sz) + c(i - sz);
        c(i) = (b(i) + beta * s) / (Ac + (hasD ? dcorr(i) : 0.0));
      });

}

// One Red-Black sweep colour of the (unit-coefficient) Poisson smoother: phi[i] = (sum - d[i]) / 6.
inline void poisSmoothColor(SField phi, SConst d, I3 e, I3 og, int g, int color) {
  SExec space;
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "sdflow::pois", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color) return;
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double s = phi(i + sx) + phi(i - sx) + phi(i + sy) + phi(i - sy) + phi(i + sz) + phi(i - sz);
        phi(i) = (s - d(i)) / 6.0;
      });

}

// MAC divergence d[i] = (u[i+sx]-u[i]) + (v[i+sy]-v[i]) + (w[i+sz]-w[i]) over inner cells.
inline void divergence(SConst u, SConst v, SConst w, SField d, I3 e, int g) {
  SExec space;
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "sdflow::diverg", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        d(i) = (u(i + sx) - u(i)) + (v(i + sy) - v(i)) + (w(i + sz) - w(i));
      });

}

// Full Red-Black Gauss-Seidel sweep (both colours) of the Poisson smoother.
inline void poisSweep(SField phi, SConst d, I3 e, I3 og, int g) {
  poisSmoothColor(phi, d, e, og, g, 0);
  poisSmoothColor(phi, d, e, og, g, 1);
}

}  // namespace sdflow

#endif  // CFD_MAC_STENCILS_HPP
