/// @file
/// @brief flow — portable (Kokkos) MAC stencil operators: Red-Black Gauss-Seidel smoothers +
/// divergence.
///
/// Kokkos port of the core stencil kernels in distributed_ns.cuh (diff_k, pois_k, diverg_k): per
/// inner cell, a 7-point stencil on the extended (inner+ghost) block, x-fastest. Red-Black
/// Gauss-Seidel is expressed as two parallel_for passes (color 0 then 1) with the global-parity
/// filter
/// ((x+ogx)+(y+ogy)+(z+ogz))&1 == color — identical to the CUDA RB-GS (within a color the updates
/// are independent, so no data race). Faithful copy of the math. Runs on any Kokkos backend.
#ifndef PECLET_FLOW_MAC_STENCILS_HPP
#define PECLET_FLOW_MAC_STENCILS_HPP

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace peclet::flow {

using SExec = Kokkos::DefaultExecutionSpace;
using SMem = SExec::memory_space;
using SField = Kokkos::View<double*, SMem>;
using SConst = Kokkos::View<const double*, SMem>;

struct I3 {
  int x, y, z;
};

KOKKOS_INLINE_FUNCTION long L3(int x, int y, int z, I3 e) {
  return static_cast<long>(x) + static_cast<long>(y) * e.x +
         static_cast<long>(z) * static_cast<long>(e.x) * e.y;
}

// One Red-Black sweep colour of the implicit-diffusion smoother:
//   c[i] = (b[i] + beta*sum_neighbours) / (Ac + dcorr[i]).  Call for colour 0 then 1.
// Host backends take the line-sweep form (one (y,z) pencil per task, stride-2 x-loop at the
// colour's parity — bit-identical, same-colour cells are independent); device keeps MDRange.
//
// ANISO DISPATCH (doc/anisotropic_metric.md §2, trap 4).  With per-axis viscous coefficients the
// numerator is `bx*(sW+sE) + by*(sS+sN) + bz*(sB+sT)`, which SUMS IN A DIFFERENT ORDER from the
// isotropic `beta*(sW+sE+sS+sN+sB+sT)` and is therefore not bit-identical at bx==by==bz.  The
// isotropic path is the reference: `Aniso == false` runs today's expression LITERALLY, and the
// per-axis body exists only for a genuinely stretched domain.
template <bool Aniso>
inline void diffSmoothColorT(SField c, SConst b, I3 e, I3 og, int g, double bx, double by,
                             double bz, double Ac, int color, SConst dcorr) {
  SExec space;
  const bool hasD = (dcorr.extent(0) != 0);
  if constexpr (std::is_same_v<typename SExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    Kokkos::parallel_for(
        "peclet::flow::diff", Kokkos::RangePolicy<SExec>(space, 0, (long)nyi * nzi),
        KOKKOS_LAMBDA(long t) {
          const int y = g + (int)(t % nyi), z = g + (int)(t / nyi);
          const long sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
          const int P = (color + og.x + og.y + y + og.z + z) & 1;
          for (int x = g + ((P ^ (g & 1)) & 1); x < e.x - g; x += 2) {
            const long i = L3(x, y, z, e);
            // nvcc forbids an extended lambda FIRST-capturing a variable inside an
            // `if constexpr`, so name the coefficients and read the six neighbours here.  Both
            // are exact: the copies are copies and the sums below keep the legacy operand order.
            const double b0 = bx, b1 = by, b2 = bz;
            const double nE = c(i + sx), nW = c(i - sx), nN = c(i + sy), nS = c(i - sy),
                         nT = c(i + sz), nB = c(i - sz);
            double s;
            if constexpr (Aniso)
              s = b0 * (nE + nW) + b1 * (nN + nS) + b2 * (nT + nB);
            else
              s = b0 * (nE + nW + nN + nS + nT + nB);
            c(i) = (b(i) + s) / (Ac + (hasD ? dcorr(i) : 0.0));
          }
        });
    return;
  }
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::diff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color)
          return;
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double b0 = bx, b1 = by, b2 = bz;  // first-capture outside the constexpr-if (nvcc)
        const double nE = c(i + sx), nW = c(i - sx), nN = c(i + sy), nS = c(i - sy),
                     nT = c(i + sz), nB = c(i - sz);
        double s;
        if constexpr (Aniso)
          s = b0 * (nE + nW) + b1 * (nN + nS) + b2 * (nT + nB);
        else
          s = b0 * (nE + nW + nN + nS + nT + nB);
        c(i) = (b(i) + s) / (Ac + (hasD ? dcorr(i) : 0.0));
      });
}

// Per-axis entry point: `aniso` selects the per-axis body, `!aniso` the legacy one (bx is beta).
inline void diffSmoothColor(SField c, SConst b, I3 e, I3 og, int g, double bx, double by, double bz,
                            double Ac, int color, SConst dcorr, bool aniso) {
  if (aniso)
    diffSmoothColorT<true>(c, b, e, og, g, bx, by, bz, Ac, color, dcorr);
  else
    diffSmoothColorT<false>(c, b, e, og, g, bx, by, bz, Ac, color, dcorr);
}

// Isotropic spelling (the legacy signature).
inline void diffSmoothColor(SField c, SConst b, I3 e, I3 og, int g, double beta, double Ac,
                            int color, SConst dcorr) {
  diffSmoothColorT<false>(c, b, e, og, g, beta, beta, beta, Ac, color, dcorr);
}

// Residual r = b - A c of the constant-coefficient folded diffusion operator over the inner
// cells (both colours): A c = (Ac + dcorr) c - beta * sum(neighbours). The residual-based
// momentum stop for the all-fluid domain-BC path (velSweepLoop's `resid` functor).
template <bool Aniso>
inline void diffResidualT(SField r, SConst c, SConst b, I3 e, int g, double bx, double by,
                          double bz, double Ac, SConst dcorr) {
  SExec space;
  const bool hasD = (dcorr.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::diff_resid", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double b0 = bx, b1 = by, b2 = bz;  // first-capture outside the constexpr-if (nvcc)
        const double nE = c(i + sx), nW = c(i - sx), nN = c(i + sy), nS = c(i - sy),
                     nT = c(i + sz), nB = c(i - sz);
        double sum;
        if constexpr (Aniso)
          sum = b0 * (nE + nW) + b1 * (nN + nS) + b2 * (nT + nB);
        else
          sum = b0 * (nE + nW + nN + nS + nT + nB);
        r(i) = b(i) - ((Ac + (hasD ? dcorr(i) : 0.0)) * c(i) - sum);
      });
}

// Per-axis entry point (see diffSmoothColor's ANISO DISPATCH note: the summation order differs).
inline void diffResidual(SField r, SConst c, SConst b, I3 e, int g, double bx, double by, double bz,
                         double Ac, SConst dcorr, bool aniso) {
  if (aniso)
    diffResidualT<true>(r, c, b, e, g, bx, by, bz, Ac, dcorr);
  else
    diffResidualT<false>(r, c, b, e, g, bx, by, bz, Ac, dcorr);
}

// Isotropic spelling (the legacy signature).
inline void diffResidual(SField r, SConst c, SConst b, I3 e, int g, double beta, double Ac,
                         SConst dcorr) {
  diffResidualT<false>(r, c, b, e, g, beta, beta, beta, Ac, dcorr);
}

// diffSmoothColor + fused max|Δ| reduction over the swept colour (see ibmRbgsStencilColorDu):
// runs as the second colour of a sweep when the momentum tolerance stop is active.
template <bool Aniso>
inline double diffSmoothColorDuT(SField c, SConst b, I3 e, I3 og, int g, double bx, double by,
                                 double bz, double Ac, int color, SConst dcorr) {
  SExec space;
  const bool hasD = (dcorr.extent(0) != 0);
  double du = 0.0;
  if constexpr (std::is_same_v<typename SExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    Kokkos::parallel_reduce(
        "peclet::flow::diff_du", Kokkos::RangePolicy<SExec>(space, 0, (long)nyi * nzi),
        KOKKOS_LAMBDA(long t, double& m) {
          const int y = g + (int)(t % nyi), z = g + (int)(t / nyi);
          const long sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
          const int P = (color + og.x + og.y + y + og.z + z) & 1;
          for (int x = g + ((P ^ (g & 1)) & 1); x < e.x - g; x += 2) {
            const long i = L3(x, y, z, e);
            // nvcc forbids an extended lambda FIRST-capturing a variable inside an
            // `if constexpr`, so name the coefficients and read the six neighbours here.  Both
            // are exact: the copies are copies and the sums below keep the legacy operand order.
            const double b0 = bx, b1 = by, b2 = bz;
            const double nE = c(i + sx), nW = c(i - sx), nN = c(i + sy), nS = c(i - sy),
                         nT = c(i + sz), nB = c(i - sz);
            double s;
            if constexpr (Aniso)
              s = b0 * (nE + nW) + b1 * (nN + nS) + b2 * (nT + nB);
            else
              s = b0 * (nE + nW + nN + nS + nT + nB);
            const double cn = (b(i) + s) / (Ac + (hasD ? dcorr(i) : 0.0));
            const double d = Kokkos::fabs(cn - c(i));
            if (d > m)
              m = d;
            c(i) = cn;
          }
        },
        Kokkos::Max<double>(du));
    return du;
  }
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_reduce(
      "peclet::flow::diff_du", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& m) {
        if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color)
          return;
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double b0 = bx, b1 = by, b2 = bz;  // first-capture outside the constexpr-if (nvcc)
        const double nE = c(i + sx), nW = c(i - sx), nN = c(i + sy), nS = c(i - sy),
                     nT = c(i + sz), nB = c(i - sz);
        double s;
        if constexpr (Aniso)
          s = b0 * (nE + nW) + b1 * (nN + nS) + b2 * (nT + nB);
        else
          s = b0 * (nE + nW + nN + nS + nT + nB);
        const double cn = (b(i) + s) / (Ac + (hasD ? dcorr(i) : 0.0));
        const double d = Kokkos::fabs(cn - c(i));
        if (d > m)
          m = d;
        c(i) = cn;
      },
      Kokkos::Max<double>(du));
  return du;
}

// Per-axis entry point (see diffSmoothColor's ANISO DISPATCH note).
inline double diffSmoothColorDu(SField c, SConst b, I3 e, I3 og, int g, double bx, double by,
                                double bz, double Ac, int color, SConst dcorr, bool aniso) {
  return aniso ? diffSmoothColorDuT<true>(c, b, e, og, g, bx, by, bz, Ac, color, dcorr)
               : diffSmoothColorDuT<false>(c, b, e, og, g, bx, by, bz, Ac, color, dcorr);
}

// Isotropic spelling (the legacy signature).
inline double diffSmoothColorDu(SField c, SConst b, I3 e, I3 og, int g, double beta, double Ac,
                                int color, SConst dcorr) {
  return diffSmoothColorDuT<false>(c, b, e, og, g, beta, beta, beta, Ac, color, dcorr);
}

// One Red-Black sweep colour of the (unit-coefficient) Poisson smoother: phi[i] = (sum - d[i]) / 6.
inline void poisSmoothColor(SField phi, SConst d, I3 e, I3 og, int g, int color) {
  SExec space;
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::pois", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color)
          return;
        const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const double s =
            phi(i + sx) + phi(i - sx) + phi(i + sy) + phi(i - sy) + phi(i + sz) + phi(i - sz);
        phi(i) = (s - d(i)) / 6.0;
      });
}

// MAC divergence d[i] = (u[i+sx]-u[i]) + (v[i+sy]-v[i]) + (w[i+sz]-w[i]) over inner cells.
inline void divergence(SConst u, SConst v, SConst w, SField d, I3 e, int g) {
  SExec space;
  using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::diverg", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
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

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_STENCILS_HPP
