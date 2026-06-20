// cfd-gpu — portable (Kokkos) global reductions over a MAC grid's inner (owned) cells.
//
// Kokkos port of the local reduction kernels in mac_reductions.cuh (reduce_block_k / dot_block_k /
// subtract_k): sum, max|.|, inner-product, and mean-subtraction over the INNER cells of an extended
// (inner+ghost) block, x-fastest layout. The block-shared-memory + per-block-atomic CUDA reduction
// becomes a single Kokkos::parallel_reduce. The MPI_Allreduce that turns these local results global
// is unchanged (host MPI) and lives in the caller. Runs on any Kokkos backend.
#ifndef CFD_MAC_REDUCTIONS_HPP
#define CFD_MAC_REDUCTIONS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>

namespace sdflow {

using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;
using DField = Kokkos::View<double*, Mem>;       // flat extended-block cell field (x-fastest)
using DConst = Kokkos::View<const double*, Mem>;

struct Ext3 {
  int x, y, z;
};

/// {sum, max|.|} reduction value.
struct SumMax {
  double sum = 0.0;
  double maxabs = 0.0;
  KOKKOS_INLINE_FUNCTION SumMax& operator+=(const SumMax& o) {
    sum += o.sum;
    maxabs = (o.maxabs > maxabs) ? o.maxabs : maxabs;
    return *this;
  }
};

// Map an inner-cell linear index c to the extended-block flat index (x-fastest, +ghost offset).
KOKKOS_INLINE_FUNCTION std::size_t innerToExt(long c, Ext3 ext, int ghost, Ext3 inner) {
  const int ix = static_cast<int>(c % inner.x);
  const int iy = static_cast<int>((c / inner.x) % inner.y);
  const int iz = static_cast<int>(c / (static_cast<long>(inner.x) * inner.y));
  return static_cast<std::size_t>(ix + ghost) +
         static_cast<std::size_t>(iy + ghost) * ext.x +
         static_cast<std::size_t>(iz + ghost) * static_cast<std::size_t>(ext.x) * ext.y;
}

/// Local sum and max|.| over the inner cells.
inline SumMax localSumMax(DConst f, Ext3 ext, int ghost, Ext3 inner) {
  const long n = static_cast<long>(inner.x) * inner.y * inner.z;
  SumMax r;
  if (n <= 0) return r;
  Kokkos::parallel_reduce(
      "sdflow::sum_max", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(long c, SumMax& acc) {
        const double v = f(innerToExt(c, ext, ghost, inner));
        acc.sum += v;
        const double a = Kokkos::fabs(v);
        if (a > acc.maxabs) acc.maxabs = a;
      },
      r);
  return r;
}

/// Local inner product <a,b> over the inner cells.
inline double localDot(DConst a, DConst b, Ext3 ext, int ghost, Ext3 inner) {
  const long n = static_cast<long>(inner.x) * inner.y * inner.z;
  double s = 0.0;
  if (n <= 0) return s;
  Kokkos::parallel_reduce(
      "sdflow::dot", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(long c, double& acc) {
        const std::size_t i = innerToExt(c, ext, ghost, inner);
        acc += a(i) * b(i);
      },
      s);
  return s;
}

/// Subtract a constant from EVERY cell of the extended block (the mean-removal scatter).
inline void subtractAll(DField f, double m) {
  const std::size_t n = f.extent(0);
  Kokkos::parallel_for(
      "sdflow::subtract", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(std::size_t i) { f(i) -= m; });
}

}  // namespace sdflow

#endif  // CFD_MAC_REDUCTIONS_HPP
