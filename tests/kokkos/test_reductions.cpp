// Correctness of the Kokkos MAC reductions (peclet::flow::localSumMax / localDot / subtractAll) over the
// inner cells of an extended (inner+ghost) block, vs a host reference. Validates the parallel_reduce
// indexing (x-fastest, +ghost) and the mean-subtraction. Runs on whatever backend Kokkos was built
// for (CUDA locally; OpenMP for CI). The MPI_Allreduce wrapper is host MPI and unchanged.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mac_reductions.hpp"

using namespace peclet::flow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int ghost = 2;
    Ext3 inner{20, 16, 12};
    Ext3 ext{inner.x + 2 * ghost, inner.y + 2 * ghost, inner.z + 2 * ghost};
    const std::size_t n = static_cast<std::size_t>(ext.x) * ext.y * ext.z;

    std::mt19937 rng(13);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::vector<double> ha(n), hb(n);
    for (std::size_t i = 0; i < n; ++i) { ha[i] = uf(rng); hb[i] = uf(rng); }

    DField a("a", n), b("b", n);
    { auto h = Kokkos::create_mirror_view(a); for (std::size_t i=0;i<n;++i) h(i)=ha[i]; Kokkos::deep_copy(a,h); }
    { auto h = Kokkos::create_mirror_view(b); for (std::size_t i=0;i<n;++i) h(i)=hb[i]; Kokkos::deep_copy(b,h); }

    // device
    SumMax sm = localSumMax(a, ext, ghost, inner);
    double dot = localDot(a, b, ext, ghost, inner);

    // host reference (same inner-cell traversal)
    double rsum = 0.0, rmax = 0.0, rdot = 0.0;
    long count = 0;
    for (int iz = 0; iz < inner.z; ++iz)
      for (int iy = 0; iy < inner.y; ++iy)
        for (int ix = 0; ix < inner.x; ++ix) {
          std::size_t idx = (std::size_t)(ix+ghost) + (std::size_t)(iy+ghost)*ext.x +
                            (std::size_t)(iz+ghost)*(std::size_t)ext.x*ext.y;
          rsum += ha[idx];
          rmax = std::fmax(rmax, std::fabs(ha[idx]));
          rdot += ha[idx]*hb[idx];
          ++count;
        }

    auto close = [](double x, double y) { return std::fabs(x - y) <= 1e-9 * (1.0 + std::fabs(y)); };
    if (!close(sm.sum, rsum) || !close(sm.maxabs, rmax) || !close(dot, rdot)) {
      std::fprintf(stderr, "FAIL: sum %.10g/%.10g  max %.10g/%.10g  dot %.10g/%.10g\n",
                   sm.sum, rsum, sm.maxabs, rmax, dot, rdot);
      status = 1;
    }

    // mean-subtraction over the WHOLE extended block, then re-sum inner cells.
    const double mean = rsum / static_cast<double>(count);
    subtractAll(a, mean);
    SumMax sm2 = localSumMax(a, ext, ghost, inner);
    const double expect2 = rsum - mean * static_cast<double>(count);
    if (!close(sm2.sum, expect2)) {
      std::fprintf(stderr, "FAIL: post-subtract inner sum %.10g != %.10g\n", sm2.sum, expect2);
      status = 1;
    }

    if (!status)
      std::printf("[mac_reductions] PASS: sum/max/dot + mean-subtract match host (%ld inner cells, exec: %s)\n",
                  count, Exec::name());
  }
  Kokkos::finalize();
  return status;
}
