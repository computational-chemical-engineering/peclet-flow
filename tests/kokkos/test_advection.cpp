// Correctness of the Kokkos staggered advection operator (sadv::advect / advect_fou) vs a host
// replication using the same templated functions. Random U,V,W,PHI on an extended (inner+ghost=2)
// block; apply the conservative Koren-TVD (and FOU) advection per inner cell on the device and on
// the host, require a match (per-cell deterministic double math). Runs on whatever backend Kokkos
// has.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "staggered_advection.hpp"

using namespace sadv;
using Mem = Kokkos::DefaultExecutionSpace::memory_space;
using DView = Kokkos::View<double*, Mem>;
using CView = Kokkos::View<const double*, Mem>;

// Host accessor mirroring ViewAcc (same indexing) for the reference.
struct HostAcc {
  const double* d;
  int ex, ey;
  double operator()(int x, int y, int z) const {
    return d[(long)x + (long)y * ex + (long)z * (long)ex * ey];
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int ghost = 2;
    const int ix0 = ghost, iy0 = ghost, iz0 = ghost;
    const int inx = 16, iny = 12, inz = 10;
    const int ex = inx + 2 * ghost, ey = iny + 2 * ghost, ez = inz + 2 * ghost;
    const std::size_t n = (std::size_t)ex * ey * ez;

    std::mt19937 rng(5);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> hU(n), hV(n), hW(n), hP(n);
    for (std::size_t i = 0; i < n; ++i) {
      hU[i] = uf(rng);
      hV[i] = uf(rng);
      hW[i] = uf(rng);
      hP[i] = uf(rng);
    }

    auto up = [&](const char* nm, std::vector<double>& h) {
      DView v(nm, n);
      auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i)
        m(i) = h[i];
      Kokkos::deep_copy(v, m);
      return v;
    };
    DView U = up("U", hU), V = up("V", hV), W = up("W", hW), P = up("P", hP);

    const long ninner = (long)inx * iny * inz;
    DView outTVD("outTVD", ninner), outFOU("outFOU", ninner);
    const int comp = 0;
    Kokkos::parallel_for(
        "advect", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, ninner),
        KOKKOS_LAMBDA(long c) {
          const int lx = (int)(c % inx), ly = (int)((c / inx) % iny),
                    lz = (int)(c / ((long)inx * iny));
          const int x = lx + ix0, y = ly + iy0, z = lz + iz0;
          ViewAcc Ua{CView(U), ex, ey}, Va{CView(V), ex, ey}, Wa{CView(W), ex, ey},
              Pa{CView(P), ex, ey};
          outTVD(c) = advect(comp, x, y, z, Ua, Va, Wa, Pa);
          outFOU(c) = advect_fou(comp, x, y, z, Ua, Va, Wa, Pa);
        });

    auto hT = Kokkos::create_mirror_view(outTVD);
    Kokkos::deep_copy(hT, outTVD);
    auto hF = Kokkos::create_mirror_view(outFOU);
    Kokkos::deep_copy(hF, outFOU);

    // host reference
    int bad = 0;
    HostAcc Uh{hU.data(), ex, ey}, Vh{hV.data(), ex, ey}, Wh{hW.data(), ex, ey},
        Ph{hP.data(), ex, ey};
    auto close = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * (1.0 + std::fabs(b)); };
    for (long c = 0; c < ninner; ++c) {
      const int lx = (int)(c % inx), ly = (int)((c / inx) % iny), lz = (int)(c / ((long)inx * iny));
      const int x = lx + ix0, y = ly + iy0, z = lz + iz0;
      const double rT = advect(comp, x, y, z, Uh, Vh, Wh, Ph);
      const double rF = advect_fou(comp, x, y, z, Uh, Vh, Wh, Ph);
      if (!close(hT(c), rT) || !close(hF(c), rF))
        ++bad;
    }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d/%ld advection cells differ\n", bad, ninner);
      status = 1;
    } else
      std::printf("[advection] PASS: %ld cells, Koren-TVD + FOU match host (exec: %s)\n", ninner,
                  Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return status;
}
