// Rung-3 relocation oracle: evaluate dem's SDF surface over a large deterministic point cloud and
// dump the RAW BITS. Run before and after the port; the files must be byte-identical. This is a
// far sharper gate than the pass/fail ctests, which carry tolerances.
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "dem_portable.hpp"
#include "narrowphase.hpp"

using namespace peclet::dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  std::vector<std::uint32_t> bits;
  {
    // deterministic point cloud + parameter sweep
    std::uint32_t st = 0xC0FFEEu;
    auto u = [&](float lo, float hi) {
      st = st * 1664525u + 1013904223u;
      const float t = (float)((st >> 8) & 0xFFFFFFu) / (float)0x1000000u;
      return lo + (hi - lo) * t;
    };

    // --- analytic leaves, swept over shapes and points -------------------------------------
    for (int s = 0; s < 40; ++s) {
      const F4 par{u(0.2f, 2.0f), u(0.2f, 2.0f), u(0.05f, 0.6f), 0.0f};
      for (int i = 0; i < 400; ++i) {
        const F3 p{u(-3, 3), u(-3, 3), u(-3, 3)};
        const float vals[4] = {sdfSphere(p, par), sdfBox(p, par), sdfHollowCylinder(p, par),
                               sdfEval(p, (i % 4), par)};
        for (float v : vals) {
          std::uint32_t b;
          std::memcpy(&b, &v, 4);
          bits.push_back(b);
        }
      }
    }

    // --- grid samplers (object + container policies) ----------------------------------------
    const int NX = 9, NY = 7, NZ = 6;
    Kokkos::View<float*, CpMem> pool("pool", (std::size_t)NX * NY * NZ);
    auto hp = Kokkos::create_mirror_view(pool);
    for (int k = 0; k < NZ; ++k)
      for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
          const float x = -1.0f + 0.3f * i, y = -1.0f + 0.3f * j, z = -1.0f + 0.3f * k;
          hp(i + NX * (j + NY * k)) = Kokkos::sqrt(x * x + y * y + z * z) - 0.55f;
        }
    Kokkos::deep_copy(pool, hp);

    ShapeDesc sd{};
    sd.type = SHAPE_GRID_SDF;
    sd.grid.nx = NX;
    sd.grid.ny = NY;
    sd.grid.nz = NZ;
    sd.grid.offset = 0;
    sd.grid.origin = peclet::core::Vec3<float>{-1.0f, -1.0f, -1.0f};
    sd.grid.invSpacing = peclet::core::Vec3<float>{1.0f / 0.3f, 1.0f / 0.3f, 1.0f / 0.3f};
    sd.grid.extension = peclet::core::geom::GridExtension::kObject;

    WallSdf w{};
    w.grid.nx = NX;
    w.grid.ny = NY;
    w.grid.nz = NZ;
    w.grid.offset = 0;
    w.grid.origin = peclet::core::Vec3<float>{-1.0f, -1.0f, -1.0f};
    w.grid.invSpacing = peclet::core::Vec3<float>{1.0f / 0.3f, 1.0f / 0.3f, 1.0f / 0.3f};
    w.grid.extension = peclet::core::geom::GridExtension::kContainer;

    const int NP = 4000;
    Kokkos::View<float*, CpMem> out("out", 2 * NP);
    std::vector<F3> pts(NP);
    for (int i = 0; i < NP; ++i)
      pts[i] = F3{u(-2.5f, 2.0f), u(-2.5f, 2.0f), u(-2.5f, 2.0f)};
    Kokkos::View<F3*, CpMem> dp("dp", NP);
    auto hdp = Kokkos::create_mirror_view(dp);
    for (int i = 0; i < NP; ++i)
      hdp(i) = pts[i];
    Kokkos::deep_copy(dp, hdp);

    Kokkos::parallel_for(
        "cap", Kokkos::RangePolicy<CpExec>(0, NP), KOKKOS_LAMBDA(int i) {
          out(2 * i) = sampleGridSdf(dp(i), sd, pool);
          out(2 * i + 1) = sampleWallSdf(dp(i), w, pool);
        });
    Kokkos::fence();
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    for (int i = 0; i < 2 * NP; ++i) {
      std::uint32_t b;
      float v = ho(i);
      std::memcpy(&b, &v, 4);
      bits.push_back(b);
    }

    // --- sdfEvalShape dispatch (analytic + grid) --------------------------------------------
    Kokkos::View<ShapeDesc*, CpMem> shp("shp", 1);
    auto hs = Kokkos::create_mirror_view(shp);
    hs(0) = sd;
    Kokkos::deep_copy(shp, hs);
    Kokkos::View<float*, CpMem> out2("out2", NP);
    Kokkos::parallel_for(
        "cap2", Kokkos::RangePolicy<CpExec>(0, NP),
        KOKKOS_LAMBDA(int i) { out2(i) = sdfEvalShape(dp(i), shp(0), pool); });
    Kokkos::fence();
    auto ho2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out2);
    for (int i = 0; i < NP; ++i) {
      std::uint32_t b;
      float v = ho2(i);
      std::memcpy(&b, &v, 4);
      bits.push_back(b);
    }
  }
  Kokkos::finalize();

  std::FILE* f = std::fopen(argv[1], "wb");
  std::fwrite(bits.data(), 4, bits.size(), f);
  std::fclose(f);
  std::printf("wrote %zu values to %s\n", bits.size(), argv[1]);
  return 0;
}
