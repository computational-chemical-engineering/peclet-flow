/// @file
/// @brief Timing probe of the direct pressure bottom's factor launch (`mg_bottom_factor`,
/// doc/vof_step_performance_design.md §13.4.3, §14 H-1) on the bottom-direct test problems:
/// `walls-y` (bottom 16x12x8, walls on y: P = 12, b = 128 -- the bubble column's bottom) and
/// `periodic` (16x12x8 all periodic: P = 16, b = 96, with the border). Not a correctness gate.
///
///   bench_bottom_factor [reps] [case: walls-y | periodic] [dump file] [T ...]
///
/// Per team size T (default 1 2 4 8 16 24; 0 = the production default): min / median ms over
/// `reps` refactors of the same operator, and whether the factor's storage (Q, Y, e, s, stat) is
/// bitwise identical to the first run's. A host backend times its host schedule, then the team
/// algorithm (the device one, the host schedule's bitwise oracle) at the same sizes. With a dump
/// file the first T's storage is written there (raw bytes, Q|Y|e|s|stat), so two builds can be
/// compared with `cmp`.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

template <class V>
std::vector<char> bytes(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const char* p = reinterpret_cast<const char*>(h.data());
  return std::vector<char>(p, p + h.span() * sizeof(typename V::value_type));
}
std::vector<char> storage(const BottomDirect<float>& D) {
  std::vector<char> out;
  for (const auto& b : {bytes(D.Q), bytes(D.Y), bytes(D.e), bytes(D.s), bytes(D.stat)})
    out.insert(out.end(), b.begin(), b.end());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const std::string cname = argc > 2 ? argv[2] : "walls-y";
    const char* dump = (argc > 3 && std::strcmp(argv[3], "-") != 0) ? argv[3] : nullptr;
    std::vector<int> Ts;
    for (int a = 4; a < argc; ++a)
      Ts.push_back(std::atoi(argv[a]));
    if (Ts.empty())
      Ts = {1, 2, 4, 8, 16, 24};
    int bc[6] = {0, 0, 1, 1, 0, 0};
    if (cname == "periodic")
      bc[2] = bc[3] = 0;
    const int nx = 32, ny = 24, nz = 16;
    CutcellMG mg;
    mg.setBoundaryConditions(bc);
    mg.init(nx, ny, nz, 2);
    mg.setAgglomerationMode(1);
    const C3 e{nx + 2, ny + 2, nz + 2};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;
    std::vector<double> hx(n), hy(n), hz(n);
    const double lr = std::log(50.0), k = 2.0 * M_PI;
    for (int z = 0; z < e.z; ++z)  // test_bottom_direct's openness (ratio 50)
      for (int y = 0; y < e.y; ++y)
        for (int x = 0; x < e.x; ++x) {
          const std::size_t i = x + (std::size_t)y * e.x + (std::size_t)z * e.x * e.y;
          const double s = 0.5 + 0.5 * std::sin(k * x / nx) * std::cos(k * (y + 0.5 * z) / ny);
          const double t = 0.5 + 0.5 * std::cos(k * (x + z) / nz);
          hx[i] = std::exp(lr * s) / 50.0;
          hy[i] = std::exp(lr * t) / 50.0;
          hz[i] = std::exp(lr * (1.0 - s)) / 50.0;
        }
    auto dev = [](const std::vector<double>& v, const char* name) {
      CCField f(name, v.size());
      auto h = Kokkos::create_mirror_view(f);
      for (std::size_t i = 0; i < v.size(); ++i)
        h(i) = v[i];
      Kokkos::deep_copy(f, h);
      return f;
    };
    CCField ox = dev(hx, "ox"), oy = dev(hy, "oy"), oz = dev(hz, "oz");
    mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
    mg.bottomForceForTest();
    const BottomPlanes pl = mg.directPlanes();
    printf("bench_bottom_factor %s: P = %d, b = %d, border %d, %d reps, concurrency %d\n",
           cname.c_str(), pl.P, pl.b, pl.border, reps, (int)CCExec().concurrency());
    std::vector<char> ref;
    const bool host = std::is_same_v<CCMem, Kokkos::HostSpace>;
    for (int alg = 0; alg < (host ? 2 : 1); ++alg)
      for (int T : Ts) {
        const bool team = alg == 1;  // on a host backend: 0 = the host schedule, 1 = the team one
        int Tu = 0;
        BottomDirect<float> D = mg.directFactorForTest<float>(T, kBottomPivotTol, &Tu, team);
        if (T != 0 && Tu != T) {
          printf("  T %3d: not available (ran %d), skipped\n", T, Tu);
          continue;
        }
        for (int w = 0; w < 3; ++w)
          mg.launchDirectFactor(D, Tu, kBottomPivotTol, /*probe=*/false, team);
        Kokkos::fence();
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
          Kokkos::Timer tm;
          mg.launchDirectFactor(D, Tu, kBottomPivotTol, /*probe=*/false, team);
          Kokkos::fence();
          ms.push_back(tm.seconds() * 1e3);
        }
        std::sort(ms.begin(), ms.end());
        const std::vector<char> st = storage(D);
        const char* same = "reference";
        if (ref.empty()) {
          ref = st;
          if (dump) {
            FILE* f = std::fopen(dump, "wb");
            std::fwrite(ref.data(), 1, ref.size(), f);
            std::fclose(f);
          }
        } else {
          same = (st == ref) ? "bitwise identical" : "DIFFERS";
        }
        printf("  %s T %3d: factor min %8.3f ms  median %8.3f ms  max %8.3f ms  (%s)\n",
               host ? (team ? "team" : "host") : "team", Tu, ms.front(), ms[ms.size() / 2],
               ms.back(), same);
      }
  }
  Kokkos::finalize();
  return 0;
}
