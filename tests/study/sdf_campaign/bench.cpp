// End-to-end: what does the exact-gradient change do to a real dem step? BOX shape (a shell, so
// contact normals are evaluated per shell point per contact) in a dense periodic box.
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>

#include "sim.hpp"
using namespace peclet::dem;
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 400, STEPS = 200;
    const float DT = (argc > 1) ? (float)std::atof(argv[1]) : 2e-4f;
    Simulation s(N + 8);
    s.setDomain(1.f, 1.f, 1.f, true, true, true);
    s.initializeShape(BOX, 0.035f, 0.f, 0.f);
    std::uint32_t st = 0xA11CEu;
    auto u = [&](float lo, float hi) {
      st = st * 1664525u + 1013904223u;
      return lo + (hi - lo) * ((float)((st >> 8) & 0xFFFFFFu) / (float)0x1000000u);
    };
    std::vector<float> pos(3 * N);
    for (auto& v : pos)
      v = u(0.06f, 0.94f);
    s.setPositions(pos);
    s.setGravity(0.f, -9.81f, 0.f);
    s.setDt(DT);
    // broad+narrow phase ONLY, on a FIXED configuration: idempotent, so both builds do exactly
    // the same work and the only difference is how the contact normal is computed.
    for (int k = 0; k < 5; ++k)
      s.computeOverlaps();  // warm-up
    Kokkos::fence();
    Kokkos::Timer t;
    for (int k = 0; k < STEPS; ++k)
      s.computeOverlaps();
    Kokkos::fence();
    const double ms = 1e3 * t.seconds() / STEPS;
    std::printf("  %d boxes, %d narrowphase passes: %.4f ms/pass  (contacts=%d)\n", N, STEPS, ms,
                s.numContacts());
  }
  Kokkos::finalize();
  return 0;
}
