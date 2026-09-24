// Deterministic single-shape trajectory dump: proves the Layer-1 shape-registry refactor left the
// existing (shapeId == 0) path bit-identical. Uses ONLY the pre-Layer-1 API so it compiles against
// both revisions.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sim.hpp"
using namespace peclet::dem;
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  std::vector<std::uint32_t> bits;
  {
    for (int shape : {(int)SPHERE, (int)BOX, (int)HOLLOW_CYLINDER}) {
      Simulation s(128);
      s.setDomain(1.f, 1.f, 1.f, true, true, true);
      s.initializeShape(shape, 0.045f, 0.08f, 0.02f);
      std::uint32_t st = 0x5EED01u;
      auto u = [&](float lo, float hi) {
        st = st * 1664525u + 1013904223u;
        return lo + (hi - lo) * ((float)((st >> 8) & 0xFFFFFFu) / (float)0x1000000u);
      };
      std::vector<float> pos(3 * 60);
      for (auto& v : pos)
        v = u(0.15f, 0.85f);
      s.setPositions(pos);
      s.setGravity(0.f, -9.81f, 0.f);
      s.setDt(2e-4f);
      const int NSTEP = std::atoi(argv[2]);
      for (int k = 0; k < NSTEP; ++k)
        s.step(2e-4f);
      s.computeOverlaps();  // narrow-phase ONLY on the given configuration -- no dynamics
      std::printf("  shape=%d  static overlap probe: contacts=%d maxOverlap=%.9g\n", shape,
                  s.numContacts(), (double)s.maxOverlap());
      for (const std::vector<float>& arr : {s.getPositions(), s.getVelocities(), s.getInvInertia()})
        for (float v : arr) {
          std::uint32_t b;
          std::memcpy(&b, &v, 4);
          bits.push_back(b);
        }
    }
  }
  Kokkos::finalize();
  std::FILE* f = std::fopen(argv[1], "wb");
  std::fwrite(bits.data(), 4, bits.size(), f);
  std::fclose(f);
  std::printf("wrote %zu values\n", bits.size());
  return 0;
}
