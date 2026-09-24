#include <cmath>
#include <cstdio>
#include <random>

#include "peclet/core/geom/scene_builder.hpp"
#include "peclet/core/geom/scene_query.hpp"
using namespace peclet::core;
using namespace peclet::core::geom;
struct Rng {
  std::mt19937_64 g{20260829ull};
  double u(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }
};
int main() {
  Rng rng;
  // fast-forward the rng exactly as the test does? The test consumed lots before this section.
  // Instead: rebuild the same scene with a FRESH rng — positions differ from the test but the
  // failure mode should reproduce statistically.
  SceneBuilder<double> b;
  const int sph = b.addLeaf(kSphere, {0.06});
  const int shaft = b.addLeaf(kHollowCylinder, {0.03, 0.4, 0.03});
  const int blade =
      b.addLeaf(kBox, {0.12, 0.02, 0.03}, Transform<double>{Vec3<double>{0.13, 0, 0}});
  const int stir = b.addUnion(shaft, blade);
  const int ell = b.addLeaf(kEllipsoid, {0.09, 0.05, 0.04});
  for (int i = 0; i < 40; ++i)
    b.addInstance(sph, Transform<double>{Vec3<double>{rng.u(0, 1), rng.u(0, 1), rng.u(0, 1)}});
  const double a45 = 0.7853981633974483;
  b.addInstance(stir, Transform<double>{Vec3<double>{0.5, 0.5, 0.5},
                                        Quat<double>{0, std::sin(a45), 0, std::cos(a45)}, 1.2});
  b.addInstance(ell, Transform<double>{Vec3<double>{0.2, 0.7, 0.3}});
  const SceneView<double> sv = b.view();
  for (bool periodic : {true, false}) {
    const PeriodicBox<double> box{1.0, 1.0, 1.0, periodic};
    CandidateGrid<double> g = buildSceneCandidateGrid(sv, {0, 0, 0}, {1, 1, 1}, box);
    const auto gv = g.view();
    const double* br = g.instBoundR.data();
    int bad = 0;
    for (int t = 0; t < 60000; ++t) {
      const Vec3<double> p{rng.u(-0.2, 1.2), rng.u(-0.2, 1.2), rng.u(-0.2, 1.2)};
      const double ref = evalScenePeriodic(sv, p, box, br);
      const double got = evalSceneGrid(sv, p, box, gv, br);
      if (got != ref) {
        if (bad < 4) {
          const long bin = gv.binOf(p);
          std::printf("[%s] p=(%.6f,%.6f,%.6f) ref=%.12g got=%.12g bin=%ld list=[",
                      periodic ? "per" : "open", p.x, p.y, p.z, ref, got, bin);
          if (bin >= 0)
            for (int k = gv.offsets[bin]; k < gv.offsets[bin + 1]; ++k)
              std::printf("%d ", gv.items[k]);
          std::printf("]\n");
          // which instance is the true argmin?
          int win = -1;
          double best = 1e300;
          for (int i = 0; i < sv.instanceCount; ++i) {
            const double d = evalInstancePeriodic(sv, i, p, box, br[i]);
            if (d < best) {
              best = d;
              win = i;
            }
          }
          std::printf("        argmin=%d dist=%.12g  boundR[argmin]=%.4f\n", win, best, br[win]);
        }
        ++bad;
      }
    }
    std::printf("%s: %d/60000 mismatches\n", periodic ? "periodic" : "open", bad);
  }
  return 0;
}
