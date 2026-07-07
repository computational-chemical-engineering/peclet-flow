// Variable-viscosity diffusion stencil (ibmBuildDiffusionVar). Verifies the assembled 7-band
// operator against a host oracle for UniformFaceProps (constant equivalence) and FieldFaceProps
// with arithmetic and harmonic face means at a viscosity jump.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "cut_cell_ibm.hpp"
#include "face_props.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

using peclet::flow::CCConst;
using peclet::flow::CCField;
using FV = Kokkos::View<float*, peclet::flow::IMem>;

float at(const FV& v, long i) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return h(i);
}

void run() {
  const int ex = 8, ey = 6, ez = 6, g = 2;
  const long n = (long)ex * ey * ez;
  const long sx = 1, sy = ex, sz = (long)ex * ey;
  FV AC("AC", n), AW("AW", n), AE("AE", n), AS("AS", n), AN("AN", n), AB("AB", n), AT("AT", n);

  // --- UniformFaceProps == constant operator: A_C = idiag + 6*beta, off = -beta.
  const double idiag = 0.1, beta = 0.5;
  peclet::flow::ibmBuildDiffusionVar(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, g,
                                     peclet::flow::UniformFaceProps{idiag, beta});
  const long i0 = (long)3 + (long)3 * sy + (long)3 * sz;  // an interior cell
  CHECK(std::fabs(at(AW, i0) - (float)(-beta)) < 1e-6);
  CHECK(std::fabs(at(AC, i0) - (float)(idiag + beta + beta + beta + beta + beta + beta)) < 1e-6);

  // --- FieldFaceProps with a viscosity jump in x at the face between cell 3 and 4: mu=1 for x<=3,
  // mu=0.1 for x>=4. Check the -x band of cell 4 (face between 3 and 4) under both means.
  CCField mu("mu", n);
  {
    auto h = Kokkos::create_mirror_view(mu);
    for (int z = 0; z < ez; ++z)
      for (int y = 0; y < ey; ++y)
        for (int x = 0; x < ex; ++x)
          h((long)x + (long)y * sy + (long)z * sz) = (x <= 3) ? 1.0 : 0.1;
    Kokkos::deep_copy(mu, h);
  }
  const long i4 = (long)4 + (long)3 * sy + (long)3 * sz;
  // arithmetic: -x face beta = 0.5*(mu(4)+mu(3)) = 0.5*(0.1+1.0) = 0.55
  peclet::flow::ibmBuildDiffusionVar(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, g,
                                     peclet::flow::FieldFaceProps{CCConst(mu), idiag, false});
  CHECK(std::fabs(at(AW, i4) - (float)(-0.55)) < 1e-6);
  // harmonic: 2*mu(4)*mu(3)/(mu(4)+mu(3)) = 2*0.1*1.0/1.1 = 0.181818
  peclet::flow::ibmBuildDiffusionVar(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, g,
                                     peclet::flow::FieldFaceProps{CCConst(mu), idiag, true});
  const float harm = (float)(2.0 * 0.1 * 1.0 / 1.1);
  CHECK(std::fabs(at(AW, i4) - (-harm)) < 1e-6);
  // +x face of cell 4 is within the mu=0.1 layer: beta = 0.1 (both means agree)
  CHECK(std::fabs(at(AE, i4) - (float)(-0.1)) < 1e-6);
  std::printf("variable-mu bands: uniform + arithmetic(-0.55) + harmonic(%.4f) vs oracle OK\n",
              harm);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
