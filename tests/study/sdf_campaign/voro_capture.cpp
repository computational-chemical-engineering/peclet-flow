// Rung-4 relocation oracle: evaluate voro's SDF providers over a deterministic sweep and dump raw
// bits. Run before and after the port; must be byte-identical. Covers the three analytic providers
// being ported AND SdfGrid/SdfSpheres, which are NOT being ported -- so the capture also proves
// they were left alone.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include "peclet/voro/sdf.hpp"

using namespace peclet::voro;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  std::vector<std::uint32_t> bits;
  {
    const int NP = 3000;
    std::uint32_t st = 0xBEEF01u;
    auto u = [&](double lo, double hi) {
      st = st * 1664525u + 1013904223u;
      return lo + (hi - lo) * ((double)((st >> 8) & 0xFFFFFFu) / (double)0x1000000u);
    };
    std::vector<double> px(NP), py(NP), pz(NP);
    for (int i = 0; i < NP; ++i) { px[i] = u(-3, 3); py[i] = u(-3, 3); pz[i] = u(-3, 3); }
    Kokkos::View<double*> X("x", NP), Y("y", NP), Z("z", NP);
    auto hx = Kokkos::create_mirror_view(X), hy = Kokkos::create_mirror_view(Y),
         hz = Kokkos::create_mirror_view(Z);
    for (int i = 0; i < NP; ++i) { hx(i) = px[i]; hy(i) = py[i]; hz(i) = pz[i]; }
    Kokkos::deep_copy(X, hx); Kokkos::deep_copy(Y, hy); Kokkos::deep_copy(Z, hz);

    // sphere / box / hollow-cylinder sweeps (several parameter sets incl. all three axes)
    const int NS = 12;
    Kokkos::View<double*> out("out", (std::size_t)NP * NS);
    Kokkos::parallel_for("cap", NP, KOKKOS_LAMBDA(int i) {
      const double x = X(i), y = Y(i), z = Z(i);
      int s = 0;
      SdfSphere<double> a{0.1, -0.2, 0.3, 1.25};
      SdfSphere<double> a2{0, 0, 0, 0.5};
      out(i * NS + s++) = a.eval(x, y, z);
      out(i * NS + s++) = a2.eval(x, y, z);
      SdfBox<double> b{0.2, 0.1, -0.3, 0.8, 1.1, 0.5};
      SdfBox<double> b2{0, 0, 0, 0.4, 0.4, 0.4};
      out(i * NS + s++) = b.eval(x, y, z);
      out(i * NS + s++) = b2.eval(x, y, z);
      for (int ax = 0; ax < 3; ++ax) {
        SdfHollowCylinder<double> c{0.05, -0.1, 0.15, 1.3, 0.7, 1.6, ax};
        out(i * NS + s++) = c.eval(x, y, z);
      }
      SdfHollowCylinder<double> c2{0, 0, 0, 1.0, 0.0, 2.0, 2};
      out(i * NS + s++) = c2.eval(x, y, z);
      // gradients through the shared central-difference helper
      double g[3];
      sdfGradient<double>(a, x, y, z, g);
      out(i * NS + s++) = g[0]; out(i * NS + s++) = g[1]; out(i * NS + s++) = g[2];
      out(i * NS + s++) = a.gradH() + b.gradH() + c2.gradH();
    });
    Kokkos::fence();
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    for (std::size_t i = 0; i < ho.extent(0); ++i) {
      std::uint64_t b64; double v = ho(i); std::memcpy(&b64, &v, 8);
      bits.push_back((std::uint32_t)(b64 & 0xFFFFFFFFu));
      bits.push_back((std::uint32_t)(b64 >> 32));
    }

    // SdfSpheres + SdfGrid (NOT ported -- capture proves they are untouched)
    const int M = 24;
    Kokkos::View<double*> cen("cen", 3 * M), rad("rad", M);
    auto hc = Kokkos::create_mirror_view(cen), hr = Kokkos::create_mirror_view(rad);
    for (int i = 0; i < M; ++i) {
      hc(3*i) = u(-1,1); hc(3*i+1) = u(-1,1); hc(3*i+2) = u(-1,1); hr(i) = u(0.1,0.4);
    }
    Kokkos::deep_copy(cen, hc); Kokkos::deep_copy(rad, hr);
    const int G = 8;
    Kokkos::View<float*> gv("gv", (std::size_t)G*G*G);
    auto hg = Kokkos::create_mirror_view(gv);
    for (int k=0;k<G;++k) for (int j=0;j<G;++j) for (int i2=0;i2<G;++i2) {
      const double xx=-1+0.3*i2, yy=-1+0.3*j, zz=-1+0.3*k;
      hg(i2 + G*(j + G*k)) = (float)(std::sqrt(xx*xx+yy*yy+zz*zz) - 0.6);
    }
    Kokkos::deep_copy(gv, hg);
    Kokkos::View<double*> out2("out2", (std::size_t)NP * 2);
    Kokkos::parallel_for("cap2", NP, KOKKOS_LAMBDA(int i) {
      SdfSpheres<double> sp{cen, rad, M, 2.0};
      auto gr = SdfGrid<double>::fromSpacing(
          Kokkos::View<const float*, peclet::core::MemSpace>(gv), G, G, G,
          -1.0, -1.0, -1.0, 0.3, 0.3, 0.3);
      out2(2 * i) = sp.eval(X(i), Y(i), Z(i));
      out2(2 * i + 1) = gr.eval(X(i), Y(i), Z(i));
    });
    Kokkos::fence();
    auto ho2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out2);
    for (std::size_t i = 0; i < ho2.extent(0); ++i) {
      std::uint64_t b64; double v = ho2(i); std::memcpy(&b64, &v, 8);
      bits.push_back((std::uint32_t)(b64 & 0xFFFFFFFFu));
      bits.push_back((std::uint32_t)(b64 >> 32));
    }
  }
  Kokkos::finalize();
  std::FILE* f = std::fopen(argv[1], "wb");
  std::fwrite(bits.data(), 4, bits.size(), f);
  std::fclose(f);
  std::printf("wrote %zu words\n", bits.size());
  return 0;
}
