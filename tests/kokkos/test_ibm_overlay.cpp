// Correctness of the Kokkos Robust-Scaled IBM overlay build (dns::ibmFillEntry) vs a HostSpace
// reference using the same code. Random cut-cell SDF configurations (fluid centre + mixed solid/fluid
// neighbours, including sandwiched double-sided axes), across point-value (SCHEME 0) and cell-average
// (1) and Dirichlet/Neumann. Compares every factor (D_rescale, dir_code, K/M/X/Nbc/R). Runs on any
// Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cut_cell_ibm.hpp"

using namespace dns;
using DSpace = Kokkos::DefaultExecutionSpace;

template <int SCHEME>
static int run(int bc) {
  const int N = 3000;
  std::mt19937 rng(100 + SCHEME * 7 + bc);
  std::uniform_real_distribution<float> uc(0.05f, 0.95f);   // fluid centre
  std::uniform_real_distribution<float> us(-0.95f, -0.05f); // solid neighbour
  std::uniform_real_distribution<float> uf(0.05f, 0.95f);   // fluid neighbour
  std::uniform_real_distribution<float> u01(0.f, 1.f);

  std::vector<float> hc(N), hn((std::size_t)N * 6);
  for (int i = 0; i < N; ++i) {
    hc[i] = uc(rng);
    // force a sandwiched axis on ~1/3 of cells for coverage
    bool sandwich_x = u01(rng) < 0.33f;
    for (int k = 0; k < 6; ++k) {
      bool solid;
      if (sandwich_x && (k == 0 || k == 1)) solid = true;
      else solid = u01(rng) < 0.5f;
      hn[(std::size_t)i * 6 + k] = solid ? us(rng) : uf(rng);
    }
  }

  // device
  Kokkos::View<float*, IMem> dc("dc", N), dn("dn", (std::size_t)N * 6);
  { auto m = Kokkos::create_mirror_view(dc); for (int i=0;i<N;++i) m(i)=hc[i]; Kokkos::deep_copy(dc,m); }
  { auto m = Kokkos::create_mirror_view(dn); for (std::size_t i=0;i<(std::size_t)N*6;++i) m(i)=hn[i]; Kokkos::deep_copy(dn,m); }
  IbmOverlay dev{ Kokkos::View<int*,IMem>("ci",N), Kokkos::View<int*,IMem>("nb",N),
                  Kokkos::View<float*,IMem>("dr",N), Kokkos::View<int*,IMem>("dirc",(std::size_t)N*6),
                  Kokkos::View<float*,IMem>("K",(std::size_t)N*6), Kokkos::View<float*,IMem>("M",(std::size_t)N*6),
                  Kokkos::View<float*,IMem>("X",(std::size_t)N*6), Kokkos::View<float*,IMem>("Nbc",(std::size_t)N*6),
                  Kokkos::View<float*,IMem>("R",(std::size_t)N*6) };
  Kokkos::parallel_for("ibm_build", Kokkos::RangePolicy<DSpace>(0, N), KOKKOS_LAMBDA(int i) {
    float sn[6]; for (int k=0;k<6;++k) sn[k]=dn((std::size_t)i*6+k);
    ibmFillEntry<SCHEME>(dev, i, i * 10, dc(i), sn, bc);
  });
  Kokkos::fence();

  // host reference (HostSpace overlay, same fill code)
  using HOV = IbmOverlayT<Kokkos::HostSpace>;
  HOV h{ Kokkos::View<int*,Kokkos::HostSpace>("hci",N), Kokkos::View<int*,Kokkos::HostSpace>("hnb",N),
         Kokkos::View<float*,Kokkos::HostSpace>("hdr",N), Kokkos::View<int*,Kokkos::HostSpace>("hdirc",(std::size_t)N*6),
         Kokkos::View<float*,Kokkos::HostSpace>("hK",(std::size_t)N*6), Kokkos::View<float*,Kokkos::HostSpace>("hM",(std::size_t)N*6),
         Kokkos::View<float*,Kokkos::HostSpace>("hX",(std::size_t)N*6), Kokkos::View<float*,Kokkos::HostSpace>("hNbc",(std::size_t)N*6),
         Kokkos::View<float*,Kokkos::HostSpace>("hR",(std::size_t)N*6) };
  for (int i = 0; i < N; ++i) {
    float sn[6]; for (int k=0;k<6;++k) sn[k]=hn[(std::size_t)i*6+k];
    ibmFillEntry<SCHEME>(h, i, i * 10, hc[i], sn, bc);
  }

  auto cmp1i = [&](Kokkos::View<int*,IMem> dv, Kokkos::View<int*,Kokkos::HostSpace> hv, std::size_t n) {
    auto m = Kokkos::create_mirror_view(dv); Kokkos::deep_copy(m, dv); int bad=0;
    for (std::size_t i=0;i<n;++i) if (m(i)!=hv(i)) ++bad; return bad; };
  auto cmp1f = [&](Kokkos::View<float*,IMem> dv, Kokkos::View<float*,Kokkos::HostSpace> hv, std::size_t n) {
    auto m = Kokkos::create_mirror_view(dv); Kokkos::deep_copy(m, dv); int bad=0;
    for (std::size_t i=0;i<n;++i) if (std::fabs((double)m(i)-(double)hv(i)) > 1e-5*(1.0+std::fabs((double)hv(i)))) ++bad; return bad; };
  int bad = 0;
  bad += cmp1i(dev.cell_index, h.cell_index, N) + cmp1i(dev.num_boundaries, h.num_boundaries, N) + cmp1i(dev.dir_code, h.dir_code, (std::size_t)N*6);
  bad += cmp1f(dev.D_rescale, h.D_rescale, N);
  bad += cmp1f(dev.K_val, h.K_val, (std::size_t)N*6) + cmp1f(dev.M_val, h.M_val, (std::size_t)N*6) +
         cmp1f(dev.X_val, h.X_val, (std::size_t)N*6) + cmp1f(dev.Nbc_val, h.Nbc_val, (std::size_t)N*6) +
         cmp1f(dev.R_val, h.R_val, (std::size_t)N*6);
  return bad;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    int bad = run<0>(0) + run<1>(0) + run<0>(1) + run<1>(1);
    if (bad) { std::fprintf(stderr, "FAIL: %d overlay-factor mismatches\n", bad); status = 1; }
    else std::printf("[ibm_overlay] PASS: Robust-Scaled overlay (point/avg x Dirichlet/Neumann) matches host (exec: %s)\n",
                     DSpace::name());
  }
  Kokkos::finalize();
  return status;
}
