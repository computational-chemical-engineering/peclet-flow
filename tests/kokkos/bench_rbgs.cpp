// Throughput benchmark of the Kokkos RB-GS Poisson sweep (the dominant solver kernel), for the
// Kokkos-vs-native-CUDA efficiency comparison. Times many full Red-Black sweeps on a fixed grid and
// reports ns/sweep + effective bandwidth. Pair with tests/cuda_bench/bench_rbgs.cu (identical kernel,
// plain nvcc) run on the same GPU.
#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>

#include "mac_stencils.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int K = (argc > 2) ? std::atoi(argv[2]) : 200;
    const int g = 1;
    peclet::flow::I3 e{N + 2 * g, N + 2 * g, N + 2 * g}, og{0, 0, 0};
    const std::size_t n = (std::size_t)e.x * e.y * e.z;
    peclet::flow::SField phi("phi", n), d("d", n);
    Kokkos::deep_copy(phi, 1.0);
    Kokkos::deep_copy(d, 0.5);

    // warmup
    for (int i = 0; i < 10; ++i) peclet::flow::poisSweep(phi, peclet::flow::SConst(d), e, og, g);
    Kokkos::fence();

    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      Kokkos::Timer t;
      for (int i = 0; i < K; ++i) peclet::flow::poisSweep(phi, peclet::flow::SConst(d), e, og, g);
      Kokkos::fence();
      double ms = t.seconds() * 1e3;
      if (ms < best) best = ms;
    }
    const double ns_per_sweep = best * 1e6 / K;
    const double cells = (double)N * N * N;
    const double ns_per_cell = ns_per_sweep / cells;
    const double gbps = cells * 64.0 / (ns_per_sweep);  // 64 B/cell (6 nbr + d + write), bytes/ns = GB/s
    std::printf("KOKKOS  N=%d sweeps=%d : %.3f ns/sweep, %.4f ns/cell, %.1f GB/s (exec %s)\n",
                N, K, ns_per_sweep, ns_per_cell, gbps, Kokkos::DefaultExecutionSpace::name());
  }
  Kokkos::finalize();
  return 0;
}
