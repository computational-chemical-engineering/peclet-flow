// cfd-gpu — the assembled multi-rank COLLOCATED solver step (collocated plan phase 5c).
//
// The collocated counterpart of test_sdflow_mpi: solves creeping (Stokes) flow through a periodic 2x2x2
// sphere packing with Solver<Colocated>, two ways -- single-rank on the full grid, and distributed
// (each rank constructs the solver with its ORB block dims, calls initMpi, setSolid with its LOCAL SDF
// block). The collocated approximate (MAC) projection runs multi-rank on exactly the same transport-core
// halo machinery as the staggered solver: the cell-velocity halo feeds centerToFace, the projected face
// field is halo-exchanged, the cut-cell pressure MG is MPI-folded. The superficial velocity <u> (hence the
// permeability k = mu*<u>/F) is reduced over ranks and must equal single-rank: np=1 bit-exact, np>1 to the
// MG-PCG reduction-order floor (the inner-product Allreduce reorders the Krylov path). Build with -DCFD_MPI.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "flow_ibm.hpp"

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;

static constexpr int N = 32, STEPS = 120;
static constexpr double RHO = 1.0, MU = 0.1, F = 1e-3, DT = 60.0;

// global sphere-packing SDF (flat x-fastest, negative inside), 2x2x2 spheres, periodic min-distance.
static std::vector<double> packingSdf(double rfrac = 0.18) {
  const double R = rfrac * N;
  std::vector<double> sdf((std::size_t)N * N * N);
  const double cs[2] = {0.25 * N, 0.75 * N};
  for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
    double best = 1e30;
    for (double sx : cs) for (double sy : cs) for (double sz : cs) {
      auto wrap = [](double d) { return d - N * std::round(d / N); };
      const double dx = wrap(x - sx), dy = wrap(y - sy), dz = wrap(z - sz);
      best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - R);
    }
    sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = best;
  }
  return sdf;
}

static void configure(Colo& s) {
  s.setRho(RHO); s.setMu(MU); s.setDt(DT); s.setBodyForce(F, 0, 0);
  s.setAdvection(false); s.setVelocityIterations(80); s.setPressureLevels(4);
  s.setPressurePcg(true, 200, 1e-9);
}

static double localUSum(Colo& s) {
  auto u = s.getVelocity(0); double sum = 0; for (double v : u) sum += v; return sum;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<double> gsdf = packingSdf();
    const double gcells = (double)N * N * N;

    // --- distributed solve ---
    peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{N, N, N});
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
    for (int z = 0; z < lnz; ++z) for (int y = 0; y < lny; ++y) for (int x = 0; x < lnx; ++x)
      lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
          gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];

    Colo sd(lnx, lny, lnz);
    sd.initMpi(N, N, N, MPI_COMM_WORLD);
    configure(sd);
    sd.setSolid(lsdf, /*cutcell_pressure=*/true);
    for (int it = 0; it < STEPS; ++it) sd.step();
    double lsum = localUSum(sd), gsum = 0;
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double k_dist = MU * (gsum / gcells) / F;
    const double div_dist = sd.maxOpenDivergence();

    // --- single-rank reference (full grid) on rank 0 ---
    double k_ref = 0.0;
    if (rank == 0) {
      Colo ref(N, N, N);
      configure(ref);
      ref.setSolid(gsdf, true);
      for (int it = 0; it < STEPS; ++it) ref.step();
      double rsum = 0; { auto u = ref.getVelocity(0); for (double v : u) rsum += v; }
      k_ref = MU * (rsum / gcells) / F;
    }
    MPI_Bcast(&k_ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double reld = std::fabs(k_dist - k_ref) / (std::fabs(k_ref) + 1e-30);
    const double tol = (size == 1) ? 1e-12 : 2e-5;  // np=1 bit-exact; np>1 the MG-PCG reduction-order floor
    if (rank == 0) std::printf("  k_dist=%.8e  k_ref=%.8e  rel=%.2e  div=%.2e  (np=%d, tol %.0e)\n",
                               k_dist, k_ref, reld, div_dist, size, tol);
    if (reld > tol || !(div_dist < 1e-5)) fail = 1;
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0) std::printf("OK (np=%d): distributed collocated Stokes permeability == single-rank\n", size);
    else std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
