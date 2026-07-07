// Dynamic load balancing: Solver::redistribute moves the distributed solver's state to a NEW ORB
// partition mid-simulation. Stokes flow through a sphere packing is run distributed on the default
// (equal-cell) decomposition for half the steps, then REDISTRIBUTED to a weighted decomposition
// (block boundaries move) and run to completion. The permeability must still match the single-rank
// reference to the same tolerance as a never-redistributed distributed run — proving the migration
// (bit-exact field movement) + the rebuild (halo / openness / IBM / MG from the migrated SDF) are
// correct. Build with -DPECLET_FLOW_MPI.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 120;
static constexpr double RHO = 1.0, MU = 0.1, F = 1e-3, DT = 60.0;

static std::vector<double> packingSdf(double rfrac = 0.18) {
  const double R = rfrac * N;
  std::vector<double> sdf((std::size_t)N * N * N);
  const double cs[2] = {0.25 * N, 0.75 * N};
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double best = 1e30;
        for (double sx : cs)
          for (double sy : cs)
            for (double sz : cs) {
              auto wrap = [](double d) { return d - N * std::round(d / N); };
              const double dx = wrap(x - sx), dy = wrap(y - sy), dz = wrap(z - sz);
              best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - R);
            }
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = best;
      }
  return sdf;
}

static void configure(IbmSolver& s) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0, 0);
  s.setAdvection(false);
  s.setVelocityIterations(80);
  // Pure RB-GS pressure (levels=1): decomposition-agnostic, so redistribute onto a WEIGHTED
  // partition works (the geometric coarse levels assume clean-coarsening; weighted-ORB co-decomp +
  // multilevel MG is the GraphAMG follow-up).
  s.setPressureLevels(1);
  s.setPressurePcg(true, 600, 1e-10);
}

// local SDF block for a decomposition's block on this rank.
static std::vector<double> localSdf(const std::vector<double>& gsdf, const BlockDecomposer<3>& dec,
                                    int rank) {
  auto b = dec.block(rank);
  const int ox = (int)b.origin[0], oy = (int)b.origin[1], oz = (int)b.origin[2];
  const int lnx = (int)b.size[0], lny = (int)b.size[1], lnz = (int)b.size[2];
  std::vector<double> l((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        l[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];
  return l;
}

static double localUSum(IbmSolver& s) {
  auto u = s.getVelocity(0);
  double sum = 0;
  for (double v : u)
    sum += v;
  return sum;
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

    // D1 = default equal-cell ORB; D2 = weighted ORB (heavy low-x half -> boundaries move).
    BlockDecomposer<3> D1((std::size_t)size, IVec<3>{N, N, N});
    std::vector<peclet::core::Real> w((std::size_t)N * N * N, 1.0);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (x < N / 2)
            w[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = 6.0;
    BlockDecomposer<3> D2((std::size_t)size, IVec<3>{N, N, N}, w);

    // distributed on D1, half the steps, REDISTRIBUTE to D2, finish.
    auto b1 = D1.block(rank);
    IbmSolver sd((int)b1.size[0], (int)b1.size[1], (int)b1.size[2]);
    sd.initMpi(D1, MPI_COMM_WORLD);
    configure(sd);
    sd.setSolid(localSdf(gsdf, D1, rank), true);
    for (int it = 0; it < STEPS / 2; ++it)
      sd.step();
    double uPre = 0, uPost = 0;
    {
      double l = localUSum(sd);
      MPI_Allreduce(&l, &uPre, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    sd.redistribute(D2);
    {
      double l = localUSum(sd);
      MPI_Allreduce(&l, &uPost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("  global-u sum pre=%.12e post=%.12e |d|=%.2e (redistribute data movement)\n",
                  uPre, uPost, std::fabs(uPre - uPost));
    for (int it = STEPS / 2; it < STEPS; ++it)
      sd.step();
    double lsum = localUSum(sd), gsum = 0;
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double k_dist = MU * (gsum / gcells) / F;
    const double div_dist = sd.maxOpenDivergence();

    // single-rank reference (validated path) on rank 0.
    double k_ref = 0.0;
    if (rank == 0) {
      IbmSolver ref(N, N, N);
      configure(ref);
      ref.setSolid(gsdf, true);
      for (int it = 0; it < STEPS; ++it)
        ref.step();
      double rsum = 0;
      for (double v : ref.getVelocity(0))
        rsum += v;
      k_ref = MU * (rsum / gcells) / F;
    }
    MPI_Bcast(&k_ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double reld = std::fabs(k_dist - k_ref) / (std::fabs(k_ref) + 1e-30);
    const double tol = (size == 1) ? 1e-12 : 2e-5;  // np=1 bit-exact; np>1 MG-PCG reduction floor
    if (rank == 0)
      std::printf(
          "  k_dist=%.8e  k_ref=%.8e  rel=%.2e  div=%.2e  (np=%d, tol %.0e, redistributed)\n",
          k_dist, k_ref, reld, div_dist, size, tol);
    if (reld > tol || !(div_dist < 1e-5))
      fail = 1;
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(totalFail == 0 ? "OK (np=%d): mid-run redistribute preserves the solution\n"
                               : "FAILED (np=%d)\n",
                size);
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
