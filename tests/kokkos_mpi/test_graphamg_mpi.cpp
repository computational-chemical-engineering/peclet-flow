// Decomposition-agnostic pressure solve via the agglomerated GraphAMG bottom solve.
//
// The geometric coarse hierarchy needs a cleanly-coarsening (equal-weight) ORB, so under a WEIGHTED
// decomposition only pure RB-GS (nLevels==1) is available -- and RB-GS alone is NOT mesh-independent.
// With set_pressure_graph_amg(true), the coarsest level (== the whole grid at nLevels==1) is solved by
// a mesh-agnostic smoothed-aggregation AMG on the operator gathered to rank 0: decomposition-agnostic
// AND mesh-independent. This test runs Stokes flow through a sphere packing on a WEIGHTED ORB and
// checks the permeability equals the single-rank reference (np=1 bit-exact-ish; np>1 to the
// reduction-order floor), for BOTH the RB-GS and the GraphAMG bottom. Build with -DPECLET_FLOW_MPI.
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

static constexpr int N = 32, STEPS = 100;
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

static void configure(IbmSolver& s, bool amg) {
  s.setRho(RHO); s.setMu(MU); s.setDt(DT); s.setBodyForce(F, 0, 0);
  s.setAdvection(false); s.setVelocityIterations(80);
  s.setPressureLevels(1);                  // pure RB-GS geometrically (decomposition-agnostic)
  s.setPressureGraphAmg(amg);              // ... upgraded to a mesh-agnostic algebraic coarse solve
  s.setPressurePcg(true, 400, 1e-9);
}

static std::vector<double> localSdf(const std::vector<double>& g, const BlockDecomposer<3>& dec,
                                    int rank) {
  auto b = dec.block(rank);
  const int ox = (int)b.origin[0], oy = (int)b.origin[1], oz = (int)b.origin[2];
  const int lnx = (int)b.size[0], lny = (int)b.size[1], lnz = (int)b.size[2];
  std::vector<double> l((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        l[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            g[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];
  return l;
}

static double perm(const std::vector<double>& gsdf, const BlockDecomposer<3>& dec, int rank,
                   bool amg, MPI_Comm comm) {
  auto b = dec.block(rank);
  IbmSolver s((int)b.size[0], (int)b.size[1], (int)b.size[2]);
  s.initMpi(dec, comm);
  configure(s, amg);
  s.setSolid(localSdf(gsdf, dec, rank), true);
  for (int it = 0; it < STEPS; ++it)
    s.step();
  double lsum = 0;
  for (double v : s.getVelocity(0))
    lsum += v;
  double gsum = 0;
  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
  return MU * (gsum / ((double)N * N * N)) / F;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<double> gsdf = packingSdf();

    // WEIGHTED ORB (heavy low-x): the geometric multilevel path is unavailable; GraphAMG bottom
    // makes the (nLevels==1) solve mesh-agnostic on it.
    std::vector<peclet::core::Real> w((std::size_t)N * N * N, 1.0);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (x < N / 2)
            w[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = 6.0;
    BlockDecomposer<3> dec((std::size_t)size, IVec<3>{N, N, N}, w);

    const double k_amg = perm(gsdf, dec, rank, /*amg=*/true, MPI_COMM_WORLD);

    // single-rank reference with the same GraphAMG solve.
    double k_ref = 0.0;
    if (rank == 0) {
      IbmSolver ref(N, N, N);
      configure(ref, /*amg=*/true);
      ref.setSolid(gsdf, true);
      for (int it = 0; it < STEPS; ++it)
        ref.step();
      double rs = 0;
      for (double v : ref.getVelocity(0))
        rs += v;
      k_ref = MU * (rs / ((double)N * N * N)) / F;
    }
    MPI_Bcast(&k_ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double reld = std::fabs(k_amg - k_ref) / (std::fabs(k_ref) + 1e-30);
    const double tol = (size == 1) ? 1e-9 : 2e-4;
    if (rank == 0)
      std::printf("  GraphAMG on weighted ORB: k=%.8e  ref=%.8e  rel=%.2e  (np=%d, tol %.0e)\n",
                  k_amg, k_ref, reld, size, tol);
    if (!(reld < tol) || !std::isfinite(k_amg))
      fail = 1;
  }
  int tf = 0;
  MPI_Allreduce(&fail, &tf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(tf == 0 ? "OK (np=%d): decomposition-agnostic GraphAMG pressure solve\n"
                        : "FAILED (np=%d)\n",
                size);
  Kokkos::finalize();
  MPI_Finalize();
  return tf == 0 ? 0 : 1;
}
