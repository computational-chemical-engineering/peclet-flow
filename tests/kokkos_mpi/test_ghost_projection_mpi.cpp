// flow — multi-rank validation of the FULL directional ghost-cell projection
// (set_ghost_projection, the 2nd-order scheme of ghost_projection.hpp — NOT face_interp mode 9).
//
// Lifted from single-rank (v1): gp-row ownership is by inner-block cell (buildGpOverlay runs over
// this rank's inner cells, closures read the exchanged g=2 sdfGp/velocity halo), the BiCGStab
// matvec stages the iterate on the solver's g=2 block so the overlay's +/-2 couplings see a
// current halo, and the fragmentation guard runs on the allgathered GLOBAL sdf (a pocket can span
// rank boundaries; every rank must agree on the main component).
//
// Sphere-packing Stokes flow, distributed (ORB blocks, initMpi) vs a full-grid single-rank
// reference on rank 0, final u compared POINTWISE (pattern of test_multiphysics_mpi):
//   - staggered  gp(2,2)  well-separated spheres
//   - staggered  gp(1,2)  TOUCHING spheres (mixed/deferred closure + contact pockets -> the
//                         global fragmentation guard is genuinely exercised across rank cuts)
//   - collocated gp(2,2)  face_interp 0 (the only face map the ghost mode admits)
// np=1 must be BIT-EXACT (catches ownership/halo mistakes cheaply); np>1 to the BiCGStab
// reduction floor (dot-product Allreduce reorder; measured ~5e-12).
//
// Found by this test (fixed in flow_ibm project()): the binary-openness operator leaves
// solid-centered cells as FREE variables; their Krylov-path-dependent phi leaked into near-wall
// fluid u through the plain projectCorrect gradient — decomposition-dependent at ~1e-2 relative
// while every gp diagnostic stayed clean. phi is now pinned to the design value 0 there.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using Stag = peclet::flow::IbmSolver;
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;

static constexpr int N = 32, STEPS = 15;
static constexpr double RHO = 1.0, MU = 0.1, F = 1e-3, DT = 20.0;

// 2x2x2 periodic sphere packing (flat x-fastest, negative inside).
static std::vector<double> packingSdf(double rfrac) {
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

template <class S>
static void configure(S& s, const std::vector<double>& lsdf, int matrixOrder, int rhsOrder) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0, 0);
  s.setAdvection(false);
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressurePcg(true, 300, 1e-9);
  s.setGhostProjection(true, matrixOrder, rhsOrder);  // BEFORE set_solid (overlay built there)
  s.setSolid(lsdf, /*cutcell_pressure=*/true);
}

// Gather per-rank inner blocks into the global field on rank 0 (test_multiphysics_mpi pattern).
static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign((std::size_t)N * N * N, 0.0);
  for (int r = 0; r < size; ++r) {
    int meta[6] = {ox, oy, oz, lnx, lny, lnz};
    if (r == 0) {
      if (rank == 0)
        for (int z = 0; z < lnz; ++z)
          for (int y = 0; y < lny; ++y)
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * N +
                                (std::size_t)(z + oz) * N * N],
                        &local[(std::size_t)y * lnx + (std::size_t)z * lnx * lny],
                        (std::size_t)lnx * sizeof(double));
      continue;
    }
    if (rank == r) {
      MPI_Send(meta, 6, MPI_INT, 0, 100 + r, MPI_COMM_WORLD);
      MPI_Send(local.data(), (int)local.size(), MPI_DOUBLE, 0, 200 + r, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(meta, 6, MPI_INT, r, 100 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> buf((std::size_t)meta[3] * meta[4] * meta[5]);
      MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 200 + r, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int z = 0; z < meta[5]; ++z)
        for (int y = 0; y < meta[4]; ++y)
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * N +
                              (std::size_t)(z + meta[2]) * N * N],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

static double relErr(const std::vector<double>& a, const std::vector<double>& b) {
  double md = 0, mb = 0;
  std::size_t am = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > md) {
      md = std::fabs(a[i] - b[i]);
      am = i;
    }
    mb = std::max(mb, std::fabs(b[i]));
  }
  if (std::getenv("GP_MPI_DEBUG"))
    std::printf("    argmax |a-b| at (%d,%d,%d)\n", (int)(am % N), (int)((am / N) % N),
                (int)(am / ((std::size_t)N * N)));
  return md / (mb + 1e-300);
}

// One distributed-vs-reference comparison for a solver type + gp order pair + geometry.
template <class S>
static int runCase(const char* name, double rfrac, int mo, int ro, int rank, int size) {
  const std::vector<double> gsdf = packingSdf(rfrac);
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{N, N, N});
  auto blk = dec.block(rank);
  const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
  const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
  std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];

  S sd(lnx, lny, lnz);
  sd.initMpi(N, N, N, MPI_COMM_WORLD);
  configure(sd, lsdf, mo, ro);
  for (int it = 0; it < STEPS; ++it)
    sd.step();
  auto gu = gatherGlobal(sd.getVelocity(0), ox, oy, oz, lnx, lny, lnz, rank, size);
  const double resd = sd.maxOpenDivergence();  // distributed diagnostic (collective)

  int fail = 0;
  if (rank == 0) {
    S ref(N, N, N);
    configure(ref, gsdf, mo, ro);
    for (int it = 0; it < STEPS; ++it)
      ref.step();
    const double eu = relErr(gu, ref.getVelocity(0));
    // np=1 is the strongest gate: the distributed path (halo exchange in place of the periodic
    // wrap, staged matvec, allgathered fragmentation guard) must be BIT-exact.
    // measured floors (host-openmp): np=1 exactly 0; np>1 <= ~5e-12 (BiCGStab dot-product
    // Allreduce reorder only — the decoupled-phi pinning removed every larger term).
    const double tol = (size == 1) ? 0.0 : 1e-9;
    const bool ok = eu <= tol;
    std::printf("  [%-12s np=%d] u rel=%.3e  tol=%.0e  div(d)=%.3e div(ref)=%.3e  %s\n", name,
                size, eu, tol, resd, ref.maxOpenDivergence(), ok ? "OK" : "FAIL");
    if (!ok)
      fail = 1;
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return fail;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fail |= runCase<Stag>("stag-gp22", 0.18, 2, 2, rank, size);
    fail |= runCase<Stag>("stag-gp12-tch", 0.26, 1, 2, rank, size);
    fail |= runCase<Colo>("colo-gp22", 0.18, 2, 2, rank, size);
    if (rank == 0)
      std::printf("GHOST PROJECTION MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
