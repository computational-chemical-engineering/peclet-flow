// flow — a mid-run redistribute on the POROUS (volume-averaged) continuity must not look like a
// jump in the porosity.
//
// The porous projection enforces d(eps)/dt + div(eps u) = 0 with d(eps)/dt = (eps^{n+1} -
// eps^n)/dt, and eps^n (`epsPrev_`) is state that no registry field carries. `redistribute`
// re-seeds it from the migrated eps^{n+1} (d(eps)/dt = 0 across the move, which leaves the
// projection RHS as it was). It used to re-seed it inside `resizeForBlock`, BEFORE the migrated
// fields were scattered into their new buffers -- so whenever a rank's block changed size it copied
// a freshly zeroed eps, and the first projection after the move saw d(eps)/dt = eps/dt: a CfdDem
// rebalance at np = 2 and 4 put the pressure off by 2.2e+02 and the velocity by 2.4 (|u| <= 0.07).
//
// Here: Stokes flow through a sphere packing with a static, z-varying porosity, run distributed on
// the equal-cell ORB for half the steps, REDISTRIBUTED onto a weighted ORB (every block changes
// size at np >= 2) and run to the end. The final velocity must match a full-grid single-rank
// reference that never redistributed, pointwise, to the solver's reduction floor (np = 1: the
// weighted "ORB" is the one block, bit for bit). Build with -DPECLET_FLOW_MPI.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 20;
static constexpr double RHO = 1.0, MU = 0.1, F = 1e-3, DT = 20.0;

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

static double epsAt(int x, int, int z) {
  return 0.6 + 0.25 * std::sin(2.0 * M_PI * z / N) + 0.05 * std::cos(2.0 * M_PI * x / N);
}

// This rank's block of a global field (x-fastest local buffer).
static std::vector<double> blockOf(const std::vector<double>& g, const BlockDecomposer<3>& dec,
                                   int rank) {
  const auto b = dec.block(rank);
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

static void configure(IbmSolver& s, const std::vector<double>& lsdf,
                      const std::vector<double>& leps) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0, 0);
  s.setAdvection(false);
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressurePcg(true, 200, 1e-10);
  s.setSolid(lsdf, /*cutcell_pressure=*/true);
  s.setPorousContinuity(true);
  s.setField("eps", leps);
  s.exchangeField("eps");
  s.syncPorousPrev();  // eps^n = eps^{n+1}: no d(eps)/dt on the first step
}

// The global field on rank 0 from every rank's inner block of `dec`.
static std::vector<double> gatherGlobal(const std::vector<double>& local,
                                        const BlockDecomposer<3>& dec, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign((std::size_t)N * N * N, 0.0);
  for (int r = 0; r < size; ++r) {
    const auto b = dec.block(r);
    const int ox = (int)b.origin[0], oy = (int)b.origin[1], oz = (int)b.origin[2];
    const int lnx = (int)b.size[0], lny = (int)b.size[1], lnz = (int)b.size[2];
    std::vector<double> buf;
    if (r == rank)
      buf = local;
    if (r != 0) {
      if (rank == r)
        MPI_Send(buf.data(), (int)buf.size(), MPI_DOUBLE, 0, 300 + r, MPI_COMM_WORLD);
      else if (rank == 0) {
        buf.resize((std::size_t)lnx * lny * lnz);
        MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 300 + r, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
    }
    if (rank == 0)
      for (int z = 0; z < lnz; ++z)
        for (int y = 0; y < lny; ++y)
          std::memcpy(
              &global[(std::size_t)ox + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N],
              &buf[(std::size_t)y * lnx + (std::size_t)z * lnx * lny],
              (std::size_t)lnx * sizeof(double));
  }
  return global;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<double> gsdf = packingSdf();
    std::vector<double> geps((std::size_t)N * N * N);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          geps[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = epsAt(x, y, z);

    // D1 = the equal-cell ORB; D2 = a weighted ORB (a heavy low-x, low-z corner) whose blocks all
    // differ in size from D1's at np >= 2 -- the case that reallocates every field buffer.
    BlockDecomposer<3> D1((std::size_t)size, IVec<3>{N, N, N});
    std::vector<peclet::core::Real> w((std::size_t)N * N * N, 1.0);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          if (x < N / 3 || z < N / 4)
            w[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = 5.0;
    BlockDecomposer<3> D2((std::size_t)size, IVec<3>{N, N, N}, w);
    int resized = 0;
    for (int k = 0; k < 3; ++k)
      resized |= D1.block(rank).size[k] != D2.block(rank).size[k];

    const auto b1 = D1.block(rank);
    IbmSolver sd((int)b1.size[0], (int)b1.size[1], (int)b1.size[2]);
    sd.initMpi(D1, MPI_COMM_WORLD);
    configure(sd, blockOf(gsdf, D1, rank), blockOf(geps, D1, rank));
    for (int it = 0; it < STEPS / 2; ++it)
      sd.step();
    sd.redistribute(D2);
    for (int it = STEPS / 2; it < STEPS; ++it)
      sd.step();
    const std::vector<double> ud = gatherGlobal(sd.getVelocity(0), D2, rank, size);
    const std::vector<double> wd = gatherGlobal(sd.getVelocity(2), D2, rank, size);
    int anyResized = 0;
    MPI_Allreduce(&resized, &anyResized, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
      IbmSolver ref(N, N, N);
      configure(ref, gsdf, geps);
      for (int it = 0; it < STEPS; ++it)
        ref.step();
      const std::vector<double> ur = ref.getVelocity(0), wr = ref.getVelocity(2);
      double du = 0, dw = 0, mu = 0;
      for (std::size_t i = 0; i < ur.size(); ++i) {
        du = std::max(du, std::fabs(ud[i] - ur[i]));
        dw = std::max(dw, std::fabs(wd[i] - wr[i]));
        mu = std::max(mu, std::fabs(ur[i]));
      }
      const double rel = std::max(du, dw) / mu;
      const double tol = (size == 1) ? 0.0 : 1e-6;  // np = 1 bit-exact; np > 1 the reduction floor
      std::printf(
          "  np=%d: blocks resized %s; max|du|=%.3e max|dw|=%.3e (max|u| %.3e) rel=%.2e"
          " (tol %.0e)\n",
          size, anyResized ? "yes" : "no", du, dw, mu, rel, tol);
      if (!(rel <= tol) || (size > 1 && !anyResized))
        fail = 1;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(totalFail == 0 ? "OK (np=%d): a porous redistribute leaves the solution unchanged\n"
                               : "FAILED (np=%d)\n",
                size);
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
