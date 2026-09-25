// flow — set_bulk_velocity under MPI: the on-device bulk-velocity (zero-net-flux) constraint.
//
// `setBulkVelocity(true, axis, U)` adds, at the end of every step(), one uniform shift to every
// inner face velocity of component `axis` so that its volume mean is U. It replaces a Python driver
// that did `u = s.get_u(); s.set_velocity(0, u - u.mean())` after each step (the bubble-column
// benchmark's closed column): the same operation, but a device reduction (a global Allreduce under
// MPI) instead of a full-field round trip through the host.
//
// Case: 32^3, x and y periodic, no-slip walls on +-z (np = 4 is 2x2x1: the ORB cuts x and y, never
// the wall-normal axis), a uniform body force f_x plus six Gaussian force_x blobs (a 3-D field, so
// the reduction has something to reorder), constant rho, MG-PCG pinned at rtol 1e-12, STEPS steps,
// target bulk velocity U = 0.02 (not 0, so the conversion of the target is exercised too).
//
// Gates:
//   * `driver`  — the device constraint against the host-driver semantics it replaces: a
//                 single-rank reference WITHOUT the constraint whose velocity is shifted on the
//                 host between steps (getVelocity -> subtract mean - U -> setVelocity). np = 1 at
//                 the reduction-order floor (host sum vs device tree), np > 1 at the MG-PCG floor
//                 of test_cellforce_mpi (1e-11 of max|u|).
//   * `mean`    — the global volume mean of u after the last step equals U to 1e-13 of max|u|.
//   * `active`  — the constraint had something to remove: |last shift| > 1e-3 (the force drives a
//                 net flow of order 1e-1 per step).
//   * `refuse`  — the constraint on the wall-bounded z axis makes step() raise on every rank.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 3;
static constexpr std::size_t GCELLS = (std::size_t)N * N * N;
static constexpr double U_BULK = 0.02;

// Six blobs along x at (i + 1/2) N/6, y = 12, z = 10 (off-centre: the flow is fully 3-D).
static std::vector<double> blobs() {
  std::vector<double> f(GCELLS, 0.0);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double s = 0.0;
        for (int i = 0; i < 6; ++i) {
          const double dx = x + 0.5 - (i + 0.5) * N / 6.0, dy = y + 0.5 - 12.0, dz = z + 0.5 - 10.0;
          s += 0.5 * std::exp(-(dx * dx + dy * dy + dz * dz) / 8.0);
        }
        f[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = s;
      }
  return f;
}

static std::vector<double> slice(const std::vector<double>& g, int ox, int oy, int oz, int lnx,
                                 int lny, int lnz) {
  std::vector<double> l((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        l[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            g[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];
  return l;
}

// BCs first (they are folded into the operators the geometry builds), then the geometry.
static void build(IbmSolver& s, std::size_t cells) {
  s.setDomainBc(4, 1, 0.0, 0.0, 0.0);  // -z wall
  s.setDomainBc(5, 1, 0.0, 0.0, 0.0);  // +z wall
  s.setPressureGeometry(std::vector<double>(cells, 10.0));
}
static void configure(IbmSolver& s, const std::vector<double>& forceLocal) {
  s.setRho(1.0);
  s.setMu(0.05);
  s.setDt(0.5);
  s.setBodyForce(0.02, 0.0, 0.0);
  s.setPressurePcg(true, 200, 1e-12);
  s.enableCellForce();
  s.setField("force_x", forceLocal);
}

static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign(GCELLS, 0.0);
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

// NaN-propagating (std::fmax drops a NaN; see test_bodyforce_ghost_mpi).
static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;
    m = std::fmax(m, d);
  }
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}
static double mean(const std::vector<double>& a) {
  double s = 0;
  for (double v : a)
    s += v;
  return s / (double)a.size();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    const std::size_t lcells = (std::size_t)lnx * lny * lnz;
    if (rank == 0)
      std::printf("BULK VELOCITY MPI np=%d  grid %d^3  block %dx%dx%d\n", size, N, lnx, lny, lnz);
    const std::vector<double> fg = blobs();

    // ---- driver + mean + active -------------------------------------------------------------
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    build(sd, lcells);
    configure(sd, slice(fg, ox, oy, oz, lnx, lny, lnz));
    sd.setBulkVelocity(true, 0, U_BULK);
    for (int it = 0; it < STEPS; ++it)
      sd.step();
    const double shift = sd.lastBulkVelocityShift();
    std::vector<double> gu[3];
    for (int c = 0; c < 3; ++c)
      gu[c] = gatherGlobal(sd.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);

    if (rank == 0) {
      IbmSolver ref(N, N, N);
      build(ref, GCELLS);
      configure(ref, fg);
      for (int it = 0; it < STEPS; ++it) {
        ref.step();
        std::vector<double> u = ref.getVelocity(0);
        const double ub = mean(u) - U_BULK;
        for (double& v : u)
          v -= ub;
        ref.setVelocity(0, u);
      }
      double du = 0, umag = 0;
      for (int c = 0; c < 3; ++c) {
        du = std::fmax(du, maxAbsDiff(gu[c], ref.getVelocity(c)));
        umag = std::fmax(umag, maxAbs(ref.getVelocity(c)));
      }
      const double tol = (size == 1 ? 1e-13 : 1e-11) * umag;
      const bool okD = du <= tol && umag > 1e-2;
      std::printf("  [driver np=%d] max|du| = %.3e (tol %.1e)  max|u| = %.3e  %s\n", size, du, tol,
                  umag, okD ? "OK" : "FAIL");
      const double dm = std::fabs(mean(gu[0]) - U_BULK);
      const bool okM = dm <= 1e-13 * umag;
      std::printf("  [mean   np=%d] |<u> - U| = %.3e  %s\n", size, dm, okM ? "OK" : "FAIL");
      const bool okA = std::fabs(shift) > 1e-3;
      std::printf("  [active np=%d] last shift = %.6e  %s\n", size, shift, okA ? "OK" : "FAIL");
      if (!(okD && okM && okA))
        fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- refuse: the wall-bounded axis --------------------------------------------------------
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(dec, MPI_COMM_WORLD);
      build(s, lcells);
      configure(s, slice(fg, ox, oy, oz, lnx, lny, lnz));
      s.setBulkVelocity(true, 2, 0.0);
      bool threw = false;
      try {
        s.step();
      } catch (const std::runtime_error&) {
        threw = true;
      }
      int t = threw ? 1 : 0, all = 0;
      MPI_Allreduce(&t, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("  [refuse np=%d] wall-bounded axis %s  %s\n", size,
                    all ? "raised" : "was ACCEPTED", all ? "OK" : "FAIL");
      if (!all)
        fail = 1;
    }
    if (rank == 0)
      std::printf("BULK VELOCITY MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
