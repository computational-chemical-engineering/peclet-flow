// flow — the distributed Anderson accelerator of a steady march (doc/steady_acceleration.md
// rev 1, gate G4c).
//
// The §11 case in cell units (one sphere of solid fraction 0.125 at the centre of a periodic
// 16^3 box, rho = mu = 1, a body force along x, Stokes) on the staggered Solver, 40 unconditional
// accelerated steps (AndersonAccelerator::step(true), window 5), two ways: distributed over np
// ranks (each rank its ORB block, initMpi, its local SDF block) and single-rank on rank 0 (the
// serial path, the full grid). Gates:
//   * max|u_np - u_1| <= 1e-8 max|u_1| after the 40 steps, every velocity component (np = 1, 2, 4);
//   * gamma, the window size and the status are BITWISE equal on all ranks after every step
//     (rank 0's broadcast decision packet, design §6.1: each rank mixes its own ghosts, which stay
//     equal to the neighbour's mixed inner values only under identical coefficients).
// Inner solves are tight (PCG rtol 1e-12, velocity residual tolerance 1e-12) and nu dt / h^2 = 0.5,
// so the momentum solve is red-black Gauss-Seidel on every block size (no V-cycle switch between
// the 16^3 single-rank block and the 8-cell distributed blocks) and converges to its tolerance:
// what remains between np is the reduction order of the pressure PCG and of the accelerator's own
// Gram sums, which is what this gate bounds.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "anderson_accelerator.hpp"
#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::AndersonAccelerator;
using peclet::flow::IbmSolver;
using peclet::flow::Staggered;

static constexpr int N = 16, STEPS = 40, WINDOW = 5;

static std::vector<double> sphereSdf() {
  const double R = std::cbrt(3.0 * 0.125 / (4.0 * M_PI)) * N, c = 0.5 * N;
  std::vector<double> sdf((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - c, dy = y + 0.5 - c, dz = z + 0.5 - c;
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return sdf;
}

static void configure(IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(0.5);
  s.setBodyForce(1e-3, 0.0, 0.0);
  s.setAdvection(false);
  s.setVelocityIterations(200);
  s.setPressureLevels(3);
  s.setPressurePcg(true, 400, 1e-12);
  s.setVelocityResidualTolerance(1e-12);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<double> gsdf = sphereSdf();

    // --- distributed ---
    auto dec =
        peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N, 0, 1.05);
    auto blk = dec.block(rank);
    const int org[3] = {(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
    const int ln[3] = {(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
    std::vector<double> lsdf((std::size_t)ln[0] * ln[1] * ln[2]);
    for (int z = 0; z < ln[2]; ++z)
      for (int y = 0; y < ln[1]; ++y)
        for (int x = 0; x < ln[0]; ++x)
          lsdf[(std::size_t)x + (std::size_t)y * ln[0] + (std::size_t)z * ln[0] * ln[1]] =
              gsdf[(std::size_t)(x + org[0]) + (std::size_t)(y + org[1]) * N +
                   (std::size_t)(z + org[2]) * N * N];
    IbmSolver sd(ln[0], ln[1], ln[2]);
    sd.initMpi(N, N, N, MPI_COMM_WORLD);
    configure(sd);
    sd.setSolid(lsdf, true);
    long gammaMismatch = 0;
    double lastRes = 0.0;
    {
      AndersonAccelerator<Staggered> acc(sd, WINDOW, 1.0);
      for (int k = 0; k < STEPS; ++k) {
        acc.step(true);
        std::vector<double> g = acc.core().gamma();
        int head[2] = {(int)g.size(), (int)acc.core().status()};
        int head0[2] = {head[0], head[1]};
        MPI_Bcast(head0, 2, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> g0 = g;
        g0.resize((std::size_t)head0[0]);
        MPI_Bcast(g0.data(), head0[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (head[0] != head0[0] || head[1] != head0[1] ||
            std::memcmp(g.data(), g0.data(), sizeof(double) * g.size()) != 0)
          ++gammaMismatch;
      }
      lastRes = acc.residual();
      if (rank == 0)
        std::printf("  distributed: status %s, residual %.6e, %d columns, %d restarts\n",
                    acc.statusName(), acc.residual(), acc.numColumns(), acc.numRestarts());
    }
    long gm = 0;
    MPI_Allreduce(&gammaMismatch, &gm, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    // gather the distributed velocity onto rank 0 (inner blocks, x-fastest)
    std::vector<std::vector<double>> ug(3);
    for (int c = 0; c < 3; ++c) {
      const std::vector<double> loc = sd.getVelocity(c);
      std::vector<int> meta(6 * size);
      int mine[6] = {org[0], org[1], org[2], ln[0], ln[1], ln[2]};
      MPI_Gather(mine, 6, MPI_INT, meta.data(), 6, MPI_INT, 0, MPI_COMM_WORLD);
      std::vector<int> counts(size), displs(size);
      int nloc = (int)loc.size();
      MPI_Gather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
      std::vector<double> all;
      if (rank == 0) {
        int off = 0;
        for (int r = 0; r < size; ++r) {
          displs[r] = off;
          off += counts[r];
        }
        all.resize((std::size_t)off);
      }
      MPI_Gatherv(loc.data(), nloc, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        ug[c].assign((std::size_t)N * N * N, 0.0);
        for (int r = 0; r < size; ++r) {
          const int* m = &meta[6 * r];
          for (int z = 0; z < m[5]; ++z)
            for (int y = 0; y < m[4]; ++y)
              for (int x = 0; x < m[3]; ++x)
                ug[c][(std::size_t)(x + m[0]) + (std::size_t)(y + m[1]) * N +
                      (std::size_t)(z + m[2]) * N * N] =
                    all[(std::size_t)displs[r] + (std::size_t)x + (std::size_t)y * m[3] +
                        (std::size_t)z * m[3] * m[4]];
        }
      }
    }

    // --- single-rank reference on rank 0 (the serial path) ---
    if (rank == 0) {
      IbmSolver ref(N, N, N);
      configure(ref);
      ref.setSolid(gsdf, true);
      AndersonAccelerator<Staggered> acc(ref, WINDOW, 1.0);
      for (int k = 0; k < STEPS; ++k)
        acc.step(true);
      std::printf("  single-rank: status %s, residual %.6e (distributed %.6e)\n", acc.statusName(),
                  acc.residual(), lastRes);
      double worst = 0.0;
      for (int c = 0; c < 3; ++c) {
        const std::vector<double> u1 = ref.getVelocity(c);
        double du = 0.0, umax = 0.0;
        for (std::size_t i = 0; i < u1.size(); ++i) {
          du = std::max(du, std::fabs(ug[c][i] - u1[i]));
          umax = std::max(umax, std::fabs(u1[i]));
        }
        const double rel = umax > 0.0 ? du / umax : du;
        worst = std::max(worst, rel);
        std::printf("  component %d: max|u_np - u_1| / max|u_1| = %.3e (max|u_1| = %.6e)\n", c, rel,
                    umax);
      }
      std::printf("  gamma / window / status mismatches across ranks: %ld\n", gm);
      if (worst > 1e-8 || gm != 0)
        fail = 1;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): distributed Anderson march == single-rank, gamma identical\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
