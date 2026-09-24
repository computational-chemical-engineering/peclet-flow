// flow — doc/vof_overlap_design.md gate G5: the block container's DEBRIS removal, distributed.
//
// The G1 scene (two R = 5 markers 16 cells apart along x, plus a speck of B's colour painted into
// A's interface band -- three cells with no cell of B above 1/2 within two cells) under a linear
// shear u = gamma (y - y_c), advected 20 steps by the block container with the block CSF on (so
// the debris pass runs). The speck is removed at the first step and its volume returned to B's
// attached interface; the removal's sums are sequential in index order on the master, so the
// result must be BITWISE independent of the decomposition.
//
// Compared bitwise against a single-rank run of the same scene (rank 0): the union colour, every
// marker's volume, and the five ledger fields. At np >= 2 the ORB cuts x, i.e. between the two
// marker centres, and the masters are round-robin, so A and B live on different ranks.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 48, NY = 32, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
static constexpr double R = 5.0, SIGMA = 320.0, AX = 16.3, BX = 32.3, CY = 16.2, CZ = 16.1;
static constexpr int SX = 21, SY0 = 15, SZ = 16;
static constexpr double SPECK[3] = {1e-5, 1e-2, 3e-3};
static constexpr double GAMMA = 0.25, DT = 0.05;
static const std::array<int, 6> BOX[2] = {{7, 7, 7, 26, 26, 26}, {18, 7, 7, 41, 26, 26}};

static double sphereFrac(double cx, int x, int y, int z) {
  const int NS = 8;
  int in = 0;
  for (int k = 0; k < NS; ++k)
    for (int j = 0; j < NS; ++j)
      for (int i = 0; i < NS; ++i) {
        const double px = x + (i + 0.5) / NS - cx, py = y + (j + 0.5) / NS - CY,
                     pz = z + (k + 0.5) / NS - CZ;
        in += (px * px + py * py + pz * pz < R * R) ? 1 : 0;
      }
  return static_cast<double>(in) / (NS * NS * NS);
}

static std::vector<double> colour(int m) {
  const auto& b = BOX[m];
  std::vector<double> c;
  for (int z = b[2]; z < b[5]; ++z)
    for (int y = b[1]; y < b[4]; ++y)
      for (int x = b[0]; x < b[3]; ++x) {
        double v = sphereFrac(m == 0 ? AX : BX, x, y, z);
        if (m == 1 && x == SX && z == SZ && y >= SY0 && y < SY0 + 3)
          v = SPECK[y - SY0];
        c.push_back(v);
      }
  return c;
}

static void configure(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  const std::size_t loc = (std::size_t)lnx * lny * lnz;
  s.setRho(1.0);
  s.setMu(0.5);
  s.setPressureGeometry(std::vector<double>(loc, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>(loc, 0.0));
  s.setSurfaceTension(SIGMA);
  s.enableVofBlocksFromColours({BOX[0], BOX[1]}, {colour(0), colour(1)});
  s.enableVofBlockCsf();  // the debris pass runs under the block CSF (removal default ON)
  // u = gamma (y - y_c) at the u faces (x + 1/2, y, z): divergence-free, periodic in x.
  std::vector<double> u(loc, 0.0), zero(loc, 0.0);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        u[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            GAMMA * ((y + oy) + 0.5 - CY);
  (void)ox;
  (void)oz;
  s.setVelocity(0, u);
  s.setVelocity(1, zero);
  s.setVelocity(2, zero);
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
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * NX +
                                (std::size_t)(z + oz) * NX * NY],
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
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * NX +
                              (std::size_t)(z + meta[2]) * NX * NY],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

// per marker: volume, debrisCells, debrisVolume, debrisReturned, debrisLost, debrisUnresolved
static constexpr int NF = 6;
static std::vector<double> ledger(IbmSolver& s) {
  std::vector<double> v;
  for (const auto& q : s.vofBlockStats()) {
    v.push_back(q.volume);
    v.push_back(static_cast<double>(q.debrisCells));
    v.push_back(q.debrisVolume);
    v.push_back(q.debrisReturned);
    v.push_back(q.debrisLost);
    v.push_back(static_cast<double>(q.debrisUnresolved));
  }
  return v;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    bool cutX = false;
    for (const auto& sz : dec.sizes())
      if ((int)sz[0] != NX)
        cutX = true;
    if (rank == 0)
      std::printf(
          "VOF BLOCK DEBRIS MPI (vof_overlap_design G5) np=%d  grid %dx%dx%d  block "
          "%dx%dx%d  x cut: %s\n",
          size, NX, NY, NZ, lnx, lny, lnz, cutX ? "yes" : "no");
    if (size > 1 && !cutX) {
      if (rank == 0)
        std::printf("  FAIL - the decomposition does not cut x, so the markers share a rank\n");
      fail = 1;
    }
    const int STEPS = (argc > 1) ? std::atoi(argv[1]) : 20;
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configure(sd, ox, oy, oz, lnx, lny, lnz);
    for (int k = 0; k < STEPS; ++k)
      sd.advectVofBlocks(DT, /*requireSolenoidal=*/false);
    // the measured entries live on each block's master (zero elsewhere): a SUM is the census,
    // and x + 0 is exact, so the sum carries the master's double bit for bit
    std::vector<double> lLoc = ledger(sd), lAll(lLoc.size(), 0.0);
    MPI_Allreduce(lLoc.data(), lAll.data(), (int)lLoc.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const std::vector<double> gC = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
    if (rank == 0) {
      IbmSolver ref(NX, NY, NZ);
      configure(ref, 0, 0, 0, NX, NY, NZ);
      const std::vector<double> r0 = ledger(ref);
      for (int k = 0; k < STEPS; ++k)
        ref.advectVofBlocks(DT, false);
      const std::vector<double> rl = ledger(ref), rC = ref.getVof();
      long nC = 0;
      for (std::size_t i = 0; i < GCELLS; ++i)
        nC += (gC[i] == rC[i]) ? 0 : 1;
      long nL = 0;
      for (std::size_t i = 0; i < rl.size(); ++i)
        nL += (lAll[i] == rl[i]) ? 0 : 1;
      const char* fn[NF] = {"volume", "cells", "vol", "returned", "lost", "unresolved"};
      for (std::size_t m = 0; m < rl.size() / NF; ++m) {
        std::printf("  marker %zu:", m);
        for (int f = 0; f < NF; ++f)
          std::printf("  %s %.17g%s", fn[f], lAll[m * NF + f],
                      lAll[m * NF + f] == rl[m * NF + f] ? "" : " (!= np1)");
        std::printf("\n");
      }
      const double painted = SPECK[0] + SPECK[1] + SPECK[2];
      std::printf(
          "  after %d steps: union colour cells differing from np=1: %ld; ledger fields "
          "differing: %ld; B returned %.17g (painted %.17g)\n",
          STEPS, nC, nL, rl[NF + 3], painted);
      double dv = 0.0;
      for (std::size_t m = 0; m < rl.size() / NF; ++m)
        dv = std::fmax(dv, std::fabs(rl[m * NF] - r0[m * NF]) / r0[m * NF]);
      std::printf("  max marker volume rel change over the run (np=1): %.3e\n", dv);
      if (!(dv <= 1e-12)) {
        std::printf("  FAIL: marker volume not conserved\n");
        fail = 1;
      }
      if (nC != 0 || nL != 0) {
        std::printf("  FAIL: not bitwise decomposition-independent\n");
        fail = 1;
      }
      if (!(rl[NF + 3] > 0.9 * painted)) {  // the gate must not be vacuous: the speck was removed
        std::printf("  FAIL: the debris removal did not fire (returned %.3e)\n", rl[NF + 3]);
        fail = 1;
      }
      if (!(rl[3] == 0.0 && rl[NF + 4] == 0.0)) {
        std::printf("  FAIL: marker A removed debris, or the return lost volume\n");
        fail = 1;
      }
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("%s\n", fail ? "FAILED" : "PASSED");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
