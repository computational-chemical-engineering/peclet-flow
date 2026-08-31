// flow — multi-rank validation of the VoF curvature cascade (rung V3, WO-O).
//
// The cascade (`src/vof/curvature.hpp` + `curvature_field.hpp`) is a PURE LOCAL STENCIL on the
// colour field's g = 3 block: three height-function tiers over a 3x3x7 patch, then a
// PLIC-volumetric paraboloid fit over a 5^3 stencil of MYC planes — reach exactly +/-3 in every
// direction, and **not one reduction anywhere in it**. So unlike every other coupled VoF gate in
// this directory, this one is required to be **BITWISE** identical at np 2 and 4, not merely
// identical to the reduction-order floor: there is no allreduce whose summation order could move.
// A failure here is a real defect, never a floor.
//
// What it can catch, in order of likelihood:
//   1. a stencil that reaches past g = 3. The whole rung was designed to fit exactly inside the
//      colour halo; one cell too far and the answer becomes decomposition-dependent, silently and
//      only in the ghost-adjacent cells (the symptom WO-E's finding 2 named).
//   2. the non-periodic colour ghost fill. `vof::clampFill` takes the value at the GLOBALLY
//      clamped index precisely so it is decomposition-independent; sequential zero-gradient axis
//      passes over the block are NOT. The height function integrates a 7-cell column, so a
//      curvature at the third inner cell from a wall reads that fill directly. `walls-z` cuts the
//      walled axis to exercise it.
//   3. the branch census. It is a per-rank local count; the sum over ranks must equal the
//      single-rank census exactly, which is a much stronger statement than the field comparison
//      alone (a cell that changes branch but happens to land on a similar value would pass the
//      field check at 1e-12 and fail this one).
//
// Two configurations on a 16x16x32 grid — the aligned ORB cuts the long z axis at np = 2 and 4 —
// with the sphere deliberately placed so its interface straddles a rank boundary, and a resolution
// (R = 4 cells, D/dx = 8) at which the PLIC-volumetric fallback provably fires on a large fraction
// of the interfacial cells. A test in which only the height-function branch runs would not gate the
// fallback's stencil reach at all.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  bool walls;  // walls +-z (exercises the clamped, non-periodic colour ghost) vs fully periodic
};

// Sharp colour at a GLOBAL cell index — a pure function of global indices, so a block gets
// bit-identical data to the corresponding part of the single-rank run. Fractions by 8^3 midpoint
// subsampling: exactness is irrelevant here (both runs see the SAME field), coverage of the
// cascade's branches is not.
static double colourAt(const Config& c, int x, int y, int z) {
  const double cx = NX / 2.0 + 0.137, cy = NY / 2.0 - 0.311;
  // periodic: sphere centred on the z rank boundary at np = 2. walls: on the quarter plane, so the
  // np = 4 boundary at z = NZ/4 cuts it too and the +-z walls stay far from the interface.
  const double cz = c.walls ? NZ / 4.0 + 0.241 : NZ / 2.0 + 0.241;
  const double R = 4.0;
  const int NS = 8;
  int in = 0;
  for (int k = 0; k < NS; ++k)
    for (int j = 0; j < NS; ++j)
      for (int i = 0; i < NS; ++i) {
        const double dx = x + (i + 0.5) / NS - cx, dy = y + (j + 0.5) / NS - cy,
                     dz = z + (k + 0.5) / NS - cz;
        if (dx * dx + dy * dy + dz * dz < R * R)
          ++in;
      }
  return static_cast<double>(in) / (NS * NS * NS);
}

template <class Fn>
static std::vector<double> blockOf(Fn f, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  std::vector<double> v((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        v[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            f(x + ox, y + oy, z + oz);
  return v;
}

static void configure(IbmSolver& s, const Config& c, int ox, int oy, int oz, int lnx, int lny,
                      int lnz) {
  s.setRho(1.0);
  s.setMu(0.01);
  s.setDt(1.0);
  if (c.walls) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z (the CUT axis)
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setVof(blockOf([&](int x, int y, int z) { return colourAt(c, x, y, z); }, ox, oy, oz, lnx, lny,
                   lnz));
}

// Gather per-rank inner blocks (x-fastest) into the global field on rank 0.
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

static long countBitDiff(const std::vector<double>& a, const std::vector<double>& b, double& mx) {
  long n = 0;
  mx = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i)
    if (a[i] != b[i]) {
      ++n;
      mx = std::fmax(mx, std::fabs(a[i] - b[i]));
    }
  return n;
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
    const int gn[3] = {NX, NY, NZ};
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != gn[a])
          cut[a] = true;
    if (rank == 0)
      std::printf("VOF CURVATURE MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"sphere-per", false}, {"sphere-walls-z", true}};

    for (const Config& c : configs) {
      // The interface must straddle a rank boundary for the test to mean anything.
      if (size > 1 && !cut[2]) {
        if (rank == 0)
          std::printf("  [%-14s np=%d] FAIL — the decomposition does NOT cut z\n", c.name, size);
        fail = 1;
        continue;
      }

      // --- distributed ---
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      sd.computeVofCurvature();
      const auto stLoc = sd.vofCurvatureStats();
      long locCounts[7] = {stLoc.interfacial, stLoc.hf,        stLoc.hfMixed,   stLoc.hfFit,
                           stLoc.pv,          stLoc.pvReduced, stLoc.noEstimate};
      long sumCounts[7] = {0, 0, 0, 0, 0, 0, 0};
      MPI_Allreduce(locCounts, sumCounts, 7, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

      const std::vector<double> gk =
          gatherGlobal(sd.getVofCurvature(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gb =
          gatherGlobal(sd.getVofCurvatureBranch(), ox, oy, oz, lnx, lny, lnz, rank, size);

      // --- single-rank full-grid reference on rank 0 ---
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        ref.computeVofCurvature();
        const auto st = ref.vofCurvatureStats();
        const long refCounts[7] = {st.interfacial, st.hf,        st.hfMixed,   st.hfFit,
                                   st.pv,          st.pvReduced, st.noEstimate};
        double dk = 0.0, db = 0.0;
        const long nk = countBitDiff(gk, ref.getVofCurvature(), dk);
        const long nb = countBitDiff(gb, ref.getVofCurvatureBranch(), db);

        bool countsOk = true;
        for (int i = 0; i < 7; ++i)
          countsOk = countsOk && (sumCounts[i] == refCounts[i]);

        std::printf(
            "  [%-14s np=%d] interfacial %ld | HF %ld  HFdir %ld  HFfit %ld  PV %ld  PVred %ld"
            "  none %ld\n",
            c.name, size, refCounts[0], refCounts[1], refCounts[2], refCounts[3], refCounts[4],
            refCounts[5], refCounts[6]);
        std::printf("      kappa: %ld/%ld cells differ (max |d| %.3e)   branch: %ld differ"
                    "   census %s\n",
                    nk, (long)GCELLS, dk, nb, countsOk ? "MATCHES" : "DIFFERS");

        // Bitwise, with no tolerance: the cascade contains no reduction.
        if (nk != 0 || nb != 0 || !countsOk) {
          std::printf("      FAIL — the curvature cascade is decomposition-DEPENDENT\n");
          fail = 1;
        }
        // The fallback must actually be exercised, or the test gates only the HF stencil.
        if (refCounts[4] + refCounts[5] == 0) {
          std::printf("      FAIL — the PLIC-volumetric fallback never fired; this configuration "
                      "no longer gates its stencil reach\n");
          fail = 1;
        }
        if (refCounts[6] != 0) {
          std::printf("      FAIL — %ld interfacial cells got NO curvature estimate\n",
                      refCounts[6]);
          fail = 1;
        }
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("%s\n", fail ? "FAILED" : "PASSED");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
