// flow — VoF rung V6 (WO-V6), multi-rank: the DYNAMIC contact angle and hysteresis on a
// decomposition that CUTS the contact line.
//
// What can only break here (the V5b MPI gate exercises none of it):
//
//   1. THE CELL-CENTRE VELOCITY. `U_cl` is measured at the ANCHOR fluid cell, which the walk along
//      n_w can reach at ghost depth 3. The velocity is therefore built on the INNER region from the
//      solver's own faces and run through the colour block's ghost policy, exactly as the wall SDF
//      and the fluid-only normals are. Building it "wherever the faces are" instead is a
//      decomposition dependence that shows up ONLY here.
//
//   2. THE 3-POINT IN-WALL SMOOTHING. Pass B averages `U_cl` over the band cell and its two
//      neighbours along t_hat, i.e. it READS depth 3 while it WRITES depth <= 2. That is why pass A
//      computes at depth <= 2 and its two outputs (`U_cl` and the validity flag) are EXCHANGED
//      before pass B runs. Without the exchange the smoothed value at a block-boundary band cell is
//      a local guess, and it reaches the inner colour through the theta the fill imposes.
//
// Gates: the imposed angle, the apparent angle, U_cl and Ca_cl pointwise BITWISE at np 1/2/4
// against a full-grid single-rank reference; the theta band fill bitwise; and a coupled
// surface-tension run at the reduction floor.
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

using peclet::flow::IbmSolver;

// A flat SDF wall with normal +x at a QUARTER-integer x (WO-S finding 5), with a liquid slab whose
// two contact lines have OPPOSITE in-wall directions — one advancing and one receding under the
// single uniform wall-tangential velocity U. The ORB cuts z at np = 2 and x+z at np = 4, so both
// contact lines are cut.
static constexpr int NX = 16, NY = 8, NZ = 32;
static constexpr double XW = 4.25, THETA_E = 60.0, SLIP = 0.1, MU = 1.0, SIGMA = 1.0, UVEL = 0.02;
static constexpr double ZL = 8.5, ZR = 24.5;  // the liquid slab, in z
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

static double sdfAt(double x, double, double) {
  return x - XW;
}
static double colourAt(int, int, int iz) {
  const double lo = std::fmax((double)iz, ZL), hi = std::fmin((double)iz + 1.0, ZR);
  return std::fmax(hi - lo, 0.0);
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
static std::vector<double> sliceOf(const std::vector<double>& g, int ox, int oy, int oz, int lnx,
                                   int lny, int lnz) {
  return blockOf(
      [&](int x, int y, int z) {
        return g[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY];
      },
      ox, oy, oz, lnx, lny, lnz);
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
// NaN-safe: `fmax(m, NaN) == m`, so a NaN field would pass a naive bitwise gate (WO-R found this).
static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d <= m))
      m = d;
  }
  return m;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
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
      std::printf(
          "VOF DYNAMIC WETTING MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s  "
          "theta_e = %.0f deg, lambda = %.2f cells, u = %.3g\n",
          size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "",
          THETA_E, SLIP, UVEL);
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does not cut z, so no contact line is cut\n");
      fail = 1;
    }
    const std::vector<double> gSdf = blockOf(
        [](int x, int y, int z) { return sdfAt(x + 0.5, y + 0.5, z + 0.5); }, 0, 0, 0, NX, NY, NZ);

    auto setup = [&](IbmSolver& s, int aox, int aoy, int aoz, int anx, int any, int anz,
                     bool hyst) {
      s.setRho(1.0);
      s.setMu(MU);
      s.setSolid(sliceOf(gSdf, aox, aoy, aoz, anx, any, anz), true);
      s.enableVof();
      s.setVof(blockOf(colourAt, aox, aoy, aoz, anx, any, anz));
      s.setSurfaceTension(SIGMA);
      // a UNIFORM wall-tangential velocity: the slab's two contact lines then have opposite t_hat,
      // so one is advancing and one receding in the same field
      s.setVelocity(2, std::vector<double>((std::size_t)anx * any * anz, UVEL));
      s.setContactAngleDynamic(THETA_E, SLIP, MU, SIGMA);
      if (hyst)
        s.setContactAngleHysteresis(100.0, 80.0);
      s.setDt(0.15 * s.capillaryDt());
    };

    // ---- 1. the imposed angle, pointwise -------------------------------------------------------
    for (int hyst = 0; hyst < 2; ++hyst) {
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      setup(sd, ox, oy, oz, lnx, lny, lnz, hyst != 0);
      const auto cdLoc = sd.contactAngleDiagnostics();
      long loc[4] = {cdLoc.dynamicCells, cdLoc.pinnedCells, cdLoc.advancingCells,
                     cdLoc.recedingCells};
      long tot[4] = {0, 0, 0, 0};
      MPI_Allreduce(loc, tot, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      std::vector<double> g[5];
      for (int q = 0; q < 5; ++q)
        g[q] = gatherGlobal(sd.getVofDynamicField(q), ox, oy, oz, lnx, lny, lnz, rank, size);
      const auto gf = gatherGlobal(sd.getVofFilledColour(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        setup(ref, 0, 0, 0, NX, NY, NZ, hyst != 0);
        const auto rd = ref.contactAngleDiagnostics();
        double d[5];
        for (int q = 0; q < 5; ++q)
          d[q] = maxAbsDiff(g[q], ref.getVofDynamicField(q));
        const double df = maxAbsDiff(gf, ref.getVofFilledColour());
        std::printf(
            "  [%s np=%d] imposed %.3e  apparent %.3e  U_cl %.3e  Ca_cl %.3e  state %.3e  "
            "band fill %.3e (all bitwise required)\n"
            "        census dynamic/pinned/advancing/receding %ld/%ld/%ld/%ld "
            "(ref %ld/%ld/%ld/%ld)  mean imposed %.6f  mean apparent %.6f  max|Ca_cl| %.6e\n",
            hyst ? "hyst " : "dyn  ", size, d[0], d[1], d[2], d[3], d[4], df, tot[0], tot[1],
            tot[2], tot[3], rd.dynamicCells, rd.pinnedCells, rd.advancingCells, rd.recedingCells,
            rd.meanImposedTheta, rd.meanApparentTheta, rd.maxCaCl);
        for (int q = 0; q < 5; ++q)
          if (!(d[q] == 0.0)) {
            std::printf("  [np=%d] FAIL — dynamic field %d is not bitwise (%.3e)\n", size, q, d[q]);
            fail = 1;
          }
        if (!(df == 0.0)) {
          std::printf("  [np=%d] FAIL — the theta band fill is not bitwise (%.3e)\n", size, df);
          fail = 1;
        }
        if (tot[0] != rd.dynamicCells || tot[1] != rd.pinnedCells ||
            tot[2] != rd.advancingCells || tot[3] != rd.recedingCells) {
          std::printf("  [np=%d] FAIL — the state census differs from the reference\n", size);
          fail = 1;
        }
      }
    }

    // ---- 2. a coupled surface-tension run, at the reduction floor -------------------------------
    {
      const int steps = 20;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      setup(sd, ox, oy, oz, lnx, lny, lnz, false);
      const double lv0 = sd.vofDiagnostics().volume;
      double v0 = 0.0;
      MPI_Allreduce(&lv0, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      long maxIt = 0;
      for (int i = 0; i < steps; ++i) {
        sd.step();
        maxIt = std::max(maxIt, sd.lastPressureIterations());
      }
      const double lv1 = sd.vofDiagnostics().volume, ls = sd.vofDiagnostics().solidSumC;
      double v1 = 0.0, solid = 0.0;
      MPI_Allreduce(&lv1, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&ls, &solid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const auto gc = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const auto gi = gatherGlobal(sd.getVofDynamicField(0), ox, oy, oz, lnx, lny, lnz, rank, size);
      std::vector<double> gu[3];
      for (int c = 0; c < 3; ++c)
        gu[c] = gatherGlobal(sd.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        setup(ref, 0, 0, 0, NX, NY, NZ, false);
        for (int i = 0; i < steps; ++i)
          ref.step();
        double du = 0;
        for (int c = 0; c < 3; ++c)
          du = std::fmax(du, maxAbsDiff(gu[c], ref.getVelocity(c)));
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double di = maxAbsDiff(gi, ref.getVofDynamicField(0));
        const double drift = (v1 - v0) / v0;
        const double tol = size == 1 ? 0.0 : 1e-9;
        std::printf(
            "  [coupled np=%d] colour %.3e  velocity %.3e  imposed theta %.3e  drift %.3e  "
            "solid sum C %.3e  pressure iters max %ld\n",
            size, dc, du, di, drift, solid, maxIt);
        if (!(dc <= tol) || !(du <= tol) || !(di <= 1e-9)) {
          std::printf("  [coupled np=%d] FAIL — above tol %.1e\n", size, tol);
          fail = 1;
        }
        if (!(solid == 0.0) || !(std::fabs(drift) <= 1e-9)) {
          std::printf("  [coupled np=%d] FAIL — solid sum %.3e drift %.3e\n", size, solid, drift);
          fail = 1;
        }
      }
    }

    int g = 0;
    MPI_Allreduce(&fail, &g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    fail = g;
    if (rank == 0)
      std::printf(fail ? "VOF DYNAMIC WETTING MPI: FAILED\n"
                       : "VOF DYNAMIC WETTING MPI: all checks passed\n");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
