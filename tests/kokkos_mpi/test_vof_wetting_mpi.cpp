// flow — VoF rung V5b (WO-S), multi-rank: the theta-consistent solid-band fill on a decomposition
// that CUTS both the wall and the drop.
//
// What can only break here (WO-Q's cut-cell MPI gate is the neutral fill and does not exercise any
// of it):
//
//   1. THE WALL NORMAL. `n_w` is a central difference of the SDF ON THE COLOUR BLOCK, evaluated at
//      solid cells out to ghost depth 2 — i.e. it reads depth 3. The SDF therefore has to go
//      through the colour field's own ghost policy after it is embedded, exactly as the
//      classification does (WO-Q finding 5), or the wall normal in the ghost band is a local guess
//      and the INNER fill inherits it through passes 2-3.
//
//   2. THE FLUID-ONLY YOUNGS NORMAL. The theta pass reads it at the ANCHOR fluid cell, which the
//      walk along `n_w` can reach at ghost depth 3 — one deeper than a 3^3 stencil can be evaluated
//      on a g = 3 block. It is therefore built on the INNER region only and exchanged, which is
//      what makes it the owner's value everywhere. Building it "wherever the stencil fits" instead
//      would be a decomposition dependence that shows up ONLY here.
//
//   3. The theta pass writes solid cells at ghost depth <= 2 like WO-Q's pass 1, so the shrinking
//      depth budget and the second exchange still have to compose.
//
// Gates: the filled colour (`vof_filled_colour`) pointwise BITWISE at np 1/2/4 against a full-grid
// single-rank reference, the band census identical, and a coupled surface-tension run at the
// reduction floor.
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

// The wall normal is +x (a flat SDF wall at a FRACTIONAL x, so the wall cells are genuinely cut;
// the quarter-cell offset rather than a half is deliberate — at exactly k + 1/2 the tangential MAC
// faces of the wall cell sit ON the SDF zero level and `buildOpenness` closes them, which isolates
// that cell tangentially, see the WO-S findings)
// and the drop sits on it centred at (xw, 8, 16). The ORB cuts z at np = 2 and x+z at np = 4, so
// the contact LINE (a circle in the plane x = xw) is cut at both.
static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr double XW = 5.25, RD = 6.0, THETA = 60.0;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

static double sdfAt(double x, double, double) {
  return x - XW;
}
// The hemispherical cap, by the same 4^3 subsampling the fluid fraction uses (C is a fraction of
// the FLUID volume).
static double colourAt(int ix, int iy, int iz) {
  int in = 0, tot = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      for (int c = 0; c < 4; ++c) {
        const double px = ix + (a + 0.5) / 4.0, py = iy + (b + 0.5) / 4.0,
                     pz = iz + (c + 0.5) / 4.0;
        if (px < XW)
          continue;
        ++tot;
        const double dx = px - XW, dy = py - 8.0, dz = pz - 16.0;
        if (dx * dx + dy * dy + dz * dz < RD * RD)
          ++in;
      }
  return tot ? (double)in / tot : 0.0;
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
static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  // WO-R2: NaN-PROPAGATING. `std::fmax(m, NaN) == m`, so the obvious loop returns 0.000e+00 for a
  // field that has gone entirely NaN and every bitwise gate built on it passes (WO-R found this on
  // a drained open-boundary run). A non-finite difference must fail, so return it.
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;  // NaN
    m = std::fmax(m, d);
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
          "VOF WETTING MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s  "
          "theta = %.0f deg\n",
          size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "",
          THETA);
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does not cut z, so the contact line is not cut\n");
      fail = 1;
    }
    const std::vector<double> gSdf = blockOf(
        [](int x, int y, int z) { return sdfAt(x + 0.5, y + 0.5, z + 0.5); }, 0, 0, 0, NX, NY, NZ);

    auto setup = [&](IbmSolver& s, int aox, int aoy, int aoz, int anx, int any, int anz,
                     bool angle) {
      s.setRho(1.0);
      s.setMu(0.05);
      s.setSolid(sliceOf(gSdf, aox, aoy, aoz, anx, any, anz), true);
      s.enableVof();
      s.setVof(blockOf(colourAt, aox, aoy, aoz, anx, any, anz));
      s.setSurfaceTension(1.0);
      if (angle)
        s.setContactAngle(THETA);
      s.setDt(0.15 * s.capillaryDt());
    };

    // ---- 1. the theta fill, pointwise ---------------------------------------------------------
    {
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      setup(sd, ox, oy, oz, lnx, lny, lnz, true);
      const auto cdLoc = sd.contactAngleDiagnostics();
      long loc[5] = {cdLoc.contactCells, cdLoc.neighbourCells, cdLoc.pureCells, cdLoc.parallelCells,
                     cdLoc.neutralCells};
      long tot[5] = {0, 0, 0, 0, 0};
      MPI_Allreduce(loc, tot, 5, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      const auto g = gatherGlobal(sd.getVofFilledColour(), ox, oy, oz, lnx, lny, lnz, rank, size);
      // the neutral fill of the SAME scene, for the ablation number
      const auto gn0 = [&] {
        IbmSolver s2(lnx, lny, lnz);
        s2.initMpi(dec, MPI_COMM_WORLD);
        setup(s2, ox, oy, oz, lnx, lny, lnz, false);
        return gatherGlobal(s2.getVofFilledColour(), ox, oy, oz, lnx, lny, lnz, rank, size);
      }();
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        setup(ref, 0, 0, 0, NX, NY, NZ, true);
        const auto rd = ref.contactAngleDiagnostics();
        const double d = maxAbsDiff(g, ref.getVofFilledColour());
        const double dn = maxAbsDiff(g, gn0);
        std::printf(
            "  [fill np=%d] theta band fill %.3e (bitwise required)  |  band census "
            "theta/neighbour/pure/parallel/neutral %ld/%ld/%ld/%ld/%ld "
            "(ref %ld/%ld/%ld/%ld/%ld)  mean apparent %.3f deg  |  "
            "max |theta-fill - neutral-fill| %.3e\n",
            size, d, tot[0], tot[1], tot[2], tot[3], tot[4], rd.contactCells, rd.neighbourCells,
            rd.pureCells, rd.parallelCells, rd.neutralCells, rd.meanApparentAngle, dn);
        if (!(d == 0.0)) {
          std::printf("  [fill np=%d] FAIL — the theta band fill is not bitwise\n", size);
          fail = 1;
        }
        if (tot[0] != rd.contactCells || tot[1] != rd.neighbourCells || tot[2] != rd.pureCells ||
            tot[3] != rd.parallelCells || tot[4] != rd.neutralCells) {
          std::printf("  [fill np=%d] FAIL — the band census differs from the reference\n", size);
          fail = 1;
        }
      }
    }

    // ---- 2. a coupled surface-tension run, at the reduction floor -------------------------------
    {
      const int steps = 25;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      setup(sd, ox, oy, oz, lnx, lny, lnz, true);
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
      std::vector<double> gu[3];
      for (int c = 0; c < 3; ++c)
        gu[c] = gatherGlobal(sd.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        setup(ref, 0, 0, 0, NX, NY, NZ, true);
        for (int i = 0; i < steps; ++i)
          ref.step();
        double du = 0;
        for (int c = 0; c < 3; ++c)
          du = std::fmax(du, maxAbsDiff(gu[c], ref.getVelocity(c)));
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double drift = (v1 - v0) / v0;
        const double tol = size == 1 ? 0.0 : 1e-9;
        std::printf(
            "  [coupled np=%d] colour %.3e  velocity %.3e  drift %.3e  solid sum C %.3e  "
            "pressure iters max %ld\n",
            size, dc, du, drift, solid, maxIt);
        if (!(dc <= tol) || !(du <= tol)) {
          std::printf("  [coupled np=%d] FAIL — colour %.3e velocity %.3e above tol %.1e\n", size,
                      dc, du, tol);
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
      std::printf(fail ? "VOF WETTING MPI: FAILED\n" : "VOF WETTING MPI: all checks passed\n");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
