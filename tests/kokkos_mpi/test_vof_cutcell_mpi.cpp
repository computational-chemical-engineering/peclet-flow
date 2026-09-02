// flow — VoF rung V5a (WO-Q), multi-rank: the colour field transported THROUGH an immersed solid
// on a decomposition that CUTS the spheres.
//
// What can only break here (nothing below is exercised by `test_vof_twophase_mpi.cpp`, which is
// solid-free):
//
//   1. THE GEOMETRY OF THE COLOUR BLOCK. `eps` and the face openness are built on the solver's
//      G = 2 block and embedded into the advector's g = 3 block — the openness with the SAME
//      one-cell shift the face velocity uses — and then run through the colour field's own ghost
//      policy so that the classification at ghost DEPTH 3 is the owner's. That last step is not
//      cosmetic: the solid-band fill reads fluid neighbours at depth 3, so a locally-guessed
//      classification there would make the INNER result decomposition-dependent. Gated bitwise.
//
//   2. THE SOLID-BAND FILL. Three passes with a shrinking depth budget (pass k writes solid cells
//      at ghost depth <= 3-k and reads only fluid cells or cells filled in an EARLIER pass),
//      followed by a second exchange. The shrinking budget is exactly what makes every read fall
//      inside the halo the exchange has already made owner-consistent. Gated bitwise, pointwise,
//      through `vof_filled_colour()`.
//
//   3. THE OPENNESS-WEIGHTED FLUX ACROSS A RANK BOUNDARY. One flux per face, formed once, so the
//      Weymouth-Yue telescoping — hence conservation — holds across the boundary too. The
//      KINEMATIC configuration below gates the transported colour BITWISE at np 1/2/4 by handing
//      every rank the SAME velocity field: each rank builds the full-grid single-rank solver,
//      runs it to a Stokes steady state (a deterministic computation, identical on every rank) and
//      scatters that velocity into its own block. Without that the velocity itself would carry the
//      pressure solve's reduction-order floor and no bitwise statement about the ADVECTION would be
//      possible.
//
//   4. The coupled step (`enable_vof` + closures + gravity through the packing) at the reduction
//      floor, with the colour volume drift measured against the projection's own divergence.
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

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

// Four periodic spheres, placed so the ORB's z cut at np = 2 and 4 goes THROUGH them.
static double sdfAt(double x, double y, double z) {
  const double cx[4] = {4.0, 11.0, 5.0, 12.0}, cy[4] = {4.0, 11.0, 12.0, 4.0},
               cz[4] = {8.0, 16.0, 24.0, 30.0}, r = 4.5;
  double best = 1e30;
  for (int s = 0; s < 4; ++s)
    for (int a = -1; a <= 1; ++a)
      for (int b = -1; b <= 1; ++b)
        for (int c = -1; c <= 1; ++c) {
          const double dx = x - (cx[s] + a * NX), dy = y - (cy[s] + b * NY),
                       dz = z - (cz[s] + c * NZ);
          const double d = std::sqrt(dx * dx + dy * dy + dz * dz) - r;
          best = d < best ? d : best;
        }
  return best;
}
static double colourAt(int, int, int z) {
  return z < NZ / 2 ? 1.0 : 0.0;
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
static double sumOf(const std::vector<double>& a) {
  double s = 0;
  for (double v : a)
    s += v;
  return s;
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
      std::printf("VOF CUTCELL MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does not cut z, so no sphere is cut\n");
      fail = 1;
    }

    const std::vector<double> gSdf = blockOf(
        [](int x, int y, int z) { return sdfAt(x + 0.5, y + 0.5, z + 0.5); }, 0, 0, 0, NX, NY, NZ);

    // ---- the deterministic Stokes velocity, computed identically on EVERY rank ----------------
    // A full-grid single-rank solver: the same arithmetic on every rank, so the velocity that
    // drives the kinematic advection carries no reduction-order dependence and the colour gate can
    // be BITWISE rather than "at the floor".
    std::vector<double> gU[3];
    std::vector<double> refColour;
    double refDrift = 0.0, refDiv = 0.0;
    const int kinSteps = 40;
    {
      IbmSolver ref(NX, NY, NZ);
      ref.setRho(1.0);
      ref.setMu(0.2);
      ref.setDt(1.0);
      ref.setBodyForce(2e-3, 1e-3, 5e-4);
      ref.setSolid(gSdf, true);
      for (int i = 0; i < 40; ++i)
        ref.step();
      refDiv = ref.maxOpenDivergence();
      for (int c = 0; c < 3; ++c)
        gU[c] = ref.getVelocity(c);
      ref.enableVof();
      ref.setVof(blockOf(colourAt, 0, 0, 0, NX, NY, NZ));
      const double v0 = ref.vofDiagnostics().volume;
      const double dtA = 0.2 / ref.vofMaxCourant();
      for (int i = 0; i < kinSteps; ++i)
        ref.advectVofKinematic(dtA);
      const auto d1 = ref.vofDiagnostics();
      refDrift = (d1.volume - v0) / v0;
      refColour = ref.getVof();
      if (rank == 0)
        std::printf(
            "  reference (full grid, every rank): max|div(open u)| %.3e, %d kinematic steps at "
            "dt %.4g, drift %.3e, clipped %.3e\n",
            refDiv, kinSteps, dtA, refDrift, d1.clippedVolume);
    }

    // ---- distributed kinematic run on the SAME velocity ---------------------------------------
    {
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      sd.setRho(1.0);
      sd.setMu(0.2);
      sd.setDt(1.0);
      sd.setSolid(sliceOf(gSdf, ox, oy, oz, lnx, lny, lnz), true);
      sd.enableVof();
      // geometry, pointwise: eps, the three face openness fields and the classification
      double dg = 0.0;
      {
        IbmSolver r2(NX, NY, NZ);
        r2.setRho(1.0);
        r2.setMu(0.2);
        r2.setDt(1.0);
        r2.setSolid(gSdf, true);
        r2.enableVof();
        for (int w = 0; w < 5; ++w) {
          const auto g = gatherGlobal(sd.getVofGeometry(w), ox, oy, oz, lnx, lny, lnz, rank, size);
          if (rank == 0)
            dg = std::fmax(dg, maxAbsDiff(g, r2.getVofGeometry(w)));
        }
      }
      for (int c = 0; c < 3; ++c)
        sd.setField(c == 0 ? "u" : (c == 1 ? "v" : "w"), sliceOf(gU[c], ox, oy, oz, lnx, lny, lnz));
      sd.setVof(blockOf(colourAt, ox, oy, oz, lnx, lny, lnz));
      // the solid-band fill, pointwise
      double dfill = 0.0;
      {
        const auto g = gatherGlobal(sd.getVofFilledColour(), ox, oy, oz, lnx, lny, lnz, rank, size);
        if (rank == 0) {
          IbmSolver r2(NX, NY, NZ);
          r2.setRho(1.0);
          r2.setMu(0.2);
          r2.setDt(1.0);
          r2.setSolid(gSdf, true);
          r2.enableVof();
          r2.setVof(blockOf(colourAt, 0, 0, 0, NX, NY, NZ));
          dfill = maxAbsDiff(g, r2.getVofFilledColour());
        }
      }
      const double divD = sd.maxOpenDivergence();
      const double locV0 = sd.vofDiagnostics().volume;
      double v0 = 0.0;
      MPI_Allreduce(&locV0, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      double cflLoc = sd.vofMaxCourant();
      const double dtA = 0.2 / cflLoc;
      for (int i = 0; i < kinSteps; ++i)
        sd.advectVofKinematic(dtA);
      const double locV1 = sd.vofDiagnostics().volume;
      double v1 = 0.0;
      MPI_Allreduce(&locV1, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const auto gc = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        const double dc = maxAbsDiff(gc, refColour);
        const double drift = (v1 - v0) / v0;
        std::printf(
            "  [kinematic np=%d] geometry %.3e  band fill %.3e  colour %.3e  "
            "drift %.3e (ref %.3e)  max|div| %.3e\n",
            size, dg, dfill, dc, drift, refDrift, divD);
        if (!(dg == 0.0) || !(dfill == 0.0) || !(dc == 0.0)) {
          std::printf("  [kinematic np=%d] FAIL — not bitwise\n", size);
          fail = 1;
        }
        if (!(std::fabs(drift) <= 1e-11)) {
          std::printf("  [kinematic np=%d] FAIL — colour volume drift %.3e\n", size, drift);
          fail = 1;
        }
      }
    }

    // ---- coupled draining through the packing (reduction floor) --------------------------------
    {
      const int steps = 30;
      const double ratio = 10.0, grav = 2e-3;
      auto setup = [&](IbmSolver& s, int aox, int aoy, int aoz, int anx, int any, int anz) {
        s.setRho(1.0);
        s.setMu(0.05);
        s.setDt(0.5);
        s.setAdvection(true);
        s.setSolid(sliceOf(gSdf, aox, aoy, aoz, anx, any, anz), true);
        s.setVof(blockOf(colourAt, aox, aoy, aoz, anx, any, anz));
        s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                           {1.0, ratio - 1.0});
        s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {0.05, 0.45});
        s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "",
                           {0.0, -grav});
      };
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      setup(sd, ox, oy, oz, lnx, lny, lnz);
      const double lv0 = sd.vofDiagnostics().volume;
      double v0 = 0.0;
      MPI_Allreduce(&lv0, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      long maxIt = 0;
      for (int i = 0; i < steps; ++i) {
        sd.step();
        maxIt = std::max(maxIt, sd.lastPressureIterations());
      }
      const double lv1 = sd.vofDiagnostics().volume;
      double v1 = 0.0;
      MPI_Allreduce(&lv1, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double lsolid = sd.vofDiagnostics().solidSumC;
      double solid = 0.0;
      MPI_Allreduce(&lsolid, &solid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double divD = sd.maxOpenDivergence();
      const auto gc = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      std::vector<double> gu[3];
      for (int c = 0; c < 3; ++c)
        gu[c] = gatherGlobal(sd.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        setup(ref, 0, 0, 0, NX, NY, NZ);
        for (int i = 0; i < steps; ++i)
          ref.step();
        double du = 0;
        for (int c = 0; c < 3; ++c)
          du = std::fmax(du, maxAbsDiff(gu[c], ref.getVelocity(c)));
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double drift = (v1 - v0) / v0;
        const double tol = size == 1 ? 0.0 : 1e-9;
        std::printf(
            "  [coupled  np=%d] colour %.3e  velocity %.3e  drift %.3e  solid sum C %.3e  "
            "max|div| %.3e  pressure iters max %ld\n",
            size, dc, du, drift, solid, divD, maxIt);
        if (!(dc <= tol) || !(du <= tol)) {
          std::printf("  [coupled  np=%d] FAIL — colour %.3e velocity %.3e above tol %.1e\n", size,
                      dc, du, tol);
          fail = 1;
        }
        if (!(solid == 0.0)) {
          std::printf("  [coupled  np=%d] FAIL — colour in solid cells %.3e\n", size, solid);
          fail = 1;
        }
        if (!(std::fabs(drift) <= 1e-8)) {
          std::printf("  [coupled  np=%d] FAIL — colour drift %.3e\n", size, drift);
          fail = 1;
        }
      }
    }
    int g = 0;
    MPI_Allreduce(&fail, &g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    fail = g;
    if (rank == 0)
      std::printf(fail ? "VOF CUTCELL MPI: FAILED\n" : "VOF CUTCELL MPI: all checks passed\n");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
