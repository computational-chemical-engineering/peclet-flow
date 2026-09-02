// flow — WO-V6b gate 6: the NAVIER SLIP wall closure under MPI.
//
// The single-rank gate (tests/study/wall_slip.py) is a slip POISEUILLE: a slit of width H between
// two flat SDF walls at QUARTER-INTEGER coordinates, driven by a uniform body force G, whose exact
// steady Stokes solution with slip length lambda at both walls is the QUADRATIC
//
//     u(z) = (G / 2 mu) * ( (z - z0)(z1 - z) + lambda H ),      H = z1 - z0,
//
// which satisfies the Robin condition exactly, so the Robust-Scaled closure must reproduce it.
// Here the same scene is run distributed and compared POINTWISE against a full-grid single-rank
// reference on rank 0 (the tests/kokkos_mpi pattern): np = 1 BITWISE, np > 1 to the reduction-order
// floor, at lambda = 0 (the no-slip control, which also gates that the lambda = 0 branch is inert
// under MPI) and lambda = 0.1 cells.
//
// THE DECOMPOSITION CUTS BOTH WALLS' CLOSURES. The ORB splits the 40-cell z axis into
// [0,16) [16,24) [24,32) [32,40) at np = 4, and the walls are placed at z = 16.25 and z = 32.25 so
// that each wall's cut cell and its SOLID neighbour land on opposite sides of a rank boundary:
// the low wall's cut cell k = 16 (theta = 0.25) is the first cell of block 1 with its solid
// neighbour k = 15 in block 0, and the high wall's cut cell k = 31 (theta = 0.75) is the last cell
// of block 2 with its solid neighbour k = 32 in block 3. Both Robin closures therefore read an
// exchanged ghost. At np = 2 (blocks [0,24) [24,40)) the cut falls in the fluid interior. The test
// prints the z-blocks and asserts that z is cut at all.
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

static constexpr int NX = 8, NY = 8, NZ = 40;
static constexpr double Z0 = 16.25, Z1 = 32.25;  // quarter-integer wall placement (WO-S finding 5)
static constexpr double MU = 1.0, RHO = 1.0, DT = 50.0, GFORCE = 1e-3;
static constexpr int STEPS = 200;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

static double sdfAt(int, int, int z) {
  const double zc = z + 0.5;
  return std::fmin(zc - Z0, Z1 - zc);
}

static std::vector<double> blockSdf(int oz, int lnx, int lny, int lnz) {
  std::vector<double> v((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        v[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = sdfAt(0, 0, z + oz);
  return v;
}

static void configure(IbmSolver& s, double lam, int oz, int lnx, int lny, int lnz) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);  // Stokes: the analytic profile is the steady Stokes solution
  s.setBodyForce(GFORCE, 0.0, 0.0);
  s.setPressurePcg(true, 400, 1e-12);
  s.setVelocityIterations(400);
  s.setSolid(blockSdf(oz, lnx, lny, lnz), /*cutcellPressure=*/true);
  if (lam > 0.0)
    s.setWallSlipLength(lam);
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
  double m = 0;  // NaN-propagating (WO-R2)
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

// max |u - u_exact| over the fluid column, relative to the peak of the exact profile.
static double analyticError(const std::vector<double>& u, double lam) {
  const double H = Z1 - Z0;
  double e = 0, s = 0;
  for (int z = 0; z < NZ; ++z) {
    const double zc = z + 0.5;
    if (zc <= Z0 || zc >= Z1)
      continue;
    const double ue = (GFORCE / (2.0 * MU)) * ((zc - Z0) * (Z1 - zc) + lam * H);
    const std::size_t i =
        (std::size_t)(NX / 2) + (std::size_t)(NY / 2) * NX + (std::size_t)z * NX * NY;
    e = std::fmax(e, std::fabs(u[i] - ue));
    s = std::fmax(s, std::fabs(ue));
  }
  return e / (s + 1e-300);
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
    if (rank == 0) {
      std::printf("WALL-SLIP MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s  "
                  "walls z = %.2f / %.2f  z-blocks:",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "", Z0, Z1);
      for (int r = 0; r < size; ++r)
        std::printf(" [%d,%d)", (int)dec.block((std::size_t)r).origin[2],
                    (int)(dec.block((std::size_t)r).origin[2] + dec.block((std::size_t)r).size[2]));
      std::printf("\n");
    }
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does NOT cut the walled z axis; this test exists "
                    "to gate the Navier closure across a rank boundary\n");
      fail = 1;
    }

    const double lambdas[] = {0.0, 0.1};
    for (double lam : lambdas) {
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, lam, oz, lnx, lny, lnz);
      for (int it = 0; it < STEPS; ++it)
        sd.step();
      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, lam, 0, NX, NY, NZ);
        for (int it = 0; it < STEPS; ++it)
          ref.step();
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double tol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
        const double aerr = analyticError(gu[0], lam);
        // The analytic tolerance is the FLOAT momentum-operator storage floor, not the scheme's
        // consistency error: the closure coefficients are stored in float, and the SAME scene at
        // lambda = 0 misses the (exact) no-slip parabola by 1.15e-6 relative. Gate both at 5e-6.
        const bool ok = du <= tol && aerr < 5e-6;
        std::printf("  [lambda=%.2f np=%d] du=%.3e (|u|=%.3e, tol %.1e) | analytic rel err "
                    "%.3e (tol 5.0e-06)  %s\n",
                    lam, size, du, umag, tol, aerr, ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("WALL-SLIP MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
