// Immersed solid TOGETHER with inflow/outflow domain BCs, DISTRIBUTED (SCALING_ISSUES #3; the
// single-rank sibling is tests/kokkos/test_openbc_solid.cpp). Every part of the fix is rank-aware
// and had to be: the SDF ghost band outside a non-periodic face is repaired only on the rank that
// owns that global face, and the boundary-face APERTURE PLANE is a ghost index that the openness
// halo exchange, the per-level fill and (across a telescope point) the plane gather all wrap over.
// That last one is why this test earns its keep -- it caught a third defect, visible AT np = 1,
// which the fix's first two halves had made matter: the exchange handed the outlet the INLET's
// aperture (1.8e-02 / 6.6e-01 from single-rank; 0.00e+00 after).
//
// So the gate is the usual one: the distributed run reproduces the single-rank run on the block it
// owns -- for a bed clear of the open faces, beds cutting each of them, and a bed cutting a wall
// (the control that says the SDF ghost extension itself is rank-consistent). The geometry is a
// local computation, so the openness fields must agree EXACTLY, not to a tolerance.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using IbmSolver = peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 4, PMAXIT = 200;
static constexpr double RHO = 1.0, MU = 1.0, UIN = 1.0, DT = 0.5, R = 0.27 * N;

static std::vector<double> sphereSdf(double cx) {
  std::vector<double> sdf((std::size_t)N * N * N);
  const double cy = 0.5 * (N - 1), cz = 0.5 * (N - 1);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x - cx, dy = y - cy, dz = z - cz;
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return sdf;
}

static std::vector<double> sphereSdfY(double cx) {  // centred ON the -y wall plane
  std::vector<double> sdf((std::size_t)N * N * N);
  const double cy = -0.5, cz = 0.5 * (N - 1);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x - cx, dy = y - cy, dz = z - cz;
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return sdf;
}

static void configure(IbmSolver& s) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);
  s.setVelocityIterations(200);
  s.setVelocityResidualTolerance(1e-10);
  s.setPressureLevels(3);
  s.setPressurePcg(true, PMAXIT, 1e-10);
  s.setDomainBc(0, 2, UIN, 0, 0);  // -x inflow
  s.setDomainBc(1, 3, 0, 0, 0);    // +x outflow
  for (int f = 2; f < 6; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);  // y/z no-slip walls
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  {
    const double centres[4] = {0.5 * N, N - 0.5, -0.5, 0.5 * N};
    const int cutWallY[4] = {0, 0, 0, 1};  // case 3: the sphere is cut by the -y WALL instead
    const char* label[4] = {"clear of the open faces", "cutting the OUTFLOW face",
                            "cutting the INFLOW face", "cutting the -y WALL"};
    peclet::core::decomp::BlockDecomposer<3> dec =
        peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    for (int cs = 0; cs < 4; ++cs) {
      const std::vector<double> gsdf =
          cutWallY[cs] ? sphereSdfY(centres[cs]) : sphereSdf(centres[cs]);
      std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
      for (int z = 0; z < lnz; ++z)
        for (int y = 0; y < lny; ++y)
          for (int x = 0; x < lnx; ++x)
            lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
                gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N +
                     (std::size_t)(z + oz) * N * N];

      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(N, N, N, MPI_COMM_WORLD);
      configure(sd);
      sd.setSolid(lsdf, true);
      long itersD = 0;
      for (int it = 0; it < STEPS; ++it) {
        sd.step();
        itersD = std::max(itersD, sd.lastPressureIterations());
      }
      const double divD = sd.maxOpenDivergenceProjected();
      const std::vector<double> ud = sd.getVelocity(0);
      std::vector<double> od[3] = {sd.getOpenness(0), sd.getOpenness(1), sd.getOpenness(2)};

      std::vector<double> ur((std::size_t)N * N * N);
      std::vector<double> orr[3] = {std::vector<double>((std::size_t)N * N * N),
                                    std::vector<double>((std::size_t)N * N * N),
                                    std::vector<double>((std::size_t)N * N * N)};
      long itersR = 0;
      double divR = 0;
      if (rank == 0) {
        IbmSolver sr(N, N, N);
        configure(sr);
        sr.setSolid(gsdf, true);
        for (int it = 0; it < STEPS; ++it) {
          sr.step();
          itersR = std::max(itersR, sr.lastPressureIterations());
        }
        divR = sr.maxOpenDivergenceProjected();
        ur = sr.getVelocity(0);
        orr[0] = sr.getOpenness(0);
        orr[1] = sr.getOpenness(1);
        orr[2] = sr.getOpenness(2);
      }
      MPI_Bcast(ur.data(), (int)ur.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&itersR, 1, MPI_LONG, 0, MPI_COMM_WORLD);
      MPI_Bcast(&divR, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      for (int q = 0; q < 3; ++q)
        MPI_Bcast(orr[q].data(), (int)orr[q].size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      {
        double dop = 0;
        for (int z = 0; z < lnz; ++z)
          for (int y = 0; y < lny; ++y)
            for (int x = 0; x < lnx; ++x)
              for (int q = 0; q < 3; ++q) {
                const std::size_t li =
                    (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
                const std::size_t gi = (std::size_t)(x + ox) + (std::size_t)(y + oy) * N +
                                       (std::size_t)(z + oz) * N * N;
                dop = std::max(dop, std::fabs(od[q][li] - orr[q][gi]));
              }
        double gdop = 0;
        MPI_Allreduce(&dop, &gdop, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (gdop != 0.0) {  // the geometry is a local computation: it must agree EXACTLY
          if (rank == 0)
            std::printf("   openness dist-vs-single max diff %.3e (must be 0)\n", gdop);
          fail = 1;
        }
      }

      double d = 0, umax = 0;
      for (int z = 0; z < lnz; ++z)
        for (int y = 0; y < lny; ++y)
          for (int x = 0; x < lnx; ++x) {
            const std::size_t li =
                (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
            const std::size_t gi =
                (std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N;
            d = std::max(d, std::fabs(ud[li] - ur[gi]));
            umax = std::max(umax, std::fabs(ur[gi]));
          }
      double g[2] = {d, umax}, gg[2];
      MPI_Allreduce(g, gg, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const double rel = gg[0] / gg[1];
      const double tol = (size == 1) ? 1e-13 : 1e-7;  // np>1: the PCG's reduction-order floor
      if (rank == 0)
        std::printf(
            "[openbc-solid-mpi] np=%d  sphere %-24s  dist-vs-single rel %.2e (tol %.0e)  "
            "iters %ld/%ld  proj max|div| %.2e/%.2e\n",
            size, label[cs], rel, tol, itersD, itersR, divD, divR);
      if (!(rel <= tol))
        fail = 1;
      if (itersD >= PMAXIT || itersR >= PMAXIT)  // the defect capped MG-PCG on the cut beds
        fail = 1;
      if (!(divD < 1e-6) || !(divR < 1e-6))  // converged; the defect sat at 1.0 and 6.7e-03
        fail = 1;
    }
  }
  int gfail = 0;
  MPI_Allreduce(&fail, &gfail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("[openbc-solid-mpi] %s\n", gfail ? "FAILED" : "OK");
  Kokkos::finalize();
  MPI_Finalize();
  return gfail;
}
