// Telescoping across a variable-density OUTFLOW (the WO-R2 outflow-coefficient path): the
// hierarchy with a telescope FORCED at level 1 must reproduce the in-place hierarchy on the same
// problem -- a sharp density jump advected out of the domain through a +z outflow with a -z inflow.
// The coarse boundary coefficient across the telescope point is coarsened from the merged stage
// (teleGatherPlane carried the high-side outflow plane), so the two hierarchies build the same
// coarse operators and the solves agree to the pressure driver's tolerance.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using IbmSolver = peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 4;
static constexpr double RHO0 = 1.0, RATIO = 100.0, MU = 0.1, DT = 0.5, WIN = 0.05;

static double rhoAt(int gz) { return (gz < N / 2) ? RATIO * RHO0 : RHO0; }

static void run(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz, int forceLevel,
                bool telescope, std::vector<double>& u, std::vector<double>& p, int& teleCount) {
  s.setRho(RHO0);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);  // x/y walls
  s.setDomainBc(4, 2, 0, 0, WIN);  // -z inflow
  s.setDomainBc(5, 3, 0, 0, 0);    // +z outflow
  s.setPressureLevels(4);
  s.setPressurePcg(true, 400, 1e-11);
  s.setPressureTelescope(telescope);
  if (telescope)
    s.setPressureTelescopeForceLevel(forceLevel);
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setDensityMode(true);
  std::vector<double> rho((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        rho[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = rhoAt(z + oz);
  s.setField("rho", rho);
  s.exchangeField("rho");
  for (int it = 0; it < STEPS; ++it)
    s.step();
  u = s.getVelocity(2);
  p = s.getPressure();
  teleCount = s.pressureTelescopeCount();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    peclet::core::decomp::BlockDecomposer<3> dec =
        peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    std::vector<double> uT, pT, uI, pI;
    int cT = 0, cI = 0;
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(N, N, N, MPI_COMM_WORLD);
      run(s, ox, oy, oz, lnx, lny, lnz, 1, true, uT, pT, cT);
    }
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(N, N, N, MPI_COMM_WORLD);
      run(s, ox, oy, oz, lnx, lny, lnz, -1, false, uI, pI, cI);
    }
    double du = 0, dp = 0, um = 0, pm = 0;
    for (std::size_t i = 0; i < uT.size(); ++i) {
      du = std::max(du, std::fabs(uT[i] - uI[i]));
      um = std::max(um, std::fabs(uI[i]));
    }
    for (std::size_t i = 0; i < pT.size(); ++i) {
      dp = std::max(dp, std::fabs(pT[i] - pI[i]));
      pm = std::max(pm, std::fabs(pI[i]));
    }
    double l[4] = {du, dp, um, pm}, g[4];
    MPI_Allreduce(l, g, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int cTg = 0;
    MPI_Allreduce(&cT, &cTg, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    const double relU = g[0] / (g[2] + 1e-300), relP = g[1] / (g[3] + 1e-300);
    const double tol = 1e-8;  // the PCG tolerance's footprint on two different-but-equivalent hierarchies
    const bool teleOk = (size == 1) || cTg >= 1;  // np=1 has one block: nothing to merge
    if (rank == 0)
      std::printf("  telescoped(force L1) vs in-place: rel du %.2e  rel dp %.2e (tol %.0e)  "
                  "telescopes %d  |u| %.3e |p| %.3e  (np=%d)\n",
                  relU, relP, tol, cTg, g[2], g[3], size);
    if (!(relU <= tol) || !(relP <= tol) || !teleOk)
      fail = 1;
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): variable-density outflow across a telescope point == in-place\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
