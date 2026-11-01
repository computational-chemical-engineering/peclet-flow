// flow — ctest `cg_guard_mpi` (np 1, 2, 4): the exit check of the two CG pressure drivers (Q18).
//
// Under the trilinear V-cycle (set_pressure_vcycle_symmetric(False), the pre-A3 preconditioner),
// above a density ratio of ~1e3 that preconditioner turns indefinite and MG-PCG / flexible CG
// either cap or stop "converged" on their recurrence residual with a field that is not a
// projection; the default SPD V-cycle (A3) keeps them converged there. The check recomputes the
// TRUE residual once per CG solve and counts a solve whose true residual exceeds max(10 rtol, 1e-8)
// r0, or that capped or broke down (diagnostics.num_pressure_unconverged_solves); it never changes
// x, the iteration or the stop. Case: a 16^3 box, walls at +-y, periodic x and z, a tilted
// two-layer VoF colour (a 22.5 degree line in x-y, extruded in z; frozen: re-set after every step),
// rho and mu by LinearMix from C, Stokes, a fixed pseudo-random initial velocity, 3 steps, rtol
// 1e-10, cap 400 (the reproduction of validation/highratio/q18, cell units). Gates, at every np
// (each rank checks its own copy; the allreduced MIN and MAX must agree):
//   G1  FCG at ratio 1e5 on the trilinear V-cycle: the count is >= 1 and the same on every rank;
//       the last checked true residual is bitwise the same on every rank;
//   G2  Chebyshev at ratio 1e4 (the variable-density default): the count stays 0;
//   G3  PCG at ratio 1 (uniform density): the count stays 0 and every true residual is <= 1
//       (the exit meets its own tolerance);
//   G4  (rank 0, a single-rank solver, the device-resident PCG and the host FCG) G1 and G3 again;
//   G5  FCG at ratio 1e4 on the default SPD V-cycle: the count stays 0 (no false alarm).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int N = 16, STEPS = 3, MAXIT = 400;
static constexpr double RTOL = 1e-10;

enum Driver { kPcg, kFcg, kCheb };

static double colourAt(int x, int y) {  // the fraction below a tilted line, 16 x 16 sub-samples
  const double th = 22.5 * M_PI / 180.0, nx = std::cos(th), ny = std::sin(th);
  const double x0 = 0.5 * N + 0.3 * nx, y0 = 0.5 * N + 0.3 * ny;
  int in = 0;
  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i)
      in += ((x + (i + 0.5) / 16 - x0) * nx + (y + (j + 0.5) / 16 - y0) * ny < 0.0) ? 1 : 0;
  return in / 256.0;
}
static double noiseAt(int c, int x, int y, int z) {  // decomposition-independent, in [-1, 1]
  unsigned h = 2166136261u;
  for (int v : {c, x, y, z})
    h = (h ^ (unsigned)v) * 16777619u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return 2.0 * (h & 0xffffff) / double(0xffffff) - 1.0;
}

struct Result {
  long count = 0;
  double lastTrue = 0.0, maxTrue = 0.0;
};

static Result run(IbmSolver& s, Driver d, double ratio, int ox, int oy, int oz, int lnx, int lny,
                  int lnz, bool trilinear = false) {
  const double muRatio = ratio / 10.0;
  s.setDt(1e4 * (ratio > 1.0 ? ratio / muRatio : 1.0));
  for (int f = 2; f < 4; ++f)
    s.setDomainBc(f, 1, 0.0, 0.0, 0.0);  // walls at -y, +y
  s.setRho(1.0);
  s.setMu(1.0);
  const std::size_t n = (std::size_t)lnx * lny * lnz;
  s.setPressureGeometry(std::vector<double>(n, 1.0));
  std::vector<double> c(n);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        c[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            colourAt(x + ox, y + oy);
  s.enableVof();
  s.setVof(c);
  s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, muRatio - 1.0});
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, ratio - 1.0});
  s.setAdvection(false);
  if (trilinear)
    s.setPressureVcycleSymmetric(false);
  if (d == kPcg)  // LAST: the density model re-selects Chebyshev
    s.setPressurePcg(true, MAXIT, RTOL);
  else if (d == kFcg)
    s.setPressureFcg(true, MAXIT, RTOL);
  else
    s.setPressureChebyshev(true, MAXIT, RTOL);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> v(n);
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          v[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
              1e-12 * noiseAt(k, x + ox, y + oy, z + oz);
    s.setVelocity(k, v);
  }
  Result r;
  for (int k = 0; k < STEPS; ++k) {
    s.step();
    s.setVof(c);  // frozen colour
    r.lastTrue = s.lastPressureTrueResidual();
    r.maxTrue = std::fmax(r.maxTrue, r.lastTrue);
  }
  r.count = s.numPressureUnconvergedSolves();
  return r;
}

// The same value on every rank? (MIN == MAX over the communicator, bitwise for the double.)
static bool sameOnAll(double v, MPI_Comm comm) {
  double lo = 0, hi = 0;
  MPI_Allreduce(&v, &lo, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&v, &hi, 1, MPI_DOUBLE, MPI_MAX, comm);
  return std::memcmp(&lo, &hi, sizeof(double)) == 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    if (rank == 0)
      std::printf("CG GUARD MPI np=%d  grid %d^3  block %dx%dx%d  rtol %.0e  cap %d\n", size, N,
                  lnx, lny, lnz, RTOL, MAXIT);
    auto check = [&](bool ok, const char* what, const Result& r) {
      int all = 0, mine = ok ? 1 : 0;
      MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("  %-44s count %ld  last true %.3e  max true %.3e  %s\n", what, r.count,
                    r.lastTrue, r.maxTrue, all ? "OK" : "FAIL");
      fails += all ? 0 : 1;
    };
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(dec, MPI_COMM_WORLD);
      const Result r = run(s, kFcg, 1e5, ox, oy, oz, lnx, lny, lnz, true);
      const bool same =
          sameOnAll((double)r.count, MPI_COMM_WORLD) && sameOnAll(r.lastTrue, MPI_COMM_WORLD);
      check(r.count >= 1 && same, "G1 FCG ratio 1e5 trilinear: flagged, same", r);
    }
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(dec, MPI_COMM_WORLD);
      const Result r = run(s, kCheb, 1e4, ox, oy, oz, lnx, lny, lnz);
      check(r.count == 0 && sameOnAll((double)r.count, MPI_COMM_WORLD),
            "G2 Chebyshev ratio 1e4: silent", r);
    }
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(dec, MPI_COMM_WORLD);
      const Result r = run(s, kPcg, 1.0, ox, oy, oz, lnx, lny, lnz);
      check(r.count == 0 && r.maxTrue <= 1.0 && sameOnAll(r.maxTrue, MPI_COMM_WORLD),
            "G3 PCG uniform rho: silent, true <= tol", r);
    }
    {
      IbmSolver s(lnx, lny, lnz);
      s.initMpi(dec, MPI_COMM_WORLD);
      const Result r = run(s, kFcg, 1e4, ox, oy, oz, lnx, lny, lnz);
      check(r.count == 0 && sameOnAll((double)r.count, MPI_COMM_WORLD),
            "G5 FCG ratio 1e4 SPD V-cycle: silent", r);
    }
    if (rank == 0) {  // G4: single-rank solvers (no initMpi)
      int f4 = 0;
      for (int k = 0; k < 2; ++k) {
        IbmSolver s(N, N, N);
        const Result r = k == 0 ? run(s, kFcg, 1e5, 0, 0, 0, N, N, N, true)
                                : run(s, kPcg, 1.0, 0, 0, 0, N, N, N);
        const bool ok = k == 0 ? r.count >= 1 : (r.count == 0 && r.maxTrue <= 1.0);
        std::printf("  %-44s count %ld  last true %.3e  max true %.3e  %s\n",
                    k == 0 ? "G4 single-rank FCG ratio 1e5 trilinear: flagged"
                           : "G4 single-rank PCG uniform rho: silent",
                    r.count, r.lastTrue, r.maxTrue, ok ? "OK" : "FAIL");
        f4 += ok ? 0 : 1;
      }
      fails += f4;
    }
    MPI_Bcast(&fails, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fails ? 1 : 0;
}
