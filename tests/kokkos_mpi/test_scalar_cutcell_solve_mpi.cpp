// Cut-cell scalar solve under MPI (doc/scalar_ibm_design.md §5.1, §5.4; WO-3).
//
// np = 1, 2, 4 on the production ORB. One global problem — an off-centre solid sphere, no-slip
// flow walls on +-x carrying scalar Dirichlet values (one a per-face profile), y and z periodic —
// solved twice on the distributed solver and on a single-rank reference of the same problem:
//   * steady, Dirichlet walls (c = 1) — the solve of G1's kind;
//   * transient, three backward-Euler steps with Robin walls and a volumetric source.
// The parallel contract of §5.4: np = 1 in the MPI build is BITWISE equal to the single-rank build
// (field, iterations, wall flux, budget); np > 1 agrees to the Krylov reduction-order floor,
// max |c - c_1| <= 1e-9 max |c_1|, iterations within +-1 per solve, and the collective getters
// (wall flux, budget) agree to the same floor; the budget identity closes on every rank count.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using IbmSolver = peclet::flow::IbmSolver;
static constexpr int G = IbmSolver::G;
static constexpr int N = 32;

static int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

static bool same(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

static std::vector<double> sphereSdf() {
  std::vector<double> s((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x - 15.3, dy = y - 16.1, dz = z - 14.7;
        s[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - 9.3;
      }
  return s;
}

// the +x face profile at global (y, z)
static double profile(int y, int z) {
  return 0.4 + 0.1 * std::sin(0.3 * y) * std::cos(0.2 * z);
}

struct Run {
  std::vector<double> c;  // the field at the solver's inner cells, x-fastest
  int iters[4] = {0, 0, 0, 0};
  double flux[2] = {0, 0}, identity[2] = {0, 0}, dmass[2] = {0, 0};
};

// Configure, then: one steady solve, then three transient steps (Robin + source).
static void configure(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  (void)ox;
  s.addScalar("c", 1.2, 1, 50, true);
  s.setScalarBc("c", 0, 2, 0.15);
  std::vector<double> prof((std::size_t)lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      prof[(std::size_t)y + (std::size_t)z * lny] = profile(y + oy, z + oz);
  s.setScalarBcProfile("c", 1, prof, lny, lnz);
  s.setScalarMaxIterations("c", 3000);
  (void)lnx;
}

static Run solveAll(IbmSolver& s) {
  Run r;
  s.setScalarWall("c", 1, 1.0, 0.0, -1);
  s.solveScalarSteady("c");
  r.iters[0] = s.scalarField("c").cut->iterations;
  r.flux[0] = s.scalarWallFlux("c")[0];
  auto b = s.scalarBudget("c");
  r.identity[0] = b.identityError / std::fabs(b.wallIn);
  s.setScalarWall("c", 2, 0.3, 0.7, -1);
  s.setScalarSource("c", 0.1);
  for (int k = 0; k < 3; ++k) {
    s.advanceScalars();
    r.iters[1 + k] = s.scalarField("c").cut->iterations;
  }
  r.flux[1] = s.scalarWallFlux("c")[0];
  b = s.scalarBudget("c");
  r.identity[1] = b.identityError / std::fabs(b.dMass);
  r.dmass[1] = b.dMass;
  return r;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Kokkos::initialize(argc, argv);
  {
    const std::vector<double> gsdf = sphereSdf();
    peclet::core::decomp::BlockDecomposer<3> dec =
        peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    const auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
              gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N +
                   (std::size_t)(z + oz) * N * N];
    auto common = [](IbmSolver& s) {
      s.setRho(1.0);
      s.setMu(1.0);
      s.setDt(2.5);
      s.setDomainBc(0, 1, 0.0, 0.0, 0.0);
      s.setDomainBc(1, 1, 0.0, 0.0, 0.0);
    };
    IbmSolver sd(lnx, lny, lnz);
    common(sd);
    sd.initMpi(N, N, N, MPI_COMM_WORLD);
    sd.setSolid(lsdf, false);
    configure(sd, ox, oy, oz, lnx, lny, lnz);
    IbmSolver sr(N, N, N);
    common(sr);
    sr.setSolid(gsdf, false);
    configure(sr, 0, 0, 0, N, N, N);
    const Run D = solveAll(sd);
    const Run R = solveAll(sr);

    // the fields by global index
    auto cd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sd.scalarField("c").c);
    auto cr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sr.scalarField("c").c);
    const long dsy = lnx + 2 * G, dsz = dsy * (lny + 2 * G), rsy = N + 2 * G,
               rsz = rsy * (N + 2 * G);
    double maxDiff = 0.0, maxRef = 0.0;
    long notSame = 0;
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x) {
          const double a = cd((x + G) + (y + G) * dsy + (z + G) * dsz);
          const double b = cr((x + ox + G) + (y + oy + G) * rsy + (z + oz + G) * rsz);
          maxDiff = std::fmax(maxDiff, std::fabs(a - b));
          maxRef = std::fmax(maxRef, std::fabs(b));
          notSame += same(a, b) ? 0 : 1;
        }
    double g[2] = {maxDiff, maxRef}, gg[2];
    MPI_Allreduce(g, gg, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    long ns = 0;
    MPI_Allreduce(&notSame, &ns, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      std::printf("np=%d: max|c - c_1| = %.2e (max|c_1| %.3f), %ld cells not bitwise\n", size,
                  gg[0], gg[1], ns);
      std::printf("  iterations np: %d %d %d %d   single: %d %d %d %d\n", D.iters[0], D.iters[1],
                  D.iters[2], D.iters[3], R.iters[0], R.iters[1], R.iters[2], R.iters[3]);
      std::printf("  wall flux np: %.15e %.15e   single: %.15e %.15e\n", D.flux[0], D.flux[1],
                  R.flux[0], R.flux[1]);
      std::printf("  budget identity (rel) np: %.1e %.1e   single: %.1e %.1e\n", D.identity[0],
                  D.identity[1], R.identity[0], R.identity[1]);
    }
    for (int k = 0; k < 2; ++k) {
      CHECK(std::fabs(D.identity[k]) <= 1e-12);
      CHECK(std::fabs(R.identity[k]) <= 1e-12);
    }
    if (size == 1) {
      CHECK(ns == 0);
      for (int k = 0; k < 4; ++k)
        CHECK(D.iters[k] == R.iters[k]);
      CHECK(same(D.flux[0], R.flux[0]) && same(D.flux[1], R.flux[1]));
    } else {
      CHECK(gg[0] <= 1e-9 * gg[1]);
      for (int k = 0; k < 4; ++k)
        CHECK(std::abs(D.iters[k] - R.iters[k]) <= 1);
      for (int k = 0; k < 2; ++k)
        CHECK(std::fabs(D.flux[k] - R.flux[k]) <= 1e-9 * std::fabs(R.flux[k]));
    }
  }
  Kokkos::finalize();
  int all = 0;
  MPI_Allreduce(&failures, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(all ? "%d failure(s)\n" : "OK\n", all);
  MPI_Finalize();
  return all ? 1 : 0;
}
