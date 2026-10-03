// Cut-cell scalar solve under MPI (doc/scalar_ibm_design.md §5.1, §5.2, §5.4; WO-3, WO-4).
//
// np = 1, 2, 4 on the production ORB. Each problem is solved on the distributed solver and on a
// single-rank reference of the same problem; the parallel contract of §5.4 (G10) is checked at
// rtol 1e-13 (ruling Q-D of WO-4, the vardensity_mpi precedent): np = 1 in the MPI build is BITWISE
// equal to the single-rank build (field, iterations, wall flux); np > 1: max |c - c_1| <= 1e-10
// max |c_1|, iterations within +-1 per solve, the wall flux to the same bound; the budget identity
// closes on every rank count. At the default rtol 1e-10 the np > 1 gap is the BiCGStab stopping
// error (the V-cycle is decomposition-independent): `g1` is also run there and printed for
// information only. Every solve runs the ScalarMG V-cycle of WO-4 (the transient steps have
// dt D/h^2 > 1).
//
//   mixed     — an off-centre solid sphere, no-slip flow walls on +-x carrying scalar Dirichlet
//               values (one a per-face profile), y and z periodic: one steady solve with Dirichlet
//               walls (c = 1), then three backward-Euler steps with Robin walls and a source;
//   g1        — G1 at R/h = 16 (box 4R = 64, walls everywhere): a Dirichlet sphere c = 1, the
//               exact field R/r as the box Dirichlet profile on all six faces, steady;
//   singular  — the periodic box, insulating walls with a flux and a source, steady: the
//               singular case of §5.1 (mean projection on every ScalarMG level, gauge kept).
//
// Plus the level table: ScalarMG's distributed table equals VelocityMG::initMpi's (in place).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using IbmSolver = peclet::flow::IbmSolver;
static constexpr int G = IbmSolver::G;

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

static int rank_ = 0, size_ = 1;

struct Block {
  int o[3], l[3];
};

struct Run {
  std::vector<int> iters;
  std::vector<double> flux;  // the body flux after each recorded stage
  std::vector<double> identity;
};

using FlowSetup = std::function<void(IbmSolver&)>;
using ScalarSetup = std::function<void(IbmSolver&, const Block&)>;
using Solve = std::function<Run(IbmSolver&)>;

static std::vector<double> sphereSdf(int n, double cx, double cy, double cz, double R) {
  std::vector<double> s((std::size_t)n * n * n);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double dx = x - cx, dy = y - cy, dz = z - cz;
        s[(std::size_t)x + (std::size_t)y * n + (std::size_t)z * n * n] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return s;
}

static void record(IbmSolver& s, Run& r, bool transient) {
  r.iters.push_back(s.scalarField("c").cut->iterations);
  r.flux.push_back(s.scalarWallFlux("c")[0]);
  auto b = s.scalarBudget("c");
  r.identity.push_back(b.identityError / std::fabs(transient ? b.dMass : b.wallIn));
}

static void compare(const char* tag, int n, const std::vector<double>& gsdf, FlowSetup flow,
                    ScalarSetup scalar, Solve solve, bool checkLevels = false, bool gated = true) {
  peclet::core::decomp::BlockDecomposer<3> dec =
      peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size_), n, n, n);
  const auto blk = dec.block(rank_);
  Block B{{(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]},
          {(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]}};
  std::vector<double> lsdf((std::size_t)B.l[0] * B.l[1] * B.l[2]);
  for (int z = 0; z < B.l[2]; ++z)
    for (int y = 0; y < B.l[1]; ++y)
      for (int x = 0; x < B.l[0]; ++x)
        lsdf[(std::size_t)x + (std::size_t)y * B.l[0] + (std::size_t)z * B.l[0] * B.l[1]] =
            gsdf[(std::size_t)(x + B.o[0]) + (std::size_t)(y + B.o[1]) * n +
                 (std::size_t)(z + B.o[2]) * n * n];
  IbmSolver sd(B.l[0], B.l[1], B.l[2]);
  flow(sd);
  sd.initMpi(n, n, n, MPI_COMM_WORLD);
  sd.setSolid(lsdf, false);
  scalar(sd, B);
  IbmSolver sr(n, n, n);
  flow(sr);
  sr.setSolid(gsdf, false);
  scalar(sr, Block{{0, 0, 0}, {n, n, n}});
  if (gated) {  // ruling Q-D: G10 is gated at rtol 1e-13
    sd.setScalarTolerance("c", 1e-13);
    sr.setScalarTolerance("c", 1e-13);
  }
  const Run D = solve(sd);
  const Run R = solve(sr);
  auto cd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sd.scalarField("c").c);
  auto cr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sr.scalarField("c").c);
  const long dsy = B.l[0] + 2 * G, dsz = dsy * (B.l[1] + 2 * G), rsy = n + 2 * G,
             rsz = rsy * (n + 2 * G);
  double maxDiff = 0.0, maxRef = 0.0;
  long notSame = 0;
  for (int z = 0; z < B.l[2]; ++z)
    for (int y = 0; y < B.l[1]; ++y)
      for (int x = 0; x < B.l[0]; ++x) {
        const double a = cd((x + G) + (y + G) * dsy + (z + G) * dsz);
        const double b = cr((x + B.o[0] + G) + (y + B.o[1] + G) * rsy + (z + B.o[2] + G) * rsz);
        maxDiff = std::fmax(maxDiff, std::fabs(a - b));
        maxRef = std::fmax(maxRef, std::fabs(b));
        notSame += same(a, b) ? 0 : 1;
      }
  double g[2] = {maxDiff, maxRef}, gg[2];
  MPI_Allreduce(g, gg, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  long ns = 0;
  MPI_Allreduce(&notSame, &ns, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  const int lvD = sd.scalarField("c").cut->mgLevels, lvR = sr.scalarField("c").cut->mgLevels;
  if (rank_ == 0) {
    std::printf(
        "[%s%s] np=%d: max|c - c_1| = %.2e (max|c_1| %.3f), %ld cells not bitwise; "
        "mg levels np %d single %d\n",
        tag, gated ? "" : " @ default rtol 1e-10, information only", size_, gg[0], gg[1], ns, lvD,
        lvR);
    std::printf("  iterations np:");
    for (int it : D.iters)
      std::printf(" %d", it);
    std::printf("   single:");
    for (int it : R.iters)
      std::printf(" %d", it);
    std::printf("\n  wall flux np: %.15e   single: %.15e\n", D.flux.back(), R.flux.back());
    double idm = 0.0;
    for (std::size_t k = 0; k < D.identity.size(); ++k)
      idm = std::fmax(idm, std::fmax(std::fabs(D.identity[k]), std::fabs(R.identity[k])));
    std::printf("  budget identity (rel) <= %.1e\n", idm);
  }
  if (!gated)
    return;
  CHECK(D.iters.size() == R.iters.size());
  for (std::size_t k = 0; k < D.identity.size(); ++k) {
    CHECK(std::fabs(D.identity[k]) <= 1e-12);
    CHECK(std::fabs(R.identity[k]) <= 1e-12);
  }
  if (size_ == 1) {
    CHECK(ns == 0);
    for (std::size_t k = 0; k < D.iters.size(); ++k)
      CHECK(D.iters[k] == R.iters[k]);
    for (std::size_t k = 0; k < D.flux.size(); ++k)
      CHECK(same(D.flux[k], R.flux[k]));
  } else {
    CHECK(gg[0] <= 1e-10 * gg[1]);
    for (std::size_t k = 0; k < D.iters.size(); ++k)
      CHECK(std::abs(D.iters[k] - R.iters[k]) <= 1);
    for (std::size_t k = 0; k < D.flux.size(); ++k)
      CHECK(std::fabs(D.flux[k] - R.flux[k]) <= 1e-10 * std::fabs(R.flux[k]));
  }
  if (checkLevels) {
    // ScalarMG's distributed level table == VelocityMG::initMpi(dec, ., comm, inPlace)'s
    const auto& mg = *sd.scalarField("c").cut->mg;
    peclet::flow::VelocityMG vmg;
    const double w[3] = {1.0, 1.0, 1.0}, hp[3] = {1.0, 1.0, 1.0};
    vmg.setMetric(w, hp);
    vmg.initMpi(dec, 64, MPI_COMM_WORLD);
    bool eq = vmg.levels() == mg.levels();
    for (int L = 0; eq && L < mg.levels(); ++L) {
      const auto& a = vmg.level(L);
      const auto& b = mg.level(L);
      eq = a.inner.x == b.inner.x && a.inner.y == b.inner.y && a.inner.z == b.inner.z &&
           a.ratio.x == b.ratio.x && a.ratio.y == b.ratio.y && a.ratio.z == b.ratio.z &&
           a.og.x == b.og.x && a.og.y == b.og.y && a.og.z == b.og.z;
    }
    int all = eq ? 1 : 0, allMin = 0;
    MPI_Allreduce(&all, &allMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank_ == 0)
      std::printf("  level table: %d levels, %s VelocityMG::initMpi\n", mg.levels(),
                  allMin ? "==" : "DIFFERS from");
    CHECK(allMin == 1);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  Kokkos::initialize(argc, argv);
  {
    // ---- mixed (the WO-3 problem) ----
    {
      const int N = 32;
      auto profile = [](int y, int z) { return 0.4 + 0.1 * std::sin(0.3 * y) * std::cos(0.2 * z); };
      compare(
          "mixed", N, sphereSdf(N, 15.3, 16.1, 14.7, 9.3),
          [](IbmSolver& s) {
            s.setRho(1.0);
            s.setMu(1.0);
            s.setDt(2.5);
            s.setDomainBc(0, 1, 0.0, 0.0, 0.0);
            s.setDomainBc(1, 1, 0.0, 0.0, 0.0);
          },
          [&](IbmSolver& s, const Block& B) {
            s.addScalar("c", 1.2, 1, 50, true);
            s.setScalarBc("c", 0, 2, 0.15);
            std::vector<double> prof((std::size_t)B.l[1] * B.l[2]);
            for (int z = 0; z < B.l[2]; ++z)
              for (int y = 0; y < B.l[1]; ++y)
                prof[(std::size_t)y + (std::size_t)z * B.l[1]] = profile(y + B.o[1], z + B.o[2]);
            s.setScalarBcProfile("c", 1, prof, B.l[1], B.l[2]);
          },
          [](IbmSolver& s) {
            Run r;
            s.setScalarWall("c", 1, 1.0, 0.0, -1);
            s.solveScalarSteady("c");
            record(s, r, false);
            s.setScalarWall("c", 2, 0.3, 0.7, -1);
            s.setScalarSource("c", 0.1);
            for (int k = 0; k < 3; ++k) {
              s.advanceScalars();
              record(s, r, true);
            }
            return r;
          },
          true);
    }
    // ---- G10 on G1 (R/h = 16, box 4R) ----
    {
      const int N = 64;
      const double R = 16.0, c0[3] = {31.87, 32.21, 31.66};
      for (const bool gated : {true, false})
        compare(
            "g1", N, sphereSdf(N, c0[0], c0[1], c0[2], R),
            [](IbmSolver& s) {
              s.setRho(1.0);
              s.setMu(1.0);
              s.setDt(1.0);
              for (int f = 0; f < 6; ++f)
                s.setDomainBc(f, 1, 0.0, 0.0, 0.0);
            },
            [&](IbmSolver& s, const Block& B) {
              s.addScalar("c", 0.7, 1, 50, true);
              for (int f = 0; f < 6; ++f) {
                const int a = f / 2;
                const int t1 = (a == 0) ? 1 : 0, t2 = (a == 2) ? 1 : 2;
                const double xa = (f % 2 == 0) ? -0.5 : N - 0.5;
                std::vector<double> prof((std::size_t)B.l[t1] * B.l[t2]);
                for (int j2 = 0; j2 < B.l[t2]; ++j2)
                  for (int j1 = 0; j1 < B.l[t1]; ++j1) {
                    double p[3];
                    p[a] = xa;
                    p[t1] = j1 + B.o[t1];
                    p[t2] = j2 + B.o[t2];
                    const double r = std::sqrt((p[0] - c0[0]) * (p[0] - c0[0]) +
                                               (p[1] - c0[1]) * (p[1] - c0[1]) +
                                               (p[2] - c0[2]) * (p[2] - c0[2]));
                    prof[(std::size_t)j1 + (std::size_t)j2 * B.l[t1]] = R / r;
                  }
                s.setScalarBc("c", f, 2, 0.0);
                s.setScalarBcProfile("c", f, prof, B.l[t1], B.l[t2]);
              }
              s.setScalarWall("c", 1, 1.0, 0.0, -1);
            },
            [](IbmSolver& s) {
              Run r;
              s.solveScalarSteady("c");
              record(s, r, false);
              return r;
            },
            false, gated);
    }
    // ---- G10 on the singular case (periodic, insulating + flux + source, steady) ----
    {
      const int N = 32;
      compare(
          "singular", N, sphereSdf(N, 15.3, 16.1, 14.7, 9.3),
          [](IbmSolver& s) {
            s.setRho(1.0);
            s.setMu(1.0);
            s.setDt(1.0);
          },
          [](IbmSolver& s, const Block&) {
            s.addScalar("c", 0.9, 1, 50, true);
            s.setScalarWall("c", 0, 0.4, 0.0, -1);
            s.setScalarSource("c", -0.05);
          },
          [](IbmSolver& s) {
            Run r;
            s.solveScalarSteady("c");
            r.iters.push_back(s.scalarField("c").cut->iterations);
            r.flux.push_back(s.scalarWallFlux("c")[0]);
            CHECK(s.scalarField("c").cut->singular);
            return r;
          });
    }
  }
  Kokkos::finalize();
  int all = 0;
  MPI_Allreduce(&failures, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank_ == 0)
    std::printf(all ? "%d failure(s)\n" : "OK\n", all);
  MPI_Finalize();
  return all ? 1 : 0;
}
