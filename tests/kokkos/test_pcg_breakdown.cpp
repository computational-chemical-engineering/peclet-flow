/// @file
/// @brief ctest `pcg_breakdown`: the breakdown exits of the device-resident single-rank MG-PCG
/// driver (doc/vof_step_performance_design.md §5.6, WO-5 / A6).
///
/// A6 keeps the Krylov scalars on the device and reads one packet {pAp, |r|inf, stop} per
/// iteration, so a non-finite r^T z is flagged on the device and only reported by the NEXT
/// iteration's packet (or, after the last iteration, by one read after the loop). The contract is
/// that every exit returns the same iteration count and sets the same failure flag as the
/// host-scalar loop, with x the pre-breakdown iterate. No healthy problem reaches these exits, so
/// the solver's test hook `setDebugBreakdown` poisons one scalar with a NaN:
///   * pAp at iteration k    -> k iterations reported, failed, x == the iterate of a maxit = k
///   solve
///   * r^T z at iteration k  -> k iterations reported, failed, x == the iterate of a maxit = k+1
///                              solve (the host loop broke AFTER that iteration's x update)
///   * r^T z at the LAST iteration (k = maxit-1) -> the same, through the post-loop read.
/// x is compared bitwise over the whole array. The problem: a single-rank periodic 16^3 box with a
/// smoothly varying face coefficient in [0.5, 1.5] (ratio 3) and a mean-free rhs; rtol 1e-30 so the
/// solve never converges inside the window.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

struct Problem {
  int n = 16;
  C3 e{18, 18, 18};
  std::size_t cells = 18 * 18 * 18;
  CCField ox, oy, oz, b;
};

Problem makeProblem() {
  Problem P;
  const int G = CutcellMG::G, n = P.n;
  const C3 e = P.e;
  std::vector<double> hx(P.cells, 0.0), hy(P.cells, 0.0), hz(P.cells, 0.0), hb(P.cells, 0.0);
  const double k = 2.0 * M_PI / n;
  double sum = 0.0;
  for (int z = 0; z < e.z; ++z)
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x) {
        const std::size_t i =
            (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
        hx[i] = 1.0 + 0.5 * std::sin(k * x) * std::cos(k * z);
        hy[i] = 1.0 + 0.5 * std::cos(k * y + 0.3);
        hz[i] = 1.0 + 0.5 * std::sin(k * (x + z));
        if (x >= G && x < e.x - G && y >= G && y < e.y - G && z >= G && z < e.z - G) {
          hb[i] = std::sin(k * x) + 0.3 * std::cos(2 * k * y) * std::sin(k * z) + 0.01 * (x - y);
          sum += hb[i];
        }
      }
  const double mean = sum / ((double)n * n * n);
  for (int z = G; z < e.z - G; ++z)
    for (int y = G; y < e.y - G; ++y)
      for (int x = G; x < e.x - G; ++x)
        hb[(std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y] -= mean;
  auto up = [&](const std::vector<double>& h, const char* name) {
    CCField v(name, P.cells);
    auto m = Kokkos::create_mirror_view(v);
    for (std::size_t i = 0; i < P.cells; ++i)
      m(i) = h[i];
    Kokkos::deep_copy(v, m);
    return v;
  };
  P.ox = up(hx, "ox");
  P.oy = up(hy, "oy");
  P.oz = up(hz, "oz");
  P.b = up(hb, "b");
  return P;
}

struct Result {
  int it = -1;
  bool failed = false;
  std::vector<double> x;
};

Result solve(const Problem& P, int maxit, int which, int iter) {
  CutcellMG mg;
  mg.init(P.n, P.n, P.n, 3);
  mg.setOpenness(CCConst(P.ox), CCConst(P.oy), CCConst(P.oz), 1.0, 1.0, 1.0);
  mg.setDebugBreakdown(which, iter);
  CCField x("x", P.cells), r("r", P.cells), p("p", P.cells), z("z", P.cells), Ap("Ap", P.cells);
  Result out;
  out.it = mg.solvePCG(P.b, x, r, p, z, Ap, maxit, 1e-30, 2, 2, 12);
  out.failed = mg.lastSolveFailed();
  auto h = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h, x);
  out.x.assign(h.data(), h.data() + P.cells);
  return out;
}

bool sameBits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

int check(const char* what, const Result& got, int wantIt, bool wantFailed, const Result& ref) {
  const bool ok = got.it == wantIt && got.failed == wantFailed && sameBits(got.x, ref.x);
  std::printf("  %-44s it %d (want %d)  failed %d (want %d)  x %s  -> %s\n", what, got.it, wantIt,
              (int)got.failed, (int)wantFailed, sameBits(got.x, ref.x) ? "bitwise" : "DIFFERS",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    std::printf("=== pcg_breakdown (A6 exits, doc/vof_step_performance_design.md §5.6) ===\n");
    const Problem P = makeProblem();
    const int K = 3, M = 6;
    const Result refK = solve(P, K, 0, -1), refK1 = solve(P, K + 1, 0, -1);
    const Result refM = solve(P, M, 0, -1);
    std::printf("  references: maxit %d -> it %d, maxit %d -> it %d, maxit %d -> it %d\n", K,
                refK.it, K + 1, refK1.it, M, refM.it);
    if (refK.it != K || refK1.it != K + 1 || refM.it != M || refK.failed || refM.failed) {
      std::printf("  FAIL: the reference solves must run to their caps unfailed\n");
      ++fails;
    }
    fails += check("NaN pAp at iteration 3", solve(P, M, 1, K), K, true, refK);
    fails += check("NaN r^T z at iteration 3", solve(P, M, 2, K), K, true, refK1);
    fails +=
        check("NaN r^T z at the last iteration (maxit 4)", solve(P, K + 1, 2, K), K, true, refK1);
    fails += check("NaN pAp at iteration 0", solve(P, M, 1, 0), 0, true, solve(P, 0, 0, -1));
    std::printf("%s\n", fails ? "pcg_breakdown: FAIL" : "pcg_breakdown: PASS");
  }
  Kokkos::finalize();
  return fails ? 1 : 0;
}
