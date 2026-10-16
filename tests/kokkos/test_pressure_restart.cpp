/// @file
/// @brief ctest `pressure_restart`: the FP64 restart of a non-converged FP32-preconditioned PCG /
/// FCG solve under 'auto' (doc/vof_projection_distributed_design.md §4, gate G-R (a)-(g) of §8).
///
/// CutcellMG level, on the column, cyl and pack problems of `mg_fp32_vcycle`
/// (mg_fp32_problems.hpp), PCG and FCG, rtol 1e-10, cap 200, with the C++ fault hook
/// CutcellMG::debugSetFp32Fault (kind 1: NaN exit scale on the at-th FP32 V-cycle of an attempt;
/// kind 2: an FP32 attempt runs at most `at` iterations):
///   (a) 'auto', kind 1 at 3: x memcmp-equal to the 'fp64' cold solve, the same iterations and
///       failure flag (false wherever 'fp64' itself does not break down), one restart, the last
///       V-cycle FP64; where the unfaulted FP32 attempt caps or breaks down by itself (cyl, pack at
///       1e4) the unfaulted 'auto' solve is checked the same way (a natural restart);
///   (b) 'auto', kind 2 at 2: as (a), and 2 discarded iterations;
///   (c) forced 'fp32', kind 1 at 3 / kind 2 at 2: no restart; the failure flag set / 2 iterations;
///   (d) 'fp64' with either fault: bitwise the solve without it;
///   (e) set_pressure_strict + 'auto' + kind 1 (at 1: the initial guard, the throw site): no
///       throw, (a) holds; strict + forced 'fp32' + kind 1 at 1: throws;
///   (f) after (a) the next 'auto' solve runs FP64 with no restart; after setVcyclePrecision(auto)
///       the next solve runs FP32 and is bitwise a fresh 'auto' solve.
/// Assembled (g): a periodic VoF bubble (ratio 100, momentum-consistent transport, the exact
/// level-0 operator, so the D5 deferral is live), 'auto' + 'projected' history 2, kind 1 at 3 on
/// step 4: step 4's p is memcmp-equal to an 'fp64' + 'zero' step from the same state, and the
/// history count advances (it is not reset).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "mg_fp32_problems.hpp"

using namespace peclet::flow;
using namespace peclet_flow_fp32_problems;

namespace {

int fails = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %-96s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok)
    ++fails;
}

// Does 'auto' run the FP32 V-cycle on this backend? (Every eligible backend until WO-A.)
const bool kAutoFp32 = true;

struct Run {
  int it = -1, restarts = 0, discarded = -1;
  bool failed = false, fp32 = false, threw = false;
  std::vector<double> x;
};
bool sameX(const Run& a, const Run& b) {
  return a.x.size() == b.x.size() &&
         std::memcmp(a.x.data(), b.x.data(), a.x.size() * sizeof(double)) == 0;
}

// One solve of P on `mg` (driver 0 = PCG, 1 = FCG) from x = 0 with the fault armed for it.
Run solveWith(CutcellMG& mg, const Problem& P, int driver, int kind, int at) {
  const std::vector<double> hb = rhsOf(P);
  const std::size_t n = hb.size();
  CCField b = toDevice(hb, "b"), x("x", n), r("r", n), p("p", n), z("z", n), zp("zp", n),
          Ap("Ap", n);
  mg.debugSetFp32Fault(kind, at);
  Run out;
  try {
    out.it = driver == 0 ? mg.solvePCG(b, x, r, p, z, Ap, 200, 1e-10, 2, 2, 12)
                         : mg.solveFCG(b, x, r, p, z, zp, Ap, 200, 1e-10, 2, 2, 12);
  } catch (const std::runtime_error&) {
    out.threw = true;
  }
  mg.debugSetFp32Fault(0, 0);
  out.failed = mg.lastSolveFailed();
  out.fp32 = mg.lastVcycleFp32();
  out.restarts = mg.vcycleRestarts();
  out.discarded = mg.lastDiscardedIters();
  out.x = toHost(x);
  return out;
}
Run fresh(const Problem& P, int mode, int driver, int kind = 0, int at = 0, bool strict = false) {
  CutcellMG mg;
  setUp(mg, P, mode);
  mg.setStrictPressure(strict);
  return solveWith(mg, P, driver, kind, at);
}
std::string itStr(const Run& r) {
  return std::to_string(r.it) + (r.failed ? " failed" : "") + (r.fp32 ? " fp32" : " fp64");
}

// (a) the restarted solve equals the 'fp64' cold solve.
bool restartedIsFp64(const Run& a, const Run& ref64) {
  return sameX(a, ref64) && a.it == ref64.it && a.failed == ref64.failed && !a.fp32 &&
         a.restarts == 1 && !a.threw;
}

void levelChecks(const Problem& P) {
  for (int driver : {0, 1}) {
    const std::string tag = P.name + (driver == 0 ? " PCG" : " FCG");
    const Run ref64 = fresh(P, CutcellMG::kVcycleFp64, driver);
    const Run auto0 = fresh(P, CutcellMG::kVcycleAuto, driver);
    std::printf("  %s: fp64 %s, auto %s\n", tag.c_str(), itStr(ref64).c_str(),
                itStr(auto0).c_str());
    if (!kAutoFp32) {
      // host 'auto' is 'fp64' (WO-A): no FP32 attempt, so the fault is inert and nothing restarts
      const Run a = fresh(P, CutcellMG::kVcycleAuto, driver, 1, 3);
      check(sameX(a, ref64) && a.it == ref64.it && !a.fp32 && a.restarts == 0,
            tag + ": host 'auto' == 'fp64', the fault inert, no restart (" + itStr(a) + ")");
    } else {
      // cyl and pack at ratio 1e4: the FP32 attempt itself caps or breaks down (so does FP64, the
      // coarsening limit), i.e. the unfaulted 'auto' solve is a NATURAL restart
      if (auto0.restarts == 1)
        check(restartedIsFp64(auto0, ref64),
              tag +
                  ": 'auto' unfaulted: the FP32 attempt does not converge -> natural restart == "
                  "fp64 cold (discarded " +
                  std::to_string(auto0.discarded) + ")");
      else
        check(auto0.fp32 && auto0.restarts == 0,
              tag + ": 'auto' unfaulted runs the FP32 V-cycle, converges, no restart");
      // (a)
      const Run a = fresh(P, CutcellMG::kVcycleAuto, driver, 1, 3);
      check(restartedIsFp64(a, ref64), tag + ": (a) kind 1 at 3 -> restart == fp64 cold (" +
                                           itStr(a) + ", discarded " + std::to_string(a.discarded) +
                                           ")");
      // (b)
      const Run b = fresh(P, CutcellMG::kVcycleAuto, driver, 2, 2);
      check(restartedIsFp64(b, ref64) && b.discarded == 2,
            tag + ": (b) kind 2 at 2 -> restart == fp64 cold, discarded " +
                std::to_string(b.discarded));
      // (e)
      const Run e = fresh(P, CutcellMG::kVcycleAuto, driver, 1, 1, /*strict=*/true);
      check(restartedIsFp64(e, ref64),
            tag + ": (e) strict + 'auto' + kind 1 at 1: no throw, restart == fp64 cold");
      // (f)
      CutcellMG mg;
      setUp(mg, P, CutcellMG::kVcycleAuto);
      const Run f0 = solveWith(mg, P, driver, 1, 3);
      const Run f1 = solveWith(mg, P, driver, 0, 0);
      check(restartedIsFp64(f0, ref64) && sameX(f1, ref64) && f1.it == ref64.it && !f1.fp32 &&
                f1.restarts == 1,
            tag + ": (f) after a restart 'auto' stays FP64, no second restart");
      mg.setVcyclePrecision(CutcellMG::kVcycleAuto);
      const Run f2 = solveWith(mg, P, driver, 0, 0);
      check(sameX(f2, auto0) && f2.it == auto0.it && f2.fp32 == auto0.fp32 &&
                f2.restarts == 1 + auto0.restarts,
            tag + ": (f) setVcyclePrecision(auto) re-arms: bitwise a fresh 'auto' solve");
    }
    // (c)
    const Run c1 = fresh(P, CutcellMG::kVcycleFp32, driver, 1, 3);
    const Run c2 = fresh(P, CutcellMG::kVcycleFp32, driver, 2, 2);
    check(c1.restarts == 0 && c1.failed && c1.fp32 && c2.restarts == 0 && c2.it == 2 && c2.fp32,
          tag + ": (c) forced 'fp32': no restart; kind 1 failed (" + itStr(c1) + "), kind 2 " +
              itStr(c2));
    // (d)
    const Run d1 = fresh(P, CutcellMG::kVcycleFp64, driver, 1, 3);
    const Run d2 = fresh(P, CutcellMG::kVcycleFp64, driver, 2, 2);
    check(sameX(d1, ref64) && d1.it == ref64.it && sameX(d2, ref64) && d2.it == ref64.it &&
              d1.restarts == 0 && d2.restarts == 0,
          tag + ": (d) 'fp64' with either fault == no fault");
    // (e), forced
    const Run e2 = fresh(P, CutcellMG::kVcycleFp32, driver, 1, 1, /*strict=*/true);
    check(e2.threw, tag + ": (e) strict + forced 'fp32' + kind 1 at 1 throws");
  }
}

// --- (g): the assembled VoF case ---------------------------------------------------------------
std::size_t idx3(int x, int y, int z, int n) {
  return (std::size_t)x + (std::size_t)y * n + (std::size_t)z * (std::size_t)n * n;
}
std::unique_ptr<IbmSolver> bubble(int vprec, int guessMode) {
  const int n = 32;
  const double rhoG = 1.0, rhoL = 100.0, r = 7.0, c = 0.5 * n;
  const std::size_t nc = (std::size_t)n * n * n;
  std::vector<double> C(nc), u(nc), v(nc, 0.0), w(nc, 0.0);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        C[idx3(x, y, z, n)] = vof::sphereCellFraction(c, c, c, r, x, y, z, 1.0, 4);
        u[idx3(x, y, z, n)] = 0.2 * std::sin(2.0 * M_PI * x / n) * std::cos(2.0 * M_PI * y / n);
      }
  for (double& cc : C)
    cc = 1.0 - cc;  // liquid (C = 1) around a gas bubble
  auto s = std::make_unique<IbmSolver>(n, n, n);
  s->setRho(rhoL);
  s->setMu(0.01);
  s->setDt(0.2);
  s->setPressureGeometry(std::vector<double>(nc, 10.0));
  s->enableVof();
  s->setVof(C);
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {rhoG, rhoL - rhoG});
  s->enableVofMomentum(rhoG, rhoL);
  s->setPressurePcg(true, 200, 1e-10);  // last: the property model re-selects Chebyshev
  s->setPressureVcyclePrecision(vprec);
  s->setPressureInitialGuess(guessMode, 2);
  s->uploadVelocity(u, v, w);
  return s;
}
void assembledCheck() {
  constexpr int kAuto = 0, kFp64 = 1, kZero = 1, kProjected = 2;
  auto A = bubble(kAuto, kProjected), B = bubble(kAuto, kProjected);
  for (int k = 1; k <= 3; ++k) {
    A->step();
    B->step();
  }
  const bool sameState = A->getPressure() == B->getPressure();
  const int histA = A->pressureGuessCount();
  // step 4 of A: the projection's FP32 attempt fails on its 3rd V-cycle and is restarted
  A->debugSetPressureFp32Fault(1, 3);
  A->step();
  A->debugSetPressureFp32Fault(0, 0);
  // step 4 of B: 'fp64' + 'zero' from the same state (host after WO-A: B unchanged, unfaulted)
  if (kAutoFp32) {
    B->setPressureVcyclePrecision(kFp64);
    B->setPressureInitialGuess(kZero, 2);
  }
  B->step();
  const std::vector<double> pA = A->getPressure(), pB = B->getPressure();
  const bool same =
      pA.size() == pB.size() && std::memcmp(pA.data(), pB.data(), pA.size() * sizeof(double)) == 0;
  std::printf(
      "  bubble: history %d before step 4 (%s), %d after; iterations A %ld B %ld; "
      "restarts A %d\n",
      histA, sameState ? "A == B" : "A != B", A->pressureGuessCount(), A->lastPressureIterations(),
      B->lastPressureIterations(), A->pressureVcycleRestarts());
  if (!kAutoFp32) {
    check(sameState && same && A->pressureVcycleRestarts() == 0 && !A->pressureVcycleFp32(),
          "bubble (g): host 'auto' == 'fp64': the fault inert, step 4 p == the unfaulted step");
    return;
  }
  check(sameState && histA == 2, "bubble (g): A and B reach step 4 bitwise, history 2");
  check(same && A->lastPressureIterations() == B->lastPressureIterations() &&
            A->pressureVcycleRestarts() == 1 && A->pressureGuessCount() == 2 &&
            !A->pressureVcycleFp32(),
        "bubble (g): step 4 p memcmp-equal to 'fp64'+'zero', one restart, history advances");
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  std::printf("pressure_restart (doc/vof_projection_distributed_design.md §4, G-R)\n");
  for (const Problem& P : {columnProblem(), cylProblem(), packProblem()})
    levelChecks(P);
  assembledCheck();
  std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
  Kokkos::finalize();
  return fails ? 1 : 0;
}
