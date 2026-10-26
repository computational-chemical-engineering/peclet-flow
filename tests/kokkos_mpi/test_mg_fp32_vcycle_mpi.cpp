/// @file
/// @brief ctest `mg_fp32_vcycle_mpi`: the distributed FP32 V-cycle data
/// (doc/vof_projection_distributed_design.md §5.1-5.3, gate G-K2 of §8).
///
/// On the periodic cyl problem of `mg_fp32_vcycle` (32^3, R = 8 cylinder, a heavy blob at ratio
/// 1e4) and a 64^3 periodic box with a heavy sphere at 1e4, distributed over MPI_COMM_WORLD with
/// the communication-avoiding coarse levels on (g = 2 where the blocks allow it):
///   G-K2: after the operator build ('fp32'), and again after the setter's rebuild from the stored
///         openness, on every level the FP32 V-cycle runs on (0 .. fp32Last), WX, WY, WZ on every
///         ghost cell (depth g) are memcmp-equal to the owner's values: a copy exchanged with the
///         level's GridHalo<float> equals the stored array on every cell;
///   the range flags are allreduced (fp32WeightsOk collective, true everywhere), the
///   configuration is carriable, and the solve stays ineligible (WO-D1 is inert: the distributed
///   eligibility comes with WO-D2a);
///   an 'fp64' build creates no FP32 halo and no FP32 arrays.
///
/// WO-D2a (§5.4-5.7, gates G-DD1, G-DD2 (a)-(e), (g) and G-R (h) of §8), the distributed FP32
/// V-cycle on the host-scalar PCG / FCG, against a single-rank reference every rank solves itself
/// (column, cyl, pack, a walled slab at 1e2, box64; rtol 1e-10, cap 200):
///   G-DD1 (np 1): one FP32 V-cycle z memcmp-equal on the inner cells; forced 'fp32' (and on a
///         device 'auto') PCG and FCG: x memcmp-equal, iterations equal;
///   G-DD2 (np > 1): (a) max|x_np - x_1| <= 1e-8 and iterations within max(1, 5 %) wherever the
///         single-rank solve converges (a configuration converging in neither mode is n/a); (b) one
///         V-cycle z within 1e-11 relative (cyl, box64, equal depth); (c) 'fp32' iterations <=
///         'fp64' + max(1, 5 %) where 'fp64' converges (every np); (d) np 2, where 'fp64'
///         converges: the attainable floor (rtol 1e-14, cap 300), fp32 <= 2 x fp64; (e) CA on == CA
///         off bitwise (cyl, box64, 30 iterations); (g) np 2: two runs bitwise (30 iterations);
///   G-R (h) (column): (a) 'auto' + kind 1 at 3 and (b) kind 2 at 2 restart to the 'fp64' cold
///   solve (on a
///         device; host 'auto' is 'fp64', the fault inert), (c) forced 'fp32' never restarts;
///         vcycleRestarts() equal on every rank; np 1 memcmp-equal to the single-rank restart.
/// WO-D2b1 (G-DD2 (f)): telescoped hierarchies -- natural (column at 6 levels, np 4 and 8) and
/// forced (setTelescopeForceLevel(1), column, np 4): fp32Last is the first telescope level, and
/// (a)-(c) as above -- (b) gated where the telescope point is the last non-bottom level (the
/// single-rank precision split); above it FP64 runs below the point (§3.2), so z is reported
/// against the single-rank FP32-vs-FP64 V-cycle difference instead.
/// doc/mg_depth_policy.md A2.4 (WO-7): lineChecks -- the zebra line smoother distributed against
/// the single-rank reference on hierarchies with triggered smoothed levels (np 1 memcmp where the
/// LR-off control is memcmp-equal; np > 1 G-DD2 (a)'s bound), FP64 and forced FP32.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "mg_fp32_problems.hpp"

using namespace peclet::flow;
using namespace peclet_flow_fp32_problems;

namespace {

int rank = 0, size = 1, fails = 0;
void check(bool ok, const std::string& what) {
  int bad = ok ? 0 : 1, any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("  %-100s %s\n", what.c_str(), any ? "FAIL" : "PASS");
  if (any)
    ++fails;
}

// This rank's block of the global openness (inner faces; setOpenness fills the ghosts).
void setUpDist(CutcellMG& mg, const Problem& P, int mode, int ca = kCaBoth, int teleForce = -1) {
  mg.setBoundaryConditions(P.bc);  // before initMpi (the per-level ghost width reads it)
  mg.setCommAvoiding(ca);
  mg.setTelescopeForceLevel(teleForce);
  mg.initMpi(P.nx, P.ny, P.nz, P.levels, MPI_COMM_WORLD);
  mg.setVcyclePrecision(mode);
  const CutcellMG::Level& l0 = mg.level(0);
  const C3 e = l0.ext, og = l0.og;
  const int g = l0.g;
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  std::vector<double> ox(n, 0.0), oy(n, 0.0), oz(n, 0.0);
  for (int z = g; z < e.z - g; ++z)
    for (int y = g; y < e.y - g; ++y)
      for (int x = g; x < e.x - g; ++x) {
        const long li = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const long gi = P.idx(x - g + og.x, y - g + og.y, z - g + og.z);
        ox[li] = P.ox[gi];
        oy[li] = P.oy[gi];
        oz[li] = P.oz[gi];
      }
  mg.setOpenness(CCConst(toDevice(ox, "ox")), CCConst(toDevice(oy, "oy")),
                 CCConst(toDevice(oz, "oz")), 1.0, 1.0, 1.0);
}

// G-K2 on every FP32 level: the cells where an exchanged copy of W differs from W.
long ringMismatches(CutcellMG& mg, int L) {
  CutcellMG::Level& lv = mg.level(L);
  long bad = 0;
  for (const VField* w : {&lv.WX, &lv.WY, &lv.WZ}) {
    VField s("scratch", w->extent(0));
    Kokkos::deep_copy(s, *w);
    lv.devf->exchange(s);
    const auto hw = toHost(*w), hs = toHost(s);
    for (std::size_t i = 0; i < hw.size(); ++i)
      if (std::memcmp(&hw[i], &hs[i], sizeof(VReal)) != 0)
        ++bad;
  }
  return bad;
}

void ringChecks(CutcellMG& mg, const std::string& tag) {
  const int last = mg.fp32Last();
  int ca = 0;
  for (int L = 0; L <= last; ++L) {
    CutcellMG::Level& lv = mg.level(L);
    const bool built = lv.devf != nullptr && lv.WX.extent(0) == lv.n && lv.wOk;
    check(built, tag + ": L" + std::to_string(L) + " (g " + std::to_string(lv.g) +
                     ") carries the FP32 data and its GridHalo<float>");
    if (!built)
      continue;
    ca += lv.g == 2 ? 1 : 0;
    const long bad = ringMismatches(mg, L);
    check(bad == 0, tag + ": G-K2 L" + std::to_string(L) + " ghost WX/WY/WZ == owner (" +
                        std::to_string(bad) + " cells differ)");
  }
  int caMax = 0;
  MPI_Allreduce(&ca, &caMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("  %s: fp32Last %d of %d levels, %d CA (g = 2) FP32 levels\n", tag.c_str(), last,
                mg.nLevels(), caMax);
  bool ok = true;
  for (int L = 0; L <= last; ++L)
    ok = mg.fp32WeightsOk(L) && ok;  // collective: the range flags are allreduced
  check(ok, tag + ": the allreduced range flags pass on every FP32 level");
}

void problemChecks(const Problem& P) {
  const std::string tag = P.name + " np" + std::to_string(size);
  {
    CutcellMG mg;
    setUpDist(mg, P, CutcellMG::kVcycleFp32);
    check(mg.fp32Last() >= 0 && mg.fp32Carriable(), tag + ": carriable distributed");
    ringChecks(mg, tag + " build");
    // the setter's rebuild from the stored openness
    mg.setVcyclePrecision(CutcellMG::kVcycleFp64);
    mg.setVcyclePrecision(CutcellMG::kVcycleFp32);
    ringChecks(mg, tag + " setter rebuild");
    const char* why = mg.fp32VcycleIneligible(true);
    check(why == nullptr, tag + ": eligible distributed (WO-D2): " + (why ? why : "eligible"));
  }
  {
    CutcellMG mg;
    setUpDist(mg, P, CutcellMG::kVcycleFp64);
    bool none = true;
    for (int L = 0; L < mg.nLevels(); ++L)
      none = none && mg.level(L).devf == nullptr && mg.level(L).WX.extent(0) == 0 &&
             mg.level(L).xf.extent(0) == 0;
    check(none, tag + ": an 'fp64' build creates no GridHalo<float> and no FP32 arrays");
  }
}

// ---- WO-D2a ------------------------------------------------------------------------------------
const bool kAutoFp32 = !CutcellMG::kHostMemory;  // WO-A: 'auto' is FP32 on device backends only
constexpr int kMaxit = 200;
constexpr double kRtol = 1e-10;

// Level 0's inner cell (x, y, z) (local, ghosts included) -> the global (single-rank) index.
long globalIndex(const Problem& P, const CutcellMG::Level& l0, int x, int y, int z) {
  return P.idx(x - l0.g + l0.og.x, y - l0.g + l0.og.y, z - l0.g + l0.og.z);
}
template <class F>
void forInner(const CutcellMG::Level& l0, F f) {
  const C3 e = l0.ext;
  const int g = l0.g;
  for (int z = g; z < e.z - g; ++z)
    for (int y = g; y < e.y - g; ++y)
      for (int x = g; x < e.x - g; ++x)
        f(x, y, z, (long)x + (long)y * e.x + (long)z * (long)e.x * e.y);
}
// This rank's block of a global (single-rank-indexed) field, ghosts 0.
std::vector<double> sliceOf(const Problem& P, const CutcellMG::Level& l0,
                            const std::vector<double>& global) {
  std::vector<double> v(l0.n, 0.0);
  forInner(l0, [&](int x, int y, int z, long li) { v[li] = global[globalIndex(P, l0, x, y, z)]; });
  return v;
}
// Over this rank's inner cells: max|a - ref| (NaN-propagating) and the memcmp-unequal count; both
// reduced over the ranks.
struct Diff {
  double maxAbs = 0.0, refMax = 0.0;
  long bits = 0;
};
Diff diffInner(const Problem& P, const CutcellMG::Level& l0, const std::vector<double>& a,
               const std::vector<double>& ref) {
  Diff d;
  forInner(l0, [&](int x, int y, int z, long li) {
    const double r = ref[globalIndex(P, l0, x, y, z)];
    const double v = std::fabs(a[li] - r);
    if (!(v <= d.maxAbs))
      d.maxAbs = v;
    d.refMax = std::max(d.refMax, std::fabs(r));
    if (std::memcmp(&a[li], &r, sizeof(double)) != 0)
      ++d.bits;
  });
  double m[2] = {d.maxAbs, d.refMax}, gm[2];
  MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  long gb = 0;
  MPI_Allreduce(&d.bits, &gb, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return Diff{gm[0], gm[1], gb};
}
std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.3e", v);
  return b;
}

struct Run {
  int it = -1, restarts = 0, discarded = -1;
  bool failed = false, fp32 = false, threw = false;
  std::vector<double> x;  // the full level-0 array (single rank: the global array)
  bool converged() const { return !threw && !failed && it < kMaxit; }
};
std::string itStr(const Run& r) {
  return r.threw
             ? std::string("threw")
             : std::to_string(r.it) + (r.failed ? " failed" : "") + (r.fp32 ? " fp32" : " fp64");
}
bool gLineRelax = true;  // A2.4: the line smoother switch every solveOn applies (default on)
// One solve of P on `mg` (driver 0 = PCG, 1 = FCG) from x = 0 with the fault armed for it.
Run solveOn(CutcellMG& mg, const std::vector<double>& hb, int driver, int kind = 0, int at = 0,
            int maxit = kMaxit, double rtol = kRtol) {
  const std::size_t n = hb.size();
  CCField b = toDevice(hb, "b"), x("x", n), r("r", n), p("p", n), z("z", n), zp("zp", n),
          Ap("Ap", n);
  mg.setLineRelaxation(gLineRelax);  // A2.4 (default on; lineChecks' control arm turns it off)
  mg.debugSetFp32Fault(kind, at);
  Run out;
  try {
    out.it = driver == 0 ? mg.solvePCG(b, x, r, p, z, Ap, maxit, rtol, 2, 2, 12)
                         : mg.solveFCG(b, x, r, p, z, zp, Ap, maxit, rtol, 2, 2, 12);
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
// The single-rank reference, solved on rank 0 and broadcast (the ranks share one GPU on a device
// backend: a redundant reference per rank would multiply the test's cost by np).
Run single(const Problem& P, int mode, int driver, int kind = 0, int at = 0) {
  Run r;
  if (rank == 0) {
    CutcellMG mg;
    setUp(mg, P, mode);
    r = solveOn(mg, rhsOf(P), driver, kind, at);
  }
  int v[6] = {r.it, r.restarts, r.discarded, r.failed, r.fp32, r.threw};
  MPI_Bcast(v, 6, MPI_INT, 0, MPI_COMM_WORLD);
  long n = (long)r.x.size();
  MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  r.x.resize((std::size_t)n);
  MPI_Bcast(r.x.data(), (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  r.it = v[0];
  r.restarts = v[1];
  r.discarded = v[2];
  r.failed = v[3] != 0;
  r.fp32 = v[4] != 0;
  r.threw = v[5] != 0;
  return r;
}
Run dist(const Problem& P, int mode, int driver, int kind = 0, int at = 0, int ca = kCaBoth,
         int maxit = kMaxit, double rtol = kRtol, int teleForce = -1) {
  CutcellMG mg;
  setUpDist(mg, P, mode, ca, teleForce);
  return solveOn(mg, sliceOf(P, mg.level(0), rhsOf(P)), driver, kind, at, maxit, rtol);
}
// The level 0 of a distributed setup of P (for the index maps of a dist() result).
struct DistLevel0 {
  CutcellMG mg;
  explicit DistLevel0(const Problem& P, int ca = kCaBoth) {
    setUpDist(mg, P, CutcellMG::kVcycleFp64, ca);
  }
  const CutcellMG::Level& l0() { return mg.level(0); }
};
bool sameRun(const Run& a, const Run& b) {
  return a.x.size() == b.x.size() &&
         std::memcmp(a.x.data(), b.x.data(), a.x.size() * sizeof(double)) == 0 && a.it == b.it;
}
// Every rank's flag set (MPI_LAND of a local bool).
bool allRanks(bool v) {
  int a = v ? 1 : 0, g = 0;
  MPI_Allreduce(&a, &g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return g == 1;
}

// One FP32 V-cycle z = B r of the mean-free r (distributed or single rank, as `mg` is set up).
std::vector<double> oneVcycle(CutcellMG& mg, const std::vector<double>& hr) {
  CutcellMG::Level& l0 = mg.level(0);
  const std::size_t n = l0.n;
  {  // the driver's smoothing schedule (2, 2, 12): a zero-iteration solve sets it
    CCField s0("s0", n), s1("s1", n), s2("s2", n), s3("s3", n), s4("s4", n), s5("s5", n);
    mg.solvePCG(s0, s1, s2, s3, s4, s5, 0, 1e-8, 2, 2, 12);
  }
  CCField r = toDevice(hr, "r"), z("z", n);
  mg.removeMean(l0, r);
  mg.precondVcycleFp32(z, r, std::ilogb(mg.maxabs(l0, r)), -1, CCField());
  return toHost(z);
}

// The true residual max|P(b - A x)| / max|P b| of a distributed solution (the attainable floor).
double trueResidual(CutcellMG& mg, const std::vector<double>& hb, const std::vector<double>& hx) {
  CutcellMG::Level& l0 = mg.level(0);
  CCField b = toDevice(hb, "b"), x = toDevice(hx, "x"), Ap("Ap", l0.n);
  mg.removeMean(l0, b);
  const double bn = mg.maxabs(l0, b);
  mg.matvecOverlap(l0, Ap, x);
  mg.axpy(b, -1.0, Ap);
  mg.removeMean(l0, b);
  return mg.maxabs(l0, b) / bn;
}

int within(int ref) {  // max(1, 5 %) of an iteration count
  return std::max(1, (int)std::ceil(0.05 * ref));
}

void d2aChecks(const Problem& P) {
  const std::string tag = P.name + " np" + std::to_string(size);
  DistLevel0 D(P);
  const CutcellMG::Level& l0 = D.l0();
  {
    CutcellMG probe;
    setUpDist(probe, P, CutcellMG::kVcycleFp32);
    const char* why = probe.fp32VcycleIneligible(true);
    if (why != nullptr) {
      check(false, tag + ": ineligible distributed: " + why);
      return;
    }
    if (rank == 0)
      std::printf("  %s: %d levels, fp32Last %d, eligible\n", tag.c_str(), probe.nLevels(),
                  probe.fp32Last());
  }
  const char* dn[2] = {"PCG", "FCG"};
  std::vector<int> modes{CutcellMG::kVcycleFp32, CutcellMG::kVcycleFp64};
  if (kAutoFp32)
    modes.push_back(CutcellMG::kVcycleAuto);
  auto mname = [](int m) {
    return m == CutcellMG::kVcycleFp32 ? "fp32" : (m == CutcellMG::kVcycleFp64 ? "fp64" : "auto");
  };
  bool fp64Converges[2] = {false, false};  // at this np, per driver: (d) runs there
  for (int driver : {0, 1}) {
    Run ref[3], got[3];
    for (std::size_t k = 0; k < modes.size(); ++k)
      ref[k] = single(P, modes[k], driver);
    // np > 1: (a) needs the configurations the single-rank solve converges on, (c) those 'fp64'
    // converges on; a configuration where neither converges (ratio 1e4: cyl, pack, box64 PCG) is
    // n/a at np > 1 and is not solved (the np = 1 memcmp of G-DD1 covers it)
    const bool run = size == 1 || ref[0].converged() || ref[1].converged();
    if (!run) {
      if (rank == 0)
        std::printf("  %-100s n/a\n", (tag + " " + dn[driver] +
                                       ": G-DD2 (a), (c) single rank converges in neither mode (" +
                                       itStr(ref[0]) + ", " + itStr(ref[1]) + ")")
                                          .c_str());
      continue;
    }
    for (std::size_t k = 0; k < modes.size(); ++k)
      got[k] = dist(P, modes[k], driver);
    fp64Converges[driver] = got[1].converged();
    for (std::size_t k = 0; k < modes.size(); ++k) {
      const std::string t = tag + " " + dn[driver] + " '" + mname(modes[k]) + "'";
      const Diff d = diffInner(P, l0, got[k].x, ref[k].x);
      const bool itEq = allRanks(got[k].it == ref[k].it);
      if (modes[k] == CutcellMG::kVcycleFp64)
        continue;  // the FP64 classes are the existing gates; here only the iteration reference
      if (!got[k].fp32 && !(modes[k] == CutcellMG::kVcycleAuto && got[k].restarts > 0)) {
        check(false, t + ": did not run the FP32 V-cycle (" + itStr(got[k]) + ")");
        continue;
      }
      if (size == 1) {  // G-DD1 (b), (c)
        check(d.bits == 0 && itEq && got[k].failed == ref[k].failed,
              t + ": G-DD1 x memcmp == single rank (" + std::to_string(d.bits) +
                  " cells differ), iterations " + itStr(got[k]) + " / " + itStr(ref[k]));
      } else if (ref[k].converged()) {  // G-DD2 (a)
        const bool itOk = std::abs(got[k].it - ref[k].it) <= within(ref[k].it);
        check(d.maxAbs <= 1e-8 && allRanks(itOk),
              t + ": G-DD2 (a) max|x - x1| " + num(d.maxAbs) + " (max|x| " + num(d.refMax) +
                  "), iterations " + itStr(got[k]) + " vs " + itStr(ref[k]));
      } else if (rank == 0) {
        std::printf("  %-100s n/a\n", (t + ": G-DD2 (a) single rank does not converge (" +
                                       itStr(ref[k]) + "), np " + itStr(got[k]))
                                          .c_str());
      }
    }
    // G-DD2 (c): 'fp32' vs 'fp64' at this np where 'fp64' converges
    if (got[1].converged())
      check(got[0].it <= got[1].it + within(got[1].it),
            tag + " " + dn[driver] + ": G-DD2 (c) 'fp32' " + itStr(got[0]) + " <= 'fp64' " +
                itStr(got[1]) + " + max(1, 5 %)");
    else if (rank == 0)
      std::printf("  %-100s n/a\n",
                  (tag + " " + dn[driver] + ": G-DD2 (c) 'fp64' does not converge (" +
                   itStr(got[1]) + "), 'fp32' " + itStr(got[0]))
                      .c_str());
  }
  // G-DD1 (a) / G-DD2 (b): one V-cycle
  const bool vcycleCase = P.name == "cyl" || P.name == "box64";
  if (size == 1 || vcycleCase) {
    CutcellMG ms, md;
    setUp(ms, P, CutcellMG::kVcycleFp32);
    setUpDist(md, P, CutcellMG::kVcycleFp32);
    const bool sameDepth = ms.nLevels() == md.nLevels();
    const std::vector<double> rhs = rhsOf(P);
    const std::vector<double> zs = oneVcycle(ms, rhs);
    const std::vector<double> zd = oneVcycle(md, sliceOf(P, md.level(0), rhs));
    const Diff d = diffInner(P, md.level(0), zd, zs);
    if (size == 1)
      check(d.bits == 0, tag + ": G-DD1 (a) one FP32 V-cycle z memcmp == single rank (" +
                             std::to_string(d.bits) + " cells differ)");
    else if (sameDepth)
      check(d.maxAbs <= 1e-11 * d.refMax,
            tag + ": G-DD2 (b) one FP32 V-cycle max|dz|/max|z| " + num(d.maxAbs / d.refMax));
    else if (rank == 0)
      std::printf("  %s: G-DD2 (b) n/a (depth %d vs single rank %d)\n", tag.c_str(), md.nLevels(),
                  ms.nLevels());
  }
  // G-DD2 (d): the attainable floor at np 2, where 'fp64' converges
  if (size == 2)
    for (int driver : {0, 1}) {
      if (!fp64Converges[driver])
        continue;
      double fl[2];
      int its[2];
      for (int k = 0; k < 2; ++k) {
        CutcellMG mg;
        setUpDist(mg, P, k == 0 ? CutcellMG::kVcycleFp32 : CutcellMG::kVcycleFp64);
        const std::vector<double> hb = sliceOf(P, mg.level(0), rhsOf(P));
        const Run r = solveOn(mg, hb, driver, 0, 0, 300, 1e-14);
        its[k] = r.it;
        fl[k] = trueResidual(mg, hb, r.x);
      }
      check(fl[0] <= 2.0 * fl[1], tag + " " + dn[driver] + ": G-DD2 (d) floor fp32 " + num(fl[0]) +
                                      " (" + std::to_string(its[0]) + " it) <= 2 x fp64 " +
                                      num(fl[1]) + " (" + std::to_string(its[1]) + " it)");
    }
  // G-DD2 (e): CA on == CA off, bitwise (periodic problems; CA needs no domain BC)
  if (size > 1 && vcycleCase) {
    // level 0 is g = 1 either way (CA widens the coarse levels only): the same layout
    // (30 iterations: bitwise equality needs no convergence)
    const Run on = dist(P, CutcellMG::kVcycleFp32, 0, 0, 0, kCaBoth, 30);
    const Run off = dist(P, CutcellMG::kVcycleFp32, 0, 0, 0, 0, 30);
    check(allRanks(sameRun(on, off)),
          tag + ": G-DD2 (e) CA on == CA off bitwise (" + itStr(on) + " / " + itStr(off) + ")");
  }
  // G-DD2 (g): two runs at np 2, bitwise
  if (size == 2) {
    const Run a = dist(P, CutcellMG::kVcycleFp32, 0, 0, 0, kCaBoth, 30),
              b = dist(P, CutcellMG::kVcycleFp32, 0, 0, 0, kCaBoth, 30);
    check(allRanks(sameRun(a, b)), tag + ": G-DD2 (g) two runs bitwise (" + itStr(a) + ")");
  }
}

// G-R (h): the restart on the distributed driver.
void restartChecks(const Problem& P) {
  const std::string tag = P.name + " np" + std::to_string(size);
  {
    CutcellMG probe;
    setUpDist(probe, P, CutcellMG::kVcycleFp32);
    if (probe.fp32VcycleIneligible(true) != nullptr)
      return;  // reported by d2aChecks
  }
  for (int driver : {0, 1}) {
    const std::string t = tag + (driver == 0 ? " PCG" : " FCG");
    const Run ref64 = dist(P, CutcellMG::kVcycleFp64, driver);
    auto restartedIsFp64 = [&](const Run& a) {
      return allRanks(sameRun(a, ref64) && a.failed == ref64.failed && !a.fp32 && a.restarts == 1 &&
                      !a.threw);
    };
    auto restartsAgree = [&](const Run& a) {
      int mn = 0, mx = 0;
      MPI_Allreduce(&a.restarts, &mn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&a.restarts, &mx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      return mn == mx;
    };
    if (!kAutoFp32) {
      const Run a = dist(P, CutcellMG::kVcycleAuto, driver, 1, 3);
      check(allRanks(sameRun(a, ref64) && !a.fp32 && a.restarts == 0),
            t + ": G-R (h) host 'auto' == 'fp64', the fault inert, no restart (" + itStr(a) + ")");
    } else {
      const Run a = dist(P, CutcellMG::kVcycleAuto, driver, 1, 3);
      check(restartedIsFp64(a) && restartsAgree(a),
            t + ": G-R (h)(a) kind 1 at 3 -> restart == fp64 cold (" + itStr(a) + ", discarded " +
                std::to_string(a.discarded) + ")");
      const Run b = dist(P, CutcellMG::kVcycleAuto, driver, 2, 2);
      check(restartedIsFp64(b) && b.discarded == 2 && restartsAgree(b),
            t + ": G-R (h)(b) kind 2 at 2 -> restart == fp64 cold, discarded " +
                std::to_string(b.discarded));
      if (size == 1) {
        const Run sa = single(P, CutcellMG::kVcycleAuto, driver, 1, 3);
        DistLevel0 D(P);
        const Diff d = diffInner(P, D.l0(), a.x, sa.x);
        check(d.bits == 0 && a.it == sa.it && a.restarts == sa.restarts,
              t + ": G-R (h) np 1 restart memcmp == the single-rank restart (" +
                  std::to_string(d.bits) + " cells differ)");
      }
    }
    const Run c1 = dist(P, CutcellMG::kVcycleFp32, driver, 1, 3);
    const Run c2 = dist(P, CutcellMG::kVcycleFp32, driver, 2, 2);
    check(allRanks(c1.restarts == 0 && c1.failed && c1.fp32 && c2.restarts == 0 && c2.it == 2 &&
                   c2.fp32),
          t + ": G-R (h)(c) forced 'fp32': no restart; kind 1 " + itStr(c1) + ", kind 2 " +
              itStr(c2));
  }
}

// G-DD2 (f) (WO-D2b1): a telescoped hierarchy, natural (teleForce < 0) or forced at teleForce.
void teleChecks(const Problem& P, int teleForce) {
  const std::string tag =
      P.name + (teleForce >= 0 ? " forced-tele L" + std::to_string(teleForce) : " natural-tele") +
      " np" + std::to_string(size);
  CutcellMG probe;
  setUpDist(probe, P, CutcellMG::kVcycleFp32, kCaBoth, teleForce);
  int tl = -1;
  for (int L = 0; L < probe.nLevels(); ++L)
    if (probe.level(L).tele) {
      tl = L;
      break;
    }
  int tlMin = 0, tlMax = 0;
  MPI_Allreduce(&tl, &tlMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&tl, &tlMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (tlMax < 0) {
    if (rank == 0)
      std::printf("  %s: does not telescope at this np (n/a)\n", tag.c_str());
    return;
  }
  check(tlMin == tlMax && probe.fp32Last() == tl,
        tag + ": the first telescope level " + std::to_string(tl) + " is fp32Last (" +
            std::to_string(probe.fp32Last()) + ") on every rank");
  const char* why = probe.fp32VcycleIneligible(true);
  check(why == nullptr, tag + ": eligible: " + (why ? why : "yes"));
  if (why != nullptr)
    return;
  if (rank == 0)
    std::printf("  %s: %d levels on rank 0, FP32 on 0..%d, FP64 below\n", tag.c_str(),
                probe.nLevels(), tl);
  const char* dn[2] = {"PCG", "FCG"};
  std::vector<int> modes{CutcellMG::kVcycleFp32, CutcellMG::kVcycleFp64};
  if (kAutoFp32)
    modes.push_back(CutcellMG::kVcycleAuto);
  for (int driver : {0, 1}) {
    Run ref[3], got[3];
    for (std::size_t k = 0; k < modes.size(); ++k) {
      ref[k] = single(P, modes[k], driver);
      got[k] = dist(P, modes[k], driver, 0, 0, kCaBoth, kMaxit, kRtol, teleForce);
    }
    for (std::size_t k = 0; k < modes.size(); ++k) {
      if (modes[k] == CutcellMG::kVcycleFp64)
        continue;
      const std::string t =
          tag + " " + dn[driver] + (modes[k] == CutcellMG::kVcycleFp32 ? " 'fp32'" : " 'auto'");
      if (!got[k].fp32) {
        check(false, t + ": did not run the FP32 V-cycle (" + itStr(got[k]) + ")");
        continue;
      }
      const Diff d = diffInner(P, probe.level(0), got[k].x, ref[k].x);
      if (ref[k].converged()) {
        const bool itOk = std::abs(got[k].it - ref[k].it) <= within(ref[k].it);
        check(d.maxAbs <= 1e-8 && allRanks(itOk),
              t + ": G-DD2 (f)(a) max|x - x1| " + num(d.maxAbs) + " (max|x| " + num(d.refMax) +
                  "), iterations " + itStr(got[k]) + " vs " + itStr(ref[k]));
      } else if (rank == 0) {
        std::printf(
            "  %-100s n/a\n",
            (t + ": G-DD2 (f)(a) single rank does not converge (" + itStr(ref[k]) + ")").c_str());
      }
    }
    if (got[1].converged())
      check(got[0].it <= got[1].it + within(got[1].it),
            tag + " " + dn[driver] + ": G-DD2 (f)(c) 'fp32' " + itStr(got[0]) + " <= 'fp64' " +
                itStr(got[1]) + " + max(1, 5 %)");
    else if (rank == 0)
      std::printf("  %-100s n/a\n",
                  (tag + " " + dn[driver] + ": G-DD2 (f)(c) 'fp64' does not converge (" +
                   itStr(got[1]) + ")")
                      .c_str());
  }
  // (b): one V-cycle against single rank (the telescope keeps the depth). (b)'s premise (§8) is a
  // V-cycle of the same precision split: where the first telescope point is the last non-bottom
  // level, the FP64 transition sits where single rank's FP32 -> FP64 bottom interface does and
  // (b) is gated; above it the distributed V-cycle runs FP64 below the telescope point (§3.2)
  // where single rank runs FP32, a different preconditioner -- reported against the single-rank
  // FP32-vs-FP64 V-cycle difference for scale, not gated.
  CutcellMG ms, md;
  setUp(ms, P, CutcellMG::kVcycleFp32);
  setUpDist(md, P, CutcellMG::kVcycleFp32, kCaBoth, teleForce);
  int depth = md.nLevels(), depthMax = 0;
  MPI_Allreduce(&depth, &depthMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const std::vector<double> rhs = rhsOf(P);
  const std::vector<double> zs = oneVcycle(ms, rhs);
  const std::vector<double> zd = oneVcycle(md, sliceOf(P, md.level(0), rhs));
  const Diff d = diffInner(P, md.level(0), zd, zs);
  if (depthMax != ms.nLevels()) {
    if (rank == 0)
      std::printf("  %s: G-DD2 (f)(b) n/a (depth %d vs single rank %d)\n", tag.c_str(), depthMax,
                  ms.nLevels());
  } else if (tl == ms.nLevels() - 2) {
    check(d.maxAbs <= 1e-11 * d.refMax,
          tag + ": G-DD2 (f)(b) one FP32 V-cycle max|dz|/max|z| " + num(d.maxAbs / d.refMax));
  } else {
    CutcellMG m64;  // the single-rank FP64 V-cycle of the same r
    setUp(m64, P, CutcellMG::kVcycleFp64);
    CutcellMG::Level& l0 = m64.level(0);
    {
      CCField s0("s0", l0.n), s1("s1", l0.n), s2("s2", l0.n), s3("s3", l0.n), s4("s4", l0.n),
          s5("s5", l0.n);
      m64.solvePCG(s0, s1, s2, s3, s4, s5, 0, 1e-8, 2, 2, 12);
    }
    CCField r = toDevice(rhs, "r"), z("z", l0.n);
    m64.removeMean(l0, r);
    m64.precondVcycle(z, r);
    const std::vector<double> z64 = toHost(z);
    double dm = 0.0, zm = 0.0;
    for (int zc = 1; zc <= P.nz; ++zc)
      for (int y = 1; y <= P.ny; ++y)
        for (int x = 1; x <= P.nx; ++x) {
          const long i = P.idx(x - 1, y - 1, zc - 1);
          dm = std::max(dm, std::fabs(z64[i] - zs[i]));
          zm = std::max(zm, std::fabs(zs[i]));
        }
    if (rank == 0)
      std::printf(
          "  %s: G-DD2 (f)(b) info: FP64 below L%d (single rank FP32 to L%d): "
          "max|dz|/max|z| %s vs single rank (single-rank fp32 vs fp64 V-cycle %s)\n",
          tag.c_str(), tl, ms.nLevels() - 2, num(d.maxAbs / d.refMax).c_str(),
          num(dm / zm).c_str());
  }
}

// doc/mg_depth_policy.md A2.4 / A2.5 G-LR (4), at the CutcellMG level: the zebra line smoother
// (default on) on hierarchies with triggered smoothed levels, distributed against the single-rank
// reference -- np 1 memcmp-equal x and equal iterations (MG-PCG, rtol 1e-10), np > 1 G-DD2 (a)'s
// bound (max|dx| <= 1e-8, iterations within max(1, 5 %)), each with LR off as the control. aniso:
// 64x64x25 periodic, a heavy blob at 1e2, AUTO depth (s = z periodic, Sherman-Morrison); anisoW:
// the same between z walls (Thomas); thinW: 64x64x4 between z walls at 5 levels, 'fp64' and forced
// 'fp32' (every FP32 level even: the FP32 line kernel, distributed through the FP32 stencil halo).
// The LR-off arm is the control (the point smoother's own np 1 / np > 1 classes on the same
// problems).
Problem lineProblem(const std::string& name, int nz, int levels, bool walls) {
  int bc[6] = {0, 0, 0, 0, 0, 0};
  if (walls)
    bc[4] = bc[5] = 1;
  auto rho = [nz](double x, double y, double z) {
    const double dx = x - 24, dy = y - 40, dz = (z - 0.5 * nz) * 3.0;
    return dx * dx + dy * dy + dz * dz < 144.0 ? 1e2 : 1.0;
  };
  return makeProblem(name, 64, 64, nz, levels, bc, [](double, double, double) { return 1.0; }, rho);
}
void lineChecks(const Problem& P, std::vector<int> modes) {
  const std::string tag = P.name + " np" + std::to_string(size);
  std::vector<int> ls;
  {
    CutcellMG ms;
    setUp(ms, P, CutcellMG::kVcycleFp64);
    ls = ms.lineSmoothedLevels();
  }
  check(!ls.empty(),
        tag + ": the hierarchy has triggered smoothed levels (" + std::to_string(ls.size()) + ")");
  int held = 0, on = 0;  // the triggered levels this rank holds (telescoping), and how many LR
  {
    CutcellMG md;
    setUpDist(md, P, CutcellMG::kVcycleFp64);
    for (int L : ls)
      if (L < md.nLevels()) {
        ++held;
        on += md.lineRelaxOn(md.level(L)) ? 1 : 0;
      }
  }
  check(on == held && (size > 1 || held == (int)ls.size()),
        tag + ": every triggered level a rank holds is line-relaxed (s whole on the blocks)");
  // LR off first: the point smoother's own np 1 class on the same problem is the control. At np 1
  // the LR memcmp is a check where the control is memcmp-equal; where it is not (aniso: the odd
  // periodic z, a pre-existing np 1 difference of the point path) the LR line is printed only.
  std::vector<bool> ctlEq(modes.size(), false);
  for (int lr : {0, 1})
    for (std::size_t k = 0; k < modes.size(); ++k) {
      const int mode = modes[k];
      gLineRelax = lr == 1;
      const std::string t =
          tag + (mode == CutcellMG::kVcycleFp32 ? " 'fp32'" : " 'fp64'") + (lr ? " LR" : " LR off");
      const Run ref = single(P, mode, 0);
      const Run got = dist(P, mode, 0);
      DistLevel0 D(P);
      const Diff d = diffInner(P, D.l0(), got.x, ref.x);
      const bool itEq = allRanks(got.it == ref.it);
      if (size == 1) {
        const bool eq = d.bits == 0 && itEq && got.failed == ref.failed;
        const std::string what = t + ": x memcmp == single rank (" + std::to_string(d.bits) +
                                 " cells differ), iterations " + itStr(got) + " / " + itStr(ref);
        if (!lr)
          ctlEq[k] = eq;
        if (lr && ctlEq[k])
          check(eq, what);
        else if (rank == 0)
          std::printf(
              "  %-100s %s\n", what.c_str(),
              lr ? "n/a (the LR-off control differs too)" : (eq ? "(control) =" : "(control) !="));
      } else {
        const bool itOk = allRanks(std::abs(got.it - ref.it) <= within(ref.it));
        check(ref.converged() && got.converged() && d.maxAbs <= 1e-8 && itOk,
              t + ": max|dx| " + num(d.maxAbs) + " <= 1e-8, iterations " + itStr(got) + " vs " +
                  itStr(ref));
      }
    }
  gLineRelax = true;
}
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    std::printf("mg_fp32_vcycle_mpi np=%d (doc/vof_projection_distributed_design.md §5, G-K2)\n",
                size);
  for (const Problem& P : {cylProblem(), boxProblem()})
    problemChecks(P);
  for (const Problem& P :
       {columnProblem(), cylProblem(), packProblem(), slabProblem(1e2), boxProblem()})
    d2aChecks(P);
  restartChecks(columnProblem());
  lineChecks(lineProblem("aniso", 25, 0, false), {CutcellMG::kVcycleFp64});
  lineChecks(lineProblem("anisoW", 25, 0, true), {CutcellMG::kVcycleFp64});
  lineChecks(lineProblem("thinW", 4, 5, true), {CutcellMG::kVcycleFp64, CutcellMG::kVcycleFp32});
  if (size == 4 || size == 8) {  // G-DD2 (f) (WO-D2b1)
    Problem c6 = columnProblem();
    c6.levels = 6;
    c6.name = "column6";
    teleChecks(c6, -1);
  }
  if (size == 4)
    teleChecks(columnProblem(), 1);
  if (rank == 0)
    std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
  Kokkos::finalize();
  MPI_Finalize();
  return fails ? 1 : 0;
}
