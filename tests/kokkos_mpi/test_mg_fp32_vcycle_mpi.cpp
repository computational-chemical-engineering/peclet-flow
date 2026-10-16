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
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
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
void setUpDist(CutcellMG& mg, const Problem& P, int mode) {
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
    check(why != nullptr && std::string(why).find("distributed") != std::string::npos,
          tag + ": still ineligible (WO-D1 inert): " + (why ? why : "(eligible!)"));
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
  if (rank == 0)
    std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
  Kokkos::finalize();
  MPI_Finalize();
  return fails ? 1 : 0;
}
