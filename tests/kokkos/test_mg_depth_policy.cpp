// doc/mg_depth_policy.md WO-1: the AUTO pressure-MG depth (CutcellMG::init with nLevels == 0),
// single rank, hierarchy only (no operator, no solve). Every row of the note's §4.4 table that fits
// in memory here: the resolved depth D and the bottom's global dims, plus the natural end of a
// badly factored grid (400^3), the 'smoother'-forced singular operator (only a <= 4-cell bottom is
// admitted), an anchored (outflow) operator (depth 4 until WO-5), and the explicit depths, which
// stay exact. The two largest rows of §4.4 (512x512x256, 1024x1024x512) and predict(levels=0) ==
// the built hierarchy are in tests/kokkos_mpi/test_mg_depth_policy_mpi.cpp: CutcellMG::predict is
// compiled with PECLET_FLOW_MPI only, and a 1024x1024x512 level 0 does not fit a test.
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

int g_fail = 0;

struct Row {
  C3 grid;
  int levels;  // 0 = AUTO
  int mode;    // setAgglomerationMode: -1 auto (the Solver default), 0 smoother, 1 agglomerated
  bool outflow;
  int wantD;
  C3 wantBottom;
  const char* what;
  std::vector<int> wantLine = {};  // Amendment A2.3: the triggered smoothed levels
};

void check(const Row& r) {
  CutcellMG mg;
  if (r.outflow) {
    const int bc[6] = {2, 3, 1, 1, 1, 1};  // inflow -x, outflow +x, walls: the anchored operator
    mg.setBoundaryConditions(bc);
  }
  mg.setAgglomerationMode(r.mode);  // before init, as Solver::setSolidInitPressureMg does
  mg.init(r.grid.x, r.grid.y, r.grid.z, r.levels);
  const std::vector<C3> ratio = mg.levelRatios();
  C3 b = r.grid;
  for (std::size_t L = 0; L + 1 < ratio.size(); ++L)
    b = C3{b.x / ratio[L].x, b.y / ratio[L].y, b.z / ratio[L].z};
  const int D = mg.nLevels();
  // Amendment A2.3: of the §4.4 rows only 16x16x4 has a triggered smoothed level, its smoothed
  // bottom L3 (2x2x2, z blocked at L1, A' 4 > theta; coordinator ruling 2026-10-26).
  const std::vector<int> line = mg.lineSmoothedLevels();
  const bool lineOk = line == r.wantLine;
  const bool ok = D == r.wantD && b.x == r.wantBottom.x && b.y == r.wantBottom.y &&
                  b.z == r.wantBottom.z && D == (int)ratio.size() && lineOk;
  if (!ok)
    ++g_fail;
  std::printf(
      "%s %4dx%4dx%4d levels=%d mode=%2d%s: D %d bottom %dx%dx%d (want %d, %dx%dx%d)%s  %s\n",
      ok ? "PASS" : "FAIL", r.grid.x, r.grid.y, r.grid.z, r.levels, r.mode,
      r.outflow ? " outflow" : "", D, b.x, b.y, b.z, r.wantD, r.wantBottom.x, r.wantBottom.y,
      r.wantBottom.z, line.empty() ? "" : " LINE-TRIGGERED", r.what);
  std::fflush(stdout);
}

// Amendment A2.3 / WO-A2a: the triggered smoothed levels (T holds and the V-cycle smooths the
// level, a smoothed bottom included) of a hierarchy whose strong axis stops coarsening.
struct LineRow {
  C3 grid;
  int levels;  // 0 = AUTO
  bool outflow;
  int wantD;
  std::vector<int> want;  // the triggered smoothed levels
  const char* what;
};

void checkLine(const LineRow& r) {
  CutcellMG mg;
  if (r.outflow) {
    const int bc[6] = {2, 3, 1, 1, 1, 1};
    mg.setBoundaryConditions(bc);
  }
  mg.setAgglomerationMode(-1);  // the Solver default
  mg.init(r.grid.x, r.grid.y, r.grid.z, r.levels);
  const int D = mg.nLevels();
  const std::vector<int> got = mg.lineSmoothedLevels();
  const std::vector<int> ax = mg.lineAxes();
  bool axisOk = true;  // every triggered level's line axis is the blocked one, z, on these grids
  for (int L : got)
    axisOk = axisOk && ax[L] == 2;
  const bool ok = D == r.wantD && got == r.want && axisOk;
  if (!ok)
    ++g_fail;
  std::string gs = "{", ws = "{", as;
  for (std::size_t i = 0; i < got.size(); ++i)
    gs += (i ? "," : "") + std::to_string(got[i]);
  for (std::size_t i = 0; i < r.want.size(); ++i)
    ws += (i ? "," : "") + std::to_string(r.want[i]);
  for (std::size_t L = 0; L < ax.size(); ++L)
    as += (L ? " " : "") + std::to_string(ax[L]);
  std::printf(
      "%s line %4dx%4dx%4d levels=%d%s: D %d triggered smoothed %s} (want %d, %s}) "
      "lineAxis [%s]  %s\n",
      ok ? "PASS" : "FAIL", r.grid.x, r.grid.y, r.grid.z, r.levels, r.outflow ? " outflow" : "", D,
      gs.c_str(), r.wantD, ws.c_str(), as.c_str(), r.what);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // Predicate spot checks (§4.1): the engine's own admission, and the small-bottom clause.
    const int per[6] = {0, 0, 0, 0, 0, 0};
    struct P {
      C3 g;
      bool anchored;
      int mode;
      bool want;
    };
    const P ps[] = {
        {{16, 16, 16}, false, -1, false},  // plane 256 > 192
        {{12, 12, 12}, false, -1, true},   // 1728 cells, plane 144
        {{24, 24, 24}, false, -1, false},  // 13824 > 8192 cells
        {{128, 4, 4}, false, -1, true},    // 2048 cells, plane 16
        {{16, 12, 8}, false, -1, true},    // the bubble column's bottom
        {{4, 4, 4}, false, 0, true},       // every axis <= 4: a smoothed bottom suffices
        {{12, 12, 12}, false, 0, false},   // 'smoother' forced: only small bottoms
        {{16, 4, 4}, true, -1, true},      // anchored: admitted at depth 4 until WO-5
    };
    for (const P& p : ps) {
      const bool got = CutcellMG::bottomAdmissible(p.g, 4, p.anchored, p.mode, per);
      if (got != p.want)
        ++g_fail;
      std::printf("%s admissible(%dx%dx%d, anchored %d, mode %d) = %d\n",
                  got == p.want ? "PASS" : "FAIL", p.g.x, p.g.y, p.g.z, (int)p.anchored, p.mode,
                  (int)got);
    }
    // §4.4, single rank, isotropic metric, the Solver's default mode (-1).
    const Row rows[] = {
        {{16, 16, 16}, 0, -1, false, 4, {2, 2, 2}, "state_hash N=16"},
        {{16, 16, 8}, 0, -1, false, 4, {2, 2, 2}, "state_hash"},
        {{16, 16, 4}, 0, -1, false, 4, {2, 2, 2}, "state_hash", {3}},
        {{24, 12, 8}, 0, -1, false, 4, {3, 3, 2}, "state_hash channel"},
        {{48, 48, 48}, 0, -1, false, 4, {6, 6, 6}, "direct at 4"},
        {{64, 64, 64}, 0, -1, false, 4, {8, 8, 8}, "direct at 4"},
        {{96, 96, 96}, 0, -1, false, 4, {12, 12, 12}, "direct at 4"},
        {{128, 96, 64}, 0, -1, false, 4, {16, 12, 8}, "bubble column: unchanged"},
        {{64, 48, 32}, 0, -1, false, 4, {8, 6, 4}, "bottom_direct_mpi grid"},
        {{128, 128, 128}, 0, -1, false, 5, {8, 8, 8}, "16^3 plane 256 -> 5"},
        {{192, 192, 192}, 0, -1, false, 5, {12, 12, 12}, "swarm 192^3"},
        {{256, 256, 256}, 0, -1, false, 6, {8, 8, 8}, "swarm 256^3"},
        {{256, 192, 128}, 0, -1, false, 5, {16, 12, 8}, "bubble column x2"},
        {{2048, 64, 64}, 0, -1, false, 5, {128, 4, 4}, "long channel"},
        {{400, 400, 400}, 0, -1, false, 5, {25, 25, 25}, "natural end, inadmissible"},
        {{128, 32, 32}, 0, -1, true, 4, {16, 4, 4}, "anchored: 4 until WO-5"},
        {{192, 192, 192}, 0, 0, false, 7, {3, 3, 3}, "'smoother' forced: <= 4 per axis"},
        {{192, 192, 192}, 0, 1, false, 5, {12, 12, 12}, "'agglomerated' forced"},
        // explicit depths stay exact
        {{192, 192, 192}, 4, -1, false, 4, {24, 24, 24}, "explicit 4"},
        {{192, 192, 192}, 7, -1, false, 7, {3, 3, 3}, "explicit 7"},
        {{16, 16, 16}, 3, -1, false, 3, {4, 4, 4}, "explicit 3"},
        {{16, 16, 16}, 1, -1, false, 1, {16, 16, 16}, "explicit 1 (pure RB-GS)"},
    };
    for (const Row& r : rows)
      check(r);
    // Amendment A2.3: triggered smoothed levels (A' = 2^(L - beta) along the blocked z).
    const LineRow lines[] = {
        {{128, 128, 25}, 0, false, 5, {2, 3}, "M10: bottom 8x8x25 direct (A' 16, not smoothed)"},
        {{128, 128, 25}, 4, false, 4, {2}, "M10 explicit 4: bottom 16x16x25 GraphAMG"},
        {{512, 512, 100}, 0, false, 7, {4, 5}, "M10b: z stops at 25"},
        {{512, 64, 25}, 0, false, 5, {2, 3}, "M10v grid: bottom 32x4x25"},
        {{34, 34, 116}, 0, false, 3, {}, "fluidized bed: no unique strong axis"},
        {{34, 34, 116}, 0, true, 3, {}, "fluidized bed, anchored"},
        {{8, 8, 18}, 0, false, 3, {}, "small bed: A' 2 = theta"},
        {{8, 8, 18}, 0, true, 3, {}, "small bed, anchored"},
    };
    for (const LineRow& r : lines)
      checkLine(r);
  }
  Kokkos::finalize();
  std::printf("mg_depth_policy: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
