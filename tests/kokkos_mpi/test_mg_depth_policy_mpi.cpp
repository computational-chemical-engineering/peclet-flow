// doc/mg_depth_policy.md WO-1: the AUTO pressure-MG depth under MPI (CutcellMG::initMpi with
// nLevels == 0), hierarchy only (no operator, no solve).
//   T1 telescoping ON (the default): the resolved depth D and every level's GLOBAL dims equal the
//      single-rank init's (rank- and backend-independence of the hierarchy, §2), for 192^3,
//      256x192x128 and 2048x64x64; predict(levels=0, telescope=true) equals the built plan.
//   T2 predict(levels=0, np=1) for the two §4.4 rows too large to build in a test (512x512x256,
//      1024x1024x512) and for the T1 grids against the single-rank init.
//   T3 telescoping OFF: D is where in-place coarsening stops, so a blocked case resolves to fewer
//      levels than single rank (40^3 on the coarse-first level-0 partition with decomposition
//      levels 2 at np 8: blocks odd below the 10^3 level, D 3 < 4 -- the aligned default ORB
//      nests for the full depth of every grid searched, so it cannot show it); at every np D <=
//      single-rank D; predict(levels=0, telescope=false) equals the built plan.
#include <mpi.h>

#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

int g_fail = 0;
int g_rank = 0;

void expect(bool ok, const char* fmt, int a = 0, int b = 0, int c = 0, int d = 0) {
  if (!ok)
    ++g_fail;
  if (g_rank == 0) {
    std::printf("%s ", ok ? "PASS" : "FAIL");
    std::printf(fmt, a, b, c, d);
    std::printf("\n");
    std::fflush(stdout);
  }
}

bool same(C3 a, C3 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Single-rank reference: the per-level global dims of CutcellMG::init(G, 0), mode auto.
std::vector<C3> singleRankDims(C3 G) {
  std::vector<C3> d;
  if (g_rank == 0) {
    CutcellMG ref;
    ref.setAgglomerationMode(-1);
    ref.init(G.x, G.y, G.z, 0);
    C3 g = G;
    for (const C3& r : ref.levelRatios()) {
      d.push_back(g);
      g = C3{g.x / r.x, g.y / r.y, g.z / r.z};
    }
  }
  int n = (int)d.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  d.resize(n);
  MPI_Bcast(d.data(), 3 * n, MPI_INT, 0, MPI_COMM_WORLD);
  return d;
}

// The built plan's global dims on rank 0 (the root of every stage holds every level), broadcast.
std::vector<CutcellMG::PlanRow> builtOnRoot(const CutcellMG& mg) {
  std::vector<CutcellMG::PlanRow> p = mg.builtPlan();
  int n = (int)p.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  p.resize(n);
  std::vector<int> buf(5 * n);
  if (g_rank == 0)
    for (int L = 0; L < n; ++L) {
      buf[5 * L + 0] = p[L].global.x;
      buf[5 * L + 1] = p[L].global.y;
      buf[5 * L + 2] = p[L].global.z;
      buf[5 * L + 3] = p[L].ranks;
      buf[5 * L + 4] = p[L].tele ? 1 : 0;
    }
  MPI_Bcast(buf.data(), 5 * n, MPI_INT, 0, MPI_COMM_WORLD);
  for (int L = 0; L < n; ++L) {
    p[L].global = C3{buf[5 * L + 0], buf[5 * L + 1], buf[5 * L + 2]};
    p[L].ranks = buf[5 * L + 3];
    p[L].tele = buf[5 * L + 4] != 0;
  }
  return p;
}

bool planEqual(const std::vector<CutcellMG::PlanRow>& a, const std::vector<CutcellMG::PlanRow>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t L = 0; L < a.size(); ++L)
    if (!same(a[L].global, b[L].global) || a[L].ranks != b[L].ranks || a[L].tele != b[L].tele)
      return false;
  return true;
}

void printPlan(const char* tag, const std::vector<CutcellMG::PlanRow>& p) {
  if (g_rank != 0)
    return;
  std::printf("  %s:", tag);
  for (const auto& r : p)
    std::printf(" %dx%dx%d[%d%s]", r.global.x, r.global.y, r.global.z, r.ranks, r.tele ? "T" : "");
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  {
    // T1 + T2 (np = 1 part): telescoping ON.
    const C3 grids[] = {{192, 192, 192}, {256, 192, 128}, {2048, 64, 64}};
    for (const C3& G : grids) {
      const std::vector<C3> ref = singleRankDims(G);
      CutcellMG mg;
      mg.setAgglomerationMode(-1);
      mg.setTelescope(true);
      mg.initMpi(G.x, G.y, G.z, 0, MPI_COMM_WORLD);
      const auto built = builtOnRoot(mg);
      printPlan("built", built);
      bool dimsOk = built.size() == ref.size();
      for (std::size_t L = 0; dimsOk && L < ref.size(); ++L)
        dimsOk = same(built[L].global, ref[L]);
      expect(dimsOk, "T1 %dx%dx%d telescope ON: D %d == single-rank D, every level's global dims",
             G.x, G.y, G.z, (int)built.size());
      const auto pred = CutcellMG::predict(G.x, G.y, G.z, size, 0, true);
      printPlan("predict", pred);
      expect(planEqual(pred, built), "T1 %dx%dx%d predict(levels=0, telescope) == built (D %d)",
             G.x, G.y, G.z, (int)pred.size());
      const auto p1 = CutcellMG::predict(G.x, G.y, G.z, 1, 0, true);
      bool p1Ok = p1.size() == ref.size();
      for (std::size_t L = 0; p1Ok && L < ref.size(); ++L)
        p1Ok = same(p1[L].global, ref[L]);
      expect(p1Ok, "T2 %dx%dx%d predict(levels=0, np=1) == single-rank init", G.x, G.y, G.z);
    }
    // T2: the rows too large to build here (§4.4): D and the bottom from predict alone.
    struct Big {
      C3 G;
      int D;
      C3 bottom;
    };
    const Big bigs[] = {{{512, 512, 256}, 6, {16, 16, 8}}, {{1024, 1024, 512}, 7, {16, 16, 8}}};
    for (const Big& b : bigs) {
      const auto p = CutcellMG::predict(b.G.x, b.G.y, b.G.z, 1, 0, true);
      expect((int)p.size() == b.D && same(p.back().global, b.bottom),
             "T2 %dx%dx%d predict(levels=0): D %d, bottom 16x16x8", b.G.x, b.G.y, b.G.z,
             (int)p.size());
    }
    // T3: telescoping OFF.
    {
      const C3 G{40, 40, 40};
      const std::vector<C3> ref = singleRankDims(G);  // 40 20 10 5: D 4
      CutcellMG mg;
      mg.setAgglomerationMode(-1);
      mg.setTelescope(false);
      const auto dec0 = CutcellMG::decomposition((std::size_t)size, G.x, G.y, G.z, 2);
      mg.initMpi(G.x, G.y, G.z, 0, MPI_COMM_WORLD, &dec0);
      const auto built = builtOnRoot(mg);
      printPlan("built (telescope OFF)", built);
      expect(built.size() <= ref.size(), "T3 40^3 telescope OFF np %d: D %d <= single-rank D %d",
             size, (int)built.size(), (int)ref.size());
      if (size == 8)
        expect(built.size() < ref.size(), "T3 40^3 telescope OFF np 8: blocked, D %d < %d",
               (int)built.size(), (int)ref.size());
      const auto pred = CutcellMG::predict(G.x, G.y, G.z, size, 0, false, 4, 2);
      expect(planEqual(pred, built), "T3 40^3 predict(levels=0, telescope=false) == built (D %d)",
             (int)pred.size());
    }
  }
  Kokkos::finalize();
  if (g_rank == 0)
    std::printf("mg_depth_policy_mpi np %d: %s (%d failures)\n", size, g_fail ? "FAIL" : "PASS",
                g_fail);
  int f = g_fail, all = 0;
  MPI_Allreduce(&f, &all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return all ? 1 : 0;
}
