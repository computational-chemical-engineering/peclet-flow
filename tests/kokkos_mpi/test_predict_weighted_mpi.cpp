// flow — the pure pressure-hierarchy forecast (CutcellMG::predict, Python flow.predict_hierarchy)
// against the hierarchy the distributed Solver actually builds, before and after
// rebalanceByWeights.
//
// For each grid and weight field, on np ranks:
//
//  A  the Solver on its default (aligned ORB) partition: its built pressure-MG table equals
//     predict(G, np, levels, telescope ON) row for row — global dims, ranks holding the level, the
//     level's block 0, the ratio to the next level, and the kind of stage out of it;
//  B  after rebalanceByWeights(w): the built table equals predict(..., &w, &align) row for row,
//     and `align` equals the alignment 2^a the rebalance returned (core's chooseAlignedWeighted at
//     its 1.05 budget, a pure function of (np, G, w)).
//
// The comparison is exact (integers and flags). Coverage is gated, so the test cannot pass on
// trivial hierarchies: at np > 1 at least one weighted case must telescope and at least one must
// be a Repartition stage (the kind only a weighted dec0 arms), and some case must align above 1.
// Build with -DPECLET_FLOW_MPI.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"

using peclet::flow::CutcellMG;
using peclet::flow::IbmSolver;
using Row = CutcellMG::PlanRow;

static constexpr int LEVELS = 8, MIN_EXTENT = 4;

static bool sameRow(const Row& a, const Row& b) {
  auto eq = [](peclet::flow::C3 p, peclet::flow::C3 q) {
    return p.x == q.x && p.y == q.y && p.z == q.z;
  };
  return eq(a.global, b.global) && eq(a.block, b.block) && eq(a.ratio, b.ratio) &&
         a.ranks == b.ranks && a.tele == b.tele && a.repartition == b.repartition;
}

static void printRows(const char* tag, const std::vector<Row>& rows) {
  for (std::size_t L = 0; L < rows.size(); ++L) {
    const Row& r = rows[L];
    std::printf(
        "    %s L%zu global %3dx%3dx%3d  ranks %2d  block0 %3dx%3dx%3d  ratio(%d,%d,%d)%s\n", tag,
        L, r.global.x, r.global.y, r.global.z, r.ranks, r.block.x, r.block.y, r.block.z, r.ratio.x,
        r.ratio.y, r.ratio.z,
        r.tele ? (r.repartition ? "  -> REPARTITION" : "  -> TELESCOPE") : "");
  }
}

// Rank 0 compares (it holds every level); the verdict is broadcast so every rank exits alike.
static bool compare(const char* what, const std::vector<Row>& built,
                    const std::vector<Row>& predicted, int rank) {
  int ok = 1;
  if (rank == 0) {
    ok = built.size() == predicted.size();
    for (std::size_t L = 0; ok && L < built.size(); ++L)
      ok = sameRow(built[L], predicted[L]);
    std::printf("  %s: %zu levels built, %zu predicted -> %s\n", what, built.size(),
                predicted.size(), ok ? "match" : "MISMATCH");
    if (!ok) {
      printRows("built    ", built);
      printRows("predicted", predicted);
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok != 0;
}

struct Grid3 {
  int x, y, z;
};

// The weight fields: a heavy lower half in z (the balanced-force / redistribute tests' kind), a
// smooth off-centre bump (odd split planes), uniform weights (the alignment can go deepest), and a
// step in x balanced exactly on the ODD plane x = 13 (weights G.x-13 below it, 13 above). On 32^3
// either even neighbour costs a 1.077 imbalance: a = 0, an odd level-0 split, so even at np = 2
// the stage out of level 0 is a Repartition (measured; on 48x32x40 an even plane fits the budget).
static std::vector<peclet::core::Real> weights(int kind, Grid3 g) {
  std::vector<peclet::core::Real> w((std::size_t)g.x * g.y * g.z, 1.0);
  for (int z = 0; z < g.z; ++z)
    for (int y = 0; y < g.y; ++y)
      for (int x = 0; x < g.x; ++x) {
        double v = 1.0;
        if (kind == 0)
          v = z < g.z / 2 ? 4.0 : 1.0;
        else if (kind == 1) {
          const double dx = (x + 0.5) / g.x - 0.3, dy = (y + 0.5) / g.y - 0.6,
                       dz = (z + 0.5) / g.z - 0.35;
          v = 1.0 + 6.0 * std::exp(-(dx * dx + dy * dy + dz * dz) / 0.02);
        } else if (kind == 3)
          v = x < 13 ? g.x - 13 : 13;
        w[(std::size_t)x + (std::size_t)y * g.x + (std::size_t)z * g.x * g.y] = v;
      }
  return w;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int teleCases = 0, repartCases = 0, alignedCases = 0;
    const Grid3 grids[] = {{32, 32, 32}, {48, 32, 40}};
    const char* names[] = {"half-z x4", "bump", "uniform", "odd step x=13"};
    for (const Grid3 g : grids) {
      for (int kind = 0; kind < 4; ++kind) {
        if (rank == 0)
          std::printf("grid %dx%dx%d, weights '%s', np=%d\n", g.x, g.y, g.z, names[kind], size);
        const auto dec =
            CutcellMG::decomposition((std::size_t)size, g.x, g.y, g.z);  // what initMpi builds
        const auto blk = dec.block((std::size_t)rank);
        const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
        IbmSolver s(lnx, lny, lnz);
        s.initMpi(g.x, g.y, g.z, MPI_COMM_WORLD);
        s.setRho(1.0);
        s.setMu(0.1);
        s.setDt(0.02);
        s.setPressureLevels(LEVELS);
        s.setPressurePcg(true, 200, 1e-6);
        s.setSolid(std::vector<double>((std::size_t)lnx * lny * lnz, 1e30), true);

        // A: the default partition.
        const auto predA = CutcellMG::predict(g.x, g.y, g.z, size, LEVELS, true, MIN_EXTENT);
        if (!compare("A default partition", s.pressureMgPlan(), predA, rank))
          fail = 1;

        // B: after the weighted rebalance.
        const auto w = weights(kind, g);
        int alignPred = 0;
        const auto predB = CutcellMG::predict(g.x, g.y, g.z, size, LEVELS, true, MIN_EXTENT, 0,
                                              1.05, &w, &alignPred);
        const int alignGot = s.rebalanceByWeights(w);
        const auto built = s.pressureMgPlan();
        if (!compare("B after rebalanceByWeights", built, predB, rank))
          fail = 1;
        if (rank == 0)
          std::printf("  alignment: rebalanceByWeights %d, predict %d -> %s\n", alignGot, alignPred,
                      alignGot == alignPred ? "match" : "MISMATCH");
        if (alignGot != alignPred)
          fail = 1;
        if (rank == 0) {
          bool tele = false, rep = false;
          for (const Row& r : built) {
            tele = tele || r.tele;
            rep = rep || r.repartition;
          }
          teleCases += tele;
          repartCases += rep;
          alignedCases += alignGot > 1;
          printRows("", built);
        }
      }
    }
    if (rank == 0) {
      std::printf("coverage: %d weighted case(s) telescope, %d repartition, %d align > 1\n",
                  teleCases, repartCases, alignedCases);
      if (alignedCases == 0 || (size > 1 && (teleCases == 0 || repartCases == 0))) {
        std::printf("FAIL: the cases no longer exercise what this test gates\n");
        fail = 1;
      }
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("%s\n", fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
