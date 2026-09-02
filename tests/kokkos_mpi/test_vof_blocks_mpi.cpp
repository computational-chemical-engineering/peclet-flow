// VoF Part III rung W0 (WO-W0), gate G4 — the DISTRIBUTED per-bubble block container.
//
// The reference is not a stored baseline: every rank additionally runs the SAME problem on the
// whole global grid as one `WyAdvector` (the np = 1 path), and compares its own sub-block of the
// UNION field bitwise. Under np = 1 that is a self-consistency check of the two gather routes;
// under np = 2 / 4 it is the decomposition-independence gate proper.
//
// Two things this gate is built to break:
//  * the masters are assigned ROUND ROBIN by block id, deliberately independent of where the
//    bubble's cells live. With more blocks than ranks (and the seeds placed away from each other)
//    at least one block's master owns none of its own cells, so its whole state arrives by message.
//  * every scene's decomposition CUTS a bubble: the ORB of 2 / 4 ranks splits the grid across the
//    seeds, so a block's gather draws from several owners and its scatter lands on several.
//
// Why bitwise is reachable: the runs of `vofBuildPieces` partition the block's box, so every
// block-local cell is written by exactly one owner's piece; the union combines with `max`, which is
// exact and commutative; and the WY update of a cell then consumes the identical doubles the
// whole-grid run consumes. A last-bit difference here means the plan double-covers or misses a
// cell, or the block ghost policy is decomposition-dependent.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "vof/block_container.hpp"
#include "vof/block_exchange.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SExec;
using peclet::flow::SField;
using peclet::flow::vof::VofBlockExchange;
using peclet::flow::vof::VofBlockSet;
using peclet::flow::vof::VofBox;
using peclet::flow::vof::WyAdvector;

constexpr int kDim = 3;
constexpr int G = 3;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

struct Seed {
  double x, y, z, r;
};

enum class Flow { Uniform, LeVeque, Cellular };

KOKKOS_INLINE_FUNCTION double cellPsi(double x, double y, double A, double phase) {
  const double PI = 3.14159265358979323846;
  return -(A / (2.0 * PI)) * Kokkos::sin(2.0 * PI * x) * Kokkos::sin(2.0 * PI * y) * phase;
}
void fillCellular(WyAdvector& a, vofscene::Block b, double h, double A, double phase) {
  SField u = a.faceU(), v = a.faceV(), w = a.faceW();
  const double invh = 1.0 / h;
  vofscene::forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int) {
        const double xn = gx * h, yn = gy * h, xp = (gx + 1) * h, yp = (gy + 1) * h;
        u(i) = (cellPsi(xp, yp, A, phase) - cellPsi(xp, yn, A, phase)) * invh;
        v(i) = -(cellPsi(xp, yp, A, phase) - cellPsi(xn, yp, A, phase)) * invh;
        w(i) = 0.0;
      });
}

void fillFlow(Flow f, WyAdvector& a, vofscene::Block b, double h, double phase) {
  if (f == Flow::Uniform)
    vofscene::fillUniform(a, b, 1.0, 0.7, -0.4);
  else if (f == Flow::LeVeque)
    vofscene::fillLeVeque(a, b, h, phase);
  else
    fillCellular(a, b, h, 1.0, phase);
}

struct SceneSpec {
  Flow flow;
  int gn;
  double dt, T;
  long steps;
  std::vector<Seed> seeds;
  const char* name;
  /// Does this scene have enough blocks that the round-robin assignment is guaranteed to leave at
  /// least one block's master owning NONE of its own cells? (Only then is it a gate rather than an
  /// accident of where the seeds landed.)
  bool orphanMaster = false;
  /// Rung W1 (WO-W12): the master-assignment mode and the re-assignment interval. `reassign > 0`
  /// makes masters CHANGE mid-run and the block colour MIGRATE; the bitwise comparison against the
  /// single-rank container (which has one rank and therefore never migrates) is then a gate on the
  /// migration being exact as well as on the exchange.
  int assign = 0;
  long reassign = 0;
};

/// A lattice of `n^3` bubbles with radii spanning ~4x, so the block cell counts span ~10x and the
/// round robin is measurably worse than LPT.
std::vector<Seed> lattice(int n, double rmin, double rmax) {
  std::vector<Seed> v;
  int k = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int m = 0; m < n; ++m, ++k)
        v.push_back(Seed{(i + 0.5) / n, (j + 0.5) / n, (m + 0.5) / n,
                         rmin + (rmax - rmin) * ((k % 8) / 7.0)});
  return v;
}

void runScene(const SceneSpec& sp, int rank, int size) {
  const int gn = sp.gn;
  const double h = 1.0 / gn;
  const std::array<bool, 3> per{true, true, true};
  const I3 gs{gn, gn, gn};

  // --------------- the flow decomposition (the SAME ORB the solver would build) ---------------
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), IVec<kDim>{gn, gn, gn});
  std::vector<VofBox> rankBox(static_cast<std::size_t>(size));
  for (int r = 0; r < size; ++r) {
    const auto b = dec.block(static_cast<std::size_t>(r));
    for (int d = 0; d < 3; ++d) {
      rankBox[r].lo[d] = static_cast<int>(b.origin[d]);
      rankBox[r].hi[d] = static_cast<int>(b.origin[d] + b.size[d]);
    }
  }
  const VofBox& me = rankBox[rank];
  const I3 ln{me.n(0), me.n(1), me.n(2)}, lo{me.lo[0], me.lo[1], me.lo[2]};

  // --------------- this rank's patch: it owns the face velocity of its own cells --------------
  WyAdvector patch;
  patch.init(ln.x, ln.y, ln.z, h, G);
  const vofscene::Block pblk = vofscene::blockOf(patch, lo);

  VofBlockSet set;
  set.init(gs, per, rank, size, h);
  auto ex = std::make_shared<VofBlockExchange>();
  ex->init(gs, per, rankBox, rank);
  VofBlockExchange::Patch pp;
  pp.e = patch.extent();
  pp.n = patch.inner();
  pp.o = lo;
  pp.g = G;
  ex->setPatch(pp, patch.faceU(), patch.faceV(), patch.faceW());
  ex->setComm(MPI_COMM_WORLD);
  set.setExchange(ex);
  set.assignMode = static_cast<peclet::flow::vof::VofMasterAssign>(sp.assign);
  set.reassignEvery = sp.reassign;
  for (const auto& s : sp.seeds)
    set.seedSphere(s.x, s.y, s.z, s.r);
  set.assignMasters();
  set.scatter(patch.colour());

  // --------------- the whole-grid reference, on every rank --------------------------------
  WyAdvector ref;
  ref.init(gn, gn, gn, h, G);
  const vofscene::Block rblk = vofscene::blockOf(ref, I3{0, 0, 0});
  const I3 re = ref.extent();
  ref.exchange = [re](SField f) { vofscene::periodicFill(f, re, G, true, true, true); };
  VofBlockSet rset;  // the SAME container on one rank: this is what the union must equal
  rset.init(gs, per, 0, 1, h);
  WyAdvector rpatch;
  rpatch.init(gn, gn, gn, h, G);
  const vofscene::Block rpblk = vofscene::blockOf(rpatch, I3{0, 0, 0});
  {
    auto rex = std::make_shared<VofBlockExchange>();
    std::vector<VofBox> one(1);
    one[0].lo[0] = one[0].lo[1] = one[0].lo[2] = 0;
    one[0].hi[0] = one[0].hi[1] = one[0].hi[2] = gn;
    rex->init(gs, per, one, 0);
    VofBlockExchange::Patch rp;
    rp.e = rpatch.extent();
    rp.n = rpatch.inner();
    rp.o = I3{0, 0, 0};
    rp.g = G;
    rex->setPatch(rp, rpatch.faceU(), rpatch.faceV(), rpatch.faceW());
    rset.setExchange(rex);
  }
  rset.assignMode = static_cast<peclet::flow::vof::VofMasterAssign>(sp.assign);
  rset.reassignEvery = sp.reassign;
  for (const auto& s : sp.seeds)
    rset.seedSphere(s.x, s.y, s.z, s.r);
  rset.assignMasters();
  rset.scatter(rpatch.colour());

  auto compare = [&](const char* when) {
    auto hLoc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), patch.colour());
    auto hRef = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rpatch.colour());
    long diff = 0;
    double worst = 0.0;
    for (int z = 0; z < ln.z; ++z)
      for (int y = 0; y < ln.y; ++y)
        for (int x = 0; x < ln.x; ++x) {
          const double a = hLoc(L3(x + G, y + G, z + G, patch.extent()));
          const double b = hRef(L3(x + lo.x + G, y + lo.y + G, z + lo.z + G, re));
          if (std::memcmp(&a, &b, sizeof(double)) != 0) {
            ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
          }
        }
    long all = 0;
    double wall = 0.0;
    MPI_Allreduce(&diff, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&worst, &wall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (all != 0 && rank == 0)
      std::printf("      %s: %ld inner cells differ (max|d| %.3e)\n", when, all, wall);
    return all;
  };
  long bad = compare("seed");
  CHECK(bad == 0);

  long firstBad = -1;
  long nMigrated = 0;
  for (long s = 0; s < sp.steps; ++s) {
    const double phase = std::cos(M_PI * (s + 0.5) * sp.dt / sp.T);
    fillFlow(sp.flow, patch, pblk, h, phase);
    fillFlow(sp.flow, rpatch, rpblk, h, phase);
    set.advect(sp.dt, patch.colour());
    rset.advect(sp.dt, rpatch.colour());
    nMigrated += set.lastReassigned();
    if (compare("step") != 0) {
      firstBad = s;
      break;
    }
  }
  CHECK(firstBad < 0);

  // per-marker volume, reduced from the masters (each block has exactly one)
  const std::size_t nb = set.count();
  std::vector<double> volLoc(nb, 0.0), volAll(nb, 0.0), volRef(nb, 0.0);
  for (std::size_t i = 0; i < nb; ++i)
    if (set.blocks()[i].mine())
      volLoc[i] = set.blocks()[i].stats().volume;
  MPI_Allreduce(volLoc.data(), volAll.data(), static_cast<int>(nb), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  for (std::size_t i = 0; i < nb; ++i)
    volRef[i] = rset.blocks()[i].stats().volume;
  double volDiff = 0.0;
  for (std::size_t i = 0; i < nb; ++i)
    volDiff = std::fmax(volDiff, std::fabs(volAll[i] - volRef[i]));

  // ownership census: how many of its OWN cells does each block's master hold?
  long ownedByMaster = 0, blocksWithNoLocalCells = 0;
  for (const auto& b : set.blocks()) {
    const VofBox ov = VofBox::intersect(b.box, rankBox[b.master]);
    if (ov.empty())
      ++blocksWithNoLocalCells;
    else
      ownedByMaster += ov.cells();
  }
  std::vector<long> mc, cc;
  set.masterCensus(mc, cc);

  if (rank == 0) {
    std::printf("  [%s] %d^3, %ld steps, np=%d, %zu blocks, local block %dx%dx%d\n", sp.name, gn,
                sp.steps, size, nb, ln.x, ln.y, ln.z);
    std::printf("      bitwise: first differing step %ld (-1 = none)   |dV| vs np=1 %.3e\n",
                firstBad, volDiff);
    std::printf("      masters per rank:");
    for (int r = 0; r < size; ++r)
      std::printf(" %ld", mc[r]);
    std::printf("   block cells per rank:");
    for (int r = 0; r < size; ++r)
      std::printf(" %ld", cc[r]);
    std::printf("   cell imbalance %.3f\n", set.cellImbalance());
    std::printf("      blocks whose master owns NONE of their cells: %ld / %zu\n",
                blocksWithNoLocalCells, nb);
    if (sp.reassign > 0)
      std::printf(
          "      assignment mode %d, re-assigned every %ld steps: %ld master CHANGES migrated "
          "(%ld B in %ld msgs on rank 0 at the last one)\n",
          sp.assign, sp.reassign, nMigrated, ex->migrateBytes(), ex->migrateMessages());
    std::printf("      exchange: gather %ld B in %ld msgs, scatter %ld B in %ld msgs (rank 0)\n",
                ex->gatherBytes(), ex->gatherMessages(), ex->scatterBytes(), ex->scatterMessages());
  }
  CHECK(volDiff < 1e-15);
  if (size > 1 && sp.orphanMaster)
    CHECK(blocksWithNoLocalCells > 0);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
      std::printf("VoF rung W0 (WO-W0) - distributed block container, backend: %s, np = %d\n",
                  SExec::name(), size);

    // G4 / G1: one bubble in the LeVeque deformation, the decomposition cutting it
    runScene(SceneSpec{Flow::LeVeque,
                       32,
                       3.0 / 768,  // CFL 0.25, the V1 gate-D schedule at 32^3
                       3.0,
                       48,
                       {{0.35, 0.35, 0.35, 0.15}},
                       "G1 one bubble / LeVeque"},
             rank, size);

    // G4 / G2: two bubbles pushed together; masters round-robin, so at np >= 2 block 1's master
    // is a different rank from block 0's and neither need own the cells
    runScene(SceneSpec{Flow::Cellular,
                       48,
                       0.9 / 216,  // CFL 0.2
                       0.9,
                       120,
                       {{0.38, 0.5, 0.5, 0.10}, {0.62, 0.5, 0.5, 0.10}},
                       "G2 two bubbles / cellular"},
             rank, size);

    // G4 / G3: three bubbles in a uniform translation -- the blocks MOVE (re-centre) while
    // distributed, and with 3 blocks over 1/2/4 ranks the round-robin leaves an idle rank at np=4
    runScene(SceneSpec{Flow::Uniform,
                       32,
                       0.25 / 32,
                       1.0,
                       40,
                       {{0.30, 0.30, 0.30, 0.12}, {0.70, 0.65, 0.35, 0.10}, {0.5, 0.2, 0.8, 0.09}},
                       "G3 three bubbles / translation",
                       true},
             rank, size);

    // W1 (WO-W12): 64 bubbles of MIXED size in the LeVeque field, LPT masters re-assigned every
    // 8 steps. The single-rank reference container never migrates (size 1), so a bitwise match
    // here is simultaneously (a) decomposition independence, (b) assignment independence and
    // (c) exactness of the migration -- the colour and the previous centroid are the only state a
    // block carries, and they travel as a contiguous message.
    runScene(SceneSpec{Flow::LeVeque,
                       48,
                       0.08 / 48,  // the LeVeque field peaks near |uf| = 2, so CFL ~ 0.16
                       1.5,
                       24,
                       lattice(4, 2.0 / 48, 8.0 / 48),
                       "W1 64 bubbles / LeVeque / LPT + re-assignment",
                       true,
                       /*assign=*/1,
                       /*reassign=*/8},
             rank, size);

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    failures = total;
    if (rank == 0)
      std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                  failures == 1 ? "" : "s");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return failures ? 1 : 0;
}
