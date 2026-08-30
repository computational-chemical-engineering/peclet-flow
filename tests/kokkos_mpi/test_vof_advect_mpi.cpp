// VoF rung V1 (WO-E) — distributed Weymouth-Yue split advection: np 2 and 4 must reproduce the
// single-rank colour field BIT FOR BIT, periodic and non-periodic.
//
// The reference is not a stored baseline: every rank additionally runs the SAME problem on the
// whole global grid as one block with a serial periodic/Neumann ghost fill (i.e. the np = 1 path),
// and compares its own sub-block bitwise. Under np = 1 that makes the test a self-consistency check
// of the two ghost-fill routes; under np = 2 / 4 it is the real decomposition-independence gate.
//
// Why bitwise identity is reachable at all: the WY update of a cell reads only its own C, the two
// face velocities, the frozen dilation flag, and the two face fluxes -- and each face flux is
// computed ONCE per face from the donor's PLIC, so the two cells sharing a face (in-rank, or across
// a rank boundary) consume the identical double. No reduction enters the update; the CFL max is the
// only all-reduce and it only gates the step. So there is no legitimate reason for a last-bit
// difference, and one appearing in ghost-adjacent cells means the halo width or the
// exchange-per-sweep choreography is wrong (WO-E escalation list).
//
// The colour field carries its OWN GridHaloTopology at ghost width 3 (VOF_PLAN.md §3 rule 1); the
// solver's G = 2 machinery is untouched -- nothing here includes flow_ibm.hpp.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "vof/advect_wy.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::WyAdvector;
using vofscene::Block;

constexpr int kDim = 3;
constexpr int G = 3;  // the colour field's own ghost width

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

enum class Scene { UniformSphere, LeVequeWalls };

/// Fill the colour + face velocities of one block for the chosen scene at step `s`.
void seedScene(Scene sc, WyAdvector& a, Block b, double h, long s, double dt, double T) {
  if (sc == Scene::UniformSphere) {
    if (s < 0)
      vofscene::initSphere(a.colour(), b, h, 0.5, 0.375, 0.3125, 0.2);
    else
      vofscene::fillUniform(a, b, 1.0, 0.7, -0.4);
  } else {
    if (s < 0)
      vofscene::initSphere(a.colour(), b, h, 0.35, 0.35, 0.35, 0.15);
    else
      vofscene::fillLeVeque(a, b, h, std::cos(M_PI * (s + 0.5) * dt / T));
  }
}

struct SceneSpec {
  Scene scene;
  IVec<kDim> gs;
  double h, dt, T;
  long steps;
  std::array<bool, kDim> periodic;
  const char* name;
};

/// Run one scene and report (a) the number of inner cells whose bits differ from the whole-grid
/// single-block run and (b) the global volume drift of the distributed run.
void runScene(const SceneSpec& sp, int rank, int size) {
  const IVec<kDim> gs = sp.gs;

  // ---------------- distributed block ----------------
  BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gs);
  GridHaloTopology<kDim> topo;
  topo.buildTopology(dec, rank, G, sp.periodic, MPI_COMM_WORLD);
  GridHalo<double> halo;
  halo.init(topo);
  const auto& idx = topo.indexer();
  const auto inner = idx.sizeInner(), ogh = idx.originInclGhost();
  const I3 origin{static_cast<int>(ogh[0]) + G, static_cast<int>(ogh[1]) + G,
                  static_cast<int>(ogh[2]) + G};

  const I3 gsi{static_cast<int>(gs[0]), static_cast<int>(gs[1]), static_cast<int>(gs[2])};

  WyAdvector adv;
  adv.init(static_cast<int>(inner[0]), static_cast<int>(inner[1]), static_cast<int>(inner[2]), sp.h,
           G);
  const Block blk = vofscene::blockOf(adv, origin);
  const std::array<bool, 3> per{sp.periodic[0], sp.periodic[1], sp.periodic[2]};
  adv.exchange = [&halo, blk, gsi, per](SField f) {
    halo.exchange(f);  // cross-rank + periodic wrap (nothing written outside a non-periodic domain)
    vofscene::clampFill(f, blk, gsi, per[0], per[1], per[2]);
  };
  adv.globalMax = [](double v) {
    double r = v;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return r;
  };

  seedScene(sp.scene, adv, blk, sp.h, -1, sp.dt, sp.T);
  adv.syncGhosts();
  double v0Loc = adv.diagnostics().sumC, v0 = 0.0;
  MPI_Allreduce(&v0Loc, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (long s = 0; s < sp.steps; ++s) {
    seedScene(sp.scene, adv, blk, sp.h, s, sp.dt, sp.T);
    adv.advect(sp.dt, s);
  }
  const auto dLoc = adv.diagnostics();
  double v1Loc = dLoc.sumC, v1 = 0.0;
  MPI_Allreduce(&v1Loc, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // ---------------- whole-grid reference on every rank (the np = 1 path) ----------------
  WyAdvector ref;
  ref.init(static_cast<int>(gs[0]), static_cast<int>(gs[1]), static_cast<int>(gs[2]), sp.h, G);
  const Block rblk = vofscene::blockOf(ref, I3{0, 0, 0});
  const I3 eRef = ref.extent();
  ref.exchange = [eRef, rblk, gsi, per](SField f) {
    vofscene::periodicFill(f, eRef, G, per[0], per[1], per[2]);
    vofscene::clampFill(f, rblk, gsi, per[0], per[1], per[2]);
  };
  seedScene(sp.scene, ref, rblk, sp.h, -1, sp.dt, sp.T);
  ref.syncGhosts();
  for (long s = 0; s < sp.steps; ++s) {
    seedScene(sp.scene, ref, rblk, sp.h, s, sp.dt, sp.T);
    ref.advect(sp.dt, s);
  }

  // ---------------- bitwise comparison of this rank's inner region ----------------
  auto hLoc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), adv.colour());
  auto hRef = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref.colour());
  long diff = 0, ghostAdjacent = 0;
  double worst = 0.0;
  for (int z = 0; z < static_cast<int>(inner[2]); ++z)
    for (int y = 0; y < static_cast<int>(inner[1]); ++y)
      for (int x = 0; x < static_cast<int>(inner[0]); ++x) {
        const double a = hLoc(L3(x + G, y + G, z + G, adv.extent()));
        const double b = hRef(L3(x + origin.x + G, y + origin.y + G, z + origin.z + G, eRef));
        if (std::memcmp(&a, &b, sizeof(double)) != 0) {
          ++diff;
          worst = std::fmax(worst, std::fabs(a - b));
          const bool edge = x == 0 || y == 0 || z == 0 || x == static_cast<int>(inner[0]) - 1 ||
                            y == static_cast<int>(inner[1]) - 1 ||
                            z == static_cast<int>(inner[2]) - 1;
          if (edge)
            ++ghostAdjacent;
        }
      }
  long diffAll = 0, edgeAll = 0;
  MPI_Allreduce(&diff, &diffAll, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ghostAdjacent, &edgeAll, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  double worstAll = 0.0;
  MPI_Allreduce(&worst, &worstAll, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0) {
    std::printf("  [%s] %dx%dx%d, %ld steps, np=%d, block %dx%dx%d:\n", sp.name,
                static_cast<int>(gs[0]), static_cast<int>(gs[1]), static_cast<int>(gs[2]), sp.steps,
                size, static_cast<int>(inner[0]), static_cast<int>(inner[1]),
                static_cast<int>(inner[2]));
    std::printf(
        "      bitwise diffs vs whole-grid run: %ld (of which ghost-adjacent %ld)"
        "   max|diff| %.3e\n",
        diffAll, edgeAll, worstAll);
    std::printf("      global volume drift %.3e   C[%.3e, %.6f]   wisps(rank0) %ld\n",
                std::fabs(v1 - v0) / v0, dLoc.minC, dLoc.maxC, dLoc.wisps);
  }
  CHECK(diffAll == 0);
  CHECK(std::fabs(v1 - v0) / v0 < 1e-13);
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
      std::printf("VoF rung V1 (WO-E) - distributed Weymouth-Yue, backend: %s, np = %d\n",
                  peclet::flow::SExec::name(), size);

    // periodic: sphere in a uniform (non-diagonal-symmetric) translation
    runScene(SceneSpec{Scene::UniformSphere,
                       IVec<kDim>{32, 24, 20},
                       1.0 / 32,
                       0.3 * (1.0 / 32),
                       1.0,
                       24,
                       {true, true, true},
                       "periodic / uniform translation"},
             rank, size);

    // non-periodic: the LeVeque deformation field on the closed unit box (u.n = 0 on all six
    // faces), Neumann colour ghosts outside the domain
    runScene(SceneSpec{Scene::LeVequeWalls,
                       IVec<kDim>{24, 24, 24},
                       1.0 / 24,
                       3.0 / 400,
                       3.0,
                       24,
                       {false, false, false},
                       "walls / LeVeque deformation"},
             rank, size);

    // mixed periodicity, to exercise a block that is periodic on one axis and walled on another
    runScene(SceneSpec{Scene::LeVequeWalls,
                       IVec<kDim>{24, 24, 24},
                       1.0 / 24,
                       3.0 / 400,
                       3.0,
                       24,
                       {true, false, true},
                       "mixed periodic/wall / LeVeque"},
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
