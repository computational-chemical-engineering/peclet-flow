// flow — multi-rank validation of the balanced-force projection (doc/collocated_varrho_forces.md
// §4.6, §8 WO-P3, gate G5-ON), on BOTH grids.
//
// The option adds, once per step, a second solve of the projection's own operator whose right-hand
// side is the constraint divergence of the face force c*beta, and a split P += X - P_b, P_b = X.
// Under a decomposition that touches: the face force's high-side plane (built on this rank from
// depth-1 ghosts of the cell force, the colour/curvature and rho), the global max|b| == 0 test (an
// allreduce), the warm start from the registered field "p_balanced", and redistribute, which must
// carry "p_balanced" with the rest of the registry.
//
// Configurations on 16x16x32 (the aligned ORB cuts the long z axis at np = 2 and 4, the axis that
// carries the walls, the stratification and the interface):
//   * `colo-hydro` — SolverColocated, frozen colour, ratio 1000, walls at +-z (the option is the
//     V8 DEFAULT here: not set explicitly);
//   * `stag-hydro` — the same column on the staggered Solver, option ON;
//   * `stag-vof`   — staggered, enable_vof, ratio 10, a uniform x drive, option ON;
//   * `colo-vof-rebalance` — SolverColocated, enable_vof, ratio 10 (V8 default ON), Stokes; after 3
//   steps
//     `rebalanceByWeights` moves the partition (a weight 4 in the lower half), then 5 more steps.
//     The single-rank reference never rebalances.
//
// Protocol (the tests/kokkos_mpi pattern): gather the distributed fields to rank 0 and compare with
// a full-grid single-rank reference. np = 1 is bitwise; np > 1 lands on the allreduce reduction-
// order floor: u 1e-11, P and P_b 1e-9, C 1e-11 relative (the scale of a rest state is g*dt, see
// test_vof_collocated_mpi). Iterations, BOTH counts: exact at np = 1; above it the cost envelope of
// test_vof_collocated_mpi (worst step within one iteration of the reference's worst, total within
// one per step) -- a warm-started solve of a rest state works on round-off, whose per-step count is
// a random walk across decompositions (measured: +-2 per step with identical totals).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using Colo = peclet::flow::Solver<peclet::flow::Colocated>;
using Stag = peclet::flow::Solver<peclet::flow::Staggered>;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr double GRAV = 0.1, DT = 1.0, RHO0 = 1.0;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  double ratio;
  bool walls;       // hydrostatic (frozen colour) vs the coupled VoF run
  int steps;        // total steps
  int rebalance;    // step index after which the partition moves (-1: never)
  bool explicitOn;  // call setBalancedForceProjection(true) (false: rely on the solver default)
  bool advect;      // momentum advection
};

static double colourAt(int, int, int z) {
  return (z < NZ / 2) ? 1.0 : 0.0;
}

template <class Fn>
static std::vector<double> blockOf(Fn f, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  std::vector<double> v((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        v[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            f(x + ox, y + oy, z + oz);
  return v;
}

template <class S>
static void configure(S& s, const Config& c, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  s.setRho(RHO0);
  s.setMu(c.walls ? 0.1 : 0.05);
  s.setDt(DT);
  s.setAdvection(c.advect);
  if (c.walls) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);
  } else {
    // a z component, so the face force c*beta = rho0 f_z/rho_f varies across the interface and
    // the balanced-force solve has a non-zero right-hand side (an x drive alone is divergence-free)
    s.setBodyForce(1e-3, 0, 5e-4);
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  const auto c0 = blockOf(colourAt, ox, oy, oz, lnx, lny, lnz);
  if (c.walls) {
    s.addField("C");
    s.setField("C", c0);
    s.exchangeField("C");
  } else {
    s.enableVof();
    s.setVof(c0);
  }
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, c.ratio - 1.0});
  if (c.walls)
    s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -GRAV});
  if (c.explicitOn)
    s.setBalancedForceProjection(true);
}

static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign(GCELLS, 0.0);
  for (int r = 0; r < size; ++r) {
    int meta[6] = {ox, oy, oz, lnx, lny, lnz};
    if (r == 0) {
      if (rank == 0)
        for (int z = 0; z < lnz; ++z)
          for (int y = 0; y < lny; ++y)
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * NX +
                                (std::size_t)(z + oz) * NX * NY],
                        &local[(std::size_t)y * lnx + (std::size_t)z * lnx * lny],
                        (std::size_t)lnx * sizeof(double));
      continue;
    }
    if (rank == r) {
      MPI_Send(meta, 6, MPI_INT, 0, 100 + r, MPI_COMM_WORLD);
      MPI_Send(local.data(), (int)local.size(), MPI_DOUBLE, 0, 200 + r, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(meta, 6, MPI_INT, r, 100 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> buf((std::size_t)meta[3] * meta[4] * meta[5]);
      MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 200 + r, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int z = 0; z < meta[5]; ++z)
        for (int y = 0; y < meta[4]; ++y)
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * NX +
                              (std::size_t)(z + meta[2]) * NX * NY],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;  // NaN-propagating (WO-R2)
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;
    m = std::fmax(m, d);
  }
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

template <class S>
static int runCase(const Config& c, const peclet::core::decomp::BlockDecomposer<3>& dec, int rank,
                   int size) {
  auto blk = dec.block(rank);
  const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
  const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
  S sd(lnx, lny, lnz);
  sd.initMpi(dec, MPI_COMM_WORLD);
  configure(sd, c, ox, oy, oz, lnx, lny, lnz);
  std::vector<long> pd, bd;
  bool moved = false;
  for (int it = 0; it < c.steps; ++it) {
    sd.step();
    pd.push_back(sd.lastPressureIterations());
    bd.push_back(sd.lastBalancedForceIterations());
    if (it == c.rebalance) {
      std::vector<peclet::core::Real> w(GCELLS, 1.0);
      for (int z = 0; z < NZ / 2; ++z)
        for (int y = 0; y < NY; ++y)
          for (int x = 0; x < NX; ++x)
            w[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = 4.0;
      const auto o0 = sd.blockOrigin();
      const long n0 = sd.nx() * sd.ny() * sd.nz();
      sd.rebalanceByWeights(w);
      const auto o1 = sd.blockOrigin();
      int mv = (o0 != o1 || n0 != sd.nx() * sd.ny() * sd.nz()) ? 1 : 0, gmv = 0;
      MPI_Allreduce(&mv, &gmv, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      moved = gmv != 0;
    }
  }
  const auto o = sd.blockOrigin();
  const int px = o[0], py = o[1], pz = o[2];
  const int qx = (int)sd.nx(), qy = (int)sd.ny(), qz = (int)sd.nz();
  std::vector<double> gu[3];
  for (int comp = 0; comp < 3; ++comp)
    gu[comp] = gatherGlobal(sd.getVelocity(comp), px, py, pz, qx, qy, qz, rank, size);
  const auto gp = gatherGlobal(sd.getPressure(), px, py, pz, qx, qy, qz, rank, size);
  const auto gb = gatherGlobal(sd.getField("p_balanced"), px, py, pz, qx, qy, qz, rank, size);
  const auto gc = gatherGlobal(sd.getField("C"), px, py, pz, qx, qy, qz, rank, size);
  int fail = 0;
  if (rank == 0) {
    S ref(NX, NY, NZ);
    configure(ref, c, 0, 0, 0, NX, NY, NZ);
    std::vector<long> pr, br;
    for (int it = 0; it < c.steps; ++it) {
      ref.step();
      pr.push_back(ref.lastPressureIterations());
      br.push_back(ref.lastBalancedForceIterations());
    }
    double du = 0, su = 0;
    for (int comp = 0; comp < 3; ++comp) {
      du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
      su = std::fmax(su, maxAbs(ref.getVelocity(comp)));
    }
    const double dp = maxAbsDiff(gp, ref.getPressure()), sp = maxAbs(ref.getPressure());
    const double db = maxAbsDiff(gb, ref.getField("p_balanced"));
    const double dc = maxAbsDiff(gc, ref.getField("C"));
    long dbit = 0, bsum = 0, maxd = 0, maxr = 0, sumd = 0, sumr = 0;
    for (std::size_t k = 0; k < pd.size(); ++k) {
      dbit = std::max(dbit, std::labs(bd[k] - br[k]));
      bsum += br[k];
      maxd = std::max(maxd, pd[k]);
      maxr = std::max(maxr, pr[k]);
      sumd += pd[k];
      sumr += pr[k];
    }
    // The balanced-force count gets the SAME cost-envelope rule as the main count. Before WO-P5
    // (warm start, initial-residual stop) the rest-state columns parted by up to 2 per step with
    // identical totals -- the solve worked on round-off. With the increment solve and its
    // full-RHS stop the rest state skips after step 1 (14 iterations in 20 steps, identical at
    // np 1/2/4), so the envelope is now slack rather than load-bearing. np = 1 stays exact.
    long bmaxd = 0, bmaxr = 0, bsumd = 0;
    for (std::size_t k = 0; k < bd.size(); ++k) {
      bmaxd = std::max(bmaxd, bd[k]);
      bmaxr = std::max(bmaxr, br[k]);
      bsumd += bd[k];
    }
    const bool bfOk = (size == 1) ? (dbit == 0)
                                  : (bmaxd <= bmaxr + 1 &&
                                     std::labs(bsumd - bsum) <= static_cast<long>(bd.size()));
    const long dpit = std::labs(sumd - sumr);
    const bool pOk =
        (size == 1) ? (pd == pr) : (maxd <= maxr + 1 && dpit <= static_cast<long>(pd.size()));
    const double uRef = std::fmax(su, c.walls ? GRAV * DT : 1e-3 * DT);
    const double tolU = (size == 1) ? 0.0 : 1e-11 * uRef;
    const double tolP = (size == 1) ? 0.0 : 1e-9 * std::fmax(sp, 1e-12);
    const double tolC = (size == 1) ? 0.0 : 1e-11;
    const bool moveOk = c.rebalance < 0 || size == 1 || moved;
    const bool ok = du <= tolU && dp <= tolP && db <= tolP && dc <= tolC && bfOk && pOk && moveOk;
    std::printf(
        "  [%-18s np=%d] du %.3e (|u| %.3e)  dP %.3e  dP_b %.3e (|P| %.3e)  dC %.3e  "
        "bf-iters sum %ld d/step %ld  p-iters max %ld/%ld sum %ld/%ld%s   %s\n",
        c.name, size, du, su, dp, db, sp, dc, bsum, dbit, maxd, maxr, sumd, sumr,
        c.rebalance >= 0 ? (moved ? "  (partition MOVED)" : "  (partition did NOT move)") : "",
        ok ? "OK" : "*** FAIL ***");
    if (!ok)
      fail = 1;
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return fail;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    if (rank == 0)
      std::printf("BALANCED-FORCE PROJECTION MPI np=%d  grid %dx%dx%d\n", size, NX, NY, NZ);
    fail |= runCase<Colo>({"colo-hydro", 1000.0, true, 20, -1, false, false}, dec, rank, size);
    fail |= runCase<Stag>({"stag-hydro", 1000.0, true, 20, -1, true, false}, dec, rank, size);
    fail |= runCase<Stag>({"stag-vof", 10.0, false, 10, -1, true, true}, dec, rank, size);
    // Stokes: the collocated advecting face field uf_ is not carried by redistribute (a
    // pre-existing gap, off this option: with the option OFF an np = 1 "move" already differs by
    // 5e-8 in u when advection is on), so the move is gated on the option's own state.
    fail |= runCase<Colo>({"colo-vof-rebalance", 10.0, false, 8, 2, false, false}, dec, rank, size);
    if (rank == 0)
      std::printf("%s\n", fail ? "FAILED" : "all balanced-force MPI gates passed");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
