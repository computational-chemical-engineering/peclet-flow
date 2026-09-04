// The MIXED velocity multigrid (immersed solid + domain BCs) distributed: the momentum solve of a
// sphere in a channel with an inflow, an outflow and four no-slip walls, at a stiff time step
// (nu*dt/dx^2 = 1e3, the packed-bed-with-inlet regime where the production RB-GS hits its sweep
// cap). Three solves of the same problem with the REAL IbmSolver:
//   (a) distributed, velocity MG (VelocityMG::setStaircaseBc on the solver's ORB, in place);
//   (b) single-rank, velocity MG  -- (a) must equal (b): np=1 bit-tight, np>1 to the MG-PCG
//       pressure solve's reduction-order floor (the momentum V-cycle itself has no reductions
//       except the update-norm stop, which is a max and therefore order-independent);
//   (c) single-rank, RB-GS to a tight update tolerance -- the accuracy gate: the V-cycle must
//       converge to the SAME fixed point (the fine operator is the sharp cut-cell stencil in both).
// Run twice: with four no-slip walls, and with the +-z walls replaced by FREE-SLIP (symmetry,
// set_domain_bc type 4) faces -- the same three-way gate exercises the type-4 velocity ghosts,
// the mixed V-cycle's slip fold and the pressure MG's Neumann rows on a slip face, distributed.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using IbmSolver = peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 4;
static constexpr double RHO = 1.0, MU = 1.0, UIN = 1e-3;
// beta = MU*DT/dx^2; default 1e3 (the packed-bed-with-inlet regime), TEST_DT overrides for sweeps
static const double DT = std::getenv("TEST_DT") ? std::atof(std::getenv("TEST_DT")) : 1e3;

static std::vector<double> sphereSdf() {
  std::vector<double> sdf((std::size_t)N * N * N);
  const double R = 0.22 * N, cx = 0.5 * N, cy = 0.5 * N, cz = 0.5 * N;
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - cz;
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return sdf;
}

static void configure(IbmSolver& s, bool vmg, bool slipZ) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);
  s.setDomainBc(0, 2, UIN, 0, 0);  // -x inflow
  s.setDomainBc(1, 3, 0, 0, 0);    // +x outflow
  for (int f = 2; f < 6; ++f)
    s.setDomainBc(f, (slipZ && f >= 4) ? 4 : 1, 0, 0, 0);  // walls (+-z free-slip in pass 2)
  s.setPressureLevels(4);
  s.setPressurePcg(true, 300, 1e-10);
  // Both solvers to the same TIGHT residual (the solver default, 1e-5, would pin the solution
  // only to ~5e-4 on this operator -- rows span 1e3 -- and the fixed-point gate needs better).
  s.setVelocityResidualTolerance(1e-10);
  if (vmg) {
    s.setVelocityMultigrid(true, 4, 60);    // up to 60 V-cycles
  } else {
    s.setVelocityIterations(20000);         // RB-GS to the same tolerance
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int pass = 0; pass < 2; ++pass) {
    const bool slipZ = (pass == 1);
    if (rank == 0)
      std::printf("[velocitymg_bc] pass %d: +-z %s\n", pass, slipZ ? "FREE-SLIP (type 4)" : "no-slip walls");
    const std::vector<double> gsdf = sphereSdf();

    // (a) distributed, velocity MG
    peclet::core::decomp::BlockDecomposer<3> dec =
        peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
              gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(N, N, N, MPI_COMM_WORLD);
    configure(sd, true, slipZ);
    sd.setSolid(lsdf, true);
    long cycles = 0;
    for (int it = 0; it < STEPS; ++it) {
      sd.step();
      cycles += sd.lastMomentumSweeps();
    }
    const std::vector<double> ud = sd.getVelocity(0);
    const double divd = sd.maxOpenDivergence();

    // (b) + (c) single-rank on rank 0, broadcast
    std::vector<double> ub((std::size_t)N * N * N), uc((std::size_t)N * N * N);
    long cyclesRef = 0, sweepsRef = 0;
    if (rank == 0) {
      IbmSolver rb(N, N, N);
      configure(rb, true, slipZ);
      rb.setSolid(gsdf, true);
      for (int it = 0; it < STEPS; ++it) {
        rb.step();
        cyclesRef += rb.lastMomentumSweeps();
      }
      ub = rb.getVelocity(0);
      IbmSolver rc(N, N, N);
      configure(rc, false, slipZ);
      rc.setSolid(gsdf, true);
      for (int it = 0; it < STEPS; ++it) {
        rc.step();
        sweepsRef += rc.lastMomentumSweeps();
      }
      uc = rc.getVelocity(0);
    }
    MPI_Bcast(ub.data(), (int)ub.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(uc.data(), (int)uc.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cyclesRef, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sweepsRef, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    double dab = 0, dbc = 0, umax = 0;
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x) {
          const std::size_t li = (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
          const std::size_t gi =
              (std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N;
          dab = std::max(dab, std::fabs(ud[li] - ub[gi]));
          dbc = std::max(dbc, std::fabs(ub[gi] - uc[gi]));
          umax = std::max(umax, std::fabs(uc[gi]));
        }
    double g[3] = {dab, dbc, umax}, gg[3];
    MPI_Allreduce(g, gg, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double relAB = gg[0] / gg[2], relBC = gg[1] / gg[2];
    const double tolAB = (size == 1) ? 1e-12 : 1e-7;  // np>1: the pressure PCG's reduction floor
    const double tolBC = 1e-5;                         // V-cycle vs converged RB-GS fixed point
    if (rank == 0)
      std::printf("  dist-vs-single (vmg) rel %.2e (tol %.0e)  vmg-vs-RBGS rel %.2e (tol %.0e)  "
                  "div %.2e  cycles/step %.1f (single %.1f) vs RB-GS sweeps/step %.1f  (np=%d)\n",
                  relAB, tolAB, relBC, tolBC, divd, cycles / (double)STEPS,
                  cyclesRef / (double)STEPS, sweepsRef / (double)STEPS, size);
    if (!(relAB <= tolAB) || !(relBC <= tolBC) || !(divd < 1e-3))  // div: sanity (open-boundary level)
      fail = 1;
    if (slipZ) {
      // the slip faces are impermeable: the wall-normal component on the +-z boundary planes
      // must be exactly 0 on every rank that owns them (w(k = 0) is the -z face; the +z face is
      // the ghost plane the gather does not return, so the inner check covers the -z side)
      const std::vector<double> wd = sd.getVelocity(2);
      double wmax = 0.0;
      if (oz == 0)
        for (int y = 0; y < lny; ++y)
          for (int x = 0; x < lnx; ++x)
            wmax = std::max(wmax, std::fabs(wd[(std::size_t)x + (std::size_t)y * lnx]));
      double wg = 0.0;
      MPI_Allreduce(&wmax, &wg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("  slip face -z: max|w| on the boundary plane %.2e (impermeable)\n", wg);
      if (!(wg == 0.0))
        fail = 1;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): mixed (solid + domain-BC) velocity MG distributed == single-rank == RB-GS "
                  "(walls, and +-z free-slip)\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
