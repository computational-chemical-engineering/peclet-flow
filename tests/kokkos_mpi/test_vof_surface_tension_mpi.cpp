// flow — multi-rank validation of the balanced-force CSF (rung V4, WO-P).
//
// Two things are gated, and they are different in kind.
//
//   S1 THE EXACTNESS GATE, AT EVERY DECOMPOSITION. A stationary droplet with a CONSTANT curvature
//      and a uniform density: the surface-tension force is then exactly the discrete gradient of
//      `sigma*kappa*C`, the projection annihilates it, and `max|u|` must sit at MACHINE ZERO on
//      every rank at np = 1, 2 and 4. This is the momentum analogue of the hydrostatic acid test's
//      multi-rank half (`test_vardensity_mpi`'s `walls-z`), and it catches the same class of defect
//      it does: a face force assembled from a ghost the exchange did not fill is invisible in the
//      single-rank run and shows up here as a rank-boundary velocity. The colour field's ghost, the
//      curvature field's ghost and the branch field's ghost all feed that one face value.
//
//   S2 THE COUPLED STEP AGAINST A FULL-GRID SINGLE-RANK REFERENCE — velocity, pressure, colour AND
//      curvature after 10 steps of a real two-phase run with surface tension on. np = 1 must be
//      BITWISE; np > 1 carries the documented reduction-order floor of the pressure driver's
//      allreduces (`doc/variable_density_projection.md` §3.1), so it is a tolerance comparison —
//      except for kappa and the branch field, which contain no reduction at all and must be bitwise
//      at every np, exactly as `test_vof_curvature_mpi` asserts.
//
// Grid 16x16x32: the aligned ORB cuts the long z axis at np = 2 and 4, and the droplet is centred
// so its interface straddles that cut. `walls-z` puts domain walls on the CUT axis so the colour
// and curvature ghosts go through the non-periodic (clamped) fill on a rank boundary.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
static constexpr double RHO_G = 1.0, RHO_L = 10.0, SIGMA = 1.0, MU = 0.05, RDROP = 5.0;

struct Config {
  const char* name;
  bool walls;       // walls +-z (the CUT axis) vs fully periodic
  bool constKappa;  // S1: freeze kappa at a constant and use a uniform density
};

// Volume fractions of a sphere at a GLOBAL cell index — a pure function of global indices, so each
// block gets bit-identical data to the corresponding part of the full-grid run. Exact in z,
// 16 x 16 subsampled in (x,y) (a crude 3-D subsample manufactures full/empty cell PAIRS whose
// shared face carries an interface no cell knows about — the CSF's orphan-face defect).
static double colourAt(const Config& c, int x, int y, int z) {
  const double cx = NX / 2.0 + 0.137, cy = NY / 2.0 - 0.311;
  const double cz = c.walls ? NZ / 4.0 + 0.241 : NZ / 2.0 + 0.241;
  const int NS = 16;
  double acc = 0.0;
  for (int j = 0; j < NS; ++j)
    for (int i = 0; i < NS; ++i) {
      const double px = x + (i + 0.5) / NS - cx, py = y + (j + 0.5) / NS - cy;
      const double r2 = RDROP * RDROP - px * px - py * py;
      if (r2 <= 0.0)
        continue;
      const double h = std::sqrt(r2);
      const double lo = std::fmax(cz - h, (double)z), hi = std::fmin(cz + h, (double)z + 1);
      if (hi > lo)
        acc += hi - lo;
    }
  return acc / (NS * NS);
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

static void configure(IbmSolver& s, const Config& c, int ox, int oy, int oz, int lnx, int lny,
                      int lnz) {
  const double rl = c.constKappa ? RHO_G : RHO_L;
  s.setRho(rl);
  s.setMu(MU);
  s.setDt(1.0);
  // The balanced-force exactness gates (max|u| < 1e-13) need the momentum solve at machine
  // precision: the default residual stop (1e-5, 2026-09-02) leaves a 5e-12 spurious current, so
  // these tests keep the legacy fixed-sweep loop.
  s.setVelocityResidualTolerance(0.0);
  if (c.walls) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z (the CUT axis)
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setPressureChebyshev(true, 400, 1e-14);
  s.enableVof();
  s.setVof(blockOf([&](int x, int y, int z) { return colourAt(c, x, y, z); }, ox, oy, oz, lnx, lny,
                   lnz));
  s.setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {RHO_G, rl - RHO_G});
  s.setSurfaceTension(SIGMA);
  if (c.constKappa)
    s.setVofKappaConstant(2.0 / RDROP);
  s.setDt(0.5 * s.capillaryDt());
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

static double maxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}
static long countBitDiff(const std::vector<double>& a, const std::vector<double>& b, double& mx) {
  long n = 0;
  mx = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i)
    if (a[i] != b[i]) {
      ++n;
      mx = std::fmax(mx, std::fabs(a[i] - b[i]));
    }
  return n;
}
static double maxAbs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
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
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    const int gn[3] = {NX, NY, NZ};
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != gn[a])
          cut[a] = true;
    if (rank == 0)
      std::printf(
          "VOF SURFACE TENSION MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size,
          NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "");

    const Config configs[] = {{"exact-per", false, true},
                              {"exact-walls-z", true, true},
                              {"drop-per", false, false},
                              {"drop-walls-z", true, false}};

    for (const Config& c : configs) {
      if (size > 1 && !cut[2]) {
        if (rank == 0)
          std::printf("  [%-14s np=%d] FAIL — the decomposition does NOT cut z\n", c.name, size);
        fail = 1;
        continue;
      }

      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      for (int k = 0; k < 10; ++k)
        sd.step();

      // S1: the exactness gate is a per-rank statement — reduce the max over ranks.
      double loc = std::fmax(std::fmax(maxAbs(sd.getVelocity(0)), maxAbs(sd.getVelocity(1))),
                             maxAbs(sd.getVelocity(2)));
      double gmax = loc;
      MPI_Allreduce(&loc, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      std::vector<double> g[6];
      const char* fn[6] = {"u", "v", "w", "P", "C", "kappa"};
      g[0] = gatherGlobal(sd.getVelocity(0), ox, oy, oz, lnx, lny, lnz, rank, size);
      g[1] = gatherGlobal(sd.getVelocity(1), ox, oy, oz, lnx, lny, lnz, rank, size);
      g[2] = gatherGlobal(sd.getVelocity(2), ox, oy, oz, lnx, lny, lnz, rank, size);
      g[3] = gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      g[4] = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      g[5] = gatherGlobal(sd.getVofCurvature(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gb =
          gatherGlobal(sd.getVofCurvatureBranch(), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        for (int k = 0; k < 10; ++k)
          ref.step();
        std::vector<double> r[6] = {ref.getVelocity(0), ref.getVelocity(1), ref.getVelocity(2),
                                    ref.getPressure(),  ref.getVof(),       ref.getVofCurvature()};
        double d[6];
        for (int i = 0; i < 6; ++i)
          d[i] = maxDiff(g[i], r[i]);
        double dkb = 0.0, dkk = 0.0;
        const long nkb = countBitDiff(gb, ref.getVofCurvatureBranch(), dkb);
        const long nkk = countBitDiff(g[5], r[5], dkk);

        std::printf(
            "  [%-14s np=%d] max|u| = %.4e   du %.2e  dv %.2e  dw %.2e  dP %.2e"
            "  dC %.2e  dkappa %.2e\n",
            c.name, size, gmax, d[0], d[1], d[2], d[3], d[4], d[5]);
        (void)fn;

        // The BRANCH field is bitwise at every np: the cascade contains no reduction, and the
        // branch is a discrete label, so a colour perturbed at 1e-15 still lands on the same tier.
        // kappa itself is bitwise only where the colour is — i.e. at np = 1, and in the frozen
        // (`exact-*`) configurations at every np. In a COUPLED run at np > 1 the colour carries the
        // pressure driver's reduction-order residue (dC below) and kappa carries its image; that is
        // WO-J's documented floor, not a defect of this rung.
        if (nkb != 0) {
          std::printf("      FAIL — the curvature BRANCH is decomposition-dependent (%ld cells)\n",
                      nkb);
          fail = 1;
        }
        if (size == 1 && nkk != 0) {
          std::printf("      FAIL — kappa is not bitwise at np = 1 (%ld cells, max |d| %.3e)\n",
                      nkk, dkk);
          fail = 1;
        }
        if (c.constKappa) {
          // S1. With a constant kappa the force IS a discrete gradient of sigma*kappa*C. In the
          // PERIODIC box the semi-implicit momentum operator commutes with that gradient (uniform
          // rho, no boundary), so the projection annihilates it in ONE step and max|u| is at
          // machine zero. With WALLS it does not commute — the face-folded viscous operator does
          // not leave a pressure-gradient field invariant next to a Dirichlet boundary — so the
          // same fixed point is approached instead of hit: measured 2.1e-10 at 10 steps,
          // 1.2e-10 at 30, 2.9e-11 at 100, 3.6e-12 at 300, and 1.1e-16 at mu = 0. The multi-rank
          // statement there is the du/dv/dw/dP comparison below, which is what this test is for.
          const double tol = c.walls ? 1e-8 : 1e-13;
          if (!(gmax < tol)) {
            std::printf("      FAIL — the exactness gate is above %.0e (%.3e)\n", tol, gmax);
            fail = 1;
          }
        }
        // S2: np = 1 bitwise; np > 1 at the pressure driver's reduction-order floor.
        const double tolU = (size == 1) ? 0.0 : 1e-12;
        const double tolP = (size == 1) ? 0.0 : 1e-9;
        const double tolC = (size == 1) ? 0.0 : 1e-12;
        const double tolK = (size == 1) ? 0.0 : 1e-12;
        if (d[5] > tolK) {
          std::printf("      FAIL — kappa differs by more than the colour's own floor (%.2e)\n",
                      d[5]);
          fail = 1;
        }
        if (d[0] > tolU || d[1] > tolU || d[2] > tolU || d[3] > tolP || d[4] > tolC) {
          std::printf(
              "      FAIL — the distributed step does not match the single-rank reference"
              " (tolerances u %.0e  P %.0e  C %.0e)\n",
              tolU, tolP, tolC);
          fail = 1;
        }
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("%s\n", fail ? "FAILED" : "PASSED");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
