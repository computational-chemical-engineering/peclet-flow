// flow — a NON-UNIFORM per-cell body force under MPI, and the init_mpi / geometry call order.
//
// The defect this test exists for (2026-09-24, found through coupling's
// `test_mpi_moving_suspension` at np = 4: rel-err 1.06e-2 against 1e-4). A caller that built the
// geometry BEFORE `initMpi` (set_pressure_geometry, then init_mpi) got a solver whose momentum
// solve was distributed but whose pressure multigrid was not: `setSolid` builds the pressure and
// velocity hierarchies for the partition in force when it runs, which before `initMpi` is the
// single-rank one, so every rank's hierarchy treated its own block as a periodic domain of its own
// and `initMpi` never rebuilt it. Each rank then ran an independent pressure solve; every PCG
// converged, and the projected velocity kept the divergence of its block-local mean. One step of
// the coupling test's Solver with a fixed Gaussian `force_x`: max|du| = 4.05e-2 at np = 4 (2x2x1,
// max|u| 0.358), 1.9e-5 at np = 2 (x cut only, where the blobs happen to be block-symmetric). One
// rank is exact by accident, since its block is the domain. The cell force was the messenger, not
// the cause: any velocity whose divergence has a non-zero mean over a rank block shows it.
//
// `initMpi` now raises when the geometry already exists, at every rank count. Two parts:
//
//   * `order`  — setPressureGeometry, then initMpi, must throw std::runtime_error. This is the
//                part that FAILS before the fix (the call returned and the run went on wrong).
//   * the cell-force comparisons, in the correct order, against a full-grid single-rank
//     reference — the coverage the existing MPI tests lacked: `test_bodyforce_ghost_mpi` drives
//     only UNIFORM forces, which a wrong neighbour, a wrong corner ghost or a block-local solve
//     all leave invisible. Gaussian blobs on the coupling test's particle lattice, including rows
//     centred ON the 16-planes the ORB cuts (np = 4 is 2x2x1 on this grid: x and y cut, each
//     periodic axis with the SAME neighbour on both sides; np = 2 cuts x):
//       - `forced-x`: constant rho -> buildRhsForced (the force read at the cell), force_x, the
//                     blob line along x — the reproducer of the report, verbatim.
//       - `var-y`:    density mode -> buildRhsVar (the force FACE-AVERAGED, so it reads the ghost
//                     ring `fillCellForceGhosts` fills), force_y, the blob line along y.
//     Both written the CFD-DEM way (enableCellForce + setField: inner cells only, no exchange).
//     Two steps, advection on (the default), so step 2 also advects the forced field.
//
// Gates: np = 1 bitwise against the reference; np > 1 at the MG-PCG reduction-order floor (the
// inner-product Allreduce reorders the Krylov path — test_sdflow_mpi, test_bodyforce_ghost_mpi),
// with the pressure driver pinned to MG-PCG at rtol 1e-12 on both sides so the default-driver
// difference (standalone V-cycle multi-rank, MG-PCG single-rank) is not what is measured.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 2;
static constexpr std::size_t GCELLS = (std::size_t)N * N * N;

struct Config {
  const char* name;
  int comp;      // forced velocity component, and the axis the six-blob line runs along
  bool density;  // uniform-rho density mode (buildRhsVar) vs constant rho (buildRhsForced)
};

// The coupling test's lattice: six blobs along axis `line` at (i + 1/2) N/6, times a 3x3 grid at
// {10, 16, 22} on the other two axes. Unwrapped Gaussians, the same global field at every np.
static std::vector<double> blobs(int line) {
  std::vector<double> f(GCELLS, 0.0);
  const double pl[3] = {10.0, 16.0, 22.0};
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double c[3] = {x + 0.5, y + 0.5, z + 0.5};
        const int a = (line + 1) % 3, b = (line + 2) % 3;
        double s = 0.0;
        for (int i = 0; i < 6; ++i)
          for (double pa : pl)
            for (double pb : pl) {
              const double dl = c[line] - (i + 0.5) * N / 6.0, da = c[a] - pa, db = c[b] - pb;
              s += -12.0 * std::exp(-(dl * dl + da * da + db * db));
            }
        f[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = s;
      }
  return f;
}

static std::vector<double> slice(const std::vector<double>& g, int ox, int oy, int oz, int lnx,
                                 int lny, int lnz) {
  std::vector<double> l((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        l[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            g[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];
  return l;
}

// Everything but the geometry, so the order part can put the geometry where it likes.
static void configure(IbmSolver& s, const Config& c, const std::vector<double>& forceLocal) {
  static const char* fn[3] = {"force_x", "force_y", "force_z"};
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(0.1);
  if (c.density)
    s.setDensityMode(true);  // uniform rho field == rho0; routes the RHS through buildRhsVar
  s.setPressurePcg(true, 200, 1e-12);  // LAST: the density mode re-selects the driver
  s.enableCellForce();                 // external-writer path (CFD-DEM): setField, no exchange
  s.setField(fn[c.comp], forceLocal);
}

// Gather per-rank inner blocks (x-fastest) into the global field on rank 0.
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
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * N +
                                (std::size_t)(z + oz) * N * N],
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
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * N +
                              (std::size_t)(z + meta[2]) * N * N],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

// NaN-propagating (see test_bodyforce_ghost_mpi: std::fmax drops a NaN and a gate built on it
// passes on a field that has gone entirely NaN).
static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != N)
          cut[a] = true;
    if (rank == 0)
      std::printf("CELLFORCE MPI np=%d  grid %d^3  block %dx%dx%d  cut axes: %s%s%s\n", size, N,
                  lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "");
    const std::vector<double> allFluid((std::size_t)lnx * lny * lnz, 10.0);

    // ---- order: a geometry built before initMpi must be refused -------------------------------
    {
      IbmSolver s(lnx, lny, lnz);
      s.setPressureGeometry(allFluid);
      bool threw = false;
      std::string what;
      try {
        s.initMpi(dec, MPI_COMM_WORLD);
      } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
      }
      // The guard is rank-local and replicated (every rank built its geometry first), so every
      // rank must agree; a rank that did not throw would be left half-initialised.
      int t = threw ? 1 : 0, all = 0;
      MPI_Allreduce(&t, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      const bool ok = all == 1 && what.find("init_mpi") != std::string::npos;
      if (rank == 0)
        std::printf("  [order    np=%d] set_pressure_geometry -> init_mpi %s  %s\n", size,
                    threw ? "raised" : "was ACCEPTED (the rank-local pressure solve)",
                    ok ? "OK" : "FAIL");
      if (!ok)
        fail = 1;
    }

    // ---- non-uniform cell force, correct order, vs the single-rank reference ------------------
    const Config configs[] = {{"forced-x", 0, false}, {"var-y", 1, true}};
    for (const Config& c : configs) {
      const std::vector<double> fg = blobs(c.comp);
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      sd.setPressureGeometry(allFluid);
      configure(sd, c, slice(fg, ox, oy, oz, lnx, lny, lnz));
      for (int it = 0; it < STEPS; ++it)
        sd.step();
      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(N, N, N);
        ref.setPressureGeometry(std::vector<double>(GCELLS, 10.0));
        configure(ref, c, fg);
        for (int it = 0; it < STEPS; ++it)
          ref.step();
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double pmag = maxAbs(ref.getPressure());
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * pmag);
        // A forced flow, not a trivially zero one: the gate must have something to compare.
        const bool ok = du <= utol && dp <= ptol && umag > 0.1;
        std::printf("  [%-8s np=%d] du=%.3e dp=%.3e (tol %.1e/%.1e) max|u|=%.3e max|p|=%.3e  %s\n",
                    c.name, size, du, dp, utol, ptol, umag, pmag, ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("CELLFORCE MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
