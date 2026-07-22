// flow — multi-rank validation of the MULTIPHYSICS layers on the assembled IbmSolver.
//
// The core incompressible solve is MPI-validated (test_sdflow_mpi); the layers on top — variable
// viscosity (setPropertyMode), variable density (setDensityMode, Chebyshev driver), the porous
// volume-averaged continuity (setPorousContinuity), and scalar transport (addScalar, periodic) —
// were structurally MPI-ready but had no multi-rank test. Each config here runs the distributed
// solver (ORB blocks, initMpi) against a full-grid single-rank reference on rank 0 and compares
// the final velocity field (and the scalar field) POINTWISE. np=1 must be bit-exact; np>1 to the
// solver reduction floor (MG-PCG dot-product Allreduce reorder; Chebyshev is reduction-free but
// shares the residual-norm checks).
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

using peclet::core::IVec;
using peclet::flow::IbmSolver;

static constexpr int N = 32, STEPS = 30;
static constexpr double RHO = 1.0, MU = 0.1, F = 1e-3, DT = 20.0;

static std::vector<double> packingSdf(double rfrac = 0.18) {
  const double R = rfrac * N;
  std::vector<double> sdf((std::size_t)N * N * N);
  const double cs[2] = {0.25 * N, 0.75 * N};
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        double best = 1e30;
        for (double sx : cs)
          for (double sy : cs)
            for (double sz : cs) {
              auto wrap = [](double d) { return d - N * std::round(d / N); };
              const double dx = wrap(x - sx), dy = wrap(y - sy), dz = wrap(z - sz);
              best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - R);
            }
        sdf[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] = best;
      }
  return sdf;
}

// Smooth global property fields (periodic-compatible).
static double muAt(int x, int y, int) {
  return MU * (1.2 + 0.5 * std::sin(2.0 * M_PI * x / N) * std::cos(2.0 * M_PI * y / N));
}
static double rhoAt(int, int, int z) { return RHO * (1.0 + 0.4 * std::sin(2.0 * M_PI * z / N)); }
static double epsAt(int, int, int z) { return 0.6 + 0.3 * std::sin(2.0 * M_PI * z / N); }
static double scalarAt(int x, int y, int z) {
  auto wrap = [](double d) { return d - N * std::round(d / N); };
  const double dx = wrap(x - 0.5 * N), dy = wrap(y - 0.5 * N), dz = wrap(z - 0.5 * N);
  return std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * 16.0));
}

// Extract this block from a global-field function (x-fastest local buffer).
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

struct Config {
  const char* name;
  bool varMu = false, varRho = false, porous = false, scalar = false, scalarBc = false;
};

static void configure(IbmSolver& s, const Config& c, const std::vector<double>& lsdf, int ox,
                      int oy, int oz, int lnx, int lny, int lnz) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0, 0);
  s.setAdvection(false);
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressurePcg(true, 200, 1e-9);
  s.setSolid(lsdf, /*cutcell_pressure=*/true);
  if (c.varMu) {
    s.setPropertyMode(true, /*harmonic=*/true);
    s.setField("mu", blockOf(muAt, ox, oy, oz, lnx, lny, lnz));
    s.exchangeField("mu");
  }
  if (c.varRho) {
    s.setDensityMode(true);  // installs the Chebyshev pressure driver by default
    s.setField("rho", blockOf(rhoAt, ox, oy, oz, lnx, lny, lnz));
    s.exchangeField("rho");
  }
  if (c.porous) {
    s.setPorousContinuity(true);
    s.setField("eps", blockOf(epsAt, ox, oy, oz, lnx, lny, lnz));
    s.exchangeField("eps");
  }
  if (c.scalar) {
    s.addScalar("c", /*D=*/0.05, /*scheme=*/1, /*iters=*/40);
    s.setField("c", blockOf(scalarAt, ox, oy, oz, lnx, lny, lnz));
    s.exchangeField("c");
  }
  if (c.scalarBc) {
    // Domain-BC scalar on the periodic flow (conjugated-transport style): hot/cold Dirichlet
    // x-faces; y/z stay periodic. Drives a conduction+advection profile through the
    // packing; validates the distributed per-face BC ownership (a rank applies a face's BC iff
    // its block touches that global face).
    s.addScalar("c", /*D=*/0.1, /*scheme=*/1, /*iters=*/40);
    // x faces: the ORB splits x first, so at np>=2 some rank does NOT touch a BC face — the
    // ownership test (touchesGlobalFace) is genuinely exercised, not trivially true.
    s.setScalarBc("c", /*face=*/0, /*type=*/2, /*value=*/1.0);
    s.setScalarBc("c", /*face=*/1, /*type=*/2, /*value=*/0.0);
    s.exchangeField("c");
  }
}

// Gather per-rank inner blocks (x-fastest, with per-rank origins/dims) into the global field on
// rank 0.
static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign((std::size_t)N * N * N, 0.0);
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

// max |a-b| / max|b| over the field.
static double relErr(const std::vector<double>& a, const std::vector<double>& b) {
  double md = 0, mb = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    md = std::max(md, std::fabs(a[i] - b[i]));
    mb = std::max(mb, std::fabs(b[i]));
  }
  return md / (mb + 1e-300);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::vector<double> gsdf = packingSdf();

    peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{N, N, N});
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
              gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N +
                   (std::size_t)(z + oz) * N * N];

    const Config configs[] = {{"varmu", true, false, false, false, false},
                              {"varrho", false, true, false, false, false},
                              {"porous", false, false, true, false, false},
                              {"scalar", false, false, false, true, false},
                              {"scalarbc", false, false, false, false, true}};

    for (const Config& c : configs) {
      // --- distributed ---
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(N, N, N, MPI_COMM_WORLD);
      configure(sd, c, lsdf, ox, oy, oz, lnx, lny, lnz);
      for (int it = 0; it < STEPS; ++it)
        sd.step();
      auto gu = gatherGlobal(sd.getVelocity(0), ox, oy, oz, lnx, lny, lnz, rank, size);
      std::vector<double> gc;
      if (c.scalar || c.scalarBc)
        gc = gatherGlobal(sd.getField("c"), ox, oy, oz, lnx, lny, lnz, rank, size);

      // --- single-rank reference on rank 0 (full grid, same config) ---
      double eu = -1, ec = -1;
      if (rank == 0) {
        IbmSolver ref(N, N, N);
        configure(ref, c, gsdf, 0, 0, 0, N, N, N);
        for (int it = 0; it < STEPS; ++it)
          ref.step();
        eu = relErr(gu, ref.getVelocity(0));
        const bool hasC = c.scalar || c.scalarBc;
        if (hasC)
          ec = relErr(gc, ref.getField("c"));
        const double tol = (size == 1) ? 1e-12 : 5e-5;
        const bool ok = eu <= tol && (!hasC || ec <= tol);
        std::printf("  [%-8s np=%d] u rel=%.3e%s  tol=%.0e  %s\n", c.name, size, eu,
                    hasC ? (std::string("  c rel=") + std::to_string(ec)).c_str() : "", tol,
                    ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("MULTIPHYSICS MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
