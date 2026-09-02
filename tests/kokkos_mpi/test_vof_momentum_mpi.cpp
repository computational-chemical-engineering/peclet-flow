// flow — multi-rank validation of MOMENTUM-CONSISTENT VoF transport (rung V2b, WO-K): `rho^c u_c`
// advected on the half-shifted MAC control volumes by the same geometric fluxes as the colour
// field.
//
// What is new here relative to `test_vof_twophase_mpi.cpp` (rung V2a), which gated the colour
// chain:
//
//   1. THE SHIFTED CONTROL VOLUMES AND THEIR OWNERSHIP. `C^c` and the transported velocity are
//      ordinary cell fields on the advector's g = 3 block, indexed in `flow`'s own low-face
//      convention precisely so that the owned control volumes ARE the block's inner region and the
//      colour halo carries them unchanged. If that correspondence were off by one along a
//      component's own axis, a control volume would be owned by two ranks or by none — invisible
//      single-rank, and visible here as a pointwise difference at the rank boundary.
//   2. SIX MORE HALO EXCHANGES PER SWEEP. `C^c` and the transported velocity are exchanged before
//      every sweep (the flux clamp reads the DONOR's own colour, which is a ghost on the first
//      owned face). Their ghost policy must be the colour's, i.e. globally-clamped on a
//      non-periodic axis (WO-E finding 2) — `walls-z` cuts the walled axis and exercises it.
//   3. THE CONSISTENCY IDENTITY UNDER DECOMPOSITION. The uniform-velocity identity is exact in
//      floating point; it must stay exact on every rank, which is a statement that the flux at a
//      rank boundary is computed identically on both sides.
//
// Comparison protocol is `tests/kokkos_mpi`'s: pointwise against a full-grid single-rank reference
// built on rank 0. np = 1 is bit-exact; np > 1 inherits the MPI reduction-order floor of the
// pressure driver (Chebyshev bounds + removeMean), which the colour and the momentum inherit
// through the velocity that advects them.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  bool walls;  // uniform-velocity identity in a walled box (else the periodic sheared sphere)
  double ratio;
  double dt;
  int steps;
};

static const double UNIF[3] = {1.0, 0.6, -0.4};

// A tilted sharp plane: a pure function of GLOBAL indices, so every block gets bit-identical data
// to the corresponding part of the single-rank run. Exact fractions from the analytic plane->volume
// relation the reconstruction inverts.
static double colourAt(int x, int y, int z) {
  double m[3] = {0.6, -0.5, 0.62456};
  const double l1 = std::fabs(m[0]) + std::fabs(m[1]) + std::fabs(m[2]);
  for (double& v : m)
    v /= l1;
  const double alpha = 0.5 * (m[0] * NX + m[1] * NY + m[2] * NZ);
  return peclet::flow::vof::plicVolume(m[0], m[1], m[2], alpha - (m[0] * x + m[1] * y + m[2] * z));
}
static double forceXAt(int, int, int z) {
  return 5e-2 * std::sin(2 * M_PI * (z + 0.5) / NZ);
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
  s.setRho(c.ratio);
  s.setMu(c.walls ? 0.0 : 0.05);
  s.setDt(c.dt);
  s.setAdvection(!c.walls);
  if (c.walls) {
    s.setDomainBc(4, 2, UNIF[0], UNIF[1], UNIF[2]);  // inflow  -z, the CUT axis
    s.setDomainBc(5, 3, 0, 0, 0);                    // outflow +z
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setVof(blockOf(colourAt, ox, oy, oz, lnx, lny, lnz));
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, c.ratio - 1.0});
  s.enableVofMomentum(1.0, c.ratio);
  if (!c.walls) {
    s.enableCellForce();
    s.setField("force_x", blockOf(forceXAt, ox, oy, oz, lnx, lny, lnz));
  }
  const std::size_t nc = (std::size_t)lnx * lny * lnz;
  if (c.walls)  // the uniform-velocity identity configuration
    s.uploadVelocity(std::vector<double>(nc, UNIF[0]), std::vector<double>(nc, UNIF[1]),
                     std::vector<double>(nc, UNIF[2]));
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
  // WO-R2: NaN-PROPAGATING. `std::fmax(m, NaN) == m`, so the obvious loop returns 0.000e+00 for a
  // field that has gone entirely NaN and every bitwise gate built on it passes (WO-R found this on
  // a drained open-boundary run). A non-finite difference must fail, so return it.
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;  // NaN
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
static double maxAbsDev(const std::vector<double>& a, double r) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v - r));
  return m;
}
// In a fully periodic box the pressure carries an arbitrary additive gauge, fixed by `removeMean`
// whose MPI_SUM order depends on the rank count. Comparing the DE-MEANED field measures the
// physics rather than that gauge; on the walled configuration the outflow Dirichlet pins P and the
// raw field is compared.
static std::vector<double> demean(std::vector<double> v) {
  double s = 0;
  for (double x : v)
    s += x;
  const double m = s / (double)v.size();
  for (double& x : v)
    x -= m;
  return v;
}
static double sumOf(const std::vector<double>& a) {
  double s = 0;
  for (double v : a)
    s += v;
  return s;
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
      std::printf("VOF MOMENTUM MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"walls-z", true, 1000.0, 0.2, 10},
                              {"shear-per", false, 10.0, 0.2, 40}};

    for (const Config& c : configs) {
      if (size > 1 && !cut[2]) {
        if (rank == 0)
          std::printf("  [%-9s np=%d] FAIL — the decomposition does NOT cut z\n", c.name, size);
        fail = 1;
        continue;
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);

      // THE CONSISTENCY IDENTITY, ON THE FIRST STEP, PER RANK (the `walls-z` configuration, which
      // is seeded with a uniform velocity): the advection's output must be uniform BITWISE — on
      // every rank, at every decomposition. That is the statement that the flux at a rank boundary
      // is computed identically on both sides.
      sd.step();
      double identLoc = 0.0;
      if (c.walls)
        for (int comp = 0; comp < 3; ++comp)
          identLoc = std::fmax(identLoc, maxAbsDev(sd.getVofAdvectedVelocity(comp), UNIF[comp]));
      double ident = 0.0;
      MPI_Allreduce(&identLoc, &ident, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      const double v0Loc = sumOf(sd.getVof());
      double v0 = 0.0;
      MPI_Allreduce(&v0Loc, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for (int it = 1; it < c.steps; ++it)
        sd.step();
      const double v1Loc = sumOf(sd.getVof());
      double v1 = 0.0;
      MPI_Allreduce(&v1Loc, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      std::vector<double> gu[3], gua[3];
      for (int comp = 0; comp < 3; ++comp) {
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
        gua[comp] =
            gatherGlobal(sd.getVofAdvectedVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      }
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        if (c.walls)
          std::printf("  [%-9s np=%d] uniform-velocity identity on the advection: %.17g%s\n",
                      c.name, size, ident, ident == 0.0 ? "   (BITWISE, all ranks)" : "");
        if (!(ident == 0.0)) {
          std::printf("  [%-9s np=%d] FAIL — the identity is not exact under decomposition\n",
                      c.name, size);
          fail = 1;
        }
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        for (int it = 0; it < c.steps; ++it)
          ref.step();
        double du = 0, umag = 0, dua = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          dua = std::fmax(dua, maxAbsDiff(gua[comp], ref.getVofAdvectedVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double dp = c.walls ? maxAbsDiff(gp, ref.getPressure())
                                  : maxAbsDiff(demean(gp), demean(ref.getPressure()));
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double pmag = maxAbs(ref.getPressure());
        const double drift = (v1 - v0) / v0;
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * pmag);
        const double ctol = (size == 1) ? 0.0 : 1e-11;
        std::printf(
            "  [%-9s np=%d] du %.3e  du_adv %.3e (tol %.1e, |u| %.2e)  dp %.3e (tol %.1e, "
            "|P| %.2e)  dC %.3e (tol %.1e)  dV/V %+.3e\n",
            c.name, size, du, dua, utol, umag, dp, ptol, pmag, dc, ctol, drift);
        if (!(du <= utol) || !(dua <= utol) || !(dp <= ptol) || !(dc <= ctol)) {
          std::printf("  [%-9s np=%d] FAIL — distributed vs single-rank\n", c.name, size);
          fail = 1;
        }
        if (!c.walls && !(std::fabs(drift) < 1e-11)) {
          std::printf("  [%-9s np=%d] FAIL — colour volume drift %.3e\n", c.name, size, drift);
          fail = 1;
        }
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf(fail ? "FAILED\n" : "OK\n");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
