// flow — multi-rank validation of the VARIABLE-DENSITY projection (VoF rung V-1, WO-A).
//
// `doc/variable_density_projection.md` §4 listed "MPI/CUDA validation deferred": the rho bridge to
// the g=1 MG block (`copyBlockShifted`, offset G-1), the `fillPropGhosts` ghost ring, and the
// Chebyshev bound estimation were *believed* MPI-ready. This test gates them.
//
// Two configurations on a 32x16x16 grid (the aligned ORB cuts only x at np = 1, 2, 4):
//
//   * `walls-z` — the WO-A hydrostatic acid test, the multi-rank twin of
//     `tests/kokkos/test_vardensity_projection.cpp`: two layers at density ratio 1000 at rest
//     between no-slip walls at +-z under the gravity closure force_z = -g*rho. From rest the face
//     force -g*rho_f divided by the face inertia rho_f/dt gives a UNIFORM w* = -g*dt, so the
//     interior divergence vanishes, the projection returns u == 0, and P accumulates exactly
//     -rho_f*g per face. A stale rho ghost, a rho bridge that misses the g=1 ghost ring, or a
//     mismatched face mean breaks the telescoping and leaves a PERMANENT spurious velocity. Both
//     gates are decomposition-independent physics: max|u| ~ 1e-16 and dP/dz == -g*rho_f at every np.
//     The pressure gate is the sharper one — see the walled-axis note below.
//
//   * `jump-x` — the rho-ghost / rho-bridge canary: a SHARP ratio-1000 density jump stacked along
//     the CUT axis, so at np = 2 and 4 the jump sits exactly on a rank boundary and the coefficient
//     rho0/rho_f at that face is assembled from an exchanged ghost. Fully periodic, driven by a
//     uniform body force (no rest state — this one is a pure np-consistency + Chebyshev-count gate).
//
// WALLED AXIS / MPI (WO-A finding, escalated — see the findings log in doc/vof_workorders.md):
// flow's per-face domain BCs are imposed by EVERY rank on its own block faces —
// `applyVelocityBcCompTo` (and the pressure-openness BC) have no `touchesGlobalFace` ownership
// test, unlike the scalar BCs (`applyScalarBc`). If the decomposition cuts a walled axis the
// domain splits into independent walled sub-columns: each one is separately hydrostatic, so the
// VELOCITY canary still reads ~1e-17 and only the PRESSURE reveals it (measured on a 16x16x32 grid
// with z cut: max|u| = 4.5e-17 but max|P_dist - P_ref| = 4.0e+02 and dP/dz off by 8x g*rho). The
// configuration below therefore keeps the walled axis uncut and ASSERTS it, so the day the
// decomposition changes this fails loudly instead of silently passing.
//
// Comparison protocol (the `tests/kokkos_mpi` pattern): the distributed run is compared POINTWISE
// against a full-grid single-rank reference built on rank 0 with the identical configuration.
// np = 1 is bit-exact; np > 1 lands on the MPI reduction-order floor — Chebyshev's bound
// estimation (`dot`) and `removeMean` both go through an MPI_SUM allreduce whose summation order is
// a function of the rank count, so np > 1 cannot be bitwise by construction. The Chebyshev V-cycle
// COUNT per step is reported and gated at +-1 (the same knife-edge on the `maxabs(r) < rtol*r0`
// stopping test is reproducible at np = 1 by changing only the OpenMP thread count).
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

static constexpr int NX = 32, NY = 16, NZ = 16, STEPS = 20;
static constexpr double RATIO = 1000.0, GRAV = 0.1, DT = 1.0, RHO0 = 1.0, FX = 1e-3;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  int axis;    // density-layering axis (2 = z for the walled column, 0 = x for the cut-axis jump)
  bool walls;  // hydrostatic acid test (else the periodic body-force consistency probe)
};

static double rhoAt(const Config& c, int x, int y, int z) {
  const int q[3] = {x, y, z}, n[3] = {NX, NY, NZ};
  return (q[c.axis] < n[c.axis] / 2) ? RATIO : 1.0;  // heavy first half
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
  s.setRho(RHO0);
  s.setMu(c.walls ? 0.0 : 0.1);  // hydrostatic: inviscid, so the balance is exact
  s.setDt(DT);
  s.setAdvection(false);
  if (c.walls) {
    s.setDomainBc(2 * c.axis + 0, 1, 0, 0, 0);
    s.setDomainBc(2 * c.axis + 1, 1, 0, 0, 0);
  } else {
    s.setBodyForce(FX, 0, 0);
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setDensityMode(true);  // installs the Chebyshev pressure driver by default
  s.setField("rho", blockOf([&](int x, int y, int z) { return rhoAt(c, x, y, z); }, ox, oy, oz,
                            lnx, lny, lnz));
  s.exchangeField("rho");
  if (c.walls)  // gravity closure: the per-cell force -g*rho, face-interpolated to -g*rho_f
    s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -GRAV});
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
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

// Discrete hydrostatic gradient along the walled column (x = NX/2, y = NY/2).
static double pressureGradientError(const std::vector<double>& p) {
  double perr = 0;
  for (int z = 1; z < NZ; ++z) {
    const std::size_t i0 =
        (std::size_t)(NX / 2) + (std::size_t)(NY / 2) * NX + (std::size_t)(z - 1) * NX * NY;
    const std::size_t i1 = i0 + (std::size_t)NX * NY;
    const double rf = 0.5 * (((z < NZ / 2) ? RATIO : 1.0) + ((z - 1 < NZ / 2) ? RATIO : 1.0));
    perr = std::fmax(perr, std::fabs((p[i1] - p[i0]) + GRAV * rf) / (GRAV * RATIO));
  }
  return perr;
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
      std::printf("VARDENSITY MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"walls-z", 2, true}, {"jump-x", 0, false}};

    for (const Config& c : configs) {
      if (c.walls && cut[c.axis]) {
        // Not a workaround: flow's per-face domain BCs have no rank-ownership test, so a cut
        // walled axis splits the column into independent hydrostatic sub-columns (see the header).
        if (rank == 0)
          std::printf("  [%-7s np=%d] FAIL — the decomposition cuts the walled axis %d; flow's "
                      "per-face domain BCs have no rank-ownership test (WO-A finding)\n",
                      c.name, size, c.axis);
        fail = 1;
        continue;
      }
      // --- distributed ---
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      std::vector<long> itd;
      for (int it = 0; it < STEPS; ++it) {
        sd.step();
        itd.push_back(sd.lastPressureIterations());
      }

      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);

      // --- single-rank full-grid reference on rank 0 ---
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        std::vector<long> itr;
        std::vector<double> refUmax;
        for (int it = 0; it < STEPS; ++it) {
          ref.step();
          itr.push_back(ref.lastPressureIterations());
          refUmax.push_back(std::fmax(maxAbs(ref.getVelocity(0)),
                                      std::fmax(maxAbs(ref.getVelocity(1)),
                                                maxAbs(ref.getVelocity(2)))));
        }
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double pmag = maxAbs(ref.getPressure());
        // np=1 bit-exact; np>1 the MPI reduction-order floor (Chebyshev bounds + removeMean).
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * pmag);
        // Chebyshev V-cycle count vs the decomposition — over the NON-DEGENERATE steps only. Once
        // the hydrostatic state has reached machine zero the solve's own r0 is round-off noise and
        // the `maxabs(r) < rtol*r0` stopping test is a knife edge: at FIXED np=1 the count sequence
        // already changes with the OpenMP thread count alone (measured: steps 7.. read
        // 15,13,13,... at 1 thread, 16,13,14,... at 2, 16,14,14,13,... at 8). Gating those steps
        // would gate round-off, not the bounds path. The window predicate is the reference's own
        // velocity magnitude entering the step.
        long dits = 0;
        int nWin = 0;
        for (std::size_t k = 0; k < itr.size(); ++k) {
          if (k > 0 && refUmax[k - 1] <= 1e-12)
            continue;
          ++nWin;
          dits = std::max(dits, std::labs(itd[k] - itr[k]));
        }
        const long itTol = 0;  // the bounds path must be decomposition-independent, exactly
        bool ok = du <= utol && dp <= ptol && dits <= itTol;
        double umax = -1, perr = -1;
        if (c.walls) {  // decomposition-independent physics gates
          umax = std::fmax(maxAbs(gu[0]), std::fmax(maxAbs(gu[1]), maxAbs(gu[2])));
          perr = pressureGradientError(gp);
          ok = ok && umax < 1e-14 && perr < 1e-11;
        }
        char extra[80] = "";
        if (c.walls)
          std::snprintf(extra, sizeof(extra), " | max|u|=%.2e dP/dz err=%.2e", umax, perr);
        std::printf("  [%-7s np=%d] du=%.3e dp=%.3e (tol %.1e/%.1e) | cheb its %ld..%ld "
                    "max-delta=%ld over %d/%d non-degenerate steps%s  %s\n",
                    c.name, size, du, dp, utol, ptol, itr.front(), itr.back(), dits, nWin, STEPS,
                    extra, ok ? "OK" : "FAIL");
        std::printf("    dist its:");
        for (long v : itd)
          std::printf(" %ld", v);
        std::printf("\n    ref  its:");
        for (long v : itr)
          std::printf(" %ld", v);
        std::printf("\n");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("VARDENSITY MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
