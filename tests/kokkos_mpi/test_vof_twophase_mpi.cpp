// flow — multi-rank validation of the COUPLED VoF chain (rung V2a, WO-J): the colour field's own
// g = 3 halo inside the solver, the G = 2 <-> g = 3 bridge, and C -> rho(C)/mu(C) -> the
// variable-density projection.
//
// What is new here relative to `test_vof_advect_mpi.cpp` (rung V1), which gated the SAME advector
// standalone: there the colour halo and the face velocities were the test's own; here they are the
// solver's. Three things can only break under MPI and only in this configuration:
//
//   1. the colour field's own `GridHaloTopology` at width 3, built on the solver's partition and
//      living alongside the G = 2 velocity halo. Two topologies of DIFFERENT ghost width are in
//      flight in the same step.
//   2. the face-velocity bridge across a rank boundary. `bridgeVelocityToVof` embeds the whole
//      G = 2 velocity block (inner + both ghost layers, filled by `fillVelGhosts`) into the g = 3
//      block with the low-face -> high-face index shift. A flux at a rank boundary must come out
//      bit-identical on both sides or the WY telescoping — hence conservation — dies exactly there.
//   3. the non-periodic colour ghost fill. `vof::clampFill` takes the value at the GLOBALLY clamped
//      index precisely so it is decomposition-independent; sequential zero-gradient axis passes
//      over the block are NOT (WO-E finding 2), and the difference reaches C through the MYC
//      stencil of the inner corner cell. `walls-z` cuts the walled axis, so this is exercised.
//
// Two configurations on a 16x16x32 grid — the aligned ORB cuts the long z axis at np = 2 and 4:
//
//   * `walls-z` — the hydrostatic acid test DRIVEN THROUGH C: a sharp two-layer colour field,
//     rho = 1 + (ratio-1)*C by a LinearMix closure, gravity force_z = -g*rho, ratio 1000,
//     walls +-z.
//     Gates dP/dz == -rho_f*g and the pointwise velocity AND pressure against the single-rank
//     reference. The pressure is the sharp gate (WO-F: a rank-unaware wall BC is invisible in the
//     velocity — each sub-column is separately hydrostatic — and reads 4.0e+02 in the pressure).
//
//   * `shear-per` — the moving-interface case: a ratio-10 sphere in a periodic box under a sheared
//     body force, advection on, 40 steps. Gates the colour field pointwise against the single-rank
//     reference and the GLOBAL colour volume. This is the configuration that catches a bridge or a
//     halo defect: the interface sweeps across the rank boundary in z.
//
// Comparison protocol (the `tests/kokkos_mpi` pattern): the distributed run is compared POINTWISE
// against a full-grid single-rank reference built on rank 0 with the identical configuration.
// np = 1 is bit-exact on every field. np > 1 cannot be bitwise by construction — Chebyshev's bound
// estimation and `removeMean` go through MPI_SUM allreduces whose summation order depends on the
// rank count — so the velocity/pressure/colour tolerances are the same reduction-order floor
// `test_vardensity_mpi.cpp` uses. The colour field itself has NO reduction in its update, so it is
// bit-exact whenever the velocity driving it is; what it inherits is the velocity's floor.
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
static constexpr double GRAV = 0.1, RHO0 = 1.0;

struct Config {
  const char* name;
  bool walls;    // hydrostatic acid test through C (else the periodic sheared sphere)
  double ratio;  // density ratio
  double dt;
  int steps;
};

// The colour field of a configuration at a GLOBAL cell index — a pure function of global indices,
// so a block gets bit-identical data to the corresponding part of the single-rank run.
static double colourAt(const Config& c, int x, int y, int z) {
  if (c.walls)
    return (z < NZ / 2) ? 1.0 : 0.0;  // sharp: heavy (liquid) below, stable stratification
  const double dx = x + 0.5 - NX / 2.0, dy = y + 0.5 - NY / 2.0, dz = z + 0.5 - NZ / 2.0;
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  return std::fmin(1.0, std::fmax(0.0, 0.5 - (r - 4.0)));
}
static double forceXAt(int, int, int z) {
  return 5e-4 * std::sin(2 * M_PI * (z + 0.5) / NZ);
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
  s.setMu(c.walls ? 0.0 : 0.05);  // hydrostatic: inviscid, so the balance is exact
  s.setDt(c.dt);
  s.setAdvection(!c.walls);
  if (c.walls) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z (the CUT axis)
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setVof(blockOf([&](int x, int y, int z) { return colourAt(c, x, y, z); }, ox, oy, oz, lnx, lny,
                   lnz));
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                     {1.0, c.ratio - 1.0});  // enables the variable-density path
  if (c.walls) {
    s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "", {0.0, -GRAV});
  } else {
    s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {0.05, 0.45});
    s.enableCellForce();
    s.setField("force_x", blockOf(forceXAt, ox, oy, oz, lnx, lny, lnz));
  }
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
static double sumOf(const std::vector<double>& a) {
  double s = 0;
  for (double v : a)
    s += v;
  return s;
}

// Discrete hydrostatic gradient along the walled column, against the ACTUAL colour field.
static double pressureGradientError(const std::vector<double>& p, const std::vector<double>& c,
                                    double ratio) {
  double perr = 0;
  for (int z = 1; z < NZ; ++z) {
    const std::size_t i0 =
        (std::size_t)(NX / 2) + (std::size_t)(NY / 2) * NX + (std::size_t)(z - 1) * NX * NY;
    const std::size_t i1 = i0 + (std::size_t)NX * NY;
    const double rf = 0.5 * ((1.0 + (ratio - 1.0) * c[i1]) + (1.0 + (ratio - 1.0) * c[i0]));
    perr = std::fmax(perr, std::fabs((p[i1] - p[i0]) + GRAV * rf) / (GRAV * ratio));
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
      std::printf("VOF TWOPHASE MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"walls-z", true, 1000.0, 1.0, 20},
                              {"shear-per", false, 10.0, 1.0, 40}};

    for (const Config& c : configs) {
      // The walled/loaded axis must be CUT: `walls-z` exists to gate the colour field's
      // non-periodic ghost fill and the domain-BC ownership on a cut axis, and `shear-per` exists
      // to sweep the interface across a rank boundary. Fail loudly if the ORB stops cutting z.
      if (size > 1 && !cut[2]) {
        if (rank == 0)
          std::printf("  [%-9s np=%d] FAIL — the decomposition does NOT cut z\n", c.name, size);
        fail = 1;
        continue;
      }
      // --- distributed ---
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      const double v0Loc = sumOf(sd.getVof());
      double v0 = 0.0;
      MPI_Allreduce(&v0Loc, &v0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for (int it = 0; it < c.steps; ++it)
        sd.step();
      const double v1Loc = sumOf(sd.getVof());
      double v1 = 0.0;
      MPI_Allreduce(&v1Loc, &v1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);

      // --- single-rank full-grid reference on rank 0 ---
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        const double rv0 = sumOf(ref.getVof());
        for (int it = 0; it < c.steps; ++it)
          ref.step();
        const double refDrift = (sumOf(ref.getVof()) - rv0) / rv0;
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const auto refC = ref.getVof();
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double dc = maxAbsDiff(gc, refC);
        const double pmag = maxAbs(ref.getPressure());
        const double drift = (v1 - v0) / v0;
        // np=1 bit-exact; np>1 the MPI reduction-order floor (Chebyshev bounds + removeMean),
        // which the colour field inherits through the velocity that advects it.
        // The absolute floor matters on `walls-z`: |u| there IS round-off (6e-13), so a purely
        // relative tolerance would be gating the reduction-order pattern of noise, not the physics.
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * pmag);
        const double ctol = (size == 1) ? 0.0 : 1e-11;
        std::printf(
            "  [%-9s np=%d] du %.3e (tol %.1e, |u| %.2e)  dp %.3e (tol %.1e, |P| %.2e)  "
            "dC %.3e (tol %.1e)  dV/V %+.3e\n",
            c.name, size, du, utol, umag, dp, ptol, pmag, dc, ctol, drift);
        if (!(du <= utol) || !(dp <= ptol) || !(dc <= ctol)) {
          std::printf("  [%-9s np=%d] FAIL — distributed vs single-rank\n", c.name, size);
          fail = 1;
        }
        // Colour volume. On `shear-per` the interface is transported by a deliberate flow and the
        // drift must be at the projection's divergence floor. On `walls-z` the colour DOES move
        // once — the pressure driver's first solve on a fresh field leaves max|u| ~ 8.6e-6 (the
        // documented varRho transient, doc/variable_density_projection.md §2), and VoF faithfully
        // advects it into a permanent ~7e-7 displacement before the velocity collapses to ~1e-13.
        // That is a property of the pressure solve, not of the advection or of the decomposition,
        // so what this test gates there is that it is DECOMPOSITION-INDEPENDENT.
        if (c.walls) {
          std::printf("  [%-9s np=%d] colour drift dist %+.4e ref %+.4e (delta %.2e)\n", c.name,
                      size, drift, refDrift, std::fabs(drift - refDrift));
          if (!(std::fabs(drift - refDrift) < 1e-13)) {
            std::printf("  [%-9s np=%d] FAIL — colour drift is decomposition-dependent\n", c.name,
                        size);
            fail = 1;
          }
        } else if (!(std::fabs(drift) < 1e-11)) {
          std::printf("  [%-9s np=%d] FAIL — colour volume drift %.3e\n", c.name, size, drift);
          fail = 1;
        }
        if (c.walls) {
          const double perr = pressureGradientError(gp, gc, c.ratio);
          const double umaxD = std::fmax(maxAbs(gu[0]), std::fmax(maxAbs(gu[1]), maxAbs(gu[2])));
          std::printf("  [%-9s np=%d] hydrostatic through C: max|u| %.3e  dP/dz rel-err %.3e\n",
                      c.name, size, umaxD, perr);
          if (!(perr < 1e-11) || !(umaxD < 1e-9)) {
            std::printf("  [%-9s np=%d] FAIL — hydrostatic balance through C\n", c.name, size);
            fail = 1;
          }
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
