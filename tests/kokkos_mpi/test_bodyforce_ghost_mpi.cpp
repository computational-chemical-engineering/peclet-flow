// flow — the per-cell BODY-FORCE ghost ring (VoF blocker WO-G).
//
// The defect this test exists for. `applyClosure` (`property_closures.hpp`) writes the INNER cells
// only — its own comment says "ghosts untouched — refilled by the field's own exchange" — and an
// external CFD-DEM writer (`field_view` + `exchange_field_add`) likewise leaves the ghost ring
// holding its deposit residue rather than the owner's value. But NOTHING used to exchange
// `force_x/y/z`: `fillPropGhosts` was called for rho / mu / drag_beta / eps and never for the force
// fields, which are zero-initialised at registration. `buildRhsVar` face-interpolates the cell force
//
//     f_f(i) = 0.5*(fb(i) + fb(i - stride_c))
//
// so on the FIRST INNER PLANE of every block (axis c) the face force read exactly HALF of the
// intended value — at every rank boundary under MPI, and single-rank at the periodic wrap plane.
// The fix gives the force fields the same per-face-rank-aware ghost fill the other cell properties
// get (`fillCellForceGhosts` -> `fillPropGhosts`: halo/periodic base + Neumann copy on domain-BC
// faces, which is what keeps f_f/rho_f == the intended acceleration at a boundary, since rho's own
// ghost is a Neumann copy).
//
// The gates are ABSOLUTE physics, not a distributed-vs-reference comparison — deliberately. The
// single-rank reference carries the identical defect, so `du = 0` proves nothing here: `per-z` and
// `per-x` at np = 1 are exactly the single-rank periodic wrap-plane variant the work order asks for,
// and they FAIL before the fix. Configurations:
//
//   * `per-z`   — fully periodic, UNIFORM rho, UNIFORM force_z from a closure, inviscid, no
//                 advection. From rest the momentum solve is (rho/dt) w = (rho/dt) w^n + f_f, so
//                 w must be EXACTLY n*F*dt/rho in every cell after n steps; a uniform w is
//                 divergence-free so the projection returns it untouched. The gate is the spread
//                 max(w) - min(w) — i.e. "the face body force is uniform", read out through the one
//                 quantity that is a pure function of it.
//   * `per-x`   — the same, force_x written by an EXTERNAL writer (`setField`, no `exchangeField`)
//                 rather than a closure: the CFD-DEM feedback path, whose ghost ring holds the
//                 deposit residue. Different axis (stride 1) than `per-z` (stride ex*ey).
//   * `walls-z` — UNIFORM rho hydrostatic column, walls +-z, gravity closure force_z = -g*rho, z
//                 CUT. This is the WHY-DID-THE-ACID-TEST-PASS control: the wall-normal velocity on
//                 the boundary plane is PINNED by the Dirichlet BC (`bcVelocityComp`, comp == axis:
//                 `at(bf) = wall`) and that face's flux openness is 0, so the halved force at a
//                 GLOBAL wall face is masked completely — this configuration passes at np = 1 even
//                 with the defect present, which is exactly why WO-A's hydrostatic acid test read
//                 2.75e-17 throughout. An INTERIOR rank boundary has no such pin, so at np = 2/4
//                 with z cut the same configuration breaks in the PRESSURE (dP/dz) while the
//                 velocity canary stays clean — the WO-F signature.
//
// np = 1 is additionally compared bitwise against a full-grid single-rank reference; np > 1 lands
// on the usual MPI reduction-order floor (Chebyshev bound estimation + removeMean go through an
// MPI_SUM whose summation order depends on the rank count) — see `test_vardensity_mpi.cpp`.
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

static constexpr int NX = 16, NY = 16, NZ = 32, STEPS = 4;
static constexpr double RHO0 = 1.0, DT = 1.0, FORCE = 0.25, GRAV = 0.1;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  int comp;      // forced velocity component (0=x, 1=y, 2=z)
  bool walls;    // hydrostatic column (walls on the comp axis) vs fully periodic
  bool closure;  // force written by a closure vs by an external setField writer
  bool needCut;  // the forced axis MUST be cut at np > 1 (loud coverage loss if it stops being)
};

static void configure(IbmSolver& s, const Config& c, int lnx, int lny, int lnz) {
  static const char* fn[3] = {"force_x", "force_y", "force_z"};
  s.setRho(RHO0);
  s.setMu(0.0);  // inviscid: the momentum solve is exactly u^n + f_f*dt/rho_f
  s.setDt(DT);
  s.setAdvection(false);
  if (c.walls) {
    s.setDomainBc(2 * c.comp + 0, 1, 0, 0, 0);
    s.setDomainBc(2 * c.comp + 1, 1, 0, 0, 0);
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setDensityMode(true);  // uniform rho field == rho0; routes the RHS through buildRhsVar
  if (c.closure)
    s.setPropertyModel(fn[c.comp], peclet::flow::ClosureKind::LinearMix, "rho", "",
                       c.walls ? std::vector<double>{0.0, -GRAV} : std::vector<double>{FORCE, 0.0});
  else {
    s.enableCellForce();  // external-writer path (CFD-DEM): field_view/setField, no exchange
    s.setField(fn[c.comp], std::vector<double>((std::size_t)lnx * lny * lnz, FORCE));
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

// Hydrostatic gradient along the walled column at (x, y) = (NX/2, NY/2), uniform rho.
// The plane index of the worst offender is reported: with the defect it is the first inner plane of
// a rank block, and never the global wall (which the Dirichlet pin masks).
static double pressureGradientError(const std::vector<double>& p, int* worstPlane) {
  double perr = 0;
  *worstPlane = -1;
  for (int z = 1; z < NZ; ++z) {
    const std::size_t i0 =
        (std::size_t)(NX / 2) + (std::size_t)(NY / 2) * NX + (std::size_t)(z - 1) * NX * NY;
    const double e = std::fabs((p[i0 + (std::size_t)NX * NY] - p[i0]) + GRAV * RHO0) / (GRAV * RHO0);
    if (e > perr) {
      perr = e;
      *worstPlane = z;
    }
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
      std::printf("BODYFORCE-GHOST MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    // z is the axis this grid's ORB cuts (np=2 -> z, np=4 -> x and z), so the two z configurations
    // require the cut; `per-x` runs regardless — at np=2 its rank boundary is the PERIODIC WRAP
    // plane in x, which is the single-rank half of the same defect and an equally hard gate.
    const Config configs[] = {{"per-z", 2, false, true, true},
                              {"per-x", 0, false, false, false},
                              {"walls-z", 2, true, true, true}};

    for (const Config& c : configs) {
      if (c.needCut && size > 1 && !cut[c.comp]) {
        if (rank == 0)
          std::printf("  [%-7s np=%d] FAIL — the decomposition does NOT cut the forced axis %d; "
                      "this test exists to gate the force ghost ACROSS a rank boundary (WO-G)\n",
                      c.name, size, c.comp);
        fail = 1;
        continue;
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, lnx, lny, lnz);
      for (int it = 0; it < STEPS; ++it)
        sd.step();

      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, NX, NY, NZ);
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
        bool ok = du <= utol && dp <= ptol;

        char extra[160] = "";
        if (!c.walls) {
          // THE gate: the face body force is uniform <=> the accumulated velocity is uniform.
          const std::vector<double>& uf = gu[c.comp];
          double lo = uf[0], hi = uf[0];
          for (double v : uf) {
            lo = std::fmin(lo, v);
            hi = std::fmax(hi, v);
          }
          const double want = STEPS * FORCE * DT / RHO0;
          const double spread = hi - lo, verr = std::fabs(0.5 * (hi + lo) - want) / want;
          // The other two components must be identically zero (no force, no coupling).
          double cross = 0;
          for (int comp = 0; comp < 3; ++comp)
            if (comp != c.comp)
              cross = std::fmax(cross, maxAbs(gu[comp]));
          ok = ok && spread <= 1e-13 * want && verr <= 1e-13 && cross <= 1e-13 * want;
          std::snprintf(extra, sizeof(extra),
                        " | u_f spread=%.3e (want 0) value=%.17g (want %.17g) cross=%.2e", spread,
                        0.5 * (hi + lo), want, cross);
        } else {
          int worst = -1;
          const double perr = pressureGradientError(gp, &worst);
          const double umax = std::fmax(maxAbs(gu[0]), std::fmax(maxAbs(gu[1]), maxAbs(gu[2])));
          ok = ok && umax < 1e-13 && perr < 1e-11;
          std::snprintf(extra, sizeof(extra), " | max|u|=%.2e dP/dz err=%.3e (worst plane z=%d)",
                        umax, perr, worst);
        }
        std::printf("  [%-7s np=%d] du=%.3e dp=%.3e (tol %.1e/%.1e)%s  %s\n", c.name, size, du, dp,
                    utol, ptol, extra, ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("BODYFORCE-GHOST MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
