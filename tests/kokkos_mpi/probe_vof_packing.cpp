// DIAGNOSTIC PROBE — NOT a gate, not a ctest (EXCLUDE_FROM_ALL). This is the instrument that
// answered docs/wo_vof_mpi_parity_gates.md; it is committed so the trace behind
// `test_vof_bc_mpi.cpp`'s packing tolerances can be reproduced instead of taken on trust.
//
//   cmake --build build_dev --target probe_vof_packing
//   OMP_NUM_THREADS=8 mpirun -np 2 ./build_dev/tests/kokkos_mpi/probe_vof_packing
//
// It runs `configurePacking`'s scene distributed and single-rank IN LOCKSTEP, gathering the colour
// after every step, so "does the disagreement appear at once or grow from round-off" is answered by
// a curve rather than one end-of-run number. (It appears at once: 4.4e-16 after step 0, 4.795e-09
// after step 1, decaying thereafter — a branch flipping, not an accumulation.)
//
// Per step it prints the max |C_np - C_ref| and where; how many cells exceed 1e-11; the max
// difference of u, v, w and p (which stay at the allreduce floor while the colour does not); the
// census of the two threshold predicates that could have explained it and do not (the `C > 0.5`
// Weymouth-Yue dilation flag and `wyIsMixed`'s wisp tolerance); and both pressure iteration counts.
//
// Knobs:
//   PROBE_STEPS=n     how many steps (default 40, as the test)
//   PROBE_KINEMATIC=1 prescribe a UNIFORM w and call advectVof() instead of step() — no pressure
//                 =2  prescribe a 3-D VARYING analytic velocity instead. This is the decisive one:
//                     it is bitwise at every np and thread count, which is what proves the VoF
//                     path innocent and became the test's `packing-kin` case. A uniform field
//                     (=1) proves much less — every value being equal hides a mis-filled ghost.
//   PROBE_ULP=1       perturb the REFERENCE's initial colour by one ulp in one far cell
//   PROBE_NOSOLID=1   drop the packing SDF
//   PROBE_RTOL=x      pin the pressure rtol (default: the solver's own 1e-10)
//   PROBE_WISP=x      set the wisp tolerance (default 1e-8, what enable_vof sets)
//   PROBE_LIST=1      list every cell differing by more than 1e-11, at full precision
//   PROBE_CELL=1      dump the corner face velocities at full precision
//   PROBE_PAD=1       per-field max difference over the PADDED block — ghosts included, which
//                     getField() cannot show
//   PROBE_PADCELL=1   list every differing padded value near the corner (14..15, 0, 0)
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

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
                      &buf[(std::size_t)meta[3] * 0 + (std::size_t)y * meta[3] +
                           (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

static std::vector<double> packingSdf(int ox, int oy, int oz, int lnx, int lny, int lnz) {
  const double sp[5][4] = {{4.5, 4.5, 11, 3.4},
                           {11.5, 11.5, 11, 3.4},
                           {4.5, 11.5, 20, 3.4},
                           {11.5, 4.5, 20, 3.4},
                           {8.0, 8.0, (double)NZ, 3.6}};
  std::vector<double> f((std::size_t)lnx * lny * lnz, 1e30);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        double d = 1e30;
        for (const auto& q : sp) {
          const double dx = x + ox + 0.5 - q[0], dy = y + oy + 0.5 - q[1], dz = z + oz + 0.5 - q[2];
          d = std::fmin(d, std::sqrt(dx * dx + dy * dy + dz * dz) - q[3]);
        }
        f[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = d;
      }
  return f;
}

static double envd(const char* k, double dflt) {
  const char* v = std::getenv(k);
  return v ? std::atof(v) : dflt;
}

static void configurePacking(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz,
                             bool perturb = false) {
  s.setRho(1.0);
  s.setMu(0.5);
  s.setDt(0.1);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, 0.5);
  s.setDomainBc(5, 3, 0, 0, 0);
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressureIterations(400);
  if (!std::getenv("PROBE_NOSOLID"))
    s.setSolid(packingSdf(ox, oy, oz, lnx, lny, lnz), true);
  s.enableVof();
  const double rtol = envd("PROBE_RTOL", 0.0);
  if (rtol > 0.0)
    s.setPressurePcg(true, 400, rtol);
  const double wisp = envd("PROBE_WISP", -1.0);
  if (wisp >= 0.0)
    s.setVofWispEps(wisp);
  std::vector<double> c0((std::size_t)lnx * lny * lnz, 0.0);
  for (int z = 0; z < lnz; ++z)
    if (z + oz >= NZ / 2)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          c0[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = 1.0;
  // PROBE_ULP: perturb the initial colour of ONE cell at the far +z end by one ulp -- a change
  // of 1.1e-16 in one cell of the initial data, far smaller than any decomposition's
  // reduction-order noise, and applied to the SINGLE-RANK run so no MPI is involved at all.
  if (perturb) {
    const std::size_t kp =
        (std::size_t)0 + (std::size_t)0 * lnx + (std::size_t)(lnz - 1) * lnx * lny;
    c0[kp] = std::nextafter(1.0, 0.0);
  }
  s.setVof(c0);
  s.setVofInflow(4, 1.0);
  s.setVofBackflow(5, 0.0);
  if (std::getenv("PROBE_KINEMATIC")) {
    // Drive the colour with a PRESCRIBED velocity built from GLOBAL cell centres: the cut-cell
    // geometry and the open boundaries stay, the pressure/velocity solve is gone. Isolates the VoF
    // path from the flow solve. PROBE_KINEMATIC=2 makes it spatially VARYING -- a uniform field
    // hides every ghost-fill and bridge difference, because all the values are equal.
    const bool vary = std::atoi(std::getenv("PROBE_KINEMATIC")) >= 2;
    std::vector<double> fu((std::size_t)lnx * lny * lnz, 0.0), fv(fu), fw(fu);
    for (int z = 0; z < lnz; ++z)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x) {
          const std::size_t k = (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
          const double gx = x + ox + 0.5, gy = y + oy + 0.5, gz = z + oz + 0.5;
          if (!vary) {
            fw[k] = 0.5;
            continue;
          }
          const double pi = 3.14159265358979323846;
          fu[k] = 0.10 * std::sin(2 * pi * gx / NX) * std::cos(2 * pi * gy / NY) *
                  std::cos(pi * gz / NZ);
          fv[k] = 0.10 * std::cos(2 * pi * gx / NX) * std::sin(2 * pi * gy / NY) *
                  std::cos(pi * gz / NZ);
          fw[k] = 0.50 + 0.10 * std::cos(2 * pi * gx / NX) * std::cos(2 * pi * gy / NY) *
                             std::sin(pi * gz / NZ);
        }
    s.setField("u", fu);
    s.setField("v", fv);
    s.setField("w", fw);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    const int steps = (int)envd("PROBE_STEPS", 40);
    const double wisp = envd("PROBE_WISP", 1e-8);

    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configurePacking(sd, ox, oy, oz, lnx, lny, lnz);

    IbmSolver* ref = nullptr;
    if (rank == 0) {
      ref = new IbmSolver(NX, NY, NZ);
      configurePacking(*ref, 0, 0, 0, NX, NY, NZ, std::getenv("PROBE_ULP") != nullptr);
      std::printf("PROBE packing np=%d steps=%d wisp=%.3e rtol=%s solid=%s\n", size, steps, wisp,
                  std::getenv("PROBE_RTOL") ? std::getenv("PROBE_RTOL") : "default",
                  std::getenv("PROBE_NOSOLID") ? "off" : "on");
      std::printf("%5s %11s %-16s %10s %11s %11s %11s %7s\n", "step", "dc", "argmax(x,y,z)",
                  "n>1e-11", "C_np", "C_ref", "dc|mixed", "its d/r");
    }
    if (rank == 0) {
      std::printf("fields:");
      for (const auto& n : sd.fieldNames())
        std::printf(" %s", n.c_str());
      std::printf("\n");
    }
    const char* flds[4] = {"u", "v", "w", "p"};
    const bool kin = std::getenv("PROBE_KINEMATIC") != nullptr;
    for (int i = 0; i < steps; ++i) {
      if (kin)
        sd.advectVof();
      else
        sd.step();
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      std::vector<std::vector<double>> gf;
      for (int f = 0; f < 4; ++f)
        gf.push_back(gatherGlobal(sd.getField(flds[f]), ox, oy, oz, lnx, lny, lnz, rank, size));
      long itd = sd.lastPressureIterations();
      const double divd = sd.maxOpenDivergence();
      if (rank == 0) {
        if (kin)
          ref->advectVof();
        else
          ref->step();
        const std::vector<double> rc = ref->getVof();
        double dfld[4] = {0, 0, 0, 0}, mfld[4] = {0, 0, 0, 0};
        for (int f = 0; f < 4; ++f) {
          const std::vector<double> rf = ref->getField(flds[f]);
          for (std::size_t k = 0; k < GCELLS; ++k) {
            dfld[f] = std::fmax(dfld[f], std::fabs(gf[f][k] - rf[k]));
            mfld[f] = std::fmax(mfld[f], std::fabs(rf[k]));
          }
        }
        long nflip05 = 0;
        double nearest05 = 1.0;
        for (std::size_t k = 0; k < GCELLS; ++k) {
          if ((gc[k] > 0.5) != (rc[k] > 0.5))
            ++nflip05;
          nearest05 = std::fmin(nearest05, std::fabs(rc[k] - 0.5));
        }
        double dc = 0;
        std::size_t am = 0;
        long nbad = 0, nflip = 0;
        for (std::size_t k = 0; k < GCELLS; ++k) {
          const double a = gc[k], b = rc[k];
          const double d = std::fabs(a - b);
          if (d > dc) {
            dc = d;
            am = k;
          }
          if (d > 1e-11)
            ++nbad;
          const bool ma = (a > wisp && a < 1.0 - wisp), mb = (b > wisp && b < 1.0 - wisp);
          if (ma != mb)
            ++nflip;
        }
        const int ax = (int)(am % NX), ay = (int)((am / NX) % NY), az = (int)(am / (NX * NY));
        std::printf(
            "%4d dc %10.3e @(%2d,%2d,%2d) C %.5f n>tol %3ld | du %8.2e dv %8.2e dw %8.2e "
            "dp %8.2e (|p| %8.2e) | flip05 %ld |C-.5|min %8.2e wispflip %ld | its %ld/%ld "
            "div %8.2e\n",
            i, dc, ax, ay, az, rc[am], nbad, dfld[0], dfld[1], dfld[2], dfld[3], mfld[3], nflip05,
            nearest05, nflip, itd, ref->lastPressureIterations(), divd);
        // PADDED comparison, ghosts included: rank 0's block starts at the global origin, so its
        // padded (i,j,k) matches the reference's padded (i,j,k) wherever both exist. This is the
        // only way to see a GHOST that differs -- getField() returns inner cells only.
        if (std::getenv("PROBE_PAD")) {
          const auto sh = sd.blockShape();
          const auto rh = ref->blockShape();
          const char* pf[5] = {"u", "v", "w", "p", "C"};
          for (int f = 0; f < 5; ++f) {
            auto vd = sd.fieldView(pf[f]);
            auto vr = ref->fieldView(pf[f]);
            auto hd = Kokkos::create_mirror_view(vd);
            Kokkos::deep_copy(hd, vd);
            auto hr2 = Kokkos::create_mirror_view(vr);
            Kokkos::deep_copy(hr2, vr);
            double md = 0;
            int bi = -1, bj = -1, bk = -1;
            bool bg = false;
            const int G2 = sd.ghostWidth();
            for (int k = 0; k < std::min(sh[2], rh[2]); ++k)
              for (int j = 0; j < std::min(sh[1], rh[1]); ++j)
                for (int ii = 0; ii < std::min(sh[0], rh[0]); ++ii) {
                  const double a =
                      hd((std::size_t)ii + (std::size_t)j * sh[0] + (std::size_t)k * sh[0] * sh[1]);
                  const double b = hr2((std::size_t)ii + (std::size_t)j * rh[0] +
                                       (std::size_t)k * rh[0] * rh[1]);
                  const double d = std::fabs(a - b);
                  if (d > md) {
                    md = d;
                    bi = ii - G2;
                    bj = j - G2;
                    bk = k - G2;
                    bg = (ii < G2 || j < G2 || k < G2 || ii >= sh[0] - G2 || j >= sh[1] - G2 ||
                          k >= sh[2] - G2);
                  }
                }
            std::printf("      pad %s max %.3e @(%3d,%3d,%3d) %s\n", pf[f], md, bi, bj, bk,
                        bg ? "GHOST" : "inner");
          }
        }
        if (std::getenv("PROBE_PADCELL")) {
          const auto sh = sd.blockShape();
          const auto rh = ref->blockShape();
          const int G2 = sd.ghostWidth();
          const char* pf[4] = {"u", "v", "w", "C"};
          for (int f = 0; f < 4; ++f) {
            auto vd = sd.fieldView(pf[f]);
            auto vr = ref->fieldView(pf[f]);
            auto hd = Kokkos::create_mirror_view(vd);
            Kokkos::deep_copy(hd, vd);
            auto h2 = Kokkos::create_mirror_view(vr);
            Kokkos::deep_copy(h2, vr);
            for (int z = -G2; z <= 1; ++z)
              for (int y = -G2; y <= 1; ++y)
                for (int x = 13; x < 13 + 3 + G2; ++x) {
                  const int i = x + G2, j = y + G2, k = z + G2;
                  if (i >= sh[0] || j >= sh[1] || k >= sh[2])
                    continue;
                  const double a =
                      hd((std::size_t)i + (std::size_t)j * sh[0] + (std::size_t)k * sh[0] * sh[1]);
                  const double b =
                      h2((std::size_t)i + (std::size_t)j * rh[0] + (std::size_t)k * rh[0] * rh[1]);
                  if (a != b)
                    std::printf("      DIFF %s (%3d,%3d,%3d) np %.17g ref %.17g d %.3e\n", pf[f], x,
                                y, z, a, b, a - b);
                }
          }
        }
        if (std::getenv("PROBE_CELL")) {
          const char* nm[3] = {"u", "v", "w"};
          std::vector<std::vector<double>> rr;
          for (int f = 0; f < 3; ++f)
            rr.push_back(ref->getField(nm[f]));
          for (int z = 0; z <= 1; ++z)
            for (int y = 0; y <= 1; ++y)
              for (int x = 13; x <= 15; ++x) {
                const std::size_t k =
                    (std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY;
                std::printf("      (%2d,%2d,%2d)", x, y, z);
                for (int f = 0; f < 3; ++f)
                  std::printf("  %s np %.17g ref %.17g", nm[f], gf[f][k], rr[f][k]);
                std::printf("\n");
              }
        }
        if (std::getenv("PROBE_LIST")) {
          long shown = 0;
          for (std::size_t k = 0; k < GCELLS && shown < 24; ++k)
            if (std::fabs(gc[k] - rc[k]) > 1e-11) {
              std::printf("      (%2d,%2d,%2d) np %.17g ref %.17g  d %.3e\n", (int)(k % NX),
                          (int)((k / NX) % NY), (int)(k / (NX * NY)), gc[k], rc[k], gc[k] - rc[k]);
              ++shown;
            }
        }
      }
    }
    delete ref;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
