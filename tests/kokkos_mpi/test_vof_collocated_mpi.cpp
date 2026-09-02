// flow — multi-rank validation of the COLLOCATED two-phase path (VoF rung V8, WO-T).
//
// What this gates that no existing MPI test does: the pieces rung V8 added to `SolverColocated`
// all live on the FACE field or on its ghost ring, and every one of them is assembled from
// exchanged property ghosts.
//
//   * the face acceleration `af(i) = dt (f_f - (P(i)-P(i-s)))/rho_f(i)` is built over the inner
//     region WIDENED BY ONE on the high side of every axis, so the plane at index `e-g` — the face
//     the divergence of the last inner cell reads, and the one the cell average reads as `af(i+s)`
//     — is formed on THIS rank from depth-1 ghosts of P, rho and the cell force. It is bitwise
//     decomposition-independent only if those ghosts are the owner's values and the arithmetic is
//     written in the same order on both sides (it is: the same `0.5*(rho(i)+rho(i-s))`);
//   * the cell correction averages `af(i)` and `af(i+s)` with the openness-0 rule, i.e. it reads
//     that same plane;
//   * the colour is bridged from `uf_/vf_/wf_` after `fillGhosts`, not from the cell field.
//
// Two configurations on 16x16x32, whose aligned ORB cuts the LONG z axis at np = 2 and 4 — the axis
// that carries the walls, the stratification and the interface, deliberately (WO-F: the walled axis
// must be CUT, and the density jump must sit on a rank boundary).
//
//   * `hydro-z`  — the hydrostatic acid test on the collocated grid, driven through a FROZEN colour
//     field at ratio 1000 with no-slip walls at +-z. The gated quantity on this grid is the FACE
//     field (the CELL field carries the approximate projection's invisible checkerboard, which
//     `centerToFace` annihilates — see tests/kokkos/test_vof_collocated.cpp T1b) plus the pointwise
//     agreement with the single-rank reference.
//   * `vof-z`    — a coupled run: `enable_vof` at ratio 10, a sharp interface stacked along the CUT
//     axis, a uniform body force, 20 steps. The colour is transported by the projected face field,
//     so this is the gate on the collocated bridge under a decomposition.
//
// Comparison protocol (the `tests/kokkos_mpi` pattern): the distributed run is compared POINTWISE
// against a full-grid single-rank reference built on rank 0. np = 1 is bit-exact; np > 1 lands on
// the MPI reduction-order floor (Chebyshev's bound estimation and `removeMean` both go through an
// MPI_SUM allreduce whose summation order depends on the rank count).
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

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr double GRAV = 0.1, DT = 1.0, RHO0 = 1.0;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  double ratio;
  bool walls;  // hydrostatic acid test (frozen colour) vs the coupled periodic VoF run
  int steps;
};

// C = 1 (heavy) in the lower half of z
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

static void configure(Colo& s, const Config& c, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  s.setRho(RHO0);
  s.setMu(c.walls ? 0.0 : 0.05);  // hydrostatic: inviscid, so the face balance is exact
  s.setDt(DT);
  s.setAdvection(!c.walls);
  if (c.walls) {
    s.setDomainBc(4, 1, 0, 0, 0);
    s.setDomainBc(5, 1, 0, 0, 0);
  } else {
    s.setBodyForce(1e-3, 0, 0);
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  const auto c0 = blockOf(colourAt, ox, oy, oz, lnx, lny, lnz);
  if (c.walls) {
    // FROZEN interface: "C" is a plain registered field the closure reads, enable_vof is never
    // called, so C supplies rho and nothing else (the `freeze` route of test_vof_twophase.cpp B2).
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
      std::printf("VOF COLLOCATED MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"hydro-z", 1000.0, true, 40}, {"vof-z", 10.0, false, 20}};

    for (const Config& c : configs) {
      // The stratification / interface axis must be CUT: that is the whole point (the face
      // acceleration's high-side plane and the colour bridge both live on a block boundary there).
      if (size > 1 && !cut[2]) {
        if (rank == 0)
          std::printf("  [%-7s np=%d] FAIL — the decomposition does NOT cut z\n", c.name, size);
        fail = 1;
        continue;
      }
      Colo sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      std::vector<long> itd;
      for (int it = 0; it < c.steps; ++it) {
        sd.step();
        itd.push_back(sd.lastPressureIterations());
      }
      std::vector<double> gu[3], gf[3];
      for (int comp = 0; comp < 3; ++comp) {
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
        gf[comp] = gatherGlobal(sd.getFaceVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      }
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gc =
          gatherGlobal(sd.getField("C"), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        Colo ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        std::vector<long> itr;
        for (int it = 0; it < c.steps; ++it) {
          ref.step();
          itr.push_back(ref.lastPressureIterations());
        }
        double du = 0, df = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          df = std::fmax(df, maxAbsDiff(gf[comp], ref.getFaceVelocity(comp)));
        }
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double dc = maxAbsDiff(gc, ref.getField("C"));
        const double su = maxAbs(ref.getVelocity(2)), sf = maxAbs(ref.getFaceVelocity(2)),
                     sp = maxAbs(ref.getPressure());
        long dit = 0;
        for (std::size_t k = 0; k < itd.size(); ++k)
          dit = std::max(dit, std::labs(itd[k] - itr[k]));
        // np = 1 must be BITWISE; np > 1 sits on the MPI reduction-order floor.
        const double tolU = (size == 1) ? 0.0 : 1e-11 * std::fmax(su, 1e-12);
        const double tolF = (size == 1) ? 0.0 : 1e-11 * std::fmax(sf, 1e-12);
        const double tolP = (size == 1) ? 0.0 : 1e-9 * std::fmax(sp, 1e-12);
        const double tolC = (size == 1) ? 0.0 : 1e-11;
        const bool ok = du <= tolU && df <= tolF && dp <= tolP && dc <= tolC && dit <= 1;
        std::printf(
            "  [%-7s np=%d] du %.3e (|u| %.3e)  duf %.3e (|uf| %.3e)  dP %.3e (|P| %.3e)"
            "  dC %.3e   d(iters) %ld   %s\n",
            c.name, size, du, su, df, sf, dp, sp, dc, dit, ok ? "OK" : "*** FAIL ***");
        if (c.walls)
          std::printf(
              "           face |uf| %.3e (the gated hydrostatic quantity), cell |u| %.3e "
              "(carries the approximate projection's invisible checkerboard)\n",
              maxAbs(ref.getFaceVelocity(2)), su);
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("%s\n", fail ? "FAILED" : "all collocated VoF MPI gates passed");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
