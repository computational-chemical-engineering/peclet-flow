// flow — doc/vof_overlap_design.md gate G5: the block container's DEBRIS removal, distributed.
//
// The G1 scene (two R = 5 markers 16 cells apart along x, plus a speck of B's colour painted into
// A's interface band -- three cells with no cell of B above 1/2 within two cells) under a linear
// shear u = gamma (y - y_c), advected 20 steps by the block container with the block CSF on (so
// the debris pass runs). The speck is removed at the first step and its volume returned to B's
// attached interface; the removal's sums are sequential in index order on the master, so the
// result must be BITWISE independent of the decomposition.
//
// Compared bitwise against a single-rank run of the same scene (rank 0): the union colour, every
// marker's volume, and the seven ledger fields (with residueReturned). At np >= 2 the ORB cuts x
// and the masters are round-robin, so A and B live on different ranks.
//
// Second scene (review finding 3c): an OVERLAPPING pair with variable density, split between ranks
// -- the phantom capillary bound activates, cuts the capillary dt by exactly
// sqrt(2 rho_min / (rho_min + rho_max)) against the set_vof_phantom_capillary_bound(False) run, and
// max S, the overlap cell count, the flag and the capillary dt are identical to np = 1.
//
// Third scene (vof_overlap_design §15): a marker CUT BY THE LOW WALL y = 0 (walls +-y), colour in
// the wall-adjacent layer, in a solenoidal field with v = 0 exactly on the wall face and v != 0 one
// face in. The block's wall-face velocity lives at global index -1, outside the domain, and is
// gathered from the wall rank's patch ghosts. np 2 cuts x through the marker and its master is
// rank 1 (round robin, marker id 1), so the out-of-domain pieces travel by MPI; np 4 also cuts y
// through its box, so ranks that own NO wall face take part. Per-marker volume <= 1e-12 and union
// colour + volumes bitwise np = 1. (With the pre-§15 zero-gradient clamp: volume drift ~1e-3.)
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 48, NY = 32, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
static constexpr double R = 5.0, SIGMA = 320.0, AX = 16.3, BX = 32.3, CY = 16.2, CZ = 16.1;
static constexpr int SX = 21, SY0 = 15, SZ = 16;
static constexpr double SPECK[3] = {1e-5, 1e-2, 3e-3};
static constexpr double GAMMA = 0.25, DT = 0.05;
static const std::array<int, 6> BOX[2] = {{7, 7, 7, 26, 26, 26}, {18, 7, 7, 41, 26, 26}};

static double sphereFrac(double cx, int x, int y, int z) {
  const int NS = 8;
  int in = 0;
  for (int k = 0; k < NS; ++k)
    for (int j = 0; j < NS; ++j)
      for (int i = 0; i < NS; ++i) {
        const double px = x + (i + 0.5) / NS - cx, py = y + (j + 0.5) / NS - CY,
                     pz = z + (k + 0.5) / NS - CZ;
        in += (px * px + py * py + pz * pz < R * R) ? 1 : 0;
      }
  return static_cast<double>(in) / (NS * NS * NS);
}

static std::vector<double> colour(int m) {
  const auto& b = BOX[m];
  std::vector<double> c;
  for (int z = b[2]; z < b[5]; ++z)
    for (int y = b[1]; y < b[4]; ++y)
      for (int x = b[0]; x < b[3]; ++x) {
        double v = sphereFrac(m == 0 ? AX : BX, x, y, z);
        if (m == 1 && x == SX && z == SZ && y >= SY0 && y < SY0 + 3)
          v = SPECK[y - SY0];
        c.push_back(v);
      }
  return c;
}

static void configure(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  const std::size_t loc = (std::size_t)lnx * lny * lnz;
  s.setRho(1.0);
  s.setMu(0.5);
  s.setPressureGeometry(std::vector<double>(loc, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>(loc, 0.0));
  s.setSurfaceTension(SIGMA);
  s.enableVofBlocksFromColours({BOX[0], BOX[1]}, {colour(0), colour(1)});
  s.enableVofBlockCsf();  // the debris pass runs under the block CSF (removal default ON)
  // u = gamma (y - y_c) at the u faces (x + 1/2, y, z): divergence-free, periodic in x.
  std::vector<double> u(loc, 0.0), zero(loc, 0.0);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        u[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            GAMMA * ((y + oy) + 0.5 - CY);
  (void)ox;
  (void)oz;
  s.setVelocity(0, u);
  s.setVelocity(1, zero);
  s.setVelocity(2, zero);
}

// ---- the PHANTOM capillary bound (vof_overlap_design §5.6/§11.3; review finding 3c): two
// markers 8 cells apart (R = 5: 2 cells of overlap), each carrying only its own sphere, with a
// variable density so the bound has a density range to act on. The x cut of np 2 / 4 falls
// between the two centres, so the S = sum_k C_k census is assembled from both masters.
static constexpr double PAX = 18.3, PBX = 26.3, RHO_L = 1.0, RHO_G = 0.1;
static const std::array<int, 6> PBOX[2] = {{9, 7, 7, 28, 26, 26}, {17, 7, 7, 36, 26, 26}};
static std::vector<double> pairColour(int m) {
  const auto& b = PBOX[m];
  std::vector<double> c;
  for (int z = b[2]; z < b[5]; ++z)
    for (int y = b[1]; y < b[4]; ++y)
      for (int x = b[0]; x < b[3]; ++x)
        c.push_back(sphereFrac(m == 0 ? PAX : PBX, x, y, z));
  return c;
}
static void configurePair(IbmSolver& s, int lnx, int lny, int lnz, bool bound) {
  const std::size_t loc = (std::size_t)lnx * lny * lnz;
  s.setRho(RHO_L);
  s.setMu(0.5);
  s.setPressureGeometry(std::vector<double>(loc, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>(loc, 0.0));
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {RHO_L, RHO_G - RHO_L});
  s.setSurfaceTension(SIGMA);
  if (!bound)
    s.setVofPhantomCapillaryBound(false);
  s.enableVofBlocksFromColours({PBOX[0], PBOX[1]}, {pairColour(0), pairColour(1)});
  s.enableVofBlockCsf();  // forms S and the census for the first step
}

// ---- a marker at the LOW wall y = 0 (§15)
static constexpr double WR[2] = {5.0, 9.0}, WC[2][3] = {{40.3, 22.2, 16.1}, {20.3, 3.2, 16.1}};
static constexpr double WA = 0.5, WDT = 0.5;
static constexpr int WSTEPS = 40;
static double sphereFrac3(const double* c, double r, int x, int y, int z) {
  const int NS = 8;
  int in = 0;
  for (int k = 0; k < NS; ++k)
    for (int j = 0; j < NS; ++j)
      for (int i = 0; i < NS; ++i) {
        const double px = x + (i + 0.5) / NS - c[0], py = y + (j + 0.5) / NS - c[1],
                     pz = z + (k + 0.5) / NS - c[2];
        in += (px * px + py * py + pz * pz < r * r) ? 1 : 0;
      }
  return static_cast<double>(in) / (NS * NS * NS);
}
static std::array<int, 6> wallBox(int m) {  // bubble box + the margin 3, inside the y walls
  std::array<int, 6> q;
  for (int d = 0; d < 3; ++d) {
    q[d] = (int)std::floor(WC[m][d] - WR[m]) - 3;
    q[3 + d] = (int)std::ceil(WC[m][d] + WR[m]) + 3;
  }
  q[1] = std::max(q[1], 0);
  q[4] = std::min(q[4], NY);
  return q;
}
static std::vector<double> wallColour(int m) {
  const auto b = wallBox(m);
  std::vector<double> c;
  for (int z = b[2]; z < b[5]; ++z)
    for (int y = b[1]; y < b[4]; ++y)
      for (int x = b[0]; x < b[3]; ++x)
        c.push_back(sphereFrac3(WC[m], WR[m], x, y, z));
  return c;
}
/// psi on the z-edge (x, y): zero on both walls, periodic in x with an EXACT wrap (x mod NX).
static double wallPsi(int x, int y) {
  const double PI = 3.14159265358979323846;
  const int xm = ((x % NX) + NX) % NX;
  return WA * (NX / (2.0 * PI)) * std::sin(2.0 * PI * xm / NX) * 4.0 * y * (NY - y) /
         (double)(NY * NY);
}
static void configureWall(IbmSolver& s, int ox, int oy, int lnx, int lny, int lnz) {
  const std::size_t loc = (std::size_t)lnx * lny * lnz;
  s.setRho(1.0);
  s.setMu(0.5);
  s.setDomainBc(2, 1, 0, 0, 0);
  s.setDomainBc(3, 1, 0, 0, 0);  // walls +-y
  s.setPressureGeometry(std::vector<double>(loc, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>(loc, 0.0));
  s.setSurfaceTension(SIGMA);
  s.enableVofBlocksFromColours({wallBox(0), wallBox(1)}, {wallColour(0), wallColour(1)});
  s.enableVofBlockCsf();  // the production configuration: debris + residue pass on
  // flow's LOW-face convention: u(i, j) on the x face i between z-edges (i, j) and (i, j + 1),
  // v(i, j) on the y face j between (i, j) and (i + 1, j) -- the discrete curl, so the discrete
  // divergence telescopes; v(., 0) = 0 exactly (psi(., 0) = 0).
  std::vector<double> u(loc, 0.0), v(loc, 0.0), w(loc, 0.0);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        const int gx = x + ox, gy = y + oy;
        const std::size_t i = (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
        u[i] = wallPsi(gx, gy + 1) - wallPsi(gx, gy);
        v[i] = -(wallPsi(gx + 1, gy) - wallPsi(gx, gy));
      }
  s.setVelocity(0, u);
  s.setVelocity(1, v);
  s.setVelocity(2, w);
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

// per marker: volume, debrisCells, debrisVolume, debrisReturned, debrisLost, debrisUnresolved,
// residueReturned (§13)
static constexpr int NF = 7;
static std::vector<double> ledger(IbmSolver& s) {
  std::vector<double> v;
  for (const auto& q : s.vofBlockStats()) {
    v.push_back(q.volume);
    v.push_back(static_cast<double>(q.debrisCells));
    v.push_back(q.debrisVolume);
    v.push_back(q.debrisReturned);
    v.push_back(q.debrisLost);
    v.push_back(static_cast<double>(q.debrisUnresolved));
    v.push_back(q.residueReturned);
  }
  return v;
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
    bool cutX = false;
    for (const auto& sz : dec.sizes())
      if ((int)sz[0] != NX)
        cutX = true;
    if (rank == 0)
      std::printf(
          "VOF BLOCK DEBRIS MPI (vof_overlap_design G5) np=%d  grid %dx%dx%d  block "
          "%dx%dx%d  x cut: %s\n",
          size, NX, NY, NZ, lnx, lny, lnz, cutX ? "yes" : "no");
    if (size > 1 && !cutX) {
      if (rank == 0)
        std::printf("  FAIL - the decomposition does not cut x, so the markers share a rank\n");
      fail = 1;
    }
    const int STEPS = (argc > 1) ? std::atoi(argv[1]) : 20;
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configure(sd, ox, oy, oz, lnx, lny, lnz);
    for (int k = 0; k < STEPS; ++k)
      sd.advectVofBlocks(DT, /*requireSolenoidal=*/false);
    // the measured entries live on each block's master (zero elsewhere): a SUM is the census,
    // and x + 0 is exact, so the sum carries the master's double bit for bit
    std::vector<double> lLoc = ledger(sd), lAll(lLoc.size(), 0.0);
    MPI_Allreduce(lLoc.data(), lAll.data(), (int)lLoc.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const std::vector<double> gC = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
    if (rank == 0) {
      IbmSolver ref(NX, NY, NZ);
      configure(ref, 0, 0, 0, NX, NY, NZ);
      const std::vector<double> r0 = ledger(ref);
      for (int k = 0; k < STEPS; ++k)
        ref.advectVofBlocks(DT, false);
      const std::vector<double> rl = ledger(ref), rC = ref.getVof();
      long nC = 0;
      for (std::size_t i = 0; i < GCELLS; ++i)
        nC += (gC[i] == rC[i]) ? 0 : 1;
      long nL = 0;
      for (std::size_t i = 0; i < rl.size(); ++i)
        nL += (lAll[i] == rl[i]) ? 0 : 1;
      const char* fn[NF] = {"volume", "cells", "vol", "returned", "lost", "unresolved", "residue"};
      for (std::size_t m = 0; m < rl.size() / NF; ++m) {
        std::printf("  marker %zu:", m);
        for (int f = 0; f < NF; ++f)
          std::printf("  %s %.17g%s", fn[f], lAll[m * NF + f],
                      lAll[m * NF + f] == rl[m * NF + f] ? "" : " (!= np1)");
        std::printf("\n");
      }
      const double painted = SPECK[0] + SPECK[1] + SPECK[2];
      std::printf(
          "  after %d steps: union colour cells differing from np=1: %ld; ledger fields "
          "differing: %ld; B returned %.17g (painted %.17g)\n",
          STEPS, nC, nL, rl[NF + 3], painted);
      double dv = 0.0;
      for (std::size_t m = 0; m < rl.size() / NF; ++m)
        dv = std::fmax(dv, std::fabs(rl[m * NF] - r0[m * NF]) / r0[m * NF]);
      std::printf("  max marker volume rel change over the run (np=1): %.3e\n", dv);
      if (!(dv <= 1e-12)) {
        std::printf("  FAIL: marker volume not conserved\n");
        fail = 1;
      }
      if (nC != 0 || nL != 0) {
        std::printf("  FAIL: not bitwise decomposition-independent\n");
        fail = 1;
      }
      if (!(rl[NF + 3] > 0.9 * painted)) {  // the gate must not be vacuous: the speck was removed
        std::printf("  FAIL: the debris removal did not fire (returned %.3e)\n", rl[NF + 3]);
        fail = 1;
      }
      if (!(rl[3] == 0.0 && rl[NF + 4] == 0.0)) {
        std::printf("  FAIL: marker A removed debris, or the return lost volume\n");
        fail = 1;
      }
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- the phantom capillary bound on an OVERLAPPING pair split between ranks
    {
      IbmSolver pb(lnx, lny, lnz), pn(lnx, lny, lnz);
      pb.initMpi(dec, MPI_COMM_WORLD);
      pn.initMpi(dec, MPI_COMM_WORLD);
      configurePair(pb, lnx, lny, lnz, true);
      configurePair(pn, lnx, lny, lnz, false);
      const auto cb = pb.vofBlockOverlapCensus(), cn = pn.vofBlockOverlapCensus();
      const double dtB = pb.capillaryDt(), dtN = pn.capillaryDt();  // collective
      const double want = std::sqrt(2.0 * RHO_G / (RHO_G + RHO_L));
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configurePair(ref, NX, NY, NZ, true);
        const auto cr = ref.vofBlockOverlapCensus();
        const double dtR = ref.capillaryDt();
        std::printf(
            "  phantom bound, overlapping pair: max S %.17g, excess %.17g, cells %ld, "
            "active %d (off-switch run: active %d); capillary dt ratio %.17g (want "
            "%.17g); np=1: max S %.17g cells %ld active %d dt %.17g\n",
            cb.maxSum, cb.excess, cb.cells, (int)cb.active, (int)cn.active, dtB / dtN, want,
            cr.maxSum, cr.cells, (int)cr.active, dtR);
        if (!(cb.active && !cn.active)) {
          std::printf("  FAIL: the phantom bound did not activate (or the off switch failed)\n");
          fail = 1;
        }
        if (!(std::fabs(dtB / dtN / want - 1.0) <= 1e-14)) {
          std::printf("  FAIL: capillary dt ratio is not sqrt(2 rho_min / (rho_min + rho_max))\n");
          fail = 1;
        }
        // max S and the cell count are exact across np (a two-term sum is order-independent and
        // the reductions are MAX / integer SUM); the excess is a floating SUM over ranks
        if (!(cb.maxSum == cr.maxSum && cb.cells == cr.cells && cb.active == cr.active &&
              dtB == dtR && std::fabs(cb.excess - cr.excess) <= 1e-12 * cr.excess)) {
          std::printf("  FAIL: the overlap census / bound differs from np = 1\n");
          fail = 1;
        }
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // ---- a marker at the LOW wall (§15), split between ranks
    {
      bool cutY = false;
      for (const auto& sz : dec.sizes())
        if ((int)sz[1] != NY)
          cutY = true;
      IbmSolver sw(lnx, lny, lnz);
      sw.initMpi(dec, MPI_COMM_WORLD);
      configureWall(sw, ox, oy, lnx, lny, lnz);
      std::vector<double> v0Loc, v0All;
      for (const auto& q : sw.vofBlockStats())
        v0Loc.push_back(q.volume);
      v0All.assign(v0Loc.size(), 0.0);
      MPI_Allreduce(v0Loc.data(), v0All.data(), (int)v0Loc.size(), MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      const int wMaster = sw.vofBlockStats()[1].master;
      double worst = 0.0;
      std::vector<double> vLoc(v0Loc.size()), vAll(v0Loc.size());
      for (int k = 0; k < WSTEPS; ++k) {
        sw.advectVofBlocks(WDT, /*requireSolenoidal=*/false);
        const auto st = sw.vofBlockStats();
        for (std::size_t m = 0; m < st.size(); ++m)
          vLoc[m] = st[m].volume;
        MPI_Allreduce(vLoc.data(), vAll.data(), (int)vLoc.size(), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        for (std::size_t m = 0; m < vAll.size(); ++m)
          worst = std::fmax(worst, std::fabs(vAll[m] / v0All[m] - 1.0));
      }
      const std::vector<double> gC =
          gatherGlobal(sw.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configureWall(ref, 0, 0, NX, NY, NZ);
        for (int k = 0; k < WSTEPS; ++k)
          ref.advectVofBlocks(WDT, false);
        const std::vector<double> rC = ref.getVof();
        const auto rs = ref.vofBlockStats();
        long nC = 0, nV = 0;
        for (std::size_t i = 0; i < GCELLS; ++i)
          nC += (gC[i] == rC[i]) ? 0 : 1;
        for (std::size_t m = 0; m < rs.size(); ++m)
          nV += (vAll[m] == rs[m].volume) ? 0 : 1;
        // the triggering condition, so the gate cannot go vacuous
        const auto wb = wallBox(1);
        const std::vector<double> wc = wallColour(1);
        double layer = 0.0;
        for (int z = 0; z < wb[5] - wb[2]; ++z)
          for (int x = 0; x < wb[3] - wb[0]; ++x)
            layer += wc[(std::size_t)x + (std::size_t)z * (wb[3] - wb[0]) * (wb[4] - wb[1])];
        const double vIn = std::fabs(wallPsi(20, 1) - wallPsi(21, 1));
        std::printf(
            "  low-wall marker: box y [%d, %d), wall-layer colour %.3f cells, |v| one face in "
            "%.3e; master rank %d; y cut %s\n",
            wb[1], wb[4], layer, vIn, wMaster, cutY ? "yes" : "no");
        std::printf(
            "  after %d steps: max marker |V/V0-1| %.3e (volumes %.17g, %.17g); union cells "
            "differing from np=1: %ld; volumes differing: %ld\n",
            WSTEPS, worst, vAll[0], vAll[1], nC, nV);
        if (!(layer > 1.0 && wb[1] == 0 && vIn > 1e-3)) {
          std::printf("  FAIL: the low-wall scene does not put colour on the wall layer\n");
          fail = 1;
        }
        if (size > 1 && wMaster == 0) {
          std::printf("  FAIL: the wall marker's master is rank 0 (no MPI gather exercised)\n");
          fail = 1;
        }
        if (!(worst <= 1e-12)) {
          std::printf("  FAIL: a marker at the low wall does not conserve its volume\n");
          fail = 1;
        }
        if (nC != 0 || nV != 0) {
          std::printf("  FAIL: the low-wall scene is not bitwise decomposition-independent\n");
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
