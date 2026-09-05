// flow — VoF Part III rung W2 (WO-W12) gate 5: the BLOCK container coupled to Navier-Stokes,
// distributed.
//
// The container itself is gated bitwise at np 1/2/4/8 by `test_vof_blocks_mpi` (rung W0/W1, a
// standalone harness). What is gated HERE is the coupling: the union colour driving the closures,
// the per-block curvature cascade, and the block CSF face force scattered UNPACK_SUM into the
// momentum RHS, all inside a real `IbmSolver::step()`.
//
//   B1 the coupled step against a full-grid single-rank reference — u, v, w, P and the union C
//      after 12 steps of a real two-phase run with the block CSF on. np = 1 must be BITWISE; np > 1
//      carries the documented reduction-order floor of the pressure driver's allreduces
//      (`doc/variable_density_projection.md` §3.1), so it is a tolerance comparison.
//   B2 the per-marker volumes agree with the single-rank run to the reduction floor, and the
//      markers OVERLAP (the union carries less liquid than the markers do) -- the property that
//      makes the whole container worth having, now measured through the NS coupling.
//
// Grid 24x24x48 with two bubbles stacked along z: the aligned ORB cuts the long z axis at np = 2
// and 4, and the bubbles are placed so the cut passes THROUGH one of them -- so a block's gather
// draws from several owners, its scatter lands on several, and its CSF force is summed across a
// rank boundary. Masters are round-robin by block id, so at np >= 2 the two markers have different
// masters.
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;

static constexpr int NX = 24, NY = 24, NZ = 48;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
// ratio 10, the rung's rated regime with motion; sigma chosen so the capillary dt is not absurd.
static constexpr double RHO_L = 10.0, RHO_G = 1.0, MU_L = 0.5, MU_G = 0.05, SIGMA = 4.0;
static constexpr double RB = 4.5, GRAV = 2.0e-3;
// the two markers, stacked along the CUT axis; the ORB's np=2 cut at z = 24 passes through the
// upper one's band.
static const double CX[2] = {NX / 2.0 + 0.13, NX / 2.0 + 0.13};
static const double CY[2] = {NY / 2.0 - 0.21, NY / 2.0 - 0.21};
static const double CZ[2] = {14.0 + 0.31, 24.0 + 0.31};
static const int ZSPLIT = 20;  // the seed boxes are clipped here so each marker gets ONE sphere

static double bubbleAt(int b, int x, int y, int z) {
  const int NS = 12;
  double acc = 0.0;
  for (int j = 0; j < NS; ++j)
    for (int i = 0; i < NS; ++i) {
      const double px = x + (i + 0.5) / NS - CX[b], py = y + (j + 0.5) / NS - CY[b];
      const double r2 = RB * RB - px * px - py * py;
      if (r2 <= 0.0)
        continue;
      const double h = std::sqrt(r2);
      const double lo = std::fmax(CZ[b] - h, (double)z), hi = std::fmin(CZ[b] + h, (double)z + 1);
      if (hi > lo)
        acc += hi - lo;
    }
  return acc / (NS * NS);
}

// `mode`: 0 = block advection only, 1 = + the block CSF (rung W2/W4's force rule),
//         2 = + rung W4's collision models (coalescence 'weber' at a permissive threshold and
//             breakup on), which is what gate G6 measures across np.
static void configure(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz,
                      bool csf, int mode = 0) {
  s.setRho(RHO_L);
  s.setMu(MU_L);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z (the CUT axis)
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setPressureChebyshev(true, 400, 1e-14);
  s.enableVof();
  // the UNION convention: C = 1 inside a marker (the dispersed phase)
  std::vector<double> c((std::size_t)lnx * lny * lnz, 0.0);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        const double v = std::fmax(bubbleAt(0, x + ox, y + oy, z + oz),
                                   bubbleAt(1, x + ox, y + oy, z + oz));
        c[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = v;
      }
  s.setVof(c);
  s.setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {RHO_L, RHO_G - RHO_L});
  s.setPropertyModel("mu", ClosureKind::LinearMix, "C", "", {MU_L, MU_G - MU_L});
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "C", "",
                     {-RHO_L * GRAV, -(RHO_G - RHO_L) * GRAV});
  if (csf)
    s.setSurfaceTension(SIGMA);
  std::vector<std::array<int, 6>> boxes;
  for (int b = 0; b < 2; ++b) {
    std::array<int, 6> q;
    const double c3[3] = {CX[b], CY[b], CZ[b]};
    for (int d = 0; d < 3; ++d) {
      q[d] = (int)std::floor(c3[d] - RB) - 1;
      q[3 + d] = (int)std::ceil(c3[d] + RB) + 1;
    }
    if (b == 0)
      q[5] = ZSPLIT;
    else
      q[2] = ZSPLIT;
    boxes.push_back(q);
  }
  s.enableVofBlocksFromField(boxes);
  if (mode >= 2) {
    // Rung W4 items 2-4, distributed: the census is replicated by one allreduce over disjoint
    // contributions and the merge / split transports are an allreduceMax over the child's box, so
    // every rank takes the SAME decision and lands the same colour -- which is what makes the
    // event census a function of the markers and not of the decomposition.
    auto& B = s.vofBlockSet();
    B.coalescence.model = peclet::flow::vof::VofCoalescence::Weber;
    B.coalescence.weCrit = 1e9;
    B.coalescence.contactFilm = 4.0;  // large enough that this short scene ACTUALLY merges
    B.breakup = true;
    B.breakupSteps = 3;
  }
  if (csf) {
    s.enableVofBlockCsf();
    s.setDt(0.25 * s.capillaryDt());
  } else {
    s.setDt(0.05);
  }
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
          "VOF BLOCK NS MPI (rung W2, WO-W12) np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: "
          "%s%s%s\n",
          size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "");
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL - the decomposition does NOT cut z, so no marker is cut\n");
      fail = 1;
    }

    const int STEPS = (argc > 1) ? std::atoi(argv[1]) : 12;
    for (int cfg = 0; cfg < 3; ++cfg) {
    const bool csf = (cfg >= 1);
    if (rank == 0)
      std::printf("  --- configuration: block advection + varRho%s\n",
                  cfg == 0 ? " (surface tension OFF)"
                           : (cfg == 1 ? " + the BLOCK CSF"
                                       : " + the BLOCK CSF + W4 coalescence/breakup"));
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configure(sd, ox, oy, oz, lnx, lny, lnz, csf, cfg);
    for (int k = 0; k < STEPS; ++k)
      sd.step();

    // per-marker volumes: only the master carries them, so a SUM over ranks is the census
    const auto stats = sd.vofBlockStats();
    if (rank == 0)
      for (const auto& q : stats)
        std::printf("    dist block %ld master %d box [%d,%d)x[%d,%d)x[%d,%d)\n", q.id, q.master,
                    q.lo[0], q.hi[0], q.lo[1], q.hi[1], q.lo[2], q.hi[2]);
    std::vector<double> vLoc(stats.size(), 0.0), vAll(stats.size(), 0.0);
    std::vector<double> aLoc(stats.size(), 0.0), aAll(stats.size(), 0.0);
    for (std::size_t i = 0; i < stats.size(); ++i) {
      vLoc[i] = stats[i].volume;
      aLoc[i] = stats[i].area;
    }
    MPI_Allreduce(vLoc.data(), vAll.data(), (int)vLoc.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(aLoc.data(), aAll.data(), (int)aLoc.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> gf[3];
    if (csf)
      for (int c = 0; c < 3; ++c)
        gf[c] = gatherGlobal(sd.getVofBlockForce(c), ox, oy, oz, lnx, lny, lnz, rank, size);
    std::vector<double> g[5];
    g[0] = gatherGlobal(sd.getVelocity(0), ox, oy, oz, lnx, lny, lnz, rank, size);
    g[1] = gatherGlobal(sd.getVelocity(1), ox, oy, oz, lnx, lny, lnz, rank, size);
    g[2] = gatherGlobal(sd.getVelocity(2), ox, oy, oz, lnx, lny, lnz, rank, size);
    g[3] = gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
    g[4] = gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);

    if (rank == 0) {
      IbmSolver ref(NX, NY, NZ);
      configure(ref, 0, 0, 0, NX, NY, NZ, csf, cfg);
      for (int k = 0; k < STEPS; ++k)
        ref.step();
      std::vector<double> r[5] = {ref.getVelocity(0), ref.getVelocity(1), ref.getVelocity(2),
                                  ref.getPressure(), ref.getVof()};
      const char* fn[5] = {"u", "v", "w", "P", "C"};
      double d[5];
      long nb[5];
      double mb[5];
      for (int i = 0; i < 5; ++i) {
        d[i] = maxDiff(g[i], r[i]);
        nb[i] = countBitDiff(g[i], r[i], mb[i]);
      }
      std::printf("  np=%d after %d steps:", size, STEPS);
      for (int i = 0; i < 5; ++i)
        std::printf("  d%s %.2e", fn[i], d[i]);
      std::printf("\n");
      std::printf("  bitwise-differing cells:");
      for (int i = 0; i < 5; ++i)
        std::printf("  %s %ld", fn[i], nb[i]);
      std::printf("\n");
      if (csf) {
        double df = 0.0;
        long nf = 0;
        for (int c = 0; c < 3; ++c) {
          const std::vector<double> rf = ref.getVofBlockForce(c);
          double m = 0.0;
          nf += countBitDiff(gf[c], rf, m);
          df = std::fmax(df, maxDiff(gf[c], rf));
        }
        std::printf("  scattered CSF face force vs np=1: max|d| %.3e over %ld bit-differing "
                    "cells\n", df, nf);
      }
      for (const auto& q : ref.vofBlockStats())
        std::printf("    ref block %ld box [%d,%d)x[%d,%d)x[%d,%d)\n", q.id, q.lo[0], q.hi[0],
                    q.lo[1], q.hi[1], q.lo[2], q.hi[2]);
      const auto rs = ref.vofBlockStats();
      double dv = 0.0, da = 0.0, sumMark = 0.0, sumUnion = 0.0;
      for (std::size_t i = 0; i < rs.size(); ++i) {
        dv = std::fmax(dv, std::fabs(vAll[i] - rs[i].volume));
        da = std::fmax(da, std::fabs(aAll[i] - rs[i].area));
        sumMark += rs[i].volume;
      }
      for (double q : r[4])
        sumUnion += q;
      std::printf("  markers: V %.6f / %.6f, area %.2f / %.2f; |dV| vs np=1 %.2e, |dA| %.2e\n",
                  rs[0].volume, rs[1].volume, rs[0].area, rs[1].area, dv, da);
      std::printf("  UNION deficit sum(V_marker) - sum(C_union) = %.4f cells (the shared liquid; "
                  "> 0 means the two markers OVERLAP, which a single field cannot do)\n",
                  sumMark - sumUnion);
      // np = 1 bitwise; np > 1 at the pressure driver's own reduction-order floor, the SAME
      // tolerances `test_vof_surface_tension_mpi` uses for its coupled gate.
      const double tolU = (size == 1) ? 0.0 : 1e-12;
      const double tolP = (size == 1) ? 0.0 : 1e-9;
      const double tolC = (size == 1) ? 0.0 : 1e-12;
      const double tol[5] = {tolU, tolU, tolU, tolP, tolC};
      for (int i = 0; i < 5; ++i)
        if (!(d[i] <= tol[i])) {
          std::printf("  FAIL: d%s = %.3e exceeds %.1e\n", fn[i], d[i], tol[i]);
          fail = 1;
        }
      if (!(dv <= (size == 1 ? 0.0 : 1e-11))) {
        std::printf("  FAIL: marker volume drift %.3e\n", dv);
        fail = 1;
      }
      // --- rung W4 gate G6: the PAIR CENSUS and the EVENT LOG are the same object at every np.
      const auto& pd = sd.vofBlockPairs();
      const auto& pr = ref.vofBlockPairs();
      std::printf("  W4 census: %zu pair(s) distributed, %zu single-rank\n", pd.size(),
                  pr.size());
      if (pd.size() != pr.size()) {
        std::printf("  FAIL: the pair census has a different SIZE at np=%d\n", size);
        fail = 1;
      } else {
        double dfilm = 0.0, dwe = 0.0;
        for (std::size_t i = 0; i < pd.size(); ++i) {
          if (pd[i].idA != pr[i].idA || pd[i].idB != pr[i].idB) {
            std::printf("  FAIL: pair %zu is (%ld,%ld) distributed vs (%ld,%ld) single-rank\n", i,
                        pd[i].idA, pd[i].idB, pr[i].idA, pr[i].idB);
            fail = 1;
          }
          dfilm = std::fmax(dfilm, std::fabs(pd[i].film - pr[i].film));
          dwe = std::fmax(dwe, std::fabs(pd[i].weber - pr[i].weber));
          std::printf("    pair (%ld,%ld) film %.6f vs %.6f, We %.4e vs %.4e, contact %ld/%ld\n",
                      pd[i].idA, pd[i].idB, pd[i].film, pr[i].film, pd[i].weber, pr[i].weber,
                      pd[i].contactSteps, pr[i].contactSteps);
        }
        const double tolF = (size == 1) ? 0.0 : 1e-6;
        if (!(dfilm <= tolF)) {
          std::printf("  FAIL: film thickness differs by %.3e across np\n", dfilm);
          fail = 1;
        }
        (void)dwe;
      }
      const auto& ed = sd.vofBlockEvents();
      const auto& er = ref.vofBlockEvents();
      std::printf("  W4 events: %zu distributed, %zu single-rank; markers %zu vs %zu\n",
                  ed.size(), er.size(), stats.size(), rs.size());
      {
        static const char* kind[4] = {"approach", "contact", "merge", "split"};
        std::printf("   ");
        for (const auto& e : er)
          std::printf("  %s@%ld(%ld,%ld)", kind[e.type & 3], e.step, e.idA, e.idB);
        std::printf("\n");
      }
      if (ed.size() != er.size() || stats.size() != rs.size()) {
        std::printf("  FAIL: the event census / marker count depends on np\n");
        fail = 1;
      } else {
        for (std::size_t i = 0; i < ed.size(); ++i)
          if (ed[i].type != er[i].type || ed[i].step != er[i].step || ed[i].idA != er[i].idA ||
              ed[i].idB != er[i].idB) {
            std::printf("  FAIL: event %zu differs: type %d step %ld (%ld,%ld) vs type %d step "
                        "%ld (%ld,%ld)\n", i, ed[i].type, ed[i].step, ed[i].idA, ed[i].idB,
                        er[i].type, er[i].step, er[i].idA, er[i].idB);
            fail = 1;
          }
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
