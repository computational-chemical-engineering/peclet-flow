// flow — dynamic load balancing on a VoF run: `rebalance_by_weights` / `Solver::redistribute` with
// a partition that actually MOVES.
//
// The defect this gates (found by WO-V9 item 4, fixed on branch `vof-rebalance`): everything the
// solver grows AFTER construction — the FieldSet's own storage (`add`: the VoF colour mirror "C",
// "kappa"/"kappa_branch", every closure target, every transported scalar), the member handles that
// ALIAS those records, the lazily-allocated per-block scratch, and the block container's per-rank
// box table — did not follow `allocateBlock` onto the new block. `redistribute` then memcpy'd the
// NEW padded extent into the OLD allocation: an AddressSanitizer heap-buffer-overflow (a
// 184320-byte write into a 115328-byte region) surfacing as `free(): invalid pointer` /
// `malloc(): unsorted double linked list corrupted` on whichever ranks GREW.
//
// What is gated here, in order of what each can only catch:
//
//   R1 THE MOVE IS EXACT DATA MOVEMENT. The global C, u, v, w, P and the transported scalar T are
//      gathered immediately before and immediately after the rebalance and must be BITWISE equal
//      (`!=`, not a tolerance) at every np — `redistributeGridFields` is pure data movement and
//      step 5's `setSolid` re-derives the geometry from the migrated SDF, so nothing about the
//      state may change. A stale view surviving the move shows up here as a differing cell long
//      before it corrupts the heap. The test FAILS if the partition did not actually move, so the
//      gate can never pass vacuously.
//
//   R2 STEPPING AFTER THE MOVE IS STILL THE SAME SOLVER. The run continues to STEPS on the new
//      partition and is compared with a full-grid single-rank reference that never redistributed.
//      np = 1 is bitwise (one rank owns everything either way); np > 1 carries the pressure
//      driver's documented allreduce reduction-order floor.
//
//   R3 THE BLOCK CONTAINER'S EXCHANGE TABLE FOLLOWS THE NEW DECOMPOSITION. Configuration 1 turns
//      on the per-bubble block container (rung W0). Its gather/scatter pieces are cut against the
//      owned box of EVERY rank; those boxes come from the `BlockDecomposer`, so a partition that
//      moves invalidates them. Gated by the same R1/R2 pair with the union colour and the
//      per-marker volumes carried through the move.
//
// Grid 24x24x48, two bubbles stacked along the long z axis (the aligned ORB's cut axis), all
// fluid — the block container is all-fluid at rung W0.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;

static constexpr int NX = 24, NY = 24, NZ = 48;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
static constexpr double RHO_L = 10.0, RHO_G = 1.0, MU_L = 0.5, MU_G = 0.05, SIGMA = 4.0;
static constexpr double RB = 4.5, GRAV = 2.0e-3;
static const double CX[2] = {NX / 2.0 + 0.13, NX / 2.0 + 0.13};
static const double CY[2] = {NY / 2.0 - 0.21, NY / 2.0 - 0.21};
static const double CZ[2] = {14.0 + 0.31, 30.0 + 0.31};
static const int ZSPLIT = 22;  // clips the seed boxes so each marker gets exactly one sphere

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

// The whole feature surface the defect lived on: VoF ("C" + the curvature mirrors), three property
// closures (registry targets "rho"/"mu"/"force_z" resolved by NAME), a transported scalar (its own
// registry buffer + its own per-block operator), and optionally the block container.
static void configure(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz,
                      bool blocks) {
  s.setRho(RHO_L);
  s.setMu(MU_L);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);  // walls +-z (the CUT axis)
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  // Pure RB-GS / Chebyshev pressure: decomposition-agnostic, so a WEIGHTED partition is admissible
  // (the geometric coarse levels assume clean coarsening — see test_redistribute_mpi).
  s.setPressureLevels(1);
  s.setPressureChebyshev(true, 400, 1e-14);
  s.enableVof();
  std::vector<double> c((std::size_t)lnx * lny * lnz, 0.0), t(c.size(), 0.0);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        const double v = std::fmax(bubbleAt(0, x + ox, y + oy, z + oz),
                                   bubbleAt(1, x + ox, y + oy, z + oz));
        const std::size_t i = (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
        c[i] = v;
        t[i] = 0.5 + 0.25 * std::sin(0.3 * (x + ox)) * std::cos(0.2 * (z + oz));
      }
  s.setVof(c);
  s.setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {RHO_L, RHO_G - RHO_L});
  s.setPropertyModel("mu", ClosureKind::LinearMix, "C", "", {MU_L, MU_G - MU_L});
  s.setPropertyModel("force_z", ClosureKind::LinearMix, "C", "",
                     {-RHO_L * GRAV, -(RHO_G - RHO_L) * GRAV});
  s.setSurfaceTension(SIGMA);
  s.addScalar("T", 0.02, 1, 30);
  s.setField("T", t);
  if (blocks) {
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
  }
  s.setDt(0.25 * s.capillaryDt());
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

// NaN-propagating (WO-R2): `std::fmax(m, NaN) == m`, so the obvious loop reports 0 for a field
// that has gone entirely NaN.
static double maxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d <= m))
      m = d;
  }
  return m;
}
static long countBitDiff(const std::vector<double>& a, const std::vector<double>& b) {
  long n = 0;
  for (std::size_t i = 0; i < b.size(); ++i)
    if (a[i] != b[i])
      ++n;
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
    const int STEPS = (argc > 1) ? std::atoi(argv[1]) : 12;
    const int MOVE_AT = STEPS / 2;

    // D1 = the solver's own default partition; D2 = the WEIGHTED ORB the rebalance produces.
    // The weight field is the interface-weighted one WO-V9 item 4 uses (1 + W*[mixed]), condensed
    // to "1 + W where the two bubbles are" so the test does not depend on the transported state.
    auto D1 = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    std::vector<peclet::core::Real> w(GCELLS, 1.0);
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
          if (std::fmax(bubbleAt(0, x, y, z), bubbleAt(1, x, y, z)) > 1e-8)
            w[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = 24.0;
    BlockDecomposer<3> D2((std::size_t)size, IVec<3>{NX, NY, NZ}, w);

    auto blk = D1.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    // The gate is vacuous unless the partition MOVES: the defect only fires on a rank whose block
    // changed size (a growing rank overflows its old allocation).
    int moved = 0;
    for (int r = 0; r < size; ++r) {
      const auto a = D1.block(r), b = D2.block(r);
      for (int d = 0; d < 3; ++d)
        if (a.origin[d] != b.origin[d] || a.size[d] != b.size[d])
          moved = 1;
    }
    if (rank == 0) {
      std::printf("VOF REDISTRIBUTE MPI np=%d  grid %dx%dx%d  block %dx%dx%d  partition moves: %s\n",
                  size, NX, NY, NZ, lnx, lny, lnz, moved ? "YES" : "no");
      for (int r = 0; r < size && size > 1; ++r) {
        const auto a = D1.block(r), b = D2.block(r);
        std::printf("    rank %d: %ldx%ldx%ld @ (%ld,%ld,%ld)  ->  %ldx%ldx%ld @ (%ld,%ld,%ld)\n", r,
                    (long)a.size[0], (long)a.size[1], (long)a.size[2], (long)a.origin[0],
                    (long)a.origin[1], (long)a.origin[2], (long)b.size[0], (long)b.size[1],
                    (long)b.size[2], (long)b.origin[0], (long)b.origin[1], (long)b.origin[2]);
      }
    }
    if (size > 1 && !moved) {
      if (rank == 0)
        std::printf("  FAIL - the weighted ORB coincides with the default one; nothing is gated\n");
      fail = 1;
    }

    for (int cfg = 0; cfg < 2; ++cfg) {
      const bool blocks = (cfg == 1);
      if (rank == 0)
        std::printf("  --- configuration: VoF + closures + scalar%s\n",
                    blocks ? " + the BLOCK container" : "");
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(D1, MPI_COMM_WORLD);
      configure(sd, ox, oy, oz, lnx, lny, lnz, blocks);
      for (int k = 0; k < MOVE_AT; ++k)
        sd.step();

      // ---- R1: the move is exact data movement -------------------------------------------
      const char* fn[6] = {"C", "u", "v", "w", "P", "T"};
      std::vector<double> pre[6], post[6];
      auto snap = [&](std::vector<double>* out, int px, int py, int pz, int qx, int qy, int qz) {
        out[0] = gatherGlobal(sd.getVof(), px, py, pz, qx, qy, qz, rank, size);
        for (int c = 0; c < 3; ++c)
          out[1 + c] = gatherGlobal(sd.getVelocity(c), px, py, pz, qx, qy, qz, rank, size);
        out[4] = gatherGlobal(sd.getPressure(), px, py, pz, qx, qy, qz, rank, size);
        out[5] = gatherGlobal(sd.getField("T"), px, py, pz, qx, qy, qz, rank, size);
      };
      snap(pre, ox, oy, oz, lnx, lny, lnz);
      std::vector<double> volPre;
      if (blocks) {
        const auto st = sd.vofBlockStats();
        std::vector<double> l(st.size(), 0.0);
        volPre.assign(st.size(), 0.0);
        for (std::size_t i = 0; i < st.size(); ++i)
          l[i] = st[i].volume;
        MPI_Allreduce(l.data(), volPre.data(), (int)l.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      }

      sd.rebalanceByWeights(w);  // <-- the call that corrupted the heap

      const auto nb = D2.block(rank);
      const int nox = (int)nb.origin[0], noy = (int)nb.origin[1], noz = (int)nb.origin[2];
      const int nlx = (int)nb.size[0], nly = (int)nb.size[1], nlz = (int)nb.size[2];
      snap(post, nox, noy, noz, nlx, nly, nlz);
      if (rank == 0) {
        std::printf("    R1 across the move (must be bitwise):");
        for (int i = 0; i < 6; ++i) {
          const long nd = countBitDiff(pre[i], post[i]);
          std::printf("  %s %ld cells / %.2e", fn[i], nd, maxDiff(pre[i], post[i]));
          if (nd != 0)
            fail = 1;
        }
        std::printf("\n");
      }
      if (blocks) {
        const auto st = sd.vofBlockStats();
        std::vector<double> l(st.size(), 0.0), volPost(st.size(), 0.0);
        for (std::size_t i = 0; i < st.size(); ++i)
          l[i] = st[i].volume;
        MPI_Allreduce(l.data(), volPost.data(), (int)l.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
          double dm = 0.0;
          for (std::size_t i = 0; i < volPost.size(); ++i)
            dm = std::fmax(dm, std::fabs(volPost[i] - volPre[i]));
          std::printf("    R3 marker volumes across the move: %zu markers, max|dV| = %.3e\n",
                      volPost.size(), dm);
          if (!(dm <= 1e-12 * std::fmax(1.0, volPre.empty() ? 1.0 : volPre[0])))
            fail = 1;
        }
      }

      // ---- R2: the run continues on the new partition ------------------------------------
      for (int k = MOVE_AT; k < STEPS; ++k)
        sd.step();
      std::vector<double> g[6];
      snap(g, nox, noy, noz, nlx, nly, nlz);

      // The CONTROL: the same distributed run that never rebalanced. It is what calibrates R2 —
      // a fixed-sweep-count RB-GS (the scalar's implicit diffusion) is decomposition-dependent by
      // construction, so an absolute tolerance here would either be vacuous or would measure the
      // decomposition and blame the redistribute. Measured on this scene at np = 2: the scalar's
      // own decomposition floor is 4.17e-05 with no rebalance at all and 4.20e-05 with one.
      IbmSolver sc(lnx, lny, lnz);
      sc.initMpi(D1, MPI_COMM_WORLD);
      configure(sc, ox, oy, oz, lnx, lny, lnz, blocks);
      for (int k = 0; k < STEPS; ++k)
        sc.step();
      std::vector<double> gc[6];
      gc[0] = gatherGlobal(sc.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      for (int c = 0; c < 3; ++c)
        gc[1 + c] = gatherGlobal(sc.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);
      gc[4] = gatherGlobal(sc.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      gc[5] = gatherGlobal(sc.getField("T"), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, 0, 0, 0, NX, NY, NZ, blocks);
        for (int k = 0; k < STEPS; ++k)
          ref.step();
        std::vector<double> r[6] = {ref.getVof(),      ref.getVelocity(0), ref.getVelocity(1),
                                    ref.getVelocity(2), ref.getPressure(), ref.getField("T")};
        std::printf("    R2 vs the single-rank reference after %d steps"
                    "  [rebalanced | never-rebalanced control]:\n     ",
                    STEPS);
        for (int i = 0; i < 6; ++i) {
          const double d = maxDiff(g[i], r[i]), dc = maxDiff(gc[i], r[i]);
          // np = 1: the control IS the reference (one rank owns everything), so `bound` collapses
          // to 1e-13 and the gate is bit-exactness. np > 1: the rebalanced run must sit at the
          // control's own floor, not merely under some absolute number.
          const double bound = std::fmax(1e-13, 4.0 * dc);
          std::printf("  d%s %.2e|%.2e", fn[i], d, dc);
          if (!(d <= bound))
            fail = 1;
        }
        std::printf("\n");
      }
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    std::printf(totalFail == 0
                    ? "OK (np=%d): a VoF (+block) run survives a MOVING rebalance bit-exactly\n"
                    : "FAILED (np=%d)\n",
                size);
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
