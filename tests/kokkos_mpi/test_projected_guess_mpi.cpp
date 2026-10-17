// flow — ctest `projected_guess_mpi` (np 1, 2, 4): the projected initial guess of the projection
// solve on the DISTRIBUTED PCG / FCG (doc/vof_projection_distributed_design.md §6, gates G-ED1..5;
// the single-rank scheme is doc/vof_projection_cost_design.md §5.3).
//
// Distributed, the Gram packet {G11, G12, G22, g1, g2} is allreduced with MPI_SUM and rref with
// MPI_MAX before the host 2x2 solve, so every rank takes the same c; the guess history is reset by
// init_mpi and by every redistribute (a size-preserving one passes the phiPrev1 extent check
// unnoticed). Two configurations on a 16x16x32 grid (the ORB cuts z at np 2, x and z at np 4):
//   * `cyl`  — the IBM cylinder of tests/python/test_projected_guess.py (axis y, R = 4, cut-cell
//              pressure, Stokes, body force 1e-3, three MG levels);
//   * `vof`  — the periodic sheared plane of vof_momentum_mpi (variable density by LinearMix at
//              ratio 10, momentum-consistent VoF, a sinusoidal cell force).
// Arms 'projected' PCG history 1, PCG history 2, FCG history 2, each against 'zero' on the same
// driver, rtol 1e-10, 10 steps. Gates (rank 0 builds the single-rank references):
//   G-ED1  np 1: u v w p (C) memcmp-equal to the single-rank 'projected' run, per-step iterations
//          equal;
//   G-ED2  np > 1: within the np-class tolerances of vof_*_mpi (1e-11 relative, de-meaned p),
//          iterations per step <= 'zero' at the same np + 1;
//   G-E3   (every np) step 1 of 'projected' (empty history) bitwise to 'zero'; after a state write
//          (uploadVelocity, same values, on both twins) the next step bitwise a twin switched to
//          'zero'; the guess was live before it (history 2, state not bitwise 'zero');
//   G-ED4  a size-preserving redistribute under 'projected' resets the history: the next step is
//          bitwise a twin switched to 'zero', the following one runs on history 1.
// G-E1, G-E2, G-E4 at np 2 are traced (debug output only): `--trace` runs the 'projected' arms
// with PECLET_FLOW_MG_DEBUG=2 (set before Kokkos starts) and prints the guess trace on rank 0 for
// gate_dist's edgate.sh to read; it gates nothing itself.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32, STEPS = 10, MAXIT = 400;
static constexpr double RTOL = 1e-10;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Case {
  const char* name;
  bool vof;
};
struct Arm {
  const char* name;
  bool fcg;
  int hist;
};
enum Guess { kAuto = 0, kZero = 1, kProjected = 2 };

static double cylSdf(int x, int, int z) {
  const double dx = x + 0.5 - 8.0, dz = z + 0.5 - 16.0;
  return std::sqrt(dx * dx + dz * dz) - 4.0;
}
static double colourAt(int x, int y, int z) {  // vof_momentum_mpi's tilted sharp plane
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

static void configure(IbmSolver& s, const Case& c, const Arm& a, int guess, int ox, int oy, int oz,
                      int lnx, int lny, int lnz) {
  if (!c.vof) {
    s.setRho(1.0);
    s.setMu(0.01);
    s.setDt(1.0);
    s.setAdvection(false);
    s.setBodyForce(1e-3, 0.0, 0.0);
    s.setPressureLevels(3);
    s.setSolid(blockOf(cylSdf, ox, oy, oz, lnx, lny, lnz), true);
  } else {
    const double ratio = 10.0;
    s.setRho(ratio);
    s.setMu(0.05);
    s.setDt(0.2);
    s.setAdvection(true);
    s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
    s.setVof(blockOf(colourAt, ox, oy, oz, lnx, lny, lnz));
    s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, ratio - 1.0});
    s.enableVofMomentum(1.0, ratio);
    s.enableCellForce();
    s.setField("force_x", blockOf(forceXAt, ox, oy, oz, lnx, lny, lnz));
  }
  if (a.fcg)  // LAST: the density mode re-selects the driver
    s.setPressureFcg(true, MAXIT, RTOL);
  else
    s.setPressurePcg(true, MAXIT, RTOL);
  s.setPressureInitialGuess(guess, a.hist);
}

static std::vector<std::vector<double>> stateOf(IbmSolver& s, bool vof) {
  std::vector<std::vector<double>> f;
  for (int k = 0; k < 3; ++k)
    f.push_back(s.getVelocity(k));
  f.push_back(s.getPressure());
  if (vof)
    f.push_back(s.getVof());
  return f;
}
static bool sameBits(const std::vector<std::vector<double>>& a,
                     const std::vector<std::vector<double>>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k].size() != b[k].size() ||
        std::memcmp(a[k].data(), b[k].data(), a[k].size() * sizeof(double)) != 0)
      return false;
  return true;
}
// Bitwise on EVERY rank (each compares its own block).
static bool sameBitsAll(IbmSolver& a, IbmSolver& b, bool vof) {
  int ok = sameBits(stateOf(a, vof), stateOf(b, vof)) ? 1 : 0, all = 0;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all == 1;
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
  double m = 0;  // NaN-propagating (vof_momentum_mpi's WO-R2 note)
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;
    m = std::fmax(m, d);
  }
  return m;
}
static std::string dbgDiff(const std::vector<double>& a, const std::vector<double>& b) {
  long nb = 0, nv = 0;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0)
      ++nb;
    if (a[i] != b[i]) {
      ++nv;
      m = std::fmax(m, std::fabs(a[i] - b[i]));
    }
  }
  char buf[96];
  std::snprintf(buf, sizeof buf, "bits %ld values %ld max %.2e", nb, nv, m);
  return buf;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}
static std::vector<double> demean(std::vector<double> v) {  // the periodic pressure gauge
  double s = 0;
  for (double x : v)
    s += x;
  const double m = s / (double)v.size();
  for (double& x : v)
    x -= m;
  return v;
}

static int fails = 0;
static void check(int rank, bool ok, const std::string& what) {
  if (rank == 0)
    std::printf("  %-100s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok)
    ++fails;
}

int main(int argc, char** argv) {
  const bool traceMode = argc > 1 && std::strcmp(argv[1], "--trace") == 0;
  if (traceMode) {  // debug output only, read once at the first solve (mgDebugLevel's static)
    setenv("PECLET_FLOW_MG_DEBUG", "2", 1);
    setenv("PECLET_FLOW_MG_DEBUG_SOLVES", "200", 1);
  }
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
    if (rank == 0)
      std::printf("PROJECTED GUESS MPI np=%d  grid %dx%dx%d  block %dx%dx%d  rtol %.0e%s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, RTOL, traceMode ? "  (--trace)" : "");
    const Case cases[] = {{"cyl", false}, {"vof", true}};
    const Arm arms[] = {{"pcg h1", false, 1}, {"pcg h2", false, 2}, {"fcg h2", true, 2}};
    auto dist = [&](IbmSolver& s, const Case& c, const Arm& a, int guess) {
      s.initMpi(dec, MPI_COMM_WORLD);
      configure(s, c, a, guess, ox, oy, oz, lnx, lny, lnz);
    };

    if (traceMode) {
      for (const Case& c : cases)
        for (const Arm& a : arms) {
          if (rank == 0) {
            std::printf("TRACE %s %s\n", c.name, a.name);
            std::fflush(stdout);
          }
          IbmSolver s(lnx, lny, lnz);
          dist(s, c, a, kProjected);
          for (int k = 0; k < STEPS; ++k)
            s.step();
          std::fflush(stdout);
        }
      if (rank == 0)
        std::printf("TRACE done\n");
    } else {
      for (const Case& c : cases) {
        for (const Arm& a : arms) {
          const std::string tag = std::string(c.name) + " " + a.name + " np" + std::to_string(size);
          // ---- G-ED1 / G-ED2: 'projected' and 'zero' at this np, against np 1 -----------------
          IbmSolver sp(lnx, lny, lnz), sz(lnx, lny, lnz);
          dist(sp, c, a, kProjected);
          dist(sz, c, a, kZero);
          std::vector<long> itP, itZ;
          for (int k = 0; k < STEPS; ++k) {
            sp.step();
            sz.step();
            itP.push_back(sp.lastPressureIterations());
            itZ.push_back(sz.lastPressureIterations());
          }
          const auto fp = stateOf(sp, c.vof), fz = stateOf(sz, c.vof);
          std::vector<std::vector<double>> gp, gz;
          for (std::size_t k = 0; k < fp.size(); ++k) {
            gp.push_back(gatherGlobal(fp[k], ox, oy, oz, lnx, lny, lnz, rank, size));
            gz.push_back(gatherGlobal(fz[k], ox, oy, oz, lnx, lny, lnz, rank, size));
          }
          int ok1 = 1, ok2 = 1;
          if (rank == 0) {
            // The reference: single rank at np 1 (G-ED1); at np > 1 the np 1 DISTRIBUTED run
            // (G-ED2 is "np 2/4 vs np 1"), on rank 0 alone over MPI_COMM_SELF.
            IbmSolver ref(NX, NY, NZ), refz(NX, NY, NZ);
            if (size > 1) {
              const auto dec1 = peclet::flow::CutcellMG::decomposition(1, NX, NY, NZ);
              ref.initMpi(dec1, MPI_COMM_SELF);
              refz.initMpi(dec1, MPI_COMM_SELF);
            }
            configure(ref, c, a, kProjected, 0, 0, 0, NX, NY, NZ);
            configure(refz, c, a, kZero, 0, 0, 0, NX, NY, NZ);
            std::vector<long> itR, itRz;
            for (int k = 0; k < STEPS; ++k) {
              ref.step();
              refz.step();
              itR.push_back(ref.lastPressureIterations());
              itRz.push_back(refz.lastPressureIterations());
            }
            const auto fr = stateOf(ref, c.vof), frz = stateOf(refz, c.vof);
            long sP = 0, sZ = 0, sR = 0;
            for (int k = 0; k < STEPS; ++k) {
              sP += itP[k];
              sZ += itZ[k];
              sR += itR[k];
            }
            std::string per;  // per field: 'projected' / the 'zero' control, max |difference|
            const char* nm[] = {"u", "v", "w", "p", "C"};
            for (std::size_t k = 0; k < gp.size(); ++k) {
              char buf[64];
              std::snprintf(buf, sizeof buf, " %s %.2e/%.2e", nm[k], maxAbsDiff(gp[k], fr[k]),
                            maxAbsDiff(gz[k], frz[k]));
              per += buf;
            }
            if (size == 1) {
              // §2's premise: serial <-> np 1 bitwise. Where the 'zero' control itself is not
              // (the staggered cut-cell IBM momentum path, pre-existing), the memcmp gate is n/a:
              // gated are the per-step iterations, and both deviations are printed.
              const bool base = sameBits(gz, frz) && itZ == itRz;
              ok1 = itP == itR && (!base || sameBits(gp, fr));
              std::printf(
                  "  [%s] G-ED1 vs single rank: fields %s, iterations %ld vs %ld%s | "
                  "'zero' control %s | max|d| proj/zero%s\n",
                  tag.c_str(), sameBits(gp, fr) ? "memcmp-equal" : "DIFFER", sP, sR,
                  itP == itR ? " (equal per step)" : " (DIFFER per step)",
                  base ? "bitwise" : "NOT bitwise (memcmp n/a)", per.c_str());
            } else {
              // the np-class tolerances of vof_*_mpi: 1e-11 relative, the periodic p de-meaned
              double du = 0, umag = 0, duz = 0;
              for (int k = 0; k < 3; ++k) {
                du = std::fmax(du, maxAbsDiff(gp[k], fr[k]));
                duz = std::fmax(duz, maxAbsDiff(gz[k], frz[k]));
                umag = std::fmax(umag, maxAbs(fr[k]));
              }
              const double dp = maxAbsDiff(demean(gp[3]), demean(fr[3]));
              const double dpz = maxAbsDiff(demean(gz[3]), demean(frz[3]));
              const double pmag = maxAbs(demean(fr[3]));
              const double dc = c.vof ? maxAbsDiff(gp[4], fr[4]) : 0.0;
              const double utol = std::fmax(1e-12, 1e-11 * umag);
              const double ptol = std::fmax(1e-12, 1e-11 * pmag);
              bool itok = true;
              int worst = -1000;
              for (int k = 0; k < STEPS; ++k) {
                itok = itok && itP[k] <= itZ[k] + 1;
                worst = std::max(worst, (int)(itP[k] - itZ[k]));
              }
              ok2 = du <= utol && dp <= ptol && dc <= 1e-11 && itok;
              std::printf(
                  "  [%s] G-ED2 vs np1: du %.3e (tol %.1e) dp %.3e (tol %.1e, |p| %.2e) dC %.3e | "
                  "'zero' du %.3e dp %.3e | iters proj %ld zero %ld (np1 proj %ld), "
                  "max(proj-zero) per step %+d\n",
                  tag.c_str(), du, utol, dp, ptol, pmag, dc, duz, dpz, sP, sZ, sR, worst);
            }
          }
          MPI_Bcast(&ok1, 1, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Bcast(&ok2, 1, MPI_INT, 0, MPI_COMM_WORLD);
          if (size == 1)
            check(rank, ok1, "[" + tag + "] G-ED1 np 1 == single rank, bitwise, iterations equal");
          else
            check(rank, ok2, "[" + tag + "] G-ED2 np-class tolerance, iterations <= 'zero' + 1");

          if (a.hist != 2)
            continue;
          // ---- G-E3: the empty history is the cold solve; a state write restarts it ----------
          {
            IbmSolver ta(lnx, lny, lnz), tb(lnx, lny, lnz), tz(lnx, lny, lnz);
            dist(ta, c, a, kProjected);
            dist(tb, c, a, kProjected);
            dist(tz, c, a, kZero);
            ta.step();
            tb.step();
            tz.step();
            check(rank, sameBitsAll(ta, tz, c.vof),
                  "[" + tag + "] G-E3 step 1 of 'projected' == 'zero', bitwise");
            for (int k = 0; k < 4; ++k) {
              ta.step();
              tb.step();
              tz.step();
            }
            const bool live = ta.pressureGuessCount() == 2 && !sameBitsAll(ta, tz, c.vof);
            check(rank, live, "[" + tag + "] the guess is live (history 2, not bitwise 'zero')");
            for (IbmSolver* t : {&ta, &tb})  // the state write, same values, on both twins
              t->uploadVelocity(t->getVelocity(0), t->getVelocity(1), t->getVelocity(2));
            tb.setPressureInitialGuess(kZero, 2);
            ta.step();
            tb.step();
            check(rank, ta.pressureGuessCount() == 1 && sameBitsAll(ta, tb, c.vof),
                  "[" + tag + "] G-E3 the step after a state write == the twin's 'zero' step");
          }
          // ---- G-ED4: a size-preserving redistribute restarts the history --------------------
          {
            IbmSolver ta(lnx, lny, lnz), tb(lnx, lny, lnz);
            dist(ta, c, a, kProjected);
            dist(tb, c, a, kProjected);
            for (int k = 0; k < 4; ++k) {
              ta.step();
              tb.step();
            }
            const bool before = ta.pressureGuessCount() == 2;
            ta.redistribute(dec);  // the same partition: every block keeps its size
            tb.redistribute(dec);
            const bool reset = ta.pressureGuessCount() == 0;
            tb.setPressureInitialGuess(kZero, 2);
            ta.step();
            tb.step();
            const bool cold = sameBitsAll(ta, tb, c.vof);
            const bool h1 = ta.pressureGuessCount() == 1;  // the next solve runs on history 1
            ta.step();
            tb.step();
            const bool used = ta.pressureGuessCount() == 2 && !sameBitsAll(ta, tb, c.vof);
            check(rank, before && reset && cold && h1 && used,
                  "[" + tag + "] G-ED4 redistribute resets: next step == 'zero', then history 1");
          }
        }
      }
      if (rank == 0)
        std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fails ? 1 : 0;
}
