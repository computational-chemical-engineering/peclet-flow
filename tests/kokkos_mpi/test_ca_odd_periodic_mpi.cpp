// Communication-avoiding (CA) red-black pairs across a periodic axis of ODD global extent.
//
// Both CA pairs (CutcellMG::smooth on the g = 2 coarse levels, IbmSolver's momentum sweep) sweep
// the 1-deep ghost ring in colour 0 with the ring slot's OWN parity. Across a periodic wrap of odd
// global extent the slot (z = -1, the image of z = nz - 1) has the opposite parity of its image, so
// the ring would be swept in the wrong colour; CA is therefore off on such a level / velocity
// decomposition and the per-colour exchange runs instead (mac_cutcell_mg.hpp, flow_ibm_mpi.hpp).
// Grid 32 x 32 x 25, periodic on every axis, so z is odd at every MG level.
//   (a) CutcellMG MG-PCG, a fixed 6 iterations, CA on and off: np 1 memcmp == single rank; at any
//       np CA on memcmp == CA off; np > 1 within test_cutcellmg_mpi's 1e-8 of single rank.
//   (b) IbmSolver Stokes through a sphere (cut-cell IBM, the CA momentum path), 4 steps, CA
//       both / momentum / pressure / off: np 1 memcmp == single rank (u, v, w, p); at any np each
//       arm memcmp == off; np > 1 velocity within test_sdflow_mpi's relative 2e-5 of single rank.
// Before the fix np 1 differed from single rank in (a) (CA on) and in (b) (both, momentum).
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "flow_ibm.hpp"
#include "mac_cutcell_mg.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::CutcellMG;
using peclet::flow::IbmSolver;

static constexpr int NX = 32, NY = 32, NZ = 25;
static int gRank = 0, gSize = 1, gFail = 0;

static void check(bool ok, const std::string& what) {
  int all = 0, v = ok ? 1 : 0;
  MPI_Allreduce(&v, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (gRank == 0)
    std::printf("  %-78s %s\n", what.c_str(), all ? "ok" : "FAIL");
  if (!all)
    ++gFail;
}

// This rank's inner-block values (x-fastest) against a global (single-rank) array: max|a - ref|
// and the memcmp-unequal count, both reduced over the ranks.
struct Diff {
  double maxAbs = 0.0, refMax = 0.0;
  long bits = 0;
};
static Diff compareBlock(const std::vector<double>& a, const std::vector<double>& ref, int ox,
                         int oy, int oz, int lnx, int lny, int lnz) {
  Diff d;
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        const double u = a[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny];
        const double r = ref[(std::size_t)(x + ox) + (std::size_t)(y + oy) * NX +
                             (std::size_t)(z + oz) * NX * NY];
        d.maxAbs = std::max(d.maxAbs, std::fabs(u - r));
        d.refMax = std::max(d.refMax, std::fabs(r));
        if (std::memcmp(&u, &r, sizeof(double)) != 0)
          ++d.bits;
      }
  MPI_Allreduce(MPI_IN_PLACE, &d.maxAbs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d.refMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d.bits, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return d;
}

static bool sameBits(const std::vector<double>& a, const std::vector<double>& b) {
  int eq = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
  MPI_Allreduce(MPI_IN_PLACE, &eq, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return eq != 0;
}

static std::string fmt(const char* f, double a, long b = 0) {
  char buf[96];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}

// ---------------------------------------------------------------------------------------------
// (a) CutcellMG: periodic all-fluid Poisson, mean-free source, 6 MG-PCG iterations, 4 levels.
static double source(int gx, int gy, int gz) {
  return std::sin(2.0 * M_PI * gx / NX) +
         std::cos(4.0 * M_PI * gy / NY) * std::sin(2.0 * M_PI * gz / NZ);
}

// Solve on `mg` (level 0 of ghost width g); returns the inner block, x-fastest.
static std::vector<double> mgSolve(CutcellMG& mg) {
  const CutcellMG::Level& l0 = mg.level(0);
  const C3 e = l0.ext, og = l0.og;
  const int g = l0.g;
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  CCField ox("ox", n), oy("oy", n), oz("oz", n);
  Kokkos::deep_copy(ox, 1.0);
  Kokkos::deep_copy(oy, 1.0);
  Kokkos::deep_copy(oz, 1.0);
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
  CCField b("b", n), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
  auto hb = Kokkos::create_mirror_view(b);
  for (std::size_t c = 0; c < n; ++c)
    hb(c) = 0.0;
  for (int k = g; k < e.z - g; ++k)
    for (int j = g; j < e.y - g; ++j)
      for (int i = g; i < e.x - g; ++i)
        hb((long)i + (long)j * e.x + (long)k * e.x * e.y) =
            source(i - g + og.x, j - g + og.y, k - g + og.z);
  Kokkos::deep_copy(b, hb);
  mg.solvePCG(b, x, r, p, z, Ap, /*maxit=*/6, /*rtol=*/1e-30, 2, 2, 8);
  auto hx = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(hx, x);
  std::vector<double> out;
  for (int k = g; k < e.z - g; ++k)
    for (int j = g; j < e.y - g; ++j)
      for (int i = g; i < e.x - g; ++i)
        out.push_back(hx((long)i + (long)j * e.x + (long)k * e.x * e.y));
  return out;
}

static void mgChecks() {
  constexpr int NLEV = 4;
  CutcellMG ref;
  ref.init(NX, NY, NZ, NLEV);
  const std::vector<double> xs = mgSolve(ref);
  std::vector<double> xd[2];
  int lnx = 0, lny = 0, lnz = 0, ox = 0, oy = 0, oz = 0;
  const int modes[2] = {peclet::flow::kCaMg, 0};
  for (int k = 0; k < 2; ++k) {
    CutcellMG mg;
    mg.setCommAvoiding(modes[k]);
    mg.initMpi(NX, NY, NZ, NLEV, MPI_COMM_WORLD);
    if (k == 0) {
      int caLevels = 0;
      for (int L = 0; L < mg.nLevels(); ++L)
        caLevels += mg.level(L).caOk ? 1 : 0;
      check(caLevels == 0, fmt("(a) no CA level on the odd periodic grid (%g of them)", caLevels));
    }
    xd[k] = mgSolve(mg);
    const CutcellMG::Level& l0 = mg.level(0);
    lnx = l0.inner.x;
    lny = l0.inner.y;
    lnz = l0.inner.z;
    ox = l0.og.x;  // the inner origin, global
    oy = l0.og.y;  // the inner origin, global
    oz = l0.og.z;  // the inner origin, global
  }
  check(sameBits(xd[0], xd[1]), "(a) MG-PCG x: CA on memcmp == CA off");
  for (int k = 0; k < 2; ++k) {
    const Diff d = compareBlock(xd[k], xs, ox, oy, oz, lnx, lny, lnz);
    const std::string t = std::string("(a) MG-PCG x, CA ") + (k == 0 ? "on" : "off");
    if (gSize == 1)
      check(d.bits == 0, t + fmt(": np 1 memcmp == single rank (max|d| %.3e, %ld cells differ)",
                                 d.maxAbs, d.bits));
    else
      check(d.maxAbs <= 1e-8, t + fmt(": max|d| vs single rank %.3e <= 1e-8", d.maxAbs));
  }
}

// ---------------------------------------------------------------------------------------------
// (b) IbmSolver: Stokes through one periodic sphere, cut-cell pressure, 4 steps.
static std::vector<double> sphereSdf() {
  std::vector<double> sdf((std::size_t)NX * NY * NZ);
  const double R = 0.3 * NZ, c[3] = {0.5 * NX, 0.5 * NY, 0.5 * NZ};
  const int n[3] = {NX, NY, NZ};
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        double d2 = 0.0;
        const int p[3] = {x, y, z};
        for (int a = 0; a < 3; ++a) {
          const double d = p[a] - c[a];
          const double w = d - n[a] * std::round(d / n[a]);
          d2 += w * w;
        }
        sdf[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = std::sqrt(d2) - R;
      }
  return sdf;
}

static void configure(IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(0.1);
  s.setDt(5.0);
  s.setBodyForce(1e-3, 2e-4, 0);
  s.setAdvection(false);
  s.setVelocityIterations(40);
  s.setPressureLevels(4);
  s.setPressurePcg(true, 200, 1e-9);
}

struct Fields {
  std::vector<double> f[4];  // u, v, w, p (inner block, x-fastest)
};
static Fields fieldsOf(IbmSolver& s) {
  Fields r;
  for (int c = 0; c < 3; ++c)
    r.f[c] = s.getVelocity(c);
  r.f[3] = s.getPressure();
  return r;
}

static void solverChecks() {
  constexpr int STEPS = 4;
  const std::vector<double> gsdf = sphereSdf();
  IbmSolver ref(NX, NY, NZ);
  configure(ref);
  // The distributed solve runs the 'direct' bottom on a host backend and GraphAMG on a device,
  // where the single-rank solver would take the device 'direct' bottom (8.3e-10 in p at np 1);
  // the np = 1 bit-exact gate pins the reference to the engine the distributed path runs.
  ref.setPressureBottomSolver(
      std::is_same_v<peclet::flow::CCExec::memory_space, Kokkos::HostSpace> ? 0 : 2);
  ref.setSolid(gsdf, /*cutcell_pressure=*/true);
  for (int it = 0; it < STEPS; ++it)
    ref.step();
  const Fields fs = fieldsOf(ref);

  const peclet::core::decomp::BlockDecomposer<3> dec =
      CutcellMG::decomposition(static_cast<std::size_t>(gSize), NX, NY, NZ);
  const auto blk = dec.block(gRank);
  const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
  const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
  std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * NX +
                 (std::size_t)(z + oz) * NX * NY];

  const int modes[4] = {peclet::flow::kCaBoth, peclet::flow::kCaMomentum, peclet::flow::kCaMg, 0};
  const char* names[4] = {"both", "momentum", "pressure", "off"};
  Fields fd[4];
  for (int k = 0; k < 4; ++k) {
    IbmSolver s(lnx, lny, lnz);
    s.setCommAvoiding(modes[k]);
    s.initMpi(NX, NY, NZ, MPI_COMM_WORLD);
    configure(s);
    s.setSolid(lsdf, true);
    for (int it = 0; it < STEPS; ++it)
      s.step();
    fd[k] = fieldsOf(s);
  }
  for (int k = 0; k < 4; ++k) {
    const std::string t = std::string("(b) CA ") + names[k];
    if (k < 3) {
      bool eq = true;
      for (int c = 0; c < 4; ++c)
        eq = sameBits(fd[k].f[c], fd[3].f[c]) && eq;
      check(eq, t + ": u, v, w, p memcmp == CA off");
    }
    if (gSize == 1) {
      long bits = 0;
      double mx = 0.0;
      for (int c = 0; c < 4; ++c) {
        const Diff d = compareBlock(fd[k].f[c], fs.f[c], ox, oy, oz, lnx, lny, lnz);
        bits += d.bits;
        mx = std::max(mx, d.maxAbs);
      }
      check(bits == 0,
            t + fmt(": np 1 u, v, w, p memcmp == single rank (max|d| %.3e, %ld differ)", mx, bits));
    } else {
      double rel = 0.0;
      for (int c = 0; c < 3; ++c) {
        const Diff d = compareBlock(fd[k].f[c], fs.f[c], ox, oy, oz, lnx, lny, lnz);
        rel = std::max(rel, d.maxAbs / (d.refMax + 1e-300));
      }
      if (gRank == 0)
        std::printf("    %s: max over u, v, w of max|d| / max|ref| = %.3e\n", names[k], rel);
      check(rel <= 2e-5, t + fmt(": velocity vs single rank, relative %.3e <= 2e-5", rel));
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);
    if (gRank == 0)
      std::printf("CA red-black pairs on a %dx%dx%d periodic grid (odd z), np %d\n", NX, NY, NZ,
                  gSize);
    mgChecks();
    solverChecks();
  }
  if (gRank == 0) {
    if (gFail == 0)
      std::printf("OK (np=%d): odd periodic extent -- CA off, np 1 == single rank\n", gSize);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d checks\n", gSize, gFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return gFail == 0 ? 0 : 1;
}
