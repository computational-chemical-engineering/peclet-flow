// symconv.cpp: gate G-SYMCONV of doc/mg_depth_policy.md Amendment A3 (A3.5, WO-S2) -- the
// production solvers (CutcellMG::solvePCG / solveFCG, schedule (2, 2, 12), rtol 1e-10, zero guess,
// rhsOf with closed cells zero, cap 500) on the rows of A3's transfer-pair matrix, with the
// symmetric V-cycle switch (setSymmetricVcycle) off and on, on one hierarchy. Host, single rank.
// The problem presets are those of the matrix tool
// (bubble_column_perf/gpu_diag/mgd/a3/symdiag.cpp). Usage: symconv <case> <levels> [LR 0|1, default
// 1]. One line per case:
//   SYMCONV <case> D<levels> LR<lr> levels <n> bottom <dims> <smoothed|agglomerated> |
//   off PCG <it>[ capped] FCG <it>[ capped] | on PCG <it>[ capped] FCG <it>[ capped]
// Build: build_diag.sh (gpu_diag/mgd/wo7d) against the tree's host libpeclet_flow_solver_mpi.a.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "mg_fp32_problems.hpp"

using namespace peclet::flow;
using namespace peclet_flow_fp32_problems;
using Level = CutcellMG::Level;

static double val(long gi) {
  unsigned long s = (unsigned long)gi * 6364136223846793005ul + 1442695040888963407ul;
  s ^= s >> 33;
  s *= 0xff51afd7ed558ccdul;
  s ^= s >> 33;
  return (double)(s >> 11) / (double)(1ul << 52) - 1.0;
}

static Problem preset(const std::string& nm, int levels) {  // symdiag.cpp's presets
  auto one = [](double, double, double) { return 1.0; };
  if (nm == "thinW" || nm == "slabP") {
    int bc[6] = {0, 0, 0, 0, nm == "thinW" ? 1 : 0, nm == "thinW" ? 1 : 0};
    auto rho = [](double x, double y, double z) {
      const double dx = x - 24, dy = y - 40, dz = (z - 2.0) * 3.0;
      return dx * dx + dy * dy + dz * dz < 144.0 ? 1e2 : 1.0;
    };
    return makeProblem(nm, 64, 64, 4, levels, bc, one, rho);
  }
  if (nm == "iso64p" || nm == "iso64w" || nm == "iso64c4") {
    const int w = nm == "iso64w" ? 1 : 0;
    int bc[6] = {w, w, w, w, w, w};
    const double ratio = nm == "iso64c4" ? 1e4 : 1e2;
    auto rho = [ratio](double x, double y, double z) {
      const double dx = x - 20, dy = y - 36, dz = z - 28;
      return dx * dx + dy * dy + dz * dz < 100.0 ? ratio : 1.0;
    };
    return makeProblem(nm, 64, 64, 64, levels, bc, one, rho);
  }
  if (nm == "uni128" || nm == "uni256" || nm == "uni256w") {
    const int N = nm == "uni128" ? 128 : 256, w = nm == "uni256w" ? 1 : 0;
    int bc[6] = {w, w, w, w, w, w};
    return makeProblem(nm, N, N, N, levels, bc, one, one);
  }
  if (nm == "cyl64") {
    int bc[6] = {0, 0, 0, 0, 0, 0};
    auto open = [](double x, double, double z) {
      return clamp01(0.5 + std::sqrt((x - 32) * (x - 32) + (z - 32) * (z - 32)) - 16.0);
    };
    return makeProblem(nm, 64, 64, 64, levels, bc, open, one);
  }
  if (nm == "iso64cut") {
    int bc[6] = {0, 0, 0, 0, 0, 0};
    auto open = [](double x, double y, double z) {
      const double dx = x - 32, dy = y - 30, dz = z - 34;
      return dx * dx + dy * dy + dz * dz < 18.0 * 18.0 ? 0.0 : 1.0;
    };
    return makeProblem(nm, 64, 64, 64, levels, bc, open, one);
  }
  const bool m10 = nm == "m10";  // else iso128
  const int nz = m10 ? 25 : 128;
  int bc[6] = {0, 0, 0, 0, 0, 0};
  std::vector<std::array<double, 4>> B;
  for (int k = 0; k < (m10 ? 12 : 40); ++k)
    B.push_back({val(100 + k) * 64 + 64, val(200 + k) * 64 + 64, (val(300 + k) * 0.5 + 0.5) * nz,
                 m10 ? 5.0 : 7.0});
  auto rho = [B](double x, double y, double z) {
    for (const auto& b : B) {
      const double dx = x - b[0], dy = y - b[1], dz = z - b[2];
      if (dx * dx + dy * dy + dz * dz < b[3] * b[3])
        return 1e-3;
    }
    return 1.0;
  };
  return makeProblem(nm, 128, 128, nz, levels, bc, one, rho);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    const std::string nm = argc > 1 ? argv[1] : "thinW";
    const int levels = argc > 2 ? std::atoi(argv[2]) : 0;
    const int lr = argc > 3 ? std::atoi(argv[3]) : 1;
    const Problem P = preset(nm, levels);
    CutcellMG mg;
    setUp(mg, P, CutcellMG::kVcycleFp64);
    mg.setLineRelaxation(lr != 0);
    Level& l0 = mg.level(0);
    const std::size_t n = l0.n;
    std::vector<double> hb = rhsOf(P);
    auto ac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l0.AC);
    for (std::size_t i = 0; i < n; ++i)
      if (!((double)ac(i) > 1e-30))
        hb[i] = 0.0;
    constexpr int kCap = 500;
    std::string out;
    for (int sym : {0, 1}) {
      mg.setSymmetricVcycle(sym != 0);
      CCField b = toDevice(hb, "b"), x("x", n), r("r", n), p("p", n), z("z", n), zp("zp", n),
              Ap("Ap", n);
      const int ip = mg.solvePCG(b, x, r, p, z, Ap, kCap, 1e-10, 2, 2, 12);
      const bool fp = mg.lastSolveFailed() || ip >= kCap;
      Kokkos::deep_copy(x, 0.0);
      const int ifc = mg.solveFCG(b, x, r, p, z, zp, Ap, kCap, 1e-10, 2, 2, 12);
      const bool ff = mg.lastSolveFailed() || ifc >= kCap;
      char buf[160];
      std::snprintf(buf, sizeof buf, " | %s PCG %d%s FCG %d%s", sym ? "on" : "off", ip,
                    fp ? " capped" : "", ifc, ff ? " capped" : "");
      out += buf;
    }
    {  // the asymmetry a of the symmetric V-cycle (ctest mg_vcycle_symmetry's measure)
      mg.setSymmetricVcycle(true);
      CCField x("x", n), y("y", n), Mx("Mx", n), My("My", n);
      for (int k = 0; k < 2; ++k) {
        auto h = Kokkos::create_mirror_view(k ? y : x);
        for (std::size_t i = 0; i < n; ++i)
          h(i) = 0.0;
        for (int zz = 0; zz < l0.inner.z; ++zz)
          for (int yy = 0; yy < l0.inner.y; ++yy)
            for (int xx = 0; xx < l0.inner.x; ++xx) {
              const long i = (long)(xx + l0.g) + (long)(yy + l0.g) * l0.ext.x +
                             (long)(zz + l0.g) * (long)l0.ext.x * l0.ext.y;
              h(i) = (double)ac(i) > 1e-30 ? val((11 + k) * 7919L + xx + 1000L * yy + 1000000L * zz)
                                           : 0.0;
            }
        Kokkos::deep_copy(k ? y : x, h);
        mg.removeMean(l0, k ? y : x);
      }
      mg.precondVcycle(Mx, x);
      mg.precondVcycle(My, y);
      const double a1 = mg.dot(l0, x, My), b1 = mg.dot(l0, y, Mx);
      char buf[64];
      std::snprintf(
          buf, sizeof buf, " | a(on) %.3e",
          std::abs(a1 - b1) / (std::sqrt(mg.dot(l0, x, x)) * std::sqrt(mg.dot(l0, My, My))));
      out += buf;
    }
    const Level& bl = mg.level(mg.nLevels() - 1);
    std::printf("SYMCONV %s D%d LR%d levels %d bottom %dx%dx%d %s%s\n", nm.c_str(), levels, lr,
                mg.nLevels(), bl.inner.x, bl.inner.y, bl.inner.z,
                mg.agglomerateBottom() ? "agglomerated" : "smoothed", out.c_str());
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
