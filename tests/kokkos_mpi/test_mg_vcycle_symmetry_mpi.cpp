/// @file
/// @brief ctest `mg_vcycle_symmetry_mpi`: the distributed half of gate G-SYM of
/// doc/mg_depth_policy.md Amendment A3 (A3.5) -- the symmetric pressure V-cycle
/// (CutcellMG::setSymmetricVcycle) distributed over MPI_COMM_WORLD (initMpi, default telescoping
/// and communication-avoiding coarse levels).
///
/// Measure as ctest `mg_vcycle_symmetry`: a = |x^T M y - y^T M x| / (|x| |M y|) of the production
/// V-cycle (precondVcycle, schedule (2, 2, 12)), x and y the single-rank test's vectors (the same
/// pseudo-random value per GLOBAL cell), closed cells zero, mean-free; dots global.
/// Checks (switch on):
///   np 1: one V-cycle z = M r memcmp-equal on the inner cells to the single-rank V-cycle (every
///         problem below);
///   every np: a <= 1e-14 on the smoothed-bottom problems of the single-rank test (thinW with line
///         relaxation on and off, the thin periodic slab, `scalar`, the 64^3 periodic and walled
///         blobs at 5 levels, cylProblem, the outflow-anchored channel); a <= 1e-8 with the
///         agglomerated bottom (64^3 periodic blob, 4 levels, forced) -- gated (and its np 1
///         memcmp) on a host backend (G-SYM: host np 2 and 4), recorded on a device backend (a
///         distributed device bottom is GraphAMG, the single-rank one the direct factor).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "mg_fp32_problems.hpp"

using namespace peclet::flow;
using namespace peclet_flow_fp32_problems;
using Level = CutcellMG::Level;

namespace {

int rank = 0, size = 1, fails = 0;
void check(bool ok, const std::string& what) {
  int bad = ok ? 0 : 1, any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("  %-96s %s\n", what.c_str(), any ? "FAIL" : "PASS");
  std::fflush(stdout);
  if (any)
    ++fails;
}
std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3e", v);
  return b;
}

constexpr double kBoundSmoothed = 1e-14, kBoundAgglomerated = 1e-8;

double val(long key) {  // as ctest mg_vcycle_symmetry
  unsigned long s = (unsigned long)key * 6364136223846793005ul + 1442695040888963407ul;
  s ^= s >> 33;
  s *= 0xff51afd7ed558ccdul;
  s ^= s >> 33;
  return (double)(s >> 11) / (double)(1ul << 52) - 1.0;
}

// test_mg_fp32_vcycle_mpi.cpp's setUpDist: this rank's block of the global openness.
void setUpDist(CutcellMG& mg, const Problem& P) {
  mg.setBoundaryConditions(P.bc);  // before initMpi (the per-level ghost width reads it)
  mg.initMpi(P.nx, P.ny, P.nz, P.levels, MPI_COMM_WORLD);
  mg.setVcyclePrecision(CutcellMG::kVcycleFp64);
  const Level& l0 = mg.level(0);
  const C3 e = l0.ext, og = l0.og;
  const int g = l0.g;
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  std::vector<double> ox(n, 0.0), oy(n, 0.0), oz(n, 0.0);
  for (int z = g; z < e.z - g; ++z)
    for (int y = g; y < e.y - g; ++y)
      for (int x = g; x < e.x - g; ++x) {
        const long li = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const long gi = P.idx(x - g + og.x, y - g + og.y, z - g + og.z);
        ox[li] = P.ox[gi];
        oy[li] = P.oy[gi];
        oz[li] = P.oz[gi];
      }
  mg.setOpenness(CCConst(toDevice(ox, "ox")), CCConst(toDevice(oy, "oy")),
                 CCConst(toDevice(oz, "oz")), 1.0, 1.0, 1.0);
}
// Level 0's inner cells: f(global inner x, y, z, local index).
template <class F>
void forInner(const Level& l0, F f) {
  const C3 e = l0.ext;
  const int g = l0.g;
  for (int z = g; z < e.z - g; ++z)
    for (int y = g; y < e.y - g; ++y)
      for (int x = g; x < e.x - g; ++x)
        f(x - g + l0.og.x, y - g + l0.og.y, z - g + l0.og.z,
          (long)x + (long)y * e.x + (long)z * (long)e.x * e.y);
}
// The random vector of ctest mg_vcycle_symmetry on this rank's block (closed cells zero),
// mean-free.
CCField randomVector(CutcellMG& mg, int seed) {
  Level& l = mg.level(0);
  CCField f("v", l.n);
  auto h = Kokkos::create_mirror_view(f);
  auto ac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l.AC);
  for (std::size_t i = 0; i < h.extent(0); ++i)
    h(i) = 0.0;
  forInner(l, [&](int x, int y, int z, long i) {
    h(i) = (double)ac(i) > 1e-30 ? val(seed * 7919L + x + 1000L * y + 1000000L * z) : 0.0;
  });
  Kokkos::deep_copy(f, h);
  mg.removeMean(l, f);
  return f;
}
void setSchedule(CutcellMG& mg) {  // the drivers' (2, 2, 12)
  const std::size_t n = mg.level(0).n;
  CCField s0("s0", n), s1("s1", n), s2("s2", n), s3("s3", n), s4("s4", n), s5("s5", n);
  mg.solvePCG(s0, s1, s2, s3, s4, s5, 0, 1e-8, 2, 2, 12);
}
double asym(CutcellMG& mg) {
  Level& l = mg.level(0);
  setSchedule(mg);
  CCField x = randomVector(mg, 11), y = randomVector(mg, 12), Mx("Mx", l.n), My("My", l.n);
  mg.precondVcycle(Mx, x);
  mg.precondVcycle(My, y);
  const double a = mg.dot(l, x, My), b = mg.dot(l, y, Mx);
  return std::abs(a - b) / (std::sqrt(mg.dot(l, x, x)) * std::sqrt(mg.dot(l, My, My)));
}
// One symmetric V-cycle z = M r, r = randomVector(seed 5), as a host array of level 0.
std::vector<double> oneVcycle(CutcellMG& mg) {
  Level& l = mg.level(0);
  setSchedule(mg);
  CCField r = randomVector(mg, 5), z("z", l.n);
  mg.precondVcycle(z, r);
  return toHost(z);
}

Problem thinProblem(bool walls) {  // as ctest mg_vcycle_symmetry
  const int w = walls ? 1 : 0;
  const int bc[6] = {0, 0, 0, 0, w, w};
  auto rho = [](double x, double y, double z) {
    const double dx = x - 24, dy = y - 40, dz = (z - 2.0) * 3.0;
    return dx * dx + dy * dy + dz * dz < 144.0 ? 1e2 : 1.0;
  };
  return makeProblem(
      walls ? "thinW" : "slabP", 64, 64, 4, 5, bc, [](double, double, double) { return 1.0; }, rho);
}
Problem blobProblem(bool walls, int levels) {
  const int w = walls ? 1 : 0;
  const int bc[6] = {w, w, w, w, w, w};
  auto rho = [](double x, double y, double z) {
    const double dx = x - 20, dy = y - 36, dz = z - 28;
    return dx * dx + dy * dy + dz * dz < 100.0 ? 1e2 : 1.0;
  };
  return makeProblem(
      std::string(walls ? "iso64w" : "iso64p") + " D" + std::to_string(levels), 64, 64, 64, levels,
      bc, [](double, double, double) { return 1.0; }, rho);
}
Problem scalarProblem() {
  const int bc[6] = {1, 1, 1, 1, 0, 0};
  auto one = [](double, double, double) { return 1.0; };
  return makeProblem("scalar", 16, 16, 4, 0, bc, one, one);
}
Problem channelProblem() {
  const int bc[6] = {2, 3, 1, 1, 1, 1};
  auto one = [](double, double, double) { return 1.0; };
  return makeProblem("channel", 32, 16, 16, 4, bc, one, one);
}

void symCase(const Problem& P, bool lr, bool agglom) {
  const std::string tag = P.name + (lr ? "" : " LR off") + " np" + std::to_string(size);
  CutcellMG mg;
  setUpDist(mg, P);
  mg.setLineRelaxation(lr);
  if (agglom)
    mg.setAgglomerationMode(1);
  mg.setSymmetricVcycle(true);
  if (rank == 0)
    std::printf("%s: %d levels, bottom %s\n", tag.c_str(), mg.nLevels(),
                mg.agglomerateBottom() ? "agglomerated" : "smoothed");
  // A distributed device bottom is GraphAMG, the single-rank one the direct factor: on a device
  // backend the agglomerated case is recorded only (G-SYM gates it on host).
  const bool gated = !(agglom && !CutcellMG::kHostMemory);
  if (size == 1 && gated) {  // np 1 init_mpi == single rank, one V-cycle, memcmp on the inner cells
    CutcellMG ref;
    setUp(ref, P, CutcellMG::kVcycleFp64);
    ref.setLineRelaxation(lr);
    if (agglom)
      ref.setAgglomerationMode(1);
    ref.setSymmetricVcycle(true);
    const std::vector<double> zd = oneVcycle(mg), zs = oneVcycle(ref);
    const Level& l0 = mg.level(0);
    long bits = 0;
    forInner(l0, [&](int x, int y, int z, long i) {
      if (std::memcmp(&zd[i], &zs[P.idx(x, y, z)], sizeof(double)) != 0)
        ++bits;
    });
    check(bits == 0, tag + ": init_mpi V-cycle memcmp-equal to single rank (" +
                         std::to_string(bits) + " cells differ)");
  }
  const double a = asym(mg);
  const double bound = agglom ? kBoundAgglomerated : kBoundSmoothed;
  if (rank == 0)
    std::printf("  a %s (bound %s)\n", num(a).c_str(), num(bound).c_str());
  if (!gated) {
    if (rank == 0)
      std::printf("  %s: a %s recorded (distributed device bottom, not gated)\n", tag.c_str(),
                  num(a).c_str());
    return;
  }
  check(a <= bound, tag + ": a " + num(a) + " <= " + num(bound));
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Kokkos::initialize(argc, argv);
  {
    const Problem thinW = thinProblem(true), slabP = thinProblem(false);
    symCase(thinW, true, false);
    symCase(thinW, false, false);
    symCase(slabP, true, false);
    symCase(scalarProblem(), true, false);
    symCase(blobProblem(false, 5), true, false);
    symCase(blobProblem(true, 5), true, false);
    symCase(cylProblem(), true, false);
    symCase(channelProblem(), true, false);
    symCase(blobProblem(false, 4), true, true);
  }
  Kokkos::finalize();
  if (rank == 0)
    std::printf("mg_vcycle_symmetry_mpi np%d: %s (%d failure%s)\n", size, fails ? "FAIL" : "PASS",
                fails, fails == 1 ? "" : "s");
  MPI_Finalize();
  return fails ? 1 : 0;
}
