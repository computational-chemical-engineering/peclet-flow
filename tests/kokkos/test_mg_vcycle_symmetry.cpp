/// @file
/// @brief ctest `mg_vcycle_symmetry`: gate G-SYM of doc/mg_depth_policy.md Amendment A3 (A3.5),
/// single rank, host and device backends -- the symmetric pressure V-cycle
/// (CutcellMG::setSymmetricVcycle: piecewise-constant prolongation P = c R^T of the cell-average
/// restriction, palindromic smoothed bottom). Also the guard against reintroducing a prolongation
/// that is not c R^T under a CG driver: a gate, not a grep.
///
/// Measure (D3 of gpu_diag/mgd/WO7_DIAG.md): a = |x^T M y - y^T M x| / (|x| |M y|) of the
/// production V-cycle M (precondVcycle, the drivers' schedule pre = post = 2, bottom 12), x and y
/// deterministic pseudo-random, closed cells zero, mean-free.
/// Checks (switch on):
///   a <= 1e-14 on the smoothed-bottom hierarchies thinW (64x64x4, z walls, a rho x 100 blob, 5
///   levels) with line relaxation on and off, the thin periodic slab (thinW periodic in z), the
///   state_hash `scalar` hierarchy (16x16x4, x/y walls, z periodic, derived depth), 64^3 periodic
///   and walled with a rho x 100 blob (5 levels), cylProblem (32^3, fractional-aperture cylinder,
///   a rho x 1e4 blob), and an outflow-anchored channel (32x16x16, -x inflow, +x outflow, y/z
///   walls, uniform);
///   a <= 1e-8 with the agglomerated bottom (64^3 periodic blob, 4 levels, forced);
///   the FP32 V-cycle (forced 'fp32' where eligible): a recorded; on a device backend PCG on thinW
///   not capped;
///   production solvePCG (rhsOf, rtol 1e-10, zero guess, cap 500): thinW <= 16 (line relaxation on
///   and off), thin periodic slab <= 14.
/// The switch-off values are printed for reference (the trilinear V-cycle's a), not gated.
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

namespace {

int fails = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %-90s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  if (!ok)
    ++fails;
}
std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3e", v);
  return b;
}

constexpr double kBoundSmoothed = 1e-14, kBoundAgglomerated = 1e-8;
constexpr int kCap = 500;
constexpr double kRtol = 1e-10;

// Deterministic pseudo-random value in [-1, 1) of a global cell key.
double val(long key) {
  unsigned long s = (unsigned long)key * 6364136223846793005ul + 1442695040888963407ul;
  s ^= s >> 33;
  s *= 0xff51afd7ed558ccdul;
  s ^= s >> 33;
  return (double)(s >> 11) / (double)(1ul << 52) - 1.0;
}
// A random level-0 vector (inner cells; ghosts 0), closed cells (AC <= 1e-30) zero, mean-free.
CCField randomVector(CutcellMG& mg, int seed) {
  Level& l = mg.level(0);
  CCField f("v", l.n);
  auto h = Kokkos::create_mirror_view(f);
  auto ac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l.AC);
  for (std::size_t i = 0; i < h.extent(0); ++i)
    h(i) = 0.0;
  for (int z = 0; z < l.inner.z; ++z)
    for (int y = 0; y < l.inner.y; ++y)
      for (int x = 0; x < l.inner.x; ++x) {
        const long i =
            (long)(x + l.g) + (long)(y + l.g) * l.ext.x + (long)(z + l.g) * (long)l.ext.x * l.ext.y;
        h(i) = (double)ac(i) > 1e-30 ? val(seed * 7919L + x + 1000L * y + 1000000L * z) : 0.0;
      }
  Kokkos::deep_copy(f, h);
  mg.removeMean(l, f);
  return f;
}
// The drivers' smoothing schedule (2, 2, 12): a zero-iteration solve sets it.
void setSchedule(CutcellMG& mg) {
  const std::size_t n = mg.level(0).n;
  CCField s0("s0", n), s1("s1", n), s2("s2", n), s3("s3", n), s4("s4", n), s5("s5", n);
  mg.solvePCG(s0, s1, s2, s3, s4, s5, 0, 1e-8, 2, 2, 12);
}
// a of the V-cycle as `mg` is configured (FP32 when its precision is forced 'fp32').
double asym(CutcellMG& mg, bool fp32) {
  Level& l = mg.level(0);
  setSchedule(mg);
  CCField x = randomVector(mg, 11), y = randomVector(mg, 12), Mx("Mx", l.n), My("My", l.n);
  if (fp32) {
    mg.precondVcycleFp32(Mx, x, std::ilogb(mg.maxabs(l, x)), -1, CCField());
    mg.precondVcycleFp32(My, y, std::ilogb(mg.maxabs(l, y)), -1, CCField());
  } else {
    mg.precondVcycle(Mx, x);
    mg.precondVcycle(My, y);
  }
  const double a = mg.dot(l, x, My), b = mg.dot(l, y, Mx);
  return std::abs(a - b) / (std::sqrt(mg.dot(l, x, x)) * std::sqrt(mg.dot(l, My, My)));
}
// Production solvePCG on rhsOf(P) (closed cells zero), zero guess.
int pcgIters(CutcellMG& mg, const Problem& P) {
  Level& l = mg.level(0);
  const std::size_t n = l.n;
  std::vector<double> hb = rhsOf(P);
  auto ac = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l.AC);
  for (std::size_t i = 0; i < n; ++i)
    if (!((double)ac(i) > 1e-30))
      hb[i] = 0.0;
  CCField b = toDevice(hb, "b"), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
  return mg.solvePCG(b, x, r, p, z, Ap, kCap, kRtol, 2, 2, 12);
}

Problem thinProblem(bool walls) {  // gpu_diag/mgd/a3/symdiag.cpp presets thinW / slabP
  const int w = walls ? 1 : 0;
  const int bc[6] = {0, 0, 0, 0, w, w};
  auto rho = [](double x, double y, double z) {
    const double dx = x - 24, dy = y - 40, dz = (z - 2.0) * 3.0;
    return dx * dx + dy * dy + dz * dz < 144.0 ? 1e2 : 1.0;
  };
  return makeProblem(
      walls ? "thinW" : "slabP", 64, 64, 4, 5, bc, [](double, double, double) { return 1.0; }, rho);
}
Problem blobProblem(bool walls, int levels) {  // symdiag.cpp presets iso64p / iso64w
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
Problem scalarProblem() {  // the state_hash `scalar` case's pressure operator
  const int bc[6] = {1, 1, 1, 1, 0, 0};
  auto one = [](double, double, double) { return 1.0; };
  return makeProblem("scalar", 16, 16, 4, 0, bc, one, one);
}
Problem channelProblem() {  // outflow-anchored: -x inflow, +x outflow (Dirichlet), y/z walls
  const int bc[6] = {2, 3, 1, 1, 1, 1};
  auto one = [](double, double, double) { return 1.0; };
  return makeProblem("channel", 32, 16, 16, 4, bc, one, one);
}

std::string hierarchy(CutcellMG& mg) {
  std::string s = std::to_string(mg.nLevels()) + " levels, bottom ";
  const Level& b = mg.level(mg.nLevels() - 1);
  s += std::to_string(b.inner.x) + "x" + std::to_string(b.inner.y) + "x" +
       std::to_string(b.inner.z) + (mg.agglomerateBottom() ? " agglomerated" : " smoothed");
  return s;
}

// a on P with the switch on (gated) and off (recorded); lr: line relaxation; agglom: forced.
void symCase(const Problem& P, bool lr, bool agglom) {
  CutcellMG mg;
  setUp(mg, P, CutcellMG::kVcycleFp64);
  mg.setLineRelaxation(lr);
  if (agglom)
    mg.setAgglomerationMode(1);
  const std::string tag = P.name + (lr ? "" : " LR off");
  std::printf("%s: %s\n", tag.c_str(), hierarchy(mg).c_str());
  mg.setSymmetricVcycle(false);
  const double aOff = asym(mg, false);
  mg.setSymmetricVcycle(true);
  const double aOn = asym(mg, false);
  std::printf("  a: trilinear/forward %s (recorded), symmetric %s\n", num(aOff).c_str(),
              num(aOn).c_str());
  check(mg.agglomerateBottom() == agglom,
        tag + ": the bottom is " + (agglom ? "agglomerated" : "smoothed"));
  const double bound = agglom ? kBoundAgglomerated : kBoundSmoothed;
  check(aOn <= bound, tag + ": a " + num(aOn) + " <= " + num(bound));
}

// The guard: production PCG iterations with the switch on (and off, recorded).
void guardCase(const Problem& P, bool lr, int bound) {
  CutcellMG mg;
  setUp(mg, P, CutcellMG::kVcycleFp64);
  mg.setLineRelaxation(lr);
  const std::string tag = P.name + (lr ? "" : " LR off");
  mg.setSymmetricVcycle(false);
  const int off = pcgIters(mg, P);
  mg.setSymmetricVcycle(true);
  const int on = pcgIters(mg, P);
  std::printf(
      "%s: production PCG rhsOf rtol 1e-10: trilinear/forward %d (recorded), symmetric %d\n",
      tag.c_str(), off, on);
  check(on <= bound, tag + ": PCG " + std::to_string(on) + " <= " + std::to_string(bound));
}

// The FP32 V-cycle (forced 'fp32'): a recorded; on a device backend PCG on thinW not capped.
void fp32Case(const Problem& P) {
  CutcellMG mg;
  setUp(mg, P, CutcellMG::kVcycleFp32);
  mg.setSymmetricVcycle(true);
  if (const char* why = mg.fp32VcycleIneligible(true)) {
    std::printf("%s FP32: not eligible (%s) -- n/a\n", P.name.c_str(), why);
    check(CutcellMG::kHostMemory, P.name + " FP32: eligible on a device backend");
    return;
  }
  const double a = asym(mg, true);
  const int it = pcgIters(mg, P);
  std::printf("%s FP32 V-cycle (forced 'fp32', symmetric): a %s (recorded); PCG %d%s\n",
              P.name.c_str(), num(a).c_str(), it, mg.lastVcycleFp32() ? "" : " (not FP32)");
  if (!CutcellMG::kHostMemory)
    check(it < kCap && !mg.lastSolveFailed(), P.name + " FP32: PCG not capped (" +
                                                  std::to_string(it) + " < " +
                                                  std::to_string(kCap) + ")");
}

}  // namespace

int main(int argc, char** argv) {
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
    fp32Case(thinW);
    guardCase(thinW, true, 16);
    guardCase(thinW, false, 16);
    guardCase(slabP, true, 14);
  }
  Kokkos::finalize();
  std::printf("mg_vcycle_symmetry: %s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails,
              fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
