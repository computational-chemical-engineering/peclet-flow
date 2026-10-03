// G-perf instrument for the cut-cell scalar (doc/scalar_ibm_design.md §11 G-perf; WO-9). An
// instrument, not a gate (ctest label `bench`): it prints, it never fails on a number.
//
//   stream  Kokkos triad a = b + s c on the default execution space, 2^26 doubles, best of 10;
//           24 bytes per element.
//   memory  bytes allocated in the default memory space from addScalar through the first steady
//           solve of ONE single-phase cut-cell scalar — the scalar field, the cut state, the
//           ScalarMG hierarchy and the Krylov vectors — per inner cell, live at the end and peak
//           (Kokkos Tools callbacks); the geometry record (one per solver, shared by every
//           cut-cell scalar) is built first and reported separately.
//   matvec  Solver::scalarCutMatvec (the G = 2 ghost fill + the 7-point band operator + the facet
//           overlay), best of `reps`; the effective bandwidth on the byte model 9 doubles per
//           extended cell (AC and six bands read, x read, y written; the overlay excluded), so a
//           lower bound on the traffic, against the triad.
//
// Setup: an n^3 periodic box (cell units), one sphere of solid fraction 0.3 (G5b's simple-cubic
// array, one period), Dirichlet walls c = 1, a uniform sink; steady.
// Usage: bench_scalar_cutcell [n = 128] [reps = 50]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using IbmSolver = peclet::flow::IbmSolver;
using CCField = peclet::flow::CCField;
using Clock = std::chrono::steady_clock;

// ---- Kokkos Tools allocation ledger (default memory space only) ----
std::map<const void*, std::uint64_t> gLive;
std::uint64_t gBytes = 0, gPeak = 0;
bool gArmed = false;

bool defaultSpace(const Kokkos::Profiling::SpaceHandle h) {
  return std::strcmp(h.name, Kokkos::DefaultExecutionSpace::memory_space::name()) == 0;
}
void onAlloc(const Kokkos::Profiling::SpaceHandle h, const char*, const void* p,
             const std::uint64_t n) {
  if (!gArmed || !defaultSpace(h))
    return;
  gLive[p] = n;
  gBytes += n;
  gPeak = gBytes > gPeak ? gBytes : gPeak;
}
void onFree(const Kokkos::Profiling::SpaceHandle h, const char*, const void* p,
            const std::uint64_t) {
  if (!defaultSpace(h))
    return;
  auto it = gLive.find(p);
  if (it == gLive.end())
    return;  // allocated before the ledger was armed
  gBytes -= it->second;
  gLive.erase(it);
}

double seconds(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double streamTriad() {
  const std::size_t n = std::size_t(1) << 26;
  Kokkos::View<double*> a("a", n), b("b", n), c("c", n);
  Kokkos::deep_copy(b, 1.0);
  Kokkos::deep_copy(c, 2.0);
  double best = 1e30;
  for (int r = 0; r < 11; ++r) {
    Kokkos::fence();
    const auto t0 = Clock::now();
    Kokkos::parallel_for(
        "bench_triad", Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) { a(i) = b(i) + 0.5 * c(i); });
    Kokkos::fence();
    if (r > 0)
      best = std::fmin(best, seconds(t0));
  }
  return 24.0 * (double)n / best / 1e9;
}

std::vector<double> sphereSdf(int n) {
  const double R = std::cbrt(0.3 * 3.0 / (4.0 * M_PI)) * n;
  const double c[3] = {0.513 * n, 0.479 * n, 0.507 * n};
  std::vector<double> s((std::size_t)n * n * n);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double dx = x + 0.5 - c[0], dy = y + 0.5 - c[1], dz = z + 0.5 - c[2];
        s[(std::size_t)x + (std::size_t)y * n + (std::size_t)z * n * n] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int n = argc > 1 ? std::atoi(argv[1]) : 128;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    Kokkos::Tools::Experimental::set_allocate_data_callback(onAlloc);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(onFree);

    const double bwStream = streamTriad();
    std::printf("G-perf backend %s, n = %d\n", Kokkos::DefaultExecutionSpace::name(), n);
    std::printf("  stream triad: %.1f GB/s\n", bwStream);

    IbmSolver s(n, n, n);
    s.setRho(1.0);
    s.setMu(1.0);
    s.setDt(1.0);
    s.setSolid(sphereSdf(n), true);
    Kokkos::fence();
    const double cells = (double)n * n * n;
    gArmed = true;
    const auto t0 = Clock::now();
    (void)s.scalarCutGeometry();  // the geometry record: built once per solver, shared by scalars
    Kokkos::fence();
    const double tGeom = seconds(t0);
    const std::uint64_t geomBytes = gBytes;
    gPeak = gBytes = 0;
    gLive.clear();  // the record stays allocated; the scalar's own bytes from here
    const auto t1 = Clock::now();
    s.addScalar("c", 0.7, 1, 50, true);
    s.setScalarWall("c", 1, 1.0, 0.0, -1);
    s.setScalarSource("c", -0.3);
    s.solveScalarSteady("c");
    Kokkos::fence();
    const double tSolve = seconds(t1);
    gArmed = false;
    auto& sc = s.scalarField("c");
    std::printf(
        "  memory: geometry record (per solver, shared) %.0f B/cell; one single-phase scalar "
        "(steady, %d MG levels) live %.0f B/cell, peak %.0f B/cell (bound 300)\n",
        (double)geomBytes / cells, sc.cut->mgLevels, (double)gBytes / cells, (double)gPeak / cells);
    std::printf("  geometry record build: %.3f s\n", tGeom);
    std::printf("  first steady solve (assembly + MG build + %d iterations): %.3f s\n",
                sc.cut->iterations, tSolve);

    const std::size_t ne = sc.c.extent(0);
    CCField x("bx", ne), y("by", ne);
    Kokkos::deep_copy(x, sc.c);
    s.scalarCutMatvec(sc, y, x);  // warm-up
    Kokkos::fence();
    double best = 1e30, sum = 0.0;
    for (int r = 0; r < reps; ++r) {
      const auto t1 = Clock::now();
      s.scalarCutMatvec(sc, y, x);
      Kokkos::fence();
      const double t = seconds(t1);
      best = std::fmin(best, t);
      sum += t;
    }
    const double bw = 9.0 * 8.0 * (double)ne / best / 1e9;
    std::printf(
        "  matvec: best %.3f ms, mean %.3f ms; effective %.1f GB/s on 72 B per extended cell = "
        "%.0f %% of the triad (red flag < 50 %%)\n",
        1e3 * best, 1e3 * sum / reps, bw, 100.0 * bw / bwStream);
  }
  Kokkos::finalize();
  return 0;
}
