// WO-V9 — the VoF profiling instrument is INERT.
//
// `set_vof_timing(True)` puts a `Kokkos::fence()` at every VoF stage boundary (the same rule
// `IbmSolver::phaseTick` already applies to the step's three coarse phases: on a device backend a
// phase boundary that does not fence bills queued work to whichever phase next reads the clock).
// A fence changes WHEN work happens and never WHAT it computes — but "never" is a claim, and this
// is the gate on it.  `useWorklist` is the same kind of claim about a re-ordering: the compacted
// reconstruction pass writes planes for exactly the cells the dense pass writes them for, and the
// flux never reads a non-mixed cell's plane, so the two must agree BIT FOR BIT.
//
//   T1  timers OFF vs ON, surface tension + momentum consistency, 30 steps: C, u, v, w, P bitwise.
//   T2  the same, through the CUT-CELL path (a sphere array, `set_solid(..., cutcell_pressure)`):
//       bitwise.  The V5a kernels are a different branch and carry the clip/census timers.
//   T3  `set_vof_worklist(False)` vs the default, both scenes: bitwise (the compaction claim).
//   T4  the timers actually MEASURE: with them armed the stage sum is > 0, the kernel breakdown
//       sums to no more than the colour stage, and `steps` counts the armed steps.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

// A NaN-propagating max|a-b| (the WO-R2 sweep: std::fmax(m, NaN) == m would hide a dead field).
double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return 1e300;
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(a[i] - b[i]);
    if (!(d <= m))  // false for NaN, which therefore becomes the answer
      m = d;
  }
  return m;
}

struct State {
  std::vector<double> C, u, v, w, P;
};

State harvest(IbmSolver& s) {
  State st;
  st.C = s.getField("C");
  st.u = s.getVelocity(0);
  st.v = s.getVelocity(1);
  st.w = s.getVelocity(2);
  st.P = s.getPressure();
  return st;
}

double diff(const State& a, const State& b) {
  double m = maxAbsDiff(a.C, b.C);
  m = std::fmax(m, maxAbsDiff(a.u, b.u));
  m = std::fmax(m, maxAbsDiff(a.v, b.v));
  m = std::fmax(m, maxAbsDiff(a.w, b.w));
  m = std::fmax(m, maxAbsDiff(a.P, b.P));
  return m;
}

// A rising drop in a periodic box: surface tension + variable density + momentum consistency, i.e.
// every VoF stage the instrument touches (advection, curvature, CSF, the bridges).
std::unique_ptr<IbmSolver> makeScene(int n, bool solid, bool timing, bool worklist) {
  auto s = std::make_unique<IbmSolver>(n, n, n);
  s->setRho(10.0);
  s->setMu(0.05);
  const double R = 0.3 * n;
  std::vector<double> sdf((std::size_t)n * n * n, 10.0);
  if (solid) {  // a 2x2x2 sphere array the interface has to move through
    const double Rs = 0.16 * n;
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
          double best = 1e30;
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
              for (int c = 0; c < 2; ++c) {
                const double cx = (0.25 + 0.5 * a) * n, cy = (0.25 + 0.5 * b) * n,
                             cz = (0.25 + 0.5 * c) * n;
                const double dx = i + 0.5 - cx, dy = j + 0.5 - cy, dz = k + 0.5 - cz;
                best = std::fmin(best, std::sqrt(dx * dx + dy * dy + dz * dz) - Rs);
              }
          sdf[(std::size_t)i + (std::size_t)j * n + (std::size_t)k * n * n] = best;
        }
    s->setSolid(sdf, true);
  } else {
    s->setPressureGeometry(sdf);
  }
  s->enableVof();
  s->setVofWorklist(worklist);
  s->setVofCurvatureWorklist(worklist);
  std::vector<double> C((std::size_t)n * n * n, 0.0);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double dx = i + 0.5 - 0.5 * n - 0.13, dy = j + 0.5 - 0.5 * n - 0.27,
                     dz = k + 0.5 - 0.5 * n - 0.11;
        // a smooth (but not symmetric) blob: fractions between 0 and 1 in a real band
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        C[(std::size_t)i + (std::size_t)j * n + (std::size_t)k * n * n] =
            std::fmin(1.0, std::fmax(0.0, 0.5 - (r - R)));
      }
  s->setField("C", C);
  s->setPropertyModel("rho", ClosureKind::LinearMix, "C", "", {1.0, 9.0});
  s->setPropertyModel("mu", ClosureKind::LinearMix, "C", "", {0.005, 0.045});
  s->setSurfaceTension(2.0);
  s->enableVofMomentum(1.0, 10.0);
  s->setPropertyModel("force_z", ClosureKind::LinearMix, "C", "", {-0.01, -0.09});
  s->setPressureChebyshev(true, 300, 1e-12);
  s->setDt(0.4 * s->capillaryDt());
  if (timing)
    s->setVofTiming(true);
  return s;
}

State run(int n, bool solid, bool timing, bool worklist, int steps) {
  auto s = makeScene(n, solid, timing, worklist);
  for (int i = 0; i < steps; ++i)
    s->step();
  return harvest(*s);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int n = 24, steps = 30;
    for (int solid = 0; solid < 2; ++solid) {
      const char* tag = solid ? "cut-cell" : "all-fluid";
      const State off = run(n, solid, false, true, steps);
      const State on = run(n, solid, true, true, steps);
      const double d1 = diff(off, on);
      std::printf("T1/T2 %-9s  timers OFF vs ON            max|d| = %.3e\n", tag, d1);
      CHECK(d1 == 0.0);
      const State nowl = run(n, solid, false, false, steps);
      const double d2 = diff(off, nowl);
      std::printf("T3    %-9s  worklists ON vs OFF         max|d| = %.3e\n", tag, d2);
      CHECK(d2 == 0.0);
    }
    // T4: the instrument reports something, and the parts do not exceed the whole.
    auto s = makeScene(n, false, true, true);
    for (int i = 0; i < 10; ++i)
      s->step();
    const auto& v = s->vofTimingReport();
    const auto& k = s->vofKernelTiming();
    const auto& q = s->vofCurvatureTiming();
    std::printf("T4    steps %ld  momAdvect %.4f s  curvature %.4f s  csf %.4f s  "
                "[recon %.4f fluxes %.4f sweep %.4f exch %.4f]\n",
                v.steps, v.momAdvect, v.curvature, v.csf, k.reconstruct, k.fluxes, k.sweep,
                k.exchange);
    CHECK(v.steps == 10);
    CHECK(v.momAdvect > 0.0);
    CHECK(v.curvature > 0.0);
    CHECK(v.csf > 0.0);
    CHECK(k.sweeps == 30);  // 3 sweeps per step
    CHECK(k.reconstruct > 0.0 && k.fluxes > 0.0 && k.sweep > 0.0);
    // the kernel breakdown is a part of the momentum-consistent colour stage
    CHECK(k.reconstruct + k.fluxes + k.sweep + k.clip + k.exchange + k.freeze <=
          v.momAdvect * 1.02 + 1e-6);
    std::printf("T4    curvature passes: compact %.4f planes %.4f height %.4f fallback %.4f "
                "census %.4f (%ld calls)\n",
                q.compact, q.planes, q.height, q.fallback, q.census, q.calls);
    CHECK(q.calls == 10);
    CHECK(q.planes > 0.0 && q.height > 0.0);
    CHECK(q.compact + q.planes + q.height + q.fallback + q.census <= v.curvature * 1.02 + 1e-6);
    s->setVofTiming(false);
    const auto& z = s->vofTimingReport();
    CHECK(z.steps == 0 && z.momAdvect == 0.0);
  }
  Kokkos::finalize();
  std::printf(failures ? "FAILED (%d)\n" : "OK (%d failures)\n", failures);
  return failures ? 1 : 0;
}
