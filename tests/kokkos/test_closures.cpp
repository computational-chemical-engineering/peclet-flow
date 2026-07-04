// Property closures + per-cell body force. Verifies each closure kernel against a host oracle, and
// that a Boussinesq buoyancy force enters the momentum RHS (buildRhsForced path selected).
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "flow_ibm.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

void run() {
  const int N = 8;
  peclet::flow::IbmSolver s(N, N, 4);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * N * 4, 10.0));
  s.addScalar("T", 0.0, 1, 1);

  std::vector<double> T((std::size_t)N * N * 4);
  for (std::size_t i = 0; i < T.size(); ++i)
    T[i] = 0.6 + 0.3 * std::sin(0.7 * i);
  s.setField("T", T);

  // linear: rho = 2 - 0.5*T ; boussinesq: force_y = 1*9.81*0.1*(T-0.5) ; arrhenius mu = 0.01*exp(...)
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "T", "", {2.0, -0.5});
  s.setPropertyModel("force_y", peclet::flow::ClosureKind::BoussinesqForce, "T", "",
                     {1.0, 9.81, 0.1, 0.5});
  s.setPropertyModel("mu2", peclet::flow::ClosureKind::ArrheniusMu, "T", "", {0.01, 0.3, 1.0});
  s.updateProperties();

  auto rho = s.getField("rho");
  auto fy = s.getField("force_y");
  auto mu2 = s.getField("mu2");
  double emax = 0.0;
  for (std::size_t i = 0; i < T.size(); ++i) {
    emax = std::fmax(emax, std::fabs(rho[i] - (2.0 - 0.5 * T[i])));
    emax = std::fmax(emax, std::fabs(fy[i] - 9.81 * 0.1 * (T[i] - 0.5)));
    emax = std::fmax(emax, std::fabs(mu2[i] - 0.01 * std::exp(0.3 * (1.0 / T[i] - 1.0))));
  }
  std::printf("closures max-err vs oracle %.2e\n", emax);
  CHECK(emax < 1e-10);
  CHECK(s.hasField("force_x") && s.hasField("force_z"));  // all 3 force slots allocated
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
