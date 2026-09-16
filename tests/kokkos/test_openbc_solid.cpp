// Immersed solid TOGETHER with inflow/outflow domain boundary conditions -- the combination no
// test anywhere covered, and the two defects that hid in the gap (SCALING_ISSUES #3,
// doc/cutcell_openbc_convergence.md). One duct (-x inflow, +x outflow, y/z no-slip walls) and four
// beds: no solid, a sphere clear of both open faces, a sphere cut by the outlet, a sphere cut by
// the inlet. All four must converge and leave the PROJECTED open divergence at solver tolerance.
//
// Before the fix:
//   * the inlet-cut bed ran MG-PCG to its iteration cap with max|div| stuck at exactly the inflow
//     velocity -- the SDF ghost outside a non-periodic face was filled by PERIODIC WRAP, so a
//     solid cell against the inlet was handed the far side's fluid and came out fully OPEN, and
//     the prescribed inflow was pushed into a cell whose pressure row is entirely closed;
//   * the outlet-cut bed converged but was SILENTLY WRONG, because the Dirichlet outlet row
//     carried the literal openness 1.0 instead of the face's own aperture, so mass left through
//     solid. That half is pinned here by its own ablation (set_outflow_operator_coefficient).
//
// Note which divergence the gate reads: `maxOpenDivergence()` refills the outflow ghost with the
// zero-gradient extrapolation BEFORE measuring, so at a partly blocked outlet it reports how far
// zero-gradient is from the mass-conserving face -- a property of the boundary condition, not a
// solver residual, and it does NOT decay. `maxOpenDivergenceProjected()` measures the corrected
// face field and is the residual of the constraint the projection actually solved.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
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

constexpr int NX = 32, NY = 16, NZ = 16, STEPS = 8, PMAXIT = 200;
constexpr double U = 1.0, R = 4.3;

// Cell centres sit at integer indices, so the domain spans [-0.5, N-0.5] on each axis: a sphere
// centred at x = -0.5 is cut in half by the inlet plane, one at x = NX-0.5 by the outlet plane.
std::vector<double> sphereSdf(double cx) {
  std::vector<double> sdf((std::size_t)NX * NY * NZ);
  const double cy = 0.5 * (NY - 1), cz = 0.5 * (NZ - 1);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        const double dx = x - cx, dy = y - cy, dz = z - cz;
        sdf[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return sdf;
}

void configure(peclet::flow::IbmSolver& s) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(0.5);
  s.setAdvection(false);
  s.setVelocityIterations(200);
  s.setVelocityResidualTolerance(1e-10);
  s.setPressureLevels(3);
  s.setPressurePcg(true, PMAXIT, 1e-10);
  s.setDomainBc(0, 2, U, 0.0, 0.0);    // -x inflow
  s.setDomainBc(1, 3, 0.0, 0.0, 0.0);  // +x outflow
  for (int f = 2; f < 6; ++f)
    s.setDomainBc(f, 1, 0.0, 0.0, 0.0);  // y/z no-slip walls
}

struct Result {
  long iters;
  double divp;
};

// cx < 0 with solid == false selects the all-fluid control.
Result run(double cx, bool solid, bool outflowOpCoeff = true) {
  peclet::flow::IbmSolver s(NX, NY, NZ);
  configure(s);
  s.setOutflowOperatorCoefficient(outflowOpCoeff);
  if (solid)
    s.setSolid(sphereSdf(cx), /*cutcellPressure=*/true);
  else
    s.setPressureGeometry(std::vector<double>((std::size_t)NX * NY * NZ, 1e30));
  Result r{0, 0.0};
  for (int it = 0; it < STEPS; ++it) {
    s.step();
    r.iters = s.lastPressureIterations();
    r.divp = s.maxOpenDivergenceProjected();
  }
  return r;
}

// A single fluid cell against the inlet plane, walled in by solid on its other five faces: the
// pressure row is entirely closed while the inflow still feeds it. No pressure field satisfies
// that, so set_solid must REJECT the geometry rather than hand the solve an inconsistent row.
bool sealedPocketRejected() {
  peclet::flow::IbmSolver s(NX, NY, NZ);
  configure(s);
  std::vector<double> sdf((std::size_t)NX * NY * NZ, -1.0);  // solid everywhere ...
  const int py = NY / 2, pz = NZ / 2;
  sdf[(std::size_t)0 + (std::size_t)py * NX + (std::size_t)pz * NX * NY] = 1.0;  // ... but one cell
  for (int x = 4; x < NX; ++x)  // plus a fluid bulk downstream, so the rest of the run is sane
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < NY; ++y)
        sdf[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = 1.0;
  try {
    s.setSolid(sdf, /*cutcellPressure=*/true);
  } catch (const std::exception& e) {
    std::printf("[openbc-solid] sealed inlet pocket rejected: %s\n", e.what());
    return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const Result empty = run(0.0, false);
    const Result clear = run(0.5 * NX, true);
    const Result cutOut = run(NX - 0.5, true);
    const Result cutIn = run(-0.5, true);
    const char* fmt = "[openbc-solid] %-28s iters %3ld  projected max|div| %.3e\n";
    std::printf(fmt, "all-fluid duct:", empty.iters, empty.divp);
    std::printf(fmt, "sphere clear of the faces:", clear.iters, clear.divp);
    std::printf(fmt, "sphere cutting the OUTLET:", cutOut.iters, cutOut.divp);
    std::printf(fmt, "sphere cutting the INLET:", cutIn.iters, cutIn.divp);

    const Result all[4] = {empty, clear, cutOut, cutIn};
    for (const Result& r : all) {
      CHECK(r.iters < PMAXIT);  // the inlet-cut bed used to burn the cap on every step
      CHECK(r.divp < 1e-7);     // measured 3e-14 .. 6.2e-9
    }

    // The outlet half, pinned by its ablation: with the Dirichlet row back on the literal openness
    // 1.0 the operator disagrees with the divergence constraint by (1 - aperture), and the
    // projected divergence PLATEAUS four orders up instead of falling to solver tolerance.
    const Result ablated = run(NX - 0.5, true, /*outflowOpCoeff=*/false);
    std::printf("[openbc-solid] %-28s iters %3ld  projected max|div| %.3e (the defect)\n",
                "OUTLET, literal-1.0 row:", ablated.iters, ablated.divp);
    CHECK(ablated.divp > 1e-4);  // measured 6.7e-3

    CHECK(sealedPocketRejected());

    if (failures == 0)
      std::printf("[openbc-solid] OK\n");
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
