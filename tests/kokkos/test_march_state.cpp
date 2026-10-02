/// @file
/// @brief ctest `march_state`: Solver::marchState(), the state descriptor of the steady march
/// (doc/steady_acceleration.md §3.1, §4.1, §5.1, §5.3, rev 1; work order WO-2).
///
/// Checks, single rank, on both grids:
///   * the field list and roles of the three §3.1 rows — staggered (u, v, w Velocity; P Carried);
///     collocated 'ghost' without projected-face advection (the same four, also reached through the
///     AUTO scheme and through set_uf_advection(False)); collocated 'ghost' with advection and the
///     projected face field (+ uf, vf, wf Carried) — each field ALIASING the solver's own buffer
///     over the full padded box;
///   * innerTolerance in three configurations: the default (= the PCG rtol), an explicit velocity
///     residual tolerance (= that value), and tolerance 0 under Chebyshev (= the Chebyshev rtol);
///   * the internal signature (dt, rho, mu, F);
///   * that each §5.3 refusal throws std::runtime_error with its own message.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/geom/scene_builder.hpp"

using peclet::core::solver::AndersonRole;
using Stag = peclet::flow::Solver<peclet::flow::Staggered>;
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok)
    ++failures;
}

constexpr int N = 12;

std::vector<double> sphereSdf(double radius) {
  std::vector<double> sdf((std::size_t)N * N * N);
  const double c = 0.5 * N;
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x + 0.5 - c, dy = y + 0.5 - c, dz = z + 0.5 - c;
        sdf[(std::size_t)x + (std::size_t)N * (y + (std::size_t)N * z)] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
      }
  return sdf;
}

std::vector<double> allFluid() {
  return std::vector<double>((std::size_t)N * N * N, 1.0);
}

template <class S>
void basics(S& s) {
  s.setRho(1.0);
  s.setMu(1.0);
  s.setDt(6.0);
  s.setBodyForce(1e-3, 0.0, 0.0);
  s.setPressurePcg(true, 200, 1e-8);
}

/// Every field is the solver's own buffer (same data pointer), over the full padded box.
template <class S>
void checkFields(S& s, const typename S::MarchState& ms, const std::vector<std::string>& names,
                 int carriedFrom, const std::string& tag) {
  check(ms.fields.size() == names.size() && ms.roles.size() == names.size(),
        tag + ": " + std::to_string(names.size()) + " fields and roles");
  if (ms.fields.size() != names.size() || ms.roles.size() != names.size())
    return;
  const std::size_t nPad = (std::size_t)ms.e.x * ms.e.y * ms.e.z;
  check(ms.G == 2 && ms.e.x == N + 4 && ms.e.y == N + 4 && ms.e.z == N + 4,
        tag + ": e = n + 2G, G = 2");
  for (std::size_t f = 0; f < names.size(); ++f) {
    const AndersonRole want = (int)f < carriedFrom ? AndersonRole::Velocity : AndersonRole::Carried;
    const bool same = ms.fields[f].data() == s.fieldView(names[f]).data();
    check(ms.roles[f] == want && ms.fields[f].extent(0) == nPad && same,
          tag + ": field " + std::to_string(f) + " is '" + names[f] + "' (" +
              (want == AndersonRole::Velocity ? "Velocity" : "Carried") + ", padded)");
  }
}

void gateRows() {
  std::printf("rows of §3.1\n");
  {
    Stag s(N, N, N);
    basics(s);
    s.setAdvection(true);
    s.setSolid(sphereSdf(3.0), true);
    const auto ms = s.marchState();
    checkFields(s, ms, {"u", "v", "w", "p"}, 3, "staggered + advection");
    check(ms.signature[0] == 6.0 && ms.signature[1] == 1.0 && ms.signature[2] == 1.0 &&
              ms.signature[3] == 1e-3 && ms.signature[4] == 0.0 && ms.signature[5] == 0.0,
          "staggered: signature (dt, rho, mu, F) in cell units");
  }
  {
    Colo s(N, N, N);  // AUTO scheme, Stokes: resolves to 'ghost' at set_solid
    basics(s);
    s.setAdvection(false);
    s.setSolid(sphereSdf(3.0), true);
    checkFields(s, s.marchState(), {"u", "v", "w", "p"}, 3, "collocated AUTO ghost, Stokes");
  }
  {
    Colo s(N, N, N);
    basics(s);
    s.setCollocatedScheme("ghost");
    s.setAdvection(true);
    s.setUfAdvection(false);
    s.setSolid(sphereSdf(3.0), true);
    checkFields(s, s.marchState(), {"u", "v", "w", "p"}, 3,
                "collocated ghost + advection, set_uf_advection(False)");
  }
  {
    Colo s(N, N, N);
    basics(s);
    s.setCollocatedScheme("ghost");
    s.setAdvection(true);
    s.setSolid(sphereSdf(3.0), true);
    const auto ms = s.marchState();
    // uf/vf/wf are not in the field registry; compare them by position after u, v, w, p.
    auto first4 = ms;
    first4.fields.resize(std::min<std::size_t>(4, ms.fields.size()));
    first4.roles.resize(std::min<std::size_t>(4, ms.roles.size()));
    checkFields(s, first4, {"u", "v", "w", "p"}, 3,
                "collocated ghost + projected-face advection (first 4)");
    bool faces = ms.fields.size() == 7;
    for (std::size_t f = 4; faces && f < 7; ++f)
      faces = ms.roles[f] == AndersonRole::Carried &&
              ms.fields[f].extent(0) == (std::size_t)ms.e.x * ms.e.y * ms.e.z;
    if (faces) {
      bool distinct = true;  // three distinct buffers, none of them u, v, w or p
      for (std::size_t f = 4; f < 7; ++f)
        for (std::size_t g = 0; g < f; ++g)
          distinct = distinct && ms.fields[f].data() != ms.fields[g].data();
      faces = distinct;
    }
    check(faces, "collocated ghost + projected-face advection: + uf, vf, wf (Carried, padded)");
  }
}

void gateTolerance() {
  std::printf("innerTolerance (§4.1)\n");
  {
    Stag s(N, N, N);
    basics(s);  // PCG rtol 1e-8, velocity residual tolerance default (< 0: follows the driver)
    check(s.marchState().innerTolerance == 1e-8, "default: the PCG rtol (1e-8)");
  }
  {
    Stag s(N, N, N);
    basics(s);
    s.setVelocityResidualTolerance(1e-9);
    check(s.marchState().innerTolerance == 1e-9, "explicit velocity residual tolerance 1e-9");
  }
  {
    Stag s(N, N, N);
    basics(s);
    s.setPressureChebyshev(true, 120, 3e-7);
    s.setVelocityResidualTolerance(0.0);
    check(s.marchState().innerTolerance == 3e-7, "tolerance 0 under Chebyshev: the Chebyshev rtol");
  }
}

/// `make` builds a solver with exactly one refused feature; marchState() must throw a message
/// containing `token` (its own refusal) and the hint.
template <class S, class Make>
void refusal(const std::string& tag, const std::string& token, Make make) {
  S s(N, N, N);
  basics(s);
  std::string what;
  try {
    make(s);
    (void)s.marchState();
  } catch (const std::runtime_error& e) {
    what = e.what();
  }
  const bool own = what.find(token) != std::string::npos;
  const bool hint = what.find("pass accelerate=False") != std::string::npos;
  check(own && hint, "refusal " + tag + ": \"" + what + "\"");
}

void sphereScene(Stag& s, double radius) {
  namespace g = peclet::core::geom;
  std::vector<int> ni(g::kNodeIntStride, 0), ii(g::kInstanceIntStride, 0);
  std::vector<double> nr(g::kNodeRealStride, 0.0), ir(g::kInstanceRealStride, 0.0);
  ni[0] = g::kSphere;
  ni[1] = -1;
  ni[2] = -1;
  nr[0] = radius;
  nr[14] = 1.0;
  nr[15] = 1.0;
  ii[0] = 0;
  ii[1] = -1;
  ir[0] = ir[1] = ir[2] = 0.5 * N;
  ir[6] = 1.0;
  ir[7] = 1.0;
  s.setScene(ni, nr, ii, ir, /*periodic=*/true);
  s.setSolidFromScene(true);
}

void gateRefusals() {
  std::printf("refusals (§5.3)\n");
  refusal<Colo>("1 collocated gauge-exact", "not the fluid-only 'ghost' projection", [](Colo& s) {
    s.setCollocatedScheme("gauge-exact");
    s.setSolid(sphereSdf(3.0), true);
  });
  refusal<Colo>("1 collocated embed", "not the fluid-only 'ghost' projection", [](Colo& s) {
    s.setCollocatedScheme("embed");
    s.setSolid(sphereSdf(3.0), true);
  });
  refusal<Stag>("2 VoF", "VoF", [](Stag& s) {
    s.setPressureGeometry(allFluid());
    s.enableVof();
  });
  refusal<Stag>("3 phase change", "phase change", [](Stag& s) {
    s.setPressureGeometry(allFluid());
    s.enablePhaseChange(1.0, 10.0, 1.0);
  });
  refusal<Stag>("4 scalar", "transported scalar", [](Stag& s) { s.addScalar("T", 0.1, 0, 1); });
  refusal<Stag>("5 porous", "porous", [](Stag& s) { s.setPorousContinuity(true); });
  refusal<Stag>("6 variable rho (closure)", "variable density or viscosity", [](Stag& s) {
    s.addField("phase");
    s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "phase", "", {1.0, 0.0});
  });
  refusal<Stag>("6 mu field", "variable density or viscosity", [](Stag& s) { s.addField("mu"); });
  refusal<Stag>("7 drag", "drag_beta", [](Stag& s) { s.enableDrag(); });
  refusal<Stag>("8 cell force", "cell force", [](Stag& s) { s.enableCellForce(); });
  refusal<Stag>("9 moving scene", "moving scene", [](Stag& s) {
    sphereScene(s, 3.0);
    s.setInstanceMotion(0, {1e-3, 0.0, 0.0}, {0.0, 0.0, 0.0}, nullptr);
  });
  refusal<Stag>("10 superficial velocity", "set_superficial_velocity",
                [](Stag& s) { s.setSuperficialVelocity(true, 0, 1e-3); });
  refusal<Stag>("11 pressure warm start", "warm start",
                [](Stag& s) { s.setPressureWarmstart(true); });
  refusal<Stag>("12 balanced-force projection", "balanced-force projection",
                [](Stag& s) { s.setBalancedForceProjection(true); });
  // A static scene is allowed.
  {
    Stag s(N, N, N);
    basics(s);
    sphereScene(s, 3.0);
    bool ok = true;
    try {
      (void)s.marchState();
    } catch (const std::exception&) {
      ok = false;
    }
    check(ok, "a static scene is allowed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    gateRows();
    gateTolerance();
    gateRefusals();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "\n%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("\nAll march-state checks passed.\n");
  return 0;
}
