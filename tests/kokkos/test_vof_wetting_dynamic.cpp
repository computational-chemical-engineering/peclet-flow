// VoF rung V6 (WO-V6) — the DYNAMIC contact angle (Cox-Voinov with an explicit slip length) and
// advancing/receding HYSTERESIS, as a PURE KERNEL test plus the one end-to-end check the model
// itself needs.
//
// G1a  THE MODEL, as arithmetic. `coxVoinovAngle` reproduces theta^3 = theta_e^3 + 9 Ca ln(1/lambda)
//      to round-off, is monotone in Ca (advancing RAISES the angle, receding LOWERS it), and the
//      clamp catches the film-entrainment branch where the cube goes non-positive.
// G1b  THE SIGN CONVENTION, as geometry. `vofWallTangent` + `vofContactLineSpeed` on a vertical
//      interface: the liquid ADVANCES when the flow points from the liquid towards the dry wall,
//      i.e. along +t_hat (the work order says -t_hat; see the header of `wetting_dynamic.hpp`).
// G1c  THE MODEL THROUGH THE SOLVER, kinematically. A flat SDF wall at a QUARTER-integer z with a
//      liquid slab and a UNIFORM wall-tangential velocity U: the box is periodic in x, so the slab
//      has two contact lines with opposite t_hat, one ADVANCING and one RECEDING in the SAME run.
//      Gate: |U_cl| == U exactly, the signs opposite and correct, and the imposed angle equal to
//      the Cox-Voinov value of the reported U_cl to 1e-10, cell by cell.
// G4a  THE HYSTERESIS SELECTOR, as a truth table, and end-to-end: with theta_r <= 90 <= theta_a the
//      vertical interface is PINNED and the imposed angle is the apparent one; outside the window
//      the advancing / receding angle is imposed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "vof/wetting_dynamic.hpp"

namespace {
using namespace peclet::flow::vof;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

const double kDeg = M_PI / 180.0;

double coxRef(double thetaEdeg, double ca, double slip) {
  const double te = thetaEdeg * kDeg;
  const double t3 = te * te * te + 9.0 * ca * (-std::log(slip));
  const double th = t3 > 0.0 ? std::cbrt(t3) : 1.0 * kDeg;
  return std::fmin(std::fmax(th, 1.0 * kDeg), 179.0 * kDeg);
}

// ------------------------------------------------------------------------------------------ G1a
void modelArithmetic() {
  std::printf(
      "G1a Cox-Voinov (Afkhami, Zaleski & Bussmann, JCP 228:5370): "
      "theta^3 = theta_e^3 + 9 Ca ln(Delta/lambda)\n");
  const double slip = 0.1, lr = -std::log(slip);
  double worst = 0.0;
  for (double thE : {30.0, 60.0, 90.0, 120.0}) {
    for (double ca : {-2e-2, -5e-3, 0.0, 5e-3, 2e-2, 5e-2}) {
      const double th = coxVoinovAngle(thE * kDeg, ca, lr, 1.0 * kDeg, 179.0 * kDeg);
      const double ref = coxRef(thE, ca, slip);
      worst = std::fmax(worst, std::fabs(th - ref));
      // the identity itself, wherever the clamp did not fire (at theta_e = 30 and Ca = -2e-2 the
      // cube goes non-positive — the film-entrainment branch — and the clamp is the answer)
      if (th > 1.0 * kDeg && th < 179.0 * kDeg) {
        const double id = th * th * th - (thE * kDeg) * (thE * kDeg) * (thE * kDeg) - 9.0 * ca * lr;
        CHECK(std::fabs(id) <= 1e-14);
      }
    }
    const double up = coxVoinovAngle(thE * kDeg, +1e-2, lr, 1.0 * kDeg, 179.0 * kDeg);
    const double dn = coxVoinovAngle(thE * kDeg, -1e-2, lr, 1.0 * kDeg, 179.0 * kDeg);
    std::printf("   theta_e %5.1f  ->  advancing (Ca=+1e-2) %7.3f deg,  receding (Ca=-1e-2) %7.3f\n",
                thE, up / kDeg, dn / kDeg);
    CHECK(up > thE * kDeg && dn < thE * kDeg);
  }
  // the film-entrainment branch: the cube goes non-positive and the clamp catches it
  const double ent = coxVoinovAngle(30.0 * kDeg, -1.0, lr, 1.0 * kDeg, 179.0 * kDeg);
  std::printf("   worst |kernel - host| over the sweep %.3e ; the Ca = -1 (film entrainment) "
              "branch clamps to %.3f deg\n",
              worst, ent / kDeg);
  CHECK(worst <= 1e-15);
  CHECK(std::fabs(ent - 1.0 * kDeg) <= 1e-15);
}

// ------------------------------------------------------------------------------------------ G1b
void signConvention() {
  std::printf("G1b the sign of U_cl (n_w = +z; the liquid ADVANCES along +t_hat)\n");
  const double nw[3] = {0.0, 0.0, 1.0};
  double worst = 0.0;
  for (double thApp : {45.0, 90.0, 135.0})
    for (double psi : {0.0, 37.0, 90.0, 200.0}) {
      const double mf[3] = {std::sin(thApp * kDeg) * std::cos(psi * kDeg),
                            std::sin(thApp * kDeg) * std::sin(psi * kDeg),
                            std::cos(thApp * kDeg)};
      double that[3], cosApp;
      const bool ok = vofWallTangent(mf, nw, 1e-6, that, cosApp);
      CHECK(ok);
      worst = std::fmax(worst, std::fabs(std::acos(cosApp) - thApp * kDeg));
      // t_hat is the in-wall azimuth of m, i.e. points from the LIQUID towards the DRY wall
      CHECK(std::fabs(that[0] - std::cos(psi * kDeg)) <= 1e-14);
      CHECK(std::fabs(that[1] - std::sin(psi * kDeg)) <= 1e-14);
      CHECK(std::fabs(that[2]) <= 1e-14);
      // a flow ALONG t_hat pushes liquid onto the dry wall: advancing, U_cl > 0
      const double uAdv[3] = {0.3 * that[0], 0.3 * that[1], 0.0};
      const double uRec[3] = {-0.3 * that[0], -0.3 * that[1], 0.0};
      CHECK(std::fabs(vofContactLineSpeed(uAdv, that) - 0.3) <= 1e-15);
      CHECK(std::fabs(vofContactLineSpeed(uRec, that) + 0.3) <= 1e-15);
    }
  std::printf("   theta_app recovered from m_f . n_w to %.3e rad; U_cl = +u.t_hat verified over "
              "3 angles x 4 azimuths\n",
              worst);
  CHECK(worst <= 1e-14);
  // an interface parallel to the wall has no contact-line direction
  const double mpar[3] = {0.0, 0.0, 1.0};
  double tp[3], cp;
  CHECK(!vofWallTangent(mpar, nw, 1e-6, tp, cp));
}

// ------------------------------------------------------------------------------------------ G4a
void hysteresisTable() {
  std::printf("G4a the hysteresis selector (theta_a = 70, theta_r = 50)\n");
  const double ta = 70.0 * kDeg, tr = 50.0 * kDeg, lr = -std::log(0.1);
  struct Row {
    double app, ca;
    int want;
  };
  const Row rows[5] = {{80.0, 0.0, kVofDynAdvancing},
                       {60.0, 0.0, kVofDynPinned},
                       {40.0, 0.0, kVofDynReceding},
                       {80.0, 1e-2, kVofDynAdvancing},
                       {40.0, -1e-2, kVofDynReceding}};
  for (const Row& r : rows) {
    double th, ca;
    // Ca is fed through mu/sigma: mu = r.ca, sigma = 1, U_cl = 1
    const int st = vofDynamicContactAngle(r.app * kDeg, 1.0, r.ca, 1.0, lr, true, ta, tr,
                                          60.0 * kDeg, 1.0 * kDeg, 179.0 * kDeg, th, ca);
    const char* nm[5] = {"none", "static", "PINNED", "advancing", "receding"};
    std::printf("   theta_app %5.1f  Ca %+8.1e  ->  %-9s  theta_imposed %7.3f deg\n", r.app, r.ca,
                nm[st], th / kDeg);
    CHECK(st == r.want);
    if (r.want == kVofDynPinned)
      CHECK(std::fabs(th - r.app * kDeg) <= 1e-15);  // the fill is idempotent: nothing moves
    else {
      const double base = r.want == kVofDynAdvancing ? 70.0 : 50.0;
      CHECK(std::fabs(th - coxRef(base, r.ca, 0.1)) <= 1e-15);
    }
  }
  // with no hysteresis the static base is always the Cox-Voinov base
  double th, ca;
  const int st = vofDynamicContactAngle(80.0 * kDeg, 1.0, 1e-2, 1.0, lr, false, ta, tr, 60.0 * kDeg,
                                        1.0 * kDeg, 179.0 * kDeg, th, ca);
  CHECK(st == kVofDynStatic);
  CHECK(std::fabs(th - coxRef(60.0, 1e-2, 0.1)) <= 1e-15);
}

// --------------------------------------------------------------------------------------- G1c/G4b
// The slit: solid below z = ZW and above NZ - ZW, so the SDF is continuous across the periodic z
// wrap. ZW is a QUARTER-integer (WO-S finding 5: at a half-integer the wall cell's tangential MAC
// faces sit ON the SDF zero level and close, and the contact line cannot move at all).
constexpr int NX = 32, NY = 8, NZ = 24;
constexpr double ZW = 4.25;
constexpr double XL = 0.5, XR = 16.5;  // the liquid slab, in x

std::vector<double> slitSdf() {
  std::vector<double> v((std::size_t)NX * NY * NZ);
  for (int z = 0; z < NZ; ++z) {
    const double zc = z + 0.5, d = std::fmin(zc - ZW, (NZ - ZW) - zc);
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x)
        v[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = d;
  }
  return v;
}
std::vector<double> slabColour() {
  std::vector<double> v((std::size_t)NX * NY * NZ);
  for (int x = 0; x < NX; ++x) {
    const double lo = std::fmax((double)x, XL), hi = std::fmin((double)x + 1.0, XR);
    const double c = std::fmax(hi - lo, 0.0);
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < NY; ++y)
        v[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY] = c;
  }
  return v;
}

void solverGates() {
  const double thE = 60.0, slip = 0.1, muL = 1.0, sigma = 1.0, U = 0.02;
  std::printf(
      "G1c the model through the solver, KINEMATIC (%dx%dx%d, flat SDF wall at z = %.2f, liquid "
      "slab x in [%.1f, %.1f), uniform u = %.3g, theta_e = %.0f, lambda = %.2f cells, mu_l = %.1f, "
      "sigma = %.1f)\n",
      NX, NY, NZ, ZW, XL, XR, U, thE, slip, muL, sigma);
  peclet::flow::IbmSolver s(NX, NY, NZ);
  s.setRho(1.0);
  s.setMu(muL);
  s.setSolid(slitSdf(), true);
  s.enableVof();
  s.setVof(slabColour());
  s.setVelocity(0, std::vector<double>((std::size_t)NX * NY * NZ, U));
  s.setContactAngleDynamic(thE, slip, muL, sigma);

  const auto imposed = s.getVofDynamicField(0);   // degrees
  const auto apparent = s.getVofDynamicField(1);  // degrees
  const auto ucl = s.getVofDynamicField(2);
  const auto state = s.getVofDynamicField(4);
  const auto at = [&](const std::vector<double>& v, int x, int y, int z) {
    return v[(std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY];
  };
  // the two contact columns, in the first solid row below the wall (z = 3)
  const int zb = 3;
  std::printf(
      "   x = 16 (m = +x, liquid on the LOW side): U_cl %+.6f  theta_app %.4f  theta_imposed "
      "%.4f\n"
      "   x =  0 (m = -x, liquid on the HIGH side): U_cl %+.6f  theta_app %.4f  theta_imposed "
      "%.4f\n",
      at(ucl, 16, 4, zb), at(apparent, 16, 4, zb), at(imposed, 16, 4, zb), at(ucl, 0, 4, zb),
      at(apparent, 0, 4, zb), at(imposed, 0, 4, zb));
  CHECK(std::fabs(at(ucl, 16, 4, zb) - U) <= 1e-12);
  CHECK(std::fabs(at(ucl, 0, 4, zb) + U) <= 1e-12);
  CHECK(std::fabs(at(apparent, 16, 4, zb) - 90.0) <= 1e-9);
  const double adv = coxRef(thE, muL * U / sigma, slip) / kDeg;
  const double rec = coxRef(thE, -muL * U / sigma, slip) / kDeg;
  std::printf(
      "   Cox-Voinov reference: advancing %.4f deg, receding %.4f deg (theta_e %.1f)  ->  "
      "advancing RAISES and receding LOWERS the imposed angle\n",
      adv, rec, thE);
  CHECK(std::fabs(at(imposed, 16, 4, zb) - adv) <= 1e-10);
  CHECK(std::fabs(at(imposed, 0, 4, zb) - rec) <= 1e-10);
  CHECK(adv > thE && rec < thE);

  // the whole field: every contact cell's imposed angle IS the Cox-Voinov value of its own U_cl
  double worst = 0.0;
  long nc = 0;
  for (std::size_t i = 0; i < imposed.size(); ++i) {
    if (state[i] < 0.5)
      continue;
    ++nc;
    const double ref = coxRef(thE, muL * ucl[i] / sigma, slip) / kDeg;
    worst = std::fmax(worst, std::fabs(imposed[i] - ref));
  }
  const auto cd = s.contactAngleDiagnostics();
  std::printf(
      "   over all %ld contact cells: max |theta_imposed - CoxVoinov(U_cl)| = %.3e deg "
      "(gate 1e-10)\n"
      "   diagnostics: dynamic %ld  pinned %ld  advancing %ld  receding %ld  mean imposed %.4f  "
      "mean apparent %.4f  max|Ca_cl| %.4e  max|U_cl| %.4e\n",
      nc, worst, cd.dynamicCells, cd.pinnedCells, cd.advancingCells, cd.recedingCells,
      cd.meanImposedTheta, cd.meanApparentTheta, cd.maxCaCl, cd.maxContactSpeed);
  CHECK(worst <= 1e-10);
  CHECK(nc > 0);
  CHECK(std::fabs(cd.maxCaCl - muL * U / sigma) <= 1e-12);

  // --- G4b hysteresis, end to end -------------------------------------------------------------
  std::printf("G4b hysteresis through the solver, same scene at REST (U_cl = 0 by mu_l -> Ca = 0)\n");
  struct HRow {
    double a, r;
    const char* what;
  };
  const HRow hrows[3] = {{120.0, 60.0, "90 inside [60,120]  -> PINNED"},
                         {70.0, 50.0, "90 above theta_a=70  -> advancing"},
                         {130.0, 110.0, "90 below theta_r=110 -> receding"}};
  for (const HRow& h : hrows) {
    peclet::flow::IbmSolver s2(NX, NY, NZ);
    s2.setRho(1.0);
    s2.setMu(muL);
    s2.setSolid(slitSdf(), true);
    s2.enableVof();
    s2.setVof(slabColour());
    s2.setContactAngle(thE);
    s2.setContactAngleHysteresis(h.a, h.r);
    const auto im = s2.getVofDynamicField(0);
    const auto st = s2.getVofDynamicField(4);
    const auto ap = s2.getVofDynamicField(1);
    const auto d2 = s2.contactAngleDiagnostics();
    std::printf(
        "   theta_a %5.1f theta_r %5.1f : %s | imposed at (16,4,3) %.4f (apparent %.4f, state "
        "%.0f) | pinned %ld advancing %ld receding %ld\n",
        h.a, h.r, h.what, at(im, 16, 4, zb), at(ap, 16, 4, zb), at(st, 16, 4, zb), d2.pinnedCells,
        d2.advancingCells, d2.recedingCells);
    const double want = (h.a >= 90.0 && h.r <= 90.0) ? at(ap, 16, 4, zb)
                                                     : (90.0 > h.a ? h.a : h.r);
    CHECK(std::fabs(at(im, 16, 4, zb) - want) <= 1e-9);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Kokkos::initialize(argc, argv);
  {
    modelArithmetic();
    signConvention();
    hysteresisTable();
    if (!std::getenv("PECLET_VOF_WETTING_KERNEL_ONLY"))
      solverGates();
  }
  Kokkos::finalize();
  if (failures)
    std::fprintf(stderr, "%d check(s) FAILED\n", failures);
  else
    std::printf("all vof dynamic-wetting checks passed\n");
  return failures ? 1 : 0;
}
