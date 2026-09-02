// VoF rung V1 (WO-E) — gate battery for Weymouth-Yue split geometric advection
// (src/vof/advect_wy.hpp) on prescribed velocity fields. Standalone: no Solver, no MPI.
//
//   A  slab translation      : an axis-normal planar interface under uniform diagonal flow is
//                              transported to round-off; volume drift over 1024 steps
//   B  sphere translation    : L1 shape error 2nd order under refinement (16/32/64);
//                              volume drift < 1e-13 over 1024 steps
//   C  Zalesak slotted disk  : one revolution, L1 error recorded and compared in order of
//                              magnitude with published geometric-VoF values
//   D  LeVeque deformation   : 3D deformation field with time reversal (T = 3) at 32^3 and 64^3
//                              (128^3 opt-in, PECLET_VOF_LEVEQUE_128=1); volume drift vs the
//                              measured discrete face divergence
//   E  CFL guard             : a step at CFL >= 0.5 aborts with a clear error
//   F  worklist neutrality   : compaction on/off is bitwise identical
//   G  the dilation trap     : recomputing H(C-1/2) between sweeps, measured (diagnostic only)
//
// The velocity fields are built so that the DISCRETE face divergence, not merely the analytic one,
// vanishes: uniform and solid-body rotation are exactly zero bitwise, and the LeVeque field is
// sampled as the discrete curl of an edge vector potential (see vof_advect_scenes.hpp).
// Weymouth-Yue conserves volume to exactly the accuracy of that divergence, so this is a
// precondition for the conservation gates, not a nicety.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "vof/advect_wy.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::SField;
using peclet::flow::vof::WyAdvector;
using vofscene::Block;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

/// A single-block periodic / walled case: the advector plus the ghost-fill hook it needs.
struct Case {
  WyAdvector adv;
  Block blk;
  std::array<bool, 3> per{true, true, true};

  void setup(int nx, int ny, int nz, double h, bool px = true, bool py = true, bool pz = true) {
    adv.init(nx, ny, nz, h, 3);
    blk = vofscene::blockOf(adv, I3{0, 0, 0});
    per = {px, py, pz};
    const I3 e = adv.extent();
    const int g = adv.ghost();
    const std::array<bool, 3> p = per;
    const Block bb = blk;
    const I3 gs{nx, ny, nz};
    adv.exchange = [e, g, p, bb, gs](SField f) {
      vofscene::periodicFill(f, e, g, p[0], p[1], p[2]);
      vofscene::clampFill(f, bb, gs, p[0], p[1], p[2]);
    };
  }
  SField c() const { return adv.colour(); }
  I3 e() const { return adv.extent(); }
  I3 n() const { return adv.inner(); }
  int g() const { return adv.ghost(); }
};

double order(double coarse, double fine) {
  return (fine > 0.0 && coarse > 0.0) ? std::log2(coarse / fine) : 0.0;
}

// =============================================================================== gate A: slab
void gateSlab() {
  std::printf("\n=== A  slab translation (planar interface, uniform diagonal flow)\n");
  const int nx = 32;
  const double h = 1.0 / nx;
  Case cs;
  cs.setup(nx, nx, nx, h);
  vofscene::initSlab(cs.c(), cs.blk, h, 0, 0.2, 0.7);
  cs.adv.syncGhosts();
  SField c0 = vofscene::copyOf(cs.c(), "slab0");
  const auto d0 = cs.adv.diagnostics();

  // CFL 0.25 with |u| = 1 in each direction; 128 steps is one full lap, run 8 laps (1024 > 1000).
  const double dt = 0.25 * h;
  vofscene::fillUniform(cs.adv, cs.blk, 1.0, 1.0, 1.0);
  for (long s = 0; s < 1024; ++s)
    cs.adv.advect(dt, s);
  const auto d1 = cs.adv.diagnostics();

  const double linf = vofscene::lInfDiff(cs.c(), c0, cs.e(), cs.n(), cs.g());
  const double l1 = vofscene::l1Diff(cs.c(), c0, cs.e(), cs.n(), cs.g());
  const double drift = std::fabs(d1.sumC - d0.sumC) / d0.sumC;
  std::printf("  1024 steps (8 laps), CFL 0.25:  Linf %.3e   L1 %.3e   volume drift %.3e\n", linf,
              l1, drift);
  std::printf("  C range [%.17g, %.17g]   wisps %ld\n", d1.minC, d1.maxC, d1.wisps);
  CHECK(drift < 1e-13);
  CHECK(linf < 1e-11);  // a plane under uniform flow is transported exactly up to accumulated eps
  CHECK(d1.minC >= -1e-14 && d1.maxC <= 1.0 + 1e-14);
}

// ============================================================================= gate B: sphere
void gateSphere() {
  std::printf("\n=== B  sphere translation (uniform diagonal flow, periodic box)\n");
  const int res[3] = {16, 32, 64};
  double err[3] = {0, 0, 0};
  for (int k = 0; k < 3; ++k) {
    const int nx = res[k];
    const double h = 1.0 / nx;
    Case cs;
    cs.setup(nx, nx, nx, h);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, 0.25);
    cs.adv.syncGhosts();
    SField c0 = vofscene::copyOf(cs.c(), "sph0");
    const auto d0 = cs.adv.diagnostics();

    const double dt = 0.25 * h;  // CFL 0.25 on each component
    const long steps = 4L * nx;  // one full diagonal lap: steps*dt*1 = 1.0 exactly
    vofscene::fillUniform(cs.adv, cs.blk, 1.0, 1.0, 1.0);
    for (long s = 0; s < steps; ++s)
      cs.adv.advect(dt, s);
    const auto d1 = cs.adv.diagnostics();

    const double l1 = vofscene::l1Diff(cs.c(), c0, cs.e(), cs.n(), cs.g());
    err[k] = l1 * h * h * h;  // volume-weighted L1 shape error
    const double drift = std::fabs(d1.sumC - d0.sumC) / d0.sumC;
    std::printf(
        "  %3d^3  %4ld steps  L1(vol) %.4e  L1/V %.4e  drift %.2e  C[%.3e,%.5f] wisps %ld\n", nx,
        steps, err[k], l1 / d0.sumC, drift, d1.minC, d1.maxC, d1.wisps);
    CHECK(drift < 1e-13);
  }
  const double o1 = order(err[0], err[1]), o2 = order(err[1], err[2]);
  std::printf("  L1 shape-error order: 16->32 %.2f   32->64 %.2f\n", o1, o2);
  CHECK(o2 > 1.7);
  CHECK(o1 > 1.5);

  // long run: the conservation gate proper (>= 1000 steps)
  std::printf("  -- long run, 32^3, 1024 steps (8 laps)\n");
  const int nx = 32;
  const double h = 1.0 / nx;
  Case cs;
  cs.setup(nx, nx, nx, h);
  vofscene::initSphere(cs.c(), cs.blk, h, 0.5, 0.5, 0.5, 0.25);
  cs.adv.syncGhosts();
  SField c0 = vofscene::copyOf(cs.c(), "sphL0");
  const auto d0 = cs.adv.diagnostics();
  vofscene::fillUniform(cs.adv, cs.blk, 1.0, 1.0, 1.0);
  const double dt = 0.25 * h;
  double worstDrift = 0.0;
  for (long s = 0; s < 1024; ++s) {
    cs.adv.advect(dt, s);
    if ((s + 1) % 128 == 0) {
      const auto d = cs.adv.diagnostics();
      worstDrift = std::fmax(worstDrift, std::fabs(d.sumC - d0.sumC) / d0.sumC);
    }
  }
  const auto d1 = cs.adv.diagnostics();
  const double l1 = vofscene::l1Diff(cs.c(), c0, cs.e(), cs.n(), cs.g());
  std::printf("  drift(worst over 8 checkpoints) %.3e   final %.3e   L1(vol) %.4e   wisps %ld\n",
              worstDrift, std::fabs(d1.sumC - d0.sumC) / d0.sumC, l1 * h * h * h, d1.wisps);
  CHECK(worstDrift < 1e-13);
  CHECK(d1.minC > -1e-14 && d1.maxC < 1.0 + 1e-14);
}

// ============================================================================ gate C: Zalesak
void gateZalesak() {
  std::printf("\n=== C  Zalesak slotted disk, one revolution (100^2 in-plane)\n");
  const int nx = 100, nz = 4;
  const double h = 1.0 / nx;
  Case cs;
  cs.setup(nx, nx, nz, h);
  vofscene::initZalesak(cs.c(), cs.blk, h, 0.5, 0.75, 0.15, 0.025, 0.85);
  cs.adv.syncGhosts();
  SField c0 = vofscene::copyOf(cs.c(), "zal0");
  const auto d0 = cs.adv.diagnostics();

  // The CFL guard is global, while the disk sits at r = 0.25 and the fastest faces are at the
  // domain corner (r ~ 0.5): the step count is set by the corner, not by the interface, so this
  // run takes ~3x the steps a solver limiting dt at the interface would. PECLET_VOF_ZALESAK_STEPS
  // exposes that (the minimum admissible count here is 629).
  // At the published 1000-step setup the GLOBAL max CFL is 0.314 (corner), above the default 3D
  // cap of 0.25, while every cell the interface ever visits stays at CFL <= 0.157. Weymouth's
  // bound is a per-flux statement, so the run is admissible and the global guard is merely a
  // conservative proxy; raise the cap here deliberately rather than shrinking dt, which would
  // break comparability with the published L1 values.
  // (Follow-up noted in VOF_PLAN.md: at V2 the solver's dt limiter should measure the CFL over
  // interface-adjacent cells, not the whole domain, or quiescent far-field corners will throttle
  // the step for nothing.)
  cs.adv.cflLimit = 0.5;
  long steps = 1000;
  if (const char* v = std::getenv("PECLET_VOF_ZALESAK_STEPS"); v && std::atol(v) > 0)
    steps = std::atol(v);
  const double T = 1.0, omega = 2.0 * M_PI / T, dt = T / steps;
  vofscene::fillRotation(cs.adv, cs.blk, h, 0.5, 0.5, omega);
  std::printf("  discrete |div| dt/h max = %.3e  (exactly zero expected)\n",
              cs.adv.maxDiscreteDivergence(dt / h));
  for (long s = 0; s < steps; ++s)
    cs.adv.advect(dt, s);
  const auto d1 = cs.adv.diagnostics();

  const double l1 = vofscene::l1Diff(cs.c(), c0, cs.e(), cs.n(), cs.g());
  const double rel = l1 / d0.sumC;
  // `rel` is the metric of THINC-scaling (Xie & Xiao 2021, eq. 27) and of most VoF papers:
  // sum|C - C_ex| / sum|C_ex|. Their published N=100 values: THINC-scaling 1.55e-2,
  // MTHINC 1.61e-2, UMTHINC 2.61e-2, THINC/QQ 3.22e-2.  `perLayer` is the per-2D-cell mean
  // absolute error E1 of Cassinelli et al. (arXiv:1903.11949, eq. 15).
  std::printf("  %ld steps  CFL %.3f   L1(abs) %.4e   L1/V %.4e   E1(per 2D cell) %.4e\n", steps,
              cs.adv.lastCfl(), l1, rel, l1 * h * h / nz);
  std::printf("  drift %.3e   C[%.3e,%.6f]   mixed %ld   wisps %ld\n",
              std::fabs(d1.sumC - d0.sumC) / d0.sumC, d1.minC, d1.maxC, d1.mixed, d1.wisps);
  CHECK(std::fabs(d1.sumC - d0.sumC) / d0.sumC < 1e-13);
  // published geometric-VoF relative shape errors for this case are ~0.01-0.02; a 10x miss is a bug
  CHECK(rel > 0.002 && rel < 0.2);
}


// ====================================================== gate C2: the interface Courant BAND (WO-R2)
//
// The band `maxCourantInterface` limits on is "this cell's colour differs from a neighbour's".
// Written with an exact `!=` it has no wisp tolerance, so the Weymouth-Yue round-off residue the
// interface leaves behind (min C ~ -3.8e-17 on this scene) keeps the whole WAKE inside the band and
// the interface-local limiter creeps back towards the global max — which is the thing it exists to
// avoid. Measured here on the published Zalesak setup, one revolution, `interfaceLocalCfl = true`
// (the setting `IbmSolver` uses; `lastCfl()` is what `vof_last_courant()` reports).
void gateZalesakBand() {
  std::printf("\n=== C2 interface Courant band on Zalesak: the wisp tolerance (WO-R2 item 4b)\n");
  const int nx = 100, nz = 4;
  const double h = 1.0 / nx;
  const long steps = 1000;
  const double T = 1.0, omega = 2.0 * M_PI / T, dt = T / steps;
  double first[2] = {0, 0}, last[2] = {0, 0}, worst[2] = {0, 0}, drift[2] = {0, 0};
  for (int k = 0; k < 2; ++k) {
    Case cs;
    cs.setup(nx, nx, nz, h);
    vofscene::initZalesak(cs.c(), cs.blk, h, 0.5, 0.75, 0.15, 0.025, 0.85);
    cs.adv.syncGhosts();
    const auto d0 = cs.adv.diagnostics();
    cs.adv.cflLimit = 0.5;             // as in gate C: the corner faces run above the 3D bound
    cs.adv.interfaceLocalCfl = true;   // the solver's setting
    cs.adv.wispEps = (k == 0) ? 0.0 : 1e-8;
    vofscene::fillRotation(cs.adv, cs.blk, h, 0.5, 0.5, omega);
    for (long t = 0; t < steps; ++t) {
      cs.adv.advect(dt, t);
      const double c = cs.adv.lastCfl();
      if (t == 0)
        first[k] = c;
      last[k] = c;
      worst[k] = std::fmax(worst[k], c);
    }
    const auto d1 = cs.adv.diagnostics();
    drift[k] = std::fabs(d1.sumC - d0.sumC) / d0.sumC;
    std::printf("  wispEps %-6g : band Courant after step 1 %.4f, after %ld %.4f, worst %.4f, "
                "volume drift %.3e\n",
                cs.adv.wispEps, first[k], steps, last[k], worst[k], drift[k]);
  }
  // The a-priori bound on a band that is "mixed cells and their face neighbours": the disk's
  // farthest point sits at r = 0.25 + 0.15 = 0.40 from the rotation centre, the band reaches one
  // cell beyond it and the reduction takes the faces of those cells, i.e. r_max + 1.5 h. That is
  // the number the guarded run must not exceed — and it is what it measures, to the printed digit.
  const double bandBound = omega * (0.40 + 1.5 * h) * dt / h;
  const double diskBound = omega * 0.40 * dt / h;
  std::printf("  a-priori: the disk's own faces reach %.4f (r = 0.40) and the band's outermost "
              "faces %.4f (r = 0.40 + 1.5h)\n",
              diskBound, bandBound);
  std::printf("  the global max (what a whole-domain limiter would report) is %.4f (corner "
              "r = 0.7071)\n",
              omega * 0.5 * M_SQRT2 * dt / h);
  CHECK(worst[1] <= 1.002 * bandBound);  // with the guard the band stays ON the interface
  CHECK(worst[0] > 1.15 * bandBound);    // without it the round-off wake widens the band
  CHECK(drift[0] < 1e-13 && drift[1] < 1e-13);  // neither touches conservation
}

// ============================================================================ gate D: LeVeque
struct LeVequeResult {
  double l1vol = 0, rel = 0, drift = 0, divmax = 0, minC = 0, maxC = 0;
  long wisps = 0;
};

LeVequeResult runLeVeque(int nx, long steps, double T) {
  const double h = 1.0 / nx;
  Case cs;
  cs.setup(nx, nx, nx, h);
  vofscene::initSphere(cs.c(), cs.blk, h, 0.35, 0.35, 0.35, 0.15);
  cs.adv.syncGhosts();
  SField c0 = vofscene::copyOf(cs.c(), "lv0");
  const auto d0 = cs.adv.diagnostics();

  const double dt = T / steps;
  LeVequeResult r;
  double worst = 0.0, divmax = 0.0;
  for (long s = 0; s < steps; ++s) {
    const double t = (s + 0.5) * dt;  // midpoint sampling => the reversal is time-symmetric
    vofscene::fillLeVeque(cs.adv, cs.blk, h, std::cos(M_PI * t / T));
    if (s % 25 == 0)
      divmax = std::fmax(divmax, cs.adv.maxDiscreteDivergence(dt / h));
    cs.adv.advect(dt, s);
    if ((s + 1) % (steps / 8) == 0) {
      const auto d = cs.adv.diagnostics();
      worst = std::fmax(worst, std::fabs(d.sumC - d0.sumC) / d0.sumC);
    }
  }
  const auto d1 = cs.adv.diagnostics();
  const double l1 = vofscene::l1Diff(cs.c(), c0, cs.e(), cs.n(), cs.g());
  r.l1vol = l1 * h * h * h;
  r.rel = l1 / d0.sumC;
  r.drift = std::fmax(worst, std::fabs(d1.sumC - d0.sumC) / d0.sumC);
  r.divmax = divmax;
  r.minC = d1.minC;
  r.maxC = d1.maxC;
  r.wisps = d1.wisps;
  return r;
}

void gateLeVeque() {
  std::printf("\n=== D  LeVeque 3D deformation with time reversal (T = 3)\n");
  const double T = 3.0;
  // Step count from a target CFL (max|u| = 2 for this field), so the resolutions are compared at
  // the SAME Courant number. Default 0.24 respects Weymouth's own 3D boundedness bound
  // 1/(2(N-1)) = 1/4 (thesis eq. A.33); PECLET_VOF_LEVEQUE_CFL raises it to probe the gap between
  // that bound and the community-standard 0.5 cap.
  double cfl = 0.24;
  if (const char* v = std::getenv("PECLET_VOF_LEVEQUE_CFL"); v && std::atof(v) > 0.0)
    cfl = std::atof(v);
  const int res[2] = {32, 64};
  const long nst[2] = {std::lround(T * 2 * res[0] / cfl), std::lround(T * 2 * res[1] / cfl)};
  double err[2];
  for (int k = 0; k < 2; ++k) {
    const LeVequeResult r = runLeVeque(res[k], nst[k], T);
    err[k] = r.l1vol;
    std::printf(
        "  %3d^3  %4ld steps  L1(vol) %.4e  L1/V %.4e  drift %.3e  |div|dt/h %.2e"
        "  C[%.2e,%.6f]  wisps %ld\n",
        res[k], nst[k], r.l1vol, r.rel, r.drift, r.divmax, r.minC, r.maxC, r.wisps);
    CHECK(r.drift < 1e-13);
  }
  std::printf("  L1 shape-error order 32->64: %.2f\n", order(err[0], err[1]));

  if (const char* v = std::getenv("PECLET_VOF_LEVEQUE_128"); v && std::atoi(v) != 0) {
    const LeVequeResult r = runLeVeque(128, std::lround(T * 2 * 128 / cfl), T);
    std::printf(
        "  128^3  3200 steps  L1(vol) %.4e  L1/V %.4e  drift %.3e  |div|dt/h %.2e"
        "  C[%.2e,%.6f]  wisps %ld\n",
        r.l1vol, r.rel, r.drift, r.divmax, r.minC, r.maxC, r.wisps);
    std::printf("  L1 shape-error order 64->128: %.2f\n", order(err[1], r.l1vol));
    CHECK(r.drift < 1e-13);
  } else {
    std::printf("  (128^3 rung skipped; set PECLET_VOF_LEVEQUE_128=1 to run it)\n");
  }
}

// =========================================================================== gate E: CFL guard
void gateCfl() {
  std::printf("\n=== E  CFL guard\n");
  Case cs;
  cs.setup(16, 16, 16, 1.0 / 16);
  vofscene::initSphere(cs.c(), cs.blk, 1.0 / 16, 0.5, 0.5, 0.5, 0.25);
  cs.adv.syncGhosts();
  vofscene::fillUniform(cs.adv, cs.blk, 1.0, 0.0, 0.0);

  // The DEFAULT must be Weymouth's proven 3D bound 1/(2(N-1)) = 0.25 (thesis eq. A.33), not the
  // 2D value 0.5 that the literature quotes without qualification. Pinned here so a future edit
  // cannot silently widen the boundedness margin.
  CHECK(cs.adv.cflLimit == 0.25);

  bool threw = false;
  std::string what;
  try {
    cs.adv.advect(0.26 * (1.0 / 16), 0);  // past the 3D bound -> must abort
  } catch (const std::runtime_error& ex) {
    threw = true;
    what = ex.what();
  }
  std::printf("  CFL = 0.26 (> default limit): %s\n", threw ? what.c_str() : "NO THROW");
  CHECK(threw);
  CHECK(what.find("CFL") != std::string::npos);

  // Weymouth's bound is inclusive, so exactly 0.25 must RUN — this is the CFL every physics
  // gate above is deliberately run at.
  threw = false;
  try {
    cs.adv.advect(0.25 * (1.0 / 16), 0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::printf("  CFL = 0.25 (at the bound): %s\n", threw ? "THREW (wrong)" : "ran");
  CHECK(!threw);

  // The cap is deliberately raisable (2D work / probing the gap): the guard must track it.
  cs.adv.cflLimit = 0.5;
  threw = false;
  try {
    cs.adv.advect(0.49 * (1.0 / 16), 0);  // under the raised cap -> must run
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::printf("  CFL = 0.49 @ cflLimit 0.5: %s\n", threw ? "THREW (wrong)" : "ran");
  CHECK(!threw);
}

// ========================================================================== gate F: worklist
void gateWorklist() {
  std::printf("\n=== F  worklist compaction is bitwise neutral\n");
  const int nx = 32;
  const double h = 1.0 / nx, T = 3.0, dt = 3.75e-3;
  std::vector<double> out[2];
  long mixed = 0;
  for (int mode = 0; mode < 2; ++mode) {
    Case cs;
    cs.setup(nx, nx, nx, h);
    cs.adv.useWorklist = (mode == 0);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.35, 0.35, 0.35, 0.15);
    cs.adv.syncGhosts();
    for (long s = 0; s < 40; ++s) {
      vofscene::fillLeVeque(cs.adv, cs.blk, h, std::cos(M_PI * (s + 0.5) * dt / T));
      cs.adv.advect(dt, s);
    }
    if (mode == 0)
      mixed = cs.adv.lastMixedCount();
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cs.c());
    out[mode].assign(hc.data(), hc.data() + hc.extent(0));
  }
  long diff = 0;
  for (std::size_t i = 0; i < out[0].size(); ++i)
    if (std::memcmp(&out[0][i], &out[1][i], sizeof(double)) != 0)
      ++diff;
  std::printf("  40 LeVeque steps, %ld mixed cells in the last sweep: %ld / %zu cells differ\n",
              mixed, diff, out[0].size());
  CHECK(diff == 0);
}

// ==================================================================== gate G: the dilation trap
void gateDilationTrap() {
  std::printf("\n=== G  the dilation-flag trap (diagnostic: what recomputing it costs)\n");
  const int nx = 32;
  const double h = 1.0 / nx, T = 3.0, dt = 3.75e-3;
  double drift[2];
  for (int mode = 0; mode < 2; ++mode) {
    Case cs;
    cs.setup(nx, nx, nx, h);
    cs.adv.debugRecomputeDilation = (mode == 1);
    vofscene::initSphere(cs.c(), cs.blk, h, 0.35, 0.35, 0.35, 0.15);
    cs.adv.syncGhosts();
    const double v0 = cs.adv.diagnostics().sumC;
    for (long s = 0; s < 200; ++s) {
      vofscene::fillLeVeque(cs.adv, cs.blk, h, std::cos(M_PI * (s + 0.5) * dt / T));
      cs.adv.advect(dt, s);
    }
    drift[mode] = std::fabs(cs.adv.diagnostics().sumC - v0) / v0;
  }
  std::printf(
      "  200 LeVeque steps, 32^3:  frozen flag  %.3e     recomputed per sweep  %.3e"
      "  (%.0fx worse)\n",
      drift[0], drift[1], drift[1] / std::fmax(drift[0], 1e-300));
  CHECK(drift[0] < 1e-13);
  CHECK(drift[1] > 1e3 * drift[0]);  // the trap must be loud here, so it stays documented
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("VoF rung V1 (WO-E) - Weymouth-Yue split advection, backend: %s\n",
                peclet::flow::SExec::name());
    gateSlab();
    gateSphere();
    gateZalesak();
    gateZalesakBand();
    gateLeVeque();
    gateCfl();
    gateWorklist();
    gateDilationTrap();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
