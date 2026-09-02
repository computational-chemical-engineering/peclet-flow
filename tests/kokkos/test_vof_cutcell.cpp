// VoF rung V5a (WO-Q) — the colour field transported THROUGH an immersed solid: openness-weighted
// Weymouth-Yue fluxes, the update in fluid-volume units, and the neutral (90 deg) solid-band fill.
//
// Gates, in the order they are run:
//
//   G3 THE TWO-BLOCK BRIDGE WITH GEOMETRY. The openness the solver hands the advector must be the
//      SAME face, i.e. shifted by one cell along the component's own axis exactly as the face
//      velocity is (`vof::copyFaceVelocity`, WO-J's 35 %-of-the-volume trap, here for the
//      openness). Two halves: (a) the geometry fields themselves, against an INDEPENDENT build of
//      `buildOpenness` / `buildCellFraction` on a g=3 block with the shift written out by hand;
//      (b) the transported colour, against a standalone `WyAdvector` driven by the same velocity
//      and the same geometry — bitwise.
//
//   G2 CONSERVATION THROUGH A PACKING (kinematic). A periodic sphere array, the single-phase
//      solver run to a Stokes steady state, then that PROJECTED velocity frozen and a liquid slab
//      advected by `advect_vof` alone. The conserved functional is `sum eps_eff C` (see
//      `vof/cutcell.hpp` rule 1); the floor is the projection's own `max|div(open u)|` and both are
//      reported. Also: zero colour in solid cells EXACTLY, boundedness in the uncut fluid cells,
//      and the clipped volume.
//
//   G5 THE 90-DEGREE NEUTRAL FILL. A liquid cap on a flat SDF wall at a HALF-INTEGER z (so the wall
//      cells are genuinely cut) with surface tension: a hemisphere is the equilibrium of the
//      neutral fill, and the apparent contact angle measured from the cap volume, its height and
//      its contact radius must come back at 90 deg.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "vof/plic.hpp"
#include "vof_advect_scenes.hpp"

namespace {
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::WyAdvector;

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

std::size_t idx(int x, int y, int z, int nx, int ny) {
  return (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
}

// ---------------------------------------------------------------- the scene
//
// A periodic array of spheres, given ANALYTICALLY so the same function can be sampled on the
// solver's inner region and on a g=3 test block (gate G3 needs both). Cell centres are at
// (i+1/2)h with h = 1 (flow works in cell units on the colour block).
struct SphereArray {
  int n = 0;
  double cx[8], cy[8], cz[8], r[8];
  int N = 0;  // periodic period, in cells
  double sdfAt(double x, double y, double z) const {
    double best = 1e30;
    for (int s = 0; s < n; ++s)
      for (int px = -1; px <= 1; ++px)
        for (int py = -1; py <= 1; ++py)
          for (int pz = -1; pz <= 1; ++pz) {
            const double dx = x - (cx[s] + px * N), dy = y - (cy[s] + py * N),
                         dz = z - (cz[s] + pz * N);
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz) - r[s];
            best = d < best ? d : best;
          }
    return best;  // > 0 in fluid (flow's convention)
  }
};

SphereArray makeArray(int N) {
  SphereArray a;
  a.N = N;
  a.n = 4;
  const double f = N / 32.0;
  const double px[4] = {6.0, 20.0, 11.0, 26.0}, py[4] = {7.0, 9.0, 24.0, 22.0},
               pz[4] = {8.0, 23.0, 19.0, 6.0};
  for (int s = 0; s < 4; ++s) {
    a.cx[s] = px[s] * f;
    a.cy[s] = py[s] * f;
    a.cz[s] = pz[s] * f;
    a.r[s] = 6.0 * f;
  }
  return a;
}

std::vector<double> arraySdfInner(const SphereArray& a, int N) {
  std::vector<double> sd((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        sd[idx(x, y, z, N, N)] = a.sdfAt(x + 0.5, y + 0.5, z + 0.5);
  return sd;
}

// The same SDF on an extended block of ghost width g (gate G3's independent geometry build).
void arraySdfBlock(const SphereArray& a, CCField dst, I3 e, int g) {
  auto h = Kokkos::create_mirror_view(dst);
  for (int z = 0; z < e.z; ++z)
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x)
        h(L3(x, y, z, e)) = a.sdfAt(x - g + 0.5, y - g + 0.5, z - g + 0.5);
  Kokkos::deep_copy(dst, h);
}

std::vector<double> hostOf(SField v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  std::vector<double> out(v.extent(0));
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = h(i);
  return out;
}

// Run the single-phase solver to a Stokes steady state on the packing and return it.
void stokesSteady(peclet::flow::IbmSolver& s, int steps) {
  for (int i = 0; i < steps; ++i)
    s.step();
}

// ================================================================ G3: the bridge with geometry
void bridgeWithGeometry() {
  const int N = 24;
  const SphereArray arr = makeArray(N);
  peclet::flow::IbmSolver s(N, N, N);
  s.setRho(1.0);
  s.setMu(0.2);
  s.setDt(1.0);
  s.setBodyForce(2e-3, 1e-3, 0.0);
  s.setSolid(arraySdfInner(arr, N), true);
  stokesSteady(s, 40);
  s.enableVof();
  CHECK(s.vofHasGeometry());

  // --- (a) the geometry fields, against an independent build on a g=3 block --------------------
  const int g = peclet::flow::IbmSolver::kVofG;
  const I3 e3{N + 2 * g, N + 2 * g, N + 2 * g};
  const long len = (long)e3.x * e3.y * e3.z;
  CCField sdf3("t::sdf3", len), ox3("t::ox3", len), oy3("t::oy3", len), oz3("t::oz3", len),
      cs3("t::cs3", len);
  arraySdfBlock(arr, sdf3, e3, g);
  peclet::flow::buildOpenness(ox3, oy3, oz3, CCConst(sdf3), C3{e3.x, e3.y, e3.z}, 1.0, 1.0, 1.0,
                              s.apertureOrder());
  peclet::flow::buildCellFraction(cs3, CCConst(sdf3), C3{e3.x, e3.y, e3.z}, 1);
  const std::vector<double> refO[3] = {hostOf(ox3), hostOf(oy3), hostOf(oz3)};
  const auto refE = hostOf(cs3);
  const long sd3[3] = {1, (long)e3.x, (long)e3.x * e3.y};

  const auto& adv = s.vofAdvector();
  double dOpen = 0.0, dEps = 0.0;
  for (int d = 0; d < 3; ++d) {
    const auto got = hostOf(adv.faceOpenness(d));
    for (int z = g; z < g + N; ++z)
      for (int y = g; y < g + N; ++y)
        for (int x = g; x < g + N; ++x) {
          const long i = L3(x, y, z, e3);
          // buildOpenness gives the LOW (-d) face of each cell; the advector wants the HIGH (+d)
          // face of the same cell, i.e. the low face of the cell one step up in d. That written-out
          // shift is exactly what `vof::copyFaceVelocity` does inside the solver.
          dOpen = std::fmax(dOpen, std::fabs(got[i] - refO[d][i + sd3[d]]));
        }
  }
  {
    const auto got = hostOf(adv.epsFraction());
    for (int z = g; z < g + N; ++z)
      for (int y = g; y < g + N; ++y)
        for (int x = g; x < g + N; ++x) {
          const long i = L3(x, y, z, e3);
          dEps = std::fmax(dEps, std::fabs(got[i] - refE[i]));
        }
  }
  std::printf(
      "G3a geometry embed: max|openness - independent| %.3e   max|eps - independent| %.3e\n", dOpen,
      dEps);
  CHECK(dOpen == 0.0);
  CHECK(dEps == 0.0);

  // --- (b) the transported colour, against a standalone advector -------------------------------
  // Seed a slab and transport it BOTH ways with the same (frozen, projected) velocity.
  std::vector<double> c0((std::size_t)N * N * N, 0.0);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        c0[idx(x, y, z, N, N)] = (z < N / 2) ? 1.0 : ((z == N / 2) ? 0.37 : 0.0);
  s.setVof(c0);

  WyAdvector ref;
  ref.init(N, N, N, 1.0, g);
  ref.interfaceLocalCfl = true;
  ref.enableGeometry();
  // The reference's geometry: the INDEPENDENT fields above, embedded with the shift by hand.
  {
    auto ho = Kokkos::create_mirror_view(ref.faceOpenness(0));
    for (int d = 0; d < 3; ++d) {
      auto hd = Kokkos::create_mirror_view(ref.faceOpenness(d));
      for (long i = 0; i < len; ++i)
        hd(i) = (i + sd3[d] < len) ? refO[d][i + sd3[d]] : 0.0;
      Kokkos::deep_copy(ref.faceOpenness(d), hd);
    }
    auto he = Kokkos::create_mirror_view(ref.epsFraction());
    for (long i = 0; i < len; ++i)
      he(i) = refE[i];
    Kokkos::deep_copy(ref.epsFraction(), he);
    (void)ho;
  }
  ref.exchange = [&ref, e3, g](SField f) {
    vofscene::periodicFill(f, e3, g, true, true, true);
    if (ref.hasGeometry() && f.data() == ref.colour().data()) {
      ref.solidBandFill();
      vofscene::periodicFill(f, e3, g, true, true, true);
    }
  };
  vofscene::periodicFill(ref.faceOpenness(0), e3, g, true, true, true);
  vofscene::periodicFill(ref.faceOpenness(1), e3, g, true, true, true);
  vofscene::periodicFill(ref.faceOpenness(2), e3, g, true, true, true);
  vofscene::periodicFill(ref.epsFraction(), e3, g, true, true, true);
  ref.classifyGeometry();
  vofscene::periodicFill(ref.kindDouble(), e3, g, true, true, true);
  ref.finalizeGeometry();
  {  // the same colour, with solid cells zeroed exactly as the solver's canonical field is
    auto hc = Kokkos::create_mirror_view(ref.colour());
    auto hk = Kokkos::create_mirror_view(ref.kindDouble());
    Kokkos::deep_copy(hk, ref.kindDouble());
    for (long i = 0; i < len; ++i)
      hc(i) = 0.0;
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          const long i = L3(x + g, y + g, z + g, e3);
          hc(i) = hk(i) > 0.5 ? 0.0 : c0[idx(x, y, z, N, N)];
        }
    Kokkos::deep_copy(ref.colour(), hc);
  }
  ref.syncGhosts();
  // The solver's own face velocity, re-indexed into the advector's high-face convention.
  const auto uu = s.getVelocity(0), vv = s.getVelocity(1), ww = s.getVelocity(2);
  {
    auto hu = Kokkos::create_mirror_view(ref.faceU());
    auto hv = Kokkos::create_mirror_view(ref.faceV());
    auto hw = Kokkos::create_mirror_view(ref.faceW());
    for (int z = 0; z < e3.z; ++z)
      for (int y = 0; y < e3.y; ++y)
        for (int x = 0; x < e3.x; ++x) {
          const int gx = ((x - g) % N + N) % N, gy = ((y - g) % N + N) % N,
                    gz = ((z - g) % N + N) % N;
          const long i = L3(x, y, z, e3);
          hu(i) = uu[idx((gx + 1) % N, gy, gz, N, N)];
          hv(i) = vv[idx(gx, (gy + 1) % N, gz, N, N)];
          hw(i) = ww[idx(gx, gy, (gz + 1) % N, N, N)];
        }
    Kokkos::deep_copy(ref.faceU(), hu);
    Kokkos::deep_copy(ref.faceV(), hv);
    Kokkos::deep_copy(ref.faceW(), hw);
  }
  const double dth = 1.0;  // h = 1
  double cflRef = ref.maxCourantInterfaceAuto(dth);
  const double dtA = cflRef > 0 ? 0.2 / cflRef : 1.0;
  s.setDt(dtA);
  const int steps = 20;
  for (int t = 0; t < steps; ++t) {
    ref.advect(dtA, t);
    s.advectVofKinematic(dtA);
  }
  const auto cs = s.getVof();
  const auto cr = hostOf(ref.colour());
  const auto kr = hostOf(ref.kindDouble());
  // The canonical "C" carries 0 in solid cells while the working block carries the neutral band
  // fill, so the comparison is over FLUID cells (the fill itself is gated by G5).
  double dmax = 0.0, fillMax = 0.0;
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const long i = L3(x + g, y + g, z + g, e3);
        if (kr[i] > 0.5) {
          fillMax = std::fmax(fillMax, cr[i]);
          continue;
        }
        dmax = std::fmax(dmax, std::fabs(cs[idx(x, y, z, N, N)] - cr[i]));
      }
  std::printf(
      "G3b solver vs standalone through the packing (%d steps, cfl 0.2, dt %.4g): max|dC| over "
      "fluid cells %.3e (standalone band fill reaches %.3f)\n",
      steps, dtA, dmax, fillMax);
  CHECK(dmax == 0.0);
}

// ================================================================ G2: conservation
void conservationThroughPacking(int N, int nAdv) {
  const SphereArray arr = makeArray(N);
  peclet::flow::IbmSolver s(N, N, N);
  s.setRho(1.0);
  s.setMu(0.2);
  s.setDt(1.0);
  s.setBodyForce(2e-3, 1e-3, 5e-4);
  s.setSolid(arraySdfInner(arr, N), true);
  stokesSteady(s, 60);
  const double div = s.maxOpenDivergence();
  const long pit = s.lastPressureIterations();
  s.enableVof();
  std::vector<double> c0((std::size_t)N * N * N, 0.0);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        c0[idx(x, y, z, N, N)] = (z < N / 2) ? 1.0 : 0.0;
  s.setVof(c0);

  const auto d0 = s.vofDiagnostics();
  const double cfl = s.vofMaxCourant();  // uses the current dt (= 1)
  const double dtA = cfl > 0 ? 0.2 / cfl : 1.0;
  double clip = 0.0, mnF = 1e30, mxF = -1e30, solidSum = 0.0;
  for (int t = 0; t < nAdv; ++t) {
    s.advectVofKinematic(dtA);
    const auto d = s.vofDiagnostics();
    clip += d.clippedVolume;
    mnF = std::fmin(mnF, d.minCFluid);
    mxF = std::fmax(mxF, d.maxCFluid);
    solidSum = std::fmax(solidSum, std::fabs(d.solidSumC));
  }
  const auto d1 = s.vofDiagnostics();
  const double drift = std::fabs(d1.volume - d0.volume) / d0.volume;
  const double rawDrift = std::fabs(d1.rawVolume - d0.rawVolume) / d0.rawVolume;
  std::printf(
      "G2 packing %d^3, %d kinematic steps at cfl 0.2 (dt %.4g):\n"
      "   max|div(open u)| %.3e (pressure iters %ld)\n"
      "   sum eps_eff C: %.15e -> %.15e   relative drift %.3e\n"
      "   sum eps     C: %.15e -> %.15e   relative drift %.3e\n"
      "   solid cells %ld, cut cells %ld, max|sum C over solid| %.3e\n"
      "   min/max C over UNCUT fluid cells %.3e / %.15e\n"
      "   clipped liquid volume, total over the run %.6e (%.3e of the liquid volume)\n",
      N, nAdv, dtA, div, pit, d0.volume, d1.volume, drift, d0.rawVolume, d1.rawVolume, rawDrift,
      d1.solidCells, d1.cutCells, solidSum, mnF, mxF, clip, clip / d0.volume);
  CHECK(div <= 1e-10);
  CHECK(drift <= 1e-11);
  CHECK(solidSum == 0.0);
  CHECK(mnF >= -1e-12);
  CHECK(mxF <= 1.0 + 1e-12);
  CHECK(d1.solidCells > 0);
  CHECK(d1.cutCells > 0);
}

// ================================================================ G5: the 90-degree neutral fill
//
// A liquid cap on a flat SDF wall. The wall occupies z < zw with zw a HALF-INTEGER in cell units,
// so the cells straddling it are genuinely cut (eps = 1/2) — a wall on a cell face would be a
// staircase and would not exercise the cut-cell path at all.
void neutralFillCap(int steps, bool zeroSolid = true) {
  const int NX = 40, NY = 40, NZ = 28;
  const double zw = 3.5;  // the wall surface, in cell units (cell k spans [k, k+1])
  const double R = 12.0;  // D/dx = 24
  const double sigma = 1.0;
  std::vector<double> sd((std::size_t)NX * NY * NZ);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x)
        sd[idx(x, y, z, NX, NY)] = (z + 0.5) - zw;  // > 0 above the wall = fluid
  // The initial hemisphere, by 4^3 midpoint subsampling (the same quadrature the fluid fraction
  // uses, so the initial colour and the geometry agree cell by cell).
  const double cx = NX * 0.5, cy = NY * 0.5;
  std::vector<double> c0((std::size_t)NX * NY * NZ, 0.0);
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        int in = 0, tot = 0;
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c) {
              const double px = x + (a + 0.5) / 4.0, py = y + (b + 0.5) / 4.0,
                           pz = z + (c + 0.5) / 4.0;
              if (pz < zw)
                continue;  // solid: C is a fraction of the FLUID volume
              ++tot;
              const double dx = px - cx, dy = py - cy, dz = pz - zw;
              if (dx * dx + dy * dy + dz * dz < R * R)
                ++in;
            }
        c0[idx(x, y, z, NX, NY)] = tot ? (double)in / tot : 0.0;
      }

  peclet::flow::IbmSolver s(NX, NY, NZ);
  s.setRho(1.0);
  s.setMu(0.05);
  s.setSolid(sd, true);
  s.enableVof();
  s.setVofSolidColourZero(zeroSolid);
  s.setVof(c0);
  s.setSurfaceTension(sigma);
  const double dtSig = s.capillaryDt();
  s.setDt(0.5 * dtSig);
  const double V00 = s.vofDiagnostics().volume;
  long maxIt = 0;
  double uTrace[4] = {0, 0, 0, 0};
  for (int i = 0; i < steps; ++i) {
    s.step();
    maxIt = std::max(maxIt, s.lastPressureIterations());
    if ((i + 1) % (steps / 4) == 0) {
      const int slot = std::min(3, (i + 1) / (steps / 4) - 1);
      double m = 0;
      for (int c = 0; c < 3; ++c)
        for (double v : s.getVelocity(c))
          m = std::fmax(m, std::fabs(v));
      uTrace[slot] = m;
    }
  }
  // The curvature census in the WALL BAND (the two fluid planes above the wall) — the WO asks for
  // it because the neutral fill is what those stencils read.
  const auto cens = [&] {
    s.computeVofCurvature();
    return s.vofCurvatureStats();
  }();
  const auto kap = s.getVofCurvature();
  const auto kbr = s.getVofCurvatureBranch();
  double kmaxWall = 0.0;
  long brWall[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int z = (int)std::ceil(zw); z < (int)std::ceil(zw) + 2; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        const std::size_t i = idx(x, y, z, NX, NY);
        const int b = (int)kbr[i];
        if (b >= 0 && b < 7)
          ++brWall[b];
        if (b != 0)
          kmaxWall = std::fmax(kmaxWall, std::fabs(kap[i]));
      }
  const auto d = s.vofDiagnostics();
  const auto cc = s.getVof();
  // h: the liquid column height on the axis (sum of C over the fluid column at the centre).
  double h = 0.0;
  {
    const int x0 = NX / 2, y0 = NY / 2;
    for (int z = 0; z < NZ; ++z)
      h += cc[idx(x0, y0, z, NX, NY)] * ((z + 1 <= zw) ? 0.0 : std::fmin(1.0, z + 1 - zw));
  }
  // a: from the liquid AREA in the first FULL fluid plane above the wall (more robust than a
  // contour, and the C = 1/2 contour of a resolved cap is its area radius to O(h^2)).
  const int z0 = (int)std::ceil(zw);
  double area = 0.0;
  for (int y = 0; y < NY; ++y)
    for (int x = 0; x < NX; ++x)
      area += cc[idx(x, y, z0, NX, NY)];
  const double a = std::sqrt(area / M_PI);
  const double theta = 2.0 * std::atan2(h, a) * 180.0 / M_PI;
  const double Rc = (a * a + h * h) / (2.0 * h);
  const double V = d.volume;
  const double V0 = 2.0 * M_PI * R * R * R / 3.0;
  // Young-Laplace: the pressure jump across the cap.
  const auto pf = s.getField("p");
  double pIn = 0.0, pOut = 0.0;
  int nIn = 0, nOut = 0;
  for (int z = 0; z < NZ; ++z)
    for (int y = 0; y < NY; ++y)
      for (int x = 0; x < NX; ++x) {
        const double dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - zw;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dz < 0.5)
          continue;
        if (r < 0.55 * R) {
          pIn += pf[idx(x, y, z, NX, NY)];
          ++nIn;
        } else if (r > 1.5 * R && r < 1.8 * R) {
          pOut += pf[idx(x, y, z, NX, NY)];
          ++nOut;
        }
      }
  const double dP = (nIn ? pIn / nIn : 0.0) - (nOut ? pOut / nOut : 0.0);
  // max|u| over the OPEN fluid only (z >= ceil(zw)+1): the velocity DOFs that live on faces inside
  // or on the wall are IBM-constrained, not flow, and reporting them as a spurious current would
  // measure the immersed boundary rather than the surface-tension balance.
  double umax = 0.0, umaxAll = 0.0;
  for (int c = 0; c < 3; ++c) {
    const auto v = s.getVelocity(c);
    for (int z = 0; z < NZ; ++z)
      for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x) {
          const double a = std::fabs(v[idx(x, y, z, NX, NY)]);
          umaxAll = std::fmax(umaxAll, a);
          if (z >= (int)std::ceil(zw) + 1)
            umax = std::fmax(umax, a);
        }
  }
  const double Ca = 0.05 * umax / sigma;  // mu of this scene
  std::printf(
      "G5 cap on a cut wall (D/dx %.0f, %d steps at 0.5 dt_sigma = %.4g, solid colour %s), "
      "pressure iters max %ld:\n"
      "   volume %.6f -> %.6f (drift %.3e); analytic hemisphere %.6f\n"
      "   h %.4f  a %.4f  ->  apparent theta %.3f deg   (target 90)\n"
      "   cap sphere radius %.4f (target %.1f); Young-Laplace dP %.6f vs 2 sigma/R %.6f "
      "(rel %.3e)\n"
      "   max|u| (open fluid) %.3e -> spurious Ca %.3e ; max|u| incl. the wall band %.3e ;\n"
      "   trace of max|u| (all cells) over the run %.2e %.2e %.2e %.2e\n"
      "   wall band: max|kappa| %.4f, branches 0/1/2/3/4/5/6 = %ld/%ld/%ld/%ld/%ld/%ld/%ld "
      "(global no_estimate %ld)\n"
      "   clipped volume last step %.3e ; clamped faces %ld ; solid sum C %.3e\n",
      2 * R, steps, 0.5 * dtSig, zeroSolid ? "ZERO" : "band fill", maxIt, V00, V,
      std::fabs(V - V00) / V00, V0, h, a, theta, Rc, R, dP, 2.0 * sigma / Rc,
      std::fabs(dP - 2.0 * sigma / Rc) / (2.0 * sigma / Rc), umax, Ca, umaxAll, uTrace[0],
      uTrace[1], uTrace[2], uTrace[3], kmaxWall, brWall[0], brWall[1], brWall[2], brWall[3],
      brWall[4], brWall[5], brWall[6], cens.noEstimate, d.clippedVolume, d.clampedFaces,
      d.solidSumC);
  CHECK(std::fabs(theta - 90.0) <= 3.0);
  CHECK(std::fabs(dP - 2.0 * sigma / Rc) / (2.0 * sigma / Rc) <= 0.01);
  if (zeroSolid)
    CHECK(d.solidSumC == 0.0);
  CHECK(std::fabs(V - V00) / V00 <= 1e-9);
}

// ================================================================ G8: momentum consistency in cut
// cells (WO-Q item 8)
//
// G8a THE CONSISTENCY IDENTITY. `enable_vof_momentum` + a packing + a UNIFORM velocity: the
//     advected velocity must come back as that same uniform value. Every term of the deviation form
//     `u_new = u_old + dev/(eps^e rho_new)` is a DIFFERENCE OF VELOCITIES, so `dev` is exactly zero
//     for a uniform field whatever `eps^e` and the face openness are — the identity is bitwise in
//     CUT cells too, not only away from them, and there is no tolerance to choose.
// G8b COUPLED DRAINING through the packing at ratio 10 with gravity and momentum consistency: the
//     colour drift per step, boundedness, `max|u|`, and the pressure iteration count against its
//     cap (rule 3b: a capped run is INVALID).
void momentumCutCells() {
  const int N = 24;
  const SphereArray arr = makeArray(N);
  const auto sdf = arraySdfInner(arr, N);
  const double U[3] = {1.0, 0.6, -0.4};
  // --- G8a ------------------------------------------------------------------------------------
  for (double R : {10.0, 1e2, 1e3}) {
    peclet::flow::IbmSolver s(N, N, N);
    s.setRho(R);
    s.setMu(0.0);
    // The CUT-CELL Courant number (`o_f |a_f| / max(eps, 0.1)`) is up to 6x the plain one in this
    // packing — at dt = 0.2, |U| = 1 the plain CFL is 0.2 but the cut one reads 1.24 and `advect()`
    // aborts, which is the limiter doing exactly its job. dt = 0.04 puts the cut number at 0.25.
    s.setDt(0.04);
    s.setSolid(sdf, true);
    s.setPressureChebyshev(true, 300, 1e-13);
    s.enableVof();
    std::vector<double> c0((std::size_t)N * N * N);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          c0[idx(x, y, z, N, N)] = peclet::flow::vof::sphereCellFraction(0.5 * N, 0.5 * N, 0.5 * N,
                                                                         0.28 * N, x, y, z, 1.0, 4);
    s.setVof(c0);
    s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, R - 1.0});
    s.enableVofMomentum(1.0, R);
    const std::size_t nc = (std::size_t)N * N * N;
    s.uploadVelocity(std::vector<double>(nc, U[0]), std::vector<double>(nc, U[1]),
                     std::vector<double>(nc, U[2]));
    s.advectVofMomentum();
    double dev = 0.0;
    for (int c = 0; c < 3; ++c)
      for (double v : s.getVofAdvectedVelocity(c))
        dev = std::fmax(dev, std::fabs(v - U[c]));
    const auto md = s.vofMomentumDiagnostics();
    std::printf(
        "G8a packing + uniform U, ratio %-6g: max|u_adv - U| = %.17g%s   (rho^e floor hit %ld, "
        "flux clamps %ld)\n",
        R, dev, dev == 0.0 ? "   BITWISE" : "", md.floored, md.clamped);
    CHECK(dev == 0.0);
    CHECK(md.floored == 0);
  }
  // --- G8b ------------------------------------------------------------------------------------
  // The body force is the ZERO-MEAN buoyancy `f_z = -g (rho - <rho>)`, not `-g rho`: in a fully
  // periodic box the projection removes only GRADIENTS, so a force with a non-zero mean accelerates
  // the whole fluid without bound and the case has no steady state to measure a drift against.
  //
  // MEASURED with the non-zero-mean force (`PECLET_VOF_CUTCELL_NONZERO_FORCE=1`), and RECORDED as
  // an open question rather than explained: with momentum consistency ON the run is clean and
  // conservative to 1e-12 for 155 steps — `C^e` inside [0,1] to the last bit, `rho^e` never
  // floored, 9-10 pressure iterations — and then at step ~160, with `max|u|` around 0.19, `C^e`
  // goes to +inf inside the ADVECTION and the velocity follows. With momentum consistency OFF the
  // identical case runs the full 200 steps (drift -2.3e-13, max|u| 0.236). So it is the
  // momentum-consistent cut-cell path, not the colour transport and not the unbounded acceleration
  // on its own; the trace shows no bounded-quantity precursor at all, which is why it is reported
  // instead of patched.
  for (int momentumOn = 1; momentumOn >= 0; --momentumOn) {
    const double R = 10.0, grav = 2e-3;
    const double liq = 0.25, rhoMean = liq * R + (1.0 - liq);
    peclet::flow::IbmSolver s(N, N, N);
    s.setRho(1.0);
    s.setMu(0.05);
    s.setDt(0.5);
    s.setSolid(sdf, true);
    s.enableVof();
    std::vector<double> c0((std::size_t)N * N * N, 0.0);
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
          c0[idx(x, y, z, N, N)] = (z >= 3 * N / 4) ? 1.0 : 0.0;  // a liquid layer on top
    s.setVof(c0);
    s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {1.0, R - 1.0});
    s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {0.05, 0.45});
    const bool nonZeroMean = std::getenv("PECLET_VOF_CUTCELL_NONZERO_FORCE") != nullptr;
    s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "rho", "",
                       {nonZeroMean ? 0.0 : grav * rhoMean, -grav});
    if (momentumOn)
      s.enableVofMomentum(1.0, R);
    const double v0 = s.vofDiagnostics().volume;
    long maxIt = 0;
    const int steps = 200;
    double umax = 0.0, clip = 0.0, mn = 1e30, mx = -1e30;
    for (int i = 0; i < steps; ++i) {
      s.step();
      maxIt = std::max(maxIt, s.lastPressureIterations());
      const auto d = s.vofDiagnostics();
      clip += d.clippedVolume;
      mn = std::fmin(mn, d.minCFluid);
      mx = std::fmax(mx, d.maxCFluid);
      if (std::getenv("PECLET_VOF_CUTCELL_TRACE") && (i % 5) == 0) {
        double um = 0;
        for (int c = 0; c < 3; ++c)
          for (double v : s.getVelocity(c))
            um = um > std::fabs(v) ? um : std::fabs(v);
        double mnCc = 0, mxCc = 0, mnRho = 0;
        long floored = 0;
        if (momentumOn) {
          const auto md = s.vofMomentumDiagnostics();
          for (int c = 0; c < 3; ++c) {
            mnCc = std::fmin(mnCc, md.minCc[c]);
            mxCc = std::fmax(mxCc, md.maxCc[c]);
          }
          mnRho = md.minRhoC;
          floored = md.floored;
        }
        std::printf(
            "   trace %3d vol %.12e minC %.3e maxC %.6f max|u| %.4e cfl %.4f iters %ld  "
            "C^e [%.3e, %.6f] minRho^e %.4e floored %ld\n",
            i, d.volume, d.minCFluid, d.maxCFluid, um, s.vofLastCourant(),
            s.lastPressureIterations(), mnCc, mxCc, mnRho, floored);
      }
    }
    for (int c = 0; c < 3; ++c)
      for (double v : s.getVelocity(c))
        umax = std::fmax(umax, std::fabs(v));
    const auto d1 = s.vofDiagnostics();
    const double drift = (d1.volume - v0) / v0;
    std::printf(
        "G8b packing draining, ratio %g, %d coupled steps, momentum consistency %s:\n"
        "   colour drift %.3e (%.3e per step), max|div(open u)| %.3e, pressure iters max %ld\n"
        "   min/max C over uncut fluid %.3e / %.15f, clipped volume total %.3e, max|u| %.3e, "
        "solid sum C %.3e\n",
        R, steps, momentumOn ? "ON" : "off", drift, drift / steps, s.maxOpenDivergence(), maxIt, mn,
        mx, clip, umax, d1.solidSumC);
    CHECK(!std::isnan(umax) && !std::isnan(drift));
    CHECK(std::fabs(drift) / steps <= 1e-10);
    CHECK(d1.solidSumC == 0.0);
    CHECK(maxIt < 200);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Kokkos::initialize(argc, argv);
  {
    bool quick = true;
    if (const char* e = std::getenv("PECLET_VOF_CUTCELL_LONG"))
      quick = (e[0] == '0');
    bridgeWithGeometry();
    conservationThroughPacking(quick ? 24 : 48, quick ? 200 : 500);
    if (!std::getenv("PECLET_VOF_CUTCELL_SKIP_G5")) {
      neutralFillCap(quick ? 120 : 200);
      // The measured ablation on what the CANONICAL colour carries inside the solid (the working
      // block always carries the neutral fill): 0 is the WO-Q contract, but the CSF and the
      // property closures read the canonical field, so a wall face then sees a full colour jump.
      if (std::getenv("PECLET_VOF_CUTCELL_FILL_ABLATION"))
        neutralFillCap(quick ? 120 : 200, false);
    }
    momentumCutCells();
  }
  Kokkos::finalize();
  if (failures)
    std::fprintf(stderr, "%d check(s) FAILED\n", failures);
  else
    std::printf("all vof cut-cell checks passed\n");
  return failures ? 1 : 0;
}
