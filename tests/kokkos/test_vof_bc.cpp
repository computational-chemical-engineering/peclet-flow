// VoF rung V-BC (WO-R) — two-phase open boundaries: inflow / outflow / wall colour, the
// out-of-domain flux rule, the inflow property ghost, and the variable-density outflow correction.
//
// Gates, in the order they run:
//
//   A  THE MASK. `vof::buildOutsideMask` must mark exactly the ghost cells that lie outside the
//      GLOBAL domain on a non-periodic axis — the same global-clamp predicate `clampFill` uses,
//      for the same reason (a per-block test would be decomposition-dependent).
//
//   B  THE FLUX RULE IS INERT WHERE THE MASK IS 0. `wyFaceFluxBc` must return `wyFaceFlux`'s value
//      BIT FOR BIT for every donor that is not marked. This is the algebraic half of gate G5: the
//      new branch cannot move an interior flux even by a rounding bit.
//
//   C  THE COLOUR BUDGET (WO-R gate G1, kinematic). A 32x32x64 box, uniform inflow at -z, outflow
//      at +z, walls elsewhere, ratio 1: a liquid slug is injected for 100 steps and then the inlet
//      switches to gas for 400 more. The gate is the EXACT budget
//         sum(eps C)(t) - sum(inflow liquid) + sum(outflow liquid) = const
//      where the fluxes are the advector's OWN boundary face fluxes (`vof_bc_volumes`). It closes
//      to round-off because a WY flux is computed once per face and enters exactly one inner
//      cell's update with one sign — the same telescoping that gives exact conservation in a
//      closed box, now with the boundary faces counted instead of cancelling.
//      Also gated: the slug leaves and nothing is left behind (sum C -> 0), and C stays in [0,1].
//
//   D  THE INFLOW PROPERTY GHOST (WO-R item 5). With rho(C) a LinearMix closure and a LIQUID
//      inlet into a GAS domain, the rho ghost band of the inflow face must read rho_liquid, not
//      the interior's rho_gas. Without the repair the inlet FACE density (the arithmetic mean of
//      inner and ghost, used by BOTH the momentum time term and the projection coefficient) is
//      wrong by up to the full density ratio.
//
//   E  BACKFLOW (inletOutlet). On an outflow face whose velocity REVERSES, the colour ghost must
//      carry the prescribed backflow colour; where it does not reverse, the zero-gradient copy
//      must survive untouched.
//
//   F  THE VARIABLE-DENSITY OUTFLOW CORRECTION KERNEL (WO-R item 4). `bcCorrectOutflowVar` must
//      reduce to `bcCorrectOutflow` BITWISE at uniform rho == rho0, and must differ from it by
//      exactly the mobility factor rho0/rho_f otherwise.
//
//   F2 ... AND THE VERDICT ON USING IT, which is NO. The solver must reach the sibling kernel
//      (that is what `set_outflow_rho_correction` switches), and the resulting divergence must be
//      measured on the PROJECTED field — `max_open_divergence()` re-imposes the zero-gradient
//      outflow face before measuring and so cannot see the difference at all (it reads 1.26e-02
//      for both branches at ratio 10, and 1.26e-09 where the projected field is at 1.41e-17).
//      Measured: at ratio 10 the plain correction leaves 8.8e-10 and the 1/rho_f one 9.2e-03,
//      because the operator's outflow-face coefficient is the RAW openness. Default OFF.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::UCField;

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

// ---------------------------------------------------------------------------------- A: the mask
void maskGate() {
  const int n = 8, g = 3;
  const I3 e{n + 2 * g, n + 2 * g, n + 2 * g};
  const long len = (long)e.x * e.y * e.z;
  UCField m("mask", len);
  // x periodic, y and z not; single block, so the origin is 0 and the global size is n.
  peclet::flow::vof::buildOutsideMask(m, e, g, I3{0, 0, 0}, I3{n, n, n}, true, false, false);
  auto h = Kokkos::create_mirror_view(m);
  Kokkos::deep_copy(h, m);
  long marked = 0, wrong = 0;
  for (int z = 0; z < e.z; ++z)
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x) {
        const bool want = (y - g < 0 || y - g >= n) || (z - g < 0 || z - g >= n);
        const bool got = h(L3(x, y, z, e)) != 0u;
        if (got != want)
          ++wrong;
        marked += got ? 1 : 0;
      }
  std::printf("A  outside mask: %ld cells marked, %ld disagreements with the global predicate\n",
              marked, wrong);
  CHECK(wrong == 0);
  CHECK(marked > 0);
}

// -------------------------------------------------------------- B: the flux rule is bitwise inert
void fluxInertGate() {
  const int n = 6, g = 3;
  const I3 e{n + 2 * g, n + 2 * g, n + 2 * g};
  const long len = (long)e.x * e.y * e.z;
  SField c("c", len), mx("mx", len), my("my", len), mz("mz", len), al("al", len);
  UCField zero("zero", len);  // all 0: nothing is "outside"
  auto hc = Kokkos::create_mirror_view(c);
  auto hm = Kokkos::create_mirror_view(mx);
  auto hn = Kokkos::create_mirror_view(my);
  auto ho = Kokkos::create_mirror_view(mz);
  auto ha = Kokkos::create_mirror_view(al);
  unsigned seed = 12345u;
  auto rnd = [&seed]() {
    seed = seed * 1103515245u + 12345u;
    return (double)((seed >> 16) & 0x7fff) / 32767.0;
  };
  for (long i = 0; i < len; ++i) {
    hc(i) = rnd();  // a fully mixed field: every donor takes the geometric branch
    const double a0 = rnd() - 0.5, a1 = rnd() - 0.5, a2 = rnd() - 0.5;
    const double s = std::fabs(a0) + std::fabs(a1) + std::fabs(a2) + 1e-12;
    hm(i) = a0 / s;
    hn(i) = a1 / s;
    ho(i) = a2 / s;
    ha(i) = peclet::flow::vof::plicAlpha(hm(i), hn(i), ho(i), hc(i));
  }
  Kokkos::deep_copy(c, hc);
  Kokkos::deep_copy(mx, hm);
  Kokkos::deep_copy(my, hn);
  Kokkos::deep_copy(mz, ho);
  Kokkos::deep_copy(al, ha);
  Kokkos::deep_copy(zero, (unsigned char)0);

  long diff = 0;
  for (int d = 0; d < 3; ++d) {
    const long sd = d == 0 ? 1 : (d == 1 ? e.x : (long)e.x * e.y);
    long nd = 0;
    Kokkos::parallel_reduce(
        "bc::flux_inert",
        Kokkos::MDRangePolicy<peclet::flow::SExec, Kokkos::Rank<3>>(peclet::flow::SExec(),
                                                                    {1, 1, 1},
                                                                    {e.x - 1, e.y - 1, e.z - 1}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long p = L3(x, y, z, e);
          for (int k = -3; k <= 3; ++k) {
            const double a = 0.07 * k;  // both signs and a == 0
            const double f0 = peclet::flow::vof::wyFaceFlux(a, p, sd, d, c, mx, my, mz, al);
            const double f1 =
                peclet::flow::vof::wyFaceFluxBc(a, p, sd, d, c, mx, my, mz, al, zero);
            if (!(f0 == f1))
              ++acc;
          }
        },
        nd);
    Kokkos::fence();
    diff += nd;
  }
  std::printf("B  wyFaceFluxBc vs wyFaceFlux with an all-zero mask: %ld bitwise disagreements\n",
              diff);
  CHECK(diff == 0);
}

// ------------------------------------------------------------------- C: the exact colour budget
// WO-R gate G1, KINEMATIC. Uniform inflow at -z, outflow at +z, walls elsewhere; the advecting
// field is the uniform w = W, which is EXACTLY discretely divergence-free (du/dx = dv/dy = 0 and w
// is constant along z), so the budget's floor is round-off and nothing else. `max_open_divergence`
// is reported to show it. (The COUPLED variant — the same box driven by step() — is in the study
// script `tests/study/vof_open_boundaries.py`, where the pressure iteration count is recorded
// against its cap; here the point is the advection identity in isolation.)
void budgetGate(int nz, int nSlug, int nAfter) {
  const int nx = 32, ny = 32;
  const double W = 1.0, dt = 0.2;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(dt);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, W);  // inflow at -z, uniform
  s.setDomainBc(5, 3, 0, 0, 0);      // outflow at +z
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>((std::size_t)nx * ny * nz, 0.0));
  s.setField("w", std::vector<double>((std::size_t)nx * ny * nz, W));
  s.setVofInflow(4, 1.0);
  s.setVofBackflow(5, 0.0);

  double budget0 = 0.0, drift = 0.0, minC = 1.0, maxC = 0.0, ledger = 0.0, peak = 0.0;
  std::vector<double> peakProfile;
  for (int i = 0; i < nSlug + nAfter; ++i) {
    if (i == nSlug)
      s.setVofInflow(4, 0.0);  // the slug ends; gas follows
    s.advectVof();
    const auto d = s.vofDiagnostics();
    const auto v = s.vofBcVolumes();
    for (int f = 0; f < 6; ++f)
      ledger += v[f];
    const double budget = d.sumC - ledger;
    if (i == 0)
      budget0 = budget;
    drift = std::fmax(drift, std::fabs(budget - budget0));
    minC = std::fmin(minC, d.minC);
    maxC = std::fmax(maxC, d.maxC);
    if (d.sumC > peak) {  // the slug at its fullest: keep its z-profile
      peak = d.sumC;
      const auto C = s.getVof();
      peakProfile.assign(nz, 0.0);
      for (int z = 0; z < nz; ++z) {
        double a = 0.0;
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            a += C[idx(x, y, z, nx, ny)];
        peakProfile[z] = a / (nx * ny);
      }
    }
  }
  {  // the slug leaves with its length intact: report the plateau length and the edge width
    int plateau = 0, edges = 0;
    for (double v : peakProfile) {
      if (v > 1.0 - 1e-12)
        ++plateau;
      else if (v > 1e-12)
        ++edges;
    }
    std::printf("C  slug at its fullest: sum(C) %.6g over %d full planes + %d partial (expected "
                "length %.4g cells)\n",
                peak, plateau, edges, peak / (nx * ny));
  }
  const auto d = s.vofDiagnostics();
  const auto tot = s.vofBcVolumesTotal();
  const double injected = tot[4];
  const double rel = (injected > 0.0) ? drift / injected : drift;
  std::printf(
      "C  budget: injected %.10g, left %.10g, remaining sum(C) %.3e; |budget drift| %.3e "
      "(rel %.3e)\n",
      injected, -tot[5], d.sumC, drift, rel);
  std::printf("C  C in [%.3e, %.17g];  max|div(open u)| of the prescribed field %.3e\n", minC, maxC,
              s.maxOpenDivergence());
  // the slug's length is preserved: what went in came out
  std::printf("C  in %.10g vs out %.10g (relative closure %.3e)\n", injected, -tot[5],
              std::fabs(injected + tot[5]) / injected);
  CHECK(rel < 1e-12);
  CHECK(minC > -1e-12);
  CHECK(maxC < 1.0 + 1e-12);
  CHECK(d.sumC < 1e-9 * injected);  // the slug left; nothing behind
  CHECK(injected > 0.0);
  CHECK(std::fabs(injected + tot[5]) / injected < 1e-9);
}

// ----------------------------------------------------------- D: the inflow property ghost (item 5)
void propGhostGate() {
  const int nx = 8, ny = 8, nz = 16;
  const double rhoG = 1.0, rhoL = 1000.0;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(rhoL);
  s.setMu(1.0);
  s.setDt(0.05);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, 0.5);
  s.setDomainBc(5, 3, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 10.0));
  s.setPressureChebyshev(true, 400, 1e-12);
  s.enableVof();
  s.setVof(std::vector<double>((std::size_t)nx * ny * nz, 0.0));  // a GAS domain
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                     {rhoG, rhoL - rhoG, 0.0, 0.0});
  s.setVofInflow(4, 1.0);  // ... fed by a LIQUID inlet
  s.step();

  auto rv = s.fieldView("rho");
  auto h = Kokkos::create_mirror_view(rv);
  Kokkos::deep_copy(h, rv);
  const int G = 2, ex = nx + 2 * G, ey = ny + 2 * G;
  auto at = [&](int x, int y, int z) { return h(idx(x, y, z, ex, ey)); };
  const double ghost = at(G + nx / 2, G + ny / 2, G - 1);  // first ghost outside the -z inflow
  const double inner = at(G + nx / 2, G + ny / 2, G);      // first inner cell (gas)
  const double face = 0.5 * (ghost + inner);
  std::printf("D  inflow -z: rho_ghost %.6g (want %.6g), rho_inner %.6g, rho_face %.6g\n", ghost,
              rhoL, inner, face);
  CHECK(std::fabs(ghost - rhoL) < 1e-10 * rhoL);
  // and the Neumann copy would have given rho_face == rho_gas; record the size of the repair
  std::printf("D  the Neumann-copy face density would have been %.6g (ratio %.4g)\n", inner,
              face / inner);
  CHECK(face / inner > 100.0);
}

// ------------------------------------------------------------------------------- E: backflow
// A pure kernel test of the inletOutlet rule: the sign convention (which velocity sign is "back
// INTO the domain" on each side), which ghost layers are written, and that a face where the fluid
// LEAVES keeps its zero-gradient band untouched. Doing it here rather than through a solve is
// deliberate: the quantity under test is the rule, and a solve would only reach it through a
// reversal that is itself hard to arrange reproducibly.
void backflowGate() {
  const int n = 8, g = 3;
  const I3 e{n + 2 * g, n + 2 * g, n + 2 * g};
  const long len = (long)e.x * e.y * e.z;
  for (int side = 0; side < 2; ++side) {
    SField c("c", len), w("w", len);
    Kokkos::deep_copy(c, 1.0);  // clampFill's zero-gradient band: liquid everywhere
    auto hw = Kokkos::create_mirror_view(w);
    Kokkos::deep_copy(hw, 0.0);
    // the advector's HIGH-face convention: the boundary z-face of side 0 is the +z face of the
    // last ghost cell (z-index g-1); of side 1 it is the +z face of the last inner cell.
    const int fa = (side == 0) ? (g - 1) : (e.z - g - 1);
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x)
        hw(L3(x, y, fa, e)) = (x < e.x / 2) ? -0.3 : +0.3;
    Kokkos::deep_copy(w, hw);
    peclet::flow::vof::bcColourBackflow(c, e, g, /*a=*/2, side, w, /*backflow=*/0.0);
    auto hc = Kokkos::create_mirror_view(c);
    Kokkos::deep_copy(hc, c);
    // inward is +z on side 0 and -z on side 1
    const int inwardHalf = (side == 0) ? 1 : 0;  // 1 => the x >= e.x/2 half has inward velocity
    long wrong = 0, layers = 0;
    const int lo = (side == 0) ? 0 : (e.z - g);
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x) {
        const bool inward = (x >= e.x / 2) == (inwardHalf == 1);
        for (int k = 0; k < g; ++k) {
          const double v = hc(L3(x, y, lo + k, e));
          if (v != (inward ? 0.0 : 1.0))
            ++wrong;
          if (inward && v == 0.0)
            ++layers;
        }
      }
    std::printf("E  backflow side %d: %ld cells disagree with the rule, %ld ghost cells took the "
                "backflow colour (all %d layers)\n",
                side, wrong, layers, g);
    CHECK(wrong == 0);
    CHECK(layers == (long)(e.x / 2) * e.y * g);
  }
}

// ------------------------------------------------- F: the variable-density outflow correction
void outflowVarGate() {
  const int n = 8, g = 2;
  const peclet::flow::B3 e{n + 2 * g, n + 2 * g, n + 2 * g};
  const long len = (long)e.x * e.y * e.z;
  peclet::flow::BField f0("f0", len), f1("f1", len), phi("phi", len), rho("rho", len);
  auto hp = Kokkos::create_mirror_view(phi);
  auto hr = Kokkos::create_mirror_view(rho);
  for (long i = 0; i < len; ++i) {
    hp(i) = 0.001 * (double)((i * 37) % 101);
    hr(i) = 1.0;
  }
  Kokkos::deep_copy(phi, hp);
  Kokkos::deep_copy(rho, hr);
  Kokkos::deep_copy(f0, 0.0);
  Kokkos::deep_copy(f1, 0.0);
  peclet::flow::bcCorrectOutflow(f0, phi, e, g, 2);
  peclet::flow::bcCorrectOutflowVar(f1, phi, rho, 1.0, e, g, 2, false);
  auto h0 = Kokkos::create_mirror_view(f0);
  auto h1 = Kokkos::create_mirror_view(f1);
  Kokkos::deep_copy(h0, f0);
  Kokkos::deep_copy(h1, f1);
  long diff = 0;
  for (long i = 0; i < len; ++i)
    if (!(h0(i) == h1(i)))
      ++diff;
  std::printf("F  bcCorrectOutflowVar at uniform rho == rho0: %ld bitwise disagreements with "
              "bcCorrectOutflow\n",
              diff);
  CHECK(diff == 0);

  // and at a real jump it must differ by exactly rho0/rho_f
  for (long i = 0; i < len; ++i)
    hr(i) = 1000.0;
  Kokkos::deep_copy(rho, hr);
  Kokkos::deep_copy(f1, 0.0);
  peclet::flow::bcCorrectOutflowVar(f1, phi, rho, 1.0, e, g, 2, false);
  Kokkos::deep_copy(h1, f1);
  double worst = 0.0;
  const long sa = (long)e.x * e.y;
  for (int y = g; y < e.y - g; ++y)
    for (int x = g; x < e.x - g; ++x) {
      const long bf = (long)x + (long)y * e.x + (long)(e.z - g) * sa;
      if (h0(bf) != 0.0)
        worst = std::fmax(worst, std::fabs(h1(bf) / h0(bf) - 1.0 / 1000.0));
    }
  std::printf("F  at rho == 1000 rho0 the correction is scaled by 1/1000 to %.3e\n", worst);
  CHECK(worst < 1e-14);
}

// ------------------------------------- F2: the outflow correction is REACHED by the solver
// Gate F proves the kernel. This proves the SOLVER calls it: two identical stratified outflow
// solvers differing only in `setOutflowRhoCorrection`, compared on the OUTFLOW BOUNDARY FACE
// itself. That face is the low face of the first GHOST cell, so `getVelocity()` (inner cells)
// cannot see it — reading it needs `fieldView("u")`. A gate that used getVelocity() would report
// "no difference" and be measuring the wrong quantity.
double outflowFaceProbe(bool corr, double ratio, double& divOut, double& uInner,
                        double& divMut) {
  const int nx = 32, ny = 4, nz = 16;
  const double rhoL = 100.0, rhoG = rhoL / ratio;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(rhoL);
  s.setMu(5.0);
  s.setDt(ratio > 1.0 ? 0.1 : 0.2);
  s.setDomainBc(4, 1, 0, 0, 0);
  s.setDomainBc(5, 1, 0, 0, 0);
  s.setDomainBc(0, 2, 0.1, 0, 0);
  s.setDomainBc(1, 3, 0, 0, 0);
  s.setVelocityIterations(60);
  s.setPressureLevels(6);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1e30));
  if (ratio > 1.0) {
    s.enableVof();
    std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
    for (int z = 0; z < nz / 2; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
          C[idx(x, y, z, nx, ny)] = 1.0;  // liquid in the lower half
    s.setVof(C);
    s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                       {rhoG, rhoL - rhoG, 0.0, 0.0});
  }
  s.setPressureFcg(true, 400, 1e-12);  // driver LAST (setDensityMode reselects Chebyshev)
  s.setOutflowRhoCorrection(corr);
  for (int i = 0; i < 5; ++i)
    s.step();
  auto uv = s.fieldView("u");
  auto h = Kokkos::create_mirror_view(uv);
  Kokkos::deep_copy(h, uv);  // BEFORE any diagnostic: maxOpenDivergence() would erase the face
  const int G = 2, ex = nx + 2 * G, ey = ny + 2 * G;
  const double face = h(idx(G + nx, G + ny / 2, G + nz - 1, ex, ey));  // gas side of the outlet
  uInner = h(idx(G + nx - 1, G + ny / 2, G + nz - 1, ex, ey));
  divOut = s.maxOpenDivergenceProjected();
  divMut = s.maxOpenDivergence();  // AFTER the projected read: this one mutates u
  return face;
}

void outflowReachedGate() {
  for (double ratio : {1.0, 10.0}) {
    double d0 = 0, d1 = 0, i0 = 0, i1 = 0, m0 = 0, m1 = 0;
    const double on = outflowFaceProbe(true, ratio, d0, i0, m0);
    const double off = outflowFaceProbe(false, ratio, d1, i1, m1);
    std::printf("F2 ratio %-6g outflow face u: with the 1/rho_f factor %.17g, without %.17g "
                "(delta %.3e)\n",
                ratio, on, off, on - off);
    std::printf("F2 ratio %-6g last inner u %.17g -> the face is %s the zero-gradient copy; "
                "max|div| projected %.3e / %.3e, mutating diagnostic %.3e / %.3e\n",
                ratio, i0, (on == i0) ? "STILL" : "NOT", d0, d1, m0, m1);
    if (ratio == 1.0) {  // constant density: the factor is exactly 1, both paths bitwise equal
      CHECK(on == off);
      CHECK(on != i0);   // ... but the projection's correction IS there
      CHECK(d0 < 1e-15);  // and the projected field is divergence-free to machine precision
    } else {
      CHECK(on != off);  // the solver must actually reach the sibling kernel
      CHECK(on != i0);
      // THE ITEM-4 VERDICT. The shipped default (no 1/rho_f, d1) must be the CONSISTENT one and
      // the ablation (d0) must be worse — the operator's outflow-face coefficient is the raw
      // openness, so the plain phi difference is the matching correction. See
      // IbmSolver::setOutflowRhoCorrection.
      CHECK(d1 < 1e-8);
      CHECK(d0 > 1e3 * d1);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    maskGate();
    fluxInertGate();
    outflowVarGate();
    outflowReachedGate();
    backflowGate();
    propGhostGate();
    const bool quick = std::getenv("PECLET_VOF_BC_QUICK") != nullptr;
    budgetGate(quick ? 32 : 64, quick ? 40 : 100, quick ? 200 : 400);
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
