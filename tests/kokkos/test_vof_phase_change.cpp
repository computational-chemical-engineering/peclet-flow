// VoF Part II, rungs P0 + P1 (WO-P01) — phase change in planar form.
//
// Gates, in the order they are run:
//
//   K1 THE PLIC POLYGON AREA. `plicArea` is the one genuinely new piece of geometry: the
//      regression removes `mdot * A * dt / rho_l` per cell, so an area that is wrong by a factor
//      is an interface that moves at the wrong speed and nothing downstream would notice. It is
//      checked against three CLOSED-FORM references (an axis-aligned plane 1, the face diagonal
//      sqrt(2), the centred hexagon 3 sqrt(3)/4 — the last two are the classic unit-cube
//      cross-sections) and against `|m|_2 dV/dalpha` by central difference over a sweep of
//      normals and volumes. The work order suggested the finite difference AS the implementation;
//      it is only exact where V(alpha) is locally linear, which is why the analytic derivative is
//      what ships and the difference is only the cross-check.
//
//   K2 THE ONE-SIDED GRADIENT FIT. On an analytic LINEAR temperature field the weighted
//      least-squares fit through the interface value must return the exact directional derivative,
//      for an arbitrary (non-axis-aligned) interface normal. This is a pure kernel statement: it
//      pins the sign of the normal distance (positive on the GAS side), the through-the-interface
//      form of the fit, and Malan's collinearity weight.
//
//   P0a REGRESSION ONLY (rho_g = rho_l, no source). A planar interface under a uniform prescribed
//      mdot, 1000 kinematic steps: the interface position must follow x(t) = x0 - mdot t/rho_l,
//      which is EXACT by construction, and the colour must stay in [0,1]. The gate is 1e-12 on
//      the position; the clip-and-redistribute ledger crosses twenty cell boundaries during the
//      run and is what makes it exact rather than merely close.
//
//   P0b THE DIVERGENCE SOURCE (density ratio 100). A CLOSED column — walls on +-x, periodic in
//      y/z — with the net vapour production balanced by a prescribed sink plane, so the domain is
//      compatible without an outflow (the varRho outflow operator is the subject of WO-R2). The
//      exact 1-D solution has the liquid at rest, the interfacial cell's own faces at the LIQUID
//      velocity (0) and the gas beyond the deposit cell at mdot(1/rho_g - 1/rho_l). All three are
//      gated, together with max|div(u) - S| at the projection floor.
//
//   P1  THE STEFAN PROBLEM. A vapour layer growing from a superheated wall into saturated liquid,
//      the mass flux computed from the one-sided gradients of a transported temperature with the
//      interfacial cells pinned at T_sat by the new per-cell Dirichlet mask. Gated here at N = 64
//      (the convergence ladder 64/128/256 lives in `tests/study/vof_stefan.py`).
//
//   INERT  With `mdot == 0` every phase-change kernel runs and the colour field must come back
//      BITWISE unchanged — the branch-level statement that the machinery adds nothing of itself.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
using peclet::flow::vof::plicAlpha;
using peclet::flow::vof::plicArea;
using peclet::flow::vof::plicVolume;
using peclet::flow::vof::hfAreaElement;          // WO-P3c
using peclet::flow::vof::hfSurfaceNormal;
using peclet::flow::vof::pvSurfaceNormal;
using peclet::flow::vof::interfaceAreaFromNormal;
using peclet::flow::vof::kAreaPlic;
using peclet::flow::vof::kAreaMetric;
using peclet::flow::vof::kAreaNormal;
using peclet::flow::vof::McVertex;                // WO-P3d
using peclet::flow::vof::mcCubeCornerArea;
using peclet::flow::vof::kMcSrcColour;
using peclet::flow::vof::kMcSrcPlic;
using peclet::flow::vof::kMcDepositCentroid;
using peclet::flow::vof::kMcDepositSplit;
using peclet::flow::vof::plicNormalizeL1;

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

// WO-P3c: run every scene below with a non-default interfacial-AREA geometry
// (`set_phase_change_area`) when `PECLET_P3C_AREA` is set, so the planar rungs can be re-taken on
// the cascade area without a second binary. Inert (and byte-identical) when the variable is unset.
template <class S>
void applyAreaModeEnv(S& s) {
  if (const char* e = std::getenv("PECLET_P3C_AREA"))
    s.setPhaseChangeArea(std::atoi(e));
}

// ============================================================ K1: the PLIC polygon area
void areaGate() {
  const double s2 = std::sqrt(2.0), hex = 3.0 * std::sqrt(3.0) / 4.0;
  CHECK(plicArea(1.0, 0.0, 0.0, plicAlpha(1.0, 0.0, 0.0, 0.5)) == 1.0);
  CHECK(plicArea(0.0, 0.0, -1.0, plicAlpha(0.0, 0.0, -1.0, 0.3)) == 1.0);
  CHECK(std::fabs(plicArea(0.5, 0.5, 0.0, plicAlpha(0.5, 0.5, 0.0, 0.5)) - s2) < 1e-15);
  const double t = 1.0 / 3.0;
  CHECK(std::fabs(plicArea(t, t, t, plicAlpha(t, t, t, 0.5)) - hex) < 1e-15);
  // a plane that misses the cell has no polygon
  CHECK(plicArea(1.0, 0.0, 0.0, -0.5) == 0.0);
  CHECK(plicArea(1.0, 0.0, 0.0, 1.5) == 0.0);

  const double ms[][3] = {{1, 0, 0},        {0.5, 0.5, 0},   {0.02, 0.98, 0},
                          {t, t, t},        {0.2, 0.3, 0.5}, {-0.2, 0.3, -0.5},
                          {1e-9, 0.4, 0.6}, {0.1, 0.1, 0.8}, {0.45, 0.45, 0.1}};
  double worst = 0.0;
  for (const auto& m : ms)
    for (double v = 0.02; v < 0.999; v += 0.02) {
      const double al = plicAlpha(m[0], m[1], m[2], v);
      const double q = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
      const double d = 1e-6;
      const double ref =
          q * (plicVolume(m[0], m[1], m[2], al + d) - plicVolume(m[0], m[1], m[2], al - d)) /
          (2 * d);
      worst = std::fmax(worst, std::fabs(plicArea(m[0], m[1], m[2], al) - ref));
    }
  std::printf("K1 plicArea: max |analytic - |m|2 dV/dalpha (FD)| = %.3e\n", worst);
  CHECK(worst < 1e-5);  // the FD's own truncation, not the kernel's error
}

// ============================================================ K6: the WO-P3d joined sheet
//
// `set_phase_change_area` 4..7 replaces the sum of per-cell pieces by ONE watertight sheet:
// marching tetrahedra (Kuhn's 6-tet split) on the dual cube whose 8 corners are cell centres. Two
// statements are gated here, both of which are what the construction is FOR:
//
//   (a) EXACT ON A PLANE, TILTED OR NOT. Give the 8 corners the exact signed distance of a plane
//       and that plane as their PLIC normal; the sheet inside the dual cube must then be exactly
//       the plane's cross-section of that cube, whose area is `plicArea` of the same plane on the
//       unit cube. This is the tilted-plane gate of the work order at kernel level, where it needs
//       no box, no ghost policy and no edge convention. (The C = 1/2 source cannot pass it and is
//       not asked to: `C(d)` is the SZ piecewise cubic, so linear interpolation of it along the
//       tets' sqrt(2) and sqrt(3) edges misplaces the vertex. It is asked for the axis-aligned
//       case, where `C` IS linear.)
//
//   (b) THE DEPOSIT IS A PARTITION. Summed over the 8 corners, both deposit rules return the same
//       total, and they still do when some corners are non-interfacial and their pieces are
//       RETARGETED — the retarget moves area between cells, never loses it. That identity is what
//       makes `Sigma_cells A` a property of the sheet and not of the booking rule.
void mcSheetGate() {
  const double ns[][3] = {{1, 0, 0},          {0, 0, -1},      {1, 1, 0},        {1, -1, 0},
                          {1, 1, 1},          {1, 2, 3},       {0.2, 0.3, 0.5},  {3, 1, -2},
                          {0.05, 1.0, 0.02},  {1, 0.5, 0.25}};
  double worstPlic = 0.0, worstSplit = 0.0, worstPart = 0.0, worstColourAxis = 0.0;
  int rows = 0, axisRows = 0;
  for (const auto& nn : ns) {
    const double q = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
    const double u[3] = {nn[0] / q, nn[1] / q, nn[2] / q};   // unit, gas-ward
    const bool axis = (std::fabs(u[0]) == 1.0 || std::fabs(u[1]) == 1.0 || std::fabs(u[2]) == 1.0);
    for (double off = -0.86; off < 0.87; off += 0.07) {
      // gas-positive distance at the dual cube's corner k, for the plane through the cube centre
      // displaced by `off` along u:  phi(x) = u . (x - (1/2,1/2,1/2)) - off
      McVertex v[8];
      bool crossed = false;
      for (int k = 0; k < 8; ++k) {
        double p[3];
        peclet::flow::vof::mcCornerPos(k, p);
        const double phi = u[0] * (p[0] - 0.5) + u[1] * (p[1] - 0.5) + u[2] * (p[2] - 0.5) - off;
        v[k].psi = phi;
        v[k].d = phi;
        v[k].n[0] = u[0];
        v[k].n[1] = u[1];
        v[k].n[2] = u[2];
        v[k].has = true;
        if (k && ((v[k].psi < 0.0) != (v[0].psi < 0.0)))
          crossed = true;
      }
      if (!crossed)
        continue;
      // the exact cross-section area: the same plane on the unit cube, L1-normalized for plicArea
      double m[3] = {u[0], u[1], u[2]};
      const double l1 = plicNormalizeL1(m);
      const double alpha = (0.5 * (m[0] + m[1] + m[2])) + off / l1;
      const double exact = plicArea(m[0], m[1], m[2], alpha);
      if (!(exact > 1e-6))
        continue;
      ++rows;
      double sumC = 0.0, sumS = 0.0;
      for (int lc = 0; lc < 8; ++lc) {
        sumC += mcCubeCornerArea(v, lc, kMcSrcPlic, kMcDepositCentroid);
        sumS += mcCubeCornerArea(v, lc, kMcSrcPlic, kMcDepositSplit);
      }
      worstPlic = std::fmax(worstPlic, std::fabs(sumC - exact) / exact);
      worstSplit = std::fmax(worstSplit, std::fabs(sumS - sumC) / exact);
      // the retarget is a permutation of the booking, not a loss: clear a few corners' `has`
      for (int drop = 1; drop < 8; drop += 3) {
        McVertex w[8];
        for (int k = 0; k < 8; ++k) {
          w[k] = v[k];
          if ((k & drop) == drop && k != 0)
            w[k].has = false;
        }
        double sc = 0.0, ss = 0.0;
        for (int lc = 0; lc < 8; ++lc) {
          sc += mcCubeCornerArea(w, lc, kMcSrcPlic, kMcDepositCentroid);
          ss += mcCubeCornerArea(w, lc, kMcSrcPlic, kMcDepositSplit);
        }
        worstPart = std::fmax(worstPart, std::fabs(sc - sumC) / exact);
        worstPart = std::fmax(worstPart, std::fabs(ss - sumC) / exact);
      }
      // the raw-C source, where it is licensed: an AXIS-ALIGNED plane, whose colour is linear in
      // the centre distance, so the interpolated crossing is the plane and the sheet is flat.
      if (axis && std::fabs(off) < 0.4) {
        ++axisRows;
        // for a unit normal along an axis the liquid fraction of a unit cell at gas-distance phi
        // is exactly clamp(1/2 - phi, 0, 1), so psi = 1/2 - C is the distance itself inside the
        // band and the linear interpolation of it IS the plane. (Outside the band C saturates,
        // which is why the sweep is restricted to a crossing well inside the cube.)
        for (int k = 0; k < 8; ++k) {
          double p[3];
          peclet::flow::vof::mcCornerPos(k, p);
          const double phi = u[0] * (p[0] - 0.5) + u[1] * (p[1] - 0.5) + u[2] * (p[2] - 0.5) - off;
          const double cc = std::fmin(1.0, std::fmax(0.0, 0.5 - phi));
          v[k].psi = 0.5 - cc;
        }
        double sa = 0.0;
        for (int lc = 0; lc < 8; ++lc)
          sa += mcCubeCornerArea(v, lc, kMcSrcColour, kMcDepositCentroid);
        worstColourAxis = std::fmax(worstColourAxis, std::fabs(sa - exact) / exact);
      }
    }
  }
  std::printf("K6 joined sheet on a PLANE (%d rows, %d axis-aligned): max rel |PLIC-source sum - "
              "exact| %.3e, |split - centroid| %.3e, |retargeted - plain| %.3e, "
              "|colour-source axis - exact| %.3e\n",
              rows, axisRows, worstPlic, worstSplit, worstPart, worstColourAxis);
  CHECK(rows > 100);
  // ~1.7e-14 relative, i.e. about 75 eps: the sheet's total is a sum of a dozen triangle areas,
  // each a square root of a cross product, so this is the round-off of the SUM and not an error of
  // the construction. The axis-aligned rows, where every factor is an exact 1, are BITWISE.
  CHECK(worstPlic < 1e-13);
  CHECK(worstSplit < 1e-13);
  CHECK(worstPart < 1e-13);
  CHECK(worstColourAxis == 0.0);
}

// ============================================================ K5: the WO-P3c area constructions
//
// A_Gamma may now come from the V3 curvature cascade's geometry instead of the MYC PLIC polygon
// (`set_phase_change_area`). The gate is that on a PLANE the three modes are the SAME number —
// which is what keeps every planar rung (P0a, P0b, P1, P2) where it was — and that the two new
// kernels are the exact objects they claim to be:
//
//   * `hfAreaElement` on the height patch a plane with normal n produces = |m|_2/|m_d| = 1/|n_d|,
//     the metric factor of the graph x_d = f(x_d1, x_d2);
//   * `hfSurfaceNormal` on the same patch returns +-n (the sign is free: `plicArea` is invariant
//     under m -> -m with alpha re-solved from the same colour, the two planes being point
//     reflections through the cell centre);
//   * `pvSurfaceNormal` of a paraboloid with no linear terms is the frame's own normal.
void areaModeGate() {
  const double ms[][3] = {{1, 0, 0},        {0, -1, 0},       {0.5, 0.5, 0},   {0.02, 0.98, 0},
                          {0.2, 0.3, 0.5},  {-0.2, 0.3, -0.5}, {0.1, 0.1, 0.8}, {0.45, 0.45, 0.1}};
  double worst1 = 0.0, worst2 = 0.0, worstMetric = 0.0, worstNrm = 0.0;
  int exactAxis = 0;
  for (const auto& m : ms) {
    const double q = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    // the column direction the cascade would pick: the largest |m_d|
    int d = 0;
    for (int k = 1; k < 3; ++k)
      if (std::fabs(m[k]) > std::fabs(m[d]))
        d = k;
    const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    for (double v = 0.05; v < 0.999; v += 0.05) {
      const double al = plicAlpha(m[0], m[1], m[2], v);
      // the 3x3 height patch of the SAME plane: x_d = (alpha - m_d1 x1 - m_d2 x2)/m_d
      double hh[9];
      for (int qq = 0; qq < 3; ++qq)
        for (int pp = 0; pp < 3; ++pp)
          hh[pp + 3 * qq] = -(m[d1] * (pp - 1) + m[d2] * (qq - 1)) / m[d];
      const double metric = hfAreaElement(hh);
      worstMetric = std::fmax(worstMetric, std::fabs(metric - q / std::fabs(m[d])));
      double ns[3];
      hfSurfaceNormal(hh, d, ns);
      const double dot = (ns[0] * m[0] + ns[1] * m[1] + ns[2] * m[2]) / q;
      worstNrm = std::fmax(worstNrm, std::fabs(std::fabs(dot) - 1.0));
      const double a0 = interfaceAreaFromNormal(kAreaPlic, m[0], m[1], m[2], al, v, ns);
      const double a1 = interfaceAreaFromNormal(kAreaMetric, m[0], m[1], m[2], al, v, ns);
      const double a2 = interfaceAreaFromNormal(kAreaNormal, m[0], m[1], m[2], al, v, ns);
      CHECK(a0 == plicArea(m[0], m[1], m[2], al));
      worst1 = std::fmax(worst1, std::fabs(a1 - a0) / a0);
      worst2 = std::fmax(worst2, std::fabs(a2 - a0) / a0);
      if (std::fabs(m[d]) == q) {  // axis-aligned: every factor is an exact 1.0
        if (a1 == a0 && a2 == a0)
          ++exactAxis;
        else
          CHECK(false);
      }
    }
  }
  std::printf("K5 area modes on a PLANE: max rel |metric - 1| %.3e, |normal| %.3e, "
              "mode1 vs mode0 %.3e, mode2 vs mode0 %.3e (%d axis-aligned rows BITWISE)\n",
              worstMetric, worstNrm, worst1, worst2, exactAxis);
  CHECK(worstMetric < 1e-15);
  CHECK(worstNrm < 1e-15);
  CHECK(worst1 < 1e-14);
  CHECK(worst2 < 1e-14);

  // the paraboloid branch: no linear terms => the frame normal itself
  const double t1[3] = {0.0, 1.0, 0.0}, t2[3] = {0.0, 0.0, 1.0}, nn[3] = {1.0, 0.0, 0.0};
  double a[6] = {0.1, 0.0, 0.0, 0.05, 0.0, 0.05}, ns[3];
  pvSurfaceNormal(a, t1, t2, nn, ns);
  CHECK(ns[0] == 1.0 && ns[1] == 0.0 && ns[2] == 0.0);
  a[1] = 0.3;  // a slope in t1 tilts it by exactly that much
  pvSurfaceNormal(a, t1, t2, nn, ns);
  CHECK(std::fabs(ns[1] + 0.3 / std::sqrt(1.09)) < 1e-15);
}

// ============================================================ K2: the one-sided gradient fit
void gradientGate() {
  // an arbitrary normal and an arbitrary linear field T = T0 + g . x
  double m[3] = {0.37, -0.51, 0.28};
  double n[3] = {1, 0, 0};
  peclet::flow::vof::pcUnitNormal(m[0], m[1], m[2], n);
  const double alpha = plicAlpha(m[0], m[1], m[2], 0.4);
  const double phic = peclet::flow::vof::pcCentreDistance(m[0], m[1], m[2], alpha);
  const double g[3] = {0.7, -1.3, 2.1};
  const double Tsat = 3.5;
  const double exact = g[0] * n[0] + g[1] * n[1] + g[2] * n[2];  // dT/dn
  // sample "pure gas" cells on the phi > 0 side of the 5^3 stencil
  peclet::flow::vof::PcGradFit fg;
  for (int dz = -2; dz <= 2; ++dz)
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx) {
        if (!dx && !dy && !dz)
          continue;
        const double w = peclet::flow::vof::pcGradWeight(dx, dy, dz, n);
        if (!(w > 0.0))
          continue;
        const double phi = peclet::flow::vof::pcOffsetDistance(phic, n, dx, dy, dz);
        if (!(phi > 0.0))
          continue;
        // T at that cell centre, expressed through its normal distance: the linear field restricted
        // to the sample points is T_sat + exact*phi + (tangential part), and a fit THROUGH the
        // interface value can only be exact if the tangential part is absent — so sample the
        // one-dimensional profile the model assumes.
        peclet::flow::vof::pcGradAdd(fg, w, phi, Tsat + exact * phi, Tsat);
      }
  const double fit = peclet::flow::vof::pcGradSolve(fg);
  std::printf("K2 one-sided gradient fit: %.17g vs exact %.17g (n = %.4f %.4f %.4f, %d samples)\n",
              fit, exact, n[0], n[1], n[2], fg.n);
  CHECK(fg.n > 10);
  CHECK(std::fabs(fit - exact) < 1e-14);
}

// ============================================================ K3: the quadratic one-sided fit
// The linear fit above is exact only on a LINEAR profile; a curved one biases it by O(T'' h), which
// (once the plane-anchored Dirichlet has removed the cell-centre mismatch) is the leading error of
// the whole rung. The quadratic fit must return the exact slope AT THE INTERFACE of a quadratic
// profile, and the linear one must not — the second half is what makes this a discriminating gate.
void quadraticGradientGate() {
  double m[3] = {0.37, -0.51, 0.28};
  double n[3] = {1, 0, 0};
  peclet::flow::vof::pcUnitNormal(m[0], m[1], m[2], n);
  const double alpha = plicAlpha(m[0], m[1], m[2], 0.4);
  const double phic = peclet::flow::vof::pcCentreDistance(m[0], m[1], m[2], alpha);
  const double Tsat = 3.5, g = 2.19, q = -0.83;  // T = Tsat + g phi + q phi^2
  peclet::flow::vof::PcGradFit f;
  for (int dz = -2; dz <= 2; ++dz)
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx) {
        if (!dx && !dy && !dz)
          continue;
        const double w = peclet::flow::vof::pcGradWeight(dx, dy, dz, n);
        if (!(w > 0.0))
          continue;
        const double phi = peclet::flow::vof::pcOffsetDistance(phic, n, dx, dy, dz);
        if (!(phi > 0.0))
          continue;
        peclet::flow::vof::pcGradAdd(f, w, phi, Tsat + g * phi + q * phi * phi, Tsat);
      }
  const double lin = peclet::flow::vof::pcGradSolve(f);
  const double quad = peclet::flow::vof::pcGradSolve2(f);
  std::printf("K3 quadratic fit on a curved profile: quad %.17g vs exact %.17g (lin %.17g, "
              "%d samples)\n", quad, g, lin, f.n);
  CHECK(std::fabs(quad - g) < 1e-13);
  CHECK(std::fabs(lin - g) > 0.1);  // the linear fit is genuinely biased here
}

// ============================================================ K4: the plane-anchored GFM distance
void gfmThetaGate() {
  // A grid-aligned plane: liquid at high x, so n = -x_hat and phi_c = 1/2 - C. The pure GAS cell is
  // at i - e_x (step s = +1 from it to the interfacial cell) and sits phi_c + 1 = 3/2 - C cells from
  // the plane; the pure LIQUID cell is at i + e_x (s = -1) and sits 1/2 + C cells from it.
  for (double C : {0.15, 0.5, 0.87}) {
    const double phic = 0.5 - C, nd = -1.0;
    const double thGas = peclet::flow::vof::pcGfmTheta(phic, nd, +1.0, 0.1, 1.9);
    const double thLiq = peclet::flow::vof::pcGfmTheta(phic, nd, -1.0, 0.1, 1.9);
    std::printf("K4 GFM theta at C = %.2f: gas %.17g (exact %.17g), liquid %.17g (exact %.17g)\n",
                C, thGas, 1.5 - C, thLiq, 0.5 + C);
    CHECK(std::fabs(thGas - (1.5 - C)) < 1e-15);
    CHECK(std::fabs(thLiq - (0.5 + C)) < 1e-15);
  }
  // A plane nearly PARALLEL to this face's axis puts the interface far along the grid line, so the
  // limit is thetaMax and the function must reach it CONTINUOUSLY — a jump there is a round-off
  // amplifier (see the note on `pcGfmTheta`; measured 4.5e-4 vs 7e-14 on the P2 MPI gate).
  CHECK(peclet::flow::vof::pcGfmTheta(0.3, 0.0, 1.0, 0.1, 1.9) == 1.9);
  double prev = peclet::flow::vof::pcGfmTheta(0.3, 1e-3, 1.0, 0.1, 1.9);
  for (double nd = 1e-3; nd > 1e-14; nd *= 0.5) {
    const double th = peclet::flow::vof::pcGfmTheta(0.3, nd, 1.0, 0.1, 1.9);
    CHECK(std::fabs(th - prev) < 1e-3);  // no jump anywhere on the way to the limit
    prev = th;
  }
  std::printf("K4 GFM theta continuity to n_d -> 0: limit %.17g (thetaMax 1.9)\n", prev);
}

// ============================================================ P0a: regression only
void p0a() {
  const int nx = 64, ny = 4, nz = 4;
  const double x0 = 32.25, mdot = 0.02, dt = 1.0;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(1.0);
  s.setMu(0.01);
  s.setDt(dt);
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, x0 - x));
  s.enableVof();
  s.setVof(C);
  s.enablePhaseChange(1.0, 1.0, 1.0);
  applyAreaModeEnv(s);
  s.setMassFluxUniform(mdot);
  auto pos = [&]() {
    const auto c = s.getVof();
    double sum = 0;
    for (double v : c)
      sum += v;
    return sum / (ny * nz);
  };
  double worst = 0.0, lo = 1e30, hi = -1e30;
  long clips = 0;
  double redist = 0.0;
  for (int k = 1; k <= 1000; ++k) {
    s.applyPhaseChange(dt);
    worst = std::fmax(worst, std::fabs(pos() - (x0 - mdot * dt * k)));
    const auto d = s.phaseChangeDiagnostics();
    lo = std::fmin(lo, d.minC);
    hi = std::fmax(hi, d.maxC);
    clips += d.deficitCells;
    redist += d.redistributed;
  }
  std::printf(
      "P0a planar regression, 1000 steps: max |x - (x0 - mdot t/rho_l)| = %.3e ; C in [%.17g, "
      "%.17g] ; %ld cell-crossings clipped, |redistributed| = %.6g\n",
      worst, lo, hi, clips, redist);
  CHECK(worst < 1e-12);
  CHECK(lo >= 0.0);
  CHECK(hi <= 1.0);
  CHECK(clips > 0);  // the run MUST cross cell boundaries or it never exercises the redistribution
}

// ============================================================ P0b: the divergence source
void p0b() {
  const int nx = 64, ny = 4, nz = 4;
  const double rg = 1.0, rl = 100.0, mdot = 0.01, x0 = 32.25;
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(rg);
  s.setMu(1e-3);
  s.setDt(1.0);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1.0));
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, x0 - x));
  s.enableVof();
  s.setVof(C);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {rg, rl - rg});
  s.enablePhaseChange(rg, rl, 1.0);
  applyAreaModeEnv(s);
  s.setMassFluxUniform(mdot);
  const double Sc = mdot * (1.0 / rg - 1.0 / rl);  // per interfacial cell (A = 1 here)
  std::vector<double> sink((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      sink[idx(nx - 6, y, z, nx, ny)] = -Sc;
  s.setDivergenceSource(sink);
  s.setPressureFcg(true, 400, 1e-12);
  for (int k = 0; k < 20; ++k)
    s.step();
  const auto u = s.getVelocity(0);
  const auto src = s.getField("pc_source");
  const auto usr = s.getField("div_source");
  const auto v = s.getVelocity(1), w = s.getVelocity(2);
  const double ug = mdot * (1.0 / rg - 1.0 / rl);
  double liq = 0.0, plat = 0.0, spread = 0.0, dmax = 0.0;
  double p0 = u[idx(40, 0, 0, nx, ny)];
  for (int x = 2; x < 30; ++x)
    liq = std::fmax(liq, std::fabs(u[idx(x, 0, 0, nx, ny)]));
  for (int x = 35; x < 56; ++x) {
    plat += u[idx(x, 0, 0, nx, ny)] / 21.0;
    spread = std::fmax(spread, std::fabs(u[idx(x, 0, 0, nx, ny)] - p0));
  }
  for (int z = 0; z + 1 < nz; ++z)
    for (int y = 0; y + 1 < ny; ++y)
      for (int x = 0; x + 1 < nx; ++x) {
        const double d = (u[idx(x + 1, y, z, nx, ny)] - u[idx(x, y, z, nx, ny)]) +
                         (v[idx(x, y + 1, z, nx, ny)] - v[idx(x, y, z, nx, ny)]) +
                         (w[idx(x, y, z + 1, nx, ny)] - w[idx(x, y, z, nx, ny)]);
        dmax =
            std::fmax(dmax, std::fabs(d - src[idx(x, y, z, nx, ny)] - usr[idx(x, y, z, nx, ny)]));
      }
  const auto d = s.phaseChangeDiagnostics();
  std::printf(
      "P0b ratio 100, closed column + balanced sink: u_gas = %.17g (exact %.17g, rel %.3e), "
      "plateau spread %.3e, max|u_liquid| %.3e, max|div - S| %.3e, source sum %.6g into %ld "
      "cells, pressure iters %d\n",
      plat, ug, (plat - ug) / ug, spread, liq, dmax, d.sourceSum, d.sourceCells,
      s.lastPressureIterations());
  CHECK(std::fabs((plat - ug) / ug) < 1e-10);
  CHECK(liq < 1e-12 * ug + 1e-15);
  CHECK(dmax < 1e-12);
  CHECK(s.lastPressureIterations() < 400);
  // the interfacial cell's own faces carry the LIQUID velocity
  CHECK(std::fabs(u[idx(32, 0, 0, nx, ny)]) < 1e-15);
  CHECK(std::fabs(u[idx(33, 0, 0, nx, ny)]) < 1e-15);
}

// ============================================================ P1: the Stefan problem
double stefanLambda(double St) {  // lambda e^{lambda^2} erf(lambda) = St/sqrt(pi)
  const double rhs = St / std::sqrt(M_PI);
  double a = 1e-8, b = 5.0;
  for (int i = 0; i < 200; ++i) {
    const double m = 0.5 * (a + b);
    ((m * std::exp(m * m) * std::erf(m) - rhs) > 0 ? b : a) = m;
  }
  return 0.5 * (a + b);
}

double stefanRun(int N, double& exact) {
  const double St = 1.0, alpha = 1.0, x0p = 0.10, xep = 0.25, Fo = 0.5;
  const double lam = stefanLambda(St);
  const double t0 = (x0p / (2 * lam)) * (x0p / (2 * lam)) / alpha;
  const double te = (xep / (2 * lam)) * (xep / (2 * lam)) / alpha;
  const int ny = 4, nz = 4;
  const double D = alpha * N * N;
  int ns = (int)std::llround((te - t0) / (Fo / D));
  const double dt = (te - t0) / ns;
  peclet::flow::IbmSolver s(N, ny, nz);
  s.setRho(1.0);
  s.setMu(1e-3);
  s.setDt(dt);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * ny * nz, 1.0));
  const double xg = x0p * N;  // interface position in cells; LIQUID at high x
  std::vector<double> C((std::size_t)N * ny * nz, 0.0), T((std::size_t)N * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < N; ++x) {
        C[idx(x, y, z, N, ny)] = std::fmin(1.0, std::fmax(0.0, (x + 1) - xg));
        const double xp = (x + 0.5) / N;
        T[idx(x, y, z, N, ny)] = xp < x0p ? 1.0 - std::erf(lam * xp / x0p) / std::erf(lam) : 0.0;
      }
  s.enableVof();
  s.setVof(C);
  s.addScalar("T", D, 1, 60);
  s.setScalarBc("T", 0, 2, 1.0);  // superheated wall
  s.setScalarBc("T", 1, 2, 0.0);  // saturated far field
  s.setField("T", T);
  s.enablePhaseChange(1.0, 1.0, 1.0);
  applyAreaModeEnv(s);
  s.setPhaseChangeThermal("T", 0.0, D, D, 0.0);
  // The rung P0/P1 treatment, pinned explicitly: WO-P23 makes the plane-anchored Dirichlet and the
  // quadratic fit the DEFAULTS, and this function is kept as the ablation that reproduces the P01
  // record (its P23 twin is `stefanRunP23`).
  s.setPhaseChangePlaneDirichlet(false);
  s.setPhaseChangeQuadraticFit(false);
  for (int k = 0; k < ns; ++k) {
    s.applyPhaseChange(dt);
    s.advanceScalars();
  }
  const auto c = s.getVof();
  double sum = 0;
  for (double q : c)
    sum += q;
  exact = 2 * lam * std::sqrt(alpha * te) * N;
  const auto d = s.phaseChangeDiagnostics();
  std::printf(
      "P1 Stefan N=%d (%d steps, St=1, lambda=%.6f): layer = %.5f cells, exact %.5f, rel %+.4f %% "
      "; C in [%.3e, %.17g], unresolved %.3e, fallback %ld\n",
      N, ns, lam, N - sum / (ny * nz), exact, 100.0 * ((N - sum / (ny * nz)) - exact) / exact,
      d.minC, d.maxC, d.unresolved, d.fallbackCells);
  CHECK(d.minC >= 0.0);
  CHECK(d.maxC <= 1.0);
  return N - sum / (ny * nz);
}

void p1() {
  double exact = 0;
  const double x = stefanRun(64, exact);
  CHECK(std::fabs(x - exact) / exact < 0.02);  // 1.31 % at N = 64; the ladder is in tests/study
}

// ============================================================ P1 with the WO-P23 defaults
// The same problem with the plane-anchored (ghost-fluid) Dirichlet and the quadratic one-sided fit.
// This is a DISCRIMINATING gate, not a repeat: the rung P0/P1 treatment is 1.31 % at N = 64 and the
// two together are two orders of magnitude better, so a tolerance that only the pair can meet is
// the statement.
double stefanRunP23(int N, double& exact) {
  const double St = 1.0, alpha = 1.0, x0p = 0.10, xep = 0.25, Fo = 0.5;
  const double lam = stefanLambda(St);
  const double t0 = (x0p / (2 * lam)) * (x0p / (2 * lam)) / alpha;
  const double te = (xep / (2 * lam)) * (xep / (2 * lam)) / alpha;
  const int ny = 4, nz = 4;
  const double D = alpha * N * N;
  int ns = (int)std::llround((te - t0) / (Fo / D));
  const double dt = (te - t0) / ns;
  peclet::flow::IbmSolver s(N, ny, nz);
  s.setRho(1.0);
  s.setMu(1e-3);
  s.setDt(dt);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 1, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * ny * nz, 1.0));
  const double xg = x0p * N;
  std::vector<double> C((std::size_t)N * ny * nz, 0.0), T((std::size_t)N * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < N; ++x) {
        C[idx(x, y, z, N, ny)] = std::fmin(1.0, std::fmax(0.0, (x + 1) - xg));
        const double xp = (x + 0.5) / N;
        T[idx(x, y, z, N, ny)] = xp < x0p ? 1.0 - std::erf(lam * xp / x0p) / std::erf(lam) : 0.0;
      }
  s.enableVof();
  s.setVof(C);
  s.addScalar("T", D, 1, 60);
  s.setScalarBc("T", 0, 2, 1.0);
  s.setScalarBc("T", 1, 2, 0.0);
  s.setField("T", T);
  s.enablePhaseChange(1.0, 1.0, 1.0);
  applyAreaModeEnv(s);
  s.setPhaseChangeThermal("T", 0.0, D, D, 0.0);
  s.setPhaseChangePlaneDirichlet(true);
  s.setPhaseChangeQuadraticFit(true);
  for (int k = 0; k < ns; ++k) {
    s.applyPhaseChange(dt);
    s.advanceScalars();
  }
  const auto c = s.getVof();
  double sum = 0;
  for (double q : c)
    sum += q;
  exact = 2 * lam * std::sqrt(alpha * te) * N;
  const double layer = N - sum / (ny * nz);
  const auto d = s.phaseChangeDiagnostics();
  std::printf("P1' Stefan N=%d, PLANE-ANCHORED Dirichlet + QUADRATIC fit: layer = %.5f cells, "
              "exact %.5f, rel %+.4f %% ; C in [%.3e, %.17g]\n",
              N, layer, exact, 100.0 * (layer - exact) / exact, d.minC, d.maxC);
  CHECK(d.minC >= 0.0);
  CHECK(d.maxC <= 1.0);
  return layer;
}

void p1p23() {
  double exact = 0;
  const double x = stefanRunP23(64, exact);
  CHECK(std::fabs(x - exact) / exact < 0.001);  // measured -0.014 %; the P0/P1 form is +1.31 %
}

// ============================================================ the consistent rho c_p T transport
// The decisive identity (the energy twin of WO-K's uniform-velocity gate): a UNIFORM temperature
// must survive an arbitrary sharp colour field advected by a uniform velocity, at ANY heat capacity
// ratio. It does iff the heat content is carried by the SAME geometric flux as the colour and
// recovered with the capacity of the SAME updated colour. An inconsistent scheme is wrong at
// O(d(rho c_p)).
void energyIdentity(double rcpRatio) {
  const int nx = 32, ny = 8, nz = 8;
  const double T0 = 3.25, U = 0.17;
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0), T((std::size_t)nx * ny * nz, T0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)  // an arbitrary sharp, tilted colour field
        C[idx(x, y, z, nx, ny)] =
            std::fmin(1.0, std::fmax(0.0, 0.37 * x + 0.21 * y - 0.13 * z - 3.4));
  peclet::flow::IbmSolver s(nx, ny, nz);
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(1.0);
  s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1.0));
  s.enableVof();
  s.setVof(C);
  s.addScalar("T", 0.0, 1, 1);
  s.setField("T", T);
  s.setVelocity(0, std::vector<double>((std::size_t)nx * ny * nz, U));
  s.enablePhaseChange(1.0, 1.0, 1.0);
  applyAreaModeEnv(s);
  s.setPhaseChangeThermal("T", T0, 1.0, 1.0, 0.0);  // T_sat = T0: a uniform field has zero mdot
  s.setPhaseChangeEnergy(1.0, rcpRatio);
  double worst = 0.0;
  for (int k = 0; k < 20; ++k) {
    s.advectVofKinematic(1.0);
    const auto t = s.getField("T");
    for (double q : t)
      worst = std::fmax(worst, std::fabs(q - T0));
  }
  std::printf("ENERGY uniform-T identity at rho c_p ratio %g: max |T - T0| = %.3e over 20 "
              "kinematic steps\n", rcpRatio, worst);
  CHECK(worst == 0.0);  // the deviation form makes this BITWISE, not "small"
}

// ============================================================ P2: the sucking interface

double suckingB(double ja, double rr) {
  double b = ja / std::sqrt(M_PI);
  for (int i = 0; i < 300; ++i) {
    const double g = b * rr;
    b = ja * std::exp(-g * g) / (std::sqrt(M_PI) * std::erfc(g));
  }
  return b;
}

// Welch & Wilson (JCP 160:662, 2000): saturated vapour against a wall, SUPERHEATED liquid beyond,
// the interface moving into the liquid and the liquid pushed out through an outlet. All of mdot
// comes from the LIQUID side, so this is the gate on the liquid half of the fit, the per-phase
// closures and the consistent rho c_p T transport at once. Full ladder in tests/study/vof_sucking.py.
void p2(int N) {
  const double ratio = 10.0, ja = 1.0, alpha_l = 1.0, x0p = 0.10, xep = 0.25, Fo = 0.5, cfl = 0.2;
  const double rr = 1.0 / ratio, b = suckingB(ja, rr);
  const double t0 = (x0p / (2 * b)) * (x0p / (2 * b)) / alpha_l;
  const double te = (xep / (2 * b)) * (xep / (2 * b)) / alpha_l;
  const int ny = 4, nz = 4;
  const double al = alpha_l * N * N, rho_l = 1.0, rho_v = rr, cpl = 1.0;
  const double k_l = al * rho_l * cpl, k_v = k_l / ratio, dT = 1.0;
  const double h_lv = rho_l * cpl * dT / (rho_v * ja);
  peclet::flow::IbmSolver s(N, ny, nz);
  s.setRho(rho_l);  // the OUTLET phase: makes the boundary coefficient the varRho one exactly
  s.setMu(1e-3);
  s.setDomainBc(0, 1, 0, 0, 0);
  s.setDomainBc(1, 3, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)N * ny * nz, 1.0));
  const double X0 = x0p * N;
  std::vector<double> C((std::size_t)N * ny * nz, 0.0), T((std::size_t)N * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < N; ++x) {
        C[idx(x, y, z, N, ny)] = std::fmin(1.0, std::fmax(0.0, (x + 1) - X0));
        const double xp = (x + 0.5) / N;
        if (xp > x0p) {
          const double sv = xp / (2 * std::sqrt(alpha_l * t0)) - b * (1.0 - rr);
          T[idx(x, y, z, N, ny)] = dT - dT * std::erfc(sv) / std::erfc(b * rr);
        }
      }
  s.enableVof();
  s.setVof(C);
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {rho_v, rho_l - rho_v});
  s.setPressureFcg(true, 4000, 1e-10);
  s.addScalar("T", al, 1, 60);
  s.setScalarBc("T", 0, 2, 0.0);
  auto farT = [&](double tt) {
    const double sv = 1.0 / (2 * std::sqrt(alpha_l * tt)) - b * (1.0 - rr);
    return dT - dT * std::erfc(sv) / std::erfc(b * rr);
  };
  s.setScalarBc("T", 1, 2, farT(t0));
  s.setField("T", T);
  s.enablePhaseChange(rho_v, rho_l, h_lv);
  applyAreaModeEnv(s);
  s.setPhaseChangeThermal("T", 0.0, k_v, k_l, 0.0);
  s.setPhaseChangeEnergy(rho_v * cpl, rho_l * cpl);
  double tcur = t0;
  int ns = 0, capped = 0;
  long itmax = 0;
  while (tcur < te) {
    const double Xd = b * std::sqrt(alpha_l / tcur) * N;
    double dt = std::fmin(cfl / std::fmax(Xd * (1.0 - rr), 1e-30), Fo / al);
    dt = std::fmin(dt, te - tcur);
    s.setDt(dt);
    s.setScalarBc("T", 1, 2, farT(tcur + dt));
    s.step();
    tcur += dt;
    ++ns;
    itmax = std::max<long>(itmax, s.lastPressureIterations());
    capped += s.lastPressureIterations() >= 4000 ? 1 : 0;
  }
  const auto c = s.getVof();
  double sum = 0;
  for (double q : c)
    sum += q;
  const double layer = N - sum / (ny * nz);
  const double exact = 2 * b * std::sqrt(alpha_l * te) * N;
  const auto d = s.phaseChangeDiagnostics();
  std::printf("P2 sucking interface N=%d (%d steps, ratio %g, Ja %g, b = %.6f): layer = %.5f "
              "cells, exact %.5f, rel %+.4f %% ; pressure iters max %d/4000 (capped %d), "
              "band_div %.3e, C in [%.3e, %.17g]\n",
              N, ns, ratio, ja, b, layer, exact, 100.0 * (layer - exact) / exact, (int)itmax, capped,
              d.bandDiv, d.minC, d.maxC);
  CHECK(capped == 0);  // rule 3b
  CHECK(std::fabs(layer - exact) / exact < 0.01);
}

// ============================================================ inertness
void inert() {
  const int nx = 32, ny = 4, nz = 4;
  std::vector<double> C((std::size_t)nx * ny * nz, 0.0);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        C[idx(x, y, z, nx, ny)] = std::fmin(1.0, std::fmax(0.0, 16.4 - x));
  peclet::flow::IbmSolver a(nx, ny, nz), b(nx, ny, nz);
  a.setDt(1.0);
  b.setDt(1.0);
  a.enableVof();
  a.setVof(C);
  b.enableVof();
  b.setVof(C);
  b.enablePhaseChange(1.0, 1.0, 1.0);
  applyAreaModeEnv(b);
  b.setMassFluxUniform(0.0);
  for (int k = 0; k < 10; ++k)
    b.applyPhaseChange(1.0);
  const auto ca = a.getVof(), cb = b.getVof();
  double diff = 0;
  for (std::size_t i = 0; i < ca.size(); ++i)
    diff = std::fmax(diff, std::fabs(ca[i] - cb[i]));
  std::printf("INERT mdot == 0: max |C_pc - C_ref| = %.3e (bitwise expected)\n", diff);
  CHECK(diff == 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    areaGate();
    areaModeGate();
    mcSheetGate();
    gradientGate();
    quadraticGradientGate();
    gfmThetaGate();
    p0a();
    p0b();
    p1();
    p1p23();
    energyIdentity(1e4);
    p2(64);
    inert();
  }
  Kokkos::finalize();
  if (failures) {
    std::fprintf(stderr, "%d CHECK(s) failed\n", failures);
    return 1;
  }
  std::printf("all phase-change gates passed\n");
  return 0;
}
