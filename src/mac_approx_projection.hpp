/// @file
/// @brief flow — collocated approximate (MAC) projection helpers (Almgren–Bell–Colella style).
///
/// The collocated solver stores all three velocity components at the cell center. Its pressure
/// coupling is the approximate projection: average the cell velocities onto a face (MAC) field,
/// make THAT field divergence-free with the existing cut-cell pressure machinery (buildCutcellOp /
/// divergOpen / projectCorrect from mac_pressure.hpp), then correct the cell-centered velocities
/// with the central-difference pressure gradient. The face field is exactly divergence-free; the
/// cell field is only approximately so — hence "approximate projection". These two kernels are the
/// only collocated-specific projection pieces; everything else (operator, divergence, MG solve,
/// rotational pressure update) is shared with the staggered solver.
#ifndef PECLET_FLOW_MAC_APPROX_PROJECTION_HPP
#define PECLET_FLOW_MAC_APPROX_PROJECTION_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // peclet::flow::C3, CCField, CCConst, CCExec, ccSampleExt

namespace peclet::flow {

// const view of the (float) IBM stencil coefficients (same alias mac_ibm.hpp defines; this header
// is included first, so declare it here for the mode-4 stencilMatvec).
using MConst = Kokkos::View<const float*, CCMem>;

// Average cell-centered velocities onto the staggered face layout: uf(i) is the velocity at the low
// (-x) face of cell i (located at i-1/2) = ½(U(i-1)+U(i)); likewise vf/wf along y/z. This
// reproduces the exact layout the staggered solver stores directly, so divergOpen / projectCorrect
// (mac_pressure.hpp) act on uf/vf/wf unchanged. Computed over the block where the axis neighbour
// exists ([1,e) per axis); the cell velocity ghosts must be filled first.
inline void centerToFace(CCField uf, CCField vf, CCField wf, CCConst U, CCConst V, CCConst W, C3 e,
                         int g) {
  (void)g;
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::center_to_face", MD(space, {1, 1, 1}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        uf(i) = 0.5 * (U(i) + U(i - sx));
        vf(i) = 0.5 * (V(i) + V(i - sy));
        wf(i) = 0.5 * (W(i) + W(i - sz));
      });
}

// Wall-aware variant of centerToFace (opt-in: Solver::setFaceInterp(1); collocated only). At a face
// whose adjacent cell centers straddle the immersed boundary (one SDF negative), the plain average
// ½(U_fluid + U_solid=0) implicitly places the wall AT THE SOLID CELL CENTER (theta = 1) regardless
// of the true axis intercept theta = sdf_c/(sdf_c - sdf_n) — an O(1) relative flux error at every
// cut face that makes the collocated drag first-order (doc/collocated_first_order_analysis.md).
// Here the face value is reconstructed from the fluid side by a WALL-ANCHORED quadratic in
// (theta - s) — exactly zero at the wall — least-squares-fitted to the fluid-side samples
// {U_c, U_far, U_far2} with the near-wall cell SHED smoothly as theta -> 0 (weight
// theta²/(theta²+0.15²)): a sliver cell's value is slaved to the wall by the momentum IBM and
// carries no independent information, so downweighting it bounds the noise amplification without
// dropping the order — the theta->0 limit is the (still 2nd-order) quadratic through
// {wall, far, far2}. A small Tikhonov term on the curvature keeps the 2x2 solve well-posed when
// far2 is unavailable; a fluid cell sandwiched along the axis falls back to the wall-anchored
// linear through U_c (bounded: xe/theta <= 1/2). Faces whose center lies behind the wall
// (theta <= 1/2) get 0, continuously (the parabola vanishes at the wall); their open area is a
// corner sliver. Interior fluid faces reproduce the plain average bit-exactly. Validated a priori
// against analytic Stokes flow past a sphere: face-value error O(h²) (the plain average is O(h)).
// Per-face stencil of the wall-aware map: up to 3 (cell index, weight) pairs such that
// uf(face) = sum_k w[k]*U(idx[k]); returns the entry count (0 = closed face or corner sliver:
// uf = 0). Shared by the forward map (centerToFaceWallAware) and its TRANSPOSE
// (transposeGradWallAware) so the two stay exact adjoints — the accuracy of the constrained steady
// state hinges on that pairing, not on the forward truncation alone.
KOKKOS_INLINE_FUNCTION int wallAwareFaceStencil(CCConst sdf, long i, long sa, int cFace, int eAxis,
                                                double xcen, long idx[3], double w[3]) {
  const double sm = sdf(i - sa), sp = sdf(i);  // low cell (coord cFace-1) / high cell (cFace)
  const bool fm = sm >= 0.0, fp = sp >= 0.0;
  if (fm && fp) {  // both fluid: the plain 1/2-1/2 average, unchanged
    idx[0] = i;
    w[0] = 0.5;
    idx[1] = i - sa;
    w[1] = 0.5;
    return 2;
  }
  if (!fm && !fp)
    return 0;                       // fully solid face (openness 0; value irrelevant)
  const long ic = fp ? i : i - sa;  // the fluid cell
  const double sc = fp ? sp : sm, ss = fp ? sm : sp;
  double th = sc / (sc - ss);  // axis intercept from the fluid center (ibmFillEntry convention)
  th = th < 1e-4 ? 1e-4 : (th > 1.0 ? 1.0 : th);
  // Evaluation abscissa in wall coordinates (theta - s): the face-CENTER distance theta - 1/2 by
  // default, or (xcen >= 0, mode 3) the OPEN-FACE-CENTROID wall distance from the static geometry
  // build. The projection's flux model is o_f * uf, so the correct uf is the open-area MEAN of the
  // velocity — the midpoint-at-centroid quadrature — not the face-center point value: the open area
  // lies on the fluid side, farther from the wall, so the (accurate) face-center value UNDERCOUNTS
  // the flux at O(h). Measured a priori (Stokes-past-sphere): centroid evaluation drops the
  // systematic flux bias by ~2 orders of magnitude at N=128.
  const double xe = (xcen >= 0.0) ? xcen : th - 0.5;
  if (xe <= 0.0)
    return 0;                             // face (or its open centroid) behind the wall
  const int cc = fp ? cFace : cFace - 1;  // fluid cell's axis coordinate
  const int up = fp ? 1 : -1;             // coordinate step AWAY from the solid
  const long dir = fp ? -sa : sa;         // index step TOWARD the solid
  const bool ok1 = (cc + up >= 0) && (cc + up < eAxis) && sdf(ic - dir) >= 0.0;
  if (!ok1) {  // sandwiched along this axis: wall-anchored linear through U_c
    idx[0] = ic;
    w[0] = Kokkos::fmin(xe / th, 2.0);  // capped (xe can exceed theta at centroid evaluation)
    return 1;
  }
  const bool ok2 = (cc + 2 * up >= 0) && (cc + 2 * up < eAxis) && sdf(ic - 2 * dir) >= 0.0;
  const double wc = th * th / (th * th + 0.0225);  // shed the sliver cell (theta* = 0.15)
  const double w2 = ok2 ? 0.25 : 0.0;
  const double xc = th, xf = th + 1.0, xg = th + 2.0;
  const double S11 = wc * xc * xc + xf * xf + w2 * xg * xg;
  const double S12 = wc * xc * xc * xc + xf * xf * xf + w2 * xg * xg * xg;
  const double S22 = wc * xc * xc * xc * xc + xf * xf * xf * xf + w2 * xg * xg * xg * xg + 0.01;
  const double det = S11 * S22 - S12 * S12;
  // coefficient of datum k (abscissa x_k, weight w_k) in T = a*xe + b*xe^2:
  //   c_k = w_k * [ x_k*(S22*xe - S12*xe^2) + x_k^2*(S11*xe^2 - S12*xe) ] / det
  const double A = (S22 * xe - S12 * xe * xe) / det, B = (S11 * xe * xe - S12 * xe) / det;
  idx[0] = ic;
  w[0] = wc * (xc * A + xc * xc * B);
  idx[1] = ic - dir;
  w[1] = xf * A + xf * xf * B;
  if (!ok2)
    return 2;
  idx[2] = ic - 2 * dir;
  w[2] = w2 * (xg * A + xg * xg * B);
  return 3;
}

KOKKOS_INLINE_FUNCTION double wallAwareFaceValue(CCConst U, CCConst sdf, long i, long sa, int cFace,
                                                 int eAxis, double xcen) {
  long idx[3];
  double w[3];
  const int n = wallAwareFaceStencil(sdf, i, sa, cFace, eAxis, xcen, idx, w);
  double v = 0.0;
  for (int k = 0; k < n; ++k)
    v += w[k] * U(idx[k]);
  return v;
}

// Static per-face geometry for the mode-3 centroid quadrature: for each face whose adjacent cell
// centers straddle the boundary, subsample the face plane (4x4 trilinear SDF), locate the OPEN-area
// centroid, and store its axis wall-distance (in cells, measured toward the solid side; two-point
// line intercept from the trilinear SDF, clamped to [0,2]). 0 => no open samples (no flux). Faces
// with both centers on one side store -1 (sentinel: stencil falls back to theta-1/2, which the
// both-fluid branch never reads anyway). One-time cost at setSolid; the geometry is static.
inline void buildFaceCentroidDist(CCField xcx, CCField xcy, CCField xcz, CCConst sdf, C3 e) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::face_centroid_dist", MD(space, {1, 1, 1}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        // face center in the sampling frame where CELL CENTERS sit at integer coordinates
        // (buildOpenness convention): -x face of cell (x,y,z) is at (x-1/2, y, z), etc.
        const double fc[3][3] = {{x - 0.5, (double)y, (double)z},
                                 {(double)x, y - 0.5, (double)z},
                                 {(double)x, (double)y, z - 0.5}};
        const long st[3] = {sx, sy, sz};
        CCField out[3] = {xcx, xcy, xcz};
        for (int a = 0; a < 3; ++a) {
          const double sm = sdf(i - st[a]), sp = sdf(i);
          if ((sm >= 0.0) == (sp >= 0.0)) {
            out[a](i) = -1.0;  // not a solid-bordering face: stencil never reads it
            continue;
          }
          const double dir = (sm < 0.0) ? -1.0 : 1.0;  // axis step TOWARD the solid side
          const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
          double c1 = 0.0, c2 = 0.0;
          int nOpen = 0;
          for (int j1 = 0; j1 < 4; ++j1)
            for (int j2 = 0; j2 < 4; ++j2) {
              const double o1 = -0.375 + 0.25 * j1, o2 = -0.375 + 0.25 * j2;
              double p[3] = {fc[a][0], fc[a][1], fc[a][2]};
              p[t1] += o1;
              p[t2] += o2;
              if (ccSampleExt(sdf, e, p[0], p[1], p[2]) > 0.0) {
                c1 += o1;
                c2 += o2;
                ++nOpen;
              }
            }
          if (nOpen == 0) {
            out[a](i) = 0.0;  // fully covered face: zero flux
            continue;
          }
          double p0[3] = {fc[a][0], fc[a][1], fc[a][2]};
          p0[t1] += c1 / nOpen;
          p0[t2] += c2 / nOpen;  // open-area centroid on the face
          double p1[3] = {p0[0], p0[1], p0[2]};
          p1[a] += dir;  // one cell toward the solid
          const double s0 = ccSampleExt(sdf, e, p0[0], p0[1], p0[2]);
          const double s1 = ccSampleExt(sdf, e, p1[0], p1[1], p1[2]);
          double d = (s0 > 0.0 && s0 - s1 > 1e-12) ? s0 / (s0 - s1) : 0.0;
          out[a](i) = d < 0.0 ? 0.0 : (d > 2.0 ? 2.0 : d);  // cap at 2 cells: never leaves the
          // fitted data range [theta, theta+2], so the stencil weights stay interpolation-bounded
        }
      });
}

// centerToFace with the wall-aware reconstruction at solid-bordering faces (see above). The SDF is
// the cell-centered field (ghosts filled); U/V/W ghosts must be filled first, as for centerToFace.
// useCen (mode 3): evaluate at the stored open-centroid wall distance xcx/xcy/xcz instead of the
// face center (the flux-consistent quadrature).
inline void centerToFaceWallAware(CCField uf, CCField vf, CCField wf, CCConst U, CCConst V,
                                  CCConst W, CCConst sdf, CCConst xcx, CCConst xcy, CCConst xcz,
                                  bool useCen, C3 e, int g) {
  (void)g;
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::center_to_face_wall", MD(space, {1, 1, 1}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        uf(i) = wallAwareFaceValue(U, sdf, i, sx, x, e.x, useCen ? xcx(i) : -1.0);
        vf(i) = wallAwareFaceValue(V, sdf, i, sy, y, e.y, useCen ? xcy(i) : -1.0);
        wf(i) = wallAwareFaceValue(W, sdf, i, sz, z, e.z, useCen ? xcz(i) : -1.0);
      });
}

// The O-WEIGHTED EXACT ADJOINT of the wall-aware constraint, applied to a cell field p:
// out(i) = sum over the (up to 6) axis faces f whose stencil references cell i of
// c_i(f)·o(f)·(p(f) - p(f-sa)) — i.e. Cᵀp for the constraint C = D·diag(o)·T. For interior fluid
// cells (o = 1, ½/½ weights) this is exactly the central difference ½(p(i+sa) - p(i-sa)).
//
// WHY the o-weighting is essential (learned the hard way): the wall-anchored T deliberately has
// row sums != 1 at cut faces (the no-slip anchor pulls them down), so the UNWEIGHTED transpose
// force Tᵀ·G·P no longer telescopes to zero over a periodic box — the fluid feels a net spurious
// pressure force proportional to the near-wall gradients, which feeds back through the flow and
// DIVERGES (the mode-3 runaway; capping the abscissa does not help because the weights were never
// the problem). With the exact adjoint the pressure does NO work on the constraint manifold
// ((u, Cᵀp) = (Cu, p) = 0), cutting the feedback unconditionally, and o·T·Tᵀ stays
// interpolation-bounded so the per-step projection remains contractive. Used (setFaceInterp(2/3))
// for BOTH the incremental predictor's -grad(P) and the projection's cell correction: at steady
// state the predictor gradient IS the pressure force, so it must be the adjoint of the constraint —
// upgrading T alone (mode 1) measurably WORSENS the drag.
inline void transposeGradWallAware(CCField out, CCConst p, CCConst sdf, CCConst o, CCConst xc,
                                   bool useCen, int axis, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::transpose_grad_wall", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const long sa = (axis == 0) ? sx : (axis == 1) ? sy : sz;
        const int eA = (axis == 0) ? e.x : (axis == 1) ? e.y : e.z;
        const int ca = (axis == 0) ? x : (axis == 1) ? y : z;
        double acc = 0.0;
        long idx[3];
        double w[3];
        for (int m = -2; m <= 3; ++m) {  // all faces whose stencil can reference this cell
          const int cf = ca + m;
          if (cf < 1 || cf >= eA)
            continue;
          const long f = i + (long)m * sa;
          const int n = wallAwareFaceStencil(sdf, f, sa, cf, eA, useCen ? xc(f) : -1.0, idx, w);
          for (int k = 0; k < n; ++k)
            if (idx[k] == i)
              acc += w[k] * o(f) * (p(f) - p(f - sa));
        }
        out(i) = acc;
      });
}

// Cell fluid volume fraction cs (mode 4): 1 in clear fluid, 0 in solid, subsampled (4x4x4 trilinear
// SDF) at cut cells. Static, one-time at setSolid. Used to weight the FV momentum time term and the
// body/pressure forcing so a cut cell is driven over its FLUID volume only (omitting cs is an O(h)
// surface error — part of what keeps modes 0-3 first-order).
inline void buildCellFraction(CCField cs, CCConst sdf, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cell_fraction", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double sc = sdf(i);
        // fully clear one cell away on all axes -> not a cut cell; cheap exact classification
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const bool f0 = sc >= 0.0;
        const bool cut = (sdf(i + sx) >= 0.0) != f0 || (sdf(i - sx) >= 0.0) != f0 ||
                         (sdf(i + sy) >= 0.0) != f0 || (sdf(i - sy) >= 0.0) != f0 ||
                         (sdf(i + sz) >= 0.0) != f0 || (sdf(i - sz) >= 0.0) != f0;
        if (!cut) {
          cs(i) = sc >= 0.0 ? 1.0 : 0.0;
          return;
        }
        int nf = 0;
        for (int j0 = 0; j0 < 4; ++j0)
          for (int j1 = 0; j1 < 4; ++j1)
            for (int j2 = 0; j2 < 4; ++j2) {
              const double o0 = -0.375 + 0.25 * j0, o1 = -0.375 + 0.25 * j1,
                           o2 = -0.375 + 0.25 * j2;
              if (ccSampleExt(sdf, e, x + o0, y + o1, z + o2) > 0.0)
                ++nf;
            }
        cs(i) = nf / 64.0;
      });
}

// FINITE-VOLUME MOMENTUM OPERATOR (mode 4): applies L_FV(U) over the fluid control volume of each
// cell (the centroid wall-gradient is validated a priori in tests/study/fv_wallflux_apriori.py):
//   L_FV(U)_i = idt·cs_i·U_i + mu·[ Σ_f o_f·(U_i − U_nbr) + Σ_a W_a·g_a^centroid(U) ]
// The o_f-weighted two-point face fluxes + the fragment-centroid wall drag = the same finite-volume
// boundary geometry the projection uses. In the interior (cs=1, o_f=1, W=0) this is exactly the
// backward-Euler diffusion operator idt·U − mu·Lap(U) — identical to the IBM matrix M there, so the
// defect correction (M·u − L_FV·u) vanishes and interior cells stay byte-identical to mode 0.
inline void fvViscousApply(CCField Lu, CCConst U, CCConst sdf, CCConst cs, CCConst ox, CCConst oy,
                           CCConst oz, double mu, double idt, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::fv_viscous_apply", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long st[3] = {sx, sy, sz};
        const long i = (long)x + (long)y * sy + (long)z * sz;
        CCConst oa[3] = {ox, oy, oz};
        double W[3], of[3];  // fragment normal per axis; low/high face openness
        double diag = 0.0, offs = 0.0, aw = 0.0;
        for (int a = 0; a < 3; ++a) {
          const double om = oa[a](i), op = oa[a](i + st[a]);
          W[a] = om - op;
          of[a] = om + op;  // low + high face openness
          diag += om + op;
          offs += om * U(i - st[a]) + op * U(i + st[a]);
          aw += (W[a] < 0.0 ? -W[a] : W[a]);
        }
        double wall = 0.0;
        if (aw > 1e-12) {  // centroid-anchored wall drag mu·Σ_a W_a g_a (a-priori-validated:
                           // fv_wallflux_apriori.py)
          double nx = 0.5 * (sdf(i + sx) - sdf(i - sx));
          double ny = 0.5 * (sdf(i + sy) - sdf(i - sy));
          double nz = 0.5 * (sdf(i + sz) - sdf(i - sz));
          double nn = Kokkos::sqrt(nx * nx + ny * ny + nz * nz);
          if (nn > 1e-12) {
            nx /= nn;
            ny /= nn;
            nz /= nn;
            const double nv[3] = {nx, ny, nz};
            const double sdi = sdf(i);
            // foot point p* = x − sdi·n̂, clamped into the sampleable block [1, e−2] so the two
            // off-cell trilinear taps (p*+2σ) never index outside the padded field (guards a NaN/
            // large sdi too — ccSampleExt only clamps integer indices, not a NaN coordinate).
            auto cl = [](double v, double hi) { return v < 1.0 ? 1.0 : (v > hi ? hi : v); };
            const double px = cl(x - sdi * nx, e.x - 2.0);
            const double py = cl(y - sdi * ny, e.y - 2.0);
            const double pz = cl(z - sdi * nz, e.z - 2.0);
            for (int a = 0; a < 3; ++a) {
              const double sg = nv[a] >= 0.0 ? 1.0 : -1.0;
              const double u1 = ccSampleExt(U, e, px + (a == 0 ? sg : 0.0),
                                            py + (a == 1 ? sg : 0.0), pz + (a == 2 ? sg : 0.0));
              const double u2 =
                  ccSampleExt(U, e, px + (a == 0 ? 2.0 * sg : 0.0), py + (a == 1 ? 2.0 * sg : 0.0),
                              pz + (a == 2 ? 2.0 * sg : 0.0));
              wall += W[a] * sg * (2.0 * u1 - 0.5 * u2);
            }
          }
        }
        // FV viscous operator = idt·cs·U + μ·[Σo_f(U_i−U_nbr) − Σ_a W_a g_a].  The wall term sign
        // is −μ Σ W_a g_a: from ∫_CV −μ∇²u = −μ[flux_out − flux_in], the wall flux g_a enters with
        // a minus, so a resistive wall (∂u/∂n<0) ADDS to the operator (dissipative), matching the
        // interior −μLap. (+ would be anti-dissipative and blows the solve up.)
        Lu(i) = idt * cs(i) * U(i) + mu * (diag * U(i) - offs - wall);
        (void)of;
      });
}

// Lagrange quadratic through (-1,a1),(0,a2),(+1,a3) evaluated at xx (Basilisk embed.h `quadratic`).
KOKKOS_INLINE_FUNCTION double eQuad(double xx, double a1, double a2, double a3) {
  return (a1 * (xx - 1.0) + a3 * (xx + 1.0)) * xx * 0.5 - a2 * (xx - 1.0) * (xx + 1.0);
}

// TRUE-NORMAL wall gradient d(U)/dn at the embedded boundary of a cut cell — the Basilisk embed.h
// `dirichlet_gradient` (no-slip U_wall = 0), in cell units (h = 1). n̂ = unit inward-to-FLUID normal
// (∇sdf), p = boundary-point offset from the cell centre (cell units). Along the dominant-|n̂| axis
// it places two image points 1 and 2 cells into the fluid, interpolates U there by TRANSVERSE
// bi-quadratic interpolation of the cell-centred values (`eQuad`×`eQuad`), and fits the quadratic
// {0 at wall, v0 at d0, v1 at d1} for a 2nd-order derivative. Fallbacks (Basilisk): if the near
// image point's 3×3 transverse stencil straddles the wall, a biased-linear `embed_interpolate`
// anchored at the fluid home cell (uses only fluid cells); if even the home cell is solid, the
// degenerate 1-point estimate through the cell centre. A-priori-validated O(h²) with 0% degenerate
// on the Stokes sphere (tests/study/fv_wallflux_apriori.py, variant C). Interior/transverse reads
// of U and sdf must be halo-filled (the velocity block carries G=2 ghosts).
KOKKOS_INLINE_FUNCTION double embedDirichletGradient(CCConst U, CCConst sdf, C3 e, int x, int y,
                                                     int z, double nx, double ny, double nz,
                                                     double px, double py, double pz) {
  const double nv[3] = {nx, ny, nz}, pv[3] = {px, py, pz};
  const int ci[3] = {x, y, z}, ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  int da = 0;
  double best = Kokkos::fabs(nv[0]);
  if (Kokkos::fabs(nv[1]) > best) {
    best = Kokkos::fabs(nv[1]);
    da = 1;
  }
  if (Kokkos::fabs(nv[2]) > best)
    da = 2;
  const int t1 = (da + 1) % 3, t2 = (da + 2) % 3;
  const double sgn = nv[da] >= 0.0 ? 1.0 : -1.0;
  auto clampi = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  auto rd = [](double v) {  // round to nearest integer, clamped to {-1,0,1} (Basilisk j/k)
    double r = v >= 0.0 ? Kokkos::floor(v + 0.5) : Kokkos::ceil(v - 0.5);
    return r > 1.0 ? 1.0 : (r < -1.0 ? -1.0 : r);
  };
  double vL[2], dL[2];
  bool defd[2];
  for (int l = 0; l < 2; ++l) {
    const int io = (l + 1) * (int)sgn;  // ±1, ±2 cells into the fluid along the dominant axis
    const double dl = ((double)io - pv[da]) / nv[da];  // distance (cells) from wall to image plane
    const double y1 = pv[t1] + dl * nv[t1], z1 = pv[t2] + dl * nv[t2];
    const int jj = (int)rd(y1), kk = (int)rd(z1);
    const double ly = y1 - jj, lz = z1 - kk;
    double vv[3][3];
    bool sf[3][3];
    for (int dk = -1; dk <= 1; ++dk)
      for (int dj = -1; dj <= 1; ++dj) {
        int cc[3];
        cc[da] = ci[da] + io;
        cc[t1] = ci[t1] + jj + dj;
        cc[t2] = ci[t2] + kk + dk;
        const long idx = (long)clampi(cc[0], ext[0]) * st[0] + (long)clampi(cc[1], ext[1]) * st[1] +
                         (long)clampi(cc[2], ext[2]) * st[2];
        vv[dj + 1][dk + 1] = U(idx);
        sf[dj + 1][dk + 1] = sdf(idx) >= 0.0;
      }
    bool full = true;
    for (int a2 = 0; a2 < 3; ++a2)
      for (int b2 = 0; b2 < 3; ++b2)
        full = full && sf[a2][b2];
    const bool home = sf[1][1];
    const double vbq =
        eQuad(lz, eQuad(ly, vv[0][0], vv[1][0], vv[2][0]), eQuad(ly, vv[0][1], vv[1][1], vv[2][1]),
              eQuad(ly, vv[0][2], vv[1][2], vv[2][2]));
    double vbl = vv[1][1];  // biased-linear embed_interpolate (fluid-side only), anchored at home
    {
      const int fp = ly >= 0.0 ? 2 : 0, fm = ly >= 0.0 ? 0 : 2;
      if (sf[fp][1])
        vbl += Kokkos::fabs(ly) * (vv[fp][1] - vv[1][1]);
      else if (sf[fm][1])
        vbl += Kokkos::fabs(ly) * (vv[1][1] - vv[fm][1]);
    }
    {
      const int fp = lz >= 0.0 ? 2 : 0, fm = lz >= 0.0 ? 0 : 2;
      if (sf[1][fp])
        vbl += Kokkos::fabs(lz) * (vv[1][fp] - vv[1][1]);
      else if (sf[1][fm])
        vbl += Kokkos::fabs(lz) * (vv[1][1] - vv[1][fm]);
    }
    vL[l] = full ? vbq : vbl;
    dL[l] = dl;
    defd[l] = full || home;
  }
  if (defd[0] && defd[1]) {  // 2-point quadratic fit (3rd-order value -> 2nd-order gradient)
    const double d0 = dL[0], d1 = dL[1], v0 = vL[0], v1 = vL[1];
    return (v0 * d1 / d0 - v1 * d0 / d1) / (d1 - d0);
  }
  if (defd[0])  // near image point only -> 1-point linear
    return vL[0] / dL[0];
  double d0 = Kokkos::fabs(pv[da] / nv[da]);  // degenerate sliver: 1-point through the cell centre
  if (d0 < 1e-3)
    d0 = 1e-3;
  const long ii = (long)ci[0] * st[0] + (long)ci[1] * st[1] + (long)ci[2] * st[2];
  return U(ii) / d0;
}

// EMBED viscous operator (setFaceInterp(5)): identical to fvViscousApply except the wall drag is
// the TRUE-NORMAL Basilisk gradient (embedDirichletGradient) rather than the axis-by-axis
// fragment-normal estimate. Per cut cell the fragment area is |W| = |o_{a−}−o_{a+}|
// (divergence-theorem normal) and the wall flux is +μ·area·d(U)/dn — algebraically the −μ·(W·∇U) of
// fvViscousApply but with the O(h²) true-normal derivative in place of the O(h) axis
// reconstruction. Interior cells (area→0) reduce to idt·cs·U + μ(diag·U − offs) exactly, so the
// mode-0 defect vanishes there.
inline void embedViscousApply(CCField Lu, CCConst U, CCConst sdf, CCConst cs, CCConst ox,
                              CCConst oy, CCConst oz, double mu, double idt, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::embed_viscous_apply", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long st[3] = {sx, sy, sz};
        const long i = (long)x + (long)y * sy + (long)z * sz;
        CCConst oa[3] = {ox, oy, oz};
        double Wv[3];
        double diag = 0.0, offs = 0.0;
        for (int a = 0; a < 3; ++a) {
          const double om = oa[a](i), op = oa[a](i + st[a]);
          Wv[a] = om - op;
          diag += om + op;
          offs += om * U(i - st[a]) + op * U(i + st[a]);
        }
        const double area = Kokkos::sqrt(Wv[0] * Wv[0] + Wv[1] * Wv[1] + Wv[2] * Wv[2]);
        double wall = 0.0;
        if (area > 1e-12) {
          double nx = 0.5 * (sdf(i + sx) - sdf(i - sx));
          double ny = 0.5 * (sdf(i + sy) - sdf(i - sy));
          double nz = 0.5 * (sdf(i + sz) - sdf(i - sz));
          const double nn = Kokkos::sqrt(nx * nx + ny * ny + nz * nz);
          if (nn > 1e-12) {
            nx /= nn;
            ny /= nn;
            nz /= nn;
            const double sdi = sdf(i);  // foot-point (boundary centroid proxy) offset, cell units
            const double dudn = embedDirichletGradient(U, sdf, e, x, y, z, nx, ny, nz, -sdi * nx,
                                                       -sdi * ny, -sdi * nz);
            wall = area * dudn;  // +μ·area·dudn == −μ·(W·∇U) with the true-normal derivative
          }
        }
        Lu(i) = idt * cs(i) * U(i) + mu * (diag * U(i) - offs + wall);
      });
}

// 7-point stencil matvec y = M·u using the stored (rs-scaled) IBM operator coefficients. Used to
// form the mode-4 defect-correction RHS  b = M·u^k − rs·L_FV(u^k) + rs·b_FV, whose fixed point
// satisfies L_FV·u* = b_FV exactly, with M only the (stable, small-cell-safe) preconditioner.
inline void stencilMatvec(CCField y, CCConst u, MConst AC, MConst AW, MConst AE, MConst AS,
                          MConst AN, MConst AB, MConst AT, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::stencil_matvec", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y2, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y2 * sy + (long)z * sz;
        y(i) = (double)AC(i) * u(i) + (double)AW(i) * u(i - sx) + (double)AE(i) * u(i + sx) +
               (double)AS(i) * u(i - sy) + (double)AN(i) * u(i + sy) + (double)AB(i) * u(i - sz) +
               (double)AT(i) * u(i + sz);
      });
}

// u -= d over the inner cells (the mode-2 correction applies the transposeGradWallAware field).
inline void subtractField(CCField u, CCConst d, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::subtract_field", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        u(i) -= d(i);
      });
}

// OPENNESS-WEIGHTED cell pressure gradient along one axis (Basilisk centered_gradient / embed cell
// correction): out(i) = (o(i)·(p(i)−p(i−sa)) + o(i+sa)·(p(i+sa)−p(i))) / (o(i)+o(i+sa)). In the
// interior (o=1) this is the central difference ½(p(i+sa)−p(i−sa)); at a CUT cell with one closed
// face (o=0) it returns the OPEN face gradient at FULL weight (not the plain projectCorrectCenter's
// ½), which is the flux-consistent pressure force the approximate projection needs at the embedded
// boundary — the O(h) under-correction the plain map makes there (analysis defect (b)). Used for
// both the incremental −grad(P^n) predictor and the projection correction of the EMBED path (mode
// 6), so the two stay the openness-adjoint of the fs-weighted divergence constraint.
inline void centerGradOpen(CCField out, CCConst p, CCConst o, int axis, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::center_grad_open", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const long sa = (axis == 0) ? sx : (axis == 1) ? sy : sz;
        const double om = o(i), op = o(i + sa);
        out(i) = (om * (p(i) - p(i - sa)) + op * (p(i + sa) - p(i))) / (om + op + 1e-12);
      });
}

// u,v,w -= openness-weighted cell pressure gradient of phi (centerGradOpen per axis). The embed
// analogue of projectCorrectCenter: the cut-cell cell velocity gets the full open-face pressure
// force.
inline void projectCorrectCenterOpen(CCField u, CCField v, CCField w, CCConst phi, CCConst ox,
                                     CCConst oy, CCConst oz, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::correct_center_open", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double omx = ox(i), opx = ox(i + sx);
        const double omy = oy(i), opy = oy(i + sy);
        const double omz = oz(i), opz = oz(i + sz);
        u(i) -= (omx * (phi(i) - phi(i - sx)) + opx * (phi(i + sx) - phi(i))) / (omx + opx + 1e-12);
        v(i) -= (omy * (phi(i) - phi(i - sy)) + opy * (phi(i + sy) - phi(i))) / (omy + opy + 1e-12);
        w(i) -= (omz * (phi(i) - phi(i - sz)) + opz * (phi(i + sz) - phi(i))) / (omz + opz + 1e-12);
      });
}

// Cell-centered velocity correction u_c -= grad_c(phi), grad_c per axis = ½·(g⁻ + g⁺) of the two
// adjacent FACE pressure-gradients, but a face that is fully CLOSED (openness 0 — a solid
// neighbour) contributes a ZERO gradient instead of reading that neighbour's φ. Rationale:
//   * interior fluid cell (both faces open): ½(φᵢ₊₁-φᵢ₋₁) — the central difference, bulk unchanged;
//   * immersed cut cell (solid neighbour): the closed face's φ is DECOUPLED (≈0, AC≈0 in the
//   operator), so
//     reading it corrupts the gradient — zeroing that face uses only the fluid-side (open)
//     gradient;
//   * domain-BC wall (Neumann, φ-ghost = interior): the closed wall face truly has ∂φ/∂n≈0, and
//   zeroing it
//     gives ½·g_open — identical to the previous central difference with the Neumann ghost (no
//     change there).
// Cut faces (0<o<1) keep their real gradient (the neighbour is fluid). Only the cell field is
// touched; the projection's face divergence-free guarantee is unaffected. phi ghosts + face
// openness must be filled first.
inline void projectCorrectCenter(CCField u, CCField v, CCField w, CCConst phi, CCConst ox,
                                 CCConst oy, CCConst oz, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::correct_center", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double gm_x = (ox(i) > 1e-12) ? (phi(i) - phi(i - sx)) : 0.0;
        const double gp_x = (ox(i + sx) > 1e-12) ? (phi(i + sx) - phi(i)) : 0.0;
        const double gm_y = (oy(i) > 1e-12) ? (phi(i) - phi(i - sy)) : 0.0;
        const double gp_y = (oy(i + sy) > 1e-12) ? (phi(i + sy) - phi(i)) : 0.0;
        const double gm_z = (oz(i) > 1e-12) ? (phi(i) - phi(i - sz)) : 0.0;
        const double gp_z = (oz(i + sz) > 1e-12) ? (phi(i + sz) - phi(i)) : 0.0;
        u(i) -= 0.5 * (gm_x + gp_x);
        v(i) -= 0.5 * (gm_y + gp_y);
        w(i) -= 0.5 * (gm_z + gp_z);
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_APPROX_PROJECTION_HPP
