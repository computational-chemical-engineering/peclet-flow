/// @file
/// @brief flow — collocated (ABC approximate-projection) variable-density + face-acceleration
///        kernels.  Rung V8 (WO-T).
///
/// The collocated solver stores u at the cell centre and couples it to the pressure through an
/// APPROXIMATE projection: average the cell velocities onto a MAC face field (`centerToFace`),
/// make THAT field discretely divergence-free, then correct the cell field.  Two things follow for
/// two-phase flow, and they are the whole of this header:
///
///  1. **Variable density.**  The face coefficient of the Poisson operator is
///     `c_f = o_f rho0/rho_f` with the ARITHMETIC face mean `rho_f = ½(rho(i)+rho(i-s))` (the
///     staggered `buildRhoCoeff`, unchanged), so the face correction must be
///     `uf -= (rho0/rho_f) (phi(i) - phi(i-s))` — `projectCorrectVar`, which already exists and is
///     already exact-adjoint on faces.  The CELL correction is then NOT a cell-centred
///     `grad(phi)/rho_c`: it is the AVERAGE OF THE TWO FACE CORRECTIONS of each axis, i.e. the same
///     averaging operator `projectCorrectCenter` applies to the plain phi differences, applied to
///     the rho-weighted ones.  Anything else and the cell sees a balance its own faces do not.
///
///  2. **Forces are face accelerations.**  On the collocated grid every body / interfacial force
///     enters the predictor at the CELL through a central difference.  With rho jumping across an
///     interface the cell balance `g_c - grad_c(P)/rho_c` is O(1) wrong at interface cells even
///     when every FACE is exactly balanced (this is the collocated form of the three-way `rho_f`
///     consistency of `doc/variable_density_projection.md` §1/§3).  So: predict `u*` WITHOUT the
///     pressure gradient and without any body force, then add the face acceleration
///
///         a_f = dt * ( f_f - (P(i) - P(i-s)) ) / rho_f ,
///         f_f = f_const + ½(fb(i)+fb(i-s)) + sigma*kappa_f*(C(i)-C(i-s))/h
///
///     to `uf*` AFTER `centerToFace(u*)` (Basilisk's `centered.h` pattern; Popinet JCP 2009 §3),
///     and give the CELL the average of the two faces' TOTAL increment `a_f - (rho0/rho_f)
///     grad_f(phi)`. A hydrostatic column and a stationary droplet are then exactly balanced on the
///     faces, and the cell sees the average of an exact zero.
///
/// A face whose openness is 0 (an immersed or domain wall) contributes ZERO to the cell average —
/// the identical rule `projectCorrectCenter` uses for phi, and the reason the near-wall cell of a
/// hydrostatic column stays at rest rather than feeling an unbalanced half-force.
///
/// Every kernel here is a SIBLING: nothing on the staggered path and nothing on the
/// constant-density collocated path reaches them (they are called only under
/// `Grid::collocated && (varRho_ || csfActive())`, both of which used to throw).
#ifndef PECLET_FLOW_COLLOCATED_VARRHO_HPP
#define PECLET_FLOW_COLLOCATED_VARRHO_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"
#include "vof/surface_tension.hpp"

namespace peclet::flow {

// The face acceleration increment of ONE component, in velocity units (already multiplied by dt):
//
//   af(i) = dt * ( fc + [½(fb(i)+fb(i-s))] - [P(i) - P(i-s)] ) / rho_f(i)
//
// `s` is the component's own face stride (the face at index `i` separates cells `i-s` and `i`, the
// solver's low-face convention).  `rho_f` = arithmetic face mean when `haveRho`, the scalar `rhoC`
// otherwise.
//
// RANGE: the inner region WIDENED BY ONE on the high side of every axis, `[g, e-g]`.  Two consumers
// need that extra plane and both need it for the same reason the face field itself does: the
// divergence of cell `i` reads the face at `i+s`, and so does the cell average below.  Every input
// it reads there is a depth-1 ghost, which every property fill has already written; and because the
// face at index `e-g` is formed from exactly the same two ghost values the neighbouring rank forms
// its own index-`g` face from, in the same order, the plane is bitwise decomposition-independent.
inline void buildFaceAccelVar(CCField af, CCConst P, CCConst rho, CCConst fb, bool haveFb,
                              CCConst o, bool haveRho, double rhoC, double fc, bool incr, double dt,
                              long s, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_accel",
      MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (o(i) <= 1e-12) {  // a CLOSED face (a wall) does not move: no acceleration on it, and
          af(i) = 0.0;        // none in the cell average either (projectCorrectCenter's rule)
          return;
        }
        const double rf = haveRho ? 0.5 * (rho(i) + rho(i - s)) : rhoC;
        const double f = fc + (haveFb ? 0.5 * (fb(i) + fb(i - s)) : 0.0) -
                         (incr ? (P(i) - P((long)i - s)) : 0.0);
        af(i) = dt * f / rf;
      });
}

// ADDITIVE balanced-force CSF at the same face, with the same `1/rho_f` and the same `dt`:
//
//   af(i) += dt * sigma * kappa_f * (C(i) - C(i-s)) / h / rho_f(i)
//
// `kappa_f` is the V4 pairing (`vof::csfFaceCurvature`): the mean of the two cells' curvatures
// where both carry one, the single available one where only one does.  The force is the
// projection's OWN difference operator applied to `sigma*kappa*C`, so for a constant kappa it lies
// exactly in the range of the operator the projection inverts and the projection annihilates it —
// the V4 rule, verbatim, moved from the staggered momentum RHS to the collocated face field.
inline void addFaceAccelCsf(CCField af, CCConst cv, CCConst kp, CCConst kb, CCConst rho, CCConst o,
                            bool haveRho, double rhoC, double sigma, double h, double dt, long s,
                            C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_csf", MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (o(i) <= 1e-12)
          return;  // closed face: pinned to 0 by buildFaceAccelVar
        const double dC = cv(i) - cv((long)i - s);
        if (dC == 0.0)
          return;  // no interface across this face -> no force, and no orphan either
        double kf = 0.0;
        vof::csfFaceCurvature(kp((long)i - s), kb((long)i - s), kp(i), kb(i), kf);
        const double rf = haveRho ? 0.5 * (rho(i) + rho(i - s)) : rhoC;
        af(i) += dt * vof::csfFaceForce(sigma, kf, dC, h) / rf;
      });
}

// uf += af over the same range the increment was built on.
inline void addFaceIncrement(CCField uf, CCConst af, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_add", MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        uf(i) += af(i);
      });
}

// Turn the face acceleration into the TOTAL face velocity increment of the step by subtracting the
// projection's own face correction:  af(i) -= (rho0/rho_f) (phi(i) - phi(i-s)).
//
// The expression is written EXACTLY as `projectCorrectVar` writes it (same grouping, same order),
// so the number the cell averages is bit-for-bit the number the face received.  With `haveRho`
// false it is the plain `projectCorrect` difference.
inline void faceAccelSubGradPhi(CCField af, CCConst phi, CCConst rho, CCConst o, bool haveRho,
                                double rho0, long s, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_subgrad",
      MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (o(i) <= 1e-12)
          return;  // closed face: the total increment stays 0 (the wall holds the balance)
        if (haveRho)
          af(i) -= rho0 / (0.5 * (rho(i) + rho(i - s))) * (phi(i) - phi((long)i - s));
        else
          af(i) -= phi(i) - phi((long)i - s);
      });
}

// The cell counterpart: u(i) += ½( af(i) + af(i+s) ), a face with openness 0 contributing 0.
// This IS `projectCorrectCenter`'s averaging operator (same predicate, same ½, same closed-face
// rule) applied to the face increments instead of to the raw phi differences.
inline void applyCellFaceAverage(CCField u, CCConst af, CCConst o, long s, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::colo_cell_avg", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double am = (o(i) > 1e-12) ? af(i) : 0.0;
        const double ap = (o((long)i + s) > 1e-12) ? af((long)i + s) : 0.0;
        u(i) += 0.5 * (am + ap);
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_COLLOCATED_VARRHO_HPP
