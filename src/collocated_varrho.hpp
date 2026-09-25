/// @file
/// @brief flow — collocated (ABC approximate-projection) variable-density kernels: the
///        mass-adjoint pair of rung V8 (doc/collocated_varrho_forces.md).
///
/// The collocated solver stores u at the cell centre and couples it to the pressure through an
/// APPROXIMATE projection: map the cell velocities onto a MAC face field, make THAT field
/// discretely divergence-free, then correct the cell field. With a variable density (or the CSF of
/// surface tension, rung V8 = `Grid::collocated && (varRho_ || csfActive())`) the step is the
/// incremental-rotational predictor of the constant-density scheme with the weights changed so
/// that the pressure force and the constraint stay an ADJOINT pair in the kinetic-energy inner
/// product (§4.3):
///
///  * the pressure, the CSF and the uniform drive `set_body_force` enter the implicit momentum
///    predictor at the cell as the finite-volume face integral with the acceleration-continuous
///    face pressure,  rho_c * ½ sum_faces o_f (F_f - w (P(j) - P(j-s))) / rho_f
///    (`buildFaceAccelVar` + `addFaceAccelCsf` build the bracket / rho_f per face; the cell RHS
///    `Solver::buildRhsColoVar` reconstructs it with `varrho::cellFromFaces`);
///  * a per-cell volumetric force enters at the CELL VALUE times the weight sum
///    W = ½ (o_lo + o_hi) of that reconstruction (`varrho::weightSum`), which is what balances
///    f = rho g against the wall's Neumann pressure (§4.5);
///  * the projection's face field is MOMENTUM-weighted, (rho_L u_L + rho_R u_R)/(rho_L + rho_R)
///    (`centerToFaceMassWeighted`), so the pressure force is exactly -M^{-1} C^T for the constraint
///    C = D O Pi_rho, for any openness;
///  * the Poisson operator keeps the face coefficient c_f = o_f rho0/rho_f with the ARITHMETIC face
///    mean (the staggered `buildRhoCoeff`), the face correction is `projectCorrectVar`, and the
///    cell correction is the SAME face-to-cell reconstruction applied to the face corrections
///    k(j) = w (rho0/rho_f) (phi(j) - phi(j-s)) (`correctCellFaceAverageVar`).
///
/// At uniform density every one of these reduces to the validated constant-density collocated
/// scheme (to round-off). The rotational (Timmermans) pressure update is unchanged. Openness is a
/// multiplicative weight applied ONCE, in the face-to-cell reconstruction and in the divergence
/// (§4.9 L1); the per-face arrays built here carry no openness.
///
/// HISTORY — do not reintroduce. WO-T (2026-09-02) made every force and the lagged pressure a MAC
/// face acceleration added after the viscous solve (Basilisk centered.h; Popinet JCP 2009 §3). It
/// was balanced but non-incremental and unstable above mu dt/(rho h^2) = 1/12, and it was retired
/// 2026-09-25. See doc/collocated_varrho_forces.md §2 and the suite-wide register.
///
/// Every kernel here is reached only under `Grid::collocated && (varRho_ || csfActive())`: nothing
/// on the staggered path and nothing on the constant-density collocated path calls them.
#ifndef PECLET_FLOW_COLLOCATED_VARRHO_HPP
#define PECLET_FLOW_COLLOCATED_VARRHO_HPP

#include <Kokkos_Core.hpp>

#include "collocated_varrho_point.hpp"
#include "mac_cutcell.hpp"
#include "policy.hpp"
#include "vof/surface_tension.hpp"

namespace peclet::flow {

// The face acceleration of ONE component, WITHOUT the openness (the cell reconstruction applies
// it):
//
//   af(i) = scale * ( fc + [½(fb(i)+fb(i-s))] - [w_a (P(i) - P(i-s))] ) / rho_f(i)
//
// `s` is the component's own face stride (the face at index `i` separates cells `i-s` and `i`, the
// solver's low-face convention). `rho_f` = arithmetic face mean when `haveRho`, the scalar `rhoC`
// otherwise. The V8 predictor calls it with `haveFb = false` (per-cell forces are volumetric: the
// cell value times the weight sum) and `scale = 1`.
//
// RANGE: the face range `[g, e-g]` on every axis. The cell reconstruction of cell `i` reads the
// face at `i+s`, so the high plane is needed; every input it reads there is a depth-1 ghost, which
// every property fill has already written, and because the face at index `e-g` is formed from
// exactly the same two ghost values the neighbouring rank forms its own index-`g` face from, in the
// same order, the plane is bitwise decomposition-independent. PHASE 2 (anisotropic cells,
// doc/anisotropic_metric.md §1.2/§3): of the three terms only the PRESSURE difference carries a
// metric. `fc` is the body force already converted per axis by Phase 1's `forceToInt(a)`, `fb` is a
// per-cell force in the same per-axis normalisation, and `1/rho_f` is unit-free -- while `-grad_a
// P'` carries `w_a = 1/h_a'^2`, exactly as it does in the staggered `buildRhs*` predictor and in
// `projectCorrect`. `wa == 1.0` isotropic (exact).
inline void buildFaceAccelVar(CCField af, CCConst P, CCConst rho, CCConst fb, bool haveFb,
                              bool haveRho, double rhoC, double fc, bool incr, double scale, long s,
                              C3 e, int g, double wa = 1.0) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_accel",
      MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double rf = haveRho ? 0.5 * (rho(i) + rho(i - s)) : rhoC;
        const double f = fc + (haveFb ? 0.5 * (fb(i) + fb(i - s)) : 0.0) -
                         (incr ? wa * (P(i) - P((long)i - s)) : 0.0);
        af(i) = scale * f / rf;
      });
}

// ADDITIVE balanced-force CSF at the same face, with the same `1/rho_f` and the same `scale`:
//
//   af(i) += scale * sigma * kappa_f * (C(i) - C(i-s)) / hGrad / rho_f(i)
//
// `hGrad` is the axis's PRESSURE-GRADIENT WEIGHT denominator `h_a'^2` (Phase 3, V3.1 — the same
// symbol the pressure face difference carries); 1.0 on every isotropic run.
//
// `kappa_f` is the V4 pairing (`vof::csfFaceCurvature`): the mean of the two cells' curvatures
// where both carry one, the single available one where only one does. The force is the
// projection's OWN difference operator applied to `sigma*kappa*C`, so for a constant kappa it lies
// exactly in the range of the pressure gradient and is balanced by a pressure (exactly from step 1
// under the balanced-force projection, §4.6).
inline void addFaceAccelCsf(CCField af, CCConst cv, CCConst kp, CCConst kb, CCConst rho,
                            bool haveRho, double rhoC, double sigma, double hGrad, double scale,
                            long s, C3 e, int g) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::colo_face_csf", MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double dC = cv(i) - cv((long)i - s);
        if (dC == 0.0)
          return;  // no interface across this face -> no force, and no orphan either
        double kf = 0.0;
        vof::csfFaceCurvature(kp((long)i - s), kb((long)i - s), kp(i), kb(i), kf);
        const double rf = haveRho ? 0.5 * (rho(i) + rho(i - s)) : rhoC;
        af(i) += scale * vof::csfFaceForce(sigma, kf, dC, hGrad) / rf;
      });
}

// The projection's constraint face field Pi_rho u: the MOMENTUM-weighted centre-to-face map
//   uf(i) = (rho(i-s) U(i-s) + rho(i) U(i)) / (rho(i-s) + rho(i))
// on the face range `[g, e-g]` of every axis (it reads the density's depth-1 ghosts and the cell
// velocity ghosts, which the caller fills first). Faces outside that range are not written: the
// divergence reads only these, and the face ghosts are re-filled after the correction.
inline void centerToFaceMassWeighted(CCField uf, CCField vf, CCField wf, CCConst U, CCConst V,
                                     CCConst W, CCConst rho, C3 e, int g) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::center_to_face_mass",
      MD(space, {g, g, g}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        uf(i) = varrho::faceMassWeighted(U(i - sx), U(i), rho(i - sx), rho(i));
        vf(i) = varrho::faceMassWeighted(V(i - sy), V(i), rho(i - sy), rho(i));
        wf(i) = varrho::faceMassWeighted(W(i - sz), W(i), rho(i - sz), rho(i));
      });
}

// The projection's CELL correction for one component: the face-to-cell reconstruction of the face
// corrections,
//   u(i) -= cellFromFaces(k(i), k(i+s), o(i), o(i+s)),   k(j) = w_a (rho0/rho_f(j)) (phi(j) -
//   phi(j-s))
// with `k` written EXACTLY as `projectCorrectVar` writes the face correction (same grouping, same
// order), so the cell is corrected by the transpose of what the constraint's faces received; with
// `haveRho` false it is the plain `projectCorrect` difference. Inner cells.
inline void correctCellFaceAverageVar(CCField u, CCConst phi, CCConst rho, CCConst o, bool haveRho,
                                      double rho0, long s, C3 e, int g, double wa = 1.0) {
  CCExec space;
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::colo_cell_corr_var", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const long j = i + s;
        const double kLo = haveRho
                               ? wa * (rho0 / (0.5 * (rho(i) + rho(i - s))) * (phi(i) - phi(i - s)))
                               : wa * (phi(i) - phi(i - s));
        const double kHi = haveRho
                               ? wa * (rho0 / (0.5 * (rho(j) + rho(j - s))) * (phi(j) - phi(j - s)))
                               : wa * (phi(j) - phi(j - s));
        u(i) -= varrho::cellFromFaces(kLo, kHi, o(i), o(j));
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_COLLOCATED_VARRHO_HPP
