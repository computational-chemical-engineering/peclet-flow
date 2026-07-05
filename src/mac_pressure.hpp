/// @file
/// @brief flow — portable (Kokkos) cut-cell pressure operator + Chorin projection.
///
/// Kokkos port of the variable-coefficient pressure machinery (mg_build_op_k / mg_smooth_var_k from
/// mac_multigrid.cuh, diverg_open_k / correct_k from distributed_ns.cuh): the open-face-weighted
/// Poisson operator A = -div(open grad) built from the staggered face openness, its red-black
/// Gauss-Seidel smoother (solid cells AC~0 decoupled), the open-weighted flux divergence, and the
/// staggered gradient correction. gf = 1/h^2 per axis (1 in grid units). Runs on any Kokkos
/// backend.
#ifndef PECLET_FLOW_MAC_PRESSURE_HPP
#define PECLET_FLOW_MAC_PRESSURE_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

// A = -div(open grad): AC = sum of the 6 face terms (openness*gf), off-diagonal across each face =
// -term. ox[i] is the -x face openness of cell i (== +x face of cell i-1). (mg_build_op_k port.)
// OpV is the operator-coefficient view type (float `mreal` to match CUDA, or double).
template <class OpV>
inline void buildCutcellOp(OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB, OpV AT, CCConst ox,
                           CCConst oy, CCConst oz, C3 e, int g, double gfx, double gfy,
                           double gfz) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cc_build_op", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double tw = ox(i) * gfx, te = ox(i + sx) * gfx;
        const double ts = oy(i) * gfy, tn = oy(i + sy) * gfy;
        const double tb = oz(i) * gfz, tt = oz(i + sz) * gfz;
        AW(i) = -tw;
        AE(i) = -te;
        AS(i) = -ts;
        AN(i) = -tn;
        AB(i) = -tb;
        AT(i) = -tt;
        AC(i) = te + tw + tn + ts + tt + tb;
      });
}

// Open-weighted flux divergence d_i = sum_f signed(o_f * face-velocity), consistent with A
// (diverg_open_k).
inline void divergOpen(CCConst u, CCConst v, CCConst w, CCConst ox, CCConst oy, CCConst oz,
                       CCField d, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::diverg_open", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        d(i) = (ox(i + sx) * u(i + sx) - ox(i) * u(i)) + (oy(i + sy) * v(i + sy) - oy(i) * v(i)) +
               (oz(i + sz) * w(i + sz) - oz(i) * w(i));
      });
}

// One red/black sweep of the variable operator: phi=(b - offdiag)/AC; AC~0 (fully solid) cells
// decoupled. b carries the negated divergence so the system is A phi = -div(u*) (matches the
// validated const-coeff sign).
template <class OpV>
inline void cutcellSmoothColor(CCField phi, CCConst b, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN,
                               OpV AB, OpV AT, C3 e, C3 og, int g, int color) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double ac = AC(i);
        if (ac < 1e-30)
          return;  // fully closed (solid) cell: decoupled, phi stays 0
        const double s = AE(i) * phi(i + sx) + AW(i) * phi(i - sx) + AN(i) * phi(i + sy) +
                         AS(i) * phi(i - sy) + AT(i) * phi(i + sz) + AB(i) * phi(i - sz);
        phi(i) = (b(i) - s) / ac;
      });
}

// y = A x for the cut-cell operator over inner cells (matvec for PCG; mg_apply_var_k port).
template <class OpV>
inline void applyCutcellOp(CCField y, CCConst x, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB,
                           OpV AT, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cc_apply", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        y(i) = AC(i) * x(i) + AE(i) * x(i + sx) + AW(i) * x(i - sx) + AN(i) * x(i + sy) +
               AS(i) * x(i - sy) + AT(i) * x(i + sz) + AB(i) * x(i - sz);
      });
}

// Projection correction u -= grad(phi) on the staggered faces (correct_k port). No openness here —
// the openness lives in the operator + divergence; closed faces carry phi~0 on both sides so stay
// unchanged.
inline void projectCorrect(CCField u, CCField v, CCField w, CCConst phi, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::correct", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= phi(i) - phi(i - sx);
        v(i) -= phi(i) - phi(i - sy);
        w(i) -= phi(i) - phi(i - sz);
      });
}

// Variable-density projection correction (sibling of projectCorrect): u_f -= (rho0/rho_f) grad(phi)
// with rho_f the arithmetic face mean — the SAME face density that scaled the Poisson coefficient
// c_f = open_f*rho0/rho_f, so the corrected open flux telescopes to A*phi exactly (discrete
// consistency; constant rho == rho0 reduces to projectCorrect identically, ratio 1.0 exact in FP).
inline void projectCorrectVar(CCField u, CCField v, CCField w, CCConst phi, CCConst rho, double rho0,
                              C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::correct_var", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        u(i) -= rho0 / (0.5 * (rho(i) + rho(i - sx))) * (phi(i) - phi(i - sx));
        v(i) -= rho0 / (0.5 * (rho(i) + rho(i - sy))) * (phi(i) - phi(i - sy));
        w(i) -= rho0 / (0.5 * (rho(i) + rho(i - sz))) * (phi(i) - phi(i - sz));
      });
}

// Variable-density Poisson face coefficients on the MG (g=1) block: c_f = open_f * rho0 / rho_f,
// rho_f = arithmetic face mean. Computed over the inner cells only — CutcellMG::setOpenness runs
// its own periodic/halo ghost fill + non-periodic boundary re-imposition on whatever level-0 fields
// it receives, exactly as for the raw openness (the coefficient "rides the openness rails"). The
// rho ghost ring of the g=1 block must be valid (bridged from the filled G=2 field).
inline void buildRhoCoeff(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy, CCConst oz,
                          CCConst rho, double rho0, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::rho_coeff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        cx(i) = ox(i) * rho0 / (0.5 * (rho(i) + rho(i - sx)));
        cy(i) = oy(i) * rho0 / (0.5 * (rho(i) + rho(i - sy)));
        cz(i) = oz(i) * rho0 / (0.5 * (rho(i) + rho(i - sz)));
      });
}

// --- Volume-averaged (porous) continuity for unresolved CFD-DEM -----------------------------------
// The proper continuity is d(eps)/dt + div(eps u) = 0 (eps = void fraction from the particles), so the
// velocity is NOT solenoidal: div(eps u) = -d(eps)/dt. These size the projection to that constraint.

// eps-weighted open-face divergence: d = div(open * eps_f * u), eps_f = arithmetic face mean. Reduces
// to divergOpen when eps == 1 everywhere (no particles).
inline void divergOpenEps(CCConst u, CCConst v, CCConst w, CCConst ox, CCConst oy, CCConst oz,
                          CCConst eps, CCField d, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::diverg_open_eps", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double exp = 0.5 * (eps(i) + eps(i + sx)), exm = 0.5 * (eps(i) + eps(i - sx));
        const double eyp = 0.5 * (eps(i) + eps(i + sy)), eym = 0.5 * (eps(i) + eps(i - sy));
        const double ezp = 0.5 * (eps(i) + eps(i + sz)), ezm = 0.5 * (eps(i) + eps(i - sz));
        d(i) = (ox(i + sx) * exp * u(i + sx) - ox(i) * exm * u(i)) +
               (oy(i + sy) * eyp * v(i + sy) - oy(i) * eym * v(i)) +
               (oz(i + sz) * ezp * w(i + sz) - oz(i) * ezm * w(i));
      });
}

// Porous Poisson face coefficient c_f = open_f * eps_f (eps_f = arithmetic face mean). Constant-density
// gas: the correction stays u -= grad(phi) (projectCorrect) so the open*eps flux telescopes to A*phi.
// (Combining with variable rho — c_f *= rho0/rho_f, projectCorrectVar — is a later composition.)
inline void buildPorousCoeff(CCField cx, CCField cy, CCField cz, CCConst ox, CCConst oy, CCConst oz,
                             CCConst eps, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::porous_coeff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        cx(i) = ox(i) * 0.5 * (eps(i) + eps(i - sx));
        cy(i) = oy(i) * 0.5 * (eps(i) + eps(i - sy));
        cz(i) = oz(i) * 0.5 * (eps(i) + eps(i - sz));
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_PRESSURE_HPP
