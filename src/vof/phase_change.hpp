/// @file
/// @brief flow — VoF phase change (Part II, rungs P0/P1): the container-free kernel set.
///
/// **Container-free by contract**, exactly like `plic.hpp`: scalars and small local arrays only —
/// no `Kokkos::View`, no grid indexing, no halo types. The solver-side driver (the block walk, the
/// fields, the MPI ghosts) lives in `flow_ibm.hpp`; everything that is *geometry or physics* is
/// here so the block container (Part III) and the AMR path can share one copy.
///
/// ## What this rung implements (VOF_PLAN §9, the Boyd & Ling 2023 / Malan et al. 2021 pattern)
///
/// 1. **Mass flux** `mdot` (kg m^-2 s^-1, solver units) on interfacial cells, either prescribed
///    (P0) or from the two one-sided pure-cell temperature gradients (P1):
///
///        mdot = ( k_g grad(T_g).n  -  k_l grad(T_l).n ) / h_lv ,   n = m/|m|_2, LIQUID -> GAS
///
///    **The sign convention matters and the work-order text has it the other way round.** The
///    interfacial energy balance is `mdot h_lv = (q_l - q_g).n` with `q = -k grad T` and `n`
///    pointing *out of the liquid*, which gives the form above. Check it on the Stefan problem:
///    superheated vapour at `x < x_G`, saturated liquid at `x > x_G`, so `n = -x_hat`,
///    `grad(T_g).n = -dT_g/dx > 0`, `grad(T_l) = 0` and `mdot = k_g |dT_g/dx| / h_lv > 0` —
///    evaporation, as it must be. Writing `(k_l grad(T_l) - k_g grad(T_g)).n` with the SAME `n`
///    (the PLIC normal, which points into the gas) would condense a superheated vapour.
///
/// 2. **Interface regression by PLIC plane shift** (never a volume source in the C equation — the
///    Hardt–Wondra smeared source leaves unresolvable residue and breaks the Weymouth–Yue bounds):
///    the liquid volume to remove from an interfacial cell is
///
///        dV = mdot * A_G * dt / rho_l ,      A_G = the PLIC polygon area of the cell
///
///    and the new colour is `C' = C - dV` — which IS the plane shift, because
///    `plicVolume(m, plicAlpha(m, C - dV)) == C - dV` identically. `plicArea` below supplies
///    `A_G` analytically (no finite difference of `plicVolume` in alpha: that is exact only where
///    `V(alpha)` is locally linear, i.e. NOT in the two cubic branches).
///
/// 3. **Clip and redistribute.** If `C - dV < 0` the cell is emptied and the deficit is pushed
///    into the face neighbours on the LIQUID side (along `-n`), weighted by `n_d^2`; the mirror
///    rule handles `C - dV > 1` (condensation) along `+n`. Conservative to round-off.
///
/// 4. **Divergence source** `S = mdot A_G (1/rho_g - 1/rho_l) / V_cell` deposited into the nearest
///    PURE GAS cell along `+n`, so the interfacial cell's face velocities stay the LIQUID velocity
///    and Weymouth–Yue advects the colour with a field it is entitled to.
///
/// ## Units
///
/// Cell size h = 1 (the solver's cell units), so `A_G` is a fraction of a cell face, `V_cell = 1`,
/// `mdot/rho` is a velocity in cells/s and `S` is a 1/s divergence. Densities are in
/// kg/cell^3, `k` in W/(cell K), `h_lv` in J/kg.
#ifndef PECLET_FLOW_VOF_PHASE_CHANGE_HPP
#define PECLET_FLOW_VOF_PHASE_CHANGE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "vof/plic.hpp"

namespace peclet::flow::vof {

// ---------------------------------------------------------------------------------------------
// (1) the PLIC polygon area
// ---------------------------------------------------------------------------------------------

/// Area of the PLIC polygon `{x in [0,1]^3 : m.x = alpha}`, as a multiple of h^2.
///
/// The plane<->volume relation gives it for free: the volume between the planes at `alpha` and
/// `alpha + d(alpha)` is a slab of thickness `d(alpha)/|m|_2` on the polygon, so
///
///     A = |m|_2 * dV/d(alpha) .
///
/// `V(alpha)` is the Scardovelli–Zaleski piecewise cubic, so `dV/d(alpha)` is a piecewise
/// quadratic and this is ANALYTIC, not a finite difference. With the components mirrored into the
/// positive octant (an isometry: the area is unchanged), L1-normalized (`n1+n2+n3 = 1`) and sorted
/// ascending, and with `w = min(a, 1-a)` (the point reflection `x -> 1-x` gives `V(a)+V(1-a) = 1`,
/// hence `V'(a) = V'(1-a)`):
///
///     6 n1 n2 n3 V(w) = w^3 - sum_i <w-n_i>^3 + sum_i <w-(1-n_i)>^3        (<y> = max(y,0))
///     2 n1 n2 n3 V'(w) = w^2 - sum_i <w-n_i>^2 + sum_i <w-(1-n_i)>^2 .
///
/// Two rearrangements make it cancellation-free as `n1 -> 0` (a nearly axis-aligned plane, which
/// is EXACTLY the planar configuration rungs P0/P1 run on):
///   * `w^2 - <w-n1>^2` is evaluated as `n1(2w-n1)` where `w > n1` — a product, not a difference
///     of two O(1) numbers;
///   * the terms `<w-(1-n_2)>^2 = <w-n1-n3>^2` and `<w-n3>^2` are PAIRED (they differ by O(n1))
///     and their difference is evaluated in closed form. Likewise for the n2 pair.
///   * `<w-(1-n1)>^2 = <w-n2-n3>^2` is identically zero: `n1 <= 1/3` (it is the minimum of three
///     numbers summing to 1) so `n2+n3 >= 2/3 > 1/2 >= w`.
/// The degenerate branches (`n1 = 0`: the polygon is a rectangle; `n2 = 0`: an axis-aligned face)
/// are taken exactly, so a grid-aligned interface returns `A = 1` to the last bit.
KOKKOS_INLINE_FUNCTION double plicArea(double mx, double my, double mz, double alpha) {
  const double ax = Kokkos::fabs(mx), ay = Kokkos::fabs(my), az = Kokkos::fabs(mz);
  const double s = ax + ay + az;
  if (!(s > 0.0))
    return 0.0;  // degenerate normal: no polygon

  double beta = alpha;  // mirror into the positive octant (plic.hpp `plicVolume` convention)
  if (mx < 0.0)
    beta += ax;
  if (my < 0.0)
    beta += ay;
  if (mz < 0.0)
    beta += az;
  const double a1 = beta / s;
  if (!(a1 > 0.0) || !(a1 < 1.0))
    return 0.0;  // the plane misses the cell

  const double n1 = Kokkos::fmin(Kokkos::fmin(ax, ay), az) / s;
  const double n3 = Kokkos::fmax(Kokkos::fmax(ax, ay), az) / s;
  const double n2 = Kokkos::fmax(1.0 - n1 - n3, 0.0);
  const double nrm = Kokkos::sqrt(n1 * n1 + n2 * n2 + n3 * n3);  // |m|_2 in the normalized frame
  const double w = Kokkos::fmin(a1, 1.0 - a1);

  if (!(n2 > 0.0))
    return nrm;  // axis-aligned plane (n3 = 1): the polygon is a whole cell face, V'(w) = 1

  if (!(n1 > 0.0)) {  // 2-D: a rectangle. V'(w) = (w > n2 ? n2 : w)/(n2 n3); <w-n3> == 0 for w<=1/2
    return nrm * ((w > n2) ? n2 : w) / (n2 * n3);
  }

  // full 3-D, cancellation-free (see the note above)
  const double A1 = (w > n1) ? n1 * (2.0 * w - n1) : w * w;
  const double u2 = w - n2, u3 = w - n3;
  const double A2 = (u2 > n1) ? n1 * (n1 - 2.0 * u2) : ((u2 > 0.0) ? -u2 * u2 : 0.0);
  const double A3 = (u3 > n1) ? n1 * (n1 - 2.0 * u3) : ((u3 > 0.0) ? -u3 * u3 : 0.0);
  return nrm * (A1 + A2 + A3) / (2.0 * n1 * n2 * n3);
}

// ---------------------------------------------------------------------------------------------
// (2) plane geometry in the cell frame
// ---------------------------------------------------------------------------------------------

/// Unit (L2) interface normal from a PLIC normal of any scale. Points INTO THE GAS (liquid -> gas),
/// the `plic.hpp` convention. Returns the L2 norm of the input (0 => `n` untouched).
KOKKOS_INLINE_FUNCTION double pcUnitNormal(double mx, double my, double mz, double n[3]) {
  const double q = Kokkos::sqrt(mx * mx + my * my + mz * mz);
  if (q > 0.0) {
    n[0] = mx / q;
    n[1] = my / q;
    n[2] = mz / q;
  }
  return q;
}

/// Signed distance (in cells) from the CELL CENTRE to the PLIC plane, measured along `n` and
/// POSITIVE ON THE GAS SIDE: `phi(x) = (m.x - alpha)/|m|_2`, evaluated at x = (1/2,1/2,1/2).
/// A cell that is mostly liquid has `phi_c < 0`.
KOKKOS_INLINE_FUNCTION double pcCentreDistance(double mx, double my, double mz, double alpha) {
  const double q = Kokkos::sqrt(mx * mx + my * my + mz * mz);
  if (!(q > 0.0))
    return 0.0;
  return (0.5 * (mx + my + mz) - alpha) / q;
}

/// Signed distance from the plane of the neighbour at INTEGER cell offset `d`, given the centre
/// distance `phic` of the reference cell: `phi_j = phi_c + d.n` (cells, positive in the gas).
KOKKOS_INLINE_FUNCTION double pcOffsetDistance(double phic, const double n[3], double dx, double dy,
                                               double dz) {
  return phic + dx * n[0] + dy * n[1] + dz * n[2];
}

// ---------------------------------------------------------------------------------------------
// (3) the one-sided, pure-cell weighted least-squares normal gradient (Malan et al. 2021)
// ---------------------------------------------------------------------------------------------

/// Accumulator for `dT/dn` fitted through the interface value: `T - T_G = G * phi`, so
/// `G = sum_j w_j phi_j (T_j - T_G) / sum_j w_j phi_j^2`. Fitting THROUGH the interface (no
/// intercept) is what makes the Dirichlet condition part of the estimator instead of a separate
/// constraint.
struct PcGradFit {
  double num = 0.0;  ///< sum w phi (T - T_G)
  double den = 0.0;  ///< sum w phi^2
  int n = 0;         ///< samples used
};

/// Malan's collinearity weight for a sample at cell offset `d`: `w = xi / |d|` with
/// `xi = (d.n)^2/|d|^2` — i.e. `w = (d.n)^2 / |d|^3`. Samples lying along the normal dominate;
/// samples in the tangent plane (which carry no normal-gradient information) get zero weight.
KOKKOS_INLINE_FUNCTION double pcGradWeight(double dx, double dy, double dz, const double n[3]) {
  const double d2 = dx * dx + dy * dy + dz * dz;
  if (!(d2 > 0.0))
    return 0.0;
  const double dn = dx * n[0] + dy * n[1] + dz * n[2];
  return (dn * dn) / (d2 * Kokkos::sqrt(d2));
}

/// Add one pure-phase sample at normal distance `phi` with temperature `T`.
KOKKOS_INLINE_FUNCTION void pcGradAdd(PcGradFit& f, double w, double phi, double T, double Tg) {
  f.num += w * phi * (T - Tg);
  f.den += w * phi * phi;
  ++f.n;
}

/// The fitted normal derivative `dT/dn` (0 if the side carried no usable sample).
KOKKOS_INLINE_FUNCTION double pcGradSolve(const PcGradFit& f) {
  return (f.den > 0.0) ? (f.num / f.den) : 0.0;
}

// ---------------------------------------------------------------------------------------------
// (4) the physics rules
// ---------------------------------------------------------------------------------------------

/// `mdot = (k_g dT_g/dn - k_l dT_l/dn)/h_lv` with `n` the PLIC normal (liquid -> gas).
/// Positive = evaporation (liquid consumed). See the file header for the sign derivation.
KOKKOS_INLINE_FUNCTION double pcMassFlux(double kg, double gradG, double kl, double gradL,
                                         double hlv) {
  return (kg * gradG - kl * gradL) / hlv;
}

/// Liquid volume (fraction of a cell) removed from an interfacial cell in one step:
/// `dV = mdot A_G dt / rho_l`. Positive = evaporation.
KOKKOS_INLINE_FUNCTION double pcRegressVolume(double mdot, double area, double dt, double rhoL) {
  return mdot * area * dt / rhoL;
}

/// The volumetric divergence source of one interfacial cell (1/s, cell volume = 1):
/// `S = mdot A_G (1/rho_g - 1/rho_l)`.
KOKKOS_INLINE_FUNCTION double pcDivSource(double mdot, double area, double rhoG, double rhoL) {
  return mdot * area * (1.0 / rhoG - 1.0 / rhoL);
}

/// Interfacial temperature under the Schrage-derived interfacial heat-transfer resistance
/// (Bureš & Sato): `T_G = T_sat + mdot R_int`. `R_int = 0` is the hard Dirichlet.
KOKKOS_INLINE_FUNCTION double pcInterfaceTemperature(double Tsat, double mdot, double Rint) {
  return Tsat + mdot * Rint;
}

/// The clip-and-redistribute allocation: which face neighbours a clipped cell's residue is pushed
/// into, and in what fractions. `sgn = -1` pushes along `-n` (a LIQUID deficit goes to the liquid
/// side), `sgn = +1` along `+n` (a condensation excess goes to the gas side). `step[d]` is the
/// integer face step (-1, 0 or +1) and `w[d]` the fraction; both are zero for `n_d == 0`.
///
/// The raw weight is `n_d^2` (the direction cosines squared, so a plane normal to an axis puts
/// everything on that axis), restricted to the neighbours `avail[d]` says can absorb it — for a
/// liquid deficit, a neighbour that still HOLDS liquid. That restriction is not cosmetic: a
/// slightly-tilted plane otherwise sends a `n_t^2` share of the deficit into an in-plane neighbour
/// that is already empty, which leaves a permanent negative colour wisp there (measured -2.5e-6 on
/// the P1 Stefan ladder before the restriction; the transverse tilt came from the red-black
/// smoother's parity asymmetry in the energy solve, so it is not avoidable upstream).
///
/// Returns false when NO neighbour can absorb the residue; the unrestricted weights are then
/// returned, so the redistribution stays exactly conservative and the caller counts the event.
KOKKOS_INLINE_FUNCTION bool pcPushWeights(const double n[3], double sgn, const bool avail[3],
                                          int step[3], double w[3]) {
  double tot = 0.0, tota = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double p = sgn * n[d];
    w[d] = p * p;
    step[d] = (p > 0.0) ? 1 : ((p < 0.0) ? -1 : 0);
    tot += w[d];
    if (avail[d])
      tota += w[d];
  }
  const bool ok = tota > 0.0;
  for (int d = 0; d < 3; ++d) {
    if (ok)
      w[d] = avail[d] ? w[d] / tota : 0.0;
    else
      w[d] = (tot > 0.0) ? w[d] / tot : 0.0;
  }
  return ok;
}

/// Is this a cell the regression acts on? The PLIC contract of `advect_wy.hpp` (`wyIsMixed`) with
/// an explicit wisp guard: a cell whose colour is round-off residue has no meaningful plane, and
/// giving it an area would inject an unbounded `mdot` into the census.
KOKKOS_INLINE_FUNCTION bool pcIsInterfacial(double c, double eps) {
  return c > eps && c < 1.0 - eps;
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_PHASE_CHANGE_HPP
