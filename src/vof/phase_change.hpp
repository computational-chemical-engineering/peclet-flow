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
  double num = 0.0;   ///< b1 = sum w phi (T - T_G)
  double den = 0.0;   ///< S2 = sum w phi^2
  double num2 = 0.0;  ///< b2 = sum w phi^2 (T - T_G)   (quadratic fit only)
  double s3 = 0.0;    ///< S3 = sum w phi^3             (quadratic fit only)
  double s4 = 0.0;    ///< S4 = sum w phi^4             (quadratic fit only)
  int n = 0;          ///< samples used
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
  const double dT = T - Tg, p2 = phi * phi;
  f.num += w * phi * dT;
  f.den += w * p2;
  f.num2 += w * p2 * dT;
  f.s3 += w * p2 * phi;
  f.s4 += w * p2 * p2;
  ++f.n;
}

/// The fitted normal derivative `dT/dn` (0 if the side carried no usable sample).
KOKKOS_INLINE_FUNCTION double pcGradSolve(const PcGradFit& f) {
  return (f.den > 0.0) ? (f.num / f.den) : 0.0;
}

/// **WO-P3f — the sample's distance to the CURVED interface, not to the tangent plane.**
///
/// `pcOffsetDistance` returns the distance to the interfacial cell's PLIC PLANE. The one-sided fit
/// then models `T` as a function of that, which is exact for a plane and FIRST ORDER in `h/R` for
/// anything curved: a sample at lateral offset `rho` from the normal line sits `rho^2/(2R)` further
/// from a sphere than from its tangent plane, and since `T` grows away from the interface every
/// off-axis sample is HOTTER than the plane model expects, so the fitted `dT/dn` — and with it
/// `mdot` — comes out systematically HIGH. Measured a priori on an exact sphere carrying an exactly
/// linear profile (`vof_scriven.py --mdot-probe --mdot-prof linear`), 128^3, sub = 16:
/// **+19.2 / +12.1 / +8.8 / +6.2 %** at R = 6 / 10 / 14 / 20 with observed order
/// **0.91 / 0.93 / 0.98** in `h/R` — a clean first-order curvature bias, and the largest single
/// error in the P3 rung.
///
/// The correction. With `kappa = div(n)` (the mean curvature of the level sets of the
/// gas-positive distance function; `-2/R` for a spherical gas bubble, whose `n` points inward),
/// the signed distance to the SURFACE of a point at plane distance `phi` and lateral offset `rho`
/// is `phi + kappa rho^2/4` to second order. Derivation, for a gas sphere of radius `R` with the
/// foot point at `R z_hat`: a point displaced `zeta` along `n = -z_hat` and `rho` tangentially sits
/// at radius `sqrt((R-zeta)^2 + rho^2) ~ R - zeta + rho^2/(2R)`, so the gas-positive distance
/// `R - r` is `zeta - rho^2/(2R)` where the plane says `zeta`; and `kappa/4 = -1/(2R)`.
KOKKOS_INLINE_FUNCTION double pcCurvedDistance(double phi, double dx, double dy, double dz,
                                               const double n[3], double kappa) {
  const double dn = dx * n[0] + dy * n[1] + dz * n[2];
  const double rho2 = Kokkos::fmax(dx * dx + dy * dy + dz * dz - dn * dn, 0.0);
  return phi + 0.25 * kappa * rho2;
}

/// **The QUADRATIC one-sided fit** (WO-P23): `T - T_G = G phi + Q phi^2`, returning `G`.
///
/// The linear fit above is a straight line through the interface value fitted to samples that
/// start ~1 cell away and reach ~2.5, so a profile with curvature `T''` biases it by
/// `O(T'' h)` — a clean FIRST-ORDER error in `mdot`, and (once the plane-anchored Dirichlet has
/// removed the cell-centre mismatch) the leading error of the whole rung. This is the
/// Aslam-quadratic lever VOF_PLAN §9 item 1 names, in its least-squares rather than PDE form:
/// the same samples, one more basis function, no extra stencil reach and no sweeps.
///
/// Normal equations of the 2-parameter weighted fit:
///     [S2 S3; S3 S4] [G; Q] = [b1; b2] ,   S_k = sum w phi^k ,  b_k = sum w phi^k (T - T_G).
/// Falls back to the linear solve when the system is ill-conditioned (fewer than two distinct
/// normal distances — the determinant is then zero to round-off), so a side that carries a single
/// usable sample still returns something rather than a division by noise.
KOKKOS_INLINE_FUNCTION double pcGradSolve2(const PcGradFit& f) {
  const double det = f.den * f.s4 - f.s3 * f.s3;
  if (f.n < 2 || !(Kokkos::fabs(det) > 1e-12 * f.den * f.s4))
    return pcGradSolve(f);
  return (f.num * f.s4 - f.num2 * f.s3) / det;
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

/// **The plane-anchored (ghost-fluid) interfacial Dirichlet condition** (WO-P23, the first item).
///
/// Rungs P0/P1 pinned the whole interfacial CELL at `T_G`, so the numerical thermal boundary sat at
/// the CELL CENTRE while the mass-flux gradient is fitted from the PLIC PLANE. The mismatch is up
/// to half a cell and it CHANGES SIGN as the interface sweeps through a cell — the first-order,
/// oscillating error component WO-P01 measured (P1 orders 1.07 on 64->128 and 1.50 on 128->256).
///
/// The repair is a PER-FACE condition, not a per-cell value. For a PURE cell `i` whose neighbour
/// `j` across the face in direction `d` is interfacial, the plane of cell `j` sits at signed
/// normal distance `phi_i = phi_c(j) - s n_d(j)` from cell `i`'s centre (`s` the face step from
/// `i` to `j`), so along the grid line it is `theta = |phi_i| / |n_d|` cells away. That face's row
/// therefore becomes the DIRICHLET row `k open (T_i - T_G)/theta` instead of the interior coupling
/// `k open (T_i - T_j)`, and the interfacial cell's own value is never read.
///
/// **Why the per-CELL value the work order describes does NOT work, measured.** Giving the
/// interfacial cell the value the one-sided profile takes at its centre is right for the neighbour
/// on the side the fit was taken from and WRONG for the neighbour on the other side: on the P1
/// Stefan ladder it heats the saturated LIQUID through the interfacial cell's liquid-side face,
/// which gives the liquid a spurious gradient, which feeds straight back into
/// `mdot = (k_g G_g - k_l G_l)/h_lv`. The error saturates instead of converging — measured
/// +6.20 / +5.62 / +5.42 % at N = 64/128/256, observed order 0.10, against +1.31 / +0.59 / +0.20 %
/// and order 1.37 for the cell-centre pinning it was meant to improve on. One cell-centred value
/// cannot serve two sides; the per-face form below can, and does (the liquid-side face then reads
/// `T_G` at its own theta, i.e. exactly zero flux for a saturated liquid).
///
/// The value to CARRY in an interfacial cell (WO-P23). With the plane-anchored rows above, no
/// neighbour reads it — the solve's only use for it is that the cell becomes PURE a few steps later
/// as the interface sweeps past, and then it IS read, both by the energy operator and by the
/// one-sided gradient fit. Leaving it at `T_G` makes every newly exposed cell start at the
/// saturation temperature, i.e. a cold spot that the diffusion then has to remove: measured on the
/// P1 Stefan ladder it holds the interface back by a clean first order (-1.31 / -0.72 / -0.36 % at
/// N = 64/128/256, order 0.93). The value the one-sided linear profile takes at the cell centre,
/// `T_G + G phi_c`, removes it. (This is the per-cell value the work order describes; it is right
/// HERE, where nothing reads it across a face, and wrong as a boundary condition — see above.)
KOKKOS_INLINE_FUNCTION double pcCarriedValue(double Tg, double grad, double phic) {
  return Tg + grad * phic;
}

/// `thetaMin` keeps the coefficient finite when the plane grazes the cell centre (the standard GFM
/// clamp); `thetaMax` bounds it when the plane sits deep inside the far cell (`|phi_c| <=
/// sqrt(3)/2`, so `theta <= 1 + sqrt(3)/2`) AND when the interface is nearly PARALLEL to this
/// face's axis, where `|phi_i|/|n_d|` diverges and the honest statement is "the interface is far
/// along this line".
///
/// **The `n_d -> 0` limit has to be `thetaMax`, and it has to be reached CONTINUOUSLY.** The first
/// version fell back to `theta = 1` below `|n_d| = 1e-6`, i.e. a JUMP of 1 -> 1.9 in that face's
/// coefficient across a threshold that a transverse MYC normal component crosses on ROUND-OFF: in a
/// quasi-1D scene the energy solve's red-black parity asymmetry gives symmetric columns ~1e-16
/// differences in T and hence ~1e-8 transverse normal components (WO-P01 finding 3), so the
/// transverse faces of every interfacial cell sit on that threshold. Measured on the P2
/// sucking-interface MPI gate at np = 1 — where the two solvers differ only by round-off —
/// `max|C_dist - C_ref|` went 3.3e-16 at 3 steps -> 2.4e-4 at 12 -> 4.5e-4 at 55 (a plateau) with
/// the jump, against 4.1e-14 / 7.4e-14 for the rung P0/P1 treatment on the same scene. A
/// discontinuous coefficient is a round-off amplifier even when the jump itself is bounded.
KOKKOS_INLINE_FUNCTION double pcGfmTheta(double phiC, double nd, double s, double thetaMin,
                                         double thetaMax) {
  const double den = Kokkos::fabs(nd);
  const double num = Kokkos::fabs(phiC - s * nd);
  if (!(num < thetaMax * den))  // includes den == 0: the limit, reached continuously
    return thetaMax;
  return Kokkos::fmax(num / den, thetaMin);
}

/// Face conductivity of the variable-coefficient energy operator, `k(C)` arithmetic in the colour
/// with ONE exception: where one of the two cells carries the per-cell Dirichlet condition
/// (an interfacial cell pinned at the plane-anchored `T_G`) the face takes the OTHER cell's `k`.
///
/// The exception is not cosmetic. A Dirichlet row is an identity row, so the only thing this face
/// coefficient does is set the conductance with which the PURE neighbour draws heat to the
/// interface. That conductance must be the pure neighbour's own phase conductivity — an arithmetic
/// mean with `k(C_iface)` puts a slab of the WRONG phase between the pure cell and a boundary
/// condition that already sits (plane-anchored) at the interface, and at the water/steam ratio
/// `k_l/k_g ~ 26` that is a 50 % error in the wall heat flux of every interfacial face.
KOKKOS_INLINE_FUNCTION double pcFaceConductivity(double ki, double kj, bool mi, bool mj) {
  if (mi && !mj)
    return kj;
  if (mj && !mi)
    return ki;
  return 0.5 * (ki + kj);
}

/// Linear-in-C phase property (`k`, `rho c_p`), written so that `C = 0` and `C = 1` return the
/// endpoint EXACTLY — the same construction as `vofPhaseRho` in `momentum_advect.hpp`.
KOKKOS_INLINE_FUNCTION double pcPhaseMix(double vg, double vl, double c) {
  return vg * (1.0 - c) + vl * c;
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

// ---------------------------------------------------------------------------------------------
// (5) WO-P3g — the SECOND-ORDER interfacial energy operator
// ---------------------------------------------------------------------------------------------

/// **WO-P3g item 3 — the curvature-consistent GFM distance.**
///
/// `pcGfmTheta` measures cell `i`'s distance to the interfacial neighbour `j`'s PLIC PLANE. On a
/// curved interface that is first order in `h/R`, exactly as it is for the one-sided fit
/// (`pcCurvedDistance`), and WO-P3f measured the consequence on the row: the GFM flux carries an
/// interface-curvature bias about TWICE the fit's (`+12.4 %` against `+6.2 %` at `R = 20`).
///
/// The correction is the same one, and it is exact to second order for a sphere. Cell `i` sits at
/// integer offset `d = -s e_d` from the interfacial cell `j`, so its plane distance is
/// `phi_i = phi_c(j) - s n_d` and its LATERAL offset from the plane's foot point is
/// `rho^2 = |d|^2 - (d.n)^2 = 1 - n_d^2` (the foot point differs from `j`'s centre by a purely
/// NORMAL displacement, so the tangential part of the offset is unchanged). The signed distance to
/// the surface is then `phi_i + kappa rho^2 / 4` with `kappa = div(n)` (`-2/R` for a gas sphere),
/// and the distance ALONG the grid line is that divided by `|n_d|` to the same order.
///
/// `kappa == 0` returns `pcGfmTheta` bitwise — that is the shipped default and the inertness
/// contract of this rung.
KOKKOS_INLINE_FUNCTION double pcGfmThetaK(double phiC, double nd, double s, double kappa,
                                          double thetaMin, double thetaMax) {
  if (kappa == 0.0)
    return pcGfmTheta(phiC, nd, s, thetaMin, thetaMax);
  const double den = Kokkos::fabs(nd);
  const double num = Kokkos::fabs(phiC - s * nd + 0.25 * kappa * (1.0 - nd * nd));
  if (!(num < thetaMax * den))
    return thetaMax;
  return Kokkos::fmax(num / den, thetaMin);
}

/// **WO-P3g item 2 — the Gibou–Fedkiw second-order one-sided (ghost-fluid) row.**
///
/// The shipped row is a TWO-POINT difference: for a pure cell `i` whose neighbour across the face
/// is interfacial, the face's interior coupling `k o (T_j - T_i)` is replaced by
/// `k o (T_Gamma - T_i)/theta`, i.e. the whole axis contributes
///
///     (1/theta)(T_Gamma - T_i)  +  1 * (T_behind - T_i)
///
/// with `T_behind` the cell one step AWAY from the interface along the same axis. That is a
/// first-order approximation of `d2T/dx2` unless `theta = 1`, and WO-P3f measured its cost on a
/// FLAT interface where nothing else is wrong: the flux the rows draw is **−17 %** at a 2.4-cell
/// thermal layer and **−5 %** at 8 cells.
///
/// The repair is the standard non-uniform three-point second difference through
/// `(T_behind, T_i, T_Gamma)` at spacings `(h, theta h)` — Gibou et al., JCP 176:205 (2002):
///
///     d2T/dx2 = 2/(h_L + h_R) [ (T_R - T_i)/h_R - (T_i - T_L)/h_L ]
///             = 2/((1 + theta) theta) (T_Gamma - T_i)  +  2/(1 + theta) (T_behind - T_i)   (h = 1)
///
/// so the COEFFICIENT FAMILY is `a_Gamma = 2/((1+theta) theta)`, `a_behind = 2/(1+theta)`, against
/// the shipped `(1/theta, 1)`. It reproduces a QUADRATIC profile EXACTLY (the three-point divided
/// difference `2 f[x_L, x_i, x_R]` is exact for quadratics at any spacing), it reduces to the
/// interior row at `theta = 1` bitwise (`2/2 = 1` both), and it is bounded over the shipped clamp
/// `theta in [0.1, 1.9]`: `a_Gamma in [0.363, 18.18]`, `a_behind in [0.690, 1.818]`. The row is
/// NON-SYMMETRIC (cell `behind` does not carry the mirror coefficient) but remains strictly
/// diagonally dominant, which is all the red–black Gauss–Seidel smoother needs.
///
/// `behind == false` — the cell one step away is itself interfacial, or that face is CLOSED, so
/// there is no third point — falls back to the shipped two-point row. This is Gibou's own rule for
/// a stencil that cannot reach a second usable node, and it is also what the `theta -> 0` clamp
/// leaves bounded.
struct PcGfmRow {
  double aGamma;   ///< coefficient of `(T_Gamma - T_i)`, times `k_f o_f`
  double aBehind;  ///< coefficient of `(T_behind - T_i)`, times that face's own `k_f o_f`
};
KOKKOS_INLINE_FUNCTION PcGfmRow pcGfmRow(double theta, bool behind, int order) {
  if (order < 2 || !behind)
    return PcGfmRow{1.0 / theta, 1.0};
  return PcGfmRow{2.0 / ((1.0 + theta) * theta), 2.0 / (1.0 + theta)};
}

/// **WO-P3g item 1 — `mdot` from the ENERGY OPERATOR's own interfacial flux.**
///
/// `q` is the heat the operator's Dirichlet rows actually draw INTO the interface from its pure
/// face neighbours, `q = sum_p a_Gamma(p) k_p o_p (T_p - T_Gamma)` (positive = the surrounding
/// phases are superheated and feed evaporation), `coefSum = sum_p a_Gamma(p) k_p o_p`, `area` the
/// interfacial area of the cell and `Rint` the Schrage/IHTR resistance of `T_Gamma = T_sat +
/// mdot R_int`. The interfacial energy balance `mdot h_lv A = q` then closes in ONE step:
///
///     mdot (h_lv A + R_int coefSum) = sum_p a_Gamma(p) k_p o_p (T_p - T_sat) .
///
/// Why this and not the least-squares fit: the heat the energy equation LOSES and the mass the
/// regression PRODUCES become the same discrete number, so the flux consistency
/// `-q_gfm / (mdot h_lv A)` is 1 by construction instead of the measured 0.95…1.04 (WO-P3f), and
/// the `mdot A` products that drive the regression and the divergence source no longer contain the
/// interfacial AREA at all — it cancels, `dV = q dt/(h_lv rho_l)`.
KOKKOS_INLINE_FUNCTION double pcOperatorMassFlux(double q, double coefSum, double area, double hlv,
                                                 double Rint) {
  const double den = hlv * area + Rint * coefSum;
  return (den > 0.0) ? q / den : 0.0;
}

/// Is this a cell the regression acts on? The PLIC contract of `advect_wy.hpp` (`wyIsMixed`) with
/// an explicit wisp guard: a cell whose colour is round-off residue has no meaningful plane, and
/// giving it an area would inject an unbounded `mdot` into the census.
KOKKOS_INLINE_FUNCTION bool pcIsInterfacial(double c, double eps) {
  return c > eps && c < 1.0 - eps;
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_PHASE_CHANGE_HPP
