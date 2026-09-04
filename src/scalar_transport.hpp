/// @file
/// @brief flow — cell-centred scalar transport (advection–diffusion) on the cut-cell grid.
///
/// A transported scalar c (temperature, concentration, phase fraction) obeys, in the solver's
/// divided-by-dt convention (dx = 1 grid units; physical diffusivity converted by the Python
/// layer):
///
///   (1/dt)(c^{n+1} - c^n) + div(open u c) = div(open D grad c) + S
///
/// Diffusion is backward-Euler implicit (an openness-weighted 7-band operator, solved by the same
/// red-black Gauss–Seidel as the pressure Poisson — cutcellSmoothColor); advection is explicit,
/// conservative flux-form, reusing the momentum limiter helpers (sadv::tvd / sou / fou_flux) with
/// the MAC face-normal velocities (staggered: C[fd].u is the -fd face velocity, co-located with the
/// face openness). Closed faces (openness 0) carry no flux and no diffusion, so an immersed solid
/// is adiabatic (zero-flux) for free and solid cells stay frozen (A_C = 1/dt, all off-diagonals 0).
///
/// This header holds the field-agnostic kernels + the per-scalar state; the Solver owns the scalars
/// and calls advanceScalars() at the end of step() with the just-projected divergence-free
/// velocity.
#ifndef PECLET_FLOW_SCALAR_TRANSPORT_HPP
#define PECLET_FLOW_SCALAR_TRANSPORT_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "mac_cutcell.hpp"
#include "staggered_advection.hpp"
#include "vof/phase_change.hpp"

namespace peclet::flow {

// Per-scalar BC on a domain face: 0 periodic (via the halo/periodic fill), 1 Neumann zero-flux
// (ghost = inner, i.e. adiabatic wall), 2 Dirichlet value (ghost = 2*value - inner reflection).
enum class ScalarBc { Periodic = 0, Neumann = 1, Dirichlet = 2 };

// One transported scalar. `c` aliases the Solver's registered field (fields_); the rest is private
// scratch on the same G=2 block. Bands are double (a scalar is cheap; no float-quantization
// concern).
struct ScalarField {
  std::string name;
  CCField c, cOld, b;                  // solution (registered), time base c^n, rhs
  CCField AC, AW, AE, AS, AN, AB, AT;  // implicit diffusion+time 7-band operator
  double D = 0.0;                      // constant diffusivity (grid units)
  int scheme = 1;                      // explicit advection flux: 0 FOU, 1 Koren TVD, 2 SOU
  int iters = 50;                      // RB-GS sweeps for the implicit diffusion solve
  int bc[6] = {0, 0, 0, 0, 0, 0};      // -x,+x,-y,+y,-z,+z (ScalarBc)
  double bcVal[6] = {0, 0, 0, 0, 0, 0};
  bool stencilBuilt = false;
  // Optional PER-CELL Dirichlet condition (WO-P01): where `dmask > 1/2` the cell's row is replaced
  // by `c = dval`. Unallocated (extent 0) by default and every consumer branches on that, so a
  // scalar that never asks for it runs the validated operator bit-for-bit. This is what pins
  // interfacial cells at the saturation temperature in the phase-change energy solve.
  CCField dmask, dval;
  // Optional CONSISTENT-ENERGY mode (WO-P23): the scalar is a temperature whose transport is done
  // GEOMETRICALLY (with the colour advection's own fluxes, `vof/energy_advect.hpp`) and whose
  // diffusion operator carries per-cell `k(C)` and a `rho c_p(C)` time term instead of the constant
  // `D`. Off, and both fields unallocated, unless `set_phase_change_energy` ran — every consumer
  // branches on that, so a scalar that never asks for it runs the validated operator bit-for-bit.
  bool energy = false;
  CCField kcell, rcp;  // k(C) and (rho c_p)(C) on the cells, refreshed by the solver each step
  CCField gfmB;        // WO-P23: the plane-anchored Dirichlet RHS contribution (allocated with it)
};

// Variable-coefficient sibling of `scalarBuildDiffusionOpen` (WO-P23):
//
//   A_C = rcp(i)*idt + sum_f k_f open_f ,   A_off = -k_f open_f
//
// with the face conductivity `k_f = pcFaceConductivity(k(i), k(j), mask(i), mask(j))` — the
// arithmetic mean of the two cells' `k(C)` EXCEPT where one of the two carries the per-cell
// Dirichlet condition, in which case the OTHER cell's `k` is used (see the derivation in
// `vof/phase_change.hpp`: a Dirichlet row is an identity row, so this coefficient's only job is to
// set the conductance with which the pure neighbour reaches a boundary condition that already sits
// at the interface).
//
// `mask` is the per-cell Dirichlet mask (all zeros if there is none).
inline void scalarBuildDiffusionVarK(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN,
                                     CCField AB, CCField AT, CCConst ox, CCConst oy, CCConst oz,
                                     CCConst kc, CCConst rcp, CCConst mask, double idt, C3 e,
                                     int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_diff_vark", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double ki = kc(i);
        const bool mi = mask(i) > 0.5;
        const double tw = vof::pcFaceConductivity(ki, kc(i - sx), mi, mask(i - sx) > 0.5) * ox(i);
        const double te = vof::pcFaceConductivity(ki, kc(i + sx), mi, mask(i + sx) > 0.5) *
                          ox(i + sx);
        const double ts = vof::pcFaceConductivity(ki, kc(i - sy), mi, mask(i - sy) > 0.5) * oy(i);
        const double tn = vof::pcFaceConductivity(ki, kc(i + sy), mi, mask(i + sy) > 0.5) *
                          oy(i + sy);
        const double tb = vof::pcFaceConductivity(ki, kc(i - sz), mi, mask(i - sz) > 0.5) * oz(i);
        const double tt = vof::pcFaceConductivity(ki, kc(i + sz), mi, mask(i + sz) > 0.5) *
                          oz(i + sz);
        AW(i) = -tw;
        AE(i) = -te;
        AS(i) = -ts;
        AN(i) = -tn;
        AB(i) = -tb;
        AT(i) = -tt;
        AC(i) = rcp(i) * idt + te + tw + tn + ts + tt + tb;
      });
}

// RHS of the consistent-energy solve: `b = rcp(i) * idt * T*`, with `T*` the GEOMETRICALLY advected
// temperature (the advective term is already in it, so there is none here). Together with the
// `rcp*idt` diagonal above this is `rho c_p (T^{n+1} - T*)/dt = div(k grad T^{n+1})`.
inline void scalarBuildRhsHeat(CCField b, CCConst cOld, CCConst rcp, double idt, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_rhs_heat", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        b(i) = rcp(i) * idt * cOld(i);
      });
}

// Replace the rows of the per-cell Dirichlet set by the identity: A_C = 1, all off-diagonals 0.
// Run AFTER scalarBuildDiffusionOpen (and after applyScalarBcStencil, which may have reopened a
// domain face on the same row).
inline void scalarMaskStencil(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN,
                              CCField AB, CCField AT, CCConst mask, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_mask_stencil", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (!(mask(i) > 0.5))
          return;
        AC(i) = 1.0;
        AW(i) = 0.0;
        AE(i) = 0.0;
        AS(i) = 0.0;
        AN(i) = 0.0;
        AB(i) = 0.0;
        AT(i) = 0.0;
      });
}

// RHS + solution seed of the per-cell Dirichlet set: b = value, and the field itself is set so the
// neighbours' first smoothing sweep already reads the imposed value.
inline void scalarMaskRhs(CCField b, CCField c, CCConst mask, CCConst val, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_mask_rhs", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (!(mask(i) > 0.5))
          return;
        b(i) = val(i);
        c(i) = val(i);
      });
}

// WO-P23: the PLANE-ANCHORED (ghost-fluid) form of the per-cell Dirichlet set. Rewrites the rows
// of the PURE cells that touch a masked (interfacial) cell so the Dirichlet value is imposed at the
// PLIC PLANE rather than at the masked cell's centre:
//
//   ... - k open (T_j - T_i)/1   ->   ... - k open (T_G - T_i)/theta ,
//
// with theta the distance (in cells) from cell i's centre to cell j's plane along the face's own
// axis (`pcGfmTheta`). The band toward the masked cell is zeroed and the Dirichlet contribution
// goes into `gfmB`, which the RHS build adds — the masked cell's own value is then never read by
// any neighbour, which is exactly what makes the condition one-sided-correct on BOTH sides. The
// Dirichlet value is `tgam` (the INTERFACE temperature `T_G`), NOT the masked cell's carried value
// `dval`: those are different numbers under the plane-anchored scheme (`pcCarriedValue`).
//
// Runs AFTER the base diffusion build (it consumes that build's band value to avoid
// double-counting, the same trick `patchScalarDirichletFace` uses) and touches only unmasked rows,
// so it commutes with `scalarMaskStencil`. `useK` selects the per-cell k(C) over the constant D.
inline void scalarMaskGfm(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN, CCField AB,
                          CCField AT, CCField gfmB, CCConst ox, CCConst oy, CCConst oz,
                          CCConst mask, CCConst tgam, CCConst gnx, CCConst gny, CCConst gnz,
                          CCConst gphi, CCConst kc, double Dconst, bool useK, double thMin,
                          double thMax, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_mask_gfm", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        gfmB(i) = 0.0;
        if (mask(i) > 0.5)
          return;  // masked rows are the identity (scalarMaskStencil); nothing to do
        const long st[3] = {sx, sy, sz};
        const double kd = useK ? kc(i) : Dconst;
        double acc = 0.0, dac = 0.0;
        for (int d = 0; d < 3; ++d)
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            const long j = i + (long)sgn * st[d];
            if (!(mask(j) > 0.5))
              continue;
            // face openness: the `-` face of the higher-indexed cell of the pair
            const double of = (d == 0)   ? ((sgn < 0) ? ox(i) : ox(i + sx))
                              : (d == 1) ? ((sgn < 0) ? oy(i) : oy(i + sy))
                                         : ((sgn < 0) ? oz(i) : oz(i + sz));
            const double nd = (d == 0) ? gnx(j) : (d == 1) ? gny(j) : gnz(j);
            const double th = vof::pcGfmTheta(gphi(j), nd, (double)sgn, thMin, thMax);
            const double coef = kd * of / th;
            // replace this face's interior coupling (band = -k_f of, AC += k_f of) by the Dirichlet
            // one, without double counting: AC += coef + band, band = 0.
            CCField band = (d == 0)   ? ((sgn < 0) ? AW : AE)
                           : (d == 1) ? ((sgn < 0) ? AS : AN)
                                      : ((sgn < 0) ? AB : AT);
            acc += coef + band(i);
            band(i) = 0.0;
            dac += coef * tgam(j);
          }
        AC(i) += acc;
        gfmB(i) = dac;
      });
}

// **WO-P3g — the SECOND-ORDER sibling of `scalarMaskGfm`.** Same contract, same inputs plus a
// per-cell curvature field `kap` and the row `order`; `order == 1 && !useKappa` reproduces
// `scalarMaskGfm` line for line (it is not called then — `advanceScalars` branches outside the
// kernel — but the two bodies are kept side by side so the difference is readable).
//
// What changes, and only this: the axis that carries the Dirichlet row is discretized by the
// non-uniform THREE-point second difference through `(T_behind, T_i, T_Gamma)` instead of the
// two-point `(T_i, T_Gamma)` one (`vof::pcGfmRow`, which writes the coefficient family out), and
// the distance `theta` is measured to the CURVED interface rather than to the interfacial cell's
// tangent plane (`vof::pcGfmThetaK`). `T_behind` is the cell one step AWAY from the interface along
// the same axis; where it is itself an identity row or its face is closed there is no third point
// and the row falls back to the shipped two-point form.
//
// The `behind` band is SCALED in place (`band *= a_behind`, `AC += (a_behind - 1) k_f o_f`), which
// is why this kernel must run after the base diffusion build and before `scalarMaskStencil`, in
// exactly the slot `scalarMaskGfm` occupies. Each axis touches only its own two bands and a masked
// pair on the same axis takes the two-point branch on both sides, so no band is written twice.
inline void scalarMaskGfm2(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN, CCField AB,
                           CCField AT, CCField gfmB, CCConst ox, CCConst oy, CCConst oz,
                           CCConst mask, CCConst tgam, CCConst gnx, CCConst gny, CCConst gnz,
                           CCConst gphi, CCConst kap, CCConst kc, double Dconst, bool useK,
                           double thMin, double thMax, int order, bool useKappa, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_mask_gfm2", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        gfmB(i) = 0.0;
        if (mask(i) > 0.5)
          return;
        const long st[3] = {sx, sy, sz};
        const double kd = useK ? kc(i) : Dconst;
        double acc = 0.0, dac = 0.0;
        for (int d = 0; d < 3; ++d)
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            const long j = i + (long)sgn * st[d];
            if (!(mask(j) > 0.5))
              continue;
            const double of = (d == 0)   ? ((sgn < 0) ? ox(i) : ox(i + sx))
                              : (d == 1) ? ((sgn < 0) ? oy(i) : oy(i + sy))
                                         : ((sgn < 0) ? oz(i) : oz(i + sz));
            const double nd = (d == 0) ? gnx(j) : (d == 1) ? gny(j) : gnz(j);
            const double kj = useKappa ? kap(j) : 0.0;
            const double th = vof::pcGfmThetaK(gphi(j), nd, (double)sgn, kj, thMin, thMax);
            CCField band = (d == 0)   ? ((sgn < 0) ? AW : AE)
                           : (d == 1) ? ((sgn < 0) ? AS : AN)
                                      : ((sgn < 0) ? AB : AT);
            CCField bandB = (d == 0)   ? ((sgn < 0) ? AE : AW)
                            : (d == 1) ? ((sgn < 0) ? AN : AS)
                                       : ((sgn < 0) ? AT : AB);
            const long jb = i - (long)sgn * st[d];
            const double bb = bandB(i);  // -k_f o_f of the face AWAY from the interface
            const bool behind = !(mask(jb) > 0.5) && (bb != 0.0);
            const vof::PcGfmRow row = vof::pcGfmRow(th, behind, order);
            const double coef = kd * of * row.aGamma;
            acc += coef + band(i);
            band(i) = 0.0;
            dac += coef * tgam(j);
            if (behind && row.aBehind != 1.0) {
              acc += -(row.aBehind - 1.0) * bb;  // bb = -k_f o_f, so this is +(a-1) k_f o_f
              bandB(i) = row.aBehind * bb;
            }
          }
        AC(i) += acc;
        gfmB(i) = dac;
      });
}

// Add the plane-anchored Dirichlet contribution to the RHS of the unmasked rows.
inline void scalarAddGfmRhs(CCField b, CCConst gfmB, CCConst mask, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_gfm_rhs", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (mask(i) > 0.5)
          return;
        b(i) += gfmB(i);
      });
}

// Build the implicit diffusion+time 7-band operator over inner cells:
//   A_C = idt + D*(ox(i)+ox(i+sx)+oy(i)+oy(i+sy)+oz(i)+oz(i+sz)),  A_off = -D*open_face.
// Openness-weighted (closed faces drop out) — the scalar analog of buildCutcellOp with the 1/dt
// diagonal. ox(i) is the -x face openness of cell i (== +x face of cell i-1), matching divergOpen.
inline void scalarBuildDiffusionOpen(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN,
                                     CCField AB, CCField AT, CCConst ox, CCConst oy, CCConst oz,
                                     double D, double idt, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_diff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double tw = D * ox(i), te = D * ox(i + sx);
        const double ts = D * oy(i), tn = D * oy(i + sy);
        const double tb = D * oz(i), tt = D * oz(i + sz);
        AW(i) = -tw;
        AE(i) = -te;
        AS(i) = -ts;
        AN(i) = -tn;
        AB(i) = -tb;
        AT(i) = -tt;
        AC(i) = idt + te + tw + tn + ts + tt + tb;
      });
}

// b = idt*c^n - div(open u c^n): explicit conservative openness-weighted advection into the RHS.
// U/V/W are the MAC face-normal velocities (staggered: C[fd].u; collocated: uf_/vf_/wf_) — U(i) is
// the -x face velocity of cell i, co-located with ox(i). scheme: 0 FOU, 1 Koren TVD, 2 SOU.
inline void scalarBuildRhs(CCField b, CCConst cOld, CCConst U, CCConst V, CCConst W, CCConst ox,
                           CCConst oy, CCConst oz, double idt, int scheme, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_rhs", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        double adv = 0.0;
        // per axis fd: flux through the +fd face (open(i+s), vel U(i+s)) minus the -fd face
        // (open(i), vel U(i)). Face value from the upwind limiter on the 4-cell stencil.
        for (int fd = 0; fd < 3; ++fd) {
          const long s = (fd == 0) ? sx : (fd == 1) ? sy : sz;
          CCConst Uf = (fd == 0) ? U : (fd == 1) ? V : W;
          CCConst Of = (fd == 0) ? ox : (fd == 1) ? oy : oz;
          const double velm = Uf(i), velp = Uf(i + s);
          const double om = Of(i), op = Of(i + s);
          const double cLL = cOld(i - 2 * s), cL = cOld(i - s), cR = cOld(i), cRR = cOld(i + s),
                       cRRR = cOld(i + 2 * s);
          double Fp, Fm;
          if (scheme == 0) {  // first-order upwind
            Fp = op * sadv::fou_flux(cR, cRR, velp);
            Fm = om * sadv::fou_flux(cL, cR, velm);
          } else if (scheme == 2) {  // second-order upwind (unlimited)
            Fp = op * sadv::sou(cL, cR, cRR, cRRR, velp);
            Fm = om * sadv::sou(cLL, cL, cR, cRR, velm);
          } else {  // Koren TVD (default)
            Fp = op * sadv::tvd(cL, cR, cRR, cRRR, velp);
            Fm = om * sadv::tvd(cLL, cL, cR, cRR, velm);
          }
          adv += Fp - Fm;
        }
        b(i) = idt * cOld(i) - adv;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_TRANSPORT_HPP
