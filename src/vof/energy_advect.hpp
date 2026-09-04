/// @file
/// @brief flow — VoF Part II (WO-P23): CONSISTENT `rho c_p T` transport, driven by the SAME
/// geometric fluxes as the colour advection of the same step (VOF_PLAN §9 item 6).
///
/// ## Why
///
/// The energy scalar of rungs P0/P1 was advected by `scalar_transport.hpp`'s Koren TVD flux, i.e.
/// by a DIFFERENT flux from the one that moved the colour. At a mixed cell the two disagree by
/// `O(a)`, and the temperature update then divides a heat content carried with one volume flux by
/// a heat capacity built from another — an artificial heating/cooling of order `d(rho c_p)`, which
/// at the water/steam `rho_l c_pl / (rho_g c_pg) ~ 2000` of the P2/P3 rungs is the whole answer.
/// Malan et al. (2021) and Boyd & Ling (2023) both fix it the same way: advect `H = (rho c_p) T`
/// with the colour's own fluxes and recover `T = H / (rho c_p)(C^{n+1})`.
///
/// ## The construction (the exact analogue of `momentum_advect.hpp`, one cell-centred unknown)
///
/// Per sweep in direction `d`, with `a_f` the face Courant number, `F_f` the geometric LIQUID
/// volume flux Weymouth-Yue already computed for the colour, and
/// `Phi_f = rcp_g (a_f - F_f) + rcp_l F_f` the heat-capacity flux of that face:
///
///     C_i += (F_-  - F_+ )          + flag_i (a_+ - a_-)
///     H_i += (Phi_- That_- - Phi_+ That_+) + rcp^_i T_i (a_+ - a_-) ,   rcp^ = rcp(flag_i)
///
/// `That_f` is the plain donor-cell (upwind) temperature of the face and `flag_i = H(C^n - 1/2)` is
/// the SAME frozen dilation indicator the colour uses; `T_i = H_i / rcp(C_i^{new})` recovers the
/// temperature. What is actually EVALUATED is the algebraically identical deviation form
/// `T_new = T_i + [Phi_-(That_- - T_i) - Phi_+(That_+ - T_i)] / rcp(C^{new})` — see `energyUpdate`.
///
/// **The identity that makes it consistent.** With `rcp(C) = rcp_g + drcp*C` and a uniform
/// `T == T0`: `Phi_- - Phi_+ = rcp_g(a_- - a_+) + drcp(F_- - F_+)` and the dilation coefficient
/// `rcp^_i = rcp_g + drcp*flag_i`, so `dH = T0 * drcp * dC = T0 * d(rcp)` exactly — `H = T0 rcp(C)`
/// is preserved and the recovery returns `T0` at ANY heat-capacity ratio, in floating point, with
/// no tuning knob. Note the dilation term is NOT gated by the flag (unlike the colour's): in a pure
/// gas cell it is `rcp_g T_i div u`, which is exactly the term that turns the conservative form
/// into the material derivative `rho c_p (dT/dt + u.grad T)` the energy balance of an evaporating
/// interface is written in. Dropping it is the classic "artificial heating at a divergence source".
///
/// ## Scope
///
/// Cut-cell geometry is NOT composed here (`enable_phase_change` already refuses an immersed
/// solid); `advect` throws if the advector carries geometry. Boundedness needs no flux clamp: the
/// flux is the colour's own, which Weymouth's proof already bounds on the pressure cell — the
/// clamp `momentum_advect.hpp` needs exists only because its control volume is HALF-SHIFTED and so
/// carries a colour the planes were not reconstructed from.
#ifndef PECLET_FLOW_VOF_ENERGY_ADVECT_HPP
#define PECLET_FLOW_VOF_ENERGY_ADVECT_HPP

#include <Kokkos_Core.hpp>
#include <functional>
#include <stdexcept>

#include "vof/advect_wy.hpp"
#include "vof/momentum_advect.hpp"  // vofMinmod
#include "vof/phase_change.hpp"

namespace peclet::flow::vof {

class VofEnergyAdvector {
 public:
  /// @param w      the colour advector whose block, planes, fluxes and face velocities are shared
  /// @param rcpG   `rho c_p` of the C = 0 phase (gas)
  /// @param rcpL   `rho c_p` of the C = 1 phase (liquid)
  void init(const WyAdvector& w, double rcpG, double rcpL) {
    if (!(rcpG > 0.0) || !(rcpL > 0.0))
      throw std::invalid_argument(
          "peclet::flow::vof::VofEnergyAdvector: both phase rho*c_p must be > 0");
    n_ = w.inner();
    e_ = w.extent();
    g_ = w.ghost();
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    rcpG_ = rcpG;
    rcpL_ = rcpL;
    T_ = SField("vof::energyT", len_);
    Tw_ = SField("vof::energyTw", len_);
  }
  bool initialized() const { return len_ > 0; }
  void setPhaseRcp(double g, double l) {
    rcpG_ = g;
    rcpL_ = l;
  }
  double phaseRcpG() const { return rcpG_; }
  double phaseRcpL() const { return rcpL_; }
  /// The transported temperature on the advector's block (inner cells mirror the solver's scalar).
  SField temperature() const { return T_; }

  /// MinMod-limited donor reconstruction of the face temperature, instead of the DEFAULT plain
  /// donor-cell (first-order upwind) value. OFF by default; turn it on deliberately and record it.
  ///
  /// Why it exists, measured: the geometric energy flux is what makes the transport CONSISTENT with
  /// the colour, but its donor-cell temperature carries the first-order upwind numerical diffusion
  /// `|u| h (1 - CFL)/2`, which thickens the thermal boundary layer and therefore LOWERS the
  /// interfacial gradient — exactly the quantity `mdot` is. On the P3 Scriven bubble at Ja = 0.5 the
  /// consistent transport with plain upwind reads -2.24 % on R(t) against -1.46 % for the scalar
  /// module's Koren TVD advection, i.e. the consistency is bought with accuracy at that Jakob
  /// number. The limited reconstruction buys it back.
  ///
  /// Why it is not the default: this is the energy twin of `momentumMuscl`, and there the slope's
  /// coefficient is unbounded in the density ratio on a control volume the sweep empties. The
  /// temperature has no such amplification (the recovery divides by `rho c_p(C^{new})`, which the
  /// flux is consistent with), but the boundedness statement is weaker than plain upwind's and the
  /// campaign's rule is that a scheme change ships as a measured switch.
  bool energyMuscl = false;

  /// Ghost policy for the temperature — the halo/periodic exchange plus the block's non-periodic
  /// zero-gradient clamp. Set by the solver; NOT the colour's own hook, because that one also runs
  /// the solid-band fill and the VoF boundary-colour rules, neither of which is a temperature.
  std::function<void(SField)> exchange;

  /// One consistent colour + energy Weymouth-Yue step. Replaces `WyAdvector::advect(dt, step)` —
  /// do not call both. `T_` must hold the current temperature on the inner cells on entry.
  void advect(WyAdvector& w, double dt, long step) {
    if (!initialized())
      throw std::runtime_error("peclet::flow::vof::VofEnergyAdvector: init() was never called");
    if (!exchange)
      throw std::runtime_error("peclet::flow::vof::VofEnergyAdvector: no `exchange` hook");
    if (w.hasGeometry())
      throw std::runtime_error(
          "peclet::flow::vof::VofEnergyAdvector: cut-cell geometry is not composed with the "
          "consistent energy transport at this rung (enable_phase_change already refuses a solid)");
    w.requireExchange();
    w.checkCourant(dt);
    const double dth = dt / w.h();
    w.freezeDilationFlag();
    w.timedExchange([&] { exchange(T_); });
    const int* perm = kWySweepPerm[static_cast<int>(step % 6)];
    for (int s = 0; s < 3; ++s) {
      const int d = perm[s];
      w.reconstruct();
      w.computeFluxes(d, dth);
      w.applySweep(d, dth);     // C^{(s+1)}: the capacity the recovery divides by
      energyUpdate(w, d, dth);  // T from the pre-sweep T, the fluxes, and that new C (swaps T_)
      w.exchangeTimed(w.colour());
      w.timedExchange([&] { exchange(T_); });  // regenerates every ghost of the swapped-in buffer
    }
  }

  /// Min/max of the transported temperature over the inner region (a boundedness read-out).
  void extrema(double& lo, double& hi) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField T = T_;
    double mn = 0.0, mx = 0.0;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    Kokkos::parallel_reduce(
        "vof::energy::min", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& a) { a = Kokkos::fmin(a, T(L3(x, y, z, e))); },
        Kokkos::Min<double>(mn));
    Kokkos::parallel_reduce(
        "vof::energy::max", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& a) { a = Kokkos::fmax(a, T(L3(x, y, z, e))); },
        Kokkos::Max<double>(mx));
    Kokkos::fence();
    lo = mn;
    hi = mx;
  }

  // nvcc requires members holding extended device lambdas to be public.

  /// The DEVIATION form of the update, and it is not cosmetic. Algebraically
  ///
  ///     T_new = H_new / rcp(C_new)
  ///           = T_i + [ Phi_- (That_- - T_i) - Phi_+ (That_+ - T_i) ] / rcp(C_new)
  ///
  /// because the three terms `-drcp dC T_i + (Phi_- - Phi_+) + rcp^_i (a_+ - a_-)` cancel exactly
  /// (that IS the consistency identity). Evolving the second form makes the cancellation happen in
  /// EXACT arithmetic instead of between two O(rcp_l T) floating-point numbers, so a uniform
  /// temperature is preserved BITWISE at any heat-capacity ratio rather than to ~1e-12 relative —
  /// the same reason `momentum_advect.hpp::momentumUpdate` evolves `u_old + dev/rho_new`.
  void energyUpdate(WyAdvector& w, int d, double dth) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sd = d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
    // The update is OUT OF PLACE: every cell reads its two upwind neighbours' temperatures, so
    // writing `T` in place is a read-write race between threads (measured: it moved the P2
    // interface position by 0.6 % and turned an order-2.8 ladder into order 0.5). `Tw_` takes the
    // new values and the two handles are swapped; `exchange` then regenerates every ghost.
    SField T = T_, Tw = Tw_, fl = w.faceFlux(), u = w.faceVel(d), c = w.colour();
    const double rg = rcpG_, rl = rcpL_;
    const double floor = 1e-12 * (rg < rl ? rg : rl);
    const bool muscl = energyMuscl;
    Kokkos::parallel_for(
        "vof::energy::update",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          const double aP = u(i) * dth, aM = u(i - sd) * dth;
          const double FP = fl(i), FM = fl(i - sd);
          const double phiP = rg * (aP - FP) + rl * FP;
          const double phiM = rg * (aM - FM) + rl * FM;
          const double ti = T(i);
          double tP, tM;
          if (!muscl) {
            tP = (aP > 0.0) ? ti : T(i + sd);
            tM = (aM > 0.0) ? T(i - sd) : ti;
          } else {
            // the flux slab of a face occupies the fraction |a| of the DONOR next to that face, so
            // its centroid sits (1/2 - |a|/2) from the donor centre TOWARDS the face
            const double wP = 0.5 - 0.5 * Kokkos::fabs(aP), wM = 0.5 - 0.5 * Kokkos::fabs(aM);
            if (aP > 0.0)
              tP = ti + wP * vofMinmod(ti - T(i - sd), T(i + sd) - ti);
            else
              tP = T(i + sd) - wP * vofMinmod(T(i + sd) - ti, T(i + 2 * sd) - T(i + sd));
            if (aM > 0.0)
              tM = T(i - sd) + wM * vofMinmod(T(i - sd) - T(i - 2 * sd), ti - T(i - sd));
            else
              tM = ti - wM * vofMinmod(ti - T(i - sd), T(i + sd) - ti);
          }
          const double cl = Kokkos::fmin(Kokkos::fmax(c(i), 0.0), 1.0);  // C AFTER this sweep
          const double r = pcPhaseMix(rg, rl, cl);
          Tw(i) = ti + (phiM * (tM - ti) - phiP * (tP - ti)) / (r > floor ? r : floor);
        });
    Kokkos::fence();
    SField tmp = T_;
    T_ = Tw_;
    Tw_ = tmp;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 0;
  long len_ = 0;
  double rcpG_ = 1.0, rcpL_ = 1.0;
  SField T_, Tw_;
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_ENERGY_ADVECT_HPP
