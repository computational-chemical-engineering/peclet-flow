/// @file
/// @brief flow — VoF rung V6 (WO-V6): the DYNAMIC contact angle (Cox-Voinov with an explicit slip
/// length) and advancing/receding HYSTERESIS with pinning.
///
/// Nothing in the V5b fill changes; only the VALUE of theta per contact cell does. `wetting.hpp`
/// already reads theta as a per-cell field (`WyAdvector::contactAngle()`), so this rung is a
/// producer for that field and nothing else — the plane construction, the anchor, the branch
/// census and passes 2-3 of the band fill are WO-S's, untouched.
///
/// ## The model — state it on every page and docstring
///
/// Afkhami, Zaleski & Bussmann, *JCP* 228:5370 (2009). A sharp-interface simulation cannot resolve
/// the microscopic contact line, so the angle it imposes at the grid scale `Delta` is not the
/// equilibrium angle but the Cox-Voinov apparent angle at that scale, with the outer cut-off set
/// to the cell size and the inner one to an EXPLICIT slip length `lambda`:
///
///     theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta / lambda),   Ca_cl = mu_l U_cl / sigma
///
/// (angles in RADIANS; `Ca_cl > 0` advancing, `< 0` receding). The point of the explicit `lambda`
/// is that the *numerical* slip of a VoF contact line is proportional to `Delta`, so a computation
/// without it silently makes the imposed angle grid-dependent — VOF_PLAN §6's "grid-dependent
/// mobility" trap. **Never report a dynamic-wetting result without stating `lambda`.**
///
/// `theta_Delta` is clamped into `[thetaMin, thetaMax]` (default 1 deg / 179 deg): the cubic has no
/// solution above the maximum receding capillary number (where a liquid film is entrained) and the
/// fill's plane construction degenerates at 0/180.
///
/// ## Hysteresis
///
/// With an advancing angle `theta_a` and a receding angle `theta_r` (Dussan; Fang et al.), the
/// contact line is PINNED while its apparent angle lies between them:
///
///     theta_app > theta_a   ->   impose theta_a  (+ the Cox-Voinov correction with U_cl)
///     theta_app < theta_r   ->   impose theta_r  (+ the correction)
///     otherwise             ->   impose theta_app  (PINNED: the fill then reproduces the
///                                                   current interface, so nothing moves)
///
/// The pinned branch is the load-bearing one and it is exact *because* the V5b fill is idempotent
/// (WO-S finding 1): writing the band with the plane of the CURRENT apparent angle is the identity
/// on the current configuration, so no Young force appears and the line does not move. A fill that
/// were not idempotent could not express pinning at all.
///
/// ## How `U_cl` is measured
///
/// `theta_app` comes from the same quantity the fill rotates: `cos(theta_app) = m_f . n_w` with
/// `m_f` the fluid-only Youngs normal of the anchor cell (`wetting.hpp`) and `n_w` the wall normal.
/// The contact line moves along the IN-WALL direction `t_hat = (m_f - (m_f.n_w) n_w)/|...|`, which
/// points from the LIQUID towards the DRY wall (because `m` points into the gas), so the liquid
/// ADVANCES when the fluid velocity points along `+t_hat`:
///
///     U_cl = + u_anchor . t_hat
///
/// (the work order writes `-t_hat`; that is a sign slip — see `vofContactLineSpeed` for the
/// derivation and the measurement that settles it).
///
/// `u_anchor` is the cell-centre velocity (the mean of the cell's two faces per axis) of the anchor
/// cell, and `U_cl` is then smoothed with a 3-POINT MEAN along `t_hat` (the cell and its two
/// in-wall neighbours in the direction of motion) — a MAC velocity next to a wall is noisy cell to
/// cell and an unsmoothed `Ca_cl` puts that noise straight into the imposed angle through the cube
/// root.
///
/// ## Container rule
///
/// The angle kernels below (`coxVoinovAngle`, `vofHysteresisBase`, `vofDynamicContactAngle`,
/// `vofWallTangent`, `vofContactLineSpeed`) are **container-free** — scalars and small arrays only,
/// the `plic.hpp` rule — so a host oracle calls them verbatim (that IS gate G1). `VofDynamicWetting`
/// below is the driver that runs them over the colour block; it owns only its own scratch views and
/// reads the advector through its public accessors, so `advect_wy.hpp` is not touched by this rung.
#ifndef PECLET_FLOW_VOF_WETTING_DYNAMIC_HPP
#define PECLET_FLOW_VOF_WETTING_DYNAMIC_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <functional>

#include "vof/advect_wy.hpp"

namespace peclet::flow::vof {

/// Which rule produced a band cell's imposed angle (`vof_dynamic_field(4)`).
enum VofDynamicState : int {
  kVofDynNone = 0,        ///< not a contact cell (no anchor / no usable normal): theta = the base
  kVofDynStatic = 1,      ///< Cox-Voinov applied to the static base (no hysteresis configured)
  kVofDynPinned = 2,      ///< theta_r <= theta_app <= theta_a: the apparent angle is imposed
  kVofDynAdvancing = 3,   ///< theta_app > theta_a: theta_a (+ Cox-Voinov)
  kVofDynReceding = 4,    ///< theta_app < theta_r: theta_r (+ Cox-Voinov)
  kVofDynStateCount = 5
};

/// Cox-Voinov / Afkhami-Zaleski-Bussmann grid-scale apparent angle, in RADIANS.
///
///     theta^3 = thetaE^3 + 9 Ca logRatio,      logRatio = ln(Delta/lambda)
///
/// Clamped into `[thetaMin, thetaMax]`; a non-positive cube (the film-entrainment branch of a fast
/// receding line) returns `thetaMin`.
KOKKOS_INLINE_FUNCTION double coxVoinovAngle(double thetaE, double ca, double logRatio,
                                             double thetaMin, double thetaMax) {
  const double t3 = thetaE * thetaE * thetaE + 9.0 * ca * logRatio;
  double th = t3 > 0.0 ? Kokkos::pow(t3, 1.0 / 3.0) : thetaMin;
  if (!(th == th))  // NaN guard: a caller-supplied NaN velocity must not reach the fill
    th = thetaE;
  return th < thetaMin ? thetaMin : (th > thetaMax ? thetaMax : th);
}

/// The hysteresis selector: which base angle the Cox-Voinov correction is applied to, and whether
/// the line is pinned. All angles in RADIANS.
KOKKOS_INLINE_FUNCTION int vofHysteresisBase(double thetaApp, double thetaA, double thetaR,
                                             double& base) {
  if (thetaApp > thetaA) {
    base = thetaA;
    return kVofDynAdvancing;
  }
  if (thetaApp < thetaR) {
    base = thetaR;
    return kVofDynReceding;
  }
  base = thetaApp;
  return kVofDynPinned;
}

/// The whole rung in one pure function: hysteresis selector, then Cox-Voinov on the selected base.
///
/// @param thetaApp  the measured apparent angle (rad)
/// @param uCl       the contact-line speed, + advancing
/// @param muL,sigma the liquid viscosity and the surface tension (`Ca_cl = muL uCl / sigma`)
/// @param logRatio  `ln(Delta/lambda)`; pass 0 to switch the Cox-Voinov correction off
/// @param hyst      whether `thetaA`/`thetaR` are configured
/// @param thetaE    the static base angle (rad) used when `hyst` is false
/// @param thetaOut  out: the angle to impose (rad)
/// @param caOut     out: `Ca_cl`
/// @return the `VofDynamicState`
KOKKOS_INLINE_FUNCTION int vofDynamicContactAngle(double thetaApp, double uCl, double muL,
                                                  double sigma, double logRatio, bool hyst,
                                                  double thetaA, double thetaR, double thetaE,
                                                  double thetaMin, double thetaMax,
                                                  double& thetaOut, double& caOut) {
  caOut = sigma > 0.0 ? muL * uCl / sigma : 0.0;
  double base = thetaE;
  int state = kVofDynStatic;
  if (hyst) {
    state = vofHysteresisBase(thetaApp, thetaA, thetaR, base);
    if (state == kVofDynPinned) {
      // The fill is idempotent, so imposing the apparent angle reproduces the current interface
      // and the line does not move. No Cox-Voinov correction: the line is not moving.
      thetaOut = base < thetaMin ? thetaMin : (base > thetaMax ? thetaMax : base);
      return state;
    }
  }
  thetaOut = coxVoinovAngle(base, caOut, logRatio, thetaMin, thetaMax);
  return state;
}

/// The in-wall (tangential) unit direction of the interface and its apparent cosine.
/// `mfIn` is the fluid-side normal (into the gas), `nwIn` the wall normal (solid -> fluid); both
/// any scale. Returns false when the interface is parallel to the wall (`|t| < tEps`), in which
/// case `that` is zeroed and no contact-line direction is defined.
KOKKOS_INLINE_FUNCTION bool vofWallTangent(const double mfIn[3], const double nwIn[3], double tEps,
                                           double that[3], double& cosApp) {
  double mn = Kokkos::sqrt(mfIn[0] * mfIn[0] + mfIn[1] * mfIn[1] + mfIn[2] * mfIn[2]);
  if (!(mn > 0.0))
    mn = 1.0;
  double wn = Kokkos::sqrt(nwIn[0] * nwIn[0] + nwIn[1] * nwIn[1] + nwIn[2] * nwIn[2]);
  if (!(wn > 0.0))
    wn = 1.0;
  const double mh[3] = {mfIn[0] / mn, mfIn[1] / mn, mfIn[2] / mn};
  const double nw[3] = {nwIn[0] / wn, nwIn[1] / wn, nwIn[2] / wn};
  cosApp = mh[0] * nw[0] + mh[1] * nw[1] + mh[2] * nw[2];
  double t[3] = {mh[0] - cosApp * nw[0], mh[1] - cosApp * nw[1], mh[2] - cosApp * nw[2]};
  const double tn = Kokkos::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  if (tn < tEps) {
    that[0] = that[1] = that[2] = 0.0;
    return false;
  }
  for (int d = 0; d < 3; ++d)
    that[d] = t[d] / tn;
  return true;
}

/// `U_cl = +u . t_hat`: positive when the LIQUID ADVANCES.
///
/// **This sign is a CORRECTION of the work order**, which writes "sign positive when the liquid
/// advances (velocity along `-t_hat`, since `m` points into the gas)". `m` points into the gas, so
/// its in-wall part `t_hat` points from the LIQUID side towards the DRY side along the wall; the
/// liquid advancing over dry wall is therefore motion along `+t_hat`, not `-t_hat`. Concretely, a
/// vertical interface with liquid at `x < x0` has `m = +x_hat` and `t_hat = +x_hat`, and a flow
/// `u = +U x_hat` pushes the liquid onto the dry wall — advancing, `Ca_cl > 0`, which Cox-Voinov
/// must turn into `theta_D > theta_e` (viscous bending RETARDS spreading; the Young force
/// `cos theta_e - cos theta_D` then opposes the motion). With the work order's sign the same flow
/// would report a receding line and Cox-Voinov would ACCELERATE the spreading — an unstable
/// feedback, which is what the gate measures.
KOKKOS_INLINE_FUNCTION double vofContactLineSpeed(const double u[3], const double that[3]) {
  return u[0] * that[0] + u[1] * that[1] + u[2] * that[2];
}

/// The driver: produces `WyAdvector::contactAngle()` from the current colour, geometry, wall SDF,
/// fluid-only normals and cell-centre velocity, once per band fill.
///
/// Decomposition independence (the same argument as WO-Q finding 5 / WO-S finding 9): pass A runs
/// at ghost depth <= 2, where every field it reads (sdf, colour, kind, fluid-only normal, cell
/// velocity) has already been made owner-correct by the block's own ghost policy; its outputs are
/// then EXCHANGED so that the 3-point smoothing of pass B, which reaches depth 3, reads the owner's
/// values too. That is what makes the imposed angle — and hence the fill — bitwise across np.
struct VofDynamicWetting {
  // ---- configuration (all angles in RADIANS) -------------------------------------------------
  bool dynamic = false;      ///< the Cox-Voinov correction is configured
  bool hysteresis = false;   ///< theta_a / theta_r are configured
  double thetaA = 0.0, thetaR = 0.0;
  double slip = 0.1;         ///< lambda, in CELLS (Delta = 1 cell), so logRatio = -ln(slip)
  double muLiquid = 0.0;     ///< the LIQUID dynamic viscosity used in Ca_cl
  double sigma = 0.0;        ///< the surface tension used in Ca_cl (the solver resolves it)
  double thetaMin = 1.0 * 3.14159265358979323846 / 180.0;
  double thetaMax = 179.0 * 3.14159265358979323846 / 180.0;
  bool smooth = true;        ///< the 3-point in-wall mean of U_cl (ablation)
  double pureEps = 1e-8;     ///< a donor this close to 0/1 carries no interface
  double tanEps = 1e-6;      ///< below this the interface is parallel to the wall

  bool active() const { return dynamic || hysteresis; }
  double logRatio() const { return dynamic ? -std::log(slip) : 0.0; }

  // ---- state ---------------------------------------------------------------------------------
  bool allocated() const { return base_.extent(0) > 0; }
  /// The STATIC per-cell base angle (radians) the selector starts from. The solver copies
  /// `WyAdvector::contactAngle()` into it whenever the static angle is (re)set, because the pass
  /// OVERWRITES `contactAngle()` and must not read back its own previous output.
  SField base() const { return base_; }
  SField imposed() const { return imposed_; }   ///< the angle written this fill (rad)
  SField apparent() const { return app_; }      ///< the measured apparent angle (rad)
  SField speed() const { return uclS_; }        ///< the SMOOTHED U_cl
  SField capillary() const { return ca_; }      ///< Ca_cl
  SField stateField() const { return stateD_; } ///< `VofDynamicState`, as a double
  SField cellVel(int d) const { return uc_[d]; }  ///< cell-centre velocity (the solver fills it)

  void allocate(long len) {
    if (base_.extent(0) == (std::size_t)len)
      return;
    base_ = SField("vof::dyn::base", len);
    imposed_ = SField("vof::dyn::imposed", len);
    app_ = SField("vof::dyn::apparent", len);
    ucl_ = SField("vof::dyn::uclRaw", len);
    uclS_ = SField("vof::dyn::ucl", len);
    wgt_ = SField("vof::dyn::valid", len);
    ca_ = SField("vof::dyn::ca", len);
    stateD_ = SField("vof::dyn::state", len);
    for (int d = 0; d < 3; ++d) {
      uc_[d] = SField("vof::dyn::cellVel", len);
      th_[d] = SField("vof::dyn::tangent", len);
    }
  }

  /// Pass A — the measurement. For every SOLID band cell at ghost depth <= 2: the wall normal, the
  /// walk to the anchor fluid cell (the SAME walk `solidBandFillPassWetting` makes, including the
  /// mixed-neighbour average where the anchor is pure phase, so the two passes agree cell for cell
  /// on which cells are contact cells), the apparent angle, the in-wall direction and the RAW
  /// contact-line speed.
  void measure(const WyAdvector& adv) {
    const I3 e = adv.extent(), n = adv.inner();
    const int g = adv.ghost();
    const long sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
    SField c = adv.colour(), sdf = adv.wallSdf();
    SField mx = adv.wettingNormal(0), my = adv.wettingNormal(1), mz = adv.wettingNormal(2);
    SField ux = uc_[0], uy = uc_[1], uz = uc_[2];
    SField ucl = ucl_, wg = wgt_, ap = app_, tx = th_[0], ty = th_[1], tz = th_[2];
    UCField kk = adv.cellKind();
    const double pe = pureEps, te = tanEps;
    Kokkos::parallel_for(
        "vof::dyn::measure",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          ucl(i) = 0.0;
          wg(i) = 0.0;
          ap(i) = 0.0;
          tx(i) = ty(i) = tz(i) = 0.0;
          if (vofGhostDepth(x, y, z, g, n.x, n.y, n.z) > 2)
            return;
          if (kk(i) != kVofSolid)
            return;
          double nw[3] = {0.5 * (sdf(i + sx) - sdf(i - sx)), 0.5 * (sdf(i + sy) - sdf(i - sy)),
                          0.5 * (sdf(i + sz) - sdf(i - sz))};
          const double nn = Kokkos::sqrt(nw[0] * nw[0] + nw[1] * nw[1] + nw[2] * nw[2]);
          if (!(nn > 1e-12))
            return;
          for (int d = 0; d < 3; ++d)
            nw[d] /= nn;
          for (int step = 1; step <= 4; ++step) {
            const int fx = x + static_cast<int>(Kokkos::round(step * nw[0]));
            const int fy = y + static_cast<int>(Kokkos::round(step * nw[1]));
            const int fz = z + static_cast<int>(Kokkos::round(step * nw[2]));
            if (fx < 0 || fy < 0 || fz < 0 || fx >= e.x || fy >= e.y || fz >= e.z)
              return;
            const long fi = L3(fx, fy, fz, e);
            if (kk(fi) != kVofFluid)
              continue;
            double mAcc[3] = {0.0, 0.0, 0.0}, uAcc[3] = {0.0, 0.0, 0.0};
            int cnt = 0;
            const double cf = c(fi);
            const double ml = Kokkos::fabs(mx(fi)) + Kokkos::fabs(my(fi)) + Kokkos::fabs(mz(fi));
            if (cf > pe && cf < 1.0 - pe && ml > 0.0) {
              mAcc[0] = mx(fi);
              mAcc[1] = my(fi);
              mAcc[2] = mz(fi);
              uAcc[0] = ux(fi);
              uAcc[1] = uy(fi);
              uAcc[2] = uz(fi);
              cnt = 1;
            } else {
              // The anchor column carries no interface: average over its MIXED fluid neighbours,
              // the same set `solidBandFillPassWetting`'s neighbour branch averages over.
              const bool inner = fx >= 1 && fy >= 1 && fz >= 1 && fx + 1 < e.x && fy + 1 < e.y &&
                                 fz + 1 < e.z;
              if (inner)
                for (int kz = -1; kz <= 1; ++kz)
                  for (int ky = -1; ky <= 1; ++ky)
                    for (int kx = -1; kx <= 1; ++kx) {
                      const long gi = L3(fx + kx, fy + ky, fz + kz, e);
                      if (kk(gi) != kVofFluid)
                        continue;
                      const double cg = c(gi);
                      if (cg <= pe || cg >= 1.0 - pe)
                        continue;
                      const double gm[3] = {mx(gi), my(gi), mz(gi)};
                      const double gl =
                          Kokkos::fabs(gm[0]) + Kokkos::fabs(gm[1]) + Kokkos::fabs(gm[2]);
                      if (!(gl > 0.0))
                        continue;
                      const double g2 =
                          Kokkos::sqrt(gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]);
                      for (int d = 0; d < 3; ++d)
                        mAcc[d] += gm[d] / g2;
                      uAcc[0] += ux(gi);
                      uAcc[1] += uy(gi);
                      uAcc[2] += uz(gi);
                      ++cnt;
                    }
            }
            if (cnt == 0)
              return;
            const double inv = 1.0 / static_cast<double>(cnt);
            const double uu[3] = {uAcc[0] * inv, uAcc[1] * inv, uAcc[2] * inv};
            double that[3], cosApp;
            const bool ok = vofWallTangent(mAcc, nw, te, that, cosApp);
            ap(i) = Kokkos::acos(cosApp < -1.0 ? -1.0 : (cosApp > 1.0 ? 1.0 : cosApp));
            wg(i) = 1.0;
            if (ok) {
              tx(i) = that[0];
              ty(i) = that[1];
              tz(i) = that[2];
              ucl(i) = vofContactLineSpeed(uu, that);
            }
            return;
          }
        });
    Kokkos::fence();
  }

  /// Pass B — the 3-point in-wall smoothing of `U_cl`, the selector and the write of theta into
  /// `adv.contactAngle()`. Call AFTER `measure()` and after the caller has exchanged `uclRaw()`
  /// and `valid()`.
  void impose(const WyAdvector& adv) {
    const I3 e = adv.extent(), n = adv.inner();
    const int g = adv.ghost();
    SField th = adv.contactAngle();
    SField bs = base_, ap = app_, ucl = ucl_, wg = wgt_, us = uclS_, ca = ca_, st = stateD_;
    SField tx = th_[0], ty = th_[1], tz = th_[2], imp = imposed_;
    UCField kk = adv.cellKind();
    const bool hyst = hysteresis, sm = smooth;
    const double ta = thetaA, tr = thetaR, mu = muLiquid, sg = sigma, lr = logRatio();
    const double tmin = thetaMin, tmax = thetaMax;
    Kokkos::parallel_for(
        "vof::dyn::impose",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          us(i) = 0.0;
          ca(i) = 0.0;
          st(i) = static_cast<double>(kVofDynNone);
          imp(i) = bs(i);
          if (vofGhostDepth(x, y, z, g, n.x, n.y, n.z) > 2)
            return;
          if (kk(i) != kVofSolid) {
            th(i) = bs(i);
            return;
          }
          if (!(wg(i) > 0.0)) {
            th(i) = bs(i);  // no contact-line data: the static base stands
            return;
          }
          double su = ucl(i), sw = 1.0;
          const double td[3] = {tx(i), ty(i), tz(i)};
          if (sm && (td[0] != 0.0 || td[1] != 0.0 || td[2] != 0.0)) {
            for (int s = -1; s <= 1; s += 2) {
              const int dx = static_cast<int>(Kokkos::round(s * td[0]));
              const int dy = static_cast<int>(Kokkos::round(s * td[1]));
              const int dz = static_cast<int>(Kokkos::round(s * td[2]));
              if (dx == 0 && dy == 0 && dz == 0)
                continue;
              const int px = x + dx, py = y + dy, pz = z + dz;
              if (px < 0 || py < 0 || pz < 0 || px >= e.x || py >= e.y || pz >= e.z)
                continue;
              const long j = L3(px, py, pz, e);
              if (!(wg(j) > 0.0))
                continue;
              su += ucl(j);
              sw += 1.0;
            }
          }
          const double u = su / sw;
          double thOut, caOut;
          const int state = vofDynamicContactAngle(ap(i), u, mu, sg, lr, hyst, ta, tr, bs(i), tmin,
                                                   tmax, thOut, caOut);
          us(i) = u;
          ca(i) = caOut;
          st(i) = static_cast<double>(state);
          imp(i) = thOut;
          th(i) = thOut;
        });
    Kokkos::fence();
  }

  SField uclRaw() const { return ucl_; }
  SField valid() const { return wgt_; }

  /// Census over the INNER region: cells per state, mean imposed / apparent angle (degrees) over
  /// the contact cells, and `max |Ca_cl|`.
  struct Census {
    long cells[kVofDynStateCount] = {0, 0, 0, 0, 0};
    double meanImposedDeg = 0.0, meanApparentDeg = 0.0, maxCa = 0.0, maxSpeed = 0.0;
    long contactCells = 0;
  };
  Census census(const WyAdvector& adv) const {
    Census r;
    if (!allocated())
      return r;
    const I3 e = adv.extent(), n = adv.inner();
    const int g = adv.ghost();
    SField wg = wgt_, imp = imposed_, ap = app_, ca = ca_, st = stateD_, us = uclS_;
    UCField kk = adv.cellKind();
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    for (int s = 0; s < kVofDynStateCount; ++s) {
      long acc = 0;
      const double sv = static_cast<double>(s);
      Kokkos::parallel_reduce(
          "vof::dyn::census_state", pol,
          KOKKOS_LAMBDA(int x, int y, int z, long& a) {
            const long i = L3(x, y, z, e);
            if (kk(i) == kVofSolid && wg(i) > 0.0 && st(i) == sv)
              ++a;
          },
          acc);
      r.cells[s] = acc;
    }
    double si = 0.0, sa = 0.0, mc = 0.0, ms = 0.0;
    long cnt = 0;
    Kokkos::parallel_reduce(
        "vof::dyn::census_mean", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& ai, double& aa, long& ac) {
          const long i = L3(x, y, z, e);
          if (kk(i) == kVofSolid && wg(i) > 0.0) {
            ai += imp(i);
            aa += ap(i);
            ++ac;
          }
        },
        si, sa, cnt);
    Kokkos::parallel_reduce(
        "vof::dyn::census_maxca", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& am) {
          const long i = L3(x, y, z, e);
          if (kk(i) == kVofSolid && wg(i) > 0.0) {
            const double v = Kokkos::fabs(ca(i));
            if (v > am)
              am = v;
          }
        },
        Kokkos::Max<double>(mc));
    Kokkos::parallel_reduce(
        "vof::dyn::census_maxu", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& as) {
          const long i = L3(x, y, z, e);
          if (kk(i) == kVofSolid && wg(i) > 0.0) {
            const double w = Kokkos::fabs(us(i));
            if (w > as)
              as = w;
          }
        },
        Kokkos::Max<double>(ms));
    Kokkos::fence();
    const double toDeg = 180.0 / 3.14159265358979323846;
    r.contactCells = cnt;
    r.meanImposedDeg = cnt ? si / static_cast<double>(cnt) * toDeg : 0.0;
    r.meanApparentDeg = cnt ? sa / static_cast<double>(cnt) * toDeg : 0.0;
    r.maxCa = mc;
    r.maxSpeed = ms;
    return r;
  }

 private:
  SField base_, imposed_, app_, ucl_, uclS_, wgt_, ca_, stateD_;
  SField uc_[3], th_[3];
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_WETTING_DYNAMIC_HPP
