/// @file
/// @brief flow — VoF rung V2b (WO-K): momentum-consistent transport of `rho^c u_c` on the
/// half-shifted (MAC) momentum control volumes, driven by the SAME geometric fluxes as the colour
/// advection of the same step.
///
/// ## Why
///
/// With mass and momentum advected by *different* fluxes, a mixed cell multiplies the gas
/// acceleration by the liquid density — a spurious interfacial momentum source of order `d(rho)`.
/// Rudman (1998) and Arrufat et al. (*Computers & Fluids* 215:104785, 2021) show the naive scheme
/// breaks down around ratio 1000 unless the resolution is absurd; with consistency a raindrop at
/// ratio 831.8 is accurate within 15 % at **15 cells/diameter** instead of ~200.
///
/// ## Indexing
///
/// The momentum control volume of component `e` is the pressure cell shifted half a cell in `e`.
/// Everything here is indexed in **`flow`'s own low-face convention**: `CV_e(i)` is the control
/// volume centred on the `-e` face of cell `i`, spanning `[x_{i-1}, x_i]` (the `+e` half of cell
/// `i-1` plus the `-e` half of cell `i`), which is exactly where the solver's velocity unknown
/// `u_e(i)` lives. Three things follow and all three are load-bearing:
/// - the owned control volumes are exactly the block's INNER region, so `C^e` and `rho^e u_e` are
///   ordinary cell fields on the advector's `g = 3` block and `WyAdvector::exchange` carries their
///   ghosts unchanged (a `+e`-shifted ownership would have put an owned value inside the halo's
///   ghost band, where the exchange would overwrite it);
/// - the bridge back to the solver is a plain `copyInner`, with no shift to get wrong;
/// - `WyAdvector`'s own face velocity `uf_e(i)` sits on the `+e` face of cell `i`
///   (`colour_field.hpp`), so the velocity AT `CV_e(i)` is `uf_e(i - s_e)`.
///
/// ## The construction
///
/// 1. **`C^e` by clipping, never by interpolation.** `C^e(i) = slab(cell i-s_e, e, [1/2, 1]) +
///    slab(cell i, e, [0, 1/2])`, each term the PLIC polyhedron of that pressure cell truncated to
///    the half cell (`plicSlabVolume` — a rescale, not a clipper). Interpolating `C` destroys
///    sharpness and is the classic error in this rung.
/// 2. **The same fluxes.** For a sweep in direction `d` the flux slab of the shifted CV is
///    - `d == e`: a slab of the SINGLE pressure cell whose centre the CV face sits on (`|a| < 1/2`
///      keeps it inside that cell's half);
///    - `d != e`: the union of two HALF cells — `[1/2,1]` of cell `p-s_e` and `[0,1/2]` of cell
///      `p`, each `|a|` deep in `d`. Two axes at once, which is why `plicBoxVolume` exists.
///    Both read the planes `WyAdvector` reconstructed for THIS sweep, between its `computeFluxes`
///    and its `applySweep` — the planes are overwritten every sweep, so the momentum sweeps are
///    interleaved with the colour sweeps rather than run after `advect()` returns.
///    The face Courant number of the shifted CV is the `e`-average of the two staggered faces;
///    summed over the three sweeps that is exactly `1/2 (div_{p-s_e} + div_p)`, so the dilation
///    term vanishes with the projection's own divergence residual — the colour's conservation
///    floor.
/// 3. **Weymouth's flux bounds, on the shifted volume's OWN colour.** The geometric flux above is
///    bounded by the liquid content of the donor CV *as the current cell planes see it*, which is
///    not the same as the ADVECTED `C^e` once the second sweep re-reconstructs from an updated `C`.
///    The gap is `O(a^2)` and does not accumulate — measured on a tilted plane at CFL 0.2/0.1/0.05/
///    0.025: `C^e` reached `-2.6e-2 / -6.0e-3 / -1.0e-3 / -3.1e-5` — but at density ratio `10^4` a
///    `-2.6e-2` undershoot drives `rho^e` to `-255`, and the recovery divides by it. The flux is
///    therefore clamped into Weymouth's own admissible interval
///    `max(0, |a| - (1 - C^e_don)) <= |F| <= min(|a|, C^e_don)` (thesis Appendix A), which is
///    exactly the hypothesis of his boundedness proof and restores `0 <= C^e <= 1`. The clamp is
///    inactive wherever the geometry is self-consistent (`clampedFluxes()` counts it), it is
///    applied to the ONE face value both neighbours share so conservation still telescopes
///    bit-exactly, and it is applied BEFORE the momentum flux is formed so the two updates keep
///    sharing one `F` — the consistency identity is untouched. For `C^e_don` exactly 0 or 1 the
///    interval collapses to the algebraic value, so full and empty control volumes stay exactly
///    stationary.
/// 4. **One frozen flag for the pair.** `flag^e = H(C^{e,n} - 1/2)`, frozen once per step from the
///    freshly built `C^e` and used unchanged by all three sweeps in BOTH the `C^e` update and the
///    `rho^e u_e` update. That shared flag is what makes the two updates telescope identically.
///    (The work order says "the same frozen dilation flag as the C advection". The flag that has to
///    be shared is the one of the pair that must telescope, and both members of that pair live on
///    the SHIFTED CV, so the structural analogue `H(C^e - 1/2)` is what is shared —
///    `WyAdvector::dilationFlag()`, the pressure-cell flag, belongs to the pressure-cell pair.
///    `useCellDilationFlag` switches to the literal reading as a measured ablation.)
/// 5. **Recovery** `u_e = (rho^e u_e) / max(rho^e, floor)`; see `rhoFloor`.
///
/// ## The consistency identity (why the uniform-velocity gate is exact)
///
/// Per sweep, with `Psi = rho_g a + (rho_l - rho_g) F` the mass flux and `rho^` the frozen-flag
/// density, the conservative pair is
///
///     C^e_i   += dC = (F_- - F_+) + flag_i (a_+ - a_-)
///     M_i     += (Psi_- u^_- - Psi_+ u^_+) + rho^_i u_i (a_+ - a_-)
///
/// with `rho^e = rho_g (1 - C^e) + rho_l C^e`, so `d(rho^e) = drho dC` — the SAME `dC`. If
/// `u_e == U` everywhere then every `u^` and every `u_i` is `U` and the two updates differ by
/// exactly the factor `U`: `dM = U d(rho^e)`, so `M = U rho^e` is preserved and `u = M/rho^e = U`
/// at ANY density ratio, with no tuning knob. An inconsistent scheme is wrong at `O(d rho)`.
///
/// The identity holds whatever the fluxes are, *provided the two updates use the same ones*. That
/// is the whole point of driving both from one set of planes, and it is why the flux clamp of
/// point 3 is free: it changes `F`, not the fact that there is one `F`. `momentumUpdate` evolves
/// the algebraically identical `u_new = u_old + dev/rho_new` form so that the identity survives in
/// FLOATING POINT too, not only in exact arithmetic — the note there explains why that matters.
///
/// ## The transported velocity in the flux
///
/// ## Cut cells (rung V5a, WO-Q item 8)
///
/// With an immersed solid the half-shifted control volume gets a fluid volume and open face areas
/// of its own, by the same half-cell averaging that defines it:
/// - fluid volume `eps^e(i) = 1/2 (eps(i - s_e) + eps(i))`;
/// - transverse face (`d != e`) openness `1/2 (o_d(i - s_e) + o_d(i))` — the average is what makes
///   the dilation sum reproduce `1/2 (div_open(i - s_e) + div_open(i))`, i.e. the same
///   half-and-half of the projection's own constraint the uncut construction relies on;
/// - AXIAL face (`d == e`) openness `eps(p)`, the CELL fraction of the pressure cell whose CENTRE
///   the face sits on — the axial CV face IS that cell-centre plane, and its fluid area fraction is
///   the cell's fluid volume fraction to the same (uniform-solid-in-cell) accuracy as everything
///   else here.
/// The liquid content of the CV is `eps(i - s_e) slabHi(i - s_e) + eps(i) slabLo(i)` and
/// `C^e = that / eps^e`: each half cell's PLIC slab fraction weighted by ITS OWN fluid volume,
/// which is the same "spread the solid uniformly over the cell" approximation the colour flux
/// makes, and which returns `C^e = 1` exactly for a uniformly liquid field whatever the geometry.
/// Both updates are then done in fluid-volume units (divide by `eps^e`), exactly as the colour is.
///
/// **The consistency identity survives all of it verbatim**, and that is the point: every term of
/// `dev` is a DIFFERENCE OF VELOCITIES, so a uniform field gives `dev = 0` bit for bit whatever
/// `eps^e` and the openness are, and `u_new = u_old` exactly — in cut cells too, not only away from
/// them. Measured on a sphere packing at ratio 1e3: 0.0.
///
/// `u^` is the CURRENT transported velocity of the donor CV (plain donor-cell upwind by default;
/// `momentumMuscl` adds a MinMod-limited slope to the flux slab's centroid, and the note on that
/// flag says why it is NOT the default at high density ratio). With a uniform field `u^ = U` bit
/// for bit either way, so the gate above is untouched. The velocity is exchanged between sweeps
/// alongside `C^e` — fluxing the FROZEN `u^n` instead would be one exchange cheaper but leaves a
/// term with gain `rho_l/rho_g` on any control volume a sweep empties (see the derivation in
/// `momentumUpdate`), which is not something to carry into a high-ratio rung to save a halo
/// exchange.
#ifndef PECLET_FLOW_VOF_MOMENTUM_ADVECT_HPP
#define PECLET_FLOW_VOF_MOMENTUM_ADVECT_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "vof/advect_wy.hpp"
#include "vof/cutcell.hpp"

namespace peclet::flow::vof {

/// Two-phase density from a fraction, written so that `C = 1` returns `rho_l` and `C = 0` returns
/// `rho_g` EXACTLY (a plain `rho_g + (rho_l - rho_g) C` does not). That exactness is what makes the
/// single-phase reduction density-ratio-independent to round-off.
KOKKOS_INLINE_FUNCTION double vofPhaseRho(double rhoG, double rhoL, double c) {
  return rhoG * (1.0 - c) + rhoL * c;
}

KOKKOS_INLINE_FUNCTION double vofMinmod(double a, double b) {
  if (!(a * b > 0.0))
    return 0.0;
  return Kokkos::fabs(a) < Kokkos::fabs(b) ? a : b;
}

/// Liquid volume of the PLIC polyhedron of cell `q` inside a box given as per-axis intervals,
/// as a fraction of a whole cell. Non-mixed cells take the algebraic branch `C * volume` — the same
/// contract `wyFaceFlux` uses, which is what keeps full/empty cells exactly stationary.
KOKKOS_INLINE_FUNCTION double vofCellBox(const SField& c, const SField& mx, const SField& my,
                                         const SField& mz, const SField& al, long q,
                                         const double lo[3], const double hi[3],
                                         double eps = 0.0) {
  const double cq = c(q);
  const double vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  // WO-R2 item 4: the SAME wisp tolerance the colour reconstruction used, or this reads a plane
  // the reconstruction pass never wrote for a wisp cell.
  if (!wyIsMixed(cq, eps))
    return cq * vol;
  return plicBoxVolume(mx(q), my(q), mz(q), al(q), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
}

/// Momentum-consistent companion to `WyAdvector`. Owns the half-shifted colour `C^e`, the conserved
/// momentum `rho^e u_e` and the recovered advected velocity for the three components, all on the
/// advector's own `g = 3` block. Drives the step itself (see the class note in `advect_wy.hpp`).
class MomentumConsistentAdvector {
 public:
  struct Diagnostics {
    double minCc[3] = {0, 0, 0};  ///< min C^e over inner momentum CVs
    double maxCc[3] = {0, 0, 0};  ///< max C^e over inner momentum CVs
    double minRhoC = 0.0;         ///< min rho^e before the floor
    long floored = 0;             ///< inner CVs where the rho^e floor bit in the last recovery
    long clamped = 0;             ///< fluxes Weymouth's admissible interval had to clamp, last step
    double sumM[3] = {0, 0, 0};   ///< sum of rho^e u_e over inner CVs AFTER the last advection
    double sumM0[3] = {0, 0, 0};  ///< the same sum at the START of the last advection (seeding)
  };

  /// @param w      the colour advector whose block, planes and face velocities are shared
  /// @param rhoG   the density of the C = 0 phase
  /// @param rhoL   the density of the C = 1 phase
  void init(const WyAdvector& w, double rhoG, double rhoL) {
    if (!(rhoG > 0.0) || !(rhoL > 0.0))
      throw std::invalid_argument(
          "peclet::flow::vof::MomentumConsistentAdvector: both phase densities must be > 0");
    n_ = w.inner();
    e_ = w.extent();
    g_ = w.ghost();
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    rhoG_ = rhoG;
    rhoL_ = rhoL;
    for (int c = 0; c < 3; ++c) {
      cc_[c] = SField("vof::Cc", len_);
      vel_[c] = SField("vof::uAdv", len_);
      flag_[c] = UCField("vof::ccFlag", len_);
    }
    fluxC_ = SField("vof::momFluxC", len_);
    fluxU_ = SField("vof::momFluxU", len_);
    aFace_ = SField("vof::momAFace", len_);
  }

  bool initialized() const { return len_ > 0; }
  void setPhaseDensities(double rhoG, double rhoL) {
    rhoG_ = rhoG;
    rhoL_ = rhoL;
  }
  /// The advected velocity of component `e`, indexed exactly like the solver's `u_e` (inner cells).
  SField advectedVelocity(int e) const { return vel_[e]; }
  /// The half-shifted colour of component `e`, same indexing.
  SField shiftedColour(int e) const { return cc_[e]; }

  /// Floor on `rho^e` in the recovery divide, in the same units as the phase densities. Default 0
  /// means "use `rhoFloorFrac * min(rho_g, rho_l)`".
  double rhoFloor = 0.0;
  double rhoFloorFrac = 1e-6;
  /// Opt-in MinMod-limited linear reconstruction of the donor velocity in the momentum flux,
  /// instead of the DEFAULT plain donor-cell (first-order upwind) value.
  ///
  /// **Off by default, and the reason is measured, not stylistic.** The velocity update is
  /// `u_new = u_old + [rho_g (a_- dv_- - a_+ dv_+) + drho (F_- dv_- - F_+ dv_+)]/rho_new` with
  /// `dv = u^ - u_old`. On a control volume whose OWN outgoing flux is the donor (`a_+ > 0`), plain
  /// upwind gives `u^_+ = u_old` and hence `dv_+ = 0` EXACTLY. A slope makes
  /// `dv_+ = (1/2 - |a|/2) * slope`, and its coefficient `drho F_+ / rho_new` is unbounded in the
  /// density ratio precisely when the sweep empties the volume (`F_+ -> C^e`, `rho_new -> rho_g`) —
  /// the volume's residual velocity is then a difference of two liquid-scale momenta. Measured on
  /// the uniform-velocity gate at ratio 1e4, 50 steps, in a double-storage build (so the solver's
  /// own float floor is out of the picture): with the slope the error grows
  /// `3.3e-16 -> 6.1e-16 -> 4.3e-14 -> 5.1e-13 -> 2.2e-10` over steps 10..50; without it, it is
  /// FLAT at `3.3e-16 ... 6.7e-16`. At ratio 1e3 the slope is harmless (6.1e-15 at 50 steps), so
  /// this is a high-ratio robustness choice, not an accuracy verdict — turn it on deliberately and
  /// re-run the ratio sweep if you do.
  bool momentumMuscl = false;
  /// Ablation: use the PRESSURE-cell frozen dilation flag `H(C^n - 1/2)` on the shifted CV instead
  /// of the structural analogue `H(C^{e,n} - 1/2)`. See the file header, point 4.
  bool useCellDilationFlag = false;
  /// Ablation: drop the Weymouth flux clamp of point 3 (the raw geometric flux). Boundedness of
  /// `C^e` is then only `O(a^2)`, which at high density ratio takes `rho^e` negative — this switch
  /// exists so that statement stays a measured number.
  bool clampFluxes = true;

  double phaseRhoG() const { return rhoG_; }
  double phaseRhoL() const { return rhoL_; }
  double lastRhoFloor() const { return floorUsed_; }
  long clampedFluxes() const { return clamped_; }

  /// One momentum-consistent Weymouth-Yue step: advances BOTH the colour field inside `w` and the
  /// half-shifted `rho^e u_e`, sharing the planes, the sweep order and the frozen flag.
  ///
  /// Replaces `WyAdvector::advect(dt, step)` — do not call both.
  void advect(WyAdvector& w, double dt, long step) {
    if (!initialized())
      throw std::runtime_error(
          "peclet::flow::vof::MomentumConsistentAdvector: init() was never called");
    if (!(w.cflLimit < 0.5))
      throw std::runtime_error(
          "peclet::flow::vof::MomentumConsistentAdvector: the shifted control volume needs "
          "|u| dt/h < 1/2 strictly (its flux slab lives inside ONE half cell), so the "
          "Weymouth-Yue cap must be < 0.5 - the default 0.25 is the proven 3D bound");
    w.requireExchange();
    w.checkCourant(dt);  // same guard, same lastCfl_, as WyAdvector::advect
    const double dth = dt / w.h();
    clamped_ = 0;
    floored_ = 0;

    w.freezeDilationFlag();  // the pressure-cell flag, for the colour update
    w.reconstruct();         // planes of C^n: what C^e is clipped out of
    if (w.hasGeometry())
      buildShiftedColourCut(w);
    else
      buildShiftedColour(w);
    freezeShiftedState(w);

    const int* perm = kWySweepPerm[static_cast<int>(step % 6)];
    for (int s = 0; s < 3; ++s) {
      const int d = perm[s];
      // The flux at the `-d` face of the first owned CV reads the donor one cell below, so C^e
      // needs a valid ghost ring for the clamp of point 3. C^e is an ordinary cell field on this
      // block (see the Indexing note), so the colour field's own exchange carries it.
      for (int e = 0; e < 3; ++e) {
        w.exchange(cc_[e]);
        w.exchange(vel_[e]);
      }
      w.reconstruct();  // idempotent for s = 0; re-reconstructs after each applied sweep
      w.computeFluxes(d, dth);
      for (int e = 0; e < 3; ++e) {
        if (w.hasGeometry()) {  // rung V5a: openness-weighted, in fluid-volume units
          momentumFluxesCut(w, d, e, dth);
          momentumUpdateCut(w, d, e);
        } else {
          momentumFluxes(w, d, e, dth);
          momentumUpdate(w, d, e);
        }
      }
      w.applySweep(d, dth);
      w.exchange(w.colour());
    }
  }

  Diagnostics diagnostics() const {
    Diagnostics dg;
    const I3 e = e_, n = n_;
    const int g = g_;
    const double rg = rhoG_, rl = rhoL_;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    double gmin = 1e300;
    for (int c = 0; c < 3; ++c) {
      SField cf = cc_[c], mf = vel_[c];
      double mn = 0.0, mx = 0.0, sm = 0.0, rmin = 0.0;
      Kokkos::parallel_reduce(
          "vof::mom::min", pol,
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            acc = Kokkos::fmin(acc, cf(L3(x, y, z, e)));
          },
          Kokkos::Min<double>(mn));
      Kokkos::parallel_reduce(
          "vof::mom::max", pol,
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            acc = Kokkos::fmax(acc, cf(L3(x, y, z, e)));
          },
          Kokkos::Max<double>(mx));
      Kokkos::parallel_reduce(
          "vof::mom::sum", pol,
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = L3(x, y, z, e);
            acc += vofPhaseRho(rg, rl, cf(i)) * mf(i);
          },
          sm);
      Kokkos::parallel_reduce(
          "vof::mom::rhomin", pol,
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            acc = Kokkos::fmin(acc, vofPhaseRho(rg, rl, cf(L3(x, y, z, e))));
          },
          Kokkos::Min<double>(rmin));
      Kokkos::fence();
      dg.minCc[c] = mn;
      dg.maxCc[c] = mx;
      dg.sumM[c] = sm;
      gmin = gmin < rmin ? gmin : rmin;
    }
    dg.minRhoC = gmin;
    dg.floored = floored_;
    dg.clamped = clamped_;
    for (int c = 0; c < 3; ++c)
      dg.sumM0[c] = seedM_[c];
    return dg;
  }

 public:
  // ---- implementation detail (public for the same nvcc reason as in WyAdvector) ----------------

  /// `C^e(i) = slab(cell i-s_e, e, [1/2,1]) + slab(cell i, e, [0,1/2])`, over the inner region.
  void buildShiftedColour(const WyAdvector& w) {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField c = w.colour(), mx = w.planeM(0), my = w.planeM(1), mz = w.planeM(2),
           al = w.planeAlpha();
    const double weps = w.wispEps;  // WO-R2 item 4: the colour reconstruction's own tolerance
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    for (int comp = 0; comp < 3; ++comp) {
      const long se = strideOf(comp);
      const int ec = comp;
      SField out = cc_[comp];
      Kokkos::parallel_for(
          "vof::mom::shifted_colour", MD(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
            lo[ec] = 0.5;
            hi[ec] = 1.0;
            const double hiHalf = vofCellBox(c, mx, my, mz, al, i - se, lo, hi, weps);
            lo[ec] = 0.0;
            hi[ec] = 0.5;
            const double loHalf = vofCellBox(c, mx, my, mz, al, i, lo, hi, weps);
            out(i) = hiHalf + loHalf;
          });
    }
    Kokkos::fence();
  }

  /// Freeze `flag^e = H(C^{e,n} - 1/2)` and seed `M = rho^e u^n_e`.
  void freezeShiftedState(const WyAdvector& w) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const bool cellFlag = useCellDilationFlag;
    UCField ccFlag = w.dilationFlag();
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    const double rg = rhoG_, rl = rhoL_;
    for (int comp = 0; comp < 3; ++comp) {
      const long se = strideOf(comp);
      SField cf = cc_[comp], mf = vel_[comp], uf = w.faceVel(comp);
      UCField fl = flag_[comp];
      Kokkos::parallel_for(
          "vof::mom::freeze", MD(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            fl(i) = cellFlag ? ccFlag(i) : (cf(i) > 0.5 ? 1u : 0u);
            mf(i) = uf(i - se);  // uf_e(i-s_e) IS the velocity at control volume i
          });
      // The momentum census at the START of this step's advection, so the conservation claim can be
      // made about the ADVECTION rather than about the whole coupled step (the momentum solve and
      // the projection both change u afterwards, and they conserve `rho_f u` — the arithmetic face
      // mean — not `rho^e u`).
      double sm = 0.0;
      Kokkos::parallel_reduce(
          "vof::mom::seedsum", MD(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = L3(x, y, z, e);
            acc += vofPhaseRho(rg, rl, cf(i)) * mf(i);
          },
          sm);
      Kokkos::fence();
      seedM_[comp] = sm;
    }
    Kokkos::fence();
  }

  /// Face Courant number, geometric liquid flux and momentum flux of the shifted CV of component
  /// `comp`, for a sweep in direction `d`. Stored at the CV on the face's `-d` side, exactly as
  /// `WyAdvector::computeFluxes` stores the colour flux — computing each face ONCE is what makes
  /// the sum telescope bit-exactly, in-rank and across a rank boundary alike.
  void momentumFluxes(const WyAdvector& w, int d, int comp, double dth) {
    const I3 e = e_;
    const int g = g_;
    const long sd = strideOf(d), se = strideOf(comp);
    const int dd = d, ec = comp;
    const double rg = rhoG_, rl = rhoL_;
    const bool muscl = momentumMuscl, clamp = clampFluxes;
    SField c = w.colour(), mx = w.planeM(0), my = w.planeM(1), mz = w.planeM(2),
           al = w.planeAlpha();
    const double weps = w.wispEps;
    SField ud = w.faceVel(d), cvU = vel_[comp], cvC = cc_[comp];
    SField fc = fluxC_, fu = fluxU_, af = aFace_;
    int lo3[3] = {g, g, g};
    const int hi3[3] = {g + n_.x, g + n_.y, g + n_.z};
    lo3[d] -= 1;  // the `-d` face of the first owned CV is the `+d` face of the CV below it
    long nclamp = 0;
    Kokkos::parallel_reduce(
        "vof::mom::flux",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {lo3[0], lo3[1], lo3[2]},
                                                      {hi3[0], hi3[1], hi3[2]}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long p = L3(x, y, z, e);
          // The shifted CV's face velocity is the `comp`-average of the two staggered faces; summed
          // over the three sweeps this is 1/2 (div_{p-s_e} + div_p) — zero for a projected field.
          const double a = 0.5 * (ud(p - se) + ud(p)) * dth;
          af(p) = a;
          if (a == 0.0) {
            fc(p) = 0.0;
            fu(p) = cvU(p);  // any finite value; the update multiplies it by a == F == 0
            return;
          }
          const double aa = a > 0.0 ? a : -a;
          double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
          double F;
          if (dd == ec) {
            // The CV face sits at the CENTRE of cell p; the |a|-deep slab therefore lives inside
            // that one cell's upwind half. |a| < 1/2 keeps it there.
            if (a > 0.0) {
              lo[ec] = 0.5 - aa;
              hi[ec] = 0.5;
            } else {
              lo[ec] = 0.5;
              hi[ec] = 0.5 + aa;
            }
            F = vofCellBox(c, mx, my, mz, al, p, lo, hi, weps);
          } else {
            // Transverse sweep: the CV face is the pressure cells' own `d` face, and the donor slab
            // is the union of the `+e` half of one cell and the `-e` half of its `e` neighbour.
            long q;
            if (a > 0.0) {
              q = p;
              lo[dd] = 1.0 - aa;
              hi[dd] = 1.0;
            } else {
              q = p + sd;
              lo[dd] = 0.0;
              hi[dd] = aa;
            }
            lo[ec] = 0.5;
            hi[ec] = 1.0;
            const double hiHalf = vofCellBox(c, mx, my, mz, al, q - se, lo, hi, weps);
            lo[ec] = 0.0;
            hi[ec] = 0.5;
            const double loHalf = vofCellBox(c, mx, my, mz, al, q, lo, hi, weps);
            F = hiHalf + loHalf;
          }
          // Weymouth's admissible interval on the DONOR control volume's own colour (thesis
          // Appendix A) — the hypothesis of his boundedness proof. Collapses to the algebraic value
          // for a full or empty donor, so full/empty CVs stay exactly stationary.
          if (clamp) {
            const double cd = cvC(a > 0.0 ? p : p + sd);
            const double up = aa < cd ? aa : cd;
            const double dn = aa - (1.0 - cd);
            const double lob = dn > 0.0 ? dn : 0.0;
            const double Fc = F > up ? up : (F < lob ? lob : F);
            if (Fc != F)
              ++acc;
            F = Fc;
          }
          if (a < 0.0)
            F = -F;
          // Donor-CV velocity, MinMod-limited to the flux slab's centroid. A uniform field gives
          // both MinMod arguments exactly 0, hence u^ = u bit for bit — the uniform-velocity gate.
          const long don = a > 0.0 ? p : p + sd;
          double slope = 0.0;
          if (muscl)
            slope = vofMinmod(cvU(don) - cvU(don - sd), cvU(don + sd) - cvU(don));
          const double sgn = a > 0.0 ? 1.0 : -1.0;
          // Only the FACE quantities are stored — the liquid flux, the Courant number and the
          // donor velocity. The mass flux `rho_g a + (rho_l - rho_g) F` and the momentum flux are
          // assembled in `momentumUpdate`, in a form that cancels exactly; see the note there.
          fc(p) = F;
          fu(p) = cvU(don) + sgn * (0.5 - 0.5 * aa) * slope;
        },
        nclamp);
    Kokkos::fence();
    clamped_ += nclamp;
  }

  /// `C^e_i += (F_- - F_+) + flag (a_+ - a_-)` and `M_i += (G_- - G_+) + rho^ u^n_i (a_+ - a_-)`,
  /// with ONE frozen flag and the SAME `a` in both — see the consistency identity in the header.
  void momentumUpdate(const WyAdvector& w, int d, int comp) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sd = strideOf(d), se = strideOf(comp);
    const double rg = rhoG_, rl = rhoL_, dr = rhoL_ - rhoG_;
    const double rmin = rhoG_ < rhoL_ ? rhoG_ : rhoL_;
    const double fl0 = rhoFloor > 0.0 ? rhoFloor : rhoFloorFrac * rmin;
    floorUsed_ = fl0;
    SField cf = cc_[comp], vf = vel_[comp], uf = w.faceVel(comp);
    SField fc = fluxC_, fu = fluxU_, af = aFace_;
    UCField fl = flag_[comp];
    long nfloor = 0;
    Kokkos::parallel_reduce(
        "vof::mom::update",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long i = L3(x, y, z, e);
          const double aP = af(i), aM = af(i - sd);
          const unsigned char k = fl(i);
          const double dil = k ? (aP - aM) : 0.0;
          // The colour increment, formed ONCE and reused by the momentum below: that literal
          // sharing is what makes the uniform-velocity identity exact rather than merely true in
          // exact arithmetic.
          const double dC = (fc(i - sd) - fc(i)) + dil;
          const double cNew = cf(i) + dC;
          cf(i) = cNew;
          // The conservative update of this control volume is
          //     rho_new u_new = rho_old u_old - (Psi_+ u^_+ - Psi_- u^_-) + rho^ u_old (a_+ - a_-)
          // with `Psi = rho_g a + drho F` the mass flux and `u^` the donor reconstruction of the
          // CURRENT velocity. Stored and evolved as `rho^e u_e`, that is a sum of rho_l-scaled
          // terms whose result is only rho_g-scaled in a gas control volume, and a volume a
          // directional sweep fills and the next one empties passes through an intermediate `rho^e`
          // a factor `ratio` larger than where it starts and ends; the rounding of that excursion
          // is eps*ratio*|a| and does NOT cancel against the colour's own. Measured, it degraded
          // the uniform-velocity gate LINEARLY IN THE DENSITY RATIO (6.7e-16 at 1e2 -> 3.2e-14 at
          // 1e4) — wearing exactly the signature of the defect this rung exists to remove, for a
          // purely floating-point reason.
          //
          // Subtracting `rho_new u_old` from both sides is algebraically free (substitute
          // `rho_old = rho_new - drho dC` and watch every `u_old` term cancel) and removes it:
          //
          //     u_new = u_old + [ rho_g (a_- dv_- - a_+ dv_+) + drho (F_- dv_- - F_+ dv_+)
          //     ]/rho_new dv_+/- = u^_+/- - u_old
          //
          // Every term is a DIFFERENCE OF VELOCITIES. For a uniform field each is exactly zero (the
          // same double, subtracted), so `u_new == u_old` bit for bit at ANY density ratio and for
          // any dC — the gate is flat in the ratio by construction, not by tolerance. This is still
          // the conservative scheme, with `rho_new` the advected colour's own density; only the
          // arithmetic is conditioned. (An earlier version fluxed the FROZEN u^n instead of the
          // current velocity; the `drho dC (u^n_i - u_old)/rho_new` term that then survives has
          // gain `ratio` on a control volume a sweep empties, and it amplified the solver's own
          // 1e-7 float-storage noise to 1e-7 at ratio 1e4 while ratio 1e3 was clean — the gain, not
          // a defect in the fluxes. Transporting the current state removes the term identically and
          // is also the faithful directional split.)
          const double uo = vf(i);
          const double dvM = fu(i - sd) - uo, dvP = fu(i) - uo;
          // The DILATION coefficient is `rho^ u^n`, with `u^n` the velocity FROZEN at the start of
          // the step — the exact analogue of Weymouth-Yue freezing `H(C^n - 1/2)`, and required for
          // the same reason. Summed over the three sweeps the dilation contributes
          // `rho^ u (a_+ - a_-)` per sweep, and only a coefficient that is a CONSTANT OF THE STEP
          // factors out of that sum to leave `rho^ u * div = 0`. With the running velocity there
          // instead, the three sweeps see three different `u` and the cancellation is lost:
          // MEASURED, per-step momentum conservation degraded to 1.4e-7 relative — first order in
          // dt, not round-off — while the colour, whose flag IS frozen, stayed exact. Written as
          // `rho^ (u^n - u_old)` because the `rho^ u_old` part has already been absorbed into the
          // deviation form above; for a uniform field `u^n == u_old` and the term is exactly zero,
          // so the consistency identity is untouched.
          const double dilCoef = (k ? rl : rg) * (uf(i - se) - uo) * (aP - aM);
          const double dev =
              rg * (aM * dvM - aP * dvP) + dr * (fc(i - sd) * dvM - fc(i) * dvP) + dilCoef;
          const double rn = vofPhaseRho(rg, rl, cNew);
          if (rn < fl0)
            ++acc;
          vf(i) = uo + dev / (rn < fl0 ? fl0 : rn);
        },
        nfloor);
    Kokkos::fence();
    floored_ += nfloor;
  }

  // ---- rung V5a (WO-Q item 8): the same construction on a CUT control volume -------------------
  // Siblings of the three kernels above, reached only through `w.hasGeometry()`. The uncut bodies
  // are untouched, so a solid-free run is byte-identical by construction.

  /// `C^e = [eps(i-s_e) slabHi(i-s_e) + eps(i) slabLo(i)] / eps^e(i)` — each half cell's PLIC slab
  /// weighted by its OWN fluid volume, then divided by the CV's fluid volume. Returns exactly 1 for
  /// a uniformly liquid field whatever the geometry.
  void buildShiftedColourCut(const WyAdvector& w) {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField c = w.colour(), mx = w.planeM(0), my = w.planeM(1), mz = w.planeM(2),
           al = w.planeAlpha(), ep = w.epsFraction();
    UCField kd = w.cellKind();
    const double weps = w.wispEps;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    for (int comp = 0; comp < 3; ++comp) {
      const long se = strideOf(comp);
      const int ec = comp;
      SField out = cc_[comp];
      Kokkos::parallel_for(
          "vof::mom::shifted_colour_cut", MD(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            const double ea = vofEpsEff(ep(i - se), kd(i - se) == kVofSolid);
            const double eb = vofEpsEff(ep(i), kd(i) == kVofSolid);
            const double ecv = 0.5 * (ea + eb);
            if (!(ecv > 0.0)) {
              out(i) = 0.0;  // the whole control volume is inside the solid
              return;
            }
            double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
            lo[ec] = 0.5;
            hi[ec] = 1.0;
            const double hiHalf = vofCellBox(c, mx, my, mz, al, i - se, lo, hi, weps);
            lo[ec] = 0.0;
            hi[ec] = 0.5;
            const double loHalf = vofCellBox(c, mx, my, mz, al, i, lo, hi, weps);
            out(i) = (ea * hiHalf + eb * loHalf) / ecv;
          });
    }
    Kokkos::fence();
  }

  /// The CV's fluid volume `eps^e(i)` (0 when the whole CV is solid).
  KOKKOS_INLINE_FUNCTION static double cvEps(const SField& ep, const UCField& kd, long i, long se) {
    return 0.5 *
           (vofEpsEff(ep(i - se), kd(i - se) == kVofSolid) + vofEpsEff(ep(i), kd(i) == kVofSolid));
  }

  /// Face Courant number, geometric liquid flux and momentum flux on a CUT shifted CV. `af(p)` is
  /// stored ALREADY openness-weighted (it is the `o a` the dilation term needs); the slab thickness
  /// used for the PLIC clipping is the UNWEIGHTED `|a|`, because that is a geometric length.
  void momentumFluxesCut(const WyAdvector& w, int d, int comp, double dth) {
    const I3 e = e_;
    const int g = g_;
    const long sd = strideOf(d), se = strideOf(comp);
    const int dd = d, ec = comp;
    const bool muscl = momentumMuscl, clamp = clampFluxes;
    SField c = w.colour(), mx = w.planeM(0), my = w.planeM(1), mz = w.planeM(2),
           al = w.planeAlpha(), ep = w.epsFraction(), od = w.faceOpenness(d);
    UCField kd = w.cellKind();
    const double weps = w.wispEps;
    SField ud = w.faceVel(d), cvU = vel_[comp], cvC = cc_[comp];
    SField fc = fluxC_, fu = fluxU_, af = aFace_;
    int lo3[3] = {g, g, g};
    const int hi3[3] = {g + n_.x, g + n_.y, g + n_.z};
    lo3[d] -= 1;
    long nclamp = 0;
    Kokkos::parallel_reduce(
        "vof::mom::flux_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {lo3[0], lo3[1], lo3[2]},
                                                      {hi3[0], hi3[1], hi3[2]}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long p = L3(x, y, z, e);
          // The CV's face open area: the AXIAL face is the cell-centre plane of cell p, so its
          // fluid area fraction is that cell's fluid VOLUME fraction; a transverse face is the two
          // half faces of cells p - s_e and p, so it is their average.
          const double ofac =
              (dd == ec) ? vofEpsEff(ep(p), kd(p) == kVofSolid) : 0.5 * (od(p - se) + od(p));
          const double a = 0.5 * (ud(p - se) + ud(p)) * dth;
          af(p) = ofac * a;
          if (a == 0.0 || ofac == 0.0) {
            fc(p) = 0.0;
            fu(p) = cvU(p);
            return;
          }
          const double aa = a > 0.0 ? a : -a;
          double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
          double F;
          if (dd == ec) {
            if (a > 0.0) {
              lo[ec] = 0.5 - aa;
              hi[ec] = 0.5;
            } else {
              lo[ec] = 0.5;
              hi[ec] = 0.5 + aa;
            }
            F = vofCellBox(c, mx, my, mz, al, p, lo, hi, weps);
          } else {
            long q;
            if (a > 0.0) {
              q = p;
              lo[dd] = 1.0 - aa;
              hi[dd] = 1.0;
            } else {
              q = p + sd;
              lo[dd] = 0.0;
              hi[dd] = aa;
            }
            lo[ec] = 0.5;
            hi[ec] = 1.0;
            const double ea = vofEpsEff(ep(q - se), kd(q - se) == kVofSolid);
            const double eb = vofEpsEff(ep(q), kd(q) == kVofSolid);
            const double ecv = 0.5 * (ea + eb);
            const double hiHalf = vofCellBox(c, mx, my, mz, al, q - se, lo, hi, weps);
            lo[ec] = 0.0;
            hi[ec] = 0.5;
            const double loHalf = vofCellBox(c, mx, my, mz, al, q, lo, hi, weps);
            // the same fluid-volume weighting as C^e, renormalized so a uniformly liquid field
            // gives F = |a| exactly (and hence, times `ofac`, the dilation's own `o|a|`)
            F = ecv > 0.0 ? (ea * hiHalf + eb * loHalf) / ecv : 0.0;
          }
          F *= ofac;
          if (clamp) {
            const long don = a > 0.0 ? p : p + sd;
            const double cd = cvC(don);
            const double sweep = ofac * aa;
            const double ecvDon = cvEps(ep, kd, don, se);
            const double Fc = vofCutFluxClamp(F, sweep, ecvDon, cd);
            if (Fc != F)
              ++acc;
            F = Fc;
          }
          if (a < 0.0)
            F = -F;
          const long don = a > 0.0 ? p : p + sd;
          double slope = 0.0;
          if (muscl)
            slope = vofMinmod(cvU(don) - cvU(don - sd), cvU(don + sd) - cvU(don));
          const double sgn = a > 0.0 ? 1.0 : -1.0;
          fc(p) = F;
          fu(p) = cvU(don) + sgn * (0.5 - 0.5 * aa) * slope;
        },
        nclamp);
    Kokkos::fence();
    clamped_ += nclamp;
  }

  /// The cut update: both increments divided by the CV's fluid volume, everything else identical —
  /// in particular `dev` is still a sum of velocity DIFFERENCES, so a uniform field is stationary
  /// bit for bit in cut cells too.
  void momentumUpdateCut(const WyAdvector& w, int d, int comp) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sd = strideOf(d), se = strideOf(comp);
    const double rg = rhoG_, rl = rhoL_, dr = rhoL_ - rhoG_;
    const double rmin = rhoG_ < rhoL_ ? rhoG_ : rhoL_;
    const double fl0 = rhoFloor > 0.0 ? rhoFloor : rhoFloorFrac * rmin;
    floorUsed_ = fl0;
    SField cf = cc_[comp], vf = vel_[comp], uf = w.faceVel(comp), ep = w.epsFraction();
    UCField kd = w.cellKind();
    SField fc = fluxC_, fu = fluxU_, af = aFace_;
    UCField fl = flag_[comp];
    long nfloor = 0;
    Kokkos::parallel_reduce(
        "vof::mom::update_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long i = L3(x, y, z, e);
          const double ecv = cvEps(ep, kd, i, se);
          if (!(ecv > 0.0))
            return;  // the control volume is entirely inside the solid
          const double aP = af(i), aM = af(i - sd);
          const unsigned char k = fl(i);
          const double dil = k ? (aP - aM) : 0.0;
          const double dC = ((fc(i - sd) - fc(i)) + dil) / ecv;
          const double cNew = cf(i) + dC;
          cf(i) = cNew;
          const double uo = vf(i);
          const double dvM = fu(i - sd) - uo, dvP = fu(i) - uo;
          const double dilCoef = (k ? rl : rg) * (uf(i - se) - uo) * (aP - aM);
          const double dev =
              rg * (aM * dvM - aP * dvP) + dr * (fc(i - sd) * dvM - fc(i) * dvP) + dilCoef;
          const double rn = vofPhaseRho(rg, rl, cNew);
          if (rn < fl0)
            ++acc;
          vf(i) = uo + dev / (ecv * (rn < fl0 ? fl0 : rn));
        },
        nfloor);
    Kokkos::fence();
    floored_ += nfloor;
  }

  long strideOf(int d) const {
    return d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 3;
  long len_ = 0;
  double rhoG_ = 1.0, rhoL_ = 1.0, floorUsed_ = 0.0;
  long floored_ = 0, clamped_ = 0;
  double seedM_[3] = {0, 0, 0};
  SField cc_[3], vel_[3], fluxC_, fluxU_, aFace_;
  UCField flag_[3];
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_MOMENTUM_ADVECT_HPP
