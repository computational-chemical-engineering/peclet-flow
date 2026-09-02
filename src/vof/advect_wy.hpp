/// @file
/// @brief flow — VoF rung V1: Weymouth & Yue (2010) directionally-split, exactly volume-conserving
/// geometric advection of a colour field on a structured block.
///
/// **Standalone.** This header owns its own extended field block with its own ghost width
/// (`g = 3`, `suite/docs/VOF_PLAN.md` §3 design rule 1) and knows nothing about `IbmSolver`, the
/// solver's `G = 2` machinery, MPI, or cut cells. The ghost refresh is a host-side callback
/// (`exchange`), so the same header drives the single-block periodic case in `tests/kokkos` and the
/// multi-rank `core::halo::GridHaloTopology` case in `tests/kokkos_mpi` with no code change.
///
/// ## The algorithm (Weymouth & Yue, JCP 229:2853, 2010)
///
/// Primary source used here: G. D. Weymouth, *Physics and learning based computational models for
/// breaking bow waves based on new boundary immersion approaches*, PhD thesis, MIT 2008
/// (dspace 1721.1/44754) — its §2.2.2 and Appendix A are the JCP paper's own derivation and the
/// full boundedness proof. Cross-checked against the restatement in Arrufat et al., *Computers &
/// Fluids* 215:104785 (2021), arXiv:1811.12327, §3.3.4.
///
/// Split the colour equation `dC/dt + div(u C) = C div(u)` into three one-dimensional sweeps
///
///     C^{l+1}_i = C^l_i - (F_{i+1/2} - F_{i-1/2}) + c_i (a_{i+1/2} - a_{i-1/2}) ,
///     a_{i+1/2} = uf_{i+1/2} dt / h  (the face Courant number),
///
/// where `F` is the **Eulerian donor-cell geometric flux**: the fluid volume of the upwind cell's
/// PLIC polyhedron inside the un-stretched slab of width |a| adjacent to the face (thesis Fig. 2-4;
/// `plicSlabVolume`). The interface is re-reconstructed before every sweep from the current C
/// (thesis: "to avoid the possibility of fluxing the same fluid into two different cells the
/// surface must be reconstructed after each sweep").
///
/// **The dilation coefficient is the frozen cell-centre indicator**
///
///     c_i = H(C^n_i - 1/2)     (thesis eq. A.28: 1 if C^n_i > 1/2, else 0)
///
/// evaluated ONCE per step from `C^n` and **used unchanged by all three sweeps**. This is the whole
/// method. Because `c_i` is a constant of the step and direction-independent, summing the three
/// sweeps gives
///
///     C^{n+1}_i - C^n_i = -sum_faces F + c_i (dt/h) sum_d (uf_{d+} - uf_{d-}) ,
///
/// whose flux part telescopes to zero over the domain and whose second part is `c_i` times the
/// *discrete* divergence — zero for a discretely solenoidal face field. Hence exact conservation,
/// to the accuracy with which the discrete face divergence vanishes (thesis §2.2.2 requirements
/// 1–3). Recomputing `c_i` between sweeps destroys exactly this cancellation, and the damage is
/// quiet: conservation degrades from ~1e-15 to ~1e-10 rather than failing outright.
///
/// Two further consequences of the same structure, both relied on here:
/// - a cell whose whole 1D neighbourhood is full (C = 1) is **exactly** stationary in floating
///   point: its fluxes are the algebraic `1 * a_{+/-}` and the dilation term is the negation of
///   the flux difference, and IEEE subtraction is antisymmetric — so `(a_- - a_+) + (a_+ - a_-)`
///   is an exact zero. Only interface cells accumulate round-off, which is why the conservation
///   floor scales with the interface area and not with the domain volume. To keep this, the flux
///   and the dilation term must scale the SAME `uf` by the SAME `dt/h` (see `applySweep`).
/// - empty cells (C = 0) stay exactly empty.
///
/// **Boundedness needs a CFL cap.** Thesis Appendix A bounds the flux by
/// `max(0, a - C) <= F <= min(a, C)` and shows that `c = H(C - 1/2)` is the only quadrature that
/// prevents both over-filling and over-emptying, *provided* `|a| <= 1/(2(N-1))` for N-dimensional
/// flow (thesis eq. A.33 / eq. 2.23) — 1/2 in 2D, **1/4 in 3D**. The widely-quoted `CFL < 0.5`
/// (including in this rung's work order) is the *2D* value; quoting it for a 3D solver is a
/// transcription error the WO inherited. `cflLimit` therefore defaults to the **proven 3D bound
/// 0.25** and `advect()` aborts at or above it; raise it deliberately (`cflLimit = 0.5`) for 2D
/// work or to probe the gap, never as a way to take bigger steps in 3D.
/// Note that conservation is *independent* of boundedness: the telescoping above holds whatever C
/// does, so an over-CFL run loses `0 <= C <= 1`, not volume — which is exactly why the failure is
/// easy to miss (volume still closes to round-off while C leaves [0,1]).
/// Empirically this margin is not tight — a sweep to CFL 0.48 on the LeVeque field never left
/// [0,1] — but the target regime (capillary-limited dt, `VOF_PLAN.md` §4 V4) sits far below both
/// bounds, so there is nothing to buy by defaulting past what is proven.
///
/// No clipping is applied at this rung (`VOF_PLAN.md` §4 V1): conservation must close to round-off
/// with nothing hiding the error. `diagnostics()` reports the wisp census instead.
#ifndef PECLET_FLOW_VOF_ADVECT_WY_HPP
#define PECLET_FLOW_VOF_ADVECT_WY_HPP

#include <cstdio>
#include <functional>
#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, SMem, I3, L3
#include "vof/cutcell.hpp"
#include "vof/plic.hpp"

namespace peclet::flow::vof {

using UCField = Kokkos::View<unsigned char*, SMem>;
using LField = Kokkos::View<long*, SMem>;

/// The mixed-cell predicate. Reconstruction and fluxing MUST agree on it: the flux reads
/// `(m, alpha)` exactly for the cells the reconstruction pass wrote, so the two tests are the same
/// function by construction (that is also what makes the worklist a pure optimization).
KOKKOS_INLINE_FUNCTION bool wyIsMixed(double c) {
  return c > 0.0 && c < 1.0;
}

/// The six permutations of (x, y, z), cycled by step index so no direction is systematically
/// favoured (`perm[step % 6]`).
inline constexpr int kWySweepPerm[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1},
                                           {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};

/// PLIC reconstruction of one cell: MYC normal from the 3^3 colour stencil + the analytic plane
/// offset. Container-free apart from the flat view read, so it stays a thin wrapper over
/// `plic.hpp`.
KOKKOS_INLINE_FUNCTION void wyReconstructCell(const SField& c, long i, long sy, long sz, SField mx,
                                              SField my, SField mz, SField alpha) {
  double st[27];
  for (int kk = -1; kk <= 1; ++kk)
    for (int jj = -1; jj <= 1; ++jj)
      for (int ii = -1; ii <= 1; ++ii)
        st[plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
  double m[3];
  mycNormal(st, m);
  mx(i) = m[0];
  my(i) = m[1];
  mz(i) = m[2];
  alpha(i) = plicAlpha(m[0], m[1], m[2], c(i));
}

/// Signed Eulerian donor-cell flux through the `dir`-face between cell `p` and cell `p + sd`,
/// as a fraction of a cell volume, positive along +dir. `a` is the face Courant number.
KOKKOS_INLINE_FUNCTION double wyFaceFlux(double a, long p, long sd, int dir, const SField& c,
                                         const SField& mx, const SField& my, const SField& mz,
                                         const SField& alpha) {
  if (a > 0.0) {  // donor is p; the outflow slab is the |a|-thick layer at its + face
    const double cd = c(p);
    return wyIsMixed(cd) ? plicSlabVolume(mx(p), my(p), mz(p), alpha(p), dir, 1.0 - a, 1.0)
                         : cd * a;
  }
  if (a < 0.0) {  // donor is p + sd; the outflow slab is the |a|-thick layer at its - face
    const long q = p + sd;
    const double cd = c(q), aa = -a;
    return -(wyIsMixed(cd) ? faceFluxVolume(mx(q), my(q), mz(q), alpha(q), dir, aa) : cd * aa);
  }
  return 0.0;
}

/// Weymouth-Yue split advection of a colour field on an extended (inner + ghost) block.
class WyAdvector {
 public:
  /// Per-step census. Volume/extrema are LOCAL to this block's inner region; a distributed caller
  /// reduces them itself (the advector stays MPI-free).
  struct Diagnostics {
    double sumC = 0.0;  ///< sum of C over inner cells (cell-volume units, not scaled by h^3)
    double minC = 0.0;  ///< min C over inner cells (may be < 0: no clipping at this rung)
    double maxC = 0.0;  ///< max C over inner cells (may be > 1)
    long mixed = 0;     ///< cells with 0 < C < 1
    long wisps = 0;     ///< cells with 0 < C < 1e-8 or 1-1e-8 < C < 1
    // --- rung V5a (WO-Q), all zero unless cut-cell geometry is attached -----------------------
    double volume = 0.0;      ///< sum of eps_eff*C over inner FLUID cells: the EXACTLY conserved
                              ///< functional of the cut-cell scheme (`cutcell.hpp` rule 1)
    double rawVolume = 0.0;   ///< sum of eps*C (raw buildCellFraction eps) over the same cells;
                              ///< differs from `volume` only on eps == 0 cells with an open face
    double solidFillSum = 0.0;  ///< sum of C over inner SOLID cells ON THE WORKING BLOCK, i.e. the
                                ///< neutral band fill. NOT the canonical colour: `IbmSolver` reports
                                ///< `solidSumC`, the sum over solid cells of the "C" field, which is
                                ///< 0 by construction.
    double solidSumC = 0.0;   ///< filled by the SOLVER from the canonical G=2 colour field
    double minCFluid = 0.0;   ///< min C over inner fluid cells with eps == 1 (uncut fluid)
    double maxCFluid = 0.0;   ///< max C over the same set
    double clippedVolume = 0.0;  ///< |liquid volume| the cut-cell clip moved during the last step
    double clippedSigned = 0.0;  ///< the same, signed (created positive, destroyed negative)
    long cutCells = 0;        ///< inner cells with 0 < eps_eff < 1
    long solidCells = 0;      ///< inner cells classified SOLID
    long clampedFaces = 0;    ///< faces Weymouth's admissible interval had to clamp, last step
  };

  /// @param nx,ny,nz  inner cell counts of this block
  /// @param h         uniform cell size
  /// @param ghost     ghost width; >= 2 is required (donor-ring reconstruction reads a 3^3 stencil
  ///                  centred one cell outside the inner region), 3 is the plan's colour-field
  ///                  width (height-function columns at V3 need it).
  void init(int nx, int ny, int nz, double h, int ghost = 3) {
    if (ghost < 2)
      throw std::invalid_argument("peclet::flow::vof::WyAdvector: ghost width must be >= 2");
    if (nx < 1 || ny < 1 || nz < 1)
      throw std::invalid_argument("peclet::flow::vof::WyAdvector: empty block");
    n_ = I3{nx, ny, nz};
    g_ = ghost;
    h_ = h;
    e_ = I3{nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    c_ = SField("vof::C", len_);
    mx_ = SField("vof::mx", len_);
    my_ = SField("vof::my", len_);
    mz_ = SField("vof::mz", len_);
    alpha_ = SField("vof::alpha", len_);
    flux_ = SField("vof::flux", len_);
    uf_ = SField("vof::uf", len_);
    vf_ = SField("vof::vf", len_);
    wf_ = SField("vof::wf", len_);
    cc_ = UCField("vof::cc", len_);
    listCap_ = static_cast<long>(n_.x + 2) * (n_.y + 2) * (n_.z + 2);
    list_ = LField("vof::worklist", listCap_);
  }

  // ---- geometry / storage ------------------------------------------------------------------
  I3 inner() const { return n_; }
  I3 extent() const { return e_; }
  int ghost() const { return g_; }
  double h() const { return h_; }
  long size() const { return len_; }
  /// Linear index of the inner-block cell (x,y,z), 0-based within the inner region.
  long index(int x, int y, int z) const { return L3(x + g_, y + g_, z + g_, e_); }

  /// Colour field on the extended block (x-fastest).
  SField colour() const { return c_; }
  /// Face velocity in direction d at the `+d` face of each cell: `uf(i,j,k)` sits at (i+1/2,j,k).
  /// The caller owns these (prescribed field at this rung) and must fill them on the ghost ring
  /// too — the `-d` face of the first inner cell is the `+d` face of a ghost cell.
  SField faceU() const { return uf_; }
  SField faceV() const { return vf_; }
  SField faceW() const { return wf_; }
  SField faceVel(int d) const { return d == 0 ? uf_ : (d == 1 ? vf_ : wf_); }

  // ---- rung V5a (WO-Q): optional cut-cell geometry ------------------------------------------
  //
  // OFF by default and EVERY kernel branches on `hasGeometry()` OUTSIDE its lambda, so with no
  // geometry attached the arithmetic executed is literally the V1 expression (not a multiplication
  // by 1.0) and the whole V1/V2a/V3/V4 battery stays byte-identical. See `vof/cutcell.hpp` for the
  // three rules the geometry branch adds and why the conserved functional is `sum eps_eff C`.
  //
  // USAGE (the solver does exactly this in `buildVofBlock`):
  //   a.enableGeometry();
  //   <fill faceOpenness(d) with the +d face openness and epsFraction() with buildCellFraction,
  //    on the INNER region, then run the block's own ghost policy on each of them>
  //   a.classifyGeometry();            // writes kindDouble() over every index except the low plane
  //   <run the block's ghost policy on kindDouble()>   // makes the low plane / ghosts the owner's
  //   a.finalizeGeometry();            // kindDouble() -> the unsigned-char classification
  bool hasGeometry() const { return hasGeom_; }
  /// Allocate the geometry views. Idempotent; does not classify.
  void enableGeometry() {
    if (len_ <= 0)
      throw std::runtime_error("peclet::flow::vof::WyAdvector: enableGeometry before init()");
    if (!hasGeom_) {
      for (int d = 0; d < 3; ++d)
        of_[d] = SField("vof::openness", len_);
      eps_ = SField("vof::eps", len_);
      kindD_ = SField("vof::kindD", len_);
      kind_ = UCField("vof::kind", len_);
      fill_ = UCField("vof::fillState", len_);
      mark_ = UCField("vof::fillMark", len_);
      hasGeom_ = true;
    }
  }
  /// Drop the geometry (back to the uncut kernels, byte-identically).
  void disableGeometry() { hasGeom_ = false; }
  /// Face openness of the `+d` face of each cell (the advector's own high-face convention — the
  /// SAME convention and the SAME one-cell shift the face velocity uses, `colour_field.hpp`).
  SField faceOpenness(int d) const { return of_[d]; }
  /// Fluid volume fraction of each cell (`buildCellFraction`, a multiple of 1/64).
  SField epsFraction() const { return eps_; }
  /// The cell classification as a DOUBLE (1 = solid), so it can ride the block's own halo exchange.
  SField kindDouble() const { return kindD_; }
  /// The cell classification (`VofCellKind`).
  UCField cellKind() const { return kind_; }
  /// Weymouth's admissible-interval clamp on the openness-weighted flux (`cutcell.hpp`
  /// `vofCutFluxClamp`). ON by default; the ablation measures what the whole-cell-PLIC-times-open-
  /// area approximation costs in boundedness and hence in clipped volume.
  bool cutFluxClamp = true;
  /// Faces the clamp had to bind, last `advect()`.
  long clampedFaces() const { return clampedFaces_; }
  /// |liquid volume| the cut-cell clip moved during the last `advect()` (0 without geometry).
  double clippedVolume() const { return clippedVolume_; }
  double clippedSigned() const { return clippedSigned_; }

  /// Classify every cell from `eps` and the six face openness values (`cutcell.hpp` rule 1) into
  /// `kindDouble()`. The `-d` face of a cell at index 0 on axis d lies outside the block, so the
  /// low plane of each axis is left FLUID here and is meant to be overwritten by the block's own
  /// ghost exchange (which puts the owner's classification there) before `finalizeGeometry()`.
  void classifyGeometry() {
    if (!hasGeom_)
      return;
    const I3 e = e_;
    const long sx = 1, sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField ox = of_[0], oy = of_[1], oz = of_[2], ep = eps_, kd = kindD_;
    Kokkos::parallel_for(
        "vof::wy::classify",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (x == 0 || y == 0 || z == 0) {
            kd(i) = 0.0;  // the exchange owns this plane
            return;
          }
          const double oLo[3] = {ox(i - sx), oy(i - sy), oz(i - sz)};
          const double oHi[3] = {ox(i), oy(i), oz(i)};
          kd(i) = vofIsSolidCell(ep(i), oLo, oHi) ? 1.0 : 0.0;
        });
    Kokkos::fence();
  }
  /// `kindDouble()` (ghost-exchanged) -> the unsigned-char classification every kernel reads.
  void finalizeGeometry() {
    if (!hasGeom_)
      return;
    SField kd = kindD_;
    UCField kk = kind_;
    Kokkos::parallel_for(
        "vof::wy::finalize_geom", Kokkos::RangePolicy<SExec>(SExec(), 0, len_),
        KOKKOS_LAMBDA(long i) { kk(i) = kd(i) > 0.5 ? kVofSolid : kVofFluid; });
    Kokkos::fence();
  }

  /// The three-pass neutral (90 deg) solid-band fill of `cutcell.hpp`. Call from the block's ghost
  /// policy AFTER the exchange + non-periodic clamp, and exchange once more afterwards so the
  /// outermost ghost layer holds its owner's filled value.
  void solidBandFill() {
    if (!hasGeom_)
      return;
    {
      UCField fs = fill_, mk = mark_, kk = kind_;
      Kokkos::parallel_for(
          "vof::wy::fill_reset", Kokkos::RangePolicy<SExec>(SExec(), 0, len_),
          KOKKOS_LAMBDA(long i) {
            fs(i) = kk(i) == kVofSolid ? kVofFillNone : kVofFillFluid;
            mk(i) = 0u;
          });
    }
    for (int k = 1; k <= 3; ++k)
      solidBandFillPass(k);
    Kokkos::fence();
  }


  // ---- hooks -------------------------------------------------------------------------------
  /// Refresh EVERY ghost layer of the given cell field (MPI exchange + domain BC fill). Required.
  std::function<void(SField)> exchange;
  /// All-reduce max across ranks; identity when unset (single block).
  std::function<double(double)> globalMax;

  /// Compaction of the reconstruction pass onto the mixed cells. Pure optimization — switching it
  /// off must reproduce the same field bit for bit (gated in `tests/kokkos/test_vof_advect.cpp`).
  bool useWorklist = true;
  /// Abort threshold on max |uf| dt / h. Defaults to **Weymouth's proven 3D boundedness bound**
  /// 1/(2(N-1)) = 0.25 (thesis eq. A.33); the familiar 0.5 is the 2D value. Raise it only
  /// deliberately (2D work, or probing the gap) — see the file header.
  double cflLimit = 0.25;
  /// Take the CFL over INTERFACE-ADJACENT cells only (mixed cells and their face neighbours)
  /// instead of over every owned face — see `maxCourantInterface()`. Default OFF so the V1
  /// standalone battery is byte-identical; `IbmSolver` turns it on (WO-J item 4).
  bool interfaceLocalCfl = false;
  /// DIAGNOSTIC ONLY — recompute the dilation flag before every sweep instead of freezing it once.
  /// This is the #1 documented trap of the method, kept switchable so the damage is a measured
  /// number rather than folklore (`tests/kokkos/test_vof_advect.cpp` gate G). Never enable it in
  /// production: it breaks the telescoping that gives exact conservation.
  bool debugRecomputeDilation = false;

  /// Bring the colour ghosts up to date. `advect()` assumes valid ghosts on entry and leaves them
  /// valid on exit, so this is only needed once after initialization.
  void syncGhosts() {
    requireExchange();
    exchange(c_);
  }

  /// One Weymouth-Yue step: freeze the dilation flag, then three sweeps in the permutation
  /// selected by `step`. Throws if the CFL cap is violated.
  void advect(double dt, long step) {
    requireExchange();
    const double dth = dt / h_;
    const double cflLocal =
        interfaceLocalCfl
            ? (hasGeom_ ? maxCourantInterfaceCut(dth) : maxCourantInterface(dth))
            : maxCourant(dth);
    const double cfl = globalMax ? globalMax(cflLocal) : cflLocal;
    lastCfl_ = cfl;
    // Weymouth's bound is INCLUSIVE (thesis eq. A.33: |a| <= 1/(2(N-1))), so a step exactly at
    // `cflLimit` is admissible and only a strictly larger one aborts. NaN propagates to an abort.
    if (!(cfl <= cflLimit)) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "peclet::flow::vof::WyAdvector: CFL = max|uf| dt/h = %.6g exceeds the "
                    "Weymouth-Yue boundedness cap %.6g (dt = %.6g, h = %.6g) - reduce dt",
                    cfl, cflLimit, dt, h_);
      throw std::runtime_error(msg);
    }

    // (1) THE dilation flag: frozen ONCE from C^n, used unchanged by all three sweeps.
    freezeDilationFlag();
    if (hasGeom_) {  // rung V5a: the clip / clamp census is per STEP, not per sweep
      clippedVolume_ = 0.0;
      clippedSigned_ = 0.0;
      clampedFaces_ = 0;
    }

    // (2) three directional sweeps
    const int* perm = kWySweepPerm[static_cast<int>(step % 6)];
    for (int s = 0; s < 3; ++s) {
      const int d = perm[s];
      if (debugRecomputeDilation && s > 0)
        freezeDilationFlag();  // the trap, on purpose — see `debugRecomputeDilation`
      reconstruct();
      computeFluxes(d, dth);
      applySweep(d, dth);
      if (hasGeom_)
        clipCutCells();  // rung V5a rule 3 (diagnostic clip, cut cells only)
      exchange(c_);
    }
    ++steps_;
  }

  double lastCfl() const { return lastCfl_; }
  long lastMixedCount() const { return mixedCount_; }

  // ---- rung V2b (WO-K) hooks -----------------------------------------------------------------
  // The momentum-consistent transport of `rho^c u_c` must be driven by the SAME PLIC planes, the
  // SAME face Courant numbers and the SAME sweep order as this colour advection. The planes are
  // re-reconstructed before EVERY sweep and overwritten in place, so a sibling advector that ran
  // after `advect()` returned would see only the last sweep's planes — the momentum sweeps have to
  // be INTERLEAVED with the colour sweeps. `MomentumConsistentAdvector` therefore drives the step
  // itself out of the public pieces below (`freezeDilationFlag` / `reconstruct` / `computeFluxes` /
  // `applySweep` / `exchange`) instead of calling `advect()`, and reads the planes through these
  // accessors between `computeFluxes` and `applySweep`, i.e. while they still describe the field
  // the sweep is about to move. Nothing here changes `advect()`.

  /// PLIC normal component `d` of the current reconstruction (valid on the inner region grown by 1,
  /// for mixed cells only — `wyIsMixed` is the contract, exactly as `wyFaceFlux` reads them).
  SField planeM(int d) const { return d == 0 ? mx_ : (d == 1 ? my_ : mz_); }
  /// PLIC plane offset of the current reconstruction.
  SField planeAlpha() const { return alpha_; }
  /// The frozen dilation indicator `H(C^n - 1/2)` of the current step (pressure cells).
  UCField dilationFlag() const { return cc_; }

  /// The interface-local (or global) Courant number for this dt, all-reduced, with the
  /// Weymouth-Yue boundedness cap enforced exactly as `advect()` enforces it — `lastCfl_` is set
  /// the same way, so a WO-K step reports the same number a WO-J step would.
  ///
  /// This duplicates the guard at the top of `advect()` rather than refactoring it: `advect()` is a
  /// validated body (V1 gates A-G) and hard rule 1 of the work orders forbids editing one. Keep the
  /// two in step if the cap ever changes.
  double checkCourant(double dt) {
    const double dth = dt / h_;
    const double cflLocal =
        interfaceLocalCfl
            ? (hasGeom_ ? maxCourantInterfaceCut(dth) : maxCourantInterface(dth))
            : maxCourant(dth);
    const double cfl = globalMax ? globalMax(cflLocal) : cflLocal;
    lastCfl_ = cfl;
    if (!(cfl <= cflLimit)) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "peclet::flow::vof::WyAdvector: CFL = max|uf| dt/h = %.6g exceeds the "
                    "Weymouth-Yue boundedness cap %.6g (dt = %.6g, h = %.6g) - reduce dt",
                    cfl, cflLimit, dt, h_);
      throw std::runtime_error(msg);
    }
    return cfl;
  }

  /// Local census over the inner region.
  Diagnostics diagnostics() const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField c = c_;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    double sum = 0.0, mn = 0.0, mx = 0.0;
    long mixed = 0, wisps = 0;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    Kokkos::parallel_reduce(
        "vof::wy::diag_sum", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) { acc += c(L3(x, y, z, e)); }, sum);
    Kokkos::parallel_reduce(
        "vof::wy::diag_min", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          acc = Kokkos::fmin(acc, c(L3(x, y, z, e)));
        },
        Kokkos::Min<double>(mn));
    Kokkos::parallel_reduce(
        "vof::wy::diag_max", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          acc = Kokkos::fmax(acc, c(L3(x, y, z, e)));
        },
        Kokkos::Max<double>(mx));
    Kokkos::parallel_reduce(
        "vof::wy::diag_counts", pol,
        KOKKOS_LAMBDA(int x, int y, int z, long& nm, long& nw) {
          const double v = c(L3(x, y, z, e));
          if (wyIsMixed(v)) {
            ++nm;
            if (v < 1e-8 || v > 1.0 - 1e-8)
              ++nw;
          }
        },
        mixed, wisps);
    Kokkos::fence();
    Diagnostics dg;
    dg.sumC = sum;
    dg.minC = mn;
    dg.maxC = mx;
    dg.mixed = mixed;
    dg.wisps = wisps;
    if (hasGeom_)
      cutDiagnostics(dg);
    return dg;
  }

  /// Max |uf| dt / h over this block's owned faces (the `+` face of every inner cell; the `-` face
  /// of the first inner cell is a neighbour's owned face, so a global max covers every face once).
  double maxCourant(double dth) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField u = uf_, v = vf_, w = wf_;
    double m = 0.0;
    Kokkos::parallel_reduce(
        "vof::wy::cfl",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = L3(x, y, z, e);
          acc = Kokkos::fmax(acc, Kokkos::fabs(u(i)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(v(i)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(w(i)));
        },
        Kokkos::Max<double>(m));
    Kokkos::fence();
    return m * dth;
  }

  /// Max |uf| dt/h over the faces of INTERFACE-ADJACENT cells only. A cell is interface-adjacent
  /// when its colour is not locally constant: it is mixed (`wyIsMixed` — the same predicate the V1
  /// worklist compacts on), or one of its six face neighbours carries a different colour.
  ///
  /// Why not the global max (`maxCourant`). Weymouth's bound is a bound on each individual FLUX,
  /// and a flux only has something to bound where boundedness can be lost. A cell whose whole 1D
  /// neighbourhood is full (or empty) is EXACTLY stationary in floating point whatever its face
  /// Courant numbers are — the algebraic flux branch and the dilation term cancel bit for bit (see
  /// the file header) — so its faces cannot make it leave [0,1] and are irrelevant to the proof.
  /// Measured in V1: on Zalesak the domain-corner faces run at 0.314 while the interface never
  /// exceeds 0.157, so a global max throttles the step by 2x for nothing (`VOF_PLAN.md` §6).
  ///
  /// The predicate is a colour DIFFERENCE, not merely `mixed`, and that difference matters: a
  /// perfectly sharp grid-aligned interface (…1,1,0,0…) has NO mixed cell at all, so a
  /// mixed-only band would be empty and the limiter toothless on exactly the configuration the
  /// hydrostatic acid test starts from. With the difference test that face is in the band, and for
  /// a resolved PLIC interface (which always has mixed cells) the two definitions coincide.
  ///
  /// A pure reduction over inner cells with a ±1 predicate, so it is decomposition-independent
  /// under the global max the caller applies. Returns 0 for a uniform colour field (no interface,
  /// no constraint). Reads C at ±1, so the colour ghosts must be valid on entry (`advect()`
  /// requires that anyway).
  double maxCourantInterface(double dth) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sx = 1, sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField c = c_, u = uf_, v = vf_, w = wf_;
    double m = 0.0;
    Kokkos::parallel_reduce(
        "vof::wy::cfl_interface",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = L3(x, y, z, e);
          const double ci = c(i);
          const bool band = wyIsMixed(ci) || c(i - sx) != ci || c(i + sx) != ci ||
                            c(i - sy) != ci || c(i + sy) != ci || c(i - sz) != ci ||
                            c(i + sz) != ci;
          if (!band)
            return;
          acc = Kokkos::fmax(acc, Kokkos::fabs(u(i)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(u(i - sx)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(v(i)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(v(i - sy)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(w(i)));
          acc = Kokkos::fmax(acc, Kokkos::fabs(w(i - sz)));
        },
        Kokkos::Max<double>(m));
    Kokkos::fence();
    // Max's identity is -inf; an empty band (a uniform colour field) means no constraint, not an
    // unconstrained step. Clamp so the reported CFL is a number and the cap test is meaningful.
    return Kokkos::fmax(m, 0.0) * dth;
  }

  /// The interface-local Courant number for this dt/h, using the CUT-CELL rule when geometry is
  /// attached (`o_f |a_f| / max(eps_i, 0.1)`, `cutcell.hpp` rule 2) and the uncut one otherwise.
  /// This is what `advect()` gates on, so a caller sizing dt sees the same number.
  double maxCourantInterfaceAuto(double dth) const {
    return hasGeom_ ? maxCourantInterfaceCut(dth) : maxCourantInterface(dth);
  }

  /// Max |discrete face divergence| * dt/h over inner cells — the exact quantity the conservation
  /// floor is set by (the dilation term contributes `c_i` times this to the global volume budget).
  double maxDiscreteDivergence(double dth) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField u = uf_, v = vf_, w = wf_;
    double m = 0.0;
    Kokkos::parallel_reduce(
        "vof::wy::divmax",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = L3(x, y, z, e);
          const double d = (u(i) - u(i - 1)) + (v(i) - v(i - sy)) + (w(i) - w(i - sz));
          acc = Kokkos::fmax(acc, Kokkos::fabs(d) * dth);
        },
        Kokkos::Max<double>(m));
    Kokkos::fence();
    return m;
  }

 public:
  // ---- implementation detail -----------------------------------------------------------------
  // These stay PUBLIC only because nvcc rejects an extended __host__ __device__ lambda whose
  // enclosing member function has private or protected access. Treat them as private.

  void requireExchange() const {
    if (!exchange)
      throw std::runtime_error("peclet::flow::vof::WyAdvector: the `exchange` hook is not set");
  }

  void freezeDilationFlag() {
    SField c = c_;
    UCField cc = cc_;
    Kokkos::parallel_for(
        "vof::wy::freeze", Kokkos::RangePolicy<SExec>(SExec(), 0, len_),
        // thesis eq. A.28: g = 1 if C^n > 1/2, else 0. Strict `>` (the tie is measure zero and the
        // proof only needs the flag to be a constant of the step).
        KOKKOS_LAMBDA(long i) { cc(i) = c(i) > 0.5 ? 1u : 0u; });
  }

  /// PLIC over the inner region grown by one cell in every direction: the donor of a face of an
  /// inner cell is at most one cell outside it. Non-mixed cells are left untouched — the flux never
  /// reads their (m, alpha), which is what keeps `useWorklist` bit-neutral.
  void reconstruct() {
    const I3 e = e_;
    const int g = g_;
    const int rx = n_.x + 2, ry = n_.y + 2, rz = n_.z + 2;
    const long region = static_cast<long>(rx) * ry * rz;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField c = c_, mx = mx_, my = my_, mz = mz_, al = alpha_;

    if (useWorklist) {
      LField list = list_;
      long cnt = 0;
      Kokkos::parallel_scan(
          "vof::wy::worklist", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
            const int ix = static_cast<int>(r % rx);
            const int iy = static_cast<int>((r / rx) % ry);
            const int iz = static_cast<int>(r / (static_cast<long>(rx) * ry));
            const long i = L3(g - 1 + ix, g - 1 + iy, g - 1 + iz, e);
            if (wyIsMixed(c(i))) {
              if (final)
                list(upd) = i;
              ++upd;
            }
          },
          cnt);
      Kokkos::fence();
      mixedCount_ = cnt;
      Kokkos::parallel_for(
          "vof::wy::plic", Kokkos::RangePolicy<SExec>(SExec(), 0, cnt),
          KOKKOS_LAMBDA(long t) { wyReconstructCell(c, list(t), sy, sz, mx, my, mz, al); });
    } else {
      Kokkos::parallel_for(
          "vof::wy::plic_dense", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r) {
            const int ix = static_cast<int>(r % rx);
            const int iy = static_cast<int>((r / rx) % ry);
            const int iz = static_cast<int>(r / (static_cast<long>(rx) * ry));
            const long i = L3(g - 1 + ix, g - 1 + iy, g - 1 + iz, e);
            if (wyIsMixed(c(i)))
              wyReconstructCell(c, i, sy, sz, mx, my, mz, al);
          });
    }
  }

  /// One flux per `d`-face touched by an inner cell, stored at the cell on the face's `-` side.
  /// Computing the face once (rather than once per adjacent cell) is what makes the flux telescope
  /// bit-exactly, in-rank and across a rank boundary alike.
  void computeFluxes(int d, double dth) {
    if (hasGeom_) {  // rung V5a: openness-weighted flux (`cutcell.hpp` eq. 1)
      computeFluxesCut(d, dth);
      return;
    }
    const I3 e = e_;
    const int g = g_;
    const long sd =
        d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
    // d-index [g-1, g+n_d): the `-` face of the first inner cell through the `+` face of the last.
    // Transverse indices stay on the inner region.
    int lo[3] = {g, g, g};
    const int hi[3] = {g + n_.x, g + n_.y, g + n_.z};
    lo[d] -= 1;
    SField c = c_, mx = mx_, my = my_, mz = mz_, al = alpha_, fl = flux_, u = faceVel(d);
    Kokkos::parallel_for(
        "vof::wy::flux",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {lo[0], lo[1], lo[2]},
                                                      {hi[0], hi[1], hi[2]}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long p = L3(x, y, z, e);
          fl(p) = wyFaceFlux(u(p) * dth, p, sd, d, c, mx, my, mz, al);
        });
  }

  /// `C_i += (F_{i-} - F_{i+}) + c_i (a_{i+} - a_{i-})`, over inner cells.
  void applySweep(int d, double dth) {
    if (hasGeom_) {  // rung V5a: the update in FLUID-VOLUME units
      applySweepCut(d, dth);
      return;
    }
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sd =
        d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
    SField c = c_, fl = flux_, u = faceVel(d);
    UCField cc = cc_;
    Kokkos::parallel_for(
        "vof::wy::update",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          // The dilation term must scale the SAME uf by the SAME dt/h as the flux, or the exact
          // cancellation in full cells (file header) is lost to rounding.
          const double aP = u(i) * dth, aM = u(i - sd) * dth;
          const double dil = cc(i) ? (aP - aM) : 0.0;
          c(i) = c(i) + (fl(i - sd) - fl(i)) + dil;
        });
  }

  // ---- rung V5a (WO-Q) kernel bodies ---------------------------------------------------------
  // Siblings of the V1 kernels above, reached only through the `hasGeom_` branch at the top of
  // each. The V1 bodies are untouched, so "no geometry" is byte-identical by construction rather
  // than by a tolerance.

  /// `F_f = o_f * wyFaceFlux(a_f, ...)`. One number per face, stored at the face's `-` side exactly
  /// as the uncut kernel stores it, so the flux sum telescopes bit-exactly whatever `o` is.
  void computeFluxesCut(int d, double dth) {
    const I3 e = e_;
    const int g = g_;
    const long sd =
        d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
    int lo[3] = {g, g, g};
    const int hi[3] = {g + n_.x, g + n_.y, g + n_.z};
    lo[d] -= 1;
    SField c = c_, mx = mx_, my = my_, mz = mz_, al = alpha_, fl = flux_, u = faceVel(d),
           o = of_[d], ep = eps_;
    UCField kd = kind_;
    const bool clamp = cutFluxClamp;
    long nclamp = 0;
    Kokkos::parallel_reduce(
        "vof::wy::flux_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {lo[0], lo[1], lo[2]},
                                                      {hi[0], hi[1], hi[2]}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const long p = L3(x, y, z, e);
          const double op = o(p);
          if (op == 0.0) {
            fl(p) = 0.0;
            return;
          }
          const double a = u(p) * dth;
          const double raw = vofCutFlux(op, wyFaceFlux(a, p, sd, d, c, mx, my, mz, al));
          if (!clamp || a == 0.0) {
            fl(p) = raw;
            return;
          }
          const long q = a > 0.0 ? p : p + sd;
          const double cq = c(q);
          if (!wyIsMixed(cq)) {  // algebraic branch: already exactly bounded, and clamping it
            fl(p) = raw;         // would break the exact full-cell cancellation
            return;
          }
          const double aa = a > 0.0 ? a : -a;
          const double mag = raw > 0.0 ? raw : -raw;
          const double cl = vofCutFluxClamp(mag, op * aa, vofEpsEff(ep(q), kd(q) == kVofSolid), cq);
          if (cl != mag)
            ++acc;
          fl(p) = a > 0.0 ? cl : -cl;
        },
        nclamp);
    Kokkos::fence();
    clampedFaces_ += nclamp;
  }

  /// `eps_i C_i += (F_- - F_+) + c_i (o_+ a_+ - o_- a_-)`, then divide by `eps_eff`. Solid cells
  /// (`kVofSolid`) are skipped entirely — their colour comes from `solidBandFill()`.
  void applySweepCut(int d, double dth) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sd =
        d == 0 ? 1 : (d == 1 ? static_cast<long>(e_.x) : static_cast<long>(e_.x) * e_.y);
    SField c = c_, fl = flux_, u = faceVel(d), o = of_[d], ep = eps_;
    UCField cc = cc_, kd = kind_;
    Kokkos::parallel_for(
        "vof::wy::update_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid)
            return;
          const double epsEff = vofEpsEff(ep(i), false);
          // Same `o_f a_f` in the flux and in the dilation term — that shared factor is what makes
          // the two contributions telescope against the projection's openness-weighted divergence.
          // `o * (u * dth)` and NOT `(o * u) * dth`: the flux forms the Courant number FIRST
          // (`wyFaceFlux(u(p) * dth, ...)`), so only this association makes the two cancel to the
          // last bit in a full cell — the property that keeps the conservation floor on the
          // interface area rather than on the domain volume (see the file header).
          const double aP = o(i) * (u(i) * dth), aM = o(i - sd) * (u(i - sd) * dth);
          const double dil = cc(i) ? (aP - aM) : 0.0;
          c(i) = c(i) + ((fl(i - sd) - fl(i)) + dil) / epsEff;
        });
  }

  /// Rule 3: clip C into [0,1] in CUT cells only (`eps_eff < 1`), accumulating the liquid volume
  /// the clip moved. A diagnostic first — if it is not negligible the flux approximation, not the
  /// clip, is what needs fixing.
  void clipCutCells() {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField c = c_, ep = eps_;
    UCField kd = kind_;
    double acc = 0.0, sgn = 0.0;
    Kokkos::parallel_reduce(
        "vof::wy::clip_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& a, double& b) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid)
            return;
          const double epsEff = vofEpsEff(ep(i), false);
          if (!(epsEff < 1.0))
            return;  // whole fluid cell: V1 boundedness applies verbatim, do not touch it
          const double cv = c(i);
          if (cv >= 0.0 && cv <= 1.0)
            return;
          double moved = 0.0;
          c(i) = vofClipCut(cv, epsEff, moved);
          a += Kokkos::fabs(moved);
          b += moved;
        },
        acc, sgn);
    Kokkos::fence();
    clippedVolume_ += acc;
    clippedSigned_ += sgn;
  }

  /// Rule 2: the interface-local Courant number in cut cells is `o_f |a_f| / max(eps_i, 0.1)`.
  /// Same interface band predicate as `maxCourantInterface` (a colour DIFFERENCE, not merely
  /// "mixed"), and solid cells are outside it by construction.
  double maxCourantInterfaceCut(double dth) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sx = 1, sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField c = c_, u = uf_, v = vf_, w = wf_, ox = of_[0], oy = of_[1], oz = of_[2], ep = eps_;
    UCField kd = kind_;
    double m = 0.0;
    Kokkos::parallel_reduce(
        "vof::wy::cfl_interface_cut",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid)
            return;
          const double ci = c(i);
          const bool band = wyIsMixed(ci) || c(i - sx) != ci || c(i + sx) != ci ||
                            c(i - sy) != ci || c(i + sy) != ci || c(i - sz) != ci ||
                            c(i + sz) != ci;
          if (!band)
            return;
          const double ei = ep(i);
          acc = Kokkos::fmax(acc, vofCutCourant(ox(i), u(i) * dth, ei));
          acc = Kokkos::fmax(acc, vofCutCourant(ox(i - sx), u(i - sx) * dth, ei));
          acc = Kokkos::fmax(acc, vofCutCourant(oy(i), v(i) * dth, ei));
          acc = Kokkos::fmax(acc, vofCutCourant(oy(i - sy), v(i - sy) * dth, ei));
          acc = Kokkos::fmax(acc, vofCutCourant(oz(i), w(i) * dth, ei));
          acc = Kokkos::fmax(acc, vofCutCourant(oz(i - sz), w(i - sz) * dth, ei));
        },
        Kokkos::Max<double>(m));
    Kokkos::fence();
    return Kokkos::fmax(m, 0.0);
  }

  /// One pass of the neutral solid-band fill. Writes solid cells at ghost depth <= 3-k, reading
  /// only fluid cells and cells filled in an EARLIER pass — a cell being written in THIS pass still
  /// carries `kVofFillNone`, so it is never read and the pass is race-free without a second buffer.
  void solidBandFillPass(int k) {
    const I3 e = e_, n = n_;
    const int g = g_, maxDepth = 3 - k;
    const long sx = 1, sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField c = c_;
    UCField fs = fill_, mk = mark_;
    Kokkos::parallel_for(
        "vof::wy::band_fill",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          if (vofGhostDepth(x, y, z, g, n.x, n.y, n.z) > maxDepth)
            return;
          const long i = L3(x, y, z, e);
          if (fs(i) != kVofFillNone)
            return;
          const long nb[6] = {i - sx, i + sx, i - sy, i + sy, i - sz, i + sz};
          double acc = 0.0;
          int cnt = 0;
          for (int q = 0; q < 6; ++q)
            if (vofFillReadable(fs(nb[q]), k)) {
              acc += c(nb[q]);
              ++cnt;
            }
          if (cnt == 0)
            return;  // no donor yet — a later pass may reach it, or it stays as it was
          c(i) = acc / cnt;
          mk(i) = 1u;
        });
    Kokkos::fence();
    Kokkos::parallel_for(
        "vof::wy::band_fill_commit", Kokkos::RangePolicy<SExec>(SExec(), 0, len_),
        KOKKOS_LAMBDA(long i) {
          if (mk(i)) {
            fs(i) = static_cast<unsigned char>(kVofFillPass0 + k);
            mk(i) = 0u;
          }
        });
    Kokkos::fence();
  }

  /// The cut-cell half of `diagnostics()` (zero-cost when no geometry is attached).
  void cutDiagnostics(Diagnostics& dg) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField c = c_, ep = eps_;
    UCField kd = kind_;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    MD pol(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z});
    double vol = 0.0, raw = 0.0, solidC = 0.0;
    long ncut = 0, nsolid = 0;
    Kokkos::parallel_reduce(
        "vof::wy::cut_sums", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& v, double& r, double& sc) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid) {
            sc += c(i);
            return;
          }
          v += vofEpsEff(ep(i), false) * c(i);
          r += ep(i) * c(i);
        },
        vol, raw, solidC);
    Kokkos::parallel_reduce(
        "vof::wy::cut_counts", pol,
        KOKKOS_LAMBDA(int x, int y, int z, long& nc, long& ns) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid) {
            ++ns;
            return;
          }
          const double ee = vofEpsEff(ep(i), false);
          if (ee < 1.0)
            ++nc;
        },
        ncut, nsolid);
    double mn = 1e300, mx = -1e300;
    Kokkos::parallel_reduce(
        "vof::wy::cut_minfluid", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& a) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid || ep(i) < 1.0)
            return;
          a = Kokkos::fmin(a, c(i));
        },
        Kokkos::Min<double>(mn));
    Kokkos::parallel_reduce(
        "vof::wy::cut_maxfluid", pol,
        KOKKOS_LAMBDA(int x, int y, int z, double& a) {
          const long i = L3(x, y, z, e);
          if (kd(i) == kVofSolid || ep(i) < 1.0)
            return;
          a = Kokkos::fmax(a, c(i));
        },
        Kokkos::Max<double>(mx));
    Kokkos::fence();
    dg.volume = vol;
    dg.rawVolume = raw;
    dg.solidFillSum = solidC;
    dg.cutCells = ncut;
    dg.solidCells = nsolid;
    dg.minCFluid = mn > 1e299 ? 0.0 : mn;
    dg.maxCFluid = mx < -1e299 ? 0.0 : mx;
    dg.clippedVolume = clippedVolume_;
    dg.clippedSigned = clippedSigned_;
    dg.clampedFaces = clampedFaces_;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 3;
  double h_ = 1.0;
  long len_ = 0, listCap_ = 0;
  SField c_, mx_, my_, mz_, alpha_, flux_, uf_, vf_, wf_;
  UCField cc_;
  LField list_;
  long mixedCount_ = 0, steps_ = 0;
  double lastCfl_ = 0.0;
  // rung V5a (WO-Q) cut-cell geometry; `hasGeom_` false => every V1 kernel runs unchanged.
  bool hasGeom_ = false;
  SField of_[3], eps_, kindD_;
  UCField kind_, fill_, mark_;
  double clippedVolume_ = 0.0, clippedSigned_ = 0.0;
  long clampedFaces_ = 0;
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_ADVECT_WY_HPP
