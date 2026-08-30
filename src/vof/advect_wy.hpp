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
    const double cflLocal = maxCourant(dth);
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

    // (2) three directional sweeps
    const int* perm = kWySweepPerm[static_cast<int>(step % 6)];
    for (int s = 0; s < 3; ++s) {
      const int d = perm[s];
      if (debugRecomputeDilation && s > 0)
        freezeDilationFlag();  // the trap, on purpose — see `debugRecomputeDilation`
      reconstruct();
      computeFluxes(d, dth);
      applySweep(d, dth);
      exchange(c_);
    }
    ++steps_;
  }

  double lastCfl() const { return lastCfl_; }
  long lastMixedCount() const { return mixedCount_; }

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
    return Diagnostics{sum, mn, mx, mixed, wisps};
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
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_ADVECT_WY_HPP
