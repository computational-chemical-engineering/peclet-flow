/// @file
/// @brief flow — VoF rung V3 (WO-O): the view-level driver for the curvature cascade.
///
/// `vof/curvature.hpp` holds the promotable, container-free kernels (WO-D signature rule). This
/// file is their block-walking driver: it owns the PLIC scratch fields, walks the colour block,
/// and produces the two output fields
///
///   * `kappa`        — 2H in units of 1/h (cell units), positive for a convex blob of liquid;
///   * `kappa_branch` — which tier of the cascade produced it (`CurvatureBranch`), as a double so
///                      it rides the ordinary field registry / IO / redistribute path.
///
/// Both live on the colour advector's **g = 3** block, which is exactly the reach the cascade
/// needs and no more:
///
///   * tier 1/2 (height functions) read a 3 x 3 x 7 patch of colour — transverse +/-1, column
///     +/-3;
///   * tier 3 (the 5^3 PV fit) reads a PLIC plane at +/-2, and each plane's MYC normal reads
///     colour at +/-1 around it — again +/-3.
///
/// So **no new halo machinery**: `IbmSolver::vofFillGhosts` on the colour field is the only
/// exchange this rung needs, and the whole computation is a pure local stencil, hence bitwise
/// decomposition-independent by construction (there is not one reduction in it).
///
/// The one deliberate cost: the PLIC planes are rebuilt here over the inner region grown by 2,
/// rather than reusing `WyAdvector`'s (`reconstruct()` covers inner+1 only, and it is a validated
/// kernel body that hard rule 1 forbids editing).
#ifndef PECLET_FLOW_VOF_CURVATURE_FIELD_HPP
#define PECLET_FLOW_VOF_CURVATURE_FIELD_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <chrono>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "vof/advect_wy.hpp"  // wyIsMixed, wyReconstructCell
#include "vof/curvature.hpp"

namespace peclet::flow::vof {

/// `wyIsMixed` with a wisp threshold: the cell carries an interface only while `eps < C < 1 - eps`.
/// `eps = 0` is `wyIsMixed` exactly (`C > 0 && C < 1`). See `VofCurvature::interfaceEps` for the
/// measurement that put it there.
KOKKOS_INLINE_FUNCTION bool vofIsInterface(double c, double eps) {
  return c > eps && c < 1.0 - eps;
}

/// WO-V9 — the per-cell bodies of the two cascade tiers, lifted VERBATIM out of the kernels they
/// used to be written inline in, so the dense (whole-region) and the compacted (interfacial-cell
/// list) kernels below can share ONE body instead of two copies that could drift apart. The only
/// substitution is the cell index: `i` is a parameter instead of `L3(x, y, z, e)`, and tier 3's
/// neighbour index `L3(x+ox, y+oy, z+oz, e)` becomes the identical `i + ox + oy*sy + oz*sz` (the
/// same integers, by the definition of L3). Nothing else changed, which is what lets the ctest
/// gate the compaction as bitwise.
template <class SF>
KOKKOS_INLINE_FUNCTION void curvHeightCell(long i, SF c, SF mx, SF my, SF mz, SF al, SF kap, SF br,
                                           long s0, long s1, long s2, double mtol, double ptW,
                                           double ieps, bool forceFb, bool oneDir, bool useFit) {
  const long st[3] = {s0, s1, s2};
        kap(i) = 0.0;
        if (!vofIsInterface(c(i), ieps)) {
          br(i) = static_cast<double>(kCurvNone);
          return;
        }
        if (forceFb) {
          br(i) = -1.0;
          return;
        }
        // Order the three column directions by the cell's own |n_d|. The normal is used as a
        // CATEGORICAL selector only — it is never differenced (see curvature.hpp).
        const double am[3] = {Kokkos::fabs(mx(i)), Kokkos::fabs(my(i)), Kokkos::fabs(mz(i))};
        int ord[3] = {0, 1, 2};
        for (int p = 1; p < 3; ++p)  // insertion sort, descending
          for (int q = p; q > 0 && am[ord[q]] > am[ord[q - 1]]; --q) {
            const int t = ord[q];
            ord[q] = ord[q - 1];
            ord[q - 1] = t;
          }
        const int ntry = oneDir ? 1 : 3;
        for (int t = 0; t < ntry; ++t) {
          const int d = ord[t];
          const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
          const long sd = st[d], s1 = st[d1], s2 = st[d2];
          double hh[9];
          int orient0 = 0;
          bool ok = true;
          for (int q = 0; q < 3 && ok; ++q)
            for (int p = 0; p < 3 && ok; ++p) {
              const long base = i + (p - 1) * s1 + (q - 1) * s2;
              double col[kHfColumn];
              for (int k = 0; k < kHfColumn; ++k)
                col[k] = c(base + (k - kHfColumn / 2) * sd);
              double h;
              int orient;
              if (!hfColumnHeight(col, kHfColumn, h, orient, mtol)) {
                ok = false;
                break;
              }
              if (orient0 == 0)
                orient0 = orient;
              else if (orient != orient0) {
                ok = false;  // the nine columns do not describe one single-valued surface
                break;
              }
              hh[p + 3 * q] = h;
            }
          if (!ok)
            continue;
          kap(i) = hfPatchKappa(hh);
          br(i) = static_cast<double>(t == 0 ? kCurvHf : kCurvHfMixed);
          return;
        }

        // Tier 2b — the MIXED height function. No direction gives nine consistent columns; pool
        // the interface positions of whichever of the 27 columns DO close and fit a paraboloid
        // through them in the target's own frame. Every cell it reads is inside the same
        // 3x3x7-per-direction footprint tier 1 already used, so it costs no extra halo.
        if (useFit) {
          const double m0 = mx(i), m1 = my(i), m2 = mz(i);
          const double nsq = m0 * m0 + m1 * m1 + m2 * m2;
          if (nsq > 0.0) {
            const double invn = 1.0 / Kokkos::sqrt(nsq);
            const double nn[3] = {m0 * invn, m1 * invn, m2 * invn};
            double t1[3], t2[3];
            curvFrame(nn, t1, t2);
            double vtx[8][3], ctr[3], area;
            const int nvt = plicPolygon(m0, m1, m2, al(i), vtx);
            polygonAreaCentroid(vtx, nvt, ctr, area);
            const double org[3] = {ctr[0] - 0.5, ctr[1] - 0.5, ctr[2] - 0.5};
            PtFit pf;
            ptFitInit(pf);
            for (int d = 0; d < 3; ++d) {
              const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
              const long sd = st[d], s1 = st[d1], s2 = st[d2];
              for (int q = -1; q <= 1; ++q)
                for (int p = -1; p <= 1; ++p) {
                  const long base = i + p * s1 + q * s2;
                  double col[kHfColumn];
                  for (int k = 0; k < kHfColumn; ++k)
                    col[k] = c(base + (k - kHfColumn / 2) * sd);
                  double hv;
                  int orient;
                  if (!hfColumnHeight(col, kHfColumn, hv, orient, mtol))
                    continue;
                  double X[3];
                  X[d1] = static_cast<double>(p);
                  X[d2] = static_cast<double>(q);
                  X[d] = orient * hv;  // the interface POSITION along d (h is the signed height)
                  ptFitAdd(pf, X, org, t1, t2, nn, ptW);
                }
            }
            double a[6];
            bool red = false;
            if (pf.npt >= 6 && ptFitSolve(pf, a, red) && !red) {
              kap(i) = paraboloidKappa(a);
              br(i) = static_cast<double>(kCurvHfFit);
              return;
            }
          }
        }
        br(i) = -1.0;  // to the fallback
}

template <class SF>
KOKKOS_INLINE_FUNCTION void curvFallbackCell(long i, SF c, SF mx, SF my, SF mz, SF al, SF kap,
                                             SF br, long sy, long sz, int gr, double dW,
                                             double cmin, double ieps) {
        if (br(i) >= 0.0)
          return;

        const double m0 = mx(i), m1 = my(i), m2 = mz(i);
        const double n2 = m0 * m0 + m1 * m1 + m2 * m2;
        if (!(n2 > 0.0)) {
          kap(i) = 0.0;
          br(i) = static_cast<double>(kCurvNoEstimate);
          return;
        }
        const double invn = 1.0 / Kokkos::sqrt(n2);
        const double nn[3] = {m0 * invn, m1 * invn, m2 * invn};
        double t1[3], t2[3];
        curvFrame(nn, t1, t2);

        // Frame origin: the target cell's own PLIC centroid, in target-centred cell units.
        double v[8][3], ctr[3], area;
        const int nv = plicPolygon(m0, m1, m2, al(i), v);
        polygonAreaCentroid(v, nv, ctr, area);
        const double org[3] = {ctr[0] - 0.5, ctr[1] - 0.5, ctr[2] - 0.5};

        PvFit fit;
        pvFitInit(fit);
        for (int oz = -gr; oz <= gr; ++oz)
          for (int oy = -gr; oy <= gr; ++oy)
            for (int ox = -gr; ox <= gr; ++ox) {
              const long j = i + (long)ox + (long)oy * sy + (long)oz * sz;
              if (!vofIsInterface(c(j), ieps))
                continue;
              const double off[3] = {static_cast<double>(ox), static_cast<double>(oy),
                                     static_cast<double>(oz)};
              pvFitAdd(fit, mx(j), my(j), mz(j), al(j), off, org, t1, t2, nn, dW, cmin);
            }

        double a[6];
        bool red = false;
        if (!pvFitSolve(fit, a, red)) {
          kap(i) = 0.0;
          br(i) = static_cast<double>(kCurvNoEstimate);
          return;
        }
        kap(i) = paraboloidKappa(a);
        br(i) = static_cast<double>(red ? kCurvPvReduced : kCurvPv);
}


/// Curvature of the colour field on an extended (inner + ghost) block.
class VofCurvature {
 public:
  /// Per-call census of the cascade over this block's inner region (LOCAL; a distributed caller
  /// sums them itself — the driver stays MPI-free, exactly as `WyAdvector` does).
  struct Stats {
    long interfacial = 0;  ///< cells carrying an interface (`vofIsInterface`, i.e. 0 < C < 1
                           ///< unless `interfaceEps` was raised) — the cells a kappa is asked for
    long hf = 0;           ///< tier 1: HF in the preferred direction
    long hfMixed = 0;      ///< tier 2a: HF in one of the other two directions
    long hfFit = 0;        ///< tier 2b: mixed HF -- paraboloid through the consistent columns
    long pv = 0;           ///< tier 3: PV paraboloid fit, full 6-parameter model
    long pvReduced = 0;    ///< tier 3 with the rank-deficient 3-parameter model
    long noEstimate = 0;   ///< NO estimate produced (must be 0 on every gated case)
  };

  /// @param ghost must be >= 3 (the column reach); the colour block's `kVofG` is 3.
  void init(int nx, int ny, int nz, int ghost) {
    if (ghost < kHfColumn / 2)
      throw std::invalid_argument(
          "peclet::flow::vof::VofCurvature: ghost width must be >= 3 (the height-function column "
          "reaches 3 cells and the 5^3 PV stencil's MYC normals reach 3 cells)");
    if (nx < 1 || ny < 1 || nz < 1)
      throw std::invalid_argument("peclet::flow::vof::VofCurvature: empty block");
    n_ = I3{nx, ny, nz};
    g_ = ghost;
    e_ = I3{nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    mx_ = SField("vof::curv::mx", len_);
    my_ = SField("vof::curv::my", len_);
    mz_ = SField("vof::curv::mz", len_);
    alpha_ = SField("vof::curv::alpha", len_);
    kappa_ = SField("vof::curv::kappa", len_);
    branch_ = SField("vof::curv::branch", len_);
    // WO-V9: the compaction lists. The grown one covers every cell `reconstructPlanes` writes; the
    // inner one every cell the cascade runs on. Sized at the full region (worst case: an entirely
    // interfacial block) so a scan can never overflow them.
    const long grown = (long)(n_.x + 2 * kPvHalf) * (n_.y + 2 * kPvHalf) * (n_.z + 2 * kPvHalf);
    listG_ = LField("vof::curv::listG", grown);
    listI_ = LField("vof::curv::listI", (long)n_.x * n_.y * n_.z);
  }

  bool ready() const { return kappa_.extent(0) != 0; }

  // ---- WO-V9: compaction, and the per-pass timers --------------------------------------------
  //
  // The cascade is the most divergent kernel in the VoF pipeline: on a resolved droplet the
  // interfacial cells are ~0.5 % of the block, so a dense `parallel_for` puts at most one or two
  // active lanes in a warp and the other thirty run the guard and idle for the whole height
  // function.  Compacting the interfacial cells into a contiguous list first makes those warps
  // full.  It is a pure re-ordering — every cell reads the same neighbours and writes the same
  // value — so the two paths are BIT-IDENTICAL and the ctest gates that.
  //
  // The timers follow `WyAdvector`'s rule exactly: fence at the boundaries only when armed.
  struct Timing {
    double planes = 0.0;    ///< the PLIC pass over the grown region
    double height = 0.0;    ///< tiers 1/2, the height-function cascade
    double fallback = 0.0;  ///< tier 3, the PLIC-volumetric paraboloid
    double census = 0.0;    ///< the branch census reduction
    double compact = 0.0;   ///< the two parallel_scans (0 when the compaction is off)
    long calls = 0;
  };
  bool useWorklist = true;
  bool timingOn = false;
  Timing tm;
  void resetTiming() { tm = Timing(); }
  const Timing& timing() const { return tm; }
  double tick_() const {
    if (!timingOn)
      return 0.0;
    Kokkos::fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void addT_(double& acc, double t0) {
    if (!timingOn)
      return;
    Kokkos::fence();
    acc += std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
               .count() -
           t0;
  }
  I3 inner() const { return n_; }
  I3 extent() const { return e_; }
  int ghost() const { return g_; }
  SField kappa() const { return kappa_; }
  SField branch() const { return branch_; }

  // ---- tunables (all measured knobs, defaults are the literature values) ---------------------
  /// Wendland support width `d` of the PV fit, in cell units (Han et al. §5: 2.5 with S = 5).
  double weightWidth = kPvWeightWidth;
  /// Tolerance of the column monotonicity test (`hfColumnHeight`). Tight enough to reject a
  /// sub-grid blob inside a column, loose enough to ignore transport round-off.
  double monoTol = 1e-6;
  /// Wendland support width of the tier-2b height-position fit, in cell units.
  double ptWeightWidth = kPtWeightWidth;
  /// Minimum `n_p . n` for a stencil polygon to enter the PV fit (`pvFitAdd`).
  double cosMin = 0.2;
  /// **Wisp threshold on the interfacial predicate.** A cell counts as carrying an interface only
  /// while `eps < C < 1 - eps`. DEFAULT 0, which is `wyIsMixed` verbatim (`C > 0 && C < 1`) and
  /// therefore byte-identical to the V3 rung as gated.
  ///
  /// Why it exists (a V4 finding, WO-P). Weymouth-Yue leaves ROUND-OFF colour residue in cells the
  /// sweeps touched — measured on a static droplet after 10 steps: `C` down to `-3e-35` and some
  /// 5200 extra cells at `0 < C < 1e-30` on a 64^3 grid, i.e. more cells than the interface itself
  /// has. Those cells satisfy `wyIsMixed` and the cascade dutifully produces a curvature for them,
  /// **from a PLIC polygon of area ~0**: measured `|kappa|` up to **1.2e+08** where the physical
  /// value is 0.125. On its own that is only a wrong number in a field nobody was reading. Under
  /// V4 it is fatal: a face between such a cell and a REAL interfacial cell has `dC = O(1)` and a
  /// face curvature `(kappa_real + 1e8)/2`, so the surface-tension force at that one face is eight
  /// orders too large. Measured on a 32^3 static droplet, unguarded: `max|u|` 4.5e-4 at step 1 ->
  /// **2.7e-1** by step 20, and at 96^3 the run trips the Weymouth-Yue CFL cap outright, while the
  /// SAME run with the curvature frozen at its (clean) initial value stays bounded at 4e-3.
  ///
  /// `Solver::setSurfaceTension` therefore sets this to 1e-8 — the same wisp threshold
  /// `WyAdvector::diagnostics` already reports against. It changes nothing on an exact colour field
  /// (no cell of any gated V3 case lies in the band) and it is the guard the literature assumes
  /// when it says wisp cleanup is unavoidable once surface tension is on (VOF_PLAN §6, after
  /// Arrufat et al. 2021).
  double interfaceEps = 0.0;
  /// DIAGNOSTIC ONLY — disable tiers 1 and 2 so every interfacial cell goes to the PV fit. Used by
  /// the ctest to measure the fallback branch on its own; never enable it in production.
  bool debugForceFallback = false;
  /// DIAGNOSTIC ONLY — disable tier 2 (the non-preferred column directions), so the branch census
  /// separates "the preferred direction alone" from "the direction cascade".
  bool debugSingleDirection = false;
  /// **Tier 2b, and it ships OFF — a measured decision, see the WO-O findings entry.**
  ///
  /// The mixed height function (a paraboloid through the interface positions of whichever columns
  /// closed, `PtFit`) is implemented and correct, and on the exact-fraction sphere it takes over
  /// exactly the 19.5-59.6 % of interfacial cells that tier 1 cannot serve. Measured on that
  /// sphere at 16/32/64 with everything else identical:
  ///
  ///     tier 2b ON   L1 2.83e-2 / 7.27e-3 / 4.21e-3   max 6.08e-2 / 4.64e-2 / 6.07e-2
  ///     tier 2b OFF  L1 3.03e-2 / 5.94e-3 / 1.32e-3   max 5.01e-2 / 1.32e-2 / 3.79e-3
  ///     order        L1 1.37 vs 2.26                  max 0.00 vs 1.86
  ///
  /// i.e. it destroys the convergence of the MAX error, on the very cells the PV fallback handles
  /// at second order. The mechanism is structural, not a parameter choice (four Wendland widths
  /// from 1.5 to 6.0 cells were swept; none converges in the max): **its data set is the columns
  /// the height function could close, which is a slope-SELECTED subset.** At a cell whose normal
  /// sits near an octant diagonal the corner columns on the steep side are exactly the ones that
  /// fail, so the surviving points sample the interface asymmetrically about the target and the
  /// quadratic fit acquires a lever-arm bias. That selection depends on the normal direction and
  /// not on h, so the bias is scale invariant — which is precisely the flat max-error curve above.
  /// The PV fallback is immune because a PLIC polygon exists in every mixed cell whatever the
  /// slope, so its 5^3 data set is symmetric.
  ///
  /// Turning it on is therefore a measurement, not a configuration. Kept because it is the WO's
  /// specified tier 2 and because the mechanism above is worth being able to re-measure.
  bool useMixedHeightFit = false;

  /// Compute the curvature over the inner region from a colour field on the SAME extended block.
  /// `c`'s ghosts must be valid on entry (the caller's exchange); nothing here communicates.
  Stats compute(SField c) {
    if (!ready())
      throw std::runtime_error("peclet::flow::vof::VofCurvature::compute: init() not called");
    const double t0 = tick_();
    if (useWorklist)
      compact(c);
    addT_(tm.compact, t0);
    const double t1 = tick_();
    reconstructPlanes(c);
    addT_(tm.planes, t1);
    const double t2 = tick_();
    heightPass(c);
    addT_(tm.height, t2);
    const double t3 = tick_();
    fallbackPass(c);
    addT_(tm.fallback, t3);
    const double t4 = tick_();
    const Stats s = census();
    addT_(tm.census, t4);
    ++tm.calls;
    return s;
  }

  /// The two compaction scans: `listG_` = every interfacial cell of the GROWN region (what
  /// `reconstructPlanes` reconstructs), `listI_` = every interfacial cell of the INNER region
  /// (what the cascade runs on).  Same predicate as the passes, so the compacted and dense paths
  /// visit exactly the same cells.
  void compact(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    const double ieps = interfaceEps;
    {
      const int rx = n.x + 2 * gr, ry = n.y + 2 * gr, rz = n.z + 2 * gr;
      const long region = (long)rx * ry * rz;
      LField list = listG_;
      long cnt = 0;
      Kokkos::parallel_scan(
          "vof::curv::compactG", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
            const int ix = (int)(r % rx);
            const int iy = (int)((r / rx) % ry);
            const int iz = (int)(r / ((long)rx * ry));
            const long i = L3(g - gr + ix, g - gr + iy, g - gr + iz, e);
            if (vofIsInterface(c(i), ieps)) {
              if (final)
                list(upd) = i;
              ++upd;
            }
          },
          cnt);
      Kokkos::fence();
      nG_ = cnt;
    }
    {
      const long region = (long)n.x * n.y * n.z;
      const int nx = n.x, ny = n.y;
      LField list = listI_;
      long cnt = 0;
      Kokkos::parallel_scan(
          "vof::curv::compactI", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
            const int ix = (int)(r % nx);
            const int iy = (int)((r / nx) % ny);
            const int iz = (int)(r / ((long)nx * ny));
            const long i = L3(g + ix, g + iy, g + iz, e);
            if (vofIsInterface(c(i), ieps)) {
              if (final)
                list(upd) = i;
              ++upd;
            }
          },
          cnt);
      Kokkos::fence();
      nI_ = cnt;
    }
  }

  // The four passes below are PUBLIC only because nvcc forbids an extended __host__ __device__
  // lambda inside a private or protected member function ("The enclosing parent function ... cannot
  // have private or protected access within its class"). They are implementation detail; call
  // `compute()`. `WyAdvector` is public for the same reason.
  //
  /// PLIC planes over the inner region grown by `kPvHalf` — every cell the 5^3 fit can read.
  /// Reads colour at +/-1 around those, i.e. +/-(kPvHalf+1) = +/-3 overall.
  void reconstructPlanes(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_;
    const double ieps = interfaceEps;
    if (useWorklist) {
      // The same two things the dense kernel does, as two coherent kernels: a branchless zeroing
      // sweep (a pure store, bandwidth-bound) and the MYC reconstruction over the compacted list.
      // Every cell ends with exactly the value the dense kernel would have written.
      Kokkos::parallel_for(
          "vof::curv::planes_zero",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(
              SExec(), {g - gr, g - gr, g - gr},
              {g + n.x + gr, g + n.y + gr, g + n.z + gr}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            mx(i) = 0.0;
            my(i) = 0.0;
            mz(i) = 0.0;
            al(i) = 0.0;
          });
      LField list = listG_;
      Kokkos::parallel_for(
          "vof::curv::planes_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nG_),
          KOKKOS_LAMBDA(long t) { wyReconstructCell(c, list(t), sy, sz, mx, my, mz, al); });
      Kokkos::fence();
      return;
    }
    Kokkos::parallel_for(
        "vof::curv::planes",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g - gr, g - gr, g - gr},
                                                      {g + n.x + gr, g + n.y + gr, g + n.z + gr}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (!vofIsInterface(c(i), ieps)) {
            mx(i) = 0.0;
            my(i) = 0.0;
            mz(i) = 0.0;
            al(i) = 0.0;
            return;
          }
          wyReconstructCell(c, i, sy, sz, mx, my, mz, al);
        });
    Kokkos::fence();
  }

  /// Tiers 1 and 2. Writes `kappa`/`branch` for every inner cell; a cell the height functions
  /// cannot serve is left with `branch = -1` for the fallback pass.
  void heightPass(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long st[3] = {1, e_.x, static_cast<long>(e_.x) * e_.y};
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, kap = kappa_, br = branch_;
    const double mtol = monoTol, ptW = ptWeightWidth, ieps = interfaceEps;
    const bool forceFb = debugForceFallback, oneDir = debugSingleDirection,
               useFit = useMixedHeightFit;
    const long s0 = st[0], s1 = st[1], s2 = st[2];
    if (useWorklist) {
      // Reset every inner cell (branchless: a pure pair of stores), then run the cascade over the
      // compacted interfacial list.  A non-interfacial cell ends at kappa = 0, branch = kCurvNone
      // and an interfacial one at whatever the shared body writes -- the dense kernel's outcome,
      // cell for cell.
      Kokkos::parallel_for(
          "vof::curv::hf_reset",
          Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                        {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            kap(i) = 0.0;
            br(i) = static_cast<double>(kCurvNone);
          });
      LField list = listI_;
      Kokkos::parallel_for(
          "vof::curv::hf_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nI_),
          KOKKOS_LAMBDA(long t) {
            curvHeightCell(list(t), c, mx, my, mz, al, kap, br, s0, s1, s2, mtol, ptW, ieps,
                           forceFb, oneDir, useFit);
          });
      Kokkos::fence();
      return;
    }
    Kokkos::parallel_for(
        "vof::curv::hf",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          curvHeightCell(L3(x, y, z, e), c, mx, my, mz, al, kap, br, s0, s1, s2, mtol, ptW, ieps,
                         forceFb, oneDir, useFit);
        });
    Kokkos::fence();
  }

  /// Tier 3, over the cells the height pass could not serve. A plain guarded `parallel_for` rather
  /// than a compacted worklist: the branch is a minority at any usable resolution, so almost every
  /// warp exits at the guard, and the fallback's larger local state is confined to this kernel.
  void fallbackPass(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, kap = kappa_, br = branch_;
    const double dW = weightWidth, cmin = cosMin, ieps = interfaceEps;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    if (useWorklist) {
      // Tier 3 is a SUBSET of the interfacial cells (those tier 1/2 could not serve), so the
      // interfacial list already removes the whole empty domain; the branch guard inside the body
      // removes the rest.
      LField list = listI_;
      Kokkos::parallel_for(
          "vof::curv::pv_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nI_),
          KOKKOS_LAMBDA(long t) {
            curvFallbackCell(list(t), c, mx, my, mz, al, kap, br, sy, sz, gr, dW, cmin, ieps);
          });
      Kokkos::fence();
      return;
    }
    Kokkos::parallel_for(
        "vof::curv::pv",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          curvFallbackCell(L3(x, y, z, e), c, mx, my, mz, al, kap, br, sy, sz, gr, dW, cmin, ieps);
        });
    Kokkos::fence();
  }

  Stats census() const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField br = branch_;
    Stats s;
    Kokkos::parallel_reduce(
        "vof::curv::census",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& ni, long& n1, long& n2, long& n2b, long& n3,
                      long& n4, long& n5) {
          const int b = static_cast<int>(br(L3(x, y, z, e)));
          if (b == kCurvNone)
            return;
          ++ni;
          if (b == kCurvHf)
            ++n1;
          else if (b == kCurvHfMixed)
            ++n2;
          else if (b == kCurvHfFit)
            ++n2b;
          else if (b == kCurvPv)
            ++n3;
          else if (b == kCurvPvReduced)
            ++n4;
          else
            ++n5;
        },
        s.interfacial, s.hf, s.hfMixed, s.hfFit, s.pv, s.pvReduced, s.noEstimate);
    Kokkos::fence();
    return s;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 0;
  long len_ = 0;
  SField mx_, my_, mz_, alpha_, kappa_, branch_;
  LField listG_, listI_;   // WO-V9: the compacted interfacial-cell lists
  long nG_ = 0, nI_ = 0;   // ... and their lengths from the last compact()
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_CURVATURE_FIELD_HPP
