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

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "vof/advect_wy.hpp"  // wyIsMixed, wyReconstructCell
#include "vof/curvature.hpp"

namespace peclet::flow::vof {

/// Curvature of the colour field on an extended (inner + ghost) block.
class VofCurvature {
 public:
  /// Per-call census of the cascade over this block's inner region (LOCAL; a distributed caller
  /// sums them itself — the driver stays MPI-free, exactly as `WyAdvector` does).
  struct Stats {
    long interfacial = 0;  ///< cells with 0 < C < 1 (the cells a curvature is asked for)
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
  }

  bool ready() const { return kappa_.extent(0) != 0; }
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
    reconstructPlanes(c);
    heightPass(c);
    fallbackPass(c);
    return census();
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
    Kokkos::parallel_for(
        "vof::curv::planes",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g - gr, g - gr, g - gr},
                                                      {g + n.x + gr, g + n.y + gr, g + n.z + gr}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (!wyIsMixed(c(i))) {
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
    const double mtol = monoTol, ptW = ptWeightWidth;
    const bool forceFb = debugForceFallback, oneDir = debugSingleDirection,
               useFit = useMixedHeightFit;
    Kokkos::parallel_for(
        "vof::curv::hf",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          kap(i) = 0.0;
          if (!wyIsMixed(c(i))) {
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
    const double dW = weightWidth, cmin = cosMin;
    Kokkos::parallel_for(
        "vof::curv::pv",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
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
                const long j = L3(x + ox, y + oy, z + oz, e);
                if (!wyIsMixed(c(j)))
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
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_CURVATURE_FIELD_HPP
