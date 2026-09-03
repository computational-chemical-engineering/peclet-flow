/// @file
/// @brief flow — WO-P3c: the view-level driver of the cascade-consistent interfacial area.
///
/// `vof/interface_area.hpp` holds the container-free kernels; this file is their block-walking
/// driver, exactly the split `curvature.hpp` / `curvature_field.hpp` uses. It produces one output
/// field on the colour advector's **g = 3** block:
///
///   * `area` — the interfacial area of each cell, in h^2 (0 in non-interfacial cells).
///
/// **Why this is a SIBLING of `VofCurvature` and not a second output of it.** Hard rule 1 of the
/// VoF campaign: never edit a validated kernel body. `VofCurvature::heightPass` /
/// `fallbackPass` are gated by rung V3's numbers; adding a write to them would put every one of
/// those numbers back in play. The cascade walk is therefore repeated here, calling the SAME
/// container-free kernels (`hfColumnHeight`, `curvFrame`, `PvFit`, `plicPolygon`) with the same
/// tolerances and the same tier order, so the two drivers agree on which tier serves a cell by
/// construction. The duplication is deliberate and is the cheaper half of the trade: a
/// consolidation (one pass emitting kappa AND area) belongs with the next re-validation of V3.
///
/// The reach is unchanged — tier 1/2 read a 3x3x7 colour patch, tier 3 a 5^3 of PLIC planes whose
/// MYC normals read +-1 around them, i.e. +-3 everywhere — so this needs no new halo and, having
/// no reduction in it except the census, is bitwise decomposition-independent by construction.
#ifndef PECLET_FLOW_VOF_INTERFACE_AREA_FIELD_HPP
#define PECLET_FLOW_VOF_INTERFACE_AREA_FIELD_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "mac_stencils.hpp"
#include "vof/advect_wy.hpp"
#include "vof/curvature.hpp"
#include "vof/curvature_field.hpp"  // vofIsInterface
#include "vof/interface_area.hpp"

namespace peclet::flow::vof {

/// Interfacial area per cell from the V3 curvature cascade's own geometry.
class VofInterfaceArea {
 public:
  /// Per-call census over the inner region (LOCAL; a distributed caller sums it).
  struct Stats {
    long interfacial = 0;
    long hf = 0;          ///< tier 1: HF in the preferred (largest |n_d|) direction
    long hfMixed = 0;     ///< tier 2a: HF in one of the other two directions
    long pv = 0;          ///< tier 3: PV paraboloid fit (full or reduced model)
    long noEstimate = 0;  ///< no cascade geometry at all — the cell keeps its MYC PLIC area
    double area = 0.0;    ///< the sum this rung exists to get right
  };

  void init(int nx, int ny, int nz, int ghost) {
    if (ghost < kHfColumn / 2)
      throw std::invalid_argument(
          "peclet::flow::vof::VofInterfaceArea: ghost width must be >= 3");
    if (nx < 1 || ny < 1 || nz < 1)
      throw std::invalid_argument("peclet::flow::vof::VofInterfaceArea: empty block");
    n_ = I3{nx, ny, nz};
    g_ = ghost;
    e_ = I3{nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    mx_ = SField("vof::area::mx", len_);
    my_ = SField("vof::area::my", len_);
    mz_ = SField("vof::area::mz", len_);
    alpha_ = SField("vof::area::alpha", len_);
    area_ = SField("vof::area::A", len_);
    branch_ = SField("vof::area::branch", len_);
  }

  bool ready() const { return area_.extent(0) != 0; }
  SField area() const { return area_; }
  SField branch() const { return branch_; }

  // ---- tunables: the SAME knobs and defaults the curvature cascade runs with ----------------
  double weightWidth = kPvWeightWidth;
  double monoTol = 1e-6;
  double cosMin = 0.2;
  /// The interfacial predicate's wisp threshold. The phase-change driver passes ITS OWN
  /// (`pcEffInterfaceEps`), because a cell this driver calls pure would get area 0 while
  /// `pcIsInterfacial` still gave it an `mdot` — the two classifications must be the same one.
  double interfaceEps = 0.0;

  Stats compute(SField c, int mode) {
    if (!ready())
      throw std::runtime_error("peclet::flow::vof::VofInterfaceArea::compute: init() not called");
    reconstructPlanes(c);
    heightPass(c, mode);
    fallbackPass(c, mode);
    return census();
  }

  // public for the same nvcc reason `VofCurvature`'s passes are.
  void reconstructPlanes(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_;
    const double ieps = interfaceEps;
    Kokkos::parallel_for(
        "vof::area::planes",
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

  /// Tiers 1 and 2a: the height function in the preferred direction, then the other two, with the
  /// identical column/consistency rules `VofCurvature::heightPass` uses. A cell that closes gets
  /// its area from the patch's own metric (or its normal); a cell that does not is left with
  /// `branch = -1` for the PV pass.
  void heightPass(SField c, int mode) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long st[3] = {1, e_.x, static_cast<long>(e_.x) * e_.y};
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, ar = area_, br = branch_;
    const double mtol = monoTol, ieps = interfaceEps;
    const int md = mode;
    Kokkos::parallel_for(
        "vof::area::hf",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          ar(i) = 0.0;
          if (!vofIsInterface(c(i), ieps)) {
            br(i) = static_cast<double>(kCurvNone);
            return;
          }
          const double am[3] = {Kokkos::fabs(mx(i)), Kokkos::fabs(my(i)), Kokkos::fabs(mz(i))};
          int ord[3] = {0, 1, 2};
          for (int p = 1; p < 3; ++p)
            for (int q = p; q > 0 && am[ord[q]] > am[ord[q - 1]]; --q) {
              const int t = ord[q];
              ord[q] = ord[q - 1];
              ord[q - 1] = t;
            }
          for (int t = 0; t < 3; ++t) {
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
                  ok = false;
                  break;
                }
                hh[p + 3 * q] = h;
              }
            if (!ok)
              continue;
            if (md == kAreaFootprint) {
              ar(i) = hfFootprintArea(hh);
            } else {
              double ns[3];
              hfSurfaceNormal(hh, d, ns);
              ar(i) = interfaceAreaFromNormal(md, mx(i), my(i), mz(i), al(i), c(i), ns);
            }
            br(i) = static_cast<double>(t == 0 ? kCurvHf : kCurvHfMixed);
            return;
          }
          br(i) = -1.0;
        });
    Kokkos::fence();
  }

  /// Tier 3: the PLIC-volumetric paraboloid fit, whose GRADIENT at the origin is the accurate
  /// normal here — the same fit whose second derivatives `paraboloidKappa` turns into curvature.
  void fallbackPass(SField c, int mode) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, ar = area_, br = branch_;
    const double dW = weightWidth, cmin = cosMin, ieps = interfaceEps;
    const int md = mode;
    Kokkos::parallel_for(
        "vof::area::pv",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (br(i) >= 0.0)
            return;
          const double m0 = mx(i), m1 = my(i), m2 = mz(i);
          const double n2 = m0 * m0 + m1 * m1 + m2 * m2;
          if (!(n2 > 0.0)) {
            ar(i) = 0.0;
            br(i) = static_cast<double>(kCurvNoEstimate);
            return;
          }
          const double invn = 1.0 / Kokkos::sqrt(n2);
          const double nn[3] = {m0 * invn, m1 * invn, m2 * invn};
          double t1[3], t2[3];
          curvFrame(nn, t1, t2);
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
                if (!vofIsInterface(c(j), ieps))
                  continue;
                const double off[3] = {static_cast<double>(ox), static_cast<double>(oy),
                                       static_cast<double>(oz)};
                pvFitAdd(fit, mx(j), my(j), mz(j), al(j), off, org, t1, t2, nn, dW, cmin);
              }
          double a[6];
          bool red = false;
          if (!pvFitSolve(fit, a, red)) {
            // no cascade geometry at all: keep the MYC PLIC area rather than emit a zero. Loud in
            // the census, exactly as `kCurvNoEstimate` is in the curvature.
            ar(i) = plicArea(m0, m1, m2, al(i));
            br(i) = static_cast<double>(kCurvNoEstimate);
            return;
          }
          double ns[3];
          pvSurfaceNormal(a, t1, t2, nn, ns);
          // The PV branch has no column and therefore no tiling footprint of its own; under
          // `kAreaFootprint` it takes variant B (the plane rebuilt on the paraboloid's normal),
          // which is the closest thing to it that a single cell can supply. The branch census
          // says how much of the total that is.
          ar(i) = interfaceAreaFromNormal(md == kAreaFootprint ? kAreaNormal : md, m0, m1, m2,
                                          al(i), c(i), ns);
          br(i) = static_cast<double>(red ? kCurvPvReduced : kCurvPv);
        });
    Kokkos::fence();
  }

  Stats census() const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField br = branch_, ar = area_;
    Stats s;
    Kokkos::parallel_reduce(
        "vof::area::census",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& ni, long& n1, long& n2, long& n3, long& n4,
                      double& acc) {
          const long i = L3(x, y, z, e);
          const int b = static_cast<int>(br(i));
          if (b == kCurvNone)
            return;
          ++ni;
          acc += ar(i);
          if (b == kCurvHf)
            ++n1;
          else if (b == kCurvHfMixed)
            ++n2;
          else if (b == kCurvPv || b == kCurvPvReduced)
            ++n3;
          else
            ++n4;
        },
        s.interfacial, s.hf, s.hfMixed, s.pv, s.noEstimate, s.area);
    Kokkos::fence();
    return s;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 0;
  long len_ = 0;
  SField mx_, my_, mz_, alpha_, area_, branch_;
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_INTERFACE_AREA_FIELD_HPP
