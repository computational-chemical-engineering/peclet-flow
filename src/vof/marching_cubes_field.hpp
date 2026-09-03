/// @file
/// @brief flow — WO-P3d: the block driver of the marching-tetrahedra interfacial area.
///
/// `vof/marching_cubes.hpp` holds the container-free kernels; this file walks the colour
/// advector's **g = 3** block with them, exactly the split `curvature.hpp` /
/// `curvature_field.hpp` and `interface_area.hpp` / `interface_area_field.hpp` already use. It is
/// a SIBLING of `VofInterfaceArea` and not a mode inside it (hard rule 1: `VofInterfaceArea`'s
/// passes carry WO-P3c's gated numbers and are not touched here).
///
/// Output: one field, `area` — the interfacial area booked to each cell, in `h^2`.
///
/// ## Reach, and why this is bitwise under MPI
///
/// A cell is a corner of the 8 dual cubes whose corners span `+-1` around it; the planes those
/// corners carry come from a MYC normal that reads `+-1`, so the whole stencil is `+-2` — inside
/// the colour block's g = 3 with a layer to spare, i.e. **no new halo**. There is no reduction in
/// it but the census, and the deposit is a GATHER with a fixed loop order (the cell re-derives the
/// triangles of each of its 8 cubes and keeps its own share), never an atomic scatter — so the
/// result is decomposition-independent bit for bit, which is the WO-P01 lesson applied to a
/// quantity that is naturally a scatter.
#ifndef PECLET_FLOW_VOF_MARCHING_CUBES_FIELD_HPP
#define PECLET_FLOW_VOF_MARCHING_CUBES_FIELD_HPP

#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "mac_stencils.hpp"
#include "vof/advect_wy.hpp"
#include "vof/curvature_field.hpp"  // vofIsInterface
#include "vof/interface_area.hpp"   // InterfaceAreaMode
#include "vof/marching_cubes.hpp"
#include "vof/phase_change.hpp"  // pcCentreDistance, pcUnitNormal
#include "vof/plic.hpp"

namespace peclet::flow::vof {

/// Interfacial area per cell from the marching-tetrahedra surface of the `C = 1/2` level set
/// (or of the PLIC-reconstructed signed distance — `set_phase_change_area` modes 4-7).
class VofMcArea {
 public:
  struct Stats {
    long cells = 0;         ///< inner cells that received area
    long orphanCells = 0;   ///< ... of which are NOT interfacial under `interfaceEps`
    double area = 0.0;      ///< the sum this rung exists to get right
    double orphanArea = 0.0;  ///< ... the part of it the phase-change consumer would drop
  };

  void init(int nx, int ny, int nz, int ghost) {
    if (ghost < 2)
      throw std::invalid_argument("peclet::flow::vof::VofMcArea: ghost width must be >= 2");
    if (nx < 1 || ny < 1 || nz < 1)
      throw std::invalid_argument("peclet::flow::vof::VofMcArea: empty block");
    n_ = I3{nx, ny, nz};
    g_ = ghost;
    e_ = I3{nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    dist_ = SField("vof::mc::d", len_);
    nx_ = SField("vof::mc::nx", len_);
    ny_ = SField("vof::mc::ny", len_);
    nz_ = SField("vof::mc::nz", len_);
    area_ = SField("vof::mc::A", len_);
  }

  bool ready() const { return area_.extent(0) != 0; }
  SField area() const { return area_; }

  /// The interfacial predicate's wisp threshold — the phase-change driver passes ITS OWN
  /// (`pcEffInterfaceEps`), so a cell this driver calls pure is the same cell `pcIsInterfacial`
  /// calls pure.
  double interfaceEps = 0.0;

  /// `mode` is an `InterfaceAreaMode` in [kAreaMcColour, kAreaMcPlicSplit].
  Stats compute(SField c, int mode) {
    if (!ready())
      throw std::runtime_error("peclet::flow::vof::VofMcArea::compute: init() not called");
    if (mode < kAreaMcColour || mode > kAreaMcPlicSplit)
      throw std::runtime_error("peclet::flow::vof::VofMcArea::compute: mode out of range");
    const int src = (mode >= kAreaMcPlic) ? kMcSrcPlic : kMcSrcColour;
    const int dep = ((mode - kAreaMcColour) & 1) ? kMcDepositSplit : kMcDepositCentroid;
    planePass(c);
    areaPass(c, src, dep);
    return census(c);
  }

  // public for the same nvcc reason `VofCurvature`'s passes are.

  /// The gas-positive centre distance and unit normal of every cell's own PLIC plane, one layer
  /// beyond the inner region (the dual cubes' corners).
  void planePass(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField dd = dist_, mx = nx_, my = ny_, mz = nz_;
    const double ieps = interfaceEps;
    Kokkos::parallel_for(
        "vof::mc::planes",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g - 1, g - 1, g - 1},
                                                      {g + n.x + 1, g + n.y + 1, g + n.z + 1}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          dd(i) = 0.0;
          mx(i) = 0.0;
          my(i) = 0.0;
          mz(i) = 0.0;
          if (!vofIsInterface(c(i), ieps))
            return;
          double st[27];
          for (int kk = -1; kk <= 1; ++kk)
            for (int jj = -1; jj <= 1; ++jj)
              for (int ii = -1; ii <= 1; ++ii)
                st[plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
          double m[3];
          mycNormal(st, m);
          double un[3] = {0.0, 0.0, 0.0};
          if (!(pcUnitNormal(m[0], m[1], m[2], un) > 0.0))
            return;
          const double al = plicAlpha(m[0], m[1], m[2], c(i));
          dd(i) = pcCentreDistance(m[0], m[1], m[2], al);
          mx(i) = un[0];
          my(i) = un[1];
          mz(i) = un[2];
        });
    Kokkos::fence();
  }

  /// The gather: each inner cell walks the 8 dual cubes it is a corner of and keeps its share.
  void areaPass(SField c, int src, int dep) {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField dd = dist_, mx = nx_, my = ny_, mz = nz_, ar = area_;
    const double ieps = interfaceEps;
    Kokkos::parallel_for(
        "vof::mc::area",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i0 = L3(x, y, z, e);
          ar(i0) = 0.0;
          // Early-out: every dual cube this cell is a corner of has its corners inside the 3^3
          // neighbourhood, so with no sign change of (C - 1/2) there nothing can cross.
          bool anyLo = false, anyHi = false;
          for (int kk = -1; kk <= 1; ++kk)
            for (int jj = -1; jj <= 1; ++jj)
              for (int ii = -1; ii <= 1; ++ii) {
                const double cv = c(L3(x + ii, y + jj, z + kk, e));
                if (cv < 0.5)
                  anyLo = true;
                else
                  anyHi = true;
              }
          if (!(anyLo && anyHi))
            return;
          double acc = 0.0;
          for (int oz = -1; oz <= 0; ++oz)
            for (int oy = -1; oy <= 0; ++oy)
              for (int ox = -1; ox <= 0; ++ox) {
                McVertex v[8];
                for (int k = 0; k < 8; ++k) {
                  const int bx = k & 1, by = (k >> 1) & 1, bz = (k >> 2) & 1;
                  const long j = L3(x + ox + bx, y + oy + by, z + oz + bz, e);
                  const double cv = c(j);
                  v[k].psi = 0.5 - cv;
                  v[k].has = vofIsInterface(cv, ieps) &&
                             (mx(j) != 0.0 || my(j) != 0.0 || mz(j) != 0.0);
                  v[k].d = dd(j);
                  v[k].n[0] = mx(j);
                  v[k].n[1] = my(j);
                  v[k].n[2] = mz(j);
                }
                const int lc = (-ox) | ((-oy) << 1) | ((-oz) << 2);
                acc += mcCubeCornerArea(v, lc, src, dep);
              }
          ar(i0) = acc;
        });
    Kokkos::fence();
  }

  Stats census(SField c) const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField ar = area_;
    const double ieps = interfaceEps;
    Stats s;
    Kokkos::parallel_reduce(
        "vof::mc::census",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                      {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& nc, long& no, double& acc, double& oacc) {
          const long i = L3(x, y, z, e);
          const double a = ar(i);
          if (!(a > 0.0))
            return;
          ++nc;
          acc += a;
          if (!vofIsInterface(c(i), ieps)) {
            ++no;
            oacc += a;
          }
        },
        s.cells, s.orphanCells, s.area, s.orphanArea);
    Kokkos::fence();
    return s;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 0;
  long len_ = 0;
  SField dist_, nx_, ny_, nz_, area_;
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_MARCHING_CUBES_FIELD_HPP
