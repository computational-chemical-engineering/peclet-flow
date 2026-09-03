/// @file
/// @brief flow — WO-P3c: the interfacial AREA taken from the CURVATURE cascade's geometry.
///
/// **Container-free by contract**, exactly like `plic.hpp` and `phase_change.hpp`: scalars and
/// small local arrays only. The view-level driver (the block walk, the height columns, the PV
/// fit) is `vof/interface_area_field.hpp`, the same split `curvature.hpp` / `curvature_field.hpp`
/// already uses.
///
/// ## Why this file exists (the measurement, WO-P3b)
///
/// Rung P0/P1 takes the interfacial area of a cell from the PLIC polygon of the **MYC** normal,
/// `A = |m|_2 dV/dalpha` (`vof::plicArea`). On a PLANE that is exact to the last bit — which is
/// every gate P0a/P0b/P1/P2 runs on. On a SPHERE it is **5.5-9.3 % LOW at R = 4...28 and does not
/// converge** (a-priori probe on exact fractions; marching cubes on the same colour field returns
/// 4 pi R^2 to 0.4 %). `plicArea` is *linear* in the normal, and rung V0 measured MYC's normal
/// error as non-convergent (order 0.83, ~28 % of mixed cells taking the Youngs branch), so a
/// non-convergent normal gives a non-convergent area. That deficit is exactly the P3 Scriven
/// growth-rate miss: the flux PER UNIT AREA is within +-0.5 % of Scriven's exact `mdot` while the
/// bubble grows as `int mdot dA`, and `beta_eff` sits 2.5 % low at every mesh.
///
/// The cure is to take `A_Gamma` from the SAME geometry the curvature comes from — the height
/// function on tiers 1/2 and the PLIC-volumetric paraboloid on tier 3 (`core/vof/curvature.hpp`).
/// Both are built from column sums / polygon volumes and never differentiate a normal, which is
/// precisely why the V3 cascade converges where MYC does not. It is also the *consistent* choice:
/// `kappa` and `A_Gamma` then describe one surface.
///
/// ## The two constructions, and what each one assumes
///
/// Write the interface locally as the graph `x_d = f(x_d1, x_d2)` over the column direction `d`.
/// The area of the piece inside one cell is `int_R sqrt(1 + f_x^2 + f_y^2)` over that piece's
/// FOOTPRINT `R` in the transverse plane, and the footprints of the cells of one column tile the
/// unit square exactly. So an area needs two things: a metric (a slope) and a footprint.
///
/// * **`kAreaMetric` (variant A, the default).** Keep the PLIC polygon's own FOOTPRINT — the
///   projection of the reconstructed polygon on the transverse plane, `A_PLIC |n_d|`, which is
///   what makes the cells of a column tile — and replace only the SLOPE:
///
///       A = A_PLIC(m_MYC, alpha) * |n_MYC . e_d| * sqrt(1 + h_x^2 + h_y^2) .
///
///   The metric is second order on a sphere (it is the height function's own, the same central
///   differences `hfPatchKappa` differentiates once more), and the footprint carries the exact
///   cell volume through `alpha`. Equivalently: `A = A_PLIC * |n_MYC.e_d| / |n*.e_d|`, i.e. the
///   MYC area with its slope factor divided out and the accurate one put back.
/// * **`kAreaNormal` (variant B).** Rebuild the whole plane with the accurate normal and the
///   cell's own colour: `A = plicArea(n*, plicAlpha(n*, C))`. No footprint bookkeeping at all —
///   the plane shift does it — but the polygon is the one a plane with THAT normal cuts, which on
///   a curved interface is not the one the transport actually carries.
///
/// Both are exact on a plane (there `h_x, h_y` are exact and `n* = n_MYC`), which is what keeps
/// rungs P0-P2 where they were.
#ifndef PECLET_FLOW_VOF_INTERFACE_AREA_HPP
#define PECLET_FLOW_VOF_INTERFACE_AREA_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "vof/curvature.hpp"  // core: hfColumnHeight, curvFrame, PvFit, paraboloidKappa, ...
#include "vof/phase_change.hpp"  // plicArea
#include "vof/plic.hpp"

namespace peclet::flow::vof {

/// Which construction turns the cascade's normal into an area. Exposed because the choice is a
/// MEASUREMENT (see the sphere probe in the WO-P3c findings), not a taste.
enum InterfaceAreaMode : int {
  kAreaPlic = 0,    ///< rung P0/P1: `plicArea` on the MYC normal. The old numbers.
  kAreaMetric = 1,  ///< variant A: the PLIC footprint x the cascade's slope. DEFAULT.
  kAreaNormal = 2,  ///< variant B: `plicArea` rebuilt on the cascade's normal.
};

/// The height function's area element `sqrt(1 + h_x^2 + h_y^2)`, with **the same central
/// differences `hfPatchKappa` uses** — so the metric and the curvature are two derivatives of one
/// interpolant, not two independent estimates.
///
/// `h` is the 3x3 patch of heights indexed `p + 3 q` exactly as `hfPatchKappa` takes it (`p` along
/// the first transverse axis `d1 = (d+1)%3`, `q` along `d2 = (d+2)%3`). The heights are SIGNED
/// (they grow with liquid whichever end the liquid is at) so the interface position is
/// `orient * h`; the area element is even in that sign and needs no orientation.
KOKKOS_INLINE_FUNCTION double hfAreaElement(const double h[9]) {
  const double hx = 0.5 * (h[2 + 3 * 1] - h[0 + 3 * 1]);
  const double hy = 0.5 * (h[1 + 3 * 2] - h[1 + 3 * 0]);
  return Kokkos::sqrt(1.0 + hx * hx + hy * hy);
}

/// The unit surface normal implied by the SAME 3x3 patch of heights, in world components.
/// `x_d = orient h(x_d1, x_d2)` has normal `(-h_x, -h_y, 1)` in the frame `(d1, d2, d)`, up to a
/// global sign that no consumer here cares about (`plicArea` is invariant under `m -> -m` with
/// `alpha` re-solved from the same colour: the two planes are point reflections of each other
/// through the cell centre, an isometry of the cube — the identity `V(a) + V(1-a) = 1` of
/// `plicArea`'s derivation).
KOKKOS_INLINE_FUNCTION void hfSurfaceNormal(const double h[9], int d, double m[3]) {
  const double hx = 0.5 * (h[2 + 3 * 1] - h[0 + 3 * 1]);
  const double hy = 0.5 * (h[1 + 3 * 2] - h[1 + 3 * 0]);
  const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
  const double inv = 1.0 / Kokkos::sqrt(1.0 + hx * hx + hy * hy);
  m[d1] = -hx * inv;
  m[d2] = -hy * inv;
  m[d] = inv;
}

/// The unit normal of the fitted paraboloid `z = a0 + a1 x + a2 y + ...` AT THE ORIGIN of its own
/// frame (the target cell's PLIC centroid), in world components: `(-a1, -a2, 1)` rotated by
/// `(t1, t2, nn)` and normalized. This is the tier-3 twin of `hfSurfaceNormal` and it is the same
/// derivative of the same fit `paraboloidKappa` differentiates twice.
KOKKOS_INLINE_FUNCTION void pvSurfaceNormal(const double a[6], const double t1[3],
                                            const double t2[3], const double nn[3], double m[3]) {
  const double inv = 1.0 / Kokkos::sqrt(1.0 + a[1] * a[1] + a[2] * a[2]);
  for (int i = 0; i < 3; ++i)
    m[i] = (-a[1] * t1[i] - a[2] * t2[i] + nn[i]) * inv;
}

/// Turn an accurate unit normal `ns` into the cell's interfacial area, given the cell's MYC plane
/// (`m`, `alpha`) and its colour `c`. `mode` selects the construction (see the file header).
///
/// The projection axis of `kAreaMetric` is the one `ns` is most aligned with — the column
/// direction on tiers 1/2 by construction, and the same choice on tier 3 — because the footprint
/// factor `|n.e_d|` is the one that stays farthest from zero there.
///
/// Returns `plicArea(m, alpha)` unchanged for `kAreaPlic`, so the old behaviour is one branch and
/// not a special case.
KOKKOS_INLINE_FUNCTION double interfaceAreaFromNormal(int mode, double mx, double my, double mz,
                                                      double alpha, double c, const double ns[3]) {
  const double aPlic = plicArea(mx, my, mz, alpha);
  if (mode == kAreaPlic)
    return aPlic;
  if (mode == kAreaNormal) {
    const double al = plicAlpha(ns[0], ns[1], ns[2], c);
    return plicArea(ns[0], ns[1], ns[2], al);
  }
  // kAreaMetric: keep the PLIC footprint, replace the slope.
  int d = 0;
  double best = Kokkos::fabs(ns[0]);
  if (Kokkos::fabs(ns[1]) > best) {
    best = Kokkos::fabs(ns[1]);
    d = 1;
  }
  if (Kokkos::fabs(ns[2]) > best) {
    best = Kokkos::fabs(ns[2]);
    d = 2;
  }
  if (!(best > 0.0))
    return aPlic;
  const double q = Kokkos::sqrt(mx * mx + my * my + mz * mz);
  if (!(q > 0.0))
    return aPlic;
  const double md = (d == 0) ? mx : ((d == 1) ? my : mz);
  return aPlic * (Kokkos::fabs(md) / q) / best;
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_INTERFACE_AREA_HPP
