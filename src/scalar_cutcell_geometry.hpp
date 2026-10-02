// Scalar cut-cell geometry: the block kernels that build the per-cell record and the facet overlay
// of doc/scalar_ibm_design.md §2 / §3 (WO-2) on flow's G = 2 extended block.
//
// The geometry itself is core's container-free `peclet::core::scheme` (WO-1): the fan-tetrahedron
// PL model (`cutCellGeometryFanTet`, `fanFaceAperture`, `snapAperture`) and the probe ladder
// (`buildProbe`). This header only gathers the 15 PL samples from the cell-centred SDF, runs
// those kernels over the block and stores the results; the Solver side (ghost exchange, scene
// body ids, MPI census reduction, lazy build) is `flow_ibm_scalars_cutcell.hpp`.
//
// Conventions (§2.2, §3.5):
//   * a corner sample is the mean of its 8 surrounding cell-centred SDF values, summed in a FIXED
//     order keyed to the corner's lowest cell — ((s000+s100)+(s010+s110)) + ((s001+s101)+(s011+
//     s111)), times 0.125 — and a face-centre sample is 0.5 (s_lo + s_hi), so a shared corner or
//     face is the same bits from both cells and on every rank;
//   * the scalar apertures sax/say/saz are SNAPPED (both ends, 1e-3), UNGATED, and one value per
//     face: sa_a(i) is the -a face of cell i, computed from the SDF for every inner cell AND for
//     the high-side face at ghost index ext - G (never exchanged — SCALING_ISSUES #3);
//   * a fluid unknown is kappa > 1e-14 with a snapped aperture sum > 0; a sealed cell is kappa > 0
//     with every snapped aperture 0 (§2.4);
//   * the operator area vector of a facet is the snapped one, A^snap = sum_a A_a (sa_a+ - sa_a-)
//   e_a
//     (single facet; none if |A^snap| < 1e-14 max A_a), or the two PL group vectors with the snap
//     residual A^snap - areaPL added to the larger (two facets) (§2.3); the centroid is the PL one;
//   * facets are compacted in x-fastest inner-cell order (an exclusive scan), so the list is the
//     same on every backend and thread count.
//
// The ONLY inputs are the SDF (both ghost layers valid, §2.5) and the internal cell size h'. None
// of the pressure openness (ox_/oy_/oz_), `cs_` or `cutOwner_` is read or written here.
#ifndef PECLET_FLOW_SCALAR_CUTCELL_GEOMETRY_HPP
#define PECLET_FLOW_SCALAR_CUTCELL_GEOMETRY_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // CCField, CCConst, CCExec, CCMem, C3
#include "peclet/core/scheme/cut_cell_geometry.hpp"
#include "peclet/core/scheme/probe_flux.hpp"
#include "policy.hpp"

namespace peclet::flow::scg {

namespace cs = peclet::core::scheme;

/// kappa above which a cell with an open face is a fluid unknown (§2.4).
inline constexpr double kKappaUnknown = 1e-14;
/// |A^snap| below this times max_a A_a: no facet (§2.3).
inline constexpr double kFacetAreaRel = 1e-14;

/// The facet overlay (§3.5): SoA, facet index fastest. Offsets are LINEAR offsets on the block.
struct ScalarFacetOverlay {
  long n = 0;                                                     ///< facets on this rank
  Kokkos::View<long*, CCMem> cell;                                ///< extended-block linear index
  Kokkos::View<double*, CCMem> alpha;                             ///< |A_phi| / V
  Kokkos::View<double* [3], Kokkos::LayoutLeft, CCMem> normal;    ///< n_phi (into the fluid)
  Kokkos::View<double* [3], Kokkos::LayoutLeft, CCMem> centroid;  ///< x_phi - cell centre
  Kokkos::View<int*, CCMem> body;                                 ///< scene instance (0: raw SDF)
  Kokkos::View<double*, CCMem> sF;                                ///< fluid probe distance
  Kokkos::View<int* [8], Kokkos::LayoutLeft, CCMem> offF;         ///< fluid probe stencil offsets
  Kokkos::View<double* [8], Kokkos::LayoutLeft, CCMem> wF;  ///< fluid probe weights (0: unused)
  Kokkos::View<std::uint8_t*, CCMem> rungF;                 ///< ladder rung (cs::kProbeR*)
  long nCut = 0;                                            ///< cells carrying >= 1 facet
  Kokkos::View<long*, CCMem> cutCell;                       ///< their extended linear index
  Kokkos::View<int*, CCMem> cellFacetStart;                 ///< CSR, nCut + 1 entries
};

/// The geometry part of `diagnostics.scalar_census` (§8.2). Counts over inner cells / facets,
/// summed over ranks; `sealedVolume` in physical units.
struct ScalarCutCensus {
  long numUnknowns = 0, numCutCells = 0, numFacets = 0, numTwoSided = 0, numThinSolid = 0,
       numSealed = 0;
  double sealedVolume = 0.0;
  long rungs[4] = {0, 0, 0, 0};  ///< fluid R0, R1a, R1b, R2
};

/// The whole record: per-cell fields on the extended block (kappa and unknown haloed; the
/// apertures computed, never exchanged) plus the facet overlay and the census.
struct ScalarCutGeometry {
  long version = -1;  ///< the Solver's geometry version this record was built for
  CCField kappa, sax, say, saz, unknown;
  ScalarFacetOverlay fac;
  ScalarCutCensus census;
};

/// Corner sample: the 8 cells whose lowest is `i0`, in the fixed order of §2.2. (`V` is any 1-D
/// view of the extended block: the device field, or a host mirror in a test.)
template <class V>
KOKKOS_INLINE_FUNCTION double cornerSample(const V& s, long i0, long sy, long sz) {
  return (((s(i0) + s(i0 + 1)) + (s(i0 + sy) + s(i0 + 1 + sy))) +
          ((s(i0 + sz) + s(i0 + 1 + sz)) + (s(i0 + sy + sz) + s(i0 + 1 + sy + sz)))) *
         0.125;
}

/// The 15 PL samples of the cell at extended linear index i (corner index bx + 2 by + 4 bz,
/// faces -x,+x,-y,+y,-z,+z).
template <class V>
KOKKOS_INLINE_FUNCTION void cellSamples(const V& s, long i, long sy, long sz, double corner[8],
                                        double face[6], double& centre) {
  for (int c = 0; c < 8; ++c) {
    const long i0 =
        i + (long)((c & 1) - 1) + (long)(((c >> 1) & 1) - 1) * sy + (long)(((c >> 2) & 1) - 1) * sz;
    corner[c] = cornerSample(s, i0, sy, sz);
  }
  const long st[3] = {1, sy, sz};
  for (int a = 0; a < 3; ++a) {
    face[2 * a] = 0.5 * (s(i - st[a]) + s(i));
    face[2 * a + 1] = 0.5 * (s(i) + s(i + st[a]));
  }
  centre = s(i);
}

/// The snapped aperture of the -a face of the cell at extended index i (the fan of
/// `cs::fanFaceAperture`: tangent axes t1 < t2, loop c00, c10, c11, c01).
KOKKOS_INLINE_FUNCTION double lowFaceAperture(const CCConst& s, long i, int a, long sy, long sz) {
  const long st[3] = {1, sy, sz};
  const int t1 = (a == 0) ? 1 : 0, t2 = (a == 2) ? 1 : 2;
  // corner (u on t1, w on t2) of the -a face: lowest cell i - e_a + (u - 1) e_t1 + (w - 1) e_t2
  auto corner = [&](int u, int w) {
    return cornerSample(s, i - st[a] + (long)(u - 1) * st[t1] + (long)(w - 1) * st[t2], sy, sz);
  };
  const double cc = 0.5 * (s(i - st[a]) + s(i));
  return cs::snapAperture(
      cs::fanFaceAperture(corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1), cc));
}

/// Kernel 1 (§2.5): the snapped scalar apertures of every face of every inner cell, including the
/// high-side face at ghost index ext - G. Faces outside that set stay 0.
inline void buildApertures(CCField sax, CCField say, CCField saz, CCConst sdf, C3 e, int g) {
  CCExec space;
  const long sy = e.x, sz = (long)e.x * e.y;
  const int hx = e.x - g, hy = e.y - g, hz = e.z - g;  // the high-face ghost index per axis
  Kokkos::parallel_for(
      "peclet::flow::scg_apertures", MDRange3<CCExec>(space, {g, g, g}, {hx + 1, hy + 1, hz + 1}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (y < hy && z < hz)
          sax(i) = lowFaceAperture(sdf, i, 0, sy, sz);
        if (x < hx && z < hz)
          say(i) = lowFaceAperture(sdf, i, 1, sy, sz);
        if (x < hx && y < hy)
          saz(i) = lowFaceAperture(sdf, i, 2, sy, sz);
      });
  space.fence();
}

/// The snapped apertures of cell i in face order -x,+x,-y,+y,-z,+z.
template <class V>
KOKKOS_INLINE_FUNCTION void cellSnapped(const V& sax, const V& say, const V& saz, long i, long sy,
                                        long sz, double ap[6]) {
  ap[0] = sax(i);
  ap[1] = sax(i + 1);
  ap[2] = say(i);
  ap[3] = say(i + sy);
  ap[4] = saz(i);
  ap[5] = saz(i + sz);
}

/// The operator facets of one cell (§2.3) from its PL record and snapped apertures: returns the
/// count (0, 1 or 2) and, when `area`/`cen` are given, the area vectors and centroids.
KOKKOS_INLINE_FUNCTION int cellFacets(const cs::CutCellGeometry& g, const double ap[6],
                                      const double h[3], double area[2][3], double cen[2][3]) {
  double aSnap[3];
  cs::facetAreaVector(ap, h, aSnap);
  if (g.nFacet == 1) {
    double maxA = cs::cutCellFaceArea(0, h);
    maxA = cs::cutCellFaceArea(1, h) > maxA ? cs::cutCellFaceArea(1, h) : maxA;
    maxA = cs::cutCellFaceArea(2, h) > maxA ? cs::cutCellFaceArea(2, h) : maxA;
    const double mag =
        Kokkos::sqrt(aSnap[0] * aSnap[0] + aSnap[1] * aSnap[1] + aSnap[2] * aSnap[2]);
    if (mag < kFacetAreaRel * maxA)
      return 0;
    for (int d = 0; d < 3; ++d) {
      area[0][d] = aSnap[d];
      cen[0][d] = g.facetCentroid[0][d];
    }
    return 1;
  }
  if (g.nFacet == 2) {
    double m[2];
    for (int k = 0; k < 2; ++k)
      m[k] = g.facetArea[k][0] * g.facetArea[k][0] + g.facetArea[k][1] * g.facetArea[k][1] +
             g.facetArea[k][2] * g.facetArea[k][2];
    const int big = m[1] > m[0] ? 1 : 0;
    for (int k = 0; k < 2; ++k)
      for (int d = 0; d < 3; ++d) {
        area[k][d] = g.facetArea[k][d] + (k == big ? aSnap[d] - g.areaPL[d] : 0.0);
        cen[k][d] = g.facetCentroid[k][d];
      }
    return 2;
  }
  return 0;
}

/// Per-cell flags of the record pass, on the INNER grid (x fastest).
struct CellPass {
  Kokkos::View<std::int8_t*, CCMem> nfac;  ///< facets of the cell (0, 1, 2)
  Kokkos::View<std::int8_t*, CCMem> kind;  ///< cs::CutCellKind of the PL record
};

/// Kernel 2 (§2.5): kappa, the unknown flag and the per-cell facet count/kind on the inner cells.
/// Cells whose 15 samples share one sign are uniform (kappa 1 or 0, no facet) — the core kernel's
/// own fast path.
inline void buildCells(CCField kappa, CCField unknown, const CellPass& cp, CCConst sdf, CCConst sax,
                       CCConst say, CCConst saz, C3 e, int g, const double hp[3]) {
  CCExec space;
  const long sy = e.x, sz = (long)e.x * e.y;
  const int nx = e.x - 2 * g, ny = e.y - 2 * g;
  const double h0 = hp[0], h1 = hp[1], h2 = hp[2];
  auto nfac = cp.nfac;
  auto kind = cp.kind;
  Kokkos::parallel_for(
      "peclet::flow::scg_cells", MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const long ii = (long)(x - g) + (long)(y - g) * nx + (long)(z - g) * (long)nx * ny;
        const double h[3] = {h0, h1, h2};
        double corner[8], face[6], centre;
        cellSamples(sdf, i, sy, sz, corner, face, centre);
        const cs::CutCellGeometry gg = cs::cutCellGeometryFanTet(corner, face, centre, h);
        double ap[6];
        cellSnapped(sax, say, saz, i, sy, sz, ap);
        const double apSum = ((ap[0] + ap[1]) + (ap[2] + ap[3])) + (ap[4] + ap[5]);
        kappa(i) = gg.kappa;
        unknown(i) = (gg.kappa > kKappaUnknown && apSum > 0.0) ? 1.0 : 0.0;
        double area[2][3], cen[2][3];
        nfac(ii) = (std::int8_t)cellFacets(gg, ap, h, area, cen);
        kind(ii) = (std::int8_t)gg.kind;
      });
  space.fence();
}

/// Kernel 3 (§2.5): exclusive scans of the facet counts and of the cut-cell indicator in x-fastest
/// inner order; allocates the overlay and fills its geometric members (not body, not the probe).
inline void buildFacets(ScalarFacetOverlay& fo, const CellPass& cp, CCConst sdf, CCConst sax,
                        CCConst say, CCConst saz, C3 e, int g, const double hp[3]) {
  const long nx = e.x - 2 * g, ny = e.y - 2 * g, nz = e.z - 2 * g;
  const long nInner = nx * ny * nz;
  auto nfac = cp.nfac;
  Kokkos::View<long*, CCMem> fStart("peclet::flow::scg_fstart", nInner);
  Kokkos::View<long*, CCMem> cStart("peclet::flow::scg_cstart", nInner);
  long totF = 0, totC = 0;
  Kokkos::parallel_scan(
      "peclet::flow::scg_scan_facets", Kokkos::RangePolicy<CCExec>(0, nInner),
      KOKKOS_LAMBDA(const long ii, long& acc, const bool fin) {
        if (fin)
          fStart(ii) = acc;
        acc += (long)nfac(ii);
      },
      totF);
  Kokkos::parallel_scan(
      "peclet::flow::scg_scan_cut", Kokkos::RangePolicy<CCExec>(0, nInner),
      KOKKOS_LAMBDA(const long ii, long& acc, const bool fin) {
        if (fin)
          cStart(ii) = acc;
        acc += nfac(ii) > 0 ? 1 : 0;
      },
      totC);
  fo.n = totF;
  fo.nCut = totC;
  fo.cell = Kokkos::View<long*, CCMem>("peclet::flow::scg_cell", totF);
  fo.alpha = Kokkos::View<double*, CCMem>("peclet::flow::scg_alpha", totF);
  fo.normal = decltype(fo.normal)("peclet::flow::scg_normal", totF);
  fo.centroid = decltype(fo.centroid)("peclet::flow::scg_centroid", totF);
  fo.body = Kokkos::View<int*, CCMem>("peclet::flow::scg_body", totF);
  fo.sF = Kokkos::View<double*, CCMem>("peclet::flow::scg_sF", totF);
  fo.offF = decltype(fo.offF)("peclet::flow::scg_offF", totF);
  fo.wF = decltype(fo.wF)("peclet::flow::scg_wF", totF);
  fo.rungF = Kokkos::View<std::uint8_t*, CCMem>("peclet::flow::scg_rungF", totF);
  fo.cutCell = Kokkos::View<long*, CCMem>("peclet::flow::scg_cutCell", totC);
  fo.cellFacetStart = Kokkos::View<int*, CCMem>("peclet::flow::scg_cellFacetStart", totC + 1);

  CCExec space;
  const long sy = e.x, sz = (long)e.x * e.y;
  const double h0 = hp[0], h1 = hp[1], h2 = hp[2];
  const double vol = h0 * h1 * h2;
  auto fcell = fo.cell;
  auto falpha = fo.alpha;
  auto fnormal = fo.normal;
  auto fcen = fo.centroid;
  auto cutCell = fo.cutCell;
  auto cfs = fo.cellFacetStart;
  Kokkos::parallel_for(
      "peclet::flow::scg_facets", Kokkos::RangePolicy<CCExec>(space, 0, nInner),
      KOKKOS_LAMBDA(const long ii) {
        if (nfac(ii) == 0)
          return;
        const long x = ii % nx, y = (ii / nx) % ny, z = ii / (nx * ny);
        const long i = (x + g) + (y + g) * sy + (z + g) * sz;
        const double h[3] = {h0, h1, h2};
        double corner[8], face[6], centre;
        cellSamples(sdf, i, sy, sz, corner, face, centre);
        const cs::CutCellGeometry gg = cs::cutCellGeometryFanTet(corner, face, centre, h);
        double ap[6];
        cellSnapped(sax, say, saz, i, sy, sz, ap);
        double area[2][3], cen[2][3];
        const int nf = cellFacets(gg, ap, h, area, cen);
        const long f0 = fStart(ii);
        for (int k = 0; k < nf; ++k) {
          const long f = f0 + k;
          const double mag = Kokkos::sqrt(area[k][0] * area[k][0] + area[k][1] * area[k][1] +
                                          area[k][2] * area[k][2]);
          fcell(f) = i;
          falpha(f) = mag / vol;
          for (int d = 0; d < 3; ++d) {
            fnormal(f, d) = area[k][d] / mag;
            fcen(f, d) = cen[k][d];
          }
        }
        cutCell(cStart(ii)) = i;
        cfs(cStart(ii)) = (int)f0;
      });
  Kokkos::deep_copy(space, Kokkos::subview(cfs, totC), (int)totF);
  space.fence();
}

/// The block `Lookup` of `cs::buildProbe` for one facet's cell: unknown flags (haloed) at lattice
/// offsets, and the phase-signed SDF trilinear at an index-space point relative to the cell.
struct BlockLookup {
  CCConst sdf, unk;
  long base, sy, sz;
  double sign;  ///< +1 fluid, -1 solid
  KOKKOS_INLINE_FUNCTION bool unknown(int dx, int dy, int dz) const {
    return unk(base + (long)dx + (long)dy * sy + (long)dz * sz) > 0.5;
  }
  KOKKOS_INLINE_FUNCTION double phi(const double xi[3]) const {
    int b[3];
    double w[8];
    cs::trilinearStencil(xi, b, w);
    double v = 0.0;
    for (int k = 0; k < 8; ++k)
      v += w[k] * sdf(base + (long)(b[0] + (k & 1)) + (long)(b[1] + ((k >> 1) & 1)) * sy +
                      (long)(b[2] + (k >> 2)) * sz);
    return sign * v;
  }
};

/// Kernel 5 (§2.5, §3.3): the fluid probe ladder of every facet. `unknown` must be haloed (and
/// zeroed beyond non-periodic global faces) and `sdf` valid on both ghost layers.
inline void buildFluidProbes(ScalarFacetOverlay& fo, CCConst sdf, CCConst unknown, C3 e,
                             const double hp[3]) {
  CCExec space;
  const long sy = e.x, sz = (long)e.x * e.y;
  const double h0 = hp[0], h1 = hp[1], h2 = hp[2];
  auto fcell = fo.cell;
  auto fnormal = fo.normal;
  auto fcen = fo.centroid;
  auto sF = fo.sF;
  auto offF = fo.offF;
  auto wF = fo.wF;
  auto rungF = fo.rungF;
  Kokkos::parallel_for(
      "peclet::flow::scg_fluid_probes", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
      KOKKOS_LAMBDA(const long f) {
        const double h[3] = {h0, h1, h2};
        const double xw[3] = {fcen(f, 0), fcen(f, 1), fcen(f, 2)};
        const double n[3] = {fnormal(f, 0), fnormal(f, 1), fnormal(f, 2)};
        const BlockLookup L{sdf, unknown, fcell(f), sy, sz, 1.0};
        const cs::ProbeResult r = cs::buildProbe(xw, n, h, L);
        sF(f) = r.s;
        rungF(f) = (std::uint8_t)r.rung;
        for (int k = 0; k < 8; ++k) {
          const bool used = k < r.n;
          offF(f, k) =
              used ? (int)((long)r.off[k][0] + (long)r.off[k][1] * sy + (long)r.off[k][2] * sz) : 0;
          wF(f, k) = used ? r.w[k] : 0.0;
        }
      });
  space.fence();
}

/// Zero `f` on every ghost layer beyond face `face` (0..5 = -x,+x,-y,+y,-z,+z), full transverse
/// extent: the halo wraps periodically, so flags beyond a non-periodic global face are cleared.
inline void zeroGhostSlab(CCField f, C3 e, int g, int face) {
  CCExec space;
  const int a = face / 2, s = face % 2;
  const int ext[3] = {e.x, e.y, e.z};
  int lo[3] = {0, 0, 0}, hi[3] = {e.x, e.y, e.z};
  lo[a] = s ? ext[a] - g : 0;
  hi[a] = s ? ext[a] : g;
  const long sy = e.x, sz = (long)e.x * e.y;
  Kokkos::parallel_for(
      "peclet::flow::scg_zero_ghost",
      MDRange3<CCExec>(space, {lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]}),
      KOKKOS_LAMBDA(int x, int y, int z) { f((long)x + (long)y * sy + (long)z * sz) = 0.0; });
  space.fence();
}

/// The census of the record on this rank (not yet summed over ranks); `sealedVolume` in internal
/// volume units.
inline ScalarCutCensus localCensus(const ScalarFacetOverlay& fo, const CellPass& cp, CCConst kappa,
                                   CCConst unknown, CCConst sax, CCConst say, CCConst saz, C3 e,
                                   int g, double vol) {
  ScalarCutCensus c;
  CCExec space;
  const long sy = e.x, sz = (long)e.x * e.y;
  const int nx = e.x - 2 * g, ny = e.y - 2 * g;
  auto nfac = cp.nfac;
  auto kind = cp.kind;
  long nU = 0, nTwo = 0, nThin = 0, nSealed = 0;
  double vSealed = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::scg_census_cells",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, long& u, long& two, long& thin, long& sealed, double& vs) {
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const long ii = (long)(x - g) + (long)(y - g) * nx + (long)(z - g) * (long)nx * ny;
        u += unknown(i) > 0.5 ? 1 : 0;
        two += nfac(ii) == 2 ? 1 : 0;
        thin += (nfac(ii) == 2 && kind(ii) == (std::int8_t)cs::CutCellKind::thinSolid) ? 1 : 0;
        double ap[6];
        cellSnapped(sax, say, saz, i, sy, sz, ap);
        const double apSum = ((ap[0] + ap[1]) + (ap[2] + ap[3])) + (ap[4] + ap[5]);
        if (kappa(i) > 0.0 && apSum == 0.0) {
          ++sealed;
          vs += kappa(i) * vol;
        }
      },
      nU, nTwo, nThin, nSealed, vSealed);
  long r[4] = {0, 0, 0, 0};
  auto rungF = fo.rungF;
  for (int k = 0; k < 4; ++k) {
    long cnt = 0;
    Kokkos::parallel_reduce(
        "peclet::flow::scg_census_rungs", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
        KOKKOS_LAMBDA(const long f, long& acc) { acc += (int)rungF(f) == k ? 1 : 0; }, cnt);
    r[k] = cnt;
  }
  c.numUnknowns = nU;
  c.numCutCells = fo.nCut;
  c.numFacets = fo.n;
  c.numTwoSided = nTwo;
  c.numThinSolid = nThin;
  c.numSealed = nSealed;
  c.sealedVolume = vSealed;
  for (int k = 0; k < 4; ++k)
    c.rungs[k] = r[k];
  return c;
}

}  // namespace peclet::flow::scg

#endif  // PECLET_FLOW_SCALAR_CUTCELL_GEOMETRY_HPP
