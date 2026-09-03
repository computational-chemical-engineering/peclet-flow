/// @file
/// @brief flow — WO-P3d: a JOINED interfacial area, from marching tetrahedra on the cell-centre
///        lattice.
///
/// **Container-free by contract**, exactly like `plic.hpp`, `interface_area.hpp` and
/// `phase_change.hpp`: scalars and small local arrays only. The block-walking driver is
/// `vof/marching_cubes_field.hpp`.
///
/// ## Why this file exists (the measurement, WO-P3c)
///
/// Every PER-CELL interfacial area — the PLIC polygon (`kAreaPlic`), the height function's metric
/// (`kAreaMetric`), the plane rebuilt on the cascade normal (`kAreaNormal`), the height function's
/// own footprint (`kAreaFootprint`) — is **first order in `h/R`** on a curved interface, and
/// WO-P3c proved it with two analytic controls that contain none of the code under test: an
/// exact-fraction circle with the EXACT radial normal and the analytic chord sums to −2.86 /
/// −1.47 / −1.42 / −0.84 % of `2 pi R` at R = 8/12/20/28, and the mode-3 footprint construction
/// with EXACT heights sums to −3.50 / −2.42 / −1.66 / −1.11 / −0.81 % at R = 8/12/20/28/40,
/// observed order **1.03**. The mechanism is not the normal, the fractions, the metric or the
/// footprint: a cell chooses its piece of the surface INDEPENDENTLY of its neighbours', so the
/// pieces overlap and gap at every cell face instead of JOINING, and the mismatch is `O(h^2 kappa)`
/// per face over `O(L/h)` faces, i.e. a relative `O(h/R)`.
///
/// The cure is a surface whose pieces join **by construction**. That is what an isosurface
/// triangulation is: one vertex per crossed lattice edge, shared by every cell that edge touches,
/// so the triangles form a closed watertight sheet and the sum of their areas is the area of ONE
/// surface rather than the sum of 6 % overlapping fragments. Marching cubes on the same colour
/// fields was already measured 3-10x closer (WO-P3c: −0.59 / −0.49 % on the cylinder rows where
/// every per-cell construction reads −1.4 … −1.6 %).
///
/// ## What is built here
///
/// **Marching TETRAHEDRA on the Kuhn 6-tet decomposition** of the dual cube (the 8 corners of a
/// dual cube are 8 CELL CENTRES). Two reasons over the 256-case cube table:
///
///  * it is branch-light and table-free — 4 corner signs give one of three topologies (nothing,
///    one triangle, one quad) and the whole case analysis is `popcount` — which is what makes it a
///    clean device kernel, and
///  * the Kuhn decomposition is **translation invariant**, so two neighbouring dual cubes split
///    their shared face on the SAME diagonal and the sheet is watertight with no ambiguous-face
///    rule at all (the classic MC hole).
///
/// ## The two things that are chosen by MEASUREMENT, not taste
///
/// **(1) What is interpolated along an edge** (`src`):
///  * `kMcSrcColour` — the raw `C = 1/2` level set, `t = psi_a/(psi_a - psi_b)` with
///    `psi = 1/2 - C`. This is the work order's design, and it is **REFUTED on a curved
///    interface**: +4.2 … +5.8 % of `4 pi R^2` over R = 4…28 with no convergence. The mechanism is
///    the combination with the tetrahedra: `C(d)` is the SZ piecewise CUBIC of the centre
///    distance, so linear interpolation of it is only accurate over a step short compared with a
///    cell, and Kuhn's tets interpolate along `sqrt(2)` face diagonals and a `sqrt(3)` body
///    diagonal. Every vertex on a long edge is misplaced along the normal, the sheet wrinkles, and
///    a wrinkle only ever ADDS area — hence a one-signed 5 % that does not converge. (Marching
///    CUBES, which interpolates only the 12 unit-length cube edges, reads −0.2 … +0.4 % on the
///    same fields — the probe prints it as the external cross-check. So the defect is the
///    combination, not either half alone.) It ships as the measured ablation.
///  * `kMcSrcPlic` — the zero of the PLIC-reconstructed signed distance. On ANY plane, tilted or
///    not, both endpoint cells' planes ARE that plane, so every vertex lands exactly on it and the
///    summed area is exact to round-off. This is the work order's named follow-on ("interpolate
///    the PLIC-reconstructed signed distance instead of C") and it is what the 0.1 % tilted-plane
///    gate needs.
///
///    *How* the two endpoint planes are combined is itself a measurement, and the two candidates
///    are three decades apart on the sphere probe. Averaging the two ROOTS reads
///    +0.50 / +0.42 / +0.14 / +0.08 % of `4 pi R^2` at R = 8/12/20/28; blending the two signed
///    distance FUNCTIONS, `Phi(s) = (1-s) phi_a(s) + s phi_b(s)` (a quadratic in `s`, solved
///    exactly), reads **+0.011 / +0.011 / +0.009 / +0.008 %**. Both are exact on a plane; the
///    blend wins because it weights each cell's plane by how NEAR the crossing is to it, so the
///    Kuhn tets' long edges (`sqrt(2)` face diagonals, `sqrt(3)` the body diagonal) stop
///    extrapolating a tangent plane most of a cell away from where it was fitted. The blend is
///    what ships.
///
/// **(2) Which cell a piece of the sheet belongs to** (`deposit`):
///  * `kMcDepositCentroid` — the whole triangle goes to the cell containing its centroid.
///  * `kMcDepositSplit` — the triangle is CLIPPED to each cell's own cube and each cell gets the
///    area of its piece. Both give the same SUM (the 8 cells of a dual cube tile it exactly), so
///    they are indistinguishable on the a-priori area probe and differ only in the per-cell
///    distribution the phase-change flux integral sees.
///
/// ## The deposit is a GATHER, never an atomic scatter
///
/// WO-P01's lesson: a bitwise-MPI phase change needs a fixed summation order. A triangle produced
/// inside a dual cube can land in any of that cube's 8 corner cells, which as a SCATTER means
/// atomics and a non-deterministic order. So the driver inverts it: a cell walks the **8 dual
/// cubes it is a corner of**, re-derives every triangle of each, and keeps only what belongs to
/// itself. The 8x redundancy is the price of determinism, and it is what makes np 1/2/4 bitwise
/// by construction rather than by tolerance.
#ifndef PECLET_FLOW_VOF_MARCHING_CUBES_HPP
#define PECLET_FLOW_VOF_MARCHING_CUBES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace peclet::flow::vof {

/// What the edge crossing interpolates.
enum McSource : int {
  kMcSrcColour = 0,  ///< the `C = 1/2` level set (the WO's design)
  kMcSrcPlic = 1,    ///< the zero of the PLIC-reconstructed signed distance (the smooth variant)
};

/// Which cell a triangle's area is booked to.
enum McDeposit : int {
  kMcDepositCentroid = 0,  ///< the whole triangle to the cell holding its centroid
  kMcDepositSplit = 1,     ///< clipped to each cell's cube, each cell gets its own piece
};

/// One dual-cube corner = one cell centre.
///
/// `psi` is the GAS-POSITIVE level value used for the sign test and for the colour interpolation
/// (`psi = 1/2 - C`). `d` is the gas-positive signed distance of the cell centre from that cell's
/// own PLIC plane (`vof::pcCentreDistance`) and `n` the unit gas-ward normal; `has` is false for a
/// cell that carries no plane (pure phase), whose `d`/`n` are then never read.
///
/// The sign conventions agree exactly: a plane through the cell centre gives `C = 1/2` by the PLIC
/// identity `V(alpha) + V(1 - alpha) = 1`, so `sign(d) == sign(1/2 - C)` for every interfacial
/// cell and the two sources classify the cube identically.
struct McVertex {
  double psi = 0.0;
  double d = 0.0;
  double n[3] = {0.0, 0.0, 0.0};
  bool has = false;
};

/// Local coordinates of dual-cube corner `k` (bit 0 = x, bit 1 = y, bit 2 = z).
KOKKOS_INLINE_FUNCTION void mcCornerPos(int k, double p[3]) {
  p[0] = static_cast<double>(k & 1);
  p[1] = static_cast<double>((k >> 1) & 1);
  p[2] = static_cast<double>((k >> 2) & 1);
}

/// The parameter `t` in [0,1] at which the level set cuts the edge from corner `a` to corner `b`.
///
/// `kMcSrcColour` is the plain linear interpolation of `psi`. `kMcSrcPlic` solves each endpoint
/// cell's own plane along the edge — `phi_a(t) = d_a + t (n_a . dv)` and
/// `phi_b(t) = d_b + (t-1)(n_b . dv)` — and averages the roots the two cells supply, which is what
/// makes a plane exact from both sides; it falls back to the `psi` interpolation when neither
/// endpoint carries a usable plane (two pure cells of opposite phase in contact, i.e. an interface
/// with no mixed cell at all).
KOKKOS_INLINE_FUNCTION double mcEdgeT(const McVertex& a, const McVertex& b, const double dv[3],
                                      int src) {
  if (src == kMcSrcPlic) {
    double acc = 0.0;
    int cnt = 0;
    if (a.has) {
      const double g = a.n[0] * dv[0] + a.n[1] * dv[1] + a.n[2] * dv[2];
      if (Kokkos::fabs(g) > 1e-12) {
        const double t = -a.d / g;
        if (t > -0.5 && t < 1.5) {
          acc += t;
          ++cnt;
        }
      }
    }
    if (b.has) {
      const double g = b.n[0] * dv[0] + b.n[1] * dv[1] + b.n[2] * dv[2];
      if (Kokkos::fabs(g) > 1e-12) {
        const double t = 1.0 - b.d / g;
        if (t > -0.5 && t < 1.5) {
          acc += t;
          ++cnt;
        }
      }
    }
    if (cnt > 0) {
      double t = acc / static_cast<double>(cnt);
      if (a.has && b.has) {
        // Phi(s) = (1-s) phi_a(s) + s phi_b(s): the linear BLEND of the two cells' own signed
        // distance functions, which is exact on a plane and weights each cell's plane by how near
        // the crossing is to it (a quadratic in s).
        const double ga = a.n[0] * dv[0] + a.n[1] * dv[1] + a.n[2] * dv[2];
        const double gb = b.n[0] * dv[0] + b.n[1] * dv[1] + b.n[2] * dv[2];
        const double A = gb - ga, B = (ga - gb) + (b.d - a.d), C0 = a.d;
        double s2 = -1.0;
        if (Kokkos::fabs(A) < 1e-12 * (Kokkos::fabs(B) + 1e-30)) {
          if (Kokkos::fabs(B) > 0.0)
            s2 = -C0 / B;
        } else {
          const double disc = B * B - 4.0 * A * C0;
          if (disc >= 0.0) {
            const double sq = Kokkos::sqrt(disc);
            const double r1 = (-B + sq) / (2.0 * A), r2 = (-B - sq) / (2.0 * A);
            const bool o1 = (r1 >= 0.0 && r1 <= 1.0), o2 = (r2 >= 0.0 && r2 <= 1.0);
            if (o1 && !o2) s2 = r1;
            else if (o2 && !o1) s2 = r2;
            else if (o1 && o2) s2 = (Kokkos::fabs(r1 - t) < Kokkos::fabs(r2 - t)) ? r1 : r2;
          }
        }
        if (s2 >= 0.0 && s2 <= 1.0)
          t = s2;
      }
      return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    }
  }
  const double den = a.psi - b.psi;
  if (!(Kokkos::fabs(den) > 0.0))
    return 0.5;
  const double t = a.psi / den;
  return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

/// Twice the area of the triangle `(p0, p1, p2)` — i.e. `|(p1-p0) x (p2-p0)|`.
KOKKOS_INLINE_FUNCTION double mcTwiceArea(const double p0[3], const double p1[3],
                                          const double p2[3]) {
  const double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
  const double vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
  const double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  return Kokkos::sqrt(cx * cx + cy * cy + cz * cz);
}

/// Area of a PLANAR polygon of `n` vertices, `1/2 |sum p_i x p_{i+1}|`.
KOKKOS_INLINE_FUNCTION double mcPolygonArea(const double p[8][3], int n) {
  if (n < 3)
    return 0.0;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1 == n) ? 0 : i + 1;
    sx += p[i][1] * p[j][2] - p[i][2] * p[j][1];
    sy += p[i][2] * p[j][0] - p[i][0] * p[j][2];
    sz += p[i][0] * p[j][1] - p[i][1] * p[j][0];
  }
  return 0.5 * Kokkos::sqrt(sx * sx + sy * sy + sz * sz);
}

/// One Sutherland-Hodgman clip of a convex polygon against the half-space
/// `keepAbove ? x_axis >= c : x_axis <= c`. Returns the new vertex count.
KOKKOS_INLINE_FUNCTION int mcClipHalf(double p[8][3], int n, int axis, double c, bool keepAbove) {
  double q[8][3];
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1 == n) ? 0 : i + 1;
    const double si = keepAbove ? (p[i][axis] - c) : (c - p[i][axis]);
    const double sj = keepAbove ? (p[j][axis] - c) : (c - p[j][axis]);
    const bool ini = si >= 0.0, inj = sj >= 0.0;
    if (ini && m < 8) {
      q[m][0] = p[i][0];
      q[m][1] = p[i][1];
      q[m][2] = p[i][2];
      ++m;
    }
    if (ini != inj && m < 8) {
      const double den = si - sj;
      const double t = (Kokkos::fabs(den) > 0.0) ? si / den : 0.0;
      q[m][0] = p[i][0] + t * (p[j][0] - p[i][0]);
      q[m][1] = p[i][1] + t * (p[j][1] - p[i][1]);
      q[m][2] = p[i][2] + t * (p[j][2] - p[i][2]);
      ++m;
    }
  }
  for (int i = 0; i < m; ++i) {
    p[i][0] = q[i][0];
    p[i][1] = q[i][1];
    p[i][2] = q[i][2];
  }
  return m;
}

/// The area of the part of one triangle inside corner `k`'s octant of the dual cube.
KOKKOS_INLINE_FUNCTION double mcClipToCorner(const double p0[3], const double p1[3],
                                             const double p2[3], int k) {
  double poly[8][3];
  for (int d = 0; d < 3; ++d) {
    poly[0][d] = p0[d];
    poly[1][d] = p1[d];
    poly[2][d] = p2[d];
  }
  int n = 3;
  for (int d = 0; d < 3 && n >= 3; ++d)
    n = mcClipHalf(poly, n, d, 0.5, ((k >> d) & 1) != 0);
  return mcPolygonArea(poly, n);
}

/// The corner in `mask` nearest the point `p`; the lowest index breaks a tie. `mask == 0` returns
/// -1.
KOKKOS_INLINE_FUNCTION int mcNearestMasked(const double p[3], int mask) {
  int best = -1;
  double bd = 0.0;
  for (int k = 0; k < 8; ++k) {
    if (!((mask >> k) & 1))
      continue;
    double q[3];
    mcCornerPos(k, q);
    const double dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (best < 0 || d2 < bd) {
      best = k;
      bd = d2;
    }
  }
  return best;
}

/// **The retarget rule.** A triangle (or a clipped piece of one) can land in a cell the wisp
/// predicate calls PURE — measured: exactly the 12 axis-tangent "pole" cells of a sphere, where
/// the cell just outside the interface is exactly `C = 1` while the `C = 1/2` sheet still passes
/// through its cube (1.5 h^2, i.e. 0.19 % of the area at R = 8). The phase-change consumer reads
/// `A` only on interfacial cells, so that area would be SILENTLY DROPPED from `int mdot dA` —
/// and, worse, the drop partly cancels the sheet's own error, which is exactly the kind of
/// cancellation this campaign has twice been caught by. So a piece whose octant's cell carries no
/// interface is booked to the nearest cell that does. `mask` is the 8-bit interfacial census of
/// the cube's corners; `mask == 0` (an interface with no mixed cell at all) keeps the geometric
/// target.
KOKKOS_INLINE_FUNCTION int mcRetarget(int k, int mask) {
  if (mask == 0 || ((mask >> k) & 1))
    return k;
  double q[3];
  mcCornerPos(k, q);
  const int t = mcNearestMasked(q, mask);
  return t < 0 ? k : t;
}

/// The part of one triangle that belongs to local corner `lc` of the dual cube.
///
/// `kMcDepositCentroid` books the whole triangle to the corner nearest its centroid (a cube's 8
/// corner OCTANTS partition it, so exactly one corner claims it); `kMcDepositSplit` clips the
/// triangle to that corner's octant `[0,1/2] x ...` and returns the clipped area. Summed over the
/// 8 corners both rules return the triangle's own area exactly — including the retargeted pieces,
/// which are moved between corners and never lost.
KOKKOS_INLINE_FUNCTION double mcTriangleToCorner(const double p0[3], const double p1[3],
                                                 const double p2[3], int lc, int deposit,
                                                 int mask) {
  if (deposit == kMcDepositCentroid) {
    double ctr[3];
    for (int d = 0; d < 3; ++d)
      ctr[d] = (p0[d] + p1[d] + p2[d]) * (1.0 / 3.0);
    int corner = 0;
    for (int d = 0; d < 3; ++d)
      if (ctr[d] >= 0.5)
        corner |= 1 << d;
    if (mask != 0 && !((mask >> corner) & 1)) {
      const int t = mcNearestMasked(ctr, mask);
      if (t >= 0)
        corner = t;
    }
    return (corner == lc) ? 0.5 * mcTwiceArea(p0, p1, p2) : 0.0;
  }
  double acc = 0.0;
  if (mask == 0 || ((mask >> lc) & 1))
    acc += mcClipToCorner(p0, p1, p2, lc);
  if (mask != 0 && mask != 255)
    for (int k = 0; k < 8; ++k)
      if (!((mask >> k) & 1) && mcRetarget(k, mask) == lc)
        acc += mcClipToCorner(p0, p1, p2, k);
  return acc;
}

/// The interfacial area inside one dual cube that belongs to its local corner `lc`.
///
/// Kuhn's 6 tetrahedra of the cube, each cut by the level set into nothing, one triangle (a 1-vs-3
/// corner split) or a quad (2-vs-2, emitted as two triangles). Every vertex sits on a tet edge and
/// therefore on a cube edge, a face diagonal or the body diagonal — all of which are shared with
/// the neighbouring cubes in the same orientation, so the sheet is closed.
KOKKOS_INLINE_FUNCTION double mcCubeCornerArea(const McVertex v[8], int lc, int src, int deposit) {
  // Kuhn's decomposition along the body diagonal 0 -> 7. Corner index = x + 2y + 4z.
  const int TET[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
                         {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7}};
  int mask = 0;
  for (int k = 0; k < 8; ++k)
    if (v[k].has)
      mask |= 1 << k;
  // No mixed cell in the cube = no reconstructible interface, and nothing this rung's consumer
  // could carry an `mdot` on. This is the same statement the rest of the rung makes (a grid-
  // aligned sharp interface has no interfacial cell and therefore no PLIC area either), and it is
  // what keeps a PERIODIC SEAM — the C = 1 | C = 0 jump a half-space wraps into — from being
  // reported as interface.
  if (mask == 0)
    return 0.0;
  double acc = 0.0;
  for (int t = 0; t < 6; ++t) {
    int code = 0;
    for (int j = 0; j < 4; ++j)
      if (v[TET[t][j]].psi < 0.0)  // liquid
        code |= 1 << j;
    if (code == 0 || code == 15)
      continue;
    int nneg = 0;
    for (int j = 0; j < 4; ++j)
      if (code & (1 << j))
        ++nneg;
    double tri[4][3];  // up to 4 crossing points (the 2-vs-2 quad)
    int ntri = 0;
    if (nneg == 1 || nneg == 3) {
      // the lone corner, and the three edges leaving it
      const bool lonesign = (nneg == 1);
      int lone = -1;
      for (int j = 0; j < 4; ++j)
        if (((code & (1 << j)) != 0) == lonesign) {
          lone = j;
          break;
        }
      for (int j = 0; j < 4; ++j) {
        if (j == lone)
          continue;
        const int ca = TET[t][lone], cb = TET[t][j];
        double pa[3], pb[3];
        mcCornerPos(ca, pa);
        mcCornerPos(cb, pb);
        const double dv[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
        const double s = mcEdgeT(v[ca], v[cb], dv, src);
        for (int d = 0; d < 3; ++d)
          tri[ntri][d] = pa[d] + s * dv[d];
        ++ntri;
      }
      acc += mcTriangleToCorner(tri[0], tri[1], tri[2], lc, deposit, mask);
    } else {
      // 2 vs 2: the quad through the four edges that join the two pairs, in cyclic order
      int neg[2], pos[2], nn = 0, np = 0;
      for (int j = 0; j < 4; ++j)
        ((code & (1 << j)) ? neg[nn++] : pos[np++]) = j;
      const int pair[4][2] = {{neg[0], pos[0]}, {neg[0], pos[1]}, {neg[1], pos[1]}, {neg[1], pos[0]}};
      for (int q = 0; q < 4; ++q) {
        const int ca = TET[t][pair[q][0]], cb = TET[t][pair[q][1]];
        double pa[3], pb[3];
        mcCornerPos(ca, pa);
        mcCornerPos(cb, pb);
        const double dv[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
        const double s = mcEdgeT(v[ca], v[cb], dv, src);
        for (int d = 0; d < 3; ++d)
          tri[q][d] = pa[d] + s * dv[d];
      }
      acc += mcTriangleToCorner(tri[0], tri[1], tri[2], lc, deposit, mask);
      acc += mcTriangleToCorner(tri[0], tri[2], tri[3], lc, deposit, mask);
    }
  }
  return acc;
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_MARCHING_CUBES_HPP
