/// @file
/// @brief flow — the sharp conservative viscous stress at a VoF interface on the staggered MAC
/// grid: band geometry (interfacial cells, virtual planes, edge owners), the per-sample PLIC
/// fractions, the kinked least-squares fit of the interface state (τ̂, D̂_nn), the implicit face
/// viscosity of the band edges and the explicit band terms.
///
/// Design: `doc/vof_sharp_viscous_stress.md` (§2, §5, §5.7, §7). Selected by
/// `set_vof_viscous_interface`, whose default "auto" resolves to it in scope (§9); "averaged"
/// never reaches this header, so it is bitwise the operator before it.
///
/// **Coordinates.** Solver block (ghost `G`), cell index `i = x + y·e.x + z·e.x·e.y`, cell `i`
/// spanning `[x, x+1] × [y, y+1] × [z, z+1]` in index space; `u_c(i)` sits on the −c face of cell
/// `i`. The colour and its PLIC planes live on the colour block (ghost `G + d`); a solver cell
/// `(x, y, z)` is colour cell `(x+d, y+d, z+d)`. Fractions are computed in index space (affine
/// invariant); normals, strains and stresses are physical in the solver's internal units, with
/// `ℓ_a = h_a/h_ref` and index velocity `v_a = ũ_a/ℓ_a` (`doc/full_viscous_stress.md` §3).
///
/// **Edge `E_ac(k)`.** The edge along b (the third axis) at `(x_a, x_c) = (k_a, k_c)`, in the
/// b-layer `k_b`: the edge shared by cells `k, k−s_a, k−s_c, k−s_a−s_c`. It is the −a edge of the
/// u_c control volume of cell `k` and the −c edge of the u_a control volume of cell `k`.
///
/// **Fixed offset order (§5.7).** Every tie and every sum over the cells of an edge enumerates them
/// by their offset, lexicographic in (dz, dy, dx), x fastest — on a block with ghost layers that is
/// the order of the LOCAL linear index, translation-invariant across a periodic seam and a rank
/// boundary (never the global index).
///
/// **The physical domain.** A cell beyond a non-periodic global face (a domain ghost) is never
/// interfacial: no rank owns it, so it has no fit (the fit runs on owned interfacial cells, §7) and
/// can be no edge's owner (§5.2: the owner is always in the τ̂ set). Its neighbour's face is no
/// virtual-plane face (§5.7 Q-B), and a link whose sample lies beyond the face is left out of the
/// fit (§5.3); a sample ON the face is kept, its ghost velocity is data.
#ifndef PECLET_FLOW_VOF_SHARP_VISCOUS_HPP
#define PECLET_FLOW_VOF_SHARP_VISCOUS_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>

#include "face_props.hpp"
#include "mac_cutcell.hpp"
#include "peclet/core/vof/plic_regions.hpp"
#include "policy.hpp"
#include "vof/curvature_field.hpp"  // vofIsInterface: THE interfacial predicate of the planes
#include "vof/plic.hpp"

namespace peclet::flow::vof {

/// Per-cell byte fields on the solver block: the band class and the fit-fallback flag.
using SvByte = Kokkos::View<std::uint8_t*, CCMem>;
using SvByteConst = Kokkos::View<const std::uint8_t*, CCMem>;

/// Band class of a cell. `kSvPlane`: the colour reconstruction gives it a plane. `kSvVirtual0 + 2a
/// + side`: a pure cell carrying the virtual plane on its −a (side 0) or +a (side 1) face.
constexpr std::uint8_t kSvNone = 0, kSvPlane = 1, kSvVirtual0 = 2;

/// The fit (§5.3) as items and rows (`SharpGeom::fitItem`): 27 block cells of 3 normal links, then
/// 144 edges of 2 shear links; 369 rows; 45 + 9 normal-equation entries (upper triangle + rhs).
constexpr int kSvNormalItems = 27, kSvItems = 27 + 144, kSvRows = 81 + 288, kSvEntries = 45 + 9;

/// The compacted interfacial cells of a fit (local linear indices), the device fit's worklist.
using SvList = Kokkos::View<long*, CCMem>;

/// The band geometry and the closure constants, by value into every kernel.
struct SharpGeom {
  int e[3] = {0, 0, 0};     ///< solver block extents
  long st[3] = {1, 0, 0};   ///< solver block strides
  long st3[3] = {1, 0, 0};  ///< colour block strides
  int d = 1;                ///< colour ghost − solver ghost
  CCConst col;              ///< colour C, colour block (ghosts valid)
  CCConst pm[3];            ///< reconstructed PLIC normals m (index space), colour block
  CCConst pa;               ///< reconstructed PLIC offsets α, colour block
  SvByteConst cls;          ///< band class, solver block
  int dlo[3] = {0, 0, 0};   ///< in-domain cells: dlo <= x_a < dhi (solver coordinates)
  int dhi[3] = {0, 0, 0};
  double eps = 0.0;           ///< interfacial threshold: the wisp guard (`set_vof_interface_eps`)
  double p0 = 0.0, p1 = 0.0;  ///< LinearMix μ(f) = p0 + p1·f (internal units)
  double muA = 0.0, muB = 0.0, mulo = 0.0, r = 1.0;  ///< μ(1), μ(0), min, μ_lo/μ_hi
  bool lowA = false;                                 ///< the C = 1 phase is the low-μ side
  double ell[3] = {1.0, 1.0, 1.0};                   ///< ℓ_a = h_a/h_ref
  VofMetric gm;                                      ///< the metric of `vofPhysNormal` (h = ℓ)

  KOKKOS_INLINE_FUNCTION long idx(const int q[3]) const {
    return (long)q[0] + (long)q[1] * st[1] + (long)q[2] * st[2];
  }
  KOKKOS_INLINE_FUNCTION long idx3(const int q[3]) const {
    return (long)(q[0] + d) + (long)(q[1] + d) * st3[1] + (long)(q[2] + d) * st3[2];
  }
  KOKKOS_INLINE_FUNCTION bool inDomain(const int q[3]) const {
    return q[0] >= dlo[0] && q[0] < dhi[0] && q[1] >= dlo[1] && q[1] < dhi[1] && q[2] >= dlo[2] &&
           q[2] < dhi[2];
  }
  KOKKOS_INLINE_FUNCTION double colour(const int q[3]) const { return col(idx3(q)); }
  KOKKOS_INLINE_FUNCTION int cellClass(const int q[3]) const { return (int)cls(idx(q)); }
  KOKKOS_INLINE_FUNCTION double mu(double f) const { return p0 + p1 * f; }
  KOKKOS_INLINE_FUNCTION double res(double f) const { return f / muA + (1.0 - f) / muB; }

  /// The plane (m, α) of a band cell of class `k`, in the cell's local index coordinates (`m·ξ <
  /// α` is the C = 1 side). Virtual (§5.7 Q-B): ξ_a = f, f = side, m = σ e_a, α = σ f, σ = +1 iff
  /// the C = 1 phase lies at ξ_a < f.
  KOKKOS_INLINE_FUNCTION void plane(const int q[3], int k, double m[3], double& al) const {
    if (k == kSvPlane) {
      const long j = idx3(q);
      m[0] = pm[0](j);
      m[1] = pm[1](j);
      m[2] = pm[2](j);
      al = pa(j);
      return;
    }
    const int v = k - kSvVirtual0, ax = v >> 1, side = v & 1;
    const bool one = colour(q) >= 1.0 - eps;  // the cell is pure; its phase
    const double s = (side == 0) ? (one ? -1.0 : 1.0) : (one ? 1.0 : -1.0);
    m[0] = m[1] = m[2] = 0.0;
    m[ax] = s;
    al = (side == 0) ? 0.0 : s;
  }

  /// The four cells of E_ac(k) in the fixed offset order, as (δa, δc): cell = k − δa e_a − δc e_c.
  KOKKOS_INLINE_FUNCTION static void edgeCells(int a, int c, int da[4], int dc[4]) {
    da[0] = 1;
    dc[0] = 1;
    if (a > c) {  // s_a > s_c: −s_a < −s_c
      da[1] = 1;
      dc[1] = 0;
      da[2] = 0;
      dc[2] = 1;
    } else {
      da[1] = 0;
      dc[1] = 1;
      da[2] = 1;
      dc[2] = 0;
    }
    da[3] = 0;
    dc[3] = 0;
  }

  /// The owner of E_ac(k) (§5.2): among its interfacial cells, the smallest |m·ξ_E − α|/|m|₂, ξ_E
  /// the edge midpoint in that cell's local coordinates; ties to the first in offset order.
  /// False when no cell of E is interfacial (E is not a band edge).
  KOKKOS_INLINE_FUNCTION bool edgeOwner(const int k[3], int a, int c, int o[3], double m[3],
                                        double& al) const {
    int da[4], dc[4];
    edgeCells(a, c, da, dc);
    const int b = 3 - a - c;
    bool found = false;
    double best = 0.0;
    for (int t = 0; t < 4; ++t) {
      int q[3] = {k[0], k[1], k[2]};
      q[a] -= da[t];
      q[c] -= dc[t];
      const int kc = cellClass(q);
      if (kc == kSvNone)
        continue;
      double mm[3], aa;
      plane(q, kc, mm, aa);
      double xi[3];
      xi[a] = (double)da[t];
      xi[c] = (double)dc[t];
      xi[b] = 0.5;
      const double nrm = Kokkos::sqrt(mm[0] * mm[0] + mm[1] * mm[1] + mm[2] * mm[2]);
      const double dist = Kokkos::fabs(mm[0] * xi[0] + mm[1] * xi[1] + mm[2] * xi[2] - aa) / nrm;
      if (!found || dist < best) {
        found = true;
        best = dist;
        o[0] = q[0];
        o[1] = q[1];
        o[2] = q[2];
        m[0] = mm[0];
        m[1] = mm[1];
        m[2] = mm[2];
        al = aa;
      }
    }
    return found;
  }

  /// The C = 1 fraction of the u_row face at E_ac(k), row ∈ {a, c} (§5.1 F_c / F_a): the plane
  /// through E normal to the OTHER edge axis, spanning [k_row − ½, k_row + ½] along row and the
  /// b-layer box; from the owner o's plane.
  KOKKOS_INLINE_FUNCTION double rowFaceFraction(const int k[3], int a, int c, int row,
                                                const int o[3], const double m[3],
                                                double al) const {
    const int p = (row == c) ? a : c;
    const int b = 3 - a - c;
    const double xiP = (double)(k[p] - o[p]);
    const double lr = (double)(k[row] - o[row]) - 0.5, lb = (double)(k[b] - o[b]);
    const int q = (p == 0) ? 1 : 0;  // the lower of the two in-plane axes
    return plicFaceFraction(m, al, p, xiP, (q == row) ? lr : lb, (q == row) ? lb : lr);
  }

  /// The C = 1 fraction of the shear link ∂_dd u_cc at its edge E(k) (§5.1 L): the plane x_cc =
  /// k_cc, a hat of half-width 1 along dd centred at k_dd, the b-layer box.
  KOKKOS_INLINE_FUNCTION double linkFraction(const int k[3], int dd, int cc, const int o[3],
                                             const double m[3], double al) const {
    const int b = 3 - dd - cc;
    return plicHatBoxFraction(m, al, cc, (double)(k[cc] - o[cc]), dd, (double)(k[dd] - o[dd]),
                              (double)(k[b] - o[b]));
  }

  /// A shear link's sample lies in the closed physical domain: the edge's b-layer is an in-domain
  /// cell layer and its (a, c) position is on or inside the faces.
  KOKKOS_INLINE_FUNCTION bool edgeInDomain(const int k[3], int a, int c) const {
    const int b = 3 - a - c;
    return k[b] >= dlo[b] && k[b] < dhi[b] && k[a] >= dlo[a] && k[a] <= dhi[a] && k[c] >= dlo[c] &&
           k[c] <= dhi[c];
  }

  /// The C = 1 fraction of a link at an edge with no owner: its in-domain cells are pure and of one
  /// phase (two adjacent opposite pure cells would carry a virtual plane), so the link lies in it.
  KOKKOS_INLINE_FUNCTION double edgePhase(const int k[3], int a, int c) const {
    int da[4], dc[4];
    edgeCells(a, c, da, dc);
    for (int t = 0; t < 4; ++t) {
      int q[3] = {k[0], k[1], k[2]};
      q[a] -= da[t];
      q[c] -= dc[t];
      if (inDomain(q))
        return colour(q) >= 1.0 - eps ? 1.0 : 0.0;
    }
    return 0.0;  // unreachable for an in-domain edge
  }

  /// τ̂ at E_ac(k) (§5.2, §5.7 Q-A): the arithmetic mean of τ̂_K over the interfacial cells K of E,
  /// summed in the fixed offset order.
  KOKKOS_INLINE_FUNCTION void edgeTau(const int k[3], int a, int c, const CCConst (&tau)[3],
                                      double t[3]) const {
    int da[4], dc[4];
    edgeCells(a, c, da, dc);
    long js[4];
    int nI = 0;
    for (int s = 0; s < 4; ++s) {
      int q[3] = {k[0], k[1], k[2]};
      q[a] -= da[s];
      q[c] -= dc[s];
      if (cellClass(q) != kSvNone)
        js[nI++] = idx(q);
    }
    t[0] = t[1] = t[2] = 0.0;
    const double w = (double)nI;
    for (int s = 0; s < nI; ++s) {
      t[0] += tau[0](js[s]) / w;
      t[1] += tau[1](js[s]) / w;
      t[2] += tau[2](js[s]) / w;
    }
  }

  /// The explicit interface-traction part of the u_row stress at E_ac(k) (§5.4 a), row ∈ {a, c}:
  ///   n_a τ̂_c (1 − μ̄ R̄_ac) + n_c τ̂_a (1 − μ̄ R̄_ca),  μ̄ = μ(F_row(E)),
  /// (n from the owner, τ̂ the interfacial mean). Zero on a non-band edge.
  KOKKOS_INLINE_FUNCTION double edgeTraction(const int k[3], int a, int c, int row,
                                             const CCConst (&tau)[3]) const {
    int o[3];
    double m[3], al;
    if (!edgeOwner(k, a, c, o, m, al))
      return 0.0;
    double n[3] = {0.0, 0.0, 0.0};
    vofPhysNormal(m, gm, n);
    const double mub = mu(rowFaceFraction(k, a, c, row, o, m, al));
    const double rac = res(linkFraction(k, a, c, o, m, al));
    const double rca = res(linkFraction(k, c, a, o, m, al));
    double t[3];
    edgeTau(k, a, c, tau, t);
    return n[a] * t[c] * (1.0 - mub * rac) + n[c] * t[a] * (1.0 - mub * rca);
  }

  /// The explicit part of the u_c stress at cell q beyond today's 2μ(C)∂_c u_c (§5.4 b):
  ///   2(μ_A − μ_B)(C̄ − C)(∂_c u_c − D̂_nn) + 2 n_c τ̂_c (1 − μ(C̄) R(C)),
  /// C̄ the c-midsection fraction of the cell's own plane. Zero unless the cell is interfacial.
  KOKKOS_INLINE_FUNCTION double cellExtra(const int q[3], int c, const CCConst (&v)[3],
                                          const CCConst (&tau)[3], CCConst dnn) const {
    const int kc = cellClass(q);
    if (kc == kSvNone)
      return 0.0;
    double m[3], al;
    plane(q, kc, m, al);
    double n[3] = {0.0, 0.0, 0.0};
    vofPhysNormal(m, gm, n);
    const double cb = plicMidsectionFraction(m, al, c), cc = colour(q);
    const long j = idx(q);
    const double mcc = v[c](j + st[c]) - v[c](j);
    return 2.0 * (muA - muB) * (cb - cc) * (mcc - dnn(j)) +
           2.0 * n[c] * tau[c](j) * (1.0 - mu(cb) * res(cc));
  }

  /// One least-squares row of the kinked model (§5.3) for link ∂_a u_c with C = 1 kernel fraction
  /// `thA` and datum `mv`: coefficient of L_pq = δ_ap δ_cq + (1−θ_lo)(r−1) n_a [(Pe_c)_p n_q +
  /// (Pe_c)_q n_p] (`fitCoef`), accumulated into the normal equations (upper triangle, row-major
  /// 9×9) by `fitRow`.
  KOKKOS_INLINE_FUNCTION void fitCoef(int a, int c, double thA, const double n[3],
                                      double co[9]) const {
    const double thlo = lowA ? thA : 1.0 - thA;
    const double kk = (1.0 - thlo) * (r - 1.0) * n[a];
    double pc[3];
    for (int p = 0; p < 3; ++p)
      pc[p] = ((p == c) ? 1.0 : 0.0) - n[c] * n[p];
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q)
        co[3 * p + q] = ((p == a && q == c) ? 1.0 : 0.0) + kk * (pc[p] * n[q] + pc[q] * n[p]);
  }
  KOKKOS_INLINE_FUNCTION void fitRow(int a, int c, double thA, double mv, const double n[3],
                                     double N[81], double rhs[9]) const {
    double co[9];
    fitCoef(a, c, thA, n, co);
    for (int s = 0; s < 9; ++s) {
      rhs[s] += co[s] * mv;
      for (int t = s; t < 9; ++t)
        N[9 * s + t] += co[s] * co[t];
    }
  }

  /// The rows of item `it` of the fit of K (§5.3), in the fit's fixed order: item it < 27 is block
  /// cell it = 9(dz+1) + 3(dy+1) + (dx+1) (offset order) with its normal links ∂_c u_c, c = x, y, z
  /// (θ̄ = C); item 27 + u is edge u = 48 pr + (its offset-order index in orientation pr, (a, c) =
  /// (0,1), (0,2), (1,2)) of the 3×3×3 block with its shear links ∂_a u_c then ∂_c u_a (θ̄ from the
  /// edge's own owner). Writes (a, c, θ̄, datum) per row; returns the row count, 3 or 2, and 0 when
  /// the sample is outside the domain.
  KOKKOS_INLINE_FUNCTION int fitItem(const int K[3], int it, const CCConst (&v)[3], int ra[3],
                                     int rc[3], double th[3], double mv[3]) const {
    if (it < kSvNormalItems) {
      const int q[3] = {K[0] + it % 3 - 1, K[1] + (it / 3) % 3 - 1, K[2] + it / 9 - 1};
      if (!inDomain(q))
        return 0;
      const double tq = colour(q);
      const long j = idx(q);
      for (int c = 0; c < 3; ++c) {
        ra[c] = rc[c] = c;
        th[c] = tq;
        mv[c] = v[c](j + st[c]) - v[c](j);
      }
      return 3;
    }
    const int u = it - kSvNormalItems, pr = u / 48, l = u % 48;
    const int a = (pr == 2) ? 1 : 0, c = (pr == 0) ? 1 : 2, b = 3 - a - c;
    const int ex = (b == 0) ? 3 : 4, ey = (b == 1) ? 3 : 4;  // the b extent is 3, the others 4
    const int k[3] = {K[0] - 1 + l % ex, K[1] - 1 + (l / ex) % ey, K[2] - 1 + l / (ex * ey)};
    if (!edgeInDomain(k, a, c))
      return 0;
    int o[3];
    double mm[3], aa;
    double tac, tca;
    if (edgeOwner(k, a, c, o, mm, aa)) {
      tac = linkFraction(k, a, c, o, mm, aa);
      tca = linkFraction(k, c, a, o, mm, aa);
    } else {
      tac = tca = edgePhase(k, a, c);
    }
    const long j = idx(k);
    ra[0] = a;
    rc[0] = c;
    th[0] = tac;
    mv[0] = (ell[c] / ell[a]) * (v[c](j) - v[c](j - st[a]));
    ra[1] = c;
    rc[1] = a;
    th[1] = tca;
    mv[1] = (ell[a] / ell[c]) * (v[a](j) - v[a](j - st[c]));
    return 2;
  }

  /// The fit of interfacial cell (X, Y, Z) (§5.3): L, the low-μ-side gradient, from the 81 normal
  /// links of the 3×3×3 block and the 288 shear links of its 144 edges (θ̄ from each edge's own
  /// owner), unit weights, Cholesky. Outputs τ̂ = μ_lo P(S n), D̂_nn = n·L n. False = fallback
  /// (smallest pivot < 1e-10·trace/9): the caller stores τ̂ = D̂ = 0. One thread; the device runs
  /// the same rows, sums and solve on a team (`sharpFit`).
  KOKKOS_INLINE_FUNCTION bool fitCell(const int K[3], const CCConst (&v)[3], double tau[3],
                                      double& dnn) const {
    double n[3] = {0.0, 0.0, 0.0};
    fitNormal(K, n);
    double N[81], rhs[9];
    for (int s = 0; s < 81; ++s)
      N[s] = 0.0;
    for (int s = 0; s < 9; ++s)
      rhs[s] = 0.0;
    for (int it = 0; it < kSvItems; ++it) {
      int ra[3], rc[3];
      double th[3], mv[3];
      const int nr = fitItem(K, it, v, ra, rc, th, mv);
      for (int q = 0; q < nr; ++q)
        fitRow(ra[q], rc[q], th[q], mv[q], n, N, rhs);
    }
    return fitSolve(N, rhs, n, mulo, tau, dnn);
  }

  /// The physical unit normal of interfacial cell K's own plane (the fit's n).
  KOKKOS_INLINE_FUNCTION void fitNormal(const int K[3], double n[3]) const {
    double m[3], al;
    plane(K, cellClass(K), m, al);
    vofPhysNormal(m, gm, n);
  }

  /// The solve of the fit's normal equations N (upper triangle, row-major 9×9, overwritten) and
  /// right-hand side, then τ̂ and D̂_nn (see `fitCell`). False = fallback.
  KOKKOS_INLINE_FUNCTION static bool fitSolve(double N[81], const double rhs[9], const double n[3],
                                              double mulo, double tau[3], double& dnn) {
    // Cholesky N = R^T R (upper R in place), pivot test against the trace.
    double tr = 0.0;
    for (int s = 0; s < 9; ++s)
      tr += N[9 * s + s];
    const double pmin = 1e-10 * tr / 9.0;
    for (int s = 0; s < 9; ++s) {
      double dpiv = N[9 * s + s];
      for (int t = 0; t < s; ++t)
        dpiv -= N[9 * t + s] * N[9 * t + s];
      if (!(dpiv >= pmin) || !(dpiv > 0.0))
        return false;
      const double rd = Kokkos::sqrt(dpiv);
      N[9 * s + s] = rd;
      for (int u = s + 1; u < 9; ++u) {
        double x = N[9 * s + u];
        for (int t = 0; t < s; ++t)
          x -= N[9 * t + s] * N[9 * t + u];
        N[9 * s + u] = x / rd;
      }
    }
    double y[9], L[9];
    for (int s = 0; s < 9; ++s) {  // R^T y = rhs
      double x = rhs[s];
      for (int t = 0; t < s; ++t)
        x -= N[9 * t + s] * y[t];
      y[s] = x / N[9 * s + s];
    }
    for (int s = 8; s >= 0; --s) {  // R L = y
      double x = y[s];
      for (int t = s + 1; t < 9; ++t)
        x -= N[9 * s + t] * L[t];
      L[s] = x / N[9 * s + s];
    }
    // S = L + L^T; (S n)_p = sum_q (L_pq + L_qp) n_q; tau = mu_lo P(S n); D_nn = n.L n.
    double sn[3], nsn = 0.0;
    dnn = 0.0;
    for (int p = 0; p < 3; ++p) {
      double x = 0.0;
      for (int q = 0; q < 3; ++q) {
        x += (L[3 * p + q] + L[3 * q + p]) * n[q];
        dnn += n[p] * L[3 * p + q] * n[q];
      }
      sn[p] = x;
      nsn += n[p] * x;
    }
    for (int p = 0; p < 3; ++p)
      tau[p] = mulo * (sn[p] - n[p] * nsn);
    return true;
  }
};

/// Which half of the time split a `SharpFaceProps` feeds (§6.1, WO-5b): the implicit band
/// (`ibmBuildDiffusionVar`) or the explicit transpose (`addFullStressRhs`).
enum class SvRole : int { kImplicit = 0, kExplicit = 1 };

/// The face viscosity under the sharp interface (§5.4 a, §6): a band edge across a (a ≠ c) takes
/// μ̄_c,E = μ(F_c(E)), the u_c row's own PLIC face fraction; the time diagonal is `VarFaceProps`'.
/// The cell link along c (§6.1, WO-5b): on an interfacial cell q (class ≠ kSvNone, virtual planes
/// included) the whole normal stress 2μ(C_q) is implicit — the implicit role returns 2μ(C_q), the
/// explicit role 0; on every other cell both roles return μ(C_q), today's split. A is unchanged.
/// Symmetric under (i, j) swap: the edge is resolved from max(i, j), the cell from min(i, j).
struct SharpFaceProps {
  VarFaceProps base;
  SharpGeom geo;
  int c = 0;
  SvRole role = SvRole::kImplicit;
  KOKKOS_INLINE_FUNCTION double idiag(long i) const { return base.idiag(i); }
  KOKKOS_INLINE_FUNCTION double beta(long i, long j) const {
    const long dd = j - i, ad = (dd < 0) ? -dd : dd;
    if (ad == geo.st[c]) {
      const long q = (dd > 0) ? i : j;  // the cell between the two u_c faces (VarFaceProps: mu(q))
      if (geo.cls(q) == kSvNone)
        return base.beta(i, j);
      return (role == SvRole::kImplicit) ? 2.0 * base.beta(i, j) : 0.0;
    }
    const int a = (ad == 1) ? 0 : ((ad == geo.st[1]) ? 1 : 2);
    const long kk = (dd > 0) ? j : i;
    // Not a band edge (none of its four cells k, k - s_a, k - s_c, k - s_a - s_c interfacial):
    // edgeOwner's answer, read without decoding k.
    const long sa = geo.st[a], sc = geo.st[c];
    if (geo.cls(kk) == kSvNone && geo.cls(kk - sa) == kSvNone && geo.cls(kk - sc) == kSvNone &&
        geo.cls(kk - sa - sc) == kSvNone)
      return base.beta(i, j);
    const long z = kk / geo.st[2], rr = kk - z * geo.st[2], y = rr / geo.st[1];
    const int k[3] = {(int)(rr - y * geo.st[1]), (int)y, (int)z};
    int o[3];
    double m[3], al;
    if (!geo.edgeOwner(k, a, c, o, m, al))
      return base.beta(i, j);
    return geo.mu(geo.rowFaceFraction(k, a, c, c, o, m, al));
  }
};

/// The band classes over the WHOLE solver block (§5.2, §5.7 Q-B). Reads the colour on a 7-point
/// stencil, so the colour block's ghost must exceed the solver's by one (`d = 1`).
inline void sharpClassify(const SharpGeom& gg, SvByte cls) {
  const SharpGeom g = gg;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::vof::sharp_classify",
      MDRange3<CCExec>(space, {0, 0, 0}, {g.e[0], g.e[1], g.e[2]}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const int q[3] = {x, y, z};
        const long i = g.idx(q);
        if (!g.inDomain(q)) {
          cls(i) = kSvNone;
          return;
        }
        const double cq = g.colour(q);
        if (vofIsInterface(cq, g.eps)) {
          cls(i) = kSvPlane;
          return;
        }
        const bool one = cq >= 1.0 - g.eps;
        for (int a = 0; a < 3; ++a)
          for (int side = 0; side < 2; ++side) {
            int nb[3] = {x, y, z};
            nb[a] += side ? 1 : -1;
            if (!g.inDomain(nb))
              continue;  // a non-periodic physical-domain face never qualifies
            const double cn = g.colour(nb);
            if (vofIsInterface(cn, g.eps))
              continue;  // the neighbour must be pure
            if ((cn >= 1.0 - g.eps) != one) {
              cls(i) = (std::uint8_t)(kSvVirtual0 + 2 * a + side);
              return;
            }
          }
        cls(i) = kSvNone;
      });
}

/// The device fit walks its items in chunks of `kSvChunkItems` (rows of a chunk: at most 91, the
/// first chunk's 27 block cells x 3 + 5 edges x 2).
constexpr int kSvChunkItems = 32, kSvChunkRows = 3 * kSvNormalItems + 2 * (kSvChunkItems - 27);

/// The first row of item `it` (row order of `SharpGeom::fitItem`); `kSvItems` gives `kSvRows`.
KOKKOS_INLINE_FUNCTION constexpr int svItemRow(int it) {
  return (it < kSvNormalItems) ? 3 * it : 3 * kSvNormalItems + 2 * (it - kSvNormalItems);
}

/// Team scratch of the device fit, one team per interfacial cell: per row of the current chunk
/// its 9 coefficients, its datum and whether it is present, and the 54 running entries (8 KB).
using SvScrD = Kokkos::View<double*, CCExec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
using SvScrI = Kokkos::View<int*, CCExec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
struct SvFitScratch {
  SvScrD co, mv, ent;
  SvScrI ok;
  static std::size_t shmem() {
    return SvScrD::shmem_size(9 * kSvChunkRows) + SvScrD::shmem_size(kSvChunkRows) +
           SvScrD::shmem_size(kSvEntries) + SvScrI::shmem_size(kSvChunkRows);
  }
  template <class Member>
  KOKKOS_INLINE_FUNCTION explicit SvFitScratch(const Member& tm)
      : co(tm.team_scratch(0), 9 * kSvChunkRows)
      , mv(tm.team_scratch(0), kSvChunkRows)
      , ent(tm.team_scratch(0), kSvEntries)
      , ok(tm.team_scratch(0), kSvChunkRows) {}
};

/// `SharpGeom::fitCell` of cell K on a whole team (the shape of `curvFallbackTeam`), per chunk of
/// items in item order: (1) the lanes map the chunk's items to their rows' coefficients and data
/// (`fitItem`, `fitCoef`); (2) ENTRY-PARALLEL accumulation: lane e continues the running sum of
/// normal-equation entry e (from 0.0 before the first chunk) over the chunk's present rows in row
/// order, with `fitRow`'s expression; then (3) one lane solves (`fitSolve`) and stores. Each entry
/// is the one-thread body's sum, row for row in the same order (the running sum is stored and
/// reloaded, never re-associated); under nvcc's FMA contraction the per-lane expression may round
/// differently from the single-thread fold (the WO-3 precedent of `curvFallbackTeam`), so the
/// device fit agrees with the host one to round-off, not bitwise. Every lane must call it (it holds
/// team barriers).
template <class Member>
KOKKOS_INLINE_FUNCTION void sharpFitTeam(const Member& tm, const SvFitScratch& sc,
                                         const SharpGeom& g, const int K[3], const CCConst (&v)[3],
                                         long i, CCField t0, CCField t1, CCField t2, CCField dnn,
                                         SvByte fb) {
  double n[3] = {0.0, 0.0, 0.0};
  g.fitNormal(K, n);  // every lane: deterministic, no broadcast
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, kSvEntries),
                       [&](const int e) { sc.ent(e) = 0.0; });
  for (int i0 = 0; i0 < kSvItems; i0 += kSvChunkItems) {
    const int i1 = (i0 + kSvChunkItems < kSvItems) ? i0 + kSvChunkItems : kSvItems;
    const int rb = svItemRow(i0), nrow = svItemRow(i1) - rb;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, i0, i1), [&](const int it) {
      int ra[3], rc[3];
      double th[3], mv[3];
      const int nr = g.fitItem(K, it, v, ra, rc, th, mv);
      const int r0 = svItemRow(it) - rb, nmax = svItemRow(it + 1) - svItemRow(it);
      for (int q = 0; q < nmax; ++q) {
        sc.ok(r0 + q) = (q < nr) ? 1 : 0;
        if (q < nr) {
          double co[9];
          g.fitCoef(ra[q], rc[q], th[q], n, co);
          for (int e = 0; e < 9; ++e)
            sc.co(9 * (r0 + q) + e) = co[e];
          sc.mv(r0 + q) = mv[q];
        }
      }
    });
    tm.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, kSvEntries), [&](const int e) {
      double acc = sc.ent(e);
      if (e >= 45) {  // rhs[s] += co[s] * mv
        const int s = e - 45;
        for (int r = 0; r < nrow; ++r)
          if (sc.ok(r))
            acc += sc.co(9 * r + s) * sc.mv(r);
      } else {  // N[9 s + t] += co[s] * co[t], t >= s, e row-major over the upper triangle
        int s = 0, t = e;
        while (t >= 9 - s) {
          t -= 9 - s;
          ++s;
        }
        t += s;
        for (int r = 0; r < nrow; ++r)
          if (sc.ok(r))
            acc += sc.co(9 * r + s) * sc.co(9 * r + t);
      }
      sc.ent(e) = acc;
    });
    tm.team_barrier();  // the chunk's rows are consumed before the next chunk overwrites them
  }
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    double N[81], rhs[9];
    for (int s = 0; s < 81; ++s)
      N[s] = 0.0;
    int e = 0;
    for (int s = 0; s < 9; ++s)
      for (int t = s; t < 9; ++t)
        N[9 * s + t] = sc.ent(e++);
    for (int s = 0; s < 9; ++s)
      rhs[s] = sc.ent(45 + s);
    double t[3] = {0.0, 0.0, 0.0}, dn = 0.0;
    std::uint8_t f = 0;
    if (!SharpGeom::fitSolve(N, rhs, n, g.mulo, t, dn)) {
      t[0] = t[1] = t[2] = 0.0;
      dn = 0.0;
      f = 1;
    }
    t0(i) = t[0];
    t1(i) = t[1];
    t2(i) = t[2];
    dnn(i) = dn;
    fb(i) = f;
  });
}

/// The τ̂/D̂_nn fit (§5.3) over the cells [lo, hi) of the solver block: interfacial cells get their
/// fit (τ̂ = D̂ = 0 and `fb = 1` on a fallback), every other cell τ̂ = D̂ = 0, `fb = 0`. Velocity
/// ghosts (incl. the (a, c) edges) must be filled to depth 2. One pass zeroes the range and
/// compacts its interfacial cells into `list` (>= the range's cell count); then on a device one
/// team per listed cell fits it (`sharpFitTeam`), on the host one thread per listed cell
/// (`SharpGeom::fitCell`) -- a range over the list, not the box, so the band's few cells are spread
/// over every thread instead of the few whose tiles hold the interface.
inline void sharpFit(const SharpGeom& gg, const CCConst (&v)[3], CCField t0, CCField t1, CCField t2,
                     CCField dnn, SvByte fb, C3 lo, C3 hi, SvList list) {
  const SharpGeom g = gg;
  const CCConst v0 = v[0], v1 = v[1], v2 = v[2];
  CCExec space;
  const int nx = hi.x - lo.x, ny = hi.y - lo.y, nz = hi.z - lo.z;
  if (nx <= 0 || ny <= 0 || nz <= 0)
    return;
  const long region = (long)nx * ny * nz;
  long nb = 0;
  Kokkos::parallel_scan(
      "peclet::flow::vof::sharp_fit_compact", Kokkos::RangePolicy<CCExec>(space, 0, region),
      KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
        const int q[3] = {lo.x + (int)(r % nx), lo.y + (int)((r / nx) % ny),
                          lo.z + (int)(r / ((long)nx * ny))};
        const long i = g.idx(q);
        if (final) {
          t0(i) = 0.0;
          t1(i) = 0.0;
          t2(i) = 0.0;
          dnn(i) = 0.0;
          fb(i) = 0;
        }
        if (g.cellClass(q) != kSvNone) {
          if (final)
            list(upd) = i;
          ++upd;
        }
      },
      nb);
  if (nb == 0)
    return;
  if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace, CCMem>::accessible) {
    using Policy = Kokkos::TeamPolicy<CCExec>;
    Policy pol(space, (int)nb, kVofWarp);
    pol.set_scratch_size(0, Kokkos::PerTeam(SvFitScratch::shmem()));
    Kokkos::parallel_for(
        "peclet::flow::vof::sharp_fit", pol, KOKKOS_LAMBDA(const Policy::member_type& tm) {
          const long i = list(tm.league_rank());
          const long z = i / g.st[2], rr = i - z * g.st[2], y = rr / g.st[1];
          const int K[3] = {(int)(rr - y * g.st[1]), (int)y, (int)z};
          const CCConst vv[3] = {v0, v1, v2};
          const SvFitScratch sc(tm);
          sharpFitTeam(tm, sc, g, K, vv, i, t0, t1, t2, dnn, fb);
        });
  } else {
    Kokkos::parallel_for(
        "peclet::flow::vof::sharp_fit", Kokkos::RangePolicy<CCExec>(space, 0, nb),
        KOKKOS_LAMBDA(long r) {
          const long i = list(r);
          const long z = i / g.st[2], rr = i - z * g.st[2], y = rr / g.st[1];
          const int q[3] = {(int)(rr - y * g.st[1]), (int)y, (int)z};
          const CCConst vv[3] = {v0, v1, v2};
          double t[3] = {0.0, 0.0, 0.0}, dn = 0.0;
          if (!g.fitCell(q, vv, t, dn)) {
            fb(i) = 1;  // the scan left tau_hat = D_nn = 0
            return;
          }
          t0(i) = t[0];
          t1(i) = t[1];
          t2(i) = t[2];
          dnn(i) = dn;
        });
  }
}

/// `sharpFit` with its own worklist (the unit tests; the solver keeps one per block).
inline void sharpFit(const SharpGeom& gg, const CCConst (&v)[3], CCField t0, CCField t1, CCField t2,
                     CCField dnn, SvByte fb, C3 lo, C3 hi) {
  const long cells = (long)(hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z);
  SvList list(
      Kokkos::view_alloc(CCExec(), "peclet::flow::vof::sharp_list", Kokkos::WithoutInitializing),
      cells > 0 ? cells : 1);
  sharpFit(gg, v, t0, t1, t2, dnn, fb, lo, hi, list);
}

/// The explicit band terms of the u_c rows over the inner cells [g, e−g)³ (§5.4): the τ-terms of
/// the four (a, c) edges and the cell midsection + τ-terms at i and i − s_c,
///   bb_c(i) += rs(i) · [ w_c (X(i) − X(i−s_c)) + Σ_a (Y(E_ac(i+s_a)) − Y(E_ac(i))) / (ℓ_a ℓ_c) ],
/// a1 < a2 in that association order. The band transpose itself is `addFullStressRhs` with
/// `SharpFaceProps`. `tau`/`dnn` ghosts filled to depth 1 incl. edges; `skipLoC` as there.
/// It runs over `list[0, nl)` (`sharpRhsCompact`: every cell whose terms can be nonzero); a cell
/// the list leaves out has every term exactly 0.
inline void addSharpViscousRhs(int c, CCField bb, CCConst rs, const CCConst (&v)[3],
                               const SharpGeom& gg, const CCConst (&tau)[3], CCConst dnn, int g,
                               double wc, bool skipLoC, SvList list, long nl) {
  const SharpGeom geo = gg;
  const int a1 = (c == 0) ? 1 : 0, a2 = (c == 2) ? 1 : 2;
  const double f1 = 1.0 / (geo.ell[a1] * geo.ell[c]), f2 = 1.0 / (geo.ell[a2] * geo.ell[c]);
  const CCConst v0 = v[0], v1 = v[1], v2 = v[2], t0 = tau[0], t1 = tau[1], t2 = tau[2];
  const int lo[3] = {g + ((skipLoC && c == 0) ? 1 : 0), g + ((skipLoC && c == 1) ? 1 : 0),
                     g + ((skipLoC && c == 2) ? 1 : 0)};
  const auto body = KOKKOS_LAMBDA(int x, int y, int z) {
    const CCConst vv[3] = {v0, v1, v2}, tt[3] = {t0, t1, t2};
    const int q[3] = {x, y, z};
    int qm[3] = {x, y, z};
    qm[c] -= 1;
    const double xc = geo.cellExtra(q, c, vv, tt, dnn) - geo.cellExtra(qm, c, vv, tt, dnn);
    int p1[3] = {x, y, z}, p2[3] = {x, y, z};
    p1[a1] += 1;
    p2[a2] += 1;
    const double y1 = geo.edgeTraction(p1, a1, c, c, tt) - geo.edgeTraction(q, a1, c, c, tt);
    const double y2 = geo.edgeTraction(p2, a2, c, c, tt) - geo.edgeTraction(q, a2, c, c, tt);
    const double t = (wc * xc + f1 * y1) + f2 * y2;
    const long i = geo.idx(q);
    bb(i) += rs(i) * t;
  };
  if (nl <= 0)
    return;
  const int lc = lo[c];
  Kokkos::parallel_for(
      "peclet::flow::vof::sharp_viscous_rhs", Kokkos::RangePolicy<CCExec>(CCExec(), 0, nl),
      KOKKOS_LAMBDA(long r) {
        const long i = list(r);
        const long z = i / geo.st[2], rr = i - z * geo.st[2], y = rr / geo.st[1];
        const int q[3] = {(int)(rr - y * geo.st[1]), (int)y, (int)z};
        if (q[c] < lc)
          return;  // skipLoC: the BC-imposed low-face node
        body(q[0], q[1], q[2]);
      });
}

/// The cells of the inner box [g, e−g)³ whose band terms (`addSharpViscousRhs`) can be nonzero
/// for some component: those with an interfacial cell in their 3×3×3 block (the cells i, i − s_c
/// and the eight further cells of the four edges all lie in it). Compacted into `list` in linear
/// order; returns their number.
inline long sharpRhsCompact(const SharpGeom& gg, int g, SvList list) {
  const SharpGeom geo = gg;
  const int nx = geo.e[0] - 2 * g, ny = geo.e[1] - 2 * g, nz = geo.e[2] - 2 * g;
  if (nx <= 0 || ny <= 0 || nz <= 0)
    return 0;
  long nl = 0;
  Kokkos::parallel_scan(
      "peclet::flow::vof::sharp_rhs_compact",
      Kokkos::RangePolicy<CCExec>(CCExec(), 0, (long)nx * ny * nz),
      KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
        const int q[3] = {g + (int)(r % nx), g + (int)((r / nx) % ny),
                          g + (int)(r / ((long)nx * ny))};
        bool hit = false;
        for (int dz = -1; dz <= 1 && !hit; ++dz)
          for (int dy = -1; dy <= 1 && !hit; ++dy)
            for (int dx = -1; dx <= 1 && !hit; ++dx) {
              const int p[3] = {q[0] + dx, q[1] + dy, q[2] + dz};
              hit = geo.cellClass(p) != kSvNone;
            }
        if (hit) {
          if (final)
            list(upd) = geo.idx(q);
          ++upd;
        }
      },
      nl);
  return nl;
}

/// `addSharpViscousRhs` with its own worklist (the unit and MPI tests).
inline void addSharpViscousRhs(int c, CCField bb, CCConst rs, const CCConst (&v)[3],
                               const SharpGeom& gg, const CCConst (&tau)[3], CCConst dnn, int g,
                               double wc, bool skipLoC) {
  const SvList list(Kokkos::view_alloc(CCExec(), "peclet::flow::vof::sharp_rhs_list",
                                       Kokkos::WithoutInitializing),
                    (std::size_t)gg.e[0] * gg.e[1] * gg.e[2]);
  const long nl = sharpRhsCompact(gg, g, list);
  addSharpViscousRhs(c, bb, rs, v, gg, tau, dnn, g, wc, skipLoC, list, nl);
}

/// The census of the band (§9, G-SV10) over [lo, hi), LOCAL to this block: interfacial cells, the
/// virtual-plane cells among them, the film cells among those (both a-neighbours pure and of the
/// other phase), and the fit fallbacks.
struct SharpCensus {
  long band = 0, virt = 0, film = 0, fallback = 0;
};

inline SharpCensus sharpCensus(const SharpGeom& gg, SvByteConst fb, C3 lo, C3 hi) {
  const SharpGeom g = gg;
  CCExec space;
  SharpCensus out;
  long cnt[4] = {0, 0, 0, 0};
  for (int w = 0; w < 4; ++w) {
    long s = 0;
    Kokkos::parallel_reduce(
        "peclet::flow::vof::sharp_census",
        MDRange3<CCExec>(space, {lo.x, lo.y, lo.z}, {hi.x, hi.y, hi.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
          const int q[3] = {x, y, z};
          const int k = g.cellClass(q);
          if (k == kSvNone)
            return;
          if (w == 0)
            acc += 1;
          else if (w == 3)
            acc += fb(g.idx(q)) ? 1 : 0;
          else if (k >= kSvVirtual0) {
            if (w == 1) {
              acc += 1;
              return;
            }
            const int ax = (k - kSvVirtual0) >> 1;
            const bool one = g.colour(q) >= 1.0 - g.eps;
            int lo3[3] = {x, y, z}, hi3[3] = {x, y, z};
            lo3[ax] -= 1;
            hi3[ax] += 1;
            bool film = g.inDomain(lo3) && g.inDomain(hi3);
            if (film) {
              const double cl = g.colour(lo3), ch = g.colour(hi3);
              film = !vofIsInterface(cl, g.eps) && !vofIsInterface(ch, g.eps) &&
                     ((cl >= 1.0 - g.eps) != one) && ((ch >= 1.0 - g.eps) != one);
            }
            acc += film ? 1 : 0;
          }
        },
        s);
    cnt[w] = s;
  }
  out.band = cnt[0];
  out.virt = cnt[1];
  out.film = cnt[2];
  out.fallback = cnt[3];
  return out;
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_SHARP_VISCOUS_HPP
