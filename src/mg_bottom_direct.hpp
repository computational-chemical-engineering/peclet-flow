/// @file
/// @brief flow — the device pressure bottom's direct preconditioner (doc/vof_step_performance_
/// design.md §13): a block-tridiagonal factorization of the agglomerated bottom operator, planes
/// along one axis, each plane's Schur complement inverted explicitly (block Thomas), in FacReal
/// arithmetic (`float` in production), applied as M inside the bottom's FP64 flexible CG.
///
/// Container-free (Views and PODs only), so it can move to core::solver later (§13.9 Q-D8).
///
/// The scheme (§13.4, normative):
///  * the bottom operator A is the level's face form {AFX, AFY, AFZ}; its diagonal is RE-SUMMED in
///    FP64, d_i = -(sum of the six face coefficients, -x, +x, -y, +y, -z, +z), so A*1 = 0 exactly
///    per fluid component (GraphAMG's rule: a face that crosses a non-periodic domain face is
///    excluded); s_i = 1/sqrt(d_i) on fluid cells with d_i > 0, else 0 (an identity row);
///  * the factored matrix is the scaled A~ = S A S, assembled in FP64 and cast to FacReal:
///    diagonal 1, off-diagonals s_i AF s_j accumulated over cell i's six faces in that order;
///  * the null space is lifted by the exact plane-local augmentation: 1/m_c added to every pair of
///    component c's cells in its last plane k_c (m_c = its cells there), so A' x = r exactly for a
///    compatible r (§13.4.2);
///  * factor (one team, at the first bottom solve after each operator change): Sigma_0 = A^_0,
///    Sigma_k = A^_k - diag(e_k) Q_{k-1} diag(e_k), Q_k = Sigma_k^{-1} by a tiled right-looking
///    Cholesky (tile kBottomTile), W = L^{-1}, Q = W^T W; a periodic slow axis with P >= 3 adds the
///    border Y = T^{-1} B and Q_B = (A^_{P-1} - diag(e_0) Y_0 - diag(e_{P-1}) Y_{P-2})^{-1};
///    a pivot <= 1e-6 (or a non-finite one) restarts the whole factor with delta = 1e-4, then 1e-2
///    on every scaled diagonal; a third failure sets facOk = 0;
///  * M(r) (inside the solve kernel): cast r~ = FacReal(s r), forward and backward block
///    substitution, the border, uncast z = s x~ on the fluid cells.
///
/// Reproducibility (§13.2): every stored or output scalar is computed by ONE thread in an order
/// fixed by the algorithm and kBottomTile, never by the team size -- no team reductions here -- so
/// the factor and M(r) are bitwise independent of T.
///
/// On a HOST backend the factor runs a second schedule of the same arithmetic (§14 H-1, A(b);
/// BottomFactorKernel::hostAssemble / hostInvert / hostBorder): every stored scalar is the team
/// algorithm's expression with its operands in the same order, but the work is cut into a few
/// barrier-free phases per tile (the team version publishes every column of every diagonal tile
/// with a barrier: ~170 per plane) and each phase computes many independent outputs in a
/// vectorizable lane loop instead of one serial dot product per thread. Its factor storage (Q, Y,
/// e, s, stat) is bitwise identical to the team algorithm's on the same backend (ctest
/// `bottom_direct` U8); only the scratch layouts of Sg, L and W differ.
#pragma once

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

// Unrolling the single-accumulator dot loops keeps several independent loads in flight; it changes
// no operation and no order (bitwise-neutral).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PECLET_BOTTOM_UNROLL _Pragma("unroll 16")
#else
#define PECLET_BOTTOM_UNROLL
#endif
// The host factor's lane loops (one output per iteration, no reduction): `omp simd` on a host
// compiler. With -ffp-contract=off (the host flags) vectorizing them changes no bit.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PECLET_BOTTOM_HOST_SIMD
#else
#define PECLET_BOTTOM_HOST_SIMD PECLET_FLOW_OMP_SIMD
#endif

inline constexpr int kBottomMaxPlane = 192;  // b cap (§13.9 Q-D5)
inline constexpr int kBottomTile = 16;       // Cholesky tile (compile time: it fixes the order)
inline constexpr double kBottomPivotTol = 1e-6;
inline constexpr int kBottomAttempts = 3;  // shifts 0, 1e-4, 1e-2
KOKKOS_INLINE_FUNCTION double bottomShift(int attempt) {
  return attempt == 0 ? 0.0 : (attempt == 1 ? 1e-4 : 1e-2);
}

// The factor ordering, decided on the host at hierarchy build from the domain BC flags (§13.4.1).
// Plane k = the coordinate along the slow axis s (P = n_s planes); the in-plane index is
// q = c[a0] + n[a0] * c[a1] with a0 < a1 the other two axes; the factor-order index is k*b + q.
struct BottomPlanes {
  int s = -1;  // slow axis; -1 = no axis gives b <= kBottomMaxPlane (ineligible)
  int a0 = 0, a1 = 1;
  int P = 0, b = 0;
  int n[3] = {0, 0, 0};             // the bottom's inner dims
  int bcf[6] = {0, 0, 0, 0, 0, 0};  // per face (-x,+x,-y,+y,-z,+z): 1 = non-periodic (excluded)
  int border = 0;                   // periodic slow axis with P >= 3
  KOKKOS_INLINE_FUNCTION bool valid() const { return s >= 0; }
};

// §13.4.1: the candidates are the non-periodic axes (all three if there are none); the longest
// wins, a tie to the higher index (z > y > x). If its plane b = n / n_s exceeds kBottomMaxPlane,
// the remaining axes are tried in decreasing length (the same tie rule); none fits -> s = -1.
inline BottomPlanes bottomChoosePlanes(C3 inner, const int bc[6]) {
  BottomPlanes p;
  p.n[0] = inner.x;
  p.n[1] = inner.y;
  p.n[2] = inner.z;
  int per[3];
  for (int f = 0; f < 6; ++f)
    p.bcf[f] = bc[f] != 0 ? 1 : 0;
  for (int a = 0; a < 3; ++a)
    per[a] = (bc[2 * a] == 0 && bc[2 * a + 1] == 0) ? 1 : 0;
  const long ncell = (long)inner.x * inner.y * inner.z;
  auto longer = [&](int a, int c) { return p.n[a] > p.n[c] || (p.n[a] == p.n[c] && a > c); };
  int first = -1;
  const bool anyWall = !per[0] || !per[1] || !per[2];
  for (int a = 0; a < 3; ++a)
    if ((!anyWall || !per[a]) && (first < 0 || longer(a, first)))
      first = a;
  auto fits = [&](int a) { return p.n[a] > 0 && ncell / p.n[a] <= kBottomMaxPlane; };
  int s = -1;
  if (fits(first)) {
    s = first;
  } else {
    int r[2], m = 0;
    for (int a = 0; a < 3; ++a)
      if (a != first)
        r[m++] = a;
    if (longer(r[1], r[0])) {
      const int t = r[0];
      r[0] = r[1];
      r[1] = t;
    }
    for (int i = 0; i < 2 && s < 0; ++i)
      if (fits(r[i]))
        s = r[i];
  }
  if (s < 0)
    return p;
  p.s = s;
  p.a0 = (s == 0) ? 1 : 0;
  p.a1 = (s == 2) ? 1 : 2;
  p.P = p.n[s];
  p.b = (int)(ncell / p.P);
  p.border = (per[s] && p.P >= 3) ? 1 : 0;
  return p;
}

// The direct engine's device storage and its M (§13.4.4, §13.4.7). Ghost width 1 (single rank).
template <class FacReal>
struct BottomDirect {
  using Mat = Kokkos::View<FacReal***, Kokkos::LayoutRight, CCMem>;
  using Sq = Kokkos::View<FacReal**, Kokkos::LayoutRight, CCMem>;
  BottomPlanes pl;
  C3 ext{0, 0, 0};
  Mat Q;        // [P][b][b]: Q(k, j, i) = (Sigma_k^{-1})_{ij} (symmetric, stored both ways)
  Mat Y;        // [P-1][b][b] (border only): Y(k, j, i) = Y_k(i, j)
  Sq e;         // [P][b]: e(k, q) = A~((q,k), (q,k-1)); e(0, .) the wrap coupling (border only)
  Sq Sg, L, W;  // [b][b] factor scratch: Sigma (lower), L, W = L^{-1} (both lower)
  Kokkos::View<double*, CCMem> s;      // [n] the scaling, factor order
  Kokkos::View<FacReal*, CCMem> g, u;  // [n] substitution vectors
  // [0] facOk, [1] restarts, [2 + a] attempt a's fail flag (one slot per attempt: a flag that
  // was reset for the next attempt could be read by a thread still leaving the last one)
  Kokkos::View<int*, CCMem> stat;

  static BottomDirect allocate(const BottomPlanes& p, C3 ext) {
    BottomDirect d;
    d.pl = p;
    d.ext = ext;
    if (!p.valid())
      return d;
    const int n = p.P * p.b;
    d.Q = Mat("peclet::flow::mg_bottom_Q", p.P, p.b, p.b);
    if (p.border)
      d.Y = Mat("peclet::flow::mg_bottom_Y", p.P - 1, p.b, p.b);
    d.e = Sq("peclet::flow::mg_bottom_e", p.P, p.b);
    d.Sg = Sq("peclet::flow::mg_bottom_sigma", p.b, p.b);
    d.L = Sq("peclet::flow::mg_bottom_L", p.b, p.b);
    d.W = Sq("peclet::flow::mg_bottom_W", p.b, p.b);
    d.s = Kokkos::View<double*, CCMem>("peclet::flow::mg_bottom_s", n);
    d.g = Kokkos::View<FacReal*, CCMem>("peclet::flow::mg_bottom_g", n);
    d.u = Kokkos::View<FacReal*, CCMem>("peclet::flow::mg_bottom_u", n);
    d.stat = Kokkos::View<int*, CCMem>("peclet::flow::mg_bottom_stat", 2 + kBottomAttempts);
    return d;
  }
  KOKKOS_INLINE_FUNCTION int n() const { return pl.P * pl.b; }
  // factor index f -> inner coordinates (0-based) and the flat ext index
  KOKKOS_INLINE_FUNCTION void coords(int f, int c[3]) const {
    const int k = f / pl.b, q = f - k * pl.b;
    const int c1 = q / pl.n[pl.a0];
    c[pl.s] = k;
    c[pl.a0] = q - c1 * pl.n[pl.a0];
    c[pl.a1] = c1;
  }
  KOKKOS_INLINE_FUNCTION int extIndex(const int c[3]) const {
    return (c[0] + 1) + (c[1] + 1) * ext.x + (c[2] + 1) * ext.x * ext.y;
  }
  KOKKOS_INLINE_FUNCTION int extOf(int f) const {
    int c[3];
    coords(f, c);
    return extIndex(c);
  }
  KOKKOS_INLINE_FUNCTION int factorIndex(const int c[3]) const {
    return c[pl.s] * pl.b + c[pl.a0] + pl.n[pl.a0] * c[pl.a1];
  }
  // Face f6 (0..5 = -x, +x, -y, +y, -z, +z) of the cell at c (ext index i): its coefficient (the
  // A2 face form: the low face of i, or of i's +axis neighbour) and the neighbour's coordinates,
  // wrapped on a periodic axis; false where it crosses a non-periodic domain face (GraphAMG's
  // exclusion: no partner, and no part of the re-summed diagonal).
  template <class OpV>
  KOKKOS_INLINE_FUNCTION bool face(const OpV& AFX, const OpV& AFY, const OpV& AFZ, const int c[3],
                                   int i, int f6, int nb[3], double& af) const {
    const int a = f6 >> 1, side = f6 & 1;
    nb[0] = c[0];
    nb[1] = c[1];
    nb[2] = c[2];
    int v = c[a] + (side ? 1 : -1);
    if (v < 0 || v >= pl.n[a]) {
      if (pl.bcf[f6])
        return false;
      v = (v < 0) ? v + pl.n[a] : v - pl.n[a];
    }
    nb[a] = v;
    const int st = (a == 0) ? 1 : (a == 1 ? ext.x : ext.x * ext.y);
    const int slotIdx = side ? i + st : i;
    af = (a == 0) ? (double)AFX(slotIdx) : (a == 1 ? (double)AFY(slotIdx) : (double)AFZ(slotIdx));
    return true;
  }

  // M (§13.4.4): z = s * x~ on the fluid cells (comp >= 0), 0 on the other inner cells, where
  // x~ = (A~')^{-1} FacReal(s r) by the stored factor. With `keep`, z's old inner values go there
  // first (the FCG's z_prev), fused into the cast phase. The caller removes the component means.
  template <class Member, class RV, class ZV, class CV>
  KOKKOS_INLINE_FUNCTION void apply(const Member& t, const RV& r, const ZV& z, const CV& comp,
                                    const ZV* keep) const {
    const int P = pl.P, b = pl.b, nn = P * b;
    const int Pt = pl.border ? P - 1 : P;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nn), [&](int f) {
      const int ei = extOf(f);
      if (keep)
        (*keep)(ei) = z(ei);
      u(f) = (FacReal)(s(f) * r(ei));
    });
    t.team_barrier();
    for (int k = 0; k < Pt; ++k) {  // forward
      const int o = k * b, op = o - b;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b), [&](int i) {
        FacReal acc = 0;
        if (k == 0) {
          PECLET_BOTTOM_UNROLL
          for (int j = 0; j < b; ++j)
            acc += Q(k, j, i) * u(o + j);
        } else {
          PECLET_BOTTOM_UNROLL
          for (int j = 0; j < b; ++j)
            acc += Q(k, j, i) * (u(o + j) - e(k, j) * g(op + j));
        }
        g(o + i) = acc;
      });
      t.team_barrier();
    }
    for (int k = Pt - 2; k >= 0; --k) {  // backward (in place: x~ overwrites g)
      const int o = k * b, on = o + b;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b), [&](int i) {
        FacReal acc = 0;
        PECLET_BOTTOM_UNROLL
        for (int j = 0; j < b; ++j)
          acc += Q(k, j, i) * (e(k + 1, j) * g(on + j));
        g(o + i) = g(o + i) - acc;
      });
      t.team_barrier();
    }
    if (pl.border) {  // x~_B = Q_B (r~_B - e_0 u_0 - e_{P-1} u_{P-2}); x~_k = u_k - Y_k x~_B
      const int oB = (P - 1) * b, oL = (P - 2) * b;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b), [&](int i) {
        FacReal acc = 0;
        PECLET_BOTTOM_UNROLL
        for (int j = 0; j < b; ++j)
          acc += Q(P - 1, j, i) * ((u(oB + j) - e(0, j) * g(j)) - e(P - 1, j) * g(oL + j));
        g(oB + i) = acc;
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, oB), [&](int f) {
        const int k = f / b, i = f - k * b;
        FacReal acc = 0;
        PECLET_BOTTOM_UNROLL
        for (int j = 0; j < b; ++j)
          acc += Y(k, j, i) * g(oB + j);
        g(f) = g(f) - acc;
      });
      t.team_barrier();
    }
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nn), [&](int f) {
      const int ei = extOf(f);
      z(ei) = (comp(ei) >= 0) ? s(f) * (double)g(f) : 0.0;
    });
    t.team_barrier();
  }
};

// The factor launch (§13.4.3): one team, TeamPolicy(1, T). Inputs: the bottom level's face form,
// its component labels (comp, ext-indexed; -1 = solid) and the per-component augmentation (kc =
// the last plane, aug = 1/m_c; BottomLabelKernel). Output: D.Q, D.Y, D.e, D.s and D.stat.
template <class FacReal, class OpV>
struct BottomFactorKernel {
  using Member = Kokkos::TeamPolicy<CCExec>::member_type;
  // Level-0 (team) scratch, a bitwise-neutral staging of the factor's working set (§13.4.7): Ls =
  // the current tile column of L (rows j0..b-1, the tile's columns; padded to kBottomTile + 1 so a
  // warp's column reads hit distinct banks), St = the current diagonal tile of Sigma.
  using SMat = Kokkos::View<FacReal**, Kokkos::LayoutRight, typename CCExec::scratch_memory_space,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static std::size_t scratchBytes(int b) {
    return SMat::shmem_size(b, kBottomTile + 1) + SMat::shmem_size(kBottomTile, kBottomTile + 1);
  }
  BottomDirect<FacReal> D;
  OpV AFX, AFY, AFZ;
  Kokkos::View<const int*, CCMem> comp, kc;
  Kokkos::View<const double*, CCMem> aug;
  double tauPiv0 = kBottomPivotTol;  // the first attempt's pivot floor (a test hook raises it)
  // A host backend runs the host schedule (hostAssemble / hostInvert / hostBorder) unless this
  // test hook asks for the team algorithm, its bitwise oracle (ctest `bottom_direct` U8).
  static constexpr bool kHostFactor = std::is_same_v<CCMem, Kokkos::HostSpace>;
  int teamAlgorithm = 0;

  // s (§13.4.1): the re-summed diagonal and the scaling, one thread per cell
  KOKKOS_INLINE_FUNCTION void scaling(const Member& t) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, D.n()), [&](int f) {
      int c[3], nb[3];
      D.coords(f, c);
      const int i = D.extIndex(c);
      double sum = 0.0, af = 0.0;
      for (int f6 = 0; f6 < 6; ++f6)
        if (D.face(AFX, AFY, AFZ, c, i, f6, nb, af))
          sum += af;
      const double d = -sum;
      D.s(f) = (comp(i) >= 0 && d > 0.0) ? 1.0 / Kokkos::sqrt(d) : 0.0;
    });
    t.team_barrier();
  }
  // e (§13.4.1): e(k, q) accumulates, over cell (q,k)'s faces in order, s_i AF s_j for the faces
  // whose neighbour lies in plane k-1 (mod P) != k -- for k = 0 only on a border axis. With P = 2
  // on a periodic axis both faces of plane 1 land in e(1, .) and e(0, .) stays 0.
  KOKKOS_INLINE_FUNCTION void couplings(const Member& t) const {
    const int P = D.pl.P;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, D.n()), [&](int f) {
      int c[3], nb[3];
      D.coords(f, c);
      const int i = D.extIndex(c), k = c[D.pl.s], q = f - k * D.pl.b;
      const int km = (k - 1 + P) % P;
      double v = 0.0, af = 0.0;
      if (km != k && (k >= 1 || D.pl.border)) {
        const double si = D.s(f);
        for (int f6 = 0; f6 < 6; ++f6)
          if (D.face(AFX, AFY, AFZ, c, i, f6, nb, af) && nb[D.pl.s] == km)
            v += si * af * D.s(D.factorIndex(nb));
      }
      D.e(k, q) = (FacReal)v;
    });
    t.team_barrier();
  }
  // A^_k(i, j) in FP64 (§13.4.1-2): diagonal 1, + s_i AF s_j over cell i's faces whose neighbour
  // is in-plane cell j (in order), + 1/m_c on component c's last plane, + delta on the diagonal.
  KOKKOS_INLINE_FUNCTION double entry(int k, int i, int j, double delta) const {
    const int b = D.pl.b, fi = k * b + i, fj = k * b + j;
    int c[3], nb[3];
    D.coords(fi, c);
    const int ei = D.extIndex(c);
    double v = (i == j) ? 1.0 : 0.0, af = 0.0;
    const double si = D.s(fi);
    for (int f6 = 0; f6 < 6; ++f6)
      if (D.face(AFX, AFY, AFZ, c, ei, f6, nb, af) && D.factorIndex(nb) == fj)
        v += si * af * D.s(fj);
    const int ci = comp(ei);
    if (ci >= 0 && ci < (int)kc.extent(0) && kc(ci) == k && comp(D.extOf(fj)) == ci)
      v += aug(ci);
    if (i == j)
      v += delta;
    return v;
  }
  // Sigma's lower triangle for plane k: mode 0 = A^_k; 1 = A^_k - diag(e_k) Q_{k-1} diag(e_k);
  // 2 = the border A^_{P-1} - diag(e_0) Y_0 - diag(e_{P-1}) Y_{P-2}.
  KOKKOS_INLINE_FUNCTION void assemble(const Member& t, int k, int mode, double delta,
                                       const SMat& St) const {
    const int b = D.pl.b, P = D.pl.P;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b * b), [&](int idx) {
      const int i = idx / b, j = idx - i * b;
      if (j > i)
        return;
      FacReal a = (FacReal)entry(k, i, j, delta);
      if (mode == 1)
        a = a - D.e(k, i) * D.Q(k - 1, i, j) * D.e(k, j);
      else if (mode == 2)
        a = (a - D.e(0, i) * D.Y(0, j, i)) - D.e(P - 1, i) * D.Y(P - 2, j, i);
      D.Sg(i, j) = a;
      if (i < kBottomTile)
        St(i, j) = a;  // the first diagonal tile
    });
    t.team_barrier();
  }
  // Q(k) = Sigma^{-1} (§13.4.3, "Dense SPD inverse"). Returns false (team-uniform) on a pivot
  // that is not > tauPiv (a NaN pivot included). Sigma's lower triangle is read from D.Sg and its
  // current diagonal tile from St; the current tile column of L is kept in Ls (and stored to D.L
  // for W). Each stored scalar is the same expression, in the same order, as without the staging.
  KOKKOS_INLINE_FUNCTION bool invert(const Member& t, int k, double tauPiv, int flag,
                                     const SMat& Ls, const SMat& St) const {
    const int b = D.pl.b;
    const auto& S = D.Sg;
    const auto& L = D.L;
    const auto& W = D.W;
    for (int j0 = 0; j0 < b; j0 += kBottomTile) {
      const int j1 = (j0 + kBottomTile < b) ? j0 + kBottomTile : b;
      const int nc = j1 - j0;
      // 1. the diagonal tile, column by column: row r of the tile (one thread) computes L(r, c)
      //    for every column c < r and, right after L(r, r - 1), its own pivot L(r, r) -- each
      //    scalar the single-thread column Cholesky's expression in its order; one barrier per
      //    column publishes the column.
      for (int c = -1; c < nc; ++c) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nc), [&](int r) {
          if (r <= c)
            return;
          const int i = j0 + r;
          if (c >= 0) {  // L(i, j) for j = j0 + c
            const int j = j0 + c;
            FacReal a2 = 0;
            for (int m = 0; m < c; ++m)
              a2 += Ls(i, m) * Ls(j, m);
            const FacReal lij = (St(r, c) - a2) / Ls(j, c);
            Ls(i, c) = lij;
            L(i, j) = lij;
          }
          if (r == c + 1) {  // the pivot of column r: every L(i, m < r) is this thread's own
            FacReal acc = 0;
            for (int m = 0; m < r; ++m)
              acc += Ls(i, m) * Ls(i, m);
            const FacReal p = St(r, r) - acc;
            if (!((double)p > tauPiv))
              D.stat(flag) = 1;
            const FacReal lii = Kokkos::sqrt(p);
            Ls(i, r) = lii;
            L(i, i) = lii;
          }
        });
        t.team_barrier();
      }
      if (D.stat(flag))
        return false;
      if (j1 < b) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, j1, b), [&](int i) {  // 2. the panel
          for (int c = 0; c < nc; ++c) {
            const int j = j0 + c;
            FacReal acc = 0;
            for (int m = 0; m < c; ++m)
              acc += Ls(i, m) * Ls(j, m);
            const FacReal lij = (S(i, j) - acc) / Ls(j, c);
            Ls(i, c) = lij;
            L(i, j) = lij;
          }
        });
        t.team_barrier();
        const int nt = b - j1;  // 3. the trailing update, i >= j below the tile
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nt * nt), [&](int idx) {
          const int i = j1 + idx / nt, j = j1 + (idx - (idx / nt) * nt);
          if (j > i)
            return;
          FacReal acc = 0;
          for (int m = 0; m < nc; ++m)
            acc += Ls(i, m) * Ls(j, m);
          const FacReal v = S(i, j) - acc;
          S(i, j) = v;
          if (i < j1 + kBottomTile)
            St(i - j1, j - j1) = v;  // the next diagonal tile
        });
        t.team_barrier();
      }
    }
    // W = L^{-1}: W(j, j) = 1 / L(j, j), W(i, j) = -(sum_{m=j}^{i-1} L(i, m) W(m, j)) / L(i, i) in
    // ascending m, by row blocks of kBottomTile: (1) every thread takes one (row i of the block,
    // column j left of it) and sums the terms m < i0 into A(r, j) (Ls's storage); (2) thread j
    // continues the SAME accumulator over the block's own rows (the L tile staged in St, its new
    // W(m, j) in registers) -- each W(i, j) is the column recurrence's expression in its order.
    const SMat A(Ls.data(), kBottomTile, b);
    for (int i0 = 0; i0 < b; i0 += kBottomTile) {
      const int i1 = (i0 + kBottomTile < b) ? i0 + kBottomTile : b;
      const int nr = i1 - i0;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nr * i0 + nr * nr), [&](int idx) {
        if (idx < nr * i0) {
          const int r = idx / i0, j = idx - r * i0, i = i0 + r;
          FacReal acc = 0;
          PECLET_BOTTOM_UNROLL
          for (int m = j; m < i0; ++m)
            acc += L(i, m) * W(m, j);
          A(r, j) = acc;
        } else {
          const int q = idx - nr * i0, r = q / nr, c = q - r * nr;
          if (c <= r)
            St(r, c) = L(i0 + r, i0 + c);
        }
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, i1), [&](int j) {
        FacReal wb[kBottomTile];  // W(i0 + r, j) of this block, this column
        for (int r = 0; r < kBottomTile; ++r) {
          if (r >= nr)
            break;
          const int i = i0 + r;
          if (j > i) {
            wb[r] = 0;
            continue;
          }
          if (j == i) {
            wb[r] = FacReal(1) / St(r, r);
            W(i, j) = wb[r];
            continue;
          }
          FacReal acc = (j < i0) ? A(r, j) : FacReal(0);
          const int m0 = (j > i0) ? j : i0;
          for (int m = m0; m < i; ++m)
            acc += St(r, m - i0) * wb[m - i0];
          wb[r] = -acc / St(r, r);
          W(i, j) = wb[r];
        }
      });
      t.team_barrier();
    }
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b * b), [&](int idx) {  // Q = W^T W
      const int i = idx / b, j = idx - i * b;
      if (j > i)
        return;
      FacReal acc = 0;
      PECLET_BOTTOM_UNROLL
      for (int m = i; m < b; ++m)
        acc += W(m, i) * W(m, j);
      D.Q(k, i, j) = acc;
      D.Q(k, j, i) = acc;
    });
    t.team_barrier();
    return true;
  }
  // The border's Y = T^{-1} B by the solve's own substitution, one column c per (c, i) pair:
  // B's block 0 = diag(e_0), block P-2 = diag(e_{P-1}).
  KOKKOS_INLINE_FUNCTION void border(const Member& t) const {
    const int P = D.pl.P, b = D.pl.b;
    for (int k = 0; k <= P - 2; ++k) {  // forward
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b * b), [&](int idx) {
        const int c = idx / b, i = idx - c * b;
        FacReal acc = 0;
        PECLET_BOTTOM_UNROLL
        for (int j = 0; j < b; ++j) {
          FacReal bj = 0;
          if (j == c)
            bj = (k == 0 ? D.e(0, j) : FacReal(0)) + (k == P - 2 ? D.e(P - 1, j) : FacReal(0));
          const FacReal br = (k == 0) ? bj : bj - D.e(k, j) * D.Y(k - 1, c, j);
          acc += D.Q(k, j, i) * br;
        }
        D.Y(k, c, i) = acc;
      });
      t.team_barrier();
    }
    for (int k = P - 3; k >= 0; --k) {  // backward
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, b * b), [&](int idx) {
        const int c = idx / b, i = idx - c * b;
        FacReal acc = 0;
        PECLET_BOTTOM_UNROLL
        for (int j = 0; j < b; ++j)
          acc += D.Q(k, j, i) * (D.e(k + 1, j) * D.Y(k + 1, c, j));
        D.Y(k, c, i) = D.Y(k, c, i) - acc;
      });
      t.team_barrier();
    }
  }

  // --- The host schedule (§14 H-1, A(b)) ------------------------------------------------------
  // The same scalars as assemble / invert / border, each the same expression with its operands in
  // the same order; what changes is who computes what between barriers. Scratch layouts (host
  // only): Sg holds Sigma's lower triangle by COLUMNS, S[j b + i] = Sigma(i, j) for i >= j; L holds
  // L by columns, Lt[m b + i] = L(i, m); W is row-major W(i, j) as on a device. A lane loop runs
  // over independent outputs (rows i, or columns j), each lane continuing its own accumulator in
  // ascending order, so vectorizing it reorders nothing. A thread takes a contiguous share of the
  // outputs, balanced by their cost (hostShare), or every T-th column where columns are far apart.

  // acc[i] += x[m b + i] * y[m b] for m = m0 .. m1-1 IN THAT ORDER, each lane i in [lo, hi): four
  // m per pass over the lanes, so acc is loaded and stored once per four terms -- each lane still
  // adds its terms one at a time in ascending m.
  KOKKOS_INLINE_FUNCTION static void hostAccumulate(FacReal* acc, const FacReal* x,
                                                    const FacReal* y, int b, int m0, int m1, int lo,
                                                    int hi) {
    int m = m0;
    for (; m + 4 <= m1; m += 4) {
      const FacReal *x0 = x + m * b, *x1 = x0 + b, *x2 = x1 + b, *x3 = x2 + b;
      const FacReal y0 = y[m * b], y1 = y[(m + 1) * b], y2 = y[(m + 2) * b], y3 = y[(m + 3) * b];
      PECLET_BOTTOM_HOST_SIMD
      for (int i = lo; i < hi; ++i) {
        FacReal a = acc[i];
        a += x0[i] * y0;
        a += x1[i] * y1;
        a += x2[i] * y2;
        a += x3[i] * y3;
        acc[i] = a;
      }
    }
    for (; m < m1; ++m) {
      const FacReal* xm = x + m * b;
      const FacReal ym = y[m * b];
      PECLET_BOTTOM_HOST_SIMD
      for (int i = lo; i < hi; ++i)
        acc[i] += xm[i] * ym;
    }
  }
  // [lo, hi): this thread's contiguous share of [0, n) when index i costs w(i) -- equal prefix
  // shares of the total. Contiguous, not round-robin: neighbouring outputs share cache lines.
  template <class Wf>
  KOKKOS_INLINE_FUNCTION static void hostShare(const Member& t, int n, const Wf& w, int& lo,
                                               int& hi) {
    const int rank = t.team_rank(), T = t.team_size();
    double tot = 0.0;
    for (int i = 0; i < n; ++i)
      tot += w(i);
    lo = n;
    hi = n;
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
      if (lo == n && cum >= tot * rank / T)
        lo = i;
      if (cum >= tot * (rank + 1) / T) {
        hi = i;
        break;
      }
      cum += w(i);
    }
    if (lo > hi)
      lo = hi;
  }
  // assemble: the team version's entry() per (i, j) re-scans cell i's six faces for every j; here
  // a row's dense values start at entry()'s initial value and each face adds its term to its own
  // partner j in face order -- per (i, j) the same additions in the same order -- then the
  // augmentation, then delta on the diagonal, then the cast and the mode's update.
  KOKKOS_INLINE_FUNCTION void hostAssemble(const Member& t, int k, int mode, double delta) const {
    const int b = D.pl.b, P = D.pl.P, s = D.pl.s;
    FacReal* S = D.Sg.data();
    int lo, hi;
    hostShare(t, b, [](int i) { return (double)(i + 1); }, lo, hi);
    for (int i = lo; i < hi; ++i) {
      double v[kBottomMaxPlane];
      for (int j = 0; j < i; ++j)
        v[j] = 0.0;
      v[i] = 1.0;
      const int fi = k * b + i;
      int c[3], nb[3];
      D.coords(fi, c);
      const int ei = D.extIndex(c);
      const double si = D.s(fi);
      double af = 0.0;
      for (int f6 = 0; f6 < 6; ++f6)
        if (D.face(AFX, AFY, AFZ, c, ei, f6, nb, af) && nb[s] == k) {
          const int fj = D.factorIndex(nb);
          if (fj - k * b <= i)
            v[fj - k * b] += si * af * D.s(fj);
        }
      const int ci = comp(ei);
      if (ci >= 0 && ci < (int)kc.extent(0) && kc(ci) == k)
        for (int j = 0; j <= i; ++j)
          if (comp(D.extOf(k * b + j)) == ci)
            v[j] += aug(ci);
      v[i] += delta;
      for (int j = 0; j <= i; ++j) {
        FacReal a = (FacReal)v[j];
        if (mode == 1)
          a = a - D.e(k, i) * D.Q(k - 1, i, j) * D.e(k, j);
        else if (mode == 2)
          a = (a - D.e(0, i) * D.Y(0, j, i)) - D.e(P - 1, i) * D.Y(P - 2, j, i);
        S[j * b + i] = a;
      }
    }
    t.team_barrier();
  }
  // invert: per tile, ONE thread factors the diagonal tile (row by row: L(i, j) for j < i, then
  // the pivot) and the panel below it (column by column, a lane per row), one barrier; the team
  // applies the trailing update (a column per thread, a lane per row), one barrier. Then W = L^{-1}
  // by rows (a lane per column; each thread a contiguous column range, rows in order) and Q = W^T W
  // by rows (a lane per column) -- one barrier each. The lane loops are long (up to b), so the
  // accumulators' loads and stores pipeline instead of forming a latency chain.
  KOKKOS_INLINE_FUNCTION bool hostInvert(const Member& t, int k, double tauPiv, int flag) const {
    const int b = D.pl.b, rank = t.team_rank(), T = t.team_size();
    FacReal* S = D.Sg.data();
    FacReal* Lt = D.L.data();
    FacReal* W = D.W.data();
    for (int j0 = 0; j0 < b; j0 += kBottomTile) {
      const int j1 = (j0 + kBottomTile < b) ? j0 + kBottomTile : b;
      const int nc = j1 - j0;
      Kokkos::single(Kokkos::PerTeam(t), [&]() {
        for (int r = 0; r < nc; ++r) {  // 1. the diagonal tile
          const int i = j0 + r;
          for (int c = 0; c < r; ++c) {
            const int j = j0 + c;
            FacReal a2 = 0;
            for (int m = 0; m < c; ++m)
              a2 += Lt[(j0 + m) * b + i] * Lt[(j0 + m) * b + j];
            Lt[j * b + i] = (S[j * b + i] - a2) / Lt[j * b + j];
          }
          FacReal acc = 0;
          for (int m = 0; m < r; ++m)
            acc += Lt[(j0 + m) * b + i] * Lt[(j0 + m) * b + i];
          const FacReal p = S[i * b + i] - acc;
          if (!((double)p > tauPiv))
            D.stat(flag) = 1;
          Lt[i * b + i] = Kokkos::sqrt(p);
        }
        if (D.stat(flag))
          return;
        for (int c = 0; c < nc; ++c) {  // 2. the panel, rows j1..b-1
          const int j = j0 + c;
          FacReal acc[kBottomMaxPlane];
          PECLET_BOTTOM_HOST_SIMD
          for (int i = j1; i < b; ++i)
            acc[i] = 0;
          hostAccumulate(acc, Lt + j0 * b, Lt + j0 * b + j, b, 0, c, j1, b);
          const FacReal ljj = Lt[j * b + j];
          FacReal* Lj = Lt + j * b;
          const FacReal* Sj = S + j * b;
          PECLET_BOTTOM_HOST_SIMD
          for (int i = j1; i < b; ++i)
            Lj[i] = (Sj[i] - acc[i]) / ljj;
        }
      });
      t.team_barrier();
      if (D.stat(flag))
        return false;
      if (j1 < b) {
        for (int j = j1 + rank; j < b; j += T) {  // 3. the trailing update, column j, rows j..b-1
          FacReal acc[kBottomMaxPlane];
          PECLET_BOTTOM_HOST_SIMD
          for (int i = j; i < b; ++i)
            acc[i] = 0;
          hostAccumulate(acc, Lt + j0 * b, Lt + j0 * b + j, b, 0, nc, j, b);
          FacReal* Sj = S + j * b;
          PECLET_BOTTOM_HOST_SIMD
          for (int i = j; i < b; ++i)
            Sj[i] = Sj[i] - acc[i];
        }
        t.team_barrier();
      }
    }
    // W(i, j) = -(sum_{m=j}^{i-1} L(i, m) W(m, j)) / L(i, i), W(i, i) = 1 / L(i, i): row i of this
    // thread's columns [ja, jz) (equal shares of the (b - j)^2 work) sums m = ja .. i-1 in order,
    // lane j taking the term m once m >= j -- each lane's sum is m = j .. i-1, ascending.
    int ja, jz;
    hostShare(t, b, [&](int j) { return (double)(b - j) * (b - j); }, ja, jz);
    for (int i = ja; i < b; ++i) {
      FacReal acc[kBottomMaxPlane];
      const int jt = (i < jz) ? i : jz;  // lanes ja .. jt-1 lie left of the diagonal
      PECLET_BOTTOM_HOST_SIMD
      for (int j = ja; j < jt; ++j)
        acc[j] = 0;
      int m = ja;
      for (; m + 4 <= i; m += 4) {  // terms m .. m+3: all four for the lanes j <= m, ...
        hostAccumulate(acc, W, Lt + i, b, m, m + 4, ja, (m + 1 < jz) ? m + 1 : jz);
        for (int d = 1; d < 4 && m + d < jz; ++d)  // ... m+d .. m+3 for lane m+d (its first)
          for (int q = m + d; q < m + 4; ++q)
            acc[m + d] += Lt[q * b + i] * W[q * b + m + d];
      }
      for (; m < i; ++m)
        hostAccumulate(acc, W, Lt + i, b, m, m + 1, ja, (m + 1 < jz) ? m + 1 : jz);
      const FacReal lii = Lt[i * b + i];
      FacReal* Wi = W + i * b;
      PECLET_BOTTOM_HOST_SIMD
      for (int j = ja; j < jt; ++j)
        Wi[j] = -acc[j] / lii;
      if (i < jz)
        Wi[i] = FacReal(1) / lii;
    }
    t.team_barrier();
    FacReal* Qk = &D.Q(k, 0, 0);
    int qa, qz;
    hostShare(t, b, [&](int i) { return (double)(b - i) * (i + 1); }, qa, qz);
    for (int i = qa; i < qz; ++i) {  // Q(i, j <= i) = sum_{m >= i} W(m, i) W(m, j)
      FacReal acc[kBottomMaxPlane];
      PECLET_BOTTOM_HOST_SIMD
      for (int j = 0; j <= i; ++j)
        acc[j] = 0;
      hostAccumulate(acc, W, W + i, b, i, b, 0, i + 1);
      for (int j = 0; j <= i; ++j) {
        Qk[i * b + j] = acc[j];
        Qk[j * b + i] = acc[j];
      }
    }
    t.team_barrier();
    return true;
  }
  // border: Y(., c, .) depends on its own column c only, so a thread runs one column through the
  // whole forward and backward sweep (a lane per row i) with no barrier between the planes.
  KOKKOS_INLINE_FUNCTION void hostBorder(const Member& t) const {
    const int P = D.pl.P, b = D.pl.b, rank = t.team_rank(), T = t.team_size();
    for (int c = rank; c < b; c += T) {
      FacReal acc[kBottomMaxPlane], br[kBottomMaxPlane];
      for (int k = 0; k <= P - 2; ++k) {  // forward
        for (int j = 0; j < b; ++j) {
          FacReal bj = 0;
          if (j == c)
            bj = (k == 0 ? D.e(0, j) : FacReal(0)) + (k == P - 2 ? D.e(P - 1, j) : FacReal(0));
          br[j] = (k == 0) ? bj : bj - D.e(k, j) * D.Y(k - 1, c, j);
        }
        PECLET_BOTTOM_HOST_SIMD
        for (int i = 0; i < b; ++i)
          acc[i] = 0;
        for (int j = 0; j < b; ++j) {
          const FacReal* Qj = &D.Q(k, j, 0);
          const FacReal bj = br[j];
          PECLET_BOTTOM_HOST_SIMD
          for (int i = 0; i < b; ++i)
            acc[i] += Qj[i] * bj;
        }
        FacReal* Yk = &D.Y(k, c, 0);
        PECLET_BOTTOM_HOST_SIMD
        for (int i = 0; i < b; ++i)
          Yk[i] = acc[i];
      }
      for (int k = P - 3; k >= 0; --k) {  // backward
        PECLET_BOTTOM_HOST_SIMD
        for (int i = 0; i < b; ++i)
          acc[i] = 0;
        for (int j = 0; j < b; ++j) {
          const FacReal* Qj = &D.Q(k, j, 0);
          const FacReal y = D.e(k + 1, j) * D.Y(k + 1, c, j);
          PECLET_BOTTOM_HOST_SIMD
          for (int i = 0; i < b; ++i)
            acc[i] += Qj[i] * y;
        }
        FacReal* Yk = &D.Y(k, c, 0);
        PECLET_BOTTOM_HOST_SIMD
        for (int i = 0; i < b; ++i)
          Yk[i] = Yk[i] - acc[i];
      }
    }
    t.team_barrier();
  }
  // operator()'s control flow (attempts, shifts, flags) on the host schedule.
  KOKKOS_INLINE_FUNCTION void hostFactor(const Member& t) const {
    const int P = D.pl.P;
    const int Pt = D.pl.border ? P - 1 : P;
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      for (int a = 0; a < kBottomAttempts; ++a)
        D.stat(2 + a) = 0;
    });
    scaling(t);  // (its barrier also publishes the cleared flags)
    couplings(t);
    int attempt = 0;
    bool ok = false;
    for (; attempt < kBottomAttempts && !ok; ++attempt) {
      const double delta = bottomShift(attempt);
      const double tauPiv = (attempt == 0) ? tauPiv0 : kBottomPivotTol;
      ok = true;
      for (int k = 0; k < Pt && ok; ++k) {
        hostAssemble(t, k, k == 0 ? 0 : 1, delta);
        ok = hostInvert(t, k, tauPiv, 2 + attempt);
      }
      if (ok && D.pl.border) {
        hostBorder(t);
        hostAssemble(t, P - 1, 2, delta);
        ok = hostInvert(t, P - 1, tauPiv, 2 + attempt);
      }
    }
    const int restarts = attempt - 1;
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      D.stat(0) = ok ? 1 : 0;
      D.stat(1) = restarts;
    });
  }

  KOKKOS_INLINE_FUNCTION void operator()(const Member& t) const {
    const int P = D.pl.P;
    const int Pt = D.pl.border ? P - 1 : P;
    if constexpr (kHostFactor) {
      if (!teamAlgorithm) {
        hostFactor(t);
        return;
      }
    }
    const SMat Ls(t.team_scratch(0), D.pl.b, kBottomTile + 1);
    const SMat St(t.team_scratch(0), kBottomTile, kBottomTile + 1);
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      for (int a = 0; a < kBottomAttempts; ++a)
        D.stat(2 + a) = 0;
    });
    scaling(t);  // (its barrier also publishes the cleared flags)
    couplings(t);
    int attempt = 0;
    bool ok = false;
    for (; attempt < kBottomAttempts && !ok; ++attempt) {
      const double delta = bottomShift(attempt);
      const double tauPiv = (attempt == 0) ? tauPiv0 : kBottomPivotTol;
      ok = true;
      for (int k = 0; k < Pt && ok; ++k) {
        assemble(t, k, k == 0 ? 0 : 1, delta, St);
        ok = invert(t, k, tauPiv, 2 + attempt, Ls, St);
      }
      if (ok && D.pl.border) {
        border(t);
        assemble(t, P - 1, 2, delta, St);
        ok = invert(t, P - 1, tauPiv, 2 + attempt, Ls, St);
      }
    }
    const int restarts = attempt - 1;
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      D.stat(0) = ok ? 1 : 0;
      D.stat(1) = restarts;
    });
  }
};

}  // namespace peclet::flow
