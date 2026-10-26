// Zebra line relaxation on the pressure-MG levels whose unique strong axis stops coarsening
// (doc/mg_depth_policy.md Amendment A2.4, WO-7). Included by mac_cutcell_mg.hpp after the FP32
// V-cycle kernels; CutcellMG drives it (lineSmoothColor / lineSmoothColorFp32) on a TRIGGERED
// smoothed level (A2.3) in place of the point red-black colour pass.
//
// Column = the line along s at fixed inner (a, b) (a < b the other axes, x-y-z order). Colour =
// (og_a + l_a + og_b + l_b) & 1 (og = parityOg(lv), l local storage indices). Per column, one
// thread solves the column's tridiagonal (s between walls) or cyclic tridiagonal (s periodic, by
// Sherman-Morrison) system for the correction, k = 0 ... m-1 (i_k the cell, t the s-stride):
//   r_k = b_k - (A x)_k   the level's full operator at the current x (FP64 cutcellApplyFaceCell |
//                         FP32 fp32Flux: x = c * 1 and b = 0 give delta = 0 exactly);
//   d_k = AC(i_k) | fp32Diag(i_k);  l_k = AF_s(i_k) | -W_s(i_k);  u_k = AF_s(i_k + t) |
//   -W_s(i_k+t); decoupled cell (AC < 1e-30 | D = 0): d_k = 1, l_k = u_k = r_k = 0. s non-periodic:
//   l_0 = u_{m-1} := 0, Thomas. s periodic: beta = l_0, alpha = u_{m-1} (the wrap face); alpha =
//   beta = 0 -> Thomas; else Sherman-Morrison with gamma = -d_0, d'_0 = d_0 - gamma, d'_{m-1} =
//   d_{m-1} - alpha beta / gamma: T'y = r and T'z = (gamma, 0, ..., 0, alpha)^T in the same passes,
//   f = (y_0 + beta y_{m-1}/gamma) / (1 + z_0 + beta z_{m-1}/gamma) (f := 0 if that denominator's
//   magnitude < eps_p), delta = y - f z. Thomas: p_0 = d'_0; p_k = d'_k - l_k c_{k-1}, p_k := d'_k
//   if p_k < eps_p d'_k (a sealed Neumann segment); c_k = u_k / p_k; y_k = (r_k - l_k y_{k-1}) /
//   p_k (z alike); back substitution y_k -= c_k y_{k+1}, k = m-2 ... 0. eps_p = 64 eps_mach
//   (1.42e-14 | 7.63e-6). x_k += delta_k.
// r_k is formed in the forward pass from the x the pass started with: x changes only in the final
// update loop, so every r_k reads the pre-pass x (the column's own cells included) and the order of
// operations is the one written above; where c, y, z live therefore changes no bit.
//
// Neighbours: ccMaskNbrs with the caller's wrap mask (0: the ghosts, filled by the caller; 7: the
// single-rank in-kernel wrap of ccWrapNbrs; the level's stencil wrap axes under MPI). The
// coefficients are read at their ghost indices as the point kernels read them.
//
// Launch: device MDRange2<CCExec> over the inner (a, b), x fastest, colour filter; host RangePolicy
// over the b-rows (each row's colour columns in a loop), serial behind hostRunSerial.
#pragma once

namespace peclet::flow {

// The inner (a, b) axes of a column along s (a < b, x-y-z order).
KOKKOS_INLINE_FUNCTION void lineAxesOf(int s, int& a, int& b) {
  a = (s == 0) ? 1 : 0;
  b = (s == 2) ? 1 : 2;
}

// FP64 row access of the face-form operator (the point smoother's cutcellSmoothFaceCell operands).
template <class OpV>
struct LineRowFp64 {
  using Real = double;
  static constexpr double kEps = 64.0 * std::numeric_limits<double>::epsilon();  // 1.42e-14
  CCField x;
  CCConst b;
  OpV AC, AFX, AFY, AFZ;
  long sx, sy, sz, t;
  int s;
  KOKKOS_INLINE_FUNCTION const OpV& afs() const { return s == 0 ? AFX : (s == 1 ? AFY : AFZ); }
  // d, l, u of row i; false (d = 1, l = u = 0) on a decoupled cell.
  KOKKOS_INLINE_FUNCTION bool coef(long i, double& d, double& l, double& u) const {
    const double ac = AC(i);
    if (ac < 1e-30) {
      d = 1.0;
      l = 0.0;
      u = 0.0;
      return false;
    }
    d = ac;
    l = afs()(i);
    u = afs()(i + t);
    return true;
  }
  KOKKOS_INLINE_FUNCTION double res(long i, const CcNbrs& w) const {
    return b(i) - cutcellApplyFaceCell(x, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym,
                                       w.zp, w.zm);
  }
};

// FP32 row access of the face-weight form (fp32SmoothCell's operands). No double in this body.
struct LineRowFp32 {
  using Real = VReal;
  static constexpr VReal kEps = 64.0f * std::numeric_limits<VReal>::epsilon();  // 7.63e-6
  VField x;
  VConst b, WX, WY, WZ;
  long sx, sy, sz, t;
  int s;
  KOKKOS_INLINE_FUNCTION const VConst& ws() const { return s == 0 ? WX : (s == 1 ? WY : WZ); }
  KOKKOS_INLINE_FUNCTION bool coef(long i, VReal& d, VReal& l, VReal& u) const {
    const VReal D = fp32Diag(WX, WY, WZ, i, sx, sy, sz);
    if (D == 0.0f) {
      d = 1.0f;
      l = 0.0f;
      u = 0.0f;
      return false;
    }
    d = D;
    l = -ws()(i);
    u = -ws()(i + t);
    return true;
  }
  KOKKOS_INLINE_FUNCTION VReal res(long i, const CcNbrs& w) const {
    return b(i) - fp32Flux(x, WX, WY, WZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym, w.zp, w.zm);
  }
};

// The column's c, y, z by k: the level-extent scratch (indexed by cell) or, for m <= kLineRegs,
// thread-local arrays (A2.6: "hold c, y, z in registers for m <= 64"). The arithmetic and its
// order are the same in both, so the result is bitwise the same (A2.4).
constexpr int kLineRegs = 64;
template <class SV>
struct LineScratchGlobal {
  SV cv, yv, zv;
  long i0, t;
  KOKKOS_INLINE_FUNCTION typename SV::reference_type c(int k) const { return cv(i0 + (long)k * t); }
  KOKKOS_INLINE_FUNCTION typename SV::reference_type y(int k) const { return yv(i0 + (long)k * t); }
  KOKKOS_INLINE_FUNCTION typename SV::reference_type z(int k) const { return zv(i0 + (long)k * t); }
};
template <class Real>
struct LineScratchLocal {
  Real cv[kLineRegs], yv[kLineRegs], zv[kLineRegs];
  KOKKOS_INLINE_FUNCTION Real& c(int k) { return cv[k]; }
  KOKKOS_INLINE_FUNCTION Real& y(int k) { return yv[k]; }
  KOKKOS_INLINE_FUNCTION Real& z(int k) { return zv[k]; }
};

// One column along s from its first cell i0 (local coordinates l3 with l3[s] = g): the A2.4 solve
// and the update x_k += delta_k, with the column's c, y, z in S.
template <class Row, class Store>
KOKKOS_INLINE_FUNCTION void lineColumnSolve(const Row& R, Store& S, long i0, int m, int l3[3], C3 n,
                                            int g, int wrapMask, bool periodic) {
  using Real = typename Row::Real;
  const Real eps = Row::kEps;
  const long t = R.t;
  // s periodic: beta = l_0, alpha = u_{m-1} (after the decoupled-cell zeroing); both zero ->
  // Thomas.
  Real alpha = Real(0), beta = Real(0);
  bool sm = false;
  if (periodic) {
    Real d, l, u;
    R.coef(i0, d, l, u);
    beta = l;
    R.coef(i0 + (long)(m - 1) * t, d, l, u);
    alpha = u;
    sm = !(alpha == Real(0) && beta == Real(0));
  }
  Real gamma = Real(0);
  Real cPrev = Real(0), yPrev = Real(0), zPrev = Real(0);
  for (int k = 0; k < m; ++k) {
    const long i = i0 + (long)k * t;
    l3[R.s] = g + k;
    Real d, l, u;
    Real r = Real(0);
    if (R.coef(i, d, l, u)) {
      const CcNbrs w = ccMaskNbrs(l3[0], l3[1], l3[2], n, g, i, R.sy, R.sz, wrapMask);
      r = R.res(i, w);
    }
    if (!periodic) {
      if (k == 0)
        l = Real(0);
      if (k == m - 1)
        u = Real(0);
    }
    Real dp = d;
    if (sm) {
      if (k == 0) {
        gamma = -d;
        dp = d - gamma;
      }
      if (k == m - 1)
        dp = d - alpha * beta / gamma;
    }
    const Real zr = !sm ? Real(0) : (k == 0 ? gamma : (k == m - 1 ? alpha : Real(0)));
    Real p, yk, zk;
    if (k == 0) {
      p = dp;
      yk = r / p;
      zk = zr / p;
    } else {
      p = dp - l * cPrev;
      if (p < eps * dp)
        p = dp;  // a sealed Neumann segment
      yk = (r - l * yPrev) / p;
      zk = (zr - l * zPrev) / p;
    }
    const Real ck = u / p;
    S.c(k) = ck;
    S.y(k) = yk;
    if (sm)
      S.z(k) = zk;
    cPrev = ck;
    yPrev = yk;
    zPrev = zk;
  }
  for (int k = m - 2; k >= 0; --k) {
    S.y(k) -= S.c(k) * S.y(k + 1);
    if (sm)
      S.z(k) -= S.c(k) * S.z(k + 1);
  }
  Real f = Real(0);
  if (sm) {
    const Real den = Real(1) + S.z(0) + beta * S.z(m - 1) / gamma;
    if (!(Kokkos::fabs(den) < eps))
      f = (S.y(0) + beta * S.y(m - 1) / gamma) / den;
  }
  for (int k = 0; k < m; ++k) {
    const long i = i0 + (long)k * t;
    if (sm)
      R.x(i) = R.x(i) + (S.y(k) - f * S.z(k));
    else
      R.x(i) = R.x(i) + S.y(k);
  }
}

// One column (la, lb) along s. c, y, z: the level-extent scratch, used when m > kLineRegs.
template <class Row, class SV>
KOKKOS_INLINE_FUNCTION void lineColumn(const Row& R, const SV& c, const SV& y, const SV& z, int la,
                                       int lb, C3 n, int g, int wrapMask, bool periodic) {
  using Real = typename Row::Real;
  int a = 0, bb = 0;
  lineAxesOf(R.s, a, bb);
  const int m = R.s == 0 ? n.x : (R.s == 1 ? n.y : n.z);
  int l3[3];
  l3[a] = la;
  l3[bb] = lb;
  l3[R.s] = g;
  const long i0 = (long)l3[0] + (long)l3[1] * R.sy + (long)l3[2] * R.sz;
  if (m <= kLineRegs) {
    LineScratchLocal<Real> S;
    lineColumnSolve(R, S, i0, m, l3, n, g, wrapMask, periodic);
  } else {
    LineScratchGlobal<SV> S{c, y, z, i0, R.t};
    lineColumnSolve(R, S, i0, m, l3, n, g, wrapMask, periodic);
  }
}

// One colour of the line smoother over every column of the level (A2.4 launch shape).
template <class Row, class SV>
inline void lineRelaxColor(const Row& R, const SV& c, const SV& y, const SV& z, C3 e, C3 og, int g,
                           int wrapMask, bool periodic, int color, const char* name,
                           const char* serialName) {
  const C3 n{e.x - 2 * g, e.y - 2 * g, e.z - 2 * g};
  int a = 0, b = 0;
  lineAxesOf(R.s, a, b);
  const int ea = a == 0 ? e.x : e.y, eb = b == 1 ? e.y : e.z;
  const int oga = a == 0 ? og.x : og.y, ogb = b == 1 ? og.y : og.z;
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nb = eb - 2 * g;
    const long cells = (long)n.x * n.y * n.z;
    auto row = KOKKOS_LAMBDA(long q) {
      const int lb = g + (int)q;
      const int P = (color + oga + ogb + lb) & 1;
      for (int la = g + ((P ^ (g & 1)) & 1); la < ea - g; la += 2)
        lineColumn(R, c, y, z, la, lb, n, g, wrapMask, periodic);
    };
    if (hostRunSerial(cells)) {
      const HostSerialRegion serialRegion(serialName);
      for (long q = 0; q < (long)nb; ++q)
        row(q);
      return;
    }
    Kokkos::parallel_for(name, Kokkos::RangePolicy<CCExec>(space, 0, (long)nb), row);
    return;
  }
  Kokkos::parallel_for(
      name, MDRange2<CCExec>(space, {g, g}, {ea - g, eb - g}), KOKKOS_LAMBDA(int la, int lb) {
        if (((oga + la + ogb + lb) & 1) != color)
          return;
        lineColumn(R, c, y, z, la, lb, n, g, wrapMask, periodic);
      });
}

// FP64: one colour of the line smoother on the face-form operator (x = phi, b = rhs).
template <class OpV>
inline void lineRelaxColorFace(CCField x, CCConst b, OpV AC, OpV AFX, OpV AFY, OpV AFZ, CCField c,
                               CCField y, CCField z, C3 e, C3 og, int g, int s, bool periodic,
                               int wrapMask, int color) {
  const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
  const long t = s == 0 ? sx : (s == 1 ? sy : sz);
  const LineRowFp64<OpV> R{x, b, AC, AFX, AFY, AFZ, sx, sy, sz, t, s};
  lineRelaxColor(R, c, y, z, e, og, g, wrapMask, periodic, color, "peclet::flow::cc_line_smooth",
                 "peclet::flow::cc_line_smooth@serial");
}

// FP32: one colour of the line smoother on the FP32 face weights.
inline void lineRelaxColorFp32(VField x, VConst b, VConst WX, VConst WY, VConst WZ, VField c,
                               VField y, VField z, C3 e, C3 og, int g, int s, bool periodic,
                               int wrapMask, int color) {
  const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
  const long t = s == 0 ? sx : (s == 1 ? sy : sz);
  const LineRowFp32 R{x, b, WX, WY, WZ, sx, sy, sz, t, s};
  lineRelaxColor(R, c, y, z, e, og, g, wrapMask, periodic, color,
                 "peclet::flow::cc_line_smooth_f32", "peclet::flow::cc_line_smooth_f32@serial");
}

}  // namespace peclet::flow
