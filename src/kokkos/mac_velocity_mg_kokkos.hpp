// cfd-gpu — portable (Kokkos) velocity (momentum) multigrid for the IBM diffusion solve: the STAIRCASE
// coarse operator.
//
// Single-GPU (periodic) port of the velocity-MG path in CUDA's DistributedPoissonMG (mac_multigrid.cuh):
// the fine level is the sharp Robust-Scaled IBM stencil As_[c] (so the residual + smoother use the TRUE
// operator and the fixed point is the exact sharp solution); the coarse levels use the geometry-aware
// STAIRCASE Helmholtz (volume fraction theta only CLASSIFIES cells: theta>=0.5 fluid / <0.5 solid-pinned,
// then a plain constant-coefficient Helmholtz at fluid cells). The fine IBM-cell residuals are excluded
// from coarsening (clean-fluid mask) and no coarse correction is pumped back into the cut-cell band
// (masked prolong); the fine smoother owns the boundary. See [[velocity-mg-design]].
//
// The whole hierarchy uses ghost width G=2 (the velocity block's width), so level 0 IS the solver's
// velocity block: the IBM stencil + RHS + solution need no g=2<->g=1 bridging. Reuses restrictAvg /
// prolongAdd (mac_cutcell_mg) and ibmRbgsStencilColor (the pin-aware variable-coeff RB-GS smoother ==
// mg_smooth_var_k). Runs on any Kokkos backend.
#ifndef CFD_MAC_VELOCITY_MG_KOKKOS_HPP
#define CFD_MAC_VELOCITY_MG_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell_mg_kokkos.hpp"  // restrictAvg, prolongAdd, FPV/FPC
#include "mac_ibm_kokkos.hpp"         // ibmRbgsStencilColor (pin smoother), MConst

namespace cfdk {

// pin-aware variable-coefficient residual (mg_residual_var_k): r = 0 at pinned (classified-solid) cells,
// else b - A x with the float operator accumulated in double.
inline void residualVarPin(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN, FPC AB,
                           FPC AT, CCConst pin, C3 e, int g) {
  CCExec space; const bool hasPin = (pin.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("cfdk::vmg_resid", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
    KOKKOS_LAMBDA(int lx, int ly, int lz) {
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const long i = (long)lx + (long)ly * sy + (long)lz * sz;
      if (hasPin && pin(i) > 0.5) { r(i) = 0.0; return; }
      const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                        (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                        (double)AB(i) * x(i - sz);
      r(i) = b(i) - Ax;
    });
  space.fence();
}

// masked trilinear prolongation (mg_prolong_masked_k): like prolongAdd but does NOT add the coarse
// correction into a fine cell whose mask < eps (the clean-fluid exclude mask is 0 at IBM cut+solid cells).
inline void prolongMasked(CCField fine, CCConst coarse, CCConst mask, C3 fext, C3 cext, int g, C3 finner,
                          C3 ratio, double eps) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("cfdk::vmg_prolong_masked", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
    KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
      const long fi = (long)(ifx + g) + (long)(ify + g) * fext.x + (long)(ifz + g) * (long)fext.x * fext.y;
      if (mask(fi) < eps) return;  // no correction into a cut/solid fine cell
      const double cx = (ratio.x == 2) ? 0.5 * ifx - 0.25 + g : ifx + g;
      const double cy = (ratio.y == 2) ? 0.5 * ify - 0.25 + g : ify + g;
      const double cz = (ratio.z == 2) ? 0.5 * ifz - 0.25 + g : ifz + g;
      const double fxw = Kokkos::floor(cx), fyw = Kokkos::floor(cy), fzw = Kokkos::floor(cz);
      const double wx = cx - fxw, wy = cy - fyw, wz = cz - fzw;
      const int x0 = (int)fxw, y0 = (int)fyw, z0 = (int)fzw;
      const long sy = cext.x, sz = (long)cext.x * cext.y;
      auto C = [&](int xx, int yy, int zz) { return coarse((long)xx + (long)yy * sy + (long)zz * sz); };
      const double c00 = C(x0, y0, z0) * (1 - wx) + C(x0 + 1, y0, z0) * wx;
      const double c10 = C(x0, y0 + 1, z0) * (1 - wx) + C(x0 + 1, y0 + 1, z0) * wx;
      const double c01 = C(x0, y0, z0 + 1) * (1 - wx) + C(x0 + 1, y0, z0 + 1) * wx;
      const double c11 = C(x0, y0 + 1, z0 + 1) * (1 - wx) + C(x0 + 1, y0 + 1, z0 + 1) * wx;
      const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
      fine(fi) += c0 * (1 - wz) + c1 * wz;
    });
  space.fence();
}

// STAIRCASE coarse operator (mg_build_velocity_op_staircase_k): theta<thresh -> identity (pinned) row;
// else a plain const-coeff Helmholtz (idiag + 2*(bx+by+bz) diagonal, per-axis -b off-diagonals).
inline void buildVelocityStaircase(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, CCConst theta,
                                   C3 e, int g, double bx, double by, double bz, double thresh, double idiag) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("cfdk::vmg_staircase", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
    KOKKOS_LAMBDA(int lx, int ly, int lz) {
      const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
      if (theta(i) < thresh) {  // classified solid -> identity row (smoother/residual pin it to 0)
        AC(i) = 1.0f; AW(i) = AE(i) = AS(i) = AN(i) = AB(i) = AT(i) = 0.0f; return;
      }
      AC(i) = (float)(idiag + 2.0 * (bx + by + bz));
      AW(i) = (float)(-bx); AE(i) = (float)(-bx);
      AS(i) = (float)(-by); AN(i) = (float)(-by);
      AB(i) = (float)(-bz); AT(i) = (float)(-bz);
    });
  space.fence();
}

inline void thresholdMask(CCField m, CCConst theta, double thresh) {  // m = 1 where theta < thresh (solid)
  CCExec space; std::size_t n = m.extent(0); CCField mm = m; CCConst th = theta;
  Kokkos::parallel_for("cfdk::vmg_threshold", Kokkos::RangePolicy<CCExec>(space, 0, n),
    KOKKOS_LAMBDA(std::size_t i) { mm(i) = (th(i) < thresh) ? 1.0 : 0.0; });
  space.fence();
}
inline void mulMask(CCField r, CCConst m) {  // r *= m (clean-fluid residual filter)
  CCExec space; std::size_t n = r.extent(0); CCField rr = r; CCConst mm = m;
  Kokkos::parallel_for("cfdk::vmg_mulmask", Kokkos::RangePolicy<CCExec>(space, 0, n),
    KOKKOS_LAMBDA(std::size_t i) { rr(i) *= mm(i); });
  space.fence();
}

// Velocity (momentum) geometric multigrid with the staircase coarse operator. All levels ghost width G=2.
class VelocityMG {
 public:
  static constexpr int G = 2;
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    std::size_t n = 0;
    CCField x, rhs, res, theta, pin, resMask;
    FPV AC, AW, AE, AS, AN, AB, AT;
  };

  // periodic uniform hierarchy (halve each axis while even and >=2, capped at nLevels).
  void init(int nx, int ny, int nz, int nLevels) {
    lv_.clear();
    C3 inner{nx, ny, nz}, cf{1, 1, 1};
    for (int L = 0; L < nLevels; ++L) {
      Level v; v.inner = inner; v.ext = C3{inner.x + 2 * G, inner.y + 2 * G, inner.z + 2 * G}; v.cfac = cf;
      v.n = (std::size_t)v.ext.x * v.ext.y * v.ext.z;
      auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
      C3 next = inner; C3 ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(inner.x)) { ratio.x = 2; next.x = inner.x / 2; }
        if (can(inner.y)) { ratio.y = 2; next.y = inner.y / 2; }
        if (can(inner.z)) { ratio.z = 2; next.z = inner.z / 2; }
      }
      v.ratio = ratio;
      v.x = CCField("vmg_x", v.n); v.rhs = CCField("vmg_rhs", v.n); v.res = CCField("vmg_res", v.n);
      v.theta = CCField("vmg_th", v.n); v.pin = CCField("vmg_pin", v.n);
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("vmg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z) break;
      inner = next; cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    lv_[0].resMask = CCField("vmg_resmask0", lv_[0].n);  // level 0 only (clean-fluid exclude)
  }
  int nLevels() const { return (int)lv_.size(); }

  // level-0 fine operator = the external IBM stencil (7 float arrays on the same G=2 block).
  void setFineStencil(FPC AC, FPC AW, FPC AE, FPC AS, FPC AN, FPC AB, FPC AT) {
    Level& f = lv_[0];
    Kokkos::deep_copy(f.AC, AC); Kokkos::deep_copy(f.AW, AW); Kokkos::deep_copy(f.AE, AE);
    Kokkos::deep_copy(f.AS, AS); Kokkos::deep_copy(f.AN, AN); Kokkos::deep_copy(f.AB, AB);
    Kokkos::deep_copy(f.AT, AT);
  }

  // staircase coarse op: level-0 pin = fine solid mask, resMask = clean-fluid mask; coarse levels classify
  // by restricted theta and build a const-coeff Helmholtz. nu_dt = mu, idiag = rho/dt, h0 = 1.
  void setStaircase(CCConst theta0, CCConst solid0, CCConst resmask0, double nu_dt, double idiag,
                    double thresh) {
    Level& f = lv_[0];
    Kokkos::deep_copy(f.theta, theta0); Kokkos::deep_copy(f.pin, solid0); Kokkos::deep_copy(f.resMask, resmask0);
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L]; Level& fin = lv_[L - 1];
      restrictAvg(c.theta, CCConst(fin.theta), c.ext, fin.ext, G, c.inner, fin.ratio);  // coarse theta = avg
      thresholdMask(c.pin, CCConst(c.theta), thresh);
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x), by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      buildVelocityStaircase(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.theta), c.ext, G, bx, by, bz,
                             thresh, idiag);
    }
  }

  // solve A x = b (b,x on the level-0 G=2 block): nvc V-cycles. Solution left in x.
  void solve(CCConst b, CCField x, int nvc, int pre, int post, int bottom) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(l0.rhs, b); Kokkos::deep_copy(l0.x, x);
    for (int v = 0; v < nvc; ++v) vcycle(0);
    Kokkos::deep_copy(x, l0.x);
  }

 public:  // (public for nvcc extended-lambda)
  void vcycle(int L) {
    Level& lv = lv_[L];
    if (L + 1 == (int)lv_.size()) { smooth(lv, bottom_); return; }  // velocity op non-singular -> no mean removal
    smooth(lv, pre_);
    fill(lv, lv.x);
    residualVarPin(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                   FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), CCConst(lv.pin), lv.ext, G);
    const bool masked = (lv.resMask.extent(0) == lv.n);  // level 0: exclude the IBM cut-cell band
    if (masked) mulMask(lv.res, CCConst(lv.resMask));
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, G, cs.inner, lv.ratio);
    Kokkos::deep_copy(cs.x, 0.0);
    vcycle(L + 1);
    fill(cs, cs.x);
    if (masked) prolongMasked(lv.x, CCConst(cs.x), CCConst(lv.resMask), lv.ext, cs.ext, G, lv.inner, lv.ratio, 0.5);
    else prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, G, lv.inner, lv.ratio);
    smooth(lv, post_);
  }
  void smooth(Level& lv, int sweeps) {
    const C3 og{0, 0, 0};
    for (int k = 0; k < sweeps; ++k)
      for (int color = 0; color < 2; ++color) {
        fill(lv, lv.x);
        ibmRbgsStencilColor(lv.x, CCConst(lv.rhs), MConst(lv.AC), MConst(lv.AW), MConst(lv.AE), MConst(lv.AS),
                            MConst(lv.AN), MConst(lv.AB), MConst(lv.AT), CCConst(lv.pin), lv.ext, og, G, color);
      }
  }
  void fill(Level& lv, CCField f) { fillAxis(lv, f, 0); fillAxis(lv, f, 1); fillAxis(lv, f, 2); }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space; C3 e = lv.ext; int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c]; const int N = N3[a]; CCField ff = f;
    Kokkos::parallel_for("cfdk::vmg_pfill", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) { ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
          ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa); } });
    space.fence();
  }

 private:
  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 8;
};

}  // namespace cfdk

#endif  // CFD_MAC_VELOCITY_MG_KOKKOS_HPP
