// cfd-gpu — portable (Kokkos) geometric multigrid for the cut-cell (variable-openness) pressure Poisson.
//
// Single-GPU (periodic) port of CUDA's DistributedPoissonMG (mac_multigrid.cuh): a level hierarchy with the
// rediscretized cut-cell operator (average-coarsen the face openness per level + re-assemble the operator at
// the coarse spacing, mg_coarsen_open_avg_k), a V-cycle with red-black Gauss-Seidel smoothing + average
// restriction + trilinear prolongation + constant-null-space (mean) removal, and an MG-PCG outer driver
// (CG preconditioned by one symmetric V-cycle). Operator stored single-precision (mreal=float) + double
// iterate, exactly as CUDA. Reuses buildCutcellOp / cutcellSmoothColor / applyCutcellOp (mac_pressure).
// Not yet ported (noted for later): Galerkin coarse op, Chebyshev smoother, semi-coarsening, domain-BC MG,
// MPI. Runs on any Kokkos backend.
#ifndef CFD_MAC_CUTCELL_MG_KOKKOS_HPP
#define CFD_MAC_CUTCELL_MG_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>

#include "mac_pressure_kokkos.hpp"
#include "mac_bc_kokkos.hpp"

namespace cfdk {

using MReal = float;                                  // operator storage = CUDA mreal
using FPV = Kokkos::View<MReal*, CCMem>;
using FPC = Kokkos::View<const MReal*, CCMem>;

// coarsen staggered face openness: each coarse face = average of the ratio_b*ratio_c fine sub-faces it spans
// (mg_coarsen_open_avg_k port).
inline void coarsenOpenAvg(CCField oxc, CCField oyc, CCField ozc, CCConst oxf, CCConst oyf, CCConst ozf,
                           C3 cext, C3 fext, int g, C3 cinner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "cfdk::coarsen_open", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const int rx = ratio.x, ry = ratio.y, rz = ratio.z;
        const int fx0 = rx * icx + g, fy0 = ry * icy + g, fz0 = rz * icz + g;
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        auto F = [&](CCConst T, int x, int y, int z) { return T((long)x + (long)y * fsy + (long)z * fsz); };
        double sx = 0, sy = 0, sz = 0;
        for (int a = 0; a < ry; ++a) for (int b = 0; b < rz; ++b) sx += F(oxf, fx0, fy0 + a, fz0 + b);
        for (int a = 0; a < rx; ++a) for (int b = 0; b < rz; ++b) sy += F(oyf, fx0 + a, fy0, fz0 + b);
        for (int a = 0; a < rx; ++a) for (int b = 0; b < ry; ++b) sz += F(ozf, fx0 + a, fy0 + b, fz0);
        const long ci = (long)(icx + g) + (long)(icy + g) * cext.x + (long)(icz + g) * (long)cext.x * cext.y;
        oxc(ci) = sx / (double)(ry * rz); oyc(ci) = sy / (double)(rx * rz); ozc(ci) = sz / (double)(rx * ry);
      });
  space.fence();
}

// residual r = b - A x for the float operator (mg_residual_var_k port).
inline void residualCutcell(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN,
                            FPC AB, FPC AT, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "cfdk::cc_residual", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                          (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                          (double)AB(i) * x(i - sz);
        r(i) = b(i) - Ax;
      });
  space.fence();
}

// average restriction (coarse = mean of ratio^3 fine children; mg_restrict_k) + trilinear prolongation
// (added to fine; mg_prolong_k). Both over inner cells.
inline void restrictAvg(CCField coarse, CCConst fine, C3 cext, C3 fext, int g, C3 cinner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "cfdk::restrict", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        double s = 0;
        for (int dz = 0; dz < ratio.z; ++dz) for (int dy = 0; dy < ratio.y; ++dy) for (int dx = 0; dx < ratio.x; ++dx) {
          const int fx = ratio.x * icx + dx + g, fy = ratio.y * icy + dy + g, fz = ratio.z * icz + dz + g;
          s += fine((long)fx + (long)fy * fsy + (long)fz * fsz);
        }
        const long ci = (long)(icx + g) + (long)(icy + g) * cext.x + (long)(icz + g) * (long)cext.x * cext.y;
        coarse(ci) = s / (double)(ratio.x * ratio.y * ratio.z);
      });
  space.fence();
}
inline void prolongAdd(CCField fine, CCConst coarse, C3 fext, C3 cext, int g, C3 finner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "cfdk::prolong", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        // coarse sample coord: coarsened axis (ratio 2) -> 0.5*ifine - 0.25 + g; kept axis (ratio 1) -> ifine+g
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
        const long fi = (long)(ifx + g) + (long)(ify + g) * fext.x + (long)(ifz + g) * (long)fext.x * fext.y;
        fine(fi) += c0 * (1 - wz) + c1 * wz;
      });
  space.fence();
}

class CutcellMG {
 public:
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    std::size_t n = 0;
    CCField x, rhs, res, ox, oy, oz;
    FPV AC, AW, AE, AS, AN, AB, AT;
  };
  static constexpr int G = 1;

  // build the periodic level hierarchy: per axis, halve inner while even and >=2 (uniform when cubic),
  // capped at nLevels (mirrors DistributedPoissonMG::init uniform path).
  void init(int nx, int ny, int nz, int nLevels) {
    lv_.clear();
    C3 inner{nx, ny, nz}, cf{1, 1, 1};
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.inner = inner;
      v.ext = C3{inner.x + 2 * G, inner.y + 2 * G, inner.z + 2 * G};
      v.cfac = cf;
      v.n = (std::size_t)v.ext.x * v.ext.y * v.ext.z;
      auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
      C3 next = inner; C3 ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(inner.x)) { ratio.x = 2; next.x = inner.x / 2; }
        if (can(inner.y)) { ratio.y = 2; next.y = inner.y / 2; }
        if (can(inner.z)) { ratio.z = 2; next.z = inner.z / 2; }
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n); v.rhs = CCField("mg_rhs", v.n); v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n); v.oy = CCField("mg_oy", v.n); v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z) break;  // nothing coarsens
      inner = next; cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
  }
  int nLevels() const { return (int)lv_.size(); }
  Level& level(int L) { return lv_[L]; }

  // per-face domain BC types {-x,+x,-y,+y,-z,+z}: 0=periodic, 1/2=Neumann (wall/inflow), 3=Dirichlet
  // (outflow). Default all-periodic -> applyBoundaryOpenness is a no-op (periodic/IBM path byte-identical).
  void setBoundaryConditions(const int bc[6]) {
    hasBC_ = false; hasOutflow_ = false;
    for (int i = 0; i < 6; ++i) { bc_[i] = bc[i]; if (bc[i]) hasBC_ = true; if (bc[i] == 3) hasOutflow_ = true; }
    removeMean_ = !hasOutflow_;  // singular all-Neumann -> remove mean; Dirichlet outflow -> non-singular
  }
  // hold the pressure/correction ghost at 0 on outflow faces (open face -> Dirichlet p=0). Call after every
  // (periodic) fill of a solution / search-direction field, on the level it lives.
  void applyOutflowGhost(C3 ext, CCField x) {
    if (!hasOutflow_) return;
    B3 e{ext.x, ext.y, ext.z};
    for (int a = 0; a < 3; ++a) for (int s = 0; s < 2; ++s) if (bc_[2 * a + s] == 3) bcZeroPressureGhost(x, e, G, a, s);
  }
  // re-impose the non-periodic boundary openness a periodic fill leaves wrong: Neumann wall/inflow -> 0
  // (closed), Dirichlet outflow -> left open. Call after every (periodic) openness fill, per level.
  void applyBoundaryOpenness(Level& lv) {
    if (!hasBC_) return;
    B3 e{lv.ext.x, lv.ext.y, lv.ext.z};
    CCField oa[3] = {lv.ox, lv.oy, lv.oz};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (t == 1 || t == 2) bcSetOpenness(oa[a], e, G, a, s, 0.0);  // wall/inflow Neumann -> closed
        else if (t == 3) bcSetOpenness(oa[a], e, G, a, s, 1.0);       // outflow -> open (periodic fill wraps wrong)
      }
  }

  // rediscretized cut-cell operator on every level from the fine face openness (idx2 = 1/dx^2 fine).
  void setOpenness(CCConst ox, CCConst oy, CCConst oz, double idx2, double idy2, double idz2) {
    Level& f = lv_[0];
    Kokkos::deep_copy(f.ox, ox); Kokkos::deep_copy(f.oy, oy); Kokkos::deep_copy(f.oz, oz);
    fillOpenness(f);  // periodic fine-level openness ghosts (the operator reads the + neighbour face);
                      // idempotent when the caller already filled them, required when it passed inner-only.
    applyBoundaryOpenness(f);  // re-impose non-periodic wall/inflow faces the periodic fill clobbered
    buildCutcellOp(f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT, CCConst(f.ox), CCConst(f.oy), CCConst(f.oz),
                   f.ext, G, idx2, idy2, idz2);
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L]; Level& fin = lv_[L - 1];
      coarsenOpenAvg(c.ox, c.oy, c.oz, CCConst(fin.ox), CCConst(fin.oy), CCConst(fin.oz), c.ext, fin.ext, G,
                     c.inner, fin.ratio);
      fillOpenness(c);  // periodic ghost openness (operator build reads the + neighbour face)
      applyBoundaryOpenness(c);  // re-impose non-periodic boundary faces per coarse level
      const double sx = 1.0 / (double)(c.cfac.x * c.cfac.x), sy = 1.0 / (double)(c.cfac.y * c.cfac.y),
                   sz = 1.0 / (double)(c.cfac.z * c.cfac.z);
      buildCutcellOp(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.ox), CCConst(c.oy), CCConst(c.oz),
                     c.ext, G, idx2 * sx, idy2 * sy, idz2 * sz);
    }
  }

  // CG preconditioned by one symmetric V-cycle (solve_pcg port). rhs on level 0; solution left in level-0 x.
  // Returns the iteration count. Scratch supplied by the caller (level-0-sized fields).
  int solvePCG(CCField b, CCField x, CCField r, CCField p, CCField z, CCField Ap, int maxit, double rtol,
               int pre, int post, int bottom) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(l0.x, x);
    auto matvec = [&](CCField y, CCField v) {
      fill(l0, v); applyOutflowGhost(l0.ext, v);
      applyCutcellOp(y, CCConst(v), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN), FPC(l0.AB),
                     FPC(l0.AT), l0.ext, G);
    };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(l0.rhs, rr); Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(zz, l0.x);
    };
    matvec(Ap, x);                                  // r = b - A x
    Kokkos::deep_copy(r, b); axpy(r, -1.0, Ap);
    removeMean(l0, r);                              // compatibility: project rhs/residual onto the range
    const double r0 = maxabs(l0, r);
    int it = 0;
    if (r0 > 0.0) {
      precond(z, r);
      Kokkos::deep_copy(p, z);
      double rz = dot(l0, r, z);
      for (; it < maxit; ++it) {
        matvec(Ap, p); removeMean(l0, Ap);
        const double pAp = dot(l0, p, Ap);
        if (pAp <= 1e-300) break;
        const double alpha = rz / pAp;
        axpy(x, alpha, p); axpy(r, -alpha, Ap); removeMean(l0, r);
        if (maxabs(l0, r) < rtol * r0) { ++it; break; }
        precond(z, r);
        const double rznew = dot(l0, r, z), beta = rznew / rz;
        aypx(p, beta, z);
        rz = rznew;
      }
    }
    Kokkos::deep_copy(l0.x, x); removeMean(l0, l0.x);
    Kokkos::deep_copy(x, l0.x);
    return it;
  }

 public:  // (public for nvcc extended-lambda rule)
  void vcycle(int L, bool sym) {
    Level& lv = lv_[L];
    if (L + 1 == (int)lv_.size()) { smooth(lv, bottom_, false); removeMean(lv, lv.x); return; }
    smooth(lv, pre_, false);
    residualCutcell(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                    FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, G);
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, G, cs.inner, lv.ratio);
    Kokkos::deep_copy(cs.x, 0.0);
    vcycle(L + 1, sym);
    fill(cs, cs.x); applyOutflowGhost(cs.ext, cs.x);
    prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, G, lv.inner, lv.ratio);
    smooth(lv, post_, /*reverse=*/sym);
    removeMean(lv, lv.x);
  }
  void smooth(Level& lv, int sweeps, bool reverse) {
    const C3 og{0, 0, 0};
    for (int k = 0; k < sweeps; ++k)
      for (int s = 0; s < 2; ++s) {
        const int color = reverse ? (1 - s) : s;
        fill(lv, lv.x); applyOutflowGhost(lv.ext, lv.x);
        cutcellSmoothColor(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                           FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, G, color);
      }
  }
  // periodic ghost fill (3 axes) of a level-sized field / the openness triple.
  void fill(Level& lv, CCField f) { fillAxis(lv, f, 0); fillAxis(lv, f, 1); fillAxis(lv, f, 2); }
  void fillOpenness(Level& lv) { fill(lv, lv.ox); fill(lv, lv.oy); fill(lv, lv.oz); }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space; C3 e = lv.ext; int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c]; const int N = N3[a]; CCField ff = f;
    Kokkos::parallel_for("cfdk::mg_pfill", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) { ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
          ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa); } });
    space.fence();
  }
  void axpy(CCField y, double a, CCField x) {
    CCExec space; CCField yy = y, xx = x; std::size_t n = y.extent(0);
    Kokkos::parallel_for("mgaxpy", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { yy(i) += a * xx(i); });
    space.fence();
  }
  void aypx(CCField y, double a, CCField x) {
    CCExec space; CCField yy = y, xx = x; std::size_t n = y.extent(0);
    Kokkos::parallel_for("mgaypx", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { yy(i) = xx(i) + a * yy(i); });
    space.fence();
  }
  // reductions / mean removal over inner FLUID cells (AC>tiny) of a level.
  double dot(Level& lv, CCField a, CCField b) {
    CCExec space; C3 e = lv.ext; CCField aa = a, bb = b; FPV ac = lv.AC; double s = 0;
    Kokkos::parallel_reduce("mgdot", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) acc += aa(i) * bb(i); }, s);
    return s;
  }
  double maxabs(Level& lv, CCField a) {
    CCExec space; C3 e = lv.ext; CCField aa = a; FPV ac = lv.AC; double m = 0;
    Kokkos::parallel_reduce("mgmax", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) { const double v = Kokkos::fabs(aa(i)); if (v > acc) acc = v; } }, Kokkos::Max<double>(m));
    return m;
  }
  void removeMean(Level& lv, CCField f) {
    if (!removeMean_) return;  // non-singular operator (Dirichlet outflow present) -> no null-space projection
    CCExec space; C3 e = lv.ext; CCField ff = f; FPV ac = lv.AC; double sum = 0; long cnt = 0;
    Kokkos::parallel_reduce("mgmeanr", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& s, long& k) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) { s += ff(i); k += 1; } }, sum, cnt);
    if (cnt == 0) return; const double mean = sum / (double)cnt;
    Kokkos::parallel_for("mgmeans", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y; if (ac(i) > 1e-30f) ff(i) -= mean; });
    space.fence();
  }

 private:
  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 4;
  int bc_[6] = {0, 0, 0, 0, 0, 0}; bool hasBC_ = false, removeMean_ = true, hasOutflow_ = false;
};

}  // namespace cfdk

#endif  // CFD_MAC_CUTCELL_MG_KOKKOS_HPP
