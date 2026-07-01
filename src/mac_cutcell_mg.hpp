/// @file
/// @brief sdflow — portable (Kokkos) geometric multigrid for the cut-cell (variable-openness) pressure Poisson.
///
/// Single-GPU (periodic) port of CUDA's DistributedPoissonMG (mac_multigrid.cuh): a level hierarchy with the
/// rediscretized cut-cell operator (average-coarsen the face openness per level + re-assemble the operator at
/// the coarse spacing, mg_coarsen_open_avg_k), a V-cycle with red-black Gauss-Seidel smoothing + average
/// restriction + trilinear prolongation + constant-null-space (mean) removal, and an MG-PCG outer driver
/// (CG preconditioned by one symmetric V-cycle). Operator stored single-precision (mreal=float) + double
/// iterate, exactly as CUDA. Reuses buildCutcellOp / cutcellSmoothColor / applyCutcellOp (mac_pressure).
/// Not yet ported (noted for later): Galerkin coarse op, Chebyshev smoother, semi-coarsening, domain-BC MG,
/// MPI. Runs on any Kokkos backend.
#ifndef PECLET_FLOW_MAC_CUTCELL_MG_HPP
#define PECLET_FLOW_MAC_CUTCELL_MG_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>

#include "mac_pressure.hpp"
#include "mac_bc.hpp"

// Multi-rank (MPI) path is opt-in: the single-GPU module never links MPI, so all distributed code is gated
// (mirrors the CUDA PECLET_FLOW_BUILD_MPI gating). When PECLET_FLOW_MPI is off, CutcellMG is byte-identical to before.
#ifdef PECLET_FLOW_MPI
#include <memory>
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#endif

namespace peclet::flow {

#ifdef PECLET_FLOW_MPI
using peclet::core::halo::GridHaloTopology;
using peclet::core::halo::GridHalo;
#endif

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
      "peclet::flow::coarsen_open", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
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

}

// residual r = b - A x for the float operator (mg_residual_var_k port).
inline void residualCutcell(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN,
                            FPC AB, FPC AT, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::cc_residual", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                          (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                          (double)AB(i) * x(i - sz);
        r(i) = b(i) - Ax;
      });

}

// average restriction (coarse = mean of ratio^3 fine children; mg_restrict_k) + trilinear prolongation
// (added to fine; mg_prolong_k). Both over inner cells.
inline void restrictAvg(CCField coarse, CCConst fine, C3 cext, C3 fext, int g, C3 cinner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::restrict", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
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

}
inline void prolongAdd(CCField fine, CCConst coarse, C3 fext, C3 cext, int g, C3 finner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::prolong", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
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

}

class CutcellMG {
 public:
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    C3 og{0, 0, 0};            // block inner origin (global red-black parity); {0,0,0} single-rank
    std::size_t n = 0;
    CCField x, rhs, res, ox, oy, oz;
    FPV AC, AW, AE, AS, AN, AB, AT;
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;                              // per-level topology (decomposed)
    std::shared_ptr<GridHalo<double>> dev;          // per-level ghost exchange
#endif
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
#ifdef PECLET_FLOW_MPI
  // Multi-rank hierarchy: coarsen the GLOBAL grid 2:1 per level; each level gets its own transport-core halo
  // over a BlockDecomposer of that level's grid (the ORB decomposition coarsens cleanly so restrict/prolong
  // stay local). Sets the distributed flag -> fill() exchanges, the reductions Allreduce, the smoother uses
  // the block's global-origin parity. Single-rank (size 1) reproduces init()'s field exactly.
  void initMpi(int gnx, int gny, int gnz, int nLevels, MPI_Comm comm) {
    lv_.clear(); distributed_ = true; comm_ = comm;
    int rank = 0, size = 1; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    std::array<bool, 3> per{true, true, true};
    C3 gs{gnx, gny, gnz}, cf{1, 1, 1};
    auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.halo = std::make_shared<GridHaloTopology<3>>();
      peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), peclet::core::IVec<3>{gs.x, gs.y, gs.z});
      v.halo->buildTopology(dec, rank, G, per, comm);
      v.dev = std::make_shared<GridHalo<double>>(); v.dev->init(*v.halo);
      const auto& idx = v.halo->indexer();
      const auto eg = idx.sizeInclGhost(), ino = idx.sizeInner(), oig = idx.originInclGhost();
      v.ext = {(int)eg[0], (int)eg[1], (int)eg[2]}; v.inner = {(int)ino[0], (int)ino[1], (int)ino[2]};
      v.og = {(int)oig[0] + G, (int)oig[1] + G, (int)oig[2] + G};  // inner origin == single-rank og=0 at origin 0
      v.cfac = cf; v.n = idx.numCellsInclGhost();
      C3 next = gs, ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(gs.x)) { ratio.x = 2; next.x = gs.x / 2; }
        if (can(gs.y)) { ratio.y = 2; next.y = gs.y / 2; }
        if (can(gs.z)) { ratio.z = 2; next.z = gs.z / 2; }
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n); v.rhs = CCField("mg_rhs", v.n); v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n); v.oy = CCField("mg_oy", v.n); v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z) break;
      gs = next; cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
  }
#endif
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
    const C3 og = lv.og;  // global red-black parity (block inner origin); {0,0,0} single-rank
    for (int k = 0; k < sweeps; ++k)
      for (int s = 0; s < 2; ++s) {
        const int color = reverse ? (1 - s) : s;
        fill(lv, lv.x); applyOutflowGhost(lv.ext, lv.x);
        cutcellSmoothColor(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                           FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, G, color);
      }
  }
  // periodic ghost fill (3 axes) of a level-sized field / the openness triple. Distributed: the per-level
  // transport-core halo (cross-rank + periodic in one call).
  void fill(Level& lv, CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) { lv.dev->exchange(f); return; }
#endif
    fillAxis(lv, f, 0); fillAxis(lv, f, 1); fillAxis(lv, f, 2);
  }
  void fillOpenness(Level& lv) { fill(lv, lv.ox); fill(lv, lv.oy); fill(lv, lv.oz); }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space; C3 e = lv.ext; int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c]; const int N = N3[a]; CCField ff = f;
    Kokkos::parallel_for("peclet::flow::mg_pfill", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) { ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
          ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa); } });

  }
  void axpy(CCField y, double a, CCField x) {
    CCExec space; CCField yy = y, xx = x; std::size_t n = y.extent(0);
    Kokkos::parallel_for("mgaxpy", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { yy(i) += a * xx(i); });

  }
  void aypx(CCField y, double a, CCField x) {
    CCExec space; CCField yy = y, xx = x; std::size_t n = y.extent(0);
    Kokkos::parallel_for("mgaypx", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { yy(i) = xx(i) + a * yy(i); });

  }
  void scale(CCField y, double a) {
    CCExec space; CCField yy = y; std::size_t n = y.extent(0);
    Kokkos::parallel_for("mgscale", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { yy(i) *= a; });

  }
  void lin(CCField out, double a, CCField x, double b, CCField y) {  // out = a*x + b*y (mg_lin_k)
    CCExec space; CCField oo = out, xx = x, yy = y; std::size_t n = out.extent(0);
    Kokkos::parallel_for("mglin", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) { oo(i) = a * xx(i) + b * yy(i); });

  }
  // zero the solid-cell entries (AC<=tiny) -> project out the solid null modes (mg_mask_solid_k).
  void maskSolid(Level& lv, CCField f) {
    CCExec space; CCField ff = f; FPV ac = lv.AC; std::size_t n = f.extent(0);
    Kokkos::parallel_for("mgmasksolid", Kokkos::RangePolicy<CCExec>(space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) { if (!(ac(i) > 1e-30f)) ff(i) = 0.0; });

  }

  // Estimate the spectral bounds [lmin,lmax] of M^{-1}A (M^{-1} = one symmetric V-cycle) by power iteration
  // (direct for the max + a shifted iteration for the min), seeded by `seed`. Communication-heavy, so the
  // CUDA driver runs it once on step 1 and reuses the bounds. Port of estimate_eigenvalues.
  void estimateEigenvalues(CCConst seed, double& lmin, double& lmax, int iters, int pre, int post, int bottom) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    Level& l0 = lv_[0]; const std::size_t n = l0.n;
    CCField v("ev_v", n), w("ev_w", n), z("ev_z", n), srhs("ev_srhs", n);
    Kokkos::deep_copy(srhs, seed);
    auto matvec = [&](CCField y, CCField x) {
      fill(l0, x); applyOutflowGhost(l0.ext, x);
      applyCutcellOp(y, CCConst(x), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN), FPC(l0.AB),
                     FPC(l0.AT), l0.ext, G);
    };
    auto applyT = [&](CCField out, CCField in) {  // out = M^{-1} A in, projected onto the fluid range
      matvec(w, in);
      Kokkos::deep_copy(l0.rhs, w); Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(out, l0.x);
      removeMean(l0, out); maskSolid(l0, out);
    };
    auto normalize = [&](CCField x) { double nr = std::sqrt(dot(l0, x, x)); if (nr > 0) scale(x, 1.0 / nr); };
    auto seedf = [&](CCField x) { Kokkos::deep_copy(x, srhs); removeMean(l0, x); maskSolid(l0, x); normalize(x); };
    seedf(v);
    lmax = 1.0;
    for (int k = 0; k < iters; ++k) { applyT(z, v); lmax = dot(l0, v, z); Kokkos::deep_copy(v, z); normalize(v); }
    seedf(v);
    double mu = 0.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v); lin(z, lmax, v, -1.0, z);  // z = lmax*v - T v
      mu = dot(l0, v, z); Kokkos::deep_copy(v, z); normalize(v);
    }
    double e_hi = lmax, e_lo = lmax - mu;        // direct (max) + shifted (min) Rayleigh estimates
    lmin = e_lo < e_hi ? e_lo : e_hi; lmax = e_lo < e_hi ? e_hi : e_lo;  // robust min/max bracket
    if (lmin < 0.02 * lmax) lmin = 0.02 * lmax;
  }

  // Chebyshev semi-iteration preconditioned by ONE symmetric V-cycle -- same goal as solvePCG but the step
  // coefficients come from the spectral bounds [a,b], so NO per-iteration global dot-products (communication-
  // light at scale). rhs on level-0 supplied as `b`; solution left in `x`. Returns the V-cycle count. Port of
  // solve_chebyshev.
  int solveChebyshev(CCField b, CCField x, int maxit, double rtol, int pre, int post, int bottom, double a, double bnd) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    Level& l0 = lv_[0]; const std::size_t n = l0.n;
    if (a > bnd) { double t = a; a = bnd; bnd = t; }  // robust to swapped bounds
    a *= 0.95; bnd *= 1.05;                            // safety margin: [a,b] must bracket the spectrum
    CCField r("cb_r", n), z("cb_z", n), d("cb_d", n), w("cb_w", n);
    auto matvec = [&](CCField y, CCField v) {
      fill(l0, v); applyOutflowGhost(l0.ext, v);
      applyCutcellOp(y, CCConst(v), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN), FPC(l0.AB),
                     FPC(l0.AT), l0.ext, G);
    };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(l0.rhs, rr); Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true); Kokkos::deep_copy(zz, l0.x);
    };
    const double theta = 0.5 * (bnd + a), delta = 0.5 * (bnd - a), sigma1 = theta / delta;
    double rho = 1.0 / sigma1;
    matvec(w, x);                                     // r = b - A x
    Kokkos::deep_copy(r, b); axpy(r, -1.0, w); removeMean(l0, r);
    const double r0 = maxabs(l0, r);
    int nvc = 0;
    if (r0 > 0.0) {
      precond(z, r); ++nvc;                           // z = M^{-1} r
      lin(d, 1.0 / theta, z, 0.0, z); axpy(x, 1.0, d);  // d = z/theta; x += d
      for (int i = 1; i < maxit; ++i) {
        matvec(w, d); axpy(r, -1.0, w); removeMean(l0, r);  // r -= A d
        if (maxabs(l0, r) < rtol * r0) break;
        precond(z, r); ++nvc;
        const double rho_new = 1.0 / (2.0 * sigma1 - rho);
        lin(d, rho_new * rho, d, 2.0 * rho_new / delta, z); axpy(x, 1.0, d);  // d update; x += d
        rho = rho_new;
      }
    }
    removeMean(l0, x);
    return nvc;
  }
  // reductions / mean removal over inner FLUID cells (AC>tiny) of a level.
  double dot(Level& lv, CCField a, CCField b) {
    CCExec space; C3 e = lv.ext; CCField aa = a, bb = b; FPV ac = lv.AC; double s = 0;
    Kokkos::parallel_reduce("mgdot", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) acc += aa(i) * bb(i); }, s);
    return allreduce(s, MPI_SUM_);
  }
  double maxabs(Level& lv, CCField a) {
    CCExec space; C3 e = lv.ext; CCField aa = a; FPV ac = lv.AC; double m = 0;
    Kokkos::parallel_reduce("mgmax", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) { const double v = Kokkos::fabs(aa(i)); if (v > acc) acc = v; } }, Kokkos::Max<double>(m));
    return allreduce(m, MPI_MAX_);
  }
  void removeMean(Level& lv, CCField f) {
    if (!removeMean_) return;  // non-singular operator (Dirichlet outflow present) -> no null-space projection
    CCExec space; C3 e = lv.ext; CCField ff = f; FPV ac = lv.AC; double sum = 0; long cnt = 0;
    Kokkos::parallel_reduce("mgmeanr", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& s, long& k) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (ac(i) > 1e-30f) { s += ff(i); k += 1; } }, sum, cnt);
    sum = allreduce(sum, MPI_SUM_); cnt = (long)allreduce((double)cnt, MPI_SUM_);  // global fluid sum + count
    if (cnt == 0) return; const double mean = sum / (double)cnt;
    Kokkos::parallel_for("mgmeans", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) { const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y; if (ac(i) > 1e-30f) ff(i) -= mean; });

  }

 private:
  enum AllOp { kSum, kMax };
  // Global reduction over ranks (no-op single-rank / non-distributed -> byte-identical to the local reduce).
  double allreduce(double v, AllOp op) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) { double g = 0; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, op == kSum ? MPI_SUM : MPI_MAX, comm_); return g; }
#endif
    (void)op; return v;
  }
  static constexpr AllOp MPI_SUM_ = kSum, MPI_MAX_ = kMax;

  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 4;
  int bc_[6] = {0, 0, 0, 0, 0, 0}; bool hasBC_ = false, removeMean_ = true, hasOutflow_ = false;
  bool distributed_ = false;
#ifdef PECLET_FLOW_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CUTCELL_MG_HPP
