/// @file
/// @brief flow — portable (Kokkos) geometric multigrid for the cut-cell (variable-openness)
/// pressure Poisson.
///
/// Single-GPU (periodic) port of CUDA's DistributedPoissonMG (mac_multigrid.cuh): a level hierarchy
/// with the rediscretized cut-cell operator (average-coarsen the face openness per level +
/// re-assemble the operator at the coarse spacing, mg_coarsen_open_avg_k), a V-cycle with red-black
/// Gauss-Seidel smoothing + average restriction + trilinear prolongation + constant-null-space
/// (mean) removal, and an MG-PCG outer driver (CG preconditioned by one symmetric V-cycle).
/// Operator stored single-precision (mreal=float) + double iterate, exactly as CUDA. Reuses
/// buildCutcellOp / cutcellSmoothColor / applyCutcellOp (mac_pressure). Not yet ported (noted for
/// later): Galerkin coarse op, Chebyshev smoother, semi-coarsening, domain-BC MG, MPI. Runs on any
/// Kokkos backend.
#ifndef PECLET_FLOW_MAC_CUTCELL_MG_HPP
#define PECLET_FLOW_MAC_CUTCELL_MG_HPP

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>

#include "ghost_projection.hpp"  // GpOverlay + gpApplyDelta (ghost-projection BiCGStab matvec)
#include "star_elimination.hpp"  // StarOverlay + starApplyDelta (mode-B fluid-only PCG matvec)
#include "mac_bc.hpp"
#include "mac_pressure.hpp"
#include "peclet/core/solver/graph_amg.hpp"  // decomposition-agnostic algebraic bottom solve

// Multi-rank (MPI) path is opt-in: the single-GPU module never links MPI, so all distributed code
// is gated (mirrors the CUDA PECLET_FLOW_BUILD_MPI gating). When PECLET_FLOW_MPI is off, CutcellMG
// is byte-identical to before.
#ifdef PECLET_FLOW_MPI
#include <memory>

#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/grid_redistribute.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#endif

namespace peclet::flow {

#ifdef PECLET_FLOW_MPI
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;
#endif

using MReal = float;  // operator storage = CUDA mreal
using FPV = Kokkos::View<MReal*, CCMem>;
using FPC = Kokkos::View<const MReal*, CCMem>;

// coarsen staggered face openness: each coarse face = average of the ratio_b*ratio_c fine sub-faces
// it spans (mg_coarsen_open_avg_k port). gc/gf: coarse/fine block ghost widths (they can differ —
// CA-eligible coarse levels carry g=2).
inline void coarsenOpenAvg(CCField oxc, CCField oyc, CCField ozc, CCConst oxf, CCConst oyf,
                           CCConst ozf, C3 cext, C3 fext, int gc, int gf, C3 cinner, C3 ratio) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::coarsen_open", MD(space, {0, 0, 0}, {cinner.x, cinner.y, cinner.z}),
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const int rx = ratio.x, ry = ratio.y, rz = ratio.z;
        const int fx0 = rx * icx + gf, fy0 = ry * icy + gf, fz0 = rz * icz + gf;
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        auto F = [&](CCConst T, int x, int y, int z) {
          return T((long)x + (long)y * fsy + (long)z * fsz);
        };
        double sx = 0, sy = 0, sz = 0;
        for (int a = 0; a < ry; ++a)
          for (int b = 0; b < rz; ++b)
            sx += F(oxf, fx0, fy0 + a, fz0 + b);
        for (int a = 0; a < rx; ++a)
          for (int b = 0; b < rz; ++b)
            sy += F(oyf, fx0 + a, fy0, fz0 + b);
        for (int a = 0; a < rx; ++a)
          for (int b = 0; b < ry; ++b)
            sz += F(ozf, fx0 + a, fy0 + b, fz0);
        const long ci =
            (long)(icx + gc) + (long)(icy + gc) * cext.x + (long)(icz + gc) * (long)cext.x * cext.y;
        oxc(ci) = sx / (double)(ry * rz);
        oyc(ci) = sy / (double)(rx * rz);
        ozc(ci) = sz / (double)(rx * ry);
      });
}

// residual r = b - A x for the float operator (mg_residual_var_k port).
inline void residualCutcell(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN,
                            FPC AB, FPC AT, C3 e, int g) {
  ccFor3(
      "peclet::flow::cc_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) +
                          (double)AW(i) * x(i - sx) + (double)AN(i) * x(i + sy) +
                          (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                          (double)AB(i) * x(i - sz);
        r(i) = b(i) - Ax;
      });
}

// Residual over a box [rlo,rhi) minus a skip box [slo,shi) — the halo-overlapped form of
// residualCutcell (interior first while the exchange is in flight, then the boundary shell).
inline void residualCutcellBox(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS,
                               FPC AN, FPC AB, FPC AT, C3 e, C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  ccFor3(
      "peclet::flow::cc_residual_box", rlo, rhi, KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;  // inside the skip box (already done by the interior pass)
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) +
                          (double)AW(i) * x(i - sx) + (double)AN(i) * x(i + sy) +
                          (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                          (double)AB(i) * x(i - sz);
        r(i) = b(i) - Ax;
      });
}

// average restriction (coarse = mean of ratio^3 fine children; mg_restrict_k) + trilinear
// prolongation (added to fine; mg_prolong_k). Both over inner cells. gc/gf: coarse/fine block
// ghost widths (CA-eligible coarse levels carry g=2, so they can differ across one transfer).
inline void restrictAvg(CCField coarse, CCConst fine, C3 cext, C3 fext, int gc, int gf, C3 cinner,
                        C3 ratio) {
  ccFor3(
      "peclet::flow::restrict", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        double s = 0;
        for (int dz = 0; dz < ratio.z; ++dz)
          for (int dy = 0; dy < ratio.y; ++dy)
            for (int dx = 0; dx < ratio.x; ++dx) {
              const int fx = ratio.x * icx + dx + gf, fy = ratio.y * icy + dy + gf,
                        fz = ratio.z * icz + dz + gf;
              s += fine((long)fx + (long)fy * fsy + (long)fz * fsz);
            }
        const long ci =
            (long)(icx + gc) + (long)(icy + gc) * cext.x + (long)(icz + gc) * (long)cext.x * cext.y;
        coarse(ci) = s / (double)(ratio.x * ratio.y * ratio.z);
      });
}
inline void prolongAdd(CCField fine, CCConst coarse, C3 fext, C3 cext, int gf, int gc, C3 finner,
                       C3 ratio) {
  ccFor3(
      "peclet::flow::prolong", C3{0, 0, 0}, C3{finner.x, finner.y, finner.z},
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        // coarse sample coord: coarsened axis (ratio 2) -> 0.5*ifine - 0.25 + gc; kept axis (ratio
        // 1) -> ifine+gc
        const double cx = (ratio.x == 2) ? 0.5 * ifx - 0.25 + gc : ifx + gc;
        const double cy = (ratio.y == 2) ? 0.5 * ify - 0.25 + gc : ify + gc;
        const double cz = (ratio.z == 2) ? 0.5 * ifz - 0.25 + gc : ifz + gc;
        const double fxw = Kokkos::floor(cx), fyw = Kokkos::floor(cy), fzw = Kokkos::floor(cz);
        const double wx = cx - fxw, wy = cy - fyw, wz = cz - fzw;
        const int x0 = (int)fxw, y0 = (int)fyw, z0 = (int)fzw;
        const long sy = cext.x, sz = (long)cext.x * cext.y;
        auto C = [&](int xx, int yy, int zz) {
          return coarse((long)xx + (long)yy * sy + (long)zz * sz);
        };
        const double c00 = C(x0, y0, z0) * (1 - wx) + C(x0 + 1, y0, z0) * wx;
        const double c10 = C(x0, y0 + 1, z0) * (1 - wx) + C(x0 + 1, y0 + 1, z0) * wx;
        const double c01 = C(x0, y0, z0 + 1) * (1 - wx) + C(x0 + 1, y0, z0 + 1) * wx;
        const double c11 = C(x0, y0 + 1, z0 + 1) * (1 - wx) + C(x0 + 1, y0 + 1, z0 + 1) * wx;
        const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
        const long fi =
            (long)(ifx + gf) + (long)(ify + gf) * fext.x + (long)(ifz + gf) * (long)fext.x * fext.y;
        fine(fi) += c0 * (1 - wz) + c1 * wz;
      });
}

// Env-gated MG diagnostics (host printf only, off unless PECLET_FLOW_MG_DEBUG is set):
//   1 = level table (per-level global/local dims + coarsening ratio) at build time
//   2 = + PCG/V-cycle residual history for the first PECLET_FLOW_MG_DEBUG_SOLVES (default 3) solves
// Used to diagnose decomposition-dependent iteration counts; no effect on the solve itself.
inline int mgDebugLevel() {
  static const int lv = [] {
    const char* e = std::getenv("PECLET_FLOW_MG_DEBUG");
    return e ? std::atoi(e) : 0;
  }();
  return lv;
}
inline int mgDebugSolves() {
  static const int n = [] {
    const char* e = std::getenv("PECLET_FLOW_MG_DEBUG_SOLVES");
    return e ? std::atoi(e) : 3;
  }();
  return n;
}

// Communication-avoiding smoothing (PECLET_FLOW_CA): exchange a 2-deep ghost layer once per
// red-black PAIR instead of 1-deep before every colour, redundantly re-smoothing the 1-deep ghost
// ring of the first colour so the second colour reads exactly the values a per-colour exchange
// would have delivered — bit-identical, at half the halo events. Consumed by CutcellMG's coarse
// levels and by the momentum RB-GS in flow_ibm.hpp. PECLET_FLOW_CA values: unset / "1" = both
// (default), "0" = off, "mom" = momentum sweeps only, "mg" = pressure-MG coarse levels only —
// the split exists to ATTRIBUTE a measured regression to one subsystem without a rebuild.
enum : int { kCaMomentum = 1, kCaMg = 2 };
inline int caSmoothingMode() {
  static const int v = [] {
    const char* e = std::getenv("PECLET_FLOW_CA");
    if (!e)
      return kCaMomentum | kCaMg;
    const std::string s(e);
    if (s == "mom" || s == "momentum")
      return (int)kCaMomentum;
    if (s == "mg")
      return (int)kCaMg;
    return std::atoi(e) != 0 ? (kCaMomentum | kCaMg) : 0;
  }();
  return v;
}

class CutcellMG {
 public:
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    C3 og{0, 0, 0};  // block inner origin in GLOBAL cells; {0,0,0} single-rank
    // This level's GLOBAL inner dims (== inner single-rank). With og it decides which ranks own a
    // global domain face — see touchesGlobalFace(lv, f).
    C3 gdim{0, 0, 0};
    std::size_t n = 0;
    // Ghost width of this level's block (1 default; 2 on distributed coarse levels eligible for
    // communication-avoiding smoothing — see initMpi). Single-rank always 1 (byte-identical).
    int g = 1;
    bool caOk = false;  // width-2 topology built and every rank's block extent >= 4
    CCField x, rhs, res, ox, oy, oz;
    FPV AC, AW, AE, AS, AN, AB, AT;
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;  // per-level topology (decomposed)
    std::shared_ptr<GridHalo<double>> dev;      // per-level ghost exchange
#endif
  };
  static constexpr int G = 1;  // level-0 / single-rank ghost width (the flow_ibm g=1 bridge)
  // The red-black parity origin for a level's smoother: the parity convention is the single-rank
  // g=1 one (parity of og+local index INCLUDING a 1-cell ghost offset), so a level with g=2 must
  // shift its origin by g-1 per axis or its colours come out swapped against the g=1 reference
  // (3 axes -> parity flips). og itself stays the true global inner origin (buildAmg needs it).
  static C3 parityOg(const Level& lv) {
    return C3{lv.og.x - lv.g + 1, lv.og.y - lv.g + 1, lv.og.z - lv.g + 1};
  }

  // build the periodic level hierarchy: per axis, halve inner while even and >=2 (uniform when
  // cubic), capped at nLevels (mirrors DistributedPoissonMG::init uniform path).
  void init(int nx, int ny, int nz, int nLevels) {
    lv_.clear();
    amg_.reset();
    gnxF_ = nx;
    gnyF_ = ny;
    gnzF_ = nz;
    C3 inner{nx, ny, nz}, cf{1, 1, 1};
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.inner = inner;
      v.gdim = inner;  // single-rank: the block IS the global grid
      v.ext = C3{inner.x + 2 * G, inner.y + 2 * G, inner.z + 2 * G};
      v.cfac = cf;
      v.n = (std::size_t)v.ext.x * v.ext.y * v.ext.z;
      auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
      C3 next = inner;
      C3 ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(inner.x)) {
          ratio.x = 2;
          next.x = inner.x / 2;
        }
        if (can(inner.y)) {
          ratio.y = 2;
          next.y = inner.y / 2;
        }
        if (can(inner.z)) {
          ratio.z = 2;
          next.z = inner.z / 2;
        }
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n);
      v.rhs = CCField("mg_rhs", v.n);
      v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n);
      v.oy = CCField("mg_oy", v.n);
      v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT})
        *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z)
        break;  // nothing coarsens
      inner = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    if (mgDebugLevel()) {
      printf("[mg] init %dx%dx%d single-rank -> %d levels (requested %d)\n", nx, ny, nz,
             (int)lv_.size(), nLevels);
      for (int L = 0; L < (int)lv_.size(); ++L)
        printf("[mg]  L%d dims %4dx%4dx%4d  ratio(%d,%d,%d)\n", L, lv_[L].inner.x, lv_[L].inner.y,
               lv_[L].inner.z, lv_[L].ratio.x, lv_[L].ratio.y, lv_[L].ratio.z);
      fflush(stdout);
    }
  }
#ifdef PECLET_FLOW_MPI
  // Multi-rank hierarchy: coarsen the GLOBAL grid 2:1 per level; each level gets its own core halo
  // over a BlockDecomposer of that level's grid (the ORB decomposition coarsens cleanly so
  // restrict/prolong stay local). Sets the distributed flag -> fill() exchanges, the reductions
  // Allreduce, the smoother uses the block's global-origin parity. Single-rank (size 1) reproduces
  // init()'s field exactly.
  // dec0: OPTIONAL shared level-0 decomposition (load-balance / CFD-DEM co-decomposition). When
  // given, level 0 uses it so the MG's level-0 block matches the caller's (possibly weighted)
  // block; the coarse levels keep the equal-weight ORB of the coarsened grid. For a weighted dec0
  // the coarse-level transfer is only clean when nLevels==1 (pure RB-GS) — use that (or the
  // decomposition-agnostic GraphAMG) for a weighted co-decomposition. nullptr => equal-weight
  // everywhere (the original behaviour, byte-identical).
  // Per-axis split alignment that makes an ORB safely coarsenable by this MG: align[k] =
  // 2^(number of times axis k can coarsen, until it turns odd) — the NATURAL MAXIMUM, independent of
  // the actual nLevels (over-aligning is harmless: coarsened() still divides cleanly at every real
  // level). Depends only on the global grid, so the solver's dec_, the mpi_block() sizing, and this
  // MG all compute the SAME value without threading nLevels. The solver builds its shared
  // decomposition with this alignment so initMpi derives nested coarse levels via coarsened().
  static peclet::core::IVec<3> coarsenAlignment(int gnx, int gny, int gnz) {
    auto can = [](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    C3 gs{gnx, gny, gnz};
    peclet::core::IVec<3> a{1, 1, 1};
    for (bool any = true; any;) {
      any = false;
      if (can(gs.x)) { a[0] *= 2; gs.x /= 2; any = true; }
      if (can(gs.y)) { a[1] *= 2; gs.y /= 2; any = true; }
      if (can(gs.z)) { a[2] *= 2; gs.z /= 2; any = true; }
    }
    // Cap at 2^(default nLevels - 1): all the 5-level hierarchy needs. The UNCAPPED natural-max
    // over-constrains the ORB on power-of-two-rich grids (e.g. 192^3 -> align 64): the split snap
    // then rounds a balanced 96|96 to 128|64 (cascading 2:1 load imbalance), and once sub-boxes
    // drop under 2*align the snap is skipped -> unaligned splits -> the even-coarsening gate
    // collapses the MG depth (measured: 192^3 np=24 pure-MPI, 27 pressure iters/step vs 9, 3.4x
    // step time). With the cap the same case decomposes perfectly evenly and keeps 5 nested
    // levels; axes whose natural alignment is smaller are unchanged, deeper hierarchies degrade
    // through the existing evenBlocks gate exactly as before.
    for (int k = 0; k < 3; ++k)
      if (a[k] > 16)
        a[k] = 16;
    return a;
  }

  // ---- coarse-first ("decompose coarse, refine upward") decomposition ---------------------------
  // Requested hierarchy depth for the LEVEL-0 DECOMPOSITION: 0 (default) = the legacy aligned-ORB
  // route above; L >= 2 = build the ORB on the grid coarsened L-1 times and refine the partition
  // upward, which guarantees L nested levels and balances on the coarse grid instead of snapping
  // fine splits afterwards. Read once from PECLET_FLOW_DECOMP_LEVELS, overridable programmatically.
  // MUST be set before the decomposition is built (i.e. before mpi_block()/init_mpi), because all
  // three call sites — mpi_block(), IbmSolver::initMpi and this class — derive the SAME partition
  // from it and would otherwise disagree about the block layout.
  static int& decompositionLevelsRef() {
    static int v = [] {
      const char* e = std::getenv("PECLET_FLOW_DECOMP_LEVELS");
      return e ? std::atoi(e) : 0;
    }();
    return v;
  }
  static int decompositionLevels() { return decompositionLevelsRef(); }
  static void setDecompositionLevels(int levels) { decompositionLevelsRef() = levels; }

  // Per-axis coarsening factor a depth-`levels` hierarchy will actually apply: 2^(levels-1), bounded
  // by that axis's factors of two (an odd axis never coarsens, so its factor stays 1).
  static peclet::core::IVec<3> refineFactor(int gnx, int gny, int gnz, int levels) {
    auto can = [](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    C3 gs{gnx, gny, gnz};
    peclet::core::IVec<3> r{1, 1, 1};
    for (int L = 1; L < levels; ++L) {
      if (can(gs.x)) { r[0] *= 2; gs.x /= 2; }
      if (can(gs.y)) { r[1] *= 2; gs.y /= 2; }
      if (can(gs.z)) { r[2] *= 2; gs.z /= 2; }
    }
    return r;
  }

  // THE shared level-0 decomposition. Every call site must go through this so the solver's block,
  // mpi_block()'s sizing and the MG's level 0 cannot drift apart.
  static peclet::core::decomp::BlockDecomposer<3> decomposition(std::size_t numBlocks, int gnx,
                                                                int gny, int gnz) {
    const int levels = decompositionLevels();
    if (levels < 2)
      return peclet::core::decomp::BlockDecomposer<3>(numBlocks, peclet::core::IVec<3>{gnx, gny, gnz},
                                                      coarsenAlignment(gnx, gny, gnz));
    // Depth and load balance pull against each other: each extra level doubles the quantum on every
    // axis that still coarsens, and a partition built on the coarse grid can only place a split on a
    // coarse-cell boundary. So rather than guess a granularity, BUILD each candidate and measure its
    // imbalance: take the deepest one that stays within budget, else keep the legacy aligned ORB.
    // The whole search is a pure function of (numBlocks, grid, levels) — every rank computes the
    // same answer without communicating.
    const double maxImbalance = [] {
      const char* e = std::getenv("PECLET_FLOW_DECOMP_MAX_IMBALANCE");
      const double v = e ? std::atof(e) : 1.05;
      return v > 1.0 ? v : 1.05;
    }();
    auto imbalanceOf = [](const peclet::core::decomp::BlockDecomposer<3>& d) {
      std::size_t hi = 0, lo = std::numeric_limits<std::size_t>::max();
      for (const auto& s : d.sizes()) {
        const std::size_t n = static_cast<std::size_t>(s[0]) * static_cast<std::size_t>(s[1]) *
                              static_cast<std::size_t>(s[2]);
        hi = n > hi ? n : hi;
        lo = n < lo ? n : lo;
      }
      return lo ? static_cast<double>(hi) / static_cast<double>(lo)
                : std::numeric_limits<double>::infinity();
    };
    for (int L = levels; L >= 2; --L) {
      const peclet::core::IVec<3> r = refineFactor(gnx, gny, gnz, L);
      if (r[0] == 1 && r[1] == 1 && r[2] == 1)
        break;  // nothing coarsens on any axis — the aligned ORB is all there is
      const std::size_t cells = static_cast<std::size_t>(gnx / r[0]) *
                                static_cast<std::size_t>(gny / r[1]) *
                                static_cast<std::size_t>(gnz / r[2]);
      if (cells < numBlocks)
        continue;  // a coarse grid thinner than the rank count would hand someone an empty block
      // Decompose the COARSE grid — telling the ORB each coarse cell's true extent, so it picks the
      // same split axes the fine grid would — then refine the partition upward. Blocks come out as
      // exact multiples of r, so every level nests for the full depth.
      peclet::core::decomp::BlockDecomposer<3> coarse;
      coarse.init(numBlocks, peclet::core::IVec<3>{gnx / r[0], gny / r[1], gnz / r[2]},
                  peclet::core::IVec<3>{1, 1, 1}, r);
      peclet::core::decomp::BlockDecomposer<3> fine = coarse.refined(r);
      if (imbalanceOf(fine) <= maxImbalance) {
        if (mgDebugLevel())
          printf("[mg] decomposition: coarse-first depth %d (refine %dx%dx%d, imbalance %.3f)\n", L,
                 r[0], r[1], r[2], imbalanceOf(fine));
        return fine;
      }
    }
    return peclet::core::decomp::BlockDecomposer<3>(numBlocks, peclet::core::IVec<3>{gnx, gny, gnz},
                                                    coarsenAlignment(gnx, gny, gnz));
  }

  void initMpi(int gnx, int gny, int gnz, int nLevels, MPI_Comm comm,
               const peclet::core::decomp::BlockDecomposer<3>* dec0 = nullptr) {
    lv_.clear();
    amg_.reset();
    distributed_ = true;
    comm_ = comm;
    gnxF_ = gnx;
    gnyF_ = gny;
    gnzF_ = gnz;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::array<bool, 3> per{true, true, true};
    C3 gs{gnx, gny, gnz}, cf{1, 1, 1};
    auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    // Coarse levels are NESTED: level 0 is the shared solver decomposition (dec0), and each coarse
    // level is the previous level's decomposition coarsened IN PLACE (same tree/leaf order, split
    // positions halved). This keeps restrict/prolong's coarse-local i <-> fine-local ratio*i mapping
    // valid on every rank. (An independent ORB per level does NOT nest — coarse blocks can split a
    // different axis than the fine level, sending restrict/prolong out of bounds.)
    peclet::core::decomp::BlockDecomposer<3> curDec;
    if (dec0) {
      curDec = *dec0;  // solver's shared decomposition (built aligned; see flow_ibm initMpi)
    } else {
      curDec = decomposition(static_cast<std::size_t>(size), gs.x, gs.y, gs.z);
    }
    // Smallest block extent (any rank, any axis) — the same on every rank (the decomposition is
    // replicated), so the per-level ghost-width decision below is rank-uniform by construction.
    auto minBlockExtent = [](const peclet::core::decomp::BlockDecomposer<3>& d) {
      long m = std::numeric_limits<long>::max();
      for (const auto& s : d.sizes())
        for (int k = 0; k < 3; ++k)
          m = std::min(m, (long)s[k]);
      return m;
    };
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.halo = std::make_shared<GridHaloTopology<3>>();
      const peclet::core::decomp::BlockDecomposer<3>& dec = curDec;
      // Communication-avoiding smoothing needs a 2-deep ghost layer; give it to the COARSE levels
      // (where every halo message is pure latency) whose blocks can carry it (extent >= 4 on every
      // rank; below that fall back to the per-colour exchange). Level 0 keeps g=1: its exchanges
      // are already overlapped with the interior sweep and the solver's g=1 bridge (openness/rhs/
      // phi staging, ghost-projection g=2 staging) assumes it. Only for the periodic/IBM operator
      // — with domain BCs (setBoundaryConditions BEFORE initMpi) every level keeps the g=1 layout,
      // so that path is byte-identical to the pre-CA code.
      v.g = (L > 0 && !hasBC_ && (caSmoothingMode() & kCaMg) && minBlockExtent(dec) >= 4) ? 2 : 1;
      v.caOk = (v.g == 2);
      v.halo->buildTopology(dec, rank, v.g, per, comm);
      v.dev = std::make_shared<GridHalo<double>>();
      v.dev->init(*v.halo);
      const auto& idx = v.halo->indexer();
      const auto eg = idx.sizeInclGhost(), ino = idx.sizeInner(), oig = idx.originInclGhost();
      v.ext = {(int)eg[0], (int)eg[1], (int)eg[2]};
      v.inner = {(int)ino[0], (int)ino[1], (int)ino[2]};
      v.og = {(int)oig[0] + v.g, (int)oig[1] + v.g,
              (int)oig[2] + v.g};  // inner origin == single-rank og=0 at origin 0
      v.gdim = gs;                 // this level's GLOBAL dims (og + inner == gdim -> owns +face)
      v.cfac = cf;
      v.n = idx.numCellsInclGhost();
      C3 next = gs, ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        // Coarsen an axis only if the GLOBAL dim can (can()) AND every rank's block is even on that
        // axis, so coarsened() nests exactly. Alignment (coarsenAlignment) makes this hold for the
        // full natural depth; on an unaligned/awkward decomposition it simply stops coarsening that
        // axis early (fewer geometric levels, GraphAMG bottom picks up the rest) — never OOB.
        auto evenBlocks = [&](int ax) {
          for (std::size_t b = 0; b < curDec.sizes().size(); ++b)
            if ((curDec.origins()[b][ax] % 2) || (curDec.sizes()[b][ax] % 2))
              return false;
          return true;
        };
        if (can(gs.x) && evenBlocks(0)) {
          ratio.x = 2;
          next.x = gs.x / 2;
        }
        if (can(gs.y) && evenBlocks(1)) {
          ratio.y = 2;
          next.y = gs.y / 2;
        }
        if (can(gs.z) && evenBlocks(2)) {
          ratio.z = 2;
          next.z = gs.z / 2;
        }
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n);
      v.rhs = CCField("mg_rhs", v.n);
      v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n);
      v.oy = CCField("mg_oy", v.n);
      v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT})
        *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z)
        break;
      gs = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
      // Next level's decomposition = this level's coarsened in place (nested; preserves rank order).
      curDec = curDec.coarsened(peclet::core::IVec<3>{ratio.x, ratio.y, ratio.z});
    }
    if (mgDebugLevel() && rank == 0) {
      printf("[mg] initMpi %dx%dx%d np=%d -> %d levels (requested %d)\n", gnx, gny, gnz, size,
             (int)lv_.size(), nLevels);
      C3 g{gnx, gny, gnz};
      for (int L = 0; L < (int)lv_.size(); ++L) {
        printf("[mg]  L%d global %4dx%4dx%4d  rank0 block %4dx%4dx%4d  ratio(%d,%d,%d)\n", L, g.x,
               g.y, g.z, lv_[L].inner.x, lv_[L].inner.y, lv_[L].inner.z, lv_[L].ratio.x,
               lv_[L].ratio.y, lv_[L].ratio.z);
        g = C3{g.x / lv_[L].ratio.x, g.y / lv_[L].ratio.y, g.z / lv_[L].ratio.z};
      }
      fflush(stdout);
    }
  }
#endif
  int nLevels() const { return (int)lv_.size(); }
  Level& level(int L) { return lv_[L]; }

  // per-face domain BC types {-x,+x,-y,+y,-z,+z}: 0=periodic, 1/2=Neumann (wall/inflow),
  // 3=Dirichlet (outflow). Default all-periodic -> applyBoundaryOpenness is a no-op (periodic/IBM
  // path byte-identical).
  void setBoundaryConditions(const int bc[6]) {
    hasBC_ = false;
    hasOutflow_ = false;
    for (int i = 0; i < 6; ++i) {
      bc_[i] = bc[i];
      if (bc[i])
        hasBC_ = true;
      if (bc[i] == 3)
        hasOutflow_ = true;
    }
    removeMean_ =
        !hasOutflow_;  // singular all-Neumann -> remove mean; Dirichlet outflow -> non-singular
  }
  // hold the pressure/correction ghost at 0 on outflow faces (open face -> Dirichlet p=0). Call
  // after every (periodic) fill of a solution / search-direction field, on the level it lives
  // (g = that level's ghost width).
  void applyOutflowGhost(const Level& lv, CCField x, int g = G) {
    if (!hasOutflow_)
      return;
    B3 e{lv.ext.x, lv.ext.y, lv.ext.z};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s)
        if (bc_[2 * a + s] == 3 && touchesGlobalFace(lv, 2 * a + s))
          bcZeroPressureGhost(x, e, g, a, s);
  }
  // Does this rank's block on level `lv` touch global domain face f? Always true single-rank
  // (og = 0 and gdim == inner), so every guarded BC application is byte-identical there.
  static bool touchesGlobalFace(const Level& lv, int f) {
    const int a = f / 2;
    const int o = (a == 0) ? lv.og.x : (a == 1) ? lv.og.y : lv.og.z;
    const int n = (a == 0) ? lv.inner.x : (a == 1) ? lv.inner.y : lv.inner.z;
    const int gd = (a == 0) ? lv.gdim.x : (a == 1) ? lv.gdim.y : lv.gdim.z;
    return (f % 2 == 0) ? (o == 0) : (o + n == gd);
  }
  // re-impose the non-periodic boundary openness a periodic fill leaves wrong: Neumann wall/inflow
  // -> 0 (closed), Dirichlet outflow -> left open. Call after every (periodic) openness fill, per
  // level.
  void applyBoundaryOpenness(Level& lv) {
    if (!hasBC_)
      return;
    B3 e{lv.ext.x, lv.ext.y, lv.ext.z};
    CCField oa[3] = {lv.ox, lv.oy, lv.oz};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (!touchesGlobalFace(lv, 2 * a + s))
          continue;  // interior rank boundary: the exchanged openness is the right value
        if (t == 1 || t == 2)
          bcSetOpenness(oa[a], e, lv.g, a, s, 0.0);  // wall/inflow Neumann -> closed
        else if (t == 3)
          bcSetOpenness(oa[a], e, lv.g, a, s, 1.0);  // outflow -> open (periodic fill wraps wrong)
      }
  }

  // rediscretized cut-cell operator on every level from the fine face openness (idx2 = 1/dx^2
  // fine).
  void setOpenness(CCConst ox, CCConst oy, CCConst oz, double idx2, double idy2, double idz2) {
    Level& f = lv_[0];
    Kokkos::deep_copy(f.ox, ox);
    Kokkos::deep_copy(f.oy, oy);
    Kokkos::deep_copy(f.oz, oz);
    fillOpenness(
        f);  // periodic fine-level openness ghosts (the operator reads the + neighbour face);
             // idempotent when the caller already filled them, required when it passed inner-only.
    applyBoundaryOpenness(
        f);  // re-impose non-periodic wall/inflow faces the periodic fill clobbered
    buildCutcellOp(f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT, CCConst(f.ox), CCConst(f.oy),
                   CCConst(f.oz), f.ext, G, idx2, idy2, idz2);
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      Level& fin = lv_[L - 1];
      coarsenOpenAvg(c.ox, c.oy, c.oz, CCConst(fin.ox), CCConst(fin.oy), CCConst(fin.oz), c.ext,
                     fin.ext, c.g, fin.g, c.inner, fin.ratio);
      fillOpenness(c);  // periodic ghost openness (operator build reads the + neighbour face)
      applyBoundaryOpenness(c);  // re-impose non-periodic boundary faces per coarse level
      const double sx = 1.0 / (double)(c.cfac.x * c.cfac.x),
                   sy = 1.0 / (double)(c.cfac.y * c.cfac.y),
                   sz = 1.0 / (double)(c.cfac.z * c.cfac.z);
      // Width-2 (CA-eligible) levels also assemble the 1-deep ghost RING of the operator (build
      // box widened by 1): the ring rows are a deterministic function of the EXCHANGED openness,
      // so they come out bit-identical to the owning rank's inner rows — the redundant ring
      // re-smoothing of the CA sweep reads them. Inner rows are computed from the same operands
      // as the g-box build (identical). g=1 levels keep the inner-only build.
      buildCutcellOp(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.ox), CCConst(c.oy),
                     CCConst(c.oz), c.ext, c.g == 2 ? c.g - 1 : c.g, idx2 * sx, idy2 * sy,
                     idz2 * sz);
    }
    // The operator (all levels, including the bottom) just changed: invalidate the agglomerated
    // GraphAMG bottom solve so the next solve rebuilds it from the CURRENT coefficients. The porous
    // and variable-rho paths rebuild the coefficients EVERY STEP — with a stale AMG bottom (frozen
    // at the first step's operator) the bottom "solve" answers a different matrix, the V-cycle
    // preconditioner drifts inconsistent/indefinite, and the outer PCG eventually breaks down and
    // NaNs the projection (observed as a sporadic, data-dependent blow-up in porous CFD-DEM). The
    // bottom level is tiny, so the per-step rebuild is negligible next to the V-cycles.
    amg_.reset();
    amgGlobalN_ = 0;
  }

  // CG preconditioned by one symmetric V-cycle (solve_pcg port). rhs on level 0; solution left in
  // level-0 x. Returns the iteration count. Scratch supplied by the caller (level-0-sized fields).
  // Optional star overlay (mode-B fluid-only constraint): the SPD Kron-elimination couplings are
  // added to the fine-level matvec only; the hierarchy/preconditioner sees the filtered 7-point
  // surrogate it was built from (the symmetric sibling of solveBiCGStab's gp overlay pattern).
  // Single-rank v1: the star kernels wrap periodically over the inner grid, so no halo work.
  int solvePCG(CCField b, CCField x, CCField r, CCField p, CCField z, CCField Ap, int maxit,
               double rtol, int pre, int post, int bottom, const StarOverlay* star = nullptr,
               int nStar = 0, C3 nnStar = C3{0, 0, 0}) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(l0.x, x);
    auto matvec = [&](CCField y, CCField v) {
      matvecOverlap(l0, y, v);
      if (star)
        starApplyDelta(y, CCConst(v), *star, nStar, nnStar, l0.ext, G, l0.ext, G);
    };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(l0.rhs, rr);
      Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(zz, l0.x);
    };
    matvec(Ap, x);  // r = b - A x
    Kokkos::deep_copy(r, b);
    axpy(r, -1.0, Ap);
    removeMean(l0, r);  // compatibility: project rhs/residual onto the range
    const double r0 = maxabs(l0, r);
    int it = 0;
    // Env-gated convergence trace (PECLET_FLOW_MG_DEBUG>=2): |b|inf, r0 and the per-iteration
    // residual, so a decomposition-dependent iteration count can be read as a rate (preconditioner
    // quality) or a floor (round-off) instead of guessed at.
    int dbgRank = 0;
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      MPI_Comm_rank(comm_, &dbgRank);
#endif
    const bool trace = mgDebugLevel() >= 2 && dbgSolve_ < mgDebugSolves() && dbgRank == 0;
    if (trace)
      printf("[mg] solve %d: r0=%.6e rtol=%.1e (pre=%d post=%d bottom=%d)\n", dbgSolve_, r0, rtol,
             pre, post, bottom);
    ++dbgSolve_;
    // Breakdown guards: a non-finite recurrence scalar means the preconditioner or operator
    // produced NaN/Inf (should not happen — the guards fail safe rather than poisoning x with a
    // NaN alpha/beta and letting the projection silently corrupt every field downstream).
    if (r0 > 0.0 && std::isfinite(r0)) {
      precond(z, r);
      Kokkos::deep_copy(p, z);
      double rz = dot(l0, r, z);
      if (!std::isfinite(rz)) {
        printf("peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z; "
               "returning zero correction\n");
        Kokkos::deep_copy(x, 0.0);
        Kokkos::deep_copy(l0.x, x);
        return 0;
      }
      for (; it < maxit; ++it) {
        matvec(Ap, p);
        if (meanRemovalAll_)
          removeMean(l0, Ap);  // A preserves mean-freeness; "fine" scope trusts that
        const double pAp = dot(l0, p, Ap);
        if (!std::isfinite(pAp) || pAp <= 1e-300)
          break;  // breakdown/converged direction: keep the last finite iterate
        const double alpha = rz / pAp;
        axpy(x, alpha, p);
        axpy(r, -alpha, Ap);
        removeMean(l0, r);
        const double rn = maxabs(l0, r);
        if (trace)
          printf("[mg]   it %3d  |r|inf=%.6e  r/r0=%.4e\n", it + 1, rn, rn / r0);
        if (rn < rtol * r0) {
          ++it;
          break;
        }
        precond(z, r);
        const double rznew = dot(l0, r, z), beta = rznew / rz;
        if (!std::isfinite(rznew))
          break;  // preconditioner breakdown: keep the last finite iterate
        aypx(p, beta, z);
        rz = rznew;
      }
    }
    Kokkos::deep_copy(l0.x, x);
    removeMean(l0, l0.x);
    Kokkos::deep_copy(x, l0.x);
    return it;
  }

  // BiCGStab preconditioned by one symmetric V-cycle, for the NONSYMMETRIC ghost-projection
  // operator A = (binary-openness 7-point op) + (per-row overlay delta, gpApplyDelta). The MG
  // hierarchy/preconditioner only ever sees the symmetric binary surrogate its levels were built
  // from (setOpenness); the overlay enters the fine-level matvec only. Same breakdown guards +
  // constant-mode (mean) removal as solvePCG, plus a stagnation guard: the nonsymmetric system's
  // left null vector is NOT exactly the constants, so the attainable residual has a small
  // compatibility floor — stop when no progress instead of burning maxit. Scratch: 7 level-0
  // fields from the caller. Returns the iteration count.
  // Distributed: the reductions/mean removal already Allreduce and the V-cycle is MPI-folded; the
  // fine-level matvec is the one gp-specific piece. The overlay couplings reach +/-2 but the MG
  // block only has a g=1 halo, so the caller passes a g=2 staging field (xg2, on its ext2 block)
  // + that block's halo: stage q's inner cells there, exchange the 2-deep halo once, read the g=1
  // halo back from the staged copy (one exchange serves both the 7-point op and the overlay), and
  // apply the overlay in ghost mode. Single-rank (h2 == nullptr) is byte-identical to before.
  int solveBiCGStab(CCField b, CCField x, CCField r, CCField rh, CCField p, CCField v, CCField t,
                    CCField z, CCField z2, int maxit, double rtol, int pre, int post, int bottom,
                    const GpOverlay& ov, int nOv, C3 nn
#ifdef PECLET_FLOW_MPI
                    ,
                    CCField xg2 = CCField(), GridHalo<double>* h2 = nullptr, C3 ext2 = C3{0, 0, 0}
#endif
  ) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    auto matvec = [&](CCField y, CCField q) {
#ifdef PECLET_FLOW_MPI
      if (distributed_ && h2) {
        stageG2(l0, q, xg2, ext2);  // inner cells g=1 block -> g=2 block
        h2->exchange(xg2);          // 2-deep halo (cross-rank + periodic)
        unstageG2(l0, q, xg2);      // whole l0 block back (fills q's g=1 halo — no 2nd exchange)
        applyOutflowGhost(l0, q);
        applyCutcellOp(y, CCConst(q), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN),
                       FPC(l0.AB), FPC(l0.AT), l0.ext, G);
        gpApplyDelta(y, CCConst(xg2), ov, nOv, nn, l0.ext, G, ext2, 2, /*useGhost=*/true);
        return;
      }
#endif
      fill(l0, q);
      applyOutflowGhost(l0, q);
      applyCutcellOp(y, CCConst(q), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN),
                     FPC(l0.AB), FPC(l0.AT), l0.ext, G);
      gpApplyDelta(y, CCConst(q), ov, nOv, nn, l0.ext, G, l0.ext, G);
    };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(l0.rhs, rr);
      Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(zz, l0.x);
    };
    matvec(t, x);  // r = b - A x  (t as scratch)
    Kokkos::deep_copy(r, b);
    axpy(r, -1.0, t);
    removeMean(l0, r);
    Kokkos::deep_copy(rh, r);  // shadow residual r^ = r_0
    const double r0n = maxabs(l0, r);
    int it = 0;
    if (r0n > 0.0 && std::isfinite(r0n)) {
      double rho = 1.0, alpha = 1.0, omega = 1.0;
      double best = r0n;
      int lastImprove = 0;
      Kokkos::deep_copy(p, 0.0);
      Kokkos::deep_copy(v, 0.0);
      for (; it < maxit; ++it) {
        const double rhoNew = dot(l0, rh, r);
        if (!std::isfinite(rhoNew) || std::fabs(rhoNew) < 1e-300)
          break;  // (rh, r) breakdown: keep the last finite iterate
        const double beta = (rhoNew / rho) * (alpha / omega);
        rho = rhoNew;
        axpy(p, -omega, v);  // p = r + beta (p - omega v)
        aypx(p, beta, r);
        precond(z, p);
        matvec(v, z);
        removeMean(l0, v);
        const double rhv = dot(l0, rh, v);
        if (!std::isfinite(rhv) || std::fabs(rhv) < 1e-300)
          break;
        alpha = rho / rhv;
        axpy(r, -alpha, v);  // r <- s = r - alpha v
        removeMean(l0, r);
        double rn = maxabs(l0, r);
        if (!std::isfinite(rn))
          break;
        if (rn < rtol * r0n) {
          axpy(x, alpha, z);
          ++it;
          break;
        }
        precond(z2, r);
        matvec(t, z2);
        removeMean(l0, t);
        const double tt = dot(l0, t, t);
        if (!std::isfinite(tt) || tt < 1e-300) {
          axpy(x, alpha, z);  // omega breakdown: take the alpha half-step and stop
          break;
        }
        omega = dot(l0, t, r) / tt;
        if (!std::isfinite(omega) || std::fabs(omega) < 1e-300) {
          axpy(x, alpha, z);
          break;
        }
        axpy(x, alpha, z);
        axpy(x, omega, z2);
        axpy(r, -omega, t);
        removeMean(l0, r);
        rn = maxabs(l0, r);
        if (!std::isfinite(rn))
          break;
        if (rn < rtol * r0n) {
          ++it;
          break;
        }
        if (rn < 0.999 * best) {
          best = rn;
          lastImprove = it;
        } else if (it - lastImprove > 30) {
          ++it;  // compatibility-floor stagnation: accept the best-so-far level
          break;
        }
      }
    }
    Kokkos::deep_copy(l0.x, x);
    removeMean(l0, l0.x);
    Kokkos::deep_copy(x, l0.x);
    return it;
  }

 public:  // (public for nvcc extended-lambda rule)
  // Per-level V-cycle wall time (PECLET_FLOW_MG_DEBUG>=3; HOST backends only — no device fence, so
  // on CUDA the numbers are launch times, not kernel times). Answers "how much of the solve is
  // spent on the small coarse levels", i.e. whether coarse-level launch overhead is worth chasing.
  void vcycle(int L, bool sym) {
    if (mgDebugLevel() >= 3) {
      const auto t0 = std::chrono::steady_clock::now();
      vcycleImpl(L, sym);
      lvTime_.resize(lv_.size(), 0.0);
      lvTime_[L] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (L == 0 && ++lvCycles_ % 50 == 0) {
        double tot = 0;
        for (std::size_t i = 0; i < lvTime_.size(); ++i)
          tot += (i + 1 < lvTime_.size() ? lvTime_[i] - lvTime_[i + 1] : lvTime_[i]);
        printf("[mg] level times over %d V-cycles (total %.3f s):\n", lvCycles_, tot);
        for (std::size_t i = 0; i < lvTime_.size(); ++i) {
          const double self = (i + 1 < lvTime_.size() ? lvTime_[i] - lvTime_[i + 1] : lvTime_[i]);
          printf("[mg]   L%zu %5dx%5dx%5d  self %7.3f s (%5.1f%%)\n", i, lv_[i].inner.x,
                 lv_[i].inner.y, lv_[i].inner.z, self, 100.0 * self / (tot + 1e-30));
        }
        fflush(stdout);
      }
      return;
    }
    vcycleImpl(L, sym);
  }
  void vcycleImpl(int L, bool sym) {
    Level& lv = lv_[L];
    if (L + 1 == (int)lv_.size()) {
      if (agglomerateBottom())
        graphAmgSolveBottom(
            lv);  // agglomerated mesh-agnostic coarse solve (decomposition-agnostic)
      else
        smooth(lv, bottom_, false);
      if (meanRemovalAll_)
        removeMean(lv, lv.x);
      return;
    }
    smooth(lv, pre_, false);
    // Refresh the halo before the residual. The smoother exchanges BEFORE each color sweep, so on
    // return the ghosts are one color-update stale: the residual — and hence the restricted coarse
    // rhs — is wrong on the block-boundary shell. That perturbation is proportional to the block
    // SURFACE, so it made the V-cycle's convergence rate decomposition-dependent: fat blocks (few
    // ranks) needed measurably more Krylov iterations than thin ones on the SAME grid (768x640x384
    // genoa: 12 iters at 12-24 ranks vs 8.1 at 96; 256^3 workstation: 8/6.5/9 at np=1/8/24 -> a
    // flat 4.0 with the refresh). Distributed: overlap the exchange with the interior residual
    // (same interior/shell split as the smoother); single-rank: the periodic wrap copy.
    auto fullResidual = [&] {
      residualCutcell(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                      FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, lv.g);
    };
    if (!resFill_) {  // PECLET_FLOW_MG_RESFILL=0: the legacy stale-ghost residual (ablation only)
      fullResidual();
    }
#ifdef PECLET_FLOW_MPI
    else if (distributed_) {
      const int g = lv.g;
      const C3 lo{g + 1, g + 1, g + 1};
      const C3 hi{lv.ext.x - g - 1, lv.ext.y - g - 1, lv.ext.z - g - 1};
      lv.dev->exchangeBegin(lv.x);
      residualCutcellBox(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW),
                         FPC(lv.AE), FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, lo, hi,
                         C3{0, 0, 0}, C3{0, 0, 0});
      lv.dev->exchangeEnd(lv.x);
      applyOutflowGhost(lv, lv.x, g);
      residualCutcellBox(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW),
                         FPC(lv.AE), FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext,
                         C3{g, g, g}, C3{lv.ext.x - g, lv.ext.y - g, lv.ext.z - g}, lo, hi);
    } else
#endif
    {
      fill(lv, lv.x);  // single-rank: the periodic wrap copy
      applyOutflowGhost(lv, lv.x, lv.g);
      fullResidual();
    }
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, cs.g, lv.g, cs.inner, lv.ratio);
    Kokkos::deep_copy(cs.x, 0.0);
    vcycle(L + 1, sym);
    fill(cs, cs.x);
    applyOutflowGhost(cs, cs.x, cs.g);
    prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, lv.g, cs.g, lv.inner, lv.ratio);
    smooth(lv, post_, /*reverse=*/sym);
    if (meanRemovalAll_ || L == 0)
      removeMean(lv, lv.x);
  }
  // Communication-avoiding smoothing on this level? Needs the width-2 topology (caOk), and the
  // periodic/IBM operator — with domain BCs the ring rows would need post-BC ghost openness the
  // exchange does not deliver, so those keep the per-colour exchange.
  bool caSmooth(const Level& lv) const { return distributed_ && lv.caOk && !hasBC_; }
  void smooth(Level& lv, int sweeps, bool reverse) {
    const C3 og = parityOg(lv);  // red-black parity origin ({0,0,0} single-rank)
#ifdef PECLET_FLOW_MPI
    if (caSmooth(lv)) {
      // Communication-avoiding pair: ONE 2-deep exchange per red-black pair instead of a 1-deep
      // exchange per colour. The first colour overlaps its exchange with the interior sweep, then
      // sweeps the boundary shell PLUS the 1-deep ghost ring — redundantly recomputing the
      // neighbour's boundary cells from the same operands the neighbour uses (2-deep x ghosts,
      // ring rows of the operator and rhs are exchanged/assembled bit-identical), so the ring
      // values come out equal to what a fresh exchange would deliver. The second colour then
      // sweeps with NO exchange: its boundary cells read only first-colour ring cells (a colour
      // never reads its own colour). Bit-identical to the per-colour exchange at half the events.
      const int g = lv.g;
      const C3 lo{g + 1, g + 1, g + 1};
      const C3 hi{lv.ext.x - g - 1, lv.ext.y - g - 1, lv.ext.z - g - 1};
      const C3 rlo{g - 1, g - 1, g - 1};
      const C3 rhi{lv.ext.x - g + 1, lv.ext.y - g + 1, lv.ext.z - g + 1};
      lv.dev->exchange(lv.rhs);  // ring rhs (owner's inner values); rhs is fixed over the sweeps
      for (int k = 0; k < sweeps; ++k) {
        const int c0 = reverse ? 1 : 0, c1 = 1 - c0;
        lv.dev->exchangeBegin(lv.x);
        cutcellSmoothColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                              FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, c0, lo,
                              hi, C3{0, 0, 0}, C3{0, 0, 0});
        lv.dev->exchangeEnd(lv.x);
        cutcellSmoothColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                              FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, c0, rlo,
                              rhi, lo, hi);
        cutcellSmoothColor(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                           FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, g, c1);
      }
      return;
    }
#endif
    for (int k = 0; k < sweeps; ++k)
      for (int s = 0; s < 2; ++s) {
        const int color = reverse ? (1 - s) : s;
#ifdef PECLET_FLOW_MPI
        if (distributed_) {
          // Overlap the per-color halo with the interior sweep: post the exchange, smooth the
          // cells whose 7-point stencil reads no ghost (they depend on neither the incoming halo
          // nor the outflow ghost), complete the exchange, then sweep the boundary shell. A
          // color's cells never read same-color cells, so this ordering is bit-identical to the
          // blocking fill-then-full-sweep (validated by the np>1 bit-exact MG tests).
          const int g = lv.g;
          const C3 lo{g + 1, g + 1, g + 1};
          const C3 hi{lv.ext.x - g - 1, lv.ext.y - g - 1, lv.ext.z - g - 1};
          lv.dev->exchangeBegin(lv.x);
          cutcellSmoothColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                                FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, color,
                                lo, hi, C3{0, 0, 0}, C3{0, 0, 0});
          lv.dev->exchangeEnd(lv.x);
          applyOutflowGhost(lv, lv.x, g);
          cutcellSmoothColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                                FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, color,
                                C3{g, g, g}, C3{lv.ext.x - g, lv.ext.y - g, lv.ext.z - g}, lo, hi);
          continue;
        }
#endif
        fill(lv, lv.x);
        applyOutflowGhost(lv, lv.x, lv.g);
        cutcellSmoothColor(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                           FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, og, lv.g, color);
      }
  }

  // --- when to agglomerate ------------------------------------------------------------------------
  // A V-cycle only converges at a rate independent of the domain if its COARSEST level is small
  // enough to be solved (essentially) exactly by the few smoother sweeps applied there. A geometric
  // hierarchy cannot always get there: an axis stops coarsening once it turns odd, and under MPI it
  // stops once any rank's block turns odd — so on a fixed per-rank block the coarsest GLOBAL grid
  // grows with the rank count and the bottom is progressively under-solved. That is the mechanism
  // behind weak-scaling curves that decay while communication stays negligible.
  //
  // Measured (single GPU, channel, Lx = 2048 x 64 x 64, everything else held): a smoothed bottom
  // needs 13.5 pressure iterations/step at 4 levels and 6.0 at 6 levels, against 4.4 at full
  // geometric depth. Agglomerating and solving that same bottom exactly gives 4.0 at BOTH 4 and 6
  // levels — depth-independent, and faster in wall-clock than the full-depth hierarchy (69.5 vs
  // 77.1 ms/step) because the extra levels cost more than the coarse solve they replace.
  //
  // NOT the default yet, and the reason is measured: on the cut-cell sphere-packing regression
  // (random_spheres, N=48) switching the bottom to the agglomerated solve makes the OUTER iteration
  // count WORSE (442 -> 622 total, +41 %) at unchanged accuracy, so the assembled coarse operator is
  // evidently not consistent with the V-cycle's on that IBM path. Until that is understood, `auto`
  // is opt-in and the legacy smoothed bottom stays the default.
  // `mode`: 0 = never / plain smoothed bottom (DEFAULT), -1 = auto, 1 = always.
  // PECLET_FLOW_AGGLOM_CELLS overrides the threshold; the ideal bottom is a handful of cells per
  // axis, and 512 is a generous cut that leaves genuinely small bottoms on the cheap path.
  bool agglomerateBottom() const {
    if (agglomMode_ == 0)
      return false;
    if (agglomMode_ == 1)
      return true;
    if (lv_.empty())
      return false;
    // Auto engages only for the SINGULAR (periodic / all-Neumann / IBM) operator. On the
    // Dirichlet-anchored (outflow) path the exact bottom measurably LOWERS the outer solve's
    // attainable floor (128x32x32 inflow/outflow channel: flux divergence floor 8e-8 smoothed vs
    // 2e-5 agglomerated at identical budgets; the CSR solution satisfies the V-cycle's own bottom
    // operator to 1e-9, so this is not operator mismatch — the anchored operator's near-null mode
    // makes the exact bottom return O(1e3 |b|) corrections whose float-hierarchy round-off the
    // smoothed bottom never generates). Until that is understood, anchored operators keep the
    // smoothed bottom; set_pressure_bottom("agglomerated") still forces it anywhere.
    if (!removeMean_)
      return false;
    // The criterion is the coarsest grid's largest EXTENT, not its cell count: what a few smoother
    // sweeps cannot fix is a mode spanning many cells along an axis, and Gauss-Seidel needs O(L^2)
    // sweeps to damp a wavelength of L cells. A 64x2x2 bottom is only 256 cells yet still 64 across
    // -- measured, that costs 6.0 pressure iterations/step against 4.0 for an exact solve.
    static const int thresh = [] {
      const char* e = std::getenv("PECLET_FLOW_AGGLOM_EXTENT");
      const int v = e ? std::atoi(e) : 4;
      return v > 0 ? v : 4;
    }();
    // coarsest GLOBAL cell count (the local block does not decide how hard the coarse solve is)
    long gx = gnxF_, gy = gnyF_, gz = gnzF_;
    for (int L = 0; L + 1 < (int)lv_.size(); ++L) {
      gx /= lv_[L].ratio.x;
      gy /= lv_[L].ratio.y;
      gz /= lv_[L].ratio.z;
    }
    return gx > thresh || gy > thresh || gz > thresh;
  }

  // --- Agglomerated GraphAMG bottom solve --------------------------------------------------------
  // Assemble the coarsest level's cut-cell operator as a GLOBAL CSR (gathered to rank 0) and build
  // a mesh-agnostic smoothed-aggregation AMG on it. Decomposition-agnostic: the CSR is keyed by
  // GLOBAL cell id (periodic-wrapped neighbours), so any (weighted) ORB gives the SAME operator.
  void buildAmg(Level& lv) {
    int gbx = gnxF_, gby = gnyF_,
        gbz = gnzF_;  // bottom global dims (coarsen by the ratios above it)
    for (int L = 0; L + 1 < (int)lv_.size(); ++L) {
      gbx /= lv_[L].ratio.x;
      gby /= lv_[L].ratio.y;
      gbz /= lv_[L].ratio.z;
    }
    amgGlobalN_ = gbx * gby * gbz;
    const int nx = lv.inner.x, ny = lv.inner.y, nz = lv.inner.z, ex = lv.ext.x, ey = lv.ext.y;
    auto host = [](FPV v) {
      auto h = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h, v);
      return h;
    };
    auto hC = host(lv.AC), hW = host(lv.AW), hE = host(lv.AE), hS = host(lv.AS), hN = host(lv.AN),
         hB = host(lv.AB), hT = host(lv.AT);
    // this rank's rows: (gid, diag) and off-diagonals (gid -> ngid, coef), periodic-wrapped.
    std::vector<int> lgid, lrow, lcol;
    std::vector<double> ldiag, lval;
    std::vector<std::uint8_t> lsolid;  // AGMG_DEBUG: identity-row marker, per local row
    amgGlobalOfLocal_.clear();
    const int band[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    const int g = lv.g;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const long p = (long)(i + g) + (long)(j + g) * ex + (long)(k + g) * ex * ey;
          const int gx = lv.og.x + i, gy = lv.og.y + j, gz = lv.og.z + k;
          const int gid = gx + gy * gbx + gz * gbx * gby;
          amgGlobalOfLocal_.push_back(gid);
          lgid.push_back(gid);
          // solid cells (all faces closed => diag 0, no coupling) get an identity row so D^-1 is
          // finite; their rhs is 0, so x stays 0 (correct — no flow inside the solid).
          const double dc = (double)hC(p);
          ldiag.push_back(dc != 0.0 ? dc : 1.0);
          lsolid.push_back(dc == 0.0 ? 1 : 0);
          const double bc[6] = {(double)hW(p), (double)hE(p), (double)hS(p),
                                (double)hN(p), (double)hB(p), (double)hT(p)};
          for (int d = 0; d < 6; ++d) {
            if (bc[d] == 0.0)
              continue;  // closed face (wall) -> no coupling
            // A face crossing the domain boundary couples to the wrapped cell ONLY on a periodic
            // axis (bc_ type 0). On a non-periodic axis an OPEN boundary face is the Dirichlet
            // outflow anchor: its coefficient lives in the diagonal (already in AC) with NO
            // off-diagonal partner — wrapping it would add a spurious top<->bottom coupling and
            // (with the mean projection skipped) a wrong, possibly indefinite bottom matrix.
            const int rx = gx + band[d][0], ry = gy + band[d][1], rz = gz + band[d][2];
            const int axis = d / 2;
            const bool crosses = (axis == 0 && (rx < 0 || rx >= gbx)) ||
                                 (axis == 1 && (ry < 0 || ry >= gby)) ||
                                 (axis == 2 && (rz < 0 || rz >= gbz));
            if (crosses && bc_[d] != 0)
              continue;  // non-periodic boundary face: Dirichlet anchor stays diagonal-only
            const int ngx = (rx % gbx + gbx) % gbx;
            const int ngy = (ry % gby + gby) % gby;
            const int ngz = (rz % gbz + gbz) % gbz;
            lrow.push_back(gid);
            lcol.push_back(ngx + ngy * gbx + ngz * gbx * gby);
            lval.push_back(bc[d]);
          }
        }
    // all-gather every rank's rows (no-op / identity single-rank): EVERY rank assembles the same
    // global CSR and builds the same AMG (redundant coarse solve).
    std::vector<int> ggid = lgid, grow = lrow, gcol = lcol;
    std::vector<double> gdiag = ldiag, gval = lval;
    std::vector<std::uint8_t> gsolid = lsolid;
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      gatherv(lgid, ggid);
      gatherv(ldiag, gdiag);
      gatherv(lrow, grow);
      gatherv(lcol, gcol);
      gatherv(lval, gval);
      gatherv(lsolid, gsolid);
    }
#endif
    amgSolid_.assign((std::size_t)amgGlobalN_, 0);
    for (std::size_t r = 0; r < ggid.size(); ++r)
      amgSolid_[(std::size_t)ggid[r]] = gsolid[r];
    {  // connected components of the operator graph (union-find over the off-diagonal edges):
      // each FLUID component carries its own constant null vector, so the null-space projection
      // must be per-component. Solid identity rows are singletons and take no projection.
      std::vector<int> parent((std::size_t)amgGlobalN_);
      for (int i = 0; i < amgGlobalN_; ++i)
        parent[(std::size_t)i] = i;
      auto find = [&](int a) {
        while (parent[(std::size_t)a] != a)
          a = parent[(std::size_t)a] = parent[(std::size_t)parent[(std::size_t)a]];
        return a;
      };
      for (std::size_t e = 0; e < grow.size(); ++e) {
        const int ra = find(grow[e]), rb = find(gcol[e]);
        if (ra != rb)
          parent[(std::size_t)ra] = rb;
      }
      amgComp_.assign((std::size_t)amgGlobalN_, -1);
      std::vector<int> remap((std::size_t)amgGlobalN_, -1);
      amgNComp_ = 0;
      for (int i = 0; i < amgGlobalN_; ++i) {
        if (amgSolid_[(std::size_t)i])
          continue;  // identity row: no null space, excluded from projection
        const int r = find(i);
        if (remap[(std::size_t)r] < 0)
          remap[(std::size_t)r] = amgNComp_++;
        amgComp_[(std::size_t)i] = remap[(std::size_t)r];
      }
    }
    if (agmgDebug()) {
      std::vector<long> csize((std::size_t)amgNComp_, 0);
      for (int i = 0; i < amgGlobalN_; ++i)
        if (amgComp_[(std::size_t)i] >= 0)
          ++csize[(std::size_t)amgComp_[(std::size_t)i]];
      printf("[agmg-build] fluid components=%d  sizes:", amgNComp_);
      for (int c = 0; c < std::min(amgNComp_, 12); ++c)
        printf(" %ld", csize[(std::size_t)c]);
      printf(amgNComp_ > 12 ? " ...\n" : "\n");
    }
    if (agmgDebug()) {
      long nSolid = 0, nTiny30 = 0, nTiny12 = 0;
      double minFluidDiag = 1e300, maxFluidDiag = 0;
      for (std::size_t r = 0; r < ggid.size(); ++r) {
        if (gsolid[r]) {
          ++nSolid;
          continue;
        }
        const double ad = std::fabs(gdiag[r]);
        minFluidDiag = std::min(minFluidDiag, ad);
        maxFluidDiag = std::max(maxFluidDiag, ad);
        nTiny30 += ad < 1e-30;
        nTiny12 += ad < 1e-12;
      }
      printf("[agmg-build] n=%d solid=%ld fluid=%ld  fluid|diag| min=%.3e max=%.3e  "
             "tiny<1e-30=%ld <1e-12=%ld\n",
             amgGlobalN_, nSolid, (long)ggid.size() - nSolid, minFluidDiag, maxFluidDiag, nTiny30,
             nTiny12);
      // Row-sum defect: the operator's null vector is the constant ONLY if every fluid row sums
      // to zero. The level coefficients are stored in float (MReal), so the diagonal is a
      // float-rounded sum of the face coefficients — a nonzero defect here bounds how far a
      // singular-consistent solve can converge.
      std::vector<double> rowsum((std::size_t)amgGlobalN_, 0.0);
      for (std::size_t r = 0; r < ggid.size(); ++r)
        rowsum[(std::size_t)ggid[r]] = gsolid[r] ? 0.0 : gdiag[r];
      for (std::size_t e = 0; e < grow.size(); ++e)
        if (!amgSolid_[(std::size_t)grow[e]])
          rowsum[(std::size_t)grow[e]] += gval[e];
      double defMax = 0, defRelMax = 0;
      for (std::size_t r = 0; r < ggid.size(); ++r)
        if (!gsolid[r]) {
          const double d = std::fabs(rowsum[(std::size_t)ggid[r]]);
          defMax = std::max(defMax, d);
          defRelMax = std::max(defRelMax, d / std::fabs(gdiag[r]));
        }
      printf("[agmg-build] fluid row-sum defect max=%.3e rel=%.3e\n", defMax, defRelMax);
      fflush(stdout);
    }
    {  // assemble the global CSR keyed by gid
      peclet::core::solver::HostCsrOp A;
      A.n = amgGlobalN_;
      A.diag.assign((std::size_t)amgGlobalN_, 0.0);
      for (std::size_t r = 0; r < ggid.size(); ++r)
        A.diag[(std::size_t)ggid[r]] = gdiag[r];
      std::vector<std::vector<std::pair<int, double>>> rows((std::size_t)amgGlobalN_);
      for (std::size_t e = 0; e < grow.size(); ++e)
        rows[(std::size_t)grow[e]].push_back({gcol[e], gval[e]});
      A.start.assign((std::size_t)amgGlobalN_ + 1, 0);
      for (int r = 0; r < amgGlobalN_; ++r)
        A.start[(std::size_t)r + 1] = A.start[(std::size_t)r] + (long)rows[(std::size_t)r].size();
      A.nbr.reserve(grow.size());
      A.coef.reserve(grow.size());
      for (int r = 0; r < amgGlobalN_; ++r)
        for (auto& [c, v] : rows[(std::size_t)r]) {
          A.nbr.push_back(c);
          A.coef.push_back(v);
        }
      // Singular (periodic / all-Neumann) path: every fluid diagonal is BY CONSTRUCTION the
      // negative sum of its off-diagonals (walls/solids contribute zero, and no Dirichlet anchor
      // exists when removeMean_). The float (MReal) level storage breaks that identity at ~5e-8
      // relative, which shifts the operator's near-null vector off the constant the null-space
      // projection assumes — measured as the inner CG flooring at ~1e-5 and burning its full
      // iteration cap every call. Resum the diagonal in double so A·1 = 0 EXACTLY per fluid row.
      if (removeMean_)
        for (int r = 0; r < amgGlobalN_; ++r)
          if (!amgSolid_[(std::size_t)r] && !rows[(std::size_t)r].empty()) {
            double s = 0;
            for (auto& [c, v] : rows[(std::size_t)r])
              s += v;
            A.diag[(std::size_t)r] = -s;
          }
      amgA_ = A;
      amg_ = std::make_shared<peclet::core::solver::GraphAMG>();
      amg_->build(A);
    }
  }
  // Solve the coarsest level with the agglomerated AMG, REDUNDANTLY: all-gather the coarse rhs,
  // every rank runs the identical GraphAMG-preconditioned CG on the identical global operator
  // (deterministic serial code on identical data => bit-identical solutions), and each extracts
  // its own block — one Allgatherv per V-cycle, no rank-0 serialization, no result broadcast.
  void graphAmgSolveBottom(Level& lv) {
    if (!amg_ && !distributed_)
      buildAmg(lv);
#ifdef PECLET_FLOW_MPI
    if (distributed_ && amgGlobalN_ == 0)
      buildAmg(lv);
#endif
    const int nx = lv.inner.x, ny = lv.inner.y, nz = lv.inner.z, ex = lv.ext.x, ey = lv.ext.y;
    const int g = lv.g;
    auto hrhs = Kokkos::create_mirror_view(lv.rhs);
    Kokkos::deep_copy(hrhs, lv.rhs);
    std::vector<double> lb;
    lb.reserve(amgGlobalOfLocal_.size());
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          lb.push_back((double)hrhs((long)(i + g) + (long)(j + g) * ex + (long)(k + g) * ex * ey));
    // all-gather rhs by global id -> b; every rank solves the identical problem.
    std::vector<double> z((std::size_t)std::max(amgGlobalN_, 1), 0.0);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::vector<int> ggid;
      std::vector<double> gb;
      gatherv(amgGlobalOfLocal_, ggid);
      gatherv(lb, gb);
      std::vector<double> b((std::size_t)amgGlobalN_, 0.0);
      for (std::size_t r = 0; r < ggid.size(); ++r)
        b[(std::size_t)ggid[r]] = gb[r];
      pcgAmg(b, z);
    } else
#endif
    {
      std::vector<double> b(lb.begin(), lb.end());
      pcgAmg(b, z);
    }
    // scatter z[gid] back into this rank's inner cells.
    auto hx = Kokkos::create_mirror_view(lv.x);
    Kokkos::deep_copy(hx, 0.0);
    std::size_t c = 0;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          hx((long)(i + g) + (long)(j + g) * ex + (long)(k + g) * ex * ey) =
              z[(std::size_t)amgGlobalOfLocal_[c++]];
    Kokkos::deep_copy(lv.x, hx);
    if (agmgDebug() && !distributed_) {
      // Consistency check: the CSR solution must satisfy the V-cycle's OWN bottom operator
      // (ghost fill + outflow ghost + 7-point apply). A large residual here means buildAmg
      // assembled a DIFFERENT matrix than the one the hierarchy applies.
      fill(lv, lv.x);
      applyOutflowGhost(lv, lv.x, lv.g);
      residualCutcell(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                      FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), lv.ext, lv.g);
      const double rn = maxabs(lv, lv.res);
      auto hb = Kokkos::create_mirror_view(lv.rhs);
      Kokkos::deep_copy(hb, lv.rhs);
      double bn = 0;
      for (std::size_t i = 0; i < hb.size(); ++i)
        bn = std::max(bn, std::fabs((double)hb(i)));
      printf("[agmg] vcycle-op residual of CSR solution: max|b-Ax|=%.3e  max|b|=%.3e  rel=%.3e\n",
             rn, bn, bn > 0 ? rn / bn : 0.0);
      fflush(stdout);
    }
  }
  // GraphAMG-preconditioned CG on the global bottom operator. For the periodic/all-Neumann case the
  // operator is singular (constant null space) and the mean must be projected out of the rhs and
  // the preconditioned residual (compatibility). With a Dirichlet outflow (removeMean_ == false)
  // the operator is NON-singular and the projection must be SKIPPED — removing the constant from a
  // non-singular system returns a wrong bottom correction and the V-cycle around it diverges.
  // Runs on rank 0 only.
  void pcgAmg(std::vector<double>& b, std::vector<double>& x) {
    const std::size_t n = (std::size_t)amgGlobalN_;
    const int dbg = agmgDebug();
    double bSolidMax = 0, bFluidMax = 0, bFluidMean = 0, bAllMean = 0;
    if (dbg) {  // rhs anatomy BEFORE the null-space projection (b is gid-ordered)
      long nf = 0;
      double sf = 0, sa = 0;
      for (std::size_t i = 0; i < n; ++i) {
        sa += b[i];
        if (i < amgSolid_.size() && amgSolid_[i])
          bSolidMax = std::max(bSolidMax, std::fabs(b[i]));
        else {
          bFluidMax = std::max(bFluidMax, std::fabs(b[i]));
          sf += b[i];
          ++nf;
        }
      }
      bFluidMean = nf ? sf / (double)nf : 0.0;
      bAllMean = n ? sa / (double)n : 0.0;
    }
    double bCompMax = 0;  // max per-component |sum(b)| / |b|_max: the per-pocket incompatibility
    if (dbg && amgNComp_ > 1) {
      std::vector<double> cs((std::size_t)amgNComp_, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        if (amgComp_[i] >= 0)
          cs[(std::size_t)amgComp_[i]] += b[i];
      for (double s : cs)
        bCompMax = std::max(bCompMax, std::fabs(s));
      bCompMax /= (bFluidMax > 0 ? bFluidMax : 1.0);
    }
    auto meanZero = [&](std::vector<double>& v) {
      if (!removeMean_)
        return;  // Dirichlet-anchored (outflow) operator: non-singular, no null space to project
      // The null space is one constant PER CONNECTED FLUID COMPONENT — solid cells are identity
      // rows (non-singular) and a coarse level can pinch fluid off into pockets, each carrying its
      // own constant. Projecting the ALL-cell mean out instead (the old code) both leaves null
      // components alive and writes a spurious value onto every solid coordinate; the next matvec
      // (identity rows) feeds that back into the residual, the effective preconditioner turns
      // nonsymmetric, and the inner CG stalls at its iteration cap (measured on random_spheres:
      // every bottom solve capped at 200 with relres up to ~1, +41% outer iterations).
      if (amgNComp_ <= 0)
        return;
      std::vector<double> m((std::size_t)amgNComp_, 0.0);
      std::vector<long> cnt((std::size_t)amgNComp_, 0);
      for (std::size_t i = 0; i < n; ++i)
        if (amgComp_[i] >= 0) {
          m[(std::size_t)amgComp_[i]] += v[i];
          ++cnt[(std::size_t)amgComp_[i]];
        }
      for (std::size_t c = 0; c < m.size(); ++c)
        m[c] = cnt[c] ? m[c] / (double)cnt[c] : 0.0;
      for (std::size_t i = 0; i < n; ++i)
        if (amgComp_[i] >= 0)
          v[i] -= m[(std::size_t)amgComp_[i]];
    };
    meanZero(b);
    x.assign(n, 0.0);
    std::vector<double> r = b, z(n), p(n), Ap(n);
    amg_->apply(r, z);
    meanZero(z);
    p = z;
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
      double s = 0;
      for (std::size_t i = 0; i < n; ++i)
        s += a[i] * c[i];
      return s;
    };
    double rz = dot(r, z), r0 = std::sqrt(dot(r, r));
    // 1e-8 is deliberate: the V-cycle's outer iteration count is unchanged from a far looser
    // bottom (measured), while 1e-10 sits at/below the double-precision floor of this
    // projected solve (rounding of the per-iteration null-space projection), where CG grinds
    // out its full iteration cap for nothing.
    int it = 0;
    for (; it < 100 && r0 > 0; ++it) {
      amgA_.apply(p, Ap);
      const double a = rz / dot(p, Ap);
      for (std::size_t i = 0; i < n; ++i) {
        x[i] += a * p[i];
        r[i] -= a * Ap[i];
      }
      if (std::sqrt(dot(r, r)) <= 1e-8 * r0)
        break;
      amg_->apply(r, z);
      meanZero(z);
      const double rzn = dot(r, z);
      const double beta = rzn / rz;
      rz = rzn;
      for (std::size_t i = 0; i < n; ++i)
        p[i] = z[i] + beta * p[i];
    }
    meanZero(x);
    if (dbg) {
      double xSolidMax = 0, xFluidMax = 0;
      for (std::size_t i = 0; i < n; ++i)
        if (i < amgSolid_.size() && amgSolid_[i])
          xSolidMax = std::max(xSolidMax, std::fabs(x[i]));
        else
          xFluidMax = std::max(xFluidMax, std::fabs(x[i]));
      const double rn = std::sqrt(dot(r, r));
      ++agmgCalls_;
      if (dbg >= 2 || agmgCalls_ <= 60 || it >= 100 || agmgCalls_ % 50 == 0) {
        printf("[agmg] call=%ld iters=%d relres=%.2e  |b|sol=%.3e |b|fl=%.3e  "
               "mean(b) fl=%.3e all=%.3e  compat=%.2e  |x|sol=%.3e |x|fl=%.3e%s\n",
               agmgCalls_, it, r0 > 0 ? rn / r0 : 0.0, bSolidMax, bFluidMax, bFluidMean, bAllMean,
               bCompMax, xSolidMax, xFluidMax, it >= 100 ? "  CAP" : "");
        fflush(stdout);
      }
    }
  }
#ifdef PECLET_FLOW_MPI
  template <class T>
  void gatherv(const std::vector<T>& local, std::vector<T>& all) {
    // REDUNDANT agglomeration: every rank receives the full concatenation (rank order, so the
    // assembled coarse problem is bit-identical on all ranks and each solves it locally with no
    // rank-0 bottleneck and no result broadcast).
    int size = 1;
    MPI_Comm_size(comm_, &size);
    const int lbytes = (int)(local.size() * sizeof(T));
    std::vector<int> bc(size), bd(size, 0);
    MPI_Allgather(&lbytes, 1, MPI_INT, bc.data(), 1, MPI_INT, comm_);
    int tot = 0;
    for (int r = 0; r < size; ++r) {
      bd[r] = tot;
      tot += bc[r];
    }
    all.resize((std::size_t)tot / sizeof(T));
    MPI_Allgatherv(local.data(), lbytes, MPI_BYTE, all.data(), bc.data(), bd.data(), MPI_BYTE,
                   comm_);
  }
#endif

  // Level-0 matvec with the halo overlapped: post the exchange, apply the interior rows while the
  // messages are in flight, land the ghosts (+ outflow ghost), apply the boundary shell. Reads v,
  // writes y (no aliasing) => bit-identical to the blocking fill-then-apply. Single-rank: the
  // blocking path.
  void matvecOverlap(Level& l0, CCField y, CCField v) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const C3 lo{G + 1, G + 1, G + 1};
      const C3 hi{l0.ext.x - G - 1, l0.ext.y - G - 1, l0.ext.z - G - 1};
      l0.dev->exchangeBegin(v);
      applyCutcellOpBox(y, CCConst(v), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN),
                        FPC(l0.AB), FPC(l0.AT), l0.ext, lo, hi, C3{0, 0, 0}, C3{0, 0, 0});
      l0.dev->exchangeEnd(v);
      applyOutflowGhost(l0, v);
      applyCutcellOpBox(y, CCConst(v), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN),
                        FPC(l0.AB), FPC(l0.AT), l0.ext, C3{G, G, G},
                        C3{l0.ext.x - G, l0.ext.y - G, l0.ext.z - G}, lo, hi);
      return;
    }
#endif
    fill(l0, v);
    applyOutflowGhost(l0, v);
    applyCutcellOp(y, CCConst(v), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE), FPC(l0.AS), FPC(l0.AN),
                   FPC(l0.AB), FPC(l0.AT), l0.ext, G);
  }
  // periodic ghost fill (3 axes) of a level-sized field / the openness triple. Distributed: the
  // per-level core halo (cross-rank + periodic in one call).
  void fill(Level& lv, CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      lv.dev->exchange(f);
      return;
    }
#endif
    fillAxis(lv, f, 0);
    fillAxis(lv, f, 1);
    fillAxis(lv, f, 2);
  }
  void fillOpenness(Level& lv) {
    fill(lv, lv.ox);
    fill(lv, lv.oy);
    fill(lv, lv.oz);
  }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space;
    C3 e = lv.ext;
    const int G = lv.g;  // shadows the class constant: this level's ghost width
    int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    const int N = N3[a];
    CCField ff = f;
    Kokkos::parallel_for(
        "peclet::flow::mg_pfill",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
        KOKKOS_LAMBDA(int p0, int p1) {
          const long base = (long)p0 * sb + (long)p1 * sc;
          for (int gl = 0; gl < G; ++gl) {
            ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
            ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
          }
        });
  }
#ifdef PECLET_FLOW_MPI
  // ghost-projection matvec staging (solveBiCGStab distributed): copy the l0 inner cells onto the
  // caller's g=2 block (whose halo then carries the overlay's +/-2 reach) ...
  void stageG2(Level& l0, CCField q, CCField xg2, C3 ext2) {
    CCExec space;
    const C3 e1 = l0.ext, nn = l0.inner;
    CCField dst = xg2, src = q;
    Kokkos::parallel_for(
        "peclet::flow::gp_stage_g2",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nn.x, nn.y, nn.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          dst((long)(x + 2) + (long)(y + 2) * ext2.x + (long)(z + 2) * (long)ext2.x * ext2.y) =
              src((long)(x + G) + (long)(y + G) * e1.x + (long)(z + G) * (long)e1.x * e1.y);
        });
  }
  // ... and read the whole l0 block (inner + its g=1 ring) back from the exchanged g=2 copy, so
  // the 7-point op's halo is current without a second exchange.
  void unstageG2(Level& l0, CCField q, CCField xg2) {
    CCExec space;
    const C3 e1 = l0.ext;
    const C3 ext2{e1.x + 2, e1.y + 2, e1.z + 2};  // same inner, gb 1 -> 2
    CCField dst = q, src = xg2;
    Kokkos::parallel_for(
        "peclet::flow::gp_unstage_g2",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {e1.x, e1.y, e1.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          dst((long)x + (long)y * e1.x + (long)z * (long)e1.x * e1.y) =
              src((long)(x + 1) + (long)(y + 1) * ext2.x + (long)(z + 1) * (long)ext2.x * ext2.y);
        });
  }
#endif
  void axpy(CCField y, double a, CCField x) {
    CCExec space;
    CCField yy = y, xx = x;
    std::size_t n = y.extent(0);
    Kokkos::parallel_for(
        "mgaxpy", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { yy(i) += a * xx(i); });
  }
  void aypx(CCField y, double a, CCField x) {
    CCExec space;
    CCField yy = y, xx = x;
    std::size_t n = y.extent(0);
    Kokkos::parallel_for(
        "mgaypx", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { yy(i) = xx(i) + a * yy(i); });
  }
  void scale(CCField y, double a) {
    CCExec space;
    CCField yy = y;
    std::size_t n = y.extent(0);
    Kokkos::parallel_for(
        "mgscale", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { yy(i) *= a; });
  }
  void lin(CCField out, double a, CCField x, double b, CCField y) {  // out = a*x + b*y (mg_lin_k)
    CCExec space;
    CCField oo = out, xx = x, yy = y;
    std::size_t n = out.extent(0);
    Kokkos::parallel_for(
        "mglin", Kokkos::RangePolicy<CCExec>(space, 0, n),
        KOKKOS_LAMBDA(std::size_t i) { oo(i) = a * xx(i) + b * yy(i); });
  }
  // zero the solid-cell entries (AC<=tiny) -> project out the solid null modes (mg_mask_solid_k).
  void maskSolid(Level& lv, CCField f) {
    CCExec space;
    CCField ff = f;
    FPV ac = lv.AC;
    std::size_t n = f.extent(0);
    Kokkos::parallel_for(
        "mgmasksolid", Kokkos::RangePolicy<CCExec>(space, 0, n), KOKKOS_LAMBDA(std::size_t i) {
          if (!(ac(i) > 1e-30f))
            ff(i) = 0.0;
        });
  }

  // Estimate the spectral bounds [lmin,lmax] of M^{-1}A (M^{-1} = one symmetric V-cycle) by power
  // iteration (direct for the max + a shifted iteration for the min), seeded by `seed`.
  // Communication-heavy, so the CUDA driver runs it once on step 1 and reuses the bounds. Port of
  // estimate_eigenvalues.
  void estimateEigenvalues(CCConst seed, double& lmin, double& lmax, int iters, int pre, int post,
                           int bottom) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    const std::size_t n = l0.n;
    CCField v("ev_v", n), w("ev_w", n), z("ev_z", n), srhs("ev_srhs", n);
    Kokkos::deep_copy(srhs, seed);
    auto matvec = [&](CCField y, CCField x) { matvecOverlap(l0, y, x); };
    auto applyT = [&](CCField out,
                      CCField in) {  // out = M^{-1} A in, projected onto the fluid range
      matvec(w, in);
      Kokkos::deep_copy(l0.rhs, w);
      Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(out, l0.x);
      removeMean(l0, out);
      maskSolid(l0, out);
    };
    auto normalize = [&](CCField x) {
      double nr = std::sqrt(dot(l0, x, x));
      if (nr > 0)
        scale(x, 1.0 / nr);
    };
    auto seedf = [&](CCField x) {
      Kokkos::deep_copy(x, srhs);
      removeMean(l0, x);
      maskSolid(l0, x);
      normalize(x);
    };
    seedf(v);
    lmax = 1.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      lmax = dot(l0, v, z);
      Kokkos::deep_copy(v, z);
      normalize(v);
    }
    seedf(v);
    double mu = 0.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      lin(z, lmax, v, -1.0, z);  // z = lmax*v - T v
      mu = dot(l0, v, z);
      Kokkos::deep_copy(v, z);
      normalize(v);
    }
    double e_hi = lmax, e_lo = lmax - mu;  // direct (max) + shifted (min) Rayleigh estimates
    lmin = e_lo < e_hi ? e_lo : e_hi;
    lmax = e_lo < e_hi ? e_hi : e_lo;  // robust min/max bracket
    if (lmin < 0.02 * lmax)
      lmin = 0.02 * lmax;
  }

  // Chebyshev semi-iteration preconditioned by ONE symmetric V-cycle -- same goal as solvePCG but
  // the step coefficients come from the spectral bounds [a,b], so NO per-iteration global
  // dot-products (communication- light at scale). rhs on level-0 supplied as `b`; solution left in
  // `x`. Returns the V-cycle count. Port of solve_chebyshev.
  int solveChebyshev(CCField b, CCField x, int maxit, double rtol, int pre, int post, int bottom,
                     double a, double bnd) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    const std::size_t n = l0.n;
    if (a > bnd) {
      double t = a;
      a = bnd;
      bnd = t;
    }  // robust to swapped bounds
    a *= 0.95;
    bnd *= 1.05;  // safety margin: [a,b] must bracket the spectrum
    CCField r("cb_r", n), z("cb_z", n), d("cb_d", n), w("cb_w", n);
    auto matvec = [&](CCField y, CCField v) { matvecOverlap(l0, y, v); };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(l0.rhs, rr);
      Kokkos::deep_copy(l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(zz, l0.x);
    };
    const double theta = 0.5 * (bnd + a), delta = 0.5 * (bnd - a), sigma1 = theta / delta;
    double rho = 1.0 / sigma1;
    matvec(w, x);  // r = b - A x
    Kokkos::deep_copy(r, b);
    axpy(r, -1.0, w);
    removeMean(l0, r);
    const double r0 = maxabs(l0, r);
    int nvc = 0;
    if (r0 > 0.0) {
      precond(z, r);
      ++nvc;  // z = M^{-1} r
      lin(d, 1.0 / theta, z, 0.0, z);
      axpy(x, 1.0, d);  // d = z/theta; x += d
      for (int i = 1; i < maxit; ++i) {
        matvec(w, d);
        axpy(r, -1.0, w);
        removeMean(l0, r);  // r -= A d
        if (maxabs(l0, r) < rtol * r0)
          break;
        precond(z, r);
        ++nvc;
        const double rho_new = 1.0 / (2.0 * sigma1 - rho);
        lin(d, rho_new * rho, d, 2.0 * rho_new / delta, z);
        axpy(x, 1.0, d);  // d update; x += d
        rho = rho_new;
      }
    }
    removeMean(l0, x);
    return nvc;
  }
  // reductions / mean removal over inner FLUID cells (AC>tiny) of a level.
  double dot(Level& lv, CCField a, CCField b) {
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a, bb = b;
    FPV ac = lv.AC;
    double s = 0;
    ccReduce3(
        "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f)
            acc += aa(i) * bb(i);
        },
        s);
    return allreduce(s, MPI_SUM_);
  }
  double maxabs(Level& lv, CCField a) {
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a;
    FPV ac = lv.AC;
    double m = 0;
    ccReduce3(
        "mgmax", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f) {
            const double v = Kokkos::fabs(aa(i));
            if (v > acc)
              acc = v;
          }
        },
        Kokkos::Max<double>(m));
    return allreduce(m, MPI_MAX_);
  }
  void removeMean(Level& lv, CCField f) {
    if (!removeMean_)
      return;  // non-singular operator (Dirichlet outflow present) -> no null-space projection
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField ff = f;
    FPV ac = lv.AC;
    double sum = 0;
    long cnt = 0;
    Kokkos::parallel_reduce(
        "mgmeanr",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {g, g, g},
                                                       {e.x - g, e.y - g, e.z - g}),
        KOKKOS_LAMBDA(int x, int y, int z, double& s, long& k) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f) {
            s += ff(i);
            k += 1;
          }
        },
        sum, cnt);
    double dcnt = (double)cnt;
    allreduceSum2(sum, dcnt);  // ONE latency hit for the {sum, count} pair (was two)
    cnt = (long)dcnt;
    if (cnt == 0)
      return;
    const double mean = sum / (double)cnt;
    Kokkos::parallel_for(
        "mgmeans",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {g, g, g},
                                                       {e.x - g, e.y - g, e.z - g}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f)
            ff(i) -= mean;
        });
  }

  // Mean-removal scope. "all" (legacy): project the nullspace out at every V-cycle level,
  // after every matvec and on every residual update — ~10 extra MPI_Allreduce latency hits per
  // Krylov iteration whose only role is FP hygiene. "fine" (DEFAULT) keeps the removals that carry the
  // algorithm (the rhs/residual projections + the fine-level V-cycle exit + the final iterate) and
  // drops the interior-level ones: A maps mean-free vectors to mean-free vectors, so the Krylov
  // space never sees the dropped components (they lie in the nullspace and are removed from the
  // final x). Validated by iteration-count parity; not bit-identical to "all".
  void setMeanRemovalScope(bool all) { meanRemovalAll_ = all; }

  // Accumulated wall time / call count of the global reductions (every dot product, residual max
  // and mean-removal funnels through allreduce()). This is THE latency-bound term of the
  // distributed pressure solve; the solver resets it per step and exposes it to Python.
  double allreduceSeconds() const { return allreduceTime_; }
  long allreduceCount() const { return allreduceCount_; }
  void resetAllreduceCounters() {
    allreduceTime_ = 0.0;
    allreduceCount_ = 0;
  }

 private:
  enum AllOp { kSum, kMax };
  // Global reduction over ranks (no-op single-rank / non-distributed -> byte-identical to the local
  // reduce).
  double allreduce(double v, AllOp op) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const auto t0 = std::chrono::steady_clock::now();
      double g = 0;
      MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, op == kSum ? MPI_SUM : MPI_MAX, comm_);
      allreduceTime_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++allreduceCount_;
      return g;
    }
#endif
    (void)op;
    return v;
  }
  // One MPI_Allreduce of a {sum, count} pair — elementwise MPI_SUM on a 2-vector is bit-identical
  // to two separate allreduces, at half the latency hits.
  void allreduceSum2(double& a, double& b) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const auto t0 = std::chrono::steady_clock::now();
      double v[2] = {a, b}, g[2] = {0.0, 0.0};
      MPI_Allreduce(v, g, 2, MPI_DOUBLE, MPI_SUM, comm_);
      allreduceTime_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++allreduceCount_;
      a = g[0];
      b = g[1];
    }
#endif
  }
  static constexpr AllOp MPI_SUM_ = kSum, MPI_MAX_ = kMax;

  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 4;
  int bc_[6] = {0, 0, 0, 0, 0, 0};
  bool hasBC_ = false, removeMean_ = true, hasOutflow_ = false;
  // Default "fine" (measured winner of the at-scale ablation, Snellius H100 8+16 GPUs: 5.5%
  // faster than "all" with identical iteration counts; single-rank the reductions are free either
  // way). setMeanRemovalScope(true) restores the legacy every-level scope.
  bool meanRemovalAll_ = false;
  bool distributed_ = false;
  int dbgSolve_ = 0;  // solve counter for the env-gated convergence trace (mgDebugLevel() >= 2)
  std::vector<double> lvTime_;  // per-level V-cycle wall time (mgDebugLevel() >= 3)
  int lvCycles_ = 0;
  // Halo refresh before the V-cycle's residual (see vcycle). ON by default — the legacy
  // stale-ghost residual is kept behind PECLET_FLOW_MG_RESFILL=0 purely as a benchmark ablation.
  bool resFill_ = [] {
    const char* e = std::getenv("PECLET_FLOW_MG_RESFILL");
    return !e || std::atoi(e) != 0;
  }();
  double allreduceTime_ = 0.0;
  long allreduceCount_ = 0;
  // --- decomposition-agnostic algebraic bottom solve (GraphAMG) ---
  // The geometric coarse hierarchy needs a cleanly-coarsening (equal-weight) ORB. Under a WEIGHTED
  // decomposition the coarse levels misalign, so the multilevel path is unavailable and only pure
  // RB-GS (nLevels==1) works. With this enabled, the coarsest level is solved by an AGGLOMERATED
  // algebraic multigrid: the operator + rhs of the coarsest level are gathered to rank 0, solved by
  // a mesh-agnostic smoothed-aggregation AMG (core::solver::GraphAMG, exact by construction on any
  // decomposition), and the solution scattered back. With nLevels==1 this makes the whole pressure
  // solve mesh-independent AND decomposition-agnostic.
  int agglomMode_ = 0;  // 0 smoothed bottom (default), -1 auto (see agglomerateBottom), 1 always
  int gnxF_ = 0, gnyF_ = 0, gnzF_ = 0;  // GLOBAL fine dims (== local single-rank)
  mutable std::shared_ptr<peclet::core::solver::GraphAMG>
      amg_;  // built once from the bottom operator
  mutable peclet::core::solver::HostCsrOp
      amgA_;  // rank 0: the assembled global bottom operator (CG matvec)
  mutable std::vector<int> amgOwnerCount_;  // rank 0: #bottom cells each rank owns (gather layout)
  mutable std::vector<int>
      amgGlobalOfLocal_;        // this rank's bottom inner cells -> global bottom index
  mutable int amgGlobalN_ = 0;  // total bottom global cells (rank 0)
  // PECLET_FLOW_AGMG_DEBUG instrumentation (see agmgDebug()): per-gid solid marker + call counter.
  mutable std::vector<std::uint8_t> amgSolid_;
  mutable std::vector<int> amgComp_;  // fluid component id per gid (-1 = solid identity row)
  mutable int amgNComp_ = 0;          // number of fluid components (null-space dimension)
  mutable long agmgCalls_ = 0;
  static int agmgDebug() {
    static const int v = [] {
      const char* e = std::getenv("PECLET_FLOW_AGMG_DEBUG");
      return e ? std::atoi(e) : 0;
    }();
    return v;
  }
#ifdef PECLET_FLOW_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif

 public:
  // Enable the agglomerated GraphAMG bottom solve (decomposition-agnostic multigrid coarse solve).
  // Rebuilds lazily on the next solve. Safe single-rank (local assemble + serial AMG).
  void setAgglomerationMode(int mode) { agglomMode_ = mode; }  // -1 auto, 0 never, 1 always
  int agglomerationMode() const { return agglomMode_; }
  void setGraphAmgBottom(bool on) {
    agglomMode_ = on ? 1 : 0;
    amg_.reset();
  }

 private:
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CUTCELL_MG_HPP
