// ScalarMG: the multigrid preconditioner of the cut-cell scalar path (doc/scalar_ibm_design.md
// §5.2, D10; WO-4 single phase). One instance per cut-cell scalar; it preconditions the SPD
// lumped-probe SURROGATE S of §4.3 — the BiCGStab of `scalar_krylov.hpp` runs on the true probe
// operator, and one V-cycle of this class is its z = M^-1 r.
//
// Level table (§5.2): VelocityMG::init / initMpi's rule verbatim — in-place coarsening per axis
// while the global dimension is even and >= 4 and (distributed) every rank's block origin and size
// are even on that axis, then the shared aspect rule `CutcellMG::mgChooseRatio` (theta = 2, read
// from setMetric's h'). No depth cap (the "full table"), no telescoping (§13 Q3). Level 0 is the
// scalar's own G = 2 block (its ghost exchange is the solver's, injected as `Fill`); coarse levels
// carry g = 1 and their own `GridHalo<double>`.
//
// Per level the surrogate is stored in FACE form, double: AC (diagonal) and AFX/AFY/AFZ, AFX(i) the
// off-diagonal coefficient of cell i's LOW x face (mac_pressure.hpp's face form). Level 0 (§4.3):
// AC = the WO-3 surrogate diagonal SAC (bands' diagonal + the lumped wall term), AF = -Lam w_a a,
// guarded toward a cell that is not an unknown exactly as `sco::buildBands` (so the level-0 sweeps
// are bitwise the WO-3 band sweeps). Coarse level L, coarse cell C (REDISCRETIZED, §5.2):
//   * face:  t_a = w_a(L) <Lam a>, w_a(L) = w_a / cfac_a^2, <Lam a> the plain average of the fine
//            sub-faces' products, coarsened recursively by `coarsenOpenAvg`'s cell body;
//   * mass:  m_C = restrictAvg of the children's m (kappa idt, + the lumped implicit outflow
//            under advection, WO-5);
//   * wall:  W_C = (1/N_L) sum over the level-0 facets under C of alpha G(s_phi) — the plain
//   average
//            of the level-0 wall terms cw, at the FINE probe distance (design Amendment A1; the
//            level's own s_L overshoots the coarse correction by ~2^L and diverged, log WO-4).
//            Robin and Dirichlet alike (Neumann: 0); in WO-7 the conjugate coupling the same way.
//            A direct gather, one thread per coarse cell over its level-0 descendants in fixed
//            order;
//   * pins:  C is pinned iff every child is (the unknown flags restricted with max); pinned rows
//   are
//            identity rows with zero correction;
//   * Dirichlet domain faces: 2 w_a(L) <Lam a_bf>, the boundary plane averaged level by level.
// Ghosts beyond a non-periodic global face are pinned on every level (the record clears the level-0
// ghost unknown flags; a coarse ghost's children are those ghosts), so a coarse correction is 0
// there before the trilinear prolongation reads it, and every face coefficient toward them is 0.
//
// Smoother: red-black Gauss-Seidel, colour = (gx + gy + gz) mod 2 from GLOBAL indices, an exchange
// before each colour. V-cycle: pre 2 sweeps R -> B, residual (after a fresh exchange), restrictAvg,
// (singular: the coarse rhs mean removed), recurse, coarse ghosts filled, prolongAdd (trilinear),
// pinned cells re-zeroed, post 2 sweeps B -> R. Bottom: 16 sweeps (8 R -> B then 8 B -> R, so the
// cycle stays symmetric) plus the mean removal when singular. The level rule (§5.2): transient with
// kappa_A = 1 + 4 dt' D' sum_a w_a < 13 uses level 0 alone (2 + 2 sweeps, the WO-3 preconditioner);
// otherwise, and always when steady, the full table. The cycle is a fixed linear operator.
//
// Steady mode WITH advection (design Amendment A2, §6.7; `Inputs::advective`, the same on every
// rank): the cycle runs on the ADVECTIVE surrogate S_adv, in band form (AC, AW..AT). Level 0 is
// the operator's 7 bands with SAC as the diagonal (no new storage). A coarse level keeps everything
// above (faces, wall, Dirichlet planes, pins; its mass carries the open-face outflow omega_open,
// not the interior outflow) and adds the summed positive parts of the sub-face fluxes, Qp/Qm per
// low face per unit level volume (the piecewise-constant Galerkin FOU): AC += the cell's outgoing
// Q, AW = AFX - Qp_x, AE = AFX(i+e_x) - Qm_x(i+e_x), ... The sweep (unknown cells of the colour),
// the residual, applySurrogate and the contraction instrument use the bands; the cycle structure,
// transfers, colours and exchanges are unchanged. Transient mode and steady mode at rest take the
// face-form path above, bitwise unchanged.
#ifndef PECLET_FLOW_SCALAR_MG_HPP
#define PECLET_FLOW_SCALAR_MG_HPP

#include <cmath>
#include <cstdint>
#include <functional>
#include <Kokkos_Core.hpp>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "mac_cutcell_mg.hpp"  // coarsenOpenAvgCell, restrictAvg, prolongAdd, CutcellMG::mgChooseRatio
#include "mac_pressure.hpp"  // cutcellSmoothColorFace, cutcellResidualFaceCell, cutcellApplyFaceCell
#include "policy.hpp"
#include "scalar_cutcell_geometry.hpp"  // ScalarFacetOverlay

namespace peclet::flow {

namespace smg {

/// Level-0 guarded face product Lam a (no metric weight): the face between cell i - st and cell i
/// carries Lam a only when both are unknowns (sco::buildBands' guard). A read-only view-like
/// functor, so `coarsenOpenAvgCell` coarsens it without a materialized level-0 field.
struct GuardedProduct {
  CCConst ap, unk;
  double lam = 0.0;
  long st = 1;
  KOKKOS_INLINE_FUNCTION double operator()(long i) const {
    return (unk(i) > 0.5 && unk(i - st) > 0.5) ? lam * ap(i) : 0.0;
  }
};

/// Level-0 mass m = kappa idt on unknowns, 0 elsewhere (read by `restrictAvgCell`). Under
/// advection (WO-5) the lumped implicit outflow of §4.3 rides on it: m = kappa idt + omega, so its
/// `restrictAvg` IS §5.2's "lumped outflow: restrictAvg of the level-0 per-cell field" (both are
/// plain child averages). `hasOut` false is the WO-4 expression verbatim.
struct MassField {
  CCConst kappa, unk, out;
  double idt = 0.0;
  bool hasOut = false;
  KOKKOS_INLINE_FUNCTION double operator()(long i) const {
    if (!(unk(i) > 0.5))
      return 0.0;
    return hasOut ? kappa(i) * idt + out(i) : kappa(i) * idt;
  }
};

/// coarse flag = max of the children's flags (1 unknown, 0 pinned), over the coarse inner cells.
template <class FV>
inline void restrictMax(CCField coarse, const FV& fine, C3 cext, C3 fext, int gc, int gf, C3 cinner,
                        C3 ratio) {
  ccFor3(
      "peclet::flow::smg_restrict_max", C3{0, 0, 0}, cinner,
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        const long fsy = fext.x, fsz = (long)fext.x * fext.y;
        double m = 0.0;
        for (int dz = 0; dz < ratio.z; ++dz)
          for (int dy = 0; dy < ratio.y; ++dy)
            for (int dx = 0; dx < ratio.x; ++dx) {
              const long fi = (long)(ratio.x * icx + dx + gf) +
                              (long)(ratio.y * icy + dy + gf) * fsy +
                              (long)(ratio.z * icz + dz + gf) * fsz;
              m = fine(fi) > m ? fine(fi) : m;
            }
        coarse((long)(icx + gc) + (long)(icy + gc) * cext.x +
               (long)(icz + gc) * (long)cext.x * cext.y) = m > 0.5 ? 1.0 : 0.0;
      });
}

/// x(i) = 0 on the inner cells that are not unknowns (re-zero the pinned cells).
inline void zeroPinned(CCField x, CCConst unk, C3 e, int g) {
  ccFor3(
      "peclet::flow::smg_zero_pinned", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (!(unk(i) > 0.5))
          x(i) = 0.0;
      });
}

/// f = 0 on the planes [lo, hi) of axis a (full transverse extent).
inline void zeroPlanes(CCField f, C3 e, int a, int lo, int hi) {
  if (hi <= lo)
    return;
  const int ext[3] = {e.x, e.y, e.z};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const long sa = st[a], sb = st[b], sc = st[c];
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::smg_zero_planes", MDRange2<CCExec>(space, {0, 0}, {ext[b], ext[c]}),
      KOKKOS_LAMBDA(int pb, int pc) {
        for (int k = lo; k < hi; ++k)
          f((long)k * sa + (long)pb * sb + (long)pc * sc) = 0.0;
      });
  space.fence();
}

/// Single-rank periodic wrap of the g ghost layers of axis `a` (VelocityMG::fillAxis).
inline void wrapAxis(CCField f, C3 e, C3 inner, int g, int a) {
  const int dims[3] = {e.x, e.y, e.z};
  const int N3[3] = {inner.x, inner.y, inner.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int N = N3[a];
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::smg_wrap", MDRange2<CCExec>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < g; ++gl) {
          f(base + (long)gl * sa) = f(base + (long)(gl + N) * sa);
          f(base + (long)(g + N + gl) * sa) = f(base + (long)(g + gl) * sa);
        }
      });
  space.fence();
}

/// r = b - A x (face form, double), inner cells.
inline void residualFace(CCField r, CCConst x, CCConst b, CCField AC, CCField AFX, CCField AFY,
                         CCField AFZ, C3 e, int g) {
  ccFor3(
      "peclet::flow::smg_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellResidualFaceCell(r, x, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                i - sy, i + sz, i - sz);
      });
}

// ---- the advective path (design Amendment A2, §6.7): band form ----------------------------------

/// Level-0 guarded positive part of the signed low-face flux: P(i) = max(sgn phi_a(i), 0) when
/// cells i - st and i are both fluid unknowns (advectionBands' implicit-coupling predicate in
/// steady mode), else 0. sgn = +1: the flow toward +a (Qp); -1: toward -a (Qm). Read by
/// `coarsenOpenAvgCell` (the GuardedProduct pattern).
struct GuardedPart {
  CCConst phi, unk;
  long st = 1;
  double sgn = 1.0;
  KOKKOS_INLINE_FUNCTION double operator()(long i) const {
    return (unk(i) > 0.5 && unk(i - st) > 0.5) ? Kokkos::fmax(sgn * phi(i), 0.0) : 0.0;
  }
};

/// The band-form off-diagonal sum of A2, in its fixed order:
/// ((AW x_W + AE x_E) + (AS x_S + AN x_N)) + (AB x_B + AT x_T).
template <class XV, class BV>
KOKKOS_INLINE_FUNCTION double bandOff(const XV& x, const BV& AW, const BV& AE, const BV& AS,
                                      const BV& AN, const BV& AB, const BV& AT, long i, long sx,
                                      long sy, long sz) {
  return ((AW(i) * x(i - sx) + AE(i) * x(i + sx)) + (AS(i) * x(i - sy) + AN(i) * x(i + sy))) +
         (AB(i) * x(i - sz) + AT(i) * x(i + sz));
}

/// One colour of the band-form Gauss-Seidel (A2), on the unknown inner cells of the colour:
/// x_i <- (rhs_i - bandOff) / AC_i. Colour = (og + l) parity, as cutcellSmoothColorFace (same
/// launch forms: host pencils, device MDRange).
template <class BV, class UV>
inline void sweepColorBand(CCField x, CCConst rhs, BV AC, BV AW, BV AE, BV AS, BV AN, BV AB, BV AT,
                           UV unk, C3 e, C3 og, int g, int color) {
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    const long cells = (long)nyi * nzi * (e.x - 2 * g);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      for (int lx = g + ((P ^ (g & 1)) & 1); lx < e.x - g; lx += 2) {
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        if (!(unk(i) > 0.5))
          continue;
        x(i) = (rhs(i) - bandOff(x, AW, AE, AS, AN, AB, AT, i, sx, sy, sz)) / AC(i);
      }
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::smg_smooth_band",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::smg_smooth_band", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        if (!(unk(i) > 0.5))
          return;
        x(i) = (rhs(i) - bandOff(x, AW, AE, AS, AN, AB, AT, i, sx, sy, sz)) / AC(i);
      });
}

/// r = b - (AC x + bandOff) over the inner cells (A2's band-form residual).
inline void residualBand(CCField r, CCConst x, CCConst b, CCField AC, CCField AW, CCField AE,
                         CCField AS, CCField AN, CCField AB, CCField AT, C3 e, int g) {
  ccFor3(
      "peclet::flow::smg_residual_band", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        r(i) = b(i) - (AC(i) * x(i) + bandOff(x, AW, AE, AS, AN, AB, AT, i, sx, sy, sz));
      });
}

}  // namespace smg

class ScalarMG {
 public:
  static constexpr int G0 = 2;  ///< level 0: the scalar's own block
  static constexpr int GC = 1;  ///< coarse levels
  using Fill = std::function<void(CCField)>;

  struct Level {
    C3 ext{0, 0, 0}, inner{0, 0, 0}, ratio{1, 1, 1}, cfac{1, 1, 1};
    C3 og{0, 0, 0};    ///< block inner origin (global red-black parity); {0,0,0} single-rank
    C3 gdim{0, 0, 0};  ///< GLOBAL dims of this level
    int g = GC;
    std::size_t n = 0;
    CCField x, rhs, res;        ///< level 0: x / rhs are the caller's (z, r); res owned
    CCField AC, AFX, AFY, AFZ;  ///< the surrogate, face form (level 0: AC = the caller's SAC)
    CCField unk;                ///< 1 unknown / 0 pinned (level 0: the geometry's flag)
    CCField mass, px, py, pz;   ///< coarse levels: m and the face products <Lam a>
    Kokkos::View<double*, CCMem> plane[6];  ///< Dirichlet domain faces: <Lam a_bf> (owning rank)
    double nUnk = 0.0;                      ///< global count of unknown cells (the singular mean)
    /// Advective path (A2): the 7 bands (level 0: the operator's AW..AT, AC = SAC; coarse: owned,
    /// allocated on the first advective build) and, on coarse levels, the summed positive parts
    /// of the sub-face fluxes on the LOW a-face, per unit level volume: qp[a] toward +a (from
    /// i - e_a into i), qm[a] toward -a.
    CCField AW, AE, AS, AN, AB, AT;
    CCField qp[3], qm[3];
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;
    std::shared_ptr<GridHalo<double>> dev;
#endif
  };

  /// Per-axis metric (VelocityMG::setMetric): w_a = 1/h_a'^2 and h_a'. Call before init.
  void setMetric(const double w[3], const double hp[3]) {
    for (int a = 0; a < 3; ++a) {
      w_[a] = w[a];
      hp_[a] = hp[a];
    }
    aniso_ = !(hp_[0] == 1.0 && hp_[1] == 1.0 && hp_[2] == 1.0);
  }
  /// Periodicity per axis (the flow's domain faces); a non-periodic axis' ghosts are pinned.
  void setPeriodic(const bool per[3]) {
    for (int a = 0; a < 3; ++a)
      per_[a] = per[a];
  }

  /// Single-rank level table (VelocityMG::init's rule, no depth cap).
  void init(int nx, int ny, int nz, Fill fill0) {
    lv_.clear();
    distributed_ = false;
    fill0_ = std::move(fill0);
    C3 inner{nx, ny, nz}, cf{1, 1, 1};
    for (int L = 0; L < kMaxLevels; ++L) {
      Level v;
      v.g = L == 0 ? G0 : GC;
      v.inner = inner;
      v.gdim = inner;
      v.ext = C3{inner.x + 2 * v.g, inner.y + 2 * v.g, inner.z + 2 * v.g};
      v.cfac = cf;
      v.n = (std::size_t)v.ext.x * v.ext.y * v.ext.z;
      const bool canA[3] = {can(inner.x), can(inner.y), can(inner.z)};
      const C3 ratio = chooseRatio(canA, cf);
      const C3 next{ratio.x == 2 ? inner.x / 2 : inner.x, ratio.y == 2 ? inner.y / 2 : inner.y,
                    ratio.z == 2 ? inner.z / 2 : inner.z};
      v.ratio = ratio;
      allocate(v, L);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z)
        break;
      inner = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
  }
#ifdef PECLET_FLOW_MPI
  /// Distributed level table: level 0 on the solver's decomposition `dec0` (inner origin `og0`),
  /// coarse levels that decomposition coarsened IN PLACE with the even-block gate
  /// (VelocityMG::initMpi(dec0, ., comm, inPlace = true)'s rule, no depth cap).
  void initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec0, MPI_Comm comm, C3 og0,
               Fill fill0) {
    lv_.clear();
    distributed_ = true;
    comm_ = comm;
    fill0_ = std::move(fill0);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::array<bool, 3> per{true, true, true};  // periodic everywhere; non-periodic ghosts zeroed
    const auto& g0 = dec0.globalSize();
    C3 gs{(int)g0[0], (int)g0[1], (int)g0[2]}, cf{1, 1, 1};
    auto evenOn = [](const peclet::core::decomp::BlockDecomposer<3>& d, int ax) {
      for (std::size_t b = 0; b < d.numBlocks(); ++b) {
        const auto blk = d.block(b);
        if (blk.origin[ax] % 2 != 0 || blk.size[ax] % 2 != 0)
          return false;
      }
      return true;
    };
    peclet::core::decomp::BlockDecomposer<3> dec = dec0;
    for (int L = 0; L < kMaxLevels; ++L) {
      Level v;
      v.g = L == 0 ? G0 : GC;
      if (L == 0) {
        const auto blk = dec.block((std::size_t)rank);
        v.inner = C3{(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
        v.ext = C3{v.inner.x + 2 * G0, v.inner.y + 2 * G0, v.inner.z + 2 * G0};
        v.og = og0;
      } else {
        v.halo = std::make_shared<GridHaloTopology<3>>();
        v.halo->buildTopology(dec, rank, v.g, per, comm);
        v.dev = std::make_shared<GridHalo<double>>();
        v.dev->init(*v.halo);
        v.dev->setLabel("smg L" + std::to_string(L));
        const auto& idx = v.halo->indexer();
        const auto eg = idx.sizeInclGhost(), ino = idx.sizeInner(), oig = idx.originInclGhost();
        v.ext = {(int)eg[0], (int)eg[1], (int)eg[2]};
        v.inner = {(int)ino[0], (int)ino[1], (int)ino[2]};
        v.og = {(int)oig[0] + v.g, (int)oig[1] + v.g, (int)oig[2] + v.g};
      }
      v.gdim = gs;
      v.cfac = cf;
      v.n = (std::size_t)v.ext.x * v.ext.y * v.ext.z;
      const bool canA[3] = {can(gs.x) && evenOn(dec, 0), can(gs.y) && evenOn(dec, 1),
                            can(gs.z) && evenOn(dec, 2)};
      const C3 ratio = chooseRatio(canA, cf);
      const C3 next{ratio.x == 2 ? gs.x / 2 : gs.x, ratio.y == 2 ? gs.y / 2 : gs.y,
                    ratio.z == 2 ? gs.z / 2 : gs.z};
      v.ratio = ratio;
      allocate(v, L);
      lv_.push_back(v);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z)
        break;
      gs = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
      dec = dec.coarsened(peclet::core::IVec<3>{ratio.x, ratio.y, ratio.z});
    }
  }
#endif

  int levels() const { return (int)lv_.size(); }
  /// Levels the last `build` set up for the V-cycle (1: level 0 alone, the transient rule).
  int levelsUsed() const { return nUse_; }
  const Level& level(int L) const { return lv_[L]; }
  /// The per-level coarsening ratios (the level table; equal to VelocityMG's on the same grid).
  std::vector<C3> levelRatios() const {
    std::vector<C3> r;
    for (const auto& v : lv_)
      r.push_back(v.ratio);
    return r;
  }

  /// Everything `build` reads from the scalar operator and the geometry record.
  struct Inputs {
    double lam = 0.0;       ///< internal Lam
    double idt = 0.0;       ///< 1/dt' (0 steady)
    bool fullTable = true;  ///< the level rule (false: level 0 alone)
    bool singular = false;  ///< steady pure-Neumann: the mean is removed on every level
    CCField SAC;            ///< level-0 surrogate diagonal (§4.3)
    CCConst kappa, unknown, sax, say, saz;
    const scg::ScalarFacetOverlay* fac = nullptr;
    Kokkos::View<const double*, CCMem> facetW;  ///< level-0 wall term per facet (cw = alpha G)
    bool dirFace[6] = {false, false, false, false, false, false};
    /// WO-5: the lumped implicit-FOU outflow per level-0 cell (§4.3; already inside SAC),
    /// restricted with the mass onto the coarse levels (§5.2). Unset (`hasOutflow` false) without
    /// advection.
    CCConst outflow;
    bool hasOutflow = false;
    /// A2 (§6.7): steady with advection -> the V-cycle runs on the ADVECTIVE surrogate S_adv on
    /// every level (band form). Level 0 = the operator's 7 bands with SAC as the diagonal; the
    /// coarse levels add the summed positive parts of the sub-face fluxes `phi`; the mass carries
    /// the open-face outflow `omegaOpen` (idt = 0) in place of `outflow`. The flag must be the same
    /// on every rank (steady && the all-rank advecting flag).
    bool advective = false;
    CCField AW, AE, AS, AN, AB, AT;  ///< the operator's level-0 bands
    CCConst phi[3];                  ///< F/V through the low a-face (st.phi, guarded)
    CCConst omegaOpen;               ///< the open faces' implicit outflow per level-0 cell
  };

  /// Rebuild the surrogate on every level from the current operator (the level table is kept).
  void build(const Inputs& in) {
    singular_ = in.singular;
    adv_ = in.advective;
    nUse_ = in.fullTable ? (int)lv_.size() : 1;
    Level& l0 = lv_[0];
    l0.AC = in.SAC;
    if (adv_) {  // A2: the operator's bands, no new storage
      l0.AW = in.AW;
      l0.AE = in.AE;
      l0.AS = in.AS;
      l0.AN = in.AN;
      l0.AB = in.AB;
      l0.AT = in.AT;
    }
    l0.unk = CCField();  // level 0's flag is the geometry's (const) view
    unk0_ = in.unknown;
    buildLevel0Faces(in);
    l0.nUnk = countUnknown(0);
    for (int f = 0; f < 6; ++f)
      dirFace_[f] = in.dirFace[f];
    if (nUse_ > 1) {
      buildCutIndex(in);
      for (int f = 0; f < 6; ++f)
        if (dirFace_[f] && touches(l0, f))
          buildPlane0(in, f);
      for (int L = 1; L < nUse_; ++L)
        buildCoarse(in, L);
    }
  }

  /// z = M^-1 r: one V-cycle on the surrogate (level 0 alone: 2 + 2 sweeps) from z = 0. `r` must
  /// be 0 on the cells that are not unknowns (the Krylov's identity rows guarantee it); in the
  /// singular case it must already be mean-free (the caller projects, and removes z's mean after).
  void apply(CCField z, CCField r) {
    Level& l0 = lv_[0];
    l0.x = z;
    l0.rhs = r;
    Kokkos::deep_copy(CCExec(), z, 0.0);
    if (nUse_ == 1) {
      for (int k = 0; k < 2; ++k) {
        sweepColor(0, 0);
        sweepColor(0, 1);
      }
      for (int k = 0; k < 2; ++k) {
        sweepColor(0, 1);
        sweepColor(0, 0);
      }
      return;
    }
    vcycle(0);
  }

  /// y = S x on level 0 (exchanges x first). For the contraction instrument and tests.
  void applySurrogate(CCField y, CCField x) {
    Level& l0 = lv_[0];
    fill(0, x);
    const C3 e = l0.ext;
    const int g = l0.g;
    if (adv_) {  // A2: S_adv at level 0 = the operator's band matvec with SAC as the diagonal
      applyCutcellOp(y, CCConst(x), l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB, l0.AT, e, g);
      return;
    }
    CCField AC = l0.AC, AFX = l0.AFX, AFY = l0.AFY, AFZ = l0.AFZ;
    CCConst xx = x;
    ccFor3(
        "peclet::flow::smg_apply", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)lx + (long)ly * sy + (long)lz * sz;
          y(i) = cutcellApplyFaceCell(xx, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                      i - sy, i + sz, i - sz);
        });
  }

  /// Power estimate of the V-cycle's error contraction rho(I - M^-1 S) on the current surrogate
  /// (§11 G-iter): e <- e - M^-1 S e from a deterministic pseudo-random start on the unknowns
  /// (a function of the GLOBAL index, so decomposition-independent), mean-free when singular.
  /// Returns the per-iteration ratios ||e_k|| / ||e_{k-1}|| (2-norm over the unknowns).
  std::vector<double> contraction(int iters, CCField e, CCField t, CCField z) {
    Level& l0 = lv_[0];
    const C3 ext = l0.ext, og = l0.og;
    const int g = l0.g;
    CCConst unk = unk0_;
    ccFor3(
        "peclet::flow::smg_seed", C3{g, g, g}, C3{ext.x - g, ext.y - g, ext.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
          std::uint64_t h = (std::uint64_t)(og.x + lx - g) * 0x9E3779B97F4A7C15ull ^
                            (std::uint64_t)(og.y + ly - g) * 0xC2B2AE3D27D4EB4Full ^
                            (std::uint64_t)(og.z + lz - g) * 0x165667B19E3779F9ull;
          h ^= h >> 29;
          h *= 0xBF58476D1CE4E5B9ull;
          h ^= h >> 32;
          e(i) = unk(i) > 0.5 ? (double)(h >> 11) * (1.0 / 9007199254740992.0) - 0.5 : 0.0;
        });
    if (singular_)
      removeMean(0, e);
    std::vector<double> ratios;
    double prev = norm2(e);
    for (int k = 0; k < iters; ++k) {
      applySurrogate(t, e);
      apply(z, t);
      if (singular_)
        removeMean(0, z);
      ccFor3(
          "peclet::flow::smg_err_update", C3{g, g, g}, C3{ext.x - g, ext.y - g, ext.z - g},
          KOKKOS_LAMBDA(int lx, int ly, int lz) {
            const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
            e(i) -= z(i);
          });
      if (singular_)  // keep the iterate off the null space (round-off would otherwise grow there)
        removeMean(0, e);
      const double cur = norm2(e);
      ratios.push_back(prev > 0.0 ? cur / prev : 0.0);
      if (!(cur > 0.0))
        break;
      // renormalize (keeps the iterate O(1) over many iterations)
      const double s = 1.0 / cur;
      ccFor3(
          "peclet::flow::smg_err_scale", C3{g, g, g}, C3{ext.x - g, ext.y - g, ext.z - g},
          KOKKOS_LAMBDA(int lx, int ly, int lz) {
            const long i = (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
            e(i) *= s;
          });
      prev = 1.0;
    }
    return ratios;
  }

  // ---- (public for nvcc extended lambdas) ----
  void vcycle(int L) {
    Level& lv = lv_[L];
    if (L + 1 == nUse_) {  // bottom: 16 sweeps, symmetric order
      for (int k = 0; k < kBottom / 2; ++k)
        sweep(L, true);
      for (int k = 0; k < kBottom / 2; ++k)
        sweep(L, false);
      if (singular_)
        removeMean(L, lv.x);
      return;
    }
    for (int k = 0; k < kPre; ++k)
      sweep(L, true);
    fill(L, lv.x);  // the smoother leaves the ghosts one colour stale
    if (adv_)
      smg::residualBand(lv.res, CCConst(lv.x), CCConst(lv.rhs), lv.AC, lv.AW, lv.AE, lv.AS, lv.AN,
                        lv.AB, lv.AT, lv.ext, lv.g);
    else
      smg::residualFace(lv.res, CCConst(lv.x), CCConst(lv.rhs), lv.AC, lv.AFX, lv.AFY, lv.AFZ,
                        lv.ext, lv.g);
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, cs.g, lv.g, cs.inner, lv.ratio);
    if (singular_)
      removeMean(L + 1, cs.rhs);
    Kokkos::deep_copy(CCExec(), cs.x, 0.0);
    vcycle(L + 1);
    fill(L + 1, cs.x);  // coarse ghosts: exchanged, 0 beyond a non-periodic global face
    prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, lv.g, cs.g, lv.inner, lv.ratio);
    smg::zeroPinned(lv.x, unkOf(L), lv.ext, lv.g);
    for (int k = 0; k < kPost; ++k)
      sweep(L, false);
  }

  /// One colour of red-black Gauss-Seidel on level L: `rb` = 0 red (global (gx+gy+gz) even), 1
  /// black. The kernel's colour counts the ghost offset (lx = gx + g on every axis), hence + 3g.
  void sweepColor(int L, int rb) {
    Level& lv = lv_[L];
    fill(L, lv.x);
    const int kc = (rb + 3 * lv.g) & 1;
    if (adv_) {
      smg::sweepColorBand(lv.x, CCConst(lv.rhs), lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB, lv.AT,
                          unkOf(L), lv.ext, lv.og, lv.g, kc);
      return;
    }
    cutcellSmoothColorFace(lv.x, CCConst(lv.rhs), lv.AC, lv.AFX, lv.AFY, lv.AFZ, lv.ext, lv.og,
                           lv.g, kc);
  }

  /// One full red-black sweep: R -> B forward, B -> R backward.
  void sweep(int L, bool fwd) {
    sweepColor(L, fwd ? 0 : 1);
    sweepColor(L, fwd ? 1 : 0);
  }

  /// Ghost exchange of a level-L field: level 0 the solver's own; coarse levels the per-level halo
  /// (or the single-rank periodic wrap), then 0 on the ghosts beyond a non-periodic global face.
  void fill(int L, CCField f) {
    if (L == 0) {
      fill0_(f);
      return;
    }
    Level& lv = lv_[L];
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      lv.dev->exchange(f);
    else
#endif
      for (int a = 0; a < 3; ++a)
        smg::wrapAxis(f, lv.ext, lv.inner, lv.g, a);
    zeroNonPeriodicGhosts(lv, f);
  }

  /// Does this rank's block on level `lv` touch global face f (0..5 = -x,+x,-y,+y,-z,+z)?
  bool touches(const Level& lv, int f) const {
    const int a = f / 2;
    const int o = (a == 0) ? lv.og.x : (a == 1) ? lv.og.y : lv.og.z;
    const int n = (a == 0) ? lv.inner.x : (a == 1) ? lv.inner.y : lv.inner.z;
    const int gn = (a == 0) ? lv.gdim.x : (a == 1) ? lv.gdim.y : lv.gdim.z;
    return (f % 2 == 0) ? (o == 0) : (o + n == gn);
  }

 private:
  static constexpr int kMaxLevels = 64;  ///< no depth cap in practice (the full table)
  static constexpr int kPre = 2, kPost = 2, kBottom = 16;

  static bool can(int d) { return (d % 2 == 0) && (d / 2 >= 2); }
  C3 chooseRatio(const bool canA[3], C3 cf) const {
    const double H[3] = {hp_[0] * (double)cf.x, hp_[1] * (double)cf.y, hp_[2] * (double)cf.z};
    return CutcellMG::mgChooseRatio(H, canA, aniso_, kAspectTheta);
  }
  static constexpr double kAspectTheta = 2.0;

  void allocate(Level& v, int L) {
    if (L == 0) {
      v.res = CCField("smg_res0", v.n);
      v.AFX = CCField("smg_afx0", v.n);
      v.AFY = CCField("smg_afy0", v.n);
      v.AFZ = CCField("smg_afz0", v.n);
      return;
    }
    v.x = CCField("smg_x", v.n);
    v.rhs = CCField("smg_rhs", v.n);
    v.res = CCField("smg_res", v.n);
    v.AC = CCField("smg_ac", v.n);
    v.AFX = CCField("smg_afx", v.n);
    v.AFY = CCField("smg_afy", v.n);
    v.AFZ = CCField("smg_afz", v.n);
    v.unk = CCField("smg_unk", v.n);
    v.mass = CCField("smg_mass", v.n);
    v.px = CCField("smg_px", v.n);
    v.py = CCField("smg_py", v.n);
    v.pz = CCField("smg_pz", v.n);
  }

  CCConst unkOf(int L) const { return L == 0 ? unk0_ : CCConst(lv_[L].unk); }

  void zeroNonPeriodicGhosts(Level& lv, CCField f) {
    const int ext[3] = {lv.ext.x, lv.ext.y, lv.ext.z};
    for (int a = 0; a < 3; ++a) {
      if (per_[a])
        continue;
      if (touches(lv, 2 * a))
        smg::zeroPlanes(f, lv.ext, a, 0, lv.g);
      if (touches(lv, 2 * a + 1))
        smg::zeroPlanes(f, lv.ext, a, ext[a] - lv.g, ext[a]);
    }
  }

  // ---- reductions ----
  void allSum(double* v, int k) const {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::vector<double> o((std::size_t)k);
      MPI_Allreduce(v, o.data(), k, MPI_DOUBLE, MPI_SUM, comm_);
      for (int j = 0; j < k; ++j)
        v[j] = o[(std::size_t)j];
    }
#endif
    (void)v;
    (void)k;
  }
  double countUnknown(int L) const {
    const Level& lv = lv_[L];
    const C3 e = lv.ext;
    const int g = lv.g;
    CCConst unk = unkOf(L);
    double s = 0.0;
    ccReduce3(
        "peclet::flow::smg_count", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz, double& acc) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          acc += unk(i) > 0.5 ? 1.0 : 0.0;
        },
        Kokkos::Sum<double>(s));
    allSum(&s, 1);
    return s;
  }
  /// f -= its mean over the unknown cells of level L (CutcellMG::removeMean's form, unweighted).
  void removeMean(int L, CCField f) {
    const Level& lv = lv_[L];
    if (!(lv.nUnk > 0.0))
      return;
    const C3 e = lv.ext;
    const int g = lv.g;
    CCConst unk = unkOf(L);
    double s = 0.0;
    ccReduce3(
        "peclet::flow::smg_mean", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz, double& acc) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          if (unk(i) > 0.5)
            acc += f(i);
        },
        Kokkos::Sum<double>(s));
    allSum(&s, 1);
    const double mean = s / lv.nUnk;
    ccFor3(
        "peclet::flow::smg_mean_sub", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          if (unk(i) > 0.5)
            f(i) -= mean;
        });
  }
  double norm2(CCField f) const {
    const Level& lv = lv_[0];
    const C3 e = lv.ext;
    const int g = lv.g;
    double s = 0.0;
    ccReduce3(
        "peclet::flow::smg_norm", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz, double& acc) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          acc += f(i) * f(i);
        },
        Kokkos::Sum<double>(s));
    allSum(&s, 1);
    return std::sqrt(s);
  }

  // ---- build ----
  /// Level-0 faces: AF_a(i) = -(Lam w_a a_a(i)) when cells i - st_a and i are both unknowns —
  /// the same expression as sco::buildBands' AW/AS/AB, so the face-form sweep is bitwise the band
  /// sweep. Written over the full extent (index 0 of each axis: 0).
  void buildLevel0Faces(const Inputs& in) {
    Level& l0 = lv_[0];
    const C3 e = l0.ext;
    const double lx = in.lam * w_[0], ly = in.lam * w_[1], lz = in.lam * w_[2];
    CCConst unk = in.unknown, sax = in.sax, say = in.say, saz = in.saz;
    CCField AFX = l0.AFX, AFY = l0.AFY, AFZ = l0.AFZ;
    ccFor3(
        "peclet::flow::smg_faces0", C3{0, 0, 0}, e, KOKKOS_LAMBDA(int x, int y, int z) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)x + (long)y * sy + (long)z * sz;
          const bool u = unk(i) > 0.5;
          const double tw = (x > 0 && u && unk(i - sx) > 0.5) ? lx * sax(i) : 0.0;
          const double ts = (y > 0 && u && unk(i - sy) > 0.5) ? ly * say(i) : 0.0;
          const double tb = (z > 0 && u && unk(i - sz) > 0.5) ? lz * saz(i) : 0.0;
          AFX(i) = -tw;
          AFY(i) = -ts;
          AFZ(i) = -tb;
        });
  }

  /// cut0_(i) = the CSR row of level-0 cell i in the facet overlay, or -1.
  void buildCutIndex(const Inputs& in) {
    const Level& l0 = lv_[0];
    if (cut0_.extent(0) != l0.n)
      cut0_ = Kokkos::View<int*, CCMem>("smg_cut0", l0.n);
    Kokkos::deep_copy(CCExec(), cut0_, -1);
    auto cut0 = cut0_;
    auto cutCell = in.fac->cutCell;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::smg_cut_index", Kokkos::RangePolicy<CCExec>(space, 0, in.fac->nCut),
        KOKKOS_LAMBDA(const long c) { cut0(cutCell(c)) = (int)c; });
    space.fence();
  }

  /// Level-0 Dirichlet boundary plane of face f: <Lam a_bf> = Lam a_bf on unknown boundary cells
  /// (sco::dirichletFaceFold's aperture), indexed (j1, j2) over the inner tangential cells.
  void buildPlane0(const Inputs& in, int f) {
    Level& l0 = lv_[0];
    const int a = f / 2, side = f % 2;
    int t1, t2;
    tangents(a, t1, t2);
    const int ext[3] = {l0.ext.x, l0.ext.y, l0.ext.z};
    const long st[3] = {1, (long)l0.ext.x, (long)l0.ext.x * l0.ext.y};
    const int g = l0.g;
    const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
    const long sa = st[a], s1 = st[t1], s2 = st[t2];
    const int aInner = side == 0 ? g : ext[a] - g - 1;
    if (l0.plane[f].extent(0) != (std::size_t)n1 * n2)
      l0.plane[f] = Kokkos::View<double*, CCMem>("smg_plane0", (long)n1 * n2);
    auto pl = l0.plane[f];
    CCConst unk = in.unknown;
    CCConst saA = a == 0 ? in.sax : (a == 1 ? in.say : in.saz);
    const double lam = in.lam;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::smg_plane0", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
          const double abf = side == 0 ? saA(i) : saA(i + sa);
          pl((long)j1 + (long)j2 * n1) = unk(i) > 0.5 ? lam * abf : 0.0;
        });
    space.fence();
  }

  static void tangents(int a, int& t1, int& t2) {
    t1 = (a == 0) ? 1 : 0;
    t2 = (a == 2) ? 1 : 2;
  }

  void buildCoarse(const Inputs& in, int L) {
    Level& c = lv_[L];
    Level& fin = lv_[L - 1];
    const C3 ratio = fin.ratio;
    // pins: max of the children's unknown flags
    if (L == 1)
      smg::restrictMax(c.unk, in.unknown, c.ext, fin.ext, c.g, fin.g, c.inner, ratio);
    else
      smg::restrictMax(c.unk, CCConst(fin.unk), c.ext, fin.ext, c.g, fin.g, c.inner, ratio);
    c.nUnk = countUnknown(L);
    // face products <Lam a>: coarsenOpenAvg's cell body on the fine products
    {
      CCField px = c.px, py = c.py, pz = c.pz;
      const C3 ce = c.ext, fe = fin.ext;
      const int gc = c.g, gf = fin.g;
      if (L == 1) {
        const long sy0 = fe.x, sz0 = (long)fe.x * fe.y;
        const smg::GuardedProduct fx{in.sax, in.unknown, in.lam, 1};
        const smg::GuardedProduct fy{in.say, in.unknown, in.lam, sy0};
        const smg::GuardedProduct fz{in.saz, in.unknown, in.lam, sz0};
        ccFor3(
            "peclet::flow::smg_coarsen_faces", C3{0, 0, 0}, c.inner,
            KOKKOS_LAMBDA(int icx, int icy, int icz) {
              coarsenOpenAvgCell(px, py, pz, fx, fy, fz, ce, fe, gc, gf, ratio, icx, icy, icz);
            });
      } else {
        CCConst fx = fin.px, fy = fin.py, fz = fin.pz;
        ccFor3(
            "peclet::flow::smg_coarsen_faces", C3{0, 0, 0}, c.inner,
            KOKKOS_LAMBDA(int icx, int icy, int icz) {
              coarsenOpenAvgCell(px, py, pz, fx, fy, fz, ce, fe, gc, gf, ratio, icx, icy, icz);
            });
      }
      fill(L, c.px);
      fill(L, c.py);
      fill(L, c.pz);
      // the low domain face of a non-periodic axis: no coupling toward the pinned ghost
      const CCField P[3] = {c.px, c.py, c.pz};
      for (int a = 0; a < 3; ++a)
        if (!per_[a] && touches(c, 2 * a))
          smg::zeroPlanes(P[a], c.ext, a, 0, c.g + 1);
    }
    // mass
    if (L == 1) {
      // A2: on the advective path the mass carries the open-face outflow omega_open (the interior
      // outflow is Q's); otherwise WO-5's lumped outflow, or nothing (WO-4) — those two verbatim
      const bool hasOut = in.advective || in.hasOutflow;
      const CCConst out = in.advective ? in.omegaOpen : (in.hasOutflow ? in.outflow : in.kappa);
      const smg::MassField m0{in.kappa, in.unknown, out, in.idt, hasOut};
      CCField cm = c.mass;
      const C3 ce = c.ext, fe = fin.ext, ci = c.inner;
      const int gc = c.g, gf = fin.g;
      ccFor3(
          "peclet::flow::smg_restrict_mass", C3{0, 0, 0}, ci,
          KOKKOS_LAMBDA(int icx, int icy, int icz) {
            restrictAvgCell(cm, m0, ce, fe, gc, gf, ratio, icx, icy, icz);
          });
    } else {
      restrictAvg(c.mass, CCConst(fin.mass), c.ext, fin.ext, c.g, fin.g, c.inner, ratio);
    }
    // Dirichlet boundary planes, averaged from the finer level's
    for (int f = 0; f < 6; ++f)
      if (dirFace_[f] && touches(c, f))
        coarsenPlane(L, f);
    // the operator
    assembleCoarse(in, L);
    if (adv_) {  // A2: the coarse advection, then the 7 bands
      coarsenAdvection(in, L);
      assembleBands(L);
    }
  }

  /// A2: Qp/Qm of level L from level L - 1 — level 1 from the guarded positive parts of the
  /// level-0 flux, deeper levels from the finer level's Q — by coarsenOpenAvgCell's sum and
  /// division in its fixed order, then one multiplication by 1/r_a (r_a the ratio on the face's
  /// normal axis): Q = sum_sub max(+-phi, 0) V / V_L. Then the ghosts (exchange / wrap, 0 beyond a
  /// non-periodic global face) and the low-face plane of every non-periodic axis zeroed, as the
  /// face products are.
  void coarsenAdvection(const Inputs& in, int L) {
    Level& c = lv_[L];
    Level& fin = lv_[L - 1];
    if (c.qp[0].extent(0) != c.n)
      for (int a = 0; a < 3; ++a) {
        c.qp[a] = CCField("smg_qp", c.n);
        c.qm[a] = CCField("smg_qm", c.n);
      }
    const C3 ratio = fin.ratio, ce = c.ext, fe = fin.ext;
    const int gc = c.g, gf = fin.g;
    const double ir[3] = {1.0 / (double)ratio.x, 1.0 / (double)ratio.y, 1.0 / (double)ratio.z};
    for (int sgn = 0; sgn < 2; ++sgn) {
      CCField qx = sgn == 0 ? c.qp[0] : c.qm[0], qy = sgn == 0 ? c.qp[1] : c.qm[1],
              qz = sgn == 0 ? c.qp[2] : c.qm[2];
      const double irx = ir[0], iry = ir[1], irz = ir[2];
      if (L == 1) {
        const long sy0 = fe.x, sz0 = (long)fe.x * fe.y;
        const double sg = sgn == 0 ? 1.0 : -1.0;
        const smg::GuardedPart fx{in.phi[0], in.unknown, 1, sg};
        const smg::GuardedPart fy{in.phi[1], in.unknown, sy0, sg};
        const smg::GuardedPart fz{in.phi[2], in.unknown, sz0, sg};
        ccFor3(
            "peclet::flow::smg_coarsen_adv", C3{0, 0, 0}, c.inner,
            KOKKOS_LAMBDA(int icx, int icy, int icz) {
              coarsenOpenAvgCell(qx, qy, qz, fx, fy, fz, ce, fe, gc, gf, ratio, icx, icy, icz);
              const long ci =
                  (long)(icx + gc) + (long)(icy + gc) * ce.x + (long)(icz + gc) * (long)ce.x * ce.y;
              qx(ci) *= irx;
              qy(ci) *= iry;
              qz(ci) *= irz;
            });
      } else {
        const CCConst fx = sgn == 0 ? fin.qp[0] : fin.qm[0], fy = sgn == 0 ? fin.qp[1] : fin.qm[1],
                      fz = sgn == 0 ? fin.qp[2] : fin.qm[2];
        ccFor3(
            "peclet::flow::smg_coarsen_adv", C3{0, 0, 0}, c.inner,
            KOKKOS_LAMBDA(int icx, int icy, int icz) {
              coarsenOpenAvgCell(qx, qy, qz, fx, fy, fz, ce, fe, gc, gf, ratio, icx, icy, icz);
              const long ci =
                  (long)(icx + gc) + (long)(icy + gc) * ce.x + (long)(icz + gc) * (long)ce.x * ce.y;
              qx(ci) *= irx;
              qy(ci) *= iry;
              qz(ci) *= irz;
            });
      }
    }
    for (int a = 0; a < 3; ++a) {
      fill(L, c.qp[a]);
      fill(L, c.qm[a]);
    }
    for (int a = 0; a < 3; ++a)
      if (!per_[a] && touches(c, 2 * a)) {
        smg::zeroPlanes(c.qp[a], c.ext, a, 0, c.g + 1);
        smg::zeroPlanes(c.qm[a], c.ext, a, 0, c.g + 1);
      }
  }

  /// A2: the coarse bands from the face form (assembleCoarse) and Q. Unknown inner cells:
  ///   AC += ((Qm_x(i) + Qp_x(i+e_x)) + (Qm_y(i) + Qp_y(i+e_y))) + (Qm_z(i) + Qp_z(i+e_z)),
  ///   AW = AFX(i) - Qp_x(i), AE = AFX(i+e_x) - Qm_x(i+e_x) (y, z alike);
  /// pinned rows stay identity rows (AC = 1, bands 0).
  void assembleBands(int L) {
    Level& c = lv_[L];
    if (c.AW.extent(0) != c.n) {
      c.AW = CCField("smg_aw", c.n);
      c.AE = CCField("smg_ae", c.n);
      c.AS = CCField("smg_as", c.n);
      c.AN = CCField("smg_an", c.n);
      c.AB = CCField("smg_ab", c.n);
      c.AT = CCField("smg_at", c.n);
    }
    const C3 e = c.ext;
    const int g = c.g;
    CCField AC = c.AC, AW = c.AW, AE = c.AE, AS = c.AS, AN = c.AN, AB = c.AB, AT = c.AT;
    CCConst AFX = c.AFX, AFY = c.AFY, AFZ = c.AFZ, unk = c.unk;
    CCConst qpx = c.qp[0], qpy = c.qp[1], qpz = c.qp[2], qmx = c.qm[0], qmy = c.qm[1],
            qmz = c.qm[2];
    Kokkos::deep_copy(CCExec(), AW, 0.0);
    Kokkos::deep_copy(CCExec(), AE, 0.0);
    Kokkos::deep_copy(CCExec(), AS, 0.0);
    Kokkos::deep_copy(CCExec(), AN, 0.0);
    Kokkos::deep_copy(CCExec(), AB, 0.0);
    Kokkos::deep_copy(CCExec(), AT, 0.0);
    ccFor3(
        "peclet::flow::smg_coarse_bands", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)lx + (long)ly * sy + (long)lz * sz;
          if (!(unk(i) > 0.5))
            return;
          AC(i) =
              AC(i) + ((qmx(i) + qpx(i + sx)) + (qmy(i) + qpy(i + sy))) + (qmz(i) + qpz(i + sz));
          AW(i) = AFX(i) - qpx(i);
          AE(i) = AFX(i + sx) - qmx(i + sx);
          AS(i) = AFY(i) - qpy(i);
          AN(i) = AFY(i + sy) - qmy(i + sy);
          AB(i) = AFZ(i) - qpz(i);
          AT(i) = AFZ(i + sz) - qmz(i + sz);
        });
  }

  void coarsenPlane(int L, int f) {
    Level& c = lv_[L];
    Level& fin = lv_[L - 1];
    const int a = f / 2;
    int t1, t2;
    tangents(a, t1, t2);
    const int cin[3] = {c.inner.x, c.inner.y, c.inner.z};
    const int fin3[3] = {fin.inner.x, fin.inner.y, fin.inner.z};
    const int rr[3] = {fin.ratio.x, fin.ratio.y, fin.ratio.z};
    const int n1 = cin[t1], n2 = cin[t2], f1 = fin3[t1];
    const int r1 = rr[t1], r2 = rr[t2];
    if (c.plane[f].extent(0) != (std::size_t)n1 * n2)
      c.plane[f] = Kokkos::View<double*, CCMem>("smg_plane", (long)n1 * n2);
    auto pc = c.plane[f];
    auto pf = fin.plane[f];
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::smg_coarsen_plane", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          double s = 0.0;
          for (int p = 0; p < r1; ++p)
            for (int q = 0; q < r2; ++q)
              s += pf((long)(r1 * j1 + p) + (long)(r2 * j2 + q) * f1);
          pc((long)j1 + (long)j2 * n1) = s / (double)(r1 * r2);
        });
    space.fence();
  }

  /// Coarse face form + diagonal: AF_a = -w_a(L) <Lam a> over the full extent; on unknown inner
  /// cells AC = m + the six face terms + W (the level-0 facet average) + the Dirichlet folds;
  /// pinned cells are identity rows.
  void assembleCoarse(const Inputs& in, int L) {
    Level& c = lv_[L];
    const C3 e = c.ext, ci = c.inner, cf = c.cfac;
    const int g = c.g;
    const double wx = w_[0] / ((double)cf.x * cf.x), wy = w_[1] / ((double)cf.y * cf.y),
                 wz = w_[2] / ((double)cf.z * cf.z);
    CCField AC = c.AC, AFX = c.AFX, AFY = c.AFY, AFZ = c.AFZ;
    CCConst px = c.px, py = c.py, pz = c.pz;
    ccFor3(
        "peclet::flow::smg_coarse_faces", C3{0, 0, 0}, e, KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          AFX(i) = -(wx * px(i));
          AFY(i) = -(wy * py(i));
          AFZ(i) = -(wz * pz(i));
        });
    // the wall term: the plain average of the level-0 wall terms (fine probe distance, A1)
    const Level& l0 = lv_[0];
    const C3 e0 = l0.ext;
    const int g0 = l0.g;
    const double invN = 1.0 / ((double)cf.x * cf.y * cf.z);
    auto cut0 = cut0_;
    CCConst unk0 = in.unknown;
    CCConst unk = c.unk, mass = c.mass;
    auto cfs = in.fac->cellFacetStart;
    auto cw = in.facetW;
    ccFor3(
        "peclet::flow::smg_coarse_diag", C3{0, 0, 0}, ci, KOKKOS_LAMBDA(int icx, int icy, int icz) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)(icx + g) + (long)(icy + g) * sy + (long)(icz + g) * sz;
          if (!(unk(i) > 0.5)) {
            AC(i) = 1.0;
            AFX(i) = AFY(i) = AFZ(i) = 0.0;
            return;
          }
          double W = 0.0;
          const long s0y = e0.x, s0z = (long)e0.x * e0.y;
          for (int dz = 0; dz < cf.z; ++dz)
            for (int dy = 0; dy < cf.y; ++dy)
              for (int dx = 0; dx < cf.x; ++dx) {
                const long i0 = (long)(cf.x * icx + dx + g0) + (long)(cf.y * icy + dy + g0) * s0y +
                                (long)(cf.z * icz + dz + g0) * s0z;
                const int row = cut0(i0);
                if (row < 0 || !(unk0(i0) > 0.5))
                  continue;
                for (int f = cfs(row); f < cfs(row + 1); ++f)
                  W += cw(f);
              }
          W *= invN;
          const double tw = wx * px(i), te = wx * px(i + sx);
          const double ts = wy * py(i), tn = wy * py(i + sy);
          const double tb = wz * pz(i), tt = wz * pz(i + sz);
          AC(i) = (mass(i) + (((tw + te) + (ts + tn)) + (tb + tt))) + W;
        });
    // Dirichlet domain faces: AC += 2 w_a(L) <Lam a_bf> on the unknown boundary cells
    const double wa[3] = {wx, wy, wz};
    for (int f = 0; f < 6; ++f) {
      if (!(dirFace_[f] && touches(c, f)))
        continue;
      const int a = f / 2, side = f % 2;
      int t1, t2;
      tangents(a, t1, t2);
      const int ext[3] = {e.x, e.y, e.z};
      const long st[3] = {1, (long)e.x, (long)e.x * e.y};
      const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
      const long sa = st[a], s1 = st[t1], s2 = st[t2];
      const int aInner = side == 0 ? g : ext[a] - g - 1;
      const double tw2 = 2.0 * wa[a];
      auto pl = c.plane[f];
      CCExec space;
      Kokkos::parallel_for(
          "peclet::flow::smg_coarse_dirichlet", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
          KOKKOS_LAMBDA(int j1, int j2) {
            const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
            if (!(unk(i) > 0.5))
              return;
            AC(i) += tw2 * pl((long)j1 + (long)j2 * n1);
          });
      space.fence();
    }
  }

  std::vector<Level> lv_;
  Fill fill0_;
  CCConst unk0_;
  Kokkos::View<int*, CCMem> cut0_;
  double w_[3] = {1.0, 1.0, 1.0};
  double hp_[3] = {1.0, 1.0, 1.0};
  bool aniso_ = false;
  bool per_[3] = {true, true, true};
  bool dirFace_[6] = {false, false, false, false, false, false};
  bool singular_ = false;
  bool adv_ = false;  ///< A2: the advective (band-form) path of the last build
  int nUse_ = 1;
  bool distributed_ = false;
#ifdef PECLET_FLOW_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_MG_HPP
