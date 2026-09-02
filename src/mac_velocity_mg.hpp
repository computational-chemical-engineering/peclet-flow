/// @file
/// @brief flow — portable (Kokkos) velocity (momentum) multigrid for the IBM diffusion solve: the
/// STAIRCASE coarse operator.
///
/// Single-GPU (periodic) port of the velocity-MG path in CUDA's DistributedPoissonMG
/// (mac_multigrid.cuh): the fine level is the sharp Robust-Scaled IBM stencil As_[c] (so the
/// residual + smoother use the TRUE operator and the fixed point is the exact sharp solution); the
/// coarse levels use the geometry-aware STAIRCASE Helmholtz (volume fraction theta only CLASSIFIES
/// cells: theta>=0.5 fluid / <0.5 solid-pinned, then a plain constant-coefficient Helmholtz at
/// fluid cells). The fine IBM-cell residuals are excluded from coarsening (clean-fluid mask) and no
/// coarse correction is pumped back into the cut-cell band (masked prolong); the fine smoother owns
/// the boundary. See [[velocity-mg-design]].
///
/// The whole hierarchy uses ghost width G=2 (the velocity block's width), so level 0 IS the
/// solver's velocity block: the IBM stencil + RHS + solution need no g=2<->g=1 bridging. Reuses
/// restrictAvg / prolongAdd (mac_cutcell_mg) and ibmRbgsStencilColor (the pin-aware variable-coeff
/// RB-GS smoother == mg_smooth_var_k). Runs on any Kokkos backend.
#ifndef PECLET_FLOW_MAC_VELOCITY_MG_HPP
#define PECLET_FLOW_MAC_VELOCITY_MG_HPP

#include <functional>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"       // restrictAvg, prolongAdd, FPV/FPC
#include "mac_ibm.hpp"              // ibmRbgsStencilColor (pin smoother), MConst
#include "staggered_advection.hpp"  // fou_operator_aniso (upwind-convective coarse op)

namespace peclet::flow {

// pin-aware variable-coefficient residual (mg_residual_var_k): r = 0 at pinned (classified-solid)
// cells, else b - A x with the float operator accumulated in double.
inline void residualVarPin(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN,
                           FPC AB, FPC AT, CCConst pin, C3 e, int g) {
  CCExec space;
  const bool hasPin = (pin.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::vmg_resid", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        if (hasPin && pin(i) > 0.5) {
          r(i) = 0.0;
          return;
        }
        const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) +
                          (double)AW(i) * x(i - sx) + (double)AN(i) * x(i + sy) +
                          (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                          (double)AB(i) * x(i - sz);
        r(i) = b(i) - Ax;
      });
}

// masked trilinear prolongation (mg_prolong_masked_k): like prolongAdd but does NOT add the coarse
// correction into a fine cell whose mask < eps (the clean-fluid exclude mask is 0 at IBM cut+solid
// cells).
inline void prolongMasked(CCField fine, CCConst coarse, CCConst mask, C3 fext, C3 cext, int g,
                          C3 finner, C3 ratio, double eps) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::vmg_prolong_masked", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        const long fi =
            (long)(ifx + g) + (long)(ify + g) * fext.x + (long)(ifz + g) * (long)fext.x * fext.y;
        if (mask(fi) < eps)
          return;  // no correction into a cut/solid fine cell
        const double cx = (ratio.x == 2) ? 0.5 * ifx - 0.25 + g : ifx + g;
        const double cy = (ratio.y == 2) ? 0.5 * ify - 0.25 + g : ify + g;
        const double cz = (ratio.z == 2) ? 0.5 * ifz - 0.25 + g : ifz + g;
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
        fine(fi) += c0 * (1 - wz) + c1 * wz;
      });
}

// STAIRCASE coarse operator (mg_build_velocity_op_staircase_k): theta<thresh -> identity (pinned)
// row; else a plain const-coeff Helmholtz (idiag + 2*(bx+by+bz) diagonal, per-axis -b
// off-diagonals).
inline void buildVelocityStaircase(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT,
                                   CCConst theta, C3 e, int g, double bx, double by, double bz,
                                   double thresh, double idiag) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::vmg_staircase", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (theta(i) <
            thresh) {  // classified solid -> identity row (smoother/residual pin it to 0)
          AC(i) = (MReal)1.0;
          AW(i) = AE(i) = AS(i) = AN(i) = AB(i) = AT(i) = (MReal)0.0;
          return;
        }
        AC(i) = (MReal)(idiag + 2.0 * (bx + by + bz));
        AW(i) = (MReal)(-bx);
        AE(i) = (MReal)(-bx);
        AS(i) = (MReal)(-by);
        AN(i) = (MReal)(-by);
        AB(i) = (MReal)(-bz);
        AT(i) = (MReal)(-bz);
      });
}

// classified-solid rows (theta < thresh) -> identity, over the inner cells (the staircase pin
// applied after an operator builder that wrote every row).
inline void pinSolidRows(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, CCConst theta,
                         C3 e, int g, double thresh) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::vmg_pin_rows", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        if (theta(i) < thresh) {
          AC(i) = (MReal)1.0;
          AW(i) = AE(i) = AS(i) = AN(i) = AB(i) = AT(i) = (MReal)0.0;
        }
      });
}

// UPWIND-CONVECTIVE coarse operator (build_adv_coarse_stencil_k): anisotropic const-coeff
// backward-Euler diffusion (per-axis beta bx/by/bz) PLUS first-order-upwind advection from the
// restricted coarse advecting velocity (scaled by s_a=1/cfac_a per face axis). M-matrix on every
// level -> stable in the advection- dominated rows. The fine residual + smoother give the exact
// sharp-IBM answer; this only sets the rate.
inline void buildAdvCoarse(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, CCConst U,
                           CCConst V, CCConst W, int comp, C3 e, int g, double bx, double by,
                           double bz, double fouw, double sx, double sy, double sz, double idiag) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::vmg_adv_coarse", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double cC = idiag + 2.0 * (bx + by + bz), cxm = -bx, cxp = -bx, cym = -by, cyp = -by,
               czm = -bz, czp = -bz;
        sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
        sadv::fou_operator_aniso(comp, x, y, z, Ua, Va, Wa, fouw, sx, sy, sz, cC, cxm, cxp, cym,
                                 cyp, czm, czp);
        AC(i) = (MReal)cC;
        AW(i) = (MReal)cxm;
        AE(i) = (MReal)cxp;
        AS(i) = (MReal)cym;
        AN(i) = (MReal)cyp;
        AB(i) = (MReal)czm;
        AT(i) = (MReal)czp;
      });
}

// CONST-COEFF anisotropic Helmholtz A = idiag*I - nu_dt*Lap (mg_const_diffusion_op_aniso_k), over
// the WHOLE extended block (the domain-BC velocity op; coarse spacing via per-axis beta). For
// cavity/BFS where the fine op is also const-coeff (no IBM stencil).
inline void buildConstAniso(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, C3 e, double bx,
                            double by, double bz, double idiag) {
  CCExec space;
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  const MReal c = (MReal)(idiag + 2.0 * (bx + by + bz)), nx = (MReal)(-bx), ny = (MReal)(-by),
              nz = (MReal)(-bz);
  Kokkos::parallel_for(
      "peclet::flow::vmg_const_aniso", Kokkos::RangePolicy<CCExec>(space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        AC(i) = c;
        AW(i) = nx;
        AE(i) = nx;
        AS(i) = ny;
        AN(i) = ny;
        AB(i) = nz;
        AT(i) = nz;
      });
}

// No-slip face-fold for the const-coeff MG operator (mg_diffusion_bc_fold_k): at a Dirichlet wall
// the tangential ghost is 2*u_wall - u_inner -> +beta moves onto the boundary-adjacent inner cell's
// diagonal (AC += beta); the dropped off-diagonal multiplies a held-0 ghost. Over the perp plane of
// face (a,s).
inline void boundaryFold(FPV AC, C3 e, int g, int a, int s, double beta) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int bic = (s == 0) ? g : (dims[a] - g - 1);  // boundary-adjacent inner cell along a
  Kokkos::parallel_for(
      "peclet::flow::vmg_bc_fold",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long i = (long)p0 * sb + (long)p1 * sc + (long)bic * sa;
        AC(i) = (MReal)((double)AC(i) + beta);
      });
}

// Fill a non-periodic boundary ghost of a coarse correction before trilinear prolongation
// (mg_fill_bc_ghost_k): Dirichlet (outflow) -> ghost 0; Neumann (wall/inflow) -> ghost = nearest
// inner (zero-gradient). Plane (a,s).
inline void fillBcGhost(CCField x, C3 e, int g, int a, int s, int dirichlet) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int na = dims[a];
  Kokkos::parallel_for(
      "peclet::flow::vmg_fill_bc_ghost",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = (long)p0 * sb + (long)p1 * sc;
        if (s == 0) {
          const double v = dirichlet ? 0.0 : x(base + (long)g * sa);
          for (int ia = 0; ia < g; ++ia)
            x(base + (long)ia * sa) = v;
        } else {
          const double v = dirichlet ? 0.0 : x(base + (long)(na - g - 1) * sa);
          for (int ia = na - g; ia < na; ++ia)
            x(base + (long)ia * sa) = v;
        }
      });
}

// zero a field on the plane at index `idx` along `axis` (held-Dirichlet boundary-face residual
// exclude).
inline void zeroPlane(CCField m, C3 e, int axis, int idx) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (axis + 1) % 3, c = (axis + 2) % 3;
  const long sa = st[axis], sb = st[b], sc = st[c];
  Kokkos::parallel_for(
      "peclet::flow::vmg_zero_plane",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { m((long)p0 * sb + (long)p1 * sc + (long)idx * sa) = 0.0; });
}

inline void thresholdMask(CCField m, CCConst theta,
                          double thresh) {  // m = 1 where theta < thresh (solid)
  CCExec space;
  std::size_t n = m.extent(0);
  CCField mm = m;
  CCConst th = theta;
  Kokkos::parallel_for(
      "peclet::flow::vmg_threshold", Kokkos::RangePolicy<CCExec>(space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) { mm(i) = (th(i) < thresh) ? 1.0 : 0.0; });
}
// max |a| over the inner cells of a G-ghosted block.
inline double maxAbsInner(CCConst a, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  double m = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::vmg_maxabs", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double d = Kokkos::fabs(a(i));
        if (d > acc)
          acc = d;
      },
      Kokkos::Max<double>(m));
  return m;
}
// max |a - b| over the inner cells of a G-ghosted block (the V-cycle update norm).
inline double maxAbsDiffInner(CCConst a, CCConst b, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  double m = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::vmg_maxabsdiff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double d = Kokkos::fabs(a(i) - b(i));
        if (d > acc)
          acc = d;
      },
      Kokkos::Max<double>(m));
  return m;
}
inline void mulMask(CCField r, CCConst m) {  // r *= m (clean-fluid residual filter)
  CCExec space;
  std::size_t n = r.extent(0);
  CCField rr = r;
  CCConst mm = m;
  Kokkos::parallel_for(
      "peclet::flow::vmg_mulmask", Kokkos::RangePolicy<CCExec>(space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) { rr(i) *= mm(i); });
}

// Velocity (momentum) geometric multigrid with the staircase coarse operator. All levels ghost
// width G=2.
class VelocityMG {
 public:
  static constexpr int G = 2;
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    C3 og{0, 0, 0};  // block inner origin (global red-black parity); {0,0,0} single-rank
    C3 gdim{0, 0, 0};  // GLOBAL dims of this level (og + inner == gdim -> owns the +face)
    std::size_t n = 0;
    CCField x, rhs, res, theta, pin, resMask;
    CCField advU, advV, advW;  // restricted advecting velocity (upwind-convective coarse op; L>=1)
    FPV AC, AW, AE, AS, AN, AB, AT;
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;  // per-level topology (decomposed)
    std::shared_ptr<GridHalo<double>> dev;      // per-level ghost exchange (ghost width G=2)
#endif
  };

  // periodic uniform hierarchy (halve each axis while even and >=2, capped at nLevels).
  void init(int nx, int ny, int nz, int nLevels) {
    lv_.clear();
    C3 inner{nx, ny, nz}, cf{1, 1, 1};
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.inner = inner;
      v.gdim = inner;
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
      v.x = CCField("vmg_x", v.n);
      v.rhs = CCField("vmg_rhs", v.n);
      v.res = CCField("vmg_res", v.n);
      v.theta = CCField("vmg_th", v.n);
      v.pin = CCField("vmg_pin", v.n);
      if (L > 0) {  // coarse advecting velocity for the upwind-convective coarse op
        v.advU = CCField("vmg_au", v.n);
        v.advV = CCField("vmg_av", v.n);
        v.advW = CCField("vmg_aw", v.n);
      }
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT})
        *p = FPV("vmg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z)
        break;
      inner = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    lv_[0].resMask =
        CCField("vmg_resmask0", lv_[0].n);  // level 0 only (clean-fluid exclude, staircase path)
  }
#ifdef PECLET_FLOW_MPI
  // Multi-rank velocity-MG: coarsen the GLOBAL grid 2:1 per level; each level gets its own G=2 core
  // halo. No global reductions here (the velocity op is non-singular -> no mean removal, no
  // Krylov), so the fold is just fill()->exchange + the block-origin red-black parity. Single-rank
  // (size 1) == init().
  void initMpi(int gnx, int gny, int gnz, int nLevels, MPI_Comm comm) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    initMpi(peclet::core::decomp::BlockDecomposer<3>(static_cast<std::size_t>(size),
                                                     peclet::core::IVec<3>{gnx, gny, gnz}),
            nLevels, comm, /*inPlace=*/false);
  }
  // The solver's hierarchy: level 0 on the caller's decomposition (the velocity block IbmSolver
  // already exchanges on), coarse levels = that decomposition coarsened IN PLACE, so an axis
  // coarsens only while every rank's block origin and size are even on it (the same gate as
  // CutcellMG without telescoping; with inPlace=false a fresh plain ORB is built per level, which
  // only the standalone ctest wants). Every rank sees the same replicated decomposition, so the
  // level count agrees globally with no communication.
  void initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec0, int nLevels, MPI_Comm comm,
               bool inPlace = true) {
    lv_.clear();
    distributed_ = true;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::array<bool, 3> per{true, true, true};
    const auto& g0 = dec0.globalSize();
    C3 gs{(int)g0[0], (int)g0[1], (int)g0[2]}, cf{1, 1, 1};
    auto can = [&](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    auto evenOn = [](const peclet::core::decomp::BlockDecomposer<3>& d, int ax) {
      for (std::size_t b = 0; b < d.numBlocks(); ++b) {
        const auto blk = d.block(b);
        if (blk.origin[ax] % 2 != 0 || blk.size[ax] % 2 != 0)
          return false;
      }
      return true;
    };
    peclet::core::decomp::BlockDecomposer<3> dec = dec0;
    for (int L = 0; L < nLevels; ++L) {
      Level v;
      v.halo = std::make_shared<GridHaloTopology<3>>();
      if (!inPlace && L > 0)
        dec = peclet::core::decomp::BlockDecomposer<3>(static_cast<std::size_t>(size),
                                                       peclet::core::IVec<3>{gs.x, gs.y, gs.z});
      v.halo->buildTopology(dec, rank, G, per, comm);
      v.dev = std::make_shared<GridHalo<double>>();
      v.dev->init(*v.halo);
      v.dev->setLabel("vmg L" + std::to_string(L));
      const auto& idx = v.halo->indexer();
      const auto eg = idx.sizeInclGhost(), ino = idx.sizeInner(), oig = idx.originInclGhost();
      v.ext = {(int)eg[0], (int)eg[1], (int)eg[2]};
      v.inner = {(int)ino[0], (int)ino[1], (int)ino[2]};
      v.og = {(int)oig[0] + G, (int)oig[1] + G, (int)oig[2] + G};
      v.gdim = gs;
      v.cfac = cf;
      v.n = idx.numCellsInclGhost();
      C3 next = gs, ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(gs.x) && (!inPlace || evenOn(dec, 0))) {
          ratio.x = 2;
          next.x = gs.x / 2;
        }
        if (can(gs.y) && (!inPlace || evenOn(dec, 1))) {
          ratio.y = 2;
          next.y = gs.y / 2;
        }
        if (can(gs.z) && (!inPlace || evenOn(dec, 2))) {
          ratio.z = 2;
          next.z = gs.z / 2;
        }
      }
      v.ratio = ratio;
      v.x = CCField("vmg_x", v.n);
      v.rhs = CCField("vmg_rhs", v.n);
      v.res = CCField("vmg_res", v.n);
      v.theta = CCField("vmg_th", v.n);
      v.pin = CCField("vmg_pin", v.n);
      if (L > 0) {
        v.advU = CCField("vmg_au", v.n);
        v.advV = CCField("vmg_av", v.n);
        v.advW = CCField("vmg_aw", v.n);
      }
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT})
        *p = FPV("vmg_A", v.n);
      lv_.push_back(v);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z)
        break;
      gs = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
      if (inPlace)
        dec = dec.coarsened(peclet::core::IVec<3>{ratio.x, ratio.y, ratio.z});
    }
    lv_[0].resMask = CCField("vmg_resmask0", lv_[0].n);
  }
#endif
  // Does this rank's block on level `lv` touch global face f (0=-x,1=+x,2=-y,3=+y,4=-z,5=+z)?
  // Single-rank: always. The domain-BC fold, the held-face residual exclude, the correction
  // ghosts and the prolongation ghosts are all applied only by the owning rank (WO-F rule).
  bool touches(const Level& lv, int f) const {
    const int a = f / 2;
    const int o = (a == 0) ? lv.og.x : (a == 1) ? lv.og.y : lv.og.z;
    const int n = (a == 0) ? lv.inner.x : (a == 1) ? lv.inner.y : lv.inner.z;
    const int gn = (a == 0) ? lv.gdim.x : (a == 1) ? lv.gdim.y : lv.gdim.z;
    return (f % 2 == 0) ? (o == 0) : (o + n == gn);
  }
  int nLevels() const { return (int)lv_.size(); }
  Level& level(int L) { return lv_[L]; }

  // level-0 fine operator = the external IBM stencil (7 float arrays on the same G=2 block).
  void setFineStencil(FPC AC, FPC AW, FPC AE, FPC AS, FPC AN, FPC AB, FPC AT) {
    Level& f = lv_[0];
    Kokkos::deep_copy(f.AC, AC);
    Kokkos::deep_copy(f.AW, AW);
    Kokkos::deep_copy(f.AE, AE);
    Kokkos::deep_copy(f.AS, AS);
    Kokkos::deep_copy(f.AN, AN);
    Kokkos::deep_copy(f.AB, AB);
    Kokkos::deep_copy(f.AT, AT);
  }

  // staircase coarse op: level-0 pin = fine solid mask, resMask = clean-fluid mask; coarse levels
  // classify by restricted theta and build a const-coeff Helmholtz. nu_dt = mu, idiag = rho/dt, h0
  // = 1.
  void setStaircase(CCConst theta0, CCConst solid0, CCConst resmask0, double nu_dt, double idiag,
                    double thresh) {
    usePin_ = true;
    useResMask_ = true;  // staircase: pin classified-solid cells + exclude the IBM band
    Level& f = lv_[0];
    Kokkos::deep_copy(f.theta, theta0);
    Kokkos::deep_copy(f.pin, solid0);
    Kokkos::deep_copy(f.resMask, resmask0);
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      Level& fin = lv_[L - 1];
      restrictAvg(c.theta, CCConst(fin.theta), c.ext, fin.ext, G, G, c.inner,
                  fin.ratio);  // coarse theta = avg
      thresholdMask(c.pin, CCConst(c.theta), thresh);
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x),
                   by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      buildVelocityStaircase(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.theta), c.ext, G,
                             bx, by, bz, thresh, idiag);
    }
  }

  // MIXED operator: an immersed solid AND domain BCs (the packed-bed-with-inlet case). Level 0 =
  // the external cut-cell stencil (setFineStencil) with the solid pin and the clean-fluid exclude,
  // PLUS the held normal-Dirichlet boundary face excluded from coarsening (as setDomainBcOp does);
  // coarse levels = the staircase Helmholtz (classified solid pinned) PLUS the domain-face folds
  // (tangential wall/inflow +beta, outflow -beta) on the owning rank. The non-periodic correction
  // ghosts are zero (fill() / the smoother keep them so), which is the Dirichlet-0 the fold
  // assumes. Needs setBC() first. comp = velocity component (the fold is component-dependent).
  // upwind: the fine stencil carries implicit first-order-upwind advection, so the coarse
  // operator adds the FOU part from the restricted advecting velocity (restrictAdvVelocities must
  // have run this step) on the fluid rows, the classified-solid rows staying pinned.
  void setStaircaseBc(int comp, CCConst theta0, CCConst solid0, CCConst resmask0, double nu_dt,
                      double idiag, double thresh, bool upwind = false, double fouw = 0.0) {
    heldComp_ = comp;
    setStaircase(theta0, solid0, resmask0, nu_dt, idiag, thresh);
    if (upwind)
      for (int L = 1; L < (int)lv_.size(); ++L) {
        Level& c = lv_[L];
        const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x),
                     by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                     bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
        buildAdvCoarse(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.advU), CCConst(c.advV),
                       CCConst(c.advW), comp, c.ext, G, bx, by, bz, fouw, 1.0 / c.cfac.x,
                       1.0 / c.cfac.y, 1.0 / c.cfac.z, idiag);
        pinSolidRows(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.theta), c.ext, G, thresh);
      }
    Level& f = lv_[0];
    {
      const int t = bc_[2 * comp];  // the -side face of the normal component lands at index G
      if ((t == 1 || t == 2) && touches(f, 2 * comp))
        zeroPlane(f.resMask, f.ext, comp, G);
    }
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x),
                   by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      for (int ff = 0; ff < 6; ++ff) {
        const int a = ff / 2, sd = ff % 2;
        const double ba = (a == 0) ? bx : (a == 1) ? by : bz;
        double dval;
        if (bc_[ff] == 3)
          dval = -ba;
        else if ((bc_[ff] == 1 || bc_[ff] == 2) && a != comp)
          dval = ba;
        else
          continue;
        if (touches(c, ff))
          boundaryFold(c.AC, c.ext, G, a, sd, dval);
      }
    }
  }

  // UPWIND-CONVECTIVE coarse op (implicit-FOU): restrict the advecting velocity u/v/w (level-0
  // block) to every coarse level (8:1 average; numerical diffusion is welcome -> keeps the
  // M-matrix). Call ONCE per Picard iteration, before buildUpwindCoarse for the 3 components (the
  // velocity is shared/frozen at u^k).
  void restrictAdvVelocities(CCConst u0, CCConst v0, CCConst w0) {
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& cs = lv_[L];
      Level& fin = lv_[L - 1];
      CCConst fu = (L == 1) ? u0 : CCConst(fin.advU), fv = (L == 1) ? v0 : CCConst(fin.advV),
              fw = (L == 1) ? w0 : CCConst(fin.advW);
      restrictAvg(cs.advU, fu, cs.ext, fin.ext, G, G, cs.inner, fin.ratio);
      fill(cs, cs.advU);
      restrictAvg(cs.advV, fv, cs.ext, fin.ext, G, G, cs.inner, fin.ratio);
      fill(cs, cs.advV);
      restrictAvg(cs.advW, fw, cs.ext, fin.ext, G, G, cs.inner, fin.ratio);
      fill(cs, cs.advW);
    }
  }
  // Build the coarse operators for component comp = aniso const-coeff diffusion + dt*FOU from the
  // restricted advecting velocity (level 0 stays the fine As_[comp] set by setFineStencil). No pin
  // / no exclude mask -- the upwind M-matrix is stable; the fine residual gives the exact sharp
  // answer. Per Picard iter, per comp.
  void buildUpwindCoarse(int comp, double nu_dt, double idiag, double fouw) {
    usePin_ = false;
    useResMask_ = false;  // upwind path: pure variable-coeff MG (no pin/exclude)
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x),
                   by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      const double sx = 1.0 / (double)c.cfac.x, sy = 1.0 / (double)c.cfac.y,
                   sz = 1.0 / (double)c.cfac.z;
      buildAdvCoarse(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.advU), CCConst(c.advV),
                     CCConst(c.advW), comp, c.ext, G, bx, by, bz, fouw, sx, sy, sz, idiag);
    }
  }

  // DOMAIN-BC const-coeff path (cavity/BFS): per-face BC types {-x,+x,-y,+y,-z,+z}
  // (0=periodic,1=wall, 2=inflow,3=outflow). Enables the non-periodic fill (periodic axes wrap;
  // non-periodic ghosts left as the caller / correction set them) + the Dirichlet/Neumann
  // prolongation ghosts.
  void setBC(const int bc[6]) {
    bcMode_ = false;
    for (int i = 0; i < 6; ++i) {
      bc_[i] = bc[i];
      if (bc[i])
        bcMode_ = true;
    }
  }
  // Re-impose the full velocity BC on the level-0 iterate before each smoother colour + the
  // residual (exactly as the RB-GS path does via fillVelGhosts(c,1)): the const-coeff smoother
  // updates the held Dirichlet faces, so without this the boundary corners drift (~2% vs RB-GS, as
  // the CUDA vmg also does). With it the vel-MG converges to the RB-GS fixed point. IbmSolver
  // supplies this per component before the solve.
  void setBcApplyL0(std::function<void(CCField)> fn) { bcApplyL0_ = std::move(fn); }
  // const-coeff aniso operator + no-slip/inflow/outflow boundary fold for component comp, on EVERY
  // level. nu_dt = mu, idiag = rho/dt, h0 = 1. Rebuilt per component (the fold is
  // component-dependent). No pin. useResMask_: exclude the HELD normal-Dirichlet boundary face
  // (a==comp, -side, wall/inflow) from coarsening -- that cell's value is pinned by the BC
  // re-imposition, so its (nonzero) residual would drive a spurious coarse correction into the
  // boundary (the ~2% drift CUDA's domain-BC vmg leaves). Excluding it makes the V-cycle converge
  // to the RB-GS fixed point (analogue of the IBM clean-fluid exclude).
  void setDomainBcOp(int comp, double nu_dt, double idiag) {
    heldComp_ = comp;
    usePin_ = false;
    Level& f = lv_[0];
    Kokkos::deep_copy(f.resMask, 1.0);
    useResMask_ = false;
    for (int s = 0; s < 1;
         ++s) {  // only the -side face index G lands inside the smoother range [G, ext-G)
      const int t = bc_[2 * comp + s];
      if (t == 1 || t == 2) {
        if (touches(f, 2 * comp + s))
          zeroPlane(f.resMask, f.ext, comp, G);
        useResMask_ = true;  // rank-uniform (the masked transfers must agree across ranks)
      }
    }
    for (int L = 0; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x),
                   by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      buildConstAniso(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, c.ext, bx, by, bz, idiag);
      for (int f = 0; f < 6; ++f) {
        const int a = f / 2, s = f % 2;
        const double ba = (a == 0) ? bx : (a == 1) ? by : bz;
        double dval;
        if (bc_[f] == 3)
          dval = -ba;  // outflow zero-gradient: every component
        else if ((bc_[f] == 1 || bc_[f] == 2) && a != comp)
          dval = ba;  // wall/inflow: tangential fold
        else
          continue;  // periodic, or the normal comp at a wall
        if (touches(c, f))
          boundaryFold(c.AC, c.ext, G, a, s, dval);
      }
    }
  }

  // solve A x = b (b,x on the level-0 G=2 block): up to nvc V-cycles. Solution left in x.
  // Two optional stops (global max under MPI):
  //   tol    > 0: max|x_{k+1} - x_k| over the inner cells <= tol * (first cycle's) -- the update
  //               criterion IbmSolver's RB-GS sweep loop uses, so cycles and sweeps compare;
  //   resTol > 0: max|b - A x| <= resTol * max|b| (the level-0 fine operator, pinned rows 0) --
  //               the criterion that measures convergence itself; a warm start that already
  //               solves the equation stops at once. resTol takes precedence when both are set.
  // Returns the cycles run; lastResidualRatio() is max|r|/max|b| at exit (resTol mode only).
  int solve(CCConst b, CCField x, int nvc, int pre, int post, int bottom, double tol = 0.0,
            MPI_Comm comm = MPI_COMM_NULL, double resTol = 0.0) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(l0.rhs, b);
    Kokkos::deep_copy(l0.x, x);
    auto gmax = [&](double v) {
#ifdef PECLET_FLOW_MPI
      if (distributed_ && comm != MPI_COMM_NULL) {
        double g = 0.0;
        MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
        return g;
      }
#else
      (void)comm;
#endif
      return v;
    };
    auto residual = [&]() {
      fill(l0, l0.x);
      if (bcApplyL0_)
        bcApplyL0_(l0.x);
      residualVarPin(l0.res, CCConst(l0.x), CCConst(l0.rhs), FPC(l0.AC), FPC(l0.AW), FPC(l0.AE),
                     FPC(l0.AS), FPC(l0.AN), FPC(l0.AB), FPC(l0.AT),
                     usePin_ ? CCConst(l0.pin) : empty_, l0.ext, G);
      // the held normal-Dirichlet face (wall / inflow, -side) is imposed, not solved: not part
      // of the convergence measure
      if (bcMode_ && heldComp_ >= 0) {
        const int t = bc_[2 * heldComp_];
        if ((t == 1 || t == 2) && touches(l0, 2 * heldComp_))
          zeroPlane(l0.res, l0.ext, heldComp_, G);
      }
      return gmax(maxAbsInner(CCConst(l0.res), l0.ext, G));
    };
    // The forcing may enter through a Dirichlet ghost (an inflow) rather than through b, so the
    // convergence scale is max(max|b|, max|A x|) with max|A x| = max|b - r| from the initial
    // residual -- max|b| alone is ~0 on the first step from rest and the stop never triggers.
    const bool useRes = resTol > 0.0;
    const bool useDu = !useRes && tol > 0.0;
    if (useDu && prev_.extent(0) != l0.n)
      prev_ = CCField("vmg_prev", l0.n);
    double du0 = 0.0, bnorm = 0.0;
    int used = nvc;
    lastResRatio_ = -1.0;
    if (useRes) {
      const double r0 = residual();
      bnorm = std::max(gmax(maxAbsInner(CCConst(l0.rhs), l0.ext, G)),
                       gmax(maxAbsDiffInner(CCConst(l0.rhs), CCConst(l0.res), l0.ext, G)));
      lastResRatio_ = bnorm > 0.0 ? r0 / bnorm : 0.0;
      if (r0 <= resTol * bnorm) {  // the warm start already solves it
        Kokkos::deep_copy(x, l0.x);
        return 0;
      }
    }
    for (int v = 0; v < nvc; ++v) {
      if (useDu)
        Kokkos::deep_copy(prev_, l0.x);
      vcycle(0);
      if (useRes) {
        const double r = residual();
        lastResRatio_ = bnorm > 0.0 ? r / bnorm : 0.0;
        if (r <= resTol * bnorm) {
          used = v + 1;
          break;
        }
      } else if (useDu) {
        const double du = gmax(maxAbsDiffInner(CCConst(l0.x), CCConst(prev_), l0.ext, G));
        if (v == 0)
          du0 = du;
        if (du <= tol * du0) {
          used = v + 1;
          break;
        }
      }
    }
    Kokkos::deep_copy(x, l0.x);
    return used;
  }
  double lastResidualRatio() const { return lastResRatio_; }

 public:  // (public for nvcc extended-lambda)
  void vcycle(int L) {
    Level& lv = lv_[L];
    const bool l0 = (L == 0);
    if (L + 1 == (int)lv_.size()) {
      smooth(lv, bottom_, l0);
      return;
    }  // velocity op non-singular -> no mean removal
    smooth(lv, pre_, l0);
    fill(lv, lv.x);
    if (l0 && bcApplyL0_)
      bcApplyL0_(lv.x);  // domain-BC: re-impose the velocity BC before the level-0 residual
    residualVarPin(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                   FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT),
                   usePin_ ? CCConst(lv.pin) : empty_, lv.ext, G);
    const bool masked =
        useResMask_ && (lv.resMask.extent(0) == lv.n);  // level 0: exclude the IBM cut-cell band
    if (masked)
      mulMask(lv.res, CCConst(lv.resMask));
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, G, G, cs.inner, lv.ratio);
    Kokkos::deep_copy(cs.x, 0.0);
    vcycle(L + 1);
    fill(cs, cs.x);
    fillProlongBcGhosts(
        cs);  // non-periodic boundary ghosts the trilinear prolong samples (domain-BC mode)
    if (masked)
      prolongMasked(lv.x, CCConst(cs.x), CCConst(lv.resMask), lv.ext, cs.ext, G, lv.inner, lv.ratio,
                    0.5);
    else
      prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, G, G, lv.inner, lv.ratio);
    smooth(lv, post_, l0);
  }
  void smooth(Level& lv, int sweeps, bool isL0) {
    const C3 og = lv.og;  // global red-black parity (block inner origin); {0,0,0} single-rank
    CCConst pin = usePin_ ? CCConst(lv.pin) : empty_;
    for (int k = 0; k < sweeps; ++k)
      for (int color = 0; color < 2; ++color) {
#ifdef PECLET_FLOW_MPI
        if (distributed_ && !bcMode_) {
          // Overlap the per-colour halo with the interior sweep (see CutcellMG::smooth) — only on
          // the periodic/IBM path: the per-colour BC hook re-imposes ghost/held cells AFTER the
          // fill in the blocking order, which the split would reorder.
          const C3 lo{G + 1, G + 1, G + 1};
          const C3 hi{lv.ext.x - G - 1, lv.ext.y - G - 1, lv.ext.z - G - 1};
          lv.dev->exchangeBegin(lv.x);
          ibmRbgsStencilColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW),
                                 FPC(lv.AE), FPC(lv.AS), FPC(lv.AN), FPC(lv.AB),
                                 FPC(lv.AT), pin, lv.ext, og, color, lo, hi, C3{0, 0, 0},
                                 C3{0, 0, 0});
          lv.dev->exchangeEnd(lv.x);
          ibmRbgsStencilColorBox(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW),
                                 FPC(lv.AE), FPC(lv.AS), FPC(lv.AN), FPC(lv.AB),
                                 FPC(lv.AT), pin, lv.ext, og, color, C3{G, G, G},
                                 C3{lv.ext.x - G, lv.ext.y - G, lv.ext.z - G}, lo, hi);
          continue;
        }
#endif
        fill(lv, lv.x);
        if (isL0 && bcApplyL0_)
          bcApplyL0_(lv.x);  // re-impose the velocity BC (held Dirichlet faces) per colour
        ibmRbgsStencilColor(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE),
                            FPC(lv.AS), FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), pin, lv.ext,
                            og, G, color);
      }
  }
  // periodic ghost fill; in domain-BC mode only the periodic axes wrap (non-periodic boundary
  // ghosts are left as the caller / correction set them -- the boundary fold + held ghost represent
  // the wall). Distributed (periodic IBM path): the per-level core halo (cross-rank + periodic in
  // one call).
  void fill(Level& lv, CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      lv.dev->exchange(f);  // cross-rank + periodic on all three axes
      if (bcMode_)
        zeroBcGhosts(lv, f);
      return;
    }
#endif
    for (int a = 0; a < 3; ++a)
      if (!bcMode_ || (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0))
        fillAxis(lv, f, a);
  }
  // Non-periodic global-face ghosts of `f` -> 0 on the owning rank (the halo exchange is built
  // periodic on every axis, so it has just wrapped the opposite side of the domain into them). A
  // coarse correction wants exactly 0 there; level 0 gets the real BC from the hook afterwards.
  void zeroBcGhosts(Level& lv, CCField f) {
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s)
        if (bc_[2 * a + s] != 0 && touches(lv, 2 * a + s))
          fillBcGhost(f, lv.ext, G, a, s, /*dirichlet=*/1);
  }
  // non-periodic boundary ghosts of a coarse correction before trilinear prolong (Dirichlet outflow
  // -> 0, Neumann wall/inflow -> zero-gradient).
  void fillProlongBcGhosts(Level& lv) {
    if (!bcMode_)
      return;
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (t == 0 || !touches(lv, 2 * a + s))
          continue;
        fillBcGhost(lv.x, lv.ext, G, a, s, t == 3 ? 1 : 0);
      }
  }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space;
    C3 e = lv.ext;
    int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    const int N = N3[a];
    CCField ff = f;
    Kokkos::parallel_for(
        "peclet::flow::vmg_pfill",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
        KOKKOS_LAMBDA(int p0, int p1) {
          const long base = (long)p0 * sb + (long)p1 * sc;
          for (int gl = 0; gl < G; ++gl) {
            ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
            ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
          }
        });
  }

 private:
  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 8;
  bool usePin_ = true,
       useResMask_ = true;  // staircase: pin + clean-fluid exclude; upwind/domain-BC: neither
  bool bcMode_ = false;
  int bc_[6] = {0, 0, 0, 0, 0, 0};          // domain-BC (non-periodic) mode
  std::function<void(CCField)> bcApplyL0_;  // re-impose the velocity BC on level 0 (domain-BC mode)
  CCConst empty_;                           // zero-extent View -> "no pin / no mask" to the kernels
  bool distributed_ = false;  // multi-rank (initMpi); fill() exchanges on every level
  CCField prev_;              // previous iterate for the update-tolerance stop (solve tol > 0)
  double lastResRatio_ = -1.0;  // max|r|/max|b| at exit of the last resTol-mode solve
  int heldComp_ = -1;           // component of the last domain-BC operator build (held-face exclude)
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_VELOCITY_MG_HPP
