/// @file
/// @brief sdflow — portable (Kokkos) velocity (momentum) multigrid for the IBM diffusion solve: the STAIRCASE
/// coarse operator.
///
/// Single-GPU (periodic) port of the velocity-MG path in CUDA's DistributedPoissonMG (mac_multigrid.cuh):
/// the fine level is the sharp Robust-Scaled IBM stencil As_[c] (so the residual + smoother use the TRUE
/// operator and the fixed point is the exact sharp solution); the coarse levels use the geometry-aware
/// STAIRCASE Helmholtz (volume fraction theta only CLASSIFIES cells: theta>=0.5 fluid / <0.5 solid-pinned,
/// then a plain constant-coefficient Helmholtz at fluid cells). The fine IBM-cell residuals are excluded
/// from coarsening (clean-fluid mask) and no coarse correction is pumped back into the cut-cell band
/// (masked prolong); the fine smoother owns the boundary. See [[velocity-mg-design]].
///
/// The whole hierarchy uses ghost width G=2 (the velocity block's width), so level 0 IS the solver's
/// velocity block: the IBM stencil + RHS + solution need no g=2<->g=1 bridging. Reuses restrictAvg /
/// prolongAdd (mac_cutcell_mg) and ibmRbgsStencilColor (the pin-aware variable-coeff RB-GS smoother ==
/// mg_smooth_var_k). Runs on any Kokkos backend.
#ifndef PECLET_FLOW_MAC_VELOCITY_MG_HPP
#define PECLET_FLOW_MAC_VELOCITY_MG_HPP

#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>

#include "mac_cutcell_mg.hpp"       // restrictAvg, prolongAdd, FPV/FPC
#include "mac_ibm.hpp"              // ibmRbgsStencilColor (pin smoother), MConst
#include "staggered_advection.hpp"  // fou_operator_aniso (upwind-convective coarse op)

namespace peclet::flow {

// pin-aware variable-coefficient residual (mg_residual_var_k): r = 0 at pinned (classified-solid) cells,
// else b - A x with the float operator accumulated in double.
inline void residualVarPin(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN, FPC AB,
                           FPC AT, CCConst pin, C3 e, int g) {
  CCExec space; const bool hasPin = (pin.extent(0) != 0);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("peclet::flow::vmg_resid", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
    KOKKOS_LAMBDA(int lx, int ly, int lz) {
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const long i = (long)lx + (long)ly * sy + (long)lz * sz;
      if (hasPin && pin(i) > 0.5) { r(i) = 0.0; return; }
      const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(i + sx) + (double)AW(i) * x(i - sx) +
                        (double)AN(i) * x(i + sy) + (double)AS(i) * x(i - sy) + (double)AT(i) * x(i + sz) +
                        (double)AB(i) * x(i - sz);
      r(i) = b(i) - Ax;
    });

}

// masked trilinear prolongation (mg_prolong_masked_k): like prolongAdd but does NOT add the coarse
// correction into a fine cell whose mask < eps (the clean-fluid exclude mask is 0 at IBM cut+solid cells).
inline void prolongMasked(CCField fine, CCConst coarse, CCConst mask, C3 fext, C3 cext, int g, C3 finner,
                          C3 ratio, double eps) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("peclet::flow::vmg_prolong_masked", MD(space, {0, 0, 0}, {finner.x, finner.y, finner.z}),
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

}

// STAIRCASE coarse operator (mg_build_velocity_op_staircase_k): theta<thresh -> identity (pinned) row;
// else a plain const-coeff Helmholtz (idiag + 2*(bx+by+bz) diagonal, per-axis -b off-diagonals).
inline void buildVelocityStaircase(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, CCConst theta,
                                   C3 e, int g, double bx, double by, double bz, double thresh, double idiag) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("peclet::flow::vmg_staircase", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
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

}

// UPWIND-CONVECTIVE coarse operator (build_adv_coarse_stencil_k): anisotropic const-coeff backward-Euler
// diffusion (per-axis beta bx/by/bz) PLUS first-order-upwind advection from the restricted coarse advecting
// velocity (scaled by s_a=1/cfac_a per face axis). M-matrix on every level -> stable in the advection-
// dominated rows. The fine residual + smoother give the exact sharp-IBM answer; this only sets the rate.
inline void buildAdvCoarse(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, CCConst U, CCConst V,
                           CCConst W, int comp, C3 e, int g, double bx, double by, double bz, double fouw,
                           double sx, double sy, double sz, double idiag) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for("peclet::flow::vmg_adv_coarse", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
    KOKKOS_LAMBDA(int x, int y, int z) {
      const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
      double cC = idiag + 2.0 * (bx + by + bz), cxm = -bx, cxp = -bx, cym = -by, cyp = -by, czm = -bz, czp = -bz;
      sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
      sadv::fou_operator_aniso(comp, x, y, z, Ua, Va, Wa, fouw, sx, sy, sz, cC, cxm, cxp, cym, cyp, czm, czp);
      AC(i) = (float)cC; AW(i) = (float)cxm; AE(i) = (float)cxp; AS(i) = (float)cym; AN(i) = (float)cyp;
      AB(i) = (float)czm; AT(i) = (float)czp;
    });

}

// CONST-COEFF anisotropic Helmholtz A = idiag*I - nu_dt*Lap (mg_const_diffusion_op_aniso_k), over the WHOLE
// extended block (the domain-BC velocity op; coarse spacing via per-axis beta). For cavity/BFS where the fine
// op is also const-coeff (no IBM stencil).
inline void buildConstAniso(FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, C3 e, double bx,
                            double by, double bz, double idiag) {
  CCExec space; const std::size_t n = (std::size_t)e.x * e.y * e.z;
  const float c = (float)(idiag + 2.0 * (bx + by + bz)), nx = (float)(-bx), ny = (float)(-by), nz = (float)(-bz);
  Kokkos::parallel_for("peclet::flow::vmg_const_aniso", Kokkos::RangePolicy<CCExec>(space, 0, n),
    KOKKOS_LAMBDA(std::size_t i) { AC(i) = c; AW(i) = nx; AE(i) = nx; AS(i) = ny; AN(i) = ny; AB(i) = nz; AT(i) = nz; });

}

// No-slip face-fold for the const-coeff MG operator (mg_diffusion_bc_fold_k): at a Dirichlet wall the
// tangential ghost is 2*u_wall - u_inner -> +beta moves onto the boundary-adjacent inner cell's diagonal
// (AC += beta); the dropped off-diagonal multiplies a held-0 ghost. Over the perp plane of face (a,s).
inline void boundaryFold(FPV AC, C3 e, int g, int a, int s, double beta) {
  CCExec space; int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3; const long sa = st[a], sb = st[b], sc = st[c];
  const int bic = (s == 0) ? g : (dims[a] - g - 1);  // boundary-adjacent inner cell along a
  Kokkos::parallel_for("peclet::flow::vmg_bc_fold", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
    KOKKOS_LAMBDA(int p0, int p1) { const long i = (long)p0 * sb + (long)p1 * sc + (long)bic * sa;
      AC(i) = (float)((double)AC(i) + beta); });

}

// Fill a non-periodic boundary ghost of a coarse correction before trilinear prolongation (mg_fill_bc_ghost_k):
// Dirichlet (outflow) -> ghost 0; Neumann (wall/inflow) -> ghost = nearest inner (zero-gradient). Plane (a,s).
inline void fillBcGhost(CCField x, C3 e, int g, int a, int s, int dirichlet) {
  CCExec space; int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3; const long sa = st[a], sb = st[b], sc = st[c]; const int na = dims[a];
  Kokkos::parallel_for("peclet::flow::vmg_fill_bc_ghost", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
    KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
      if (s == 0) { const double v = dirichlet ? 0.0 : x(base + (long)g * sa); for (int ia = 0; ia < g; ++ia) x(base + (long)ia * sa) = v; }
      else { const double v = dirichlet ? 0.0 : x(base + (long)(na - g - 1) * sa); for (int ia = na - g; ia < na; ++ia) x(base + (long)ia * sa) = v; } });

}

// zero a field on the plane at index `idx` along `axis` (held-Dirichlet boundary-face residual exclude).
inline void zeroPlane(CCField m, C3 e, int axis, int idx) {
  CCExec space; int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
  const int b = (axis + 1) % 3, c = (axis + 2) % 3; const long sa = st[axis], sb = st[b], sc = st[c];
  Kokkos::parallel_for("peclet::flow::vmg_zero_plane", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
    KOKKOS_LAMBDA(int p0, int p1) { m((long)p0 * sb + (long)p1 * sc + (long)idx * sa) = 0.0; });

}

inline void thresholdMask(CCField m, CCConst theta, double thresh) {  // m = 1 where theta < thresh (solid)
  CCExec space; std::size_t n = m.extent(0); CCField mm = m; CCConst th = theta;
  Kokkos::parallel_for("peclet::flow::vmg_threshold", Kokkos::RangePolicy<CCExec>(space, 0, n),
    KOKKOS_LAMBDA(std::size_t i) { mm(i) = (th(i) < thresh) ? 1.0 : 0.0; });

}
inline void mulMask(CCField r, CCConst m) {  // r *= m (clean-fluid residual filter)
  CCExec space; std::size_t n = r.extent(0); CCField rr = r; CCConst mm = m;
  Kokkos::parallel_for("peclet::flow::vmg_mulmask", Kokkos::RangePolicy<CCExec>(space, 0, n),
    KOKKOS_LAMBDA(std::size_t i) { rr(i) *= mm(i); });

}

// Velocity (momentum) geometric multigrid with the staircase coarse operator. All levels ghost width G=2.
class VelocityMG {
 public:
  static constexpr int G = 2;
  struct Level {
    C3 ext, inner, ratio{2, 2, 2}, cfac{1, 1, 1};
    C3 og{0, 0, 0};            // block inner origin (global red-black parity); {0,0,0} single-rank
    std::size_t n = 0;
    CCField x, rhs, res, theta, pin, resMask;
    CCField advU, advV, advW;  // restricted advecting velocity (upwind-convective coarse op; L>=1)
    FPV AC, AW, AE, AS, AN, AB, AT;
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;                       // per-level topology (decomposed)
    std::shared_ptr<GridHalo<double>> dev;   // per-level ghost exchange (ghost width G=2)
#endif
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
      if (L > 0) {  // coarse advecting velocity for the upwind-convective coarse op
        v.advU = CCField("vmg_au", v.n); v.advV = CCField("vmg_av", v.n); v.advW = CCField("vmg_aw", v.n);
      }
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("vmg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z) break;
      inner = next; cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    lv_[0].resMask = CCField("vmg_resmask0", lv_[0].n);  // level 0 only (clean-fluid exclude, staircase path)
  }
#ifdef PECLET_FLOW_MPI
  // Multi-rank velocity-MG: coarsen the GLOBAL grid 2:1 per level; each level gets its own G=2 transport-core
  // halo. No global reductions here (the velocity op is non-singular -> no mean removal, no Krylov), so the
  // fold is just fill()->exchange + the block-origin red-black parity. Single-rank (size 1) == init().
  void initMpi(int gnx, int gny, int gnz, int nLevels, MPI_Comm comm) {
    lv_.clear(); distributed_ = true;
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
      v.og = {(int)oig[0] + G, (int)oig[1] + G, (int)oig[2] + G}; v.cfac = cf;
      v.n = idx.numCellsInclGhost();
      C3 next = gs, ratio{1, 1, 1};
      if (L + 1 < nLevels) {
        if (can(gs.x)) { ratio.x = 2; next.x = gs.x / 2; }
        if (can(gs.y)) { ratio.y = 2; next.y = gs.y / 2; }
        if (can(gs.z)) { ratio.z = 2; next.z = gs.z / 2; }
      }
      v.ratio = ratio;
      v.x = CCField("vmg_x", v.n); v.rhs = CCField("vmg_rhs", v.n); v.res = CCField("vmg_res", v.n);
      v.theta = CCField("vmg_th", v.n); v.pin = CCField("vmg_pin", v.n);
      if (L > 0) { v.advU = CCField("vmg_au", v.n); v.advV = CCField("vmg_av", v.n); v.advW = CCField("vmg_aw", v.n); }
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("vmg_A", v.n);
      lv_.push_back(v);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z) break;
      gs = next; cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    lv_[0].resMask = CCField("vmg_resmask0", lv_[0].n);
  }
#endif
  int nLevels() const { return (int)lv_.size(); }
  Level& level(int L) { return lv_[L]; }

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
    usePin_ = true; useResMask_ = true;  // staircase: pin classified-solid cells + exclude the IBM band
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

  // UPWIND-CONVECTIVE coarse op (implicit-FOU): restrict the advecting velocity u/v/w (level-0 block) to
  // every coarse level (8:1 average; numerical diffusion is welcome -> keeps the M-matrix). Call ONCE per
  // Picard iteration, before buildUpwindCoarse for the 3 components (the velocity is shared/frozen at u^k).
  void restrictAdvVelocities(CCConst u0, CCConst v0, CCConst w0) {
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& cs = lv_[L]; Level& fin = lv_[L - 1];
      CCConst fu = (L == 1) ? u0 : CCConst(fin.advU), fv = (L == 1) ? v0 : CCConst(fin.advV),
              fw = (L == 1) ? w0 : CCConst(fin.advW);
      restrictAvg(cs.advU, fu, cs.ext, fin.ext, G, cs.inner, fin.ratio); fill(cs, cs.advU);
      restrictAvg(cs.advV, fv, cs.ext, fin.ext, G, cs.inner, fin.ratio); fill(cs, cs.advV);
      restrictAvg(cs.advW, fw, cs.ext, fin.ext, G, cs.inner, fin.ratio); fill(cs, cs.advW);
    }
  }
  // Build the coarse operators for component comp = aniso const-coeff diffusion + dt*FOU from the restricted
  // advecting velocity (level 0 stays the fine As_[comp] set by setFineStencil). No pin / no exclude mask --
  // the upwind M-matrix is stable; the fine residual gives the exact sharp answer. Per Picard iter, per comp.
  void buildUpwindCoarse(int comp, double nu_dt, double idiag, double fouw) {
    usePin_ = false; useResMask_ = false;  // upwind path: pure variable-coeff MG (no pin/exclude)
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x), by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      const double sx = 1.0 / (double)c.cfac.x, sy = 1.0 / (double)c.cfac.y, sz = 1.0 / (double)c.cfac.z;
      buildAdvCoarse(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, CCConst(c.advU), CCConst(c.advV),
                     CCConst(c.advW), comp, c.ext, G, bx, by, bz, fouw, sx, sy, sz, idiag);
    }
  }

  // DOMAIN-BC const-coeff path (cavity/BFS): per-face BC types {-x,+x,-y,+y,-z,+z} (0=periodic,1=wall,
  // 2=inflow,3=outflow). Enables the non-periodic fill (periodic axes wrap; non-periodic ghosts left as the
  // caller / correction set them) + the Dirichlet/Neumann prolongation ghosts.
  void setBC(const int bc[6]) {
    bcMode_ = false;
    for (int i = 0; i < 6; ++i) { bc_[i] = bc[i]; if (bc[i]) bcMode_ = true; }
  }
  // Re-impose the full velocity BC on the level-0 iterate before each smoother colour + the residual (exactly
  // as the RB-GS path does via fillVelGhosts(c,1)): the const-coeff smoother updates the held Dirichlet faces,
  // so without this the boundary corners drift (~2% vs RB-GS, as the CUDA vmg also does). With it the vel-MG
  // converges to the RB-GS fixed point. IbmSolver supplies this per component before the solve.
  void setBcApplyL0(std::function<void(CCField)> fn) { bcApplyL0_ = std::move(fn); }
  // const-coeff aniso operator + no-slip/inflow/outflow boundary fold for component comp, on EVERY level.
  // nu_dt = mu, idiag = rho/dt, h0 = 1. Rebuilt per component (the fold is component-dependent). No pin.
  // useResMask_: exclude the HELD normal-Dirichlet boundary face (a==comp, -side, wall/inflow) from
  // coarsening -- that cell's value is pinned by the BC re-imposition, so its (nonzero) residual would drive a
  // spurious coarse correction into the boundary (the ~2% drift CUDA's domain-BC vmg leaves). Excluding it
  // makes the V-cycle converge to the RB-GS fixed point (analogue of the IBM clean-fluid exclude).
  void setDomainBcOp(int comp, double nu_dt, double idiag) {
    usePin_ = false;
    Level& f = lv_[0];
    Kokkos::deep_copy(f.resMask, 1.0);
    useResMask_ = false;
    for (int s = 0; s < 1; ++s) {  // only the -side face index G lands inside the smoother range [G, ext-G)
      const int t = bc_[2 * comp + s];
      if (t == 1 || t == 2) { zeroPlane(f.resMask, f.ext, comp, G); useResMask_ = true; }
    }
    for (int L = 0; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      const double bx = nu_dt / (double)(c.cfac.x * c.cfac.x), by = nu_dt / (double)(c.cfac.y * c.cfac.y),
                   bz = nu_dt / (double)(c.cfac.z * c.cfac.z);
      buildConstAniso(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, c.ext, bx, by, bz, idiag);
      for (int f = 0; f < 6; ++f) {
        const int a = f / 2, s = f % 2;
        const double ba = (a == 0) ? bx : (a == 1) ? by : bz;
        double dval;
        if (bc_[f] == 3) dval = -ba;                              // outflow zero-gradient: every component
        else if ((bc_[f] == 1 || bc_[f] == 2) && a != comp) dval = ba;  // wall/inflow: tangential fold
        else continue;                                            // periodic, or the normal comp at a wall
        boundaryFold(c.AC, c.ext, G, a, s, dval);
      }
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
    const bool l0 = (L == 0);
    if (L + 1 == (int)lv_.size()) { smooth(lv, bottom_, l0); return; }  // velocity op non-singular -> no mean removal
    smooth(lv, pre_, l0);
    fill(lv, lv.x);
    if (l0 && bcApplyL0_) bcApplyL0_(lv.x);  // domain-BC: re-impose the velocity BC before the level-0 residual
    residualVarPin(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AW), FPC(lv.AE), FPC(lv.AS),
                   FPC(lv.AN), FPC(lv.AB), FPC(lv.AT), usePin_ ? CCConst(lv.pin) : empty_, lv.ext, G);
    const bool masked = useResMask_ && (lv.resMask.extent(0) == lv.n);  // level 0: exclude the IBM cut-cell band
    if (masked) mulMask(lv.res, CCConst(lv.resMask));
    Level& cs = lv_[L + 1];
    restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, G, cs.inner, lv.ratio);
    Kokkos::deep_copy(cs.x, 0.0);
    vcycle(L + 1);
    fill(cs, cs.x);
    fillProlongBcGhosts(cs);  // non-periodic boundary ghosts the trilinear prolong samples (domain-BC mode)
    if (masked) prolongMasked(lv.x, CCConst(cs.x), CCConst(lv.resMask), lv.ext, cs.ext, G, lv.inner, lv.ratio, 0.5);
    else prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, G, lv.inner, lv.ratio);
    smooth(lv, post_, l0);
  }
  void smooth(Level& lv, int sweeps, bool isL0) {
    const C3 og = lv.og;  // global red-black parity (block inner origin); {0,0,0} single-rank
    CCConst pin = usePin_ ? CCConst(lv.pin) : empty_;
    for (int k = 0; k < sweeps; ++k)
      for (int color = 0; color < 2; ++color) {
        fill(lv, lv.x);
        if (isL0 && bcApplyL0_) bcApplyL0_(lv.x);  // re-impose the velocity BC (held Dirichlet faces) per colour
        ibmRbgsStencilColor(lv.x, CCConst(lv.rhs), MConst(lv.AC), MConst(lv.AW), MConst(lv.AE), MConst(lv.AS),
                            MConst(lv.AN), MConst(lv.AB), MConst(lv.AT), pin, lv.ext, og, G, color);
      }
  }
  // periodic ghost fill; in domain-BC mode only the periodic axes wrap (non-periodic boundary ghosts are
  // left as the caller / correction set them -- the boundary fold + held ghost represent the wall).
  // Distributed (periodic IBM path): the per-level transport-core halo (cross-rank + periodic in one call).
  void fill(Level& lv, CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_ && !bcMode_) { lv.dev->exchange(f); return; }
#endif
    for (int a = 0; a < 3; ++a)
      if (!bcMode_ || (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)) fillAxis(lv, f, a);
  }
  // non-periodic boundary ghosts of a coarse correction before trilinear prolong (Dirichlet outflow -> 0,
  // Neumann wall/inflow -> zero-gradient).
  void fillProlongBcGhosts(Level& lv) {
    if (!bcMode_) return;
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (t == 0) continue;
        fillBcGhost(lv.x, lv.ext, G, a, s, t == 3 ? 1 : 0);
      }
  }
  void fillAxis(Level& lv, CCField f, int axis) {
    CCExec space; C3 e = lv.ext; int N3[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    int dims[3] = {e.x, e.y, e.z}; long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c]; const int N = N3[a]; CCField ff = f;
    Kokkos::parallel_for("peclet::flow::vmg_pfill", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) { const long base = (long)p0 * sb + (long)p1 * sc;
        for (int gl = 0; gl < G; ++gl) { ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
          ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa); } });

  }

 private:
  std::vector<Level> lv_;
  int pre_ = 2, post_ = 2, bottom_ = 8;
  bool usePin_ = true, useResMask_ = true;  // staircase: pin + clean-fluid exclude; upwind/domain-BC: neither
  bool bcMode_ = false; int bc_[6] = {0, 0, 0, 0, 0, 0};  // domain-BC (non-periodic) mode
  std::function<void(CCField)> bcApplyL0_;  // re-impose the velocity BC on level 0 (domain-BC mode)
  CCConst empty_;                           // zero-extent View -> "no pin / no mask" to the kernels
  bool distributed_ = false;                // multi-rank (initMpi); periodic IBM path -> fill() exchanges
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_VELOCITY_MG_HPP
