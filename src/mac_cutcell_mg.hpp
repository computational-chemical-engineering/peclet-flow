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

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ghost_projection.hpp"  // GpOverlay + gpApplyDelta (ghost-projection BiCGStab matvec)
#include "mac_bc.hpp"
#include "mac_pressure.hpp"
#include "mg_bottom_direct.hpp"              // §13: the direct preconditioner of the device bottom
#include "peclet/core/solver/graph_amg.hpp"  // decomposition-agnostic algebraic bottom solve
#include "star_elimination.hpp"  // StarOverlay + starApplyDelta (mode-B fluid-only PCG matvec)

// Multi-rank (MPI) path is opt-in: the single-GPU module never links MPI, so all distributed code
// is gated (mirrors the CUDA PECLET_FLOW_BUILD_MPI gating). When PECLET_FLOW_MPI is off, CutcellMG
// is byte-identical to before.
#ifdef PECLET_FLOW_MPI
#include <memory>
#include <optional>
#include <stdexcept>

#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/grid_redistribute.hpp"
#include "peclet/core/decomp/redistribute_topology.hpp"
#include "peclet/core/decomp/stage_comm.hpp"
#include "peclet/core/decomp/stage_target.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"
#include "policy.hpp"
#endif

namespace peclet::flow {

#ifdef PECLET_FLOW_MPI
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;
#endif

// Operator/level storage precision. Float (the CUDA-era default) breaks the singular row-sum
// identity A*1=0 at ~eps_f relative per row; under high face-coefficient contrast (order-2
// apertures span 3 decades) the defect on mixed large+tiny rows is amplified to ~eps_f*contrast,
// which perturbs the near-null vector the mean-removal deflation assumes and floors/rebounds the
// CG-family drivers near r/r0 ~ 1e-6 (see doc/collocated_paper_plan.md row 55). The bottom AMG
// already re-sums its diagonal in double for exactly this reason. -DPECLET_FLOW_OPERATOR_DOUBLE
// switches the whole hierarchy to double (A/B instrument; ~2x operator memory).
//
// EVERY store into an FPV/FV must go through MReal, or the switch is silently partial. WO-M found
// three families of hard `(float)` casts that survived the templating and clamped their operator to
// float even in a double build: `IbmSolver::buildAdvStencil` / `buildAdvStencilVar` (the implicit-
// FOU momentum stencil), `IbmSolver::addDragDiagonal` (the CFD-DEM face drag — its symptom was a
// porous steady drag balance stuck at 4.8e-8 in BOTH builds), and five sites in
// `mac_velocity_mg.hpp` (the velocity-MG staircase / upwind-coarse / const-aniso operators and the
// no-slip fold). They now cast to MReal, which is byte-identical when MReal is float.
#ifdef PECLET_FLOW_OPERATOR_DOUBLE
using MReal = double;
#else
using MReal = float;  // operator storage = CUDA mreal
#endif
using FPV = Kokkos::View<MReal*, CCMem>;
using FPC = Kokkos::View<const MReal*, CCMem>;
// The FP32 V-cycle's storage and arithmetic type (doc/vof_projection_cost_design.md §4.4.2).
using VReal = float;  // PRECISION-EXEMPT: FP32 V-cycle preconditioner below the exact FP64 Krylov
                      // (doc/vof_projection_cost_design.md §4)

// coarsen staggered face openness: each coarse face = average of the ratio_b*ratio_c fine sub-faces
// it spans (mg_coarsen_open_avg_k port). gc/gf: coarse/fine block ghost widths (they can differ —
// CA-eligible coarse levels carry g=2). The cell body is the A0 form (§5.2).
template <class CV, class FV>
KOKKOS_INLINE_FUNCTION void coarsenOpenAvgCell(const CV& oxc, const CV& oyc, const CV& ozc,
                                               const FV& oxf, const FV& oyf, const FV& ozf, C3 cext,
                                               C3 fext, int gc, int gf, C3 ratio, int icx, int icy,
                                               int icz) {
  const int rx = ratio.x, ry = ratio.y, rz = ratio.z;
  const int fx0 = rx * icx + gf, fy0 = ry * icy + gf, fz0 = rz * icz + gf;
  const long fsy = fext.x, fsz = (long)fext.x * fext.y;
  auto F = [&](const FV& T, int x, int y, int z) {
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
}
inline void coarsenOpenAvg(CCField oxc, CCField oyc, CCField ozc, CCConst oxf, CCConst oyf,
                           CCConst ozf, C3 cext, C3 fext, int gc, int gf, C3 cinner, C3 ratio) {
  ccFor3(
      "peclet::flow::coarsen_open", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        coarsenOpenAvgCell(oxc, oyc, ozc, oxf, oyf, ozf, cext, fext, gc, gf, ratio, icx, icy, icz);
      });
}

// --- WO-R2 item 1: the OUTFLOW face coefficient on a variable-density operator ----------------
//
// `applyBoundaryOpenness` used to re-impose the literal 1.0 at a Dirichlet (outflow) domain face on
// EVERY level, which under `varRho` overwrote `buildRhoCoeff`'s `open_f * rho0/rho_f` with 1 and
// made the operator row disagree with the projection correction by the full density ratio (WO-R's
// headline defect: the Nusselt film's low-side outlet, `max|w|` 1.455 against a film `u_max` of
// 0.312). The face index convention makes the two sides ASYMMETRIC:
//   * the LOW  domain face of an axis is the INNER index `g`      -> `buildRhoCoeff` (and, on a
//     coarse level, `coarsenOpenAvg`) already wrote the right value there; the fix is simply not
//     to overwrite it.
//   * the HIGH domain face is the GHOST index `dims-g`            -> no kernel writes it, and the
//     periodic/halo ghost fill wraps the opposite boundary's value into it. It must be saved
//     before the fill and restored after (level 0, from the caller's field), and coarsened from
//     the fine level's own restored plane (coarse levels).
// These three helpers do the save / restore / coarsen of ONE domain-face plane. They are only
// reached when the caller sets `setOutflowCoefficient(true)` (the varRho pressure build); with it
// off the literal-1.0 path is untouched and byte-identical.
inline void mgSaveFacePlane(CCField dst, CCConst src, C3 e, int g, int a, int s) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int bf = (s == 0) ? g : (dims[a] - g);
  const int db = dims[b];
  using MD = MDRange2<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::mg_save_face_plane", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        dst((long)p0 + (long)p1 * db) = src((long)p0 * sb + (long)p1 * sc + (long)bf * sa);
      });
}
inline void mgRestoreFacePlane(CCField dst, CCConst src, C3 e, int g, int a, int s) {
  CCExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int bf = (s == 0) ? g : (dims[a] - g);
  const int db = dims[b];
  using MD = MDRange2<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::mg_restore_face_plane", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        dst((long)p0 * sb + (long)p1 * sc + (long)bf * sa) = src((long)p0 + (long)p1 * db);
      });
}
// Coarse domain-face plane = the area average of the ratio_b*ratio_c fine sub-faces it spans —
// the same rule `coarsenOpenAvg` applies to every interior face, restricted to the one plane the
// coarsening loop cannot reach (the high side lives on a ghost index). The coarse plane's own
// TRANSVERSE ghost ring is filled by clamping the fine transverse index into range: it is read
// only by the redundant ring rows of a distributed CA (g=2) level's smoother, and it is exact
// whenever the coefficient is constant along the outlet (every constant-density case, and every
// ratio-1 case, so the byte-identity and MPI gates are unaffected).
inline void mgCoarsenFacePlane(CCField oc, CCConst of, C3 cext, C3 fext, int gc, int gf, C3 cinner,
                               C3 ratio, int a, int s) {
  CCExec space;
  int cd[3] = {cext.x, cext.y, cext.z}, fd[3] = {fext.x, fext.y, fext.z};
  int ci[3] = {cinner.x, cinner.y, cinner.z}, rt[3] = {ratio.x, ratio.y, ratio.z};
  long cst[3] = {1, (long)cext.x, (long)cext.x * cext.y};
  long fst[3] = {1, (long)fext.x, (long)fext.x * fext.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long csa = cst[a], csb = cst[b], csc = cst[c];
  const long fsa = fst[a], fsb = fst[b], fsc = fst[c];
  const int cbf = (s == 0) ? gc : (cd[a] - gc);
  const int fbf = (s == 0) ? gf : (fd[a] - gf);
  const int rb = rt[b], rc = rt[c];
  const int fdb = fd[b], fdc = fd[c];
  const double inv = 1.0 / (double)(rb * rc);
  using MD = MDRange2<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::mg_coarsen_face_plane", MD(space, {0, 0}, {cd[b], cd[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const int ib = p0 - gc, ic = p1 - gc;  // coarse INNER indices (negative in the ghost ring)
        double sum = 0.0;
        for (int q0 = 0; q0 < rb; ++q0)
          for (int q1 = 0; q1 < rc; ++q1) {
            int fb = rb * ib + gf + q0, fc = rc * ic + gf + q1;
            fb = fb < 0 ? 0 : (fb >= fdb ? fdb - 1 : fb);
            fc = fc < 0 ? 0 : (fc >= fdc ? fdc - 1 : fc);
            sum += of((long)fb * fsb + (long)fc * fsc + (long)fbf * fsa);
          }
        oc((long)p0 * csb + (long)p1 * csc + (long)cbf * csa) = sum * inv;
      });
  (void)ci;
}

// residual r = b - A x for the float operator (mg_residual_var_k port). A0 cell body (§5.2).
template <class RV, class XV, class BV, class OpV>
KOKKOS_INLINE_FUNCTION void cutcellResidualCell(const RV& r, const XV& x, const BV& b,
                                                const OpV& AC, const OpV& AW, const OpV& AE,
                                                const OpV& AS, const OpV& AN, const OpV& AB,
                                                const OpV& AT, long i, long xp, long xm, long yp,
                                                long ym, long zp, long zm) {
  const double Ax = (double)AC(i) * x(i) + (double)AE(i) * x(xp) + (double)AW(i) * x(xm) +
                    (double)AN(i) * x(yp) + (double)AS(i) * x(ym) + (double)AT(i) * x(zp) +
                    (double)AB(i) * x(zm);
  r(i) = b(i) - Ax;
}
inline void residualCutcell(CCField r, CCConst x, CCConst b, FPC AC, FPC AW, FPC AE, FPC AS, FPC AN,
                            FPC AB, FPC AT, C3 e, int g) {
  ccFor3(
      "peclet::flow::cc_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellResidualCell(r, x, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                            i + sz, i - sz);
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
        cutcellResidualCell(r, x, b, AC, AW, AE, AS, AN, AB, AT, i, i + sx, i - sx, i + sy, i - sy,
                            i + sz, i - sz);
      });
}

// Face-form siblings of residualCutcell / residualCutcellBox (A2, see mac_pressure.hpp).
inline void residualCutcellFace(CCField r, CCConst x, CCConst b, FPC AC, FPC AFX, FPC AFY, FPC AFZ,
                                C3 e, int g) {
  ccFor3(
      "peclet::flow::cc_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellResidualFaceCell(r, x, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                i - sy, i + sz, i - sz);
      });
}
inline void residualCutcellBoxFace(CCField r, CCConst x, CCConst b, FPC AC, FPC AFX, FPC AFY,
                                   FPC AFZ, C3 e, C3 rlo, C3 rhi, C3 slo, C3 shi) {
  if (rhi.x <= rlo.x || rhi.y <= rlo.y || rhi.z <= rlo.z)
    return;
  ccFor3(
      "peclet::flow::cc_residual_box", rlo, rhi, KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (lx >= slo.x && lx < shi.x && ly >= slo.y && ly < shi.y && lz >= slo.z && lz < shi.z)
          return;  // inside the skip box (already done by the interior pass)
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        cutcellResidualFaceCell(r, x, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + sx, i - sx, i + sy,
                                i - sy, i + sz, i - sz);
      });
}

// --- A3: fused periodic wrap (doc/vof_step_performance_design.md §4.3, §5.4) -------------------
// On a single-rank level whose ghosts a periodic fill would only COPY from the inner cells (no
// outflow face, no overlay on the solve), the smoother, the residual and the Krylov matvec read a
// boundary cell's periodic neighbour directly through the wrapped index instead of from a ghost
// filled by a launch before each pass: per axis a, for inner coordinate k in [g, g+n_a),
//   minus neighbour = (k == g)         ? i + (n_a-1) s_a : i - s_a
//   plus  neighbour = (k == g+n_a-1)   ? i - (n_a-1) s_a : i + s_a.
// That is the very cell the fill copies into the ghost, so a read-only pass (residual, matvec)
// reads the identical value. A red-black colour pass does too exactly when every inner dimension is
// EVEN: the wrapped neighbour then has the other colour, which this pass does not update, so it
// still holds the value the fill copied before the pass (the caller checks the parity). Only the
// field being smoothed or applied is wrapped; the coefficients are static over a solve and are read
// at their ghost indices as before (AFX(i+sx) at the high ghost holds the periodic face, and a wall
// face's coefficient is 0 there, multiplying the wrapped value exactly as it multiplied the ghost).
struct CcNbrs {
  long xp, xm, yp, ym, zp, zm;
};
KOKKOS_INLINE_FUNCTION CcNbrs ccWrapNbrs(int lx, int ly, int lz, C3 n, int g, long i, long sy,
                                         long sz) {
  CcNbrs w;
  w.xm = (lx == g) ? i + (long)(n.x - 1) : i - 1;
  w.xp = (lx == g + n.x - 1) ? i - (long)(n.x - 1) : i + 1;
  w.ym = (ly == g) ? i + (long)(n.y - 1) * sy : i - sy;
  w.yp = (ly == g + n.y - 1) ? i - (long)(n.y - 1) * sy : i + sy;
  w.zm = (lz == g) ? i + (long)(n.z - 1) * sz : i - sz;
  w.zp = (lz == g + n.z - 1) ? i - (long)(n.z - 1) * sz : i + sz;
  return w;
}
// S (doc/vof_projection_cost_design.md §9, WO-S): the host row of an A3 wrap kernel with the row
// ends PEELED. Per row the y/z neighbour offsets once (ccWrapNbrs' selects, loop-invariant in a
// row); the first and last cell of the row take ccWrapNbrs (today's body); the interior reads
// i +- 1, so the x-neighbour loads are affine instead of gathers. Same cells, same body, same
// operands: bitwise to the ccWrapNbrs form. `cell(i, xp, xm, yp, ym, zp, zm)` is called for the
// cells lx = lx0, lx0 + step, ... <= g + n.x - 1 of the row (step 2: one colour of the smoother).
template <class Cell>
KOKKOS_INLINE_FUNCTION void ccWrapRowPeeled(int ly, int lz, C3 e, C3 n, int g, int lx0, int step,
                                            const Cell& cell) {
  const long sy = e.x, sz = (long)e.x * e.y;
  const long base = (long)ly * sy + (long)lz * sz;
  const long dym = (ly == g) ? (long)(n.y - 1) * sy : -sy;
  const long dyp = (ly == g + n.y - 1) ? -(long)(n.y - 1) * sy : sy;
  const long dzm = (lz == g) ? (long)(n.z - 1) * sz : -sz;
  const long dzp = (lz == g + n.z - 1) ? -(long)(n.z - 1) * sz : sz;
  const int lxLast = g + n.x - 1;
  if (lx0 > lxLast)
    return;
  const int last = lx0 + ((lxLast - lx0) / step) * step;  // the row's last cell of the pass
  auto scalar = [&](int lx) {
    const long i = (long)lx + base;
    const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
    cell(i, w.xp, w.xm, w.yp, w.ym, w.zp, w.zm);
  };
  int lo = lx0, hi = last;  // inclusive
  if (lo == g) {
    scalar(lo);
    lo += step;
  }
  const bool peelLast = hi == lxLast && hi >= lo;
  if (peelLast)
    hi -= step;
  PECLET_FLOW_OMP_SIMD  // same as the unpeeled loop: independent cells
      for (int lx = lo; lx <= hi; lx += step) {
    const long i = (long)lx + base;
    cell(i, i + 1, i - 1, i + dyp, i + dym, i + dzp, i + dzm);
  }
  if (peelLast)
    scalar(lxLast);
}
// Every inner cell of a wrap kernel on a host backend, row-peeled (ccFor3's host structure: the
// (y, z) row partition, the serial cutoff).
template <class Cell>
inline void ccWrapForPeeled(const char* name, C3 e, C3 n, int g, const Cell& cell) {
  CCExec space;
  const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
  const long rows = (long)nyi * nzi;
  auto row = [=](long t) {
    ccWrapRowPeeled(g + (int)(t % nyi), g + (int)(t / nyi), e, n, g, g, 1, cell);
  };
  if (hostRunSerial(rows * (e.x - 2 * g))) {
    for (long t = 0; t < rows; ++t)
      row(t);
    return;
  }
  Kokkos::parallel_for(name, Kokkos::RangePolicy<CCExec>(space, 0, rows), row);
}
// cutcellSmoothColorFace with wrapped neighbour reads (n = the level's inner dims, all even).
inline void cutcellSmoothColorFaceWrap(CCField phi, CCConst b, FPC AC, FPC AFX, FPC AFY, FPC AFZ,
                                       C3 e, C3 n, C3 og, int g, int color) {
  CCExec space;
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    const long cells = (long)nyi * nzi * (e.x - 2 * g);
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      // S: the row ends peeled (same-colour cells are independent: ccFor3's contract)
      ccWrapRowPeeled(ly, lz, e, n, g, g + ((P ^ (g & 1)) & 1), 2,
                      [&](long i, long xp, long xm, long yp, long ym, long zp, long zm) {
                        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, xp, xm, yp,
                                              ym, zp, zm);
                      });
    };
    if (hostRunSerial(cells)) {  // coarse MG level: the fork/join costs more than the sweep
      for (long t = 0; t < (long)nyi * nzi; ++t)
        pencil(t);
      return;
    }
    Kokkos::parallel_for("peclet::flow::cc_smooth",
                         Kokkos::RangePolicy<CCExec>(space, 0, (long)nyi * nzi), pencil);
    return;
  }
  using MD = MDRange3<CCExec>;
  Kokkos::parallel_for(
      "peclet::flow::cc_smooth", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
          return;
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym,
                              w.zp, w.zm);
      });
}
// residualCutcellFace with wrapped reads of x.
inline void residualCutcellFaceWrap(CCField r, CCConst x, CCConst b, FPC AC, FPC AFX, FPC AFY,
                                    FPC AFZ, C3 e, C3 n, int g) {
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {  // S
    const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
    ccWrapForPeeled("peclet::flow::cc_residual", e, n, g,
                    [=](long i, long xp, long xm, long yp, long ym, long zp, long zm) {
                      cutcellResidualFaceCell(r, x, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, xp, xm, yp,
                                              ym, zp, zm);
                    });
    return;
  }
  ccFor3(
      "peclet::flow::cc_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        cutcellResidualFaceCell(r, x, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym,
                                w.zp, w.zm);
      });
}
// applyCutcellOpFace / applyCutcellOpExact with wrapped reads of x (the level-0 Krylov matvec).
inline void applyCutcellOpFaceWrap(CCField y, CCConst x, FPC AC, FPC AFX, FPC AFY, FPC AFZ, C3 e,
                                   C3 n, int g) {
  ccFor3(
      "peclet::flow::cc_apply", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        y(i) = cutcellApplyFaceCell(x, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym,
                                    w.zp, w.zm);
      });
}
inline void applyCutcellOpExactWrap(CCField y, CCConst x, CCConst ox, CCConst oy, CCConst oz, C3 e,
                                    C3 n, int g, double gfx, double gfy, double gfz) {
  if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {  // S
    const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
    ccWrapForPeeled("peclet::flow::cc_apply_exact", e, n, g,
                    [=](long i, long xp, long xm, long yp, long ym, long zp, long zm) {
                      y(i) = cutcellApplyExactCell(x, ox, oy, oz, i, sx, sy, sz, xp, xm, yp, ym, zp,
                                                   zm, gfx, gfy, gfz);
                    });
    return;
  }
  ccFor3(
      "peclet::flow::cc_apply_exact", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        y(i) = cutcellApplyExactCell(x, ox, oy, oz, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym, w.zp,
                                     w.zm, gfx, gfy, gfz);
      });
}

// average restriction (coarse = mean of ratio^3 fine children; mg_restrict_k) + trilinear
// prolongation (added to fine; mg_prolong_k). Both over inner cells. gc/gf: coarse/fine block
// ghost widths (CA-eligible coarse levels carry g=2, so they can differ across one transfer).
// A0 cell bodies (§5.2).
template <class CV, class FV>
KOKKOS_INLINE_FUNCTION void restrictAvgCell(const CV& coarse, const FV& fine, C3 cext, C3 fext,
                                            int gc, int gf, C3 ratio, int icx, int icy, int icz) {
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
}
template <class FV, class CV>
KOKKOS_INLINE_FUNCTION void prolongAddCell(const FV& fine, const CV& coarse, C3 fext, C3 cext,
                                           int gf, int gc, C3 ratio, int ifx, int ify, int ifz) {
  // The coarse sample coordinate is 0.5*ifine - 0.25 + gc on a coarsened axis (ratio 2) and
  // ifine + gc on a kept one (ratio 1); its floor is the lower coarse index and its fraction the
  // weight. §14 H-2: both in integer arithmetic. The coordinate is exact in double, so its floor
  // and fraction are exactly
  //   ratio 2: x0 = (i >> 1) + gc - 1 + (i & 1),  w = 0.25 (i odd) or 0.75 (i even);
  //   ratio 1: x0 = i + gc,                        w = 0,
  // and the interpolation below (textually unchanged) sees the same operands -- bit-identical,
  // without three floors and six FP <-> int conversions per fine cell.
  const int x0 = (ratio.x == 2) ? (ifx >> 1) + gc - 1 + (ifx & 1) : ifx + gc;
  const int y0 = (ratio.y == 2) ? (ify >> 1) + gc - 1 + (ify & 1) : ify + gc;
  const int z0 = (ratio.z == 2) ? (ifz >> 1) + gc - 1 + (ifz & 1) : ifz + gc;
  const double wx = (ratio.x == 2) ? ((ifx & 1) ? 0.25 : 0.75) : 0.0;
  const double wy = (ratio.y == 2) ? ((ify & 1) ? 0.25 : 0.75) : 0.0;
  const double wz = (ratio.z == 2) ? ((ifz & 1) ? 0.25 : 0.75) : 0.0;
  const long sy = cext.x, sz = (long)cext.x * cext.y;
  auto C = [&](int xx, int yy, int zz) { return coarse((long)xx + (long)yy * sy + (long)zz * sz); };
  const double c00 = C(x0, y0, z0) * (1 - wx) + C(x0 + 1, y0, z0) * wx;
  const double c10 = C(x0, y0 + 1, z0) * (1 - wx) + C(x0 + 1, y0 + 1, z0) * wx;
  const double c01 = C(x0, y0, z0 + 1) * (1 - wx) + C(x0 + 1, y0, z0 + 1) * wx;
  const double c11 = C(x0, y0 + 1, z0 + 1) * (1 - wx) + C(x0 + 1, y0 + 1, z0 + 1) * wx;
  const double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
  const long fi =
      (long)(ifx + gf) + (long)(ify + gf) * fext.x + (long)(ifz + gf) * (long)fext.x * fext.y;
  fine(fi) += c0 * (1 - wz) + c1 * wz;
}
// The subtract half of CutcellMG::removeMean (A0 cell body).
template <class FV, class AV>
KOKKOS_INLINE_FUNCTION void meanSubtractCell(const FV& f, const AV& ac, long i, double mean) {
  if (ac(i) > 1e-30f)
    f(i) -= mean;
}
inline void restrictAvg(CCField coarse, CCConst fine, C3 cext, C3 fext, int gc, int gf, C3 cinner,
                        C3 ratio) {
  ccFor3(
      "peclet::flow::restrict", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        restrictAvgCell(coarse, fine, cext, fext, gc, gf, ratio, icx, icy, icz);
      });
}
// A5 (doc/vof_step_performance_design.md §5.5): restrictAvg that also zeroes the coarse ITERATE on
// the same inner cells in the same kernel, replacing the full-array zero fill of `cs.x` the V-cycle
// launched after the restriction. The coarse ghosts are no longer zeroed; every consumer fills,
// wrap-reads or overwrites them first (the smoother and the residual fill or wrap before reading,
// GraphAMG writes the whole array). VelocityMG keeps the plain restrictAvg.
inline void restrictAvgZeroX(CCField coarse, CCField coarseX, CCConst fine, C3 cext, C3 fext,
                             int gc, int gf, C3 cinner, C3 ratio) {
  ccFor3(
      "peclet::flow::restrict", C3{0, 0, 0}, C3{cinner.x, cinner.y, cinner.z},
      KOKKOS_LAMBDA(int icx, int icy, int icz) {
        restrictAvgCell(coarse, fine, cext, fext, gc, gf, ratio, icx, icy, icz);
        coarseX((long)(icx + gc) + (long)(icy + gc) * cext.x +
                (long)(icz + gc) * (long)cext.x * cext.y) = 0.0;
      });
}
// D (doc/vof_projection_cost_design.md §4.4.2): the FP32 V-cycle's face weights
// WX = fl32(t_x), t_x = ox * gfx (POSITIVE; likewise y, z) over [0, ext), from the operands of
// cutcellBuildFaceOpCell. With `fp64` the same kernel also writes the FP64 face form with that
// cell body, unchanged (setOpenness's build); without it only the weights (the precision setter's
// rebuild from the stored openness). Returns 1 when a face weight t of the box violates
// t == 0 || 1e-30 <= t <= 1e30 (the face-range check, §4.4.1 (5)), else 0. One max reduction.
template <class OpV, class WV>
inline int buildCutcellOpFaceFp32(OpV AC, OpV AFX, OpV AFY, OpV AFZ, WV WX, WV WY, WV WZ,
                                  CCConst ox, CCConst oy, CCConst oz, C3 e, int gb, double gfx,
                                  double gfy, double gfz, bool fp64) {
  int bad = 0;
  ccReduce3(
      "peclet::flow::cc_build_op", C3{0, 0, 0}, C3{e.x, e.y, e.z},
      KOKKOS_LAMBDA(int lx, int ly, int lz, int& acc) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        if (fp64) {
          const bool diag =
              lx >= gb && lx < e.x - gb && ly >= gb && ly < e.y - gb && lz >= gb && lz < e.z - gb;
          cutcellBuildFaceOpCell(AC, AFX, AFY, AFZ, ox, oy, oz, i, sx, sy, sz, gfx, gfy, gfz, diag);
        }
        const double tw = ox(i) * gfx;
        const double ts = oy(i) * gfy;
        const double tb = oz(i) * gfz;
        WX(i) = (VReal)tw;
        WY(i) = (VReal)ts;
        WZ(i) = (VReal)tb;
        const bool okw = tw == 0.0 || (tw >= 1e-30 && tw <= 1e30);
        const bool oks = ts == 0.0 || (ts >= 1e-30 && ts <= 1e30);
        const bool okb = tb == 0.0 || (tb >= 1e-30 && tb <= 1e30);
        if (!(okw && oks && okb))
          acc = 1;
      },
      Kokkos::Max<int>(bad));
  return bad > 0 ? 1 : 0;  // the identity of Max<int> is INT_MIN: no violating face leaves it
}
inline void prolongAdd(CCField fine, CCConst coarse, C3 fext, C3 cext, int gf, int gc, C3 finner,
                       C3 ratio) {
  ccFor3(
      "peclet::flow::prolong", C3{0, 0, 0}, C3{finner.x, finner.y, finner.z},
      KOKKOS_LAMBDA(int ifx, int ify, int ifz) {
        prolongAddCell(fine, coarse, fext, cext, gf, gc, ratio, ifx, ify, ifz);
      });
}

}  // namespace peclet::flow
#include "mg_fp32_vcycle.hpp"  // D: the FP32 V-cycle kernels (doc/vof_projection_cost_design.md §4)
namespace peclet::flow {

// --- The direct bottom solve (doc/vof_step_performance_design.md §5.7, §5.14, §13, §14 H-1) -----
// On every backend (host backends since §14 H-1) an ELIGIBLE agglomerated bottom (single rank,
// singular operator, <= 8192 inner cells, 1-64 fluid components (BottomLabelKernel), an axis whose
// planes hold <= 192 cells; CutcellMG::directBottomIneligible) is solved in ONE launch instead of
// the host GraphAMG round trip: a single team runs flexible CG (Polak-Ribiere) in FP64 on the
// bottom level's own operator, preconditioned by the block-tridiagonal FP32 direct factor of
// mg_bottom_direct.hpp (§13; factored by its own launch once per operator change), to a relative
// infinity-norm residual of tau = 1e-5 (E3), capped at 100 iterations. The constants are fixed, not
// setters. Ghost width 1 (single rank). B1's V-cycle preconditioner over a geometric sub-hierarchy
// (§5.7) was retired by §13 D-4; its code is in the history (the WO-6 / WO-11 commits).
inline constexpr long kBottomMaxCells = 8192;
// tau: 1e-5 since E3 (§7 WO-6, Q2): 1e-8, 1e-6 and 1e-5 give the SAME outer iteration count on
// every one of the 50 bubble-column steps (653 total).
inline constexpr double kBottomTau = 1e-5;
inline constexpr int kBottomMaxComponents = 64;  // B1b (§5.14): more -> GraphAMG
inline constexpr int kBottomCap = 100;
// §14 H-1: the team size of the solve launch on a HOST backend, min(this, team_size_max) -- one
// Zen CCX (the factor launch: kBottomHostFactorTeam). The factor and M are bitwise independent of T
// by construction (§13.2), and on a host backend the FCG's reductions are single-lane
// (BottomKernel::teamSum / teamMax), so the host bottom's bits do not depend on T or
// OMP_NUM_THREADS; T affects speed only.
inline constexpr int kBottomHostTeam = 8;
// §14 H-1, A(b): the FACTOR launch's host team size. The host schedule (mg_bottom_direct.hpp) has
// ~2.5 barriers per tile and one-thread diagonal tiles, so it scales little past two threads:
// bench_bottom_factor (16x12x8 walls-y, P = 12, b = 128) on a loaded 24-core Zen 3, ms per factor,
// T = 1 / 2 / 3 / 4 / 8: 1.44 / 1.09 / 0.93-1.79 / 0.86-1.70 / 1.43-1.88 -- two is the fastest
// size that is also stable under load. Bitwise free, like kBottomHostTeam. Genoa (Snellius job
// 27783375, 1 task x 24 cores, 30 reps, median, walls-y P12 b128): T = 1 / 2 / 4 / 6 / 8 / 24:
// 2.10 / 1.51 / 1.24 / 1.08 / 1.20 / 2.46 ms -- six is the best there (Q-H3), hence 6.
inline constexpr int kBottomHostFactorTeam = 6;

struct BottomLevel {
  C3 ext{0, 0, 0}, inner{0, 0, 0};
  FPC AC, AFX, AFY, AFZ;
};

// B1b (§5.14): the fluid components of the bottom level. Deterministic min-label propagation over
// the faces whose coefficient is > 0 (the face form stores the band sign: AF < 0) between two fluid
// cells (AC > 1e-30, removeMean's predicate), iterated to its fixed point inside one team kernel:
// every fluid cell ends with the smallest flat index of its component, whatever the update order
// (Jacobi sweeps between two buffers here). Then, in flat (label) order, one thread numbers the
// components 0, 1, ... and counts their cells. Solid cells get -1. Out: comp (ids), cnt (cells per
// component, the first kBottomMaxComponents), nc(0) = the number of components. Geometry time only.
struct BottomLabelKernel {
  using Member = Kokkos::TeamPolicy<CCExec>::member_type;
  C3 ext{0, 0, 0}, inner{0, 0, 0};
  FPC AC, AFX, AFY, AFZ;
  Kokkos::View<int*, CCMem> comp, tmp;
  Kokkos::View<long*, CCMem> cnt;
  Kokkos::View<int*, CCMem> nc;
  // §13.4.2: with a slow axis (slow >= 0), each component's last plane kc(c) (the largest
  // coordinate along `slow` among its cells) and aug(c) = 1/m_c, m_c = its cells in that plane.
  int slow = -1;
  Kokkos::View<int*, CCMem> kc;
  Kokkos::View<double*, CCMem> aug;
  KOKKOS_INLINE_FUNCTION long cell(int q, int& lx, int& ly, int& lz) const {
    const int t = q / inner.x;
    lx = 1 + (q - t * inner.x);
    lz = t / inner.y;
    ly = 1 + (t - lz * inner.y);
    lz += 1;
    return (long)lx + (long)ly * ext.x + (long)lz * (long)ext.x * ext.y;
  }
  // one Jacobi sweep src -> dst; returns the number of labels that moved
  KOKKOS_INLINE_FUNCTION int sweep(const Member& t, const Kokkos::View<int*, CCMem>& src,
                                   const Kokkos::View<int*, CCMem>& dst) const {
    const int ninner = inner.x * inner.y * inner.z;
    const long sy = ext.x, sz = (long)ext.x * ext.y;
    int moved = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(t, ninner),
        [&](int q, int& acc) {
          int lx, ly, lz;
          const long i = cell(q, lx, ly, lz);
          int m = src(i);
          if (m >= 0) {
            const CcNbrs w = ccWrapNbrs(lx, ly, lz, inner, 1, i, sy, sz);
            const long nb[6] = {w.xp, w.xm, w.yp, w.ym, w.zp, w.zm};
            const double cf[6] = {(double)AFX(i + 1), (double)AFX(i),      (double)AFY(i + sy),
                                  (double)AFY(i),     (double)AFZ(i + sz), (double)AFZ(i)};
            for (int f = 0; f < 6; ++f) {
              const int lj = src(nb[f]);
              if (cf[f] < 0.0 && lj >= 0 && lj < m)
                m = lj;
            }
            if (m < src(i))
              acc += 1;
          }
          dst(i) = m;
        },
        moved);  // a sum (the identity of Max<int> is INT_MIN, not 0)
    return moved;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& t) const {
    const int ninner = inner.x * inner.y * inner.z;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ninner), [&](int q) {
      int lx, ly, lz;
      const long i = cell(q, lx, ly, lz);
      comp(i) = (AC(i) > 1e-30f) ? (int)i : -1;
    });
    t.team_barrier();
    for (;;) {  // two sweeps per round, so the labels end in `comp`
      sweep(t, comp, tmp);
      t.team_barrier();
      const int moved = sweep(t, tmp, comp);
      t.team_barrier();
      if (!moved)
        break;
    }
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      int n = 0;
      for (int q = 0; q < ninner; ++q) {  // number the roots in label (flat-index) order
        int lx, ly, lz;
        const long i = cell(q, lx, ly, lz);
        if (comp(i) == (int)i)
          tmp(i) = n++;
      }
      for (int c = 0; c < kBottomMaxComponents; ++c)
        cnt(c) = 0;
      for (int q = 0; q < ninner; ++q) {
        int lx, ly, lz;
        const long i = cell(q, lx, ly, lz);
        if (comp(i) >= 0) {
          const int id = tmp(comp(i));
          comp(i) = id;
          if (id < kBottomMaxComponents)
            cnt(id) += 1;
        }
      }
      nc(0) = n;
      if (slow >= 0) {
        long m[kBottomMaxComponents];
        for (int c = 0; c < kBottomMaxComponents; ++c) {
          kc(c) = -1;
          m[c] = 0;
        }
        for (int pass = 0; pass < 2; ++pass)
          for (int q = 0; q < ninner; ++q) {
            int lx, ly, lz;
            const long i = cell(q, lx, ly, lz);
            const int c = comp(i);
            if (c < 0 || c >= kBottomMaxComponents)
              continue;
            const int k = (slow == 0 ? lx : (slow == 1 ? ly : lz)) - 1;
            if (pass == 0 && k > kc(c))
              kc(c) = k;
            else if (pass == 1 && k == kc(c))
              m[c] += 1;
          }
        for (int c = 0; c < kBottomMaxComponents; ++c)
          aug(c) = m[c] > 0 ? 1.0 / (double)m[c] : 0.0;
      }
    });
  }
};

// The bottom solve kernel: one TeamPolicy(1, T) launch per bottom solve. FCG in FP64 on the stored
// operator (A0 cell bodies); M = the direct factor `dir` (FacReal = float in production), whose
// dir.stat(0) = 0 (a factor that failed even with the largest shift) takes the failure path.
template <class FacReal = float>  // PRECISION-EXEMPT: the FP32 direct factor under an FP64 FCG
struct BottomKernel {
  using Member = Kokkos::TeamPolicy<CCExec>::member_type;
  BottomDirect<FacReal> dir;
  BottomLevel lv;
  CCField x, b, r, p, z, zp, ap;  // the bottom's solution and rhs (its x, rhs), FCG vectors
  int precondOnly = 0;            // test hook: z = M(b) and nothing else
  int noMeanForTest = 0;          // test hook: M without the component means' removal
  // B1b (§5.14): the bottom's fluid components -- comp(i) in [0, nc) on fluid cells, -1 on cells
  // with AC <= 1e-30 (which keep x = 0) -- their cell counts, and a scratch for their means.
  int nc = 1;
  Kokkos::View<const int*, CCMem> comp;
  Kokkos::View<const long*, CCMem> ccnt;
  Kokkos::View<double*, CCMem> cmean;
  Kokkos::View<double*, CCMem> ks;  // the Krylov slots; ks(flagSlot) = 1 on a non-finite scalar
  int flagSlot = 0;
  Kokkos::View<int*, CCMem> info;  // info(0) = inner iterations of the last solve

  // Every phase is one TeamThreadRange pass over the level followed by a barrier; the matvec reads
  // periodic neighbours through wrapped indices (the A3 rule, §4.3), with 32-bit index arithmetic.
  KOKKOS_INLINE_FUNCTION static int nInner(const BottomLevel& l) {
    return l.inner.x * l.inner.y * l.inner.z;
  }
  KOKKOS_INLINE_FUNCTION static int nExt(const BottomLevel& l) {
    return l.ext.x * l.ext.y * l.ext.z;
  }
  // inner cell q (x fastest) -> local coordinates (ghost width 1) and the flat index
  KOKKOS_INLINE_FUNCTION static long cell(const BottomLevel& l, int q, int& lx, int& ly, int& lz) {
    const int nx = l.inner.x, ny = l.inner.y;
    const int t = q / nx;
    lx = 1 + (q - t * nx);
    lz = t / ny;
    ly = 1 + (t - lz * ny);
    lz += 1;
    return (long)lx + (long)ly * l.ext.x + (long)lz * (long)l.ext.x * l.ext.y;
  }
  // §14 H-1: on a HOST backend every reduction of the FCG skeleton (the dots, the max-norms, the
  // per-component sums of the mean projections) is computed by ONE lane, in ascending inner index
  // (= ascending ext index) over the same cell set, and broadcast to the team: its bits are then
  // independent of the team size and of OMP_NUM_THREADS (about 8 x 1536 FMA per FCG iteration).
  // A device backend keeps the team reductions.
  static constexpr bool kHostLane = std::is_same_v<CCMem, Kokkos::HostSpace>;
  // sum over q in [0, n) of f(q, acc) (f adds its term to acc)
  template <class F>
  KOKKOS_INLINE_FUNCTION static double teamSum(const Member& t, int n, const F& f) {
    double s = 0.0;
    if constexpr (kHostLane) {
      Kokkos::single(
          Kokkos::PerTeam(t),
          [&](double& v) {
            double acc = 0.0;
            for (int q = 0; q < n; ++q)
              f(q, acc);
            v = acc;
          },
          s);
    } else {
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, n), f, s);
    }
    return s;
  }
  // max over q in [0, n) of f(q, acc) (f raises acc to its term), from 0
  template <class F>
  KOKKOS_INLINE_FUNCTION static double teamMax(const Member& t, int n, const F& f) {
    double s = 0.0;
    if constexpr (kHostLane) {
      Kokkos::single(
          Kokkos::PerTeam(t),
          [&](double& v) {
            double acc = 0.0;
            for (int q = 0; q < n; ++q)
              f(q, acc);
            v = acc;
          },
          s);
    } else {
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, n), f, Kokkos::Max<double>(s));
    }
    return s;
  }
  KOKKOS_INLINE_FUNCTION double dot(const Member& t, const BottomLevel& l, const CCField& a,
                                    const CCField& c) const {
    return teamSum(t, nInner(l), [&](int q, double& acc) {
      int lx, ly, lz;
      const long i = cell(l, q, lx, ly, lz);
      if (l.AC(i) > 1e-30f)
        acc += a(i) * c(i);
    });
  }
  // The bottom's mean removal, per fluid component (§5.14): one team reduction per label, in label
  // order, then one subtraction pass.
  KOKKOS_INLINE_FUNCTION void removeMeanBottom(const Member& t, const CCField& f) const {
    const BottomLevel& l = lv;
    for (int c = 0; c < nc; ++c) {
      const double s = teamSum(t, nInner(l), [&](int q, double& acc) {
        int lx, ly, lz;
        const long i = cell(l, q, lx, ly, lz);
        if (comp(i) == c)
          acc += f(i);
      });
      Kokkos::single(Kokkos::PerTeam(t), [&]() { cmean(c) = s / (double)ccnt(c); });
    }
    t.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nInner(l)), [&](int q) {
      int lx, ly, lz;
      const long i = cell(l, q, lx, ly, lz);
      const int c = comp(i);
      if (c >= 0)
        f(i) -= cmean(c);
    });
    t.team_barrier();
  }
  // z = M r (§13.4.4): the direct factor's M, then the component means' removal (step 5). With
  // `keep`, z's previous values go there first.
  KOKKOS_INLINE_FUNCTION void applyM(const Member& t, const CCField& R, const CCField& Z,
                                     const CCField* keep) const {
    dir.apply(t, R, Z, comp, keep);
    if (!noMeanForTest)
      removeMeanBottom(t, Z);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& t) const {
    const BottomLevel& B0 = lv;
    if (precondOnly) {
      applyM(t, b, z, nullptr);
      return;
    }
    const long sy = B0.ext.x, sz = (long)B0.ext.x * B0.ext.y;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nExt(B0)), [&](int i) { x(i) = 0.0; });
    // r = b on the fluid cells and 0 on the solid ones (GraphAMG's identity rows with a zero rhs)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nInner(B0)), [&](int q) {
      int lx, ly, lz;
      const long i = cell(B0, q, lx, ly, lz);
      r(i) = (comp(i) >= 0) ? b(i) : 0.0;
    });
    t.team_barrier();
    removeMeanBottom(t, r);  // b -= mean_fluid(b), per component
    // max|r| over the fluid cells
    auto rmax = [&](int q, double& acc) {
      int lx, ly, lz;
      const long i = cell(B0, q, lx, ly, lz);
      if (B0.AC(i) > 1e-30f) {
        const double v = Kokkos::fabs(r(i));
        if (v > acc)
          acc = v;
      }
    };
    const double r0 = teamMax(t, nInner(B0), rmax);
    int it = 0;
    // a direct factor that failed even with the largest shift (non-finite input): the failure path
    bool fail = dir.stat(0) == 0;
    if (r0 != 0.0 && !fail) {
      applyM(t, r, z, nullptr);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nExt(B0)), [&](int i) { p(i) = z(i); });
      t.team_barrier();
      double rz = dot(t, B0, r, z);
      while (it < kBottomCap) {
        ++it;
        // Ap = A_b p: the band apply reading p's periodic neighbours through the wrap
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nInner(B0)), [&](int q) {
          int lx, ly, lz;
          const long i = cell(B0, q, lx, ly, lz);
          const CcNbrs w = ccWrapNbrs(lx, ly, lz, B0.inner, 1, i, sy, sz);
          ap(i) = cutcellApplyFaceCell(p, B0.AC, B0.AFX, B0.AFY, B0.AFZ, i, 1L, sy, sz, w.xp, w.xm,
                                       w.yp, w.ym, w.zp, w.zm);
        });
        t.team_barrier();
        const double pAp = dot(t, B0, p, ap);
        if (!(Kokkos::isfinite(pAp) && pAp > 1e-300)) {
          fail = !Kokkos::isfinite(pAp);  // a tiny pAp is a converged direction, not a failure
          break;
        }
        const double a = rz / pAp;
        // x += a p; r -= a Ap on the fluid cells (a cell with AC <= 1e-30 keeps x = 0), then
        // r -= its component mean, then max|r|
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nInner(B0)), [&](int q) {
          int lx, ly, lz;
          const long i = cell(B0, q, lx, ly, lz);
          if (comp(i) >= 0) {
            x(i) += a * p(i);
            r(i) -= a * ap(i);
          }
        });
        t.team_barrier();
        removeMeanBottom(t, r);
        const double rn = teamMax(t, nInner(B0), rmax);
        if (rn <= kBottomTau * r0)
          break;
        applyM(t, r, z, &zp);  // zp = z (the previous one), z = M r
        // <r, z - zp>: Polak-Ribiere (robust to M's rounding)
        const double rzc = teamSum(t, nInner(B0), [&](int q, double& acc) {
          int lx, ly, lz;
          const long i = cell(B0, q, lx, ly, lz);
          if (B0.AC(i) > 1e-30f)
            acc += r(i) * (z(i) - zp(i));
        });
        const double beta = rzc / rz;
        rz = dot(t, B0, r, z);
        if (!(Kokkos::isfinite(rz) && Kokkos::isfinite(beta))) {
          fail = true;
          break;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nInner(B0)), [&](int q) {
          int lx, ly, lz;
          const long i = cell(B0, q, lx, ly, lz);
          p(i) = z(i) + beta * p(i);
        });
        t.team_barrier();
      }
    }
    if (fail) {  // x = 0 on the bottom and the device flag; the host turns it into solveFailed_
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nExt(B0)), [&](int i) { x(i) = 0.0; });
      Kokkos::single(Kokkos::PerTeam(t), [&]() { ks(flagSlot) = 1.0; });
      t.team_barrier();
    } else {
      removeMeanBottom(t, x);  // x -= mean_fluid(x), per component
    }
    Kokkos::single(Kokkos::PerTeam(t), [&]() { info(0) = it; });
  }
};

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

// Communication-avoiding smoothing (`set_comm_avoiding`): exchange a 2-deep ghost layer once per
// red-black PAIR instead of 1-deep before every colour, redundantly re-smoothing the 1-deep ghost
// ring of the first colour so the second colour reads exactly the values a per-colour exchange
// would have delivered — bit-identical, at half the halo events. Consumed by CutcellMG's coarse
// levels and by the momentum RB-GS in flow_ibm.hpp. The two subsystems are switched independently
// so a measured regression can be ATTRIBUTED to one of them without a rebuild; both are on by
// default.
enum : int { kCaMomentum = 1, kCaMg = 2, kCaBoth = kCaMomentum | kCaMg };

class CutcellMG {
 public:
#ifdef PECLET_FLOW_MPI
  struct Telescope;  // defined after Level
#endif
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
    // A2 face form (mac_pressure.hpp): the diagonal + one coefficient per face, full extent.
    FPV AC, AFX, AFY, AFZ;
    // H-1 (doc/vof_projection_cost_design.md §8, host single rank only): the count of inner FLUID
    // cells (AC > 1e-30f, the reductions' predicate) and whether that is every inner cell, set by
    // setOpenness at each operator build (noteFluidCells). nFluid < 0: not known -> today's path.
    long nFluid = -1;
    bool allFluid = false;
    // D (doc/vof_projection_cost_design.md §4.4.2; single rank, non-bottom levels, built only while
    // the V-cycle precision is not 'fp64'): the FP32 face weights w_f = fl32(t_f) > 0 -- no
    // diagonal in any precision, D_i is summed from the six weights -- and the FP32 iterates.
    // wOk: this level's weights are built from the current openness and passed the face-range
    // check (§4.4.1 (5)).
    Kokkos::View<VReal*, CCMem> WX, WY, WZ, xf, rhsf, resf;
    bool wOk = false;
#ifdef PECLET_FLOW_MPI
    std::shared_ptr<GridHaloTopology<3>> halo;  // per-level topology (decomposed)
    std::shared_ptr<GridHalo<double>> dev;      // per-level ghost exchange
    // This level's communicator: comm_ above the first telescope point, the roots-only
    // sub-communicator below it. Every collective a LEVEL performs (removeMean's Allreduce, the
    // bottom's gather) must use this, not comm_ — see Telescope.
    MPI_Comm comm = MPI_COMM_NULL;
    // Set on level L when the transition L -> L+1 telescopes: L+1 lives on FEWER ranks (the ORB
    // tree truncated one or more levels, BlockDecomposer::agglomerated), and the V-cycle moves
    // this level's residual down / correction up through a gather / scatter within disjoint rank
    // groups. Null on an ordinary in-place transition.
    std::shared_ptr<Telescope> tele;
#endif
  };
#ifdef PECLET_FLOW_MPI
  // Coarse-level telescoping (docs/MG_TELESCOPING_PLAN.md). When a per-rank block turns odd the
  // in-place coarsening (coarsened(): coarse-local i <-> fine-local 2i on the SAME rank) is
  // blocked, and without this the hierarchy simply stops -- at 384^3 on 1536 ranks it dies at a
  // block of 3x6x4 with the coarsest global extent still 48, and the pressure iteration count
  // grows 16.6 -> 38.7 across the ladder. Merging ORB siblings restores parity (the merged block's
  // origin is its parent's split value), so: gather level L at ITS OWN resolution onto the group
  // roots, and let the roots continue the geometric hierarchy on a sub-communicator. Ranks that
  // are not roots idle below L (they join the gather and the scatter, on the parent comm, and skip
  // the recursion). restrictAvg/prolongAdd/coarsenOpenAvg are untouched: they read fine INNER
  // cells only / write fine INNER cells only, so the stage needs no halo of its own.
  //
  // The target, the communicators and the movement are core's (suite decision "Coarse-level
  // redistribution lives in core; the hierarchies stay in the methods",
  // amr/docs/amr_mg_core_boundary.md §7): the depth search is `chooseStageTarget` /
  // `shallowestLiftableMerge` with flow's per-axis lift rule as the predicate, the group / roots
  // communicators are `makeStageComm`'s, and the gather / scatter of a level field is a
  // `RedistributeTopology` built once here and run every V-cycle. What stays in flow: this struct
  // (flow's per-level record of a stage), the stage buffers, the scatter's ADD (core's backward
  // overwrites) and the WO-R2 outflow ghost-plane gather (teleGatherPlane), which moves a plane
  // beyond the inner block that the inner-cell topology does not describe.
  struct Telescope {
    // group = this rank's group (the ranks merged into one block, owner first), sub = the roots
    // only (MPI_COMM_NULL on members). Owned and freed by the StageComm.
    peclet::core::decomp::StageComm comm;
    bool root() const { return comm.active; }  // this rank owns the merged block
    bool repartition() const { return comm.kind == peclet::core::decomp::StageKind::Repartition; }
    int nMembers = 1;
    int g = 1;                   // ghost width of level L (the stage buffers use it)
    C3 mInner{}, mOg{}, mExt{};  // merged block (root): inner dims, global origin, extent
    // per-member fine block geometry in group-rank order (root first), root only
    std::vector<C3> memO, memS;
    // the movement of a level-L field: this rank's inner cells <-> the merged block's (root)
    peclet::core::decomp::RedistributeTopology<3, double> move;
    // Repartition only: the level's decomposition and the target's (replicated), which the
    // outflow ghost-plane movement (teleGatherPlaneRepartition) intersects.
    peclet::core::decomp::BlockDecomposer<3> srcDec, dstDec;
    std::vector<double> back;    // teleScatterAdd's landing buffer, level L's padded layout
    CCField res, x, ox, oy, oz;  // stage buffers at resolution L on the merged block (root)
  };
#endif
  static constexpr int G = 1;  // level-0 / single-rank ghost width (the flow_ibm g=1 bridge)
  // The red-black parity origin for a level's smoother: the parity convention is the single-rank
  // g=1 one (parity of og+local index INCLUDING a 1-cell ghost offset), so a level with g=2 must
  // shift its origin by g-1 per axis or its colours come out swapped against the g=1 reference
  // (3 axes -> parity flips). og itself stays the true global inner origin (buildAmg needs it).
  static C3 parityOg(const Level& lv) {
    return C3{lv.og.x - lv.g + 1, lv.og.y - lv.g + 1, lv.og.z - lv.g + 1};
  }

  // ---- the aspect-ratio coarsening rule (doc/anisotropic_metric.md §5) -------------------------
  // Shared by BOTH geometric hierarchies (VelocityMG calls it too): given this level's spacings
  // H[a] = hp[a] * cfac[a] and the per-axis coarsenability canA[a] that today's tests produce
  // (can(dim) single-rank, plus evenOn(dec, a) under MPI), return the per-axis coarsening ratio.
  //
  //   aniso == false : ratio_a = 2 iff canA[a]                     -- TODAY'S DECISION, VERBATIM
  //   aniso == true  : ratio_a = 2 iff canA[a] and H[a] < theta * min_{b : canA[b]} H[b]
  //
  // i.e. always coarsen the finest coarsenable axis and DEFER one that is already at least theta
  // times coarser (§5.2: the spread halves per level until it is below theta and then stays
  // there, so no coarse operator is ever more than theta^2 anisotropic in coefficient).  On an
  // isotropic metric every H in the candidate set is equal, 1 < theta, and the result is today's
  // table bit for bit -- which is why the isotropic path is gated on the flag and not on the
  // arithmetic (§5.4: the rule WOULD change the table after a telescoping merge).
  //
  // A DEFERRED axis is not a BLOCKED one: initMpi's telescope trigger keeps reading canA, so
  // deferral alone never merges ranks (trap 6).
  static C3 mgChooseRatio(const double H[3], const bool canA[3], bool aniso, double theta) {
    C3 r{1, 1, 1};
    if (!aniso) {
      if (canA[0])
        r.x = 2;
      if (canA[1])
        r.y = 2;
      if (canA[2])
        r.z = 2;
      return r;
    }
    double hmin = 0.0;
    bool any = false;
    for (int a = 0; a < 3; ++a)
      if (canA[a] && (!any || H[a] < hmin)) {
        hmin = H[a];
        any = true;
      }
    if (!any)
      return r;
    const double lim = theta * hmin;
    if (canA[0] && H[0] < lim)
      r.x = 2;
    if (canA[1] && H[1] < lim)
      r.y = 2;
    if (canA[2] && H[2] < lim)
      r.z = 2;
    return r;
  }

  /// Per-axis cell spacings h_a' = h_a / hRef of the physical domain (doc/anisotropic_metric.md
  /// §1.1).  Call BEFORE init/initMpi (trap 5): the level table is built there, long before
  /// setOpenness, and the aspect rule of §5 needs the metric at that point.  The default
  /// (1, 1, 1) is the isotropic / cell-unit path, on which the rule is inert BY CONSTRUCTION
  /// (`aniso_` is false, so mgChooseRatio returns today's decision verbatim).
  void setMetric(const double hp[3]) {
    for (int a = 0; a < 3; ++a)
      hp_[a] = hp[a];
    aniso_ = !(hp_[0] == 1.0 && hp_[1] == 1.0 && hp_[2] == 1.0);
  }
  /// The per-level coarsening ratio actually chosen (the level table of §5, gate §8.5).
  std::vector<C3> levelRatios() const {
    std::vector<C3> r;
    r.reserve(lv_.size());
    for (const auto& v : lv_)
      r.push_back(v.ratio);
    return r;
  }

  // build the periodic level hierarchy: per axis, halve inner while even and >=2 (uniform when
  // cubic), capped at nLevels (mirrors DistributedPoissonMG::init uniform path).
  void init(int nx, int ny, int nz, int nLevels) {
    lv_.clear();
    opBuilt_ = false;  // D: no operator (and no FP32 weights) on the new hierarchy yet
    amg_.reset();
    eigWarm_ = false;  // D3: the kept power-iteration vectors belong to the old hierarchy
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
        // §5.1: the candidate set is today's `can()` per axis; the aspect rule then defers an axis
        // that is already >= theta times coarser than the finest candidate.  Inert when !aniso_.
        const bool canA[3] = {can(inner.x), can(inner.y), can(inner.z)};
        const double H[3] = {hp_[0] * (double)cf.x, hp_[1] * (double)cf.y, hp_[2] * (double)cf.z};
        ratio = mgChooseRatio(H, canA, aniso_, aspectTheta_);
        if (ratio.x == 2)
          next.x = inner.x / 2;
        if (ratio.y == 2)
          next.y = inner.y / 2;
        if (ratio.z == 2)
          next.z = inner.z / 2;
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n);
      v.rhs = CCField("mg_rhs", v.n);
      v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n);
      v.oy = CCField("mg_oy", v.n);
      v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AFX, &v.AFY, &v.AFZ})
        *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (next.x == inner.x && next.y == inner.y && next.z == inner.z)
        break;  // nothing coarsens
      inner = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
    }
    buildBottomStore();  // the direct bottom's storage (single rank, every backend)
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
  // block, and each coarse level is the previous one coarsened in place while every block stays
  // even. nullptr => the solver's own aligned ORB (`decomposition()`), byte-identical to before.
  // A WEIGHTED dec0 (rebalanceByWeights) generally has ODD splits: in-place coarsening blocks at
  // the shallowest odd split and a telescope stage fires there — at level 0 when the ORB root
  // split is odd. The sibling-merge search alone then lands at the shallowest even ORB depth,
  // which is ONE RANK (d = 0) whenever the root split is odd: the whole level gathered to one rank
  // every V-cycle. Measured (SCALING_ISSUES #2, 96^3 heap, np = 8): projection 2.25-2.6x slower,
  // iterations unchanged. The two escapes this comment used to recommend do NOT escape: the
  // GraphAMG bottom leaves that level-0 telescope untouched (0.085 -> 0.218 s), and nLevels == 1
  // either runs the redundant GraphAMG on the whole grid (the `auto` bottom, ~55x slower) or pure
  // RB-GS (the smoother bottom, slower than the collapse it replaces). What bounds it is
  // setRepartition(true), which the Solver sets for a weighted dec0: a stage never hands a rank
  // more cells than the largest level-0 block, and a merge that would is replaced by a Repartition
  // onto a proportional ORB of the level's grid on np_L ranks (core's chooseStageTarget with
  // maxBlockCells > 0, amr/docs/amr_mg_core_boundary.md §11.1-§11.3).

  // flow's LIFT rule — "every block can coarsen in place one more time": every block is even in
  // origin and size on every axis whose GLOBAL extent `gs` can still halve (can()). An axis that
  // cannot halve is not a constraint. This is the predicate core's stage policy
  // (peclet::core::decomp::chooseStageTarget) is parameterised on, shared by initMpi and predict.
  static bool teleLiftable(const peclet::core::decomp::BlockDecomposer<3>& d, C3 gs) {
    auto can = [](int e) { return (e % 2 == 0) && (e / 2 >= 2); };
    const int ga[3] = {gs.x, gs.y, gs.z};
    for (int ax = 0; ax < 3; ++ax) {
      if (!can(ga[ax]))
        continue;
      for (std::size_t b = 0; b < d.sizes().size(); ++b)
        if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
          return false;
    }
    return true;
  }
  // Per-axis split alignment that makes an ORB safely coarsenable by this MG: align[k] =
  // 2^(number of times axis k can coarsen, until it turns odd) — the NATURAL MAXIMUM, independent
  // of the actual nLevels (over-aligning is harmless: coarsened() still divides cleanly at every
  // real level). Depends only on the global grid, so the solver's dec_, the mpi_block() sizing, and
  // this MG all compute the SAME value without threading nLevels. The solver builds its shared
  // decomposition with this alignment so initMpi derives nested coarse levels via coarsened().
  static peclet::core::IVec<3> coarsenAlignment(int gnx, int gny, int gnz) {
    auto can = [](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    C3 gs{gnx, gny, gnz};
    peclet::core::IVec<3> a{1, 1, 1};
    for (bool any = true; any;) {
      any = false;
      if (can(gs.x)) {
        a[0] *= 2;
        gs.x /= 2;
        any = true;
      }
      if (can(gs.y)) {
        a[1] *= 2;
        gs.y /= 2;
        any = true;
      }
      if (can(gs.z)) {
        a[2] *= 2;
        gs.z /= 2;
        any = true;
      }
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
  // `levels` selects how the LEVEL-0 DECOMPOSITION is built: 0 (the default) = the legacy
  // aligned-ORB route above; L >= 2 = build the ORB on the grid coarsened L-1 times and refine the
  // partition upward, which guarantees L nested levels and balances on the coarse grid instead of
  // snapping fine splits afterwards. It is a PARAMETER, not process state: `decomposition()` is a
  // pure function of (numBlocks, grid, levels, maxImbalance), so every rank — and every call site
  // that must agree on the partition (`flow.mpi_block`, `IbmSolver::initMpi`) — computes the same
  // answer without communicating, and two solvers in one process can differ.
  // Per-axis coarsening factor a depth-`levels` hierarchy will actually apply: 2^(levels-1),
  // bounded by that axis's factors of two (an odd axis never coarsens, so its factor stays 1).
  static peclet::core::IVec<3> refineFactor(int gnx, int gny, int gnz, int levels) {
    auto can = [](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    C3 gs{gnx, gny, gnz};
    peclet::core::IVec<3> r{1, 1, 1};
    for (int L = 1; L < levels; ++L) {
      if (can(gs.x)) {
        r[0] *= 2;
        gs.x /= 2;
      }
      if (can(gs.y)) {
        r[1] *= 2;
        gs.y /= 2;
      }
      if (can(gs.z)) {
        r[2] *= 2;
        gs.z /= 2;
      }
    }
    return r;
  }

  // THE shared level-0 decomposition. Every call site must go through this so the solver's block,
  // mpi_block()'s sizing and the MG's level 0 cannot drift apart.
  static peclet::core::decomp::BlockDecomposer<3> decomposition(std::size_t numBlocks, int gnx,
                                                                int gny, int gnz, int levels = 0,
                                                                double maxImbalance = 1.05) {
    if (levels < 2)
      return peclet::core::decomp::BlockDecomposer<3>(
          numBlocks, peclet::core::IVec<3>{gnx, gny, gnz}, coarsenAlignment(gnx, gny, gnz));
    // Depth and load balance pull against each other: each extra level doubles the quantum on every
    // axis that still coarsens, and a partition built on the coarse grid can only place a split on
    // a coarse-cell boundary. So rather than guess a granularity, BUILD each candidate and measure
    // its imbalance: take the deepest one that stays within budget, else keep the legacy aligned
    // ORB. The whole search is a pure function of (numBlocks, grid, levels) — every rank computes
    // the same answer without communicating.
    if (!(maxImbalance > 1.0))
      maxImbalance = 1.05;
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
    opBuilt_ = false;  // D: no operator (and no FP32 weights) on the new hierarchy yet
    amg_.reset();
    eigWarm_ = false;  // D3: the kept power-iteration vectors belong to the old hierarchy
    distributed_ = true;
    buildBottomStore();  // single-rank only: none (the distributed path keeps GraphAMG)
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
    // positions halved). This keeps restrict/prolong's coarse-local i <-> fine-local ratio*i
    // mapping valid on every rank. (An independent ORB per level does NOT nest — coarse blocks can
    // split a different axis than the fine level, sending restrict/prolong out of bounds.)
    peclet::core::decomp::BlockDecomposer<3> curDec;
    if (dec0) {
      curDec = *dec0;  // solver's shared decomposition (built aligned; see flow_ibm initMpi)
    } else {
      curDec = decomposition(static_cast<std::size_t>(size), gs.x, gs.y, gs.z);
    }
    // Below a telescope point the hierarchy continues on a sub-communicator of the group roots:
    // these track the communicator/rank a level is built on (rank == block index at every level,
    // by construction of the sub-communicator's keys).
    MPI_Comm curComm = comm;
    int curRank = rank;
    teleActive_ = true;
    // setRepartition: a stage never hands a rank more cells than the largest LEVEL-0 block
    // (design §11.1 — the finest level's largest block, replicated); 0 = sibling merges only.
    const peclet::core::Index maxBlockCells =
        repartition_ ? peclet::core::decomp::largestBlockCells(curDec) : 0;
    auto evenOn = [](const peclet::core::decomp::BlockDecomposer<3>& d, int ax) {
      for (std::size_t b = 0; b < d.sizes().size(); ++b)
        if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
          return false;
      return true;
    };
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
      v.g = (L > 0 && !hasBC_ && (caMode_ & kCaMg) && minBlockExtent(dec) >= 4) ? 2 : 1;
      v.caOk = (v.g == 2);
      v.halo->buildTopology(dec, curRank, v.g, per, curComm);
      v.comm = curComm;
      v.dev = std::make_shared<GridHalo<double>>();
      v.dev->init(*v.halo);
      v.dev->setLabel("mg L" + std::to_string(L) + " g" + std::to_string(v.g) + " ranks" +
                      std::to_string(curDec.numBlocks()));
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
      bool idleBelow = false;
      if (L + 1 < nLevels) {
        // Coarsen an axis only if the GLOBAL dim can (can()) AND every rank's block is even on that
        // axis, so coarsened() nests exactly. Alignment (coarsenAlignment) makes this hold for the
        // full natural depth; on an unaligned/awkward decomposition it used to simply stop
        // coarsening that axis (fewer geometric levels) — with telescoping ON, a blocked axis
        // instead merges ORB siblings onto fewer ranks and carries on (see Telescope).
        namespace cdec = peclet::core::decomp;
        const bool canAny = can(gs.x) || can(gs.y) || can(gs.z);
        // flow's lift rule, the predicate core's stage policy is parameterised on.
        auto liftable = [&](const cdec::BlockDecomposer<3>& d) { return teleLiftable(d, gs); };
        const bool blocked = !liftable(curDec);
        const bool tooSmall = teleMinExtent_ > 0 &&
                              cdec::minBlockExtent(curDec) < (peclet::core::Index)teleMinExtent_;
        const bool doTele = canAny && curDec.numBlocks() > 1 &&
                            ((telescope_ && (blocked || tooSmall)) || teleForce_ == L);
        if (doTele) {
          // Fewest merges (largest tree depth) at which EVERY still-coarsenable axis has even
          // blocks — "any axis" would be wrong: it can unblock z and leave x frozen at the extent
          // that matters. depth 0 (one block: origin 0, size gs, even by can()) always qualifies;
          // with the economic trigger a merged candidate must also stay fat enough after the
          // halving that follows. That search is core's `shallowestLiftableMerge`, and behind
          // flow's trigger it is `chooseStageTarget` with maxBlockCells = 0 (byte-identical to the
          // search flow carried inline until S4). The test-only forced telescope runs the search
          // where in-place coarsening is legal, i.e. the search on its own.
          std::optional<cdec::StageTarget<3>> t;
          if (teleForce_ == L) {
            if (auto m = cdec::shallowestLiftableMerge(curDec, liftable, teleMinExtent_)) {
              t.emplace();
              t->kind = cdec::StageKind::SiblingMerge;
              t->dec = std::move(m->dec);
              t->ownerOf = std::move(m->ownerOf);
              t->groupOf = std::move(m->groupOf);
            }
          } else {
            t = cdec::chooseStageTarget(curDec, peclet::core::IVec<3>{gs.x, gs.y, gs.z}, liftable,
                                        teleMinExtent_, maxBlockCells);
            if (t->kind != cdec::StageKind::SiblingMerge && t->kind != cdec::StageKind::Repartition)
              throw std::logic_error(
                  "CutcellMG::initMpi: the telescope search found no merge (depth 0 always lifts "
                  "under flow's rule)");
          }
          if (t) {
            auto T = std::make_shared<Telescope>();
            T->comm = cdec::makeStageComm(curComm, *t);
            T->g = v.g;
            // The target block this rank's data moves into: its merge group's (SiblingMerge — the
            // root owns it, the members only feed it), or the one it owns (Repartition; none on a
            // rank that idles below).
            const int tb = T->repartition() ? T->comm.myTargetBlock : T->comm.myGroup;
            if (tb >= 0) {
              const auto mb = t->dec.block((std::size_t)tb);
              T->mOg = C3{(int)mb.origin[0], (int)mb.origin[1], (int)mb.origin[2]};
              T->mInner = C3{(int)mb.size[0], (int)mb.size[1], (int)mb.size[2]};
              T->mExt = C3{T->mInner.x + 2 * T->g, T->mInner.y + 2 * T->g, T->mInner.z + 2 * T->g};
            }
            if (T->repartition()) {
              T->srcDec = curDec;
              T->dstDec = t->dec;
            }
            // members in group-comm rank order == ascending old block index (makeStageComm)
            for (const int b : T->comm.members) {
              const auto fb = curDec.block((std::size_t)b);
              T->memO.push_back(C3{(int)fb.origin[0], (int)fb.origin[1], (int)fb.origin[2]});
              T->memS.push_back(C3{(int)fb.size[0], (int)fb.size[1], (int)fb.size[2]});
            }
            T->nMembers = (int)T->comm.members.size();
            // The movement: a global cell of this rank's level-L block lives at its padded slot
            // (og, g, ext of the level); on the root, a cell of the merged block at the stage
            // buffer's (mOg, g, mExt). Built once; teleGather / teleScatterAdd run it.
            {
              const C3 lo = v.og, le = v.ext, mo = T->mOg, me = T->mExt;
              const int lg = v.g, mg = T->g;
              using peclet::core::Index;
              auto srcIdx = [&](const peclet::core::IVec<3>& c) {
                return (Index)(c[0] - lo.x + lg) + (Index)(c[1] - lo.y + lg) * le.x +
                       (Index)(c[2] - lo.z + lg) * (Index)le.x * le.y;
              };
              auto dstIdx = [&](const peclet::core::IVec<3>& c) {
                return (Index)(c[0] - mo.x + mg) + (Index)(c[1] - mo.y + mg) * me.x +
                       (Index)(c[2] - mo.z + mg) * (Index)me.x * me.y;
              };
              T->move.build(curDec, *t, T->comm, srcIdx, dstIdx, /*id=*/L);
              T->back.assign(v.n, 0.0);
            }
            if (T->root()) {
              const std::size_t mn = (std::size_t)T->mExt.x * T->mExt.y * T->mExt.z;
              T->res = CCField("tele_res", mn);
              T->x = CCField("tele_x", mn);
              T->ox = CCField("tele_ox", mn);
              T->oy = CCField("tele_oy", mn);
              T->oz = CCField("tele_oz", mn);
            }
            v.tele = T;
            if (T->root()) {
              curDec = t->dec;
              curComm = T->comm.sub;
              MPI_Comm_rank(curComm, &curRank);
            } else {
              idleBelow = true;  // this rank holds levels 0..L only
            }
          }
        }
        auto evenBlocks = [&](int ax) { return evenOn(curDec, ax); };
        // §5.1 again, with the MPI candidate set: an axis can coarsen when the GLOBAL dim can and
        // every rank's (possibly just-merged) block is even on it.  The aspect rule then defers
        // one that is already >= theta times coarser.  Note `blocked` above is computed from
        // `can()` and `evenOn()` ALONE (trap 6): an axis deferred here is not blocked and must not
        // trigger a telescope merge.  Every rank has the same doubles and the same decomposition,
        // so this is a pure function -- no communication.
        const bool canA[3] = {can(gs.x) && evenBlocks(0), can(gs.y) && evenBlocks(1),
                              can(gs.z) && evenBlocks(2)};
        const double H[3] = {hp_[0] * (double)cf.x, hp_[1] * (double)cf.y, hp_[2] * (double)cf.z};
        ratio = mgChooseRatio(H, canA, aniso_, aspectTheta_);
        if (ratio.x == 2)
          next.x = gs.x / 2;
        if (ratio.y == 2)
          next.y = gs.y / 2;
        if (ratio.z == 2)
          next.z = gs.z / 2;
      }
      v.ratio = ratio;
      v.x = CCField("mg_x", v.n);
      v.rhs = CCField("mg_rhs", v.n);
      v.res = CCField("mg_res", v.n);
      v.ox = CCField("mg_ox", v.n);
      v.oy = CCField("mg_oy", v.n);
      v.oz = CCField("mg_oz", v.n);
      for (FPV* p : {&v.AC, &v.AFX, &v.AFY, &v.AFZ})
        *p = FPV("mg_A", v.n);
      lv_.push_back(v);
      if (idleBelow) {
        teleActive_ = false;
        break;
      }
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z)
        break;
      gs = next;
      cf = C3{cf.x * ratio.x, cf.y * ratio.y, cf.z * ratio.z};
      // Next level's decomposition = this level's coarsened in place (nested; preserves rank
      // order).
      curDec = curDec.coarsened(peclet::core::IVec<3>{ratio.x, ratio.y, ratio.z});
    }
    if (mgDebugLevel() && rank == 0) {
      printf("[mg] initMpi %dx%dx%d np=%d -> %d levels (requested %d)%s\n", gnx, gny, gnz, size,
             (int)lv_.size(), nLevels, telescope_ ? "  [telescope ON]" : "");
      C3 g{gnx, gny, gnz};
      int ranks = size;
      for (int L = 0; L < (int)lv_.size(); ++L) {
        printf(
            "[mg]  L%d global %4dx%4dx%4d  ranks %5d  rank0 block %4dx%4dx%4d  ratio(%d,%d,%d)%s\n",
            L, g.x, g.y, g.z, ranks, lv_[L].inner.x, lv_[L].inner.y, lv_[L].inner.z, lv_[L].ratio.x,
            lv_[L].ratio.y, lv_[L].ratio.z,
            lv_[L].tele ? (lv_[L].tele->repartition() ? "  -> REPARTITION" : "  -> TELESCOPE")
                        : "");
        if (lv_[L].tele) {
          int sub = 1;
          MPI_Comm_size(lv_[L].tele->comm.sub, &sub);
          ranks = sub;
        }
        g = C3{g.x / lv_[L].ratio.x, g.y / lv_[L].ratio.y, g.z / lv_[L].ratio.z};
      }
      fflush(stdout);
    }
  }
#endif
  int nLevels() const { return (int)lv_.size(); }
  Level& level(int L) { return lv_[L]; }

  // per-face domain BC types {-x,+x,-y,+y,-z,+z}: 0=periodic, 1/2/4=Neumann (wall/inflow/
  // free-slip), 3=Dirichlet (outflow). Default all-periodic -> applyBoundaryOpenness is a no-op
  // (periodic/IBM path byte-identical).
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
  // Zero-gradient (Neumann) pressure/correction ghost on a wall/inflow face (BC type 1/2), the
  // counterpart of applyOutflowGhost's Dirichlet ghost. WHY IT EXISTS (WO-H, 2026-08-30): the
  // per-level ghost fill is PERIODIC on all three axes (fill()/GridHalo), so on a walled face a
  // level's `x` ghost carries the value from the OPPOSITE side of the domain. Every *operator*
  // consumer is immune — the wall face openness is 0, so the smoother/residual/matvec multiply that
  // ghost by AW/AE/... = 0 — but `prolongAdd` is NOT: trilinear interpolation reads the coarse
  // ghost with weight 1/4 whatever the openness, so the fine cells against a wall were receiving a
  // quarter of the coarse correction from the far wall. That teleport is a long-range coupling
  // present in P and absent from R, i.e. exactly the asymmetry that broke MG-PCG on domain-BC grids
  // (measured: dense-M skew ||M-M^T||F/||M||F 3.5-5.4 % wall-bounded vs 0.8 % periodic, and PCG
  // 200/200 vs 7). With the zero-gradient ghost the boundary fine cell simply takes the coarse
  // value (0.25*c0 + 0.75*c0 = c0), which is also what the constant-mode-preserving prolongation
  // must do. Call after every (periodic) fill of a level's solution field that a prolongation will
  // read.
  //
  // This is the pressure-side counterpart of `VelocityMG::fillProlongBcGhosts` /
  // `fillBcGhost` (mac_velocity_mg.hpp), the port of the retired CUDA `mg_fill_bc_ghost_k`: the
  // VELOCITY multigrid has always done both halves (Dirichlet 0 AND Neumann zero-gradient) before
  // its trilinear prolongation. The pressure MG only ever received the Dirichlet half
  // (applyOutflowGhost) — CLAUDE.md's "the trilinear prolongation fills the non-periodic boundary
  // ghosts (Neumann -> zero-gradient, Dirichlet -> 0)" described the intent, not the code.
  void applyNeumannGhost(const Level& lv, CCField x, int g = G) { applyNeumannGhostT(lv, x, g); }
  // D: the FP32 coarse iterate's zero-gradient ghost before the FP32 prolongation (same copies).
  void applyNeumannGhost(const Level& lv, VField x, int g = G) { applyNeumannGhostT(lv, x, g); }
  template <class V>
  void applyNeumannGhostT(const Level& lv, V x, int g) {
    if (!hasBC_ || !bcGhost_)
      return;
    B3 e{lv.ext.x, lv.ext.y, lv.ext.z};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if ((t == 1 || t == 2 || t == 4) && touchesGlobalFace(lv, 2 * a + s))
          bcNeumannGhostT(x, e, g, a, s);
      }
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
  // `fine` = the next FINER level (null on level 0). Only the outflow-coefficient path reads it.
  void applyBoundaryOpenness(Level& lv, Level* fine = nullptr) {
    if (fine) {
      CCField fo[3] = {fine->ox, fine->oy, fine->oz};
      applyBoundaryOpennessFrom(lv, fo, fine->ext, fine->g, fine->ratio);
    } else {
      applyBoundaryOpennessFrom(lv, nullptr, C3{0, 0, 0}, 0, C3{1, 1, 1});
    }
  }
  // `fo`/`fext`/`fg`/`fratio` describe the FINE openness this level coarsens its outflow
  // coefficient plane from: the finer Level in place, or -- across a telescope point -- the merged
  // stage buffers (Telescope::ox/oy/oz on T.mExt with T.g), whose high-side outflow plane
  // teleGatherPlane carried over with the inner cells.
  void applyBoundaryOpennessFrom(Level& lv, const CCField* fo, C3 fext, int fg, C3 fratio) {
    if (!hasBC_)
      return;
    B3 e{lv.ext.x, lv.ext.y, lv.ext.z};
    CCField oa[3] = {lv.ox, lv.oy, lv.oz};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (!touchesGlobalFace(lv, 2 * a + s))
          continue;  // interior rank boundary: the exchanged openness is the right value
        if (t == 1 || t == 2 || t == 4) {
          bcSetOpenness(oa[a], e, lv.g, a, s, 0.0);  // wall/inflow/free-slip Neumann -> closed
        } else if (t == 3) {
          if (!outflowCoeff_) {
            bcSetOpenness(oa[a], e, lv.g, a, s, 1.0);  // outflow -> open (periodic fill wraps)
            continue;
          }
          // WO-R2 item 1: the caller handed a COEFFICIENT field (open_f*rho0/rho_f), so the
          // literal 1.0 would overwrite it. Low side = an inner index the caller / the coarsening
          // already wrote correctly -> leave it. High side = a ghost index the fill wrapped ->
          // restore (level 0) or coarsen from the finer level's own restored plane.
          if (s == 0)
            continue;
          if (fo == nullptr) {
            mgRestoreFacePlane(oa[a], CCConst(bcPlane_[a]), lv.ext, lv.g, a, 1);
          } else {
            mgCoarsenFacePlane(oa[a], CCConst(fo[a]), lv.ext, fext, lv.g, fg, lv.inner, fratio, a,
                               1);
          }
        }
      }
  }
  // Save the HIGH-side outflow face plane of a coefficient field before the (periodic/halo) ghost
  // fill destroys it. Called on level 0 only; the coarse levels re-derive theirs by averaging.
  void saveOutflowPlanes(Level& lv) {
    CCField oa[3] = {lv.ox, lv.oy, lv.oz};
    int dims[3] = {lv.ext.x, lv.ext.y, lv.ext.z};
    for (int a = 0; a < 3; ++a) {
      if (bc_[2 * a + 1] != 3 || !touchesGlobalFace(lv, 2 * a + 1))
        continue;
      const int b = (a + 1) % 3, c = (a + 2) % 3;
      const std::size_t np = (std::size_t)dims[b] * dims[c];
      if (bcPlane_[a].extent(0) != np)
        bcPlane_[a] = CCField(
            Kokkos::view_alloc("peclet::flow::mg_bcplane", Kokkos::WithoutInitializing), np);
      mgSaveFacePlane(bcPlane_[a], CCConst(oa[a]), lv.ext, lv.g, a, 1);
    }
  }

  // H-1 (§8): count a level's inner fluid cells once per operator build, with the reductions'
  // predicate, so the host reductions can drop the AC read on an all-fluid level and removeMean
  // its count reduction. Host single rank only; the device and the distributed path keep their
  // kernels (§2 device reduction caveat; the distributed count needs its Allreduce anyway).
  void noteFluidCells(Level& lv) {
    lv.nFluid = -1;
    lv.allFluid = false;
    if (!kHostMemory || distributed_)
      return;
    C3 e = lv.ext;
    const int g = lv.g;
    FPV ac = lv.AC;
    long cnt = 0;
    ccReduce3(
        "mgfluidcount", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, long& k) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f)
            k += 1;
        },
        cnt);
    lv.nFluid = cnt;
    lv.allFluid = cnt == (long)lv.inner.x * lv.inner.y * lv.inner.z;
  }
  // H-1: the host single-rank all-fluid fast path of the reductions below (same cells, same
  // order as the masked body; the device never takes it).
  bool hostAllFluid(const Level& lv) const {
    return kHostMemory && !distributed_ && lv.nFluid >= 0 && lv.allFluid;
  }

  // H-2a (§8): the level-0 face field of axis a, as the destination of a caller's per-step
  // coefficient builder (single rank, no outflow face: setOpenness then overwrites every ghost by
  // the periodic fill and re-imposes the wall faces, so the field's previous ghost content cannot
  // leak; the outflow-coefficient plane snapshot would read it, so outflow keeps the copies).
  CCField level0Coefficient(int a) const {
    const Level& f = lv_[0];
    return a == 0 ? f.ox : (a == 1 ? f.oy : f.oz);
  }
  bool distributed() const { return distributed_; }

  // rediscretized cut-cell operator on every level from the fine face openness (idx2 = 1/dx^2
  // fine).
  void setOpenness(CCConst ox, CCConst oy, CCConst oz, double idx2, double idy2, double idz2) {
    Level& f = lv_[0];
    // Retained for the exact (matrix-free) level-0 apply, which re-derives the face coefficient
    // t_f = open_f * gf instead of reading the assembled band. All three call sites pass grid
    // units (1.0) today; store them rather than assume it.
    gfx_ = idx2;
    gfy_ = idy2;
    gfz_ = idz2;
    // H-2a (§8): a caller that built its coefficients straight into level 0 (level0Coefficient)
    // hands the same Views back -- nothing to stage.
    if (ox.data() != f.ox.data())
      Kokkos::deep_copy(f.ox, ox);
    if (oy.data() != f.oy.data())
      Kokkos::deep_copy(f.oy, oy);
    if (oz.data() != f.oz.data())
      Kokkos::deep_copy(f.oz, oz);
    if (outflowCoeff_)
      saveOutflowPlanes(f);  // WO-R2: the high-side outflow coefficient lives on a ghost index
                             // that the fill below wraps over -- snapshot it first.
    fillOpenness(
        f);  // periodic fine-level openness ghosts (the operator reads the + neighbour face);
             // idempotent when the caller already filled them, required when it passed inner-only.
    applyBoundaryOpenness(
        f);            // re-impose non-periodic wall/inflow faces the periodic fill clobbered
    if (fp32Build(0))  // D: the FP32 weights in the same kernel (§4.4.2)
      f.wOk = buildCutcellOpFaceFp32(f.AC, f.AFX, f.AFY, f.AFZ, f.WX, f.WY, f.WZ, CCConst(f.ox),
                                     CCConst(f.oy), CCConst(f.oz), f.ext, G, idx2, idy2, idz2,
                                     true) == 0;
    else
      buildCutcellOpFace(f.AC, f.AFX, f.AFY, f.AFZ, CCConst(f.ox), CCConst(f.oy), CCConst(f.oz),
                         f.ext, G, idx2, idy2, idz2);
    noteFluidCells(f);
#ifdef PECLET_FLOW_MPI
    // A telescope point gathers this level's openness onto the group roots (all group ranks take
    // part); the next level coarsens from that stage. A rank idling below holds no next level, so
    // the loop simply ends for it after the gather.
    if (lv_[0].tele) {
      teleGather(lv_[0], lv_[0].ox, lv_[0].tele->ox);
      teleGather(lv_[0], lv_[0].oy, lv_[0].tele->oy);
      teleGather(lv_[0], lv_[0].oz, lv_[0].tele->oz);
      teleGatherOutflowPlanes(lv_[0]);
    }
#endif
    for (int L = 1; L < (int)lv_.size(); ++L) {
      Level& c = lv_[L];
      Level& fin = lv_[L - 1];
#ifdef PECLET_FLOW_MPI
      if (fin.tele) {
        // Across a telescope point the fine openness is the merged STAGE (inner cells gathered by
        // teleGather; the WO-R2 high-side outflow coefficient plane, a ghost index, gathered by
        // teleGatherPlane), so the coarse boundary coefficient is coarsened from the stage exactly
        // as the in-place path coarsens it from the finer level.
        Telescope& T = *fin.tele;
        coarsenOpenAvg(c.ox, c.oy, c.oz, CCConst(T.ox), CCConst(T.oy), CCConst(T.oz), c.ext, T.mExt,
                       c.g, T.g, c.inner, fin.ratio);
        fillOpenness(c);
        CCField so[3] = {T.ox, T.oy, T.oz};
        applyBoundaryOpennessFrom(c, so, T.mExt, T.g, fin.ratio);
      } else {
#endif
        coarsenOpenAvg(c.ox, c.oy, c.oz, CCConst(fin.ox), CCConst(fin.oy), CCConst(fin.oz), c.ext,
                       fin.ext, c.g, fin.g, c.inner, fin.ratio);
        fillOpenness(c);  // periodic ghost openness (operator build reads the + neighbour face)
        applyBoundaryOpenness(c, &fin);  // re-impose non-periodic boundary faces per coarse level
#ifdef PECLET_FLOW_MPI
      }
#endif
      const double sx = 1.0 / (double)(c.cfac.x * c.cfac.x),
                   sy = 1.0 / (double)(c.cfac.y * c.cfac.y),
                   sz = 1.0 / (double)(c.cfac.z * c.cfac.z);
      // Width-2 (CA-eligible) levels also assemble the 1-deep ghost RING of the operator (build
      // box widened by 1): the ring rows are a deterministic function of the EXCHANGED openness,
      // so they come out bit-identical to the owning rank's inner rows — the redundant ring
      // re-smoothing of the CA sweep reads them. Inner rows are computed from the same operands
      // as the g-box build (identical). g=1 levels keep the inner-only build.
      if (fp32Build(L))  // D: the FP32 weights in the same kernel (§4.4.2)
        c.wOk =
            buildCutcellOpFaceFp32(c.AC, c.AFX, c.AFY, c.AFZ, c.WX, c.WY, c.WZ, CCConst(c.ox),
                                   CCConst(c.oy), CCConst(c.oz), c.ext, c.g == 2 ? c.g - 1 : c.g,
                                   idx2 * sx, idy2 * sy, idz2 * sz, true) == 0;
      else
        buildCutcellOpFace(c.AC, c.AFX, c.AFY, c.AFZ, CCConst(c.ox), CCConst(c.oy), CCConst(c.oz),
                           c.ext, c.g == 2 ? c.g - 1 : c.g, idx2 * sx, idy2 * sy, idz2 * sz);
      noteFluidCells(c);
#ifdef PECLET_FLOW_MPI
      if (c.tele) {
        teleGather(c, c.ox, c.tele->ox);
        teleGather(c, c.oy, c.tele->oy);
        teleGather(c, c.oz, c.tele->oz);
        teleGatherOutflowPlanes(c);
      }
#endif
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
    // §13: the direct factor is rebuilt at the next direct bottom solve; the components
    // (condition 6) are labelled once per hierarchy, here at its first operator build.
    facStale_ = true;
    if (!bottomConnKnown_ && bottomStore_)
      evalBottomComponents();
    opBuilt_ = true;
#ifndef NDEBUG
    checkFp32DecoupledSets();
#endif
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
    if (!distributed_)  // A6: the single-rank driver keeps its Krylov scalars on the device
      return solvePCGResident(b, x, r, p, z, Ap, maxit, rtol, pre, post, bottom, star, nStar,
                              nnStar);
    solveFailed_ = false;  // ISSUES sweep item 6: per-solve breakdown flag
    OverlayScope overlay(overlaySolve_, star != nullptr);
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(CCExec(), l0.x, x);
    auto matvec = [&](CCField y, CCField v) {
      matvecOverlap(l0, y, v);
      if (star)
        starApplyDelta(y, CCConst(v), *star, nStar, nnStar, l0.ext, G, l0.ext, G, exactResidual_);
    };
    auto precond = [&](CCField zz, CCField rr) { precondVcycle(zz, rr); };  // A5
    matvec(Ap, x);                                                          // r = b - A x
    Kokkos::deep_copy(CCExec(), r, b);
    axpy(r, -1.0, Ap);
    removeMean(l0, r);  // compatibility: project rhs/residual onto the range
    const double r0 = maxabs(l0, r);
    // The stop reference: r0 (the default), or the caller's FULL right-hand-side norm when a
    // stop reference is set (setStopReference; only the balanced-force projection sets it).
    // With it unset rref IS r0, so `rtol * rref` is bit-for-bit the old test.
    const double rref = stopRef_ > 0.0 ? stopRef_ : r0;
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
      Kokkos::deep_copy(CCExec(), p, z);
      double rz = dot(l0, r, z);
      if (!std::isfinite(rz)) {
        // ISSUES sweep item 6: this is a FAILED solve, not a converged one. It used to print to
        // stdout, zero the correction and return 0 iterations, so a caller's rule-3b "no capped
        // pressure solve" check passed while the projection had been handed nothing. Report the
        // cap and raise the flag; set_pressure_strict(True) turns it into a throw.
        solveFailed_ = true;
        printf(
            "peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z; "
            "returning zero correction (reported as %d/%d iterations, i.e. a CAPPED solve)\n",
            maxit, maxit);
        Kokkos::deep_copy(CCExec(), x, 0.0);
        Kokkos::deep_copy(CCExec(), l0.x, x);
        if (strictPressure_)
          throw std::runtime_error(
              "peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z "
              "(set_pressure_strict)");
        return maxit;
      }
      for (; it < maxit; ++it) {
        matvec(Ap, p);
        if (meanRemovalAll_)
          removeMean(l0, Ap);  // A preserves mean-freeness; "fine" scope trusts that
        const double pAp = dot(l0, p, Ap);
        if (!std::isfinite(pAp) || pAp <= 1e-300) {
          if (!std::isfinite(pAp))
            solveFailed_ = true;  // ISSUES sweep item 6 (pAp <= 1e-300 is a CONVERGED direction)
          break;                  // keep the last finite iterate
        }
        const double alpha = rz / pAp;
        axpy(x, alpha, p);
        axpy(r, -alpha, Ap);
        removeMean(l0, r);
        const double rn = maxabs(l0, r);
        if (trace)
          printf("[mg]   it %3d  |r|inf=%.6e  r/r0=%.4e\n", it + 1, rn, rn / r0);
        if (rn < rtol * rref) {
          ++it;
          break;
        }
        precond(z, r);
        const double rznew = dot(l0, r, z), beta = rznew / rz;
        if (!std::isfinite(rznew)) {
          solveFailed_ = true;  // ISSUES sweep item 6: a breakdown, not a convergence
          break;                // preconditioner breakdown: keep the last finite iterate
        }
        aypx(p, beta, z);
        rz = rznew;
      }
    }
    Kokkos::deep_copy(CCExec(), l0.x, x);
    removeMean(l0, l0.x);
    Kokkos::deep_copy(CCExec(), x, l0.x);
    return it;
  }

  // --- A6 (doc/vof_step_performance_design.md §4.4, §5.6): device-resident Krylov scalars -------
  // The single-rank MG-PCG driver. The same recurrence as solvePCG's loop, statement for statement,
  // with its scalars kept on the device: the reductions land in slots of ks_ (0-d Views over one
  // small device array) with the same policy and functor as dot / maxabs / removeMean, alpha and
  // beta are formed by one-thread kernels with the same IEEE divisions, and the host reads ONE
  // packet {pAp, |r|inf, stop} per iteration -- what the stop test needs -- instead of five scalars
  // (two dots, the max, two mean-removal sums). Bit-identical to the host-scalar loop; every exit
  // returns the same iteration count and sets the same solveFailed_:
  //   * pAp non-finite or <= 1e-300: x and r untouched, break without ++it (failed iff non-finite);
  //   * r^T z non-finite: flagged on the device; the NEXT iteration's packet reports it (its update
  //     is skipped, x and r untouched) and the index of the iteration that computed it is returned
  //     -- or, if that was the last one, the read after the loop does;
  //   * |r|inf < rtol r0: ++it, break.
  // The distributed path keeps solvePCG's host-scalar loop unchanged (a device-resident Allreduce
  // is not in this design).
  enum : int {
    kPAp = 0,
    kRn = 1,
    kStop = 2,
    kBottomFlag = 3,  // the device bottom met a non-finite scalar (read with the packet)
    kRz = 4,
    kRzNew = 5,
    kBeta = 6,
    kAlpha = 7,
    kMsum = 8
  };
  static constexpr int kNSlots = 9;
  static constexpr int kPacket = 4;  // the per-iteration host packet {pAp, rn, stop, bottomFlag}
  static constexpr double kBrkPAp = 1.0, kBrkRz = 2.0;  // the stop codes (slot kStop)
  Kokkos::View<double*, CCMem> ks_;                     // the scalar slots
  Kokkos::View<long, CCMem> kcnt_;                      // removeMean's fluid-cell count
  Kokkos::View<double*, Kokkos::HostSpace> kpk_;  // the host packet {pAp, rn, stop, bottomFlag}
  void ensureScalars() {
    if (ks_.extent(0) == (std::size_t)kNSlots)
      return;
    ks_ = Kokkos::View<double*, CCMem>("peclet::flow::mg_krylov_scalars", kNSlots);
    kcnt_ = Kokkos::View<long, CCMem>("peclet::flow::mg_mean_count");
    kpk_ = Kokkos::View<double*, Kokkos::HostSpace>("peclet::flow::mg_krylov_packet", kPacket);
  }
  Kokkos::View<double, CCMem> slot(int k) const {
    return Kokkos::View<double, CCMem>(ks_.data() + k);
  }
  double readSlot(int k) const {
    double v = 0.0;
    Kokkos::deep_copy(v, slot(k));
    return v;
  }
  void setSlot(int k, double v) { Kokkos::deep_copy(CCExec(), slot(k), v); }
  // dot / maxabs into a slot: the SAME policy and functor as dot() / maxabs(), result on the
  // device.
  void dotTo(Level& lv, CCField a, CCField b, int k) {
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a, bb = b;
    FPV ac = lv.AC;
    if (hostAllFluid(lv)) {  // H-1: no AC read
      ccReduce3Lanes(
          "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            acc += aa(i) * bb(i);
          },
          slot(k));
      return;
    }
    ccReduce3Lanes(
        "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (ac(i) > 1e-30f)
            acc += aa(i) * bb(i);
        },
        slot(k));
  }
  // H-2b (§8): p = z and r^T z into slot k. On the host single-rank fused-wrap V-cycle (even
  // level 0 with a coarser level) in ONE pass over the inner cells, with dotTo's body and order:
  // there no pass writes a ghost of z (precondVcycle zeroes it, the wrap smoother and residual
  // fill nothing, prolongAdd writes inner cells), so z's ghosts are +0 and p's, never copied, stay
  // the +0 they always hold. Elsewhere the copy and dotTo (§2: a device sum-fusion can change the
  // reduction tree).
  void copyDotTo(Level& lv, CCField p, CCField r, CCField z, int k) {
    if (!(kHostMemory && fusedWrapSmooth(lv) && lv_.size() > 1)) {
      Kokkos::deep_copy(CCExec(), p, z);
      dotTo(lv, r, z, k);
      return;
    }
    C3 e = lv.ext;
    const int g = lv.g;
    CCField pp = p, rr = r, zz = z;
    FPV ac = lv.AC;
    const bool all = hostAllFluid(lv);
    ccReduce3Lanes(
        "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int zc, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)zc * (long)e.x * e.y;
          pp(i) = zz(i);
          if (all || ac(i) > 1e-30f)
            acc += rr(i) * zz(i);
        },
        slot(k));
  }
  // C (doc/vof_projection_cost_design.md §6): eligibility of the fused host Krylov kernels of
  // solvePCGResident -- host memory, single rank with the precomputed fluid count (H-1), the A3
  // wrap reads (no outflow face, no overlay), a singular operator (the mean projection runs), and
  // level 0 at the matvec's ghost width. Elsewhere the separate kernels (the device keeps them:
  // §2, a device sum-fusion can change the reduction tree).
  // C4: a pending request of solvePCGResident to form r^T z (slot k; and p = z when p is
  // allocated) in the level-0 exit pass of the next V-cycle; k < 0 = none (every other driver).
  struct ExitDot {
    int k = -1;
    CCField p;
    bool done = false;
  };
  ExitDot exitDot_;
  bool hostKrylovFusion() const {
    const Level& l0 = lv_[0];
    return kHostMemory && fusedWrapReads() && removeMean_ && l0.nFluid >= 0 && l0.g == G;
  }
  // C1 (§6 (i)): y = A v (matvecOverlap's A3 wrap cell body, exact flux or band form) and
  // v^T y into slot k in ONE lane pass over the inner cells, with dotTo's mask, operand order and
  // lane order: bitwise to matvecOverlap(l0, y, v) + dotTo(l0, v, y, k). hostKrylovFusion() only.
  void matvecDotTo(Level& l0, CCField y, CCField v, int k) {
    const C3 e = l0.ext, n = l0.inner;
    const int g = G;
    CCField yy = y;
    CCConst vv = v;
    FPV ac = l0.AC;
    const bool all = hostAllFluid(l0);
    if (exactResidual_) {
      CCConst ox = l0.ox, oy = l0.oy, oz = l0.oz;
      const double gfx = gfx_, gfy = gfy_, gfz = gfz_;
      ccReduce3Lanes(
          "peclet::flow::cc_apply_exact_dot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int lx, int ly, int lz, double& acc) {
            const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
            const long i = (long)lx + (long)ly * sy + (long)lz * sz;
            const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
            const double a = cutcellApplyExactCell(vv, ox, oy, oz, i, sx, sy, sz, w.xp, w.xm, w.yp,
                                                   w.ym, w.zp, w.zm, gfx, gfy, gfz);
            yy(i) = a;
            if (all || ac(i) > 1e-30f)
              acc += vv(i) * a;
          },
          slot(k));
      return;
    }
    FPC AC = l0.AC, AFX = l0.AFX, AFY = l0.AFY, AFZ = l0.AFZ;
    ccReduce3Lanes(
        "peclet::flow::cc_apply_dot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz, double& acc) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)lx + (long)ly * sy + (long)lz * sz;
          const CcNbrs w = ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
          const double a = cutcellApplyFaceCell(vv, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm,
                                                w.yp, w.ym, w.zp, w.zm);
          yy(i) = a;
          if (all || ac(i) > 1e-30f)
            acc += vv(i) * a;
        },
        slot(k));
  }
  // C2 (§6 (ii), amendment 4): x += alpha p, r -= alpha Ap (mgpcg_update's expressions, skipped
  // once the stop flag is set) over the INNER cells, and the fluid sum of the new r into kMsum
  // with removeMeanHostCounted's body and lane order -- bitwise to mgpcg_update + that sum on
  // every inner cell. The ghost entries of x and r are no longer updated: no reader consumes
  // them (the A3 wrap reads wrap inner cells, the V-cycle reads r's inner cells, the reductions
  // and the final removeMean run over inner cells). A stopped iteration adds nothing; its sum is
  // never used (the stop-guarded subtract skips). hostKrylovFusion() only.
  void pcgUpdateSum(Level& l0, CCField x, CCField r, CCField p, CCField Ap) {
    const C3 e = l0.ext;
    const int g = l0.g;
    CCField xx = x, rr = r, pp = p, aa = Ap;
    FPV ac = l0.AC;
    const bool all = l0.allFluid;
    auto ks = ks_;
    ccReduce3Lanes(
        "mgpcg_update_sum", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x_, int y, int z, double& s) {
          if (ks(kStop) != 0.0)
            return;
          const long i = (long)x_ + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double a = ks(kAlpha), na = -a;
          xx(i) += a * pp(i);
          rr(i) += na * aa(i);
          if (all || ac(i) > 1e-30f)
            s += rr(i);
        },
        Kokkos::Sum<double, CCMem>(slot(kMsum)));
  }
  // C3 (§6 (iii)): removeMeanHostCounted's stop-guarded subtract (mean = kMsum / nFluid) and
  // maxabsTo's max|f| over the fluid cells into slot k in one pass. Max is order-free: bitwise
  // to mgmeans + mgmax. hostKrylovFusion() only.
  void meanSubtractMaxTo(Level& lv, CCField f, int k) {
    const C3 e = lv.ext;
    const int g = lv.g;
    CCField ff = f;
    FPV ac = lv.AC;
    const bool all = lv.allFluid;
    const long cnt = lv.nFluid;
    auto ks = ks_;
    ccReduce3(
        "mgmeans_max", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int z, double& m) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (!(cnt == 0 || ks(kStop) != 0.0)) {
            const double mean = ks(kMsum) / (double)cnt;
            if (all)
              ff(i) -= mean;
            else
              meanSubtractCell(ff, ac, i, mean);
          }
          if (all || ac(i) > 1e-30f) {
            const double v = Kokkos::fabs(ff(i));
            if (v > m)
              m = v;
          }
        },
        Kokkos::Max<double, CCMem>(slot(k)));
  }
  void maxabsTo(Level& lv, CCField a, int k) {
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a;
    FPV ac = lv.AC;
    if (hostAllFluid(lv)) {  // H-1: no AC read
      ccReduce3(
          "mgmax", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            const double v = Kokkos::fabs(aa(i));
            if (v > acc)
              acc = v;
          },
          Kokkos::Max<double, CCMem>(slot(k)));
      return;
    }
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
        Kokkos::Max<double, CCMem>(slot(k)));
  }
  // Test hook (tests/kokkos/test_pcg_breakdown.cpp): poison the pAp (which = 1) or r^T z
  // (which = 2) of iteration `iter` of the next single-rank PCG solves with a NaN; which = 0 (the
  // default) disarms it. Exercises the breakdown exits, which no healthy problem reaches.
  void setDebugBreakdown(int which, int iter) {
    dbgBrkWhich_ = which;
    dbgBrkIter_ = iter;
  }
  int solvePCGResident(CCField b, CCField x, CCField r, CCField p, CCField z, CCField Ap, int maxit,
                       double rtol, int pre, int post, int bottom, const StarOverlay* star,
                       int nStar, C3 nnStar) {
    solveFailed_ = false;  // ISSUES sweep item 6: per-solve breakdown flag
    OverlayScope overlay(overlaySolve_, star != nullptr);
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    ensureScalars();
    Level& l0 = lv_[0];  // H-2b: l0.x is not staged (no reader between or inside the solves)
    auto matvec = [&](CCField y, CCField v) {
      matvecOverlap(l0, y, v);
      if (star)
        starApplyDelta(y, CCConst(v), *star, nStar, nnStar, l0.ext, G, l0.ext, G, exactResidual_);
    };
    auto precond = [&](CCField zz, CCField rr) { precondVcycle(zz, rr); };  // A5
    if (zeroGuess_) {
      // H-2c (§8): the caller guarantees x0 = +0, so A x0 = +0 in every cell and
      // b + (-1)(+0) = b for every b (+-0 included): r = b with no matvec. The matvec's only
      // side effect on x, the ghost fill of the non-wrap path, is kept.
      if (!fusedWrapReads()) {
        fill(l0, x);
        applyOutflowGhost(l0, x);
      }
      Kokkos::deep_copy(CCExec(), r, b);
    } else {
      matvec(Ap, x);  // r = b - A x
      Kokkos::deep_copy(CCExec(), r, b);
      axpy(r, -1.0, Ap);
    }
    removeMean(l0, r);                // compatibility: project rhs/residual onto the range
    const double r0 = maxabs(l0, r);  // host read (once per solve)
    int it = 0;
    const bool trace = mgDebugLevel() >= 2 && dbgSolve_ < mgDebugSolves();
    if (trace)
      printf("[mg] solve %d: r0=%.6e rtol=%.1e (pre=%d post=%d bottom=%d)\n", dbgSolve_, r0, rtol,
             pre, post, bottom);
    ++dbgSolve_;
    const bool fuseC = hostKrylovFusion();  // C1-C4 (§6): the fused host kernels
    struct ExitDotScope {                   // C4's request never outlives this solve
      ExitDot& d;
      ~ExitDotScope() { d = ExitDot{}; }
    } exitDotScope{exitDot_};
    // D (§4.4.5): the FP32 V-cycle on an eligible solve; it takes e = ilogb(max|r|) of the vector
    // it preconditions (r0 before the loop, the packet's rn inside it) and forms r^T z in its exit
    const bool f32 = fp32VcycleSolve();
    lastVcycleFp32_ = f32;
    const bool health = trace || healthTrace_;
    healthLog_.clear();
    int eNext = 0;
    // C4: r^T z from the V-cycle's level-0 exit pass (and p = z there under copyDotTo's fusion
    // condition); a V-cycle without that exit (one level) leaves `done` false -> the old kernels
    auto precondDot = [&](int k, CCField pz) {
      if (f32) {
        precondVcycleFp32(z, r, eNext, k, pz);
        return true;
      }
      exitDot_ = ExitDot{fuseC ? k : -1, pz, false};
      precond(z, r);
      const bool done = exitDot_.done;
      exitDot_ = ExitDot{};
      return done;
    };
    if (r0 > 0.0 && std::isfinite(r0)) {
      const bool copyFused = kHostMemory && fusedWrapSmooth(l0) && lv_.size() > 1;
      eNext = std::ilogb(r0);
      if (!precondDot(kRz, copyFused ? p : CCField()))
        copyDotTo(l0, p, r, z, kRz);  // p = z; r^T z into kRz
      else if (!copyFused)
        Kokkos::deep_copy(CCExec(), p, z);
      const double rz0 = readSlot(kRz);  // host read (the initial guard)
      if (!std::isfinite(rz0)) {
        solveFailed_ = true;  // see solvePCG
        printf(
            "peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z; "
            "returning zero correction (reported as %d/%d iterations, i.e. a CAPPED solve)\n",
            maxit, maxit);
        Kokkos::deep_copy(CCExec(), x, 0.0);
        Kokkos::deep_copy(CCExec(), l0.x, x);
        if (strictPressure_)
          throw std::runtime_error(
              "peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z "
              "(set_pressure_strict)");
        return maxit;
      }
      setSlot(kStop, 0.0);
      auto ks = ks_;
      const std::size_t n = x.extent(0);
      CCField xx = x, rr = r, pp = p, zz = z, aa = Ap;
      const double brkPAp = kBrkPAp, brkRz = kBrkRz;  // by value into the kernels
      bool exited = false;
      for (; it < maxit; ++it) {
        if (fuseC && !meanRemovalAll_) {
          matvecDotTo(l0, Ap, p, kPAp);  // C1: Ap = A p and p^T Ap in one pass (star == nullptr)
        } else {
          matvec(Ap, p);
          if (meanRemovalAll_)
            removeMean(l0, Ap);  // A preserves mean-freeness; "fine" scope trusts that
          dotTo(l0, p, Ap, kPAp);
        }
        if (dbgBrkWhich_ == 1 && it == dbgBrkIter_)
          setSlot(kPAp, std::numeric_limits<double>::quiet_NaN());
        Kokkos::parallel_for(  // alpha = rz / pAp, or the pAp breakdown flag
            "mgpcg_alpha", Kokkos::RangePolicy<CCExec>(CCExec(), 0, 1), KOKKOS_LAMBDA(int) {
              const double pAp = ks(kPAp);
              if (ks(kStop) == 0.0) {
                if (Kokkos::isfinite(pAp) && pAp > 1e-300)
                  ks(kAlpha) = ks(kRz) / pAp;
                else
                  ks(kStop) = brkPAp;
              }
            });
        if (fuseC) {
          pcgUpdateSum(l0, x, r, p, Ap);  // C2: the update + the fluid sum of r
          meanSubtractMaxTo(l0, r, kRn);  // C3: the stop-guarded subtract + max|r|
        } else {
          Kokkos::parallel_for(  // x += alpha p; r -= alpha Ap (axpy's expressions)
              "mgpcg_update", Kokkos::RangePolicy<CCExec>(CCExec(), 0, n),
              KOKKOS_LAMBDA(std::size_t i) {
                if (ks(kStop) != 0.0)
                  return;
                const double a = ks(kAlpha), na = -a;
                xx(i) += a * pp(i);
                rr(i) += na * aa(i);
              });
          removeMean(l0, r, /*stopGuard=*/true);
          maxabsTo(l0, r, kRn);
        }
        Kokkos::deep_copy(kpk_, Kokkos::subview(ks_, std::make_pair(0, kPacket)));  // THE read
        const double pAp = kpk_(0), rn = kpk_(1), stop = kpk_(2);
        noteBottomFlag(kpk_(3));  // a non-finite scalar in a device bottom solve
        if (stop == kBrkRz) {     // r^T z of the previous iteration was non-finite
          solveFailed_ = true;    // ISSUES sweep item 6: a breakdown, not a convergence
          --it;                   // report the iteration that computed it
          exited = true;
          break;
        }
        if (stop == kBrkPAp) {
          if (!std::isfinite(pAp))
            solveFailed_ = true;  // ISSUES sweep item 6 (pAp <= 1e-300 is a CONVERGED direction)
          exited = true;
          break;  // keep the last finite iterate
        }
        if (trace)
          printf("[mg]   it %3d  |r|inf=%.6e  r/r0=%.4e\n", it + 1, rn, rn / r0);
        if (rn < rtol * r0) {
          ++it;
          exited = true;
          break;
        }
        eNext = std::ilogb(rn);
        const double rzOld = health ? dot(l0, r, z) : 0.0;  // r_{k+1}^T z_k (the instrument)
        if (!precondDot(kRzNew, CCField()))  // C4: r^T z_new from the V-cycle exit pass
          dotTo(l0, r, z, kRzNew);
        if (health) {
          const double rzNew = readSlot(kRzNew);
          const double ratio = rzNew != 0.0 ? std::abs(rzOld / rzNew) : 0.0;
          healthLog_.push_back(ratio);
          if (trace)
            printf("[mg]   it %3d  health |r^T z_k| / |r^T z_{k+1}| = %.3e%s\n", it + 1, ratio,
                   f32 ? " (fp32 V-cycle)" : "");
        }
        if (dbgBrkWhich_ == 2 && it == dbgBrkIter_)
          setSlot(kRzNew, std::numeric_limits<double>::quiet_NaN());
        Kokkos::parallel_for(  // beta = r^T z_new / r^T z; rz = r^T z_new -- or the flag
            "mgpcg_beta", Kokkos::RangePolicy<CCExec>(CCExec(), 0, 1), KOKKOS_LAMBDA(int) {
              const double rznew = ks(kRzNew);
              if (Kokkos::isfinite(rznew)) {
                ks(kBeta) = rznew / ks(kRz);
                ks(kRz) = rznew;
              } else {
                ks(kStop) = brkRz;
              }
            });
        Kokkos::parallel_for(  // p = z + beta p (aypx's expression)
            "mgaypx", Kokkos::RangePolicy<CCExec>(CCExec(), 0, n), KOKKOS_LAMBDA(std::size_t i) {
              if (ks(kStop) != 0.0)
                return;
              pp(i) = zz(i) + ks(kBeta) * pp(i);
            });
      }
      if (!exited && maxit > 0 && readSlot(kStop) == kBrkRz) {
        solveFailed_ = true;  // the last iteration's r^T z was non-finite
        it = maxit - 1;
      }
    }
    removeMean(l0, x);  // H-2b: in place (was the l0.x round trip; same cells, same values)
    return it;
  }

  // FLEXIBLE CG (Notay 2000 / Golub-Ye "inexact preconditioned CG"; the Polak-Ribiere form of
  // Axelsson's generalized CG), preconditioned by the SAME one symmetric V-cycle as solvePCG.
  //
  // The ONLY difference from solvePCG is the beta recurrence: Fletcher-Reeves
  //     beta = r_{k+1}^T z_{k+1} / (r_k^T z_k)
  // is replaced by Polak-Ribiere
  //     beta = r_{k+1}^T (z_{k+1} - z_k) / (r_k^T z_k),
  // at the cost of one extra stored vector (z_k) and one extra global dot per iteration. The two
  // forms are ALGEBRAICALLY IDENTICAL when the preconditioner is a fixed SPD operator, because
  // then r_{k+1} is M^{-1}A-orthogonal to z_k, i.e. r_{k+1}^T z_k = 0 exactly -- so on a healthy
  // problem FCG must reproduce PCG's iteration count (that equality is the sanity gate). When the
  // preconditioner is NOT symmetric w.r.t. the fine operator, that orthogonality is lost, the
  // Fletcher-Reeves numerator is contaminated by a term CG has no right to, and the iteration
  // stalls; Polak-Ribiere subtracts exactly that term, which is why FCG converging where PCG
  // stalls is a DIAGNOSIS ("the preconditioner is nonsymmetric") and not merely a fix.
  //
  // Everything else -- matvec (incl. the optional star overlay), preconditioner, mean removal,
  // the maxabs(r) < rtol*r0 stopping estimate, the breakdown guards, the final mean removal -- is
  // identical to solvePCG line for line. Keep the two in sync if either is ever changed.
  // Scratch: solvePCG's five level-0 fields plus `zp` (the previous preconditioned residual).
  int solveFCG(CCField b, CCField x, CCField r, CCField p, CCField z, CCField zp, CCField Ap,
               int maxit, double rtol, int pre, int post, int bottom,
               const StarOverlay* star = nullptr, int nStar = 0, C3 nnStar = C3{0, 0, 0}) {
    solveFailed_ = false;  // ISSUES sweep item 6: per-solve breakdown flag
    OverlayScope overlay(overlaySolve_, star != nullptr);
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    Kokkos::deep_copy(CCExec(), l0.x, x);
    auto matvec = [&](CCField y, CCField v) {
      matvecOverlap(l0, y, v);
      if (star)
        starApplyDelta(y, CCConst(v), *star, nStar, nnStar, l0.ext, G, l0.ext, G, exactResidual_);
    };
    // D (§4.4.5): the FP32 V-cycle on an eligible solve, e = ilogb(max|r|); FCG keeps its own dots
    const bool f32 = fp32VcycleSolve();
    lastVcycleFp32_ = f32;
    healthLog_.clear();
    auto precond = [&](CCField zz, CCField rr, double rn) {
      if (f32)
        precondVcycleFp32(zz, rr, std::ilogb(rn), -1, CCField());
      else
        precondVcycle(zz, rr);  // A5
    };
    matvec(Ap, x);  // r = b - A x
    Kokkos::deep_copy(CCExec(), r, b);
    axpy(r, -1.0, Ap);
    removeMean(l0, r);  // compatibility: project rhs/residual onto the range
    const double r0 = maxabs(l0, r);
    // The stop reference: r0 (the default), or the caller's FULL right-hand-side norm when a
    // stop reference is set (setStopReference; only the balanced-force projection sets it).
    // With it unset rref IS r0, so `rtol * rref` is bit-for-bit the old test.
    const double rref = stopRef_ > 0.0 ? stopRef_ : r0;
    int it = 0;
    int dbgRank = 0;
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      MPI_Comm_rank(comm_, &dbgRank);
#endif
    const bool trace = mgDebugLevel() >= 2 && dbgSolve_ < mgDebugSolves() && dbgRank == 0;
    if (trace)
      printf("[mg] fcg solve %d: r0=%.6e rtol=%.1e (pre=%d post=%d bottom=%d)\n", dbgSolve_, r0,
             rtol, pre, post, bottom);
    ++dbgSolve_;
    if (r0 > 0.0 && std::isfinite(r0)) {
      precond(z, r, r0);
      Kokkos::deep_copy(CCExec(), p, z);
      double rz = dot(l0, r, z);
      if (!std::isfinite(rz)) {
        solveFailed_ = true;  // ISSUES sweep item 6 -- see solvePCG for the mechanism
        printf(
            "peclet::flow CutcellMG::solveFCG: preconditioner produced non-finite z; "
            "returning zero correction (reported as %d/%d iterations, i.e. a CAPPED solve)\n",
            maxit, maxit);
        Kokkos::deep_copy(CCExec(), x, 0.0);
        Kokkos::deep_copy(CCExec(), l0.x, x);
        if (strictPressure_)
          throw std::runtime_error(
              "peclet::flow CutcellMG::solveFCG: preconditioner produced non-finite z "
              "(set_pressure_strict)");
        return maxit;
      }
      for (; it < maxit; ++it) {
        matvec(Ap, p);
        if (meanRemovalAll_)
          removeMean(l0, Ap);  // A preserves mean-freeness; "fine" scope trusts that
        const double pAp = dot(l0, p, Ap);
        if (!std::isfinite(pAp) || pAp <= 1e-300) {
          if (!std::isfinite(pAp))
            solveFailed_ = true;  // ISSUES sweep item 6 (pAp <= 1e-300 is a CONVERGED direction)
          break;                  // keep the last finite iterate
        }
        const double alpha = rz / pAp;
        axpy(x, alpha, p);
        axpy(r, -alpha, Ap);
        removeMean(l0, r);
        const double rn = maxabs(l0, r);
        if (trace)
          printf("[mg]   it %3d  |r|inf=%.6e  r/r0=%.4e\n", it + 1, rn, rn / r0);
        if (rn < rtol * rref) {
          ++it;
          break;
        }
        Kokkos::deep_copy(CCExec(), zp, z);  // z_k, before the preconditioner overwrites it
        precond(z, r, rn);
        const double rznew = dot(l0, r, z), rzcross = dot(l0, r, zp);
        if (!std::isfinite(rznew) || !std::isfinite(rzcross)) {
          solveFailed_ = true;  // ISSUES sweep item 6: a breakdown, not a convergence
          break;                // preconditioner breakdown: keep the last finite iterate
        }
        const double beta = (rznew - rzcross) / rz;  // Polak-Ribiere: r^T (z_{k+1} - z_k) / r^T z
        // DIAGNOSTIC (PECLET_FLOW_MG_DEBUG>=2): `pr` is |r_{k+1}^T z_k| / |r_{k+1}^T z_{k+1}| --
        // the term Fletcher-Reeves keeps and Polak-Ribiere removes. With a preconditioner that is
        // symmetric w.r.t. the fine operator this is EXACTLY zero in exact arithmetic, so a
        // measured pr ~ 1e-14 says "the V-cycle is symmetric here (and FCG == PCG by construction)"
        // while pr = O(1) says "it is not" -- a direct read-out of the hypothesis, per iteration,
        // that needs no extra solve.
        if (trace)
          printf("[mg]   it %3d  beta=%.6e  pr=%.3e (|r^T z_k| / |r^T z_{k+1}|)\n", it + 1, beta,
                 rznew != 0.0 ? std::abs(rzcross / rznew) : 0.0);
        if (healthTrace_)  // §4.4.5: FCG forms this dot anyway
          healthLog_.push_back(rznew != 0.0 ? std::abs(rzcross / rznew) : 0.0);
        aypx(p, beta, z);
        rz = rznew;
      }
    }
    Kokkos::deep_copy(CCExec(), l0.x, x);
    removeMean(l0, l0.x);
    Kokkos::deep_copy(CCExec(), x, l0.x);
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
                    const GpOverlayReal<MReal>& ov, int nOv, C3 nn
#ifdef PECLET_FLOW_MPI
                    ,
                    CCField xg2 = CCField(), GridHalo<double>* h2 = nullptr, C3 ext2 = C3{0, 0, 0}
#endif
  ) {
    solveFailed_ = false;                       // ISSUES sweep item 6: per-solve breakdown flag
    OverlayScope overlay(overlaySolve_, true);  // A3: the gp overlay keeps today's fills
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
        applyCutcellOpFace(y, CCConst(q), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ), l0.ext,
                           G);
        gpApplyDelta(y, CCConst(xg2), ov, nOv, nn, l0.ext, G, ext2, 2, /*useGhost=*/true);
        return;
      }
#endif
      fill(l0, q);
      applyOutflowGhost(l0, q);
      applyCutcellOpFace(y, CCConst(q), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ), l0.ext,
                         G);
      gpApplyDelta(y, CCConst(q), ov, nOv, nn, l0.ext, G, l0.ext, G);
    };
    auto precond = [&](CCField zz, CCField rr) {
      Kokkos::deep_copy(CCExec(), l0.rhs, rr);
      Kokkos::deep_copy(CCExec(), l0.x, 0.0);
      vcycle(0, /*sym=*/true);
      Kokkos::deep_copy(CCExec(), zz, l0.x);
    };
    matvec(t, x);  // r = b - A x  (t as scratch)
    Kokkos::deep_copy(CCExec(), r, b);
    axpy(r, -1.0, t);
    removeMean(l0, r);
    Kokkos::deep_copy(CCExec(), rh, r);  // shadow residual r^ = r_0
    const double r0n = maxabs(l0, r);
    int it = 0;
    if (r0n > 0.0 && std::isfinite(r0n)) {
      double rho = 1.0, alpha = 1.0, omega = 1.0;
      double best = r0n;
      int lastImprove = 0;
      Kokkos::deep_copy(CCExec(), p, 0.0);
      Kokkos::deep_copy(CCExec(), v, 0.0);
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
    Kokkos::deep_copy(CCExec(), l0.x, x);
    removeMean(l0, l0.x);
    Kokkos::deep_copy(CCExec(), x, l0.x);
    return it;
  }

 public:  // (public for nvcc extended-lambda rule)
          // Per-level V-cycle wall time (PECLET_FLOW_MG_DEBUG>=3; HOST backends only — no device
          // fence, so on CUDA the numbers are launch times, not kernel times). Answers "how much of
          // the solve is spent on the small coarse levels", i.e. whether coarse-level launch
          // overhead is worth chasing.
#ifdef PECLET_FLOW_MPI

  // Telescope data movement (host-staged; the stage is a coarse level, i.e. small), through the
  // stage's core RedistributeTopology. Gather: every member's INNER cells land in the merged
  // block at (member origin - merged origin + g) on the group root (core's forward: a group
  // Gatherv). Scatter-add: the inverse (core's backward: a group Scatterv, which OVERWRITES its
  // landing buffer), then each member ADDS the received box into `dst`'s inner cells (prolongAdd
  // is additive — flow's arithmetic, not the movement's). Pure data movement, bitwise.
  void teleGather(const Level& lv, CCField src, CCField dst) {
    Telescope& T = *lv.tele;
    auto hs = Kokkos::create_mirror_view(src);
    Kokkos::deep_copy(hs, src);
    if (!T.root()) {
      T.move.forward({hs.data()}, {});
      return;
    }
    auto hd = Kokkos::create_mirror_view(dst);
    T.move.forward({hs.data()}, {hd.data()});
    Kokkos::deep_copy(dst, hd);
  }
  // WO-R2 outflow coefficient across a telescope point: the HIGH-side outflow coefficient of a
  // level lives on the first ghost index beyond the inner block along the outflow axis (a plane
  // teleGather's inner-only pack leaves behind). Gather that plane from the members touching the
  // global +face into the merged stage buffer's own ghost plane, so the coarse level coarsens its
  // boundary coefficient from the stage exactly as it would from a finer level in place.
  void teleGatherOutflowPlanes(const Level& lv) {
    if (!outflowCoeff_)
      return;
    CCField src[3] = {lv.ox, lv.oy, lv.oz};
    CCField dst[3] = {lv.tele->ox, lv.tele->oy, lv.tele->oz};
    for (int a = 0; a < 3; ++a)
      if (bc_[2 * a + 1] == 3) {
        if (lv.tele->repartition())
          teleGatherPlaneRepartition(lv, src[a], dst[a], a);
        else
          teleGatherPlane(lv, src[a], dst[a], a);
      }
  }
  // The same plane across a REPARTITION stage, where there are no merge groups: a source block
  // touching the global +face sends each target block touching it the part of its plane that lies
  // in the target's transverse (b, c) rectangle — the box intersections, computed from the two
  // replicated decompositions, exactly as core's RedistributeTopology does for the inner cells.
  // One MPI_Alltoallv on the level's communicator (every rank takes part; most counts are zero),
  // cells in (c, b) order on both sides; the target lands them on its stage buffer's own ghost
  // plane. Pure data movement.
  void teleGatherPlaneRepartition(const Level& lv, CCField src, CCField dst, int a) {
    Telescope& T = *lv.tele;
    using Blk = peclet::core::decomp::Block<3>;
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const long gd[3] = {lv.gdim.x, lv.gdim.y, lv.gdim.z};
    int rank = 0, np = 1;
    MPI_Comm_rank(T.comm.parent, &rank);
    MPI_Comm_size(T.comm.parent, &np);
    auto touches = [&](const Blk& k) { return (long)(k.origin[a] + k.size[a]) == gd[a]; };
    // transverse overlap [lo, hi) of two blocks on axes b and c; false when empty
    auto overlap = [&](const Blk& p, const Blk& q, long lo[2], long hi[2]) {
      const int ax[2] = {b, c};
      for (int e = 0; e < 2; ++e) {
        lo[e] = (long)std::max(p.origin[ax[e]], q.origin[ax[e]]);
        hi[e] = (long)std::min(p.origin[ax[e]] + p.size[ax[e]], q.origin[ax[e]] + q.size[ax[e]]);
        if (hi[e] <= lo[e])
          return false;
      }
      return true;
    };
    std::vector<int> sc((std::size_t)np, 0), sd((std::size_t)np, 0), rc((std::size_t)np, 0),
        rd((std::size_t)np, 0);
    std::vector<double> sb, rb;
    const Blk me = T.srcDec.block((std::size_t)rank);
    if (touches(me)) {
      const int g = lv.g;
      const int ext[3] = {lv.ext.x, lv.ext.y, lv.ext.z};
      const long st[3] = {1, (long)lv.ext.x, (long)lv.ext.x * lv.ext.y};
      auto hs = Kokkos::create_mirror_view(src);
      Kokkos::deep_copy(hs, src);
      const long pa = (long)(ext[a] - g) * st[a];
      for (std::size_t t = 0; t < T.dstDec.numBlocks(); ++t) {
        long lo[2], hi[2];
        const Blk tb = T.dstDec.block(t);
        if (!touches(tb) || !overlap(me, tb, lo, hi))
          continue;
        sd[t] = (int)sb.size();
        for (long k = lo[1]; k < hi[1]; ++k)
          for (long j = lo[0]; j < hi[0]; ++j)
            sb.push_back(hs(pa + (j - (long)me.origin[b] + g) * st[b] +
                            (k - (long)me.origin[c] + g) * st[c]));
        sc[t] = (int)sb.size() - sd[t];
      }
    }
    const bool land = T.root() && touches(T.dstDec.block((std::size_t)rank));
    if (land) {
      const Blk tb = T.dstDec.block((std::size_t)rank);
      int n = 0;
      for (std::size_t s2 = 0; s2 < T.srcDec.numBlocks(); ++s2) {
        long lo[2], hi[2];
        const Blk sk = T.srcDec.block(s2);
        if (!touches(sk) || !overlap(sk, tb, lo, hi))
          continue;
        rd[s2] = n;
        rc[s2] = (int)((hi[0] - lo[0]) * (hi[1] - lo[1]));
        n += rc[s2];
      }
      rb.resize((std::size_t)n);
    }
    MPI_Alltoallv(sb.data(), sc.data(), sd.data(), MPI_DOUBLE, rb.data(), rc.data(), rd.data(),
                  MPI_DOUBLE, T.comm.parent);
    if (!land)
      return;
    const Blk tb = T.dstDec.block((std::size_t)rank);
    const long mst[3] = {1, (long)T.mExt.x, (long)T.mExt.x * T.mExt.y};
    const int mext[3] = {T.mExt.x, T.mExt.y, T.mExt.z};
    auto hd = Kokkos::create_mirror_view(dst);
    Kokkos::deep_copy(hd, dst);
    const long pa = (long)(mext[a] - T.g) * mst[a];
    for (std::size_t s2 = 0; s2 < T.srcDec.numBlocks(); ++s2) {
      if (rc[s2] == 0)
        continue;
      long lo[2], hi[2];
      overlap(T.srcDec.block(s2), tb, lo, hi);
      const double* q = rb.data() + rd[s2];
      for (long k = lo[1]; k < hi[1]; ++k)
        for (long j = lo[0]; j < hi[0]; ++j)
          hd(pa + (j - (long)tb.origin[b] + T.g) * mst[b] +
             (k - (long)tb.origin[c] + T.g) * mst[c]) = *q++;
    }
    Kokkos::deep_copy(dst, hd);
  }
  void teleGatherPlane(const Level& lv, CCField src, CCField dst, int a) {
    Telescope& T = *lv.tele;
    const int g = lv.g, b = (a + 1) % 3, c = (a + 2) % 3;
    const int ext[3] = {lv.ext.x, lv.ext.y, lv.ext.z},
              inn[3] = {lv.inner.x, lv.inner.y, lv.inner.z};
    const long st[3] = {1, (long)lv.ext.x, (long)lv.ext.x * lv.ext.y};
    const bool mine = touchesGlobalFace(lv, 2 * a + 1);
    // pack my plane (ghost index ext[a]-g along a; inner ranges along b, c) if I touch the +face
    std::vector<double> sb;
    if (mine) {
      auto hs = Kokkos::create_mirror_view(src);
      Kokkos::deep_copy(hs, src);
      sb.resize((std::size_t)inn[b] * inn[c]);
      const long pa = (long)(ext[a] - g) * st[a];
      for (int k = 0; k < inn[c]; ++k)
        for (int j = 0; j < inn[b]; ++j)
          sb[(std::size_t)j + (std::size_t)k * inn[b]] =
              hs(pa + (long)(j + g) * st[b] + (long)(k + g) * st[c]);
    }
    // the root knows which members touch the face from their block boxes (replicated)
    std::vector<int> counts, displs;
    std::vector<double> rb;
    const int gd[3] = {lv.gdim.x, lv.gdim.y, lv.gdim.z};
    if (T.root()) {
      int disp = 0;
      for (int m = 0; m < T.nMembers; ++m) {
        const int mo[3] = {T.memO[m].x, T.memO[m].y, T.memO[m].z};
        const int ms[3] = {T.memS[m].x, T.memS[m].y, T.memS[m].z};
        const int n = (mo[a] + ms[a] == gd[a]) ? ms[b] * ms[c] : 0;
        counts.push_back(n);
        displs.push_back(disp);
        disp += n;
      }
      rb.resize((std::size_t)disp);
    }
    MPI_Gatherv(sb.data(), (int)sb.size(), MPI_DOUBLE, T.root() ? rb.data() : nullptr,
                T.root() ? counts.data() : nullptr, T.root() ? displs.data() : nullptr, MPI_DOUBLE,
                0, T.comm.group);
    if (!T.root())
      return;
    const int mext[3] = {T.mExt.x, T.mExt.y, T.mExt.z};
    const long mst[3] = {1, (long)T.mExt.x, (long)T.mExt.x * T.mExt.y};
    const int mog[3] = {T.mOg.x, T.mOg.y, T.mOg.z};
    auto hd = Kokkos::create_mirror_view(dst);
    Kokkos::deep_copy(hd, dst);
    const long pa = (long)(mext[a] - T.g) * mst[a];
    for (int m = 0; m < T.nMembers; ++m) {
      if (counts[m] == 0)
        continue;
      const int mo[3] = {T.memO[m].x, T.memO[m].y, T.memO[m].z};
      const int ms[3] = {T.memS[m].x, T.memS[m].y, T.memS[m].z};
      const double* q = rb.data() + displs[m];
      for (int k = 0; k < ms[c]; ++k)
        for (int j = 0; j < ms[b]; ++j)
          hd(pa + (long)(mo[b] - mog[b] + j + T.g) * mst[b] +
             (long)(mo[c] - mog[c] + k + T.g) * mst[c]) =
              q[(std::size_t)j + (std::size_t)k * ms[b]];
    }
    Kokkos::deep_copy(dst, hd);
  }
  void teleScatterAdd(const Level& lv, CCField src, CCField dst) {
    Telescope& T = *lv.tele;
    const int g = lv.g;
    if (T.root()) {
      auto hs = Kokkos::create_mirror_view(src);
      Kokkos::deep_copy(hs, src);
      T.move.backward({hs.data()}, {T.back.data()});
    } else {
      T.move.backward({}, {T.back.data()});
    }
    auto hd = Kokkos::create_mirror_view(dst);
    Kokkos::deep_copy(hd, dst);
    for (int k = 0; k < lv.inner.z; ++k)
      for (int j = 0; j < lv.inner.y; ++j)
        for (int i = 0; i < lv.inner.x; ++i) {
          const long id =
              (long)(i + g) + (long)(j + g) * lv.ext.x + (long)(k + g) * (long)lv.ext.x * lv.ext.y;
          hd(id) += T.back[(std::size_t)id];
        }
    Kokkos::deep_copy(dst, hd);
  }
#endif
  // A5 (§5.5): z = M r with M one symmetric V-cycle, run IN PLACE on the caller's vectors: level
  // 0's `rhs` and `x` are rebound to `rr` and `zz` (shallow View assignment) for the cycle and
  // restored after it, replacing the `rhs <- r`, `x <- 0`, `z <- x` full-field copies. Relies on
  // level 0 never WRITING `rhs` (only a communication-avoiding level exchanges its rhs, and those
  // are coarse).
  void precondVcycle(CCField zz, CCField rr) {
    Level& l0 = lv_[0];
    assert(!l0.caOk && "A5: level 0 must not exchange (write) its rhs");
    struct Restore {
      Level& l;
      CCField rhs, x;
      ~Restore() {
        l.rhs = rhs;
        l.x = x;
      }
    } restore{l0, l0.rhs, l0.x};
    l0.rhs = rr;
    l0.x = zz;
    Kokkos::deep_copy(CCExec(), zz, 0.0);
    vcycle(0, /*sym=*/true);
  }
  void vcycle(int L, bool sym) {
    if (L == 0) {  // D (§4.4.6): every FP64 V-cycle passes here, the FP32 one never does
      lastVcycleFp32_ = false;
      if (vprec_ == kVcycleFp32) {
        const char* why = fp32VcycleIneligible(true);
        throw std::runtime_error(
            std::string("set_pressure_vcycle_precision('fp32'): not eligible here: ") +
            (why ? why : fp32VcycleIneligible(false)));
      }
    }
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
    // The bottom is the last level THIS rank holds that is not a telescope point: a rank idling
    // below a telescope point ends its lv_ at that point and must take the transition branch
    // (gather, skip, scatter), never the bottom solve.
    bool isBottom = (L + 1 == (int)lv_.size());
#ifdef PECLET_FLOW_MPI
    if (lv.tele)
      isBottom = false;
#endif
    if (isBottom) {
      if (bottomSolver_ == kBottomDirect) {
        if (const char* why = directBottomIneligible())
          throw std::runtime_error(
              std::string("set_pressure_bottom_solver('direct'): not eligible here: ") + why);
      }
      if (agglomerateBottom()) {
        if (bottomSolver_ != kBottomAlgebraic && directBottomIneligible() == nullptr)
          directBottomSolve();  // §13: FCG + the FP32 direct factor (device, one launch)
        else
          graphAmgSolveBottom(
              lv);  // agglomerated mesh-agnostic coarse solve (decomposition-agnostic)
      } else
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
      residualCutcellFace(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX),
                          FPC(lv.AFY), FPC(lv.AFZ), lv.ext, lv.g);
    };
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const int g = lv.g;
      const C3 lo{g + 1, g + 1, g + 1};
      const C3 hi{lv.ext.x - g - 1, lv.ext.y - g - 1, lv.ext.z - g - 1};
      lv.dev->exchangeBegin(lv.x);
      residualCutcellBoxFace(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX),
                             FPC(lv.AFY), FPC(lv.AFZ), lv.ext, lo, hi, C3{0, 0, 0}, C3{0, 0, 0});
      lv.dev->exchangeEnd(lv.x);
      applyOutflowGhost(lv, lv.x, g);
      residualCutcellBoxFace(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX),
                             FPC(lv.AFY), FPC(lv.AFZ), lv.ext, C3{g, g, g},
                             C3{lv.ext.x - g, lv.ext.y - g, lv.ext.z - g}, lo, hi);
    } else
#endif
    {
      if (fusedWrapReads()) {  // A3: the residual reads the periodic wrap directly, no fill
        residualCutcellFaceWrap(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX),
                                FPC(lv.AFY), FPC(lv.AFZ), lv.ext, lv.inner, lv.g);
      } else {
        fill(lv, lv.x);  // single-rank: the periodic wrap copy
        applyOutflowGhost(lv, lv.x, lv.g);
        fullResidual();
      }
    }
#ifdef PECLET_FLOW_MPI
    if (lv.tele) {
      // Telescoped transition: the residual goes DOWN at this level's own resolution onto the group
      // root (the merged block is coarsenable; this rank's block is not), the roots restrict /
      // recurse / prolong on the sub-communicator, and the correction comes back the same way.
      Telescope& T = *lv.tele;
      teleGather(lv, lv.res, T.res);
      if (T.root()) {
        Level& cs = lv_[L + 1];
        restrictAvg(cs.rhs, CCConst(T.res), cs.ext, T.mExt, cs.g, T.g, cs.inner, lv.ratio);
        Kokkos::deep_copy(CCExec(), cs.x, 0.0);
        vcycle(L + 1, sym);
        fill(cs, cs.x);
        applyOutflowGhost(cs, cs.x, cs.g);
        applyNeumannGhost(cs, cs.x, cs.g);
        Kokkos::deep_copy(CCExec(), T.x, 0.0);
        prolongAdd(T.x, CCConst(cs.x), T.mExt, cs.ext, T.g, cs.g, T.mInner, lv.ratio);
      }
      teleScatterAdd(lv, T.x, lv.x);
    } else
#endif
    {
      Level& cs = lv_[L + 1];
      if (!distributed_) {  // A5: the restriction zeroes cs.x's inner cells itself
        restrictAvgZeroX(cs.rhs, cs.x, CCConst(lv.res), cs.ext, lv.ext, cs.g, lv.g, cs.inner,
                         lv.ratio);
      } else {
        restrictAvg(cs.rhs, CCConst(lv.res), cs.ext, lv.ext, cs.g, lv.g, cs.inner, lv.ratio);
        Kokkos::deep_copy(CCExec(), cs.x, 0.0);
      }
      vcycle(L + 1, sym);
      fill(cs, cs.x);
      applyOutflowGhost(cs, cs.x, cs.g);
      // The trilinear prolongation reads the coarse ghost with weight 1/4 regardless of openness,
      // so a domain-BC face needs its real ghost policy here (Dirichlet 0 above, Neumann
      // zero-gradient below) instead of the periodic wrap fill() just wrote. See applyNeumannGhost.
      applyNeumannGhost(cs, cs.x, cs.g);
      prolongAdd(lv.x, CCConst(cs.x), lv.ext, cs.ext, lv.g, cs.g, lv.inner, lv.ratio);
    }
    smooth(lv, post_, /*reverse=*/sym);
    if (L == 0 && exitDot_.k >= 0) {  // C4: the exit subtract also forms r^T z (solvePCGResident)
      removeMeanDotTo(lv, exitDot_.k, exitDot_.p);
      exitDot_.done = true;
      return;
    }
    if (meanRemovalAll_ || L == 0)
      removeMean(lv, lv.x);
  }
  // Communication-avoiding smoothing on this level? Needs the width-2 topology (caOk), and the
  // periodic/IBM operator — with domain BCs the ring rows would need post-BC ghost openness the
  // exchange does not deliver, so those keep the per-colour exchange.
  bool caSmooth(const Level& lv) const { return distributed_ && lv.caOk && !hasBC_; }
  // A3 (§5.4) eligibility of the fused periodic-wrap reads: single rank, no outflow face (whose
  // Dirichlet ghost is not a copy), and no overlay on the current solve (the star /
  // ghost-projection paths keep today's fills). The smoother additionally needs every inner
  // dimension even (see cutcellSmoothColorFaceWrap); the read-only residual and matvec need no
  // parity.
  bool fusedWrapReads() const { return !distributed_ && !hasOutflow_ && !overlaySolve_; }
  bool fusedWrapSmooth(const Level& lv) const {
    return fusedWrapReads() && lv.inner.x % 2 == 0 && lv.inner.y % 2 == 0 && lv.inner.z % 2 == 0;
  }
  // --- D: the FP32 V-cycle preconditioner (doc/vof_projection_cost_design.md §4) -------------
  // The V-cycle precision mode (§4.4.6): 'auto' = FP32 where fp32VcycleIneligible() is null, else
  // FP64; 'fp64'; 'fp32' = FP32 or a raise naming the failed condition.
  enum : int { kVcycleAuto = 0, kVcycleFp64 = 1, kVcycleFp32 = 2 };
  // Does level L carry the FP32 data (§4.4.2)? Single rank, no outflow face, a non-bottom level,
  // and the mode is not 'fp64'. Allocates the level's six FP32 arrays (zero) on first use; a level
  // that does not carry them is marked stale (wOk = false).
  bool fp32Build(int L) {
    Level& lv = lv_[L];
    if (vprec_ == kVcycleFp64 || distributed_ || hasOutflow_ || L + 1 >= (int)lv_.size()) {
      lv.wOk = false;
      return false;
    }
    if (lv.WX.extent(0) != lv.n) {
      lv.WX = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_wx", lv.n);
      lv.WY = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_wy", lv.n);
      lv.WZ = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_wz", lv.n);
      lv.xf = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_xf", lv.n);
      lv.rhsf = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_rhsf", lv.n);
      lv.resf = Kokkos::View<VReal*, CCMem>("peclet::flow::mg_resf", lv.n);
    }
    return true;
  }
  // The FP32 weights of every carrying level from the STORED level openness (the setter's
  // rebuild; setOpenness builds them in its own kernel). Same operands as setOpenness: level L's
  // metric is gf * (1 / cfac^2), which is gf itself on level 0.
  void buildFp32Weights() {
    if (!opBuilt_)
      return;  // the next setOpenness builds them
    for (int L = 0; L < (int)lv_.size(); ++L) {
      Level& lv = lv_[L];
      if (!fp32Build(L))
        continue;
      const double sx = 1.0 / (double)(lv.cfac.x * lv.cfac.x),
                   sy = 1.0 / (double)(lv.cfac.y * lv.cfac.y),
                   sz = 1.0 / (double)(lv.cfac.z * lv.cfac.z);
      lv.wOk = buildCutcellOpFaceFp32(lv.AC, lv.AFX, lv.AFY, lv.AFZ, lv.WX, lv.WY, lv.WZ,
                                      CCConst(lv.ox), CCConst(lv.oy), CCConst(lv.oz), lv.ext, lv.g,
                                      gfx_ * sx, gfy_ * sy, gfz_ * sz, false) == 0;
    }
#ifndef NDEBUG
    checkFp32DecoupledSets();
#endif
  }
  // §4.4.1, §4.4.6: does this solve run the FP32 V-cycle? Evaluated once per solve by
  // solvePCGResident and solveFCG; 'fp32' raises naming the failed condition where ineligible.
  bool fp32VcycleSolve() {
    if (vprec_ == kVcycleFp64)
      return false;
    if (const char* why = fp32VcycleIneligible(true)) {
      if (vprec_ == kVcycleFp32)
        throw std::runtime_error(
            std::string("set_pressure_vcycle_precision('fp32'): not eligible here: ") + why);
      return false;
    }
    return true;
  }
  // §4.4.5 health instrument: |r_{k+1}^T z_k| / |r_{k+1}^T z_{k+1}| per iteration of the
  // single-rank PCG / FCG, formed (one extra dot) only while tracing (PECLET_FLOW_MG_DEBUG >= 2,
  // printed) or under this test hook (recorded, healthLog(): the last solve's ratios).
  void setHealthTrace(bool on) { healthTrace_ = on; }
  const std::vector<double>& healthLog() const { return healthLog_; }
  // §4.4.4: one FP32 V-cycle z = M r on an eligible solve (the caller checked
  // fp32VcycleIneligible). e = ilogb(max|r|) of the vector being preconditioned: the input is
  // scaled by 2^-e and the output by 2^e, both exact. k >= 0: r^T z over the fluid cells into slot
  // k (and p = z when p is allocated) in the exit pass; k < 0: no dot. z's ghosts are not written
  // (the A3 wrap paths never read them). Uses the schedule (pre_, post_, bottom_) of the driver.
  void precondVcycleFp32(CCField zz, CCField rr, int e, int k, CCField p) {
    Level& l0 = lv_[0];
    ensureScalars();
    // 1. entry: rhsf = fl32(2^-e r); the first pre-smooth colour pass from x = 0 (colour 0)
    entryFp32(l0.xf, l0.rhsf, CCConst(rr), VConst(l0.WX), VConst(l0.WY), VConst(l0.WZ),
              std::ldexp(1.0, -e), l0.ext, parityOg(l0), l0.g, /*color=*/0, pre_ > 0);
    vcycleFp32Impl(0, /*sym=*/true);                 // 2., 3.
    exitFp32(l0, zz, rr, std::ldexp(1.0, e), k, p);  // 4.
  }
  // The red-black sweeps of smooth() on the FP32 data (wrap reads; every non-bottom level is
  // fusedWrapSmooth by eligibility). skipFirst: the entry pass already did the first colour.
  void smoothFp32(Level& lv, int sweeps, bool reverse, bool skipFirst) {
    const C3 og = parityOg(lv);
    for (int k = 0; k < sweeps; ++k)
      for (int s = 0; s < 2; ++s) {
        if (skipFirst) {
          skipFirst = false;
          continue;
        }
        const int color = reverse ? (1 - s) : s;
        cutcellSmoothColorFp32Wrap(lv.xf, VConst(lv.rhsf), VConst(lv.WX), VConst(lv.WY),
                                   VConst(lv.WZ), lv.ext, lv.inner, og, lv.g, color);
      }
  }
  // §4.4.4 item 2 (and 3 at the bottom interface): vcycleImpl's single-rank schedule on the FP32
  // data of level L < bottom.
  void vcycleFp32Impl(int L, bool sym) {
    Level& lv = lv_[L];
    smoothFp32(lv, pre_, false, /*skipFirst=*/L == 0 && pre_ > 0);
    residualFp32Wrap(lv.resf, VConst(lv.xf), VConst(lv.rhsf), VConst(lv.WX), VConst(lv.WY),
                     VConst(lv.WZ), lv.ext, lv.inner, lv.g);
    Level& cs = lv_[L + 1];
    if (L + 2 == (int)lv_.size()) {  // the bottom: its FP64 x, rhs and engine, as today
      restrictFp32ToDZeroX(cs.rhs, cs.x, VConst(lv.resf), cs.ext, lv.ext, cs.g, lv.g, cs.inner,
                           lv.ratio);
      vcycle(L + 1, sym);
      fill(cs, cs.x);
      applyOutflowGhost(cs, cs.x, cs.g);
      applyNeumannGhost(cs, cs.x, cs.g);
      prolongFp32FromD(lv.xf, CCConst(cs.x), lv.ext, cs.ext, lv.g, cs.g, lv.inner, lv.ratio);
    } else {
      restrictFp32ZeroX(cs.rhsf, cs.xf, VConst(lv.resf), cs.ext, lv.ext, cs.g, lv.g, cs.inner,
                        lv.ratio);
      vcycleFp32Impl(L + 1, sym);
      fillWrap(cs, cs.xf);  // fill()'s single-rank periodic copy (no outflow face: eligibility)
      applyNeumannGhost(cs, cs.xf, cs.g);
      prolongFp32(lv.xf, VConst(cs.xf), lv.ext, cs.ext, lv.g, cs.g, lv.inner, lv.ratio);
    }
    smoothFp32(lv, post_, /*reverse=*/sym, false);
  }
  // §4.4.4 item 4, the level-0 EXIT (replaces removeMean(l0, z) and the r^T z dot): the fluid
  // mean m of (double)xf (C0's lane order on a host backend), then one pass z = s ((double)xf - m)
  // on fluid cells and z = s (double)xf on the other inner cells (s = 2^e), accumulating r^T z over
  // the fluid cells into slot k (C4's order) when k >= 0, and p = z when p is allocated.
  void exitFp32(Level& l0, CCField zz, CCField rr, double s, int k, CCField p) {
    const C3 e = l0.ext;
    const int g = l0.g;
    VConst xf = l0.xf;
    CCField z = zz, pp = p;
    CCConst r = rr;
    FPV ac = l0.AC;
    const bool counted = kHostMemory && l0.nFluid >= 0;  // H-1: the precomputed fluid count
    const bool all = counted && l0.allFluid;
    const long cnt0 = l0.nFluid;
    auto ks = ks_;
    auto kc = kcnt_;
    if (counted)
      ccReduce3Lanes(
          "peclet::flow::mgmeanr_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int zc, double& a) {
            const long i = (long)x + (long)y * e.x + (long)zc * (long)e.x * e.y;
            if (all || ac(i) > 1e-30f)
              a += (double)xf(i);
          },
          Kokkos::Sum<double, CCMem>(slot(kMsum)));
    else
      ccReduce3Lanes(
          "peclet::flow::mgmeanr_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int zc, double& a, long& c) {
            const long i = (long)x + (long)y * e.x + (long)zc * (long)e.x * e.y;
            if (ac(i) > 1e-30f) {
              a += (double)xf(i);
              c += 1;
            }
          },
          Kokkos::Sum<double, CCMem>(slot(kMsum)), Kokkos::Sum<long, CCMem>(kcnt_));
    const bool copyP = p.extent(0) != 0;
    auto body = KOKKOS_LAMBDA(int x, int y, int zc, double& acc) {
      const long i = (long)x + (long)y * e.x + (long)zc * (long)e.x * e.y;
      const long cnt = counted ? cnt0 : kc();
      const bool fluid = all || ac(i) > 1e-30f;
      const double xd = (double)xf(i);
      const double zi = (fluid && cnt != 0) ? s * (xd - ks(kMsum) / (double)cnt) : s * xd;
      z(i) = zi;
      if (copyP)
        pp(i) = zi;
      if (fluid)
        acc += r(i) * zi;
    };
    if (k >= 0) {
      ccReduce3Lanes("peclet::flow::mgmeans_dot_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
                     body, slot(k));
    } else {
      ccFor3(
          "peclet::flow::mgmeans_f32", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int zc) {
            double unused = 0.0;
            body(x, y, zc, unused);
          });
    }
  }
  // Test hook (tests/kokkos/test_mg_fp32_vcycle.cpp): mark / unmark an overlay solve, as
  // OverlayScope does for the star and ghost-projection drivers.
  void debugSetOverlaySolve(bool on) { overlaySolve_ = on; }
  void setVcyclePrecision(int mode) {
    vprec_ = mode;
    buildFp32Weights();  // takes effect at once (§4.4.6)
  }
  int vcyclePrecision() const { return vprec_; }
  // The precision the last solve's V-cycles used (§4.4.6): true = FP32.
  bool lastVcycleFp32() const { return lastVcycleFp32_; }
  // The decoupled cells of a carrying level, counted two ways: FP32 (all six weights 0, D_i == 0)
  // and FP64 (AC < 1e-30). The face-range check makes the sets equal (§4.4.1 (5)).
  void fp32DecoupledCounts(int L, long& n32, long& n64) {
    const Level& lv = lv_[L];
    const C3 e = lv.ext;
    const int g = lv.g;
    Kokkos::View<const VReal*, CCMem> WX = lv.WX, WY = lv.WY, WZ = lv.WZ;
    FPV AC = lv.AC;
    n32 = 0;
    n64 = 0;
    ccReduce3(
        "peclet::flow::mg_fp32_decoupled", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int lx, int ly, int lz, long& a, long& b) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)lx + (long)ly * sy + (long)lz * sz;
          if (WX(i) == 0.0f && WX(i + sx) == 0.0f && WY(i) == 0.0f && WY(i + sy) == 0.0f &&
              WZ(i) == 0.0f && WZ(i + sz) == 0.0f)
            a += 1;
          if (AC(i) < 1e-30)
            b += 1;
        },
        n32, n64);
  }
#ifndef NDEBUG
  void checkFp32DecoupledSets() {
    for (int L = 0; L < (int)lv_.size(); ++L)
      if (lv_[L].wOk) {
        long n32 = 0, n64 = 0;
        fp32DecoupledCounts(L, n32, n64);
        assert(n32 == n64 && "D: the FP32 decoupled set must equal the FP64 AC < 1e-30 set");
      }
  }
#endif
  // Eligibility of the FP32 V-cycle for the current solve (§4.4.1): nullptr when every condition
  // holds, else the first failed one. `krylov`: the preconditioner is called from
  // solvePCGResident or solveFCG (condition 1).
  const char* fp32VcycleIneligible(bool krylov) const {
    if (!krylov)
      return "the V-cycle is not the preconditioner of the single-rank PCG or FCG driver";
    if (distributed_)
      return "the solve is distributed (multi-rank)";
    if (hasOutflow_)
      return "an outflow face is present";
    if (overlaySolve_)
      return "the solve carries an overlay (star couplings / ghost projection)";
    if (lv_.size() < 2)
      return "the hierarchy has a single level";
    for (std::size_t L = 0; L + 1 < lv_.size(); ++L)
      if (!fusedWrapSmooth(lv_[L]))
        return "a non-bottom level has an odd inner dimension";
    if (!(exactResidual_ || std::is_same_v<MReal, double>))
      return "the outer operator is float (MReal = float without the exact residual)";
    // §4.2 P8: the scheme removes the mean at the level-0 exit only ("fine scope, as today"); the
    // legacy every-level scope (set_pressure_mean_removal('all')) keeps the FP64 V-cycle.
    if (meanRemovalAll_)
      return "the mean-removal scope is 'all' (the FP32 V-cycle removes the mean at level 0 only)";
    for (std::size_t L = 0; L + 1 < lv_.size(); ++L)
      if (!lv_[L].wOk)
        return "a face weight is outside {0} or [1e-30, 1e30] (or the FP32 weights are not built)";
    return nullptr;
  }
  // Marks a solve that carries an overlay (star couplings / the ghost-projection BiCGStab) for its
  // whole duration, every exit path included.
  struct OverlayScope {
    bool& flag;
    OverlayScope(bool& f, bool on) : flag(f) { flag = on; }
    ~OverlayScope() { flag = false; }
  };
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
        cutcellSmoothColorBoxFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                                  FPC(lv.AFZ), lv.ext, og, c0, lo, hi, C3{0, 0, 0}, C3{0, 0, 0});
        lv.dev->exchangeEnd(lv.x);
        cutcellSmoothColorBoxFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                                  FPC(lv.AFZ), lv.ext, og, c0, rlo, rhi, lo, hi);
        cutcellSmoothColorFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                               FPC(lv.AFZ), lv.ext, og, g, c1);
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
          cutcellSmoothColorBoxFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                                    FPC(lv.AFZ), lv.ext, og, color, lo, hi, C3{0, 0, 0},
                                    C3{0, 0, 0});
          lv.dev->exchangeEnd(lv.x);
          applyOutflowGhost(lv, lv.x, g);
          cutcellSmoothColorBoxFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                                    FPC(lv.AFZ), lv.ext, og, color, C3{g, g, g},
                                    C3{lv.ext.x - g, lv.ext.y - g, lv.ext.z - g}, lo, hi);
          continue;
        }
#endif
        if (fusedWrapSmooth(lv)) {  // A3: no fill; the pass reads the periodic wrap directly
          cutcellSmoothColorFaceWrap(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                                     FPC(lv.AFZ), lv.ext, lv.inner, og, lv.g, color);
          continue;
        }
        fill(lv, lv.x);
        applyOutflowGhost(lv, lv.x, lv.g);
        cutcellSmoothColorFace(lv.x, CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX), FPC(lv.AFY),
                               FPC(lv.AFZ), lv.ext, og, lv.g, color);
      }
  }

  // --- when to agglomerate
  // ------------------------------------------------------------------------ A V-cycle only
  // converges at a rate independent of the domain if its COARSEST level is small enough to be
  // solved (essentially) exactly by the few smoother sweeps applied there. A geometric hierarchy
  // cannot always get there: an axis stops coarsening once it turns odd, and under MPI it stops
  // once any rank's block turns odd — so on a fixed per-rank block the coarsest GLOBAL grid grows
  // with the rank count and the bottom is progressively under-solved. That is the mechanism behind
  // weak-scaling curves that decay while communication stays negligible.
  //
  // Measured (single GPU, channel, Lx = 2048 x 64 x 64, everything else held): a smoothed bottom
  // needs 13.5 pressure iterations/step at 4 levels and 6.0 at 6 levels, against 4.4 at full
  // geometric depth. Agglomerating and solving that same bottom exactly gives 4.0 at BOTH 4 and 6
  // levels — depth-independent, and faster in wall-clock than the full-depth hierarchy (69.5 vs
  // 77.1 ms/step) because the extra levels cost more than the coarse solve they replace.
  //
  // NOT the default yet, and the reason is measured: on the cut-cell sphere-packing regression
  // (random_spheres, N=48) switching the bottom to the agglomerated solve makes the OUTER iteration
  // count WORSE (442 -> 622 total, +41 %) at unchanged accuracy, so the assembled coarse operator
  // is evidently not consistent with the V-cycle's on that IBM path. Until that is understood,
  // `auto` is opt-in and the legacy smoothed bottom stays the default. `mode`: 0 = never / plain
  // smoothed bottom (DEFAULT), -1 = auto, 1 = always. `setAgglomerationExtent` moves the
  // threshold; the ideal bottom is a handful of cells per axis.
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
    const int thresh = agglomExtent_ > 0 ? agglomExtent_ : 4;
    // coarsest GLOBAL cell count (the local block does not decide how hard the coarse solve is)
    long gx = gnxF_, gy = gnyF_, gz = gnzF_;
    for (int L = 0; L + 1 < (int)lv_.size(); ++L) {
      gx /= lv_[L].ratio.x;
      gy /= lv_[L].ratio.y;
      gz /= lv_[L].ratio.z;
    }
    return gx > thresh || gy > thresh || gz > thresh;
  }

  // --- The direct bottom solve (doc/vof_step_performance_design.md §5.7, §5.14, §13, §14 H-1) ---
  static constexpr bool kHostMemory =
      std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>;
  // Eligibility (§13.4.6: §5.7 conditions 1-6 -- 6 = B1b's 1-64 fluid components -- and an axis
  // whose planes hold at most kBottomMaxPlane cells): nullptr when the device bottom may run, else
  // the first failed condition. Host-cheap: the components are labelled once per hierarchy
  // (evalBottomComponents, at the first setOpenness after init) and cached -- the rho / eps / drag
  // rescaling of the per-step setOpenness never closes a face, so it cannot change the answer.
  const char* directBottomIneligible() const {
    if (distributed_)
      return "the solve is distributed (multi-rank)";
    if (!removeMean_ || hasOutflow_)
      return "the operator is not the singular one (an outflow face is present)";
    if (!agglomerateBottom())
      return "the bottom is not agglomerated (set_pressure_bottom / set_pressure_bottom_extent)";
    if (lv_.empty() ||
        (long)lv_.back().inner.x * lv_.back().inner.y * lv_.back().inner.z > kBottomMaxCells)
      return "the bottom level has more than 8192 inner cells";
    if (!bottomConnKnown_)
      return "the pressure operator has not been built (no setOpenness since the hierarchy)";
    if (!bottomConnected_)
      return "the bottom level has no fluid cell or more than 64 fluid components (B1b)";
    if (!dirPlanes_.valid())
      return "no axis gives planes of at most 192 cells (the direct factor's cap)";
    return nullptr;
  }
  // The bottom's storage (§13.4.7): single rank (any backend since §14 H-1), a bottom of at most
  // kBottomMaxCells cells with ghost width 1; otherwise none and the engine is ineligible.
  // The FCG vectors, the component labels and their scratch, the plane ordering and the factor.
  void buildBottomStore() {
    facStale_ = true;
    bottomConnKnown_ = false;
    bottomConnected_ = false;
    bottomStore_ = false;
    dirPlanes_ = BottomPlanes{};
    if (distributed_ || lv_.empty())
      return;
    const Level& bt = lv_.back();
    if (bt.g != 1 || (long)bt.inner.x * bt.inner.y * bt.inner.z > kBottomMaxCells)
      return;
    bottomStore_ = true;
    for (CCField* f : {&bottomR_, &bottomP_, &bottomZ_, &bottomZp_, &bottomAp_})
      *f = CCField("mg_bottom_fcg", bt.n);
    bottomInfo_ = Kokkos::View<int*, CCMem>("peclet::flow::mg_bottom_info", 1);
    bottomComp_ = Kokkos::View<int*, CCMem>("peclet::flow::mg_bottom_comp", bt.n);
    bottomLabTmp_ = Kokkos::View<int*, CCMem>("peclet::flow::mg_bottom_labtmp", bt.n);
    bottomCompCnt_ =
        Kokkos::View<long*, CCMem>("peclet::flow::mg_bottom_compcnt", kBottomMaxComponents);
    bottomCompMean_ = CCField("peclet::flow::mg_bottom_compmean", kBottomMaxComponents);
    dirKc_ = Kokkos::View<int*, CCMem>("peclet::flow::mg_bottom_kc", kBottomMaxComponents);
    dirAug_ = Kokkos::View<double*, CCMem>("peclet::flow::mg_bottom_aug", kBottomMaxComponents);
    dirPlanes_ = bottomChoosePlanes(bt.inner, bc_);
    dir_ = BottomDirect<float>::allocate(dirPlanes_, bt.ext);  // PRECISION-EXEMPT: §13 FP32 factor
    dirTeam_ = 0;
    dirFacTeam_ = 0;
  }
  // Condition 6, as B1b replaced it (§5.14): the bottom level's fluid components, labelled on the
  // device by BottomLabelKernel -- eligible with 1..64 of them (the register's per-component
  // projector for the agglomerated bottom; more go to GraphAMG) -- and, along the slow axis, each
  // component's last plane and 1/m_c (§13.4.2). One team launch and one scalar read, at the first
  // operator build of a hierarchy (geometry time; the rho / eps / drag rescaling of the per-step
  // setOpenness never closes or opens a face, so the labels stay valid).
  void evalBottomComponents() {
    bottomConnKnown_ = true;
    bottomConnected_ = false;
    bottomNComp_ = 0;
    if (!bottomStore_)
      return;
    const Level& bt = lv_.back();
    BottomLabelKernel k;
    k.ext = bt.ext;
    k.inner = bt.inner;
    k.AC = bt.AC;
    k.AFX = bt.AFX;
    k.AFY = bt.AFY;
    k.AFZ = bt.AFZ;
    k.comp = bottomComp_;
    k.tmp = bottomLabTmp_;
    k.cnt = bottomCompCnt_;
    k.nc = bottomInfo_;  // info(0) doubles as the component-count landing slot here
    k.slow = dirPlanes_.s;
    k.kc = dirKc_;
    k.aug = dirAug_;
    Kokkos::TeamPolicy<CCExec> probe(CCExec(), 1, 1);
    const int T = std::min(1024, probe.team_size_max(k, Kokkos::ParallelForTag()));
    Kokkos::parallel_for("peclet::flow::mg_bottom_label",
                         Kokkos::TeamPolicy<CCExec>(CCExec(), 1, T), k);
    Kokkos::deep_copy(bottomNComp_, Kokkos::subview(bottomInfo_, 0));
    bottomConnected_ = bottomNComp_ >= 1 && bottomNComp_ <= kBottomMaxComponents;
    if (mgDebugLevel())
      printf("[mg] device bottom: %d fluid component(s) on the %dx%dx%d bottom%s\n", bottomNComp_,
             bt.inner.x, bt.inner.y, bt.inner.z,
             bottomConnected_ ? "" : " -> GraphAMG (more than 64, or none)");
  }
  template <class FR = float>  // PRECISION-EXEMPT: the FP32 direct factor under an FP64 FCG
  BottomKernel<FR> makeBottomKernel(bool precondOnly) {
    ensureScalars();
    Level& bt = lv_.back();
    BottomKernel<FR> k;
    k.lv.ext = bt.ext;
    k.lv.inner = bt.inner;
    k.lv.AC = bt.AC;
    k.lv.AFX = bt.AFX;
    k.lv.AFY = bt.AFY;
    k.lv.AFZ = bt.AFZ;
    k.x = bt.x;
    k.b = bt.rhs;
    k.r = bottomR_;
    k.p = bottomP_;
    k.z = bottomZ_;
    k.zp = bottomZp_;
    k.ap = bottomAp_;
    k.precondOnly = precondOnly ? 1 : 0;
    k.ks = ks_;
    k.flagSlot = kBottomFlag;
    k.info = bottomInfo_;
    k.nc = bottomNComp_;
    k.comp = bottomComp_;
    k.ccnt = bottomCompCnt_;
    k.cmean = bottomCompMean_;
    return k;
  }
  // A non-finite scalar in a device bottom solve (the kernel's device flag) fails the solve. `v`
  // is the flag as read by the PCG packet or by lastSolveFailed(); a set flag is cleared on the
  // device so it is reported once.
  void noteBottomFlag(double v) {
    bottomFlagPending_ = false;
    if (v != 0.0) {
      solveFailed_ = true;
      setSlot(kBottomFlag, 0.0);
    }
  }
  // A launch's team size: min(1024, team_size_max) when team <= 0 -- min(hostTeam,
  // team_size_max) on a host backend (§14 H-1) -- else `team` clamped to team_size_max; with
  // `probe` false a positive `team` (a size this kernel already ran with) is used as is.
  // `scratch` = the team scratch (level 0) the kernel requests, in bytes.
  template <class K>
  static int directTeam(const K& k, int team, bool probe, std::size_t scratch = 0,
                        int hostTeam = kBottomHostTeam) {
    if (team > 0 && !probe)
      return team;
    Kokkos::TeamPolicy<CCExec> pol(CCExec(), 1, 1);
    if (scratch)
      pol.set_scratch_size(0, Kokkos::PerTeam(scratch));
    const int tmax = std::min(1024, pol.team_size_max(k, Kokkos::ParallelForTag()));
    if (team > 0)
      return std::min(team, tmax);
    return kHostMemory ? std::min(hostTeam, tmax) : tmax;
  }
  // The factor launch: one team, T = directTeam(team) (the U3 hook passes T; the factor is bitwise
  // independent of it). `teamAlgorithm` (a test hook, U8) runs the team algorithm on a host
  // backend instead of the host schedule. Returns the team size used.
  template <class FR>
  int launchDirectFactor(const BottomDirect<FR>& D, int team, double tauPiv0, bool probe = true,
                         bool teamAlgorithm = false) {
    const Level& bt = lv_.back();
    BottomFactorKernel<FR, FPC> k;
    k.D = D;
    k.AFX = bt.AFX;
    k.AFY = bt.AFY;
    k.AFZ = bt.AFZ;
    k.comp = bottomComp_;
    k.kc = dirKc_;
    k.aug = dirAug_;
    k.tauPiv0 = tauPiv0;
    k.teamAlgorithm = teamAlgorithm ? 1 : 0;
    const std::size_t scratch = BottomFactorKernel<FR, FPC>::scratchBytes(D.pl.b);
    const int T = directTeam(k, team, probe, scratch, kBottomHostFactorTeam);
    Kokkos::parallel_for(
        "peclet::flow::mg_bottom_factor",
        Kokkos::TeamPolicy<CCExec>(CCExec(), 1, T).set_scratch_size(0, Kokkos::PerTeam(scratch)),
        k);
    return T;
  }
  // The FCG kernel with M = the direct factor `D`: the production solve (precondOnly = false) or,
  // for the tests, z = M(lv.rhs) alone. Returns the team size used.
  template <class FR>
  int launchDirectKernel(const BottomDirect<FR>& D, bool precondOnly, int team, bool noMean,
                         CCField zOut = CCField(), CCField rIn = CCField(), bool probe = true) {
    BottomKernel<FR> k = makeBottomKernel<FR>(precondOnly);
    k.dir = D;
    k.noMeanForTest = noMean ? 1 : 0;
    if (precondOnly) {
      k.z = zOut;
      k.b = rIn;
    }
    const int T = directTeam(k, team, probe);
    Kokkos::parallel_for("peclet::flow::mg_bottom_direct",
                         Kokkos::TeamPolicy<CCExec>(CCExec(), 1, T), k);
    return T;
  }
  // The production direct bottom solve: lv.x = A_b^{-1} lv.rhs (component-mean-free) by the FCG
  // preconditioned by the FP32 factor, refactored first if setOpenness changed the operator (a
  // host flag, not a device read).
  void directBottomSolve() {
    const bool first = dirTeam_ == 0;
    if (facStale_) {
      dirFacTeam_ = launchDirectFactor(dir_, dirFacTeam_, dirPivotTol_, /*probe=*/false);
      facStale_ = false;
      if (mgDebugLevel() >= 2 && dbgSolve_ <= mgDebugSolves()) {
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dir_.stat);
        printf("[mg]     direct bottom factor: ok %d, restarts %d (team %d)\n", h(0), h(1),
               dirFacTeam_);
      }
    }
    dirTeam_ =
        launchDirectKernel(dir_, false, dirTeam_, false, CCField(), CCField(), /*probe=*/false);
    if (first && mgDebugLevel()) {
      printf(
          "[mg] direct bottom: %dx%dx%d, slow axis %d, P = %d, b = %d, border %d; team sizes "
          "factor %d, solve %d\n",
          lv_.back().inner.x, lv_.back().inner.y, lv_.back().inner.z, dirPlanes_.s, dirPlanes_.P,
          dirPlanes_.b, dirPlanes_.border, dirFacTeam_, dirTeam_);
      fflush(stdout);
    }
    bottomFlagPending_ = true;
    if (mgDebugLevel() >= 2 && dbgSolve_ <= mgDebugSolves()) {
      int it = 0;
      Kokkos::deep_copy(it, Kokkos::subview(bottomInfo_, 0));
      printf("[mg]     direct bottom: %d inner iterations (team %d)\n", it, dirTeam_);
    }
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
    auto hC = host(lv.AC), hAFX = host(lv.AFX), hAFY = host(lv.AFY), hAFZ = host(lv.AFZ);
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
          // A2 face form: AW = AFX(p), AE = AFX(p+1), ... (the band values themselves).
          const double bc[6] = {(double)hAFX(p), (double)hAFX(p + 1),
                                (double)hAFY(p), (double)hAFY(p + ex),
                                (double)hAFZ(p), (double)hAFZ(p + (long)ex * ey)};
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
      gatherv(lgid, ggid, lv.comm);
      gatherv(ldiag, gdiag, lv.comm);
      gatherv(lrow, grow, lv.comm);
      gatherv(lcol, gcol, lv.comm);
      gatherv(lval, gval, lv.comm);
      gatherv(lsolid, gsolid, lv.comm);
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
      printf(
          "[agmg-build] n=%d solid=%ld fluid=%ld  fluid|diag| min=%.3e max=%.3e  "
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
      gatherv(amgGlobalOfLocal_, ggid, lv.comm);
      gatherv(lb, gb, lv.comm);
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
      residualCutcellFace(lv.res, CCConst(lv.x), CCConst(lv.rhs), FPC(lv.AC), FPC(lv.AFX),
                          FPC(lv.AFY), FPC(lv.AFZ), lv.ext, lv.g);
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
      // CG breakdown guard (core's GraphAMG pcg has the same one): p^T A p = 0 means the search
      // direction lies in the null space, i.e. the projected rhs has no range component left and
      // x is already the answer. It happens when the bottom rhs is round-off on a constant: the
      // projection's rounded mean differs from the value by an ulp and leaves an exactly CONSTANT
      // residual (8 x -0x1.ffffffffffffcp-103 -> 8 x 0x1p-154 on the uniform periodic porous bed),
      // the AMG maps it to z = 0, and rz / pAp = 0/0 poisoned x with NaN, which the outer PCG then
      // reported as a non-finite preconditioner and a CAPPED solve.
      const double pAp = dot(p, Ap);
      if (!(pAp > 0.0))
        break;
      const double a = rz / pAp;
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
        printf(
            "[agmg] call=%ld iters=%d relres=%.2e  |b|sol=%.3e |b|fl=%.3e  "
            "mean(b) fl=%.3e all=%.3e  compat=%.2e  |x|sol=%.3e |x|fl=%.3e%s\n",
            agmgCalls_, it, r0 > 0 ? rn / r0 : 0.0, bSolidMax, bFluidMax, bFluidMean, bAllMean,
            bCompMax, xSolidMax, xFluidMax, it >= 100 ? "  CAP" : "");
        fflush(stdout);
      }
    }
  }
#ifdef PECLET_FLOW_MPI
  template <class T>
  void gatherv(const std::vector<T>& local, std::vector<T>& all, MPI_Comm c) {
    // REDUNDANT agglomeration: every rank receives the full concatenation (rank order, so the
    // assembled coarse problem is bit-identical on all ranks and each solves it locally with no
    // rank-0 bottleneck and no result broadcast).
    int size = 1;
    MPI_Comm_size(c, &size);
    const int lbytes = (int)(local.size() * sizeof(T));
    std::vector<int> bc(size), bd(size, 0);
    MPI_Allgather(&lbytes, 1, MPI_INT, bc.data(), 1, MPI_INT, c);
    int tot = 0;
    for (int r = 0; r < size; ++r) {
      bd[r] = tot;
      tot += bc[r];
    }
    all.resize((std::size_t)tot / sizeof(T));
    MPI_Allgatherv(local.data(), lbytes, MPI_BYTE, all.data(), bc.data(), bd.data(), MPI_BYTE, c);
  }
#endif

  // Level-0 matvec with the halo overlapped: post the exchange, apply the interior rows while the
  // messages are in flight, land the ghosts (+ outflow ghost), apply the boundary shell. Reads v,
  // writes y (no aliasing) => bit-identical to the blocking fill-then-apply. Single-rank: the
  // blocking path.
  void matvecOverlap(Level& l0, CCField y, CCField v) {
    const bool ex = exactResidual_;  // P1: exact double flux-form apply instead of the bands
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const C3 lo{G + 1, G + 1, G + 1};
      const C3 hi{l0.ext.x - G - 1, l0.ext.y - G - 1, l0.ext.z - G - 1};
      l0.dev->exchangeBegin(v);
      if (ex)
        applyCutcellOpExactBox(y, CCConst(v), CCConst(l0.ox), CCConst(l0.oy), CCConst(l0.oz),
                               l0.ext, lo, hi, C3{0, 0, 0}, C3{0, 0, 0}, gfx_, gfy_, gfz_);
      else
        applyCutcellOpBoxFace(y, CCConst(v), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ),
                              l0.ext, lo, hi, C3{0, 0, 0}, C3{0, 0, 0});
      l0.dev->exchangeEnd(v);
      applyOutflowGhost(l0, v);
      if (ex)
        applyCutcellOpExactBox(y, CCConst(v), CCConst(l0.ox), CCConst(l0.oy), CCConst(l0.oz),
                               l0.ext, C3{G, G, G}, C3{l0.ext.x - G, l0.ext.y - G, l0.ext.z - G},
                               lo, hi, gfx_, gfy_, gfz_);
      else
        applyCutcellOpBoxFace(y, CCConst(v), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ),
                              l0.ext, C3{G, G, G}, C3{l0.ext.x - G, l0.ext.y - G, l0.ext.z - G}, lo,
                              hi);
      return;
    }
#endif
    if (fusedWrapReads()) {  // A3: the matvec reads the periodic wrap directly, no fill
      if (ex)
        applyCutcellOpExactWrap(y, CCConst(v), CCConst(l0.ox), CCConst(l0.oy), CCConst(l0.oz),
                                l0.ext, l0.inner, G, gfx_, gfy_, gfz_);
      else
        applyCutcellOpFaceWrap(y, CCConst(v), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ),
                               l0.ext, l0.inner, G);
      return;
    }
    fill(l0, v);
    applyOutflowGhost(l0, v);
    if (ex)
      applyCutcellOpExact(y, CCConst(v), CCConst(l0.ox), CCConst(l0.oy), CCConst(l0.oz), l0.ext, G,
                          gfx_, gfy_, gfz_);
    else
      applyCutcellOpFace(y, CCConst(v), FPC(l0.AC), FPC(l0.AFX), FPC(l0.AFY), FPC(l0.AFZ), l0.ext,
                         G);
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
    if (lv.inner.x < lv.g || lv.inner.y < lv.g || lv.inner.z < lv.g) {
      // a ghost band wider than the block: the sequential passes copy ghost from ghost, which the
      // one-shot wrap does not reproduce -- keep them
      fillAxis(lv, f, 0);
      fillAxis(lv, f, 1);
      fillAxis(lv, f, 2);
      return;
    }
    fillWrap(lv, f);
  }
  // The single-rank periodic fill in ONE launch over the ghost shell: every ghost cell takes the
  // value of the inner cell its three coordinates wrap to. That is exactly what the three
  // sequential `fillAxis` passes (x, then y, then z, each over the full transverse extent, ghosts
  // included) leave behind -- a ghost already written by an earlier pass is overwritten by the
  // later pass with the wrapped-inner value, and inner cells are only ever read -- so the two are
  // bit-identical (pure copies). The V-cycle calls this before every smoother colour, so the three
  // launches it replaces were two thirds of the solve's kernel launches, and on the coarse levels
  // the launch, not the copy, is the cost.
  template <class V>  // CCField; D: also the FP32 coarse iterate (VField), the same copies
  void fillWrap(Level& lv, V f) {
    CCExec space;
    const C3 e = lv.ext;
    const int G = lv.g;  // shadows the class constant: this level's ghost width
    const int nx = lv.inner.x, ny = lv.inner.y, nz = lv.inner.z;
    const long exy = (long)e.x * e.y;
    // the shell as three slabs: z-ghost planes (all x, y), then y-ghost rows of the inner z range
    // (all x), then x-ghost cells of the inner (y, z) range
    const long nZ = 2L * G * exy, nY = (long)nz * 2 * G * e.x, nX = (long)nz * ny * 2 * G;
    if constexpr (std::is_same_v<typename CCExec::memory_space, Kokkos::HostSpace>) {
      // A3 (§4.3, §5.4): on a host backend the same three slabs go ROW by row -- the index math
      // (one division) per row, a contiguous copy along x inside it -- serially below the cutoff.
      // The one-thread-per-cell form below paid a 64-bit div/mod per ghost cell (85 us per
      // launch at 24 threads). Every ghost cell is still written once, from the inner cell its
      // three coordinates wrap to, so the result is identical. No simd mark: a row's source and
      // destination can share the row (the x-ghost slab).
      const long rZ = 2L * G * e.y, rY = (long)nz * 2 * G, rX = (long)nz * ny;
      V ff = f;
      auto copyRow = [=](long dst, long src) {  // one full x row, wrapped along x
        for (int x = 0; x < G; ++x)
          ff(dst + x) = ff(src + x + nx);
        for (int x = G; x < G + nx; ++x)
          ff(dst + x) = ff(src + x);
        for (int x = G + nx; x < e.x; ++x)
          ff(dst + x) = ff(src + x - nx);
      };
      auto row = [=](long t) {
        if (t < rZ) {  // z-ghost plane row (y, zg)
          const int l = (int)(t / e.y), y = (int)(t - (long)l * e.y);
          const int z = l < G ? l : nz + l;
          const int sy = y < G ? y + ny : (y >= G + ny ? y - ny : y);
          const int sz = z < G ? z + nz : z - nz;
          copyRow((long)y * e.x + (long)z * exy, (long)sy * e.x + (long)sz * exy);
        } else if (t < rZ + rY) {  // y-ghost row (yg, z) of the inner z range
          const long t2 = t - rZ;
          const int zi = (int)(t2 / (2 * G)), l = (int)(t2 - (long)zi * 2 * G);
          const int y = l < G ? l : ny + l, z = G + zi;
          const int sy = y < G ? y + ny : y - ny;
          copyRow((long)y * e.x + (long)z * exy, (long)sy * e.x + (long)z * exy);
        } else {  // the x-ghost cells of inner row (y, z)
          const long t3 = t - rZ - rY;
          const int zi = (int)(t3 / ny), yi = (int)(t3 - (long)zi * ny);
          const long base = (long)(G + yi) * e.x + (long)(G + zi) * exy;
          for (int x = 0; x < G; ++x)
            ff(base + x) = ff(base + x + nx);
          for (int x = G + nx; x < e.x; ++x)
            ff(base + x) = ff(base + x - nx);
        }
      };
      if (hostRunSerial(nZ + nY + nX)) {
        for (long t = 0; t < rZ + rY + rX; ++t)
          row(t);
        return;
      }
      Kokkos::parallel_for("peclet::flow::mg_pfill3",
                           Kokkos::RangePolicy<CCExec>(space, 0, rZ + rY + rX), row);
      return;
    }
    V ff = f;
    Kokkos::parallel_for(
        "peclet::flow::mg_pfill3", Kokkos::RangePolicy<CCExec>(space, 0, nZ + nY + nX),
        KOKKOS_LAMBDA(long t) {
          int x, y, z;
          if (t < nZ) {
            const int l = (int)(t / exy);
            const long r = t - (long)l * exy;
            x = (int)(r % e.x);
            y = (int)(r / e.x);
            z = l < G ? l : nz + l;
          } else if (t < nZ + nY) {
            const long t2 = t - nZ, row = 2L * G * e.x;
            const long zi = t2 / row, r = t2 - zi * row;
            const int l = (int)(r / e.x);
            x = (int)(r % e.x);
            y = l < G ? l : ny + l;
            z = G + (int)zi;
          } else {
            const long t3 = t - nZ - nY, row = (long)ny * 2 * G;
            const long zi = t3 / row, r = t3 - zi * row;
            const int l = (int)(r % (2 * G));
            x = l < G ? l : nx + l;
            y = G + (int)(r / (2 * G));
            z = G + (int)zi;
          }
          const int sx = x < G ? x + nx : (x >= G + nx ? x - nx : x);
          const int sy = y < G ? y + ny : (y >= G + ny ? y - ny : y);
          const int sz = z < G ? z + nz : (z >= G + nz ? z - nz : z);
          ff((long)x + (long)y * e.x + (long)z * exy) =
              ff((long)sx + (long)sy * e.x + (long)sz * exy);
        });
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
        "peclet::flow::mg_pfill", MDRange2<CCExec>(space, {0, 0}, {dims[b], dims[c]}),
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
        "peclet::flow::gp_stage_g2", MDRange3<CCExec>(space, {0, 0, 0}, {nn.x, nn.y, nn.z}),
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
        "peclet::flow::gp_unstage_g2", MDRange3<CCExec>(space, {0, 0, 0}, {e1.x, e1.y, e1.z}),
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

  // A zero-filled level-0 work vector that persists across solves: the Chebyshev driver and its
  // eigenvalue estimate used to allocate (and free) eight of these per step -- a
  // cudaMalloc/cudaFree pair each, and cudaFree synchronises the device. Same contents as a fresh
  // View (zeros), so the solves are unchanged bit for bit.
  CCField workVector(int slot, std::size_t n, const char* label) {
    CCField& v = work_[slot];
    if (v.extent(0) != n)
      v = CCField(label, n);
    else
      Kokkos::deep_copy(CCExec(), v, 0.0);
    return v;
  }

  // How estimateEigenvalues starts its two power iterations (D3, doc/vof_step_performance_design.md
  // §5.13). `Cold`: both from `seed` (the right-hand side), nothing kept -- the pre-D3 estimate,
  // and still the balanced-force pre-projection's. `ColdKeep`: the same arithmetic, and the last
  // v_max / v_min iterates are kept. `Warm`: each iteration starts from its kept iterate (masked,
  // mean-removed and normalised exactly as the rhs seed is) and the kept iterates are replaced by
  // the new last ones; `seed` is not read. Warm requires eigenWarmReady().
  enum class EigenStart { Cold, ColdKeep, Warm };
  // True when a ColdKeep or Warm estimate has run on the current hierarchy (init/initMpi drop it).
  bool eigenWarmReady() const {
    return eigWarm_ && !lv_.empty() && eigVmax_.extent(0) == lv_[0].n &&
           eigVmin_.extent(0) == lv_[0].n;
  }
  // The kept iterates are cross-step state: the Solver's `redistribute` carries them across a
  // repartition (inner cells migrated like a registry field) and hands them back here.
  CCField eigenWarmMax() const { return eigVmax_; }
  CCField eigenWarmMin() const { return eigVmin_; }
  void setEigenWarm(CCField vmax, CCField vmin) {
    eigVmax_ = vmax;
    eigVmin_ = vmin;
    eigWarm_ = true;
  }

  // Estimate the spectral bounds [lmin,lmax] of M^{-1}A (M^{-1} = one symmetric V-cycle) by power
  // iteration (direct for the max + a shifted iteration for the min), seeded by `seed` (or, with
  // EigenStart::Warm, by the iterates a previous estimate kept). Communication-heavy, so the CUDA
  // driver runs it once on step 1 and reuses the bounds. Port of estimate_eigenvalues.
  void estimateEigenvalues(CCConst seed, double& lmin, double& lmax, int iters, int pre, int post,
                           int bottom, EigenStart start = EigenStart::Cold) {
    solveFailed_ = false;  // ISSUES sweep item 6: per-solve breakdown flag
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    Level& l0 = lv_[0];
    const std::size_t n = l0.n;
    CCField v = workVector(0, n, "ev_v"), w = workVector(1, n, "ev_w"),
            z = workVector(2, n, "ev_z"), srhs = workVector(3, n, "ev_srhs");
    const bool warm = start == EigenStart::Warm, keep = start != EigenStart::Cold;
    if (!warm)
      Kokkos::deep_copy(CCExec(), srhs, seed);
    auto matvec = [&](CCField y, CCField x) { matvecOverlap(l0, y, x); };
    auto applyT = [&](CCField out,
                      CCField in) {  // out = M^{-1} A in, projected onto the fluid range
      matvec(w, in);
      precondVcycle(out, w);  // A5
      removeMean(l0, out);
      maskSolid(l0, out);
    };
    auto normalize = [&](CCField x) {
      double nr = std::sqrt(dot(l0, x, x));
      if (nr > 0)
        scale(x, 1.0 / nr);
      return nr;
    };
    auto seedf = [&](CCField x, CCField src) {
      Kokkos::deep_copy(CCExec(), x, src);
      removeMean(l0, x);
      maskSolid(l0, x);
      return normalize(x);
    };
    auto keepf = [&](CCField& dst, CCField x, const char* label) {
      if (dst.extent(0) != n)
        dst = CCField(label, n);
      Kokkos::deep_copy(CCExec(), dst, x);
    };
    const double nrMax = seedf(v, warm ? eigVmax_ : srhs);
    lmax = 1.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      lmax = dot(l0, v, z);
      Kokkos::deep_copy(CCExec(), v, z);
      normalize(v);
    }
    if (keep)
      keepf(eigVmax_, v, "ev_vmax");
    const double nrMin = seedf(v, warm ? eigVmin_ : srhs);
    double mu = 0.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      lin(z, lmax, v, -1.0, z);  // z = lmax*v - T v
      mu = dot(l0, v, z);
      Kokkos::deep_copy(CCExec(), v, z);
      normalize(v);
    }
    if (keep) {
      keepf(eigVmin_, v, "ev_vmin");
      // Only a non-degenerate estimate may seed a warm one: a zero seed (a zero right-hand side,
      // e.g. a fluid at rest) leaves zero iterates and lmax = 0, from which a warm estimate would
      // return zero bounds -- and a Chebyshev solve on them NaN, which maxabs cannot see.
      eigWarm_ =
          nrMax > 0.0 && nrMin > 0.0 && std::isfinite(lmax) && lmax > 0.0 && std::isfinite(mu);
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
  // `guard3` (D3, §5.13, for a solve on warm-started bounds): if the residual after 3 iterations
  // is not <= r0 the solve stops there and chebyshevGuardTripped() reports it, so the caller can
  // re-estimate cold and redo the solve. Off, the loop is the pre-D3 loop.
  int solveChebyshev(CCField b, CCField x, int maxit, double rtol, int pre, int post, int bottom,
                     double a, double bnd, bool guard3 = false) {
    solveFailed_ = false;  // ISSUES sweep item 6: per-solve breakdown flag
    chebGuardTripped_ = false;
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
    CCField r = workVector(4, n, "cb_r"), z = workVector(5, n, "cb_z"),
            d = workVector(6, n, "cb_d"), w = workVector(7, n, "cb_w");
    auto matvec = [&](CCField y, CCField v) { matvecOverlap(l0, y, v); };
    auto precond = [&](CCField zz, CCField rr) { precondVcycle(zz, rr); };  // A5
    const double theta = 0.5 * (bnd + a), delta = 0.5 * (bnd - a), sigma1 = theta / delta;
    double rho = 1.0 / sigma1;
    matvec(w, x);  // r = b - A x
    Kokkos::deep_copy(CCExec(), r, b);
    axpy(r, -1.0, w);
    removeMean(l0, r);
    const double r0 = maxabs(l0, r);
    // The stop reference: r0 (the default), or the caller's FULL right-hand-side norm when a
    // stop reference is set (setStopReference; only the balanced-force projection sets it).
    // With it unset rref IS r0, so `rtol * rref` is bit-for-bit the old test.
    const double rref = stopRef_ > 0.0 ? stopRef_ : r0;
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
        const double rn = maxabs(l0, r);
        if (rn < rtol * rref)
          break;
        if (guard3 && i == 3 && !(rn <= r0)) {  // growing after 3 iterations: bounds gone stale
          chebGuardTripped_ = true;
          break;
        }
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
  // --- stop reference for a solve relative to the FULL right-hand side (the balanced-force
  // projection, flow doc/collocated_varrho_forces.md §4.6 / WO-P5). ref <= 0 restores the
  // default (stop relative to the initial residual). Callers set it around ONE solve only.
  void setStopReference(double ref) { stopRef_ = ref; }
  // H-2c (§8): the caller guarantees the initial iterate of the next single-rank PCG solves is
  // +0 everywhere (projectSolve's cold path), so the solve skips A x0. Set around ONE solve.
  void setZeroInitialGuess(bool on) { zeroGuess_ = on; }
  // Did the last solveChebyshev(..., guard3 = true) stop on its 3-iteration growth guard?
  bool chebyshevGuardTripped() const { return chebGuardTripped_; }
  // max|b| over the fluid cells after the null-space (mean) projection -- the drivers' norm.
  double rhsNorm(CCField b, CCField scratch) {
    Level& l0 = lv_[0];
    Kokkos::deep_copy(scratch, b);
    removeMean(l0, scratch);
    return maxabs(l0, scratch);
  }
  // max|b - A x| in the same norm, computed exactly as the drivers compute their r0.
  double residualNorm(CCField b, CCField x, CCField r, CCField Ap) {
    Level& l0 = lv_[0];
    matvecOverlap(l0, Ap, x);
    Kokkos::deep_copy(r, b);
    axpy(r, -1.0, Ap);
    removeMean(l0, r);
    return maxabs(l0, r);
  }
  // reductions / mean removal over inner FLUID cells (AC>tiny) of a level.
  double dot(Level& lv, CCField a, CCField b) {
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a, bb = b;
    FPV ac = lv.AC;
    double s = 0;
    if (hostAllFluid(lv))  // H-1: no AC read
      ccReduce3Lanes(
          "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            acc += aa(i) * bb(i);
          },
          s);
    else
      ccReduce3Lanes(
          "mgdot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            if (ac(i) > 1e-30f)
              acc += aa(i) * bb(i);
          },
          s);
#ifdef PECLET_FLOW_MPI
    return allreduce(s, MPI_SUM_, lv.comm);
#else
    return allreduce(s, MPI_SUM_);
#endif
  }
  double maxabs(Level& lv, CCField a) {
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField aa = a;
    FPV ac = lv.AC;
    double m = 0;
    if (hostAllFluid(lv))  // H-1: no AC read
      ccReduce3(
          "mgmax", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            const double v = Kokkos::fabs(aa(i));
            if (v > acc)
              acc = v;
          },
          Kokkos::Max<double>(m));
    else
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
#ifdef PECLET_FLOW_MPI
    return allreduce(m, MPI_MAX_, lv.comm);
#else
    return allreduce(m, MPI_MAX_);
#endif
  }
  // H-1 (§8, host single rank): removeMean with the level's precomputed fluid count -- the sum
  // reduces in the same 4-lane pencil order (C0) as the {sum, count} pair, and an all-fluid level
  // reads no AC. Same cells, same order, same mean: bitwise to the single-rank path below.
  void removeMeanHostCounted(Level& lv, CCField f, bool stopGuard) {
    meanSumHostCounted(lv, f);
    meanSubtractHostCounted(lv, f, stopGuard);
  }
  // The sum half of removeMeanHostCounted: the fluid sum of f into kMsum (4-lane order).
  void meanSumHostCounted(Level& lv, CCField f) {
    C3 e = lv.ext;
    const int g = lv.g;
    CCField ff = f;
    FPV ac = lv.AC;
    const bool all = lv.allFluid;
    if (all)
      ccReduce3Lanes(
          "mgmeanr", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& s) {
            s += ff((long)x + (long)y * e.x + (long)z * (long)e.x * e.y);
          },
          Kokkos::Sum<double, CCMem>(slot(kMsum)));
    else
      ccReduce3Lanes(
          "mgmeanr", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z, double& s) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            if (ac(i) > 1e-30f)
              s += ff(i);
          },
          Kokkos::Sum<double, CCMem>(slot(kMsum)));
  }
  // The subtract half of removeMeanHostCounted: f -= kMsum / nFluid on the fluid cells (skipped
  // with `stopGuard` once the PCG stop flag is set).
  void meanSubtractHostCounted(Level& lv, CCField f, bool stopGuard) {
    C3 e = lv.ext;
    const int g = lv.g;
    CCField ff = f;
    FPV ac = lv.AC;
    const bool all = lv.allFluid;
    auto ks = ks_;
    const long cnt = lv.nFluid;
    const bool guard = stopGuard;
    ccFor3(
        "mgmeans", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g}, KOKKOS_LAMBDA(int x, int y, int z) {
          if (cnt == 0 || (guard && ks(kStop) != 0.0))
            return;
          const double mean = ks(kMsum) / (double)cnt;
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (all)
            ff(i) -= mean;
          else
            meanSubtractCell(ff, ac, i, mean);
        });
  }
  // C4 (§6 amendment 2): the level-0 V-cycle exit removeMean of z = lv.x (removeMeanHostCounted:
  // the sum, then the subtract) with r^T z (r = lv.rhs) accumulated in the subtract pass, with
  // dotTo's mask, operand order and 4-lane order -- bitwise to removeMean(lv, z) + dotTo(lv, r,
  // z, k). With `p` allocated also p = z (copyDotTo's H-2b fusion, under its own condition).
  void removeMeanDotTo(Level& lv, int k, CCField p) {
    meanSumHostCounted(lv, lv.x);
    const C3 e = lv.ext;
    const int g = lv.g;
    CCField zz = lv.x, rr = lv.rhs, pp = p;
    FPV ac = lv.AC;
    const bool all = lv.allFluid, copyP = p.extent(0) != 0;
    const long cnt = lv.nFluid;
    auto ks = ks_;
    ccReduce3Lanes(
        "mgmeans_dot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
        KOKKOS_LAMBDA(int x, int y, int zc, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)zc * (long)e.x * e.y;
          if (cnt != 0) {
            const double mean = ks(kMsum) / (double)cnt;
            if (all)
              zz(i) -= mean;
            else
              meanSubtractCell(zz, ac, i, mean);
          }
          if (copyP)
            pp(i) = zz(i);
          if (all || ac(i) > 1e-30f)
            acc += rr(i) * zz(i);
        },
        slot(k));
  }
  // Single rank (A6, §5.6): the {sum, count} reduce into device slots with the same policy and
  // functor, and the subtract kernel forms mean = sum / count itself -- no host read. With
  // `stopGuard` the subtract is skipped once the PCG stop flag is set (solvePCGResident keeps r
  // exactly as the host-scalar loop leaves it at a break). Distributed: the Allreduce path below.
  void removeMean(Level& lv, CCField f, bool stopGuard = false) {
    if (!removeMean_)
      return;  // non-singular operator (Dirichlet outflow present) -> no null-space projection
    CCExec space;
    C3 e = lv.ext;
    const int g = lv.g;
    CCField ff = f;
    FPV ac = lv.AC;
    // the {sum, count} body; ccReduce3Lanes runs it in the 4-lane pencil order on a host backend
    // (WO-C0, doc/vof_projection_cost_design.md §6) and as the MDRange reduction it always was on a
    // device
    auto body = KOKKOS_LAMBDA(int x, int y, int z, double& s, long& k) {
      const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
      if (ac(i) > 1e-30f) {
        s += ff(i);
        k += 1;
      }
    };
    if (!distributed_) {
      ensureScalars();
      if (kHostMemory && lv.nFluid >= 0) {  // H-1: the precomputed count, no count reduction
        removeMeanHostCounted(lv, f, stopGuard);
        return;
      }
      ccReduce3Lanes("mgmeanr", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g}, body,
                     Kokkos::Sum<double, CCMem>(slot(kMsum)), Kokkos::Sum<long, CCMem>(kcnt_));
      auto ks = ks_;
      auto kc = kcnt_;
      const bool guard = stopGuard;
      ccFor3(
          "mgmeans", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long cnt = kc();
            if (cnt == 0 || (guard && ks(kStop) != 0.0))
              return;
            const double mean = ks(kMsum) / (double)cnt;
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            meanSubtractCell(ff, ac, i, mean);
          });
      return;
    }
    double sum = 0;
    long cnt = 0;
    ccReduce3Lanes("mgmeanr", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g}, body, sum, cnt);
    double dcnt = (double)cnt;
    allreduceSum2(sum, dcnt
#ifdef PECLET_FLOW_MPI
                  ,
                  lv.comm
#endif
    );  // ONE latency hit for the {sum, count} pair (was two)
    cnt = (long)dcnt;
    if (cnt == 0)
      return;
    const double mean = sum / (double)cnt;
    ccFor3(
        "mgmeans", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g}, KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          meanSubtractCell(ff, ac, i, mean);
        });
  }

  // Mean-removal scope. "all" (legacy): project the nullspace out at every V-cycle level,
  // after every matvec and on every residual update — ~10 extra MPI_Allreduce latency hits per
  // Krylov iteration whose only role is FP hygiene. "fine" (DEFAULT) keeps the removals that carry
  // the algorithm (the rhs/residual projections + the fine-level V-cycle exit + the final iterate)
  // and drops the interior-level ones: A maps mean-free vectors to mean-free vectors, so the Krylov
  // space never sees the dropped components (they lie in the nullspace and are removed from the
  // final x). Validated by iteration-count parity; not bit-identical to "all".
  void setMeanRemovalScope(bool all) { meanRemovalAll_ = all; }

  // Accumulated wall time / call count of the global reductions (every dot product, residual max
  // and mean-removal funnels through allreduce()). This is THE latency-bound term of the
  // distributed pressure solve; the solver resets it per step and exposes it to Python.
  double allreduceSeconds() const { return allreduceTime_; }
  long allreduceCount() const { return allreduceCount_; }
  // ISSUES sweep item 6: did the LAST solve driver give up on a non-finite recurrence scalar?
  // A failing solve returns the ITERATION CAP (so a rule-3b "no capped solve" check sees it) and
  // sets this; it used to print to stdout, zero the correction and report 0 iterations.
  // The device bottom flags a non-finite scalar on the device; a driver whose host reads do not
  // carry it (all but the single-rank PCG packet) has it folded in here, by one scalar read.
  bool lastSolveFailed() {
    if (bottomFlagPending_) {
      double v = 0.0;
      Kokkos::deep_copy(v, slot(kBottomFlag));
      noteBottomFlag(v);
    }
    return solveFailed_;
  }
  void resetAllreduceCounters() {
    allreduceTime_ = 0.0;
    allreduceCount_ = 0;
  }

 private:
  enum AllOp { kSum, kMax };
#ifdef PECLET_FLOW_MPI
  using MPI_Comm_or_void = MPI_Comm;
  static MPI_Comm nullptr_comm() { return MPI_COMM_NULL; }
#else
  using MPI_Comm_or_void = void*;
  static void* nullptr_comm() { return nullptr; }
#endif
  // Global reduction over ranks (no-op single-rank / non-distributed -> byte-identical to the local
  // reduce).
  double allreduce(double v, AllOp op, MPI_Comm_or_void c = nullptr_comm()) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const auto t0 = std::chrono::steady_clock::now();
      double g = 0;
      MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, op == kSum ? MPI_SUM : MPI_MAX,
                    c == nullptr_comm() ? comm_ : c);
      allreduceTime_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++allreduceCount_;
      return g;
    }
#endif
    (void)op;
    return v;
  }
  // One MPI_Allreduce of a {sum, count} pair — elementwise MPI_SUM on a 2-vector is bit-identical
  // to two separate allreduces, at half the latency hits.
  void allreduceSum2(double& a, double& b, MPI_Comm_or_void c = nullptr_comm()) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const auto t0 = std::chrono::steady_clock::now();
      double v[2] = {a, b}, g[2] = {0.0, 0.0};
      MPI_Allreduce(v, g, 2, MPI_DOUBLE, MPI_SUM, c == nullptr_comm() ? comm_ : c);
      allreduceTime_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++allreduceCount_;
      a = g[0];
      b = g[1];
    }
#endif
  }
  static constexpr AllOp MPI_SUM_ = kSum, MPI_MAX_ = kMax;

  std::vector<Level> lv_;
  CCField work_[8];  // workVector's slots (0-3 the eigenvalue estimate, 4-7 the Chebyshev driver)
  // D3 (§5.13): the last v_max / v_min power iterates of a ColdKeep/Warm estimate, persistent
  // across solves (never zeroed), valid while eigWarm_ (dropped by init/initMpi).
  CCField eigVmax_, eigVmin_;
  bool eigWarm_ = false;
  bool chebGuardTripped_ = false;  // see chebyshevGuardTripped()
  // The physical metric (setMetric), doc/anisotropic_metric.md §5: h_a' = h_a/hRef and the flag
  // that engages the aspect-ratio coarsening rule.  (1,1,1)/false is the isotropic lattice, on
  // which mgChooseRatio reproduces today's level table exactly.
  double hp_[3] = {1.0, 1.0, 1.0};
  bool aniso_ = false;
  int pre_ = 2, post_ = 2, bottom_ = 4;
  int bc_[6] = {0, 0, 0, 0, 0, 0};
  bool hasBC_ = false, removeMean_ = true, hasOutflow_ = false;
  // WO-R2 item 1. OFF by default and then every path is byte-identical to before it existed: the
  // outflow face keeps the literal 1.0 the raw-openness operator wants. The varRho pressure build
  // (IbmSolver::project) turns it on for its own setOpenness call, because the field it hands over
  // is the COEFFICIENT open_f*rho0/rho_f and the 1.0 would overwrite the boundary rows of it.
  bool outflowCoeff_ = false;
  CCField bcPlane_[3];  // level-0 high-side outflow coefficient plane, per axis (saveOutflowPlanes)
  // Default "fine" (measured winner of the at-scale ablation, Snellius H100 8+16 GPUs: 5.5%
  // faster than "all" with identical iteration counts; single-rank the reductions are free either
  // way). setMeanRemovalScope(true) restores the legacy every-level scope.
  bool meanRemovalAll_ = false;
  bool distributed_ = false;
  bool overlaySolve_ = false;              // A3: the current solve has an overlay (OverlayScope)
  int dbgBrkWhich_ = 0, dbgBrkIter_ = -1;  // setDebugBreakdown (test hook)
  int dbgSolve_ = 0;  // solve counter for the env-gated convergence trace (mgDebugLevel() >= 2)
  // ISSUES sweep item 6: did the LAST Krylov solve give up on a non-finite recurrence
  // scalar (a preconditioner or operator that produced NaN/Inf)? Reset at the head of
  // every solve. See `lastSolveFailed()`.
  bool solveFailed_ = false;
  // The device bottom (§5.7, §5.14, §13): the FCG vectors, the kernel's info slot, the component
  // labels (condition 6, cached per hierarchy), and whether a device solve ran since its flag was
  // last read.
  CCField bottomR_, bottomP_, bottomZ_, bottomZp_, bottomAp_;
  Kokkos::View<int*, CCMem> bottomInfo_;  // [0] = inner iterations of the last device solve
  bool bottomConnKnown_ = false, bottomConnected_ = false;  // condition 6 (B1b), per hierarchy
  int bottomNComp_ = 0;                                     // the bottom's fluid components
  Kokkos::View<int*, CCMem> bottomComp_, bottomLabTmp_;  // component ids (-1 solid), label scratch
  Kokkos::View<long*, CCMem> bottomCompCnt_;             // cells per component
  CCField bottomCompMean_;                               // the kernel's per-component means
  bool bottomFlagPending_ = false;  // a device solve ran since the device flag was last read
  int bottomSolver_ = 0;            // kBottomAuto / kBottomDirect / kBottomAlgebraic
  // §13, the direct preconditioner: the bottom storage exists (single rank, <= 8192 cells, device
  // or forced by a hook), the plane ordering, the FP32 factor, the per-component augmentation, the
  // factor's staleness (set by setOpenness), the two kernels' team sizes, the first attempt's
  // pivot floor (kBottomPivotTol; the U5 hook raises it).
  bool bottomStore_ = false;
  BottomPlanes dirPlanes_;
  BottomDirect<float> dir_;  // PRECISION-EXEMPT: the FP32 direct factor under an FP64 FCG (§13)
  Kokkos::View<int*, CCMem> dirKc_;
  Kokkos::View<double*, CCMem> dirAug_;
  bool facStale_ = true;
  int dirTeam_ = 0, dirFacTeam_ = 0;
  double dirPivotTol_ = kBottomPivotTol;
  double stopRef_ = -1.0;       // see setStopReference
  bool zeroGuess_ = false;      // see setZeroInitialGuess
  std::vector<double> lvTime_;  // per-level V-cycle wall time (mgDebugLevel() >= 3)
  int lvCycles_ = 0;
  // Zero-gradient (Neumann) coarse ghost before the prolongation on wall/inflow faces — the WO-H
  // symmetry repair (see applyNeumannGhost). ON by default; `setCoarseGhost(false)` restores the
  // pre-2026-08-30 periodic-wrap ghost purely as a MEASUREMENT ABLATION (it reinstates the
  // asymmetry that stalls MG-PCG on every 3-D wall-bounded grid — never a production setting).
  // Inert on periodic/IBM problems (hasBC_ == false), where the fix is a no-op either way.
  bool bcGhost_ = true;
  // The aspect-ratio coarsening threshold `theta` of doc/anisotropic_metric.md §5.1
  // (`setAspectThreshold`; only ever consulted on an ANISOTROPIC metric). 1e9 restores full
  // coarsening (the §8.5 item (c) ablation), 1.4142 is the sqrt(2) variant §5.2 weighs against 2.0.
  double aspectTheta_ = 2.0;
  // Communication-avoiding smoothing (see the kCa* enum): both subsystems on by default. Only
  // kCaMg is read here; the momentum half lives in IbmSolver.
  int caMode_ = kCaBoth;
  // ISSUES sweep item 6. `setStrictPressure(true)` turns a non-finite preconditioner output into a
  // throw instead of a printed line + a zero correction. Off by default: the shipped behaviour is
  // to report the cap through `last_pressure_iterations()` and raise the `pressure_solve_failed()`
  // flag, so a rule-3b check catches it without changing control flow.
  bool strictPressure_ = false;
  // P1 of the defect-correction campaign — the exact (matrix-free, double, flux-form) level-0
  // operator apply in the residual and the Krylov matvec. See mac_cutcell.hpp.
  bool exactResidual_ = false;
  // D (§4.4.6): the V-cycle precision mode (kVcycleAuto / kVcycleFp64 / kVcycleFp32); 'fp64' until
  // WO-D3's promotion. opBuilt_: setOpenness has built the operator of the current hierarchy.
  int vprec_ = kVcycleFp64;
  bool opBuilt_ = false;
  bool lastVcycleFp32_ = false;
  bool healthTrace_ = false;       // §4.4.5 test hook (setHealthTrace)
  std::vector<double> healthLog_;  // the last single-rank PCG / FCG solve's health ratios
  // The `agglomerateBottom()` auto criterion: agglomerate once the coarsest GLOBAL grid exceeds
  // this many cells on any axis (`setAgglomerationExtent`).
  int agglomExtent_ = 4;
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
  // Coarse-level telescoping (see Telescope). DEFAULT ON since 2026-09-02 (the FoxBerry ladder);
  // `setTelescope(false)` restores the in-place-only hierarchy, which is byte-identical to before
  // telescoping existed. teleForce_ > 0 forces a telescope at that level even when in-place
  // coarsening is legal (tests: compare the two hierarchies on one problem).
  bool telescope_ = true;
  int teleForce_ = -1;
  bool teleActive_ = true;  // false on a rank that idles below a telescope point
  // Economic trigger (MueLu's "min rows per proc", PETSc's reduction factor): once a level's
  // smallest block extent drops below this, merge -- and merge far enough that the merged blocks
  // clear it -- even if in-place coarsening is still legal. A 1x3x3 block on 1024 ranks is a
  // halo exchange with nine cells of work behind it. 0 disables (merge only when blocked).
  int teleMinExtent_ = 4;
  // Repartition stages (setRepartition): bound the block a telescope stage may hand a rank by the
  // largest level-0 block, repartitioning the level onto a proportional ORB on fewer ranks where a
  // sibling merge would exceed it. OFF by default (sibling merges only, byte-identical to S4); the
  // Solver switches it on for a weighted level-0 decomposition.
  bool repartition_ = false;
  int gnxF_ = 0, gnyF_ = 0, gnzF_ = 0;  // GLOBAL fine dims (== local single-rank)
  // Fine-level 1/h^2 per axis, as handed to setOpenness. Only the exact (matrix-free) level-0
  // apply reads them; the bands carry gf already folded in.
  double gfx_ = 1.0, gfy_ = 1.0, gfz_ = 1.0;
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
  void setAgglomerationExtent(int cells) { agglomExtent_ = cells > 0 ? cells : 4; }
  int agglomerationExtent() const { return agglomExtent_; }
  void setTelescope(bool on) { telescope_ = on; }
  /// Neumann coarse ghost on wall/inflow faces before the prolongation (see bcGhost_).
  void setCoarseGhost(bool on) { bcGhost_ = on; }
  bool coarseGhost() const { return bcGhost_; }
  /// Anisotropic-coarsening aspect threshold theta (see aspectTheta_).
  void setAspectThreshold(double theta) { aspectTheta_ = theta; }
  double aspectThreshold() const { return aspectTheta_; }
  /// Communication-avoiding smoothing mask (kCaMomentum | kCaMg); only kCaMg is read here.
  void setCommAvoiding(int mask) { caMode_ = mask; }
  int commAvoiding() const { return caMode_; }
  /// Throw instead of reporting when the preconditioner returns a non-finite correction.
  void setStrictPressure(bool on) { strictPressure_ = on; }
  bool strictPressure() const { return strictPressure_; }
  /// The exact double flux-form level-0 apply in the residual and the Krylov matvec.
  void setExactResidual(bool on) { exactResidual_ = on; }
  bool exactResidual() const { return exactResidual_; }
  // WO-R2 item 1: the next setOpenness receives a variable-density COEFFICIENT field, so the
  // Dirichlet (outflow) domain-face rows must carry the caller's coefficient rather than the
  // literal openness 1.0. See applyBoundaryOpenness. Reset it for a raw-openness build.
  void setOutflowCoefficient(bool on) { outflowCoeff_ = on; }
  bool outflowCoefficient() const { return outflowCoeff_; }
  // Diagnostics for the tests / the prediction tool: telescope points this rank passed through,
  // and the coarsest GLOBAL grid of the hierarchy (valid on a rank that holds every level, e.g.
  // rank 0, which is always a group root).
  int telescopeCount() const {
    int n = 0;
#ifdef PECLET_FLOW_MPI
    for (const Level& l : lv_)
      n += l.tele ? 1 : 0;
#endif
    return n;
  }
  // Repartition stages this rank passed through (a subset of telescopeCount()).
  int repartitionCount() const {
    int n = 0;
#ifdef PECLET_FLOW_MPI
    for (const Level& l : lv_)
      n += (l.tele && l.tele->repartition()) ? 1 : 0;
#endif
    return n;
  }
  C3 coarsestGlobal() const {
    C3 g{gnxF_, gnyF_, gnzF_};
    for (std::size_t L = 0; L + 1 < lv_.size(); ++L)
      g = C3{g.x / lv_[L].ratio.x, g.y / lv_[L].ratio.y, g.z / lv_[L].ratio.z};
    return g;
  }
#ifdef PECLET_FLOW_MPI
  // The hierarchy initMpi WOULD build, as a pure function of (grid, rank count, levels, telescope,
  // decomposition mode) — no MPI, no allocation, so it runs on a laptop for any rank count. One row
  // per level: global dims, ranks holding it, block dims (block 0), ratio to the next level, and
  // whether the transition out of it telescopes. This is the pre-flight for a job (P1 of
  // docs/MG_TELESCOPING_PLAN.md) and replaces launching `np` oversubscribed processes to find out.
  // NOTE (Phase 2 C3): this planner models the ISOTROPIC rule -- it takes no metric, so it does
  // NOT apply the aspect-ratio deferral of doc/anisotropic_metric.md §5.  On an anisotropic
  // `extent` read the real table off the built hierarchy instead
  // (`Solver.pressure_mg_level_ratios()` / `CutcellMG::levelRatios()`).  Threading `hp` through
  // here and through scripts/check_decomposition.py is a follow-up the note does not order.
  struct PlanRow {
    C3 global, block, ratio;
    int ranks;
    bool tele;
    bool repartition = false;  // the stage out of this level is a Repartition (tele is set too)
  };
  // `decompLevels`/`maxImbalance` are the LEVEL-0 decomposition's parameters (what
  // Solver::setDecomposition takes), not the MG level count `nLevels`: predict is a pure function,
  // so a caller pre-flighting a coarse-first job must state the depth it will run with. Before the
  // env-var retirement these came from a process-global static, which is exactly why they are
  // parameters now.
  //
  // `weights` (optional) are per-cell weights of the global grid, x-fastest (gnx*gny*gnz) -- what
  // Solver::rebalanceByWeights takes -- and the forecast is then the hierarchy AFTER that call:
  // level 0 is the partition it builds, core's aligned weighted ORB `chooseAlignedWeighted(np, G,
  // w)` at its 1.05 budget (`decompLevels` / `maxImbalance` are ignored, as the rebalance ignores
  // Solver::setDecomposition), and the stages are those the Solver arms for a weighted dec0
  // (setRepartition(true): `chooseStageTarget` with maxBlockCells = the largest level-0 block, so
  // a stage may be a Repartition). `*align`, when given, receives that partition's alignment 2^a
  // -- rebalanceByWeights' return value. Without weights the forecast is unchanged.
  static std::vector<PlanRow> predict(int gnx, int gny, int gnz, int np, int nLevels,
                                      bool telescope, int minExtent = 4, int decompLevels = 0,
                                      double maxImbalance = 1.05,
                                      const std::vector<peclet::core::Real>* weights = nullptr,
                                      int* align = nullptr) {
    using Dec = peclet::core::decomp::BlockDecomposer<3>;
    std::vector<PlanRow> rows;
    auto can = [](int d) { return (d % 2 == 0) && (d / 2 >= 2); };
    auto evenOn = [](const Dec& d, int ax) {
      for (std::size_t b = 0; b < d.sizes().size(); ++b)
        if ((d.origins()[b][ax] % 2) || (d.sizes()[b][ax] % 2))
          return false;
      return true;
    };
    Dec cur;
    peclet::core::Index maxBlockCells = 0;  // 0: sibling merges only (an unweighted dec0)
    if (weights) {
      if (weights->size() != (std::size_t)gnx * (std::size_t)gny * (std::size_t)gnz)
        throw std::invalid_argument(
            "predict: weights must hold one value per global cell (gnx*gny*gnz, x-fastest)");
      auto c = peclet::core::decomp::chooseAlignedWeighted(
          static_cast<std::size_t>(np), peclet::core::IVec<3>{gnx, gny, gnz}, *weights);
      cur = std::move(c.dec);
      maxBlockCells = peclet::core::decomp::largestBlockCells(cur);
      if (align)
        *align = 1 << c.a;
    } else {
      cur = decomposition(static_cast<std::size_t>(np), gnx, gny, gnz, decompLevels, maxImbalance);
      if (align)
        *align = 1;
    }
    C3 gs{gnx, gny, gnz};
    for (int L = 0; L < nLevels; ++L) {
      PlanRow r;
      r.global = gs;
      r.ranks = (int)cur.numBlocks();
      r.block = C3{(int)cur.sizes()[0][0], (int)cur.sizes()[0][1], (int)cur.sizes()[0][2]};
      r.tele = false;
      r.ratio = C3{1, 1, 1};
      C3 next = gs;
      if (L + 1 < nLevels) {
        const bool canAny = can(gs.x) || can(gs.y) || can(gs.z);
        auto liftable = [&](const Dec& d) { return teleLiftable(d, gs); };
        const bool blocked = !liftable(cur);
        const bool tooSmall = minExtent > 0 && peclet::core::decomp::minBlockExtent(cur) <
                                                   (peclet::core::Index)minExtent;
        // initMpi's trigger + core's policy (maxBlockCells = 0: flow's sibling-merge search,
        // which never returns a Repartition; > 0 for a weighted dec0, as setRepartition arms it)
        if (telescope && (blocked || tooSmall) && canAny && cur.numBlocks() > 1) {
          auto t = peclet::core::decomp::chooseStageTarget(
              cur, peclet::core::IVec<3>{gs.x, gs.y, gs.z}, liftable, minExtent, maxBlockCells);
          if (t.kind == peclet::core::decomp::StageKind::SiblingMerge ||
              t.kind == peclet::core::decomp::StageKind::Repartition) {
            r.repartition = t.kind == peclet::core::decomp::StageKind::Repartition;
            cur = std::move(t.dec);
            r.tele = true;
          }
        }
        if (can(gs.x) && evenOn(cur, 0)) {
          r.ratio.x = 2;
          next.x = gs.x / 2;
        }
        if (can(gs.y) && evenOn(cur, 1)) {
          r.ratio.y = 2;
          next.y = gs.y / 2;
        }
        if (can(gs.z) && evenOn(cur, 2)) {
          r.ratio.z = 2;
          next.z = gs.z / 2;
        }
      }
      rows.push_back(r);
      if (next.x == gs.x && next.y == gs.y && next.z == gs.z)
        break;
      gs = next;
      cur = cur.coarsened(peclet::core::IVec<3>{r.ratio.x, r.ratio.y, r.ratio.z});
    }
    return rows;
  }
  // The hierarchy initMpi BUILT, in predict()'s row format -- the oracle predict is tested
  // against. Rank 0 holds every level (it is the root of every stage), so there this is the whole
  // table; a rank idling below a stage gets the levels it holds.
  std::vector<PlanRow> builtPlan() const {
    std::vector<PlanRow> rows;
    for (const Level& l : lv_) {
      PlanRow r;
      r.global = l.gdim;
      r.block = l.inner;
      r.ratio = l.ratio;
      r.ranks = 1;
      if (l.comm != MPI_COMM_NULL)
        MPI_Comm_size(l.comm, &r.ranks);
      r.tele = (bool)l.tele;
      r.repartition = l.tele && l.tele->repartition();
      rows.push_back(r);
    }
    return rows;
  }
#endif  // PECLET_FLOW_MPI (predict)
  bool telescope() const { return telescope_; }
  void setTelescopeForceLevel(int L) { teleForce_ = L; }  // tests only; -1 = never force
  void setTelescopeMinExtent(int e) { teleMinExtent_ = e; }
  /// Repartition stages for a weighted level-0 decomposition (see repartition_); before initMpi.
  void setRepartition(bool on) { repartition_ = on; }
  bool repartition() const { return repartition_; }
  // Number of levels THIS rank holds (fewer than the hierarchy's on a rank idling below a
  // telescope point) and the hierarchy's rank count per level (replicated).
  int localLevels() const { return (int)lv_.size(); }
  int agglomerationMode() const { return agglomMode_; }
  void setGraphAmgBottom(bool on) {
    agglomMode_ = on ? 1 : 0;
    amg_.reset();
  }
  // Engine selection for an agglomerated bottom (§13 D-5, §14 H-1): kBottomAuto = the FCG bottom
  // preconditioned by the direct factor wherever directBottomIneligible() is null (every backend
  // since §14 H-1), GraphAMG otherwise; kBottomDirect = that engine or a raise naming the failed
  // condition; kBottomAlgebraic = GraphAMG always (the A/B instrument).
  enum : int { kBottomAuto = 0, kBottomDirect = 1, kBottomAlgebraic = 2 };
  void setBottomSolver(int m) { bottomSolver_ = m; }
  int bottomSolver() const { return bottomSolver_; }
  // Test hooks (tests/kokkos/test_bottom_direct.cpp). Since §14 H-1 every backend builds the
  // bottom's storage at init (single rank); `bottomForceForTest` builds it (and labels the
  // components) if it is missing.
  bool bottomConnected() const { return bottomConnected_; }
  int bottomComponents() const { return bottomNComp_; }
  void bottomForceForTest() {
    if (!bottomStore_) {
      buildBottomStore();
      evalBottomComponents();
    }
  }
  // §13 hooks.
  // `directFactorForTest<FR>`: a fresh factor of the bottom in FR arithmetic (team T, 0 = the
  // default), `*Tused` = the team size it ran with; `teamAlgorithm`: by the team algorithm even on
  // a host backend (the host schedule's oracle, U8). `directApplyForTest`: z = M(r) by that factor
  // through the FCG kernel's precondition-only path (`noMean`: without the component means'
  // removal); returns the team size. `directSolveForTest`: the production solve (FP32 factor,
  // lazily refactored) on the bottom's rhs; returns the inner iterations.
  const BottomPlanes& directPlanes() const { return dirPlanes_; }
  template <class FR>
  BottomDirect<FR> directFactorForTest(int team, double tauPiv0 = kBottomPivotTol,
                                       int* Tused = nullptr, bool teamAlgorithm = false) {
    BottomDirect<FR> D = BottomDirect<FR>::allocate(dirPlanes_, lv_.back().ext);
    const int T = launchDirectFactor(D, team, tauPiv0, true, teamAlgorithm);
    Kokkos::fence();
    if (Tused)
      *Tused = T;
    return D;
  }
  template <class FR>
  int directApplyForTest(const BottomDirect<FR>& D, CCField z, CCField r, int team,
                         bool noMean = false) {
    const int T = launchDirectKernel(D, true, team, noMean, z, r);
    Kokkos::fence();
    return T;
  }
  int directSolveForTest() {
    directBottomSolve();
    int it = 0;
    Kokkos::deep_copy(it, Kokkos::subview(bottomInfo_, 0));
    return it;
  }
  // §14 H-1 (U3 on a host backend): refactor and solve the bottom's rhs at team size `team`
  // (clamped to team_size_max), `*Tused` = the size both launches ran with (-1 if they differ);
  // returns the inner iterations. The production team sizes are re-probed at the next solve.
  int directSolveTeamForTest(int team, int* Tused = nullptr) {
    const int Tf = launchDirectFactor(dir_, team, dirPivotTol_);
    const int Ts = launchDirectKernel(dir_, false, team, false);
    bottomFlagPending_ = true;
    Kokkos::fence();
    if (Tused)
      *Tused = (Tf == Ts) ? Ts : -1;
    dirTeam_ = 0;
    dirFacTeam_ = 0;
    facStale_ = true;
    int it = 0;
    Kokkos::deep_copy(it, Kokkos::subview(bottomInfo_, 0));
    return it;
  }
  int directRestarts() const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dir_.stat);
    return h(1);
  }
  void directPivotTolForTest(double t) { dirPivotTol_ = t; }
  void directMarkStaleForTest() { facStale_ = true; }

 private:
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_CUTCELL_MG_HPP
