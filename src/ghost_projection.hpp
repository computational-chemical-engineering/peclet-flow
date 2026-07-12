/// @file
/// @brief flow — directional ghost-cell IBM projection overlay (experimental second staggered
/// IBM).
///
/// Point-based finite-difference projection near the immersed boundary, NO openness factors: the
/// divergence of a fluid-centered pressure cell uses plain face differences; a face whose
/// staggered velocity point is solid is closed by the momentum IBM's 1-D wall-anchored quadratic
/// along the face's own axis (poly_D/poly_Nc/poly_N_nb of cut_cell_ibm.hpp, reused verbatim):
///
///     poly_D(th) * u_ghost = 2*u_bc + poly_Nc(th)*u_near + poly_N_nb(th)*u_far
///     th = sdf_near/(sdf_near - sdf_ghost), clamped [GP_THETA_MIN, 1]
///
/// Substituting the corrected velocity u = u* - grad(phi) makes the closure implicit in phi: the
/// Poisson row gains couplings to phi(+/-1), phi(+/-2) along the axis (up to 13-point,
/// nonsymmetric). The symmetric part is realised as the BINARY-openness 7-point operator
/// (gpBinaryOpenness -> buildCutcellOp -> the whole CutcellMG hierarchy unchanged, as the
/// preconditioner); the nonsymmetric remainder is a compact per-row overlay applied by
/// gpApplyDelta (matvec) and gpDivergDelta (RHS + diagnostic), both scaled by the per-row
/// conditioning rescale rho = min(1, min_f D_f) — the D_rescale analog.
///
/// Face-state cascade per (cell, axis, side), sdf >= 0 fluid, classification from CELL-CENTERED
/// sdf with face values = mean of the two adjacent centers (identical to the trilinear
/// ccSampleExt at the +/-0.5 staggered offsets, so it agrees with the momentum solid masks):
///   COUPLED   face point fluid AND neighbor center fluid -> standard +/- (phi_i - phi_nb)
///   (sandwich) both own face points solid, center fluid  -> BC_ONLY both sides (no fluid
///             unknown lives between the walls: the wall BCs alone determine the axis)
///   QUAD      face point solid, near+far sources usable  -> quadratic closure, th in (0,1]
///   LIN       face point solid, only near source         -> th*u_g = u_bc + (th-1)*u_near
///   (sliver)  face point fluid but neighbor center solid -> same QUAD/LIN with EXTENDED
///             th = 1 + sdf_g/(sdf_g - sdf_beyond) in (1,2)  (interpolation, D > 2)
///   BC_ONLY   no usable fluid source                     -> u_face = u_bc, no phi coupling
///   EXPLICIT  sliver with no crossing on the u-line      -> face flux = u* (no phi coupling)
/// Rows with no phi coupling at all are decoupled (phi = 0, RHS zeroed).
///
/// v1 scope: single-rank, periodic + IBM only (neighbor access uses explicit periodic wrap over
/// the INNER grid, so the +/-2 reach never depends on the ghost-layer depth), stationary walls
/// (u_bc = 0 at runtime; the w_bc weights are stored for the parity tests / future moving walls).
/// Reference implementation + gates: tests/study/ghost_projection_apriori.py.
#ifndef PECLET_FLOW_GHOST_PROJECTION_HPP
#define PECLET_FLOW_GHOST_PROJECTION_HPP

#include <Kokkos_Core.hpp>

#include <cstdint>

#include "cut_cell_ibm.hpp"  // poly_D / poly_Nc / poly_N_nb
#include "mac_cutcell.hpp"   // CCField/CCConst, C3, CCExec

namespace peclet::flow {

constexpr float GP_THETA_MIN = 1e-4f;

enum GpState : int8_t {
  GP_COUPLED = 0,
  GP_QUAD = 1,
  GP_LIN = 2,
  GP_BC_ONLY = 3,
  GP_EXPLICIT = 4,
};

/// Per-overlay-row SoA. Face slot k = 2*axis + (0 = plus side, 1 = minus side), matching the
/// momentum overlay's direction order {+x,-x,+y,-y,+z,-z}. Weights are the UNSCALED closure
/// coefficients (u_face = w_bc*u_bc + w_n1*u_near + w_n2*u_far); the row rescale rho is applied
/// at row level by the delta kernels.
template <class Space>
struct GpOverlayT {
  Kokkos::View<int*, Space> cell;         // packed INNER flat index x + y*nx + z*nx*ny
  Kokkos::View<float*, Space> rescale;    // rho = min(1, min_f D_f)
  Kokkos::View<int8_t*, Space> coupled;   // 1 if the row has any phi coupling at all
  Kokkos::View<int8_t*, Space> state;     // [slot*6+k]
  Kokkos::View<float*, Space> th;         // [slot*6+k] (parity/diagnostics)
  Kokkos::View<float*, Space> w_bc, w_n1, w_n2;  // [slot*6+k]
};
using GpOverlay = GpOverlayT<CCMem>;

inline GpOverlay gpMakeOverlay(long n) {
  GpOverlay ov;
  ov.cell = Kokkos::View<int*, CCMem>("gp_cell", n);
  ov.rescale = Kokkos::View<float*, CCMem>("gp_rescale", n);
  ov.coupled = Kokkos::View<int8_t*, CCMem>("gp_coupled", n);
  ov.state = Kokkos::View<int8_t*, CCMem>("gp_state", 6 * n);
  ov.th = Kokkos::View<float*, CCMem>("gp_th", 6 * n);
  ov.w_bc = Kokkos::View<float*, CCMem>("gp_wbc", 6 * n);
  ov.w_n1 = Kokkos::View<float*, CCMem>("gp_wn1", 6 * n);
  ov.w_n2 = Kokkos::View<float*, CCMem>("gp_wn2", 6 * n);
  return ov;
}

struct GpFace {
  int8_t state;
  float th, wbc, w1, w2, D;
};

/// Classify + fill ONE face from its 1-D sdf samples along the face's axis.
///   sg  = ghost (this) face sdf     sn = near face sdf (the cell's other face on this axis)
///   sf  = far face sdf              sb = beyond-ghost face sdf (one further into the solid)
///   snb = neighbor center sdf       sc1/sc2 = centers needed by the near/far face gradients
///   otherSolid = the cell's other face on this axis is solid (sandwich detection)
/// Pure function of the samples — shared verbatim with the host parity test.
KOKKOS_INLINE_FUNCTION GpFace gpClassifyFace(float sg, float sn, float sf, float sb, float snb,
                                             float sc1, float sc2, bool otherSolid) {
  GpFace f{GP_COUPLED, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  if (sg >= 0.0f && snb >= 0.0f)
    return f;  // COUPLED
  if (sg < 0.0f && otherSolid) {
    f.state = GP_BC_ONLY;  // sandwich: both own faces solid, wall BCs determine the axis
    f.wbc = 1.0f;
    return f;
  }
  float th;
  if (sg < 0.0f) {  // standard ghost (near face fluid guaranteed: not sandwich)
    th = sn / (sn - sg);
    th = th < GP_THETA_MIN ? GP_THETA_MIN : (th > 1.0f ? 1.0f : th);
  } else {  // sliver: face point fluid, neighbor center solid
    if (sb >= 0.0f) {
      f.state = GP_EXPLICIT;  // no crossing on the u-line: explicit u* flux, no phi coupling
      return f;
    }
    th = 1.0f + sg / (sg - sb);  // extended theta in (1,2): evaluation INSIDE the data hull
    const float lo = 1.0f + GP_THETA_MIN;
    th = th < lo ? lo : (th > 2.0f ? 2.0f : th);
  }
  f.th = th;
  const bool src1 = (sn >= 0.0f) && (sc1 >= 0.0f);
  const bool src2 = (sf >= 0.0f) && (sc2 >= 0.0f);
  if (!src1) {
    f.state = GP_BC_ONLY;
    f.wbc = 1.0f;
    return f;
  }
  if (src2) {
    f.state = GP_QUAD;
    f.D = poly_D(th);
    f.wbc = 2.0f / f.D;
    f.w1 = poly_Nc(th) / f.D;
    f.w2 = poly_N_nb(th) / f.D;
  } else {
    f.state = GP_LIN;
    f.D = th;
    f.wbc = 1.0f / th;
    f.w1 = (th - 1.0f) / th;
  }
  return f;
}

/// Fill one overlay row from the per-axis sample sets. F[a][m+1] = face sdf at face index i+m
/// (m = -1..2, face i+m sits between centers i+m-1 and i+m); Cq[a][q+2] = center sdf at i+q
/// (q = -2..2; Cq[a][2] = own center, fluid by construction). Returns true if any face is
/// non-COUPLED (i.e. the row belongs in the overlay).
template <class OV>
KOKKOS_INLINE_FUNCTION bool gpFillRow(const OV& ov, int slot, int cellInner, const float F[3][4],
                                      const float Cq[3][5]) {
  bool any = false;
  bool anyPhi = false;
  float rho = 1.0f;
  GpFace faces[6];
  for (int a = 0; a < 3; ++a) {
    const bool solidM = F[a][1] < 0.0f;  // own minus face (m=0)
    const bool solidP = F[a][2] < 0.0f;  // own plus face (m=1)
    // minus side (k = 2a+1): ghost m=0, near m=1, far m=2, beyond m=-1; nb center q=-1;
    // gradient cells q=+1, q=+2.
    faces[2 * a + 1] = gpClassifyFace(F[a][1], F[a][2], F[a][3], F[a][0], Cq[a][1], Cq[a][3],
                                      Cq[a][4], solidP);
    // plus side (k = 2a): ghost m=1, near m=0, far m=-1, beyond m=2; nb center q=+1;
    // gradient cells q=-1, q=-2.
    faces[2 * a] = gpClassifyFace(F[a][2], F[a][1], F[a][0], F[a][3], Cq[a][3], Cq[a][1],
                                  Cq[a][0], solidM);
  }
  for (int k = 0; k < 6; ++k) {
    const GpFace& f = faces[k];
    if (f.state != GP_COUPLED)
      any = true;
    if (f.state == GP_COUPLED || f.state == GP_QUAD || f.state == GP_LIN)
      anyPhi = true;
    if ((f.state == GP_QUAD || f.state == GP_LIN) && f.D < rho)
      rho = f.D;
  }
  if (!any)
    return false;
  ov.cell(slot) = cellInner;
  ov.rescale(slot) = rho;
  ov.coupled(slot) = anyPhi ? 1 : 0;
  for (int k = 0; k < 6; ++k) {
    ov.state(slot * 6 + k) = faces[k].state;
    ov.th(slot * 6 + k) = faces[k].th;
    ov.w_bc(slot * 6 + k) = faces[k].wbc;
    ov.w_n1(slot * 6 + k) = faces[k].w1;
    ov.w_n2(slot * 6 + k) = faces[k].w2;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION int gpWrap(int v, int n) {
  v %= n;
  return v < 0 ? v + n : v;
}

/// Build the overlay over the inner grid nn from the cell-centered sdf on the extended block
/// (ext, ghost width g). Neighbor access wraps periodically over the inner grid (v1 is
/// single-rank periodic), so the +/-2 reach never depends on g. Overlay arrays must be sized for
/// the worst case; returns the row count. idMap (size nn.x*nn.y*nn.z) gets slot or -1.
inline int buildGpOverlay(CCConst sdf, C3 ext, int g, C3 nn, const GpOverlay& ov,
                          Kokkos::View<int*, CCMem> idMap, Kokkos::View<int, CCMem> counter) {
  CCExec space;
  Kokkos::deep_copy(space, counter, 0);
  Kokkos::deep_copy(space, idMap, -1);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::gp_build_overlay", MD(space, {0, 0, 0}, {nn.x, nn.y, nn.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        auto S = [&](int dx, int dy, int dz) {
          const long i = (long)(gpWrap(x + dx, nn.x) + g) +
                         (long)(gpWrap(y + dy, nn.y) + g) * ext.x +
                         (long)(gpWrap(z + dz, nn.z) + g) * (long)ext.x * ext.y;
          return (float)sdf(i);
        };
        const float sc = S(0, 0, 0);
        if (sc < 0.0f)
          return;  // solid-centered: decoupled row (phi = 0), not in the overlay
        float F[3][4], Cq[3][5];
        const int d[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int a = 0; a < 3; ++a) {
          for (int q = -2; q <= 2; ++q)
            Cq[a][q + 2] = S(q * d[a][0], q * d[a][1], q * d[a][2]);
          for (int m = -1; m <= 2; ++m)  // face i+m = mean of centers i+m-1, i+m
            F[a][m + 1] = 0.5f * (Cq[a][m + 1] + Cq[a][m + 2]);
        }
        // cheap pre-check: fully interior rows (all six 1st neighbors fluid on faces+centers)
        bool clean = true;
        for (int a = 0; a < 3; ++a)
          clean = clean && F[a][1] >= 0.0f && F[a][2] >= 0.0f && Cq[a][1] >= 0.0f &&
                  Cq[a][3] >= 0.0f;
        if (clean)
          return;
        const int inner = x + y * nn.x + z * nn.x * nn.y;
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        if (!gpFillRow(ov, slot, inner, F, Cq)) {
          // all faces COUPLED after all (pre-check was conservative): release the slot lazily by
          // marking it inert (zero-weight row on its own cell).
          ov.cell(slot) = inner;
          ov.rescale(slot) = 1.0f;
          ov.coupled(slot) = 1;
          for (int k = 0; k < 6; ++k) {
            ov.state(slot * 6 + k) = GP_COUPLED;
            ov.th(slot * 6 + k) = 1.0f;
            ov.w_bc(slot * 6 + k) = 0.0f;
            ov.w_n1(slot * 6 + k) = 0.0f;
            ov.w_n2(slot * 6 + k) = 0.0f;
          }
        } else {
          idMap(inner) = slot;
        }
      });
  int cnt = 0;
  Kokkos::deep_copy(cnt, counter);
  return cnt;
}

/// Binary openness for the symmetric MG surrogate, on the extended-block layout of buildOpenness:
/// o(face) = 1 iff the face point is fluid AND both adjacent cell centers are fluid, else 0.
/// buildCutcellOp on this field gives exactly the COUPLED part of the ghost operator; the whole
/// CutcellMG hierarchy (coarsening, smoothing, GraphAMG bottom) runs on it unchanged.
inline void gpBinaryOpenness(CCField ox, CCField oy, CCField oz, CCConst sdf, C3 ext) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::gp_binary_openness", MD(space, {0, 0, 0}, {ext.x, ext.y, ext.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sy = ext.x, sz = (long)ext.x * ext.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
        auto S = [&](int xx, int yy, int zz) {
          return sdf((long)cl(xx, ext.x) + (long)cl(yy, ext.y) * sy + (long)cl(zz, ext.z) * sz);
        };
        const double sc = S(x, y, z);
        const double sw = S(x - 1, y, z), ss = S(x, y - 1, z), sb = S(x, y, z - 1);
        ox(i) = (0.5 * (sw + sc) >= 0.0 && sw >= 0.0 && sc >= 0.0) ? 1.0 : 0.0;
        oy(i) = (0.5 * (ss + sc) >= 0.0 && ss >= 0.0 && sc >= 0.0) ? 1.0 : 0.0;
        oz(i) = (0.5 * (sb + sc) >= 0.0 && sb >= 0.0 && sc >= 0.0) ? 1.0 : 0.0;
      });
}

/// Overlay matvec correction: y(r) = rho_r * (y(r) + closure-face phi terms), where y currently
/// holds the binary-openness (COUPLED-part) matvec. x/y live on a block of ghost width gb with
/// extents extb (works for both the g=1 MG block and the g=2 solver block); neighbor access
/// wraps over the inner grid nn. Face at relative index m couples cells (i+m-1, i+m); the div
/// coefficient c = sgn*w contributes A x += -c*x(i+m) + c*x(i+m-1)  =>  delta = sgn*w*(x_cm -
/// x_cp). Distinct rows per thread: no atomics.
inline void gpApplyDelta(CCField y, CCConst x, const GpOverlay& ov, int nOv, C3 nn, C3 extb,
                         int gb) {
  if (nOv <= 0)
    return;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::gp_apply_delta", Kokkos::RangePolicy<CCExec>(space, 0, nOv),
      KOKKOS_LAMBDA(int s) {
        const int inner = ov.cell(s);
        const int ix = inner % nn.x, iy = (inner / nn.x) % nn.y, iz = inner / (nn.x * nn.y);
        auto X = [&](int a, int q) {  // phi at cell offset q along axis a
          const int cx = a == 0 ? gpWrap(ix + q, nn.x) : ix;
          const int cy = a == 1 ? gpWrap(iy + q, nn.y) : iy;
          const int cz = a == 2 ? gpWrap(iz + q, nn.z) : iz;
          return x((long)(cx + gb) + (long)(cy + gb) * extb.x +
                   (long)(cz + gb) * (long)extb.x * extb.y);
        };
        const long r = (long)(ix + gb) + (long)(iy + gb) * extb.x +
                       (long)(iz + gb) * (long)extb.x * extb.y;
        double delta = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int8_t st = ov.state(s * 6 + k);
          if (st != GP_QUAD && st != GP_LIN)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;  // odd k = minus side
          const int mn = (k & 1) ? 1 : 0;    // near-face relative index
          const int mf = (k & 1) ? 2 : -1;   // far-face relative index
          const double w1 = ov.w_n1(s * 6 + k), w2 = ov.w_n2(s * 6 + k);
          delta += sgn * w1 * (X(a, mn - 1) - X(a, mn));
          if (st == GP_QUAD)
            delta += sgn * w2 * (X(a, mf - 1) - X(a, mf));
        }
        const double rho = ov.rescale(s);
        y(r) = ov.coupled(s) ? rho * (y(r) + delta) : 0.0;
      });
}

/// Overlay divergence correction: d(r) = rho_r * (d(r) + closure/BC/explicit face values), where
/// d currently holds the binary-openness divergence (divergOpen on the binary fields: COUPLED
/// faces only). u/v/w and d live on the solver block (extents extb, ghost width gb). u_bc = 0
/// (v1 stationary walls). Rows with no phi coupling are zeroed (decoupled). Used identically for
/// the RHS div(u*) and the post-correction diagnostic — the diagnostic IS the residual.
inline void gpDivergDelta(CCField d, CCConst u, CCConst v, CCConst w, const GpOverlay& ov,
                          int nOv, C3 nn, C3 extb, int gb) {
  if (nOv <= 0)
    return;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::gp_diverg_delta", Kokkos::RangePolicy<CCExec>(space, 0, nOv),
      KOKKOS_LAMBDA(int s) {
        const int inner = ov.cell(s);
        const int ix = inner % nn.x, iy = (inner / nn.x) % nn.y, iz = inner / (nn.x * nn.y);
        auto U = [&](int a, int m) {  // face-field value at face index i+m along axis a
          const int cx = a == 0 ? gpWrap(ix + m, nn.x) : ix;
          const int cy = a == 1 ? gpWrap(iy + m, nn.y) : iy;
          const int cz = a == 2 ? gpWrap(iz + m, nn.z) : iz;
          const long i = (long)(cx + gb) + (long)(cy + gb) * extb.x +
                         (long)(cz + gb) * (long)extb.x * extb.y;
          return a == 0 ? u(i) : (a == 1 ? v(i) : w(i));
        };
        const long r = (long)(ix + gb) + (long)(iy + gb) * extb.x +
                       (long)(iz + gb) * (long)extb.x * extb.y;
        double dd = d(r);
        for (int k = 0; k < 6; ++k) {
          const int8_t st = ov.state(s * 6 + k);
          if (st == GP_COUPLED)
            continue;
          const int a = k / 2;
          const int sgn = (k & 1) ? -1 : 1;
          const int mg = (k & 1) ? 0 : 1;  // the closed face's own index
          const int mn = (k & 1) ? 1 : 0;
          const int mf = (k & 1) ? 2 : -1;
          if (st == GP_EXPLICIT) {
            dd += sgn * U(a, mg);  // sliver without crossing: explicit u* flux
            continue;
          }
          if (st == GP_BC_ONLY)
            continue;  // u_bc = 0 (v1); w_bc kept in the overlay for moving walls
          double val = ov.w_n1(s * 6 + k) * U(a, mn);  // + w_bc*u_bc (= 0)
          if (st == GP_QUAD)
            val += ov.w_n2(s * 6 + k) * U(a, mf);
          dd += sgn * val;
        }
        d(r) = ov.coupled(s) ? ov.rescale(s) * dd : 0.0;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_GHOST_PROJECTION_HPP
