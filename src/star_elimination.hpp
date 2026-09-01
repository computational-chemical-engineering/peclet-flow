/// @file
/// @brief flow — Design B of the fluid-only collocated constraint (route 2b): Kron (star-mesh)
/// elimination of solid-centered pressure DOFs from the aperture Poisson graph.
///
/// Each solid-centered cell s with aperture-open faces to fluid-centered axis neighbours j is
/// eliminated by the conductance-weighted average  phibar_s = sum_j w_j phi_j,
/// w_j = a_sj / D_s, D_s = sum_j a_sj — the unique weights for which the eliminated operator is
/// SYMMETRIC (coupling a_si a_sj / D_s), and equivalently the least-squares value of phi_s in the
/// aperture metric. The full operator is
///     A_B = A_filtered  +  S_star,
///     A_filtered = 7-point cut-cell op on the FILTERED openness (faces with a solid-centered
///                  side closed — Design A's operator, which also feeds the MG hierarchy as the
///                  symmetric preconditioner),
///     (S_star x)_i = sum_s a_si (x_i - phibar_s(x))    [same sign convention as applyCutcellOp],
/// S_star is PSD (phi^T S phi = sum_s D_s Var_w(phi) >= 0), so A_B is SPD and MG-PCG applies with
/// the filtered hierarchy as surrogate preconditioner (the ghost projection's overlay pattern,
/// but symmetric). The divergence keeps the ORIGINAL apertures on fluid rows (throat flux is
/// counted), solid rows are dropped from the constraint (their rhs is masked; their phi stays 0),
/// and the face correction at a fluid|solid face uses (phibar_s - phi_i) so fluid control
/// volumes end exactly divergence-free — operator == correction by construction.
/// See doc/fluid_only_constraint_plan.md. v1: single-rank, periodic+IBM, collocated only.
#ifndef PECLET_FLOW_STAR_ELIMINATION_HPP
#define PECLET_FLOW_STAR_ELIMINATION_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"  // CCField/CCConst, C3, CCExec, CCMem

namespace peclet::flow {

/// One entry per eliminated solid-centered cell: packed INNER flat index + the apertures of its
/// (up to 6) faces to FLUID-CENTERED axis neighbours (order: +x,-x,+y,-y,+z,-z; 0 where the
/// neighbour is solid-centered or the face is closed). D = the stored row sum (> 0 by
/// construction — cells with no open fluid face are not entered).
struct StarOverlay {
  Kokkos::View<int*, CCMem> cell;
  // [slot*6+k]. DOUBLE since P2: this is the masked face openness, and the openness source
  // (Level::ox/oy/oz) is double — storing it float made the star row a float-rounded copy of a
  // coefficient the solver already holds exactly. Gate-off reads it back through (float) so the
  // default path is bitwise what it was; see starAval().
  Kokkos::View<double*, CCMem> a;
};

inline StarOverlay starMakeOverlay(long n) {
  StarOverlay ov;
  ov.cell = Kokkos::View<int*, CCMem>("star_cell", n);
  ov.a = Kokkos::View<double*, CCMem>("star_a", 6 * n);
  return ov;
}

/// One stored aperture, at the precision the gate selects. Gate OFF returns the float-rounded
/// value the overlay used to store, so the default path is bitwise unchanged by the widening;
/// gate ON returns the exact double the openness actually carries.
KOKKOS_INLINE_FUNCTION double starAval(const StarOverlay& ov, long k, bool exact) {
  const double v = ov.a(k);
  return exact ? v : (double)(float)v;
}

KOKKOS_INLINE_FUNCTION int starWrap(int v, int n) {
  v %= n;
  return v < 0 ? v + n : v;
}

/// Count + fill the star overlay from the cell-centered sdf and the ORIGINAL (unfiltered)
/// apertures on the extended block (ext, ghost width g); nn = inner extents. Single-rank periodic
/// (neighbour access wraps over the inner grid). Pass ov with capacity >= the count of a prior
/// sizing call (fill = ov.cell.extent(0) > 0). Returns the row count via the host copy of
/// counter.
inline int buildStarOverlay(CCConst sdf, CCConst ox, CCConst oy, CCConst oz, C3 ext, int g, C3 nn,
                            const StarOverlay& ov, Kokkos::View<int, CCMem> counter) {
  CCExec space;
  Kokkos::deep_copy(counter, 0);
  const bool fill = ov.cell.extent(0) > 0;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::star_build", MD(space, {0, 0, 0}, {nn.x, nn.y, nn.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
        const long st[3] = {sx, sy, sz};
        const long i = (long)(x + g) + (long)(y + g) * sy + (long)(z + g) * sz;
        if (sdf(i) >= 0.0)
          return;  // fluid-centered: not eliminated
        CCConst oa[3] = {ox, oy, oz};
        double av[6];
        double D = 0.0;
        for (int a2 = 0; a2 < 3; ++a2) {
          const double op = oa[a2](i + st[a2]), om = oa[a2](i);  // +side face / -side face
          const bool fp = sdf(i + st[a2]) >= 0.0, fm = sdf(i - st[a2]) >= 0.0;
          av[2 * a2] = (fp && op > 0.0) ? op : 0.0;
          av[2 * a2 + 1] = (fm && om > 0.0) ? om : 0.0;
          D += av[2 * a2] + av[2 * a2 + 1];
        }
        if (D <= 0.0)
          return;  // no open fluid face: fully decoupled solid cell
        const int slot = Kokkos::atomic_fetch_add(&counter(), 1);
        if (fill) {
          ov.cell(slot) = x + y * nn.x + z * nn.x * nn.y;
          for (int k = 0; k < 6; ++k)
            ov.a(slot * 6 + k) = av[k];
        }
      });
  space.fence();
  auto h = Kokkos::create_mirror_view(counter);
  Kokkos::deep_copy(h, counter);
  return h();
}

/// y += S_star x over the inner cells of the (extY, gY) block, x read from the (extX, gX) block,
/// both periodic via inner-grid wrap. Sign convention of applyCutcellOp (positive-definite):
/// row i gains a_si (x_i - phibar_s). Atomic adds (several stars can touch one row).
inline void starApplyDelta(CCField y, CCConst x, const StarOverlay& ov, int nOv, C3 nn, C3 extY,
                           int gY, C3 extX, int gX) {
  if (nOv <= 0)
    return;
  CCExec space;
  const bool exact = exactResidual();
  Kokkos::parallel_for(
      "peclet::flow::star_apply", Kokkos::RangePolicy<CCExec>(space, 0, nOv),
      KOKKOS_LAMBDA(int s) {
        const int inner = ov.cell(s);
        const int ix = inner % nn.x, iy = (inner / nn.x) % nn.y, iz = inner / (nn.x * nn.y);
        auto idx = [&](int cx, int cy, int cz, C3 ext, int gb) {
          return (long)(starWrap(cx, nn.x) + gb) + (long)(starWrap(cy, nn.y) + gb) * ext.x +
                 (long)(starWrap(cz, nn.z) + gb) * (long)ext.x * ext.y;
        };
        const int nb[6][3] = {{ix + 1, iy, iz}, {ix - 1, iy, iz}, {ix, iy + 1, iz},
                              {ix, iy - 1, iz}, {ix, iy, iz + 1}, {ix, iy, iz - 1}};
        double D = 0.0, num = 0.0;
        double av[6], xv[6];
        for (int k = 0; k < 6; ++k) {
          av[k] = starAval(ov, (long)s * 6 + k, exact);
          if (av[k] <= 0.0)
            continue;
          xv[k] = x(idx(nb[k][0], nb[k][1], nb[k][2], extX, gX));
          D += av[k];
          num += av[k] * xv[k];
        }
        if (exact) {
          // FLUX FORM (P2, the P1 trick applied to the star row). Algebraically identical to the
          // phibar form below --
          //   a_k (x_k - phibar) = (a_k/D)(D x_k - sum_j a_j x_j) = (a_k/D) sum_j a_j (x_k - x_j)
          // -- but every term is a coefficient times a DIFFERENCE, so a constant x is annihilated
          // BITWISE. The phibar form is not: phibar = sum(a x)/sum(a) does not return x exactly
          // for constant x even in double (the two sums round independently, then divide), so the
          // star delta leaked a nonzero row contribution into A*1 that P1's exact 7-point apply
          // could not cancel. Composing the two is the whole point of the rung: the level-0
          // operator is the 7-point part PLUS this delta, and A*1 = 0 needs both halves exact.
          for (int k = 0; k < 6; ++k) {
            if (av[k] <= 0.0)
              continue;
            double acc = 0.0;
            for (int j = 0; j < 6; ++j)
              if (av[j] > 0.0)
                acc += av[j] * (xv[k] - xv[j]);
            Kokkos::atomic_add(&y(idx(nb[k][0], nb[k][1], nb[k][2], extY, gY)), (av[k] / D) * acc);
          }
          return;
        }
        const double phibar = num / D;
        for (int k = 0; k < 6; ++k) {
          if (av[k] <= 0.0)
            continue;
          Kokkos::atomic_add(&y(idx(nb[k][0], nb[k][1], nb[k][2], extY, gY)),
                             av[k] * (xv[k] - phibar));
        }
      });
}

/// Fix the face correction at fluid|solid faces: projectCorrect applied -(phi_hi - phi_lo) with
/// the solid side's (decoupled) phi = 0; the mode-B correction wants the solid value phibar_s.
/// Difference per face: +phibar_s when s is the LOW cell, -phibar_s when s is the HIGH cell.
/// Each such face has exactly one solid side (both-solid faces carry no constraint), so no
/// atomics. phi is read on the (extP, gP) block; the face fields live on the (ext, g) block with
/// the o(i)-is-low-face convention (face k=+side of s == low face of the + neighbour).
inline void starCorrectFaces(CCField uf, CCField vf, CCField wf, CCConst phi,
                             const StarOverlay& ov, int nOv, C3 nn, C3 ext, int g, C3 extP,
                             int gP) {
  if (nOv <= 0)
    return;
  CCExec space;
  const bool exact = exactResidual();
  Kokkos::parallel_for(
      "peclet::flow::star_correct_faces", Kokkos::RangePolicy<CCExec>(space, 0, nOv),
      KOKKOS_LAMBDA(int s) {
        const int inner = ov.cell(s);
        const int ix = inner % nn.x, iy = (inner / nn.x) % nn.y, iz = inner / (nn.x * nn.y);
        auto idxP = [&](int cx, int cy, int cz) {
          return (long)(starWrap(cx, nn.x) + gP) + (long)(starWrap(cy, nn.y) + gP) * extP.x +
                 (long)(starWrap(cz, nn.z) + gP) * (long)extP.x * extP.y;
        };
        auto idxF = [&](int cx, int cy, int cz) {
          return (long)(starWrap(cx, nn.x) + g) + (long)(starWrap(cy, nn.y) + g) * ext.x +
                 (long)(starWrap(cz, nn.z) + g) * (long)ext.x * ext.y;
        };
        const int nb[6][3] = {{ix + 1, iy, iz}, {ix - 1, iy, iz}, {ix, iy + 1, iz},
                              {ix, iy - 1, iz}, {ix, iy, iz + 1}, {ix, iy, iz - 1}};
        double D = 0.0, num = 0.0;
        for (int k = 0; k < 6; ++k) {
          const double a = starAval(ov, (long)s * 6 + k, exact);
          if (a <= 0.0)
            continue;
          D += a;
          num += a * phi(idxP(nb[k][0], nb[k][1], nb[k][2]));
        }
        // phibar itself IS the wanted quantity here (the eliminated cell's reconstructed value),
        // not an operator row, so there is no difference form to take -- only the aperture
        // precision matters, and it must match the operator that was actually solved.
        const double phibar = num / D;
        CCField fa[3] = {uf, vf, wf};
        for (int k = 0; k < 6; ++k) {
          if (starAval(ov, (long)s * 6 + k, exact) <= 0.0)
            continue;
          const int a2 = k / 2;
          if ((k & 1) == 0) {
            // + side: face is the LOW face of the + neighbour; s is the LOW cell -> += phibar
            fa[a2](idxF(nb[k][0], nb[k][1], nb[k][2])) += phibar;
          } else {
            // - side: face is s's own LOW face; s is the HIGH cell -> -= phibar
            fa[a2](idxF(ix, iy, iz)) -= phibar;
          }
        }
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_STAR_ELIMINATION_HPP
