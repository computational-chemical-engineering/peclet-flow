/// @file
/// @brief flow — VoF rung V2a (WO-J): the bridge between the solver's `G = 2` block and the colour
/// field's own `g = 3` working block, plus the colour field's ghost policy.
///
/// `suite/docs/VOF_PLAN.md` §3 design rule 1 keeps the solver's global `G = 2` untouched and gives
/// the colour field its own, wider halo. That leaves exactly two block extents in play and every
/// bridge between them is a place to get an offset wrong — the functions here are that bridge, kept
/// in one file so there is one place to read.
///
/// Two kinds of transfer are needed and they are NOT the same:
/// - **inner-region only** (the colour field itself). The two blocks disagree about how many ghost
///   layers they have but agree about the inner cells; each block's ghosts are then filled by that
///   block's OWN policy. `IbmSolver::copyInner` already does this both ways, so nothing here.
/// - **whole source block embedded in the destination** (the face velocities). The advector reads
///   the face velocity one cell OUTSIDE its inner region (the `-d` face of the first inner cell is
///   a ghost cell's `+d` face), and the solver's velocity block carries exactly those values in its
///   own ghost ring once `fillVelGhosts` has run. `copyFaceVelocity` places the whole `G = 2` block
///   (inner + both ghost layers) inside the `g = 3` block, so the advector's ghost layers hold real
///   face velocities and only the outermost layer — which no kernel reads — is left alone.
///
/// **The two codes index a staggered face differently and the shift is load-bearing.** In `flow`,
/// `u(i)` is the **low** (`-x`) face of cell `i`: `projectCorrect` writes
/// `u(i) -= phi(i) - phi(i-sx)` (the gradient between cells `i-1` and `i`) and `divergOpen` forms
/// `d(i) = ox(i+sx)*u(i+sx) - ox(i)*u(i)`. In `WyAdvector`, `uf(i)` is the **high** (`+x`) face of
/// cell `i`: `wyFaceFlux(u(p), p, sd, …)` fluxes from `p` into `p+sd`, and the dilation term is
/// `u(i) - u(i-sd)`. So the bridge must shift by one cell ALONG the component's own axis:
///
///     u_advector(i) = u_solver(i + s_d)
///
/// Getting this wrong is invisible in a uniform flow and invisible in the discrete divergence of
/// EACH AXIS SEPARATELY (the shifted x-difference is still the solver's exact zero, just evaluated
/// at cell `i-1`) — but the advector sums the three axes AT ONE CELL, and the unshifted bridge
/// hands it `div_x(i-1) + div_y(i-s_y) + div_z(i-s_z)`, three different cells whose sum the
/// projection never constrained. Weymouth-Yue then adds `H(C-½)` times that non-zero "divergence"
/// to every full cell's budget: MEASURED with the shift omitted, a sheared ratio-10 sphere over
/// 1000 coupled steps gained **35 % of its volume** while the solver's own `max|div(open·u)|` read
/// 7e-11. That is the failure mode to remember — the conservation gate catches it and nothing else
/// does.
///
/// The ghost fills are the production twins of the ones the V1 test harness uses
/// (`tests/kokkos/vof_advect_scenes.hpp`), including WO-E finding 2: the non-periodic fill must
/// take the value at the **globally clamped** index, never sequential zero-gradient axis passes
/// over the block, or the fill is decomposition-DEPENDENT and the difference reaches the colour
/// field through the MYC stencil of the inner corner cell.
#ifndef PECLET_FLOW_VOF_COLOUR_FIELD_HPP
#define PECLET_FLOW_VOF_COLOUR_FIELD_HPP

#include <Kokkos_Core.hpp>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3

namespace peclet::flow::vof {

/// Embed the ENTIRE source extended block (inner + its `sg` ghost layers) into a LARGER destination
/// extended block, converting the staggered face index of velocity component `dir` from the
/// solver's low-face convention to the advector's high-face convention:
///
///     dst(x + s - [dir==0], y + s - [dir==1], z + s - [dir==2]) = src(x, y, z),   s = dg - sg
///
/// i.e. a plain concentric embed on the two transverse axes and one cell less on the component's
/// own axis (`u_adv(i) = u_solver(i + s_dir)` — see the file header; omitting this shift is a
/// silent, 35 %-scale conservation defect). Requires equal inner extents and `dg > sg`, which the
/// only caller satisfies (`g = 3` colour block from the `G = 2` velocity block).
inline void copyFaceVelocity(SField dst, I3 de, int dg, SField src, I3 se, int sg, int dir) {
  const int s = dg - sg;
  const int ox = s - (dir == 0 ? 1 : 0), oy = s - (dir == 1 ? 1 : 0), oz = s - (dir == 2 ? 1 : 0);
  Kokkos::parallel_for(
      "vof::copyFaceVelocity",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {se.x, se.y, se.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        dst(L3(x + ox, y + oy, z + oz, de)) = src(L3(x, y, z, se));
      });
}

/// The INVERSE of `copyFaceVelocity`, restricted to the inner region: bring a face-indexed field
/// back from the advector's `g = 3` block to the solver's `G = 2` block, undoing the high-face ->
/// low-face shift.
///
///     dst(x + dg, y + dg, z + dg) = src(x + sg - [dir==0], y + sg - [dir==1], z + sg - [dir==2])
///
/// for `x,y,z` over the inner region `n`. Used at rung V2b (WO-K) to hand the momentum-consistent
/// advected velocity back to the momentum RHS. The source index runs one cell BELOW the advector's
/// inner region along `dir` — that is exactly the momentum control volume the solver's first
/// velocity unknown on that axis belongs to, and `MomentumConsistentAdvector::cvRange` computes
/// there for this reason (the two index sets are in one-to-one correspondence, so nothing is
/// dropped or duplicated across a rank boundary).
inline void copyAdvectedVelocity(SField dst, I3 de, int dg, SField src, I3 se, int sg, int dir,
                                 I3 n) {
  const int ox = sg - (dir == 0 ? 1 : 0), oy = sg - (dir == 1 ? 1 : 0), oz = sg - (dir == 2 ? 1 : 0);
  Kokkos::parallel_for(
      "vof::copyAdvectedVelocity",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {n.x, n.y, n.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        dst(L3(x + dg, y + dg, z + dg, de)) = src(L3(x + ox, y + oy, z + oz, se));
      });
}

/// Periodic wrap of a single-block extended field at ghost width `g` (the single-rank ghost path;
/// under MPI `core::halo::GridHalo` does this and the cross-rank exchange in one call). Sequential
/// axis passes, so edges and corners come out right.
inline void periodicFill(SField f, I3 e, int g, bool px, bool py, bool pz) {
  const long st[3] = {1, e.x, static_cast<long>(e.x) * e.y};
  const int dims[3] = {e.x, e.y, e.z};
  const bool per[3] = {px, py, pz};
  for (int a = 0; a < 3; ++a) {
    if (!per[a])
      continue;
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    const int N = dims[a] - 2 * g;
    Kokkos::parallel_for(
        "vof::periodicFill",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(SExec(), {0, 0}, {dims[b], dims[c]}),
        KOKKOS_LAMBDA(int p0, int p1) {
          const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
          for (int gl = 0; gl < g; ++gl) {
            f(base + static_cast<long>(gl) * sa) = f(base + static_cast<long>(gl + N) * sa);
            f(base + static_cast<long>(g + N + gl) * sa) = f(base + static_cast<long>(g + gl) * sa);
          }
        });
    Kokkos::fence();
  }
}

/// Zero-gradient (Neumann) fill of the ghost cells that lie OUTSIDE the global domain on a
/// non-periodic axis: each such cell takes the value at its globally CLAMPED index.
///
/// `o` is the GLOBAL index of this block's inner cell (0,0,0) and `gs` the global grid size. The
/// clamp is per axis and global, which is what makes it decomposition-independent (WO-E finding 2:
/// sequential zero-gradient axis passes over the block are NOT — a corner cell outside the domain
/// in y whose x-neighbourhood is an interior halo gets a different value on a different
/// decomposition, and that difference reaches C through the MYC stencil of the inner corner cell).
/// The clamped source is provably inside the block's extended range and is either an inner cell or
/// a ghost the exchange has already filled, so call this AFTER the halo exchange / periodic fill.
inline void clampFill(SField f, I3 e, int g, I3 o, I3 gs, bool px, bool py, bool pz) {
  if (px && py && pz)
    return;
  const bool p0 = px, p1 = py, p2 = pz;
  const int q0 = gs.x, q1 = gs.y, q2 = gs.z;
  Kokkos::parallel_for(
      "vof::clampFill",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const int gx = x - g + o.x, gy = y - g + o.y, gz = z - g + o.z;
        const int cx = p0 ? gx : (gx < 0 ? 0 : (gx >= q0 ? q0 - 1 : gx));
        const int cy = p1 ? gy : (gy < 0 ? 0 : (gy >= q1 ? q1 - 1 : gy));
        const int cz = p2 ? gz : (gz < 0 ? 0 : (gz >= q2 ? q2 - 1 : gz));
        if (cx != gx || cy != gy || cz != gz)
          f(L3(x, y, z, e)) = f(L3(cx - o.x + g, cy - o.y + g, cz - o.z + g, e));
      });
  Kokkos::fence();
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_COLOUR_FIELD_HPP
