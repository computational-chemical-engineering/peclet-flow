/// @file
/// @brief flow — collocated (cell-centered) momentum advection (Koren TVD + FOU).
///
/// The collocated counterpart of staggered_advection.hpp. All three velocity components live at the
/// cell center, so a single cell is the control volume for every component and the advecting normal
/// velocity at a face is the cell->face average of that face's normal component — independent of
/// which component is being advected (contrast sadv::adv_vel, which interpolates onto a staggered
/// control volume). The flux reconstruction (Koren limiter / FOU) is otherwise identical, so this
/// header reuses sadv::koren / tvd / fou_flux / ViewAcc and only re-expresses adv_vel + the
/// conservative operators.
///
/// ADVECTING FIELD (`uf` argument of every operator below, plumbed from
/// `Solver::ufAdvVelocity()`): the natural advecting field is the PROJECTED, discretely
/// divergence-free MAC face velocity `uf_/vf_/wf_` that the approximate projection produced last
/// step — the field the projection just made solenoidal IS the conservative advective flux
/// (Almgren–Bell–Colella; doc/flow_colocated_plan.md §1 step 3). That is the production choice
/// (`uf == true`). The phase-2 form, the plain cell->face average 1/2(U(i)+U(i+1)) of the stored
/// cell velocities, survives as `uf == false`: it is what the solver uses before the first
/// projection has built a face field, and the developer-tier ablation
/// `diagnostics.set_uf_advection(False)` (see doc/uf_advection.md).
#ifndef PECLET_FLOW_COLOCATED_ADVECTION_HPP
#define PECLET_FLOW_COLOCATED_ADVECTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "staggered_advection.hpp"  // sadv::koren / tvd / fou_flux / ViewAcc (reused verbatim)

namespace cadv {

// Advecting (normal) velocity at the +fd face of the cell at (x,y,z). Independent of the advected
// component `comp` (all three components are co-located at the center, so one cell is the control
// volume for all of them and one face flux serves all three).
//
// `uf` selects what U/V/W ARE:
//   true  — the projected MAC face field `uf_/vf_/wf_` in flow's LOW-face convention (U(i) is the
//           velocity at the -x face of cell i, i.e. at i-1/2), so the +fd face of cell (x,y,z) is
//           the low face of its +fd neighbour: index (x+1,y,z) for fd=0. Read verbatim — this is
//           the field the projection made solenoidal, so in the ALL-FLUID bulk (every face
//           openness 1) the advective flux sum telescopes to zero exactly. At a CUT cell it does
//           not: the projection zeroes the openness-WEIGHTED divergence
//           sum(o_f u_f^+ - o_f u_f^-) (divergOpen) while the flux form here is unweighted, so
//           the band carries a residual — small, and far smaller than the O(h^2) cell divergence
//           the average carries everywhere. Same trade amr records (docs/decisions/amr.md:13:
//           "div-free bulk, residual-small band").
//   false — the cell-centered velocities; the face value is their average 1/2(U(i)+U(i+1)).
template <class A>
KOKKOS_INLINE_FUNCTION double adv_vel(int /*comp*/, int fd, int x, int y, int z, A U, A V, A W,
                                      bool uf) {
  if (fd == 0)
    return uf ? U(x + 1, y, z) : 0.5 * (U(x, y, z) + U(x + 1, y, z));
  if (fd == 1)
    return uf ? V(x, y + 1, z) : 0.5 * (V(x, y, z) + V(x, y + 1, z));
  return uf ? W(x, y, z + 1) : 0.5 * (W(x, y, z) + W(x, y, z + 1));
}

// Conservative Koren-TVD advection A = sum_dir (F_plus - F_minus) of component comp; PHI is its
// field.
template <class A>
KOKKOS_INLINE_FUNCTION double advect(int comp, int x, int y, int z, A U, A V, A W, A PHI, bool uf) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W, uf);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W, uf);
    const double Fp =
        sadv::tvd(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                  PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    const double Fm =
        sadv::tvd(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                  PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// Conservative second-order-upwind advection (SOU flux; same advecting velocities as
// cadv::advect).
template <class A>
KOKKOS_INLINE_FUNCTION double advect_sou(int comp, int x, int y, int z, A U, A V, A W, A PHI,
                                         bool uf) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W, uf);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W, uf);
    const double Fp =
        sadv::sou(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                  PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    const double Fm =
        sadv::sou(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                  PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// FOU advection OPERATOR coefficients added to a cell's 7-point stencil (consistent with
// advect_fou): diagonal cC gets max(velp,0)-min(velm,0) >= 0, off-diagonals <= 0. Added (not
// assigned) into out-params.
template <class A>
KOKKOS_INLINE_FUNCTION void fou_operator(int comp, int x, int y, int z, A U, A V, A W, double dt,
                                         double& cC, double& cxm, double& cxp, double& cym,
                                         double& cyp, double& czm, double& czp, bool uf) {
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W, uf);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W, uf);
    cC += dt * (Kokkos::fmax(velp, 0.0) - Kokkos::fmin(velm, 0.0));
    const double cp = dt * Kokkos::fmin(velp, 0.0), cm = dt * (-Kokkos::fmax(velm, 0.0));
    if (fd == 0) {
      cxp += cp;
      cxm += cm;
    } else if (fd == 1) {
      cyp += cp;
      cym += cm;
    } else {
      czp += cp;
      czm += cm;
    }
  }
}

// Conservative first-order-upwind advection of comp (low-order flux, same advecting velocities).
template <class A>
KOKKOS_INLINE_FUNCTION double advect_fou(int comp, int x, int y, int z, A U, A V, A W, A PHI,
                                         bool uf) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W, uf);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W, uf);
    out += sadv::fou_flux(PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velp) -
           sadv::fou_flux(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), velm);
  }
  return out;
}

}  // namespace cadv

#endif  // PECLET_FLOW_COLOCATED_ADVECTION_HPP
