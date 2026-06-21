/// @file
/// @brief sdflow — collocated (cell-centered) momentum advection (Koren TVD + FOU).
///
/// The collocated counterpart of staggered_advection.hpp. All three velocity components live at the cell
/// center, so a single cell is the control volume for every component and the advecting normal velocity at
/// a face is the cell->face average of that face's normal component — independent of which component is
/// being advected (contrast sadv::adv_vel, which interpolates onto a staggered control volume). The flux
/// reconstruction (Koren limiter / FOU) is otherwise identical, so this header reuses sadv::koren / tvd /
/// fou_flux / ViewAcc and only re-expresses adv_vel + the conservative operators.
///
/// NOTE: in phase 2 (no pressure) the advecting velocity is the plain cell->face average of the stored
/// cell velocities. Once the approximate projection lands (phase 3), the natural advecting field is the
/// projected, divergence-free face velocity — this header's adv_vel is where that swap happens.
#ifndef CFD_COLOCATED_ADVECTION_HPP
#define CFD_COLOCATED_ADVECTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "staggered_advection.hpp"  // sadv::koren / tvd / fou_flux / ViewAcc (reused verbatim)

namespace cadv {

// Advecting (normal) velocity at the +fd face of the cell at (x,y,z): the cell->face average of the
// fd-component. Independent of the advected component `comp` (all components are co-located at the center).
template <class A>
KOKKOS_INLINE_FUNCTION double adv_vel(int /*comp*/, int fd, int x, int y, int z, A U, A V, A W) {
  if (fd == 0) return 0.5 * (U(x, y, z) + U(x + 1, y, z));
  if (fd == 1) return 0.5 * (V(x, y, z) + V(x, y + 1, z));
  return 0.5 * (W(x, y, z) + W(x, y, z + 1));
}

// Conservative Koren-TVD advection A = sum_dir (F_plus - F_minus) of component comp; PHI is its field.
template <class A>
KOKKOS_INLINE_FUNCTION double advect(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    const double Fp = sadv::tvd(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                                PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    const double Fm = sadv::tvd(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                                PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// FOU advection OPERATOR coefficients added to a cell's 7-point stencil (consistent with advect_fou):
// diagonal cC gets max(velp,0)-min(velm,0) >= 0, off-diagonals <= 0. Added (not assigned) into out-params.
template <class A>
KOKKOS_INLINE_FUNCTION void fou_operator(int comp, int x, int y, int z, A U, A V, A W, double dt,
                                         double& cC, double& cxm, double& cxp, double& cym, double& cyp,
                                         double& czm, double& czp) {
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    cC += dt * (Kokkos::fmax(velp, 0.0) - Kokkos::fmin(velm, 0.0));
    const double cp = dt * Kokkos::fmin(velp, 0.0), cm = dt * (-Kokkos::fmax(velm, 0.0));
    if (fd == 0) { cxp += cp; cxm += cm; }
    else if (fd == 1) { cyp += cp; cym += cm; }
    else { czp += cp; czm += cm; }
  }
}

// Conservative first-order-upwind advection of comp (low-order flux, same advecting velocities).
template <class A>
KOKKOS_INLINE_FUNCTION double advect_fou(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = cadv::adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = cadv::adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    out += sadv::fou_flux(PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velp) -
           sadv::fou_flux(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), velm);
  }
  return out;
}

}  // namespace cadv

#endif  // CFD_COLOCATED_ADVECTION_HPP
