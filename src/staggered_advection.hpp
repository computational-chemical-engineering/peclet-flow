/// @file
/// @brief flow — portable (Kokkos) staggered MAC momentum advection (Koren TVD + FOU).
///
/// Kokkos port of staggered_advection.cuh: the Koren-limited conservative advection operator and
/// its first-order-upwind variant, on the local extended block (direct strides; ghosts
/// halo-exchanged). The math is a faithful copy (double precision, conservative flux form, +/-2
/// cell reach). A View accessor replaces the CUDA LocAcc. The functions are KOKKOS_INLINE_FUNCTION
/// so the same code runs on device and is reused by the host reference in the test.
#ifndef PECLET_FLOW_STAGGERED_ADVECTION_HPP
#define PECLET_FLOW_STAGGERED_ADVECTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace sadv {

KOKKOS_INLINE_FUNCTION double koren(double up_m1, double up, double down, double vel) {
  const double num = up - up_m1, den = down - up;
  double r = (Kokkos::fabs(den) < 1e-10) ? 0.0 : num / den;
  if (Kokkos::fabs(den) < 1e-10 && Kokkos::fabs(num) < 1e-10)
    r = 1.0;
  const double psi =
      Kokkos::fmax(0.0, Kokkos::fmin(2.0 * r, Kokkos::fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
  return vel * (up + 0.5 * psi * (down - up));
}
KOKKOS_INLINE_FUNCTION double tvd(double LL, double L, double R, double RR, double vel) {
  return (vel > 0.0) ? koren(LL, L, R, vel) : koren(RR, R, L, vel);
}
KOKKOS_INLINE_FUNCTION double fou_flux(double L, double R, double vel) {
  return vel * (vel > 0.0 ? L : R);
}
// Second-order upwind (SOU) face flux: linear extrapolation from the two upwind
// cells (unlimited; 2nd-order everywhere, including smooth extrema where the TVD
// limiter clips to 1st order). vel>0 uses LL,L; vel<0 uses RR,R.
KOKKOS_INLINE_FUNCTION double sou(double LL, double L, double R, double RR, double vel) {
  return (vel > 0.0) ? vel * (1.5 * L - 0.5 * LL) : vel * (1.5 * R - 0.5 * RR);
}

// Local extended-block accessor over a Kokkos View (direct strides, x-fastest; no wrap).
struct ViewAcc {
  Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space> d;
  int ex, ey;
  KOKKOS_INLINE_FUNCTION double operator()(int x, int y, int z) const {
    return d(static_cast<long>(x) + static_cast<long>(y) * ex +
             static_cast<long>(z) * static_cast<long>(ex) * ey);
  }
};

// Advecting velocity at the +fd face of the comp control volume at (x,y,z).
template <class A>
KOKKOS_INLINE_FUNCTION double adv_vel(int comp, int fd, int x, int y, int z, A U, A V, A W) {
  if (comp == 0) {
    if (fd == 0)
      return 0.5 * (U(x, y, z) + U(x + 1, y, z));
    if (fd == 1)
      return 0.5 * (V(x - 1, y + 1, z) + V(x, y + 1, z));
    return 0.5 * (W(x - 1, y, z + 1) + W(x, y, z + 1));
  }
  if (comp == 1) {
    if (fd == 0)
      return 0.5 * (U(x + 1, y - 1, z) + U(x + 1, y, z));
    if (fd == 1)
      return 0.5 * (V(x, y, z) + V(x, y + 1, z));
    return 0.5 * (W(x, y - 1, z + 1) + W(x, y, z + 1));
  }
  if (fd == 0)
    return 0.5 * (U(x + 1, y, z - 1) + U(x + 1, y, z));
  if (fd == 1)
    return 0.5 * (V(x, y + 1, z - 1) + V(x, y + 1, z));
  return 0.5 * (W(x, y, z) + W(x, y, z + 1));
}

// Conservative Koren-TVD advection A = sum_dir (F_plus - F_minus) of component comp; PHI is its
// field.
template <class A>
KOKKOS_INLINE_FUNCTION double advect(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    const double Fp = tvd(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                          PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    const double Fm = tvd(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                          PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// Conservative second-order-upwind advection (same control volume / advecting
// velocities as advect(); SOU flux instead of the Koren limiter).
template <class A>
KOKKOS_INLINE_FUNCTION double advect_sou(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    const double Fp = sou(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                          PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    const double Fm = sou(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                          PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// FOU advection OPERATOR coefficients added to a cell's 7-point stencil (consistent with advect_fou
// applied to the field): diagonal cC gets max(velp,0)-min(velm,0) >= 0, off-diagonals <= 0. Added
// (not assigned) into the out-params. Port of fou_operator (staggered_advection.cuh).
template <class A>
KOKKOS_INLINE_FUNCTION void fou_operator(int comp, int x, int y, int z, A U, A V, A W, double dt,
                                         double& cC, double& cxm, double& cxp, double& cym,
                                         double& cyp, double& czm, double& czp) {
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
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

// Anisotropic (per-axis inverse spacing) FOU operator for the velocity multigrid coarse levels:
// the advecting velocity along face-axis fd is scaled by s_fd = 1/h_fd. sx=sy=sz=1 == fou_operator.
template <class A>
KOKKOS_INLINE_FUNCTION void fou_operator_aniso(int comp, int x, int y, int z, A U, A V, A W,
                                               double dt, double sx, double sy, double sz,
                                               double& cC, double& cxm, double& cxp, double& cym,
                                               double& cyp, double& czm, double& czp) {
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double s = (fd == 0) ? sx : (fd == 1) ? sy : sz;
    const double velp = s * adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = s * adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
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
KOKKOS_INLINE_FUNCTION double advect_fou(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    const int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    const double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    const double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    out += fou_flux(PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velp) -
           fou_flux(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), velm);
  }
  return out;
}

}  // namespace sadv

#endif  // PECLET_FLOW_STAGGERED_ADVECTION_HPP
