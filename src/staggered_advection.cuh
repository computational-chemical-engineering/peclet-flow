/// @file
/// @brief Staggered MAC momentum advection (Koren TVD), shared by the solver and tests.
// cfd-gpu — staggered MAC momentum advection (Koren TVD), shared by the distributed solver and tests.
//
// Replicates cfd_solver.cu's get_advection_velocity (2-point staggered interpolation of the advecting
// velocity) and get_tvd_flux / tvd_flux_koren (Koren limiter, sign-upwinded), in double precision and
// in conservative flux form A = sum_dir (F_plus - F_minus). The operator reaches +/-2 cells, so any
// distributed use needs ghost width 2. A field accessor templates the index mapping so the same code
// serves the full periodic grid (wrapping get_idx) and a local extended block (direct strides).
#pragma once

#include "cfd_solver.cuh"  // get_idx

namespace sadv {

__device__ inline double koren(double up_m1, double up, double down, double vel) {
  double num = up - up_m1, den = down - up;
  double r = (fabs(den) < 1e-10) ? 0.0 : num / den;
  if (fabs(den) < 1e-10 && fabs(num) < 1e-10) r = 1.0;
  double psi = fmax(0.0, fmin(2.0 * r, fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
  return vel * (up + 0.5 * psi * (down - up));
}
__device__ inline double tvd(double LL, double L, double R, double RR, double vel) {
  return (vel > 0.0) ? koren(LL, L, R, vel) : koren(RR, R, L, vel);
}

// Full periodic grid accessor (wraps via get_idx).
struct FullAcc {
  const double* d;
  int3 res;
  __device__ double operator()(int x, int y, int z) const { return d[get_idx(x, y, z, res)]; }
};
// Local extended-block accessor (direct, no wrap — ghosts are halo-exchanged).
struct LocAcc {
  const double* d;
  int3 e;
  __device__ double operator()(int x, int y, int z) const {
    return d[(long)x + (long)y * e.x + (long)z * (long)e.x * e.y];
  }
};

// Advecting velocity at the +face_dir face of the comp control volume at (x,y,z); mirrors
// get_advection_velocity in cfd_solver.cu exactly.
template <class A>
__device__ inline double adv_vel(int comp, int fd, int x, int y, int z, A U, A V, A W) {
  if (comp == 0) {
    if (fd == 0) return 0.5 * (U(x, y, z) + U(x + 1, y, z));
    if (fd == 1) return 0.5 * (V(x - 1, y + 1, z) + V(x, y + 1, z));
    return 0.5 * (W(x - 1, y, z + 1) + W(x, y, z + 1));
  }
  if (comp == 1) {
    if (fd == 0) return 0.5 * (U(x + 1, y - 1, z) + U(x + 1, y, z));
    if (fd == 1) return 0.5 * (V(x, y, z) + V(x, y + 1, z));
    return 0.5 * (W(x, y - 1, z + 1) + W(x, y, z + 1));
  }
  if (fd == 0) return 0.5 * (U(x + 1, y, z - 1) + U(x + 1, y, z));
  if (fd == 1) return 0.5 * (V(x, y + 1, z - 1) + V(x, y + 1, z));
  return 0.5 * (W(x, y, z) + W(x, y, z + 1));
}

// Conservative advection A = sum_dir (F_plus - F_minus) of component `comp`; PHI is its field.
template <class A>
__device__ inline double advect(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    double Fp = tvd(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                    PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    double Fm = tvd(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                    PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

// ---- first-order upwind (FOU) variant, for the implicit-FOU deferred correction ----
// FOU face flux: vel * (upwind value across the face L|R).
__device__ inline double fou_flux(double L, double R, double vel) { return vel * (vel > 0.0 ? L : R); }

// Conservative FOU advection of `comp` (same advecting velocities as advect(), low-order flux).
template <class A>
__device__ inline double advect_fou(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    out += fou_flux(PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velp) -
           fou_flux(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), velm);
  }
  return out;
}

// The FOU advection OPERATOR coefficients added to a cell's 7-point stencil (consistent with
// advect_fou applied to the field): diagonal gets max(velp,0)-min(velm,0) >= 0 (diagonal dominance),
// off-diagonals are <= 0. Returns via the out-params (added in, not assigned). cmp is the component.
template <class A>
__device__ inline void fou_operator(int comp, int x, int y, int z, A U, A V, A W, double dt, double& cC,
                                    double& cxm, double& cxp, double& cym, double& cyp, double& czm,
                                    double& czp) {
  for (int fd = 0; fd < 3; ++fd) {
    int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    cC += dt * (fmax(velp, 0.0) - fmin(velm, 0.0));
    double cp = dt * fmin(velp, 0.0), cm = dt * (-fmax(velm, 0.0));
    if (fd == 0) { cxp += cp; cxm += cm; }
    else if (fd == 1) { cyp += cp; cym += cm; }
    else { czp += cp; czm += cm; }
  }
}

// Coarse-level FOU operator for the velocity multigrid. Identical donor-cell upwinding to fou_operator,
// but with a PER-AXIS inverse advective spacing s = (sx,sy,sz) = (1/h_x, 1/h_y, 1/h_z): the conservative
// FOU coefficient is dt*vel/h_a, and fou_operator assumes h=1, so the advecting velocity along face-axis
// `fd` is scaled by s_fd. On the coarse level h_a = h0*cfac_a, hence s_a = 1/(h0*cfac_a). sx=sy=sz=1
// reproduces fou_operator exactly. Upwinding keeps the diagonal dominant -> M-matrix (RB-GS stable).
template <class A>
__device__ inline void fou_operator_aniso(int comp, int x, int y, int z, A U, A V, A W, double dt,
                                          double sx, double sy, double sz, double& cC, double& cxm,
                                          double& cxp, double& cym, double& cyp, double& czm,
                                          double& czp) {
  for (int fd = 0; fd < 3; ++fd) {
    int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    double s = (fd == 0) ? sx : (fd == 1) ? sy : sz;
    double velp = s * adv_vel(comp, fd, x, y, z, U, V, W);
    double velm = s * adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    cC += dt * (fmax(velp, 0.0) - fmin(velm, 0.0));
    double cp = dt * fmin(velp, 0.0), cm = dt * (-fmax(velm, 0.0));
    if (fd == 0) { cxp += cp; cxm += cm; }
    else if (fd == 1) { cyp += cp; cym += cm; }
    else { czp += cp; czm += cm; }
  }
}

}  // namespace sadv
