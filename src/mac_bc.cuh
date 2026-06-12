// cfd-gpu -- native domain boundary conditions for the MAC staggered velocity + cut-cell pressure.
//
// Periodic faces are handled by the halo (it wraps). NON-periodic faces are left unfilled by the halo
// (transport-core grid_halo.hpp:78 "BC fills these"); these kernels fill them:
//   * bc_velocity_comp_k  -- velocity ghost cells (Dirichlet / no-slip / moving wall), MAC staggered.
//   * bc_zero_openness_k   -- zero the boundary-face openness on a wall face -> the cut-cell pressure
//                            operator gives homogeneous Neumann there for free (alpha_f = 0).
//
// MAC convention (matches mac_cutcell.cuh / set_ibm_solid offsets): component a is stored at the -a face
// of its cell (offset -0.5 along axis a). So for axis a, the a-component is NORMAL to an a-face and the
// other two are TANGENTIAL. Domain a-boundary face position is the -a face of the first inner cell (low,
// along-index g) or the +a face of the last inner cell (high, along-index na-g).
//   Normal component:     boundary-face value = wall_a ; ghosts = odd reflection about the boundary.
//   Tangential component: ghosts = 2*wall - (reflected inner) so the wall value interpolates correctly.
// Ghost width is g (2 here); both ghost layers are filled.
#pragma once

#include <cuda_runtime.h>

namespace cfdmpi {
namespace bcdetail {

// Fill component `comp` (0=u,1=v,2=w) ghosts for ONE domain face: axis a in {0,1,2}, side s (0=low/-,
// 1=high/+). wall_comp = the comp-component of the prescribed wall velocity. Launched over the perp plane.
//   fold=0 (EXPLICIT, e.g. advection): tangential ghost = reflection (2*wall - inner); normal: boundary
//          face + odd-reflected ghosts. The ghost carries the wall value for explicit stencils.
//   fold=1 (IMPLICIT diffusion):       wall ghosts = 0 (the face is DROPPED); only the normal boundary
//          face is set. The dropped wall face's beta is moved to the diagonal by bc_diffusion_fold_k, so
//          the implicit solve never reads a wall ghost (no one-sweep Gauss-Seidel lag).
__global__ void bc_velocity_comp_k(double* f, int3 ext, int g, int a, int s, int comp, double wall_comp,
                                   int fold) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;  // the two perpendicular axes
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  size_t base = (size_t)p0 * strides[b] + (size_t)p1 * strides[c];
  size_t sa = strides[a];
  int na = dims[a];
  int bf = (s == 0) ? g : (na - g);  // along-axis index of the normal boundary face
  auto at = [&](int ia) -> double& { return f[base + (size_t)ia * sa]; };
  if (comp == a) {                   // NORMAL component: direct Dirichlet face (never lags) -- fold-agnostic
    at(bf) = wall_comp;              // boundary-face Dirichlet value
    if (s == 0)
      for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall_comp - at(2 * bf - ia);       // odd reflection
    else
      for (int ia = na - g + 1; ia < na; ++ia) at(ia) = 2.0 * wall_comp - at(2 * bf - ia);
  } else if (fold) {                 // TANGENTIAL, implicit: drop the wall face (ghost -> 0). Its beta is
    if (s == 0)                      //   moved to the diagonal + RHS by bc_diffusion_fold_k -> u_inner
      for (int ia = 0; ia < g; ++ia) at(ia) = 0.0;                                      //   stays implicit
    else
      for (int ia = na - g; ia < na; ++ia) at(ia) = 0.0;
  } else {                           // TANGENTIAL, explicit: reflection ghost (cell-centred; about bf-0.5)
    if (s == 0)
      for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall_comp - at(2 * bf - 1 - ia);
    else
      for (int ia = na - g; ia < na; ++ia) at(ia) = 2.0 * wall_comp - at(2 * bf - 1 - ia);
  }
}

// Build the implicit-diffusion face-fold for ONE tangential wall face (axis a, side s; comp != a): at the
// wall-adjacent inner cell, add beta to the diagonal correction `dcorr` (the dropped face's coefficient)
// and 2*beta*wall_comp to the RHS fold `brhs` (the Dirichlet contribution). Accumulates (corners get
// contributions from each wall). Launched over the perp plane.
__global__ void bc_diffusion_fold_k(double* dcorr, double* brhs, int3 ext, int g, int a, int s,
                                    double beta, double wall_comp) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  int bic = (s == 0) ? g : (dims[a] - g - 1);  // wall-adjacent inner cell along a
  size_t i = (size_t)p0 * strides[b] + (size_t)p1 * strides[c] + (size_t)bic * strides[a];
  dcorr[i] += beta;
  brhs[i] += 2.0 * beta * wall_comp;
}

// Zero the a-component face openness on a wall face (axis a, side s) -> homogeneous Neumann pressure.
// The a-face openness o_a[i] is the -a face of cell i; the domain boundary face is along-index g (low) or
// na-g (high). Launched over the perp plane.
__global__ void bc_zero_openness_k(double* oa, int3 ext, int g, int a, int s) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  int na = dims[a];
  int bf = (s == 0) ? g : (na - g);
  oa[(size_t)p0 * strides[b] + (size_t)p1 * strides[c] + (size_t)bf * strides[a]] = 0.0;
}

}  // namespace bcdetail
}  // namespace cfdmpi
