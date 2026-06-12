// cfd-gpu -- native domain boundary conditions for the MAC staggered velocity + cut-cell pressure.
//
// Periodic faces are handled by the halo (it wraps). NON-periodic faces are left unfilled by the halo
// (transport-core grid_halo.hpp:78 "BC fills these"); these kernels fill them:
//   * bc_velocity_comp_k  -- velocity ghost cells (Dirichlet / no-slip / moving wall), MAC staggered.
//   * bc_outflow_comp_k   -- zero-gradient (Neumann) OUTFLOW velocity ghost (du/dn = 0).
//   * bc_zero_openness_k   -- zero the boundary-face openness on a wall face -> the cut-cell pressure
//                            operator gives homogeneous Neumann there for free (alpha_f = 0).
//   * bc_zero_pressure_ghost_k -- hold the pressure ghost at 0 on an OUTFLOW face -> Dirichlet p=0 (the
//                            face stays open in the operator; non-singular -> no mean removal).
//   * correct_outflow_k    -- projection correction of the high-side outflow normal face (mass exit).
//
// Open boundaries split the face openness into two roles: the OPERATOR openness (pressure matrix) is 0 at
// walls + inflow (Neumann) and open at outflow (Dirichlet); the FLUX openness (divergence/correction) is
// open at inflow + outflow so their flux is counted. See distributed_ns.cuh.
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
// prof (optional): a per-(b,c)-position INLET PROFILE laid out [p0][p1][3] with row stride prof_nc; when
// non-null the prescribed value is prof[(p0*prof_nc+p1)*3+comp] instead of the scalar wall_comp (e.g. a
// parabolic channel inlet / the backward-facing-step partial inlet). Tangential profile values feed the
// explicit path; the implicit tangential fold still uses the scalar wall (set it for a sheared inlet).
__global__ void bc_velocity_comp_k(double* f, int3 ext, int g, int a, int s, int comp, double wall_comp,
                                   int fold, const double* prof = nullptr, int prof_nc = 0) {
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
  double wc = prof ? prof[((size_t)p0 * prof_nc + p1) * 3 + comp] : wall_comp;
  auto at = [&](int ia) -> double& { return f[base + (size_t)ia * sa]; };
  if (comp == a) {                   // NORMAL component: direct Dirichlet face (never lags) -- fold-agnostic
    at(bf) = wc;                     // boundary-face Dirichlet value
    if (s == 0)
      for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wc - at(2 * bf - ia);       // odd reflection
    else
      for (int ia = na - g + 1; ia < na; ++ia) at(ia) = 2.0 * wc - at(2 * bf - ia);
  } else if (fold) {                 // TANGENTIAL, implicit: drop the wall face (ghost -> 0). Its beta is
    if (s == 0)                      //   moved to the diagonal + RHS by bc_diffusion_fold_k -> u_inner
      for (int ia = 0; ia < g; ++ia) at(ia) = 0.0;                              //   stays implicit
    else
      for (int ia = na - g; ia < na; ++ia) at(ia) = 0.0;
  } else {                           // TANGENTIAL, explicit: reflection ghost (cell-centred; about bf-0.5)
    if (s == 0)
      for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
    else
      for (int ia = na - g; ia < na; ++ia) at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
  }
}

// Build the implicit-diffusion face-fold for ONE boundary face (axis a, side s): at the boundary-adjacent
// inner cell, add `dval` to the diagonal correction `dcorr` (the dropped face's coefficient) and `bval` to
// the RHS fold `brhs`. Accumulates (corners get contributions from each face). Two uses:
//   Dirichlet wall (2*wall - u_inner ghost): dval = +beta, bval = +2*beta*wall  (tangential only).
//   Zero-gradient outflow (u_ghost = u_inner): dval = -beta, bval = 0           (every component).
// Launched over the perp plane.
__global__ void bc_diffusion_fold_k(double* dcorr, double* brhs, int3 ext, int g, int a, int s,
                                    double dval, double bval) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  int bic = (s == 0) ? g : (dims[a] - g - 1);  // boundary-adjacent inner cell along a
  size_t i = (size_t)p0 * strides[b] + (size_t)p1 * strides[c] + (size_t)bic * strides[a];
  dcorr[i] += dval;
  brhs[i] += bval;
}

// Zero-gradient (Neumann) OUTFLOW velocity ghost for component `comp` on ONE face (axis a, side s). The
// open boundary copies the last interior plane outward (du/dn = 0). fold=0 (explicit): ghosts = nearest
// interior value. fold=1 (implicit diffusion): ghosts = 0 (the face is dropped; its -beta is folded into
// the diagonal by bc_diffusion_fold_k, keeping u_inner implicit). The normal component (comp==a) also
// fills the boundary face itself (index g low / na-g high); tangential fills cell ghosts only.
__global__ void bc_outflow_comp_k(double* f, int3 ext, int g, int a, int s, int comp, int fold) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  size_t base = (size_t)p0 * strides[b] + (size_t)p1 * strides[c];
  size_t sa = strides[a];
  int na = dims[a];
  auto at = [&](int ia) -> double& { return f[base + (size_t)ia * sa]; };
  if (s == 0) {
    int src = (comp == a) ? g + 1 : g;   // nearest interior value
    int last = (comp == a) ? g : g - 1;  // furthest ghost index to fill (incl. normal boundary face g)
    double v = fold ? 0.0 : at(src);
    for (int ia = 0; ia <= last; ++ia) at(ia) = v;
  } else {
    int src = na - g - 1;                // nearest interior value (boundary face na-g for normal)
    double v = fold ? 0.0 : at(src);
    for (int ia = na - g; ia < na; ++ia) at(ia) = v;
  }
}

// Zero the pressure ghost plane on an OUTFLOW face (axis a, side s) -> Dirichlet p=0 there. The cut-cell
// operator keeps that face open (aperture != 0), so its smoother reads phi[ghost]; holding the ghost at 0
// (the smoother updates interior-only and the non-periodic halo skips it) imposes p=0. Low ghosts
// [0,g-1], high ghosts [na-g, na-1]. Launched over the perp plane.
__global__ void bc_zero_pressure_ghost_k(double* phi, int3 ext, int g, int a, int s) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  size_t base = (size_t)p0 * strides[b] + (size_t)p1 * strides[c];
  size_t sa = strides[a];
  int na = dims[a];
  int lo = (s == 0) ? 0 : (na - g), hi = (s == 0) ? (g - 1) : (na - 1);
  for (int ia = lo; ia <= hi; ++ia) phi[base + (size_t)ia * sa] = 0.0;
}

// Projection correction of the HIGH-side OUTFLOW normal-velocity face (the +a domain face at index na-g),
// which correct_k misses (it updates interior -a faces only, i.e. up to index na-g-1). With the Dirichlet
// ghost phi[na-g]=0: u_face -= (phi[na-g] - phi[na-g-1]) = +phi[na-g-1]. This lets the outflow respond to
// the pressure (the global mass-conservation mechanism). Low-side outflow is handled by correct_k itself.
__global__ void correct_outflow_k(double* f, const double* phi, int3 ext, int g, int a) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  size_t sa = strides[a];
  size_t bf = (size_t)p0 * strides[b] + (size_t)p1 * strides[c] + (size_t)(dims[a] - g) * sa;
  f[bf] -= phi[bf] - phi[bf - sa];
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
