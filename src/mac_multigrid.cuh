/// @file
/// @brief Distributed geometric multigrid pressure Poisson (rediscretized, BC-aware, semi-coarsening).
// cfd-gpu -- distributed geometric multigrid for the periodic constant-coefficient pressure Poisson.
//
// The pure-Neumann pressure solve (Lap phi = rhs, periodic, all-fluid) is the canonical inner solve of
// the projection. This builds a V-cycle on a hierarchy of MacGridHalo levels: level L is the global
// grid at res>>L. Because transport-core's ORB cuts power-of-two grids at midpoints, each rank owns the
// SAME spatial sub-box halved at every level (asserted in init), so restriction (8:1 average) and
// prolongation (trilinear) are LOCAL within a rank's block -- only the per-level ghost exchange crosses
// ranks. Mean removal between levels uses the distributed reductions (mac_reductions.cuh).
//
// Operator (A = -Laplacian, spacing h per level, h doubling each level):
//   A x = b ;  A_C = 6/h^2 , A_off = -1/h^2.  RB-GS:  x[i] = (sum_neighbours + h^2 b[i]) / 6.
//   residual r[i] = b[i] - (6 x[i] - sum_neighbours)/h^2.
// Everything is double, so the distributed V-cycle reproduces a serial full-grid V-cycle cell-for-cell
// (the only per-cell difference, halo vs in-kernel wrap, yields identical neighbour values).
#pragma once

#include "tpx/common/mpi.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mac_halo.cuh"
#include "mac_reductions.cuh"

namespace cfdmpi {

// Mixed precision: the 7-point operator coefficients (AC..AT), streamed by every smoother sweep,
// residual and matvec, are stored in single precision; the iterate x, RHS b, residual r and all the
// Krylov vectors / dot products stay double. Each coefficient*vector product promotes float*double ->
// double, so the smoother and CG arithmetic run in double on a float-stored operator. The build-time
// transmissibilities tx/ty/tz (used once to assemble the operator and to coarsen it) stay double.
// Consequence: the achievable residual floors near the single-precision level of A (~1e-6..1e-7
// relative), matching the production solver's float pressure operator. Same alias as mac_ibm.cuh
// (identical-type redefinition is legal); flip to double to recover the original convergence depth.
using mreal = float;

namespace mgdetail {

// RB-GS sweep over inner cells; colour by GLOBAL parity so the colouring is consistent across blocks.
__global__ void mg_smooth_k(double* __restrict__ x, const double* __restrict__ b, int3 ext, int3 og,
                            int g, double h2, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double sum = x[i + sx] + x[i - sx] + x[i + sy] + x[i - sy] + x[i + sz] + x[i - sz];
  x[i] = (sum + h2 * b[i]) / 6.0;
}

// ---- variable-coefficient fine-level operator (the SDF / cut-cell case) ----
// FACE-TERM ASSEMBLY CONVENTION (octree/AMR forward-compatible -- see doc/sdflow_multigrid_plan.md):
// a finite-volume face contributes t_f = aperture_f * gf, where gf = A_f/(d_f*V_cell) is the face's
// GEOMETRIC FACTOR (area / centre-distance / cell volume) and aperture_f is the cut-cell open fraction.
// gf is the "geometry provider": a per-axis COMPILE-TIME CONSTANT on the uniform Cartesian grid
// (gf_x = 1/dx^2, ...) that a later octree port would supply per face -- so the numerics below stay
// mesh-agnostic. The operator is assembled purely from face terms: diagonal = sum_f t_f, off-diagonal
// across face f = -t_f. (cc_face_term inlines to aperture*gf, so this is byte-identical / zero-cost.)
__device__ __forceinline__ double cc_face_term(double aperture, double gf) { return aperture * gf; }

// Assemble the symmetric 7-point cut-cell pressure operator by looping the 6 faces of each cell. ox[i] is
// the aperture of the -x face of cell i (== the +x face of cell i-1), so the +x face is ox[i+sx]; shared
// faces give matching coefficients on both sides -> symmetric across block/rank boundaries. (The Galerkin
// path passes pre-scaled transmissibilities with gf=1; the convention still holds, gf just absorbs into
// the coefficient.) Inputs ox/oy/oz are double; AC..AT stored single precision, the diagonal summed in
// double then cast. Mirrors compute_pressure_operator_kernel.
__global__ void mg_build_op_k(mreal* AC, mreal* AW, mreal* AE, mreal* AS, mreal* AN, mreal* AB,
                              mreal* AT, const double* ox, const double* oy, const double* oz,
                              int3 ext, int g, double gfx, double gfy, double gfz) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  // the 6 faces of cell i: low (-) and high (+) across each axis. aperture = openness; gf = axis factor.
  double tw = cc_face_term(ox[i],      gfx);  // -x  (low face reads o[i])
  double te = cc_face_term(ox[i + sx], gfx);  // +x  (high face == neighbour's -x face)
  double ts = cc_face_term(oy[i],      gfy);  // -y
  double tn = cc_face_term(oy[i + sy], gfy);  // +y
  double tb = cc_face_term(oz[i],      gfz);  // -z
  double tt = cc_face_term(oz[i + sz], gfz);  // +z
  AW[i] = (mreal)(-tw);  AE[i] = (mreal)(-te);  // off-diagonal across each face = -t_f
  AS[i] = (mreal)(-ts);  AN[i] = (mreal)(-tn);
  AB[i] = (mreal)(-tb);  AT[i] = (mreal)(-tt);
  AC[i] = (mreal)(te + tw + tn + ts + tt + tb);  // diagonal = sum of face terms (original add order)
}

// constant-coefficient diffusion operator A = I - nu_dt * Laplacian over the whole extended block:
// A_C = 1 + 6*beta, off-diagonals = -beta with beta = nu_dt / h^2. Non-singular (the I term), so no
// mean removal. Used on the coarse levels of the velocity-diffusion multigrid.
__global__ void mg_const_diffusion_op_k(mreal* AC, mreal* AW, mreal* AE, mreal* AS, mreal* AN,
                                        mreal* AB, mreal* AT, int3 ext, double beta) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  AC[i] = (mreal)(1.0 + 6.0 * beta);
  mreal nb = (mreal)(-beta);
  AW[i] = nb; AE[i] = nb; AS[i] = nb; AN[i] = nb; AB[i] = nb; AT[i] = nb;
}

__global__ void mg_smooth_var_k(double* __restrict__ x, const double* __restrict__ b,
                                const mreal* AC, const mreal* AW, const mreal* AE, const mreal* AS,
                                const mreal* AN, const mreal* AB, const mreal* AT, int3 ext, int3 og,
                                int g, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double ac = AC[i];
  if (ac < 1e-30) return;  // fully closed (solid) cell
  double s = (double)AE[i] * x[i + sx] + (double)AW[i] * x[i - sx] + (double)AN[i] * x[i + sy] +
             (double)AS[i] * x[i - sy] + (double)AT[i] * x[i + sz] + (double)AB[i] * x[i - sz];
  x[i] = (b[i] - s) / ac;
}

__global__ void mg_residual_var_k(double* __restrict__ r, const double* __restrict__ x,
                                  const double* __restrict__ b, const mreal* AC, const mreal* AW,
                                  const mreal* AE, const mreal* AS, const mreal* AN, const mreal* AB,
                                  const mreal* AT, int3 ext, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  // Ax with the float operator, accumulated in double (the residual floor is set by A's float storage).
  double Ax = (double)AC[i] * x[i] + (double)AE[i] * x[i + sx] + (double)AW[i] * x[i - sx] +
              (double)AN[i] * x[i + sy] + (double)AS[i] * x[i - sy] + (double)AT[i] * x[i + sz] +
              (double)AB[i] * x[i - sz];
  r[i] = b[i] - Ax;
}

// y = A x for the variable operator (inner cells); used as the matvec in PCG.
__global__ void mg_apply_var_k(double* __restrict__ y, const double* __restrict__ x, const mreal* AC,
                               const mreal* AW, const mreal* AE, const mreal* AS, const mreal* AN,
                               const mreal* AB, const mreal* AT, int3 ext, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  y[i] = (double)AC[i] * x[i] + (double)AE[i] * x[i + sx] + (double)AW[i] * x[i - sx] +
         (double)AN[i] * x[i + sy] + (double)AS[i] * x[i - sy] + (double)AT[i] * x[i + sz] +
         (double)AB[i] * x[i - sz];
}
__global__ void mg_axpy_k(double* y, double a, const double* x, long n) {  // y += a*x
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

// One step of the Chebyshev (point-Jacobi-preconditioned) smoother over inner cells:
//   p <- cp*p + cr * D^{-1} r ;  x <- x + p     (D = diag(A) = AC, r = b - A x supplied by the caller)
// cp=0 on the first step. Solid cells (AC==0) are skipped (decoupled by the IBM/cut-cell operator).
__global__ void mg_cheb_step_k(double* __restrict__ x, double* __restrict__ p,
                               const double* __restrict__ r, const mreal* __restrict__ AC, double cp,
                               double cr, int3 ext, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  double ac = AC[i];
  if (ac < 1e-30) { p[i] = 0.0; return; }  // fully closed (solid) cell
  double pv = cp * p[i] + cr * (r[i] / ac);
  p[i] = pv;
  x[i] += pv;
}
__global__ void mg_aypx_k(double* y, double a, const double* x, long n) {  // y = x + a*y
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + a * y[i];
}
__global__ void mg_scale_k(double* y, double s, long n) {  // y *= s
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] *= s;
}
__global__ void mg_lin_k(double* o, double a, const double* x, double b, const double* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;  // o = a*x + b*y
  if (i < n) o[i] = a * x[i] + b * y[i];
}
__global__ void mg_mask_solid_k(double* v, const mreal* AC, long n) {  // zero the operator null space
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;                // (solid cells: AC==0)
  if (i < n && AC[i] < (mreal)1e-30) v[i] = 0.0;
}

__global__ void mg_residual_k(double* __restrict__ r, const double* __restrict__ x,
                              const double* __restrict__ b, int3 ext, int g, double invh2) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double sum = x[i + sx] + x[i - sx] + x[i + sy] + x[i - sy] + x[i + sz] + x[i - sz];
  r[i] = b[i] - invh2 * (6.0 * x[i] - sum);
}

// Full-weighting average of the fine residual into the coarse rhs over the rx*ry*rz children (ratio is 2
// on a coarsened axis, 1 on a kept axis -> semi-coarsening). Local: reads fine INNER cells only (origins
// align), so no exchange needed. ratio = {2,2,2} reproduces the 8:1 average exactly.
__global__ void mg_restrict_k(double* __restrict__ coarse, const double* __restrict__ fine, int3 cext,
                              int3 fext, int g, int3 cinner, int3 ratio) {
  int icx = blockIdx.x * blockDim.x + threadIdx.x;
  int icy = blockIdx.y * blockDim.y + threadIdx.y;
  int icz = blockIdx.z * blockDim.z + threadIdx.z;
  if (icx >= cinner.x || icy >= cinner.y || icz >= cinner.z) return;
  size_t fsy = fext.x, fsz = (size_t)fext.x * fext.y;
  double sum = 0.0;
  for (int dz = 0; dz < ratio.z; ++dz)
    for (int dy = 0; dy < ratio.y; ++dy)
      for (int dx = 0; dx < ratio.x; ++dx) {
        int fx = ratio.x * icx + dx + g, fy = ratio.y * icy + dy + g, fz = ratio.z * icz + dz + g;
        sum += fine[(size_t)fx + (size_t)fy * fsy + (size_t)fz * fsz];
      }
  size_t ci = (size_t)(icx + g) + (size_t)(icy + g) * cext.x + (size_t)(icz + g) * cext.x * cext.y;
  coarse[ci] = sum / (double)(ratio.x * ratio.y * ratio.z);
}

__device__ inline double trilerp_ext(const double* c, double x, double y, double z, int3 cext) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  size_t sy = cext.x, sz = (size_t)cext.x * cext.y;
  auto F = [&](int xx, int yy, int zz) { return c[(size_t)xx + (size_t)yy * sy + (size_t)zz * sz]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy;
  double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

// Trilinear prolongation of the coarse correction, added to the fine solution. Needs the coarse ghost
// layer filled first (the stencil reaches one coarse cell beyond the block edge); sampling is then
// direct in the coarse extended block (no wrap). Coarse local sample coord = 0.5*ifine - 0.25 + g.
__global__ void mg_prolong_k(double* __restrict__ fine, const double* __restrict__ coarse, int3 fext,
                             int3 cext, int g, int3 finner, int3 ratio) {
  int ifx = blockIdx.x * blockDim.x + threadIdx.x;
  int ify = blockIdx.y * blockDim.y + threadIdx.y;
  int ifz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ifx >= finner.x || ify >= finner.y || ifz >= finner.z) return;
  // coarsened axis (ratio 2): cell-centred coarse sample 0.5*ifine - 0.25 + g; kept axis (ratio 1): the
  // fine and coarse cells coincide, so the (integer) coordinate ifine+g selects it (trilerp weight 1).
  double cx = ratio.x == 2 ? 0.5 * ifx - 0.25 + g : (double)(ifx + g);
  double cy = ratio.y == 2 ? 0.5 * ify - 0.25 + g : (double)(ify + g);
  double cz = ratio.z == 2 ? 0.5 * ifz - 0.25 + g : (double)(ifz + g);
  size_t fi = (size_t)(ifx + g) + (size_t)(ify + g) * fext.x + (size_t)(ifz + g) * fext.x * fext.y;
  fine[fi] += trilerp_ext(coarse, cx, cy, cz, cext);
}

// ---- Galerkin (aggregation/variational) coarsening for the variable-coefficient operator ----
// With piecewise-constant aggregation transfers P (injection) and R = P^T (8-cell sum), the variational
// coarse operator A_c = P^T A_f P stays a 7-point stencil whose face transmissibility is the SUM of the
// four fine-grid face transmissibilities spanning the coarse face. So the coarse operators are derived
// from the fine cut-cell operator (they "see" the solid), unlike the re-discretised constant-coefficient
// coarse operator. The matching cycle uses injection prolongation + summation restriction (both local,
// no halo) and the variable smoother/residual on every level.

// staggered face transmissibility T = openness / h^2 (the fine-level operator coefficient input)
__global__ void mg_scale_to_T_k(double* tx, double* ty, double* tz, const double* ox,
                                const double* oy, const double* oz, long n, double ix, double iy,
                                double iz) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  tx[i] = ox[i] * ix;
  ty[i] = oy[i] * iy;
  tz[i] = oz[i] * iz;
}

// coarse face transmissibility = sum of the 4 fine faces spanning the coarse face (= P^T A_f P, local).
// fine_local child base of coarse cell (clx,cly,clz) is (2*clx-g, 2*cly-g, 2*clz-g); the coarse -x face
// is the 4 fine -x faces at that base x across the (y,z) children. Fine indices are clamped (only the
// outermost coarse ghosts, never read by the operator assembly, would go out of range).
__global__ void mg_agg_T_k(double* txc, double* tyc, double* tzc, const double* txf,
                           const double* tyf, const double* tzf, int3 cext, int3 fext, int g) {
  int clx = blockIdx.x * blockDim.x + threadIdx.x;
  int cly = blockIdx.y * blockDim.y + threadIdx.y;
  int clz = blockIdx.z * blockDim.z + threadIdx.z;
  if (clx >= cext.x || cly >= cext.y || clz >= cext.z) return;
  int fx0 = 2 * clx - g, fy0 = 2 * cly - g, fz0 = 2 * clz - g;
  size_t fsy = fext.x, fsz = (size_t)fext.x * fext.y;
  auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  auto F = [&](const double* T, int x, int y, int z) {
    return T[(size_t)cl(x, fext.x) + (size_t)cl(y, fext.y) * fsy + (size_t)cl(z, fext.z) * fsz];
  };
  double sx = 0, sy = 0, sz = 0;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b) {
      sx += F(txf, fx0, fy0 + a, fz0 + b);
      sy += F(tyf, fx0 + a, fy0, fz0 + b);
      sz += F(tzf, fx0 + a, fy0 + b, fz0);
    }
  size_t i = (size_t)clx + (size_t)cly * cext.x + (size_t)clz * (size_t)cext.x * cext.y;
  txc[i] = sx;
  tyc[i] = sy;
  tzc[i] = sz;
}

// ---- geometric rediscretization: coarsen the face OPENNESS (area fraction), not the transmissibility ----
// The coarse face spans 4 fine sub-faces of 1/4 its area each, so the coarse openness (open-area
// fraction) is the AVERAGE of the 4 fine opennesses. Rebuilding the operator from this coarsened openness
// at the coarse spacing (idx2 -> idx2/4^L) gives a genuine cut-cell discretization on every level --
// unlike the Galerkin sum (mg_agg_T_k) or the geometry-blind constant-coefficient coarse operator.
// Runs over INNER coarse cells (their 4 fine faces are all in the fine inner block); the caller then
// halo-exchanges the result to fill the coarse ghost openness (periodic), which the operator build reads.
__global__ void mg_coarsen_open_avg_k(double* oxc, double* oyc, double* ozc, const double* oxf,
                                      const double* oyf, const double* ozf, int3 cext, int3 fext, int g,
                                      int3 cinner, int3 ratio) {
  int icx = blockIdx.x * blockDim.x + threadIdx.x;
  int icy = blockIdx.y * blockDim.y + threadIdx.y;
  int icz = blockIdx.z * blockDim.z + threadIdx.z;
  if (icx >= cinner.x || icy >= cinner.y || icz >= cinner.z) return;
  int rx = ratio.x, ry = ratio.y, rz = ratio.z;
  int fx0 = rx * icx + g, fy0 = ry * icy + g, fz0 = rz * icz + g;  // fine-local base of the children
  size_t fsy = fext.x, fsz = (size_t)fext.x * fext.y;
  auto F = [&](const double* T, int x, int y, int z) {
    return T[(size_t)x + (size_t)y * fsy + (size_t)z * fsz];
  };
  // each coarse -a face spans the (ratio_b * ratio_c) fine sub-faces across its two perpendicular axes;
  // the open-area fraction is their average. ratio = {2,2,2} -> the original 4-face (0.25) average.
  double sx = 0, sy = 0, sz = 0;
  for (int a = 0; a < ry; ++a)
    for (int b = 0; b < rz; ++b) sx += F(oxf, fx0, fy0 + a, fz0 + b);
  for (int a = 0; a < rx; ++a)
    for (int b = 0; b < rz; ++b) sy += F(oyf, fx0 + a, fy0, fz0 + b);
  for (int a = 0; a < rx; ++a)
    for (int b = 0; b < ry; ++b) sz += F(ozf, fx0 + a, fy0 + b, fz0);
  size_t ci = (size_t)(icx + g) + (size_t)(icy + g) * cext.x + (size_t)(icz + g) * cext.x * cext.y;
  oxc[ci] = sx / (double)(ry * rz);
  oyc[ci] = sy / (double)(rx * rz);
  ozc[ci] = sz / (double)(rx * ry);
}

// R = P^T : coarse residual = SUM of the 8 fine children (not the 1/8 average used by geometric MG)
__global__ void mg_restrict_sum_k(double* coarse, const double* fine, int3 cext, int3 fext, int g,
                                  int3 cinner) {
  int icx = blockIdx.x * blockDim.x + threadIdx.x;
  int icy = blockIdx.y * blockDim.y + threadIdx.y;
  int icz = blockIdx.z * blockDim.z + threadIdx.z;
  if (icx >= cinner.x || icy >= cinner.y || icz >= cinner.z) return;
  size_t fsy = fext.x, fsz = (size_t)fext.x * fext.y;
  double sum = 0.0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        int fx = 2 * icx + dx + g, fy = 2 * icy + dy + g, fz = 2 * icz + dz + g;
        sum += fine[(size_t)fx + (size_t)fy * fsy + (size_t)fz * fsz];
      }
  size_t ci = (size_t)(icx + g) + (size_t)(icy + g) * cext.x + (size_t)(icz + g) * cext.x * cext.y;
  coarse[ci] = sum;
}

// P : piecewise-constant injection -- each fine cell adds its coarse parent's correction (reads coarse
// INNER cells only, so no coarse halo exchange is needed before prolongation).
__global__ void mg_prolong_inject_k(double* fine, const double* coarse, int3 fext, int3 cext, int g,
                                    int3 finner) {
  int ifx = blockIdx.x * blockDim.x + threadIdx.x;
  int ify = blockIdx.y * blockDim.y + threadIdx.y;
  int ifz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ifx >= finner.x || ify >= finner.y || ifz >= finner.z) return;
  int flx = ifx + g, fly = ify + g, flz = ifz + g;
  int clx = (flx + g) / 2, cly = (fly + g) / 2, clz = (flz + g) / 2;
  size_t fi = (size_t)flx + (size_t)fly * fext.x + (size_t)flz * (size_t)fext.x * fext.y;
  size_t ci = (size_t)clx + (size_t)cly * cext.x + (size_t)clz * (size_t)cext.x * cext.y;
  fine[fi] += coarse[ci];
}

// ---- non-periodic domain boundaries (cavity / channel / step) on coarse levels ----
// The inner openness coarsening + the periodic ghost exchange do NOT fill a non-periodic boundary face,
// so the coarse operator there is undefined. Re-apply the boundary condition per level: set the boundary
// FACE openness (axis a, side s) -- Neumann wall/inflow -> 0 (closed), Dirichlet outflow -> open (1, the
// all-fluid coarse area fraction). Over the perpendicular plane. bf = the along-axis index of the face.
__global__ void mg_set_face_openness_k(double* o, int3 ext, int g, int a, int s, double value) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  int bf = (s == 0) ? g : (dims[a] - g);
  o[(size_t)p0 * strides[b] + (size_t)p1 * strides[c] + (size_t)bf * strides[a]] = value;
}

// Fill the non-periodic boundary ghosts of a coarse correction before trilinear prolongation (which
// samples one coarse cell beyond the block edge). Dirichlet (outflow): ghost = 0 (homogeneous correction).
// Neumann (wall/inflow): ghost = nearest inner cell (zero-gradient extension). Over the perpendicular plane.
__global__ void mg_fill_bc_ghost_k(double* x, int3 ext, int g, int a, int s, int dirichlet) {
  int dims[3] = {ext.x, ext.y, ext.z};
  size_t strides[3] = {1, (size_t)ext.x, (size_t)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  int p0 = blockIdx.x * blockDim.x + threadIdx.x;
  int p1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (p0 >= dims[b] || p1 >= dims[c]) return;
  size_t base = (size_t)p0 * strides[b] + (size_t)p1 * strides[c];
  size_t sa = strides[a];
  int na = dims[a];
  auto at = [&](int ia) -> double& { return x[base + (size_t)ia * sa]; };
  if (s == 0) {
    double v = dirichlet ? 0.0 : at(g);
    for (int ia = 0; ia < g; ++ia) at(ia) = v;
  } else {
    double v = dirichlet ? 0.0 : at(na - g - 1);
    for (int ia = na - g; ia < na; ++ia) at(ia) = v;
  }
}

}  // namespace mgdetail

struct MGLevel {
  MacGridHalo mac;
  double* x = nullptr;    // solution / correction (extended block)
  double* rhs = nullptr;  // right-hand side
  double* res = nullptr;  // residual scratch
  double h = 1.0;         // grid spacing
  size_t n = 0;
  int3 ext{}, og{}, inner{};
  int g = 1;
  int3 ratio{2, 2, 2};    // coarsening ratio from THIS level to the next coarser (per axis, 1 or 2)
  int3 cfac{1, 1, 1};     // cumulative coarsening factor vs level 0 (per axis, power of two) -> spacing
  // variable-coefficient operator (fine level, or every level under Galerkin coarsening); single
  // precision (mreal) -- the streamed matrix. x/rhs/res above stay double.
  bool variable = false;
  mreal *AC = nullptr, *AW = nullptr, *AE = nullptr, *AS = nullptr, *AN = nullptr, *AB = nullptr,
        *AT = nullptr;
  // staggered face transmissibilities (Galerkin only, build-time scratch): tx[i] = T of -x face of i
  double *tx = nullptr, *ty = nullptr, *tz = nullptr;
  double* cheb_p = nullptr;  // Chebyshev smoother direction vector (when enabled)
};

/// @brief Distributed geometric multigrid V-cycle solver for the pressure Poisson system.
///
/// Builds a hierarchy of MacGridHalo levels (each rank owns the same sub-box, coarsened) and runs a
/// V-cycle with red-black Gauss-Seidel (or Chebyshev) smoothing. Three coarse-operator strategies:
/// rediscretized (re-build the cut-cell operator from average-coarsened face openness on every level —
/// the recommended, grid-independent choice), Galerkin (variational A_c = PᵀAP), and constant-coefficient.
/// Usable standalone (V-cycles), as an SPD preconditioner for CG (solve_pcg), or with a Chebyshev outer
/// accelerator (solve_chebyshev, communication-light at scale). Handles non-periodic domain boundaries
/// (Neumann walls / inflow, Dirichlet outflow) per level (setBoundaryConditions) and semi-coarsening for
/// thin (quasi-2D) grids (init's `semi` flag). All arithmetic is double on a single-precision-stored
/// operator. @see DistributedNS
class DistributedPoissonMG {
 public:
  // global_res must be power-of-two and divisible by 2^(n_levels-1) (with even per-rank blocks). h0 is
  // the finest spacing; comm the communicator. ghost is the halo width of every level (1 suffices for
  // the 7-point stencil; pass a wider width to share a layout with a host solver). Asserts
  // level-to-level partition alignment.
  // semi=false: uniform 2:1 coarsening on every axis (the periodic / IBM / diffusion path; exactly
  // n_levels levels). semi=true: per-axis coarsening -- an axis is halved only while it stays even and
  // >= 2, so a thin axis (quasi-2D nz=4) freezes while the others keep coarsening; the hierarchy stops
  // early when no axis can coarsen. ratio={2,2,2} on every level reproduces the uniform path exactly.
  void init(int3 global_res, int rank, int size, double h0, int n_levels, MPI_Comm comm, int ghost = 1,
            std::array<bool, 3> periodic = {true, true, true}, bool semi = false) {
    comm_ = comm;
    levels_.clear();
    int3 res = global_res, cfac = {1, 1, 1};
    for (int L = 0; L < n_levels; ++L) {
      levels_.push_back(std::make_unique<MGLevel>());
      MGLevel& lv = *levels_[L];
      lv.mac.init(res, rank, size, periodic, ghost, comm);
      lv.g = lv.mac.ghost;
      lv.ext = lv.mac.local_ext;
      lv.og = lv.mac.origin_incl_ghost;
      lv.inner = lv.mac.inner_res();
      lv.n = lv.mac.num_local_cells();
      lv.cfac = cfac;
      lv.h = h0 * (double)cfac.x;  // (scalar h: the uniform/const-coeff path, where cfac is isotropic)
      cudaMalloc(&lv.x, lv.n * 8);
      cudaMalloc(&lv.rhs, lv.n * 8);
      cudaMalloc(&lv.res, lv.n * 8);
      cudaMemset(lv.x, 0, lv.n * 8);
      cudaMemset(lv.rhs, 0, lv.n * 8);
      cudaMemset(lv.res, 0, lv.n * 8);
      // decide this level's coarsening ratio to the next coarser level
      int3 ratio = {1, 1, 1}, next = res;
      auto coarsen = [&](int v) { return semi ? (v % 2 == 0 && v / 2 >= 2) : (v % 2 == 0); };
      if (coarsen(res.x)) { ratio.x = 2; next.x = res.x / 2; }
      if (coarsen(res.y)) { ratio.y = 2; next.y = res.y / 2; }
      if (coarsen(res.z)) { ratio.z = 2; next.z = res.z / 2; }
      lv.ratio = ratio;
      if (L > 0) {
        const MGLevel& f = *levels_[L - 1];
        int3 rt = f.ratio;  // fine -> this (coarse) ratio
        int fsx = f.og.x + f.g, fsy = f.og.y + f.g, fsz = f.og.z + f.g;
        int csx = lv.og.x + lv.g, csy = lv.og.y + lv.g, csz = lv.og.z + lv.g;
        bool aligned = (f.inner.x == rt.x * lv.inner.x) && (f.inner.y == rt.y * lv.inner.y) &&
                       (f.inner.z == rt.z * lv.inner.z) && (fsx == rt.x * csx) && (fsy == rt.y * csy) &&
                       (fsz == rt.z * csz);
        if (!aligned) {
          int rk = 0;
          MPI_Comm_rank(comm, &rk);
          fprintf(stderr,
                  "[DistributedPoissonMG] rank %d: level %d block not aligned to level %d (ratio %d,%d,%d; "
                  "fine inner %d,%d,%d start %d,%d,%d ; coarse inner %d,%d,%d start %d,%d,%d). Requires "
                  "power-of-two global_res with even per-rank blocks.\n",
                  rk, L, L - 1, rt.x, rt.y, rt.z, f.inner.x, f.inner.y, f.inner.z, fsx, fsy, fsz,
                  lv.inner.x, lv.inner.y, lv.inner.z, csx, csy, csz);
          MPI_Abort(comm, 1);
        }
      }
      if (next.x == res.x && next.y == res.y && next.z == res.z) break;  // nothing coarsens -> stop
      res = next;
      cfac = make_int3(cfac.x * ratio.x, cfac.y * ratio.y, cfac.z * ratio.z);
    }
  }

  void free() {
    for (auto& lp : levels_) {
      MGLevel& lv = *lp;
      for (double** p : {&lv.x, &lv.rhs, &lv.res, &lv.tx, &lv.ty, &lv.tz, &lv.cheb_p}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
      }
      for (mreal** p : {&lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB, &lv.AT}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
      }
    }
    levels_.clear();
  }

  // Install a variable-coefficient operator on the fine level from staggered face openness fields on
  // the extended block: ox[i] = openness of the -x face of cell i (similarly oy, oz). The caller fills
  // ox/oy/oz on the WHOLE extended block (incl. ghosts); a face is a deterministic function of its
  // global position, so every rank derives matching values for shared faces (no exchange needed).
  // idx2 = 1/dx^2, etc.
  //
  // galerkin=false: only the fine level is variable; coarse levels stay constant-coefficient (mirrors
  //   the serial use_periodic_operator = level>0). Cheap, but a poor coarse model for stiff cut cells.
  // galerkin=true: build the variational coarse operators A_c = P^T A_f P (aggregation) -- the coarse
  //   face transmissibility is the sum of the 4 fine faces it spans -- on EVERY level, so the coarse
  //   grids see the geometry. The V-cycle then uses injection prolongation + summation restriction.
  // Toggle the constant null-space (mean) removal. Default true for the all-Neumann pressure operator;
  // set false when a Dirichlet face (e.g. a pressure outflow) makes the operator non-singular.
  void setRemoveMean(bool on) { remove_mean_ = on; }

  // Per-face domain boundary types {-x,+x,-y,+y,-z,+z}: 0=periodic, 1/2=Neumann (wall/inflow), 3=Dirichlet
  // (outflow). Default all-0 (periodic) -> the boundary handling below is a no-op, so the periodic / IBM
  // pressure path stays byte-identical. Set by the solver for native-BC problems so the rediscretized
  // coarse operators + the trilinear prolongation get the right non-periodic boundary treatment per level.
  void setBoundaryConditions(const int bc[6]) {
    has_bc_ = false;
    for (int i = 0; i < 6; ++i) {
      bc_type_[i] = bc[i];
      if (bc[i] != 0) has_bc_ = true;
    }
  }

  void setFineVariableOperator(const double* ox, const double* oy, const double* oz, double idx2,
                               double idy2, double idz2, bool galerkin = false) {
    galerkin_ = galerkin;
    MGLevel& f = *levels_[0];
    auto alloc_op = [](MGLevel& lv) {
      if (!lv.AC)
        for (mreal** p : {&lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB, &lv.AT})
          cudaMalloc(p, lv.n * sizeof(mreal));
    };
    auto alloc_T = [](MGLevel& lv) {
      if (!lv.tx)
        for (double** p : {&lv.tx, &lv.ty, &lv.tz}) cudaMalloc(p, lv.n * 8);
    };
    dim3 blk(8, 8, 8);
    auto grd = [](const MGLevel& lv) {
      return dim3((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    };
    alloc_op(f);
    f.variable = true;

    if (!galerkin) {
      mgdetail::mg_build_op_k<<<grd(f), blk>>>(f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT, ox, oy, oz,
                                               f.ext, f.g, idx2, idy2, idz2);
      return;
    }

    // fine transmissibility T = openness/h^2; fine operator assembled from T (idx2=1)
    alloc_T(f);
    int t1 = 256, b1 = (int)((f.n + t1 - 1) / t1);
    mgdetail::mg_scale_to_T_k<<<b1, t1>>>(f.tx, f.ty, f.tz, ox, oy, oz, (long)f.n, idx2, idy2, idz2);
    mgdetail::mg_build_op_k<<<grd(f), blk>>>(f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT, f.tx, f.ty,
                                             f.tz, f.ext, f.g, 1.0, 1.0, 1.0);
    // coarsen the transmissibilities and re-assemble the variational operator on every coarse level
    for (int L = 1; L < (int)levels_.size(); ++L) {
      MGLevel& c = *levels_[L];
      MGLevel& fin = *levels_[L - 1];
      alloc_op(c);
      alloc_T(c);
      c.variable = true;
      dim3 gAll((c.ext.x + 7) / 8, (c.ext.y + 7) / 8, (c.ext.z + 7) / 8);
      mgdetail::mg_agg_T_k<<<gAll, blk>>>(c.tx, c.ty, c.tz, fin.tx, fin.ty, fin.tz, c.ext, fin.ext,
                                          c.g);
      mgdetail::mg_build_op_k<<<grd(c), blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, c.tx, c.ty,
                                               c.tz, c.ext, c.g, 1.0, 1.0, 1.0);
    }
  }

  // Geometric REDISCRETIZED variable operator (the recommended cut-cell path): build the fine operator
  // from the face openness, then on every coarse level average-coarsen the openness (mg_coarsen_open_avg_k
  // + a periodic ghost exchange) and re-assemble the cut-cell operator at the coarse spacing. Every level
  // is a genuine discretization (consistent, unlike Galerkin aggregation), and the V-cycle uses the
  // geometric transfers (average restriction + trilinear prolongation, galerkin_=false) -- identical to
  // the working constant-coefficient path but with a geometry-aware coarse operator. Coarse openness is
  // stashed in the per-level tx/ty/tz scratch. idx2 = 1/dx^2 on the fine level.
  void setFineVariableOperatorRediscretized(const double* ox, const double* oy, const double* oz,
                                            double idx2, double idy2, double idz2) {
    galerkin_ = false;  // geometric transfers (average restrict + trilinear prolong)
    auto alloc_op = [](MGLevel& lv) {
      if (!lv.AC)
        for (mreal** p : {&lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB, &lv.AT})
          cudaMalloc(p, lv.n * sizeof(mreal));
    };
    auto alloc_T = [](MGLevel& lv) {
      if (!lv.tx)
        for (double** p : {&lv.tx, &lv.ty, &lv.tz}) cudaMalloc(p, lv.n * 8);
    };
    dim3 blk(8, 8, 8);
    auto grd = [](const MGLevel& lv) {
      return dim3((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    };
    MGLevel& f = *levels_[0];
    alloc_op(f);
    f.variable = true;
    mgdetail::mg_build_op_k<<<grd(f), blk>>>(f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT, ox, oy, oz, f.ext,
                                             f.g, idx2, idy2, idz2);
    const double *pox = ox, *poy = oy, *poz = oz;  // fine openness feeding the next coarsening
    for (int L = 1; L < (int)levels_.size(); ++L) {
      MGLevel& c = *levels_[L];
      MGLevel& fin = *levels_[L - 1];
      alloc_op(c);
      alloc_T(c);
      c.variable = true;
      mgdetail::mg_coarsen_open_avg_k<<<grd(c), blk>>>(c.tx, c.ty, c.tz, pox, poy, poz, c.ext, fin.ext,
                                                       c.g, c.inner, fin.ratio);
      c.mac.exchange(c.tx);  // periodic ghost openness (the operator build reads the +neighbour face)
      c.mac.exchange(c.ty);
      c.mac.exchange(c.tz);
      if (has_bc_) applyBoundaryOpenness(c);  // re-impose non-periodic boundary faces (coarsen+exchange
                                              //   leave them undefined): Neumann -> 0, Dirichlet -> open
      // per-axis coarse spacing (semi-coarsening is anisotropic): idx2_a = idx2 / cfac_a^2.
      double sx = 1.0 / (double)(c.cfac.x * c.cfac.x), sy = 1.0 / (double)(c.cfac.y * c.cfac.y),
             sz = 1.0 / (double)(c.cfac.z * c.cfac.z);
      mgdetail::mg_build_op_k<<<grd(c), blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, c.tx, c.ty, c.tz,
                                               c.ext, c.g, idx2 * sx, idy2 * sy, idz2 * sz);
      pox = c.tx;
      poy = c.ty;
      poz = c.tz;
    }
  }

  // Velocity-diffusion multigrid: constant-coefficient A = I - nu_dt*Laplacian on the COARSE levels (the
  // bulk diffusion, coarse spacing). Geometric (trilinear) transfers; NO mean removal (non-singular).
  // Component-independent -> build once; then setDiffusionFine() swaps the fine stencil per component.
  void setDiffusionCoarse(double nu_dt, double h0) {
    galerkin_ = false;
    remove_mean_ = false;
    dim3 blk(8, 8, 8);
    for (int L = 1; L < (int)levels_.size(); ++L) {
      MGLevel& c = *levels_[L];
      if (!c.AC)
        for (mreal** p : {&c.AC, &c.AW, &c.AE, &c.AS, &c.AN, &c.AB, &c.AT})
          cudaMalloc(p, c.n * sizeof(mreal));
      c.variable = true;
      double hL = h0 * (double)(1 << L), beta = nu_dt / (hL * hL);
      dim3 gAll((c.ext.x + 7) / 8, (c.ext.y + 7) / 8, (c.ext.z + 7) / 8);
      mgdetail::mg_const_diffusion_op_k<<<gAll, blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, c.ext,
                                                       beta);
    }
  }

  // Install the fine-level 7-point stencil (copied from external A_C..A_T device arrays -- e.g. the
  // IBM-modified diffusion stencil). fineA = {AC,AW,AE,AS,AN,AB,AT}, in the level-0 layout.
  void setDiffusionFine(mreal* const fineA[7]) {
    MGLevel& f = *levels_[0];
    if (!f.AC)
      for (mreal** p : {&f.AC, &f.AW, &f.AE, &f.AS, &f.AN, &f.AB, &f.AT})
        cudaMalloc(p, f.n * sizeof(mreal));
    f.variable = true;
    mreal* fdst[7] = {f.AC, f.AW, f.AE, f.AS, f.AN, f.AB, f.AT};
    for (int k = 0; k < 7; ++k)
      cudaMemcpy(fdst[k], fineA[k], f.n * sizeof(mreal), cudaMemcpyDeviceToDevice);
  }

  // Use a degree-`pre/post` Chebyshev smoother (point-Jacobi preconditioned) instead of Red-Black
  // Gauss-Seidel on the variable-coefficient levels. The damped band is [cheb_b/eig_ratio, cheb_b] with
  // cheb_b = 2.1 (a rigorous upper bound on rho(D^{-1}A) for this M-matrix operator, +5% margin); a
  // wider band (larger eig_ratio) covers more of the high-frequency spectrum and keeps the smoother
  // non-amplifying for the CG preconditioner. Call after the operators are built (setFineVariable...).
  void enableChebyshev(double eig_ratio = 30.0) {
    cheb_enabled_ = true;
    cheb_b_ = 2.1;
    cheb_a_ = cheb_b_ / eig_ratio;
    for (auto& lp : levels_) {
      MGLevel& lv = *lp;
      if (lv.variable && !lv.cheb_p) {
        cudaMalloc(&lv.cheb_p, lv.n * sizeof(double));
        cudaMemset(lv.cheb_p, 0, lv.n * sizeof(double));
      }
    }
  }

  MGLevel& level(int L) { return *levels_[L]; }
  int n_levels() const { return (int)levels_.size(); }

  // Run n_vcycles of the V-cycle starting at level 0 (rhs on level 0 must be set; x starts at 0).
  void solve(int n_vcycles, int pre, int post, int bottom) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    // DIAGNOSTIC (env SDFLOW_VMG_TRACE): print the level-0 residual reduction per V-cycle for the first
    // few solve() calls -- answers "is the V-cycle actually reducing the residual, or effectively a no-op?".
    static int trace_left = -1;
    if (trace_left < 0) trace_left = (std::getenv("SDFLOW_VMG_TRACE") ? 4 : 0);
    bool trace = trace_left > 0;
    if (trace) {
      MGLevel& l0 = *levels_[0];
      residual(l0);
      std::fprintf(stderr, "[VMG] levels=%zu max|r0|=%.4e\n", levels_.size(),
                   mac_max_abs(l0.res, l0.mac, comm_));
    }
    for (int v = 0; v < n_vcycles; ++v) {
      vcycle(0);
      if (trace) {
        MGLevel& l0 = *levels_[0];
        residual(l0);
        std::fprintf(stderr, "[VMG]   cycle %d max|r|=%.4e\n", v, mac_max_abs(l0.res, l0.mac, comm_));
      }
    }
    if (trace_left > 0) trace_left--;
  }

  // Conjugate Gradients preconditioned by ONE symmetric V-cycle (M^{-1}). For the stiff cut-cell /
  // strongly-variable operator the standalone (unsmoothed-aggregation) V-cycle is only a modest
  // smoother, but as an SPD preconditioner it makes CG converge robustly. Requires the variable
  // operator (level 0 .variable). rhs on level 0 is the RHS; the solution is left in level-0 x.
  // Returns the iteration count; stops when max|r| < rtol*max|r0| or max_iter is reached.
  int solve_pcg(int max_iter, double rtol, int pre, int post, int bottom) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    MGLevel& l0 = *levels_[0];
    int g = l0.g;
    long n = (long)l0.n;
    dim3 blk(8, 8, 8);
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    int t1 = 256, b1 = (int)((n + t1 - 1) / t1);

    double *b, *x, *r, *p, *z, *Ap;
    for (double** q : {&b, &x, &r, &p, &z, &Ap}) cudaMalloc(q, n * 8);
    cudaMemcpy(b, l0.rhs, n * 8, cudaMemcpyDeviceToDevice);  // PCG RHS (rhs is reused as V-cycle scratch)
    cudaMemcpy(x, l0.x, n * 8, cudaMemcpyDeviceToDevice);    // initial guess

    auto matvec = [&](double* y, double* v) {  // y = A v
      l0.mac.exchange(v);
      mgdetail::mg_apply_var_k<<<grd, blk>>>(y, v, l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB, l0.AT,
                                             l0.ext, g);
    };
    auto precond = [&](double* zz, double* rr) {  // zz = M^{-1} rr  (one symmetric V-cycle)
      cudaMemcpy(l0.rhs, rr, n * 8, cudaMemcpyDeviceToDevice);
      cudaMemset(l0.x, 0, n * 8);
      vcycle(0, /*sym=*/true);
      cudaMemcpy(zz, l0.x, n * 8, cudaMemcpyDeviceToDevice);
    };

    matvec(Ap, x);                                  // r = b - A x
    cudaMemcpy(r, b, n * 8, cudaMemcpyDeviceToDevice);
    mgdetail::mg_axpy_k<<<b1, t1>>>(r, -1.0, Ap, n);
    double r0 = mac_max_abs(r, l0.mac, comm_);
    int it = 0;
    if (r0 > 0.0) {
      precond(z, r);
      cudaMemcpy(p, z, n * 8, cudaMemcpyDeviceToDevice);
      double rz = mac_dot(r, z, l0.mac, comm_);
      for (; it < max_iter; ++it) {
        matvec(Ap, p);
        double pAp = mac_dot(p, Ap, l0.mac, comm_);
        double alpha = rz / pAp;
        mgdetail::mg_axpy_k<<<b1, t1>>>(x, alpha, p, n);    // x += alpha p
        mgdetail::mg_axpy_k<<<b1, t1>>>(r, -alpha, Ap, n);  // r -= alpha Ap
        if (mac_max_abs(r, l0.mac, comm_) < rtol * r0) {
          ++it;
          break;
        }
        precond(z, r);
        double rz_new = mac_dot(r, z, l0.mac, comm_);
        double beta = rz_new / rz;
        mgdetail::mg_aypx_k<<<b1, t1>>>(p, beta, z, n);  // p = z + beta p
        rz = rz_new;
      }
    }
    cudaMemcpy(l0.x, x, n * 8, cudaMemcpyDeviceToDevice);
    if (remove_mean_) mac_remove_mean(l0.x, l0.mac, comm_);
    cudaMemcpy(l0.rhs, b, n * 8, cudaMemcpyDeviceToDevice);  // restore RHS (precond used it as scratch)
    for (double* q : {b, x, r, p, z, Ap}) cudaFree(q);
    return it;
  }

  // Estimate the spectral bounds [lmin, lmax] of the preconditioned operator M^{-1}A (M = one symmetric
  // V-cycle) by power iteration. lmax: power iteration on M^{-1}A; lmin: power iteration on lmax*I-M^{-1}A.
  // This is a ONE-TIME setup cost (the operator is fixed, so the bounds are reused across every solve),
  // so its reductions do NOT count against the per-solve communication budget Chebyshev is meant to cut.
  void estimate_eigenvalues(double& lmin, double& lmax, int iters, int pre, int post, int bottom) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    MGLevel& l0 = *levels_[0];
    int g = l0.g;
    long n = (long)l0.n;
    dim3 blk(8, 8, 8);
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    int t1 = 256, b1 = (int)((n + t1 - 1) / t1);
    double *v, *w, *z, *srhs, *sx;
    for (double** q : {&v, &w, &z, &srhs, &sx}) cudaMalloc(q, n * 8);
    cudaMemcpy(srhs, l0.rhs, n * 8, cudaMemcpyDeviceToDevice);  // preserve caller state (precond scratch)
    cudaMemcpy(sx, l0.x, n * 8, cudaMemcpyDeviceToDevice);
    auto matvec = [&](double* y, double* x) {
      l0.mac.exchange(x);
      mgdetail::mg_apply_var_k<<<grd, blk>>>(y, x, l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB, l0.AT, l0.ext, g);
    };
    auto mask = [&](double* x) { mgdetail::mg_mask_solid_k<<<b1, t1>>>(x, l0.AC, n); };
    auto applyT = [&](double* out, double* in) {  // out = M^{-1} A in, projected onto the fluid range
      matvec(w, in);
      cudaMemcpy(l0.rhs, w, n * 8, cudaMemcpyDeviceToDevice);
      cudaMemset(l0.x, 0, n * 8);
      vcycle(0, /*sym=*/true);
      cudaMemcpy(out, l0.x, n * 8, cudaMemcpyDeviceToDevice);
      if (remove_mean_) mac_remove_mean(out, l0.mac, comm_);  // project out the constant null mode
      mask(out);                                              // project out the solid-cell null modes
    };
    auto normalize = [&](double* x) {
      double nr = sqrt(mac_dot(x, x, l0.mac, comm_));
      if (nr > 0) mgdetail::mg_scale_k<<<b1, t1>>>(x, 1.0 / nr, n);
    };
    auto seed = [&](double* x) {  // mean-zero, solid-free, normalized seed
      cudaMemcpy(x, srhs, n * 8, cudaMemcpyDeviceToDevice);
      if (remove_mean_) mac_remove_mean(x, l0.mac, comm_);
      mask(x);
      normalize(x);
    };
    seed(v);
    lmax = 1.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      lmax = mac_dot(v, z, l0.mac, comm_);  // Rayleigh quotient (v normalized)
      cudaMemcpy(v, z, n * 8, cudaMemcpyDeviceToDevice);
      normalize(v);
    }
    seed(v);
    double mu = 0.0;
    for (int k = 0; k < iters; ++k) {
      applyT(z, v);
      mgdetail::mg_lin_k<<<b1, t1>>>(z, lmax, v, -1.0, z, n);  // z = lmax*v - T v
      mu = mac_dot(v, z, l0.mac, comm_);
      cudaMemcpy(v, z, n * 8, cudaMemcpyDeviceToDevice);
      normalize(v);
    }
    double e_hi = lmax, e_lo = lmax - mu;  // direct (max) and shifted (min) Rayleigh estimates
    lmin = e_lo < e_hi ? e_lo : e_hi;      // robust bracket: a tight cluster can make the direct power
    lmax = e_lo < e_hi ? e_hi : e_lo;      // iteration under-resolve the max, so take min/max of the two
    if (lmin < 0.02 * lmax) lmin = 0.02 * lmax;
    cudaMemcpy(l0.rhs, srhs, n * 8, cudaMemcpyDeviceToDevice);  // restore
    cudaMemcpy(l0.x, sx, n * 8, cudaMemcpyDeviceToDevice);
    for (double* q : {v, w, z, srhs, sx}) cudaFree(q);
  }

  // Chebyshev semi-iteration preconditioned by ONE symmetric V-cycle (M^{-1}). Same goal as solve_pcg,
  // but the step coefficients come from the spectral bounds [a,b]=[lmin,lmax] of M^{-1}A (from
  // estimate_eigenvalues) -- so NO per-iteration global dot-products. Per iteration: 1 V-cycle + 1 matvec
  // + axpys + the residual check (1 reduction). At scale this avoids PCG's 2 dot-product Allreduce/iter.
  // rhs on level 0 is the RHS; the solution is left in level-0 x. Returns the V-cycle (iteration) count.
  int solve_chebyshev(int max_iter, double rtol, int pre, int post, int bottom, double a, double b) {
    pre_ = pre; post_ = post; bottom_ = bottom;
    MGLevel& l0 = *levels_[0];
    int g = l0.g;
    long n = (long)l0.n;
    dim3 blk(8, 8, 8);
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    int t1 = 256, b1 = (int)((n + t1 - 1) / t1);
    if (a > b) { double t = a; a = b; b = t; }     // robust to swapped bounds
    a *= 0.95; b *= 1.05;                           // safety margin: [a,b] must bracket the spectrum
    double *rhs, *x, *r, *z, *d, *w;
    for (double** q : {&rhs, &x, &r, &z, &d, &w}) cudaMalloc(q, n * 8);
    cudaMemcpy(rhs, l0.rhs, n * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpy(x, l0.x, n * 8, cudaMemcpyDeviceToDevice);
    auto matvec = [&](double* y, double* v) {
      l0.mac.exchange(v);
      mgdetail::mg_apply_var_k<<<grd, blk>>>(y, v, l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB, l0.AT, l0.ext, g);
    };
    auto precond = [&](double* zz, double* rr) {  // zz = M^{-1} rr (one symmetric V-cycle)
      cudaMemcpy(l0.rhs, rr, n * 8, cudaMemcpyDeviceToDevice);
      cudaMemset(l0.x, 0, n * 8);
      vcycle(0, /*sym=*/true);
      cudaMemcpy(zz, l0.x, n * 8, cudaMemcpyDeviceToDevice);
    };
    double theta = 0.5 * (b + a), delta = 0.5 * (b - a), sigma1 = theta / delta, rho = 1.0 / sigma1;
    matvec(w, x);  // r = rhs - A x
    cudaMemcpy(r, rhs, n * 8, cudaMemcpyDeviceToDevice);
    mgdetail::mg_axpy_k<<<b1, t1>>>(r, -1.0, w, n);
    double r0 = mac_max_abs(r, l0.mac, comm_);
    int nvc = 0;
    if (r0 > 0.0) {
      precond(z, r); ++nvc;                                        // z = M^{-1} r
      mgdetail::mg_lin_k<<<b1, t1>>>(d, 1.0 / theta, z, 0.0, z, n);  // d = z / theta
      mgdetail::mg_axpy_k<<<b1, t1>>>(x, 1.0, d, n);                 // x += d
      for (int i = 1; i < max_iter; ++i) {
        matvec(w, d);
        mgdetail::mg_axpy_k<<<b1, t1>>>(r, -1.0, w, n);             // r -= A d
        if (mac_max_abs(r, l0.mac, comm_) < rtol * r0) break;
        precond(z, r); ++nvc;
        double rho_new = 1.0 / (2.0 * sigma1 - rho);
        mgdetail::mg_lin_k<<<b1, t1>>>(d, rho_new * rho, d, 2.0 * rho_new / delta, z, n);  // d update
        mgdetail::mg_axpy_k<<<b1, t1>>>(x, 1.0, d, n);             // x += d
        rho = rho_new;
      }
    }
    cudaMemcpy(l0.x, x, n * 8, cudaMemcpyDeviceToDevice);
    if (remove_mean_) mac_remove_mean(l0.x, l0.mac, comm_);
    cudaMemcpy(l0.rhs, rhs, n * 8, cudaMemcpyDeviceToDevice);
    for (double* q : {rhs, x, r, z, d, w}) cudaFree(q);
    return nvc;
  }

 private:
  // reverse=true sweeps black-then-red (the transpose ordering) so that a forward pre-smooth + reverse
  // post-smooth makes the V-cycle symmetric -> usable as an SPD preconditioner for CG.
  void smooth(MGLevel& lv, int sweeps, bool reverse = false) {
    // Chebyshev is a polynomial in A -> symmetric by construction, so `reverse` (the RB-GS symmetry
    // trick for the CG preconditioner) is unnecessary; one degree-`sweeps` pass serves pre and post.
    if (cheb_enabled_ && lv.variable) {
      smoothCheb(lv, sweeps);
      return;
    }
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    double h2 = lv.h * lv.h;
    for (int k = 0; k < sweeps; ++k) {
      for (int s = 0; s < 2; ++s) {
        int color = reverse ? (1 - s) : s;
        lv.mac.exchange(lv.x);
        if (lv.variable)
          mgdetail::mg_smooth_var_k<<<grd, blk>>>(lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS, lv.AN,
                                                  lv.AB, lv.AT, lv.ext, lv.og, lv.g, color);
        else
          mgdetail::mg_smooth_k<<<grd, blk>>>(lv.x, lv.rhs, lv.ext, lv.og, lv.g, h2, color);
      }
    }
  }

  // Degree-`degree` Chebyshev smoother on the variable operator, point-Jacobi preconditioned. The
  // damped interval is [cheb_a_, cheb_b_] in the spectrum of D^{-1}A; cheb_b_ > 2 >= rho(D^{-1}A)
  // (Gershgorin: the M-matrix rows have |off-diag| sum == diagonal -> spectrum in [0,2] on every level),
  // so the smoother never amplifies. Each step costs one residual (exchange + matvec) like a GS sweep,
  // but fewer halo exchanges (one per degree vs two per GS sweep).
  void smoothCheb(MGLevel& lv, int degree) {
    if (degree < 1) return;
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    double a = cheb_a_, b = cheb_b_;
    double theta = 0.5 * (b + a), delta = 0.5 * (b - a), sigma = theta / delta, rho = 1.0 / sigma;
    residual(lv);  // lv.res = b - A x
    mgdetail::mg_cheb_step_k<<<grd, blk>>>(lv.x, lv.cheb_p, lv.res, lv.AC, /*cp=*/0.0,
                                           /*cr=*/1.0 / theta, lv.ext, lv.g);
    for (int k = 1; k < degree; ++k) {
      residual(lv);
      double rho_new = 1.0 / (2.0 * sigma - rho);
      mgdetail::mg_cheb_step_k<<<grd, blk>>>(lv.x, lv.cheb_p, lv.res, lv.AC, /*cp=*/rho * rho_new,
                                             /*cr=*/2.0 * rho_new / delta, lv.ext, lv.g);
      rho = rho_new;
    }
  }

  void residual(MGLevel& lv) {
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    lv.mac.exchange(lv.x);
    if (lv.variable)
      mgdetail::mg_residual_var_k<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS,
                                                lv.AN, lv.AB, lv.AT, lv.ext, lv.g);
    else
      mgdetail::mg_residual_k<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.ext, lv.g, 1.0 / (lv.h * lv.h));
  }

  // sym=true makes the cycle symmetric (reverse post-smooth) for use as a CG preconditioner; the
  // default solve() path uses sym=false, preserving the exact behaviour the cell-for-cell tests check.
  void vcycle(int L, bool sym = false) {
    MGLevel& lv = *levels_[L];
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);

    if (L + 1 == (int)levels_.size()) {  // bottom: smooth + remove null space
      smooth(lv, bottom_);
      if (remove_mean_) mac_remove_mean(lv.x, lv.mac, comm_);
      return;
    }

    smooth(lv, pre_);

    // residual (needs current ghosts); variable operator on the fine level, constant on coarse
    residual(lv);

    // restrict residual -> coarse rhs (local); reset coarse x. Galerkin uses R = P^T (sum); the
    // geometric path uses full-weighting (1/8 average).
    MGLevel& cs = *levels_[L + 1];
    dim3 cgrd((cs.inner.x + 7) / 8, (cs.inner.y + 7) / 8, (cs.inner.z + 7) / 8);
    if (galerkin_)
      mgdetail::mg_restrict_sum_k<<<cgrd, blk>>>(cs.rhs, lv.res, cs.ext, lv.ext, lv.g, cs.inner);
    else
      mgdetail::mg_restrict_k<<<cgrd, blk>>>(cs.rhs, lv.res, cs.ext, lv.ext, lv.g, cs.inner, lv.ratio);
    cudaMemset(cs.x, 0, cs.n * 8);

    vcycle(L + 1, sym);

    // prolong coarse correction -> fine. Galerkin uses injection (P, no coarse halo); the geometric
    // path uses trilinear interpolation (needs the coarse ghosts).
    if (galerkin_) {
      mgdetail::mg_prolong_inject_k<<<grd, blk>>>(lv.x, cs.x, lv.ext, cs.ext, lv.g, lv.inner);
    } else {
      cs.mac.exchange(cs.x);
      if (has_bc_) fillBCGhosts(cs);  // non-periodic boundary ghosts the trilinear prolong samples:
                                      //   Neumann -> zero-gradient, Dirichlet -> 0
      mgdetail::mg_prolong_k<<<grd, blk>>>(lv.x, cs.x, lv.ext, cs.ext, lv.g, lv.inner, lv.ratio);
    }

    smooth(lv, post_, /*reverse=*/sym);
    if (remove_mean_) mac_remove_mean(lv.x, lv.mac, comm_);
  }

  // which domain-boundary faces this rank owns at level lv (low/high along each axis).
  void bcFaceFlags(const MGLevel& lv, bool lo[3], bool hi[3]) const {
    int g = lv.g;
    int3 og = lv.og, N = lv.mac.global_res, e = lv.ext;
    lo[0] = og.x + g == 0;  lo[1] = og.y + g == 0;  lo[2] = og.z + g == 0;
    hi[0] = og.x + e.x - g == N.x;  hi[1] = og.y + e.y - g == N.y;  hi[2] = og.z + e.z - g == N.z;
  }
  // re-impose the boundary FACE openness on a coarse level (Neumann -> 0, Dirichlet outflow -> open).
  void applyBoundaryOpenness(MGLevel& c) {
    bool lo[3], hi[3];
    bcFaceFlags(c, lo, hi);
    int dims[3] = {c.ext.x, c.ext.y, c.ext.z};
    double* oarr[3] = {c.tx, c.ty, c.tz};
    for (int a = 0; a < 3; ++a) {
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      for (int s = 0; s < 2; ++s) {
        int face = 2 * a + s, t = bc_type_[face];
        if (t == 0 || (s == 0 ? !lo[a] : !hi[a])) continue;
        mgdetail::mg_set_face_openness_k<<<grd, blk>>>(oarr[a], c.ext, c.g, a, s, t == 3 ? 1.0 : 0.0);
      }
    }
  }
  // fill a coarse correction's non-periodic boundary ghosts before trilinear prolongation.
  void fillBCGhosts(MGLevel& c) {
    bool lo[3], hi[3];
    bcFaceFlags(c, lo, hi);
    int dims[3] = {c.ext.x, c.ext.y, c.ext.z};
    for (int a = 0; a < 3; ++a) {
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      for (int s = 0; s < 2; ++s) {
        int face = 2 * a + s, t = bc_type_[face];
        if (t == 0 || (s == 0 ? !lo[a] : !hi[a])) continue;
        mgdetail::mg_fill_bc_ghost_k<<<grd, blk>>>(c.x, c.ext, c.g, a, s, t == 3 ? 1 : 0);
      }
    }
  }

  std::vector<std::unique_ptr<MGLevel>> levels_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int pre_ = 2, post_ = 2, bottom_ = 8;
  bool galerkin_ = false;     // variational (aggregation) coarse operators + injection/sum transfers
  bool remove_mean_ = true;   // false for non-singular operators (e.g. diffusion I - nu*dt*Lap)
  int bc_type_[6] = {0, 0, 0, 0, 0, 0};  // per-face domain BC (0=periodic); see setBoundaryConditions
  bool has_bc_ = false;       // any non-periodic face -> apply the coarse-level boundary handling
  bool cheb_enabled_ = false;       // Chebyshev smoother (variable levels) instead of Red-Black GS
  double cheb_a_ = 0.0, cheb_b_ = 0.0;  // damped spectral band [a,b] of D^{-1}A
};

}  // namespace cfdmpi
