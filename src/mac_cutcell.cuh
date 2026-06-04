// cfd-gpu -- cut-cell pressure-operator face openness from an SDF, on a MacGridHalo extended block.
//
// Reproduces cfd's compute_fluid_fraction_kernel (gradient-normalised fluid fraction) + the operator
// mask (compute_pressure_operator_kernel: frac_u_l = (sdf_face>0)? frac_u : 0) to produce the staggered
// face openness ox/oy/oz consumed by DistributedPoissonMG::setFineVariableOperator: ox[i] = openness of
// the -x face of cell i. The fraction arithmetic is factored into cc_fraction_core so the distributed
// (direct-indexed) build and a serial (wrap-indexed) reference share identical math -> the distributed
// coefficients match the serial ones cell-for-cell. The SDF must be provided on the WHOLE extended
// block (inner + ghost); with ghost width 2 the fraction stencil (reach 1.5 around a face) is in-bounds
// for every inner cell and its immediate high-side neighbour, which is all setFineVariableOperator reads.
#pragma once

#include <cuda_runtime.h>

namespace cfdmpi {
namespace ccdetail {

// Masked fluid fraction of a face from its SDF samples (centre + the 6 axis neighbours at +/-1).
// type: 1 = x-face (u), 2 = y-face (v), 3 = z-face (w). Mirrors compute_fluid_fraction_kernel; the
// (sd<=0 -> 0) mask folds in compute_pressure_operator_kernel's (sdf_face>0) gate.
__host__ __device__ inline double cc_fraction_core(double sd, double sxp, double sxm, double syp,
                                                   double sym, double szp, double szm, int type,
                                                   double dx, double dy, double dz) {
  if (sd <= 0.0) return 0.0;  // face centre inside solid -> closed
  double gx = (sxp - sxm) / (2.0 * dx);
  double gy = (syp - sym) / (2.0 * dy);
  double gz = (szp - szm) / (2.0 * dz);
  double gmag = sqrt(gx * gx + gy * gy + gz * gz);
  if (gmag < 1e-6) gmag = 1e-6;
  double nx = gx / gmag, ny = gy / gmag, nz = gz / gmag;
  double denom = 0.0;
  if (type == 1)
    denom = fabs(ny) * dy + fabs(nz) * dz;
  else if (type == 2)
    denom = fabs(nx) * dx + fabs(nz) * dz;
  else
    denom = fabs(nx) * dx + fabs(ny) * dy;
  if (denom < 1e-9) denom = 1e-9;
  double frac = 0.5 + sd / denom;
  if (frac < 0.0) frac = 0.0;
  if (frac > 1.0) frac = 1.0;
  return frac;
}

// Trilinear SDF sample in the extended block (local coords), clamped to [0, ext-1] in each axis. Clamp
// only fires on the outermost ghosts, which the operator build never reads (their stencils stay
// in-bounds), so it never affects a used coefficient.
__device__ inline double cc_sample_ext(const double* __restrict__ sdf, int3 ext, double x, double y,
                                       double z) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
  int x1 = cl(x0 + 1, ext.x), y1 = cl(y0 + 1, ext.y), z1 = cl(z0 + 1, ext.z);
  x0 = cl(x0, ext.x); y0 = cl(y0, ext.y); z0 = cl(z0, ext.z);
  size_t sy = ext.x, sz = (size_t)ext.x * ext.y;
  auto F = [&](int xx, int yy, int zz) { return sdf[(size_t)xx + (size_t)yy * sy + (size_t)zz * sz]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x1, y0, z0) * wx;
  double c10 = F(x0, y1, z0) * (1 - wx) + F(x1, y1, z0) * wx;
  double c01 = F(x0, y0, z1) * (1 - wx) + F(x1, y0, z1) * wx;
  double c11 = F(x0, y1, z1) * (1 - wx) + F(x1, y1, z1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy;
  double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

__device__ inline double cc_face_open_ext(const double* sdf, int3 ext, double fx, double fy,
                                          double fz, int type, double dx, double dy, double dz) {
  double sd = cc_sample_ext(sdf, ext, fx, fy, fz);
  if (sd <= 0.0) return 0.0;
  double e = 1.0;
  return cc_fraction_core(sd, cc_sample_ext(sdf, ext, fx + e, fy, fz),
                          cc_sample_ext(sdf, ext, fx - e, fy, fz),
                          cc_sample_ext(sdf, ext, fx, fy + e, fz),
                          cc_sample_ext(sdf, ext, fx, fy - e, fz),
                          cc_sample_ext(sdf, ext, fx, fy, fz + e),
                          cc_sample_ext(sdf, ext, fx, fy, fz - e), type, dx, dy, dz);
}

// Fill the staggered face openness for every extended cell: ox[i] = openness of the -x face of cell i
// (face centre at local (lx-0.5, ly, lz)), oy/oz analogous. Launch over the whole extended block.
__global__ void cc_build_open_k(double* ox, double* oy, double* oz, const double* sdf, int3 ext,
                                double dx, double dy, double dz) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
  ox[i] = cc_face_open_ext(sdf, ext, lx - 0.5, ly, lz, 1, dx, dy, dz);
  oy[i] = cc_face_open_ext(sdf, ext, lx, ly - 0.5, lz, 2, dx, dy, dz);
  oz[i] = cc_face_open_ext(sdf, ext, lx, ly, lz - 0.5, 3, dx, dy, dz);
}

}  // namespace ccdetail
}  // namespace cfdmpi
