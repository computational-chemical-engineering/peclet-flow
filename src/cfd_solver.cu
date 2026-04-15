#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "cfd_solver.cuh"
#include "cfd_solver_ibm_kernels.cuh"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda/functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  }

// --------------------------------------------------------
// Device Helper: 3D Periodic Indexing
// --------------------------------------------------------
// Defined in cfd_solver.cuh

// Forward Declarations
// Note: compute_ibm_geometry_kernel is templated and defined in header.
static void free_ibm_storage(IBM_Data &data);
static void alloc_ibm_storage(IBM_Data &data, size_t n);
static void replace_int_buffer(int *&ptr, size_t n);
static void validate_compact_ibm_host(const char *label, const IBM_Data &data,
                                      const int *ibm_id_map, int num_elements);
static void validate_fluid_indices_host(const char *label,
                                        const int *fluid_indices,
                                        int num_fluid_cells,
                                        const int *ibm_id_map,
                                        int num_elements);

__global__ void
modify_stencil_ibm_kernel(float *__restrict__ A_C, float *__restrict__ A_W,
                          float *__restrict__ A_E, float *__restrict__ A_S,
                          float *__restrict__ A_N, float *__restrict__ A_B,
                          float *__restrict__ A_T, float *__restrict__ a_inhom,
                          IBM_Data ibm_data, float u_bc_val);

__global__ void populate_ibm_scaling_kernel(float *__restrict__ d_rescale,
                                            IBM_Data ibm_data,
                                            int num_elements);

__global__ void apply_ibm_scaling_kernel(float *vector, IBM_Data ibm_data,
                                         int num_elements);

template <typename T>
__global__ void solve_rbgs_stencil_kernel(
    T *__restrict__ phi, const float *__restrict__ A_C,
    const float *__restrict__ A_W, const float *__restrict__ A_E,
    const float *__restrict__ A_S, const float *__restrict__ A_N,
    const float *__restrict__ A_B, const float *__restrict__ A_T,
    const float *__restrict__ B_RHS, int3 res, bool is_red);

__global__ void project_velocity_kernel(
    double *__restrict__ u, double *__restrict__ v, double *__restrict__ w,
    const float *__restrict__ p, const float *__restrict__ frac_u,
    const float *__restrict__ frac_v, const float *__restrict__ frac_w,
    const float *__restrict__ sdf, int3 res, float3 spacing);

__global__ void apply_correction_kernel(double *__restrict__ u,
                                        const float *__restrict__ du,
                                        int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  u[idx] += (double)du[idx];
}

__global__ void scale_field_kernel(double *__restrict__ field, double scale,
                                   int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  field[idx] *= scale;
}

// Helper: Trilinear Interpolation with Periodic Wrap
__device__ float sample_field(const float *__restrict__ field, float x, float y,
                              float z, int3 res) {
  // 1. Wrap coords to [0, dim)
  float rx = fmodf(x, (float)res.x);
  if (rx < 0)
    rx += res.x;
  float ry = fmodf(y, (float)res.y);
  if (ry < 0)
    ry += res.y;
  float rz = fmodf(z, (float)res.z);
  if (rz < 0)
    rz += res.z;

  // 2. Base indices
  int x0 = (int)floorf(rx);
  int y0 = (int)floorf(ry);
  int z0 = (int)floorf(rz);

  // 3. Fractions
  float fx = rx - x0;
  float fy = ry - y0;
  float fz = rz - z0;

  // 4. Neighbors (x0, x1, etc.)
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  // 5. Fetch 8 samples (using get_idx for wrapping x1,y1,z1)
  float c000 = field[get_idx(x0, y0, z0, res)];
  float c100 = field[get_idx(x1, y0, z0, res)];
  float c010 = field[get_idx(x0, y1, z0, res)];
  float c110 = field[get_idx(x1, y1, z0, res)];

  float c001 = field[get_idx(x0, y0, z1, res)];
  float c101 = field[get_idx(x1, y0, z1, res)];
  float c011 = field[get_idx(x0, y1, z1, res)];
  float c111 = field[get_idx(x1, y1, z1, res)];

  // 6. Interpolate
  float c00 = c000 * (1.0f - fx) + c100 * fx;
  float c10 = c010 * (1.0f - fx) + c110 * fx;
  float c01 = c001 * (1.0f - fx) + c101 * fx;
  float c11 = c011 * (1.0f - fx) + c111 * fx;

  float c0 = c00 * (1.0f - fy) + c10 * fy;
  float c1 = c01 * (1.0f - fy) + c11 * fy;

  return c0 * (1.0f - fz) + c1 * fz;
}

// --------------------------------------------------------
// SDF Sampling Helpers (local copy to avoid cross-TU device linkage)
// --------------------------------------------------------
__device__ inline float sample_sdf_interp_local(float x, float y, float z,
                                                const float *__restrict__ sdf,
                                                int3 res) {
  float fx = floorf(x);
  float fy = floorf(y);
  float fz = floorf(z);

  float wx = x - fx;
  float wy = y - fy;
  float wz = z - fz;

  int ix = (int)fx;
  int iy = (int)fy;
  int iz = (int)fz;

  int x0 = (ix % res.x + res.x) % res.x;
  int y0 = (iy % res.y + res.y) % res.y;
  int z0 = (iz % res.z + res.z) % res.z;

  int x1 = (x0 + 1) % res.x;
  int y1 = (y0 + 1) % res.y;
  int z1 = (z0 + 1) % res.z;

  float c000 = sdf[z0 * res.y * res.x + y0 * res.x + x0];
  float c100 = sdf[z0 * res.y * res.x + y0 * res.x + x1];
  float c010 = sdf[z0 * res.y * res.x + y1 * res.x + x0];
  float c110 = sdf[z0 * res.y * res.x + y1 * res.x + x1];
  float c001 = sdf[z1 * res.y * res.x + y0 * res.x + x0];
  float c101 = sdf[z1 * res.y * res.x + y0 * res.x + x1];
  float c011 = sdf[z1 * res.y * res.x + y1 * res.x + x0];
  float c111 = sdf[z1 * res.y * res.x + y1 * res.x + x1];

  float c00 = c000 * (1.0f - wx) + c100 * wx;
  float c10 = c010 * (1.0f - wx) + c110 * wx;
  float c01 = c001 * (1.0f - wx) + c101 * wx;
  float c11 = c011 * (1.0f - wx) + c111 * wx;

  float c0 = c00 * (1.0f - wy) + c10 * wy;
  float c1 = c01 * (1.0f - wy) + c11 * wy;

  return c0 * (1.0f - wz) + c1 * wz;
}

__device__ inline float sample_sdf_component(const float *__restrict__ sdf,
                                             int3 res, int idx_x, int idx_y,
                                             int idx_z, int comp_idx) {
  float3 offset = make_float3(0.0f, 0.0f, 0.0f);
  if (comp_idx == 0) {
    offset = make_float3(-0.5f, 0.0f, 0.0f);
  } else if (comp_idx == 1) {
    offset = make_float3(0.0f, -0.5f, 0.0f);
  } else {
    offset = make_float3(0.0f, 0.0f, -0.5f);
  }
  return sample_sdf_interp_local(idx_x + offset.x, idx_y + offset.y,
                                 idx_z + offset.z, sdf, res);
}

__device__ inline double
get_face_velocity_for_flux(const double *__restrict__ vel,
                           const float *__restrict__ sdf, int3 res,
                           float3 offset, int idx_x, int idx_y, int idx_z,
                           float frac, float bc_val, int axis) {
  if (frac <= 0.0f) {
    return 0.0f;
  }

  float sdf_face = sample_sdf_interp_local(idx_x + offset.x, idx_y + offset.y,
                                           idx_z + offset.z, sdf, res);
  float face_val = vel[get_idx(idx_x, idx_y, idx_z, res)];
  if (sdf_face >= 0.0f) {
    return face_val;
  }

  int dx = (axis == 0) ? 1 : 0;
  int dy = (axis == 1) ? 1 : 0;
  int dz = (axis == 2) ? 1 : 0;

  float sdf_p =
      sample_sdf_interp_local(idx_x + dx + offset.x, idx_y + dy + offset.y,
                              idx_z + dz + offset.z, sdf, res);
  if (sdf_p >= 0.0f) {
    return vel[get_idx(idx_x + dx, idx_y + dy, idx_z + dz, res)];
  }

  float sdf_m =
      sample_sdf_interp_local(idx_x - dx + offset.x, idx_y - dy + offset.y,
                              idx_z - dz + offset.z, sdf, res);
  if (sdf_m >= 0.0f) {
    return vel[get_idx(idx_x - dx, idx_y - dy, idx_z - dz, res)];
  }

  return bc_val;
}

// --------------------------------------------------------
// Velocity Mask Kernel (Face-Centered SDF)
// --------------------------------------------------------
// Sets u, v, w to 0 if the face center lies inside solid (SDF < 0).
__global__ void apply_face_sdf_mask_kernel(double *__restrict__ u,
                                           double *__restrict__ v,
                                           double *__restrict__ w,
                                           const float *__restrict__ sdf,
                                           int3 res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  float sdf_u = sample_sdf_interp_local(idx_x - 0.5f, idx_y, idx_z, sdf, res);
  if (sdf_u <= 0.0f) {
    u[get_idx(idx_x, idx_y, idx_z, res)] = 0.0f;
  }

  float sdf_v = sample_sdf_interp_local(idx_x, idx_y - 0.5f, idx_z, sdf, res);
  if (sdf_v <= 0.0f) {
    v[get_idx(idx_x, idx_y, idx_z, res)] = 0.0f;
  }

  float sdf_w = sample_sdf_interp_local(idx_x, idx_y, idx_z - 0.5f, sdf, res);
  if (sdf_w <= 0.0f) {
    w[get_idx(idx_x, idx_y, idx_z, res)] = 0.0f;
  }
}

// --------------------------------------------------------
// Max Velocity Kernel (for CFL)
// --------------------------------------------------------
__device__ float atomicMaxFloat(float *addr, float val) {
  int *addr_as_int = (int *)addr;
  int old = *addr_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void compute_max_velocity_kernel(const double *__restrict__ u,
                                            const double *__restrict__ v,
                                            const double *__restrict__ w,
                                            float *max_vel, int3 res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Velocity magnitude squared or L1 norm?
  // Schemes usually bounded by max(|u|, |v|, |w|).
  // Let's compute max(|u|, |v|, |w|) per cell since our CFL is dx/u_max.
  // Actually, we should check u/dx, v/dy, w/dz?
  // Assuming dx=dy=dz for now or taking simplified max.
  // Conservative: max(|u|, |v|, |w|)

  float u_val = fabsf((float)u[idx]);
  float v_val = fabsf((float)v[idx]);
  float w_val = fabsf((float)w[idx]);

  float local_max = fmaxf(u_val, fmaxf(v_val, w_val));

  // Reduce to global max
  // Simple atomic for now. number of collisions might be high but it's executed
  // once per step. Shared memory reduction would be better for perf.
  atomicMaxFloat(max_vel, local_max);
}

// --------------------------------------------------------
// Red-Black Gauss-Seidel Kernel (with Neumann BC for Solids)
// --------------------------------------------------------
// is_red: true if we update Red nodes ((x+y+z)%2 == 0), false for Black
// --------------------------------------------------------
// Red-Black Gauss-Seidel Kernel (IBM Modified)
// --------------------------------------------------------
// --------------------------------------------------------
// Base Pressure Stencil Kernel
// --------------------------------------------------------
// Computes the standard Geometric Laplacian coefficients into A_*, B
__global__ void compute_pressure_stencil_kernel(
    float *__restrict__ A_C, float *__restrict__ A_W, float *__restrict__ A_E,
    float *__restrict__ A_S, float *__restrict__ A_N, float *__restrict__ A_B,
    float *__restrict__ A_T, float *__restrict__ B_RHS,
    const float *__restrict__ frac_u, const float *__restrict__ frac_v,
    const float *__restrict__ frac_w, const float *__restrict__ sdf,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, int3 res, float3 spacing) {

  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  float inv_dx2 = 1.0f / (spacing.x * spacing.x);
  float inv_dy2 = 1.0f / (spacing.y * spacing.y);
  float inv_dz2 = 1.0f / (spacing.z * spacing.z);

  // Neighbors indices
  int idx_xp = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_xm = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_yp = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_ym = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_zp = get_idx(idx_x, idx_y, idx_z + 1, res);
  int idx_zm = get_idx(idx_x, idx_y, idx_z - 1, res);

  // Face-centered SDF classification for pressure coupling
  float sdf_u_r = sample_sdf_interp_local(idx_x + 0.5f, idx_y, idx_z, sdf, res);
  float sdf_u_l = sample_sdf_interp_local(idx_x - 0.5f, idx_y, idx_z, sdf, res);
  float sdf_v_n = sample_sdf_interp_local(idx_x, idx_y + 0.5f, idx_z, sdf, res);
  float sdf_v_s = sample_sdf_interp_local(idx_x, idx_y - 0.5f, idx_z, sdf, res);
  float sdf_w_t = sample_sdf_interp_local(idx_x, idx_y, idx_z + 0.5f, sdf, res);
  float sdf_w_b = sample_sdf_interp_local(idx_x, idx_y, idx_z - 0.5f, sdf, res);

  float frac_u_r = (sdf_u_r > 0.0f) ? frac_u[idx_xp] : 0.0f;
  float frac_u_l = (sdf_u_l > 0.0f) ? frac_u[idx] : 0.0f;
  float frac_v_n = (sdf_v_n > 0.0f) ? frac_v[idx_yp] : 0.0f;
  float frac_v_s = (sdf_v_s > 0.0f) ? frac_v[idx] : 0.0f;
  float frac_w_t = (sdf_w_t > 0.0f) ? frac_w[idx_zp] : 0.0f;
  float frac_w_b = (sdf_w_b > 0.0f) ? frac_w[idx] : 0.0f;

  float ax_p = frac_u_r * inv_dx2;
  float ax_m = frac_u_l * inv_dx2;
  float ay_p = frac_v_n * inv_dy2;
  float ay_m = frac_v_s * inv_dy2;
  float az_p = frac_w_t * inv_dz2;
  float az_m = frac_w_b * inv_dz2;

  double du_dx = (u[idx_xp] * frac_u_r - u[idx] * frac_u_l) / spacing.x;
  double dv_dy = (v[idx_yp] * frac_v_n - v[idx] * frac_v_s) / spacing.y;
  double dw_dz = (w[idx_zp] * frac_w_t - w[idx] * frac_w_b) / spacing.z;

  double div = du_dx + dv_dy + dw_dz;
  A_E[idx] = -ax_p;
  A_W[idx] = -ax_m;
  A_N[idx] = -ay_p;
  A_S[idx] = -ay_m;
  A_T[idx] = -az_p;
  A_B[idx] = -az_m;

  // A_C is sum of absolute values of neighbors (Diagonal Dominance)
  A_C[idx] = ax_p + ax_m + ay_p + ay_m + az_p + az_m;
  // Initialize RHS
  B_RHS[idx] = (float)(-div);
}

// Device Helper: Get Advection Velocity (Interpolated)
// --------------------------------------------------------
// comp_idx: 0=U, 1=V, 2=W (Which equation we are solving)
// direction: 0=X, 1=Y, 2=Z (Which advection term: d/dx, d/dy, d/dz)
// idx: target cell index (i, j, k)
// Returns: Advecting velocity at the corresponding face
//
// Grid Staggering:
// U is at (i, j+1/2, k+1/2)
// V is at (i+1/2, j, k+1/2)
// W is at (i+1/2, j+1/2, k)
// Helper: Get Advection Velocity (Interpolated) - Double precision for accuracy
__device__ inline double get_advection_velocity(const double *__restrict__ u,
                                                const double *__restrict__ v,
                                                const double *__restrict__ w,
                                                int comp_idx, int face_dir,
                                                int3 idx, int3 res) {
  // comp_idx: 0=U, 1=V, 2=W (Scalar being advected)
  // face_dir: 0=X, 1=Y, 2=Z (Direction of flux)
  // idx: Indices of the Scalar Cell (Control Volume)
  // We want velocity component `face_dir` at the `face_dir`+ face of the CV.

  int idx_x = idx.x;
  int idx_y = idx.y;
  int idx_z = idx.z;

  if (comp_idx == 0) {   // U-Momentum
    if (face_dir == 0) { // X-Face (x=i+0.5)
      double u_c = u[get_idx(idx_x, idx_y, idx_z, res)];
      double u_e = u[get_idx(idx_x + 1, idx_y, idx_z, res)];
      return 0.5 * (u_c + u_e);
    } else if (face_dir == 1) { // Y-Face (y=j+1)
      double v_l = v[get_idx(idx_x - 1, idx_y + 1, idx_z, res)];
      double v_r = v[get_idx(idx_x, idx_y + 1, idx_z, res)];
      return 0.5 * (v_l + v_r);
    } else { // Z-Face (z=k+1)
      double w_l = w[get_idx(idx_x - 1, idx_y, idx_z + 1, res)];
      double w_r = w[get_idx(idx_x, idx_y, idx_z + 1, res)];
      return 0.5 * (w_l + w_r);
    }
  }

  if (comp_idx == 1) {   // V-Momentum
    if (face_dir == 0) { // X-Face (x=i+1)
      double u_s = u[get_idx(idx_x + 1, idx_y - 1, idx_z, res)];
      double u_n = u[get_idx(idx_x + 1, idx_y, idx_z, res)];
      return 0.5 * (u_s + u_n);
    } else if (face_dir == 1) { // Y-Face (y=j+0.5)
      double v_c = v[get_idx(idx_x, idx_y, idx_z, res)];
      double v_n = v[get_idx(idx_x, idx_y + 1, idx_z, res)];
      return 0.5 * (v_c + v_n);
    } else { // Z-Face (z=k+1)
      double w_s = w[get_idx(idx_x, idx_y - 1, idx_z + 1, res)];
      double w_n = w[get_idx(idx_x, idx_y, idx_z + 1, res)];
      return 0.5 * (w_s + w_n);
    }
  }

  if (comp_idx == 2) {   // W-Momentum
    if (face_dir == 0) { // X-Face (x=i+1)
      double u_b = u[get_idx(idx_x + 1, idx_y, idx_z - 1, res)];
      double u_t = u[get_idx(idx_x + 1, idx_y, idx_z, res)];
      return 0.5 * (u_b + u_t);
    } else if (face_dir == 1) { // Y-Face (y=j+1)
      double v_b = v[get_idx(idx_x, idx_y + 1, idx_z - 1, res)];
      double v_t = v[get_idx(idx_x, idx_y + 1, idx_z, res)];
      return 0.5 * (v_b + v_t);
    } else { // Z-Face (z=k+0.5)
      double w_c = w[get_idx(idx_x, idx_y, idx_z, res)];
      double w_f = w[get_idx(idx_x, idx_y, idx_z + 1, res)];
      return 0.5 * (w_c + w_f);
    }
  }

  return 0.0;
}

// Helper: Koren Limiter Flux
// u_face: Advecting velocity
// phi_up_m1: i-1 (Upstream - 1)
// phi_up:    i   (Upstream)
// phi_down:  i+1 (Downstream)
// (Indices relative to flow direction)
__device__ inline float tvd_flux_koren(float phi_up_m1, float phi_up,
                                       float phi_down, float u_face) {
  // Gradient Ratio r
  float num = phi_up - phi_up_m1;
  float den = phi_down - phi_up;

  // Avoid division by zero
  float r = (fabsf(den) < 1e-10f) ? 0.0f : num / den;
  if (fabsf(den) < 1e-10f && fabsf(num) < 1e-10f)
    r = 1.0f; // Uniform region

  // Koren Limiter Psi(r)
  // max(0, min(2r, min((1+2r)/3, 2)))
  float psi =
      fmaxf(0.0f, fminf(2.0f * r, fminf((1.0f + 2.0f * r) / 3.0f, 2.0f)));

  // TVD Flux = FOU + 0.5 * Psi(r) * (F_CDS - F_FOU)
  // F_CDS - F_FOU = u * 0.5 * (phi_down - phi_up)
  // So Flux = u * (phi_up + 0.5 * psi * (phi_down - phi_up))
  return u_face * (phi_up + 0.5f * psi * (phi_down - phi_up));
}

// Wrapper to handle direction
__device__ inline float get_tvd_flux(float phi_LL, float phi_L, float phi_R,
                                     float phi_RR, float u_face) {
  if (u_face > 0.0f) {
    // Flow L -> R. Upstream: L, LL. Downstream: R.
    return tvd_flux_koren(phi_LL, phi_L, phi_R, u_face);
  } else {
    // Flow R -> L. Upstream: R, RR. Downstream: L.
    return tvd_flux_koren(phi_RR, phi_R, phi_L, u_face);
  }
}

__device__ inline float get_ibm_ratio(const int *__restrict__ ibm_id_map,
                                      IBM_Data ibm_data, int idx, int dir) {
  if (ibm_id_map == nullptr) {
    return 1.0f;
  }
  int list_idx = ibm_id_map[idx];
  if (list_idx < 0) {
    return 1.0f;
  }
  return ibm_data.R_val[list_idx * 6 + dir];
}

// --------------------------------------------------------
// Generic RB-GS Stencil Solver
// --------------------------------------------------------
template <typename T>
__global__ void solve_rbgs_stencil_kernel(
    T *__restrict__ phi, const float *__restrict__ A_C,
    const float *__restrict__ A_W, const float *__restrict__ A_E,
    const float *__restrict__ A_S, const float *__restrict__ A_N,
    const float *__restrict__ A_B, const float *__restrict__ A_T,
    const float *__restrict__ B_RHS, int3 res, bool is_red) {

  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int sum_coord = idx_x + idx_y + idx_z;
  if ((sum_coord % 2 == 0) != is_red)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Load Coefficients
  float ac = A_C[idx];
  // If ac is zero (or close), don't divide.
  if (fabsf(ac) < 1e-12f)
    return;

  // Neighbors indices
  int idx_E = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_W = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_N = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_S = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_T = get_idx(idx_x, idx_y, idx_z + 1, res);
  int idx_B = get_idx(idx_x, idx_y, idx_z - 1, res);

  // Compute SumNeighbors
  // eq: A_c * phi_c + Sum( A_nb * phi_nb ) = B
  // phi_c = (B - Sum) / A_c

  T sum = A_E[idx] * phi[idx_E] + A_W[idx] * phi[idx_W] +
          A_N[idx] * phi[idx_N] + A_S[idx] * phi[idx_S] +
          A_T[idx] * phi[idx_T] + A_B[idx] * phi[idx_B];

  phi[idx] = (B_RHS[idx] - sum) / ac;
}

// ... Projection Kernel ... (Unchanged)
__global__ void project_velocity_kernel(
    double *__restrict__ u, double *__restrict__ v, double *__restrict__ w,
    const float *__restrict__ p, const float *__restrict__ frac_u,
    const float *__restrict__ frac_v, const float *__restrict__ frac_w,
    const float *__restrict__ sdf, int3 res, float3 spacing) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Paper convention: u = u* - ∇φ (no dt/rho factor)
  // The ρ/(θΔt) factor is in the pressure update (Eq. 14)
  float scale = 1.0f;

  // Use standard gradients - the frac-weighted Laplacian in the Poisson solve
  // ensures divergence-free result. If frac=0, face is blocked and u is set to
  // 0.

  {
    float sdf_u = sample_sdf_interp_local(idx_x - 0.5f, idx_y, idx_z, sdf, res);
    if (sdf_u > 0.0f && frac_u[idx] > 0.0f) {
      int idx_prev = get_idx(idx_x - 1, idx_y, idx_z, res);
      float dp_dx = (p[idx] - p[idx_prev]) / spacing.x;
      u[idx] -= scale * dp_dx;
    } else {
      u[idx] = 0.0f;
    }
  }
  {
    float sdf_v = sample_sdf_interp_local(idx_x, idx_y - 0.5f, idx_z, sdf, res);
    if (sdf_v > 0.0f && frac_v[idx] > 0.0f) {
      int idy_prev = get_idx(idx_x, idx_y - 1, idx_z, res);
      float dp_dy = (p[idx] - p[idy_prev]) / spacing.y;
      v[idx] -= scale * dp_dy;
    } else {
      v[idx] = 0.0f;
    }
  }
  {
    float sdf_w = sample_sdf_interp_local(idx_x, idx_y, idx_z - 0.5f, sdf, res);
    if (sdf_w > 0.0f && frac_w[idx] > 0.0f) {
      int idz_prev = get_idx(idx_x, idx_y, idx_z - 1, res);
      float dp_dz = (p[idx] - p[idz_prev]) / spacing.z;
      w[idx] -= scale * dp_dz;
    } else {
      w[idx] = 0.0f;
    }
  }
}

// --------------------------------------------------------
// Pressure Update from Phi Kernel (Incremental Pressure Correction)
// --------------------------------------------------------
// Implements Eq. 14 from Robust_Scaled_IBM_Solver.tex directly:
//
//   δp = (ρ/(θΔt))*φ + ρ*(u·∇φ) - μ*∇²φ
//
// where φ satisfies the Poisson equation: ∇²φ = div(u)
//
// This formulation is appropriate for large Δt (steady-state simulations)
// because the (ρ/(θΔt)) term naturally diminishes as Δt → ∞.
__global__ void update_pressure_from_phi_kernel(
    double *__restrict__ p, const float *__restrict__ phi,
    const double *__restrict__ p_old, const double *__restrict__ u,
    const double *__restrict__ v, const double *__restrict__ w,
    const float *__restrict__ rhs_phi, int3 res, float3 spacing, float dt,
    float theta, float rho, float mu) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Simple pressure update: δp = (ρ/Δt)*φ
  // The convection and diffusion terms in Eq. 14 can cause instability
  // when the Poisson equation doesn't account for them consistently.
  // The basic projection uses just the temporal term.
  double delta_p = (double)((rho / dt) * phi[idx] + mu * rhs_phi[idx]);

  // Update pressure: p = p_old + δp
  p[idx] = p[idx] + delta_p;
}

__global__ void shift_pressure_kernel(double *__restrict__ p, int num_elements,
                                      double offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  p[idx] -= offset;
}

template <int Component>
__global__ void add_brinkman_drag_kernel(
    float *__restrict__ residual, float *__restrict__ A_C,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const float *__restrict__ vol_frac, int3 res,
    float3 spacing, float theta, float mu) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  const int idx = get_idx(idx_x, idx_y, idx_z, res);
  const float inv_dx2 = 1.0f / (spacing.x * spacing.x);
  const float inv_dy2 = 1.0f / (spacing.y * spacing.y);
  const float inv_dz2 = 1.0f / (spacing.z * spacing.z);
  const float fluid = fminf(fmaxf(vol_frac[idx], 0.0f), 1.0f);
  const float solid = 1.0f - fluid;
  const float diff_diag = 2.0f * mu * (inv_dx2 + inv_dy2 + inv_dz2);

  const float drag = diff_diag * solid / fmaxf(fluid, 5.0e-2f);
  const double vel =
      (Component == 0 ? u[idx] : (Component == 1 ? v[idx] : w[idx]));

  A_C[idx] += theta * drag;
  residual[idx] -= static_cast<float>(theta * drag * vel);
}

CFDSolver::CFDSolver(int3 res, float3 spacing)
    : num_elements(res.x * res.y * res.z), rho_(1.0f), mu_(1.0f), pin_idx(0),
      v_max_iter_(5), p_max_iter_(50), theta_(1.0f), outer_iterations_(2),
      outer_tol_(-1.0f) {
  grid.res = res;
  grid.spacing = spacing;
  grid.num_elements = num_elements; // Fix: Initialize grid member
  grid.body_force_density_ = make_float3(0.0f, 0.0f, 0.0f);
  grid.u_bc_ = make_float3(0.0f, 0.0f, 0.0f);

  // Allocate MacGrid arrays
  CHECK_CUDA(cudaMalloc(&grid.u, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.v, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.w, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.p, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.rhs, num_elements * sizeof(float)));

  // Stencil Arrays
  CHECK_CUDA(cudaMalloc(&grid.A_C, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_W, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_E, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_S, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_N, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_B, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.A_T, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.B_RHS, num_elements * sizeof(float)));

  // Newton-Raphson Buffers
  CHECK_CUDA(cudaMalloc(&grid.u_old, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.v_old, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.w_old, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.p_old, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.p_prev, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.res_u, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_v, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_w, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_u_pre, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_v_pre, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_w_pre, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_u_post, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_v_post, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.res_w_post, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.explicit_u, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.explicit_v, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.explicit_w, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&grid.phi, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.du, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.dv, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.dw, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.div_pre, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.div_post, num_elements * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&grid.sdf, num_elements * sizeof(float)));

  // Initialize to zero
  CHECK_CUDA(cudaMemset(grid.u, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.v, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.w, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.p, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.rhs, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.sdf, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.u_old, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.v_old, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.w_old, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.p_old, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.p_prev, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.res_u, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_v, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_w, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_u_pre, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_v_pre, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_w_pre, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_u_post, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_v_post, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.res_w_post, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.explicit_u, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.explicit_v, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.explicit_w, 0, num_elements * sizeof(double)));
  CHECK_CUDA(cudaMemset(grid.phi, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.du, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.dv, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.dw, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.div_pre, 0, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(grid.div_post, 0, num_elements * sizeof(float)));

  // Surface Fractions
  CHECK_CUDA(cudaMalloc(&grid.frac_u, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.frac_v, num_elements * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&grid.frac_w, num_elements * sizeof(float)));

  // IBM Allocations (SoA)
  size_t n = num_elements;
  CHECK_CUDA(cudaMalloc(&grid.ibm_id_map, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&grid.d_inhom_scratch, n * sizeof(float)));

  // Staggered IBM U
  CHECK_CUDA(cudaMalloc(&grid.ibm_id_map_u, n * sizeof(int)));

  // Staggered IBM V
  CHECK_CUDA(cudaMalloc(&grid.ibm_id_map_v, n * sizeof(int)));

  // Staggered IBM W
  CHECK_CUDA(cudaMalloc(&grid.ibm_id_map_w, n * sizeof(int)));

  // Allocate fluid indices (will be populated later)
  grid.ibm_data = IBM_Data{};
  grid.ibm_data_u = IBM_Data{};
  grid.ibm_data_v = IBM_Data{};
  grid.ibm_data_w = IBM_Data{};
  grid.fluid_indices_u = nullptr;
  grid.fluid_indices_v = nullptr;
  grid.fluid_indices_w = nullptr;

  grid.num_ibm_cells = 0;
  grid.num_ibm_cells_u = 0;
  grid.num_ibm_cells_v = 0;
  grid.num_ibm_cells_w = 0;
  // pin_idx is set in initialize() using the max SDF cell.
  // For Neumann BC problems, pressure is only defined up to a constant.
  pin_idx = 0;
}

CFDSolver::~CFDSolver() {
  free_pressure_multigrid();
  free_velocity_multigrid();
  CHECK_CUDA(cudaFree(grid.u));
  CHECK_CUDA(cudaFree(grid.v));
  CHECK_CUDA(cudaFree(grid.w));
  CHECK_CUDA(cudaFree(grid.p));
  CHECK_CUDA(cudaFree(grid.rhs));

  CHECK_CUDA(cudaFree(grid.A_C));
  CHECK_CUDA(cudaFree(grid.A_W));
  CHECK_CUDA(cudaFree(grid.A_E));
  CHECK_CUDA(cudaFree(grid.A_S));
  CHECK_CUDA(cudaFree(grid.A_N));
  CHECK_CUDA(cudaFree(grid.A_B));
  CHECK_CUDA(cudaFree(grid.A_T));
  CHECK_CUDA(cudaFree(grid.B_RHS));

  CHECK_CUDA(cudaFree(grid.u_old));
  CHECK_CUDA(cudaFree(grid.v_old));
  CHECK_CUDA(cudaFree(grid.w_old));
  CHECK_CUDA(cudaFree(grid.p_old));
  CHECK_CUDA(cudaFree(grid.p_prev));
  CHECK_CUDA(cudaFree(grid.res_u));
  CHECK_CUDA(cudaFree(grid.res_v));
  CHECK_CUDA(cudaFree(grid.res_w));
  CHECK_CUDA(cudaFree(grid.res_u_pre));
  CHECK_CUDA(cudaFree(grid.res_v_pre));
  CHECK_CUDA(cudaFree(grid.res_w_pre));
  CHECK_CUDA(cudaFree(grid.res_u_post));
  CHECK_CUDA(cudaFree(grid.res_v_post));
  CHECK_CUDA(cudaFree(grid.res_w_post));
  CHECK_CUDA(cudaFree(grid.explicit_u));
  CHECK_CUDA(cudaFree(grid.explicit_v));
  CHECK_CUDA(cudaFree(grid.explicit_w));
  CHECK_CUDA(cudaFree(grid.phi));
  CHECK_CUDA(cudaFree(grid.du));
  CHECK_CUDA(cudaFree(grid.dv));
  CHECK_CUDA(cudaFree(grid.dw));
  CHECK_CUDA(cudaFree(grid.div_pre));
  CHECK_CUDA(cudaFree(grid.div_post));
  CHECK_CUDA(cudaFree(grid.sdf));

  CHECK_CUDA(cudaFree(grid.frac_u));
  CHECK_CUDA(cudaFree(grid.frac_v));
  CHECK_CUDA(cudaFree(grid.frac_w));

  if (grid.ibm_id_map != nullptr)
    CHECK_CUDA(cudaFree(grid.ibm_id_map));
  CHECK_CUDA(cudaFree(grid.d_inhom_scratch));

  free_ibm_storage(grid.ibm_data);

  // Free Staggered
  if (grid.ibm_id_map_u != nullptr)
    CHECK_CUDA(cudaFree(grid.ibm_id_map_u));
  free_ibm_storage(grid.ibm_data_u);

  if (grid.ibm_id_map_v != nullptr)
    CHECK_CUDA(cudaFree(grid.ibm_id_map_v));
  free_ibm_storage(grid.ibm_data_v);

  if (grid.ibm_id_map_w != nullptr)
    CHECK_CUDA(cudaFree(grid.ibm_id_map_w));
  free_ibm_storage(grid.ibm_data_w);

  if (grid.fluid_indices_u != nullptr)
    CHECK_CUDA(cudaFree(grid.fluid_indices_u));
  if (grid.fluid_indices_v != nullptr)
    CHECK_CUDA(cudaFree(grid.fluid_indices_v));
  if (grid.fluid_indices_w != nullptr)
    CHECK_CUDA(cudaFree(grid.fluid_indices_w));
}

void CFDSolver::set_theta_(float theta) { theta_ = theta; }
void CFDSolver::set_rho(float rho) { rho_ = rho; }
void CFDSolver::set_mu(float mu) { mu_ = mu; }

void CFDSolver::set_pressure_solver_params(int iter) { p_max_iter_ = iter; }

void CFDSolver::set_velocity_solver_params(int iter) { v_max_iter_ = iter; }

void CFDSolver::set_debug_stats(bool enabled) {
  debug_stats_enabled_ = enabled;
}

std::vector<float> CFDSolver::get_debug_stats() const {
  std::vector<float> stats;
  stats.reserve(11);
  for (int i = 0; i < 3; ++i) {
    stats.push_back(debug_stats_.res_before[i]);
  }
  for (int i = 0; i < 3; ++i) {
    stats.push_back(debug_stats_.res_after[i]);
  }
  for (int i = 0; i < 3; ++i) {
    stats.push_back(debug_stats_.corr_max[i]);
  }
  stats.push_back(debug_stats_.div_before);
  stats.push_back(debug_stats_.div_after);
  return stats;
}

std::vector<std::vector<float>> CFDSolver::get_debug_fields() const {
  std::vector<std::vector<float>> fields(8);
  for (auto &field : fields) {
    field.resize(num_elements);
  }

  CHECK_CUDA(cudaMemcpy(fields[0].data(), grid.res_u_pre,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[1].data(), grid.res_v_pre,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[2].data(), grid.res_w_pre,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[3].data(), grid.res_u_post,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[4].data(), grid.res_v_post,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[5].data(), grid.res_w_post,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[6].data(), grid.div_pre,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(fields[7].data(), grid.div_post,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));

  return fields;
}

void CFDSolver::set_debug_cell(int3 cell) {
  debug_cell_ = cell;
  debug_cell_enabled_ = true;
  debug_cell_info_.assign(59, 0.0f);
}

std::vector<float> CFDSolver::get_debug_cell_info() const {
  return debug_cell_info_;
}

float CFDSolver::compute_max_velocity() {

  float *d_max_vel;
  CHECK_CUDA(cudaMalloc(&d_max_vel, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_max_vel, 0, sizeof(float)));

  dim3 threads(256);
  dim3 blocks((num_elements + 255) / 256); // 1D layout is fine for reduction

  compute_max_velocity_kernel<<<blocks, threads>>>(grid.u, grid.v, grid.w,
                                                   d_max_vel, grid.res);
  CHECK_CUDA(cudaGetLastError());

  float h_max_vel;
  CHECK_CUDA(
      cudaMemcpy(&h_max_vel, d_max_vel, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_max_vel));

  return h_max_vel;
}

struct AbsValue {
  __host__ __device__ float operator()(float value) const {
    return fabsf(value);
  }
};

struct IndexedSquareValue {
  const float *data;
  const int *indices;

  __host__ __device__ double operator()(int i) const {
    double value = static_cast<double>(data[indices[i]]);
    return value * value;
  }
};

static float max_abs_device(const float *d_data, size_t n) {
  if (n == 0) {
    return 0.0f;
  }
  thrust::device_ptr<const float> ptr(d_data);
  return thrust::transform_reduce(ptr, ptr + n, AbsValue(), 0.0f,
                                  cuda::maximum<float>());
}

static double indexed_square_sum_device(const float *d_data,
                                        const int *d_indices, size_t n) {
  if (n == 0) {
    return 0.0;
  }
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + static_cast<int>(n);
  IndexedSquareValue op{d_data, d_indices};
  return thrust::transform_reduce(first, last, op, 0.0, thrust::plus<double>());
}

static float active_rms_device(const float *d_data, const int *fluid_indices,
                               int num_fluid_cells,
                               const IBM_Data &ibm_data) {
  // The outer fixed-point convergence should be measured over the active
  // velocity unknowns only. That keeps the stopping rule comparable between
  // RBGS and multigrid and avoids a grid-size-dependent max norm driven by a
  // few hotspot cells near IBM interfaces.
  const int num_ibm_cells = ibm_data.num_active_cells;
  const size_t count =
      static_cast<size_t>(num_fluid_cells) + static_cast<size_t>(num_ibm_cells);
  if (count == 0) {
    return 0.0f;
  }
  double sum_sq = indexed_square_sum_device(d_data, fluid_indices,
                                            static_cast<size_t>(num_fluid_cells));
  sum_sq += indexed_square_sum_device(d_data, ibm_data.cell_index,
                                      static_cast<size_t>(num_ibm_cells));
  return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
}

static void free_ibm_storage(IBM_Data &data) {
  if (data.cell_index != nullptr)
    CHECK_CUDA(cudaFree(data.cell_index));
  if (data.D_rescale != nullptr)
    CHECK_CUDA(cudaFree(data.D_rescale));
  if (data.num_boundaries != nullptr)
    CHECK_CUDA(cudaFree(data.num_boundaries));
  if (data.dir_code != nullptr)
    CHECK_CUDA(cudaFree(data.dir_code));
  if (data.K_val != nullptr)
    CHECK_CUDA(cudaFree(data.K_val));
  if (data.M_val != nullptr)
    CHECK_CUDA(cudaFree(data.M_val));
  if (data.X_val != nullptr)
    CHECK_CUDA(cudaFree(data.X_val));
  if (data.Nbc_val != nullptr)
    CHECK_CUDA(cudaFree(data.Nbc_val));
  if (data.R_val != nullptr)
    CHECK_CUDA(cudaFree(data.R_val));
  data = IBM_Data{};
}

static void alloc_ibm_storage(IBM_Data &data, size_t n) {
  // IBM metadata is sparse in realistic packings, so the heavy SoA storage is
  // allocated by the active cut-cell count rather than by the full grid size.
  if (n == 0) {
    data = IBM_Data{};
    return;
  }
  CHECK_CUDA(cudaMalloc(&data.cell_index, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&data.D_rescale, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&data.num_boundaries, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&data.dir_code, n * 6 * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&data.K_val, n * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&data.M_val, n * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&data.X_val, n * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&data.Nbc_val, n * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&data.R_val, n * 6 * sizeof(float)));
  data.num_active_cells = 0;
}

static void replace_int_buffer(int *&ptr, size_t n) {
  if (ptr != nullptr)
    CHECK_CUDA(cudaFree(ptr));
  ptr = nullptr;
  if (n > 0)
    CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(int)));
}

static void validate_compact_ibm_host(const char *label, const IBM_Data &data,
                                      const int *ibm_id_map,
                                      int num_elements) {
  // Optional host-side validation for the compact IBM layout. This is kept
  // behind an env var so it can be enabled when debugging allocator changes
  // without affecting normal solver performance.
  const int n = data.num_active_cells;
  if (n <= 0) {
    return;
  }

  std::vector<int> host_cells(n);
  std::vector<float> host_d(n);
  std::vector<int> host_map(num_elements);
  CHECK_CUDA(cudaMemcpy(host_cells.data(), data.cell_index, n * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_d.data(), data.D_rescale, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_map.data(), ibm_id_map, num_elements * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned char> seen(static_cast<size_t>(num_elements), 0);
  for (int i = 0; i < n; ++i) {
    const int idx = host_cells[i];
    if (idx < 0 || idx >= num_elements) {
      throw std::runtime_error(std::string(label) +
                               ": compact IBM cell index out of range");
    }
    if (host_map[idx] != i) {
      throw std::runtime_error(std::string(label) +
                               ": ibm_id_map does not point back to compact "
                               "list entry");
    }
    if (seen[idx] != 0) {
      throw std::runtime_error(std::string(label) +
                               ": duplicate compact IBM cell index");
    }
    seen[idx] = 1;
    if (!std::isfinite(host_d[i])) {
      throw std::runtime_error(std::string(label) +
                               ": non-finite D_rescale in compact IBM data");
    }
  }
}

static void validate_fluid_indices_host(const char *label,
                                        const int *fluid_indices,
                                        int num_fluid_cells,
                                        const int *ibm_id_map,
                                        int num_elements) {
  if (num_fluid_cells <= 0) {
    return;
  }

  std::vector<int> host_fluid(num_fluid_cells);
  std::vector<int> host_map(num_elements);
  CHECK_CUDA(cudaMemcpy(host_fluid.data(), fluid_indices,
                        num_fluid_cells * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_map.data(), ibm_id_map, num_elements * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned char> seen(static_cast<size_t>(num_elements), 0);
  for (int i = 0; i < num_fluid_cells; ++i) {
    const int idx = host_fluid[i];
    if (idx < 0 || idx >= num_elements) {
      throw std::runtime_error(std::string(label) +
                               ": fluid index out of range");
    }
    if (host_map[idx] != -1) {
      throw std::runtime_error(std::string(label) +
                               ": fluid index overlaps with IBM map");
    }
    if (seen[idx] != 0) {
      throw std::runtime_error(std::string(label) +
                               ": duplicate fluid index");
    }
    seen[idx] = 1;
  }
}

static float sample_sdf_interp_host(float x, float y, float z, const float *sdf,
                                    int3 res) {
  float fx = std::floorf(x);
  float fy = std::floorf(y);
  float fz = std::floorf(z);

  float wx = x - fx;
  float wy = y - fy;
  float wz = z - fz;

  int ix = static_cast<int>(fx);
  int iy = static_cast<int>(fy);
  int iz = static_cast<int>(fz);

  int x0 = (ix % res.x + res.x) % res.x;
  int y0 = (iy % res.y + res.y) % res.y;
  int z0 = (iz % res.z + res.z) % res.z;

  int x1 = (x0 + 1) % res.x;
  int y1 = (y0 + 1) % res.y;
  int z1 = (z0 + 1) % res.z;

  int stride_xy = res.x * res.y;
  int idx000 = z0 * stride_xy + y0 * res.x + x0;
  int idx100 = z0 * stride_xy + y0 * res.x + x1;
  int idx010 = z0 * stride_xy + y1 * res.x + x0;
  int idx110 = z0 * stride_xy + y1 * res.x + x1;
  int idx001 = z1 * stride_xy + y0 * res.x + x0;
  int idx101 = z1 * stride_xy + y0 * res.x + x1;
  int idx011 = z1 * stride_xy + y1 * res.x + x0;
  int idx111 = z1 * stride_xy + y1 * res.x + x1;

  float c000 = sdf[idx000];
  float c100 = sdf[idx100];
  float c010 = sdf[idx010];
  float c110 = sdf[idx110];
  float c001 = sdf[idx001];
  float c101 = sdf[idx101];
  float c011 = sdf[idx011];
  float c111 = sdf[idx111];

  float c00 = c000 * (1.0f - wx) + c100 * wx;
  float c10 = c010 * (1.0f - wx) + c110 * wx;
  float c01 = c001 * (1.0f - wx) + c101 * wx;
  float c11 = c011 * (1.0f - wx) + c111 * wx;

  float c0 = c00 * (1.0f - wy) + c10 * wy;
  float c1 = c01 * (1.0f - wy) + c11 * wy;

  return c0 * (1.0f - wz) + c1 * wz;
}

void CFDSolver::initialize(const SDFData &sdf_data) {
  if (sdf_data.size() != num_elements) {
    std::cerr << "SDF size mismatch!" << std::endl;
    return;
  }
  float max_df = -std::numeric_limits<float>::infinity();
  int max_idx = 0;
  for (size_t i = 0; i < sdf_data.sdf_values.size(); ++i) {
    float val = sdf_data.sdf_values[i];
    if (val > max_df) {
      max_df = val;
      max_idx = static_cast<int>(i);
    }
  }
  pin_idx = max_idx;
  CHECK_CUDA(cudaMemcpy(grid.sdf, sdf_data.sdf_values.data(),
                        num_elements * sizeof(float), cudaMemcpyHostToDevice));

  // Initialize IBM Geometry
  ibm_geometry_dirty_ = true;
  update_ibm_geometry();
  pressure_mg_built_ = false;
  velocity_mg_built_ = false;
  if (pressure_multigrid_enabled_) {
    build_pressure_multigrid();
  }
  if (velocity_multigrid_enabled_) {
    build_velocity_multigrid();
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Forward Declaration of Kernel (it's in another file, need to link or extern)
// Ideally put prototype in header or extern here.
// extern __global__ void compute_ibm_geometry_kernel(
//    IBM_Data ibm_data, int *ibm_id_map, const float *__restrict__ sdf, int3
//    res, float3 spacing, int *counter, float3 offset, int bc_type);

// Functor for Thrust to identify pure fluid cells
struct is_fluid_functor {
  const float *sdf;
  const int *ibm_id_map;
  float3 offset;
  int3 res;

  is_fluid_functor(const float *_sdf, const int *_ibm_id_map, float3 _offset,
                   int3 _res)
      : sdf(_sdf), ibm_id_map(_ibm_id_map), offset(_offset), res(_res) {}

  __device__ bool operator()(int idx) const {
    // Condition 1: Not an IBM cell
    if (ibm_id_map[idx] != -1) {
      return false;
    }

    // Condition 2: Not a solid cell (sample SDF at the component-specific
    // location)
    int z = idx / (res.x * res.y);
    int y = (idx % (res.x * res.y)) / res.x;
    int x = idx % res.x;

    float sdf_val =
        sample_sdf_interp(x + offset.x, y + offset.y, z + offset.z, sdf, res);
    return sdf_val > 0.0f;
  }
};

template <typename T>
__global__ void restrict_average_velmg_kernel(T *__restrict__ coarse,
                                              const T *__restrict__ fine,
                                              int3 coarse_res, int3 fine_res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= coarse_res.x || idx_y >= coarse_res.y || idx_z >= coarse_res.z)
    return;

  T sum = static_cast<T>(0);
  for (int dz = 0; dz < 2; ++dz) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        sum += fine[get_idx(2 * idx_x + dx, 2 * idx_y + dy, 2 * idx_z + dz,
                            fine_res)];
      }
    }
  }
  coarse[get_idx(idx_x, idx_y, idx_z, coarse_res)] = static_cast<T>(0.125) * sum;
}

__global__ void compute_residual_velmg_kernel(
    float *__restrict__ residual, const float *__restrict__ x,
    const float *__restrict__ A_C, const float *__restrict__ A_W,
    const float *__restrict__ A_E, const float *__restrict__ A_S,
    const float *__restrict__ A_N, const float *__restrict__ A_B,
    const float *__restrict__ A_T, const float *__restrict__ rhs, int3 res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);
  int idx_E = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_W = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_N = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_S = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_T = get_idx(idx_x, idx_y, idx_z + 1, res);
  int idx_B = get_idx(idx_x, idx_y, idx_z - 1, res);

  float Ax = A_C[idx] * x[idx] + A_W[idx] * x[idx_W] + A_E[idx] * x[idx_E] +
             A_S[idx] * x[idx_S] + A_N[idx] * x[idx_N] + A_B[idx] * x[idx_B] +
             A_T[idx] * x[idx_T];
  residual[idx] = rhs[idx] - Ax;
}

__global__ void prolongate_trilinear_add_velmg_kernel(
    float *__restrict__ fine, const float *__restrict__ coarse, int3 fine_res,
    int3 coarse_res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx_x >= fine_res.x || idx_y >= fine_res.y || idx_z >= fine_res.z)
    return;

  float x = 0.5f * static_cast<float>(idx_x) - 0.25f;
  float y = 0.5f * static_cast<float>(idx_y) - 0.25f;
  float z = 0.5f * static_cast<float>(idx_z) - 0.25f;
  fine[get_idx(idx_x, idx_y, idx_z, fine_res)] +=
      sample_field(coarse, x, y, z, coarse_res);
}

static float sample_field_interp_host_velmg(const std::vector<float> &field,
                                            int3 res, float x, float y,
                                            float z) {
  float fx = floorf(x);
  float fy = floorf(y);
  float fz = floorf(z);
  float wx = x - fx;
  float wy = y - fy;
  float wz = z - fz;
  int ix = static_cast<int>(fx);
  int iy = static_cast<int>(fy);
  int iz = static_cast<int>(fz);
  int x0 = (ix % res.x + res.x) % res.x;
  int y0 = (iy % res.y + res.y) % res.y;
  int z0 = (iz % res.z + res.z) % res.z;
  int x1 = (x0 + 1) % res.x;
  int y1 = (y0 + 1) % res.y;
  int z1 = (z0 + 1) % res.z;
  auto fetch = [&](int xx, int yy, int zz) {
    return field[zz * res.y * res.x + yy * res.x + xx];
  };
  float c000 = fetch(x0, y0, z0);
  float c100 = fetch(x1, y0, z0);
  float c010 = fetch(x0, y1, z0);
  float c110 = fetch(x1, y1, z0);
  float c001 = fetch(x0, y0, z1);
  float c101 = fetch(x1, y0, z1);
  float c011 = fetch(x0, y1, z1);
  float c111 = fetch(x1, y1, z1);
  float c00 = c000 * (1.0f - wx) + c100 * wx;
  float c10 = c010 * (1.0f - wx) + c110 * wx;
  float c01 = c001 * (1.0f - wx) + c101 * wx;
  float c11 = c011 * (1.0f - wx) + c111 * wx;
  float c0 = c00 * (1.0f - wy) + c10 * wy;
  float c1 = c01 * (1.0f - wy) + c11 * wy;
  return c0 * (1.0f - wz) + c1 * wz;
}

static void build_sdf_hierarchy_host_velmg(
    const float *fine_sdf_device, int3 fine_res, float3 fine_spacing,
    int max_levels, std::vector<int3> &level_res,
    std::vector<float3> &level_spacing,
    std::vector<std::vector<float>> &host_sdf_levels) {
  level_res.clear();
  level_spacing.clear();
  host_sdf_levels.clear();

  const int n = fine_res.x * fine_res.y * fine_res.z;
  level_res.push_back(fine_res);
  level_spacing.push_back(fine_spacing);
  host_sdf_levels.emplace_back(n);
  CHECK_CUDA(cudaMemcpy(host_sdf_levels[0].data(), fine_sdf_device,
                        n * sizeof(float), cudaMemcpyDeviceToHost));

  while (static_cast<int>(level_res.size()) < max_levels) {
    int3 curr_res = level_res.back();
    if ((curr_res.x % 2) != 0 || (curr_res.y % 2) != 0 || (curr_res.z % 2) != 0)
      break;
    if (curr_res.x < 8 || curr_res.y < 8 || curr_res.z < 8)
      break;
    int3 coarse_res =
        make_int3(curr_res.x / 2, curr_res.y / 2, curr_res.z / 2);
    float3 curr_spacing = level_spacing.back();
    float3 coarse_spacing = make_float3(2.0f * curr_spacing.x,
                                        2.0f * curr_spacing.y,
                                        2.0f * curr_spacing.z);
    std::vector<float> coarse_sdf(coarse_res.x * coarse_res.y * coarse_res.z);
    const auto &curr_sdf = host_sdf_levels.back();
    for (int z = 0; z < coarse_res.z; ++z) {
      for (int y = 0; y < coarse_res.y; ++y) {
        for (int x = 0; x < coarse_res.x; ++x) {
          coarse_sdf[z * coarse_res.y * coarse_res.x + y * coarse_res.x + x] =
              sample_field_interp_host_velmg(curr_sdf, curr_res,
                                             2.0f * x + 0.5f, 2.0f * y + 0.5f,
                                             2.0f * z + 0.5f);
        }
      }
    }
    level_res.push_back(coarse_res);
    level_spacing.push_back(coarse_spacing);
    host_sdf_levels.push_back(std::move(coarse_sdf));
  }
}

void CFDSolver::update_ibm_geometry() {
  int *d_counter;
  CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

  dim3 threads(8, 8, 8);
  dim3 blocks((grid.res.x + threads.x - 1) / threads.x,
              (grid.res.y + threads.y - 1) / threads.y,
              (grid.res.z + threads.z - 1) / threads.z);

  auto count_geo_pass = [&](float3 offset) -> int {
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    count_ibm_cells_kernel<<<blocks, threads>>>(grid.sdf, grid.res, offset,
                                                d_counter);
    int count;
    CHECK_CUDA(
        cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
  };

  auto run_geo_pass = [&](IBM_Data &data, int *map, float3 offset,
                          int bc_type, int count) -> int {
    // Two-pass build:
    // 1. count candidate IBM cells to size the compact buffers
    // 2. fill the compact SoA and keep the actual fill count
    //
    // The second count is retained because the fill kernel is the source of
    // truth for the compact indexing and catches any mismatch immediately.
    free_ibm_storage(data);
    alloc_ibm_storage(data, static_cast<size_t>(count));
    if (count == 0) {
      data.num_active_cells = 0;
      return 0;
    }
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    if (ibm_scheme_ == 1) {
      compute_ibm_geometry_kernel<1>
          <<<blocks, threads>>>(data, map, grid.sdf, grid.res, grid.spacing,
                                d_counter, offset, bc_type);
    } else {
      compute_ibm_geometry_kernel<0>
          <<<blocks, threads>>>(data, map, grid.sdf, grid.res, grid.spacing,
                                d_counter, offset, bc_type);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    int actual_count = 0;
    CHECK_CUDA(cudaMemcpy(&actual_count, d_counter, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (actual_count > count) {
      throw std::runtime_error(
          "IBM fill pass exceeded the allocated compact storage.");
    }
    data.num_active_cells = actual_count;
    return actual_count;
  };

  // 0. Update Area Fractions (for Projection & Pressure Stencil)
  compute_fluid_fraction_kernel<<<blocks, threads>>>(
      grid.sdf, grid.frac_u, grid.res, grid.spacing,
      make_float3(-0.5f, 0.0f, 0.0f), 1);
  compute_fluid_fraction_kernel<<<blocks, threads>>>(
      grid.sdf, grid.frac_v, grid.res, grid.spacing,
      make_float3(0.0f, -0.5f, 0.0f), 2);
  compute_fluid_fraction_kernel<<<blocks, threads>>>(
      grid.sdf, grid.frac_w, grid.res, grid.spacing,
      make_float3(0.0f, 0.0f, -0.5f), 3);
  CHECK_CUDA(cudaGetLastError());

  const int pressure_ibm_count = count_geo_pass(make_float3(0.0f, 0.0f, 0.0f));
  const int u_ibm_count = count_geo_pass(make_float3(-0.5f, 0.0f, 0.0f));
  const int v_ibm_count = count_geo_pass(make_float3(0.0f, -0.5f, 0.0f));
  const int w_ibm_count = count_geo_pass(make_float3(0.0f, 0.0f, -0.5f));

  // 1. Centered (Pressure) - Neumann (1)
  grid.num_ibm_cells = run_geo_pass(grid.ibm_data, grid.ibm_id_map,
                                    make_float3(0.0f, 0.0f, 0.0f), 1,
                                    pressure_ibm_count);

  // 2. U (Face X: 0.0 x) - Dirichlet (0)
  grid.num_ibm_cells_u = run_geo_pass(grid.ibm_data_u, grid.ibm_id_map_u,
                                      make_float3(-0.5f, 0.0f, 0.0f), 0,
                                      u_ibm_count);

  // 3. V (Face Y: 0.0 y) - Dirichlet (0)
  grid.num_ibm_cells_v = run_geo_pass(grid.ibm_data_v, grid.ibm_id_map_v,
                                      make_float3(0.0f, -0.5f, 0.0f), 0,
                                      v_ibm_count);

  // 4. W (Face Z: 0.0 z) - Dirichlet (0)
  grid.num_ibm_cells_w = run_geo_pass(grid.ibm_data_w, grid.ibm_id_map_w,
                                      make_float3(0.0f, 0.0f, -0.5f), 0,
                                      w_ibm_count);

  CHECK_CUDA(cudaFree(d_counter));

  // 5. Classify pure fluid cells
  auto classify_fluid = [&](int *&fluid_indices, int &num_fluid_cells,
                            int *ibm_id_map, float3 offset) {
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + num_elements;
    is_fluid_functor pred(grid.sdf, ibm_id_map, offset, grid.res);
    try {
      // As with the IBM list, size the fluid list from a count pass and retain
      // the copy_if result as the final authoritative count.
      const int predicted_count =
          thrust::count_if(thrust::device, first, last, pred);
      replace_int_buffer(fluid_indices, static_cast<size_t>(predicted_count));
      num_fluid_cells = 0;
      if (predicted_count > 0) {
        thrust::device_ptr<int> fluid_indices_ptr(fluid_indices);
        auto end_it =
            thrust::copy_if(thrust::device, first, last, fluid_indices_ptr, pred);
        num_fluid_cells = static_cast<int>(end_it - fluid_indices_ptr);
      }
    } catch (const std::exception &e) {
      std::cerr << "Thrust error during fluid cell classification: " << e.what()
                << std::endl;
      exit(1);
    }
  };

  classify_fluid(grid.fluid_indices_u, grid.num_fluid_cells_u,
                 grid.ibm_id_map_u, make_float3(-0.5f, 0.0f, 0.0f));
  classify_fluid(grid.fluid_indices_v, grid.num_fluid_cells_v,
                 grid.ibm_id_map_v, make_float3(0.0f, -0.5f, 0.0f));
  classify_fluid(grid.fluid_indices_w, grid.num_fluid_cells_w,
                 grid.ibm_id_map_w, make_float3(0.0f, 0.0f, -0.5f));

    if (std::getenv("PNM_VALIDATE_IBM") != nullptr) {
      // Expensive host-side consistency check for the compact indexing.
      validate_compact_ibm_host("pressure", grid.ibm_data, grid.ibm_id_map,
                                num_elements);
    validate_compact_ibm_host("u", grid.ibm_data_u, grid.ibm_id_map_u,
                              num_elements);
    validate_compact_ibm_host("v", grid.ibm_data_v, grid.ibm_id_map_v,
                              num_elements);
    validate_compact_ibm_host("w", grid.ibm_data_w, grid.ibm_id_map_w,
                              num_elements);
    validate_fluid_indices_host("fluid_u", grid.fluid_indices_u,
                                grid.num_fluid_cells_u, grid.ibm_id_map_u,
                                num_elements);
    validate_fluid_indices_host("fluid_v", grid.fluid_indices_v,
                                grid.num_fluid_cells_v, grid.ibm_id_map_v,
                                num_elements);
    validate_fluid_indices_host("fluid_w", grid.fluid_indices_w,
                                grid.num_fluid_cells_w, grid.ibm_id_map_w,
                                num_elements);
  }

  std::cout << "IBM Geometry Updated." << std::endl;
  std::cout << "  Pressure/Center: " << grid.num_ibm_cells << " IBM cells"
            << std::endl;
  std::cout << "  U-Face: " << grid.num_ibm_cells_u << " IBM cells, "
            << grid.num_fluid_cells_u << " fluid cells" << std::endl;
  std::cout << "  V-Face: " << grid.num_ibm_cells_v << " IBM cells, "
            << grid.num_fluid_cells_v << " fluid cells" << std::endl;
  std::cout << "  W-Face: " << grid.num_ibm_cells_w << " IBM cells, "
            << grid.num_fluid_cells_w << " fluid cells" << std::endl;
  ibm_geometry_dirty_ = false;
  pressure_mg_built_ = false;
  velocity_mg_built_ = false;
}

void CFDSolver::set_body_force(float3 force_density) {
  // Store force density directly (N/m^3)
  grid.body_force_density_ = force_density;
}

void CFDSolver::set_boundary_velocity(float3 u_bc) { grid.u_bc_ = u_bc; }

void CFDSolver::set_ibm_scheme(int scheme) {
  if (ibm_scheme_ != scheme) {
    ibm_scheme_ = scheme;
    ibm_geometry_dirty_ = true;
  }
}

// Getters
std::vector<double> CFDSolver::get_u() const {
  std::vector<double> host_u(num_elements);
  CHECK_CUDA(cudaMemcpy(host_u.data(), grid.u, num_elements * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host_u;
}

std::vector<double> CFDSolver::get_v() const {
  std::vector<double> host_v(num_elements);
  CHECK_CUDA(cudaMemcpy(host_v.data(), grid.v, num_elements * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host_v;
}

std::vector<double> CFDSolver::get_w() const {
  std::vector<double> host_w(num_elements);
  CHECK_CUDA(cudaMemcpy(host_w.data(), grid.w, num_elements * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host_w;
}

std::vector<double> CFDSolver::get_p() const {
  std::vector<double> host_p(num_elements);
  CHECK_CUDA(cudaMemcpy(host_p.data(), grid.p, num_elements * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host_p;
}

float CFDSolver::get_momentum_residual_max(bool fluid_only) {
  std::vector<float> host_res_u(num_elements);
  std::vector<float> host_res_v(num_elements);
  std::vector<float> host_res_w(num_elements);

  CHECK_CUDA(cudaMemcpy(host_res_u.data(), grid.res_u,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_res_v.data(), grid.res_v,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_res_w.data(), grid.res_w,
                        num_elements * sizeof(float), cudaMemcpyDeviceToHost));

  float max_abs = 0.0f;
  if (!fluid_only) {
    for (size_t i = 0; i < num_elements; ++i) {
      float ru = std::fabs(host_res_u[i]);
      float rv = std::fabs(host_res_v[i]);
      float rw = std::fabs(host_res_w[i]);
      if (ru > max_abs)
        max_abs = ru;
      if (rv > max_abs)
        max_abs = rv;
      if (rw > max_abs)
        max_abs = rw;
    }
    return max_abs;
  }

  std::vector<float> host_sdf(num_elements);
  CHECK_CUDA(cudaMemcpy(host_sdf.data(), grid.sdf, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int3 res = grid.res;
  int stride_xy = res.x * res.y;
  for (int z = 0; z < res.z; ++z) {
    for (int y = 0; y < res.y; ++y) {
      for (int x = 0; x < res.x; ++x) {
        int idx = z * stride_xy + y * res.x + x;

        float sdf_u =
            sample_sdf_interp_host(x - 0.5f, y, z, host_sdf.data(), res);
        if (sdf_u > 0.0f) {
          float ru = std::fabs(host_res_u[idx]);
          if (ru > max_abs)
            max_abs = ru;
        }

        float sdf_v =
            sample_sdf_interp_host(x, y - 0.5f, z, host_sdf.data(), res);
        if (sdf_v > 0.0f) {
          float rv = std::fabs(host_res_v[idx]);
          if (rv > max_abs)
            max_abs = rv;
        }

        float sdf_w =
            sample_sdf_interp_host(x, y, z - 0.5f, host_sdf.data(), res);
        if (sdf_w > 0.0f) {
          float rw = std::fabs(host_res_w[idx]);
          if (rw > max_abs)
            max_abs = rw;
        }
      }
    }
  }

  return max_abs;
}

// Helper for NAN checking
__global__ void check_nan_kernel(float *data, int n, const char *label) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (isnan(data[idx])) {
      printf("NAN DETECTED in %s at index %d\n", label, idx);
      // asm("trap;");
    }
  }
}
void check_field_nan(float *d_data, int n, const char *label) {
  check_nan_kernel<<<(n + 255) / 256, 256>>>(d_data, n, label);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

// --------------------------------------------------------
// FUSED IBM KERNEL
// --------------------------------------------------------
template <int COMPONENT>
__global__ void compute_fused_ibm_kernel(
    // Input State (Double)
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const double *__restrict__ p,
    double *__restrict__ explicit_term,
    // SDF for solid masking
    const float *__restrict__ sdf,
    // IBM Geometry

    IBM_Data ibm_data,
    // Output: Residual (Float)
    float *__restrict__ residual,
    // Output: Jacobian (Float)
    float *__restrict__ A_C, float *__restrict__ A_W, float *__restrict__ A_E,
    float *__restrict__ A_S, float *__restrict__ A_N, float *__restrict__ A_B,
    float *__restrict__ A_T,
    // Parameters
    int3 res, float3 spacing, float dt, float rho, float mu, float3 body_force,
    float theta, float u_bc_val, bool compute_explicit) {

  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm_data.num_active_cells)
    return;

  int idx = ibm_data.cell_index[list_idx];

  // ========================================
  // SOLID FACE MASKING (Fused)
  // ========================================
  // This should not be strictly necessary if the list generation is correct,
  // but as a safety check.
  float offset_x = (COMPONENT == 0) ? -0.5f : 0.0f;
  float offset_y = (COMPONENT == 1) ? -0.5f : 0.0f;
  float offset_z = (COMPONENT == 2) ? -0.5f : 0.0f;

  int idx_x = idx % res.x;
  int idx_y = (idx / res.x) % res.y;
  int idx_z = idx / (res.x * res.y);

  float sdf_face = sample_sdf_interp(idx_x + offset_x, idx_y + offset_y,
                                     idx_z + offset_z, sdf, res);
  if (sdf_face <= 0.0f) {
    A_C[idx] = 1.0f;
    A_W[idx] = 0.0f;
    A_E[idx] = 0.0f;
    A_S[idx] = 0.0f;
    A_N[idx] = 0.0f;
    A_B[idx] = 0.0f;
    A_T[idx] = 0.0f;
    residual[idx] = 0.0f;
    return;
  }

  // Neighbor indices
  int idx_W = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_E = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_S = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_N = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_B = get_idx(idx_x, idx_y, idx_z - 1, res);
  int idx_T = get_idx(idx_x, idx_y, idx_z + 1, res);

  // ========================================
  // STEP 1: Build Base Stencil (Double Precision)
  // ========================================
  double d_inv_dx = 1.0 / (double)spacing.x;
  double d_inv_dy = 1.0 / (double)spacing.y;
  double d_inv_dz = 1.0 / (double)spacing.z;
  double d_inv_dx2 = d_inv_dx * d_inv_dx;
  double d_inv_dy2 = d_inv_dy * d_inv_dy;
  double d_inv_dz2 = d_inv_dz * d_inv_dz;
  double d_rho = (double)rho;
  double d_mu = (double)mu;
  double d_theta = (double)theta;
  double d_dt = (double)dt;

  // Diffusion coefficients
  double diff_corr = 2.0 * d_mu * (d_inv_dx2 + d_inv_dy2 + d_inv_dz2);
  double val_C = diff_corr;
  double val_W = -d_mu * d_inv_dx2;
  double val_E = -d_mu * d_inv_dx2;
  double val_S = -d_mu * d_inv_dy2;
  double val_N = -d_mu * d_inv_dy2;
  double val_B = -d_mu * d_inv_dz2;
  double val_T = -d_mu * d_inv_dz2;

  // Get advection velocity at this location (double precision)
  double uc = 0.0, vc = 0.0, wc = 0.0;

#define AVG4_D(v1, v2, v3, v4) (0.25 * ((v1) + (v2) + (v3) + (v4)))

  if (COMPONENT == 0) { // U at (i, j+0.5, k+0.5)
    uc = u[idx];
    vc = AVG4_D(v[get_idx(idx_x - 1, idx_y, idx_z, res)],
                v[get_idx(idx_x, idx_y, idx_z, res)],
                v[get_idx(idx_x - 1, idx_y + 1, idx_z, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z, res)]);
    wc = AVG4_D(w[get_idx(idx_x - 1, idx_y, idx_z, res)],
                w[get_idx(idx_x, idx_y, idx_z, res)],
                w[get_idx(idx_x - 1, idx_y, idx_z + 1, res)],
                w[get_idx(idx_x, idx_y, idx_z + 1, res)]);
  } else if (COMPONENT == 1) { // V at (i+0.5, j, k+0.5)
    vc = v[idx];
    uc = AVG4_D(u[get_idx(idx_x, idx_y - 1, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y - 1, idx_z, res)],
                u[get_idx(idx_x, idx_y, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z, res)]);
    wc = AVG4_D(w[get_idx(idx_x, idx_y - 1, idx_z, res)],
                w[get_idx(idx_x, idx_y, idx_z, res)],
                w[get_idx(idx_x, idx_y - 1, idx_z + 1, res)],
                w[get_idx(idx_x, idx_y, idx_z + 1, res)]);
  } else { // W at (i+0.5, j+0.5, k)
    wc = w[idx];
    uc = AVG4_D(u[get_idx(idx_x, idx_y, idx_z - 1, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z - 1, res)],
                u[get_idx(idx_x, idx_y, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z, res)]);
    vc = AVG4_D(v[get_idx(idx_x, idx_y, idx_z - 1, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z - 1, res)],
                v[get_idx(idx_x, idx_y, idx_z, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z, res)]);
  }
#undef AVG4_D

  // Central differencing near IBM walls
  double term_x = 0.5 * d_rho * uc * d_inv_dx;
  val_E += term_x;
  val_W -= term_x;
  double term_y = 0.5 * d_rho * vc * d_inv_dy;
  val_S -= term_y;
  val_N += term_y;
  double term_z = 0.5 * d_rho * wc * d_inv_dz;
  val_B -= term_z;
  val_T += term_z;

  // Build RHS: (rho/dt)*u_old + theta*force + (1-theta)*explicit
  double force = 0.0;
  double gp_current = 0.0;
  const double *phi_field = nullptr;

  if (COMPONENT == 0) {
    force = (double)body_force.x;
    gp_current = (p[idx] - p[idx_W]) * d_inv_dx;
    phi_field = u;
  } else if (COMPONENT == 1) {
    force = (double)body_force.y;
    gp_current = (p[idx] - p[idx_S]) * d_inv_dy;
    phi_field = v;
  } else {
    force = (double)body_force.z;
    gp_current = (p[idx] - p[idx_B]) * d_inv_dz;
    phi_field = w;
  }

  // ========================================
  // STEP 3: IBM Modification (Double Precision)
  // ========================================
  // The list_idx is the global thread ID, which is correct for this kernel
  double D_rescale = (double)ibm_data.D_rescale[list_idx];
  val_C *= D_rescale;

  double orig_vals[6] = {val_E, val_W, val_N, val_S, val_T, val_B};
  double mod_E = 0.0, mod_W = 0.0, mod_N = 0.0;
  double mod_S = 0.0, mod_T = 0.0, mod_B = 0.0;
  double inhom_accum = 0.0;

  for (int k = 0; k < 6; k++) {
    int entry = list_idx * 6 + k;
    double K = (double)ibm_data.K_val[entry];
    double M = (double)ibm_data.M_val[entry];
    double X = (double)ibm_data.X_val[entry];
    double Nbc = (double)ibm_data.Nbc_val[entry];
    double val_nb = orig_vals[k];
    val_C += val_nb * K;
    inhom_accum += Nbc * (double)u_bc_val * val_nb;

    switch (k) {
    case 0:
      mod_E += orig_vals[0] * (D_rescale * M - 1.0);
      mod_W += orig_vals[0] * X;
      break;
    case 1:
      mod_W += orig_vals[1] * (D_rescale * M - 1.0);
      mod_E += orig_vals[1] * X;
      break;
    case 2:
      mod_N += orig_vals[2] * (D_rescale * M - 1.0);
      mod_S += orig_vals[2] * X;
      break;
    case 3:
      mod_S += orig_vals[3] * (D_rescale * M - 1.0);
      mod_N += orig_vals[3] * X;
      break;
    case 4:
      mod_T += orig_vals[4] * (D_rescale * M - 1.0);
      mod_B += orig_vals[4] * X;
      break;
    case 5:
      mod_B += orig_vals[5] * (D_rescale * M - 1.0);
      mod_T += orig_vals[5] * X;
      break;
    }
  }

  val_E = orig_vals[0] + mod_E;
  val_W = orig_vals[1] + mod_W;
  val_N = orig_vals[2] + mod_N;
  val_S = orig_vals[3] + mod_S;
  val_T = orig_vals[4] + mod_T;
  val_B = orig_vals[5] + mod_B;

  // ========================================
  // STEP 4: Compute Residual (Double Precision)
  // ========================================
  double phi_C = phi_field[idx];
  double phi_W = phi_field[idx_W];
  double phi_E = phi_field[idx_E];
  double phi_S = phi_field[idx_S];
  double phi_N = phi_field[idx_N];
  double phi_B = phi_field[idx_B];
  double phi_T = phi_field[idx_T];

  double Ax = val_C * phi_C + val_W * phi_W + val_E * phi_E + val_S * phi_S +
              val_N * phi_N + val_B * phi_B + val_T * phi_T;

  double res_double = D_rescale * (force - gp_current) - Ax;
  if (explicit_term != nullptr)
    if (compute_explicit)
      explicit_term[idx] = D_rescale * (d_rho / d_dt) * phi_field[idx] +
                           (1.0 - d_theta) * res_double;
    else
      res_double = explicit_term[idx] -
                   D_rescale * (d_rho / d_dt) * phi_field[idx] +
                   d_theta * res_double;

  // ========================================
  // STEP 5: Store Outputs (Float)
  // ========================================
  residual[idx] = (float)res_double;
  A_C[idx] = theta * (float)val_C + (float)(D_rescale * (d_rho / d_dt));
  A_W[idx] = theta * (float)val_W;
  A_E[idx] = theta * (float)val_E;
  A_S[idx] = theta * (float)val_S;
  A_N[idx] = theta * (float)val_N;
  A_B[idx] = theta * (float)val_B;
  A_T[idx] = theta * (float)val_T;
}

// --------------------------------------------------------
// FUSED FLUID KERNEL
// --------------------------------------------------------
template <int COMPONENT>
__global__ void compute_fused_fluid_kernel(
    // Input State (Double)
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const double *__restrict__ p,
    double *__restrict__ explicit_term,
    // Output: Residual (Float)
    float *__restrict__ residual,
    // Output: Jacobian (Float)
    float *__restrict__ A_C, float *__restrict__ A_W, float *__restrict__ A_E,
    float *__restrict__ A_S, float *__restrict__ A_N, float *__restrict__ A_B,
    float *__restrict__ A_T,
    // Cell lists
    const int *__restrict__ cell_indices, int num_cells,
    // Parameters
    int3 res, float3 spacing, float dt, float rho, float mu, float3 body_force,
    float theta, bool compute_explicit) {

  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid >= num_cells)
    return;

  int idx = cell_indices[global_tid];

  // Neighbor indices
  int idx_x = idx % res.x;
  int idx_y = (idx / res.x) % res.y;
  int idx_z = idx / (res.x * res.y);

  int idx_W = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_E = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_S = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_N = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_B = get_idx(idx_x, idx_y, idx_z - 1, res);
  int idx_T = get_idx(idx_x, idx_y, idx_z + 1, res);

  // ========================================
  // STEP 1: Build Base Stencil (Double Precision)
  // ========================================
  double d_inv_dx = 1.0 / (double)spacing.x;
  double d_inv_dy = 1.0 / (double)spacing.y;
  double d_inv_dz = 1.0 / (double)spacing.z;
  double d_inv_dx2 = d_inv_dx * d_inv_dx;
  double d_inv_dy2 = d_inv_dy * d_inv_dy;
  double d_inv_dz2 = d_inv_dz * d_inv_dz;
  double d_rho = (double)rho;
  double d_mu = (double)mu;
  double d_theta = (double)theta;
  double d_dt = (double)dt;

  // Diffusion coefficients
  double diff_corr = 2.0 * d_mu * (d_inv_dx2 + d_inv_dy2 + d_inv_dz2);
  double val_C = diff_corr;
  double val_W = -d_mu * d_inv_dx2;
  double val_E = -d_mu * d_inv_dx2;
  double val_S = -d_mu * d_inv_dy2;
  double val_N = -d_mu * d_inv_dy2;
  double val_B = -d_mu * d_inv_dz2;
  double val_T = -d_mu * d_inv_dz2;

  // Get advection velocity at this location (double precision)
  double uc = 0.0, vc = 0.0, wc = 0.0;

#define AVG4_D(v1, v2, v3, v4) (0.25 * ((v1) + (v2) + (v3) + (v4)))

  if (COMPONENT == 0) { // U at (i, j+0.5, k+0.5)
    uc = u[idx];
    vc = AVG4_D(v[get_idx(idx_x - 1, idx_y, idx_z, res)],
                v[get_idx(idx_x, idx_y, idx_z, res)],
                v[get_idx(idx_x - 1, idx_y + 1, idx_z, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z, res)]);
    wc = AVG4_D(w[get_idx(idx_x - 1, idx_y, idx_z, res)],
                w[get_idx(idx_x, idx_y, idx_z, res)],
                w[get_idx(idx_x - 1, idx_y, idx_z + 1, res)],
                w[get_idx(idx_x, idx_y, idx_z + 1, res)]);
  } else if (COMPONENT == 1) { // V at (i+0.5, j, k+0.5)
    vc = v[idx];
    uc = AVG4_D(u[get_idx(idx_x, idx_y - 1, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y - 1, idx_z, res)],
                u[get_idx(idx_x, idx_y, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z, res)]);
    wc = AVG4_D(w[get_idx(idx_x, idx_y - 1, idx_z, res)],
                w[get_idx(idx_x, idx_y, idx_z, res)],
                w[get_idx(idx_x, idx_y - 1, idx_z + 1, res)],
                w[get_idx(idx_x, idx_y, idx_z + 1, res)]);
  } else { // W at (i+0.5, j+0.5, k)
    wc = w[idx];
    uc = AVG4_D(u[get_idx(idx_x, idx_y, idx_z - 1, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z - 1, res)],
                u[get_idx(idx_x, idx_y, idx_z, res)],
                u[get_idx(idx_x + 1, idx_y, idx_z, res)]);
    vc = AVG4_D(v[get_idx(idx_x, idx_y, idx_z - 1, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z - 1, res)],
                v[get_idx(idx_x, idx_y, idx_z, res)],
                v[get_idx(idx_x, idx_y + 1, idx_z, res)]);
  }
#undef AVG4_D

  // FOU Upwind Advection (for Jacobian diagonal dominance)
  double term_x = d_rho * uc * d_inv_dx;
  if (uc > 0.0) {
    val_C += term_x;
    val_W -= term_x;
  } else {
    val_C -= term_x;
    val_E += term_x;
  }

  double term_y = d_rho * vc * d_inv_dy;
  if (vc > 0.0) {
    val_C += term_y;
    val_S -= term_y;
  } else {
    val_C -= term_y;
    val_N += term_y;
  }

  double term_z = d_rho * wc * d_inv_dz;
  if (wc > 0.0) {
    val_C += term_z;
    val_B -= term_z;
  } else {
    val_C -= term_z;
    val_T += term_z;
  }

  double force = 0.0;
  double gp_current = 0.0;
  const double *phi_field = nullptr;

  if (COMPONENT == 0) {
    force = (double)body_force.x;
    gp_current = (p[idx] - p[idx_W]) * d_inv_dx;
    phi_field = u;
  } else if (COMPONENT == 1) {
    force = (double)body_force.y;
    gp_current = (p[idx] - p[idx_S]) * d_inv_dy;
    phi_field = v;
  } else {
    force = (double)body_force.z;
    gp_current = (p[idx] - p[idx_B]) * d_inv_dz;
    phi_field = w;
  }

  // ========================================
  // STEP 2: TVD Advection Correction (Double Precision)
  // ========================================
  double tvd_corr = 0.0;
  {
    double phi_LL, phi_L, phi_R, phi_RR;
    double u_face, f_tvd, f_fou;

    // Right face (i+1)
    phi_LL = phi_field[idx_W];
    phi_L = phi_field[idx];
    phi_R = phi_field[idx_E];
    phi_RR = phi_field[get_idx(idx_x + 2, idx_y, idx_z, res)];

    if (COMPONENT == 0)
      u_face = 0.5 * (phi_field[idx] + phi_field[idx_E]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 0,
                                      make_int3(idx_x, idx_y, idx_z), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_x_r = (f_tvd - f_fou);

    // Left face (i)
    phi_LL = phi_field[get_idx(idx_x - 2, idx_y, idx_z, res)];
    phi_L = phi_field[idx_W];
    phi_R = phi_field[idx];
    phi_RR = phi_field[idx_E];

    if (COMPONENT == 0)
      u_face = 0.5 * (phi_field[idx_W] + phi_field[idx]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 0,
                                      make_int3(idx_x - 1, idx_y, idx_z), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_x_l = (f_tvd - f_fou);

    tvd_corr += (diff_x_r - diff_x_l) * d_inv_dx;

    // Y-direction flux correction
    phi_LL = phi_field[idx_S];
    phi_L = phi_field[idx];
    phi_R = phi_field[idx_N];
    phi_RR = phi_field[get_idx(idx_x, idx_y + 2, idx_z, res)];

    if (COMPONENT == 1)
      u_face = 0.5 * (phi_field[idx] + phi_field[idx_N]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 1,
                                      make_int3(idx_x, idx_y, idx_z), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_y_t = (f_tvd - f_fou);

    phi_LL = phi_field[get_idx(idx_x, idx_y - 2, idx_z, res)];
    phi_L = phi_field[idx_S];
    phi_R = phi_field[idx];
    phi_RR = phi_field[idx_N];

    if (COMPONENT == 1)
      u_face = 0.5 * (phi_field[idx_S] + phi_field[idx]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 1,
                                      make_int3(idx_x, idx_y - 1, idx_z), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_y_b = (f_tvd - f_fou);

    tvd_corr += (diff_y_t - diff_y_b) * d_inv_dy;

    // Z-direction flux correction
    phi_LL = phi_field[idx_B];
    phi_L = phi_field[idx];
    phi_R = phi_field[idx_T];
    phi_RR = phi_field[get_idx(idx_x, idx_y, idx_z + 2, res)];

    if (COMPONENT == 2)
      u_face = 0.5 * (phi_field[idx] + phi_field[idx_T]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 2,
                                      make_int3(idx_x, idx_y, idx_z), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_z_t = (f_tvd - f_fou);

    phi_LL = phi_field[get_idx(idx_x, idx_y, idx_z - 2, res)];
    phi_L = phi_field[idx_B];
    phi_R = phi_field[idx];
    phi_RR = phi_field[idx_T];

    if (COMPONENT == 2)
      u_face = 0.5 * (phi_field[idx_B] + phi_field[idx]);
    else
      u_face = get_advection_velocity(u, v, w, COMPONENT, 2,
                                      make_int3(idx_x, idx_y, idx_z - 1), res);

    f_tvd = (double)get_tvd_flux((float)phi_LL, (float)phi_L, (float)phi_R,
                                 (float)phi_RR, (float)u_face);
    f_fou = (u_face > 0) ? u_face * phi_L : u_face * phi_R;
    double diff_z_b = (f_tvd - f_fou);

    tvd_corr += (diff_z_t - diff_z_b) * d_inv_dz;
  }

  // ========================================
  // STEP 4: Compute Residual (Double Precision)
  // ========================================
  double phi_C = phi_field[idx];
  double phi_W = phi_field[idx_W];
  double phi_E = phi_field[idx_E];
  double phi_S = phi_field[idx_S];
  double phi_N = phi_field[idx_N];
  double phi_B = phi_field[idx_B];
  double phi_T = phi_field[idx_T];

  double Ax = val_C * phi_C + val_W * phi_W + val_E * phi_E + val_S * phi_S +
              val_N * phi_N + val_B * phi_B + val_T * phi_T;

  double res_double = force - gp_current - d_rho * tvd_corr - Ax;
  if (explicit_term != nullptr)
    if (compute_explicit)
      explicit_term[idx] =
          (d_rho / d_dt) * phi_field[idx] + (1.0 - d_theta) * res_double;
    else
      res_double = explicit_term[idx] - (d_rho / d_dt) * phi_field[idx] +
                   d_theta * res_double;

  // ========================================
  // STEP 5: Store Outputs (Float)
  // ========================================

  residual[idx] = (float)res_double;
  A_C[idx] = theta * (float)val_C + rho / dt;
  A_W[idx] = theta * (float)val_W;
  A_E[idx] = theta * (float)val_E;
  A_S[idx] = theta * (float)val_S;
  A_N[idx] = theta * (float)val_N;
  A_B[idx] = theta * (float)val_B;
  A_T[idx] = theta * (float)val_T;
}

void CFDSolver::velocity_bind_fine_level(const double *u, const double *v,
                                         const double *w, const double *p,
                                         float *x, float *rhs, float *A_C,
                                         float *A_W, float *A_E, float *A_S,
                                         float *A_N, float *A_B, float *A_T) {
  if (velocity_mg_levels_.empty())
    return;
  VelocityMGLevel &fine = velocity_mg_levels_.front();
  fine.res = grid.res;
  fine.spacing = grid.spacing;
  fine.num_elements = grid.num_elements;
  fine.sdf = grid.sdf;
  fine.u = const_cast<double *>(u);
  fine.v = const_cast<double *>(v);
  fine.w = const_cast<double *>(w);
  fine.p = const_cast<double *>(p);
  fine.ibm_data_u = grid.ibm_data_u;
  fine.ibm_data_v = grid.ibm_data_v;
  fine.ibm_data_w = grid.ibm_data_w;
  fine.ibm_id_map_u = grid.ibm_id_map_u;
  fine.ibm_id_map_v = grid.ibm_id_map_v;
  fine.ibm_id_map_w = grid.ibm_id_map_w;
  fine.fluid_indices_u = grid.fluid_indices_u;
  fine.fluid_indices_v = grid.fluid_indices_v;
  fine.fluid_indices_w = grid.fluid_indices_w;
  fine.num_ibm_cells_u = grid.num_ibm_cells_u;
  fine.num_ibm_cells_v = grid.num_ibm_cells_v;
  fine.num_ibm_cells_w = grid.num_ibm_cells_w;
  fine.num_fluid_cells_u = grid.num_fluid_cells_u;
  fine.num_fluid_cells_v = grid.num_fluid_cells_v;
  fine.num_fluid_cells_w = grid.num_fluid_cells_w;
  fine.x = x;
  fine.rhs = rhs;
  fine.residual = grid.B_RHS;
  fine.A_C = A_C;
  fine.A_W = A_W;
  fine.A_E = A_E;
  fine.A_S = A_S;
  fine.A_N = A_N;
  fine.A_B = A_B;
  fine.A_T = A_T;
}

void CFDSolver::velocity_smooth_level(VelocityMGLevel &level, int sweeps) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                (level.res.y + block.y - 1) / block.y,
                (level.res.z + block.z - 1) / block.z);
  for (int k = 0; k < sweeps; ++k) {
    solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
        level.x, level.A_C, level.A_W, level.A_E, level.A_S, level.A_N,
        level.A_B, level.A_T, level.rhs, level.res, true);
    solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
        level.x, level.A_C, level.A_W, level.A_E, level.A_S, level.A_N,
        level.A_B, level.A_T, level.rhs, level.res, false);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void CFDSolver::velocity_project_coarse_state(int level_idx, float dt,
                                              float theta) {
  if (level_idx <= 0 ||
      level_idx >= static_cast<int>(velocity_mg_levels_.size()) ||
      !pressure_multigrid_enabled_) {
    return;
  }
  if (!pressure_mg_built_) {
    build_pressure_multigrid();
  }
  if (level_idx >= static_cast<int>(pressure_mg_levels_.size())) {
    return;
  }

  VelocityMGLevel &vel_level = velocity_mg_levels_[level_idx];
  PressureMGLevel &pressure_level = pressure_mg_levels_[level_idx];
  if (!pressure_level.owns_state) {
    return;
  }
  if (vel_level.res.x != pressure_level.res.x ||
      vel_level.res.y != pressure_level.res.y ||
      vel_level.res.z != pressure_level.res.z) {
    return;
  }

  const size_t state_bytes =
      static_cast<size_t>(vel_level.num_elements) * sizeof(double);
  const size_t scalar_bytes =
      static_cast<size_t>(vel_level.num_elements) * sizeof(float);

  CHECK_CUDA(cudaMemcpy(pressure_level.u, vel_level.u, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(pressure_level.v, vel_level.v, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(pressure_level.w, vel_level.w, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(pressure_level.p, vel_level.p, state_bytes,
                        cudaMemcpyDeviceToDevice));
  const int pressure_cycles = pressure_mg_v_cycles_ * (level_idx + 1);
  for (int cycle = 0; cycle < pressure_cycles; ++cycle) {
    CHECK_CUDA(cudaMemset(pressure_level.x, 0, scalar_bytes));
    CHECK_CUDA(cudaMemset(pressure_level.x_applied, 0, scalar_bytes));
    CHECK_CUDA(cudaMemset(pressure_level.rhs, 0, scalar_bytes));
    CHECK_CUDA(cudaMemset(pressure_level.residual, 0, scalar_bytes));
    pressure_compute_level_rhs(pressure_level, true);
    pressure_v_cycle(level_idx, dt, theta);
    pressure_apply_level_projection(pressure_level, dt, theta, true);
  }

  CHECK_CUDA(cudaMemcpy(vel_level.u, pressure_level.u, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(vel_level.v, pressure_level.v, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(vel_level.w, pressure_level.w, state_bytes,
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(vel_level.p, pressure_level.p, state_bytes,
                        cudaMemcpyDeviceToDevice));
}

void CFDSolver::velocity_update_coarse_operators(int component, float dt,
                                                 float theta) {
  if (velocity_mg_levels_.size() < 2)
    return;

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 block1d(256);
  for (size_t level_idx = 1; level_idx < velocity_mg_levels_.size(); ++level_idx) {
    VelocityMGLevel &fine = velocity_mg_levels_[level_idx - 1];
    VelocityMGLevel &coarse = velocity_mg_levels_[level_idx];
    dim3 coarse_grid_dim((coarse.res.x + block.x - 1) / block.x,
                         (coarse.res.y + block.y - 1) / block.y,
                         (coarse.res.z + block.z - 1) / block.z);

    restrict_average_velmg_kernel<<<coarse_grid_dim, block>>>(
        coarse.u, fine.u, coarse.res, fine.res);
    restrict_average_velmg_kernel<<<coarse_grid_dim, block>>>(
        coarse.v, fine.v, coarse.res, fine.res);
    restrict_average_velmg_kernel<<<coarse_grid_dim, block>>>(
        coarse.w, fine.w, coarse.res, fine.res);
    restrict_average_velmg_kernel<<<coarse_grid_dim, block>>>(
        coarse.p, fine.p, coarse.res, fine.res);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    velocity_project_coarse_state(static_cast<int>(level_idx), dt, theta);

    CHECK_CUDA(cudaMemset(coarse.A_C, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_W, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_E, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_S, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_N, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_B, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.A_T, 0, coarse.num_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(coarse.residual, 0, coarse.num_elements * sizeof(float)));

    if (component == 0) {
      dim3 fluid_grid((coarse.num_fluid_cells_u + 255) / 256);
      compute_fused_fluid_kernel<0><<<fluid_grid, block1d>>>(
          coarse.u, coarse.v, coarse.w, coarse.p, nullptr, coarse.residual,
          coarse.A_C, coarse.A_W, coarse.A_E, coarse.A_S, coarse.A_N,
          coarse.A_B, coarse.A_T, coarse.fluid_indices_u,
          coarse.num_fluid_cells_u, coarse.res, coarse.spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, false);
      add_brinkman_drag_kernel<0><<<coarse_grid_dim, block>>>(
          coarse.residual, coarse.A_C, coarse.u, coarse.v, coarse.w,
          coarse.vol_frac_u, coarse.res, coarse.spacing, theta, mu_);
    } else if (component == 1) {
      dim3 fluid_grid((coarse.num_fluid_cells_v + 255) / 256);
      compute_fused_fluid_kernel<1><<<fluid_grid, block1d>>>(
          coarse.u, coarse.v, coarse.w, coarse.p, nullptr, coarse.residual,
          coarse.A_C, coarse.A_W, coarse.A_E, coarse.A_S, coarse.A_N,
          coarse.A_B, coarse.A_T, coarse.fluid_indices_v,
          coarse.num_fluid_cells_v, coarse.res, coarse.spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, false);
      add_brinkman_drag_kernel<1><<<coarse_grid_dim, block>>>(
          coarse.residual, coarse.A_C, coarse.u, coarse.v, coarse.w,
          coarse.vol_frac_v, coarse.res, coarse.spacing, theta, mu_);
    } else {
      dim3 fluid_grid((coarse.num_fluid_cells_w + 255) / 256);
      compute_fused_fluid_kernel<2><<<fluid_grid, block1d>>>(
          coarse.u, coarse.v, coarse.w, coarse.p, nullptr, coarse.residual,
          coarse.A_C, coarse.A_W, coarse.A_E, coarse.A_S, coarse.A_N,
          coarse.A_B, coarse.A_T, coarse.fluid_indices_w,
          coarse.num_fluid_cells_w, coarse.res, coarse.spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, false);
      add_brinkman_drag_kernel<2><<<coarse_grid_dim, block>>>(
          coarse.residual, coarse.A_C, coarse.u, coarse.v, coarse.w,
          coarse.vol_frac_w, coarse.res, coarse.spacing, theta, mu_);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

void CFDSolver::velocity_v_cycle(int level_idx) {
  VelocityMGLevel &level = velocity_mg_levels_[level_idx];
  const bool is_bottom =
      (level_idx + 1 == static_cast<int>(velocity_mg_levels_.size()));
  if (is_bottom) {
    velocity_smooth_level(level, velocity_mg_bottom_sweeps_);
    return;
  }

  velocity_smooth_level(level, velocity_mg_pre_sweeps_);

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                (level.res.y + block.y - 1) / block.y,
                (level.res.z + block.z - 1) / block.z);
  compute_residual_velmg_kernel<<<grid_dim, block>>>(
      level.residual, level.x, level.A_C, level.A_W, level.A_E, level.A_S,
      level.A_N, level.A_B, level.A_T, level.rhs, level.res);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  VelocityMGLevel &coarse = velocity_mg_levels_[level_idx + 1];
  CHECK_CUDA(cudaMemset(coarse.x, 0, coarse.num_elements * sizeof(float)));
  dim3 coarse_grid_dim((coarse.res.x + block.x - 1) / block.x,
                       (coarse.res.y + block.y - 1) / block.y,
                       (coarse.res.z + block.z - 1) / block.z);
  restrict_average_velmg_kernel<<<coarse_grid_dim, block>>>(
      coarse.rhs, level.residual, coarse.res, level.res);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  velocity_v_cycle(level_idx + 1);

  prolongate_trilinear_add_velmg_kernel<<<grid_dim, block>>>(
      level.x, coarse.x, level.res, coarse.res);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  velocity_smooth_level(level, velocity_mg_post_sweeps_);
}

void CFDSolver::solve_velocity_with_multigrid(
    int component, const double *u, const double *v, const double *w,
    const double *p, float *x, float *rhs, float *A_C, float *A_W,
    float *A_E, float *A_S, float *A_N, float *A_B, float *A_T, float dt,
    float theta) {
  if (!velocity_mg_built_) {
    build_velocity_multigrid();
  }
  if (velocity_mg_levels_.empty())
    return;

  velocity_bind_fine_level(u, v, w, p, x, rhs, A_C, A_W, A_E, A_S, A_N, A_B,
                           A_T);
  velocity_update_coarse_operators(component, dt, theta);
  CHECK_CUDA(cudaMemset(x, 0, grid.num_elements * sizeof(float)));
  for (int cycle = 0; cycle < velocity_mg_v_cycles_; ++cycle) {
    velocity_v_cycle(0);
  }
}

void CFDSolver::free_velocity_multigrid() {
  auto free_float = [&](float *ptr) {
    if (ptr != nullptr)
      CHECK_CUDA(cudaFree(ptr));
  };
  auto free_double = [&](double *ptr) {
    if (ptr != nullptr)
      CHECK_CUDA(cudaFree(ptr));
  };
  auto free_int = [&](int *ptr) {
    if (ptr != nullptr)
      CHECK_CUDA(cudaFree(ptr));
  };
  auto free_ibm = [&](IBM_Data &data) {
    if (data.cell_index != nullptr)
      CHECK_CUDA(cudaFree(data.cell_index));
    if (data.D_rescale != nullptr)
      CHECK_CUDA(cudaFree(data.D_rescale));
    if (data.num_boundaries != nullptr)
      CHECK_CUDA(cudaFree(data.num_boundaries));
    if (data.dir_code != nullptr)
      CHECK_CUDA(cudaFree(data.dir_code));
    if (data.K_val != nullptr)
      CHECK_CUDA(cudaFree(data.K_val));
    if (data.M_val != nullptr)
      CHECK_CUDA(cudaFree(data.M_val));
    if (data.X_val != nullptr)
      CHECK_CUDA(cudaFree(data.X_val));
    if (data.Nbc_val != nullptr)
      CHECK_CUDA(cudaFree(data.Nbc_val));
    if (data.R_val != nullptr)
      CHECK_CUDA(cudaFree(data.R_val));
    data = IBM_Data{};
  };

  for (auto &level : velocity_mg_levels_) {
    if (!level.owns_storage)
      continue;
    free_float(level.sdf);
    free_double(level.u);
    free_double(level.v);
    free_double(level.w);
    free_double(level.p);
    free_float(level.A_C);
    free_float(level.A_W);
    free_float(level.A_E);
    free_float(level.A_S);
    free_float(level.A_N);
    free_float(level.A_B);
    free_float(level.A_T);
    free_float(level.x);
    free_float(level.rhs);
    free_float(level.residual);
    free_float(level.vol_frac_u);
    free_float(level.vol_frac_v);
    free_float(level.vol_frac_w);
    free_int(level.ibm_id_map_u);
    free_int(level.ibm_id_map_v);
    free_int(level.ibm_id_map_w);
    free_int(level.fluid_indices_u);
    if (level.fluid_indices_v != level.fluid_indices_u) {
      free_int(level.fluid_indices_v);
    }
    if (level.fluid_indices_w != level.fluid_indices_u &&
        level.fluid_indices_w != level.fluid_indices_v) {
      free_int(level.fluid_indices_w);
    }
    free_ibm(level.ibm_data_u);
    free_ibm(level.ibm_data_v);
    free_ibm(level.ibm_data_w);
  }
  velocity_mg_levels_.clear();
  velocity_mg_built_ = false;
}

void CFDSolver::build_velocity_multigrid() {
  free_velocity_multigrid();
  if (!velocity_multigrid_enabled_)
    return;

  std::vector<int3> level_res;
  std::vector<float3> level_spacing;
  std::vector<std::vector<float>> host_sdf_levels;
  build_sdf_hierarchy_host_velmg(grid.sdf, grid.res, grid.spacing,
                                 velocity_mg_max_levels_, level_res,
                                 level_spacing, host_sdf_levels);

  velocity_mg_levels_.resize(level_res.size());
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  for (size_t level_idx = 0; level_idx < velocity_mg_levels_.size(); ++level_idx) {
    VelocityMGLevel &level = velocity_mg_levels_[level_idx];
    level.res = level_res[level_idx];
    level.spacing = level_spacing[level_idx];
    level.num_elements = level.res.x * level.res.y * level.res.z;

    if (level_idx == 0) {
      level.owns_storage = false;
      continue;
    }

    size_t float_bytes = static_cast<size_t>(level.num_elements) * sizeof(float);
    size_t double_bytes = static_cast<size_t>(level.num_elements) * sizeof(double);

    CHECK_CUDA(cudaMalloc(&level.sdf, float_bytes));
    CHECK_CUDA(cudaMemcpy(level.sdf, host_sdf_levels[level_idx].data(), float_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&level.u, double_bytes));
    CHECK_CUDA(cudaMalloc(&level.v, double_bytes));
    CHECK_CUDA(cudaMalloc(&level.w, double_bytes));
    CHECK_CUDA(cudaMalloc(&level.p, double_bytes));
    CHECK_CUDA(cudaMalloc(&level.vol_frac_u, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.vol_frac_v, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.vol_frac_w, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_C, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_W, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_E, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_S, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_N, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_B, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.A_T, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.x, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.rhs, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.residual, float_bytes));
    CHECK_CUDA(cudaMalloc(&level.fluid_indices_u,
                          static_cast<size_t>(level.num_elements) * sizeof(int)));
    level.fluid_indices_v = level.fluid_indices_u;
    level.fluid_indices_w = level.fluid_indices_u;

    CHECK_CUDA(cudaMemset(level.u, 0, double_bytes));
    CHECK_CUDA(cudaMemset(level.v, 0, double_bytes));
    CHECK_CUDA(cudaMemset(level.w, 0, double_bytes));
    CHECK_CUDA(cudaMemset(level.p, 0, double_bytes));
    CHECK_CUDA(cudaMemset(level.x, 0, float_bytes));
    CHECK_CUDA(cudaMemset(level.rhs, 0, float_bytes));
    CHECK_CUDA(cudaMemset(level.residual, 0, float_bytes));
    thrust::sequence(thrust::device, level.fluid_indices_u,
                     level.fluid_indices_u + level.num_elements, 0);

    dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                  (level.res.y + block.y - 1) / block.y,
                  (level.res.z + block.z - 1) / block.z);
    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.vol_frac_u, level.res, level.spacing,
        make_float3(-0.5f, 0.0f, 0.0f), 0);
    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.vol_frac_v, level.res, level.spacing,
        make_float3(0.0f, -0.5f, 0.0f), 0);
    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.vol_frac_w, level.res, level.spacing,
        make_float3(0.0f, 0.0f, -0.5f), 0);
    level.num_ibm_cells_u = 0;
    level.num_ibm_cells_v = 0;
    level.num_ibm_cells_w = 0;
    level.num_fluid_cells_u = level.num_elements;
    level.num_fluid_cells_v = level.num_elements;
    level.num_fluid_cells_w = level.num_elements;
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    level.owns_storage = true;
  }

  velocity_mg_built_ = true;
}

void CFDSolver::set_velocity_multigrid_enabled(bool enabled) {
  velocity_multigrid_enabled_ = enabled;
  velocity_mg_built_ = false;
  if (!enabled)
    free_velocity_multigrid();
}

void CFDSolver::set_velocity_multigrid_params(int max_levels, int pre_sweeps,
                                              int post_sweeps,
                                              int bottom_sweeps,
                                              int v_cycles) {
  velocity_mg_max_levels_ = max_levels;
  velocity_mg_pre_sweeps_ = pre_sweeps;
  velocity_mg_post_sweeps_ = post_sweeps;
  velocity_mg_bottom_sweeps_ = bottom_sweeps;
  velocity_mg_v_cycles_ = v_cycles;
  velocity_mg_built_ = false;
}

// --------------------------------------------------------
// Picard / Newton Step Implementation
// --------------------------------------------------------
void CFDSolver::step(float dt) {
  int3 res = grid.res;
  float3 spacing = grid.spacing;
  int num_elements = grid.num_elements;

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid_dim((res.x + block.x - 1) / block.x,
                (res.y + block.y - 1) / block.y,
                (res.z + block.z - 1) / block.z);

  dim3 block1d(256);
  dim3 grid1d((num_elements + 255) / 256);
  float *d_debug_cell = nullptr;
  int dbg_idx = 0;
  auto read_debug_val = [&](const double *ptr) -> float {
    double val = 0.0;
    CHECK_CUDA(cudaMemcpy(&val, ptr + dbg_idx, sizeof(double),
                          cudaMemcpyDeviceToHost));
    return (float)val;
  };
  if (debug_cell_enabled_) {
    CHECK_CUDA(cudaMalloc(&d_debug_cell, 44 * sizeof(float)));
    int x = (debug_cell_.x % res.x + res.x) % res.x;
    int y = (debug_cell_.y % res.y + res.y) % res.y;
    int z = (debug_cell_.z % res.z + res.z) % res.z;
    dbg_idx = z * res.y * res.x + y * res.x + x;
  }

  // Static-geometry cases should reuse IBM preprocessing across steps.
  if (ibm_geometry_dirty_) {
    update_ibm_geometry();
  }

  // Enforce zero velocity inside solid faces before using them in advection.
  /*  apply_face_sdf_mask_kernel<<<grid_dim, block>>>(grid.u, grid.v, grid.w,
                                                    grid.sdf, res);
  */
  // Save previous state (for Time Derivative in RHS)
  CHECK_CUDA(cudaMemcpy(grid.u_old, grid.u, num_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(grid.v_old, grid.v, num_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(grid.w_old, grid.w, num_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  // Save previous time step pressure for the (1-theta) term.
  CHECK_CUDA(cudaMemcpy(grid.p_prev, grid.p, num_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  // Initialize previous pressure for incremental correction.
  CHECK_CUDA(cudaMemcpy(grid.p_old, grid.p, num_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  if (debug_cell_enabled_) {
    debug_cell_info_[44] = read_debug_val(grid.u);
    debug_cell_info_[45] = read_debug_val(grid.v);
    debug_cell_info_[46] = read_debug_val(grid.w);
  }

  // Theta Parameter
  float theta = theta_;
  float max_corr_u = 0.0f;
  float max_corr_v = 0.0f;
  float max_corr_w = 0.0f;

  bool compute_explicit = true;
  last_outer_iterations_used_ = 0;
  for (int outer = 0; outer < outer_iterations_; ++outer) {
    last_outer_iterations_used_ = outer + 1;

    // --- Solve U (Fused Delta Form) ---
    CHECK_CUDA(cudaMemset(grid.A_C, 0.0, num_elements * sizeof(float)));
    // Launch fluid kernel
    int num_fluid_u = grid.num_fluid_cells_u;
    if (num_fluid_u > 0) {
      dim3 fluid_grid1d((num_fluid_u + 255) / 256);
      compute_fused_fluid_kernel<0><<<fluid_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_u, grid.res_u, grid.A_C,
          grid.A_W, grid.A_E, grid.A_S, grid.A_N, grid.A_B, grid.A_T,
          grid.fluid_indices_u, num_fluid_u, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }
    // Launch IBM kernel
    int num_ibm_u = grid.ibm_data_u.num_active_cells;
    if (num_ibm_u > 0) {
      dim3 ibm_grid1d((num_ibm_u + 255) / 256);
      compute_fused_ibm_kernel<0><<<ibm_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_u, grid.sdf,
          grid.ibm_data_u, grid.res_u, grid.A_C, grid.A_W, grid.A_E, grid.A_S,
          grid.A_N, grid.A_B, grid.A_T, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, grid.u_bc_.x, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemset(grid.du, 0, num_elements * sizeof(float)));
    if (velocity_multigrid_enabled_) {
      if (!velocity_mg_built_) {
        build_velocity_multigrid();
      }
      solve_velocity_with_multigrid(0, grid.u, grid.v, grid.w, grid.p, grid.du,
                                    grid.res_u, grid.A_C, grid.A_W, grid.A_E,
                                    grid.A_S, grid.A_N, grid.A_B, grid.A_T, dt,
                                    theta);
    } else {
      for (int k = 0; k < v_max_iter_; k++) {
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.du, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_u, res, true);
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.du, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_u, res, false);
      }
    }
    apply_correction_kernel<<<grid1d, block1d>>>(grid.u, grid.du, num_elements);
    max_corr_u = max_abs_device(grid.du, num_elements);

    // --- Solve V (Fused Delta Form) ---
    CHECK_CUDA(cudaMemset(grid.A_C, 0.0, num_elements * sizeof(float)));
    int num_fluid_v = grid.num_fluid_cells_v;
    if (num_fluid_v > 0) {
      dim3 fluid_grid1d((num_fluid_v + 255) / 256);
      compute_fused_fluid_kernel<1><<<fluid_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_v, grid.res_v, grid.A_C,
          grid.A_W, grid.A_E, grid.A_S, grid.A_N, grid.A_B, grid.A_T,
          grid.fluid_indices_v, num_fluid_v, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }
    int num_ibm_v = grid.ibm_data_v.num_active_cells;
    if (num_ibm_v > 0) {
      dim3 ibm_grid1d((num_ibm_v + 255) / 256);
      compute_fused_ibm_kernel<1><<<ibm_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_v, grid.sdf,
          grid.ibm_data_v, grid.res_v, grid.A_C, grid.A_W, grid.A_E, grid.A_S,
          grid.A_N, grid.A_B, grid.A_T, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, grid.u_bc_.y, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }

    if (debug_stats_enabled_) {
      CHECK_CUDA(cudaMemcpy(grid.res_v_pre, grid.res_v,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      debug_stats_.res_before[1] = max_abs_device(grid.res_v, num_elements);
    }

    CHECK_CUDA(cudaMemset(grid.dv, 0, num_elements * sizeof(float)));
    if (velocity_multigrid_enabled_) {
      if (!velocity_mg_built_) {
        build_velocity_multigrid();
      }
      solve_velocity_with_multigrid(1, grid.u, grid.v, grid.w, grid.p, grid.dv,
                                    grid.res_v, grid.A_C, grid.A_W, grid.A_E,
                                    grid.A_S, grid.A_N, grid.A_B, grid.A_T, dt,
                                    theta);
    } else {
      for (int k = 0; k < v_max_iter_; k++) {
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.dv, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_v, res, true);
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.dv, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_v, res, false);
      }
    }

    apply_correction_kernel<<<grid1d, block1d>>>(grid.v, grid.dv, num_elements);
    max_corr_v = max_abs_device(grid.dv, num_elements);
    if (debug_stats_enabled_) {
      debug_stats_.corr_max[1] = max_corr_v;
      int num_fluid_v = grid.num_fluid_cells_v;
      if (num_fluid_v > 0) {
        dim3 fluid_grid1d((num_fluid_v + 255) / 256);
        compute_fused_fluid_kernel<1><<<fluid_grid1d, block1d>>>(
            grid.u, grid.v, grid.w, grid.p, grid.explicit_v, grid.res_v,
            grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N, grid.A_B,
            grid.A_T, grid.fluid_indices_v, num_fluid_v, res, spacing, dt, rho_,
            mu_, grid.body_force_density_, theta, compute_explicit);
      }
      int num_ibm_v = grid.ibm_data_v.num_active_cells;
      if (num_ibm_v > 0) {
        dim3 ibm_grid1d((num_ibm_v + 255) / 256);
        compute_fused_ibm_kernel<1><<<ibm_grid1d, block1d>>>(
            grid.u, grid.v, grid.w, grid.p, grid.explicit_v, grid.sdf,
            grid.ibm_data_v, grid.res_v, grid.A_C, grid.A_W, grid.A_E, grid.A_S,
            grid.A_N, grid.A_B, grid.A_T, res, spacing, dt, rho_, mu_,
            grid.body_force_density_, theta, grid.u_bc_.y, compute_explicit);
      }
      CHECK_CUDA(cudaMemcpy(grid.res_v_post, grid.res_v,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      debug_stats_.res_after[1] = max_abs_device(grid.res_v, num_elements);
    }
    if (debug_cell_enabled_) {
      debug_cell_info_[50] = read_debug_val(grid.u);
      debug_cell_info_[51] = read_debug_val(grid.v);
      debug_cell_info_[52] = read_debug_val(grid.w);
    }

    // --- Solve W (Fused Delta Form) ---
    CHECK_CUDA(cudaMemset(grid.A_C, 0.0, num_elements * sizeof(float)));
    int num_fluid_w = grid.num_fluid_cells_w;
    if (num_fluid_w > 0) {
      dim3 fluid_grid1d((num_fluid_w + 255) / 256);
      compute_fused_fluid_kernel<2><<<fluid_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_w, grid.res_w, grid.A_C,
          grid.A_W, grid.A_E, grid.A_S, grid.A_N, grid.A_B, grid.A_T,
          grid.fluid_indices_w, num_fluid_w, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }
    int num_ibm_w = grid.ibm_data_w.num_active_cells;
    if (num_ibm_w > 0) {
      dim3 ibm_grid1d((num_ibm_w + 255) / 256);
      compute_fused_ibm_kernel<2><<<ibm_grid1d, block1d>>>(
          grid.u, grid.v, grid.w, grid.p, grid.explicit_w, grid.sdf,
          grid.ibm_data_w, grid.res_w, grid.A_C, grid.A_W, grid.A_E, grid.A_S,
          grid.A_N, grid.A_B, grid.A_T, res, spacing, dt, rho_, mu_,
          grid.body_force_density_, theta, grid.u_bc_.z, compute_explicit);
      CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaMemset(grid.dw, 0, num_elements * sizeof(float)));
    if (velocity_multigrid_enabled_) {
      if (!velocity_mg_built_) {
        build_velocity_multigrid();
      }
      solve_velocity_with_multigrid(2, grid.u, grid.v, grid.w, grid.p, grid.dw,
                                    grid.res_w, grid.A_C, grid.A_W, grid.A_E,
                                    grid.A_S, grid.A_N, grid.A_B, grid.A_T, dt,
                                    theta);
    } else {
      for (int k = 0; k < v_max_iter_; k++) {
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.dw, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_w, res, true);
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.dw, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.res_w, res, false);
      }
    }

    apply_correction_kernel<<<grid1d, block1d>>>(grid.w, grid.dw, num_elements);
    max_corr_w = max_abs_device(grid.dw, num_elements);

    // Incremental Pressure Correction
    CHECK_CUDA(cudaMemcpy(grid.p_old, grid.p, num_elements * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(grid.phi, 0, num_elements * sizeof(float)));
    if (pressure_multigrid_enabled_) {
      if (!pressure_mg_built_) {
        build_pressure_multigrid();
      }

      compute_pressure_rhs_kernel<<<grid_dim, block>>>(
          grid.B_RHS, grid.frac_u, grid.frac_v, grid.frac_w, grid.sdf, grid.u,
          grid.v, grid.w, res, spacing);
      CHECK_CUDA(cudaGetLastError());
      remove_mean_from_device_vector(grid.B_RHS, num_elements);
      CHECK_CUDA(cudaMemcpy(grid.rhs, grid.B_RHS, num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));

      PressureMGLevel &fine_level = pressure_mg_levels_.front();
      CHECK_CUDA(cudaMemcpy(fine_level.rhs, grid.B_RHS,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      CHECK_CUDA(cudaMemset(fine_level.x, 0, num_elements * sizeof(float)));

      for (int cycle = 0; cycle < pressure_mg_v_cycles_; ++cycle) {
        pressure_v_cycle(0, dt, theta);
      }

      CHECK_CUDA(cudaMemcpy(grid.phi, fine_level.x,
                            num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    } else {
      compute_pressure_stencil_kernel<<<grid_dim, block>>>(
          grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N, grid.A_B,
          grid.A_T, grid.B_RHS, grid.frac_u, grid.frac_v, grid.frac_w,
          grid.sdf, grid.u, grid.v, grid.w, res, spacing);
      CHECK_CUDA(cudaMemcpy(grid.rhs, grid.B_RHS, num_elements * sizeof(float),
                            cudaMemcpyDeviceToDevice));

      for (int k = 0; k < p_max_iter_; k++) {
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.phi, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.B_RHS, res, true);
        solve_rbgs_stencil_kernel<<<grid_dim, block>>>(
            grid.phi, grid.A_C, grid.A_W, grid.A_E, grid.A_S, grid.A_N,
            grid.A_B, grid.A_T, grid.B_RHS, res, false);
      }
    }

    project_velocity_kernel<<<grid_dim, block>>>(
        grid.u, grid.v, grid.w, grid.phi, grid.frac_u, grid.frac_v, grid.frac_w,
        grid.sdf, res, spacing);

    update_pressure_from_phi_kernel<<<grid_dim, block>>>(
        grid.p, grid.phi, grid.p_old, grid.u, grid.v, grid.w, grid.rhs, res,
        spacing, dt, theta, rho_, mu_);
    double p_pin = 0.0;
    CHECK_CUDA(cudaMemcpy(&p_pin, grid.p + pin_idx, sizeof(double),
                          cudaMemcpyDeviceToHost));
    shift_pressure_kernel<<<grid1d, block1d>>>(grid.p, num_elements, p_pin);

    if (outer_tol_ > 0.0f) {
      float corr_metric_u = max_corr_u;
      float corr_metric_v = max_corr_v;
      float corr_metric_w = max_corr_w;
      if (outer_convergence_mode_ == 1) {
        corr_metric_u = active_rms_device(grid.du, grid.fluid_indices_u,
                                          grid.num_fluid_cells_u,
                                          grid.ibm_data_u);
        corr_metric_v = active_rms_device(grid.dv, grid.fluid_indices_v,
                                          grid.num_fluid_cells_v,
                                          grid.ibm_data_v);
        corr_metric_w = active_rms_device(grid.dw, grid.fluid_indices_w,
                                          grid.num_fluid_cells_w,
                                          grid.ibm_data_w);
      }
      float max_corr =
          fmaxf(corr_metric_u, fmaxf(corr_metric_v, corr_metric_w));
      if (max_corr < outer_tol_) {
        break;
      }
    }
    compute_explicit = false;
  }
}

// --------------------------------------------------------
// Find Pin Index Kernel
// --------------------------------------------------------
__global__ void find_pin_idx_kernel(const float *__restrict__ sdf, int *pin_idx,
                                    int num_elements) {}

std::vector<float> CFDSolver::get_fluid_fraction(int type, float3 offset) {
  std::vector<float> h_frac(num_elements);
  float *d_frac;
  CHECK_CUDA(cudaMalloc(&d_frac, num_elements * sizeof(float)));

  dim3 threads(8, 8, 8);
  dim3 blocks((grid.res.x + threads.x - 1) / threads.x,
              (grid.res.y + threads.y - 1) / threads.y,
              (grid.res.z + threads.z - 1) / threads.z);

  compute_fluid_fraction_kernel<<<blocks, threads>>>(
      grid.sdf, d_frac, grid.res, grid.spacing, offset, type);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_frac.data(), d_frac, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_frac));

  return h_frac;
}

void CFDSolver::set_u(const std::vector<double> &h_u) {
  if (h_u.size() != num_elements) {
    throw std::runtime_error("set_u: Input size mismatch");
  }
  CHECK_CUDA(cudaMemcpy(grid.u, h_u.data(), num_elements * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void CFDSolver::set_v(const std::vector<double> &h_v) {
  if (h_v.size() != num_elements) {
    throw std::runtime_error("set_v: Input size mismatch");
  }
  CHECK_CUDA(cudaMemcpy(grid.v, h_v.data(), num_elements * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void CFDSolver::set_w(const std::vector<double> &h_w) {
  if (h_w.size() != num_elements) {
    throw std::runtime_error("set_w: Input size mismatch");
  }
  CHECK_CUDA(cudaMemcpy(grid.w, h_w.data(), num_elements * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void CFDSolver::set_p(const std::vector<double> &h_p) {
  if (h_p.size() != num_elements) {
    throw std::runtime_error("set_p: Input size mismatch");
  }
  CHECK_CUDA(cudaMemcpy(grid.p, h_p.data(), num_elements * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void CFDSolver::scale_state(double velocity_scale, double pressure_scale) {
  const int threads = 256;
  const int blocks = (num_elements + threads - 1) / threads;

  if (velocity_scale != 1.0) {
    scale_field_kernel<<<blocks, threads>>>(grid.u, velocity_scale,
                                            num_elements);
    scale_field_kernel<<<blocks, threads>>>(grid.v, velocity_scale,
                                            num_elements);
    scale_field_kernel<<<blocks, threads>>>(grid.w, velocity_scale,
                                            num_elements);
  }
  if (pressure_scale != 1.0) {
    scale_field_kernel<<<blocks, threads>>>(grid.p, pressure_scale,
                                            num_elements);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Kernel to subtract mean from RHS
__global__ void subtract_mean_kernel(float *__restrict__ rhs, float mean,
                                     int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  rhs[idx] -= mean;
}

std::vector<float> CFDSolver::get_ibm_scaling(int component_idx) {
  int num_elements = grid.num_elements;
  std::vector<float> h_scale(num_elements, 1.0f);

  float *d_scale;
  CHECK_CUDA(cudaMalloc(&d_scale, num_elements * sizeof(float)));

  // Initialize device array to 1.0
  thrust::device_ptr<float> ptr(d_scale);
  thrust::fill(ptr, ptr + num_elements, 1.0f);

  IBM_Data *ibm_data_ptr = nullptr;
  if (component_idx == 0)
    ibm_data_ptr = &grid.ibm_data_u;
  else if (component_idx == 1)
    ibm_data_ptr = &grid.ibm_data_v;
  else if (component_idx == 2)
    ibm_data_ptr = &grid.ibm_data_w;

  if (ibm_data_ptr && ibm_data_ptr->num_active_cells > 0) {
    populate_ibm_scaling_kernel<<<(ibm_data_ptr->num_active_cells + 255) / 256,
                                  256>>>(d_scale, *ibm_data_ptr, num_elements);
    CHECK_CUDA(cudaGetLastError());
  }

  CHECK_CUDA(cudaMemcpy(h_scale.data(), d_scale, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_scale));

  return h_scale;
}

void CFDSolver::set_outer_iterations(int iterations) {
  outer_iterations_ = iterations;
}

void CFDSolver::set_outer_tolerance(float tol) { outer_tol_ = tol; }

void CFDSolver::set_outer_convergence_mode(int mode) {
  outer_convergence_mode_ = mode;
}
