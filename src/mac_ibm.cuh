// cfd-gpu -- Robust-Scaled cut-cell IBM for the velocity (momentum) solve on a MacGridHalo extended
// block. Ports the production solver's velocity IBM (cfd_solver_ibm.cu / _ibm_kernels.cuh): per cut cell
// the SDF geometry gives polynomial factors (D_rescale, K/M/X/Nbc) that are baked into the velocity
// diffusion stencil A_C..A_T plus an inhomogeneous Dirichlet term, eliminating the solid ghost values
// and enforcing the wall velocity. The bake (ibm_modify_stencil_k) only edits each cut cell's OWN row,
// so it distributes cleanly; only the SDF geometry needed porting to extended-block (no-wrap) sampling.
//
// The IBM math is factored into ibm_fill_entry (shared by the extended build here and a serial wrap
// reference) so the distributed coefficients match the serial ones cell-for-cell.
#pragma once

#include <cuda_runtime.h>

#include "cfd_solver.cuh"               // IBM_Data, get_idx
#include "cfd_solver_ibm_kernels.cuh"   // poly_* (Robust-Scaled polynomials)
#include "mac_cutcell.cuh"              // cc_sample_ext (clamped extended-block SDF sampling)

namespace cfdmpi {
namespace ibmdetail {

// Decide whether the staggered point (sdf_c) with its 6 axis neighbours (sdf_n) is an IBM cut cell:
// fluid centre with at least one solid neighbour.
__host__ __device__ inline bool ibm_is_cut(float sdf_c, const float sdf_n[6]) {
  if (sdf_c <= 0.0f) return false;
  for (int k = 0; k < 6; ++k)
    if (sdf_n[k] < 0.0f) return true;
  return false;
}

// Fill one IBM_Data entry (list_idx) for a cut cell at linear index c_idx from its 7 SDF samples.
// Verbatim port of compute_ibm_geometry_kernel's per-cell body (Dirichlet bc_type=0 / Neumann=1,
// SCHEME 0 point-value / 1 cell-average). Indexing-agnostic: c_idx is whatever layout the caller uses.
template <int SCHEME>
__device__ inline void ibm_fill_entry(IBM_Data ibm, int list_idx, int c_idx, float sdf_c,
                                      const float sdf_n[6], float3 spacing, int bc_type) {
  ibm.cell_index[list_idx] = c_idx;
  ibm.num_boundaries[list_idx] = 6;
  bool is_ghost[6];
  float xi_vals[6], D_vals[6];
  for (int k = 0; k < 6; ++k) {
    bool is_solid_n = (sdf_n[k] < 0.0f);
    if (is_solid_n) {  // centre is fluid (cut cell), so sign differs => interface
      is_ghost[k] = true;
      if (bc_type == 0) {
        float theta = sdf_c / (sdf_c - sdf_n[k]);
        if (theta < 1e-4f) theta = 1e-4f;
        if (theta > 1.0f) theta = 1.0f;
        xi_vals[k] = theta;
        D_vals[k] = (SCHEME == 0) ? poly_D(theta) : poly_D_avg(theta);
      } else {
        xi_vals[k] = 0.5f;
        D_vals[k] = 1.0f;
      }
    } else {
      is_ghost[k] = false;
      xi_vals[k] = 1.0f;
      D_vals[k] = 1e9f;
    }
  }

  if (bc_type == 0) {
    bool is_sandwich[3] = {is_ghost[0] && is_ghost[1], is_ghost[2] && is_ghost[3],
                           is_ghost[4] && is_ghost[5]};
    float D_sandwich[3] = {0, 0, 0};
    for (int a = 0; a < 3; ++a)
      if (is_sandwich[a])
        D_sandwich[a] = (SCHEME == 0) ? poly_D_sandwich(xi_vals[2 * a + 1], xi_vals[2 * a])
                                      : poly_D_sandwich_avg(xi_vals[2 * a + 1], xi_vals[2 * a]);

    float min_D_abs = 1e30f, D_rescale = 1.0f;
    auto update_min = [&](float val) {
      if (fabsf(val) < min_D_abs) { min_D_abs = fabsf(val); D_rescale = val; }
    };
    for (int axis = 0; axis < 3; ++axis) {
      if (is_sandwich[axis])
        update_min(D_sandwich[axis]);
      else {
        if (is_ghost[2 * axis]) update_min(D_vals[2 * axis]);
        if (is_ghost[2 * axis + 1]) update_min(D_vals[2 * axis + 1]);
      }
    }
    ibm.D_rescale[list_idx] = D_rescale;

    for (int axis = 0; axis < 3; ++axis) {
      int km = 2 * axis + 1, kp = 2 * axis;
      bool sandwich = is_sandwich[axis], g_p = is_ghost[kp], g_m = is_ghost[km];
      float D_axis = sandwich ? D_sandwich[axis]
                              : (g_p ? D_vals[kp] : (g_m ? D_vals[km] : D_rescale));
      float R = D_rescale / D_axis;
      if (fabsf(D_axis) < 1e-9f) R = 1.0f;
      ibm.R_val[list_idx * 6 + kp] = R;
      ibm.R_val[list_idx * 6 + km] = R;

      if (sandwich) {
        if (SCHEME == 0) {
          ibm.K_val[list_idx * 6 + kp] = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]) * R;
          ibm.K_val[list_idx * 6 + km] = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]) * R;
          ibm.Nbc_val[list_idx * 6 + kp] =
              (poly_Nbc_pp_sw(xi_vals[km], xi_vals[kp]) + poly_Nbc_mp_sw(xi_vals[km], xi_vals[kp])) * R;
          ibm.Nbc_val[list_idx * 6 + km] =
              (poly_Nbc_pp_sw(xi_vals[kp], xi_vals[km]) + poly_Nbc_mp_sw(xi_vals[kp], xi_vals[km])) * R;
        } else {
          ibm.K_val[list_idx * 6 + kp] = poly_N_c_sandwich_avg(xi_vals[km], xi_vals[kp]) * R;
          ibm.K_val[list_idx * 6 + km] = poly_N_c_sandwich_avg(xi_vals[kp], xi_vals[km]) * R;
          ibm.Nbc_val[list_idx * 6 + kp] = (poly_Nbc_pp_sw_avg(xi_vals[km], xi_vals[kp]) +
                                            poly_Nbc_mp_sw_avg(xi_vals[km], xi_vals[kp])) * R;
          ibm.Nbc_val[list_idx * 6 + km] = (poly_Nbc_pp_sw_avg(xi_vals[kp], xi_vals[km]) +
                                            poly_Nbc_mp_sw_avg(xi_vals[kp], xi_vals[km])) * R;
        }
        ibm.M_val[list_idx * 6 + kp] = 0.0f; ibm.X_val[list_idx * 6 + kp] = 0.0f;
        ibm.M_val[list_idx * 6 + km] = 0.0f; ibm.X_val[list_idx * 6 + km] = 0.0f;
      } else {
        for (int side = 0; side < 2; ++side) {
          int kk = side == 0 ? kp : km;
          if (is_ghost[kk]) {
            if (SCHEME == 0) {
              ibm.K_val[list_idx * 6 + kk] = poly_Nc(xi_vals[kk]) * R;
              ibm.X_val[list_idx * 6 + kk] = poly_N_nb(xi_vals[kk]) * R;
              ibm.Nbc_val[list_idx * 6 + kk] = poly_Nbc(xi_vals[kk]) * R;
            } else {
              ibm.K_val[list_idx * 6 + kk] = poly_Nc_avg(xi_vals[kk]) * R;
              ibm.X_val[list_idx * 6 + kk] = poly_Nnb_avg(xi_vals[kk]) * R;
              ibm.Nbc_val[list_idx * 6 + kk] = poly_Nbc_avg(xi_vals[kk]) * R;
            }
            ibm.M_val[list_idx * 6 + kk] = 0.0f;
          } else {
            ibm.K_val[list_idx * 6 + kk] = 0.0f;
            ibm.M_val[list_idx * 6 + kk] = 1.0f;
            ibm.X_val[list_idx * 6 + kk] = 0.0f;
            ibm.Nbc_val[list_idx * 6 + kk] = 0.0f;
          }
        }
      }
      ibm.dir_code[list_idx * 6 + kp] = kp;
      ibm.dir_code[list_idx * 6 + km] = km;
    }
  } else {  // Neumann
    ibm.D_rescale[list_idx] = 1.0f;
    for (int k = 0; k < 6; ++k) {
      ibm.dir_code[list_idx * 6 + k] = k;
      ibm.R_val[list_idx * 6 + k] = 1.0f;
      ibm.K_val[list_idx * 6 + k] = is_ghost[k] ? 1.0f : 0.0f;
      ibm.M_val[list_idx * 6 + k] = is_ghost[k] ? 0.0f : 1.0f;
      ibm.X_val[list_idx * 6 + k] = 0.0f;
      ibm.Nbc_val[list_idx * 6 + k] = 0.0f;
    }
  }
}

// Gather the 7 SDF samples (centre + 6 axis neighbours at +/-1) at a staggered point of an extended-
// block cell, using the clamped extended sampler (no wrap).
__device__ inline void ibm_gather_ext(const double* sdf, int3 ext, double lx, double ly, double lz,
                                      float3 off, float& sc, float sn[6]) {
  sc = (float)ccdetail::cc_sample_ext(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
  const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int k = 0; k < 6; ++k)
    sn[k] = (float)ccdetail::cc_sample_ext(sdf, ext, lx + d[k][0] + off.x, ly + d[k][1] + off.y,
                                           lz + d[k][2] + off.z);
}

// count cut cells over inner cells (sizing pass)
__global__ void ibm_count_ext_k(const double* sdf, int3 ext, int g, float3 off, int* counter) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  float sc, sn[6];
  ibm_gather_ext(sdf, ext, lx, ly, lz, off, sc, sn);
  if (ibm_is_cut(sc, sn)) atomicAdd(counter, 1);
}

// fill the IBM_Data SoA + id_map over inner cells of the extended block (no wrap)
template <int SCHEME>
__global__ void ibm_geometry_ext_k(IBM_Data ibm, int* id_map, const double* sdf, int3 ext, int g,
                                   float3 spacing, int* counter, float3 off, int bc_type) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t idx = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  float sc, sn[6];
  ibm_gather_ext(sdf, ext, lx, ly, lz, off, sc, sn);
  if (!ibm_is_cut(sc, sn)) {
    id_map[idx] = -1;
    return;
  }
  int list_idx = atomicAdd(counter, 1);
  id_map[idx] = list_idx;
  ibm_fill_entry<SCHEME>(ibm, list_idx, (int)idx, sc, sn, spacing, bc_type);
}

// Bake the IBM factors into the (double) velocity diffusion stencil + inhomogeneous Dirichlet term.
// Verbatim port of modify_stencil_ibm_kernel (which is itself indexing-agnostic), retyped for double
// stencils. a_inhom[c_idx] accumulates the wall-velocity contribution to be subtracted from the RHS.
__global__ void ibm_modify_stencil_k(double* A_C, double* A_W, double* A_E, double* A_S, double* A_N,
                                     double* A_B, double* A_T, double* a_inhom, IBM_Data ibm,
                                     float u_bc_val) {
  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm.num_active_cells) return;
  int c = ibm.cell_index[list_idx];
  float descale = ibm.D_rescale[list_idx];
  double orig[6] = {A_E[c], A_W[c], A_N[c], A_S[c], A_T[c], A_B[c]};
  A_C[c] *= descale;
  double mod[6] = {0, 0, 0, 0, 0, 0};
  double inhom = 0.0;
  for (int k = 0; k < 6; ++k) {
    float K = ibm.K_val[list_idx * 6 + k], M = ibm.M_val[list_idx * 6 + k];
    float X = ibm.X_val[list_idx * 6 + k], Nbc = ibm.Nbc_val[list_idx * 6 + k];
    double vnb = orig[k];
    A_C[c] += vnb * K;
    inhom += (double)Nbc * u_bc_val * vnb;
    int opp = k ^ 1;  // 0<->1, 2<->3, 4<->5
    mod[k] += vnb * ((double)descale * M - 1.0);
    mod[opp] += vnb * X;
  }
  A_E[c] = orig[0] + mod[0];
  A_W[c] = orig[1] + mod[1];
  A_N[c] = orig[2] + mod[2];
  A_S[c] = orig[3] + mod[3];
  A_T[c] = orig[4] + mod[4];
  A_B[c] = orig[5] + mod[5];
  if (a_inhom) a_inhom[c] += inhom;
}

// build the backward-Euler velocity diffusion stencil over the extended block: A_C = 1 + 6 beta,
// off-diagonals = -beta (dx=1, beta = theta*dt*nu). RHS b is the explicit term; inhom is subtracted.
__global__ void ibm_build_diffusion_k(double* A_C, double* A_W, double* A_E, double* A_S, double* A_N,
                                      double* A_B, double* A_T, int3 ext, double beta) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  A_C[i] = 1.0 + 6.0 * beta;
  A_W[i] = -beta; A_E[i] = -beta; A_S[i] = -beta; A_N[i] = -beta; A_B[i] = -beta; A_T[i] = -beta;
}

// one Red-Black sweep of the (IBM-modified) stencil: A_C x = b - sum(A_off x_nbr), global parity.
__global__ void ibm_rbgs_stencil_k(double* x, const double* b, const double* A_C, const double* A_W,
                                   const double* A_E, const double* A_S, const double* A_N,
                                   const double* A_B, const double* A_T, int3 ext, int3 og, int g,
                                   int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double ac = A_C[i];
  if (fabs(ac) < 1e-30) return;
  double s = A_E[i] * x[i + sx] + A_W[i] * x[i - sx] + A_N[i] * x[i + sy] + A_S[i] * x[i - sy] +
             A_T[i] * x[i + sz] + A_B[i] * x[i - sz];
  x[i] = (b[i] - s) / ac;
}

}  // namespace ibmdetail

// allocate / free the IBM_Data SoA for `n` cut cells (host helpers)
inline IBM_Data ibm_alloc(int n) {
  IBM_Data d{};
  d.num_active_cells = n;
  int m = n > 0 ? n : 1;
  cudaMalloc(&d.cell_index, m * sizeof(int));
  cudaMalloc(&d.D_rescale, m * sizeof(float));
  cudaMalloc(&d.num_boundaries, m * sizeof(int));
  cudaMalloc(&d.dir_code, 6 * m * sizeof(int));
  cudaMalloc(&d.K_val, 6 * m * sizeof(float));
  cudaMalloc(&d.M_val, 6 * m * sizeof(float));
  cudaMalloc(&d.X_val, 6 * m * sizeof(float));
  cudaMalloc(&d.Nbc_val, 6 * m * sizeof(float));
  cudaMalloc(&d.R_val, 6 * m * sizeof(float));
  return d;
}
inline void ibm_free(IBM_Data& d) {
  for (void* p : {(void*)d.cell_index, (void*)d.D_rescale, (void*)d.num_boundaries, (void*)d.dir_code,
                  (void*)d.K_val, (void*)d.M_val, (void*)d.X_val, (void*)d.Nbc_val, (void*)d.R_val})
    if (p) cudaFree(p);
  d = IBM_Data{};
}

}  // namespace cfdmpi
