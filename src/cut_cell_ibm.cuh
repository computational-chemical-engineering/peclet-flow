/// @file
/// @brief Shared cut-cell IBM primitives: periodic grid index, the Robust-Scaled IBM overlay struct, and
/// the boundary-distance polynomials. Extracted from the retired pnm_backend so the canonical `sdflow`
/// solver (and its tests) own these directly. Header-only; used by staggered_advection.cuh and mac_ibm.cuh.
#pragma once

#include <cuda_runtime.h>

// Device helper for periodic indexing: linear index I = x + y*nx + z*nx*ny, wrapping each axis.
__device__ inline int get_idx(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

// SoA overlay for the Robust-Scaled cut-cell IBM: per cut cell, the baked-in stencil-modification factors
// (K/M/X/Nbc/R) and the D_rescale row scaling. Built from the SDF geometry, applied to the momentum stencil
// (see mac_ibm.cuh / doc/ibm_overlay.md).
struct IBM_Data {
  int num_active_cells;

  // Per-cell data
  int *cell_index;      // [num_active] global grid index
  float *D_rescale;     // [num_active] row scaling factor D_rescale
  int *num_boundaries;  // [num_active] number of modified directions

  // Per-direction data (size 6 * num_active; access [list_idx * 6 + k], k = 0..5 = X+,X-,Y+,Y-,Z+,Z-)
  int *dir_code;   // direction code per modified face
  float *K_val;    // factor K (fold neighbour into the centre)
  float *M_val;    // factor M (scale the neighbour coefficient)
  float *X_val;    // factor X (cross-term, non-zero only for a sandwiched axis)
  float *Nbc_val;  // geometric factor N_bc * R (multiplies the wall velocity u_bc)
  float *R_val;    // D_rescale / D_axis ratio per direction
};

// --------------------------------------------------------
// Robust-Scaled IBM boundary-distance polynomials (xi = fractional fluid distance to the wall along a face).
// Table 1: point-value schemes.
__device__ inline float poly_D(float xi) { return xi * (1.0f + xi); }
__device__ inline float poly_N_nb(float xi) { return xi * (1.0f - xi); }
__device__ inline float poly_Nc(float xi) { return 2.0f * (xi * xi - 1.0f); }
__device__ inline float poly_Nbc(float xi) { return 2.0f; }

// Table 1: cell-average schemes.
__device__ inline float poly_D_avg(float xi) { return xi * (1.0f + xi) - 1.0f / 12.0f; }
__device__ inline float poly_Nnb_avg(float xi) { return xi * (1.0f - xi) + 1.0f / 12.0f; }
__device__ inline float poly_Nc_avg(float xi) { return 2.0f * (xi * xi - 1.0f) - 1.0f / 6.0f; }
__device__ inline float poly_Nbc_avg(float xi) { return 2.0f; }

// Table 2: sandwiched (double-sided) point-value. Neighbours at xi_m (minus) and xi_p (plus).
__device__ inline float poly_D_sandwich(float xi_m, float xi_p) { return xi_m * xi_p; }
__device__ inline float poly_N_c_sandwich(float xi_m, float xi_p) {
  return (xi_m + 1.0f) * (xi_p - 1.0f);  // N_{c,+} factor for the plus-side ghost
}
__device__ inline float poly_Nbc_pp_sw(float xi_m, float xi_p) {
  return (xi_m / (xi_m + xi_p)) * (1.0f + xi_m);
}
__device__ inline float poly_Nbc_mp_sw(float xi_m, float xi_p) {
  return (xi_p / (xi_m + xi_p)) * (1.0f - xi_p);
}

// Table 2: sandwiched (double-sided) cell-average.
__device__ inline float poly_D_sandwich_avg(float xi_m, float xi_p) { return xi_m * xi_p - 1.0f / 12.0f; }
__device__ inline float poly_N_c_sandwich_avg(float xi_m, float xi_p) {
  return (xi_m + 1.0f) * (xi_p - 1.0f) - 1.0f / 12.0f;
}
__device__ inline float poly_Nbc_pp_sw_avg(float xi_m, float xi_p) {
  return (xi_m / (xi_m + xi_p)) * (1.0f + xi_m) - 1.0f / 12.0f;
}
__device__ inline float poly_Nbc_mp_sw_avg(float xi_m, float xi_p) {
  return (xi_p / (xi_m + xi_p)) * (1.0f - xi_p) + 1.0f / 12.0f;
}
