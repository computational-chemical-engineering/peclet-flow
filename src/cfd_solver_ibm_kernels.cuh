#pragma once
#include "cfd_solver.cuh"
#include <cstdio>

// --------------------------------------------------------
// Robust Scaled IBM Geometry Helper: Polynomials
// --------------------------------------------------------

// Table 1: Point-value schemes
__device__ inline float poly_D(float xi) { return xi * (1.0f + xi); }
__device__ inline float poly_N_nb(float xi) { return xi * (1.0f - xi); }
__device__ inline float poly_Nc(float xi) { return 2.0f * (xi * xi - 1.0f); }
__device__ inline float poly_Nbc(float xi) { return 2.0f; }

// Table 1: Cell-average schemes
__device__ inline float poly_D_avg(float xi) { return xi * (1.0f + xi) - 1.0f / 12.0f; }
__device__ inline float poly_Nnb_avg(float xi) {
  return xi * (1.0f - xi) + 1.0f / 12.0f;
}
__device__ inline float poly_Nc_avg(float xi) {
  return 2.0f * (xi * xi - 1.0f) - 1.0f / 6.0f;
}
__device__ inline float poly_Nbc_avg(float xi) { return 2.0f; }

// Table 2: Sandwiched (Double-sided) Point-value
// Neighbors at xi_m (minus) and xi_p (plus)
__device__ inline float poly_D_sandwich(float xi_m, float xi_p) {
  return (xi_m + xi_p) * xi_m * xi_p;
}
__device__ inline float poly_N_c_sandwich(float xi_m, float xi_p) {
  // N_{c,+} term (factor for Plus-side ghost)
  return (xi_m + xi_p) * (xi_m + 1.0f) * (xi_p - 1.0f);
}

// Sandwich Boundary Factors (Point-Value)
__device__ inline float poly_Nbc_pp_sw(float xi_m, float xi_p) { return xi_m * (1.0f + xi_m); }
__device__ inline float poly_Nbc_mp_sw(float xi_m, float xi_p) { return xi_p * (1.0f - xi_p); }

// Table 2: Sandwiched (Double-sided) Cell-average
__device__ inline float poly_D_sandwich_avg(float xi_m, float xi_p) {
  return (xi_m + xi_p) * (xi_m * xi_p - 1.0f / 12.0f);
}
__device__ inline float poly_N_c_sandwich_avg(float xi_m, float xi_p) {
  // factor for Plus-side ghost
  return (xi_m + xi_p) * ((xi_m + 1.0f) * (xi_p - 1.0f) - 1.0f / 12.0f);
}
__device__ inline float poly_Nbc_pp_sw_avg(float xi_m, float xi_p) {
  return xi_m * (1.0f + xi_m) - 1.0f / 12.0f;
}
__device__ inline float poly_Nbc_mp_sw_avg(float xi_m, float xi_p) {
  return xi_p * (1.0f - xi_p) + 1.0f / 12.0f;
}

// Trilinear Interpolation of periodic SDF
__device__ inline float sample_sdf_interp(float x, float y, float z,
                                   const float *__restrict__ sdf, int3 res) {
  // 1. Floor to get base index
  float fx = floorf(x);
  float fy = floorf(y);
  float fz = floorf(z);

  // 2. Fraction
  float wx = x - fx;
  float wy = y - fy;
  float wz = z - fz;

  // 3. Base integer indices (wrapped)
  int ix = (int)fx;
  int iy = (int)fy;
  int iz = (int)fz;

  // 4. Neighbors
  int x0 = (ix % res.x + res.x) % res.x;
  int y0 = (iy % res.y + res.y) % res.y;
  int z0 = (iz % res.z + res.z) % res.z;

  int x1 = (x0 + 1) % res.x;
  int y1 = (y0 + 1) % res.y;
  int z1 = (z0 + 1) % res.z;

  // 5. Fetch 8 corners
  // Order: 000, 100, 010, 110, 001, 101, 011, 111
  float c000 = sdf[z0 * res.y * res.x + y0 * res.x + x0];
  float c100 = sdf[z0 * res.y * res.x + y0 * res.x + x1];
  float c010 = sdf[z0 * res.y * res.x + y1 * res.x + x0];
  float c110 = sdf[z0 * res.y * res.x + y1 * res.x + x1];
  float c001 = sdf[z1 * res.y * res.x + y0 * res.x + x0];
  float c101 = sdf[z1 * res.y * res.x + y0 * res.x + x1];
  float c011 = sdf[z1 * res.y * res.x + y1 * res.x + x0];
  float c111 = sdf[z1 * res.y * res.x + y1 * res.x + x1];

  // 6. Interpolate (Lerp X, then Y, then Z)
  float c00 = c000 * (1.0f - wx) + c100 * wx;
  float c10 = c010 * (1.0f - wx) + c110 * wx;
  float c01 = c001 * (1.0f - wx) + c101 * wx;
  float c11 = c011 * (1.0f - wx) + c111 * wx;

  float c0 = c00 * (1.0f - wy) + c10 * wy;
  float c1 = c01 * (1.0f - wy) + c11 * wy;

  return c0 * (1.0f - wz) + c1 * wz;
}

// --------------------------------------------------------
// Compute IBM Geometry Kernel (Robust Scaled)
// --------------------------------------------------------
static __global__ void count_ibm_cells_kernel(const float *__restrict__ sdf,
                                              int3 res, float3 offset,
                                              int *counter) {
  // Lightweight count pass for compact IBM allocation. The subsequent geometry
  // kernel performs the authoritative fill and records the final active count.
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  float sdf_c = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);
  if (sdf_c <= 0.0f)
    return;

  const int dirs[6][3] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
      {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
  };
  for (int k = 0; k < 6; ++k) {
    float sdf_n = sample_sdf_interp(idx_x + dirs[k][0] + offset.x,
                                    idx_y + dirs[k][1] + offset.y,
                                    idx_z + dirs[k][2] + offset.z, sdf, res);
    if (sdf_n < 0.0f) {
      atomicAdd(counter, 1);
      return;
    }
  }
}

// bc_type: 0 = Dirichlet (Robust Scaled Polynomials)
//          1 = Neumann (Zero Gradient: phi_g = phi_c => K=1, M=0)
template <int SCHEME>
__global__ void compute_ibm_geometry_kernel(IBM_Data ibm_data, int *ibm_id_map,
                                            const float *__restrict__ sdf,
                                            int3 res, float3 spacing,
                                            int *counter, float3 offset,
                                            int bc_type) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);

  // Sample SDF
  float sdf_c = sample_sdf_interp(idx_x + offset.x, idx_y + offset.y,
                                  idx_z + offset.z, sdf, res);

  // Determine if this cell is solid or fluid
  bool is_solid_c = (sdf_c <= 0.0f);
  if (is_solid_c) {
    ibm_id_map[idx] = -1;
    return;
  }

  // Identify neighbors
  float D_vals[6];
  float xi_vals[6];
  bool is_ghost[6];
  bool is_sandwich[3];

  int num_interface_neighbors = 0;

  // Check Directions
  auto check_dir = [&](int k, int dx, int dy, int dz) {
    float sdf_n =
        sample_sdf_interp(idx_x + dx + offset.x, idx_y + dy + offset.y,
                          idx_z + dz + offset.z, sdf, res);
    
    // Interface detected if sign differs
    bool is_solid_n = (sdf_n < 0.0f);
    
    if (is_solid_c != is_solid_n) {
      is_ghost[k] = true;
      if (bc_type == 0) {
        // Theta works regardless of sign direction (ratio of distances)
        float theta = sdf_c / (sdf_c - sdf_n);
        if (theta < 1e-4f)
          theta = 1e-4f;
        if (theta > 1.0f)
          theta = 1.0f;
        xi_vals[k] = theta;
        
        if (SCHEME == 0) {
            D_vals[k] = poly_D(theta); 
        } else {
            D_vals[k] = poly_D_avg(theta);
        }
      } else {
        // Neumann: K=1, M=0
        xi_vals[k] = 0.5f;
        D_vals[k] = 1.0f;
      }
      num_interface_neighbors++;
    } else {
      is_ghost[k] = false;
      xi_vals[k] = 1.0f;
      D_vals[k] = 1e9f;
    }
  };

  check_dir(0, 1, 0, 0);  // +X
  check_dir(1, -1, 0, 0); // -X
  check_dir(2, 0, 1, 0);  // +Y
  check_dir(3, 0, -1, 0); // -Y
  check_dir(4, 0, 0, 1);  // +Z
  check_dir(5, 0, 0, -1); // -Z

  // Add to list ONLY if it touches an interface
  // (Interior Fluid and Interior Solid are skipped)
  if (num_interface_neighbors == 0) {
    ibm_id_map[idx] = -1;
    return;
  }

  // Allocate
  int list_idx = atomicAdd(counter, 1);
  ibm_id_map[idx] = list_idx;
  ibm_data.cell_index[list_idx] = idx;
  ibm_data.num_boundaries[list_idx] = 6;

    // Dirichlet (Robust Scaled) Calculation
  if (bc_type == 0) {
    is_sandwich[0] = is_ghost[0] && is_ghost[1];
    is_sandwich[1] = is_ghost[2] && is_ghost[3];
    is_sandwich[2] = is_ghost[4] && is_ghost[5];

    float D_sandwich[3] = {0, 0, 0};
    if (is_sandwich[0]) {
      if (SCHEME == 0) D_sandwich[0] = poly_D_sandwich(xi_vals[1], xi_vals[0]);
      else D_sandwich[0] = poly_D_sandwich_avg(xi_vals[1], xi_vals[0]);
    }
    if (is_sandwich[1]) {
      if (SCHEME == 0) D_sandwich[1] = poly_D_sandwich(xi_vals[3], xi_vals[2]);
      else D_sandwich[1] = poly_D_sandwich_avg(xi_vals[3], xi_vals[2]);
    }
    if (is_sandwich[2]) {
      if (SCHEME == 0) D_sandwich[2] = poly_D_sandwich(xi_vals[5], xi_vals[4]);
      else D_sandwich[2] = poly_D_sandwich_avg(xi_vals[5], xi_vals[4]);
    }

    float min_D_abs = 1e30f;
    float D_rescale = 1.0f;
    auto update_min = [&](float val) {
      if (fabsf(val) < min_D_abs) {
        min_D_abs = fabsf(val);
        D_rescale = val;
      }
    };

    for (int axis = 0; axis < 3; axis++) {
      if (is_sandwich[axis]) update_min(D_sandwich[axis]);
      else {
        if (is_ghost[2 * axis]) update_min(D_vals[2 * axis]);
        if (is_ghost[2 * axis + 1]) update_min(D_vals[2 * axis + 1]);
      }
    }

    ibm_data.D_rescale[list_idx] = D_rescale;

    for (int axis = 0; axis < 3; axis++) {
      int km = 2 * axis + 1;
      int kp = 2 * axis;

      float D_axis = 0.0f;
      bool sandwich = is_sandwich[axis];
      bool g_p = is_ghost[kp];
      bool g_m = is_ghost[km];

      if (sandwich)
        D_axis = D_sandwich[axis];
      else if (g_p)
        D_axis = D_vals[kp];
      else if (g_m)
        D_axis = D_vals[km];
      else
        D_axis = D_rescale;

      float R = D_rescale / D_axis;
      if (fabsf(D_axis) < 1e-9f)
        R = 1.0f;
      ibm_data.R_val[list_idx * 6 + kp] = R;
      ibm_data.R_val[list_idx * 6 + km] = R;

      if (sandwich) {
        if (SCHEME == 0) {
            float N_c_plus = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]);
            ibm_data.K_val[list_idx * 6 + kp] = N_c_plus * R;
            
            float N_c_minus = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]);
            ibm_data.K_val[list_idx * 6 + km] = N_c_minus * R;

            float Npp = poly_Nbc_pp_sw(xi_vals[km], xi_vals[kp]);
            float Nmp = poly_Nbc_mp_sw(xi_vals[km], xi_vals[kp]);
            ibm_data.Nbc_val[list_idx * 6 + kp] = (Npp + Nmp) * R;

            float Nmm = poly_Nbc_pp_sw(xi_vals[kp], xi_vals[km]);
            float Npm = poly_Nbc_mp_sw(xi_vals[kp], xi_vals[km]);
            ibm_data.Nbc_val[list_idx * 6 + km] = (Nmm + Npm) * R;
        } else {
            float N_c_plus = poly_N_c_sandwich_avg(xi_vals[km], xi_vals[kp]);
            ibm_data.K_val[list_idx * 6 + kp] = N_c_plus * R;
            
            float N_c_minus = poly_N_c_sandwich_avg(xi_vals[kp], xi_vals[km]);
            ibm_data.K_val[list_idx * 6 + km] = N_c_minus * R;

            float Npp = poly_Nbc_pp_sw_avg(xi_vals[km], xi_vals[kp]);
            float Nmp = poly_Nbc_mp_sw_avg(xi_vals[km], xi_vals[kp]);
            ibm_data.Nbc_val[list_idx * 6 + kp] = (Npp + Nmp) * R;

            float Nmm = poly_Nbc_pp_sw_avg(xi_vals[kp], xi_vals[km]);
            float Npm = poly_Nbc_mp_sw_avg(xi_vals[kp], xi_vals[km]);
            ibm_data.Nbc_val[list_idx * 6 + km] = (Nmm + Npm) * R;
        }
        ibm_data.M_val[list_idx * 6 + kp] = 0.0f;
        ibm_data.X_val[list_idx * 6 + kp] = 0.0f;
        ibm_data.M_val[list_idx * 6 + km] = 0.0f;
        ibm_data.X_val[list_idx * 6 + km] = 0.0f;
      } else {
        if (is_ghost[kp]) {
          if (SCHEME == 0) {
              ibm_data.K_val[list_idx * 6 + kp] = poly_Nc(xi_vals[kp]) * R;
              ibm_data.X_val[list_idx * 6 + kp] = poly_N_nb(xi_vals[kp]) * R;
              ibm_data.Nbc_val[list_idx * 6 + kp] = poly_Nbc(xi_vals[kp]) * R;
          } else {
              ibm_data.K_val[list_idx * 6 + kp] = poly_Nc_avg(xi_vals[kp]) * R;
              ibm_data.X_val[list_idx * 6 + kp] = poly_Nnb_avg(xi_vals[kp]) * R;
              ibm_data.Nbc_val[list_idx * 6 + kp] = poly_Nbc_avg(xi_vals[kp]) * R;
          }
          ibm_data.M_val[list_idx * 6 + kp] = 0.0f;
        } else {
          ibm_data.K_val[list_idx * 6 + kp] = 0.0f;
          ibm_data.M_val[list_idx * 6 + kp] = 1.0f;
          ibm_data.X_val[list_idx * 6 + kp] = 0.0f;
          ibm_data.Nbc_val[list_idx * 6 + kp] = 0.0f;
        }
        if (is_ghost[km]) {
          if (SCHEME == 0) {
              ibm_data.K_val[list_idx * 6 + km] = poly_Nc(xi_vals[km]) * R;
              ibm_data.X_val[list_idx * 6 + km] = poly_N_nb(xi_vals[km]) * R;
              ibm_data.Nbc_val[list_idx * 6 + km] = poly_Nbc(xi_vals[km]) * R;
          } else {
              ibm_data.K_val[list_idx * 6 + km] = poly_Nc_avg(xi_vals[km]) * R;
              ibm_data.X_val[list_idx * 6 + km] = poly_Nnb_avg(xi_vals[km]) * R;
              ibm_data.Nbc_val[list_idx * 6 + km] = poly_Nbc_avg(xi_vals[km]) * R;
          }
          ibm_data.M_val[list_idx * 6 + km] = 0.0f;
        } else {
          ibm_data.K_val[list_idx * 6 + km] = 0.0f;
          ibm_data.M_val[list_idx * 6 + km] = 1.0f;
          ibm_data.X_val[list_idx * 6 + km] = 0.0f;
          ibm_data.Nbc_val[list_idx * 6 + km] = 0.0f;
        }
      }
      ibm_data.dir_code[list_idx * 6 + kp] = kp;
      ibm_data.dir_code[list_idx * 6 + km] = km;
    }

  } else {
    // Neumann (Simple K=1, M=0)
    ibm_data.D_rescale[list_idx] = 1.0f;
    for (int k = 0; k < 6; k++) {
      ibm_data.dir_code[list_idx * 6 + k] = k;
      ibm_data.R_val[list_idx * 6 + k] = 1.0f;
      if (is_ghost[k]) {
        // Ghost: phi_g = phi_c
        // Term in stencil: A_nb * phi_g = A_nb * phi_c.
        // Modify: A_c += A_nb * 1. A_nb *= 0.
        ibm_data.K_val[list_idx * 6 + k] = 1.0f;
        ibm_data.M_val[list_idx * 6 + k] = 0.0f;
        ibm_data.X_val[list_idx * 6 + k] = 0.0f;
        ibm_data.Nbc_val[list_idx * 6 + k] = 0.0f;
      } else {
        // Fluid: No Change
        ibm_data.K_val[list_idx * 6 + k] = 0.0f;
        ibm_data.M_val[list_idx * 6 + k] = 1.0f;
        ibm_data.X_val[list_idx * 6 + k] = 0.0f;
        ibm_data.Nbc_val[list_idx * 6 + k] = 0.0f;
      }
    }
  }
}
