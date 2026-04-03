#include "cfd_solver.cuh"
#include "cfd_solver_ibm_kernels.cuh"

#include <cmath>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace {

inline int get_idx_host(int x, int y, int z, int3 res) {
  x = (x % res.x + res.x) % res.x;
  y = (y % res.y + res.y) % res.y;
  z = (z % res.z + res.z) % res.z;
  return z * res.y * res.x + y * res.x + x;
}

float sample_field_interp_host(const std::vector<float> &field, int3 res, float x,
                               float y, float z) {
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

__device__ float sample_field_interp_device(const float *__restrict__ field,
                                            float x, float y, float z,
                                            int3 res) {
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

template <typename T>
__global__ void solve_rbgs_pressure_kernel(
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

  if (((idx_x + idx_y + idx_z) % 2 == 0) != is_red)
    return;

  int idx = get_idx(idx_x, idx_y, idx_z, res);
  float ac = A_C[idx];
  if (fabsf(ac) < 1e-12f)
    return;

  int idx_E = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_W = get_idx(idx_x - 1, idx_y, idx_z, res);
  int idx_N = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_S = get_idx(idx_x, idx_y - 1, idx_z, res);
  int idx_T = get_idx(idx_x, idx_y, idx_z + 1, res);
  int idx_B = get_idx(idx_x, idx_y, idx_z - 1, res);

  T sum = A_E[idx] * phi[idx_E] + A_W[idx] * phi[idx_W] +
          A_N[idx] * phi[idx_N] + A_S[idx] * phi[idx_S] +
          A_T[idx] * phi[idx_T] + A_B[idx] * phi[idx_B];
  phi[idx] = (B_RHS[idx] - sum) / ac;
}

__global__ void compute_residual_kernel(
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

__global__ void restrict_cell_average_kernel(float *__restrict__ coarse,
                                             const float *__restrict__ fine,
                                             int3 coarse_res, int3 fine_res) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= coarse_res.x || idx_y >= coarse_res.y || idx_z >= coarse_res.z)
    return;

  float sum = 0.0f;
  for (int dz = 0; dz < 2; ++dz) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        sum += fine[get_idx(2 * idx_x + dx, 2 * idx_y + dy, 2 * idx_z + dz,
                            fine_res)];
      }
    }
  }
  coarse[get_idx(idx_x, idx_y, idx_z, coarse_res)] = 0.125f * sum;
}

__global__ void prolongate_trilinear_add_kernel(
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
      sample_field_interp_device(coarse, x, y, z, coarse_res);
}

__global__ void subtract_mean_mg_kernel(float *__restrict__ data, float mean,
                                        int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  data[idx] -= mean;
}

} // namespace

__global__ void compute_pressure_operator_kernel(
    float *__restrict__ A_C, float *__restrict__ A_W, float *__restrict__ A_E,
    float *__restrict__ A_S, float *__restrict__ A_N, float *__restrict__ A_B,
    float *__restrict__ A_T, const float *__restrict__ frac_u,
    const float *__restrict__ frac_v, const float *__restrict__ frac_w,
    const float *__restrict__ sdf, int3 res, float3 spacing) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  float inv_dx2 = 1.0f / (spacing.x * spacing.x);
  float inv_dy2 = 1.0f / (spacing.y * spacing.y);
  float inv_dz2 = 1.0f / (spacing.z * spacing.z);

  int idx_xp = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_yp = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_zp = get_idx(idx_x, idx_y, idx_z + 1, res);

  float sdf_u_r = sample_sdf_interp(idx_x + 0.5f, idx_y, idx_z, sdf, res);
  float sdf_u_l = sample_sdf_interp(idx_x - 0.5f, idx_y, idx_z, sdf, res);
  float sdf_v_n = sample_sdf_interp(idx_x, idx_y + 0.5f, idx_z, sdf, res);
  float sdf_v_s = sample_sdf_interp(idx_x, idx_y - 0.5f, idx_z, sdf, res);
  float sdf_w_t = sample_sdf_interp(idx_x, idx_y, idx_z + 0.5f, sdf, res);
  float sdf_w_b = sample_sdf_interp(idx_x, idx_y, idx_z - 0.5f, sdf, res);

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

  A_E[idx] = -ax_p;
  A_W[idx] = -ax_m;
  A_N[idx] = -ay_p;
  A_S[idx] = -ay_m;
  A_T[idx] = -az_p;
  A_B[idx] = -az_m;
  A_C[idx] = ax_p + ax_m + ay_p + ay_m + az_p + az_m;
}

__global__ void compute_pressure_rhs_kernel(
    float *__restrict__ B_RHS, const float *__restrict__ frac_u,
    const float *__restrict__ frac_v, const float *__restrict__ frac_w,
    const float *__restrict__ sdf, const double *__restrict__ u,
    const double *__restrict__ v, const double *__restrict__ w, int3 res,
    float3 spacing) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx_x >= res.x || idx_y >= res.y || idx_z >= res.z)
    return;
  int idx = get_idx(idx_x, idx_y, idx_z, res);

  int idx_xp = get_idx(idx_x + 1, idx_y, idx_z, res);
  int idx_yp = get_idx(idx_x, idx_y + 1, idx_z, res);
  int idx_zp = get_idx(idx_x, idx_y, idx_z + 1, res);

  float sdf_u_r = sample_sdf_interp(idx_x + 0.5f, idx_y, idx_z, sdf, res);
  float sdf_u_l = sample_sdf_interp(idx_x - 0.5f, idx_y, idx_z, sdf, res);
  float sdf_v_n = sample_sdf_interp(idx_x, idx_y + 0.5f, idx_z, sdf, res);
  float sdf_v_s = sample_sdf_interp(idx_x, idx_y - 0.5f, idx_z, sdf, res);
  float sdf_w_t = sample_sdf_interp(idx_x, idx_y, idx_z + 0.5f, sdf, res);
  float sdf_w_b = sample_sdf_interp(idx_x, idx_y, idx_z - 0.5f, sdf, res);

  float frac_u_r = (sdf_u_r > 0.0f) ? frac_u[idx_xp] : 0.0f;
  float frac_u_l = (sdf_u_l > 0.0f) ? frac_u[idx] : 0.0f;
  float frac_v_n = (sdf_v_n > 0.0f) ? frac_v[idx_yp] : 0.0f;
  float frac_v_s = (sdf_v_s > 0.0f) ? frac_v[idx] : 0.0f;
  float frac_w_t = (sdf_w_t > 0.0f) ? frac_w[idx_zp] : 0.0f;
  float frac_w_b = (sdf_w_b > 0.0f) ? frac_w[idx] : 0.0f;

  double du_dx = (u[idx_xp] * frac_u_r - u[idx] * frac_u_l) / spacing.x;
  double dv_dy = (v[idx_yp] * frac_v_n - v[idx] * frac_v_s) / spacing.y;
  double dw_dz = (w[idx_zp] * frac_w_t - w[idx] * frac_w_b) / spacing.z;
  B_RHS[idx] = static_cast<float>(-(du_dx + dv_dy + dw_dz));
}

void CFDSolver::remove_mean_from_device_vector(float *d_data, int num_elements) {
  if (num_elements <= 0)
    return;
  thrust::device_ptr<float> ptr(d_data);
  float sum = thrust::reduce(ptr, ptr + num_elements, 0.0f, thrust::plus<float>());
  float mean = sum / static_cast<float>(num_elements);
  subtract_mean_mg_kernel<<<(num_elements + 255) / 256, 256>>>(d_data, mean,
                                                                num_elements);
  CHECK_CUDA(cudaGetLastError());
}

void CFDSolver::pressure_smooth_level(PressureMGLevel &level, int sweeps) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                (level.res.y + block.y - 1) / block.y,
                (level.res.z + block.z - 1) / block.z);
  for (int k = 0; k < sweeps; ++k) {
    solve_rbgs_pressure_kernel<<<grid_dim, block>>>(
        level.x, level.A_C, level.A_W, level.A_E, level.A_S, level.A_N,
        level.A_B, level.A_T, level.rhs, level.res, true);
    solve_rbgs_pressure_kernel<<<grid_dim, block>>>(
        level.x, level.A_C, level.A_W, level.A_E, level.A_S, level.A_N,
        level.A_B, level.A_T, level.rhs, level.res, false);
  }
  CHECK_CUDA(cudaGetLastError());
}

void CFDSolver::pressure_v_cycle(int level_idx) {
  PressureMGLevel &level = pressure_mg_levels_[level_idx];
  const bool is_bottom = (level_idx + 1 == static_cast<int>(pressure_mg_levels_.size()));
  if (is_bottom) {
    pressure_smooth_level(level, pressure_mg_bottom_sweeps_);
    remove_mean_from_device_vector(level.x, level.num_elements);
    return;
  }

  pressure_smooth_level(level, pressure_mg_pre_sweeps_);

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                (level.res.y + block.y - 1) / block.y,
                (level.res.z + block.z - 1) / block.z);

  compute_residual_kernel<<<grid_dim, block>>>(
      level.residual, level.x, level.A_C, level.A_W, level.A_E, level.A_S,
      level.A_N, level.A_B, level.A_T, level.rhs, level.res);
  CHECK_CUDA(cudaGetLastError());

  PressureMGLevel &coarse = pressure_mg_levels_[level_idx + 1];
  CHECK_CUDA(cudaMemset(coarse.x, 0, coarse.num_elements * sizeof(float)));
  dim3 coarse_grid_dim((coarse.res.x + block.x - 1) / block.x,
                       (coarse.res.y + block.y - 1) / block.y,
                       (coarse.res.z + block.z - 1) / block.z);
  restrict_cell_average_kernel<<<coarse_grid_dim, block>>>(
      coarse.rhs, level.residual, coarse.res, level.res);
  CHECK_CUDA(cudaGetLastError());
  remove_mean_from_device_vector(coarse.rhs, coarse.num_elements);

  pressure_v_cycle(level_idx + 1);

  prolongate_trilinear_add_kernel<<<grid_dim, block>>>(level.x, coarse.x,
                                                       level.res, coarse.res);
  CHECK_CUDA(cudaGetLastError());

  pressure_smooth_level(level, pressure_mg_post_sweeps_);
  remove_mean_from_device_vector(level.x, level.num_elements);
}

void CFDSolver::free_pressure_multigrid() {
  auto free_if = [](float *ptr) {
    if (ptr != nullptr)
      CHECK_CUDA(cudaFree(ptr));
  };
  for (auto &level : pressure_mg_levels_) {
    free_if(level.sdf);
    free_if(level.frac_u);
    free_if(level.frac_v);
    free_if(level.frac_w);
    free_if(level.A_C);
    free_if(level.A_W);
    free_if(level.A_E);
    free_if(level.A_S);
    free_if(level.A_N);
    free_if(level.A_B);
    free_if(level.A_T);
    free_if(level.x);
    free_if(level.rhs);
    free_if(level.residual);
  }
  pressure_mg_levels_.clear();
  pressure_mg_built_ = false;
}

void CFDSolver::build_pressure_multigrid() {
  free_pressure_multigrid();
  if (!pressure_multigrid_enabled_)
    return;

  std::vector<int3> level_res;
  std::vector<float3> level_spacing;
  std::vector<std::vector<float>> host_sdf_levels;

  level_res.push_back(grid.res);
  level_spacing.push_back(grid.spacing);
  host_sdf_levels.emplace_back(grid.num_elements);
  CHECK_CUDA(cudaMemcpy(host_sdf_levels[0].data(), grid.sdf,
                        grid.num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  while (static_cast<int>(level_res.size()) < pressure_mg_max_levels_) {
    int3 fine_res = level_res.back();
    if ((fine_res.x % 2) != 0 || (fine_res.y % 2) != 0 || (fine_res.z % 2) != 0)
      break;
    if (fine_res.x < 8 || fine_res.y < 8 || fine_res.z < 8)
      break;

    int3 coarse_res = make_int3(fine_res.x / 2, fine_res.y / 2, fine_res.z / 2);
    float3 fine_spacing = level_spacing.back();
    float3 coarse_spacing =
        make_float3(2.0f * fine_spacing.x, 2.0f * fine_spacing.y, 2.0f * fine_spacing.z);

    std::vector<float> coarse_sdf(coarse_res.x * coarse_res.y * coarse_res.z);
    const auto &fine_sdf = host_sdf_levels.back();
    for (int z = 0; z < coarse_res.z; ++z) {
      for (int y = 0; y < coarse_res.y; ++y) {
        for (int x = 0; x < coarse_res.x; ++x) {
          coarse_sdf[get_idx_host(x, y, z, coarse_res)] =
              sample_field_interp_host(fine_sdf, fine_res, 2.0f * x + 0.5f,
                                       2.0f * y + 0.5f, 2.0f * z + 0.5f);
        }
      }
    }

    level_res.push_back(coarse_res);
    level_spacing.push_back(coarse_spacing);
    host_sdf_levels.push_back(std::move(coarse_sdf));
  }

  pressure_mg_levels_.resize(level_res.size());
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

  for (size_t level_idx = 0; level_idx < pressure_mg_levels_.size(); ++level_idx) {
    PressureMGLevel &level = pressure_mg_levels_[level_idx];
    level.res = level_res[level_idx];
    level.spacing = level_spacing[level_idx];
    level.num_elements = level.res.x * level.res.y * level.res.z;

    size_t bytes = static_cast<size_t>(level.num_elements) * sizeof(float);
    CHECK_CUDA(cudaMalloc(&level.sdf, bytes));
    CHECK_CUDA(cudaMalloc(&level.frac_u, bytes));
    CHECK_CUDA(cudaMalloc(&level.frac_v, bytes));
    CHECK_CUDA(cudaMalloc(&level.frac_w, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_C, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_W, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_E, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_S, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_N, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_B, bytes));
    CHECK_CUDA(cudaMalloc(&level.A_T, bytes));
    CHECK_CUDA(cudaMalloc(&level.x, bytes));
    CHECK_CUDA(cudaMalloc(&level.rhs, bytes));
    CHECK_CUDA(cudaMalloc(&level.residual, bytes));

    CHECK_CUDA(cudaMemcpy(level.sdf, host_sdf_levels[level_idx].data(), bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(level.x, 0, bytes));
    CHECK_CUDA(cudaMemset(level.rhs, 0, bytes));
    CHECK_CUDA(cudaMemset(level.residual, 0, bytes));

    dim3 grid_dim((level.res.x + block.x - 1) / block.x,
                  (level.res.y + block.y - 1) / block.y,
                  (level.res.z + block.z - 1) / block.z);

    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.frac_u, level.res, level.spacing,
        make_float3(-0.5f, 0.0f, 0.0f), 1);
    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.frac_v, level.res, level.spacing,
        make_float3(0.0f, -0.5f, 0.0f), 2);
    compute_fluid_fraction_kernel<<<grid_dim, block>>>(
        level.sdf, level.frac_w, level.res, level.spacing,
        make_float3(0.0f, 0.0f, -0.5f), 3);
    compute_pressure_operator_kernel<<<grid_dim, block>>>(
        level.A_C, level.A_W, level.A_E, level.A_S, level.A_N, level.A_B,
        level.A_T, level.frac_u, level.frac_v, level.frac_w, level.sdf,
        level.res, level.spacing);
    CHECK_CUDA(cudaGetLastError());
  }

  pressure_mg_built_ = true;
}

void CFDSolver::set_pressure_multigrid_enabled(bool enabled) {
  pressure_multigrid_enabled_ = enabled;
  pressure_mg_built_ = false;
  if (!enabled)
    free_pressure_multigrid();
}

void CFDSolver::set_pressure_multigrid_params(int max_levels, int pre_sweeps,
                                              int post_sweeps,
                                              int bottom_sweeps,
                                              int v_cycles) {
  pressure_mg_max_levels_ = max_levels;
  pressure_mg_pre_sweeps_ = pre_sweeps;
  pressure_mg_post_sweeps_ = post_sweeps;
  pressure_mg_bottom_sweeps_ = bottom_sweeps;
  pressure_mg_v_cycles_ = v_cycles;
  pressure_mg_built_ = false;
}
