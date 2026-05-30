// Validate that transport-core's halo reproduces cfd-gpu's periodic operators on cfd's real field
// layout, at the ghost widths the solver needs:
//   radius 1 — diffusion / Laplacian (7-point);
//   radius 2 — Koren TVD advection (reads phi_LL..phi_RR, i.e. reaches +/-2).
// For each radius R we compute a separable +/-1..R stencil two ways: (serial) full grid on the GPU
// using cfd's own get_idx() periodic wrapping; (distributed) per-rank extended block with ghost
// width R, filled by MacGridHalo::exchange(), then a local stencil with direct neighbour indexing.
// The distributed inner-cell results must equal the serial reference cell-for-cell.
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // cfd's get_idx (the exact periodic indexing the solver uses)
#include "mac_halo.cuh"

__host__ __device__ inline double field_val(int x, int y, int z) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL;
}

__global__ void init_full_kernel(double* f, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  f[get_idx(x, y, z, res)] = field_val(x, y, z);
}

// Serial reference: separable +/-1..R stencil via cfd's periodic get_idx.
template <int R>
__global__ void stencil_full_kernel(const double* f, double* out, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  double c = f[get_idx(x, y, z, res)];
  double s = 0.0;
#pragma unroll
  for (int d = 1; d <= R; ++d) {
    double w = 1.0 / d;
    s += w * (f[get_idx(x + d, y, z, res)] + f[get_idx(x - d, y, z, res)] +
              f[get_idx(x, y + d, z, res)] + f[get_idx(x, y - d, z, res)] +
              f[get_idx(x, y, z + d, res)] + f[get_idx(x, y, z - d, res)]);
  }
  out[get_idx(x, y, z, res)] = s - 6.0 * c;
}

// Distributed: same stencil on the extended local block with direct strides (ghosts pre-filled).
template <int R>
__global__ void stencil_local_kernel(const double* f, double* out, int3 ext, int ghost) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
  long idx = lx * sx + ly * sy + lz * sz;
  double c = f[idx];
  double s = 0.0;
#pragma unroll
  for (int d = 1; d <= R; ++d) {
    double w = 1.0 / d;
    s += w * (f[idx + d * sx] + f[idx - d * sx] + f[idx + d * sy] + f[idx - d * sy] +
              f[idx + d * sz] + f[idx - d * sz]);
  }
  out[idx] = s - 6.0 * c;
}

// Returns local mismatch count for stencil radius R with ghost width R.
template <int R>
int run_case(int3 res, int rank, int size) {
  dim3 blk(8, 8, 8);

  // Serial reference on the full grid.
  size_t nfull = (size_t)res.x * res.y * res.z;
  double *d_full = nullptr, *d_ref = nullptr;
  cudaMalloc(&d_full, nfull * sizeof(double));
  cudaMalloc(&d_ref, nfull * sizeof(double));
  dim3 grdF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  init_full_kernel<<<grdF, blk>>>(d_full, res);
  stencil_full_kernel<R><<<grdF, blk>>>(d_full, d_ref, res);
  std::vector<double> ref(nfull);
  cudaMemcpy(ref.data(), d_ref, nfull * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_full);
  cudaFree(d_ref);

  // Distributed via MacGridHalo at ghost width R.
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/R, MPI_COMM_WORLD);
  int3 ext = mac.local_ext;
  size_t nloc = mac.num_local_cells();

  std::vector<double> h_loc(nloc, 0.0);
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        int gx = lx + mac.origin_incl_ghost.x, gy = ly + mac.origin_incl_ghost.y,
            gz = lz + mac.origin_incl_ghost.z;
        h_loc[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y] = field_val(gx, gy, gz);
      }

  double *d_loc = nullptr, *d_out = nullptr;
  cudaMalloc(&d_loc, nloc * sizeof(double));
  cudaMalloc(&d_out, nloc * sizeof(double));
  cudaMemcpy(d_loc, h_loc.data(), nloc * sizeof(double), cudaMemcpyHostToDevice);

  mac.exchange(d_loc);  // fill the width-R ghost layer from neighbours

  dim3 grdL((ext.x + 7) / 8, (ext.y + 7) / 8, (ext.z + 7) / 8);
  stencil_local_kernel<R><<<grdL, blk>>>(d_loc, d_out, ext, mac.ghost);
  std::vector<double> out(nloc);
  cudaMemcpy(out.data(), d_out, nloc * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_loc);
  cudaFree(d_out);

  int fail = 0;
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        int gx = lx + mac.origin_incl_ghost.x, gy = ly + mac.origin_incl_ghost.y,
            gz = lz + mac.origin_incl_ghost.z;
        double got = out[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y];
        double exp = ref[(size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y];
        if (fabs(got - exp) > 1e-9) ++fail;
      }
  return fail;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(40, 28, 24);

  int fail1 = run_case<1>(res, rank, size);  // diffusion reach
  int fail2 = run_case<2>(res, rank, size);  // advection reach (Koren TVD)

  int total = 0, local = fail1 + fail2;
  MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): MacGridHalo reproduces cfd's periodic stencils at ghost width 1 and 2\n",
             size);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches (R1=%d R2=%d on rank0)\n", size, total, fail1,
              fail2);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
