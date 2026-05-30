// Validate that transport-core's halo reproduces cfd-gpu's periodic operator on cfd's real field
// layout. We compute the 7-point periodic Laplacian of a double field two ways:
//   (serial)      full grid on the GPU using cfd's own get_idx() periodic wrapping;
//   (distributed) per-rank extended block: fill ghosts with MacGridHalo::exchange(), then a local
//                 7-point stencil with direct (non-wrapping) neighbour indexing.
// The distributed inner-cell results must equal the serial reference at the corresponding global
// cells (to round-off). This proves the shared decomposition + async halo preserves cfd's operator,
// i.e. the solver's stencils can run block-distributed with halo exchange replacing global wrapping.
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // cfd's get_idx (the exact periodic indexing the solver uses)
#include "mac_halo.cuh"

// Deterministic field value from global coordinates (host + device identical).
__host__ __device__ inline double field_val(int x, int y, int z) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL;
}

// Serial reference: full-grid init + periodic Laplacian via cfd's get_idx.
__global__ void init_full_kernel(double* f, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  f[get_idx(x, y, z, res)] = field_val(x, y, z);
}
__global__ void lap_full_kernel(const double* f, double* lap, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  double c = f[get_idx(x, y, z, res)];
  double s = f[get_idx(x + 1, y, z, res)] + f[get_idx(x - 1, y, z, res)] +
             f[get_idx(x, y + 1, z, res)] + f[get_idx(x, y - 1, z, res)] +
             f[get_idx(x, y, z + 1, res)] + f[get_idx(x, y, z - 1, res)];
  lap[get_idx(x, y, z, res)] = s - 6.0 * c;
}

// Distributed: Laplacian on the extended local block with direct neighbour strides (no wrap; ghosts
// were filled by exchange). ext = extended dims; only inner cells [ghost, ext-ghost) are written.
__global__ void lap_local_kernel(const double* f, double* lap, int3 ext, int ghost) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
  long idx = lx * sx + ly * sy + lz * sz;
  double c = f[idx];
  double s = f[idx + sx] + f[idx - sx] + f[idx + sy] + f[idx - sy] + f[idx + sz] + f[idx - sz];
  lap[idx] = s - 6.0 * c;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(40, 28, 24);

  // --- Serial reference on the full grid (cfd get_idx) ---
  size_t nfull = (size_t)res.x * res.y * res.z;
  double *d_full = nullptr, *d_full_lap = nullptr;
  cudaMalloc(&d_full, nfull * sizeof(double));
  cudaMalloc(&d_full_lap, nfull * sizeof(double));
  dim3 blk(8, 8, 8);
  dim3 grdF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  init_full_kernel<<<grdF, blk>>>(d_full, res);
  lap_full_kernel<<<grdF, blk>>>(d_full, d_full_lap, res);
  std::vector<double> ref(nfull);
  cudaMemcpy(ref.data(), d_full_lap, nfull * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_full);
  cudaFree(d_full_lap);

  // --- Distributed via MacGridHalo ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, MPI_COMM_WORLD);
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

  double *d_loc = nullptr, *d_loc_lap = nullptr;
  cudaMalloc(&d_loc, nloc * sizeof(double));
  cudaMalloc(&d_loc_lap, nloc * sizeof(double));
  cudaMemcpy(d_loc, h_loc.data(), nloc * sizeof(double), cudaMemcpyHostToDevice);

  mac.exchange(d_loc);  // fill ghosts from neighbours (replaces global wrapping)

  dim3 grdL((ext.x + 7) / 8, (ext.y + 7) / 8, (ext.z + 7) / 8);
  lap_local_kernel<<<grdL, blk>>>(d_loc, d_loc_lap, ext, mac.ghost);
  std::vector<double> loc_lap(nloc);
  cudaMemcpy(loc_lap.data(), d_loc_lap, nloc * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_loc);
  cudaFree(d_loc_lap);

  // Compare inner cells to the serial reference.
  int fail = 0;
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        int gx = lx + mac.origin_incl_ghost.x, gy = ly + mac.origin_incl_ghost.y,
            gz = lz + mac.origin_incl_ghost.z;
        double got = loc_lap[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y];
        double exp = ref[(size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y];
        if (fabs(got - exp) > 1e-9) ++fail;
      }

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): MacGridHalo reproduces cfd's periodic Laplacian on the MAC grid\n", size);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
