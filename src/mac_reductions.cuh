// transport-core / cfd-gpu -- distributed global reductions over a MacGridHalo's owned (inner) cells.
//
// The serial solver reduces over the full grid (thrust::reduce for the pressure mean removal,
// atomicMax for the max-velocity / CFL). Distributed: each rank reduces over its INNER cells (skipping
// the ghost layer, which duplicates neighbours), then MPI_Allreduce. These are the null-space removal
// (remove_mean, for the pure-Neumann pressure) and the CFL/convergence max -- both needed by the
// distributed multigrid pressure solve and any real distributed step().
#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include "mac_halo.cuh"

namespace cfdmpi {
namespace rdetail {

__device__ inline double atomicMaxDouble(double* addr, double val) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    old = atomicCAS(a, assumed,
                    __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// reduce over INNER cells of the extended block (x-fastest layout): sum and max(|.|)
__global__ void reduce_inner_k(const double* __restrict__ f, int3 ext, int ghost, int3 inner,
                               double* dsum, double* dmax) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= inner.x || iy >= inner.y || iz >= inner.z) return;
  size_t idx = (size_t)(ix + ghost) + (size_t)(iy + ghost) * ext.x +
               (size_t)(iz + ghost) * ext.x * ext.y;
  double v = f[idx];
  atomicAdd(dsum, v);
  atomicMaxDouble(dmax, fabs(v));
}

__global__ void subtract_k(double* f, size_t n, double m) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) f[i] -= m;
}

}  // namespace rdetail

// Reduce a double cell-field (extended-block layout) over INNER cells across ALL ranks:
// out_sum = global sum, out_maxabs = global max|.|.
inline void mac_reduce(const double* d_field, const MacGridHalo& h, MPI_Comm comm, double* out_sum,
                       double* out_maxabs) {
  double *dsum = nullptr, *dmax = nullptr;
  cudaMalloc(&dsum, sizeof(double));
  cudaMalloc(&dmax, sizeof(double));
  double zero = 0.0, ninf = -1e300;
  cudaMemcpy(dsum, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dmax, &ninf, sizeof(double), cudaMemcpyHostToDevice);

  int3 inner = h.inner_res();
  if (inner.x > 0 && inner.y > 0 && inner.z > 0) {
    dim3 blk(8, 8, 8);
    dim3 grd((inner.x + 7) / 8, (inner.y + 7) / 8, (inner.z + 7) / 8);
    rdetail::reduce_inner_k<<<grd, blk>>>(d_field, h.local_ext, h.ghost, inner, dsum, dmax);
  }
  double lsum = 0.0, lmax = -1e300;
  cudaMemcpy(&lsum, dsum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&lmax, dmax, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dsum);
  cudaFree(dmax);

  double gsum = 0.0, gmax = 0.0;
  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
  *out_sum = gsum;
  *out_maxabs = gmax;
}

// Global max(|.|) over owned cells (CFL / convergence).
inline double mac_max_abs(const double* d_field, const MacGridHalo& h, MPI_Comm comm) {
  double s, m;
  mac_reduce(d_field, h, comm, &s, &m);
  return m;
}

// Subtract the global mean (over the full grid) -- the pure-Neumann pressure null-space removal.
// Subtracts over the WHOLE extended block: every rank subtracts the same mean, so the ghost layer
// stays equal to the neighbours' (also mean-subtracted) inner cells -- no halo exchange needed.
inline void mac_remove_mean(double* d_field, const MacGridHalo& h, MPI_Comm comm) {
  double s, m;
  mac_reduce(d_field, h, comm, &s, &m);
  long long gcount =
      (long long)h.global_res.x * (long long)h.global_res.y * (long long)h.global_res.z;
  double mean = s / static_cast<double>(gcount);
  size_t n = h.num_local_cells();
  rdetail::subtract_k<<<(unsigned)((n + 255) / 256), 256>>>(d_field, n, mean);
}

}  // namespace cfdmpi
