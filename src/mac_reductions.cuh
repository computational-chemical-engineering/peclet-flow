// transport-core / cfd-gpu -- distributed global reductions over a MacGridHalo's owned (inner) cells.
//
// The serial solver reduces over the full grid (thrust::reduce for the pressure mean removal,
// atomicMax for the max-velocity / CFL). Distributed: each rank reduces over its INNER cells (skipping
// the ghost layer, which duplicates neighbours), then MPI_Allreduce. These are the null-space removal
// (remove_mean, for the pure-Neumann pressure) and the CFL/convergence max -- both needed by the
// distributed multigrid pressure solve and any real distributed step().
//
// Performance (these run many times per PCG / V-cycle iteration): each call does a SHARED-MEMORY block
// reduction (one atomic per block, not per cell -- the per-cell single-address atomic was catastrophic
// under contention), reuses a process-lifetime pinned scratch (no per-call cudaMalloc/cudaFree), and the
// hot single-quantity paths (mac_max_abs / mac_dot / mac_remove_mean) do ONE MPI_Allreduce instead of
// two. mac_reduce keeps the dual sum+max form (two collectives) for the diagnostics that need both.
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

// Process-lifetime reduction scratch: device accumulators d[2] = {sum, max} and a pinned host readback
// h[2]. Lazily allocated once; intentionally never freed (avoids a static-destruction-order cudaFree
// after the CUDA context is gone). Single host thread per rank, so no synchronisation needed.
struct RedScratch {
  double* d = nullptr;
  double* h = nullptr;
};
inline RedScratch& red_scratch() {
  static RedScratch s = [] {
    RedScratch r;
    cudaMalloc(&r.d, 2 * sizeof(double));
    cudaHostAlloc(&r.h, 2 * sizeof(double), cudaHostAllocDefault);
    return r;
  }();
  return s;
}

constexpr int RBLK = 256;          // reduction block size
inline int red_grid(long ncells) {  // one atomic per block; cap the grid, blocks grid-stride
  long g = (ncells + RBLK - 1) / RBLK;
  return (int)(g < 1 ? 1 : (g > 1024 ? 1024 : g));
}

__global__ void set2_k(double* d, double a, double b) {
  d[0] = a;
  d[1] = b;
}

// Block reduction over INNER cells (x-fastest layout, linearised + grid-strided): block-local sum (and
// max|.| if DO_MAX) via shared memory, then ONE atomic per block into d[0] (and d[1]).
template <bool DO_MAX>
__global__ void reduce_block_k(const double* __restrict__ f, int3 ext, int ghost, int3 inner,
                               long ncells, double* dsum, double* dmax) {
  __shared__ double ssum[RBLK];
  __shared__ double smax[RBLK];
  int tid = threadIdx.x;
  double ls = 0.0, lm = -1e300;
  for (long c = (long)blockIdx.x * blockDim.x + tid; c < ncells; c += (long)gridDim.x * blockDim.x) {
    int ix = (int)(c % inner.x);
    int iy = (int)((c / inner.x) % inner.y);
    int iz = (int)(c / ((long)inner.x * inner.y));
    size_t idx = (size_t)(ix + ghost) + (size_t)(iy + ghost) * ext.x +
                 (size_t)(iz + ghost) * (size_t)ext.x * ext.y;
    double v = f[idx];
    ls += v;
    if (DO_MAX) lm = fmax(lm, fabs(v));
  }
  ssum[tid] = ls;
  if (DO_MAX) smax[tid] = lm;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      ssum[tid] += ssum[tid + s];
      if (DO_MAX) smax[tid] = fmax(smax[tid], smax[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(dsum, ssum[0]);
    if (DO_MAX) atomicMaxDouble(dmax, smax[0]);
  }
}

// Block reduction of the inner-product sum a[i]*b[i].
__global__ void dot_block_k(const double* __restrict__ a, const double* __restrict__ b, int3 ext,
                            int ghost, int3 inner, long ncells, double* dsum) {
  __shared__ double ssum[RBLK];
  int tid = threadIdx.x;
  double ls = 0.0;
  for (long c = (long)blockIdx.x * blockDim.x + tid; c < ncells; c += (long)gridDim.x * blockDim.x) {
    int ix = (int)(c % inner.x);
    int iy = (int)((c / inner.x) % inner.y);
    int iz = (int)(c / ((long)inner.x * inner.y));
    size_t idx = (size_t)(ix + ghost) + (size_t)(iy + ghost) * ext.x +
                 (size_t)(iz + ghost) * (size_t)ext.x * ext.y;
    ls += a[idx] * b[idx];
  }
  ssum[tid] = ls;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(dsum, ssum[0]);
}

__global__ void subtract_k(double* f, size_t n, double m) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) f[i] -= m;
}

// local block reduction into the scratch accumulators (sum in d[0], and max|.| in d[1] if DO_MAX)
template <bool DO_MAX>
inline void local_reduce(const double* d_field, const MacGridHalo& h, RedScratch& sc) {
  set2_k<<<1, 1>>>(sc.d, 0.0, -1e300);
  int3 inner = h.inner_res();
  long n = (long)inner.x * inner.y * inner.z;
  if (n > 0)
    reduce_block_k<DO_MAX><<<red_grid(n), RBLK>>>(d_field, h.local_ext, h.ghost, inner, n, sc.d,
                                                  sc.d + 1);
}

}  // namespace rdetail

// Reduce a double cell-field (extended-block layout) over INNER cells across ALL ranks:
// out_sum = global sum, out_maxabs = global max|.|.  (Dual form: two collectives; for diagnostics that
// genuinely need both. The hot single-quantity paths below do one.)
inline void mac_reduce(const double* d_field, const MacGridHalo& h, MPI_Comm comm, double* out_sum,
                       double* out_maxabs) {
  auto& sc = rdetail::red_scratch();
  rdetail::local_reduce<true>(d_field, h, sc);
  cudaMemcpy(sc.h, sc.d, 2 * sizeof(double), cudaMemcpyDeviceToHost);
  double lsum = sc.h[0], lmax = sc.h[1], gsum = 0.0, gmax = 0.0;
  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
  *out_sum = gsum;
  *out_maxabs = gmax;
}

// Global max(|.|) over owned cells (CFL / convergence) -- ONE collective.
inline double mac_max_abs(const double* d_field, const MacGridHalo& h, MPI_Comm comm) {
  auto& sc = rdetail::red_scratch();
  rdetail::local_reduce<true>(d_field, h, sc);
  cudaMemcpy(sc.h + 1, sc.d + 1, sizeof(double), cudaMemcpyDeviceToHost);
  double lmax = sc.h[1], gmax = 0.0;
  MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
  return gmax;
}

// Global inner-product <a, b> over owned cells (for Krylov methods) -- ONE collective.
inline double mac_dot(const double* a, const double* b, const MacGridHalo& h, MPI_Comm comm) {
  auto& sc = rdetail::red_scratch();
  rdetail::set2_k<<<1, 1>>>(sc.d, 0.0, -1e300);
  int3 inner = h.inner_res();
  long n = (long)inner.x * inner.y * inner.z;
  if (n > 0)
    rdetail::dot_block_k<<<rdetail::red_grid(n), rdetail::RBLK>>>(a, b, h.local_ext, h.ghost, inner, n,
                                                                 sc.d);
  cudaMemcpy(sc.h, sc.d, sizeof(double), cudaMemcpyDeviceToHost);
  double lsum = sc.h[0], gsum = 0.0;
  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
  return gsum;
}

// Subtract the global mean (over the full grid) -- the pure-Neumann pressure null-space removal -- with
// ONE collective. Subtracts over the WHOLE extended block: every rank subtracts the same mean, so the
// ghost layer stays equal to the neighbours' (also mean-subtracted) inner cells -- no halo exchange.
inline void mac_remove_mean(double* d_field, const MacGridHalo& h, MPI_Comm comm) {
  auto& sc = rdetail::red_scratch();
  rdetail::local_reduce<false>(d_field, h, sc);
  cudaMemcpy(sc.h, sc.d, sizeof(double), cudaMemcpyDeviceToHost);
  double lsum = sc.h[0], gsum = 0.0;
  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
  long long gcount =
      (long long)h.global_res.x * (long long)h.global_res.y * (long long)h.global_res.z;
  double mean = gsum / static_cast<double>(gcount);
  size_t n = h.num_local_cells();
  rdetail::subtract_k<<<(unsigned)((n + 255) / 256), 256>>>(d_field, n, mean);
}

}  // namespace cfdmpi
