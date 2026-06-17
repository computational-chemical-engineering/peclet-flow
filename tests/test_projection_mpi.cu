// Step 5: distributed pressure projection (Chorin) on an all-fluid periodic staggered MAC grid.
//
// Composes everything: exchange the staggered velocities (u,v,w) -> cell-centred divergence ->
// Poisson solve for phi (Red-Black Gauss-Seidel, halo exchange between sweeps) -> subtract grad(phi)
// from the velocities. Serial (full grid, cfd get_idx) and distributed (extended blocks, MacGridHalo
// exchanges) run identical kernels, so the projected velocity must match cell-for-cell. This is the
// canonical incompressible-flow update and the last building block before threading into step().
//
// Staggering (cfd): u(i,j,k) sits on the cell's -x face (x=i), so cell i has faces u[i] and u[i+1].
//   div[i]    = (u[i+1]-u[i]) + (v[j+1]-v[j]) + (w[k+1]-w[k])     (dx=1)
//   u[i]     -= phi[i] - phi[i-1]   (grad on the -x face);  v,w analogous on their faces.
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cut_cell_ibm.cuh"  // cfd's get_idx
#include "mac_halo.cuh"

static constexpr int kIters = 40;

__host__ __device__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

__global__ void init_vel_full(double* u, double* v, double* w, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = hash01(x, y, z, 1);
  v[i] = hash01(x, y, z, 2);
  w[i] = hash01(x, y, z, 3);
}

__global__ void divergence_full(const double* u, const double* v, const double* w, double* div,
                                int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  div[i] = (u[get_idx(x + 1, y, z, res)] - u[i]) + (v[get_idx(x, y + 1, z, res)] - v[i]) +
           (w[get_idx(x, y, z + 1, res)] - w[i]);
}

__global__ void gs_poisson_full(double* phi, const double* div, int3 res, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (((x + y + z) & 1) != color) return;
  double s = phi[get_idx(x + 1, y, z, res)] + phi[get_idx(x - 1, y, z, res)] +
             phi[get_idx(x, y + 1, z, res)] + phi[get_idx(x, y - 1, z, res)] +
             phi[get_idx(x, y, z + 1, res)] + phi[get_idx(x, y, z - 1, res)];
  int i = get_idx(x, y, z, res);
  phi[i] = (s - div[i]) / 6.0;  // Lap(phi) = div, dx=1
}

__global__ void correct_full(double* u, double* v, double* w, const double* phi, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] -= phi[i] - phi[get_idx(x - 1, y, z, res)];
  v[i] -= phi[i] - phi[get_idx(x, y - 1, z, res)];
  w[i] -= phi[i] - phi[get_idx(x, y, z - 1, res)];
}

// --- distributed (extended block, direct strides) ---
struct Str {
  long sx, sy, sz;
};
__device__ inline long lidx(int lx, int ly, int lz, Str s) { return lx * s.sx + ly * s.sy + lz * s.sz; }

__global__ void divergence_local(const double* u, const double* v, const double* w, double* div,
                                 int3 ext, int ghost) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  Str s{1, ext.x, (long)ext.x * ext.y};
  long i = lidx(lx, ly, lz, s);
  div[i] = (u[i + s.sx] - u[i]) + (v[i + s.sy] - v[i]) + (w[i + s.sz] - w[i]);
}

__global__ void gs_poisson_local(double* phi, const double* div, int3 ext, int3 og, int ghost,
                                 int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  if ((((lx + og.x) + (ly + og.y) + (lz + og.z)) & 1) != color) return;
  Str s{1, ext.x, (long)ext.x * ext.y};
  long i = lidx(lx, ly, lz, s);
  double sum = phi[i + s.sx] + phi[i - s.sx] + phi[i + s.sy] + phi[i - s.sy] + phi[i + s.sz] +
               phi[i - s.sz];
  phi[i] = (sum - div[i]) / 6.0;
}

__global__ void correct_local(double* u, double* v, double* w, const double* phi, int3 ext,
                              int ghost) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  Str s{1, ext.x, (long)ext.x * ext.y};
  long i = lidx(lx, ly, lz, s);
  u[i] -= phi[i] - phi[i - s.sx];
  v[i] -= phi[i] - phi[i - s.sy];
  w[i] -= phi[i] - phi[i - s.sz];
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);
  int3 res = make_int3(40, 28, 24);
  dim3 blk(8, 8, 8);

  // --- Serial reference ---
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u, *v, *w, *phi, *dv;
  cudaMalloc(&u, nf * 8);
  cudaMalloc(&v, nf * 8);
  cudaMalloc(&w, nf * 8);
  cudaMalloc(&phi, nf * 8);
  cudaMalloc(&dv, nf * 8);
  cudaMemset(phi, 0, nf * 8);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  init_vel_full<<<gF, blk>>>(u, v, w, res);
  divergence_full<<<gF, blk>>>(u, v, w, dv, res);
  for (int k = 0; k < kIters; ++k) {
    gs_poisson_full<<<gF, blk>>>(phi, dv, res, 0);
    gs_poisson_full<<<gF, blk>>>(phi, dv, res, 1);
  }
  correct_full<<<gF, blk>>>(u, v, w, phi, res);
  std::vector<double> ru(nf), rv(nf), rw(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rv.data(), v, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rw.data(), w, nf * 8, cudaMemcpyDeviceToHost);
  cudaFree(u); cudaFree(v); cudaFree(w); cudaFree(phi); cudaFree(dv);

  // --- Distributed ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/1, MPI_COMM_WORLD);
  int3 ext = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  std::vector<double> hu(nl, 0), hv(nl, 0), hw(nl, 0);
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
        hu[i] = hash01(lx + og.x, ly + og.y, lz + og.z, 1);
        hv[i] = hash01(lx + og.x, ly + og.y, lz + og.z, 2);
        hw[i] = hash01(lx + og.x, ly + og.y, lz + og.z, 3);
      }
  double *eu, *ev, *ew, *ephi, *edv;
  cudaMalloc(&eu, nl * 8); cudaMalloc(&ev, nl * 8); cudaMalloc(&ew, nl * 8);
  cudaMalloc(&ephi, nl * 8); cudaMalloc(&edv, nl * 8);
  cudaMemcpy(eu, hu.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ev, hv.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ew, hw.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemset(ephi, 0, nl * 8);
  dim3 gL((ext.x + 7) / 8, (ext.y + 7) / 8, (ext.z + 7) / 8);

  mac.exchange(eu); mac.exchange(ev); mac.exchange(ew);  // need +1 face neighbours for divergence
  divergence_local<<<gL, blk>>>(eu, ev, ew, edv, ext, mac.ghost);
  for (int k = 0; k < kIters; ++k) {
    mac.exchange(ephi);
    gs_poisson_local<<<gL, blk>>>(ephi, edv, ext, og, mac.ghost, 0);
    mac.exchange(ephi);
    gs_poisson_local<<<gL, blk>>>(ephi, edv, ext, og, mac.ghost, 1);
  }
  mac.exchange(ephi);  // need -1 neighbours for the gradient correction
  correct_local<<<gL, blk>>>(eu, ev, ew, ephi, ext, mac.ghost);
  std::vector<double> lu(nl), lv(nl), lw(nl);
  cudaMemcpy(lu.data(), eu, nl * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(lv.data(), ev, nl * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(lw.data(), ew, nl * 8, cudaMemcpyDeviceToHost);
  cudaFree(eu); cudaFree(ev); cudaFree(ew); cudaFree(ephi); cudaFree(edv);

  int fail = 0;
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
        size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                    (size_t)(lz + og.z) * res.x * res.y;
        if (fabs(lu[li] - ru[gi]) > 1e-9 || fabs(lv[li] - rv[gi]) > 1e-9 ||
            fabs(lw[li] - rw[gi]) > 1e-9)
          ++fail;
      }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): distributed Chorin projection matches serial (u,v,w cell-for-cell)\n",
             size);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
