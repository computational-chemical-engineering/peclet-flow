// Step 4: distributed implicit diffusion via Red-Black Gauss-Seidel, validated against serial RB-GS.
//
// Solve (I - beta*Lap) phi = b iteratively, i.e. phi_i <- (b_i + beta * sum_6nbr phi_nbr) / (1+6beta),
// sweeping red cells then black cells. This is the iterative pattern the real solver uses for the
// momentum and pressure systems. The new ingredient vs Step 3 is a halo exchange after EACH colour
// sweep, so the next colour reads up-to-date neighbour values across block boundaries.
//
// Colouring uses GLOBAL parity (gx+gy+gz)&1 so red/black is consistent across block boundaries.
// Serial (full grid, get_idx) and distributed (extended block, exchange between sweeps) run identical
// arithmetic and iteration counts, so they must agree cell-for-cell — only if the halo is correct
// after every sweep. All-fluid (no IBM); ghost width 1 (7-point).
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // cfd's get_idx
#include "mac_halo.cuh"

static constexpr int kIters = 30;

__host__ __device__ inline double field_val(int x, int y, int z) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL;
}

__global__ void init_full(double* phi, double* b, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  double v = field_val(x, y, z);
  int i = get_idx(x, y, z, res);
  phi[i] = v;
  b[i] = v;
}

__global__ void gs_full(double* phi, const double* b, int3 res, double beta, double Ac, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (((x + y + z) & 1) != color) return;
  double s = phi[get_idx(x + 1, y, z, res)] + phi[get_idx(x - 1, y, z, res)] +
             phi[get_idx(x, y + 1, z, res)] + phi[get_idx(x, y - 1, z, res)] +
             phi[get_idx(x, y, z + 1, res)] + phi[get_idx(x, y, z - 1, res)];
  int i = get_idx(x, y, z, res);
  phi[i] = (b[i] + beta * s) / Ac;
}

__global__ void gs_local(double* phi, const double* b, int3 ext, int3 og, int ghost, double beta,
                         double Ac, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
  if (((gx + gy + gz) & 1) != color) return;  // global parity -> consistent across blocks
  long sx = 1, sy = ext.x, sz = (long)ext.x * ext.y;
  long idx = lx * sx + ly * sy + lz * sz;
  double s = phi[idx + sx] + phi[idx - sx] + phi[idx + sy] + phi[idx - sy] + phi[idx + sz] +
             phi[idx - sz];
  phi[idx] = (b[idx] + beta * s) / Ac;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(40, 28, 24);
  double beta = 0.1, Ac = 1.0 + 6.0 * beta;  // implicit diffusion, diagonally dominant
  dim3 blk(8, 8, 8);

  // --- Serial reference: full-grid RB-GS ---
  size_t nfull = (size_t)res.x * res.y * res.z;
  double *d_phi = nullptr, *d_b = nullptr;
  cudaMalloc(&d_phi, nfull * sizeof(double));
  cudaMalloc(&d_b, nfull * sizeof(double));
  dim3 grdF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  init_full<<<grdF, blk>>>(d_phi, d_b, res);
  for (int k = 0; k < kIters; ++k) {
    gs_full<<<grdF, blk>>>(d_phi, d_b, res, beta, Ac, 0);
    gs_full<<<grdF, blk>>>(d_phi, d_b, res, beta, Ac, 1);
  }
  std::vector<double> ref(nfull);
  cudaMemcpy(ref.data(), d_phi, nfull * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_phi);
  cudaFree(d_b);

  // --- Distributed: extended block, exchange after each colour sweep ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/1, MPI_COMM_WORLD);
  int3 ext = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nloc = mac.num_local_cells();
  std::vector<double> h(nloc, 0.0);
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx)
        h[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y] =
            field_val(lx + og.x, ly + og.y, lz + og.z);

  double *e_phi = nullptr, *e_b = nullptr;
  cudaMalloc(&e_phi, nloc * sizeof(double));
  cudaMalloc(&e_b, nloc * sizeof(double));
  cudaMemcpy(e_phi, h.data(), nloc * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(e_b, h.data(), nloc * sizeof(double), cudaMemcpyHostToDevice);
  dim3 grdL((ext.x + 7) / 8, (ext.y + 7) / 8, (ext.z + 7) / 8);
  for (int k = 0; k < kIters; ++k) {
    mac.exchange(e_phi);  // refresh ghosts (black current) before red
    gs_local<<<grdL, blk>>>(e_phi, e_b, ext, og, mac.ghost, beta, Ac, 0);
    mac.exchange(e_phi);  // refresh red ghosts before black
    gs_local<<<grdL, blk>>>(e_phi, e_b, ext, og, mac.ghost, beta, Ac, 1);
  }
  std::vector<double> loc(nloc);
  cudaMemcpy(loc.data(), e_phi, nloc * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(e_phi);
  cudaFree(e_b);

  int fail = 0;
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        double got = loc[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y];
        double exp = ref[(size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y];
        if (fabs(got - exp) > 1e-9) ++fail;
      }

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): distributed RB-GS implicit diffusion matches serial over %d iters\n", size,
             kIters);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
