// Step 3: distributed explicit scalar advection-diffusion over a time loop, validated against a
// serial reference cell-for-cell. The advection uses the Koren TVD reconstruction (the same limiter
// cfd uses), so the operator reaches +/-2 and needs ghost width 2. Serial (full grid, cfd get_idx)
// and distributed (extended local block + MacGridHalo exchange each step) call the SAME cell_update
// device function — they can only agree if the width-2 halo is correct on every block boundary at
// every step. This exercises the exact explicit-operator pattern the distributed solver will use:
//   each step:  exchange ghosts  ->  update inner cells.
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // cfd's get_idx
#include "mac_halo.cuh"

static constexpr int kSteps = 25;

__host__ __device__ inline double field_val(int x, int y, int z) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL;
}

// Koren TVD face flux given upwind-ordered values (far-upwind, upwind, downwind) and face velocity.
__host__ __device__ inline double koren_face(double far_up, double up, double down, double vel) {
  double denom = down - up;
  double r = (fabs(denom) > 1e-12) ? (up - far_up) / denom : 0.0;
  double psi = fmax(0.0, fmin(fmin(2.0 * r, (2.0 + r) / 3.0), 2.0));
  return vel * (up + 0.5 * psi * denom);
}

// n[axis][0..4] = phi at offsets -2,-1,0,+1,+2 along axis (n[*][2] is the shared centre).
__host__ __device__ inline double cell_update(const double n[3][5], const double a[3], double nu,
                                              double dt, const double dx[3]) {
  double total = 0.0;
  for (int ax = 0; ax < 3; ++ax) {
    double m2 = n[ax][0], m1 = n[ax][1], c = n[ax][2], p1 = n[ax][3], p2 = n[ax][4];
    double vel = a[ax];
    double fp = (vel >= 0.0) ? koren_face(m1, c, p1, vel) : koren_face(p2, p1, c, vel);
    double fm = (vel >= 0.0) ? koren_face(m2, m1, c, vel) : koren_face(p1, c, m1, vel);
    double adv = (fp - fm) / dx[ax];
    double diff = nu * (p1 - 2.0 * c + m1) / (dx[ax] * dx[ax]);
    total += -adv + diff;
  }
  return n[0][2] + dt * total;
}

__constant__ double c_a[3];
__constant__ double c_dx[3];

__global__ void init_full(double* f, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  f[get_idx(x, y, z, res)] = field_val(x, y, z);
}

__global__ void step_full(const double* f, double* fn, int3 res, double nu, double dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  double n[3][5];
  for (int o = -2; o <= 2; ++o) {
    n[0][o + 2] = f[get_idx(x + o, y, z, res)];
    n[1][o + 2] = f[get_idx(x, y + o, z, res)];
    n[2][o + 2] = f[get_idx(x, y, z + o, res)];
  }
  fn[get_idx(x, y, z, res)] = cell_update(n, c_a, nu, dt, c_dx);
}

__global__ void step_local(const double* f, double* fn, int3 ext, int ghost, double nu, double dt) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < ghost || ly < ghost || lz < ghost) return;
  if (lx >= ext.x - ghost || ly >= ext.y - ghost || lz >= ext.z - ghost) return;
  long s[3] = {1, ext.x, (long)ext.x * ext.y};
  long idx0 = lx * s[0] + ly * s[1] + lz * s[2];
  double n[3][5];
  for (int o = -2; o <= 2; ++o) {
    n[0][o + 2] = f[idx0 + o * s[0]];
    n[1][o + 2] = f[idx0 + o * s[1]];
    n[2][o + 2] = f[idx0 + o * s[2]];
  }
  fn[idx0] = cell_update(n, c_a, nu, dt, c_dx);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(40, 28, 24);
  double a[3] = {0.5, -0.3, 0.2}, dx[3] = {1.0, 1.0, 1.0};
  double nu = 0.1, dt = 0.2;  // CFL ~0.5, diffusion number ~0.02 — stable
  cudaMemcpyToSymbol(c_a, a, sizeof(a));
  cudaMemcpyToSymbol(c_dx, dx, sizeof(dx));
  dim3 blk(8, 8, 8);

  // --- Serial reference: full-grid time loop with cfd get_idx ---
  size_t nfull = (size_t)res.x * res.y * res.z;
  double *d_a = nullptr, *d_b = nullptr;
  cudaMalloc(&d_a, nfull * sizeof(double));
  cudaMalloc(&d_b, nfull * sizeof(double));
  dim3 grdF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  init_full<<<grdF, blk>>>(d_a, res);
  for (int s = 0; s < kSteps; ++s) {
    step_full<<<grdF, blk>>>(d_a, d_b, res, nu, dt);
    std::swap(d_a, d_b);
  }
  std::vector<double> ref(nfull);
  cudaMemcpy(ref.data(), d_a, nfull * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);

  // --- Distributed: extended block, exchange ghosts each step ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/2, MPI_COMM_WORLD);
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
  double *e_a = nullptr, *e_b = nullptr;
  cudaMalloc(&e_a, nloc * sizeof(double));
  cudaMalloc(&e_b, nloc * sizeof(double));
  cudaMemcpy(e_a, h_loc.data(), nloc * sizeof(double), cudaMemcpyHostToDevice);
  dim3 grdL((ext.x + 7) / 8, (ext.y + 7) / 8, (ext.z + 7) / 8);
  for (int s = 0; s < kSteps; ++s) {
    mac.exchange(e_a);  // fill width-2 ghosts, then update inner cells
    step_local<<<grdL, blk>>>(e_a, e_b, ext, mac.ghost, nu, dt);
    std::swap(e_a, e_b);
  }
  std::vector<double> loc(nloc);
  cudaMemcpy(loc.data(), e_a, nloc * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(e_a);
  cudaFree(e_b);

  int fail = 0;
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        int gx = lx + mac.origin_incl_ghost.x, gy = ly + mac.origin_incl_ghost.y,
            gz = lz + mac.origin_incl_ghost.z;
        double got = loc[(size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y];
        double exp = ref[(size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y];
        if (fabs(got - exp) > 1e-9) ++fail;
      }

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): distributed Koren advection-diffusion matches serial over %d steps\n",
             size, kSteps);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
