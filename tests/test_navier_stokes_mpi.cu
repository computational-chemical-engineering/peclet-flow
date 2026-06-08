// Step 11: full distributed Navier-Stokes (DistributedNS with advection) vs a serial reference.
//
// The DistributedNS solver, with set_advection(true), does each step: explicit Koren advection
// folded into the momentum RHS + implicit diffusion (RB-GS) + Chorin projection. We validate it
// against an independent SERIAL full-grid integration of the identical scheme (same kernels, same
// iteration counts), requiring the distributed result to match cell-for-cell over multiple steps,
// np=1,2,4. This is the rigorous distribution check for the full nonlinear solver (cell-for-cell is
// exact regardless of iterative-solve convergence). Periodic, no solid.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"
#include "staggered_advection.cuh"

using dns::DistributedNS;

static constexpr int kSteps = 10;
static constexpr int kDiff = 30;
static constexpr int kPois = 40;

__host__ __device__ inline double hv(int x, int y, int z, int s) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(s * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}
__host__ __device__ inline double init_u(int gx, int gy, int gz, double k) {
  return cos(k * gx) * sin(k * (gy + 0.5)) + 0.2 * hv(gx, gy, gz, 1);
}
__host__ __device__ inline double init_v(int gx, int gy, int gz, double k) {
  return -sin(k * (gx + 0.5)) * cos(k * gy) + 0.2 * hv(gx, gy, gz, 2);
}
__host__ __device__ inline double init_w(int gx, int gy, int gz, double k) {
  return 0.1 * hv(gx, gy, gz, 3);
}

// ----- serial full-grid kernels (get_idx wrapping) -----
__global__ void init_full(double* u, double* v, double* w, int3 res, double k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = init_u(x, y, z, k);
  v[i] = init_v(x, y, z, k);
  w[i] = init_w(x, y, z, k);
}
__global__ void advect_rhs_full(int comp, const double* u, const double* v, const double* w,
                                const double* phi, double* b, double dt, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  double A = sadv::advect(comp, x, y, z, sadv::FullAcc{u, res}, sadv::FullAcc{v, res},
                          sadv::FullAcc{w, res}, sadv::FullAcc{phi, res});
  b[i] = phi[i] - dt * A;
}
__global__ void diff_full(double* c, const double* b, int3 res, double beta, double Ac, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (((x + y + z) & 1) != color) return;
  double s = c[get_idx(x + 1, y, z, res)] + c[get_idx(x - 1, y, z, res)] +
             c[get_idx(x, y + 1, z, res)] + c[get_idx(x, y - 1, z, res)] +
             c[get_idx(x, y, z + 1, res)] + c[get_idx(x, y, z - 1, res)];
  int i = get_idx(x, y, z, res);
  c[i] = (b[i] + beta * s) / Ac;
}
__global__ void diverg_full(const double* u, const double* v, const double* w, double* d, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  d[i] = (u[get_idx(x + 1, y, z, res)] - u[i]) + (v[get_idx(x, y + 1, z, res)] - v[i]) +
         (w[get_idx(x, y, z + 1, res)] - w[i]);
}
__global__ void pois_full(double* phi, const double* d, int3 res, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (((x + y + z) & 1) != color) return;
  double s = phi[get_idx(x + 1, y, z, res)] + phi[get_idx(x - 1, y, z, res)] +
             phi[get_idx(x, y + 1, z, res)] + phi[get_idx(x, y - 1, z, res)] +
             phi[get_idx(x, y, z + 1, res)] + phi[get_idx(x, y, z - 1, res)];
  int i = get_idx(x, y, z, res);
  phi[i] = (s - d[i]) / 6.0;
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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = 24;
  int3 res = make_int3(N, N, 8);
  double nu = 0.02, dt = 0.2, k = 2.0 * M_PI / N, beta = nu * dt, Ac = 1.0 + 6.0 * beta;
  dim3 blk(8, 8, 4), gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);
  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };

  // ----- serial reference -----
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u = alloc(nf), *v = alloc(nf), *w = alloc(nf), *phi = alloc(nf), *dvg = alloc(nf);
  double* b[3] = {alloc(nf), alloc(nf), alloc(nf)};
  init_full<<<gF, blk>>>(u, v, w, res, k);
  double* comp[3] = {u, v, w};
  for (int s = 0; s < kSteps; ++s) {
    for (int c = 0; c < 3; ++c) advect_rhs_full<<<gF, blk>>>(c, u, v, w, comp[c], b[c], dt, res);
    for (int c = 0; c < 3; ++c)
      for (int it = 0; it < kDiff; ++it) {
        diff_full<<<gF, blk>>>(comp[c], b[c], res, beta, Ac, 0);
        diff_full<<<gF, blk>>>(comp[c], b[c], res, beta, Ac, 1);
      }
    diverg_full<<<gF, blk>>>(u, v, w, dvg, res);
    cudaMemset(phi, 0, nf * 8);
    for (int it = 0; it < kPois; ++it) {
      pois_full<<<gF, blk>>>(phi, dvg, res, 0);
      pois_full<<<gF, blk>>>(phi, dvg, res, 1);
    }
    correct_full<<<gF, blk>>>(u, v, w, phi, res);
  }
  std::vector<double> ru(nf), rv(nf), rw(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rv.data(), v, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rw.data(), w, nf * 8, cudaMemcpyDeviceToHost);
  for (double* p : {u, v, w, phi, dvg, b[0], b[1], b[2]}) cudaFree(p);

  // ----- distributed via DistributedNS (advection on) -----
  DistributedNS sol;
  sol.init(res, rank, size, nu, dt);
  sol.set_advection(true);
  int3 e = sol.ext(), og = sol.origin_incl_ghost();
  int g = sol.ghost();
  std::size_t n = sol.num_cells();
  std::vector<double> hu(n, 0), hvv(n, 0), hw(n, 0);
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        hu[i] = init_u(gx, gy, gz, k);
        hvv[i] = init_v(gx, gy, gz, k);
        hw[i] = init_w(gx, gy, gz, k);
      }
  sol.upload_velocity(hu.data(), hvv.data(), hw.data());
  for (int s = 0; s < kSteps; ++s) sol.step(kDiff, kPois);
  std::vector<double> du(n), dv(n), dw(n);
  sol.download_velocity(du.data(), dv.data(), dw.data());

  int fail = 0;
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                    (size_t)(lz + og.z) * res.x * res.y;
        if (fabs(du[i] - ru[gi]) > 1e-9 || fabs(dv[i] - rv[gi]) > 1e-9 ||
            fabs(dw[i] - rw[gi]) > 1e-9)
          ++fail;
      }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): full distributed Navier-Stokes matches serial cell-for-cell (%d steps)\n",
             size, kSteps);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
