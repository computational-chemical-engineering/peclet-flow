// Step 12 (capstone): full distributed Navier-Stokes flow AROUND an SDF solid.
//
// Exercises the DistributedNS solver with BOTH nonlinear advection and an SDF solid (sphere,
// no-slip by per-cell velocity masking). Validated against an independent serial full-grid
// integration of the identical scheme, cell-for-cell over multiple steps, np=1,2,4 — the complete
// distributed incompressible-flow capability (decomposition + async halo + advection + projection +
// solids) in one test. Periodic box with a body force driving flow past the sphere.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"
#include "staggered_advection.cuh"

using dns::DistributedNS;

static constexpr int kSteps = 8;
static constexpr int kDiff = 25;
static constexpr int kPois = 30;

struct Sph { double cx, cy, cz, R; };
__constant__ Sph c_sph;

__host__ __device__ inline bool solid_at(int gx, int gy, int gz, Sph s) {
  double dx = (gx + 0.5) - s.cx, dy = (gy + 0.5) - s.cy, dz = (gz + 0.5) - s.cz;
  return dx * dx + dy * dy + dz * dz < s.R * s.R;  // cell-centre inside sphere
}

// ----- serial full-grid kernels -----
__global__ void init_full(double* u, double* v, double* w, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = 1.0; v[i] = 0.0; w[i] = 0.0;  // uniform inflow in x
}
__global__ void advect_rhs_full(int comp, const double* u, const double* v, const double* w,
                                const double* phi, double* b, double idt, double f, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  double A = sadv::advect(comp, x, y, z, sadv::FullAcc{u, res}, sadv::FullAcc{v, res},
                          sadv::FullAcc{w, res}, sadv::FullAcc{phi, res});
  b[i] = idt * phi[i] - A + f;  // divided convention: momentum equation scaled by 1/dt
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
__global__ void mask_one_full(double* c, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (solid_at(x, y, z, c_sph)) c[get_idx(x, y, z, res)] = 0.0;
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
  double nu = 0.05, dt = 0.2, fx = 0.02, idt = 1.0 / dt, beta = nu, Ac = idt + 6.0 * beta;
  Sph sph{N * 0.5, N * 0.5, res.z * 0.5, N * 0.18};
  cudaMemcpyToSymbol(c_sph, &sph, sizeof(Sph));
  dim3 blk(8, 8, 4), gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);
  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };

  // ----- serial reference -----
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u = alloc(nf), *v = alloc(nf), *w = alloc(nf), *phi = alloc(nf), *dvg = alloc(nf);
  double* b[3] = {alloc(nf), alloc(nf), alloc(nf)};
  init_full<<<gF, blk>>>(u, v, w, res);
  mask_one_full<<<gF, blk>>>(u, res);
  mask_one_full<<<gF, blk>>>(v, res);
  mask_one_full<<<gF, blk>>>(w, res);
  double* comp[3] = {u, v, w};
  double f[3] = {fx, 0, 0};
  for (int s = 0; s < kSteps; ++s) {
    for (int c = 0; c < 3; ++c) advect_rhs_full<<<gF, blk>>>(c, u, v, w, comp[c], b[c], idt, f[c], res);
    for (int c = 0; c < 3; ++c)
      for (int it = 0; it < kDiff; ++it) {
        diff_full<<<gF, blk>>>(comp[c], b[c], res, beta, Ac, 0);
        mask_one_full<<<gF, blk>>>(comp[c], res);
        diff_full<<<gF, blk>>>(comp[c], b[c], res, beta, Ac, 1);
        mask_one_full<<<gF, blk>>>(comp[c], res);
      }
    diverg_full<<<gF, blk>>>(u, v, w, dvg, res);
    cudaMemset(phi, 0, nf * 8);
    for (int it = 0; it < kPois; ++it) {
      pois_full<<<gF, blk>>>(phi, dvg, res, 0);
      pois_full<<<gF, blk>>>(phi, dvg, res, 1);
    }
    correct_full<<<gF, blk>>>(u, v, w, phi, res);
    mask_one_full<<<gF, blk>>>(u, res);
    mask_one_full<<<gF, blk>>>(v, res);
    mask_one_full<<<gF, blk>>>(w, res);
  }
  std::vector<double> ru(nf), rv(nf), rw(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rv.data(), v, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rw.data(), w, nf * 8, cudaMemcpyDeviceToHost);
  for (double* p : {u, v, w, phi, dvg, b[0], b[1], b[2]}) cudaFree(p);

  // ----- distributed via DistributedNS (advection + solid) -----
  DistributedNS sol;
  sol.init(res, rank, size, 1.0, nu, dt);
  sol.set_incremental_pressure(false);  // serial reference is classical (non-incremental) Chorin
  sol.set_advection(true);
  sol.set_body_force(fx, 0, 0);
  int3 e = sol.ext(), og = sol.origin_incl_ghost();
  int g = sol.ghost();
  std::size_t n = sol.num_cells();
  std::vector<double> hu(n, 0), hvv(n, 0), hw(n, 0), solid(n, 0.0);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx) {
        int gx = ((lx + og.x) % N + N) % N, gy = ((ly + og.y) % N + N) % N,
            gz = ((lz + og.z) % res.z + res.z) % res.z;
        if (solid_at(gx, gy, gz, sph))
          solid[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = 1.0;
      }
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx)
        hu[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = 1.0;  // uniform inflow
  sol.upload_velocity(hu.data(), hvv.data(), hw.data());
  sol.set_solid(solid);  // also masks the initial field
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
      printf("OK (np=%d): distributed Navier-Stokes flow around SDF solid matches serial (%d steps)\n",
             size, kSteps);
    else
      fprintf(stderr, "FAILED (np=%d): %d mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
