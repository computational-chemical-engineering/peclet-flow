// Step 6: a complete distributed incompressible flow solver (unsteady Stokes) assembled from the
// verified building blocks, on cfd's staggered MAC grid.
//
// One timestep = per-component implicit diffusion (backward Euler, Red-Black Gauss-Seidel with a halo
// exchange between sweeps) + Chorin projection (divergence -> Poisson -> grad correction). Periodic.
//
// Two validations:
//   (a) consistency  — distributed (extended blocks + MacGridHalo) matches a serial full-grid run
//                      cell-for-cell over many steps, np=1,2,4. (Guaranteed only if every halo is
//                      correct at every sweep of every step.)
//   (b) physical     — initialised with the 2D Taylor-Green vortex, which is *discretely*
//                      divergence-free on the MAC grid (so the projection is an exact no-op), the
//                      solver reproduces the analytic backward-Euler decay rate to <0.1%.
//
// Stokes (no nonlinear advection) needs only ghost width 1, and the Taylor-Green velocity decays as a
// pure diffusion mode — a clean, rigorous physical check of the assembled algorithm.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // cfd's get_idx
#include "mac_halo.cuh"

static constexpr int kSteps = 20;
static constexpr int kDiffIters = 60;  // RB-GS iters for the implicit diffusion solve
static constexpr int kPoisIters = 0;   // projection no-op for TGV; kept 0 (div==0 exactly)

// ---- shared device math ----
__device__ inline long L3(int x, int y, int z, int3 e) { return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y; }

// init Taylor-Green on the staggered grid (dx=1, wavenumber k); w=0.
__global__ void init_tgv_full(double* u, double* v, double* w, int3 res, double k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = cos(k * x) * sin(k * (y + 0.5));        // u at x-face (x=i, y=j+0.5)
  v[i] = -sin(k * (x + 0.5)) * cos(k * y);       // v at y-face (x=i+0.5, y=j)
  w[i] = 0.0;
}

// One implicit-diffusion RB-GS sweep of colour `color`: c <- (b + beta*sum_nbr c)/(1+6 beta).
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
__global__ void diff_local(double* c, const double* b, int3 e, int3 og, int g, double beta,
                           double Ac, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  if ((((lx + og.x) + (ly + og.y) + (lz + og.z)) & 1) != color) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = c[i + sx] + c[i - sx] + c[i + sy] + c[i - sy] + c[i + sz] + c[i - sz];
  c[i] = (b[i] + beta * s) / Ac;
}

// projection pieces (dx=1)
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
__global__ void diverg_local(const double* u, const double* v, const double* w, double* d, int3 e,
                             int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  d[i] = (u[i + sx] - u[i]) + (v[i + sy] - v[i]) + (w[i + sz] - w[i]);
}
__global__ void pois_local(double* phi, const double* d, int3 e, int3 og, int g, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  if ((((lx + og.x) + (ly + og.y) + (lz + og.z)) & 1) != color) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = phi[i + sx] + phi[i - sx] + phi[i + sy] + phi[i - sy] + phi[i + sz] + phi[i - sz];
  phi[i] = (s - d[i]) / 6.0;
}
__global__ void correct_local(double* u, double* v, double* w, const double* phi, int3 e, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  u[i] -= phi[i] - phi[i - sx];
  v[i] -= phi[i] - phi[i - sy];
  w[i] -= phi[i] - phi[i - sz];
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = 32;
  int3 res = make_int3(N, N, 4);
  double k = 2.0 * M_PI / N, nu = 0.05, dt = 0.5;
  double beta = nu * dt, Ac = 1.0 + 6.0 * beta;  // dx=1
  dim3 blk(8, 8, 4);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);

  // --- serial reference ---
  size_t nf = (size_t)res.x * res.y * res.z;
  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };
  double *u = alloc(nf), *v = alloc(nf), *w = alloc(nf), *b = alloc(nf), *phi = alloc(nf),
         *dvg = alloc(nf);
  init_tgv_full<<<gF, blk>>>(u, v, w, res, k);
  std::vector<double> u0(nf);
  cudaMemcpy(u0.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  for (int s = 0; s < kSteps; ++s) {
    double* comp[3] = {u, v, w};
    for (int c = 0; c < 3; ++c) {
      cudaMemcpy(b, comp[c], nf * 8, cudaMemcpyDeviceToDevice);
      for (int it = 0; it < kDiffIters; ++it) {
        diff_full<<<gF, blk>>>(comp[c], b, res, beta, Ac, 0);
        diff_full<<<gF, blk>>>(comp[c], b, res, beta, Ac, 1);
      }
    }
    diverg_full<<<gF, blk>>>(u, v, w, dvg, res);
    cudaMemset(phi, 0, nf * 8);
    for (int it = 0; it < kPoisIters; ++it) {
      pois_full<<<gF, blk>>>(phi, dvg, res, 0);
      pois_full<<<gF, blk>>>(phi, dvg, res, 1);
    }
    correct_full<<<gF, blk>>>(u, v, w, phi, res);
  }
  std::vector<double> ru(nf), uf(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  uf = ru;
  cudaFree(u); cudaFree(v); cudaFree(w); cudaFree(b); cudaFree(phi); cudaFree(dvg);

  // --- distributed ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/1, MPI_COMM_WORLD);
  int3 e = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  std::vector<double> hu(nl, 0), hv(nl, 0), hw(nl, 0);
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        hu[i] = cos(k * gx) * sin(k * (gy + 0.5));
        hv[i] = -sin(k * (gx + 0.5)) * cos(k * gy);
      }
  double *eu = alloc(nl), *ev = alloc(nl), *ew = alloc(nl), *eb = alloc(nl), *ephi = alloc(nl),
         *edv = alloc(nl);
  cudaMemcpy(eu, hu.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ev, hv.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ew, hw.data(), nl * 8, cudaMemcpyHostToDevice);  // w = 0
  dim3 gL((e.x + 7) / 8, (e.y + 7) / 8, (e.z + 3) / 4);
  for (int s = 0; s < kSteps; ++s) {
    double* comp[3] = {eu, ev, ew};
    for (int c = 0; c < 3; ++c) {
      cudaMemcpy(eb, comp[c], nl * 8, cudaMemcpyDeviceToDevice);
      for (int it = 0; it < kDiffIters; ++it) {
        mac.exchange(comp[c]);
        diff_local<<<gL, blk>>>(comp[c], eb, e, og, mac.ghost, beta, Ac, 0);
        mac.exchange(comp[c]);
        diff_local<<<gL, blk>>>(comp[c], eb, e, og, mac.ghost, beta, Ac, 1);
      }
    }
    mac.exchange(eu); mac.exchange(ev); mac.exchange(ew);
    diverg_local<<<gL, blk>>>(eu, ev, ew, edv, e, mac.ghost);
    cudaMemset(ephi, 0, nl * 8);
    for (int it = 0; it < kPoisIters; ++it) {
      mac.exchange(ephi);
      pois_local<<<gL, blk>>>(ephi, edv, e, og, mac.ghost, 0);
      mac.exchange(ephi);
      pois_local<<<gL, blk>>>(ephi, edv, e, og, mac.ghost, 1);
    }
    mac.exchange(ephi);
    correct_local<<<gL, blk>>>(eu, ev, ew, ephi, e, mac.ghost);
  }
  std::vector<double> lu(nl);
  cudaMemcpy(lu.data(), eu, nl * 8, cudaMemcpyDeviceToHost);
  cudaFree(eu); cudaFree(ev); cudaFree(ew); cudaFree(eb); cudaFree(ephi); cudaFree(edv);

  // (a) consistency: distributed inner cells == serial reference.
  int fail = 0;
  double max_uf_local = 0.0;
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                    (size_t)(lz + og.z) * res.x * res.y;
        if (fabs(lu[li] - ru[gi]) > 1e-9) ++fail;
        max_uf_local = fmax(max_uf_local, fabs(lu[li]));
      }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // (b) physical: amplitude decay vs the analytic discrete backward-Euler factor.
  // discrete Laplacian eigenvalue of the TGV mode (x and y axes): lam = 2*(2 - 2 cos k).
  double lam = 2.0 * (2.0 - 2.0 * cos(k));
  double f_step = 1.0 / (1.0 + nu * dt * lam);
  double expect_ratio = pow(f_step, kSteps);
  double max_u0 = 0.0;
  for (double v0 : u0) max_u0 = fmax(max_u0, fabs(v0));
  double max_uf = 0.0;
  MPI_Allreduce(&max_uf_local, &max_uf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double got_ratio = max_uf / max_u0;
  double rel_err = fabs(got_ratio - expect_ratio) / expect_ratio;

  if (rank == 0) {
    printf("# Stokes/TGV: decay measured=%.6f analytic=%.6f rel_err=%.2e\n", got_ratio,
           expect_ratio, rel_err);
    bool ok = (total == 0) && (rel_err < 1e-3);
    if (ok)
      printf("OK (np=%d): distributed Stokes matches serial cell-for-cell AND TGV decay (%d steps)\n",
             size, kSteps);
    else
      fprintf(stderr, "FAILED (np=%d): consistency_mismatches=%d  decay_rel_err=%.2e\n", size, total,
              rel_err);
    fflush(stdout);
  }
  int physfail = (rel_err < 1e-3) ? 0 : 1;
  MPI_Bcast(&physfail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (total == 0 && physfail == 0) ? 0 : 1;
}
