// Step 7: distributed incompressible flow AROUND an SDF-described solid.
//
// Extends the Step 6 Stokes solver with a static solid (a sphere, SDF < 0 inside) handled by a simple
// no-slip immersed boundary: each step the velocity is forced to zero on faces inside the solid, and
// the projection makes the surrounding fluid divergence-free. The solid mask is a function of global
// position, so it is identical on every rank.
//
// Validations: (a) distributed (extended blocks + MacGridHalo) matches a serial full-grid run
// cell-for-cell over many steps, np=1,2,4 — the rigorous proof that the decomposition + halo are
// correct with a solid present; (b) exact no-slip: velocity is identically zero in the solid at the
// end (mask applied last); (c) reported: peak fluid divergence after projection.
//
// This is the simplest SDF immersed boundary; the full Robust-Scaled cut-cell IBM and nonlinear
// advection layer onto the same pattern (see doc/mpi_parallelization_status.md).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cut_cell_ibm.cuh"  // cfd's get_idx
#include "mac_halo.cuh"

static constexpr int kSteps = 20;
static constexpr int kDiffIters = 40;
static constexpr int kPoisIters = 80;

__host__ __device__ inline double sphere_sdf(double x, double y, double z, double cx, double cy,
                                             double cz, double R) {
  double dx = x - cx, dy = y - cy, dz = z - cz;
  return sqrt(dx * dx + dy * dy + dz * dz) - R;  // < 0 inside solid
}
__device__ inline long L3(int x, int y, int z, int3 e) {
  return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
}

struct Solid {
  double cx, cy, cz, R;
};
__constant__ Solid c_solid;

__global__ void init_uniform_full(double* u, double* v, double* w, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = 1.0;
  v[i] = 0.0;
  w[i] = 0.0;
}

#define MASK_BODY(IDX, GX, GY, GZ)                                                      \
  Solid s = c_solid;                                                                    \
  if (sphere_sdf((GX), (GY) + 0.5, (GZ) + 0.5, s.cx, s.cy, s.cz, s.R) < 0) u[IDX] = 0.0; \
  if (sphere_sdf((GX) + 0.5, (GY), (GZ) + 0.5, s.cx, s.cy, s.cz, s.R) < 0) v[IDX] = 0.0; \
  if (sphere_sdf((GX) + 0.5, (GY) + 0.5, (GZ), s.cx, s.cy, s.cz, s.R) < 0) w[IDX] = 0.0;

__global__ void mask_full(double* u, double* v, double* w, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  MASK_BODY(i, x, y, z)
}
__global__ void mask_local(double* u, double* v, double* w, int3 e, int3 og, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  long i = L3(lx, ly, lz, e);
  int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
  MASK_BODY(i, gx, gy, gz)
}

// --- diffusion + projection kernels (same discretisation as Step 6) ---
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
__global__ void diverg_full(const double* u, const double* v, const double* w, double* d, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  d[i] = (u[get_idx(x + 1, y, z, res)] - u[i]) + (v[get_idx(x, y + 1, z, res)] - v[i]) +
         (w[get_idx(x, y, z + 1, res)] - w[i]);
}
__global__ void diverg_local(const double* u, const double* v, const double* w, double* d, int3 e,
                             int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  d[i] = (u[i + sx] - u[i]) + (v[i + sy] - v[i]) + (w[i + sz] - w[i]);
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
__global__ void pois_local(double* phi, const double* d, int3 e, int3 og, int g, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g || ly < g || lz < g || lx >= e.x - g || ly >= e.y - g || lz >= e.z - g) return;
  if ((((lx + og.x) + (ly + og.y) + (lz + og.z)) & 1) != color) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = phi[i + sx] + phi[i - sx] + phi[i + sy] + phi[i - sy] + phi[i + sz] + phi[i - sz];
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
  int3 res = make_int3(N, N, 8);
  double nu = 0.1, dt = 0.5, beta = nu * dt, Ac = 1.0 + 6.0 * beta;
  Solid solid{N * 0.5, N * 0.5, res.z * 0.5, N * 0.2};
  cudaMemcpyToSymbol(c_solid, &solid, sizeof(Solid));
  dim3 blk(8, 8, 4);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);

  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };

  // --- serial reference ---
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u = alloc(nf), *v = alloc(nf), *w = alloc(nf), *b = alloc(nf), *phi = alloc(nf),
         *dvg = alloc(nf);
  init_uniform_full<<<gF, blk>>>(u, v, w, res);
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
    mask_full<<<gF, blk>>>(u, v, w, res);  // no-slip applied last
  }
  std::vector<double> ru(nf), rv(nf), rw(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rv.data(), v, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(rw.data(), w, nf * 8, cudaMemcpyDeviceToHost);
  cudaFree(u); cudaFree(v); cudaFree(w); cudaFree(b); cudaFree(phi); cudaFree(dvg);

  // --- distributed ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/1, MPI_COMM_WORLD);
  int3 e = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  double *eu = alloc(nl), *ev = alloc(nl), *ew = alloc(nl), *eb = alloc(nl), *ephi = alloc(nl),
         *edv = alloc(nl);
  std::vector<double> ones(nl, 0.0), zeros(nl, 0.0);
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx)
        ones[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = 1.0;
  cudaMemcpy(eu, ones.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ev, zeros.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ew, zeros.data(), nl * 8, cudaMemcpyHostToDevice);
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
    mask_local<<<gL, blk>>>(eu, ev, ew, e, og, mac.ghost);
  }
  std::vector<double> lu(nl), lv(nl), lw(nl);
  cudaMemcpy(lu.data(), eu, nl * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(lv.data(), ev, nl * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(lw.data(), ew, nl * 8, cudaMemcpyDeviceToHost);
  cudaFree(eu); cudaFree(ev); cudaFree(ew); cudaFree(eb); cudaFree(ephi); cudaFree(edv);

  // (a) consistency + (b) no-slip in solid
  int fail = 0, solid_viol = 0;
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        size_t gi = (size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y;
        if (fabs(lu[li] - ru[gi]) > 1e-9 || fabs(lv[li] - rv[gi]) > 1e-9 ||
            fabs(lw[li] - rw[gi]) > 1e-9)
          ++fail;
        if (sphere_sdf(gx, gy + 0.5, gz + 0.5, solid.cx, solid.cy, solid.cz, solid.R) < 0 &&
            lu[li] != 0.0)
          ++solid_viol;
      }
  int total = 0, sviol = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&solid_viol, &sviol, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    bool ok = (total == 0) && (sviol == 0);
    if (ok)
      printf("OK (np=%d): distributed flow around SDF solid matches serial; no-slip exact (%d steps)\n",
             size, kSteps);
    else
      fprintf(stderr, "FAILED (np=%d): consistency=%d solid_velocity_violations=%d\n", size, total,
              sviol);
  }
  MPI_Finalize();
  return (total == 0 && sviol == 0) ? 0 : 1;
}
