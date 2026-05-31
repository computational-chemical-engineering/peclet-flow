// Step 10: distributed staggered momentum advection operator (cfd's Koren TVD scheme), validated.
//
// Replicates cfd_solver.cu's get_advection_velocity (2-point staggered interpolation of the advecting
// velocity) and get_tvd_flux / tvd_flux_koren (Koren limiter, sign-upwinded). For each velocity
// component it computes the conservative advection A = sum_dir (F_plus - F_minus) (dx=1), which
// reaches +/-2 cells -> ghost width 2.
//
// Validations: (a) distributed (extended blocks + MacGridHalo, width 2) == serial full-grid
// cell-for-cell, np=1,2,4; (b) conservation — sum over all cells of A is ~0 (the flux form telescopes
// over the periodic domain), so advection conserves total momentum. This is the nonlinear operator
// needed to extend the distributed Stokes solver to full Navier-Stokes.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"
#include "mac_halo.cuh"

__host__ __device__ inline double hashv(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

// Koren TVD flux (double), matching cfd's tvd_flux_koren / get_tvd_flux.
__device__ inline double koren(double up_m1, double up, double down, double vel) {
  double num = up - up_m1, den = down - up;
  double r = (fabs(den) < 1e-10) ? 0.0 : num / den;
  if (fabs(den) < 1e-10 && fabs(num) < 1e-10) r = 1.0;
  double psi = fmax(0.0, fmin(2.0 * r, fmin((1.0 + 2.0 * r) / 3.0, 2.0)));
  return vel * (up + 0.5 * psi * (down - up));
}
__device__ inline double tvd(double LL, double L, double R, double RR, double vel) {
  return (vel > 0.0) ? koren(LL, L, R, vel) : koren(RR, R, L, vel);
}

// Field accessors: full grid wraps (get_idx); local extended block indexes directly.
struct FullAcc {
  const double* d;
  int3 res;
  __device__ double operator()(int x, int y, int z) const { return d[get_idx(x, y, z, res)]; }
};
struct LocAcc {
  const double* d;
  int3 e;
  __device__ double operator()(int x, int y, int z) const {
    return d[(long)x + (long)y * e.x + (long)z * (long)e.x * e.y];
  }
};

// Advecting velocity at the +face_dir face of the comp control volume at (x,y,z). Mirrors
// get_advection_velocity in cfd_solver.cu exactly.
template <class A>
__device__ inline double adv_vel(int comp, int fd, int x, int y, int z, A U, A V, A W) {
  if (comp == 0) {
    if (fd == 0) return 0.5 * (U(x, y, z) + U(x + 1, y, z));
    if (fd == 1) return 0.5 * (V(x - 1, y + 1, z) + V(x, y + 1, z));
    return 0.5 * (W(x - 1, y, z + 1) + W(x, y, z + 1));
  }
  if (comp == 1) {
    if (fd == 0) return 0.5 * (U(x + 1, y - 1, z) + U(x + 1, y, z));
    if (fd == 1) return 0.5 * (V(x, y, z) + V(x, y + 1, z));
    return 0.5 * (W(x, y - 1, z + 1) + W(x, y, z + 1));
  }
  if (fd == 0) return 0.5 * (U(x + 1, y, z - 1) + U(x + 1, y, z));
  if (fd == 1) return 0.5 * (V(x, y + 1, z - 1) + V(x, y + 1, z));
  return 0.5 * (W(x, y, z) + W(x, y, z + 1));
}

// Conservative advection A = sum_dir (F_plus - F_minus); PHI is the advected component field.
template <class A>
__device__ inline double advect(int comp, int x, int y, int z, A U, A V, A W, A PHI) {
  double out = 0.0;
  for (int fd = 0; fd < 3; ++fd) {
    int ox = (fd == 0), oy = (fd == 1), oz = (fd == 2);
    double velp = adv_vel(comp, fd, x, y, z, U, V, W);
    double velm = adv_vel(comp, fd, x - ox, y - oy, z - oz, U, V, W);
    double Fp = tvd(PHI(x - ox, y - oy, z - oz), PHI(x, y, z), PHI(x + ox, y + oy, z + oz),
                    PHI(x + 2 * ox, y + 2 * oy, z + 2 * oz), velp);
    double Fm = tvd(PHI(x - 2 * ox, y - 2 * oy, z - 2 * oz), PHI(x - ox, y - oy, z - oz),
                    PHI(x, y, z), PHI(x + ox, y + oy, z + oz), velm);
    out += Fp - Fm;
  }
  return out;
}

__global__ void init_full(double* u, double* v, double* w, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  u[i] = hashv(x, y, z, 1);
  v[i] = hashv(x, y, z, 2);
  w[i] = hashv(x, y, z, 3);
}
__global__ void adv_full(int comp, const double* u, const double* v, const double* w,
                         const double* phi, double* A, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  A[get_idx(x, y, z, res)] =
      advect(comp, x, y, z, FullAcc{u, res}, FullAcc{v, res}, FullAcc{w, res}, FullAcc{phi, res});
}
__global__ void adv_local(int comp, const double* u, const double* v, const double* w,
                          const double* phi, double* A, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  A[(long)x + (long)y * e.x + (long)z * (long)e.x * e.y] =
      advect(comp, x, y, z, LocAcc{u, e}, LocAcc{v, e}, LocAcc{w, e}, LocAcc{phi, e});
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);
  int3 res = make_int3(40, 28, 24);
  dim3 blk(8, 8, 4), gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);
  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };

  // serial fields + reference advection for each component
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u = alloc(nf), *v = alloc(nf), *w = alloc(nf), *A = alloc(nf);
  init_full<<<gF, blk>>>(u, v, w, res);
  std::vector<std::vector<double>> ref(3, std::vector<double>(nf));
  double* comp_f[3] = {u, v, w};
  for (int c = 0; c < 3; ++c) {
    adv_full<<<gF, blk>>>(c, u, v, w, comp_f[c], A, res);
    cudaMemcpy(ref[c].data(), A, nf * 8, cudaMemcpyDeviceToHost);
  }
  std::vector<double> hu(nf), hv(nf), hw(nf);
  cudaMemcpy(hu.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hv.data(), v, nf * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hw.data(), w, nf * 8, cudaMemcpyDeviceToHost);
  cudaFree(u); cudaFree(v); cudaFree(w); cudaFree(A);

  // distributed, ghost width 2
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/2, MPI_COMM_WORLD);
  int3 e = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  std::vector<double> lu(nl, 0), lv(nl, 0), lw(nl, 0);
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                    (size_t)(lz + og.z) * res.x * res.y;
        lu[i] = hu[gi]; lv[i] = hv[gi]; lw[i] = hw[gi];
      }
  double *eu = alloc(nl), *ev = alloc(nl), *ew = alloc(nl), *eA = alloc(nl);
  cudaMemcpy(eu, lu.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ev, lv.data(), nl * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ew, lw.data(), nl * 8, cudaMemcpyHostToDevice);
  mac.exchange(eu); mac.exchange(ev); mac.exchange(ew);
  dim3 gL((e.x + 7) / 8, (e.y + 7) / 8, (e.z + 3) / 4);
  double* comp_e[3] = {eu, ev, ew};

  int fail = 0;
  double cons_l[3] = {0, 0, 0};
  for (int c = 0; c < 3; ++c) {
    adv_local<<<gL, blk>>>(c, eu, ev, ew, comp_e[c], eA, e, mac.ghost);
    std::vector<double> la(nl);
    cudaMemcpy(la.data(), eA, nl * 8, cudaMemcpyDeviceToHost);
    for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
      for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
        for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
          size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
          size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                      (size_t)(lz + og.z) * res.x * res.y;
          if (fabs(la[i] - ref[c][gi]) > 1e-9) ++fail;
          cons_l[c] += la[i];
        }
  }
  cudaFree(eu); cudaFree(ev); cudaFree(ew); cudaFree(eA);

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  double cons[3];
  MPI_Allreduce(cons_l, cons, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double cmax = fmax(fabs(cons[0]), fmax(fabs(cons[1]), fabs(cons[2])));
  if (rank == 0) {
    printf("# advection: sum|A| (momentum non-conservation) = %.2e\n", cmax);
    bool ok = (total == 0) && (cmax < 1e-8);
    if (ok)
      printf("OK (np=%d): distributed Koren momentum advection matches serial AND conserves momentum\n",
             size);
    else
      fprintf(stderr, "FAILED (np=%d): consistency=%d conservation=%.2e\n", size, total, cmax);
  }
  int cfail = (cmax < 1e-8) ? 0 : 1;
  MPI_Bcast(&cfail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (total == 0 && cfail == 0) ? 0 : 1;
}
