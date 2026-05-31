// Step 8: distributed Poiseuille channel flow — physical validation against the analytic balance.
//
// A body force g drives flow in x between two solid walls (masked, no-slip) at the y-boundaries;
// periodic in x and z. At steady state the momentum balance is nu * d2u/dy2 + g = 0, whose solution
// is the parabolic Poiseuille profile. Because a parabola has constant second difference, the DISCRETE
// steady solution satisfies nu*(u[j+1]-2u[j]+u[j-1]) + g = 0 exactly in the interior — a clean,
// mesh-exact physical check (this is what cfd's verify_poiseuille.py checks).
//
// Validations: (a) distributed (extended blocks + MacGridHalo) == serial full-grid cell-for-cell,
// np=1,2,4; (b) interior fluid momentum residual |nu*Lap_y(u) + g| ~ 0 (steady Poiseuille balance).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"
#include "mac_halo.cuh"

static constexpr int kSteps = 400;
static constexpr int kDiffIters = 80;

__device__ inline long L3(int x, int y, int z, int3 e) {
  return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
}

// b = u + dt*g  (body force folded into the implicit-diffusion RHS)
__global__ void rhs_full(const double* u, double* b, int3 res, double dtg) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  b[i] = u[i] + dtg;
}
__global__ void rhs_local(const double* u, double* b, int3 e, int g_, double dtg) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g_ || ly < g_ || lz < g_ || lx >= e.x - g_ || ly >= e.y - g_ || lz >= e.z - g_) return;
  long i = L3(lx, ly, lz, e);
  b[i] = u[i] + dtg;
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
__global__ void diff_local(double* c, const double* b, int3 e, int3 og, int g_, double beta,
                           double Ac, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx < g_ || ly < g_ || lz < g_ || lx >= e.x - g_ || ly >= e.y - g_ || lz >= e.z - g_) return;
  if ((((lx + og.x) + (ly + og.y) + (lz + og.z)) & 1) != color) return;
  long i = L3(lx, ly, lz, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = c[i + sx] + c[i - sx] + c[i + sy] + c[i - sy] + c[i + sz] + c[i - sz];
  c[i] = (b[i] + beta * s) / Ac;
}
// walls: solid (u=0) for gy < wall or gy >= Ny-wall
__global__ void mask_full(double* u, int3 res, int wall) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  if (y < wall || y >= res.y - wall) u[get_idx(x, y, z, res)] = 0.0;
}
// Mask ALL local cells (inner + ghost) by their wrapped global y, so walls stay zero everywhere
// without needing an extra exchange. Enforced after every Gauss-Seidel sweep (hard Dirichlet).
__global__ void mask_local(double* u, int3 e, int3 og, int wall, int Ny) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= e.x || ly >= e.y || lz >= e.z) return;
  int gy = ((ly + og.y) % Ny + Ny) % Ny;
  if (gy < wall || gy >= Ny - wall) u[L3(lx, ly, lz, e)] = 0.0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = 32;
  int3 res = make_int3(N, N, 4);
  int wall = 4;
  // Large dt: backward Euler is unconditionally stable, so we march to steady state quickly.
  double nu = 0.1, dt = 20.0, g = 1e-3, beta = nu * dt, Ac = 1.0 + 6.0 * beta, dtg = dt * g;
  dim3 blk(8, 8, 4);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 3) / 4);
  auto alloc = [](size_t n) { double* p; cudaMalloc(&p, n * 8); return p; };

  // --- serial reference ---
  size_t nf = (size_t)res.x * res.y * res.z;
  double *u = alloc(nf), *b = alloc(nf);
  cudaMemset(u, 0, nf * 8);
  for (int s = 0; s < kSteps; ++s) {
    rhs_full<<<gF, blk>>>(u, b, res, dtg);
    for (int it = 0; it < kDiffIters; ++it) {
      diff_full<<<gF, blk>>>(u, b, res, beta, Ac, 0);
      mask_full<<<gF, blk>>>(u, res, wall);  // hard Dirichlet no-slip within the solve
      diff_full<<<gF, blk>>>(u, b, res, beta, Ac, 1);
      mask_full<<<gF, blk>>>(u, res, wall);
    }
  }
  std::vector<double> ru(nf);
  cudaMemcpy(ru.data(), u, nf * 8, cudaMemcpyDeviceToHost);
  cudaFree(u); cudaFree(b);

  // --- distributed ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/1, MPI_COMM_WORLD);
  int3 e = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  double *eu = alloc(nl), *eb = alloc(nl);
  cudaMemset(eu, 0, nl * 8);
  dim3 gL((e.x + 7) / 8, (e.y + 7) / 8, (e.z + 3) / 4);
  for (int s = 0; s < kSteps; ++s) {
    rhs_local<<<gL, blk>>>(eu, eb, e, mac.ghost, dtg);
    for (int it = 0; it < kDiffIters; ++it) {
      mac.exchange(eu);
      diff_local<<<gL, blk>>>(eu, eb, e, og, mac.ghost, beta, Ac, 0);
      mask_local<<<gL, blk>>>(eu, e, og, wall, res.y);
      mac.exchange(eu);
      diff_local<<<gL, blk>>>(eu, eb, e, og, mac.ghost, beta, Ac, 1);
      mask_local<<<gL, blk>>>(eu, e, og, wall, res.y);
    }
  }
  std::vector<double> lu(nl);
  cudaMemcpy(lu.data(), eu, nl * 8, cudaMemcpyDeviceToHost);
  cudaFree(eu); cudaFree(eb);

  // (a) consistency
  int fail = 0;
  for (int lz = mac.ghost; lz < e.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < e.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < e.x - mac.ghost; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        size_t gi = (size_t)(lx + og.x) + (size_t)(ly + og.y) * res.x +
                    (size_t)(lz + og.z) * res.x * res.y;
        if (fabs(lu[li] - ru[gi]) > 1e-9) ++fail;
      }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // (b) physical: interior momentum residual nu*Lap_y(u) + g ~ 0, computed on the serial field.
  double max_res = 0.0, u_peak = 0.0;
  if (rank == 0) {
    auto IDX = [&](int x, int y, int z) {
      return (size_t)x + (size_t)y * res.x + (size_t)z * res.x * res.y;
    };
    for (int y = wall + 1; y < res.y - wall - 1; ++y) {
      double uy = ru[IDX(0, y, 0)];
      u_peak = fmax(u_peak, uy);
      double lap_y = ru[IDX(0, y + 1, 0)] - 2.0 * uy + ru[IDX(0, y - 1, 0)];
      max_res = fmax(max_res, fabs(nu * lap_y + g));
    }
  }

  // Discrete parabola has zeros at the wall-cell centres (y = wall-1 and y = Ny-wall), so the
  // effective channel width is Ny-2*wall+1 cells and the peak is g*W^2/(8*nu).
  double W = res.y - 2.0 * wall + 1.0;
  double u_peak_analytic = g * W * W / (8.0 * nu);
  double peak_err = fabs(u_peak - u_peak_analytic) / u_peak_analytic;
  int physfail = (max_res < 1e-6 && peak_err < 0.02) ? 0 : 1;
  if (rank == 0) {
    printf("# Poiseuille: u_peak=%.5f analytic=%.5f peak_err=%.2e  max|nu*Lap_y u + g|=%.2e\n",
           u_peak, u_peak_analytic, peak_err, max_res);
    if (total == 0 && physfail == 0)
      printf("OK (np=%d): distributed channel flow matches serial AND analytic Poiseuille profile\n",
             size);
    else
      fprintf(stderr, "FAILED (np=%d): consistency=%d  max_residual=%.2e  peak_err=%.2e\n", size,
              total, max_res, peak_err);
  }
  MPI_Bcast(&physfail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return (total == 0 && physfail == 0) ? 0 : 1;
}
