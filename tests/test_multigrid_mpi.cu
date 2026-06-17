// Distributed geometric multigrid (periodic constant-coefficient Poisson) vs a serial full-grid
// reference V-cycle. Both run the identical sequence -- RB-GS smooth (red/black by global parity),
// residual, 8:1 restriction, trilinear prolongation, mean removal -- so the distributed result must
// match the serial one cell-for-cell (halo vs in-kernel wrap give identical neighbour values; the only
// divergence is ~1e-16 in the global-mean reduction order). np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cut_cell_ibm.cuh"  // get_idx (periodic wrap)
#include "mac_halo.cuh"
#include "mac_multigrid.cuh"

__host__ __device__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

// ---- serial full-grid reference kernels (periodic wrap) ----
__global__ void s_fill_rhs(double* b, int3 res, int seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  b[get_idx(x, y, z, res)] = hash01(x, y, z, seed);
}

__global__ void s_smooth(double* x, const double* b, int3 res, double h2, int color) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
      iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= res.x || iy >= res.y || iz >= res.z) return;
  if (((ix + iy + iz) & 1) != color) return;
  int i = get_idx(ix, iy, iz, res);
  double sum = x[get_idx(ix + 1, iy, iz, res)] + x[get_idx(ix - 1, iy, iz, res)] +
               x[get_idx(ix, iy + 1, iz, res)] + x[get_idx(ix, iy - 1, iz, res)] +
               x[get_idx(ix, iy, iz + 1, res)] + x[get_idx(ix, iy, iz - 1, res)];
  x[i] = (sum + h2 * b[i]) / 6.0;
}

__global__ void s_residual(double* r, const double* x, const double* b, int3 res, double invh2) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
      iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= res.x || iy >= res.y || iz >= res.z) return;
  int i = get_idx(ix, iy, iz, res);
  double sum = x[get_idx(ix + 1, iy, iz, res)] + x[get_idx(ix - 1, iy, iz, res)] +
               x[get_idx(ix, iy + 1, iz, res)] + x[get_idx(ix, iy - 1, iz, res)] +
               x[get_idx(ix, iy, iz + 1, res)] + x[get_idx(ix, iy, iz - 1, res)];
  r[i] = b[i] - invh2 * (6.0 * x[i] - sum);
}

__global__ void s_restrict(double* c, const double* f, int3 cres, int3 fres) {
  int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y * blockDim.y + threadIdx.y,
      cz = blockIdx.z * blockDim.z + threadIdx.z;
  if (cx >= cres.x || cy >= cres.y || cz >= cres.z) return;
  double sum = 0.0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx)
        sum += f[get_idx(2 * cx + dx, 2 * cy + dy, 2 * cz + dz, fres)];
  c[get_idx(cx, cy, cz, cres)] = 0.125 * sum;
}

__device__ inline double s_trilerp(const double* c, double x, double y, double z, int3 res) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto F = [&](int xx, int yy, int zz) { return c[get_idx(xx, yy, zz, res)]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy;
  double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

__global__ void s_prolong(double* f, const double* c, int3 fres, int3 cres) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
      iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= fres.x || iy >= fres.y || iz >= fres.z) return;
  double x = 0.5 * ix - 0.25, y = 0.5 * iy - 0.25, z = 0.5 * iz - 0.25;
  f[get_idx(ix, iy, iz, fres)] += s_trilerp(c, x, y, z, cres);
}

struct SLevel {
  double *x, *rhs, *res;
  int3 res3;
  double h;
  int n;
};

static void s_remove_mean(double* d, int n) {  // host sum -> subtract (reference)
  std::vector<double> h(n);
  cudaMemcpy(h.data(), d, n * 8, cudaMemcpyDeviceToHost);
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += h[i];
  double m = s / n;
  for (int i = 0; i < n; ++i) h[i] -= m;
  cudaMemcpy(d, h.data(), n * 8, cudaMemcpyHostToDevice);
}

static int s_pre, s_post, s_bottom;
static std::vector<SLevel> slev;

static void s_smooth_lvl(SLevel& lv, int sweeps) {
  dim3 blk(8, 8, 8);
  dim3 grd((lv.res3.x + 7) / 8, (lv.res3.y + 7) / 8, (lv.res3.z + 7) / 8);
  double h2 = lv.h * lv.h;
  for (int k = 0; k < sweeps; ++k) {
    s_smooth<<<grd, blk>>>(lv.x, lv.rhs, lv.res3, h2, 0);
    s_smooth<<<grd, blk>>>(lv.x, lv.rhs, lv.res3, h2, 1);
  }
}

static void s_vcycle(int L) {
  SLevel& lv = slev[L];
  dim3 blk(8, 8, 8);
  dim3 grd((lv.res3.x + 7) / 8, (lv.res3.y + 7) / 8, (lv.res3.z + 7) / 8);
  if (L + 1 == (int)slev.size()) {
    s_smooth_lvl(lv, s_bottom);
    s_remove_mean(lv.x, lv.n);
    return;
  }
  s_smooth_lvl(lv, s_pre);
  s_residual<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.res3, 1.0 / (lv.h * lv.h));
  SLevel& cs = slev[L + 1];
  dim3 cgrd((cs.res3.x + 7) / 8, (cs.res3.y + 7) / 8, (cs.res3.z + 7) / 8);
  s_restrict<<<cgrd, blk>>>(cs.rhs, lv.res, cs.res3, lv.res3);
  cudaMemset(cs.x, 0, cs.n * 8);
  s_vcycle(L + 1);
  s_prolong<<<grd, blk>>>(lv.x, cs.x, lv.res3, cs.res3);
  s_smooth_lvl(lv, s_post);
  s_remove_mean(lv.x, lv.n);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  const int n_levels = 4, n_vcycles = 3, pre = 2, post = 2, bottom = 12, seed = 7;
  const double h0 = 1.0;
  dim3 blk(8, 8, 8);

  // ---- serial reference (rank 0 computes; all ranks redo it cheaply for a local compare) ----
  slev.resize(n_levels);
  for (int L = 0; L < n_levels; ++L) {
    int3 r = make_int3(res.x >> L, res.y >> L, res.z >> L);
    slev[L].res3 = r;
    slev[L].n = r.x * r.y * r.z;
    slev[L].h = h0 * (1 << L);
    cudaMalloc(&slev[L].x, slev[L].n * 8);
    cudaMalloc(&slev[L].rhs, slev[L].n * 8);
    cudaMalloc(&slev[L].res, slev[L].n * 8);
    cudaMemset(slev[L].x, 0, slev[L].n * 8);
    cudaMemset(slev[L].rhs, 0, slev[L].n * 8);
    cudaMemset(slev[L].res, 0, slev[L].n * 8);
  }
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  s_fill_rhs<<<gF, blk>>>(slev[0].rhs, res, seed);
  s_remove_mean(slev[0].rhs, slev[0].n);  // solvability
  s_pre = pre;
  s_post = post;
  s_bottom = bottom;
  for (int v = 0; v < n_vcycles; ++v) s_vcycle(0);
  std::vector<double> sx(slev[0].n);
  cudaMemcpy(sx.data(), slev[0].x, slev[0].n * 8, cudaMemcpyDeviceToHost);
  for (auto& lv : slev) {
    cudaFree(lv.x);
    cudaFree(lv.rhs);
    cudaFree(lv.res);
  }

  // ---- distributed ----
  cfdmpi::DistributedPoissonMG mg;
  mg.init(res, rank, size, h0, n_levels, MPI_COMM_WORLD);
  cfdmpi::MGLevel& l0 = mg.level(0);
  // fill level-0 rhs inner cells from the same hash, then remove mean (matches the serial rhs)
  std::vector<double> hb(l0.n, 0.0);
  for (int lz = l0.g; lz < l0.ext.z - l0.g; ++lz)
    for (int ly = l0.g; ly < l0.ext.y - l0.g; ++ly)
      for (int lx = l0.g; lx < l0.ext.x - l0.g; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * l0.ext.x + (size_t)lz * l0.ext.x * l0.ext.y;
        hb[i] = hash01(lx + l0.og.x, ly + l0.og.y, lz + l0.og.z, seed);
      }
  cudaMemcpy(l0.rhs, hb.data(), l0.n * 8, cudaMemcpyHostToDevice);
  cfdmpi::mac_remove_mean(l0.rhs, l0.mac, MPI_COMM_WORLD);

  // initial residual (x=0) is just rhs: record its global max for a convergence check
  double r0_sum = 0.0, r0_max = 0.0;
  cfdmpi::mac_reduce(l0.rhs, l0.mac, MPI_COMM_WORLD, &r0_sum, &r0_max);

  mg.solve(n_vcycles, pre, post, bottom);

  // final residual on level 0: r = rhs - A x
  {
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    l0.mac.exchange(l0.x);
    cfdmpi::mgdetail::mg_residual_k<<<grd, blk>>>(l0.res, l0.x, l0.rhs, l0.ext, l0.g,
                                                  1.0 / (l0.h * l0.h));
  }
  double rf_sum = 0.0, rf_max = 0.0;
  cfdmpi::mac_reduce(l0.res, l0.mac, MPI_COMM_WORLD, &rf_sum, &rf_max);

  std::vector<double> lx(l0.n);
  cudaMemcpy(lx.data(), l0.x, l0.n * 8, cudaMemcpyDeviceToHost);

  // compare distributed inner cells to the serial full-grid solution
  double maxd = 0.0;
  for (int iz = l0.g; iz < l0.ext.z - l0.g; ++iz)
    for (int iy = l0.g; iy < l0.ext.y - l0.g; ++iy)
      for (int ix = l0.g; ix < l0.ext.x - l0.g; ++ix) {
        size_t li = (size_t)ix + (size_t)iy * l0.ext.x + (size_t)iz * l0.ext.x * l0.ext.y;
        int gx = ix + l0.og.x, gy = iy + l0.og.y, gz = iz + l0.og.z;
        size_t gi = (size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y;
        maxd = fmax(maxd, fabs(lx[li] - sx[gi]));
      }
  mg.free();

  double gmaxd = 0.0;
  MPI_Reduce(&maxd, &gmaxd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  int fail = 0;
  if (rank == 0) {
    double conv = rf_max / (r0_max + 1e-300);  // residual reduction over the solve
    bool match_ok = (gmaxd <= 1e-8 && !std::isnan(gmaxd));
    bool conv_ok = (conv < 0.1 && !std::isnan(conv));  // a real solver drops the residual
    fail = (match_ok && conv_ok) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  levels=%d vcycles=%d (pre=%d post=%d bottom=%d)\n", size, res.x,
           res.y, res.z, n_levels, n_vcycles, pre, post, bottom);
    printf("  distributed vs serial V-cycle: max|d| = %.3e   %s\n", gmaxd,
           match_ok ? "match-OK" : "MATCH-FAIL");
    printf("  residual max: initial %.3e -> final %.3e  (reduction %.3e)   %s\n", r0_max, rf_max,
           conv, conv_ok ? "converge-OK" : "CONVERGE-FAIL");
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
