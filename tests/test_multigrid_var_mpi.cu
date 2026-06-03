// Distributed multigrid with a VARIABLE-COEFFICIENT fine-level operator (the SDF / cut-cell case) vs a
// serial full-grid reference V-cycle. The fine level uses per-cell 7-point coefficients assembled from
// staggered face openness o(face) -- a deterministic function of the face's global position (here an
// analytic periodic sphere SDF, openness in [0.1,1], a 10x coefficient ratio). Coarse levels stay
// constant-coefficient (mirrors the serial use_periodic_operator = level>0). Because a face coefficient
// is a function of its global position, every rank derives matching values for shared faces, so the
// operator is symmetric across block boundaries and the distributed V-cycle must reproduce the serial
// one cell-for-cell. np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"  // get_idx (periodic wrap)
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

// periodic (min-image) sphere SDF: negative inside the solid sphere, positive in fluid
__host__ __device__ inline double psdf(double x, double y, double z, int3 res) {
  double cx = res.x * 0.5, cy = res.y * 0.5, cz = res.z * 0.5, R = res.x * 0.3;
  double dx = x - cx, dy = y - cy, dz = z - cz;
  dx -= res.x * round(dx / res.x);
  dy -= res.y * round(dy / res.y);
  dz -= res.z * round(dz / res.z);
  return sqrt(dx * dx + dy * dy + dz * dz) - R;
}
// face openness in [0.1, 1.0]: a smooth, strictly positive transmissibility (well-posed; true solid
// cut cells would use openness=0 with the A_C guard in mg_smooth_var_k, same assembly).
__host__ __device__ inline double openf(double sdf) { return 0.1 + 0.45 * (1.0 + tanh(sdf)); }

// ---- distributed: fill face-openness on the whole extended block from global coords ----
__global__ void d_fill_open(double* ox, double* oy, double* oz, int3 ext, int3 og, int3 res) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y,
      lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
  double gx = og.x + lx, gy = og.y + ly, gz = og.z + lz;
  ox[i] = openf(psdf(gx - 0.5, gy, gz, res));
  oy[i] = openf(psdf(gx, gy - 0.5, gz, res));
  oz[i] = openf(psdf(gx, gy, gz - 0.5, res));
}

// ---- serial full-grid reference (periodic wrap) ----
__global__ void s_fill_open(double* ox, double* oy, double* oz, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  ox[i] = openf(psdf(x - 0.5, y, z, res));
  oy[i] = openf(psdf(x, y - 0.5, z, res));
  oz[i] = openf(psdf(x, y, z - 0.5, res));
}

__global__ void s_build_op(double* AC, double* AW, double* AE, double* AS, double* AN, double* AB,
                           double* AT, const double* ox, const double* oy, const double* oz, int3 res,
                           double idx2, double idy2, double idz2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  double te = ox[get_idx(x + 1, y, z, res)] * idx2, tw = ox[i] * idx2;
  double tn = oy[get_idx(x, y + 1, z, res)] * idy2, ts = oy[i] * idy2;
  double tt = oz[get_idx(x, y, z + 1, res)] * idz2, tb = oz[i] * idz2;
  AE[i] = -te; AW[i] = -tw; AN[i] = -tn; AS[i] = -ts; AT[i] = -tt; AB[i] = -tb;
  AC[i] = te + tw + tn + ts + tt + tb;
}

__global__ void s_smooth_var(double* x, const double* b, const double* AC, const double* AW,
                             const double* AE, const double* AS, const double* AN, const double* AB,
                             const double* AT, int3 res, int color) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
      iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= res.x || iy >= res.y || iz >= res.z) return;
  if (((ix + iy + iz) & 1) != color) return;
  int i = get_idx(ix, iy, iz, res);
  double ac = AC[i];
  if (ac < 1e-300) return;
  double s = AE[i] * x[get_idx(ix + 1, iy, iz, res)] + AW[i] * x[get_idx(ix - 1, iy, iz, res)] +
             AN[i] * x[get_idx(ix, iy + 1, iz, res)] + AS[i] * x[get_idx(ix, iy - 1, iz, res)] +
             AT[i] * x[get_idx(ix, iy, iz + 1, res)] + AB[i] * x[get_idx(ix, iy, iz - 1, res)];
  x[i] = (b[i] - s) / ac;
}

__global__ void s_residual_var(double* r, const double* x, const double* b, const double* AC,
                               const double* AW, const double* AE, const double* AS, const double* AN,
                               const double* AB, const double* AT, int3 res) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y,
      iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= res.x || iy >= res.y || iz >= res.z) return;
  int i = get_idx(ix, iy, iz, res);
  double Ax = AC[i] * x[i] + AE[i] * x[get_idx(ix + 1, iy, iz, res)] +
              AW[i] * x[get_idx(ix - 1, iy, iz, res)] + AN[i] * x[get_idx(ix, iy + 1, iz, res)] +
              AS[i] * x[get_idx(ix, iy - 1, iz, res)] + AT[i] * x[get_idx(ix, iy, iz + 1, res)] +
              AB[i] * x[get_idx(ix, iy, iz - 1, res)];
  r[i] = b[i] - Ax;
}

// constant-coefficient coarse-level kernels (identical to test_multigrid_mpi)
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
  double *AC, *AW, *AE, *AS, *AN, *AB, *AT;
  int3 res3;
  double h;
  int n;
  bool variable;
};
static void s_remove_mean(double* d, int n) {
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
  for (int k = 0; k < sweeps; ++k)
    for (int c = 0; c < 2; ++c) {
      if (lv.variable)
        s_smooth_var<<<grd, blk>>>(lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB, lv.AT,
                                   lv.res3, c);
      else
        s_smooth<<<grd, blk>>>(lv.x, lv.rhs, lv.res3, h2, c);
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
  if (lv.variable)
    s_residual_var<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB,
                                 lv.AT, lv.res3);
  else
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
  const int n_levels = 4, n_vcycles = 4, pre = 2, post = 2, bottom = 12, seed = 9;
  const double h0 = 1.0, idx2 = 1.0 / (h0 * h0);
  dim3 blk(8, 8, 8);

  // ---- serial reference (each rank recomputes; cheap) ----
  slev.resize(n_levels);
  for (int L = 0; L < n_levels; ++L) {
    int3 r = make_int3(res.x >> L, res.y >> L, res.z >> L);
    SLevel& lv = slev[L];
    lv.res3 = r;
    lv.n = r.x * r.y * r.z;
    lv.h = h0 * (1 << L);
    lv.variable = (L == 0);
    cudaMalloc(&lv.x, lv.n * 8);
    cudaMalloc(&lv.rhs, lv.n * 8);
    cudaMalloc(&lv.res, lv.n * 8);
    cudaMemset(lv.x, 0, lv.n * 8);
    cudaMemset(lv.rhs, 0, lv.n * 8);
    cudaMemset(lv.res, 0, lv.n * 8);
    if (lv.variable) {
      for (double** p : {&lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB, &lv.AT}) cudaMalloc(p, lv.n * 8);
      double *ox, *oy, *oz;
      cudaMalloc(&ox, lv.n * 8); cudaMalloc(&oy, lv.n * 8); cudaMalloc(&oz, lv.n * 8);
      dim3 gF((r.x + 7) / 8, (r.y + 7) / 8, (r.z + 7) / 8);
      s_fill_open<<<gF, blk>>>(ox, oy, oz, r);
      s_build_op<<<gF, blk>>>(lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB, lv.AT, ox, oy, oz, r, idx2,
                              idx2, idx2);
      cudaFree(ox); cudaFree(oy); cudaFree(oz);
    } else {
      lv.AC = lv.AW = lv.AE = lv.AS = lv.AN = lv.AB = lv.AT = nullptr;
    }
  }
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  s_fill_rhs<<<gF, blk>>>(slev[0].rhs, res, seed);
  s_remove_mean(slev[0].rhs, slev[0].n);
  s_pre = pre; s_post = post; s_bottom = bottom;
  for (int v = 0; v < n_vcycles; ++v) s_vcycle(0);
  std::vector<double> sx(slev[0].n);
  cudaMemcpy(sx.data(), slev[0].x, slev[0].n * 8, cudaMemcpyDeviceToHost);
  for (auto& lv : slev) {
    cudaFree(lv.x); cudaFree(lv.rhs); cudaFree(lv.res);
    if (lv.variable)
      for (double* p : {lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB, lv.AT}) cudaFree(p);
  }

  // ---- distributed ----
  cfdmpi::DistributedPoissonMG mg;
  mg.init(res, rank, size, h0, n_levels, MPI_COMM_WORLD);
  cfdmpi::MGLevel& l0 = mg.level(0);
  // build the variable operator from analytic face openness on the extended block
  double *ox, *oy, *oz;
  cudaMalloc(&ox, l0.n * 8); cudaMalloc(&oy, l0.n * 8); cudaMalloc(&oz, l0.n * 8);
  dim3 gE((l0.ext.x + 7) / 8, (l0.ext.y + 7) / 8, (l0.ext.z + 7) / 8);
  d_fill_open<<<gE, blk>>>(ox, oy, oz, l0.ext, l0.og, res);
  mg.setFineVariableOperator(ox, oy, oz, idx2, idx2, idx2);
  cudaFree(ox); cudaFree(oy); cudaFree(oz);

  // fill level-0 rhs from the same hash, remove mean
  std::vector<double> hb(l0.n, 0.0);
  for (int lz = l0.g; lz < l0.ext.z - l0.g; ++lz)
    for (int ly = l0.g; ly < l0.ext.y - l0.g; ++ly)
      for (int lx = l0.g; lx < l0.ext.x - l0.g; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * l0.ext.x + (size_t)lz * l0.ext.x * l0.ext.y;
        hb[i] = hash01(lx + l0.og.x, ly + l0.og.y, lz + l0.og.z, seed);
      }
  cudaMemcpy(l0.rhs, hb.data(), l0.n * 8, cudaMemcpyHostToDevice);
  cfdmpi::mac_remove_mean(l0.rhs, l0.mac, MPI_COMM_WORLD);

  double r0_sum = 0, r0_max = 0;
  cfdmpi::mac_reduce(l0.rhs, l0.mac, MPI_COMM_WORLD, &r0_sum, &r0_max);

  mg.solve(n_vcycles, pre, post, bottom);

  // final residual (variable operator)
  {
    dim3 grd((l0.inner.x + 7) / 8, (l0.inner.y + 7) / 8, (l0.inner.z + 7) / 8);
    l0.mac.exchange(l0.x);
    cfdmpi::mgdetail::mg_residual_var_k<<<grd, blk>>>(l0.res, l0.x, l0.rhs, l0.AC, l0.AW, l0.AE,
                                                      l0.AS, l0.AN, l0.AB, l0.AT, l0.ext, l0.g);
  }
  double rf_sum = 0, rf_max = 0;
  cfdmpi::mac_reduce(l0.res, l0.mac, MPI_COMM_WORLD, &rf_sum, &rf_max);

  std::vector<double> lx(l0.n);
  cudaMemcpy(lx.data(), l0.x, l0.n * 8, cudaMemcpyDeviceToHost);
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
    double conv = rf_max / (r0_max + 1e-300);
    bool match_ok = (gmaxd <= 1e-8 && !std::isnan(gmaxd));
    bool conv_ok = (conv < 0.2 && !std::isnan(conv));
    fail = (match_ok && conv_ok) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  levels=%d vcycles=%d  variable fine operator (openness 0.1..1)\n",
           size, res.x, res.y, res.z, n_levels, n_vcycles);
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
