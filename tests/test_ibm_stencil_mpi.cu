// Distributed Robust-Scaled velocity IBM (mac_ibm.cuh) vs a serial full-grid reference: build the
// u-velocity diffusion stencil, run the IBM geometry from a sphere SDF, and bake the cut-cell factors
// into the stencil + inhomogeneous Dirichlet term. Both share the IBM math (ibm_fill_entry,
// ibm_modify_stencil_k); only the SDF sampling (extended clamp vs periodic wrap, both on a double SDF)
// and the cell linear index differ. So the IBM-modified A_C..A_T and the inhom term must match
// cell-for-cell. Validates the velocity IBM is assembled identically under decomposition (incl. cut
// cells straddling block boundaries). np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cut_cell_ibm.cuh"
#include "mac_halo.cuh"
#include "mac_ibm.cuh"

using namespace cfdmpi::ibmdetail;

__host__ __device__ inline double psdf(double x, double y, double z, int3 res) {
  double cx = res.x * 0.5, cy = res.y * 0.5, cz = res.z * 0.5, R = res.x * 0.3;
  double dx = x - cx, dy = y - cy, dz = z - cz;
  dx -= res.x * round(dx / res.x);
  dy -= res.y * round(dy / res.y);
  dz -= res.z * round(dz / res.z);
  return sqrt(dx * dx + dy * dy + dz * dz) - R;
}

// double, periodic-wrap trilinear (matches cc_sample_ext's formula with get_idx fetches)
__device__ inline double s_sample_wrap(const double* sdf, int3 res, double x, double y, double z) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto F = [&](int xx, int yy, int zz) { return sdf[get_idx(xx, yy, zz, res)]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}
__global__ void s_fill_sdf(double* sdf, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  sdf[get_idx(x, y, z, res)] = psdf(x, y, z, res);
}
__global__ void s_ibm_count(const double* sdf, int3 res, float3 off, int* counter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  float sc = (float)s_sample_wrap(sdf, res, x + off.x, y + off.y, z + off.z), sn[6];
  for (int k = 0; k < 6; ++k)
    sn[k] = (float)s_sample_wrap(sdf, res, x + d[k][0] + off.x, y + d[k][1] + off.y,
                                 z + d[k][2] + off.z);
  if (ibm_is_cut(sc, sn)) atomicAdd(counter, 1);
}
template <int SCHEME>
__global__ void s_ibm_geometry(IBM_Data ibm, int* id_map, const double* sdf, int3 res, float3 spacing,
                               int* counter, float3 off, int bc_type) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int idx = get_idx(x, y, z, res);
  const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  float sc = (float)s_sample_wrap(sdf, res, x + off.x, y + off.y, z + off.z), sn[6];
  for (int k = 0; k < 6; ++k)
    sn[k] = (float)s_sample_wrap(sdf, res, x + d[k][0] + off.x, y + d[k][1] + off.y,
                                 z + d[k][2] + off.z);
  if (!ibm_is_cut(sc, sn)) { id_map[idx] = -1; return; }
  int list_idx = atomicAdd(counter, 1);
  id_map[idx] = list_idx;
  ibm_fill_entry<SCHEME>(ibm, list_idx, idx, sc, sn, spacing, bc_type);
}
__global__ void s_build_diffusion(cfdmpi::mreal* AC, cfdmpi::mreal* AW, cfdmpi::mreal* AE,
                                  cfdmpi::mreal* AS, cfdmpi::mreal* AN, cfdmpi::mreal* AB,
                                  cfdmpi::mreal* AT, int n, double beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  AC[i] = (cfdmpi::mreal)(1.0 + 6.0 * beta);
  cfdmpi::mreal nb = (cfdmpi::mreal)(-beta);
  AW[i] = nb; AE[i] = nb; AS[i] = nb; AN[i] = nb; AB[i] = nb; AT[i] = nb;
}

// allocate the IBM_Data SoA for `n` cut cells
static IBM_Data alloc_ibm(int n) {
  IBM_Data d{};
  d.num_active_cells = n;
  int m = n > 0 ? n : 1;
  cudaMalloc(&d.cell_index, m * sizeof(int));
  cudaMalloc(&d.D_rescale, m * sizeof(float));
  cudaMalloc(&d.num_boundaries, m * sizeof(int));
  cudaMalloc(&d.dir_code, 6 * m * sizeof(int));
  cudaMalloc(&d.K_val, 6 * m * sizeof(float));
  cudaMalloc(&d.M_val, 6 * m * sizeof(float));
  cudaMalloc(&d.X_val, 6 * m * sizeof(float));
  cudaMalloc(&d.Nbc_val, 6 * m * sizeof(float));
  cudaMalloc(&d.R_val, 6 * m * sizeof(float));
  return d;
}
static void free_ibm(IBM_Data& d) {
  for (void* p : {(void*)d.cell_index, (void*)d.D_rescale, (void*)d.num_boundaries, (void*)d.dir_code,
                  (void*)d.K_val, (void*)d.M_val, (void*)d.X_val, (void*)d.Nbc_val, (void*)d.R_val})
    if (p) cudaFree(p);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  float3 spacing = make_float3(1, 1, 1);
  float3 off = make_float3(-0.5f, 0.0f, 0.0f);  // u-faces
  const float u_bc = 0.7f;                       // moving-wall BC -> nonzero inhom term
  const double beta = 0.3;                        // theta*dt*nu
  dim3 blk(8, 8, 8);
  size_t nf = (size_t)res.x * res.y * res.z;

  // ---- serial reference: stencil[7] + inhom over the full grid ----
  double* sdf;
  cudaMalloc(&sdf, nf * 8);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  s_fill_sdf<<<gF, blk>>>(sdf, res);
  int* cnt;
  cudaMalloc(&cnt, sizeof(int));
  cudaMemset(cnt, 0, sizeof(int));
  s_ibm_count<<<gF, blk>>>(sdf, res, off, cnt);
  int s_count = 0;
  cudaMemcpy(&s_count, cnt, sizeof(int), cudaMemcpyDeviceToHost);
  IBM_Data s_ibm = alloc_ibm(s_count);
  int* s_map;
  cudaMalloc(&s_map, nf * sizeof(int));
  cudaMemset(cnt, 0, sizeof(int));
  s_ibm_geometry<0><<<gF, blk>>>(s_ibm, s_map, sdf, res, spacing, cnt, off, 0);
  cfdmpi::mreal *sAC, *sAW, *sAE, *sAS, *sAN, *sAB, *sAT;
  double* sIN;
  for (cfdmpi::mreal** p : {&sAC, &sAW, &sAE, &sAS, &sAN, &sAB, &sAT})
    cudaMalloc(p, nf * sizeof(cfdmpi::mreal));
  cudaMalloc(&sIN, nf * 8);
  cudaMemset(sIN, 0, nf * 8);
  s_build_diffusion<<<(unsigned)((nf + 255) / 256), 256>>>(sAC, sAW, sAE, sAS, sAN, sAB, sAT, (int)nf,
                                                           beta);
  cudaMemcpy(&s_ibm.num_active_cells, cnt, sizeof(int), cudaMemcpyDeviceToHost);
  if (s_ibm.num_active_cells > 0)
    ibm_modify_stencil_k<<<(s_ibm.num_active_cells + 255) / 256, 256>>>(
        sAC, sAW, sAE, sAS, sAN, sAB, sAT, sIN, nullptr, s_ibm, u_bc);
  std::vector<std::vector<double>> S(8, std::vector<double>(nf));
  cfdmpi::mreal* sp[7] = {sAC, sAW, sAE, sAS, sAN, sAB, sAT};
  std::vector<cfdmpi::mreal> stmp(nf);
  for (int k = 0; k < 7; ++k) {
    cudaMemcpy(stmp.data(), sp[k], nf * sizeof(cfdmpi::mreal), cudaMemcpyDeviceToHost);
    for (size_t ii = 0; ii < nf; ++ii) S[k][ii] = stmp[ii];
  }
  cudaMemcpy(S[7].data(), sIN, nf * 8, cudaMemcpyDeviceToHost);
  cudaFree(sdf);
  for (cfdmpi::mreal* p : {sAC, sAW, sAE, sAS, sAN, sAB, sAT}) cudaFree(p);
  cudaFree(sIN);
  cudaFree(s_map); free_ibm(s_ibm);

  // ---- distributed: same on the extended block ----
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost=*/2, MPI_COMM_WORLD);
  int3 e = mac.local_ext, og = mac.origin_incl_ghost;
  int g = mac.ghost;
  size_t nl = mac.num_local_cells();
  std::vector<double> hs(nl);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx)
        hs[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] =
            psdf(lx + og.x, ly + og.y, lz + og.z, res);
  double* dsdf;
  cudaMalloc(&dsdf, nl * 8);
  cudaMemcpy(dsdf, hs.data(), nl * 8, cudaMemcpyHostToDevice);
  dim3 gI((mac.inner_res().x + 7) / 8, (mac.inner_res().y + 7) / 8, (mac.inner_res().z + 7) / 8);
  cudaMemset(cnt, 0, sizeof(int));
  ibm_count_ext_k<<<gI, blk>>>(dsdf, e, g, off, cnt);
  int d_count = 0;
  cudaMemcpy(&d_count, cnt, sizeof(int), cudaMemcpyDeviceToHost);
  IBM_Data d_ibm = alloc_ibm(d_count);
  int* d_map;
  cudaMalloc(&d_map, nl * sizeof(int));
  cudaMemset(cnt, 0, sizeof(int));
  ibm_geometry_ext_k<0><<<gI, blk>>>(d_ibm, d_map, dsdf, e, g, spacing, cnt, off, 0);
  // Mixed precision: the stencil A_C..A_T is single precision (cfdmpi::mreal); the inhom term is double.
  cfdmpi::mreal *dAC, *dAW, *dAE, *dAS, *dAN, *dAB, *dAT;
  double* dIN;
  for (cfdmpi::mreal** p : {&dAC, &dAW, &dAE, &dAS, &dAN, &dAB, &dAT})
    cudaMalloc(p, nl * sizeof(cfdmpi::mreal));
  cudaMalloc(&dIN, nl * 8);
  cudaMemset(dIN, 0, nl * 8);
  dim3 gE((e.x + 7) / 8, (e.y + 7) / 8, (e.z + 7) / 8);
  ibm_build_diffusion_k<<<gE, blk>>>(dAC, dAW, dAE, dAS, dAN, dAB, dAT, e, beta, /*idiag=*/1.0);
  cudaMemcpy(&d_ibm.num_active_cells, cnt, sizeof(int), cudaMemcpyDeviceToHost);
  if (d_ibm.num_active_cells > 0)
    ibm_modify_stencil_k<<<(d_ibm.num_active_cells + 255) / 256, 256>>>(
        dAC, dAW, dAE, dAS, dAN, dAB, dAT, dIN, nullptr, d_ibm, u_bc);
  std::vector<std::vector<double>> D(8, std::vector<double>(nl));
  cfdmpi::mreal* dp[7] = {dAC, dAW, dAE, dAS, dAN, dAB, dAT};
  std::vector<cfdmpi::mreal> tmp(nl);
  for (int k = 0; k < 7; ++k) {
    cudaMemcpy(tmp.data(), dp[k], nl * sizeof(cfdmpi::mreal), cudaMemcpyDeviceToHost);
    for (size_t ii = 0; ii < nl; ++ii) D[k][ii] = tmp[ii];
  }
  cudaMemcpy(D[7].data(), dIN, nl * 8, cudaMemcpyDeviceToHost);

  // compare inner cells cell-for-cell
  double maxd = 0.0;
  long ncut_local = 0;
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        size_t gi = (size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y;
        for (int k = 0; k < 8; ++k) maxd = fmax(maxd, fabs(D[k][li] - S[k][gi]));
      }
  ncut_local = d_ibm.num_active_cells;
  cudaFree(dsdf);
  for (cfdmpi::mreal* p : {dAC, dAW, dAE, dAS, dAN, dAB, dAT}) cudaFree(p);
  cudaFree(dIN);
  cudaFree(d_map); free_ibm(d_ibm); cudaFree(cnt);

  double gmaxd = 0.0;
  long gcut = 0;
  MPI_Reduce(&maxd, &gmaxd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ncut_local, &gcut, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  int fail = 0;
  if (rank == 0) {
    // Decomposition invariance: the single-precision stencil + double inhom are assembled by identical
    // per-cell ops on both paths, so distributed == serial bit-for-bit regardless of the block split.
    fail = (gmaxd > 1e-12 || std::isnan(gmaxd) || gcut == 0) ? 1 : 0;
    printf("np=%d  res=%dx%dx%d  velocity IBM (u-faces, sphere SDF, Robust-Scaled, float stencil)\n",
           size, res.x, res.y, res.z);
    printf("  cut cells (total) = %ld\n", gcut);
    printf("  distributed vs serial IBM-modified stencil + inhom: max|d| = %.3e   %s\n", gmaxd,
           fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
