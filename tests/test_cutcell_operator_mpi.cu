// Distributed cut-cell pressure operator (mac_cutcell.cuh: cfd's gradient-normalised fluid fraction,
// masked, -> staggered face openness -> 7-point A_C..A_T) vs a serial full-grid reference. Both build
// the operator from the SAME analytic periodic sphere SDF, sharing the fraction arithmetic
// (cc_fraction_core); the distributed build reads ghost SDF values that equal the serial wrapped ones,
// so the per-cell coefficients must match cell-for-cell. This validates the real SDF/fraction operator
// is assembled identically under domain decomposition (incl. faces straddling block boundaries). np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfd_solver.cuh"          // get_idx (periodic wrap)
#include "distributed_stokes.cuh"  // pulls in mac_cutcell.cuh + mac_multigrid.cuh

using cfdmpi::ccdetail::cc_fraction_core;

__host__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

__host__ __device__ inline double psdf(double x, double y, double z, int3 res) {
  double cx = res.x * 0.5, cy = res.y * 0.5, cz = res.z * 0.5, R = res.x * 0.3;
  double dx = x - cx, dy = y - cy, dz = z - cz;
  dx -= res.x * round(dx / res.x);
  dy -= res.y * round(dy / res.y);
  dz -= res.z * round(dz / res.z);
  return sqrt(dx * dx + dy * dy + dz * dz) - R;
}

// ---- serial full-grid reference (periodic wrap) ----
__global__ void s_fill_sdf(double* sdf, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  sdf[get_idx(x, y, z, res)] = psdf(x, y, z, res);
}
__device__ inline double s_sample_wrap(const double* sdf, int3 res, double x, double y, double z) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  auto F = [&](int xx, int yy, int zz) { return sdf[get_idx(xx, yy, zz, res)]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy;
  double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}
__device__ inline double s_face_open(const double* sdf, int3 res, double fx, double fy, double fz,
                                     int type) {
  double sd = s_sample_wrap(sdf, res, fx, fy, fz);
  if (sd <= 0.0) return 0.0;
  double e = 1.0;
  return cc_fraction_core(sd, s_sample_wrap(sdf, res, fx + e, fy, fz),
                          s_sample_wrap(sdf, res, fx - e, fy, fz),
                          s_sample_wrap(sdf, res, fx, fy + e, fz),
                          s_sample_wrap(sdf, res, fx, fy - e, fz),
                          s_sample_wrap(sdf, res, fx, fy, fz + e),
                          s_sample_wrap(sdf, res, fx, fy, fz - e), type, 1.0, 1.0, 1.0);
}
__global__ void s_build_open(double* ox, double* oy, double* oz, const double* sdf, int3 res) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  ox[i] = s_face_open(sdf, res, x - 0.5, y, z, 1);
  oy[i] = s_face_open(sdf, res, x, y - 0.5, z, 2);
  oz[i] = s_face_open(sdf, res, x, y, z - 0.5, 3);
}
__global__ void s_build_op(double* AC, double* AW, double* AE, double* AS, double* AN, double* AB,
                           double* AT, const double* ox, const double* oy, const double* oz, int3 res,
                           double idx2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= res.x || y >= res.y || z >= res.z) return;
  int i = get_idx(x, y, z, res);
  double te = ox[get_idx(x + 1, y, z, res)] * idx2, tw = ox[i] * idx2;
  double tn = oy[get_idx(x, y + 1, z, res)] * idx2, ts = oy[i] * idx2;
  double tt = oz[get_idx(x, y, z + 1, res)] * idx2, tb = oz[i] * idx2;
  AE[i] = -te; AW[i] = -tw; AN[i] = -tn; AS[i] = -ts; AT[i] = -tt; AB[i] = -tb;
  AC[i] = te + tw + tn + ts + tt + tb;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  dim3 blk(8, 8, 8);
  size_t nf = (size_t)res.x * res.y * res.z;

  // ---- serial reference coefficients (full grid) ----
  double *sdf, *ox, *oy, *oz, *sAC, *sAW, *sAE, *sAS, *sAN, *sAB, *sAT;
  for (double** p : {&sdf, &ox, &oy, &oz, &sAC, &sAW, &sAE, &sAS, &sAN, &sAB, &sAT})
    cudaMalloc(p, nf * 8);
  dim3 gF((res.x + 7) / 8, (res.y + 7) / 8, (res.z + 7) / 8);
  s_fill_sdf<<<gF, blk>>>(sdf, res);
  s_build_open<<<gF, blk>>>(ox, oy, oz, sdf, res);
  s_build_op<<<gF, blk>>>(sAC, sAW, sAE, sAS, sAN, sAB, sAT, ox, oy, oz, res, 1.0);
  std::vector<std::vector<double>> S(7, std::vector<double>(nf));
  double* sptr[7] = {sAC, sAW, sAE, sAS, sAN, sAB, sAT};
  for (int k = 0; k < 7; ++k) cudaMemcpy(S[k].data(), sptr[k], nf * 8, cudaMemcpyDeviceToHost);
  for (double* p : {sdf, ox, oy, oz, sAC, sAW, sAE, sAS, sAN, sAB, sAT}) cudaFree(p);

  // ---- distributed: build the cut-cell operator on the MG fine level ----
  cfdmpi::DistributedPoissonMG mg;
  mg.init(res, rank, size, /*h0=*/1.0, /*n_levels=*/2, MPI_COMM_WORLD, /*ghost=*/2);
  cfdmpi::MGLevel& l0 = mg.level(0);
  int3 e = l0.ext, og = l0.og;
  // SDF on the whole extended block from global coords (periodic psdf -> ghosts match the serial wrap)
  std::vector<double> hs(l0.n);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx)
        hs[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] =
            psdf(lx + og.x, ly + og.y, lz + og.z, res);
  double *dsdf, *dox, *doy, *doz;
  for (double** p : {&dsdf, &dox, &doy, &doz}) cudaMalloc(p, l0.n * 8);
  cudaMemcpy(dsdf, hs.data(), l0.n * 8, cudaMemcpyHostToDevice);
  dim3 gE((e.x + 7) / 8, (e.y + 7) / 8, (e.z + 7) / 8);
  cfdmpi::ccdetail::cc_build_open_k<<<gE, blk>>>(dox, doy, doz, dsdf, e, 1.0, 1.0, 1.0);
  mg.setFineVariableOperator(dox, doy, doz, 1.0, 1.0, 1.0);
  for (double* p : {dsdf, dox, doy, doz}) cudaFree(p);

  std::vector<std::vector<double>> D(7, std::vector<double>(l0.n));
  double* dptr[7] = {l0.AC, l0.AW, l0.AE, l0.AS, l0.AN, l0.AB, l0.AT};
  for (int k = 0; k < 7; ++k) cudaMemcpy(D[k].data(), dptr[k], l0.n * 8, cudaMemcpyDeviceToHost);

  // compare inner-cell coefficients to the serial reference
  double maxd = 0.0;
  int g = l0.g;
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        size_t li = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        int gx = lx + og.x, gy = ly + og.y, gz = lz + og.z;
        size_t gi = (size_t)gx + (size_t)gy * res.x + (size_t)gz * res.x * res.y;
        for (int k = 0; k < 7; ++k) maxd = fmax(maxd, fabs(D[k][li] - S[k][gi]));
      }
  mg.free();

  double gmaxd = 0.0;
  MPI_Reduce(&maxd, &gmaxd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // ---- part 2: drive DistributedStokes through set_cutcell_pressure_operator. The stiff cut-cell
  // operator needs CG accelerating the Galerkin V-cycle (a plain V-cycle stalls); compare the two. ----
  auto run_cutcell = [&](bool pcg, int budget, double& dv0, double& dv1, double& rms1) {
    dstokes::DistributedStokes s;
    s.init(res, rank, size, /*nu=*/0.0, /*dt=*/0.1);
    int3 e2 = s.ext(), og2 = s.origin_incl_ghost();
    int g2 = s.ghost();
    size_t n2 = s.num_cells();
    std::vector<double> sdfx(n2), solid(n2, 0.0), hu(n2, 0), hv(n2, 0), hw(n2, 0);
    for (int lz = 0; lz < e2.z; ++lz)
      for (int ly = 0; ly < e2.y; ++ly)
        for (int lx = 0; lx < e2.x; ++lx) {
          size_t i = (size_t)lx + (size_t)ly * e2.x + (size_t)lz * e2.x * e2.y;
          double sd = psdf(lx + og2.x, ly + og2.y, lz + og2.z, res);
          sdfx[i] = sd;
          solid[i] = (sd < 0.0) ? 1.0 : 0.0;
          if (lx >= g2 && ly >= g2 && lz >= g2 && lx < e2.x - g2 && ly < e2.y - g2 &&
              lz < e2.z - g2) {
            hu[i] = hash01(lx + og2.x, ly + og2.y, lz + og2.z, 11);
            hv[i] = hash01(lx + og2.x, ly + og2.y, lz + og2.z, 22);
            hw[i] = hash01(lx + og2.x, ly + og2.y, lz + og2.z, 33);
          }
        }
    s.upload_velocity(hu.data(), hv.data(), hw.data());
    // NB: no set_solid here -- the cut-cell operator handles the geometry through the open fractions;
    // velocity masking would zero partially-open solid-adjacent faces and reintroduce divergence
    // (an inconsistency between masking and the open-weighted operator).
    (void)solid;
    s.set_cutcell_pressure_operator(sdfx, /*galerkin=*/true);  // real cut-cell operator, Galerkin coarse
    if (pcg) s.set_pressure_pcg(true, budget, 1e-10);
    dv0 = s.max_open_divergence();                 // cut-cell flux divergence (what is projected)
    s.step(/*n_diff=*/0, /*n_pois=*/budget);       // budget = V-cycles (no PCG) or CG iteration cap
    dv1 = s.max_open_divergence();
    rms1 = s.rms_open_divergence();
  };

  double vc_dv0, vc_dv1, vc_rms, pc_dv0, pc_dv1, pc_rms;
  run_cutcell(/*pcg=*/false, /*budget=*/12, vc_dv0, vc_dv1, vc_rms);   // Galerkin V-cycles alone
  run_cutcell(/*pcg=*/true, /*budget=*/80, pc_dv0, pc_dv1, pc_rms);    // CG + Galerkin V-cycle

  int fail = 0;
  if (rank == 0) {
    bool coeff_ok = (gmaxd <= 1e-12 && !std::isnan(gmaxd));
    // CG drives the bulk (RMS) flux divergence to ~0; the max-norm floor is a single thin cut cell.
    bool pc_ok = std::isfinite(pc_rms) && pc_rms < 1e-6 * vc_rms;
    bool better = pc_rms < 0.01 * vc_rms;
    fail = (coeff_ok && pc_ok && better) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  cut-cell operator (sphere SDF, real fluid fraction)\n", size, res.x,
           res.y, res.z);
    printf("  [1] distributed vs serial coefficients: max|d| = %.3e   %s\n", gmaxd,
           coeff_ok ? "OK" : "FAIL");
    printf("  [2] Galerkin V-cycle projection (12):  rms|flux div| %.3e\n", vc_rms);
    printf("  [3] CG + Galerkin projection:          rms|flux div| %.3e   (max %.3e)   %.1fx better rms   %s\n",
           pc_rms, pc_dv1, vc_rms / pc_rms, (pc_ok && better) ? "OK" : "FAIL");
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
