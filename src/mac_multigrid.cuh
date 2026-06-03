// cfd-gpu -- distributed geometric multigrid for the periodic constant-coefficient pressure Poisson.
//
// The pure-Neumann pressure solve (Lap phi = rhs, periodic, all-fluid) is the canonical inner solve of
// the projection. This builds a V-cycle on a hierarchy of MacGridHalo levels: level L is the global
// grid at res>>L. Because transport-core's ORB cuts power-of-two grids at midpoints, each rank owns the
// SAME spatial sub-box halved at every level (asserted in init), so restriction (8:1 average) and
// prolongation (trilinear) are LOCAL within a rank's block -- only the per-level ghost exchange crosses
// ranks. Mean removal between levels uses the distributed reductions (mac_reductions.cuh).
//
// Operator (A = -Laplacian, spacing h per level, h doubling each level):
//   A x = b ;  A_C = 6/h^2 , A_off = -1/h^2.  RB-GS:  x[i] = (sum_neighbours + h^2 b[i]) / 6.
//   residual r[i] = b[i] - (6 x[i] - sum_neighbours)/h^2.
// Everything is double, so the distributed V-cycle reproduces a serial full-grid V-cycle cell-for-cell
// (the only per-cell difference, halo vs in-kernel wrap, yields identical neighbour values).
#pragma once

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mac_halo.cuh"
#include "mac_reductions.cuh"

namespace cfdmpi {
namespace mgdetail {

// RB-GS sweep over inner cells; colour by GLOBAL parity so the colouring is consistent across blocks.
__global__ void mg_smooth_k(double* __restrict__ x, const double* __restrict__ b, int3 ext, int3 og,
                            int g, double h2, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double sum = x[i + sx] + x[i - sx] + x[i + sy] + x[i - sy] + x[i + sz] + x[i - sz];
  x[i] = (sum + h2 * b[i]) / 6.0;
}

// ---- variable-coefficient fine-level operator (the SDF / cut-cell case) ----
// Assemble the 7-point coefficients from staggered face transmissibilities. ox[i] is the openness of
// the -x face of cell i (== the +x face of cell i-1); the +x face of cell i is ox[i+1]. The face is
// shared, so neighbouring ranks derive the SAME coefficient for it -> the operator is symmetric across
// block boundaries. Mirrors compute_pressure_operator_kernel (A_E = -frac_r*inv_dx2, etc.); here the
// face openness stands in for the masked fluid fraction.
__global__ void mg_build_op_k(double* AC, double* AW, double* AE, double* AS, double* AN, double* AB,
                              double* AT, const double* ox, const double* oy, const double* oz,
                              int3 ext, int g, double idx2, double idy2, double idz2) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double te = ox[i + sx] * idx2, tw = ox[i] * idx2;
  double tn = oy[i + sy] * idy2, ts = oy[i] * idy2;
  double tt = oz[i + sz] * idz2, tb = oz[i] * idz2;
  AE[i] = -te;
  AW[i] = -tw;
  AN[i] = -tn;
  AS[i] = -ts;
  AT[i] = -tt;
  AB[i] = -tb;
  AC[i] = te + tw + tn + ts + tt + tb;
}

__global__ void mg_smooth_var_k(double* __restrict__ x, const double* __restrict__ b,
                                const double* AC, const double* AW, const double* AE,
                                const double* AS, const double* AN, const double* AB,
                                const double* AT, int3 ext, int3 og, int g, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double ac = AC[i];
  if (ac < 1e-300) return;  // fully closed (solid) cell
  double s = AE[i] * x[i + sx] + AW[i] * x[i - sx] + AN[i] * x[i + sy] + AS[i] * x[i - sy] +
             AT[i] * x[i + sz] + AB[i] * x[i - sz];
  x[i] = (b[i] - s) / ac;
}

__global__ void mg_residual_var_k(double* __restrict__ r, const double* __restrict__ x,
                                  const double* __restrict__ b, const double* AC, const double* AW,
                                  const double* AE, const double* AS, const double* AN,
                                  const double* AB, const double* AT, int3 ext, int g) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double Ax = AC[i] * x[i] + AE[i] * x[i + sx] + AW[i] * x[i - sx] + AN[i] * x[i + sy] +
              AS[i] * x[i - sy] + AT[i] * x[i + sz] + AB[i] * x[i - sz];
  r[i] = b[i] - Ax;
}

__global__ void mg_residual_k(double* __restrict__ r, const double* __restrict__ x,
                              const double* __restrict__ b, int3 ext, int g, double invh2) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  double sum = x[i + sx] + x[i - sx] + x[i + sy] + x[i - sy] + x[i + sz] + x[i - sz];
  r[i] = b[i] - invh2 * (6.0 * x[i] - sum);
}

// 8:1 average of the fine residual into the coarse rhs. Local: the 8 fine cells are in this rank's
// fine block (origins align), so no exchange is needed (reads fine INNER cells only).
__global__ void mg_restrict_k(double* __restrict__ coarse, const double* __restrict__ fine, int3 cext,
                              int3 fext, int g, int3 cinner) {
  int icx = blockIdx.x * blockDim.x + threadIdx.x;
  int icy = blockIdx.y * blockDim.y + threadIdx.y;
  int icz = blockIdx.z * blockDim.z + threadIdx.z;
  if (icx >= cinner.x || icy >= cinner.y || icz >= cinner.z) return;
  size_t fsy = fext.x, fsz = (size_t)fext.x * fext.y;
  double sum = 0.0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        int fx = 2 * icx + dx + g, fy = 2 * icy + dy + g, fz = 2 * icz + dz + g;
        sum += fine[(size_t)fx + (size_t)fy * fsy + (size_t)fz * fsz];
      }
  size_t ci = (size_t)(icx + g) + (size_t)(icy + g) * cext.x + (size_t)(icz + g) * cext.x * cext.y;
  coarse[ci] = 0.125 * sum;
}

__device__ inline double trilerp_ext(const double* c, double x, double y, double z, int3 cext) {
  double fx = floor(x), fy = floor(y), fz = floor(z);
  double wx = x - fx, wy = y - fy, wz = z - fz;
  int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
  size_t sy = cext.x, sz = (size_t)cext.x * cext.y;
  auto F = [&](int xx, int yy, int zz) { return c[(size_t)xx + (size_t)yy * sy + (size_t)zz * sz]; };
  double c00 = F(x0, y0, z0) * (1 - wx) + F(x0 + 1, y0, z0) * wx;
  double c10 = F(x0, y0 + 1, z0) * (1 - wx) + F(x0 + 1, y0 + 1, z0) * wx;
  double c01 = F(x0, y0, z0 + 1) * (1 - wx) + F(x0 + 1, y0, z0 + 1) * wx;
  double c11 = F(x0, y0 + 1, z0 + 1) * (1 - wx) + F(x0 + 1, y0 + 1, z0 + 1) * wx;
  double c0 = c00 * (1 - wy) + c10 * wy;
  double c1 = c01 * (1 - wy) + c11 * wy;
  return c0 * (1 - wz) + c1 * wz;
}

// Trilinear prolongation of the coarse correction, added to the fine solution. Needs the coarse ghost
// layer filled first (the stencil reaches one coarse cell beyond the block edge); sampling is then
// direct in the coarse extended block (no wrap). Coarse local sample coord = 0.5*ifine - 0.25 + g.
__global__ void mg_prolong_k(double* __restrict__ fine, const double* __restrict__ coarse, int3 fext,
                             int3 cext, int g, int3 finner) {
  int ifx = blockIdx.x * blockDim.x + threadIdx.x;
  int ify = blockIdx.y * blockDim.y + threadIdx.y;
  int ifz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ifx >= finner.x || ify >= finner.y || ifz >= finner.z) return;
  double cx = 0.5 * ifx - 0.25 + g, cy = 0.5 * ify - 0.25 + g, cz = 0.5 * ifz - 0.25 + g;
  size_t fi = (size_t)(ifx + g) + (size_t)(ify + g) * fext.x + (size_t)(ifz + g) * fext.x * fext.y;
  fine[fi] += trilerp_ext(coarse, cx, cy, cz, cext);
}

}  // namespace mgdetail

struct MGLevel {
  MacGridHalo mac;
  double* x = nullptr;    // solution / correction (extended block)
  double* rhs = nullptr;  // right-hand side
  double* res = nullptr;  // residual scratch
  double h = 1.0;         // grid spacing
  size_t n = 0;
  int3 ext{}, og{}, inner{};
  int g = 1;
  // variable-coefficient operator (fine level only); null on constant-coefficient levels
  bool variable = false;
  double *AC = nullptr, *AW = nullptr, *AE = nullptr, *AS = nullptr, *AN = nullptr, *AB = nullptr,
         *AT = nullptr;
};

// Distributed V-cycle solver for the periodic constant-coefficient Poisson on power-of-two grids.
class DistributedPoissonMG {
 public:
  // global_res must be power-of-two and divisible by 2^(n_levels-1) (with even per-rank blocks). h0 is
  // the finest spacing; comm the communicator. Asserts level-to-level partition alignment.
  void init(int3 global_res, int rank, int size, double h0, int n_levels, MPI_Comm comm) {
    comm_ = comm;
    levels_.clear();
    for (int L = 0; L < n_levels; ++L) levels_.push_back(std::make_unique<MGLevel>());
    for (int L = 0; L < n_levels; ++L) {
      int3 r = make_int3(global_res.x >> L, global_res.y >> L, global_res.z >> L);
      MGLevel& lv = *levels_[L];
      lv.mac.init(r, rank, size, {true, true, true}, /*ghost=*/1, comm);
      lv.g = lv.mac.ghost;
      lv.ext = lv.mac.local_ext;
      lv.og = lv.mac.origin_incl_ghost;
      lv.inner = lv.mac.inner_res();
      lv.n = lv.mac.num_local_cells();
      lv.h = h0 * (double)(1 << L);
      cudaMalloc(&lv.x, lv.n * 8);
      cudaMalloc(&lv.rhs, lv.n * 8);
      cudaMalloc(&lv.res, lv.n * 8);
      cudaMemset(lv.x, 0, lv.n * 8);
      cudaMemset(lv.rhs, 0, lv.n * 8);
      cudaMemset(lv.res, 0, lv.n * 8);
      if (L > 0) {
        const MGLevel& f = *levels_[L - 1];
        // inner-block start = origin_incl_ghost + ghost; coarse start must be exactly half the fine
        // start, and coarse inner half the fine inner, so 2:1 coarsening stays local per rank.
        int fsx = f.og.x + f.g, fsy = f.og.y + f.g, fsz = f.og.z + f.g;
        int csx = lv.og.x + lv.g, csy = lv.og.y + lv.g, csz = lv.og.z + lv.g;
        bool aligned = (f.inner.x == 2 * lv.inner.x) && (f.inner.y == 2 * lv.inner.y) &&
                       (f.inner.z == 2 * lv.inner.z) && (fsx == 2 * csx) && (fsy == 2 * csy) &&
                       (fsz == 2 * csz);
        if (!aligned) {
          int rk = 0;
          MPI_Comm_rank(comm, &rk);
          fprintf(stderr,
                  "[DistributedPoissonMG] rank %d: level %d block not aligned to level %d "
                  "(fine inner %d,%d,%d start %d,%d,%d ; coarse inner %d,%d,%d start %d,%d,%d). "
                  "Requires power-of-two global_res with even per-rank blocks.\n",
                  rk, L, L - 1, f.inner.x, f.inner.y, f.inner.z, fsx, fsy, fsz, lv.inner.x,
                  lv.inner.y, lv.inner.z, csx, csy, csz);
          MPI_Abort(comm, 1);
        }
      }
    }
  }

  void free() {
    for (auto& lp : levels_) {
      MGLevel& lv = *lp;
      for (double** p : {&lv.x, &lv.rhs, &lv.res, &lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB,
                         &lv.AT}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
      }
    }
    levels_.clear();
  }

  // Install a variable-coefficient operator on the fine level (level 0) from staggered face
  // openness/transmissibility fields on the extended block: ox[i] = openness of the -x face of cell i
  // (similarly oy, oz). The caller fills ox/oy/oz on the WHOLE extended block (incl. ghosts); because
  // a face is a deterministic function of its global position, every rank derives matching values for
  // shared faces (no exchange needed). idx2 = 1/dx^2, etc. Coarse levels stay constant-coefficient
  // (mirrors the serial use_periodic_operator = level>0).
  void setFineVariableOperator(const double* ox, const double* oy, const double* oz, double idx2,
                               double idy2, double idz2) {
    MGLevel& lv = *levels_[0];
    if (!lv.AC) {
      for (double** p : {&lv.AC, &lv.AW, &lv.AE, &lv.AS, &lv.AN, &lv.AB, &lv.AT})
        cudaMalloc(p, lv.n * 8);
    }
    lv.variable = true;
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    mgdetail::mg_build_op_k<<<grd, blk>>>(lv.AC, lv.AW, lv.AE, lv.AS, lv.AN, lv.AB, lv.AT, ox, oy, oz,
                                          lv.ext, lv.g, idx2, idy2, idz2);
  }

  MGLevel& level(int L) { return *levels_[L]; }
  int n_levels() const { return (int)levels_.size(); }

  // Run n_vcycles of the V-cycle starting at level 0 (rhs on level 0 must be set; x starts at 0).
  void solve(int n_vcycles, int pre, int post, int bottom) {
    pre_ = pre;
    post_ = post;
    bottom_ = bottom;
    for (int v = 0; v < n_vcycles; ++v) vcycle(0);
  }

 private:
  void smooth(MGLevel& lv, int sweeps) {
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    double h2 = lv.h * lv.h;
    for (int k = 0; k < sweeps; ++k) {
      for (int color = 0; color < 2; ++color) {
        lv.mac.exchange(lv.x);
        if (lv.variable)
          mgdetail::mg_smooth_var_k<<<grd, blk>>>(lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS, lv.AN,
                                                  lv.AB, lv.AT, lv.ext, lv.og, lv.g, color);
        else
          mgdetail::mg_smooth_k<<<grd, blk>>>(lv.x, lv.rhs, lv.ext, lv.og, lv.g, h2, color);
      }
    }
  }

  void residual(MGLevel& lv) {
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);
    lv.mac.exchange(lv.x);
    if (lv.variable)
      mgdetail::mg_residual_var_k<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.AC, lv.AW, lv.AE, lv.AS,
                                                lv.AN, lv.AB, lv.AT, lv.ext, lv.g);
    else
      mgdetail::mg_residual_k<<<grd, blk>>>(lv.res, lv.x, lv.rhs, lv.ext, lv.g, 1.0 / (lv.h * lv.h));
  }

  void vcycle(int L) {
    MGLevel& lv = *levels_[L];
    dim3 blk(8, 8, 8);
    dim3 grd((lv.inner.x + 7) / 8, (lv.inner.y + 7) / 8, (lv.inner.z + 7) / 8);

    if (L + 1 == (int)levels_.size()) {  // bottom: smooth + remove null space
      smooth(lv, bottom_);
      mac_remove_mean(lv.x, lv.mac, comm_);
      return;
    }

    smooth(lv, pre_);

    // residual (needs current ghosts); variable operator on the fine level, constant on coarse
    residual(lv);

    // restrict residual -> coarse rhs (local); reset coarse x
    MGLevel& cs = *levels_[L + 1];
    dim3 cgrd((cs.inner.x + 7) / 8, (cs.inner.y + 7) / 8, (cs.inner.z + 7) / 8);
    mgdetail::mg_restrict_k<<<cgrd, blk>>>(cs.rhs, lv.res, cs.ext, lv.ext, lv.g, cs.inner);
    cudaMemset(cs.x, 0, cs.n * 8);

    vcycle(L + 1);

    // prolong coarse correction -> fine (needs coarse ghosts)
    cs.mac.exchange(cs.x);
    mgdetail::mg_prolong_k<<<grd, blk>>>(lv.x, cs.x, lv.ext, cs.ext, lv.g, lv.inner);

    smooth(lv, post_);
    mac_remove_mean(lv.x, lv.mac, comm_);
  }

  std::vector<std::unique_ptr<MGLevel>> levels_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int pre_ = 2, post_ = 2, bottom_ = 8;
};

}  // namespace cfdmpi
