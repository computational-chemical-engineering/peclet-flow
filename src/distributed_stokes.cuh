// cfd-gpu — a small reusable distributed incompressible (unsteady Stokes) solver.
//
// Consolidates the building blocks validated in tests/test_*_mpi.cu into one component: a staggered
// MAC velocity field decomposed into rank-owned blocks (transport-core ORB), advanced each step by
// per-component implicit diffusion (backward Euler, Red-Black Gauss-Seidel with halo exchange between
// sweeps) + Chorin projection, with optional body force and SDF-described solids (no-slip by
// per-cell velocity masking). Periodic. Host-staged GPU halo (see ../transport-core).
//
// Scope: Stokes (no nonlinear advection) — the validated core. Nonlinear advection (Step 3 Koren
// operator, ghost width 2) and the full Robust-Scaled cut-cell IBM layer on top; see
// doc/mpi_parallelization_status.md.
#pragma once

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "cfd_solver.cuh"  // get_idx (unused at runtime here; kept for convention parity)
#include "mac_halo.cuh"
#include "mac_multigrid.cuh"  // DistributedPoissonMG (opt-in multigrid pressure solve)
#include "staggered_advection.cuh"

namespace dstokes {

namespace detail {
__device__ inline long L3(int x, int y, int z, int3 e) {
  return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
}
// dst = -src over the whole extended block (build the multigrid RHS b = -div: the V-cycle solves
// A x = b with A = -Laplacian, so x converges to phi with Lap phi = div, matching pois_k).
__global__ void neg_k(double* dst, const double* src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = -src[i];
}
// b = comp + dt*f   (body force folded into the implicit-diffusion RHS), inner cells.
__global__ void rhs_k(const double* c, double* b, double dtf, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e);
  b[i] = c[i] + dtf;
}
// b = comp - dt*advect(comp) + dt*f : explicit nonlinear advection folded into the diffusion RHS.
__global__ void advect_rhs_k(int comp, const double* u, const double* v, const double* w,
                             const double* phi, double* b, double dt, double f, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  double A = sadv::advect(comp, x, y, z, sadv::LocAcc{u, e}, sadv::LocAcc{v, e}, sadv::LocAcc{w, e},
                          sadv::LocAcc{phi, e});
  long i = L3(x, y, z, e);
  b[i] = phi[i] - dt * A + dt * f;
}
// one Red-Black diffusion sweep: c <- (b + beta*sum_nbr c)/(1+6 beta), global parity.
__global__ void diff_k(double* c, const double* b, int3 e, int3 og, int g, double beta, double Ac,
                       int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = c[i + sx] + c[i - sx] + c[i + sy] + c[i - sy] + c[i + sz] + c[i - sz];
  c[i] = (b[i] + beta * s) / Ac;
}
// zero velocity where the cell is solid (mask 1.0 = solid). Masks all cells (inner + ghost) so walls
// stay consistent without an extra exchange.
__global__ void mask_k(double* c, const double* solid, int3 e) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= e.x || y >= e.y || z >= e.z) return;
  long i = L3(x, y, z, e);
  if (solid[i] > 0.5) c[i] = 0.0;
}
__global__ void diverg_k(const double* u, const double* v, const double* w, double* d, int3 e,
                         int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  d[i] = (u[i + sx] - u[i]) + (v[i + sy] - v[i]) + (w[i + sz] - w[i]);
}
__global__ void pois_k(double* phi, const double* d, int3 e, int3 og, int g, int color) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = phi[i + sx] + phi[i - sx] + phi[i + sy] + phi[i - sy] + phi[i + sz] + phi[i - sz];
  phi[i] = (s - d[i]) / 6.0;
}
__global__ void correct_k(double* u, double* v, double* w, const double* phi, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  u[i] -= phi[i] - phi[i - sx];
  v[i] -= phi[i] - phi[i - sy];
  w[i] -= phi[i] - phi[i - sz];
}
}  // namespace detail

class DistributedStokes {
 public:
  void init(int3 global_res, int rank, int size, double nu, double dt,
            MPI_Comm comm = MPI_COMM_WORLD) {
    // Ghost width 2 covers the Koren advection reach (diffusion/projection use only width 1).
    mac_.init(global_res, rank, size, {true, true, true}, /*ghost_width=*/2, comm);
    comm_ = comm;
    nu_ = nu;
    dt_ = dt;
    n_ = mac_.num_local_cells();
    for (double** p : {&u_, &v_, &w_, &phi_, &div_, &solid_, &b_[0], &b_[1], &b_[2]})
      cudaMalloc(p, n_ * sizeof(double));
    cudaMemset(u_, 0, n_ * 8);
    cudaMemset(v_, 0, n_ * 8);
    cudaMemset(w_, 0, n_ * 8);
    cudaMemset(solid_, 0, n_ * 8);
    ext_ = mac_.local_ext;
    blk_ = dim3(8, 8, 4);
    grd_ = dim3((ext_.x + 7) / 8, (ext_.y + 7) / 8, (ext_.z + 3) / 4);
  }
  ~DistributedStokes() {
    for (double* p : {u_, v_, w_, phi_, div_, solid_, b_[0], b_[1], b_[2]})
      if (p) cudaFree(p);
    if (mg_built_) mg_.free();
  }
  DistributedStokes() = default;
  DistributedStokes(const DistributedStokes&) = delete;
  DistributedStokes& operator=(const DistributedStokes&) = delete;

  const MacGridHalo& halo() const { return mac_; }
  int ghost() const { return mac_.ghost; }
  int3 ext() const { return ext_; }
  int3 origin_incl_ghost() const { return mac_.origin_incl_ghost; }
  std::size_t num_cells() const { return n_; }
  double* u() { return u_; }
  double* v() { return v_; }
  double* w() { return w_; }

  void set_body_force(double fx, double fy, double fz) { fx_ = fx; fy_ = fy; fz_ = fz; }
  // Enable explicit nonlinear advection (full Navier-Stokes instead of Stokes).
  void set_advection(bool on) { advection_ = on; }

  // Opt-in: solve the pressure Poisson with the distributed geometric multigrid V-cycle
  // (src/mac_multigrid.cuh) instead of the single-level Red-Black Gauss-Seidel. Both solve the same
  // periodic constant-coefficient Laplacian; the V-cycle converges far faster per unit work. Requires
  // a power-of-two global resolution divisible by 2^(n_levels-1) with even per-rank blocks (the MG
  // asserts this). step()'s n_pois argument then counts V-cycles, not GS sweeps. Built lazily on the
  // first step so it can be configured after init().
  void set_pressure_multigrid(bool on, int n_levels = 4, int pre = 2, int post = 2, int bottom = 12) {
    mg_enabled_ = on;
    mg_levels_ = n_levels;
    mg_pre_ = pre;
    mg_post_ = post;
    mg_bottom_ = bottom;
  }

  // Upload the per-cell solid mask (extended-block layout; 1.0 = solid). Caller builds it (e.g. from
  // an SDF) for all local cells including ghosts (mask is a function of global position).
  void set_solid(const std::vector<double>& solid_ext) {
    cudaMemcpy(solid_, solid_ext.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
    has_solid_ = true;
    apply_mask(u_); apply_mask(v_); apply_mask(w_);
  }

  void upload_velocity(const double* u_ext, const double* v_ext, const double* w_ext) {
    cudaMemcpy(u_, u_ext, n_ * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(v_, v_ext, n_ * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(w_, w_ext, n_ * 8, cudaMemcpyHostToDevice);
  }
  void download_velocity(double* u_ext, double* v_ext, double* w_ext) {
    cudaMemcpy(u_ext, u_, n_ * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_ext, v_, n_ * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_ext, w_, n_ * 8, cudaMemcpyDeviceToHost);
  }
  // Refresh the ghost layer of all velocity components (so a host-side stencil over the downloaded
  // extended array sees current neighbour values at block boundaries).
  void exchange_all() { mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_); }

  // Assemble the global field of a device component (u_/v_/w_) onto `root` in x-fastest order.
  // Returns the global array on root, empty elsewhere. Each rank contributes its block's inner cells;
  // block geometry is known to every rank from the (replicated) decomposition.
  std::vector<double> gather_to_root(double* comp_dev, int root = 0) {
    std::vector<double> host(n_);
    cudaMemcpy(host.data(), comp_dev, n_ * sizeof(double), cudaMemcpyDeviceToHost);
    int g = mac_.ghost;
    int3 e = ext_;
    int3 isz = make_int3(e.x - 2 * g, e.y - 2 * g, e.z - 2 * g);
    std::vector<double> loc((size_t)isz.x * isz.y * isz.z);
    size_t t = 0;
    for (int bz = 0; bz < isz.z; ++bz)
      for (int by = 0; by < isz.y; ++by)
        for (int bx = 0; bx < isz.x; ++bx)
          loc[t++] = host[(size_t)(bx + g) + (size_t)(by + g) * e.x + (size_t)(bz + g) * e.x * e.y];

    const auto& gs = mac_.dec.globalSize();
    std::vector<double> global;
    if (mac_.rank != root) {
      MPI_Send(loc.data(), (int)loc.size(), MPI_DOUBLE, root, 901, comm_);
      return global;
    }
    global.assign((size_t)gs[0] * gs[1] * gs[2], 0.0);
    std::vector<double> buf;
    for (int r = 0; r < mac_.size; ++r) {
      auto o = mac_.dec.origins()[r];
      auto s = mac_.dec.sizes()[r];
      size_t cnt = (size_t)s[0] * s[1] * s[2];
      const double* src = loc.data();
      if (r != root) {
        buf.resize(cnt);
        MPI_Recv(buf.data(), (int)cnt, MPI_DOUBLE, r, 901, comm_, MPI_STATUS_IGNORE);
        src = buf.data();
      }
      size_t t2 = 0;
      for (long bz = 0; bz < s[2]; ++bz)
        for (long by = 0; by < s[1]; ++by)
          for (long bx = 0; bx < s[0]; ++bx) {
            long gx = o[0] + bx, gy = o[1] + by, gz = o[2] + bz;
            global[(size_t)gx + (size_t)gy * gs[0] + (size_t)gz * gs[0] * gs[1]] = src[t2++];
          }
    }
    return global;
  }

  // Advance one timestep: implicit diffusion (+body force, +no-slip) for each component, then
  // projection. n_diff GS iterations per component, n_pois for the pressure Poisson.
  void step(int n_diff, int n_pois) {
    using namespace detail;
    double beta = nu_ * dt_, Ac = 1.0 + 6.0 * beta;
    double* comp[3] = {u_, v_, w_};
    double f[3] = {fx_, fy_, fz_};
    int g = mac_.ghost;

    // Build the per-component diffusion RHS. With advection on, all three use the SAME n-level
    // velocity (u,v,w), so compute every RHS before any component is updated.
    if (advection_) {
      mac_.exchange(u_);
      mac_.exchange(v_);
      mac_.exchange(w_);
      for (int c = 0; c < 3; ++c)
        advect_rhs_k<<<grd_, blk_>>>(c, u_, v_, w_, comp[c], b_[c], dt_, f[c], ext_, g);
    } else {
      for (int c = 0; c < 3; ++c) rhs_k<<<grd_, blk_>>>(comp[c], b_[c], dt_ * f[c], ext_, g);
    }

    for (int c = 0; c < 3; ++c) {
      for (int it = 0; it < n_diff; ++it) {
        mac_.exchange(comp[c]);
        diff_k<<<grd_, blk_>>>(comp[c], b_[c], ext_, mac_.origin_incl_ghost, g, beta, Ac, 0);
        if (has_solid_) apply_mask(comp[c]);
        mac_.exchange(comp[c]);
        diff_k<<<grd_, blk_>>>(comp[c], b_[c], ext_, mac_.origin_incl_ghost, g, beta, Ac, 1);
        if (has_solid_) apply_mask(comp[c]);
      }
    }
    // projection
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    diverg_k<<<grd_, blk_>>>(u_, v_, w_, div_, ext_, g);
    if (mg_enabled_) {
      // multigrid pressure solve: n_pois V-cycles of A phi = -div (same periodic Laplacian as pois_k)
      ensure_mg_built();
      cfdmpi::MGLevel& l0 = mg_.level(0);  // ghost-2 layout identical to this solver's blocks
      int threads = 256, blocks = (int)((n_ + threads - 1) / threads);
      neg_k<<<blocks, threads>>>(l0.rhs, div_, (long)n_);
      cudaMemset(l0.x, 0, n_ * 8);
      mg_.solve(n_pois, mg_pre_, mg_post_, mg_bottom_);
      cudaMemcpy(phi_, l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(phi_, 0, n_ * 8);
      for (int it = 0; it < n_pois; ++it) {
        mac_.exchange(phi_);
        pois_k<<<grd_, blk_>>>(phi_, div_, ext_, mac_.origin_incl_ghost, g, 0);
        mac_.exchange(phi_);
        pois_k<<<grd_, blk_>>>(phi_, div_, ext_, mac_.origin_incl_ghost, g, 1);
      }
    }
    mac_.exchange(phi_);
    correct_k<<<grd_, blk_>>>(u_, v_, w_, phi_, ext_, g);
    if (has_solid_) { apply_mask(u_); apply_mask(v_); apply_mask(w_); }
  }

 private:
  void apply_mask(double* c) { detail::mask_k<<<grd_, blk_>>>(c, solid_, ext_); }
  void ensure_mg_built() {
    if (mg_built_) return;
    // same ORB decomposition + ghost width (2) as this solver, so MG level-0 blocks share the layout.
    mg_.init(mac_.global_res, mac_.rank, mac_.size, /*h0=*/1.0, mg_levels_, comm_, /*ghost=*/mac_.ghost);
    mg_built_ = true;
  }

  MacGridHalo mac_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  double nu_ = 0, dt_ = 0, fx_ = 0, fy_ = 0, fz_ = 0;
  bool has_solid_ = false;
  bool advection_ = false;
  std::size_t n_ = 0;
  int3 ext_{}, origin_{};
  dim3 blk_, grd_;
  double *u_ = nullptr, *v_ = nullptr, *w_ = nullptr, *phi_ = nullptr, *div_ = nullptr,
         *solid_ = nullptr;
  double* b_[3] = {nullptr, nullptr, nullptr};
  // opt-in multigrid pressure solve (built lazily on first step)
  cfdmpi::DistributedPoissonMG mg_;
  bool mg_enabled_ = false, mg_built_ = false;
  int mg_levels_ = 4, mg_pre_ = 2, mg_post_ = 2, mg_bottom_ = 12;
};

}  // namespace dstokes
