/// @file
/// @brief DistributedNS: the distributed (MPI-optional) cut-cell Navier-Stokes solver (sdflow core).
// cfd-gpu — DistributedNS: a reusable distributed incompressible Navier–Stokes solver.
//
// Consolidates the building blocks validated in tests/test_*_mpi.cu into one component: a staggered
// MAC velocity field decomposed into rank-owned blocks (transport-core ORB), advanced each step by
// per-component implicit diffusion (backward Euler, Red-Black Gauss-Seidel with halo exchange between
// sweeps) + Chorin projection. Full Navier–Stokes: optional nonlinear Koren TVD advection (explicit or
// implicit-FOU deferred correction, set_advection / set_implicit_advection), Picard outer iterations,
// body force, and SDF solids via the Robust-Scaled cut-cell IBM (or simple masking). The pressure solve
// is single-level RB-GS, geometric/Galerkin multigrid, or CG; cut-cell pressure operator optional.
// Periodic. Host-staged GPU halo (see ../transport-core). Full status: doc/mpi_parallelization_status.md.
//
// (Historically named DistributedStokes when it was Stokes-only; now a full NS solver.)
#pragma once

#include "tpx/common/mpi.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "cfd_solver.cuh"  // get_idx (unused at runtime here; kept for convention parity)
#include "mac_cutcell.cuh"    // cut-cell face openness from an SDF
#include "mac_halo.cuh"
#include "mac_ibm.cuh"        // Robust-Scaled velocity IBM (cut-cell no-slip)
#include "mac_bc.cuh"         // native domain boundary conditions (velocity ghosts + wall openness)
#include "mac_multigrid.cuh"  // DistributedPoissonMG (opt-in multigrid pressure solve)
#include "staggered_advection.cuh"

namespace dns {

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
// implicit-FOU deferred correction: build the velocity stencil = backward-Euler diffusion
// (I - nu*dt*Lap) + dt*FOU_advection(u^k). The FOU upwind terms add to the diagonal -> diagonally
// dominant -> stable at high Re. Built per Picard iteration (advecting velocity u,v,w frozen at u^k).
template <int COMP>
__global__ void build_adv_stencil_k(cfdmpi::mreal* A_C, cfdmpi::mreal* A_W, cfdmpi::mreal* A_E,
                                    cfdmpi::mreal* A_S, cfdmpi::mreal* A_N, cfdmpi::mreal* A_B,
                                    cfdmpi::mreal* A_T, const double* u, const double* v,
                                    const double* w, int3 e, int g, double beta, double dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e);
  // assemble in double (the advecting velocity, diffusion and FOU upwind terms), store float.
  double cC = 1.0 + 6.0 * beta, cxm = -beta, cxp = -beta, cym = -beta, cyp = -beta, czm = -beta,
         czp = -beta;
  sadv::fou_operator(COMP, x, y, z, sadv::LocAcc{u, e}, sadv::LocAcc{v, e}, sadv::LocAcc{w, e}, dt, cC,
                     cxm, cxp, cym, cyp, czm, czp);
  A_C[i] = (cfdmpi::mreal)cC; A_W[i] = (cfdmpi::mreal)cxm; A_E[i] = (cfdmpi::mreal)cxp;
  A_S[i] = (cfdmpi::mreal)cym; A_N[i] = (cfdmpi::mreal)cyp; A_B[i] = (cfdmpi::mreal)czm;
  A_T[i] = (cfdmpi::mreal)czp;
}
// Velocity-multigrid COARSE-level operator for the implicit-FOU (upwind-convective) path: an anisotropic
// constant-coefficient backward-Euler diffusion (per-axis beta bx,by,bz) PLUS a first-order-upwind
// advection built from the RESTRICTED coarse advecting velocity (u,v,w on the coarse level). It mirrors
// build_adv_stencil_k (the fine As_[c]) but at coarse spacing: bx=nu_dt/h_x^2 and the FOU velocity is
// scaled by s_a=1/h_a per face axis (h_a = h0*cfac_a). The fine residual+smoother give the exact fine
// (sharp-IBM) answer; this coarse op only sets the convergence rate, and upwinding keeps it an M-matrix.
// Launched over INNER cells (smoother/residual read AC..AT only there); the +-1 reach hits exchanged ghosts.
template <int COMP>
__global__ void build_adv_coarse_stencil_k(cfdmpi::mreal* A_C, cfdmpi::mreal* A_W, cfdmpi::mreal* A_E,
                                           cfdmpi::mreal* A_S, cfdmpi::mreal* A_N, cfdmpi::mreal* A_B,
                                           cfdmpi::mreal* A_T, const double* u, const double* v,
                                           const double* w, int3 e, int g, double bx, double by,
                                           double bz, double dt, double sx, double sy, double sz) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e);
  double cC = 1.0 + 2.0 * (bx + by + bz), cxm = -bx, cxp = -bx, cym = -by, cyp = -by, czm = -bz,
         czp = -bz;
  sadv::fou_operator_aniso(COMP, x, y, z, sadv::LocAcc{u, e}, sadv::LocAcc{v, e}, sadv::LocAcc{w, e}, dt,
                           sx, sy, sz, cC, cxm, cxp, cym, cyp, czm, czp);
  A_C[i] = (cfdmpi::mreal)cC; A_W[i] = (cfdmpi::mreal)cxm; A_E[i] = (cfdmpi::mreal)cxp;
  A_S[i] = (cfdmpi::mreal)cym; A_N[i] = (cfdmpi::mreal)cyp; A_B[i] = (cfdmpi::mreal)czm;
  A_T[i] = (cfdmpi::mreal)czp;
}
// add the explicit FOU term to the RHS: b += dt*FOU(u^k) (so b = u^n + dt*f - dt*(Koren - FOU))
template <int COMP>
__global__ void add_fou_rhs_k(double* b, const double* u, const double* v, const double* w,
                              const double* phi, int3 e, int g, double dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e);
  b[i] += dt * sadv::advect_fou(COMP, x, y, z, sadv::LocAcc{u, e}, sadv::LocAcc{v, e},
                                sadv::LocAcc{w, e}, sadv::LocAcc{phi, e});
}

// one Red-Black diffusion sweep: c <- (b + beta*sum_nbr c)/(1+6 beta), global parity.
__global__ void diff_k(double* c, const double* b, int3 e, int3 og, int g, double beta, double Ac,
                       int color, const double* dcorr = nullptr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double s = c[i + sx] + c[i - sx] + c[i + sy] + c[i - sy] + c[i + sz] + c[i - sz];
  // dcorr (domain-BC face-fold): a wall face is dropped (its ghost is 0 in s) and its beta is added to
  // the diagonal here, so the implicit solve never reads a lagged wall ghost.
  c[i] = (b[i] + beta * s) / (Ac + (dcorr ? dcorr[i] : 0.0));
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
// cut-cell flux divergence: open-face-weighted, consistent with the cut-cell operator
// A = -div(open grad). ox[i] is the -x face openness of cell i (== +x face of cell i-1).
__global__ void diverg_open_k(const double* u, const double* v, const double* w, const double* ox,
                              const double* oy, const double* oz, double* d, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  d[i] = (ox[i + sx] * u[i + sx] - ox[i] * u[i]) + (oy[i + sy] * v[i + sy] - oy[i] * v[i]) +
         (oz[i + sz] * w[i + sz] - oz[i] * w[i]);
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
// Incremental pressure correction: subtract the accumulated-potential gradient from the momentum RHS,
// b_c -= dPhi/dx_c (same staggered stencil as correct_k), so the predictor carries -grad(p^n)/rho.
__global__ void sub_gradpot_k(double* bu, double* bv, double* bw, const double* P, int3 e, int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
  bu[i] -= P[i] - P[i - sx];
  bv[i] -= P[i] - P[i - sy];
  bw[i] -= P[i] - P[i - sz];
}
// Rotational incremental update of the accumulated potential Phi (= dt/rho * p):
//   Phi += phi - nu*dt*div(u*)   <=>   p += (rho/dt)phi - mu*div(u*)   (Timmermans rotational form).
// Inner cells. Solid cells stay ~0 (phi and div are ~0 there), so grad(Phi) at solid faces is consistent
// without a special mask. At steady state div->0 and phi->0, so Phi (the pressure) stabilises exactly.
__global__ void pot_update_k(double* P, const double* phi, const double* div, double nudt, int3 e,
                             int g) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y,
      z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < g || y < g || z < g || x >= e.x - g || y >= e.y - g || z >= e.z - g) return;
  long i = L3(x, y, z, e);
  P[i] += phi[i] - nudt * div[i];
}
}  // namespace detail

/// @brief Distributed (MPI-optional) cut-cell IBM incompressible Navier-Stokes solver — the `sdflow` core.
///
/// Solves the incompressible Navier-Stokes equations on a staggered MAC grid decomposed into per-rank
/// blocks (transport-core ORB + asynchronous ghost halo). The solid is described by an SDF; a Robust-Scaled
/// cut-cell Immersed Boundary Method imposes no-slip / moving-wall conditions and a matching cut-cell
/// pressure operator. Each step() advances one time step: a (semi-)implicit diffusion solve, optional
/// advection (Koren TVD, explicit or implicit deferred-correction), and a pressure projection whose Poisson
/// system is solved by the geometric multigrid (::cfdmpi::DistributedPoissonMG) with MG-PCG or Chebyshev
/// acceleration. Supports native per-face domain boundary conditions (periodic / wall / inflow / outflow)
/// and per-position inlet profiles. Lengths are in grid units (dx = 1); set nu and dt at construction.
class DistributedNS {
 public:
  void init(int3 global_res, int rank, int size, double nu, double dt,
            MPI_Comm comm = MPI_COMM_WORLD) {
    // Ghost width 2 covers the Koren advection reach (diffusion/projection use only width 1). periodic_
    // is {true,true,true} unless set_domain_bc made an axis non-periodic (the halo then leaves the
    // physical-boundary ghosts for apply_velocity_bc to fill -- transport-core grid_halo.hpp:78).
    mac_.init(global_res, rank, size, periodic_, /*ghost_width=*/2, comm);
    comm_ = comm;
    nu_ = nu;
    dt_ = dt;
    n_ = mac_.num_local_cells();
    for (double** p : {&u_, &v_, &w_, &phi_, &div_, &solid_, &phitot_, &b_[0], &b_[1], &b_[2], &u_old_,
                       &v_old_, &w_old_, &up_, &vp_, &wp_})
      cudaMalloc(p, n_ * sizeof(double));
    cudaMemset(u_, 0, n_ * 8);
    cudaMemset(v_, 0, n_ * 8);
    cudaMemset(w_, 0, n_ * 8);
    cudaMemset(solid_, 0, n_ * 8);
    cudaMemset(phitot_, 0, n_ * 8);  // accumulated pressure potential starts at 0
    cudaMemset(phi_, 0, n_ * 8);     // projection potential (also the pressure-solve warm-start seed)
    ext_ = mac_.local_ext;
    blk_ = dim3(8, 8, 4);
    grd_ = dim3((ext_.x + 7) / 8, (ext_.y + 7) / 8, (ext_.z + 3) / 4);
    // per-component streams + exchange engines for the concurrent velocity solve (share mac_'s topology)
    for (int c = 0; c < 3; ++c) {
      cudaStreamCreate(&vstreams_[c]);
      vexch_[c].init(mac_.halo);
    }
    cudaEventCreateWithFlags(&vrhs_evt_, cudaEventDisableTiming);
    vstreams_init_ = true;
  }
  ~DistributedNS() {
    for (double* p : {u_, v_, w_, phi_, div_, solid_, phitot_, b_[0], b_[1], b_[2], ox_, oy_, oz_,
                      bx_, by_, bz_, u_old_, v_old_, w_old_, up_, vp_, wp_})
      if (p) cudaFree(p);
    for (int c = 0; c < 3; ++c) {
      if (idmap_[c]) cudaFree(idmap_[c]);
      if (inhom_[c]) cudaFree(inhom_[c]);
      if (solidmask_[c]) cudaFree(solidmask_[c]);
      if (descale_[c]) cudaFree(descale_[c]);
      if (bc_dcorr_[c]) cudaFree(bc_dcorr_[c]);
      if (bc_brhs_[c]) cudaFree(bc_brhs_[c]);
      for (int k = 0; k < 7; ++k)
        if (As_[c][k]) cudaFree(As_[c][k]);
      if (ibmdata_[c].cell_index) cfdmpi::ibm_free(ibmdata_[c]);
    }
    for (double* p : bc_prof_) if (p) cudaFree(p);
    for (auto& v : {&vadv_u_, &vadv_v_, &vadv_w_})
      for (double* p : *v)
        if (p) cudaFree(p);
    for (double* p : vfine_) if (p) cudaFree(p);
    for (double* p : vresmask_) if (p) cudaFree(p);
    for (double* p : vtheta_lvl_) if (p) cudaFree(p);
    if (mg_built_) mg_.free();
    if (vmg_built_) vmg_.free();
    if (vstreams_init_) {
      for (int c = 0; c < 3; ++c)
        if (vstreams_[c]) cudaStreamDestroy(vstreams_[c]);
      if (vrhs_evt_) cudaEventDestroy(vrhs_evt_);
    }
  }
  DistributedNS() = default;
  DistributedNS(const DistributedNS&) = delete;
  DistributedNS& operator=(const DistributedNS&) = delete;

  const MacGridHalo& halo() const { return mac_; }
  int ghost() const { return mac_.ghost; }
  int3 ext() const { return ext_; }
  int3 origin_incl_ghost() const { return mac_.origin_incl_ghost; }
  std::size_t num_cells() const { return n_; }
  double* u() { return u_; }
  double* v() { return v_; }
  double* w() { return w_; }
  double* phi() { return phi_; }  // projection potential; pressure p = rho/dt * phi (Chorin)
  // The field to scale by rho/dt for the physical pressure: the accumulated potential under the
  // incremental scheme, else the per-step Chorin projection potential.
  double* pressure_potential() { return incremental_ ? phitot_ : phi_; }

  // Run the independent u/v/w IBM RB-GS momentum solves on 3 concurrent CUDA streams (default on).
  // Overlaps at small per-component sizes where one stencil does not saturate the GPU; ~no effect once
  // saturated. Only affects the cut-cell IBM RB-GS velocity path (not the velocity-multigrid path).
  void set_velocity_streams(bool on) { vstreams_enabled_ = on; }

  void set_body_force(double fx, double fy, double fz) { fx_ = fx; fy_ = fy; fz_ = fz; }
  // Enable explicit nonlinear advection (full Navier-Stokes instead of Stokes).
  void set_advection(bool on) { advection_ = on; }

  // Incremental-rotational pressure correction (vs classical non-incremental Chorin). The momentum
  // predictor carries -grad(p^n) and the accumulated potential Phi is updated by the rotational form
  // Phi += phi - nu*dt*div(u*). Same steady velocity; more accurate transient + near-wall pressure
  // (no Chorin splitting boundary layer). Default off here (keeps the cell-for-cell Chorin tests exact);
  // the sdflow Python module turns it on by default.
  void set_incremental_pressure(bool on) { incremental_ = on; }

  // Implicit-FOU deferred-correction advection (requires the IBM operator). The first-order-upwind part
  // of advection is solved implicitly (added to the momentum stencil -> diagonally dominant), and the
  // (Koren - FOU) correction is explicit, so the scheme stays Koren TVD at convergence but is robust at
  // high Reynolds number (no longer CFL-limited like the fully-explicit path). Velocity solve is RB-GS
  // (the stencil changes each Picard iteration); n_diff counts RB-GS sweeps.
  void set_implicit_advection(bool on) { implicit_fou_ = on; }

  // Picard (defect-correction) outer iterations within each timestep: each iteration re-lags the
  // advection at the latest velocity and re-projects, converging the nonlinear advection/pressure
  // coupling that a single fractional step leaves. tol>0 stops early when the max velocity change over
  // an outer iteration drops below it. iters=1 (default) reproduces the single-pass scheme exactly.
  void set_outer_iterations(int iters) { outer_iters_ = iters < 1 ? 1 : iters; }
  void set_outer_tolerance(double tol) { outer_tol_ = tol; }
  int last_outer_iterations() const { return last_outer_iterations_; }
  double last_outer_correction() const { return last_outer_correction_; }

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

  // Install the real cut-cell pressure operator on the multigrid fine level from an SDF (extended-block
  // layout, all cells incl. ghosts, negative inside solid -- a function of global position, like
  // set_solid). The staggered face openness (cfd's gradient-normalised fluid fraction, masked) replaces
  // the constant-coefficient Laplacian on level 0 (coarse levels stay constant-coefficient, mirroring
  // the serial multigrid). Enables and builds the multigrid; call after init(). Spacing is unit (dx=1),
  // matching diverg_k/correct_k.
  // coarse_mode selects how the multigrid coarse-level operators are built:
  //   0 = REDISCRETIZED (default, recommended): average-coarsen the face openness and re-assemble the
  //       cut-cell operator per level -- a genuine, consistent discretization on every grid. Grid-
  //       independent V-cycle (rho ~0.15 flat in N); the correct + efficient choice.
  //   1 = GALERKIN: variational (unsmoothed-aggregation) coarse operators. Inconsistent for the cut-cell
  //       system (V-cycle rho -> 1 with N); kept only for comparison. Do not use for production.
  //   2 = CONST: geometry-blind constant-coefficient coarse levels. Correct but a weak coarse model.
  void set_cutcell_pressure_operator(const std::vector<double>& sdf_ext, int coarse_mode = 0) {
    mg_enabled_ = true;
    ensure_mg_built();
    if (!ox_) for (double** p : {&ox_, &oy_, &oz_}) cudaMalloc(p, n_ * sizeof(double));
    double* sdf = nullptr;
    cudaMalloc(&sdf, n_ * sizeof(double));
    cudaMemcpy(sdf, sdf_ext.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
    dim3 gE((ext_.x + 7) / 8, (ext_.y + 7) / 8, (ext_.z + 3) / 4);
    // OPERATOR openness alpha (ox_/oy_/oz_): walls/inflow zeroed (Neumann), outflow kept open (Dirichlet).
    cfdmpi::ccdetail::cc_build_open_k<<<gE, dim3(8, 8, 4)>>>(ox_, oy_, oz_, sdf, ext_, 1.0, 1.0, 1.0);
    apply_wall_openness();
    if (has_open_boundary_) {
      // FLUX openness beta (bx_/by_/bz_): pure geometry, NOT zeroed at domain faces, so the divergence and
      // projection count the inflow/outflow flux even where the operator is Neumann. Solids still close it.
      if (!bx_) for (double** p : {&bx_, &by_, &bz_}) cudaMalloc(p, n_ * sizeof(double));
      cfdmpi::ccdetail::cc_build_open_k<<<gE, dim3(8, 8, 4)>>>(bx_, by_, bz_, sdf, ext_, 1.0, 1.0, 1.0);
      mg_.setRemoveMean(false);  // the Dirichlet outflow face makes the operator non-singular
    }
    // tell the MG the per-face boundary types so the coarse rediscretized operators + the trilinear
    // prolongation get the right non-periodic treatment per level (no-op if all-periodic).
    if (has_domain_bc_) mg_.setBoundaryConditions(bc_type_);
    if (coarse_mode == 0)
      mg_.setFineVariableOperatorRediscretized(ox_, oy_, oz_, 1.0, 1.0, 1.0);
    else
      mg_.setFineVariableOperator(ox_, oy_, oz_, 1.0, 1.0, 1.0, /*galerkin=*/coarse_mode == 1);
    cudaFree(sdf);
    cutcell_ = true;
  }
  // flux-openness pointers for the divergence/correction: beta when an open boundary exists, else alpha.
  double* fox() { return has_open_boundary_ ? bx_ : ox_; }
  double* foy() { return has_open_boundary_ ? by_ : oy_; }
  double* foz() { return has_open_boundary_ ? bz_ : oz_; }

  // Solve the pressure Poisson with CG preconditioned by the multigrid V-cycle instead of plain V-cycles.
  // Converges the stiff cut-cell operator in fewer iterations; requires the cut-cell operator
  // (set_cutcell_pressure_operator). step()'s n_pois is then ignored in favour of max_iter/rtol. The core
  // default is OFF (standalone V-cycles); the sdflow module enables it by default on a single rank.
  void set_pressure_pcg(bool on, int max_iter = 60, double rtol = 1e-8) {
    pcg_ = on;
    if (on) cheb_ = false;  // mutually exclusive outer accelerators
    pcg_maxit_ = max_iter;
    pcg_rtol_ = rtol;
  }

  // Solve the pressure Poisson with Chebyshev semi-iteration accelerating the multigrid V-cycle, instead
  // of CG (set_pressure_pcg) or plain V-cycles. Same convergence as MG-PCG (~matches its iteration count)
  // but with NO per-iteration global dot-products -- communication-light, intended for large multi-GPU
  // where PCG's reductions are latency-bound. The spectral bounds of M^{-1}A are estimated once (lazily,
  // on the first step). Requires the cut-cell operator. Overrides PCG when both are set.
  void set_pressure_chebyshev(bool on, int max_iter = 60, double rtol = 1e-8) {
    cheb_ = on;
    if (on) pcg_ = false;
    cheb_maxit_ = max_iter;
    cheb_rtol_ = rtol;
    cheb_bounds_set_ = false;  // re-estimate (e.g. if solver settings changed)
  }

  // Warm-start the pressure solve from the previous step's projection potential (opt-in; default off).
  // A steady-march optimization: consecutive phi's are similar, so the previous phi is a good initial
  // guess (fewer PCG iters / a more-converged phi per fixed sweep count). Off = cold start (zero guess),
  // the bit-exact behaviour the cell-for-cell serial-reference tests assume.
  void set_pressure_warmstart(bool on) { pwarm_ = on; }

  // Solve the IBM velocity diffusion with geometric multigrid (constant-coefficient coarse operators)
  // instead of plain RB-GS -- far fewer iterations when the diffusion is stiff (large nu*dt). Requires
  // the IBM operator (set_ibm_solid). n_diff then counts V-cycles.
  void set_velocity_multigrid(bool on, int n_levels = 3, int v_cycles = 4) {
    vmg_enabled_ = on;
    vmg_levels_ = n_levels;
    vmg_vcycles_ = v_cycles;
  }

  // Use the geometry-aware VOLUME-FRACTION coarse operator (smoothed momentum balance) + masked,
  // volume-weighted transfers for the IBM velocity-MG, instead of the geometry-blind const-coeff coarse.
  // Opt-in (default off = const-coeff, byte-identical). eps = volume fraction below which a COARSE cell is
  // treated as solid (small-cell fix, safe on coarse levels only). Requires set_ibm_solid + vmg enabled.
  void set_velocity_mg_volfrac(bool on, double eps = 0.1, bool mask_xfer = false, bool res_mask = true) {
    vmg_volfrac_ = on;
    vmg_eps_ = eps;
    vmg_mask_xfer_ = mask_xfer;
    vmg_res_mask_ = res_mask;  // zero cut-cell residual before restriction (only the coarsenable region)
  }

  // Robust-Scaled cut-cell IBM no-slip for the momentum solve, from an SDF (extended-block layout, all
  // cells incl. ghosts, negative inside solid). Builds the per-component (u/v/w) IBM geometry and bakes
  // it into a static backward-Euler diffusion stencil + inhomogeneous Dirichlet term (u_bc = wall
  // velocity, default no-slip). The diffusion sweeps then use the modified stencil instead of the
  // constant-coefficient operator + velocity masking -- the accurate cut-cell boundary treatment.
  // Static dt/nu (the stencil bakes beta = nu*dt). Call after init().
  void set_ibm_solid(const std::vector<double>& sdf_ext, float3 u_bc = make_float3(0, 0, 0)) {
    using namespace cfdmpi::ibmdetail;
    double beta = nu_ * dt_;
    dim3 blk(8, 8, 8);
    dim3 gE((ext_.x + 7) / 8, (ext_.y + 7) / 8, (ext_.z + 3) / 4);
    float3 spacing = make_float3(1, 1, 1);
    float3 offs[3] = {make_float3(-0.5f, 0, 0), make_float3(0, -0.5f, 0), make_float3(0, 0, -0.5f)};
    float ubc[3] = {u_bc.x, u_bc.y, u_bc.z};
    ubc_ibm_ = u_bc;

    double* sdf = nullptr;
    cudaMalloc(&sdf, n_ * 8);
    cudaMemcpy(sdf, sdf_ext.data(), n_ * 8, cudaMemcpyHostToDevice);
    int* counter = nullptr;
    cudaMalloc(&counter, sizeof(int));

    for (int c = 0; c < 3; ++c) {
      // GeometryProvider: geometry -> overlay (the one call site an octree port replaces).
      build_ibm_overlay(c, sdf, offs[c], spacing, counter);

      // base operator (face loop) + overlay apply (mesh-agnostic):
      if (!As_[c][0])
        for (int k = 0; k < 7; ++k) cudaMalloc(&As_[c][k], n_ * sizeof(cfdmpi::mreal));
      if (!inhom_[c]) cudaMalloc(&inhom_[c], n_ * 8);
      if (!solidmask_[c]) cudaMalloc(&solidmask_[c], n_ * 8);
      if (!descale_[c]) cudaMalloc(&descale_[c], n_ * 8);
      ibm_solid_mask_k<<<gE, blk>>>(solidmask_[c], sdf, ext_, offs[c]);
      // fluid volume fraction theta (for the opt-in volume-fraction velocity-MG coarse operator); static.
      if (!vfine_[c]) cudaMalloc(&vfine_[c], n_ * 8);
      ibm_volfrac_k<<<gE, blk>>>(vfine_[c], sdf, ext_, offs[c]);
      mac_.exchange(vfine_[c]);  // periodic ghost theta (the coarse build + restriction read neighbours)
      // coarse-coupling mask: 1 only at clean fluid interior; 0 at IBM cut cells + solid (the coarse grid
      // couples only where its clean operator matches the fine one -> no overshoot at the boundary band/solid)
      if (!vresmask_[c]) cudaMalloc(&vresmask_[c], n_ * 8);
      ibm_clean_fluid_mask_k<<<gE, blk>>>(vresmask_[c], sdf, ext_, offs[c]);
      ibm_build_diffusion_k<<<gE, blk>>>(As_[c][0], As_[c][1], As_[c][2], As_[c][3], As_[c][4],
                                         As_[c][5], As_[c][6], ext_, beta);
      cudaMemset(inhom_[c], 0, n_ * 8);
      int t1 = 256, b1 = (int)((n_ + t1 - 1) / t1);
      ibm_fill_k<<<b1, t1>>>(descale_[c], 1.0, (long)n_);  // Robust-Scaled RHS scale (1 outside cut cells)
      if (ibmdata_[c].num_active_cells > 0)
        ibm_modify_stencil_k<<<(ibmdata_[c].num_active_cells + 255) / 256, 256>>>(
            As_[c][0], As_[c][1], As_[c][2], As_[c][3], As_[c][4], As_[c][5], As_[c][6], inhom_[c],
            descale_[c], ibmdata_[c], ubc[c]);
    }
    cudaFree(sdf);
    cudaFree(counter);
    ibm_enabled_ = true;
  }

  // Set a domain-boundary condition on one of the 6 faces (0=-x,1=+x,2=-y,3=+y,4=-z,5=+z). type:
  // 0=PERIODIC (default), 1=NOSLIP wall, 2=DIRICHLET velocity / INFLOW (vel), 3=OUTFLOW (zero-gradient
  // velocity + Dirichlet p=0). Call BEFORE the first step / set_solid -- it fixes the halo periodicity (an
  // axis becomes non-periodic if either of its faces is non-periodic). Velocity ghosts on these faces are
  // filled by apply_velocity_bc; the pressure operator gets Neumann at walls/inflow (zeroed face openness)
  // and Dirichlet at outflow (open face + ghost p=0). An open boundary (outflow, or an inflow with a
  // non-zero normal velocity) splits the openness: see set_cutcell_pressure_operator.
  void set_domain_bc(int face, int type, float3 vel = make_float3(0, 0, 0)) {
    bc_type_[face] = type;
    bc_vel_[face] = vel;
    int ax = face / 2;
    periodic_[ax] = (bc_type_[2 * ax] == 0 && bc_type_[2 * ax + 1] == 0);
    has_domain_bc_ = !(periodic_[0] && periodic_[1] && periodic_[2]);
    // an open boundary makes the pressure operator differ from the flux divergence: outflow (Dirichlet), or
    // an inflow (Dirichlet velocity) whose NORMAL component carries mass across the boundary.
    has_open_boundary_ = false;
    for (int fc = 0; fc < 6; ++fc) {
      int a = fc / 2;
      float3 v = bc_vel_[fc];
      double vn = a == 0 ? v.x : (a == 1 ? v.y : v.z);
      if (bc_type_[fc] == 3 || (bc_type_[fc] == 2 && vn != 0.0)) has_open_boundary_ = true;
    }
    if (has_prof_[0] || has_prof_[1] || has_prof_[2] || has_prof_[3] || has_prof_[4] || has_prof_[5])
      has_open_boundary_ = true;
  }

  // Prescribe a per-(b,c)-position INLET PROFILE on a face (b,c = the two axes perpendicular to face/2).
  // prof is the global plane flattened [Nb][Nc][3] (Nb=global_res[b], Nc=global_res[c]); it sets the face
  // to inflow (type 2). Call BEFORE the first step. Use for a parabolic channel inlet or the partial inlet
  // of a backward-facing step (parabola over the open part, 0 over the step face).
  void set_domain_bc_profile(int face, const std::vector<double>& prof, int nb, int nc) {
    bc_prof_host_[face] = prof;
    bc_prof_nb_[face] = nb;
    bc_prof_nc_[face] = nc;
    has_prof_[face] = true;
    if (bc_type_[face] == 0) set_domain_bc(face, 2);  // a profile implies inflow (updates periodicity)
    bc_prof_built_ = false;
    has_open_boundary_ = true;
  }
  // Build each face's local device profile (sized to this rank's extended perp-plane) from the stored
  // global plane; ghost rows clamp to the edge. Lazy (needs ext_/origin from init).
  void ensure_bc_profiles() {
    if (bc_prof_built_) return;
    int3 og = mac_.origin_incl_ghost, N = mac_.global_res;
    int dims[3] = {ext_.x, ext_.y, ext_.z}, orig[3] = {og.x, og.y, og.z}, gres[3] = {N.x, N.y, N.z};
    for (int face = 0; face < 6; ++face) {
      if (!has_prof_[face]) continue;
      int a = face / 2, b = (a + 1) % 3, c = (a + 2) % 3;
      int Lb = dims[b], Lc = dims[c], Nb = bc_prof_nb_[face], Nc = bc_prof_nc_[face];
      (void)gres;
      std::vector<double> local((size_t)Lb * Lc * 3);
      for (int p0 = 0; p0 < Lb; ++p0)
        for (int p1 = 0; p1 < Lc; ++p1) {
          int gb = orig[b] + p0, gc = orig[c] + p1;
          gb = gb < 0 ? 0 : (gb >= Nb ? Nb - 1 : gb);  // clamp ghost rows to the plane edge
          gc = gc < 0 ? 0 : (gc >= Nc ? Nc - 1 : gc);
          for (int comp = 0; comp < 3; ++comp)
            local[((size_t)p0 * Lc + p1) * 3 + comp] =
                bc_prof_host_[face][((size_t)gb * Nc + gc) * 3 + comp];
        }
      if (!bc_prof_[face]) cudaMalloc(&bc_prof_[face], local.size() * 8);
      cudaMemcpy(bc_prof_[face], local.data(), local.size() * 8, cudaMemcpyHostToDevice);
    }
    bc_prof_built_ = true;
  }

  // Global max of the cut-cell flux divergence |div(open u)| over owned cells (the quantity the
  // cut-cell projection drives to zero). Uses div_ as scratch; call between steps.
  double max_open_divergence() {
    if (!cutcell_) return 0.0;
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    detail::diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, fox(), foy(), foz(), div_, ext_, mac_.ghost);
    return cfdmpi::mac_max_abs(div_, mac_, comm_);
  }
  // RMS of the cut-cell flux divergence (bulk measure; less sensitive to a single thin cut cell)
  double rms_open_divergence() {
    if (!cutcell_) return 0.0;
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    detail::diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, fox(), foy(), foz(), div_, ext_, mac_.ghost);
    double ss = cfdmpi::mac_dot(div_, div_, mac_, comm_);
    auto gs = mac_.dec.globalSize();
    double N = (double)gs[0] * gs[1] * gs[2];
    return std::sqrt(ss / N);
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
    double* old[3] = {u_old_, v_old_, w_old_};
    double* prev[3] = {up_, vp_, wp_};
    double f[3] = {fx_, fy_, fz_};
    int g = mac_.ghost;
    int threads = 256, blocks = (int)((n_ + threads - 1) / threads);

    // time-derivative base u^n (fixed for the whole timestep)
    for (int c = 0; c < 3; ++c) cudaMemcpy(old[c], comp[c], n_ * 8, cudaMemcpyDeviceToDevice);

    // incremental pressure: Phi^n is fixed for the whole step; refresh its ghosts once for the
    // predictor's -grad(Phi) term (re-used by every Picard iteration).
    if (incremental_) mac_.exchange(phitot_);

    int max_outer = outer_iters_;
    last_outer_iterations_ = 0;
    for (int outer = 0; outer < max_outer; ++outer) {
      last_outer_iterations_ = outer + 1;
      if (outer_tol_ > 0)
        for (int c = 0; c < 3; ++c) cudaMemcpy(prev[c], comp[c], n_ * 8, cudaMemcpyDeviceToDevice);

      // Per-component diffusion RHS b = u^n + dt*f - dt*advect(u^k): time-derivative from u^n, advection
      // lagged at the current iterate u^k. (Single-pass outer=0 has u^k == u^n -> identical to before.)
      if (advection_) {
        mac_.exchange(u_);
        mac_.exchange(v_);
        mac_.exchange(w_);
        apply_velocity_bc();  // explicit advection reads the reflection wall ghosts
        for (int c = 0; c < 3; ++c)
          advect_rhs_k<<<grd_, blk_>>>(c, u_, v_, w_, comp[c], b_[c], dt_, f[c], ext_, g);
        // retarget the time base from u^k to u^n:  b += (u^n - u^k)
        for (int c = 0; c < 3; ++c) {
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[c], 1.0, old[c], (long)n_);
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[c], -1.0, comp[c], (long)n_);
        }
        // Implicit-FOU deferred correction: b += dt*FOU(u^k) -> b = u^n + dt*f - dt*(Koren - FOU). This is
        // geometry-independent, needed whenever the velocity operator carries the implicit FOU term: the
        // IBM stencil As_[c] (rebuilt below) or the domain-BC vmg levels (built in the velocity solve). It
        // requires an operator that includes FOU, so only when ibm_enabled_ (As_) or vmg_enabled_ (levels).
        if (implicit_fou_ && (ibm_enabled_ || vmg_enabled_)) {
          add_fou_rhs_k<0><<<grd_, blk_>>>(b_[0], u_, v_, w_, u_, ext_, g, dt_);
          add_fou_rhs_k<1><<<grd_, blk_>>>(b_[1], u_, v_, w_, v_, ext_, g, dt_);
          add_fou_rhs_k<2><<<grd_, blk_>>>(b_[2], u_, v_, w_, w_, ext_, g, dt_);
        }
        if (implicit_fou_ && ibm_enabled_) {
          // rebuild the IBM velocity stencil = diffusion + dt*FOU(u^k), then re-apply the IBM bake
          // (descale_/inhom_). (The domain-BC FOU operator is built into the vmg levels, not here.)
          float ub[3] = {ubc_ibm_.x, ubc_ibm_.y, ubc_ibm_.z};
          for (int c = 0; c < 3; ++c) {
            cfdmpi::mreal* A[7] = {As_[c][0], As_[c][1], As_[c][2], As_[c][3],
                                   As_[c][4], As_[c][5], As_[c][6]};
            if (c == 0)
              build_adv_stencil_k<0><<<grd_, blk_>>>(A[0], A[1], A[2], A[3], A[4], A[5], A[6], u_, v_,
                                                     w_, ext_, g, beta, dt_);
            else if (c == 1)
              build_adv_stencil_k<1><<<grd_, blk_>>>(A[0], A[1], A[2], A[3], A[4], A[5], A[6], u_, v_,
                                                     w_, ext_, g, beta, dt_);
            else
              build_adv_stencil_k<2><<<grd_, blk_>>>(A[0], A[1], A[2], A[3], A[4], A[5], A[6], u_, v_,
                                                     w_, ext_, g, beta, dt_);
            cfdmpi::ibmdetail::ibm_fill_k<<<blocks, threads>>>(descale_[c], 1.0, (long)n_);
            cudaMemset(inhom_[c], 0, n_ * 8);
            if (ibmdata_[c].num_active_cells > 0)
              cfdmpi::ibmdetail::ibm_modify_stencil_k<<<(ibmdata_[c].num_active_cells + 255) / 256,
                                                        256>>>(A[0], A[1], A[2], A[3], A[4], A[5], A[6],
                                                               inhom_[c], descale_[c], ibmdata_[c],
                                                               ub[c]);
          }
        }
      } else {
        for (int c = 0; c < 3; ++c) rhs_k<<<grd_, blk_>>>(old[c], b_[c], dt_ * f[c], ext_, g);
      }

      // incremental predictor: b_c -= dPhi/dx_c  (carry -grad(p^n)/rho into the momentum RHS, before
      // the IBM RHS scaling so it is descaled like the other source terms).
      if (incremental_)
        sub_gradpot_k<<<grd_, blk_>>>(b_[0], b_[1], b_[2], phitot_, ext_, g);

    if (ibm_enabled_) {
      // Robust-Scaled cut-cell IBM: solve the IBM-modified stencil A_ibm comp = b - inhom (the
      // Dirichlet wall velocity is baked into inhom). No velocity masking -- the IBM eliminates the
      // solid ghost couplings.
      int ibthreads = 256, ibblocks = (int)((n_ + ibthreads - 1) / ibthreads);
      // Robust-Scaled RHS: b'_c = D_rescale * b_c - inhom (scale at cut cells, then the Dirichlet term)
      for (int c = 0; c < 3; ++c) {
        cfdmpi::ibmdetail::ibm_scale_k<<<ibblocks, ibthreads>>>(b_[c], descale_[c], (long)n_);
        cfdmpi::mgdetail::mg_axpy_k<<<ibblocks, ibthreads>>>(b_[c], -1.0, inhom_[c], (long)n_);
      }
      static int dbg = -1;
      if (dbg < 0) dbg = (std::getenv("SDFLOW_VMG_TRACE") ? 2 : 0);
      if (dbg > 0) { std::fprintf(stderr, "[VMG-DISPATCH] ibm=%d vmg_enabled=%d implicit_fou=%d vstreams=%d\n",
                                  (int)ibm_enabled_, (int)vmg_enabled_, (int)implicit_fou_, (int)vstreams_enabled_); dbg--; }
      if (vmg_enabled_ && !implicit_fou_) {  // diffusion-only vmg (advection explicit in the RHS)
        ensure_vmg_built();
        for (int c = 0; c < 3; ++c) {
          vmg_.setDiffusionFine(As_[c]);  // fine = this component's float IBM stencil (coarse built once)
          // opt-in: geometry-aware volume-fraction coarse op (smoothed momentum balance) + masked
          // volume-weighted transfers, rebuilt per component (static geometry, cheap). Level 0 stays As_[c]
          // -> the residual + smoother use the TRUE operator, so the fixed point is the exact sharp solution.
          if (vmg_volfrac_)
            vmg_.setVelocityVolfracCoarse(vfine_[c], vtheta_lvl_.data(), nu_ * dt_, 1.0, vmg_eps_,
                                          vmg_mask_xfer_, vmg_res_mask_ ? vresmask_[c] : nullptr);
          cfdmpi::MGLevel& l0 = vmg_.level(0);
          cudaMemcpy(l0.rhs, b_[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cudaMemcpy(l0.x, comp[c], n_ * 8, cudaMemcpyDeviceToDevice);  // initial guess
          vmg_.solve(vmg_vcycles_, vmg_pre_, vmg_post_, vmg_bottom_);  // (n_diff ignored under vmg)
          cudaMemcpy(comp[c], l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
          detail::mask_k<<<grd_, blk_>>>(comp[c], solidmask_[c], ext_);
        }
      } else if (vmg_enabled_ && implicit_fou_) {
        // Upwind-convective velocity-MG: the fine As_[c] (already rebuilt above = diffusion + dt*FOU(u^k) +
        // IBM bake) is the fine operator; the coarse operators are anisotropic const-coeff diffusion + a
        // coarse FOU from the restricted u^k. Rebuilt every Picard iteration since the advecting velocity
        // changes. Upwinding keeps every level an M-matrix, so RB-GS smoothing + coarse correction stay
        // stable in the advection-dominated rows; the fine residual guarantees the exact sharp-IBM answer.
        ensure_vmg_built();
        restrict_vmg_adv_velocities();
        for (int c = 0; c < 3; ++c) {
          build_vmg_adv_stencil(c, /*include_fine=*/false);  // coarse levels; level 0 = fine As_[c]
          vmg_.setDiffusionFine(As_[c]);
          cfdmpi::MGLevel& l0 = vmg_.level(0);
          cudaMemcpy(l0.rhs, b_[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cudaMemcpy(l0.x, comp[c], n_ * 8, cudaMemcpyDeviceToDevice);  // initial guess
          vmg_.solve(vmg_vcycles_, vmg_pre_, vmg_post_, vmg_bottom_);
          cudaMemcpy(comp[c], l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
          detail::mask_k<<<grd_, blk_>>>(comp[c], solidmask_[c], ext_);
        }
      } else if (vstreams_enabled_) {
        // 3-stream concurrent velocity solve: the u/v/w IBM RB-GS chains are independent -> one stream
        // each. exchangeOnStream has no device-wide sync, so the components overlap (at small sizes; the
        // GPU saturates at large). The RB-GS colours within a component stay ordered on its stream.
        cudaEventRecord(vrhs_evt_, 0);  // b_ + comp were built on the default stream
        for (int c = 0; c < 3; ++c) cudaStreamWaitEvent(vstreams_[c], vrhs_evt_, 0);
        for (int it = 0; it < n_diff; ++it)
          for (int color = 0; color < 2; ++color)
            for (int c = 0; c < 3; ++c) {
              vexch_[c].exchangeOnStream(comp[c], vstreams_[c], /*tag=*/c);
              cfdmpi::ibmdetail::ibm_rbgs_stencil_k<<<grd_, blk_, 0, vstreams_[c]>>>(
                  comp[c], b_[c], As_[c][0], As_[c][1], As_[c][2], As_[c][3], As_[c][4], As_[c][5],
                  As_[c][6], ext_, mac_.origin_incl_ghost, g, color);
            }
        for (int c = 0; c < 3; ++c)
          detail::mask_k<<<grd_, blk_, 0, vstreams_[c]>>>(comp[c], solidmask_[c], ext_);
        for (int c = 0; c < 3; ++c) cudaStreamSynchronize(vstreams_[c]);  // join before projection
      } else {
        for (int c = 0; c < 3; ++c) {
          for (int it = 0; it < n_diff; ++it) {
            for (int color = 0; color < 2; ++color) {
              mac_.exchange(comp[c]);
              cfdmpi::ibmdetail::ibm_rbgs_stencil_k<<<grd_, blk_>>>(
                  comp[c], b_[c], As_[c][0], As_[c][1], As_[c][2], As_[c][3], As_[c][4], As_[c][5],
                  As_[c][6], ext_, mac_.origin_incl_ghost, g, color);
            }
          }
          detail::mask_k<<<grd_, blk_>>>(comp[c], solidmask_[c], ext_);  // zero the decoupled solid
        }
      }
    } else {
      // Domain-BC walls are folded into the implicit diffusion: drop the wall ghosts (fold=1), add the
      // dropped face's beta to the diagonal (bc_dcorr_) and 2*beta*wall to the RHS (bc_brhs_). This keeps
      // u_inner implicit -- no one-sweep Gauss-Seidel lag on the wall term (the explicit advection above
      // still used the reflection ghost). setup_bc_diffusion() is a no-op once built / without domain BC.
      setup_bc_diffusion();
      if (has_domain_bc_)
        for (int c = 0; c < 3; ++c)
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[c], 1.0, bc_brhs_[c], (long)n_);
      if (vmg_enabled_ && !implicit_fou_) {
        // Velocity multigrid (const-coeff I - nu_dt*Lap + no-slip face-fold per component). The fold puts
        // +beta on the boundary diagonal and the wall ghost is held 0 (non-periodic halo skips it +
        // interior-only smoother), exactly the fine RB-GS representation; b_ already carries bc_brhs_.
        ensure_vmg_built();
        for (int c = 0; c < 3; ++c) {
          vmg_.setDiffusionConstAllLevels(nu_ * dt_, 1.0);     // component-independent base
          vmg_.setDiffusionBoundaryFold(c, bc_type_, nu_ * dt_, 1.0);  // component-dependent wall fold
          mac_.exchange(comp[c]);
          apply_velocity_bc_comp(comp[c], c, /*fold=*/1);      // zero wall ghosts + set normal Dirichlet face
          cfdmpi::MGLevel& l0 = vmg_.level(0);
          cudaMemcpy(l0.rhs, b_[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cudaMemcpy(l0.x, comp[c], n_ * 8, cudaMemcpyDeviceToDevice);
          vmg_.solve(vmg_vcycles_, vmg_pre_, vmg_post_, vmg_bottom_);
          cudaMemcpy(comp[c], l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
          if (has_solid_) apply_mask(comp[c]);
        }
      } else if (vmg_enabled_ && implicit_fou_) {
        // Upwind-convective velocity multigrid on the domain-BC path (cavity/BFS, no IBM stencil). The
        // operator on EVERY level is anisotropic const-coeff diffusion + dt*FOU(u^k) from the restricted
        // advecting velocity, with the no-slip/inflow/outflow wall fold applied on top (same fold as the
        // diffusion-only path; the fold acts on the diffusion wall term). b_ already carries bc_brhs_ +
        // the dt*FOU deferred correction. Upwinding keeps every level an M-matrix -> stable at high CFL.
        ensure_vmg_built();
        restrict_vmg_adv_velocities();
        for (int c = 0; c < 3; ++c) {
          build_vmg_adv_stencil(c, /*include_fine=*/true);            // diffusion + FOU on all levels
          vmg_.setDiffusionBoundaryFold(c, bc_type_, nu_ * dt_, 1.0);  // wall fold (per component)
          mac_.exchange(comp[c]);
          apply_velocity_bc_comp(comp[c], c, /*fold=*/1);
          cfdmpi::MGLevel& l0 = vmg_.level(0);
          cudaMemcpy(l0.rhs, b_[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cudaMemcpy(l0.x, comp[c], n_ * 8, cudaMemcpyDeviceToDevice);
          vmg_.solve(vmg_vcycles_, vmg_pre_, vmg_post_, vmg_bottom_);
          cudaMemcpy(comp[c], l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
          if (has_solid_) apply_mask(comp[c]);
        }
      } else
      for (int c = 0; c < 3; ++c) {
        const double* dcorr = has_domain_bc_ ? bc_dcorr_[c] : nullptr;
        for (int it = 0; it < n_diff; ++it) {
          mac_.exchange(comp[c]);
          apply_velocity_bc_comp(comp[c], c, /*fold=*/1);
          diff_k<<<grd_, blk_>>>(comp[c], b_[c], ext_, mac_.origin_incl_ghost, g, beta, Ac, 0, dcorr);
          if (has_solid_) apply_mask(comp[c]);
          mac_.exchange(comp[c]);
          apply_velocity_bc_comp(comp[c], c, /*fold=*/1);
          diff_k<<<grd_, blk_>>>(comp[c], b_[c], ext_, mac_.origin_incl_ghost, g, beta, Ac, 1, dcorr);
          if (has_solid_) apply_mask(comp[c]);
        }
      }
    }
    // projection (cut-cell operator -> open-weighted flux divergence RHS; else plain divergence)
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    apply_velocity_bc();  // domain BC: set boundary normal velocity + ghosts before the divergence
    if (cutcell_)
      diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, fox(), foy(), foz(), div_, ext_, g);
    else
      diverg_k<<<grd_, blk_>>>(u_, v_, w_, div_, ext_, g);
    if (mg_enabled_) {
      // multigrid pressure solve: n_pois V-cycles of A phi = -div (same periodic Laplacian as pois_k)
      ensure_mg_built();
      cfdmpi::MGLevel& l0 = mg_.level(0);  // ghost-2 layout identical to this solver's blocks
      int threads = 256, blocks = (int)((n_ + threads - 1) / threads);
      neg_k<<<blocks, threads>>>(l0.rhs, div_, (long)n_);
      // warm start: seed the solve with the previous step's projection potential (consecutive phi's are
      // similar along a steady march -> fewer PCG iters / a more-converged phi per fixed V-cycle count).
      if (pwarm_)
        cudaMemcpy(l0.x, phi_, n_ * 8, cudaMemcpyDeviceToDevice);
      else
        cudaMemset(l0.x, 0, n_ * 8);
      // Dirichlet outflow: hold the pressure ghost at 0 (the smoother is interior-only + the non-periodic
      // halo skips it, so 0 persists through every sweep -> p=0 at the outflow).
      if (has_open_boundary_) apply_outflow_pressure_ghost(l0.x);
      if (cheb_ && cutcell_) {
        // Chebyshev semi-iteration (no per-iteration global dot-products -> communication-light at scale).
        // Estimate the spectral bounds of M^{-1}A once, lazily, on the first solve (the RHS = -div is a
        // good non-trivial seed by then; the operator is fixed so the bounds are reused every step).
        if (!cheb_bounds_set_) {
          mg_.estimate_eigenvalues(cheb_a_, cheb_b_, /*iters=*/15, mg_pre_, mg_post_, mg_bottom_);
          cheb_bounds_set_ = true;
        }
        mg_.solve_chebyshev(cheb_maxit_, cheb_rtol_, mg_pre_, mg_post_, mg_bottom_, cheb_a_, cheb_b_);
      } else if (pcg_ && cutcell_) {
        mg_.solve_pcg(pcg_maxit_, pcg_rtol_, mg_pre_, mg_post_, mg_bottom_);
      } else {
        mg_.solve(n_pois, mg_pre_, mg_post_, mg_bottom_);
      }
      cudaMemcpy(phi_, l0.x, n_ * 8, cudaMemcpyDeviceToDevice);
      if (has_open_boundary_) apply_outflow_pressure_ghost(phi_);  // ghost p=0 for correct_k / next warm
    } else {
      if (!pwarm_) cudaMemset(phi_, 0, n_ * 8);  // else keep the previous phi_ as the RB-GS warm start
      for (int it = 0; it < n_pois; ++it) {
        mac_.exchange(phi_);
        pois_k<<<grd_, blk_>>>(phi_, div_, ext_, mac_.origin_incl_ghost, g, 0);
        mac_.exchange(phi_);
        pois_k<<<grd_, blk_>>>(phi_, div_, ext_, mac_.origin_incl_ghost, g, 1);
      }
    }
    mac_.exchange(phi_);
    correct_k<<<grd_, blk_>>>(u_, v_, w_, phi_, ext_, g);
    // high-side outflow normal face (correct_k misses it): u_out -= (0 - phi_inner) -> mass leaves.
    if (has_open_boundary_) apply_outflow_correction();
    // re-impose Dirichlet walls/inflow; do_outflow=false keeps the just-corrected outflow velocity.
    apply_velocity_bc(/*do_outflow=*/false);
    if (has_solid_) { apply_mask(u_); apply_mask(v_); apply_mask(w_); }
    // the projection's grad(phi) correction touches the solid too; re-impose no-slip there so the
    // decoupled solid velocity cannot accumulate (the IBM operator keeps the fluid consistent).
    if (ibm_enabled_)
      for (int c = 0; c < 3; ++c) detail::mask_k<<<grd_, blk_>>>(comp[c], solidmask_[c], ext_);

      // outer-iteration convergence: max velocity change over this Picard iteration. Uses b_[0] as
      // scratch (free post-projection) so div_ (= div(u*)) survives for the incremental update below.
      if (outer_tol_ > 0) {
        double corr = 0.0;
        for (int c = 0; c < 3; ++c) {
          cudaMemcpy(b_[0], comp[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[0], -1.0, prev[c], (long)n_);
          corr = fmax(corr, cfdmpi::mac_max_abs(b_[0], mac_, comm_));
        }
        last_outer_correction_ = corr;
        if (corr < outer_tol_) break;
      }
    }  // outer Picard loop

    // incremental-rotational pressure update, ONCE per step: Phi += phi - nu*dt*div(u*). phi_ is the
    // final projection potential and div_ the final div(u*) (preserved past the convergence check).
    if (incremental_)
      pot_update_k<<<grd_, blk_>>>(phitot_, phi_, div_, beta, ext_, g);
  }

 private:
  // GeometryProvider (Cartesian, SDF -> IBM overlay): count cut cells, (re)allocate the overlay SoA, and
  // fill it from the SDF for velocity component `comp`. This is the ONE call site an octree port replaces
  // (tree-walk + per-cell SDF -> the same ibm_fill_entry); the rest of the momentum solve is mesh-agnostic
  // (base face operator + ibm_modify_stencil_k overlay apply). See mac_ibm.cuh / doc/ibm_overlay.md.
  void build_ibm_overlay(int comp, const double* sdf, float3 off, float3 spacing, int* counter) {
    using namespace cfdmpi::ibmdetail;
    int g = mac_.ghost;
    int3 inner = mac_.inner_res();
    dim3 blk(8, 8, 8);
    dim3 gI((inner.x + 7) / 8, (inner.y + 7) / 8, (inner.z + 7) / 8);
    if (!idmap_[comp]) cudaMalloc(&idmap_[comp], n_ * sizeof(int));
    cudaMemset(counter, 0, sizeof(int));
    ibm_count_ext_k<<<gI, blk>>>(sdf, ext_, g, off, counter);
    int cnt = 0;
    cudaMemcpy(&cnt, counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (ibmdata_[comp].cell_index) cfdmpi::ibm_free(ibmdata_[comp]);
    ibmdata_[comp] = cfdmpi::ibm_alloc(cnt);
    cudaMemset(counter, 0, sizeof(int));
    ibm_geometry_ext_k<0><<<gI, blk>>>(ibmdata_[comp], idmap_[comp], sdf, ext_, g, spacing, counter, off,
                                       /*bc=Dirichlet*/ 0);
    cudaMemcpy(&ibmdata_[comp].num_active_cells, counter, sizeof(int), cudaMemcpyDeviceToHost);
  }
  void apply_mask(double* c) { detail::mask_k<<<grd_, blk_>>>(c, solid_, ext_); }

  // --- domain boundary conditions (apply only on the faces THIS rank owns at the domain boundary) ---
  void bc_face_flags(bool atlo[3], bool athi[3]) const {
    int g = mac_.ghost;
    int3 og = mac_.origin_incl_ghost, N = mac_.global_res;
    atlo[0] = og.x + g == 0;  atlo[1] = og.y + g == 0;  atlo[2] = og.z + g == 0;
    athi[0] = og.x + ext_.x - g == N.x;  athi[1] = og.y + ext_.y - g == N.y;
    athi[2] = og.z + ext_.z - g == N.z;
  }
  // fill the boundary ghosts of ONE velocity component (call after its halo exchange). fold=0 ->
  // reflection / zero-gradient ghosts (explicit stencils); fold=1 -> ghosts dropped to 0 (implicit
  // diffusion). do_outflow=false skips outflow faces (used after the projection, which already corrected
  // the outflow velocity -- re-imposing zero-gradient would clobber it).
  void apply_velocity_bc_comp(double* f, int comp, int fold = 0, bool do_outflow = true) {
    if (!has_domain_bc_) return;
    ensure_bc_profiles();
    bool atlo[3], athi[3];
    bc_face_flags(atlo, athi);
    int g = mac_.ghost, dims[3] = {ext_.x, ext_.y, ext_.z};
    for (int a = 0; a < 3; ++a) {
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      for (int s = 0; s < 2; ++s) {
        int face = 2 * a + s;
        if (bc_type_[face] == 0 || (s == 0 ? !atlo[a] : !athi[a])) continue;
        if (bc_type_[face] == 3) {  // OUTFLOW: zero-gradient (fold=0) / dropped (fold=1)
          if (do_outflow) cfdmpi::bcdetail::bc_outflow_comp_k<<<grd, blk>>>(f, ext_, g, a, s, comp, fold);
          continue;
        }
        float3 wv = bc_vel_[face];
        double wc = comp == 0 ? wv.x : (comp == 1 ? wv.y : wv.z);  // scalar fallback / tangential fold value
        cfdmpi::bcdetail::bc_velocity_comp_k<<<grd, blk>>>(f, ext_, g, a, s, comp, wc, fold,
                                                           bc_prof_[face], dims[cc]);
      }
    }
  }
  void apply_velocity_bc(bool do_outflow = true) {
    apply_velocity_bc_comp(u_, 0, 0, do_outflow);
    apply_velocity_bc_comp(v_, 1, 0, do_outflow);
    apply_velocity_bc_comp(w_, 2, 0, do_outflow);
  }
  // Precompute the implicit-diffusion face-fold (once; beta = nu*dt is fixed): per component, a diagonal
  // correction bc_dcorr_[c] and an RHS fold bc_brhs_[c] at the boundary-adjacent inner cells.
  //   Dirichlet wall (tangential only): dcorr += beta, brhs += 2*beta*wall   (drop the 2*wall-u ghost).
  //   Zero-gradient outflow (every comp): dcorr -= beta, brhs += 0           (drop the u_inner ghost).
  // The diffusion then drops the boundary ghosts and folds their coefficient into the diagonal -- no
  // lagged boundary ghost in the implicit solve.
  void setup_bc_diffusion() {
    if (!has_domain_bc_ || bc_diff_built_) return;
    double beta = nu_ * dt_;
    bool atlo[3], athi[3];
    bc_face_flags(atlo, athi);
    int g = mac_.ghost, dims[3] = {ext_.x, ext_.y, ext_.z};
    for (int comp = 0; comp < 3; ++comp) {
      if (!bc_dcorr_[comp]) cudaMalloc(&bc_dcorr_[comp], n_ * 8);
      if (!bc_brhs_[comp]) cudaMalloc(&bc_brhs_[comp], n_ * 8);
      cudaMemset(bc_dcorr_[comp], 0, n_ * 8);
      cudaMemset(bc_brhs_[comp], 0, n_ * 8);
      for (int a = 0; a < 3; ++a) {
        int b = (a + 1) % 3, cc = (a + 2) % 3;
        dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
        for (int s = 0; s < 2; ++s) {
          int face = 2 * a + s;
          if (bc_type_[face] == 0 || (s == 0 ? !atlo[a] : !athi[a])) continue;
          double dval, bval;
          if (bc_type_[face] == 3) {       // OUTFLOW (zero-gradient): every component
            dval = -beta; bval = 0.0;
          } else if (comp != a) {          // Dirichlet WALL: tangential only (normal face held directly)
            float3 wv = bc_vel_[face];
            double wc = comp == 0 ? wv.x : (comp == 1 ? wv.y : wv.z);
            dval = beta; bval = 2.0 * beta * wc;
          } else {
            continue;
          }
          cfdmpi::bcdetail::bc_diffusion_fold_k<<<grd, blk>>>(bc_dcorr_[comp], bc_brhs_[comp], ext_, g, a,
                                                              s, dval, bval);
        }
      }
    }
    bc_diff_built_ = true;
  }
  // zero the OPERATOR-openness (ox_/oy_/oz_ = alpha) on wall/inflow faces (-> Neumann pressure). Outflow
  // (type 3) is LEFT OPEN (alpha != 0 -> Dirichlet p=0). Call at operator build.
  void apply_wall_openness() {
    if (!has_domain_bc_ || !ox_) return;
    bool atlo[3], athi[3];
    bc_face_flags(atlo, athi);
    int g = mac_.ghost, dims[3] = {ext_.x, ext_.y, ext_.z};
    double* oarr[3] = {ox_, oy_, oz_};
    for (int a = 0; a < 3; ++a) {
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      for (int s = 0; s < 2; ++s) {
        int face = 2 * a + s;
        if (bc_type_[face] == 0 || bc_type_[face] == 3 || (s == 0 ? !atlo[a] : !athi[a])) continue;
        cfdmpi::bcdetail::bc_zero_openness_k<<<grd, blk>>>(oarr[a], ext_, g, a, s);
      }
    }
  }
  // hold the pressure ghost at 0 on every owned outflow face -> Dirichlet p=0 there.
  void apply_outflow_pressure_ghost(double* p) {
    bool atlo[3], athi[3];
    bc_face_flags(atlo, athi);
    int g = mac_.ghost, dims[3] = {ext_.x, ext_.y, ext_.z};
    for (int a = 0; a < 3; ++a) {
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      for (int s = 0; s < 2; ++s) {
        int face = 2 * a + s;
        if (bc_type_[face] != 3 || (s == 0 ? !atlo[a] : !athi[a])) continue;
        cfdmpi::bcdetail::bc_zero_pressure_ghost_k<<<grd, blk>>>(p, ext_, g, a, s);
      }
    }
  }
  // correct the high-side outflow normal-velocity face (correct_k handles the low side + the interior).
  void apply_outflow_correction() {
    bool atlo[3], athi[3];
    bc_face_flags(atlo, athi);
    int g = mac_.ghost, dims[3] = {ext_.x, ext_.y, ext_.z};
    double* comp[3] = {u_, v_, w_};
    for (int a = 0; a < 3; ++a) {
      if (bc_type_[2 * a + 1] != 3 || !athi[a]) continue;  // +a face is outflow and owned by this rank
      int b = (a + 1) % 3, cc = (a + 2) % 3;
      dim3 blk(16, 16), grd((dims[b] + 15) / 16, (dims[cc] + 15) / 16);
      cfdmpi::bcdetail::correct_outflow_k<<<grd, blk>>>(comp[a], phi_, ext_, g, a);
    }
  }
  void ensure_vmg_built() {
    if (vmg_built_) return;
    // domain-BC (cavity/BFS): semi-coarsening so a thin (quasi-2D) axis freezes while the wide axes keep
    // coarsening -> a deep hierarchy on shallow grids. semi self-limits (init breaks when nothing coarsens),
    // so the requested level count need not be clamped. IBM/porous path stays uniform + clamp_levels.
    bool semi = has_domain_bc_;
    int nl = semi ? vmg_levels_ : clamp_levels(vmg_levels_);
    vmg_.init(mac_.global_res, mac_.rank, mac_.size, /*h0=*/1.0, nl, comm_, mac_.ghost, periodic_, semi);
    if (!has_domain_bc_)
      vmg_.setDiffusionCoarse(nu_ * dt_, 1.0);  // IBM: coarse built once (fine swapped per comp); BC path
                                                // rebuilds all levels per component via setDiffusionConstAllLevels
    // Upwind-convective coarse operator (opt-in, implicit-FOU path): per-level scratch for the restricted
    // advecting velocity. Level 0 is the fine operator (IBM: As_[c]; domain-BC: built from u_/v_/w_), so
    // [0] is left null. Zero-init so non-periodic boundary ghosts read 0 (no spurious coarse advective flux
    // at walls; the halo exchange leaves them untouched and the restriction only writes inner cells).
    if (implicit_fou_) {
      int N = vmg_.n_levels();
      vadv_u_.assign(N, nullptr);
      vadv_v_.assign(N, nullptr);
      vadv_w_.assign(N, nullptr);
      for (int L = 1; L < N; ++L) {
        std::size_t nb = vmg_.level(L).n * 8;
        for (double** p : {&vadv_u_[L], &vadv_v_[L], &vadv_w_[L]}) {
          cudaMalloc(p, nb);
          cudaMemset(*p, 0, nb);
        }
      }
      // domain-BC: allocate the per-level operator arrays (AC..AT) and set the const-coeff MG flags
      // (galerkin off, no mean removal); build_vmg_adv_stencil then overwrites them each Picard iteration.
      // The IBM path's setDiffusionCoarse (above) already did this for levels 1..N-1.
      if (has_domain_bc_) vmg_.setDiffusionConstAllLevels(nu_ * dt_, 1.0);
    }
    // volume-fraction coarse-op scratch: one coarse theta buffer per coarse level (reused across the 3
    // components and across steps; the geometry is static). Index L>=1; [0] unused (level 0 = vfine_[c]).
    if (vmg_volfrac_ && !has_domain_bc_) {
      int N = vmg_.n_levels();
      vtheta_lvl_.assign(N, nullptr);
      for (int L = 1; L < N; ++L) cudaMalloc(&vtheta_lvl_[L], vmg_.level(L).n * 8);
    }
    // NB: a geometry-aware (rediscretized) velocity coarse operator was tried and *diverges* -- the
    // Robust-Scaled fine stencil is row-scaled by D_rescale at cut cells, so it is inconsistent with a
    // clean I-beta*L coarse operator under geometric transfers (and the staggered velocity geometry is
    // not the cell-face openness). The const-coeff coarse converges but slowly; velocity RB-GS
    // (set_velocity_solver_params) is the exact, recommended default. Proper velocity MG = deferred.
    vmg_built_ = true;
  }
  // Upwind-convective velocity-MG (implicit-FOU): restrict the advecting velocity u_,v_,w_ to every coarse
  // level (8:1 volume average; numerical diffusion is welcome on the coarse -> keeps the M-matrix) and fill
  // the coarse ghosts. Called once per Picard iteration (velocities frozen at u^k), reused by all 3 comps.
  void restrict_vmg_adv_velocities() {
    dim3 blk(8, 8, 8);
    int N = vmg_.n_levels();
    const double* fsrc[3] = {u_, v_, w_};
    std::vector<double*>* dst[3] = {&vadv_u_, &vadv_v_, &vadv_w_};
    for (int L = 1; L < N; ++L) {
      cfdmpi::MGLevel& cs = vmg_.level(L);
      cfdmpi::MGLevel& fn = vmg_.level(L - 1);
      dim3 cgrd((cs.inner.x + 7) / 8, (cs.inner.y + 7) / 8, (cs.inner.z + 7) / 8);
      for (int c = 0; c < 3; ++c) {
        const double* fine = (L == 1) ? fsrc[c] : (*dst[c])[L - 1];
        double* coarse = (*dst[c])[L];
        cfdmpi::mgdetail::mg_restrict_k<<<cgrd, blk>>>(coarse, fine, cs.ext, fn.ext, fn.g, cs.inner,
                                                       fn.ratio);
        cs.mac.exchange(coarse);
      }
    }
  }
  // Build the velocity stencils (AC..AT) for component `comp`: anisotropic const-coeff backward-Euler
  // diffusion (per-axis beta from cfac) + first-order-upwind advection from the restricted coarse velocity
  // (scaled by 1/cfac per face axis). `include_fine` selects which levels are written: the IBM path keeps
  // level 0 as the fine As_[comp] (include_fine=false, levels 1..N-1); the domain-BC path has no IBM
  // stencil, so it also builds level 0 from u_,v_,w_ (include_fine=true) and folds the walls afterwards.
  // Call after restrict_vmg_adv_velocities(), per component, before setDiffusionBoundaryFold + solve.
  void build_vmg_adv_stencil(int comp, bool include_fine) {
    dim3 blk(8, 8, 4);
    int N = vmg_.n_levels();
    double* l0[3] = {u_, v_, w_};
    for (int L = include_fine ? 0 : 1; L < N; ++L) {
      cfdmpi::MGLevel& c = vmg_.level(L);
      if (!c.AC)
        for (cfdmpi::mreal** p : {&c.AC, &c.AW, &c.AE, &c.AS, &c.AN, &c.AB, &c.AT})
          cudaMalloc(p, c.n * sizeof(cfdmpi::mreal));
      c.variable = true;
      double hx = (double)c.cfac.x, hy = (double)c.cfac.y, hz = (double)c.cfac.z;  // h0 = 1
      double bx = nu_ * dt_ / (hx * hx), by = nu_ * dt_ / (hy * hy), bz = nu_ * dt_ / (hz * hz);
      double sx = 1.0 / hx, sy = 1.0 / hy, sz = 1.0 / hz;
      dim3 grd((c.ext.x + 7) / 8, (c.ext.y + 7) / 8, (c.ext.z + 3) / 4);
      double* u = (L == 0) ? l0[0] : vadv_u_[L];
      double* v = (L == 0) ? l0[1] : vadv_v_[L];
      double* w = (L == 0) ? l0[2] : vadv_w_[L];
      if (comp == 0)
        detail::build_adv_coarse_stencil_k<0><<<grd, blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, u, v,
                                                    w, c.ext, c.g, bx, by, bz, dt_, sx, sy, sz);
      else if (comp == 1)
        detail::build_adv_coarse_stencil_k<1><<<grd, blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, u, v,
                                                    w, c.ext, c.g, bx, by, bz, dt_, sx, sy, sz);
      else
        detail::build_adv_coarse_stencil_k<2><<<grd, blk>>>(c.AC, c.AW, c.AE, c.AS, c.AN, c.AB, c.AT, u, v,
                                                    w, c.ext, c.g, bx, by, bz, dt_, sx, sy, sz);
    }
  }
  // Cap the requested MG level count so no axis coarsens below 2 cells (else the ORB decomposer divides
  // by a zero dimension -- e.g. a quasi-2D nz=4 grid cannot take the default 4 levels: 4>>3 = 0).
  int clamp_levels(int want) const {
    int md = mac_.global_res.x;
    if (mac_.global_res.y < md) md = mac_.global_res.y;
    if (mac_.global_res.z < md) md = mac_.global_res.z;
    int lev = want;
    while (lev > 1 && (md >> (lev - 1)) < 2) --lev;
    return lev;
  }
  // Levels achievable under SEMI-coarsening: halve each axis only while it stays even and >= 2, so a thin
  // axis (quasi-2D nz=4) freezes while the others keep coarsening -> more depth than uniform clamp_levels.
  int semi_level_count(int want) const {
    int3 r = mac_.global_res;
    int lev = 1;
    while (lev < want) {
      bool any = false;
      if (r.x % 2 == 0 && r.x / 2 >= 2) { r.x /= 2; any = true; }
      if (r.y % 2 == 0 && r.y / 2 >= 2) { r.y /= 2; any = true; }
      if (r.z % 2 == 0 && r.z / 2 >= 2) { r.z /= 2; any = true; }
      if (!any) break;
      ++lev;
    }
    return lev;
  }
  void ensure_mg_built() {
    if (mg_built_) return;
    // same ORB decomposition + ghost width (2) as this solver, so MG level-0 blocks share the layout.
    // Native-BC problems use semi-coarsening (thin axes freeze -> deeper hierarchy on the wide axes); the
    // periodic / IBM (porous) path keeps uniform coarsening + clamp_levels -> byte-identical.
    bool semi = has_domain_bc_;
    int nlev = semi ? semi_level_count(mg_levels_) : clamp_levels(mg_levels_);
    mg_.init(mac_.global_res, mac_.rank, mac_.size, /*h0=*/1.0, nlev, comm_, /*ghost=*/mac_.ghost,
             periodic_, semi);
    mg_built_ = true;
  }

  MacGridHalo mac_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  double nu_ = 0, dt_ = 0, fx_ = 0, fy_ = 0, fz_ = 0;
  bool has_solid_ = false;
  bool advection_ = false;
  bool implicit_fou_ = false;
  std::size_t n_ = 0;
  int3 ext_{}, origin_{};
  dim3 blk_, grd_;
  double *u_ = nullptr, *v_ = nullptr, *w_ = nullptr, *phi_ = nullptr, *div_ = nullptr,
         *solid_ = nullptr;
  // incremental-rotational pressure: accumulated velocity potential Phi (= dt/rho * pressure)
  double* phitot_ = nullptr;
  bool incremental_ = false;
  double* b_[3] = {nullptr, nullptr, nullptr};
  // Picard outer loop: time-base u^n (old) + per-iteration snapshot (prev) for the convergence test
  double *u_old_ = nullptr, *v_old_ = nullptr, *w_old_ = nullptr;
  double *up_ = nullptr, *vp_ = nullptr, *wp_ = nullptr;
  int outer_iters_ = 1;
  double outer_tol_ = -1.0;
  int last_outer_iterations_ = 0;
  double last_outer_correction_ = 0.0;
  // opt-in multigrid pressure solve (built lazily on first step)
  cfdmpi::DistributedPoissonMG mg_;
  bool mg_enabled_ = false, mg_built_ = false;
  int mg_levels_ = 4, mg_pre_ = 2, mg_post_ = 2, mg_bottom_ = 12;
  // opt-in velocity-diffusion multigrid (for the IBM momentum solve)
  cfdmpi::DistributedPoissonMG vmg_;
  bool vmg_enabled_ = false, vmg_built_ = false;
  int vmg_levels_ = 3, vmg_vcycles_ = 4, vmg_pre_ = 2, vmg_post_ = 2, vmg_bottom_ = 10;
  // upwind-convective velocity-MG (implicit-FOU path): per-level restricted advecting velocity. Index L>=1
  // holds the coarse-level scratch; [0] is left null and aliases u_/v_/w_ as the level-0 restriction source.
  std::vector<double*> vadv_u_, vadv_v_, vadv_w_;
  // volume-fraction velocity-MG coarse operator (IBM diffusion path, opt-in). vfine_[c] = per-component
  // fine fluid volume fraction theta (static, from the SDF); vtheta_lvl_ = coarse theta scratch reused
  // across components/levels (index L>=1). See doc/velocity_mg_plan.md (Phases 1+3; NOT the un-scale).
  bool vmg_volfrac_ = false;
  bool vmg_mask_xfer_ = false;
  bool vmg_res_mask_ = true;   // zero the cut-cell residual before restriction (the consistent-residual fix)
  double vmg_eps_ = 0.1;
  double* vfine_[3] = {nullptr, nullptr, nullptr};
  double* vresmask_[3] = {nullptr, nullptr, nullptr};  // 0 at IBM cut cells, 1 elsewhere (per component)
  std::vector<double*> vtheta_lvl_;
  // cut-cell pressure operator: staggered face openness + the flux-divergence flag
  double *ox_ = nullptr, *oy_ = nullptr, *oz_ = nullptr;
  bool cutcell_ = false;
  // Robust-Scaled velocity IBM: per-component (u/v/w) modified diffusion stencil + Dirichlet inhom
  bool ibm_enabled_ = false;
  IBM_Data ibmdata_[3] = {};
  int* idmap_[3] = {nullptr, nullptr, nullptr};
  cfdmpi::mreal* As_[3][7] = {};  // momentum-solve matrix: single precision (see mac_ibm.cuh mreal)
  double* inhom_[3] = {nullptr, nullptr, nullptr};
  double* solidmask_[3] = {nullptr, nullptr, nullptr};
  double* descale_[3] = {nullptr, nullptr, nullptr};  // Robust-Scaled per-cell RHS scale (D_rescale)
  float3 ubc_ibm_ = make_float3(0, 0, 0);             // IBM wall velocity (for per-iteration re-bake)
  // Domain boundary conditions (per face 0=-x,1=+x,2=-y,3=+y,4=-z,5=+z; type 0=periodic,1=noslip,2=dirichlet)
  std::array<bool, 3> periodic_ = {true, true, true};
  int bc_type_[6] = {0, 0, 0, 0, 0, 0};
  float3 bc_vel_[6] = {};
  bool has_domain_bc_ = false;
  bool has_open_boundary_ = false;                     // inflow(normal)/outflow -> alpha (operator) != beta
  double* bc_dcorr_[3] = {nullptr, nullptr, nullptr};  // implicit-diffusion face-fold: +-beta diagonal
  double* bc_brhs_[3] = {nullptr, nullptr, nullptr};   //   correction + 2*beta*wall RHS, per component
  double* bx_ = nullptr, *by_ = nullptr, *bz_ = nullptr;  // FLUX openness (divergence/correction); open at
                                                          //   inflow+outflow (ox_/oy_/oz_ = operator alpha)
  bool bc_diff_built_ = false;
  // per-face inlet velocity profile (e.g. parabolic channel inlet / backward-step partial inlet)
  double* bc_prof_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};  // device, local plane
  std::vector<double> bc_prof_host_[6];                  // global plane [Nb][Nc][3], set before init
  int bc_prof_nb_[6] = {0, 0, 0, 0, 0, 0}, bc_prof_nc_[6] = {0, 0, 0, 0, 0, 0};
  bool has_prof_[6] = {false, false, false, false, false, false};
  bool bc_prof_built_ = false;
  // CG-accelerated pressure solve (for the stiff cut-cell operator)
  bool pcg_ = false;
  int pcg_maxit_ = 60;
  double pcg_rtol_ = 1e-8;
  bool cheb_ = false;            // Chebyshev semi-iteration outer accelerator (communication-light)
  int cheb_maxit_ = 60;
  double cheb_rtol_ = 1e-8;
  double cheb_a_ = 0.0, cheb_b_ = 0.0;  // estimated spectral bounds of M^{-1}A (set lazily on step 1)
  bool cheb_bounds_set_ = false;
  bool pwarm_ = false;  // warm-start the pressure solve from the previous step's projection potential
                        // (opt-in; off keeps the cold-start bit-exact behaviour the ctests assume)
  // 3-stream concurrent velocity solve: the independent u/v/w IBM RB-GS momentum solves run on their
  // own CUDA streams with per-component exchange engines (separate host-staged buffers -> safe at any
  // rank count). Overlaps at small per-component sizes; ~no effect once one stencil saturates the GPU.
  bool vstreams_enabled_ = true;
  bool vstreams_init_ = false;
  cudaStream_t vstreams_[3] = {};
  tpx::halo::DeviceGridExchange<double> vexch_[3];
  cudaEvent_t vrhs_evt_ = nullptr;
};

}  // namespace dns
