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

class DistributedNS {
 public:
  void init(int3 global_res, int rank, int size, double nu, double dt,
            MPI_Comm comm = MPI_COMM_WORLD) {
    // Ghost width 2 covers the Koren advection reach (diffusion/projection use only width 1).
    mac_.init(global_res, rank, size, {true, true, true}, /*ghost_width=*/2, comm);
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
                      u_old_, v_old_, w_old_, up_, vp_, wp_})
      if (p) cudaFree(p);
    for (int c = 0; c < 3; ++c) {
      if (idmap_[c]) cudaFree(idmap_[c]);
      if (inhom_[c]) cudaFree(inhom_[c]);
      if (solidmask_[c]) cudaFree(solidmask_[c]);
      if (descale_[c]) cudaFree(descale_[c]);
      for (int k = 0; k < 7; ++k)
        if (As_[c][k]) cudaFree(As_[c][k]);
      if (ibmdata_[c].cell_index) cfdmpi::ibm_free(ibmdata_[c]);
    }
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
    // staggered face openness, kept for the cut-cell flux divergence (projection RHS + diagnostic)
    cfdmpi::ccdetail::cc_build_open_k<<<gE, dim3(8, 8, 4)>>>(ox_, oy_, oz_, sdf, ext_, 1.0, 1.0, 1.0);
    if (coarse_mode == 0)
      mg_.setFineVariableOperatorRediscretized(ox_, oy_, oz_, 1.0, 1.0, 1.0);
    else
      mg_.setFineVariableOperator(ox_, oy_, oz_, 1.0, 1.0, 1.0, /*galerkin=*/coarse_mode == 1);
    cudaFree(sdf);
    cutcell_ = true;
  }

  // Solve the pressure Poisson with CG preconditioned by the (Galerkin) V-cycle instead of plain
  // V-cycles. Needed to actually converge the stiff cut-cell operator; requires the cut-cell operator
  // (set_cutcell_pressure_operator). step()'s n_pois is then ignored in favour of max_iter/rtol.
  void set_pressure_pcg(bool on, int max_iter = 60, double rtol = 1e-8) {
    pcg_ = on;
    pcg_maxit_ = max_iter;
    pcg_rtol_ = rtol;
  }

  // Solve the IBM velocity diffusion with geometric multigrid (constant-coefficient coarse operators)
  // instead of plain RB-GS -- far fewer iterations when the diffusion is stiff (large nu*dt). Requires
  // the IBM operator (set_ibm_solid). n_diff then counts V-cycles.
  void set_velocity_multigrid(bool on, int n_levels = 3, int v_cycles = 4) {
    vmg_enabled_ = on;
    vmg_levels_ = n_levels;
    vmg_vcycles_ = v_cycles;
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
    int g = mac_.ghost;
    int3 inner = mac_.inner_res();
    dim3 blk(8, 8, 8);
    dim3 gI((inner.x + 7) / 8, (inner.y + 7) / 8, (inner.z + 7) / 8);
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
      if (!idmap_[c]) cudaMalloc(&idmap_[c], n_ * sizeof(int));
      cudaMemset(counter, 0, sizeof(int));
      ibm_count_ext_k<<<gI, blk>>>(sdf, ext_, g, offs[c], counter);
      int cnt = 0;
      cudaMemcpy(&cnt, counter, sizeof(int), cudaMemcpyDeviceToHost);
      if (ibmdata_[c].cell_index) cfdmpi::ibm_free(ibmdata_[c]);
      ibmdata_[c] = cfdmpi::ibm_alloc(cnt);
      cudaMemset(counter, 0, sizeof(int));
      ibm_geometry_ext_k<0><<<gI, blk>>>(ibmdata_[c], idmap_[c], sdf, ext_, g, spacing, counter,
                                         offs[c], /*bc=Dirichlet*/ 0);
      cudaMemcpy(&ibmdata_[c].num_active_cells, counter, sizeof(int), cudaMemcpyDeviceToHost);

      if (!As_[c][0])
        for (int k = 0; k < 7; ++k) cudaMalloc(&As_[c][k], n_ * sizeof(cfdmpi::mreal));
      if (!inhom_[c]) cudaMalloc(&inhom_[c], n_ * 8);
      if (!solidmask_[c]) cudaMalloc(&solidmask_[c], n_ * 8);
      if (!descale_[c]) cudaMalloc(&descale_[c], n_ * 8);
      ibm_solid_mask_k<<<gE, blk>>>(solidmask_[c], sdf, ext_, offs[c]);
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

  // Global max of the cut-cell flux divergence |div(open u)| over owned cells (the quantity the
  // cut-cell projection drives to zero). Uses div_ as scratch; call between steps.
  double max_open_divergence() {
    if (!cutcell_) return 0.0;
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    detail::diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, ox_, oy_, oz_, div_, ext_, mac_.ghost);
    return cfdmpi::mac_max_abs(div_, mac_, comm_);
  }
  // RMS of the cut-cell flux divergence (bulk measure; less sensitive to a single thin cut cell)
  double rms_open_divergence() {
    if (!cutcell_) return 0.0;
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    detail::diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, ox_, oy_, oz_, div_, ext_, mac_.ghost);
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
        for (int c = 0; c < 3; ++c)
          advect_rhs_k<<<grd_, blk_>>>(c, u_, v_, w_, comp[c], b_[c], dt_, f[c], ext_, g);
        // retarget the time base from u^k to u^n:  b += (u^n - u^k)
        for (int c = 0; c < 3; ++c) {
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[c], 1.0, old[c], (long)n_);
          cfdmpi::mgdetail::mg_axpy_k<<<blocks, threads>>>(b_[c], -1.0, comp[c], (long)n_);
        }
        if (implicit_fou_ && ibm_enabled_) {
          // deferred correction: b += dt*FOU(u^k) -> b = u^n + dt*f - dt*(Koren - FOU); and rebuild the
          // velocity stencil = diffusion + dt*FOU(u^k), then re-apply the IBM bake (descale_/inhom_).
          add_fou_rhs_k<0><<<grd_, blk_>>>(b_[0], u_, v_, w_, u_, ext_, g, dt_);
          add_fou_rhs_k<1><<<grd_, blk_>>>(b_[1], u_, v_, w_, v_, ext_, g, dt_);
          add_fou_rhs_k<2><<<grd_, blk_>>>(b_[2], u_, v_, w_, w_, ext_, g, dt_);
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
      if (vmg_enabled_ && !implicit_fou_) {  // implicit-FOU changes the stencil each iter -> RB-GS
        ensure_vmg_built();
        for (int c = 0; c < 3; ++c) {
          vmg_.setDiffusionFine(As_[c]);  // fine = this component's float IBM stencil (coarse built once)
          cfdmpi::MGLevel& l0 = vmg_.level(0);
          cudaMemcpy(l0.rhs, b_[c], n_ * 8, cudaMemcpyDeviceToDevice);
          cudaMemcpy(l0.x, comp[c], n_ * 8, cudaMemcpyDeviceToDevice);  // initial guess
          vmg_.solve(vmg_vcycles_, vmg_pre_, vmg_post_, vmg_bottom_);  // (n_diff ignored under vmg)
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
    }
    // projection (cut-cell operator -> open-weighted flux divergence RHS; else plain divergence)
    mac_.exchange(u_); mac_.exchange(v_); mac_.exchange(w_);
    if (cutcell_)
      diverg_open_k<<<grd_, blk_>>>(u_, v_, w_, ox_, oy_, oz_, div_, ext_, g);
    else
      diverg_k<<<grd_, blk_>>>(u_, v_, w_, div_, ext_, g);
    if (mg_enabled_) {
      // multigrid pressure solve: n_pois V-cycles of A phi = -div (same periodic Laplacian as pois_k)
      ensure_mg_built();
      cfdmpi::MGLevel& l0 = mg_.level(0);  // ghost-2 layout identical to this solver's blocks
      int threads = 256, blocks = (int)((n_ + threads - 1) / threads);
      neg_k<<<blocks, threads>>>(l0.rhs, div_, (long)n_);
      cudaMemset(l0.x, 0, n_ * 8);
      if (pcg_ && cutcell_)
        mg_.solve_pcg(pcg_maxit_, pcg_rtol_, mg_pre_, mg_post_, mg_bottom_);
      else
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
  void apply_mask(double* c) { detail::mask_k<<<grd_, blk_>>>(c, solid_, ext_); }
  void ensure_vmg_built() {
    if (vmg_built_) return;
    vmg_.init(mac_.global_res, mac_.rank, mac_.size, /*h0=*/1.0, vmg_levels_, comm_, mac_.ghost);
    vmg_.setDiffusionCoarse(nu_ * dt_, 1.0);  // const-coeff diffusion coarse operators (built once)
    vmg_built_ = true;
  }
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
  // CG-accelerated pressure solve (for the stiff cut-cell operator)
  bool pcg_ = false;
  int pcg_maxit_ = 60;
  double pcg_rtol_ = 1e-8;
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
