/// @file
/// @brief Robust-Scaled cut-cell IBM for the velocity (momentum) solve.
// cfd-gpu -- Robust-Scaled cut-cell IBM for the velocity (momentum) solve on a MacGridHalo extended
// block. Ports the production solver's velocity IBM (cfd_solver_ibm.cu / _ibm_kernels.cuh): per cut cell
// the SDF geometry gives polynomial factors (D_rescale, K/M/X/Nbc) that are baked into the velocity
// diffusion stencil A_C..A_T plus an inhomogeneous Dirichlet term, eliminating the solid ghost values
// and enforcing the wall velocity. The bake (ibm_modify_stencil_k) only edits each cut cell's OWN row,
// so it distributes cleanly; only the SDF geometry needed porting to extended-block (no-wrap) sampling.
//
// The IBM math is factored into ibm_fill_entry (shared by the extended build here and a serial wrap
// reference) so the distributed coefficients match the serial ones cell-for-cell.
//
// ===========================================================================================
// SPARSE-OVERLAY ARCHITECTURE (octree/AMR forward-compatible -- see doc/ibm_overlay.md)
// -------------------------------------------------------------------------------------------
// The momentum operator is a mesh-agnostic BASE face operator + a sparse IBM OVERLAY. The IBM is
// deliberately row-based / non-symmetric and is NEVER multigridded -- momentum is legitimately
// non-conservative across an immersed boundary (the wall exerts force), and the D_rescale row scaling is
// Gauss-Seidel-invariant but not MG-invariant. So we keep it row-based and treat it as an overlay on the
// face architecture, not a competitor to it. Three layers + two mesh-specific providers:
//
//   1. BASE operator       ibm_build_diffusion_k   A = I - beta*L (face loop, beta per face = nu*dt*gf).
//   2. OVERLAY (data)      IBM_Data (cfd_solver.cuh) -- a sparse SoA of cut cells: {cell handle
//                          cell_index; per-face neighbour/direction hook dir_code + coefficients
//                          R/K/M/X/Nbc; D_rescale}. Read-only here (shared with pnm_backend).
//   3. APPLY               ibm_modify_stencil_k -- loop the overlay, modify each cut cell's OWN row.
//
//   Provider A (GEOMETRY): geometry -> overlay entries. Cartesian = ibm_gather_ext (SDF sampling) +
//     ibm_fill_entry (already indexing-agnostic). Octree = tree-walk + per-cell SDF, the SAME ibm_fill_entry.
//   Provider B (CONNECTIVITY): per face -> (base-stencil slot, opposite face for the X cross-term).
//     Cartesian = the implicit 7-point (slot order below; OPP[k]=k^1). Octree = the tree (via dir_code).
//
// An octree port replaces only Provider A and Provider B; the numerics (base, overlay coefficients,
// apply math) are unchanged. See doc/ibm_overlay.md for the extension steps.
// ===========================================================================================
#pragma once

#include <cuda_runtime.h>

#include "cfd_solver.cuh"               // IBM_Data, get_idx
#include "cfd_solver_ibm_kernels.cuh"   // poly_* (Robust-Scaled polynomials)
#include "mac_cutcell.cuh"              // cc_sample_ext (clamped extended-block SDF sampling)

namespace cfdmpi {

// Mixed precision: the momentum-solve matrix (the velocity diffusion/advection stencil streamed every
// Red-Black sweep) is stored in single precision, while the iterate, RHS, residual, inhomogeneous
// Dirichlet term and the Robust-Scaled RHS factor stay double. The stencil only sets the OPERATOR, so
// float storage perturbs it at ~1e-7 (the matrix-in-float choice); all accumulation that feeds the
// solution (the diagonal assembly, inhom) is done in double and stored, so the converged velocity keeps
// double accuracy. Flip to double to recover the original bit-for-bit behaviour.
using mreal = float;

namespace ibmdetail {

// Decide whether the staggered point (sdf_c) with its 6 axis neighbours (sdf_n) is an IBM cut cell:
// fluid centre with at least one solid neighbour.
__host__ __device__ inline bool ibm_is_cut(float sdf_c, const float sdf_n[6]) {
  if (sdf_c <= 0.0f) return false;
  for (int k = 0; k < 6; ++k)
    if (sdf_n[k] < 0.0f) return true;
  return false;
}

// Fill one IBM_Data entry (list_idx) for a cut cell at linear index c_idx from its 7 SDF samples.
// Verbatim port of compute_ibm_geometry_kernel's per-cell body (Dirichlet bc_type=0 / Neumann=1,
// SCHEME 0 point-value / 1 cell-average). Indexing-agnostic: c_idx is whatever layout the caller uses.
template <int SCHEME>
__device__ inline void ibm_fill_entry(IBM_Data ibm, int list_idx, int c_idx, float sdf_c,
                                      const float sdf_n[6], float3 spacing, int bc_type) {
  ibm.cell_index[list_idx] = c_idx;
  ibm.num_boundaries[list_idx] = 6;
  bool is_ghost[6];
  float xi_vals[6], D_vals[6];
  for (int k = 0; k < 6; ++k) {
    bool is_solid_n = (sdf_n[k] < 0.0f);
    if (is_solid_n) {  // centre is fluid (cut cell), so sign differs => interface
      is_ghost[k] = true;
      if (bc_type == 0) {
        float theta = sdf_c / (sdf_c - sdf_n[k]);
        if (theta < 1e-4f) theta = 1e-4f;
        if (theta > 1.0f) theta = 1.0f;
        xi_vals[k] = theta;
        D_vals[k] = (SCHEME == 0) ? poly_D(theta) : poly_D_avg(theta);
      } else {
        xi_vals[k] = 0.5f;
        D_vals[k] = 1.0f;
      }
    } else {
      is_ghost[k] = false;
      xi_vals[k] = 1.0f;
      D_vals[k] = 1e9f;
    }
  }

  if (bc_type == 0) {
    bool is_sandwich[3] = {is_ghost[0] && is_ghost[1], is_ghost[2] && is_ghost[3],
                           is_ghost[4] && is_ghost[5]};
    float D_sandwich[3] = {0, 0, 0};
    for (int a = 0; a < 3; ++a)
      if (is_sandwich[a])
        D_sandwich[a] = (SCHEME == 0) ? poly_D_sandwich(xi_vals[2 * a + 1], xi_vals[2 * a])
                                      : poly_D_sandwich_avg(xi_vals[2 * a + 1], xi_vals[2 * a]);

    float min_D_abs = 1e30f, D_rescale = 1.0f;
    auto update_min = [&](float val) {
      if (fabsf(val) < min_D_abs) { min_D_abs = fabsf(val); D_rescale = val; }
    };
    for (int axis = 0; axis < 3; ++axis) {
      if (is_sandwich[axis])
        update_min(D_sandwich[axis]);
      else {
        if (is_ghost[2 * axis]) update_min(D_vals[2 * axis]);
        if (is_ghost[2 * axis + 1]) update_min(D_vals[2 * axis + 1]);
      }
    }
    ibm.D_rescale[list_idx] = D_rescale;

    for (int axis = 0; axis < 3; ++axis) {
      int km = 2 * axis + 1, kp = 2 * axis;
      bool sandwich = is_sandwich[axis], g_p = is_ghost[kp], g_m = is_ghost[km];
      float D_axis = sandwich ? D_sandwich[axis]
                              : (g_p ? D_vals[kp] : (g_m ? D_vals[km] : D_rescale));
      float R = D_rescale / D_axis;
      if (fabsf(D_axis) < 1e-9f) R = 1.0f;
      ibm.R_val[list_idx * 6 + kp] = R;
      ibm.R_val[list_idx * 6 + km] = R;

      if (sandwich) {
        if (SCHEME == 0) {
          ibm.K_val[list_idx * 6 + kp] = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]) * R;
          ibm.K_val[list_idx * 6 + km] = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]) * R;
          ibm.Nbc_val[list_idx * 6 + kp] =
              (poly_Nbc_pp_sw(xi_vals[km], xi_vals[kp]) + poly_Nbc_mp_sw(xi_vals[km], xi_vals[kp])) * R;
          ibm.Nbc_val[list_idx * 6 + km] =
              (poly_Nbc_pp_sw(xi_vals[kp], xi_vals[km]) + poly_Nbc_mp_sw(xi_vals[kp], xi_vals[km])) * R;
        } else {
          ibm.K_val[list_idx * 6 + kp] = poly_N_c_sandwich_avg(xi_vals[km], xi_vals[kp]) * R;
          ibm.K_val[list_idx * 6 + km] = poly_N_c_sandwich_avg(xi_vals[kp], xi_vals[km]) * R;
          ibm.Nbc_val[list_idx * 6 + kp] = (poly_Nbc_pp_sw_avg(xi_vals[km], xi_vals[kp]) +
                                            poly_Nbc_mp_sw_avg(xi_vals[km], xi_vals[kp])) * R;
          ibm.Nbc_val[list_idx * 6 + km] = (poly_Nbc_pp_sw_avg(xi_vals[kp], xi_vals[km]) +
                                            poly_Nbc_mp_sw_avg(xi_vals[kp], xi_vals[km])) * R;
        }
        ibm.M_val[list_idx * 6 + kp] = 0.0f; ibm.X_val[list_idx * 6 + kp] = 0.0f;
        ibm.M_val[list_idx * 6 + km] = 0.0f; ibm.X_val[list_idx * 6 + km] = 0.0f;
      } else {
        for (int side = 0; side < 2; ++side) {
          int kk = side == 0 ? kp : km;
          if (is_ghost[kk]) {
            if (SCHEME == 0) {
              ibm.K_val[list_idx * 6 + kk] = poly_Nc(xi_vals[kk]) * R;
              ibm.X_val[list_idx * 6 + kk] = poly_N_nb(xi_vals[kk]) * R;
              ibm.Nbc_val[list_idx * 6 + kk] = poly_Nbc(xi_vals[kk]) * R;
            } else {
              ibm.K_val[list_idx * 6 + kk] = poly_Nc_avg(xi_vals[kk]) * R;
              ibm.X_val[list_idx * 6 + kk] = poly_Nnb_avg(xi_vals[kk]) * R;
              ibm.Nbc_val[list_idx * 6 + kk] = poly_Nbc_avg(xi_vals[kk]) * R;
            }
            ibm.M_val[list_idx * 6 + kk] = 0.0f;
          } else {
            ibm.K_val[list_idx * 6 + kk] = 0.0f;
            ibm.M_val[list_idx * 6 + kk] = 1.0f;
            ibm.X_val[list_idx * 6 + kk] = 0.0f;
            ibm.Nbc_val[list_idx * 6 + kk] = 0.0f;
          }
        }
      }
      ibm.dir_code[list_idx * 6 + kp] = kp;
      ibm.dir_code[list_idx * 6 + km] = km;
    }
  } else {  // Neumann
    ibm.D_rescale[list_idx] = 1.0f;
    for (int k = 0; k < 6; ++k) {
      ibm.dir_code[list_idx * 6 + k] = k;
      ibm.R_val[list_idx * 6 + k] = 1.0f;
      ibm.K_val[list_idx * 6 + k] = is_ghost[k] ? 1.0f : 0.0f;
      ibm.M_val[list_idx * 6 + k] = is_ghost[k] ? 0.0f : 1.0f;
      ibm.X_val[list_idx * 6 + k] = 0.0f;
      ibm.Nbc_val[list_idx * 6 + k] = 0.0f;
    }
  }
}

// Gather the 7 SDF samples (centre + 6 axis neighbours at +/-1) at a staggered point of an extended-
// block cell, using the clamped extended sampler (no wrap).
__device__ inline void ibm_gather_ext(const double* sdf, int3 ext, double lx, double ly, double lz,
                                      float3 off, float& sc, float sn[6]) {
  sc = (float)ccdetail::cc_sample_ext(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
  const int d[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int k = 0; k < 6; ++k)
    sn[k] = (float)ccdetail::cc_sample_ext(sdf, ext, lx + d[k][0] + off.x, ly + d[k][1] + off.y,
                                           lz + d[k][2] + off.z);
}

// count cut cells over inner cells (sizing pass)
__global__ void ibm_count_ext_k(const double* sdf, int3 ext, int g, float3 off, int* counter) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  float sc, sn[6];
  ibm_gather_ext(sdf, ext, lx, ly, lz, off, sc, sn);
  if (ibm_is_cut(sc, sn)) atomicAdd(counter, 1);
}

// fill the IBM_Data SoA + id_map over inner cells of the extended block (no wrap)
template <int SCHEME>
__global__ void ibm_geometry_ext_k(IBM_Data ibm, int* id_map, const double* sdf, int3 ext, int g,
                                   float3 spacing, int* counter, float3 off, int bc_type) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  size_t idx = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  float sc, sn[6];
  ibm_gather_ext(sdf, ext, lx, ly, lz, off, sc, sn);
  if (!ibm_is_cut(sc, sn)) {
    id_map[idx] = -1;
    return;
  }
  int list_idx = atomicAdd(counter, 1);
  id_map[idx] = list_idx;
  ibm_fill_entry<SCHEME>(ibm, list_idx, (int)idx, sc, sn, spacing, bc_type);
}

// Bake the IBM factors into the (double) velocity diffusion stencil + inhomogeneous Dirichlet term.
// Verbatim port of modify_stencil_ibm_kernel (which is itself indexing-agnostic), retyped for double
// stencils. a_inhom[c_idx] accumulates the wall-velocity contribution to be subtracted from the RHS.
// rhs_scale[c] receives D_rescale at each cut cell (caller pre-fills 1.0 elsewhere): the Robust-Scaled
// RHS is b'_c = D_rescale * b_c - a_inhom, so the RHS at cut cells must be scaled by D_rescale to match
// the A_C *= D_rescale below -- otherwise a thin cut cell (tiny A_C, unscaled b) gives a huge velocity.
// The CONNECTIVITY provider (Cartesian 7-point). The overlay apply is otherwise mesh-agnostic: it only
// needs, per cut-cell face k, (a) the base-stencil slot the face's coefficient lives in, and (b) the
// OPPOSITE face -- the row-structured X cross-term folds face k's coefficient into its opposite. Here
// faces k=0..5 are {x+, x-, y+, y-, z+, z-}: slot order is the natural {A_E,A_W,A_N,A_S,A_T,A_B} below,
// and OPP[k] is the opposite face (== k^1 on Cartesian). An octree supplies these per overlay entry
// (variable face count, neighbour handles via ibm.dir_code, tree-defined "opposite"); the math below is
// unchanged. cc_face_term has no analogue here -- momentum is row-based, not a face flux (see ibm_fill_entry).
__global__ void ibm_modify_stencil_k(mreal* A_C, mreal* A_W, mreal* A_E, mreal* A_S, mreal* A_N,
                                     mreal* A_B, mreal* A_T, double* a_inhom, double* rhs_scale,
                                     IBM_Data ibm, float u_bc_val) {
  int list_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (list_idx >= ibm.num_active_cells) return;
  constexpr int OPP[6] = {1, 0, 3, 2, 5, 4};  // opposite face per face (Cartesian connectivity)
  int c = ibm.cell_index[list_idx];
  float descale = ibm.D_rescale[list_idx];
  if (rhs_scale) rhs_scale[c] = descale;  // double: same value scales A_C and the RHS -> ratio cancels
  double orig[6] = {A_E[c], A_W[c], A_N[c], A_S[c], A_T[c], A_B[c]};  // base coeff per face k (slot order)
  double aC = (double)A_C[c] * (double)descale;  // diagonal assembled in double, stored float
  double mod[6] = {0, 0, 0, 0, 0, 0};
  double inhom = 0.0;
  for (int k = 0; k < 6; ++k) {
    float K = ibm.K_val[list_idx * 6 + k], M = ibm.M_val[list_idx * 6 + k];
    float X = ibm.X_val[list_idx * 6 + k], Nbc = ibm.Nbc_val[list_idx * 6 + k];
    double vnb = orig[k];
    aC += vnb * K;
    inhom += (double)Nbc * u_bc_val * vnb;
    mod[k] += vnb * ((double)descale * M - 1.0);
    mod[OPP[k]] += vnb * X;  // cross-term: fold face k's coefficient into its opposite face
  }
  A_C[c] = (mreal)aC;
  A_E[c] = (mreal)(orig[0] + mod[0]);
  A_W[c] = (mreal)(orig[1] + mod[1]);
  A_N[c] = (mreal)(orig[2] + mod[2]);
  A_S[c] = (mreal)(orig[3] + mod[3]);
  A_T[c] = (mreal)(orig[4] + mod[4]);
  A_B[c] = (mreal)(orig[5] + mod[5]);
  if (a_inhom) a_inhom[c] += inhom;
}

// build the backward-Euler velocity diffusion stencil over the extended block: A_C = 1 + 6 beta,
// off-diagonals = -beta (dx=1, beta = theta*dt*nu). RHS b is the explicit term; inhom is subtracted.
__global__ void ibm_build_diffusion_k(mreal* A_C, mreal* A_W, mreal* A_E, mreal* A_S, mreal* A_N,
                                      mreal* A_B, mreal* A_T, int3 ext, double beta) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  A_C[i] = (mreal)(1.0 + 6.0 * beta);
  mreal nb = (mreal)(-beta);
  A_W[i] = nb; A_E[i] = nb; A_S[i] = nb; A_N[i] = nb; A_B[i] = nb; A_T[i] = nb;
}

__global__ void ibm_fill_k(double* a, double v, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = v;
}
__global__ void ibm_scale_k(double* a, const double* s, long n) {  // a *= s (elementwise)
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= s[i];
}

// Fluid VOLUME FRACTION for a velocity component at the staggered point: a smoothed Heaviside of the SDF,
// theta = clamp(0.5 + sdf, 0, 1) (cell size 1; sdf>0 in fluid). 1 deep in fluid, 0 deep in solid, a linear
// ramp across the one-cell interface band. The volumetric analogue of the pressure path's face openness;
// feeds the velocity multigrid's rediscretized coarse operator (smoothed momentum balance). NOT used by the
// sharp fine IBM operator -- only by the geometry-aware coarse levels.
__global__ void ibm_volfrac_k(double* theta, const double* sdf, int3 ext, float3 off) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  double sd = ccdetail::cc_sample_ext(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
  double t = 0.5 + sd;
  theta[i] = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

// Residual-restriction / prolongation mask for the IBM velocity multigrid: 1.0 only at CLEAN FLUID INTERIOR
// cells (fluid centre, no solid neighbour -> the fine row is the standard coarsenable operator), 0.0 at IBM
// cut cells AND solid cells. The coarse grid is coupled ONLY where its clean volume-fraction operator matches
// the fine operator: the row-scaled cut-cell rows and the (live-during-the-cycle, masked-only-at-the-end)
// solid rows are both excluded, so the coarse correction never overshoots the immersed-boundary band or the
// stiff 1+6*beta solid interior (the fine smoother owns both). Self-contained from the SDF (clamped sampler
// handles ghosts), independent of the inner-only id_map.
__global__ void ibm_clean_fluid_mask_k(double* m, const double* sdf, int3 ext, float3 off) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  float sc, sn[6];
  ibm_gather_ext(sdf, ext, lx, ly, lz, off, sc, sn);
  bool solid = (sc <= 0.0f);
  m[i] = (solid || ibm_is_cut(sc, sn)) ? 0.0 : 1.0;
}

// solid mask for a velocity component: 1.0 where the staggered SDF point is inside the solid, else 0.
__global__ void ibm_solid_mask_k(double* mask, const double* sdf, int3 ext, float3 off) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  int lz = blockIdx.z * blockDim.z + threadIdx.z;
  if (lx >= ext.x || ly >= ext.y || lz >= ext.z) return;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * (size_t)ext.x * ext.y;
  double sd = ccdetail::cc_sample_ext(sdf, ext, lx + off.x, ly + off.y, lz + off.z);
  mask[i] = (sd < 0.0) ? 1.0 : 0.0;
}

// one Red-Black sweep of the (IBM-modified) stencil: A_C x = b - sum(A_off x_nbr), global parity.
// Mixed precision: float matrix coefficients, double iterate x and RHS b. Each product float*double
// promotes to double and the sum/divide are in double, so the Gauss-Seidel update keeps double accuracy
// on a float-stored operator.
__global__ void ibm_rbgs_stencil_k(double* x, const double* b, const mreal* A_C, const mreal* A_W,
                                   const mreal* A_E, const mreal* A_S, const mreal* A_N,
                                   const mreal* A_B, const mreal* A_T, const double* solidmask, int3 ext,
                                   int3 og, int g, int color) {
  int lx = blockIdx.x * blockDim.x + threadIdx.x + g;
  int ly = blockIdx.y * blockDim.y + threadIdx.y + g;
  int lz = blockIdx.z * blockDim.z + threadIdx.z + g;
  if (lx >= ext.x - g || ly >= ext.y - g || lz >= ext.z - g) return;
  if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color) return;
  size_t sx = 1, sy = ext.x, sz = (size_t)ext.x * ext.y;
  size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * sz;
  // Pin solid cells to 0 instead of running the decoupled 1+6*beta diffusion on them. The cut-cell rows
  // already zero every fluid->solid coupling, so the fluid update is unchanged (bit-identical); this just
  // keeps the slaved solid DOFs at 0 throughout the sweeps rather than masking them only at the end.
  if (solidmask && solidmask[i] > 0.5) {
    x[i] = 0.0;
    return;
  }
  double ac = A_C[i];
  if (fabs(ac) < 1e-30) return;
  double s = (double)A_E[i] * x[i + sx] + (double)A_W[i] * x[i - sx] + (double)A_N[i] * x[i + sy] +
             (double)A_S[i] * x[i - sy] + (double)A_T[i] * x[i + sz] + (double)A_B[i] * x[i - sz];
  x[i] = (b[i] - s) / ac;
}

}  // namespace ibmdetail

// allocate / free the IBM_Data SoA for `n` cut cells (host helpers)
inline IBM_Data ibm_alloc(int n) {
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
inline void ibm_free(IBM_Data& d) {
  for (void* p : {(void*)d.cell_index, (void*)d.D_rescale, (void*)d.num_boundaries, (void*)d.dir_code,
                  (void*)d.K_val, (void*)d.M_val, (void*)d.X_val, (void*)d.Nbc_val, (void*)d.R_val})
    if (p) cudaFree(p);
  d = IBM_Data{};
}

}  // namespace cfdmpi
