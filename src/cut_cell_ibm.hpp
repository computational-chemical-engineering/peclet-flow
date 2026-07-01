/// @file
/// @brief flow — portable (Kokkos) Robust-Scaled cut-cell IBM primitives + per-cut-cell overlay build.
///
/// Kokkos port of cut_cell_ibm.cuh (the boundary-distance polynomials) and ibm_fill_entry from
/// mac_ibm.cuh (the per-cut-cell stencil-modification factors K/M/X/Nbc/R + D_rescale). Faithful copy
/// of the Dirichlet/Neumann, point-value(SCHEME 0)/cell-average(1), and sandwiched (double-sided) cases.
/// Output factors are written into Kokkos Views (SoA, [list_idx*6+k]); the build kernel fills one entry
/// per cut cell. KOKKOS_INLINE_FUNCTION so the math is shared with the host reference.
#ifndef PECLET_FLOW_CUT_CELL_IBM_HPP
#define PECLET_FLOW_CUT_CELL_IBM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace peclet::flow {

using IMem = Kokkos::DefaultExecutionSpace::memory_space;

// ---- boundary-distance polynomials (verbatim from cut_cell_ibm.cuh) ----
KOKKOS_INLINE_FUNCTION float poly_D(float xi) { return xi * (1.0f + xi); }
KOKKOS_INLINE_FUNCTION float poly_N_nb(float xi) { return xi * (1.0f - xi); }
KOKKOS_INLINE_FUNCTION float poly_Nc(float xi) { return 2.0f * (xi * xi - 1.0f); }
KOKKOS_INLINE_FUNCTION float poly_Nbc(float) { return 2.0f; }
KOKKOS_INLINE_FUNCTION float poly_D_avg(float xi) { return xi * (1.0f + xi) - 1.0f / 12.0f; }
KOKKOS_INLINE_FUNCTION float poly_Nnb_avg(float xi) { return xi * (1.0f - xi) + 1.0f / 12.0f; }
KOKKOS_INLINE_FUNCTION float poly_Nc_avg(float xi) { return 2.0f * (xi * xi - 1.0f) - 1.0f / 6.0f; }
KOKKOS_INLINE_FUNCTION float poly_Nbc_avg(float) { return 2.0f; }
KOKKOS_INLINE_FUNCTION float poly_D_sandwich(float xi_m, float xi_p) { return xi_m * xi_p; }
KOKKOS_INLINE_FUNCTION float poly_N_c_sandwich(float xi_m, float xi_p) { return (xi_m + 1.0f) * (xi_p - 1.0f); }
KOKKOS_INLINE_FUNCTION float poly_Nbc_pp_sw(float xi_m, float xi_p) { return (xi_m / (xi_m + xi_p)) * (1.0f + xi_m); }
KOKKOS_INLINE_FUNCTION float poly_Nbc_mp_sw(float xi_m, float xi_p) { return (xi_p / (xi_m + xi_p)) * (1.0f - xi_p); }
KOKKOS_INLINE_FUNCTION float poly_D_sandwich_avg(float xi_m, float xi_p) { return xi_m * xi_p - 1.0f / 12.0f; }
KOKKOS_INLINE_FUNCTION float poly_N_c_sandwich_avg(float xi_m, float xi_p) { return (xi_m + 1.0f) * (xi_p - 1.0f) - 1.0f / 12.0f; }
KOKKOS_INLINE_FUNCTION float poly_Nbc_pp_sw_avg(float xi_m, float xi_p) { return (xi_m / (xi_m + xi_p)) * (1.0f + xi_m) - 1.0f / 12.0f; }
KOKKOS_INLINE_FUNCTION float poly_Nbc_mp_sw_avg(float xi_m, float xi_p) { return (xi_p / (xi_m + xi_p)) * (1.0f - xi_p) + 1.0f / 12.0f; }

// IBM overlay output (SoA Views; per-direction arrays are size 6*num_cells). Templated on the memory
// space so the device build and a HostSpace reference share the same fill code.
template <class Space>
struct IbmOverlayT {
  Kokkos::View<int*, Space> cell_index;
  Kokkos::View<int*, Space> num_boundaries;
  Kokkos::View<float*, Space> D_rescale;
  Kokkos::View<int*, Space> dir_code;
  Kokkos::View<float*, Space> K_val, M_val, X_val, Nbc_val, R_val;
};
using IbmOverlay = IbmOverlayT<IMem>;

// Fill one overlay entry (list_idx) for a cut cell from its 7 SDF samples. Verbatim port of
// ibm_fill_entry<SCHEME>. bc_type: 0 = Dirichlet, 1 = Neumann.
template <int SCHEME, class OV>
KOKKOS_INLINE_FUNCTION void ibmFillEntry(const OV& o, int list_idx, int c_idx, float sdf_c,
                                         const float sdf_n[6], int bc_type) {
  o.cell_index(list_idx) = c_idx;
  o.num_boundaries(list_idx) = 6;
  bool is_ghost[6];
  float xi_vals[6], D_vals[6];
  for (int k = 0; k < 6; ++k) {
    if (sdf_n[k] < 0.0f) {
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
    bool is_sandwich[3] = {is_ghost[0] && is_ghost[1], is_ghost[2] && is_ghost[3], is_ghost[4] && is_ghost[5]};
    float D_sandwich[3] = {0, 0, 0};
    for (int a = 0; a < 3; ++a)
      if (is_sandwich[a])
        D_sandwich[a] = (SCHEME == 0) ? poly_D_sandwich(xi_vals[2 * a + 1], xi_vals[2 * a])
                                      : poly_D_sandwich_avg(xi_vals[2 * a + 1], xi_vals[2 * a]);
    float min_D_abs = 1e30f, D_rescale = 1.0f;
    auto update_min = [&](float val) { if (Kokkos::fabs(val) < min_D_abs) { min_D_abs = Kokkos::fabs(val); D_rescale = val; } };
    for (int axis = 0; axis < 3; ++axis) {
      if (is_sandwich[axis]) update_min(D_sandwich[axis]);
      else { if (is_ghost[2 * axis]) update_min(D_vals[2 * axis]); if (is_ghost[2 * axis + 1]) update_min(D_vals[2 * axis + 1]); }
    }
    o.D_rescale(list_idx) = D_rescale;

    for (int axis = 0; axis < 3; ++axis) {
      int km = 2 * axis + 1, kp = 2 * axis;
      bool sandwich = is_sandwich[axis], g_p = is_ghost[kp], g_m = is_ghost[km];
      float D_axis = sandwich ? D_sandwich[axis] : (g_p ? D_vals[kp] : (g_m ? D_vals[km] : D_rescale));
      float R = D_rescale / D_axis;
      if (Kokkos::fabs(D_axis) < 1e-9f) R = 1.0f;
      o.R_val(list_idx * 6 + kp) = R;
      o.R_val(list_idx * 6 + km) = R;
      if (sandwich) {
        if (SCHEME == 0) {
          o.K_val(list_idx * 6 + kp) = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]) * R;
          o.K_val(list_idx * 6 + km) = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]) * R;
          o.Nbc_val(list_idx * 6 + kp) = (poly_Nbc_pp_sw(xi_vals[km], xi_vals[kp]) + poly_Nbc_mp_sw(xi_vals[km], xi_vals[kp])) * R;
          o.Nbc_val(list_idx * 6 + km) = (poly_Nbc_pp_sw(xi_vals[kp], xi_vals[km]) + poly_Nbc_mp_sw(xi_vals[kp], xi_vals[km])) * R;
        } else {
          o.K_val(list_idx * 6 + kp) = poly_N_c_sandwich_avg(xi_vals[km], xi_vals[kp]) * R;
          o.K_val(list_idx * 6 + km) = poly_N_c_sandwich_avg(xi_vals[kp], xi_vals[km]) * R;
          o.Nbc_val(list_idx * 6 + kp) = (poly_Nbc_pp_sw_avg(xi_vals[km], xi_vals[kp]) + poly_Nbc_mp_sw_avg(xi_vals[km], xi_vals[kp])) * R;
          o.Nbc_val(list_idx * 6 + km) = (poly_Nbc_pp_sw_avg(xi_vals[kp], xi_vals[km]) + poly_Nbc_mp_sw_avg(xi_vals[kp], xi_vals[km])) * R;
        }
        o.M_val(list_idx * 6 + kp) = 0.0f; o.X_val(list_idx * 6 + kp) = 0.0f;
        o.M_val(list_idx * 6 + km) = 0.0f; o.X_val(list_idx * 6 + km) = 0.0f;
      } else {
        for (int side = 0; side < 2; ++side) {
          int kk = side == 0 ? kp : km;
          if (is_ghost[kk]) {
            if (SCHEME == 0) {
              o.K_val(list_idx * 6 + kk) = poly_Nc(xi_vals[kk]) * R;
              o.X_val(list_idx * 6 + kk) = poly_N_nb(xi_vals[kk]) * R;
              o.Nbc_val(list_idx * 6 + kk) = poly_Nbc(xi_vals[kk]) * R;
            } else {
              o.K_val(list_idx * 6 + kk) = poly_Nc_avg(xi_vals[kk]) * R;
              o.X_val(list_idx * 6 + kk) = poly_Nnb_avg(xi_vals[kk]) * R;
              o.Nbc_val(list_idx * 6 + kk) = poly_Nbc_avg(xi_vals[kk]) * R;
            }
            o.M_val(list_idx * 6 + kk) = 0.0f;
          } else {
            o.K_val(list_idx * 6 + kk) = 0.0f; o.M_val(list_idx * 6 + kk) = 1.0f;
            o.X_val(list_idx * 6 + kk) = 0.0f; o.Nbc_val(list_idx * 6 + kk) = 0.0f;
          }
        }
      }
      o.dir_code(list_idx * 6 + kp) = kp;
      o.dir_code(list_idx * 6 + km) = km;
    }
  } else {  // Neumann
    o.D_rescale(list_idx) = 1.0f;
    for (int k = 0; k < 6; ++k) {
      o.dir_code(list_idx * 6 + k) = k;
      o.R_val(list_idx * 6 + k) = 1.0f;
      o.K_val(list_idx * 6 + k) = is_ghost[k] ? 1.0f : 0.0f;
      o.M_val(list_idx * 6 + k) = is_ghost[k] ? 0.0f : 1.0f;
      o.X_val(list_idx * 6 + k) = 0.0f;
      o.Nbc_val(list_idx * 6 + k) = 0.0f;
    }
  }
}

// Build the backward-Euler velocity diffusion stencil over the extended block (divided convention):
// A_C = idiag + 6*beta, off-diagonals = -beta (dx=1). idiag = 1/dt, beta = nu.
inline void ibmBuildDiffusion(Kokkos::View<float*, IMem> AC, Kokkos::View<float*, IMem> AW,
                              Kokkos::View<float*, IMem> AE, Kokkos::View<float*, IMem> AS,
                              Kokkos::View<float*, IMem> AN, Kokkos::View<float*, IMem> AB,
                              Kokkos::View<float*, IMem> AT, int ex, int ey, int ez, double beta,
                              double idiag) {
  Kokkos::DefaultExecutionSpace space;
  const std::size_t n = (std::size_t)ex * ey * ez;
  const float nb = (float)(-beta), c = (float)(idiag + 6.0 * beta);
  Kokkos::parallel_for(
      "peclet::flow::ibm_build_diff", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        AC(i) = c; AW(i) = nb; AE(i) = nb; AS(i) = nb; AN(i) = nb; AB(i) = nb; AT(i) = nb;
      });

}

// Apply the Robust-Scaled overlay to the momentum stencil at each cut cell (port of
// ibm_modify_stencil_k): modify A_C / 6 off-diagonals + accumulate the inhomogeneous (wall-velocity)
// term and store the row scaling. Each cut cell owns a distinct grid index c -> no races.
inline void ibmModifyStencil(Kokkos::View<float*, IMem> AC, Kokkos::View<float*, IMem> AW,
                             Kokkos::View<float*, IMem> AE, Kokkos::View<float*, IMem> AS,
                             Kokkos::View<float*, IMem> AN, Kokkos::View<float*, IMem> AB,
                             Kokkos::View<float*, IMem> AT, Kokkos::View<double*, IMem> a_inhom,
                             Kokkos::View<double*, IMem> rhs_scale, const IbmOverlay& ibm,
                             int numActive, float u_bc_val) {
  Kokkos::DefaultExecutionSpace space;
  const bool hasInhom = (a_inhom.extent(0) != 0), hasScale = (rhs_scale.extent(0) != 0);
  Kokkos::parallel_for(
      "peclet::flow::ibm_modify", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, numActive),
      KOKKOS_LAMBDA(int list_idx) {
        const int OPP[6] = {1, 0, 3, 2, 5, 4};
        const int c = ibm.cell_index(list_idx);
        const float descale = ibm.D_rescale(list_idx);
        if (hasScale) rhs_scale(c) = descale;
        const double orig[6] = {AE(c), AW(c), AN(c), AS(c), AT(c), AB(c)};
        double aC = (double)AC(c) * (double)descale;
        double mod[6] = {0, 0, 0, 0, 0, 0};
        double inhom = 0.0;
        for (int k = 0; k < 6; ++k) {
          const float K = ibm.K_val(list_idx * 6 + k), M = ibm.M_val(list_idx * 6 + k);
          const float X = ibm.X_val(list_idx * 6 + k), Nbc = ibm.Nbc_val(list_idx * 6 + k);
          const double vnb = orig[k];
          aC += vnb * K;
          inhom += (double)Nbc * u_bc_val * vnb;
          mod[k] += vnb * ((double)descale * M - 1.0);
          mod[OPP[k]] += vnb * X;
        }
        AC(c) = (float)aC;
        AE(c) = (float)(orig[0] + mod[0]);
        AW(c) = (float)(orig[1] + mod[1]);
        AN(c) = (float)(orig[2] + mod[2]);
        AS(c) = (float)(orig[3] + mod[3]);
        AT(c) = (float)(orig[4] + mod[4]);
        AB(c) = (float)(orig[5] + mod[5]);
        if (hasInhom) a_inhom(c) += inhom;
      });

}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_CUT_CELL_IBM_HPP
