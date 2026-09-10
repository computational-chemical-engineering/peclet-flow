/// @file
/// @brief flow — portable (Kokkos) Robust-Scaled cut-cell IBM primitives + per-cut-cell overlay
/// build.
///
/// Kokkos port of cut_cell_ibm.cuh (the boundary-distance polynomials) and ibm_fill_entry from
/// mac_ibm.cuh (the per-cut-cell stencil-modification factors K/M/X/Nbc/R + D_rescale). Faithful
/// copy of the Dirichlet/Neumann, point-value(SCHEME 0)/cell-average(1), and sandwiched
/// (double-sided) cases. Output factors are written into Kokkos Views (SoA, [list_idx*6+k]); the
/// build kernel fills one entry per cut cell. KOKKOS_INLINE_FUNCTION so the math is shared with the
/// host reference.
///
/// QUALITY_PLAN G.6 (precision as a typed policy): the closure polynomials and the overlay are
/// templated on `Real` (the operator storage type, `IbmSolver::mreal` = `MReal`). The default
/// build instantiates every one of these at `Real = float`, so the generated code -- same
/// operations, same integer-exact literals -- is BIT-IDENTICAL to the pre-G.6 hardcoded-float
/// version; a `-DPECLET_FLOW_OPERATOR_DOUBLE` build instantiates the SAME templates at
/// `Real = double`, so the overlay build no longer narrows the SDF/theta samples (and every
/// K/M/X/Nbc/R/D_rescale factor derived from them) to float before storing them.
#ifndef PECLET_FLOW_CUT_CELL_IBM_HPP
#define PECLET_FLOW_CUT_CELL_IBM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace peclet::flow {

using IMem = Kokkos::DefaultExecutionSpace::memory_space;

// ---- boundary-distance polynomials (verbatim from cut_cell_ibm.cuh, templated on Real: G.6) ----
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_D(Real xi) {
  return xi * (Real(1) + xi);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_N_nb(Real xi) {
  return xi * (Real(1) - xi);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nc(Real xi) {
  return Real(2) * (xi * xi - Real(1));
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc(Real) {
  return Real(2);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_D_avg(Real xi) {
  return xi * (Real(1) + xi) - Real(1) / Real(12);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nnb_avg(Real xi) {
  return xi * (Real(1) - xi) + Real(1) / Real(12);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nc_avg(Real xi) {
  return Real(2) * (xi * xi - Real(1)) - Real(1) / Real(6);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc_avg(Real) {
  return Real(2);
}
// ---- Navier-slip generalization of the one-sided Dirichlet polynomials (WO-V6b) ----
//
// The shipped closure builds the quadratic p through (u_m at -1, u_c at 0, u_g at +1) and imposes
// the DIRICHLET condition p(theta) = u_b at the wall crossing. The Navier condition on the
// TANGENTIAL velocity is the Robin condition
//
//     u_t(wall) - u_body = lambda * d(u_t)/dn        (n = the INWARD, fluid-side normal)
//
// which along the axis (solid on the +x side, so n = -x) reads  p(theta) + lam*p'(theta) = u_b,
// with `lam` the slip length measured ALONG THIS AXIS (see ibmBuildOverlay: lam = s*lambda/|n_a|,
// s = 1 - n_c^2 the tangential weight of the component this DOF carries). Solving for u_g with
//     P = theta + lam,   Q = theta^2 + 2*lam*theta
// gives  (P+Q) u_g = 2 u_b - 2 u_c (1-Q) + u_m (P-Q), i.e. exactly the shipped polynomials with
// three additive lam-terms and an UNCHANGED Nbc = 2:
//
//     D   = theta(1+theta)   + lam(1+2 theta)
//     X   = theta(1-theta)   + lam(1-2 theta)
//     K   = 2(theta^2-1)     + 4 lam theta
//
// lam = 0 reproduces poly_D / poly_N_nb / poly_Nc identically (the call sites branch on lam > 0 so
// that no extra rounding is introduced at all). lam -> infinity gives the free-slip (zero
// normal-derivative) closure: at theta = 1/2, u_g -> u_c.
//
// NOTE the closure is stored in FLOAT by default (double under -DPECLET_FLOW_OPERATOR_DOUBLE): the
// Robin datum is indistinguishable from no-slip once lam*(1+2 theta) drops below the storage
// epsilon times D(theta), i.e. below lam ~ 3e-8 cells in the float build (measured floor in the
// findings). Every physically meaningful lambda is orders of magnitude above it.
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_D_slip(Real xi, Real lam) {
  return poly_D(xi) + lam * (Real(1) + Real(2) * xi);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_N_nb_slip(Real xi, Real lam) {
  return poly_N_nb(xi) + lam * (Real(1) - Real(2) * xi);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nc_slip(Real xi, Real lam) {
  return poly_Nc(xi) + Real(4) * lam * xi;
}

template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_D_sandwich(Real xi_m, Real xi_p) {
  return xi_m * xi_p;
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_N_c_sandwich(Real xi_m, Real xi_p) {
  return (xi_m + Real(1)) * (xi_p - Real(1));
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc_pp_sw(Real xi_m, Real xi_p) {
  return (xi_m / (xi_m + xi_p)) * (Real(1) + xi_m);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc_mp_sw(Real xi_m, Real xi_p) {
  return (xi_p / (xi_m + xi_p)) * (Real(1) - xi_p);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_D_sandwich_avg(Real xi_m, Real xi_p) {
  return xi_m * xi_p - Real(1) / Real(12);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_N_c_sandwich_avg(Real xi_m, Real xi_p) {
  return (xi_m + Real(1)) * (xi_p - Real(1)) - Real(1) / Real(12);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc_pp_sw_avg(Real xi_m, Real xi_p) {
  return (xi_m / (xi_m + xi_p)) * (Real(1) + xi_m) - Real(1) / Real(12);
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real poly_Nbc_mp_sw_avg(Real xi_m, Real xi_p) {
  return (xi_p / (xi_m + xi_p)) * (Real(1) - xi_p) + Real(1) / Real(12);
}

// IBM overlay output (SoA Views; per-direction arrays are size 6*num_cells). Templated on the
// memory space AND the storage precision (G.6) so the device build and a HostSpace reference
// share the same fill code, and a `-DPECLET_FLOW_OPERATOR_DOUBLE` build carries the overlay in
// double end to end instead of narrowing it to float at this SoA.
template <class Space, class Real = float>
struct IbmOverlayT {
  using value_type = Real;
  Kokkos::View<int*, Space> cell_index;
  Kokkos::View<int*, Space> num_boundaries;
  Kokkos::View<Real*, Space> D_rescale;
  Kokkos::View<int*, Space> dir_code;
  Kokkos::View<Real*, Space> K_val, M_val, X_val, Nbc_val, R_val;
};
// Float alias kept for callers that have not opted into a typed overlay (none left in-tree after
// G.6; mac_ibm.hpp defines the canonical `IbmOverlay = IbmOverlayT<IMem, mreal>`, mreal = MReal).
template <class Real>
using IbmOverlayReal = IbmOverlayT<IMem, Real>;

// Fill one overlay entry (list_idx) for a cut cell from its 7 SDF samples. Verbatim port of
// ibm_fill_entry<SCHEME>. bc_type: 0 = Dirichlet, 1 = Neumann. thEx (optional, may be nullptr):
// per-direction EXACT wall-crossing fractions theta from the cut cell toward each of the 6
// neighbours (analytic-SDF capability, setExactCrossings) — a finite thEx[k] overrides the
// linear-interpolated theta; non-finite entries fall back. `Real` (the overlay's own storage
// type, `OV::value_type`) is deduced from `o` — every existing call site is unchanged.
template <int SCHEME, class OV>
KOKKOS_INLINE_FUNCTION void ibmFillEntry(const OV& o, int list_idx, int c_idx,
                                         typename OV::value_type sdf_c,
                                         const typename OV::value_type sdf_n[6], int bc_type,
                                         const typename OV::value_type* thEx,
                                         const typename OV::value_type* lamAxis = nullptr,
                                         int* sandwichSkipped = nullptr) {
  using Real = typename OV::value_type;
  o.cell_index(list_idx) = c_idx;
  o.num_boundaries(list_idx) = 6;
  bool is_ghost[6];
  Real xi_vals[6], D_vals[6];
  for (int k = 0; k < 6; ++k) {
    if (sdf_n[k] < Real(0)) {
      is_ghost[k] = true;
      if (bc_type == 0) {
        Real theta = sdf_c / (sdf_c - sdf_n[k]);
        if (thEx != nullptr && Kokkos::isfinite(thEx[k]))
          theta = thEx[k];
        if (theta < Real(1e-4))
          theta = Real(1e-4);
        if (theta > Real(1))
          theta = Real(1);
        xi_vals[k] = theta;
        const Real lam = (lamAxis != nullptr && SCHEME == 0) ? lamAxis[k >> 1] : Real(0);
        D_vals[k] = lam > Real(0) ? poly_D_slip(theta, lam)
                                  : ((SCHEME == 0) ? poly_D(theta) : poly_D_avg(theta));
      } else {
        xi_vals[k] = Real(0.5);
        D_vals[k] = Real(1);
      }
    } else {
      is_ghost[k] = false;
      xi_vals[k] = Real(1);
      D_vals[k] = Real(1e9);
    }
  }

  if (bc_type == 0) {
    bool is_sandwich[3] = {is_ghost[0] && is_ghost[1], is_ghost[2] && is_ghost[3],
                           is_ghost[4] && is_ghost[5]};
    Real D_sandwich[3] = {Real(0), Real(0), Real(0)};
    for (int a = 0; a < 3; ++a)
      if (is_sandwich[a])
        D_sandwich[a] = (SCHEME == 0) ? poly_D_sandwich(xi_vals[2 * a + 1], xi_vals[2 * a])
                                      : poly_D_sandwich_avg(xi_vals[2 * a + 1], xi_vals[2 * a]);
    Real min_D_abs = Real(1e30), D_rescale = Real(1);
    auto update_min = [&](Real val) {
      if (Kokkos::fabs(val) < min_D_abs) {
        min_D_abs = Kokkos::fabs(val);
        D_rescale = val;
      }
    };
    for (int axis = 0; axis < 3; ++axis) {
      if (is_sandwich[axis])
        update_min(D_sandwich[axis]);
      else {
        if (is_ghost[2 * axis])
          update_min(D_vals[2 * axis]);
        if (is_ghost[2 * axis + 1])
          update_min(D_vals[2 * axis + 1]);
      }
    }
    o.D_rescale(list_idx) = D_rescale;

    for (int axis = 0; axis < 3; ++axis) {
      int km = 2 * axis + 1, kp = 2 * axis;
      bool sandwich = is_sandwich[axis], g_p = is_ghost[kp], g_m = is_ghost[km];
      // SANDWICHED axis (both neighbours solid = a one-cell fluid gap): the Navier closure is NOT
      // applied. A slip length is a sub-cell wall model and a gap that a single cell spans does
      // not resolve one; the axis keeps its validated no-slip Dirichlet closure. Counted (not
      // silent) through sandwichSkipped so a geometry where it matters is visible.
      if (sandwich && lamAxis != nullptr && lamAxis[axis] > Real(0) && sandwichSkipped != nullptr)
        Kokkos::atomic_fetch_add(sandwichSkipped, 1);
      Real D_axis =
          sandwich ? D_sandwich[axis] : (g_p ? D_vals[kp] : (g_m ? D_vals[km] : D_rescale));
      Real R = D_rescale / D_axis;
      if (Kokkos::fabs(D_axis) < Real(1e-9))
        R = Real(1);
      o.R_val(list_idx * 6 + kp) = R;
      o.R_val(list_idx * 6 + km) = R;
      if (sandwich) {
        if (SCHEME == 0) {
          o.K_val(list_idx * 6 + kp) = poly_N_c_sandwich(xi_vals[km], xi_vals[kp]) * R;
          o.K_val(list_idx * 6 + km) = poly_N_c_sandwich(xi_vals[kp], xi_vals[km]) * R;
          o.Nbc_val(list_idx * 6 + kp) = (poly_Nbc_pp_sw(xi_vals[km], xi_vals[kp]) +
                                          poly_Nbc_mp_sw(xi_vals[km], xi_vals[kp])) *
                                         R;
          o.Nbc_val(list_idx * 6 + km) = (poly_Nbc_pp_sw(xi_vals[kp], xi_vals[km]) +
                                          poly_Nbc_mp_sw(xi_vals[kp], xi_vals[km])) *
                                         R;
        } else {
          o.K_val(list_idx * 6 + kp) = poly_N_c_sandwich_avg(xi_vals[km], xi_vals[kp]) * R;
          o.K_val(list_idx * 6 + km) = poly_N_c_sandwich_avg(xi_vals[kp], xi_vals[km]) * R;
          o.Nbc_val(list_idx * 6 + kp) = (poly_Nbc_pp_sw_avg(xi_vals[km], xi_vals[kp]) +
                                          poly_Nbc_mp_sw_avg(xi_vals[km], xi_vals[kp])) *
                                         R;
          o.Nbc_val(list_idx * 6 + km) = (poly_Nbc_pp_sw_avg(xi_vals[kp], xi_vals[km]) +
                                          poly_Nbc_mp_sw_avg(xi_vals[kp], xi_vals[km])) *
                                         R;
        }
        o.M_val(list_idx * 6 + kp) = Real(0);
        o.X_val(list_idx * 6 + kp) = Real(0);
        o.M_val(list_idx * 6 + km) = Real(0);
        o.X_val(list_idx * 6 + km) = Real(0);
      } else {
        for (int side = 0; side < 2; ++side) {
          int kk = side == 0 ? kp : km;
          if (is_ghost[kk]) {
            const Real lam = (lamAxis != nullptr && SCHEME == 0) ? lamAxis[axis] : Real(0);
            if (lam > Real(0)) {
              o.K_val(list_idx * 6 + kk) = poly_Nc_slip(xi_vals[kk], lam) * R;
              o.X_val(list_idx * 6 + kk) = poly_N_nb_slip(xi_vals[kk], lam) * R;
              o.Nbc_val(list_idx * 6 + kk) = poly_Nbc(xi_vals[kk]) * R;
            } else if (SCHEME == 0) {
              o.K_val(list_idx * 6 + kk) = poly_Nc(xi_vals[kk]) * R;
              o.X_val(list_idx * 6 + kk) = poly_N_nb(xi_vals[kk]) * R;
              o.Nbc_val(list_idx * 6 + kk) = poly_Nbc(xi_vals[kk]) * R;
            } else {
              o.K_val(list_idx * 6 + kk) = poly_Nc_avg(xi_vals[kk]) * R;
              o.X_val(list_idx * 6 + kk) = poly_Nnb_avg(xi_vals[kk]) * R;
              o.Nbc_val(list_idx * 6 + kk) = poly_Nbc_avg(xi_vals[kk]) * R;
            }
            o.M_val(list_idx * 6 + kk) = Real(0);
          } else {
            o.K_val(list_idx * 6 + kk) = Real(0);
            o.M_val(list_idx * 6 + kk) = Real(1);
            o.X_val(list_idx * 6 + kk) = Real(0);
            o.Nbc_val(list_idx * 6 + kk) = Real(0);
          }
        }
      }
      o.dir_code(list_idx * 6 + kp) = kp;
      o.dir_code(list_idx * 6 + km) = km;
    }
  } else {  // Neumann
    o.D_rescale(list_idx) = Real(1);
    for (int k = 0; k < 6; ++k) {
      o.dir_code(list_idx * 6 + k) = k;
      o.R_val(list_idx * 6 + k) = Real(1);
      o.K_val(list_idx * 6 + k) = is_ghost[k] ? Real(1) : Real(0);
      o.M_val(list_idx * 6 + k) = is_ghost[k] ? Real(0) : Real(1);
      o.X_val(list_idx * 6 + k) = Real(0);
      o.Nbc_val(list_idx * 6 + k) = Real(0);
    }
  }
}

// Sampled-theta entry point (the historical signature; all existing call sites unchanged).
template <int SCHEME, class OV>
KOKKOS_INLINE_FUNCTION void ibmFillEntry(const OV& o, int list_idx, int c_idx,
                                         typename OV::value_type sdf_c,
                                         const typename OV::value_type sdf_n[6], int bc_type) {
  ibmFillEntry<SCHEME>(o, list_idx, c_idx, sdf_c, sdf_n, bc_type, nullptr, nullptr, nullptr);
}

// Build the backward-Euler velocity diffusion stencil over the extended block (divided convention):
// A_C = idiag + 2*((bx+by)+bz), off-diagonals = -b_a per derivative axis a. idiag = rho/dt,
// b_a = mu' * w_a with w_a = 1/h_a'^2 the per-axis metric weight (doc/anisotropic_metric.md §2).
//
// ASSOCIATION ORDER IS PART OF THE CONTRACT (note trap 3).  With bx == by == bz == beta:
// (beta+beta) is exact, +beta rounds once to fl(3*beta), and the *2 is exact, so the diagonal is
// fl(6*beta) == 6.0*beta to the last bit — the isotropic operator is BIT-IDENTICAL to the
// pre-Phase-2 `idiag + 6.0*beta`.  Any other grouping breaks that.
template <class MV>
inline void ibmBuildDiffusion(MV AC, MV AW, MV AE, MV AS, MV AN, MV AB, MV AT, int ex, int ey,
                              int ez, double bx, double by, double bz, double idiag) {
  Kokkos::DefaultExecutionSpace space;
  const std::size_t n = (std::size_t)ex * ey * ez;
  using MVreal = typename MV::non_const_value_type;
  const MVreal nbx = (MVreal)(-bx), nby = (MVreal)(-by), nbz = (MVreal)(-bz),
               c = (MVreal)(idiag + 2.0 * ((bx + by) + bz));
  Kokkos::parallel_for(
      "peclet::flow::ibm_build_diff", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        AC(i) = c;
        AW(i) = nbx;
        AE(i) = nbx;
        AS(i) = nby;
        AN(i) = nby;
        AB(i) = nbz;
        AT(i) = nbz;
      });
}

// Isotropic spelling (b_x = b_y = b_z = beta), kept for the kernel-level tests and any caller
// with no metric: bit-identical to the pre-Phase-2 kernel by the association note above.
template <class MV>
inline void ibmBuildDiffusion(MV AC, MV AW, MV AE, MV AS, MV AN, MV AB, MV AT, int ex, int ey,
                              int ez, double beta, double idiag) {
  ibmBuildDiffusion(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, beta, beta, beta, idiag);
}

// Variable-viscosity backward-Euler diffusion stencil (sibling of ibmBuildDiffusion): the face
// off-diagonal is -beta_face (per-face viscosity from FaceProps) and A_C = idiag(i) + sum of the 6
// face betas. Built over INNER cells (neighbour mu at i+-stride must be valid — fill the mu ghosts
// first). Face means are computed in double, cast to float once (mirroring the constant path).
// FaceProps: UniformFaceProps reproduces the constant operator; FieldFaceProps reads a mu field.
template <class FaceProps, class MV>
inline void ibmBuildDiffusionVar(MV AC, MV AW, MV AE, MV AS, MV AN, MV AB, MV AT, int ex, int ey,
                                 int ez, int g, FaceProps fp, double wx = 1.0, double wy = 1.0,
                                 double wz = 1.0) {
  Kokkos::DefaultExecutionSpace space;
  using MD = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::ibm_build_diff_var", MD(space, {g, g, g}, {ex - g, ey - g, ez - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = ex, sz = (long)ex * ey;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        // The face viscosity times the metric weight w_a of the face's OWN axis
        // (doc/anisotropic_metric.md §2); w == 1.0 exactly on the isotropic path.
        const double bw = fp.beta(i, i - sx) * wx, be = fp.beta(i, i + sx) * wx;
        const double bs = fp.beta(i, i - sy) * wy, bn = fp.beta(i, i + sy) * wy;
        const double bb = fp.beta(i, i - sz) * wz, bt = fp.beta(i, i + sz) * wz;
        AW(i) = (typename MV::non_const_value_type)(-bw);
        AE(i) = (typename MV::non_const_value_type)(-be);
        AS(i) = (typename MV::non_const_value_type)(-bs);
        AN(i) = (typename MV::non_const_value_type)(-bn);
        AB(i) = (typename MV::non_const_value_type)(-bb);
        AT(i) = (typename MV::non_const_value_type)(-bt);
        AC(i) = (typename MV::non_const_value_type)(fp.idiag(i) + bw + be + bs + bn + bb + bt);
      });
}

// Apply the Robust-Scaled overlay to the momentum stencil at each cut cell (port of
// ibm_modify_stencil_k): modify A_C / 6 off-diagonals + accumulate the inhomogeneous
// (wall-velocity) term and store the row scaling. Each cut cell owns a distinct grid index c -> no
// races. `OV` (deduced): the overlay's own storage type is `OV::value_type` (G.6).
// `u_bc` (optional): per-cell wall velocity of THIS component, on the extended block -- the
// kinematic no-slip datum for MOVING geometry (Layer 3 rung 2). An EMPTY View falls back to the
// scalar u_bc_val, which is what keeps a static solver bit-identical: the accumulated term is
// (double)Nbc * 0.0f * vnb either way, the same three roundings in the same order.
template <class MV, class OV>
inline void ibmModifyStencil(
    MV AC, MV AW, MV AE, MV AS, MV AN, MV AB, MV AT, Kokkos::View<double*, IMem> a_inhom,
    Kokkos::View<double*, IMem> rhs_scale, const OV& ibm, int numActive,
    typename OV::value_type u_bc_val,
    Kokkos::View<const double*, IMem> u_bc = Kokkos::View<const double*, IMem>()) {
  Kokkos::DefaultExecutionSpace space;
  const bool hasInhom = (a_inhom.extent(0) != 0), hasScale = (rhs_scale.extent(0) != 0);
  const bool hasWallVel = (u_bc.extent(0) != 0);
  Kokkos::parallel_for(
      "peclet::flow::ibm_modify", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, numActive),
      KOKKOS_LAMBDA(int list_idx) {
        const int OPP[6] = {1, 0, 3, 2, 5, 4};
        const int c = ibm.cell_index(list_idx);
        const auto descale = ibm.D_rescale(list_idx);
        if (hasScale)
          rhs_scale(c) = descale;
        const double ubc = hasWallVel ? u_bc(c) : (double)u_bc_val;
        const double orig[6] = {AE(c), AW(c), AN(c), AS(c), AT(c), AB(c)};
        double aC = (double)AC(c) * (double)descale;
        double mod[6] = {0, 0, 0, 0, 0, 0};
        double inhom = 0.0;
        for (int k = 0; k < 6; ++k) {
          const auto K = ibm.K_val(list_idx * 6 + k), M = ibm.M_val(list_idx * 6 + k);
          const auto X = ibm.X_val(list_idx * 6 + k), Nbc = ibm.Nbc_val(list_idx * 6 + k);
          const double vnb = orig[k];
          aC += vnb * (double)K;
          inhom += (double)Nbc * ubc * vnb;
          mod[k] += vnb * ((double)descale * (double)M - 1.0);
          mod[OPP[k]] += vnb * (double)X;
        }
        AC(c) = (typename MV::non_const_value_type)aC;
        AE(c) = (typename MV::non_const_value_type)(orig[0] + mod[0]);
        AW(c) = (typename MV::non_const_value_type)(orig[1] + mod[1]);
        AN(c) = (typename MV::non_const_value_type)(orig[2] + mod[2]);
        AS(c) = (typename MV::non_const_value_type)(orig[3] + mod[3]);
        AT(c) = (typename MV::non_const_value_type)(orig[4] + mod[4]);
        AB(c) = (typename MV::non_const_value_type)(orig[5] + mod[5]);
        if (hasInhom)
          a_inhom(c) += inhom;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_CUT_CELL_IBM_HPP
