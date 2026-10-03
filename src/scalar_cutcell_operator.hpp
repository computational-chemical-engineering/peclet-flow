// Cut-cell scalar operator: the stored true operator of doc/scalar_ibm_design.md §4.1 (single
// phase, WO-3) — the 7 bands, the facet overlay coefficients, the right-hand side, the lumped
// level-0 surrogate diagonal of §4.3 — and the per-facet flux evaluation the budget, the wall-flux
// getter and the facet diagnostics share. Block kernels on flow's G = 2 extended block; the Solver
// side (unit conversions, refusals, the solve, the getters) is `flow_ibm_scalars_cutcell.hpp`.
//
// The fluid row of unknown i, per unit FULL-cell volume V (§1.4, backward Euler; idt = 0 steady):
//
//   kappa_i idt (c_i - c_i^n) + sum_a sum_+- Lam w_a a (c_i - c_nb) + sum_phi cw_phi u_p,phi + B_i
//     = kappa_i S_i + sum_phi rw_phi + (Dirichlet domain faces: 2 Lam w_a a_bf g_bf)
//
// with cw = alpha G and rw = alpha G g (Dirichlet G = Lam/s, Robin G = 1/(s/Lam + 1/k)) or cw = 0
// and rw = alpha q (Neumann). The bands hold kappa idt, the face diffusion Lam w a and the
// Dirichlet-domain fold 2 Lam w a_bf; the overlay adds cw times the probe interpolation. Every row
// that is not a fluid unknown is an identity row (AC = 1, bands 0, rhs 0), so c stays 0 there.
//
// A face toward a cell that is not an unknown carries no band (the guard of ruling D-WO3-4). By the
// geometry's invariant (kappa = 0 => every aperture 0; a sealed cell has every aperture 0) it is a
// no-op on every interior face; at a non-periodic global face the ghost's unknown flag was cleared
// by the record, which is exactly the "band toward the ghost is 0" of §1.4's domain-face table.
#ifndef PECLET_FLOW_SCALAR_CUTCELL_OPERATOR_HPP
#define PECLET_FLOW_SCALAR_CUTCELL_OPERATOR_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <limits>
#include <string>
#include <vector>

#include "mac_cutcell.hpp"  // CCField, CCConst, CCExec, CCMem, C3, ccFor3
#include "peclet/core/scheme/probe_flux.hpp"
#include "policy.hpp"
#include "scalar_cutcell_geometry.hpp"

namespace peclet::flow {

/// Wall condition per body (§1.1, §8.1 `set_scalar_wall`), in the caller's PHYSICAL units; the
/// conversion to internal units is done at advance time (§1.2).
struct ScalarWallSpec {
  int type = 0;              ///< 0 neumann, 1 dirichlet, 2 robin
  double value = 0.0;        ///< neumann: q (flux into the fluid, c L/T); dirichlet/robin: g
  double coefficient = 0.0;  ///< robin: k (L/T)
};

/// The per-cut-cell-scalar state (`ScalarField::cut`). Settings are stored verbatim (physical);
/// the operator members are rebuilt on every advance or steady solve (ruling D-WO3-3).
struct ScalarCutState {
  // ---- settings (physical) ----
  ScalarWallSpec wallDefault;  ///< every body without an instance override
  std::vector<std::pair<int, ScalarWallSpec>> wallInstance;  ///< instance overrides, last wins
  double rtol = 1e-10;         ///< max-norm relative residual tolerance (§5.1)
  int maxit = 200;             ///< BiCGStab iteration cap (§5.1)
  double sourceConst = 0.0;    ///< volumetric source S (c/T), uniform part
  CCField sourceField;         ///< or per cell (physical), when allocated
  bool sourceIsField = false;  ///< set_scalar_source was given an array (sourceField)
  bool hasProfile[6] = {false, false, false, false, false, false};
  std::vector<double> profile[6];  ///< Dirichlet domain-face profile (t1 fastest)
  int profileN[6][2] = {};         ///< its tangential extents
  // ---- the operator of the last advance / steady solve ----
  CCField SAC;                            ///< level-0 surrogate diagonal (bands + lumped wall)
  Kokkos::View<double*, CCMem> cw, rw;    ///< per facet (§4.1)
  Kokkos::View<double*, CCMem> gFace[6];  ///< Dirichlet domain-face value per face cell
  bool dirFace[6] = {false, false, false, false, false, false};
  double lam = 0.0;                   ///< internal Lam = D tRef/hRef^2 of that operator
  double idt = 0.0;                   ///< 1/dt' (0 steady)
  double dt = 0.0;                    ///< dt' (internal)
  double srcFactor = 0.0;             ///< S' = S * srcFactor (tRef)
  std::vector<int> wtType;            ///< the resolved per-body wall table of that operator
  std::vector<double> wtK, wtG, wtQ;  ///< (internal k', g, q')
  long numUnknowns = 0;               ///< global fluid unknowns (the singular mean)
  bool steady = false, built = false, singular = false;
  // ---- Krylov scratch and statistics ----
  CCField kr, krh, kp, kv, kt, kz, kz2;
  CCField kb, kq;  ///< singular case: the projected rhs, and the preconditioner's rhs copy
  int iterations = 0;
  double residual = 0.0;  ///< final TRUE residual max|b - A c| / ref
  bool converged = true;
  bool warnedNoConv = false;
  double incompatibility = 0.0;  ///< steady singular case: |sum b| / sum |b| before projection
};

namespace sco {

/// Face aperture of axis `a` at the low face of cell i (`sa_a(i)`).
KOKKOS_INLINE_FUNCTION double lowAp(const CCConst& sax, const CCConst& say, const CCConst& saz,
                                    int a, long i) {
  return a == 0 ? sax(i) : (a == 1 ? say(i) : saz(i));
}

/// Zero `c` on every inner cell that is not a fluid unknown (§1.3: re-zeroed at the start of every
/// advance or steady solve, since a user set_field may have written them).
inline void zeroNonUnknown(CCField c, CCConst unk, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_zero_nonunknown", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (!(unk(i) > 0.5))
          c(i) = 0.0;
      });
}

/// The 7 bands (§4.1): AC = kappa idt + sum of the six face terms Lam w_a a (guarded toward a cell
/// that is not an unknown), off-diagonals -Lam w_a a; identity rows elsewhere.
inline void buildBands(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN, CCField AB,
                       CCField AT, CCConst kappa, CCConst unk, CCConst sax, CCConst say,
                       CCConst saz, double lam, double idt, const double w[3], C3 e, int g) {
  const double lx = lam * w[0], ly = lam * w[1], lz = lam * w[2];
  ccFor3(
      "peclet::flow::sco_bands", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5)) {
          AC(i) = 1.0;
          AW(i) = AE(i) = AS(i) = AN(i) = AB(i) = AT(i) = 0.0;
          return;
        }
        const double tw = unk(i - sx) > 0.5 ? lx * sax(i) : 0.0;
        const double te = unk(i + sx) > 0.5 ? lx * sax(i + sx) : 0.0;
        const double ts = unk(i - sy) > 0.5 ? ly * say(i) : 0.0;
        const double tn = unk(i + sy) > 0.5 ? ly * say(i + sy) : 0.0;
        const double tb = unk(i - sz) > 0.5 ? lz * saz(i) : 0.0;
        const double tt = unk(i + sz) > 0.5 ? lz * saz(i + sz) : 0.0;
        AW(i) = -tw;
        AE(i) = -te;
        AS(i) = -ts;
        AN(i) = -tn;
        AB(i) = -tb;
        AT(i) = -tt;
        AC(i) = kappa(i) * idt + ((tw + te) + (ts + tn)) + (tb + tt);
      });
}

/// Base right-hand side: b = kappa (idt c^n + S'), identity rows 0. `src` is the per-cell source
/// (physical) when `useField`, else the constant `srcConst`; both times `srcFactor` (= tRef).
inline void buildRhs(CCField b, CCConst cOld, CCConst kappa, CCConst unk, double idt,
                     double srcConst, CCConst src, bool useField, double srcFactor, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_rhs", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (!(unk(i) > 0.5)) {
          b(i) = 0.0;
          return;
        }
        const double S = (useField ? src(i) : srcConst) * srcFactor;
        b(i) = kappa(i) * (idt * cOld(i) + S);
      });
}

/// The tangential axes of face axis `a`, in x, y, z order (t1 < t2): the layout of a domain-face
/// profile (§8.1 `set_scalar_bc`), t1 fastest.
KOKKOS_INLINE_FUNCTION void faceTangents(int a, int& t1, int& t2) {
  t1 = (a == 0) ? 1 : 0;
  t2 = (a == 2) ? 1 : 2;
}

/// Dirichlet domain face (§1.4 B_i): for the inner boundary cell i of face (a, side), if it is an
/// unknown, AC += 2 Lam w_a a_bf and b += 2 Lam w_a a_bf g_bf, a_bf = the boundary face's snapped
/// aperture (the low face of the first cell, or the high face at ext - G — both in the record).
inline void dirichletFaceFold(CCField AC, CCField b, CCConst unk, CCConst saA,
                              Kokkos::View<const double*, CCMem> gv, double lamW, int a, int side,
                              C3 e, int g) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_dirichlet_face", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        if (!(unk(i) > 0.5))
          return;
        const double abf = side == 0 ? saA(i) : saA(i + sa);
        const double t = 2.0 * lamW * abf;
        AC(i) += t;
        b(i) += t * gv((long)j1 + (long)j2 * n1);
      });
  space.fence();
}

/// Sum over one Dirichlet domain face of V t (g_bf - c_i): the flux INTO the domain through it,
/// per internal time (budget `boundary_in`). Summed in a fixed (j1, j2) order on the host.
inline double dirichletFaceInflux(CCConst c, CCConst unk, CCConst saA,
                                  Kokkos::View<const double*, CCMem> gv, double lamW, double vol,
                                  int a, int side, C3 e, int g) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  Kokkos::View<double*, CCMem> q("peclet::flow::sco_face_influx", (long)n1 * n2);
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_dirichlet_face_influx", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        const long k = (long)j1 + (long)j2 * n1;
        if (!(unk(i) > 0.5)) {
          q(k) = 0.0;
          return;
        }
        const double abf = side == 0 ? saA(i) : saA(i + sa);
        q(k) = vol * (2.0 * lamW * abf) * (gv(k) - c(i));
      });
  space.fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
  double s = 0.0;
  for (long k = 0; k < (long)h.extent(0); ++k)
    s += h(k);
  return s;
}

/// The budget's residual r = b - A c in FLUX form (§9): r = b - kappa idt c - sum_faces
/// t (c_i - c_nb) with the same guarded face terms t as `buildBands`; the caller then subtracts the
/// overlay (`overlayApply(..., subtract)`) and the Dirichlet-domain folds
/// (`dirichletFaceResidual`). Equal to the band form b - A c in exact arithmetic; its round-off is
/// eps |t (c_i - c_nb)| instead of eps t |c|, and a face's two contributions are exact negatives,
/// so the identity of §1.6.1 closes to the round-off of the fluxes themselves even at large
/// dt D / h^2. Identity rows give r = 0. The solve itself never uses this form.
inline void residualFluxForm(CCField r, CCConst c, CCConst b, CCConst kappa, CCConst unk,
                             CCConst sax, CCConst say, CCConst saz, double lam, double idt,
                             const double w[3], C3 e, int g) {
  const double lx = lam * w[0], ly = lam * w[1], lz = lam * w[2];
  ccFor3(
      "peclet::flow::sco_residual_flux", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5)) {
          r(i) = 0.0;
          return;
        }
        const double ci = c(i);
        const double tw = unk(i - sx) > 0.5 ? lx * sax(i) : 0.0;
        const double te = unk(i + sx) > 0.5 ? lx * sax(i + sx) : 0.0;
        const double ts = unk(i - sy) > 0.5 ? ly * say(i) : 0.0;
        const double tn = unk(i + sy) > 0.5 ? ly * say(i + sy) : 0.0;
        const double tb = unk(i - sz) > 0.5 ? lz * saz(i) : 0.0;
        const double tt = unk(i + sz) > 0.5 ? lz * saz(i + sz) : 0.0;
        const double flux = ((tw * (ci - c(i - sx)) + te * (ci - c(i + sx))) +
                             (ts * (ci - c(i - sy)) + tn * (ci - c(i + sy)))) +
                            (tb * (ci - c(i - sz)) + tt * (ci - c(i + sz)));
        r(i) = b(i) - kappa(i) * idt * ci - flux;
      });
}

/// r(i) -= 2 Lam w_a a_bf c_i on the inner boundary cells of a Dirichlet domain face (the LHS part
/// of `dirichletFaceFold`, for the flux-form residual).
inline void dirichletFaceResidual(CCField r, CCConst c, CCConst unk, CCConst saA, double lamW,
                                  int a, int side, C3 e, int g) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_dirichlet_face_residual", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        if (!(unk(i) > 0.5))
          return;
        const double abf = side == 0 ? saA(i) : saA(i + sa);
        r(i) -= 2.0 * lamW * abf * c(i);
      });
  space.fence();
}

/// Device copy of the per-body wall table, resolved (instance override or default) and converted
/// to internal units: type, k', g, q'.
struct WallTable {
  Kokkos::View<int*, CCMem> type;
  Kokkos::View<double*, CCMem> k, gval, q;
  int nb = 0;
};

/// Per-facet overlay coefficients (§4.1): cw = alpha G, rw = alpha G g (Dirichlet/Robin) or cw = 0,
/// rw = alpha q (Neumann); both 0 on a facet whose cell is not an unknown.
inline void facetCoefficients(Kokkos::View<double*, CCMem> cw, Kokkos::View<double*, CCMem> rw,
                              const scg::ScalarFacetOverlay& fo, CCConst unk, const WallTable& wt,
                              double lam) {
  auto fcell = fo.cell;
  auto falpha = fo.alpha;
  auto fbody = fo.body;
  auto sF = fo.sF;
  auto type = wt.type;
  auto kk = wt.k;
  auto gg = wt.gval;
  auto qq = wt.q;
  const int nb = wt.nb;
  const double kInf = std::numeric_limits<double>::infinity();  // Dirichlet: G = Lam / s
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_facet_coeffs", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
      KOKKOS_LAMBDA(const long f) {
        if (!(unk(fcell(f)) > 0.5)) {
          cw(f) = 0.0;
          rw(f) = 0.0;
          return;
        }
        int bd = fbody(f);
        if (bd < 0 || bd >= nb)
          bd = 0;
        const int t = type(bd);
        const double al = falpha(f);
        if (t == 0) {
          cw(f) = 0.0;
          rw(f) = al * qq(bd);
          return;
        }
        const double G = peclet::core::scheme::wallConductance(sF(f), lam, t == 1 ? kInf : kk(bd));
        cw(f) = al * G;
        rw(f) = al * G * gg(bd);
      });
  space.fence();
}

/// y(i) += sum over the facets of cut cell i of cw * (probe interpolation of x): the overlay pass
/// of the matvec (§4.1 step 3). One thread per cut cell, facets in CSR order: deterministic, no
/// atomics.
/// `subtract` = true gives y(i) -= that sum instead (the budget's flux-form residual).
inline void overlayApply(CCField y, CCConst x, const scg::ScalarFacetOverlay& fo,
                         Kokkos::View<const double*, CCMem> cw, bool subtract = false) {
  auto cutCell = fo.cutCell;
  auto cfs = fo.cellFacetStart;
  auto offF = fo.offF;
  auto wF = fo.wF;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_overlay_apply", Kokkos::RangePolicy<CCExec>(space, 0, fo.nCut),
      KOKKOS_LAMBDA(const long c) {
        const long i = cutCell(c);
        double acc = 0.0;
        for (int f = cfs(c); f < cfs(c + 1); ++f) {
          double up = 0.0;
          for (int k = 0; k < 8; ++k)
            up += wF(f, k) * x(i + (long)offF(f, k));
          acc += cw(f) * up;
        }
        if (subtract)
          y(i) -= acc;
        else
          y(i) += acc;
      });
  space.fence();
}

/// b(i) += sum over the facets of cut cell i of rw (the wall part of the right-hand side).
inline void overlayRhs(CCField b, const scg::ScalarFacetOverlay& fo,
                       Kokkos::View<const double*, CCMem> rw) {
  auto cutCell = fo.cutCell;
  auto cfs = fo.cellFacetStart;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_overlay_rhs", Kokkos::RangePolicy<CCExec>(space, 0, fo.nCut),
      KOKKOS_LAMBDA(const long c) {
        double acc = 0.0;
        for (int f = cfs(c); f < cfs(c + 1); ++f)
          acc += rw(f);
        b(cutCell(c)) += acc;
      });
  space.fence();
}

/// The level-0 surrogate diagonal (§4.3): SAC = AC + sum of the cut cell's cw (the probe weights
/// lumped, sum w_k = 1). Off-diagonals are the bands themselves.
inline void surrogateDiagonal(CCField SAC, CCConst AC, const scg::ScalarFacetOverlay& fo,
                              Kokkos::View<const double*, CCMem> cw) {
  Kokkos::deep_copy(CCExec(), SAC, AC);
  auto cutCell = fo.cutCell;
  auto cfs = fo.cellFacetStart;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_surrogate_diag", Kokkos::RangePolicy<CCExec>(space, 0, fo.nCut),
      KOKKOS_LAMBDA(const long c) {
        double acc = 0.0;
        for (int f = cfs(c); f < cfs(c + 1); ++f)
          acc += cw(f);
        SAC(cutCell(c)) += acc;
      });
  space.fence();
}

/// Per facet: the probe value u_p of `x` and the wall flux INTO the fluid per unit cell volume per
/// internal time, rw - cw u_p (§1.4: the row carries + cw u_p - rw = alpha Q, the flux out).
inline void facetFlux(Kokkos::View<double*, CCMem> up, Kokkos::View<double*, CCMem> qin, CCConst x,
                      const scg::ScalarFacetOverlay& fo, Kokkos::View<const double*, CCMem> cw,
                      Kokkos::View<const double*, CCMem> rw) {
  auto fcell = fo.cell;
  auto offF = fo.offF;
  auto wF = fo.wF;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_facet_flux", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
      KOKKOS_LAMBDA(const long f) {
        const long i = fcell(f);
        double u = 0.0;
        for (int k = 0; k < 8; ++k)
          u += wF(f, k) * x(i + (long)offF(f, k));
        up(f) = u;
        qin(f) = rw(f) - cw(f) * u;
      });
  space.fence();
}

// ---- inner-cell reductions (this rank; the Solver adds the MPI_Allreduce) ----------------------

/// sum over inner cells of a(i) b(i).
inline double dotLocal(CCConst a, CCConst b, C3 e, int g) {
  double s = 0.0;
  ccReduce3(
      "peclet::flow::sco_dot", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        acc += a(i) * b(i);
      },
      s);
  return s;
}

/// (sum a b, sum c c) over inner cells in one pass.
inline void dot2Local(CCConst a, CCConst b, CCConst c, C3 e, int g, double& ab, double& cc) {
  CCExec space;
  double s1 = 0.0, s2 = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_dot2", MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        p1 += a(i) * b(i);
        p2 += c(i) * c(i);
      },
      s1, s2);
  ab = s1;
  cc = s2;
}

/// max over inner cells of |a|.
inline double maxabsLocal(CCConst a, C3 e, int g) {
  double m = 0.0;
  ccReduce3(
      "peclet::flow::sco_maxabs", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double v = Kokkos::fabs(a(i));
        if (v > acc)
          acc = v;
      },
      Kokkos::Max<double>(m));
  return m;
}

/// (sum a, sum |a|) over the inner cells that are unknowns.
inline void sumUnknownLocal(CCConst a, CCConst unk, C3 e, int g, double& sum, double& sumAbs) {
  CCExec space;
  double s1 = 0.0, s2 = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_sum_unknown",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (unk(i) > 0.5) {
          p1 += a(i);
          p2 += Kokkos::fabs(a(i));
        }
      },
      s1, s2);
  sum = s1;
  sumAbs = s2;
}

/// a(i) += d on the inner cells that are unknowns.
inline void addOnUnknown(CCField a, CCConst unk, double d, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_add_unknown", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (unk(i) > 0.5)
          a(i) += d;
      });
}

/// (sum kappa c, sum kappa) over the unknowns (times V by the caller).
inline void kappaMomentsLocal(CCConst c, CCConst kappa, CCConst unk, C3 e, int g, double& kc,
                              double& k) {
  CCExec space;
  double s1 = 0.0, s2 = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_kappa_moments",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (unk(i) > 0.5) {
          p1 += kappa(i) * c(i);
          p2 += kappa(i);
        }
      },
      s1, s2);
  kc = s1;
  k = s2;
}

/// sum over unknowns of kappa S' (the source part of the right-hand side, per unit V).
inline double sourceSumLocal(CCConst kappa, CCConst unk, double srcConst, CCConst src,
                             bool useField, double srcFactor, C3 e, int g) {
  double s = 0.0;
  ccReduce3(
      "peclet::flow::sco_source_sum", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (unk(i) > 0.5)
          acc += kappa(i) * ((useField ? src(i) : srcConst) * srcFactor);
      },
      s);
  return s;
}

/// max over facets of cw (0 when there is no Dirichlet/Robin facet: the steady singular test).
inline double maxFacetLocal(Kokkos::View<const double*, CCMem> cw) {
  double m = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_max_cw", Kokkos::RangePolicy<CCExec>(0, (long)cw.extent(0)),
      KOKKOS_LAMBDA(const long f, double& acc) {
        if (cw(f) > acc)
          acc = cw(f);
      },
      Kokkos::Max<double>(m));
  return m;
}

}  // namespace sco
}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_CUTCELL_OPERATOR_HPP
