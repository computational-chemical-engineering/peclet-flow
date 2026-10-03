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
#include <memory>
#include <string>
#include <vector>

#include "mac_cutcell.hpp"  // CCField, CCConst, CCExec, CCMem, C3, ccFor3
#include "peclet/core/scheme/probe_flux.hpp"
#include "policy.hpp"
#include "scalar_cutcell_geometry.hpp"
#include "staggered_advection.hpp"  // sadv::tvd / sou / fou_flux, the legacy reconstructions

namespace peclet::flow {

class ScalarMG;  // scalar_mg.hpp (WO-4): the preconditioner, one per cut-cell scalar

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
  /// Mean-gradient (closure) mode (§1.5, WO-6): the field holds theta, c = G.x + theta, in the
  /// frame moving with the fluid's mean velocity. G physical (c/L); (0, 0, 0) is off.
  double meanGrad[3] = {0.0, 0.0, 0.0};
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
  bool meanGradOn = false;            ///< that operator carried the mean-gradient terms (§1.5)
  double gPhys[3] = {0.0, 0.0, 0.0};  ///< its G (physical, c/L)
  double gInt[3] = {0.0, 0.0, 0.0};   ///< its G' = G hRef (c per internal length)
  double ubar[3] = {0.0, 0.0, 0.0};   ///< its moving-frame velocity U'bar = sum U_i / sum kappa_i V
  // ---- Krylov scratch and statistics ----
  CCField kr, krh, kp, kv, kt, kz, kz2;
  CCField kb, kq;  ///< singular case: the projected rhs, and the preconditioner's rhs copy
  int iterations = 0;
  double residual = 0.0;  ///< final TRUE residual max|b - A c| / ref
  bool converged = true;
  bool warnedNoConv = false;
  double incompatibility = 0.0;  ///< steady singular case: |sum b| / sum |b| before projection
  // ---- ScalarMG (§5.2): level table per geometry version / block, coefficients per build ----
  std::shared_ptr<ScalarMG> mg;
  long mgVersion = -1;  ///< the geometry version its level table was built for
  std::size_t mgN = 0;  ///< the block size it was built for
  int mgLevels = 1;     ///< levels the last solve's V-cycle used (census `mg_levels`)
  // ---- advection (WO-5, §6): the face flux of the last advance and its classification ----
  CCField
      phi[3];  ///< F/V through the LOW a-face of cell i (internal 1/T), guarded (sco::faceFluxes)
  CCField Phi[3];   ///< explicit face fluxes phi c*(c^n), 0 on implicit faces (§6.2)
  CCField small;    ///< 1 on a small cell (§6.3), inner cells + ghost layer 1
  CCField outflow;  ///< lumped implicit outflow per cell (inside SAC; ScalarMG coarse mass, §5.2)
  /// A2 (§6.7, D-WO5-3): the implicit outflow the open-face rows add to AC, summed over the cell's
  /// open faces and clamped at 0 (implicit backflow), 0 elsewhere — the advective path's coarse
  /// mass. Written by steady advecting solves only (`outflow` stays the transient path's input).
  CCField omegaOpen;
  double maxCellPeclet = 0.0;  ///< census `max_cell_peclet` (A2): max |phi_a| / (Lam' w_a)
  bool openFace[6] = {false, false, false, false, false, false};  ///< open face rows built (WO-5b)
  bool openInflow[6] = {false, false, false, false, false, false};  ///< ... and it is an inflow
  bool advecting = false;     ///< some face carried flux: else every advection kernel was skipped
  long numSmall = 0;          ///< census `num_small_cells`
  long numImplicitFaces = 0;  ///< census `num_implicit_faces` (faces carrying flux, implicit)
  long numFluxFaces = 0;      ///< faces carrying flux (the implicit fraction's denominator)
  long numGuardedFaces = 0;   ///< faces whose projection flux the guard zeroed (a no-op reading)
  double bulkCourant = 0.0;   ///< census `bulk_courant`: C_bulk of §6.3 (0 steady)
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

// ---- advection and small cells (WO-5, §6) -------------------------------------------------------
//
// phi_a(i) is the volume flux through the LOW a-face of cell i per unit FULL-cell volume per
// internal time, positive along +a: open * vel with the projection's own openness and face velocity
// (Solver::scalarFaceFlux, §6.1). The velocity is an INDEX velocity (cells per tRef), so the A_a/V
// = 1/h'_a of F/V is already inside it (doc/anisotropic_metric.md §2). The flux out of cell i
// through its high a-face is phi_a(i + s_a), through its low a-face -phi_a(i).

/// phi = open * vel on every face whose two cells are inside the block (0 on the block's lowest
/// layer), and 0 where either cell is not a fluid unknown: the advective analogue of the band guard
/// of ruling D-WO3-4 (a non-unknown row is an identity row and cannot take its half of the flux; at
/// a non-periodic global face the cleared ghost flag zeroes the wall/slip face, where u.n = 0).
inline void faceFluxes(CCField phi, CCConst vel, CCConst open, CCConst unk, int a, C3 e) {
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const long s = st[a];
  ccFor3(
      "peclet::flow::sco_face_flux", C3{0, 0, 0}, e, KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const int xa = a == 0 ? x : (a == 1 ? y : z);
        if (xa < 1 || !(unk(i) > 0.5) || !(unk(i - s) > 0.5)) {
          phi(i) = 0.0;
          return;
        }
        phi(i) = open(i) * vel(i);
      });
}

/// (faces carrying flux, faces whose nonzero open * vel the guard zeroed) over the low faces of the
/// inner cells (each interior face exactly once over all ranks; a non-periodic high domain face is
/// not counted).
inline void faceFluxCountsLocal(CCConst phi, CCConst vel, CCConst open, int a, C3 e, int g,
                                long& nFlux, long& nGuarded) {
  (void)a;
  CCExec space;
  long n1 = 0, n2 = 0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_flux_counts",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, long& p1, long& p2) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (phi(i) != 0.0)
          ++p1;
        else if (open(i) * vel(i) != 0.0)
          ++p2;
      },
      n1, n2);
  nFlux = n1;
  nGuarded = n2;
}

/// max |phi| over the low faces of the inner cells (0: no advection this step).
inline double maxAbsFluxLocal(CCConst phx, CCConst phy, CCConst phz, C3 e, int g) {
  double m = 0.0;
  ccReduce3(
      "peclet::flow::sco_max_flux", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double v = Kokkos::fmax(Kokkos::fabs(phx(i)),
                                      Kokkos::fmax(Kokkos::fabs(phy(i)), Kokkos::fabs(phz(i))));
        if (v > acc)
          acc = v;
      },
      Kokkos::Max<double>(m));
  return m;
}

/// A2 census `max_cell_peclet` on this rank: the max over the interior flux faces (the low faces
/// of the inner cells whose two cells are fluid unknowns) and the axes of |phi_a| / (lam w_a), the
/// aperture-weighted face Peclet number |u_a| a h_a / D (lam = 0 with flux: +inf).
inline double maxCellPecletLocal(CCConst phx, CCConst phy, CCConst phz, CCConst unk, double lam,
                                 const double w[3], C3 e, int g) {
  const double dx = lam * w[0], dy = lam * w[1], dz = lam * w[2];
  double m = 0.0;
  ccReduce3(
      "peclet::flow::sco_max_cell_peclet", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const double P[3] = {unk(i - sx) > 0.5 ? Kokkos::fabs(phx(i)) / dx : 0.0,
                             unk(i - sy) > 0.5 ? Kokkos::fabs(phy(i)) / dy : 0.0,
                             unk(i - sz) > 0.5 ? Kokkos::fabs(phz(i)) / dz : 0.0};
        for (int k = 0; k < 3; ++k)
          if (P[k] > acc)  // (0 / 0 = NaN when lam = 0 and no flux: skipped)
            acc = P[k];
      },
      Kokkos::Max<double>(m));
  return m > 0.0 ? m : 0.0;
}

/// omega = max(omega, 0) on the inner cells (A2: implicit backflow never enters the coarse mass).
inline void clampNonNegative(CCField omega, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_clamp_omega", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (omega(i) < 0.0)
          omega(i) = 0.0;
      });
}

/// Out_i = sum over the six faces of max(F_out, 0), per unit V (§6.3), in a fixed order.
KOKKOS_INLINE_FUNCTION double cellOutflow(const CCConst& phx, const CCConst& phy,
                                          const CCConst& phz, long i, long sy, long sz) {
  return ((Kokkos::fmax(phx(i + 1), 0.0) + Kokkos::fmax(-phx(i), 0.0)) +
          (Kokkos::fmax(phy(i + sy), 0.0) + Kokkos::fmax(-phy(i), 0.0))) +
         (Kokkos::fmax(phz(i + sz), 0.0) + Kokkos::fmax(-phz(i), 0.0));
}

/// C_bulk of §6.3 on this rank: max over the inner FULL cells (an unknown with kappa = 1 and all
/// six snapped apertures 1) of dt Out_i.
inline double bulkCourantLocal(CCConst phx, CCConst phy, CCConst phz, CCConst kappa, CCConst unk,
                               CCConst sax, CCConst say, CCConst saz, double dt, C3 e, int g) {
  double m = 0.0;
  ccReduce3(
      "peclet::flow::sco_bulk_courant", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5) || !(kappa(i) >= 1.0) || !(sax(i) >= 1.0) || !(sax(i + 1) >= 1.0) ||
            !(say(i) >= 1.0) || !(say(i + sy) >= 1.0) || !(saz(i) >= 1.0) || !(saz(i + sz) >= 1.0))
          return;
        const double v = dt * cellOutflow(phx, phy, phz, i, sy, sz);
        if (v > acc)
          acc = v;
      },
      Kokkos::Max<double>(m));
  return m;
}

/// The small-cell flags of §6.3 over the inner cells and ghost layer 1 (from the face fluxes and
/// kappa, which carry ghosts: no exchange, deterministic): small(i) = 1 iff i is an unknown,
/// kappa_i < 1 and dt Out_i > thr kappa_i, thr = max(1/2, C_bulk). 0 elsewhere on the block.
inline void smallCells(CCField small, CCConst phx, CCConst phy, CCConst phz, CCConst kappa,
                       CCConst unk, double dt, double thr, C3 e, int g) {
  Kokkos::deep_copy(CCExec(), small, 0.0);
  ccFor3(
      "peclet::flow::sco_small_cells", C3{g - 1, g - 1, g - 1},
      C3{e.x - g + 1, e.y - g + 1, e.z - g + 1}, KOKKOS_LAMBDA(int x, int y, int z) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        const double k = kappa(i);
        if (!(unk(i) > 0.5) || !(k < 1.0))
          return;
        small(i) = dt * cellOutflow(phx, phy, phz, i, sy, sz) > thr * k ? 1.0 : 0.0;
      });
}

/// Count of small inner cells (census `num_small_cells`).
inline long countSmallLocal(CCConst small, C3 e, int g) {
  long n = 0;
  ccReduce3(
      "peclet::flow::sco_count_small", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (small(i) > 0.5)
          ++acc;
      },
      n);
  return n;
}

/// Faces carrying flux that are implicit, over the low faces of the inner cells (census
/// `num_implicit_faces`): steady, or either adjacent cell small. A face toward a non-unknown (an
/// open domain face, WO-5b) is counted by openFaceCountsLocal instead.
inline long countImplicitLocal(CCConst phi, CCConst small, CCConst unk, int a, bool steady, C3 e,
                               int g) {
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const long s = st[a];
  long n = 0;
  ccReduce3(
      "peclet::flow::sco_count_implicit", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        if (phi(i) != 0.0 && unk(i) > 0.5 && unk(i - s) > 0.5 &&
            (steady || small(i) > 0.5 || small(i - s) > 0.5))
          ++acc;
      },
      n);
  return n;
}

/// The implicit FOU part of the operator (§1.4, §6.3; all faces when steady, §6.7): on every
/// unknown inner cell, for each face that is implicit (steady, or either cell small), AC +=
/// max(F_out, 0) and the band toward the neighbour += min(F_out, 0); `outflow` = the sum of the
/// max(F_out, 0) added (the surrogate's lumped outflow, §4.3), 0 on every other inner cell. A face
/// flux is one number read by both cells, so the face's two contributions are exact negatives
/// (conservation). A face toward a non-unknown is skipped: its flux is 0 by the guard of
/// faceFluxes except on an open domain face, whose row is openFaceAdvection's (WO-5b).
inline void advectionBands(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN, CCField AB,
                           CCField AT, CCField outflow, CCConst phx, CCConst phy, CCConst phz,
                           CCConst small, CCConst unk, bool steady, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_adv_bands", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5)) {
          outflow(i) = 0.0;
          return;
        }
        const bool si = small(i) > 0.5;
        // F_out per face: W, E, S, N, B, T; implicit flag per face
        const double F[6] = {-phx(i), phx(i + sx), -phy(i), phy(i + sy), -phz(i), phz(i + sz)};
        const long nb[6] = {i - sx, i + sx, i - sy, i + sy, i - sz, i + sz};
        double d[6], o[6];
        for (int f = 0; f < 6; ++f) {
          const bool imp = (steady || si || small(nb[f]) > 0.5) && unk(nb[f]) > 0.5;
          d[f] = imp ? Kokkos::fmax(F[f], 0.0) : 0.0;
          o[f] = imp ? Kokkos::fmin(F[f], 0.0) : 0.0;
        }
        const double om = ((d[0] + d[1]) + (d[2] + d[3])) + (d[4] + d[5]);
        outflow(i) = om;
        AC(i) += om;
        AW(i) += o[0];
        AE(i) += o[1];
        AS(i) += o[2];
        AN(i) += o[3];
        AB(i) += o[4];
        AT(i) += o[5];
      });
}

/// The explicit face fluxes of §6.2 on the low a-faces of the cells g .. ext_a - g (the high face
/// of the last inner cell included), transverse inner: 0 on an implicit face (steady, or either
/// cell small); else Phi = phi c*(c^n) with the legacy reconstruction (`sadv`: 0 fou, 1 koren, 2
/// sou), first-order upwind when any cell of the stencil (two upwind, one downwind) is not a fluid
/// unknown — the ghosts beyond a non-periodic global face included (their flags are cleared). A
/// face toward a non-unknown is skipped (0 by the guard of faceFluxes, or an open domain face,
/// whose flux openFaceAdvection carries; WO-5b).
inline void explicitFaceFluxes(CCField Phi, CCConst phi, CCConst c, CCConst unk, CCConst small,
                               int a, int scheme, bool steady, C3 e, int g) {
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const long s = st[a];
  const C3 hi{a == 0 ? e.x - g + 1 : e.x - g, a == 1 ? e.y - g + 1 : e.y - g,
              a == 2 ? e.z - g + 1 : e.z - g};
  Kokkos::deep_copy(CCExec(), Phi, 0.0);
  ccFor3(
      "peclet::flow::sco_explicit_flux", C3{g, g, g}, hi, KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double v = phi(i);
        if (v == 0.0 || steady || small(i) > 0.5 || small(i - s) > 0.5 || !(unk(i) > 0.5) ||
            !(unk(i - s) > 0.5))
          return;
        const double cLL = c(i - 2 * s), cL = c(i - s), cR = c(i), cRR = c(i + s);
        bool fou = scheme == 0;
        if (!fou)
          fou = v > 0.0 ? !(unk(i - 2 * s) > 0.5 && unk(i - s) > 0.5 && unk(i) > 0.5)
                        : !(unk(i + s) > 0.5 && unk(i) > 0.5 && unk(i - s) > 0.5);
        Phi(i) =
            fou ? sadv::fou_flux(cL, cR, v)
                : (scheme == 2 ? sadv::sou(cLL, cL, cR, cRR, v) : sadv::tvd(cLL, cL, cR, cRR, v));
      });
}

/// b(i) -= the net explicit outflow sum_a (Phi_a(i + s_a) - Phi_a(i)) on the unknown inner cells.
inline void explicitAdvectionRhs(CCField b, CCConst Px, CCConst Py, CCConst Pz, CCConst unk, C3 e,
                                 int g) {
  ccFor3(
      "peclet::flow::sco_explicit_rhs", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        b(i) -= ((Px(i + sx) - Px(i)) + (Py(i + sy) - Py(i))) + (Pz(i + sz) - Pz(i));
      });
}

/// r(i) -= the implicit faces' net outflow sum F_out c_up at c (the budget's flux-form residual,
/// §9): per face exactly the `advectionBands` terms, so a face's two contributions are exact
/// negatives (an open domain face's: openFaceResidual).
inline void advectionResidual(CCField r, CCConst c, CCConst phx, CCConst phy, CCConst phz,
                              CCConst small, CCConst unk, bool steady, C3 e, int g) {
  ccFor3(
      "peclet::flow::sco_adv_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const bool si = small(i) > 0.5;
        const double F[6] = {-phx(i), phx(i + sx), -phy(i), phy(i + sy), -phz(i), phz(i + sz)};
        const long nb[6] = {i - sx, i + sx, i - sy, i + sy, i - sz, i + sz};
        const double ci = c(i);
        double q[6];
        for (int f = 0; f < 6; ++f) {
          const bool imp = (steady || si || small(nb[f]) > 0.5) && unk(nb[f]) > 0.5;
          q[f] = imp ? (F[f] > 0.0 ? F[f] * ci : F[f] * c(nb[f])) : 0.0;
        }
        r(i) -= ((q[0] + q[1]) + (q[2] + q[3])) + (q[4] + q[5]);
      });
}

// ---- open domain faces (WO-5b; §1.4 B_i, §6.5, ruling D-WO5-3) ---------------------------------
//
// A flow inflow / outflow global face carries the boundary-face flux the projection constrained,
// open * vel at the boundary face (the low face of the first inner cell, or the first GHOST index
// on the high side), captured right after the projection (Solver::scalarCaptureOpenFaceFlux) as a
// plane over the block's FULL transverse extent, index j1 + j2 * ext[t1] (faceTangents order).
// openFaceFluxes writes it into phi_a at the boundary face, so Out_i, C_bulk and the small flags
// of §6.3 see it; the interior kernels above skip every face toward a non-unknown, so its row is
// carried here alone, per §1.4's domain-face table:
//   inflow   b_i += F_in g_bf (g_bf the scalar's Dirichlet value; always on the rhs);
//   outflow  F_out c_i (upwind on the inner cell: the zero-gradient exit), explicit at c^n on the
//            rhs, or implicit (AC += F_out) when the inner cell is small, and always when steady.
// F_out is the volume flux per unit V out of the inner cell through that face: -phi on the low
// side, +phi on the high side.

/// phi(bf) = plane value on the boundary face of (a, side), over the full transverse extent, where
/// the inner-side cell is an unknown; 0 elsewhere on that plane.
inline void openFaceFluxes(CCField phi, Kokkos::View<const double*, CCMem> plane, CCConst unk,
                           int a, int side, C3 e, int g) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int m1 = ext[t1], m2 = ext[t2];
  const long bfa = side == 0 ? g : ext[a] - g;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_open_face_flux", MDRange2<CCExec>(space, {0, 0}, {m1, m2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long bf = bfa * sa + (long)j1 * s1 + (long)j2 * s2;
        const long in = side == 0 ? bf : bf - sa;
        phi(bf) = unk(in) > 0.5 ? plane((long)j1 + (long)j2 * m1) : 0.0;
      });
  space.fence();
}

/// The open face's row terms on the unknown inner boundary cells of (a, side) (table above):
/// inflow b += F_in g (gv over the inner tangential range, j1 + j2 n1, as dirichletFaceFold);
/// outflow implicit (steady or small) AC += F_out and outflow += F_out (the surrogate's lumped
/// outflow, as advectionBands), explicit b -= F_out c^n. Runs after advectionBands.
/// A2: `omega` (when `wOmega`) += F_out on the implicit outflow rows as well (omega_open).
inline void openFaceAdvection(CCField AC, CCField b, CCField outflow, CCConst cOld, CCConst phi,
                              CCConst small, CCConst unk, Kokkos::View<const double*, CCMem> gv,
                              bool inflow, int a, int side, bool steady, C3 e, int g,
                              CCField omega = CCField(), bool wOmega = false) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_open_face_adv", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        if (!(unk(i) > 0.5))
          return;
        const double Fout = side == 0 ? -phi(i) : phi(i + sa);
        if (inflow) {
          b(i) -= Fout * gv((long)j1 + (long)j2 * n1);
        } else if (steady || small(i) > 0.5) {
          AC(i) += Fout;
          outflow(i) += Fout;
          if (wOmega)
            omega(i) += Fout;
        } else {
          b(i) -= Fout * cOld(i);
        }
      });
  space.fence();
}

/// r(i) -= F_out c_i on the implicit outflow faces of (a, side) (the budget's flux-form residual;
/// the inflow and explicit terms sit in b).
inline void openFaceResidual(CCField r, CCConst c, CCConst phi, CCConst small, CCConst unk,
                             bool inflow, int a, int side, bool steady, C3 e, int g) {
  if (inflow)
    return;
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_open_face_residual", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        if (!(unk(i) > 0.5) || !(steady || small(i) > 0.5))
          return;
        const double Fout = side == 0 ? -phi(i) : phi(i + sa);
        r(i) -= Fout * c(i);
      });
  space.fence();
}

/// Sum over one open face of V times the advective flux INTO the domain through it, per internal
/// time (budget `boundary_in`): inflow F_in g; outflow -F_out c_i at c (implicit) or c^n
/// (explicit). Summed in a fixed (j1, j2) order on the host.
inline double openFaceInflux(CCConst c, CCConst cOld, CCConst phi, CCConst small, CCConst unk,
                             Kokkos::View<const double*, CCMem> gv, bool inflow, double vol, int a,
                             int side, bool steady, C3 e, int g) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  Kokkos::View<double*, CCMem> q("peclet::flow::sco_open_face_influx", (long)n1 * n2);
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_open_face_influx", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        const long k = (long)j1 + (long)j2 * n1;
        if (!(unk(i) > 0.5)) {
          q(k) = 0.0;
          return;
        }
        const double Fout = side == 0 ? -phi(i) : phi(i + sa);
        const double cv = inflow ? gv(k) : ((steady || small(i) > 0.5) ? c(i) : cOld(i));
        q(k) = -vol * (Fout * cv);
      });
  space.fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
  double s = 0.0;
  for (long k = 0; k < (long)h.extent(0); ++k)
    s += h(k);
  return s;
}

/// The census counts of one open face over this rank's inner boundary cells: (faces carrying flux
/// on the HIGH side — faceFluxCountsLocal already counts the low side's —, the implicit ones of
/// either side: outflow with steady or a small inner cell).
inline void openFaceCountsLocal(CCConst phi, CCConst small, CCConst unk, bool inflow, int a,
                                int side, bool steady, C3 e, int g, long& nFlux, long& nImpl) {
  int t1, t2;
  faceTangents(a, t1, t2);
  const int ext[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int n1 = ext[t1] - 2 * g, n2 = ext[t2] - 2 * g;
  const long sa = st[a], s1 = st[t1], s2 = st[t2];
  const int aInner = side == 0 ? g : ext[a] - g - 1;
  CCExec space;
  long p = 0, q = 0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_open_face_counts", MDRange2<CCExec>(space, {0, 0}, {n1, n2}),
      KOKKOS_LAMBDA(int j1, int j2, long& c1, long& c2) {
        const long i = (long)aInner * sa + (long)(j1 + g) * s1 + (long)(j2 + g) * s2;
        const double v = side == 0 ? phi(i) : phi(i + sa);
        if (!(unk(i) > 0.5) || v == 0.0)
          return;
        if (side == 1)
          ++c1;
        if (!inflow && (steady || small(i) > 0.5))
          ++c2;
      },
      p, q);
  nFlux = p;
  nImpl = q;
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

// ---- mean-gradient (closure) mode (WO-6, §1.5) --------------------------------------------------
//
// The unknown is theta, c = G.x + theta. Wherever the discretization evaluates c at a point x it
// uses theta + G.x and moves the G.x part to the right-hand side. Face differences use the TRUE
// displacement +-h'_a e_a (never a wrapped index difference); a probe stencil cell or a domain-face
// centre uses its global position, unwrapped relative to the cut cell (an extended-block index
// past a periodic face is simply its unwrapped position).

/// G.x at the centre of extended-block cell j (physical; `cellCentres`' formula, unwrapped).
struct MeanGradPosition {
  double G[3] = {0.0, 0.0, 0.0};    ///< the physical mean gradient (c/L)
  double org[3] = {0.0, 0.0, 0.0};  ///< the physical lower corner of the global grid
  double h[3] = {1.0, 1.0, 1.0};    ///< the physical cell size
  long base[3] = {0, 0, 0};         ///< og_a - g: global index of extended-block index 0
  long sy = 1, sz = 1;              ///< extended-block strides
  KOKKOS_INLINE_FUNCTION double operator()(long j) const {
    const long z = j / sz, y = (j - z * sz) / sy, x = j - z * sz - y * sy;
    const double px = org[0] + ((double)(base[0] + x) + 0.5) * h[0];
    const double py = org[1] + ((double)(base[1] + y) + 0.5) * h[1];
    const double pz = org[2] + ((double)(base[2] + z) + 0.5) * h[2];
    return (G[0] * px + G[1] * py) + G[2] * pz;
  }
};

/// Faces (§1.5): b(i) += sum_a gf_a (a_a^+ - a_a^-) over the coupled faces (the bands' guard),
/// gf_a = Lam w_a h'_a G'_a — the G.x part of the face terms Lam w_a a (c_i - c_nb) moved to the
/// right-hand side. Unknown inner cells only.
inline void meanGradientFaceRhs(CCField b, CCConst unk, CCConst sax, CCConst say, CCConst saz,
                                const double gf[3], C3 e, int g) {
  const double gx = gf[0], gy = gf[1], gz = gf[2];
  ccFor3(
      "peclet::flow::sco_mg_face_rhs", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const double aw = unk(i - sx) > 0.5 ? sax(i) : 0.0;
        const double ae = unk(i + sx) > 0.5 ? sax(i + sx) : 0.0;
        const double as = unk(i - sy) > 0.5 ? say(i) : 0.0;
        const double an = unk(i + sy) > 0.5 ? say(i + sy) : 0.0;
        const double ab = unk(i - sz) > 0.5 ? saz(i) : 0.0;
        const double at = unk(i + sz) > 0.5 ? saz(i + sz) : 0.0;
        b(i) += (gx * (ae - aw) + gy * (an - as)) + gz * (at - ab);
      });
}

/// Dirichlet/Robin probes (§1.5): rw(f) -= cw(f) sum_k w_k G.x_k, so that rw = alpha G_phi (g -
/// sum_k w_k G.x_k); x_k the global centres of the stencil cells. Neumann facets (cw = 0) and
/// facets of non-unknown cells (cw = rw = 0) are unchanged.
inline void meanGradientProbeShift(Kokkos::View<double*, CCMem> rw, const scg::ScalarFacetOverlay& fo,
                                   Kokkos::View<const double*, CCMem> cw,
                                   const MeanGradPosition& pos) {
  auto fcell = fo.cell;
  auto offF = fo.offF;
  auto wF = fo.wF;
  CCExec space;
  Kokkos::parallel_for(
      "peclet::flow::sco_mg_probe_shift", Kokkos::RangePolicy<CCExec>(space, 0, fo.n),
      KOKKOS_LAMBDA(const long f) {
        if (cw(f) == 0.0)
          return;
        const long i = fcell(f);
        double gxp = 0.0;
        for (int k = 0; k < 8; ++k)
          gxp += wF(f, k) * pos(i + (long)offF(f, k));
        rw(f) -= cw(f) * gxp;
      });
  space.fence();
}

/// U'_a/V of cell i (§1.5): the volume-integrated cell velocity sum_f F_out (x_f - x_i) per unit
/// full-cell volume, = (h'_a/2) (phi_a(i + s_a) + phi_a(i)) (internal length per internal time).
KOKKOS_INLINE_FUNCTION void cellVelocity(const CCConst& phx, const CCConst& phy, const CCConst& phz,
                                         long i, long sy, long sz, const double hh[3],
                                         double u[3]) {
  u[0] = hh[0] * (phx(i + 1) + phx(i));
  u[1] = hh[1] * (phy(i + sy) + phy(i));
  u[2] = hh[2] * (phz(i + sz) + phz(i));
}

/// (sum U'_x/V, sum U'_y/V, sum U'_z/V) over the unknown inner cells (the moving frame's
/// numerator; its denominator is sum kappa, `kappaMomentsLocal`). hh = h'/2.
inline void cellVelocitySumLocal(CCConst phx, CCConst phy, CCConst phz, CCConst unk,
                                 const double hh[3], C3 e, int g, double s[3]) {
  CCExec space;
  const double h0 = hh[0], h1 = hh[1], h2 = hh[2];
  double a = 0.0, b = 0.0, c = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_mg_cell_velocity_sum",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2, double& p3) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const double hq[3] = {h0, h1, h2};
        double u[3];
        cellVelocity(phx, phy, phz, i, sy, sz, hq, u);
        p1 += u[0];
        p2 += u[1];
        p3 += u[2];
      },
      a, b, c);
  s[0] = a;
  s[1] = b;
  s[2] = c;
}

/// Advection (§1.5): b(i) -= G'.(U'_i/V - kappa_i U'bar), the exact linear part of the advective
/// term in the frame moving with U'bar. Sums to zero over the unknowns (exactly, in exact
/// arithmetic), which keeps steady problems compatible.
inline void meanGradientAdvectionRhs(CCField b, CCConst phx, CCConst phy, CCConst phz,
                                     CCConst kappa, CCConst unk, const double gi[3],
                                     const double hh[3], const double ub[3], C3 e, int g) {
  const double g0 = gi[0], g1 = gi[1], g2 = gi[2], h0 = hh[0], h1 = hh[1], h2 = hh[2], u0 = ub[0],
               u1 = ub[1], u2 = ub[2];
  ccFor3(
      "peclet::flow::sco_mg_adv_rhs", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const double hq[3] = {h0, h1, h2};
        double u[3];
        cellVelocity(phx, phy, phz, i, sy, sz, hq, u);
        const double k = kappa(i);
        b(i) -= (g0 * (u[0] - k * u0) + g1 * (u[1] - k * u1)) + g2 * (u[2] - k * u2);
      });
}

/// `scalar_mean_flux` (§8.1), this rank's sums (internal; the caller divides by the global cell
/// count): diffusive, over the low faces of the inner cells whose two cells are unknowns,
/// sum -Lam a (theta_i - theta_{i - s_a} + G'_a h'_a) / h'_a; advective, over the unknown inner
/// cells, sum (U'_i/V - kappa_i U'bar) theta_i (zero when the operator carried no advection).
inline void meanFluxLocal(CCConst th, CCConst kappa, CCConst unk, CCConst sax, CCConst say,
                          CCConst saz, CCConst phx, CCConst phy, CCConst phz, bool advecting,
                          double lam, const double gi[3], const double hp[3], const double ub[3],
                          C3 e, int g, double diff[3], double adv[3]) {
  CCExec space;
  const double g0 = gi[0], g1 = gi[1], g2 = gi[2], hp0 = hp[0], hp1 = hp[1], hp2 = hp[2];
  const double h0 = 0.5 * hp[0], h1 = 0.5 * hp[1], h2 = 0.5 * hp[2], u0 = ub[0], u1 = ub[1],
               u2 = ub[2];
  double d0 = 0.0, d1 = 0.0, d2 = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_mean_flux_diff",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2, double& p3) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        if (unk(i - 1) > 0.5)
          p1 += -lam * sax(i) * ((th(i) - th(i - 1)) + g0 * hp0) / hp0;
        if (unk(i - sy) > 0.5)
          p2 += -lam * say(i) * ((th(i) - th(i - sy)) + g1 * hp1) / hp1;
        if (unk(i - sz) > 0.5)
          p3 += -lam * saz(i) * ((th(i) - th(i - sz)) + g2 * hp2) / hp2;
      },
      d0, d1, d2);
  diff[0] = d0;
  diff[1] = d1;
  diff[2] = d2;
  adv[0] = adv[1] = adv[2] = 0.0;
  if (!advecting)
    return;
  double a0 = 0.0, a1 = 0.0, a2 = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::sco_mean_flux_adv",
      MDRange3<CCExec>(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z, double& p1, double& p2, double& p3) {
        const long sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)x + (long)y * sy + (long)z * sz;
        if (!(unk(i) > 0.5))
          return;
        const double hq[3] = {h0, h1, h2};
        double u[3];
        cellVelocity(phx, phy, phz, i, sy, sz, hq, u);
        const double k = kappa(i), t = th(i);
        p1 += (u[0] - k * u0) * t;
        p2 += (u[1] - k * u1) * t;
        p3 += (u[2] - k * u2) * t;
      },
      a0, a1, a2);
  adv[0] = a0;
  adv[1] = a1;
  adv[2] = a2;
}

}  // namespace sco
}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_CUTCELL_OPERATOR_HPP
