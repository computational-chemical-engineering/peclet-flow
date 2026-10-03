// BiCGStab for the cut-cell scalar operator (doc/scalar_ibm_design.md §5.1, D9/D11).
//
// Modelled statement for statement on `CutcellMG::solveBiCGStab` (mac_cutcell_mg.hpp): the same
// recurrence, the same breakdown guards, the same stagnation guard, and the mean projection of
// r / v / t in the singular case. Three pieces are injected by the caller through `Ops`: the matvec
// of the TRUE (probe) operator, the preconditioner z = M^-1 r (one ScalarMG V-cycle on the SPD
// surrogate, scalar_mg.hpp; level 0 alone = 2 + 2 red-black sweeps under the transient level
// rule), and the reductions (`dot`, `dot2`,
// `maxabs`, `removeMean`), which carry the MPI_Allreduce. Vectors are block fields on flow's G = 2
// block; vector operations run over inner cells, and identity rows keep every Krylov vector exactly
// 0 at the cells that are not unknowns, so the dots need no mask.
//
// Stopping (§5.1): ref = max(max|b|, max|A x0|) (A x0 = b - r0 comes free); stop when the
// RECURRENCE residual max|r_k| <= rtol ref, checked at each half-step. At exit the TRUE residual is
// recomputed once; if it exceeds 10 rtol ref the iteration restarts once from it (within the same
// iteration budget). The max-norm stop decision is reduction-order independent. A non-finite
// residual raises.
#ifndef PECLET_FLOW_SCALAR_KRYLOV_HPP
#define PECLET_FLOW_SCALAR_KRYLOV_HPP

#include <cmath>
#include <functional>
#include <Kokkos_Core.hpp>
#include <stdexcept>

#include "mac_cutcell.hpp"

namespace peclet::flow {

/// Outcome of one `scalarBiCGStab` call.
struct ScalarKrylovResult {
  int iterations = 0;    ///< BiCGStab iterations (both passes when it restarted)
  double ref = 0.0;      ///< max(max|b|, max|A x0|)
  double trueRes = 0.0;  ///< max|b - A x| at exit (after the restart, if any)
  bool restarted = false;
  bool converged = true;  ///< trueRes <= 10 rtol ref
};

/// A Krylov vector (§5.1): the fluid field and, for a conjugate scalar, the solid field (WO-7).
/// Both are block fields on the G = 2 block; `solid` says whether `s` takes part.
struct ScalarVec {
  CCField f, s;
  bool solid = false;
};

namespace skr {

/// y += a x over inner cells.
inline void axpy(CCField y, double a, CCConst x, C3 e, int g) {
  ccFor3(
      "peclet::flow::skr_axpy", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        y(i) += a * x(i);
      });
}
/// y = x + a y over inner cells.
inline void aypx(CCField y, double a, CCConst x, C3 e, int g) {
  ccFor3(
      "peclet::flow::skr_aypx", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        y(i) = x(i) + a * y(i);
      });
}
/// y = b - y over inner cells (the residual from a matvec result held in y).
inline void residualFrom(CCField y, CCConst b, C3 e, int g) {
  ccFor3(
      "peclet::flow::skr_residual", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        y(i) = b(i) - y(i);
      });
}

/// The same three operations on a ScalarVec, phase by phase.
inline void axpy(const ScalarVec& y, double a, const ScalarVec& x, C3 e, int g) {
  axpy(y.f, a, CCConst(x.f), e, g);
  if (y.solid)
    axpy(y.s, a, CCConst(x.s), e, g);
}
inline void aypx(const ScalarVec& y, double a, const ScalarVec& x, C3 e, int g) {
  aypx(y.f, a, CCConst(x.f), e, g);
  if (y.solid)
    aypx(y.s, a, CCConst(x.s), e, g);
}
inline void residualFrom(const ScalarVec& y, const ScalarVec& b, C3 e, int g) {
  residualFrom(y.f, CCConst(b.f), e, g);
  if (y.solid)
    residualFrom(y.s, CCConst(b.s), e, g);
}
inline void copy(const ScalarVec& y, const ScalarVec& x) {
  Kokkos::deep_copy(CCExec(), y.f, x.f);
  if (y.solid)
    Kokkos::deep_copy(CCExec(), y.s, x.s);
}
inline void zero(const ScalarVec& y) {
  Kokkos::deep_copy(CCExec(), y.f, 0.0);
  if (y.solid)
    Kokkos::deep_copy(CCExec(), y.s, 0.0);
}

}  // namespace skr

/// The injected pieces as callables (host-side calls that launch kernels; the overhead of the
/// indirection is nothing next to a block-wide kernel).
struct ScalarKrylovOps {
  C3 e{0, 0, 0};
  int g = 0;
  std::function<void(const ScalarVec&, const ScalarVec&)> matvec, precond;
  std::function<double(const ScalarVec&, const ScalarVec&)> dot;
  std::function<void(const ScalarVec&, const ScalarVec&, const ScalarVec&, double&, double&)> dot2;
  std::function<double(const ScalarVec&)> maxabs;
  std::function<void(const ScalarVec&)> removeMean;
};

/// BiCGStab on A x = b. `ops` provides
///   void matvec(const ScalarVec& y, const ScalarVec& x)   y = A x (exchanges x's ghosts itself)
///   void precond(const ScalarVec& z, const ScalarVec& r)  z = M^-1 r (a fixed linear operator)
///   double dot(const ScalarVec& a, const ScalarVec& b)    global inner-cell sum, both phases
///   void dot2(a, b, c, double& ab, double& cc)            fused (a.b, c.c), one reduction
///   double maxabs(const ScalarVec& a)                     global inner-cell max |a|
///   void removeMean(const ScalarVec& a)                   singular case only: the unknown-row mean
///   C3 e; int g;                                          the block
/// Scratch: r, rh, p, v, t, z, z2. x holds the initial guess and the result.
template <class Ops>
ScalarKrylovResult scalarBiCGStab(Ops& ops, const ScalarVec& b, const ScalarVec& x,
                                  const ScalarVec& r, const ScalarVec& rh, const ScalarVec& p,
                                  const ScalarVec& v, const ScalarVec& t, const ScalarVec& z,
                                  const ScalarVec& z2, int maxit, double rtol, bool singular) {
  const C3 e = ops.e;
  const int g = ops.g;
  ScalarKrylovResult res;
  ops.matvec(t, x);  // t = A x0
  const double ax0 = ops.maxabs(t);
  const double bmax = ops.maxabs(b);
  res.ref = bmax > ax0 ? bmax : ax0;
  if (!std::isfinite(res.ref))
    throw std::runtime_error("scalar cut-cell solve: non-finite right-hand side or initial guess");
  const double thresh = rtol * res.ref;
  int it = 0;
  // One BiCGStab pass from the current x (CutcellMG::solveBiCGStab's loop, with the §5.1 stop).
  auto pass = [&]() {
    ops.matvec(t, x);  // r = b - A x  (t as scratch)
    skr::copy(r, t);
    skr::residualFrom(r, b, e, g);
    if (singular)
      ops.removeMean(r);
    skr::copy(rh, r);  // shadow residual r^ = r_0
    const double r0n = ops.maxabs(r);
    if (!std::isfinite(r0n))
      throw std::runtime_error("scalar cut-cell solve: non-finite residual");
    if (!(r0n > thresh))
      return;
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double best = r0n;
    int lastImprove = it;
    skr::zero(p);
    skr::zero(v);
    for (; it < maxit; ++it) {
      const double rhoNew = ops.dot(rh, r);
      if (!std::isfinite(rhoNew) || std::fabs(rhoNew) < 1e-300)
        break;  // (rh, r) breakdown: keep the last finite iterate
      const double beta = (rhoNew / rho) * (alpha / omega);
      rho = rhoNew;
      skr::axpy(p, -omega, v, e, g);  // p = r + beta (p - omega v)
      skr::aypx(p, beta, r, e, g);
      ops.precond(z, p);
      ops.matvec(v, z);
      if (singular)
        ops.removeMean(v);
      const double rhv = ops.dot(rh, v);
      if (!std::isfinite(rhv) || std::fabs(rhv) < 1e-300)
        break;
      alpha = rho / rhv;
      skr::axpy(r, -alpha, v, e, g);  // r <- s = r - alpha v
      if (singular)
        ops.removeMean(r);
      double rn = ops.maxabs(r);
      if (!std::isfinite(rn))
        throw std::runtime_error("scalar cut-cell solve: non-finite residual");
      if (rn <= thresh) {
        skr::axpy(x, alpha, z, e, g);
        ++it;
        break;
      }
      ops.precond(z2, r);
      ops.matvec(t, z2);
      if (singular)
        ops.removeMean(t);
      double tr = 0.0, tt = 0.0;
      ops.dot2(t, r, t, tr, tt);
      if (!std::isfinite(tt) || tt < 1e-300) {
        skr::axpy(x, alpha, z, e, g);  // omega breakdown: take the alpha half-step, stop
        ++it;
        break;
      }
      omega = tr / tt;
      if (!std::isfinite(omega) || std::fabs(omega) < 1e-300) {
        skr::axpy(x, alpha, z, e, g);
        ++it;
        break;
      }
      skr::axpy(x, alpha, z, e, g);
      skr::axpy(x, omega, z2, e, g);
      skr::axpy(r, -omega, t, e, g);
      if (singular)
        ops.removeMean(r);
      rn = ops.maxabs(r);
      if (!std::isfinite(rn))
        throw std::runtime_error("scalar cut-cell solve: non-finite residual");
      if (rn <= thresh) {
        ++it;
        break;
      }
      if (rn < 0.999 * best) {
        best = rn;
        lastImprove = it;
      } else if (it - lastImprove > 30) {
        ++it;  // stagnation: accept the best-so-far level
        break;
      }
    }
  };
  auto trueResidual = [&]() {
    ops.matvec(t, x);
    skr::copy(r, t);
    skr::residualFrom(r, b, e, g);
    if (singular)
      ops.removeMean(r);
    const double rn = ops.maxabs(r);
    if (!std::isfinite(rn))
      throw std::runtime_error("scalar cut-cell solve: non-finite residual");
    return rn;
  };
  if (res.ref > 0.0) {
    pass();
    res.trueRes = trueResidual();
    if (res.trueRes > 10.0 * thresh && it < maxit) {
      res.restarted = true;
      pass();
      res.trueRes = trueResidual();
    }
  }
  res.iterations = it;
  res.converged = res.trueRes <= 10.0 * thresh;
  return res;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_KRYLOV_HPP
