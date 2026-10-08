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
//
// Device-resident scalars (WO-9b; flow's A6, CutcellMG::solvePCGResident): on a single rank and
// for a non-singular system the pass keeps rho, alpha, omega and beta on the device. The
// reductions land in slots of a small device array (the SAME policies and functors as the host
// dots, so the same values); one-thread kernels form the coefficients with the same IEEE
// divisions and run the breakdown tests; the vector updates read their coefficient from the slot
// and skip once a stop code is set. The host reads ONE packet {stop, max|r|} per half-step — what
// the max-norm stop test needs — instead of five scalars per iteration. Every exit returns the
// same iterate and iteration count as the host-scalar pass, bitwise. The distributed and the
// singular solves keep the host-scalar pass (their reductions are an MPI_Allreduce or feed the
// mean projection on the host).
#ifndef PECLET_FLOW_SCALAR_KRYLOV_HPP
#define PECLET_FLOW_SCALAR_KRYLOV_HPP

#include <cmath>
#include <functional>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <utility>

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

/// The device slots of the resident pass, and the stop codes (exact in double).
namespace skr {
enum : int {
  kRho = 0,
  kAlpha = 1,
  kOmega = 2,
  kBeta = 3,
  kRhoNew = 4,
  kRhv = 5,
  kTr = 6,
  kTt = 7,
  kStop = 8,  ///< the packet the host reads: {stop, max|r_f|, max|r_s|}
  kRn = 9,
  kRnS = 10,
  kNSlots = 11
};
constexpr double kBrkRho = 1.0, kBrkRhv = 2.0, kBrkTt = 3.0, kBrkOmega = 4.0;
using Slot = Kokkos::View<double, CCMem>;
inline Slot slot(const Kokkos::View<double*, CCMem>& ks, int k) {
  return Slot(ks.data() + k);
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
  /// The resident pass (single rank): `ks` (>= skr::kNSlots device doubles) and `pk` (3 host
  /// doubles) persistent scratch; dotTo / dot2To / maxabsTo the reductions of dot / dot2 / maxabs
  /// into device slots (maxabsTo: the fluid part and the solid part, combined on the host with
  /// std::fmax exactly as `maxabs` does).
  bool resident = false;
  Kokkos::View<double*, CCMem> ks;
  Kokkos::View<double*, Kokkos::HostSpace> pk;
  std::function<void(const ScalarVec&, const ScalarVec&, skr::Slot)> dotTo;
  std::function<void(const ScalarVec&, const ScalarVec&, const ScalarVec&, skr::Slot, skr::Slot)>
      dot2To;
  std::function<void(const ScalarVec&, skr::Slot, skr::Slot)> maxabsTo;
};

/// Arm `ops` for the resident pass with the persistent slots of `st` (`ks`, `pk`, allocated once).
template <class State>
inline void residentKrylov(State& st, ScalarKrylovOps& ops) {
  if (st.ks.extent(0) < (std::size_t)skr::kNSlots)
    st.ks = Kokkos::View<double*, CCMem>("peclet::flow::skr_slots", skr::kNSlots);
  if (st.pk.extent(0) < 3)
    st.pk = Kokkos::View<double*, Kokkos::HostSpace>("peclet::flow::skr_packet", 3);
  ops.resident = true;
  ops.ks = st.ks;
  ops.pk = st.pk;
}

namespace skr {

/// One-thread kernel on the slots (the coefficient arithmetic of the host pass, on the device).
template <class F>
inline void scalarKernel(const char* name, F f) {
  Kokkos::parallel_for(name, Kokkos::RangePolicy<CCExec>(CCExec(), 0, 1), f);
}

inline void residentInit(const Kokkos::View<double*, CCMem>& ks) {
  scalarKernel(
      "peclet::flow::skr_init", KOKKOS_LAMBDA(int) {
        ks(kRho) = 1.0;
        ks(kAlpha) = 1.0;
        ks(kOmega) = 1.0;
        ks(kStop) = 0.0;
      });
}
/// (rh, r) breakdown, else beta = (rhoNew / rho) (alpha / omega) and rho = rhoNew.
inline void residentBeta(const Kokkos::View<double*, CCMem>& ks) {
  scalarKernel(
      "peclet::flow::skr_beta", KOKKOS_LAMBDA(int) {
        const double rhoNew = ks(kRhoNew);
        if (!Kokkos::isfinite(rhoNew) || Kokkos::fabs(rhoNew) < 1e-300) {
          ks(kStop) = kBrkRho;
          return;
        }
        ks(kBeta) = (rhoNew / ks(kRho)) * (ks(kAlpha) / ks(kOmega));
        ks(kRho) = rhoNew;
      });
}
/// (rh, v) breakdown, else alpha = rho / (rh, v).
inline void residentAlpha(const Kokkos::View<double*, CCMem>& ks) {
  scalarKernel(
      "peclet::flow::skr_alpha", KOKKOS_LAMBDA(int) {
        if (ks(kStop) != 0.0)
          return;
        const double rhv = ks(kRhv);
        if (!Kokkos::isfinite(rhv) || Kokkos::fabs(rhv) < 1e-300) {
          ks(kStop) = kBrkRhv;
          return;
        }
        ks(kAlpha) = ks(kRho) / rhv;
      });
}
/// The tt breakdown, else omega = (t, r) / (t, t) and its breakdown.
inline void residentOmega(const Kokkos::View<double*, CCMem>& ks) {
  scalarKernel(
      "peclet::flow::skr_omega", KOKKOS_LAMBDA(int) {
        const double tt = ks(kTt);
        if (!Kokkos::isfinite(tt) || tt < 1e-300) {
          ks(kStop) = kBrkTt;
          return;
        }
        const double om = ks(kTr) / tt;
        if (!Kokkos::isfinite(om) || Kokkos::fabs(om) < 1e-300) {
          ks(kStop) = kBrkOmega;
          return;
        }
        ks(kOmega) = om;
      });
}
/// p = r + beta (p - omega v): axpy(p, -omega, v) then aypx(p, beta, r), per phase; skipped on a
/// stop.
inline void residentP(const ScalarVec& p, const ScalarVec& v, const ScalarVec& r,
                      const Kokkos::View<double*, CCMem>& ks, C3 e, int g) {
  CCField pf = p.f, ps = p.s, vf = v.f, vs = v.s, rf = r.f, rs = r.s;
  const bool two = p.solid;
  ccFor3(
      "peclet::flow::skr_p", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (ks(kStop) != 0.0)
          return;
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        const double na = -ks(kOmega), beta = ks(kBeta);
        pf(i) += na * vf(i);
        pf(i) = rf(i) + beta * pf(i);
        if (two) {
          ps(i) += na * vs(i);
          ps(i) = rs(i) + beta * ps(i);
        }
      });
}
/// r <- s = r - alpha v (axpy(r, -alpha, v)); skipped on a stop.
inline void residentS(const ScalarVec& r, const ScalarVec& v,
                      const Kokkos::View<double*, CCMem>& ks, C3 e, int g) {
  CCField rf = r.f, rs = r.s, vf = v.f, vs = v.s;
  const bool two = r.solid;
  ccFor3(
      "peclet::flow::skr_s", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        if (ks(kStop) != 0.0)
          return;
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        const double na = -ks(kAlpha);
        rf(i) += na * vf(i);
        if (two)
          rs(i) += na * vs(i);
      });
}
/// x += alpha z (the half-step exit).
inline void residentXHalf(const ScalarVec& x, const ScalarVec& z,
                          const Kokkos::View<double*, CCMem>& ks, C3 e, int g) {
  CCField xf = x.f, xs = x.s, zf = z.f, zs = z.s;
  const bool two = x.solid;
  ccFor3(
      "peclet::flow::skr_x_half", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        const double a = ks(kAlpha);
        xf(i) += a * zf(i);
        if (two)
          xs(i) += a * zs(i);
      });
}
/// x += alpha z; x += omega z2; r -= omega t (the three axpys) — or, after a tt / omega
/// breakdown, the alpha half-step alone.
inline void residentXR(const ScalarVec& x, const ScalarVec& r, const ScalarVec& z,
                       const ScalarVec& z2, const ScalarVec& t,
                       const Kokkos::View<double*, CCMem>& ks, C3 e, int g) {
  CCField xf = x.f, xs = x.s, rf = r.f, rs = r.s, zf = z.f, zs = z.s, z2f = z2.f, z2s = z2.s,
          tf = t.f, ts = t.s;
  const bool two = x.solid;
  ccFor3(
      "peclet::flow::skr_x_r", C3{g, g, g}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
        const double a = ks(kAlpha);
        if (ks(kStop) != 0.0) {
          xf(i) += a * zf(i);
          if (two)
            xs(i) += a * zs(i);
          return;
        }
        const double om = ks(kOmega), nom = -om;
        xf(i) += a * zf(i);
        xf(i) += om * z2f(i);
        rf(i) += nom * tf(i);
        if (two) {
          xs(i) += a * zs(i);
          xs(i) += om * z2s(i);
          rs(i) += nom * ts(i);
        }
      });
}
}  // namespace skr

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
  // The same pass with the scalars on the device (WO-9b; see the file comment). Statement for
  // statement `pass` above, through the skr::resident* kernels.
  auto passResident = [&]() {
    ops.matvec(t, x);
    skr::copy(r, t);
    skr::residualFrom(r, b, e, g);
    skr::copy(rh, r);
    const double r0n = ops.maxabs(r);
    if (!std::isfinite(r0n))
      throw std::runtime_error("scalar cut-cell solve: non-finite residual");
    if (!(r0n > thresh))
      return;
    double best = r0n;
    int lastImprove = it;
    skr::zero(p);
    skr::zero(v);
    const auto& ks = ops.ks;
    skr::residentInit(ks);
    // the packet {stop, max|r|} (both phases combined with std::fmax, as ops.maxabs)
    auto readPacket = [&](double& stop) {
      ops.maxabsTo(r, skr::slot(ks, skr::kRn), skr::slot(ks, skr::kRnS));
      Kokkos::deep_copy(ops.pk,
                        Kokkos::subview(ks, std::make_pair((int)skr::kStop, (int)skr::kStop + 3)));
      stop = ops.pk(0);
      return x.solid ? std::fmax(ops.pk(1), ops.pk(2)) : ops.pk(1);
    };
    for (; it < maxit; ++it) {
      ops.dotTo(rh, r, skr::slot(ks, skr::kRhoNew));
      skr::residentBeta(ks);
      skr::residentP(p, v, r, ks, e, g);  // p = r + beta (p - omega v)
      ops.precond(z, p);
      ops.matvec(v, z);
      ops.dotTo(rh, v, skr::slot(ks, skr::kRhv));
      skr::residentAlpha(ks);
      skr::residentS(r, v, ks, e, g);  // r <- s = r - alpha v
      double stop = 0.0;
      double rn = readPacket(stop);  // THE read of the half-step
      if (stop != 0.0)
        break;  // (rh, r) or (rh, v) breakdown: keep the last finite iterate
      if (!std::isfinite(rn))
        throw std::runtime_error("scalar cut-cell solve: non-finite residual");
      if (rn <= thresh) {
        skr::residentXHalf(x, z, ks, e, g);
        ++it;
        break;
      }
      ops.precond(z2, r);
      ops.matvec(t, z2);
      ops.dot2To(t, r, t, skr::slot(ks, skr::kTr), skr::slot(ks, skr::kTt));
      skr::residentOmega(ks);
      skr::residentXR(x, r, z, z2, t, ks, e, g);
      rn = readPacket(stop);  // THE read of the full step
      if (stop != 0.0) {
        ++it;  // tt / omega breakdown: the alpha half-step was taken, stop
        break;
      }
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
  const bool resident = ops.resident && !singular;
  if (res.ref > 0.0) {
    resident ? passResident() : pass();
    res.trueRes = trueResidual();
    if (res.trueRes > 10.0 * thresh && it < maxit) {
      res.restarted = true;
      resident ? passResident() : pass();
      res.trueRes = trueResidual();
    }
  }
  res.iterations = it;
  res.converged = res.trueRes <= 10.0 * thresh;
  return res;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_KRYLOV_HPP
