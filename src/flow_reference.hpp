/// @file
/// @brief flow — assembled single-GPU periodic Navier-Stokes step on Kokkos fields.
///
/// Composes the ported operators into a Chorin projection step on a staggered MAC grid (periodic box,
/// ghost width 2): explicit Koren-TVD advection (optional) + backward-Euler implicit diffusion
/// (RB-GS) -> divergence -> pressure Poisson (RB-GS + mean removal) -> projection correction. Divided
/// convention (operator (1/dt)I - nu*Lap). Single GPU: the "halo" is a periodic ghost wrap. This is
/// the cfd analog of the packing demStep — it wires the validated kernels into a runnable solver.
#ifndef PECLET_FLOW_SDFLOW_HPP
#define PECLET_FLOW_SDFLOW_HPP

#include <Kokkos_Core.hpp>

#include "mac_mg.hpp"
#include "mac_reductions.hpp"
#include "mac_stencils.hpp"
#include "mac_transfer.hpp"
#include "staggered_advection.hpp"

namespace peclet::flow {

class FlowReference {
 public:
  using F = Kokkos::View<double*, SMem>;
  static constexpr int G = 2;

  FlowReference(int n, double nu, double dt) : N_(n), nu_(nu), dt_(dt), mg_(n) {
    e_ = I3{N_ + 2 * G, N_ + 2 * G, N_ + 2 * G};
    ne_ = (std::size_t)e_.x * e_.y * e_.z;
    u_ = F("u", ne_); v_ = F("v", ne_); w_ = F("w", ne_);
    us_ = F("us", ne_); vs_ = F("vs", ne_); ws_ = F("ws", ne_);
    bu_ = F("bu", ne_); bv_ = F("bv", ne_); bw_ = F("bw", ne_);
    phi_ = F("phi", ne_); div_ = F("div", ne_);
    // g=1 scratch for the multigrid level-0 (MG uses ghost width 1; the solver uses 2).
    eMg_ = I3{N_ + 2, N_ + 2, N_ + 2};
    const std::size_t nm = (std::size_t)eMg_.x * eMg_.y * eMg_.z;
    dMg_ = F("dMg", nm); phiMg_ = F("phiMg", nm);
  }

  F& u() { return u_; }
  F& v() { return v_; }
  F& w() { return w_; }
  int N() const { return N_; }
  I3 ext() const { return e_; }
  void setBodyForce(double fx, double fy, double fz) { fx_ = fx; fy_ = fy; fz_ = fz; }
  void setAdvection(bool on) { advect_ = on; }
  void setIterations(int nDiff, int nPois) { nDiff_ = nDiff; nPois_ = nPois; }
  // Pressure solver: multigrid V-cycles (default) or plain RB-GS (useMg=false).
  void setPressureMultigrid(bool useMg, int nVcycles) { useMg_ = useMg; nVcycles_ = nVcycles; }

  void step() {
    const I3 og{0, 0, 0};
    periodicFill(u_); periodicFill(v_); periodicFill(w_);
    buildRhs();  // bu/bv/bw = idiag*u + f - advect
    // implicit diffusion (RB-GS) per component, starting guess = current field
    Kokkos::deep_copy(us_, u_); Kokkos::deep_copy(vs_, v_); Kokkos::deep_copy(ws_, w_);
    const double Ac = idiag() + 6.0 * nu_;
    for (int it = 0; it < nDiff_; ++it) {
      diffuseComp(us_, bu_, Ac);
      diffuseComp(vs_, bv_, Ac);
      diffuseComp(ws_, bw_, Ac);
    }
    // projection: Lap(phi) = div(u*); u = u* - grad(phi)
    periodicFill(us_); periodicFill(vs_); periodicFill(ws_);
    divergence(SConst(us_), SConst(vs_), SConst(ws_), div_, e_, G);
    Kokkos::deep_copy(phi_, 0.0);
    if (useMg_) {  // geometric multigrid V-cycles (fast); bridge g=2 <-> g=1 by inner-cell copy
      copyInner(dMg_, eMg_, 1, SConst(div_), e_, G);
      Kokkos::deep_copy(phiMg_, 0.0);
      mg_.solve(phiMg_, SConst(dMg_), nVcycles_);
      copyInner(phi_, e_, G, SConst(phiMg_), eMg_, 1);
    } else {  // plain RB-GS (slow; kept for comparison)
      for (int it = 0; it < nPois_; ++it) {
        periodicFill(phi_);
        poisSweep(phi_, SConst(div_), e_, og, G);
        removeMean(phi_);
      }
    }
    periodicFill(phi_);
    correct(us_, vs_, ws_, SConst(phi_), T3{e_.x, e_.y, e_.z}, G);
    Kokkos::deep_copy(u_, us_); Kokkos::deep_copy(v_, vs_); Kokkos::deep_copy(w_, ws_);
  }

  // L2 over inner cells of a component (for diagnostics / validation).
  double l2(F f) {
    SumMax s = localSumMaxSq(SConst(f));
    return std::sqrt(s.sum);
  }

  // max|div(u)| over inner cells (projection quality).
  double maxDivU() {
    periodicFill(u_); periodicFill(v_); periodicFill(w_);
    divergence(SConst(u_), SConst(v_), SConst(w_), div_, e_, G);
    return reduceInner(SConst(div_), false).maxabs;
  }

  // (public: nvcc forbids extended __host__ __device__ lambdas inside private/protected members.)
  double idiag() const { return 1.0 / dt_; }

  // Copy the N^3 inner cells between two extended blocks of different ghost width.
  void copyInner(F dst, I3 de, int dg, SConst src, I3 se, int sg) {
    SExec space; const int N = N_;
    Kokkos::parallel_for(
        "peclet::flow::copyInner", Kokkos::RangePolicy<SExec>(space, 0, (long)N * N * N),
        KOKKOS_LAMBDA(long c) {
          const int ix = (int)(c % N), iy = (int)((c / N) % N), iz = (int)(c / ((long)N * N));
          const long di = (long)(ix+dg) + (long)(iy+dg)*de.x + (long)(iz+dg)*(long)de.x*de.y;
          const long si = (long)(ix+sg) + (long)(iy+sg)*se.x + (long)(iz+sg)*(long)se.x*se.y;
          dst(di) = src(si);
        });

  }

  void diffuseComp(F x, F b, double Ac) {
    periodicFill(x);
    diffSmoothColor(x, SConst(b), e_, I3{0, 0, 0}, G, nu_, Ac, 0, SConst());
    periodicFill(x);
    diffSmoothColor(x, SConst(b), e_, I3{0, 0, 0}, G, nu_, Ac, 1, SConst());
  }

  void buildRhs() {
    SExec space; const double id = idiag(); const bool adv = advect_;
    const double fx = fx_, fy = fy_, fz = fz_; const I3 e = e_;
    F u = u_, v = v_, w = w_, bu = bu_, bv = bv_, bw = bw_;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::sdflow_rhs", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double au = 0, av = 0, aw = 0;
          if (adv) {
            sadv::ViewAcc U{SConst(u), e.x, e.y}, V{SConst(v), e.x, e.y}, W{SConst(w), e.x, e.y};
            au = sadv::advect(0, x, y, z, U, V, W, U);
            av = sadv::advect(1, x, y, z, U, V, W, V);
            aw = sadv::advect(2, x, y, z, U, V, W, W);
          }
          bu(i) = id * u(i) + fx - au;
          bv(i) = id * v(i) + fy - av;
          bw(i) = id * w(i) + fz - aw;
        });

  }

  void removeMean(F f) {
    SumMax s = localSumMax(SConst(f));
    const double mean = s.sum / ((double)N_ * N_ * N_);
    SExec space; const I3 e = e_;
    Kokkos::parallel_for("peclet::flow::submean", Kokkos::RangePolicy<SExec>(space, 0, ne_),
                         KOKKOS_LAMBDA(std::size_t i) { f(i) -= mean; });

  }

  // sum + max|.| over inner cells (reuse mac_reductions math inline for the periodic block).
  SumMax localSumMax(SConst f) { return reduceInner(f, false); }
  SumMax localSumMaxSq(SConst f) { return reduceInner(f, true); }
  SumMax reduceInner(SConst f, bool sq) {
    SExec space; const I3 e = e_; const int N = N_;
    SumMax r;
    Kokkos::parallel_reduce(
        "peclet::flow::sdflow_reduce", Kokkos::RangePolicy<SExec>(space, 0, (long)N * N * N),
        KOKKOS_LAMBDA(long c, SumMax& acc) {
          const int ix = (int)(c % N), iy = (int)((c / N) % N), iz = (int)(c / ((long)N * N));
          const long i = (long)(ix + G) + (long)(iy + G) * e.x + (long)(iz + G) * (long)e.x * e.y;
          const double val = f(i);
          acc.sum += sq ? val * val : val;
          const double a = Kokkos::fabs(val);
          if (a > acc.maxabs) acc.maxabs = a;
        },
        r);
    return r;
  }

  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners).
  void periodicFill(F f) {
    fillAxis(f, 0); fillAxis(f, 1); fillAxis(f, 2);
  }
  void fillAxis(F f, int axis) {
    SExec space; const I3 e = e_; const int N = N_;
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    F ff = f;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>;
    // copy the two ghost slabs from the wrapped inner planes; over the FULL perp extent so corners fill.
    Kokkos::parallel_for(
        "peclet::flow::pfill", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
          const long base = (long)p0 * sb + (long)p1 * sc;
          for (int gl = 0; gl < G; ++gl) {
            // low ghost gl  <- inner plane (gl + N)   ; high ghost (G+N+gl) <- inner plane (G+gl)
            ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
            ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
          }
        });

  }

 private:
  int N_; double nu_, dt_;
  I3 e_; std::size_t ne_;
  F u_, v_, w_, us_, vs_, ws_, bu_, bv_, bw_, phi_, div_;
  MgPoisson mg_;
  I3 eMg_; F dMg_, phiMg_;
  double fx_ = 0, fy_ = 0, fz_ = 0;
  bool advect_ = true, useMg_ = true;
  int nDiff_ = 20, nPois_ = 50, nVcycles_ = 8;
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SDFLOW_HPP
