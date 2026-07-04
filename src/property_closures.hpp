/// @file
/// @brief flow — device property closures: material properties / body forces as functions of fields.
///
/// A closure writes one target cell field (a material property like rho/mu, or a momentum body-force
/// component) as a pointwise function of one or two input fields (a transported scalar, pressure, a
/// phase fraction). Dispatch is a host-side enum switch launching ONE dedicated Kokkos kernel per
/// closure kind — no per-cell Python, no device virtual dispatch. The Solver applies its closure list
/// in registration order at the top of step() (properties frozen over the step; segregated coupling).
///
/// This is the seam for field–field coupling: Boussinesq buoyancy (force from temperature),
/// temperature-dependent viscosity (Arrhenius), composition-dependent density (linear mixture), and
/// tabulated properties. A user escape hatch (set_field on a property) bypasses closures entirely.
#ifndef PECLET_FLOW_PROPERTY_CLOSURES_HPP
#define PECLET_FLOW_PROPERTY_CLOSURES_HPP

#include <Kokkos_Core.hpp>
#include <array>

#include "mac_cutcell.hpp"

namespace peclet::flow {

enum class ClosureKind {
  LinearMix,        // out = p0 + p1*in0 + p2*in1
  BoussinesqForce,  // out = p0*p1*p2*(in0 - p3)   [rho0, g, beta, T0] -> buoyancy body force
  ArrheniusMu,      // out = p0*exp(p1*(1/in0 - 1/p2))   [mu_ref, B, Tref]
  Table1D           // out = piecewise-linear interp of (tabX, tabY) at in0
};

// A registered closure. `out`/`in0`/`in1` are resolved from the field registry at registration; a
// later redistribution that reallocates fields must re-resolve them.
struct Closure {
  ClosureKind kind;
  CCField out;
  CCConst in0, in1;
  std::array<double, 4> p{{0, 0, 0, 0}};
  CCField tabX, tabY;  // Table1D nodes (ascending tabX)
  int nTab = 0;
};

// Apply one closure over the inner cells (ghosts untouched — refilled by the field's own exchange).
inline void applyClosure(const Closure& cl, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  const ClosureKind kind = cl.kind;
  CCField out = cl.out;
  CCConst in0 = cl.in0, in1 = cl.in1;
  const double p0 = cl.p[0], p1 = cl.p[1], p2 = cl.p[2], p3 = cl.p[3];
  const bool haveIn1 = (in1.data() != nullptr) && (in1.extent(0) == out.extent(0));
  CCConst tx = cl.tabX, ty = cl.tabY;
  const int nTab = cl.nTab;
  Kokkos::parallel_for(
      "peclet::flow::apply_closure", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double v;
        switch (kind) {
          case ClosureKind::LinearMix:
            v = p0 + p1 * in0(i) + (haveIn1 ? p2 * in1(i) : 0.0);
            break;
          case ClosureKind::BoussinesqForce:
            v = p0 * p1 * p2 * (in0(i) - p3);
            break;
          case ClosureKind::ArrheniusMu:
            v = p0 * Kokkos::exp(p1 * (1.0 / in0(i) - 1.0 / p2));
            break;
          default: {  // Table1D: clamped piecewise-linear interpolation
            const double s = in0(i);
            if (nTab <= 0) {
              v = 0.0;
            } else if (s <= tx(0)) {
              v = ty(0);
            } else if (s >= tx(nTab - 1)) {
              v = ty(nTab - 1);
            } else {
              int lo = 0, hi = nTab - 1;  // binary search for the bracketing node
              while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                if (tx(mid) <= s)
                  lo = mid;
                else
                  hi = mid;
              }
              const double t = (s - tx(lo)) / (tx(hi) - tx(lo));
              v = ty(lo) + t * (ty(hi) - ty(lo));
            }
          }
        }
        out(i) = v;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_PROPERTY_CLOSURES_HPP
